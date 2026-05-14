use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{FullSpectrumKramersKronigInput, full_spectrum_kramers_kronig};
use refeff_io::{
    DrudeDatData, EpsDatData, FullSpectrumInput, OpconsDatData,
    opcons_dat_from_fullspectrum_epsilon_minus_one, read_drude_dat, read_eps_dat, write_opcons_dat,
};

use crate::work_dir_for_input;

const OMEGA_MATCH_TOLERANCE: f64 = 1.0e-10;

/// Run cached FEFF `FULLSPECTRUM` optical-table generation beside an input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether FEFF FULLSPECTRUM has cached dielectric data ready for optical tables.
pub(crate) fn has_cached_optical_inputs(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("fullspectrum.inp");
    if !input_path.is_file() || !work_dir.join("eps.dat").is_file() {
        return Ok(false);
    }
    Ok(read_input(work_dir)?.m_full_spectrum > 0)
}

/// Write `opcons.dat`, `opconsKK.dat`, and `opcons0.dat` from cached `eps.dat`.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.m_full_spectrum <= 0 {
        return Ok(0);
    }

    let eps_path = work_dir.join("eps.dat");
    let eps = read_eps_dat(&eps_path)
        .with_context(|| format!("failed to read {}", eps_path.display()))?;

    let drude_epsilon = read_optional_drude_epsilon(work_dir, &eps.omega)?;
    let total_epsilon = add_optional_drude_epsilon(&eps.epsilon, drude_epsilon.as_ref());
    let direct = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM optical constants from eps.dat".to_string()],
        &eps,
        &total_epsilon,
    )?;
    write_opcons(work_dir, "opcons.dat", &direct)?;

    let kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.epsilon)
        .context("failed to compute FULLSPECTRUM Kramers-Kronig table")?;
    let kk = add_optional_drude_epsilon(&kk_bound, drude_epsilon.as_ref());
    let kk_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM Kramers-Kronig optical constants from eps.dat".to_string()],
        &eps,
        &kk,
    )?;
    write_opcons(work_dir, "opconsKK.dat", &kk_table)?;

    let background_kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.background_epsilon)
        .context("failed to compute FULLSPECTRUM background Kramers-Kronig table")?;
    let background_kk = add_optional_drude_epsilon(&background_kk_bound, drude_epsilon.as_ref());
    let background_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM atomic-background optical constants from eps.dat".to_string()],
        &eps,
        &background_kk,
    )?;
    write_opcons(work_dir, "opcons0.dat", &background_table)?;

    Ok(eps.point_count())
}

fn read_optional_drude_epsilon(
    work_dir: &Path,
    omega: &Array1<f64>,
) -> Result<Option<Array1<Complex64>>> {
    let path = work_dir.join("drude.dat");
    if !path.is_file() {
        return Ok(None);
    }
    let drude =
        read_drude_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    validate_drude_grid(&drude, omega)?;
    Ok(Some(drude.epsilon))
}

fn validate_drude_grid(drude: &DrudeDatData, omega: &Array1<f64>) -> Result<()> {
    if drude.point_count() != omega.len() {
        bail!(
            "drude.dat has {} row(s) but eps.dat has {} row(s)",
            drude.point_count(),
            omega.len()
        );
    }
    for (index, (&drude_omega, &eps_omega)) in drude.omega.iter().zip(omega.iter()).enumerate() {
        let scale = drude_omega.abs().max(eps_omega.abs()).max(1.0);
        if (drude_omega - eps_omega).abs() > OMEGA_MATCH_TOLERANCE * scale {
            bail!(
                "drude.dat omega row {} ({drude_omega}) does not match eps.dat omega ({eps_omega})",
                index + 1
            );
        }
    }
    Ok(())
}

fn add_optional_drude_epsilon(
    bound: &Array1<Complex64>,
    drude: Option<&Array1<Complex64>>,
) -> Array1<Complex64> {
    drude.map_or_else(
        || bound.clone(),
        |drude| {
            Array1::from_iter(
                bound
                    .iter()
                    .zip(drude.iter())
                    .map(|(&bound, &free)| bound + free),
            )
        },
    )
}

fn read_input(work_dir: &Path) -> Result<FullSpectrumInput> {
    let input_path = work_dir.join("fullspectrum.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    FullSpectrumInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn opcons_from_eps(
    header_lines: Vec<String>,
    eps: &EpsDatData,
    epsilon_minus_one: &Array1<Complex64>,
) -> Result<OpconsDatData> {
    opcons_dat_from_fullspectrum_epsilon_minus_one(
        header_lines,
        eps.omega.view(),
        epsilon_minus_one.view(),
    )
    .context("failed to compute FULLSPECTRUM optical constants")
}

fn kramers_kronig_epsilon_minus_one(
    omega: &Array1<f64>,
    epsilon_minus_one: &Array1<Complex64>,
) -> Result<Array1<Complex64>> {
    let epsilon2 = epsilon_minus_one.mapv(|value| value.im);
    let epsilon1 = full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
        omega: omega.view(),
        epsilon2: epsilon2.view(),
    })?;
    Ok(Array1::from_iter(epsilon1.iter().zip(epsilon2.iter()).map(
        |(&real, &imaginary)| Complex64::new(real, imaginary),
    )))
}

fn write_opcons(work_dir: &Path, file_name: &str, data: &OpconsDatData) -> Result<()> {
    let path = work_dir.join(file_name);
    write_opcons_dat(&path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_optical_inputs, run_in_dir};
    use anyhow::Result;
    use ndarray::array;
    use num_complex::Complex64;
    use refeff_io::{
        DrudeDatData, EpsDatData, FullSpectrumInput, fullspectrum_input_string, read_opcons_dat,
        write_drude_dat, write_eps_dat,
    };

    fn sample_eps_dat() -> EpsDatData {
        EpsDatData {
            header_lines: vec!["# sample eps.dat".to_string()],
            omega: array![1.0, 2.0, 4.0, 7.0],
            epsilon: array![
                Complex64::new(0.2, 0.05),
                Complex64::new(0.4, 0.12),
                Complex64::new(0.1, 0.07),
                Complex64::new(0.3, 0.03),
            ],
            background_epsilon: array![
                Complex64::new(0.1, 0.02),
                Complex64::new(0.2, 0.04),
                Complex64::new(0.05, 0.025),
                Complex64::new(0.15, 0.01),
            ],
            sigma: array![0.01, 0.02, 0.03, 0.04],
        }
    }

    fn write_fullspectrum_input(path: &std::path::Path, flag: i32) -> Result<()> {
        std::fs::write(
            path.join("fullspectrum.inp"),
            fullspectrum_input_string(&FullSpectrumInput {
                m_full_spectrum: flag,
            })?,
        )?;
        Ok(())
    }

    #[test]
    fn detects_cached_optical_inputs_only_when_enabled_and_complete() -> Result<()> {
        let temp = tempfile::tempdir()?;
        assert!(!has_cached_optical_inputs(temp.path())?);

        write_fullspectrum_input(temp.path(), 0)?;
        write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;
        assert!(!has_cached_optical_inputs(temp.path())?);

        write_fullspectrum_input(temp.path(), 1)?;
        assert!(has_cached_optical_inputs(temp.path())?);
        Ok(())
    }

    #[test]
    fn skips_disabled_fullspectrum_without_reading_eps_dat() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fullspectrum_input(temp.path(), 0)?;

        assert_eq!(run_in_dir(temp.path())?, 0);
        assert!(!temp.path().join("opcons.dat").exists());
        Ok(())
    }

    #[test]
    fn adds_cached_drude_term_to_fullspectrum_optical_tables() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let eps = sample_eps_dat();
        let drude = DrudeDatData {
            gamma_ev: 0.5,
            plasma_frequency_ev: 2.0,
            omega: eps.omega.clone(),
            epsilon: array![
                Complex64::new(-0.01, 0.005),
                Complex64::new(-0.02, 0.006),
                Complex64::new(-0.03, 0.007),
                Complex64::new(-0.04, 0.008),
            ],
        };
        write_fullspectrum_input(temp.path(), 1)?;
        write_eps_dat(temp.path().join("eps.dat"), &eps)?;
        write_drude_dat(temp.path().join("drude.dat"), &drude)?;

        assert_eq!(run_in_dir(temp.path())?, eps.point_count());

        let opcons = read_opcons_dat(temp.path().join("opcons.dat"))?;
        let opcons_kk = read_opcons_dat(temp.path().join("opconsKK.dat"))?;
        let opcons0 = read_opcons_dat(temp.path().join("opcons0.dat"))?;
        for ((actual, bound), free) in opcons
            .epsilon_minus_one
            .iter()
            .zip(eps.epsilon.iter())
            .zip(drude.epsilon.iter())
        {
            assert!((actual.re - (bound.re + free.re)).abs() < 1.0e-10);
            assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
        }
        for ((actual, bound), free) in opcons_kk
            .epsilon_minus_one
            .iter()
            .zip(eps.epsilon.iter())
            .zip(drude.epsilon.iter())
        {
            assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
        }
        for ((actual, bound), free) in opcons0
            .epsilon_minus_one
            .iter()
            .zip(eps.background_epsilon.iter())
            .zip(drude.epsilon.iter())
        {
            assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
        }
        Ok(())
    }

    #[test]
    fn writes_cached_fullspectrum_optical_tables_from_eps_dat() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let eps = sample_eps_dat();
        write_fullspectrum_input(temp.path(), 1)?;
        write_eps_dat(temp.path().join("eps.dat"), &eps)?;

        assert_eq!(run_in_dir(temp.path())?, eps.point_count());

        let opcons = read_opcons_dat(temp.path().join("opcons.dat"))?;
        let opcons_kk = read_opcons_dat(temp.path().join("opconsKK.dat"))?;
        let opcons0 = read_opcons_dat(temp.path().join("opcons0.dat"))?;

        assert_eq!(opcons.point_count(), eps.point_count());
        assert_eq!(opcons_kk.point_count(), eps.point_count());
        assert_eq!(opcons0.point_count(), eps.point_count());
        for (actual, expected) in opcons.epsilon_minus_one.iter().zip(eps.epsilon.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-10);
            assert!((actual.im - expected.im).abs() < 1.0e-10);
        }
        for (actual, expected) in opcons_kk.epsilon_minus_one.iter().zip(eps.epsilon.iter()) {
            assert!((actual.im - expected.im).abs() < 1.0e-10);
        }
        for (actual, expected) in opcons0
            .epsilon_minus_one
            .iter()
            .zip(eps.background_epsilon.iter())
        {
            assert!((actual.im - expected.im).abs() < 1.0e-10);
        }
        Ok(())
    }
}
