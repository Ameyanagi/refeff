use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{FullSpectrumKramersKronigInput, full_spectrum_kramers_kronig};
use refeff_io::{
    DrudeDatData, EpsDatData, FullSpectrumInput, HamakerDatData, ModuleLogData, OpconsDatData,
    OscStrDatData, fullspectrum_number_density_from_pot_bin,
    opcons_dat_from_fullspectrum_epsilon_minus_one, read_drude_dat, read_eps_dat, read_hamaker_dat,
    read_module_log_dat, read_osc_str_dat, read_pot_bin, sumrules_dat_from_opcons, write_drude_dat,
    write_hamaker_dat, write_module_log_dat, write_opcons_dat, write_osc_str_dat,
    write_sumrules_dat,
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
///
/// Existing FULLSPECTRUM sidecars such as `drude.dat`, `osc_str.dat`, and
/// `hamaker.dat` are validated and re-rendered when present, along with
/// optional `logfullspectrum.dat` module diagnostics.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.m_full_spectrum <= 0 {
        return Ok(0);
    }

    let eps_path = work_dir.join("eps.dat");
    let eps = read_eps_dat(&eps_path)
        .with_context(|| format!("failed to read {}", eps_path.display()))?;

    let drude = read_optional_drude_cache(work_dir, &eps.omega)?;
    write_optional_sidecar_caches(work_dir)?;
    let total_epsilon =
        add_optional_drude_epsilon(&eps.epsilon, drude.as_ref().map(|data| &data.epsilon));
    let direct = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM optical constants from eps.dat".to_string()],
        &eps,
        &total_epsilon,
    )?;
    write_opcons(work_dir, "opcons.dat", &direct)?;

    let kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.epsilon)
        .context("failed to compute FULLSPECTRUM Kramers-Kronig table")?;
    let kk = add_optional_drude_epsilon(&kk_bound, drude.as_ref().map(|data| &data.epsilon));
    let kk_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM Kramers-Kronig optical constants from eps.dat".to_string()],
        &eps,
        &kk,
    )?;
    write_opcons(work_dir, "opconsKK.dat", &kk_table)?;
    write_optional_sumrules(work_dir, &kk_table)?;

    let background_kk_bound = kramers_kronig_epsilon_minus_one(&eps.omega, &eps.background_epsilon)
        .context("failed to compute FULLSPECTRUM background Kramers-Kronig table")?;
    let background_kk = add_optional_drude_epsilon(
        &background_kk_bound,
        drude.as_ref().map(|data| &data.epsilon),
    );
    let background_table = opcons_from_eps(
        vec!["# refeff FULLSPECTRUM atomic-background optical constants from eps.dat".to_string()],
        &eps,
        &background_kk,
    )?;
    write_opcons(work_dir, "opcons0.dat", &background_table)?;

    Ok(eps.point_count())
}

fn write_optional_sumrules(work_dir: &Path, opcons_kk: &OpconsDatData) -> Result<()> {
    let Some(number_density) = read_optional_sumrules_number_density(work_dir)? else {
        return Ok(());
    };
    let sumrules = sumrules_dat_from_opcons(number_density, opcons_kk)
        .context("failed to compute FULLSPECTRUM sum rules")?;
    let path = work_dir.join("sumrules.dat");
    write_sumrules_dat(&path, &sumrules)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn read_optional_sumrules_number_density(work_dir: &Path) -> Result<Option<f64>> {
    let path = work_dir.join("pot.bin");
    if !path.is_file() {
        return Ok(None);
    }
    let pot = read_pot_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut atomic_numbers = Vec::new();
    for atomic_number in pot.atomic_numbers.iter().copied() {
        if !atomic_numbers.contains(&atomic_number) {
            atomic_numbers.push(atomic_number);
        }
    }
    let density = atomic_numbers
        .into_iter()
        .map(|atomic_number| fullspectrum_number_density_from_pot_bin(atomic_number, &pot))
        .collect::<refeff_io::Result<Vec<_>>>()
        .context("failed to estimate FULLSPECTRUM sum-rule number densities")?
        .into_iter()
        .filter(|density| *density > 0.0)
        .reduce(f64::min)
        .context("pot.bin did not provide a positive FULLSPECTRUM number density")?;
    Ok(Some(density))
}

fn read_optional_drude_cache(work_dir: &Path, omega: &Array1<f64>) -> Result<Option<DrudeDatData>> {
    let path = work_dir.join("drude.dat");
    if !path.is_file() {
        return Ok(None);
    }
    let drude =
        read_drude_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    validate_drude_grid(&drude, omega)?;
    write_drude_cache(&path, &drude)?;
    Ok(Some(drude))
}

fn write_optional_sidecar_caches(work_dir: &Path) -> Result<()> {
    write_optional_osc_str_cache(&work_dir.join("osc_str.dat"))?;
    write_optional_hamaker_cache(&work_dir.join("hamaker.dat"))?;
    write_optional_module_log(&work_dir.join("logfullspectrum.dat"))?;
    Ok(())
}

fn write_optional_osc_str_cache(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_osc_str_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_osc_str_cache(path, &data)
}

fn write_optional_hamaker_cache(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_hamaker_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_hamaker_cache(path, &data)
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
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

fn write_drude_cache(path: &Path, data: &DrudeDatData) -> Result<()> {
    write_drude_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_osc_str_cache(path: &Path, data: &OscStrDatData) -> Result<()> {
    write_osc_str_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_hamaker_cache(path: &Path, data: &HamakerDatData) -> Result<()> {
    write_hamaker_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_optical_inputs, run_in_dir};
    use anyhow::Result;
    use ndarray::array;
    use num_complex::Complex64;
    use refeff_io::{
        DrudeDatData, EpsDatData, FullSpectrumInput, HamakerDatData, ModuleLogData, OscStrDatData,
        OscStrRow, fullspectrum_input_string, read_drude_dat, read_hamaker_dat,
        read_module_log_dat, read_opcons_dat, read_osc_str_dat, write_drude_dat, write_eps_dat,
        write_hamaker_dat, write_module_log_dat, write_osc_str_dat,
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

    fn sample_osc_str_dat() -> OscStrDatData {
        OscStrDatData {
            header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
            rows: vec![OscStrRow {
                component: "Cu".to_string(),
                edge: "K".to_string(),
                core_hole_index: 1,
                effective_electron_count: 5.123,
            }],
        }
    }

    fn sample_hamaker_dat() -> HamakerDatData {
        HamakerDatData {
            header_lines: vec!["# cached hamaker transform".to_string()],
            omega: array![1.0, 2.0, 4.0],
            imaginary_axis_epsilon: array![
                Complex64::new(0.35, 0.0),
                Complex64::new(0.25, 0.0),
                Complex64::new(0.10, 0.0),
            ],
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating full spectrum optical constants ...".to_string(),
                "Done with module: FULLSPECTRUM.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
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
        let expected_drude = read_drude_dat(temp.path().join("drude.dat"))?;

        assert_eq!(run_in_dir(temp.path())?, eps.point_count());

        assert_eq!(
            read_drude_dat(temp.path().join("drude.dat"))?,
            expected_drude
        );
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
    fn preserves_cached_fullspectrum_sidecars() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let eps = sample_eps_dat();
        write_fullspectrum_input(temp.path(), 1)?;
        write_eps_dat(temp.path().join("eps.dat"), &eps)?;
        write_osc_str_dat(temp.path().join("osc_str.dat"), &sample_osc_str_dat())?;
        write_hamaker_dat(temp.path().join("hamaker.dat"), &sample_hamaker_dat())?;
        write_module_log_dat(
            temp.path().join("logfullspectrum.dat"),
            &sample_module_log(),
        )?;
        let expected_osc_str = read_osc_str_dat(temp.path().join("osc_str.dat"))?;
        let expected_hamaker = read_hamaker_dat(temp.path().join("hamaker.dat"))?;
        let expected_log = read_module_log_dat(temp.path().join("logfullspectrum.dat"))?;

        assert_eq!(run_in_dir(temp.path())?, eps.point_count());

        assert_eq!(
            read_osc_str_dat(temp.path().join("osc_str.dat"))?,
            expected_osc_str
        );
        assert_eq!(
            read_hamaker_dat(temp.path().join("hamaker.dat"))?,
            expected_hamaker
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logfullspectrum.dat"))?,
            expected_log
        );
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
        assert!(!temp.path().join("sumrules.dat").exists());
        assert!(!temp.path().join("hamaker.dat").exists());
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
