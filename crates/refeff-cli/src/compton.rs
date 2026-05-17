use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    ComptonGridInput as CoreComptonGridInput, ComptonWindow as CoreComptonWindow,
    compton_build_grid, compton_profiles,
};
use refeff_io::{
    ComptonDatData, ComptonInput, JzzpDatData, RhozzpDatData, read_jzzp_dat, read_rhozzp_dat,
    write_compton_dat, write_rhozzp_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF COMPTON profile stage beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether cached FEFF COMPTON outputs can satisfy the requested work.
pub(crate) fn has_cached_outputs(work_dir: &Path) -> Result<bool> {
    let input = read_input(work_dir)?;
    if !input.run || input.switches.force_recalc_jzzp {
        return Ok(false);
    }

    let has_profile_cache = !input.switches.jpq || work_dir.join("jzzp.dat").is_file();
    let has_rhozzp_cache = !input.switches.rhozzp || work_dir.join("rhozzp.dat").is_file();
    Ok(has_profile_cache && has_rhozzp_cache && (input.switches.jpq || input.switches.rhozzp))
}

/// Run the FEFF COMPTON cached-output path.
///
/// Profile output is generated from an existing `jzzp.dat` cache. Requested
/// `rhozzp.dat` diagnostics are validated and re-rendered from the cached text
/// output when present. Rebuilding either cache from RHORRP density callbacks is
/// still outside the supported path.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.run {
        return Ok(0);
    }
    if input.switches.force_recalc_jzzp {
        bail!("COMPTON forced jzzp.dat recalculation requires the unported density callback path");
    }

    let rhozzp_rows = if input.switches.rhozzp {
        let rhozzp = read_cached_rhozzp(work_dir)?;
        let point_count = rhozzp.point_count();
        write_cached_rhozzp(work_dir, &rhozzp)?;
        point_count
    } else {
        0
    };

    if input.switches.jpq {
        let cache_path = work_dir.join("jzzp.dat");
        let cache = read_jzzp_dat(&cache_path)
            .with_context(|| format!("failed to read {}", cache_path.display()))?;
        let profile = calculate_profile(&input, &cache)?;
        let point_count = profile.point_count();
        let output_path = work_dir.join("compton.dat");
        write_compton_dat(&output_path, &profile)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        Ok(point_count + rhozzp_rows)
    } else {
        Ok(rhozzp_rows)
    }
}

fn read_input(work_dir: &Path) -> Result<ComptonInput> {
    let input_path = work_dir.join("compton.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    ComptonInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn calculate_profile(input: &ComptonInput, cache: &JzzpDatData) -> Result<ComptonDatData> {
    validate_cache_matches_input(input, cache)?;
    let grid = compton_build_grid(CoreComptonGridInput {
        ns: cache.ns,
        nphi: cache.nphi,
        nz: cache.nz,
        nzp: cache.nzp,
        smax: cache.smax,
        phimax: cache.phimax,
        zmax: cache.zmax,
        zpmax: cache.zpmax,
        norman_radius: 0.0,
        qhat: input.qhat,
    })
    .context("failed to build COMPTON integration grid from jzzp.dat")?;
    let window = profile_window(input.window.window_type);
    let momentum = compton_momentum_grid(input)?;
    let profile = compton_profiles(
        &grid,
        cache.values.view(),
        momentum.view(),
        window,
        input.window.cutoff,
    )
    .context("failed to evaluate COMPTON profile")?;

    Ok(ComptonDatData {
        header_lines: compton_dat_header(input, cache),
        ns: Some(cache.ns),
        nphi: Some(cache.nphi),
        nz: Some(cache.nz),
        nzp: Some(cache.nzp),
        zpmax: Some(cache.zpmax),
        temperature_ev: Some(input.temperature),
        momentum,
        profile,
    })
}

fn read_cached_rhozzp(work_dir: &Path) -> Result<RhozzpDatData> {
    let path = work_dir.join("rhozzp.dat");
    read_rhozzp_dat(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn write_cached_rhozzp(work_dir: &Path, data: &RhozzpDatData) -> Result<()> {
    let path = work_dir.join("rhozzp.dat");
    write_rhozzp_dat(&path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn validate_cache_matches_input(input: &ComptonInput, cache: &JzzpDatData) -> Result<()> {
    let ns = positive_i32_to_usize("ns", input.grid.ns)?;
    let nphi = positive_i32_to_usize("nphi", input.grid.nphi)?;
    let nz = positive_i32_to_usize("nz", input.grid.nz)?;
    let nzp = positive_i32_to_usize("nzp", input.grid.nzp)?;
    if (ns, nphi, nz, nzp) != (cache.ns, cache.nphi, cache.nz, cache.nzp) {
        bail!(
            "COMPTON jzzp.dat grid ({}, {}, {}, {}) does not match compton.inp grid ({ns}, {nphi}, {nz}, {nzp})",
            cache.ns,
            cache.nphi,
            cache.nz,
            cache.nzp
        );
    }
    Ok(())
}

fn positive_i32_to_usize(name: &'static str, value: i32) -> Result<usize> {
    if value <= 0 {
        bail!("COMPTON {name} must be positive, got {value}");
    }
    usize::try_from(value).with_context(|| format!("COMPTON {name} is out of range: {value}"))
}

fn compton_momentum_grid(input: &ComptonInput) -> Result<Array1<f64>> {
    if !input.momentum.pqmax.is_finite() || input.momentum.pqmax < 0.0 {
        bail!(
            "COMPTON pqmax must be finite and nonnegative, got {}",
            input.momentum.pqmax
        );
    }
    let npq = positive_i32_to_usize("npq", input.momentum.npq)?;
    if npq < 2 {
        bail!("COMPTON npq must be at least 2, got {npq}");
    }
    let scale = input.momentum.pqmax / (npq - 1) as f64;
    Ok(Array1::from_iter(
        (0..npq).map(|index| index as f64 * scale),
    ))
}

fn profile_window(window_type: i32) -> CoreComptonWindow {
    match window_type {
        0 => CoreComptonWindow::Rectangular,
        1 => CoreComptonWindow::CosineSquared,
        _ => CoreComptonWindow::Unwindowed,
    }
}

fn compton_dat_header(input: &ComptonInput, cache: &JzzpDatData) -> Vec<String> {
    vec![
        " # Compton profile, J(pq)".to_string(),
        format!(" # ns:          {:4}", cache.ns),
        format!(" # nphi:        {:4}", cache.nphi),
        format!(" # nz:          {:4}", cache.nz),
        format!(" # nzp:         {:4}", cache.nzp),
        format!(" # zpmax:   {:17.13}", cache.zpmax),
        format!(" # temperature (eV): {:14.7E}", input.temperature),
        " #----------------------------".to_string(),
        " # pq               J".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use ndarray::{Array2, ShapeBuilder};
    use refeff_io::{
        JzzpDatData, RhozzpDatData, parse_compton_dat, read_compton_dat, read_rhozzp_dat,
        write_jzzp_dat, write_rhozzp_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn compton_module_writes_profile_from_jzzp_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_compton_input(temp.path(), " T F F")?;
        write_jzzp_dat(
            temp.path().join("jzzp.dat"),
            &JzzpDatData {
                ns: 2,
                nphi: 2,
                nz: 3,
                nzp: 3,
                smax: 1.0,
                phimax: std::f64::consts::PI,
                zmax: 1.0,
                zpmax: 1.0,
                values: Array2::from_shape_fn((3, 3).f(), |(z, zp)| {
                    0.2 + z as f64 * 0.1 + zp as f64 * 0.05
                }),
            },
        )?;

        let count = run_in_dir(temp.path())?;

        let output = read_compton_dat(temp.path().join("compton.dat"))?;
        assert_eq!(count, 3);
        assert_eq!(output.point_count(), 3);
        assert_close(output.momentum[0], 0.0, 1.0e-12);
        assert_close(output.momentum[2], 1.0, 1.0e-12);
        assert!(output.profile.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn compton_module_preserves_cached_rhozzp_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let expected = sample_rhozzp_data();
        write_minimal_compton_input(temp.path(), " F T F")?;
        write_rhozzp_dat(temp.path().join("rhozzp.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected.point_count());
        assert_eq!(read_rhozzp_dat(temp.path().join("rhozzp.dat"))?, expected);
        assert!(!temp.path().join("compton.dat").exists());
        Ok(())
    }

    #[test]
    fn compton_module_rejects_missing_rhozzp_cache_until_density_callback_is_ported() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_minimal_compton_input(temp.path(), " T T F")?;

        let error = run_in_dir(temp.path())
            .err()
            .context("rhozzp generation should require the density callback path")?;

        assert!(
            error.to_string().contains("rhozzp.dat"),
            "unexpected error: {error:#}"
        );
        assert!(!temp.path().join("compton.dat").exists());
        Ok(())
    }

    #[test]
    fn compton_module_rejects_forced_jzzp_recalculation_until_density_callback_is_ported()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_minimal_compton_input(temp.path(), " T F T")?;

        let error = run_in_dir(temp.path())
            .err()
            .context("forced jzzp recalculation should require the density callback path")?;

        assert!(
            error.to_string().contains(
                "forced jzzp.dat recalculation requires the unported density callback path"
            )
        );
        assert!(!temp.path().join("compton.dat").exists());
        Ok(())
    }

    #[test]
    fn compton_module_matches_feff_reference_profile_from_cache_when_present() -> Result<()> {
        let Some(zip_path) = reference_compton_zip()? else {
            eprintln!("skipping COMPTON reference test; Cu REFERENCE.zip not found");
            return Ok(());
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            eprintln!("skipping COMPTON reference test; unzip command not found");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let input_text =
            String::from_utf8(unzip_reference_entry(&zip_path, "REFERENCE/compton.inp")?)?
                .replace(" T T F", " T F F");
        std::fs::write(temp.path().join("compton.inp"), input_text)?;
        std::fs::write(
            temp.path().join("jzzp.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/jzzp.dat")?,
        )?;
        let expected = parse_compton_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/compton.dat",
        )?)?)?;

        let count = run_in_dir(temp.path())?;

        let actual = read_compton_dat(temp.path().join("compton.dat"))?;
        assert_eq!(count, expected.point_count());
        assert_eq!(actual.point_count(), expected.point_count());
        for ((actual_momentum, expected_momentum), (actual_profile, expected_profile)) in actual
            .momentum
            .iter()
            .zip(expected.momentum.iter())
            .zip(actual.profile.iter().zip(expected.profile.iter()))
        {
            assert_close(*actual_momentum, *expected_momentum, 5.0e-7);
            assert_close(*actual_profile, *expected_profile, 4.0e-5);
        }
        Ok(())
    }

    fn reference_compton_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/COMPTON/Cu/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn write_minimal_compton_input(work_dir: &Path, switch_line: &str) -> Result<()> {
        std::fs::write(
            work_dir.join("compton.inp"),
            format!(
                concat!(
                    "run compton module?\n",
                    "           1\n",
                    "pqmax, npq\n",
                    "   1.00000000              3\n",
                    "ns, nphi, nz, nzp\n",
                    "   2   2   3   3\n",
                    "smax, phimax, zmax, zpmax\n",
                    "      1.00000      3.14159      1.00000      1.00000\n",
                    "jpq? rhozzp? force_recalc_jzzp?\n",
                    "{}\n",
                    "window_type (0=Step, 1=Hann), window_cutoff\n",
                    "           0   0.00000000    \n",
                    "temperature (in eV)\n",
                    "      0.00000\n",
                    "set_chemical_potential? chemical_potential(eV)\n",
                    " F   0.00000000    \n",
                    "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?\n",
                    " F F F F F\n",
                    "qhat_x qhat_y qhat_z\n",
                    "   0.0000000000000000        0.0000000000000000        1.0000000000000000     \n",
                ),
                switch_line
            ),
        )?;
        Ok(())
    }

    fn sample_rhozzp_data() -> RhozzpDatData {
        RhozzpDatData {
            header_lines: vec![" # rhozzp diagnostic".to_string()],
            z_prime: ndarray::Array1::from_vec(vec![0.01, 0.51, 1.01]),
            density: ndarray::Array1::from_vec(vec![0.45, 0.35, 0.15]),
        }
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to read {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
