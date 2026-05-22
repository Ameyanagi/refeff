use super::run_in_dir;
use anyhow::{Context, Result};
use ndarray::{Array2, ShapeBuilder};
use refeff_io::{
    JzzpDatData, ModuleLogData, RhozzpDatData, parse_compton_dat, read_compton_dat, read_jzzp_dat,
    read_module_log_dat, read_rhozzp_dat, write_jzzp_dat, write_module_log_dat, write_rhozzp_dat,
};
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn compton_module_writes_profile_from_jzzp_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected_cache = JzzpDatData {
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
    };
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected_cache)?;
    write_module_log_dat(temp.path().join("logcompton.dat"), &sample_module_log())?;
    let expected_log = read_module_log_dat(temp.path().join("logcompton.dat"))?;

    let count = run_in_dir(temp.path())?;

    let output = read_compton_dat(temp.path().join("compton.dat"))?;
    assert_eq!(count, 3);
    assert_eq!(output.point_count(), 3);
    assert_eq!(read_jzzp_dat(temp.path().join("jzzp.dat"))?, expected_cache);
    assert_eq!(
        read_module_log_dat(temp.path().join("logcompton.dat"))?,
        expected_log
    );
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
    write_module_log_dat(temp.path().join("logcompton.dat"), &sample_module_log())?;
    let expected_log = read_module_log_dat(temp.path().join("logcompton.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, expected.point_count());
    assert_eq!(read_rhozzp_dat(temp.path().join("rhozzp.dat"))?, expected);
    assert_eq!(
        read_module_log_dat(temp.path().join("logcompton.dat"))?,
        expected_log
    );
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_rejects_missing_rhozzp_cache_until_density_callback_is_ported() -> Result<()> {
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
fn compton_module_rejects_forced_jzzp_recalculation_until_density_callback_is_ported() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F T")?;

    let error = run_in_dir(temp.path())
        .err()
        .context("forced jzzp recalculation should require the density callback path")?;

    assert!(
        error
            .to_string()
            .contains("forced jzzp.dat recalculation requires the unported density callback path")
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
    let input_text = String::from_utf8(unzip_reference_entry(&zip_path, "REFERENCE/compton.inp")?)?
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

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating Compton scattering ...".to_string(),
            "Done with module: COMPTON.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
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
