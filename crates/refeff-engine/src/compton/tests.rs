use super::{has_supported_outputs, run_in_dir};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::{Complex32, Complex64};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    ConfigDatData, ConfigDatPotential, FmsCluster, FmsControl, FmsDebye, FmsInput, GeomDat,
    GeomDatRow, JzzpDatData, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
    PotBinData, PotBinScalars, PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp,
    PotRun, PotScattering, PotThermal, PotTolerances, RHORRP_POT_BIN_RADIAL_DX,
    RhorrpGgDiagBinData, RhorrpGgSliceBinData, RhozzpDatData, config_dat_string, fms_input_string,
    geom_dat_string, jzzp_dat_string, parse_compton_dat, phase_bin_string, pot_input_string,
    read_compton_dat, read_jzzp_dat, read_module_log_dat, read_rhozzp_dat,
    rhorrp_gg_diag_bin_bytes, rhorrp_gg_slice_bin_bytes, rhozzp_dat_string, write_jzzp_dat,
    write_module_log_dat, write_pot_bin, write_rhozzp_dat,
};
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn compton_module_writes_profile_from_jzzp_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected_cache = sample_jzzp_data();
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
fn compton_module_generates_missing_module_log_from_jzzp_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &sample_jzzp_data())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_module_log_dat(temp.path().join("logcompton.dat"))?,
        module_log_with_lines(&[
            "Calculating Compton scattering ...",
            "FEFF-serial using 1 thread.",
            "Calculating Compton profile",
            "Reusing previously calculated j(z,z')",
            "Calculate j(pq)",
            "Done with module: Compton scattering.",
        ])
    );
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
fn compton_module_generates_rhozzp_log_lines_from_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T T F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &sample_jzzp_data())?;
    write_rhozzp_dat(temp.path().join("rhozzp.dat"), &sample_rhozzp_data())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_eq!(
        read_module_log_dat(temp.path().join("logcompton.dat"))?,
        module_log_with_lines(&[
            "Calculating Compton scattering ...",
            "FEFF-serial using 1 thread.",
            "Calculating rho(z,z')",
            "Calculating Compton profile",
            "Reusing previously calculated j(z,z')",
            "Calculate j(pq)",
            "Done with module: Compton scattering.",
        ])
    );
    Ok(())
}

#[test]
fn compton_module_rejects_missing_rhozzp_cache_without_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T T F")?;

    let error = run_in_dir(temp.path())
        .err()
        .context("rhozzp generation should require RHORRP handoff files")?;

    assert!(
        error.to_string().contains("rhozzp.dat"),
        "unexpected error: {error:#}"
    );
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_does_not_claim_malformed_jzzp_cache_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F F")?;
    std::fs::write(temp.path().join("jzzp.dat"), b"not a jzzp.dat cache\n")?;

    assert!(!has_supported_outputs(temp.path())?);
    Ok(())
}

#[test]
fn compton_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_jzzp_data();
    std::fs::write(
        temp.path().join("compton.inp"),
        b"not a compton.inp handoff\n",
    )?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected)?;

    assert!(!has_supported_outputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed COMPTON input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("compton.inp"), "{chain}");
    assert_eq!(read_jzzp_dat(temp.path().join("jzzp.dat"))?, expected);
    assert!(!temp.path().join("compton.dat").exists());
    assert!(!temp.path().join("logcompton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_does_not_claim_cached_jzzp_with_malformed_rhorrp_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_jzzp_data();
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected)?;
    write_rhorrp_callback_handoffs(temp.path())?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

    assert!(!has_supported_outputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed RHORRP callback source should block cached COMPTON completion")?;
    let chain = format!("{error:#}");

    assert!(chain.contains("phase.bin"), "{chain}");
    assert_eq!(read_jzzp_dat(temp.path().join("jzzp.dat"))?, expected);
    assert!(!temp.path().join("compton.dat").exists());
    assert!(!temp.path().join("logcompton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_does_not_claim_orphan_outputs_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected_jzzp = sample_jzzp_data();
    let expected_rhozzp = sample_rhozzp_data();
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected_jzzp)?;
    write_rhozzp_dat(temp.path().join("rhozzp.dat"), &expected_rhozzp)?;

    assert!(!has_supported_outputs(temp.path())?);
    assert_eq!(read_jzzp_dat(temp.path().join("jzzp.dat"))?, expected_jzzp);
    assert_eq!(
        read_rhozzp_dat(temp.path().join("rhozzp.dat"))?,
        expected_rhozzp
    );
    assert!(!temp.path().join("compton.dat").exists());
    assert!(!temp.path().join("logcompton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_does_not_claim_incompatible_jzzp_cache_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut cache = sample_jzzp_data();
    cache.zpmax = 1.5;
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &cache)?;

    assert!(!has_supported_outputs(temp.path())?);
    Ok(())
}

#[test]
fn compton_module_does_not_claim_malformed_rhozzp_cache_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " F T F")?;
    std::fs::write(temp.path().join("rhozzp.dat"), b"not a rhozzp.dat cache\n")?;

    assert!(!has_supported_outputs(temp.path())?);
    Ok(())
}

#[test]
fn compton_module_rejects_forced_jzzp_recalculation_without_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F T")?;

    let error = run_in_dir(temp.path())
        .err()
        .context("forced jzzp recalculation should require RHORRP handoff files")?;

    assert!(
        error.to_string().contains(
            "forced jzzp.dat recalculation requires RHORRP density callback handoff files"
        )
    );
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_rejects_jzzp_limit_mismatch_without_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut cache = sample_jzzp_data();
    cache.zpmax = 1.5;
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &cache)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("jzzp limit mismatch should require RHORRP handoff files")?;

    assert!(
        error.to_string().contains("cached jzzp.dat zpmax"),
        "unexpected error: {error:#}"
    );
    assert!(
        error
            .to_string()
            .contains("recalculation requires RHORRP density callback handoff files")
    );
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_generates_jzzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F T")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let cache = read_jzzp_dat(temp.path().join("jzzp.dat"))?;
    assert_eq!((cache.ns, cache.nphi, cache.nz, cache.nzp), (2, 2, 3, 3));
    assert!(cache.values.iter().all(|value| value.is_finite()));
    let profile = read_compton_dat(temp.path().join("compton.dat"))?;
    assert_eq!(profile.point_count(), 3);
    assert!(profile.profile.iter().all(|value| value.is_finite()));
    let log = read_module_log_dat(temp.path().join("logcompton.dat"))?;
    assert_log_contains(&log, "Saving j(z,z')");
    Ok(())
}

#[test]
fn compton_module_recovers_malformed_jzzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F F")?;
    std::fs::write(temp.path().join("jzzp.dat"), "not a jzzp.dat cache\n")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let cache = read_jzzp_dat(temp.path().join("jzzp.dat"))?;
    assert_eq!((cache.ns, cache.nphi, cache.nz, cache.nzp), (2, 2, 3, 3));
    assert!(cache.values.iter().all(|value| value.is_finite()));
    let profile = read_compton_dat(temp.path().join("compton.dat"))?;
    assert_eq!(profile.point_count(), 3);
    assert!(profile.profile.iter().all(|value| value.is_finite()));
    let log = read_module_log_dat(temp.path().join("logcompton.dat"))?;
    assert_log_contains(&log, "Saving j(z,z')");
    Ok(())
}

#[test]
fn compton_module_regenerates_stale_readable_jzzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " T F F")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    run_in_dir(temp.path())?;
    let expected_cache = read_jzzp_dat(temp.path().join("jzzp.dat"))?;
    let expected_profile = read_compton_dat(temp.path().join("compton.dat"))?;
    let mut stale_cache = expected_cache.clone();
    stale_cache.values[[0, 0]] += 0.125;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &stale_cache)?;
    std::fs::remove_file(temp.path().join("logcompton.dat"))?;

    let count = run_in_dir(temp.path())?;

    let actual_cache = read_jzzp_dat(temp.path().join("jzzp.dat"))?;
    let actual_profile = read_compton_dat(temp.path().join("compton.dat"))?;
    assert_eq!(count, expected_profile.point_count());
    assert_ne!(
        jzzp_dat_string(&stale_cache)?,
        jzzp_dat_string(&expected_cache)?
    );
    assert_eq!(
        jzzp_dat_string(&actual_cache)?,
        jzzp_dat_string(&expected_cache)?
    );
    assert_eq!(actual_profile.point_count(), expected_profile.point_count());
    for ((actual_momentum, expected_momentum), (actual_value, expected_value)) in actual_profile
        .momentum
        .iter()
        .zip(expected_profile.momentum.iter())
        .zip(
            actual_profile
                .profile
                .iter()
                .zip(expected_profile.profile.iter()),
        )
    {
        assert_close(*actual_momentum, *expected_momentum, 1.0e-12);
        assert_close(*actual_value, *expected_value, 1.0e-12);
    }
    let log = read_module_log_dat(temp.path().join("logcompton.dat"))?;
    assert_log_contains(&log, "Saving j(z,z')");
    Ok(())
}

#[test]
fn compton_module_generates_rhozzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " F T F")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    let count = run_in_dir(temp.path())?;

    let rhozzp = read_rhozzp_dat(temp.path().join("rhozzp.dat"))?;
    assert_eq!(count, 1000);
    assert_eq!(rhozzp.point_count(), 1000);
    assert!(rhozzp.density.iter().all(|value| value.is_finite()));
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_recovers_malformed_rhozzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " F T F")?;
    std::fs::write(temp.path().join("rhozzp.dat"), "not a rhozzp.dat cache\n")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    let count = run_in_dir(temp.path())?;

    let rhozzp = read_rhozzp_dat(temp.path().join("rhozzp.dat"))?;
    assert_eq!(count, 1000);
    assert_eq!(rhozzp.point_count(), 1000);
    assert!(rhozzp.density.iter().all(|value| value.is_finite()));
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_regenerates_stale_readable_rhozzp_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_input(temp.path(), " F T F")?;
    write_rhorrp_callback_handoffs(temp.path())?;

    run_in_dir(temp.path())?;
    let expected = read_rhozzp_dat(temp.path().join("rhozzp.dat"))?;
    let mut stale = expected.clone();
    stale.density[0] += 0.125;
    write_rhozzp_dat(temp.path().join("rhozzp.dat"), &stale)?;

    let count = run_in_dir(temp.path())?;

    let actual = read_rhozzp_dat(temp.path().join("rhozzp.dat"))?;
    assert_eq!(count, expected.point_count());
    assert_ne!(rhozzp_dat_string(&stale)?, rhozzp_dat_string(&expected)?);
    assert_eq!(rhozzp_dat_string(&actual)?, rhozzp_dat_string(&expected)?);
    assert!(!temp.path().join("compton.dat").exists());
    Ok(())
}

#[test]
fn compton_module_matches_feff_reference_profile_from_cache_when_present() -> Result<()> {
    let Some(zip_path) = reference_compton_zip()? else {
        crate::require_fixture!("COMPTON reference test; Cu REFERENCE.zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        crate::require_fixture!("COMPTON reference test; unzip command not found");
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

fn sample_jzzp_data() -> JzzpDatData {
    JzzpDatData {
        ns: 2,
        nphi: 2,
        nz: 3,
        nzp: 3,
        smax: 1.0,
        phimax: (std::f64::consts::PI * 100_000.0).round() / 100_000.0,
        zmax: 1.0,
        zpmax: 1.0,
        values: Array2::from_shape_fn((3, 3).f(), |(z, zp)| {
            0.2 + z as f64 * 0.1 + zp as f64 * 0.05
        }),
    }
}

fn sample_rhozzp_data() -> RhozzpDatData {
    RhozzpDatData {
        header_lines: vec![" # rhozzp diagnostic".to_string()],
        z_prime: ndarray::Array1::from_vec(vec![0.01, 0.51, 1.01]),
        density: ndarray::Array1::from_vec(vec![0.45, 0.35, 0.15]),
    }
}

fn write_rhorrp_callback_handoffs(work_dir: &Path) -> Result<()> {
    write_pot_bin(work_dir.join("pot.bin"), &sample_callback_pot_bin())?;
    std::fs::write(
        work_dir.join("config.dat"),
        config_dat_string(&sample_callback_config_dat())?,
    )?;
    std::fs::write(
        work_dir.join("phase.bin"),
        phase_bin_string(&sample_callback_phase_bin())?,
    )?;
    std::fs::write(
        work_dir.join("pot.inp"),
        pot_input_string(&sample_callback_pot_input())?,
    )?;
    std::fs::write(
        work_dir.join("fms.inp"),
        fms_input_string(&sample_callback_fms_input())?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_callback_geom_dat())?,
    )?;
    std::fs::write(
        work_dir.join("gg_diag.bin"),
        rhorrp_gg_diag_bin_bytes(&sample_callback_gg_diag())?,
    )?;
    std::fs::write(
        work_dir.join("gg_slice.bin"),
        rhorrp_gg_slice_bin_bytes(&sample_callback_gg_slice())?,
    )?;
    Ok(())
}

fn sample_callback_geom_dat() -> GeomDat {
    GeomDat {
        nat: 2,
        nph: 1,
        model_atoms: vec![1, 2],
        atoms: vec![
            GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                boundary: 0,
            },
            GeomDatRow {
                index: 2,
                x: 0.8,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 0,
            },
        ],
    }
}

fn sample_callback_pot_bin() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["COMPTON RHORRP callback test".to_string()],
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            interstitial_potential: -1.2,
            interstitial_density: 0.03,
            edge_position: 9.1,
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            core_valence_energy: -3.0,
            density_radius: 1.7,
            fermi_momentum: 0.9,
            total_charge: 42.0,
            total_volume: 11.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12, 13]),
        muffin_tin_radii: Array1::from_vec(vec![1.1, 1.2]),
        norman_indices: Array1::from_vec(vec![20, 21]),
        atomic_numbers: Array1::from_vec(vec![29, 8]),
        kappa: Array1::from_iter(-20..=20),
        norman_radii: Array1::from_vec(vec![2.1, 2.2]),
        overlap_factors: Array1::from_vec(vec![0.9, 0.8]),
        max_overlap_factors: Array1::from_vec(vec![1.3, 1.4]),
        potential_multiplicities: Array1::from_vec(vec![1.0, 1.0]),
        ionization: Array1::from_vec(vec![0.0, 1.0]),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            0.001 * (row + 1) as f64
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            -0.001 * (row + 1) as f64
        }),
        large_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        large_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        electron_density: callback_radial_matrix(potentials, 0.01),
        coulomb_potential: callback_radial_matrix(potentials, -0.02),
        total_potential: callback_radial_matrix(potentials, -0.03),
        valence_density: callback_radial_matrix(potentials, 0.004),
        valence_potential: callback_radial_matrix(potentials, -0.005),
        magnetization_density: callback_radial_matrix(potentials, 0.0002),
        orbital_occupancy: Array2::from_shape_fn(
            (POT_BIN_ORBITALS, potentials),
            |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
        ),
        orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
            -10.0 + orbital as f64 * 0.25
        }),
        occupied_orbital_indices: Array2::from_shape_fn(
            (POT_BIN_IORB_SLOTS, potentials),
            |(slot, _)| slot as i32 - 5,
        ),
        norman_charges: Array1::from_vec(vec![28.5, 7.5]),
        valence_occupancy: Array2::from_shape_fn((1, potentials), |(_, potential)| {
            potential as f64
        }),
        raw_text: None,
    }
}

fn callback_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

fn sample_callback_config_dat() -> ConfigDatData {
    let mut first_occupations = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
    let mut first_valence = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
    first_occupations[0] = 1.0;
    first_occupations[1] = 2.0;
    first_valence[1] = 0.5;

    let mut second_occupations = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
    let mut second_valence = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
    second_occupations[0] = 2.0;
    second_occupations[1] = 2.0;
    second_occupations[2] = 1.0;
    second_valence[2] = 1.0;

    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![
            ConfigDatPotential {
                potential_index: 0,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: first_occupations,
                valence_occupations: first_valence,
                spin_occupations: None,
            },
            ConfigDatPotential {
                potential_index: 1,
                atomic_number: 8,
                element: "O".to_string(),
                occupations: second_occupations,
                valence_occupations: second_valence,
                spin_occupations: None,
            },
        ],
    }
}

fn sample_callback_phase_bin() -> PhaseBinData {
    let spin_count = 1;
    let energy_grid = Array1::from_vec(vec![
        Complex64::new(0.15, 0.02),
        Complex64::new(0.15, 0.04),
        Complex64::new(0.20, 0.04),
        Complex64::new(0.25, 0.04),
    ]);
    let energy_count = energy_grid.len();
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: energy_count,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 1,
        transition_count: 1,
        q_count: 1,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: 0.045,
            edge_energy: 9.8,
        },
        energy_grid,
        reference_energy: Array2::zeros((energy_count, spin_count)),
        potentials: vec![
            sample_callback_phase_potential(29, "Cu", energy_count, spin_count),
            sample_callback_phase_potential(8, "O", energy_count, spin_count),
        ],
        transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
        raw_pads: None,
    }
}

fn sample_callback_phase_potential(
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
) -> PhaseBinPotential {
    PhaseBinPotential {
        lmax: 0,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::zeros((energy_count, 1, spin_count)),
    }
}

fn sample_callback_pot_input() -> PotInput {
    PotInput {
        control: PotControl {
            mpot: 1,
            nph: 1,
            ntitle: 1,
            ihole: 1,
            ipr1: 0,
            iafolp: 0,
            ixc: 0,
            ispec: 0,
            iscfxc: 0,
        },
        run: PotRun {
            nmix: 0,
            nohole: 0,
            jumprm: 0,
            inters: 0,
            nscmt: 0,
            icoul: 0,
            lfms1: 0,
            iunf: 0,
        },
        titles: vec!["COMPTON RHORRP callback test".to_string()],
        scattering: PotScattering {
            gamach: 0.0,
            rgrd: RHORRP_POT_BIN_RADIAL_DX,
            ca1: 0.0,
            ecv: 0.0,
            totvol: 1.0,
            rfms1: 0.0,
            corval_emin: 0.0,
        },
        potentials: vec![
            PotPotential {
                z: 29,
                lmaxsc: 0,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
            PotPotential {
                z: 8,
                lmaxsc: 0,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
        ],
        external_pot: false,
        start_from_file: false,
        overlap_shells: vec![Vec::<PotOverlapShell>::new(), Vec::<PotOverlapShell>::new()],
        chsh_type: 0,
        config_type: 1,
        thermal: PotThermal {
            scf_temperature: 0.0,
            scf_thermal_vxc: 0,
            iscfth: 0,
            xntol: 0.0,
            nmu: 0,
            negrid: 0,
            emaxscf: 0.0,
        },
        finite_nucleus: false,
        warn_ion: false,
        ramp: PotRamp {
            ramp_scf: false,
            rfms_start: 0.0,
            nramp: 0,
        },
        tolerances: PotTolerances {
            tolmu: 0.0,
            tolq: 0.0,
            tolqp: 0.0,
        },
    }
}

fn sample_callback_fms_input() -> FmsInput {
    FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: 0,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 3.0,
            rdirec: 0.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 0.0,
            thetad: 0.0,
            sig2g: 0.0,
        },
        lmaxph: vec![0, 0],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    }
}

fn sample_callback_gg_diag() -> RhorrpGgDiagBinData {
    RhorrpGgDiagBinData {
        values: Array4::from_elem((4, 2, 1, 1), Complex32::new(0.0, 0.0)),
    }
}

fn sample_callback_gg_slice() -> RhorrpGgSliceBinData {
    RhorrpGgSliceBinData {
        values: Array3::from_elem((4, 1, 2), Complex32::new(0.0, 0.0)),
    }
}

fn module_log_with_lines(lines: &[&str]) -> ModuleLogData {
    ModuleLogData {
        lines: lines.iter().map(|line| (*line).to_string()).collect(),
        line_terminators: vec!["\n".to_string(); lines.len()],
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

fn assert_log_contains(log: &ModuleLogData, expected: &str) {
    assert!(
        log.lines.iter().any(|line| line.contains(expected)),
        "expected log to contain {expected:?}, got {:?}",
        log.lines
    );
}
