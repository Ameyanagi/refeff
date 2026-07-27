use super::*;
use refeff_io::{
    BandEnergyMesh, BandInput, CONFIG_DAT_ORBITAL_COUNT, CfAverage, ConfigDatData,
    ConfigDatPotential, FmsCluster, FmsControl, FmsDebye, FmsInput, GeomDat, GeomDatRow, GgDatData,
    GgDatSection, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GtrBinData,
    KmeshDatData, KmeshMetadata, KmeshRow, LossDatData, ReciprocalCell, ReciprocalInput,
    ReciprocalKMesh, RixsBroadening, RixsEnergyWindow, RixsInput, RixsSwitches, XsphAdvanced,
    XsphControl, XsphGrid, XsphInput, XsphInputSourceFormat, band_input_string, fms_input_string,
    geom_dat_string, global_input_string, pot_input_string, read_config_dat, read_kmesh_dat,
    read_phase_bin, reciprocal_input_string, rixs_input_string, write_aphase_hubbard_bin,
    write_config_dat, write_gg_bin, write_gtr_bin, write_kmesh_dat, xsph_input_string,
};

#[test]
fn full_run_completes_from_cached_pot_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_misc_dat(output.join("misc.dat"), &sample_misc_dat())?;
    write_convergence_scf(output.join("convergence.scf"), &sample_convergence_scf())?;
    write_convergence_scf_fine(
        output.join("convergence.scf.fine"),
        &sample_convergence_scf_fine(),
    )?;
    write_fort16(output.join("fort.16"), &sample_fort16())?;
    let expected_misc = read_misc_dat(output.join("misc.dat"))?;
    let expected_convergence = read_convergence_scf(output.join("convergence.scf"))?;
    let expected_convergence_fine = read_convergence_scf_fine(output.join("convergence.scf.fine"))?;
    let expected_fort16 = read_fort16(output.join("fort.16"))?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    assert_eq!(read_misc_dat(output.join("misc.dat"))?, expected_misc);
    assert_eq!(
        read_convergence_scf(output.join("convergence.scf"))?,
        expected_convergence
    );
    assert_eq!(
        read_convergence_scf_fine(output.join("convergence.scf.fine"))?,
        expected_convergence_fine
    );
    assert_eq!(read_fort16(output.join("fort.16"))?, expected_fort16);
    Ok(())
}

#[test]
fn full_run_generates_clean_xanes_cu_xmu_from_source_handoffs() -> Result<()> {
    let Some(source_input) = stock_xanes_cu_feff_input()? else {
        require_fixture!("clean XANES/Cu full-run acceptance; stock feff.inp not found");
    };

    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    std::fs::copy(source_input, &input)?;

    run_feff_to_dir(&input, &output)?;

    for name in [
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "vtot.dat",
        "phase.bin",
        "xsect.dat",
        "fms.bin",
        "feff.bin",
        "xmu.dat",
    ] {
        assert!(
            output.join(name).is_file(),
            "clean XANES/Cu run should generate {name}"
        );
    }
    assert!(read_wscrn_dat(output.join("wscrn.dat"))?.row_count() > 0);
    assert!(read_xmu_dat(output.join("xmu.dat"))?.point_count() > 0);
    Ok(())
}

#[test]
fn full_run_generates_clean_exafs_cu_chi_close_to_feff() -> Result<()> {
    const MAX_RELATIVE_L2: f64 = 5.0e-5;

    let Some(reference_dir) = reference_exafs_cu_full_run_case()? else {
        require_fixture!("clean EXAFS/Cu full-run parity; FEFF feff.inp/chi.dat not found");
    };

    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::copy(reference_dir.join("feff.inp"), &input)?;

    run_feff_to_dir(&input, &output)?;

    let actual = read_chi_dat(output.join("chi.dat"))?;
    let expected = read_chi_dat(reference_dir.join("chi.dat"))?;
    assert_eq!(
        actual.point_count(),
        expected.point_count(),
        "clean EXAFS/Cu chi.dat point count"
    );

    for row in 0..actual.point_count() {
        assert_float_close_with_tolerance(
            actual.wave_number[row],
            expected.wave_number[row],
            1.0e-12,
            &format!("clean EXAFS/Cu chi.dat wave number {row}"),
        );
    }

    let squared_error = actual
        .chi
        .iter()
        .zip(&expected.chi)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>();
    let squared_reference_norm = expected.chi.iter().map(|value| value.powi(2)).sum::<f64>();
    assert!(
        squared_reference_norm > 0.0,
        "FEFF EXAFS/Cu chi reference must have a non-zero L2 norm"
    );
    let relative_l2 = (squared_error / squared_reference_norm).sqrt();
    assert!(
        relative_l2 <= MAX_RELATIVE_L2,
        "clean EXAFS/Cu chi.dat relative L2 {relative_l2:.6e} exceeds {MAX_RELATIVE_L2:.6e}"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_pot_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(
        temp.path().join("pot.bin"),
        &sample_atomic_config_pot_bin_data(),
    )?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "pot"),
        "orphan pot.bin/apot.bin cache without pot.inp should not report POT complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_recovers_incomplete_pot_cache_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    let mut apot = sample_apot_bin_data();
    apot.sections.truncate(1);
    write_apot_bin(output.join("apot.bin"), &apot)?;

    run_feff_to_dir(&input, &output)?;

    assert!(read_apot_bin(output.join("apot.bin"))?.sections.len() > 1);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_does_not_advertise_malformed_shared_atomic_pot_log_before_atomic_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_misc_dat(output.join("misc.dat"), &sample_misc_dat())?;
    write_convergence_scf(output.join("convergence.scf"), &sample_convergence_scf())?;
    write_convergence_scf_fine(
        output.join("convergence.scf.fine"),
        &sample_convergence_scf_fine(),
    )?;
    write_fort16(output.join("fort.16"), &sample_fort16())?;
    std::fs::write(output.join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("malformed shared ATOM/POT log should fail before POT recovery")?;

    let message = format!("{error:#?}");
    assert!(
        message.contains("no supported cached stages were run"),
        "{message}"
    );
    assert!(
        message.contains("failed to run supported atomic stage"),
        "{message}"
    );
    assert!(message.contains("failed to read"), "{message}");
    assert!(message.contains("log1.dat"), "{message}");
    Ok(())
}

#[test]
fn full_run_skips_disabled_cached_pot_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_disabled_pot_cached_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    run_feff_to_dir(&input, &output)?;
    assert!(!output.join("pot00.dat").exists());
    assert!(!output.join("log1.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_atomic_and_pot_from_rdinp_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.inp").is_file());
    assert!(output.join("geom.dat").is_file());
    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot.bin").is_file());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_external_pot_from_mtdp_handoff_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_external_potential_no_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_pot_mtdp_data(),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(message.contains("corrected_momentum"), "{message}");
    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
    assert_eq!(pot.muffin_tin_indices[0], 7);
    assert!((pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
    assert!((pot.scalars.interstitial_potential + 0.75).abs() < 1.0e-10);
    assert!((pot.scalars.fermi_level + 0.10).abs() < 1.0e-10);
    assert!((pot.total_potential[(0, 0)] + 1.0).abs() < 1.0e-10);
    assert!((pot.total_potential[(2, 0)] + 1.2).abs() < 1.0e-10);
    assert!((pot.total_potential[(3, 0)] + 0.75).abs() < 1.0e-10);
    assert!((pot.electron_density[(0, 0)] - 0.11).abs() < 1.0e-10);
    assert!((pot.electron_density[(2, 0)] - 0.13).abs() < 1.0e-10);
    assert!((pot.electron_density[(3, 0)] - pot.scalars.interstitial_density).abs() < 1.0e-10);
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_restart_no_scf_pot_from_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_restart_no_scf_input(&input)?;
    let mut restart = sample_pot_bin_data();
    restart.atomic_numbers[0] = 4;
    restart.scalars.fermi_level = -0.125;
    restart.scalars.interstitial_potential = -0.275;
    restart.scalars.interstitial_density = 0.019;
    restart.total_potential.fill(-0.41);
    restart.electron_density.fill(0.023);
    restart.coulomb_potential.fill(123.0);
    restart.valence_density.fill(456.0);
    write_pot_bin(output.join("pot.bin"), &restart)?;
    let restart = read_pot_bin(output.join("pot.bin"))?;

    run_feff_to_dir(&input, &output)?;

    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
    assert_eq!(pot.total_potential, restart.total_potential);
    assert_eq!(pot.electron_density, restart.electron_density);
    assert!((pot.scalars.fermi_level - restart.scalars.fermi_level).abs() < 1.0e-10);
    assert!(
        (pot.scalars.interstitial_potential - restart.scalars.interstitial_potential).abs()
            < 1.0e-10
    );
    assert!(
        (pot.scalars.interstitial_density - restart.scalars.interstitial_density).abs() < 1.0e-10
    );
    assert_ne!(pot.coulomb_potential, restart.coulomb_potential);
    assert_ne!(pot.valence_density, restart.valence_density);
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_finite_nucleus_no_scf_pot_from_source_before_ff2x_zero_normalization_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_highz_no_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph=6 file(s)"), "{message}");
    assert!(message.contains("genfmt=3 file(s)"), "{message}");
    assert!(
        message.contains("FF2X xmu.dat normalization is zero"),
        "{message}"
    );
    assert!(!message.contains("pot-input="), "{message}");
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert!(pot_input.finite_nucleus);
    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
    assert!(pot.scalars.interstitial_density > 0.0);
    assert!(pot.scalars.fermi_level.is_finite());
    assert!(pot.large_components.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_high_exchange_no_scf_pot_from_source_before_xsph_valence_delta_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_high_exchange_no_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("valence_delta"), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert_eq!(pot_input.control.ixc, 6);
    assert_eq!(pot_input.run.nscmt, 0);
    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
    assert!(
        pot.total_potential
            .iter()
            .zip(pot.valence_potential.iter())
            .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
        "EXCHANGE 6 no-SCF full-run source path should preserve separate valence potential"
    );
    assert!(pot.scalars.fermi_level.is_finite());
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_applies_restart_after_external_pot_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_external_restart_no_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_pot_mtdp_data(),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;
    let mut restart = sample_pot_bin_data();
    restart.atomic_numbers[0] = 4;
    restart.scalars.fermi_level = -0.125;
    restart.scalars.interstitial_potential = -0.275;
    restart.scalars.interstitial_density = 0.019;
    restart.total_potential.fill(-0.41);
    restart.electron_density.fill(0.023);
    write_pot_bin(output.join("pot.bin"), &restart)?;
    let restart = read_pot_bin(output.join("pot.bin"))?;

    run_feff_to_dir(&input, &output)?;

    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
    assert_eq!(pot.muffin_tin_indices[0], 7);
    assert!((pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
    assert_eq!(pot.total_potential, restart.total_potential);
    assert_eq!(pot.electron_density, restart.electron_density);
    assert!((pot.scalars.fermi_level - restart.scalars.fermi_level).abs() < 1.0e-10);
    assert!(
        (pot.scalars.interstitial_potential - restart.scalars.interstitial_potential).abs()
            < 1.0e-10
    );
    assert!(
        (pot.scalars.interstitial_density - restart.scalars.interstitial_density).abs() < 1.0e-10
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_pot_bin_from_rdinp_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    std::fs::write(output.join("pot.bin"), "not pot.bin\n")?;
    std::fs::write(output.join("apot.bin"), "not apot.bin\n")?;

    run_feff_to_dir(&input, &output)?;

    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert_eq!(pot.potential_count(), 2);
    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_pot_bin_from_rdinp_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;

    run_feff_to_dir(&input, &output)?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    let mut stale_pot = expected_pot.clone();
    stale_pot.scalars.interstitial_density += 0.25;
    write_pot_bin(output.join("pot.bin"), &stale_pot)?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_apot_bin_from_rdinp_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;

    run_feff_to_dir(&input, &output)?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    let mut stale_apot = expected_apot.clone();
    add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
    write_apot_bin(output.join("apot.bin"), &stale_apot)?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_scf_pot_and_apot_from_rdinp_sources_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still expose the next source boundary")?;
    let message = error.to_string();
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to run FEFF xsph stage"), "{chain}");
    assert!(chain.contains("corrected_momentum"), "{chain}");
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    let mut stale_pot = expected_pot.clone();
    stale_pot.scalars.interstitial_density += 0.25;
    write_pot_bin(output.join("pot.bin"), &stale_pot)?;
    let mut stale_apot = expected_apot.clone();
    add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
    write_apot_bin(output.join("apot.bin"), &stale_apot)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still expose the next source boundary")?;

    let message = error.to_string();
    assert!(message.contains("pot="), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    let chain = format!("{error:#}");
    assert!(
        !chain.contains("failed to run supported pot stage"),
        "{chain}"
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_missing_scf_apot_from_rdinp_sources_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still expose the next source boundary")?;
    let message = error.to_string();
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to run FEFF xsph stage"), "{chain}");
    assert!(chain.contains("corrected_momentum"), "{chain}");
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    std::fs::remove_file(output.join("apot.bin"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still expose the next source boundary")?;

    let message = error.to_string();
    assert!(message.contains("pot="), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    let chain = format!("{error:#}");
    assert!(
        !chain.contains("failed to run supported pot stage"),
        "{chain}"
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_rejects_cached_pot_when_no_scf_source_selector_is_unsupported() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;
    execute_rdinp(&input, &output)?;
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let mut pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert_eq!(pot_input.run.nscmt, 0, "smoke fixture should be no-SCF");
    pot_input.control.iscfxc = 0;
    std::fs::write(output.join("pot.inp"), pot_input_string(&pot_input)?)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;

    let error = run_supported_cached_modules(&output)
        .err()
        .context("unsupported current no-SCF sources must not validate an older POT cache")?;
    let chain = format!("{error:#}");
    assert!(
        chain.contains("failed to run supported pot stage"),
        "{chain}"
    );
    assert!(chain.contains("iscfxc selector 0 is invalid"), "{chain}");
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(!output.join("pot00.dat").exists());
    assert!(!output.join("pot01.dat").exists());
    if output.join("log1.dat").is_file() {
        let log = std::fs::read_to_string(output.join("log1.dat"))?;
        assert!(
            !log.contains("Done with module: potentials."),
            "failed POT refresh must not advertise a completed POT stage: {log}"
        );
    }
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_pot_geometry_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;
    execute_rdinp(&input, &output)?;
    std::fs::write(output.join("geom.dat"), "not a geom.dat handoff\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports.iter().any(|report| {
            report.name == "pot" || report.name == "pot-input" || report.name == "pot-scf-source"
        }),
        "malformed POT geom.dat source should not report POT completion or validation-only handoffs: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!output.join("pot.bin").exists());
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_pot_when_geometry_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;
    execute_rdinp(&input, &output)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    std::fs::write(output.join("geom.dat"), "not a geom.dat handoff\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports.iter().any(|report| {
            report.name == "pot" || report.name == "pot-input" || report.name == "pot-scf-source"
        }),
        "malformed POT geom.dat source should block cached POT completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_pot_when_config_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;
    execute_rdinp(&input, &output)?;
    let pot_input_path = output.join("pot.inp");
    let mut pot_input = refeff_io::PotInput::parse_str(
        &pot_input_path,
        &std::fs::read_to_string(&pot_input_path)?,
    )?;
    pot_input.config_type = 2;
    pot_input.run.nscmt = 0;
    std::fs::write(&pot_input_path, pot_input_string(&pot_input)?)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    std::fs::write(output.join("config.inp"), "not a config.inp handoff\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports.iter().any(|report| {
            matches!(
                report.name,
                "atomic" | "atomic-config" | "pot" | "pot-input" | "pot-scf-source"
            )
        }),
        "malformed POT config.inp source should block cached ATOMIC/POT completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(!output.join("config.dat").exists());
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_reference_pot_scf_output_from_sources() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        require_fixture!(
            "POT full-run scheduler reference test; generated EXAFS/Cu reference not found"
        );
    };
    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "pot" && report.count > 0),
        "missing POT source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }
    Ok(())
}

#[test]
fn full_run_scheduler_runs_gecl4_true_scf_pot_row_parity_from_sources() -> Result<()> {
    let Some(reference_dir) = reference_xanes_gecl4_source_dir()? else {
        require_fixture!("GeCl4 POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_xanes_gecl4_pot_zip()? else {
        require_fixture!("GeCl4 POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("GeCl4 POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    let source_pot = reference_dir.join("pot.inp");
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert!(
        input.run.nscmt > 0,
        "GeCl4 reference should exercise the SCF branch"
    );
    input.run.nscmt = 1;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(reference_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 5);
    for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_electron_density_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_nio_hubbard_true_scf_pot_electron_density_parity_from_sources()
-> Result<()> {
    let Some(reference_dir) = reference_hubbard_nio_source_dir()? else {
        require_fixture!("NiO POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_hubbard_nio_pot_zip()? else {
        require_fixture!("NiO POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("NiO POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    let source_pot = reference_dir.join("pot.inp");
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 2,
        "NiO reference should exercise multiple unique potentials"
    );
    assert_eq!(
        input.run.nohole, -1,
        "NiO reference should exercise the screened core-hole branch"
    );
    assert!(
        input.run.nscmt > 0,
        "NiO reference should exercise the SCF branch"
    );
    input.run.nscmt = 2;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(reference_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 6);
    for name in [
        "pot.bin",
        "apot.bin",
        "pot00.dat",
        "pot01.dat",
        "pot02.dat",
        "log1.dat",
    ] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_electron_density_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(generated.valence_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_matches_nio_hubbard_bounded_feff_pot_reference_when_present() -> Result<()> {
    let Some(reference_pot) = reference_hubbard_nio_bounded_feff_pot_bin()? else {
        require_fixture!(
            "NiO POT bounded full-run parity test; no REFEFF_NIO_BOUNDED_FEFF_POT_BIN or reference-work/tmp/feff-pot-nio-bounded.*/pot.bin found"
        );
    };
    let Some(reference_dir) = reference_hubbard_nio_source_dir()? else {
        require_fixture!("NiO POT bounded full-run parity test; source reference not found");
    };

    let temp = tempfile::tempdir()?;
    let source_pot = reference_dir.join("pot.inp");
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert!(
        input.run.nscmt > 2,
        "NiO reference should be a longer SCF run than the bounded parity target"
    );
    input.run.nscmt = 2;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(reference_dir.join("geom.dat"), temp.path().join("geom.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 6);
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(reference_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_ldos_spin_true_scf_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_ldos_xanes_cu_spin_no_fms_xsph_source_dir()? else {
        require_fixture!("LDOS spin Cu POT full-run scheduler test; source reference not found");
    };
    let source_pot = reference_dir.join("pot.inp");
    let source_geom = reference_dir.join("geom.dat");
    if !source_pot.is_file() || !source_geom.is_file() {
        require_fixture!("LDOS spin Cu POT full-run scheduler test; POT sources not found");
    }

    let temp = tempfile::tempdir()?;
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 1,
        "LDOS spin Cu reference should exercise two potential columns"
    );
    assert_eq!(
        input.run.nohole, 2,
        "LDOS spin Cu reference should exercise final-state screening"
    );
    assert!(
        input.run.nscmt > 0,
        "LDOS spin Cu reference should exercise the SCF branch"
    );
    input.run.nscmt = 1;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(source_geom, temp.path().join("geom.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing LDOS spin Cu POT source report")?;
    assert_eq!(pot_report.count, 5);
    assert_eq!(pot_report.unit, "file(s)");
    assert!(
        !reports.iter().any(|report| report.name == "pot-scf-source"),
        "bounded LDOS spin Cu POT source output should complete instead of reporting a loop boundary: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(reference_dir.join("pot.bin"))?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.len(),
        reference.potential_multiplicities.len()
    );
    assert!(
        generated
            .potential_multiplicities
            .iter()
            .all(|multiplicity| multiplicity.is_finite() && *multiplicity > 0.0)
    );
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(generated.valence_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_bn_positive_totvol_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_bn_source_dir()? else {
        require_fixture!("BN POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_bn_pot_zip()? else {
        require_fixture!("BN POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("BN POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    let source_pot = reference_dir.join("pot.inp");
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 2,
        "BN reference should exercise three potential columns"
    );
    assert!(
        input.scattering.totvol > 0.0,
        "BN reference should exercise positive totvol"
    );
    assert!(
        input.run.nscmt > 0,
        "BN reference should exercise the SCF branch"
    );
    input.run.nscmt = 1;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(reference_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 6);
    for name in [
        "pot.bin",
        "apot.bin",
        "pot00.dat",
        "pot01.dat",
        "pot02.dat",
        "log1.dat",
    ] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_float_close_with_tolerance(
        generated.scalars.total_volume,
        reference.scalars.total_volume,
        1.0e-10,
        "BN POT total volume",
    );
    assert_pot_bin_reference_geometry_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_matches_bn_positive_totvol_bounded_feff_pot_reference_when_present()
-> Result<()> {
    let Some(reference_pot) = reference_bn_positive_totvol_bounded_feff_pot_bin()? else {
        require_fixture!(
            "BN POT bounded full-run parity test; no REFEFF_BN_POSITIVE_TOTVOL_BOUNDED_FEFF_POT_BIN or reference-work/tmp/feff-pot-bn-positive-totvol-bounded.*/pot.bin found"
        );
    };
    let Some(reference_dir) = reference_bn_source_dir()? else {
        require_fixture!("BN POT bounded full-run parity test; source reference not found");
    };

    let temp = tempfile::tempdir()?;
    let source_pot = reference_dir.join("pot.inp");
    let mut input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 2,
        "BN reference should exercise three potential columns"
    );
    assert!(
        input.scattering.totvol > 0.0,
        "BN reference should exercise positive totvol"
    );
    assert!(
        input.run.nscmt > 1,
        "BN reference should be a longer SCF run than the bounded parity target"
    );
    input.run.nscmt = 1;
    std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
    std::fs::copy(reference_dir.join("geom.dat"), temp.path().join("geom.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 6);
    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(reference_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_float_close_with_tolerance(
        generated.scalars.total_volume,
        reference.scalars.total_volume,
        1.0e-10,
        "BN POT total volume",
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_ybco_no_scf_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_ybco_source_dir()? else {
        require_fixture!("YBCO POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_ybco_pot_zip()? else {
        require_fixture!("YBCO POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("YBCO POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let source_pot = reference_dir.join("pot.inp");
    let input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 4,
        "YBCO reference should exercise five potential columns"
    );
    assert_eq!(input.run.nscmt, 0, "YBCO reference should be no-SCF");
    assert_eq!(
        input.run.nohole, -1,
        "YBCO reference should exercise screened core-hole bookkeeping"
    );
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 8);
    for name in [
        "pot.bin",
        "apot.bin",
        "pot00.dat",
        "pot01.dat",
        "pot02.dat",
        "pot03.dat",
        "pot04.dat",
        "log1.dat",
    ] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }

    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_sf6_no_scf_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_sf6_source_dir()? else {
        require_fixture!("SF6 POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_sf6_pot_zip()? else {
        require_fixture!("SF6 POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("SF6 POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let source_pot = reference_dir.join("pot.inp");
    let input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 1,
        "SF6 reference should exercise two potential columns"
    );
    assert_eq!(input.run.nscmt, 0, "SF6 reference should be no-SCF");
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 5);
    for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }

    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_mnf2_xmcd_no_scf_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_xmcd_mnf2_source_dir()? else {
        require_fixture!("MnF2 XMCD POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_xmcd_mnf2_pot_zip()? else {
        require_fixture!("MnF2 XMCD POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("MnF2 XMCD POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let source_pot = reference_dir.join("pot.inp");
    let input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 3,
        "MnF2 reference should exercise four potential columns"
    );
    assert_eq!(input.run.nscmt, 0, "MnF2 reference should be no-SCF");
    assert_eq!(
        input.run.nohole, -1,
        "MnF2 reference should exercise screened core-hole bookkeeping"
    );
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 7);
    for name in [
        "pot.bin",
        "apot.bin",
        "pot00.dat",
        "pot01.dat",
        "pot02.dat",
        "pot03.dat",
        "log1.dat",
    ] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }

    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_gd_l1_xmcd_no_scf_pot_source_output() -> Result<()> {
    let Some(reference_dir) = reference_xmcd_gd_l1_source_dir()? else {
        require_fixture!("Gd L1 XMCD POT full-run scheduler test; source reference not found");
    };
    let Some(zip_path) = reference_xmcd_gd_l1_pot_zip()? else {
        require_fixture!("Gd L1 XMCD POT full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("Gd L1 XMCD POT full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["pot.inp", "geom.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let source_pot = reference_dir.join("pot.inp");
    let input =
        refeff_io::PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
    assert_eq!(
        input.control.nph, 1,
        "Gd L1 reference should use two potentials"
    );
    assert_eq!(input.run.nscmt, 0, "Gd L1 reference should be no-SCF");
    assert_eq!(
        input.run.nohole, -1,
        "Gd L1 reference should exercise screened core-hole bookkeeping"
    );
    let expected_pot = temp.path().join("expected-pot.bin");
    std::fs::write(
        &expected_pot,
        unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let pot_report = reports
        .iter()
        .find(|report| report.name == "pot")
        .context("missing POT source report")?;
    assert_eq!(pot_report.count, 5);
    for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
        assert!(
            temp.path().join(name).is_file(),
            "expected full-run scheduler to write {name}"
        );
    }

    let generated = read_pot_bin(temp.path().join("pot.bin"))?;
    let reference = read_pot_bin(&expected_pot)?;
    assert_eq!(generated.potential_count(), reference.potential_count());
    assert_eq!(
        generated.atomic_numbers.to_vec(),
        reference.atomic_numbers.to_vec()
    );
    assert_eq!(generated.nohole, reference.nohole);
    assert_eq!(generated.ihole, reference.ihole);
    assert_eq!(
        generated.potential_multiplicities.to_vec(),
        reference.potential_multiplicities.to_vec()
    );
    assert_pot_bin_reference_rows_close(&generated, &reference);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(generated.electron_density.iter().any(|value| *value != 0.0));
    assert!(read_apot_bin(temp.path().join("apot.bin")).is_ok());
    Ok(())
}

#[test]
fn full_run_generates_regular_core_hole_iterative_pot_before_xsph_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream XSPH stage should still expose the next source boundary")?;

    let message = error.to_string();
    assert!(message.contains("atomic=4 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    assert!(output.join("pot.inp").is_file());
    assert!(output.join("geom.dat").is_file());
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert!(pot_input.run.nohole < 0);
    assert!(output.join("config.dat").is_file());
    assert!(output.join("apot.bin").is_file());
    assert!(read_pot_bin(output.join("pot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_writes_regular_core_hole_pot_before_xsph_boundary() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_iterative_scf_input(&input)?;
    execute_rdinp(&input, &output)?;

    let reports = run_supported_cached_modules(&output)?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "pot" && report.count > 0),
        "regular core-hole scheduler should report POT completion: {reports:?}"
    );
    assert!(
        reports
            .iter()
            .any(|report| report.name == "xsph-emesh" && report.count > 0),
        "regular core-hole scheduler should report XSPH emesh handoff completion: {reports:?}"
    );
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert!(pot_input.run.nohole < 0);
    assert!(output.join("config.dat").is_file());
    assert!(output.join("apot.bin").is_file());
    assert!(read_pot_bin(output.join("pot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ...")),
        "regular core-hole POT source path should write POT SCF log: {log:?}"
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials.")),
        "regular core-hole POT source path should write completed POT log: {log:?}"
    );
    Ok(())
}

#[test]
fn full_run_treats_restart_pot_bin_as_scf_source_not_final_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_restart_iterative_scf_input(&input)?;
    let mut restart = sample_pot_bin_data();
    restart.atomic_numbers[0] = 4;
    write_pot_bin(output.join("pot.bin"), &restart)?;
    let expected_restart = read_pot_bin(output.join("pot.bin"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("required POT stage should still require complete source handoffs or caches")?;

    let message = error.to_string();
    assert!(
        message.contains("pot-scf-source=1 source bundle(s)"),
        "{message}"
    );
    assert!(!message.contains("pot="), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    let chain = format!("{error:#}");
    assert!(
        chain.contains("POT required stage needs complete source handoffs"),
        "{chain}"
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_restart);
    assert!(output.join("config.dat").is_file());
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_writes_restart_iterative_scf_pot_from_compatible_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");

    write_restart_iterative_scf_input(&input)?;
    std::fs::copy(seed_output.join("pot.bin"), output.join("pot.bin"))?;
    let restart = read_pot_bin(output.join("pot.bin"))?;

    run_feff_to_dir(&input, &output)?;

    let generated = read_pot_bin(output.join("pot.bin"))?;
    assert_ne!(
        generated.electron_density, restart.electron_density,
        "compatible START_FROM_FILE SCF should replace the restart pot.bin with terminal source output"
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ...")),
        "restart SCF source path should write POT SCF log: {log:?}"
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials.")),
        "restart SCF source path should write completed POT log: {log:?}"
    );
    Ok(())
}

#[test]
fn full_run_writes_external_iterative_scf_pot_from_compatible_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");
    let seed_pot = read_pot_bin(seed_output.join("pot.bin"))?;

    write_external_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;

    run_feff_to_dir(&input, &output)?;

    let generated = read_pot_bin(output.join("pot.bin"))?;
    assert_ne!(
        generated, seed_pot,
        "compatible EXTPOT SCF source route should not preserve the normal seed output"
    );
    assert_eq!(generated.potential_count(), 1);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ...")),
        "external SCF source path should write POT SCF log: {log:?}"
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials.")),
        "external SCF source path should write completed POT log: {log:?}"
    );
    Ok(())
}

#[test]
fn full_run_regenerates_stale_external_scf_pot_from_compatible_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");
    let seed_pot = read_pot_bin(seed_output.join("pot.bin"))?;

    write_external_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;

    run_feff_to_dir(&input, &output)?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;

    let mut stale_pot = expected_pot.clone();
    stale_pot.scalars.interstitial_density += 0.25;
    write_pot_bin(output.join("pot.bin"), &stale_pot)?;
    let mut stale_apot = expected_apot.clone();
    add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
    write_apot_bin(output.join("apot.bin"), &stale_apot)?;

    run_feff_to_dir(&input, &output)?;
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_external_pot_when_sort_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");
    let seed_pot = read_pot_bin(seed_output.join("pot.bin"))?;

    write_external_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;

    run_feff_to_dir(&input, &output)?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    std::fs::write(output.join("sort.aip"), "not a sort.aip handoff\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports
            .iter()
            .any(|report| matches!(report.name, "pot" | "pot-input" | "pot-scf-source")),
        "malformed external POT sort source should block cached POT completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_external_pot_when_mtdp_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");
    let seed_pot = read_pot_bin(seed_output.join("pot.bin"))?;

    write_external_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;

    run_feff_to_dir(&input, &output)?;
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    std::fs::write(output.join("GeCl4.04.dft.mtdp"), "not an MTDP handoff\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports
            .iter()
            .any(|report| matches!(report.name, "pot" | "pot-input" | "pot-scf-source")),
        "malformed external POT MTDP source should block cached POT completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    Ok(())
}

#[test]
fn full_run_writes_external_restart_iterative_scf_pot_from_compatible_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let seed_input = temp.path().join("seed.inp");
    let seed_output = temp.path().join("seed");
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&seed_output)?;
    std::fs::create_dir_all(&output)?;
    write_iterative_scf_input(&seed_input)?;
    let seed_error = run_feff_to_dir(&seed_input, &seed_output)
        .err()
        .context("seed run should stop after writing source-backed POT output")?;
    let seed_message = seed_error.to_string();
    assert!(seed_message.contains("pot=4 file(s)"), "{seed_message}");
    let seed_pot = read_pot_bin(seed_output.join("pot.bin"))?;

    write_external_restart_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_scf_mtdp_data(&seed_pot),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;
    write_pot_bin(output.join("pot.bin"), &seed_pot)?;
    let restart = read_pot_bin(output.join("pot.bin"))?;

    run_feff_to_dir(&input, &output)?;

    let generated = read_pot_bin(output.join("pot.bin"))?;
    assert_ne!(
        generated.electron_density, restart.electron_density,
        "compatible EXTPOT + START_FROM_FILE SCF should replace the restart pot.bin with terminal source output"
    );
    assert_eq!(generated.potential_count(), 1);
    assert!(generated.scalars.fermi_level.is_finite());
    assert!(generated.scalars.interstitial_density > 0.0);
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ...")),
        "external restart SCF source path should write POT SCF log: {log:?}"
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials.")),
        "external restart SCF source path should write completed POT log: {log:?}"
    );
    Ok(())
}

#[test]
fn full_run_treats_external_restart_pot_bin_as_scf_source_not_final_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_external_restart_iterative_scf_input(&input)?;
    refeff_io::write_mtdp(
        output.join("GeCl4.04.dft.mtdp"),
        &sample_external_pot_mtdp_data(),
    )?;
    std::fs::write(output.join("sort.aip"), "0\n")?;
    let mut restart = sample_pot_bin_data();
    restart.atomic_numbers[0] = 4;
    restart.scalars.fermi_level = -0.125;
    restart.scalars.interstitial_potential = -0.275;
    restart.scalars.interstitial_density = 0.019;
    restart.total_potential.fill(-0.41);
    restart.electron_density.fill(0.023);
    write_pot_bin(output.join("pot.bin"), &restart)?;
    let expected_restart = read_pot_bin(output.join("pot.bin"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("required POT stage should still require complete source handoffs or caches")?;

    let message = error.to_string();
    assert!(
        message.contains("pot-scf-source=1 source bundle(s)"),
        "{message}"
    );
    assert!(!message.contains("pot="), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    let chain = format!("{error:#}");
    assert!(
        chain.contains("POT required stage needs complete source handoffs"),
        "{chain}"
    );
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert!(pot_input.external_pot);
    assert!(pot_input.start_from_file);
    assert!(pot_input.run.nscmt > 0);
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_restart);
    assert!(output.join("config.dat").is_file());
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_carries_highz_finite_nucleus_iterative_pot_source_to_repeat_boundary() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_highz_iterative_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("required POT stage should still require complete source handoffs or caches")?;

    let message = error.to_string();
    assert!(
        message.contains("pot-scf-source=1 source bundle(s)"),
        "{message}"
    );
    assert!(!message.contains("pot-input="), "{message}");
    let chain = format!("{error:#}");
    assert!(
        chain.contains("POT required stage needs complete source handoffs"),
        "{chain}"
    );
    let pot_text = std::fs::read_to_string(output.join("pot.inp"))?;
    let pot_input = refeff_io::PotInput::parse_str(output.join("pot.inp"), &pot_text)?;
    assert!(pot_input.finite_nucleus);
    assert!(output.join("config.dat").is_file());
    assert!(output.join("apot.bin").is_file());
    assert!(!output.join("pot.bin").exists());
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_validates_high_exchange_iterative_pot_source_before_xsph_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_high_exchange_iterative_scf_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(!message.contains("pot-scf-source="), "{message}");
    assert!(!message.contains("pot-input="), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    let pot = read_pot_bin(output.join("pot.bin"))?;
    assert!(
        pot.total_potential
            .iter()
            .zip(pot.valence_potential.iter())
            .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
        "EXCHANGE 5 full-run source path should preserve separate valence potential"
    );
    assert!(output.join("apot.bin").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_high_exchange_scf_pot_from_rdinp_sources_before_xsph_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_high_exchange_iterative_scf_input(&input)?;

    let first_error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;
    let first_message = format!("{first_error:#?}");
    assert!(first_message.contains("pot=4 file(s)"), "{first_message}");
    assert!(
        !first_message.contains("pot-scf-source="),
        "{first_message}"
    );
    assert!(
        first_message.contains("failed to run FEFF xsph stage"),
        "{first_message}"
    );
    let expected_pot = read_pot_bin(output.join("pot.bin"))?;
    let expected_apot = read_apot_bin(output.join("apot.bin"))?;
    assert!(
        expected_pot
            .total_potential
            .iter()
            .zip(expected_pot.valence_potential.iter())
            .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
        "EXCHANGE 5 source path should preserve separate valence potential"
    );

    let mut stale_pot = expected_pot.clone();
    stale_pot.scalars.interstitial_density += 0.25;
    write_pot_bin(output.join("pot.bin"), &stale_pot)?;
    let mut stale_apot = expected_apot.clone();
    add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
    write_apot_bin(output.join("apot.bin"), &stale_apot)?;

    let second_error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream stage should still require more source state")?;
    let second_message = format!("{second_error:#?}");
    assert!(second_message.contains("pot=4 file(s)"), "{second_message}");
    assert!(
        !second_message.contains("pot-scf-source="),
        "{second_message}"
    );
    assert!(
        second_message.contains("failed to run FEFF xsph stage"),
        "{second_message}"
    );
    assert_eq!(read_pot_bin(output.join("pot.bin"))?, expected_pot);
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected_apot);
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("log1.dat").is_file());
    Ok(())
}

#[test]
fn full_run_completes_from_cached_atomic_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    run_feff_to_dir(&input, &output)?;

    assert!(
        read_apot_bin(output.join("apot.bin"))?.sections.len()
            > sample_apot_bin_data().sections.len()
    );
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_pot_input_for_atomic_or_pot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("pot.inp"), b"not a pot.inp handoff\n")?;
    write_pot_bin(
        temp.path().join("pot.bin"),
        &sample_atomic_config_pot_bin_data(),
    )?;
    write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin_data())?;
    let pot = read_pot_bin(temp.path().join("pot.bin"))?;
    let apot = read_apot_bin(temp.path().join("apot.bin"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| {
            !matches!(
                report.name,
                "atomic" | "atomic-config" | "pot" | "pot-input" | "pot-scf-source"
            )
        }),
        "malformed pot.inp should not report ATOMIC or POT complete: {:?}",
        reports
    );
    assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, pot);
    assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, apot);
    assert!(!temp.path().join("config.dat").exists());
    assert!(!temp.path().join("log1.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_atomic_log_for_config_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    std::fs::write(output.join("config.dat"), "not config.dat\n")?;
    std::fs::write(output.join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;
    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_atomic_fpf0_from_source_handoff() -> Result<()> {
    let Some(reference_dir) = reference_atomic_dir()? else {
        require_fixture!(
            "ATOM fpf0 full-run recovery test; generated EXAFS/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    std::fs::copy(reference_dir.join("apot.bin"), output.join("apot.bin"))?;
    std::fs::copy(reference_dir.join("fort.16"), output.join("fort.16"))?;
    std::fs::write(output.join("fpf0.dat"), "not fpf0.dat\n")?;
    let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

    run_feff_to_dir(&reference_dir.join("feff.inp"), &output)?;
    assert_fpf0_close(&read_fpf0_dat(output.join("fpf0.dat"))?, &expected_fpf0);
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module:")),
        "{log:?}"
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_atomic_cache_from_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    std::fs::write(output.join("apot.bin"), "not apot.bin\n")?;

    run_feff_to_dir(&input, &output)?;

    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_replaces_malformed_atomic_apot_from_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    std::fs::write(output.join("apot.bin"), "not apot.bin\n")?;

    run_feff_to_dir(&input, &output)?;
    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_atomic_log_after_source_apot_replaces_malformed_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    std::fs::write(output.join("apot.bin"), "not apot.bin\n")?;
    std::fs::write(output.join("config.dat"), "not config.dat\n")?;
    std::fs::write(output.join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_refreshes_xsph_outputs_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_cached_input(&input)?;
    let phase = sample_phase_bin_data();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(output.join("phase.bin"), &phase)?;
    write_xsect_dat(output.join("xsect.dat"), &xsect)?;
    write_mpse_dat(output.join("mpse.dat"), &sample_mpse_dat())?;
    write_emesh_dat(output.join("emesh.dat"), &sample_emesh_dat())?;
    write_emesh_bin(output.join("emesh.bin"), &sample_emesh_bin())?;
    let seed_phase = read_phase_bin(output.join("phase.bin"))?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    let phase = read_phase_bin(output.join("phase.bin"))?;
    assert!(
        phase.energy_count > seed_phase.energy_count,
        "completed no-SCF run should refresh the small synthetic phase cache"
    );
    assert!(read_xsect_dat(output.join("xsect.dat"))?.energy_count() > 0);
    assert!(read_mpse_dat(output.join("mpse.dat"))?.point_count() > 0);
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    let log = read_module_log_dat(output.join("log2.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating cross-section and phases ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("    absorption cross section"))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: cross-section and phases (XSPH)."))
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_cache_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_cached_input(&input)?;
    std::fs::write(output.join("phase.bin"), "not phase.bin\n")?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    assert!(read_phase_bin(output.join("phase.bin")).is_ok());
    assert!(read_xsect_dat(output.join("xsect.dat")).is_ok());
    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert!(output.join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_xsph_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let phase = sample_phase_bin_data();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| {
            !matches!(
                report.name,
                "xsph" | "xsph-phase" | "xsph-phase-text" | "xsph-emesh"
            )
        }),
        "orphan phase.bin/xsect.dat cache without xsph.inp should not report XSPH complete: {:?}",
        reports
    );
    assert_eq!(read_phase_bin(temp.path().join("phase.bin"))?, phase);
    assert_eq!(read_xsect_dat(temp.path().join("xsect.dat"))?, xsect);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_xsph_with_stale_xsect_energy_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_cached_input(&temp.path().join("xsph.inp"))?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    let mut stale_xsect = sample_xsect_dat_for_phase(&phase);
    stale_xsect.energy_grid_ev[0] += Complex64::new(1.0, 0.0);
    write_xsect_dat(temp.path().join("xsect.dat"), &stale_xsect)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "same-shape stale xsect.dat energy grid should not report completed XSPH: {reports:?}"
    );
    assert!(
        (read_xsect_dat(temp.path().join("xsect.dat"))?.energy_grid_ev[0].re
            - stale_xsect.energy_grid_ev[0].re)
            .abs()
            <= 5.0e-5
    );
    Ok(())
}

#[test]
fn full_run_generates_xsph_outputs_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(phase.potentials[0].label, "Cu");
    assert!(
        phase.potentials[0]
            .phase_shifts
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );

    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));

    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(read_fms_bin(output.join("fms.bin"))?.energy_count > 0);
    let paths = read_paths_dat(output.join("paths.dat"))?;
    assert!(!paths.titles.is_empty());
    let feff = read_feff_bin(output.join("feff.bin"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert!(feff.energy_count() > 0);
    assert!(!read_list_dat(output.join("list.dat"))?.titles.is_empty());
    let xmu = read_xmu_dat(output.join("xmu.dat"))?;
    let chi = read_chi_dat(output.join("chi.dat"))?;
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(xmu.mu.iter().all(|value| value.is_finite()));
    assert!(chi.chi.iter().all(|value| value.is_finite()));
    let log = read_module_log_dat(output.join("log2.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: cross-section and phases (XSPH)."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_generates_xanes_cu_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_xanes_cu_xsph_source_dir()? else {
        require_fixture!("XSPH XANES/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "normal XANES XSPH source handoff should report a completed XSPH stage: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "XANES/Cu scheduler xsect.dat",
    );
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_debye_dm_xanes_cu_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_debye_dm_xanes_cu_xsph_source_dir()? else {
        require_fixture!("XSPH DEBYE/DM/XANES/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin", "config.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH DEBYE/DM/XANES/Cu source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete DEBYE/DM/XANES/Cu XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "DEBYE/DM/XANES/Cu scheduler xsect.dat",
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_exafs_cu_scf_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_exafs_cu_scf_xsph_source_dir()? else {
        require_fixture!("XSPH EXAFS/Cu_SCF full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin", "config.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH EXAFS/Cu_SCF source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete EXAFS/Cu_SCF XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "EXAFS/Cu_SCF scheduler xsect.dat",
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_danes_cu_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_danes_cu_xsph_source_dir()? else {
        require_fixture!("XSPH DANES/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH DANES/Cu source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH DANES source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete DANES/Cu XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close_with_tolerance(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "DANES/Cu scheduler xsect.dat",
        5.0e-5,
        0.20,
        1.0e-4,
        0.25,
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_xes_cu_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(zip_path) = reference_xes_cu_xsph_zip()? else {
        require_fixture!("XSPH XES/Cu full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("XSPH XES/Cu full-run scheduler test; unzip command not found");
    }

    let reference = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
        "phase.bin",
        "xsect.dat",
        "emesh.dat",
        "emesh.bin",
    ] {
        std::fs::write(
            reference.path().join(name),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
        )
        .with_context(|| format!("failed to extract XES/Cu {name}"))?;
    }

    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
    ] {
        std::fs::copy(reference.path().join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference.path().join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference.path().join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference.path().join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(!temp.path().join("emesh.dat").exists());
    assert!(!temp.path().join("emesh.bin").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH XES/Cu source report")?;
    assert!(
        report.count >= 6,
        "completed XSPH XES source report should include base phase/xsect sidecars and AXAFS: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete XES/Cu XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        1.0e-4,
    );
    assert_reference_xsect_dat_close(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "XES/Cu scheduler xsect.dat",
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(
        temp.path().join("axafs.dat").is_file(),
        "XES/Cu ipr2 source handoff should generate axafs.dat"
    );
    assert!(temp.path().join("mpse.dat").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_fprime_gecl4_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_fprime_gecl4_xsph_source_dir()? else {
        require_fixture!("XSPH FPRIME/GeCl4 full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin", "config.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH FPRIME/GeCl4 source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH FPRIME source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete FPRIME/GeCl4 XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert!(
        !temp.path().join("mpse.dat").exists(),
        "FPRIME XSPH source handoff should not require mpse.dat"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close_with_tolerance(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "FPRIME/GeCl4 scheduler xsect.dat",
        5.0e-5,
        0.20,
        5.0e-5,
        0.25,
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_elnes_cu_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_elnes_cu_xsph_source_dir()? else {
        require_fixture!("XSPH ELNES/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH ELNES/Cu source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete ELNES/Cu XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close_allow_zero_with_tolerance(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "ELNES/Cu scheduler xsect.dat",
        3.0e-4,
        0.20,
        3.0e-4,
        0.25,
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_ldos_spin_no_fms_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let Some(reference_dir) = reference_ldos_xanes_cu_spin_no_fms_xsph_source_dir()? else {
        require_fixture!(
            "XSPH LDOS/XANES_Cu_spin_no_fms full-run scheduler test; reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "wscrn.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH LDOS/XANES_Cu_spin_no_fms source report")?;
    assert!(
        report.count >= 5,
        "completed XSPH LDOS source report should include base phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "complete LDOS/XANES_Cu_spin_no_fms XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        5.0e-5,
    );
    assert_reference_xsect_dat_close(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &expected_xsect,
        "LDOS/XANES_Cu_spin_no_fms scheduler xsect.dat",
    );
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        5.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        5.0e-5,
    );
    assert!(temp.path().join("mpse.dat").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_remaining_ldos_xsph_reference_phase_and_xsect_from_source_handoffs()
-> Result<()> {
    let mut fixtures = Vec::new();
    if let Some(reference_dir) = reference_ldos_xanes_cu_fms_xsph_source_dir()? {
        fixtures.push(("LDOS/XANES_Cu_fms", reference_dir));
    } else {
        crate::record_missing_fixture!(
            "XSPH LDOS/XANES_Cu_fms full-run scheduler test; reference not found"
        );
    }
    if let Some(reference_dir) = reference_ldos_xanes_cu_spin_fms_short_xsph_source_dir()? {
        fixtures.push(("LDOS/XANES_Cu_spin_fms_short", reference_dir));
    } else {
        crate::record_missing_fixture!(
            "XSPH LDOS/XANES_Cu_spin_fms_short full-run scheduler test; reference not found"
        );
    }

    if fixtures.is_empty() {
        return Ok(());
    }

    for (label, reference_dir) in fixtures {
        let temp = tempfile::tempdir()?;
        for name in [
            "xsph.inp",
            "global.inp",
            "pot.bin",
            "config.dat",
            "wscrn.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))
                .with_context(|| format!("failed to copy {label} {name}"))?;
        }
        let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
        let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
        let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
        let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

        assert!(!temp.path().join("phase.bin").exists());
        assert!(!temp.path().join("xsect.dat").exists());
        let reports = run_supported_cached_modules(temp.path())?;

        let report = reports
            .iter()
            .find(|report| report.name == "xsph")
            .with_context(|| format!("missing XSPH {label} source report"))?;
        assert!(
            report.count >= 5,
            "completed XSPH {label} source report should include base phase/xsect sidecars: {reports:?}"
        );
        assert!(
            !reports.iter().any(|report| report.name == "xsph-phase"),
            "complete {label} XSPH source handoff should report xsph, not xsph-phase: {reports:?}"
        );
        assert_reference_phase_bin_close(
            &read_phase_bin(temp.path().join("phase.bin"))?,
            &expected_phase,
            5.0e-5,
        );
        let xsect_label = format!("{label} scheduler xsect.dat");
        assert_reference_xsect_dat_close(
            &read_xsect_dat(temp.path().join("xsect.dat"))?,
            &expected_xsect,
            &xsect_label,
        );
        assert_emesh_dat_close(
            &read_emesh_dat(temp.path().join("emesh.dat"))?,
            &expected_emesh,
            5.0e-5,
        );
        assert_emesh_bin_close(
            &read_emesh_bin(temp.path().join("emesh.bin"))?,
            &expected_emesh_bin,
            5.0e-5,
        );
        assert!(temp.path().join("mpse.dat").is_file());
        assert!(temp.path().join("log2.dat").is_file());
    }
    Ok(())
}

#[test]
fn full_run_scheduler_runs_nrixs_gecl4_xsph_source_handoff() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_gecl4_xsph_phase_dir()? else {
        require_fixture!("XSPH NRIXS full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin", "config.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing XSPH source report")?;
    assert!(
        report.count >= 6,
        "completed NRIXS XSPH source report should include phase/xsect/xsecl sidecars: {reports:?}"
    );
    assert!(
        !reports.iter().any(|report| report.name == "xsph-phase"),
        "completed NRIXS XSPH source handoff should report the full XSPH stage: {reports:?}"
    );
    assert_reference_phase_bin_close(
        &read_phase_bin(temp.path().join("phase.bin"))?,
        &expected_phase,
        1.0e-4,
    );
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
            || xsect.cross_section.iter().any(|value| value.norm() > 0.0),
        "expected source-generated NRIXS/JAS xsect.dat rows"
    );
    assert!(temp.path().join("xsecl.dat").is_file());
    assert!(temp.path().join("xsecl2.dat").is_file());
    assert!(temp.path().join("xsecl.bin").is_file());
    assert_emesh_dat_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        1.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_without_xsectjas_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    let expected_phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "incomplete NRIXS xsectjas sidecar set should not report completed XSPH: {reports:?}"
    );
    assert!(
        !temp.path().join("xsecl.dat").exists(),
        "scheduler should not synthesize unsupported xsectjas text sidecar"
    );
    assert!(
        !temp.path().join("xsecl2.dat").exists(),
        "scheduler should not synthesize unsupported xsectjas secondary text sidecar"
    );
    assert!(
        !temp.path().join("xsecl.bin").exists(),
        "scheduler should not synthesize unsupported xsectjas binary sidecar"
    );
    assert_eq!(
        read_phase_bin(temp.path().join("phase.bin"))?,
        expected_phase
    );
    assert_eq!(
        read_xsect_dat(temp.path().join("xsect.dat"))?,
        expected_xsect
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsectjas_energy_grid()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let mut stale_xsecl = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    stale_xsecl.energy[0] += 1.0;
    refeff_io::write_xsecl_dat(temp.path().join("xsecl.dat"), &stale_xsecl)?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsectjas energy grid should not report completed XSPH: {reports:?}"
    );
    let preserved = refeff_io::read_xsecl_dat(temp.path().join("xsecl.dat"))?;
    assert!((preserved.energy[0] - stale_xsecl.energy[0]).abs() <= 5.0e-5);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_malformed_xsecl_text() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let malformed_xsecl = b"not an xsecl.dat cache\n";
    std::fs::write(temp.path().join("xsecl.dat"), malformed_xsecl)?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "malformed NRIXS xsecl.dat cache should not report completed XSPH: {reports:?}"
    );
    assert_eq!(
        std::fs::read(temp.path().join("xsecl.dat"))?,
        malformed_xsecl
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsecl2_energy_grid() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_xsecl2 = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    stale_xsecl2.energy[0] += 1.0;
    refeff_io::write_xsecl2_dat(temp.path().join("xsecl2.dat"), &stale_xsecl2)?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsectjas secondary energy grid should not report completed XSPH: {reports:?}"
    );
    let preserved = refeff_io::read_xsecl2_dat(temp.path().join("xsecl2.dat"))?;
    assert!((preserved.energy[0] - stale_xsecl2.energy[0]).abs() <= 5.0e-5);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_malformed_xsecl2_text() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let malformed_xsecl2 = b"not an xsecl2.dat cache\n";
    std::fs::write(temp.path().join("xsecl2.dat"), malformed_xsecl2)?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "malformed NRIXS xsecl2.dat cache should not report completed XSPH: {reports:?}"
    );
    assert_eq!(
        std::fs::read(temp.path().join("xsecl2.dat"))?,
        malformed_xsecl2
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsecl_text_sum() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let mut stale_xsecl = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    stale_xsecl.channel_sum[0] += Complex64::new(1.0, 0.0);
    refeff_io::write_xsecl_dat(temp.path().join("xsecl.dat"), &stale_xsecl)?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsectjas text sum should not report completed XSPH: {reports:?}"
    );
    let preserved = refeff_io::read_xsecl_dat(temp.path().join("xsecl.dat"))?;
    assert!((preserved.channel_sum[0].re - stale_xsecl.channel_sum[0].re).abs() <= 5.0e-5);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsecl2_text_sum() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_xsecl2 = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    stale_xsecl2.channel_sum[0] += Complex64::new(1.0, 0.0);
    refeff_io::write_xsecl2_dat(temp.path().join("xsecl2.dat"), &stale_xsecl2)?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsectjas secondary text sum should not report completed XSPH: {reports:?}"
    );
    let preserved = refeff_io::read_xsecl2_dat(temp.path().join("xsecl2.dat"))?;
    assert!((preserved.channel_sum[0].re - stale_xsecl2.channel_sum[0].re).abs() <= 5.0e-5);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_mismatched_xsecl_text_channels()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_xsecl2 = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    let expanded_channels = Array2::from_shape_fn((phase.energy_count, 2), |(row, channel)| {
        if channel == 0 {
            stale_xsecl2.channel_cross_sections[(row, 0)]
        } else {
            Complex64::new(0.02 * (row + 1) as f64, 0.003 * (row + 1) as f64)
        }
    });
    stale_xsecl2.channel_sum = Array1::from_shape_fn(phase.energy_count, |row| {
        expanded_channels[(row, 0)] + expanded_channels[(row, 1)]
    });
    stale_xsecl2.channel_cross_sections = expanded_channels;
    refeff_io::write_xsecl2_dat(temp.path().join("xsecl2.dat"), &stale_xsecl2)?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "mismatched NRIXS xsectjas text channel counts should not report completed XSPH: {reports:?}"
    );
    assert_eq!(
        refeff_io::read_xsecl2_dat(temp.path().join("xsecl2.dat"))?.channel_count(),
        2
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_mismatched_xsecl_text_header()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_xsecl2 = sample_nrixs_xsecl_dat_from_phase(&phase)?;
    stale_xsecl2.header.core_hole_width += 1.0;
    refeff_io::write_xsecl2_dat(temp.path().join("xsecl2.dat"), &stale_xsecl2)?;
    refeff_io::write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_nrixs_xsecl_bin_from_phase(&phase),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "mismatched NRIXS xsectjas text headers should not report completed XSPH: {reports:?}"
    );
    assert!(
        (refeff_io::read_xsecl2_dat(temp.path().join("xsecl2.dat"))?
            .header
            .core_hole_width
            - stale_xsecl2.header.core_hole_width)
            .abs()
            <= 5.0e-5
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsecl_bin_contract() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_bin = sample_nrixs_xsecl_bin_from_phase(&phase);
    stale_bin.transitions.pop();
    refeff_io::write_xsecl_bin(temp.path().join("xsecl.bin"), &stale_bin)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsecl.bin header contract should not report completed XSPH: {reports:?}"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_stale_xsecl_bin_final_state_count()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let mut stale_bin = sample_nrixs_xsecl_bin_from_phase(&phase);
    let atom_cross_sections = stale_bin.atom_cross_sections.clone();
    stale_bin.atom_cross_sections = Array2::from_shape_fn(
        (atom_cross_sections.nrows(), atom_cross_sections.ncols() - 1),
        |(energy, final_state)| atom_cross_sections[(energy, final_state)],
    );
    refeff_io::write_xsecl_bin(temp.path().join("xsecl.bin"), &stale_bin)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "stale NRIXS xsecl.bin final-state contract should not report completed XSPH: {reports:?}"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_nrixs_xsph_with_malformed_xsecl_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_nrixs_xsectjas_cached_xsph_input(temp.path())?;
    let phase = sample_phase_bin_data();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    refeff_io::write_xsecl_dat(
        temp.path().join("xsecl.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    refeff_io::write_xsecl2_dat(
        temp.path().join("xsecl2.dat"),
        &sample_nrixs_xsecl_dat_from_phase(&phase)?,
    )?;
    let malformed_bin = b"not an xsecl.bin cache\n";
    std::fs::write(temp.path().join("xsecl.bin"), malformed_bin)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "malformed NRIXS xsecl.bin cache should not report completed XSPH: {reports:?}"
    );
    assert_eq!(std::fs::read(temp.path().join("xsecl.bin"))?, malformed_bin);
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_log_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    std::fs::write(output.join("log2.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    let log = read_module_log_dat(output.join("log2.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: cross-section and phases (XSPH)."))
    );
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_phase_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    std::fs::write(output.join("phase.bin"), "not phase.bin\n")?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_xsph_phase_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let expected_phase = read_phase_bin(output.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(output.join("xsect.dat"))?;
    let mut stale_phase = expected_phase.clone();
    stale_phase.potentials[0].phase_shifts[(0, 0, 0)] += Complex64::new(0.25, -0.125);
    write_phase_bin(output.join("phase.bin"), &stale_phase)?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(read_phase_bin(output.join("phase.bin"))?, expected_phase);
    assert_eq!(read_xsect_dat(output.join("xsect.dat"))?, expected_xsect);
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_xsph_transition_moments_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let expected_phase = read_phase_bin(output.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(output.join("xsect.dat"))?;
    let mut stale_phase = expected_phase.clone();
    stale_phase.transition_moments[(0, 0, 0, 0)] += Complex64::new(0.125, -0.25);
    write_phase_bin(output.join("phase.bin"), &stale_phase)?;

    run_feff_to_dir(&input, &output)?;

    assert_phase_transition_moments_close(
        &read_phase_bin(output.join("phase.bin"))?,
        &expected_phase,
        1.0e-8,
    );
    assert_eq!(read_xsect_dat(output.join("xsect.dat"))?, expected_xsect);
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_bootstraps_hubbard_control_through_ordinary_xsph_and_fms_without_v_source() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_hubbard_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("gg.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(!output.join("v_hubbard.bin").exists());
    assert!(!output.join("aphase_hubbard.bin").exists());
    Ok(())
}

#[test]
fn full_run_generates_hubbard_active_xsph_source_after_source_apot_before_fms_handoff_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_hubbard_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_v_hubbard_bin(output.join("v_hubbard.bin"), &sample_xsph_v_hubbard_bin())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("active Hubbard FMS source handoff should require complete Hubbard sources")?;

    let message = format!("{error:#?}");
    assert!(message.contains("atomic=4 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph=6 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF fms stage"),
        "{message}"
    );
    assert!(
        message.contains(
            "active Hubbard FMS source generation requires aphase_hubbard.bin and transformation_hubbard.bin"
        ),
        "{message}"
    );
    assert!(!message.contains("fms=5 file(s)"), "{message}");
    assert!(!message.contains("genfmt=3 file(s)"), "{message}");
    assert!(!message.contains("ff2x=3 file(s)"), "{message}");

    let phase = read_phase_bin(output.join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert!(output.join("xsect.dat").is_file());
    let aphase = read_aphase_hubbard_bin_inferred(
        output.join("aphase_hubbard.bin"),
        phase.energy_count,
        phase.potential_count(),
    )?;
    assert_eq!(aphase.potential_count(), phase.potential_count());
    assert_eq!(aphase.energy_count(), phase.energy_count);
    assert!(
        aphase
            .values
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn full_run_regenerates_stale_hubbard_active_xsph_aphase_before_fms_handoff_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_hubbard_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_v_hubbard_bin(output.join("v_hubbard.bin"), &sample_xsph_v_hubbard_bin())?;
    run_feff_to_dir(&input, &output)
        .err()
        .context("active Hubbard XSPH source handoff should stop at FMS source requirement")?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let expected_aphase = read_aphase_hubbard_bin_inferred(
        output.join("aphase_hubbard.bin"),
        phase.energy_count,
        phase.potential_count(),
    )?;
    let mut stale_aphase = expected_aphase.clone();
    stale_aphase.values[(0, 0, 0, 0, 0)] += Complex64::new(0.25, -0.125);
    write_aphase_hubbard_bin(output.join("aphase_hubbard.bin"), &stale_aphase)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("active Hubbard FMS source handoff should require complete Hubbard sources")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph=6 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF fms stage"),
        "{message}"
    );
    assert!(
        message.contains(
            "active Hubbard FMS source generation requires aphase_hubbard.bin and transformation_hubbard.bin"
        ),
        "{message}"
    );
    assert_eq!(
        read_aphase_hubbard_bin_inferred(
            output.join("aphase_hubbard.bin"),
            phase.energy_count,
            phase.potential_count(),
        )?,
        expected_aphase
    );
    Ok(())
}

#[test]
fn xsph_module_alias_generates_source_outputs_from_normal_potential_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    write_xsph_source_input(&input)?;
    execute_rdinp(&input, temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_xsph_source_config_dat(),
    )?;

    run_module("xsph", input)?;

    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert!(
        phase.potentials[0]
            .phase_shifts
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );

    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_composes_source_atomic_apot_into_xsph_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        1
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_global_multipole_xsph_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_e2_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed global-multipole XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 5,
        "global-multipole XSPH source report should include phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete global-multipole source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "screen" || report.name == "screen-wscrn"),
        "global-multipole XSPH source scheduling should not depend on SCREEN: {reports:?}"
    );
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(
        (3..phase.transition_count).any(|transition| {
            (0..phase.energy_count)
                .any(|energy| phase.transition_moments[(energy, 0, transition, 0)].norm() > 0.0)
        }),
        "expected global.inp E2 controls to populate higher XSPH transition slots"
    );
    Ok(())
}

#[test]
fn full_run_uses_global_multipole_controls_for_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_e2_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(
        (3..phase.transition_count).any(|transition| {
            (0..phase.energy_count)
                .any(|energy| phase.transition_moments[(energy, 0, transition, 0)].norm() > 0.0)
        }),
        "expected global.inp E2 controls to populate higher XSPH transition slots"
    );
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_ignores_pmbse_for_positive_izstd_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_positive_izstd_pmbse_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_xsph_positive_izstd_source_pot_bin(),
    )?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert_eq!(xsph_input.advanced.izstd, 1);
    assert_eq!(xsph_input.advanced.ipmbse, 3);
    assert_eq!(xsph_input.advanced.itdlda, 2);
    assert_eq!(xsph_input.advanced.nonlocal, 2);
    assert_eq!(xsph_input.advanced.ibasis, 6);

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_positive_izstd_xsph_while_ignoring_pmbse_from_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_positive_izstd_pmbse_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_xsph_positive_izstd_source_pot_bin(),
    )?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed positive-izstd PMBSE-reset XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 5,
        "positive-izstd PMBSE-reset XSPH source report should include phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete positive-izstd PMBSE-reset source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "screen" || report.name == "screen-wscrn"),
        "positive-izstd PMBSE-reset scheduling should ignore PMBSE/SCRN sidecars: {reports:?}"
    );

    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert_eq!(xsph_input.advanced.izstd, 1);
    assert_eq!(xsph_input.advanced.ipmbse, 3);
    assert_eq!(xsph_input.advanced.itdlda, 2);
    assert_eq!(xsph_input.advanced.nonlocal, 2);
    assert_eq!(xsph_input.advanced.ibasis, 6);

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn full_run_scheduler_generates_tdlda_xsedge_from_pmbse_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed TDLDA/PMBSE XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 4,
        "TDLDA/PMBSE XSPH source report should include phase/xsedge sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete TDLDA/PMBSE source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_scheduler_generates_file_basis_tdlda_xsedge_from_pmbse_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_file_basis_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;
    write_full_run_tdlda_file_basis_orbitals(&output)?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed file-basis TDLDA/PMBSE XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 4,
        "file-basis TDLDA/PMBSE XSPH source report should include phase/xsedge sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete file-basis TDLDA/PMBSE source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_file_basis_tdlda_xsedge_without_projector_files() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_file_basis_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        !reports.iter().any(|report| report.name == "xsph"),
        "file-basis TDLDA/PMBSE without Vila/Orbs projectors must not report completed XSPH: {reports:?}"
    );
    assert!(
        reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "incomplete file-basis TDLDA/PMBSE should still expose partial XSPH source progress: {reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(!output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_generated_basis_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed generated-basis TDLDA/PMBSE XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 4,
        "generated-basis TDLDA/PMBSE XSPH source report should include phase/xsedge sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete generated-basis TDLDA/PMBSE source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_file_basis_tdlda_xsedge_from_pmbse_source_handoffs()
-> Result<()> {
    assert_full_run_scheduler_regenerates_stale_unsplit_tdlda_xsedge(
        write_xsph_tdlda_pmbse_file_basis_source_input,
        true,
        "file-basis",
    )
}

#[test]
fn full_run_scheduler_regenerates_stale_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs()
-> Result<()> {
    assert_full_run_scheduler_regenerates_stale_unsplit_tdlda_xsedge(
        write_xsph_tdlda_pmbse_generated_basis_source_input,
        false,
        "generated-basis",
    )
}

#[test]
fn full_run_scheduler_ignores_malformed_ordinary_xsect_for_tdlda_xsedge_source_handoff()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;
    std::fs::write(
        output.join("xsect.dat"),
        "not an ordinary xsect.dat cache\n",
    )?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("stale ordinary xsect.dat should not block TDLDA xsedge generation")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 4,
        "TDLDA/PMBSE XSPH source report should include phase/xsedge sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete TDLDA/PMBSE source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert_eq!(
        std::fs::read_to_string(output.join("xsect.dat"))?,
        "not an ordinary xsect.dat cache\n"
    );

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_regenerates_stale_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_source_input(&input)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;

    let first_error = run_feff_to_dir(&input, &output)
        .err()
        .context("TDLDA xsedge full run should stop at GENFMT path source requirement")?;

    let first_message = format!("{first_error:#?}");
    assert!(first_message.contains("xsph="), "{first_message}");
    assert!(
        first_message.contains("failed to run FEFF genfmt stage"),
        "{first_message}"
    );
    assert!(
        first_message.contains("GENFMT generation requires cached feff.bin/list.dat outputs"),
        "{first_message}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let expected = read_xsedge_dat(output.join("xsedge.dat"))?;
    let mut stale = expected.clone();
    stale.total_single_particle[0] += 25.0;
    stale.total_screened[0] += 12.5;
    if let Some(plus) = stale.plus_branch_single_particle.as_mut() {
        plus[0] += 7.0;
    }
    write_xsedge_dat(output.join("xsedge.dat"), &stale)?;
    assert_ne!(read_xsedge_dat(output.join("xsedge.dat"))?, expected);

    let second_error = run_feff_to_dir(&input, &output)
        .err()
        .context("TDLDA xsedge full run should stop at GENFMT path source requirement")?;

    let second_message = format!("{second_error:#?}");
    assert!(second_message.contains("xsph="), "{second_message}");
    assert!(
        second_message.contains("failed to run FEFF genfmt stage"),
        "{second_message}"
    );
    assert!(
        second_message.contains("GENFMT generation requires cached feff.bin/list.dat outputs"),
        "{second_message}"
    );
    assert_eq!(read_xsedge_dat(output.join("xsedge.dat"))?, expected);
    assert!(!output.join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_file_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_file_basis_source_input(&input)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;
    write_full_run_tdlda_file_basis_orbitals(&output)?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "file-basis TDLDA xsedge full run should stop at GENFMT path source requirement",
    )?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph="), "{message}");
    assert!(!message.contains("xsph-phase="), "{message}");
    assert!(
        message.contains("failed to run FEFF genfmt stage"),
        "{message}"
    );
    assert!(
        message.contains("GENFMT generation requires cached feff.bin/list.dat outputs"),
        "{message}"
    );
    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert_eq!(xsph_input.advanced.ipmbse, 2);
    assert_eq!(xsph_input.advanced.itdlda, 2);
    assert_eq!(xsph_input.advanced.ibasis, 1);
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_generates_generated_basis_tdlda_xsedge_from_pmbse_sources_before_genfmt_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_generated_basis_source_input(&input)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "generated-basis TDLDA xsedge full run should stop at GENFMT path source requirement",
    )?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph="), "{message}");
    assert!(!message.contains("xsph-phase="), "{message}");
    assert!(
        message.contains("failed to run FEFF genfmt stage"),
        "{message}"
    );
    assert!(
        message.contains("GENFMT generation requires cached feff.bin/list.dat outputs"),
        "{message}"
    );
    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert_eq!(xsph_input.advanced.ipmbse, 2);
    assert_eq!(xsph_input.advanced.itdlda, 2);
    assert_eq!(xsph_input.advanced.ibasis, 2);
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let xsedge = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 4);
    assert!(!xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn full_run_rejects_cached_tdlda_xsedge_when_pmbse_source_bundle_is_malformed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_tdlda_pmbse_source_input(&input)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    std::fs::write(output.join("listedges.pmbse"), "Oddp1\n")?;
    std::fs::write(
        output.join("xsedge.dat"),
        "\
  0.00000  1 2 3 4 5 6
  1.00000  1 2 3 4 5 6
  2.00000  1 2 3 4 5 6
  2.50000  1 2 3 4 5 6
  3.00000  1 2 3 4 5 6
  3.50000  1 2 3 4 5 6
",
    )?;
    let cached = read_xsedge_dat(output.join("xsedge.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("malformed PMBSE source should block cached xsedge completion")?;

    let message = error.to_string();
    assert!(!message.contains("xsph="), "{message}");
    assert!(!message.contains("xsph-phase="), "{message}");
    let chain = format!("{error:#}");
    assert!(chain.contains("PMBSE"), "{chain}");
    assert_eq!(read_xsedge_dat(output.join("xsedge.dat"))?, cached);
    assert!(!output.join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_two_spin_filtered_xsph_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_two_spin_filtered_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    let reports = run_supported_cached_modules(&output)?;

    let report = reports
        .iter()
        .find(|report| report.name == "xsph")
        .context("missing completed two-spin filtered XSPH source report")?;
    assert_eq!(report.unit, "file(s)");
    assert!(
        report.count >= 5,
        "two-spin filtered XSPH source report should include phase/xsect sidecars: {reports:?}"
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "xsph-phase" || report.name == "xsph-emesh"),
        "complete two-spin filtered source bundle should report xsph, not a partial handoff: {reports:?}"
    );
    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert!(xsph_input.spinph.iter().any(|spin| *spin != 0.0));
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(phase.transition_moments.dim().3, 2);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    Ok(())
}

#[test]
fn full_run_generates_two_spin_filtered_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_two_spin_filtered_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    run_feff_to_dir(&input, &output)?;

    let xsph_text = std::fs::read_to_string(output.join("xsph.inp"))?;
    let xsph_input = refeff_io::XsphInput::parse_str(output.join("xsph.inp"), &xsph_text)?;
    assert!(xsph_input.spinph.iter().any(|spin| *spin != 0.0));
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(phase.transition_moments.dim().3, 2);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_fprime_xsph_outputs_from_source_handoffs_before_ff2x_extension_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_fprime_phase_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("FPRIME XSPH full run should stop at FF2X extension-grid requirement")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph=5 file(s)"), "{message}");
    assert!(message.contains("fms=3 file(s)"), "{message}");
    assert!(message.contains("mkgtr=3 file(s)"), "{message}");
    assert!(message.contains("genfmt=3 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF ff2x stage"),
        "{message}"
    );
    assert!(
        message.contains("FF2X FPRIME requires positive-axis extension rows"),
        "{message}"
    );
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(output.join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_phase_as_fprime_source_outputs_before_ff2x_extension_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_fprime_phase_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    std::fs::write(output.join("phase.bin"), "not phase.bin\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("FPRIME XSPH full run should stop at FF2X extension-grid requirement")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph=5 file(s)"), "{message}");
    assert!(message.contains("fms=3 file(s)"), "{message}");
    assert!(message.contains("mkgtr=3 file(s)"), "{message}");
    assert!(message.contains("genfmt=3 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF ff2x stage"),
        "{message}"
    );
    assert!(
        message.contains("FF2X FPRIME requires positive-axis extension rows"),
        "{message}"
    );
    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(output.join("log2.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_xes_xsph_outputs_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_xes_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_xsph_screened_source_pot_bin(),
    )?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );

    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(!output.join("axafs.dat").is_file());
    assert!(output.join("mpse.dat").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_xanes_cu_screen_reference_tables_from_source_handoffs() -> Result<()>
{
    let Some(reference_dir) = reference_xanes_cu_screen_source_dir()? else {
        require_fixture!("SCREEN XANES/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "screen.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "fms.inp",
        "geom.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_wscrn = read_wscrn_dat(reference_dir.join("wscrn.dat"))?;
    let expected_vtot = read_vtot_dat(reference_dir.join("vtot.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "screen")
        .context("missing SCREEN source report")?;
    assert_eq!(
        report.count,
        expected_wscrn.row_count() + expected_vtot.row_count()
    );
    assert!(
        !reports.iter().any(|report| report.name == "screen-wscrn"),
        "complete SCREEN source handoff should report screen, not screen-wscrn: {reports:?}"
    );
    assert_wscrn_reference_close(
        &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
        &expected_wscrn,
        1.0e-4,
    );
    assert_vtot_reference_close(
        &read_vtot_dat(temp.path().join("vtot.dat"))?,
        &expected_vtot,
        1.0e-4,
    );
    assert!(temp.path().join("logscreen.dat").is_file());
    assert!(
        !temp.path().join("gg.bin").is_file(),
        "SCREEN inline source-grid FMS path should not require cached gg.bin"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_screen_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let Some(reference_dir) = reference_xanes_cu_screen_source_dir()? else {
        require_fixture!("SCREEN malformed source scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "screen.inp",
        "pot.bin",
        "config.dat",
        "fms.inp",
        "geom.dat",
        "wscrn.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;
    let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "screen" || report.name == "screen-wscrn"),
        "malformed SCREEN phase source should not report SCREEN completion: {:?}",
        reports
    );
    assert_wscrn_reference_close(
        &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
        &expected_wscrn,
        1.0e-12,
    );
    assert!(!temp.path().join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_crpa_reference_from_screen_source_handoffs() -> Result<()> {
    let Some(zip_path) = reference_crpa_zip()? else {
        require_fixture!("CRPA full-run scheduler test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("CRPA full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for entry in ["crpa.inp", "pot.bin", "config.dat", "geom.dat", "fms.inp"] {
        std::fs::write(
            temp.path().join(entry),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
        )?;
    }
    let mut screen_input = unzip_reference_entry(&zip_path, "REFERENCE/screen.inp")?;
    if !String::from_utf8_lossy(&screen_input).contains("icore") {
        screen_input.extend_from_slice(b" icore          -1\n");
    }
    std::fs::write(temp.path().join("screen.inp"), screen_input)?;

    let expected = tempfile::tempdir()?;
    for entry in ["crpa.dat", "wscrn.dat"] {
        std::fs::write(
            expected.path().join(entry),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
        )?;
    }
    let expected_crpa = read_crpa_dat(expected.path().join("crpa.dat"))?;
    let expected_wscrn = read_wscrn_dat(expected.path().join("wscrn.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "crpa")
        .context("missing CRPA source report")?;
    assert_eq!(report.count, 2 + expected_wscrn.row_count());
    assert!(
        !reports.iter().any(|report| report.name == "crpa-wscrn"),
        "complete CRPA source handoff should report crpa, not crpa-wscrn: {reports:?}"
    );
    assert!(
        !temp.path().join("phase.bin").is_file(),
        "CRPA inline source-grid path should not require cached phase.bin"
    );
    assert!(
        !temp.path().join("gg.bin").is_file(),
        "CRPA inline source-grid path should not require cached gg.bin"
    );
    assert_crpa_reference_close(
        &read_crpa_dat(temp.path().join("crpa.dat"))?,
        &expected_crpa,
        1.0e-5,
    );
    assert_wscrn_screened_reference_close(
        &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
        &expected_wscrn,
        1.0e-5,
    );
    assert!(temp.path().join("logscrn.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_screen_before_xes_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_xes_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_xsph_screened_source_pot_bin(),
    )?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;

    run_feff_to_dir(&input, &output)?;

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert!(wscrn.radius_bohr.len() >= vtot.radius_bohr.len());
    for row in 0..vtot.radius_bohr.len() {
        assert_close(wscrn.radius_bohr[row], vtot.radius_bohr[row]);
    }
    assert!(
        wscrn
            .screened_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(wscrn.screened_potential.iter().any(|value| *value != 0.0));
    assert!(read_module_log_dat(output.join("logscreen.dat")).is_ok());
    assert!(read_phase_bin(output.join("phase.bin"))?.energy_count > 0);
    assert!(read_xsect_dat(output.join("xsect.dat"))?.energy_count() > 0);
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_missing_xsph_rl_from_cached_phase_and_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input_with_rlprint(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    let radial = read_xsph_rl_dat(output.join("rl.dat"))?;
    assert!(radial.record_count() > 0);
    assert!(radial.records.iter().all(|record| {
        record.regular_large.len() == radial.radial_count()
            && record.regular_small.len() == radial.radial_count()
    }));
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_rl_from_cached_phase_and_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input_with_rlprint(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;
    std::fs::write(output.join("rl.dat"), "not rl.dat\n")?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    let radial = read_xsph_rl_dat(output.join("rl.dat"))?;
    assert!(radial.record_count() > 0);
    assert!(radial.records.iter().all(|record| {
        record.regular_large.len() == radial.radial_count()
            && record.regular_small.len() == radial.radial_count()
    }));
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_log_for_missing_rl_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_source_input_with_rlprint(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;
    std::fs::write(output.join("log2.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let xsect = read_xsect_dat(output.join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(read_xsph_rl_dat(output.join("rl.dat"))?.record_count() > 0);
    let log = read_module_log_dat(output.join("log2.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: cross-section and phases (XSPH)."))
    );
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_xsph_emesh_from_phase_handoff_before_xsph_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_emesh_phase_handoff_input(&input)?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("phase-only XSPH emesh handoff should stop at full XSPH source requirement")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(
        message.contains("XSPH required stage needs complete phase.bin/xsect.dat caches"),
        "{message}"
    );
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(!output.join("xsect.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_xsph_emesh_from_phase_handoff_before_xsph_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_emesh_phase_handoff_input(&input)?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    std::fs::write(output.join("emesh.dat"), "not emesh.dat\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("phase-only XSPH emesh recovery should stop at full XSPH source requirement")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(
        message.contains("XSPH required stage needs complete phase.bin/xsect.dat caches"),
        "{message}"
    );
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(!output.join("xsect.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_xsph_phase_text_and_emesh_from_phase_handoff_before_xsph_axafs_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_phase_text_cached_input(&input)?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    std::fs::write(output.join("phase00.dat"), "stale phase text\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("phase-text XSPH handoff should stop at AXAFS source requirement")?;

    let message = format!("{error:#?}");
    assert!(message.contains("xsph-phase-text=4 file(s)"), "{message}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(
        message.contains("XSPH AXAFS generation requires xsect.dat cross-section handoff"),
        "{message}"
    );
    for name in ["phase00.dat", "phmin00.dat", "phase01.dat", "phmin01.dat"] {
        assert!(output.join(name).is_file(), "missing {name}");
    }
    let phase00 = std::fs::read_to_string(output.join("phase00.dat"))?;
    assert_ne!(phase00, "stale phase text\n");
    assert!(phase00.contains("unique pot,  lmax, ne"));
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(!output.join("xsect.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_xsph_emesh_from_pot_handoff_before_xsph_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_emesh_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still require the phase-shift solver after emesh handoff")?;

    let message = format!("{error:#?}");
    assert!(
        message.contains("supported cached stages run: xsph-emesh=2 file(s)"),
        "{message}"
    );
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(!output.join("phase.bin").exists());
    assert!(!output.join("xsect.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_xsph_emesh_from_pot_handoff_when_phase_cache_is_malformed_before_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_emesh_source_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    std::fs::write(output.join("phase.bin"), "not phase.bin\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still require the phase-shift solver after emesh handoff")?;

    let message = format!("{error:#?}");
    assert!(
        message.contains("supported cached stages run: xsph-emesh=2 file(s)"),
        "{message}"
    );
    assert!(read_phase_bin(output.join("phase.bin")).is_err());
    assert!(read_emesh_dat(output.join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(output.join("emesh.bin"))?.point_count() > 0);
    assert!(!output.join("xsect.dat").exists());
    Ok(())
}

#[test]
fn full_run_preserves_cached_self_stage_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_self_cached_input(&input)?;
    write_exc_dat(output.join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(output.join("exc.dat"))?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert_eq!(read_exc_dat(output.join("exc.dat"))?, expected);
    Ok(())
}

#[test]
fn full_run_regenerates_stale_self_exc_dat_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_self_source_input(&input)?;
    write_full_run_self_source_handoffs(&output)?;
    write_exc_dat(output.join("exc.dat"), &sample_exc_dat())?;
    let stale = read_exc_dat(output.join("exc.dat"))?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let actual = read_exc_dat(output.join("exc.dat"))?;
    assert_ne!(actual, stale);
    assert_eq!(actual.pole_count(), 4);
    assert!(actual.has_auxiliary_weight());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_mpse_cu_self_reference_exc_from_loss_source_handoff() -> Result<()>
{
    let Some((reference_dir, zip_path)) = reference_self_mpse_cu_case()? else {
        require_fixture!("SELF MPSE/Cu full-run scheduler test; reference not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("SELF MPSE/Cu full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_self_source_input(&input)?;
    execute_rdinp(&input, &output)?;
    for name in ["xsph.inp", "loss.dat"] {
        std::fs::copy(reference_dir.join(name), output.join(name))?;
    }
    std::fs::write(
        temp.path().join("expected-exc.dat"),
        unzip_reference_entry(&zip_path, "REFERENCE/exc.dat")?,
    )?;
    let expected = read_exc_dat(temp.path().join("expected-exc.dat"))?;

    assert!(!output.join("exc.dat").exists());
    let reports = run_supported_cached_modules(&output)?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "self" && report.count == expected.pole_count()),
        "complete MPSE/Cu SELF source handoff should report generated excitation poles: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let actual = read_exc_dat(output.join("exc.dat"))?;
    assert_self_exc_dat_reference_close(&actual, &expected);
    Ok(())
}

#[test]
fn full_run_normalizes_fms_cached_stage_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_cached_input(&input)?;
    let expected_gg = sample_full_run_fms_gg_data();
    write_gg_bin(output.join("gg.bin"), &expected_gg)?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let actual_gg = read_gg_bin(output.join("gg.bin"))?;
    assert!(
        actual_gg.section_count() > expected_gg.section_count(),
        "completed run should expand the small synthetic FMS cache"
    );
    assert_gg_data_values_eq(&read_gg_dat(output.join("gg.dat"))?, &actual_gg);
    let log = read_module_log_dat(output.join("log3.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line == "FMS calculation of full Green's function ...")
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line == "Done with module: FMS.")
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_fms_gtr_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_full_run_orphan_gtr_bin(),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fms"),
        "orphan gtrNN.bin cache without fms.inp should not report FMS complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(temp.path().join("gtr00.bin").is_file());
    assert!(!temp.path().join("log3.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_fms_dmdw_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_fms_dmdw_source_handoffs(temp.path())?;
    std::fs::write(temp.path().join("feff.dym"), b"not a dym source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fms"),
        "malformed FMS DMDW .dym source should not report FMS complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("log3.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_fms_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("fms.inp"),
        fms_input_string(&sample_full_run_fms_source_input())?,
    )?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&sample_band_global_input(1))?,
    )?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&sample_full_run_fms_dmdw_geom())?,
    )?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;
    let gg = sample_full_run_fms_gg_data();
    write_gg_bin(temp.path().join("gg.bin"), &gg)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fms"),
        "malformed FMS phase source should not report cached FMS completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_gg_data_values_eq(&read_gg_bin(temp.path().join("gg.bin"))?, &gg);
    assert!(!temp.path().join("log3.dat").exists());
    Ok(())
}

#[test]
fn full_run_normalizes_fms_gg_dat_after_recovery_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_cached_input(&input)?;
    let gg = sample_full_run_fms_gg_data();
    write_gg_bin(output.join("gg.bin"), &gg)?;
    std::fs::write(output.join("gg.dat"), b"not a gg.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let actual_gg = read_gg_bin(output.join("gg.bin"))?;
    assert!(
        actual_gg.section_count() > gg.section_count(),
        "completed run should expand the small synthetic FMS cache"
    );
    assert_gg_data_values_eq(&read_gg_dat(output.join("gg.dat"))?, &actual_gg);
    Ok(())
}

#[test]
fn full_run_generates_fms_outputs_from_phase_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_source_input(&input)?;
    write_phase_bin(
        output.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    let phase = read_phase_bin(output.join("phase.bin"))?;
    let gg = read_gg_dat(output.join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));
    assert_eq!(read_gg_bin(output.join("gg.bin"))?, gg);

    let fms = read_fms_bin(output.join("fms.bin"))?;
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_eq!(fms.highest_potential_index, 1);
    assert!(fms.spectra.iter().any(|value| value.norm() > 1.0e-8));
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());

    let gtr = read_gtr_dat(output.join("gtr.dat"))?;
    assert_eq!(gtr.energy.len(), phase.energy_count);
    assert!(gtr.trace.iter().any(|value| value.norm() > 1.0e-8));

    let log = read_module_log_dat(output.join("log3.dat"))?;
    for expected in [
        "FMS calculation of full Green's function ...",
        "Done with module: FMS.",
        "MKGTR: Tracing over Green's function ...",
        "Done with module: MKGTR.",
    ] {
        assert!(
            log.lines.iter().any(|line| line == expected),
            "expected log line {expected:?}, got {:?}",
            log.lines
        );
    }
    Ok(())
}

#[test]
fn full_run_recovers_malformed_fms_gg_caches_from_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_source_input(&input)?;
    write_phase_bin(
        output.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;
    std::fs::write(output.join("gg.bin"), b"not a gg.bin cache\n")?;
    std::fs::write(output.join("gg.dat"), b"not a gg.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;

    let gg = read_gg_dat(output.join("gg.dat"))?;
    assert!(gg.section_count() > 0);
    assert_eq!(read_gg_bin(output.join("gg.bin"))?, gg);
    let fms = read_fms_bin(output.join("fms.bin"))?;
    assert_eq!(gg.section_count(), fms.energy_count);
    let gtr = read_gtr_dat(output.join("gtr.dat"))?;
    assert_eq!(gtr.energy.len(), fms.energy_count);
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_fms_outputs_with_classical_debye() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_classical_debye_source_input(&input)?;
    write_phase_bin(
        output.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    let gg = read_gg_dat(output.join("gg.dat"))?;
    assert!(gg.section_count() > 0);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));
    let fms = read_fms_bin(output.join("fms.bin"))?;
    assert_eq!(gg.section_count(), fms.energy_count);
    assert!(
        read_gtr_dat(output.join("gtr.dat"))?
            .trace
            .iter()
            .any(|value| value.norm() > 1.0e-8)
    );
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());

    let log = read_module_log_dat(output.join("log3.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line == "Applying Debye-Waller factors using the Classical Debye model.")
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line == "Done with module: MKGTR.")
    );
    Ok(())
}

fn assert_band_atomic_overlap_boundary(message: &str) {
    assert!(message.contains("atomic-config=1 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF atomic stage"),
        "{message}"
    );
    assert!(
        message.contains("failed to generate ATOM apot.bin from pot.inp/geom.dat source handoffs"),
        "{message}"
    );
    assert!(
        message.contains("failed to overlap ATOM potential 1"),
        "{message}"
    );
    assert!(message.contains("InvalidRadius"), "{message}");
    assert!(message.contains("radius: 0.0"), "{message}");
}

#[test]
fn full_run_executes_cached_band_stage_before_atomic_overlap_radius_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input_with_freeprop(&input, false)?;
    write_bandstructure_dat(
        output.join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    let expected = read_bandstructure_dat(output.join("bandstructure.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band=3 file(s)"), "{message}");
    assert_eq!(
        read_bandstructure_dat(output.join("bandstructure.dat"))?,
        expected
    );
    let log = read_module_log_dat(output.join("logband.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating band structure ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Solving band structure."))
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_band_log_from_reciprocal_handoff_before_atomic_overlap_radius_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input_with_freeprop(&input, false)?;
    write_bandstructure_dat(
        output.join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    std::fs::write(output.join("logband.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band=3 file(s)"), "{message}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    let log = read_module_log_dat(output.join("logband.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating band structure ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Solving band structure."))
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_band_log_from_pre_solver_handoff_before_atomic_overlap_radius_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input(&input)?;
    write_bandstructure_dat(
        output.join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    write_kmesh_dat(output.join("kmesh.dat"), &sample_single_kmesh_dat())?;
    std::fs::write(output.join("logband.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band=3 file(s)"), "{message}");
    assert!(!message.contains("band-handoff="), "{message}");
    let log = read_module_log_dat(output.join("logband.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating band structure ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Solving band structure."))
    );
    Ok(())
}

#[test]
fn full_run_skips_malformed_band_cache_after_source_handoffs_before_required_module_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_bandstructure_input(&input)?;
    std::fs::write(output.join("bandstructure.dat"), "not bandstructure.dat\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("BAND should skip malformed bandstructure cache after source handoffs")?;

    let message = format!("{error:#?}");
    assert!(message.contains("atomic=4 file(s)"), "{message}");
    assert!(message.contains("pot=5 file(s)"), "{message}");
    assert!(message.contains("xsph=6 file(s)"), "{message}");
    assert!(message.contains("fms=3 file(s)"), "{message}");
    assert!(message.contains("mkgtr=3 file(s)"), "{message}");
    assert!(message.contains("path=1 path(s)"), "{message}");
    assert!(!message.contains("band="), "{message}");
    Ok(())
}

#[test]
fn full_run_skips_malformed_band_log_after_source_handoffs_before_required_module_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_bandstructure_input(&input)?;
    write_bandstructure_dat(
        output.join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    std::fs::write(output.join("logband.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("BAND should skip malformed log cache after source handoffs")?;

    let message = format!("{error:#?}");
    assert!(message.contains("atomic=4 file(s)"), "{message}");
    assert!(message.contains("pot=5 file(s)"), "{message}");
    assert!(message.contains("xsph=6 file(s)"), "{message}");
    assert!(message.contains("fms=3 file(s)"), "{message}");
    assert!(message.contains("mkgtr=3 file(s)"), "{message}");
    assert!(message.contains("path=1 path(s)"), "{message}");
    assert!(!message.contains("band="), "{message}");
    Ok(())
}

#[test]
fn full_run_validates_band_handoff_when_malformed_bandstructure_exists_before_solver_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input_with_freeprop(&input, false)?;
    std::fs::write(output.join("bandstructure.dat"), "not bandstructure.dat\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("BAND should validate reciprocal handoff before solver output")?;

    let message = format!("{error:#?}");
    assert!(message.contains("band-handoff=2 file(s)"), "{message}");
    assert!(!message.contains("band="), "{message}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert!(!output.join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_band_log_for_pre_solver_handoff_before_atomic_overlap_radius_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input_with_freeprop(&input, false)?;
    std::fs::write(output.join("bandstructure.dat"), "not bandstructure.dat\n")?;
    std::fs::write(output.join("logband.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band-handoff=3 file(s)"), "{message}");
    assert!(!message.contains("band="), "{message}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    let log = read_module_log_dat(output.join("logband.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating band structure ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Solving band structure."))
    );
    Ok(())
}

#[test]
fn full_run_generates_kmesh_from_reciprocal_handoff_before_atomic_overlap_radius_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_reciprocal_bandstructure_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band-handoff=2 file(s)"), "{message}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert_eq!(
        kmesh.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert!(!output.join("bandstructure.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_kmesh_from_reciprocal_handoff_before_atomic_overlap_radius_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input(&input)?;
    std::fs::write(output.join("kmesh.dat"), "not kmesh.dat\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band-handoff=2 file(s)"), "{message}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert_eq!(
        kmesh.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert!(!output.join("bandstructure.dat").exists());
    Ok(())
}

#[test]
fn full_run_validates_band_reciprocal_handoff_before_atomic_overlap_radius_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_bandstructure_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should reject the reciprocal BAND source geometry")?;

    let message = format!("{error:#?}");
    assert_band_atomic_overlap_boundary(&message);
    assert!(message.contains("band-handoff=2 file(s)"), "{message}");
    assert!(output.join("kmesh.dat").is_file());
    assert!(!output.join("bandstructure.dat").exists());
    assert!(!output.join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_band_phase_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&sample_single_potential_reciprocal_input(8))?,
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "band" || report.name == "band-handoff"),
        "malformed BAND phase source should not report BAND completion or pre-solver handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_band_reciprocal_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_band_handoff_phase_bin(),
    )?;
    std::fs::write(temp.path().join("reciprocal.inp"), "not reciprocal input\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "band" || report.name == "band-handoff"),
        "malformed BAND reciprocal source should not report BAND completion or pre-solver handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_band_when_reciprocal_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_single_kmesh_dat())?;
    std::fs::write(temp.path().join("reciprocal.inp"), "not reciprocal input\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "band"),
        "malformed BAND reciprocal source should block cached BAND completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_band_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_single_kmesh_dat())?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "band"),
        "malformed BAND phase source should block cached BAND completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_band_when_global_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_single_kmesh_dat())?;
    std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "band"),
        "malformed BAND global source should block cached BAND completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_band_when_fms_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    write_bandstructure_dat(
        temp.path().join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_single_kmesh_dat())?;
    std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports.iter().any(|report| report.name == "band"),
        "malformed BAND FMS source should block cached BAND completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_band_global_spin_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_rel_band_scheduler_handoffs(temp.path())?;
    std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "band" || report.name == "band-handoff"),
        "malformed BAND global source should not report BAND completion or pre-solver handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_band_fms_lmaxph_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_rel_band_scheduler_handoffs(temp.path())?;
    std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "band" || report.name == "band-handoff"),
        "malformed BAND FMS source should not report BAND completion or pre-solver handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_one_spin_rel_bandstructure_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_rel_band_scheduler_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 5),
        "missing completed BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed rel source bundle should not report as validation-only: {reports:?}"
    );
    let global_text = std::fs::read_to_string(temp.path().join("global.inp"))?;
    let global = refeff_io::GlobalInput::parse_str(temp.path().join("global.inp"), &global_text)?;
    assert_eq!(global.control.ispin, 1);
    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_one_spin_rel_bandstructure_from_source_handoffs()
-> Result<()> {
    assert_full_run_scheduler_regenerates_stale_one_spin_rel_bandstructure(false, 5)
}

#[test]
fn full_run_scheduler_regenerates_stale_one_spin_rel_freeprop_bandstructure_from_source_handoffs()
-> Result<()> {
    assert_full_run_scheduler_regenerates_stale_one_spin_rel_bandstructure(true, 4)
}

#[test]
fn full_run_scheduler_generates_graphite_kmesh_from_reference_reciprocal_handoff() -> Result<()> {
    let Some((reference_dir, zip_path)) = reference_graphite_band_handoff()? else {
        require_fixture!("BAND Graphite full-run scheduler test; reference handoff not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("BAND Graphite full-run scheduler test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    write_enabled_band_handoff_input(temp.path())?;
    std::fs::copy(
        reference_dir.join("reciprocal.inp"),
        temp.path().join("reciprocal.inp"),
    )?;
    let expected_kmesh_path = temp.path().join("expected-kmesh.dat");
    std::fs::write(
        &expected_kmesh_path,
        unzip_reference_entry(&zip_path, "REFERENCE/kmesh.dat")?,
    )?;
    let expected_kmesh = read_kmesh_dat(&expected_kmesh_path)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "kmesh" && report.count == 1),
        "missing KSPACE kmesh source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports
            .iter()
            .any(|report| report.name == "band" || report.name == "band-handoff"),
        "Graphite kmesh handoff should not report completed or validation-only BAND: {reports:?}"
    );
    assert_kmesh_dat_close(
        &read_kmesh_dat(temp.path().join("kmesh.dat"))?,
        &expected_kmesh,
        6.0e-4,
    );
    assert!(!temp.path().join("bandstructure.dat").exists());
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_cr2gec_reference_bandstructure_from_source_handoffs() -> Result<()>
{
    let Some(reference_dir) = reference_cr2gec_generated_band_output()? else {
        require_fixture!("BAND Cr2GeC full-run scheduler test; FEFF band run not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "band.inp",
        "reciprocal.inp",
        "fms.inp",
        "global.inp",
        "phase.bin",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy Cr2GeC BAND handoff {name}"))?;
    }
    let expected = read_bandstructure_dat(reference_dir.join("bandstructure.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().any(|report| report.name == "band"),
        "missing completed Cr2GeC BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed Cr2GeC source bundle should not report as validation-only: {reports:?}"
    );
    assert_bandstructure_dat_close(
        &read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        &expected,
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_bandstructure_values_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_freeprop_band_scheduler_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing completed BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed source bundle should not report as validation-only: {reports:?}"
    );
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
    let mut stale = expected.clone();
    let (_, row) = stale
        .rows
        .iter_mut()
        .enumerate()
        .find(|(_, row)| !row.bands.is_empty())
        .context("source bandstructure should contain at least one band value")?;
    row.bands[0] += 0.25;
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing regenerated BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "regenerated source bundle should not report as validation-only: {reports:?}"
    );
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_freeprop_bandstructure_band_counts_from_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_freeprop_band_scheduler_handoffs(temp.path())?;

    run_supported_cached_modules(temp.path())?;
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
    let mut stale = expected.clone();
    for row in &mut stale.rows {
        row.bands = Array1::zeros(0);
    }
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing regenerated freeprop BAND source report after stale band counts: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "regenerated source bundle should not report as validation-only: {reports:?}"
    );
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_two_spin_degenerate_bandstructure_from_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_two_spin_degenerate_band_scheduler_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 5),
        "missing completed two-spin BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed two-spin source bundle should not report as validation-only: {reports:?}"
    );
    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_two_spin_non_degenerate_bandstructure_from_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_two_spin_non_degenerate_band_scheduler_handoffs(temp.path(), false)?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_ne!(
        phase.reference_energy[(0, 0)],
        phase.reference_energy[(0, 1)]
    );

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 5),
        "missing completed non-degenerate two-spin BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed non-degenerate two-spin source bundle should not report as validation-only: {reports:?}"
    );
    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_two_spin_non_degenerate_freeprop_bandstructure_from_source_handoffs()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_two_spin_non_degenerate_band_scheduler_handoffs(temp.path(), true)?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_ne!(
        phase.reference_energy[(0, 0)],
        phase.reference_energy[(0, 1)]
    );

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing completed non-degenerate two-spin freeprop BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed non-degenerate two-spin freeprop source bundle should not report as validation-only: {reports:?}"
    );
    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_two_spin_bandstructure_from_source_handoffs() -> Result<()>
{
    assert_full_run_scheduler_regenerates_stale_two_spin_bandstructure(false, 5)
}

#[test]
fn full_run_scheduler_regenerates_stale_two_spin_freeprop_bandstructure_from_source_handoffs()
-> Result<()> {
    assert_full_run_scheduler_regenerates_stale_two_spin_bandstructure(true, 4)
}

fn assert_full_run_scheduler_regenerates_stale_one_spin_rel_bandstructure(
    freeprop: bool,
    expected_report_count: usize,
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_band_scheduler_handoffs(
        temp.path(),
        freeprop,
        &sample_band_handoff_phase_bin(),
    )?;
    let global_text = std::fs::read_to_string(temp.path().join("global.inp"))?;
    let global = refeff_io::GlobalInput::parse_str(temp.path().join("global.inp"), &global_text)?;
    assert_eq!(global.control.ispin, 1);

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == expected_report_count),
        "missing completed one-spin rel BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed one-spin rel source bundle should not report as validation-only: {reports:?}"
    );
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
    let mut stale = expected.clone();
    stale
        .rows
        .first_mut()
        .context("one-spin rel source bandstructure should contain at least one k-point")?
        .k_point[0] += 0.5;
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == expected_report_count),
        "missing regenerated one-spin rel BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "regenerated one-spin rel source bundle should not report as validation-only: {reports:?}"
    );
    assert_bandstructure_dat_close(
        &read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        &expected,
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_freeprop_bandstructure_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_freeprop_band_scheduler_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing completed freeprop BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed freeprop source bundle should not report as validation-only: {reports:?}"
    );
    assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat"))?.k_point_count() > 0);
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

fn assert_full_run_scheduler_regenerates_stale_two_spin_bandstructure(
    freeprop: bool,
    expected_report_count: usize,
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_single_potential_two_spin_non_degenerate_band_scheduler_handoffs(temp.path(), freeprop)?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_ne!(
        phase.reference_energy[(0, 0)],
        phase.reference_energy[(0, 1)]
    );

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == expected_report_count),
        "missing completed two-spin BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed two-spin source bundle should not report as validation-only: {reports:?}"
    );
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
    let mut stale = expected.clone();
    stale
        .rows
        .iter_mut()
        .find(|row| !row.bands.is_empty())
        .context("two-spin source bandstructure should contain at least one band")?
        .bands[0] += 0.5;
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == expected_report_count),
        "missing regenerated two-spin BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "regenerated two-spin source bundle should not report as validation-only: {reports:?}"
    );
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_freeprop_bandstructure_from_source_handoffs() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_single_potential_freeprop_band_scheduler_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing completed freeprop BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "completed freeprop source bundle should not report as validation-only: {reports:?}"
    );
    let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
    let mut stale = expected.clone();
    stale
        .rows
        .iter_mut()
        .find(|row| !row.bands.is_empty())
        .context("freeprop source bandstructure should contain at least one band")?
        .bands[0] += 99.0;
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;
    assert!(
        reports
            .iter()
            .any(|report| report.name == "band" && report.count == 4),
        "missing regenerated freeprop BAND source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        !reports.iter().any(|report| report.name == "band-handoff"),
        "regenerated freeprop source bundle should not report as validation-only: {reports:?}"
    );
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        expected
    );
    assert!(temp.path().join("kmesh.dat").is_file());
    assert!(temp.path().join("logband.dat").is_file());
    Ok(())
}

#[test]
fn full_run_regenerates_malformed_no_fms_ldos_from_radial_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    let rhoc = sample_rhoc_dat()?;
    write_rhoc_dat(output.join("rhoc00.dat"), &rhoc)?;
    std::fs::write(output.join("ldos00.dat"), "not ldos.dat\n")?;

    run_feff_to_dir(&input, &output)?;

    let ldos = read_ldos_dat(output.join("ldos00.dat"))?;
    let regenerated_rhoc = read_rhoc_dat(output.join("rhoc00.dat"))?;
    assert_eq!(ldos.energy_ev, regenerated_rhoc.energy_ev);
    assert_eq!(ldos.density, regenerated_rhoc.density);
    assert_ne!(regenerated_rhoc, rhoc);
    assert!(ldos.density.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
    assert_eq!(
        read_module_log_dat(output.join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_xanes_cu_no_fms_ldos_reference_tables_from_source_handoffs()
-> Result<()> {
    let Some((source_dir, expected_dir)) = reference_ldos_xanes_cu_no_fms_source_case()? else {
        require_fixture!("LDOS XANES/Cu no-FMS full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(expected_dir.join("ldos.inp"), temp.path().join("ldos.inp"))?;
    for name in [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
    ] {
        std::fs::copy(source_dir.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy {name} for LDOS scheduler reference"))?;
    }

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "ldos" && report.count == 4),
        "missing completed LDOS source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    for potential in 0..=1 {
        let generated_ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let generated_rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        let reference_ldos = read_ldos_dat(expected_dir.join(format!("ldos{potential:02}.dat")))?;
        let reference_rhoc = read_rhoc_dat(expected_dir.join(format!("rhoc{potential:02}.dat")))?;
        assert_ldos_reference_table_close(
            &generated_ldos,
            &reference_ldos,
            &format!("XANES/Cu no-FMS ldos{potential:02}.dat"),
        );
        assert_ldos_reference_table_close(
            &generated_rhoc,
            &reference_rhoc,
            &format!("XANES/Cu no-FMS rhoc{potential:02}.dat"),
        );
        assert_eq!(generated_ldos.energy_ev, generated_rhoc.energy_ev);
        assert_eq!(generated_ldos.density, generated_rhoc.density);
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_xanes_cu_no_fms_ldos_tables_from_source_handoffs()
-> Result<()> {
    let Some((source_dir, expected_dir)) = reference_ldos_xanes_cu_no_fms_source_case()? else {
        require_fixture!("stale LDOS XANES/Cu no-FMS full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(expected_dir.join("ldos.inp"), temp.path().join("ldos.inp"))?;
    for name in [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
    ] {
        std::fs::copy(source_dir.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy {name} for stale LDOS scheduler test"))?;
    }

    let first_reports = run_supported_cached_modules(temp.path())?;
    assert!(
        first_reports
            .iter()
            .any(|report| report.name == "ldos" && report.count == 4),
        "missing initial LDOS source report: {:?}",
        first_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert_ldos_reference_table_close(
        &expected_ldos,
        &read_ldos_dat(expected_dir.join("ldos00.dat"))?,
        "initial XANES/Cu no-FMS ldos00.dat",
    );
    assert_ldos_reference_table_close(
        &expected_rhoc,
        &read_rhoc_dat(expected_dir.join("rhoc00.dat"))?,
        "initial XANES/Cu no-FMS rhoc00.dat",
    );

    let mut stale_ldos = expected_ldos.clone();
    let mut stale_rhoc = expected_rhoc.clone();
    stale_ldos.density[(0, 0)] += 0.25;
    stale_rhoc.density[(0, 0)] += 0.125;
    write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &stale_rhoc)?;

    let second_reports = run_supported_cached_modules(temp.path())?;
    assert!(
        second_reports
            .iter()
            .any(|report| report.name == "ldos" && report.count == 4),
        "missing regenerated LDOS source report: {:?}",
        second_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        expected_ldos
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        expected_rhoc
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ldos_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("ldos.inp"), b"not an ldos.inp handoff\n")?;
    let ldos = sample_ldos_dat()?;
    let rhoc = sample_rhoc_dat()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "ldos" && report.name != "ldos-kmesh"),
        "malformed ldos.inp should not report LDOS complete: {:?}",
        reports
    );
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_exafs_cu_ff2x_reference_spectra_from_source_handoffs() -> Result<()>
{
    let Some(reference_dir) = reference_exafs_cu_ff2x_source_dir()? else {
        require_fixture!("FF2X EXAFS/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["ff2x.inp", "feff.bin", "list.dat", "xsect.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "ff2x")
        .context("missing FF2X source report")?;
    assert_eq!(report.count, 3);
    assert_ff2x_chi_reference_close(
        &read_chi_dat(temp.path().join("chi.dat"))?,
        &read_chi_dat(reference_dir.join("chi.dat"))?,
    );
    assert_ff2x_xmu_reference_close(
        &read_xmu_dat(temp.path().join("xmu.dat"))?,
        &read_xmu_dat(reference_dir.join("xmu.dat"))?,
    );
    assert!(temp.path().join("log6.dat").is_file());
    Ok(())
}

fn pot_stage_output_count(work_dir: &std::path::Path) -> usize {
    let fixed_outputs = ["pot.bin", "apot.bin", "chemical.dat", "log1.dat"]
        .into_iter()
        .filter(|name| work_dir.join(name).is_file())
        .count();
    let rendered_outputs = std::fs::read_dir(work_dir)
        .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", work_dir.display()))
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(index) = name
                .strip_prefix("pot")
                .and_then(|name| name.strip_suffix(".dat"))
            else {
                return false;
            };
            index.len() == 2 && index.bytes().all(|byte| byte.is_ascii_digit())
        })
        .count();
    fixed_outputs + rendered_outputs
}

fn assert_rixs_downstream_solver_error(message: &str, work_dir: &std::path::Path) {
    assert!(
        message.contains("atomic=4 file(s)") || message.contains("atomic=3 file(s)"),
        "{message}"
    );
    let pot_count = pot_stage_output_count(work_dir);
    assert!(
        pot_count > 0,
        "no POT stage outputs in {}",
        work_dir.display()
    );
    assert!(
        message.contains(&format!("pot={pot_count} file(s)")),
        "{message}"
    );
    assert!(
        message.contains("failed to run FEFF rixs stage"),
        "{message}"
    );
}

fn assert_incomplete_explicit_rixs_handoff_error(message: &str) {
    assert!(
        message.contains("failed to prepare the RIXS two-edge solver workflow"),
        "{message}"
    );
    assert!(
        message.contains(
            "incomplete RIXS two-edge handoff: found 7 of 9 required files; refusing to mix edge calculations"
        ),
        "{message}"
    );
    assert!(
        !message.contains("failed to run RIXS edge calculation"),
        "{message}"
    );
    assert!(
        !message.contains("failed to run FEFF rixs stage"),
        "{message}"
    );
}

fn assert_rixs_l3_preparation_xsph_error(message: &str) {
    assert!(
        message.contains("failed to prepare the RIXS two-edge solver workflow"),
        "{message}"
    );
    assert!(
        message.contains("failed to run RIXS edge calculation L3"),
        "{message}"
    );
    assert!(message.contains("/L3"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(
        message.contains(
            "XSPH phase generation requires cached phase.bin or supported pot/config source handoffs"
        ),
        "{message}"
    );
    assert!(
        !message.contains("failed to run RIXS edge calculation VAL"),
        "{message}"
    );
    assert!(
        !message.contains("failed to run FEFF rixs stage"),
        "{message}"
    );
}

#[test]
fn full_run_preserves_cached_rixs_stage_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
    let expected_map = read_rixs_map(output.join("rixsET.dat"))?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert_eq!(read_rixs_map(output.join("rixsET.dat"))?, expected_map);
    let herfd = read_rixs_line(output.join("herfd.dat"))?;
    assert_eq!(herfd.energy_ev.to_vec(), vec![11_540.0, 11_541.0]);
    assert_eq!(herfd.channels.shape(), &[2, 2]);
    assert_close(herfd.channels[(0, 0)], 1.0e-6);
    assert_close(herfd.channels[(0, 1)], 1.2e-6);
    assert_close(herfd.channels[(1, 0)], 4.0e-6);
    assert_close(herfd.channels[(1, 1)], 4.2e-6);
    assert!(!output.join("xasEI.dat").exists());
    assert!(!output.join("xasEF.dat").exists());
    assert!(!output.join("rixsEE.dat").exists());
    let log = read_module_log_dat(output.join("logrixs.dat"))?;
    assert!(log.lines.iter().any(|line| line == "Reading data."));
    assert!(log.lines.iter().any(|line| line == "Writing results."));
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_rixs_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("rixs.inp"), b"not a rixs.inp handoff\n")?;
    let map = sample_rixs_map_data();
    write_rixs_map(temp.path().join("rixsET.dat"), &map)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "rixs" && report.name != "rixs-handoff"),
        "malformed rixs.inp should not report RIXS complete: {:?}",
        reports
    );
    assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
    assert!(!temp.path().join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_rixs_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let map = sample_rixs_map_data();
    write_rixs_map(temp.path().join("rixsET.dat"), &map)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "rixs" && report.name != "rixs-handoff"),
        "orphan rixsET.dat cache without rixs.inp should not report RIXS complete: {:?}",
        reports
    );
    assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, map);
    assert!(!temp.path().join("herfd.dat").exists());
    assert!(!temp.path().join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_rixs_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    let map = sample_rixs_map_data();
    write_rixs_map(output.join("rixsET.dat"), &map)?;
    std::fs::write(output.join("phase.bin"), b"not a phase.bin source\n")?;

    let reports = run_supported_cached_modules(&output)?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "rixs" && report.name != "rixs-handoff"),
        "malformed RIXS phase source should not report RIXS complete: {:?}",
        reports
    );
    assert_eq!(read_rixs_map(output.join("rixsET.dat"))?, map);
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_rixs_herfd_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
    std::fs::write(output.join("herfd.dat"), "not a RIXS line\n")?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let herfd = read_rixs_line(output.join("herfd.dat"))?;
    assert_eq!(herfd.energy_ev.to_vec(), vec![11_540.0, 11_541.0]);
    assert_eq!(herfd.channels.shape(), &[2, 2]);
    assert_close(herfd.channels[(0, 0)], 1.0e-6);
    assert_close(herfd.channels[(0, 1)], 1.2e-6);
    assert_close(herfd.channels[(1, 0)], 4.0e-6);
    assert_close(herfd.channels[(1, 1)], 4.2e-6);
    Ok(())
}

#[test]
fn full_run_recovers_malformed_rixs_log_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
    std::fs::write(output.join("logrixs.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let herfd = read_rixs_line(output.join("herfd.dat"))?;
    assert_eq!(herfd.energy_ev.to_vec(), vec![11_540.0, 11_541.0]);
    assert_eq!(herfd.channels.shape(), &[2, 2]);
    let log = read_module_log_dat(output.join("logrixs.dat"))?;
    assert!(log.lines.iter().any(|line| line == "Reading data."));
    assert!(log.lines.iter().any(|line| line == "Writing results."));
    Ok(())
}

#[test]
fn full_run_rejects_malformed_rixs_cache_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    std::fs::write(output.join("rixsET.dat"), "not a RIXS map\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("malformed RIXS cache should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_rixs_downstream_solver_error(&message, &output);
    assert!(
        message.contains(
            "RIXS generation requires cached spectra or complete phase/rl/wscrn/xsect source handoffs"
        ),
        "{message}"
    );
    assert!(
        !message.contains("supported cached stages run: rixs="),
        "{message}"
    );
    assert!(!output.join("herfd.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_rejects_malformed_rixs_log_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
    refeff_io::write_rixs_line(
        output.join("herfd.dat"),
        &refeff_io::RixsLineData {
            header_lines: Vec::new(),
            energy_ev: ndarray::Array1::from_vec(vec![11_540.0, 11_541.0]),
            channels: ndarray::Array2::from_shape_vec(
                (2, 2),
                vec![1.0e-6, 1.2e-6, 4.0e-6, 4.2e-6],
            )?,
        },
    )?;
    std::fs::write(output.join("logrixs.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("malformed RIXS log should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_rixs_downstream_solver_error(&message, &output);
    assert!(
        message.contains(
            "RIXS generation requires cached spectra or complete phase/rl/wscrn/xsect source handoffs"
        ),
        "{message}"
    );
    assert!(
        !message.contains("supported cached stages run: rixs="),
        "{message}"
    );
    Ok(())
}

#[test]
fn full_run_derives_rixs_final_outputs_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_edges_dat(output.join("edges.dat"), &sample_rixs_edges_dat())?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_square_map_data())?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());

    let herfd = read_rixs_line(output.join("herfd.dat"))?;
    assert_eq!(herfd.energy_ev.len(), 3);
    assert_close(herfd.channels[(0, 0)], 1.0);
    assert_close(herfd.channels[(1, 0)], 5.0);
    assert_close(herfd.channels[(2, 0)], 9.0);
    assert!(
        !read_rixs_line(output.join("xasEI.dat"))?
            .energy_ev
            .is_empty()
    );
    assert!(
        !read_rixs_line(output.join("xasEF.dat"))?
            .energy_ev
            .is_empty()
    );
    let rixs_ee = read_rixs_map(output.join("rixsEE.dat"))?;
    assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
    assert_eq!(rixs_ee.channels.ncols(), 2);
    Ok(())
}

#[test]
fn full_run_skip_calc_rewrites_rixs_outputs_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_skip_calc_cached_input(&input)?;
    write_edges_dat(output.join("edges.dat"), &sample_rixs_edges_dat())?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_square_map_data())?;
    refeff_io::write_rixs_line(
        output.join("xasEI.dat"),
        &refeff_io::RixsLineData {
            header_lines: vec!["# stale xasEI".to_string()],
            energy_ev: Array1::from_vec(vec![-1.0]),
            channels: Array2::from_shape_vec((1, 2), vec![999.0, 1000.0])?,
        },
    )?;
    write_rixs_map(output.join("rixsEE.dat"), &sample_rixs_map_data())?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let rixs_input = std::fs::read_to_string(output.join("rixs.inp"))?;
    assert!(rixs_input.contains(" Readpoles, SkipCalc, MBConv, ReadSigma\n T T F F\n"));

    let xas_ei = read_rixs_line(output.join("xasEI.dat"))?;
    assert_ne!(xas_ei.energy_ev.to_vec(), vec![-1.0]);
    assert_ne!(xas_ei.channels[(0, 0)], 999.0);
    let rixs_ee = read_rixs_map(output.join("rixsEE.dat"))?;
    assert_eq!(rixs_ee.block_lengths, vec![3, 3, 3]);
    assert_eq!(rixs_ee.channels.ncols(), 2);
    Ok(())
}

#[test]
fn full_run_skip_calc_generates_rixs_satellite_outputs_during_complete_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(output.join("XES"))?;
    write_rixs_skip_calc_mbconv_cached_input(&input)?;
    write_edges_dat(output.join("edges.dat"), &sample_rixs_edges_dat())?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_square_map_data())?;
    write_xmu_dat(output.join("XES").join("xmu.dat"), &sample_xes_xmu_dat())?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    let rixs_input = std::fs::read_to_string(output.join("rixs.inp"))?;
    assert!(rixs_input.contains(" Readpoles, SkipCalc, MBConv, ReadSigma\n T T T F\n"));
    assert!(output.join("rixsET-sat.dat").is_file());
    assert!(output.join("herfd-sat.dat").is_file());
    assert!(output.join("xasEI-sat.dat").is_file());
    assert!(output.join("xasEF-sat.dat").is_file());
    let rixs_ee_sat = read_rixs_map(output.join("rixsEE-sat.dat"))?;
    assert_eq!(rixs_ee_sat.block_lengths, vec![3, 3, 3]);
    assert_eq!(rixs_ee_sat.channels.ncols(), 2);
    let log = read_module_log_dat(output.join("logrixs.dat"))?;
    assert_eq!(
        log.lines
            .iter()
            .filter(|line| line.as_str() == "Writing results.")
            .count(),
        2
    );
    Ok(())
}

#[test]
fn full_run_rejects_malformed_rixs_xes_source_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(output.join("XES"))?;
    write_rixs_skip_calc_mbconv_cached_input(&input)?;
    write_edges_dat(output.join("edges.dat"), &sample_rixs_edges_dat())?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_square_map_data())?;
    std::fs::write(output.join("XES").join("xmu.dat"), "not an xmu cache\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("malformed XES satellite source should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_rixs_downstream_solver_error(&message, &output);
    assert!(message.contains("failed to read"), "{message}");
    assert!(message.contains("XES/xmu.dat"), "{message}");
    assert!(
        message.contains("at least one spectrum row is required"),
        "{message}"
    );
    assert!(!message.contains("rixs="), "{message}");
    assert!(
        !message.contains("failed to run supported rixs stage"),
        "{message}"
    );
    assert!(!output.join("herfd-sat.dat").exists());
    assert!(!output.join("xasEI-sat.dat").exists());
    assert!(!output.join("xasEF-sat.dat").exists());
    assert!(!output.join("rixsEE-sat.dat").exists());
    Ok(())
}

#[test]
fn full_run_rejects_inconsistent_rixs_solver_handoffs_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase_1.bin"), &phase)?;
    write_phase_bin(output.join("phase_2.bin"), &phase)?;
    write_xsph_rl_dat(output.join("rl_1.dat"), &sample_rixs_full_run_rl_dat())?;
    write_xsph_rl_dat(output.join("rl_2.dat"), &sample_rixs_full_run_rl_dat())?;
    write_wscrn_dat(
        output.join("wscrn_1.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_wscrn_dat(
        output.join("wscrn_2.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_xsect_dat(output.join("xsect_2.dat"), &sample_xsect_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("inconsistent RIXS solver handoffs should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_incomplete_explicit_rixs_handoff_error(&message);
    assert!(!output.join("L3").exists());
    assert!(!output.join("VAL").exists());
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("herfd.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_rixs_global_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = RixsInput {
        run: true,
        broadening: RixsBroadening {
            gam_ch: 0.000_135_051_2,
            gam_exp_1: 0.000_135_051_2,
            gam_exp_2: 0.000_135_051_2,
        },
        energy_window: RixsEnergyWindow {
            emin_i: 0.0,
            emax_i: 0.0,
            emin_f: 0.0,
            emax_f: 0.0,
        },
        xmu: -367_493_090.027_428_2,
        switches: RixsSwitches {
            read_poles: true,
            skip_calc: false,
            mbconv: true,
            read_sigma: false,
        },
        edges: vec!["L3".to_string(), "VAL".to_string()],
    };
    std::fs::write(temp.path().join("rixs.inp"), rixs_input_string(&input)?)?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(temp.path().join("phase_1.bin"), &phase)?;
    write_phase_bin(temp.path().join("phase_2.bin"), &phase)?;
    std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        !reports
            .iter()
            .any(|report| report.name == "rixs" || report.name == "rixs-handoff"),
        "malformed RIXS global source should not report RIXS completion or solver handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("rixsET.dat").exists());
    assert!(!temp.path().join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_ignores_malformed_shared_rixs_handoffs_when_edge_handoffs_are_explicit()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = RixsInput {
        run: true,
        broadening: RixsBroadening {
            gam_ch: 0.000_135_051_2,
            gam_exp_1: 0.000_135_051_2,
            gam_exp_2: 0.000_135_051_2,
        },
        energy_window: RixsEnergyWindow {
            emin_i: 0.0,
            emax_i: 0.0,
            emin_f: 0.0,
            emax_f: 0.0,
        },
        xmu: -367_493_090.027_428_2,
        switches: RixsSwitches {
            read_poles: true,
            skip_calc: false,
            mbconv: true,
            read_sigma: false,
        },
        edges: vec!["L3".to_string(), "VAL".to_string()],
    };
    std::fs::write(temp.path().join("rixs.inp"), rixs_input_string(&input)?)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&sample_band_global_input(1))?,
    )?;
    write_complete_rixs_full_run_source_handoff(temp.path())?;
    std::fs::write(temp.path().join("phase.bin"), "not a phase.bin cache\n")?;
    std::fs::write(temp.path().join("rl.dat"), "not an rl.dat handoff\n")?;
    std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat handoff\n")?;
    std::fs::write(temp.path().join("gg.bin"), "not a gg.bin handoff\n")?;
    std::fs::write(temp.path().join("xsect.dat"), "not an xsect.dat handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    let report = reports
        .iter()
        .find(|report| report.name == "rixs")
        .context("missing completed RIXS source report")?;
    assert_eq!(report.count, 6);
    assert_eq!(report.unit, "file(s)");
    assert!(
        reports.iter().all(|report| report.name != "rixs-handoff"),
        "complete edge-specific RIXS sources should not fall back to validation-only handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_rixs_map(temp.path().join("rixsET.dat"))?.point_count(),
        4
    );
    assert_eq!(
        read_rixs_line(temp.path().join("herfd.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_line(temp.path().join("xasEI.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_line(temp.path().join("xasEF.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_map(temp.path().join("rixsEE.dat"))?.point_count(),
        4
    );
    assert!(temp.path().join("logrixs.dat").is_file());
    assert!(read_phase_bin(temp.path().join("phase.bin")).is_err());
    Ok(())
}

#[test]
fn full_run_scheduler_prefers_xsph_mpse_source_over_stale_read_sigma_cache() -> Result<()> {
    let source_only = tempfile::tempdir()?;
    write_read_sigma_rixs_scheduler_source_handoff(source_only.path())?;
    write_read_sigma_xsph_mpse_source_handoff(source_only.path())?;

    let source_reports = run_supported_cached_modules(source_only.path())?;

    assert!(
        source_reports
            .iter()
            .any(|report| report.name == "rixs" && report.count == 6),
        "missing completed source-only ReadSigma RIXS report: {:?}",
        source_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        source_reports.iter().all(|report| report.name != "xsph"),
        "MPSE source fixture should not be scheduled as a standalone XSPH stage: {:?}",
        source_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let source_map = read_rixs_map(source_only.path().join("rixsET.dat"))?;
    assert!(!source_only.path().join("mpse.dat").exists());

    let stale_with_source = tempfile::tempdir()?;
    write_read_sigma_rixs_scheduler_source_handoff(stale_with_source.path())?;
    write_read_sigma_xsph_mpse_source_handoff(stale_with_source.path())?;
    let stale_mpse = sample_mpse_dat();
    let source_mpse = xsph::generate_mpse_dat_from_source_handoff(stale_with_source.path())?
        .context("missing generated XSPH MPSE source handoff")?;
    assert_ne!(source_mpse.energy_ev, stale_mpse.energy_ev);
    write_mpse_dat(stale_with_source.path().join("mpse.dat"), &stale_mpse)?;

    let stale_source_reports = run_supported_cached_modules(stale_with_source.path())?;

    assert!(
        stale_source_reports
            .iter()
            .any(|report| report.name == "rixs" && report.count == 6),
        "missing completed stale-cache ReadSigma RIXS report: {:?}",
        stale_source_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        stale_source_reports
            .iter()
            .all(|report| report.name != "xsph"),
        "stale-cache ReadSigma fixture should not be repaired by a standalone XSPH stage: {:?}",
        stale_source_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_mpse_dat(stale_with_source.path().join("mpse.dat"))?,
        stale_mpse
    );
    assert_eq!(
        read_rixs_map(stale_with_source.path().join("rixsET.dat"))?,
        source_map
    );

    let stale_only = tempfile::tempdir()?;
    write_read_sigma_rixs_scheduler_source_handoff(stale_only.path())?;
    write_mpse_dat(stale_only.path().join("mpse.dat"), &stale_mpse)?;

    let stale_only_reports = run_supported_cached_modules(stale_only.path())?;

    assert!(
        stale_only_reports
            .iter()
            .any(|report| report.name == "rixs" && report.count == 6),
        "missing completed stale-only ReadSigma RIXS report: {:?}",
        stale_only_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_ne!(
        read_rixs_map(stale_only.path().join("rixsET.dat"))?,
        source_map,
        "RIXS scheduler output should follow generated XSPH MPSE source instead of stale mpse.dat"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_uses_xsph_mpse_source_for_malformed_read_sigma_cache() -> Result<()> {
    let source_only = tempfile::tempdir()?;
    write_read_sigma_rixs_scheduler_source_handoff(source_only.path())?;
    write_read_sigma_xsph_mpse_source_handoff(source_only.path())?;

    let source_reports = run_supported_cached_modules(source_only.path())?;

    assert!(
        source_reports
            .iter()
            .any(|report| report.name == "rixs" && report.count == 6),
        "missing completed source-only ReadSigma RIXS report: {:?}",
        source_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let source_map = read_rixs_map(source_only.path().join("rixsET.dat"))?;

    let malformed_with_source = tempfile::tempdir()?;
    write_read_sigma_rixs_scheduler_source_handoff(malformed_with_source.path())?;
    write_read_sigma_xsph_mpse_source_handoff(malformed_with_source.path())?;
    let malformed_mpse = b"not an mpse.dat cache\n";
    std::fs::write(
        malformed_with_source.path().join("mpse.dat"),
        malformed_mpse,
    )?;

    let malformed_reports = run_supported_cached_modules(malformed_with_source.path())?;

    assert!(
        malformed_reports
            .iter()
            .any(|report| report.name == "rixs" && report.count == 6),
        "missing completed malformed-cache ReadSigma RIXS report: {:?}",
        malformed_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(
        malformed_reports.iter().all(|report| report.name != "xsph"),
        "malformed-cache ReadSigma fixture should not be repaired by a standalone XSPH stage: {:?}",
        malformed_reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        std::fs::read(malformed_with_source.path().join("mpse.dat"))?,
        malformed_mpse
    );
    assert_eq!(
        read_rixs_map(malformed_with_source.path().join("rixsET.dat"))?,
        source_map
    );
    Ok(())
}

#[test]
fn full_run_writes_rixs_outputs_from_complete_sources_during_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_complete_rixs_full_run_source_handoff(&output)?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert_eq!(read_rixs_map(output.join("rixsET.dat"))?.point_count(), 4);
    assert_eq!(read_rixs_line(output.join("herfd.dat"))?.point_count(), 2);
    assert_eq!(read_rixs_line(output.join("xasEI.dat"))?.point_count(), 2);
    assert_eq!(read_rixs_line(output.join("xasEF.dat"))?.point_count(), 2);
    assert_eq!(read_rixs_map(output.join("rixsEE.dat"))?.point_count(), 4);
    assert!(output.join("logrixs.dat").is_file());
    Ok(())
}

#[test]
fn full_run_writes_rixs_satellite_outputs_from_complete_sources_during_no_scf_run() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(output.join("XES"))?;
    write_rixs_mbconv_cached_input(&input)?;
    write_complete_rixs_full_run_source_handoff(&output)?;
    write_xmu_dat(output.join("XES").join("xmu.dat"), &sample_xes_xmu_dat())?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("apot.bin").is_file());
    assert_eq!(read_rixs_map(output.join("rixsET.dat"))?.point_count(), 4);
    assert_eq!(
        read_rixs_map(output.join("rixsET-sat.dat"))?.point_count(),
        4
    );
    assert_eq!(
        read_rixs_line(output.join("herfd-sat.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_line(output.join("xasEI-sat.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_line(output.join("xasEF-sat.dat"))?.point_count(),
        2
    );
    assert_eq!(
        read_rixs_map(output.join("rixsEE-sat.dat"))?.point_count(),
        4
    );
    let log = read_module_log_dat(output.join("logrixs.dat"))?;
    assert_eq!(
        log.lines
            .iter()
            .filter(|line| line.as_str() == "Writing results.")
            .count(),
        2
    );
    Ok(())
}

#[test]
fn full_run_recovers_shared_phase_before_rixs_gg_mismatch_after_no_scf_pot_completion() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    std::fs::write(output.join("phase.bin"), "not a phase.bin cache\n")?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase_1.bin"), &phase)?;
    write_phase_bin(output.join("phase_2.bin"), &phase)?;
    write_xsph_rl_dat(output.join("rl_1.dat"), &sample_rixs_full_run_rl_dat())?;
    write_xsph_rl_dat(output.join("rl_2.dat"), &sample_rixs_full_run_rl_dat())?;
    write_wscrn_dat(
        output.join("wscrn_1.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_wscrn_dat(
        output.join("wscrn_2.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_xsect_dat(output.join("xsect_2.dat"), &sample_xsect_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("inconsistent edge-specific RIXS phases should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_incomplete_explicit_rixs_handoff_error(&message);
    assert!(!output.join("L3").exists());
    assert!(!output.join("VAL").exists());
    assert_eq!(
        std::fs::read(output.join("phase.bin"))?,
        b"not a phase.bin cache\n"
    );
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("herfd.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_uses_explicit_rixs_screening_before_gg_mismatch_after_no_scf_pot_completion()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase_1.bin"), &phase)?;
    write_phase_bin(output.join("phase_2.bin"), &phase)?;
    write_xsph_rl_dat(output.join("rl_1.dat"), &sample_rixs_full_run_rl_dat())?;
    write_xsph_rl_dat(output.join("rl_2.dat"), &sample_rixs_full_run_rl_dat())?;
    write_wscrn_dat(
        output.join("wscrn_1.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_wscrn_dat(
        output.join("wscrn_2.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_xsect_dat(output.join("xsect_2.dat"), &sample_xsect_dat())?;
    std::fs::write(output.join("vtot.dat"), "not a vtot.dat table\n")?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("inconsistent edge-specific RIXS screening should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_incomplete_explicit_rixs_handoff_error(&message);
    assert!(!output.join("L3").exists());
    assert!(!output.join("VAL").exists());
    assert!(!output.join("wscrn.dat").exists());
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_reports_malformed_rixs_final_cache_gg_mismatch_after_no_scf_pot_completion()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase_1.bin"), &phase)?;
    write_phase_bin(output.join("phase_2.bin"), &phase)?;
    write_xsph_rl_dat(output.join("rl_1.dat"), &sample_rixs_full_run_rl_dat())?;
    write_xsph_rl_dat(output.join("rl_2.dat"), &sample_rixs_full_run_rl_dat())?;
    write_wscrn_dat(
        output.join("wscrn_1.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_wscrn_dat(
        output.join("wscrn_2.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_xsect_dat(output.join("xsect_2.dat"), &sample_xsect_dat())?;
    std::fs::write(output.join("herfd.dat"), "not a RIXS line\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("inconsistent RIXS handoffs should fail before replacing malformed final cache")?;

    let message = format!("{error:#?}");
    assert_incomplete_explicit_rixs_handoff_error(&message);
    assert!(!output.join("L3").exists());
    assert!(!output.join("VAL").exists());
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_rixs_wscrn_before_source_error_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("incomplete RIXS sources should fail after recovering shared screening")?;

    let message = format!("{error:#?}");
    assert_rixs_l3_preparation_xsph_error(&message);
    assert!(read_wscrn_dat(output.join("L3").join("wscrn.dat")).is_ok());
    assert!(!output.join("VAL").exists());
    assert!(!output.join("wscrn.dat").exists());
    assert!(!output.join("phase_1.bin").exists());
    assert!(!output.join("phase_2.bin").exists());
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_rejects_unrecoverable_rixs_wscrn_after_no_scf_pot_completion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    std::fs::write(output.join("vtot.dat"), "not a vtot.dat table\n")?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("unrecoverable RIXS screening should fail the required RIXS stage")?;

    let message = format!("{error:#?}");
    assert_rixs_l3_preparation_xsph_error(&message);
    assert!(read_wscrn_dat(output.join("L3").join("wscrn.dat")).is_ok());
    assert!(!output.join("VAL").exists());
    assert!(!message.contains("rixs-handoff="), "{message}");
    assert!(!output.join("wscrn.dat").exists());
    assert!(!output.join("phase_1.bin").exists());
    assert!(!output.join("phase_2.bin").exists());
    assert!(!output.join("rixsET.dat").exists());
    assert!(!output.join("logrixs.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_screen_wscrn_for_required_rixs_handoff_before_solver_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_screen_handoff_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase.bin"), &phase)?;
    write_xsect_dat(
        output.join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    write_gg_bin(output.join("gg.bin"), &sample_rixs_full_run_gg_data(2, 4))?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_xsph_rl_dat(output.join("rl.dat"), &sample_rixs_full_run_rl_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("RIXS should validate the recovered SCREEN handoff before the solver gate")?;

    let message = format!("{error:#?}");
    assert_rixs_l3_preparation_xsph_error(&message);
    assert!(read_wscrn_dat(output.join("L3").join("wscrn.dat")).is_ok());
    assert!(!output.join("VAL").exists());
    assert!(!output.join("phase_1.bin").exists());
    assert!(!output.join("phase_2.bin").exists());
    assert!(!output.join("wscrn.dat").exists());
    assert!(!output.join("rixsET.dat").exists());
    Ok(())
}

#[test]
fn full_run_executes_cached_rhorrp_before_downstream_xsph_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rhorrp_cached_input(&input)?;
    write_rhorrp_density_text(
        output.join("density.dat"),
        &sample_rhorrp_density_text_data(),
    )?;
    let expected_density = read_rhorrp_density_text(output.join("density.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still require complete caches after cached RHORRP")?;

    let message = format!("{error:#?}");
    assert!(message.contains("rhorrp=1 file(s)"), "{message}");
    assert!(message.contains("pot=5 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(
        message.contains("XSPH required stage needs complete phase.bin/xsect.dat caches"),
        "{message}"
    );
    assert_eq!(
        read_rhorrp_density_text(output.join("density.dat"))?,
        expected_density
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_rhorrp_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("density.inp"), b"line density.dat 0.0\n")?;
    write_rhorrp_density_text(
        temp.path().join("density.dat"),
        &sample_rhorrp_density_text_data(),
    )?;
    let density = read_rhorrp_density_text(temp.path().join("density.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "rhorrp"),
        "malformed density.inp should not report RHORRP complete: {:?}",
        reports
    );
    assert_eq!(
        read_rhorrp_density_text(temp.path().join("density.dat"))?,
        density
    );
    assert!(!temp.path().join("logrhorrp.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_rhorrp_when_core_source_is_malformed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("density.inp"),
        concat!("line density.dat 0.0 0.0 0.0 core\n", "1.0 0.0 0.0 2\n",),
    )?;
    write_rhorrp_density_text(
        temp.path().join("density.dat"),
        &sample_rhorrp_density_text_data(),
    )?;
    let expected_density = read_rhorrp_density_text(temp.path().join("density.dat"))?;
    write_pot_bin(
        temp.path().join("pot.bin"),
        &sample_rhorrp_core_density_pot_bin(),
    )?;
    std::fs::write(temp.path().join("geom.dat"), b"not geom.dat\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "rhorrp"),
        "malformed RHORRP core source should not report cached RHORRP complete: {:?}",
        reports
    );
    assert_eq!(
        read_rhorrp_density_text(temp.path().join("density.dat"))?,
        expected_density
    );
    assert!(!temp.path().join("logrhorrp.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_rhorrp_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_rhorrp_density_text(
        temp.path().join("density.dat"),
        &sample_rhorrp_density_text_data(),
    )?;
    let expected_density = read_rhorrp_density_text(temp.path().join("density.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "rhorrp"),
        "orphan density.dat cache without density.inp should not report RHORRP complete: {:?}",
        reports
    );
    assert_eq!(
        read_rhorrp_density_text(temp.path().join("density.dat"))?,
        expected_density
    );
    assert!(!temp.path().join("logrhorrp.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_rhorrp_core_density_from_pot_cache_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rhorrp_core_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_rhorrp_core_density_pot_bin(),
    )?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still reject the generated RHORRP source stack")?;

    let message = format!("{error:#?}");
    assert!(message.contains("rhorrp=1 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(message.contains("corrected_momentum"), "{message}");
    let density = read_rhorrp_density_text(output.join("density.dat"))?;
    assert_eq!(density.point_count(), 2);
    assert!(
        density
            .density_per_angstrom3
            .iter()
            .all(|value| *value > 0.0)
    );
    assert!(density.nearest.is_some());
    let log = read_module_log_dat(output.join("logrhorrp.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line == "Calculate density: density.dat (       2 total points)")
    );
    Ok(())
}

#[test]
fn full_run_xsph_discovery_declines_rhorrp_pot_refresh_when_xcpot_stops() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rhorrp_core_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_rhorrp_core_density_pot_bin(),
    )?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still reject the generated RHORRP source stack")?;
    let message = format!("{error:#?}");
    assert!(message.contains("rhorrp=1 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(message.contains("corrected_momentum"), "{message}");
    assert!(!message.contains("xsph="), "{message}");
    assert!(!message.contains("xsph-phase="), "{message}");

    assert!(!xsph::has_supported_xsph_output(&output)?);
    assert!(!xsph::has_supported_tdlda_xsedge_output(&output)?);
    assert!(!xsph::has_supported_phase_handoff(&output)?);

    let error = xsph::run_supported_phase_handoff_in_dir(&output)
        .err()
        .context("explicit XSPH phase handoff should stay strict")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to evaluate XSPH xcpot"), "{chain}");
    assert!(chain.contains("corrected_momentum"), "{chain}");
    Ok(())
}

#[test]
fn full_run_recovers_malformed_rhorrp_core_density_from_pot_cache_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rhorrp_core_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_rhorrp_core_density_pot_bin(),
    )?;
    std::fs::write(output.join("density.dat"), b"not RHORRP density\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still reject the recovered RHORRP source stack")?;

    let message = format!("{error:#?}");
    assert!(message.contains("rhorrp=1 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(message.contains("corrected_momentum"), "{message}");
    let density = read_rhorrp_density_text(output.join("density.dat"))?;
    assert_eq!(density.point_count(), 2);
    assert!(
        density
            .density_per_angstrom3
            .iter()
            .all(|value| *value > 0.0)
    );
    assert!(density.nearest.is_some());
    Ok(())
}

#[test]
fn full_run_regenerates_stale_rhorrp_core_density_from_pot_cache_before_xsph_corrected_momentum_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rhorrp_core_source_input(&input)?;
    write_pot_bin(
        output.join("pot.bin"),
        &sample_rhorrp_core_density_pot_bin(),
    )?;
    write_rhorrp_density_text(
        output.join("density.dat"),
        &sample_rhorrp_density_text_data(),
    )?;
    let stale_density = read_rhorrp_density_text(output.join("density.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH should still reject the regenerated RHORRP source stack")?;

    let message = format!("{error:#?}");
    assert!(message.contains("rhorrp=1 file(s)"), "{message}");
    assert!(message.contains("pot=4 file(s)"), "{message}");
    assert!(message.contains("xsph-emesh=2 file(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF xsph stage"),
        "{message}"
    );
    assert!(message.contains("corrected_momentum"), "{message}");
    let density = read_rhorrp_density_text(output.join("density.dat"))?;
    assert_ne!(density, stale_density);
    assert_eq!(density.point_count(), 2);
    assert!(
        density
            .density_per_angstrom3
            .iter()
            .all(|value| *value > 0.0)
    );
    assert!(density.nearest.is_some());
    Ok(())
}

#[test]
fn full_run_generates_atomic_apot_handoff_for_incomplete_wpot_cache_before_xsph_error() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_atomic_pot_handoffs_from_rdinp() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(output.join("pot.inp").is_file());
    assert!(output.join("geom.dat").is_file());
    assert!(output.join("pot.bin").is_file());
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("pot01.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_validates_existing_atomic_config_while_generating_source_apot_before_xsph_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    write_config_dat(
        output.join("config.dat"),
        &sample_atomic_config_handoff_dat(),
    )?;
    let expected = read_config_dat(output.join("config.dat"))?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(read_config_dat(output.join("config.dat"))?, expected);
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("log1.dat").is_file());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_replaces_existing_atomic_log_while_generating_source_apot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_config_dat(
        output.join("config.dat"),
        &sample_atomic_config_handoff_dat(),
    )?;
    std::fs::write(output.join("log1.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    let log = read_module_log_dat(output.join("log1.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating SCF potentials ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_atomic_config_while_generating_source_apot_before_xsph_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    std::fs::write(output.join("config.dat"), "not config.dat\n")?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_stale_atomic_config_while_generating_source_apot_before_xsph_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_atomic_config_pot_bin_data())?;
    let mut stale = sample_atomic_config_handoff_dat();
    stale.potentials.truncate(1);
    write_config_dat(output.join("config.dat"), &stale)?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_config_dat(output.join("config.dat"))?.potential_count(),
        2
    );
    assert!(read_apot_bin(output.join("apot.bin")).is_ok());
    assert!(output.join("pot00.dat").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

fn sample_atomic_config_pot_bin_data() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["CLI atomic config handoff smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.0,
            fermi_level: 0.0,
            interstitial_potential: 0.0,
            interstitial_density: 0.0,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12, 12]),
        muffin_tin_radii: Array1::from_vec(vec![1.1, 1.1]),
        norman_indices: Array1::from_vec(vec![40, 40]),
        atomic_numbers: Array1::from_vec(vec![29, 29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.1, 2.1]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_external_pot_mtdp_data() -> refeff_io::MtdpData {
    refeff_io::MtdpData {
        radial_count: 3,
        atomic_numbers: Array1::from_vec(vec![4]),
        atom_coordinates: Array2::zeros((1, 3)),
        atom_radii: Array1::from_vec(vec![1.25]),
        atom_radius_indices: Array1::from_vec(vec![7]),
        atom_density: Array2::from_shape_vec((3, 1), vec![0.11, 0.12, 0.13])
            .expect("sample MTDP atom density shape"),
        atom_potential: Array2::from_shape_vec((3, 1), vec![-1.0, -1.1, -1.2])
            .expect("sample MTDP atom potential shape"),
        empty_sphere_coordinates: Array2::zeros((0, 3)),
        empty_sphere_radii: Array1::zeros(0),
        empty_sphere_radius_indices: Array1::zeros(0),
        empty_sphere_density: Array2::zeros((3, 0)),
        empty_sphere_potential: Array2::zeros((3, 0)),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: -0.08,
    }
}

fn sample_external_scf_mtdp_data(seed: &PotBinData) -> refeff_io::MtdpData {
    let mut atom_density = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
    let mut atom_potential = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
    for row in 0..POT_BIN_RADIAL_POINTS {
        atom_density[(row, 0)] = seed.electron_density[(row, 0)];
        atom_potential[(row, 0)] = seed.total_potential[(row, 0)];
    }
    atom_density[(0, 0)] += 1.0e-6;
    atom_density[(2, 0)] += 2.0e-6;
    atom_potential[(0, 0)] = -1.0;
    atom_potential[(1, 0)] = -1.1;
    atom_potential[(2, 0)] = -1.2;
    refeff_io::MtdpData {
        radial_count: POT_BIN_RADIAL_POINTS,
        atomic_numbers: Array1::from_vec(vec![4]),
        atom_coordinates: Array2::zeros((1, 3)),
        atom_radii: Array1::from_vec(vec![1.25]),
        atom_radius_indices: Array1::from_vec(vec![7]),
        atom_density,
        atom_potential,
        empty_sphere_coordinates: Array2::zeros((0, 3)),
        empty_sphere_radii: Array1::zeros(0),
        empty_sphere_radius_indices: Array1::zeros(0),
        empty_sphere_density: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
        empty_sphere_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: -0.08,
    }
}

fn sample_atomic_config_handoff_dat() -> ConfigDatData {
    ConfigDatData {
        header_lines: Vec::new(),
        potentials: (0..2)
            .map(|index| ConfigDatPotential {
                potential_index: index,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: Array1::from_shape_fn(CONFIG_DAT_ORBITAL_COUNT, |orbital| {
                    if orbital == 0 { 2.0 } else { 0.0 }
                }),
                valence_occupations: Array1::from_shape_fn(CONFIG_DAT_ORBITAL_COUNT, |orbital| {
                    if orbital == 0 { 1.0 } else { 0.0 }
                }),
                spin_occupations: None,
            })
            .collect(),
    }
}

fn sample_full_run_fms_gg_data() -> GgDatData {
    GgDatData {
        sections: vec![
            GgDatSection {
                section_number: 1,
                values: Array2::from_shape_fn((2, 2), |(row, column)| {
                    let value = 1.0 + row as f64 + 2.0 * column as f64;
                    Complex64::new(value, -0.25 * value)
                }),
                raw_prefix_lines: None,
            },
            GgDatSection {
                section_number: 2,
                values: Array2::from_shape_fn((2, 2), |(row, column)| {
                    let value = 5.0 + row as f64 + column as f64;
                    Complex64::new(value, -0.5 * value)
                }),
                raw_prefix_lines: None,
            },
        ],
    }
}

fn sample_full_run_orphan_gtr_bin() -> GtrBinData {
    GtrBinData {
        point_count_declared: 2,
        horizontal_count: 1,
        danes_extension_count: 1,
        highest_potential_index: 0,
        fms_mode: 2,
        values: Array3::from_shape_fn((2, 1, 1), |(energy, _, _)| {
            Complex64::new(0.1 * (energy + 1) as f64, -0.05 * energy as f64)
        }),
    }
}

fn write_full_run_fms_dmdw_source_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("fms.inp"),
        fms_input_string(&sample_full_run_fms_dmdw_input())?,
    )?;
    write_phase_bin(
        work_dir.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_band_global_input(1))?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_full_run_fms_dmdw_geom())?,
    )?;
    std::fs::write(
        work_dir.join("dmdw.inp"),
        "   1\n   1\n   1    190.000\n   0\nfeff.dym\n   0\n",
    )?;
    Ok(())
}

fn sample_full_run_fms_dmdw_input() -> FmsInput {
    FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: 5,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 3.0,
            rdirec: 5.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 190.0,
            thetad: 315.0,
            sig2g: 0.0,
        },
        lmaxph: vec![1, 1],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    }
}

fn sample_full_run_fms_source_input() -> FmsInput {
    FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: -1,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 3.0,
            rdirec: 5.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 190.0,
            thetad: 315.0,
            sig2g: 0.0,
        },
        lmaxph: vec![1, 1],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    }
}

fn sample_full_run_fms_dmdw_geom() -> GeomDat {
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
                x: 1.4,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 0,
            },
        ],
    }
}

fn assert_gg_data_values_eq(actual: &GgDatData, expected: &GgDatData) {
    assert_eq!(actual.sections.len(), expected.sections.len());
    for (actual, expected) in actual.sections.iter().zip(&expected.sections) {
        assert_eq!(actual.section_number, expected.section_number);
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
            assert!(
                (*actual - *expected).norm() <= 1.0e-12,
                "expected {actual:?} to match {expected:?}"
            );
        }
    }
}

fn add_to_first_real_apot_matrix_value(apot: &mut ApotBinData, delta: f64) {
    let Some(values) = apot.sections.iter_mut().find_map(|section| {
        if let ApotBinPayload::Matrix(matrix) = &mut section.payload
            && let ApotBinMatrixValues::Real(values) = &mut matrix.values
        {
            return Some(values);
        }
        None
    }) else {
        panic!("sample apot.bin should contain a real matrix section");
    };
    values[(0, 0)] += delta;
}

fn assert_phase_transition_moments_close(
    actual: &PhaseBinData,
    expected: &PhaseBinData,
    tolerance: f64,
) {
    assert_eq!(
        actual.transition_moments.dim(),
        expected.transition_moments.dim()
    );
    for ((energy, q, transition, spin), actual) in actual.transition_moments.indexed_iter() {
        let expected = expected.transition_moments[(energy, q, transition, spin)];
        let difference = (*actual - expected).norm();
        let tolerance = tolerance * expected.norm().max(1.0);
        assert!(
            difference <= tolerance,
            "transition moment differs at ({energy}, {q}, {transition}, {spin}) by {difference:e}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:e}"
        );
    }
}

fn assert_reference_phase_bin_close(
    actual: &PhaseBinData,
    expected: &PhaseBinData,
    tolerance: f64,
) {
    assert_eq!(actual.spin_count, expected.spin_count);
    assert_eq!(actual.energy_count, expected.energy_count);
    assert_eq!(actual.main_energy_count, expected.main_energy_count);
    assert_eq!(
        actual.auxiliary_energy_count,
        expected.auxiliary_energy_count
    );
    assert_eq!(actual.ihole, expected.ihole);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_eq!(actual.pad_width, expected.pad_width);
    assert_float_close_with_tolerance(
        actual.scalars.average_norman_radius,
        expected.scalars.average_norman_radius,
        tolerance,
        "phase average Norman radius",
    );
    assert_float_close_with_tolerance(
        actual.scalars.fermi_level,
        expected.scalars.fermi_level,
        tolerance,
        "phase Fermi level",
    );
    assert_float_close_with_tolerance(
        actual.scalars.edge_energy,
        expected.scalars.edge_energy,
        tolerance,
        "phase edge energy",
    );
    assert_complex_array1_close(
        "phase energy grid",
        &actual.energy_grid,
        &expected.energy_grid,
        tolerance,
    );
    assert_eq!(
        actual.reference_energy.dim(),
        expected.reference_energy.dim()
    );
    for ((energy, spin), actual) in actual.reference_energy.indexed_iter() {
        let expected = expected.reference_energy[(energy, spin)];
        assert_complex_close_with_tolerance(
            *actual,
            expected,
            tolerance,
            &format!("phase reference energy ({energy}, {spin})"),
        );
    }
    assert_eq!(actual.potential_count(), expected.potential_count());
    for (potential_index, (actual, expected)) in actual
        .potentials
        .iter()
        .zip(expected.potentials.iter())
        .enumerate()
    {
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_eq!(actual.label, expected.label);
        assert_eq!(actual.lmax, expected.lmax);
        assert_eq!(actual.phase_shifts.dim(), expected.phase_shifts.dim());
        for ((energy, angular, spin), actual) in actual.phase_shifts.indexed_iter() {
            let expected = expected.phase_shifts[(energy, angular, spin)];
            assert_complex_close_with_tolerance(
                *actual,
                expected,
                tolerance,
                &format!("phase shift ({potential_index}, {energy}, {angular}, {spin})"),
            );
        }
    }
    assert!(
        actual
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
}

fn assert_reference_xsect_dat_close(actual: &XsectDatData, expected: &XsectDatData, label: &str) {
    assert_reference_xsect_dat_close_with_tolerance(
        actual, expected, label, 1.0e-5, 0.20, 2.0e-5, 0.25,
    );
}

fn assert_reference_xsect_dat_close_with_tolerance(
    actual: &XsectDatData,
    expected: &XsectDatData,
    label: &str,
    background_absolute: f64,
    background_relative: f64,
    cross_section_absolute: f64,
    cross_section_relative: f64,
) {
    assert_reference_xsect_dat_close_with_tolerance_and_nonzero(
        actual,
        expected,
        label,
        background_absolute,
        background_relative,
        cross_section_absolute,
        cross_section_relative,
        true,
    );
}

fn assert_reference_xsect_dat_close_allow_zero_with_tolerance(
    actual: &XsectDatData,
    expected: &XsectDatData,
    label: &str,
    background_absolute: f64,
    background_relative: f64,
    cross_section_absolute: f64,
    cross_section_relative: f64,
) {
    assert_reference_xsect_dat_close_with_tolerance_and_nonzero(
        actual,
        expected,
        label,
        background_absolute,
        background_relative,
        cross_section_absolute,
        cross_section_relative,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_reference_xsect_dat_close_with_tolerance_and_nonzero(
    actual: &XsectDatData,
    expected: &XsectDatData,
    label: &str,
    background_absolute: f64,
    background_relative: f64,
    cross_section_absolute: f64,
    cross_section_relative: f64,
    require_nonzero: bool,
) {
    assert_eq!(actual.energy_count(), expected.energy_count(), "{label}");
    assert_eq!(
        actual.main_energy_count, expected.main_energy_count,
        "{label}"
    );
    assert_eq!(actual.fermi_index, expected.fermi_index, "{label}");
    assert_complex_array1_close(
        &format!("{label} energy grid"),
        &actual.energy_grid_ev,
        &expected.energy_grid_ev,
        1.0e-8,
    );
    assert_real_column_close_mixed(
        &format!("{label} normalized background"),
        &actual.normalized_background,
        &expected.normalized_background,
        background_absolute,
        background_relative,
    );
    assert_complex_column_close_mixed(
        &format!("{label} cross section"),
        &actual.cross_section,
        &expected.cross_section,
        cross_section_absolute,
        cross_section_relative,
    );
    if require_nonzero {
        assert!(
            actual.cross_section.iter().any(|value| value.norm() > 0.0),
            "{label} should contain nonzero cross-section rows"
        );
    }
}

fn assert_real_column_close_mixed(
    label: &str,
    actual: &ndarray::Array1<f64>,
    expected: &ndarray::Array1<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        let difference = (actual - expected).abs();
        assert!(
            difference <= tolerance,
            "{label} row {index} differs by {difference:e}: actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
        );
    }
}

fn assert_complex_column_close_mixed(
    label: &str,
    actual: &ndarray::Array1<Complex64>,
    expected: &ndarray::Array1<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let real_tolerance = absolute_tolerance + relative_tolerance * expected.re.abs();
        let real_difference = (actual.re - expected.re).abs();
        assert!(
            real_difference <= real_tolerance,
            "{label} row {index} real differs by {real_difference:e}: actual={:e}, expected={:e}, tolerance={real_tolerance:e}",
            actual.re,
            expected.re
        );
        let imaginary_tolerance = absolute_tolerance + relative_tolerance * expected.im.abs();
        let imaginary_difference = (actual.im - expected.im).abs();
        assert!(
            imaginary_difference <= imaginary_tolerance,
            "{label} row {index} imaginary differs by {imaginary_difference:e}: actual={:e}, expected={:e}, tolerance={imaginary_tolerance:e}",
            actual.im,
            expected.im
        );
    }
}

fn assert_wscrn_reference_close(actual: &WscrnDatData, expected: &WscrnDatData, tolerance: f64) {
    assert_eq!(actual.row_count(), expected.row_count());
    assert_float_columns_close(
        "wscrn.dat radius",
        &actual.radius_bohr,
        &expected.radius_bohr,
        tolerance,
    );
    assert_float_columns_close(
        "wscrn.dat screened potential",
        &actual.screened_potential,
        &expected.screened_potential,
        tolerance,
    );
    assert_float_columns_close(
        "wscrn.dat core-hole potential",
        &actual.core_hole_potential,
        &expected.core_hole_potential,
        tolerance,
    );
}

fn assert_vtot_reference_close(actual: &VtotDatData, expected: &VtotDatData, tolerance: f64) {
    assert_eq!(actual.header_lines, expected.header_lines);
    assert_eq!(actual.row_count(), expected.row_count());
    assert_float_columns_close(
        "vtot.dat radius",
        &actual.radius_bohr,
        &expected.radius_bohr,
        tolerance,
    );
    assert_float_columns_close(
        "vtot.dat total potential",
        &actual.total_potential,
        &expected.total_potential,
        tolerance,
    );
    assert_float_columns_close(
        "vtot.dat screened core-hole potential",
        &actual.screened_core_hole_potential,
        &expected.screened_core_hole_potential,
        tolerance,
    );
}

fn assert_crpa_reference_close(actual: &CrpaDatData, expected: &CrpaDatData, tolerance: f64) {
    for (label, actual, expected) in [
        ("crpa.dat Hubbard U", actual.hubbard_u, expected.hubbard_u),
        (
            "crpa.dat occupation",
            actual.occupation,
            expected.occupation,
        ),
        ("crpa.dat bare U", actual.bare_u, expected.bare_u),
    ] {
        assert_float_close_with_tolerance(actual, expected, tolerance, label);
    }
}

fn assert_wscrn_screened_reference_close(
    actual: &WscrnDatData,
    expected: &WscrnDatData,
    tolerance: f64,
) {
    assert_eq!(actual.row_count(), expected.row_count());
    assert_float_columns_close(
        "CRPA wscrn.dat radius",
        &actual.radius_bohr,
        &expected.radius_bohr,
        tolerance,
    );
    assert_float_columns_close(
        "CRPA wscrn.dat screened potential",
        &actual.screened_potential,
        &expected.screened_potential,
        tolerance,
    );
}

fn assert_ff2x_chi_reference_close(actual: &ChiDatData, expected: &ChiDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    for row in 0..actual.point_count() {
        assert_float_close_with_tolerance(
            actual.wave_number[row],
            expected.wave_number[row],
            1.0e-12,
            &format!("FF2X chi.dat wave number {row}"),
        );
        assert_float_close_with_tolerance(
            actual.chi[row],
            expected.chi[row],
            5.0e-7,
            &format!("FF2X chi.dat chi {row}"),
        );
        assert_float_close_with_tolerance(
            actual.magnitude[row],
            expected.magnitude[row],
            5.0e-7,
            &format!("FF2X chi.dat magnitude {row}"),
        );
        assert_float_close_with_tolerance(
            actual.phase[row],
            expected.phase[row],
            2.0e-5,
            &format!("FF2X chi.dat phase {row}"),
        );
    }
}

fn assert_ff2x_xmu_reference_close(actual: &XmuDatData, expected: &XmuDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_eq!(
        actual.normalization.is_some(),
        expected.normalization.is_some()
    );
    if let (Some(actual), Some(expected)) = (actual.normalization, expected.normalization) {
        assert_float_close_with_tolerance(actual, expected, 5.0e-9, "FF2X xmu.dat normalization");
    }
    for row in 0..actual.point_count() {
        assert_float_close_with_tolerance(
            actual.photon_energy_ev[row],
            expected.photon_energy_ev[row],
            1.0e-3,
            &format!("FF2X xmu.dat photon energy {row}"),
        );
        assert_float_close_with_tolerance(
            actual.relative_energy_ev[row],
            expected.relative_energy_ev[row],
            1.0e-3,
            &format!("FF2X xmu.dat relative energy {row}"),
        );
        assert_float_close_with_tolerance(
            actual.wave_number[row],
            expected.wave_number[row],
            1.0e-12,
            &format!("FF2X xmu.dat wave number {row}"),
        );
        assert_float_close_with_tolerance(
            actual.mu[row],
            expected.mu[row],
            1.0e-5,
            &format!("FF2X xmu.dat mu {row}"),
        );
        assert_float_close_with_tolerance(
            actual.mu0[row],
            expected.mu0[row],
            1.0e-5,
            &format!("FF2X xmu.dat mu0 {row}"),
        );
        assert_float_close_with_tolerance(
            actual.chi[row],
            expected.chi[row],
            1.0e-6,
            &format!("FF2X xmu.dat chi {row}"),
        );
    }
}

fn assert_loss_dat_reference_close(actual: &LossDatData, expected: &LossDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_float_columns_close(
        "OPCONS loss.dat energy",
        &actual.energy_ev,
        &expected.energy_ev,
        2.0e-6,
    );
    assert_float_columns_close("OPCONS loss.dat loss", &actual.loss, &expected.loss, 2.0e-5);
}

fn assert_self_exc_dat_reference_close(actual: &ExcDatData, expected: &ExcDatData) {
    assert_eq!(actual.header_lines, expected.header_lines);
    assert_float_columns_close(
        "SELF exc.dat energy",
        &actual.energy_ev,
        &expected.energy_ev,
        5.0e-10,
    );
    assert_float_columns_close(
        "SELF exc.dat broadening",
        &actual.broadening_ev,
        &expected.broadening_ev,
        5.0e-10,
    );
    assert_float_columns_close(
        "SELF exc.dat oscillator strength",
        &actual.oscillator_strength,
        &expected.oscillator_strength,
        5.0e-10,
    );
    match (&actual.auxiliary_weight, &expected.auxiliary_weight) {
        (Some(actual), Some(expected)) => {
            assert_float_columns_close("SELF exc.dat auxiliary weight", actual, expected, 5.0e-10)
        }
        (None, None) => {}
        _ => panic!("SELF exc.dat auxiliary weight presence mismatch"),
    }
}

fn assert_float_columns_close(
    label: &str,
    actual: &ndarray::Array1<f64>,
    expected: &ndarray::Array1<f64>,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_float_close_with_tolerance(actual, expected, tolerance, &format!("{label} {index}"));
    }
}

fn assert_emesh_dat_close(actual: &EmeshDatData, expected: &EmeshDatData, tolerance: f64) {
    assert_float_close_with_tolerance(
        actual.edge_hartree,
        expected.edge_hartree,
        tolerance,
        "emesh edge Hartree",
    );
    assert_float_close_with_tolerance(
        actual.bohr_angstrom,
        expected.bohr_angstrom,
        tolerance,
        "emesh Bohr radius",
    );
    assert_float_close_with_tolerance(actual.edge_ev, expected.edge_ev, tolerance, "emesh edge eV");
    assert_eq!(actual.spectrum, expected.spectrum);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_eq!(actual.indices, expected.indices);
    assert_eq!(actual.energy_ev.len(), expected.energy_ev.len());
    for (index, (&actual, &expected)) in actual
        .energy_ev
        .iter()
        .zip(expected.energy_ev.iter())
        .enumerate()
    {
        assert_float_close_with_tolerance(
            actual,
            expected,
            tolerance,
            &format!("emesh energy eV {index}"),
        );
    }
    assert_eq!(
        actual.wave_number_inverse_angstrom.len(),
        expected.wave_number_inverse_angstrom.len()
    );
    for (index, (&actual, &expected)) in actual
        .wave_number_inverse_angstrom
        .iter()
        .zip(expected.wave_number_inverse_angstrom.iter())
        .enumerate()
    {
        assert_float_close_with_tolerance(
            actual,
            expected,
            tolerance,
            &format!("emesh wave number {index}"),
        );
    }
}

fn assert_emesh_bin_close(actual: &EmeshBinData, expected: &EmeshBinData, tolerance: f64) {
    assert_eq!(actual.point_count_declared, expected.point_count_declared);
    assert_eq!(actual.horizontal_count, expected.horizontal_count);
    assert_eq!(actual.danes_extension_count, expected.danes_extension_count);
    assert_complex_array1_close(
        "emesh.bin energy",
        &actual.energy_hartree,
        &expected.energy_hartree,
        tolerance,
    );
}

fn assert_kmesh_dat_close(actual: &KmeshDatData, expected: &KmeshDatData, tolerance: f64) {
    assert_eq!(actual.rows.len(), expected.rows.len());
    for (index, (actual, expected)) in actual.rows.iter().zip(expected.rows.iter()).enumerate() {
        assert_eq!(actual.index, expected.index);
        assert_eq!(actual.metadata, expected.metadata);
        for axis in 0..3 {
            assert_float_close_with_tolerance(
                actual.k_point[axis],
                expected.k_point[axis],
                tolerance,
                &format!("kmesh row {index} axis {axis}"),
            );
        }
        assert_float_close_with_tolerance(
            actual.weight,
            expected.weight,
            tolerance,
            &format!("kmesh row {index} weight"),
        );
    }
}

fn assert_bandstructure_dat_close(actual: &BandstructureDatData, expected: &BandstructureDatData) {
    const KPOINT_TOLERANCE: f64 = 5.0e-4;
    const BAND_VALUE_TOLERANCE: f64 = 5.0e-5;

    assert_eq!(
        actual.header_lines.len(),
        expected.header_lines.len(),
        "bandstructure.dat header line count changed"
    );
    for (index, (actual, expected)) in actual
        .header_lines
        .iter()
        .zip(expected.header_lines.iter())
        .enumerate()
    {
        assert!(
            actual.split_whitespace().eq(expected.split_whitespace()),
            "bandstructure.dat header line {index} changed: actual={actual:?}, expected={expected:?}"
        );
    }

    assert_eq!(
        actual.rows.len(),
        expected.rows.len(),
        "bandstructure.dat k-point row count changed"
    );
    for (row_index, (actual, expected)) in actual.rows.iter().zip(expected.rows.iter()).enumerate()
    {
        assert_eq!(actual.index, expected.index);
        for axis in 0..3 {
            assert_float_close_with_tolerance(
                actual.k_point[axis],
                expected.k_point[axis],
                KPOINT_TOLERANCE,
                &format!("bandstructure row {row_index} k-point axis {axis}"),
            );
        }
        assert_eq!(
            actual.bands.len(),
            expected.bands.len(),
            "bandstructure row {row_index} band count changed"
        );
        for (band_index, (&actual, &expected)) in
            actual.bands.iter().zip(expected.bands.iter()).enumerate()
        {
            assert_float_close_with_tolerance(
                actual,
                expected,
                BAND_VALUE_TOLERANCE,
                &format!("bandstructure row {row_index} band {band_index}"),
            );
        }
    }
}

fn assert_complex_array1_close(
    label: &str,
    actual: &ndarray::Array1<Complex64>,
    expected: &ndarray::Array1<Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_complex_close_with_tolerance(
            actual,
            expected,
            tolerance,
            &format!("{label} {index}"),
        );
    }
}

fn assert_complex_close_with_tolerance(
    actual: Complex64,
    expected: Complex64,
    tolerance: f64,
    label: &str,
) {
    let difference = (actual - expected).norm();
    let tolerance = tolerance * expected.norm().max(1.0);
    assert!(
        difference <= tolerance,
        "{label} differs by {difference:e}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:e}"
    );
}

fn assert_float_close_with_tolerance(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let difference = (actual - expected).abs();
    let tolerance = tolerance * expected.abs().max(1.0);
    assert!(
        difference <= tolerance,
        "{label} differs by {difference:e}: actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

fn assert_ldos_reference_table_close(actual: &LdosDatData, expected: &LdosDatData, label: &str) {
    assert_eq!(actual.fermi_level_ev, expected.fermi_level_ev, "{label}");
    assert_eq!(actual.charge_transfer, expected.charge_transfer, "{label}");
    assert_eq!(actual.electron_counts, expected.electron_counts, "{label}");
    if actual.atom_count.is_some() || expected.atom_count.is_none() {
        assert_eq!(actual.atom_count, expected.atom_count, "{label}");
    }
    assert_eq!(
        actual.lorentzian_hwhh_ev, expected.lorentzian_hwhh_ev,
        "{label}"
    );
    assert_eq!(actual.energy_ev.len(), expected.energy_ev.len(), "{label}");
    assert_eq!(actual.density.dim(), expected.density.dim(), "{label}");
    for (row, (actual, expected)) in actual
        .energy_ev
        .iter()
        .zip(expected.energy_ev.iter())
        .enumerate()
    {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 5.0e-4,
            "{label}: energy[{row}] actual={actual}, expected={expected}, diff={diff}"
        );
    }
    for ((row, column), actual) in actual.density.indexed_iter() {
        let expected = expected.density[(row, column)];
        let diff = (actual - expected).abs();
        let rel = diff / expected.abs().max(1.0e-30);
        assert!(
            diff <= 5.0e-5 || rel <= 1.5e-3,
            "{label}: density[{row},{column}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
        );
    }
}

fn assert_pot_bin_reference_rows_close(generated: &PotBinData, reference: &PotBinData) {
    assert_pot_bin_reference_electron_density_rows_close(generated, reference);
    assert_pot_row_values_close(
        "POT valence density",
        generated.valence_density.iter().copied(),
        reference.valence_density.iter().copied(),
        1.0,
    );
}

fn assert_pot_bin_reference_electron_density_rows_close(
    generated: &PotBinData,
    reference: &PotBinData,
) {
    assert_pot_bin_reference_geometry_rows_close(generated, reference);
    assert_pot_row_values_close(
        "POT electron density",
        generated.electron_density.iter().copied(),
        reference.electron_density.iter().copied(),
        1.0,
    );
}

fn assert_pot_bin_reference_geometry_rows_close(generated: &PotBinData, reference: &PotBinData) {
    assert_pot_row_values_close(
        "POT muffin-tin radii",
        generated.muffin_tin_radii.iter().copied(),
        reference.muffin_tin_radii.iter().copied(),
        2.5e-1,
    );
    assert_pot_row_values_close(
        "POT Norman radii",
        generated.norman_radii.iter().copied(),
        reference.norman_radii.iter().copied(),
        2.5e-1,
    );
    assert_pot_row_values_close(
        "POT overlap factors",
        generated.overlap_factors.iter().copied(),
        reference.overlap_factors.iter().copied(),
        2.5e-1,
    );
}

fn assert_pot_row_values_close<A, E>(label: &str, actual: A, expected: E, relative_tolerance: f64)
where
    A: IntoIterator<Item = f64>,
    E: IntoIterator<Item = f64>,
{
    let actual = actual.into_iter().collect::<Vec<_>>();
    let expected = expected.into_iter().collect::<Vec<_>>();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length changed from FEFF reference"
    );

    let mut compared = 0usize;
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            assert!(
                actual == expected,
                "{label}[{index}] finite mismatch: actual={actual}, expected={expected}"
            );
            continue;
        }
        compared += 1;
        let allowed = relative_tolerance * expected.abs().max(1.0);
        let diff = (actual - expected).abs();
        assert!(
            diff <= allowed,
            "{label}[{index}] differs from FEFF reference: actual={actual}, expected={expected}, diff={diff}, allowed={allowed}"
        );
    }
    assert!(compared > 0, "{label} comparison was empty");
}

fn assert_fpf0_close(actual: &Fpf0DatData, expected: &Fpf0DatData) {
    assert_eq!(actual.atomic_number, expected.atomic_number);
    assert_fpf0_value_close(
        actual.total_energy_fprime,
        expected.total_energy_fprime,
        1.0e-6,
        "total_energy_fprime",
    );
    assert_fpf0_value_close(
        actual.relativistic_correction,
        expected.relativistic_correction,
        1.0e-6,
        "relativistic_correction",
    );
    assert_eq!(actual.oscillators, expected.oscillators);
    assert_eq!(actual.form_factor_momentum, expected.form_factor_momentum);
    assert_eq!(actual.form_factor_count(), expected.form_factor_count());
    for (index, (&actual, &expected)) in actual
        .form_factor
        .iter()
        .zip(expected.form_factor.iter())
        .enumerate()
    {
        assert_fpf0_value_close(actual, expected, 1.5e-4, &format!("form_factor[{index}]"));
    }
}

fn assert_fpf0_value_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{label} differs by {difference:e}: actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

fn sample_single_kmesh_dat() -> KmeshDatData {
    KmeshDatData {
        rows: vec![KmeshRow {
            index: 1,
            k_point: [0.0, 0.0, 0.0],
            weight: 1.0,
            metadata: Some(KmeshMetadata {
                requested_points: 1,
                irreducible_points: 1,
                divisions: [1, 1, 1],
            }),
        }],
    }
}

fn write_reciprocal_bandstructure_input(path: &Path) -> Result<()> {
    write_reciprocal_bandstructure_input_with_freeprop(path, true)
}

fn write_reciprocal_bandstructure_input_with_freeprop(path: &Path, freeprop: bool) -> Result<()> {
    let freeprop_flag = if freeprop { "T" } else { "F" };
    std::fs::write(
        path,
        format!(
            r#"
TITLE Cu reciprocal kmesh run
EDGE K
BANDSTRUCTURE -5.0 10.0 0.25 2 8 {freeprop_flag}
RECIPROCAL
KMESH 8 0
TARGET 1
SGROUP 221
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        ),
    )?;
    Ok(())
}

fn write_single_potential_rel_band_scheduler_handoffs(work_dir: &Path) -> Result<()> {
    write_single_potential_band_scheduler_handoffs(
        work_dir,
        false,
        &sample_band_handoff_phase_bin(),
    )
}

fn write_single_potential_freeprop_band_scheduler_handoffs(work_dir: &Path) -> Result<()> {
    write_single_potential_band_scheduler_handoffs(work_dir, true, &sample_band_handoff_phase_bin())
}

fn write_single_potential_two_spin_degenerate_band_scheduler_handoffs(
    work_dir: &Path,
) -> Result<()> {
    write_single_potential_band_scheduler_handoffs(
        work_dir,
        false,
        &sample_two_spin_degenerate_band_handoff_phase_bin(),
    )
}

fn write_single_potential_two_spin_non_degenerate_band_scheduler_handoffs(
    work_dir: &Path,
    freeprop: bool,
) -> Result<()> {
    let mut phase = sample_two_spin_degenerate_band_handoff_phase_bin();
    phase.reference_energy[(0, 1)].re += 0.01;
    write_single_potential_band_scheduler_handoffs(work_dir, freeprop, &phase)
}

fn write_single_potential_band_scheduler_handoffs(
    work_dir: &Path,
    freeprop: bool,
    phase: &PhaseBinData,
) -> Result<()> {
    let input = BandInput {
        mband: 1,
        energy_mesh: BandEnergyMesh {
            emin: -5.0,
            emax: 10.0,
            estep: 0.25,
        },
        nkp: 2,
        ikpath: 1,
        freeprop,
    };
    std::fs::write(work_dir.join("band.inp"), band_input_string(&input)?)?;
    write_phase_bin(work_dir.join("phase.bin"), phase)?;
    std::fs::write(
        work_dir.join("reciprocal.inp"),
        reciprocal_input_string(&sample_single_potential_reciprocal_input(8))?,
    )?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_band_global_input(1))?,
    )?;
    Ok(())
}

fn write_enabled_band_handoff_input(work_dir: &Path) -> Result<()> {
    let input = BandInput {
        mband: 1,
        energy_mesh: BandEnergyMesh {
            emin: -5.0,
            emax: 10.0,
            estep: 0.25,
        },
        nkp: 2,
        ikpath: 1,
        freeprop: false,
    };
    std::fs::write(work_dir.join("band.inp"), band_input_string(&input)?)?;
    Ok(())
}

fn sample_single_potential_reciprocal_input(total_kpoints: i32) -> ReciprocalInput {
    ReciprocalInput {
        ispace: 0,
        cell: Some(ReciprocalCell {
            lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: "P".to_string(),
            space_group_hm: "Pm-3m".to_string(),
            space_group: 221,
            atom_count: 1,
            absorber: 1,
            core_hole: 1,
            k_mesh: ReciprocalKMesh {
                total: total_kpoints,
                x: total_kpoints,
                y: 0,
                z: 0,
                kind: 3,
                use_symmetry: false,
            },
            positions: vec![[0.0, 0.0, 0.0]],
            potentials: vec![0],
            labels: vec!["Cu".to_string()],
            stretch: [0.0, 0.0, 0.0],
        }),
    }
}

fn sample_band_global_input(ispin: i32) -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: 0,
            ispin,
            le2: 0,
            elpty: 0.0,
            angks: 0.0,
            l2lp: 0,
            do_nrixs: 0,
            ldecmx: 0,
            lj: 0,
        },
        evec: [0.0, 0.0, 1.0],
        xivec: [1.0, 0.0, 0.0],
        spvec: [0.0, 0.0, 1.0],
        polarization_tensor: [[0.0; 6]; 3],
        norms: GlobalNorms {
            evnorm: 1.0,
            xivnorm: 1.0,
            spvnorm: 1.0,
        },
        q_control: GlobalQControl {
            nq: 0,
            imdff: 0,
            qaverage: false,
            mixdff: false,
        },
        q_vectors: Vec::new(),
        mdff: None,
    }
}

fn sample_rixs_edges_dat() -> EdgesDatData {
    EdgesDatData {
        header_lines: vec!["# emu, M_kk, gam".to_string()],
        rows: vec![
            EdgesDatRow {
                chemical_potential: 10.0,
                matrix_element: 1.0,
                core_hole_width: 0.1,
            },
            EdgesDatRow {
                chemical_potential: 4.0,
                matrix_element: 1.0,
                core_hole_width: 0.1,
            },
        ],
    }
}

fn sample_rixs_square_map_data() -> RixsMapData {
    let mut first_energy_ev = Vec::new();
    let mut second_energy_ev = Vec::new();
    let mut channels = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            let value = (row * 3 + col + 1) as f64;
            first_energy_ev.push((10.0 + col as f64) * 27.211_396);
            second_energy_ev.push((4.0 + row as f64) * 27.211_396);
            channels.push(value);
            channels.push(value + 100.0);
        }
    }

    RixsMapData {
        header_lines: Vec::new(),
        block_lengths: vec![3, 3, 3],
        first_energy_ev: Array1::from_vec(first_energy_ev),
        second_energy_ev: Array1::from_vec(second_energy_ev),
        channels: Array2::from_shape_vec((9, 2), channels).expect("valid RIXS sample shape"),
    }
}

fn sample_rixs_full_run_rl_dat() -> XsphRlDatData {
    let energy_count = 2;
    let angular_limit = 1;
    let radial_count = 4;
    let mut records = Vec::new();
    for energy in 0..energy_count {
        for angular in 0..=angular_limit {
            records.push(XsphRlDatRecord {
                energy: Complex64::new(0.25 + energy as f64, 0.01 * energy as f64),
                angular_momentum: angular,
                phase_shift: Complex64::new(0.02 * (energy + 1) as f64, 0.01 * angular as f64),
                regular_large: Array1::from_shape_fn(radial_count, |radial| {
                    Complex64::new(
                        0.1 * (energy + 1) as f64 + 0.01 * angular as f64 + 0.001 * radial as f64,
                        0.0,
                    )
                }),
                regular_small: Array1::from_shape_fn(radial_count, |radial| {
                    Complex64::new(
                        0.05 * (energy + 1) as f64
                            + 0.005 * angular as f64
                            + 0.0005 * radial as f64,
                        0.0,
                    )
                }),
            });
        }
    }

    XsphRlDatData {
        muffin_tin_radius: 1.4,
        angular_limit,
        radial_match_index_1based: radial_count,
        log_step: 0.05,
        grid_origin: 8.8,
        records,
    }
}

fn sample_rixs_full_run_wscrn_dat(row_count: usize) -> WscrnDatData {
    WscrnDatData {
        header_lines: vec![" # r       w_scrn(r)      v_ch(r)".to_string()],
        radius_bohr: Array1::from_shape_fn(row_count, |row| 0.001 + 0.0001 * row as f64),
        screened_potential: Array1::from_shape_fn(row_count, |row| 26.0 + 0.01 * row as f64),
        core_hole_potential: Array1::from_shape_fn(row_count, |row| 28.0 + 0.01 * row as f64),
    }
}

fn sample_rixs_full_run_gg_data(section_count: usize, order: usize) -> GgDatData {
    GgDatData {
        sections: (0..section_count)
            .map(|section| GgDatSection {
                section_number: section + 1,
                values: Array2::from_shape_fn((order, order), |(row, column)| {
                    Complex64::new(
                        0.1 * (section + 1) as f64 + 0.01 * row as f64,
                        -0.02 * column as f64,
                    )
                }),
                raw_prefix_lines: None,
            })
            .collect(),
    }
}

fn write_complete_rixs_full_run_source_handoff(output: &Path) -> Result<()> {
    let phase = sample_fms_source_phase_bin_data();
    write_phase_bin(output.join("phase_1.bin"), &phase)?;
    write_phase_bin(output.join("phase_2.bin"), &phase)?;
    write_xsph_rl_dat(output.join("rl_1.dat"), &sample_rixs_full_run_rl_dat())?;
    write_xsph_rl_dat(output.join("rl_2.dat"), &sample_rixs_full_run_rl_dat())?;
    write_wscrn_dat(
        output.join("wscrn_1.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_wscrn_dat(
        output.join("wscrn_2.dat"),
        &sample_rixs_full_run_wscrn_dat(4),
    )?;
    write_gg_bin(output.join("gg_1.bin"), &sample_rixs_full_run_gg_data(2, 4))?;
    write_gg_bin(output.join("gg_2.bin"), &sample_rixs_full_run_gg_data(2, 4))?;
    write_xsect_dat(output.join("xsect_2.dat"), &sample_xsect_dat())?;
    write_edges_dat(output.join("edges.dat"), &sample_rixs_edges_dat())?;
    Ok(())
}

fn write_read_sigma_rixs_scheduler_source_handoff(work_dir: &Path) -> Result<()> {
    let input = RixsInput {
        run: true,
        broadening: RixsBroadening {
            gam_ch: 0.000_135_051_2,
            gam_exp_1: 0.000_135_051_2,
            gam_exp_2: 0.000_135_051_2,
        },
        energy_window: RixsEnergyWindow {
            emin_i: 0.0,
            emax_i: 0.0,
            emin_f: 0.0,
            emax_f: 0.0,
        },
        xmu: -367_493_090.027_428_2,
        switches: RixsSwitches {
            read_poles: true,
            skip_calc: false,
            mbconv: true,
            read_sigma: true,
        },
        edges: vec!["L3".to_string(), "VAL".to_string()],
    };
    std::fs::write(work_dir.join("rixs.inp"), rixs_input_string(&input)?)?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_band_global_input(1))?,
    )?;
    write_complete_rixs_full_run_source_handoff(work_dir)?;
    Ok(())
}

fn write_read_sigma_xsph_mpse_source_handoff(work_dir: &Path) -> Result<()> {
    let input = XsphInput {
        control: XsphControl {
            mphase: 0,
            ipr2: 0,
            ixc: 0,
            ixc0: 0,
            ispec: 0,
            lreal: 0,
            lfms2: 0,
            nph: 0,
            l2lp: 0,
            i_plsmn: 0,
            n_poles: 100,
            i_gamma_ch: 0,
            i_grid: 0,
            i_core_state: -1,
            iscfxc: 11,
        },
        vr0: 0.0,
        vi0: 0.0,
        lmaxph: vec![1],
        pot_labels: vec!["Cu".to_string()],
        grid: XsphGrid {
            rgrd: 0.05,
            rfms2: 0.0,
            gamach: 1.0,
            xkstep: 0.05,
            xkmax: 10.0,
            vixan: 0.0,
            eps0: 0.0,
            egap: 0.0,
        },
        spinph: vec![0.0],
        advanced: XsphAdvanced {
            izstd: 0,
            ifxc: 0,
            ipmbse: 0,
            itdlda: 0,
            nonlocal: 0,
            ibasis: 0,
        },
        electronic_temperature: 0.0,
        chsh_type: 0,
        decomposition_channels: -1,
        lopt: false,
        print_rl: false,
        source_format: XsphInputSourceFormat::modern(),
    };
    std::fs::write(work_dir.join("xsph.inp"), xsph_input_string(&input)?)?;
    write_phase_bin(
        work_dir.join("phase.bin"),
        &sample_read_sigma_mpse_phase_bin(),
    )?;
    write_pot_bin(work_dir.join("pot.bin"), &sample_xsph_source_pot_bin())?;
    Ok(())
}

fn sample_read_sigma_mpse_phase_bin() -> PhaseBinData {
    let mut phase = sample_phase_bin_data();
    phase.potentials.truncate(1);
    phase.fermi_index = 0;
    for energy in 0..phase.energy_count {
        phase.reference_energy[(energy, 0)].im = 0.03 + 0.005 * energy as f64;
    }
    phase
}

fn write_rixs_screen_handoff_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu RIXS SCREEN handoff run
EDGE L3 VAL
COREHOLE RPA
CONTROL 1 1 1 1 1 1
FMS 5.5
RIXS 0.1 0.1
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

/// Local scalar comparator (F6): delegates to the named [`Tol::PHASE_SHIFT`]
/// profile, which reproduces this module's original
/// `1.0e-10 * expected.abs().max(1.0)` formula exactly (see the `Tol` doc
/// comment) while gaining the shared max-abs/max-rel/RMS reporting.
fn assert_close(actual: f64, expected: f64) {
    Tol::PHASE_SHIFT.assert(actual, expected);
}

fn sample_rhorrp_core_density_pot_bin() -> PotBinData {
    let mut data = sample_pot_bin_data();
    for radial in 0..POT_BIN_RADIAL_POINTS {
        data.large_components[(radial, 0, 0)] = 1.0;
    }
    data
}

fn write_xsph_source_input(path: &Path) -> Result<()> {
    write_xsph_source_input_with_rlprint_flag(path, false)
}

fn write_nrixs_xsectjas_cached_xsph_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("xsph.inp"),
        concat!(
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc\n",
            "   1   0   0   0   1   0   0   1  30   0 100   0   0  -1  11\n",
            "vr0, vi0\n",
            "      0.00000      0.00000\n",
            " lmaxph(0:nph)\n",
            "   1   1\n",
            " potlbl(iph)\n",
            "Cu    O     \n",
            "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap\n",
            "      0.05000      0.00000      1.00000      0.05000     10.00000      0.00000      0.00000      0.00000\n",
            "spinph(0:nph)\n",
            "      0.00000      0.00000\n",
            "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis\n",
            "   0   0   0   0   0   0\n",
            "electronic temperature\n",
            "      0.00000\n",
            "ChSh_Type:\n",
            "   0\n",
            " the number of decomposition channels ; only used for nrixs\n",
            "   -1\n",
            "lopt\n",
            " F\n",
            "PrintRL\n",
            " F\n",
        ),
    )?;
    Ok(())
}

fn sample_nrixs_xsecl_dat_from_phase(phase: &PhaseBinData) -> Result<refeff_io::XseclDatData> {
    let fermi_index =
        usize::try_from(phase.fermi_index).context("sample phase fermi index is negative")?;
    let energy = Array1::from_iter(phase.energy_grid.iter().map(|energy| {
        (energy.re - phase.scalars.edge_energy + phase.scalars.fermi_level)
            * refeff_core::FEFF_HARTREE_EV
    }));
    let channel_cross_sections =
        Array2::from_shape_fn((phase.energy_count, 1), |(row, _channel)| {
            Complex64::new(0.01 * (row + 1) as f64, -0.005 * (row + 1) as f64)
        });
    let channel_sum =
        Array1::from_shape_fn(phase.energy_count, |row| channel_cross_sections[(row, 0)]);
    Ok(refeff_io::XseclDatData {
        header: refeff_io::XseclDatHeader {
            real_energy_count: phase.main_energy_count,
            fermi_index,
            edge: phase.scalars.edge_energy,
            emu: phase.scalars.fermi_level,
            core_hole_width: 0.1,
        },
        energy,
        channel_cross_sections,
        channel_sum,
    })
}

fn sample_nrixs_xsecl_bin_from_phase(phase: &PhaseBinData) -> refeff_io::XseclBinData {
    refeff_io::XseclBinData {
        pad_width: phase.pad_width,
        initial_state_j: 1,
        transitions: (0..phase.transition_count)
            .map(|transition| refeff_io::XseclBinTransition {
                final_state_kappa: if transition % 2 == 0 { -1 } else { 2 },
                decomposition_channel: transition as i32,
                total_angular_momentum_channel: transition as i32,
                orbital_angular_momentum: transition as i32,
            })
            .collect(),
        atom_cross_sections: Array2::from_shape_fn(
            (phase.energy_count, phase.final_state_count),
            |(row, state)| {
                Complex64::new(
                    0.02 * (row + 1) as f64 + 0.001 * state as f64,
                    -0.01 * (row + 1) as f64,
                )
            },
        ),
        raw_atom_cross_section_pad: None,
    }
}

fn write_self_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu SELF source run
SELF
MPSE 1 4
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

fn write_full_run_self_source_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("xsph.inp"),
        concat!(
            "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc\n",
            "   1   0   0   0   1   0   0   0   0   1   4   0   0  -1   0\n",
            "vr0, vi0\n",
            "      0.00000      0.00000\n",
            " lmaxph(0:nph)\n",
            "   3\n",
            " potlbl(iph)\n",
            "Cu    \n",
            "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap\n",
            "      0.05000      6.00000      1.72900      0.07000      8.00000      0.00000     12.00000      0.00000\n",
            "spinph(0:nph)\n",
            "      0.00000\n",
            "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis\n",
            "   0   0   0   0   0   0\n",
            "electronic temperature\n",
            "      0.00000\n",
            "ChSh_Type:\n",
            "   0\n",
            " the number of decomposition channels ; only used for nrixs\n",
            "   -1\n",
            "lopt\n",
            " F\n",
            "PrintRL\n",
            " F\n",
        ),
    )?;
    std::fs::write(
        work_dir.join("loss.dat"),
        concat!(
            "5.000000E+00 1.800000E-01\n",
            "1.200000E+01 4.500000E-01\n",
            "2.500000E+01 3.200000E-01\n",
            "6.000000E+01 2.000000E-01\n",
            "1.200000E+02 1.100000E-01\n",
            "2.500000E+02 5.000000E-02\n",
            "5.000000E+02 2.000000E-02\n",
        ),
    )?;
    Ok(())
}

fn write_xsph_hubbard_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu Hubbard XSPH source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
HUBBARD 4.0 0.5 0.0 2
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_phase_text_cached_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XSPH phase text cache run
CONTROL 0 1 0 0 0 0
PRINT 0 2 0 0 0 0
RPATH 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

fn write_xsph_emesh_phase_handoff_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XSPH emesh phase handoff run
CONTROL 0 1 0 0 0 0
RPATH 5.5
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

fn write_xsph_e2_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XSPH global multipole source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
POLARIZATION 1.0 0.0 0.0
ELLIPTICITY 0.25 0.0 1.0 0.0
MULTIPOLE 2 0
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 3
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_positive_izstd_pmbse_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu positive izstd PMBSE reset source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
TDLDA 0
PMBSE 3 2 0 6
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_tdlda_pmbse_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu active TDLDA PMBSE source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
PMBSE 2 0 5 0
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_tdlda_pmbse_file_basis_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu active TDLDA PMBSE file-basis source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
PMBSE 2 0 5 1
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_tdlda_pmbse_generated_basis_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu active TDLDA PMBSE generated-basis source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
PMBSE 2 0 5 2
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_two_spin_filtered_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu two-spin filtered XSPH source run
EDGE K
SPIN 1
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
MULTIPOLE 0 1
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1 0.01 1.0
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_fprime_phase_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu FPRIME XSPH phase handoff run
FPRIME -5.0 10.0
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
ICORE 1
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_xes_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XES XSPH source run
CONTROL 1 1 1 1 1 1
PRINT 0 1 0 0 0 0
COREHOLE RPA
XES 8.0 0.07 0.0
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_external_potential_no_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be external-potential POT source run
EDGE K
CONTROL 1 1 1 1 1 1
EXTPOT
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_restart_no_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be restart POT source run
EDGE K
CONTROL 1 1 1 1 1 1
RESTART
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_highz_no_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be finite-nucleus POT source run
EDGE K
CONTROL 1 1 1 1 1 1
HIGHZ
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_high_exchange_no_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be high-exchange no-SCF POT source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 6 0.0 0.0
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_external_restart_no_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be external restart POT source run
EDGE K
CONTROL 1 1 1 1 1 1
EXTPOT
RESTART
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_restart_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be restart iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
RESTART
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_external_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be external iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
EXTPOT
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_external_restart_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be external restart iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
EXTPOT
RESTART
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_highz_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be finite-nucleus iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
HIGHZ
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_high_exchange_iterative_scf_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Be high-exchange iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 5 0.0 0.0
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
    )?;
    Ok(())
}

fn write_disabled_pot_cached_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu disabled POT cache run
CONTROL 0 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    Ok(())
}

fn write_xsph_emesh_source_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu XSPH emesh handoff run
EDGE K
CONTROL 0 1 0 0 0 0
EXCHANGE 2 0.0 0.0
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;
    Ok(())
}

fn write_xsph_source_input_with_rlprint(path: &Path) -> Result<()> {
    write_xsph_source_input_with_rlprint_flag(path, true)
}

fn write_xsph_source_input_with_rlprint_flag(path: &Path, print_rl: bool) -> Result<()> {
    let rlprint = if print_rl { "RLPRINT\n" } else { "" };
    std::fs::write(
        path,
        format!(
            r#"
TITLE Cu XSPH source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 2 0.0 0.0
RSIGMA
ICORE 1
EGRID
user_grid
4.0 0.01
POTENTIALS
0 29 Cu 1 1
ATOMS
0.0 0.0 0.0 0 Cu
{rlprint}END
"#
        ),
    )?;
    Ok(())
}

fn write_full_run_split_pmbse_xmu_sources(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("listedges.pmbse"),
        "Oddp1\nEvenp1\nOddm1\nEvenm1\n",
    )?;
    write_full_run_pmbse_xmu_channel(
        work_dir,
        "Oddp1",
        &[100.0, 101.0, 102.0, 103.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
    )?;
    write_full_run_pmbse_xmu_channel(
        work_dir,
        "Evenp1",
        &[102.0, 102.5, 103.0, 103.5],
        &[0.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 3.0],
        &[9.0, 19.0, 29.0, 39.0],
    )?;
    write_full_run_pmbse_xmu_channel(
        work_dir,
        "Oddm1",
        &[100.0, 101.0, 102.0, 103.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0, 7.0],
    )?;
    write_full_run_pmbse_xmu_channel(
        work_dir,
        "Evenm1",
        &[102.0, 102.5, 103.0, 103.5],
        &[0.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 3.0],
        &[49.0, 59.0, 69.0, 79.0],
    )?;
    Ok(())
}

fn write_full_run_pmbse_xmu_channel(
    work_dir: &Path,
    channel_dir: &str,
    photon_energy_ev: &[f64],
    relative_energy_ev: &[f64],
    wave_number: &[f64],
    chi: &[f64],
) -> Result<()> {
    let channel_path = work_dir.join(channel_dir);
    std::fs::create_dir_all(&channel_path)?;
    let mu0 = Array1::from_elem(photon_energy_ev.len(), 1.0);
    let chi = Array1::from_vec(chi.to_vec());
    let mu = &mu0 + &chi;
    let data = XmuDatData {
        header_lines: vec!["# PMBSE xmu.dat full-run test channel".to_string()],
        normalization: None,
        photon_energy_ev: Array1::from_vec(photon_energy_ev.to_vec()),
        relative_energy_ev: Array1::from_vec(relative_energy_ev.to_vec()),
        wave_number: Array1::from_vec(wave_number.to_vec()),
        mu,
        mu0,
        chi,
    };
    write_xmu_dat(channel_path.join("xmu.dat"), &data)?;
    Ok(())
}

fn write_full_run_tdlda_file_basis_orbitals(work_dir: &Path) -> Result<()> {
    let orbital_dir = work_dir.join("Vila").join("Orbs");
    std::fs::create_dir_all(&orbital_dir)?;
    std::fs::write(
        orbital_dir.join("mg.3p.dat"),
        full_run_tdlda_file_basis_orbital_text(2.0),
    )?;
    std::fs::write(
        orbital_dir.join("mg.4p.dat"),
        full_run_tdlda_file_basis_orbital_text(3.0),
    )?;
    Ok(())
}

fn full_run_tdlda_file_basis_orbital_text(scale: f64) -> String {
    (1..=10)
        .map(|index| {
            let radius = 0.05 * index as f64;
            format!("{radius:.8} {scale:.8}\n")
        })
        .collect()
}

fn assert_full_run_scheduler_regenerates_stale_unsplit_tdlda_xsedge(
    input_writer: fn(&Path) -> Result<()>,
    write_file_basis_orbitals: bool,
    branch_label: &str,
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    input_writer(&input)?;
    execute_rdinp(&input, &output)?;
    let mut pot = sample_xsph_source_pot_bin();
    pot.ihole = 4;
    write_pot_bin(output.join("pot.bin"), &pot)?;
    write_config_dat(output.join("config.dat"), &sample_xsph_source_config_dat())?;
    write_full_run_split_pmbse_xmu_sources(&output)?;
    if write_file_basis_orbitals {
        write_full_run_tdlda_file_basis_orbitals(&output)?;
    }

    let first_reports = run_supported_cached_modules(&output)?;
    let first_report = first_reports
        .iter()
        .find(|report| report.name == "xsph")
        .with_context(|| {
            format!("missing initial completed {branch_label} TDLDA/PMBSE XSPH source report")
        })?;
    assert_eq!(first_report.unit, "file(s)");
    assert!(
        first_report.count >= 4,
        "{branch_label} TDLDA/PMBSE source report should include phase/xsedge sidecars: {first_reports:?}"
    );
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("emesh.dat").is_file());
    assert!(output.join("emesh.bin").is_file());
    assert!(output.join("xsedge.dat").is_file());
    assert!(!output.join("xsect.dat").is_file());

    let expected = read_xsedge_dat(output.join("xsedge.dat"))?;
    assert_eq!(expected.row_count(), 4);
    assert!(!expected.has_branch_columns());
    let mut stale = expected.clone();
    stale.total_single_particle[0] += 25.0;
    stale.total_screened[0] += 12.5;
    write_xsedge_dat(output.join("xsedge.dat"), &stale)?;
    assert_ne!(read_xsedge_dat(output.join("xsedge.dat"))?, expected);

    let second_reports = run_supported_cached_modules(&output)?;
    let second_report = second_reports
        .iter()
        .find(|report| report.name == "xsph")
        .with_context(|| {
            format!("missing regenerated {branch_label} TDLDA/PMBSE XSPH source report")
        })?;
    assert_eq!(second_report.unit, "file(s)");
    assert!(
        second_report.count >= 1,
        "{branch_label} TDLDA/PMBSE scheduler should regenerate stale xsedge.dat: {second_reports:?}"
    );
    assert_eq!(read_xsedge_dat(output.join("xsedge.dat"))?, expected);
    assert!(!output.join("xsect.dat").is_file());
    Ok(())
}

fn sample_xsph_source_pot_bin() -> PotBinData {
    let potentials = 1;
    let occupied_orbitals = 12;
    let mut large_components = Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    let mut small_components = Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    let mut large_coefficients =
        Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials));
    let mut small_coefficients =
        Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials));
    let mut orbital_occupancy = Array2::zeros((POT_BIN_ORBITALS, potentials));

    for orbital in 0..occupied_orbitals {
        orbital_occupancy[(orbital, 0)] = if orbital == 3 { 0.2 } else { 0.0 };
        for row in 0..POT_BIN_RADIAL_POINTS {
            let r = (row + 1) as f64;
            let o = (orbital + 1) as f64;
            large_components[(row, orbital, 0)] =
                0.012 * o * (0.035 * r * o).sin() * (-0.002 * r).exp();
            small_components[(row, orbital, 0)] =
                0.007 * o * (0.029 * r * o).cos() * (-0.0025 * r).exp();
        }
        for coefficient in 0..POT_BIN_COEFFICIENTS {
            let c = (coefficient + 1) as f64;
            let o = (orbital + 1) as f64;
            large_coefficients[(coefficient, orbital, 0)] =
                0.006 * c + 0.0009 * o * (0.17 * c * o).cos();
            small_coefficients[(coefficient, orbital, 0)] =
                -0.004 * c + 0.0006 * o * (0.13 * c * o).sin();
        }
    }

    PotBinData {
        titles: vec!["XSPH normal phase smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.5,
            fermi_level: 0.0,
            interstitial_potential: -0.45,
            interstitial_density: 0.06,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![184]),
        muffin_tin_radii: Array1::from_vec(vec![1.42]),
        norman_indices: Array1::from_vec(vec![220]),
        atomic_numbers: Array1::from_vec(vec![29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.0]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            let r = (row + 1) as f64;
            0.018 * (0.041 * r).sin() * (-0.002 * r).exp()
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            let r = (row + 1) as f64;
            0.006 * (0.033 * r).cos() * (-0.0025 * r).exp()
        }),
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        electron_density: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            0.055 + 0.000_01 * row as f64
        }),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            -0.52 + 0.000_4 * row as f64
        }),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy,
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_xsph_screened_source_pot_bin() -> PotBinData {
    let mut data = sample_xsph_source_pot_bin();
    data.nohole = 2;
    data
}

fn sample_xsph_positive_izstd_source_pot_bin() -> PotBinData {
    let mut data = sample_xsph_source_pot_bin();
    for orbital in 4..POT_BIN_ORBITALS {
        data.orbital_occupancy[(orbital, 0)] = 0.0;
        for row in 0..POT_BIN_RADIAL_POINTS {
            data.large_components[(row, orbital, 0)] = 0.0;
            data.small_components[(row, orbital, 0)] = 0.0;
        }
        for coefficient in 0..POT_BIN_COEFFICIENTS {
            data.large_coefficients[(coefficient, orbital, 0)] = 0.0;
            data.small_coefficients[(coefficient, orbital, 0)] = 0.0;
        }
    }
    data
}

fn sample_xsph_v_hubbard_bin() -> HubbardVnlmBinData {
    let angular_limit = 1;
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    let mut next = 0.05;
    let values = Array4::from_shape_fn((1, 2, angular_count, magnetic_count), |_| {
        let value = next;
        next += 0.05;
        value
    });
    HubbardVnlmBinData {
        angular_limit,
        values,
    }
}

fn sample_xsph_source_config_dat() -> ConfigDatData {
    let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    occupations[0] = 1.8;
    occupations[1] = 1.0;
    occupations[2] = 0.7;
    occupations[3] = 0.5;

    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![ConfigDatPotential {
            potential_index: 0,
            atomic_number: 29,
            element: "Cu".to_string(),
            occupations,
            valence_occupations: Array1::zeros(CONFIG_DAT_ORBITAL_COUNT),
            spin_occupations: None,
        }],
    }
}

#[test]
fn full_run_executes_cached_opcons_stage_before_atomic_geom_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_opcons_input(&input)?;
    std::fs::write(
        output.join("opconsCu.dat"),
        concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
    )?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should still require geom.dat after cached OPCONS")?;

    let message = format!("{error:#?}");
    assert!(message.contains("atomic-config=1 file(s)"), "{message}");
    assert!(message.contains("pot-input=1 file(s)"), "{message}");
    assert!(message.contains("opcons=3 row(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF atomic stage"),
        "{message}"
    );
    assert!(
        message.contains("ATOM source apot.bin generation requires geom.dat handoff"),
        "{message}"
    );
    assert!(output.join("loss.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_mpse_cu_opcons_reference_loss_from_source_tables() -> Result<()> {
    let Some(zip_path) = reference_opcons_zip()? else {
        require_fixture!("OPCONS full-run reference test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("OPCONS full-run reference test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for entry in ["opcons.inp", "opconsCu.dat", "pot.bin"] {
        std::fs::write(
            temp.path().join(entry),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
        )?;
    }
    let expected_loss = parse_loss_dat(&String::from_utf8(unzip_reference_entry(
        &zip_path,
        "REFERENCE/loss.dat",
    )?)?)?;

    assert!(!temp.path().join("loss.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "opcons" && report.count == expected_loss.point_count()),
        "complete MPSE/Cu_OPCONS source tables should report generated loss rows: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let actual_loss = parse_loss_dat(&std::fs::read_to_string(temp.path().join("loss.dat"))?)?;
    assert_loss_dat_reference_close(&actual_loss, &expected_loss);
    Ok(())
}

#[test]
fn full_run_scheduler_runs_opcons_before_mpse_xsph_phase_generation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = XsphInput {
        control: XsphControl {
            mphase: 1,
            ipr2: 0,
            ixc: 0,
            ixc0: 0,
            ispec: 1,
            lreal: 0,
            lfms2: 0,
            nph: 0,
            l2lp: 0,
            i_plsmn: 1,
            n_poles: 3,
            i_gamma_ch: 0,
            i_grid: 0,
            i_core_state: -1,
            iscfxc: 11,
        },
        vr0: 0.0,
        vi0: 0.0,
        lmaxph: vec![1],
        pot_labels: vec!["Cu".to_string()],
        grid: XsphGrid {
            rgrd: 0.05,
            rfms2: 0.0,
            gamach: 1.0,
            xkstep: 0.05,
            xkmax: 5.0,
            vixan: 0.0,
            eps0: 0.0,
            egap: 0.0,
        },
        spinph: vec![0.0],
        advanced: XsphAdvanced {
            izstd: 0,
            ifxc: 0,
            ipmbse: 0,
            itdlda: 0,
            nonlocal: 0,
            ibasis: 0,
        },
        electronic_temperature: 0.0,
        chsh_type: 0,
        decomposition_channels: -1,
        lopt: false,
        print_rl: false,
        source_format: XsphInputSourceFormat::modern(),
    };
    std::fs::write(temp.path().join("xsph.inp"), xsph_input_string(&input)?)?;
    std::fs::write(
        temp.path().join("opcons.inp"),
        "run_opcons\n T\nprint_eps\n F\nNumDens(0:nphx)\n  1.0000000000000000\n",
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_xsph_source_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_xsph_source_config_dat(),
    )?;

    assert!(!temp.path().join("loss.dat").exists());
    assert!(!temp.path().join("phase.bin").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    let opcons_position = reports
        .iter()
        .position(|report| report.name == "opcons")
        .context("missing OPCONS scheduler report")?;
    let xsph_position = reports
        .iter()
        .position(|report| report.name == "xsph")
        .context("missing completed MPSE XSPH scheduler report")?;
    assert!(
        opcons_position < xsph_position,
        "OPCONS loss generation must precede MPSE XSPH: {reports:?}"
    );
    assert!(temp.path().join("loss.dat").is_file());
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("xsect.dat").is_file());
    assert!(temp.path().join("mpse.dat").is_file());
    Ok(())
}

#[test]
fn full_run_generates_missing_opcons_epsdb_table_before_atomic_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_opcons_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOM should still require geom.dat after OPCONS epsdb source generation")?;

    let message = format!("{error:#?}");
    assert!(message.contains("atomic-config=1 file(s)"), "{message}");
    assert!(message.contains("pot-input=1 file(s)"), "{message}");
    assert!(message.contains("opcons=181 row(s)"), "{message}");
    assert!(
        message.contains("failed to run FEFF atomic stage"),
        "{message}"
    );
    assert!(
        message.contains("ATOM source apot.bin generation requires geom.dat handoff"),
        "{message}"
    );
    assert!(output.join("config.dat").is_file());
    assert!(output.join("opconsCu.dat").is_file());
    assert!(output.join("loss.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_opcons_table() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin_data())?;
    write_minimal_opcons_input(temp.path())?;
    std::fs::write(temp.path().join("opconsCu.dat"), b"not an opcons table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "opcons"),
        "malformed OPCONS source table should not report OPCONS complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_opcons_when_pot_source_is_malformed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_opcons_input(temp.path())?;
    std::fs::write(
        temp.path().join("opconsCu.dat"),
        concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
    )?;
    std::fs::write(temp.path().join("pot.bin"), b"not a pot.bin source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "opcons"),
        "malformed OPCONS pot.bin source should not report OPCONS complete: {:?}",
        reports
    );
    assert!(!temp.path().join("loss.dat").exists());
    assert!(!temp.path().join("epsilon.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_opcons_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("opcons.inp"),
        b"not an opcons.inp handoff\n",
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "opcons"),
        "malformed OPCONS input should not report OPCONS complete: {:?}",
        reports
    );
    assert!(!temp.path().join("loss.dat").exists());
    assert!(!temp.path().join("epsilon.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_opcons_table_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("opconsCu.dat"),
        concat!(" 1.0 1.0 0.5\n", " 2.0 2.0 1.0\n", " 3.0 3.0 1.5\n"),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "opcons"),
        "orphan opconsCu.dat table without opcons.inp should not report OPCONS complete: {:?}",
        reports
    );
    assert!(!temp.path().join("loss.dat").exists());
    assert!(!temp.path().join("epsilon.dat").exists());
    Ok(())
}

fn write_minimal_opcons_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("opcons.inp"),
        "run_opcons\n T\nprint_eps\n F\nNumDens(0:nphx)\n  1.0000000000000000\n",
    )?;
    Ok(())
}
