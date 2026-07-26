use super::*;
use ndarray::{Array2, Array4, Array5, Array6};
use num_complex::Complex32;
use refeff_io::feff_bin::FEFF_BIN_BOHR;
use refeff_io::pot_bin::POT_BIN_DEFAULT_PAD_WIDTH;
use refeff_io::{
    CONFIG_DAT_ORBITAL_COUNT, CfAverage, ConfigDatData, ConfigDatPotential, CrpaInput, CumDatData,
    CumDatEntry, Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput, FmsCluster, FmsControl,
    FmsDebye, FmsInput, FullSpectrumInput, GeomDat, GeomDatRow, GlobalControl, GlobalInput,
    GlobalNorms, GlobalQControl, HubbardInput, HubbardLdosGtrBinData, HubbardLdosGtrMBinData,
    HubbardLdosGtrOffBinData, KmeshMetadata, LdosMagneticDatData, MdffInput, PathsControl,
    PathsCriteria, PathsInput, PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp,
    PotRun, PotScattering, PotThermal, PotTolerances, RHORRP_POT_BIN_RADIAL_DX,
    RhorrpGgDiagBinData, RhorrpGgSliceBinData, config_dat_string, crpa_input_string,
    ff2x_input_string, fms_input_string, fullspectrum_input_string, geom_dat_string,
    global_input_string, hubbard_input_string, mdff_input_string, paths_input_string,
    pot_input_string, read_cum_dat, read_kmesh_dat, read_lmdos_dat, read_rhocm_dat,
    rhorrp_gg_diag_bin_bytes, rhorrp_gg_slice_bin_bytes, write_cum_dat, write_hubbard_ldos_gtr_bin,
    write_hubbard_ldos_gtr_m_bin, write_hubbard_ldos_gtr_off_bin, write_lmdos_dat, write_rhocm_dat,
};

#[test]
fn full_run_skips_compton_stage_when_jzzp_cache_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_compton_cached_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop when no supported COMPTON cache is available")?;

    assert!(
        error
            .to_string()
            .contains("no supported cached stages were run")
    );
    assert!(!output.join("compton.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_compton_jzzp_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_module_input(temp.path(), " T F F")?;
    std::fs::write(temp.path().join("jzzp.dat"), b"not a jzzp.dat cache\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "compton"),
        "malformed standalone COMPTON jzzp.dat should not report COMPTON complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_compton_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("compton.inp"),
        b"not a compton.inp handoff\n",
    )?;
    write_jzzp_dat(temp.path().join("jzzp.dat"), &sample_jzzp_data())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "compton"),
        "malformed compton.inp should not report COMPTON complete: {:?}",
        reports
    );
    assert!(!temp.path().join("compton.dat").exists());
    assert!(!temp.path().join("logcompton.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_compton_when_rhorrp_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_module_input(temp.path(), " T F F")?;
    let expected = sample_jzzp_data();
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected)?;
    write_full_run_compton_rhorrp_callback_handoffs(temp.path())?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "compton"),
        "cached COMPTON output with malformed RHORRP source should not report COMPTON complete: {:?}",
        reports
    );
    assert_eq!(read_jzzp_dat(temp.path().join("jzzp.dat"))?, expected);
    assert!(!temp.path().join("compton.dat").exists());
    assert!(!temp.path().join("logcompton.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_compton_outputs_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected_jzzp = sample_jzzp_data();
    let expected_rhozzp = sample_rhozzp_data();
    write_jzzp_dat(temp.path().join("jzzp.dat"), &expected_jzzp)?;
    write_rhozzp_dat(temp.path().join("rhozzp.dat"), &expected_rhozzp)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "compton"),
        "orphan COMPTON outputs without compton.inp should not report COMPTON complete: {:?}",
        reports
    );
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
fn full_run_scheduler_does_not_report_malformed_band_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("band.inp"), b"not a band.inp handoff\n")?;
    let bandstructure = sample_bandstructure_dat();
    write_bandstructure_dat(temp.path().join("bandstructure.dat"), &bandstructure)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "band" && report.name != "band-handoff"),
        "malformed band.inp should not report BAND complete: {:?}",
        reports
    );
    assert_eq!(
        read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
        bandstructure
    );
    assert!(!temp.path().join("logband.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_compton_rhozzp_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_compton_module_input(temp.path(), " F T F")?;
    std::fs::write(temp.path().join("rhozzp.dat"), b"not a rhozzp.dat cache\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "compton"),
        "malformed standalone COMPTON rhozzp.dat should not report COMPTON complete: {:?}",
        reports
    );
    Ok(())
}

fn write_minimal_compton_module_input(work_dir: &Path, switches: &str) -> Result<()> {
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
            switches
        ),
    )?;
    Ok(())
}

#[test]
fn full_run_executes_cached_compton_stage_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_compton_cached_input(&input)?;
    write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop after cached COMPTON boundary")?;
    let message = error.to_string();
    assert!(message.contains("compton=3 row(s)"), "{message}");
    assert_eq!(
        read_compton_dat(output.join("compton.dat"))?.point_count(),
        3
    );
    assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
    let log = read_module_log_dat(output.join("logcompton.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating Compton profile"))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Reusing previously calculated j(z,z')"))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: Compton scattering."))
    );
    Ok(())
}

#[test]
fn full_run_preserves_cached_compton_rhozzp_stage_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_compton_rhozzp_cached_input(&input)?;
    write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;
    write_rhozzp_dat(output.join("rhozzp.dat"), &sample_rhozzp_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop after cached COMPTON/RHOZZP boundary")?;
    let message = error.to_string();
    assert!(message.contains("compton=6 row(s)"), "{message}");
    assert_eq!(
        read_compton_dat(output.join("compton.dat"))?.point_count(),
        3
    );
    assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
    assert_eq!(
        read_rhozzp_dat(output.join("rhozzp.dat"))?,
        sample_rhozzp_data()
    );
    let log = read_module_log_dat(output.join("logcompton.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating rho(z,z')"))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: Compton scattering."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_compton_outputs_from_rhorrp_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_full_run_compton_rhorrp_input(&input)?;
    execute_rdinp(&input, &output)?;
    write_full_run_compton_rhorrp_callback_handoffs(&output)?;
    assert!(crate::rhorrp::has_rhorrp_density_callback_source(&output));

    let first_reports = run_supported_cached_modules(&output)?;
    assert!(
        first_reports
            .iter()
            .any(|report| report.name == "compton" && report.count == 1003),
        "{:?}",
        first_reports
    );
    let expected_jzzp = read_jzzp_dat(output.join("jzzp.dat"))?;
    let expected_rhozzp = read_rhozzp_dat(output.join("rhozzp.dat"))?;

    let mut stale_jzzp = expected_jzzp.clone();
    stale_jzzp.values[[0, 0]] += 0.125;
    write_jzzp_dat(output.join("jzzp.dat"), &stale_jzzp)?;
    let mut stale_rhozzp = expected_rhozzp.clone();
    stale_rhozzp.density[0] += 0.125;
    write_rhozzp_dat(output.join("rhozzp.dat"), &stale_rhozzp)?;
    std::fs::remove_file(output.join("logcompton.dat"))?;

    let second_reports = run_supported_cached_modules(&output)?;
    assert!(
        second_reports
            .iter()
            .any(|report| report.name == "compton" && report.count == 1003),
        "{:?}",
        second_reports
    );
    assert_ne!(read_jzzp_dat(output.join("jzzp.dat"))?, stale_jzzp);
    assert_ne!(read_rhozzp_dat(output.join("rhozzp.dat"))?, stale_rhozzp);
    assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, expected_jzzp);
    assert_eq!(read_rhozzp_dat(output.join("rhozzp.dat"))?, expected_rhozzp);
    let log = read_module_log_dat(output.join("logcompton.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating rho(z,z')"))
    );
    assert!(log.lines.iter().any(|line| line.contains("Saving j(z,z')")));
    Ok(())
}

#[test]
fn full_run_executes_cached_crpa_stage_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_crpa_dat(output.join("crpa.dat"), &sample_crpa_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop after cached CRPA boundary")?;

    let message = error.to_string();
    assert!(message.contains("crpa=2 row(s)"), "{message}");
    assert_eq!(read_crpa_dat(output.join("crpa.dat"))?, sample_crpa_dat());
    let log = read_module_log_dat(output.join("logscrn.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains(" Calculating Hubbard U."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains(" Done with Hubbard U calculation."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_crpa_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("crpa.inp"), b"not a crpa.inp handoff\n")?;
    write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "crpa" | "crpa-wscrn")),
        "malformed crpa.inp should not report CRPA complete: {:?}",
        reports
    );
    assert!(!temp.path().join("logscrn.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_crpa_when_screen_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("crpa.inp"),
        crpa_input_string(&CrpaInput {
            enabled: true,
            rcut: 3.5,
            l: 2,
        })?,
    )?;
    write_crpa_dat(temp.path().join("crpa.dat"), &sample_crpa_dat())?;
    std::fs::write(
        temp.path().join("screen.inp"),
        b"not a screen.inp handoff\n",
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "crpa" | "crpa-wscrn")),
        "malformed screen.inp source should not report CRPA complete: {:?}",
        reports
    );
    assert_eq!(
        read_crpa_dat(temp.path().join("crpa.dat"))?,
        sample_crpa_dat()
    );
    assert!(!temp.path().join("wscrn.dat").exists());
    assert!(!temp.path().join("logscrn.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_crpa_wscrn_from_vtot_and_apot_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_crpa_dat(output.join("crpa.dat"), &sample_crpa_dat())?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    std::fs::write(output.join("wscrn.dat"), "not a wscrn.dat table\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop after CRPA recovery boundary")?;

    let message = error.to_string();
    assert!(message.contains("crpa=5 row(s)"), "{message}");
    assert_eq!(read_crpa_dat(output.join("crpa.dat"))?, sample_crpa_dat());

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );

    let log = read_module_log_dat(output.join("logscrn.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains(" Calculating Hubbard U."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains(" Done with Hubbard U calculation."))
    );
    Ok(())
}

#[test]
fn full_run_generates_crpa_wscrn_handoff_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("CRPA wscrn handoff should still require downstream source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("crpa-wscrn=3 row(s)"), "{message}");
    assert!(!output.join("crpa.dat").exists());
    assert!(!output.join("logscrn.dat").exists());
    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    Ok(())
}

#[test]
fn full_run_generates_crpa_wscrn_handoff_for_malformed_crpa_cache_before_source_requirement_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    std::fs::write(output.join("crpa.dat"), "not a crpa.dat table\n")?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("CRPA wscrn handoff should still require downstream source state")?;

    let message = format!("{error:#?}");
    assert!(message.contains("crpa-wscrn=3 row(s)"), "{message}");
    assert!(!message.contains("crpa.dat"), "{message}");
    assert!(!output.join("logscrn.dat").exists());
    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    Ok(())
}

#[test]
fn full_run_scheduler_runs_crpa_source_output_from_reference_handoffs() -> Result<()> {
    let Some(zip_path) = reference_crpa_zip()? else {
        require_fixture!("CRPA full-run scheduler source test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("CRPA full-run scheduler source test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for entry in [
        "band.inp",
        "crpa.inp",
        "pot.bin",
        "config.dat",
        "geom.dat",
        "fms.inp",
    ] {
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

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "crpa" && report.count > 1),
        "missing CRPA source output report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(temp.path().join("crpa.dat").is_file());
    assert!(temp.path().join("wscrn.dat").is_file());
    assert!(temp.path().join("logscrn.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_crpa_screen_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("crpa.inp"),
        concat!(
            " do_CRPA           1\n",
            " rcut   3.5000000000000000     \n",
            " l_crpa           2\n",
        ),
    )?;
    std::fs::write(temp.path().join("screen.inp"), "not a screen.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| !matches!(
            report.name,
            "screen" | "screen-wscrn" | "crpa" | "crpa-wscrn"
        )),
        "malformed CRPA SCREEN source handoff should not report SCREEN/CRPA completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("crpa.dat").exists());
    assert!(!temp.path().join("wscrn.dat").exists());
    assert!(!temp.path().join("logscrn.dat").exists());
    assert!(!temp.path().join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_runs_screen_source_output_from_reference_handoffs() -> Result<()> {
    let Some(zip_path) = reference_crpa_zip()? else {
        require_fixture!("SCREEN full-run scheduler source test; reference zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        require_fixture!("SCREEN full-run scheduler source test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    for entry in ["pot.bin", "config.dat", "geom.dat", "fms.inp"] {
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

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "screen" && report.count > 1),
        "missing SCREEN source output report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(temp.path().join("wscrn.dat").is_file());
    assert!(temp.path().join("vtot.dat").is_file());
    assert!(temp.path().join("logscreen.dat").is_file());
    Ok(())
}

#[test]
fn full_run_does_not_advertise_malformed_crpa_log_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_crpa_dat(output.join("crpa.dat"), &sample_crpa_dat())?;
    std::fs::write(output.join("logscrn.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop without advertising malformed CRPA log")?;

    let message = error.to_string();
    assert!(!message.contains("crpa="), "{message}");
    Ok(())
}

#[test]
fn full_run_does_not_treat_crpa_wscrn_sidecar_as_screen_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("CRPA should still require crpa.dat or a complete source assembly")?;

    let message = error.to_string();
    assert!(
        message.contains("no supported cached stages were run"),
        "{message}"
    );
    assert!(!message.contains("screen="), "{message}");
    assert!(!output.join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn full_run_executes_cached_screen_stage_before_xsph_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH source handoff should still require phase sources after SCREEN")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen=6 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );
    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, vtot);
    assert_eq!(
        read_module_log_dat(output.join("logscreen.dat"))?,
        sample_screen_module_log()
    );
    Ok(())
}

#[test]
fn full_run_does_not_advertise_malformed_screen_log_before_xsph_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    std::fs::write(output.join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN log rejection",
    )?;

    let message = error.to_string();
    assert!(!message.contains("screen="), "{message}");
    Ok(())
}

#[test]
fn full_run_recovers_screen_wscrn_from_vtot_and_apot_before_xsph_source_requirement() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN wscrn recovery",
    )?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen-wscrn=3 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, vtot);
    assert!(!output.join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_screen_log_from_vtot_and_apot_before_xsph_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    std::fs::write(output.join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN wscrn recovery",
    )?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen-wscrn=3 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, vtot);
    assert!(read_module_log_dat(output.join("logscreen.dat")).is_err());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_screen_wscrn_from_vtot_and_apot_before_xsph_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    std::fs::write(output.join("wscrn.dat"), "not a wscrn.dat table\n")?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN wscrn recovery",
    )?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen-wscrn=3 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, vtot);
    Ok(())
}

#[test]
fn full_run_recovers_stale_screen_wscrn_from_vtot_and_apot_before_xsph_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
    let mut stale = sample_wscrn_dat();
    stale.screened_potential[0] += 1.0;
    write_wscrn_dat(output.join("wscrn.dat"), &stale)?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN wscrn recovery",
    )?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen-wscrn=3 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = sample_vtot_dat();
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, vtot);
    Ok(())
}

#[test]
fn full_run_recovers_malformed_screen_vtot_from_wscrn_and_pot_before_xsph_source_requirement()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
    std::fs::write(output.join("vtot.dat"), "not a vtot.dat table\n")?;
    std::fs::write(output.join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output).err().context(
        "XSPH source handoff should still require phase sources after SCREEN vtot recovery",
    )?;

    let chain = format!("{error:#}");
    assert!(chain.contains("screen=6 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = sample_wscrn_dat();
    let vtot = read_vtot_dat(output.join("vtot.dat"))?;
    assert_eq!(vtot.radius_bohr, wscrn.radius_bohr);
    assert_eq!(vtot.screened_core_hole_potential, wscrn.screened_potential);
    assert_eq!(vtot.total_potential.len(), vtot.radius_bohr.len());
    assert!(vtot.total_potential.iter().all(|value| value.is_finite()));
    assert!(
        vtot.total_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert_eq!(
        read_module_log_dat(output.join("logscreen.dat"))?,
        sample_screen_module_log()
    );
    Ok(())
}

#[test]
fn full_run_prefers_recoverable_screen_wscrn_handoff_before_xsph_source_requirement() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
    let mut source_vtot = sample_vtot_dat();
    source_vtot.screened_core_hole_potential[0] += 1.0;
    write_vtot_dat(output.join("vtot.dat"), &source_vtot)?;
    std::fs::write(output.join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("XSPH source handoff should still require phase sources after SCREEN recovery")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("screen-wscrn=3 row(s)"), "{chain}");
    assert!(
        chain.contains("XSPH phase generation requires cached phase.bin"),
        "{chain}"
    );

    let wscrn = read_wscrn_dat(output.join("wscrn.dat"))?;
    let vtot = read_vtot_dat(output.join("vtot.dat"))?;
    assert_eq!(vtot.radius_bohr, source_vtot.radius_bohr);
    assert_eq!(
        vtot.screened_core_hole_potential,
        source_vtot.screened_core_hole_potential
    );
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert_eq!(vtot.total_potential.len(), vtot.radius_bohr.len());
    assert!(vtot.total_potential.iter().all(|value| value.is_finite()));
    assert!(
        vtot.total_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    assert!(read_module_log_dat(output.join("logscreen.dat")).is_err());
    Ok(())
}

#[test]
fn full_run_completes_from_cached_ldos_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    write_ldos_dat(output.join("ldos00.dat"), &sample_ldos_dat()?)?;

    run_feff_to_dir(&input, &output)?;

    let ldos = read_ldos_dat(output.join("ldos00.dat"))?;
    assert!(!ldos.energy_ev.is_empty());
    assert_eq!(ldos.density.nrows(), ldos.energy_ev.len());
    assert!(ldos.density.ncols() >= 4);
    assert!(ldos.energy_ev.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().any(|value| value.abs() > 1.0e-12));
    let rhoc = read_rhoc_dat(output.join("rhoc00.dat"))?;
    assert_eq!(rhoc.energy_ev.len(), ldos.energy_ev.len());
    assert_eq!(rhoc.density.dim(), ldos.density.dim());
    assert!(rhoc.density.iter().all(|value| value.is_finite()));
    assert!(rhoc.density.iter().any(|value| value.abs() > 1.0e-12));
    assert_eq!(
        read_module_log_dat(output.join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_ldos_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat()?)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "orphan ldosNN.dat cache without ldos.inp should not report LDOS complete: {:?}",
        reports
    );
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

fn assert_full_run_ldos_rhoc_pair(output: &Path) -> Result<()> {
    let ldos = read_ldos_dat(output.join("ldos00.dat"))?;
    let rhoc = read_rhoc_dat(output.join("rhoc00.dat"))?;

    assert!(!ldos.energy_ev.is_empty());
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density.dim(), rhoc.density.dim());
    assert_eq!(ldos.density.nrows(), ldos.energy_ev.len());
    assert!(ldos.density.ncols() >= 4);
    assert!(ldos.energy_ev.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().all(|value| value.is_finite()));
    assert!(rhoc.density.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().any(|value| value.abs() > 1.0e-12));
    assert!(rhoc.density.iter().any(|value| value.abs() > 1.0e-12));
    Ok(())
}

#[test]
fn full_run_recovers_malformed_rhoc_from_ldos() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    let ldos = sample_ldos_dat()?;
    write_ldos_dat(output.join("ldos00.dat"), &ldos)?;
    std::fs::write(output.join("rhoc00.dat"), "not an rhoc table\n")?;

    run_feff_to_dir(&input, &output)?;

    assert_full_run_ldos_rhoc_pair(&output)?;
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
fn full_run_generates_missing_ldos_from_rhoc() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    let mut rhoc = sample_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(output.join("rhoc00.dat"), &rhoc)?;

    run_feff_to_dir(&input, &output)?;

    assert_full_run_ldos_rhoc_pair(&output)?;
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
fn full_run_recovers_malformed_ldos_from_rhoc() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    let mut rhoc = sample_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(output.join("rhoc00.dat"), &rhoc)?;
    std::fs::write(output.join("ldos00.dat"), "not an ldos table\n")?;

    run_feff_to_dir(&input, &output)?;

    let ldos = read_ldos_dat(output.join("ldos00.dat"))?;
    assert_full_run_ldos_rhoc_pair(&output)?;
    assert!(ldos.header_lines.iter().any(|line| line.contains("sDOS")));
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_ldos_log_from_rhoc() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    let mut rhoc = sample_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(output.join("rhoc00.dat"), &rhoc)?;
    std::fs::write(output.join("ldos00.dat"), "not an ldos table\n")?;
    std::fs::write(output.join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    run_feff_to_dir(&input, &output)?;

    assert_full_run_ldos_rhoc_pair(&output)?;
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
fn full_run_generates_ordinary_spin_no_fms_ldos_from_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_spin_ldos_cached_input(&input)?;
    let mut rhoc = sample_spin_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(output.join("rhoc00.dat"), &rhoc)?;

    run_feff_to_dir(&input, &output)?;

    let ldos = read_ldos_dat(output.join("ldos00.dat"))?;
    let rhoc = read_rhoc_dat(output.join("rhoc00.dat"))?;
    assert!(!ldos.is_spin_resolved());
    assert!(!rhoc.is_spin_resolved());
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density.dim(), (3, 4));
    assert_eq!(rhoc.density.dim(), (3, 4));
    assert!(ldos.density.iter().all(|value| value.is_finite()));
    assert!(rhoc.density.iter().all(|value| value.is_finite()));
    assert!(ldos.density.iter().any(|value| value.abs() > 1.0e-12));
    assert!(rhoc.density.iter().any(|value| value.abs() > 1.0e-12));
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_recovers_malformed_spin_ldos_from_rhoc_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_spin_ldos_module_input(temp.path())?;
    let mut rhoc = sample_spin_ldos_dat()?;
    rhoc.header_lines.clear();
    rhoc.fermi_level_ev = None;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not a spin ldos table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("recoverable spin LDOS cache should report LDOS")?;
    assert_eq!(ldos_report.count, 2);
    assert_eq!(ldos_report.unit, "file(s)");
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert!(ldos.is_spin_resolved());
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert!(
        ldos.header_lines
            .iter()
            .any(|line| line.contains("sDOS(up)") && line.contains("sDOS(down)"))
    );
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_generates_missing_spin_rhoc_from_ldos_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_spin_ldos_module_input(temp.path())?;
    let ldos = sample_spin_ldos_dat()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("recoverable spin RHOC cache should report LDOS")?;
    assert_eq!(ldos_report.count, 2);
    assert_eq!(ldos_report.unit, "file(s)");
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert!(rhoc.is_spin_resolved());
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_recovers_malformed_spin_rhoc_from_ldos_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_spin_ldos_module_input(temp.path())?;
    let ldos = sample_spin_ldos_dat()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not a spin rhoc table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("recoverable spin RHOC cache should report LDOS")?;
    assert_eq!(ldos_report.count, 2);
    assert_eq!(ldos_report.unit, "file(s)");
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert!(rhoc.is_spin_resolved());
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_generates_ldos_kmesh_from_reciprocal_handoff_before_atomic_source_overlap_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_reciprocal_ldos_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOMIC source generation should still validate reciprocal LDOS geometry")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("ldos-kmesh=1 file(s)"), "{chain}");
    assert!(
        chain.contains("failed to generate ATOM apot.bin from pot.inp/geom.dat source handoffs"),
        "{chain}"
    );
    assert!(
        chain.contains("radius must be positive and finite"),
        "{chain}"
    );
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
    assert!(!output.join("ldos00.dat").exists());
    assert!(!output.join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ldos_reciprocal_kmesh_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ldos_module_input(temp.path())?;
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        "not a reciprocal.inp handoff\n",
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "ldos" | "ldos-kmesh" | "kmesh")),
        "malformed LDOS reciprocal handoff should not report LDOS/KSPACE completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("kmesh.dat").exists());
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ldos_wavefunction_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ldos_module_input(temp.path())?;
    write_pot_bin(
        temp.path().join("pot.bin"),
        &sample_full_run_compton_callback_pot_bin(),
    )?;
    std::fs::write(
        temp.path().join("config.dat"),
        config_dat_string(&sample_full_run_compton_callback_config_dat())?,
    )?;
    std::fs::write(
        temp.path().join("pot.inp"),
        pot_input_string(&sample_full_run_compton_callback_pot_input())?,
    )?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "malformed LDOS wavefunction source handoff should not report LDOS completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("rhoc00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_ldos_when_wavefunction_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ldos_module_input(temp.path())?;
    write_pot_bin(
        temp.path().join("pot.bin"),
        &sample_full_run_compton_callback_pot_bin(),
    )?;
    std::fs::write(
        temp.path().join("config.dat"),
        config_dat_string(&sample_full_run_compton_callback_config_dat())?,
    )?;
    std::fs::write(
        temp.path().join("pot.inp"),
        pot_input_string(&sample_full_run_compton_callback_pot_input())?,
    )?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;
    let ldos = sample_ldos_dat()?;
    let rhoc = sample_rhoc_dat()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| report.name != "ldos" && report.name != "ldos-kmesh"),
        "malformed LDOS wavefunction source handoff should not report cached LDOS completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_ldos_log_for_kmesh_handoff_before_atomic_source_overlap_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_ldos_input(&input)?;
    std::fs::write(output.join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOMIC source generation should still validate reciprocal LDOS geometry")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("ldos-kmesh=2 file(s)"), "{chain}");
    assert!(
        chain.contains("failed to generate ATOM apot.bin from pot.inp/geom.dat source handoffs"),
        "{chain}"
    );
    assert!(
        chain.contains("radius must be positive and finite"),
        "{chain}"
    );
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert_eq!(
        read_module_log_dat(output.join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    assert!(!output.join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_ldos_kmesh_for_malformed_ldos_cache_before_atomic_source_overlap_error()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_reciprocal_ldos_input(&input)?;
    std::fs::write(output.join("ldos00.dat"), "not an ldos table\n")?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("ATOMIC source generation should still validate reciprocal LDOS geometry")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("ldos-kmesh=1 file(s)"), "{chain}");
    assert!(
        chain.contains("failed to generate ATOM apot.bin from pot.inp/geom.dat source handoffs"),
        "{chain}"
    );
    assert!(
        chain.contains("radius must be positive and finite"),
        "{chain}"
    );
    assert!(!chain.contains("ldos00.dat"), "{chain}");
    let kmesh = read_kmesh_dat(output.join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert!(!output.join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_reports_active_hubbard_ldos_with_matching_source_contracts() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_active_hubbard_gtr_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_active_hubbard_gtr_m_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off00.bin"),
        &sample_active_hubbard_gtr_off_source_contract(1, 1, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("complete active-Hubbard LDOS source/cache contract should report LDOS")?;
    assert_eq!(ldos_report.count, 4);
    assert_eq!(ldos_report.unit, "file(s)");
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        sample_rhoc_dat()?
    );
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_repairs_active_hubbard_ldos_ordinary_pair_from_valid_rhoc() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input_with_lfms2(temp.path(), 0)?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("recoverable active-Hubbard LDOS ordinary pair should report LDOS")?;
    assert_eq!(ldos_report.count, 4);
    assert_eq!(ldos_report.unit, "file(s)");
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    let expected_rhoc = sample_rhoc_dat()?;
    assert_eq!(ldos.energy_ev, expected_rhoc.energy_ev);
    assert_eq!(ldos.density, expected_rhoc.density);
    assert_eq!(rhoc, expected_rhoc);
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_repairs_active_hubbard_ldos_rhoc_ordinary_pair_from_valid_ldos() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input_with_lfms2(temp.path(), 0)?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not a rhoc table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("recoverable active-Hubbard RHOC ordinary pair should report LDOS")?;
    assert_eq!(ldos_report.count, 4);
    assert_eq!(ldos_report.unit, "file(s)");
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    let expected_ldos = sample_ldos_dat()?;
    assert_eq!(ldos, expected_ldos);
    assert_eq!(rhoc.energy_ev, expected_ldos.energy_ev);
    assert_eq!(rhoc.density, expected_ldos.density);
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_ldos_energy_grid_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut ldos = sample_ldos_dat()?;
    ldos.energy_ev[1] += 0.25;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale ordinary ldos energy grid should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        sample_rhoc_dat()?
    );
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_rhoc_energy_grid_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut rhoc = sample_rhoc_dat()?;
    rhoc.energy_ev[1] += 0.25;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale ordinary rhoc energy grid should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_ordinary_density_layout_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut rhoc = sample_rhoc_dat()?;
    rhoc.density = Array2::from_shape_vec(
        (3, 6),
        vec![
            5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4, 9.0E-4, 10.0E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4,
            9.1E-4, 10.1E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4, 9.2E-4, 10.2E-4,
        ],
    )?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale ordinary density layout should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_lmdos_energy_grid_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut lmdos = sample_active_hubbard_lmdos_dat()?;
    lmdos.energy_ev[1] += 0.25;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale magnetic energy grid should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        sample_active_hubbard_rhocm_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_rhocm_energy_grid_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut rhocm = sample_active_hubbard_rhocm_dat()?;
    rhocm.energy_ev[0] -= 0.25;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale rhocm energy grid should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_magnetic_layout_is_stale()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    let mut rhocm = sample_active_hubbard_rhocm_dat()?;
    rhocm.angular_limit = 0;
    rhocm.density =
        Array2::from_shape_vec((3, 2), vec![9.0e-4, 8.0e-4, 9.1e-4, 8.1e-4, 9.2e-4, 8.2e-4])?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with stale magnetic layout should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
    Ok(())
}

#[test]
fn full_run_scheduler_rejects_malformed_active_hubbard_magnetic_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    std::fs::write(
        temp.path().join("lmdos00.dat"),
        "not a magnetic ldos table\n",
    )?;

    let error = run_supported_cached_modules(temp.path())
        .err()
        .context("malformed active-Hubbard magnetic sidecar should fail scheduler validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("lmdos00.dat"), "{chain}");
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_rejects_malformed_active_hubbard_rhocm_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    std::fs::write(
        temp.path().join("rhocm00.dat"),
        "not a magnetic rhoc table\n",
    )?;

    let error = run_supported_cached_modules(temp.path())
        .err()
        .context("malformed active-Hubbard rhocm sidecar should fail scheduler validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("rhocm00.dat"), "{chain}");
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_ignores_non_hubbard_magnetic_sidecar_files() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ldos_module_input(temp.path())?;
    let ldos = sample_ldos_dat()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(
        temp.path().join("lmdos00.dat"),
        "not a magnetic ldos table\n",
    )?;
    std::fs::write(
        temp.path().join("rhocm00.dat"),
        "not a magnetic rhoc table\n",
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    let ldos_report = reports
        .iter()
        .find(|report| report.name == "ldos")
        .context("ordinary non-Hubbard LDOS cache should ignore stray magnetic sidecars")?;
    assert_eq!(ldos_report.count, 2);
    assert_eq!(ldos_report.unit, "file(s)");
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("lmdos00.dat"))?,
        "not a magnetic ldos table\n"
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("rhocm00.dat"))?,
        "not a magnetic rhoc table\n"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_omits_cached_potential()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr01.bin"),
        &sample_active_hubbard_gtr_source_contract(1, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with incomplete gtr source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos01.dat"))?,
        sample_ldos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_layout_conflicts() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_active_hubbard_gtr_source_contract(2, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with conflicting gtr source layout should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_and_gtr_m_layouts_conflict()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;

    let mut lmdos = sample_active_hubbard_lmdos_dat()?;
    lmdos.angular_limit = 0;
    lmdos.density =
        Array2::from_shape_vec((3, 2), vec![1.0e-4, 2.0e-4, 1.1e-4, 2.1e-4, 1.2e-4, 2.2e-4])?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;

    let mut rhocm = sample_active_hubbard_rhocm_dat()?;
    rhocm.angular_limit = 0;
    rhocm.density =
        Array2::from_shape_vec((3, 2), vec![9.0e-4, 8.0e-4, 9.1e-4, 8.1e-4, 9.2e-4, 8.2e-4])?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_active_hubbard_gtr_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_active_hubbard_gtr_m_source_contract(0, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with conflicting gtr/gtr_m source layouts should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
    assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_m_omits_cached_potential()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m01.bin"),
        &sample_active_hubbard_gtr_m_source_contract(1, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with incomplete gtr_m source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos01.dat"))?,
        sample_ldos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_m_layout_conflicts() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_active_hubbard_gtr_m_source_contract(0, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with conflicting gtr_m source layout should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_off_omits_cached_potential()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off01.bin"),
        &sample_active_hubbard_gtr_off_source_contract(1, 1, 3, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with incomplete gtr_off source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos01.dat"))?,
        sample_ldos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_active_hubbard_ldos_when_gtr_off_layout_conflicts()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_ldos_cached_input(temp.path())?;
    write_active_hubbard_cached_tables(temp.path(), "00")?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off00.bin"),
        &sample_active_hubbard_gtr_off_source_contract(1, 1, 2, 1),
    )?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ldos"),
        "active-Hubbard LDOS cache with conflicting gtr_off source layout should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("logdos.dat").exists());
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        sample_active_hubbard_lmdos_dat()?
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_eels_stage_before_ff2x_polarization_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eels_cached_input(&input)?;
    write_eels_dat(output.join("eels.dat"), &sample_eels_dat())?;
    write_module_log_dat(output.join("logeels.dat"), &sample_eels_module_log())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("FF2X should still require polarization sources after EELS")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("eels=3 row(s)"), "{chain}");
    assert!(
        chain.contains("FF2X polarization 2 generation requires"),
        "{chain}"
    );
    assert_eq!(read_eels_dat(output.join("eels.dat"))?, sample_eels_dat());
    assert_eq!(
        read_module_log_dat(output.join("logeels.dat"))?,
        sample_eels_module_log()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_eels_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("eels.inp"), b"not an eels.inp handoff\n")?;
    write_eels_dat(temp.path().join("eels.dat"), &sample_eels_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eels"),
        "malformed eels.inp should not report EELS complete: {:?}",
        reports
    );
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_eels_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_eels_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eels"),
        "orphan eels.dat cache without eels.inp should not report EELS complete: {:?}",
        reports
    );
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_eels_from_xmu_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eels_cached_input(&input)?;
    write_full_run_eels_xmu_sources(&output)?;
    std::fs::write(output.join("eels.dat"), b"not an eels.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;
    let eels = read_eels_dat(output.join("eels.dat"))?;
    assert_eq!(eels.point_count(), sample_xmu_dat().point_count());
    assert!(eels.has_tensor());
    assert!(output.join("logeels.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_stale_eels_from_xmu_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eels_cached_input(&input)?;
    write_full_run_eels_xmu_sources(&output)?;
    let stale = sample_eels_dat();
    write_eels_dat(output.join("eels.dat"), &stale)?;

    run_feff_to_dir(&input, &output)?;
    let eels = read_eels_dat(output.join("eels.dat"))?;
    assert_eq!(eels.point_count(), sample_xmu_dat().point_count());
    assert_ne!(eels.energy_loss_ev, stale.energy_loss_ev);
    assert!(eels.has_tensor());
    assert!(output.join("logeels.dat").is_file());
    Ok(())
}

#[test]
fn full_run_scheduler_generates_elnes_cu_eels_reference_from_xmu_source_handoffs() -> Result<()> {
    let Some(reference_dir) = reference_elnes_cu_eels_source_dir()? else {
        require_fixture!("EELS full-run reference test; ELNES/Cu source fixture not found");
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(reference_dir.join("eels.inp"), temp.path().join("eels.inp"))?;
    copy_elnes_cu_eels_source_handoffs(&reference_dir, temp.path())?;
    let expected = read_eels_dat(reference_dir.join("eels.dat"))?;

    assert!(!temp.path().join("eels.dat").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "eels" && report.count == expected.point_count()),
        "complete ELNES/Cu EELS source handoff should report generated eels rows: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert!(actual.has_tensor());
    assert!(temp.path().join("logeels.dat").is_file());
    assert_eels_reference_close(&actual, &expected);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_eels_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_eels_opconskk_input(temp.path())?;
    std::fs::write(temp.path().join("opconsKK10.dat"), b"not an opcons table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eels"),
        "malformed standalone EELS opconsKK source should not report EELS complete: {:?}",
        reports
    );
    Ok(())
}

fn write_minimal_eels_opconskk_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("eels.inp"),
        concat!(
            "calculate ELNES?\n",
            "   1\n",
            "average? relativistic? cross-terms? Which input?\n",
            "   1   1   1   2   8\n",
            "polarizations to be used ; min step max\n",
            "  10   1  10\n",
            "beam energy in eV\n",
            " 200000.00000\n",
            "beam direction in arbitrary units\n",
            "      0.00000      0.00000      1.00000\n",
            "collection and convergence semiangle in rad\n",
            "      0.00150      0.00020\n",
            "qmesh - radial and angular grid size\n",
            "   3   2\n",
            "detector positions - two angles in rad\n",
            "      0.00000      0.00000\n",
            "calculate magic angle if magic=1\n",
            "   0\n",
            "energy for magic angle - eV above threshold\n",
            "      0.00000\n",
        ),
    )?;
    Ok(())
}

fn write_minimal_eels_xmu_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("eels.inp"),
        concat!(
            "calculate ELNES?\n",
            "   1\n",
            "average? relativistic? cross-terms? Which input?\n",
            "   0   1   1   1   4\n",
            "polarizations to be used ; min step max\n",
            "   1   1   9\n",
            "beam energy in eV\n",
            " 300000.00000\n",
            "beam direction in arbitrary units\n",
            "      0.00000      1.00000      0.00000\n",
            "collection and convergence semiangle in rad\n",
            "      0.00240      0.00000\n",
            "qmesh - radial and angular grid size\n",
            "   5   3\n",
            "detector positions - two angles in rad\n",
            "      0.00000      0.00000\n",
            "calculate magic angle if magic=1\n",
            "   0\n",
            "energy for magic angle - eV above threshold\n",
            "      0.00000\n",
        ),
    )?;
    Ok(())
}

fn write_minimal_mdff_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("mdff.inp"),
        mdff_input_string(&MdffInput {
            task: 1,
            q_input: 2,
        })?,
    )?;
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_eels_when_opconskk_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_eels_opconskk_input(temp.path())?;
    let expected = sample_eels_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;
    std::fs::write(temp.path().join("opconsKK10.dat"), b"not an opcons table\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eels"),
        "cached EELS output with malformed opconsKK source should not report EELS complete: {:?}",
        reports
    );
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn full_run_executes_cached_eelsmdff_stage_before_eels_xmu_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eelsmdff_cached_input(&input)?;
    write_mdff_dat(output.join("mdff.dat"), &sample_mdff_dat()?)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("EELS should still require xmu source after cached EELS-MDFF")?;

    let chain = format!("{error:#}");
    assert!(chain.contains("eelsmdff=2 row(s)"), "{chain}");
    assert!(chain.contains("failed to run FEFF eels stage"), "{chain}");
    assert!(chain.contains("xmu.dat"), "{chain}");
    assert_eq!(read_mdff_dat(output.join("mdff.dat"))?, sample_mdff_dat()?);
    let log = read_module_log_dat(output.join("logmdff.dat"))?;
    assert!(log.lines.iter().any(|line| line
        == "Calculating MDFF for given experimental parameters - e.g. for simulating an EELS experiment"));
    assert!(
        log.lines
            .iter()
            .any(|line| line == "Module mdff is finished.  Exiting.")
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_eelsmdff_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut global = sample_full_run_path_global();
    global.q_control.imdff = 3;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    std::fs::write(temp.path().join("mdff.inp"), b"not an mdff.inp handoff\n")?;
    write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eelsmdff"),
        "malformed mdff.inp should not report EELS-MDFF complete: {:?}",
        reports
    );
    assert!(!temp.path().join("logmdff.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_eelsmdff_when_global_input_is_malformed() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("global.inp"), b"not a global input\n")?;
    write_minimal_mdff_input(temp.path())?;
    let expected = sample_mdff_dat()?;
    write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eelsmdff"),
        "malformed global.inp should not report cached EELS-MDFF complete: {:?}",
        reports
    );
    assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
    assert!(!temp.path().join("logmdff.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_eelsmdff_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut global = sample_full_run_path_global();
    global.q_control.imdff = 3;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    let expected = sample_mdff_dat()?;
    write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eelsmdff"),
        "orphan mdff.dat cache without mdff.inp should not report EELS-MDFF complete: {:?}",
        reports
    );
    assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
    assert!(!temp.path().join("logmdff.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_eelsmdff_when_xmu_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut global = sample_full_run_path_global();
    global.q_control.imdff = 3;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_minimal_mdff_input(temp.path())?;
    write_minimal_eels_xmu_input(temp.path())?;
    write_full_run_eels_xmu_sources(temp.path())?;
    let expected = sample_mdff_dat()?;
    write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;
    std::fs::write(temp.path().join("xmu09.dat"), b"not an xmu source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "eelsmdff"),
        "cached EELS-MDFF output with malformed xmu source should not report EELS-MDFF complete: {:?}",
        reports
    );
    assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
    assert!(!temp.path().join("logmdff.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_eelsmdff_from_xmu_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eelsmdff_cached_input(&input)?;
    write_full_run_eels_xmu_sources(&output)?;
    std::fs::write(output.join("mdff.dat"), b"not an mdff.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;
    let mdff = read_mdff_dat(output.join("mdff.dat"))?;
    assert_eq!(mdff.point_count(), sample_xmu_dat().point_count());
    assert_eq!(mdff.channel_count(), 5);
    assert!(output.join("logmdff.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_stale_eelsmdff_from_xmu_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eelsmdff_cached_input(&input)?;
    write_full_run_eels_xmu_sources(&output)?;
    let stale = sample_mdff_dat()?;
    write_mdff_dat(output.join("mdff.dat"), &stale)?;

    run_feff_to_dir(&input, &output)?;
    let mdff = read_mdff_dat(output.join("mdff.dat"))?;
    assert_eq!(mdff.point_count(), sample_xmu_dat().point_count());
    assert_eq!(mdff.channel_count(), 5);
    assert_ne!(mdff.channel_count(), stale.channel_count());
    assert!(output.join("logmdff.dat").is_file());
    Ok(())
}

fn write_full_run_eels_xmu_sources(work_dir: &Path) -> Result<()> {
    for index in 1..=9 {
        write_xmu_dat(
            work_dir.join(full_run_eels_xmu_source_filename(index)),
            &sample_xmu_dat(),
        )?;
    }
    Ok(())
}

fn copy_elnes_cu_eels_source_handoffs(reference_dir: &Path, work_dir: &Path) -> Result<()> {
    for name in [
        "xmu.dat",
        "xmu02.dat",
        "xmu03.dat",
        "xmu04.dat",
        "xmu05.dat",
        "xmu06.dat",
        "xmu07.dat",
        "xmu08.dat",
        "xmu09.dat",
    ] {
        std::fs::copy(reference_dir.join(name), work_dir.join(name))?;
    }
    Ok(())
}

fn full_run_eels_xmu_source_filename(index: usize) -> String {
    match index {
        1 => "xmu.dat".to_string(),
        2..=9 => format!("xmu0{index}.dat"),
        10 => "xmu10.dat".to_string(),
        _ => format!("xmu{index}.dat"),
    }
}

fn assert_eels_reference_close(actual: &EelsDatData, expected: &EelsDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_eq!(actual.has_tensor(), expected.has_tensor());
    assert_float_series_close(
        "eels.energy_loss_ev",
        actual.energy_loss_ev.iter(),
        expected.energy_loss_ev.iter(),
        1.0e-8,
        1.0e-8,
    );
    assert_float_series_close(
        "eels.total",
        actual.total.iter(),
        expected.total.iter(),
        5.0e-5,
        1.0e-20,
    );
    assert_float_series_close(
        "eels.atomic_background",
        actual.atomic_background.iter(),
        expected.atomic_background.iter(),
        5.0e-5,
        1.0e-20,
    );
    assert_float_series_close(
        "eels.fine_structure",
        actual.fine_structure.iter(),
        expected.fine_structure.iter(),
        5.0e-5,
        1.0e-20,
    );
    match (&actual.tensor, &expected.tensor) {
        (Some(actual), Some(expected)) => {
            assert_eq!(
                actual.shape(),
                expected.shape(),
                "eels.tensor shape mismatch"
            );
            for ((row, column), &actual_value) in actual.indexed_iter() {
                assert_float_value_close(
                    actual_value,
                    expected[(row, column)],
                    5.0e-5,
                    1.0e-20,
                    &format!("eels.tensor[{row},{column}]"),
                );
            }
        }
        (None, None) => {}
        _ => panic!("eels.tensor presence mismatch"),
    }
}

fn assert_float_series_close<'a, A, E>(
    label: &str,
    actual: A,
    expected: E,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) where
    A: ExactSizeIterator<Item = &'a f64>,
    E: ExactSizeIterator<Item = &'a f64>,
{
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&actual_value, &expected_value)) in actual.zip(expected).enumerate() {
        assert_float_value_close(
            actual_value,
            expected_value,
            relative_tolerance,
            absolute_tolerance,
            &format!("{label}[{index}]"),
        );
    }
}

fn assert_float_value_close(
    actual: f64,
    expected: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    label: &str,
) {
    let tolerance = absolute_tolerance.max(relative_tolerance * expected.abs().max(1.0));
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual} != {expected} within {tolerance}"
    );
}

fn assert_dmdw_reference_close(actual: &DmdwOutData, expected: &DmdwOutData) {
    assert_eq!(
        actual.mass_enhancement_header, expected.mass_enhancement_header,
        "DMDW mass-enhancement header mismatch"
    );
    match (&actual.header, &expected.header) {
        (Some(actual), Some(expected)) => {
            assert_eq!(
                actual.lanczos_recursion_order, expected.lanczos_recursion_order,
                "DMDW Lanczos recursion order mismatch"
            );
            assert_dmdw_temperature_close(
                &actual.temperature,
                &expected.temperature,
                "DMDW header temperature",
            );
            assert_eq!(
                actual.dynamical_matrix_file, expected.dynamical_matrix_file,
                "DMDW dynamical matrix file mismatch"
            );
        }
        (None, None) => {}
        _ => panic!("DMDW header presence mismatch"),
    }
    assert_eq!(
        actual.sections.len(),
        expected.sections.len(),
        "DMDW section count mismatch"
    );
    for (index, (actual, expected)) in actual
        .sections
        .iter()
        .zip(expected.sections.iter())
        .enumerate()
    {
        assert_dmdw_section_close(actual, expected, index);
    }
}

fn assert_dmdw_section_close(
    actual: &DmdwOutSection,
    expected: &DmdwOutSection,
    section_index: usize,
) {
    assert_eq!(
        actual.subject, expected.subject,
        "DMDW section {section_index} subject mismatch"
    );
    assert_eq!(
        actual.projected_dos_component_computed, expected.projected_dos_component_computed,
        "DMDW section {section_index} projected-DOS marker mismatch"
    );
    assert_eq!(
        actual.pdos_poles.len(),
        expected.pdos_poles.len(),
        "DMDW section {section_index} pole count mismatch"
    );
    for (pole_index, (actual, expected)) in actual
        .pdos_poles
        .iter()
        .zip(expected.pdos_poles.iter())
        .enumerate()
    {
        assert_dmdw_float_close(
            actual.frequency_thz,
            expected.frequency_thz,
            &format!("DMDW section {section_index} pole {pole_index} frequency"),
        );
        assert_dmdw_float_close(
            actual.weight,
            expected.weight,
            &format!("DMDW section {section_index} pole {pole_index} weight"),
        );
    }
    match (&actual.einstein, &expected.einstein) {
        (Some(actual), Some(expected)) => {
            assert_dmdw_float_close(
                actual.frequency_thz,
                expected.frequency_thz,
                &format!("DMDW section {section_index} Einstein frequency"),
            );
            assert_dmdw_float_close(
                actual.temperature_kelvin,
                expected.temperature_kelvin,
                &format!("DMDW section {section_index} Einstein temperature"),
            );
            assert_dmdw_float_close(
                actual.effective_force_constant_n_per_m,
                expected.effective_force_constant_n_per_m,
                &format!("DMDW section {section_index} Einstein force constant"),
            );
        }
        (None, None) => {}
        _ => panic!("DMDW section {section_index} Einstein presence mismatch"),
    }
    assert_eq!(
        actual.moments.len(),
        expected.moments.len(),
        "DMDW section {section_index} moment count mismatch"
    );
    for (moment_index, (actual, expected)) in actual
        .moments
        .iter()
        .zip(expected.moments.iter())
        .enumerate()
    {
        assert_eq!(
            actual.order, expected.order,
            "DMDW section {section_index} moment {moment_index} order mismatch"
        );
        assert_dmdw_float_close(
            actual.moment_thz_power_n,
            expected.moment_thz_power_n,
            &format!("DMDW section {section_index} moment {moment_index} value"),
        );
        assert_dmdw_optional_float_close(
            actual.frequency_thz,
            expected.frequency_thz,
            &format!("DMDW section {section_index} moment {moment_index} frequency"),
        );
        assert_dmdw_optional_float_close(
            actual.temperature_kelvin,
            expected.temperature_kelvin,
            &format!("DMDW section {section_index} moment {moment_index} temperature"),
        );
        assert_dmdw_optional_float_close(
            actual.effective_force_constant_n_per_m,
            expected.effective_force_constant_n_per_m,
            &format!("DMDW section {section_index} moment {moment_index} force constant"),
        );
    }
    assert_dmdw_optional_float_close(
        actual.reduced_mass_amu,
        expected.reduced_mass_amu,
        &format!("DMDW section {section_index} reduced mass"),
    );
    assert_dmdw_optional_float_close(
        actual.path_length_angstrom,
        expected.path_length_angstrom,
        &format!("DMDW section {section_index} path length"),
    );
    assert_dmdw_optional_float_close(
        actual.sigma2_1e_minus_3_angstrom2,
        expected.sigma2_1e_minus_3_angstrom2,
        &format!("DMDW section {section_index} sigma2"),
    );
    assert_dmdw_temperature_values_close(
        &actual.sigma2_by_temperature,
        &expected.sigma2_by_temperature,
        &format!("DMDW section {section_index} sigma2-by-temperature"),
    );
    assert_dmdw_optional_float_close(
        actual.vibrational_free_energy_ev,
        expected.vibrational_free_energy_ev,
        &format!("DMDW section {section_index} vibrational free energy"),
    );
    assert_dmdw_temperature_values_close(
        &actual.vibrational_free_energy_by_temperature,
        &expected.vibrational_free_energy_by_temperature,
        &format!("DMDW section {section_index} vibrational-free-energy-by-temperature"),
    );
    assert_dmdw_optional_float_close(
        actual.u2_1e_minus_3_angstrom2,
        expected.u2_1e_minus_3_angstrom2,
        &format!("DMDW section {section_index} u2"),
    );
    assert_dmdw_temperature_values_close(
        &actual.u2_by_temperature,
        &expected.u2_by_temperature,
        &format!("DMDW section {section_index} u2-by-temperature"),
    );
}

fn assert_dmdw_temperature_close(
    actual: &DmdwOutTemperature,
    expected: &DmdwOutTemperature,
    label: &str,
) {
    match (actual, expected) {
        (DmdwOutTemperature::Single(actual), DmdwOutTemperature::Single(expected)) => {
            assert_dmdw_float_close(*actual, *expected, label);
        }
        (DmdwOutTemperature::ListedBelow, DmdwOutTemperature::ListedBelow) => {}
        _ => panic!("{label} mismatch"),
    }
}

fn assert_dmdw_temperature_values_close(
    actual: &[refeff_io::DmdwOutTemperatureValue],
    expected: &[refeff_io::DmdwOutTemperatureValue],
    label: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{label} row count mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_dmdw_float_close(
            actual.temperature_kelvin,
            expected.temperature_kelvin,
            &format!("{label} {index} temperature"),
        );
        assert_dmdw_float_close(
            actual.value,
            expected.value,
            &format!("{label} {index} value"),
        );
    }
}

fn assert_dmdw_optional_float_close(actual: Option<f64>, expected: Option<f64>, label: &str) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_dmdw_float_close(actual, expected, label),
        (None, None) => {}
        _ => panic!("{label} presence mismatch"),
    }
}

fn assert_dmdw_float_close(actual: f64, expected: f64, label: &str) {
    assert_float_value_close(actual, expected, 2.0e-4, 5.0e-4, label);
}

#[test]
fn full_run_scheduler_executes_cached_dmdw_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_input(temp.path())?;
    write_dmdw_out(temp.path().join("dmdw.out"), &sample_dmdw_out())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "dmdw" && report.count == 1),
        "missing cached DMDW report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_dmdw_out(temp.path().join("dmdw.out"))?,
        sample_dmdw_out()
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_dmdw_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("dmdw.inp"), b"not a dmdw.inp handoff\n")?;
    write_dmdw_out(temp.path().join("dmdw.out"), &sample_dmdw_out())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "dmdw"),
        "malformed dmdw.inp should not report DMDW complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_dmdw_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_dmdw_out();
    write_dmdw_out(temp.path().join("dmdw.out"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "dmdw"),
        "orphan dmdw.out cache without dmdw.inp should not report DMDW complete: {:?}",
        reports
    );
    let actual = read_dmdw_out(temp.path().join("dmdw.out"))?;
    assert_dmdw_reference_close(&actual, &expected);
    Ok(())
}

#[test]
fn full_run_scheduler_recovers_malformed_dmdw_cache_from_dym_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_source_handoffs(temp.path())?;
    std::fs::write(temp.path().join("dmdw.out"), b"not a dmdw.out cache\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "dmdw" && report.count == 1),
        "missing DMDW source output report after malformed cache repair: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(read_dmdw_out(temp.path().join("dmdw.out"))?.section_count() > 0);
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_dmdw_cache_from_dym_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_source_handoffs(temp.path())?;
    let stale = sample_dmdw_out();
    write_dmdw_out(temp.path().join("dmdw.out"), &stale)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "dmdw" && report.count == 1),
        "missing DMDW source output report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let output = read_dmdw_out(temp.path().join("dmdw.out"))?;
    assert_ne!(output, stale);
    assert_eq!(output.section_count(), 1);
    Ok(())
}

#[test]
fn full_run_scheduler_runs_dmdw_source_output_from_dym_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_source_handoffs(temp.path())?;

    assert!(!temp.path().join("dmdw.out").exists());

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "dmdw" && report.count > 0),
        "missing DMDW source output report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(read_dmdw_out(temp.path().join("dmdw.out"))?.section_count() > 0);
    Ok(())
}

#[test]
fn full_run_scheduler_generates_debye_dm_exafs_cu_dmdw_reference_from_dym_source() -> Result<()> {
    let Some(reference_dir) = reference_debye_dm_exafs_cu_dmdw_source_dir()? else {
        require_fixture!("DMDW DEBYE/DM/EXAFS/Cu full-run scheduler test; reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["dmdw.inp", "feff.dym"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected = read_dmdw_out(reference_dir.join("dmdw.out"))?;

    assert!(!temp.path().join("dmdw.out").exists());
    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "dmdw" && report.count == expected.section_count()),
        "complete DEBYE/DM/EXAFS/Cu DMDW source handoff should report generated sections: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let actual = read_dmdw_out(temp.path().join("dmdw.out"))?;
    assert_dmdw_reference_close(&actual, &expected);
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_dmdw_dym_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_input(temp.path())?;
    std::fs::write(temp.path().join("feff.dym"), b"not a dym source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "dmdw"),
        "malformed standalone DMDW .dym source should not report DMDW complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_dmdw_when_dym_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_dmdw_input(temp.path())?;
    let expected = sample_dmdw_out();
    write_dmdw_out(temp.path().join("dmdw.out"), &expected)?;
    std::fs::write(temp.path().join("feff.dym"), b"not a dym source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "dmdw"),
        "cached DMDW output with malformed .dym source should not report DMDW complete: {:?}",
        reports
    );
    assert_dmdw_reference_close(&read_dmdw_out(temp.path().join("dmdw.out"))?, &expected);
    Ok(())
}

#[test]
fn full_run_scheduler_runs_sfconv_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_sfconv_source_handoffs(temp.path())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "sfconv" && report.count == 1),
        "missing SFCONV source report: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let rendered = std::fs::read_to_string(temp.path().join("xmu.dat"))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    assert!(temp.path().join("specfunct.dat").is_file());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        "Calculating S0^2 ...\nDone with module: S0^2.\r\n\n"
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_sfconv_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("sfconv.inp"),
        b"not an sfconv.inp handoff\n",
    )?;
    write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "sfconv" | "self")),
        "malformed sfconv.inp should not report SFCONV or SELF complete: {:?}",
        reports
    );
    assert!(!temp.path().join("logsfconv.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_sfconv_or_self_outputs_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let expected_exc = read_exc_dat(temp.path().join("exc.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "sfconv" | "self")),
        "orphan SFCONV/SELF outputs without sfconv.inp should not report complete stages: {:?}",
        reports
    );
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, expected_exc);
    assert!(!temp.path().join("logsfconv.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_sfconv_target_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_sfconv_source_handoffs(temp.path())?;
    std::fs::write(temp.path().join("xmu.dat"), "not an xmu.dat target\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "sfconv"),
        "malformed SFCONV target source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("specfunct.dat").exists());
    assert!(!temp.path().join("apl.dat").exists());
    assert!(!temp.path().join("logsfconv.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_mark_unsupported_self_source_complete() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_self_source_handoffs(temp.path(), 1)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "self"),
        "unsupported SELF source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("exc.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_self_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_self_source_handoffs(temp.path(), 0)?;
    std::fs::write(temp.path().join("xsph.inp"), "not an xsph.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "self"),
        "malformed SELF xsph.inp source should not be reported complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("exc.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_self_when_xsph_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_self_source_handoffs(temp.path(), 0)?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(temp.path().join("exc.dat"))?;
    std::fs::write(temp.path().join("xsph.inp"), "not an xsph.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "self"),
        "cached SELF output with malformed xsph source should not report SELF complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, expected);
    Ok(())
}

fn write_full_run_dmdw_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        concat!(
            "   1\n",
            "   1\n",
            "   1    450.000\n",
            "   0\n",
            "feff.dym\n",
            "   1\n",
            "   2   1   2          10.00\n",
        ),
    )?;
    Ok(())
}

fn write_full_run_dmdw_source_handoffs(work_dir: &Path) -> Result<()> {
    write_full_run_dmdw_input(work_dir)?;

    let mut force_constants = Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            force_constants[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as f64 + 0.001 * component as f64;
        }
    }

    write_dym(
        work_dir.join("feff.dym"),
        &DymData {
            dym_type: 1,
            atomic_numbers: Array1::from_vec(vec![29, 29, 29]),
            atomic_masses: Array1::from_vec(vec![63.546, 63.546, 63.546]),
            coordinates: DymCoordinates::Cartesian(Array2::from_shape_vec(
                (3, 3),
                vec![0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 1.7, 0.0],
            )?),
            force_constants,
            type2_metadata: None,
            dipole_derivatives: None,
        },
    )?;
    Ok(())
}

fn write_full_run_self_source_handoffs(work_dir: &Path, ixc: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("sfconv.inp"),
        concat!(
            "msfconv, ipse, ipsk\n",
            "   0   1   0\n",
            "wsigk, cen\n",
            "      0.00000      0.00000\n",
            "ispec, ipr6\n",
            "   0   0\n",
            "cfname\n",
            "NULL        \n",
        ),
    )?;
    std::fs::write(
        work_dir.join("xsph.inp"),
        format!(
            concat!(
                "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc\n",
                "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}\n",
                "vr0, vi0\n",
                "{:13.5}{:13.5}\n",
                " lmaxph(0:nph)\n",
                "{:4}\n",
                " potlbl(iph)\n",
                "Cu    \n",
                "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap\n",
                "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}\n",
                "spinph(0:nph)\n",
                "{:13.5}\n",
                "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis\n",
                "{:4}{:4}{:4}{:4}{:4}{:4}\n",
                "electronic temperature\n",
                "{:13.5}\n",
                "ChSh_Type:\n",
                "{:4}\n",
                " the number of decomposition channels ; only used for nrixs\n",
                "{:5}\n",
                "lopt\n",
                " F\n",
                "PrintRL\n",
                " F\n",
            ),
            1,
            0,
            ixc,
            ixc,
            1,
            0,
            0,
            0,
            0,
            1,
            4,
            0,
            0,
            -1,
            0,
            0.0,
            0.0,
            3,
            0.05,
            6.0,
            1.729,
            0.07,
            8.0,
            0.0,
            12.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            -1
        ),
    )?;

    let mut loss = String::new();
    for (energy, value) in [
        (5.0, 0.18),
        (12.0, 0.45),
        (25.0, 0.32),
        (60.0, 0.20),
        (120.0, 0.11),
        (250.0, 0.05),
        (500.0, 0.02),
    ] {
        loss.push_str(&format!("{energy:12.6} {value:12.6}\n"));
    }
    std::fs::write(work_dir.join("loss.dat"), loss)?;
    Ok(())
}

fn write_full_run_sfconv_source_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("sfconv.inp"),
        concat!(
            "msfconv, ipse, ipsk\n",
            "   1   0   0\n",
            "wsigk, cen\n",
            "      0.00000      0.00000\n",
            "ispec, ipr6\n",
            "   0   0\n",
            "cfname\n",
            "NULL        \n",
        ),
    )?;

    let mut text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
    )
    .to_string();
    for row in 0..24 {
        let row = row as f64;
        text.push_str(&format!(
            "  {:10.4} {:10.4} {:10.4} {:13.6E} {:13.6E} {:13.6E}\n",
            100.0 + 5.0 * row,
            1.0 + 5.0 * row,
            0.20 + 0.02 * row,
            1.0 + 0.01 * row,
            0.80 + 0.005 * row,
            0.20 + 0.005 * row
        ));
    }
    std::fs::write(work_dir.join("xmu.dat"), text)?;
    Ok(())
}

#[test]
fn full_run_completes_from_cached_path_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_path_cached_input(&input)?;
    write_paths_dat(output.join("paths.dat"), &sample_paths_dat())?;

    run_feff_to_dir(&input, &output)?;

    let paths = read_paths_dat(output.join("paths.dat"))?;
    assert!(!paths.titles.is_empty());
    assert!(!paths.paths.is_empty());
    let log = read_module_log_dat(output.join("log4.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Pathfinder: finding scattering paths..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Eliminating path degeneracies"))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: pathfinder."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_paths_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("paths.inp"), b"not a paths.inp handoff\n")?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "path"),
        "malformed paths.inp should not report PATH complete: {:?}",
        reports
    );
    assert!(!temp.path().join("log4.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_path_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_paths_dat();
    write_paths_dat(temp.path().join("paths.dat"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "path"),
        "orphan paths.dat cache without paths.inp should not report PATH complete: {:?}",
        reports
    );
    assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
    assert!(!temp.path().join("log4.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_path_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_full_run_paths_module_input(temp.path())?;
    write_full_run_path_generation_handoffs(temp.path())?;
    let expected = sample_paths_dat();
    write_paths_dat(temp.path().join("paths.dat"), &expected)?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "path"),
        "cached PATH output with malformed phase source should not report PATH complete: {:?}",
        reports
    );
    assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
    assert!(!temp.path().join("log4.dat").exists());
    Ok(())
}

#[test]
fn full_run_recovers_malformed_paths_dat_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_path_source_handoff_input(&input)?;
    write_full_run_path_generation_handoffs(&output)?;
    std::fs::write(output.join("paths.dat"), b"not a paths.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;

    let paths = read_paths_dat(output.join("paths.dat"))?;
    assert!(!paths.paths.is_empty());
    assert!(
        paths
            .paths
            .iter()
            .any(|path| path.index == 1 && path.degeneracy > 0.0)
    );
    let log = read_module_log_dat(output.join("log4.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Pathfinder: finding scattering paths..."))
    );
    assert!(log.lines.iter().any(|line| line.contains("Unique paths")));
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_recovers_stale_paths_dat_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_path_source_handoff_input(&input)?;
    write_full_run_path_generation_handoffs(&output)?;
    let stale = sample_paths_dat();
    write_paths_dat(output.join("paths.dat"), &stale)?;

    run_feff_to_dir(&input, &output)?;

    let paths = read_paths_dat(output.join("paths.dat"))?;
    assert!(!paths.paths.is_empty());
    assert_ne!(paths.titles, stale.titles);
    assert!(
        paths
            .paths
            .iter()
            .any(|path| path.index == 1 && path.degeneracy > 0.0)
    );
    let log = read_module_log_dat(output.join("log4.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Pathfinder: finding scattering paths..."))
    );
    assert!(log.lines.iter().any(|line| line.contains("Unique paths")));
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("list.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(output.join("chi.dat").is_file());
    Ok(())
}

#[test]
fn full_run_writes_ss_paths_dat_before_required_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_single_scattering_paths_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("full run should stop after single-scattering PATH handoff")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: atomic-config=1 file(s)")
    );
    let paths = read_paths_dat(output.join("paths.dat"))?;
    assert_eq!(paths.paths.len(), 1);
    assert_eq!(paths.titles[0], " Cu SS paths run");
    assert_eq!(
        paths.titles[1],
        " Single scattering paths from ss lines cards in feff input"
    );
    let path = &paths.paths[0];
    assert_eq!(path.index, 29);
    assert_eq!(path.leg_count(), 2);
    assert_eq!(path.degeneracy, 48.0);
    assert_eq!(path.atoms[0].potential_index, 1);
    assert_eq!(path.atoms[0].label, "Cu1");
    assert_eq!(path.atoms[1].potential_index, 0);
    assert_eq!(path.atoms[1].label, "Cu0");
    let paths_input = std::fs::read_to_string(output.join("paths.inp"))?;
    assert!(paths_input.contains("   0   0   0   7   0"));
    assert!(!output.join("log4.dat").exists());
    Ok(())
}

fn write_path_source_handoff_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu PATH source handoff run
CONTROL 1 1 1 1 1 1
RPATH 2.05
POTENTIALS
0 29 Cu0
1 29 Cu1
2 29 Cu2
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
0.0 0.0 3.0 2 Cu2
END
"#,
    )?;
    Ok(())
}

fn write_full_run_paths_module_input(work_dir: &Path) -> Result<()> {
    let input = PathsInput {
        control: PathsControl {
            mpath: 1,
            ms: 1,
            nncrit: 0,
            nlegxx: 3,
            ipr4: 0,
        },
        criteria: PathsCriteria {
            critpw: 0.0,
            pcritk: 0.0,
            pcrith: 0.0,
            rmax: 2.05,
            rfms2: 0.5,
        },
        ica: -1,
    };
    std::fs::write(work_dir.join("paths.inp"), paths_input_string(&input)?)?;
    Ok(())
}

fn write_reciprocal_ldos_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu reciprocal LDOS kmesh run
LDOS -1 1 0.1 3 0
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
    )?;
    Ok(())
}

fn write_minimal_ldos_module_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("ldos.inp"),
        concat!(
            "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
            "   1   0   0   0   0       3   11\n",
            "rfms2, emin, emax, eimag, rgrd\n",
            "     -1.00000     -1.00000      1.00000      0.10000      0.05000\n",
            "rdirec, toler1, toler2\n",
            "     -1.00000      0.00100      0.00100\n",
            " lmaxph(0:nph)\n",
            "   3   3\n",
            "ldostype\n",
            "   0\n",
        ),
    )?;
    Ok(())
}

fn write_spin_ldos_module_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("ldos.inp"),
        concat!(
            "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
            "   1   0   0   1   0       3   11\n",
            "rfms2, emin, emax, eimag, rgrd\n",
            "     -1.00000     -1.00000      1.00000      0.10000      0.05000\n",
            "rdirec, toler1, toler2\n",
            "     -1.00000      0.00100      0.00100\n",
            " lmaxph(0:nph)\n",
            "   3   3\n",
            "ldostype\n",
            "   0\n",
        ),
    )?;
    Ok(())
}

fn write_active_hubbard_ldos_cached_input(work_dir: &Path) -> Result<()> {
    write_active_hubbard_ldos_cached_input_with_lfms2(work_dir, 2)
}

fn write_active_hubbard_ldos_cached_input_with_lfms2(work_dir: &Path, lfms2: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("ldos.inp"),
        format!(
            concat!(
                "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                "   1{:4}   0   0   0       3    0\n",
                "rfms2, emin, emax, eimag, rgrd\n",
                "      0.00000     -1.00000      1.00000      0.10000      0.05000\n",
                "rdirec, toler1, toler2\n",
                "      0.00000      0.00100      0.00100\n",
                " lmaxph(0:nph)\n",
                "   1   1\n",
            ),
            lfms2
        ),
    )?;
    let input = HubbardInput {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: 4.0,
        j: 0.5,
        fermi_shift: 0.0,
        l: 1,
    };
    std::fs::write(work_dir.join("hubbard.inp"), hubbard_input_string(&input)?)?;
    Ok(())
}

fn write_active_hubbard_cached_tables(work_dir: &Path, index: &str) -> Result<()> {
    write_ldos_dat(
        work_dir.join(format!("ldos{index}.dat")),
        &sample_ldos_dat()?,
    )?;
    write_rhoc_dat(
        work_dir.join(format!("rhoc{index}.dat")),
        &sample_rhoc_dat()?,
    )?;
    write_lmdos_dat(
        work_dir.join(format!("lmdos{index}.dat")),
        &sample_active_hubbard_lmdos_dat()?,
    )?;
    write_rhocm_dat(
        work_dir.join(format!("rhocm{index}.dat")),
        &sample_active_hubbard_rhocm_dat()?,
    )?;
    Ok(())
}

fn sample_active_hubbard_gtr_source_contract(
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrBinData {
    let angular_count = angular_limit + 1;
    HubbardLdosGtrBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        angular_limit,
        values: Array4::from_elem(
            (2, energy_count, potential_count, angular_count),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_active_hubbard_gtr_m_source_contract(
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrMBinData {
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    HubbardLdosGtrMBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        angular_limit,
        values: Array5::from_elem(
            (
                2,
                energy_count,
                potential_count,
                angular_count,
                magnetic_count,
            ),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_active_hubbard_gtr_off_source_contract(
    hubbard_l: usize,
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrOffBinData {
    let angular_count = angular_limit + 1;
    let order = (hubbard_l + 1) * (hubbard_l + 1);
    HubbardLdosGtrOffBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        hubbard_l,
        angular_limit,
        values: Array6::from_elem(
            (
                angular_count,
                2,
                energy_count,
                potential_count,
                order,
                order,
            ),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_active_hubbard_lmdos_dat() -> Result<LdosMagneticDatData> {
    Ok(LdosMagneticDatData {
        header_lines: vec![
            "#  Fermi level (eV):  -3.777".to_string(),
            concat!(
                "#      e   s(+0)DOS-up   p(-1)DOS-up   p(+0)DOS-up   p(+1)DOS-up",
                "   s(+0)DOS-dn   p(-1)DOS-dn   p(+0)DOS-dn   p(+1)DOS-dn"
            )
            .to_string(),
        ],
        fermi_level_ev: Some(-3.777),
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        angular_limit: 1,
        energy_ev: Array1::from_vec(vec![-1.0, 0.0, 1.0]),
        density: Array2::from_shape_vec(
            (3, 8),
            vec![
                1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4, 1.1E-4, 2.1E-4,
                3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4, 1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4,
                5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4,
            ],
        )?,
    })
}

fn sample_active_hubbard_rhocm_dat() -> Result<LdosMagneticDatData> {
    Ok(LdosMagneticDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        angular_limit: 1,
        energy_ev: Array1::from_vec(vec![-1.0, 0.0, 1.0]),
        density: Array2::from_shape_vec(
            (3, 8),
            vec![
                9.0E-4, 8.0E-4, 7.0E-4, 6.0E-4, 5.0E-4, 4.0E-4, 3.0E-4, 2.0E-4, 9.1E-4, 8.1E-4,
                7.1E-4, 6.1E-4, 5.1E-4, 4.1E-4, 3.1E-4, 2.1E-4, 9.2E-4, 8.2E-4, 7.2E-4, 6.2E-4,
                5.2E-4, 4.2E-4, 3.2E-4, 2.2E-4,
            ],
        )?,
    })
}

fn write_full_run_path_generation_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_full_run_path_geom())?,
    )?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_full_run_path_global())?,
    )?;
    write_phase_bin(
        work_dir.join("phase.bin"),
        &sample_full_run_path_phase_bin(),
    )?;
    Ok(())
}

fn sample_full_run_path_geom() -> GeomDat {
    GeomDat {
        nat: 5,
        nph: 2,
        model_atoms: vec![1, 2, 4],
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
                x: 1.0,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 2,
            },
            GeomDatRow {
                index: 3,
                x: 0.0,
                y: 2.0,
                z: 0.0,
                iph: 1,
                boundary: 0,
            },
            GeomDatRow {
                index: 4,
                x: 0.0,
                y: 0.0,
                z: 3.0,
                iph: 2,
                boundary: 3,
            },
            GeomDatRow {
                index: 5,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                iph: 2,
                boundary: 1,
            },
        ],
    }
}

fn sample_full_run_path_global() -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: 0,
            ispin: 0,
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

fn sample_full_run_path_phase_bin() -> PhaseBinData {
    let energy_count = 5;
    let spin_count = 1;
    let transition_count = 1;
    let energy_grid = Array1::from_iter(
        (0..energy_count)
            .map(|energy| Complex64::new(0.1 + 0.2 * energy as f64, 0.002 * (energy + 1) as f64)),
    );
    let reference_energy = Array2::zeros((energy_count, spin_count));
    let potentials = (0..3)
        .map(|potential| PhaseBinPotential {
            lmax: 1,
            atomic_number: 29,
            label: format!("Cu{potential}"),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, 3, spin_count),
                |(energy, signed_l, _)| {
                    Complex64::new(
                        0.03 * (potential + 1) as f64
                            + 0.01 * energy as f64
                            + 0.02 * signed_l as f64,
                        0.002 * (signed_l + 1) as f64,
                    )
                },
            ),
        })
        .collect();
    let transition_moments = Array4::<Complex64>::from_shape_fn(
        (energy_count, 1, transition_count, spin_count),
        |(energy, _, _, _)| Complex64::new(0.1 + 0.01 * energy as f64, 0.0),
    );

    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: energy_count,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 2,
        pad_width: 8,
        final_state_count: transition_count,
        transition_count,
        q_count: 1,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: 0.0,
            edge_energy: 8_979.0,
        },
        energy_grid,
        reference_energy,
        potentials,
        transition_moments,
        raw_pads: None,
    }
}

#[test]
fn full_run_regenerates_stale_cached_genfmt_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_genfmt_cached_input(&input)?;
    write_feff_bin(output.join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(output.join("list.dat"), &sample_list_dat())?;
    let stale_feff = read_feff_bin(output.join("feff.bin"))?;
    let stale_list = read_list_dat(output.join("list.dat"))?;

    run_feff_to_dir(&input, &output)?;
    let feff = read_feff_bin(output.join("feff.bin"))?;
    let list = read_list_dat(output.join("list.dat"))?;
    assert_ne!(feff, stale_feff);
    assert_ne!(list, stale_list);
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.order, 2);
    assert!(!feff.paths.is_empty());
    assert!(!list.entries.is_empty());
    let log = read_module_log_dat(output.join("log5.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating EXAFS parameters ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters (GENFMT)."))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_genfmt_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        b"not a genfmt.inp handoff\n",
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "genfmt"),
        "malformed genfmt.inp should not report GENFMT complete: {:?}",
        reports
    );
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_genfmt_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "genfmt"),
        "orphan feff.bin/list.dat cache without genfmt.inp should not report GENFMT complete: {:?}",
        reports
    );
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_genfmt_when_phase_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        concat!(
            "mfeff, ipr5, iorder, critcw, wnstar\n",
            "   1   0       2      4.00000    F\n",
            " the number of decomposi\n",
            "   -1\n",
        ),
    )?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&sample_full_run_path_global())?,
    )?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "genfmt"),
        "malformed GENFMT phase handoff should not report cached GENFMT complete: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn full_run_generates_genfmt_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_genfmt_source_handoff_input(&input)?;
    write_phase_bin(
        output.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_paths_dat(output.join("paths.dat"), &sample_paths_dat())?;

    run_feff_to_dir(&input, &output)?;

    let feff = read_feff_bin(output.join("feff.bin"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.order, 2);
    assert_eq!(feff.energy_count(), 2);
    assert_eq!(feff.paths.len(), 1);
    assert_eq!(
        read_list_dat(output.join("list.dat"))?.titles,
        sample_paths_dat().titles
    );
    let log = read_module_log_dat(output.join("log5.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters"))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_genfmt_nstar_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        concat!(
            "mfeff, ipr5, iorder, critcw, wnstar\n",
            "   1   0       2      4.00000    T\n",
            " the number of decomposi\n",
            "   -1\n",
        ),
    )?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&sample_full_run_path_global())?,
    )?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "genfmt" && report.count == 4),
        "GENFMT source handoff with wnstar should report base plus nstar outputs: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    let mut stale_nstar = expected_nstar.clone();
    stale_nstar.entries[0].nstar += 1.0;
    write_nstar_dat(temp.path().join("nstar.dat"), &stale_nstar)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "genfmt" && report.count == 4),
        "stale GENFMT nstar cache should still report repairable source handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_genfmt_feffl_from_decomposed_jas_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        concat!(
            "mfeff, ipr5, iorder, critcw, wnstar\n",
            "   1   0       2      4.00000    T\n",
            " the number of decomposi\n",
            "   1\n",
        ),
    )?;
    let mut global = sample_full_run_path_global();
    global.control.ipol = 1;
    global.control.l2lp = 1;
    global.control.do_nrixs = 1;
    global.control.ldecmx = 1;
    global.q_control.qaverage = true;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_full_run_path_phase_bin(),
    )?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "genfmt" && report.count == 5),
        "decomposed GENFMTJAS source handoff should report base, nstar, and feffl outputs: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    let expected_feffl = read_feffl_bin(
        temp.path().join("feffl.bin"),
        expected_feff.pad_width,
        expected_feff.paths.len(),
        expected_feff.energy_count(),
        1,
    )?;
    let mut stale_feffl = expected_feffl.clone();
    stale_feffl.amplitudes[(0, 0, 0, 0)] += 0.5;
    write_feffl_bin(temp.path().join("feffl.bin"), &stale_feffl)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "genfmt" && report.count == 5),
        "stale decomposed GENFMT feffl cache should report repairable source handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    assert_eq!(
        read_feffl_bin(
            temp.path().join("feffl.bin"),
            expected_feff.pad_width,
            expected_feff.paths.len(),
            expected_feff.energy_count(),
            1,
        )?,
        expected_feffl
    );
    Ok(())
}

#[test]
fn full_run_recovers_malformed_genfmt_feff_bin_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_genfmt_source_handoff_input(&input)?;
    write_phase_bin(
        output.join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_paths_dat(output.join("paths.dat"), &sample_paths_dat())?;
    std::fs::write(output.join("feff.bin"), b"not a feff.bin cache\n")?;
    write_list_dat(output.join("list.dat"), &sample_list_dat())?;

    run_feff_to_dir(&input, &output)?;

    let feff = read_feff_bin(output.join("feff.bin"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.order, 2);
    assert_eq!(feff.energy_count(), 2);
    assert_eq!(
        read_list_dat(output.join("list.dat"))?.titles,
        sample_paths_dat().titles
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_genfmt_when_malformed_fms_input_blocks_source() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        concat!(
            "mfeff, ipr5, iorder, critcw, wnstar\n",
            "   1   0       2      4.00000    F\n",
            " the number of decomposi\n",
            "   -1\n",
        ),
    )?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&sample_full_run_path_global())?,
    )?;
    write_phase_bin(
        temp.path().join("phase.bin"),
        &sample_fms_source_phase_bin_data(),
    )?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .all(|report| !matches!(report.name, "fms" | "genfmt")),
        "malformed FMS input should block FMS and downstream GENFMT source completion: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(!temp.path().join("feff.bin").exists());
    assert!(!temp.path().join("list.dat").exists());
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn full_run_completes_from_cached_ff2x_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ff2x_cached_input(&input)?;
    write_xmu_dat(output.join("xmu.dat"), &sample_xmu_dat())?;
    write_chi_dat(output.join("chi.dat"), &sample_chi_dat())?;
    write_danes_dat(output.join("danes.dat"), &sample_danes_dat())?;
    write_prexmu_dat(output.join("prexmu.dat"), &sample_xscorr_complex_table())?;
    write_residue_dat(output.join("residue.dat"), &sample_xscorr_complex_table())?;
    write_contour_dat(output.join("contour.dat"), &sample_xscorr_complex_table())?;
    write_curve_dat(output.join("curve.dat"), &sample_xscorr_curve_dat())?;
    write_xscorr_raw_dat(output.join("raw.dat"), &sample_xscorr_raw_dat())?;

    run_feff_to_dir(&input, &output)?;

    let xmu = read_xmu_dat(output.join("xmu.dat"))?;
    let chi = read_chi_dat(output.join("chi.dat"))?;
    assert_ne!(xmu, sample_xmu_dat());
    assert_ne!(chi, sample_chi_dat());
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(xmu.mu.iter().all(|value| value.is_finite()));
    assert!(xmu.mu0.iter().all(|value| value.is_finite()));
    assert!(xmu.mu.iter().any(|value| value.abs() > 1.0e-12));
    assert!(chi.chi.iter().all(|value| value.is_finite()));
    assert_eq!(
        read_danes_dat(output.join("danes.dat"))?,
        sample_danes_dat()
    );
    assert_eq!(
        read_prexmu_dat(output.join("prexmu.dat"))?,
        sample_xscorr_complex_table()
    );
    assert_eq!(
        read_residue_dat(output.join("residue.dat"))?,
        sample_xscorr_complex_table()
    );
    assert_eq!(
        read_contour_dat(output.join("contour.dat"))?,
        sample_xscorr_complex_table()
    );
    assert_eq!(
        read_curve_dat(output.join("curve.dat"))?,
        sample_xscorr_curve_dat()
    );
    assert_eq!(
        read_xscorr_raw_dat(output.join("raw.dat"))?,
        sample_xscorr_raw_dat()
    );
    let log = read_module_log_dat(output.join("log6.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Calculating XAS spectra ..."))
    );
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: XAS spectra (FF2X"))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ff2x_cache_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ff2x_input(temp.path())?;
    std::fs::write(temp.path().join("xmu.dat"), b"not an xmu.dat cache\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ff2x"),
        "malformed FF2X final cache without source handoffs should not report FF2X complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ff2x_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("ff2x.inp"), b"not an ff2x.inp handoff\n")?;
    write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ff2x"),
        "malformed ff2x.inp should not report FF2X complete: {:?}",
        reports
    );
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_ff2x_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_xmu_dat();
    write_xmu_dat(temp.path().join("xmu.dat"), &expected)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ff2x"),
        "orphan xmu.dat cache without ff2x.inp should not report FF2X complete: {:?}",
        reports
    );
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_ff2x_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ff2x_input(temp.path())?;
    std::fs::write(temp.path().join("xsect.dat"), b"not an xsect.dat source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ff2x"),
        "malformed FF2X source handoff should not report FF2X complete: {:?}",
        reports
    );
    assert!(!temp.path().join("xmu.dat").exists());
    assert!(!temp.path().join("chi.dat").exists());
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_cached_ff2x_when_xsect_source_handoff_is_malformed()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_ff2x_input(temp.path())?;
    let expected = sample_xmu_dat();
    write_xmu_dat(temp.path().join("xmu.dat"), &expected)?;
    std::fs::write(temp.path().join("xsect.dat"), b"not an xsect.dat source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "ff2x"),
        "malformed FF2X xsect source should block cached FF2X completion: {:?}",
        reports
    );
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

fn write_minimal_ff2x_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("ff2x.inp"),
        r#"mchi, ispec, idwopt, ipr6, mbconv, absolu, iGammaCH
    1    1    0    0    0    0    0
vrcorr, vicorr, s02, critcw
      0.00000      0.00000      1.00000      0.00000
tk, thetad, alphat, thetae, sig2g, sig_gk
      0.00000      0.00000      0.00000      0.00000      0.00000      0.00000
momentum transfer
      0.00000      0.00000      0.00000
 the number of decomposi
    0
electronic temperature
      0.00000
"#,
    )?;
    Ok(())
}

fn write_sig3_ff2x_input(work_dir: &Path) -> Result<()> {
    let input = Ff2xInput {
        control: Ff2xControl {
            mchi: 1,
            ispec: 0,
            idwopt: 0,
            ipr6: 0,
            mbconv: 0,
            absolu: 0,
            i_gamma_ch: 0,
        },
        corrections: Ff2xCorrections {
            vrcorr: 0.0,
            vicorr: 0.0,
            s02: 1.0,
            critcw: 4.0,
        },
        debye: Ff2xDebye {
            tk: 190.0,
            thetad: 315.0,
            alphat: 0.034,
            thetae: 400.0,
            sig2g: 0.0,
            sig_gk: 0.0,
        },
        momentum_transfer: [0.0, 0.0, 0.0],
        decomposition_channels: -1,
        electronic_temperature: 0.0,
    };
    std::fs::write(work_dir.join("ff2x.inp"), ff2x_input_string(&input)?)?;
    Ok(())
}

fn sample_full_run_single_scattering_feff_bin_data() -> FeffBinData {
    let mut feff = sample_feff_bin_data();
    feff.paths = vec![FeffBinPath {
        index: 17,
        degeneracy: 12.0,
        effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
        criterion: 100.0,
        potential_indices: Array1::from_vec(vec![1, 0]),
        positions: Array2::from_shape_fn((2, 3), |(leg, axis)| match (leg, axis) {
            (0, 0) => 2.5 / FEFF_BIN_BOHR,
            (0, 1..=2) => 0.0,
            (1, 0..=2) => 0.0,
            _ => 0.0,
        }),
        beta: Array1::from_vec(vec![0.1, 0.2]),
        eta: Array1::from_vec(vec![0.3, 0.4]),
        leg_distances: Array1::from_vec(vec![2.5 / FEFF_BIN_BOHR, 2.5 / FEFF_BIN_BOHR]),
        amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
        phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
    }];
    feff
}

fn sample_full_run_single_scattering_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.001,
            amplitude_ratio: 100.0,
            degeneracy: 12.0,
            leg_count: 2,
            effective_half_path_length_angstrom: 2.5,
        }],
    }
}

#[test]
fn full_run_generates_ff2x_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ff2x_source_handoff_input(&input)?;
    write_feff_bin(output.join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(output.join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_ff2x_source_xsect_dat())?;

    run_feff_to_dir(&input, &output)?;

    let chi = read_chi_dat(output.join("chi.dat"))?;
    let xmu = read_xmu_dat(output.join("xmu.dat"))?;
    assert_eq!(chi.point_count(), 8);
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(
        chi.header_lines
            .iter()
            .any(|line| line.contains("S02=") && line.contains("Debye_temp="))
    );
    let log = read_module_log_dat(output.join("log6.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: XAS spectra (FF2X"))
    );
    Ok(())
}

#[test]
fn full_run_scheduler_regenerates_stale_ff2x_cum_from_sig3_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sig3_ff2x_input(temp.path())?;
    write_feff_bin(
        temp.path().join("feff.bin"),
        &sample_full_run_single_scattering_feff_bin_data(),
    )?;
    write_list_dat(
        temp.path().join("list.dat"),
        &sample_full_run_single_scattering_list_dat(),
    )?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_ff2x_source_xsect_dat(),
    )?;
    let stale_cum = CumDatData {
        einstein_temperature: 25.0,
        thermal_expansion: 0.5,
        entries: vec![CumDatEntry {
            path_index: 99,
            first_cumulant_angstrom: 1.0,
            sigma2_angstrom2: 2.0,
            third_cumulant_angstrom3: 3.0,
        }],
    };
    write_cum_dat(temp.path().join("cum.dat"), &stale_cum)?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports
            .iter()
            .any(|report| report.name == "ff2x" && report.count == 4),
        "stale FF2X cum.dat cache should report repairable source handoff: {:?}",
        reports
            .iter()
            .map(|report| (report.name, report.count, report.unit))
            .collect::<Vec<_>>()
    );
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("log6.dat").is_file());
    let cum = read_cum_dat(temp.path().join("cum.dat"))?;
    assert_ne!(cum, stale_cum);
    assert_eq!(cum.einstein_temperature, 400.0);
    assert_eq!(cum.thermal_expansion, 0.034);
    assert_eq!(cum.entries.len(), 1);
    assert_eq!(cum.entries[0].path_index, 17);
    Ok(())
}

#[test]
fn full_run_recovers_malformed_ff2x_chi_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ff2x_source_handoff_input(&input)?;
    write_feff_bin(output.join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(output.join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_ff2x_source_xsect_dat())?;
    std::fs::write(output.join("chi.dat"), b"not a chi.dat cache\n")?;

    run_feff_to_dir(&input, &output)?;

    let chi = read_chi_dat(output.join("chi.dat"))?;
    let xmu = read_xmu_dat(output.join("xmu.dat"))?;
    assert_eq!(chi.point_count(), 8);
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(
        chi.header_lines
            .iter()
            .any(|line| line.contains("S02=") && line.contains("Debye_temp="))
    );
    Ok(())
}

#[test]
fn full_run_completes_from_cached_fullspectrum_stage() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fullspectrum_cached_input(&input)?;
    write_eps_dat(output.join("eps.dat"), &sample_fullspectrum_eps_dat())?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
    write_osc_str_dat(
        output.join("osc_str.dat"),
        &sample_fullspectrum_osc_str_dat(),
    )?;
    write_hamaker_dat(
        output.join("hamaker.dat"),
        &sample_fullspectrum_hamaker_dat(),
    )?;
    write_module_log_dat(
        output.join("logfullspectrum.dat"),
        &sample_fullspectrum_module_log(),
    )?;
    let expected_osc_str = read_osc_str_dat(output.join("osc_str.dat"))?;
    let expected_hamaker = read_hamaker_dat(output.join("hamaker.dat"))?;
    let expected_log = read_module_log_dat(output.join("logfullspectrum.dat"))?;

    run_feff_to_dir(&input, &output)?;

    assert_eq!(
        read_opcons_dat(output.join("opconsKK.dat"))?.point_count(),
        4
    );
    assert_eq!(
        read_sumrules_dat(output.join("sumrules.dat"))?.point_count(),
        4
    );
    assert!(output.join("opcons.dat").is_file());
    assert!(output.join("opcons0.dat").is_file());
    assert_eq!(
        read_osc_str_dat(output.join("osc_str.dat"))?,
        expected_osc_str
    );
    assert_eq!(
        read_hamaker_dat(output.join("hamaker.dat"))?,
        expected_hamaker
    );
    assert_eq!(
        read_module_log_dat(output.join("logfullspectrum.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_fullspectrum_eps_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_fullspectrum_input(temp.path())?;
    std::fs::write(temp.path().join("eps.dat"), b"not an eps.dat cache\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fullspectrum"),
        "malformed FULLSPECTRUM eps.dat should not report FULLSPECTRUM complete: {:?}",
        reports
    );
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_fullspectrum_pot_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_minimal_fullspectrum_input(temp.path())?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_fullspectrum_eps_dat())?;
    std::fs::write(temp.path().join("pot.bin"), b"not a pot.bin source\n")?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fullspectrum"),
        "malformed FULLSPECTRUM pot.bin source should not report FULLSPECTRUM complete: {:?}",
        reports
    );
    assert!(!temp.path().join("opcons.dat").exists());
    assert!(!temp.path().join("opconsKK.dat").exists());
    assert!(!temp.path().join("opcons0.dat").exists());
    assert!(!temp.path().join("sumrules.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_malformed_fullspectrum_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("fullspectrum.inp"),
        b"not a fullspectrum.inp handoff\n",
    )?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_fullspectrum_eps_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fullspectrum"),
        "malformed FULLSPECTRUM input should not report FULLSPECTRUM complete: {:?}",
        reports
    );
    assert!(!temp.path().join("opcons.dat").exists());
    assert!(!temp.path().join("opconsKK.dat").exists());
    assert!(!temp.path().join("opcons0.dat").exists());
    Ok(())
}

#[test]
fn full_run_scheduler_does_not_report_orphan_fullspectrum_eps_cache_without_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_fullspectrum_eps_dat())?;

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fullspectrum"),
        "orphan eps.dat cache without fullspectrum.inp should not report FULLSPECTRUM complete: {:?}",
        reports
    );
    assert!(!temp.path().join("opcons.dat").exists());
    assert!(!temp.path().join("opconsKK.dat").exists());
    assert!(!temp.path().join("opcons0.dat").exists());
    assert!(!temp.path().join("logfullspectrum.dat").exists());
    Ok(())
}

fn write_minimal_fullspectrum_input(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("fullspectrum.inp"),
        fullspectrum_input_string(&FullSpectrumInput { m_full_spectrum: 1 })?,
    )?;
    Ok(())
}

fn write_full_run_compton_rhorrp_callback_handoffs(work_dir: &Path) -> Result<()> {
    write_pot_bin(
        work_dir.join("pot.bin"),
        &sample_full_run_compton_callback_pot_bin(),
    )?;
    std::fs::write(
        work_dir.join("config.dat"),
        config_dat_string(&sample_full_run_compton_callback_config_dat())?,
    )?;
    write_phase_bin(
        work_dir.join("phase.bin"),
        &sample_full_run_compton_callback_phase_bin(),
    )?;
    std::fs::write(
        work_dir.join("pot.inp"),
        pot_input_string(&sample_full_run_compton_callback_pot_input())?,
    )?;
    std::fs::write(
        work_dir.join("fms.inp"),
        fms_input_string(&sample_full_run_compton_callback_fms_input())?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_full_run_compton_callback_geom_dat())?,
    )?;
    std::fs::write(
        work_dir.join("gg_diag.bin"),
        rhorrp_gg_diag_bin_bytes(&sample_full_run_compton_callback_gg_diag())?,
    )?;
    std::fs::write(
        work_dir.join("gg_slice.bin"),
        rhorrp_gg_slice_bin_bytes(&sample_full_run_compton_callback_gg_slice())?,
    )?;
    Ok(())
}

fn write_full_run_compton_rhorrp_input(path: &Path) -> Result<()> {
    std::fs::write(
        path,
        r#"
TITLE Cu compton RHORRP source run
EDGE K
COMPTON 1.0 3 0
RHOZZP
CGRID 1.0 2 2 3 3
POTENTIALS
0 29 Cu
1 8 O
ATOMS
0.0 0.0 0.0 0 Cu0
0.8 0.0 0.0 1 O1
END
"#,
    )?;
    Ok(())
}

fn sample_full_run_compton_callback_geom_dat() -> GeomDat {
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

fn sample_full_run_compton_callback_pot_bin() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["full-run COMPTON RHORRP callback test".to_string()],
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
        electron_density: full_run_compton_callback_radial_matrix(potentials, 0.01),
        coulomb_potential: full_run_compton_callback_radial_matrix(potentials, -0.02),
        total_potential: full_run_compton_callback_radial_matrix(potentials, -0.03),
        valence_density: full_run_compton_callback_radial_matrix(potentials, 0.004),
        valence_potential: full_run_compton_callback_radial_matrix(potentials, -0.005),
        magnetization_density: full_run_compton_callback_radial_matrix(potentials, 0.0002),
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

fn full_run_compton_callback_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

fn sample_full_run_compton_callback_config_dat() -> ConfigDatData {
    let mut first_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut first_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    first_occupations[0] = 1.0;
    first_occupations[1] = 2.0;
    first_valence[1] = 0.5;

    let mut second_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut second_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
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

fn sample_full_run_compton_callback_phase_bin() -> PhaseBinData {
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
            sample_full_run_compton_callback_phase_potential(29, "Cu", energy_count, spin_count),
            sample_full_run_compton_callback_phase_potential(8, "O", energy_count, spin_count),
        ],
        transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
        raw_pads: None,
    }
}

fn sample_full_run_compton_callback_phase_potential(
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

fn sample_full_run_compton_callback_pot_input() -> PotInput {
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
            iscfxc: 11,
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
        titles: vec!["full-run COMPTON RHORRP callback test".to_string()],
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

fn sample_full_run_compton_callback_fms_input() -> FmsInput {
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

fn sample_full_run_compton_callback_gg_diag() -> RhorrpGgDiagBinData {
    RhorrpGgDiagBinData {
        values: Array4::from_elem((4, 2, 1, 1), Complex32::new(0.0, 0.0)),
    }
}

fn sample_full_run_compton_callback_gg_slice() -> RhorrpGgSliceBinData {
    RhorrpGgSliceBinData {
        values: Array3::from_elem((4, 1, 2), Complex32::new(0.0, 0.0)),
    }
}
