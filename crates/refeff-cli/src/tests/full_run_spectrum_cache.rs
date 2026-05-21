use super::*;

#[test]
fn full_run_skips_compton_stage_when_jzzp_cache_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_compton_cached_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("no supported cached stages were run")
    );
    assert!(!output.join("compton.dat").exists());
    Ok(())
}

#[test]
fn full_run_executes_cached_compton_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_compton_cached_input(&input)?;
    write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;
    write_module_log_dat(output.join("logcompton.dat"), &sample_compton_module_log())?;
    let expected_log = read_module_log_dat(output.join("logcompton.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: compton=3 row(s)")
    );
    assert_eq!(
        read_compton_dat(output.join("compton.dat"))?.point_count(),
        3
    );
    assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
    assert_eq!(
        read_module_log_dat(output.join("logcompton.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn full_run_preserves_cached_compton_rhozzp_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_compton_rhozzp_cached_input(&input)?;
    write_jzzp_dat(output.join("jzzp.dat"), &sample_jzzp_data())?;
    write_rhozzp_dat(output.join("rhozzp.dat"), &sample_rhozzp_data())?;
    write_module_log_dat(output.join("logcompton.dat"), &sample_compton_module_log())?;
    let expected_log = read_module_log_dat(output.join("logcompton.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: compton=6 row(s)")
    );
    assert_eq!(
        read_compton_dat(output.join("compton.dat"))?.point_count(),
        3
    );
    assert_eq!(read_jzzp_dat(output.join("jzzp.dat"))?, sample_jzzp_data());
    assert_eq!(
        read_rhozzp_dat(output.join("rhozzp.dat"))?,
        sample_rhozzp_data()
    );
    assert_eq!(
        read_module_log_dat(output.join("logcompton.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_crpa_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_crpa_cached_input(&input)?;
    write_crpa_dat(output.join("crpa.dat"), &sample_crpa_dat())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: crpa=1 row(s)")
    );
    assert_eq!(read_crpa_dat(output.join("crpa.dat"))?, sample_crpa_dat());
    Ok(())
}

#[test]
fn full_run_executes_cached_screen_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_screen_cached_input(&input)?;
    write_wscrn_dat(output.join("wscrn.dat"), &sample_wscrn_dat())?;
    write_vtot_dat(output.join("vtot.dat"), &sample_vtot_dat())?;
    write_module_log_dat(output.join("logscreen.dat"), &sample_screen_module_log())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: screen=6 row(s)")
    );
    assert_eq!(
        read_wscrn_dat(output.join("wscrn.dat"))?,
        sample_wscrn_dat()
    );
    assert_eq!(read_vtot_dat(output.join("vtot.dat"))?, sample_vtot_dat());
    assert_eq!(
        read_module_log_dat(output.join("logscreen.dat"))?,
        sample_screen_module_log()
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_ldos_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_ldos_cached_input(&input)?;
    write_ldos_dat(output.join("ldos00.dat"), &sample_ldos_dat()?)?;
    write_module_log_dat(output.join("logdos.dat"), &sample_ldos_module_log())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: ldos=1 file(s)")
    );
    assert_eq!(
        read_ldos_dat(output.join("ldos00.dat"))?,
        sample_ldos_dat()?
    );
    assert_eq!(
        read_module_log_dat(output.join("logdos.dat"))?,
        sample_ldos_module_log()
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_eels_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eels_cached_input(&input)?;
    write_eels_dat(output.join("eels.dat"), &sample_eels_dat())?;
    write_module_log_dat(output.join("logeels.dat"), &sample_eels_module_log())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: eels=3 row(s)")
    );
    assert_eq!(read_eels_dat(output.join("eels.dat"))?, sample_eels_dat());
    assert_eq!(
        read_module_log_dat(output.join("logeels.dat"))?,
        sample_eels_module_log()
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_eelsmdff_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_eelsmdff_cached_input(&input)?;
    write_mdff_dat(output.join("mdff.dat"), &sample_mdff_dat()?)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: eelsmdff=2 row(s)")
    );
    assert_eq!(read_mdff_dat(output.join("mdff.dat"))?, sample_mdff_dat()?);
    Ok(())
}

#[test]
fn full_run_executes_cached_dmdw_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_dmdw_cached_input(&input)?;
    std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())?;
    write_dmdw_out(output.join("dmdw.out"), &sample_dmdw_out())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: dmdw=1 section(s)")
    );
    assert_eq!(read_dmdw_out(output.join("dmdw.out"))?, sample_dmdw_out());
    Ok(())
}

#[test]
fn full_run_executes_cached_path_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_path_cached_input(&input)?;
    write_paths_dat(output.join("paths.dat"), &sample_paths_dat())?;
    write_module_log_dat(output.join("log4.dat"), &sample_path_module_log())?;
    let expected_log = read_module_log_dat(output.join("log4.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: path=1 path(s)")
    );
    assert_eq!(
        read_paths_dat(output.join("paths.dat"))?,
        sample_paths_dat()
    );
    assert_eq!(read_module_log_dat(output.join("log4.dat"))?, expected_log);
    Ok(())
}

#[test]
fn full_run_executes_cached_genfmt_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_genfmt_cached_input(&input)?;
    write_feff_bin(output.join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(output.join("list.dat"), &sample_list_dat())?;
    write_module_log_dat(output.join("log5.dat"), &sample_genfmt_module_log())?;
    let expected_feff = read_feff_bin(output.join("feff.bin"))?;
    let expected_list = read_list_dat(output.join("list.dat"))?;
    let expected_log = read_module_log_dat(output.join("log5.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: genfmt=3 file(s)")
    );
    assert_eq!(read_feff_bin(output.join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(output.join("list.dat"))?, expected_list);
    assert_eq!(read_module_log_dat(output.join("log5.dat"))?, expected_log);
    Ok(())
}

#[test]
fn full_run_executes_cached_ff2x_stage_before_unported_module_error() -> Result<()> {
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
    write_module_log_dat(output.join("log6.dat"), &sample_ff2x_module_log())?;
    let expected_log = read_module_log_dat(output.join("log6.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: ff2x=9 file(s)")
    );
    assert_eq!(read_xmu_dat(output.join("xmu.dat"))?, sample_xmu_dat());
    assert_eq!(read_chi_dat(output.join("chi.dat"))?, sample_chi_dat());
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
    assert_eq!(read_module_log_dat(output.join("log6.dat"))?, expected_log);
    Ok(())
}

#[test]
fn full_run_executes_cached_fullspectrum_stage_before_unported_module_error() -> Result<()> {
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

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: fullspectrum=4 row(s)")
    );
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
