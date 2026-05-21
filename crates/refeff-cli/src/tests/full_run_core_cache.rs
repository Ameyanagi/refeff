use super::*;

#[test]
fn full_run_executes_cached_wpot_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;
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

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    let message = error.to_string();
    assert!(message.contains("atomic=1 file(s)"));
    assert!(message.contains("wpot=5 file(s)"));
    assert!(output.join("pot00.dat").is_file());
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
fn full_run_executes_cached_atomic_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_apot_bin(output.join("apot.bin"), &sample_apot_bin_data())?;
    let expected = read_apot_bin(output.join("apot.bin"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: atomic=1 file(s)")
    );
    assert_eq!(read_apot_bin(output.join("apot.bin"))?, expected);
    Ok(())
}

#[test]
fn full_run_executes_cached_xsph_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_xsph_cached_input(&input)?;
    write_phase_bin(output.join("phase.bin"), &sample_phase_bin_data())?;
    write_xsect_dat(output.join("xsect.dat"), &sample_xsect_dat())?;
    write_mpse_dat(output.join("mpse.dat"), &sample_mpse_dat())?;
    write_emesh_dat(output.join("emesh.dat"), &sample_emesh_dat())?;
    write_emesh_bin(output.join("emesh.bin"), &sample_emesh_bin())?;
    write_module_log_dat(output.join("log2.dat"), &sample_xsph_module_log())?;
    let expected_phase = read_phase_bin(output.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(output.join("xsect.dat"))?;
    let expected_mpse = read_mpse_dat(output.join("mpse.dat"))?;
    let expected_emesh = read_emesh_dat(output.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(output.join("emesh.bin"))?;
    let expected_log = read_module_log_dat(output.join("log2.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: xsph=6 file(s)")
    );
    assert_eq!(read_phase_bin(output.join("phase.bin"))?, expected_phase);
    assert_eq!(read_xsect_dat(output.join("xsect.dat"))?, expected_xsect);
    assert_eq!(read_mpse_dat(output.join("mpse.dat"))?, expected_mpse);
    assert_eq!(read_emesh_dat(output.join("emesh.dat"))?, expected_emesh);
    assert_eq!(
        read_emesh_bin(output.join("emesh.bin"))?,
        expected_emesh_bin
    );
    assert_eq!(read_module_log_dat(output.join("log2.dat"))?, expected_log);
    Ok(())
}

#[test]
fn full_run_executes_cached_self_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_self_cached_input(&input)?;
    write_exc_dat(output.join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(output.join("exc.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: self=2 pole(s)")
    );
    assert_eq!(read_exc_dat(output.join("exc.dat"))?, expected);
    Ok(())
}

#[test]
fn full_run_executes_cached_fms_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_fms_cached_input(&input)?;
    write_fms_bin(output.join("fms.bin"), &sample_fms_bin_data())?;
    write_module_log_dat(output.join("log3.dat"), &sample_fms_module_log())?;
    let expected_fms = read_fms_bin(output.join("fms.bin"))?;
    let expected_log = read_module_log_dat(output.join("log3.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: fms=2 file(s)")
    );
    assert_eq!(read_fms_bin(output.join("fms.bin"))?, expected_fms);
    assert_eq!(read_module_log_dat(output.join("log3.dat"))?, expected_log);
    Ok(())
}

#[test]
fn full_run_executes_cached_band_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_bandstructure_input(&input)?;
    write_bandstructure_dat(
        output.join("bandstructure.dat"),
        &sample_bandstructure_dat(),
    )?;
    let expected = read_bandstructure_dat(output.join("bandstructure.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: band=1 file(s)")
    );
    assert_eq!(
        read_bandstructure_dat(output.join("bandstructure.dat"))?,
        expected
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_rixs_stage_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_rixs_cached_input(&input)?;
    write_rixs_map(output.join("rixsET.dat"), &sample_rixs_map_data())?;
    write_module_log_dat(output.join("logrixs.dat"), &sample_rixs_module_log())?;
    let expected_map = read_rixs_map(output.join("rixsET.dat"))?;
    let expected_log = read_module_log_dat(output.join("logrixs.dat"))?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: rixs=2 file(s)")
    );
    assert_eq!(read_rixs_map(output.join("rixsET.dat"))?, expected_map);
    assert_eq!(
        read_module_log_dat(output.join("logrixs.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn full_run_executes_cached_rhorrp_stage_before_unported_module_error() -> Result<()> {
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
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: rhorrp=1 file(s)")
    );
    assert_eq!(
        read_rhorrp_density_text(output.join("density.dat"))?,
        expected_density
    );
    Ok(())
}

#[test]
fn full_run_skips_incomplete_wpot_cache_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    std::fs::create_dir_all(&output)?;
    write_minimal_input(&input)?;
    write_pot_bin(output.join("pot.bin"), &sample_pot_bin_data())?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("no supported cached stages were run")
    );
    assert!(!output.join("pot00.dat").exists());
    Ok(())
}

#[test]
fn full_run_executes_cached_opcons_stage_before_unported_module_error() -> Result<()> {
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
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("supported cached stages run: opcons=3 row(s)")
    );
    assert!(output.join("loss.dat").is_file());
    Ok(())
}

#[test]
fn full_run_skips_opcons_stage_when_tables_are_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_opcons_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(
        error
            .to_string()
            .contains("no supported cached stages were run")
    );
    assert!(!output.join("loss.dat").exists());
    Ok(())
}
