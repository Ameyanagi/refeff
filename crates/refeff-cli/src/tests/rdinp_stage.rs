use super::*;

#[test]
fn rdinp_stage_writes_supported_outputs_to_requested_dir() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;

    let report = execute_rdinp(&input, &output)?;

    assert_eq!(report.cards, 6);
    assert_eq!(report.atoms, 2);
    assert_eq!(report.potentials, 2);
    assert!(
        report
            .stdout
            .as_deref()
            .is_some_and(|stdout| stdout.starts_with("Launching FEFF version"))
    );
    assert!(output.join("atoms.dat").is_file());
    assert!(output.join("geom.dat").is_file());
    assert!(output.join(".dimensions.dat").is_file());
    assert!(output.join("log.dat").is_file());
    assert!(output.join("rixs.inp").is_file());
    assert!(!output.join(".feff.error").exists());
    Ok(())
}

#[test]
fn rdinp_stage_copies_relative_dmdw_auxiliary_to_requested_dir() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    let dym_dir = temp.path().join("dym");
    std::fs::create_dir_all(&dym_dir)?;
    write_dmdw_input(&input)?;
    std::fs::write(dym_dir.join("force.dym"), minimal_dym_text())?;

    execute_rdinp(&input, &output)?;

    assert_eq!(
        std::fs::read_to_string(output.join("dym").join("force.dym"))?,
        minimal_dym_text()
    );
    Ok(())
}

#[test]
fn full_run_writes_rdinp_outputs_before_unported_module_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;

    let error = run_feff_to_dir(&input, &output)
        .err()
        .context("downstream modules should still be unported")?;

    assert!(error.to_string().contains("completed rdinp"));
    assert!(
        error
            .to_string()
            .contains("no supported cached stages were run")
    );
    assert!(output.join("pot.inp").is_file());
    assert!(output.join("xsph.inp").is_file());
    Ok(())
}
