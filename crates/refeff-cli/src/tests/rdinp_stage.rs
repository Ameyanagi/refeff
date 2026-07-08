use clap::Parser as _;

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
fn full_run_completes_minimal_cu_smoke_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_minimal_input(&input)?;

    run_feff_to_dir(&input, &output)?;

    assert!(output.join("pot.inp").is_file());
    assert!(output.join("xsph.inp").is_file());
    assert!(output.join("config.dat").is_file());
    assert!(output.join("pot.bin").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("fms.bin").is_file());
    assert!(output.join("paths.dat").is_file());
    assert!(output.join("feff.bin").is_file());
    assert!(output.join("chi.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn cli_run_writes_full_smoke_output_to_requested_dir() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("cli-out");
    write_minimal_input(&input)?;

    let cli = Cli::try_parse_from([
        "refeff",
        "run",
        "--input",
        &input.display().to_string(),
        "--output",
        &output.display().to_string(),
    ])?;
    run_cli(cli)?;

    assert!(output.join("pot.bin").is_file());
    assert!(output.join("phase.bin").is_file());
    assert!(output.join("xsect.dat").is_file());
    assert!(output.join("xmu.dat").is_file());
    assert!(!temp.path().join("xmu.dat").exists());
    Ok(())
}
