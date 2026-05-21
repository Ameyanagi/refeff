use super::*;

#[test]
fn extracts_debye_dynamical_matrix_options() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())?;
    let input = FeffInput::parse_str(
        &input_path,
        r#"
DEBYE 450 315 5 feff.dym 6 0 1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let debye = doc.debye.context("missing DEBYE options")?;
    ensure!(debye.idwopt == 5, "unexpected idwopt: {}", debye.idwopt);
    assert_eq!(debye.dym_file.as_deref(), Some("feff.dym"));
    assert_eq!(debye.dmdw_order, 6);
    assert_eq!(debye.dmdw_type, 0);
    assert_eq!(debye.dmdw_route, 1);
    let dym_input = doc.dym_input.context("missing DMDW auxiliary")?;
    assert_eq!(dym_input.output_name, "feff.dym");
    assert_eq!(dym_input.text, minimal_dym_text());
    Ok(())
}

#[test]
fn rejects_dmdw_auxiliary_parent_output_paths() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let input_path = temp.path().join("feff.inp");
    let input = FeffInput::parse_str(&input_path, "DEBYE 450 315 5 ../force.dym\nEND\n")?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("DMDW parent path should be rejected")?;

    ensure!(
        error.to_string().contains("output directory"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_non_numeric_potential_atomic_numbers() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 XXX Te
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("non-numeric POTENTIALS Z token should be rejected")?;

    ensure!(
        error.to_string().contains("XXX"),
        "unexpected error: {error}"
    );
    Ok(())
}
