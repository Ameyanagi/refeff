use super::*;

#[test]
fn listed_reciprocal_requires_handoff_even_when_real_follows_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
REAL
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("listed RECIPROCAL should require reciprocal handoff cards")?;
    ensure!(
        error
            .to_string()
            .contains("KMESH and TARGET are required for RECIPROCAL card"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn later_reciprocal_overrides_real_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
REAL
RECIPROCAL
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("RECIPROCAL should still require reciprocal handoff cards")?;
    ensure!(
        error
            .to_string()
            .contains("KMESH and TARGET are required for RECIPROCAL card"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_reciprocal_without_lattice_or_cif_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("RECIPROCAL should require one structure source")?;
    ensure!(
        error
            .to_string()
            .contains("use either LATTICE or CIF with RECIPROCAL card"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_reciprocal_with_lattice_and_cif_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
CIF dummy.cif
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("RECIPROCAL should reject simultaneous LATTICE and CIF")?;
    ensure!(
        error
            .to_string()
            .contains("use either LATTICE or CIF with RECIPROCAL card"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_strfac_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
STRFAC 1.0 2.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete STRFAC should be rejected")?;
    ensure!(
        error.to_string().contains("STRFAC requires three values"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_standalone_incomplete_strfac_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str("feff.inp", "STRFAC\nEND\n")?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("standalone incomplete STRFAC should be rejected")?;
    assert!(
        error.to_string().contains("STRFAC requires three values"),
        "{error}"
    );
    Ok(())
}

#[test]
fn rejects_cgrid_without_compton_or_rhozzp_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CGRID 10.0 32 32 32 120
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("CGRID should require COMPTON or RHOZZP")?;
    ensure!(
        error
            .to_string()
            .contains("Cannot use CGRID without COMPTON or RHOZZP.  Exiting."),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_hubbard_with_reciprocal_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
HUBBARD 1.0 0.5 0.0 2
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("HUBBARD should be rejected with RECIPROCAL")?;
    ensure!(
        error
            .to_string()
            .contains("Cannot use RECIPROCAL with HUBBARD."),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn converts_reciprocal_lattice_coordinates_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
COORDINATES 1
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let reciprocal = doc
        .reciprocal_input
        .as_ref()
        .and_then(|input| input.cell.as_ref())
        .context("missing reciprocal cell")?;

    assert_eq!(doc.coordinate_mode, 1);
    assert_eq!(reciprocal.positions[0], [0.0, 0.0, 0.0]);
    assert_eq!(reciprocal.positions[1], [0.5, 0.0, 0.0]);
    assert!(
        doc.atoms
            .iter()
            .any(|atom| atom.ipot == 1 && (atom.x.abs() - 1.0).abs() < 1.0e-12)
    );
    Ok(())
}

#[test]
fn accepts_bare_reciprocal_sgroup_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
SGROUP
LATTICE P 2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
ATOMS
0.0 0.0 0.0 1 Cu0
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let reciprocal = doc
        .reciprocal_input
        .as_ref()
        .and_then(|input| input.cell.as_ref())
        .context("missing reciprocal cell")?;

    assert_eq!(reciprocal.space_group, 1);
    assert!(doc.active_cards.iter().any(|card| card == "SGROUP"));
    Ok(())
}
