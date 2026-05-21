use super::*;

#[test]
fn extracts_single_scattering_cards_and_scales_distance() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RMULTIPLIER 2.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 2.99
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.overlap_shells.len(), 2);
    assert_eq!(doc.overlap_shells[0].potential_index, 0);
    assert_eq!(doc.overlap_shells[0].neighbor_potential_index, 1);
    assert_eq!(doc.overlap_shells[0].count, 12);
    ensure!(
        (doc.overlap_shells[0].distance - 5.10532).abs() < 1.0e-12,
        "unexpected scaled OVERLAP distance: {}",
        doc.overlap_shells[0].distance
    );
    assert_eq!(doc.single_scattering_paths.len(), 1);
    let path = doc
        .single_scattering_paths
        .first()
        .context("missing SS path")?;
    assert_eq!(path.index, 29);
    assert_eq!(path.potential_index, 1);
    assert_eq!(path.degeneracy, 48.0);
    ensure!(
        (path.distance - 5.98).abs() < 1.0e-12,
        "unexpected scaled SS distance: {}",
        path.distance
    );
    Ok(())
}

#[test]
fn rejects_single_scattering_cards_without_overlap() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
SS 29 1 48 2.99
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("SS without OVERLAP should be rejected")?;

    ensure!(
        error
            .to_string()
            .contains("SS cards require an OVERLAP card"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_single_scattering_cards_without_overlap_rows() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
SS 29 1 48 2.99
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("SS without OVERLAP rows should be rejected")?;

    ensure!(
        error
            .to_string()
            .contains("SS cards require OVERLAP shell rows"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_atoms_with_overlap_geometry() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
ATOMS
0 0 0 0 Cu0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("ATOMS with OVERLAP should be rejected")?;

    ensure!(
        error.to_string().contains("cannot use ATOMS and OVERLAP"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_manual_overlap_factors() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
AFOLP 1.30
FOLP 1 1.2
FOLP 2 0.8
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.afolp, 1.30);
    assert_eq!(doc.overlap_factors.len(), 2);
    assert_eq!(doc.overlap_factors[0].potential_index, 1);
    assert_eq!(doc.overlap_factors[0].factor, 1.2);
    assert_eq!(doc.overlap_factors[1].potential_index, 2);
    assert_eq!(doc.overlap_factors[1].factor, 0.8);
    Ok(())
}

#[test]
fn extracts_ionization_cards() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
ION 1 0.2
ION 2 -0.1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.ionizations.len(), 2);
    assert_eq!(doc.ionizations[0].potential_index, 1);
    assert_eq!(doc.ionizations[0].value, 0.2);
    assert_eq!(doc.ionizations[1].potential_index, 2);
    assert_eq!(doc.ionizations[1].value, -0.1);
    Ok(())
}
