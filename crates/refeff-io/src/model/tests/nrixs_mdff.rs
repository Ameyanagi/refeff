use super::*;

#[test]
fn rejects_nrixs_without_xanes_or_exafs_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NRIXS 1 0.0 0.0 1.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("NRIXS without XANES or EXAFS should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("NRIXS must be combined with XANES or EXAFS"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_nrixs_card_like_feff() -> anyhow::Result<()> {
    for (source, expected) in [
        ("XANES\nNRIXS\nEND\n", "NRIXS requires nq"),
        ("XANES\nNRIXS 1\nEND\n", "NRIXS card requires nq qx qy qz"),
        (
            "XANES\nNRIXS -1\nEND\n",
            "NRIXS q-average card requires nq and q",
        ),
    ] {
        let input = FeffInput::parse_str("feff.inp", source)?;
        let error = FeffDocument::from_input(&input)
            .err()
            .with_context(|| format!("input should be rejected: {source:?}"))?;
        ensure!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }
    Ok(())
}

#[test]
fn rejects_multiple_spectroscopy_cards_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
EXAFS
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("multiple spectroscopy cards should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("ERROR more than one type of spectroscopy selected"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_ldec_without_nrixs_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
LDEC 4
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("LDEC without NRIXS should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("LDEC and LJMAX cards only allowed with NRIXS"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_nrixs_forbidden_cards_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIXS 1 0.0 0.0 1.0
POLARIZATION 1.0 0.0 0.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("NRIXS with POLARIZATION should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("card is explicitly forbidden for NRIXS"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_nrixs_multi_q_rows_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIXS 2 0.0 0.0 2.0 0.25
1.0 0.0 0.0 0.75
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
    assert_eq!(nrixs.nq, 2);
    assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
    assert_eq!(
        nrixs.q_vectors.as_slice(),
        &[
            NrixsQVector {
                vector: [0.0, 0.0, 2.0],
                norm: 2.0,
                weight: [0.25, 0.0],
            },
            NrixsQVector {
                vector: [1.0, 0.0, 0.0],
                norm: 1.0,
                weight: [0.75, 0.0],
            },
        ]
    );
    Ok(())
}

#[test]
fn extracts_nrixs_qaverage_complex_weights_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIXS -2 2.0 0.25 0.10
3.0 0.75 0.20
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
    assert_eq!(nrixs.nq, 2);
    assert!(nrixs.qaverage);
    assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
    assert_eq!(nrixs.qnorm, 2.0);
    assert_eq!(
        nrixs.q_vectors.as_slice(),
        &[
            NrixsQVector {
                vector: [0.0, 0.0, 2.0],
                norm: 2.0,
                weight: [0.25, 0.10],
            },
            NrixsQVector {
                vector: [0.0, 0.0, 3.0],
                norm: 3.0,
                weight: [0.75, 0.20],
            },
        ]
    );
    Ok(())
}

#[test]
fn extracts_mdff_handoff_controls_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIXS 1 0.0 0.0 2.0
MDFF 1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.mdff.imdff, 1);
    assert_eq!(doc.mdff.qqmdff, -1.0);
    assert_eq!(doc.mdff.cosmdff_angle, 0.0);

    let error = FeffDocument::from_input(&FeffInput::parse_str("feff.inp", "MDFF 1\nEND\n")?)
        .err()
        .context("MDFF 1 without NRIXS should be rejected")?;
    ensure!(
        error.to_string().contains("MDFF 1 requires NRIXS"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_mdff2_generated_qprime_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIXS 2 0.0 0.0 2.0 1.0
1.0 0.0 0.0 1.0
MDFF 2 3.0 45.0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
    assert_eq!(doc.mdff.imdff, 2);
    assert_eq!(doc.mdff.qqmdff, 3.0);
    assert_eq!(doc.mdff.cosmdff_angle, 45.0);
    assert!((nrixs.q_vectors[1].vector[0] - 0.0).abs() < 1.0e-12);
    assert!((nrixs.q_vectors[1].vector[1] - 2.121_320_343_559_643).abs() < 1.0e-12);
    assert!((nrixs.q_vectors[1].vector[2] - 2.121_320_343_559_643).abs() < 1.0e-12);
    assert_eq!(nrixs.q_vectors[1].norm, 1.0);

    let error = FeffDocument::from_input(&FeffInput::parse_str(
        "feff.inp",
        "XANES\nNRIXS 1 0.0 0.0 2.0\nMDFF 2\nEND\n",
    )?)
    .err()
    .context("MDFF 2 without nq=2 should be rejected")?;
    ensure!(
        error.to_string().contains("MDFF 2 requires NRIXS nq=2"),
        "unexpected error: {error}"
    );
    Ok(())
}
