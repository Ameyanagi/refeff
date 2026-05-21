use super::*;

#[test]
fn extracts_jump_removal_aliases() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
JUMP
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.jump_removal);
    assert_eq!(doc.active_cards, ["JUMPRM"]);
    assert_eq!(doc.input_cards, ["JUMPRM"]);
    Ok(())
}

#[test]
fn extracts_nogeom_output_switch() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NOGEOM
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.no_geom);
    assert_eq!(doc.active_cards, ["NOGEOM"]);
    assert_eq!(doc.input_cards, ["NOGEOM"]);
    Ok(())
}

#[test]
fn extracts_interstitial_alias_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
INTE 1 1.25
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let interstitial = doc.interstitial.context("missing INTERSTITIAL")?;
    assert_eq!(interstitial.mode, 1);
    assert_eq!(interstitial.volume_scale, 1.25);
    assert_eq!(doc.active_cards, ["INTERSTITIAL"]);
    assert_eq!(doc.input_cards, ["INTERSTITIAL"]);
    Ok(())
}

#[test]
fn accepts_blank_card_defaults_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NLEG
INTE
MULTIPOLE
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.nleg, Some(7));
    assert_eq!(
        doc.interstitial,
        Some(Interstitial {
            mode: 0,
            volume_scale: 0.0,
        })
    );
    assert_eq!(doc.le2, 0);
    assert_eq!(doc.l2lp, 0);
    assert_eq!(doc.active_cards, ["NLEG", "INTERSTITIAL", "MULT"]);
    Ok(())
}

#[test]
fn rejects_bare_rpath_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str("feff.inp", "RPATH\nEND\n")?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("bare RPATH should be rejected")?;
    assert!(error.to_string().contains("RPATH requires rmax"), "{error}");
    Ok(())
}

#[test]
fn extracts_hubbard_alias_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
HUBB 3.0 0.5 -0.1 2
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.hubbard.i_hubbard, 2);
    assert_eq!(doc.hubbard.mldos_hubb, 2);
    assert_eq!(doc.hubbard.u, 3.0);
    assert_eq!(doc.hubbard.j, 0.5);
    assert_eq!(doc.hubbard.fermi_shift, -0.1);
    assert_eq!(doc.hubbard.l, 2);
    assert_eq!(doc.active_cards, ["HUBBARD"]);
    assert_eq!(doc.input_cards, ["HUBBARD"]);
    Ok(())
}

#[test]
fn extracts_nrixs_alias_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIX 1 0.0 0.0 2.0
LDEC 4
LJMAX 2
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    let nrixs = doc.nrixs.as_ref().context("missing NRIXS")?;
    assert_eq!(doc.ispec, 1);
    assert_eq!(nrixs.nq, 1);
    assert!(!nrixs.qaverage);
    assert_eq!(nrixs.qvec, [0.0, 0.0, 2.0]);
    assert_eq!(nrixs.qnorm, 2.0);
    assert_eq!(
        nrixs.q_vectors.as_slice(),
        &[NrixsQVector {
            vector: [0.0, 0.0, 2.0],
            norm: 2.0,
            weight: [1.0, 0.0],
        }]
    );
    assert_eq!(nrixs.ldecmx, 4);
    assert_eq!(nrixs.lj, 2);
    assert_eq!(doc.active_cards, ["XANES", "NRIXS", "LJMAX", "LDECMX"]);
    assert_eq!(doc.input_cards, ["XANES", "NRIXS", "LDECMX", "LJMAX"]);
    Ok(())
}
