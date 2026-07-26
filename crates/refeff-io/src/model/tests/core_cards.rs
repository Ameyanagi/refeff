use super::*;

#[test]
fn rejects_unknown_root_card_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "unknown-card.inp",
        "TITLE unknown card\nNOTAFEFFCARD 1 2 3\nEND\n",
    )?;
    let error = FeffDocument::from_input(&input).expect_err("unknown card must fail");
    assert!(error.to_string().contains("unknown-card.inp:2"));
    assert!(error.to_string().contains("Keyword unrecognized."));
    Ok(())
}

#[test]
fn extracts_common_structure_cards() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
SCF 5.0 0 40 0.3
EXCHANGE 0 1.0 2.0
EXAFS 20.0
FMS 4.0 1 0 0.002 0.003 20.0
COMPTON 7.0 300 1
RHOZZP
CGRID 12.0 20 21 22 23
DEBYE 190 315 0
RPATH 5.5
DIMS 100 4
LDOS -30 20 0.1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.0 0.0 0.0 1 Cu1 1.0 1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(
        doc.active_cards,
        [
            "ATOMS",
            "CONTROL",
            "EXCHANGE",
            "TITLE",
            "RPATH",
            "DEBYE",
            "PRINT",
            "POTENTIALS",
            "EXAFS",
            "EDGE",
            "SCF",
            "FMS",
            "LDOS",
            "S02",
            "DIMS",
            "COMPTON",
            "RHOZZP",
            "CGRID"
        ]
    );
    assert_eq!(doc.titles, ["Cu crystal"]);
    let edge = doc.edge.context("missing parsed edge")?;
    assert_eq!(edge.label, "K");
    assert_eq!(doc.s02, Some(1.0));
    assert_eq!(doc.control, Some([1, 1, 1, 1, 1, 1]));
    assert_eq!(doc.scf.as_ref().map(|scf| scf.iterations), Some(40));
    assert_eq!(
        doc.exchange.as_ref().map(|exchange| exchange.vr0),
        Some(1.0)
    );
    assert_eq!(doc.exafs.as_ref().map(|exafs| exafs.xkmax), Some(20.0));
    assert_eq!(doc.ispec, 5);
    assert_eq!(doc.fms.as_ref().map(|fms| fms.radius), Some(4.0));
    assert_eq!(doc.fms.as_ref().map(|fms| fms.lfms), Some(1));
    assert_eq!(doc.fms.as_ref().map(|fms| fms.rdirec), Some(8.0));
    assert!(doc.compton.do_compton);
    assert!(doc.compton.do_rhozzp);
    assert!(doc.compton.force_jzzp);
    assert_eq!(doc.compton.pqmax, 7.0);
    assert_eq!(doc.compton.npq, 300);
    assert_eq!(doc.compton.ns, 20);
    assert_eq!(doc.compton.nphi, 21);
    assert_eq!(doc.compton.nz, 22);
    assert_eq!(doc.compton.nzp, 23);
    assert_eq!(doc.compton.zpmax, 12.0);
    assert_eq!(
        doc.debye.as_ref().map(|debye| debye.temperature),
        Some(190.0)
    );
    assert_eq!(doc.rpath, Some(5.5));
    assert_eq!(doc.dims, Some(DimensionLimits { nclusx: 100, lx: 4 }));
    assert_eq!(doc.ldos.as_ref().map(|ldos| ldos.eimag), Some(0.1));
    assert_eq!(doc.potentials.len(), 2);
    assert_eq!(doc.atoms.len(), 2);
    assert_eq!(doc.atoms[1].tag.as_deref(), Some("Cu1"));
    Ok(())
}

#[test]
fn normalizes_unavailable_debye_selector_like_feff() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        temp.path().join("spring.inp"),
        concat!(
            "* res wmax dosfit acut\n",
            " VDOS 0.03 0.5 1\n",
            "\n",
            " STRETCHES\n",
            " 0 1 27.9 2.\n",
        ),
    )?;
    let input = FeffInput::parse_str(&input_path, "DEBYE 450 315 7\nEND\n")?;

    let document = FeffDocument::from_input(&input)?;
    let debye = document.debye.context("missing parsed DEBYE card")?;

    assert_eq!(debye.requested_idwopt, 7);
    assert_eq!(debye.idwopt, 2);
    assert!(document.spring_input_text.is_some());
    Ok(())
}

#[test]
fn expands_feff7_control_and_print_cards_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONTROL 0 1 0 1
PRINT 5 2 1 4
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.control, Some([0, 0, 0, 1, 0, 1]));
    assert_eq!(doc.print, Some([5, 5, 5, 2, 1, 4]));
    Ok(())
}

#[test]
fn pads_incomplete_control_and_print_cards_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONTROL
PRINT 1 2
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.control, Some([0, 0, 0, 0, 0, 0]));
    assert_eq!(doc.print, Some([1, 2, 0, 0, 0, 0]));
    Ok(())
}

#[test]
fn active_cards_use_feff_token_order_and_alias_names() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Alias test
PLASMON 2
SFCONV
WARNION
CONFIG card 1
2 1 0
XNCD
RMAX 4.5
POTENTIAL
0 29 Cu
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(
        doc.active_cards,
        [
            "TITLE",
            "RPATH",
            "POTENTIALS",
            "XMCD",
            "MPSE",
            "SFCONV",
            "CONFIGURATION",
            "WARN"
        ]
    );
    assert_eq!(
        doc.input_cards,
        [
            "TITLE",
            "MPSE",
            "SFCONV",
            "WARN",
            "CONFIGURATION",
            "XMCD",
            "RPATH",
            "POTENTIALS"
        ]
    );
    Ok(())
}

#[test]
fn extracts_xsph_advanced_tdl_and_pmbse_controls_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TDLDA 7
PMBSE 3 4 5 6
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.xsph_advanced.izstd, 1);
    assert_eq!(doc.xsph_advanced.ifxc, 7);
    assert_eq!(doc.xsph_advanced.ipmbse, 3);
    assert_eq!(doc.xsph_advanced.itdlda, 2);
    assert_eq!(doc.xsph_advanced.nonlocal, 4);
    assert_eq!(doc.xsph_advanced.ibasis, 6);

    let pmbse_only = FeffInput::parse_str(
        "feff.inp",
        r#"
PMBSE 3 4 5 6
END
"#,
    )?;
    let doc = FeffDocument::from_input(&pmbse_only)?;
    assert_eq!(doc.xsph_advanced.izstd, 0);
    assert_eq!(doc.xsph_advanced.ifxc, 5);
    assert_eq!(doc.xsph_advanced.ipmbse, 3);
    assert_eq!(doc.xsph_advanced.itdlda, 2);
    assert_eq!(doc.xsph_advanced.nonlocal, 4);
    assert_eq!(doc.xsph_advanced.ibasis, 6);
    Ok(())
}

#[test]
fn extracts_cfaverage_and_absorber_potential_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CFAVERAGE 1 0 0
POTENTIALS
1 29 Cu
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.cfaverage.nabs, 3);
    assert_eq!(doc.cfaverage.iphabs, 1);
    assert_eq!(doc.cfaverage.rclabs, 100000.0);
    assert_eq!(doc.potentials.len(), 2);
    assert_eq!(doc.potentials[0].ipot, 0);
    assert_eq!(doc.potentials[0].z, Some(29));
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("Cu"));
    assert_eq!(doc.potentials[1].ipot, 1);

    let limited = FeffInput::parse_str(
        "feff.inp",
        r#"
CFAVERAGE 1 2 5.0
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 1 Cu0
1.0 0.0 0.0 1 Cu1
2.0 0.0 0.0 1 Cu2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&limited)?;
    assert_eq!(doc.cfaverage.nabs, 2);
    assert_eq!(doc.cfaverage.iphabs, 1);
    assert_eq!(doc.cfaverage.rclabs, 5.0);
    Ok(())
}

#[test]
fn extracts_common_control_aliases_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITL Alias controls
CONT 1 0 1 0 1 0
PRIN 3 4 5 6 7 8
EXCH 2 1.25 0.5 9
CORR -1.5 0.75
RGRI 0.03
CORE NONE
UNFR
ABSO
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.titles, ["Alias controls"]);
    assert_eq!(doc.control, Some([1, 0, 1, 0, 1, 0]));
    assert_eq!(doc.print, Some([3, 4, 5, 6, 7, 8]));
    let exchange = doc.exchange.context("missing EXCHANGE alias")?;
    assert_eq!(exchange.ixc, 2);
    assert_eq!(exchange.vr0, 1.25);
    assert_eq!(exchange.vi0, 0.5);
    assert_eq!(exchange.ixc0, Some(9));
    assert_eq!(doc.corrections, [-1.5, 0.75]);
    assert_eq!(doc.rgrid, 0.03);
    assert_eq!(doc.nohole, 0);
    assert!(doc.unfreezef);
    assert!(doc.absolute);
    assert_eq!(
        doc.active_cards,
        [
            "CONTROL",
            "EXCHANGE",
            "TITLE",
            "PRINT",
            "CORRECTIONS",
            "RGRID",
            "UNFREEZEF",
            "ABSOLUTE",
            "COREHOLE"
        ]
    );
    assert_eq!(
        doc.input_cards,
        [
            "TITLE",
            "CONTROL",
            "PRINT",
            "EXCHANGE",
            "CORRECTIONS",
            "RGRID",
            "COREHOLE",
            "UNFREEZEF",
            "ABSOLUTE"
        ]
    );
    Ok(())
}

#[test]
fn rejects_incomplete_exchange_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EXCHANGE 5
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete EXCHANGE should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("EXCHANGE requires ixc, vr0, and vi0"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_magic_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
ELNES
MAGIC
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete MAGIC should be rejected")?;
    ensure!(
        error.to_string().contains("MAGIC requires emagic"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_eels_rows_like_feff() -> anyhow::Result<()> {
    for (source, expected) in [
        ("ELNES\nEND\n", "ELNES requires beam-energy row"),
        (
            "ELNES\n200 1 1 1 1 1\nEND\n",
            "ELNES requires collection-angle row",
        ),
        (
            "ELNES\n200 0 1 1 1 1\nEND\n",
            "ELNES requires beam-direction row",
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
fn rejects_incomplete_corrections_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CORRECTIONS -1.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete CORRECTIONS should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("CORRECTIONS requires real and imaginary shifts"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_path_criteria_like_feff() -> anyhow::Result<()> {
    for (source, expected) in [
        ("CRITERIA 3.0\nEND\n", "CRITERIA requires critcw and critpw"),
        (
            "PCRITERIA 0.7\nEND\n",
            "PCRITERIA requires pcritk and pcrith",
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
fn rejects_incomplete_crpa_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CRPA 2
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete CRPA should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("CRPA requires l and rcut values"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_hubbard_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
HUBBARD 1.0 0.5
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete HUBBARD should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("HUBBARD requires U, J, fermi_shift, and l values"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_redundant_nohole_and_corehole_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
COREHOLE FSR
NOHOLE
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("NOHOLE plus COREHOLE should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("NOHOLE and COREHOLE cards are mutually exclusive"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_invalid_hole_selector_like_feff() -> anyhow::Result<()> {
    for (source, expected) in [
        ("HOLE\nEND\n", "HOLE requires ihole"),
        ("HOLE 0\nEND\n", "HOLE ihole must be positive"),
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
