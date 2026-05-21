use super::*;
use anyhow::{Context as _, ensure};
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

#[test]
fn extracts_four_character_control_aliases_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITL Prefix aliases
EDGE K
HOLE 1 0.77
SCF 5.0 1 40 0.3
FMS 4.0 1 0 0.002 0.003 7.0
DEBY 190 315 0
RPAT 4.5
CRIT 3.1 2.2
PCRI 0.7 0.8
RMUL 2.0
AFOL 1.25
FOLP 1 1.35
PLAS 3 50
ELNE
100.0 1 1 1 1 1
15.0 20.0
8 6
3.0 4.0
MAGI 7112.0
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.titles, ["Prefix aliases"]);
    assert_eq!(doc.hole, Some(1));
    assert_eq!(doc.s02, Some(0.77));
    assert_eq!(doc.critcw, 3.1);
    assert_eq!(doc.critpw, 2.2);
    assert_eq!(doc.pcritk, 0.7);
    assert_eq!(doc.pcrith, 0.8);
    assert_eq!(doc.r_multiplier, 2.0);
    assert_eq!(doc.rpath, Some(9.0));
    assert_eq!(doc.scf.as_ref().map(|scf| scf.radius), Some(10.0));
    assert_eq!(doc.fms.as_ref().map(|fms| fms.radius), Some(8.0));
    assert_eq!(
        doc.debye.as_ref().map(|debye| debye.temperature),
        Some(190.0)
    );
    assert_eq!(doc.afolp, 1.25);
    assert_eq!(doc.overlap_factors.len(), 1);
    assert_eq!(doc.overlap_factors[0].factor, 1.35);
    assert_eq!(doc.i_plsmn, 3);
    assert_eq!(doc.n_poles, 50);
    assert!(doc.eels.enabled);
    assert_eq!(doc.eels.magic, 1);
    assert_eq!(doc.eels.magic_energy, 7112.0);
    assert_eq!(doc.atoms[1].x, 2.0);
    assert_eq!(
        doc.active_cards,
        [
            "ATOMS",
            "HOLE",
            "TITLE",
            "FOLP",
            "RPATH",
            "DEBYE",
            "RMULT",
            "POTENTIALS",
            "CRITERIA",
            "PCRITERIA",
            "AFOLP",
            "EDGE",
            "SCF",
            "FMS",
            "MPSE",
            "ELNES",
            "MAGIC"
        ]
    );
    Ok(())
}

#[test]
fn extracts_block_alias_rows_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTE
0 29 Cu0
1 29 Cu1
ATOM
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.potentials.len(), 2);
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("Cu0"));
    assert_eq!(doc.atoms.len(), 2);
    assert_eq!(doc.atoms[1].tag.as_deref(), Some("Cu1"));
    assert_eq!(doc.active_cards, ["ATOMS", "POTENTIALS"]);
    assert_eq!(doc.input_cards, ["POTENTIALS", "ATOMS"]);
    Ok(())
}

#[test]
fn extracts_compton_aliases_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
COMP 7.0 300 1
RHOZ
CGRI 12.0 20 21 22 23
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.ispec, 5);
    assert_eq!(doc.nohole, 0);
    assert!(doc.compton.do_compton);
    assert!(doc.compton.do_rhozzp);
    assert!(doc.compton.force_jzzp);
    assert_eq!(doc.compton.pqmax, 7.0);
    assert_eq!(doc.compton.npq, 300);
    assert_eq!(doc.compton.zpmax, 12.0);
    assert_eq!(doc.compton.ns, 20);
    assert_eq!(doc.compton.nphi, 21);
    assert_eq!(doc.compton.nz, 22);
    assert_eq!(doc.compton.nzp, 23);
    assert_eq!(doc.active_cards, ["COMPTON", "RHOZZP", "CGRID"]);
    assert_eq!(doc.input_cards, ["COMPTON", "RHOZZP", "CGRID"]);
    Ok(())
}

#[test]
fn extracts_spectroscopy_aliases_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANE 20.0 1.25 0.2
POLA 1.0 0.0 0.0
ELLI 0.25 0.0 1.0 0.0
MULT 2 1
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.ispec, 1);
    assert_eq!(doc.ipol, 1);
    assert_eq!(doc.polarization_vector, [1.0, 0.0, 0.0]);
    assert_eq!(doc.ellipticity, 0.25);
    assert_eq!(doc.incidence_vector, [0.0, 1.0, 0.0]);
    assert_eq!(doc.le2, 2);
    assert_eq!(doc.l2lp, 1);
    assert_eq!(doc.spectrum_grid.xkmax, 20.0);
    assert_eq!(doc.spectrum_grid.xkstep, 1.25);
    assert_eq!(doc.spectrum_grid.vixan, 0.2);
    assert_eq!(
        doc.active_cards,
        ["XANES", "POLARIZATION", "ELLIPTICITY", "MULT"]
    );
    assert_eq!(
        doc.input_cards,
        ["XANES", "POLARIZATION", "ELLIPTICITY", "MULT"]
    );

    let fprime = FeffDocument::from_input(&FeffInput::parse_str(
        "feff.inp",
        "FPRI -5.0 10.0 0.25\nEND\n",
    )?)?;
    assert_eq!(fprime.ispec, 4);
    assert_eq!(fprime.spectrum_grid.xkmax, -5.0);
    assert_eq!(fprime.spectrum_grid.xkstep, 10.0);
    assert_eq!(fprime.spectrum_grid.vixan, 0.25);
    assert_eq!(fprime.active_cards, ["FPRIME"]);

    let exafs = FeffDocument::from_input(&FeffInput::parse_str("feff.inp", "EXAF 15.0\nEND\n")?)?;
    assert_eq!(exafs.exafs.as_ref().map(|exafs| exafs.xkmax), Some(15.0));
    assert_eq!(exafs.spectrum_grid.xkmax, 15.0);
    assert_eq!(exafs.active_cards, ["EXAFS"]);
    Ok(())
}

#[test]
fn accepts_blank_edge_iorder_and_spin_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE
IORDER
SPIN
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.edge.as_ref().map(|edge| edge.label.as_str()), Some("K"));
    assert_eq!(doc.iorder, 0);
    assert_eq!(doc.spin, 0);
    assert_eq!(doc.spin_vector, [0.0, 0.0, 0.0]);
    assert_eq!(doc.active_cards, ["IORD", "SPIN", "EDGE"]);
    Ok(())
}

#[test]
fn accepts_blank_handoff_defaults_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Context audit
EDGE K
EXAFS 20
SYMMETRY
EGRID
CHBROADENING
DIMS
NUMDENS
CORVAL
SCFTH
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.path_symmetry, -1);
    assert!(doc.egrid_records.is_empty());
    assert_eq!(doc.xsph_handoff.core_hole_broadening, 0);
    assert_eq!(doc.dims, Some(DimensionLimits { nclusx: 0, lx: 0 }));
    assert_eq!(doc.opcons_input.number_densities, [0.0, -1.0]);
    assert_eq!(doc.corval_emin, -70.0);
    assert_eq!(doc.scf_thermal.iscfth, 0);
    assert_eq!(doc.scf_thermal.negrid, 400);
    assert_eq!(doc.scf_thermal.nmu, 100);
    assert_eq!(doc.scf_thermal.emaxscf, 5.0);
    assert_eq!(
        doc.active_cards,
        [
            "ATOMS",
            "TITLE",
            "POTENTIALS",
            "EXAFS",
            "EDGE",
            "SYMMETRY",
            "EGRID",
            "CHBROADENING",
            "DIMS",
            "NUMD",
            "CORVAL",
            "SCFTH"
        ]
    );
    Ok(())
}

#[test]
fn rejects_incomplete_polarization_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
POLARIZATION 1.0 0.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete POLARIZATION should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("POLARIZATION requires x, y, and z"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_ellipticity_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
ELLIPTICITY 0.25 0.0 1.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete ELLIPTICITY should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("ELLIPTICITY requires ellipticity and incident direction"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_fprime_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
FPRIME -5.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete FPRIME should be rejected")?;
    ensure!(
        error.to_string().contains("FPRIME requires emin and emax"),
        "unexpected error: {error}"
    );
    Ok(())
}

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

#[test]
fn extracts_density_alias_payload_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
DENS
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.ispec, 5);
    assert_eq!(doc.density_records.len(), 2);
    assert_eq!(doc.active_cards, ["DENS"]);
    assert_eq!(doc.input_cards, ["DENS"]);
    Ok(())
}

#[test]
fn extracts_configuration_alias_payload_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONF card 1
2 1 0
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.config_type, 2);
    assert_eq!(doc.config_records, ["2 1 0"]);
    assert_eq!(doc.active_cards, ["CONFIGURATION"]);
    assert_eq!(doc.input_cards, ["CONFIGURATION"]);
    Ok(())
}

#[test]
fn extracts_external_potential_restart_switches() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EXTPOT
RESTART
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.external_pot);
    assert!(doc.restart_from_pot_bin);
    assert_eq!(doc.active_cards, ["EXTPOT", "RESTART"]);
    Ok(())
}

#[test]
fn extracts_chemical_shift_alias() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CHSH 3
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.chsh_type, 3);
    assert_eq!(doc.active_cards, ["CHSHIFT"]);
    Ok(())
}

#[test]
fn extracts_corval_and_highz_handoff_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CORV -120
HIGHZ
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.corval_emin, -120.0);
    assert!(doc.finite_nucleus);
    assert_eq!(doc.active_cards, ["CORVAL", "HIGHZ"]);
    Ok(())
}

#[test]
fn extracts_warnion_alias_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
WARN
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.warn_ion);
    assert_eq!(doc.active_cards, ["WARN"]);
    Ok(())
}

#[test]
fn extracts_scf_tail_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCFTH 1 6.5 640 80 5e-5
SCFR 2.0 4
TOLS 0.1 0.002 0.0003
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.scf_thermal.iscfth, 1);
    assert_eq!(doc.scf_thermal.emaxscf, 6.5);
    assert_eq!(doc.scf_thermal.negrid, 640);
    assert_eq!(doc.scf_thermal.nmu, 80);
    assert_eq!(doc.scf_thermal.xntol, 5.0e-5);
    assert!(doc.scf_ramp.enabled);
    assert_eq!(doc.scf_ramp.rfms_start, 2.0);
    assert_eq!(doc.scf_ramp.nramp, 4);
    assert_eq!(doc.scf_tolerances.tolmu, 0.001);
    assert_eq!(doc.scf_tolerances.tolq, 0.002);
    assert_eq!(doc.scf_tolerances.tolqp, 0.0003);
    assert_eq!(doc.active_cards, ["SCFTH", "SCFR", "TOLS"]);
    Ok(())
}

#[test]
fn scales_scf_tolerances_for_negative_tols_factor() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TOLS -2
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.scf_tolerances.tolmu, -0.002);
    assert_eq!(doc.scf_tolerances.tolq, -0.002);
    assert_eq!(doc.scf_tolerances.tolqp, -0.0004);
    assert_eq!(doc.active_cards, ["TOLS"]);
    Ok(())
}

#[test]
fn extracts_ff2x_convolution_and_damping_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
MBCONV
SIG2 0.012
SIG3 0.034 250
SIGGK 0.056
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.many_body_convolution);
    assert_eq!(doc.fine_structure_damping.sig2g, 0.012);
    assert_eq!(doc.fine_structure_damping.alphat, 0.034);
    assert_eq!(doc.fine_structure_damping.thetae, 250.0);
    assert_eq!(doc.fine_structure_damping.sig_gk, 0.056);
    assert_eq!(doc.active_cards, ["SIG2", "SIG3", "MBCONV", "SIGGK"]);
    Ok(())
}

#[test]
fn ignores_empty_siggk_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SIGGK
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.fine_structure_damping.sig_gk, 0.0);
    assert_eq!(doc.active_cards, ["SIGGK"]);
    Ok(())
}

#[test]
fn extracts_sfconv_alias_and_controls_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
SO2C
SELF
SFSE 2.5
RCONV 10.25 longfilename.dat
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.sfconv);
    assert_eq!(
        doc.sfconv_input.control,
        SfconvControl {
            msfconv: 1,
            ipse: 1,
            ipsk: 1,
        }
    );
    assert_eq!(doc.sfconv_input.window.wsigk, 2.5);
    assert_eq!(doc.sfconv_input.window.cen, 10.25);
    assert_eq!(doc.sfconv_input.spectrum.ispec, 1);
    assert_eq!(doc.sfconv_input.spectrum.ipr6, 0);
    assert_eq!(doc.sfconv_input.cfname, "longfilename");
    assert_eq!(
        doc.active_cards,
        ["XANES", "SFCONV", "SELF", "SFSE", "RCONV"]
    );
    assert_eq!(
        doc.input_cards,
        ["XANES", "SFCONV", "SELF", "SFSE", "RCONV"]
    );
    Ok(())
}

#[test]
fn extracts_genfmt_and_real_phase_switches() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
IORDER 4
POLARIZATION 1 0 0
RPHASES
NSTAR
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.iorder, 4);
    assert_eq!(doc.lreal, 2);
    assert!(doc.nstar);
    assert_eq!(
        doc.active_cards,
        ["IORD", "POLARIZATION", "RPHASES", "NSTAR"]
    );
    Ok(())
}

#[test]
fn disables_nstar_without_linear_polarization_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NSTAR
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(!doc.nstar);
    assert_eq!(doc.active_cards, ["NSTAR"]);
    Ok(())
}

#[test]
fn extracts_path_symmetry_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SYMMETRY 3
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.path_symmetry, 3);
    assert_eq!(doc.active_cards, ["SYMMETRY"]);
    Ok(())
}

#[test]
fn clamps_invalid_path_symmetry_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SYMMETRY 9
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.path_symmetry, -1);
    assert_eq!(doc.active_cards, ["SYMMETRY"]);
    Ok(())
}

#[test]
fn extracts_bandstructure_handoff_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.band_input.mband, 1);
    assert_eq!(doc.band_input.energy_mesh.emin, -5.0);
    assert_eq!(doc.band_input.energy_mesh.emax, 10.0);
    assert_eq!(doc.band_input.energy_mesh.estep, 0.25);
    assert_eq!(doc.band_input.ikpath, 2);
    assert_eq!(doc.band_input.nkp, 64);
    assert!(doc.band_input.freeprop);
    assert_eq!(doc.active_cards, ["BAND"]);
    Ok(())
}

#[test]
fn rejects_incomplete_bandstructure_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
BANDSTRUCTURE -5.0 10.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete BANDSTRUCTURE should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("BANDSTRUCTURE requires emin emax estep ikpath"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_fullspectrum_handoff_switch() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
FULLSPECTRUM
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.full_spectrum_input.m_full_spectrum, 1);
    assert_eq!(doc.active_cards, ["FULLSPECTRUM"]);
    Ok(())
}

#[test]
fn extracts_scxc_handoff_controls_in_input_order() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCXC 22
TEMP 0.25 12
SCXC 21
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.electronic_temperature, 0.25);
    assert_eq!(doc.iscfxc, 21);
    assert_eq!(doc.active_cards, ["TEMP", "SCXC"]);
    assert_eq!(doc.input_cards, ["SCXC", "TEMP", "SCXC"]);
    Ok(())
}

#[test]
fn rejects_invalid_scxc_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCXC 99
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("invalid SCXC should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("SCXC iscfxc must be one of 11, 12, 21, or 22"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_screen_handoff_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCREEN rfms 5.5
SCREEN ner 64.4
SCREEN lfxc 2.4
SCREEN ermin 2e-3
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.screen_input.rfms, 5.5);
    assert_eq!(doc.screen_input.ner, 64);
    assert_eq!(doc.screen_input.lfxc, 2);
    assert_eq!(doc.screen_input.ermin, 0.002);
    assert_eq!(doc.screen_input.nei, 20);
    assert_eq!(doc.active_cards, ["SCREEN"]);
    assert_eq!(doc.input_cards, ["SCREEN", "SCREEN", "SCREEN", "SCREEN"]);
    Ok(())
}

#[test]
fn rejects_incomplete_screen_card_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCREEN rfms
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete SCREEN card should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("SCREEN requires keyword and value"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_unknown_screen_keyword_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCREEN unknown 1.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("unknown SCREEN keyword should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("unrecognized SCREEN keyword \"unknown\""),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn extracts_xsph_handoff_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CHBROADENING 1
SETEDGE
EPS0 -2.0
EGAP 1.25
CHWIDTH 0.75
RLPRINT
ICORE 3
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(doc.xsph_handoff.core_hole_broadening, 1);
    assert_eq!(doc.xsph_handoff.core_state, 3);
    assert_eq!(doc.xsph_handoff.eps0, -2.0);
    assert_eq!(doc.xsph_handoff.egap, 1.25);
    assert_eq!(doc.xsph_handoff.core_hole_width, Some(0.75));
    assert!(doc.xsph_handoff.set_edge);
    assert!(doc.xsph_handoff.print_radial_wavefunctions);
    assert_eq!(
        doc.active_cards,
        [
            "CHBROADENING",
            "SETE",
            "EPS0",
            "EGAP",
            "CHWIDTH",
            "RLPR",
            "ICOR"
        ]
    );
    Ok(())
}

#[test]
fn extracts_opcons_number_density_controls() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
OPCONS
NUMDENS 0 8.5
NUMDENS 2 4.25
PREPS
POTENTIALS
0 29 Cu0
1 29 Cu1
2 8 O
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.opcons);
    assert!(doc.opcons_input.run_opcons);
    assert!(doc.opcons_input.print_eps);
    assert_eq!(doc.opcons_input.number_densities, vec![8.5, -1.0, 4.25]);
    assert_eq!(doc.active_cards, ["POTENTIALS", "OPCONS", "NUMD", "PREP"]);
    Ok(())
}

#[test]
fn extracts_opcons_alias_controls_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
OPCO
NUMD 0 8.5
PREP
POTENTIALS
0 29 Cu0
END
"#,
    )?;

    let doc = FeffDocument::from_input(&input)?;
    assert!(doc.opcons);
    assert!(doc.opcons_input.run_opcons);
    assert!(doc.opcons_input.print_eps);
    assert_eq!(doc.opcons_input.number_densities, vec![8.5, -1.0]);
    assert_eq!(doc.active_cards, ["POTENTIALS", "OPCONS", "NUMD", "PREP"]);
    assert_eq!(doc.input_cards, ["OPCONS", "NUMD", "PREP", "POTENTIALS"]);
    Ok(())
}

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

#[test]
fn generates_potentials_for_cif_without_potentials_card() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("two-site.cif");
    std::fs::write(
        &cif_path,
        r#"
data_two_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
    )?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF two-site.cif
TARGET 2
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].ipot, 0);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[0].xnatph, Some(0.01));
    assert_eq!(doc.potentials[1].ipot, 1);
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].tag.as_deref(), Some("H"));
    assert_eq!(doc.potentials[1].xnatph, Some(1.0));
    assert_eq!(doc.potentials[2].ipot, 2);
    assert_eq!(doc.potentials[2].z, Some(8));
    assert_eq!(doc.potentials[2].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[2].xnatph, Some(1.0));
    Ok(())
}

#[test]
fn cif_equivalence_two_generates_atomic_number_potentials() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("three-site.cif");
    std::fs::write(
        &cif_path,
        r#"
data_three_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H1 0.0 0.0 0.0
H2 0.25 0.0 0.0
O1 0.5 0.5 0.5
"#,
    )?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF three-site.cif
TARGET 3
EQUI 2
FMS 4.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.cif_equivalence, 2);
    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].ipot, 0);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[0].tag.as_deref(), Some("O"));
    assert_eq!(doc.potentials[1].ipot, 1);
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].xnatph, Some(2.0));
    assert_eq!(doc.potentials[2].ipot, 2);
    assert_eq!(doc.potentials[2].z, Some(8));
    assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
    assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
    assert!(doc.active_cards.iter().any(|card| card == "EQUIVALENCE"));
    Ok(())
}

#[test]
fn rejects_bare_cif_equivalence_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EQUIVALENCE
END
"#,
    )?;

    let error = FeffDocument::from_input(&input).expect_err("bare EQUIVALENCE should fail");
    assert!(
        error
            .to_string()
            .contains("EQUIVALENCE requires a selector")
    );
    Ok(())
}

#[test]
fn cif_equivalence_four_collapses_when_potential_limit_is_exceeded() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("many-site.cif");
    let mut cif = String::from(
        r#"
data_many_sites
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
"#,
    );
    for index in 0..32 {
        let symbol = if index % 2 == 0 { "H" } else { "O" };
        let x = index as f64 / 64.0;
        cif.push_str(&format!("{symbol}{index} {x:.6} 0.0 0.0\n"));
    }
    std::fs::write(&cif_path, cif)?;

    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF many-site.cif
TARGET 2
EQUIVALENCE 4
FMS 4.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(doc.cif_equivalence, 4);
    assert_eq!(doc.potentials.len(), 3);
    assert_eq!(doc.potentials[0].z, Some(8));
    assert_eq!(doc.potentials[1].z, Some(1));
    assert_eq!(doc.potentials[1].xnatph, Some(16.0));
    assert_eq!(doc.potentials[2].z, Some(8));
    assert_eq!(doc.potentials[2].xnatph, Some(16.0));
    assert!(doc.atoms.iter().any(|atom| atom.ipot == 1));
    assert!(!doc.atoms.iter().any(|atom| atom.ipot == 3));
    Ok(())
}

#[test]
fn generates_atoms_for_cif_without_atoms_card() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let cif_path = temp.path().join("two-site.cif");
    std::fs::write(
        &cif_path,
        r#"
data_two_site
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
H 0.0 0.0 0.0
O 0.5 0.5 0.5
"#,
    )?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF two-site.cif
TARGET 2
FMS 4.0
RMULTIPLIER 2.0
EDGE K
XANES
END
"#,
    )?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;

    assert!(!doc.atoms.is_empty());
    assert_eq!(doc.atoms[0].ipot, 0);
    assert_eq!(
        (
            doc.atoms[0].x.round() as i32,
            doc.atoms[0].y.round() as i32,
            doc.atoms[0].z.round() as i32,
        ),
        (0, 0, 0)
    );
    assert!(
        doc.atoms
            .iter()
            .any(|atom| atom.ipot == 1 && (atom.x.abs() - 4.0).abs() < 1.0e-9)
    );
    Ok(())
}

fn minimal_dym_text() -> &'static str {
    concat!(
        "    1\n",
        "    1\n",
        "   29\n",
        "   63.546000\n",
        "    0.00000000    0.00000000    0.00000000\n",
        "    1    1\n",
        "  1.000000E+00  0.000000E+00  0.000000E+00\n",
        "  0.000000E+00  1.000000E+00  0.000000E+00\n",
        "  0.000000E+00  0.000000E+00  1.000000E+00\n",
    )
}
