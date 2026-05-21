use super::*;

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
