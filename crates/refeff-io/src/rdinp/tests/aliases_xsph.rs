use super::*;

#[test]
fn writes_common_control_aliases_into_module_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITL Alias controls
EDGE K
CONT 1 0 1 0 1 0
PRIN 3 4 5 6 7 8
EXCH 2 1.25 0.5 9
CORR -1.5 0.75
RGRI 0.03
CORE NONE
UNFR
ABSO
POTENTIALS
0 29 Cu
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
    let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;

    assert_eq!(pot.titles, ["Alias controls"]);
    assert_eq!(pot.control.ipr1, 3);
    assert_eq!(pot.control.ixc, 2);
    assert_eq!(pot.run.nohole, 0);
    assert_eq!(pot.run.iunf, 1);
    assert_eq!(pot.scattering.rgrd, 0.03);
    assert_eq!(xsph.control.ipr2, 4);
    assert_eq!(xsph.control.ixc, 2);
    assert_eq!(xsph.control.ixc0, 9);
    assert_eq!(xsph.vr0, 1.25);
    assert_eq!(xsph.vi0, 0.5);
    assert_eq!(xsph.grid.rgrd, 0.03);
    assert_eq!(ff2x.control.ipr6, 8);
    assert_eq!(ff2x.control.absolu, 1);
    assert_eq!(ff2x.corrections.vrcorr, -1.5);
    assert_eq!(ff2x.corrections.vicorr, 0.75);
    Ok(())
}

#[test]
fn expands_feff7_control_and_print_into_module_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE FEFF7 control print compatibility
EDGE K
XANES
CONTROL 0 1 0 1
PRINT 5 2 1 4
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
    let fms = crate::FmsInput::parse_str("fms.inp", &fms_inp_string(&doc)?)?;
    let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;
    let genfmt = crate::GenfmtInput::parse_str("genfmt.inp", &genfmt_inp_string(&doc)?)?;
    let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;

    assert_eq!(pot.control.mpot, 0);
    assert_eq!(pot.control.ipr1, 5);
    assert_eq!(xsph.control.mphase, 0);
    assert_eq!(xsph.control.ipr2, 5);
    assert_eq!(fms.control.mfms, 0);
    assert_eq!(paths.control.mpath, 1);
    assert_eq!(paths.control.ipr4, 2);
    assert_eq!(genfmt.control.mfeff, 0);
    assert_eq!(genfmt.control.ipr5, 1);
    assert_eq!(ff2x.control.mchi, 1);
    assert_eq!(ff2x.control.ipr6, 4);
    Ok(())
}

#[test]
fn writes_four_character_control_aliases_into_module_inputs() -> Result<()> {
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
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let fms = crate::FmsInput::parse_str("fms.inp", &fms_inp_string(&doc)?)?;
    let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;
    let eels = crate::EelsInput::parse_str("eels.inp", &eels_inp_string(&doc)?)?;

    assert_eq!(pot.titles, ["Prefix aliases"]);
    assert_eq!(pot.control.ihole, 1);
    assert_eq!(pot.scattering.rfms1, 10.0);
    assert_eq!(pot.potentials[1].folp, 1.35);
    assert_eq!(fms.cluster.rfms2, 8.0);
    assert_eq!(fms.cluster.rdirec, 7.0);
    assert_eq!(fms.debye.tk, 190.0);
    assert_eq!(fms.debye.thetad, 315.0);
    assert_eq!(paths.criteria.critpw, 2.2);
    assert_eq!(paths.criteria.pcritk, 0.7);
    assert_eq!(paths.criteria.pcrith, 0.8);
    assert_eq!(paths.criteria.rmax, 9.0);
    assert_eq!(paths.criteria.rfms2, 8.0);
    assert!(eels.calculate_elnes);
    assert_eq!(eels.magic, 1);
    assert_eq!(eels.magic_energy, 7112.0);
    Ok(())
}

#[test]
fn writes_xsph_inp_for_copper_exafs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
EXAFS 20.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let xsph = xsph_inp_string(&doc)?;

    assert!(xsph.contains("   1   0   0   0   0   0   0   1   0   0 100   0   0  -1  11\n"));
    assert!(xsph.contains("Cu    Cu    \n"));
    assert!(xsph.contains("      0.05000     -1.00000      1.72919      0.07000     20.00000"));
    Ok(())
}

#[test]
fn writes_xsph_core_hole_controls_into_module_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
CHBROADENING 1
CHWIDTH 0.75
EPS0 -2.0
EGAP 1.25
SETEDGE
RLPRINT
ICORE 3
POTENTIALS
0 29 Cu
1 29 Cu
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;
    let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_inp_string(&doc)?)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let log = rdinp_log_dat(&doc)?;

    assert_eq!(xsph.control.i_gamma_ch, 1);
    assert_eq!(xsph.control.i_core_state, 3);
    assert_eq!(xsph.grid.gamach, 0.75);
    assert_eq!(xsph.grid.eps0, -2.0);
    assert_eq!(xsph.grid.egap, 1.25);
    assert!(xsph.lopt);
    assert!(xsph.print_rl);
    assert_eq!(ff2x.control.i_gamma_ch, 1);
    assert_eq!(pot.scattering.gamach, 0.75);
    assert_eq!(log.core_hole_lifetime_ev, Some(0.75));
    Ok(())
}

#[test]
fn writes_tdl_and_pmbse_controls_into_xsph_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE TDLDA PMBSE smoke
EDGE K
TDLDA 7
PMBSE 3 4 5 6
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let text = xsph_inp_string(&doc)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &text)?;

    assert_eq!(xsph.advanced.izstd, 1);
    assert_eq!(xsph.advanced.ifxc, 7);
    assert_eq!(xsph.advanced.ipmbse, 3);
    assert_eq!(xsph.advanced.itdlda, 2);
    assert_eq!(xsph.advanced.nonlocal, 4);
    assert_eq!(xsph.advanced.ibasis, 6);
    assert_eq!(crate::xsph_input_string(&xsph)?, text);
    Ok(())
}
