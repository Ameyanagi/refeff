use super::*;

#[test]
fn writes_single_scattering_paths_dat_from_ss_cards() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE SS smoke
CONTROL 1 1 1 1 1 1
RPATH 6.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 5.98
SS 30 1 2 8.0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let paths = single_scattering_paths_dat_string(&doc)?;

    assert_eq!(
        paths,
        concat!(
            " SS smoke\n",
            " Single scattering paths from ss lines cards in feff input\n",
            " -----------------------------------------------------------------------\n",
            "  29   2  48.000  index,nleg,degeneracy,r=  5.9800\n",
            " single scattering\n",
            "    5.980000    0.000000    0.000000   1 'Cu1   '\n",
            "    0.000000    0.000000    0.000000   0 'Cu0   '  x,y,z,ipot\n",
        )
    );

    let parsed = parse_paths_dat(&paths)?;
    assert_eq!(parsed.paths.len(), 1);
    assert_eq!(parsed.paths[0].index, 29);
    assert_eq!(parsed.paths[0].atoms[0].label, "Cu1");

    let outputs = text_outputs(&doc)?;
    assert_eq!(
        outputs.get("paths.dat").map(String::as_str),
        Some(paths.as_str())
    );
    Ok(())
}

#[test]
fn omits_single_scattering_paths_dat_when_path_module_is_disabled() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONTROL 1 1 1 0 1 1
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 1 1 1 2.0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert!(!text_outputs(&doc)?.contains_key("paths.dat"));
    Ok(())
}

#[test]
fn writes_overlap_geometry_into_module_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE SS smoke
CONTROL 1 1 1 1 1 1
RPATH 6.0
POTENTIALS
0 29 Cu0
1 29 Cu1
OVERLAP 0
1 12 2.55266
OVERLAP 1
0 12 2.55266
SS 29 1 48 5.98
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains(concat!(
        "OVERLAP option: novr(iph)\n",
        "   1   1\n",
        " iphovr  nnovr rovr \n",
        "    1   12      2.55266\n",
        "    0   12      2.55266\n",
        "ChSh_Type:\n",
    )));
    assert!(paths_inp_string(&doc)?.starts_with(concat!(
        "mpath, ms, nncrit, nlegxx, ipr4\n",
        "   0   0   0   7   0\n",
    )));
    assert!(fms_inp_string(&doc)?.starts_with(concat!("mfms, idwopt, minv\n", "   0  -1   0\n",)));
    Ok(())
}

#[test]
fn writes_manual_folp_factors_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
AFOLP 1.30
FOLP 1 1.2
FOLP 2 0.8
POTENTIALS
0 29 Cu0
1 29 Cu1
2 1 H
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.starts_with(concat!(
        "mpot, nph, ntitle, ihole, ipr1, iafolp, ixc,ispec, iscfxc\n",
        "   1   2   1   1   0  -1   0   0  11\n",
    )));
    assert!(pot.contains(concat!(
        " iz, lmaxsc, xnatph, xion, folp\n",
        "   29    2        1.0000000000        0.0000000000        1.0000000000\n",
        "   29    2        1.0000000000        0.0000000000        1.2000000000\n",
        "    1    1        1.0000000000        0.0000000000        0.8000000000\n",
    )));

    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
    assert_eq!(parsed.control.iafolp, -1);
    assert_eq!(parsed.potentials[0].folp, 1.0);
    assert_eq!(parsed.potentials[1].folp, 1.2);
    assert_eq!(parsed.potentials[2].folp, 0.8);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_ionization_values_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
ION 1 0.2
ION 2 -0.1
POTENTIALS
0 29 Cu0
1 29 Cu1
2 8 O
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains(concat!(
        " iz, lmaxsc, xnatph, xion, folp\n",
        "   29    2        1.0000000000        0.0000000000        1.1500000000\n",
        "   29    2        1.0000000000        0.2000000000        1.1500000000\n",
        "    8    2        1.0000000000       -0.1000000000        1.1500000000\n",
    )));

    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
    assert_eq!(parsed.potentials[0].xion, 0.0);
    assert_eq!(parsed.potentials[1].xion, 0.2);
    assert_eq!(parsed.potentials[2].xion, -0.1);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_interstitial_alias_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
INTE 1 1.25
POTENTIALS
0 29 Cu0
1 29 Cu1
ATOMS
0.0 0.0 0.0 0 Cu0
2.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot_text = pot_inp_string(&doc)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;

    assert_eq!(pot.run.inters, 1);
    assert_eq!(pot.scattering.totvol, 20.0);
    assert_eq!(crate::pot_input_string(&pot)?, pot_text);
    Ok(())
}

#[test]
fn writes_hubbard_alias_into_hubbard_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
HUBB 3.0 0.5 -0.1 2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let hubbard_text = hubbard_inp_string(&doc);
    let hubbard = crate::HubbardInput::parse_str("hubbard.inp", &hubbard_text)?;

    assert_eq!(hubbard.i_hubbard, 2);
    assert_eq!(hubbard.mldos_hubb, 2);
    assert_eq!(hubbard.u, 3.0);
    assert_eq!(hubbard.j, 0.5);
    assert_eq!(hubbard.fermi_shift, -0.1);
    assert_eq!(hubbard.l, 2);
    assert_eq!(crate::hubbard_input_string(&hubbard)?, hubbard_text);
    Ok(())
}

#[test]
fn writes_jump_removal_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
JUMPRM
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains(concat!(
        "nmix, nohole, jumprm, inters, nscmt, icoul, lfms1, iunf\n",
        "   1  -1   1   0   0   0   0   0\n",
    )));

    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
    assert_eq!(parsed.run.jumprm, 1);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_external_potential_restart_switches_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EXTPOT
RESTART
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains(concat!(
        "ExternalPot switch, StartFromFile switch\n",
        " T T\n",
    )));

    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
    assert!(parsed.external_pot);
    assert!(parsed.start_from_file);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_chemical_shift_type_into_pot_and_xsph_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CHSHIFT 3
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;
    let xsph = xsph_inp_string(&doc)?;

    assert!(pot.contains(concat!("ChSh_Type:\n", "   3\n")));
    assert!(xsph.contains(concat!("ChSh_Type:\n", "   3\n")));

    let parsed_pot = crate::PotInput::parse_str("pot.inp", &pot)?;
    let parsed_xsph = crate::XsphInput::parse_str("xsph.inp", &xsph)?;
    assert_eq!(parsed_pot.chsh_type, 3);
    assert_eq!(parsed_xsph.chsh_type, 3);
    assert_eq!(crate::pot_input_string(&parsed_pot)?, pot);
    assert_eq!(crate::xsph_input_string(&parsed_xsph)?, xsph);
    Ok(())
}

#[test]
fn writes_corval_and_highz_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CORVAL -120
HIGHZ
WARNION
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains("gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin\n"));
    assert!(pot.contains("   -120.00000\n"));
    assert!(pot.contains(concat!("FiniteNucleus, WarnIon\n", " T T\n",)));

    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;
    assert_eq!(parsed.scattering.corval_emin, -120.0);
    assert!(parsed.finite_nucleus);
    assert!(parsed.warn_ion);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_warn_alias_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
WARN
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot_text = pot_inp_string(&doc)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;

    assert!(pot.warn_ion);
    assert_eq!(crate::pot_input_string(&pot)?, pot_text);
    Ok(())
}

#[test]
fn writes_scf_tail_controls_into_pot_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SCFTH 1 6.5 640 80 5e-5
SCFR 2.0 4
TOLS 0.1 0.002 0.0003
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = pot_inp_string(&doc)?;
    let parsed = crate::PotInput::parse_str("pot.inp", &pot)?;

    assert_eq!(parsed.thermal.iscfth, 1);
    assert_eq!(parsed.thermal.emaxscf, 6.5);
    assert_eq!(parsed.thermal.negrid, 640);
    assert_eq!(parsed.thermal.nmu, 80);
    assert_eq!(parsed.thermal.xntol, 5.0e-5);
    assert!(parsed.ramp.ramp_scf);
    assert_eq!(parsed.ramp.rfms_start, 2.0);
    assert_eq!(parsed.ramp.nramp, 4);
    assert_eq!(parsed.tolerances.tolmu, 0.001);
    assert_eq!(parsed.tolerances.tolq, 0.002);
    assert_eq!(parsed.tolerances.tolqp, 0.0003);
    assert_eq!(crate::pot_input_string(&parsed)?, pot);
    Ok(())
}

#[test]
fn writes_scxc_into_module_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TEMP 0.25 12
SCXC 22
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let pot_text = pot_inp_string(&doc)?;
    let xsph_text = xsph_inp_string(&doc)?;
    let ldos_text = ldos_inp_string(&doc)?;
    let ff2x_text = ff2x_inp_string(&doc)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_text)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_text)?;
    let ldos = crate::LdosInput::parse_str("ldos.inp", &ldos_text)?;
    let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_text)?;

    assert_eq!(pot.control.iscfxc, 22);
    assert_eq!(xsph.control.iscfxc, 22);
    assert_eq!(ldos.control.iscfxc, 22);
    assert_eq!(xsph.electronic_temperature, 0.25);
    assert_eq!(ff2x.electronic_temperature, 0.25);
    assert_eq!(crate::pot_input_string(&pot)?, pot_text);
    assert_eq!(crate::xsph_input_string(&xsph)?, xsph_text);
    assert_eq!(crate::ldos_input_string(&ldos)?, ldos_text);
    assert_eq!(crate::ff2x_input_string(&ff2x)?, ff2x_text);
    Ok(())
}
