use super::{
    atoms_dat_string, compton_inp_string, config_inp_string, density_inp_string,
    dimensions_dat_string, dmdw_inp_string, eels_inp_string, ff2x_inp_string, fms_inp_string,
    fullspectrum_inp_string_for_document, genfmt_inp_string, geom_dat_string, global_inp_string,
    grid_inp_string, hubbard_inp_string, ldos_inp_string, opcons_inp_string, paths_inp_string,
    pot_inp_string, rdinp_error_log_string, rdinp_log_dat, rdinp_log_dat_string,
    rdinp_stdout_string, reciprocal_inp_string, rixs_inp_string, screen_inp_string_for_document,
    sfconv_inp_string, single_scattering_paths_dat_string, text_outputs, xsph_inp_string,
};
use crate::global_input::GlobalInput;
use crate::{FeffDocument, FeffInput, IoError, Result, parse_log_dat, parse_paths_dat};

#[test]
fn writes_atoms_dat_with_feff_widths() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 2.0 2.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let atoms = atoms_dat_string(&doc)?;

    assert_eq!(
        atoms,
        "natx =        2\n    x       y        z       iph  \n      0.00000      0.00000      0.00000   0      0.00000\n      1.00000      2.00000      2.00000   1      3.00000\n"
    );
    Ok(())
}

#[test]
fn writes_config_inp_from_config_card_payload() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONFIG card 1
0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let config = config_inp_string(&doc)?;

    assert_eq!(config.lines().next().map(str::len), Some(150));
    assert!(config.starts_with("0 Cu 1s -2 2s -2 2p -2 -4"));
    Ok(())
}

#[test]
fn writes_configuration_alias_payload_into_config_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
CONF card 1
2 1 0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let config = outputs.get("config.inp").ok_or_else(|| IoError::Parse {
        path: "feff.inp".into(),
        line: 0,
        message: "missing config.inp".to_string(),
    })?;

    assert_eq!(config.lines().next().map(str::len), Some(150));
    assert!(config.starts_with("2 1 0"));
    Ok(())
}

#[test]
fn writes_grid_inp_from_egrid_payload_tokens() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EGRID
e_grid    -15  -1.0  1.0
e_grid  last 10.0 0.1
k_grid  last 5.0 0.05
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    assert_eq!(
        doc.egrid_records,
        [
            "e_grid -15 -1.0 1.0",
            "e_grid last 10.0 0.1",
            "k_grid last 5.0 0.05"
        ]
    );

    assert_eq!(
        grid_inp_string(&doc)?,
        " e_grid -15 -1.0 1.0 \n e_grid last 10.0 0.1 \n k_grid last 5.0 0.05 \n"
    );
    Ok(())
}

#[test]
fn writes_empty_grid_inp_for_blank_egrid_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EGRID
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;

    assert!(doc.egrid_records.is_empty());
    assert_eq!(grid_inp_string(&doc)?, "");
    assert_eq!(outputs.get("grid.inp").map(String::as_str), Some(""));
    Ok(())
}

#[test]
fn writes_density_inp_only_from_density_block() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
DENSITY
line line.dat 0.0 0.0 0.0 core
1.0 0.0 0.0 101
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        density_inp_string(&doc)?,
        "line line.dat 0.0 0.0 0.0 core\n1.0 0.0 0.0 101\n"
    );
    assert!(text_outputs(&doc)?.contains_key("density.inp"));

    let input_without_density = FeffInput::parse_str("feff.inp", "END\n")?;
    let doc_without_density = FeffDocument::from_input(&input_without_density)?;
    assert!(!text_outputs(&doc_without_density)?.contains_key("density.inp"));
    Ok(())
}

#[test]
fn writes_density_alias_payload_into_density_inp() -> Result<()> {
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
    let outputs = text_outputs(&doc)?;

    assert_eq!(
        outputs.get("density.inp").map(String::as_str),
        Some("line line.dat 0.0 0.0 0.0 core\n1.0 0.0 0.0 101\n")
    );
    Ok(())
}

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

#[test]
fn writes_ff2x_convolution_and_damping_controls() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EXAFS 20.0
MBCONV
SIG2 0.012
SIG3 0.034 250
SIGGK 0.056
FMS 3.0
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let ff2x_text = ff2x_inp_string(&doc)?;
    let ff2x = crate::Ff2xInput::parse_str("ff2x.inp", &ff2x_text)?;

    assert_eq!(ff2x.control.mbconv, 1);
    assert_eq!(ff2x.debye.sig2g, 0.012);
    assert_eq!(ff2x.debye.alphat, 0.034);
    assert_eq!(ff2x.debye.thetae, 250.0);
    assert_eq!(ff2x.debye.sig_gk, 0.056);
    assert_eq!(crate::ff2x_input_string(&ff2x)?, ff2x_text);
    assert!(fms_inp_string(&doc)?.contains(concat!(
        "tk, thetad, sig2g\n",
        "      0.00000      0.00000      0.01200\n",
    )));
    Ok(())
}

#[test]
fn writes_genfmt_and_real_phase_switches() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
IORDER 4
POLARIZATION 1 0 0
RPHASES
NSTAR
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let genfmt_text = genfmt_inp_string(&doc)?;
    let genfmt = crate::GenfmtInput::parse_str("genfmt.inp", &genfmt_text)?;
    let xsph_text = xsph_inp_string(&doc)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_text)?;

    assert_eq!(genfmt.control.iorder, 4);
    assert!(genfmt.control.wnstar);
    assert_eq!(crate::genfmt_input_string(&genfmt)?, genfmt_text);
    assert_eq!(xsph.control.lreal, 2);
    assert_eq!(crate::xsph_input_string(&xsph)?, xsph_text);
    assert!(
        rdinp_stdout_string(&doc)?
            .contains(" Real phase shifts only will be used.  FEFF results will be unreliable.\n")
    );
    Ok(())
}

#[test]
fn writes_path_symmetry_into_paths_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SYMMETRY 3
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let paths_text = paths_inp_string(&doc)?;
    let paths = crate::PathsInput::parse_str("paths.inp", &paths_text)?;

    assert_eq!(paths.ica, 3);
    assert_eq!(crate::paths_input_string(&paths)?, paths_text);
    assert!(
        rdinp_stdout_string(&doc)?
            .contains(" SYMMETRY CARD - fixing icase to    3 in module PATH.\n")
    );
    Ok(())
}

#[test]
fn nrixs_overrides_path_symmetry_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
SYMMETRY 3
XANES
NRIXS 1 0.0 0.0 1.0
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let paths = crate::PathsInput::parse_str("paths.inp", &paths_inp_string(&doc)?)?;

    assert_eq!(doc.path_symmetry, 3);
    assert_eq!(paths.ica, 7);
    Ok(())
}

#[test]
fn writes_nrixs_alias_into_global_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
NRIX 1 0.0 0.0 2.0
LDEC 4
LJMAX 2
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global_text = global_inp_string(&doc)?;
    let global = GlobalInput::parse_str("global.inp", &global_text)?;

    assert_eq!(global.control.do_nrixs, 1);
    assert_eq!(global.control.ldecmx, 4);
    assert_eq!(global.control.lj, 2);
    assert_eq!(global.control.l2lp, 30);
    assert_eq!(global.q_control.nq, 1);
    assert_eq!(global.q_vectors[0].q, [0.0, 0.0, 2.0]);
    assert_eq!(crate::global_input_string(&global)?, global_text);
    Ok(())
}

#[test]
fn writes_bandstructure_into_band_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
BANDSTRUCTURE -5.0 10.0 0.25 2 64 T
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let band_text = outputs.get("band.inp").ok_or_else(|| IoError::Parse {
        path: "feff.inp".into(),
        line: 0,
        message: "missing band.inp output".to_string(),
    })?;
    let band = crate::BandInput::parse_str("band.inp", band_text)?;

    assert_eq!(band.mband, 1);
    assert_eq!(band.energy_mesh.emin, -5.0);
    assert_eq!(band.energy_mesh.emax, 10.0);
    assert_eq!(band.energy_mesh.estep, 0.25);
    assert_eq!(band.ikpath, 2);
    assert_eq!(band.nkp, 64);
    assert!(band.freeprop);
    assert_eq!(crate::band_input_string(&band)?, *band_text);
    assert!(rdinp_stdout_string(&doc)?.contains("BANDSTRUCTURE card is experimental.\n"));
    Ok(())
}

#[test]
fn writes_fullspectrum_switch_into_fullspectrum_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
FULLSPECTRUM
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let fullspectrum_text = outputs
        .get("fullspectrum.inp")
        .ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing fullspectrum.inp output".to_string(),
        })?;
    let fullspectrum = crate::FullSpectrumInput::parse_str("fullspectrum.inp", fullspectrum_text)?;

    assert_eq!(fullspectrum.m_full_spectrum, 1);
    assert_eq!(
        crate::fullspectrum_input_string(&fullspectrum)?,
        *fullspectrum_text
    );
    assert_eq!(
        fullspectrum_inp_string_for_document(&doc)?,
        *fullspectrum_text
    );
    Ok(())
}

#[test]
fn writes_sfconv_alias_controls_into_sfconv_inp() -> Result<()> {
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
    let sfconv_text = sfconv_inp_string(&doc)?;
    let sfconv = crate::SfconvInput::parse_str("sfconv.inp", &sfconv_text)?;

    assert_eq!(sfconv.control.msfconv, 1);
    assert_eq!(sfconv.control.ipse, 1);
    assert_eq!(sfconv.control.ipsk, 1);
    assert_eq!(sfconv.window.wsigk, 2.5);
    assert_eq!(sfconv.window.cen, 10.25);
    assert_eq!(sfconv.spectrum.ispec, 1);
    assert_eq!(sfconv.spectrum.ipr6, 0);
    assert_eq!(sfconv.cfname, "longfilename");
    assert_eq!(crate::sfconv_input_string(&sfconv)?, sfconv_text);
    Ok(())
}

#[test]
fn writes_opcons_number_density_controls() -> Result<()> {
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
    let opcons_text = opcons_inp_string(&doc)?;
    let opcons = crate::OpconsInput::parse_str("opcons.inp", &opcons_text)?;

    assert!(opcons.run_opcons);
    assert!(opcons.print_eps);
    assert_eq!(opcons.number_densities, vec![8.5, -1.0, 4.25]);
    assert_eq!(crate::opcons_input_string(&opcons)?, opcons_text);
    Ok(())
}

#[test]
fn writes_opcons_alias_controls_into_opcons_inp() -> Result<()> {
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
    let opcons_text = opcons_inp_string(&doc)?;
    let opcons = crate::OpconsInput::parse_str("opcons.inp", &opcons_text)?;

    assert!(opcons.run_opcons);
    assert!(opcons.print_eps);
    assert_eq!(opcons.number_densities, vec![8.5, -1.0]);
    assert_eq!(crate::opcons_input_string(&opcons)?, opcons_text);
    Ok(())
}

#[test]
fn writes_screen_controls_into_screen_inp() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE K
SCREEN rfms 5.5
SCREEN ner 64.4
SCREEN eimax 3.25
SCREEN icore 2
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let screen_text = outputs.get("screen.inp").ok_or_else(|| IoError::Parse {
        path: "feff.inp".into(),
        line: 0,
        message: "missing screen.inp output".to_string(),
    })?;
    let screen = crate::ScreenInput::parse_str("screen.inp", screen_text)?;

    assert_eq!(screen.rfms, 5.5);
    assert_eq!(screen.ner, 64);
    assert_eq!(screen.eimax, 3.25);
    assert_eq!(screen.icore, 2);
    assert_eq!(crate::screen_input_string(&screen)?, *screen_text);
    assert_eq!(screen_inp_string_for_document(&doc)?, *screen_text);
    assert!(rdinp_stdout_string(&doc)?.contains(":INFO  User provides options for screen.inp\n"));
    Ok(())
}

#[test]
fn writes_reciprocal_inp_from_lattice_atoms() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 1000 0
TARGET 1
LATTICE P 2.456
0.86603 -0.5000 0.00000
0.00000 1.00000 0.00000
0.00000 0.00000 2.72638
ATOMS
0.00000 0.00000 0.68160 1 C1
0.00000 0.00000 2.04479 1 C1
0.57735 0.00000 0.68160 2 C2
0.28868 0.50000 2.04479 2 C2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let reciprocal = outputs
        .get("reciprocal.inp")
        .ok_or_else(|| IoError::Parse {
            path: "test".into(),
            line: 0,
            message: "missing reciprocal.inp".to_string(),
        })?;

    assert!(reciprocal.starts_with("ispace\n   0\n"));
    assert!(reciprocal.contains("      2.12697     -1.22800      0.00000\n"));
    assert!(
        reciprocal
            .contains("        1000        1000           0           0           1           0\n")
    );
    assert!(reciprocal.contains("           1           1           2           2\n"));
    Ok(())
}

#[test]
fn writes_reciprocal_coordinates_one_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE COORDINATES smoke
EDGE K
RECIPROCAL
KMESH 10 0
TARGET 1
FMS 2.0
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
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
    let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;
    let reciprocal_input = doc
        .reciprocal_input
        .as_ref()
        .ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "missing reciprocal input".to_string(),
        })?;
    let reciprocal = crate::reciprocal_input_string(reciprocal_input)?;

    assert_eq!(atoms.atoms[0].iph, 0);
    assert_eq!(atoms.atoms[1].distance, 1.0);
    assert_eq!(atoms.atoms[2].distance, 1.0);
    assert_eq!(geom.atoms[1].x, -1.0);
    assert_eq!(geom.atoms[2].x, 1.0);
    assert!(reciprocal.contains("      0.50000      0.00000      0.00000\n"));
    Ok(())
}

#[test]
fn logs_real_and_reciprocal_cards_in_input_order() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RECIPROCAL
KMESH 10 0
TARGET 1
LATTICE P 1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
REAL
EDGE K
POTENTIALS
0 29 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let outputs = text_outputs(&doc)?;
    let expected_reciprocal = reciprocal_inp_string();

    assert!(!doc.reciprocal);
    assert!(doc.reciprocal_input.is_none());
    assert!(
        rdinp_stdout_string(&doc)?
            .contains("Working in reciprocal space.\nWorking in real space.\n")
    );
    assert_eq!(
        outputs.get("reciprocal.inp").map(String::as_str),
        Some(expected_reciprocal.as_str())
    );
    Ok(())
}

#[test]
fn writes_global_inp_defaults() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global = global_inp_string(&doc)?;

    assert!(global.contains(" nabs, iphabs - CFAVERAGE data\n       1       0 100000.00000\n"));
    assert!(global.contains(" polarization tensor \n      0.33333"));
    Ok(())
}

#[test]
fn writes_cfaverage_into_global_pot_and_geom_outputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE CFAVERAGE smoke
EDGE K
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
    let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;
    let geom = crate::GeomDat::parse_str("geom.dat", &geom_dat_string(&doc)?)?;

    assert_eq!(global.cfaverage.nabs, 3);
    assert_eq!(global.cfaverage.iphabs, 1);
    assert_eq!(global.cfaverage.rclabs, 100000.0);
    assert_eq!(pot.control.nph, 1);
    assert_eq!(pot.potentials[0].z, 29);
    assert_eq!(pot.potentials[0].xnatph, 1.0);
    assert_eq!(pot.potentials[1].z, 29);
    assert_eq!(pot.potentials[1].xnatph, 2.0);
    assert_eq!(atoms.atoms[0].iph, 1);
    assert_eq!(geom.model_atoms, [1, 2]);
    assert_eq!(geom.atoms[0].iph, 0);
    assert_eq!(geom.atoms[1].iph, 1);
    Ok(())
}

#[test]
fn writes_global_inp_linear_polarization_tensor() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POLARIZATION 1 0 0
MULTIPOLE 2
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let text = global_inp_string(&doc)?;
    let global = GlobalInput::parse_str("global.inp", &text)?;

    assert_eq!(global.control.ipol, 1);
    assert_eq!(global.control.le2, 0);
    assert_eq!(
        global.polarization_tensor,
        [
            [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            [0.0; 6],
            [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        ]
    );
    Ok(())
}

#[test]
fn writes_spectroscopy_aliases_into_global_and_xsph_inputs() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE K
XANE 20.0 1.25 0.2
POLA 1.0 0.0 0.0
ELLI 0.25 0.0 1.0 0.0
MULT 2 1
POTENTIALS
0 29 Cu
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let global = GlobalInput::parse_str("global.inp", &global_inp_string(&doc)?)?;
    let xsph = crate::XsphInput::parse_str("xsph.inp", &xsph_inp_string(&doc)?)?;

    assert_eq!(global.control.ipol, 1);
    assert_eq!(global.control.le2, 2);
    assert_eq!(global.control.l2lp, 1);
    assert_eq!(global.control.elpty, 0.25);
    assert_eq!(global.xivec, [0.0, 1.0, 0.0]);
    assert_eq!(global.norms.xivnorm, 1.0);
    assert_eq!(xsph.control.ispec, 1);
    assert_eq!(xsph.grid.xkstep, 1.25);
    assert_eq!(xsph.grid.xkmax, 20.0);
    assert_eq!(xsph.grid.vixan, 0.2);
    Ok(())
}

#[test]
fn rejects_zero_length_global_polarization_vector() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POLARIZATION 0 0 0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    assert!(global_inp_string(&doc).is_err());
    Ok(())
}

#[test]
fn writes_dimensions_dat_from_cluster_radius() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
RPATH 2.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
3.0 0.0 0.0 1 Cu2
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        dimensions_dat_string(&doc)?,
        "           2           3           1           1\n"
    );
    Ok(())
}

#[test]
fn writes_geom_dat_with_sorted_relative_cluster() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
2.0 0.0 0.0 1 Cu2
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        geom_dat_string(&doc)?,
        concat!(
            "nat, nph =     3    1\n",
            "    1    2\n",
            " iat     x       y        z       iph  \n",
            " -----------------------------------------------------------------------\n",
            "   1      0.00000      0.00000      0.00000   0   1\n",
            "   2      1.00000      0.00000      0.00000   1   1\n",
            "   3      2.00000      0.00000      0.00000   1   1\n",
        )
    );
    Ok(())
}

#[test]
fn nogeom_suppresses_geom_dat_output_like_feff() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
NOGEOM
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
    let outputs = text_outputs(&doc)?;

    assert!(doc.no_geom);
    assert!(outputs.contains_key("atoms.dat"));
    assert!(!outputs.contains_key("geom.dat"));
    Ok(())
}

#[test]
fn writes_compton_inp_from_cards() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
COMPTON
RHOZZP
CGRID 10 32 32 32 120
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        compton_inp_string(&doc)?,
        concat!(
            "run compton module?\n",
            "           1\n",
            "pqmax, npq\n",
            "   5.00000000            1000\n",
            "ns, nphi, nz, nzp\n",
            "  32  32  32 120\n",
            "smax, phimax, zmax, zpmax\n",
            "      0.00000      6.28319      0.00000     10.00000\n",
            "jpq? rhozzp? force_recalc_jzzp?\n",
            " T T F\n",
            "window_type (0=Step, 1=Hann), window_cutoff\n",
            "           1   0.00000000    \n",
            "temperature (in eV)\n",
            "      0.00000\n",
            "set_chemical_potential? chemical_potential(eV)\n",
            " F   0.00000000    \n",
            "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?\n",
            " F F F F F\n",
            "qhat_x qhat_y qhat_z\n",
            "   0.0000000000000000        0.0000000000000000        1.0000000000000000     \n",
        )
    );
    Ok(())
}

#[test]
fn writes_compton_aliases_into_compton_inp() -> Result<()> {
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
    let compton_text = compton_inp_string(&doc)?;
    let compton = crate::ComptonInput::parse_str("compton.inp", &compton_text)?;

    assert!(compton.run);
    assert_eq!(compton.momentum.pqmax, 7.0);
    assert_eq!(compton.momentum.npq, 300);
    assert_eq!(compton.grid.ns, 20);
    assert_eq!(compton.grid.nphi, 21);
    assert_eq!(compton.grid.nz, 22);
    assert_eq!(compton.grid.nzp, 23);
    assert_eq!(compton.limits.zpmax, 12.0);
    assert!(compton.switches.jpq);
    assert!(compton.switches.rhozzp);
    assert!(compton.switches.force_recalc_jzzp);
    assert_eq!(crate::compton_input_string(&compton)?, compton_text);
    Ok(())
}

#[test]
fn writes_pot_inp_for_copper_defaults() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
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
    let pot = pot_inp_string(&doc)?;

    assert!(pot.contains("   1   1   1   1   0   0   0   0  11\n"));
    assert!(pot.contains("      1.72919      0.05000      0.00000    -40.00000"));
    assert!(pot.contains("   29    2        1.0000000000"));
    Ok(())
}

#[test]
fn writes_block_alias_rows_into_structure_outputs() -> Result<()> {
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
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

    assert_eq!(pot.control.nph, 1);
    assert_eq!(pot.potentials.len(), 2);
    assert_eq!(pot.potentials[0].z, 29);
    assert_eq!(atoms.atoms.len(), 2);
    assert_eq!(atoms.atoms[1].iph, 1);
    Ok(())
}

#[test]
fn writes_cif_equivalence_two_into_structure_outputs() -> Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
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
    )
    .map_err(|source| IoError::io(&cif_path, source))?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(
        &input_path,
        r#"
CIF three-site.cif
TARGET 3
EQUIVALENCE 2
FMS 4.0
EDGE K
XANES
END
"#,
    )
    .map_err(|source| IoError::io(&input_path, source))?;

    let input = FeffInput::parse_file(&input_path)?;
    let doc = FeffDocument::from_input(&input)?;
    let pot = crate::PotInput::parse_str("pot.inp", &pot_inp_string(&doc)?)?;
    let atoms = crate::AtomsDat::parse_str("atoms.dat", &atoms_dat_string(&doc)?)?;

    assert_eq!(pot.control.nph, 2);
    assert_eq!(pot.potentials.len(), 3);
    assert_eq!(pot.potentials[0].z, 8);
    assert_eq!(pot.potentials[1].z, 1);
    assert_eq!(pot.potentials[1].xnatph, 200.0);
    assert_eq!(pot.potentials[2].z, 8);
    assert!(atoms.atoms.iter().any(|atom| atom.iph == 1));
    assert!(atoms.atoms.iter().all(|atom| atom.iph <= 2));
    Ok(())
}

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

#[test]
fn writes_rdinp_log_dat_for_copper_exafs_summary() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
DEBYE 190 315 0
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
LDOS -30 20 0.1
EXAFS 20.0
RPATH 5.5
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
    let log = rdinp_log_dat_string(&doc)?;

    assert_eq!(
        log,
        concat!(
            "Launching FEFF version FEFF 10.0.0\n",
            "Core hole lifetime is   1.729 eV.\n",
            "Your calculation:\n",
            " Cu crystal\n",
            "Cu K edge EXAFS using FSR corehole.\n",
            "Using:     * Debye-Waller factors\n",
            "Using cards:   ATOMS CONTROL TITLE RPATH DEBYE PRINT POTENTIALS EXAFS EDGE LDOS S02\n",
            "\n",
        )
    );
    assert_eq!(parse_log_dat(&log)?.features, vec!["Debye-Waller factors"]);
    Ok(())
}

#[test]
fn writes_rdinp_stdout_only_absorber_spin_default_diagnostic() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Gd_L1 hcp
XMCD
EDGE L1
SPIN 1
POTENTIALS
0 64 Gd
1 64 Gd
ATOMS
0.0 0.0 0.0 0 Gd0
1.0 0.0 0.0 1 Gd1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;
    let log = rdinp_log_dat_string(&doc)?;
    let stdout = rdinp_stdout_string(&doc)?;

    assert!(!log.contains("\n           1\n"));
    assert!(stdout.contains("Core hole lifetime is   5.533 eV.\n           1\n"));
    assert!(stdout.contains("No spin set in POTENTIALS card. Using default spins:\n"));
    Ok(())
}

#[test]
fn writes_rdinp_error_log_for_highz_template_failure() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE test_element
HIGHZ
POTENTIALS
       0    XXX   Te
END
"#,
    )?;
    let error = FeffDocument::from_input(&input)
        .err()
        .ok_or_else(|| IoError::Parse {
            path: "feff.inp".into(),
            line: 0,
            message: "HIGHZ template should fail".to_string(),
        })?;

    assert_eq!(
        rdinp_error_log_string(&input, &error)?,
        concat!(
            "Launching FEFF version FEFF 10.0.0\n",
            "Using finite nucleus.\n",
            " Error reading input, bad line follows:\n",
            " 0    XXX   Te\n",
            "RDINP fatal error.\n",
        )
    );
    Ok(())
}

#[test]
fn writes_legacy_rdinp_error_logs_for_blank_context_cards() -> Result<()> {
    let cases = [
        (
            "HOLE",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                " Use NOHOLE to calculate without core hole.  Only ihole greater than zero are allowed.\n",
                "RDINP\n",
            ),
        ),
        (
            "OVERLAP",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                " Cannot use ATOMS and OVERLAP in the same feff.inp.\n",
                "RDINP\n",
            ),
        ),
        (
            "RCONV",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                " RCONV\n",
                " RCONV\n",
                " Token        0\n",
                " Keyword unrecognized.\n",
                " See FEFF document -- some old features are no longer available.\n",
                "RDINP-2\n",
            ),
        ),
        (
            "BAND",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "BANDSTRUCTURE card is experimental.\n",
                "BANDSTRUCTURE requires at least: emin  emax  estep  ikpath\n",
                "\n",
            ),
        ),
        (
            "COORDINATES",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Attempt to enter funky lattice coordinates.\n",
                "Please stick to one of the formats described in the manual.\n",
                "Exiting now.\n",
            ),
        ),
        (
            "MDFF",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "NRIXS type MDFF calculation selected - summed over all q,q' pairs.\n",
                "ERROR - the selected MDFF option is only available with the NRIXS card.\n",
                "RDINP\n",
            ),
        ),
        ("SCREEN", "Launching FEFF version FEFF 10.0.0\n"),
        (
            "SCXC",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Error: iscfxc should take one of the values 11 for vBH, 12 for PZ, 21 for PDW, or 22 for KSDT ... stopping\n",
            ),
        ),
        (
            "LJMAX",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Core hole lifetime is   1.729 eV.\n",
            ),
        ),
        (
            "CGRID",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                "Core hole lifetime is   1.729 eV.\n",
            ),
        ),
        (
            "ELNES",
            concat!(
                "Launching FEFF version FEFF 10.0.0\n",
                " Error reading input, bad line follows:\n",
                " POTENTIALS\n",
                "RDINP fatal error.\n",
            ),
        ),
    ];

    for (card, expected) in cases {
        let input = FeffInput::parse_str("feff.inp", &context_card_input(card))?;
        let error = FeffDocument::from_input(&input)
            .err()
            .ok_or_else(|| IoError::Parse {
                path: "feff.inp".into(),
                line: 0,
                message: format!("{card} should fail in context audit"),
            })?;

        assert_eq!(rdinp_error_log_string(&input, &error)?, expected, "{card}");
    }
    Ok(())
}

fn context_card_input(card: &str) -> String {
    let mut input = String::new();
    input.push_str("TITLE Context audit\n");
    if card != "EDGE" {
        input.push_str("EDGE K\n");
    }
    input.push_str("EXAFS 20\n");
    input.push_str(card);
    input.push('\n');
    input.push_str("POTENTIALS\n0 29 Cu\nATOMS\n0.0 0.0 0.0 0 Cu\nEND\n");
    input
}

#[test]
fn writes_dmdw_inp_for_dynamical_matrix_debye() -> Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
    let input_path = temp.path().join("feff.inp");
    std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
        .map_err(|source| IoError::io("feff.dym", source))?;
    let input = FeffInput::parse_str(
        &input_path,
        r#"
DEBYE 450 315 5 feff.dym 6 0 1
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        dmdw_inp_string(&doc)?,
        "   1\n   6\n   1    450.000\n   0\nfeff.dym\n   1\n   2   1   0           2.08\n"
    );
    Ok(())
}

#[test]
fn copies_dym_file_for_dynamical_matrix_debye() -> Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
    let input_path = temp.path().join("feff.inp");
    let dym_text = minimal_dym_text();
    std::fs::write(temp.path().join("custom.dym"), dym_text)
        .map_err(|source| IoError::io("custom.dym", source))?;
    let input = FeffInput::parse_str(
        &input_path,
        r#"
DEBYE 450 315 5 custom.dym 6 0 1
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    let dym_input = doc.dym_input.as_ref().ok_or_else(|| IoError::Parse {
        path: input_path.clone(),
        line: 0,
        message: "missing DMDW auxiliary".to_string(),
    })?;
    assert_eq!(dym_input.output_name, "custom.dym");
    assert_eq!(dym_input.text, dym_text);
    assert_eq!(
        text_outputs(&doc)?.get("custom.dym").map(String::as_str),
        Some(dym_text)
    );
    Ok(())
}

#[test]
fn copies_spring_inp_for_emm_and_recursion_debye() -> Result<()> {
    let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
    let input_path = temp.path().join("feff.inp");
    let spring_text = concat!(
        "* res wmax dosfit acut\n",
        " VDOS 0.03 0.5 1\n",
        "\n",
        " STRETCHES\n",
        " 0 1 27.9 2.\n",
    );
    std::fs::write(temp.path().join("spring.inp"), spring_text)
        .map_err(|source| IoError::io("spring.inp", source))?;
    for idwopt in [1, 2] {
        let input = FeffInput::parse_str(&input_path, &format!("DEBYE 450 315 {idwopt}\nEND\n"))?;
        let doc = FeffDocument::from_input(&input)?;

        assert_eq!(doc.spring_input_text.as_deref(), Some(spring_text));
        assert_eq!(
            text_outputs(&doc)?.get("spring.inp").map(String::as_str),
            Some(spring_text)
        );
    }
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

#[test]
fn writes_rixs_inp_defaults() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE K
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert_eq!(
        rixs_inp_string(&doc)?,
        concat!(
            " m_run\n",
            "           0\n",
            " gam_ch, gam_exp(1), gam_exp(2)\n",
            "        0.0001350512        0.0001350512        0.0001350512\n",
            " EMinI, EMaxI, EMinF, EMaxF\n",
            "        0.0000000000        0.0000000000        0.0000000000        0.0000000000\n",
            " xmu\n",
            "  -367493090.02742821     \n",
            " Readpoles, SkipCalc, MBConv, ReadSigma\n",
            " T F F F\n",
            " nEdges\n",
            "           1\n",
            " Edge           1\n",
            " K\n",
        )
    );
    Ok(())
}
