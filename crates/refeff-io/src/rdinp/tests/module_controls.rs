use super::*;

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
