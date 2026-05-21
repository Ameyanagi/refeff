use super::*;

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
