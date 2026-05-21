use super::*;

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
