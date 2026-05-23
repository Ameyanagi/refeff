use super::*;

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
