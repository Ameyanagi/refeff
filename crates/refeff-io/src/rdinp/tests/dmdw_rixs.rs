use super::*;

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

#[test]
fn writes_rixs_inp_optional_switches_from_rixs_card() -> Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
EDGE L3 L2
RIXS 0.1 0.2 3.0 F T T T
END
"#,
    )?;
    let doc = FeffDocument::from_input(&input)?;

    assert!(doc.rixs.run);
    assert!(!doc.rixs.read_poles);
    assert!(doc.rixs.skip_calc);
    assert!(doc.rixs.mbconv);
    assert!(doc.rixs.read_sigma);

    let rixs = rixs_inp_string(&doc)?;
    assert!(rixs.contains(" Readpoles, SkipCalc, MBConv, ReadSigma\n F T T T\n"));
    assert!(rixs.contains(" L3\n Edge           2\n L2\n"));
    Ok(())
}
