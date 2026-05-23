use super::*;

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
