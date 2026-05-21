use super::*;

#[test]
fn rejects_incomplete_polarization_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
POLARIZATION 1.0 0.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete POLARIZATION should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("POLARIZATION requires x, y, and z"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_ellipticity_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
XANES
ELLIPTICITY 0.25 0.0 1.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete ELLIPTICITY should be rejected")?;
    ensure!(
        error
            .to_string()
            .contains("ELLIPTICITY requires ellipticity and incident direction"),
        "unexpected error: {error}"
    );
    Ok(())
}

#[test]
fn rejects_incomplete_fprime_like_feff() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
FPRIME -5.0
END
"#,
    )?;

    let error = FeffDocument::from_input(&input)
        .err()
        .context("incomplete FPRIME should be rejected")?;
    ensure!(
        error.to_string().contains("FPRIME requires emin and emax"),
        "unexpected error: {error}"
    );
    Ok(())
}
