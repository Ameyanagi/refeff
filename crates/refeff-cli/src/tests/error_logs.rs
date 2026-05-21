use super::*;

#[test]
fn failed_rdinp_writes_feff_style_error_log() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("feff.inp");
    let output = temp.path().join("out");
    write_highz_template_input(&input)?;

    let error = execute_rdinp(&input, &output)
        .err()
        .context("HIGHZ template should fail during rdinp extraction")?;

    assert!(error.to_string().contains("XXX"));
    assert_eq!(
        std::fs::read_to_string(output.join("log.dat"))?,
        concat!(
            "Launching FEFF version FEFF 10.0.0\n",
            "Using finite nucleus.\n",
            " Error reading input, bad line follows:\n",
            " 0    XXX   Te\n",
            "RDINP fatal error.\n",
        )
    );
    assert_eq!(
        std::fs::read_to_string(output.join(".feff.error"))?,
        rdinp::rdinp_error_sentinel_string()
    );
    Ok(())
}
