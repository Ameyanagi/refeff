use super::*;

#[test]
fn parses_all_local_reference_examples_when_present() -> anyhow::Result<()> {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples");
    if !examples_dir.exists() {
        eprintln!("skipping local reference example parse; feff10/examples not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();
    ensure!(
        inputs.len() == 44,
        "unexpected local FEFF example count: {}",
        inputs.len()
    );

    for input in inputs {
        let parsed = FeffInput::parse_file(&input)
            .with_context(|| format!("failed to parse {}", input.display()))?;
        if input
            .components()
            .any(|component| component.as_os_str() == "HIGHZ")
        {
            let error = FeffDocument::from_input(&parsed).err().with_context(|| {
                format!("HIGHZ template should be invalid: {}", input.display())
            })?;
            ensure!(
                error.to_string().contains("XXX"),
                "unexpected HIGHZ template error for {}: {error}",
                input.display()
            );
        } else {
            FeffDocument::from_input(&parsed)
                .with_context(|| format!("failed to extract {}", input.display()))?;
        }
    }

    let mut spring_inputs = Vec::new();
    collect_named_files(&examples_dir, "spring.inp", &mut spring_inputs)?;
    spring_inputs.sort();
    for input in &spring_inputs {
        let text = std::fs::read_to_string(input)
            .with_context(|| format!("failed to read {}", input.display()))?;
        SpringInput::parse_str(&text)
            .with_context(|| format!("failed to parse {}", input.display()))?;
    }
    ensure!(
        !spring_inputs.is_empty(),
        "no local FEFF spring.inp examples found"
    );

    let mpse_dat = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/src/SELF/mpse.dat");
    if mpse_dat.exists() {
        let text = std::fs::read_to_string(&mpse_dat)
            .with_context(|| format!("failed to read {}", mpse_dat.display()))?;
        parse_mpse_dat(&text).with_context(|| format!("failed to parse {}", mpse_dat.display()))?;
    }
    Ok(())
}
