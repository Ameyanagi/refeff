use super::*;

#[test]
fn matches_generated_reference_rdinp_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated reference output comparison; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input in inputs {
        let output_dir = input
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input)
            .with_context(|| format!("failed to parse {}", input.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input.display()))?;
        let outputs = rdinp::text_outputs(&document)
            .with_context(|| format!("failed to render rdinp outputs for {}", input.display()))?;
        ensure_supported_reference_rdinp_outputs_are_rendered(output_dir, outputs.keys())?;

        let standalone_ldos_reference = is_ldos_reference_dir(output_dir);
        let generated_periodic_structure = parsed.card("CIF").is_some()
            || (parsed.card("RECIPROCAL").is_some() && parsed.card("LATTICE").is_some());
        for (name, actual) in outputs {
            let expected_path = output_dir.join(name.as_ref());
            if !expected_path.exists() {
                if standalone_ldos_reference {
                    continue;
                }
                bail!(
                    "unexpected generated rdinp output {name} for {}",
                    input.display()
                );
            }
            let expected = std::fs::read_to_string(&expected_path)
                .with_context(|| format!("failed to read {}", expected_path.display()))?;

            if standalone_ldos_reference && name.as_ref() == "ldos.inp" {
                // LDOS category fixtures keep standalone LDOS module inputs in
                // ldos.inp. The module-input roundtrip test covers those files;
                // this test compares only RDINP-owned outputs.
                continue;
            }

            if generated_periodic_structure && matches!(name.as_ref(), "atoms.dat" | "geom.dat") {
                // Periodic equal-distance shells are sensitive to FEFF's compiler-level
                // floating-point tie order. Keep this semantic until that ordering is
                // reproduced byte-for-byte for CIF and reciprocal LATTICE expansion.
                ensure_periodic_structure_matches(&name, &expected_path, &expected, &actual)
                    .with_context(|| {
                        format!("{name} structural mismatch for {}", input.display())
                    })?;
                compared += 1;
                continue;
            }

            ensure!(
                actual == expected,
                "{name} mismatch for {}: {}",
                input.display(),
                first_mismatch(&expected, &actual)
            );
            compared += 1;
        }
    }

    ensure!(compared > 0, "no generated rdinp outputs found");
    Ok(())
}

fn is_ldos_reference_dir(path: &Path) -> bool {
    path.parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "LDOS")
}

#[test]
fn matches_generated_reference_rdinp_log_dat_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated log.dat comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("log.dat");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input_path.display()))?;
        let actual = rdinp::rdinp_log_dat_string(&document)
            .with_context(|| format!("failed to render log.dat for {}", input_path.display()))?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "log.dat mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    ensure!(compared > 0, "no generated log.dat examples found");
    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_error_log_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated rdinp error-log comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if !output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("log.dat");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let error = FeffDocument::from_input(&parsed).err().with_context(|| {
            format!(
                "expected rdinp extraction to fail for {}",
                input_path.display()
            )
        })?;
        let actual = rdinp::rdinp_error_log_string(&parsed, &error).with_context(|| {
            format!(
                "failed to render rdinp error log for {}",
                input_path.display()
            )
        })?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "rdinp error log mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    if compared == 0 {
        eprintln!("skipping generated rdinp error-log comparison; no failing FEFF golden inputs");
        return Ok(());
    }

    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_error_sentinel_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated rdinp error-sentinel comparison; reference-work/golden not found"
        );
        return Ok(());
    }

    let mut sentinels = Vec::new();
    collect_named_files(&golden_dir, ".feff.error", &mut sentinels)?;
    sentinels.sort();
    if sentinels.is_empty() {
        eprintln!(
            "skipping generated rdinp error-sentinel comparison; no FEFF sentinel files found"
        );
        return Ok(());
    }

    let actual = rdinp::rdinp_error_sentinel_string();
    for path in sentinels {
        let expected =
            std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        ensure!(
            actual.as_bytes() == expected.as_slice(),
            ".feff.error mismatch for {}",
            path.display()
        );
    }

    Ok(())
}

#[test]
fn matches_generated_reference_rdinp_stdout_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated rdinp stdout comparison; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input_path in inputs {
        let output_dir = input_path
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input_path.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }
        let expected_path = output_dir.join("rdinp.stdout");
        if !expected_path.exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input_path)
            .with_context(|| format!("failed to parse {}", input_path.display()))?;
        let document = FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input_path.display()))?;
        let actual = rdinp::rdinp_stdout_string(&document).with_context(|| {
            format!("failed to render rdinp stdout for {}", input_path.display())
        })?;
        let expected = std::fs::read_to_string(&expected_path)
            .with_context(|| format!("failed to read {}", expected_path.display()))?;

        ensure!(
            actual == expected,
            "rdinp stdout mismatch for {}",
            input_path.display()
        );
        compared += 1;
    }

    ensure!(compared > 0, "no generated rdinp stdout examples found");
    Ok(())
}
