use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use refeff_io::{
    ComptonInput, CrpaInput, EelsInput, FeffDocument, FeffInput, Ff2xInput, FmsInput, GenfmtInput,
    GlobalInput, HubbardInput, LdosInput, PathsInput, PotInput, RixsInput, ScreenInput,
    SfconvInput, XsphInput, rdinp,
};

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
        FeffDocument::from_input(&parsed)
            .with_context(|| format!("failed to extract {}", input.display()))?;
    }
    Ok(())
}

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

        for (name, actual) in outputs {
            let expected_path = output_dir.join(name);
            if !expected_path.exists() {
                continue;
            }
            let expected = std::fs::read_to_string(&expected_path)
                .with_context(|| format!("failed to read {}", expected_path.display()))?;

            ensure!(
                actual == expected,
                "{name} mismatch for {}",
                input.display()
            );
            compared += 1;
        }
    }

    ensure!(compared > 0, "no generated rdinp outputs found");
    Ok(())
}

#[test]
fn parses_generated_reference_handoff_outputs_when_present() -> anyhow::Result<()> {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!("skipping generated handoff parser coverage; reference-work/golden not found");
        return Ok(());
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs)?;
    inputs.sort();
    ensure!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut parsed_count = 0_usize;
    for input in inputs {
        let output_dir = input
            .parent()
            .with_context(|| format!("golden input has no parent: {}", input.display()))?;
        if output_dir.join(".feff.error").exists() {
            continue;
        }

        parsed_count += parse_handoff_file(output_dir, "global.inp", GlobalInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "compton.inp", ComptonInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "crpa.inp", CrpaInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "eels.inp", EelsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "hubbard.inp", HubbardInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "pot.inp", PotInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "screen.inp", ScreenInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "sfconv.inp", SfconvInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "xsph.inp", XsphInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "fms.inp", FmsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "ldos.inp", LdosInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "paths.inp", PathsInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "genfmt.inp", GenfmtInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "ff2x.inp", Ff2xInput::parse_str)?;
        parsed_count += parse_handoff_file(output_dir, "rixs.inp", RixsInput::parse_str)?;
    }

    ensure!(parsed_count > 0, "no generated handoff files parsed");
    Ok(())
}

fn parse_handoff_file<T>(
    output_dir: &Path,
    name: &str,
    parse: impl FnOnce(PathBuf, &str) -> refeff_io::Result<T>,
) -> anyhow::Result<usize> {
    let path = output_dir.join(name);
    if !path.exists() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse(path.clone(), &text).with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(1)
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_feff_inputs(&path, inputs)?;
        } else if path.file_name().is_some_and(|name| name == "feff.inp") {
            inputs.push(path);
        }
    }
    Ok(())
}
