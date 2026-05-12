use std::path::{Path, PathBuf};

use refeff_io::{FeffDocument, FeffInput, rdinp};

#[test]
fn parses_all_local_reference_examples_when_present() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples");
    if !examples_dir.exists() {
        eprintln!("skipping local reference example parse; feff10/examples not found");
        return;
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs);
    inputs.sort();
    assert_eq!(inputs.len(), 44, "unexpected local FEFF example count");

    for input in inputs {
        let parsed = FeffInput::parse_file(&input).unwrap_or_else(|err| {
            panic!("failed to parse {}: {err}", input.display());
        });
        FeffDocument::from_input(&parsed).unwrap_or_else(|err| {
            panic!("failed to extract {}: {err}", input.display());
        });
    }
}

#[test]
fn matches_generated_reference_rdinp_outputs_when_present() {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-work/golden");
    if !golden_dir.exists() {
        eprintln!(
            "skipping generated reference output comparison; reference-work/golden not found"
        );
        return;
    }

    let mut inputs = Vec::new();
    collect_feff_inputs(&golden_dir, &mut inputs);
    inputs.sort();
    assert!(!inputs.is_empty(), "no generated FEFF golden inputs found");

    let mut compared = 0_usize;
    for input in inputs {
        let output_dir = input.parent().expect("golden input has parent");
        if output_dir.join(".feff.error").exists() {
            continue;
        }

        let parsed = FeffInput::parse_file(&input).unwrap_or_else(|err| {
            panic!("failed to parse {}: {err}", input.display());
        });
        let document = FeffDocument::from_input(&parsed).unwrap_or_else(|err| {
            panic!("failed to extract {}: {err}", input.display());
        });
        let outputs = rdinp::text_outputs(&document).unwrap_or_else(|err| {
            panic!(
                "failed to render rdinp outputs for {}: {err}",
                input.display()
            );
        });

        for (name, actual) in outputs {
            let expected_path = output_dir.join(name);
            if !expected_path.exists() {
                continue;
            }
            let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|err| {
                panic!("failed to read {}: {err}", expected_path.display());
            });

            assert_eq!(actual, expected, "{name} mismatch for {}", input.display());
            compared += 1;
        }
    }

    assert!(compared > 0, "no generated rdinp outputs found");
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read example dir") {
        let entry = entry.expect("read example entry");
        let path = entry.path();
        if path.is_dir() {
            collect_feff_inputs(&path, inputs);
        } else if path.file_name().is_some_and(|name| name == "feff.inp") {
            inputs.push(path);
        }
    }
}
