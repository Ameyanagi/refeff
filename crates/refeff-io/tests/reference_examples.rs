use std::path::{Path, PathBuf};

use refeff_io::{FeffDocument, FeffInput};

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
