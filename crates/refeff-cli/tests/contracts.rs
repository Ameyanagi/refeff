//! Exercise real process boundaries; all three relevant directories are disposable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::process::{Command, Output};

const INPUT: &str =
    "TITLE contract\nEDGE K\nCONTROL 0 0 0 0 0 0\nPOTENTIALS\n0 29 Cu\nATOMS\n0 0 0 0 Cu\nEND\n";

fn invoke(binary: &str, cwd: &Path, args: &[&str]) -> Output {
    Command::new(binary)
        .current_dir(cwd)
        .env_remove("REFEFF_THREADS")
        .args(args)
        .output()
        .expect("start CLI")
}

#[test]
fn module_directory_and_json_are_explicit() {
    let root = tempfile::tempdir().unwrap();
    let cwd = root.path().join("process");
    let work = root.path().join("handoffs");
    let sources = root.path().join("sources");
    for dir in [&cwd, &work, &sources] {
        std::fs::create_dir(dir).unwrap();
    }
    let input = sources.join("custom.inp");
    std::fs::write(&input, INPUT).unwrap();
    let output = invoke(
        env!("CARGO_BIN_EXE_refeff"),
        &cwd,
        &[
            "--json",
            "-C",
            work.to_str().unwrap(),
            "module",
            "rdinp",
            "-i",
            input.to_str().unwrap(),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("one JSON document");
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["data"]["module"], "rdinp");
    assert!(work.join("pot.inp").exists());
    assert!(!cwd.join("pot.inp").exists());
    assert!(!sources.join("pot.inp").exists());
}

#[test]
fn module_without_dir_uses_input_parent() {
    let root = tempfile::tempdir().unwrap();
    let inputs = root.path().join("inputs");
    std::fs::create_dir(&inputs).unwrap();
    let input = inputs.join("feff.inp");
    std::fs::write(&input, INPUT).unwrap();
    let output = invoke(
        env!("CARGO_BIN_EXE_refeff"),
        root.path(),
        &["module", "rdinp", "-i", input.to_str().unwrap()],
    );
    assert!(output.status.success());
    assert!(inputs.join("pot.inp").exists());
    assert!(!root.path().join("pot.inp").exists());
}

#[test]
fn check_rejects_semantic_errors_without_writes() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("feff.inp");
    for text in [
        String::new(),
        INPUT.replace("EDGE K", "EDGE invalid"),
        INPUT.replace("EDGE K", "EDGE K\nS02 NaN"),
        INPUT.replace("0 0 0 0 Cu", "0 0 0 99 Cu"),
    ] {
        std::fs::write(&input, text).unwrap();
        let output = invoke(
            env!("CARGO_BIN_EXE_refeff"),
            root.path(),
            &["--json", "check"],
        );
        assert_eq!(
            output.status.code(),
            Some(3),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(json["ok"], false);
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
    }
}

#[test]
fn syntax_only_accepts_partial_inputs() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("feff.inp"), "").unwrap();
    assert!(
        invoke(
            env!("CARGO_BIN_EXE_refeff"),
            root.path(),
            &["check", "--syntax-only"]
        )
        .status
        .success()
    );
}

#[test]
fn frontend_names_share_exit_codes() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("feff.inp"), "UNRECOGNIZED_CARD 1\n").unwrap();
    for binary in [env!("CARGO_BIN_EXE_refeff"), env!("CARGO_BIN_EXE_feff")] {
        assert_eq!(
            invoke(binary, root.path(), &["check"]).status.code(),
            Some(3)
        );
    }
}

#[test]
fn quiet_and_json_do_not_mix_human_output() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("feff.inp"), INPUT).unwrap();
    let output = invoke(
        env!("CARGO_BIN_EXE_refeff"),
        root.path(),
        &["--quiet", "check"],
    );
    assert!(output.status.success());
    assert!(output.stdout.is_empty() && output.stderr.is_empty());
}

#[test]
fn usage_errors_are_one_json_document() {
    let root = tempfile::tempdir().unwrap();
    let result = invoke(
        env!("CARGO_BIN_EXE_refeff"),
        root.path(),
        &["--json", "--not-an-option"],
    );
    assert_eq!(result.status.code(), Some(2));
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["ok"], false);
    assert_eq!(value["error"]["code"], "usage");
}

#[test]
fn informational_json_is_also_one_document() {
    let root = tempfile::tempdir().unwrap();
    for args in [
        vec!["--json", "--help"],
        vec!["--json", "--version"],
        vec!["--json", "completions", "bash"],
    ] {
        let result = invoke(env!("CARGO_BIN_EXE_refeff"), root.path(), &args);
        assert!(result.status.success());
        let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(value["ok"], true);
    }
}
#[test]
fn init_refuses_overwrite_and_plan_is_read_only() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        invoke(env!("CARGO_BIN_EXE_refeff"), root.path(), &["init"])
            .status
            .success()
    );
    let original = std::fs::read(root.path().join("feff.inp")).unwrap();
    assert!(
        !invoke(env!("CARGO_BIN_EXE_refeff"), root.path(), &["init"])
            .status
            .success()
    );
    assert_eq!(
        std::fs::read(root.path().join("feff.inp")).unwrap(),
        original
    );
    let result = invoke(
        env!("CARGO_BIN_EXE_refeff"),
        root.path(),
        &["plan", "-o", "never-created", "--json"],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(
        value["data"]["stages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|stage| stage["name"] == "ff2x")
    );
    assert!(!root.path().join("never-created").exists());
}
