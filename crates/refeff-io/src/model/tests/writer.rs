use super::*;
use std::path::PathBuf;

/// Read a `feff.inp` file shipped with the FEFF10 examples, relative to the
/// workspace root.
fn read_example(relative_path: &str) -> anyhow::Result<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../feff10/examples")
        .join(relative_path);
    std::fs::read_to_string(&path)
        .with_context(|| format!("reading example input {}", path.display()))
}

/// Fields that record which literal card spellings were used
/// (`active_cards`/`input_cards`) and the originating file path are
/// bookkeeping, not modeled physics: [`feff_inp_string`] is free to pick a
/// different (but equivalent) canonical card for the same field, so exclude
/// them before comparing two documents for round-trip equivalence.
fn normalize(mut document: FeffDocument) -> FeffDocument {
    document.source = PathBuf::from("normalized");
    document.active_cards.clear();
    document.input_cards.clear();
    document
}

fn assert_round_trips(relative_path: &str) -> anyhow::Result<()> {
    let text = read_example(relative_path)?;
    let input = FeffInput::parse_str(relative_path, &text)?;
    let original = FeffDocument::from_input(&input)
        .with_context(|| format!("parsing original example input {relative_path}"))?;

    let rendered = feff_inp_string(&original)
        .with_context(|| format!("rendering feff.inp text for {relative_path}"))?;
    let rendered_input = FeffInput::parse_str("rendered-feff.inp", &rendered)
        .with_context(|| format!("re-parsing rendered feff.inp for {relative_path}\n{rendered}"))?;
    let round_tripped = FeffDocument::from_input(&rendered_input)
        .with_context(|| format!("re-extracting FeffDocument for {relative_path}\n{rendered}"))?;

    ensure!(
        normalize(original) == normalize(round_tripped),
        "round-trip mismatch for {relative_path}; rendered feff.inp was:\n{rendered}"
    );
    Ok(())
}

#[test]
fn round_trips_exafs_cu_example() -> anyhow::Result<()> {
    assert_round_trips("EXAFS/Cu/feff.inp")
}

#[test]
fn round_trips_xanes_gecl4_example() -> anyhow::Result<()> {
    assert_round_trips("XANES/GeCl_4/feff.inp")
}

#[test]
fn round_trips_exafs_gecl4_example() -> anyhow::Result<()> {
    assert_round_trips("EXAFS/GeCl_4/feff.inp")
}

#[test]
fn renders_canonical_cards_for_a_hand_built_document() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE Cu crystal
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
SCF 5.0 0 40 0.3
EXCHANGE 0 1.0 2.0
EXAFS 20.0
FMS 4.0 1 0 0.002 0.003 20.0
DEBYE 190 315 0
RPATH 5.5
DIMS 100 4
LDOS -30 20 0.1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.0 0.0 0.0 1 Cu1 1.0 1
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;

    let rendered = feff_inp_string(&document)?;
    ensure!(rendered.contains("TITLE Cu crystal"), "{rendered}");
    ensure!(rendered.contains("EDGE K"), "{rendered}");
    ensure!(rendered.contains("S02 1"), "{rendered}");
    ensure!(rendered.contains("POTENTIALS"), "{rendered}");
    ensure!(rendered.contains("ATOMS"), "{rendered}");
    ensure!(rendered.trim_end().ends_with("END"), "{rendered}");

    let rendered_input = FeffInput::parse_str("rendered.inp", &rendered)?;
    let round_tripped = FeffDocument::from_input(&rendered_input)?;
    ensure!(
        normalize(document) == normalize(round_tripped),
        "round-trip mismatch; rendered feff.inp was:\n{rendered}"
    );
    Ok(())
}

#[test]
fn write_feff_inp_writes_a_reparsable_file() -> anyhow::Result<()> {
    let input = FeffInput::parse_str(
        "feff.inp",
        r#"
TITLE minimal
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
    )?;
    let document = FeffDocument::from_input(&input)?;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("feff.inp");
    write_feff_inp(&path, &document)?;

    let round_tripped_input = FeffInput::parse_file(&path)?;
    let round_tripped = FeffDocument::from_input(&round_tripped_input)?;
    ensure!(
        normalize(document) == normalize(round_tripped),
        "file round-trip mismatch for {}",
        path.display()
    );
    Ok(())
}
