//! Audits the checked FEFF10 production-scope contract.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const SCOPE_MANIFEST: &str = "compatibility/feff10.json";
const UPSTREAM_PRODUCTION_MAKEFILE: &str = "feff10/src/Makefile";

#[derive(Debug, Deserialize)]
struct ScopeManifest {
    schema_version: u32,
    upstream: Upstream,
    main_binaries: Vec<String>,
    production_executables: Vec<ExecutableMapping>,
    rust_extensions: Vec<String>,
    excluded_tools: Vec<String>,
    required_card_token_ids: Vec<u32>,
    stock_workflows: Vec<String>,
    highz: HighzScope,
}

#[derive(Debug, Deserialize)]
struct Upstream {
    repository: String,
    revision: String,
    reference_compiler: String,
}

#[derive(Debug, Deserialize)]
struct ExecutableMapping {
    feff10: String,
    rust: String,
}

#[derive(Debug, Deserialize)]
struct HighzScope {
    first_atomic_number: u32,
    last_atomic_number: u32,
    known_reference_failures: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ScopeAuditReport {
    schema_version: u32,
    upstream_repository: String,
    upstream_revision: String,
    reference_compiler: String,
    production_executables: usize,
    rust_extensions: usize,
    excluded_tools: usize,
    card_tokens: usize,
    stock_workflows: usize,
    highz_cases: usize,
    missing_binaries: Vec<String>,
    missing_upstream_executables: Vec<String>,
    unexpected_card_ids: Vec<u32>,
    missing_card_ids: Vec<u32>,
    missing_workflows: Vec<String>,
    unexpected_workflows: Vec<String>,
    revision_mismatch: Option<String>,
    advertised_excluded_tools: Vec<String>,
}

impl ScopeAuditReport {
    fn passed(&self) -> bool {
        self.missing_binaries.is_empty()
            && self.missing_upstream_executables.is_empty()
            && self.unexpected_card_ids.is_empty()
            && self.missing_card_ids.is_empty()
            && self.missing_workflows.is_empty()
            && self.unexpected_workflows.is_empty()
            && self.revision_mismatch.is_none()
            && self.advertised_excluded_tools.is_empty()
    }
}

pub(crate) fn print_scope_audit(detail: bool, json_out: Option<&Path>) -> Result<()> {
    let root = workspace_root()?;
    let report = scope_audit(&root)?;
    println!(
        "FEFF10 scope: executables={} extensions={} cards={} workflows={} HIGHZ={} passed={}",
        report.production_executables,
        report.rust_extensions,
        report.card_tokens,
        report.stock_workflows,
        report.highz_cases,
        report.passed()
    );
    if detail {
        println!(
            "reference: {} @ {} ({})",
            report.upstream_repository, report.upstream_revision, report.reference_compiler
        );
        print_items("missing binaries", &report.missing_binaries);
        print_items(
            "unmapped upstream production executables",
            &report.missing_upstream_executables,
        );
        print_items("missing workflows", &report.missing_workflows);
        print_items("unexpected workflows", &report.unexpected_workflows);
        print_items(
            "advertised excluded tools",
            &report.advertised_excluded_tools,
        );
        if !report.missing_card_ids.is_empty() {
            println!("missing card ids: {:?}", report.missing_card_ids);
        }
        if !report.unexpected_card_ids.is_empty() {
            println!("unexpected card ids: {:?}", report.unexpected_card_ids);
        }
        if let Some(mismatch) = &report.revision_mismatch {
            println!("reference revision mismatch: {mismatch}");
        }
    }
    if let Some(path) = json_out {
        write_json(path, &report)?;
        println!("wrote scope audit json: {}", path.display());
    }
    anyhow::ensure!(report.passed(), "FEFF10 production-scope audit failed");
    Ok(())
}

fn print_items(label: &str, items: &[String]) {
    if !items.is_empty() {
        println!("{label}: {}", items.join(", "));
    }
}

fn write_json(path: &Path, report: &ScopeAuditReport) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

fn scope_audit(root: &Path) -> Result<ScopeAuditReport> {
    let manifest = read_scope_manifest(root)?;
    anyhow::ensure!(manifest.schema_version == 1, "unsupported scope schema");
    ensure_unique("main binary", &manifest.main_binaries)?;
    ensure_unique("Rust extension", &manifest.rust_extensions)?;
    ensure_unique("excluded tool", &manifest.excluded_tools)?;
    ensure_unique("workflow", &manifest.stock_workflows)?;
    ensure_unique("card token", &manifest.required_card_token_ids)?;

    let cargo = std::fs::read_to_string(root.join("crates/refeff-cli/Cargo.toml"))?;
    let mut required_binaries = manifest.main_binaries.clone();
    required_binaries.extend(
        manifest
            .production_executables
            .iter()
            .map(|mapping| mapping.rust.clone()),
    );
    let missing_binaries = required_binaries
        .into_iter()
        .filter(|name| {
            !cargo.contains(&format!("name = \"{name}\""))
                || !root
                    .join("crates/refeff-cli/src/bin")
                    .join(format!("{name}.rs"))
                    .is_file()
        })
        .collect::<Vec<_>>();

    let upstream_names = manifest
        .production_executables
        .iter()
        .map(|mapping| mapping.feff10.as_str())
        .collect::<Vec<_>>();
    ensure_unique("FEFF10 executable", &upstream_names)?;
    let declared_upstream_executables = upstream_names
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let built_upstream_executables = upstream_production_executables(root)?;
    let missing_upstream_executables =
        omitted_upstream_executables(&built_upstream_executables, &declared_upstream_executables);

    let cards_source = std::fs::read_to_string(root.join("crates/refeff-io/src/model/cards.rs"))?;
    let implemented_card_ids = card_token_ids(&cards_source);
    let required_card_ids = manifest
        .required_card_token_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let missing_card_ids = required_card_ids
        .difference(&implemented_card_ids)
        .copied()
        .collect();
    let unexpected_card_ids = implemented_card_ids
        .difference(&required_card_ids)
        .copied()
        .collect();

    let expected_workflows = manifest
        .stock_workflows
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let actual_workflows = discover_workflows(&root.join("feff10/examples"))?;
    let (missing_workflows, unexpected_workflows) = if actual_workflows.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (
            expected_workflows
                .difference(&actual_workflows)
                .cloned()
                .collect(),
            actual_workflows
                .difference(&expected_workflows)
                .cloned()
                .collect(),
        )
    };

    let revision_mismatch = crate::manifest::feff10_git_rev(&root.join("feff10"))
        .filter(|actual| actual != &manifest.upstream.revision)
        .map(|actual| format!("expected {}, found {actual}", manifest.upstream.revision));
    let cli_source = std::fs::read_to_string(root.join("crates/refeff-cli/src/lib.rs"))?;
    let advertised_excluded_tools = ["Inpgen", "inpgen"]
        .into_iter()
        .filter(|needle| cli_source.contains(needle))
        .map(str::to_string)
        .collect();

    let highz_cases = manifest
        .highz
        .last_atomic_number
        .checked_sub(manifest.highz.first_atomic_number)
        .and_then(|value| value.checked_add(1))
        .context("invalid HIGHZ range")? as usize;
    anyhow::ensure!(
        manifest
            .highz
            .known_reference_failures
            .iter()
            .all(|value| *value >= manifest.highz.first_atomic_number
                && *value <= manifest.highz.last_atomic_number),
        "HIGHZ known failure is outside the configured range"
    );

    Ok(ScopeAuditReport {
        schema_version: manifest.schema_version,
        upstream_repository: manifest.upstream.repository,
        upstream_revision: manifest.upstream.revision,
        reference_compiler: manifest.upstream.reference_compiler,
        production_executables: manifest.production_executables.len(),
        rust_extensions: manifest.rust_extensions.len(),
        excluded_tools: manifest.excluded_tools.len(),
        card_tokens: manifest.required_card_token_ids.len(),
        stock_workflows: manifest.stock_workflows.len(),
        highz_cases,
        missing_binaries,
        missing_upstream_executables,
        unexpected_card_ids,
        missing_card_ids,
        missing_workflows,
        unexpected_workflows,
        revision_mismatch,
        advertised_excluded_tools,
    })
}

fn upstream_production_executables(root: &Path) -> Result<BTreeSet<String>> {
    let path = root.join(UPSTREAM_PRODUCTION_MAKEFILE);
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    makefile_word_variable(&source, "EXECUTABLES")
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn omitted_upstream_executables(
    built: &BTreeSet<String>,
    declared: &BTreeSet<String>,
) -> Vec<String> {
    built.difference(declared).cloned().collect()
}

/// Reads a whitespace-separated Make variable while ignoring commented-out
/// assignments and honoring line continuations. A later `=` assignment
/// replaces an earlier value, while `+=` appends to it.
fn makefile_word_variable(source: &str, variable: &str) -> Result<BTreeSet<String>> {
    let mut logical_line = String::new();
    let mut words = BTreeSet::new();
    let mut found = false;

    for physical_line in source.lines() {
        let uncommented = physical_line
            .split_once('#')
            .map_or(physical_line, |(before, _)| before)
            .trim_end();
        let continued = uncommented.ends_with('\\');
        let fragment = uncommented.strip_suffix('\\').unwrap_or(uncommented).trim();
        if !fragment.is_empty() {
            if !logical_line.is_empty() {
                logical_line.push(' ');
            }
            logical_line.push_str(fragment);
        }
        if continued {
            continue;
        }

        if let Some((append, value)) = makefile_assignment(&logical_line, variable) {
            found = true;
            if !append {
                words.clear();
            }
            for word in value.split_whitespace() {
                anyhow::ensure!(
                    word.chars()
                        .all(|character| character.is_ascii_alphanumeric()
                            || matches!(character, '_' | '-')),
                    "unsupported {variable} token {word:?}"
                );
                words.insert(word.to_owned());
            }
        }
        logical_line.clear();
    }

    if !logical_line.is_empty() {
        if let Some((append, value)) = makefile_assignment(&logical_line, variable) {
            found = true;
            if !append {
                words.clear();
            }
            words.extend(value.split_whitespace().map(str::to_owned));
        }
    }

    anyhow::ensure!(found, "missing active {variable} assignment");
    anyhow::ensure!(!words.is_empty(), "active {variable} assignment is empty");
    Ok(words)
}

fn makefile_assignment<'a>(line: &'a str, variable: &str) -> Option<(bool, &'a str)> {
    let rest = line.trim_start().strip_prefix(variable)?;
    let rest = rest.trim_start();
    if let Some(value) = rest.strip_prefix("+=") {
        Some((true, value.trim()))
    } else if let Some(value) = rest
        .strip_prefix(":=")
        .or_else(|| rest.strip_prefix("?="))
        .or_else(|| rest.strip_prefix('='))
    {
        Some((false, value.trim()))
    } else {
        None
    }
}

fn read_scope_manifest(root: &Path) -> Result<ScopeManifest> {
    let path = root.join(SCOPE_MANIFEST);
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&source).with_context(|| format!("failed to parse {}", path.display()))
}

fn ensure_unique<T>(label: &str, values: &[T]) -> Result<()>
where
    T: Ord + std::fmt::Debug,
{
    let mut seen = BTreeSet::new();
    for value in values {
        anyhow::ensure!(seen.insert(value), "duplicate {label}: {value:?}");
    }
    Ok(())
}

fn card_token_ids(source: &str) -> BTreeSet<u32> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.split_once("Some((")?.1;
            rest.split_once(',')?.0.trim().parse().ok()
        })
        .collect()
}

fn discover_workflows(examples: &Path) -> Result<BTreeSet<String>> {
    if !examples.is_dir() {
        return Ok(BTreeSet::new());
    }
    let mut inputs = Vec::new();
    collect_feff_inputs(examples, &mut inputs)?;
    inputs
        .into_iter()
        .map(|input| {
            input
                .parent()
                .context("feff.inp has no parent")?
                .strip_prefix(examples)
                .context("workflow is outside examples")
                .map(|path| path.to_string_lossy().replace('\\', "/"))
        })
        .collect()
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_feff_inputs(&path, inputs)?;
        } else if path.file_name().is_some_and(|name| name == "feff.inp") {
            inputs.push(path);
        }
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    loop {
        if current.join(SCOPE_MANIFEST).is_file() && current.join("Cargo.toml").is_file() {
            return Ok(current);
        }
        anyhow::ensure!(current.pop(), "could not locate workspace root");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_feff_card_ids() {
        let ids = card_token_ids(
            r#"
            "ATOM" => Some((1, "ATOMS")),
            "MARK" | "TARG" => Some((71, "TARGET")),
            "#,
        );
        assert_eq!(ids, BTreeSet::from([1, 71]));
    }

    #[test]
    fn extracts_only_active_makefile_executables() {
        let executables = makefile_word_variable(
            r#"
            # EXECUTABLES = stale omitted
            EXECUTABLES = atomic dmdw \
                dym2feffinp
            EXECUTABLES += xsph # an inline explanation
            EXECUTABLES_EXTRA = ignored
            "#,
            "EXECUTABLES",
        )
        .expect("active assignment should parse");

        assert_eq!(
            executables,
            BTreeSet::from([
                "atomic".to_owned(),
                "dmdw".to_owned(),
                "dym2feffinp".to_owned(),
                "xsph".to_owned(),
            ])
        );
    }

    #[test]
    fn rejects_missing_active_makefile_assignment() {
        let error = makefile_word_variable("# EXECUTABLES = only-commented", "EXECUTABLES")
            .expect_err("commented assignment must not count");
        assert!(error.to_string().contains("missing active EXECUTABLES"));
    }

    #[test]
    fn detects_upstream_executable_omitted_from_manifest() {
        let built = BTreeSet::from([
            "atomic".to_owned(),
            "dmdw".to_owned(),
            "dym2feffinp".to_owned(),
        ]);
        let declared = BTreeSet::from(["atomic".to_owned(), "dmdw".to_owned()]);

        assert_eq!(
            omitted_upstream_executables(&built, &declared),
            vec!["dym2feffinp"]
        );
    }
}
