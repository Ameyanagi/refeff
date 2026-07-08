//! Verifies that compatibility-matrix `evidence`/`verification_gate` strings
//! reference test filters that still resolve to a real `#[test]` function
//! somewhere in the workspace, so a test rename can't silently rot the
//! release gate (F8).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::compatibility_matrix::compatibility_evidence_entries;
use crate::port_status::{collect_rust_files, test_name_from_line};

/// A `cargo test -p <crate> <filter>` reference extracted from a
/// compatibility-matrix row that does not resolve to any `#[test]` function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DanglingEvidenceReference {
    pub(crate) row_id: &'static str,
    pub(crate) source: &'static str,
    pub(crate) crate_name: String,
    pub(crate) test_filter: String,
}

pub(crate) fn print_verify_evidence(workspace_root: Option<PathBuf>) -> Result<()> {
    let workspace_root = workspace_root.unwrap_or_else(default_workspace_root);
    let dangling = dangling_evidence_references(&workspace_root)?;
    if dangling.is_empty() {
        println!(
            "verify-evidence: every compatibility-matrix evidence test reference resolves to a workspace #[test] function"
        );
        return Ok(());
    }

    println!(
        "verify-evidence: {} dangling evidence test reference(s) found",
        dangling.len()
    );
    println!("row\tsource\tcrate\ttest_filter");
    for reference in &dangling {
        println!(
            "{}\t{}\t{}\t{}",
            reference.row_id, reference.source, reference.crate_name, reference.test_filter
        );
    }
    anyhow::bail!(
        "{} compatibility-matrix evidence reference(s) do not match any workspace #[test] function; rename or update the matrix row",
        dangling.len()
    );
}

/// Default workspace root: the parent directory of the `xtask` crate.
pub(crate) fn default_workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(PathBuf::new, Path::to_path_buf)
}

pub(crate) fn dangling_evidence_references(
    workspace_root: &Path,
) -> Result<Vec<DanglingEvidenceReference>> {
    let mut references = Vec::new();
    for entry in compatibility_evidence_entries() {
        collect_references(entry.id, "evidence", entry.evidence, &mut references);
        if let Some(verification_gate) = entry.verification_gate {
            collect_references(
                entry.id,
                "verification_gate",
                verification_gate,
                &mut references,
            );
        }
    }

    let mut test_names_by_crate: HashMap<String, Vec<String>> = HashMap::new();
    let mut dangling = Vec::new();
    for reference in references {
        if !test_names_by_crate.contains_key(&reference.crate_name) {
            let names = crate_test_names(workspace_root, &reference.crate_name)?;
            test_names_by_crate.insert(reference.crate_name.clone(), names);
        }
        let test_names = test_names_by_crate
            .get(&reference.crate_name)
            .map_or(&[][..], Vec::as_slice);
        let resolved = test_names
            .iter()
            .any(|name| name.contains(&reference.test_filter));
        if !resolved {
            dangling.push(reference);
        }
    }
    dangling.sort_by(|left, right| {
        (left.row_id, left.source, left.test_filter.as_str()).cmp(&(
            right.row_id,
            right.source,
            right.test_filter.as_str(),
        ))
    });
    Ok(dangling)
}

fn collect_references(
    row_id: &'static str,
    source: &'static str,
    text: &str,
    references: &mut Vec<DanglingEvidenceReference>,
) {
    for (crate_name, test_filter) in cargo_test_filters(text) {
        references.push(DanglingEvidenceReference {
            row_id,
            source,
            crate_name,
            test_filter,
        });
    }
}

/// Extracts `(crate, test_filter)` pairs from the first `-p <crate> <token>`
/// following each `cargo test` occurrence (stopping at `&&`), mirroring what
/// a copy-pasted `cargo test` invocation would actually run.
fn cargo_test_filters(text: &str) -> Vec<(String, String)> {
    let mut filters = Vec::new();
    let mut search_from = 0_usize;
    while let Some(relative_pos) = text[search_from..].find("cargo test") {
        let start = search_from + relative_pos;
        let segment_end = text[start..]
            .find("&&")
            .map_or(text.len(), |offset| start + offset);
        let segment = &text[start..segment_end];
        if let Some(flag_pos) = segment.find(" -p ") {
            let mut tokens = segment[flag_pos + " -p ".len()..].split_whitespace();
            if let (Some(crate_name), Some(filter)) = (tokens.next(), tokens.next()) {
                let filter = filter.trim_matches(|ch: char| ch == ',' || ch == ';');
                if !filter.is_empty() {
                    filters.push((crate_name.to_string(), filter.to_string()));
                }
            }
        }
        search_from = segment_end.max(start + "cargo test".len());
    }
    filters
}

fn crate_test_names(workspace_root: &Path, crate_name: &str) -> Result<Vec<String>> {
    let crate_dir = if crate_name == "xtask" {
        workspace_root.join("xtask")
    } else {
        workspace_root.join("crates").join(crate_name)
    };
    if !crate_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    collect_rust_files(&crate_dir, &mut files)?;
    files.sort();

    let mut names = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("failed to read {}", file.display()))?;
        collect_test_names(&text, &mut names);
    }
    Ok(names)
}

fn collect_test_names(source: &str, names: &mut Vec<String>) {
    let lines = source.lines().collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        if line.trim() != "#[test]" {
            continue;
        }
        if let Some(name) = lines
            .iter()
            .skip(index + 1)
            .take(6)
            .find_map(|candidate| test_name_from_line(candidate.trim()))
        {
            names.push(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_test_filters_extracts_first_token_after_crate_flag() {
        let filters = cargo_test_filters(
            "cargo test --profile release -p refeff-cli pot_scf && cargo run --profile release -p xtask -- compatibility-matrix --row pot.scf-retry-exhaustion --fail-on-open",
        );

        assert_eq!(
            filters,
            vec![("refeff-cli".to_string(), "pot_scf".to_string())]
        );
    }

    #[test]
    fn cargo_test_filters_strips_trailing_punctuation() {
        let filters = cargo_test_filters(
            "cargo test --profile release -p refeff-cli iterative_scf_outputs_with_high_exchange, high_exchange_iterative, and high_exchange_scf cover module generation",
        );

        assert_eq!(
            filters,
            vec![(
                "refeff-cli".to_string(),
                "iterative_scf_outputs_with_high_exchange".to_string()
            )]
        );
    }

    #[test]
    fn cargo_test_filters_ignores_evidence_without_a_cargo_test_invocation() {
        let filters =
            cargo_test_filters("xtask port-status reports pot as source-handoff supported");

        assert!(filters.is_empty());
    }

    #[test]
    fn collect_test_names_finds_function_after_test_attribute() {
        let mut names = Vec::new();
        collect_test_names(
            r#"
#[test]
fn pot_module_matches_nio_hubbard_bounded_feff_reference_when_present() -> Result<()> {
    Ok(())
}

#[ignore = "slow"]
#[test]
fn another_test() {}
"#,
            &mut names,
        );

        assert_eq!(
            names,
            vec![
                "pot_module_matches_nio_hubbard_bounded_feff_reference_when_present".to_string(),
                "another_test".to_string(),
            ]
        );
    }

    #[test]
    fn dangling_evidence_references_flags_unresolved_test_filter() -> Result<()> {
        let root = temp_workspace_root("verify-evidence-dangling")?;
        let crate_dir = root.join("crates/refeff-cli/src");
        std::fs::create_dir_all(&crate_dir)?;
        std::fs::write(
            crate_dir.join("lib.rs"),
            "#[test]\nfn some_other_test() {}\n",
        )?;

        let dangling = dangling_evidence_references(&root)?;
        assert!(
            dangling
                .iter()
                .any(|reference| reference.test_filter == "tdlda_xsedge")
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn dangling_evidence_references_resolves_known_test_filter() -> Result<()> {
        let root = temp_workspace_root("verify-evidence-resolved")?;
        let crate_dir = root.join("crates/refeff-cli/src");
        std::fs::create_dir_all(&crate_dir)?;
        std::fs::write(
            crate_dir.join("lib.rs"),
            "#[test]\nfn tdlda_xsedge_writes_generated_basis_output() {}\n",
        )?;

        let dangling = dangling_evidence_references(&root)?;
        assert!(
            !dangling
                .iter()
                .any(|reference| reference.test_filter == "tdlda_xsedge")
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    fn temp_workspace_root(prefix: &str) -> Result<PathBuf> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("refeff-{prefix}-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }
}
