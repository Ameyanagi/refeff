//! Port-status reporting for FEFF module migration coverage.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PortStatusReport {
    modules: Vec<PortModuleStatus>,
}

impl PortStatusReport {
    fn unported_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| module.has_unported_gate)
            .count()
    }

    fn reference_covered_unported_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| module.has_unported_gate && module.has_reference_coverage)
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PortModuleStatus {
    module: String,
    has_unported_gate: bool,
    has_reference_coverage: bool,
    has_cache_path: bool,
    unported_reasons: Vec<String>,
}

pub(crate) fn print_port_status(cli_src: Option<PathBuf>, fail_on_unported: bool) -> Result<()> {
    let cli_src = cli_src.unwrap_or_else(default_cli_src_dir);
    let report = port_status_report(&cli_src)?;
    println!(
        "module status: modules={} unported={} unported_reference_covered={}",
        report.modules.len(),
        report.unported_count(),
        report.reference_covered_unported_count()
    );
    println!("module\tstate\treference\tcache\treason");
    for module in &report.modules {
        println!(
            "{}\t{}\t{}\t{}\t{}",
            module.module,
            if module.has_unported_gate {
                "unported"
            } else {
                "supported"
            },
            bool_status(module.has_reference_coverage),
            bool_status(module.has_cache_path),
            module.unported_reasons.join(" | ")
        );
    }

    if fail_on_unported && report.unported_count() > 0 {
        anyhow::bail!(
            "{} module(s) still contain explicit unported gates",
            report.unported_count()
        );
    }
    Ok(())
}

fn port_status_report(cli_src: &Path) -> Result<PortStatusReport> {
    let mut modules = Vec::new();
    for (module, text) in module_sources(cli_src)? {
        modules.push(module_status_from_source(&module, &text));
    }

    modules.sort_by(|left, right| left.module.cmp(&right.module));
    Ok(PortStatusReport { modules })
}

fn module_sources(cli_src: &Path) -> Result<Vec<(String, String)>> {
    let mut modules = Vec::new();
    for entry in std::fs::read_dir(cli_src)
        .with_context(|| format!("failed to read {}", cli_src.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", cli_src.display()))?;
        let path = entry.path();
        if path.is_file() {
            let Some(module) = flat_module_name(&path) else {
                continue;
            };
            let mut text = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let sidecar_dir = cli_src.join(&module);
            if sidecar_dir.is_dir() {
                text.push('\n');
                text.push_str(&read_module_directory_source(&sidecar_dir)?);
            }
            modules.push((module, text));
        } else if path.is_dir() {
            let Some(module) = directory_module_name(&path) else {
                continue;
            };
            if cli_src.join(format!("{module}.rs")).is_file() {
                continue;
            }
            let text = read_module_directory_source(&path)?;
            modules.push((module, text));
        }
    }
    modules.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(modules)
}

fn flat_module_name(path: &Path) -> Option<String> {
    if path.extension().is_none_or(|extension| extension != "rs") {
        return None;
    }
    let module = path.file_stem()?.to_str()?;
    if matches!(module, "lib" | "tests") {
        return None;
    }
    Some(module.to_string())
}

fn directory_module_name(path: &Path) -> Option<String> {
    let module = path.file_name()?.to_str()?;
    if matches!(module, "bin" | "tests") || !path.join("mod.rs").is_file() {
        return None;
    }
    Some(module.to_string())
}

fn read_module_directory_source(path: &Path) -> Result<String> {
    let mut rust_files = Vec::new();
    collect_rust_files(path, &mut rust_files)?;
    rust_files.sort();

    let mut source = String::new();
    for file in rust_files {
        source.push_str(
            &std::fs::read_to_string(&file)
                .with_context(|| format!("failed to read {}", file.display()))?,
        );
        source.push('\n');
    }
    Ok(source)
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files)?;
        } else if path.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn module_status_from_source(module: &str, source: &str) -> PortModuleStatus {
    let unported_reasons = unported_reasons_from_source(source);
    PortModuleStatus {
        module: module.to_string(),
        has_unported_gate: !unported_reasons.is_empty(),
        has_reference_coverage: has_reference_coverage(source),
        has_cache_path: has_cache_path(source),
        unported_reasons,
    }
}

fn unported_reasons_from_source(source: &str) -> Vec<String> {
    let mut reasons = Vec::new();
    let mut in_bail = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.contains("bail!(") {
            in_bail = true;
        }
        if in_bail
            && let Some(reason) = unported_reason_from_bail_line(trimmed)
            && !reasons.contains(&reason)
        {
            reasons.push(reason);
        }
        if in_bail && trimmed.contains(");") {
            in_bail = false;
        }
    }
    reasons
}

fn unported_reason_from_bail_line(line: &str) -> Option<String> {
    if line.contains("requires the unported")
        || line.contains("still unported")
        || line.contains("unported density callback path")
    {
        Some(extract_first_string(line).unwrap_or_else(|| {
            line.trim_start_matches("anyhow::bail!(")
                .trim_start_matches("bail!(")
                .trim()
                .trim_end_matches(',')
                .trim_end_matches(';')
                .trim_end_matches(')')
                .to_string()
        }))
    } else {
        None
    }
}

fn extract_first_string(line: &str) -> Option<String> {
    let start = line.find('"')?;
    let rest = &line[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn has_reference_coverage(source: &str) -> bool {
    [
        "generated_reference_when_present",
        "checks_generated_reference_when_present",
        "roundtrips_reference_zip_when_present",
        "matches_feff_reference",
        "roundtrips_generated_reference",
    ]
    .iter()
    .any(|needle| source.contains(needle))
}

fn has_cache_path(source: &str) -> bool {
    ["has_cached", "cached-output", "cached output"]
        .iter()
        .any(|needle| source.contains(needle))
}

fn bool_status(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn default_cli_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(PathBuf::new, Path::to_path_buf)
        .join("crates/refeff-cli/src")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporary_work_dir;

    #[test]
    fn port_status_detects_unported_reference_and_cache_markers() {
        let source = r#"
/// The EXAMPLE numerical solver is still unported.
pub(crate) fn has_cached_example_output() -> anyhow::Result<bool> { Ok(true) }
fn run() -> anyhow::Result<()> {
    anyhow::bail!("EXAMPLE generation requires the unported EXAMPLE numerical solver");
}
#[test]
fn example_module_roundtrips_generated_reference_when_present() {}
"#;

        let status = module_status_from_source("example", source);

        assert_eq!(status.module, "example");
        assert!(status.has_unported_gate);
        assert!(status.has_reference_coverage);
        assert!(status.has_cache_path);
        assert_eq!(status.unported_reasons.len(), 1);
    }

    #[test]
    fn port_status_report_scans_cli_module_sources() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-port-status-test")?;
        std::fs::write(
            root.join("atomic.rs"),
            r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("ATOM generation requires the unported ATOM numerical solver");
}
#[test]
fn atomic_module_roundtrips_generated_reference_when_present() {}
"#,
        )?;
        std::fs::write(root.join("wpot.rs"), "pub(crate) fn run_in_dir() {}\n")?;
        std::fs::write(root.join("eels.rs"), "pub(crate) fn run_in_dir() {}\n")?;
        std::fs::create_dir(root.join("eels"))?;
        std::fs::write(
            root.join("eels/tests.rs"),
            "#[test]\nfn eels_module_roundtrips_generated_reference_when_present() {}\n",
        )?;
        std::fs::create_dir(root.join("dmdw"))?;
        std::fs::write(root.join("dmdw/mod.rs"), "pub(crate) fn run_in_dir() {}\n")?;
        std::fs::write(
            root.join("dmdw/tests.rs"),
            "#[test]\nfn dmdw_module_roundtrips_generated_reference_when_present() {}\n",
        )?;
        std::fs::write(
            root.join("lib.rs"),
            "this workspace root module should not be counted\n",
        )?;
        std::fs::write(
            root.join("tests.rs"),
            "this external CLI test module should not be counted\n",
        )?;

        let report = port_status_report(&root)?;

        assert_eq!(report.modules.len(), 4);
        assert_eq!(report.unported_count(), 1);
        assert_eq!(report.reference_covered_unported_count(), 1);
        assert_eq!(report.modules[0].module, "atomic");
        assert_eq!(report.modules[1].module, "dmdw");
        assert!(report.modules[1].has_reference_coverage);
        assert_eq!(report.modules[2].module, "eels");
        assert!(report.modules[2].has_reference_coverage);
        assert_eq!(report.modules[3].module, "wpot");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
