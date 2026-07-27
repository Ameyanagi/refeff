//! Port-status reporting for FEFF module migration coverage.

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::compatibility_matrix::compatibility_open_items;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PortStatusReport {
    modules: Vec<PortModuleStatus>,
}

impl PortStatusReport {
    fn module_count(&self) -> usize {
        self.modules.len()
    }

    fn supported_count(&self) -> usize {
        self.module_count() - self.unported_count()
    }

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

    fn source_handoff_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| !module.source_handoff_markers.is_empty())
            .count()
    }

    fn ignored_parity_check_count(&self) -> usize {
        self.modules
            .iter()
            .map(|module| module.ignored_parity_checks.len())
            .sum()
    }

    fn ignored_parity_module_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| !module.ignored_parity_checks.is_empty())
            .count()
    }

    fn guarded_branch_count(&self) -> usize {
        self.modules
            .iter()
            .map(|module| module.guarded_branch_reasons.len())
            .sum()
    }

    fn guarded_branch_module_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| !module.guarded_branch_reasons.is_empty())
            .count()
    }

    fn ungated_module_count(&self) -> usize {
        self.modules
            .iter()
            .filter(|module| !module.has_unported_gate && module.guarded_branch_reasons.is_empty())
            .count()
    }

    fn module_support_percent(&self) -> f64 {
        percent(self.supported_count(), self.module_count())
    }

    fn ungated_module_percent(&self) -> f64 {
        percent(self.ungated_module_count(), self.module_count())
    }

    fn unported_percent(&self) -> f64 {
        percent(self.unported_count(), self.module_count())
    }

    fn source_handoff_percent(&self) -> f64 {
        percent(self.source_handoff_count(), self.module_count())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PortModuleStatus {
    module: String,
    has_unported_gate: bool,
    has_reference_coverage: bool,
    has_cache_path: bool,
    source_handoff_markers: Vec<String>,
    ignored_parity_checks: Vec<String>,
    guarded_branch_reasons: Vec<String>,
    unported_reasons: Vec<String>,
    next_step: Option<&'static str>,
}

pub(crate) fn print_port_status(
    cli_src: Option<PathBuf>,
    fail_on_unported: bool,
    fail_on_ignored_parity: bool,
    fail_on_guarded_branches: bool,
    detail: bool,
    json_out: Option<&Path>,
) -> Result<()> {
    let cli_src = cli_src.unwrap_or_else(default_cli_src_dir);
    let report = port_status_report(&cli_src)?;
    println!(
        "module status: modules={} unported={} unported_reference_covered={} source_handoff={} supported={} module_support={:.1}% source_handoff_percent={:.1}% unported_percent={:.1}% ungated_modules={} ungated_module_percent={:.1}% guarded_branches={} guarded_branch_modules={} ignored_parity_checks={} ignored_parity_modules={}",
        report.module_count(),
        report.unported_count(),
        report.reference_covered_unported_count(),
        report.source_handoff_count(),
        report.supported_count(),
        report.module_support_percent(),
        report.source_handoff_percent(),
        report.unported_percent(),
        report.ungated_module_count(),
        report.ungated_module_percent(),
        report.guarded_branch_count(),
        report.guarded_branch_module_count(),
        report.ignored_parity_check_count(),
        report.ignored_parity_module_count()
    );
    println!("module\tstate\treference\tcache\tsource_handoff\tguarded_branches\treason");
    for module in &report.modules {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            module.module,
            module_state(module),
            bool_status(module.has_reference_coverage),
            bool_status(module.has_cache_path),
            bool_status(!module.source_handoff_markers.is_empty()),
            module.guarded_branch_reasons.len(),
            module_display_reason(module)
        );
    }
    if detail {
        print_port_detail(&report)?;
    }
    if let Some(json_out) = json_out {
        write_port_status_json_report(json_out, &report)?;
        println!("wrote port status json: {}", json_out.display());
    }

    enforce_port_status_gates(
        &report,
        fail_on_unported,
        fail_on_ignored_parity,
        fail_on_guarded_branches,
    )
}

fn write_port_status_json_report(path: &Path, report: &PortStatusReport) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, port_status_json_report(report)?)?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct PortStatusSummaryJson {
    modules: usize,
    unported: usize,
    unported_reference_covered: usize,
    source_handoff: usize,
    supported: usize,
    module_support_percent: f64,
    source_handoff_percent: f64,
    unported_percent: f64,
    ungated_modules: usize,
    ungated_module_percent: f64,
    guarded_branches: usize,
    guarded_branch_modules: usize,
    ignored_parity_checks: usize,
    ignored_parity_modules: usize,
}

#[derive(Debug, Serialize)]
struct PortModuleStatusJson<'a> {
    module: &'a str,
    state: &'static str,
    reference: bool,
    cache: bool,
    source_handoff: bool,
    source_handoff_markers: &'a [String],
    unported_reasons: &'a [String],
    guarded_branch_reasons: &'a [String],
    ignored_parity_checks: &'a [String],
    next: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct PortStatusReportJson<'a> {
    summary: PortStatusSummaryJson,
    modules: Vec<PortModuleStatusJson<'a>>,
}

/// Rounds to one decimal place, matching the previous `{:.1}` text formatting
/// so the JSON percent fields keep the same numeric value.
fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn port_status_json_report(report: &PortStatusReport) -> Result<String> {
    let json = PortStatusReportJson {
        summary: PortStatusSummaryJson {
            modules: report.module_count(),
            unported: report.unported_count(),
            unported_reference_covered: report.reference_covered_unported_count(),
            source_handoff: report.source_handoff_count(),
            supported: report.supported_count(),
            module_support_percent: round1(report.module_support_percent()),
            source_handoff_percent: round1(report.source_handoff_percent()),
            unported_percent: round1(report.unported_percent()),
            ungated_modules: report.ungated_module_count(),
            ungated_module_percent: round1(report.ungated_module_percent()),
            guarded_branches: report.guarded_branch_count(),
            guarded_branch_modules: report.guarded_branch_module_count(),
            ignored_parity_checks: report.ignored_parity_check_count(),
            ignored_parity_modules: report.ignored_parity_module_count(),
        },
        modules: report
            .modules
            .iter()
            .map(|module| PortModuleStatusJson {
                module: &module.module,
                state: module_state(module),
                reference: module.has_reference_coverage,
                cache: module.has_cache_path,
                source_handoff: !module.source_handoff_markers.is_empty(),
                source_handoff_markers: &module.source_handoff_markers,
                unported_reasons: &module.unported_reasons,
                guarded_branch_reasons: &module.guarded_branch_reasons,
                ignored_parity_checks: &module.ignored_parity_checks,
                next: module.next_step,
            })
            .collect(),
    };
    serde_json::to_string_pretty(&json).context("failed to serialize port status json")
}

fn enforce_port_status_gates(
    report: &PortStatusReport,
    fail_on_unported: bool,
    fail_on_ignored_parity: bool,
    fail_on_guarded_branches: bool,
) -> Result<()> {
    if fail_on_unported && report.unported_count() > 0 {
        anyhow::bail!(
            "{} module(s) still contain explicit unported gates",
            report.unported_count()
        );
    }
    if fail_on_guarded_branches && report.guarded_branch_count() > 0 {
        anyhow::bail!(
            "{} guarded production branch(es) remain across {} module(s)",
            report.guarded_branch_count(),
            report.guarded_branch_module_count()
        );
    }
    if fail_on_ignored_parity && report.ignored_parity_check_count() > 0 {
        anyhow::bail!(
            "{} ignored parity check(s) remain across {} module(s)",
            report.ignored_parity_check_count(),
            report.ignored_parity_module_count()
        );
    }
    Ok(())
}

fn print_port_detail(report: &PortStatusReport) -> Result<()> {
    print!("{}", port_detail_text(report)?);
    Ok(())
}

fn port_detail_text(report: &PortStatusReport) -> Result<String> {
    let mut text = String::new();
    writeln!(&mut text)?;
    writeln!(&mut text, "remaining port detail:")?;
    writeln!(&mut text, "module\tblocker\tnext")?;
    for module in report
        .modules
        .iter()
        .filter(|module| module.has_unported_gate)
    {
        writeln!(
            &mut text,
            "{}\t{}\t{}",
            module.module,
            module.unported_reasons.join(" | "),
            module
                .next_step
                .unwrap_or("remove the remaining explicit unported gate")
        )?;
    }
    writeln!(&mut text)?;
    writeln!(&mut text, "source handoff detail:")?;
    writeln!(&mut text, "module\tcoverage")?;
    for module in report
        .modules
        .iter()
        .filter(|module| !module.source_handoff_markers.is_empty())
    {
        writeln!(
            &mut text,
            "{}\t{}",
            module.module,
            module.source_handoff_markers.join(" | ")
        )?;
    }
    let open_items = compatibility_open_items();
    if !open_items.is_empty() {
        writeln!(&mut text)?;
        writeln!(&mut text, "branch compatibility blockers:")?;
        writeln!(
            &mut text,
            "module\trow\tworkflow\tstatus\trequirement\tnext\tverify"
        )?;
        for item in open_items {
            writeln!(
                &mut text,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                item.module,
                item.id,
                item.workflow,
                item.status,
                item.requirement,
                item.next_action.unwrap_or(""),
                item.verification_gate.unwrap_or("")
            )?;
        }
    }
    if report.guarded_branch_count() > 0 {
        writeln!(&mut text)?;
        writeln!(&mut text, "guarded production branches:")?;
        writeln!(&mut text, "module\tbranch_guard")?;
        for module in report
            .modules
            .iter()
            .filter(|module| !module.guarded_branch_reasons.is_empty())
        {
            for reason in &module.guarded_branch_reasons {
                writeln!(&mut text, "{}\t{}", module.module, reason)?;
            }
        }
    }
    if report.ignored_parity_check_count() > 0 {
        writeln!(&mut text)?;
        writeln!(&mut text, "ignored parity checks:")?;
        writeln!(&mut text, "module\tcheck")?;
        for module in report
            .modules
            .iter()
            .filter(|module| !module.ignored_parity_checks.is_empty())
        {
            for check in &module.ignored_parity_checks {
                writeln!(&mut text, "{}\t{}", module.module, check)?;
            }
        }
    }
    writeln!(&mut text)?;
    writeln!(&mut text, "completion plan:")?;
    writeln!(&mut text, "artifact\trole")?;
    writeln!(
        &mut text,
        "docs/FEFF_RUST_PORT_PLAN.md\tactive implementation order and test cadence"
    )?;
    Ok(text)
}

fn port_status_report(cli_src: &Path) -> Result<PortStatusReport> {
    let mut modules = Vec::new();
    let workspace_root = workspace_root_for_cli_src(cli_src);
    for (module, text) in module_sources(cli_src)? {
        let evidence = workspace_root
            .as_deref()
            .map(|root| module_evidence_source(root, &module))
            .transpose()?
            .unwrap_or_default();
        modules.push(module_status_from_source_with_evidence(
            &module, &text, &evidence,
        ));
    }
    if let Some(driver_status) = driver_status(cli_src)? {
        modules.push(driver_status);
    }

    modules.sort_by(|left, right| left.module.cmp(&right.module));
    Ok(PortStatusReport { modules })
}

fn driver_status(cli_src: &Path) -> Result<Option<PortModuleStatus>> {
    let path = cli_src.join("lib.rs");
    if !path.is_file() {
        return Ok(None);
    }
    let mut text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    append_optional_source_file(&cli_src.join("tests.rs"), &mut text)?;
    append_optional_source_dir(&cli_src.join("tests"), &mut text)?;
    let mut status = module_status_from_source("feff", &text);
    status.has_cache_path = false;
    status.source_handoff_markers.clear();
    Ok((status.has_unported_gate || !status.ignored_parity_checks.is_empty()).then_some(status))
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

fn workspace_root_for_cli_src(cli_src: &Path) -> Option<PathBuf> {
    Some(cli_src.parent()?.parent()?.parent()?.to_path_buf())
}

fn module_evidence_source(workspace_root: &Path, module: &str) -> Result<String> {
    let mut source = String::new();
    for path in module_evidence_paths(workspace_root, module) {
        append_optional_source_file(&path, &mut source)?;
    }
    for path in module_evidence_dirs(workspace_root, module) {
        append_optional_source_dir(&path, &mut source)?;
    }
    Ok(source)
}

fn module_evidence_paths(workspace_root: &Path, module: &str) -> Vec<PathBuf> {
    let engine_tests = workspace_root.join("crates/refeff-engine/src/tests");
    let io_reference_tests = workspace_root.join("crates/refeff-io/tests/reference_examples");
    match module {
        "fullspectrum" => vec![
            workspace_root.join("crates/refeff-engine/src/fullspectrum/tests.rs"),
            engine_tests.join("full_run_spectrum_cache.rs"),
            io_reference_tests.join("spectrum_outputs.rs"),
        ],
        "opcons" => vec![
            engine_tests.join("module_aliases.rs"),
            engine_tests.join("full_run_core_cache.rs"),
        ],
        "wpot" => vec![
            engine_tests.join("module_aliases.rs"),
            io_reference_tests.join("handoff_outputs.rs"),
        ],
        _ => Vec::new(),
    }
}

fn module_evidence_dirs(workspace_root: &Path, module: &str) -> Vec<PathBuf> {
    match module {
        "fullspectrum" => vec![workspace_root.join("crates/refeff-core/src/fullspectrum/tests")],
        _ => Vec::new(),
    }
}

fn append_optional_source_file(path: &Path, source: &mut String) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    source.push_str(
        &std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?,
    );
    source.push('\n');
    Ok(())
}

fn append_optional_source_dir(path: &Path, source: &mut String) -> Result<()> {
    if !path.is_dir() {
        return Ok(());
    }
    source.push_str(&read_module_directory_source(path)?);
    source.push('\n');
    Ok(())
}

pub(crate) fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
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
    module_status_from_source_with_evidence(module, source, "")
}

fn module_status_from_source_with_evidence(
    module: &str,
    source: &str,
    evidence: &str,
) -> PortModuleStatus {
    let unported_reasons = unported_reasons_from_source(source);
    let has_unported_gate = !unported_reasons.is_empty();
    let next_step = has_unported_gate
        .then(|| module_next_step(module, &unported_reasons))
        .flatten();
    let source_handoff_markers = source_handoff_markers_from_source(source, evidence);
    let ignored_parity_checks = ignored_parity_checks_from_source(source);
    let mut guarded_branch_reasons = guarded_branch_reasons_from_source(source);
    extend_static_guarded_branch_reasons(module, source, &mut guarded_branch_reasons);
    PortModuleStatus {
        module: module.to_string(),
        has_unported_gate,
        has_reference_coverage: has_reference_coverage(source) || has_reference_coverage(evidence),
        has_cache_path: has_cache_path(module, source) || has_cache_path(module, evidence),
        source_handoff_markers,
        ignored_parity_checks,
        guarded_branch_reasons,
        unported_reasons,
        next_step,
    }
}

fn module_next_step(module: &str, unported_reasons: &[String]) -> Option<&'static str> {
    match module {
        "atomic" => Some("finish remaining ATOM finite-nucleus and reference-parity gaps"),
        "pot" => Some(
            "close POT SCF numerical parity against FEFF references and remove remaining unsupported input gates",
        ),
        "feff" => Some(
            "replace the top-level full-run numerical fallback with fully ported module orchestration",
        ),
        _ if unported_reasons.iter().any(|reason| {
            reason.contains("requires the unported") || reason.contains("still unported")
        }) =>
        {
            Some("replace the remaining cached-output fallback with a source-backed Rust driver")
        }
        _ => None,
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

fn guarded_branch_reasons_from_source(source: &str) -> Vec<String> {
    let mut reasons = Vec::new();
    let mut in_guard_macro = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.contains("bail!(") || trimmed.contains("ensure!(") {
            in_guard_macro = true;
        }
        if in_guard_macro
            && let Some(reason) = guarded_branch_reason_from_macro_line(trimmed)
            && !reasons.contains(&reason)
        {
            reasons.push(reason);
        }
        if in_guard_macro && trimmed.contains(");") {
            in_guard_macro = false;
        }
    }
    reasons
}

fn extend_static_guarded_branch_reasons(_module: &str, _source: &str, _reasons: &mut Vec<String>) {}

fn guarded_branch_reason_from_macro_line(line: &str) -> Option<String> {
    let reason = extract_first_string(line)?;
    if is_guarded_branch_reason(&reason) {
        Some(normalize_unported_reason(reason))
    } else {
        None
    }
}

fn is_guarded_branch_reason(reason: &str) -> bool {
    reason.contains("does not yet support")
        || reason.contains("currently supports")
        || reason.contains("supports only")
        || (reason.contains("supports ") && reason.contains(" only"))
}

fn unported_reason_from_bail_line(line: &str) -> Option<String> {
    if line.contains("requires the unported")
        || line.contains("still unported")
        || line.contains("unported density callback path")
        || line.contains("full FEFF numerical execution is not implemented yet")
    {
        let reason = extract_first_string(line).unwrap_or_else(|| {
            line.trim_start_matches("anyhow::bail!(")
                .trim_start_matches("bail!(")
                .trim()
                .trim_end_matches(',')
                .trim_end_matches(';')
                .trim_end_matches(')')
                .to_string()
        });
        Some(normalize_unported_reason(reason))
    } else {
        None
    }
}

fn normalize_unported_reason(reason: String) -> String {
    let without_placeholders = replace_format_placeholders(&reason);
    let static_part = without_placeholders
        .split_once(';')
        .map_or(without_placeholders.as_str(), |(prefix, _)| prefix);
    let static_part = static_part
        .split_once(", got ")
        .map_or(static_part, |(prefix, _)| prefix);
    static_part
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(':')
        .to_string()
}

fn replace_format_placeholders(reason: &str) -> String {
    let mut normalized = String::with_capacity(reason.len());
    let mut chars = reason.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '{' {
            for inner in chars.by_ref() {
                if inner == '}' {
                    break;
                }
            }
            normalized.push_str("<value>");
        } else {
            normalized.push(ch);
        }
    }
    normalized
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
        "renders_generated_reference_wpot_outputs_when_present",
    ]
    .iter()
    .any(|needle| source.contains(needle))
}

fn has_cache_path(module: &str, source: &str) -> bool {
    ["has_cached", "cached-output", "cached output"]
        .iter()
        .any(|needle| source.contains(needle))
        || match module {
            "opcons" => source.contains("has_complete_table_inputs"),
            "wpot" => source.contains("read_pot_bin") && source.contains("read_apot_bin"),
            _ => false,
        }
}

fn source_handoff_markers_from_source(source: &str, evidence: &str) -> Vec<String> {
    let combined = [source, evidence].join("\n");
    let mut markers = Vec::new();
    push_marker_if(
        &mut markers,
        combined.contains("has_supported_") || combined.contains("run_supported_"),
        "supported-runner",
    );
    push_marker_if(
        &mut markers,
        combined.contains("source handoff")
            || combined.contains("source-backed")
            || combined.contains("source_handoff"),
        "source-handoff",
    );
    push_marker_if(
        &mut markers,
        combined.contains("_from_handoffs") || combined.contains("from_handoffs"),
        "typed-handoff-adapter",
    );
    push_marker_if(
        &mut markers,
        combined.contains("write_missing_or_unusable")
            || combined.contains("recover_")
            || combined.contains("_needs_generation"),
        "repair-or-generation",
    );
    markers
}

fn push_marker_if(markers: &mut Vec<String>, condition: bool, marker: &'static str) {
    if condition && !markers.iter().any(|existing| existing == marker) {
        markers.push(marker.to_string());
    }
}

fn ignored_parity_checks_from_source(source: &str) -> Vec<String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut checks = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("#[ignore") {
            continue;
        }

        let reason = extract_first_string(trimmed);
        let name = lines
            .iter()
            .skip(index + 1)
            .take(12)
            .filter_map(|candidate| test_name_from_line(candidate.trim()))
            .next();
        let check = match (name, reason) {
            (Some(name), Some(reason)) => format!("{name}: {reason}"),
            (Some(name), None) => name,
            (None, Some(reason)) => format!("ignored test: {reason}"),
            (None, None) => "ignored test".to_string(),
        };
        if !checks.contains(&check) {
            checks.push(check);
        }
    }
    checks
}

pub(crate) fn test_name_from_line(line: &str) -> Option<String> {
    let start = line.find("fn ")? + "fn ".len();
    let name = line[start..]
        .split(|ch: char| ch == '(' || ch == '<' || ch.is_whitespace())
        .next()?;
    (!name.is_empty()).then(|| name.to_string())
}

fn bool_status(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn module_state(module: &PortModuleStatus) -> &'static str {
    if module.has_unported_gate {
        "unported"
    } else if !module.guarded_branch_reasons.is_empty() {
        "guarded"
    } else {
        "supported"
    }
}

fn module_display_reason(module: &PortModuleStatus) -> String {
    let mut reasons = Vec::new();
    reasons.extend(module.unported_reasons.iter().cloned());
    reasons.extend(module.guarded_branch_reasons.iter().cloned());
    reasons.join(" | ")
}

fn percent(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

fn default_cli_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(PathBuf::new, Path::to_path_buf)
        .join("crates/refeff-engine/src")
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
    fn port_status_assigns_unported_next_step() {
        let source = r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("ATOM finite-nucleus source path is still unported");
}
"#;

        let status = module_status_from_source("atomic", source);

        assert_eq!(
            status.next_step,
            Some("finish remaining ATOM finite-nucleus and reference-parity gaps")
        );

        let pot_status = module_status_from_source(
            "pot",
            r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("POT self-consistent potential generation requires the unported POT numerical solver");
}
"#,
        );
        assert_eq!(
            pot_status.next_step,
            Some(
                "close POT SCF numerical parity against FEFF references and remove remaining unsupported input gates"
            )
        );
    }

    #[test]
    fn port_status_uses_module_specific_external_evidence() {
        let opcons = module_status_from_source_with_evidence(
            "opcons",
            "pub(crate) fn has_complete_table_inputs(work_dir: &Path) -> Result<bool> { Ok(true) }",
            "#[test]\nfn opcons_module_matches_feff_reference_loss_when_present() {}\n",
        );
        assert!(opcons.has_reference_coverage);
        assert!(opcons.has_cache_path);

        let wpot = module_status_from_source_with_evidence(
            "wpot",
            "fn run() { read_pot_bin(&pot_path); read_apot_bin(&apot_path); }",
            "#[test]\nfn renders_generated_reference_wpot_outputs_when_present() {}\n",
        );
        assert!(wpot.has_reference_coverage);
        assert!(wpot.has_cache_path);

        let fullspectrum = module_status_from_source_with_evidence(
            "fullspectrum",
            "pub(crate) fn has_cached_optical_inputs(work_dir: &Path) -> Result<bool> { Ok(true) }",
            "#[test]\nfn drude_term_matches_feff_reference_algorithm() {}\n",
        );
        assert!(fullspectrum.has_reference_coverage);
        assert!(fullspectrum.has_cache_path);
    }

    #[test]
    fn port_status_tracks_source_handoff_markers() {
        let source = r#"
/// Generate a source-backed handoff without entering the old solver.
pub(crate) fn has_supported_example_handoff() -> anyhow::Result<bool> { Ok(true) }
pub(crate) fn run_supported_example_handoff_in_dir() -> anyhow::Result<usize> { Ok(1) }
fn build() {
    let _ = example_setup_from_handoffs();
    let _ = write_missing_or_unusable_example_from_source();
}
"#;

        let status = module_status_from_source("example", source);

        assert_eq!(
            status.source_handoff_markers,
            vec![
                "supported-runner",
                "source-handoff",
                "typed-handoff-adapter",
                "repair-or-generation",
            ]
        );
    }

    #[test]
    fn port_status_tracks_ignored_parity_checks() {
        let source = r#"
#[ignore = "reference parity release gate"]
#[test]
fn example_module_matches_reference_when_present() {}

#[ignore = "slow parity check"]
fn second_reference_gate() {}
"#;

        let status = module_status_from_source("example", source);

        assert_eq!(
            status.ignored_parity_checks,
            vec![
                "example_module_matches_reference_when_present: reference parity release gate",
                "second_reference_gate: slow parity check",
            ]
        );
    }

    #[test]
    fn port_status_tracks_guarded_branch_reasons() {
        let source = r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("FF2X generation does not yet support NRIXS/JAS do_nrixs=1");
}

fn validate(value: i32) -> anyhow::Result<()> {
    anyhow::ensure!(
        value == 0,
        "EXAMPLE source generation supports mode A only"
    );
    Ok(())
}

fn source_parity_rejection(value: i32) -> anyhow::Result<()> {
    anyhow::ensure!(
        value == 0,
        "XSPH positive-izstd xsect generation rejects nonrelativistic M1 because FEFF radint stops for mult=1 with ifl < 0"
    );
    Ok(())
}

fn invalid_input(kind: i32) -> anyhow::Result<()> {
    anyhow::bail!("unsupported EELS input source {kind}; expected 1 for xmu or 2 for opconsKK");
}
"#;

        let status = module_status_from_source("example", source);

        assert_eq!(module_state(&status), "guarded");
        assert_eq!(
            status.guarded_branch_reasons,
            vec![
                "FF2X generation does not yet support NRIXS/JAS do_nrixs=1",
                "EXAMPLE source generation supports mode A only",
            ]
        );
    }

    #[test]
    fn port_status_does_not_static_guard_source_backed_nrixs_xsectjas_generation() {
        let status = module_status_from_source(
            "xsph",
            r#"
fn nrixs_xsectjas_requested() -> bool { true }
fn prepare_required_nrixs_spectrum_sidecars() {}
"#,
        );

        assert_eq!(module_state(&status), "supported");
        assert!(status.guarded_branch_reasons.is_empty());
    }

    #[test]
    fn port_status_detail_omits_empty_ignored_parity_section() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
/// Generate a source-backed handoff without entering the old solver.
pub(crate) fn run_supported_example_handoff_in_dir() -> anyhow::Result<()> { Ok(()) }
"#,
            )],
        };

        let detail = port_detail_text(&report).expect("port detail should format");

        assert!(!detail.contains("ignored parity checks:"));
        assert!(!detail.contains("guarded production branches:"));
        assert!(detail.contains("completion plan:"));
        assert!(
            detail.contains(
                "docs/FEFF_RUST_PORT_PLAN.md\tactive implementation order and test cadence"
            )
        );
    }

    #[test]
    fn port_status_detail_omits_branch_compatibility_blockers_when_matrix_is_closed() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                "pub(crate) fn run_supported_example_handoff_in_dir() -> anyhow::Result<()> { Ok(()) }",
            )],
        };

        let detail = port_detail_text(&report).expect("port detail should format");

        assert!(!detail.contains("branch compatibility blockers:"));
        assert!(!detail.contains("needs-coverage"));
    }

    #[test]
    fn port_status_json_report_contains_summary_modules_and_escaped_values() {
        let report = PortStatusReport {
            modules: vec![PortModuleStatus {
                module: "example\"module".to_string(),
                has_unported_gate: true,
                has_reference_coverage: true,
                has_cache_path: false,
                source_handoff_markers: vec!["source-handoff".to_string()],
                ignored_parity_checks: vec!["gate\none".to_string()],
                guarded_branch_reasons: vec!["branch guard".to_string()],
                unported_reasons: vec!["needs solver".to_string()],
                next_step: Some("replace with source driver"),
            }],
        };

        let json = port_status_json_report(&report).expect("port status json should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("port status json should parse");

        assert_eq!(value["summary"]["modules"], 1);
        assert_eq!(value["summary"]["unported"], 1);
        let module = &value["modules"][0];
        assert_eq!(module["module"], "example\"module");
        assert_eq!(module["state"], "unported");
        assert_eq!(module["reference"], true);
        assert_eq!(module["cache"], false);
        assert_eq!(
            module["source_handoff_markers"],
            serde_json::json!(["source-handoff"])
        );
        assert_eq!(
            module["ignored_parity_checks"],
            serde_json::json!(["gate\none"])
        );
        assert_eq!(module["next"], "replace with source driver");
    }

    #[test]
    fn write_port_status_json_report_creates_parent_directory() -> Result<()> {
        let root = temporary_work_dir("refeff-port-status-json-test")?;
        let path = root.join("nested/port-status.json");
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                "pub(crate) fn run_supported_example_handoff_in_dir() -> anyhow::Result<()> { Ok(()) }",
            )],
        };

        write_port_status_json_report(&path, &report)?;

        let json = std::fs::read_to_string(&path)?;
        assert!(json.contains("\"module\": \"example\""));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn port_status_detail_lists_ignored_parity_section_when_present() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
#[ignore = "reference parity release gate"]
#[test]
fn example_module_matches_reference_when_present() {}
"#,
            )],
        };

        let detail = port_detail_text(&report).expect("port detail should format");

        assert!(detail.contains("ignored parity checks:"));
        assert!(detail.contains(
            "example\texample_module_matches_reference_when_present: reference parity release gate"
        ));
        assert!(detail.contains("completion plan:"));
    }

    #[test]
    fn port_status_detail_lists_guarded_branch_section_when_present() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("EXAMPLE generation does not yet support branch=2");
}
"#,
            )],
        };

        let detail = port_detail_text(&report).expect("port detail should format");

        assert!(detail.contains("guarded production branches:"));
        assert!(detail.contains("example\tEXAMPLE generation does not yet support branch=2"));
    }

    #[test]
    fn port_status_gate_can_fail_on_ignored_parity_checks() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
#[ignore = "reference parity release gate"]
#[test]
fn example_module_matches_reference_when_present() {}
"#,
            )],
        };

        let error = enforce_port_status_gates(&report, false, true, false)
            .expect_err("ignored parity gate should fail");

        assert_eq!(
            error.to_string(),
            "1 ignored parity check(s) remain across 1 module(s)"
        );
    }

    #[test]
    fn port_status_gate_can_fail_on_guarded_branches() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("EXAMPLE source generation does not yet support branch=2");
}
"#,
            )],
        };

        let error = enforce_port_status_gates(&report, false, false, true)
            .expect_err("guarded branch gate should fail");

        assert_eq!(
            error.to_string(),
            "1 guarded production branch(es) remain across 1 module(s)"
        );
    }

    #[test]
    fn port_status_gate_prioritizes_unported_failure() {
        let report = PortStatusReport {
            modules: vec![module_status_from_source(
                "example",
                r#"
fn run() -> anyhow::Result<()> {
    anyhow::bail!("EXAMPLE generation requires the unported EXAMPLE numerical solver");
}
#[ignore = "reference parity release gate"]
#[test]
fn example_module_matches_reference_when_present() {}
"#,
            )],
        };

        let error = enforce_port_status_gates(&report, true, true, true)
            .expect_err("unported gate should fail first");

        assert_eq!(
            error.to_string(),
            "1 module(s) still contain explicit unported gates"
        );
    }

    #[test]
    fn port_status_normalizes_formatted_unported_reasons() {
        let source = r#"
fn run_sfconv() -> anyhow::Result<()> {
    anyhow::bail!(
        "SFCONV S0^2 convolution requires the unported SO2CONV numerical driver; discovered {} target file(s): {}; read {} existing target data file(s){}{}",
        target_count,
        target_summary,
        cache_count,
        material_summary,
        cache_summary
    );
}

fn run_rhorrp() -> anyhow::Result<()> {
    anyhow::bail!(
        "RHORRP density generation requires the unported RHORRP numerical solver; missing cached output {}",
        output_path.display()
    );
}
"#;

        let status = module_status_from_source("formatted", source);

        assert_eq!(
            status.unported_reasons,
            vec![
                "SFCONV S0^2 convolution requires the unported SO2CONV numerical driver",
                "RHORRP density generation requires the unported RHORRP numerical solver",
            ]
        );
    }

    #[test]
    fn port_status_report_scans_cli_module_sources() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-port-status-test")?;
        std::fs::write(
            root.join("atomic.rs"),
            r#"
	fn run() -> anyhow::Result<()> {
	    anyhow::bail!("ATOM finite-nucleus path is still unported");
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
        assert_eq!(report.supported_count(), 3);
        assert!((report.module_support_percent() - 75.0).abs() < f64::EPSILON);
        assert!((report.unported_percent() - 25.0).abs() < f64::EPSILON);
        assert_eq!(report.reference_covered_unported_count(), 1);
        assert_eq!(report.modules[0].module, "atomic");
        assert_eq!(report.modules[1].module, "dmdw");
        assert!(report.modules[1].has_reference_coverage);
        assert_eq!(report.modules[2].module, "eels");
        assert!(report.modules[2].has_reference_coverage);
        assert_eq!(report.modules[3].module, "wpot");
        assert_eq!(report.ignored_parity_check_count(), 0);
        assert_eq!(report.ignored_parity_module_count(), 0);
        assert_eq!(report.guarded_branch_count(), 0);
        assert_eq!(report.guarded_branch_module_count(), 0);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn port_status_report_scans_workspace_module_evidence() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-port-status-evidence-test")?;
        let cli_src = root.join("crates/refeff-cli/src");
        let cli_tests = cli_src.join("tests");
        std::fs::create_dir_all(&cli_tests)?;
        std::fs::write(
            cli_src.join("opcons.rs"),
            "pub(crate) fn has_complete_table_inputs() -> bool { true }\n",
        )?;
        std::fs::write(
            cli_tests.join("module_aliases.rs"),
            "#[test]\nfn opcons_module_matches_feff_reference_loss_when_present() {}\n",
        )?;

        let report = port_status_report(&cli_src)?;
        let opcons = report
            .modules
            .iter()
            .find(|module| module.module == "opcons")
            .expect("opcons status is reported");

        assert!(opcons.has_reference_coverage);
        assert!(opcons.has_cache_path);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn port_status_report_tracks_top_level_feff_driver_gate() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-port-status-driver-test")?;
        std::fs::write(root.join("atomic.rs"), "pub(crate) fn run_in_dir() {}\n")?;
        std::fs::write(
            root.join("lib.rs"),
            r#"
fn run_feff_to_dir() -> anyhow::Result<()> {
    anyhow::bail!(
        "full FEFF numerical execution is not implemented yet; completed rdinp for {} cards",
        cards
    )
}
"#,
        )?;

        let report = port_status_report(&root)?;

        assert_eq!(report.modules.len(), 2);
        assert_eq!(report.unported_count(), 1);
        assert_eq!(report.modules[0].module, "atomic");
        assert_eq!(report.modules[1].module, "feff");
        assert_eq!(
            report.modules[1].unported_reasons,
            vec!["full FEFF numerical execution is not implemented yet"]
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
