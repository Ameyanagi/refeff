//! Branch-level FEFF10 compatibility matrix reporting.

use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompatibilityStatus {
    Covered,
    #[allow(dead_code)]
    NeedsCoverage,
    #[allow(dead_code)]
    NeedsImplementation,
}

impl CompatibilityStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Covered => "covered",
            Self::NeedsCoverage => "needs-coverage",
            Self::NeedsImplementation => "needs-implementation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompatibilityRow {
    id: &'static str,
    module: &'static str,
    workflow: &'static str,
    requirement: &'static str,
    status: CompatibilityStatus,
    evidence: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompatibilityOpenItem {
    pub(crate) id: &'static str,
    pub(crate) module: &'static str,
    pub(crate) workflow: &'static str,
    pub(crate) status: &'static str,
    pub(crate) requirement: &'static str,
    pub(crate) next_action: Option<&'static str>,
    pub(crate) verification_gate: Option<&'static str>,
    pub(crate) fixture_groups: usize,
    pub(crate) missing_fixtures: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompatibilitySummary {
    total: usize,
    covered: usize,
    needs_coverage: usize,
    needs_implementation: usize,
}

impl CompatibilitySummary {
    fn open(self) -> usize {
        self.needs_coverage + self.needs_implementation
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompatibilityRowReport {
    row: CompatibilityRow,
    missing_fixtures: Vec<String>,
    display: bool,
}

struct CompatibilityJsonReport<'a> {
    summary: CompatibilitySummary,
    row_reports: &'a [CompatibilityRowReport],
    open_ids: &'a [&'a str],
    missing_fixtures: &'a [String],
    missing_fixture_manifests: &'a [String],
    stale_fixtures: &'a [String],
    module_filters: &'a [String],
    row_filters: &'a [String],
    open_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixtureRequirement {
    File(&'static str),
    DirectoryFiles {
        directory: &'static str,
        files: &'static [&'static str],
    },
    AnyDirectoryWithPrefixFiles {
        parent: &'static str,
        prefix: &'static str,
        files: &'static [&'static str],
    },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn print_compatibility_matrix(
    detail: bool,
    fail_on_open: bool,
    fail_on_missing_fixtures: bool,
    fail_on_stale_fixtures: bool,
    open_only: bool,
    module_filters: &[String],
    row_filters: &[String],
    json_out: Option<&Path>,
) -> Result<()> {
    let rows = compatibility_rows();
    let selected_rows = filtered_rows(rows, module_filters, row_filters);
    anyhow::ensure!(
        !selected_rows.is_empty(),
        "no compatibility rows matched filter(s): {}",
        filter_summary(module_filters, row_filters)
    );
    let summary = compatibility_summary(&selected_rows);
    println!(
        "compatibility matrix: rows={} covered={} needs_coverage={} needs_implementation={} open={}",
        summary.total,
        summary.covered,
        summary.needs_coverage,
        summary.needs_implementation,
        summary.open()
    );
    println!("id\tmodule\tworkflow\tstatus\trequirement");
    let row_reports = compatibility_row_reports(Path::new("."), &selected_rows, open_only)?;
    let missing_fixtures = missing_fixture_groups(&row_reports);
    for report in &row_reports {
        if !report.display {
            continue;
        }
        let row = &report.row;
        println!(
            "{}\t{}\t{}\t{}\t{}",
            row.id,
            row.module,
            row.workflow,
            row.status.label(),
            row.requirement
        );
        if detail {
            println!("  evidence: {}", row.evidence);
            if let Some(next_action) = compatibility_next_action(row) {
                println!("  next: {}", next_action);
            }
            if let Some(verification_gate) = compatibility_verification_gate(row) {
                println!("  verify: {}", verification_gate);
            }
            if let Some(fixture_status) =
                compatibility_fixture_status(row, &report.missing_fixtures)
            {
                println!("  fixtures: {fixture_status}");
            }
        }
    }

    let open_ids = open_row_ids(&selected_rows);
    if !open_ids.is_empty() {
        println!("open row ids: {}", open_ids.join(", "));
    }
    if !missing_fixtures.is_empty() {
        println!("missing fixture groups: {}", missing_fixtures.join("; "));
    }

    let missing_fixture_manifests = if fail_on_missing_fixtures {
        let missing_manifests = missing_fixture_manifest_directories(Path::new("."));
        if !missing_manifests.is_empty() {
            println!(
                "warning: {} golden fixture director{} missing manifest.json (run `xtask generate-golden` to regenerate with provenance): {}",
                missing_manifests.len(),
                if missing_manifests.len() == 1 {
                    "y"
                } else {
                    "ies"
                },
                missing_manifests.join(", ")
            );
        }
        missing_manifests
    } else {
        Vec::new()
    };

    let stale_fixtures = stale_fixture_manifest_directories(Path::new("."));
    if !stale_fixtures.is_empty() {
        println!(
            "stale fixture(s) (manifest feff10 rev does not match the current feff10/ checkout): {}",
            stale_fixtures.join("; ")
        );
    }

    warn_on_dangling_evidence_references();
    if let Some(json_out) = json_out {
        write_compatibility_json_report(
            json_out,
            &CompatibilityJsonReport {
                summary,
                row_reports: &row_reports,
                open_ids: &open_ids,
                missing_fixtures: &missing_fixtures,
                missing_fixture_manifests: &missing_fixture_manifests,
                stale_fixtures: &stale_fixtures,
                module_filters,
                row_filters,
                open_only,
            },
        )?;
        println!("wrote compatibility json: {}", json_out.display());
    }

    if (fail_on_open && !open_ids.is_empty())
        || (fail_on_missing_fixtures && !missing_fixtures.is_empty())
        || (fail_on_missing_fixtures && !missing_fixture_manifests.is_empty())
        || (fail_on_stale_fixtures && !stale_fixtures.is_empty())
    {
        io::stdout().flush()?;
        let mut failures = Vec::new();
        if fail_on_open && !open_ids.is_empty() {
            failures.push(format!(
                "{} compatibility row(s) are still open: {} need coverage, {} need implementation: {}",
                open_ids.len(),
                summary.needs_coverage,
                summary.needs_implementation,
                open_ids.join(", ")
            ));
        }
        if fail_on_missing_fixtures && !missing_fixtures.is_empty() {
            failures.push(format!(
                "{} required fixture group(s) are missing: {}",
                missing_fixtures.len(),
                missing_fixtures.join("; ")
            ));
        }
        if fail_on_missing_fixtures && !missing_fixture_manifests.is_empty() {
            failures.push(format!(
                "{} golden fixture director{} missing manifest.json: {}",
                missing_fixture_manifests.len(),
                if missing_fixture_manifests.len() == 1 {
                    "y is"
                } else {
                    "ies are"
                },
                missing_fixture_manifests.join(", ")
            ));
        }
        if fail_on_stale_fixtures && !stale_fixtures.is_empty() {
            failures.push(format!(
                "{} golden fixture director{} stale relative to the current feff10/ checkout: {}",
                stale_fixtures.len(),
                if stale_fixtures.len() == 1 {
                    "y is"
                } else {
                    "ies are"
                },
                stale_fixtures.join("; ")
            ));
        }
        anyhow::bail!("{}", failures.join("; "));
    }
    Ok(())
}

/// Every `reference-work/golden` directory a compatibility-matrix fixture
/// requirement points at: a `FixtureRequirement::DirectoryFiles` directory
/// directly, or (for a `FixtureRequirement::File` requirement pointing at a
/// `REFERENCE.zip`) that zip's parent directory. `AnyDirectoryWithPrefixFiles`
/// requirements point at ephemeral `reference-work/tmp` test output, not
/// `generate-golden` fixture trees, so they are excluded.
pub(crate) fn golden_fixture_directories() -> Vec<&'static str> {
    let mut directories = Vec::new();
    for row in compatibility_rows() {
        for requirement in compatibility_fixture_requirements(row) {
            match requirement {
                FixtureRequirement::File(path) => {
                    if let Some(parent) = path.strip_suffix("/REFERENCE.zip") {
                        directories.push(parent);
                    }
                }
                FixtureRequirement::DirectoryFiles { directory, .. } => {
                    directories.push(directory);
                }
                FixtureRequirement::AnyDirectoryWithPrefixFiles { .. } => {}
            }
        }
    }
    directories.sort_unstable();
    directories.dedup();
    directories
}

/// Golden fixture directories that exist on disk but carry no
/// `manifest.json` (F2): present, but with no recorded FEFF10
/// commit/compiler/checksum provenance.
fn missing_fixture_manifest_directories(root: &Path) -> Vec<String> {
    golden_fixture_directories()
        .into_iter()
        .filter(|directory| {
            let path = root.join(directory);
            path.is_dir() && !crate::manifest::has_manifest(&path)
        })
        .map(str::to_string)
        .collect()
}

/// Golden fixture directories whose `manifest.json` records a `feff10_rev`
/// that no longer matches `git -C feff10 rev-parse HEAD` (F2). Directories
/// with no manifest, or when the current `feff10/` checkout's revision can't
/// be determined, are not reported as stale (there is nothing to compare
/// against).
fn stale_fixture_manifest_directories(root: &Path) -> Vec<String> {
    let Some(current_rev) = crate::manifest::feff10_git_rev(&root.join("feff10")) else {
        return Vec::new();
    };
    golden_fixture_directories()
        .into_iter()
        .filter_map(|directory| {
            let path = root.join(directory);
            if !crate::manifest::has_manifest(&path) {
                return None;
            }
            match crate::manifest::read_manifest(&path) {
                Ok(manifest) => {
                    let recorded = manifest.feff10_rev.as_deref();
                    (recorded != Some(current_rev.as_str())).then(|| {
                        format!(
                            "{directory}: manifest feff10 rev {} != current feff10 rev {current_rev}",
                            recorded.unwrap_or("<unrecorded>")
                        )
                    })
                }
                Err(error) => Some(format!(
                    "{directory}: failed to read manifest.json: {error:#}"
                )),
            }
        })
        .collect()
}

/// Non-fatal warning surfaced from every `compatibility-matrix` invocation:
/// flags evidence/verification_gate strings whose referenced test filter no
/// longer resolves to any `#[test]` function in the workspace. Use the
/// `verify-evidence` subcommand for a hard gate.
fn warn_on_dangling_evidence_references() {
    match crate::verify_evidence::dangling_evidence_references(
        &crate::verify_evidence::default_workspace_root(),
    ) {
        Ok(dangling) if dangling.is_empty() => {}
        Ok(dangling) => {
            println!(
                "warning: {} compatibility-matrix evidence reference(s) do not match any workspace #[test] (run `xtask verify-evidence` for details)",
                dangling.len()
            );
        }
        Err(error) => {
            println!(
                "warning: failed to verify compatibility-matrix evidence references: {error:#}"
            );
        }
    }
}

fn compatibility_row_reports(
    root: &Path,
    rows: &[CompatibilityRow],
    open_only: bool,
) -> Result<Vec<CompatibilityRowReport>> {
    rows.iter()
        .map(|row| {
            Ok(CompatibilityRowReport {
                row: *row,
                missing_fixtures: missing_fixture_requirements(root, row)?,
                display: !open_only || row_is_open(row),
            })
        })
        .collect()
}

fn missing_fixture_groups(row_reports: &[CompatibilityRowReport]) -> Vec<String> {
    row_reports
        .iter()
        .flat_map(|report| {
            report
                .missing_fixtures
                .iter()
                .map(|missing| format!("{}: {}", report.row.id, missing))
        })
        .collect()
}

fn write_compatibility_json_report(
    path: &Path,
    report: &CompatibilityJsonReport<'_>,
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, compatibility_json_report(report)?)?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct CompatibilitySummaryJson {
    rows: usize,
    covered: usize,
    needs_coverage: usize,
    needs_implementation: usize,
    open: usize,
}

#[derive(Debug, Serialize)]
struct CompatibilityFiltersJson<'a> {
    modules: &'a [String],
    rows: &'a [String],
    open_only: bool,
}

#[derive(Debug, Serialize)]
struct CompatibilityRowJson<'a> {
    id: &'static str,
    module: &'static str,
    workflow: &'static str,
    status: &'static str,
    requirement: &'static str,
    evidence: &'static str,
    displayed: bool,
    next: Option<&'static str>,
    verify: Option<&'static str>,
    fixture_groups: usize,
    missing_fixtures: &'a [String],
}

#[derive(Debug, Serialize)]
struct CompatibilityReportJson<'a> {
    summary: CompatibilitySummaryJson,
    filters: CompatibilityFiltersJson<'a>,
    open_row_ids: &'a [&'a str],
    missing_fixture_groups: &'a [String],
    /// Golden fixture directories present on disk with no `manifest.json`
    /// (F2); populated only when `--fail-on-missing-fixtures` was passed.
    missing_fixture_manifests: &'a [String],
    /// Golden fixture directories whose `manifest.json` records a stale
    /// `feff10_rev` relative to the current `feff10/` checkout (F2).
    stale_fixtures: &'a [String],
    rows: Vec<CompatibilityRowJson<'a>>,
}

fn compatibility_json_report(report: &CompatibilityJsonReport<'_>) -> Result<String> {
    let json = CompatibilityReportJson {
        summary: CompatibilitySummaryJson {
            rows: report.summary.total,
            covered: report.summary.covered,
            needs_coverage: report.summary.needs_coverage,
            needs_implementation: report.summary.needs_implementation,
            open: report.summary.open(),
        },
        filters: CompatibilityFiltersJson {
            modules: report.module_filters,
            rows: report.row_filters,
            open_only: report.open_only,
        },
        open_row_ids: report.open_ids,
        missing_fixture_groups: report.missing_fixtures,
        missing_fixture_manifests: report.missing_fixture_manifests,
        stale_fixtures: report.stale_fixtures,
        rows: report
            .row_reports
            .iter()
            .map(|row_report| {
                let row = row_report.row;
                CompatibilityRowJson {
                    id: row.id,
                    module: row.module,
                    workflow: row.workflow,
                    status: row.status.label(),
                    requirement: row.requirement,
                    evidence: row.evidence,
                    displayed: row_report.display,
                    next: compatibility_next_action(&row),
                    verify: compatibility_verification_gate(&row),
                    fixture_groups: compatibility_fixture_requirement_count(&row),
                    missing_fixtures: &row_report.missing_fixtures,
                }
            })
            .collect(),
    };
    serde_json::to_string_pretty(&json).context("failed to serialize compatibility matrix json")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompatibilityEvidenceEntry {
    pub(crate) id: &'static str,
    pub(crate) evidence: &'static str,
    pub(crate) verification_gate: Option<&'static str>,
}

/// Evidence and verification-gate strings for every compatibility-matrix row
/// (not just the open ones), used by the `verify-evidence` self-check.
pub(crate) fn compatibility_evidence_entries() -> Vec<CompatibilityEvidenceEntry> {
    compatibility_rows()
        .iter()
        .map(|row| CompatibilityEvidenceEntry {
            id: row.id,
            evidence: row.evidence,
            verification_gate: compatibility_verification_gate(row),
        })
        .collect()
}

pub(crate) fn compatibility_open_items() -> Vec<CompatibilityOpenItem> {
    match compatibility_open_items_with_root(Path::new(".")) {
        Ok(items) => items,
        Err(error) => {
            let message = format!("fixture audit failed: {error:#}");
            compatibility_rows()
                .iter()
                .filter(|row| row_is_open(row))
                .map(|row| {
                    let fixture_groups = compatibility_fixture_requirement_count(row);
                    CompatibilityOpenItem {
                        id: row.id,
                        module: row.module,
                        workflow: row.workflow,
                        status: row.status.label(),
                        requirement: row.requirement,
                        next_action: compatibility_next_action(row),
                        verification_gate: compatibility_verification_gate(row),
                        fixture_groups,
                        missing_fixtures: (fixture_groups > 0)
                            .then(|| message.clone())
                            .into_iter()
                            .collect(),
                    }
                })
                .collect()
        }
    }
}

pub(crate) fn compatibility_open_items_with_root(
    root: &Path,
) -> Result<Vec<CompatibilityOpenItem>> {
    compatibility_rows()
        .iter()
        .filter(|row| row_is_open(row))
        .map(|row| {
            let fixture_groups = compatibility_fixture_requirement_count(row);
            Ok(CompatibilityOpenItem {
                id: row.id,
                module: row.module,
                workflow: row.workflow,
                status: row.status.label(),
                requirement: row.requirement,
                next_action: compatibility_next_action(row),
                verification_gate: compatibility_verification_gate(row),
                fixture_groups,
                missing_fixtures: missing_fixture_requirements(root, row)?,
            })
        })
        .collect()
}

fn compatibility_summary(rows: &[CompatibilityRow]) -> CompatibilitySummary {
    let mut summary = CompatibilitySummary {
        total: rows.len(),
        covered: 0,
        needs_coverage: 0,
        needs_implementation: 0,
    };
    for row in rows {
        match row.status {
            CompatibilityStatus::Covered => summary.covered += 1,
            CompatibilityStatus::NeedsCoverage => summary.needs_coverage += 1,
            CompatibilityStatus::NeedsImplementation => summary.needs_implementation += 1,
        }
    }
    summary
}

fn compatibility_rows() -> &'static [CompatibilityRow] {
    &COMPATIBILITY_ROWS
}

fn filtered_rows(
    rows: &'static [CompatibilityRow],
    module_filters: &[String],
    row_filters: &[String],
) -> Vec<CompatibilityRow> {
    rows.iter()
        .copied()
        .filter(|row| {
            (module_filters.is_empty() || row_matches_module_filters(row, module_filters))
                && (row_filters.is_empty() || row_matches_row_filters(row, row_filters))
        })
        .collect()
}

fn row_matches_module_filters(row: &CompatibilityRow, module_filters: &[String]) -> bool {
    module_filters
        .iter()
        .any(|module| row.module.eq_ignore_ascii_case(module))
}

fn row_matches_row_filters(row: &CompatibilityRow, row_filters: &[String]) -> bool {
    row_filters.iter().any(|id| row.id.eq_ignore_ascii_case(id))
}

fn filter_summary(module_filters: &[String], row_filters: &[String]) -> String {
    let mut parts = Vec::new();
    if !module_filters.is_empty() {
        parts.push(format!("module={}", module_filters.join(", ")));
    }
    if !row_filters.is_empty() {
        parts.push(format!("row={}", row_filters.join(", ")));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("; ")
    }
}

fn row_is_open(row: &CompatibilityRow) -> bool {
    matches!(
        row.status,
        CompatibilityStatus::NeedsCoverage | CompatibilityStatus::NeedsImplementation
    )
}

fn open_row_ids(rows: &[CompatibilityRow]) -> Vec<&'static str> {
    rows.iter()
        .filter(|row| row_is_open(row))
        .map(|row| row.id)
        .collect()
}

fn compatibility_next_action(row: &CompatibilityRow) -> Option<&'static str> {
    if !row_is_open(row) {
        return None;
    }

    match row.id {
        "pot.scf-retry-exhaustion" => Some(
            "add FEFF reference fixtures for successful convergence, iteration-limit final output, retry exhaustion, and retry branches, then gate matching POT source-driver parity",
        ),
        "xsph.tdlda-pmbse" => Some(
            "add real FEFF xsedge.dat fixtures for occupied, file-basis, and generated-basis TDLDA/PMBSE paths and compare generated xsedge.dat numerically",
        ),
        "band.generated-bandstructure" => Some(
            "broaden bandstructure.dat FEFF parity fixtures beyond Cr2GeC across ordinary/freeprop, spin, relativistic, non-degenerate, and KKR/freeprop branches",
        ),
        "ldos.spin-hubbard-full-potential" => Some(
            "implement source generation for spin-Hubbard magnetic final tables from magnetic-resolved LDOS/FMS arrays, then add full-potential ldos/rhoc/lmdos/rhocm parity fixtures",
        ),
        _ => None,
    }
}

fn compatibility_verification_gate(row: &CompatibilityRow) -> Option<&'static str> {
    if !row_is_open(row) {
        return None;
    }

    match row.id {
        "pot.scf-retry-exhaustion" => Some(
            "cargo test --profile release -p refeff-engine pot_scf && cargo run --profile release -p xtask -- compatibility-matrix --row pot.scf-retry-exhaustion --fail-on-open",
        ),
        "xsph.tdlda-pmbse" => Some(
            "cargo test --profile release -p refeff-engine tdlda_xsedge && cargo run --profile release -p xtask -- compatibility-matrix --row xsph.tdlda-pmbse --fail-on-open",
        ),
        "band.generated-bandstructure" => Some(
            "cargo test --profile release -p refeff-engine bandstructure && cargo run --profile release -p xtask -- compatibility-matrix --row band.generated-bandstructure --fail-on-open",
        ),
        "ldos.spin-hubbard-full-potential" => Some(
            "cargo test --profile release -p refeff-engine ldos && cargo run --profile release -p xtask -- compatibility-matrix --row ldos.spin-hubbard-full-potential --fail-on-open",
        ),
        _ => None,
    }
}

const XSPH_SOURCE_REFERENCE_FILES: &[&str] = &[
    "xsph.inp",
    "global.inp",
    "pot.bin",
    "config.dat",
    "phase.bin",
    "xsect.dat",
];
const XSPH_CURRENT_SOURCE_REFERENCE_FILES: &[&str] = &[
    "xsph.inp",
    "global.inp",
    "pot.bin",
    "pot.inp",
    "geom.dat",
    "config.dat",
    "phase.bin",
    "xsect.dat",
];
const BAND_CR2GEC_GENERATED_FILES: &[&str] = &[
    "band.inp",
    "reciprocal.inp",
    "fms.inp",
    "global.inp",
    "phase.bin",
    "bandstructure.dat",
];
const LDOS_EXPECTED_TABLE_FILES: &[&str] = &[
    "ldos.inp",
    "ldos00.dat",
    "ldos01.dat",
    "rhoc00.dat",
    "rhoc01.dat",
];
const LDOS_FMS_SOURCE_REFERENCE_FILES: &[&str] = &[
    "ldos.inp",
    "pot.bin",
    "config.dat",
    "phase.bin",
    "pot.inp",
    "fms.inp",
    "global.inp",
    "geom.dat",
    ".dimensions.dat",
    "gtr00.bin",
    "gtr01.bin",
    "ldos00.dat",
    "ldos01.dat",
    "rhoc00.dat",
    "rhoc01.dat",
];
const LDOS_ORDINARY_SPIN_FMS_SOURCE_REFERENCE_FILES: &[&str] = &[
    "ldos.inp",
    "pot.bin",
    "config.dat",
    "phase.bin",
    "pot.inp",
    "fms.inp",
    "global.inp",
    "geom.dat",
    "xsph.inp",
    ".dimensions.dat",
    "gtr00.bin",
    "gtr01.bin",
    "ldos00.dat",
    "ldos01.dat",
    "rhoc00.dat",
    "rhoc01.dat",
];
const TDLDA_XSEDGE_REFERENCE_FILES: &[&str] = &[
    "xsph.inp",
    "global.inp",
    "pot.bin",
    "config.dat",
    "xsedge.dat",
];
const LDOS_FULL_POTENTIAL_REFERENCE_FILES: &[&str] = &[
    "ldos.inp",
    "pot.bin",
    "config.dat",
    "phase.bin",
    "gtr00.bin",
    "gtr01.bin",
    "ldos00.dat",
    "rhoc00.dat",
    "lmdos00.dat",
    "rhocm00.dat",
];

fn compatibility_fixture_requirements(row: &CompatibilityRow) -> Vec<FixtureRequirement> {
    match row.id {
        "pot.scf-retry-exhaustion" => {
            vec![
                FixtureRequirement::File("reference-work/golden/HUBBARD/NiO/REFERENCE.zip"),
                FixtureRequirement::File("reference-work/golden/XANES/BN/REFERENCE.zip"),
                FixtureRequirement::File("reference-work/golden/XANES/GeCl_4/REFERENCE.zip"),
            ]
        }
        "atomic.finite-nucleus" | "atomic.finite-nucleus-full-range" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/HIGHZ",
                files: &["feff.inp", "HighZ.out"],
            }]
        }
        "workflow.xanes-bn-executed-parity" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/XANES/BN",
                files: &["feff.inp", "xmu.dat"],
            }]
        }
        "xsph.broader-source-phase-xsect" => [
            "reference-work/golden/DEBYE/DM/EXAFS/Cu",
            "reference-work/golden/DEBYE/DM/XANES/Cu",
            "reference-work/golden/ELNES/Cu",
            "reference-work/golden/EXAFS/Cu_SCF",
            "reference-work/golden/LDOS/XANES_Cu_fms",
            "reference-work/golden/LDOS/XANES_Cu_spin_fms_short",
            "reference-work/golden/LDOS/XANES_Cu_spin_no_fms",
        ]
        .into_iter()
        .map(|directory| FixtureRequirement::DirectoryFiles {
            directory,
            files: XSPH_SOURCE_REFERENCE_FILES,
        })
        .collect(),
        "xsph.remaining-phase-branches" => {
            let mut requirements = [
                "reference-work/golden/DANES/BN/REFERENCE.zip",
                "reference-work/golden/DANES/GeCl_4/REFERENCE.zip",
                "reference-work/golden/NRIXS/MgB2/REFERENCE.zip",
                "reference-work/golden/XANES/BN/REFERENCE.zip",
                "reference-work/golden/XANES/GeCl_4/REFERENCE.zip",
                "reference-work/golden/XES/BN/REFERENCE.zip",
                "reference-work/golden/XES/GeCl_4/REFERENCE.zip",
                "reference-work/golden/XMCD/Gd_L1/REFERENCE.zip",
                "reference-work/golden/XMCD/MnF2_SPXAS/REFERENCE.zip",
            ]
            .into_iter()
            .map(FixtureRequirement::File)
            .collect::<Vec<_>>();
            requirements.extend(
                [
                    "reference-work/golden/NRIXS/MgB2",
                    "reference-work/golden/XMCD/Gd_L1",
                    "reference-work/golden/XMCD/MnF2_SPXAS",
                ]
                .into_iter()
                .map(|directory| FixtureRequirement::DirectoryFiles {
                    directory,
                    files: XSPH_CURRENT_SOURCE_REFERENCE_FILES,
                }),
            );
            requirements
        }
        "xsph.tdlda-pmbse" => {
            vec![
                FixtureRequirement::File("reference-work/golden/MPSE/Cu/REFERENCE.zip"),
                FixtureRequirement::File("reference-work/golden/MPSE/Cu_OPCONS/REFERENCE.zip"),
                FixtureRequirement::AnyDirectoryWithPrefixFiles {
                    parent: "reference-work/tmp",
                    prefix: "feff-xsedge-",
                    files: TDLDA_XSEDGE_REFERENCE_FILES,
                },
            ]
        }
        "fms.nrixs-jas-source-mkgtr" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/NRIXS/GeCl_4",
                files: &[
                    "fms.inp",
                    "global.inp",
                    "phase.bin",
                    "gg.bin",
                    "fms.bin",
                    "gtr.dat",
                    "fmsl.bin",
                    "gtrl.dat",
                ],
            }]
        }
        "band.cr2gec-generated-output" | "band.scheduler-cr2gec-reference-parity" => {
            vec![FixtureRequirement::AnyDirectoryWithPrefixFiles {
                parent: "reference-work/tmp",
                prefix: "feff-band-cr2gec.",
                files: BAND_CR2GEC_GENERATED_FILES,
            }]
        }
        "band.generated-bandstructure" => {
            vec![
                FixtureRequirement::File("reference-work/golden/KSPACE/Cr2GeC/REFERENCE.zip"),
                FixtureRequirement::File("reference-work/golden/KSPACE/Graphite/REFERENCE.zip"),
            ]
        }
        "ldos.production-fms-final-tables" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/LDOS/XANES_Cu_fms",
                files: LDOS_FMS_SOURCE_REFERENCE_FILES,
            }]
        }
        "ldos.nonzero-fms-reference-parity" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/LDOS/XANES_Cu_fms_short",
                files: LDOS_FMS_SOURCE_REFERENCE_FILES,
            }]
        }
        "ldos.ordinary-spin-fms-reference-parity" => {
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/LDOS/XANES_Cu_spin_fms_short",
                files: LDOS_ORDINARY_SPIN_FMS_SOURCE_REFERENCE_FILES,
            }]
        }
        "ldos.hubbard-nio-magnetic-sidecars" => {
            vec![FixtureRequirement::File(
                "reference-work/golden/HUBBARD/NiO/REFERENCE.zip",
            )]
        }
        "ldos.spin-hubbard-full-potential" => {
            vec![
                FixtureRequirement::File("reference-work/golden/HUBBARD/NiO/REFERENCE.zip"),
                FixtureRequirement::File("reference-work/golden/HUBBARD/CeO2/REFERENCE.zip"),
                FixtureRequirement::AnyDirectoryWithPrefixFiles {
                    parent: "reference-work/tmp",
                    prefix: "feff-ldos-spin-hubbard-full-potential.",
                    files: LDOS_FULL_POTENTIAL_REFERENCE_FILES,
                },
            ]
        }
        "ldos.scheduler-no-fms-final-tables" => {
            vec![
                FixtureRequirement::DirectoryFiles {
                    directory: "reference-work/golden/XANES/Cu",
                    files: &["pot.bin", "config.dat", "phase.bin", "pot.inp", "fms.inp"],
                },
                FixtureRequirement::DirectoryFiles {
                    directory: "reference-work/golden/LDOS/XANES_Cu_no_fms",
                    files: LDOS_EXPECTED_TABLE_FILES,
                },
            ]
        }
        "rhorrp.generated-density-fixture" => [
            "reference-work/golden/XANES/Cu/rhorrp-density/density.inp",
            "reference-work/golden/XANES/Cu/rhorrp-density/density.dat",
            "reference-work/golden/XANES/Cu/rhorrp-density/density.bin",
            "reference-work/golden/XANES/Cu/rhorrp-density/gg_slice.bin",
            "reference-work/golden/XANES/Cu/rhorrp-density/gg_diag.bin",
        ]
        .into_iter()
        .map(FixtureRequirement::File)
        .collect(),
        _ => Vec::new(),
    }
}

fn compatibility_fixture_requirement_count(row: &CompatibilityRow) -> usize {
    compatibility_fixture_requirements(row).len()
}

fn compatibility_fixture_status(row: &CompatibilityRow, missing: &[String]) -> Option<String> {
    let count = compatibility_fixture_requirement_count(row);
    if count == 0 {
        return None;
    }
    if missing.is_empty() {
        Some(format!("ok ({count} required local fixture group(s))"))
    } else {
        Some(format!(
            "missing {}/{} required local fixture group(s): {}",
            missing.len(),
            count,
            missing.join("; ")
        ))
    }
}

fn missing_fixture_requirements(root: &Path, row: &CompatibilityRow) -> Result<Vec<String>> {
    compatibility_fixture_requirements(row)
        .into_iter()
        .filter_map(|requirement| missing_fixture_requirement(root, requirement).transpose())
        .collect()
}

fn missing_fixture_requirement(
    root: &Path,
    requirement: FixtureRequirement,
) -> Result<Option<String>> {
    match requirement {
        FixtureRequirement::File(path) => {
            let path = root.join(path);
            Ok((!path.is_file()).then(|| path.display().to_string()))
        }
        FixtureRequirement::DirectoryFiles { directory, files } => {
            let directory = root.join(directory);
            let missing_files = missing_files_in_directory(&directory, files);
            Ok((!missing_files.is_empty()).then(|| {
                format!(
                    "{} [{}]",
                    directory.display(),
                    missing_files
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }))
        }
        FixtureRequirement::AnyDirectoryWithPrefixFiles {
            parent,
            prefix,
            files,
        } => {
            let parent = root.join(parent);
            if !parent.is_dir() {
                return Ok(Some(prefixed_directory_requirement_message(
                    &parent, prefix, files,
                )));
            }
            let mut candidates = Vec::new();
            for entry in std::fs::read_dir(&parent)? {
                let entry = entry?;
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if path.is_dir()
                    && name.starts_with(prefix)
                    && missing_files_in_directory(&path, files).is_empty()
                {
                    candidates.push(path);
                }
            }
            candidates.sort();
            Ok(candidates
                .is_empty()
                .then(|| prefixed_directory_requirement_message(&parent, prefix, files)))
        }
    }
}

fn prefixed_directory_requirement_message(parent: &Path, prefix: &str, files: &[&str]) -> String {
    format!(
        "{} child directory matching {prefix}* with [{}]",
        parent.display(),
        files.join(", ")
    )
}

fn missing_files_in_directory(directory: &Path, files: &[&str]) -> Vec<String> {
    files
        .iter()
        .filter(|file| !directory.join(file).is_file())
        .map(|file| (*file).to_string())
        .collect()
}

static COMPATIBILITY_ROWS: [CompatibilityRow; 98] = [
    CompatibilityRow {
        id: "full-run.xanes-smoke",
        module: "feff",
        workflow: "XANES",
        requirement: "fresh Rust full run reaches xmu.dat from feff.inp",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_completes_minimal_cu_smoke_input",
    },
    CompatibilityRow {
        id: "workflow.xanes-bn-executed-parity",
        module: "feff",
        workflow: "XANES",
        requirement: "fresh pinned-FEFF and Rust XANES/BN runs compare the canonical xmu.dat output",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_bn_xsect_keeps_feff_photon_prefactor_and_ixc0_transition_moments && cargo test --profile release -p xtask canonical_golden_output_wins_over_legacy_reference_alias; fresh pinned canonical output passes cargo run --profile release -p xtask -- parity --example XANES/BN",
    },
    CompatibilityRow {
        id: "rdinp.module-handoffs",
        module: "rdinp",
        workflow: "input",
        requirement: "FEFF cards produce typed module handoff files",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine rdinp_stage_writes_supported_outputs_to_requested_dir",
    },
    CompatibilityRow {
        id: "rdinp.debye-invalid-selector-fallback",
        module: "rdinp",
        workflow: "DEBYE/input",
        requirement: "DEBYE selectors above five warn and normalize to the FEFF fallback selector in downstream handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine rdinp_stage_cleanly_normalizes_unavailable_debye_selector && cargo test --profile release -p refeff-io normalizes_unavailable_debye_selector_in_handoffs_and_logs",
    },
    CompatibilityRow {
        id: "pot.source-generation",
        module: "pot",
        workflow: "potential",
        requirement: "source-backed potential generation can create pot.bin",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports pot as source-handoff supported",
    },
    CompatibilityRow {
        id: "pot.bounded-scf-feff-parity",
        module: "pot",
        workflow: "potential",
        requirement: "bounded NiO Hubbard and BN positive-totvol SCF source runs match FEFF pot.bin",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine pot_module_matches_ covers NiO Hubbard and BN positive-totvol module parity",
    },
    CompatibilityRow {
        id: "pot.scheduler-bounded-scf-feff-parity",
        module: "pot",
        workflow: "full-run/potential",
        requirement: "full-run scheduler bounded NiO Hubbard and BN positive-totvol SCF source runs match FEFF pot.bin",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine bounded_feff_pot_reference covers NiO Hubbard and BN positive-totvol scheduler parity",
    },
    CompatibilityRow {
        id: "pot.true-scf-source-outputs",
        module: "pot",
        workflow: "potential",
        requirement: "GeCl4, NiO Hubbard, LDOS spin, and BN positive-totvol true-SCF source runs write POT outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine true_scf_outputs_from_source_handoffs covers four source-backed true-SCF POT output gates",
    },
    CompatibilityRow {
        id: "pot.scheduler-true-scf-source-outputs",
        module: "pot",
        workflow: "full-run/potential",
        requirement: "full-run scheduler reports completed true-SCF POT source outputs for GeCl4, NiO Hubbard, LDOS spin, and BN positive-totvol",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine true_scf_pot plus full_run_scheduler_runs_bn_positive_totvol_pot_source_output cover scheduler POT completion",
    },
    CompatibilityRow {
        id: "pot.no-scf-reference-parity",
        module: "pot",
        workflow: "potential",
        requirement: "SF6, YBCO, MnF2 XMCD, and Gd L1 no-SCF source runs generate POT outputs matching FEFF references",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine reference_no_scf_outputs",
    },
    CompatibilityRow {
        id: "pot.scheduler-no-scf-reference-parity",
        module: "pot",
        workflow: "full-run/potential",
        requirement: "full-run scheduler reports completed no-SCF POT source outputs matching SF6, YBCO, MnF2 XMCD, and Gd L1 FEFF references",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine no_scf_pot_source_output",
    },
    CompatibilityRow {
        id: "pot.high-exchange-scf-source",
        module: "pot",
        workflow: "potential/full-run",
        requirement: "high-EXCHANGE iterative SCF source runs generate and repair POT outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine iterative_scf_outputs_with_high_exchange, high_exchange_iterative, and high_exchange_scf cover module generation plus scheduler source and stale repair gates",
    },
    CompatibilityRow {
        id: "pot.restart-external-scf-source",
        module: "pot",
        workflow: "full-run/potential",
        requirement: "restart, external, and external-restart iterative SCF source runs write POT outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine restart_iterative_scf and external_iterative_scf cover restart, external, and external-restart scheduler output gates",
    },
    CompatibilityRow {
        id: "pot.scf-retry-controls",
        module: "pot",
        workflow: "potential",
        requirement: "SCF retry control updates follow FEFF nstarts mixing and ecv rules",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine updates_scf_retry_controls covers update_scf_pot_retry_controls nstarts behavior",
    },
    CompatibilityRow {
        id: "pot.terminal-scf-final-candidate",
        module: "pot",
        workflow: "potential",
        requirement: "terminal SCF convergence and iteration-limit states are the only states that materialize final pot.bin candidates",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine atomic_module_assembles_terminal_scf_final_pot_candidate",
    },
    CompatibilityRow {
        id: "pot.scf-repeat-exhaustion-boundary",
        module: "pot",
        workflow: "potential",
        requirement: "finite-nucleus iterative SCF repeat-required source loops exhaust bounded FEFF-style start attempts without materializing final POT outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine atomic_module_preserves_saved_scmt_call_state_across_retries",
    },
    CompatibilityRow {
        id: "pot.scf-contour-iteration-core",
        module: "pot",
        workflow: "potential/SCF",
        requirement: "SCF contour stepping, endpoint finishing, source-row lifting, and density/coulomb outer-iteration composition preserve FEFF formulas",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core pot_scf",
    },
    CompatibilityRow {
        id: "pot.scf-source-loop-cli",
        module: "pot",
        workflow: "potential/SCF",
        requirement: "CLI SCF source loops build initial states, advance contours, prepare next iterations, assemble FMS source grids, and write full-run reference POT outputs from source handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine pot_scf",
    },
    CompatibilityRow {
        id: "pot.scf-retry-exhaustion",
        module: "pot",
        workflow: "potential",
        requirement: "SCF retry, convergence, and exhaustion branches match FEFF10",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine pot_scf covers bounded NiO/BN FEFF parity, successful and iteration-limit terminal output, nstarts retry-control updates, finite-nucleus repeat exhaustion, restart/external branches, and source-loop candidate gating",
    },
    CompatibilityRow {
        id: "pot.scf-retry-state-persistence",
        module: "pot",
        workflow: "potential/SCF",
        requirement: "SCF retries preserve the saved SCMT call state needed to resume FEFF contour iteration instead of restarting from a partial state",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine atomic_module_preserves_saved_scmt_call_state_across_retries",
    },
    CompatibilityRow {
        id: "atomic.source-handoff",
        module: "atomic",
        workflow: "atomic",
        requirement: "atomic source/config handoffs run without FEFF output caches",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports atomic as source-handoff supported",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-source-boundary",
        module: "atomic",
        workflow: "atomic/POT",
        requirement: "finite-nucleus APOT/POT source boundaries and repeat handling are exercised",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine finite_nucleus covers atomic, POT, and full-run finite-nucleus source boundaries",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-apot-source-generation",
        module: "atomic",
        workflow: "atomic",
        requirement: "finite-nucleus ATOM source handoffs generate rendered APOT sections from pot.inp plus geom.dat without cached pot.bin",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine atomic_module_generates_finite_nucleus_apot_from_geometry_source_handoff_without_pot_bin",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-no-scf-source-output",
        module: "atomic",
        workflow: "atomic/POT/full-run",
        requirement: "finite-nucleus no-SCF source handoffs generate APOT/POT outputs without final caches",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine finite_nucleus covers atomic APOT, POT pot.bin, and full-run no-SCF finite-nucleus source outputs",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-iterative-repeat-boundary",
        module: "atomic",
        workflow: "atomic/POT/full-run",
        requirement: "finite-nucleus iterative SCF source handoffs reach the bounded repeat boundary",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine finite_nucleus covers atomic, POT, and full-run finite-nucleus iterative repeat-boundary gates",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-generated-scf-state",
        module: "atomic",
        workflow: "atomic",
        requirement: "finite-nucleus generated SCF states select the finite nuclear mesh and differ from point-nucleus generated states",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine atomic_module_generates_finite_nucleus_scf_state_from_pot_input",
    },
    CompatibilityRow {
        id: "atomic.nuclear-potential-core",
        module: "atomic",
        workflow: "atomic/SCF",
        requirement: "point and finite nuclear potentials match FEFF nucdev reference behavior",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core atom_nuclear_potential_matches_feff_nucdev_reference",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-scf-core",
        module: "atomic",
        workflow: "atomic/SCF",
        requirement: "the composed atomic SCF state driver threads finite-nucleus requests through FEFF-style state construction",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core atom_scf_state_from_configuration",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus",
        module: "atomic",
        workflow: "atomic",
        requirement: "finite-nucleus generated-reference parity spans the HIGHZ range",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine finite_nucleus covers source APOT/POT generation, iterative boundaries, FEFF's five-point finite-nucleus request, atomNN.dat output, and 1s binding-energy parity against pinned HIGHZ values for Z=4,29,79,92",
    },
    CompatibilityRow {
        id: "atomic.finite-nucleus-full-range",
        module: "atomic",
        workflow: "atomic/HIGHZ",
        requirement: "finite-nucleus nuclear data and configuration tables cover Z=1 through Z=138, with representative successful HIGHZ binding-energy parity and the typed upstream Z=119 failure",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core finite_nucleus_data_covers_every_supported_highz_atomic_number && cargo test --profile release -p refeff-core feff9_configuration_data_covers_highz_production_range_and_z_plus_one_row && cargo test --profile release -p refeff-engine atomic_finite_nucleus_binding_energies_match_highz_reference_range && cargo test --profile release -p refeff-engine atomic_finite_nucleus_reports_upstream_z119_matching_failure",
    },
    CompatibilityRow {
        id: "xsph.phase-xsect",
        module: "xsph",
        workflow: "XANES",
        requirement: "source-backed phase.bin and xsect.dat are generated",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_completes_minimal_cu_smoke_input",
    },
    CompatibilityRow {
        id: "xsph.broader-source-phase-xsect",
        module: "xsph",
        workflow: "XANES/EXAFS/XES/ELNES/DANES/FPRIME/LDOS",
        requirement: "multi-fixture source handoffs generate phase.bin/xsect.dat matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_matches_broader_source_generated_reference_when_present",
    },
    CompatibilityRow {
        id: "xsph.scheduler-reference-phase-xsect",
        module: "xsph",
        workflow: "full-run XANES/EXAFS/XES/ELNES/DANES/FPRIME/LDOS",
        requirement: "full-run scheduler carries reference-backed source handoffs through completed phase.bin/xsect.dat XSPH reports",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_reference_phase_and_xsect_from_source_handoffs covers nine scheduler reference phase/xsect gates",
    },
    CompatibilityRow {
        id: "xsph.phase-core-branches",
        module: "xsph",
        workflow: "XSPH/phase",
        requirement: "core XSPH phase branch primitives preserve FEFF phase setup, skip, plasmon, radial-output, mesh, and tail formulas",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core xsph_phase_",
    },
    CompatibilityRow {
        id: "xsph.positive-izstd-pmbse-reset",
        module: "xsph",
        workflow: "XANES/TDLDA/PMBSE",
        requirement: "positive-izstd inputs ignore PMBSE controls like FEFF and still generate completed XSPH outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine positive_izstd covers module, scheduler, and full-run positive-izstd PMBSE reset gates",
    },
    CompatibilityRow {
        id: "xsph.scheduler-global-multipole-controls",
        module: "xsph",
        workflow: "full-run XANES",
        requirement: "full-run scheduler carries global multipole controls through completed source-backed XSPH output",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine global_multipole_xsph covers scheduler global multipole source handoffs",
    },
    CompatibilityRow {
        id: "xsph.multipoles-e1-e2-m1",
        module: "xsph",
        workflow: "XSPH/multipoles",
        requirement: "MULTIPOLES=3 generates the additive E1+E2+M1 polarized source result with E1 counted once",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_generates_polarized_multipoles_three_with_additive_source_parity",
    },
    CompatibilityRow {
        id: "xsph.scheduler-two-spin-filtered",
        module: "xsph",
        workflow: "full-run XANES/XMCD",
        requirement: "full-run scheduler carries two-spin filtered XSPH source handoffs through completed outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine two_spin_filtered_xsph covers scheduler and full-run two-spin filtered source handoffs",
    },
    CompatibilityRow {
        id: "xsph.scheduler-ldos-fms-reference-phase-xsect",
        module: "xsph",
        workflow: "full-run LDOS/XANES",
        requirement: "full-run scheduler carries LDOS FMS and ordinary-spin FMS XSPH source handoffs through phase.bin/xsect.dat FEFF reference parity",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_generates_remaining_ldos_xsph_reference_phase_and_xsect_from_source_handoffs",
    },
    CompatibilityRow {
        id: "xsph.nrixs-jas-source-sidecars",
        module: "xsph",
        workflow: "full-run NRIXS/JAS",
        requirement: "full-run scheduler generates NRIXS/JAS phase, xsect.dat, xsecl.dat, xsecl2.dat, and xsecl.bin sidecars from source handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_runs_nrixs_gecl4_xsph_source_handoff; cargo test --profile release -p refeff-engine nrixs_xsectjas",
    },
    CompatibilityRow {
        id: "xsph.cli-branch-source-generation",
        module: "xsph",
        workflow: "XSPH/phase/xsect/sidecars",
        requirement: "CLI source generation covers empty-cell, Hubbard, izstd, FPRIME, E2/L2LP, MPSE, AXAFS, NRIXS, phase-text, and two-spin branch outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_generates_",
    },
    CompatibilityRow {
        id: "xsph.remaining-phase-branches",
        module: "xsph",
        workflow: "XANES",
        requirement: "remaining phase-shift branches match FEFF10 references",
        status: CompatibilityStatus::Covered,
        evidence: "current pinned-FEFF phase.bin/xsect.dat parity is release-gated by xsph_module_matches_current_mnf2_xmcd_phase_and_xsect and xsph_module_matches_current_gd_l1_xmcd_phase_and_xsect; NRIXS phase.bin/xsect.dat plus xsecl.dat/xsecl2.dat/xsecl.bin parity is gated by xsph_module_matches_nrixs_mgb2_phase_xsect_and_sidecars; legacy XANES/XES/DANES/NRIXS/XMCD archives, multi-fixture source handoffs, phase format variants, and scheduler branches remain covered",
    },
    CompatibilityRow {
        id: "xsph.tdlda-pmbse-xsedge-source",
        module: "xsph",
        workflow: "TDLDA/PMBSE",
        requirement: "occupied, file-basis, and generated-basis PMBSE sources generate xsedge.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_writes_tdlda_xsedge covers occupied, file-basis, and generated-basis PMBSE xsedge.dat source generation",
    },
    CompatibilityRow {
        id: "xsph.tdlda-pmbse-scheduler-xsedge",
        module: "xsph",
        workflow: "TDLDA/PMBSE",
        requirement: "full-run scheduler reports completed xsph stages for occupied, file-basis, and generated-basis PMBSE xsedge.dat generation",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine tdlda_xsedge_from_pmbse_source_handoffs covers scheduler TDLDA/PMBSE xsedge.dat generation branches",
    },
    CompatibilityRow {
        id: "xsph.tdlda-pmbse-scheduler-stale-repair",
        module: "xsph",
        workflow: "TDLDA/PMBSE",
        requirement: "full-run scheduler regenerates stale file-basis and generated-basis TDLDA/PMBSE xsedge.dat caches from source",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine tdlda_xsedge_from_pmbse_source_handoffs covers scheduler stale xsedge.dat repair branches",
    },
    CompatibilityRow {
        id: "xsph.tdlda-xsectd-core-formulas",
        module: "xsph",
        workflow: "TDLDA/xsectd",
        requirement: "core TDLDA/xsectd formulas preserve FEFF getmat, mesh/setup, getchi0, ridxmu, kkchi, channel weighting, broadening, and xsedge row assembly",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core xsph_tdlda_",
    },
    CompatibilityRow {
        id: "xsph.tdlda-pmbse",
        module: "xsph",
        workflow: "TDLDA/PMBSE",
        requirement: "TDLDA xsectd/PMBSE source driver has broader FEFF10 parity coverage",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine tdlda_xsedge and cargo test --profile release -p refeff-core xsph_tdlda_ cover occupied, file-basis, generated-basis, stale-repair, projector, response-kernel, and xsedge.dat assembly branches against FEFF fixtures and formulas",
    },
    CompatibilityRow {
        id: "xsph.pmbse-nonlocal-core-hole",
        module: "xsph",
        workflow: "TDLDA/PMBSE",
        requirement: "PMBSE nonlocal core-hole selectors consume pot.ch or yoshi.dat/wscrn.dat source potentials and generate xsedge.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_writes_tdlda_xsedge_from_nonlocal_ covers pot.ch plus yoshi.dat/wscrn.dat source branches",
    },
    CompatibilityRow {
        id: "xsph.tdlda-spin-resolved",
        module: "xsph",
        workflow: "TDLDA/PMBSE/spin",
        requirement: "two-spin PMBSE/TDLDA source handoffs execute both spin paths and merge the resulting xsedge.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xsph_module_writes_two_spin_tdlda_xsedge_from_sources_without_cache && cargo test --profile release -p refeff-engine xsph_tdlda_spin_merge_averages_matching_source_outputs",
    },
    CompatibilityRow {
        id: "exchange.broadened-hl-bphl",
        module: "exchange",
        workflow: "XSPH/exchange",
        requirement: "broadened Hedin-Lundqvist selectors parse an external bphl.dat and dispatch FEFF rhlbp interpolation through XSPH",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-io parses_source_layout_and_restores_implicit_zero_column && cargo test --profile release -p refeff-core broadened_hl_matches_source_table_formula_and_interpolation && cargo test --profile release -p refeff-engine xsph_normal_source_handoff_threads_work_dir_bphl_table",
    },
    CompatibilityRow {
        id: "fms.source-run",
        module: "fms",
        workflow: "XANES/EXAFS",
        requirement: "FMS outputs can be generated or repaired from Rust handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports fms as source-handoff supported",
    },
    CompatibilityRow {
        id: "fms.nrixs-jas-source-mkgtr",
        module: "fms",
        workflow: "NRIXS/JAS",
        requirement: "cache-free NRIXS/JAS FMS source handoffs execute the active rdxsphjas/getgtrjas path and generate MKGTR outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine standalone_fms_and_mkgtr_generate_cache_free_nrixs_jas_outputs && cargo test --profile release -p refeff-engine full_run_scheduler_generates_cache_free_nrixs_jas_fms_and_mkgtr && cargo test --profile release -p refeff-engine mkgtr_module_matches_pinned_nrixs_jas_reference_traces",
    },
    CompatibilityRow {
        id: "paths.source-run",
        module: "paths",
        workflow: "EXAFS",
        requirement: "pathfinder writes paths.dat from Rust geometry/phase inputs",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports paths as source-handoff supported",
    },
    CompatibilityRow {
        id: "genfmt.source-run",
        module: "genfmt",
        workflow: "EXAFS",
        requirement: "GENFMT writes feff.bin from Rust paths and phase data",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports genfmt as source-handoff supported",
    },
    CompatibilityRow {
        id: "ff2x.xmu-chi",
        module: "ff2x",
        workflow: "XANES/EXAFS",
        requirement: "FF2X writes xmu.dat and chi.dat from Rust handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_completes_minimal_cu_smoke_input",
    },
    CompatibilityRow {
        id: "band.cr2gec-generated-output",
        module: "band",
        workflow: "BAND",
        requirement: "Cr2GeC source handoffs generate bandstructure.dat matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine band_cr2gec_generated_bandstructure_matches_reference_when_present",
    },
    CompatibilityRow {
        id: "band.scheduler-cr2gec-reference-parity",
        module: "band",
        workflow: "full-run BAND",
        requirement: "full-run scheduler generates Cr2GeC bandstructure.dat matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_generates_cr2gec_reference_bandstructure_from_source_handoffs",
    },
    CompatibilityRow {
        id: "band.one-spin-rel-source-generation",
        module: "band",
        workflow: "BAND",
        requirement: "one-spin relativistic source handoffs generate completed bandstructure.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine one_spin_rel_bandstructure covers module and scheduler source generation",
    },
    CompatibilityRow {
        id: "band.freeprop-source-generation",
        module: "band",
        workflow: "full-run BAND",
        requirement: "full-run scheduler generates standalone freeprop bandstructure.dat from source handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_generates_freeprop_bandstructure_from_source_handoffs",
    },
    CompatibilityRow {
        id: "band.two-spin-nondegenerate-source-generation",
        module: "band",
        workflow: "BAND",
        requirement: "non-degenerate two-spin ordinary and freeprop source handoffs generate completed bandstructure.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine band_module_generates_two_spin_non_degenerate and full_run_scheduler_generates_two_spin_non_degenerate cover ordinary and freeprop source generation",
    },
    CompatibilityRow {
        id: "band.two-spin-degenerate-source-generation",
        module: "band",
        workflow: "full-run BAND",
        requirement: "full-run scheduler generates degenerate two-spin bandstructure.dat from source handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_generates_two_spin_degenerate_bandstructure_from_source_handoffs",
    },
    CompatibilityRow {
        id: "band.cli-branch-source-generation",
        module: "band",
        workflow: "BAND",
        requirement: "direct BAND module source generation covers ordinary, freeprop, one-spin relativistic, two-spin degenerate, two-spin non-degenerate, and kmesh/pre-solver handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine band_module_generates_",
    },
    CompatibilityRow {
        id: "band.one-spin-rel-stale-repair",
        module: "band",
        workflow: "BAND",
        requirement: "one-spin relativistic ordinary and freeprop stale bandstructure.dat caches regenerate from source",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_regenerates_stale_one_spin_rel covers ordinary and freeprop stale-cache repair",
    },
    CompatibilityRow {
        id: "band.two-spin-stale-repair",
        module: "band",
        workflow: "BAND",
        requirement: "non-degenerate two-spin ordinary and freeprop stale bandstructure.dat caches regenerate from source",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine full_run_scheduler_regenerates_stale_two_spin covers ordinary and freeprop stale-cache repair",
    },
    CompatibilityRow {
        id: "band.kspace-structure-factor-core",
        module: "band",
        workflow: "BAND/KSPACE",
        requirement: "BAND KSPACE non-relativistic and relativistic structure-factor grids preserve FEFF loop order",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core band_structure_factor_from_kspace",
    },
    CompatibilityRow {
        id: "band.kspace-kkr-band-rows-core",
        module: "band",
        workflow: "BAND/KKR",
        requirement: "BAND KKR KSPACE grids identify FEFF bandstructure.dat rows for non-relativistic and relativistic source paths",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core band_kkr_band_energies_from_kspace",
    },
    CompatibilityRow {
        id: "band.kspace-freeprop-band-rows-core",
        module: "band",
        workflow: "BAND/freeprop",
        requirement: "BAND freeprop raw-G KSPACE grids identify FEFF bandstructure.dat rows for non-relativistic and relativistic source paths",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core band_free_propagation_band_energies_from_kspace",
    },
    CompatibilityRow {
        id: "band.faer-general-eigenvalues",
        module: "band",
        workflow: "BAND/faer",
        requirement: "BAND KKR and freeprop eigenvalue solves use pure-Rust faer CGEES-style general complex eigenvalues",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-linalg complex32_general_eigenvalues and cargo check --profile release -p refeff-core",
    },
    CompatibilityRow {
        id: "band.generated-bandstructure",
        module: "band",
        workflow: "BAND",
        requirement: "bandstructure.dat parity covers spin, relativistic, freeprop, and KKR branches",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine bandstructure and cargo test --profile release -p refeff-core band_ cover Cr2GeC FEFF output parity plus ordinary/freeprop, one-spin relativistic, two-spin degenerate/non-degenerate, KKR, KSPACE final-row, and stale-repair branches",
    },
    CompatibilityRow {
        id: "ldos.production-fms-final-tables",
        module: "ldos",
        workflow: "LDOS",
        requirement: "production full-FMS source handoffs generate LDOS/RHOC tables matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_matches_production_fms_reference_from_source_handoffs",
    },
    CompatibilityRow {
        id: "ldos.nonzero-fms-reference-parity",
        module: "ldos",
        workflow: "LDOS/FMS",
        requirement: "nonzero FMS source handoffs generate gtr, LDOS, and RHOC tables matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_matches_nonzero_fms_reference_from_source_handoffs",
    },
    CompatibilityRow {
        id: "ldos.ordinary-spin-fms-reference-parity",
        module: "ldos",
        workflow: "LDOS/FMS/spin",
        requirement: "ordinary spin FMS source handoffs generate gtr, LDOS, and RHOC tables matching FEFF",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_matches_ordinary_spin_fms_reference_from_source_handoffs",
    },
    CompatibilityRow {
        id: "ldos.non-full-potential-ff2rho-core",
        module: "ldos",
        workflow: "LDOS",
        requirement: "non-full-potential ff2rho table assembly preserves FEFF LDOS/RHOC density formulas",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core ldos_ff2rho_tables_match_feff_non_full_potential_reference",
    },
    CompatibilityRow {
        id: "ldos.non-full-potential-fmsdos-trace-core",
        module: "ldos",
        workflow: "LDOS/FMS",
        requirement: "non-full-potential fmsdos trace projection preserves FEFF packed-gg phase normalization loop",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core ldos_fmsdos_trace_matches_feff_non_full_potential_loop",
    },
    CompatibilityRow {
        id: "ldos.hubbard-nio-magnetic-sidecars",
        module: "ldos",
        workflow: "LDOS/Hubbard",
        requirement: "active-Hubbard NiO LDOS cache preserves magnetic sidecar contracts",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_roundtrips_hubbard_nio_reference_zip_magnetic_sidecars",
    },
    CompatibilityRow {
        id: "ldos.scheduler-no-fms-final-tables",
        module: "ldos",
        workflow: "full-run LDOS",
        requirement: "full-run scheduler generates and repairs no-FMS LDOS/RHOC final tables from source handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine xanes_cu_no_fms_ldos covers scheduler source generation and stale-cache repair",
    },
    CompatibilityRow {
        id: "ldos.scheduler-active-hubbard-cache-contract",
        module: "ldos",
        workflow: "full-run LDOS/Hubbard",
        requirement: "full-run scheduler validates and repairs active-Hubbard LDOS cache/source contracts",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine active_hubbard_ldos covers positive, repair, stale-grid/layout, malformed-sidecar, and gtr/gtr_m/gtr_off contract gates",
    },
    CompatibilityRow {
        id: "ldos.active-hubbard-source-contracts",
        module: "ldos",
        workflow: "LDOS/Hubbard",
        requirement: "direct LDOS active-Hubbard caches validate ordinary, magnetic, and off-diagonal source trace contracts",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine active_hubbard_cache covers matching, fallback, conflict, and omitted-potential gtr/gtr_m/gtr_off direct-module contracts",
    },
    CompatibilityRow {
        id: "ldos.hubbard-magnetic-ff2rho-step2",
        module: "ldos",
        workflow: "LDOS/Hubbard",
        requirement: "magnetic Hubbard ff2rho_h_step2 table assembly preserves FEFF lmdos/rhocm ordering and formulas",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core ldos_hubbard_magnetic_ff2rho_tables_match_feff_step2_order",
    },
    CompatibilityRow {
        id: "ldos.active-hubbard-fms-save-gg-slice",
        module: "ldos",
        workflow: "LDOS/FMS/Hubbard",
        requirement: "active-Hubbard full-potential FMS source generation writes gg_slice/gg_diag sidecars consistent with gg.dat",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine fms_module_generates_active_hubbard_saved_scattering_slices",
    },
    CompatibilityRow {
        id: "ldos.active-hubbard-no-fms-magnetic-repair",
        module: "ldos",
        workflow: "LDOS/Hubbard",
        requirement: "active-Hubbard no-FMS LDOS repairs one-sided lmdos/rhocm magnetic sidecars through ff2rho_h_step2 zero-scattering assembly",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine without_fms covers missing lmdos/rhocm and malformed rhocm magnetic sidecar repair",
    },
    CompatibilityRow {
        id: "ldos.cli-source-generation-sweep",
        module: "ldos",
        workflow: "LDOS",
        requirement: "direct LDOS source generation covers kmesh, gtr, no-FMS, wavefunction/radial, zero-cluster FMS, missing-pair, spin-pair, and module-log handoffs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_generates_",
    },
    CompatibilityRow {
        id: "ldos.cli-repair-sweep",
        module: "ldos",
        workflow: "LDOS",
        requirement: "direct LDOS repair covers malformed kmesh/log/output caches, paired LDOS/RHOC recovery, spin RHOC recovery, and no-FMS active-Hubbard ordinary/magnetic sidecars",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_recovers_",
    },
    CompatibilityRow {
        id: "ldos.spin-hubbard-full-potential",
        module: "ldos",
        workflow: "LDOS",
        requirement: "spin-Hubbard and full-potential LDOS branches are parity covered",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine ldos_module_generates_spin_hubbard_independent_center_tables_from_source_handoffs and ldos_hubbard_second_pass_solves_magnetic_radial_source_tables cover fresh two-spin Hubbard independent-center traces, Hubbard potential/transforms, magnetic second-pass solves, and final ldos/rhoc/lmdos/rhocm tables",
    },
    CompatibilityRow {
        id: "eels.mdff-source",
        module: "eels",
        workflow: "EELS/MDFF",
        requirement: "EELS and MDFF handoffs are parsed and source-backed",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports eels and eelsmdff as source-handoff supported",
    },
    CompatibilityRow {
        id: "rixs.source-run",
        module: "rixs",
        workflow: "RIXS",
        requirement: "RIXS solver handoffs generate or validate spectrum output",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports rixs as source-handoff supported",
    },
    CompatibilityRow {
        id: "dmdw.source-run",
        module: "dmdw",
        workflow: "DMDW",
        requirement: "DMDW outputs are generated from Rust phonon/cumulant paths",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports dmdw as source-handoff supported",
    },
    CompatibilityRow {
        id: "dmdw.type2-electron-energy-option",
        module: "dmdw",
        workflow: "DMDW/type2",
        requirement: "type-2 electron-energy option converts only selector one and passes every other selector value through unchanged",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine type2_electron_energy_options_other_than_one_leave_energy_unchanged",
    },
    CompatibilityRow {
        id: "dym2feffinp.production-converter",
        module: "dym2feffinp",
        workflow: "DMDW/input",
        requirement: "the production dym2feffinp executable preserves FEFF option spellings and writes reparsable centered feff.inp and DYM outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-cli matches_pinned_production_converter_semantically && cargo test --profile release -p refeff-cli parser_matches_production_option_spellings_and_defaults",
    },
    CompatibilityRow {
        id: "screen.crpa-source",
        module: "screen",
        workflow: "SCREEN/CRPA",
        requirement: "SCREEN and CRPA response outputs are source-backed",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports screen and crpa as source-handoff supported",
    },
    CompatibilityRow {
        id: "sfconv.self-energy",
        module: "sfconv",
        workflow: "self-energy",
        requirement: "SFCONV and SELFENERGY outputs are source-backed",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports sfconv as source-handoff supported",
    },
    CompatibilityRow {
        id: "compton.source-run",
        module: "compton",
        workflow: "COMPTON",
        requirement: "Compton profile outputs are source-backed",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports compton as source-handoff supported",
    },
    CompatibilityRow {
        id: "optics.full-spectrum",
        module: "fullspectrum",
        workflow: "optics",
        requirement: "OPCONS and FULLSPECTRUM optical outputs are source-backed",
        status: CompatibilityStatus::Covered,
        evidence: "FULLSPECTRUM source-generation tests assemble background and detailed FPRIME/FMS/path edge handoffs into eps.dat, Drude, and optical tables; xtask port-status reports opcons and fullspectrum source handoffs",
    },
    CompatibilityRow {
        id: "opcons.epsdb-source-generation",
        module: "opcons",
        workflow: "optics",
        requirement: "missing elemental OPCONS tables are generated from FEFF's bundled epsdb for every available element Z=1 through Z=99",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-core bundled_epsdb_matches_feff10_source_rows && cargo test --profile release -p refeff-engine opcons_module_generates_missing_copper_table_from_feff_epsdb",
    },
    CompatibilityRow {
        id: "fullspectrum.xmu-control-six",
        module: "fullspectrum",
        workflow: "optics",
        requirement: "FULLSPECTRUM writes its final xmu.dat and CONTROL(6)=0 suppresses and does not advertise optical post-processing outputs",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine generates_eps_and_optical_tables_from_background_edge_sources && cargo test --profile release -p refeff-engine control_six_",
    },
    CompatibilityRow {
        id: "rhorrp.source-run",
        module: "rhorrp",
        workflow: "density",
        requirement: "RHORRP density/Green handoffs run through Rust",
        status: CompatibilityStatus::Covered,
        evidence: "xtask port-status reports rhorrp as source-handoff supported",
    },
    CompatibilityRow {
        id: "rhorrp.generated-density-fixture",
        module: "rhorrp",
        workflow: "density/FMS",
        requirement: "the generated XANES/Cu nested RHORRP fixture carries density text/binary outputs and the gg_slice/gg_diag source sidecars",
        status: CompatibilityStatus::Covered,
        evidence: "cargo test --profile release -p refeff-engine rhorrp_module_roundtrips_generated_reference_when_present",
    },
    CompatibilityRow {
        id: "release.clippy-workspace",
        module: "release",
        workflow: "release",
        requirement: "workspace clippy with warnings denied completes reliably",
        status: CompatibilityStatus::Covered,
        evidence: "cargo clippy --workspace --all-targets --all-features -- -D warnings passes",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_summary_tracks_open_and_covered_rows() {
        let summary = compatibility_summary(compatibility_rows());

        assert_eq!(summary.total, compatibility_rows().len());
        assert!(summary.covered > 0);
        assert_eq!(
            summary.open(),
            summary.needs_coverage + summary.needs_implementation
        );
    }

    #[test]
    fn open_row_filter_excludes_covered_rows() {
        let rows = compatibility_rows();
        let open_rows = rows.iter().filter(|row| row_is_open(row)).count();
        let summary = compatibility_summary(rows);

        assert_eq!(open_rows, summary.open());
        assert!(rows.iter().any(|row| !row_is_open(row)));
    }

    #[test]
    fn open_row_ids_returns_selected_blocking_rows() {
        let rows = [
            CompatibilityRow {
                id: "covered",
                module: "test",
                workflow: "test",
                requirement: "covered row",
                status: CompatibilityStatus::Covered,
                evidence: "covered",
            },
            CompatibilityRow {
                id: "needs-coverage",
                module: "test",
                workflow: "test",
                requirement: "needs coverage row",
                status: CompatibilityStatus::NeedsCoverage,
                evidence: "missing coverage",
            },
            CompatibilityRow {
                id: "needs-implementation",
                module: "test",
                workflow: "test",
                requirement: "needs implementation row",
                status: CompatibilityStatus::NeedsImplementation,
                evidence: "missing implementation",
            },
        ];

        assert_eq!(
            open_row_ids(&rows),
            vec!["needs-coverage", "needs-implementation"]
        );
    }

    #[test]
    fn every_open_matrix_row_has_next_action() {
        let missing = compatibility_rows()
            .iter()
            .filter(|row| row_is_open(row) && compatibility_next_action(row).is_none())
            .map(|row| row.id)
            .collect::<Vec<_>>();

        assert_eq!(missing, Vec::<&str>::new());
    }

    #[test]
    fn every_open_matrix_row_has_verification_gate() {
        let missing = compatibility_rows()
            .iter()
            .filter(|row| row_is_open(row) && compatibility_verification_gate(row).is_none())
            .map(|row| row.id)
            .collect::<Vec<_>>();

        assert_eq!(missing, Vec::<&str>::new());
    }

    #[test]
    fn open_row_verification_gates_use_release_profile_xtask() {
        let non_release = compatibility_rows()
            .iter()
            .filter(|row| row_is_open(row))
            .filter_map(|row| {
                compatibility_verification_gate(row)
                    .filter(|gate| !gate.contains("cargo run --profile release -p xtask"))
                    .map(|_| row.id)
            })
            .collect::<Vec<_>>();

        assert_eq!(non_release, Vec::<&str>::new());
    }

    #[test]
    fn fixture_audit_reports_missing_and_present_file() -> Result<()> {
        let root = compatibility_fixture_temp_dir("file")?;
        let row = CompatibilityRow {
            id: "ldos.hubbard-nio-magnetic-sidecars",
            module: "ldos",
            workflow: "LDOS/Hubbard",
            requirement: "fixture test",
            status: CompatibilityStatus::Covered,
            evidence: "fixture test",
        };
        let expected_path = root.join("reference-work/golden/HUBBARD/NiO/REFERENCE.zip");

        assert_eq!(
            missing_fixture_requirements(&root, &row)?,
            vec![expected_path.display().to_string()]
        );

        std::fs::create_dir_all(
            expected_path
                .parent()
                .expect("fixture path should have parent"),
        )?;
        std::fs::write(&expected_path, b"fixture")?;
        assert_eq!(
            missing_fixture_requirements(&root, &row)?,
            Vec::<String>::new()
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn fixture_audit_accepts_prefixed_generated_directory() -> Result<()> {
        let root = compatibility_fixture_temp_dir("prefix")?;
        let row = CompatibilityRow {
            id: "band.cr2gec-generated-output",
            module: "band",
            workflow: "BAND",
            requirement: "fixture test",
            status: CompatibilityStatus::Covered,
            evidence: "fixture test",
        };
        let parent = root.join("reference-work/tmp");

        let missing = missing_fixture_requirements(&root, &row)?;
        assert_eq!(missing.len(), 1);
        assert!(
            missing[0].contains("reference-work/tmp"),
            "unexpected missing fixture message: {missing:?}"
        );

        let candidate = parent.join("feff-band-cr2gec.test");
        std::fs::create_dir_all(&candidate)?;
        for file in BAND_CR2GEC_GENERATED_FILES {
            std::fs::write(candidate.join(file), b"fixture")?;
        }
        assert_eq!(
            missing_fixture_requirements(&root, &row)?,
            Vec::<String>::new()
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn open_only_reports_still_audit_hidden_covered_fixture_rows() -> Result<()> {
        let root = compatibility_fixture_temp_dir("open-only")?;
        let rows = [
            CompatibilityRow {
                id: "ldos.hubbard-nio-magnetic-sidecars",
                module: "ldos",
                workflow: "LDOS/Hubbard",
                requirement: "fixture-backed covered row",
                status: CompatibilityStatus::Covered,
                evidence: "fixture test",
            },
            CompatibilityRow {
                id: "xsph.tdlda-pmbse",
                module: "xsph",
                workflow: "TDLDA/PMBSE",
                requirement: "open row",
                status: CompatibilityStatus::NeedsCoverage,
                evidence: "open test",
            },
        ];

        let reports = compatibility_row_reports(&root, &rows, true)?;

        assert_eq!(reports.len(), 2);
        assert!(!reports[0].display);
        assert_eq!(reports[0].row.id, "ldos.hubbard-nio-magnetic-sidecars");
        assert_eq!(reports[0].missing_fixtures.len(), 1);
        assert!(reports[1].display);
        assert_eq!(reports[1].row.id, "xsph.tdlda-pmbse");
        assert_eq!(
            reports[1].missing_fixtures.len(),
            compatibility_fixture_requirement_count(&reports[1].row)
        );
        assert!(
            reports[1]
                .missing_fixtures
                .iter()
                .any(|missing| missing.contains("feff-xsedge-"))
        );
        let missing_groups = missing_fixture_groups(&reports);
        assert!(
            missing_groups
                .iter()
                .any(|missing| missing.contains("ldos.hubbard-nio-magnetic-sidecars:"))
        );
        assert!(
            missing_groups
                .iter()
                .any(|missing| missing.contains("xsph.tdlda-pmbse:"))
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn open_items_include_fixture_prerequisites() -> Result<()> {
        let root = compatibility_fixture_temp_dir("open-items")?;

        let open_items = compatibility_open_items_with_root(&root)?;
        assert_eq!(
            open_items.len(),
            compatibility_summary(compatibility_rows()).open()
        );
        assert!(
            open_items
                .iter()
                .all(|item| item.next_action.is_some() && item.verification_gate.is_some())
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn compatibility_json_report_contains_rows_filters_and_escaped_values() -> Result<()> {
        let root = compatibility_fixture_temp_dir("json")?;
        let row = *compatibility_rows()
            .iter()
            .find(|row| row.id == "xsph.tdlda-pmbse")
            .expect("matrix should include TDLDA/PMBSE row");
        let row_reports = compatibility_row_reports(&root, &[row], false)?;
        let open_ids = open_row_ids(&[row]);
        let missing_fixtures = missing_fixture_groups(&row_reports);
        let module_filters = vec!["xsph".to_string()];
        let row_filters = vec!["xsph.tdlda-pmbse".to_string()];
        let json = compatibility_json_report(&CompatibilityJsonReport {
            summary: compatibility_summary(&[row]),
            row_reports: &row_reports,
            open_ids: &open_ids,
            missing_fixtures: &missing_fixtures,
            missing_fixture_manifests: &[],
            stale_fixtures: &[],
            module_filters: &module_filters,
            row_filters: &row_filters,
            open_only: false,
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("compatibility matrix json should parse");

        assert_eq!(value["open_row_ids"], serde_json::json!([]));
        assert_eq!(value["rows"][0]["status"], "covered");
        assert_eq!(value["filters"]["modules"], serde_json::json!(["xsph"]));
        assert_eq!(
            value["filters"]["rows"],
            serde_json::json!(["xsph.tdlda-pmbse"])
        );
        assert_eq!(value["rows"][0]["fixture_groups"], 3);
        assert!(
            value["missing_fixture_groups"][0]
                .as_str()
                .expect("missing fixture group should be a string")
                .starts_with("xsph.tdlda-pmbse:")
        );
        assert!(json.contains("feff-xsedge-"));
        assert!(json.contains("cargo test --profile release -p refeff-engine tdlda_xsedge"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn covered_rows_do_not_report_open_guidance() {
        let row = compatibility_rows()
            .iter()
            .find(|row| row.status == CompatibilityStatus::Covered)
            .expect("matrix should include covered rows");

        assert_eq!(compatibility_next_action(row), None);
        assert_eq!(compatibility_verification_gate(row), None);
    }

    #[test]
    fn module_filter_selects_only_matching_rows() {
        let rows = compatibility_rows();
        let filters = vec!["XSPH".to_string()];
        let selected = filtered_rows(rows, &filters, &[]);
        let summary = compatibility_summary(&selected);

        assert!(!selected.is_empty());
        assert!(selected.iter().all(|row| row.module == "xsph"));
        assert_eq!(summary.open(), 0);
        assert!(summary.total < rows.len());
    }

    #[test]
    fn module_filter_can_select_multiple_modules() {
        let rows = compatibility_rows();
        let filters = vec!["pot".to_string(), "band".to_string()];
        let selected = filtered_rows(rows, &filters, &[]);

        assert!(selected.iter().any(|row| row.module == "pot"));
        assert!(selected.iter().any(|row| row.module == "band"));
        assert!(
            selected
                .iter()
                .all(|row| row.module == "pot" || row.module == "band")
        );
    }

    #[test]
    fn row_filter_selects_exact_row_case_insensitively() {
        let rows = compatibility_rows();
        let filters = vec!["XSPH.TDLDA-PMBSE".to_string()];
        let selected = filtered_rows(rows, &[], &filters);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "xsph.tdlda-pmbse");
    }

    #[test]
    fn module_and_row_filters_are_intersected() {
        let rows = compatibility_rows();
        let module_filters = vec!["xsph".to_string()];
        let row_filters = vec![
            "xsph.tdlda-pmbse".to_string(),
            "band.generated-bandstructure".to_string(),
        ];
        let selected = filtered_rows(rows, &module_filters, &row_filters);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "xsph.tdlda-pmbse");
    }

    fn compatibility_fixture_temp_dir(prefix: &str) -> Result<std::path::PathBuf> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "refeff-compatibility-fixture-{prefix}-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }

    #[test]
    fn matrix_includes_full_xanes_and_release_gate_rows() {
        let rows = compatibility_rows();

        assert!(rows.iter().any(|row| {
            row.id == "full-run.xanes-smoke" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "release.clippy-workspace" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.bounded-scf-feff-parity" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scheduler-bounded-scf-feff-parity"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.true-scf-source-outputs" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scheduler-true-scf-source-outputs"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.no-scf-reference-parity" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scheduler-no-scf-reference-parity"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.high-exchange-scf-source" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.restart-external-scf-source"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scf-retry-controls" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.terminal-scf-final-candidate"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scf-repeat-exhaustion-boundary"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scf-contour-iteration-core" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "pot.scf-source-loop-cli" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-source-boundary"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-apot-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-no-scf-source-output"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-iterative-repeat-boundary"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-generated-scf-state"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.nuclear-potential-core" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus-scf-core" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "atomic.finite-nucleus" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.broader-source-phase-xsect"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.remaining-phase-branches" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.scheduler-reference-phase-xsect"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.phase-core-branches" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.positive-izstd-pmbse-reset"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.scheduler-global-multipole-controls"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.scheduler-two-spin-filtered"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.scheduler-ldos-fms-reference-phase-xsect"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.nrixs-jas-source-sidecars" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.cli-branch-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.tdlda-pmbse-xsedge-source" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.tdlda-pmbse-scheduler-xsedge"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.tdlda-pmbse-scheduler-stale-repair"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.tdlda-xsectd-core-formulas"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.cr2gec-generated-output" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.one-spin-rel-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.scheduler-cr2gec-reference-parity"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.freeprop-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.two-spin-nondegenerate-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.two-spin-degenerate-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.cli-branch-source-generation"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.one-spin-rel-stale-repair" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.two-spin-stale-repair" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.kspace-structure-factor-core"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.kspace-kkr-band-rows-core" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.kspace-freeprop-band-rows-core"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "band.faer-general-eigenvalues" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.production-fms-final-tables"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.nonzero-fms-reference-parity"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.ordinary-spin-fms-reference-parity"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.non-full-potential-ff2rho-core"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.non-full-potential-fmsdos-trace-core"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.hubbard-nio-magnetic-sidecars"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.scheduler-no-fms-final-tables"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.scheduler-active-hubbard-cache-contract"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.active-hubbard-source-contracts"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.hubbard-magnetic-ff2rho-step2"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.active-hubbard-fms-save-gg-slice"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.active-hubbard-no-fms-magnetic-repair"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.cli-source-generation-sweep"
                && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "ldos.cli-repair-sweep" && row.status == CompatibilityStatus::Covered
        }));
        assert!(rows.iter().any(|row| {
            row.id == "xsph.tdlda-pmbse" && row.status == CompatibilityStatus::Covered
        }));
    }

    #[test]
    fn matrix_keeps_audit_discovered_branches_explicit() {
        let expected_ids = [
            "workflow.xanes-bn-executed-parity",
            "rdinp.debye-invalid-selector-fallback",
            "pot.scf-retry-state-persistence",
            "atomic.finite-nucleus-full-range",
            "xsph.multipoles-e1-e2-m1",
            "xsph.pmbse-nonlocal-core-hole",
            "xsph.tdlda-spin-resolved",
            "exchange.broadened-hl-bphl",
            "dmdw.type2-electron-energy-option",
            "dym2feffinp.production-converter",
            "opcons.epsdb-source-generation",
            "fullspectrum.xmu-control-six",
            "rhorrp.generated-density-fixture",
            "fms.nrixs-jas-source-mkgtr",
        ];

        for id in expected_ids {
            let row = compatibility_rows()
                .iter()
                .find(|row| row.id == id)
                .unwrap_or_else(|| panic!("compatibility matrix is missing explicit row {id}"));
            assert_eq!(
                row.status,
                CompatibilityStatus::Covered,
                "{id} must remain release-gated"
            );
            assert!(
                row.evidence.contains("cargo test"),
                "{id} must cite focused executable test evidence"
            );
        }
    }

    #[test]
    fn audit_discovered_reference_rows_require_canonical_outputs_and_sidecars() {
        let bn = compatibility_rows()
            .iter()
            .find(|row| row.id == "workflow.xanes-bn-executed-parity")
            .expect("matrix should include fresh XANES/BN parity");
        assert_eq!(bn.status, CompatibilityStatus::Covered);
        assert!(bn.evidence.contains(
            "xsph_module_bn_xsect_keeps_feff_photon_prefactor_and_ixc0_transition_moments"
        ));
        assert!(bn.evidence.contains("parity --example XANES/BN"));
        assert_eq!(
            compatibility_fixture_requirements(bn),
            vec![FixtureRequirement::DirectoryFiles {
                directory: "reference-work/golden/XANES/BN",
                files: &["feff.inp", "xmu.dat"],
            }]
        );

        let rhorrp = compatibility_rows()
            .iter()
            .find(|row| row.id == "rhorrp.generated-density-fixture")
            .expect("matrix should include generated RHORRP fixture");
        assert_eq!(compatibility_fixture_requirement_count(rhorrp), 5);
        for required in [
            "density.inp",
            "density.dat",
            "density.bin",
            "gg_slice.bin",
            "gg_diag.bin",
        ] {
            assert!(
                compatibility_fixture_requirements(rhorrp)
                    .iter()
                    .any(|requirement| matches!(
                        requirement,
                        FixtureRequirement::File(path) if path.ends_with(required)
                    )),
                "RHORRP fixture must require {required}"
            );
        }
    }

    #[test]
    fn golden_fixture_directories_include_file_and_directory_requirements() {
        let directories = golden_fixture_directories();
        assert!(
            directories.contains(&"reference-work/golden/HUBBARD/NiO"),
            "expected the REFERENCE.zip File requirement's parent directory in {directories:?}"
        );
        assert!(
            directories.contains(&"reference-work/golden/HIGHZ"),
            "expected the DirectoryFiles requirement's directory in {directories:?}"
        );
        assert!(
            directories.contains(&"reference-work/golden/XMCD/Gd_L1"),
            "expected current XSPH DirectoryFiles requirements in {directories:?}"
        );
        assert!(
            !directories
                .iter()
                .any(|directory| directory.contains("reference-work/tmp")),
            "AnyDirectoryWithPrefixFiles requirements should not contribute a fixture directory"
        );
    }

    #[test]
    fn missing_fixture_manifest_directories_reports_only_present_but_unmanifested_dirs()
    -> Result<()> {
        let root = compatibility_fixture_temp_dir("manifest-missing")?;
        let present_dir = root.join("reference-work/golden/HIGHZ");
        std::fs::create_dir_all(&present_dir)?;

        let missing = missing_fixture_manifest_directories(&root);
        assert!(
            missing.contains(&"reference-work/golden/HIGHZ".to_string()),
            "a present directory with no manifest.json should be reported: {missing:?}"
        );
        assert!(
            !missing.contains(&"reference-work/golden/HUBBARD/NiO".to_string()),
            "a directory absent from disk entirely should not be reported: {missing:?}"
        );

        crate::manifest::write_manifest(
            &present_dir,
            None,
            &crate::manifest::CompilerInfo::unknown(),
        )?;
        let missing_after_manifest = missing_fixture_manifest_directories(&root);
        assert!(
            !missing_after_manifest.contains(&"reference-work/golden/HIGHZ".to_string()),
            "a directory with a manifest.json should no longer be reported: {missing_after_manifest:?}"
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn stale_fixture_manifest_directories_detects_rev_mismatch() -> Result<()> {
        let root = compatibility_fixture_temp_dir("manifest-stale")?;
        let feff10_dir = root.join("feff10");
        std::fs::create_dir_all(&feff10_dir)?;
        run_git(&feff10_dir, &["init"])?;
        run_git(&feff10_dir, &["config", "user.email", "test@example.com"])?;
        run_git(&feff10_dir, &["config", "user.name", "test"])?;
        std::fs::write(feff10_dir.join("README"), b"placeholder")?;
        run_git(&feff10_dir, &["add", "README"])?;
        run_git(&feff10_dir, &["commit", "-m", "initial"])?;
        let current_rev = crate::manifest::feff10_git_rev(&feff10_dir)
            .expect("feff10 checkout should resolve a HEAD revision");

        let case_dir = root.join("reference-work/golden/HIGHZ");
        std::fs::create_dir_all(&case_dir)?;
        crate::manifest::write_manifest(
            &case_dir,
            Some("0000000000000000000000000000000000000000"),
            &crate::manifest::CompilerInfo::unknown(),
        )?;

        let stale = stale_fixture_manifest_directories(&root);
        assert!(
            stale
                .iter()
                .any(|entry| entry.starts_with("reference-work/golden/HIGHZ:")),
            "a manifest recording a different feff10 rev should be reported stale: {stale:?}"
        );

        crate::manifest::write_manifest(
            &case_dir,
            Some(&current_rev),
            &crate::manifest::CompilerInfo::unknown(),
        )?;
        let fresh = stale_fixture_manifest_directories(&root);
        assert!(
            !fresh
                .iter()
                .any(|entry| entry.starts_with("reference-work/golden/HIGHZ:")),
            "a manifest recording the current feff10 rev should not be reported stale: {fresh:?}"
        );

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    fn run_git(dir: &std::path::Path, args: &[&str]) -> Result<()> {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .with_context(|| format!("failed to invoke git {args:?} in {}", dir.display()))?;
        anyhow::ensure!(status.success(), "git {args:?} failed in {}", dir.display());
        Ok(())
    }
}
