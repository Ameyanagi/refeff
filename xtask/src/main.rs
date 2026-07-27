#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

use std::env;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use ndarray::Array1;
use refeff_io::{
    FeffDocument, FeffInput, FmsInput, WscrnDatData, XsphInput, fms_input_string, parse_danes_dat,
    parse_emesh_bin, parse_fms_bin, parse_gg_bin_bytes, parse_gtr_dat, parse_phase_bin,
    parse_xmu_dat, parse_xsect_dat, rdinp, read_wscrn_dat, wscrn_dat_string, xsph_input_string,
};
use serde::Serialize;

mod compatibility_matrix;
mod manifest;
mod parity;
mod port_status;
mod rixs_reference;
mod scope_manifest;
mod verify_evidence;

use compatibility_matrix::{
    CompatibilityOpenItem, compatibility_open_items, print_compatibility_matrix,
};
use port_status::print_port_status;
use verify_evidence::print_verify_evidence;

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Xtask {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ScopeAudit {
        #[arg(long)]
        detail: bool,
        #[arg(long, value_name = "PATH")]
        json_out: Option<PathBuf>,
    },
    PortStatus {
        #[arg(long)]
        cli_src: Option<PathBuf>,
        #[arg(long)]
        fail_on_unported: bool,
        #[arg(long)]
        fail_on_ignored_parity: bool,
        #[arg(long)]
        fail_on_guarded_branches: bool,
        #[arg(long)]
        detail: bool,
        #[arg(long, value_name = "PATH")]
        json_out: Option<PathBuf>,
    },
    CompatibilityMatrix {
        #[arg(long)]
        detail: bool,
        #[arg(long)]
        fail_on_open: bool,
        #[arg(long)]
        fail_on_missing_fixtures: bool,
        #[arg(long)]
        fail_on_stale_fixtures: bool,
        #[arg(long)]
        open_only: bool,
        #[arg(long = "module", value_name = "NAME")]
        modules: Vec<String>,
        #[arg(long = "row", value_name = "ID")]
        rows: Vec<String>,
        #[arg(long, value_name = "PATH")]
        json_out: Option<PathBuf>,
    },
    ReleaseReadiness {
        #[arg(long)]
        detail: bool,
        #[arg(long)]
        open_only: bool,
        #[arg(long = "module", value_name = "NAME")]
        modules: Vec<String>,
        #[arg(long = "row", value_name = "ID")]
        rows: Vec<String>,
        #[arg(long, value_name = "PATH")]
        port_json_out: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        compatibility_json_out: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        json_out: Option<PathBuf>,
    },
    ReferenceTests {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
    },
    VerifyEvidence {
        #[arg(long)]
        workspace_root: Option<PathBuf>,
    },
    GenerateGolden {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
        #[arg(long, default_value = "reference-work/golden")]
        out_dir: PathBuf,
        #[arg(long)]
        example: Vec<String>,
        #[arg(long)]
        no_build: bool,
        #[arg(long)]
        force: bool,
        #[arg(long, value_enum, default_value_t = ReferenceProgram::Feff)]
        program: ReferenceProgram,
    },
    /// Hash existing compatibility fixtures and record their pinned FEFF
    /// checkout provenance without regenerating the numerical artifacts.
    StampGoldenManifests {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },
    BenchE2e {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
        #[arg(long)]
        example: Vec<String>,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long)]
        reference: bool,
    },
    /// Run the Rust `refeff` pipeline against a golden fixture's `feff.inp`,
    /// gate its canonical output, and report every file diff (F1).
    Parity {
        /// Golden case to run, e.g. `XANES/BN`
        /// (`reference-work/golden/XANES/BN`).
        #[arg(long)]
        example: String,
        #[arg(long, value_name = "PATH")]
        json_out: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReferenceProgram {
    Feff,
    Rdinp,
}

impl ReferenceProgram {
    fn binary_candidates(self, ref_dir: &Path) -> [PathBuf; 2] {
        match self {
            Self::Feff => [ref_dir.join("bin/Seq/feff"), ref_dir.join("bin/feff")],
            Self::Rdinp => [ref_dir.join("bin/Seq/rdinp"), ref_dir.join("bin/rdinp")],
        }
    }

    fn log_prefix(self) -> &'static str {
        match self {
            Self::Feff => "feff",
            Self::Rdinp => "rdinp",
        }
    }
}

fn main() {
    if let Err(error) = run() {
        let _ = io::stdout().flush();
        println!("Error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let xtask = Xtask::parse();
    match xtask.command {
        Command::ScopeAudit { detail, json_out } => {
            scope_manifest::print_scope_audit(detail, json_out.as_deref())?;
        }
        Command::PortStatus {
            cli_src,
            fail_on_unported,
            fail_on_ignored_parity,
            fail_on_guarded_branches,
            detail,
            json_out,
        } => print_port_status(
            cli_src,
            fail_on_unported,
            fail_on_ignored_parity,
            fail_on_guarded_branches,
            detail,
            json_out.as_deref(),
        )?,
        Command::CompatibilityMatrix {
            detail,
            fail_on_open,
            fail_on_missing_fixtures,
            fail_on_stale_fixtures,
            open_only,
            modules,
            rows,
            json_out,
        } => print_compatibility_matrix(
            detail,
            fail_on_open,
            fail_on_missing_fixtures,
            fail_on_stale_fixtures,
            open_only,
            &modules,
            &rows,
            json_out.as_deref(),
        )?,
        Command::ReleaseReadiness {
            detail,
            open_only,
            modules,
            rows,
            port_json_out,
            compatibility_json_out,
            json_out,
        } => print_release_readiness(
            detail,
            open_only,
            &modules,
            &rows,
            port_json_out.as_deref(),
            compatibility_json_out.as_deref(),
            json_out.as_deref(),
        )?,
        Command::ReferenceTests { ref_dir } => run_reference_tests(ref_dir)?,
        Command::VerifyEvidence { workspace_root } => print_verify_evidence(workspace_root)?,
        Command::GenerateGolden {
            ref_dir,
            out_dir,
            example,
            no_build,
            force,
            program,
        } => generate_golden(ref_dir, &out_dir, &example, !no_build, force, program)?,
        Command::StampGoldenManifests { ref_dir, force } => {
            stamp_golden_manifests(ref_dir, force)?;
        }
        Command::BenchE2e {
            ref_dir,
            example,
            iterations,
            reference,
        } => bench_e2e(ref_dir, &example, iterations, reference)?,
        Command::Parity { example, json_out } => {
            parity::run_parity(&example, json_out.as_deref())?;
        }
    }
    Ok(())
}

fn print_release_readiness(
    detail: bool,
    open_only: bool,
    modules: &[String],
    rows: &[String],
    port_json_out: Option<&Path>,
    compatibility_json_out: Option<&Path>,
    json_out: Option<&Path>,
) -> Result<()> {
    println!("release readiness: pinned production-scope gate");
    let scope_status = scope_manifest::print_scope_audit(detail, None)
        .context("strict FEFF10 production-scope gate failed");

    println!("release readiness: strict module-support gate");
    let port_status = print_port_status(None, true, true, true, detail, port_json_out)
        .context("strict port-status release gate failed");

    println!("release readiness: strict branch-level compatibility gate");
    let compatibility = print_compatibility_matrix(
        detail,
        true,
        true,
        true,
        open_only,
        modules,
        rows,
        compatibility_json_out,
    )
    .context("strict compatibility-matrix release gate failed");

    if let Some(json_out) = json_out {
        write_release_readiness_json_report(
            json_out,
            &ReleaseReadinessJsonReport {
                detail,
                open_only,
                modules,
                rows,
                port_json_out,
                compatibility_json_out,
                scope_status: &scope_status,
                port_status: &port_status,
                compatibility: &compatibility,
            },
        )?;
        println!("wrote release readiness json: {}", json_out.display());
    }

    let mut failures = Vec::new();
    for result in [scope_status, port_status, compatibility] {
        if let Err(error) = result {
            failures.push(format!("{error:#}"));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("release readiness failed:\n- {}", failures.join("\n- "))
    }
}

struct ReleaseReadinessJsonReport<'a> {
    detail: bool,
    open_only: bool,
    modules: &'a [String],
    rows: &'a [String],
    port_json_out: Option<&'a Path>,
    compatibility_json_out: Option<&'a Path>,
    scope_status: &'a Result<()>,
    port_status: &'a Result<()>,
    compatibility: &'a Result<()>,
}

fn write_release_readiness_json_report(
    path: &Path,
    report: &ReleaseReadinessJsonReport<'_>,
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, release_readiness_json_report(report)?)?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct ReleaseReadinessFiltersJson<'a> {
    modules: &'a [String],
    rows: &'a [String],
    open_only: bool,
    detail: bool,
}

#[derive(Debug, Serialize)]
struct GateResultJson {
    passed: bool,
    artifact: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompatibilityOpenItemJson<'a> {
    id: &'static str,
    module: &'static str,
    workflow: &'static str,
    status: &'static str,
    requirement: &'static str,
    next: Option<&'static str>,
    verify: Option<&'static str>,
    fixture_groups: usize,
    missing_fixtures: &'a [String],
}

#[derive(Debug, Serialize)]
struct ReleaseReadinessReportJson<'a> {
    passed: bool,
    filters: ReleaseReadinessFiltersJson<'a>,
    production_scope: GateResultJson,
    port_status: GateResultJson,
    compatibility_matrix: GateResultJson,
    open_compatibility_items: Vec<CompatibilityOpenItemJson<'a>>,
}

fn release_readiness_json_report(report: &ReleaseReadinessJsonReport<'_>) -> Result<String> {
    let scope_passed = report.scope_status.is_ok();
    let port_passed = report.port_status.is_ok();
    let compatibility_passed = report.compatibility.is_ok();
    let open_items = selected_compatibility_open_items(report.modules, report.rows);
    let json = ReleaseReadinessReportJson {
        passed: scope_passed && port_passed && compatibility_passed,
        filters: ReleaseReadinessFiltersJson {
            modules: report.modules,
            rows: report.rows,
            open_only: report.open_only,
            detail: report.detail,
        },
        production_scope: GateResultJson {
            passed: scope_passed,
            artifact: None,
            error: result_error_string(report.scope_status),
        },
        port_status: GateResultJson {
            passed: port_passed,
            artifact: report.port_json_out.map(display_path),
            error: result_error_string(report.port_status),
        },
        compatibility_matrix: GateResultJson {
            passed: compatibility_passed,
            artifact: report.compatibility_json_out.map(display_path),
            error: result_error_string(report.compatibility),
        },
        open_compatibility_items: open_items
            .iter()
            .map(|item| CompatibilityOpenItemJson {
                id: item.id,
                module: item.module,
                workflow: item.workflow,
                status: item.status,
                requirement: item.requirement,
                next: item.next_action,
                verify: item.verification_gate,
                fixture_groups: item.fixture_groups,
                missing_fixtures: &item.missing_fixtures,
            })
            .collect(),
    };
    serde_json::to_string_pretty(&json).context("failed to serialize release readiness json")
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn selected_compatibility_open_items(
    module_filters: &[String],
    row_filters: &[String],
) -> Vec<CompatibilityOpenItem> {
    compatibility_open_items()
        .into_iter()
        .filter(|item| {
            (module_filters.is_empty()
                || module_filters
                    .iter()
                    .any(|module| item.module.eq_ignore_ascii_case(module)))
                && (row_filters.is_empty()
                    || row_filters
                        .iter()
                        .any(|row| item.id.eq_ignore_ascii_case(row)))
        })
        .collect()
}

fn result_error_string(result: &Result<()>) -> Option<String> {
    result.as_ref().err().map(|error| format!("{error:#}"))
}

#[derive(Debug, Default)]
struct RustBenchSummary {
    runs: usize,
    successful: usize,
    failed: usize,
    output_files: usize,
    output_bytes: usize,
    duration: Duration,
    errors: Vec<String>,
}

#[derive(Debug, Default)]
struct ReferenceBenchSummary {
    runs: usize,
    successful: usize,
    failed: usize,
    duration: Duration,
    errors: Vec<String>,
}

#[derive(Debug)]
struct RustRdinpRun {
    output_files: usize,
    output_bytes: usize,
}

fn bench_e2e(
    ref_dir: Option<PathBuf>,
    examples: &[String],
    iterations: usize,
    compare_reference: bool,
) -> Result<()> {
    anyhow::ensure!(iterations > 0, "iterations must be positive");
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("feff10"));
    let examples_dir = ref_dir.join("examples");

    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();
    inputs.retain(|input| selected_example(input, &examples_dir, examples));
    anyhow::ensure!(
        !inputs.is_empty(),
        "no FEFF examples matched the benchmark selection"
    );

    let work_dir = temporary_work_dir("refeff-e2e-bench")?;
    let rust_dir = work_dir.join("rust");
    let reference_dir = work_dir.join("reference");

    let rust_summary = bench_rust_rdinp(&inputs, iterations, &rust_dir)?;
    println!(
        "rust rdinp: inputs={} iterations={} runs={} ok={} failed={} time={:.6}s avg/run={:.6}s outputs={} bytes={}",
        inputs.len(),
        iterations,
        rust_summary.runs,
        rust_summary.successful,
        rust_summary.failed,
        duration_seconds(rust_summary.duration),
        average_seconds(rust_summary.duration, rust_summary.runs),
        rust_summary.output_files,
        rust_summary.output_bytes
    );
    print_sample_errors("rust rdinp", &rust_summary.errors);

    if compare_reference {
        let ref_dir = ref_dir.canonicalize()?;
        let driver = reference_driver(&ref_dir, ReferenceProgram::Rdinp)?;
        let reference_summary =
            bench_reference_rdinp(&driver, &examples_dir, &inputs, iterations, &reference_dir)?;
        println!(
            "feff10 rdinp: inputs={} iterations={} runs={} ok={} failed={} time={:.6}s avg/run={:.6}s",
            inputs.len(),
            iterations,
            reference_summary.runs,
            reference_summary.successful,
            reference_summary.failed,
            duration_seconds(reference_summary.duration),
            average_seconds(reference_summary.duration, reference_summary.runs)
        );
        print_sample_errors("feff10 rdinp", &reference_summary.errors);
    }

    if let Err(error) = std::fs::remove_dir_all(&work_dir) {
        eprintln!(
            "warning: failed to remove benchmark work directory {}: {error}",
            work_dir.display()
        );
    }
    Ok(())
}

fn bench_rust_rdinp(
    inputs: &[PathBuf],
    iterations: usize,
    output_root: &Path,
) -> Result<RustBenchSummary> {
    let mut summary = RustBenchSummary::default();
    let start = Instant::now();
    for iteration in 0..iterations {
        for (index, input) in inputs.iter().enumerate() {
            summary.runs += 1;
            let output_dir = output_root.join(format!("iter-{iteration}/input-{index:04}"));
            match run_rust_rdinp_to_dir(input, &output_dir) {
                Ok(run) => {
                    summary.successful += 1;
                    summary.output_files += run.output_files;
                    summary.output_bytes += run.output_bytes;
                }
                Err(error) => {
                    summary.failed += 1;
                    if summary.errors.len() < 8 {
                        summary.errors.push(format!("{}: {error}", input.display()));
                    }
                }
            }
        }
    }
    summary.duration = start.elapsed();
    Ok(summary)
}

fn run_rust_rdinp_to_dir(input: &Path, output_dir: &Path) -> Result<RustRdinpRun> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let sentinel = rdinp::rdinp_error_sentinel_string();
    let sentinel_bytes = write_output(output_dir, ".feff.error", &sentinel)?;

    let parsed = FeffInput::parse_file(input)?;
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(error) => {
            let content = rdinp::rdinp_error_log_string(&parsed, &error)?;
            let log_bytes = write_output(output_dir, "log.dat", &content)?;
            return Ok(RustRdinpRun {
                output_files: 2,
                output_bytes: sentinel_bytes + log_bytes,
            });
        }
    };

    let mut output_files = 0_usize;
    let mut output_bytes = 0_usize;
    for (name, content) in rdinp::text_outputs(&document)? {
        output_files += 1;
        output_bytes += write_output(output_dir, name.as_ref(), &content)?;
    }
    if let Ok(content) = rdinp::rdinp_log_dat_string(&document) {
        output_files += 1;
        output_bytes += write_output(output_dir, "log.dat", &content)?;
    }
    match std::fs::remove_file(output_dir.join(".feff.error")) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to remove {}",
                    output_dir.join(".feff.error").display()
                )
            });
        }
    }
    Ok(RustRdinpRun {
        output_files,
        output_bytes,
    })
}

fn write_output(output_dir: &Path, name: &str, content: &str) -> Result<usize> {
    let output_path = output_dir.join(name);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&output_path, content)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(content.len())
}

fn bench_reference_rdinp(
    driver: &Path,
    examples_dir: &Path,
    inputs: &[PathBuf],
    iterations: usize,
    output_root: &Path,
) -> Result<ReferenceBenchSummary> {
    let mut summary = ReferenceBenchSummary::default();
    for iteration in 0..iterations {
        for (index, input) in inputs.iter().enumerate() {
            summary.runs += 1;
            let Some(parent) = input.parent() else {
                summary.failed += 1;
                if summary.errors.len() < 8 {
                    summary
                        .errors
                        .push(format!("{} has no parent directory", input.display()));
                }
                continue;
            };
            let rel = parent.strip_prefix(examples_dir).with_context(|| {
                format!(
                    "{} is not under examples directory {}",
                    parent.display(),
                    examples_dir.display()
                )
            })?;
            let output_dir = output_root.join(format!("iter-{iteration}/input-{index:04}"));
            std::fs::create_dir_all(&output_dir)
                .with_context(|| format!("failed to create {}", output_dir.display()))?;
            copy_dir(parent, &output_dir)?;

            let start = Instant::now();
            let output = std::process::Command::new(driver)
                .current_dir(&output_dir)
                .output()
                .with_context(|| format!("failed to run reference rdinp for {}", rel.display()))?;
            summary.duration += start.elapsed();
            if output.status.success() {
                summary.successful += 1;
            } else {
                summary.failed += 1;
                if summary.errors.len() < 8 {
                    summary.errors.push(format!(
                        "{} failed with status {}",
                        rel.display(),
                        output.status
                    ));
                }
            }
        }
    }
    Ok(summary)
}

fn selected_example(input: &Path, examples_dir: &Path, examples: &[String]) -> bool {
    if examples.is_empty() {
        return true;
    }
    input
        .parent()
        .and_then(|parent| parent.strip_prefix(examples_dir).ok())
        .map(|rel| selected_relative_example(rel, examples))
        .unwrap_or(false)
}

fn selected_relative_example(relative: &Path, examples: &[String]) -> bool {
    let relative_text = relative.to_string_lossy();
    examples.iter().any(|pattern| {
        if pattern.eq_ignore_ascii_case("RIXS") {
            relative == Path::new("RIXS")
        } else {
            relative_text.contains(pattern)
        }
    })
}

fn temporary_work_dir(prefix: &str) -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_millis();
    let path = env::temp_dir().join(format!("{prefix}-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    Ok(path)
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn average_seconds(duration: Duration, runs: usize) -> f64 {
    if runs == 0 {
        0.0
    } else {
        duration.as_secs_f64() / runs as f64
    }
}

fn print_sample_errors(label: &str, errors: &[String]) {
    for error in errors {
        eprintln!("{label} sample error: {error}");
    }
}

fn generate_golden(
    ref_dir: Option<PathBuf>,
    out_dir: &Path,
    examples: &[String],
    build_reference: bool,
    force: bool,
    program: ReferenceProgram,
) -> Result<()> {
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("feff10"));
    let ref_dir = ref_dir.canonicalize()?;
    let examples_dir = ref_dir.join("examples");

    let compiler = if build_reference {
        build_reference_feff(&ref_dir)?
    } else {
        manifest::CompilerInfo::unknown()
    };
    let feff10_rev = manifest::feff10_git_rev(&ref_dir);
    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();

    for input in inputs {
        let parent = input
            .parent()
            .with_context(|| format!("{} has no parent directory", input.display()))?;
        let rel = parent.strip_prefix(&examples_dir)?;
        if !examples.is_empty() && !selected_relative_example(rel, examples) {
            continue;
        }

        let dest = out_dir.join(rel);
        if dest.exists() {
            if force {
                std::fs::remove_dir_all(&dest)?;
            } else {
                anyhow::bail!(
                    "{} already exists; pass --force to replace it",
                    dest.display()
                );
            }
        }
        std::fs::create_dir_all(&dest)?;
        copy_dir(parent, &dest)?;
        clean_hubbard_nio_generated_caches(rel, &dest)?;

        // HIGHZ is an upstream parameterized harness: its feff.inp contains
        // the literal `XXX`, and `runall` expands atomic numbers 1..=138.
        // Preserve the pinned upstream harness and checked reference report
        // as one provenance-tracked fixture instead of attempting to run the
        // invalid template once.
        if rel == Path::new("HIGHZ") {
            remove_empty_feff_error(&dest)?;
            manifest::write_manifest(&dest, feff10_rev.as_deref(), &compiler)
                .with_context(|| format!("failed to write manifest.json in {}", dest.display()))?;
            println!("generated {} (parameterized HIGHZ harness)", dest.display());
            continue;
        }

        if program == ReferenceProgram::Feff {
            capture_standalone_rdinp_if_available(&ref_dir, &dest)?;
        }
        let output = run_reference_program(&ref_dir, program, &dest)?;
        std::fs::write(
            dest.join(format!("{}.stdout", program.log_prefix())),
            &output.stdout,
        )?;
        std::fs::write(
            dest.join(format!("{}.stderr", program.log_prefix())),
            &output.stderr,
        )?;
        if !output.success && !is_expected_danes_errorfile_cleanup_exit(rel, &output) {
            anyhow::bail!(
                "{} reference failed for {} with status {}",
                program.log_prefix(),
                rel.display(),
                output.status
            );
        }
        if program == ReferenceProgram::Feff {
            repair_and_validate_hubbard_nio_generation(
                &ref_dir,
                feff10_rev.as_deref(),
                rel,
                &dest,
                &output,
            )?;
            generate_compatibility_reference_subcases(&ref_dir, rel, &dest)?;
            quarantine_invalid_danes_gecl4_outputs(rel, &dest)?;
            quarantine_invalid_debye_fecn6_outputs(rel, &dest)?;
            if rel == Path::new("RIXS") {
                let native_commit = feff10_rev
                    .as_deref()
                    .context("exact RIXS generation requires a pinned FEFF10 Git commit")?;
                generate_exact_native_rixs_reference(&ref_dir, &dest, native_commit)?;
            }
        }
        manifest::write_manifest(&dest, feff10_rev.as_deref(), &compiler)
            .with_context(|| format!("failed to write manifest.json in {}", dest.display()))?;
        println!("generated {}", dest.display());
    }

    Ok(())
}

const HUBBARD_NIO_EXAMPLE: &str = "HUBBARD/NiO";
const MKGTR_UNEXPECTED_GG_EOF: &str = "Error: Unexpected end of record while reading from gg.bin.";
const HUBBARD_NIO_PROVENANCE_FILE: &str = ".hubbard-mkgtr-provenance.json";
const HUBBARD_NIO_EXPECTED_ENERGY_COUNT: usize = 83;
const HUBBARD_NIO_MAX_GG_BYTES: u64 = 64 * 1024 * 1024;
const HUBBARD_NIO_GG_DESCRIPTOR_PREFIX: &[u8] = b"#DF# This section written in ";
const HUBBARD_NIO_GG_DESCRIPTOR: &[u8] = b"#DF# This section written in sanitized FEFF oracle.\n";
const HUBBARD_NIO_GG_NORMALIZATION: &str = "for every generated Hubbard NiO gg.bin section, replace only the '#DF# This section written in <four uninitialized descriptor bytes>.' record and its immediate standalone '.' continuation with '#DF# This section written in sanitized FEFF oracle.\\n'; copy every other byte unchanged";
const HUBBARD_NIO_GENERATED_CACHES: &[&str] = &[
    "phase.bin",
    "gg.bin",
    "fms.bin",
    "gtr.dat",
    "gtr00.bin",
    "gtr01.bin",
    "gtr02.bin",
    "gtr_m00.bin",
    "gtr_m01.bin",
    "gtr_m02.bin",
    "gtr_off.dat",
    "gtr_off00.bin",
    "gtr_off01.bin",
    "gtr_off02.bin",
    "xmu.dat",
    HUBBARD_NIO_PROVENANCE_FILE,
];
const HUBBARD_NIO_DOWNSTREAM_CACHES: &[&str] = &[
    "fms.bin",
    "gtr.dat",
    "gtr00.bin",
    "gtr01.bin",
    "gtr02.bin",
    "gtr_m00.bin",
    "gtr_m01.bin",
    "gtr_m02.bin",
    "gtr_off.dat",
    "gtr_off00.bin",
    "gtr_off01.bin",
    "gtr_off02.bin",
    "xmu.dat",
    HUBBARD_NIO_PROVENANCE_FILE,
];

#[derive(Debug, Serialize)]
struct HubbardMkgtrProvenance {
    schema_version: u8,
    generator: &'static str,
    native_commit: String,
    normalization_operation: &'static str,
    phase_sha256: String,
    original_gg_sha256: String,
    sanitized_gg_sha256: String,
    descriptor_records: usize,
    continuation_lines_removed: usize,
    mkgtr_executable: &'static str,
    ff2x_executable: &'static str,
    fms_sha256: String,
    gtr_sha256: String,
    xmu_sha256: String,
}

#[derive(Debug)]
struct HubbardGgSanitization {
    bytes: Vec<u8>,
    descriptor_records: usize,
    continuation_lines_removed: usize,
}

fn clean_hubbard_nio_generated_caches(example: &Path, case_dir: &Path) -> Result<usize> {
    if example != Path::new(HUBBARD_NIO_EXAMPLE) {
        return Ok(0);
    }
    remove_hubbard_generated_files(case_dir, HUBBARD_NIO_GENERATED_CACHES)
}

fn remove_hubbard_generated_files(case_dir: &Path, names: &[&str]) -> Result<usize> {
    let mut removed = 0;
    for name in names {
        let path = case_dir.join(name);
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
            }
        };
        anyhow::ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "refusing to clean non-regular Hubbard cache {}",
            path.display()
        );
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to clean stale Hubbard cache {}", path.display()))?;
        removed += 1;
    }
    Ok(removed)
}

fn repair_and_validate_hubbard_nio_generation(
    ref_dir: &Path,
    native_commit: Option<&str>,
    example: &Path,
    case_dir: &Path,
    output: &ReferenceOutput,
) -> Result<bool> {
    if example != Path::new(HUBBARD_NIO_EXAMPLE) {
        return Ok(false);
    }
    let result = (|| -> Result<bool> {
        let repair = repair_hubbard_nio_mkgtr(
            ref_dir,
            native_commit.context(
                "Hubbard NiO MKGTR descriptor repair requires a pinned FEFF10 Git commit",
            )?,
            case_dir,
            output,
        )?;
        validate_hubbard_nio_generation(example, case_dir, output, repair.is_some())?;
        if let Some(provenance) = repair {
            write_hubbard_mkgtr_provenance(case_dir, &provenance)?;
        }
        Ok(true)
    })();
    fail_closed_hubbard_generation(case_dir, result)
}

fn fail_closed_hubbard_generation(case_dir: &Path, result: Result<bool>) -> Result<bool> {
    match result {
        Ok(validated) => Ok(validated),
        Err(validation_error) => {
            let validation_message = format!("{validation_error:#}");
            std::fs::remove_dir_all(case_dir).with_context(|| {
                format!(
                    "Hubbard reference generation failed validation ({validation_message}); \
                     also failed to remove incomplete case {}",
                    case_dir.display()
                )
            })?;
            Err(validation_error.context(format!(
                "removed incomplete Hubbard reference case {}",
                case_dir.display()
            )))
        }
    }
}

fn repair_hubbard_nio_mkgtr(
    ref_dir: &Path,
    native_commit: &str,
    case_dir: &Path,
    output: &ReferenceOutput,
) -> Result<Option<HubbardMkgtrProvenance>> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.contains(MKGTR_UNEXPECTED_GG_EOF) && !stderr.contains(MKGTR_UNEXPECTED_GG_EOF) {
        return Ok(None);
    }
    anyhow::ensure!(
        native_commit.len() == 40
            && native_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Hubbard NiO repair requires a lowercase 40-digit native commit"
    );

    let phase_bytes = read_required_generated_file(case_dir, "phase.bin")?;
    let phase_text =
        std::str::from_utf8(&phase_bytes).context("generated phase.bin is not UTF-8 text")?;
    let phase = parse_phase_bin(phase_text).context("generated phase.bin is invalid")?;
    anyhow::ensure!(
        phase.energy_count == HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
        "generated Hubbard NiO phase.bin has {} energy point(s), expected {}",
        phase.energy_count,
        HUBBARD_NIO_EXPECTED_ENERGY_COUNT
    );

    let gg_path = case_dir.join("gg.bin");
    let gg_bytes = read_bounded_hubbard_gg(&gg_path)?;
    let original_gg = parse_gg_bin_bytes(&gg_bytes).context("generated gg.bin is invalid")?;
    anyhow::ensure!(
        original_gg.section_count() == HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
        "generated Hubbard NiO gg.bin has {} section(s), expected {}",
        original_gg.section_count(),
        HUBBARD_NIO_EXPECTED_ENERGY_COUNT
    );
    let sanitized = sanitize_hubbard_nio_gg_descriptors(&gg_bytes)?;
    let sanitized_gg =
        parse_gg_bin_bytes(&sanitized.bytes).context("sanitized Hubbard NiO gg.bin is invalid")?;
    anyhow::ensure!(
        gg_numeric_payloads_equal(&original_gg, &sanitized_gg),
        "Hubbard NiO GG descriptor repair changed a section number or numeric payload"
    );
    let original_gg_sha256 = manifest::sha256_hex(&gg_bytes);
    let sanitized_gg_sha256 = manifest::sha256_hex(&sanitized.bytes);
    anyhow::ensure!(
        original_gg_sha256 != sanitized_gg_sha256,
        "Hubbard NiO MKGTR reported the descriptor defect, but sanitization changed no bytes"
    );
    replace_regular_file_atomic(&gg_path, &sanitized.bytes)?;

    remove_hubbard_generated_files(case_dir, HUBBARD_NIO_DOWNSTREAM_CACHES)?;
    let mkgtr = run_hubbard_repair_module(ref_dir, case_dir, "mkgtr")?;
    let repaired_stdout = String::from_utf8_lossy(&mkgtr.stdout);
    let repaired_stderr = String::from_utf8_lossy(&mkgtr.stderr);
    anyhow::ensure!(
        !repaired_stdout.contains(MKGTR_UNEXPECTED_GG_EOF)
            && !repaired_stderr.contains(MKGTR_UNEXPECTED_GG_EOF),
        "Hubbard NiO MKGTR still rejected gg.bin after descriptor-only repair"
    );
    append_hubbard_repair_log(case_dir, "mkgtr descriptor repair", &mkgtr)?;
    let ff2x = run_hubbard_repair_module(ref_dir, case_dir, "ff2x")?;
    append_hubbard_repair_log(case_dir, "ff2x after MKGTR descriptor repair", &ff2x)?;

    Ok(Some(HubbardMkgtrProvenance {
        schema_version: 1,
        generator: "xtask generate-golden Hubbard NiO descriptor repair",
        native_commit: native_commit.to_string(),
        normalization_operation: HUBBARD_NIO_GG_NORMALIZATION,
        phase_sha256: manifest::sha256_hex(&phase_bytes),
        original_gg_sha256,
        sanitized_gg_sha256,
        descriptor_records: sanitized.descriptor_records,
        continuation_lines_removed: sanitized.continuation_lines_removed,
        mkgtr_executable: "bin/Seq/mkgtr",
        ff2x_executable: "bin/Seq/ff2x",
        fms_sha256: manifest::sha256_hex(&read_required_generated_file(case_dir, "fms.bin")?),
        gtr_sha256: manifest::sha256_hex(&read_required_generated_file(case_dir, "gtr.dat")?),
        xmu_sha256: manifest::sha256_hex(&read_required_generated_file(case_dir, "xmu.dat")?),
    }))
}

fn validate_hubbard_nio_generation(
    example: &Path,
    case_dir: &Path,
    output: &ReferenceOutput,
    repaired_internal_mkgtr_failure: bool,
) -> Result<bool> {
    if example != Path::new(HUBBARD_NIO_EXAMPLE) {
        return Ok(false);
    }

    let fms_input_path = case_dir.join("fms.inp");
    let fms_input_text = std::fs::read_to_string(&fms_input_path)
        .with_context(|| format!("failed to read {}", fms_input_path.display()))?;
    let fms_input = FmsInput::parse_str(&fms_input_path, &fms_input_text)
        .with_context(|| format!("failed to parse {}", fms_input_path.display()))?;
    anyhow::ensure!(
        fms_input.do_fms != 0,
        "{HUBBARD_NIO_EXAMPLE} unexpectedly disabled its required FMS calculation"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::ensure!(
        repaired_internal_mkgtr_failure
            || (!stdout.contains(MKGTR_UNEXPECTED_GG_EOF)
                && !stderr.contains(MKGTR_UNEXPECTED_GG_EOF)),
        "{HUBBARD_NIO_EXAMPLE} MKGTR rejected the freshly generated gg.bin; \
         refusing stale archive/root FMS or spectrum caches"
    );

    let phase_bytes = read_required_generated_file(case_dir, "phase.bin")?;
    let phase_text =
        std::str::from_utf8(&phase_bytes).context("generated phase.bin is not UTF-8 text")?;
    let phase = parse_phase_bin(phase_text).context("generated phase.bin is invalid")?;

    let gg_bytes = read_required_generated_file(case_dir, "gg.bin")?;
    let gg = parse_gg_bin_bytes(&gg_bytes).context("generated gg.bin is invalid")?;

    let fms_bytes = read_required_generated_file(case_dir, "fms.bin")?;
    let fms_text =
        std::str::from_utf8(&fms_bytes).context("generated fms.bin is not UTF-8 text")?;
    let fms = parse_fms_bin(fms_text).context("generated fms.bin is invalid")?;

    let gtr_bytes = read_required_generated_file(case_dir, "gtr.dat")?;
    let gtr_text =
        std::str::from_utf8(&gtr_bytes).context("generated gtr.dat is not UTF-8 text")?;
    let gtr = parse_gtr_dat(gtr_text).context("generated gtr.dat is invalid")?;

    let xmu_bytes = read_required_generated_file(case_dir, "xmu.dat")?;
    let xmu_text =
        std::str::from_utf8(&xmu_bytes).context("generated xmu.dat is not UTF-8 text")?;
    let xmu = parse_xmu_dat(xmu_text).context("generated xmu.dat is invalid")?;

    validate_hubbard_fms_cache_contract(
        phase.energy_count,
        gg.section_count(),
        fms.energy_count,
        gtr.row_count(),
        xmu.chi.iter().any(|value| *value != 0.0),
    )?;
    Ok(true)
}

fn read_bounded_hubbard_gg(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect generated Hubbard GG {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "generated Hubbard GG {} is not a regular file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= HUBBARD_NIO_MAX_GG_BYTES,
        "generated Hubbard GG {} is {} bytes, exceeding the {}-byte limit",
        path.display(),
        metadata.len(),
        HUBBARD_NIO_MAX_GG_BYTES
    );
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read generated Hubbard GG {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() as u64 == metadata.len(),
        "generated Hubbard GG {} changed length while it was read",
        path.display()
    );
    Ok(bytes)
}

fn sanitize_hubbard_nio_gg_descriptors(input: &[u8]) -> Result<HubbardGgSanitization> {
    anyhow::ensure!(
        input.len() as u64 <= HUBBARD_NIO_MAX_GG_BYTES,
        "generated Hubbard GG is {} bytes, exceeding the {}-byte limit",
        input.len(),
        HUBBARD_NIO_MAX_GG_BYTES
    );
    let mut output = Vec::with_capacity(input.len());
    let mut cursor = 0usize;
    let mut descriptor_records = 0usize;
    let mut continuation_lines_removed = 0usize;
    while cursor < input.len() {
        let line_end = next_hubbard_gg_line_end(input, cursor);
        let line = &input[cursor..line_end];
        if !line.starts_with(b"#DF#") {
            output.extend_from_slice(line);
            cursor = line_end;
            continue;
        }
        anyhow::ensure!(
            line.starts_with(HUBBARD_NIO_GG_DESCRIPTOR_PREFIX)
                && line.len() == HUBBARD_NIO_GG_DESCRIPTOR_PREFIX.len() + 4,
            "generated Hubbard GG contains an unexpected #DF# descriptor"
        );
        descriptor_records += 1;
        anyhow::ensure!(
            descriptor_records <= HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
            "generated Hubbard GG contains too many #DF# records"
        );
        output.extend_from_slice(HUBBARD_NIO_GG_DESCRIPTOR);
        cursor = line_end;

        anyhow::ensure!(
            cursor < input.len(),
            "generated Hubbard GG ends immediately after a #DF# descriptor"
        );
        let continuation_end = next_hubbard_gg_line_end(input, cursor);
        let continuation = &input[cursor..continuation_end];
        anyhow::ensure!(
            continuation == b".\n" || continuation == b".",
            "generated Hubbard GG descriptor is not followed by the exact standalone '.' record"
        );
        let header_end = next_hubbard_gg_line_end(input, continuation_end);
        let header = input
            .get(continuation_end..header_end)
            .context("generated Hubbard GG descriptor continuation has no following header")?;
        anyhow::ensure!(
            header.starts_with(b"#H#"),
            "generated Hubbard GG has data outside the bounded #DF# continuation position"
        );
        continuation_lines_removed += 1;
        cursor = continuation_end;
    }
    anyhow::ensure!(
        descriptor_records == HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
        "generated Hubbard GG contains {descriptor_records} #DF# record(s), expected \
         {HUBBARD_NIO_EXPECTED_ENERGY_COUNT}"
    );
    Ok(HubbardGgSanitization {
        bytes: output,
        descriptor_records,
        continuation_lines_removed,
    })
}

fn next_hubbard_gg_line_end(bytes: &[u8], start: usize) -> usize {
    bytes[start..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(bytes.len(), |offset| start + offset + 1)
}

fn gg_numeric_payloads_equal(left: &refeff_io::GgDatData, right: &refeff_io::GgDatData) -> bool {
    left.sections.len() == right.sections.len()
        && left
            .sections
            .iter()
            .zip(&right.sections)
            .all(|(left, right)| {
                left.section_number == right.section_number && left.values == right.values
            })
}

fn read_required_generated_file(case_dir: &Path, name: &str) -> Result<Vec<u8>> {
    let path = case_dir.join(name);
    let metadata = std::fs::symlink_metadata(&path).with_context(|| {
        format!(
            "required generated Hubbard handoff {} is missing",
            path.display()
        )
    })?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "required generated Hubbard handoff {} is not a regular file",
        path.display()
    );
    std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn replace_regular_file_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "refusing to replace non-regular generated file {}",
        path.display()
    );
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let temporary = parent.join(format!(
        ".{}.replace-{}-{stamp}",
        name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temporary.display()))?;
        std::fs::rename(&temporary, path).with_context(|| {
            format!(
                "failed to atomically publish {} as {}",
                temporary.display(),
                path.display()
            )
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;
    anyhow::ensure!(
        std::fs::read(path).with_context(|| format!("failed to verify {}", path.display()))?
            == bytes,
        "published generated file {} failed byte verification",
        path.display()
    );
    Ok(())
}

fn run_hubbard_repair_module(
    ref_dir: &Path,
    case_dir: &Path,
    module: &'static str,
) -> Result<std::process::Output> {
    let executable = ref_dir.join("bin/Seq").join(module);
    anyhow::ensure!(
        executable.is_file(),
        "missing Hubbard repair executable {}",
        executable.display()
    );
    let output = std::process::Command::new(&executable)
        .current_dir(case_dir)
        .output()
        .with_context(|| format!("failed to run {}", executable.display()))?;
    anyhow::ensure!(
        output.status.success(),
        "Hubbard NiO {module} repair failed with status {}",
        output.status
    );
    Ok(output)
}

fn append_hubbard_repair_log(
    case_dir: &Path,
    description: &str,
    output: &std::process::Output,
) -> Result<()> {
    for (suffix, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        let path = case_dir.join(format!("feff.{suffix}"));
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to append {}", path.display()))?;
        writeln!(file, "\n== {description} ==")?;
        file.write_all(bytes)?;
    }
    Ok(())
}

fn write_hubbard_mkgtr_provenance(
    case_dir: &Path,
    provenance: &HubbardMkgtrProvenance,
) -> Result<()> {
    validate_hubbard_mkgtr_provenance(case_dir, provenance)?;
    let mut bytes =
        serde_json::to_vec_pretty(provenance).context("failed to serialize Hubbard provenance")?;
    bytes.push(b'\n');
    let path = case_dir.join(HUBBARD_NIO_PROVENANCE_FILE);
    match std::fs::read(&path) {
        Ok(existing) if existing == bytes => return Ok(()),
        Ok(_) => anyhow::bail!(
            "refusing to overwrite different Hubbard provenance {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let temporary = case_dir.join(format!(
        ".hubbard-mkgtr-provenance.tmp-{}-{stamp}",
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temporary.display()))?;
        std::fs::rename(&temporary, &path).with_context(|| {
            format!(
                "failed to atomically publish {} as {}",
                temporary.display(),
                path.display()
            )
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;
    anyhow::ensure!(
        std::fs::read(&path).with_context(|| format!("failed to verify {}", path.display()))?
            == bytes,
        "published Hubbard provenance {} failed byte verification",
        path.display()
    );
    Ok(())
}

fn validate_hubbard_mkgtr_provenance(
    case_dir: &Path,
    provenance: &HubbardMkgtrProvenance,
) -> Result<()> {
    anyhow::ensure!(
        provenance.schema_version == 1
            && provenance.generator == "xtask generate-golden Hubbard NiO descriptor repair"
            && provenance.normalization_operation == HUBBARD_NIO_GG_NORMALIZATION
            && provenance.mkgtr_executable == "bin/Seq/mkgtr"
            && provenance.ff2x_executable == "bin/Seq/ff2x",
        "Hubbard NiO provenance has an unexpected schema or generator contract"
    );
    anyhow::ensure!(
        is_lowercase_hex_digest(&provenance.native_commit, 40)
            && is_lowercase_hex_digest(&provenance.phase_sha256, 64)
            && is_lowercase_hex_digest(&provenance.original_gg_sha256, 64)
            && is_lowercase_hex_digest(&provenance.sanitized_gg_sha256, 64)
            && is_lowercase_hex_digest(&provenance.fms_sha256, 64)
            && is_lowercase_hex_digest(&provenance.gtr_sha256, 64)
            && is_lowercase_hex_digest(&provenance.xmu_sha256, 64),
        "Hubbard NiO provenance contains an invalid commit or artifact digest"
    );
    anyhow::ensure!(
        provenance.descriptor_records == HUBBARD_NIO_EXPECTED_ENERGY_COUNT
            && provenance.continuation_lines_removed == HUBBARD_NIO_EXPECTED_ENERGY_COUNT
            && provenance.original_gg_sha256 != provenance.sanitized_gg_sha256,
        "Hubbard NiO provenance does not record the exact bounded descriptor repair"
    );
    for (name, expected) in [
        ("phase.bin", &provenance.phase_sha256),
        ("gg.bin", &provenance.sanitized_gg_sha256),
        ("fms.bin", &provenance.fms_sha256),
        ("gtr.dat", &provenance.gtr_sha256),
        ("xmu.dat", &provenance.xmu_sha256),
    ] {
        let actual = manifest::sha256_hex(&read_required_generated_file(case_dir, name)?);
        anyhow::ensure!(
            actual == *expected,
            "Hubbard NiO provenance digest for {name} does not match the published artifact"
        );
    }
    Ok(())
}

fn is_lowercase_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_hubbard_fms_cache_contract(
    phase_energy_count: usize,
    gg_section_count: usize,
    fms_energy_count: usize,
    gtr_row_count: usize,
    has_nonzero_chi: bool,
) -> Result<()> {
    anyhow::ensure!(
        phase_energy_count > 0,
        "generated Hubbard phase.bin has no energy points"
    );
    anyhow::ensure!(
        gg_section_count == phase_energy_count,
        "generated Hubbard gg.bin has {gg_section_count} section(s), but phase.bin has \
         {phase_energy_count} energy point(s)"
    );
    anyhow::ensure!(
        fms_energy_count == phase_energy_count,
        "generated Hubbard fms.bin has {fms_energy_count} energy point(s), but phase.bin has \
         {phase_energy_count}"
    );
    anyhow::ensure!(
        gtr_row_count == phase_energy_count,
        "generated Hubbard gtr.dat has {gtr_row_count} row(s), but phase.bin has \
         {phase_energy_count} energy point(s)"
    );
    anyhow::ensure!(
        has_nonzero_chi,
        "generated Hubbard xmu.dat has identically zero chi after active FMS"
    );
    Ok(())
}

fn is_expected_danes_errorfile_cleanup_exit(example: &Path, output: &ReferenceOutput) -> bool {
    if output.success || example != Path::new("DANES/GeCl_4") {
        return false;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout.contains("Done with module: XAS spectra")
        && stderr.contains("file = '.feff.error'")
        && stderr.contains("Fortran runtime error: File cannot be deleted")
}

fn remove_empty_feff_error(case_dir: &Path) -> Result<bool> {
    let path = case_dir.join(".feff.error");
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    };
    if metadata.len() != 0 {
        return Ok(false);
    }
    std::fs::remove_file(&path)
        .with_context(|| format!("failed to remove empty {}", path.display()))?;
    Ok(true)
}

fn capture_standalone_rdinp_if_available(ref_dir: &Path, case_dir: &Path) -> Result<bool> {
    let Ok(driver) = reference_driver(ref_dir, ReferenceProgram::Rdinp) else {
        return Ok(false);
    };
    run_reference_subprogram(
        &driver,
        case_dir,
        "rdinp",
        "standalone FEFF10 RDINP reference",
    )?;
    Ok(true)
}

#[derive(Debug, Serialize)]
struct ReferenceFallbackProvenance {
    schema_version: u8,
    reason: &'static str,
    archive: &'static str,
    archive_sha256: String,
    archive_members: Vec<&'static str>,
    quarantined: Vec<QuarantinedReferenceArtifact>,
}

#[derive(Debug, Serialize)]
struct QuarantinedReferenceArtifact {
    path: &'static str,
    sha256: Option<String>,
    validation_error: String,
}

const MAX_FALLBACK_ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FALLBACK_ARCHIVE_MEMBER_BYTES: u64 = 16 * 1024 * 1024;

fn quarantine_invalid_danes_gecl4_outputs(example: &Path, case_dir: &Path) -> Result<bool> {
    if example != Path::new("DANES/GeCl_4") {
        return Ok(false);
    }

    let artifact_names = [
        "phase.bin",
        "emesh.bin",
        "xmu.dat",
        "danes.dat",
        "xsect.dat",
        "fms.bin",
    ];
    let mut invalid = Vec::new();
    for name in artifact_names {
        let path = case_dir.join(name);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                invalid.push(QuarantinedReferenceArtifact {
                    path: name,
                    sha256: None,
                    validation_error: "generated file is missing".to_string(),
                });
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        if let Err(validation_error) = validate_danes_gecl4_artifact(name, &bytes) {
            invalid.push(QuarantinedReferenceArtifact {
                path: name,
                sha256: Some(manifest::sha256_hex(&bytes)),
                validation_error,
            });
        }
    }
    if invalid.is_empty() {
        return Ok(false);
    }

    let archive_path = case_dir.join("REFERENCE.zip");
    let archive_bytes = read_fallback_archive_bytes(&archive_path)?;
    validate_danes_gecl4_archive(&archive_path)?;

    let provenance = ReferenceFallbackProvenance {
        schema_version: 1,
        reason: "pinned FEFF10 generated a non-finite/inconsistent DANES GeCl_4 mesh; invalid local artifacts are excluded rather than weakening strict codecs",
        archive: "REFERENCE.zip",
        archive_sha256: manifest::sha256_hex(&archive_bytes),
        archive_members: vec![
            "REFERENCE/xmu.dat",
            "REFERENCE/danes.dat",
            "REFERENCE/xsect.dat",
            "REFERENCE/fms.bin",
        ],
        quarantined: invalid,
    };
    quarantine_reference_artifacts(case_dir, &provenance)?;
    Ok(true)
}

fn validate_danes_gecl4_artifact(name: &str, bytes: &[u8]) -> std::result::Result<(), String> {
    match name {
        "phase.bin" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| format!("phase.bin is not UTF-8 text: {error}"))?;
            parse_phase_bin(text)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        "emesh.bin" => parse_emesh_bin(bytes)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        "xmu.dat" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| format!("xmu.dat is not UTF-8 text: {error}"))?;
            parse_xmu_dat(text)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        "danes.dat" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| format!("danes.dat is not UTF-8 text: {error}"))?;
            parse_danes_dat(text)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        "xsect.dat" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| format!("xsect.dat is not UTF-8 text: {error}"))?;
            parse_xsect_dat(text)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        "fms.bin" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|error| format!("fms.bin is not UTF-8 text: {error}"))?;
            parse_fms_bin(text)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        _ => Err(format!("unsupported DANES GeCl_4 artifact {name}")),
    }
}

fn validate_danes_gecl4_archive(path: &Path) -> Result<()> {
    let required = [
        ("REFERENCE/xmu.dat", "xmu.dat"),
        ("REFERENCE/danes.dat", "danes.dat"),
        ("REFERENCE/xsect.dat", "xsect.dat"),
        ("REFERENCE/fms.bin", "fms.bin"),
    ];

    for (member, artifact_name) in required {
        let bytes = read_exact_archive_member(path, member)?;
        validate_danes_gecl4_artifact(artifact_name, &bytes).map_err(|error| {
            anyhow::anyhow!(
                "{} member {member} is not a valid finite {artifact_name}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

fn quarantine_invalid_debye_fecn6_outputs(example: &Path, case_dir: &Path) -> Result<bool> {
    if example != Path::new("DEBYE/DM/XANES/FeCN_6") {
        return Ok(false);
    }

    let stdout_path = case_dir.join("feff.stdout");
    let stdout = std::fs::read_to_string(&stdout_path)
        .with_context(|| format!("failed to read {}", stdout_path.display()))?;
    let xmu_path = case_dir.join("xmu.dat");
    let xmu_bytes = std::fs::read(&xmu_path)
        .with_context(|| format!("failed to read {}", xmu_path.display()))?;
    let xmu_text = std::str::from_utf8(&xmu_bytes)
        .with_context(|| format!("{} is not UTF-8 text", xmu_path.display()))?;
    let xmu = parse_xmu_dat(xmu_text)
        .with_context(|| format!("failed to validate {}", xmu_path.display()))?;

    let internal_gg_error =
        stdout.contains("Error: Unexpected end of record while reading from gg.bin.");
    let missing_fms_or_gtr =
        !case_dir.join("fms.bin").is_file() || !case_dir.join("gtr.dat").is_file();
    let zero_fine_structure = xmu.chi.iter().all(|value| *value == 0.0);
    if !(internal_gg_error && missing_fms_or_gtr && zero_fine_structure) {
        return Ok(false);
    }

    let archive_path = case_dir.join("REFERENCE.zip");
    let archive_bytes = read_fallback_archive_bytes(&archive_path)?;
    let archived_xmu = read_exact_archive_member(&archive_path, "REFERENCE/xmu.dat")?;
    let archived_text = std::str::from_utf8(&archived_xmu)
        .context("REFERENCE.zip!/REFERENCE/xmu.dat is not UTF-8 text")?;
    let archived = parse_xmu_dat(archived_text)
        .context("REFERENCE.zip!/REFERENCE/xmu.dat is not a valid finite xmu.dat")?;
    anyhow::ensure!(
        archived.chi.iter().any(|value| *value != 0.0),
        "{} archived xmu.dat also has identically zero fine structure",
        archive_path.display()
    );

    let legacy_path = case_dir.join("referencexmu.dat");
    let legacy_bytes = std::fs::read(&legacy_path)
        .with_context(|| format!("failed to read {}", legacy_path.display()))?;
    anyhow::ensure!(
        legacy_bytes == archived_xmu,
        "{} does not match exact REFERENCE.zip!/REFERENCE/xmu.dat provenance",
        legacy_path.display()
    );

    let provenance = ReferenceFallbackProvenance {
        schema_version: 1,
        reason: "pinned FEFF10 MKGTR rejected gg.bin after an embedded-LF descriptor, omitted FMS/GTR handoffs, and FF2X emitted identically zero chi",
        archive: "REFERENCE.zip",
        archive_sha256: manifest::sha256_hex(&archive_bytes),
        archive_members: vec!["REFERENCE/xmu.dat"],
        quarantined: vec![QuarantinedReferenceArtifact {
            path: "xmu.dat",
            sha256: Some(manifest::sha256_hex(&xmu_bytes)),
            validation_error:
                "chi is identically zero after FEFF reported unexpected EOF reading gg.bin"
                    .to_string(),
        }],
    };
    quarantine_reference_artifacts(case_dir, &provenance)?;
    Ok(true)
}

fn read_exact_archive_member(path: &Path, member: &str) -> Result<Vec<u8>> {
    read_exact_archive_member_with_limit(path, member, MAX_FALLBACK_ARCHIVE_MEMBER_BYTES)
}

fn read_fallback_archive_bytes(path: &Path) -> Result<Vec<u8>> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
    anyhow::ensure!(
        metadata.len() <= MAX_FALLBACK_ARCHIVE_BYTES,
        "{} declares {} byte(s), exceeding the {}-byte fallback archive limit",
        path.display(),
        metadata.len(),
        MAX_FALLBACK_ARCHIVE_BYTES
    );
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_FALLBACK_ARCHIVE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    anyhow::ensure!(
        u64::try_from(bytes.len()).unwrap_or(u64::MAX) <= MAX_FALLBACK_ARCHIVE_BYTES,
        "{} streamed more than the {}-byte fallback archive limit",
        path.display(),
        MAX_FALLBACK_ARCHIVE_BYTES
    );
    Ok(bytes)
}

fn read_exact_archive_member_with_limit(
    path: &Path,
    member: &str,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    anyhow::ensure!(max_bytes > 0, "archive member byte limit must be positive");
    let intended_path = Path::new(member);
    anyhow::ensure!(
        intended_path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_))),
        "archive member {member:?} is not an exact normalized relative path"
    );
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("failed to read {}", path.display()))?;
    let matches =
        exact_central_directory_member_count(path, archive.central_directory_start(), member)?;
    anyhow::ensure!(
        matches == 1,
        "{} must contain exactly one {member}, found {}",
        path.display(),
        matches
    );
    let entry = archive
        .by_name(member)
        .with_context(|| format!("failed to read {member} from {}", path.display()))?;
    anyhow::ensure!(
        entry.is_file(),
        "{} member {member} is not a regular file",
        path.display()
    );
    anyhow::ensure!(
        entry.enclosed_name().as_deref() == Some(intended_path),
        "{} member {member} did not resolve to the exact intended path",
        path.display()
    );
    anyhow::ensure!(
        entry.size() <= max_bytes,
        "{} member {member} declares {} decompressed byte(s), exceeding the {max_bytes}-byte limit",
        path.display(),
        entry.size()
    );
    let read_limit = max_bytes
        .checked_add(1)
        .context("archive member byte limit overflow")?;
    let mut bytes = Vec::new();
    entry
        .take(read_limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to extract {member} from {}", path.display()))?;
    anyhow::ensure!(
        u64::try_from(bytes.len()).unwrap_or(u64::MAX) <= max_bytes,
        "{} member {member} streamed more than the {max_bytes}-byte decompressed limit despite its declared size",
        path.display()
    );
    Ok(bytes)
}

fn exact_central_directory_member_count(
    path: &Path,
    central_directory_start: u64,
    member: &str,
) -> Result<usize> {
    const CENTRAL_DIRECTORY_SIGNATURE: [u8; 4] = *b"PK\x01\x02";
    const END_SIGNATURES: [[u8; 4]; 3] = [*b"PK\x05\x06", *b"PK\x06\x06", *b"PK\x05\x05"];
    const CENTRAL_FIXED_BYTES_AFTER_SIGNATURE: usize = 42;
    const MAX_CENTRAL_DIRECTORY_ENTRIES: usize = 100_000;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    file.seek(SeekFrom::Start(central_directory_start))
        .with_context(|| format!("failed to seek to central directory in {}", path.display()))?;
    let mut exact_count = 0_usize;
    for entry_index in 0..=MAX_CENTRAL_DIRECTORY_ENTRIES {
        let mut signature = [0_u8; 4];
        file.read_exact(&mut signature).with_context(|| {
            format!(
                "failed to read central directory signature {entry_index} in {}",
                path.display()
            )
        })?;
        if signature != CENTRAL_DIRECTORY_SIGNATURE {
            anyhow::ensure!(
                END_SIGNATURES.contains(&signature),
                "{} has invalid central directory signature {:02x?} after {entry_index} entries",
                path.display(),
                signature
            );
            return Ok(exact_count);
        }
        anyhow::ensure!(
            entry_index < MAX_CENTRAL_DIRECTORY_ENTRIES,
            "{} exceeds the {MAX_CENTRAL_DIRECTORY_ENTRIES}-entry archive safety limit",
            path.display()
        );

        let mut fixed = [0_u8; CENTRAL_FIXED_BYTES_AFTER_SIGNATURE];
        file.read_exact(&mut fixed).with_context(|| {
            format!(
                "failed to read central directory entry {entry_index} in {}",
                path.display()
            )
        })?;
        let name_len = usize::from(u16::from_le_bytes([fixed[24], fixed[25]]));
        let extra_len = u64::from(u16::from_le_bytes([fixed[26], fixed[27]]));
        let comment_len = u64::from(u16::from_le_bytes([fixed[28], fixed[29]]));
        let mut name = vec![0_u8; name_len];
        file.read_exact(&mut name).with_context(|| {
            format!(
                "failed to read central directory name {entry_index} in {}",
                path.display()
            )
        })?;
        if name == member.as_bytes() {
            exact_count += 1;
        }
        let skip = extra_len
            .checked_add(comment_len)
            .context("central directory variable-length fields overflow")?;
        file.seek(SeekFrom::Current(
            i64::try_from(skip).context("central directory field length exceeds i64")?,
        ))
        .with_context(|| {
            format!(
                "failed to skip central directory metadata {entry_index} in {}",
                path.display()
            )
        })?;
    }
    anyhow::bail!(
        "{} central directory did not terminate within the entry safety limit",
        path.display()
    )
}

fn quarantine_reference_artifacts(
    case_dir: &Path,
    provenance: &ReferenceFallbackProvenance,
) -> Result<()> {
    write_reference_fallback_marker_atomic(case_dir, provenance)?;
    for artifact in &provenance.quarantined {
        let path = case_dir.join(artifact.path);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to quarantine {}", path.display()));
            }
        }
    }
    Ok(())
}

fn write_reference_fallback_marker_atomic(
    case_dir: &Path,
    provenance: &ReferenceFallbackProvenance,
) -> Result<()> {
    let marker_path = case_dir.join(".reference-fallback.json");
    let mut marker = serde_json::to_vec_pretty(provenance)
        .context("failed to serialize reference fallback provenance")?;
    marker.push(b'\n');

    match std::fs::read(&marker_path) {
        Ok(existing) if existing == marker => return Ok(()),
        Ok(_) => anyhow::bail!(
            "refusing to overwrite different fallback provenance at {}",
            marker_path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", marker_path.display()));
        }
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let temporary_path = case_dir.join(format!(
        ".reference-fallback.json.tmp-{}-{stamp}",
        std::process::id()
    ));
    let write_result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)
            .with_context(|| format!("failed to create {}", temporary_path.display()))?;
        file.write_all(&marker)
            .with_context(|| format!("failed to write {}", temporary_path.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temporary_path.display()))?;
        std::fs::rename(&temporary_path, &marker_path).with_context(|| {
            format!(
                "failed to atomically publish {} as {}",
                temporary_path.display(),
                marker_path.display()
            )
        })
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    write_result
}

fn stamp_golden_manifests(ref_dir: Option<PathBuf>, force: bool) -> Result<()> {
    let workspace_root = env::current_dir().context("failed to resolve the workspace root")?;
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| workspace_root.join("feff10"))
        .canonicalize()
        .context("failed to resolve the pinned FEFF10 checkout")?;
    let feff10_rev = manifest::feff10_git_rev(&ref_dir)
        .context("the reference directory is not a FEFF10 git checkout")?;
    let compiler = manifest::CompilerInfo::unknown();
    let mut written = 0usize;

    for relative in compatibility_matrix::golden_fixture_directories() {
        let case_dir = workspace_root.join(relative);
        if !case_dir.is_dir() || (!force && manifest::has_manifest(&case_dir)) {
            continue;
        }
        manifest::write_manifest(&case_dir, Some(&feff10_rev), &compiler)
            .with_context(|| format!("failed to stamp {}", case_dir.display()))?;
        written += 1;
        println!("stamped {relative}");
    }

    println!("stamped {written} compatibility fixture manifest(s) at FEFF10 revision {feff10_rev}");
    Ok(())
}

fn generate_exact_native_rixs_reference(
    ref_dir: &Path,
    generated_case: &Path,
    native_commit: &str,
) -> Result<()> {
    let temporary = temporary_work_dir("refeff-native-rixs-golden")?;
    let generation =
        generate_exact_native_rixs_reference_in(ref_dir, generated_case, native_commit, &temporary);
    let cleanup = std::fs::remove_dir_all(&temporary)
        .with_context(|| format!("failed to remove {}", temporary.display()));
    match (generation, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
    }
}

fn generate_exact_native_rixs_reference_in(
    ref_dir: &Path,
    generated_case: &Path,
    native_commit: &str,
    temporary: &Path,
) -> Result<()> {
    anyhow::ensure!(
        native_commit.len() == 40
            && native_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "exact RIXS generation requires a lowercase 40-digit native commit"
    );
    let work_dir = temporary.join("work");
    std::fs::create_dir_all(&work_dir)
        .with_context(|| format!("failed to create {}", work_dir.display()))?;
    let source_input = generated_case.join("feff.inp");
    let source = std::fs::read_to_string(&source_input)
        .with_context(|| format!("failed to read {}", source_input.display()))?;
    let (incident_edge, final_edge) = exact_rixs_edge_pair(&source)?;
    anyhow::ensure!(
        incident_edge == "L3" && final_edge == "VAL",
        "exact RIXS generator supports only the stock EDGE L3 VAL example"
    );
    std::fs::write(work_dir.join("feff.inp"), &source)
        .with_context(|| format!("failed to stage {}", source_input.display()))?;

    let mut edge_provenance = Vec::new();
    for (edge, destination_index) in [("L3", 1usize), ("VAL", 2usize)] {
        let edge_dir = work_dir.join(edge);
        std::fs::create_dir_all(&edge_dir)
            .with_context(|| format!("failed to create {}", edge_dir.display()))?;
        let derived_input = exact_rixs_edge_input(&source, &incident_edge, edge)?;
        std::fs::write(edge_dir.join("feff.inp"), &derived_input)
            .with_context(|| format!("failed to write {edge} RIXS edge input"))?;

        let output = run_reference_program(ref_dir, ReferenceProgram::Feff, &edge_dir)
            .with_context(|| format!("failed to run native FEFF {edge} RIXS edge calculation"))?;
        std::fs::write(edge_dir.join("feff.stdout"), &output.stdout)?;
        std::fs::write(edge_dir.join("feff.stderr"), &output.stderr)?;
        anyhow::ensure!(
            output.success,
            "native FEFF {edge} RIXS edge calculation failed with status {}",
            output.status
        );

        for name in ["phase.bin", "rl.dat", "gg.bin"] {
            let path = edge_dir.join(name);
            let metadata = std::fs::symlink_metadata(&path).with_context(|| {
                format!("native FEFF {edge} edge did not produce {}", path.display())
            })?;
            anyhow::ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "native FEFF {edge} handoff {} is not a regular file",
                path.display()
            );
        }
        let original_gg_path = edge_dir.join("gg.bin");
        let original_gg = rixs_reference::read_bounded_regular_file(
            &original_gg_path,
            rixs_reference::MAX_GG_BYTES,
            "native edge GG",
        )?;
        let sanitized = rixs_reference::sanitize_native_gg_descriptors(&original_gg)
            .with_context(|| format!("failed to normalize {edge} GG descriptors"))?;
        let original_gg_sha256 = manifest::sha256_hex(&original_gg);
        let sanitized_gg_sha256 = manifest::sha256_hex(&sanitized.bytes);
        anyhow::ensure!(
            original_gg_sha256 != sanitized_gg_sha256,
            "native FEFF {edge} GG did not contain the current-source descriptor defect"
        );

        copy_native_rixs_handoff(
            &edge_dir.join("phase.bin"),
            &work_dir.join(format!("phase_{destination_index}.bin")),
        )?;
        copy_native_rixs_handoff(
            &edge_dir.join("rl.dat"),
            &work_dir.join(format!("rl_{destination_index}.dat")),
        )?;
        std::fs::write(
            work_dir.join(format!("gg_{destination_index}.bin")),
            &sanitized.bytes,
        )
        .with_context(|| format!("failed to stage sanitized {edge} GG"))?;
        if edge == "L3" {
            copy_native_rixs_handoff(&edge_dir.join("wscrn.dat"), &work_dir.join("wscrn_1.dat"))?;
        } else {
            copy_native_rixs_handoff(&edge_dir.join("xsect.dat"), &work_dir.join("xsect_2.dat"))?;
        }

        edge_provenance.push(rixs_reference::NativeRixsEdgeProvenance {
            edge: edge.to_string(),
            derived_input_sha256: manifest::sha256_hex(derived_input.as_bytes()),
            original_gg_sha256,
            sanitized_gg_sha256,
            descriptor_records: sanitized.descriptor_records,
            continuation_lines_removed: sanitized.continuation_lines_removed,
        });
    }

    let zero_screen = exact_rixs_val_zero_screen(
        &work_dir.join("L3").join("wscrn.dat"),
        &work_dir.join("VAL").join("rl.dat"),
    )?;
    std::fs::write(work_dir.join("wscrn_2.dat"), &zero_screen)
        .context("failed to stage native RIXS VAL zero-screen handoff")?;
    let zero_screen_data = refeff_io::parse_wscrn_dat(
        std::str::from_utf8(&zero_screen)
            .context("generated native RIXS VAL zero-screen handoff is not valid UTF-8")?,
    )?;

    for (name, description) in [
        ("rdinp", "FEFF10 RDINP for exact RIXS reference"),
        ("atomic", "FEFF10 ATOMIC for exact RIXS reference"),
        ("rixs", "FEFF10 RIXS current-source oracle"),
    ] {
        run_reference_subprogram(
            &ref_dir.join("bin/Seq").join(name),
            &work_dir,
            name,
            description,
        )?;
    }

    let staged_map_path = work_dir.join(rixs_reference::MAP_FILE_NAME);
    let map_validation = rixs_reference::validate_current_source_map_file(&staged_map_path)?;
    let map_bytes = rixs_reference::read_bounded_regular_file(
        &staged_map_path,
        rixs_reference::MAX_MAP_BYTES,
        "native RIXS map",
    )?;
    let provenance = rixs_reference::NativeRixsProvenance {
        schema_version: rixs_reference::PROVENANCE_SCHEMA_VERSION,
        generator: "xtask generate-golden exact RIXS native-current-source oracle".to_string(),
        native_commit: native_commit.to_string(),
        normalization_operation: rixs_reference::GG_NORMALIZATION_OPERATION.to_string(),
        edges: edge_provenance,
        val_zero_screen: rixs_reference::NativeRixsZeroScreenProvenance {
            derivation: "zero both potential columns on exp(-x0 + row*dx), using VAL/rl.dat dx/x0 and the native L3/wscrn.dat row count".to_string(),
            row_count: zero_screen_data.row_count(),
            sha256: manifest::sha256_hex(&zero_screen),
        },
        solver: rixs_reference::NativeRixsSolverProvenance {
            executable: "bin/Seq/rixs".to_string(),
            output_sha256: manifest::sha256_hex(&map_bytes),
            map_order: rixs_reference::MAP_ORDER,
            point_count: rixs_reference::MAP_POINT_COUNT,
            peak_row: map_validation.peak_row,
            peak_first_energy_ev: map_validation.peak_first_energy_ev,
            peak_second_energy_ev: map_validation.peak_second_energy_ev,
            peak_intensity: map_validation.peak_intensity,
        },
    };
    let mut provenance_bytes =
        serde_json::to_vec_pretty(&provenance).context("failed to serialize RIXS provenance")?;
    provenance_bytes.push(b'\n');

    atomic_publish_case_file(
        generated_case,
        Path::new(rixs_reference::MAP_FILE_NAME),
        &map_bytes,
    )?;
    atomic_publish_case_file(
        generated_case,
        Path::new(rixs_reference::PROVENANCE_FILE_NAME),
        &provenance_bytes,
    )?;
    rixs_reference::validate_published_reference(generated_case)?;
    remove_legacy_rixs_reference_alias(generated_case)?;
    Ok(())
}

fn exact_rixs_edge_pair(source: &str) -> Result<(String, String)> {
    let mut edge_cards = source.lines().filter_map(|line| {
        let mut fields = line.trim_start().split_ascii_whitespace();
        let keyword = fields.next()?;
        keyword
            .eq_ignore_ascii_case("EDGE")
            .then(|| fields.map(str::to_string).collect::<Vec<_>>())
    });
    let edges = edge_cards
        .next()
        .context("stock RIXS input has no EDGE card")?;
    anyhow::ensure!(
        edge_cards.next().is_none(),
        "stock RIXS input has more than one EDGE card"
    );
    anyhow::ensure!(
        edges.len() == 2,
        "stock RIXS EDGE card must name exactly two edges"
    );
    Ok((edges[0].to_ascii_uppercase(), edges[1].to_ascii_uppercase()))
}

fn exact_rixs_edge_input(
    source: &str,
    incident_edge: &str,
    calculation_edge: &str,
) -> Result<String> {
    anyhow::ensure!(
        incident_edge == "L3" && matches!(calculation_edge, "L3" | "VAL"),
        "exact RIXS edge derivation supports only L3 and VAL"
    );
    let mut output = String::new();
    if calculation_edge == "VAL" {
        output.push_str("COREHOLE NONE\n");
    } else {
        output.push_str("ICORE 4\nCOREHOLE RPA\n");
    }
    output.push_str("RLPRINT\nEDGE L3\nXANES 20\n");

    for line in source.lines() {
        let card = line
            .trim_start()
            .split_ascii_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();
        if matches!(
            card.as_str(),
            "COREHOLE" | "HOLE" | "ICORE" | "RLPRINT" | "EDGE" | "RIXS" | "XES" | "XANES"
        ) {
            continue;
        }
        output.push_str(line.trim_end_matches('\r'));
        output.push('\n');
    }
    Ok(output)
}

fn copy_native_rixs_handoff(source: &Path, destination: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(source)
        .with_context(|| format!("failed to inspect native RIXS handoff {}", source.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "native RIXS handoff {} must be a regular non-symlink file",
        source.display()
    );
    std::fs::copy(source, destination).with_context(|| {
        format!(
            "failed to stage native RIXS handoff {} as {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn exact_rixs_val_zero_screen(incident_wscrn: &Path, final_rl: &Path) -> Result<Vec<u8>> {
    let incident = read_wscrn_dat(incident_wscrn)
        .with_context(|| format!("failed to read {}", incident_wscrn.display()))?;
    let row_count = incident.row_count();
    anyhow::ensure!(
        row_count > 0 && row_count <= 4096,
        "native incident wscrn.dat row count {row_count} is outside the bounded RIXS range"
    );
    let rl_bytes =
        rixs_reference::read_bounded_regular_file(final_rl, 64 * 1024 * 1024, "native VAL rl.dat")?;
    let rl_text = std::str::from_utf8(&rl_bytes)
        .with_context(|| format!("{} is not valid UTF-8", final_rl.display()))?;
    let second_line = rl_text
        .lines()
        .nth(1)
        .context("native VAL rl.dat has no dx/x0 header row")?;
    let values = second_line
        .split_ascii_whitespace()
        .map(str::parse::<f64>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("native VAL rl.dat has an invalid dx/x0 header row")?;
    anyhow::ensure!(
        values.len() == 2 && values[0].is_finite() && values[0] > 0.0 && values[1].is_finite(),
        "native VAL rl.dat dx/x0 header is invalid"
    );
    let (dx, x0) = (values[0], values[1]);
    let zero_screen = WscrnDatData {
        header_lines: Vec::new(),
        radius_bohr: Array1::from_shape_fn(row_count, |row| (-x0 + row as f64 * dx).exp()),
        screened_potential: Array1::zeros(row_count),
        core_hole_potential: Array1::zeros(row_count),
    };
    Ok(wscrn_dat_string(&zero_screen)?.into_bytes())
}

fn atomic_publish_case_file(case_dir: &Path, name: &Path, bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        name.components().count() == 1
            && name
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "RIXS publication target must be one normal path component: {}",
        name.display()
    );
    let case_metadata = std::fs::symlink_metadata(case_dir)
        .with_context(|| format!("failed to inspect {}", case_dir.display()))?;
    anyhow::ensure!(
        case_metadata.file_type().is_dir() && !case_metadata.file_type().is_symlink(),
        "RIXS publication root {} must be a regular directory",
        case_dir.display()
    );
    let destination = case_dir.join(name);
    match std::fs::symlink_metadata(&destination) {
        Ok(metadata) => {
            anyhow::ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "refusing to overwrite non-regular RIXS artifact {}",
                destination.display()
            );
            let existing = std::fs::read(&destination)
                .with_context(|| format!("failed to read {}", destination.display()))?;
            anyhow::ensure!(
                existing == bytes,
                "refusing to overwrite different RIXS artifact {}",
                destination.display()
            );
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", destination.display()));
        }
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    let temporary = case_dir.join(format!(
        ".{}.rixs-tmp-{}-{stamp}",
        name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync {}", temporary.display()))?;
        std::fs::rename(&temporary, &destination).with_context(|| {
            format!(
                "failed to publish {} as {}",
                temporary.display(),
                destination.display()
            )
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result?;
    anyhow::ensure!(
        std::fs::read(&destination)
            .with_context(|| format!("failed to verify {}", destination.display()))?
            == bytes,
        "published RIXS artifact {} failed byte verification",
        destination.display()
    );
    Ok(())
}

fn remove_legacy_rixs_reference_alias(case_dir: &Path) -> Result<bool> {
    let path = case_dir.join(rixs_reference::LEGACY_MAP_FILE_NAME);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    };
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "refusing to remove non-regular legacy RIXS reference {}",
        path.display()
    );
    std::fs::remove_file(&path)
        .with_context(|| format!("failed to remove stale {}", path.display()))?;
    Ok(true)
}

fn generate_compatibility_reference_subcases(
    ref_dir: &Path,
    example: &Path,
    generated_case: &Path,
) -> Result<()> {
    if example == Path::new("XANES/Cu") {
        generate_cu_tdlda_reference(ref_dir, generated_case)?;
        generate_cu_rhorrp_reference(ref_dir, generated_case)?;
    }
    Ok(())
}

fn generate_cu_rhorrp_reference(ref_dir: &Path, generated_case: &Path) -> Result<()> {
    // Keep this beneath the stock XANES/Cu case: its parent manifest hashes
    // nested files recursively, so the reference is provenance-tracked
    // without changing the stock-workflow manifest count.
    let destination = generated_case.join("rhorrp-density");
    std::fs::create_dir_all(&destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;

    for name in [
        ".dimensions.dat",
        "global.inp",
        "geom.dat",
        "pot.inp",
        "pot.bin",
        "reciprocal.inp",
        "xsph.inp",
        "phase.bin",
        "hubbard.inp",
        "config.dat",
    ] {
        let source = generated_case.join(name);
        anyhow::ensure!(
            source.is_file(),
            "missing XANES/Cu RHORRP reference input {}",
            source.display()
        );
        std::fs::copy(&source, destination.join(name)).with_context(|| {
            format!("failed to copy RHORRP reference input {}", source.display())
        })?;
    }

    let source_fms_path = generated_case.join("fms.inp");
    let source_fms_text = std::fs::read_to_string(&source_fms_path)
        .with_context(|| format!("failed to read {}", source_fms_path.display()))?;
    let mut fms = FmsInput::parse_str(&source_fms_path, &source_fms_text)
        .with_context(|| format!("failed to parse {}", source_fms_path.display()))?;
    fms.save_gg_slice = true;
    std::fs::write(destination.join("fms.inp"), fms_input_string(&fms)?)?;
    std::fs::write(
        destination.join("density.inp"),
        concat!(
            "line density.dat 0.0 0.0 0.0 core\n",
            "1.0 0.0 0.0 2\n",
            "line density.bin 0.0 0.0 0.0 core\n",
            "1.0 0.0 0.0 2\n",
        ),
    )?;

    run_reference_subprogram(
        &ref_dir.join("bin/Seq/fms"),
        &destination,
        "fms",
        "FEFF10 FMS RHORRP compatibility reference",
    )?;
    anyhow::ensure!(
        destination.join("gg_slice.bin").is_file() && destination.join("gg_diag.bin").is_file(),
        "FEFF10 FMS RHORRP compatibility reference did not produce gg_slice.bin and gg_diag.bin"
    );

    run_reference_subprogram(
        &ref_dir.join("bin/Seq/rhorrp"),
        &destination,
        "rhorrp",
        "FEFF10 RHORRP compatibility reference",
    )?;
    anyhow::ensure!(
        destination.join("density.dat").is_file() && destination.join("density.bin").is_file(),
        "FEFF10 RHORRP compatibility reference did not produce density.dat and density.bin"
    );
    Ok(())
}

fn run_reference_subprogram(
    executable: &Path,
    work_dir: &Path,
    log_prefix: &str,
    description: &str,
) -> Result<()> {
    anyhow::ensure!(
        executable.is_file(),
        "missing {description} executable {}",
        executable.display()
    );
    let output = std::process::Command::new(executable)
        .current_dir(work_dir)
        .output()
        .with_context(|| format!("failed to run {}", executable.display()))?;
    std::fs::write(
        work_dir.join(format!("{log_prefix}.stdout")),
        &output.stdout,
    )?;
    std::fs::write(
        work_dir.join(format!("{log_prefix}.stderr")),
        &output.stderr,
    )?;
    anyhow::ensure!(
        output.status.success(),
        "{description} failed with status {}",
        output.status
    );
    Ok(())
}

fn generate_cu_tdlda_reference(ref_dir: &Path, generated_case: &Path) -> Result<()> {
    let destination = generated_case.join("tdlda-occupied");
    std::fs::create_dir_all(&destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;

    for name in [
        ".dimensions.dat",
        "global.inp",
        "pot.bin",
        "pot.inp",
        "geom.dat",
        "config.dat",
        "reciprocal.inp",
        "wscrn.dat",
        "xmu.dat",
        "hubbard.inp",
    ] {
        let source = generated_case.join(name);
        if source.is_file() {
            std::fs::copy(&source, destination.join(name)).with_context(|| {
                format!("failed to copy TDLDA reference input {}", source.display())
            })?;
        }
    }

    let source_xsph_path = generated_case.join("xsph.inp");
    let source_xsph_text = std::fs::read_to_string(&source_xsph_path)
        .with_context(|| format!("failed to read {}", source_xsph_path.display()))?;
    let mut xsph = XsphInput::parse_str(&source_xsph_path, &source_xsph_text)
        .with_context(|| format!("failed to parse {}", source_xsph_path.display()))?;
    xsph.advanced.izstd = 0;
    xsph.advanced.ifxc = 0;
    xsph.advanced.ipmbse = 2;
    xsph.advanced.itdlda = 2;
    xsph.advanced.nonlocal = 0;
    xsph.advanced.ibasis = 0;
    std::fs::write(destination.join("xsph.inp"), xsph_input_string(&xsph)?)?;
    std::fs::write(destination.join("listedges.pmbse"), ".\n")?;

    let executable = ref_dir.join("bin/Seq/xsph");
    anyhow::ensure!(
        executable.is_file(),
        "missing FEFF10 XSPH executable {}",
        executable.display()
    );
    let output = std::process::Command::new(&executable)
        .current_dir(&destination)
        .output()
        .with_context(|| format!("failed to run {}", executable.display()))?;
    std::fs::write(destination.join("xsph.stdout"), &output.stdout)?;
    std::fs::write(destination.join("xsph.stderr"), &output.stderr)?;
    anyhow::ensure!(
        output.status.success(),
        "FEFF10 XSPH TDLDA reference failed with status {}",
        output.status
    );
    anyhow::ensure!(
        destination.join("phase.bin").is_file() && destination.join("xsedge.dat").is_file(),
        "FEFF10 XSPH TDLDA reference did not produce phase.bin and xsedge.dat"
    );
    Ok(())
}

fn build_reference_feff(ref_dir: &Path) -> Result<manifest::CompilerInfo> {
    let src = ref_dir.join("src");
    let mut command = std::process::Command::new("make");
    command.arg("all").arg("band").current_dir(&src);
    let compiler = if !command_exists("ifort") && command_exists("gfortran") {
        let flags = "-ffree-line-length-none -cpp -O3 -fallow-argument-mismatch";
        command
            .arg("F90=gfortran")
            .arg(format!("FLAGS={flags}"))
            .arg("MPIF90=gfortran")
            .arg(format!("MPIFLAGS={flags}"));
        manifest::CompilerInfo {
            name: "gfortran".to_string(),
            flags: flags.to_string(),
        }
    } else if command_exists("ifort") {
        manifest::CompilerInfo {
            name: "ifort".to_string(),
            flags: "Makefile default (FLAGS unset)".to_string(),
        }
    } else {
        manifest::CompilerInfo::unknown()
    };

    let status = command.status()?;
    if !status.success() {
        anyhow::bail!("failed to build FEFF reference in {}", src.display());
    }
    Ok(compiler)
}

struct ReferenceOutput {
    success: bool,
    status: String,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_reference_program(
    ref_dir: &Path,
    program: ReferenceProgram,
    work_dir: &Path,
) -> Result<ReferenceOutput> {
    if let Ok(driver) = reference_driver(ref_dir, program) {
        let output = std::process::Command::new(driver)
            .current_dir(work_dir)
            .output()?;
        return Ok(ReferenceOutput {
            success: output.status.success(),
            status: output.status.to_string(),
            stdout: output.stdout,
            stderr: output.stderr,
        });
    }
    anyhow::ensure!(
        program == ReferenceProgram::Feff,
        "no {} reference driver found under {}",
        program.log_prefix(),
        ref_dir.display()
    );

    let sequence = [
        "rdinp",
        "dmdw",
        "atomic",
        "pot",
        "ldos",
        "screen",
        "crpa",
        "opconsat",
        "xsph",
        "fms",
        "mkgtr",
        "band",
        "path",
        "genfmt",
        "ff2x",
        "sfconv",
        "fullspectrum",
        "compton",
        "eels",
        "rhorrp",
        "rixs",
    ];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    for name in sequence {
        let executable = ref_dir.join("bin/Seq").join(name);
        anyhow::ensure!(
            executable.is_file(),
            "missing FEFF10 module executable {}",
            executable.display()
        );
        let output = std::process::Command::new(&executable)
            .current_dir(work_dir)
            .output()
            .with_context(|| format!("failed to run {}", executable.display()))?;
        stdout.extend_from_slice(format!("\n== {name} ==\n").as_bytes());
        stdout.extend_from_slice(&output.stdout);
        stderr.extend_from_slice(format!("\n== {name} ==\n").as_bytes());
        stderr.extend_from_slice(&output.stderr);
        if !output.status.success() {
            return Ok(ReferenceOutput {
                success: false,
                status: format!("{name}: {}", output.status),
                stdout,
                stderr,
            });
        }
    }
    Ok(ReferenceOutput {
        success: true,
        status: "all module drivers succeeded".to_string(),
        stdout,
        stderr,
    })
}

fn command_exists(command: &str) -> bool {
    let command_path = Path::new(command);
    if command_path.components().count() > 1 {
        return command_path.is_file();
    }

    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|dir| dir.join(command).is_file())
}

fn reference_driver(ref_dir: &Path, program: ReferenceProgram) -> Result<PathBuf> {
    program
        .binary_candidates(ref_dir)
        .into_iter()
        .find(|candidate| candidate.exists())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no {} reference driver found under {}; run xtask generate-golden without --no-build or build FEFF manually",
                program.log_prefix(),
                ref_dir.display()
            )
        })
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            std::fs::create_dir_all(&dst)?;
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

fn run_reference_tests(ref_dir: Option<PathBuf>) -> Result<()> {
    let ref_dir = ref_dir
        .or_else(|| env::var_os("FEFF10_REF").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("feff10"));
    let examples_dir = ref_dir.join("examples");
    let mut inputs = Vec::new();
    collect_feff_inputs(&examples_dir, &mut inputs)?;
    inputs.sort();

    let mut total_cards = 0_usize;
    let mut total_atoms = 0_usize;
    let mut total_potentials = 0_usize;
    let mut skipped_invalid_templates = 0_usize;
    for input in &inputs {
        let parsed = match FeffInput::parse_file(input) {
            Ok(parsed) => parsed,
            Err(error) => {
                let message = error.to_string();
                if is_expected_invalid_template(input, &message) {
                    skipped_invalid_templates += 1;
                    continue;
                }
                return Err(error.into());
            }
        };
        let document = match FeffDocument::from_input(&parsed) {
            Ok(document) => document,
            Err(error) => {
                let message = error.to_string();
                if is_expected_invalid_template(input, &message) {
                    skipped_invalid_templates += 1;
                    continue;
                }
                return Err(error.into());
            }
        };
        total_cards += parsed.cards().count();
        total_atoms += document.atoms.len();
        total_potentials += document.potentials.len();
    }

    println!(
        "parsed {} FEFF examples: cards={} atoms={} potentials={} skipped_invalid_templates={}",
        inputs.len() - skipped_invalid_templates,
        total_cards,
        total_atoms,
        total_potentials,
        skipped_invalid_templates
    );
    Ok(())
}

fn is_expected_invalid_template(input: &Path, message: &str) -> bool {
    input
        .components()
        .any(|component| component.as_os_str() == "HIGHZ")
        && message.contains("XXX")
}

fn collect_feff_inputs(dir: &Path, inputs: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_feff_inputs(&path, inputs)?;
        } else if path.file_name().is_some_and(|name| name == "feff.inp") {
            inputs.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rixs_edge_derivation_is_bounded_to_stock_l3_val() -> Result<()> {
        let source = "TITLE Pt\nEDGE L3 VAL\nCOREHOLE RPA\nRIXS\nXES -20 15 0.05\nEGRID\ne_grid -15 -1 1\nEND\n";
        assert_eq!(
            exact_rixs_edge_pair(source)?,
            ("L3".to_string(), "VAL".to_string())
        );

        let incident = exact_rixs_edge_input(source, "L3", "L3")?;
        assert!(incident.starts_with("ICORE 4\nCOREHOLE RPA\nRLPRINT\nEDGE L3\nXANES 20\n"));
        assert_eq!(incident.matches("EDGE L3").count(), 1);
        assert!(!incident.lines().any(|line| line == "RIXS"));
        assert!(!incident.lines().any(|line| line.starts_with("XES")));

        let final_state = exact_rixs_edge_input(source, "L3", "VAL")?;
        assert!(final_state.starts_with("COREHOLE NONE\nRLPRINT\nEDGE L3\nXANES 20\n"));
        assert!(!final_state.contains("ICORE"));
        assert!(exact_rixs_edge_input(source, "K", "VAL").is_err());
        assert!(exact_rixs_edge_pair("EDGE L3\n").is_err());
        Ok(())
    }

    #[test]
    fn exact_rixs_selection_does_not_match_nrixs() {
        let selection = vec!["RIXS".to_string()];
        assert!(selected_relative_example(Path::new("RIXS"), &selection));
        assert!(!selected_relative_example(
            Path::new("NRIXS/GeCl_4"),
            &selection
        ));
        assert!(!selected_relative_example(
            Path::new("NRIXS/MgB2"),
            &selection
        ));
    }

    #[test]
    fn exact_rixs_zero_screen_uses_val_rl_grid_and_incident_row_count() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-rixs-zero-screen-test")?;
        let incident = root.join("wscrn.dat");
        std::fs::write(
            &incident,
            concat!(
                "# native incident screen\n",
                "    0.1000000000E-03    0.1000000000E+01    0.2000000000E+01\n",
                "    0.2000000000E-03    0.1000000000E+01    0.2000000000E+01\n",
                "    0.3000000000E-03    0.1000000000E+01    0.2000000000E+01\n",
            ),
        )?;
        let rl = root.join("rl.dat");
        std::fs::write(&rl, "    0.28E+01 5 198\n    0.50E-01 0.88E+01\n")?;

        let bytes = exact_rixs_val_zero_screen(&incident, &rl)?;
        let parsed = refeff_io::parse_wscrn_dat(std::str::from_utf8(&bytes)?)?;

        assert_eq!(parsed.row_count(), 3);
        assert!((parsed.radius_bohr[0] - (-8.8_f64).exp()).abs() < 1.0e-13);
        assert!((parsed.radius_bohr[2] - (-8.7_f64).exp()).abs() < 1.0e-13);
        assert!(parsed.screened_potential.iter().all(|value| *value == 0.0));
        assert!(parsed.core_hole_potential.iter().all(|value| *value == 0.0));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn rixs_atomic_publication_is_path_safe_and_byte_idempotent() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-rixs-publish-test")?;
        let target = Path::new("rixsET.dat");

        atomic_publish_case_file(&root, target, b"validated native map\n")?;
        atomic_publish_case_file(&root, target, b"validated native map\n")?;
        assert_eq!(std::fs::read(root.join(target))?, b"validated native map\n");
        assert!(atomic_publish_case_file(&root, target, b"different\n").is_err());
        assert!(atomic_publish_case_file(&root, Path::new("../escape"), b"x").is_err());
        assert!(
            !root
                .parent()
                .unwrap_or(Path::new("."))
                .join("escape")
                .exists()
        );
        assert!(
            std::fs::read_dir(&root)?
                .filter_map(std::result::Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().contains(".rixs-tmp-"))
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn highz_filter_removes_only_empty_feff_error() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-highz-filter-test")?;
        let sentinel = root.join(".feff.error");
        std::fs::write(&sentinel, [])?;
        anyhow::ensure!(remove_empty_feff_error(&root)?);
        anyhow::ensure!(!sentinel.exists());

        std::fs::write(&sentinel, "real failure\n")?;
        anyhow::ensure!(!remove_empty_feff_error(&root)?);
        anyhow::ensure!(std::fs::read_to_string(&sentinel)? == "real failure\n");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn hubbard_nio_clean_generation_removes_only_generated_caches() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-hubbard-clean-test")?;
        for name in HUBBARD_NIO_GENERATED_CACHES {
            std::fs::write(root.join(name), format!("stale {name}\n"))?;
        }
        std::fs::write(root.join("feff.inp"), "HUBBARD 8 0 0 2\n")?;
        std::fs::write(root.join("REFERENCE.zip"), b"opaque pinned archive")?;

        anyhow::ensure!(
            clean_hubbard_nio_generated_caches(Path::new(HUBBARD_NIO_EXAMPLE), &root)?
                == HUBBARD_NIO_GENERATED_CACHES.len()
        );
        for name in HUBBARD_NIO_GENERATED_CACHES {
            anyhow::ensure!(!root.join(name).exists(), "stale cache {name} survived");
        }
        anyhow::ensure!(root.join("feff.inp").is_file());
        anyhow::ensure!(root.join("REFERENCE.zip").is_file());

        std::fs::write(root.join("xmu.dat"), "other example output\n")?;
        anyhow::ensure!(clean_hubbard_nio_generated_caches(Path::new("HUBBARD/CeO2"), &root)? == 0);
        anyhow::ensure!(root.join("xmu.dat").is_file());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn hubbard_nio_internal_mkgtr_failure_removes_incomplete_case() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-hubbard-fail-closed-test")?;
        std::fs::write(
            root.join("fms.inp"),
            concat!(
                "mfms, idwopt, minv\n",
                "   1   0   0\n",
                "rfms2, rdirec, toler1, toler2\n",
                "      4.00000      9.00000      0.00100      0.00100\n",
                "tk, thetad, sig2g\n",
                "      0.00000      0.00000      0.00000\n",
                " lmaxph(0:nph)\n",
                "   1   2\n",
                " the number of decomposi\n",
                "   -1\n",
                " save_gg_slice\n",
                "F\n",
                "do_fms\n",
                "   1\n",
            ),
        )?;
        std::fs::write(root.join("REFERENCE.zip"), b"stale cache bundle")?;
        let output = ReferenceOutput {
            success: true,
            status: "exit status: 0".to_string(),
            stdout: format!(
                "MKGTR: Tracing over Green's function ...\n{MKGTR_UNEXPECTED_GG_EOF}\n"
            )
            .into_bytes(),
            stderr: Vec::new(),
        };

        let validation =
            validate_hubbard_nio_generation(Path::new(HUBBARD_NIO_EXAMPLE), &root, &output, false);
        let error = fail_closed_hubbard_generation(&root, validation)
            .expect_err("internal MKGTR failure must reject the generated case");
        anyhow::ensure!(
            error
                .to_string()
                .contains("removed incomplete Hubbard reference case"),
            "unexpected fail-closed error: {error:#}"
        );
        anyhow::ensure!(
            format!("{error:#}").contains("refusing stale archive/root"),
            "MKGTR root cause was lost: {error:#}"
        );
        anyhow::ensure!(
            !root.exists(),
            "incomplete Hubbard case survived validation failure"
        );
        Ok(())
    }

    #[test]
    fn hubbard_nio_gg_repair_changes_only_exact_bounded_descriptors() -> Result<()> {
        let mut original = Vec::new();
        let mut expected = Vec::new();
        for section in 1..=HUBBARD_NIO_EXPECTED_ENERGY_COUNT {
            let prefix = format!("#SN#   Section:    {section}\n");
            let payload = format!(
                "#H#\n#DT# 2D complex array with sizes    1   1\n {section}.0 {section}.5\n"
            );
            original.extend_from_slice(prefix.as_bytes());
            original.extend_from_slice(b"#DF# This section written in \x10,\t\n.\n");
            original.extend_from_slice(payload.as_bytes());
            expected.extend_from_slice(prefix.as_bytes());
            expected.extend_from_slice(HUBBARD_NIO_GG_DESCRIPTOR);
            expected.extend_from_slice(payload.as_bytes());
        }

        let repaired = sanitize_hubbard_nio_gg_descriptors(&original)?;
        anyhow::ensure!(repaired.bytes == expected);
        anyhow::ensure!(
            repaired.descriptor_records == HUBBARD_NIO_EXPECTED_ENERGY_COUNT
                && repaired.continuation_lines_removed == HUBBARD_NIO_EXPECTED_ENERGY_COUNT
        );

        let mut data_like = original;
        let continuation = data_like
            .windows(3)
            .position(|window| window == b"\n.\n")
            .context("missing test descriptor continuation")?
            + 1;
        data_like[continuation] = b'1';
        anyhow::ensure!(sanitize_hubbard_nio_gg_descriptors(&data_like).is_err());
        Ok(())
    }

    #[test]
    fn hubbard_nio_provenance_is_atomic_idempotent_and_artifact_bound() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-hubbard-provenance-test")?;
        for (name, bytes) in [
            ("phase.bin", b"phase".as_slice()),
            ("gg.bin", b"sanitized gg".as_slice()),
            ("fms.bin", b"fms".as_slice()),
            ("gtr.dat", b"gtr".as_slice()),
            ("xmu.dat", b"xmu".as_slice()),
        ] {
            std::fs::write(root.join(name), bytes)?;
        }
        let provenance = HubbardMkgtrProvenance {
            schema_version: 1,
            generator: "xtask generate-golden Hubbard NiO descriptor repair",
            native_commit: "0".repeat(40),
            normalization_operation: HUBBARD_NIO_GG_NORMALIZATION,
            phase_sha256: manifest::sha256_hex(b"phase"),
            original_gg_sha256: "1".repeat(64),
            sanitized_gg_sha256: manifest::sha256_hex(b"sanitized gg"),
            descriptor_records: HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
            continuation_lines_removed: HUBBARD_NIO_EXPECTED_ENERGY_COUNT,
            mkgtr_executable: "bin/Seq/mkgtr",
            ff2x_executable: "bin/Seq/ff2x",
            fms_sha256: manifest::sha256_hex(b"fms"),
            gtr_sha256: manifest::sha256_hex(b"gtr"),
            xmu_sha256: manifest::sha256_hex(b"xmu"),
        };

        write_hubbard_mkgtr_provenance(&root, &provenance)?;
        write_hubbard_mkgtr_provenance(&root, &provenance)?;
        anyhow::ensure!(root.join(HUBBARD_NIO_PROVENANCE_FILE).is_file());
        anyhow::ensure!(
            !std::fs::read_dir(&root)?.any(|entry| entry.is_ok_and(|entry| entry
                .file_name()
                .to_string_lossy()
                .contains(".hubbard-mkgtr-provenance.tmp-")))
        );

        std::fs::write(root.join("xmu.dat"), b"changed")?;
        anyhow::ensure!(validate_hubbard_mkgtr_provenance(&root, &provenance).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn hubbard_fms_cache_contract_rejects_every_incoherent_boundary() -> Result<()> {
        validate_hubbard_fms_cache_contract(83, 83, 83, 83, true)?;
        for invalid in [
            (0, 0, 0, 0, true),
            (83, 82, 83, 83, true),
            (83, 83, 82, 83, true),
            (83, 83, 83, 82, true),
            (83, 83, 83, 83, false),
        ] {
            anyhow::ensure!(
                validate_hubbard_fms_cache_contract(
                    invalid.0, invalid.1, invalid.2, invalid.3, invalid.4
                )
                .is_err(),
                "incoherent Hubbard cache contract unexpectedly passed: {invalid:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn danes_accepts_only_terminal_errorfile_cleanup_exit() {
        let completed = ReferenceOutput {
            success: false,
            status: "exit status: 2".to_string(),
            stdout: b"Done with module: XAS spectra (FF2X: DW + final sum over paths).\n".to_vec(),
            stderr: b"At line 42 of file m_errorfile.f90 (unit = 77, file = '.feff.error')\nFortran runtime error: File cannot be deleted\n".to_vec(),
        };
        assert!(is_expected_danes_errorfile_cleanup_exit(
            Path::new("DANES/GeCl_4"),
            &completed
        ));

        let mut incomplete = ReferenceOutput {
            success: false,
            status: completed.status.clone(),
            stdout: Vec::new(),
            stderr: completed.stderr.clone(),
        };
        assert!(!is_expected_danes_errorfile_cleanup_exit(
            Path::new("DANES/GeCl_4"),
            &incomplete
        ));
        incomplete.stdout = completed.stdout.clone();
        assert!(!is_expected_danes_errorfile_cleanup_exit(
            Path::new("DANES/BN"),
            &incomplete
        ));
    }

    #[cfg(unix)]
    #[test]
    fn standalone_rdinp_capture_writes_stdout_and_stderr() -> Result<()> {
        use std::os::unix::fs::PermissionsExt as _;

        let root = temporary_work_dir("refeff-xtask-rdinp-capture-test")?;
        let bin = root.join("reference/bin/Seq");
        let case_dir = root.join("case");
        std::fs::create_dir_all(&bin)?;
        std::fs::create_dir_all(&case_dir)?;
        let driver = bin.join("rdinp");
        std::fs::write(
            &driver,
            "#!/bin/sh\nprintf 'pinned rdinp output\\n'\nprintf 'pinned rdinp warning\\n' >&2\n",
        )?;
        let mut permissions = std::fs::metadata(&driver)?.permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&driver, permissions)?;

        anyhow::ensure!(capture_standalone_rdinp_if_available(
            &root.join("reference"),
            &case_dir
        )?);
        anyhow::ensure!(
            std::fs::read_to_string(case_dir.join("rdinp.stdout"))? == "pinned rdinp output\n"
        );
        anyhow::ensure!(
            std::fs::read_to_string(case_dir.join("rdinp.stderr"))? == "pinned rdinp warning\n"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn invalid_danes_gecl4_outputs_use_validated_archive_fallback() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-gecl-fallback-test")?;
        std::fs::write(root.join("phase.bin"), "invalid phase\n")?;
        std::fs::write(root.join("emesh.bin"), b"invalid emesh")?;
        std::fs::write(root.join("xmu.dat"), "1 2 3 NaN 5 6\n")?;
        std::fs::write(root.join("danes.dat"), "1 2 3 4 5 6 NaN\n")?;
        std::fs::write(root.join("xsect.dat"), "invalid xsect\n")?;
        std::fs::write(root.join("fms.bin"), "invalid fms\n")?;

        let archive_file = std::fs::File::create(root.join("REFERENCE.zip"))?;
        let mut archive = zip::ZipWriter::new(archive_file);
        let options = zip::write::SimpleFileOptions::default();
        archive.start_file("REFERENCE/xmu.dat", options)?;
        archive.write_all(b"1 2 3 4 5 6\n")?;
        archive.start_file("REFERENCE/danes.dat", options)?;
        archive.write_all(b"1 2 3 4 5 6 7\n")?;
        archive.start_file("REFERENCE/xsect.dat", options)?;
        archive.write_all(b"# Cu crystal\n#  -----------------------------------------------------------------------\n#   8.50000E-01  1.50000E-01  2.40000E+00  9.1000000E+00 -4.0000000E-01 method to calculate xsect\n#   1.2300000E+00      2      1 gamach in eV, # of points on horizontal axis\n#       em              xsnorm            xsec  \n  1.250000000E+00  1.00000E-02  2.00000E+00  3.00000E+00 -4.00000E-01\n  1.500000000E+00  2.00000E-02  2.50000E+00  3.50000E+00 -5.00000E-01\n")?;
        archive.start_file("REFERENCE/fms.bin", options)?;
        archive.write_all(b"FMS rfms=-1.0000\n   3   2   0   1   8   0\n")?;
        archive.finish()?;

        anyhow::ensure!(quarantine_invalid_danes_gecl4_outputs(
            Path::new("DANES/GeCl_4"),
            &root
        )?);
        for name in [
            "phase.bin",
            "emesh.bin",
            "xmu.dat",
            "danes.dat",
            "xsect.dat",
            "fms.bin",
        ] {
            anyhow::ensure!(!root.join(name).exists(), "{name} was not quarantined");
        }
        let marker: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            root.join(".reference-fallback.json"),
        )?)?;
        anyhow::ensure!(marker["schema_version"] == 1);
        anyhow::ensure!(
            marker["archive_members"]
                == serde_json::json!([
                    "REFERENCE/xmu.dat",
                    "REFERENCE/danes.dat",
                    "REFERENCE/xsect.dat",
                    "REFERENCE/fms.bin"
                ])
        );
        anyhow::ensure!(
            marker["quarantined"]
                .as_array()
                .is_some_and(|items| items.len() == 6)
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn invalid_debye_fecn6_xmu_uses_matching_archive_reference() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-fecn-fallback-test")?;
        std::fs::write(
            root.join("feff.stdout"),
            "Error: Unexpected end of record while reading from gg.bin.\n",
        )?;
        std::fs::write(root.join("xmu.dat"), "1 2 3 4 5 0\n")?;
        let pinned_xmu = b"1 2 3 4 5 0.25\n";
        std::fs::write(root.join("referencexmu.dat"), pinned_xmu)?;

        let archive_file = std::fs::File::create(root.join("REFERENCE.zip"))?;
        let mut archive = zip::ZipWriter::new(archive_file);
        archive.start_file(
            "REFERENCE/xmu.dat",
            zip::write::SimpleFileOptions::default(),
        )?;
        archive.write_all(pinned_xmu)?;
        archive.finish()?;

        anyhow::ensure!(quarantine_invalid_debye_fecn6_outputs(
            Path::new("DEBYE/DM/XANES/FeCN_6"),
            &root
        )?);
        anyhow::ensure!(!root.join("xmu.dat").exists());
        anyhow::ensure!(root.join("referencexmu.dat").exists());
        let marker: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            root.join(".reference-fallback.json"),
        )?)?;
        anyhow::ensure!(marker["archive_members"] == serde_json::json!(["REFERENCE/xmu.dat"]));
        anyhow::ensure!(marker["quarantined"][0]["path"] == "xmu.dat");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn archive_fallback_rejects_declared_oversized_member() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-archive-size-test")?;
        let archive_path = root.join("REFERENCE.zip");
        let archive_file = std::fs::File::create(&archive_path)?;
        let mut archive = zip::ZipWriter::new(archive_file);
        archive.start_file(
            "REFERENCE/xmu.dat",
            zip::write::SimpleFileOptions::default(),
        )?;
        archive.write_all(&[b'x'; 17])?;
        archive.finish()?;

        let error = read_exact_archive_member_with_limit(&archive_path, "REFERENCE/xmu.dat", 16)
            .expect_err("oversized declared member must be rejected");
        anyhow::ensure!(
            error.to_string().contains("declares 17")
                && error.to_string().contains("16-byte limit"),
            "unexpected oversized-member error: {error:#}"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn archive_fallback_rejects_member_with_lying_declared_size() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-archive-lying-size-test")?;
        let archive_path = root.join("REFERENCE.zip");
        let archive_file = std::fs::File::create(&archive_path)?;
        let mut archive = zip::ZipWriter::new(archive_file);
        archive.start_file(
            "REFERENCE/xmu.dat",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated),
        )?;
        archive.write_all(&[b'x'; 64])?;
        archive.finish()?;

        let mut bytes = std::fs::read(&archive_path)?;
        let central_header = bytes
            .windows(4)
            .position(|window| window == b"PK\x01\x02")
            .context("test archive has no central directory header")?;
        bytes[central_header + 24..central_header + 28].copy_from_slice(&1_u32.to_le_bytes());
        std::fs::write(&archive_path, bytes)?;

        read_exact_archive_member_with_limit(&archive_path, "REFERENCE/xmu.dat", 16)
            .expect_err("member that lies about decompressed size must be rejected");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn archive_fallback_rejects_duplicate_exact_member() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-archive-duplicate-test")?;
        let archive_path = root.join("REFERENCE.zip");
        let archive_file = std::fs::File::create(&archive_path)?;
        let mut archive = zip::ZipWriter::new(archive_file);
        for (name, payload) in [
            ("REFERENCE/xmu.dat", b"first".as_slice()),
            ("REFERENCE/xmv.dat", b"second".as_slice()),
        ] {
            archive.start_file(name, zip::write::SimpleFileOptions::default())?;
            archive.write_all(payload)?;
        }
        archive.finish()?;
        let mut bytes = std::fs::read(&archive_path)?;
        let alias = b"REFERENCE/xmv.dat";
        let exact = b"REFERENCE/xmu.dat";
        let matches = bytes
            .windows(alias.len())
            .enumerate()
            .filter_map(|(index, window)| (window == alias).then_some(index))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            matches.len() == 2,
            "test archive alias appeared {} time(s), expected local and central names",
            matches.len()
        );
        for index in matches {
            bytes[index..index + exact.len()].copy_from_slice(exact);
        }
        std::fs::write(&archive_path, bytes)?;

        let error = read_exact_archive_member(&archive_path, "REFERENCE/xmu.dat")
            .expect_err("duplicate exact archive member must be rejected");
        anyhow::ensure!(
            error
                .to_string()
                .contains("exactly one REFERENCE/xmu.dat, found 2"),
            "unexpected duplicate-member error: {error:#}"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn archive_fallback_rejects_non_normalized_requested_member() -> Result<()> {
        let error = read_exact_archive_member_with_limit(
            Path::new("unused.zip"),
            "REFERENCE/../xmu.dat",
            16,
        )
        .expect_err("non-normalized archive member must be rejected before opening the archive");
        anyhow::ensure!(
            error
                .to_string()
                .contains("not an exact normalized relative path"),
            "unexpected non-normalized-member error: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn fallback_marker_is_published_before_any_quarantine_failure() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-marker-first-test")?;
        std::fs::create_dir(root.join("artifact.dat"))?;
        let provenance = ReferenceFallbackProvenance {
            schema_version: 1,
            reason: "test fallback",
            archive: "REFERENCE.zip",
            archive_sha256: "test-sha256".to_string(),
            archive_members: vec!["REFERENCE/artifact.dat"],
            quarantined: vec![QuarantinedReferenceArtifact {
                path: "artifact.dat",
                sha256: None,
                validation_error: "test invalid artifact".to_string(),
            }],
        };

        let error = quarantine_reference_artifacts(&root, &provenance)
            .expect_err("removing a directory as a file must fail");
        anyhow::ensure!(
            error.to_string().contains("failed to quarantine"),
            "unexpected quarantine error: {error:#}"
        );
        anyhow::ensure!(
            root.join(".reference-fallback.json").is_file(),
            "authorization marker was not published before quarantine failed"
        );
        anyhow::ensure!(
            root.join("artifact.dat").is_dir(),
            "failed quarantine unexpectedly removed the artifact directory"
        );
        anyhow::ensure!(
            !std::fs::read_dir(&root)?.any(|entry| entry
                .is_ok_and(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))),
            "atomic marker left a temporary file behind"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn rust_rdinp_bench_runner_writes_outputs() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-rdinp-test")?;
        let input_dir = root.join("input");
        std::fs::create_dir_all(&input_dir)?;
        let input = input_dir.join("feff.inp");
        std::fs::write(
            &input,
            r#"
TITLE Cu smoke test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu absorber
2.0 0.0 0.0 1 Cu shell
"#,
        )?;

        let run = run_rust_rdinp_to_dir(&input, &root.join("out"))?;
        anyhow::ensure!(run.output_files > 0, "rdinp benchmark wrote no files");
        anyhow::ensure!(run.output_bytes > 0, "rdinp benchmark wrote no bytes");
        anyhow::ensure!(
            root.join("out/atoms.dat").exists(),
            "rdinp benchmark did not write atoms.dat"
        );
        anyhow::ensure!(
            !root.join("out/.feff.error").exists(),
            "rdinp benchmark left a stale FEFF error sentinel"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn rust_rdinp_bench_runner_writes_error_sentinel() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-rdinp-error-test")?;
        let input_dir = root.join("input");
        std::fs::create_dir_all(&input_dir)?;
        let input = input_dir.join("feff.inp");
        std::fs::write(&input, "HOLE\nEND\n")?;

        let run = run_rust_rdinp_to_dir(&input, &root.join("out"))?;

        anyhow::ensure!(run.output_files == 2, "unexpected output count");
        anyhow::ensure!(
            std::fs::read_to_string(root.join("out/.feff.error"))?
                == rdinp::rdinp_error_sentinel_string(),
            "rdinp benchmark did not write the FEFF error sentinel"
        );
        anyhow::ensure!(
            root.join("out/log.dat").exists(),
            "rdinp benchmark did not write log.dat"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn reference_tests_skip_expected_highz_template() -> Result<()> {
        let root = temporary_work_dir("refeff-xtask-reference-test")?;
        let valid_dir = root.join("examples/EXAFS/Cu");
        std::fs::create_dir_all(&valid_dir)?;
        std::fs::write(
            valid_dir.join("feff.inp"),
            r#"
TITLE Cu smoke test
EDGE K
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu absorber
2.0 0.0 0.0 1 Cu shell
END
"#,
        )?;

        let highz_dir = root.join("examples/HIGHZ");
        std::fs::create_dir_all(&highz_dir)?;
        std::fs::write(
            highz_dir.join("feff.inp"),
            r#"
TITLE test_element
NOHOLE
HIGHZ
POTENTIALS
0 XXX Te
1 XXX Te
ATOMS
0.0 0.0 0.0 0 Te0
0.0 0.0 2.0 1 Te1
END
"#,
        )?;

        let result = run_reference_tests(Some(root.clone()));
        std::fs::remove_dir_all(root)?;
        result
    }

    #[test]
    fn release_readiness_command_parses_matrix_filters() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "release-readiness",
            "--detail",
            "--open-only",
            "--module",
            "xsph",
            "--module",
            "ldos",
            "--row",
            "xsph.tdlda-pmbse",
        ])?;

        match xtask.command {
            Command::ReleaseReadiness {
                detail,
                open_only,
                modules,
                rows,
                port_json_out,
                compatibility_json_out,
                json_out,
            } => {
                assert!(detail);
                assert!(open_only);
                assert_eq!(modules, vec!["xsph".to_string(), "ldos".to_string()]);
                assert_eq!(rows, vec!["xsph.tdlda-pmbse".to_string()]);
                assert_eq!(port_json_out, None);
                assert_eq!(compatibility_json_out, None);
                assert_eq!(json_out, None);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn port_status_command_parses_json_output_path() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "port-status",
            "--json-out",
            "target/port-status.json",
        ])?;

        match xtask.command {
            Command::PortStatus {
                json_out,
                cli_src,
                fail_on_unported,
                fail_on_ignored_parity,
                fail_on_guarded_branches,
                detail,
            } => {
                assert_eq!(json_out, Some(PathBuf::from("target/port-status.json")));
                assert_eq!(cli_src, None);
                assert!(!fail_on_unported);
                assert!(!fail_on_ignored_parity);
                assert!(!fail_on_guarded_branches);
                assert!(!detail);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn compatibility_matrix_command_parses_json_output_path() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "compatibility-matrix",
            "--json-out",
            "target/compatibility-matrix.json",
        ])?;

        match xtask.command {
            Command::CompatibilityMatrix {
                json_out,
                detail,
                fail_on_open,
                fail_on_missing_fixtures,
                fail_on_stale_fixtures,
                open_only,
                modules,
                rows,
            } => {
                assert_eq!(
                    json_out,
                    Some(PathBuf::from("target/compatibility-matrix.json"))
                );
                assert!(!detail);
                assert!(!fail_on_open);
                assert!(!fail_on_missing_fixtures);
                assert!(!fail_on_stale_fixtures);
                assert!(!open_only);
                assert!(modules.is_empty());
                assert!(rows.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn compatibility_matrix_command_parses_fail_on_stale_fixtures() -> Result<()> {
        let xtask =
            Xtask::try_parse_from(["xtask", "compatibility-matrix", "--fail-on-stale-fixtures"])?;

        match xtask.command {
            Command::CompatibilityMatrix {
                fail_on_stale_fixtures,
                fail_on_missing_fixtures,
                ..
            } => {
                assert!(fail_on_stale_fixtures);
                assert!(!fail_on_missing_fixtures);
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn parity_command_parses_example_and_json_out() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "parity",
            "--example",
            "XANES/BN",
            "--json-out",
            "target/parity.json",
        ])?;

        match xtask.command {
            Command::Parity { example, json_out } => {
                assert_eq!(example, "XANES/BN");
                assert_eq!(json_out, Some(PathBuf::from("target/parity.json")));
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn release_readiness_command_parses_compatibility_json_output_path() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "release-readiness",
            "--compatibility-json-out",
            "target/release-readiness-compatibility.json",
        ])?;

        match xtask.command {
            Command::ReleaseReadiness {
                port_json_out,
                compatibility_json_out,
                json_out,
                detail,
                open_only,
                modules,
                rows,
            } => {
                assert_eq!(port_json_out, None);
                assert_eq!(json_out, None);
                assert_eq!(
                    compatibility_json_out,
                    Some(PathBuf::from("target/release-readiness-compatibility.json"))
                );
                assert!(!detail);
                assert!(!open_only);
                assert!(modules.is_empty());
                assert!(rows.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn release_readiness_command_parses_port_json_output_path() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "release-readiness",
            "--port-json-out",
            "target/release-readiness-port-status.json",
        ])?;

        match xtask.command {
            Command::ReleaseReadiness {
                port_json_out,
                compatibility_json_out,
                json_out,
                detail,
                open_only,
                modules,
                rows,
            } => {
                assert_eq!(
                    port_json_out,
                    Some(PathBuf::from("target/release-readiness-port-status.json"))
                );
                assert_eq!(compatibility_json_out, None);
                assert_eq!(json_out, None);
                assert!(!detail);
                assert!(!open_only);
                assert!(modules.is_empty());
                assert!(rows.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn release_readiness_command_parses_json_output_path() -> Result<()> {
        let xtask = Xtask::try_parse_from([
            "xtask",
            "release-readiness",
            "--json-out",
            "target/release-readiness.json",
        ])?;

        match xtask.command {
            Command::ReleaseReadiness {
                json_out,
                port_json_out,
                compatibility_json_out,
                detail,
                open_only,
                modules,
                rows,
            } => {
                assert_eq!(
                    json_out,
                    Some(PathBuf::from("target/release-readiness.json"))
                );
                assert_eq!(port_json_out, None);
                assert_eq!(compatibility_json_out, None);
                assert!(!detail);
                assert!(!open_only);
                assert!(modules.is_empty());
                assert!(rows.is_empty());
            }
            other => panic!("unexpected command: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn release_readiness_json_report_lists_selected_open_items_and_errors() {
        let modules = vec!["xsph".to_string()];
        let rows = vec!["xsph.tdlda-pmbse".to_string()];
        let scope_status = Ok(());
        let port_status = Ok(());
        let compatibility = Err(anyhow::anyhow!("matrix failed"));
        let port_json_out = Path::new("target/port.json");
        let compatibility_json_out = Path::new("target/compat.json");

        let json = release_readiness_json_report(&ReleaseReadinessJsonReport {
            detail: true,
            open_only: true,
            modules: &modules,
            rows: &rows,
            port_json_out: Some(port_json_out),
            compatibility_json_out: Some(compatibility_json_out),
            scope_status: &scope_status,
            port_status: &port_status,
            compatibility: &compatibility,
        })
        .expect("release readiness json should serialize");
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("release readiness json should parse");

        assert_eq!(value["passed"], false);
        assert_eq!(value["production_scope"]["passed"], true);
        assert_eq!(value["filters"]["modules"], serde_json::json!(["xsph"]));
        assert_eq!(
            value["filters"]["rows"],
            serde_json::json!(["xsph.tdlda-pmbse"])
        );
        assert_eq!(value["port_status"]["artifact"], "target/port.json");
        assert_eq!(
            value["compatibility_matrix"]["artifact"],
            "target/compat.json"
        );
        assert_eq!(value["compatibility_matrix"]["error"], "matrix failed");
        let open_items = value["open_compatibility_items"]
            .as_array()
            .expect("open_compatibility_items should be an array");
        assert!(open_items.is_empty());
    }

    #[test]
    fn write_release_readiness_json_report_creates_parent_directory() -> Result<()> {
        let root = temporary_work_dir("refeff-release-readiness-json-test")?;
        let path = root.join("nested/release-readiness.json");
        let modules = Vec::new();
        let rows = Vec::new();
        let scope_status = Ok(());
        let port_status = Ok(());
        let compatibility = Ok(());

        write_release_readiness_json_report(
            &path,
            &ReleaseReadinessJsonReport {
                detail: false,
                open_only: false,
                modules: &modules,
                rows: &rows,
                port_json_out: None,
                compatibility_json_out: None,
                scope_status: &scope_status,
                port_status: &port_status,
                compatibility: &compatibility,
            },
        )?;

        let json = std::fs::read_to_string(&path)?;
        assert!(json.contains("\"passed\": true"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
