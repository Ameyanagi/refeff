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
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use refeff_io::{
    FeffDocument, FeffInput, FmsInput, XsphInput, fms_input_string, rdinp, xsph_input_string,
};
use serde::Serialize;

mod compatibility_matrix;
mod manifest;
mod parity;
mod port_status;
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
        .map(|rel| {
            let rel = rel.to_string_lossy();
            examples.iter().any(|pattern| rel.contains(pattern))
        })
        .unwrap_or(false)
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
        let rel_string = rel.to_string_lossy();
        if !examples.is_empty() && !examples.iter().any(|pattern| rel_string.contains(pattern)) {
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

        // HIGHZ is an upstream parameterized harness: its feff.inp contains
        // the literal `XXX`, and `runall` expands atomic numbers 1..=138.
        // Preserve the pinned upstream harness and checked reference report
        // as one provenance-tracked fixture instead of attempting to run the
        // invalid template once.
        if rel == Path::new("HIGHZ") {
            manifest::write_manifest(&dest, feff10_rev.as_deref(), &compiler)
                .with_context(|| format!("failed to write manifest.json in {}", dest.display()))?;
            println!("generated {} (parameterized HIGHZ harness)", dest.display());
            continue;
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
        if !output.success {
            anyhow::bail!(
                "{} reference failed for {} with status {}",
                program.log_prefix(),
                rel.display(),
                output.status
            );
        }
        if program == ReferenceProgram::Feff {
            generate_compatibility_reference_subcases(&ref_dir, rel, &dest)?;
        }
        manifest::write_manifest(&dest, feff10_rev.as_deref(), &compiler)
            .with_context(|| format!("failed to write manifest.json in {}", dest.display()))?;
        println!("generated {}", dest.display());
    }

    Ok(())
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
