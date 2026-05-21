#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use refeff_io::{FeffDocument, FeffInput, rdinp};

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Xtask {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    PortStatus {
        #[arg(long)]
        cli_src: Option<PathBuf>,
        #[arg(long)]
        fail_on_unported: bool,
    },
    ReferenceTests {
        #[arg(long)]
        ref_dir: Option<PathBuf>,
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

fn main() -> Result<()> {
    let xtask = Xtask::parse();
    match xtask.command {
        Command::PortStatus {
            cli_src,
            fail_on_unported,
        } => print_port_status(cli_src, fail_on_unported)?,
        Command::ReferenceTests { ref_dir } => run_reference_tests(ref_dir)?,
        Command::GenerateGolden {
            ref_dir,
            out_dir,
            example,
            no_build,
            force,
            program,
        } => generate_golden(ref_dir, &out_dir, &example, !no_build, force, program)?,
        Command::BenchE2e {
            ref_dir,
            example,
            iterations,
            reference,
        } => bench_e2e(ref_dir, &example, iterations, reference)?,
    }
    Ok(())
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

fn print_port_status(cli_src: Option<PathBuf>, fail_on_unported: bool) -> Result<()> {
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
    for entry in std::fs::read_dir(cli_src)
        .with_context(|| format!("failed to read {}", cli_src.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", cli_src.display()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }
        let Some(module) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if matches!(module, "lib" | "tests") {
            continue;
        }

        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        modules.push(module_status_from_source(module, &text));
    }

    modules.sort_by(|left, right| left.module.cmp(&right.module));
    Ok(PortStatusReport { modules })
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

    if build_reference {
        build_reference_feff(&ref_dir)?;
    }
    let driver = reference_driver(&ref_dir, program)?;

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

        let output = std::process::Command::new(&driver)
            .current_dir(&dest)
            .output()?;
        std::fs::write(
            dest.join(format!("{}.stdout", program.log_prefix())),
            &output.stdout,
        )?;
        std::fs::write(
            dest.join(format!("{}.stderr", program.log_prefix())),
            &output.stderr,
        )?;
        if !output.status.success() {
            anyhow::bail!(
                "{} reference failed for {} with status {}",
                program.log_prefix(),
                rel.display(),
                output.status
            );
        }
        println!("generated {}", dest.display());
    }

    Ok(())
}

fn build_reference_feff(ref_dir: &Path) -> Result<()> {
    let src = ref_dir.join("src");
    let mut command = std::process::Command::new("make");
    command.arg("all").current_dir(&src);
    if !command_exists("ifort") && command_exists("gfortran") {
        let flags = "-ffree-line-length-none -cpp -O3 -fallow-argument-mismatch";
        command
            .arg("F90=gfortran")
            .arg(format!("FLAGS={flags}"))
            .arg("MPIF90=gfortran")
            .arg(format!("MPIFLAGS={flags}"));
    }

    let status = command.status()?;
    if !status.success() {
        anyhow::bail!("failed to build FEFF reference in {}", src.display());
    }
    Ok(())
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
        std::fs::write(
            root.join("lib.rs"),
            "this workspace root module should not be counted\n",
        )?;
        std::fs::write(
            root.join("tests.rs"),
            "this external CLI test module should not be counted\n",
        )?;

        let report = port_status_report(&root)?;

        assert_eq!(report.modules.len(), 2);
        assert_eq!(report.unported_count(), 1);
        assert_eq!(report.reference_covered_unported_count(), 1);
        assert_eq!(report.modules[0].module, "atomic");
        assert_eq!(report.modules[1].module, "wpot");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
