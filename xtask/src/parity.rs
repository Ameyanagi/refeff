//! `cargo xtask parity --example XANES/BN` (F1): the parity front door.
//!
//! Copies a golden fixture's `feff.inp` into a scratch directory under
//! `target/`, runs the Rust `refeff` pipeline against it, then diffs every
//! file the run produced against the golden fixture tree with a generic,
//! Fortran-float-aware text differ (falling back to a byte comparison for
//! `.bin` files), printing a per-file max-abs/max-rel/RMS/first-divergence
//! table plus an optional JSON artifact. Parity evidence previously lived
//! only inside individual test modules, where a first failing assert
//! aborted the whole test with no overall picture.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

const GOLDEN_ROOT: &str = "reference-work/golden";
const DEFAULT_REL_TOLERANCE: f64 = 1e-6;
const DEFAULT_ABS_TOLERANCE: f64 = 1e-12;
/// Golden-directory files that are driver logs or packaging artifacts, not
/// fixture output to diff against.
const IGNORED_GOLDEN_FILE_NAMES: &[&str] = &["manifest.json", "REFERENCE.zip"];
const IGNORED_GOLDEN_FILE_SUFFIXES: &[&str] = &[".stdout", ".stderr"];

/// Run `xtask parity --example <example>`.
pub(crate) fn run_parity(example: &str, json_out: Option<&Path>) -> Result<()> {
    let golden_dir = PathBuf::from(GOLDEN_ROOT).join(example);
    anyhow::ensure!(
        golden_dir.is_dir(),
        "no golden fixture tree found at {} (run `cargo xtask generate-golden --example {example}` \
         against a feff10/ checkout first, or check that --example matches a \
         reference-work/golden/<category>/<name> directory)",
        golden_dir.display()
    );
    let golden_input = golden_dir.join("feff.inp");
    anyhow::ensure!(
        golden_input.is_file(),
        "{} has no feff.inp; it is not a runnable golden case",
        golden_dir.display()
    );

    build_refeff_binary()?;

    let scratch_dir = scratch_dir_for(example);
    if scratch_dir.exists() {
        std::fs::remove_dir_all(&scratch_dir).with_context(|| {
            format!(
                "failed to clear stale parity scratch dir {}",
                scratch_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(&scratch_dir)
        .with_context(|| format!("failed to create {}", scratch_dir.display()))?;
    let scratch_input = scratch_dir.join("feff.inp");
    std::fs::copy(&golden_input, &scratch_input)
        .with_context(|| format!("failed to copy {} into scratch dir", golden_input.display()))?;
    println!(
        "parity: running refeff against {} in {}",
        golden_input.display(),
        scratch_dir.display()
    );

    let run_output = run_refeff(&scratch_input, &scratch_dir)?;
    if !run_output.status.success() {
        println!(
            "warning: refeff run exited with {} (comparing whatever it did produce)",
            run_output.status
        );
        let stderr = String::from_utf8_lossy(&run_output.stderr);
        if !stderr.trim().is_empty() {
            println!("{stderr}");
        }
    }

    let comparisons = compare_against_golden(&golden_dir, &scratch_dir)?;
    print_comparison_table(example, &comparisons);

    if let Some(json_out) = json_out {
        write_json_report(json_out, example, &golden_dir, &scratch_dir, &comparisons)?;
        println!("wrote parity json: {}", json_out.display());
    }

    let failing: Vec<&str> = comparisons
        .iter()
        .filter(|comparison| !comparison.passed)
        .map(|comparison| comparison.name.as_str())
        .collect();
    anyhow::ensure!(
        failing.is_empty(),
        "{}/{} compared file(s) diverged from golden {}: {}",
        failing.len(),
        comparisons.len(),
        golden_dir.display(),
        failing.join(", ")
    );
    Ok(())
}

fn scratch_dir_for(example: &str) -> PathBuf {
    let sanitized = example.replace(['/', '\\'], "_");
    PathBuf::from("target/xtask-parity").join(sanitized)
}

fn build_refeff_binary() -> Result<()> {
    let status = std::process::Command::new("cargo")
        .args(["build", "-p", "refeff-cli", "--bin", "refeff"])
        .status()
        .context("failed to invoke `cargo build -p refeff-cli --bin refeff`")?;
    anyhow::ensure!(
        status.success(),
        "`cargo build -p refeff-cli --bin refeff` failed"
    );
    Ok(())
}

fn run_refeff(input: &Path, output_dir: &Path) -> Result<std::process::Output> {
    std::process::Command::new("cargo")
        .args(["run", "-q", "-p", "refeff-cli", "--bin", "refeff", "--"])
        .arg("run")
        .arg("-i")
        .arg(input)
        .arg("-o")
        .arg(output_dir)
        .output()
        .context("failed to invoke `cargo run -p refeff-cli --bin refeff -- run`")
}

/// One row of the parity comparison table.
#[derive(Debug, Clone, Serialize)]
struct FileComparison {
    /// The produced or golden-only file name being reported on.
    name: String,
    kind: ComparisonKind,
    max_abs: Option<f64>,
    max_rel: Option<f64>,
    rms: Option<f64>,
    first_divergence: Option<String>,
    passed: bool,
    detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ComparisonKind {
    Numeric,
    Binary,
    /// A file `refeff run` produced with no corresponding golden reference
    /// file to diff it against; not counted as a pass or a fail.
    NoGoldenReference,
    /// A golden reference file with no corresponding produced file.
    MissingProduced,
}

fn compare_against_golden(golden_dir: &Path, scratch_dir: &Path) -> Result<Vec<FileComparison>> {
    let golden_files = comparable_golden_files(golden_dir)?;
    let mut golden_by_target: std::collections::BTreeMap<String, PathBuf> =
        std::collections::BTreeMap::new();
    for path in golden_files {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        golden_by_target.insert(golden_target_name(name), path);
    }

    let mut produced_files = Vec::new();
    collect_files(scratch_dir, &mut produced_files)?;
    produced_files.sort();

    let mut comparisons = Vec::new();
    let mut matched_targets = std::collections::BTreeSet::new();
    for produced in &produced_files {
        let rel = produced
            .strip_prefix(scratch_dir)
            .unwrap_or(produced)
            .to_string_lossy()
            .replace('\\', "/");
        let Some(name) = produced.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        match golden_by_target.get(name) {
            Some(golden_path) => {
                matched_targets.insert(name.to_string());
                comparisons.push(compare_files(&rel, golden_path, produced)?);
            }
            None => comparisons.push(FileComparison {
                name: rel,
                kind: ComparisonKind::NoGoldenReference,
                max_abs: None,
                max_rel: None,
                rms: None,
                first_divergence: None,
                passed: true,
                detail: "produced, no golden counterpart to compare against".to_string(),
            }),
        }
    }

    for (target, golden_path) in &golden_by_target {
        if matched_targets.contains(target) {
            continue;
        }
        comparisons.push(FileComparison {
            name: golden_relative_name(golden_dir, golden_path),
            kind: ComparisonKind::MissingProduced,
            max_abs: None,
            max_rel: None,
            rms: None,
            first_divergence: None,
            passed: false,
            detail: "golden reference exists, but refeff run did not produce it".to_string(),
        });
    }

    comparisons.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(comparisons)
}

fn golden_relative_name(golden_dir: &Path, golden_path: &Path) -> String {
    golden_path
        .strip_prefix(golden_dir)
        .unwrap_or(golden_path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Maps a golden file's own name to the produced file name it should be
/// diffed against: `reference_xmu.dat`/`referencexmu.dat` -> `xmu.dat`,
/// anything else compares against a produced file of the identical name
/// (e.g. `atoms.dat`, `geom.dat`, the RDINP-generated `*.inp` handoffs).
fn golden_target_name(golden_name: &str) -> String {
    golden_name
        .strip_prefix("reference_")
        .or_else(|| golden_name.strip_prefix("reference"))
        .filter(|rest| !rest.is_empty())
        .unwrap_or(golden_name)
        .to_string()
}

fn comparable_golden_files(golden_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files(golden_dir, &mut files)?;
    files.retain(|path| {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        !IGNORED_GOLDEN_FILE_NAMES.contains(&name)
            && !IGNORED_GOLDEN_FILE_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
    });
    Ok(files)
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn compare_files(name: &str, golden_path: &Path, produced_path: &Path) -> Result<FileComparison> {
    let is_binary = golden_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("bin"));
    if is_binary {
        return compare_binary_files(name, golden_path, produced_path);
    }
    compare_text_files(name, golden_path, produced_path)
}

fn compare_binary_files(
    name: &str,
    golden_path: &Path,
    produced_path: &Path,
) -> Result<FileComparison> {
    let golden_bytes = std::fs::read(golden_path)
        .with_context(|| format!("failed to read {}", golden_path.display()))?;
    let produced_bytes = std::fs::read(produced_path)
        .with_context(|| format!("failed to read {}", produced_path.display()))?;

    if golden_bytes == produced_bytes {
        return Ok(FileComparison {
            name: name.to_string(),
            kind: ComparisonKind::Binary,
            max_abs: None,
            max_rel: None,
            rms: None,
            first_divergence: None,
            passed: true,
            detail: format!("binary match ({} bytes)", golden_bytes.len()),
        });
    }

    let mut first_diff = None;
    let mut differing = 0_usize;
    for (offset, (golden_byte, produced_byte)) in
        golden_bytes.iter().zip(produced_bytes.iter()).enumerate()
    {
        if golden_byte != produced_byte {
            differing += 1;
            if first_diff.is_none() {
                first_diff = Some(offset);
            }
        }
    }
    let length_mismatch = golden_bytes.len() != produced_bytes.len();
    let detail = format!(
        "binary differs: {differing} byte(s) differ over the shared {} byte(s){}",
        golden_bytes.len().min(produced_bytes.len()),
        if length_mismatch {
            format!(
                " (length mismatch: golden {} vs produced {})",
                golden_bytes.len(),
                produced_bytes.len()
            )
        } else {
            String::new()
        }
    );
    Ok(FileComparison {
        name: name.to_string(),
        kind: ComparisonKind::Binary,
        max_abs: None,
        max_rel: None,
        rms: None,
        first_divergence: first_diff.map(|offset| format!("byte {offset}")),
        passed: false,
        detail,
    })
}

#[derive(Debug, Clone, Copy)]
struct Tolerance {
    rel: f64,
    abs: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            rel: DEFAULT_REL_TOLERANCE,
            abs: DEFAULT_ABS_TOLERANCE,
        }
    }
}

/// Per-format tolerance dispatch; every format currently uses the default
/// (rel 1e-6 / abs 1e-12). Kept as its own function so a format known to
/// need a looser or tighter tolerance can be added without touching the
/// differ itself.
fn tolerance_for(_name: &str) -> Tolerance {
    Tolerance::default()
}

fn compare_text_files(
    name: &str,
    golden_path: &Path,
    produced_path: &Path,
) -> Result<FileComparison> {
    let golden_text = std::fs::read_to_string(golden_path)
        .with_context(|| format!("failed to read {}", golden_path.display()))?;
    let produced_text = std::fs::read_to_string(produced_path)
        .with_context(|| format!("failed to read {}", produced_path.display()))?;
    let tolerance = tolerance_for(name);

    let golden_lines: Vec<&str> = golden_text.lines().collect();
    let produced_lines: Vec<&str> = produced_text.lines().collect();

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut numeric_count = 0_usize;
    let mut first_divergence: Option<String> = None;
    let mut any_mismatch = false;

    let line_count = golden_lines.len().max(produced_lines.len());
    for line_index in 0..line_count {
        let golden_line = golden_lines.get(line_index).copied().unwrap_or_default();
        let produced_line = produced_lines.get(line_index).copied().unwrap_or_default();
        let golden_tokens: Vec<&str> = golden_line.split_whitespace().collect();
        let produced_tokens: Vec<&str> = produced_line.split_whitespace().collect();
        let token_count = golden_tokens.len().max(produced_tokens.len());

        for token_index in 0..token_count {
            let golden_token = golden_tokens.get(token_index).copied();
            let produced_token = produced_tokens.get(token_index).copied();
            let (golden_token, produced_token) = match (golden_token, produced_token) {
                (Some(golden), Some(produced)) => (golden, produced),
                _ => {
                    any_mismatch = true;
                    if first_divergence.is_none() {
                        first_divergence = Some(format!(
                            "line {} token {}: token count mismatch ({:?} vs {:?})",
                            line_index + 1,
                            token_index + 1,
                            golden_token,
                            produced_token
                        ));
                    }
                    continue;
                }
            };
            if golden_token == produced_token {
                continue;
            }

            match (
                parse_fortran_float(golden_token),
                parse_fortran_float(produced_token),
            ) {
                (Some(golden_value), Some(produced_value)) => {
                    let diff = (golden_value - produced_value).abs();
                    let scale = golden_value.abs().max(produced_value.abs());
                    let rel = if scale > 0.0 { diff / scale } else { 0.0 };
                    max_abs = max_abs.max(diff);
                    max_rel = max_rel.max(rel);
                    sum_squared += diff * diff;
                    numeric_count += 1;
                    let threshold = tolerance.abs.max(tolerance.rel * scale);
                    if diff > threshold {
                        any_mismatch = true;
                        if first_divergence.is_none() {
                            first_divergence = Some(format!(
                                "line {} token {}: {golden_token} vs {produced_token} (abs diff {diff:e})",
                                line_index + 1,
                                token_index + 1
                            ));
                        }
                    }
                }
                _ => {
                    any_mismatch = true;
                    if first_divergence.is_none() {
                        first_divergence = Some(format!(
                            "line {} token {}: {golden_token} vs {produced_token} (non-numeric)",
                            line_index + 1,
                            token_index + 1
                        ));
                    }
                }
            }
        }
    }

    let rms = if numeric_count > 0 {
        Some((sum_squared / numeric_count as f64).sqrt())
    } else {
        None
    };
    let passed = !any_mismatch;
    let detail = if passed {
        format!("ok ({numeric_count} numeric token(s) compared)")
    } else {
        first_divergence
            .clone()
            .unwrap_or_else(|| "mismatch".to_string())
    };

    Ok(FileComparison {
        name: name.to_string(),
        kind: ComparisonKind::Numeric,
        max_abs: (numeric_count > 0).then_some(max_abs),
        max_rel: (numeric_count > 0).then_some(max_rel),
        rms,
        first_divergence,
        passed,
        detail,
    })
}

/// Parses a Fortran-formatted floating-point token, tolerating the `D`/`d`
/// double-precision exponent marker (`1.234D-05`) in addition to `E`/`e`.
fn parse_fortran_float(token: &str) -> Option<f64> {
    if token.is_empty() {
        return None;
    }
    if token.contains(['D', 'd']) {
        token.replace(['D', 'd'], "E").parse::<f64>().ok()
    } else {
        token.parse::<f64>().ok()
    }
}

fn print_comparison_table(example: &str, comparisons: &[FileComparison]) {
    println!("parity: {example}");
    println!(
        "{:<28} {:<8} {:<12} {:<12} {:<12} first-divergence / detail",
        "file", "status", "max-abs", "max-rel", "rms"
    );
    for comparison in comparisons {
        let status = match comparison.kind {
            ComparisonKind::NoGoldenReference => "SKIP",
            _ if comparison.passed => "PASS",
            _ => "FAIL",
        };
        println!(
            "{:<28} {:<8} {:<12} {:<12} {:<12} {}",
            comparison.name,
            status,
            format_metric(comparison.max_abs),
            format_metric(comparison.max_rel),
            format_metric(comparison.rms),
            comparison
                .first_divergence
                .as_deref()
                .unwrap_or(&comparison.detail)
        );
    }
    let passed = comparisons
        .iter()
        .filter(|comparison| comparison.kind != ComparisonKind::NoGoldenReference)
        .filter(|comparison| comparison.passed)
        .count();
    let compared = comparisons
        .iter()
        .filter(|comparison| comparison.kind != ComparisonKind::NoGoldenReference)
        .count();
    println!("parity summary: {passed}/{compared} compared file(s) passed");
}

fn format_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:e}"))
}

#[derive(Debug, Serialize)]
struct ParityJsonReport<'a> {
    example: &'a str,
    golden_dir: String,
    scratch_dir: String,
    default_tolerance: ToleranceJson,
    files: &'a [FileComparison],
}

#[derive(Debug, Serialize)]
struct ToleranceJson {
    rel: f64,
    abs: f64,
}

fn write_json_report(
    path: &Path,
    example: &str,
    golden_dir: &Path,
    scratch_dir: &Path,
    comparisons: &[FileComparison],
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let report = ParityJsonReport {
        example,
        golden_dir: golden_dir.display().to_string(),
        scratch_dir: scratch_dir.display().to_string(),
        default_tolerance: ToleranceJson {
            rel: DEFAULT_REL_TOLERANCE,
            abs: DEFAULT_ABS_TOLERANCE,
        },
        files: comparisons,
    };
    let json =
        serde_json::to_string_pretty(&report).context("failed to serialize parity report json")?;
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_target_name_strips_reference_prefixes() {
        assert_eq!(golden_target_name("referencexmu.dat"), "xmu.dat");
        assert_eq!(golden_target_name("reference_compton.dat"), "compton.dat");
        assert_eq!(golden_target_name("atoms.dat"), "atoms.dat");
        assert_eq!(golden_target_name("reference"), "reference");
    }

    #[test]
    fn parse_fortran_float_accepts_d_exponent() {
        assert_eq!(parse_fortran_float("1.5D+02"), Some(150.0));
        assert_eq!(parse_fortran_float("1.5d-02"), Some(0.015));
        assert_eq!(parse_fortran_float("1.5e+02"), Some(150.0));
        assert_eq!(parse_fortran_float("not-a-number"), None);
        assert_eq!(parse_fortran_float(""), None);
    }

    #[test]
    fn compare_text_files_passes_within_tolerance_and_fails_outside_it() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-text-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&root)?;
        let golden = root.join("golden.dat");
        let close = root.join("close.dat");
        let far = root.join("far.dat");
        std::fs::write(&golden, "label 1.000000D+00 2.000000D+00\n")?;
        std::fs::write(&close, "label 1.0000001D+00 2.000000D+00\n")?;
        std::fs::write(&far, "label 1.500000D+00 2.000000D+00\n")?;

        let close_comparison = compare_files("close.dat", &golden, &close)?;
        assert!(
            close_comparison.passed,
            "a tiny relative difference should pass: {close_comparison:?}"
        );

        let far_comparison = compare_files("far.dat", &golden, &far)?;
        assert!(
            !far_comparison.passed,
            "a large relative difference should fail: {far_comparison:?}"
        );
        assert!(far_comparison.first_divergence.is_some());

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn compare_binary_files_reports_byte_level_differences() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-binary-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root)?;
        let golden = root.join("golden.bin");
        let produced = root.join("produced.bin");
        std::fs::write(&golden, [1_u8, 2, 3, 4])?;
        std::fs::write(&produced, [1_u8, 2, 9, 4])?;

        let comparison = compare_files("produced.bin", &golden, &produced)?;
        assert_eq!(comparison.kind, ComparisonKind::Binary);
        assert!(!comparison.passed);
        assert_eq!(comparison.first_divergence.as_deref(), Some("byte 2"));

        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn run_parity_reports_a_clear_error_for_a_missing_golden_tree() {
        let error = run_parity("NO/SUCH/CASE", None).expect_err("missing golden tree should error");
        let message = format!("{error:#}");
        assert!(
            message.contains("no golden fixture tree found"),
            "unexpected error message: {message}"
        );
    }
}
