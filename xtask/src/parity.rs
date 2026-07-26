//! `cargo xtask parity --example XANES/BN` (F1): the parity front door.
//!
//! Copies a golden fixture's `feff.inp` into a scratch directory under
//! `target/`, runs the Rust `refeff` pipeline against it, then diffs every
//! file the run produced against the golden fixture tree with a generic,
//! Fortran-float-aware text differ and typed format comparators. The complete
//! table remains diagnostic; the workflow's canonical primary output is the
//! release-blocking result. Parity evidence previously lived only inside
//! individual test modules, where a first failing assert aborted the whole
//! test with no overall picture.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use refeff_io::codec::{FeffCodec, FileFormat, NumericTolerance, Representation, identify_format};
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
    let required_targets = required_parity_targets(example);
    print_comparison_table(example, &comparisons, required_targets);

    if let Some(json_out) = json_out {
        write_json_report(json_out, example, &golden_dir, &scratch_dir, &comparisons)?;
        println!("wrote parity json: {}", json_out.display());
    }

    let failing: Vec<&str> = comparisons
        .iter()
        .filter(|comparison| {
            required_targets.contains(&comparison_file_name(&comparison.name)) && !comparison.passed
        })
        .map(|comparison| comparison.name.as_str())
        .collect();
    let present_required = required_targets
        .iter()
        .filter(|target| {
            comparisons
                .iter()
                .any(|comparison| comparison_file_name(&comparison.name) == **target)
        })
        .count();
    anyhow::ensure!(
        present_required == required_targets.len(),
        "only {present_required}/{} required primary output(s) were represented in the parity report",
        required_targets.len()
    );
    anyhow::ensure!(
        failing.is_empty(),
        "{}/{} required primary output(s) diverged from golden {}: {}",
        failing.len(),
        required_targets.len(),
        golden_dir.display(),
        failing.join(", ")
    );
    Ok(())
}

fn required_parity_targets(example: &str) -> &'static [&'static str] {
    let workflow = example
        .split(['/', '\\'])
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    match workflow.as_str() {
        "EXAFS" => &["chi.dat"],
        "EELS" | "ELNES" | "EXELFS" => &["eels.dat"],
        "COMPTON" => &["compton.dat"],
        "BAND" => &["bandstructure.dat"],
        "DMDW" | "DEBYE" => &["dmdw.out"],
        "RIXS" => &["rixsET.dat"],
        // XANES is the documented parity front door. FPRIME, DANES,
        // XES, and NRIXS also publish their final spectrum through xmu.dat.
        _ => &["xmu.dat"],
    }
}

fn comparison_file_name(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
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
    SemanticBinary,
    Binary,
    /// A file `refeff run` produced with no corresponding golden reference
    /// file to diff it against; not counted as a pass or a fail.
    NoGoldenReference,
    /// A golden reference file with no corresponding produced file.
    MissingProduced,
}

fn compare_against_golden(golden_dir: &Path, scratch_dir: &Path) -> Result<Vec<FileComparison>> {
    let golden_files = comparable_golden_files(golden_dir)?;
    let golden_by_target = golden_files_by_target(golden_files)?;

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

fn golden_files_by_target(
    golden_files: Vec<PathBuf>,
) -> Result<std::collections::BTreeMap<String, PathBuf>> {
    let mut by_target = std::collections::BTreeMap::new();
    for path in golden_files {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let target = golden_target_name(name);
        match by_target.get(&target) {
            None => {
                by_target.insert(target, path);
            }
            Some(previous) => {
                let previous_name = previous
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                let previous_is_canonical = previous_name == target;
                let current_is_canonical = name == target;
                match (previous_is_canonical, current_is_canonical) {
                    (false, true) => {
                        // A freshly generated FEFF output is stronger evidence
                        // than a legacy `reference*` file shipped with an
                        // example. Prefer it regardless of directory order.
                        by_target.insert(target, path);
                    }
                    (true, false) => {}
                    _ => anyhow::bail!(
                        "ambiguous golden files {} and {} both target produced file {}",
                        previous.display(),
                        path.display(),
                        target
                    ),
                }
            }
        }
    }
    Ok(by_target)
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
    for entry in std::fs::read_dir(golden_dir)
        .with_context(|| format!("failed to read {}", golden_dir.display()))?
    {
        let path = entry
            .with_context(|| format!("failed to read {}", golden_dir.display()))?
            .path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
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
    let descriptor = identify_format(golden_path).or_else(|| identify_format(name));
    if let Some(descriptor) =
        descriptor.filter(|descriptor| descriptor.format == FileFormat::XmuDat)
    {
        return compare_xmu_files(name, golden_path, produced_path, descriptor.tolerance);
    }
    if let Some(descriptor) =
        descriptor.filter(|descriptor| descriptor.representation == Representation::Binary)
    {
        if supports_semantic_binary_comparison(descriptor.format) {
            return compare_semantic_binary_files(
                name,
                golden_path,
                produced_path,
                descriptor.format,
                descriptor.tolerance,
            );
        }
        return compare_binary_files(name, golden_path, produced_path);
    }
    if descriptor.is_none()
        && golden_path
            .extension()
            .is_some_and(|extension| extension == "bin")
    {
        return compare_binary_files(name, golden_path, produced_path);
    }
    compare_text_files(name, golden_path, produced_path)
}

fn compare_xmu_files(
    name: &str,
    golden_path: &Path,
    produced_path: &Path,
    tolerance: NumericTolerance,
) -> Result<FileComparison> {
    let golden_bytes = std::fs::read(golden_path)
        .with_context(|| format!("failed to read {}", golden_path.display()))?;
    let produced_bytes = std::fs::read(produced_path)
        .with_context(|| format!("failed to read {}", produced_path.display()))?;
    let golden = <refeff_io::XmuDatData as FeffCodec>::decode(golden_path, &golden_bytes)
        .with_context(|| format!("failed to decode golden {}", golden_path.display()))?;
    let produced = <refeff_io::XmuDatData as FeffCodec>::decode(produced_path, &produced_bytes)
        .with_context(|| format!("failed to decode produced {}", produced_path.display()))?;

    compare_named_numeric_fields(
        name,
        &[
            (
                "photon-energy",
                golden.photon_energy_ev.as_slice().unwrap_or(&[]),
                produced.photon_energy_ev.as_slice().unwrap_or(&[]),
            ),
            (
                "relative-energy",
                golden.relative_energy_ev.as_slice().unwrap_or(&[]),
                produced.relative_energy_ev.as_slice().unwrap_or(&[]),
            ),
            (
                "wave-number",
                golden.wave_number.as_slice().unwrap_or(&[]),
                produced.wave_number.as_slice().unwrap_or(&[]),
            ),
            (
                "mu",
                golden.mu.as_slice().unwrap_or(&[]),
                produced.mu.as_slice().unwrap_or(&[]),
            ),
            (
                "mu0",
                golden.mu0.as_slice().unwrap_or(&[]),
                produced.mu0.as_slice().unwrap_or(&[]),
            ),
            (
                "chi",
                golden.chi.as_slice().unwrap_or(&[]),
                produced.chi.as_slice().unwrap_or(&[]),
            ),
        ],
        tolerance,
    )
}

fn compare_named_numeric_fields(
    name: &str,
    fields: &[(&str, &[f64], &[f64])],
    tolerance: NumericTolerance,
) -> Result<FileComparison> {
    let mut max_abs = 0.0_f64;
    let mut max_relative_l2 = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut value_count = 0_usize;
    let mut first_divergence = None;

    for &(field_name, golden, produced) in fields {
        if golden.len() != produced.len() {
            first_divergence.get_or_insert_with(|| {
                format!(
                    "{field_name} length differs: golden {} vs produced {}",
                    golden.len(),
                    produced.len()
                )
            });
            continue;
        }

        let mut field_diff_squared = 0.0_f64;
        let mut golden_squared = 0.0_f64;
        let mut produced_squared = 0.0_f64;
        for (&golden_value, &produced_value) in golden.iter().zip(produced) {
            let diff = golden_value - produced_value;
            max_abs = max_abs.max(diff.abs());
            field_diff_squared += diff * diff;
            golden_squared += golden_value * golden_value;
            produced_squared += produced_value * produced_value;
        }
        sum_squared += field_diff_squared;
        value_count += golden.len();

        let field_l2 = field_diff_squared.sqrt();
        let scale_l2 = golden_squared.sqrt().max(produced_squared.sqrt());
        let relative_l2 = if scale_l2 > 0.0 {
            field_l2 / scale_l2
        } else {
            0.0
        };
        max_relative_l2 = max_relative_l2.max(relative_l2);
        let absolute_l2 = tolerance.absolute * (golden.len() as f64).sqrt();
        if field_l2 > absolute_l2.max(tolerance.relative * scale_l2) && first_divergence.is_none() {
            first_divergence = Some(format!(
                "{field_name} relative L2 {relative_l2:e} exceeds {:e}",
                tolerance.relative
            ));
        }
    }

    let passed = first_divergence.is_none();
    Ok(FileComparison {
        name: name.to_string(),
        kind: ComparisonKind::Numeric,
        max_abs: (value_count > 0).then_some(max_abs),
        max_rel: (value_count > 0).then_some(max_relative_l2),
        rms: (value_count > 0).then(|| (sum_squared / value_count as f64).sqrt()),
        first_divergence: first_divergence.clone(),
        passed,
        detail: if passed {
            format!(
                "semantic spectrum match ({} field(s), {value_count} values)",
                fields.len()
            )
        } else {
            first_divergence.unwrap_or_else(|| "semantic spectrum mismatch".to_string())
        },
    })
}

fn supports_semantic_binary_comparison(format: FileFormat) -> bool {
    matches!(
        format,
        FileFormat::EmeshBin | FileFormat::ChiaBin | FileFormat::GtrBin | FileFormat::GgBin
    )
}

#[derive(Debug)]
struct SemanticBinaryPayload {
    metadata: Vec<(String, i64)>,
    values: Vec<f64>,
}

fn compare_semantic_binary_files(
    name: &str,
    golden_path: &Path,
    produced_path: &Path,
    format: FileFormat,
    tolerance: NumericTolerance,
) -> Result<FileComparison> {
    let golden_bytes = std::fs::read(golden_path)
        .with_context(|| format!("failed to read {}", golden_path.display()))?;
    let produced_bytes = std::fs::read(produced_path)
        .with_context(|| format!("failed to read {}", produced_path.display()))?;
    let golden = semantic_binary_payload(format, golden_path, &golden_bytes)
        .with_context(|| format!("failed to decode golden {}", golden_path.display()))?;
    let produced = semantic_binary_payload(format, produced_path, &produced_bytes)
        .with_context(|| format!("failed to decode produced {}", produced_path.display()))?;

    if golden.metadata != produced.metadata {
        let first = golden
            .metadata
            .iter()
            .zip(&produced.metadata)
            .find(|(golden, produced)| golden != produced)
            .map_or_else(
                || {
                    format!(
                        "metadata count differs: golden {} vs produced {}",
                        golden.metadata.len(),
                        produced.metadata.len()
                    )
                },
                |(golden, produced)| {
                    format!(
                        "metadata differs: {}={} vs {}={}",
                        golden.0, golden.1, produced.0, produced.1
                    )
                },
            );
        return Ok(FileComparison {
            name: name.to_string(),
            kind: ComparisonKind::SemanticBinary,
            max_abs: None,
            max_rel: None,
            rms: None,
            first_divergence: Some(first.clone()),
            passed: false,
            detail: first,
        });
    }

    if golden.values.len() != produced.values.len() {
        let detail = format!(
            "numeric value count differs: golden {} vs produced {}",
            golden.values.len(),
            produced.values.len()
        );
        return Ok(FileComparison {
            name: name.to_string(),
            kind: ComparisonKind::SemanticBinary,
            max_abs: None,
            max_rel: None,
            rms: None,
            first_divergence: Some(detail.clone()),
            passed: false,
            detail,
        });
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut first_divergence = None;
    for (index, (&golden_value, &produced_value)) in
        golden.values.iter().zip(&produced.values).enumerate()
    {
        let diff = (golden_value - produced_value).abs();
        let scale = golden_value.abs().max(produced_value.abs());
        let relative = if scale > 0.0 { diff / scale } else { 0.0 };
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(relative);
        sum_squared += diff * diff;
        let threshold = tolerance.absolute.max(tolerance.relative * scale);
        if diff > threshold && first_divergence.is_none() {
            first_divergence = Some(format!(
                "numeric value {}: {golden_value:e} vs {produced_value:e} (abs diff {diff:e})",
                index + 1
            ));
        }
    }
    let value_count = golden.values.len();
    let rms = (value_count > 0).then(|| (sum_squared / value_count as f64).sqrt());
    let passed = first_divergence.is_none();
    let detail = if passed {
        format!("semantic match ({value_count} numeric value(s) compared)")
    } else {
        first_divergence
            .clone()
            .unwrap_or_else(|| "semantic binary mismatch".to_string())
    };
    Ok(FileComparison {
        name: name.to_string(),
        kind: ComparisonKind::SemanticBinary,
        max_abs: (value_count > 0).then_some(max_abs),
        max_rel: (value_count > 0).then_some(max_rel),
        rms,
        first_divergence,
        passed,
        detail,
    })
}

fn semantic_binary_payload(
    format: FileFormat,
    path: &Path,
    bytes: &[u8],
) -> Result<SemanticBinaryPayload> {
    match format {
        FileFormat::EmeshBin => {
            let data = <refeff_io::EmeshBinData as FeffCodec>::decode(path, bytes)?;
            Ok(SemanticBinaryPayload {
                metadata: vec![
                    metadata("point-count", data.point_count_declared)?,
                    metadata("horizontal-count", data.horizontal_count)?,
                    metadata("danes-extension-count", data.danes_extension_count)?,
                ],
                values: complex64_values(data.energy_hartree.iter()),
            })
        }
        FileFormat::ChiaBin => {
            let data = <refeff_io::ChiaBinData as FeffCodec>::decode(path, bytes)?;
            Ok(SemanticBinaryPayload {
                metadata: vec![metadata("value-count", data.values.len())?],
                values: complex64_values(data.values.iter()),
            })
        }
        FileFormat::GtrBin => {
            let data = <refeff_io::GtrBinData as FeffCodec>::decode(path, bytes)?;
            Ok(SemanticBinaryPayload {
                metadata: vec![
                    metadata("point-count", data.point_count_declared)?,
                    metadata("horizontal-count", data.horizontal_count)?,
                    metadata("danes-extension-count", data.danes_extension_count)?,
                    metadata("highest-potential-index", data.highest_potential_index)?,
                    ("fms-mode".to_string(), i64::from(data.fms_mode)),
                    metadata("energy-count", data.energy_count())?,
                    metadata("potential-count", data.potential_count())?,
                    metadata("angular-count", data.angular_channel_count())?,
                ],
                values: complex64_values(data.values.iter()),
            })
        }
        FileFormat::GgBin => {
            let data = <refeff_io::GgDatData as FeffCodec>::decode(path, bytes)?;
            let mut metadata_values = vec![metadata("section-count", data.section_count())?];
            let mut values = Vec::new();
            for (index, section) in data.sections.iter().enumerate() {
                metadata_values.push(metadata(
                    &format!("section-{}-number", index + 1),
                    section.section_number,
                )?);
                metadata_values.push(metadata(
                    &format!("section-{}-rows", index + 1),
                    section.row_count(),
                )?);
                metadata_values.push(metadata(
                    &format!("section-{}-columns", index + 1),
                    section.column_count(),
                )?);
                values.extend(complex64_values(section.values.iter()));
            }
            Ok(SemanticBinaryPayload {
                metadata: metadata_values,
                values,
            })
        }
        _ => anyhow::bail!("{format:?} has no semantic binary comparator"),
    }
}

fn metadata(label: &str, value: usize) -> Result<(String, i64)> {
    Ok((
        label.to_string(),
        i64::try_from(value).with_context(|| format!("{label} does not fit in i64"))?,
    ))
}

fn complex64_values<'a>(values: impl IntoIterator<Item = &'a num_complex::Complex64>) -> Vec<f64> {
    values
        .into_iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
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

/// Per-format semantic tolerance from the central FEFF format registry.
fn tolerance_for(name: &str) -> Tolerance {
    identify_format(name).map_or_else(Tolerance::default, |descriptor| Tolerance {
        rel: descriptor.tolerance.relative,
        abs: descriptor.tolerance.absolute,
    })
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

fn print_comparison_table(
    example: &str,
    comparisons: &[FileComparison],
    required_targets: &[&str],
) {
    println!("parity: {example}");
    println!(
        "{:<28} {:<8} {:<12} {:<12} {:<12} first-divergence / detail",
        "file", "status", "max-abs", "max-rel", "rms"
    );
    for comparison in comparisons {
        let status = match comparison.kind {
            ComparisonKind::NoGoldenReference => "SKIP",
            _ if !comparison.passed
                && !required_targets.contains(&comparison_file_name(&comparison.name)) =>
            {
                "WARN"
            }
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
    let required_passed = comparisons
        .iter()
        .filter(|comparison| required_targets.contains(&comparison_file_name(&comparison.name)))
        .filter(|comparison| comparison.passed)
        .count();
    let advisory_differences = comparisons
        .iter()
        .filter(|comparison| {
            comparison.kind != ComparisonKind::NoGoldenReference
                && !required_targets.contains(&comparison_file_name(&comparison.name))
                && !comparison.passed
        })
        .count();
    println!(
        "parity gate: {required_passed}/{} required primary output(s) passed; \
         {advisory_differences} auxiliary diagnostic difference(s) reported",
        required_targets.len()
    );
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
    fn required_parity_targets_follow_workflow_primary_outputs() {
        assert_eq!(required_parity_targets("XANES/BN"), &["xmu.dat"]);
        assert_eq!(required_parity_targets("EXAFS/Cu"), &["chi.dat"]);
        assert_eq!(
            required_parity_targets("BAND/Cr2GeC"),
            &["bandstructure.dat"]
        );
    }

    #[test]
    fn canonical_golden_output_wins_over_legacy_reference_alias() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-canonical-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root)?;
        let legacy = root.join("referencexmu.dat");
        let canonical = root.join("xmu.dat");
        std::fs::write(&legacy, "legacy\n")?;
        std::fs::write(&canonical, "fresh\n")?;

        let files = golden_files_by_target(vec![legacy, canonical.clone()])?;

        assert_eq!(files.get("xmu.dat"), Some(&canonical));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn comparable_golden_files_ignore_nested_compatibility_subcases() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("refeff-xtask-parity-nested-{}", std::process::id()));
        let nested = root.join("rhorrp-density");
        std::fs::create_dir_all(&nested)?;
        std::fs::write(root.join("xmu.dat"), "top-level\n")?;
        std::fs::write(nested.join("density.dat"), "auxiliary\n")?;

        let files = comparable_golden_files(&root)?;

        assert_eq!(files, vec![root.join("xmu.dat")]);
        std::fs::remove_dir_all(root)?;
        Ok(())
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
    fn compare_xmu_uses_header_independent_column_l2_tolerance() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-xmu-test-{}",
            std::process::id()
        ));
        let golden_dir = root.join("golden");
        let produced_dir = root.join("produced");
        std::fs::create_dir_all(&golden_dir)?;
        std::fs::create_dir_all(&produced_dir)?;
        let golden = golden_dir.join("xmu.dat");
        let close = produced_dir.join("xmu.dat");
        let far = produced_dir.join("far-xmu.dat");
        std::fs::write(&golden, "# pinned FEFF header\n1 2 3 4 5 6\n2 3 4 5 6 7\n")?;
        std::fs::write(
            &close,
            "# Rust header is intentionally different\n1 2 3 4.00008 5 6\n2 3 4 5.00010 6 7\n",
        )?;
        std::fs::write(
            &far,
            "# Rust header is intentionally different\n1 2 3 4.8 5 6\n2 3 4 6.0 6 7\n",
        )?;

        let close_comparison = compare_files("xmu.dat", &golden, &close)?;
        assert!(
            close_comparison.passed,
            "small column L2 drift should pass: {close_comparison:?}"
        );
        let far_comparison = compare_files("xmu.dat", &golden, &far)?;
        assert!(
            !far_comparison.passed,
            "large column L2 drift should fail: {far_comparison:?}"
        );
        assert!(far_comparison.max_rel.is_some_and(|value| value > 0.1));

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
    fn compare_gg_dat_uses_decoded_byte_payload() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-gg-dat-test-{}",
            std::process::id()
        ));
        let golden_dir = root.join("golden");
        let produced_dir = root.join("produced");
        std::fs::create_dir_all(&golden_dir)?;
        std::fs::create_dir_all(&produced_dir)?;
        let golden_path = golden_dir.join("gg.dat");
        let produced_path = produced_dir.join("gg.dat");
        let data = refeff_io::GgDatData {
            sections: vec![refeff_io::GgDatSection {
                section_number: 1,
                values: ndarray::arr2(&[[num_complex::Complex64::new(1.0, 0.5)]]),
                raw_prefix_lines: None,
            }],
        };
        let mut bytes = data.encode()?;
        let descriptor = bytes
            .windows(3)
            .position(|window| window == b"txt")
            .expect("canonical gg.dat descriptor should contain txt");
        bytes[descriptor..descriptor + 3].copy_from_slice(&[0xc0, 0xae, 0xa6]);
        std::fs::write(&golden_path, &bytes)?;
        std::fs::write(&produced_path, &bytes)?;

        let comparison = compare_files("gg.dat", &golden_path, &produced_path)?;

        assert_eq!(comparison.kind, ComparisonKind::SemanticBinary);
        assert!(comparison.passed, "comparison should pass: {comparison:?}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn compare_emesh_bin_uses_decoded_numeric_values() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "refeff-xtask-parity-emesh-test-{}",
            std::process::id()
        ));
        let golden_dir = root.join("golden");
        let produced_dir = root.join("produced");
        std::fs::create_dir_all(&golden_dir)?;
        std::fs::create_dir_all(&produced_dir)?;
        let golden_path = golden_dir.join("emesh.bin");
        let produced_path = produced_dir.join("emesh.bin");
        let golden = refeff_io::EmeshBinData {
            point_count_declared: 1,
            horizontal_count: 1,
            danes_extension_count: 0,
            energy_hartree: ndarray::arr1(&[num_complex::Complex64::new(1.0, 0.5)]),
        };
        let produced = refeff_io::EmeshBinData {
            energy_hartree: ndarray::arr1(&[num_complex::Complex64::new(1.0 + 5.0e-7, 0.5)]),
            ..golden.clone()
        };
        std::fs::write(&golden_path, golden.encode()?)?;
        std::fs::write(&produced_path, produced.encode()?)?;

        let comparison = compare_files("emesh.bin", &golden_path, &produced_path)?;

        assert_eq!(comparison.kind, ComparisonKind::SemanticBinary);
        assert!(comparison.passed, "comparison should pass: {comparison:?}");
        assert!(comparison.max_abs.is_some_and(|value| value > 0.0));
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
