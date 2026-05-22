use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    ChiDatData, DanesDatData, Ff2xInput, ModuleLogData, XmuDatData, XmulDatData,
    XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData, read_chi_dat, read_contour_dat,
    read_curve_dat, read_danes_dat, read_module_log_dat, read_prexmu_dat, read_residue_dat,
    read_xmu_dat, read_xmul_dat, read_xscorr_raw_dat, write_chi_dat, write_contour_dat,
    write_curve_dat, write_danes_dat, write_module_log_dat, write_prexmu_dat, write_residue_dat,
    write_xmu_dat, write_xmul_dat, write_xscorr_raw_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF FF2X cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FF2X run can be satisfied from existing spectrum caches.
pub(crate) fn has_cached_ff2x_output(work_dir: &Path) -> Result<bool> {
    if cached_output_paths(work_dir)?.is_empty() {
        return Ok(false);
    }
    Ok(ff2x_enabled(&read_input(work_dir)?))
}

/// Run the FEFF FF2X cached-output path from existing final-spectrum files.
///
/// The FF2X spectrum assembler is still unported. This keeps cached FEFF final
/// spectra usable by validating and re-rendering typed `xmu.dat`,
/// `chi.dat`/`chipNNNN.dat`, `xmul.dat`, and `danes.dat` outputs, plus
/// optional XSCORR diagnostic sidecars and `log6.dat` module diagnostics.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !ff2x_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        bail!("FF2X spectrum generation requires the unported FF2X numerical solver");
    }

    for output in &outputs {
        match output.kind {
            CachedOutputKind::Xmu => {
                let data = read_xmu_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmu_cache(&output.path, &data)?;
            }
            CachedOutputKind::Chi => {
                let data = read_chi_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_chi_cache(&output.path, &data)?;
            }
            CachedOutputKind::Xmul => {
                let data = read_xmul_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmul_cache(&output.path, &data)?;
            }
            CachedOutputKind::Danes => {
                let data = read_danes_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_danes_cache(&output.path, &data)?;
            }
        }
    }

    let sidecar_count = write_optional_xscorr_sidecars(work_dir)?;
    let log_count = write_optional_module_log(&work_dir.join("log6.dat"))?;
    Ok(outputs.len() + sidecar_count + log_count)
}

fn ff2x_enabled(input: &Ff2xInput) -> bool {
    input.control.mchi != 0
}

fn read_input(work_dir: &Path) -> Result<Ff2xInput> {
    let input_path = work_dir.join("ff2x.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    Ff2xInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_xmu_cache(path: &Path, data: &XmuDatData) -> Result<()> {
    write_xmu_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_chi_cache(path: &Path, data: &ChiDatData) -> Result<()> {
    write_chi_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xmul_cache(path: &Path, data: &XmulDatData) -> Result<()> {
    write_xmul_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_danes_cache(path: &Path, data: &DanesDatData) -> Result<()> {
    write_danes_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_xscorr_sidecars(work_dir: &Path) -> Result<usize> {
    Ok(write_optional_prexmu_cache(&work_dir.join("prexmu.dat"))?
        + write_optional_residue_cache(&work_dir.join("residue.dat"))?
        + write_optional_contour_cache(&work_dir.join("contour.dat"))?
        + write_optional_curve_cache(&work_dir.join("curve.dat"))?
        + write_optional_raw_cache(&work_dir.join("raw.dat"))?)
}

fn write_optional_prexmu_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_prexmu_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_prexmu_cache(path, &data)?;
    Ok(1)
}

fn write_optional_residue_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_residue_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_residue_cache(path, &data)?;
    Ok(1)
}

fn write_optional_contour_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_contour_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_contour_cache(path, &data)?;
    Ok(1)
}

fn write_optional_curve_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_curve_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_curve_cache(path, &data)?;
    Ok(1)
}

fn write_optional_raw_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_xscorr_raw_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_raw_cache(path, &data)?;
    Ok(1)
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)?;
    Ok(1)
}

fn write_prexmu_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_prexmu_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_residue_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_residue_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_contour_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_contour_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_curve_cache(path: &Path, data: &XscorrCurveDatData) -> Result<()> {
    write_curve_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_raw_cache(path: &Path, data: &XscorrRawDatData) -> Result<()> {
    write_xscorr_raw_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Xmu,
    Chi,
    Xmul,
    Danes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedOutputPath {
    path: PathBuf,
    kind: CachedOutputKind,
}

fn cached_output_paths(work_dir: &Path) -> Result<Vec<CachedOutputPath>> {
    let mut outputs = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", work_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let kind = if name == "xmu.dat" {
            Some(CachedOutputKind::Xmu)
        } else if name == "chi.dat" || is_chip_dat_name(name) {
            Some(CachedOutputKind::Chi)
        } else if name == "xmul.dat" {
            Some(CachedOutputKind::Xmul)
        } else if name == "danes.dat" {
            Some(CachedOutputKind::Danes)
        } else {
            None
        };
        if let Some(kind) = kind {
            outputs.push(CachedOutputPath {
                path: entry.path(),
                kind,
            });
        }
    }

    outputs.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(outputs)
}

fn is_chip_dat_name(name: &str) -> bool {
    name.strip_prefix("chip")
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
mod tests;
