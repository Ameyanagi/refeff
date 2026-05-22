use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    FeffBinData, GenfmtInput, ListDatData, read_feff_bin, read_feffl_bin, read_list_dat,
    read_module_log_dat, write_feff_bin, write_feffl_bin, write_list_dat, write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF GENFMT cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF GENFMT run can be satisfied from existing `feff.bin` caches.
pub(crate) fn has_cached_genfmt_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("feff.bin").is_file() || !work_dir.join("list.dat").is_file() {
        return Ok(false);
    }
    Ok(genfmt_enabled(&read_input(work_dir)?))
}

/// Run the FEFF GENFMT cached-output path from existing handoff files.
///
/// The GENFMT curved-wave path formatter is still unported. This keeps cached
/// FEFF output usable by validating and re-rendering `feff.bin`, suffixed
/// `feffNN.bin` files, `list.dat`, suffixed `listNN.dat` files, and optional
/// `feffl.bin` NRIXS decomposition caches plus optional `log5.dat`
/// diagnostics.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !genfmt_enabled(&input) {
        return Ok(0);
    }

    let caches = cached_output_paths(work_dir)?;
    if !caches.has_required_base_outputs() {
        bail!("GENFMT path-format generation requires the unported GENFMT numerical solver");
    }

    let mut written = 0_usize;
    let mut base_feff = None;
    for path in caches.feff_bins {
        let data =
            read_feff_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if path.file_name().and_then(|name| name.to_str()) == Some("feff.bin") {
            base_feff = Some(data.clone());
        }
        write_feff_cache(&path, &data)?;
        written += 1;
    }

    for path in caches.list_dats {
        let data =
            read_list_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        write_list_cache(&path, &data)?;
        written += 1;
    }

    if let Some(path) = caches.feffl_bin {
        let base = base_feff.context("feffl.bin cache requires feff.bin metadata")?;
        let max_channel = decomposition_channel(&input)?;
        let data = read_feffl_bin(
            &path,
            base.pad_width,
            base.paths.len(),
            base.energy_count(),
            max_channel,
        )
        .with_context(|| format!("failed to read {}", path.display()))?;
        write_feffl_bin(&path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
    }

    written += write_optional_module_log(&work_dir.join("log5.dat"))?;
    Ok(written)
}

fn genfmt_enabled(input: &GenfmtInput) -> bool {
    input.control.mfeff != 0
}

fn decomposition_channel(input: &GenfmtInput) -> Result<usize> {
    if input.decomposition_channels < 0 {
        bail!("feffl.bin cache requires a nonnegative GENFMT decomposition channel count");
    }
    Ok(input.decomposition_channels as usize)
}

fn read_input(work_dir: &Path) -> Result<GenfmtInput> {
    let input_path = work_dir.join("genfmt.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    GenfmtInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_feff_cache(path: &Path, data: &FeffBinData) -> Result<()> {
    write_feff_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_list_cache(path: &Path, data: &ListDatData) -> Result<()> {
    write_list_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

#[derive(Debug, Default)]
struct GenfmtCachePaths {
    feff_bins: Vec<PathBuf>,
    list_dats: Vec<PathBuf>,
    feffl_bin: Option<PathBuf>,
}

impl GenfmtCachePaths {
    fn has_required_base_outputs(&self) -> bool {
        self.feff_bins
            .iter()
            .any(|path| file_name_is(path, "feff.bin"))
            && self
                .list_dats
                .iter()
                .any(|path| file_name_is(path, "list.dat"))
    }
}

fn cached_output_paths(work_dir: &Path) -> Result<GenfmtCachePaths> {
    let mut paths = GenfmtCachePaths::default();
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
        if is_feff_bin_cache_name(name) {
            paths.feff_bins.push(entry.path());
        } else if is_list_dat_cache_name(name) {
            paths.list_dats.push(entry.path());
        } else if name == "feffl.bin" {
            paths.feffl_bin = Some(entry.path());
        }
    }

    paths.feff_bins.sort();
    paths.list_dats.sort();
    Ok(paths)
}

fn is_feff_bin_cache_name(name: &str) -> bool {
    if name == "feff.bin" {
        return true;
    }
    name.strip_prefix("feff")
        .and_then(|tail| tail.strip_suffix(".bin"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_list_dat_cache_name(name: &str) -> bool {
    if name == "list.dat" {
        return true;
    }
    name.strip_prefix("list")
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn file_name_is(path: &Path, expected: &str) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some(expected)
}

#[cfg(test)]
mod tests;
