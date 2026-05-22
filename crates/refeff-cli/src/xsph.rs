use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    EmeshBinData, EmeshDatData, ModuleLogData, MpseDatData, PhaseBinData, XseclBinData,
    XseclDatData, XsectDatData, XsphInput, read_axafs_dat, read_emesh_bin, read_emesh_dat,
    read_module_log_dat, read_mpse_dat, read_phase_bin, read_xsecl_bin, read_xsecl_dat,
    read_xsecl2_dat, read_xsect_dat, write_axafs_dat, write_emesh_bin, write_emesh_dat,
    write_module_log_dat, write_mpse_dat, write_phase_bin, write_xsecl_bin, write_xsecl_dat,
    write_xsecl2_dat, write_xsect_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF XSPH cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF XSPH run can be satisfied from existing phase caches.
pub(crate) fn has_cached_xsph_output(work_dir: &Path) -> Result<bool> {
    let caches = XsphCachePaths::new(work_dir);
    if !caches.has_required_base_outputs() {
        return Ok(false);
    }
    Ok(xsph_enabled(&read_input(work_dir)?))
}

/// Run the FEFF XSPH cached-output path from existing phase-shift files.
///
/// The XSPH phase-shift solver is still unported. This keeps cached FEFF
/// phase directories usable by validating and re-rendering typed `phase.bin`,
/// `xsect.dat`, and optional NRIXS `xsecl.dat`/`xsecl2.dat`/`xsecl.bin`
/// AXAFS `axafs.dat`, MPSE `mpse.dat`, phase-mesh `emesh.dat`/`emesh.bin`,
/// and `log2.dat` diagnostic handoffs.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) {
        return Ok(0);
    }

    let caches = XsphCachePaths::new(work_dir);
    if !caches.has_required_base_outputs() {
        bail!("XSPH phase-shift generation requires the unported XSPH numerical solver");
    }

    let phase = read_phase_bin(&caches.phase_bin)
        .with_context(|| format!("failed to read {}", caches.phase_bin.display()))?;
    write_phase_cache(&caches.phase_bin, &phase)?;

    let xsect = read_xsect_dat(&caches.xsect_dat)
        .with_context(|| format!("failed to read {}", caches.xsect_dat.display()))?;
    write_xsect_cache(&caches.xsect_dat, &xsect)?;

    let mut written = 2_usize;
    if caches.axafs_dat.is_file() {
        let data = read_axafs_dat(&caches.axafs_dat)
            .with_context(|| format!("failed to read {}", caches.axafs_dat.display()))?;
        write_axafs_cache(&caches.axafs_dat, &data)?;
        written += 1;
    }
    if caches.xsecl_dat.is_file() {
        let data = read_xsecl_dat(&caches.xsecl_dat)
            .with_context(|| format!("failed to read {}", caches.xsecl_dat.display()))?;
        write_xsecl_cache(&caches.xsecl_dat, &data)?;
        written += 1;
    }
    if caches.xsecl2_dat.is_file() {
        let data = read_xsecl2_dat(&caches.xsecl2_dat)
            .with_context(|| format!("failed to read {}", caches.xsecl2_dat.display()))?;
        write_xsecl2_cache(&caches.xsecl2_dat, &data)?;
        written += 1;
    }
    if caches.xsecl_bin.is_file() {
        let data = read_xsecl_bin(&caches.xsecl_bin, phase.pad_width, phase.energy_count)
            .with_context(|| format!("failed to read {}", caches.xsecl_bin.display()))?;
        write_xsecl_bin_cache(&caches.xsecl_bin, &data)?;
        written += 1;
    }
    if caches.mpse_dat.is_file() {
        let data = read_mpse_dat(&caches.mpse_dat)
            .with_context(|| format!("failed to read {}", caches.mpse_dat.display()))?;
        write_mpse_cache(&caches.mpse_dat, &data)?;
        written += 1;
    }
    if caches.emesh_dat.is_file() {
        let data = read_emesh_dat(&caches.emesh_dat)
            .with_context(|| format!("failed to read {}", caches.emesh_dat.display()))?;
        write_emesh_cache(&caches.emesh_dat, &data)?;
        written += 1;
    }
    if caches.emesh_bin.is_file() {
        let data = read_emesh_bin(&caches.emesh_bin)
            .with_context(|| format!("failed to read {}", caches.emesh_bin.display()))?;
        write_emesh_bin_cache(&caches.emesh_bin, &data)?;
        written += 1;
    }
    written += write_optional_module_log(&caches.log2_dat)?;

    Ok(written)
}

fn xsph_enabled(input: &XsphInput) -> bool {
    input.control.mphase != 0
}

fn read_input(work_dir: &Path) -> Result<XsphInput> {
    let input_path = work_dir.join("xsph.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    XsphInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_phase_cache(path: &Path, data: &PhaseBinData) -> Result<()> {
    write_phase_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsect_cache(path: &Path, data: &XsectDatData) -> Result<()> {
    write_xsect_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_axafs_cache(path: &Path, data: &refeff_io::AxafsDatData) -> Result<()> {
    write_axafs_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl_cache(path: &Path, data: &XseclDatData) -> Result<()> {
    write_xsecl_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl2_cache(path: &Path, data: &XseclDatData) -> Result<()> {
    write_xsecl2_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl_bin_cache(path: &Path, data: &XseclBinData) -> Result<()> {
    write_xsecl_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_mpse_cache(path: &Path, data: &MpseDatData) -> Result<()> {
    write_mpse_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_emesh_cache(path: &Path, data: &EmeshDatData) -> Result<()> {
    write_emesh_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_emesh_bin_cache(path: &Path, data: &EmeshBinData) -> Result<()> {
    write_emesh_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
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

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XsphCachePaths {
    phase_bin: PathBuf,
    xsect_dat: PathBuf,
    axafs_dat: PathBuf,
    xsecl_dat: PathBuf,
    xsecl2_dat: PathBuf,
    xsecl_bin: PathBuf,
    mpse_dat: PathBuf,
    emesh_dat: PathBuf,
    emesh_bin: PathBuf,
    log2_dat: PathBuf,
}

impl XsphCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            phase_bin: work_dir.join("phase.bin"),
            xsect_dat: work_dir.join("xsect.dat"),
            axafs_dat: work_dir.join("axafs.dat"),
            xsecl_dat: work_dir.join("xsecl.dat"),
            xsecl2_dat: work_dir.join("xsecl2.dat"),
            xsecl_bin: work_dir.join("xsecl.bin"),
            mpse_dat: work_dir.join("mpse.dat"),
            emesh_dat: work_dir.join("emesh.dat"),
            emesh_bin: work_dir.join("emesh.bin"),
            log2_dat: work_dir.join("log2.dat"),
        }
    }

    fn has_required_base_outputs(&self) -> bool {
        self.phase_bin.is_file() && self.xsect_dat.is_file()
    }
}

#[cfg(test)]
mod tests;
