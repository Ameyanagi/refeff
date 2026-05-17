use std::path::Path;

use anyhow::{Context, Result};
use refeff_io::{
    Fort16Data, MiscDatData, ScfConvergenceData, potential_dat_outputs_from_bins, read_apot_bin,
    read_convergence_scf, read_convergence_scf_fine, read_fort16, read_misc_dat, read_pot_bin,
    write_convergence_scf, write_convergence_scf_fine, write_fort16, write_misc_dat,
};

use crate::work_dir_for_input;

/// Run FEFF `wpot`-compatible potential output generation beside an input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Write `potNN.dat` files from `pot.bin` and `apot.bin` in a work directory.
///
/// Existing potential-stage diagnostic sidecars are also validated and
/// re-rendered so cached FEFF POT directories keep their typed handoff files.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let pot_path = work_dir.join("pot.bin");
    let apot_path = work_dir.join("apot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let apot = read_apot_bin(&apot_path)
        .with_context(|| format!("failed to read {}", apot_path.display()))?;
    let outputs = potential_dat_outputs_from_bins(&pot, &apot)
        .context("failed to render FEFF wpot potential outputs")?;
    let count = outputs.len();
    for (name, content) in outputs {
        let output_path = work_dir.join(&name);
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    let count = count
        + write_optional_misc_dat(&work_dir.join("misc.dat"))?
        + write_optional_convergence_scf(&work_dir.join("convergence.scf"))?
        + write_optional_convergence_scf_fine(&work_dir.join("convergence.scf.fine"))?
        + write_optional_fort16(&work_dir.join("fort.16"))?;
    Ok(count)
}

fn write_optional_misc_dat(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_misc_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_misc_cache(path, &data)?;
    Ok(1)
}

fn write_optional_convergence_scf(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_convergence_scf(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_convergence_scf_cache(path, &data)?;
    Ok(1)
}

fn write_optional_convergence_scf_fine(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_convergence_scf_fine(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    write_convergence_scf_fine_cache(path, &data)?;
    Ok(1)
}

fn write_optional_fort16(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_fort16(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_fort16_cache(path, &data)?;
    Ok(1)
}

fn write_misc_cache(path: &Path, data: &MiscDatData) -> Result<()> {
    write_misc_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_convergence_scf_cache(path: &Path, data: &ScfConvergenceData) -> Result<()> {
    write_convergence_scf(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_convergence_scf_fine_cache(path: &Path, data: &ScfConvergenceData) -> Result<()> {
    write_convergence_scf_fine(path, data)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn write_fort16_cache(path: &Path, data: &Fort16Data) -> Result<()> {
    write_fort16(path, data).with_context(|| format!("failed to write {}", path.display()))
}
