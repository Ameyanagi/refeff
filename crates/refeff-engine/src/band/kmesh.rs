use std::path::Path;

use anyhow::{Context, Result};
use refeff_io::{ReciprocalInput, kmesh_dat_from_reciprocal_cell, read_kmesh_dat, write_kmesh_dat};

pub(crate) fn write_optional_or_generated_kmesh(work_dir: &Path, path: &Path) -> Result<usize> {
    if kmesh_needs_generation(work_dir, path)? {
        return write_generated_kmesh(work_dir, path);
    }
    if !path.is_file() {
        return Ok(0);
    }
    write_cached_kmesh(path)
}

pub(crate) fn prepare_optional_or_generated_kmesh(work_dir: &Path, path: &Path) -> Result<()> {
    if !kmesh_needs_generation(work_dir, path)? {
        if path.is_file() {
            read_kmesh_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
        }
        return Ok(());
    }

    let reciprocal_path = work_dir.join("reciprocal.inp");
    let input = read_reciprocal_input(&reciprocal_path)?;
    if let Some(cell) = input.cell.as_ref() {
        kmesh_dat_from_reciprocal_cell(cell)
            .with_context(|| format!("failed to generate {}", path.display()))?;
    }
    Ok(())
}

pub(crate) fn has_supported_kmesh_handoff(work_dir: &Path) -> Result<bool> {
    let path = work_dir.join("kmesh.dat");
    Ok(kmesh_needs_generation(work_dir, &path).unwrap_or(false))
}

pub(crate) fn has_reciprocal_kmesh_source_handoff(work_dir: &Path) -> Result<bool> {
    Ok(can_generate_kmesh(work_dir).unwrap_or(false))
}

pub(crate) fn run_supported_kmesh_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    write_optional_or_generated_kmesh(work_dir, &work_dir.join("kmesh.dat"))
}

fn write_cached_kmesh(path: &Path) -> Result<usize> {
    let data =
        read_kmesh_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_kmesh_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

pub(crate) fn kmesh_needs_generation(work_dir: &Path, path: &Path) -> Result<bool> {
    if !path.is_file() {
        return can_generate_kmesh(work_dir);
    }
    if read_kmesh_dat(path).is_ok() {
        return Ok(false);
    }
    can_generate_kmesh(work_dir)
}

fn write_generated_kmesh(work_dir: &Path, path: &Path) -> Result<usize> {
    if !can_generate_kmesh(work_dir)? {
        return Ok(0);
    }

    let reciprocal_path = work_dir.join("reciprocal.inp");
    let input = read_reciprocal_input(&reciprocal_path)?;
    let Some(cell) = input.cell.as_ref() else {
        return Ok(0);
    };

    let data = kmesh_dat_from_reciprocal_cell(cell)
        .with_context(|| format!("failed to generate {}", path.display()))?;
    write_kmesh_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn can_generate_kmesh(work_dir: &Path) -> Result<bool> {
    let reciprocal_path = work_dir.join("reciprocal.inp");
    if !reciprocal_path.is_file() {
        return Ok(false);
    }

    let input = read_reciprocal_input(&reciprocal_path)?;
    let Some(cell) = input.cell.as_ref() else {
        return Ok(false);
    };
    Ok(!cell.k_mesh.use_symmetry)
}

pub(crate) fn read_reciprocal_input(path: &Path) -> Result<ReciprocalInput> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    ReciprocalInput::parse_str(path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
}
