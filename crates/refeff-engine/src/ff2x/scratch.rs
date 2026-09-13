//! Declared FF2X comparison inputs and temporary workspace ownership.
use super::*;

pub(super) struct Ff2xScratchWorkDir {
    directory: tempfile::TempDir,
}

impl Ff2xScratchWorkDir {
    pub(super) fn copy_source_files_from(work_dir: &Path) -> Result<Self> {
        let directory = crate::execution::temporary_workspace("refeff-ff2x-source-")
            .context("failed to create FF2X source scratch directory")?;
        let scratch = Self { directory };
        copy_ff2x_source_files(work_dir, scratch.path())?;
        Ok(scratch)
    }

    pub(super) fn path(&self) -> &Path {
        self.directory.path()
    }
}

fn copy_ff2x_source_files(work_dir: &Path, scratch_dir: &Path) -> Result<()> {
    // Inputs used by FF2X source preparation and its optional damping branches.
    // Final spectra are deliberately excluded from the comparison workspace.
    let names = [
        "ff2x.inp",
        "global.inp",
        "eels.inp",
        "atoms.dat",
        "spring.inp",
        "dmdw.inp",
        "xsect.dat",
        "xsecl.bin",
        "feff.bin",
        "feffl.bin",
        "list.dat",
        "fms.bin",
        "fmsl.bin",
        "phase.bin",
        "pot.bin",
        "geom.dat",
        "paths.dat",
        "hubbard.inp",
        "reciprocal.inp",
        "prexmu.dat",
        "residue.dat",
        "contour.dat",
        "curve.dat",
        "raw.dat",
        "cum.dat",
        "chia.bin",
        FF2X_CFAVERAGE_STATE_FILE,
    ];
    let mut sources: Vec<PathBuf> = names.into_iter().map(PathBuf::from).collect();
    for entry in std::fs::read_dir(work_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if (name.starts_with("feff") && name.ends_with(".bin"))
            || (name.starts_with("list") && name.ends_with(".dat"))
        {
            sources.push(PathBuf::from(name.as_ref()));
        }
    }
    if let Some(calculation) = read_ff2x_dmdw_calculation(work_dir)? {
        let path = PathBuf::from(calculation.dym_file);
        // Absolute declared sources remain absolute in the copied input.
        if path.is_relative() {
            anyhow::ensure!(
                path.components()
                    .all(|part| matches!(part, std::path::Component::Normal(_))),
                "FF2X scratch dependency must not escape the workspace: {}",
                path.display()
            );
            sources.push(path);
        }
    }
    sources.sort();
    sources.dedup();
    for relative in sources {
        let source = work_dir.join(&relative);
        if !source.is_file() {
            continue;
        }
        let target = scratch_dir.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&source, &target)
            .with_context(|| format!("failed to copy FF2X dependency {}", source.display()))?;
    }
    Ok(())
}
