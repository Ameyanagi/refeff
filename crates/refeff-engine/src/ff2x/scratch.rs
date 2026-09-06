//! Declared FF2X comparison inputs and temporary workspace ownership.
use super::*;

pub(super) struct Ff2xScratchWorkDir {
    path: PathBuf,
}

impl Ff2xScratchWorkDir {
    pub(super) fn copy_source_files_from(work_dir: &Path) -> Result<Self> {
        for attempt in 0..100_u32 {
            let path = std::env::temp_dir().join(format!(
                "refeff-ff2x-source-{}-{}-{attempt}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .context("system clock is before UNIX_EPOCH")?
                    .as_nanos()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    let scratch = Self { path };
                    copy_ff2x_source_files(work_dir, scratch.path())?;
                    return Ok(scratch);
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()));
                }
            }
        }
        bail!("failed to create unique FF2X source scratch directory");
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Ff2xScratchWorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
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
