use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    DensityInput, read_rhorrp_density_bin, read_rhorrp_density_text,
    rhorrp_density_filename_is_binary, write_rhorrp_density_bin, write_rhorrp_density_text,
};

use crate::work_dir_for_input;

/// Run the supported FEFF RHORRP cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF RHORRP run can be satisfied from existing density caches.
pub(crate) fn has_cached_rhorrp_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("density.inp").is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if input.grids.is_empty() {
        return Ok(false);
    }

    Ok(cached_output_paths(work_dir, &input)?
        .iter()
        .all(|output| output.path.is_file()))
}

/// Run the FEFF RHORRP cached-output path from existing density-grid outputs.
///
/// The full density-matrix solver is still unported. This preserves cached FEFF
/// `DENSITY` runs by validating and re-rendering typed ASCII and binary
/// RHORRP density-grid output files named by `density.inp`.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.grids.is_empty() {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir, &input)?;
    let missing = outputs.iter().find(|output| !output.path.is_file());
    if let Some(output) = missing {
        bail!(
            "RHORRP density generation requires the unported RHORRP numerical solver; missing cached output {}",
            output.path.display()
        );
    }

    for output in &outputs {
        match output.kind {
            CachedOutputKind::Text => {
                let data = read_rhorrp_density_text(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_density_text(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
            }
            CachedOutputKind::Binary => {
                let data = read_rhorrp_density_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_density_bin(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
            }
        }
    }

    Ok(outputs.len())
}

fn read_input(work_dir: &Path) -> Result<DensityInput> {
    let input_path = work_dir.join("density.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    DensityInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Text,
    Binary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedOutputPath {
    path: PathBuf,
    kind: CachedOutputKind,
}

fn cached_output_paths(work_dir: &Path, input: &DensityInput) -> Result<Vec<CachedOutputPath>> {
    input
        .grids
        .iter()
        .map(|grid| {
            let path = output_path(work_dir, &grid.filename)?;
            let kind = if rhorrp_density_filename_is_binary(&grid.filename) {
                CachedOutputKind::Binary
            } else {
                CachedOutputKind::Text
            };
            Ok(CachedOutputPath { path, kind })
        })
        .collect()
}

fn output_path(work_dir: &Path, filename: &str) -> Result<PathBuf> {
    let path = Path::new(filename);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("RHORRP density output filename must stay within the work directory: {filename}");
    }
    Ok(work_dir.join(path))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_rhorrp_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{arr1, arr2};
    use refeff_io::{
        RhorrpDensityBinData, RhorrpDensityTextData, RhorrpNearestAtomColumns,
        read_rhorrp_density_bin, read_rhorrp_density_text, write_rhorrp_density_bin,
        write_rhorrp_density_text,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn rhorrp_module_skips_empty_density_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("density.inp"), "")?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_rhorrp_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn rhorrp_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RHORRP should require the numerical solver")?;

        assert!(
            error.to_string().contains(
                "RHORRP density generation requires the unported RHORRP numerical solver"
            )
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        let expected_text = sample_density_text();
        let expected_bin = sample_density_bin();
        write_rhorrp_density_text(temp.path().join("density.dat"), &expected_text)?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &expected_bin)?;
        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert!(has_cached_rhorrp_output(temp.path())?);
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_rhorrp_dir()? else {
            eprintln!("skipping RHORRP reference test; generated RHORRP reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in ["density.inp", "density.dat", "density.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }

        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        Ok(())
    }

    fn write_density_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("density.inp"),
            concat!(
                "line density.dat 0.0 0.0 0.0 core\n",
                "1.0 0.0 0.0 2\n",
                "line density.bin 0.0 0.0 0.0 core\n",
                "1.0 0.0 0.0 2\n",
            ),
        )?;
        Ok(())
    }

    fn sample_density_text() -> RhorrpDensityTextData {
        RhorrpDensityTextData {
            points_angstrom: arr2(&[[0.0, 0.0, 0.0], [0.529_177_249, 0.0, 0.0]]),
            density_per_angstrom3: arr1(&[1.0, 2.0]),
            nearest: Some(RhorrpNearestAtomColumns {
                displacement_bohr: arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atom_indices: arr1(&[0, 0]),
                potential_indices: arr1(&[0, 0]),
            }),
        }
    }

    fn sample_density_bin() -> RhorrpDensityBinData {
        RhorrpDensityBinData {
            origin_angstrom: [0.0, 0.0, 0.0],
            axes_angstrom: arr2(&[[1.0], [0.0], [0.0]]),
            points_per_axis: vec![2],
            density_per_angstrom3: arr1(&[1.0, 2.0]),
        }
    }

    fn reference_rhorrp_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/RHORRP");
        let required = ["density.inp", "density.dat", "density.bin"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
