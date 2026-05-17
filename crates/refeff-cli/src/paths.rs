use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{PathsDatData, PathsInput, read_paths_dat, write_paths_dat};

use crate::work_dir_for_input;

/// Run the supported FEFF PATH cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF PATH run can be satisfied from an existing `paths.dat`.
pub(crate) fn has_cached_paths_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("paths.dat").is_file() {
        return Ok(false);
    }
    Ok(path_enabled(&read_input(work_dir)?))
}

/// Run the FEFF PATH cached-output path from an existing `paths.dat`.
///
/// The multiple-scattering pathfinder is still unported. This path preserves
/// FEFF-compatible cache directories by validating and re-rendering the typed
/// `paths.dat` handoff when `paths.inp` says the PATH module would run.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !path_enabled(&input) {
        return Ok(0);
    }

    let output_path = work_dir.join("paths.dat");
    if !output_path.is_file() {
        bail!("PATH pathfinder generation requires the unported PATH numerical solver");
    }

    let data = read_paths_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let path_count = data.paths.len();
    write_cached_output(&output_path, &data)?;
    Ok(path_count)
}

fn path_enabled(input: &PathsInput) -> bool {
    input.control.mpath != 0
}

fn read_input(work_dir: &Path) -> Result<PathsInput> {
    let input_path = work_dir.join("paths.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    PathsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &PathsDatData) -> Result<()> {
    write_paths_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_paths_output, run_in_dir};
    use anyhow::{Context, Result};
    use refeff_io::{
        PathsControl, PathsCriteria, PathsDatAtom, PathsDatData, PathsDatPath, PathsInput,
        paths_input_string, read_paths_dat, write_paths_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn path_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 0)?;
        write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_paths_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn path_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled PATH should require the numerical solver")?;

        assert!(
            error
                .to_string()
                .contains("PATH pathfinder generation requires the unported PATH numerical solver")
        );
        Ok(())
    }

    #[test]
    fn path_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_paths_input(temp.path(), 1)?;
        let expected = sample_paths_dat();
        write_paths_dat(temp.path().join("paths.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert!(has_cached_paths_output(temp.path())?);
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        Ok(())
    }

    #[test]
    fn path_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_paths_dir()? else {
            eprintln!("skipping PATH reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(
            reference_dir.join("paths.inp"),
            temp.path().join("paths.inp"),
        )?;
        std::fs::copy(
            reference_dir.join("paths.dat"),
            temp.path().join("paths.dat"),
        )?;
        let expected = read_paths_dat(temp.path().join("paths.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected.paths.len());
        assert_eq!(read_paths_dat(temp.path().join("paths.dat"))?, expected);
        Ok(())
    }

    fn write_paths_input(work_dir: &Path, mpath: i32) -> Result<()> {
        let input = PathsInput {
            control: PathsControl {
                mpath,
                ms: mpath,
                nncrit: 0,
                nlegxx: 7,
                ipr4: 0,
            },
            criteria: PathsCriteria {
                critpw: 2.5,
                pcritk: 0.0,
                pcrith: 0.0,
                rmax: 5.5,
                rfms2: -1.0,
            },
            ica: -1,
        };
        std::fs::write(work_dir.join("paths.inp"), paths_input_string(&input)?)?;
        Ok(())
    }

    fn sample_paths_dat() -> PathsDatData {
        PathsDatData {
            titles: vec![
                "PATH  Rmax= 5.500,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
            ],
            paths: vec![PathsDatPath {
                index: 1,
                degeneracy: 12.0,
                effective_half_path_length_angstrom: 2.5527,
                row_header:
                    "      x           y           z     ipot  label      rleg      beta        eta"
                        .to_string(),
                atoms: vec![
                    PathsDatAtom {
                        position_angstrom: [-1.805, -1.805, 0.0],
                        potential_index: 1,
                        label: "Cu".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(180.0),
                        eta_degrees: Some(0.0),
                    },
                    PathsDatAtom {
                        position_angstrom: [0.0, 0.0, 0.0],
                        potential_index: 0,
                        label: "Cu".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(180.0),
                        eta_degrees: Some(0.0),
                    },
                ],
            }],
        }
    }

    fn reference_paths_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["paths.inp", "paths.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
