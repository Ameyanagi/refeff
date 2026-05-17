use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{
    EelsDatData, EelsInput, ModuleLogData, read_eels_dat, read_module_log_dat, write_eels_dat,
    write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF EELS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF EELS run can be satisfied from an existing `eels.dat` cache.
pub(crate) fn has_cached_eels_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("eels.dat").is_file() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    Ok(input.calculate_elnes)
}

/// Run the FEFF EELS cached-output path from an existing `eels.dat`.
///
/// The EELS/ELNES/EXELFS spectrum generator is still unported. This path keeps
/// cached FEFF spectra available to downstream compatibility tests by
/// validating and re-rendering the typed spectrum table plus optional raw
/// module logs.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.calculate_elnes {
        return Ok(0);
    }

    let output_path = work_dir.join("eels.dat");
    if !output_path.is_file() {
        bail!("EELS spectrum generation requires the unported EELS numerical solver");
    }

    let data = read_eels_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let point_count = data.point_count();
    write_cached_output(&output_path, &data)?;
    write_optional_module_log(&work_dir.join("logeels.dat"))?;
    Ok(point_count)
}

fn read_input(work_dir: &Path) -> Result<EelsInput> {
    let input_path = work_dir.join("eels.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    EelsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &EelsDatData) -> Result<()> {
    write_eels_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use ndarray::array;
    use refeff_io::{
        EelsDatData, ModuleLogData, read_eels_dat, read_module_log_dat, write_eels_dat,
        write_module_log_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn eels_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_eels_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("eels.dat").exists());
        Ok(())
    }

    #[test]
    fn eels_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_eels_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled EELS should require the numerical solver")?;

        assert!(
            error
                .to_string()
                .contains("EELS spectrum generation requires the unported EELS numerical solver")
        );
        Ok(())
    }

    #[test]
    fn eels_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_eels_input(temp.path(), true)?;
        let expected = sample_eels_dat();
        let expected_log = sample_module_log();
        write_eels_dat(temp.path().join("eels.dat"), &expected)?;
        write_module_log_dat(temp.path().join("logeels.dat"), &expected_log)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("logeels.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn eels_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_eels_dir()? else {
            eprintln!("skipping EELS reference test; generated ELNES/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("eels.inp"), temp.path().join("eels.inp"))?;
        std::fs::copy(reference_dir.join("eels.dat"), temp.path().join("eels.dat"))?;
        let expected = read_eels_dat(temp.path().join("eels.dat"))?;

        let count = run_in_dir(temp.path())?;

        let actual = read_eels_dat(temp.path().join("eels.dat"))?;
        assert_eq!(count, expected.point_count());
        assert!(actual.has_tensor());
        assert_eq!(actual, expected);
        Ok(())
    }

    fn write_eels_input(work_dir: &Path, enabled: bool) -> Result<()> {
        std::fs::write(
            work_dir.join("eels.inp"),
            format!(
                concat!(
                    "calculate ELNES?\n",
                    "{:4}\n",
                    "average? relativistic? cross-terms? Which input?\n",
                    "{:4}{:4}{:4}{:4}{:4}\n",
                    "polarizations to be used ; min step max\n",
                    "{:4}{:4}{:4}\n",
                    "beam energy in eV\n",
                    "{:13.5}\n",
                    "beam direction in arbitrary units\n",
                    "{:13.5}{:13.5}{:13.5}\n",
                    "collection and convergence semiangle in rad\n",
                    "{:13.5}{:13.5}\n",
                    "qmesh - radial and angular grid size\n",
                    "{:4}{:4}\n",
                    "detector positions - two angles in rad\n",
                    "{:13.5}{:13.5}\n",
                    "calculate magic angle if magic=1\n",
                    "{:4}\n",
                    "energy for magic angle - eV above threshold\n",
                    "{:13.5}\n"
                ),
                i32::from(enabled),
                0,
                1,
                1,
                1,
                4,
                1,
                1,
                9,
                300_000.0,
                0.0,
                1.0,
                0.0,
                0.0024,
                0.0,
                5,
                3,
                0.0,
                0.0,
                0,
                0.0,
            ),
        )?;
        Ok(())
    }

    fn sample_eels_dat() -> EelsDatData {
        EelsDatData {
            header_lines: vec![
                "# Orientation averaged EELS calculation".to_string(),
                "#  Energy       total         atomic-bg     fine-struct".to_string(),
            ],
            energy_loss_ev: array![8979.41, 8980.98, 8982.40],
            total: array![0.123_014E-12, 0.146_285E-12, 0.176_683E-12],
            atomic_background: array![0.138_430E-12, 0.166_322E-12, 0.203_202E-12],
            fine_structure: array![-0.154_167E-13, -0.200_377E-13, -0.265_188E-13],
            tensor: None,
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating EELS spectrum ...".to_string(),
                "Done with module: EELS.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn reference_eels_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/ELNES/Cu");
        let required = ["eels.inp", "eels.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
