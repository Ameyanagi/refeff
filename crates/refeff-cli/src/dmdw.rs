use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{DmdwInput, DmdwOutData, read_dmdw_out, write_dmdw_out};

use crate::work_dir_for_input;

/// Run the supported FEFF DMDW cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF DMDW run can be satisfied from an existing `dmdw.out` cache.
pub(crate) fn has_cached_dmdw_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("dmdw.out").is_file() {
        return Ok(false);
    }
    Ok(matches!(read_input(work_dir)?, DmdwInput::Enabled(_)))
}

/// Run the FEFF DMDW cached-output path from an existing `dmdw.out`.
///
/// The dynamical-matrix Debye-Waller solver is still unported. This preserves
/// FEFF-compatible cached diagnostics by validating and re-rendering the typed
/// `dmdw.out` report.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    if matches!(read_input(work_dir)?, DmdwInput::Disabled) {
        return Ok(0);
    }

    let output_path = work_dir.join("dmdw.out");
    if !output_path.is_file() {
        bail!("DMDW Debye-Waller generation requires the unported DMDW numerical solver");
    }

    let data = read_dmdw_out(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let section_count = data.section_count();
    write_cached_output(&output_path, &data)?;
    Ok(section_count)
}

fn read_input(work_dir: &Path) -> Result<DmdwInput> {
    let input_path = work_dir.join("dmdw.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    DmdwInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &DmdwOutData) -> Result<()> {
    write_dmdw_out(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use refeff_io::{
        DmdwOutData, DmdwOutHeader, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature,
        read_dmdw_out, write_dmdw_out,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn dmdw_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_disabled_dmdw_input(temp.path())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("dmdw.out").exists());
        Ok(())
    }

    #[test]
    fn dmdw_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_enabled_dmdw_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled DMDW should require the numerical solver")?;

        assert!(
            error.to_string().contains(
                "DMDW Debye-Waller generation requires the unported DMDW numerical solver"
            )
        );
        Ok(())
    }

    #[test]
    fn dmdw_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_enabled_dmdw_input(temp.path())?;
        let expected = sample_dmdw_out();
        write_dmdw_out(temp.path().join("dmdw.out"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(read_dmdw_out(temp.path().join("dmdw.out"))?, expected);
        Ok(())
    }

    #[test]
    fn dmdw_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_dmdw_dir()? else {
            eprintln!(
                "skipping DMDW reference test; generated DEBYE/DM/EXAFS/Cu reference not found"
            );
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("dmdw.inp"), temp.path().join("dmdw.inp"))?;
        std::fs::copy(reference_dir.join("dmdw.out"), temp.path().join("dmdw.out"))?;
        let expected = read_dmdw_out(temp.path().join("dmdw.out"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected.section_count());
        assert_eq!(read_dmdw_out(temp.path().join("dmdw.out"))?, expected);
        Ok(())
    }

    fn write_disabled_dmdw_input(work_dir: &Path) -> Result<()> {
        std::fs::write(work_dir.join("dmdw.inp"), "-999\n")?;
        Ok(())
    }

    fn write_enabled_dmdw_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("dmdw.inp"),
            concat!(
                "   1\n",
                "   2\n",
                "   1    450.000\n",
                "   0\n",
                "feff.dym\n",
                "   1\n",
                "   2   1   0          29.78\n",
            ),
        )?;
        Ok(())
    }

    fn sample_dmdw_out() -> DmdwOutData {
        let mut section = DmdwOutSection::new(DmdwOutSubject::PathIndices(vec![1, 2]));
        section.reduced_mass_amu = Some(31.773);
        section.path_length_angstrom = Some(2.5323);
        section.sigma2_1e_minus_3_angstrom2 = Some(11.8576);

        DmdwOutData {
            header: Some(DmdwOutHeader {
                lanczos_recursion_order: 2,
                temperature: DmdwOutTemperature::Single(450.0),
                dynamical_matrix_file: "feff.dym".to_string(),
            }),
            sections: vec![section],
        }
    }

    fn reference_dmdw_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/DEBYE/DM/EXAFS/Cu");
        let required = ["dmdw.inp", "dmdw.out"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
