use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{CrpaDatData, CrpaInput, read_crpa_dat, write_crpa_dat};

use crate::work_dir_for_input;

/// Run the supported FEFF CRPA cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF CRPA run can be satisfied from an existing `crpa.dat` cache.
pub(crate) fn has_cached_crpa_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("crpa.dat").is_file() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    Ok(input.enabled)
}

/// Run the FEFF CRPA cached-output path from an existing `crpa.dat`.
///
/// The CRPA numerical Hubbard-`U` solver is still unported. This keeps disabled
/// handoff files compatible and lets cached FEFF output pass through the typed
/// Rust renderer for module-level orchestration tests.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.enabled {
        return Ok(0);
    }

    let output_path = work_dir.join("crpa.dat");
    if !output_path.is_file() {
        bail!("CRPA Hubbard-U generation requires the unported CRPA numerical solver");
    }

    let data = read_crpa_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    write_cached_output(&output_path, &data)?;
    Ok(1)
}

fn read_input(work_dir: &Path) -> Result<CrpaInput> {
    let input_path = work_dir.join("crpa.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    CrpaInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &CrpaDatData) -> Result<()> {
    write_crpa_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use refeff_io::{CrpaDatData, read_crpa_dat, write_crpa_dat};
    use std::path::Path;

    #[test]
    fn crpa_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("crpa.dat").exists());
        Ok(())
    }

    #[test]
    fn crpa_module_rejects_enabled_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled CRPA should require the numerical solver")?;

        assert!(
            error
                .to_string()
                .contains("CRPA Hubbard-U generation requires the unported CRPA numerical solver")
        );
        Ok(())
    }

    #[test]
    fn crpa_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected = CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        };
        write_crpa_dat(temp.path().join("crpa.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(read_crpa_dat(temp.path().join("crpa.dat"))?, expected);
        Ok(())
    }

    fn write_crpa_input(work_dir: &Path, enabled: bool) -> Result<()> {
        std::fs::write(
            work_dir.join("crpa.inp"),
            format!(
                concat!(" do_CRPA{:12}\n", " rcut{:21.16}     \n", " l_crpa{:12}\n",),
                i32::from(enabled),
                3.5,
                2
            ),
        )?;
        Ok(())
    }
}
