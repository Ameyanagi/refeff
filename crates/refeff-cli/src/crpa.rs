use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{
    CrpaDatData, CrpaInput, ModuleLogData, read_crpa_dat, read_module_log_dat, write_crpa_dat,
    write_module_log_dat,
};

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
    Ok(1 + write_optional_module_log(&work_dir.join("logscrn.dat"))?)
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

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use refeff_io::{
        CrpaDatData, ModuleLogData, parse_crpa_dat, parse_module_log_dat, read_crpa_dat,
        read_module_log_dat, write_crpa_dat, write_module_log_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

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

    #[test]
    fn crpa_module_roundtrips_cached_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_crpa_input(temp.path(), true)?;
        let expected_output = CrpaDatData {
            header_lines: vec!["U, n, U_Bare".to_string()],
            hubbard_u: 0.197_879_035_252_010,
            occupation: 1.0,
            bare_u: 0.694_283_422_651_496,
        };
        let expected_log = sample_module_log();
        write_crpa_dat(temp.path().join("crpa.dat"), &expected_output)?;
        write_module_log_dat(temp.path().join("logscrn.dat"), &expected_log)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_output
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn crpa_module_roundtrips_reference_zip_when_present() -> Result<()> {
        let Some(zip_path) = reference_crpa_zip()? else {
            eprintln!("skipping CRPA reference test; CRPA REFERENCE.zip not found");
            return Ok(());
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            eprintln!("skipping CRPA reference test; unzip command not found");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("crpa.inp"),
            unzip_reference_entry(&zip_path, "REFERENCE/crpa.inp")?,
        )?;
        std::fs::write(
            temp.path().join("crpa.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/crpa.dat")?,
        )?;
        std::fs::write(
            temp.path().join("logscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/logscrn.dat")?,
        )?;
        let expected_output = parse_crpa_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/crpa.dat",
        )?)?)?;
        let expected_log = parse_module_log_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/logscrn.dat",
        )?)?)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_crpa_dat(temp.path().join("crpa.dat"))?,
            expected_output
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logscrn.dat"))?,
            expected_log
        );
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

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                " Calculating Hubbard U.".to_string(),
                " Done with Hubbard U calculation.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn reference_crpa_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/CRPA/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }
}
