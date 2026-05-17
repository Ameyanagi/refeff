use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{ScreenInput, WscrnDatData, read_wscrn_dat, write_wscrn_dat};

use crate::work_dir_for_input;

/// Run the supported FEFF SCREEN cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF SCREEN run can be satisfied from an existing `wscrn.dat`.
pub(crate) fn has_cached_screen_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("wscrn.dat").is_file() {
        return Ok(false);
    }
    read_input(work_dir)?;
    Ok(true)
}

/// Run the FEFF SCREEN cached-output path from an existing `wscrn.dat`.
///
/// The screened-core-hole solver is still unported. This path preserves the
/// module boundary for existing FEFF caches by validating and re-rendering the
/// radial screened-potential table.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    read_input(work_dir)?;

    let output_path = work_dir.join("wscrn.dat");
    if !output_path.is_file() {
        bail!("SCREEN screened-core-hole generation requires the unported SCREEN numerical solver");
    }

    let data = read_wscrn_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let row_count = data.row_count();
    write_cached_output(&output_path, &data)?;
    Ok(row_count)
}

fn read_input(work_dir: &Path) -> Result<ScreenInput> {
    let input_path = work_dir.join("screen.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    ScreenInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &WscrnDatData) -> Result<()> {
    write_wscrn_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::run_in_dir;
    use anyhow::{Context, Result};
    use ndarray::array;
    use refeff_io::{WscrnDatData, rdinp, read_wscrn_dat, write_wscrn_dat};
    use std::path::Path;

    #[test]
    fn screen_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN should require the numerical solver without wscrn.dat")?;

        assert!(error.to_string().contains(
            "SCREEN screened-core-hole generation requires the unported SCREEN numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn screen_module_roundtrips_cached_wscrn_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let expected = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_wscrn_dat(temp.path().join("wscrn.dat"))?, expected);
        Ok(())
    }

    fn write_screen_input(work_dir: &Path) -> Result<()> {
        std::fs::write(work_dir.join("screen.inp"), rdinp::screen_inp_string())?;
        Ok(())
    }

    fn sample_wscrn_dat() -> WscrnDatData {
        WscrnDatData {
            header_lines: vec![" # r       w_scrn(r)      v_ch(r)".to_string()],
            radius_bohr: array![
                0.150_733_046_3E-03,
                0.158_461_294_9E-03,
                0.166_585_779_2E-03
            ],
            screened_potential: array![
                0.267_288_234_6E+02,
                0.267_288_167_8E+02,
                0.267_288_030_6E+02
            ],
            core_hole_potential: array![
                0.291_616_524_4E+02,
                0.291_616_457_6E+02,
                0.291_616_320_4E+02
            ],
        }
    }
}
