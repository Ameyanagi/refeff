use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::SfconvInput;

use crate::work_dir_for_input;

/// Run FEFF `SFCONV` startup behavior beside the requested input.
///
/// The full `SO2CONV` spectral-function convolution is still unported. This
/// function preserves the FEFF module boundary for disabled SFCONV inputs by
/// parsing `sfconv.inp` and creating/truncating `logsfconv.dat`.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Run the supported disabled `SFCONV` path from an existing `sfconv.inp`.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let log_path = work_dir.join("logsfconv.dat");
    std::fs::write(&log_path, "")
        .with_context(|| format!("failed to write {}", log_path.display()))?;

    if input.control.msfconv == 1 {
        bail!("SFCONV S0^2 convolution requires the unported SO2CONV driver");
    }

    Ok(0)
}

fn read_input(work_dir: &Path) -> Result<SfconvInput> {
    let input_path = work_dir.join("sfconv.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    SfconvInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

#[cfg(test)]
mod tests {
    use super::{run_for_input, run_in_dir};
    use anyhow::{Context, Result};
    use std::path::Path;

    #[test]
    fn sfconv_module_writes_empty_log_when_disabled() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_sfconv_input(temp.path(), 0)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert_eq!(
            std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
            ""
        );
        Ok(())
    }

    #[test]
    fn sfconv_module_uses_input_parent_directory() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_sfconv_input(temp.path(), 0)?;

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 0);
        assert!(temp.path().join("logsfconv.dat").is_file());
        Ok(())
    }

    #[test]
    fn sfconv_module_rejects_enabled_convolution_until_so2conv_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_sfconv_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled SFCONV should require the SO2CONV driver")?;

        assert!(
            error
                .to_string()
                .contains("S0^2 convolution requires the unported SO2CONV driver")
        );
        assert_eq!(
            std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
            ""
        );
        Ok(())
    }

    fn write_sfconv_input(work_dir: &Path, msfconv: i32) -> Result<()> {
        std::fs::write(
            work_dir.join("sfconv.inp"),
            format!(
                concat!(
                    "msfconv, ipse, ipsk\n",
                    "{:4}{:4}{:4}\n",
                    "wsigk, cen\n",
                    "{:13.5}{:13.5}\n",
                    "ispec, ipr6\n",
                    "{:4}{:4}\n",
                    "cfname\n",
                    "NULL        \n",
                ),
                msfconv, 0, 0, 0.0, 0.0, 0, 0
            ),
        )?;
        Ok(())
    }
}
