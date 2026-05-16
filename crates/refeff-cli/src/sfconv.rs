use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{
    EelsInput, SfconvInput, SfconvSo2convTarget, read_list_dat, sfconv_so2conv_targets,
};

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
    let log_text = if input.control.msfconv == 1 {
        "Calculating S0^2 ...\n"
    } else {
        ""
    };
    std::fs::write(&log_path, log_text)
        .with_context(|| format!("failed to write {}", log_path.display()))?;

    if input.control.msfconv == 1 {
        let targets = so2conv_targets_for_dir(work_dir, &input)?;
        bail!(
            "SFCONV S0^2 convolution requires the unported SO2CONV numerical driver; discovered {} target file(s): {}",
            targets.len(),
            so2conv_target_summary(&targets)
        );
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

fn so2conv_targets_for_dir(
    work_dir: &Path,
    input: &SfconvInput,
) -> Result<Vec<SfconvSo2convTarget>> {
    let list = if input.spectrum.ispec.abs() == 0 && input.spectrum.ipr6 >= 2 {
        let list_path = work_dir.join("list.dat");
        Some(read_list_dat(&list_path).with_context(|| {
            format!(
                "failed to read SO2CONV path selection from {}",
                list_path.display()
            )
        })?)
    } else {
        None
    };
    let eels = read_optional_eels_input(work_dir)?;
    sfconv_so2conv_targets(input, list.as_ref(), eels.as_ref())
        .context("failed to select SO2CONV target files")
}

fn read_optional_eels_input(work_dir: &Path) -> Result<Option<EelsInput>> {
    let path = work_dir.join("eels.inp");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    EelsInput::parse_str(&path, &text)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn so2conv_target_summary(targets: &[SfconvSo2convTarget]) -> String {
    let names = targets
        .iter()
        .take(6)
        .map(|target| target.file_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if targets.len() > 6 {
        format!("{names}, ...")
    } else {
        names
    }
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
    fn sfconv_module_rejects_enabled_convolution_after_target_selection() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_sfconv_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled SFCONV should require the SO2CONV driver")?;

        assert!(
            error
                .to_string()
                .contains("S0^2 convolution requires the unported SO2CONV numerical driver")
        );
        assert!(error.to_string().contains("xmu.dat, chi.dat"));
        assert_eq!(
            std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
            "Calculating S0^2 ...\n"
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
