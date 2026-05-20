use std::path::Path;

use anyhow::{Context, Result, bail};
use refeff_io::{
    GlobalInput, MdffDatData, MdffInput, read_mdff_dat, read_module_log_dat, write_mdff_dat,
    write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF EELS-MDFF cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF EELS-MDFF run can be satisfied from an existing `mdff.dat` cache.
pub(crate) fn has_cached_mdff_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("mdff.dat").is_file() {
        return Ok(false);
    }
    if !global_requests_mdff(work_dir)? {
        return Ok(false);
    }
    Ok(work_dir.join("mdff.inp").is_file())
}

/// Run the FEFF EELS-MDFF cached-output path from an existing `mdff.dat`.
///
/// The EELS-MDFF numerical reducer is still unported. This path preserves the
/// module boundary for generated `mdff.inp` handoffs and validates cached
/// complex spectra so downstream compatibility tests can reuse FEFF output.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    if !global_requests_mdff(work_dir)? {
        return Ok(0);
    }

    read_input(work_dir)?;
    let output_path = work_dir.join("mdff.dat");
    if !output_path.is_file() {
        bail!("EELS-MDFF spectrum generation requires the unported EELS-MDFF numerical solver");
    }

    let data = read_mdff_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    let point_count = data.point_count();
    write_cached_output(&output_path, &data)?;
    write_optional_module_log(&work_dir.join("logmdff.dat"))?;
    Ok(point_count)
}

fn global_requests_mdff(work_dir: &Path) -> Result<bool> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(true);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let global = GlobalInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(global.q_control.imdff == 3)
}

fn read_input(work_dir: &Path) -> Result<MdffInput> {
    let input_path = work_dir.join("mdff.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    MdffInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &MdffDatData) -> Result<()> {
    write_mdff_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    let line_count = data.line_count();
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(line_count)
}

#[cfg(test)]
mod tests {
    use super::{has_cached_mdff_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;
    use refeff_io::{
        CfAverage, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, MdffDatData, MdffInput,
        ModuleLogData, global_input_string, mdff_input_string, read_mdff_dat, read_module_log_dat,
        write_mdff_dat, write_module_log_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn eelsmdff_module_skips_non_mdff_global_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 0)?;
        write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_mdff_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn eelsmdff_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_mdff_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled EELS-MDFF should require the numerical solver")?;

        assert!(error.to_string().contains(
            "EELS-MDFF spectrum generation requires the unported EELS-MDFF numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn eelsmdff_module_roundtrips_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        let expected = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;
        write_module_log_dat(temp.path().join("logmdff.dat"), &sample_module_log())?;
        let expected_log = read_module_log_dat(temp.path().join("logmdff.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert!(has_cached_mdff_output(temp.path())?);
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("logmdff.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn eelsmdff_module_checks_generated_reference_when_present() -> Result<()> {
        let Some(rdinp) = reference_rdinp()? else {
            eprintln!("skipping EELS-MDFF reference test; FEFF10 rdinp not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("feff.inp"), reference_mdff_input())?;
        let output = Command::new(rdinp).current_dir(temp.path()).output()?;
        if !output.status.success() {
            anyhow::bail!(
                "FEFF10 rdinp failed for EELS-MDFF reference input\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let global_text = std::fs::read_to_string(temp.path().join("global.inp"))?;
        let global = GlobalInput::parse_str(temp.path().join("global.inp"), &global_text)?;

        assert_eq!(global.q_control.imdff, 3);
        assert!(!temp.path().join("mdff.dat").exists());
        assert!(!has_cached_mdff_output(temp.path())?);
        Ok(())
    }

    fn write_mdff_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("mdff.inp"),
            mdff_input_string(&MdffInput {
                task: 1,
                q_input: 2,
            })?,
        )?;
        Ok(())
    }

    fn write_global_input(work_dir: &Path, imdff: i32) -> Result<()> {
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&GlobalInput {
                cfaverage: CfAverage {
                    nabs: 1,
                    iphabs: 0,
                    rclabs: 0.0,
                },
                control: GlobalControl {
                    ipol: 0,
                    ispin: 0,
                    le2: 0,
                    elpty: 0.0,
                    angks: 0.0,
                    l2lp: 0,
                    do_nrixs: 0,
                    ldecmx: -1,
                    lj: -1,
                },
                evec: [0.0; 3],
                xivec: [0.0, 1.0, 0.0],
                spvec: [0.0; 3],
                polarization_tensor: [[0.0; 6]; 3],
                norms: GlobalNorms {
                    evnorm: 0.0,
                    xivnorm: 1.0,
                    spvnorm: 0.0,
                },
                q_control: GlobalQControl {
                    nq: 0,
                    imdff,
                    qaverage: true,
                    mixdff: false,
                },
                q_vectors: Vec::new(),
                mdff: None,
            })?,
        )?;
        Ok(())
    }

    fn sample_mdff_dat() -> Result<MdffDatData> {
        Ok(MdffDatData {
            header_lines: vec![
                "# Orientation sensitive EELS calculation - beam energy =    300keV".to_string(),
                "#  Energy       total".to_string(),
            ],
            energy_loss_ev: Array1::from_vec(vec![10.0, 12.5]),
            spectrum: Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.25),
                    Complex64::new(0.5, -0.1),
                    Complex64::new(1.2, 0.2),
                    Complex64::new(0.8, -0.05),
                ],
            )?,
        })
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Starting MDFF module.".to_string(),
                "Module mdff is finished.  Exiting.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn reference_rdinp() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        for candidate in [
            workspace.join("feff10/bin/Seq/rdinp"),
            workspace.join("feff10/bin/rdinp"),
        ] {
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }

    fn reference_mdff_input() -> &'static str {
        r#"
TITLE Cu EELS-MDFF reference handoff
ELNES
300
0 1 0
2.4 0.0
5 3
0.0 0.0
MDFF 3
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#
    }
}
