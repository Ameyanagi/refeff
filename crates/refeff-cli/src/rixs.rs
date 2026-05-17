use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    RixsInput, RixsLineData, RixsMapData, read_rixs_line, read_rixs_map, write_rixs_line,
    write_rixs_map,
};

use crate::work_dir_for_input;

/// Run the supported FEFF RIXS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF RIXS run can be satisfied from existing spectrum caches.
pub(crate) fn has_cached_rixs_output(work_dir: &Path) -> Result<bool> {
    if cached_output_paths(work_dir)?.is_empty() {
        return Ok(false);
    }
    Ok(rixs_enabled(&read_input(work_dir)?))
}

/// Run the FEFF RIXS cached-output path from existing map and line spectra.
///
/// The RIXS numerical solver is still unported. This preserves cached FEFF
/// RIXS output directories by validating and re-rendering typed two-axis maps
/// (`rixsET*`, `rixsEE*`, `rixs0.dat`, `rixs1.dat`) and one-axis line spectra
/// (`herfd*`, `xasEI*`, `xasEF*`, `xas0.dat`, `xas1.dat`).
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !rixs_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        bail!("RIXS spectrum generation requires the unported RIXS numerical solver");
    }

    for output in &outputs {
        match output.kind {
            CachedOutputKind::Map => {
                let data = read_rixs_map(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rixs_map_cache(&output.path, &data)?;
            }
            CachedOutputKind::Line => {
                let data = read_rixs_line(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rixs_line_cache(&output.path, &data)?;
            }
        }
    }

    Ok(outputs.len())
}

fn rixs_enabled(input: &RixsInput) -> bool {
    input.run
}

fn read_input(work_dir: &Path) -> Result<RixsInput> {
    let input_path = work_dir.join("rixs.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    RixsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_rixs_map_cache(path: &Path, data: &RixsMapData) -> Result<()> {
    write_rixs_map(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_rixs_line_cache(path: &Path, data: &RixsLineData) -> Result<()> {
    write_rixs_line(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Map,
    Line,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedOutputPath {
    path: PathBuf,
    kind: CachedOutputKind,
}

fn cached_output_paths(work_dir: &Path) -> Result<Vec<CachedOutputPath>> {
    let mut outputs = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", work_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let kind = if is_rixs_map_name(name) {
            Some(CachedOutputKind::Map)
        } else if is_rixs_line_name(name) {
            Some(CachedOutputKind::Line)
        } else {
            None
        };
        if let Some(kind) = kind {
            outputs.push(CachedOutputPath {
                path: entry.path(),
                kind,
            });
        }
    }

    outputs.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(outputs)
}

fn is_rixs_map_name(name: &str) -> bool {
    matches!(
        name,
        "rixsET.dat"
            | "rixsET-sat.dat"
            | "rixsEE.dat"
            | "rixsEE-sat.dat"
            | "rixsEI.dat"
            | "rixsEI-sat.dat"
            | "rixs0.dat"
            | "rixs1.dat"
    )
}

fn is_rixs_line_name(name: &str) -> bool {
    matches!(
        name,
        "herfd.dat"
            | "herfd-sat.dat"
            | "xasEI.dat"
            | "xasEI-sat.dat"
            | "xasEF.dat"
            | "xasEF-sat.dat"
            | "xas0.dat"
            | "xas1.dat"
    )
}

#[cfg(test)]
mod tests {
    use super::{has_cached_rixs_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2};
    use refeff_io::{
        RixsBroadening, RixsEnergyWindow, RixsInput, RixsLineData, RixsMapData, RixsSwitches,
        read_rixs_line, read_rixs_map, rixs_input_string, write_rixs_line, write_rixs_map,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn rixs_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), false)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_rixs_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn rixs_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RIXS should require the numerical solver")?;

        assert!(
            error
                .to_string()
                .contains("RIXS spectrum generation requires the unported RIXS numerical solver")
        );
        Ok(())
    }

    #[test]
    fn rixs_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rixs_input(temp.path(), true)?;
        write_rixs_map(temp.path().join("rixsET.dat"), &sample_rixs_map())?;
        write_rixs_map(temp.path().join("rixsEE-sat.dat"), &sample_rixs_map())?;
        write_rixs_line(temp.path().join("herfd.dat"), &sample_rixs_line())?;
        write_rixs_line(temp.path().join("xasEF-sat.dat"), &sample_rixs_line())?;

        let expected_map = read_rixs_map(temp.path().join("rixsET.dat"))?;
        let expected_map_sat = read_rixs_map(temp.path().join("rixsEE-sat.dat"))?;
        let expected_line = read_rixs_line(temp.path().join("herfd.dat"))?;
        let expected_line_sat = read_rixs_line(temp.path().join("xasEF-sat.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(has_cached_rixs_output(temp.path())?);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, expected_map);
        assert_eq!(
            read_rixs_map(temp.path().join("rixsEE-sat.dat"))?,
            expected_map_sat
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            expected_line
        );
        assert_eq!(
            read_rixs_line(temp.path().join("xasEF-sat.dat"))?,
            expected_line_sat
        );
        Ok(())
    }

    #[test]
    fn rixs_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_rixs_dir()? else {
            eprintln!("skipping RIXS reference test; generated RIXS reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("rixs.inp"), temp.path().join("rixs.inp"))?;
        std::fs::copy(
            reference_dir.join("referencerixsET.dat"),
            temp.path().join("rixsET.dat"),
        )?;
        std::fs::copy(
            reference_dir.join("referenceherfd.dat"),
            temp.path().join("herfd.dat"),
        )?;
        std::fs::copy(
            reference_dir.join("referenceherfd-sat.dat"),
            temp.path().join("herfd-sat.dat"),
        )?;

        let expected_map = read_rixs_map(temp.path().join("rixsET.dat"))?;
        let expected_herfd = read_rixs_line(temp.path().join("herfd.dat"))?;
        let expected_herfd_sat = read_rixs_line(temp.path().join("herfd-sat.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_rixs_map(temp.path().join("rixsET.dat"))?, expected_map);
        assert_eq!(
            read_rixs_line(temp.path().join("herfd.dat"))?,
            expected_herfd
        );
        assert_eq!(
            read_rixs_line(temp.path().join("herfd-sat.dat"))?,
            expected_herfd_sat
        );
        Ok(())
    }

    fn write_rixs_input(work_dir: &Path, run: bool) -> Result<()> {
        let input = RixsInput {
            run,
            broadening: RixsBroadening {
                gam_ch: 0.000_135_051_2,
                gam_exp_1: 0.000_135_051_2,
                gam_exp_2: 0.000_135_051_2,
            },
            energy_window: RixsEnergyWindow {
                emin_i: 0.0,
                emax_i: 0.0,
                emin_f: 0.0,
                emax_f: 0.0,
            },
            xmu: -367_493_090.027_428_2,
            switches: RixsSwitches {
                read_poles: true,
                skip_calc: false,
                mbconv: true,
                read_sigma: false,
            },
            edges: vec!["L3".to_string(), "VAL".to_string()],
        };
        std::fs::write(work_dir.join("rixs.inp"), rixs_input_string(&input)?)?;
        Ok(())
    }

    fn sample_rixs_map() -> RixsMapData {
        RixsMapData {
            header_lines: vec!["# sample RIXS map".to_string()],
            block_lengths: vec![2, 2],
            first_energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0, 11_540.0, 11_541.0]),
            second_energy_ev: Array1::from_vec(vec![-15.0, -15.0, -14.0, -14.0]),
            channels: Array2::from_shape_fn((4, 2), |(row, channel)| {
                1.0e-6 * (row + 1) as f64 + 2.0e-7 * channel as f64
            }),
        }
    }

    fn sample_rixs_line() -> RixsLineData {
        RixsLineData {
            header_lines: vec!["# sample RIXS line".to_string()],
            energy_ev: Array1::from_vec(vec![11_540.0, 11_541.0, 11_542.0]),
            channels: Array2::from_shape_fn((3, 2), |(row, channel)| {
                2.5e-6 * (row + 1) as f64 + 1.0e-7 * channel as f64
            }),
        }
    }

    fn reference_rixs_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/RIXS");
        let required = [
            "rixs.inp",
            "referencerixsET.dat",
            "referenceherfd.dat",
            "referenceherfd-sat.dat",
        ];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
