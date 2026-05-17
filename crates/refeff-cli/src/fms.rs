use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    FmsBinData, FmsInput, FmslBinData, GgDatData, GtrBinData, GtrDatData, GtrlDatData,
    read_fms_bin, read_fmsl_bin, read_gg_bin, read_gg_dat, read_gtr_bin, read_gtr_dat,
    read_gtrl_dat, write_fms_bin, write_fmsl_bin, write_gg_bin, write_gg_dat, write_gtr_bin,
    write_gtr_dat, write_gtrl_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF FMS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FMS/MKGTR run can be satisfied from existing caches.
pub(crate) fn has_cached_fms_output(work_dir: &Path) -> Result<bool> {
    if cached_output_paths(work_dir)?.is_empty() {
        return Ok(false);
    }
    Ok(fms_enabled(&read_input(work_dir)?))
}

/// Run the FEFF FMS/MKGTR cached-output path from existing handoff files.
///
/// The full multiple-scattering solver and Green's-function trace builder are
/// still unported. This preserves cached FEFF directories by validating and
/// re-rendering typed `gg.bin`/`gg.dat`, `fms.bin`, `fmsl.bin`, `gtr.dat`,
/// `gtrNN.bin`, and `gtrl.dat` handoffs.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !fms_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        bail!("FMS Green's-function generation requires the unported FMS numerical solver");
    }

    let fms_metadata = if outputs
        .iter()
        .any(|output| output.kind == CachedOutputKind::FmslBin)
    {
        let fms_path = work_dir.join("fms.bin");
        Some(
            read_fms_bin(&fms_path)
                .with_context(|| format!("failed to read {}", fms_path.display()))?,
        )
    } else {
        None
    };

    for output in &outputs {
        match output.kind {
            CachedOutputKind::FmsBin => {
                let data = read_fms_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_fms_cache(&output.path, &data)?;
            }
            CachedOutputKind::FmslBin => {
                let metadata = fms_metadata
                    .as_ref()
                    .context("fmsl.bin cache requires fms.bin metadata")?;
                let max_channel = decomposition_channel(&input)?;
                let data = read_fmsl_bin(
                    &output.path,
                    metadata.pad_width,
                    metadata.energy_count,
                    max_channel,
                )
                .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_fmsl_cache(&output.path, &data)?;
            }
            CachedOutputKind::GgBin => {
                let data = read_gg_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gg_bin_cache(&output.path, &data)?;
            }
            CachedOutputKind::GgDat => {
                let data = read_gg_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gg_dat_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrBin => {
                let data = read_gtr_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtr_bin_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrDat => {
                let data = read_gtr_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtr_dat_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrlDat => {
                let data = read_gtrl_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtrl_dat_cache(&output.path, &data)?;
            }
        }
    }

    Ok(outputs.len())
}

fn fms_enabled(input: &FmsInput) -> bool {
    input.control.mfms != 0
}

fn decomposition_channel(input: &FmsInput) -> Result<usize> {
    if input.decomposition_channels < 0 {
        bail!("fmsl.bin cache requires a nonnegative FMS decomposition channel count");
    }
    Ok(input.decomposition_channels as usize)
}

fn read_input(work_dir: &Path) -> Result<FmsInput> {
    let input_path = work_dir.join("fms.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    FmsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_fms_cache(path: &Path, data: &FmsBinData) -> Result<()> {
    write_fms_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_fmsl_cache(path: &Path, data: &FmslBinData) -> Result<()> {
    write_fmsl_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gg_bin_cache(path: &Path, data: &GgDatData) -> Result<()> {
    write_gg_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gg_dat_cache(path: &Path, data: &GgDatData) -> Result<()> {
    write_gg_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtr_bin_cache(path: &Path, data: &GtrBinData) -> Result<()> {
    write_gtr_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtr_dat_cache(path: &Path, data: &GtrDatData) -> Result<()> {
    write_gtr_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtrl_dat_cache(path: &Path, data: &GtrlDatData) -> Result<()> {
    write_gtrl_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    FmsBin,
    FmslBin,
    GgBin,
    GgDat,
    GtrBin,
    GtrDat,
    GtrlDat,
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
        let kind = if name == "fms.bin" {
            Some(CachedOutputKind::FmsBin)
        } else if name == "fmsl.bin" {
            Some(CachedOutputKind::FmslBin)
        } else if name == "gg.bin" {
            Some(CachedOutputKind::GgBin)
        } else if name == "gg.dat" {
            Some(CachedOutputKind::GgDat)
        } else if name == "gtr.dat" {
            Some(CachedOutputKind::GtrDat)
        } else if name == "gtrl.dat" {
            Some(CachedOutputKind::GtrlDat)
        } else if is_gtr_bin_name(name) {
            Some(CachedOutputKind::GtrBin)
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

fn is_gtr_bin_name(name: &str) -> bool {
    name.strip_prefix("gtr")
        .and_then(|tail| tail.strip_suffix(".bin"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_fms_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex64;
    use refeff_io::{
        FmsBinData, FmsCluster, FmsControl, FmsDebye, FmsInput, FmslBinData, GgDatData,
        GgDatSection, GtrBinData, GtrDatData, GtrlDatData, fms_input_string, parse_gtrl_dat,
        read_fms_bin, read_fmsl_bin, read_gg_bin, read_gg_dat, read_gtr_bin, read_gtr_dat,
        read_gtrl_dat, write_fms_bin, write_fmsl_bin, write_gg_bin, write_gg_dat, write_gtr_bin,
        write_gtr_dat, write_gtrl_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn fms_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 0, -1)?;
        write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_fms_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn fms_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 1, -1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled FMS should require the numerical solver")?;

        assert!(error.to_string().contains(
            "FMS Green's-function generation requires the unported FMS numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn fms_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 1, 2)?;
        write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;
        write_fmsl_bin(temp.path().join("fmsl.bin"), &sample_fmsl_bin())?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_gg_dat())?;
        write_gg_dat(temp.path().join("gg.dat"), &sample_gg_dat())?;
        write_gtr_dat(temp.path().join("gtr.dat"), &sample_gtr_dat())?;
        write_gtr_bin(temp.path().join("gtr00.bin"), &sample_gtr_bin())?;
        write_gtrl_dat(temp.path().join("gtrl.dat"), &sample_gtrl_dat()?)?;

        let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let expected_fmsl = read_fmsl_bin(
            temp.path().join("fmsl.bin"),
            expected_fms.pad_width,
            expected_fms.energy_count,
            2,
        )?;
        let expected_gg_bin = read_gg_bin(temp.path().join("gg.bin"))?;
        let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
        let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
        let expected_gtr_bin = read_gtr_bin(temp.path().join("gtr00.bin"))?;
        let expected_gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 7);
        assert!(has_cached_fms_output(temp.path())?);
        assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
        assert_eq!(
            read_fmsl_bin(
                temp.path().join("fmsl.bin"),
                expected_fms.pad_width,
                expected_fms.energy_count,
                2,
            )?,
            expected_fmsl
        );
        assert_eq!(read_gg_bin(temp.path().join("gg.bin"))?, expected_gg_bin);
        assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
        assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
        assert_eq!(
            read_gtr_bin(temp.path().join("gtr00.bin"))?,
            expected_gtr_bin
        );
        assert_eq!(read_gtrl_dat(temp.path().join("gtrl.dat"))?, expected_gtrl);
        Ok(())
    }

    #[test]
    fn fms_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_fms_dir()? else {
            eprintln!("skipping FMS reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
        for name in required {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        for name in ["gg.bin", "gtrl.dat", "fmsl.bin"] {
            let source = reference_dir.join(name);
            if source.is_file() {
                std::fs::copy(source, temp.path().join(name))?;
            }
        }
        copy_gtr_bin_references(&reference_dir, temp.path())?;

        let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
        let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert!(count >= required.len() - 1);
        assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
        assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
        assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
        Ok(())
    }

    fn write_fms_input(work_dir: &Path, mfms: i32, decomposition_channels: i32) -> Result<()> {
        let input = FmsInput {
            control: FmsControl {
                mfms,
                idwopt: 0,
                minv: 0,
            },
            cluster: FmsCluster {
                rfms2: -1.0,
                rdirec: -1.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: FmsDebye {
                tk: 190.0,
                thetad: 315.0,
                sig2g: 0.0,
            },
            lmaxph: vec![2, 2],
            decomposition_channels,
            save_gg_slice: false,
            do_fms: 0,
        };
        std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
        Ok(())
    }

    fn sample_fms_bin() -> FmsBinData {
        FmsBinData {
            cluster_radius_angstrom: 5.5,
            energy_count: 2,
            main_energy_count: 1,
            auxiliary_energy_count: 0,
            highest_potential_index: 1,
            pad_width: 8,
            declared_spectrum_count: Some(2),
            spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
                Complex64::new(
                    0.25 * (energy + 1) as f64 + spectrum as f64,
                    -0.05 * (energy + 1) as f64 - spectrum as f64,
                )
            }),
        }
    }

    fn sample_fmsl_bin() -> FmslBinData {
        FmslBinData {
            pad_width: 8,
            max_decomposition_channel: 2,
            traces: Array3::from_shape_fn((2, 3, 3), |(energy, lg2, lg1)| {
                Complex64::new(
                    energy as f64 + 0.1 * lg2 as f64 + 0.01 * lg1 as f64,
                    -(energy as f64) - 0.2 * lg2 as f64 - 0.02 * lg1 as f64,
                )
            }),
        }
    }

    fn sample_gg_dat() -> GgDatData {
        GgDatData {
            sections: vec![
                GgDatSection {
                    section_number: 1,
                    values: Array2::from_shape_fn((2, 2), |(row, column)| {
                        let value = 1.0 + row as f64 + 2.0 * column as f64;
                        Complex64::new(value, -0.5 * value)
                    }),
                    raw_prefix_lines: None,
                },
                GgDatSection {
                    section_number: 2,
                    values: Array2::from_shape_fn((1, 2), |(_, column)| {
                        let value = 5.0 + column as f64;
                        Complex64::new(value, -value - 0.5)
                    }),
                    raw_prefix_lines: None,
                },
            ],
        }
    }

    fn sample_gtr_dat() -> GtrDatData {
        GtrDatData {
            energy: Array1::from_vec(vec![
                Complex64::new(-0.138_801, 0.031_773),
                Complex64::new(-0.137_401, 0.031_773),
                Complex64::new(55.866_911, 0.031_773),
            ]),
            trace: Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.624_106, 1.081_113),
            ]),
        }
    }

    fn sample_gtr_bin() -> GtrBinData {
        GtrBinData {
            point_count_declared: 2,
            horizontal_count: 1,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            values: Array3::from_shape_fn((2, 2, 2), |(energy, potential, angular)| {
                let value = energy as f64 + 0.1 * potential as f64 + 0.01 * angular as f64;
                Complex64::new(value, -value)
            }),
        }
    }

    fn sample_gtrl_dat() -> Result<GtrlDatData> {
        Ok(parse_gtrl_dat(
            r#"    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01
    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01
"#,
        )?)
    }

    fn copy_gtr_bin_references(source_dir: &Path, target_dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(source_dir)
            .with_context(|| format!("failed to read {}", source_dir.display()))?
        {
            let entry = entry
                .with_context(|| format!("failed to read entry in {}", source_dir.display()))?;
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
            if super::is_gtr_bin_name(name) {
                std::fs::copy(entry.path(), target_dir.join(name))?;
            }
        }
        Ok(())
    }

    fn reference_fms_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
