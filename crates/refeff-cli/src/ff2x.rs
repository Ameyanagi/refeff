use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    ChiDatData, DanesDatData, Ff2xInput, XmuDatData, XmulDatData, read_chi_dat, read_danes_dat,
    read_xmu_dat, read_xmul_dat, write_chi_dat, write_danes_dat, write_xmu_dat, write_xmul_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF FF2X cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FF2X run can be satisfied from existing spectrum caches.
pub(crate) fn has_cached_ff2x_output(work_dir: &Path) -> Result<bool> {
    if cached_output_paths(work_dir)?.is_empty() {
        return Ok(false);
    }
    Ok(ff2x_enabled(&read_input(work_dir)?))
}

/// Run the FEFF FF2X cached-output path from existing final-spectrum files.
///
/// The FF2X spectrum assembler is still unported. This keeps cached FEFF final
/// spectra usable by validating and re-rendering typed `xmu.dat`,
/// `chi.dat`/`chipNNNN.dat`, `xmul.dat`, and `danes.dat` outputs.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !ff2x_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        bail!("FF2X spectrum generation requires the unported FF2X numerical solver");
    }

    for output in &outputs {
        match output.kind {
            CachedOutputKind::Xmu => {
                let data = read_xmu_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmu_cache(&output.path, &data)?;
            }
            CachedOutputKind::Chi => {
                let data = read_chi_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_chi_cache(&output.path, &data)?;
            }
            CachedOutputKind::Xmul => {
                let data = read_xmul_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmul_cache(&output.path, &data)?;
            }
            CachedOutputKind::Danes => {
                let data = read_danes_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_danes_cache(&output.path, &data)?;
            }
        }
    }

    Ok(outputs.len())
}

fn ff2x_enabled(input: &Ff2xInput) -> bool {
    input.control.mchi != 0
}

fn read_input(work_dir: &Path) -> Result<Ff2xInput> {
    let input_path = work_dir.join("ff2x.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    Ff2xInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_xmu_cache(path: &Path, data: &XmuDatData) -> Result<()> {
    write_xmu_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_chi_cache(path: &Path, data: &ChiDatData) -> Result<()> {
    write_chi_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xmul_cache(path: &Path, data: &XmulDatData) -> Result<()> {
    write_xmul_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_danes_cache(path: &Path, data: &DanesDatData) -> Result<()> {
    write_danes_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Xmu,
    Chi,
    Xmul,
    Danes,
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
        let kind = if name == "xmu.dat" {
            Some(CachedOutputKind::Xmu)
        } else if name == "chi.dat" || is_chip_dat_name(name) {
            Some(CachedOutputKind::Chi)
        } else if name == "xmul.dat" {
            Some(CachedOutputKind::Xmul)
        } else if name == "danes.dat" {
            Some(CachedOutputKind::Danes)
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

fn is_chip_dat_name(name: &str) -> bool {
    name.strip_prefix("chip")
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::{has_cached_ff2x_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::Array1;
    use refeff_io::{
        ChiDatData, DanesDatData, Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput, XmuDatData,
        ff2x_input_string, read_chi_dat, read_danes_dat, read_xmu_dat, write_chi_dat,
        write_danes_dat, write_xmu_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn ff2x_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ff2x_input(temp.path(), 0)?;
        write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_ff2x_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn ff2x_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ff2x_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled FF2X should require the numerical solver")?;

        assert!(
            error
                .to_string()
                .contains("FF2X spectrum generation requires the unported FF2X numerical solver")
        );
        Ok(())
    }

    #[test]
    fn ff2x_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ff2x_input(temp.path(), 1)?;
        let xmu = sample_xmu_dat();
        let chi = sample_chi_dat();
        let danes = sample_danes_dat();
        write_xmu_dat(temp.path().join("xmu.dat"), &xmu)?;
        write_chi_dat(temp.path().join("chi.dat"), &chi)?;
        write_danes_dat(temp.path().join("danes.dat"), &danes)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert!(has_cached_ff2x_output(temp.path())?);
        assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, xmu);
        assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, chi);
        assert_eq!(read_danes_dat(temp.path().join("danes.dat"))?, danes);
        Ok(())
    }

    #[test]
    fn ff2x_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_ff2x_dir()? else {
            eprintln!("skipping FF2X reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in ["ff2x.inp", "xmu.dat", "chi.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
        let expected_chi = read_chi_dat(temp.path().join("chi.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
        assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, expected_chi);
        Ok(())
    }

    fn write_ff2x_input(work_dir: &Path, mchi: i32) -> Result<()> {
        let input = Ff2xInput {
            control: Ff2xControl {
                mchi,
                ispec: 0,
                idwopt: 0,
                ipr6: 0,
                mbconv: 0,
                absolu: 0,
                i_gamma_ch: 0,
            },
            corrections: Ff2xCorrections {
                vrcorr: 0.0,
                vicorr: 0.0,
                s02: 1.0,
                critcw: 4.0,
            },
            debye: Ff2xDebye {
                tk: 190.0,
                thetad: 315.0,
                alphat: 0.0,
                thetae: 0.0,
                sig2g: 0.0,
                sig_gk: 0.0,
            },
            momentum_transfer: [0.0, 0.0, 0.0],
            decomposition_channels: -1,
            electronic_temperature: 0.0,
        };
        std::fs::write(work_dir.join("ff2x.inp"), ff2x_input_string(&input)?)?;
        Ok(())
    }

    fn sample_xmu_dat() -> XmuDatData {
        XmuDatData {
            header_lines: vec![
                "# # Cu                                                           FEFF 10.0"
                    .to_string(),
                "# xsedge+ 50, used to normalize mu           1.234500E+00".to_string(),
            ],
            normalization: Some(1.2345),
            photon_energy_ev: Array1::from_vec(vec![8979.0, 8980.0, 8981.0]),
            relative_energy_ev: Array1::from_vec(vec![0.0, 1.0, 2.0]),
            wave_number: Array1::from_vec(vec![0.0, 0.512, 0.724]),
            mu: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            mu0: Array1::from_vec(vec![0.9, 0.95, 1.0]),
            chi: Array1::from_vec(vec![0.1, 0.15, 0.2]),
        }
    }

    fn sample_chi_dat() -> ChiDatData {
        ChiDatData {
            header_lines: vec![
                "# # Cu                                                           FEFF 10.0"
                    .to_string(),
                "#       k          chi          mag           phase @#".to_string(),
            ],
            wave_number: Array1::from_vec(vec![0.0, 0.05, 0.1]),
            chi: Array1::from_vec(vec![-0.115_938_3, -0.119_413_8, -0.122_912_6]),
            magnitude: Array1::from_vec(vec![0.270_227_8, 0.272_670_8, 0.275_083_6]),
            phase: Array1::from_vec(vec![-2.698_164, -2.688_285, -2.678_386]),
            phase_minus_2kr: None,
            ckp_real: None,
            ckp_imag: None,
        }
    }

    fn sample_danes_dat() -> DanesDatData {
        DanesDatData {
            header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
            energy_ev: Array1::from_vec(vec![-18.690, -17.122, -15.703]),
            matsubara: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            sommerfeld: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            anomalous: Array1::from_vec(vec![10.097, 10.603, 11.159]),
            tail: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
            total: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
            difference: Array1::from_vec(vec![-5.4576, -5.6591, -5.8651]),
        }
    }

    fn reference_ff2x_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["ff2x.inp", "xmu.dat", "chi.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
