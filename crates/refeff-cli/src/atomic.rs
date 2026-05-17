use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    PotInput, read_apot_bin, read_config_dat, read_fpf0_dat, read_module_log_dat, write_apot_bin,
    write_config_dat, write_fpf0_dat, write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF `ATOM` cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF `ATOM` run can be satisfied from an existing `apot.bin`.
pub(crate) fn has_cached_atomic_output(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.apot_bin.is_file() {
        return Ok(false);
    }
    Ok(atomic_enabled(&read_input(work_dir)?))
}

/// Run FEFF `ATOM` compatibility from existing atomic-potential caches.
///
/// The atomic-potential numerical solver is still unported. This preserves the
/// FEFF module boundary for directories that already contain `apot.bin`, and
/// validates optional `config.dat`, `fpf0.dat`, and `log1.dat` handoff files
/// that FEFF writes around the atomic-potential stage.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    if !caches.apot_bin.is_file() {
        bail!("ATOM atomic-potential generation requires the unported ATOM numerical solver");
    }

    let apot = read_apot_bin(&caches.apot_bin)
        .with_context(|| format!("failed to read {}", caches.apot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;

    let mut written = 1_usize;
    written += write_optional_config(&caches.config_dat)?;
    written += write_optional_fpf0(&caches.fpf0_dat)?;
    written += write_optional_module_log(&caches.log1_dat)?;
    Ok(written)
}

fn atomic_enabled(input: &PotInput) -> bool {
    input.control.mpot == 1
}

fn read_input(work_dir: &Path) -> Result<PotInput> {
    let input_path = work_dir.join("pot.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    PotInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_optional_config(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_config_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_config_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_optional_fpf0(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_fpf0_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_fpf0_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AtomicCachePaths {
    apot_bin: PathBuf,
    config_dat: PathBuf,
    fpf0_dat: PathBuf,
    log1_dat: PathBuf,
}

impl AtomicCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            apot_bin: work_dir.join("apot.bin"),
            config_dat: work_dir.join("config.dat"),
            fpf0_dat: work_dir.join("fpf0.dat"),
            log1_dat: work_dir.join("log1.dat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{has_cached_atomic_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2};
    use refeff_io::pot_bin::POT_BIN_RADIAL_POINTS;
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, ConfigDatData, ConfigDatPotential, FeffDocument, FeffInput, Fpf0DatData,
        Fpf0Oscillator, ModuleLogData, PotInput, rdinp, read_apot_bin, read_config_dat,
        read_fpf0_dat, read_module_log_dat, write_apot_bin, write_config_dat, write_fpf0_dat,
        write_module_log_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn atomic_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 0)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled ATOM should require the numerical solver")?;

        assert!(error.to_string().contains(
            "ATOM atomic-potential generation requires the unported ATOM numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn atomic_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        let apot_path = temp.path().join("apot.bin");
        let config_path = temp.path().join("config.dat");
        let fpf0_path = temp.path().join("fpf0.dat");
        let log_path = temp.path().join("log1.dat");
        write_apot_bin(&apot_path, &sample_apot_bin())?;
        write_config_dat(&config_path, &sample_config_dat())?;
        write_fpf0_dat(&fpf0_path, &sample_fpf0_dat())?;
        write_module_log_dat(&log_path, &sample_module_log())?;
        let expected_apot = read_apot_bin(&apot_path)?;
        let expected_config = read_config_dat(&config_path)?;
        let expected_fpf0 = read_fpf0_dat(&fpf0_path)?;
        let expected_log = read_module_log_dat(&log_path)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(has_cached_atomic_output(temp.path())?);
        assert_eq!(read_apot_bin(&apot_path)?, expected_apot);
        assert_eq!(read_config_dat(&config_path)?, expected_config);
        assert_eq!(read_fpf0_dat(&fpf0_path)?, expected_fpf0);
        assert_eq!(read_module_log_dat(&log_path)?, expected_log);
        Ok(())
    }

    #[test]
    fn atomic_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            eprintln!("skipping ATOM reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        for name in ["config.dat", "fpf0.dat", "log1.dat"] {
            let source = reference_dir.join(name);
            if source.is_file() {
                std::fs::copy(source, temp.path().join(name))?;
            }
        }
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        let expected_config = optional_config_dat(temp.path().join("config.dat"))?;
        let expected_fpf0 = optional_fpf0_dat(temp.path().join("fpf0.dat"))?;
        let expected_log = optional_module_log(temp.path().join("log1.dat"))?;

        let count = run_in_dir(temp.path())?;

        let optional_count = [
            expected_config.as_ref().map(|_| 1_usize),
            expected_fpf0.as_ref().map(|_| 1_usize),
            expected_log.as_ref().map(|_| 1_usize),
        ]
        .into_iter()
        .flatten()
        .sum::<usize>();
        assert_eq!(count, 1 + optional_count);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        if let Some(expected) = expected_config {
            assert_eq!(read_config_dat(temp.path().join("config.dat"))?, expected);
        }
        if let Some(expected) = expected_fpf0 {
            assert_eq!(read_fpf0_dat(temp.path().join("fpf0.dat"))?, expected);
        }
        if let Some(expected) = expected_log {
            assert_eq!(read_module_log_dat(temp.path().join("log1.dat"))?, expected);
        }
        Ok(())
    }

    fn write_pot_input(work_dir: &Path, mpot: i32) -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu atomic smoke test
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut pot_input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        pot_input.control.mpot = mpot;
        std::fs::write(
            work_dir.join("pot.inp"),
            refeff_io::pot_input_string(&pot_input)?,
        )?;
        Ok(())
    }

    fn sample_apot_bin() -> ApotBinData {
        ApotBinData {
            sections: vec![ApotBinSection {
                section_number: 8,
                headers: vec![
                    "rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
                ],
                header_texts: vec![
                    " rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
                ],
                column_labels: vec![],
                column_label_text: None,
                payload: ApotBinPayload::Matrix(ApotBinMatrix {
                    value_type: ApotBinType::Double,
                    values: ApotBinMatrixValues::Real(Array2::from_shape_fn(
                        (POT_BIN_RADIAL_POINTS, 2),
                        |(row, potential)| 0.015 * (row + 1) as f64 + 0.25 * potential as f64,
                    )),
                }),
                trailing_headers: vec![],
                trailing_header_texts: vec![],
            }],
        }
    }

    fn sample_config_dat() -> ConfigDatData {
        ConfigDatData {
            header_lines: Vec::new(),
            potentials: vec![ConfigDatPotential {
                potential_index: 0,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.1),
                valence_occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.01),
                spin_occupations: None,
            }],
        }
    }

    fn sample_fpf0_dat() -> Fpf0DatData {
        Fpf0DatData {
            atomic_number: 29,
            total_energy_fprime: -0.125,
            relativistic_correction: 0.075,
            oscillators: vec![Fpf0Oscillator {
                oscillator_strength: 1.25,
                excitation_energy: -8.98,
                orbital_index: 1,
            }],
            form_factor_momentum: Array1::from_vec(vec![0.0, 0.5, 1.0]),
            form_factor: Array1::from_vec(vec![29.0, 28.1, 25.7]),
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating atomic potentials ...".to_string(),
                "Done with module: atomic potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn optional_config_dat(path: impl AsRef<Path>) -> Result<Option<ConfigDatData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_config_dat(path)?));
        }
        Ok(None)
    }

    fn optional_fpf0_dat(path: impl AsRef<Path>) -> Result<Option<Fpf0DatData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_fpf0_dat(path)?));
        }
        Ok(None)
    }

    fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_module_log_dat(path)?));
        }
        Ok(None)
    }

    fn reference_atomic_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/Cu"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("apot.bin").is_file()))
    }
}
