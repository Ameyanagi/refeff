use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    BandInput, ModuleLogData, read_bandstructure_dat, read_kmesh_dat, read_module_log_dat,
    write_bandstructure_dat, write_kmesh_dat, write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF `BAND` cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF `BAND` run can be satisfied from an existing band cache.
pub(crate) fn has_cached_band_output(work_dir: &Path) -> Result<bool> {
    let caches = BandCachePaths::new(work_dir);
    if !caches.bandstructure_dat.is_file() {
        return Ok(false);
    }
    Ok(band_enabled(&read_input(work_dir)?))
}

/// Run FEFF `BAND` compatibility from existing band-structure caches.
///
/// The KKR band-structure numerical solver is still unported. This preserves
/// the FEFF module boundary for disabled BAND inputs and validates
/// `bandstructure.dat` plus optional `kmesh.dat` and `logband.dat` caches for
/// enabled runs.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let caches = BandCachePaths::new(work_dir);
    if !band_enabled(&input) {
        write_empty_module_log(&caches.logband_dat)?;
        return Ok(0);
    }

    if !caches.bandstructure_dat.is_file() {
        bail!("BAND band-structure generation requires the unported BAND numerical solver");
    }

    let data = read_bandstructure_dat(&caches.bandstructure_dat)
        .with_context(|| format!("failed to read {}", caches.bandstructure_dat.display()))?;
    write_bandstructure_dat(&caches.bandstructure_dat, &data)
        .with_context(|| format!("failed to write {}", caches.bandstructure_dat.display()))?;

    let mut written = 1_usize;
    written += write_optional_kmesh(&caches.kmesh_dat)?;
    written += write_optional_module_log(&caches.logband_dat)?;
    Ok(written)
}

fn band_enabled(input: &BandInput) -> bool {
    input.mband == 1
}

fn read_input(work_dir: &Path) -> Result<BandInput> {
    let input_path = work_dir.join("band.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    BandInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_optional_kmesh(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_kmesh_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_kmesh_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
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

fn write_empty_module_log(path: &Path) -> Result<()> {
    let data = ModuleLogData {
        lines: Vec::new(),
        line_terminators: Vec::new(),
    };
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BandCachePaths {
    bandstructure_dat: PathBuf,
    kmesh_dat: PathBuf,
    logband_dat: PathBuf,
}

impl BandCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            bandstructure_dat: work_dir.join("bandstructure.dat"),
            kmesh_dat: work_dir.join("kmesh.dat"),
            logband_dat: work_dir.join("logband.dat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{has_cached_band_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::Array1;
    use refeff_io::{
        BandEnergyMesh, BandInput, BandstructureDatData, BandstructureRow, KmeshDatData,
        KmeshMetadata, KmeshRow, ModuleLogData, band_input_string, read_bandstructure_dat,
        read_kmesh_dat, read_module_log_dat, write_bandstructure_dat, write_kmesh_dat,
        write_module_log_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn band_module_writes_empty_log_for_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_band_output(temp.path())?);
        assert!(read_module_log_dat(temp.path().join("logband.dat"))?.is_empty());
        Ok(())
    }

    #[test]
    fn band_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled BAND should require the numerical solver")?;

        assert!(error.to_string().contains(
            "BAND band-structure generation requires the unported BAND numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn band_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        let band_path = temp.path().join("bandstructure.dat");
        let kmesh_path = temp.path().join("kmesh.dat");
        let log_path = temp.path().join("logband.dat");
        write_bandstructure_dat(&band_path, &sample_bandstructure_dat())?;
        write_kmesh_dat(&kmesh_path, &sample_kmesh_dat())?;
        write_module_log_dat(&log_path, &sample_module_log())?;
        let expected_band = read_bandstructure_dat(&band_path)?;
        let expected_kmesh = read_kmesh_dat(&kmesh_path)?;
        let expected_log = read_module_log_dat(&log_path)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert!(has_cached_band_output(temp.path())?);
        assert_eq!(read_bandstructure_dat(&band_path)?, expected_band);
        assert_eq!(read_kmesh_dat(&kmesh_path)?, expected_kmesh);
        assert_eq!(read_module_log_dat(&log_path)?, expected_log);
        Ok(())
    }

    #[test]
    fn band_module_uses_disabled_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_band_dir()? else {
            eprintln!(
                "skipping BAND reference test; generated KSPACE/Graphite reference not found"
            );
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("band.inp"), temp.path().join("band.inp"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(read_module_log_dat(temp.path().join("logband.dat"))?.is_empty());
        Ok(())
    }

    fn write_band_input(work_dir: &Path, enabled: bool) -> Result<()> {
        let input = BandInput {
            mband: if enabled { 1 } else { 0 },
            energy_mesh: BandEnergyMesh {
                emin: -5.0,
                emax: 10.0,
                estep: 0.25,
            },
            nkp: 2,
            ikpath: 1,
            freeprop: false,
        };
        std::fs::write(work_dir.join("band.inp"), band_input_string(&input)?)?;
        Ok(())
    }

    fn sample_bandstructure_dat() -> BandstructureDatData {
        BandstructureDatData {
            header_lines: vec![
                " # grid of            2  k-points.".to_string(),
                " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000".to_string(),
                " # Found between            1  and            2  number of bands.".to_string(),
            ],
            rows: vec![
                BandstructureRow {
                    index: 1,
                    k_point: [0.0, 0.5, 0.25],
                    bands: Array1::from_vec(vec![-5.0, 1.25]),
                },
                BandstructureRow {
                    index: 2,
                    k_point: [0.5, 0.25, 0.0],
                    bands: Array1::from_vec(vec![0.75]),
                },
            ],
        }
    }

    fn sample_kmesh_dat() -> KmeshDatData {
        KmeshDatData {
            rows: vec![
                KmeshRow {
                    index: 1,
                    k_point: [0.0, 0.5, 0.25],
                    weight: 0.75,
                    metadata: Some(KmeshMetadata {
                        requested_points: 100,
                        irreducible_points: 2,
                        divisions: [4, 5, 6],
                    }),
                },
                KmeshRow {
                    index: 2,
                    k_point: [0.5, 0.25, 0.0],
                    weight: 0.25,
                    metadata: None,
                },
            ],
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating band structure ...".to_string(),
                " Done with module: band structure.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\r\n".to_string()],
        }
    }

    fn reference_band_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/KSPACE/Graphite"));
        Ok(reference.filter(|path| path.join("band.inp").is_file()))
    }
}
