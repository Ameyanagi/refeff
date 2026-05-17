use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    FeffBinData, GenfmtInput, ListDatData, read_feff_bin, read_feffl_bin, read_list_dat,
    write_feff_bin, write_feffl_bin, write_list_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF GENFMT cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF GENFMT run can be satisfied from existing `feff.bin` caches.
pub(crate) fn has_cached_genfmt_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("feff.bin").is_file() || !work_dir.join("list.dat").is_file() {
        return Ok(false);
    }
    Ok(genfmt_enabled(&read_input(work_dir)?))
}

/// Run the FEFF GENFMT cached-output path from existing handoff files.
///
/// The GENFMT curved-wave path formatter is still unported. This keeps cached
/// FEFF output usable by validating and re-rendering `feff.bin`, suffixed
/// `feffNN.bin` files, `list.dat`, suffixed `listNN.dat` files, and optional
/// `feffl.bin` NRIXS decomposition caches.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !genfmt_enabled(&input) {
        return Ok(0);
    }

    let caches = cached_output_paths(work_dir)?;
    if !caches.has_required_base_outputs() {
        bail!("GENFMT path-format generation requires the unported GENFMT numerical solver");
    }

    let mut written = 0_usize;
    let mut base_feff = None;
    for path in caches.feff_bins {
        let data =
            read_feff_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if path.file_name().and_then(|name| name.to_str()) == Some("feff.bin") {
            base_feff = Some(data.clone());
        }
        write_feff_cache(&path, &data)?;
        written += 1;
    }

    for path in caches.list_dats {
        let data =
            read_list_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        write_list_cache(&path, &data)?;
        written += 1;
    }

    if let Some(path) = caches.feffl_bin {
        let base = base_feff.context("feffl.bin cache requires feff.bin metadata")?;
        let max_channel = decomposition_channel(&input)?;
        let data = read_feffl_bin(
            &path,
            base.pad_width,
            base.paths.len(),
            base.energy_count(),
            max_channel,
        )
        .with_context(|| format!("failed to read {}", path.display()))?;
        write_feffl_bin(&path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
    }

    Ok(written)
}

fn genfmt_enabled(input: &GenfmtInput) -> bool {
    input.control.mfeff != 0
}

fn decomposition_channel(input: &GenfmtInput) -> Result<usize> {
    if input.decomposition_channels < 0 {
        bail!("feffl.bin cache requires a nonnegative GENFMT decomposition channel count");
    }
    Ok(input.decomposition_channels as usize)
}

fn read_input(work_dir: &Path) -> Result<GenfmtInput> {
    let input_path = work_dir.join("genfmt.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    GenfmtInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_feff_cache(path: &Path, data: &FeffBinData) -> Result<()> {
    write_feff_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_list_cache(path: &Path, data: &ListDatData) -> Result<()> {
    write_list_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Default)]
struct GenfmtCachePaths {
    feff_bins: Vec<PathBuf>,
    list_dats: Vec<PathBuf>,
    feffl_bin: Option<PathBuf>,
}

impl GenfmtCachePaths {
    fn has_required_base_outputs(&self) -> bool {
        self.feff_bins
            .iter()
            .any(|path| file_name_is(path, "feff.bin"))
            && self
                .list_dats
                .iter()
                .any(|path| file_name_is(path, "list.dat"))
    }
}

fn cached_output_paths(work_dir: &Path) -> Result<GenfmtCachePaths> {
    let mut paths = GenfmtCachePaths::default();
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
        if is_feff_bin_cache_name(name) {
            paths.feff_bins.push(entry.path());
        } else if is_list_dat_cache_name(name) {
            paths.list_dats.push(entry.path());
        } else if name == "feffl.bin" {
            paths.feffl_bin = Some(entry.path());
        }
    }

    paths.feff_bins.sort();
    paths.list_dats.sort();
    Ok(paths)
}

fn is_feff_bin_cache_name(name: &str) -> bool {
    if name == "feff.bin" {
        return true;
    }
    name.strip_prefix("feff")
        .and_then(|tail| tail.strip_suffix(".bin"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn is_list_dat_cache_name(name: &str) -> bool {
    if name == "list.dat" {
        return true;
    }
    name.strip_prefix("list")
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

fn file_name_is(path: &Path, expected: &str) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some(expected)
}

#[cfg(test)]
mod tests {
    use super::{has_cached_genfmt_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;
    use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
    use refeff_io::{
        FeffBinData, FeffBinPath, FeffBinPotential, GenfmtControl, GenfmtInput, ListDatData,
        ListDatEntry, genfmt_input_string, read_feff_bin, read_list_dat, write_feff_bin,
        write_list_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn genfmt_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_genfmt_input(temp.path(), 0)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_genfmt_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn genfmt_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_genfmt_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled GENFMT should require the numerical solver")?;

        assert!(error.to_string().contains(
            "GENFMT path-format generation requires the unported GENFMT numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn genfmt_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_genfmt_input(temp.path(), 1)?;
        let feff = sample_feff_bin_data();
        let list = sample_list_dat();
        write_feff_bin(temp.path().join("feff.bin"), &feff)?;
        write_list_dat(temp.path().join("list.dat"), &list)?;
        let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
        let expected_list = read_list_dat(temp.path().join("list.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert!(has_cached_genfmt_output(temp.path())?);
        assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
        assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
        Ok(())
    }

    #[test]
    fn genfmt_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_genfmt_dir()? else {
            eprintln!("skipping GENFMT reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in ["genfmt.inp", "feff.bin", "list.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
        let expected_list = read_list_dat(temp.path().join("list.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
        assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
        Ok(())
    }

    fn write_genfmt_input(work_dir: &Path, mfeff: i32) -> Result<()> {
        let input = GenfmtInput {
            control: GenfmtControl {
                mfeff,
                ipr5: 0,
                iorder: 2,
                critcw: 4.0,
                wnstar: false,
            },
            decomposition_channels: -1,
        };
        std::fs::write(work_dir.join("genfmt.inp"), genfmt_input_string(&input)?)?;
        Ok(())
    }

    fn sample_feff_bin_data() -> FeffBinData {
        FeffBinData {
            version: "refeff-test".to_string(),
            pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
            ihole: 1,
            order: 2,
            initial_angular_momentum: 0,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potentials: vec![
                FeffBinPotential {
                    label: "Cu".to_string(),
                    atomic_number: 29,
                },
                FeffBinPotential {
                    label: "O".to_string(),
                    atomic_number: 8,
                },
            ],
            central_phase_shift: Array1::from_vec(vec![
                Complex64::new(0.1, -0.01),
                Complex64::new(0.2, -0.02),
                Complex64::new(0.3, -0.03),
            ]),
            complex_momentum: Array1::from_vec(vec![
                Complex64::new(1.0, 0.1),
                Complex64::new(1.1, 0.2),
                Complex64::new(1.2, 0.3),
            ]),
            real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
            paths: vec![FeffBinPath {
                index: 17,
                degeneracy: 4.0,
                effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
                criterion: 12.5,
                potential_indices: Array1::from_vec(vec![0, 1, 0]),
                positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                    (0, 0..=2) => 0.0,
                    (1, 0) => 1.0,
                    (1, 1) => 0.5,
                    (1, 2) => 0.0,
                    (2, 0) => -1.0,
                    (2, 1) => 0.25,
                    (2, 2) => 0.0,
                    _ => 0.0,
                }),
                beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
                eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
                leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
                amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
                phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
            }],
            raw_text: None,
        }
    }

    fn sample_list_dat() -> ListDatData {
        ListDatData {
            titles: vec!["PATH  Rmax= 6.000".to_string()],
            entries: vec![ListDatEntry {
                path_index: 17,
                sigma2: 0.0,
                amplitude_ratio: 12.5,
                degeneracy: 4.0,
                leg_count: 3,
                effective_half_path_length_angstrom: 2.5,
            }],
        }
    }

    fn reference_genfmt_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["genfmt.inp", "feff.bin", "list.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
