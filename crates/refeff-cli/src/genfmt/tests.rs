use super::{has_cached_genfmt_output, run_in_dir};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
use refeff_io::{
    FeffBinData, FeffBinPath, FeffBinPotential, GenfmtControl, GenfmtInput, ListDatData,
    ListDatEntry, ModuleLogData, genfmt_input_string, read_feff_bin, read_list_dat,
    read_module_log_dat, write_feff_bin, write_list_dat, write_module_log_dat,
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

    assert!(
        error.to_string().contains(
            "GENFMT path-format generation requires the unported GENFMT numerical solver"
        )
    );
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
    write_module_log_dat(temp.path().join("log5.dat"), &sample_module_log())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_log = read_module_log_dat(temp.path().join("log5.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(has_cached_genfmt_output(temp.path())?);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_module_log_dat(temp.path().join("log5.dat"))?,
        expected_log
    );
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
    let log_source = reference_dir.join("log5.dat");
    if log_source.is_file() {
        std::fs::copy(log_source, temp.path().join("log5.dat"))?;
    }
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_log = optional_module_log(temp.path().join("log5.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2 + usize::from(expected_log.is_some()));
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log5.dat"))?, expected);
    }
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

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS parameters ...".to_string(),
            "Done with module: EXAFS parameters (GENFMT).".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_module_log_dat(path)?))
    } else {
        Ok(None)
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
