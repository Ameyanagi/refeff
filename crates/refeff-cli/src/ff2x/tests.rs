use super::{has_cached_ff2x_output, run_in_dir};
use anyhow::{Context, Result};
use ndarray::Array1;
use num_complex::Complex64;
use refeff_io::{
    ChiDatData, DanesDatData, Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput, IoError,
    ModuleLogData, XmuDatData, XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData,
    ff2x_input_string, read_chi_dat, read_contour_dat, read_curve_dat, read_danes_dat,
    read_module_log_dat, read_prexmu_dat, read_residue_dat, read_xmu_dat, read_xscorr_raw_dat,
    write_chi_dat, write_contour_dat, write_curve_dat, write_danes_dat, write_module_log_dat,
    write_prexmu_dat, write_residue_dat, write_xmu_dat, write_xscorr_raw_dat,
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
    let xscorr = sample_xscorr_complex_table();
    let curve = sample_xscorr_curve_dat();
    let raw = sample_xscorr_raw_dat();
    let log = sample_module_log();
    write_xmu_dat(temp.path().join("xmu.dat"), &xmu)?;
    write_chi_dat(temp.path().join("chi.dat"), &chi)?;
    write_danes_dat(temp.path().join("danes.dat"), &danes)?;
    write_prexmu_dat(temp.path().join("prexmu.dat"), &xscorr)?;
    write_residue_dat(temp.path().join("residue.dat"), &xscorr)?;
    write_contour_dat(temp.path().join("contour.dat"), &xscorr)?;
    write_curve_dat(temp.path().join("curve.dat"), &curve)?;
    write_xscorr_raw_dat(temp.path().join("raw.dat"), &raw)?;
    write_module_log_dat(temp.path().join("log6.dat"), &log)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 9);
    assert!(has_cached_ff2x_output(temp.path())?);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, chi);
    assert_eq!(read_danes_dat(temp.path().join("danes.dat"))?, danes);
    assert_eq!(read_prexmu_dat(temp.path().join("prexmu.dat"))?, xscorr);
    assert_eq!(read_residue_dat(temp.path().join("residue.dat"))?, xscorr);
    assert_eq!(read_contour_dat(temp.path().join("contour.dat"))?, xscorr);
    assert_eq!(read_curve_dat(temp.path().join("curve.dat"))?, curve);
    assert_eq!(read_xscorr_raw_dat(temp.path().join("raw.dat"))?, raw);
    assert_eq!(read_module_log_dat(temp.path().join("log6.dat"))?, log);
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
    let mut sidecar_count = 0_usize;
    for name in [
        "prexmu.dat",
        "residue.dat",
        "contour.dat",
        "curve.dat",
        "raw.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
            sidecar_count += 1;
        }
    }
    let has_log = reference_dir.join("log6.dat").is_file();
    if has_log {
        std::fs::copy(reference_dir.join("log6.dat"), temp.path().join("log6.dat"))?;
    }
    let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let expected_chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let expected_prexmu =
        optional_read(temp.path().join("prexmu.dat"), |path| read_prexmu_dat(path))?;
    let expected_residue = optional_read(temp.path().join("residue.dat"), |path| {
        read_residue_dat(path)
    })?;
    let expected_contour = optional_read(temp.path().join("contour.dat"), |path| {
        read_contour_dat(path)
    })?;
    let expected_curve = optional_read(temp.path().join("curve.dat"), |path| read_curve_dat(path))?;
    let expected_raw = optional_read(temp.path().join("raw.dat"), |path| {
        read_xscorr_raw_dat(path)
    })?;
    let expected_log = optional_module_log(temp.path().join("log6.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2 + sidecar_count + usize::from(has_log));
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, expected_chi);
    if let Some(expected) = expected_prexmu {
        assert_eq!(read_prexmu_dat(temp.path().join("prexmu.dat"))?, expected);
    }
    if let Some(expected) = expected_residue {
        assert_eq!(read_residue_dat(temp.path().join("residue.dat"))?, expected);
    }
    if let Some(expected) = expected_contour {
        assert_eq!(read_contour_dat(temp.path().join("contour.dat"))?, expected);
    }
    if let Some(expected) = expected_curve {
        assert_eq!(read_curve_dat(temp.path().join("curve.dat"))?, expected);
    }
    if let Some(expected) = expected_raw {
        assert_eq!(read_xscorr_raw_dat(temp.path().join("raw.dat"))?, expected);
    }
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log6.dat"))?, expected);
    }
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

fn sample_xscorr_complex_table() -> XscorrComplexTable {
    XscorrComplexTable {
        energy_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_020_637_731_56, 0.000_120_322_770_8),
            Complex64::new(-0.000_021_177_763_91, 0.000_123_685_052_9),
        ]),
    }
}

fn sample_xscorr_curve_dat() -> XscorrCurveDatData {
    XscorrCurveDatData {
        energy: Array1::from_vec(vec![
            Complex64::new(-0.138_801_301_5, 0.000_183_746_545),
            Complex64::new(-0.138_801_301_5, 0.000_367_493_09),
        ]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_028_662, 0.000_237_48),
            Complex64::new(-0.000_028_683, 0.000_237_44),
        ]),
    }
}

fn sample_xscorr_raw_dat() -> XscorrRawDatData {
    XscorrRawDatData {
        temperature_hartree: 0.0,
        electronic_temperature_ev: 0.0,
        loss_ev: 0.864_59,
        fermi_energy_ev: -3.776_977_18,
        pole_count: 0,
        omega_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        cchi: Array1::from_vec(vec![
            Complex64::new(-0.000_016_299_5, 0.000_115_24),
            Complex64::new(-0.000_016_898_337_65, 0.000_118_558_222_9),
        ]),
        one_minus_fermi: Array1::from_vec(vec![0.5, 0.514_017_875_2]),
        xmu0: Array1::from_vec(vec![
            Complex64::new(-0.000_032_599, 0.000_230_48),
            Complex64::new(-0.000_032_875, 0.000_230_65),
        ]),
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS ...".to_string(),
            "Done with module: EXAFS spectra.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
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

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_module_log_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_read<T>(
    path: impl AsRef<Path>,
    read: impl FnOnce(&Path) -> std::result::Result<T, IoError>,
) -> Result<Option<T>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read(path)?))
    } else {
        Ok(None)
    }
}
