use super::{
    EELS_THETA0_RAD, eels_source_filename, has_cached_eels_output,
    has_supported_eels_source_handoff, run_in_dir,
};
use anyhow::{Context, Result};
use ndarray::{ArrayView1, ArrayView2, array};
use num_complex::Complex64;
use refeff_io::{
    EelsDatData, EelsGos1DatData, EelsGos2DatData, EelsMagicDatData, ModuleLogData, OpconsDatData,
    read_eels_dat, read_eels_gos1_dat, read_eels_gos2_dat, read_eels_magic_dat,
    read_module_log_dat, write_eels_dat, write_eels_gos1_dat, write_eels_gos2_dat,
    write_eels_magic_dat, write_module_log_dat, write_opcons_dat,
};
use std::path::{Path, PathBuf};

#[test]
fn eels_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_input(temp.path(), false)?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    assert!(!has_cached_eels_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!temp.path().join("eels.dat").exists());
    Ok(())
}

#[test]
fn eels_module_rejects_enabled_generation_without_source_spectra() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_input(temp.path(), true)?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("enabled EELS should require source spectra")?;

    let message = error.to_string();
    assert!(message.contains("failed to read"));
    assert!(message.contains("xmu.dat"));
    Ok(())
}

#[test]
fn eels_module_rejects_malformed_cached_output_without_source_spectra() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_input(temp.path(), true)?;
    std::fs::write(temp.path().join("eels.dat"), b"not an eels.dat cache\n")?;

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed EELS cache should require source spectra")?;

    let message = error.to_string();
    assert!(message.contains("failed to read"));
    assert!(message.contains("eels.dat"));
    Ok(())
}

#[test]
fn eels_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_eels_dat();
    std::fs::write(temp.path().join("eels.inp"), b"not an eels.inp handoff\n")?;
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    assert!(!has_cached_eels_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed EELS input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("eels.inp"), "{chain}");
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn eels_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_eels_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    assert!(!has_cached_eels_output(temp.path())?);
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn eels_module_does_not_claim_malformed_opconskk_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_opconskk_input(temp.path())?;
    std::fs::write(temp.path().join("opconsKK10.dat"), b"not an opcons table\n")?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    assert!(!has_cached_eels_output(temp.path())?);
    Ok(())
}

#[test]
fn eels_module_does_not_claim_cached_output_with_malformed_opconskk_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_opconskk_input(temp.path())?;
    let expected = sample_eels_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;
    std::fs::write(temp.path().join("opconsKK10.dat"), b"not an opcons table\n")?;

    assert!(!has_supported_eels_source_handoff(temp.path())?);
    assert!(!has_cached_eels_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed EELS opconsKK source should block cached EELS completion")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("opconsKK10.dat"), "{chain}");
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert!(!temp.path().join("logeels.dat").exists());
    Ok(())
}

#[test]
fn eels_module_roundtrips_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_input(temp.path(), true)?;
    let expected = sample_eels_dat();
    let expected_log = sample_module_log();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;
    write_module_log_dat(temp.path().join("logeels.dat"), &expected_log)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(read_eels_dat(temp.path().join("eels.dat"))?, expected);
    assert_eq!(
        read_module_log_dat(temp.path().join("logeels.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn eels_module_roundtrips_cached_magic_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_input_with_magic(temp.path(), true, 1, 40.0)?;
    let expected = sample_eels_dat();
    let expected_magic = sample_magic_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;
    write_eels_magic_dat(temp.path().join("magic.dat"), &expected_magic)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, expected.point_count());
    assert_eq!(
        read_eels_magic_dat(temp.path().join("magic.dat"))?,
        expected_magic
    );
    Ok(())
}

#[test]
fn eels_module_roundtrips_cached_gos_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_gos_input(temp.path())?;
    let expected = sample_eels_dat();
    let expected_gos1 = sample_gos1_dat();
    let expected_gos2 = sample_gos2_dat();
    write_eels_dat(temp.path().join("eels.dat"), &expected)?;
    write_eels_gos1_dat(temp.path().join("gos1.txt"), &expected_gos1)?;
    write_eels_gos2_dat(temp.path().join("gos2.txt"), &expected_gos2)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, expected.point_count());
    assert_eq!(
        read_eels_gos1_dat(temp.path().join("gos1.txt"))?,
        expected_gos1
    );
    assert_eq!(
        read_eels_gos2_dat(temp.path().join("gos2.txt"))?,
        expected_gos2
    );
    Ok(())
}

#[test]
fn eels_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_eels_dir()? else {
        eprintln!("skipping EELS reference test; generated ELNES/Cu reference not found");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(reference_dir.join("eels.inp"), temp.path().join("eels.inp"))?;
    std::fs::copy(reference_dir.join("eels.dat"), temp.path().join("eels.dat"))?;
    let expected = read_eels_dat(temp.path().join("eels.dat"))?;

    let count = run_in_dir(temp.path())?;

    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert_eq!(count, expected.point_count());
    assert!(actual.has_tensor());
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn eels_module_generates_reference_from_xmu_sources_when_cache_missing() -> Result<()> {
    let Some(reference_dir) = reference_eels_dir()? else {
        eprintln!("skipping EELS generation test; generated ELNES/Cu reference not found");
        return Ok(());
    };
    if !reference_has_xmu_sources(&reference_dir) {
        eprintln!("skipping EELS generation test; generated ELNES/Cu xmu sources not found");
        return Ok(());
    }

    let temp = tempfile::tempdir()?;
    std::fs::copy(reference_dir.join("eels.inp"), temp.path().join("eels.inp"))?;
    copy_reference_xmu_sources(&reference_dir, temp.path())?;
    let expected = read_eels_dat(reference_dir.join("eels.dat"))?;

    assert!(has_supported_eels_source_handoff(temp.path())?);
    assert!(has_cached_eels_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert_eq!(count, expected.point_count());
    assert!(actual.has_tensor());
    assert!(
        actual
            .header_lines
            .iter()
            .any(|line| line.contains("Orientation sensitive EELS"))
    );
    assert!(temp.path().join("logeels.dat").is_file());
    assert_eels_dat_close(&actual, &expected);
    Ok(())
}

#[test]
fn eels_module_generates_from_opconskk_sources_when_cache_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_opconskk_input(temp.path())?;
    write_opcons_dat(temp.path().join("opconsKK10.dat"), &sample_opcons_dat())?;

    assert!(has_supported_eels_source_handoff(temp.path())?);
    assert!(has_cached_eels_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert_eq!(count, sample_opcons_dat().point_count());
    assert_eq!(actual.point_count(), count);
    assert!(!actual.has_tensor());
    assert!(
        actual
            .header_lines
            .iter()
            .any(|line| line.contains("Orientation averaged EELS"))
    );
    assert!(temp.path().join("logeels.dat").is_file());
    Ok(())
}

#[test]
fn eels_module_recovers_malformed_cached_output_from_opconskk_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_opconskk_input(temp.path())?;
    write_opcons_dat(temp.path().join("opconsKK10.dat"), &sample_opcons_dat())?;
    std::fs::write(temp.path().join("eels.dat"), b"not an eels.dat cache\n")?;

    assert!(has_supported_eels_source_handoff(temp.path())?);
    assert!(has_cached_eels_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert_eq!(count, sample_opcons_dat().point_count());
    assert_eq!(actual.point_count(), count);
    assert!(!actual.has_tensor());
    assert!(temp.path().join("logeels.dat").is_file());
    Ok(())
}

#[test]
fn eels_module_regenerates_stale_cached_output_from_opconskk_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eels_opconskk_input(temp.path())?;
    let source = sample_opcons_dat();
    write_opcons_dat(temp.path().join("opconsKK10.dat"), &source)?;
    write_eels_dat(temp.path().join("eels.dat"), &sample_eels_dat())?;
    let stale = read_eels_dat(temp.path().join("eels.dat"))?;

    assert!(has_supported_eels_source_handoff(temp.path())?);
    assert!(has_cached_eels_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    let actual = read_eels_dat(temp.path().join("eels.dat"))?;
    assert_eq!(count, source.point_count());
    assert_eq!(actual.point_count(), count);
    assert_ne!(actual.energy_loss_ev, stale.energy_loss_ev);
    assert_array_close(
        "energy_loss_ev",
        actual.energy_loss_ev.view(),
        source.energy_ev.view(),
        1.0e-8,
        1.0e-8,
    );
    assert!(!actual.has_tensor());
    assert!(temp.path().join("logeels.dat").is_file());
    Ok(())
}

#[test]
fn eels_module_generates_magic_dat_from_xmu_sources_when_requested() -> Result<()> {
    let Some(reference_dir) = reference_eels_dir()? else {
        eprintln!("skipping EELS magic test; generated ELNES/Cu reference not found");
        return Ok(());
    };
    if !reference_has_xmu_sources(&reference_dir) {
        eprintln!("skipping EELS magic test; generated ELNES/Cu xmu sources not found");
        return Ok(());
    }

    let temp = tempfile::tempdir()?;
    write_eels_input_with_magic(temp.path(), true, 1, 40.0)?;
    copy_reference_xmu_sources(&reference_dir, temp.path())?;

    let count = run_in_dir(temp.path())?;

    let magic = read_eels_magic_dat(temp.path().join("magic.dat"))?;
    assert!(count > 0);
    assert_eq!(magic.point_count(), 5);
    assert_eq!(magic.point_counts.to_vec(), vec![3, 12, 27, 48, 75]);
    assert_close(
        "first magic collection angle",
        magic.rows[(0, 0)],
        EELS_THETA0_RAD,
        1.0e-8,
        1.0e-12,
    );
    assert_close(
        "last magic collection angle",
        magic.rows[(4, 0)],
        0.0024,
        1.0e-8,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn eels_module_generates_gos_outputs_from_xmu_sources_when_requested() -> Result<()> {
    let Some(reference_dir) = reference_eels_dir()? else {
        eprintln!("skipping EELS GOS test; generated ELNES/Cu reference not found");
        return Ok(());
    };
    if !reference_has_xmu_sources(&reference_dir) {
        eprintln!("skipping EELS GOS test; generated ELNES/Cu xmu sources not found");
        return Ok(());
    }

    let temp = tempfile::tempdir()?;
    write_eels_gos_input(temp.path())?;
    copy_reference_xmu_sources(&reference_dir, temp.path())?;

    let count = run_in_dir(temp.path())?;

    let gos1 = read_eels_gos1_dat(temp.path().join("gos1.txt"))?;
    let gos2 = read_eels_gos2_dat(temp.path().join("gos2.txt"))?;
    assert!(count > 0);
    assert_eq!(gos1.point_count(), 20);
    assert_eq!(gos2.q_count(), 20);
    assert_eq!(gos2.energy_count(), count);
    assert_eq!(gos2.element_label, "OXYG");
    assert_eq!(gos2.edge_label, "1S1/2");
    assert!(temp.path().join("eels.dat").is_file());
    assert!(temp.path().join("logeels.dat").is_file());
    Ok(())
}

fn write_eels_input(work_dir: &Path, enabled: bool) -> Result<()> {
    write_eels_input_with_magic(work_dir, enabled, 0, 0.0)
}

fn write_eels_input_with_magic(
    work_dir: &Path,
    enabled: bool,
    magic: i32,
    magic_energy: f64,
) -> Result<()> {
    let mut input = EelsInputFixture::xmu(enabled);
    input.magic = magic;
    input.magic_energy = magic_energy;
    write_eels_input_fixture(work_dir, input)
}

fn write_eels_gos_input(work_dir: &Path) -> Result<()> {
    let mut input = EelsInputFixture::xmu(true);
    input.calculation_mode = 9;
    input.average = 1;
    input.beam_direction = [0.0, 0.0, 1.0];
    write_eels_input_fixture(work_dir, input)
}

fn write_eels_opconskk_input(work_dir: &Path) -> Result<()> {
    write_eels_input_fixture(
        work_dir,
        EelsInputFixture {
            average: 1,
            input_source: 2,
            spectrum_column: 8,
            polarization: [10, 1, 10],
            beam_energy: 200_000.0,
            beam_direction: [0.0, 0.0, 1.0],
            collection_angle: 0.0015,
            convergence_angle: 0.0002,
            qmesh: [3, 2],
            ..EelsInputFixture::xmu(true)
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct EelsInputFixture {
    calculation_mode: i32,
    average: i32,
    relativistic: i32,
    cross_terms: i32,
    input_source: i32,
    spectrum_column: i32,
    polarization: [i32; 3],
    beam_energy: f64,
    beam_direction: [f64; 3],
    collection_angle: f64,
    convergence_angle: f64,
    qmesh: [i32; 2],
    detector: [f64; 2],
    magic: i32,
    magic_energy: f64,
}

impl EelsInputFixture {
    fn xmu(enabled: bool) -> Self {
        Self {
            calculation_mode: i32::from(enabled),
            average: 0,
            relativistic: 1,
            cross_terms: 1,
            input_source: 1,
            spectrum_column: 4,
            polarization: [1, 1, 9],
            beam_energy: 300_000.0,
            beam_direction: [0.0, 1.0, 0.0],
            collection_angle: 0.0024,
            convergence_angle: 0.0,
            qmesh: [5, 3],
            detector: [0.0, 0.0],
            magic: 0,
            magic_energy: 0.0,
        }
    }
}

fn write_eels_input_fixture(work_dir: &Path, input: EelsInputFixture) -> Result<()> {
    std::fs::write(
        work_dir.join("eels.inp"),
        format!(
            concat!(
                "calculate ELNES?\n",
                "{:4}\n",
                "average? relativistic? cross-terms? Which input?\n",
                "{:4}{:4}{:4}{:4}{:4}\n",
                "polarizations to be used ; min step max\n",
                "{:4}{:4}{:4}\n",
                "beam energy in eV\n",
                "{:13.5}\n",
                "beam direction in arbitrary units\n",
                "{:13.5}{:13.5}{:13.5}\n",
                "collection and convergence semiangle in rad\n",
                "{:13.5}{:13.5}\n",
                "qmesh - radial and angular grid size\n",
                "{:4}{:4}\n",
                "detector positions - two angles in rad\n",
                "{:13.5}{:13.5}\n",
                "calculate magic angle if magic=1\n",
                "{:4}\n",
                "energy for magic angle - eV above threshold\n",
                "{:13.5}\n"
            ),
            input.calculation_mode,
            input.average,
            input.relativistic,
            input.cross_terms,
            input.input_source,
            input.spectrum_column,
            input.polarization[0],
            input.polarization[1],
            input.polarization[2],
            input.beam_energy,
            input.beam_direction[0],
            input.beam_direction[1],
            input.beam_direction[2],
            input.collection_angle,
            input.convergence_angle,
            input.qmesh[0],
            input.qmesh[1],
            input.detector[0],
            input.detector[1],
            input.magic,
            input.magic_energy,
        ),
    )?;
    Ok(())
}

fn sample_eels_dat() -> EelsDatData {
    EelsDatData {
        header_lines: vec![
            "# Orientation averaged EELS calculation".to_string(),
            "#  Energy       total         atomic-bg     fine-struct".to_string(),
        ],
        energy_loss_ev: array![8979.41, 8980.98, 8982.40],
        total: array![0.123_014E-12, 0.146_285E-12, 0.176_683E-12],
        atomic_background: array![0.138_430E-12, 0.166_322E-12, 0.203_202E-12],
        fine_structure: array![-0.154_167E-13, -0.200_377E-13, -0.265_188E-13],
        tensor: None,
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EELS spectrum ...".to_string(),
            "Done with module: EELS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn sample_magic_dat() -> EelsMagicDatData {
    EelsMagicDatData {
        header_lines: vec![
            "#    beta        sp2        pi        sigmadip        total".to_string(),
        ],
        rows: array![
            [
                0.000_050_000,
                0.047,
                0.000_000_001,
                0.000_000_020,
                0.000_000_021
            ],
            [
                0.002_400_000,
                0.091,
                0.000_000_022,
                0.000_000_220,
                0.000_000_242
            ],
        ],
        point_counts: array![3_usize, 75],
    }
}

fn sample_gos1_dat() -> EelsGos1DatData {
    EelsGos1DatData {
        q_values: array![0.050_319_876_699, 0.106_573_941_39],
        strengths: array![27_431.800_619_716, 84_730.813_273_548],
    }
}

fn sample_gos2_dat() -> EelsGos2DatData {
    EelsGos2DatData {
        element_label: "OXYG".to_string(),
        edge_label: "1S1/2".to_string(),
        q_scale: 0.6859,
        q_log_step: 0.1294,
        edge_parameter: 100.0,
        energy_start_ev: 100.0,
        energy_step_ev: 10.0,
        strengths: array![
            [1_200_166.9, 260_695.67, 27_431.801],
            [3_841_354.5, 810_931.18, 84_730.813],
        ],
    }
}

fn sample_opcons_dat() -> OpconsDatData {
    OpconsDatData {
        header_lines: vec![
            "# opconsKK EELS source".to_string(),
            "# E eps1 eps2 n1 n2 mu R loss".to_string(),
        ],
        energy_ev: array![100.0, 120.0, 140.0],
        epsilon_minus_one: array![
            Complex64::new(0.2, 0.03),
            Complex64::new(0.25, 0.04),
            Complex64::new(0.28, 0.05),
        ],
        refractive_index_minus_one: array![
            Complex64::new(0.08, 0.010),
            Complex64::new(0.09, 0.012),
            Complex64::new(0.10, 0.014),
        ],
        absorption_coefficient: array![0.20, 0.24, 0.28],
        reflectivity: array![0.02, 0.03, 0.04],
        loss: array![0.0010, 0.0013, 0.0017],
    }
}

fn reference_eels_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/ELNES/Cu");
    let required = ["eels.inp", "eels.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_has_xmu_sources(path: &Path) -> bool {
    (1..=9).all(|index| path.join(eels_source_filename("xmu", index)).is_file())
}

fn copy_reference_xmu_sources(reference_dir: &Path, work_dir: &Path) -> Result<()> {
    for index in 1..=9 {
        let name = eels_source_filename("xmu", index);
        std::fs::copy(reference_dir.join(&name), work_dir.join(name))?;
    }
    Ok(())
}

fn assert_eels_dat_close(actual: &EelsDatData, expected: &EelsDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_eq!(actual.has_tensor(), expected.has_tensor());
    assert_array_close(
        "energy_loss_ev",
        actual.energy_loss_ev.view(),
        expected.energy_loss_ev.view(),
        1.0e-8,
        1.0e-8,
    );
    assert_array_close(
        "total",
        actual.total.view(),
        expected.total.view(),
        5.0e-5,
        1.0e-20,
    );
    assert_array_close(
        "atomic_background",
        actual.atomic_background.view(),
        expected.atomic_background.view(),
        5.0e-5,
        1.0e-20,
    );
    assert_array_close(
        "fine_structure",
        actual.fine_structure.view(),
        expected.fine_structure.view(),
        5.0e-5,
        1.0e-20,
    );
    if let (Some(actual), Some(expected)) = (&actual.tensor, &expected.tensor) {
        assert_matrix_close("tensor", actual.view(), expected.view(), 5.0e-5, 1.0e-20);
    }
}

fn assert_array_close(
    name: &str,
    actual: ArrayView1<'_, f64>,
    expected: ArrayView1<'_, f64>,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert_close(
            &format!("{name}[{index}]"),
            actual_value,
            expected_value,
            relative_tolerance,
            absolute_tolerance,
        );
    }
}

fn assert_matrix_close(
    name: &str,
    actual: ArrayView2<'_, f64>,
    expected: ArrayView2<'_, f64>,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) {
    assert_eq!(actual.shape(), expected.shape(), "{name} shape mismatch");
    for ((row, column), &actual_value) in actual.indexed_iter() {
        assert_close(
            &format!("{name}[{row},{column}]"),
            actual_value,
            expected[(row, column)],
            relative_tolerance,
            absolute_tolerance,
        );
    }
}

fn assert_close(
    name: &str,
    actual: f64,
    expected: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) {
    let tolerance = absolute_tolerance.max(relative_tolerance * actual.abs().max(expected.abs()));
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{name}: actual {actual:e} expected {expected:e} diff {difference:e} tolerance {tolerance:e}"
    );
}
