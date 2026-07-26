use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, arr1, arr2};
use num_complex::Complex64;
use refeff_core::{
    EelsReadSpectrumInput, FEFF_MDFF_AUTOMATIC_THETA_X, FEFF_MDFF_AUTOMATIC_THETA_Y,
    MdffAutomaticQGridInput, MdffManualQGridInput, MdffSpectrumInput, eels_read_spectrum,
    mdff_automatic_q_grid, mdff_manual_q_grid, mdff_spectrum,
};
use refeff_io::{
    EelsInput, GlobalInput, GlobalQVector, MdffDatData, MdffInput, ModuleLogData, read_mdff_dat,
    read_module_log_dat, write_mdff_dat, write_module_log_dat,
};

use crate::eels;
use crate::work_dir_for_input;

const MANUAL_MDFF_Q_VECTORS: [[f64; 2]; 3] = [[0.0, 0.0], [0.0, -0.23240], [-0.03755, -0.03755]];
const MANUAL_MDFF_Q_PRIME_AMPLITUDE: Complex64 = Complex64::new(0.8, -0.2);

/// Run the supported FEFF EELS-MDFF path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF EELS-MDFF run can be satisfied from cache or supported source
/// handoffs.
pub(crate) fn has_cached_mdff_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("mdff.inp").is_file() {
        return Ok(false);
    }
    if !global_requests_mdff_for_discovery(work_dir) {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    let output_path = work_dir.join("mdff.dat");
    if output_path.is_file() && read_mdff_dat(&output_path).is_ok() {
        if validate_declared_mdff_source_handoffs(work_dir, &input).is_err() {
            return Ok(false);
        }
        return Ok(true);
    }

    has_supported_mdff_source_handoff_for_input(work_dir, &input)
}

/// Whether FEFF EELS-MDFF can generate `mdff.dat` from source-backed EELS
/// spectra and supported q/q-prime controls.
pub(crate) fn has_supported_mdff_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("mdff.inp").is_file() {
        return Ok(false);
    }
    if !global_requests_mdff_for_discovery(work_dir) {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    has_supported_mdff_source_handoff_for_input(work_dir, &input)
}

fn has_mdff_source_spectra(work_dir: &Path, input: &MdffInput) -> Result<bool> {
    has_supported_mdff_source_handoff_for_input(work_dir, input)
}

fn has_supported_mdff_source_handoff_for_input(work_dir: &Path, input: &MdffInput) -> Result<bool> {
    if !work_dir.join("eels.inp").is_file() {
        return Ok(false);
    }
    if !matches!(input.q_input, 1 | 2) {
        return Ok(false);
    }
    let eels_input = read_eels_input(work_dir)?;
    Ok(eels_input.calculate_elnes && eels::has_eels_source_spectra(work_dir, &eels_input))
}

/// Run the supported FEFF EELS-MDFF path.
///
/// Existing `mdff.dat` files are validated and re-rendered. When source spectra
/// are available, stale readable caches are regenerated from those spectra. The
/// FEFF manual q/q-prime branch (`q_input=1`) and the hardcoded two-position
/// automatic branch (`q_input=2`) can also be generated from the same source
/// spectra consumed by EELS. FEFF accepts tasks 1/2/3 on the same output path
/// and only changes the module log task marker, so the Rust source generator
/// mirrors that behavior.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    if !global_requests_mdff(work_dir)? {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    let output_path = work_dir.join("mdff.dat");
    let data = if output_path.is_file() {
        match read_mdff_dat(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))
        {
            Ok(data) => {
                generate_mdff_if_stale_against_source(work_dir, &input, &data)?.unwrap_or(data)
            }
            Err(error) => {
                if !has_mdff_source_spectra(work_dir, &input)? {
                    return Err(error);
                }
                generate_mdff_output(work_dir, &input)?
            }
        }
    } else {
        generate_mdff_output(work_dir, &input)?
    };
    let point_count = data.point_count();
    write_cached_output(&output_path, &data)?;
    write_or_generate_module_log(&work_dir.join("logmdff.dat"), &input)?;
    Ok(point_count)
}

fn generate_mdff_if_stale_against_source(
    work_dir: &Path,
    input: &MdffInput,
    cached: &MdffDatData,
) -> Result<Option<MdffDatData>> {
    validate_declared_mdff_source_handoffs(work_dir, input)?;
    if !matches!(input.q_input, 1 | 2) || !work_dir.join("eels.inp").is_file() {
        return Ok(None);
    }
    let generated = match generate_mdff_output(work_dir, input) {
        Ok(generated) => generated,
        Err(_) => return Ok(None),
    };
    if mdff_dat_matches_source(cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn mdff_dat_matches_source(cached: &MdffDatData, source: &MdffDatData) -> bool {
    const ENERGY_ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
    const ENERGY_RELATIVE_TOLERANCE: f64 = 1.0e-8;
    const SPECTRUM_ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
    const SPECTRUM_RELATIVE_TOLERANCE: f64 = 5.0e-5;

    cached.point_count() == source.point_count()
        && cached.channel_count() == source.channel_count()
        && real_slices_match(
            &cached.energy_loss_ev,
            &source.energy_loss_ev,
            ENERGY_ABSOLUTE_TOLERANCE,
            ENERGY_RELATIVE_TOLERANCE,
        )
        && complex_matrix_match(
            &cached.spectrum,
            &source.spectrum,
            SPECTRUM_ABSOLUTE_TOLERANCE,
            SPECTRUM_RELATIVE_TOLERANCE,
        )
}

fn validate_declared_mdff_source_handoffs(work_dir: &Path, input: &MdffInput) -> Result<()> {
    if !matches!(input.q_input, 1 | 2) || !work_dir.join("eels.inp").is_file() {
        return Ok(());
    }
    let eels_input = match read_eels_input(work_dir) {
        Ok(input) => input,
        Err(error) => {
            if any_eels_source_handoff_file(work_dir) {
                return Err(error);
            }
            return Ok(());
        }
    };
    if !eels_input.calculate_elnes {
        return Ok(());
    }
    eels::validate_declared_eels_source_handoffs(work_dir, &eels_input)
        .context("failed to validate EELS-MDFF source spectra")
}

fn any_eels_source_handoff_file(work_dir: &Path) -> bool {
    ["xmu", "opconsKK"].iter().any(|prefix| {
        (1..=10).any(|index| work_dir.join(eels_source_filename(prefix, index)).is_file())
    })
}

fn eels_source_filename(prefix: &str, index: usize) -> String {
    match index {
        1 => format!("{prefix}.dat"),
        2..=9 => format!("{prefix}0{index}.dat"),
        10 => format!("{prefix}10.dat"),
        _ => format!("{prefix}{index}.dat"),
    }
}

fn global_requests_mdff(work_dir: &Path) -> Result<bool> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(true);
    }
    Ok(read_global_input(&path)?.q_control.imdff == 3)
}

fn global_requests_mdff_for_discovery(work_dir: &Path) -> bool {
    matches!(global_requests_mdff(work_dir), Ok(true))
}

fn read_global_input(path: &Path) -> Result<GlobalInput> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    GlobalInput::parse_str(path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn read_input(work_dir: &Path) -> Result<MdffInput> {
    let input_path = work_dir.join("mdff.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    MdffInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn read_eels_input(work_dir: &Path) -> Result<EelsInput> {
    let input_path = work_dir.join("eels.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    EelsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn generate_mdff_output(work_dir: &Path, input: &MdffInput) -> Result<MdffDatData> {
    let eels_input = read_eels_input(work_dir)?;
    if !eels_input.calculate_elnes {
        bail!("EELS-MDFF generation requires enabled EELS/ELNES controls");
    }
    let sources = eels::read_eels_sources(work_dir, &eels_input)?;
    let source_views = eels::eels_source_views(&sources);
    let readsp = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &source_views,
        orientation_averaged: eels_input.control.average != 0,
        cross_terms: eels_input.control.cross_terms != 0,
        polarization_min: eels::positive_usize(
            "polarization minimum",
            eels_input.polarization.min,
        )?,
        polarization_step: eels::positive_usize("polarization step", eels_input.polarization.step)?,
        polarization_max: eels::positive_usize(
            "polarization maximum",
            eels_input.polarization.max,
        )?,
    })
    .context("failed to assemble EELS-MDFF source spectra")?;

    let (q_vectors, classical_q_lengths, amplitudes) = match input.q_input {
        1 => {
            let beam = manual_mdff_beam(work_dir)?;
            let q_grid = mdff_manual_q_grid(MdffManualQGridInput {
                incident_energy_ev: eels_input.beam_energy,
                q_vectors: beam.q_vectors.view(),
                energy_count: readsp.energy_loss_ev.len(),
                relativistic: eels_input.control.relativistic != 0,
            })
            .context("failed to build FEFF manual-q MDFF grid")?;
            (
                q_grid.q_vectors,
                q_grid.classical_q_lengths,
                beam.amplitudes,
            )
        }
        2 => {
            let theta_x = arr1(&FEFF_MDFF_AUTOMATIC_THETA_X);
            let theta_y = arr1(&FEFF_MDFF_AUTOMATIC_THETA_Y);
            let q_grid = mdff_automatic_q_grid(MdffAutomaticQGridInput {
                incident_energy_ev: eels_input.beam_energy,
                energy_loss_ev: readsp.energy_loss_ev.view(),
                beam_direction: eels_input.beam_direction,
                theta_x: theta_x.view(),
                theta_y: theta_y.view(),
                relativistic: eels_input.control.relativistic != 0,
            })
            .context("failed to build FEFF automatic-q MDFF grid")?;
            (
                q_grid.q_vectors,
                q_grid.classical_q_lengths,
                automatic_mdff_amplitudes(theta_x.len()),
            )
        }
        value => bail!("unsupported EELS-MDFF q_input selector {value}; expected 1 or 2"),
    };
    let spectrum = mdff_spectrum(MdffSpectrumInput {
        incident_energy_ev: eels_input.beam_energy,
        energy_loss_ev: readsp.energy_loss_ev.view(),
        transition_tensor: readsp.transition_tensor.view(),
        q_vectors: q_vectors.view(),
        classical_q_lengths: classical_q_lengths.view(),
        amplitudes: amplitudes.view(),
        relativistic: eels_input.control.relativistic != 0,
    })
    .context("failed to compute EELS-MDFF spectrum")?;

    Ok(MdffDatData {
        header_lines: mdff_header_lines(&eels_input),
        energy_loss_ev: spectrum.energy_loss_ev,
        spectrum: spectrum.spectrum,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct ManualMdffBeam {
    q_vectors: Array2<f64>,
    amplitudes: Array1<Complex64>,
}

fn manual_mdff_beam(work_dir: &Path) -> Result<ManualMdffBeam> {
    let global_path = work_dir.join("global.inp");
    if global_path.is_file() {
        let global = read_global_input(&global_path)?;
        if let Some(beam) = manual_mdff_beam_from_global(&global)? {
            return Ok(beam);
        }
    }
    Ok(ManualMdffBeam {
        q_vectors: manual_mdff_q_vectors(),
        amplitudes: manual_mdff_amplitudes(),
    })
}

fn manual_mdff_beam_from_global(global: &GlobalInput) -> Result<Option<ManualMdffBeam>> {
    if global.q_vectors.is_empty() {
        return Ok(None);
    }

    let q_count = global.q_vectors.len();
    let mut q_vectors = Array2::zeros((3, q_count));
    let mut amplitudes = Vec::with_capacity(q_count);
    for (index, vector) in global.q_vectors.iter().enumerate() {
        q_vectors[(0, index)] = vector.q[0];
        q_vectors[(1, index)] = vector.q[1];
        q_vectors[(2, index)] = vector.q[2];
        amplitudes.push(normalized_mdff_amplitude(index, vector)?);
    }

    Ok(Some(ManualMdffBeam {
        q_vectors,
        amplitudes: Array1::from_vec(amplitudes),
    }))
}

fn normalized_mdff_amplitude(index: usize, vector: &GlobalQVector) -> Result<Complex64> {
    let amplitude = Complex64::new(vector.weight[0], vector.weight[1]);
    let norm = amplitude.norm();
    if !norm.is_finite() || norm == 0.0 {
        bail!(
            "EELS-MDFF global q-vector {} has invalid zero beam amplitude ({}, {})",
            index + 1,
            vector.weight[0],
            vector.weight[1]
        );
    }
    Ok(amplitude / norm)
}

fn manual_mdff_q_vectors() -> Array2<f64> {
    arr2(&MANUAL_MDFF_Q_VECTORS)
}

fn manual_mdff_amplitudes() -> Array1<Complex64> {
    let normalized_q_prime = MANUAL_MDFF_Q_PRIME_AMPLITUDE / MANUAL_MDFF_Q_PRIME_AMPLITUDE.norm();
    arr1(&[Complex64::new(1.0, 0.0), normalized_q_prime])
}

fn automatic_mdff_amplitudes(q_count: usize) -> Array1<Complex64> {
    Array1::from_elem(q_count, Complex64::new(1.0, 0.0))
}

fn real_slices_match(
    cached: &Array1<f64>,
    source: &Array1<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.len() == source.len()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_matrix_match(
    cached: &Array2<Complex64>,
    source: &Array2<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.shape() == source.shape()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            scalar_matches(cached.re, source.re, absolute_tolerance, relative_tolerance)
                && scalar_matches(cached.im, source.im, absolute_tolerance, relative_tolerance)
        })
}

fn scalar_matches(
    cached: f64,
    source: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    (cached - source).abs() <= absolute_tolerance + relative_tolerance * source.abs()
}

fn mdff_header_lines(input: &EelsInput) -> Vec<String> {
    let mut lines = Vec::new();
    let beam_energy_kev = input.beam_energy / 1000.0;
    if input.control.average != 0 {
        lines.push(format!(
            "# Orientation averaged EELS calculation - beam energy = {beam_energy_kev:6.0}keV"
        ));
    } else {
        lines.push(format!(
            "# Orientation sensitive EELS calculation - beam energy = {beam_energy_kev:6.0}keV"
        ));
        lines.push(format!(
            "# Sample to beam orientation : {:8.3} {:8.3} {:8.3} ",
            input.beam_direction[0], input.beam_direction[1], input.beam_direction[2]
        ));
    }
    lines.push(format!(
        "# Collection and convergence semiangle: {:10.3} {:10.3}   ; # points: {:5} x{:2}",
        input.angles.collection * 1000.0,
        input.angles.convergence * 1000.0,
        input.qmesh.radial,
        input.qmesh.angular
    ));
    lines.push(format!(
        "# Detector position: {:10.4} {:10.4} ",
        input.detector[0], input.detector[1]
    ));
    lines.push(mdff_relativity_header(input).to_string());
    lines.push("#  Energy       total".to_string());
    lines
}

fn mdff_relativity_header(input: &EelsInput) -> &'static str {
    match (
        input.control.relativistic != 0,
        input.control.cross_terms != 0,
    ) {
        (true, true) => "# Relativistic and cross-terms.",
        (true, false) => "# Relativistic, no cross-terms.",
        (false, true) => "# Nonrelativistic and cross-terms.",
        (false, false) => "# Nonrelativistic, no cross-terms.",
    }
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

fn write_or_generate_module_log(path: &Path, input: &MdffInput) -> Result<()> {
    if path.is_file() {
        write_optional_module_log(path)?;
        return Ok(());
    }
    write_module_log_dat(path, &generated_mdff_module_log(input)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn generated_mdff_module_log(input: &MdffInput) -> Result<ModuleLogData> {
    let mut lines = Vec::new();
    lines.push(match input.q_input {
        1 => "Calculating MDFF for user-specified q,q' - e.g. for plotting".to_string(),
        2 => {
            "Calculating MDFF for given experimental parameters - e.g. for simulating an EELS experiment"
                .to_string()
        }
        value => bail!("unsupported EELS-MDFF q_input selector {value}; expected 1 or 2"),
    });
    lines.push("Starting MDFF module.".to_string());
    lines.push("Reading Sigma tensor from file.".to_string());
    lines.push(match input.task {
        1 => "Calculating EELS cross-section.".to_string(),
        2 => "Calculating MDFF.".to_string(),
        3 => "Calculating CAMDFF.".to_string(),
        value => bail!("unsupported EELS-MDFF task selector {value}; expected 1, 2, or 3"),
    });
    lines.push("Converting XAS to EELS.".to_string());
    lines.push("Creating headers.".to_string());
    lines.push("Entering big loop over energy.".to_string());
    lines.push("Module mdff is finished.  Exiting.".to_string());

    Ok(ModuleLogData {
        line_terminators: vec!["\n".to_string(); lines.len()],
        lines,
    })
}

#[cfg(test)]
mod tests {
    use super::{has_cached_mdff_output, has_supported_mdff_source_handoff, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, ArrayView1};
    use num_complex::Complex64;
    use refeff_io::{
        CfAverage, EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh, GlobalControl,
        GlobalInput, GlobalNorms, GlobalQControl, GlobalQVector, MdffDatData, MdffInput,
        ModuleLogData, XmuDatData, eels_input_string, global_input_string, mdff_input_string,
        read_mdff_dat, read_module_log_dat, write_mdff_dat, write_module_log_dat, write_xmu_dat,
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
        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn eelsmdff_module_ignores_malformed_global_without_mdff_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;
        write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn eelsmdff_module_does_not_claim_malformed_global_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_mdff_input(temp.path())?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;
        let expected = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed global input should fail through explicit EELS-MDFF run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("global.inp"), "{chain}");
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert!(!temp.path().join("logmdff.dat").exists());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_requires_source_spectra_for_generation() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("enabled EELS-MDFF should require source spectra")?;

        assert!(error.to_string().contains("failed to read"));
        Ok(())
    }

    #[test]
    fn eelsmdff_module_rejects_malformed_cached_output_without_source_spectra() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        std::fs::write(temp.path().join("mdff.dat"), b"not an mdff.dat cache\n")?;

        assert!(!has_cached_mdff_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed EELS-MDFF cache should require source spectra")?;

        let message = error.to_string();
        assert!(message.contains("failed to read"));
        assert!(message.contains("mdff.dat"));
        Ok(())
    }

    #[test]
    fn eelsmdff_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        let expected = sample_mdff_dat()?;
        std::fs::write(temp.path().join("mdff.inp"), b"not an mdff.inp handoff\n")?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed EELS-MDFF input should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("mdff.inp"), "{chain}");
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert!(!temp.path().join("logmdff.dat").exists());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        let expected = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert!(!temp.path().join("logmdff.dat").exists());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_generates_manual_q_output_from_xmu_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_manual_q_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;

        assert!(has_supported_mdff_source_handoff(temp.path())?);
        assert!(has_cached_mdff_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 5);
        assert_eq!(data.energy_loss_ev.as_slice(), Some(&[12.5, 45.0][..]));
        assert_complex_row_close(
            data.spectrum.row(0),
            &[
                Complex64::new(4.724_605_070_321_539e2, 1.429_465_454_164_265e1),
                Complex64::new(2.615_870_550_430_711e2, 0.0),
                Complex64::new(1.180_953_902_256_069_6e2, 2.952_384_755_640_174e1),
                Complex64::new(6.091_677_205_903_636e1, -1.522_919_301_475_909e1),
                Complex64::new(3.186_128_970_443_955_3e1, 0.0),
            ],
        );
        assert_complex_row_close(
            data.spectrum.row(1),
            &[
                Complex64::new(1.833_341_576_389_011e2, 4.403_187_327_576_675),
                Complex64::new(1.065_745_816_311_581_4e2, 0.0),
                Complex64::new(4.181_761_738_897_701e1, 1.045_440_434_724_425_3e1),
                Complex64::new(2.420_486_807_867_031_2e1, -6.051_217_019_667_578),
                Complex64::new(1.073_709_054_009_563_2e1, 0.0),
            ],
        );
        let log = read_module_log_dat(temp.path().join("logmdff.dat"))?;
        assert!(
            log.lines
                .iter()
                .any(|line| line == "Calculating MDFF for user-specified q,q' - e.g. for plotting")
        );
        Ok(())
    }

    #[test]
    fn eelsmdff_module_generates_automatic_q_output_from_xmu_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;

        assert!(has_supported_mdff_source_handoff(temp.path())?);
        assert!(has_cached_mdff_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 5);
        assert_eq!(data.energy_loss_ev.as_slice(), Some(&[12.5, 45.0][..]));
        for row in data.spectrum.rows() {
            let channel_sum = row.iter().skip(1).copied().sum::<Complex64>();
            assert_close(row[0].re, channel_sum.re);
            assert_close(row[0].im, channel_sum.im);
            assert!(row[0].norm() > 0.0);
        }
        let log = read_module_log_dat(temp.path().join("logmdff.dat"))?;
        assert!(log.lines.iter().any(|line| {
            line == "Calculating MDFF for given experimental parameters - e.g. for simulating an EELS experiment"
        }));
        Ok(())
    }

    #[test]
    fn eelsmdff_module_recovers_malformed_cached_output_from_xmu_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;
        std::fs::write(temp.path().join("mdff.dat"), b"not an mdff.dat cache\n")?;

        assert!(has_supported_mdff_source_handoff(temp.path())?);
        assert!(has_cached_mdff_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 5);
        assert!(temp.path().join("logmdff.dat").is_file());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_regenerates_stale_cached_output_from_xmu_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;
        let stale = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &stale)?;

        assert!(has_supported_mdff_source_handoff(temp.path())?);
        assert!(has_cached_mdff_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 5);
        assert_ne!(data.channel_count(), stale.channel_count());
        assert_eq!(data.energy_loss_ev.as_slice(), Some(&[12.5, 45.0][..]));
        assert!(temp.path().join("logmdff.dat").is_file());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_does_not_claim_cached_output_with_malformed_xmu_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;
        let expected = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;
        std::fs::write(temp.path().join("xmu09.dat"), b"not an xmu source\n")?;

        assert!(!has_supported_mdff_source_handoff(temp.path())?);
        assert!(!has_cached_mdff_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed EELS-MDFF xmu source should block cached MDFF completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("xmu09.dat"), "{chain}");
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert!(!temp.path().join("logmdff.dat").exists());
        Ok(())
    }

    #[test]
    fn eelsmdff_module_generates_manual_q_mdff_and_camdff_tasks() -> Result<()> {
        for (task, expected_log_line) in [(2, "Calculating MDFF."), (3, "Calculating CAMDFF.")] {
            let temp = tempfile::tempdir()?;
            write_global_input(temp.path(), 3)?;
            write_manual_q_mdff_input_with_task(temp.path(), task)?;
            write_manual_q_eels_input(temp.path())?;
            write_manual_q_xmu_sources(temp.path())?;

            assert!(has_cached_mdff_output(temp.path())?);
            let count = run_in_dir(temp.path())?;

            assert_eq!(count, 2);
            let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
            assert_eq!(data.point_count(), 2);
            assert_eq!(data.channel_count(), 5);
            let log = read_module_log_dat(temp.path().join("logmdff.dat"))?;
            assert!(
                log.lines.iter().any(|line| line == expected_log_line),
                "task={task} log lines: {:?}",
                log.lines
            );
        }
        Ok(())
    }

    #[test]
    fn eelsmdff_module_uses_global_q_vectors_for_manual_q_generation() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input_with_q_vectors(temp.path(), 3, sample_global_q_vectors())?;
        write_manual_q_mdff_input(temp.path())?;
        write_manual_q_eels_input(temp.path())?;
        write_manual_q_xmu_sources(temp.path())?;

        assert!(has_cached_mdff_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_mdff_dat(temp.path().join("mdff.dat"))?;
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.channel_count(), 10);
        assert!(data.spectrum[(0, 9)].norm() > 0.0);
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
    fn eelsmdff_module_generates_missing_module_log_from_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        write_mdff_input(temp.path())?;
        let expected = sample_mdff_dat()?;
        write_mdff_dat(temp.path().join("mdff.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_mdff_dat(temp.path().join("mdff.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("logmdff.dat"))?,
            generated_sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn eelsmdff_module_generates_manual_q_and_mdff_task_log_lines() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_global_input(temp.path(), 3)?;
        std::fs::write(
            temp.path().join("mdff.inp"),
            mdff_input_string(&MdffInput {
                task: 2,
                q_input: 1,
            })?,
        )?;
        write_mdff_dat(temp.path().join("mdff.dat"), &sample_mdff_dat()?)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let log = read_module_log_dat(temp.path().join("logmdff.dat"))?;
        assert!(
            log.lines
                .iter()
                .any(|line| line == "Calculating MDFF for user-specified q,q' - e.g. for plotting")
        );
        assert!(log.lines.iter().any(|line| line == "Calculating MDFF."));
        Ok(())
    }

    #[test]
    fn eelsmdff_module_checks_generated_reference_when_present() -> Result<()> {
        let Some(rdinp) = reference_rdinp()? else {
            crate::require_fixture!("EELS-MDFF reference test; FEFF10 rdinp not found");
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

    fn write_manual_q_mdff_input(work_dir: &Path) -> Result<()> {
        write_manual_q_mdff_input_with_task(work_dir, 1)
    }

    fn write_manual_q_mdff_input_with_task(work_dir: &Path, task: i32) -> Result<()> {
        std::fs::write(
            work_dir.join("mdff.inp"),
            mdff_input_string(&MdffInput { task, q_input: 1 })?,
        )?;
        Ok(())
    }

    fn write_manual_q_eels_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("eels.inp"),
            eels_input_string(&manual_q_eels_input())?,
        )?;
        Ok(())
    }

    fn manual_q_eels_input() -> EelsInput {
        EelsInput {
            calculate_elnes: true,
            calculation_mode: 1,
            control: EelsControl {
                average: 0,
                relativistic: 1,
                cross_terms: 1,
                input: 1,
                spectrum_column: 4,
            },
            polarization: EelsPolarization {
                min: 1,
                step: 1,
                max: 9,
            },
            beam_energy: 300_000.0,
            beam_direction: [0.0, 1.0, 0.0],
            angles: EelsAngles {
                collection: 0.0024,
                convergence: 0.0,
            },
            qmesh: EelsQMesh {
                radial: 5,
                angular: 3,
            },
            detector: [0.0, 0.0],
            magic: 0,
            magic_energy: 0.0,
        }
    }

    fn write_manual_q_xmu_sources(work_dir: &Path) -> Result<()> {
        let tensor = manual_q_transition_tensor();
        for polarization in 1..=9 {
            let component = polarization - 1;
            let row = component / 3;
            let column = component % 3;
            let mu = Array1::from_iter((0..2).map(|energy| tensor[(energy, row, column)]));
            write_xmu_dat(
                work_dir.join(mdff_xmu_source_filename(polarization)),
                &XmuDatData {
                    header_lines: vec![format!("# xmu{polarization:02} MDFF source")],
                    normalization: None,
                    photon_energy_ev: Array1::from_vec(vec![12.5, 45.0]),
                    relative_energy_ev: Array1::from_vec(vec![12.5, 45.0]),
                    wave_number: Array1::from_vec(vec![0.25, 0.5]),
                    mu: mu.clone(),
                    mu0: Array1::zeros(2),
                    chi: mu,
                },
            )?;
        }
        Ok(())
    }

    fn manual_q_transition_tensor() -> Array3<f64> {
        Array3::from_shape_fn((2, 3, 3), |(energy, row, column)| {
            let i = (energy + 1) as f64;
            let j1 = (row + 1) as f64;
            let j2 = (column + 1) as f64;
            0.025 * i + 0.11 * j1 - 0.04 * j2 + 0.003 * i * j1 * j2
        })
    }

    fn mdff_xmu_source_filename(index: usize) -> String {
        match index {
            1 => "xmu.dat".to_string(),
            2..=9 => format!("xmu0{index}.dat"),
            10 => "xmu10.dat".to_string(),
            _ => format!("xmu{index}.dat"),
        }
    }

    fn write_global_input(work_dir: &Path, imdff: i32) -> Result<()> {
        write_global_input_with_q_vectors(work_dir, imdff, Vec::new())
    }

    fn write_global_input_with_q_vectors(
        work_dir: &Path,
        imdff: i32,
        q_vectors: Vec<GlobalQVector>,
    ) -> Result<()> {
        let q_count = i32::try_from(q_vectors.len()).context("q-vector count should fit i32")?;
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
                    do_nrixs: i32::from(q_count > 0),
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
                    nq: q_count,
                    imdff,
                    qaverage: true,
                    mixdff: false,
                },
                q_vectors,
                mdff: None,
            })?,
        )?;
        Ok(())
    }

    fn sample_global_q_vectors() -> Vec<GlobalQVector> {
        vec![
            GlobalQVector {
                q: [0.0, 0.0, -0.03755],
                norm: 0.03755,
                weight: [2.0, 0.0],
                trig: [-1.0, 0.0, 1.0, 0.0],
            },
            GlobalQVector {
                q: [0.0, -0.23240, -0.03755],
                norm: 0.235_415_948_907_885_9,
                weight: [0.0, -3.0],
                trig: [
                    -0.159_506_628_813_621_66,
                    0.987_196_794_294_329_4,
                    0.0,
                    -1.0,
                ],
            },
            GlobalQVector {
                q: [0.061, 0.025, -0.041],
                norm: 0.077_620_873_481_300_12,
                weight: [1.0, 1.0],
                trig: [
                    -0.528_208_254_578_435_3,
                    0.849_114_860_709_097_4,
                    0.925_769_429_955_506_9,
                    0.379_888_290_965_371_7,
                ],
            },
        ]
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

    fn generated_sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating MDFF for given experimental parameters - e.g. for simulating an EELS experiment".to_string(),
                "Starting MDFF module.".to_string(),
                "Reading Sigma tensor from file.".to_string(),
                "Calculating EELS cross-section.".to_string(),
                "Converting XAS to EELS.".to_string(),
                "Creating headers.".to_string(),
                "Entering big loop over energy.".to_string(),
                "Module mdff is finished.  Exiting.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 8],
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

    fn assert_complex_row_close(actual: ArrayView1<'_, Complex64>, expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_close(actual.re, expected.re);
            assert_close(actual.im, expected.im);
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, diff={}, tolerance={tolerance}",
            (actual - expected).abs()
        );
    }
}
