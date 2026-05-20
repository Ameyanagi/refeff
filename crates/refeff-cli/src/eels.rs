use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, ArrayView1};
use refeff_core::{
    EelsCollectionDependenceInput, EelsGosInput, EelsMeshInput, EelsMeshMode, EelsReadSpectrum,
    EelsReadSpectrumInput, EelsReadSpectrumSource, EelsSpectrumInput, FEFF_ELECTRON_REST_ENERGY_EV,
    FEFF_HBARC_ATOMIC, eels_collection_angle_dependence, eels_generalized_oscillator_strength,
    eels_read_spectrum, eels_spectrum, electron_wavelength_atomic_units,
};
use refeff_io::{
    EelsDatData, EelsGos1DatData, EelsGos2DatData, EelsInput, EelsMagicDatData, ModuleLogData,
    OpconsDatData, XmuDatData, eels_gos_dat_from_table, eels_magic_dat_from_collection_table,
    read_eels_dat, read_eels_gos1_dat, read_eels_gos2_dat, read_eels_magic_dat,
    read_module_log_dat, read_opcons_dat, read_xmu_dat, write_eels_dat, write_eels_gos1_dat,
    write_eels_gos2_dat, write_eels_magic_dat, write_module_log_dat,
};

use crate::work_dir_for_input;

const EELS_THETA0_RAD: f64 = 0.05 / 1000.0;
const EELS_XMU_INPUT: i32 = 1;
const EELS_OPCONS_KK_INPUT: i32 = 2;
const EELS_MAX_POLARIZATION_INDEX: usize = 10;

struct OwnedEelsSource {
    polarization_index: usize,
    energy_loss_ev: Array1<f64>,
    selected_spectrum: Array1<f64>,
    atomic_background: Array1<f64>,
    header_lines: Vec<String>,
}

struct GeneratedEelsOutput {
    spectrum: EelsDatData,
    magic: Option<EelsMagicDatData>,
    gos: Option<(EelsGos1DatData, EelsGos2DatData)>,
}

/// Run the supported FEFF EELS path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF EELS run has a cached output or enough source spectra.
pub(crate) fn has_cached_eels_output(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("eels.inp");
    if !input_path.is_file() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    if !input.calculate_elnes {
        return Ok(false);
    }
    if work_dir.join("eels.dat").is_file() {
        return Ok(true);
    }
    Ok(has_eels_source_spectra(work_dir, &input))
}

/// Run the FEFF EELS output path from cached or source spectra.
///
/// Existing `eels.dat` files are validated and re-rendered. When the cache is
/// missing, FEFF-style `xmu*.dat` or `opconsKK*.dat` source spectra are reduced
/// through the ported `readsp` tensor assembly and EELS q-integration routines.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !input.calculate_elnes {
        return Ok(0);
    }

    let output_path = work_dir.join("eels.dat");
    if output_path.is_file() {
        let data = read_eels_dat(&output_path)
            .with_context(|| format!("failed to read {}", output_path.display()))?;
        let point_count = data.point_count();
        write_cached_output(&output_path, &data)?;
        write_optional_module_log(&work_dir.join("logeels.dat"))?;
        write_optional_magic_dat(&work_dir.join("magic.dat"))?;
        write_optional_gos_dat(work_dir)?;
        return Ok(point_count);
    }

    let generated = generate_eels_output(work_dir, &input)?;
    let point_count = generated.spectrum.point_count();
    write_cached_output(&output_path, &generated.spectrum)?;
    if let Some(magic) = &generated.magic {
        write_magic_output(&work_dir.join("magic.dat"), magic)?;
    }
    if let Some((gos1, gos2)) = &generated.gos {
        write_gos_output(work_dir, gos1, gos2)?;
    }
    write_generated_module_log(
        work_dir,
        &input,
        point_count,
        generated.magic.as_ref(),
        generated.gos.as_ref(),
    )?;
    Ok(point_count)
}

fn has_eels_source_spectra(work_dir: &Path, input: &EelsInput) -> bool {
    let Ok(indices) = eels_source_indices(input) else {
        return false;
    };
    let Ok(prefix) = eels_source_prefix(input.control.input) else {
        return false;
    };
    indices
        .into_iter()
        .all(|index| work_dir.join(eels_source_filename(prefix, index)).is_file())
}

fn generate_eels_output(work_dir: &Path, input: &EelsInput) -> Result<GeneratedEelsOutput> {
    let sources = read_eels_sources(work_dir, input)?;
    let source_views = eels_source_views(&sources);
    let readsp = eels_read_spectrum(EelsReadSpectrumInput {
        sources: &source_views,
        orientation_averaged: input.control.average != 0,
        cross_terms: input.control.cross_terms != 0,
        polarization_min: positive_usize("polarization minimum", input.polarization.min)?,
        polarization_step: positive_usize("polarization step", input.polarization.step)?,
        polarization_max: positive_usize("polarization maximum", input.polarization.max)?,
    })
    .context("failed to assemble EELS source spectra")?;

    let spectrum = eels_spectrum(EelsSpectrumInput {
        incident_energy_ev: input.beam_energy,
        beam_direction: input.beam_direction,
        mesh: eels_mesh_input(input)?,
        energy_loss_ev: readsp.energy_loss_ev.view(),
        transition_tensor: readsp.transition_tensor.view(),
        atomic_background: readsp.atomic_background.view(),
        relativistic: input.control.relativistic != 0,
    })
    .context("failed to compute EELS spectrum")?;

    let magic = if input.magic == 0 {
        None
    } else {
        Some(eels_magic_output(input, &readsp)?)
    };
    let gos = if is_gos_mode(input) {
        Some(eels_gos_output(input, &readsp)?)
    } else {
        None
    };

    Ok(GeneratedEelsOutput {
        spectrum: EelsDatData {
            header_lines: eels_header_lines(input, &sources),
            energy_loss_ev: readsp.energy_loss_ev,
            total: spectrum.total,
            atomic_background: spectrum.background,
            fine_structure: spectrum.fine_structure,
            tensor: (input.control.average == 0).then_some(spectrum.partials),
        },
        magic,
        gos,
    })
}

fn eels_gos_output(
    input: &EelsInput,
    readsp: &EelsReadSpectrum,
) -> Result<(EelsGos1DatData, EelsGos2DatData)> {
    if input.control.average == 0 {
        bail!("EELS GOS mode requires orientation averaging");
    }
    let prefactors = eels_prefactors(input.beam_energy, readsp.energy_loss_ev.view())?;
    let averaged_spectrum = scaled_tensor_component(&readsp.transition_tensor, &prefactors, 0, 0);
    let table = eels_generalized_oscillator_strength(EelsGosInput {
        incident_energy_ev: input.beam_energy,
        energy_loss_ev: readsp.energy_loss_ev.view(),
        averaged_spectrum: averaged_spectrum.view(),
        relativistic: input.control.relativistic != 0,
    })
    .context("failed to compute EELS GOS tables")?;
    Ok(eels_gos_dat_from_table(table))
}

fn eels_magic_output(input: &EelsInput, readsp: &EelsReadSpectrum) -> Result<EelsMagicDatData> {
    let prefactors = eels_prefactors(input.beam_energy, readsp.energy_loss_ev.view())?;
    let sigma_x = scaled_tensor_component(&readsp.transition_tensor, &prefactors, 0, 0);
    let sigma_y = scaled_tensor_component(&readsp.transition_tensor, &prefactors, 1, 1);
    let pi_spectrum = scaled_tensor_component(&readsp.transition_tensor, &prefactors, 2, 2);
    let table = eels_collection_angle_dependence(EelsCollectionDependenceInput {
        incident_energy_ev: input.beam_energy,
        beam_direction: input.beam_direction,
        mesh: eels_mesh_input(input)?,
        magic_energy_ev: input.magic_energy,
        energy_loss_ev: readsp.energy_loss_ev.view(),
        sigma_x_spectrum: sigma_x.view(),
        sigma_y_spectrum: sigma_y.view(),
        pi_spectrum: pi_spectrum.view(),
        relativistic: input.control.relativistic != 0,
    })
    .context("failed to compute EELS magic-angle table")?;
    Ok(eels_magic_dat_from_collection_table(
        table.rows,
        table.point_counts,
    ))
}

fn eels_prefactors(
    incident_energy_ev: f64,
    energy_loss_ev: ArrayView1<'_, f64>,
) -> Result<Array1<f64>> {
    let incident_wavelength = electron_wavelength_atomic_units(incident_energy_ev)
        .context("failed to compute incident electron wavelength")?;
    let beam_factor = (1.0 + incident_energy_ev / FEFF_ELECTRON_REST_ENERGY_EV).powi(2)
        / std::f64::consts::PI
        * FEFF_HBARC_ATOMIC;
    let prefactors = energy_loss_ev
        .iter()
        .map(|&loss| {
            let scattered_energy_ev = incident_energy_ev - loss;
            electron_wavelength_atomic_units(scattered_energy_ev).map(|scattered_wavelength| {
                incident_wavelength / scattered_wavelength * beam_factor / loss
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to compute scattered electron wavelength")?;
    Ok(Array1::from_vec(prefactors))
}

fn scaled_tensor_component(
    tensor: &ndarray::Array3<f64>,
    prefactors: &Array1<f64>,
    row: usize,
    column: usize,
) -> Array1<f64> {
    Array1::from_iter(
        (0..tensor.dim().0)
            .map(|energy_index| tensor[(energy_index, row, column)] * prefactors[energy_index]),
    )
}

fn read_eels_sources(work_dir: &Path, input: &EelsInput) -> Result<Vec<OwnedEelsSource>> {
    eels_source_indices(input)?
        .into_iter()
        .map(|index| read_eels_source(work_dir, input, index))
        .collect()
}

fn read_eels_source(work_dir: &Path, input: &EelsInput, index: usize) -> Result<OwnedEelsSource> {
    let prefix = eels_source_prefix(input.control.input)?;
    let path = work_dir.join(eels_source_filename(prefix, index));
    if input.control.input == EELS_XMU_INPUT {
        let data =
            read_xmu_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        source_from_xmu(index, input.control.spectrum_column, data)
    } else {
        let data =
            read_opcons_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        source_from_opcons(index, input.control.spectrum_column, data)
    }
}

fn source_from_xmu(index: usize, column: i32, data: XmuDatData) -> Result<OwnedEelsSource> {
    let selected_spectrum = match column {
        1 => data.photon_energy_ev.clone(),
        2 => data.relative_energy_ev.clone(),
        3 => data.wave_number.clone(),
        4 => data.mu.clone(),
        5 => data.mu0.clone(),
        6 => data.chi.clone(),
        _ => bail!("EELS xmu spectrum column {column} is outside the supported 1..=6 range"),
    };
    Ok(OwnedEelsSource {
        polarization_index: index,
        energy_loss_ev: data.photon_energy_ev,
        selected_spectrum,
        atomic_background: data.mu0,
        header_lines: data.header_lines,
    })
}

fn source_from_opcons(index: usize, column: i32, data: OpconsDatData) -> Result<OwnedEelsSource> {
    let selected_spectrum = match column {
        1 => data.energy_ev.clone(),
        2 => data.epsilon_minus_one.mapv(|value| value.re),
        3 => data.epsilon_minus_one.mapv(|value| value.im),
        4 => data.refractive_index_minus_one.mapv(|value| value.re),
        5 => data.refractive_index_minus_one.mapv(|value| value.im),
        6 => data.absorption_coefficient.clone(),
        7 => data.reflectivity.clone(),
        8 => data.loss.clone(),
        _ => bail!("EELS opconsKK spectrum column {column} is outside the supported 1..=8 range"),
    };
    Ok(OwnedEelsSource {
        polarization_index: index,
        energy_loss_ev: data.energy_ev,
        selected_spectrum,
        atomic_background: data
            .refractive_index_minus_one
            .mapv(|refractive_index| refractive_index.im),
        header_lines: data.header_lines,
    })
}

fn eels_source_views(sources: &[OwnedEelsSource]) -> Vec<EelsReadSpectrumSource<'_>> {
    sources
        .iter()
        .map(|source| EelsReadSpectrumSource {
            polarization_index: source.polarization_index,
            energy_loss_ev: source.energy_loss_ev.view(),
            selected_spectrum: source.selected_spectrum.view(),
            atomic_background: source.atomic_background.view(),
        })
        .collect()
}

fn eels_source_indices(input: &EelsInput) -> Result<Vec<usize>> {
    let min = positive_usize("polarization minimum", input.polarization.min)?;
    let step = positive_usize("polarization step", input.polarization.step)?;
    let max = positive_usize("polarization maximum", input.polarization.max)?;
    if min > max || max > EELS_MAX_POLARIZATION_INDEX {
        bail!(
            "invalid EELS polarization range: min={min}, step={step}, max={max}; expected 1..=10"
        );
    }
    Ok((min..=max).step_by(step).collect())
}

fn positive_usize(name: &'static str, value: i32) -> Result<usize> {
    if value <= 0 {
        bail!("EELS {name} must be positive, got {value}");
    }
    usize::try_from(value).with_context(|| format!("failed to convert EELS {name}"))
}

fn eels_source_prefix(input_kind: i32) -> Result<&'static str> {
    match input_kind {
        EELS_XMU_INPUT => Ok("xmu"),
        EELS_OPCONS_KK_INPUT => Ok("opconsKK"),
        _ => bail!(
            "unsupported EELS input source {input_kind}; expected 1 for xmu or 2 for opconsKK"
        ),
    }
}

fn eels_source_filename(prefix: &str, index: usize) -> String {
    match index {
        1 => format!("{prefix}.dat"),
        2..=9 => format!("{prefix}0{index}.dat"),
        10 => format!("{prefix}10.dat"),
        _ => format!("{prefix}{index}.dat"),
    }
}

fn eels_mesh_input(input: &EelsInput) -> Result<EelsMeshInput> {
    Ok(EelsMeshInput {
        collection_angle: input.angles.collection,
        convergence_angle: input.angles.convergence,
        theta0: EELS_THETA0_RAD,
        theta_x_center: input.detector[0],
        theta_y_center: input.detector[1],
        radial_count: positive_usize("radial q-mesh count", input.qmesh.radial)?,
        angular_count: positive_usize("angular q-mesh count", input.qmesh.angular)?,
        mode: if is_gos_mode(input) {
            EelsMeshMode::OneDimensional
        } else {
            EelsMeshMode::Logarithmic
        },
    })
}

fn is_gos_mode(input: &EelsInput) -> bool {
    input.calculation_mode == 9
}

fn eels_header_lines(input: &EelsInput, sources: &[OwnedEelsSource]) -> Vec<String> {
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
    lines.push("# Units are a_0^2 / eV.  Multiply by 28.00 10^-18  to get cm^-2 / eV.  Or by 28 to get Mbarn / eV.".to_string());
    lines.push(eels_relativity_header(input));
    lines.extend(eels_source_header_lines(sources));
    lines.push(eels_column_header(input).to_string());
    lines
}

fn eels_relativity_header(input: &EelsInput) -> String {
    match (
        input.control.relativistic != 0,
        input.control.cross_terms != 0,
    ) {
        (true, true) => "# Relativistic and cross-terms.",
        (true, false) => "# Relativistic, no cross-terms.",
        (false, true) => "# Nonrelativistic and cross-terms.",
        (false, false) => "# Nonrelativistic, no cross-terms.",
    }
    .to_string()
}

fn eels_source_header_lines(sources: &[OwnedEelsSource]) -> Vec<String> {
    sources
        .last()
        .into_iter()
        .flat_map(|source| source.header_lines.iter())
        .filter(|line| eels_keeps_source_header_line(line))
        .take(5)
        .cloned()
        .collect()
}

fn eels_keeps_source_header_line(line: &str) -> bool {
    let prefix = line
        .chars()
        .skip_while(|character| matches!(character, ' ' | '#'))
        .take(3)
        .collect::<String>();
    matches!(prefix.as_str(), "FMS" | "Gam" | "S02" | "POT" | "Ene")
}

fn eels_column_header(input: &EelsInput) -> &'static str {
    if input.control.average != 0 {
        "#  Energy       total         atomic-bg     fine-struct"
    } else {
        concat!(
            "#  Energy       total         atomic-bg     fine-struct   xx            xy            xz",
            "            yx            yy            yz            zx            zy            zz"
        )
    }
}

fn read_input(work_dir: &Path) -> Result<EelsInput> {
    let input_path = work_dir.join("eels.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    EelsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(path: &Path, data: &EelsDatData) -> Result<()> {
    write_eels_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn write_optional_magic_dat(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_eels_magic_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_magic_output(path, &data)
}

fn write_magic_output(path: &Path, data: &EelsMagicDatData) -> Result<()> {
    write_eels_magic_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_gos_dat(work_dir: &Path) -> Result<()> {
    let gos1_path = work_dir.join("gos1.txt");
    if gos1_path.is_file() {
        let data = read_eels_gos1_dat(&gos1_path)
            .with_context(|| format!("failed to read {}", gos1_path.display()))?;
        write_eels_gos1_dat(&gos1_path, &data)
            .with_context(|| format!("failed to write {}", gos1_path.display()))?;
    }
    let gos2_path = work_dir.join("gos2.txt");
    if gos2_path.is_file() {
        let data = read_eels_gos2_dat(&gos2_path)
            .with_context(|| format!("failed to read {}", gos2_path.display()))?;
        write_eels_gos2_dat(&gos2_path, &data)
            .with_context(|| format!("failed to write {}", gos2_path.display()))?;
    }
    Ok(())
}

fn write_gos_output(work_dir: &Path, gos1: &EelsGos1DatData, gos2: &EelsGos2DatData) -> Result<()> {
    let gos1_path = work_dir.join("gos1.txt");
    write_eels_gos1_dat(&gos1_path, gos1)
        .with_context(|| format!("failed to write {}", gos1_path.display()))?;
    let gos2_path = work_dir.join("gos2.txt");
    write_eels_gos2_dat(&gos2_path, gos2)
        .with_context(|| format!("failed to write {}", gos2_path.display()))
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_generated_module_log(
    work_dir: &Path,
    input: &EelsInput,
    point_count: usize,
    magic: Option<&EelsMagicDatData>,
    gos: Option<&(EelsGos1DatData, EelsGos2DatData)>,
) -> Result<()> {
    let path = work_dir.join("logeels.dat");
    if path.is_file() {
        return write_optional_module_log(&path);
    }
    let mut lines = vec![
        "Calculating EELS spectra ...".to_string(),
        format!("Beam energy={:8.2} keV", input.beam_energy / 1000.0),
        format!(
            "Beam direction={:6.3} {:6.3} {:6.3} in coordinate frame of feff.inp",
            input.beam_direction[0], input.beam_direction[1], input.beam_direction[2]
        ),
        format!(
            "Collection semiangle={:6.2} mrad  convergence semiangle={:6.2} mrad",
            input.angles.collection * 1000.0,
            input.angles.convergence * 1000.0
        ),
        format!("Generated eels.dat with {point_count} energy point(s)."),
    ];
    if let Some(magic) = magic {
        lines.push(format!(
            "Generated magic.dat with {} collection-angle row(s).",
            magic.point_count()
        ));
    }
    if let Some((gos1, gos2)) = gos {
        lines.push(format!(
            "Generated gos1.txt with {} q row(s) and gos2.txt with {} energy row(s).",
            gos1.point_count(),
            gos2.energy_count()
        ));
    }
    lines.push("Done with module: EELS.".to_string());
    let data = ModuleLogData {
        line_terminators: vec!["\n".to_string(); lines.len()],
        lines,
    };
    write_module_log(&path, &data)
}

#[cfg(test)]
mod tests {
    use super::{EELS_THETA0_RAD, eels_source_filename, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{ArrayView1, ArrayView2, array};
    use refeff_io::{
        EelsDatData, EelsGos1DatData, EelsGos2DatData, EelsMagicDatData, ModuleLogData,
        read_eels_dat, read_eels_gos1_dat, read_eels_gos2_dat, read_eels_magic_dat,
        read_module_log_dat, write_eels_dat, write_eels_gos1_dat, write_eels_gos2_dat,
        write_eels_magic_dat, write_module_log_dat,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn eels_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_eels_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("eels.dat").exists());
        Ok(())
    }

    #[test]
    fn eels_module_rejects_enabled_generation_without_source_spectra() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_eels_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled EELS should require source spectra")?;

        let message = error.to_string();
        assert!(message.contains("failed to read"));
        assert!(message.contains("xmu.dat"));
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
                i32::from(enabled),
                0,
                1,
                1,
                1,
                4,
                1,
                1,
                9,
                300_000.0,
                0.0,
                1.0,
                0.0,
                0.0024,
                0.0,
                5,
                3,
                0.0,
                0.0,
                magic,
                magic_energy,
            ),
        )?;
        Ok(())
    }

    fn write_eels_gos_input(work_dir: &Path) -> Result<()> {
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
                9,
                1,
                1,
                1,
                1,
                4,
                1,
                1,
                9,
                300_000.0,
                0.0,
                0.0,
                1.0,
                0.0024,
                0.0,
                5,
                3,
                0.0,
                0.0,
                0,
                0.0,
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
        for (index, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
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
        let tolerance =
            absolute_tolerance.max(relative_tolerance * actual.abs().max(expected.abs()));
        let difference = (actual - expected).abs();
        assert!(
            difference <= tolerance,
            "{name}: actual {actual:e} expected {expected:e} diff {difference:e} tolerance {tolerance:e}"
        );
    }
}
