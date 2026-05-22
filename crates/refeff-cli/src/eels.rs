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
mod tests;
