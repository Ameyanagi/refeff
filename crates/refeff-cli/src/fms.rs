use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array3, Axis, ShapeBuilder};
use num_complex::Complex32;
use refeff_core::{
    Complex, FEFF_BOHR_ANGSTROM, MkgtrGreenTraceInput, TransitionBMatrixInput,
    core_hole_quantum_numbers, mkgtr_green_trace, transition_b_matrix,
};
use refeff_io::{
    FmsBinData, FmsInput, FmslBinData, GgDatData, GlobalInput, GtrBinData, GtrDatData, GtrlDatData,
    PhaseBinData, read_fms_bin, read_fmsl_bin, read_gg_bin, read_gg_dat, read_gtr_bin,
    read_gtr_dat, read_gtrl_dat, read_module_log_dat, read_phase_bin, write_fms_bin,
    write_fmsl_bin, write_gg_bin, write_gg_dat, write_gtr_bin, write_gtr_dat, write_gtrl_dat,
    write_module_log_dat,
};

use crate::work_dir_for_input;

/// Run the supported FEFF FMS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FMS/MKGTR run can be satisfied from existing caches.
pub(crate) fn has_cached_fms_output(work_dir: &Path) -> Result<bool> {
    if cached_output_paths(work_dir)?.is_empty() {
        return Ok(false);
    }
    Ok(fms_enabled(&read_input(work_dir)?))
}

/// Run the FEFF FMS/MKGTR cached-output path from existing handoff files.
///
/// The full multiple-scattering solver is still unported. This preserves
/// cached FEFF directories by validating and re-rendering typed
/// `gg.bin`/`gg.dat`, `fms.bin`, `fmsl.bin`, `gtr.dat`, `gtrNN.bin`,
/// `gtrl.dat`, and optional `log3.dat` diagnostic handoffs. When a cached
/// absorber `gg` matrix exists with `phase.bin` and non-NRIXS `global.inp`,
/// the port also folds the Green's functions through MKGTR to generate missing
/// `fms.bin` and `gtr.dat` files.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !fms_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        bail!("FMS Green's-function generation requires the unported FMS numerical solver");
    }

    let fms_metadata = if outputs
        .iter()
        .any(|output| output.kind == CachedOutputKind::FmslBin)
    {
        let fms_path = work_dir.join("fms.bin");
        Some(
            read_fms_bin(&fms_path)
                .with_context(|| format!("failed to read {}", fms_path.display()))?,
        )
    } else {
        None
    };

    for output in &outputs {
        match output.kind {
            CachedOutputKind::FmsBin => {
                let data = read_fms_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_fms_cache(&output.path, &data)?;
            }
            CachedOutputKind::FmslBin => {
                let metadata = fms_metadata
                    .as_ref()
                    .context("fmsl.bin cache requires fms.bin metadata")?;
                let max_channel = decomposition_channel(&input)?;
                let data = read_fmsl_bin(
                    &output.path,
                    metadata.pad_width,
                    metadata.energy_count,
                    max_channel,
                )
                .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_fmsl_cache(&output.path, &data)?;
            }
            CachedOutputKind::GgBin => {
                let data = read_gg_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gg_bin_cache(&output.path, &data)?;
            }
            CachedOutputKind::GgDat => {
                let data = read_gg_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gg_dat_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrBin => {
                let data = read_gtr_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtr_bin_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrDat => {
                let data = read_gtr_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtr_dat_cache(&output.path, &data)?;
            }
            CachedOutputKind::GtrlDat => {
                let data = read_gtrl_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_gtrl_dat_cache(&output.path, &data)?;
            }
        }
    }

    let generated = generate_mkgtr_outputs_from_cached_gg(work_dir, &input, &outputs)?;

    Ok(outputs.len() + generated + write_optional_module_log(&work_dir.join("log3.dat"))?)
}

fn fms_enabled(input: &FmsInput) -> bool {
    input.control.mfms != 0
}

fn decomposition_channel(input: &FmsInput) -> Result<usize> {
    if input.decomposition_channels < 0 {
        bail!("fmsl.bin cache requires a nonnegative FMS decomposition channel count");
    }
    Ok(input.decomposition_channels as usize)
}

fn read_input(work_dir: &Path) -> Result<FmsInput> {
    let input_path = work_dir.join("fms.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    FmsInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_fms_cache(path: &Path, data: &FmsBinData) -> Result<()> {
    write_fms_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_fmsl_cache(path: &Path, data: &FmslBinData) -> Result<()> {
    write_fmsl_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gg_bin_cache(path: &Path, data: &GgDatData) -> Result<()> {
    write_gg_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gg_dat_cache(path: &Path, data: &GgDatData) -> Result<()> {
    write_gg_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtr_bin_cache(path: &Path, data: &GtrBinData) -> Result<()> {
    write_gtr_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtr_dat_cache(path: &Path, data: &GtrDatData) -> Result<()> {
    write_gtr_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_gtrl_dat_cache(path: &Path, data: &GtrlDatData) -> Result<()> {
    write_gtrl_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn generate_mkgtr_outputs_from_cached_gg(
    work_dir: &Path,
    input: &FmsInput,
    outputs: &[CachedOutputPath],
) -> Result<usize> {
    let fms_path = work_dir.join("fms.bin");
    let gtr_path = work_dir.join("gtr.dat");
    let needs_fms = !fms_path.is_file();
    let needs_gtr = !gtr_path.is_file();
    if !needs_fms && !needs_gtr {
        return Ok(0);
    }

    let Some(gg_output) = cached_gg_output(outputs) else {
        return Ok(0);
    };
    let phase_path = work_dir.join("phase.bin");
    let global_path = work_dir.join("global.inp");
    if !phase_path.is_file() || !global_path.is_file() {
        return Ok(0);
    }

    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let global_text = std::fs::read_to_string(&global_path)
        .with_context(|| format!("failed to read {}", global_path.display()))?;
    let global = GlobalInput::parse_str(&global_path, &global_text)
        .with_context(|| format!("failed to parse {}", global_path.display()))?;
    if global.control.do_nrixs != 0 {
        return Ok(0);
    }

    let gg = read_cached_gg(gg_output)?;
    let generated = build_mkgtr_outputs(input, &global, &phase, &gg)?;
    let mut count = 0;
    if needs_fms {
        write_fms_cache(&fms_path, &generated.fms)?;
        count += 1;
    }
    if needs_gtr {
        write_gtr_dat_cache(&gtr_path, &generated.gtr)?;
        count += 1;
    }
    Ok(count)
}

fn cached_gg_output(outputs: &[CachedOutputPath]) -> Option<&CachedOutputPath> {
    outputs
        .iter()
        .find(|output| output.kind == CachedOutputKind::GgBin)
        .or_else(|| {
            outputs
                .iter()
                .find(|output| output.kind == CachedOutputKind::GgDat)
        })
}

fn read_cached_gg(output: &CachedOutputPath) -> Result<GgDatData> {
    match output.kind {
        CachedOutputKind::GgBin => read_gg_bin(&output.path)
            .with_context(|| format!("failed to read {}", output.path.display())),
        CachedOutputKind::GgDat => read_gg_dat(&output.path)
            .with_context(|| format!("failed to read {}", output.path.display())),
        _ => bail!("internal FMS error: expected gg cache path"),
    }
}

struct GeneratedMkgtrOutputs {
    fms: FmsBinData,
    gtr: GtrDatData,
}

fn build_mkgtr_outputs(
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
) -> Result<GeneratedMkgtrOutputs> {
    let absorber_lmax = absorber_lmax(input)?;
    let active_spin_channels = active_spin_channels(global, phase)?;
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .with_context(|| format!("failed to map ihole {} to core-hole kappa", phase.ihole))?;
    let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
        lmax: absorber_lmax,
        initial_kappa: core_hole.kappa,
        polarization: global.control.ipol,
        polarization_tensor: polarization_tensor(global),
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channels: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })
    .context("failed to build MKGTR transition B matrix")?;
    let green_functions = green_functions_from_gg(gg, phase.energy_count)?;
    let transition_moments = phase.transition_moments.index_axis(Axis(1), 0);
    let trace = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels,
        green_functions: green_functions.view(),
        transition_matrices: &[transition_matrix],
        transition_moments,
    })
    .context("failed to fold cached gg matrices into MKGTR trace")?;

    let fms = FmsBinData {
        cluster_radius_angstrom: input.cluster.rfms2 * FEFF_BOHR_ANGSTROM,
        energy_count: phase.energy_count,
        main_energy_count: phase.main_energy_count,
        auxiliary_energy_count: phase.auxiliary_energy_count,
        highest_potential_index: phase
            .potential_count()
            .checked_sub(1)
            .context("phase.bin requires at least one potential")?,
        pad_width: phase.pad_width,
        declared_spectrum_count: Some(0),
        spectra: trace.traces.clone(),
    };
    let gtr = GtrDatData {
        energy: phase.energy_grid.clone(),
        trace: trace.traces.row(0).to_owned(),
    };
    Ok(GeneratedMkgtrOutputs { fms, gtr })
}

fn absorber_lmax(input: &FmsInput) -> Result<usize> {
    let value = *input
        .lmaxph
        .first()
        .context("FMS input requires lmaxph(0) for MKGTR trace generation")?;
    if value < 0 {
        bail!("FMS lmaxph(0) must be nonnegative for MKGTR trace generation");
    }
    usize::try_from(value).context("failed to convert FMS lmaxph(0)")
}

fn active_spin_channels(global: &GlobalInput, phase: &PhaseBinData) -> Result<usize> {
    if phase.spin_count == 0 {
        bail!("phase.bin requires at least one spin channel for MKGTR trace generation");
    }
    if global.control.ispin.abs() == 1 {
        Ok(phase.spin_count)
    } else {
        Ok(1)
    }
}

fn polarization_tensor(global: &GlobalInput) -> [[Complex; 3]; 3] {
    let mut tensor = [[Complex::new(0.0, 0.0); 3]; 3];
    for (row_index, row) in global.polarization_tensor.iter().enumerate() {
        tensor[row_index] = [
            Complex::new(row[0], row[1]),
            Complex::new(row[2], row[3]),
            Complex::new(row[4], row[5]),
        ];
    }
    tensor
}

fn green_functions_from_gg(gg: &GgDatData, energy_count: usize) -> Result<Array3<Complex32>> {
    if gg.sections.len() != energy_count {
        bail!(
            "gg cache section count {} does not match phase.bin energy count {energy_count}",
            gg.sections.len()
        );
    }
    let first = gg
        .sections
        .first()
        .context("gg cache requires at least one section")?;
    let (rows, columns) = first.shape();
    if rows == 0 || rows != columns {
        bail!("gg cache sections must be nonempty square matrices");
    }

    let mut green_functions = Array3::zeros((energy_count, rows, columns).f());
    for (energy, section) in gg.sections.iter().enumerate() {
        let shape = section.shape();
        if shape != (rows, columns) {
            bail!(
                "gg cache section {} shape {:?} does not match first section shape {:?}",
                section.section_number,
                shape,
                (rows, columns)
            );
        }
        for row in 0..rows {
            for column in 0..columns {
                green_functions[(energy, row, column)] =
                    narrow_complex64_to_complex32(section.values[(row, column)], "gg")?;
            }
        }
    }
    Ok(green_functions)
}

fn narrow_complex64_to_complex32(value: Complex, table: &'static str) -> Result<Complex32> {
    let narrowed = Complex32::new(value.re as f32, value.im as f32);
    if value.re.is_finite()
        && value.im.is_finite()
        && narrowed.re.is_finite()
        && narrowed.im.is_finite()
    {
        Ok(narrowed)
    } else {
        bail!("{table} contains a non-finite or out-of-range complex value")
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    FmsBin,
    FmslBin,
    GgBin,
    GgDat,
    GtrBin,
    GtrDat,
    GtrlDat,
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
        let kind = if name == "fms.bin" {
            Some(CachedOutputKind::FmsBin)
        } else if name == "fmsl.bin" {
            Some(CachedOutputKind::FmslBin)
        } else if name == "gg.bin" {
            Some(CachedOutputKind::GgBin)
        } else if name == "gg.dat" {
            Some(CachedOutputKind::GgDat)
        } else if name == "gtr.dat" {
            Some(CachedOutputKind::GtrDat)
        } else if name == "gtrl.dat" {
            Some(CachedOutputKind::GtrlDat)
        } else if is_gtr_bin_name(name) {
            Some(CachedOutputKind::GtrBin)
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

fn is_gtr_bin_name(name: &str) -> bool {
    name.strip_prefix("gtr")
        .and_then(|tail| tail.strip_suffix(".bin"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
mod tests;
