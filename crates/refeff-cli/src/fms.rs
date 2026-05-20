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
mod tests {
    use super::{has_cached_fms_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4, Axis, ShapeBuilder};
    use num_complex::Complex64;
    use refeff_core::{
        MkgtrGreenTraceInput, TransitionBMatrixInput, core_hole_quantum_numbers, mkgtr_green_trace,
        transition_b_matrix,
    };
    use refeff_io::{
        CfAverage, FmsBinData, FmsCluster, FmsControl, FmsDebye, FmsInput, FmslBinData, GgDatData,
        GgDatSection, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GtrBinData,
        GtrDatData, GtrlDatData, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
        fms_input_string, global_input_string, parse_gtrl_dat, read_fms_bin, read_fmsl_bin,
        read_gg_bin, read_gg_dat, read_gtr_bin, read_gtr_dat, read_gtrl_dat, read_module_log_dat,
        write_fms_bin, write_fmsl_bin, write_gg_bin, write_gg_dat, write_gtr_bin, write_gtr_dat,
        write_gtrl_dat, write_module_log_dat, write_phase_bin,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn fms_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 0, -1)?;
        write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_fms_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn fms_module_rejects_generation_until_solver_is_ported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 1, -1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled FMS should require the numerical solver")?;

        assert!(error.to_string().contains(
            "FMS Green's-function generation requires the unported FMS numerical solver"
        ));
        Ok(())
    }

    #[test]
    fn fms_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input(temp.path(), 1, 2)?;
        write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;
        write_fmsl_bin(temp.path().join("fmsl.bin"), &sample_fmsl_bin())?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_gg_dat())?;
        write_gg_dat(temp.path().join("gg.dat"), &sample_gg_dat())?;
        write_gtr_dat(temp.path().join("gtr.dat"), &sample_gtr_dat())?;
        write_gtr_bin(temp.path().join("gtr00.bin"), &sample_gtr_bin())?;
        write_gtrl_dat(temp.path().join("gtrl.dat"), &sample_gtrl_dat()?)?;
        write_module_log_dat(temp.path().join("log3.dat"), &sample_module_log())?;

        let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let expected_fmsl = read_fmsl_bin(
            temp.path().join("fmsl.bin"),
            expected_fms.pad_width,
            expected_fms.energy_count,
            2,
        )?;
        let expected_gg_bin = read_gg_bin(temp.path().join("gg.bin"))?;
        let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
        let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
        let expected_gtr_bin = read_gtr_bin(temp.path().join("gtr00.bin"))?;
        let expected_gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;
        let expected_log = read_module_log_dat(temp.path().join("log3.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 8);
        assert!(has_cached_fms_output(temp.path())?);
        assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
        assert_eq!(
            read_fmsl_bin(
                temp.path().join("fmsl.bin"),
                expected_fms.pad_width,
                expected_fms.energy_count,
                2,
            )?,
            expected_fmsl
        );
        assert_eq!(read_gg_bin(temp.path().join("gg.bin"))?, expected_gg_bin);
        assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
        assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
        assert_eq!(
            read_gtr_bin(temp.path().join("gtr00.bin"))?,
            expected_gtr_bin
        );
        assert_eq!(read_gtrl_dat(temp.path().join("gtrl.dat"))?, expected_gtrl);
        assert_eq!(
            read_module_log_dat(temp.path().join("log3.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn fms_module_generates_mkgtr_outputs_from_cached_gg() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_fms_input_with_lmax(temp.path(), 1, -1, &[1])?;
        let global = sample_global_input();
        std::fs::write(
            temp.path().join("global.inp"),
            global_input_string(&global)?,
        )?;
        let phase = sample_phase_bin();
        write_phase_bin(temp.path().join("phase.bin"), &phase)?;
        let gg = sample_mkgtr_gg();
        write_gg_bin(temp.path().join("gg.bin"), &gg)?;

        let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
        assert_eq!(fms.declared_spectrum_count, Some(0));
        assert_eq!(fms.energy_count, phase.energy_count);
        assert_eq!(fms.main_energy_count, phase.main_energy_count);
        assert_eq!(fms.highest_potential_index, phase.potential_count() - 1);
        assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 1.0e-8);
        assert_eq!(gtr.energy, phase.energy_grid);
        assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);
        Ok(())
    }

    #[test]
    fn fms_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_fms_dir()? else {
            eprintln!("skipping FMS reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
        for name in required {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        for name in ["gg.bin", "gtrl.dat", "fmsl.bin", "log3.dat"] {
            let source = reference_dir.join(name);
            if source.is_file() {
                std::fs::copy(source, temp.path().join(name))?;
            }
        }
        copy_gtr_bin_references(&reference_dir, temp.path())?;

        let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
        let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
        let expected_log = optional_module_log(temp.path().join("log3.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert!(count >= required.len() - 1);
        assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
        assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
        assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
        if let Some(expected) = expected_log {
            assert_eq!(read_module_log_dat(temp.path().join("log3.dat"))?, expected);
        }
        Ok(())
    }

    fn write_fms_input(work_dir: &Path, mfms: i32, decomposition_channels: i32) -> Result<()> {
        write_fms_input_with_lmax(work_dir, mfms, decomposition_channels, &[2, 2])
    }

    fn write_fms_input_with_lmax(
        work_dir: &Path,
        mfms: i32,
        decomposition_channels: i32,
        lmaxph: &[i32],
    ) -> Result<()> {
        let input = FmsInput {
            control: FmsControl {
                mfms,
                idwopt: 0,
                minv: 0,
            },
            cluster: FmsCluster {
                rfms2: -1.0,
                rdirec: -1.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: FmsDebye {
                tk: 190.0,
                thetad: 315.0,
                sig2g: 0.0,
            },
            lmaxph: lmaxph.to_vec(),
            decomposition_channels,
            save_gg_slice: false,
            do_fms: 0,
        };
        std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
        Ok(())
    }

    fn sample_global_input() -> GlobalInput {
        GlobalInput {
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
                do_nrixs: 0,
                ldecmx: 0,
                lj: 0,
            },
            evec: [0.0, 0.0, 1.0],
            xivec: [1.0, 0.0, 0.0],
            spvec: [0.0, 0.0, 1.0],
            polarization_tensor: [
                [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
            ],
            norms: GlobalNorms {
                evnorm: 1.0,
                xivnorm: 1.0,
                spvnorm: 1.0,
            },
            q_control: GlobalQControl {
                nq: 0,
                imdff: 0,
                qaverage: false,
                mixdff: false,
            },
            q_vectors: Vec::new(),
            mdff: None,
        }
    }

    fn sample_phase_bin() -> PhaseBinData {
        let energy_count = 2;
        let spin_count = 1;
        let transition_count = 8;
        let energy_grid =
            Array1::from_vec(vec![Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)]);
        let reference_energy =
            Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
                Complex64::new(0.01 * (energy + 1) as f64, -0.02 * spin as f64)
            });
        let phase_shifts =
            Array3::from_shape_fn((energy_count, 3, spin_count), |(energy, angular, spin)| {
                Complex64::new(
                    0.1 * (energy + 1) as f64 + 0.01 * angular as f64,
                    -0.005 * spin as f64,
                )
            });
        let mut transition_moments =
            Array4::<Complex64>::zeros((energy_count, 1, transition_count, spin_count).f());
        for energy in 0..energy_count {
            for transition in 0..transition_count {
                transition_moments[(energy, 0, transition, 0)] = Complex64::new(
                    0.25 + 0.1 * energy as f64 + 0.03 * transition as f64,
                    -0.02 * transition as f64,
                );
            }
        }

        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 1,
            pad_width: 8,
            final_state_count: transition_count,
            transition_count,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: 0.0,
                edge_energy: 8_979.0,
            },
            energy_grid,
            reference_energy,
            potentials: vec![PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts,
            }],
            transition_moments,
            raw_pads: None,
        }
    }

    fn sample_mkgtr_gg() -> GgDatData {
        GgDatData {
            sections: (0..2)
                .map(|energy| GgDatSection {
                    section_number: energy + 1,
                    values: Array2::from_shape_fn((4, 4), |(row, column)| {
                        let base =
                            0.15 + 0.2 * energy as f64 + 0.03 * row as f64 + 0.01 * column as f64;
                        Complex64::new(base, -0.5 * base)
                    }),
                    raw_prefix_lines: None,
                })
                .collect(),
        }
    }

    fn expected_mkgtr_trace(
        global: &GlobalInput,
        phase: &PhaseBinData,
        gg: &GgDatData,
        lmax: usize,
    ) -> Result<Array2<Complex64>> {
        let core_hole = core_hole_quantum_numbers(phase.ihole)?;
        let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
            lmax,
            initial_kappa: core_hole.kappa,
            polarization: global.control.ipol,
            polarization_tensor: super::polarization_tensor(global),
            multipole: global.control.le2,
            trace_orbital: false,
            spin: global.control.ispin,
            spin_channels: phase.spin_count,
            spin_vector_angle: global.control.angks,
        })?;
        let green_functions = super::green_functions_from_gg(gg, phase.energy_count)?;
        let transition_moments = phase.transition_moments.index_axis(Axis(1), 0);
        Ok(mkgtr_green_trace(MkgtrGreenTraceInput {
            active_spin_channels: 1,
            green_functions: green_functions.view(),
            transition_matrices: &[transition_matrix],
            transition_moments,
        })?
        .traces)
    }

    fn assert_complex_table_close(
        actual: ndarray::ArrayView2<'_, Complex64>,
        expected: ndarray::ArrayView2<'_, Complex64>,
        tolerance: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= tolerance,
                "complex table mismatch at {index}: actual={actual:?} expected={expected:?}"
            );
        }
    }

    fn assert_complex_vec_close(
        actual: ndarray::ArrayView1<'_, Complex64>,
        expected: ndarray::ArrayView1<'_, Complex64>,
        tolerance: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= tolerance,
                "complex vector mismatch at {index}: actual={actual:?} expected={expected:?}"
            );
        }
    }

    fn sample_fms_bin() -> FmsBinData {
        FmsBinData {
            cluster_radius_angstrom: 5.5,
            energy_count: 2,
            main_energy_count: 1,
            auxiliary_energy_count: 0,
            highest_potential_index: 1,
            pad_width: 8,
            declared_spectrum_count: Some(2),
            spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
                Complex64::new(
                    0.25 * (energy + 1) as f64 + spectrum as f64,
                    -0.05 * (energy + 1) as f64 - spectrum as f64,
                )
            }),
        }
    }

    fn sample_fmsl_bin() -> FmslBinData {
        FmslBinData {
            pad_width: 8,
            max_decomposition_channel: 2,
            traces: Array3::from_shape_fn((2, 3, 3), |(energy, lg2, lg1)| {
                Complex64::new(
                    energy as f64 + 0.1 * lg2 as f64 + 0.01 * lg1 as f64,
                    -(energy as f64) - 0.2 * lg2 as f64 - 0.02 * lg1 as f64,
                )
            }),
        }
    }

    fn sample_gg_dat() -> GgDatData {
        GgDatData {
            sections: vec![
                GgDatSection {
                    section_number: 1,
                    values: Array2::from_shape_fn((2, 2), |(row, column)| {
                        let value = 1.0 + row as f64 + 2.0 * column as f64;
                        Complex64::new(value, -0.5 * value)
                    }),
                    raw_prefix_lines: None,
                },
                GgDatSection {
                    section_number: 2,
                    values: Array2::from_shape_fn((1, 2), |(_, column)| {
                        let value = 5.0 + column as f64;
                        Complex64::new(value, -value - 0.5)
                    }),
                    raw_prefix_lines: None,
                },
            ],
        }
    }

    fn sample_gtr_dat() -> GtrDatData {
        GtrDatData {
            energy: Array1::from_vec(vec![
                Complex64::new(-0.138_801, 0.031_773),
                Complex64::new(-0.137_401, 0.031_773),
                Complex64::new(55.866_911, 0.031_773),
            ]),
            trace: Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.624_106, 1.081_113),
            ]),
        }
    }

    fn sample_gtr_bin() -> GtrBinData {
        GtrBinData {
            point_count_declared: 2,
            horizontal_count: 1,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            values: Array3::from_shape_fn((2, 2, 2), |(energy, potential, angular)| {
                let value = energy as f64 + 0.1 * potential as f64 + 0.01 * angular as f64;
                Complex64::new(value, -value)
            }),
        }
    }

    fn sample_gtrl_dat() -> Result<GtrlDatData> {
        Ok(parse_gtrl_dat(
            r#"    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01
    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01
"#,
        )?)
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "FMS calculation of full Green's function ...".to_string(),
                "Done with module: FMS.".to_string(),
                "MKGTR: Tracing over Green's function ...".to_string(),
                "Done with module: MKGTR.".to_string(),
            ],
            line_terminators: vec![
                "\n".to_string(),
                "\n".to_string(),
                "\n".to_string(),
                "\n".to_string(),
            ],
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

    fn copy_gtr_bin_references(source_dir: &Path, target_dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(source_dir)
            .with_context(|| format!("failed to read {}", source_dir.display()))?
        {
            let entry = entry
                .with_context(|| format!("failed to read entry in {}", source_dir.display()))?;
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
            if super::is_gtr_bin_name(name) {
                std::fs::copy(entry.path(), target_dir.join(name))?;
            }
        }
        Ok(())
    }

    fn reference_fms_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
