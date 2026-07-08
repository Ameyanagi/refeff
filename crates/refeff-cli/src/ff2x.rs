use std::f64::consts::PI;
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use num_complex::Complex64;
use refeff_core::{
    FEFF_ALPHA_INV, FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, Ff2xExcitationConvolutionInput,
    FprimeCorrectionInput, Real, SpringDynamicalMatrix, SpringDynamicalMatrixInput,
    SpringEquationOfMotionInput, SpringInput, SpringRecursionInput, SpringRecursionState,
    classical_debye_waller_factor, conv as lorentz_convolve, dmdw_debye_waller_factors_from_poles,
    dmdw_lanczos_coefficients, dmdw_lanczos_pole_spectrum, dmdw_mass_weighted_dynamical_matrix,
    dmdw_path_motion, dmdw_project_seed_vector, dmdw_rigid_body_projection_modes,
    equation_of_motion_debye_waller_factor, ff2x_excitation_convolve, fprime_correction,
    morse_einstein_cumulants, parse_spring_input, quantum_debye_waller_factor,
    recursion_debye_waller_factor, remove_phase_jump, spring_dynamical_matrix, terp, terp1, terpc,
    thermal_expansion_cumulants, update_spring_recursion_state, wave_number_from_hartree,
};
use refeff_io::{
    ChiDatData, ChiaBinData, CumDatData, CumDatEntry, DanesDatData, DmdwCalculation, DmdwInput,
    EelsInput, FMS_BIN_DEFAULT_PAD_WIDTH, FeffBinData, FeffBinPath, FefflBinData, Ff2xInput,
    FmsBinData, FmslBinData, GeomDat, GlobalInput, ListDatData, ModuleLogData,
    SfconvSo2convFeffPathData, XmuDatData, XmulDatData, XmulDatFromNrixsDecompositionInput,
    XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData, XseclBinData, XsectFf2xHandoff,
    chi_dat_string, danes_dat_string, read_chi_dat, read_chia_bin, read_contour_dat, read_cum_dat,
    read_curve_dat, read_danes_dat, read_dym, read_feff_bin, read_feffl_bin, read_fms_bin,
    read_fmsl_bin, read_list_dat, read_module_log_dat, read_phase_bin, read_prexmu_dat,
    read_residue_dat, read_xmu_dat, read_xmul_dat, read_xscorr_raw_dat, read_xsecl_bin,
    read_xsect_dat, write_chi_dat, write_chia_bin, write_contour_dat, write_cum_dat,
    write_curve_dat, write_danes_dat, write_module_log_dat, write_prexmu_dat, write_residue_dat,
    write_sfconv_so2conv_feff_path_data, write_xmu_dat, write_xmul_dat, write_xscorr_raw_dat,
    xmu_dat_string, xmul_dat_from_nrixs_decomposition, xmul_dat_string, xsect_dat_ff2x_handoff,
};

use crate::{fms, work_dir_for_input};

const FF2X_EPS4: Real = 1.0e-4;
const FF2X_EPS: Real = 1.0e-16;
const FF2X_FEFF_VERSION: &str = "FEFF 10.0.0";
const FF2X_DMDW_MATCH_TOLERANCE_BOHR: Real = 0.01;

/// Run the supported FEFF FF2X cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FF2X run can be satisfied from caches or source handoffs.
pub(crate) fn has_cached_ff2x_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("ff2x.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !ff2x_enabled(&input) {
        return Ok(false);
    }
    let outputs = cached_output_paths(work_dir)?;
    if !outputs.is_empty() && validate_cached_outputs_readable(&outputs).is_ok() {
        if validate_declared_ff2x_source_handoffs(work_dir, &input).is_err() {
            return Ok(false);
        }
        return Ok(true);
    }
    match has_ff2x_generation_handoffs(work_dir, &input) {
        Ok(supported) => Ok(supported),
        Err(_) => Ok(false),
    }
}

/// Run FEFF FF2X from source handoffs or existing final-spectrum files.
///
/// When `xsect.dat` plus the matching `feff.bin`/`list.dat` handoffs are
/// present, this generates EXAFS, regular XANES, DANES, or FPRIME spectra from
/// the Rust FF2X assembler. Existing `xmu.dat`, `chi.dat`/`chipNNNN.dat`,
/// `xmul.dat`, and `danes.dat` caches are still validated and re-rendered, plus
/// optional XSCORR diagnostic sidecars and `log6.dat` module diagnostics.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !ff2x_enabled(&input) {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir)?;
    if outputs.is_empty() {
        if let Some(generated_count) = evaluate_optional_path_damping_handoffs(work_dir, &input)? {
            return Ok(generated_count);
        }
        bail!(
            "FF2X spectrum generation requires cached final-spectrum output or xsect.dat/feff.bin/list.dat source handoffs"
        );
    }
    if let Err(cache_error) = validate_cached_outputs_readable(&outputs) {
        if has_ff2x_generation_handoffs(work_dir, &input).unwrap_or(false)
            && let Some(generated_count) =
                evaluate_optional_path_damping_handoffs(work_dir, &input)?
        {
            return Ok(generated_count);
        }
        return Err(cache_error);
    }
    if cached_exafs_outputs_are_stale_against_source(work_dir, &input)?
        && let Some(generated_count) = evaluate_optional_path_damping_handoffs(work_dir, &input)?
    {
        return Ok(generated_count);
    }
    if cached_non_exafs_outputs_are_stale_against_source(work_dir, &input, &outputs)?
        && let Some(generated_count) = evaluate_optional_path_damping_handoffs(work_dir, &input)?
    {
        return Ok(generated_count);
    }

    for output in &outputs {
        match output.kind {
            CachedOutputKind::Xmu => {
                let data = read_xmu_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmu_cache(&output.path, &data)?;
            }
            CachedOutputKind::Chi => {
                let data = read_chi_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_chi_cache(&output.path, &data)?;
            }
            CachedOutputKind::Xmul => {
                let data = read_xmul_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_xmul_cache(&output.path, &data)?;
            }
            CachedOutputKind::Danes => {
                let data = read_danes_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_danes_cache(&output.path, &data)?;
            }
        }
    }

    let diagnostic_count = write_optional_cached_diagnostic_outputs(work_dir, &input)?;
    let sidecar_count = write_optional_xscorr_sidecars(work_dir)?;
    let log_count = write_or_generate_module_log(&work_dir.join("log6.dat"))?;
    Ok(outputs.len() + diagnostic_count + sidecar_count + log_count)
}

fn validate_cached_outputs_readable(outputs: &[CachedOutputPath]) -> Result<()> {
    for output in outputs {
        match output.kind {
            CachedOutputKind::Xmu => {
                read_xmu_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
            }
            CachedOutputKind::Chi => {
                read_chi_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
            }
            CachedOutputKind::Xmul => {
                read_xmul_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
            }
            CachedOutputKind::Danes => {
                read_danes_dat(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
            }
        }
    }
    Ok(())
}

fn validate_declared_ff2x_source_handoffs(work_dir: &Path, input: &Ff2xInput) -> Result<()> {
    let xsect_path = work_dir.join("xsect.dat");
    if xsect_path.is_file() {
        let xsect = read_xsect_dat(&xsect_path)
            .with_context(|| format!("failed to read {}", xsect_path.display()))?;
        xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv)?;
    }

    read_optional_global_input(work_dir)?;
    for polarization in ff2x_generation_polarizations(work_dir)? {
        let files = ff2x_generation_files(work_dir, polarization)?;
        if files.feff_path.is_file() {
            read_feff_bin(&files.feff_path)
                .with_context(|| format!("failed to read {}", files.feff_path.display()))?;
        }
        if files.list_path.is_file() {
            read_list_dat(&files.list_path)
                .with_context(|| format!("failed to read {}", files.list_path.display()))?;
        }
    }

    Ok(())
}

#[derive(Debug)]
struct Ff2xExafsSourceOutputs {
    files: Ff2xGenerationFiles,
    chi: ChiDatData,
    xmu: XmuDatData,
}

fn cached_exafs_outputs_are_stale_against_source(
    work_dir: &Path,
    input: &Ff2xInput,
) -> Result<bool> {
    let Some(generated) = build_exafs_source_outputs_if_applicable(work_dir, input)? else {
        return Ok(false);
    };

    let cached_chi = match read_chi_dat(&generated.files.chi_path) {
        Ok(data) => data,
        Err(_) => return Ok(true),
    };
    if chi_dat_string(&cached_chi)? != chi_dat_string(&generated.chi)? {
        return Ok(true);
    }

    let cached_xmu = match read_xmu_dat(&generated.files.xmu_path) {
        Ok(data) => data,
        Err(_) => return Ok(true),
    };
    Ok(xmu_dat_string(&cached_xmu)? != xmu_dat_string(&generated.xmu)?)
}

fn build_exafs_source_outputs_if_applicable(
    work_dir: &Path,
    input: &Ff2xInput,
) -> Result<Option<Ff2xExafsSourceOutputs>> {
    if fms::blocks_downstream_source_generation(work_dir)? {
        return Ok(None);
    }
    if input.control.ispec != 0
        || ff2x_xmu_effective_ispec(input.control.ispec).is_some()
        || input.control.ispec == 3
        || input.control.ispec == 4
    {
        return Ok(None);
    }

    let xsect_path = work_dir.join("xsect.dat");
    if !xsect_path.is_file() {
        return Ok(None);
    }
    let xsect_dat = read_xsect_dat(&xsect_path)
        .with_context(|| format!("failed to read {}", xsect_path.display()))?;
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    let global = read_optional_global_input(work_dir)?;
    ff2x_validate_generation_supported(input, global.as_ref())?;
    if ff2x_nrixs_decomposition_channel(input, global.as_ref())?.is_some()
        || ff2x_nrixs_non_decomposed(global.as_ref())
        || ff2x_configuration_average_nabs(global.as_ref())?.is_some()
    {
        return Ok(None);
    }

    let polarizations = ff2x_generation_polarizations(work_dir)?;
    if polarizations.len() != 1 || polarizations[0] != Ff2xPolarizationSpec::base() {
        return Ok(None);
    }
    let files = ff2x_generation_files(work_dir, polarizations[0])?;
    if !files.feff_path.is_file() || !files.list_path.is_file() {
        return Ok(None);
    }

    let feff = read_feff_bin(&files.feff_path)
        .with_context(|| format!("failed to read {}", files.feff_path.display()))?;
    let list = read_list_dat(&files.list_path)
        .with_context(|| format!("failed to read {}", files.list_path.display()))?;
    let prepared = ff2x_prepared_paths_with_imaginary_correction(
        Some(work_dir),
        input,
        &feff,
        &list,
        ff2x_effective_imaginary_correction_hartree(input, &xsect),
    )?;
    let path_summary_header_lines = ff2x_path_summary_header_lines(&prepared);
    let pre_table_header_lines =
        ff2x_pre_table_header_lines(input, &xsect, &list, &path_summary_header_lines);
    let momentum_grid = ff2x_generation_momentum_grid(input, &feff, &xsect)?;
    let path_sum = ff2x_generation_path_sum(input, &feff, &prepared, &xsect, &momentum_grid)?;
    let mut chi = ff2x_chi_dat_from_path_sum(
        input,
        &momentum_grid,
        &path_sum,
        &pre_table_header_lines,
        prepared.len(),
        list.entries.len(),
    )?;
    let output_energy = ff2x_output_energy_grid_for_input(input, &feff, &xsect, &momentum_grid)?;
    let (corrected_background, path_chi) =
        ff2x_mbconv_components(input, &xsect, &output_energy, &path_sum)?;
    if input.control.mbconv > 0 {
        chi.chi = path_chi.clone();
    }
    if input.control.ipr6 == 4 {
        chi = ff2x_chi_dat_with_ckp_columns(chi, &feff, &xsect, &momentum_grid)?;
    }
    let xscorr = ff2x_atomic_xscorr_with_background(input, &xsect, corrected_background.view())?;
    let xmu = ff2x_xmu_dat_from_components(Ff2xXmuComponents {
        input,
        xsect: &xsect,
        momentum_grid: &momentum_grid,
        output_energy: &output_energy,
        path_sum: &path_sum,
        path_chi: path_chi.view(),
        corrected_background: corrected_background.view(),
        corrected_atomic_cross_section: xscorr.corrected_atomic_cross_section.view(),
        pre_table_header_lines: &pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    Ok(Some(Ff2xExafsSourceOutputs { files, chi, xmu }))
}

fn cached_non_exafs_outputs_are_stale_against_source(
    work_dir: &Path,
    input: &Ff2xInput,
    outputs: &[CachedOutputPath],
) -> Result<bool> {
    if outputs.is_empty()
        || source_generation_is_base_exafs_without_extra_state(work_dir, input)?
        || !has_ff2x_generation_handoffs(work_dir, input)?
    {
        return Ok(false);
    }

    let scratch = Ff2xScratchWorkDir::copy_source_files_from(work_dir)?;
    let Some(_) = evaluate_optional_path_damping_handoffs(scratch.path(), input)? else {
        return Ok(false);
    };

    for output in outputs {
        let Some(file_name) = output.path.file_name() else {
            continue;
        };
        let generated_path = scratch.path().join(file_name);
        if !generated_path.is_file() {
            continue;
        }
        if cached_output_differs_from_generated(&output.path, &generated_path, output.kind)? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn source_generation_is_base_exafs_without_extra_state(
    work_dir: &Path,
    input: &Ff2xInput,
) -> Result<bool> {
    if input.control.ispec != 0 || ff2x_xmu_effective_ispec(input.control.ispec).is_some() {
        return Ok(false);
    }
    let global = read_optional_global_input(work_dir)?;
    if ff2x_nrixs_decomposition_channel(input, global.as_ref())?.is_some()
        || ff2x_nrixs_non_decomposed(global.as_ref())
        || ff2x_configuration_average_nabs(global.as_ref())?.is_some()
    {
        return Ok(false);
    }
    let polarizations = ff2x_generation_polarizations(work_dir)?;
    Ok(polarizations.len() == 1 && polarizations[0] == Ff2xPolarizationSpec::base())
}

fn cached_output_differs_from_generated(
    cached_path: &Path,
    generated_path: &Path,
    kind: CachedOutputKind,
) -> Result<bool> {
    match kind {
        CachedOutputKind::Xmu => Ok(xmu_dat_string(
            &read_xmu_dat(cached_path)
                .with_context(|| format!("failed to read {}", cached_path.display()))?,
        )? != xmu_dat_string(
            &read_xmu_dat(generated_path)
                .with_context(|| format!("failed to read {}", generated_path.display()))?,
        )?),
        CachedOutputKind::Chi => Ok(chi_dat_string(
            &read_chi_dat(cached_path)
                .with_context(|| format!("failed to read {}", cached_path.display()))?,
        )? != chi_dat_string(
            &read_chi_dat(generated_path)
                .with_context(|| format!("failed to read {}", generated_path.display()))?,
        )?),
        CachedOutputKind::Xmul => Ok(xmul_dat_string(
            &read_xmul_dat(cached_path)
                .with_context(|| format!("failed to read {}", cached_path.display()))?,
        )? != xmul_dat_string(
            &read_xmul_dat(generated_path)
                .with_context(|| format!("failed to read {}", generated_path.display()))?,
        )?),
        CachedOutputKind::Danes => Ok(danes_dat_string(
            &read_danes_dat(cached_path)
                .with_context(|| format!("failed to read {}", cached_path.display()))?,
        )? != danes_dat_string(
            &read_danes_dat(generated_path)
                .with_context(|| format!("failed to read {}", generated_path.display()))?,
        )?),
    }
}

struct Ff2xScratchWorkDir {
    path: PathBuf,
}

impl Ff2xScratchWorkDir {
    fn copy_source_files_from(work_dir: &Path) -> Result<Self> {
        for attempt in 0..100_u32 {
            let path = std::env::temp_dir().join(format!(
                "refeff-ff2x-source-{}-{}-{attempt}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .context("system clock is before UNIX_EPOCH")?
                    .as_nanos()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    copy_ff2x_source_files(work_dir, &path)?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()));
                }
            }
        }
        bail!("failed to create unique FF2X source scratch directory");
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Ff2xScratchWorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn copy_ff2x_source_files(work_dir: &Path, scratch_dir: &Path) -> Result<()> {
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
        if is_ff2x_final_spectrum_name(name) {
            continue;
        }
        std::fs::copy(entry.path(), scratch_dir.join(name)).with_context(|| {
            format!(
                "failed to copy {} into {}",
                entry.path().display(),
                scratch_dir.display()
            )
        })?;
    }
    Ok(())
}

fn ff2x_enabled(input: &Ff2xInput) -> bool {
    input.control.mchi == 1
}

fn read_input(work_dir: &Path) -> Result<Ff2xInput> {
    let input_path = work_dir.join("ff2x.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    Ff2xInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn read_optional_global_input(work_dir: &Path) -> Result<Option<GlobalInput>> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = GlobalInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(input))
}

fn read_optional_eels_input(work_dir: &Path) -> Result<Option<EelsInput>> {
    let path = work_dir.join("eels.inp");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = EelsInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(input))
}

fn read_geom_dat(work_dir: &Path) -> Result<GeomDat> {
    let geom_path = work_dir.join("geom.dat");
    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))
}

fn has_ff2x_generation_handoffs(work_dir: &Path, input: &Ff2xInput) -> Result<bool> {
    if fms::blocks_downstream_source_generation(work_dir)? {
        return Ok(false);
    }
    if matches!(input.control.idwopt, 1 | 2)
        && (!work_dir.join("spring.inp").is_file() || !work_dir.join("geom.dat").is_file())
    {
        return Ok(false);
    }

    let xsect_path = work_dir.join("xsect.dat");
    if !xsect_path.is_file() {
        return Ok(false);
    }
    let xsect = read_xsect_dat(&xsect_path)
        .with_context(|| format!("failed to read {}", xsect_path.display()))?;
    if xsect.main_energy_count >= xsect.energy_grid_ev.len() {
        return Ok(false);
    }
    if xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv).is_err() {
        return Ok(false);
    }
    let global = read_optional_global_input(work_dir)?;
    if ff2x_validate_generation_supported(input, global.as_ref()).is_err() {
        return Ok(false);
    }
    let nrixs_decomposition_channel = ff2x_nrixs_decomposition_channel(input, global.as_ref())
        .ok()
        .flatten();
    for polarization in ff2x_generation_polarizations(work_dir)? {
        let files = ff2x_generation_files(work_dir, polarization)?;
        if !files.feff_path.is_file() || !files.list_path.is_file() {
            return Ok(false);
        }
        if nrixs_decomposition_channel.is_some()
            && (!work_dir.join("feffl.bin").is_file() || !work_dir.join("xsecl.bin").is_file())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn write_xmu_cache(path: &Path, data: &XmuDatData) -> Result<()> {
    write_xmu_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_chi_cache(path: &Path, data: &ChiDatData) -> Result<()> {
    write_chi_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xmul_cache(path: &Path, data: &XmulDatData) -> Result<()> {
    write_xmul_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_danes_cache(path: &Path, data: &DanesDatData) -> Result<()> {
    write_danes_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_xscorr_sidecars(work_dir: &Path) -> Result<usize> {
    Ok(write_optional_prexmu_cache(&work_dir.join("prexmu.dat"))?
        + write_optional_residue_cache(&work_dir.join("residue.dat"))?
        + write_optional_contour_cache(&work_dir.join("contour.dat"))?
        + write_optional_curve_cache(&work_dir.join("curve.dat"))?
        + write_optional_raw_cache(&work_dir.join("raw.dat"))?
        + write_optional_cum_cache(&work_dir.join("cum.dat"))?)
}

fn write_optional_prexmu_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_prexmu_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_prexmu_cache(path, &data)?;
    Ok(1)
}

fn write_optional_residue_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_residue_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_residue_cache(path, &data)?;
    Ok(1)
}

fn write_optional_contour_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_contour_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_contour_cache(path, &data)?;
    Ok(1)
}

fn write_optional_curve_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_curve_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_curve_cache(path, &data)?;
    Ok(1)
}

fn write_optional_raw_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_xscorr_raw_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_raw_cache(path, &data)?;
    Ok(1)
}

fn write_optional_cum_cache(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_cum_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_cum_cache(path, &data)?;
    Ok(1)
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)?;
    Ok(1)
}

fn write_or_generate_module_log(path: &Path) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_module_log(path, &generated_ff2x_module_log())?;
    Ok(1)
}

fn generated_ff2x_module_log() -> ModuleLogData {
    ff2x_module_log_from_lines(vec![
        "Calculating XAS spectra ...".to_string(),
        "Done with module: XAS spectra (FF2X: DW + final sum over paths).".to_string(),
    ])
}

fn write_or_generate_generation_module_log(
    path: &Path,
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    used_path_count: usize,
) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_module_log(
        path,
        &generated_ff2x_generation_module_log(input, xsect, used_path_count),
    )?;
    Ok(1)
}

fn generated_ff2x_generation_module_log(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    used_path_count: usize,
) -> ModuleLogData {
    let mut lines = vec!["Calculating XAS spectra ...".to_string()];
    if input.debye.alphat > 0.0 {
        lines.push(format!(
            "    1st and 3rd cumulants, alphat = {:20.4E}",
            input.debye.alphat
        ));
    }
    if ff2x_has_effective_energy_correction(input, xsect) {
        lines.push(format!(
            "    Energy zero shift, vr, vi {:14.5E}{:14.5E}",
            input.corrections.vrcorr,
            ff2x_effective_imaginary_correction_ev(input, xsect)
        ));
    }
    lines.push(format!(
        "    Use all paths with cw amplitude ratio{:7.2}%",
        input.corrections.critcw
    ));
    if ff2x_uses_debye_waller_correction(input) {
        lines.push(format!(
            "    S02{:7.3}  Temp{:8.2}  Debye temp{:8.2}  Global sig2{:9.5}",
            xsect.amplitude_reduction, input.debye.tk, input.debye.thetad, input.debye.sig2g
        ));
    } else {
        lines.push(format!(
            "    S02{:7.3}  Global sig2{:9.5}",
            xsect.amplitude_reduction, input.debye.sig2g
        ));
    }
    if used_path_count > 0
        && ff2x_uses_debye_waller_correction(input)
        && let Some(line) = ff2x_debye_log_line(input.control.idwopt)
    {
        lines.push(line.to_string());
    }
    lines.push("Done with module: XAS spectra (FF2X: DW + final sum over paths).".to_string());
    ff2x_module_log_from_lines(lines)
}

fn ff2x_module_log_from_lines(lines: Vec<String>) -> ModuleLogData {
    let line_terminators = vec!["\n".to_string(); lines.len()];
    ModuleLogData {
        lines,
        line_terminators,
    }
}

fn write_prexmu_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_prexmu_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_residue_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_residue_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_contour_cache(path: &Path, data: &XscorrComplexTable) -> Result<()> {
    write_contour_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_curve_cache(path: &Path, data: &XscorrCurveDatData) -> Result<()> {
    write_curve_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_raw_cache(path: &Path, data: &XscorrRawDatData) -> Result<()> {
    write_xscorr_raw_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_cum_cache(path: &Path, data: &CumDatData) -> Result<()> {
    write_cum_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_cached_diagnostic_outputs(work_dir: &Path, input: &Ff2xInput) -> Result<usize> {
    if input.control.ipr6 < 2 && input.debye.alphat <= 0.0 {
        return Ok(0);
    }

    let xsect_path = work_dir.join("xsect.dat");
    if !xsect_path.is_file() {
        return Ok(0);
    }
    let xsect_dat = read_xsect_dat(&xsect_path)
        .with_context(|| format!("failed to read {}", xsect_path.display()))?;
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    let mut written = 0_usize;
    for polarization in ff2x_generation_polarizations(work_dir)? {
        let files = ff2x_generation_files(work_dir, polarization)?;
        if !files.feff_path.is_file() || !files.list_path.is_file() {
            continue;
        }
        written += write_cached_diagnostic_outputs_for_files(work_dir, input, &xsect, &files)?;
    }
    Ok(written)
}

fn write_cached_diagnostic_outputs_for_files(
    work_dir: &Path,
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    files: &Ff2xGenerationFiles,
) -> Result<usize> {
    let feff = read_feff_bin(&files.feff_path)
        .with_context(|| format!("failed to read {}", files.feff_path.display()))?;
    let list = read_list_dat(&files.list_path)
        .with_context(|| format!("failed to read {}", files.list_path.display()))?;
    let prepared = ff2x_prepared_paths_with_imaginary_correction(
        Some(work_dir),
        input,
        &feff,
        &list,
        ff2x_effective_imaginary_correction_hartree(input, xsect),
    )?;

    let mut written = 0_usize;
    if input.debye.alphat > 0.0 {
        let damping = prepared.iter().map(|path| path.damping).collect::<Vec<_>>();
        write_ff2x_cum_dat(&work_dir.join("cum.dat"), input, &damping)?;
    }
    if input.control.ipr6 >= 2 && input.control.ispec == 0 {
        let momentum_grid = ff2x_generation_momentum_grid(input, &feff, xsect)?;
        let path_sum = ff2x_generation_path_sum(input, &feff, &prepared, xsect, &momentum_grid)?;
        written += write_ff2x_chip_outputs(Ff2xChipOutputInputs {
            work_dir,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
        })?;
    }
    if input.control.ipr6 >= 3 {
        written += write_ff2x_feff_path_outputs(work_dir, &feff, &list, xsect)?;
    }
    Ok(written)
}

fn evaluate_optional_path_damping_handoffs(
    work_dir: &Path,
    input: &Ff2xInput,
) -> Result<Option<usize>> {
    if fms::blocks_downstream_source_generation(work_dir)? {
        return Ok(None);
    }

    let xsect_path = work_dir.join("xsect.dat");
    let xsect = if xsect_path.is_file() {
        let xsect_dat = read_xsect_dat(&xsect_path)
            .with_context(|| format!("failed to read {}", xsect_path.display()))?;
        Some(xsect_dat_ff2x_handoff(
            &xsect_dat,
            input.corrections.s02,
            input.control.mbconv,
        )?)
    } else {
        None
    };

    if let Some(xsect) = xsect {
        let global = read_optional_global_input(work_dir)?;
        ff2x_validate_generation_supported(input, global.as_ref())?;
        let mut generated_count = 0_usize;
        let mut wrote_common_outputs = false;
        for polarization in ff2x_generation_polarizations(work_dir)? {
            let files = ff2x_generation_files(work_dir, polarization)?;
            if !files.feff_path.is_file() || !files.list_path.is_file() {
                if !wrote_common_outputs && generated_count == 0 {
                    return Ok(None);
                }
                bail!(
                    "FF2X polarization {} generation requires {} and {}",
                    polarization.index,
                    files.feff_path.display(),
                    files.list_path.display()
                );
            }
            generated_count += evaluate_generation_for_polarization(
                work_dir,
                input,
                &xsect,
                &files,
                global.as_ref(),
                !wrote_common_outputs,
            )?;
            wrote_common_outputs = true;
        }
        return Ok(Some(generated_count));
    }

    let base_files = ff2x_generation_files(work_dir, Ff2xPolarizationSpec::base())?;
    if !base_files.feff_path.is_file() || !base_files.list_path.is_file() {
        return Ok(None);
    }
    let feff = read_feff_bin(&base_files.feff_path)
        .with_context(|| format!("failed to read {}", base_files.feff_path.display()))?;
    let list = read_list_dat(&base_files.list_path)
        .with_context(|| format!("failed to read {}", base_files.list_path.display()))?;
    let prepared = ff2x_prepared_paths_with_imaginary_correction(
        Some(work_dir),
        input,
        &feff,
        &list,
        ff2x_imaginary_correction_hartree(input),
    )?;
    if input.debye.alphat > 0.0 {
        let damping = prepared.iter().map(|path| path.damping).collect::<Vec<_>>();
        write_ff2x_cum_dat(&work_dir.join("cum.dat"), input, &damping)?;
    }

    Ok(None)
}

fn evaluate_generation_for_polarization(
    work_dir: &Path,
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    files: &Ff2xGenerationFiles,
    global: Option<&GlobalInput>,
    write_common_outputs: bool,
) -> Result<usize> {
    let feff = read_feff_bin(&files.feff_path)
        .with_context(|| format!("failed to read {}", files.feff_path.display()))?;
    let list = read_list_dat(&files.list_path)
        .with_context(|| format!("failed to read {}", files.list_path.display()))?;
    let prepared = ff2x_prepared_paths_with_imaginary_correction(
        Some(work_dir),
        input,
        &feff,
        &list,
        ff2x_effective_imaginary_correction_hartree(input, xsect),
    )?;
    let path_summary_header_lines = ff2x_path_summary_header_lines(&prepared);
    let mut generated_count = 0_usize;
    if input.debye.alphat > 0.0 {
        let damping = prepared.iter().map(|path| path.damping).collect::<Vec<_>>();
        write_ff2x_cum_dat(&work_dir.join("cum.dat"), input, &damping)?;
        if write_common_outputs {
            generated_count += 1;
        }
    }
    let pre_table_header_lines =
        ff2x_pre_table_header_lines(input, xsect, &list, &path_summary_header_lines);
    let momentum_grid = ff2x_generation_momentum_grid(input, &feff, xsect)?;
    let nrixs_non_decomposed = ff2x_nrixs_non_decomposed(global);
    if let Some(max_decomposition_channel) = ff2x_nrixs_decomposition_channel(input, global)? {
        generated_count += write_ff2x_nrixs_xmul_outputs(Ff2xNrixsOutputInputs {
            work_dir,
            write_module_log: write_common_outputs,
            input,
            xsect,
            list: &list,
            feff: &feff,
            prepared: &prepared,
            max_decomposition_channel,
            pre_table_header_lines: &pre_table_header_lines,
        })?;
        return Ok(generated_count);
    }
    let mut path_sum = ff2x_generation_path_sum(input, &feff, &prepared, xsect, &momentum_grid)?;
    let mut configuration_average_trace = None;
    if let Some(nabs) = ff2x_configuration_average_nabs(global)? {
        let trace = ff2x_configuration_average_current_trace(
            work_dir,
            input,
            xsect,
            &feff,
            files.polarization.fms_spectrum_index,
            &momentum_grid,
            &path_sum,
        )?;
        match ff2x_configuration_average_action(work_dir, trace.view(), nabs)? {
            Ff2xConfigurationAverageAction::IntermediateWritten { count } => {
                generated_count += count;
                if write_common_outputs {
                    generated_count += write_or_generate_generation_module_log(
                        &work_dir.join("log6.dat"),
                        input,
                        xsect,
                        prepared.len(),
                    )?;
                }
                return Ok(generated_count);
            }
            Ff2xConfigurationAverageAction::FinalTrace(total) => {
                if ff2x_xmu_effective_ispec(input.control.ispec).is_none()
                    && input.control.ispec != 3
                    && input.control.ispec != 4
                {
                    path_sum = Ff2xPathSum {
                        total,
                        paths: path_sum.paths,
                    };
                } else {
                    configuration_average_trace = Some(total);
                }
            }
        }
    }
    if nrixs_non_decomposed && ff2x_xmu_effective_ispec(input.control.ispec).is_some() {
        generated_count += write_ff2x_nrixs_xmu_outputs(Ff2xXanesOutputInputs {
            work_dir,
            xmu_path: &files.xmu_path,
            fms_spectrum_index: files.polarization.fms_spectrum_index,
            write_module_log: write_common_outputs,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
            configuration_average_trace: configuration_average_trace
                .as_ref()
                .map(|trace| trace.view()),
            pre_table_header_lines: &pre_table_header_lines,
        })?;
        return Ok(generated_count);
    }
    if ff2x_xmu_effective_ispec(input.control.ispec).is_some() {
        generated_count += write_ff2x_xanes_outputs(Ff2xXanesOutputInputs {
            work_dir,
            xmu_path: &files.xmu_path,
            fms_spectrum_index: files.polarization.fms_spectrum_index,
            write_module_log: write_common_outputs,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
            configuration_average_trace: configuration_average_trace
                .as_ref()
                .map(|trace| trace.view()),
            pre_table_header_lines: &pre_table_header_lines,
        })?;
        return Ok(generated_count);
    }
    if input.control.ispec == 3 {
        generated_count += write_ff2x_danes_outputs(Ff2xAnomalousOutputInputs {
            work_dir,
            xmu_path: &files.xmu_path,
            fms_spectrum_index: files.polarization.fms_spectrum_index,
            write_module_log: write_common_outputs,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
            configuration_average_trace: configuration_average_trace
                .as_ref()
                .map(|trace| trace.view()),
            pre_table_header_lines: &pre_table_header_lines,
        })?;
        return Ok(generated_count);
    }
    if input.control.ispec == 4 {
        generated_count += write_ff2x_fprime_outputs(Ff2xAnomalousOutputInputs {
            work_dir,
            xmu_path: &files.xmu_path,
            fms_spectrum_index: files.polarization.fms_spectrum_index,
            write_module_log: write_common_outputs,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
            configuration_average_trace: configuration_average_trace
                .as_ref()
                .map(|trace| trace.view()),
            pre_table_header_lines: &pre_table_header_lines,
        })?;
        return Ok(generated_count);
    }

    let mut chi = ff2x_chi_dat_from_path_sum(
        input,
        &momentum_grid,
        &path_sum,
        &pre_table_header_lines,
        prepared.len(),
        list.entries.len(),
    )?;
    let output_energy = ff2x_output_energy_grid_for_input(input, &feff, xsect, &momentum_grid)?;
    let (corrected_background, path_chi) =
        ff2x_mbconv_components(input, xsect, &output_energy, &path_sum)?;
    if input.control.mbconv > 0 {
        chi.chi = path_chi.clone();
    }
    if input.control.ipr6 == 4 {
        chi = ff2x_chi_dat_with_ckp_columns(chi, &feff, xsect, &momentum_grid)?;
    }
    let xscorr = ff2x_atomic_xscorr_with_background(input, xsect, corrected_background.view())?;
    let xmu = ff2x_xmu_dat_from_components(Ff2xXmuComponents {
        input,
        xsect,
        momentum_grid: &momentum_grid,
        output_energy: &output_energy,
        path_sum: &path_sum,
        path_chi: path_chi.view(),
        corrected_background: corrected_background.view(),
        corrected_atomic_cross_section: xscorr.corrected_atomic_cross_section.view(),
        pre_table_header_lines: &pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    write_chi_cache(&files.chi_path, &chi)?;
    write_xmu_cache(&files.xmu_path, &xmu)?;
    generated_count += 2;
    if input.control.ipr6 >= 2 {
        generated_count += write_ff2x_chip_outputs(Ff2xChipOutputInputs {
            work_dir,
            input,
            xsect,
            list: &list,
            feff: &feff,
            momentum_grid: &momentum_grid,
            prepared: &prepared,
            path_sum: &path_sum,
        })?;
    }
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, &feff, &list, xsect)?;
    }
    if write_common_outputs {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    Ok(generated_count)
}

enum Ff2xConfigurationAverageAction {
    IntermediateWritten { count: usize },
    FinalTrace(Array1<Complex64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ff2xConfigurationAverageState {
    absorber_count: usize,
    accumulated_count: usize,
    trace_len: usize,
}

const FF2X_CFAVERAGE_STATE_FILE: &str = ".refeff-ff2x-cfaverage-state";

fn ff2x_configuration_average_nabs(global: Option<&GlobalInput>) -> Result<Option<usize>> {
    let Some(global) = global else {
        return Ok(None);
    };
    if global.cfaverage.nabs <= 1 {
        return Ok(None);
    }
    let nabs = usize::try_from(global.cfaverage.nabs)
        .context("failed to convert FF2X CFAVERAGE absorber count")?;
    Ok(Some(nabs))
}

fn ff2x_configuration_average_current_trace(
    work_dir: &Path,
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    feff: &FeffBinData,
    fms_spectrum_index: usize,
    momentum_grid: &Ff2xMomentumGrid,
    path_sum: &Ff2xPathSum,
) -> Result<Array1<Complex64>> {
    if ff2x_xmu_effective_ispec(input.control.ispec).is_some() {
        ff2x_validate_xanes_fms_grid(feff, xsect, momentum_grid)?;
        let fms_trace = ff2x_xanes_optional_fms_trace(work_dir, xsect, fms_spectrum_index)?;
        return ff2x_xanes_combined_trace(xsect, fms_trace.view(), path_sum);
    }

    if input.control.ispec == 3 {
        let (_, fms_trace) =
            ff2x_danes_extension_and_fms_trace(work_dir, xsect, fms_spectrum_index)?;
        return ff2x_path_sum_with_fms_trace("DANES", xsect, path_sum, fms_trace.view());
    }

    if input.control.ispec == 4 {
        let fms_trace = ff2x_fprime_fms_trace(work_dir, xsect, fms_spectrum_index)?;
        return ff2x_path_sum_with_fms_trace("FPRIME", xsect, path_sum, fms_trace.view());
    }

    Ok(path_sum.total.clone())
}

fn ff2x_configuration_average_action(
    work_dir: &Path,
    current_trace: ArrayView1<'_, Complex64>,
    absorber_count: usize,
) -> Result<Ff2xConfigurationAverageAction> {
    let chia_path = work_dir.join("chia.bin");
    let state_path = work_dir.join(FF2X_CFAVERAGE_STATE_FILE);
    if !chia_path.is_file() {
        let trace = ff2x_scaled_configuration_average_trace(current_trace, absorber_count)?;
        let count = write_ff2x_chia_bin(&chia_path, trace.view())?;
        write_ff2x_configuration_average_state(
            &state_path,
            Ff2xConfigurationAverageState {
                absorber_count,
                accumulated_count: 1,
                trace_len: current_trace.len(),
            },
        )?;
        return Ok(Ff2xConfigurationAverageAction::IntermediateWritten { count });
    }

    let accumulated = read_chia_bin(&chia_path)
        .with_context(|| format!("failed to read {}", chia_path.display()))?;
    let state = read_optional_ff2x_configuration_average_state(&state_path)?.unwrap_or(
        Ff2xConfigurationAverageState {
            absorber_count,
            accumulated_count: absorber_count.saturating_sub(1),
            trace_len: accumulated.values.len(),
        },
    );
    validate_ff2x_configuration_average_state(&state, absorber_count, accumulated.values.len())?;
    let updated_trace =
        ff2x_next_configuration_average_trace(&accumulated, current_trace, absorber_count)?;
    let accumulated_count = state
        .accumulated_count
        .checked_add(1)
        .context("FF2X CFAVERAGE accumulated absorber count overflowed")?;
    if accumulated_count < absorber_count {
        write_ff2x_chia_bin(&chia_path, updated_trace.view())?;
        write_ff2x_configuration_average_state(
            &state_path,
            Ff2xConfigurationAverageState {
                absorber_count,
                accumulated_count,
                trace_len: updated_trace.len(),
            },
        )?;
        return Ok(Ff2xConfigurationAverageAction::IntermediateWritten { count: 1 });
    }

    std::fs::remove_file(&chia_path)
        .with_context(|| format!("failed to remove {}", chia_path.display()))?;
    remove_optional_ff2x_configuration_average_state(&state_path)?;
    Ok(Ff2xConfigurationAverageAction::FinalTrace(updated_trace))
}

fn ff2x_scaled_configuration_average_trace(
    current_trace: ArrayView1<'_, Complex64>,
    absorber_count: usize,
) -> Result<Array1<Complex64>> {
    if absorber_count == 0 {
        bail!("FF2X CFAVERAGE absorber count must be positive");
    }
    let scale = 1.0 / absorber_count as Real;
    Ok(Array1::from_iter(
        current_trace.iter().map(|&value| value * scale),
    ))
}

fn ff2x_next_configuration_average_trace(
    accumulated: &ChiaBinData,
    current_trace: ArrayView1<'_, Complex64>,
    absorber_count: usize,
) -> Result<Array1<Complex64>> {
    if accumulated.values.len() != current_trace.len() {
        bail!(
            "FF2X chia.bin contains {} row(s), but current trace has {} row(s)",
            accumulated.values.len(),
            current_trace.len()
        );
    }
    let current = ff2x_scaled_configuration_average_trace(current_trace, absorber_count)?;
    Ok(Array1::from_iter(
        accumulated
            .values
            .iter()
            .zip(current.iter())
            .map(|(&accumulated, &current)| accumulated + current),
    ))
}

fn ff2x_path_sum_with_fms_trace(
    label: &'static str,
    xsect: &XsectFf2xHandoff,
    path_sum: &Ff2xPathSum,
    fms_trace: ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>> {
    if path_sum.total.len() != xsect.energy_count() {
        bail!(
            "FF2X {label} path sum has {} rows for {} xsect.dat energies",
            path_sum.total.len(),
            xsect.energy_count()
        );
    }
    if fms_trace.len() != xsect.energy_count() {
        bail!(
            "FF2X {label} FMS trace length {} does not match xsect.dat energy count {}",
            fms_trace.len(),
            xsect.energy_count()
        );
    }
    Ok(Array1::from_iter(
        path_sum
            .total
            .iter()
            .zip(fms_trace.iter())
            .map(|(&path, &fms)| path + fms * xsect.amplitude_reduction),
    ))
}

fn ff2x_configuration_or_path_sum_with_fms_trace(
    label: &'static str,
    xsect: &XsectFf2xHandoff,
    path_sum: &Ff2xPathSum,
    fms_trace: ArrayView1<'_, Complex64>,
    configuration_average_trace: Option<ArrayView1<'_, Complex64>>,
) -> Result<Array1<Complex64>> {
    if let Some(trace) = configuration_average_trace {
        if trace.len() != xsect.energy_count() {
            bail!(
                "FF2X {label} configuration-average trace length {} does not match xsect.dat energy count {}",
                trace.len(),
                xsect.energy_count()
            );
        }
        return Ok(trace.to_owned());
    }
    ff2x_path_sum_with_fms_trace(label, xsect, path_sum, fms_trace)
}

fn write_ff2x_chia_bin(path: &Path, trace: ArrayView1<'_, Complex64>) -> Result<usize> {
    let data = ChiaBinData {
        values: trace.iter().copied().collect(),
    };
    write_chia_bin(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_ff2x_configuration_average_state(
    path: &Path,
    state: Ff2xConfigurationAverageState,
) -> Result<()> {
    let text = format!(
        "absorber_count {}\naccumulated_count {}\ntrace_len {}\n",
        state.absorber_count, state.accumulated_count, state.trace_len
    );
    std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))
}

fn read_optional_ff2x_configuration_average_state(
    path: &Path,
) -> Result<Option<Ff2xConfigurationAverageState>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(Some(parse_ff2x_configuration_average_state(&text)?))
}

fn parse_ff2x_configuration_average_state(text: &str) -> Result<Ff2xConfigurationAverageState> {
    let mut absorber_count = None;
    let mut accumulated_count = None;
    let mut trace_len = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else {
            continue;
        };
        let Some(value) = fields.next() else {
            bail!("FF2X CFAVERAGE state line {line:?} is missing a value");
        };
        if fields.next().is_some() {
            bail!("FF2X CFAVERAGE state line {line:?} has extra fields");
        }
        let parsed = value
            .parse::<usize>()
            .with_context(|| format!("failed to parse FF2X CFAVERAGE state value {value:?}"))?;
        match key {
            "absorber_count" => absorber_count = Some(parsed),
            "accumulated_count" => accumulated_count = Some(parsed),
            "trace_len" => trace_len = Some(parsed),
            _ => bail!("FF2X CFAVERAGE state contains unknown key {key:?}"),
        }
    }
    Ok(Ff2xConfigurationAverageState {
        absorber_count: absorber_count.context("FF2X CFAVERAGE state missing absorber_count")?,
        accumulated_count: accumulated_count
            .context("FF2X CFAVERAGE state missing accumulated_count")?,
        trace_len: trace_len.context("FF2X CFAVERAGE state missing trace_len")?,
    })
}

fn validate_ff2x_configuration_average_state(
    state: &Ff2xConfigurationAverageState,
    absorber_count: usize,
    trace_len: usize,
) -> Result<()> {
    if state.absorber_count != absorber_count {
        bail!(
            "FF2X CFAVERAGE state absorber_count {} does not match global.inp nabs {}",
            state.absorber_count,
            absorber_count
        );
    }
    if state.accumulated_count == 0 || state.accumulated_count >= absorber_count {
        bail!(
            "FF2X CFAVERAGE state accumulated_count {} must be in 1..{}",
            state.accumulated_count,
            absorber_count
        );
    }
    if state.trace_len != trace_len {
        bail!(
            "FF2X CFAVERAGE state trace_len {} does not match chia.bin row count {}",
            state.trace_len,
            trace_len
        );
    }
    Ok(())
}

fn remove_optional_ff2x_configuration_average_state(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

fn ff2x_validate_generation_supported(
    input: &Ff2xInput,
    global: Option<&GlobalInput>,
) -> Result<()> {
    ff2x_nrixs_decomposition_channel(input, global)?;
    Ok(())
}

fn ff2x_nrixs_decomposition_channel(
    input: &Ff2xInput,
    global: Option<&GlobalInput>,
) -> Result<Option<usize>> {
    let Some(global) = global else {
        return Ok(None);
    };
    if global.control.do_nrixs != 1 {
        return Ok(None);
    }
    if global.control.ldecmx <= 0 {
        return Ok(None);
    }
    let max_decomposition_channel = usize::try_from(global.control.ldecmx)
        .context("FF2X NRIXS/JAS ldecmx must be nonnegative")?;
    if input.decomposition_channels >= 0
        && usize::try_from(input.decomposition_channels)
            .ok()
            .is_some_and(|input_channel| input_channel != max_decomposition_channel)
    {
        bail!(
            "FF2X NRIXS/JAS ff2x.inp decomposition channel {} does not match global.inp ldecmx {}",
            input.decomposition_channels,
            max_decomposition_channel
        );
    }
    Ok(Some(max_decomposition_channel))
}

fn ff2x_nrixs_non_decomposed(global: Option<&GlobalInput>) -> bool {
    global.is_some_and(|global| global.control.do_nrixs == 1 && global.control.ldecmx <= 0)
}

fn ff2x_xmu_effective_ispec(ispec: i32) -> Option<i32> {
    if ispec.abs() > 0 && ispec < 3 {
        Some(ispec.abs())
    } else {
        None
    }
}

fn ff2x_generation_polarizations(work_dir: &Path) -> Result<Vec<Ff2xPolarizationSpec>> {
    let Some(eels) = read_optional_eels_input(work_dir)? else {
        return Ok(vec![Ff2xPolarizationSpec::base()]);
    };
    if !eels.calculate_elnes {
        return Ok(vec![Ff2xPolarizationSpec::base()]);
    }

    let min = eels.polarization.min;
    let step = eels.polarization.step;
    let max = eels.polarization.max;
    if min < 1 || max > 10 || min > max || step <= 0 {
        bail!(
            "FF2X EELS polarization range must satisfy 1 <= min <= max <= 10 and step > 0, got min={min}, step={step}, max={max}"
        );
    }
    if (max - min) % step != 0 {
        bail!("FF2X EELS polarization range {min}:{step}:{max} does not reach max");
    }

    let mut polarizations = Vec::new();
    let mut index = min;
    while index <= max {
        polarizations.push(Ff2xPolarizationSpec {
            index,
            fms_spectrum_index: usize::try_from(index - min)
                .context("FF2X EELS polarization offset overflowed")?,
        });
        index += step;
    }
    Ok(polarizations)
}

fn ff2x_generation_files(
    work_dir: &Path,
    polarization: Ff2xPolarizationSpec,
) -> Result<Ff2xGenerationFiles> {
    Ok(Ff2xGenerationFiles {
        polarization,
        feff_path: work_dir.join(ff2x_polarized_file_name(
            "feff",
            ".bin",
            polarization.index,
        )?),
        list_path: work_dir.join(ff2x_polarized_file_name(
            "list",
            ".dat",
            polarization.index,
        )?),
        xmu_path: work_dir.join(ff2x_polarized_file_name("xmu", ".dat", polarization.index)?),
        chi_path: work_dir.join(ff2x_polarized_file_name("chi", ".dat", polarization.index)?),
    })
}

fn ff2x_polarized_file_name(
    stem: &'static str,
    suffix: &'static str,
    index: i32,
) -> Result<String> {
    match index {
        1 => Ok(format!("{stem}{suffix}")),
        2..=9 => Ok(format!("{stem}0{index}{suffix}")),
        10 => Ok(format!("{stem}10{suffix}")),
        _ => bail!("FF2X polarization index must be 1..=10, got {index}"),
    }
}

type DebyeWallerFn = for<'a> fn(
    Real,
    Real,
    Real,
    ndarray::ArrayView2<'a, Real>,
    &[usize],
) -> std::result::Result<Real, refeff_core::DebyeError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ff2xPathDamping {
    pub(crate) path_index: usize,
    pub(crate) total_sigma2_angstrom2: Real,
    pub(crate) global_sigma2_angstrom2: Real,
    pub(crate) user_sigma2_angstrom2: Real,
    pub(crate) debye_sigma2_angstrom2: Real,
    pub(crate) cumulants: Option<Ff2xPathCumulants>,
    pub(crate) criterion: Real,
    pub(crate) degeneracy: Real,
    pub(crate) leg_count: usize,
    pub(crate) effective_half_path_length_angstrom: Real,
    pub(crate) effective_half_path_length_bohr: Real,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ff2xPathCumulants {
    pub(crate) first_cumulant_bohr: Real,
    pub(crate) third_cumulant_bohr3: Real,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xPreparedPath {
    pub(crate) damping: Ff2xPathDamping,
    pub(crate) amplitude: Array1<Real>,
    pub(crate) phase: Array1<Real>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xPathSignal {
    pub(crate) path_index: usize,
    pub(crate) signal: Array1<Complex64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xPathSum {
    pub(crate) total: Array1<Complex64>,
    pub(crate) paths: Vec<Ff2xPathSignal>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xDecomposedPathSignal {
    pub(crate) path_index: usize,
    /// Decomposed path signal as `(energy, lg2, lg1)`.
    pub(crate) signal: Array3<Complex64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xDecomposedPathSum {
    /// Decomposed path sum as `(energy, lg2, lg1)`.
    pub(crate) total: Array3<Complex64>,
    pub(crate) paths: Vec<Ff2xDecomposedPathSignal>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xMomentumGrid {
    pub(crate) output_momentum: Array1<Real>,
    pub(crate) interpolation_momentum: Array1<Real>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xOutputEnergyGrid {
    pub(crate) fermi_energy_hartree: Real,
    pub(crate) photon_energy_hartree: Array1<Real>,
    pub(crate) relative_energy_hartree: Array1<Real>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ff2xPolarizationSpec {
    index: i32,
    fms_spectrum_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Ff2xGenerationFiles {
    polarization: Ff2xPolarizationSpec,
    feff_path: PathBuf,
    list_path: PathBuf,
    xmu_path: PathBuf,
    chi_path: PathBuf,
}

impl Ff2xPolarizationSpec {
    fn base() -> Self {
        Self {
            index: 1,
            fms_spectrum_index: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Ff2xXmuComponents<'a> {
    pub(crate) input: &'a Ff2xInput,
    pub(crate) xsect: &'a XsectFf2xHandoff,
    pub(crate) momentum_grid: &'a Ff2xMomentumGrid,
    pub(crate) output_energy: &'a Ff2xOutputEnergyGrid,
    pub(crate) path_sum: &'a Ff2xPathSum,
    pub(crate) path_chi: ArrayView1<'a, Real>,
    pub(crate) corrected_background: ArrayView1<'a, Real>,
    pub(crate) corrected_atomic_cross_section: ArrayView1<'a, Real>,
    pub(crate) pre_table_header_lines: &'a [String],
    pub(crate) used_path_count: usize,
    pub(crate) total_path_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Ff2xNrixsXmulComponents<'a> {
    pub(crate) header_lines: &'a [String],
    pub(crate) max_decomposition_channel: usize,
    pub(crate) xsect: &'a XsectFf2xHandoff,
    pub(crate) channel_background: ArrayView2<'a, Real>,
    pub(crate) normalized_fine_structure: ArrayView3<'a, Real>,
}

#[derive(Debug, Clone, Copy)]
struct Ff2xNrixsOutputInputs<'a> {
    work_dir: &'a Path,
    write_module_log: bool,
    input: &'a Ff2xInput,
    xsect: &'a XsectFf2xHandoff,
    list: &'a ListDatData,
    feff: &'a FeffBinData,
    prepared: &'a [Ff2xPreparedPath],
    max_decomposition_channel: usize,
    pre_table_header_lines: &'a [String],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Ff2xXscorrInput<'a> {
    pub(crate) ispec: i32,
    pub(crate) energy_grid_hartree: ArrayView1<'a, Complex64>,
    pub(crate) main_energy_count: usize,
    pub(crate) fermi_index: usize,
    pub(crate) cross_section: ArrayView1<'a, Complex64>,
    pub(crate) background: ArrayView1<'a, Real>,
    pub(crate) path_chi: ArrayView1<'a, Complex64>,
    pub(crate) real_correction_hartree: Real,
    pub(crate) electronic_temperature_ev: Real,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ff2xXscorrResult {
    pub(crate) cchi: Array1<Complex64>,
    pub(crate) corrected_atomic_cross_section: Array1<Real>,
}

#[derive(Debug, Clone, Copy)]
struct Ff2xXanesOutputInputs<'a> {
    work_dir: &'a Path,
    xmu_path: &'a Path,
    fms_spectrum_index: usize,
    write_module_log: bool,
    input: &'a Ff2xInput,
    xsect: &'a XsectFf2xHandoff,
    list: &'a ListDatData,
    feff: &'a FeffBinData,
    momentum_grid: &'a Ff2xMomentumGrid,
    prepared: &'a [Ff2xPreparedPath],
    path_sum: &'a Ff2xPathSum,
    configuration_average_trace: Option<ArrayView1<'a, Complex64>>,
    pre_table_header_lines: &'a [String],
}

#[derive(Debug, Clone, PartialEq)]
struct Ff2xXanesCorrectedComponents {
    total: Array1<Real>,
    atomic: Array1<Real>,
    fine_structure: Array1<Real>,
}

#[derive(Debug, Clone, Copy)]
struct Ff2xXanesXmuInputs<'a> {
    input: &'a Ff2xInput,
    xsect: &'a XsectFf2xHandoff,
    momentum_grid: &'a Ff2xMomentumGrid,
    output_energy: &'a Ff2xOutputEnergyGrid,
    corrected: &'a Ff2xXanesCorrectedComponents,
    corrected_background: ArrayView1<'a, Real>,
    pre_table_header_lines: &'a [String],
    used_path_count: usize,
    total_path_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct Ff2xAnomalousOutputInputs<'a> {
    work_dir: &'a Path,
    xmu_path: &'a Path,
    fms_spectrum_index: usize,
    write_module_log: bool,
    input: &'a Ff2xInput,
    xsect: &'a XsectFf2xHandoff,
    list: &'a ListDatData,
    feff: &'a FeffBinData,
    momentum_grid: &'a Ff2xMomentumGrid,
    prepared: &'a [Ff2xPreparedPath],
    path_sum: &'a Ff2xPathSum,
    configuration_average_trace: Option<ArrayView1<'a, Complex64>>,
    pre_table_header_lines: &'a [String],
}

#[derive(Debug, Clone, Copy)]
struct Ff2xAnomalousXmuInputs<'a> {
    work_dir: &'a Path,
    fms_spectrum_index: usize,
    input: &'a Ff2xInput,
    feff: &'a FeffBinData,
    xsect: &'a XsectFf2xHandoff,
    momentum_grid: &'a Ff2xMomentumGrid,
    path_sum: &'a Ff2xPathSum,
    configuration_average_trace: Option<ArrayView1<'a, Complex64>>,
    pre_table_header_lines: &'a [String],
    used_path_count: usize,
    total_path_count: usize,
}

fn ff2x_chi_dat_from_path_sum(
    input: &Ff2xInput,
    momentum_grid: &Ff2xMomentumGrid,
    path_sum: &Ff2xPathSum,
    pre_table_header_lines: &[String],
    used_path_count: usize,
    total_path_count: usize,
) -> Result<ChiDatData> {
    if momentum_grid.output_momentum.len() != path_sum.total.len() {
        bail!(
            "FF2X chi.dat row build got {} output momentum points for {} path-sum points",
            momentum_grid.output_momentum.len(),
            path_sum.total.len()
        );
    }

    let mut wave_number = Array1::<Real>::zeros(path_sum.total.len());
    let mut chi = Array1::<Real>::zeros(path_sum.total.len());
    let mut magnitude = Array1::<Real>::zeros(path_sum.total.len());
    let mut phase = Array1::<Real>::zeros(path_sum.total.len());
    for (row, &value) in path_sum.total.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            bail!("FF2X chi.dat row {row} has non-finite path sum {value:?}");
        }
        wave_number[row] = momentum_grid.output_momentum[row] / FEFF_BOHR_ANGSTROM;
        chi[row] = if input.control.ispec.abs() == 3 {
            value.re
        } else {
            value.im
        };
        magnitude[row] = value.norm();
        phase[row] = if value.norm() > 0.0 {
            value.im.atan2(value.re)
        } else {
            0.0
        };
        if row > 0 {
            phase[row] = remove_phase_jump(phase[row], phase[row - 1])
                .context("FF2X chi.dat phase jump removal")?;
        }
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        "# -----------------------------------------------------------------------".to_string(),
        "#       k          chi          mag           phase @#".to_string(),
    ]);

    Ok(ChiDatData {
        header_lines,
        wave_number,
        chi,
        magnitude,
        phase,
        phase_minus_2kr: None,
        ckp_real: None,
        ckp_imag: None,
    })
}

fn ff2x_chi_dat_with_ckp_columns(
    mut chi: ChiDatData,
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<ChiDatData> {
    if chi.phase_minus_2kr.is_some() {
        bail!("FF2X chi.dat ckp columns cannot be combined with per-path phase columns");
    }
    if chi.point_count() != momentum_grid.output_momentum.len() {
        bail!(
            "FF2X chi.dat ckp column build got {} chi rows for {} output momenta",
            chi.point_count(),
            momentum_grid.output_momentum.len()
        );
    }

    let (ckp_real, ckp_imag) = ff2x_chi_ckp_columns(feff, xsect, momentum_grid)?;
    chi.ckp_real = Some(ckp_real);
    chi.ckp_imag = Some(ckp_imag);
    Ok(chi)
}

fn ff2x_chi_ckp_columns(
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<(Array1<Real>, Array1<Real>)> {
    let output_count = momentum_grid.output_momentum.len();
    if momentum_grid.interpolation_momentum.len() != output_count {
        bail!(
            "FF2X chi.dat ckp output got {} interpolation momenta for {} output momenta",
            momentum_grid.interpolation_momentum.len(),
            output_count
        );
    }

    let main_energy_count = xsect.main_energy_count;
    if main_energy_count < 4 {
        bail!(
            "FF2X chi.dat ckp output requires at least four main xsect.dat points for cubic interpolation, got {main_energy_count}"
        );
    }
    if feff.complex_momentum.len() < main_energy_count {
        bail!(
            "FF2X chi.dat ckp output needs {main_energy_count} FEFF complex momentum points, got {}",
            feff.complex_momentum.len()
        );
    }
    if xsect.wave_number.len() < main_energy_count {
        bail!(
            "FF2X chi.dat ckp output needs {main_energy_count} xsect.dat wave-number points, got {}",
            xsect.wave_number.len()
        );
    }

    let source_momentum = xsect
        .wave_number
        .as_slice()
        .context("FF2X xsect.dat wave-number grid is not contiguous")?;
    let source_momentum = &source_momentum[..main_energy_count];
    let source_complex_momentum = feff
        .complex_momentum
        .iter()
        .take(main_energy_count)
        .copied()
        .collect::<Vec<_>>();

    let mut ckp_real = Array1::<Real>::zeros(output_count);
    let mut ckp_imag = Array1::<Real>::zeros(output_count);
    for (row, &momentum) in momentum_grid.interpolation_momentum.iter().enumerate() {
        if !momentum.is_finite() {
            bail!("FF2X chi.dat ckp interpolation momentum {row} is not finite: {momentum}");
        }
        let ckp = terpc(source_momentum, &source_complex_momentum, 3, momentum)
            .with_context(|| format!("FF2X chi.dat ckp interpolation at row {row}"))?
            .value;
        if !(ckp.re.is_finite() && ckp.im.is_finite()) {
            bail!("FF2X chi.dat ckp interpolation row {row} returned non-finite {ckp:?}");
        }
        ckp_real[row] = ckp.re / FEFF_BOHR_ANGSTROM;
        ckp_imag[row] = ckp.im / FEFF_BOHR_ANGSTROM;
    }

    Ok((ckp_real, ckp_imag))
}

struct Ff2xChipOutputInputs<'a> {
    work_dir: &'a Path,
    input: &'a Ff2xInput,
    xsect: &'a XsectFf2xHandoff,
    list: &'a ListDatData,
    feff: &'a FeffBinData,
    momentum_grid: &'a Ff2xMomentumGrid,
    prepared: &'a [Ff2xPreparedPath],
    path_sum: &'a Ff2xPathSum,
}

fn write_ff2x_chip_outputs(inputs: Ff2xChipOutputInputs<'_>) -> Result<usize> {
    let Ff2xChipOutputInputs {
        work_dir,
        input,
        xsect,
        list,
        feff,
        momentum_grid,
        prepared,
        path_sum,
    } = inputs;

    if prepared.len() != path_sum.paths.len() {
        bail!(
            "FF2X chip output got {} prepared paths for {} path signals",
            prepared.len(),
            path_sum.paths.len()
        );
    }

    let mut count = 0_usize;
    for (prepared_path, path_signal) in prepared.iter().zip(&path_sum.paths) {
        if prepared_path.damping.path_index != path_signal.path_index {
            bail!(
                "FF2X chip output path mismatch: prepared {} but signal {}",
                prepared_path.damping.path_index,
                path_signal.path_index
            );
        }
        let data = ff2x_chip_dat_from_path_signal(
            input,
            xsect,
            list,
            feff,
            momentum_grid,
            prepared_path,
            path_signal,
        )?;
        write_chi_cache(
            &work_dir.join(format!("chip{:04}.dat", path_signal.path_index)),
            &data,
        )?;
        count += 1;
    }
    Ok(count)
}

fn ff2x_chip_dat_from_path_signal(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    list: &ListDatData,
    feff: &FeffBinData,
    momentum_grid: &Ff2xMomentumGrid,
    prepared_path: &Ff2xPreparedPath,
    path_signal: &Ff2xPathSignal,
) -> Result<ChiDatData> {
    let output_count = momentum_grid.output_momentum.len();
    if momentum_grid.interpolation_momentum.len() != output_count {
        bail!(
            "FF2X chip output got {} interpolation momenta for {} output momenta",
            momentum_grid.interpolation_momentum.len(),
            output_count
        );
    }
    if path_signal.signal.len() != output_count {
        bail!(
            "FF2X chip output path {} has {} signal points for {} output momenta",
            path_signal.path_index,
            path_signal.signal.len(),
            output_count
        );
    }
    if feff.complex_momentum.is_empty() || feff.real_momentum.is_empty() {
        bail!("FF2X chip output requires a non-empty FEFF momentum grid");
    }

    let first_complex_momentum = feff.complex_momentum[0];
    let first_real_momentum = feff.real_momentum[0];
    if !(first_complex_momentum.re.is_finite()
        && first_complex_momentum.im.is_finite()
        && first_real_momentum.is_finite())
    {
        bail!(
            "FF2X chip output got non-finite first momentum: ck={first_complex_momentum:?}, xk={first_real_momentum}"
        );
    }
    let momentum_reference = (first_complex_momentum * first_complex_momentum).re
        - first_real_momentum * first_real_momentum.abs();
    let reff = prepared_path.damping.effective_half_path_length_bohr;
    if !(momentum_reference.is_finite() && reff.is_finite()) {
        bail!("FF2X chip output got non-finite reference momentum or reff");
    }

    let mut wave_number = Array1::<Real>::zeros(output_count);
    let mut chi = Array1::<Real>::zeros(output_count);
    let mut magnitude = Array1::<Real>::zeros(output_count);
    let mut phase = Array1::<Real>::zeros(output_count);
    let mut phase_minus_2kr = Array1::<Real>::zeros(output_count);
    for row in 0..output_count {
        let output_momentum = momentum_grid.output_momentum[row];
        let interpolation_momentum = momentum_grid.interpolation_momentum[row];
        let signal = path_signal.signal[row];
        if !(output_momentum.is_finite()
            && interpolation_momentum.is_finite()
            && signal.re.is_finite()
            && signal.im.is_finite())
        {
            bail!(
                "FF2X chip output path {} row {row} contains non-finite values",
                path_signal.path_index
            );
        }
        let ckp = Complex64::new(
            output_momentum * output_momentum.abs() + momentum_reference,
            0.0,
        )
        .sqrt();
        let attenuation = (2.0 * reff * -ckp.im).exp();
        let corrected_signal = signal * attenuation;
        let corrected_phase = if corrected_signal.norm() > 0.0 {
            corrected_signal.im.atan2(corrected_signal.re)
        } else {
            0.0
        };

        wave_number[row] = output_momentum / FEFF_BOHR_ANGSTROM;
        chi[row] = corrected_signal.im;
        magnitude[row] = corrected_signal.norm();
        phase[row] = if row > 0 {
            remove_phase_jump(corrected_phase, phase[row - 1])
                .context("FF2X chip phase jump removal")?
        } else {
            corrected_phase
        };
        phase_minus_2kr[row] = phase[row] - 2.0 * interpolation_momentum * reff;
    }

    Ok(ChiDatData {
        header_lines: ff2x_chip_header_lines(input, xsect, list, &prepared_path.damping),
        wave_number,
        chi,
        magnitude,
        phase,
        phase_minus_2kr: Some(phase_minus_2kr),
        ckp_real: None,
        ckp_imag: None,
    })
}

fn ff2x_chip_header_lines(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    list: &ListDatData,
    damping: &Ff2xPathDamping,
) -> Vec<String> {
    let mut lines = list
        .titles
        .iter()
        .map(|title| format!("# {}", title.trim_end()))
        .collect::<Vec<_>>();
    if ff2x_uses_debye_waller_correction(input) {
        lines.push(format!(
            " S02{:7.3}  Temp{:8.2}  Debye temp{:8.2}  Global sig2{:9.5}",
            xsect.amplitude_reduction, input.debye.tk, input.debye.thetad, input.debye.sig2g
        ));
    } else {
        lines.push(format!(
            " S02{:7.3}                                        Global sig2{:9.5}",
            xsect.amplitude_reduction, input.debye.sig2g
        ));
    }
    if input.debye.alphat > 0.0 {
        lines.push(format!(
            " 1st and 3rd cumulants, alphat = {:20.4E}",
            input.debye.alphat
        ));
    }
    if ff2x_has_energy_correction(input) {
        lines.push(format!(
            " Energy zero shift, vr, vi {:14.5E}{:14.5E}",
            ff2x_real_correction_hartree(input),
            ff2x_imaginary_correction_hartree(input)
        ));
    }
    lines.push(format!(
        " Debye-waller factor {:14.6E} {:14.6E}",
        damping.total_sigma2_angstrom2 / FEFF_BOHR_ANGSTROM.powi(2),
        damping
            .cumulants
            .map(|cumulants| cumulants.third_cumulant_bohr3)
            .unwrap_or(0.0)
    ));
    lines.push(
        " -----------------------------------------------------------------------".to_string(),
    );
    lines
        .push("       k         chi           mag          phase        phase-2kr  @#".to_string());
    lines
}

fn ff2x_mbconv_components(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    output_energy: &Ff2xOutputEnergyGrid,
    path_sum: &Ff2xPathSum,
) -> Result<(Array1<Real>, Array1<Real>)> {
    let mut background = xsect.normalized_background.clone();
    let mut path_chi = ff2x_path_sum_chi(input, path_sum)?;

    if input.control.mbconv <= 0 {
        return Ok((background, path_chi));
    }

    let main_energy_count = xsect.main_energy_count;
    if main_energy_count > xsect.energy_count() {
        bail!(
            "FF2X mbconv main energy count {} exceeds xsect energy count {}",
            main_energy_count,
            xsect.energy_count()
        );
    }
    if background.len() != xsect.energy_count() {
        bail!(
            "FF2X mbconv background length {} does not match xsect energy count {}",
            background.len(),
            xsect.energy_count()
        );
    }
    if path_chi.len() != output_energy.photon_energy_hartree.len() {
        bail!(
            "FF2X mbconv path chi length {} does not match output energy length {}",
            path_chi.len(),
            output_energy.photon_energy_hartree.len()
        );
    }

    let xsect_energy =
        Array1::from_iter(xsect.omega_hartree.iter().take(main_energy_count).copied());
    let xsect_background = Array1::from_iter(background.iter().take(main_energy_count).copied());
    let convolved_background = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: xsect_energy.view(),
        xmu: xsect_background.view(),
        fermi_energy: ff2x_mbconv_fermi_energy(output_energy.fermi_energy_hartree, xsect),
        amplitude_reduction: xsect.file_amplitude_reduction,
        relaxation_energy: xsect.relaxation_energy,
        plasmon_frequency: xsect.plasmon_frequency * 0.5,
    })
    .context("FF2X mbconv background excitation convolution")?;
    for (row, &value) in convolved_background.iter().enumerate() {
        background[row] = value;
    }

    path_chi = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: output_energy.photon_energy_hartree.view(),
        xmu: path_chi.view(),
        fermi_energy: ff2x_mbconv_fermi_energy(output_energy.fermi_energy_hartree, xsect),
        amplitude_reduction: xsect.file_amplitude_reduction,
        relaxation_energy: xsect.relaxation_energy,
        plasmon_frequency: xsect.plasmon_frequency * 0.5,
    })
    .context("FF2X mbconv path excitation convolution")?;

    Ok((background, path_chi))
}

fn ff2x_mbconv_fermi_energy(fermi_energy_hartree: Real, xsect: &XsectFf2xHandoff) -> Real {
    let first_omega = xsect
        .omega_hartree
        .first()
        .copied()
        .unwrap_or(fermi_energy_hartree);
    if (fermi_energy_hartree - first_omega).abs() <= FF2X_EPS {
        first_omega
    } else {
        fermi_energy_hartree
    }
}

fn write_ff2x_xanes_outputs(inputs: Ff2xXanesOutputInputs<'_>) -> Result<usize> {
    let Ff2xXanesOutputInputs {
        work_dir,
        xmu_path,
        fms_spectrum_index,
        write_module_log,
        input,
        xsect,
        list,
        feff,
        momentum_grid,
        prepared,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
    } = inputs;

    if path_sum.paths.len() != prepared.len() {
        bail!(
            "FF2X XANES path sum has {} paths for {} prepared paths",
            path_sum.paths.len(),
            prepared.len()
        );
    }
    ff2x_validate_xanes_fms_grid(feff, xsect, momentum_grid)?;

    let fms_trace = ff2x_xanes_optional_fms_trace(work_dir, xsect, fms_spectrum_index)?;
    let combined_trace = if let Some(trace) = configuration_average_trace {
        if trace.len() != xsect.energy_count() {
            bail!(
                "FF2X XANES configuration-average trace length {} does not match xsect.dat energy count {}",
                trace.len(),
                xsect.energy_count()
            );
        }
        trace.to_owned()
    } else {
        ff2x_xanes_combined_trace(xsect, fms_trace.view(), path_sum)?
    };
    let output_energy = ff2x_output_energy_grid_for_input(input, feff, xsect, momentum_grid)?;
    let corrected_background = ff2x_xanes_corrected_background(input, xsect, &output_energy)?;
    let corrected = ff2x_xanes_corrected_components(
        input,
        xsect,
        combined_trace.view(),
        corrected_background.view(),
    )?;
    let corrected =
        ff2x_xanes_apply_vicorr_convolution(input, xsect, list.entries.len(), corrected)?;
    let xmu = ff2x_xanes_xmu_dat_from_components(Ff2xXanesXmuInputs {
        input,
        xsect,
        momentum_grid,
        output_energy: &output_energy,
        corrected: &corrected,
        corrected_background: corrected_background.view(),
        pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    write_xmu_cache(xmu_path, &xmu)?;
    let mut generated_count = 1;
    if write_module_log {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, feff, list, xsect)?;
    }
    Ok(generated_count)
}

fn write_ff2x_nrixs_xmu_outputs(inputs: Ff2xXanesOutputInputs<'_>) -> Result<usize> {
    let Ff2xXanesOutputInputs {
        work_dir,
        xmu_path,
        fms_spectrum_index,
        write_module_log,
        input,
        xsect,
        list,
        feff,
        momentum_grid,
        prepared,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
    } = inputs;

    if path_sum.paths.len() != prepared.len() {
        bail!(
            "FF2X NRIXS/JAS path sum has {} paths for {} prepared paths",
            path_sum.paths.len(),
            prepared.len()
        );
    }
    ff2x_validate_xanes_fms_grid(feff, xsect, momentum_grid)?;

    let fms_trace = ff2x_xanes_optional_fms_trace(work_dir, xsect, fms_spectrum_index)?;
    let combined_trace = if let Some(trace) = configuration_average_trace {
        if trace.len() != xsect.energy_count() {
            bail!(
                "FF2X NRIXS/JAS configuration-average trace length {} does not match xsect.dat energy count {}",
                trace.len(),
                xsect.energy_count()
            );
        }
        trace.to_owned()
    } else {
        ff2x_xanes_combined_trace(xsect, fms_trace.view(), path_sum)?
    };
    let output_energy = ff2x_output_energy_grid_for_input(input, feff, xsect, momentum_grid)?;
    let corrected_background = ff2x_xanes_corrected_background(input, xsect, &output_energy)?;
    let corrected = ff2x_xanes_corrected_components(
        input,
        xsect,
        combined_trace.view(),
        corrected_background.view(),
    )?;
    let corrected =
        ff2x_xanes_apply_vicorr_convolution(input, xsect, list.entries.len(), corrected)?;
    let xmu = ff2x_nrixs_xmu_dat_from_components(Ff2xXanesXmuInputs {
        input,
        xsect,
        momentum_grid,
        output_energy: &output_energy,
        corrected: &corrected,
        corrected_background: corrected_background.view(),
        pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    write_xmu_cache(xmu_path, &xmu)?;
    let mut generated_count = 1;
    if write_module_log {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, feff, list, xsect)?;
    }
    Ok(generated_count)
}

fn write_ff2x_danes_outputs(inputs: Ff2xAnomalousOutputInputs<'_>) -> Result<usize> {
    let Ff2xAnomalousOutputInputs {
        work_dir,
        xmu_path,
        fms_spectrum_index,
        write_module_log,
        input,
        xsect,
        list,
        feff,
        momentum_grid,
        prepared,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
    } = inputs;

    let xmu = ff2x_danes_xmu_dat_from_components(Ff2xAnomalousXmuInputs {
        work_dir,
        fms_spectrum_index,
        input,
        feff,
        xsect,
        momentum_grid,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    write_xmu_cache(xmu_path, &xmu)?;
    let mut generated_count = 1;
    if write_module_log {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, feff, list, xsect)?;
    }
    Ok(generated_count)
}

fn write_ff2x_fprime_outputs(inputs: Ff2xAnomalousOutputInputs<'_>) -> Result<usize> {
    let Ff2xAnomalousOutputInputs {
        work_dir,
        xmu_path,
        fms_spectrum_index,
        write_module_log,
        input,
        xsect,
        list,
        feff,
        momentum_grid,
        prepared,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
    } = inputs;

    let xmu = ff2x_fprime_xmu_dat_from_components(Ff2xAnomalousXmuInputs {
        work_dir,
        fms_spectrum_index,
        input,
        feff,
        xsect,
        momentum_grid,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
        used_path_count: prepared.len(),
        total_path_count: list.entries.len(),
    })?;

    write_xmu_cache(xmu_path, &xmu)?;
    let mut generated_count = 1;
    if write_module_log {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, feff, list, xsect)?;
    }
    Ok(generated_count)
}

fn ff2x_validate_xanes_fms_grid(
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<()> {
    if momentum_grid.output_momentum.len() != xsect.main_energy_count {
        bail!(
            "FF2X XANES output grid got {} momenta for {} xsect.dat main points",
            momentum_grid.output_momentum.len(),
            xsect.main_energy_count
        );
    }
    if momentum_grid.interpolation_momentum.len() != xsect.main_energy_count {
        bail!(
            "FF2X XANES interpolation grid got {} momenta for {} xsect.dat main points",
            momentum_grid.interpolation_momentum.len(),
            xsect.main_energy_count
        );
    }
    if feff.real_momentum.len() < xsect.main_energy_count {
        bail!(
            "FF2X XANES generation needs {} FEFF momenta, got {}",
            xsect.main_energy_count,
            feff.real_momentum.len()
        );
    }
    if xsect.wave_number.len() < xsect.main_energy_count {
        bail!(
            "FF2X XANES generation needs {} xsect.dat momenta, got {}",
            xsect.main_energy_count,
            xsect.wave_number.len()
        );
    }
    for row in 0..xsect.main_energy_count {
        let feff_momentum = feff.real_momentum[row];
        let xsect_momentum = xsect.wave_number[row];
        if !(feff_momentum.is_finite() && xsect_momentum.is_finite()) {
            bail!(
                "FF2X XANES grid row {row} has non-finite momentum: feff.bin={feff_momentum}, xsect.dat={xsect_momentum}"
            );
        }
        let squared_delta = feff_momentum.powi(2) - xsect_momentum.powi(2);
        let relative_delta = squared_delta.abs() / (feff_momentum.powi(2) + 0.001);
        if squared_delta.abs() > FF2X_EPS4 && relative_delta > FF2X_EPS4 {
            bail!(
                "FF2X XANES Emesh in feff.bin and xsect.dat different at row {row}: feff k={}, xsect k={}, delta={squared_delta}",
                feff_momentum / FEFF_BOHR_ANGSTROM,
                xsect_momentum / FEFF_BOHR_ANGSTROM
            );
        }
    }
    Ok(())
}

#[cfg(test)]
fn ff2x_xanes_fms_trace(xsect: &XsectFf2xHandoff, fms: &FmsBinData) -> Result<Array1<Complex64>> {
    ff2x_xanes_fms_trace_for_spectrum(xsect, fms, 0)
}

fn ff2x_xanes_fms_trace_for_spectrum(
    xsect: &XsectFf2xHandoff,
    fms: &FmsBinData,
    spectrum_index: usize,
) -> Result<Array1<Complex64>> {
    if fms.energy_count != xsect.energy_count() {
        bail!(
            "FF2X XANES fms.bin energy count {} does not match xsect.dat count {}",
            fms.energy_count,
            xsect.energy_count()
        );
    }
    if fms.main_energy_count != xsect.main_energy_count {
        bail!(
            "FF2X XANES fms.bin main energy count {} does not match xsect.dat count {}",
            fms.main_energy_count,
            xsect.main_energy_count
        );
    }
    if fms.spectrum_count() == 0 {
        bail!("FF2X XANES fms.bin contains no FMS spectra");
    }
    if spectrum_index >= fms.spectrum_count() {
        bail!(
            "FF2X XANES fms.bin spectrum index {} is outside spectrum count {}",
            spectrum_index,
            fms.spectrum_count()
        );
    }

    Ok(fms
        .spectra
        .row(spectrum_index)
        .mapv(|value| value * xsect.amplitude_reduction))
}

fn ff2x_xanes_optional_fms_trace(
    work_dir: &Path,
    xsect: &XsectFf2xHandoff,
    spectrum_index: usize,
) -> Result<Array1<Complex64>> {
    let fms_path = work_dir.join("fms.bin");
    if !fms_path.is_file() {
        return Ok(Array1::<Complex64>::zeros(xsect.energy_count()));
    }
    let fms = read_fms_bin(&fms_path)
        .with_context(|| format!("failed to read {}", fms_path.display()))?;
    ff2x_xanes_fms_trace_for_spectrum(xsect, &fms, spectrum_index)
}

fn ff2x_nrixs_optional_fmsl_trace(
    work_dir: &Path,
    xsect: &XsectFf2xHandoff,
    max_decomposition_channel: usize,
) -> Result<FmslBinData> {
    let channel_count = max_decomposition_channel.checked_add(1).with_context(|| {
        format!(
            "FF2X NRIXS decomposition channel count overflows for ldecmx={max_decomposition_channel}"
        )
    })?;
    let fmsl_path = work_dir.join("fmsl.bin");
    if !fmsl_path.is_file() {
        return Ok(FmslBinData {
            pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
            max_decomposition_channel,
            traces: Array3::<Complex64>::zeros((
                xsect.energy_count(),
                channel_count,
                channel_count,
            )),
        });
    }

    let fms_path = work_dir.join("fms.bin");
    if !fms_path.is_file() {
        bail!(
            "FF2X NRIXS fmsl.bin handoff requires {} for PAD and energy metadata",
            fms_path.display()
        );
    }
    let fms = read_fms_bin(&fms_path)
        .with_context(|| format!("failed to read {}", fms_path.display()))?;
    if fms.energy_count != xsect.energy_count() {
        bail!(
            "FF2X NRIXS fms.bin has {} energy row(s), but xsect.dat has {}",
            fms.energy_count,
            xsect.energy_count()
        );
    }
    if fms.main_energy_count != xsect.main_energy_count {
        bail!(
            "FF2X NRIXS fms.bin has {} main energy row(s), but xsect.dat has {}",
            fms.main_energy_count,
            xsect.main_energy_count
        );
    }

    let fmsl = read_fmsl_bin(
        &fmsl_path,
        fms.pad_width,
        fms.energy_count,
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to read {}", fmsl_path.display()))?;
    ff2x_validate_finite_complex3("fmsl.bin trace", fmsl.traces.view())?;
    Ok(fmsl)
}

fn ff2x_xanes_combined_trace(
    xsect: &XsectFf2xHandoff,
    fms_trace: ArrayView1<'_, Complex64>,
    path_sum: &Ff2xPathSum,
) -> Result<Array1<Complex64>> {
    if fms_trace.len() != xsect.energy_count() {
        bail!(
            "FF2X XANES FMS trace length {} does not match xsect.dat count {}",
            fms_trace.len(),
            xsect.energy_count()
        );
    }
    if path_sum.total.len() != xsect.main_energy_count {
        bail!(
            "FF2X XANES path contribution length {} does not match xsect.dat main count {}",
            path_sum.total.len(),
            xsect.main_energy_count
        );
    }

    let mut combined = fms_trace.to_owned();
    for row in 0..xsect.main_energy_count {
        combined[row] += path_sum.total[row];
    }
    Ok(combined)
}

fn ff2x_nrixs_combined_decomposed_trace(
    xsect: &XsectFf2xHandoff,
    fmsl: &FmslBinData,
    path_sum: &Ff2xDecomposedPathSum,
) -> Result<Array3<Complex64>> {
    if !xsect.amplitude_reduction.is_finite() {
        bail!(
            "FF2X NRIXS amplitude reduction is not finite: {}",
            xsect.amplitude_reduction
        );
    }
    let channel_count = fmsl
        .max_decomposition_channel
        .checked_add(1)
        .with_context(|| {
            format!(
                "FF2X fmsl.bin decomposition channel count overflows for ldecmx={}",
                fmsl.max_decomposition_channel
            )
        })?;
    let fms_shape = fmsl.traces.shape();
    let expected_fms_shape = [xsect.energy_count(), channel_count, channel_count];
    if fms_shape != expected_fms_shape {
        bail!(
            "FF2X fmsl.bin trace shape {:?} does not match expected {:?}",
            fms_shape,
            expected_fms_shape
        );
    }
    let path_shape = path_sum.total.shape();
    let expected_path_shape = [xsect.main_energy_count, channel_count, channel_count];
    if path_shape != expected_path_shape {
        bail!(
            "FF2X NRIXS decomposed path-sum shape {:?} does not match expected {:?}",
            path_shape,
            expected_path_shape
        );
    }
    ff2x_validate_finite_complex3("fmsl.bin trace", fmsl.traces.view())?;
    ff2x_validate_finite_complex3("NRIXS decomposed path sum", path_sum.total.view())?;

    let mut combined = fmsl.traces.mapv(|value| value * xsect.amplitude_reduction);
    for row in 0..xsect.main_energy_count {
        for lg2 in 0..channel_count {
            for lg1 in 0..channel_count {
                combined[(row, lg2, lg1)] += path_sum.total[(row, lg2, lg1)];
            }
        }
    }
    ff2x_validate_finite_complex3("NRIXS combined decomposed trace", combined.view())?;
    Ok(combined)
}

fn ff2x_nrixs_total_single_electron_response(
    channel_background: ArrayView2<'_, Real>,
) -> Result<Array1<Real>> {
    let row_count = channel_background.nrows();
    let channel_count = channel_background.ncols();
    if row_count == 0 {
        bail!("FF2X NRIXS total response requires at least one energy row");
    }
    if channel_count == 0 {
        bail!("FF2X NRIXS total response requires at least one channel column");
    }
    for ((row, channel), value) in channel_background.indexed_iter() {
        if !value.is_finite() {
            bail!(
                "FF2X NRIXS channel background row {row} channel {channel} is not finite: {value}"
            );
        }
    }

    Ok(Array1::from_iter(
        channel_background
            .axis_iter(Axis(0))
            .map(|row| row.iter().copied().sum::<Real>()),
    ))
}

fn ff2x_nrixs_xmul_output_grid(xsect: &XsectFf2xHandoff) -> Result<(Array1<Real>, Array1<Real>)> {
    let main_energy_count = xsect.main_energy_count;
    if main_energy_count == 0 {
        bail!("FF2X NRIXS xmul grid requires at least one main energy row");
    }
    let energy_count = xsect.energy_count();
    if main_energy_count > energy_count {
        bail!(
            "FF2X NRIXS xmul grid main energy count {main_energy_count} exceeds xsect.dat energy count {energy_count}"
        );
    }
    if xsect.omega_hartree.len() != energy_count {
        bail!(
            "FF2X NRIXS xmul grid omega length {} does not match xsect.dat energy count {}",
            xsect.omega_hartree.len(),
            energy_count
        );
    }
    if xsect.wave_number.len() != energy_count {
        bail!(
            "FF2X NRIXS xmul grid wave-number length {} does not match xsect.dat energy count {}",
            xsect.wave_number.len(),
            energy_count
        );
    }

    let mut photon_energy_ev = Array1::<Real>::zeros(main_energy_count);
    let mut wave_number = Array1::<Real>::zeros(main_energy_count);
    for row in 0..main_energy_count {
        let omega = xsect.omega_hartree[row];
        if !omega.is_finite() {
            bail!("FF2X NRIXS xmul grid omega row {row} is not finite: {omega}");
        }
        let source_wave_number = xsect.wave_number[row];
        if !source_wave_number.is_finite() {
            bail!(
                "FF2X NRIXS xmul grid source wave number row {row} is not finite: {source_wave_number}"
            );
        }

        let energy_ev = omega * FEFF_HARTREE_EV;
        if !energy_ev.is_finite() {
            bail!("FF2X NRIXS xmul grid photon energy row {row} is not finite: {energy_ev}");
        }
        let output_wave_number = source_wave_number / FEFF_BOHR_ANGSTROM;
        if !output_wave_number.is_finite() {
            bail!(
                "FF2X NRIXS xmul grid output wave number row {row} is not finite: {output_wave_number}"
            );
        }

        photon_energy_ev[row] = energy_ev;
        wave_number[row] = output_wave_number;
    }
    Ok((photon_energy_ev, wave_number))
}

fn ff2x_nrixs_xmul_dat_from_components(input: Ff2xNrixsXmulComponents<'_>) -> Result<XmulDatData> {
    let (photon_energy_ev, wave_number) = ff2x_nrixs_xmul_output_grid(input.xsect)?;
    xmul_dat_from_nrixs_decomposition(XmulDatFromNrixsDecompositionInput {
        header_lines: input.header_lines,
        max_decomposition_channel: input.max_decomposition_channel,
        photon_energy_ev: photon_energy_ev.view(),
        wave_number: wave_number.view(),
        channel_background: input.channel_background,
        normalized_fine_structure: input.normalized_fine_structure,
    })
    .context("FF2X NRIXS xmul.dat source assembly")
}

fn write_ff2x_nrixs_xmul_outputs(inputs: Ff2xNrixsOutputInputs<'_>) -> Result<usize> {
    let Ff2xNrixsOutputInputs {
        work_dir,
        write_module_log,
        input,
        xsect,
        list,
        feff,
        prepared,
        max_decomposition_channel,
        pre_table_header_lines,
    } = inputs;

    let feffl_path = work_dir.join("feffl.bin");
    let feffl = read_feffl_bin(
        &feffl_path,
        feff.pad_width,
        prepared.len(),
        feff.energy_count(),
        max_decomposition_channel,
    )
    .with_context(|| format!("failed to read {}", feffl_path.display()))?;
    let output_momentum = ff2x_nrixs_decomposed_output_momentum(xsect)?;
    let path_sum = ff2x_sum_decomposed_paths(feff, &feffl, prepared, output_momentum.view())?;
    let fmsl = ff2x_nrixs_optional_fmsl_trace(work_dir, xsect, max_decomposition_channel)?;
    let combined = ff2x_nrixs_combined_decomposed_trace(xsect, &fmsl, &path_sum)?;

    let xsecl_path = work_dir.join("xsecl.bin");
    let xsecl = read_xsecl_bin(&xsecl_path, feff.pad_width, xsect.energy_count())
        .with_context(|| format!("failed to read {}", xsecl_path.display()))?;
    let channel_cross_sections = ff2x_nrixs_channel_cross_sections_from_xsecl(
        &xsecl,
        max_decomposition_channel,
        xsect.energy_count(),
    )?;
    let (channel_background, normalized_fine_structure) = ff2x_nrixs_corrected_xmul_components(
        input,
        xsect,
        combined.view(),
        channel_cross_sections.view(),
    )?;
    let xmul = ff2x_nrixs_xmul_dat_from_components(Ff2xNrixsXmulComponents {
        header_lines: pre_table_header_lines,
        max_decomposition_channel,
        xsect,
        channel_background: channel_background.view(),
        normalized_fine_structure: normalized_fine_structure.view(),
    })?;

    write_xmul_cache(&work_dir.join("xmul.dat"), &xmul)?;
    let mut generated_count = 1;
    if input.control.ipr6 >= 3 {
        generated_count += write_ff2x_feff_path_outputs(work_dir, feff, list, xsect)?;
    }
    if write_module_log {
        generated_count += write_or_generate_generation_module_log(
            &work_dir.join("log6.dat"),
            input,
            xsect,
            prepared.len(),
        )?;
    }
    Ok(generated_count)
}

fn ff2x_nrixs_decomposed_output_momentum(xsect: &XsectFf2xHandoff) -> Result<Array1<Real>> {
    if xsect.main_energy_count == 0 {
        bail!("FF2X NRIXS decomposed path output requires at least one main energy row");
    }
    if xsect.wave_number.len() < xsect.main_energy_count {
        bail!(
            "FF2X NRIXS decomposed path output needs {} xsect.dat momenta, got {}",
            xsect.main_energy_count,
            xsect.wave_number.len()
        );
    }
    for row in 0..xsect.main_energy_count {
        let momentum = xsect.wave_number[row];
        if !momentum.is_finite() {
            bail!("FF2X NRIXS decomposed path momentum row {row} is not finite: {momentum}");
        }
    }
    Ok(Array1::from_iter(
        xsect
            .wave_number
            .iter()
            .take(xsect.main_energy_count)
            .copied(),
    ))
}

#[cfg(test)]
fn ff2x_nrixs_channel_background_from_xsecl(
    xsecl: &XseclBinData,
    max_decomposition_channel: usize,
    main_energy_count: usize,
) -> Result<Array2<Real>> {
    let channel_cross_sections = ff2x_nrixs_channel_cross_sections_from_xsecl(
        xsecl,
        max_decomposition_channel,
        main_energy_count,
    )?;
    let channel_count = channel_cross_sections.ncols();
    let mut background = Array2::<Real>::zeros((main_energy_count, channel_count));
    for row in 0..main_energy_count {
        for channel in 0..channel_count {
            background[(row, channel)] =
                channel_cross_sections[(row, channel)].im / FEFF_HARTREE_EV;
        }
    }
    Ok(background)
}

fn ff2x_nrixs_channel_cross_sections_from_xsecl(
    xsecl: &XseclBinData,
    max_decomposition_channel: usize,
    required_energy_count: usize,
) -> Result<Array2<Complex64>> {
    let channel_count = max_decomposition_channel
        .checked_add(1)
        .context("FF2X NRIXS channel count overflowed")?;
    if xsecl.energy_count() < required_energy_count {
        bail!(
            "FF2X NRIXS xsecl.bin has {} energy row(s), but xsect.dat needs {} row(s)",
            xsecl.energy_count(),
            required_energy_count
        );
    }
    if xsecl.transition_index_count() > xsecl.final_state_count() {
        bail!(
            "FF2X NRIXS xsecl.bin has {} transition index row(s), but only {} final-state cross-section column(s)",
            xsecl.transition_index_count(),
            xsecl.final_state_count()
        );
    }

    let mut channel_cross_sections =
        Array2::<Complex64>::zeros((xsecl.energy_count(), channel_count));
    for (transition_index, transition) in xsecl.transitions.iter().enumerate() {
        let channel = usize::try_from(transition.decomposition_channel).with_context(|| {
            format!(
                "FF2X NRIXS xsecl.bin transition {transition_index} has negative decomposition channel {}",
                transition.decomposition_channel
            )
        })?;
        if channel > max_decomposition_channel {
            continue;
        }
        for row in 0..xsecl.energy_count() {
            let value = xsecl.atom_cross_sections[(row, transition_index)];
            if !(value.re.is_finite() && value.im.is_finite()) {
                bail!(
                    "FF2X NRIXS xsecl.bin row {row} transition {transition_index} background is not finite: {value:?}"
                );
            }
            channel_cross_sections[(row, channel)] += value;
        }
    }
    Ok(channel_cross_sections)
}

fn ff2x_nrixs_corrected_xmul_components(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    combined: ArrayView3<'_, Complex64>,
    channel_cross_sections: ArrayView2<'_, Complex64>,
) -> Result<(Array2<Real>, Array3<Real>)> {
    let energy_count = xsect.energy_count();
    let main_energy_count = xsect.main_energy_count;
    let combined_shape = combined.shape();
    let channel_shape = channel_cross_sections.shape();
    if combined_shape.len() != 3 {
        bail!("FF2X NRIXS corrected xmul trace must be 3D, got shape {combined_shape:?}");
    }
    if combined_shape[0] != energy_count {
        bail!(
            "FF2X NRIXS corrected xmul trace has {} energy rows, expected {}",
            combined_shape[0],
            energy_count
        );
    }
    if channel_shape[0] < energy_count
        || channel_shape[1] != combined_shape[1]
        || combined_shape[1] != combined_shape[2]
    {
        bail!(
            "FF2X NRIXS corrected xmul channel shape {:?} does not match trace shape {:?}",
            channel_shape,
            combined_shape
        );
    }
    if xsect.normalized_background.len() != energy_count {
        bail!(
            "FF2X NRIXS corrected xmul background length {} does not match xsect energy count {}",
            xsect.normalized_background.len(),
            energy_count
        );
    }

    let channel_count = combined_shape[1];
    let zero_path_chi = Array1::<Complex64>::zeros(energy_count);
    let mut channel_background = Array2::<Real>::zeros((main_energy_count, channel_count));
    for channel in 0..channel_count {
        let mut cross_section = Array1::<Complex64>::zeros(energy_count);
        for row in 0..energy_count {
            cross_section[row] = channel_cross_sections[(row, channel)];
        }
        let correction = ff2x_xscorr(Ff2xXscorrInput {
            ispec: input.control.ispec,
            energy_grid_hartree: xsect.energy_grid_hartree.view(),
            main_energy_count,
            fermi_index: xsect.fermi_index,
            cross_section: cross_section.view(),
            background: xsect.normalized_background.view(),
            path_chi: zero_path_chi.view(),
            real_correction_hartree: ff2x_real_correction_hartree(input),
            electronic_temperature_ev: input.electronic_temperature,
        })?;
        for row in 0..main_energy_count {
            channel_background[(row, channel)] =
                (cross_section[row] + correction[row]).im / FEFF_HARTREE_EV;
        }
    }

    let total_background = ff2x_nrixs_total_single_electron_response(channel_background.view())?;
    let mut normalized = Array3::<Real>::zeros((main_energy_count, channel_count, channel_count));
    let zero_cross_section = Array1::<Complex64>::zeros(energy_count);
    for angular in 0..channel_count {
        for l_star in 0..channel_count {
            let mut path_chi = Array1::<Complex64>::zeros(energy_count);
            for row in 0..energy_count {
                path_chi[row] = combined[(row, l_star, angular)];
            }
            let mut cross_section = zero_cross_section.clone();
            if l_star == angular {
                for row in 0..energy_count {
                    cross_section[row] = channel_cross_sections[(row, angular)];
                }
            }
            let correction = ff2x_xscorr(Ff2xXscorrInput {
                ispec: input.control.ispec,
                energy_grid_hartree: xsect.energy_grid_hartree.view(),
                main_energy_count,
                fermi_index: xsect.fermi_index,
                cross_section: cross_section.view(),
                background: xsect.normalized_background.view(),
                path_chi: path_chi.view(),
                real_correction_hartree: ff2x_real_correction_hartree(input),
                electronic_temperature_ev: input.electronic_temperature,
            })?;
            for row in 0..main_energy_count {
                let total = (cross_section[row]
                    + xsect.normalized_background[row] * path_chi[row]
                    + correction[row])
                    .im;
                let atomic = if l_star == angular {
                    channel_background[(row, angular)] * FEFF_HARTREE_EV
                } else {
                    0.0
                };
                let denominator = total_background[row] * FEFF_HARTREE_EV;
                let value = if denominator.abs() > FF2X_EPS {
                    (total - atomic) / denominator
                } else {
                    0.0
                };
                if !value.is_finite() {
                    bail!(
                        "FF2X NRIXS corrected xmul fine structure row {row} channel ({l_star},{angular}) is not finite: {value}"
                    );
                }
                normalized[(row, l_star, angular)] = value;
            }
        }
    }

    Ok((channel_background, normalized))
}

fn ff2x_validate_finite_complex3(
    label: &'static str,
    values: ArrayView3<'_, Complex64>,
) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            bail!("FF2X {label} value {index} is not finite: {value:?}");
        }
    }
    Ok(())
}

fn ff2x_xanes_corrected_background(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    output_energy: &Ff2xOutputEnergyGrid,
) -> Result<Array1<Real>> {
    let mut background = xsect.normalized_background.clone();
    if input.control.mbconv <= 0 {
        return Ok(background);
    }

    let main_energy_count = xsect.main_energy_count;
    if main_energy_count > xsect.energy_count() {
        bail!(
            "FF2X XANES mbconv main energy count {} exceeds xsect energy count {}",
            main_energy_count,
            xsect.energy_count()
        );
    }
    if background.len() != xsect.energy_count() {
        bail!(
            "FF2X XANES mbconv background length {} does not match xsect energy count {}",
            background.len(),
            xsect.energy_count()
        );
    }

    let energy = Array1::from_iter(xsect.omega_hartree.iter().take(main_energy_count).copied());
    let xsnorm = Array1::from_iter(background.iter().take(main_energy_count).copied());
    let convolved_background = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: energy.view(),
        xmu: xsnorm.view(),
        fermi_energy: ff2x_mbconv_fermi_energy(output_energy.fermi_energy_hartree, xsect),
        amplitude_reduction: xsect.file_amplitude_reduction,
        relaxation_energy: xsect.relaxation_energy,
        plasmon_frequency: xsect.plasmon_frequency * 0.5,
    })
    .context("FF2X XANES mbconv background excitation convolution")?;
    for (row, &value) in convolved_background.iter().enumerate() {
        background[row] = value;
    }

    Ok(background)
}

fn ff2x_xanes_corrected_components(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    fms_trace: ArrayView1<'_, Complex64>,
    corrected_background: ArrayView1<'_, Real>,
) -> Result<Ff2xXanesCorrectedComponents> {
    if fms_trace.len() != xsect.energy_count() {
        bail!(
            "FF2X XANES FMS trace length {} does not match xsect.dat count {}",
            fms_trace.len(),
            xsect.energy_count()
        );
    }
    if corrected_background.len() != xsect.energy_count() {
        bail!(
            "FF2X XANES corrected background length {} does not match xsect.dat count {}",
            corrected_background.len(),
            xsect.energy_count()
        );
    }
    let ispec = ff2x_xmu_effective_ispec(input.control.ispec)
        .context("FF2X XANES corrected components require a regular XANES ispec")?;

    let total_correction = ff2x_xscorr(Ff2xXscorrInput {
        ispec,
        energy_grid_hartree: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        fermi_index: xsect.fermi_index,
        cross_section: xsect.cross_section.view(),
        background: corrected_background,
        path_chi: fms_trace,
        real_correction_hartree: ff2x_real_correction_hartree(input),
        electronic_temperature_ev: input.electronic_temperature,
    })?;

    let zero_trace = Array1::<Complex64>::zeros(xsect.energy_count());
    let atomic_correction = ff2x_xscorr(Ff2xXscorrInput {
        ispec,
        energy_grid_hartree: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        fermi_index: xsect.fermi_index,
        cross_section: xsect.cross_section.view(),
        background: corrected_background,
        path_chi: zero_trace.view(),
        real_correction_hartree: ff2x_real_correction_hartree(input),
        electronic_temperature_ev: input.electronic_temperature,
    })?;

    let mut total = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut atomic = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut fine_structure = Array1::<Real>::zeros(xsect.main_energy_count);
    for row in 0..xsect.main_energy_count {
        let total_value = xsect.cross_section[row]
            + corrected_background[row] * fms_trace[row]
            + total_correction[row];
        let atomic_value = xsect.cross_section[row] + atomic_correction[row];
        total[row] = total_value.im;
        atomic[row] = atomic_value.im;
        fine_structure[row] = total[row] - atomic[row];
    }

    Ok(Ff2xXanesCorrectedComponents {
        total,
        atomic,
        fine_structure,
    })
}

fn ff2x_xanes_apply_vicorr_convolution(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    total_path_count: usize,
    mut corrected: Ff2xXanesCorrectedComponents,
) -> Result<Ff2xXanesCorrectedComponents> {
    let imaginary_correction = ff2x_effective_imaginary_correction_hartree(input, xsect);
    if imaginary_correction <= FF2X_EPS4 || total_path_count != 0 {
        return Ok(corrected);
    }

    let main_energy_count = xsect.main_energy_count;
    if xsect.omega_hartree.len() < main_energy_count {
        bail!(
            "FF2X XANES vicorr convolution needs {} omega points, got {}",
            main_energy_count,
            xsect.omega_hartree.len()
        );
    }
    if corrected.total.len() != main_energy_count
        || corrected.atomic.len() != main_energy_count
        || corrected.fine_structure.len() != main_energy_count
    {
        bail!("FF2X XANES vicorr convolution component lengths do not match main energy count");
    }

    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("FF2X XANES omega grid is not contiguous")?;
    let spectrum = (0..main_energy_count)
        .map(|row| Complex64::new(corrected.total[row], corrected.atomic[row]))
        .collect::<Vec<_>>();
    let convolved = lorentz_convolve(&omega[..main_energy_count], &spectrum, imaginary_correction)
        .context("FF2X XANES vicorr Lorentzian convolution")?;

    for row in 0..main_energy_count {
        corrected.total[row] = convolved[row].re;
        corrected.atomic[row] = convolved[row].im;
        corrected.fine_structure[row] = corrected.total[row] - corrected.atomic[row];
    }
    Ok(corrected)
}

fn ff2x_xanes_xmu_dat_from_components(inputs: Ff2xXanesXmuInputs<'_>) -> Result<XmuDatData> {
    let Ff2xXanesXmuInputs {
        input,
        xsect,
        momentum_grid,
        output_energy,
        corrected,
        corrected_background,
        pre_table_header_lines,
        used_path_count,
        total_path_count,
    } = inputs;
    let output_count = momentum_grid.output_momentum.len();
    if output_count != xsect.main_energy_count {
        bail!(
            "FF2X XANES xmu.dat row build got {} output rows for {} main xsect.dat points",
            output_count,
            xsect.main_energy_count
        );
    }
    if output_energy.photon_energy_hartree.len() != output_count
        || output_energy.relative_energy_hartree.len() != output_count
    {
        bail!("FF2X XANES output energy grid length does not match output count {output_count}");
    }
    if corrected.total.len() != output_count
        || corrected.atomic.len() != output_count
        || corrected.fine_structure.len() != output_count
    {
        bail!("FF2X XANES corrected component lengths do not match output count {output_count}");
    }

    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("FF2X xsect.dat omega grid is not contiguous")?;
    if corrected_background.len() != xsect.energy_count() {
        bail!(
            "FF2X XANES xmu.dat corrected background length {} does not match xsect.dat count {}",
            corrected_background.len(),
            xsect.energy_count()
        );
    }
    let ispec = ff2x_xmu_effective_ispec(input.control.ispec)
        .context("FF2X XANES xmu.dat rows require a regular XANES ispec")?;
    let background = corrected_background
        .as_slice()
        .context("FF2X XANES corrected background grid is not contiguous")?;
    let omega = &omega[..xsect.main_energy_count];
    let background = &background[..xsect.main_energy_count];
    let mut normalization = if input.control.absolu == 1 {
        1.0
    } else {
        let normalization_energy = if ispec == 2 {
            output_energy.fermi_energy_hartree
        } else {
            output_energy.fermi_energy_hartree + 50.0 / FEFF_HARTREE_EV
        };
        terp(omega, background, 1, normalization_energy)
            .context("FF2X XANES xmu.dat xsedge interpolation")?
            .value
    };
    if !normalization.is_finite() {
        bail!("FF2X XANES xmu.dat normalization is not finite: {normalization}");
    }
    if normalization == 0.0 {
        bail!("FF2X XANES xmu.dat normalization is zero");
    }
    if input.control.absolu == 1 {
        normalization = 1.0;
    }

    let mut photon_energy_ev = Array1::<Real>::zeros(output_count);
    let mut relative_energy_ev = Array1::<Real>::zeros(output_count);
    let mut wave_number = Array1::<Real>::zeros(output_count);
    let mut mu = Array1::<Real>::zeros(output_count);
    let mut mu0 = Array1::<Real>::zeros(output_count);
    let mut chi = Array1::<Real>::zeros(output_count);
    for row in 0..output_count {
        let k_inverse_angstrom = momentum_grid.output_momentum[row] / FEFF_BOHR_ANGSTROM;
        let mut fine_structure = corrected.fine_structure[row];
        if input.debye.sig_gk > FF2X_EPS4 {
            fine_structure *= (-(input.debye.sig_gk * k_inverse_angstrom).powi(2)).exp();
        }
        photon_energy_ev[row] = output_energy.photon_energy_hartree[row] * FEFF_HARTREE_EV;
        relative_energy_ev[row] = output_energy.relative_energy_hartree[row] * FEFF_HARTREE_EV;
        wave_number[row] = k_inverse_angstrom;
        mu0[row] = corrected.atomic[row] / normalization;
        chi[row] = fine_structure / normalization;
        mu[row] = mu0[row] + chi[row];
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        format!("# xsedge+ 50, used to normalize mu {normalization:20.4E}"),
        "# -----------------------------------------------------------------------".to_string(),
        "# omega    e    k    mu    mu0     chi     @#".to_string(),
    ]);

    Ok(XmuDatData {
        header_lines,
        normalization: Some(normalization),
        photon_energy_ev,
        relative_energy_ev,
        wave_number,
        mu,
        mu0,
        chi,
    })
}

fn ff2x_nrixs_xmu_dat_from_components(inputs: Ff2xXanesXmuInputs<'_>) -> Result<XmuDatData> {
    let Ff2xXanesXmuInputs {
        input,
        xsect,
        momentum_grid,
        output_energy,
        corrected,
        corrected_background: _,
        pre_table_header_lines,
        used_path_count,
        total_path_count,
    } = inputs;
    let output_count = momentum_grid.output_momentum.len();
    if output_count != xsect.main_energy_count {
        bail!(
            "FF2X NRIXS/JAS xmu.dat row build got {} output rows for {} main xsect.dat points",
            output_count,
            xsect.main_energy_count
        );
    }
    if output_energy.photon_energy_hartree.len() != output_count
        || output_energy.relative_energy_hartree.len() != output_count
    {
        bail!(
            "FF2X NRIXS/JAS output energy grid length does not match output count {output_count}"
        );
    }
    if corrected.total.len() != output_count
        || corrected.atomic.len() != output_count
        || corrected.fine_structure.len() != output_count
    {
        bail!(
            "FF2X NRIXS/JAS corrected component lengths do not match output count {output_count}"
        );
    }

    let mut photon_energy_ev = Array1::<Real>::zeros(output_count);
    let mut relative_energy_ev = Array1::<Real>::zeros(output_count);
    let mut wave_number = Array1::<Real>::zeros(output_count);
    let mut mu = Array1::<Real>::zeros(output_count);
    let mut mu0 = Array1::<Real>::zeros(output_count);
    let mut chi = Array1::<Real>::zeros(output_count);
    for row in 0..output_count {
        let k_inverse_angstrom = momentum_grid.output_momentum[row] / FEFF_BOHR_ANGSTROM;
        let mut fine_structure = corrected.fine_structure[row];
        if input.debye.sig_gk > FF2X_EPS4 {
            fine_structure *= (-(input.debye.sig_gk * k_inverse_angstrom).powi(2)).exp();
        }
        photon_energy_ev[row] = output_energy.photon_energy_hartree[row] * FEFF_HARTREE_EV;
        relative_energy_ev[row] = output_energy.relative_energy_hartree[row] * FEFF_HARTREE_EV;
        wave_number[row] = k_inverse_angstrom;
        mu0[row] = corrected.atomic[row] / FEFF_HARTREE_EV;
        chi[row] = fine_structure / FEFF_HARTREE_EV;
        mu[row] = corrected.total[row] / FEFF_HARTREE_EV;
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        "# Contribution to S(q,w) from a single electron".to_string(),
        "# -----------------------------------------------------------------------".to_string(),
        "# omega    e    k   S(qw)  S^0(qw)  chi_q*S^0(qw)       @#".to_string(),
    ]);

    Ok(XmuDatData {
        header_lines,
        normalization: None,
        photon_energy_ev,
        relative_energy_ev,
        wave_number,
        mu,
        mu0,
        chi,
    })
}

fn ff2x_danes_xmu_dat_from_components(inputs: Ff2xAnomalousXmuInputs<'_>) -> Result<XmuDatData> {
    let Ff2xAnomalousXmuInputs {
        work_dir,
        fms_spectrum_index,
        input,
        feff,
        xsect,
        momentum_grid,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
        used_path_count,
        total_path_count,
    } = inputs;
    let (extension_count, fms_trace) =
        ff2x_danes_extension_and_fms_trace(work_dir, xsect, fms_spectrum_index)?;
    if momentum_grid.output_momentum.len() != xsect.energy_count()
        || momentum_grid.interpolation_momentum.len() != xsect.energy_count()
    {
        bail!(
            "FF2X DANES momentum grid lengths must match xsect.dat energy count {}",
            xsect.energy_count()
        );
    }
    if path_sum.total.len() != xsect.energy_count() {
        bail!(
            "FF2X DANES path sum has {} rows for {} xsect.dat energies",
            path_sum.total.len(),
            xsect.energy_count()
        );
    }

    let output_energy = ff2x_output_energy_grid_for_input(input, feff, xsect, momentum_grid)?;
    let normalized_background =
        ff2x_fprime_corrected_background(input, xsect, output_energy.fermi_energy_hartree)?;
    let normalization = ff2x_fprime_normalization(
        input,
        xsect,
        normalized_background.view(),
        output_energy.fermi_energy_hartree,
    )?;
    let (converted_cross_section, converted_background) = ff2x_fprime_units(
        xsect,
        normalized_background.view(),
        output_energy.fermi_energy_hartree,
    )?;
    let path_chi = ff2x_configuration_or_path_sum_with_fms_trace(
        "DANES",
        xsect,
        path_sum,
        fms_trace.view(),
        configuration_average_trace,
    )?;
    let zero_path_chi = Array1::<Complex64>::zeros(xsect.energy_count());
    let real_correction = ff2x_real_correction_hartree(input);
    let imaginary_correction = ff2x_imaginary_correction_hartree(input);
    let correction = fprime_correction(FprimeCorrectionInput {
        edge_reference_energy: output_energy.fermi_energy_hartree,
        energy: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        extension_count,
        fermi_index: xsect.fermi_index,
        cross_section: converted_cross_section.view(),
        background: converted_background.view(),
        path_chi: path_chi.view(),
        real_correction,
        imaginary_correction,
    })
    .context("FF2X DANES total fprime correction")?;
    let atomic_correction = fprime_correction(FprimeCorrectionInput {
        edge_reference_energy: output_energy.fermi_energy_hartree,
        energy: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        extension_count,
        fermi_index: xsect.fermi_index,
        cross_section: converted_cross_section.view(),
        background: converted_background.view(),
        path_chi: zero_path_chi.view(),
        real_correction,
        imaginary_correction,
    })
    .context("FF2X DANES atomic fprime correction")?;

    let mut photon_energy_ev = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut relative_energy_ev = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut wave_number = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut mu = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut mu0 = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut chi = Array1::<Real>::zeros(xsect.main_energy_count);

    for row in 0..xsect.main_energy_count {
        let path_term = converted_background[row] * path_chi[row];
        let total_real = (path_term + correction[row]).re;
        let atomic_real = atomic_correction[row].re;
        let chi_real = total_real - atomic_real;
        photon_energy_ev[row] = xsect.omega_hartree[row] * FEFF_HARTREE_EV;
        relative_energy_ev[row] = xsect.energy_grid_hartree[row].re * FEFF_HARTREE_EV;
        wave_number[row] = momentum_grid.output_momentum[row] / FEFF_BOHR_ANGSTROM;
        mu[row] = -total_real;
        mu0[row] = -atomic_real;
        chi[row] = -chi_real;
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        format!("# xsedge+ 50, used to normalize mu {normalization:20.4E}"),
        "# -----------------------------------------------------------------------".to_string(),
        "# omega    e    k    mu    mu0     chi     @#".to_string(),
    ]);

    Ok(XmuDatData {
        header_lines,
        normalization: Some(normalization),
        photon_energy_ev,
        relative_energy_ev,
        wave_number,
        mu,
        mu0,
        chi,
    })
}

fn ff2x_fprime_xmu_dat_from_components(inputs: Ff2xAnomalousXmuInputs<'_>) -> Result<XmuDatData> {
    let Ff2xAnomalousXmuInputs {
        work_dir,
        fms_spectrum_index,
        input,
        feff,
        xsect,
        momentum_grid,
        path_sum,
        configuration_average_trace,
        pre_table_header_lines,
        used_path_count,
        total_path_count,
    } = inputs;
    let extension_count = ff2x_fprime_extension_count(xsect)?;
    if momentum_grid.output_momentum.len() != xsect.energy_count()
        || momentum_grid.interpolation_momentum.len() != xsect.energy_count()
    {
        bail!(
            "FF2X FPRIME momentum grid lengths must match xsect.dat energy count {}",
            xsect.energy_count()
        );
    }
    if path_sum.total.len() != xsect.energy_count() {
        bail!(
            "FF2X FPRIME path sum has {} rows for {} xsect.dat energies",
            path_sum.total.len(),
            xsect.energy_count()
        );
    }

    let output_energy = ff2x_output_energy_grid_for_input(input, feff, xsect, momentum_grid)?;
    let normalized_background =
        ff2x_fprime_corrected_background(input, xsect, output_energy.fermi_energy_hartree)?;
    let normalization = ff2x_fprime_normalization(
        input,
        xsect,
        normalized_background.view(),
        output_energy.fermi_energy_hartree,
    )?;
    let (converted_cross_section, converted_background) = ff2x_fprime_units(
        xsect,
        normalized_background.view(),
        output_energy.fermi_energy_hartree,
    )?;
    let fms_trace = ff2x_fprime_fms_trace(work_dir, xsect, fms_spectrum_index)?;
    let path_chi = ff2x_configuration_or_path_sum_with_fms_trace(
        "FPRIME",
        xsect,
        path_sum,
        fms_trace.view(),
        configuration_average_trace,
    )?;
    let zero_path_chi = Array1::<Complex64>::zeros(xsect.energy_count());
    let real_correction = ff2x_real_correction_hartree(input);
    let imaginary_correction = ff2x_imaginary_correction_hartree(input);
    let correction = fprime_correction(FprimeCorrectionInput {
        edge_reference_energy: output_energy.fermi_energy_hartree,
        energy: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        extension_count,
        fermi_index: xsect.fermi_index,
        cross_section: converted_cross_section.view(),
        background: converted_background.view(),
        path_chi: path_chi.view(),
        real_correction,
        imaginary_correction,
    })
    .context("FF2X FPRIME total fprime correction")?;
    let atomic_correction = fprime_correction(FprimeCorrectionInput {
        edge_reference_energy: output_energy.fermi_energy_hartree,
        energy: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        extension_count,
        fermi_index: xsect.fermi_index,
        cross_section: converted_cross_section.view(),
        background: converted_background.view(),
        path_chi: zero_path_chi.view(),
        real_correction,
        imaginary_correction,
    })
    .context("FF2X FPRIME atomic fprime correction")?;

    let mut photon_energy_ev = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut relative_energy_ev = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut f_prime_total = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut f_prime_atomic = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut f_double_prime_total = Array1::<Real>::zeros(xsect.main_energy_count);
    let mut f_double_prime_atomic = Array1::<Real>::zeros(xsect.main_energy_count);

    for row in 0..xsect.main_energy_count {
        let path_term = converted_background[row] * path_chi[row];
        let total_real = (path_term + correction[row]).re;
        let atomic_real = atomic_correction[row].re;
        photon_energy_ev[row] = xsect.omega_hartree[row] * FEFF_HARTREE_EV;
        relative_energy_ev[row] = xsect.energy_grid_hartree[row].re * FEFF_HARTREE_EV;
        f_prime_total[row] = -total_real;
        f_prime_atomic[row] = -atomic_real;
        f_double_prime_total[row] = converted_background[row] + path_term.im;
        f_double_prime_atomic[row] = converted_background[row];
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        format!("# xsedge+ 50, used to normalize mu {normalization:20.4E}"),
        "# -----------------------------------------------------------------------".to_string(),
        "# omega    e    f'    f'0    f''    f''0     @#".to_string(),
    ]);

    Ok(XmuDatData {
        header_lines,
        normalization: Some(normalization),
        photon_energy_ev,
        relative_energy_ev,
        wave_number: f_prime_total,
        mu: f_prime_atomic,
        mu0: f_double_prime_total,
        chi: f_double_prime_atomic,
    })
}

fn ff2x_fprime_extension_count(xsect: &XsectFf2xHandoff) -> Result<usize> {
    if xsect.main_energy_count >= xsect.energy_count() {
        bail!(
            "FF2X FPRIME requires positive-axis extension rows after ne1={}, got {} total rows",
            xsect.main_energy_count,
            xsect.energy_count()
        );
    }
    let extension_count = xsect.energy_count() - xsect.main_energy_count;
    if extension_count < 4 {
        bail!(
            "FF2X FPRIME requires at least 4 positive-axis extension rows, got {extension_count}"
        );
    }
    Ok(extension_count)
}

fn ff2x_fprime_corrected_background(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    fermi_energy_hartree: Real,
) -> Result<Array1<Real>> {
    let mut background = xsect.normalized_background.clone();
    if input.control.mbconv <= 0 {
        return Ok(background);
    }

    let main_energy_count = xsect.main_energy_count;
    let xsect_energy =
        Array1::from_iter(xsect.omega_hartree.iter().take(main_energy_count).copied());
    let xsect_background = Array1::from_iter(background.iter().take(main_energy_count).copied());
    let convolved_background = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: xsect_energy.view(),
        xmu: xsect_background.view(),
        fermi_energy: ff2x_mbconv_fermi_energy(fermi_energy_hartree, xsect),
        amplitude_reduction: xsect.file_amplitude_reduction,
        relaxation_energy: xsect.relaxation_energy,
        plasmon_frequency: xsect.plasmon_frequency * 0.5,
    })
    .context("FF2X FPRIME mbconv background excitation convolution")?;
    for (row, &value) in convolved_background.iter().enumerate() {
        background[row] = value;
    }
    Ok(background)
}

fn ff2x_fprime_normalization(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    background: ArrayView1<'_, Real>,
    fermi_energy_hartree: Real,
) -> Result<Real> {
    if background.len() != xsect.energy_count() {
        bail!(
            "FF2X FPRIME background length {} does not match xsect.dat count {}",
            background.len(),
            xsect.energy_count()
        );
    }
    if input.control.absolu == 1 {
        return Ok(1.0);
    }
    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("FF2X FPRIME omega grid is not contiguous")?;
    let background = background
        .as_slice()
        .context("FF2X FPRIME background grid is not contiguous")?;
    let edge_plus_50 = fermi_energy_hartree + 50.0 / FEFF_HARTREE_EV;
    let normalization = terp(
        &omega[..xsect.main_energy_count],
        &background[..xsect.main_energy_count],
        1,
        edge_plus_50,
    )
    .context("FF2X FPRIME xsedge interpolation")?
    .value;
    if !normalization.is_finite() {
        bail!("FF2X FPRIME normalization is not finite: {normalization}");
    }
    if normalization == 0.0 {
        bail!("FF2X FPRIME normalization is zero");
    }
    Ok(normalization)
}

fn ff2x_fprime_units(
    xsect: &XsectFf2xHandoff,
    background: ArrayView1<'_, Real>,
    fermi_energy_hartree: Real,
) -> Result<(Array1<Complex64>, Array1<Real>)> {
    if background.len() != xsect.energy_count() {
        bail!(
            "FF2X FPRIME unit conversion got {} background rows for {} xsect.dat rows",
            background.len(),
            xsect.energy_count()
        );
    }
    let mut cross_section = Array1::<Complex64>::zeros(xsect.energy_count());
    let mut normalized_background = Array1::<Real>::zeros(xsect.energy_count());
    for row in 0..xsect.energy_count() {
        let energy = xsect.energy_grid_hartree[row].re + fermi_energy_hartree;
        if !energy.is_finite() {
            bail!("FF2X FPRIME unit conversion row {row} got non-finite energy {energy}");
        }
        let prefactor = 4.0 * PI * FEFF_ALPHA_INV / energy * FEFF_BOHR_ANGSTROM.powi(2);
        if !prefactor.is_finite() || prefactor == 0.0 {
            bail!("FF2X FPRIME unit conversion row {row} got invalid prefactor {prefactor}");
        }
        let scale = FEFF_ALPHA_INV.powi(2) / prefactor;
        cross_section[row] = xsect.cross_section[row] * scale;
        normalized_background[row] = background[row] * scale;
    }
    Ok((cross_section, normalized_background))
}

fn ff2x_fprime_fms_trace(
    work_dir: &Path,
    xsect: &XsectFf2xHandoff,
    spectrum_index: usize,
) -> Result<Array1<Complex64>> {
    let fms_path = work_dir.join("fms.bin");
    if !fms_path.is_file() {
        return Ok(Array1::<Complex64>::zeros(xsect.energy_count()));
    }
    let fms = read_fms_bin(&fms_path)
        .with_context(|| format!("failed to read {}", fms_path.display()))?;
    if fms.energy_count != xsect.energy_count() {
        bail!(
            "FF2X FPRIME fms.bin energy count {} does not match xsect.dat count {}",
            fms.energy_count,
            xsect.energy_count()
        );
    }
    if fms.spectrum_count() == 0 {
        bail!("FF2X FPRIME fms.bin contains no FMS spectra");
    }
    if spectrum_index >= fms.spectrum_count() {
        bail!(
            "FF2X FPRIME fms.bin spectrum index {} is outside spectrum count {}",
            spectrum_index,
            fms.spectrum_count()
        );
    }
    Ok(fms.spectra.row(spectrum_index).to_owned())
}

fn ff2x_danes_extension_and_fms_trace(
    work_dir: &Path,
    xsect: &XsectFf2xHandoff,
    spectrum_index: usize,
) -> Result<(usize, Array1<Complex64>)> {
    let fms_path = work_dir.join("fms.bin");
    if fms_path.is_file() {
        let fms = read_fms_bin(&fms_path)
            .with_context(|| format!("failed to read {}", fms_path.display()))?;
        if fms.energy_count != xsect.energy_count() {
            bail!(
                "FF2X DANES fms.bin energy count {} does not match xsect.dat count {}",
                fms.energy_count,
                xsect.energy_count()
            );
        }
        if fms.main_energy_count != xsect.main_energy_count {
            bail!(
                "FF2X DANES fms.bin main energy count {} does not match xsect.dat ne1 {}",
                fms.main_energy_count,
                xsect.main_energy_count
            );
        }
        if fms.spectrum_count() == 0 {
            bail!("FF2X DANES fms.bin contains no FMS spectra");
        }
        if spectrum_index >= fms.spectrum_count() {
            bail!(
                "FF2X DANES fms.bin spectrum index {} is outside spectrum count {}",
                spectrum_index,
                fms.spectrum_count()
            );
        }
        ff2x_validate_danes_extension_count(fms.auxiliary_energy_count, xsect)?;
        return Ok((
            fms.auxiliary_energy_count,
            fms.spectra.row(spectrum_index).to_owned(),
        ));
    }

    let phase_path = work_dir.join("phase.bin");
    if phase_path.is_file() {
        let phase = read_phase_bin(&phase_path)
            .with_context(|| format!("failed to read {}", phase_path.display()))?;
        if phase.energy_count != xsect.energy_count() {
            bail!(
                "FF2X DANES phase.bin energy count {} does not match xsect.dat count {}",
                phase.energy_count,
                xsect.energy_count()
            );
        }
        if phase.main_energy_count != xsect.main_energy_count {
            bail!(
                "FF2X DANES phase.bin main energy count {} does not match xsect.dat ne1 {}",
                phase.main_energy_count,
                xsect.main_energy_count
            );
        }
        ff2x_validate_danes_extension_count(phase.auxiliary_energy_count, xsect)?;
        return Ok((
            phase.auxiliary_energy_count,
            Array1::<Complex64>::zeros(xsect.energy_count()),
        ));
    }

    bail!("FF2X DANES generation requires fms.bin or phase.bin to determine ne3")
}

fn ff2x_validate_danes_extension_count(
    extension_count: usize,
    xsect: &XsectFf2xHandoff,
) -> Result<()> {
    let trailing_count = xsect.energy_count() - xsect.main_energy_count;
    if extension_count > trailing_count {
        bail!("FF2X DANES ne3 {extension_count} exceeds trailing xsect.dat rows {trailing_count}");
    }
    let contour_count = trailing_count - extension_count;
    if contour_count < 3 {
        bail!("FF2X DANES vertical contour requires at least 3 points, got {contour_count}");
    }
    Ok(())
}

fn write_ff2x_feff_path_outputs(
    work_dir: &Path,
    feff: &FeffBinData,
    list: &ListDatData,
    xsect: &XsectFf2xHandoff,
) -> Result<usize> {
    let selected_paths = ff2x_listed_feff_paths(feff, list)?;
    write_ff2x_files_dat(&work_dir.join("files.dat"), xsect, &selected_paths)?;

    for path in &selected_paths {
        let data = ff2x_feff_path_data(feff, xsect, path)?;
        write_sfconv_so2conv_feff_path_data(
            work_dir.join(format!("feff{:04}.dat", path.path.index)),
            &data,
        )
        .with_context(|| {
            format!(
                "failed to write {}",
                work_dir
                    .join(format!("feff{:04}.dat", path.path.index))
                    .display()
            )
        })?;
    }

    Ok(selected_paths.len() + 1)
}

#[derive(Debug, Clone, Copy)]
struct Ff2xListedFeffPath<'a> {
    ordinal: usize,
    path: &'a FeffBinPath,
}

fn ff2x_listed_feff_paths<'a>(
    feff: &'a FeffBinData,
    list: &'a ListDatData,
) -> Result<Vec<Ff2xListedFeffPath<'a>>> {
    list.entries
        .iter()
        .map(|entry| {
            let (index, path) = feff
                .paths
                .iter()
                .enumerate()
                .find(|(_, path)| path.index == entry.path_index)
                .with_context(|| {
                    format!(
                        "FF2X feffNNNN output could not find list.dat path {} in feff.bin",
                        entry.path_index
                    )
                })?;
            Ok(Ff2xListedFeffPath {
                ordinal: index + 1,
                path,
            })
        })
        .collect()
}

fn write_ff2x_files_dat(
    path: &Path,
    xsect: &XsectFf2xHandoff,
    paths: &[Ff2xListedFeffPath<'_>],
) -> Result<()> {
    let mut out = String::new();
    for title in &xsect.titles {
        writeln!(out, "# {}", title.trim_end())?;
    }
    writeln!(
        out,
        " -----------------------------------------------------------------------"
    )?;
    writeln!(
        out,
        "    file        sig2   amp ratio    deg    nlegs  r effective"
    )?;
    for selected in paths {
        writeln!(
            out,
            " {:<12}{:>8.5}{:>10.3}{:>10.3}{:>6}{:>9.4}",
            format!("feff{:04}.dat", selected.path.index),
            0.0,
            selected.path.criterion,
            selected.path.degeneracy,
            selected.path.leg_count(),
            selected.path.effective_half_path_length_bohr * FEFF_BOHR_ANGSTROM
        )?;
    }

    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
}

fn ff2x_feff_path_data(
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    selected: &Ff2xListedFeffPath<'_>,
) -> Result<SfconvSo2convFeffPathData> {
    let point_count = xsect.main_energy_count;
    ff2x_validate_feff_path_data_inputs(feff, selected.path, point_count)?;

    let path = selected.path;
    let reff = path.effective_half_path_length_bohr;
    let initial_phase = f64::from(feff.initial_angular_momentum) * PI;
    let mut wave_number_inverse_angstrom = Array1::<Real>::zeros(point_count);
    let mut central_phase = Array1::<Real>::zeros(point_count);
    let mut effective_amplitude = Array1::<Real>::zeros(point_count);
    let mut effective_phase = Array1::<Real>::zeros(point_count);
    let mut reduction_factor = Array1::<Real>::zeros(point_count);
    let mut mean_free_path_angstrom = Array1::<Real>::zeros(point_count);
    let mut real_momentum_inverse_angstrom = Array1::<Real>::zeros(point_count);

    for row in 0..point_count {
        let cchi =
            Complex64::new(path.phase[row].cos(), path.phase[row].sin()) * path.amplitude[row];
        let complex_momentum = feff.complex_momentum[row];
        let mean_free_path = if complex_momentum.im.abs() > FF2X_EPS {
            1.0 / complex_momentum.im
        } else {
            1.0e10
        };
        let redfac = (-2.0 * feff.central_phase_shift[row].im).exp();
        let central_phase_raw = 2.0 * feff.central_phase_shift[row].re;
        let central_phase_unwrapped = if row > 0 {
            remove_phase_jump(central_phase_raw, central_phase[row - 1] - initial_phase)
                .context("FF2X feffNNNN central phase jump removal")?
        } else {
            central_phase_raw
        };
        let effective_signal =
            cchi * feff.real_momentum[row] * reff.powi(2) * (2.0 * reff / mean_free_path).exp()
                / redfac;
        let effective_phase_raw = if cchi.norm() < FF2X_EPS {
            0.0
        } else {
            cchi.im.atan2(cchi.re)
        };
        let effective_phase_unwrapped = if row > 0 {
            remove_phase_jump(
                effective_phase_raw,
                effective_phase[row - 1] + central_phase[row - 1],
            )
            .context("FF2X feffNNNN effective phase jump removal")?
        } else {
            effective_phase_raw
        };

        let values = [
            feff.real_momentum[row] / FEFF_BOHR_ANGSTROM,
            central_phase_unwrapped + initial_phase,
            effective_signal.norm() * FEFF_BOHR_ANGSTROM,
            effective_phase_unwrapped - central_phase_unwrapped - initial_phase,
            redfac,
            mean_free_path * FEFF_BOHR_ANGSTROM,
            complex_momentum.re / FEFF_BOHR_ANGSTROM,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            bail!(
                "FF2X feffNNNN output path {} row {row} contains non-finite values",
                path.index
            );
        }

        wave_number_inverse_angstrom[row] = values[0];
        central_phase[row] = values[1];
        effective_amplitude[row] = values[2];
        effective_phase[row] = values[3];
        reduction_factor[row] = values[4];
        mean_free_path_angstrom[row] = values[5];
        real_momentum_inverse_angstrom[row] = values[6];
    }

    Ok(SfconvSo2convFeffPathData {
        header_lines: ff2x_feff_path_header_lines(feff, xsect, selected)?,
        leg_count: path.leg_count(),
        degeneracy: path.degeneracy,
        effective_half_path_length_angstrom: path.effective_half_path_length_bohr
            * FEFF_BOHR_ANGSTROM,
        wave_number_inverse_angstrom,
        central_phase,
        effective_amplitude,
        effective_phase,
        reduction_factor,
        mean_free_path_angstrom,
        real_momentum_inverse_angstrom,
    })
}

fn ff2x_validate_feff_path_data_inputs(
    feff: &FeffBinData,
    path: &FeffBinPath,
    point_count: usize,
) -> Result<()> {
    if point_count == 0 {
        bail!("FF2X feffNNNN output requires at least one energy point");
    }
    if feff.energy_count() < point_count {
        bail!(
            "FF2X feffNNNN output needs {point_count} FEFF energy points, got {}",
            feff.energy_count()
        );
    }
    if path.amplitude.len() < point_count || path.phase.len() < point_count {
        bail!(
            "FF2X feffNNNN output path {} has {} amplitude and {} phase points for {point_count} rows",
            path.index,
            path.amplitude.len(),
            path.phase.len()
        );
    }
    let (position_legs, position_axes) = path.positions.dim();
    if position_legs != path.leg_count() || position_axes != 3 {
        bail!(
            "FF2X feffNNNN output path {} positions shape is {position_legs}x{position_axes}, expected {}x3",
            path.index,
            path.leg_count()
        );
    }
    Ok(())
}

fn ff2x_feff_path_header_lines(
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    selected: &Ff2xListedFeffPath<'_>,
) -> Result<Vec<String>> {
    let path = selected.path;
    let mut lines = Vec::new();
    for title in &xsect.titles {
        lines.push(format!("# {}", title.trim_end()));
    }
    lines.push(format!(
        " Path{:5}      icalc {:7}",
        selected.ordinal, feff.order
    ));
    lines.push(
        " -----------------------------------------------------------------------".to_string(),
    );
    lines.push(format!(
        " {:>3}{:>8.3}{:>9.4}{:>10.4}{:>11.5} nleg, deg, reff, rnrmav(bohr), edge",
        path.leg_count(),
        path.degeneracy,
        path.effective_half_path_length_bohr * FEFF_BOHR_ANGSTROM,
        feff.average_norman_radius,
        feff.edge_energy * FEFF_HARTREE_EV
    ));
    lines.push("        x         y         z   pot at#".to_string());
    let absorbing_leg = path.leg_count() - 1;
    lines.push(ff2x_feff_path_leg_header_line(
        feff,
        path,
        absorbing_leg,
        true,
    )?);
    for leg in 0..absorbing_leg {
        lines.push(ff2x_feff_path_leg_header_line(feff, path, leg, false)?);
    }
    lines.push(
        "    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#"
            .to_string(),
    );
    Ok(lines)
}

fn ff2x_feff_path_leg_header_line(
    feff: &FeffBinData,
    path: &FeffBinPath,
    leg: usize,
    absorbing_atom: bool,
) -> Result<String> {
    let potential_index = path.potential_indices[leg];
    let potential = feff.potentials.get(potential_index).with_context(|| {
        format!(
            "FF2X feffNNNN output path {} leg {} references missing potential {}",
            path.index,
            leg + 1,
            potential_index
        )
    })?;
    let label = ff2x_potential_label(&potential.label);
    let suffix = if absorbing_atom {
        "   absorbing atom"
    } else {
        ""
    };
    Ok(format!(
        " {:>10.4}{:>10.4}{:>10.4}{:>3}{:>4} {:<6}{suffix}",
        path.positions[(leg, 0)] * FEFF_BOHR_ANGSTROM,
        path.positions[(leg, 1)] * FEFF_BOHR_ANGSTROM,
        path.positions[(leg, 2)] * FEFF_BOHR_ANGSTROM,
        potential_index,
        potential.atomic_number,
        label
    ))
}

fn ff2x_potential_label(label: &str) -> String {
    label.chars().take(6).collect()
}

fn ff2x_output_energy_grid_for_input(
    input: &Ff2xInput,
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<Ff2xOutputEnergyGrid> {
    ff2x_output_energy_grid(
        ff2x_shifted_output_edge_hartree(input, feff)?,
        xsect,
        momentum_grid,
    )
}

fn ff2x_shifted_output_edge_hartree(input: &Ff2xInput, feff: &FeffBinData) -> Result<Real> {
    if !feff.edge_energy.is_finite() {
        bail!(
            "FF2X output edge energy from feff.bin is not finite: {}",
            feff.edge_energy
        );
    }
    let real_correction = ff2x_real_correction_hartree(input);
    if !real_correction.is_finite() {
        bail!("FF2X output real correction is not finite: {real_correction}");
    }
    if real_correction.abs() > FF2X_EPS4 {
        Ok(feff.edge_energy - real_correction)
    } else {
        Ok(feff.edge_energy)
    }
}

fn ff2x_output_energy_grid(
    edge_hartree: Real,
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<Ff2xOutputEnergyGrid> {
    if !edge_hartree.is_finite() {
        bail!("FF2X output energy grid got non-finite edge energy: {edge_hartree}");
    }
    if xsect.energy_grid_hartree.is_empty() {
        bail!("FF2X output energy grid requires at least one xsect.dat energy point");
    }
    if xsect.energy_grid_hartree.len() != xsect.omega_hartree.len() {
        bail!(
            "FF2X xsect.dat energy length {} does not match omega length {}",
            xsect.energy_grid_hartree.len(),
            xsect.omega_hartree.len()
        );
    }

    let first_energy = xsect.energy_grid_hartree[0].re;
    let first_omega = xsect.omega_hartree[0];
    if !(first_energy.is_finite() && first_omega.is_finite()) {
        bail!(
            "FF2X output energy grid got non-finite first xsect point: energy={first_energy}, omega={first_omega}"
        );
    }
    let fermi_energy_hartree = edge_hartree + first_omega - first_energy;

    let mut photon_energy_hartree = Array1::<Real>::zeros(momentum_grid.output_momentum.len());
    let mut relative_energy_hartree = Array1::<Real>::zeros(momentum_grid.output_momentum.len());
    for (row, &momentum) in momentum_grid.output_momentum.iter().enumerate() {
        if !momentum.is_finite() {
            bail!("FF2X output momentum {row} is not finite: {momentum}");
        }
        let kinetic_energy = if momentum < 0.0 {
            -momentum.powi(2) / 2.0
        } else {
            momentum.powi(2) / 2.0
        };
        photon_energy_hartree[row] = kinetic_energy + fermi_energy_hartree;
        relative_energy_hartree[row] =
            photon_energy_hartree[row] - fermi_energy_hartree + edge_hartree;
    }

    Ok(Ff2xOutputEnergyGrid {
        fermi_energy_hartree,
        photon_energy_hartree,
        relative_energy_hartree,
    })
}

pub(crate) fn ff2x_atomic_xscorr_with_background(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    background: ArrayView1<'_, Real>,
) -> Result<Ff2xXscorrResult> {
    let zero_path_chi = Array1::<Complex64>::zeros(xsect.energy_count());
    if background.len() != xsect.energy_count() {
        bail!(
            "FF2X atomic xscorr background length {} does not match xsect energy count {}",
            background.len(),
            xsect.energy_count()
        );
    }
    let cchi = ff2x_xscorr(Ff2xXscorrInput {
        ispec: input.control.ispec,
        energy_grid_hartree: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        fermi_index: xsect.fermi_index,
        cross_section: xsect.cross_section.view(),
        background,
        path_chi: zero_path_chi.view(),
        real_correction_hartree: ff2x_real_correction_hartree(input),
        electronic_temperature_ev: 0.0,
    })?;
    let mut corrected_atomic_cross_section = xsect.cross_section.mapv(|value| value.im);
    for row in 0..xsect.main_energy_count {
        corrected_atomic_cross_section[row] = (xsect.cross_section[row] + cchi[row]).im;
    }

    Ok(Ff2xXscorrResult {
        cchi,
        corrected_atomic_cross_section,
    })
}

fn ff2x_xscorr(input: Ff2xXscorrInput<'_>) -> Result<Array1<Complex64>> {
    if input.electronic_temperature_ev > 0.0 {
        return ff2x_thermal_xscorr(input);
    }
    let Ff2xXscorrInput {
        ispec,
        energy_grid_hartree,
        main_energy_count,
        fermi_index,
        cross_section,
        background,
        path_chi,
        real_correction_hartree,
        electronic_temperature_ev,
    } = input;
    if !(electronic_temperature_ev.is_finite() && electronic_temperature_ev >= 0.0) {
        bail!(
            "FF2X xscorr electronic temperature must be finite and nonnegative, got {electronic_temperature_ev}"
        );
    }
    let energy_count = energy_grid_hartree.len();
    if main_energy_count < 2 {
        bail!("FF2X xscorr requires at least two horizontal energy points");
    }
    if main_energy_count >= energy_count {
        bail!(
            "FF2X xscorr requires contour points after the horizontal grid: ne1={}, ne={}",
            main_energy_count,
            energy_count
        );
    }
    if fermi_index >= main_energy_count {
        bail!(
            "FF2X xscorr Fermi index {} is outside horizontal energy count {}",
            fermi_index,
            main_energy_count
        );
    }
    if cross_section.len() != energy_count
        || background.len() != energy_count
        || path_chi.len() != energy_count
    {
        bail!(
            "FF2X xscorr input lengths do not match energy count {}: xsec={}, xsnorm={}, chia={}",
            energy_count,
            cross_section.len(),
            background.len(),
            path_chi.len()
        );
    }
    if !real_correction_hartree.is_finite() {
        bail!("FF2X xscorr real correction is not finite: {real_correction_hartree}");
    }

    let mut energy = energy_grid_hartree.to_owned();
    let mut xmu = Array1::<Complex64>::zeros(energy_count);
    for row in 0..energy_count {
        let energy_value = energy[row];
        let xsec = cross_section[row];
        let xsnorm = background[row];
        let chia = path_chi[row];
        if !(energy_value.re.is_finite()
            && energy_value.im.is_finite()
            && xsec.re.is_finite()
            && xsec.im.is_finite()
            && xsnorm.is_finite()
            && chia.re.is_finite()
            && chia.im.is_finite())
        {
            bail!("FF2X xscorr input row {row} contains non-finite values");
        }
        xmu[row] = xsec + chia * xsnorm;
    }

    let xloss = energy[0].im;
    if !(xloss.is_finite() && xloss > 0.0) {
        bail!("FF2X xscorr requires positive finite xloss, got {xloss}");
    }
    let mut fermi_energy = energy[energy_count - 1].re;
    if !fermi_energy.is_finite() {
        bail!("FF2X xscorr Fermi energy is not finite: {fermi_energy}");
    }
    let omega = energy
        .iter()
        .take(main_energy_count)
        .map(|energy| energy.re)
        .collect::<Vec<_>>();

    let fermi_scale = if real_correction_hartree.abs() > FF2X_EPS4 {
        let fermi_xmu = xmu[fermi_index];
        if fermi_xmu.norm() == 0.0 {
            bail!("FF2X xscorr cannot rescale vertical contour from zero Fermi xmu");
        }
        fermi_energy -= real_correction_hartree;
        let horizontal_xmu = xmu
            .iter()
            .take(main_energy_count)
            .copied()
            .collect::<Vec<_>>();
        let shifted_fermi_xmu = terpc(&omega, &horizontal_xmu, 1, fermi_energy)
            .context("FF2X xscorr real-correction interpolation")?
            .value;
        for row in main_energy_count..energy_count {
            energy[row] -= Complex64::new(real_correction_hartree, 0.0);
        }
        let scale = shifted_fermi_xmu / fermi_xmu;
        for row in main_energy_count..energy_count {
            xmu[row] *= scale;
        }
        scale
    } else {
        Complex64::new(1.0, 0.0)
    };

    let mut contour_energy = Vec::new();
    let mut contour_value = Vec::new();
    for row in main_energy_count..energy_count {
        if energy[row].im < xloss {
            contour_energy.push(energy[row]);
            contour_value.push(xmu[row]);
        }
    }
    if real_correction_hartree.abs() > FF2X_EPS4 {
        contour_energy.push(Complex64::new(fermi_energy, xloss));
        contour_value.push(fermi_scale * xmu[fermi_index]);
    } else {
        contour_energy.push(energy[fermi_index]);
        contour_value.push(xmu[fermi_index]);
    }
    if ispec != 2 {
        for row in 0..main_energy_count {
            if energy[row].re - fermi_energy > FF2X_EPS4 {
                contour_energy.push(energy[row]);
                contour_value.push(xmu[row]);
            }
        }
    } else {
        for row in (0..main_energy_count).rev() {
            if fermi_energy - energy[row].re > FF2X_EPS4 {
                contour_energy.push(energy[row]);
                contour_value.push(xmu[row]);
            }
        }
    }
    if contour_energy.len() < 2 {
        bail!(
            "FF2X xscorr contour requires at least two points, got {}",
            contour_energy.len()
        );
    }

    let mut cchi = Array1::<Complex64>::zeros(main_energy_count);
    for row in 0..main_energy_count {
        let omega_row = omega[row];
        let xmu0 = if omega_row >= fermi_energy {
            if ispec == 2 {
                xmu[fermi_index] * fermi_scale
            } else {
                xmu[row]
            }
        } else if ispec == 2 {
            xmu[row]
        } else {
            xmu[fermi_index] * fermi_scale
        };
        let e1 = Complex64::new(omega_row, xloss);
        let e2 = Complex64::new(omega_row, -xloss);
        let dele = omega_row - fermi_energy;
        let step = ff2x_xscorr_astep(xloss, dele)?;
        let mut convolution = xmu0 * step;
        if ispec == 2 {
            convolution = xmu0 - convolution;
        }

        let dele_for_lorenz = if dele.abs() < FF2X_EPS4 { 0.0 } else { dele };
        let mut correction = ff2x_xscorr_lorenz(xloss, contour_energy[0].im, dele_for_lorenz)?
            * (contour_value[0] - xmu0)
            * Complex64::new(0.0, contour_energy[0].im);

        for segment in 0..(contour_energy.len() - 1) {
            let z1 = contour_energy[segment];
            let z2 = contour_energy[segment + 1];
            let f1 = contour_value[segment] - xmu0;
            let f2 = contour_value[segment + 1] - xmu0;
            let dz = z2 - z1;
            if dz.norm() == 0.0 {
                bail!("FF2X xscorr contour has duplicate segment point {segment}");
            }

            let mut segment_integral = Complex64::new(0.0, 0.0);
            if (z1 - e1).norm() > FF2X_EPS4 && (z2 - e1).norm() > FF2X_EPS4 {
                segment_integral = ((z2 - e1) / (z1 - e1)).ln() * (f1 * (z2 - e1) + f2 * (e1 - z1));
            }
            segment_integral -= ((z2 - e2) / (z1 - e2)).ln() * (f1 * (z2 - e2) + f2 * (e2 - z1));
            correction += segment_integral / dz / Complex64::new(0.0, 2.0 * std::f64::consts::PI);
        }

        if ispec == 2 {
            correction = -correction;
        }
        cchi[row] = convolution + correction - xmu[row];
    }

    Ok(cchi)
}

fn ff2x_thermal_xscorr(input: Ff2xXscorrInput<'_>) -> Result<Array1<Complex64>> {
    let Ff2xXscorrInput {
        ispec,
        energy_grid_hartree,
        main_energy_count,
        fermi_index: _,
        cross_section,
        background,
        path_chi,
        real_correction_hartree,
        electronic_temperature_ev,
    } = input;
    let energy_count = energy_grid_hartree.len();
    if main_energy_count < 2 {
        bail!("FF2X thermal xscorr requires at least two horizontal energy points");
    }
    if cross_section.len() != energy_count
        || background.len() != energy_count
        || path_chi.len() != energy_count
    {
        bail!(
            "FF2X thermal xscorr input lengths do not match energy count {}: xsec={}, xsnorm={}, chia={}",
            energy_count,
            cross_section.len(),
            background.len(),
            path_chi.len()
        );
    }
    if !(electronic_temperature_ev.is_finite() && electronic_temperature_ev > 0.0) {
        bail!(
            "FF2X thermal xscorr electronic temperature must be positive and finite, got {electronic_temperature_ev}"
        );
    }
    if !real_correction_hartree.is_finite() {
        bail!("FF2X thermal xscorr real correction is not finite: {real_correction_hartree}");
    }

    let thermal_auxiliary_count = 10usize;
    let fermi_extra_count = 1usize;
    let pole_start = main_energy_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(thermal_auxiliary_count))
        .context("FF2X thermal xscorr mesh size overflowed")?;
    let minimum_count = pole_start
        .checked_add(fermi_extra_count)
        .context("FF2X thermal xscorr minimum mesh size overflowed")?;
    if energy_count < minimum_count {
        bail!(
            "FF2X thermal xscorr requires at least {minimum_count} xsect.dat rows for ne1={} thermal contour layout, got {energy_count}",
            main_energy_count
        );
    }

    let mut xmu = Array1::<Complex64>::zeros(energy_count);
    for row in 0..energy_count {
        let energy = energy_grid_hartree[row];
        let xsec = cross_section[row];
        let xsnorm = background[row];
        let chia = path_chi[row];
        if !(energy.re.is_finite()
            && energy.im.is_finite()
            && xsec.re.is_finite()
            && xsec.im.is_finite()
            && xsnorm.is_finite()
            && chia.re.is_finite()
            && chia.im.is_finite())
        {
            bail!("FF2X thermal xscorr input row {row} contains non-finite values");
        }
        xmu[row] = xsec + chia * xsnorm;
    }

    let xloss = energy_grid_hartree[main_energy_count].im;
    if !(xloss.is_finite() && xloss > 0.0) {
        bail!("FF2X thermal xscorr requires positive finite loss width, got {xloss}");
    }
    let mut fermi_energy = energy_grid_hartree[energy_count - fermi_extra_count].re;
    if !fermi_energy.is_finite() {
        bail!("FF2X thermal xscorr Fermi energy is not finite: {fermi_energy}");
    }
    if real_correction_hartree.abs() > FF2X_EPS4 {
        fermi_energy -= real_correction_hartree;
    }

    let temperature_hartree = electronic_temperature_ev / FEFF_HARTREE_EV;
    if !(temperature_hartree.is_finite() && temperature_hartree > 0.0) {
        bail!(
            "FF2X thermal xscorr temperature in Hartree must be positive and finite, got {temperature_hartree}"
        );
    }

    let horizontal_energy = energy_grid_hartree
        .iter()
        .take(main_energy_count)
        .copied()
        .collect::<Vec<_>>();
    let horizontal_xmu = xmu
        .iter()
        .take(main_energy_count)
        .copied()
        .collect::<Vec<_>>();
    let loss_xmu = xmu
        .iter()
        .skip(main_energy_count)
        .take(main_energy_count)
        .copied()
        .collect::<Vec<_>>();

    let pole_end = energy_count - fermi_extra_count;
    let mut cchi = Array1::<Complex64>::zeros(main_energy_count);
    for row in 0..main_energy_count {
        let omega = energy_grid_hartree[row].re;
        let xmu0 = loss_xmu[row];

        let fermi_lorentz =
            ff2x_thermal_fermi_lorentz_integral(omega, temperature_hartree, fermi_energy, xloss)?;
        let mut convolution = xmu0 * fermi_lorentz;
        if ispec != 2 {
            convolution = xmu0 - convolution;
        }

        let mut residue = Complex64::new(0.0, 0.0);
        for pole_row in pole_start..pole_end {
            let mut pole = energy_grid_hartree[pole_row];
            if real_correction_hartree.abs() > FF2X_EPS4 {
                pole -= Complex64::new(real_correction_hartree, 0.0);
            }
            let numerator = xmu[pole_row] - xmu0;
            residue += numerator * xloss
                / ((pole - Complex64::new(omega, 0.0)).powi(2)
                    + Complex64::new(xloss * xloss, 0.0));
        }
        residue *= Complex64::new(0.0, -2.0 * temperature_hartree);
        if ispec != 2 {
            residue = -residue;
        }

        let contour = ff2x_thermal_cchi_eff(
            &horizontal_energy,
            &horizontal_xmu,
            fermi_energy,
            temperature_hartree,
            xmu0,
            xloss,
            omega,
            ispec,
        )?;
        cchi[row] = convolution + contour + residue - horizontal_xmu[row];
        if !(cchi[row].re.is_finite() && cchi[row].im.is_finite()) {
            bail!(
                "FF2X thermal xscorr row {row} produced non-finite cchi {:?}",
                cchi[row]
            );
        }
    }

    Ok(cchi)
}

fn ff2x_thermal_fermi_lorentz_integral(
    omega: Real,
    temperature_hartree: Real,
    fermi_energy: Real,
    xloss: Real,
) -> Result<Real> {
    if !(omega.is_finite()
        && temperature_hartree.is_finite()
        && fermi_energy.is_finite()
        && xloss.is_finite())
    {
        bail!("FF2X thermal fermi-lorentz integral got non-finite input");
    }
    if temperature_hartree <= 0.0 || xloss <= 0.0 {
        bail!(
            "FF2X thermal fermi-lorentz integral requires positive temperature and loss, got T={temperature_hartree}, loss={xloss}"
        );
    }

    let analytic_count = 20_000usize;
    let omega_max = 1.0;
    let domega = omega_max / (analytic_count as Real - 1.0);
    let mut result = 0.0;

    for index in 0..(analytic_count - 1) {
        let z = 1.0 / (-omega_max + index as Real * domega);
        if (z - fermi_energy) / temperature_hartree < 100.0 {
            let fermi = ff2x_thermal_fermi_real(z, temperature_hartree, fermi_energy);
            result += fermi * ff2x_thermal_cauchy_real(z, omega, xloss) * z.powi(2);
        }
    }

    for index in 0..(analytic_count - 1) {
        let negative = -omega_max + index as Real * domega;
        let fermi = ff2x_thermal_fermi_real(negative, temperature_hartree, fermi_energy);
        result += fermi * ff2x_thermal_cauchy_real(negative, omega, xloss);

        let positive = index as Real * domega;
        let fermi = ff2x_thermal_fermi_real(positive, temperature_hartree, fermi_energy);
        result += fermi * ff2x_thermal_cauchy_real(positive, omega, xloss);
    }

    for index in (2..=analytic_count).rev() {
        let z = 1.0 / (omega_max - (analytic_count - index) as Real * domega);
        if (z - fermi_energy) / temperature_hartree < 100.0 {
            let fermi = ff2x_thermal_fermi_real(z, temperature_hartree, fermi_energy);
            result += fermi * ff2x_thermal_cauchy_real(z, omega, xloss) * z.powi(2);
        }
    }

    result *= domega;
    if result.is_finite() {
        Ok(result)
    } else {
        bail!("FF2X thermal fermi-lorentz integral produced non-finite value {result}");
    }
}

#[allow(clippy::too_many_arguments)]
fn ff2x_thermal_cchi_eff(
    energy: &[Complex64],
    xmu: &[Complex64],
    fermi_energy: Real,
    temperature_hartree: Real,
    xmu0: Complex64,
    xloss: Real,
    omega: Real,
    ispec: i32,
) -> Result<Complex64> {
    if energy.len() != xmu.len() {
        bail!(
            "FF2X thermal contour interpolation length mismatch: energy={}, xmu={}",
            energy.len(),
            xmu.len()
        );
    }
    if energy.len() < 2 {
        bail!("FF2X thermal contour interpolation requires at least two points");
    }
    if temperature_hartree <= 0.0 {
        bail!("FF2X thermal contour interpolation requires positive temperature");
    }
    let first_energy = energy[0].re;
    let last_energy = energy[energy.len() - 1].re;
    let mut window_size = 10.0;
    if fermi_energy > first_energy {
        window_size = ((fermi_energy - first_energy) / temperature_hartree)
            .floor()
            .min(10.0);
    }
    if !(window_size.is_finite() && window_size > 0.0) {
        bail!(
            "FF2X thermal contour interpolation window collapsed: fermi={fermi_energy}, first={first_energy}, T={temperature_hartree}"
        );
    }

    let interpolation_count = 10_000usize;
    let defermi = 2.0 * window_size * temperature_hartree / interpolation_count as Real;
    let fermi_low = fermi_energy - window_size * temperature_hartree;
    let fermi_high = fermi_energy + window_size * temperature_hartree;
    if fermi_low < first_energy || fermi_high > last_energy {
        bail!(
            "FF2X thermal contour interpolation window [{fermi_low}, {fermi_high}] is outside horizontal grid [{first_energy}, {last_energy}]"
        );
    }

    let real_energy = energy.iter().map(|value| value.re).collect::<Vec<_>>();
    let search_low = ff2x_thermal_binary_search_1based(fermi_low, &real_energy)?;
    let search_high = ff2x_thermal_binary_search_1based(fermi_high, &real_energy)?;
    let ind_low = search_low.saturating_sub(1).max(1);
    let ind_high = search_high.min(energy.len());
    let interpolated_len = ind_low
        .checked_add(interpolation_count)
        .and_then(|count| count.checked_add(energy.len() - ind_high + 1))
        .context("FF2X thermal contour interpolation length overflowed")?;

    let mut interp_energy = Vec::with_capacity(interpolated_len);
    let mut interp_xmu = Vec::with_capacity(interpolated_len);
    for index in 0..ind_low {
        interp_energy.push(energy[index]);
        interp_xmu.push(xmu[index]);
    }
    for index in 0..interpolation_count {
        let real = fermi_low + index as Real * defermi;
        let value = Complex64::new(real, energy[0].im);
        interp_energy.push(value);
        interp_xmu.push(ff2x_thermal_interp1d(real, &real_energy, xmu)?);
    }
    interp_energy.push(Complex64::new(fermi_high, energy[0].im));
    interp_xmu.push(ff2x_thermal_interp1d(fermi_high, &real_energy, xmu)?);
    for index in ind_high..energy.len() {
        interp_energy.push(energy[index]);
        interp_xmu.push(xmu[index]);
    }

    let e1 = Complex64::new(omega, xloss);
    let e2 = Complex64::new(omega, -xloss);
    let mut result = Complex64::new(0.0, 0.0);
    for segment in 0..(interp_energy.len() - 1) {
        let z1 = interp_energy[segment];
        let z2 = interp_energy[segment + 1];
        let k1 = interp_xmu[segment] - xmu0;
        let k2 = interp_xmu[segment + 1] - xmu0;
        let mut f1 = ff2x_thermal_fermi_real(z1.re, temperature_hartree, fermi_energy);
        let mut f2 = ff2x_thermal_fermi_real(z2.re, temperature_hartree, fermi_energy);
        if ispec != 2 {
            f1 = 1.0 - f1;
            f2 = 1.0 - f2;
        }
        let dz = z2 - z1;
        if dz.norm() == 0.0 {
            bail!("FF2X thermal contour interpolation has duplicate segment point {segment}");
        }

        if z1.re > fermi_low && z1.re < fermi_high {
            result += 0.5
                * dz
                * (k1 * f1 * ff2x_thermal_cauchy(z1, omega, xloss)
                    + k2 * f2 * ff2x_thermal_cauchy(z2, omega, xloss));
        } else {
            let mut segment_integral = Complex64::new(0.0, 0.0);
            if (z1 - e1).norm() > FF2X_EPS4 && (z2 - e1).norm() > FF2X_EPS4 {
                segment_integral =
                    -((z2 - e1) / (z1 - e1)).ln() * (k1 * f1 * (z2 - e1) + k2 * f2 * (e1 - z1));
            }
            segment_integral +=
                ((z2 - e2) / (z1 - e2)).ln() * (k1 * f1 * (z2 - e2) + k2 * f2 * (e2 - z1));
            result -= segment_integral / dz / Complex64::new(0.0, 2.0 * PI);
        }
    }

    if result.re.is_finite() && result.im.is_finite() {
        Ok(result)
    } else {
        bail!("FF2X thermal contour interpolation produced non-finite value {result:?}");
    }
}

fn ff2x_thermal_interp1d(x0: Real, x: &[Real], y: &[Complex64]) -> Result<Complex64> {
    if x.len() != y.len() || x.len() < 2 {
        bail!(
            "FF2X thermal interp1d needs matching arrays of at least two points, got x={}, y={}",
            x.len(),
            y.len()
        );
    }
    if x[0] < x0 && x0 < x[x.len() - 1] {
        let idx_1based = ff2x_thermal_binary_search_1based(x0, x)?;
        let upper = idx_1based
            .checked_sub(1)
            .context("FF2X thermal interp1d got invalid upper index")?;
        let lower = upper
            .checked_sub(1)
            .context("FF2X thermal interp1d got invalid lower index")?;
        let x1 = x[lower];
        let x2 = x[upper];
        if x2 == x1 {
            bail!("FF2X thermal interp1d got duplicate source x value {x1}");
        }
        Ok((y[upper] - y[lower]) * ((x0 - x1) / (x2 - x1)) + y[lower])
    } else if x[0] == x0 {
        Ok(y[0])
    } else if (x[x.len() - 1] - x0).abs() < 1.0e-10 {
        Ok(y[y.len() - 1])
    } else {
        bail!(
            "FF2X thermal interp1d point {x0} is outside [{}, {}]",
            x[0],
            x[x.len() - 1]
        );
    }
}

fn ff2x_thermal_binary_search_1based(x0: Real, x: &[Real]) -> Result<usize> {
    if x.is_empty() {
        bail!("FF2X thermal binary search requires a non-empty grid");
    }
    if x0 >= x[x.len() - 1] {
        return Ok(x.len() + 1);
    }
    if x0 <= x[0] {
        return Ok(1);
    }
    let index = x.partition_point(|value| *value < x0);
    Ok(index + 1)
}

fn ff2x_thermal_fermi_real(energy: Real, temperature_hartree: Real, fermi_energy: Real) -> Real {
    if temperature_hartree > 0.0 {
        let reduced = (energy - fermi_energy) / temperature_hartree;
        if reduced < 100.0 {
            1.0 / (reduced.exp() + 1.0)
        } else {
            0.0
        }
    } else if energy > fermi_energy {
        1.0
    } else {
        0.0
    }
}

fn ff2x_thermal_cauchy(x: Complex64, omega: Real, xloss: Real) -> Complex64 {
    (xloss / PI) / ((x - Complex64::new(omega, 0.0)).powi(2) + Complex64::new(xloss * xloss, 0.0))
}

fn ff2x_thermal_cauchy_real(energy: Real, omega: Real, xloss: Real) -> Real {
    (xloss / PI) / ((energy - omega).powi(2) + xloss.powi(2))
}

fn ff2x_xscorr_lorenz(xloss: Real, width: Real, delta_energy: Real) -> Result<Complex64> {
    if !(xloss.is_finite() && width.is_finite() && delta_energy.is_finite()) {
        bail!(
            "FF2X xscorr lorenz got non-finite inputs: xloss={xloss}, width={width}, delta={delta_energy}"
        );
    }
    let denominator = Complex64::new(
        xloss.powi(2) - width.powi(2) + delta_energy.powi(2),
        -2.0 * width * delta_energy,
    );
    if denominator.norm() == 0.0 {
        bail!("FF2X xscorr lorenz denominator is zero");
    }
    Ok((xloss / std::f64::consts::PI) / denominator)
}

fn ff2x_xscorr_astep(xloss: Real, delta_energy: Real) -> Result<Real> {
    if !(xloss.is_finite() && delta_energy.is_finite()) {
        bail!("FF2X xscorr astep got non-finite inputs: xloss={xloss}, delta={delta_energy}");
    }
    if xloss == 0.0 {
        bail!("FF2X xscorr astep requires non-zero xloss");
    }
    Ok((0.5 + (delta_energy / xloss).atan() / std::f64::consts::PI).clamp(0.0, 1.0))
}

fn ff2x_xmu_dat_from_components(components: Ff2xXmuComponents<'_>) -> Result<XmuDatData> {
    let Ff2xXmuComponents {
        input,
        xsect,
        momentum_grid,
        output_energy,
        path_sum,
        path_chi,
        corrected_background,
        corrected_atomic_cross_section,
        pre_table_header_lines,
        used_path_count,
        total_path_count,
    } = components;
    let main_energy_count = xsect.main_energy_count;
    if main_energy_count < 2 {
        bail!(
            "FF2X xmu.dat row build requires at least two main xsect.dat points, got {main_energy_count}"
        );
    }
    if main_energy_count > xsect.energy_count() {
        bail!(
            "FF2X xsect.dat main energy count {} exceeds total energy count {}",
            main_energy_count,
            xsect.energy_count()
        );
    }
    if corrected_background.len() != xsect.energy_count() {
        bail!(
            "FF2X corrected background length {} does not match xsect energy count {}",
            corrected_background.len(),
            xsect.energy_count()
        );
    }
    if corrected_atomic_cross_section.len() != xsect.energy_count() {
        bail!(
            "FF2X corrected atomic cross-section length {} does not match xsect energy count {}",
            corrected_atomic_cross_section.len(),
            xsect.energy_count()
        );
    }
    let output_count = momentum_grid.output_momentum.len();
    if momentum_grid.interpolation_momentum.len() != output_count {
        bail!(
            "FF2X xmu.dat row build got {} interpolation momenta for {} output momenta",
            momentum_grid.interpolation_momentum.len(),
            output_count
        );
    }
    if output_energy.photon_energy_hartree.len() != output_count
        || output_energy.relative_energy_hartree.len() != output_count
    {
        bail!(
            "FF2X output energy grid length does not match output momentum length {}",
            output_count
        );
    }
    if path_sum.total.len() != output_count {
        bail!(
            "FF2X xmu.dat row build got {} path-sum points for {} output momenta",
            path_sum.total.len(),
            output_count
        );
    }
    if path_chi.len() != output_count {
        bail!(
            "FF2X xmu.dat row build got {} path chi points for {} output momenta",
            path_chi.len(),
            output_count
        );
    }

    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("FF2X xsect.dat omega grid is not contiguous")?;
    let wave_number = xsect
        .wave_number
        .as_slice()
        .context("FF2X xsect.dat wave-number grid is not contiguous")?;
    let background = corrected_background
        .as_slice()
        .context("FF2X corrected background grid is not contiguous")?;
    let atomic_cross_section = corrected_atomic_cross_section
        .as_slice()
        .context("FF2X corrected atomic cross-section grid is not contiguous")?;
    let omega = &omega[..main_energy_count];
    let wave_number = &wave_number[..main_energy_count];
    let background = &background[..main_energy_count];
    let atomic_cross_section = &atomic_cross_section[..main_energy_count];

    let mut normalization = if input.control.absolu == 1 {
        1.0
    } else {
        let edge_plus_50 = output_energy.fermi_energy_hartree + 50.0 / FEFF_HARTREE_EV;
        terp(omega, background, 1, edge_plus_50)
            .context("FF2X xmu.dat xsedge interpolation")?
            .value
    };
    if !normalization.is_finite() {
        bail!("FF2X xmu.dat normalization is not finite: {normalization}");
    }
    if normalization == 0.0 {
        bail!("FF2X xmu.dat normalization is zero");
    }
    if input.control.absolu == 1 {
        normalization = 1.0;
    }

    let fermi_output_index = ff2x_fermi_output_index(momentum_grid);
    let mut photon_energy_ev = Array1::<Real>::zeros(output_count);
    let mut relative_energy_ev = Array1::<Real>::zeros(output_count);
    let mut output_wave_number = Array1::<Real>::zeros(output_count);
    let mut mu = Array1::<Real>::zeros(output_count);
    let mut mu0 = Array1::<Real>::zeros(output_count);
    let mut chi = Array1::<Real>::zeros(output_count);

    for row in 0..output_count {
        let interpolation_momentum = momentum_grid.interpolation_momentum[row];
        if !interpolation_momentum.is_finite() {
            bail!(
                "FF2X xmu.dat interpolation momentum {row} is not finite: {interpolation_momentum}"
            );
        }
        let xsec0 = terp(wave_number, atomic_cross_section, 1, interpolation_momentum)
            .with_context(|| format!("FF2X xmu.dat cross-section interpolation at row {row}"))?
            .value;
        let xsnor0 = terp(wave_number, background, 1, interpolation_momentum)
            .with_context(|| format!("FF2X xmu.dat background interpolation at row {row}"))?
            .value;
        let chi_source =
            if output_energy.photon_energy_hartree[row] >= output_energy.fermi_energy_hartree {
                row
            } else {
                fermi_output_index
            };
        let chi0 = xsnor0 * path_chi[chi_source];

        photon_energy_ev[row] = output_energy.photon_energy_hartree[row] * FEFF_HARTREE_EV;
        relative_energy_ev[row] = output_energy.relative_energy_hartree[row] * FEFF_HARTREE_EV;
        output_wave_number[row] = momentum_grid.output_momentum[row] / FEFF_BOHR_ANGSTROM;
        if input.control.ispec.abs() == 3 {
            mu[row] = -(xsec0 + chi0);
            mu0[row] = -xsec0;
            chi[row] = -chi0;
        } else {
            mu[row] = (chi0 + xsec0) / normalization;
            mu0[row] = xsec0 / normalization;
            chi[row] = path_chi[row];
        }
    }

    let mut header_lines = pre_table_header_lines.to_vec();
    header_lines.extend([
        format!("#  {used_path_count:4}/{total_path_count:4} paths used"),
        format!("# xsedge+50, used to normalize mu {normalization:20.4E}"),
        "# -----------------------------------------------------------------------".to_string(),
        "# omega    e    k    mu    mu0     chi     @#".to_string(),
    ]);

    Ok(XmuDatData {
        header_lines,
        normalization: Some(normalization),
        photon_energy_ev,
        relative_energy_ev,
        wave_number: output_wave_number,
        mu,
        mu0,
        chi,
    })
}

fn ff2x_path_sum_chi(input: &Ff2xInput, path_sum: &Ff2xPathSum) -> Result<Array1<Real>> {
    let mut chi = Array1::<Real>::zeros(path_sum.total.len());
    for (row, &value) in path_sum.total.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            bail!("FF2X path sum row {row} is not finite: {value:?}");
        }
        chi[row] = if input.control.ispec.abs() == 3 {
            value.re
        } else {
            value.im
        };
    }
    Ok(chi)
}

fn ff2x_fermi_output_index(momentum_grid: &Ff2xMomentumGrid) -> usize {
    momentum_grid
        .interpolation_momentum
        .iter()
        .position(|momentum| *momentum >= FF2X_EPS4)
        .map_or_else(
            || momentum_grid.interpolation_momentum.len().saturating_sub(1),
            |index| index.saturating_sub(1),
        )
}

#[cfg(test)]
fn ff2x_prepared_paths(
    input: &Ff2xInput,
    feff: &FeffBinData,
    list: &ListDatData,
) -> Result<Vec<Ff2xPreparedPath>> {
    ff2x_prepared_paths_with_imaginary_correction(
        None,
        input,
        feff,
        list,
        ff2x_imaginary_correction_hartree(input),
    )
}

fn ff2x_prepared_paths_with_imaginary_correction(
    work_dir: Option<&Path>,
    input: &Ff2xInput,
    feff: &FeffBinData,
    list: &ListDatData,
    imaginary_correction_hartree: Real,
) -> Result<Vec<Ff2xPreparedPath>> {
    let damping = ff2x_path_damping_in_dir(work_dir, input, feff, list)?;
    let mut prepared = Vec::with_capacity(damping.len());

    for path_damping in damping {
        let path = feff
            .paths
            .iter()
            .find(|path| path.index == path_damping.path_index)
            .with_context(|| {
                format!(
                    "FF2X list.dat path {} is missing from feff.bin",
                    path_damping.path_index
                )
            })?;
        let (amplitude, phase) = ff2x_damped_amplitude_phase(
            input,
            feff,
            path,
            &path_damping,
            imaginary_correction_hartree,
        )?;
        prepared.push(Ff2xPreparedPath {
            damping: path_damping,
            amplitude,
            phase,
        });
    }

    Ok(prepared)
}

fn ff2x_path_summary_header_lines(prepared: &[Ff2xPreparedPath]) -> Vec<String> {
    prepared
        .iter()
        .map(|path| ff2x_path_summary_header_line(&path.damping))
        .collect()
}

fn ff2x_pre_table_header_lines(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    list: &ListDatData,
    path_summary_header_lines: &[String],
) -> Vec<String> {
    let mut lines = ff2x_wrhead_lines(input, xsect, list);
    lines.extend(path_summary_header_lines.iter().cloned());
    lines
}

fn ff2x_wrhead_lines(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
    list: &ListDatData,
) -> Vec<String> {
    let mut titles = xsect
        .titles
        .iter()
        .chain(list.titles.iter())
        .map(|title| title.trim_end())
        .filter(|title| !title.is_empty());
    let first_title = titles.next().unwrap_or("Untitled");

    let mut lines = Vec::new();
    lines.push(ff2x_wrhead_first_title_line(first_title));
    lines.extend(titles.map(|title| format!("# {}", ff2x_rdhead_title_record(title))));
    lines.push(format!(
        "#  S02={:5.3}  Temp={:7.2}  Debye_temp={:7.2}  Global_sig2={:8.5}",
        xsect.amplitude_reduction, input.debye.tk, input.debye.thetad, input.debye.sig2g
    ));
    if input.debye.alphat > 0.0 {
        lines.push(format!(
            "#  1st and 3rd cumulants, alphat = {:20.4E}",
            input.debye.alphat
        ));
    }
    lines.push(format!(
        "#  Energy zero shift, vr, vi {:14.5E}{:14.5E}",
        input.corrections.vrcorr,
        ff2x_effective_imaginary_correction_ev(input, xsect)
    ));
    if input.corrections.critcw > 0.0 {
        lines.push(format!(
            "#  Curved wave amplitude ratio filter {:7.3}%",
            input.corrections.critcw
        ));
    }
    lines.push(
        "#     file         sig2 tot  cw amp ratio   deg  nlegs   reff  inp sig2".to_string(),
    );
    lines
}

fn ff2x_uses_debye_waller_correction(input: &Ff2xInput) -> bool {
    input.debye.tk > 1.0e-3
}

fn ff2x_debye_log_line(idwopt: i32) -> Option<&'static str> {
    match idwopt {
        0 => Some("Applying Debye-Waller factors using a Correlated Debye model."),
        1 => Some("Applying Debye-Waller factors using the Equation-of-Motion method."),
        2 => Some("Applying Debye-Waller factors using the Recursion method."),
        3 => Some("Applying Debye-Waller factors using the Classical Debye model."),
        4 => Some("Applying Debye-Waller factors using the sig.dat file."),
        5 => Some("Applying Debye-Waller factors using the ab-initio Dynamical Matrix model."),
        _ => None,
    }
}

fn ff2x_has_energy_correction(input: &Ff2xInput) -> bool {
    ff2x_real_correction_hartree(input).abs() >= FF2X_EPS4
        || ff2x_imaginary_correction_hartree(input).abs() >= FF2X_EPS4
}

fn ff2x_has_effective_energy_correction(input: &Ff2xInput, xsect: &XsectFf2xHandoff) -> bool {
    ff2x_real_correction_hartree(input).abs() >= FF2X_EPS4
        || ff2x_effective_imaginary_correction_hartree(input, xsect).abs() >= FF2X_EPS4
}

fn ff2x_effective_imaginary_correction_hartree(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
) -> Real {
    if ff2x_xmu_effective_ispec(input.control.ispec).is_some() && input.control.i_gamma_ch == 1 {
        xsect.core_hole_width_hartree * 0.5
    } else {
        ff2x_imaginary_correction_hartree(input)
    }
}

fn ff2x_effective_imaginary_correction_ev(input: &Ff2xInput, xsect: &XsectFf2xHandoff) -> Real {
    ff2x_effective_imaginary_correction_hartree(input, xsect) * FEFF_HARTREE_EV
}

fn ff2x_wrhead_first_title_line(title: &str) -> String {
    let record = ff2x_rdhead_title_record(title);
    let title_field = record.chars().take(55).collect::<String>();
    let prefix = format!("# {title_field:<55}");
    format!("{prefix:<65}{FF2X_FEFF_VERSION}")
}

fn ff2x_rdhead_title_record(title: &str) -> String {
    format!("# {}", title.trim_end())
}

fn ff2x_path_summary_header_line(damping: &Ff2xPathDamping) -> String {
    let mut line = format!(
        "#  {:>10}     {:>9.5}{:>10.2}{:>10.2}{:>6}{:>9.4}",
        damping.path_index,
        damping.total_sigma2_angstrom2,
        damping.criterion,
        damping.degeneracy,
        damping.leg_count,
        damping.effective_half_path_length_angstrom,
    );
    if damping.user_sigma2_angstrom2.abs() > 0.000_001 {
        let _ = write!(line, "{:>9.5}", damping.user_sigma2_angstrom2);
    }
    line
}

fn ff2x_sum_prepared_paths(
    feff: &FeffBinData,
    prepared: &[Ff2xPreparedPath],
    output_momentum: ArrayView1<'_, Real>,
) -> Result<Ff2xPathSum> {
    let source_len = ff2x_real_momentum_interpolation_len(feff)?;
    ff2x_sum_prepared_paths_with_source_len(feff, prepared, source_len, output_momentum)
}

fn ff2x_sum_prepared_paths_with_source_len(
    feff: &FeffBinData,
    prepared: &[Ff2xPreparedPath],
    source_len: usize,
    output_momentum: ArrayView1<'_, Real>,
) -> Result<Ff2xPathSum> {
    ff2x_validate_path_sum_source_len_inputs(feff, prepared, source_len, output_momentum)?;
    let source_momentum = feff
        .real_momentum
        .as_slice()
        .context("FF2X feff.bin real momentum grid is not contiguous")?;
    let source_momentum = &source_momentum[..source_len];
    let mut total = Array1::<Complex64>::zeros(output_momentum.len());
    let mut paths = Vec::with_capacity(prepared.len());

    for path in prepared {
        let amplitude = path.amplitude.as_slice().with_context(|| {
            format!(
                "FF2X path {} amplitude is not contiguous",
                path.damping.path_index
            )
        })?;
        let amplitude = &amplitude[..source_len];
        let phase = path.phase.as_slice().with_context(|| {
            format!(
                "FF2X path {} phase is not contiguous",
                path.damping.path_index
            )
        })?;
        let phase = &phase[..source_len];
        let mut signal = Array1::<Complex64>::zeros(output_momentum.len());
        for (row, &momentum) in output_momentum.iter().enumerate() {
            let interpolated_amplitude =
                terp1(source_momentum, amplitude, momentum).with_context(|| {
                    format!(
                        "FF2X path {} amplitude interpolation",
                        path.damping.path_index
                    )
                })?;
            let interpolated_phase =
                terp1(source_momentum, phase, momentum).with_context(|| {
                    format!("FF2X path {} phase interpolation", path.damping.path_index)
                })?;
            let phase_angle =
                2.0 * momentum * path.damping.effective_half_path_length_bohr + interpolated_phase;
            let path_signal = Complex64::new(
                interpolated_amplitude * phase_angle.cos(),
                interpolated_amplitude * phase_angle.sin(),
            );
            signal[row] = path_signal;
            total[row] += path_signal;
        }
        paths.push(Ff2xPathSignal {
            path_index: path.damping.path_index,
            signal,
        });
    }

    Ok(Ff2xPathSum { total, paths })
}

fn ff2x_sum_decomposed_paths(
    feff: &FeffBinData,
    feffl: &FefflBinData,
    prepared: &[Ff2xPreparedPath],
    output_momentum: ArrayView1<'_, Real>,
) -> Result<Ff2xDecomposedPathSum> {
    let source_len = ff2x_real_momentum_interpolation_len(feff)?;
    ff2x_sum_decomposed_paths_with_source_len(feff, feffl, prepared, source_len, output_momentum)
}

fn ff2x_sum_decomposed_paths_with_source_len(
    feff: &FeffBinData,
    feffl: &FefflBinData,
    prepared: &[Ff2xPreparedPath],
    source_len: usize,
    output_momentum: ArrayView1<'_, Real>,
) -> Result<Ff2xDecomposedPathSum> {
    let channel_count = ff2x_validate_decomposed_path_sum_inputs(
        feff,
        feffl,
        prepared,
        source_len,
        output_momentum,
    )?;
    let source_momentum = feff
        .real_momentum
        .as_slice()
        .context("FF2X feff.bin real momentum grid is not contiguous")?;
    let source_momentum = &source_momentum[..source_len];
    let mut total =
        Array3::<Complex64>::zeros((output_momentum.len(), channel_count, channel_count));
    let mut paths = Vec::with_capacity(prepared.len());

    for (record_index, path) in prepared.iter().enumerate() {
        let mut signal =
            Array3::<Complex64>::zeros((output_momentum.len(), channel_count, channel_count));
        for lg2 in 0..channel_count {
            for lg1 in 0..channel_count {
                let amplitude = (0..source_len)
                    .map(|energy| feffl.amplitudes[(record_index, lg2, lg1, energy)])
                    .collect::<Vec<_>>();
                let phase = (0..source_len)
                    .map(|energy| feffl.phases[(record_index, lg2, lg1, energy)])
                    .collect::<Vec<_>>();
                for (row, &momentum) in output_momentum.iter().enumerate() {
                    let interpolated_amplitude = terp1(source_momentum, &amplitude, momentum)
                        .with_context(|| {
                            format!(
                                "FF2X decomposed path {} channel ({lg2},{lg1}) amplitude interpolation",
                                path.damping.path_index
                            )
                        })?;
                    let interpolated_phase = terp1(source_momentum, &phase, momentum)
                        .with_context(|| {
                            format!(
                                "FF2X decomposed path {} channel ({lg2},{lg1}) phase interpolation",
                                path.damping.path_index
                            )
                        })?;
                    let phase_angle = 2.0 * momentum * path.damping.effective_half_path_length_bohr
                        + interpolated_phase;
                    let path_signal = Complex64::new(
                        interpolated_amplitude * phase_angle.cos(),
                        interpolated_amplitude * phase_angle.sin(),
                    );
                    signal[(row, lg2, lg1)] = path_signal;
                    total[(row, lg2, lg1)] += path_signal;
                }
            }
        }
        paths.push(Ff2xDecomposedPathSignal {
            path_index: path.damping.path_index,
            signal,
        });
    }

    Ok(Ff2xDecomposedPathSum { total, paths })
}

fn ff2x_generation_momentum_grid(
    input: &Ff2xInput,
    feff: &FeffBinData,
    xsect: &XsectFf2xHandoff,
) -> Result<Ff2xMomentumGrid> {
    if input.control.ispec == 3 {
        return ff2x_danes_momentum_grid(input, xsect);
    }
    ff2x_momentum_grid(input, feff)
}

fn ff2x_generation_path_sum(
    input: &Ff2xInput,
    feff: &FeffBinData,
    prepared: &[Ff2xPreparedPath],
    xsect: &XsectFf2xHandoff,
    momentum_grid: &Ff2xMomentumGrid,
) -> Result<Ff2xPathSum> {
    if input.control.ispec == 3 {
        return ff2x_sum_prepared_paths_with_source_len(
            feff,
            prepared,
            xsect.main_energy_count,
            momentum_grid.interpolation_momentum.view(),
        );
    }
    ff2x_sum_prepared_paths(feff, prepared, momentum_grid.interpolation_momentum.view())
}

fn ff2x_danes_momentum_grid(
    input: &Ff2xInput,
    xsect: &XsectFf2xHandoff,
) -> Result<Ff2xMomentumGrid> {
    let real_correction_hartree = ff2x_real_correction_hartree(input);
    if !real_correction_hartree.is_finite() {
        bail!(
            "FF2X DANES real correction is not finite after Hartree conversion: {}",
            real_correction_hartree
        );
    }

    let mut output = Vec::with_capacity(xsect.energy_count());
    for (index, &source_momentum) in xsect.wave_number.iter().enumerate() {
        if !source_momentum.is_finite() {
            bail!("FF2X DANES xsect.dat momentum {index} is not finite: {source_momentum}");
        }
        let shifted_energy =
            (source_momentum * source_momentum.abs() + 2.0 * real_correction_hartree) / 2.0;
        output.push(wave_number_from_hartree(shifted_energy));
    }
    let output_momentum = Array1::from_vec(output);
    Ok(Ff2xMomentumGrid {
        interpolation_momentum: output_momentum.clone(),
        output_momentum,
    })
}

fn ff2x_momentum_grid(input: &Ff2xInput, feff: &FeffBinData) -> Result<Ff2xMomentumGrid> {
    if ff2x_uses_source_aligned_momentum(input) {
        return ff2x_source_aligned_momentum_grid(input, feff);
    }
    ff2x_exafs_fine_momentum_grid(input, feff, feff.energy_count().saturating_mul(100))
}

fn ff2x_uses_source_aligned_momentum(input: &Ff2xInput) -> bool {
    let ispec = input.control.ispec;
    ff2x_xmu_effective_ispec(ispec).is_some() || ispec == 3 || ispec == 4
}

fn ff2x_source_aligned_momentum_grid(
    input: &Ff2xInput,
    feff: &FeffBinData,
) -> Result<Ff2xMomentumGrid> {
    let output_momentum = ff2x_source_aligned_output_momentum(input, feff)?;
    Ok(Ff2xMomentumGrid {
        interpolation_momentum: output_momentum.clone(),
        output_momentum,
    })
}

fn ff2x_source_aligned_output_momentum(
    input: &Ff2xInput,
    feff: &FeffBinData,
) -> Result<Array1<Real>> {
    let real_correction_hartree = ff2x_real_correction_hartree(input);
    if !real_correction_hartree.is_finite() {
        bail!(
            "FF2X real correction is not finite after Hartree conversion: {}",
            real_correction_hartree
        );
    }

    let source_len = ff2x_real_momentum_interpolation_len(feff)?;
    let source_momentum = feff
        .real_momentum
        .as_slice()
        .context("FF2X feff.bin real momentum grid is not contiguous")?;
    let mut output = Vec::with_capacity(source_len);
    for (index, &source_momentum) in source_momentum[..source_len].iter().enumerate() {
        if !source_momentum.is_finite() {
            bail!("FF2X source momentum {index} is not finite: {source_momentum}");
        }
        let shifted_energy =
            (source_momentum * source_momentum.abs() + 2.0 * real_correction_hartree) / 2.0;
        output.push(wave_number_from_hartree(shifted_energy));
    }
    Ok(Array1::from_vec(output))
}

fn ff2x_exafs_fine_momentum_grid(
    input: &Ff2xInput,
    feff: &FeffBinData,
    max_points: usize,
) -> Result<Ff2xMomentumGrid> {
    if max_points == 0 {
        bail!("FF2X EXAFS fine momentum grid requires a positive point limit");
    }
    let source = feff
        .real_momentum
        .as_slice()
        .context("FF2X feff.bin real momentum grid is not contiguous")?;
    let source_len = ff2x_real_momentum_interpolation_len(feff)?;
    let source = &source[..source_len];

    let real_correction_hartree = ff2x_real_correction_hartree(input);
    if !real_correction_hartree.is_finite() {
        bail!(
            "FF2X real correction is not finite after Hartree conversion: {}",
            real_correction_hartree
        );
    }
    let delta = 0.05 * FEFF_BOHR_ANGSTROM;
    let first_energy = source[0].signum() * source[0].powi(2) / 2.0 + real_correction_hartree;
    let first_shifted = wave_number_from_hartree(first_energy);
    let mut offset = (first_shifted / delta).trunc() as i64;
    if first_shifted > 0.0 {
        offset += 1;
    }
    let output_start = offset as Real * delta;
    let source_max = source[source.len() - 1];

    let mut output_momentum = Vec::new();
    let mut interpolation_momentum = Vec::new();
    for point in 0..max_points {
        let output = output_start + delta * point as Real;
        let sign = if output < 0.0 { -1.0 } else { 1.0 };
        let interpolation_energy = sign * output.powi(2) / 2.0 - real_correction_hartree;
        let interpolation = wave_number_from_hartree(interpolation_energy);
        if interpolation > source_max + FF2X_EPS4 {
            break;
        }
        output_momentum.push(output);
        interpolation_momentum.push(interpolation);
    }
    if output_momentum.is_empty() {
        bail!("FF2X EXAFS fine momentum grid has no points inside the source grid");
    }

    Ok(Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(output_momentum),
        interpolation_momentum: Array1::from_vec(interpolation_momentum),
    })
}

fn ff2x_validate_path_sum_source_len_inputs(
    feff: &FeffBinData,
    prepared: &[Ff2xPreparedPath],
    source_len: usize,
    output_momentum: ArrayView1<'_, Real>,
) -> Result<()> {
    let energy_count = feff.energy_count();
    if feff.real_momentum.len() != energy_count {
        bail!(
            "FF2X feff.bin real momentum length {} does not match energy count {}",
            feff.real_momentum.len(),
            energy_count
        );
    }
    if source_len < 2 || source_len > energy_count {
        bail!("FF2X path summation source length {source_len} must be in 2..={energy_count}");
    }
    for index in 0..source_len {
        let momentum = feff.real_momentum[index];
        if !momentum.is_finite() {
            bail!("FF2X source momentum {index} is not finite: {momentum}");
        }
        if index > 0 && momentum <= feff.real_momentum[index - 1] {
            bail!(
                "FF2X source momentum grid must be strictly increasing through row {source_len}; row {index} has {momentum} after {}",
                feff.real_momentum[index - 1]
            );
        }
    }
    for (index, &momentum) in output_momentum.iter().enumerate() {
        if !momentum.is_finite() {
            bail!("FF2X output momentum {index} is not finite: {momentum}");
        }
    }
    for path in prepared {
        if path.amplitude.len() != energy_count {
            bail!(
                "FF2X prepared path {} amplitude length {} does not match energy count {}",
                path.damping.path_index,
                path.amplitude.len(),
                energy_count
            );
        }
        if path.phase.len() != energy_count {
            bail!(
                "FF2X prepared path {} phase length {} does not match energy count {}",
                path.damping.path_index,
                path.phase.len(),
                energy_count
            );
        }
        for (index, &value) in path.amplitude.iter().enumerate() {
            if !value.is_finite() {
                bail!(
                    "FF2X prepared path {} amplitude {} is not finite: {}",
                    path.damping.path_index,
                    index,
                    value
                );
            }
        }
        for (index, &value) in path.phase.iter().enumerate() {
            if !value.is_finite() {
                bail!(
                    "FF2X prepared path {} phase {} is not finite: {}",
                    path.damping.path_index,
                    index,
                    value
                );
            }
        }
    }
    Ok(())
}

fn ff2x_validate_decomposed_path_sum_inputs(
    feff: &FeffBinData,
    feffl: &FefflBinData,
    prepared: &[Ff2xPreparedPath],
    source_len: usize,
    output_momentum: ArrayView1<'_, Real>,
) -> Result<usize> {
    ff2x_validate_path_sum_source_len_inputs(feff, prepared, source_len, output_momentum)?;
    let channel_count = feffl
        .max_decomposition_channel
        .checked_add(1)
        .with_context(|| {
            format!(
                "FF2X feffl.bin decomposition channel count overflows for ldecmx={}",
                feffl.max_decomposition_channel
            )
        })?;
    let expected_shape = vec![
        prepared.len(),
        channel_count,
        channel_count,
        feff.energy_count(),
    ];
    let amplitude_shape = feffl.amplitudes.shape().to_vec();
    if amplitude_shape != expected_shape {
        bail!(
            "FF2X feffl.bin amplitude shape {:?} does not match expected {:?}",
            amplitude_shape,
            expected_shape
        );
    }
    let phase_shape = feffl.phases.shape().to_vec();
    if phase_shape != expected_shape {
        bail!(
            "FF2X feffl.bin phase shape {:?} does not match expected {:?}",
            phase_shape,
            expected_shape
        );
    }
    for (index, &value) in feffl.amplitudes.iter().enumerate() {
        if !value.is_finite() {
            bail!("FF2X feffl.bin amplitude value {index} is not finite: {value}");
        }
    }
    for (index, &value) in feffl.phases.iter().enumerate() {
        if !value.is_finite() {
            bail!("FF2X feffl.bin phase value {index} is not finite: {value}");
        }
    }
    Ok(channel_count)
}

fn ff2x_real_momentum_interpolation_len(feff: &FeffBinData) -> Result<usize> {
    let source = feff
        .real_momentum
        .as_slice()
        .context("FF2X feff.bin real momentum grid is not contiguous")?;
    if source.len() < 2 {
        bail!(
            "FF2X momentum interpolation requires at least two source points, got {}",
            source.len()
        );
    }
    for (index, &momentum) in source.iter().enumerate() {
        if !momentum.is_finite() {
            bail!("FF2X source momentum {index} is not finite: {momentum}");
        }
    }

    let mut interpolation_len = source.len();
    for index in 1..source.len() {
        if source[index] <= source[index - 1] {
            interpolation_len = index;
            break;
        }
    }
    if interpolation_len < 2 {
        bail!(
            "FF2X source momentum grid must have at least two increasing points; row 1 has {} after {}",
            source[1],
            source[0]
        );
    }
    if interpolation_len < source.len() {
        for (offset, &momentum) in source[interpolation_len..].iter().enumerate() {
            if momentum.abs() > FF2X_EPS4 {
                let index = interpolation_len + offset;
                bail!(
                    "FF2X source momentum grid stops increasing at row {interpolation_len}, but row {index} contains non-padding value {momentum}"
                );
            }
        }
    }
    Ok(interpolation_len)
}

fn ff2x_damped_amplitude_phase(
    input: &Ff2xInput,
    feff: &FeffBinData,
    path: &FeffBinPath,
    damping: &Ff2xPathDamping,
    imaginary_correction_hartree: Real,
) -> Result<(Array1<Real>, Array1<Real>)> {
    let energy_count = feff.energy_count();
    if feff.complex_momentum.len() != energy_count {
        bail!(
            "FF2X feff.bin complex momentum length {} does not match energy count {}",
            feff.complex_momentum.len(),
            energy_count
        );
    }
    if path.amplitude.len() != energy_count {
        bail!(
            "FF2X feff.bin path {} amplitude length {} does not match energy count {}",
            path.index,
            path.amplitude.len(),
            energy_count
        );
    }
    if path.phase.len() != energy_count {
        bail!(
            "FF2X feff.bin path {} phase length {} does not match energy count {}",
            path.index,
            path.phase.len(),
            energy_count
        );
    }

    let s02 = if input.control.mbconv > 0 {
        1.0
    } else {
        input.corrections.s02
    };
    let sigma2_bohr2 = damping.total_sigma2_angstrom2 / FEFF_BOHR_ANGSTROM.powi(2);
    let (first_cumulant_bohr, third_cumulant_bohr3) =
        damping.cumulants.map_or((0.0, 0.0), |cumulants| {
            (
                cumulants.first_cumulant_bohr,
                cumulants.third_cumulant_bohr3,
            )
        });

    let mut amplitude = Array1::<Real>::zeros(energy_count);
    let mut phase = Array1::<Real>::zeros(energy_count);
    for energy in 0..energy_count {
        let ck = feff.complex_momentum[energy];
        let ck2 = ck * ck;
        let ck3 = ck2 * ck;
        let dw = (ck2 * Complex64::new(-2.0 * sigma2_bohr2, 0.0)).exp()
            * (ck * Complex64::new(0.0, 2.0 * first_cumulant_bohr)).exp()
            * (ck3 * Complex64::new(0.0, -4.0 * third_cumulant_bohr3 / 3.0)).exp();
        let phdw = if dw.norm() > 0.0 {
            dw.im.atan2(dw.re)
        } else {
            0.0
        };
        let imaginary_correction = ff2x_imaginary_correction_amplitude(
            imaginary_correction_hartree,
            ck,
            path.effective_half_path_length_bohr,
        )?;
        amplitude[energy] =
            path.amplitude[energy] * imaginary_correction * dw.norm() * s02 * damping.degeneracy;
        phase[energy] = path.phase[energy] + phdw;
    }

    for energy in 1..energy_count {
        phase[energy] = remove_phase_jump(phase[energy], phase[energy - 1])
            .context("FF2X path phase jump removal")?;
    }

    Ok((amplitude, phase))
}

fn ff2x_imaginary_correction_amplitude(
    imaginary_correction_hartree: Real,
    ck: Complex64,
    effective_half_path_length_bohr: Real,
) -> Result<Real> {
    if imaginary_correction_hartree.abs() < FF2X_EPS4 {
        return Ok(1.0);
    }
    if !(ck.re.is_finite()
        && ck.im.is_finite()
        && imaginary_correction_hartree.is_finite()
        && effective_half_path_length_bohr.is_finite())
    {
        bail!(
            "FF2X imaginary correction received non-finite inputs: ck={ck:?}, vicorr={}, reff={}",
            imaginary_correction_hartree,
            effective_half_path_length_bohr
        );
    }

    let shifted_momentum =
        (ck * ck + Complex64::new(0.0, 2.0 * imaginary_correction_hartree)).sqrt();
    let attenuation = ck.im - shifted_momentum.im;
    Ok((2.0 * effective_half_path_length_bohr * attenuation).exp())
}

fn ff2x_real_correction_hartree(input: &Ff2xInput) -> Real {
    input.corrections.vrcorr / FEFF_HARTREE_EV
}

fn ff2x_imaginary_correction_hartree(input: &Ff2xInput) -> Real {
    input.corrections.vicorr / FEFF_HARTREE_EV
}

#[cfg(test)]
fn ff2x_path_damping(
    input: &Ff2xInput,
    feff: &FeffBinData,
    list: &ListDatData,
) -> Result<Vec<Ff2xPathDamping>> {
    ff2x_path_damping_in_dir(None, input, feff, list)
}

fn ff2x_path_damping_in_dir(
    work_dir: Option<&Path>,
    input: &Ff2xInput,
    feff: &FeffBinData,
    list: &ListDatData,
) -> Result<Vec<Ff2xPathDamping>> {
    let mut damping = Vec::new();
    let dwcorr = input.debye.tk > 1.0e-3;
    let dmdw_context = if dwcorr && input.control.idwopt == 5 {
        let work_dir =
            work_dir.context("FF2X idwopt=5 path damping requires dmdw.inp beside ff2x.inp")?;
        Some(ff2x_dmdw_context(work_dir)?)
    } else {
        None
    };
    let mut spring_context = if dwcorr && matches!(input.control.idwopt, 1 | 2) {
        let work_dir =
            work_dir.context("FF2X spring path damping requires spring.inp beside ff2x.inp")?;
        Some(ff2x_spring_recursion_context(work_dir, feff)?)
    } else {
        None
    };

    for entry in &list.entries {
        let path = feff
            .paths
            .iter()
            .find(|path| path.index == entry.path_index)
            .with_context(|| {
                format!(
                    "FF2X list.dat path {} is missing from feff.bin",
                    entry.path_index
                )
            })?;
        if path.criterion < input.corrections.critcw {
            continue;
        }

        let debye_sigma2_angstrom2 = if dwcorr && input.control.idwopt >= 0 {
            ff2x_debye_sigma2(
                input,
                feff,
                path,
                dmdw_context.as_ref(),
                spring_context.as_mut(),
            )?
        } else {
            0.0
        };
        let total_sigma2_angstrom2 = input.debye.sig2g + entry.sigma2 + debye_sigma2_angstrom2;
        let cumulants = ff2x_path_cumulants(input, feff, path, total_sigma2_angstrom2)?;
        damping.push(Ff2xPathDamping {
            path_index: path.index,
            total_sigma2_angstrom2,
            global_sigma2_angstrom2: input.debye.sig2g,
            user_sigma2_angstrom2: entry.sigma2,
            debye_sigma2_angstrom2,
            cumulants,
            criterion: path.criterion,
            degeneracy: path.degeneracy,
            leg_count: path.leg_count(),
            effective_half_path_length_angstrom: path.effective_half_path_length_bohr
                * FEFF_BOHR_ANGSTROM,
            effective_half_path_length_bohr: path.effective_half_path_length_bohr,
        });
    }

    Ok(damping)
}

struct Ff2xDmdwContext {
    atom_positions_bohr: Array2<Real>,
    atom_masses: Array1<Real>,
    mass_weighted_matrix: Array2<Real>,
    rigid_body_modes: Array2<Real>,
    temperatures: Array1<Real>,
    pole_count: usize,
}

struct Ff2xSpringRecursionContext {
    spring: SpringInput,
    matrix: SpringDynamicalMatrix,
    state: SpringRecursionState,
}

fn ff2x_spring_recursion_context(
    work_dir: &Path,
    feff: &FeffBinData,
) -> Result<Ff2xSpringRecursionContext> {
    let spring_path = work_dir.join("spring.inp");
    let spring_text = std::fs::read_to_string(&spring_path)
        .with_context(|| format!("failed to read {}", spring_path.display()))?;
    let spring = parse_spring_input(&spring_text)
        .with_context(|| format!("failed to parse {}", spring_path.display()))?;
    let geom = read_geom_dat(work_dir)?;
    let (positions, atomic_numbers, potential_indices, absorber_index) =
        ff2x_spring_atom_table(&geom, feff)?;
    let matrix = spring_dynamical_matrix(SpringDynamicalMatrixInput {
        spring: &spring,
        atom_positions_angstrom: positions.view(),
        atomic_numbers: &atomic_numbers,
        potential_indices: &potential_indices,
        absorber_index,
    })
    .context("failed to build FF2X idwopt=2 spring dynamical matrix")?;
    Ok(Ff2xSpringRecursionContext {
        spring,
        matrix,
        state: SpringRecursionState::new(feff.potentials.len()),
    })
}

type Ff2xSpringAtomTable = (Array2<Real>, Vec<usize>, Vec<usize>, usize);

fn ff2x_spring_atom_table(geom: &GeomDat, feff: &FeffBinData) -> Result<Ff2xSpringAtomTable> {
    if geom.atoms.is_empty() {
        bail!("FF2X idwopt=2 spring damping requires nonempty geom.dat");
    }
    let mut positions = Array2::<Real>::zeros((geom.atoms.len(), 3));
    let mut atomic_numbers = Vec::with_capacity(geom.atoms.len());
    let mut potential_indices = Vec::with_capacity(geom.atoms.len());
    let mut absorber_index = None;
    for (index, atom) in geom.atoms.iter().enumerate() {
        let potential = usize::try_from(atom.iph).with_context(|| {
            format!(
                "geom.dat atom {} has negative potential {}",
                atom.index, atom.iph
            )
        })?;
        let feff_potential = feff.potentials.get(potential).with_context(|| {
            format!(
                "geom.dat atom {} references missing feff.bin potential {}",
                atom.index, potential
            )
        })?;
        positions[(index, 0)] = atom.x;
        positions[(index, 1)] = atom.y;
        positions[(index, 2)] = atom.z;
        atomic_numbers.push(feff_potential.atomic_number);
        potential_indices.push(potential);
        if potential == 0 {
            absorber_index = Some(index);
        }
    }
    let absorber_index = absorber_index
        .context("FF2X idwopt=2 spring damping requires an absorber atom with potential 0")?;
    Ok((positions, atomic_numbers, potential_indices, absorber_index))
}

fn ff2x_dmdw_context(work_dir: &Path) -> Result<Ff2xDmdwContext> {
    let calculation = read_ff2x_dmdw_calculation(work_dir)?
        .context("FF2X idwopt=5 path damping requires enabled dmdw.inp")?;
    validate_ff2x_dmdw_calculation(&calculation)?;
    let pole_count =
        usize::try_from(calculation.order).context("failed to convert DMDW Lanczos order")?;
    let dym_path = work_dir.join(&calculation.dym_file);
    let dym =
        read_dym(&dym_path).with_context(|| format!("failed to read {}", dym_path.display()))?;
    let atom_positions_bohr = dym.coordinates.cartesian_positions();
    let mass_weighted =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())
            .context("failed to build DMDW mass-weighted dynamical matrix for FF2X")?;
    let rigid_body_modes =
        dmdw_rigid_body_projection_modes(atom_positions_bohr.view(), dym.atomic_masses.view())
            .context("failed to build DMDW rigid-body modes for FF2X")?;

    Ok(Ff2xDmdwContext {
        atom_positions_bohr,
        atom_masses: dym.atomic_masses,
        mass_weighted_matrix: mass_weighted.matrix,
        rigid_body_modes: rigid_body_modes.projection_modes,
        temperatures: ff2x_dmdw_temperatures(&calculation)?,
        pole_count,
    })
}

fn read_ff2x_dmdw_calculation(work_dir: &Path) -> Result<Option<DmdwCalculation>> {
    let input_path = work_dir.join("dmdw.inp");
    if !input_path.is_file() {
        return Ok(None);
    }
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let input = DmdwInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))?;
    match input {
        DmdwInput::Disabled => Ok(None),
        DmdwInput::Enabled(calculation) => Ok(Some(calculation)),
    }
}

fn validate_ff2x_dmdw_calculation(calculation: &DmdwCalculation) -> Result<()> {
    if calculation.order <= 0 {
        bail!(
            "FF2X idwopt=5 requires a positive DMDW Lanczos order, got {}",
            calculation.order
        );
    }
    if calculation.temperature_flag <= 0 {
        bail!(
            "FF2X idwopt=5 requires a positive DMDW temperature count, got {}",
            calculation.temperature_flag
        );
    }
    if !calculation.temperature.is_finite() {
        bail!("FF2X idwopt=5 requires a finite DMDW temperature");
    }
    if calculation.temperature_flag > 1 {
        let temperature_max = calculation
            .temperature_max
            .context("FF2X idwopt=5 DMDW multi-temperature input requires an upper temperature")?;
        if !temperature_max.is_finite() {
            bail!("FF2X idwopt=5 requires a finite DMDW upper temperature");
        }
    }
    if calculation.dym_file.trim().is_empty() {
        bail!("FF2X idwopt=5 requires a DMDW dynamical-matrix filename");
    }
    Ok(())
}

fn ff2x_dmdw_temperatures(calculation: &DmdwCalculation) -> Result<Array1<Real>> {
    let temperature_count =
        usize::try_from(calculation.temperature_flag).context("invalid DMDW temperature count")?;
    if temperature_count == 1 {
        return Ok(Array1::from_vec(vec![calculation.temperature]));
    }
    let temperature_max = calculation
        .temperature_max
        .context("DMDW multi-temperature run requires an upper temperature")?;
    let (start, end) = if temperature_max < calculation.temperature {
        (temperature_max, calculation.temperature)
    } else {
        (calculation.temperature, temperature_max)
    };
    Ok(Array1::linspace(start, end, temperature_count))
}

fn ff2x_path_cumulants(
    input: &Ff2xInput,
    feff: &FeffBinData,
    path: &FeffBinPath,
    total_sigma2_angstrom2: Real,
) -> Result<Option<Ff2xPathCumulants>> {
    if input.debye.alphat <= 0.0 || path.leg_count() != 2 {
        return Ok(None);
    }

    let sigma2_bohr2 = total_sigma2_angstrom2 / FEFF_BOHR_ANGSTROM.powi(2);
    let (first_cumulant_bohr, third_cumulant_bohr3) = if input.debye.thetae <= 0.0 {
        let central_atomic_number = ff2x_path_atomic_number(feff, path, path.potential_indices[1])?;
        let neighbor_atomic_number =
            ff2x_path_atomic_number(feff, path, path.potential_indices[0])?;
        let cumulants = thermal_expansion_cumulants(
            central_atomic_number,
            neighbor_atomic_number,
            sigma2_bohr2,
            input.debye.alphat,
            input.debye.thetad,
            path.effective_half_path_length_bohr,
        )
        .context("FF2X thermal-expansion cumulants")?;
        (cumulants.first, cumulants.third)
    } else {
        let cumulants = morse_einstein_cumulants(
            sigma2_bohr2,
            input.debye.tk,
            input.debye.alphat,
            input.debye.thetae,
        )
        .context("FF2X Morse-Einstein cumulants")?;
        (cumulants.first, cumulants.third)
    };

    Ok(Some(Ff2xPathCumulants {
        first_cumulant_bohr,
        third_cumulant_bohr3,
    }))
}

fn ff2x_debye_sigma2(
    input: &Ff2xInput,
    feff: &FeffBinData,
    path: &FeffBinPath,
    dmdw_context: Option<&Ff2xDmdwContext>,
    spring_context: Option<&mut Ff2xSpringRecursionContext>,
) -> Result<Real> {
    let debye_waller: DebyeWallerFn = match input.control.idwopt {
        0 => quantum_debye_waller_factor,
        2 => {
            let context =
                spring_context.context("FF2X idwopt=2 path damping requires spring handoffs")?;
            return ff2x_spring_recursion_sigma2(context, input, path);
        }
        1 => {
            let context =
                spring_context.context("FF2X idwopt=1 path damping requires spring handoffs")?;
            return ff2x_spring_equation_of_motion_sigma2(context, input, path);
        }
        3 => classical_debye_waller_factor,
        5 => {
            let context =
                dmdw_context.context("FF2X idwopt=5 path damping requires DMDW handoffs")?;
            return ff2x_dmdw_sigma2(context, path);
        }
        value => bail!(
            "FF2X path damping received unexpected idwopt={} Debye-Waller damping",
            value
        ),
    };
    let positions = ff2x_path_positions_angstrom(path)?;
    let atomic_numbers = ff2x_path_atomic_numbers(feff, path)?;

    debye_waller(
        input.debye.tk,
        input.debye.thetad,
        feff.average_norman_radius,
        positions.view(),
        &atomic_numbers,
    )
    .context("FF2X Debye-Waller path damping")
}

fn ff2x_spring_equation_of_motion_sigma2(
    context: &mut Ff2xSpringRecursionContext,
    input: &Ff2xInput,
    path: &FeffBinPath,
) -> Result<Real> {
    let positions = ff2x_path_positions_angstrom(path)?;
    let result = equation_of_motion_debye_waller_factor(SpringEquationOfMotionInput {
        matrix: &context.matrix,
        spring: &context.spring,
        temperature: input.debye.tk,
        path_positions_angstrom: positions.view(),
    })
    .with_context(|| {
        format!(
            "failed to compute FF2X path {} idwopt=1 Equation-of-Motion Debye-Waller factor",
            path.index
        )
    })?;
    update_spring_recursion_state(
        &mut context.state,
        &context.matrix,
        positions.view(),
        result.sigma2,
    )
    .with_context(|| {
        format!(
            "failed to update FF2X path {} idwopt=1 Equation-of-Motion Debye-Waller state",
            path.index
        )
    })?;
    Ok(result.sigma2)
}

fn ff2x_spring_recursion_sigma2(
    context: &mut Ff2xSpringRecursionContext,
    input: &Ff2xInput,
    path: &FeffBinPath,
) -> Result<Real> {
    let positions = ff2x_path_positions_angstrom(path)?;
    let result = recursion_debye_waller_factor(SpringRecursionInput {
        matrix: &context.matrix,
        temperature: input.debye.tk,
        path_positions_angstrom: positions.view(),
        state: Some(&context.state),
    })
    .with_context(|| {
        format!(
            "failed to compute FF2X path {} idwopt=2 Recursion-method Debye-Waller factor",
            path.index
        )
    })?;
    update_spring_recursion_state(
        &mut context.state,
        &context.matrix,
        positions.view(),
        result.sigma2,
    )
    .with_context(|| {
        format!(
            "failed to update FF2X path {} idwopt=2 Recursion-method Debye-Waller state",
            path.index
        )
    })?;
    Ok(result.sigma2)
}

fn ff2x_dmdw_sigma2(context: &Ff2xDmdwContext, path: &FeffBinPath) -> Result<Real> {
    let path_atoms = ff2x_dmdw_path_atom_indices(context, path)?;
    let motion = dmdw_path_motion(
        context.atom_positions_bohr.view(),
        context.atom_masses.view(),
        &path_atoms,
    )
    .context("failed to build FF2X DMDW path motion")?;
    let seed = dmdw_project_seed_vector(
        motion.initial_vector.view(),
        context.rigid_body_modes.view(),
    )
    .context("failed to project FF2X DMDW path seed")?;
    let coefficients = dmdw_lanczos_coefficients(
        context.mass_weighted_matrix.view(),
        seed.view(),
        context.pole_count,
    )
    .context("failed to compute FF2X DMDW Lanczos coefficients")?;
    let spectrum = dmdw_lanczos_pole_spectrum(
        context.pole_count,
        coefficients.alpha.view(),
        coefficients.beta.view(),
    )
    .context("failed to compute FF2X DMDW Lanczos pole spectrum")?;
    let sigma2 = dmdw_debye_waller_factors_from_poles(
        context.temperatures.view(),
        motion.reduced_mass,
        spectrum.angular_frequencies.view(),
        spectrum.weights.view(),
    )
    .context("failed to compute FF2X DMDW Debye-Waller factor")?;
    sigma2
        .get(0)
        .copied()
        .context("FF2X DMDW Debye-Waller calculation produced no sigma2 values")
}

fn ff2x_dmdw_path_atom_indices(
    context: &Ff2xDmdwContext,
    path: &FeffBinPath,
) -> Result<Vec<usize>> {
    let positions = ff2x_dmdw_path_positions_bohr(path)?;
    positions
        .outer_iter()
        .map(|position| ff2x_dmdw_atom_index(context, position))
        .collect()
}

fn ff2x_dmdw_path_positions_bohr(path: &FeffBinPath) -> Result<Array2<Real>> {
    let leg_count = path.leg_count();
    if leg_count == 0 {
        bail!("FF2X feff.bin path {} has no legs", path.index);
    }
    let (position_rows, position_columns) = path.positions.dim();
    if position_rows != leg_count || position_columns != 3 {
        bail!(
            "FF2X feff.bin path {} has positions shape ({}, {}), expected ({}, 3)",
            path.index,
            position_rows,
            position_columns,
            leg_count
        );
    }

    let mut positions = Array2::<Real>::zeros((leg_count, 3));
    for axis in 0..3 {
        positions[(0, axis)] = path.positions[(leg_count - 1, axis)];
    }
    for leg in 0..leg_count.saturating_sub(1) {
        for axis in 0..3 {
            positions[(leg + 1, axis)] = path.positions[(leg, axis)];
        }
    }
    Ok(positions)
}

fn ff2x_dmdw_atom_index(
    context: &Ff2xDmdwContext,
    position_bohr: ndarray::ArrayView1<'_, Real>,
) -> Result<usize> {
    let mut matched = None;
    for (index, row) in context.atom_positions_bohr.outer_iter().enumerate() {
        let distance = ((row[0] - position_bohr[0]).powi(2)
            + (row[1] - position_bohr[1]).powi(2)
            + (row[2] - position_bohr[2]).powi(2))
        .sqrt();
        if distance < FF2X_DMDW_MATCH_TOLERANCE_BOHR {
            if matched.is_some() {
                bail!(
                    "FF2X DMDW atom match is ambiguous for position {:?}",
                    position_bohr.to_vec()
                );
            }
            matched = Some(index);
        }
    }
    matched.with_context(|| {
        format!(
            "FF2X DMDW could not match path atom position {:?} to the dynamical matrix",
            position_bohr.to_vec()
        )
    })
}

fn ff2x_path_positions_angstrom(path: &FeffBinPath) -> Result<Array2<Real>> {
    let leg_count = path.leg_count();
    if leg_count == 0 {
        bail!("FF2X feff.bin path {} has no legs", path.index);
    }
    let (position_rows, position_columns) = path.positions.dim();
    if position_rows != leg_count || position_columns != 3 {
        bail!(
            "FF2X feff.bin path {} has positions shape ({}, {}), expected ({}, 3)",
            path.index,
            position_rows,
            position_columns,
            leg_count
        );
    }

    let mut positions = Array2::<Real>::zeros((leg_count + 1, 3));
    for axis in 0..3 {
        positions[(0, axis)] = path.positions[(leg_count - 1, axis)] * FEFF_BOHR_ANGSTROM;
    }
    for leg in 0..leg_count {
        for axis in 0..3 {
            positions[(leg + 1, axis)] = path.positions[(leg, axis)] * FEFF_BOHR_ANGSTROM;
        }
    }
    Ok(positions)
}

fn ff2x_path_atomic_numbers(feff: &FeffBinData, path: &FeffBinPath) -> Result<Vec<usize>> {
    let leg_count = path.leg_count();
    if leg_count == 0 {
        bail!("FF2X feff.bin path {} has no legs", path.index);
    }
    let mut atomic_numbers = Vec::with_capacity(leg_count + 1);
    atomic_numbers.push(ff2x_path_atomic_number(
        feff,
        path,
        path.potential_indices[leg_count - 1],
    )?);
    for potential_index in path.potential_indices.iter().copied() {
        atomic_numbers.push(ff2x_path_atomic_number(feff, path, potential_index)?);
    }
    Ok(atomic_numbers)
}

fn ff2x_path_atomic_number(
    feff: &FeffBinData,
    path: &FeffBinPath,
    potential_index: usize,
) -> Result<usize> {
    feff.potentials
        .get(potential_index)
        .map(|potential| potential.atomic_number)
        .with_context(|| {
            format!(
                "FF2X feff.bin path {} references missing potential {}",
                path.index, potential_index
            )
        })
}

fn write_ff2x_cum_dat(path: &Path, input: &Ff2xInput, damping: &[Ff2xPathDamping]) -> Result<()> {
    let entries = damping
        .iter()
        .filter_map(|path| {
            path.cumulants.map(|cumulants| CumDatEntry {
                path_index: path.path_index,
                first_cumulant_angstrom: cumulants.first_cumulant_bohr * FEFF_BOHR_ANGSTROM,
                sigma2_angstrom2: path.total_sigma2_angstrom2,
                third_cumulant_angstrom3: cumulants.third_cumulant_bohr3
                    * FEFF_BOHR_ANGSTROM.powi(3),
            })
        })
        .collect::<Vec<_>>();
    let data = CumDatData {
        einstein_temperature: input.debye.thetae,
        thermal_expansion: input.debye.alphat,
        entries,
    };
    write_cum_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Xmu,
    Chi,
    Xmul,
    Danes,
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
        let kind = if is_xmu_dat_name(name) {
            Some(CachedOutputKind::Xmu)
        } else if is_chi_dat_name(name) || is_chip_dat_name(name) {
            Some(CachedOutputKind::Chi)
        } else if name == "xmul.dat" {
            Some(CachedOutputKind::Xmul)
        } else if name == "danes.dat" {
            Some(CachedOutputKind::Danes)
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

fn is_ff2x_final_spectrum_name(name: &str) -> bool {
    is_xmu_dat_name(name)
        || is_chi_dat_name(name)
        || is_chip_dat_name(name)
        || name == "xmul.dat"
        || name == "danes.dat"
}

fn is_xmu_dat_name(name: &str) -> bool {
    name == "xmu.dat" || is_polarized_dat_name(name, "xmu")
}

fn is_chi_dat_name(name: &str) -> bool {
    name == "chi.dat" || is_polarized_dat_name(name, "chi")
}

fn is_polarized_dat_name(name: &str, stem: &str) -> bool {
    name.strip_prefix(stem)
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| {
            matches!(
                index,
                "02" | "03" | "04" | "05" | "06" | "07" | "08" | "09" | "10"
            )
        })
}

fn is_chip_dat_name(name: &str) -> bool {
    name.strip_prefix("chip")
        .and_then(|tail| tail.strip_suffix(".dat"))
        .is_some_and(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
}

#[cfg(test)]
mod tests;
