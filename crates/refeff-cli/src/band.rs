use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use refeff_io::{
    BandInput, BandPreSolverHandoffSetup, BandstructureDatData, FmsInput, GlobalInput,
    ModuleLogData, PhaseBinData, ReciprocalCell, band_k_path_setup_from_handoffs,
    band_pre_solver_setup_from_handoffs, band_pre_solver_setup_from_handoffs_with_lmaxph,
    band_search_setup_from_handoffs,
    bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs,
    bandstructure_dat_from_kspace_free_propagation_rel_handoffs,
    bandstructure_dat_from_kspace_free_propagation_spin_degenerate_handoffs,
    bandstructure_dat_from_kspace_free_propagation_spin_resolved_handoffs,
    bandstructure_dat_from_kspace_non_rel_handoffs, bandstructure_dat_from_kspace_rel_handoffs,
    bandstructure_dat_from_kspace_spin_degenerate_handoffs,
    bandstructure_dat_from_kspace_spin_resolved_handoffs, read_bandstructure_dat,
    read_module_log_dat, read_phase_bin, write_bandstructure_dat, write_module_log_dat,
};

use crate::work_dir_for_input;

pub(crate) mod kmesh;
use kmesh::{prepare_optional_or_generated_kmesh, write_optional_or_generated_kmesh};

pub(crate) const BAND_SOURCE_REQUIREMENT_ERROR: &str = "BAND generation requires cached bandstructure.dat or complete phase/reciprocal source handoffs";

/// Run the supported FEFF `BAND` cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_band_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_pre_solver_handoff(work_dir)? {
        return run_supported_pre_solver_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF `BAND` run can be satisfied from an existing band cache.
pub(crate) fn has_cached_band_output(work_dir: &Path) -> Result<bool> {
    let caches = BandCachePaths::new(work_dir);
    if !work_dir.join("band.inp").is_file() || !caches.bandstructure_dat.is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !band_enabled(&input) {
        return Ok(false);
    }
    Ok(can_use_cached_band_output(work_dir, &caches, &input))
}

/// Whether partial BAND source state can be parsed and validated before the
/// complete source-output requirement is evaluated.
pub(crate) fn has_supported_pre_solver_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("band.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !band_enabled(&input) {
        return Ok(false);
    }

    let caches = BandCachePaths::new(work_dir);
    if caches.bandstructure_dat.is_file() && can_use_cached_band_output(work_dir, &caches, &input) {
        return Ok(false);
    }
    let count = match supported_pre_solver_handoff_file_count(work_dir, &caches, &input) {
        Ok(count) => count,
        Err(_) => return Ok(false),
    };
    if count == 0 {
        return Ok(false);
    }

    match build_optional_band_pre_solver_setup(work_dir, &caches.phase_bin, &input) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Run the Rust-backed BAND source handoff files when the supported branch is complete.
///
/// Complete ordinary/freeprop non-relativistic `phase.bin` plus
/// `reciprocal.inp` handoffs, including spin-degenerate and non-degenerate
/// spin-resolved multi-spin branches, now produce `bandstructure.dat`. Partial
/// or unsupported handoffs still stop at validation/kmesh generation before the
/// complete source-output requirement.
pub(crate) fn run_supported_pre_solver_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    if !work_dir.join("band.inp").is_file() {
        return Ok(0);
    }
    let input = read_input(work_dir)?;
    if !band_enabled(&input) {
        return Ok(0);
    }

    let caches = BandCachePaths::new(work_dir);
    if caches.bandstructure_dat.is_file() && can_use_cached_band_output(work_dir, &caches, &input) {
        return Ok(0);
    }
    let count = match supported_pre_solver_handoff_file_count(work_dir, &caches, &input) {
        Ok(count) => count,
        Err(_) => return Ok(0),
    };
    if count == 0 {
        return Ok(0);
    }

    match write_source_bandstructure_if_supported(work_dir, &caches, &input) {
        Ok(Some(band_written)) => {
            let kmesh_written = write_optional_or_generated_kmesh(work_dir, &caches.kmesh_dat)?;
            let log_written = write_or_recover_module_log(&caches.logband_dat, true)?;
            return Ok(count + band_written + kmesh_written + log_written);
        }
        Ok(None) => {}
        Err(_) => return Ok(0),
    }

    match build_optional_band_pre_solver_setup(work_dir, &caches.phase_bin, &input) {
        Ok(()) => {
            let written = write_optional_or_generated_kmesh(work_dir, &caches.kmesh_dat)?;
            let log_written =
                recover_existing_module_log_if_malformed(&caches.logband_dat, count + written > 0)?;
            Ok(count + written + log_written)
        }
        Err(_) => Ok(0),
    }
}

/// Run FEFF `BAND` compatibility from source handoffs or existing band caches.
///
/// Ordinary and freeprop non-relativistic phase/reciprocal source handoffs are
/// generated directly. Missing or incomplete source state reports a normal
/// source requirement after validating deterministic pre-solver state.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let caches = BandCachePaths::new(work_dir);
    let kmesh_source_handoff = kmesh::kmesh_needs_generation(work_dir, &caches.kmesh_dat)?;
    let mut written = write_optional_or_generated_kmesh(work_dir, &caches.kmesh_dat)?;
    if !band_enabled(&input) {
        write_empty_module_log(&caches.logband_dat)?;
        return Ok(written);
    }

    if !caches.bandstructure_dat.is_file() {
        if let Some(band_written) =
            write_source_bandstructure_if_supported(work_dir, &caches, &input)?
        {
            written += band_written;
            written += write_or_recover_module_log(&caches.logband_dat, true)?;
            return Ok(written);
        }
        build_optional_band_pre_solver_setup(work_dir, &caches.phase_bin, &input)?;
        bail!(BAND_SOURCE_REQUIREMENT_ERROR);
    }

    let data = match read_bandstructure_dat(&caches.bandstructure_dat)
        .with_context(|| format!("failed to read {}", caches.bandstructure_dat.display()))
    {
        Ok(data) => data,
        Err(error) => {
            if supported_pre_solver_handoff_file_count(work_dir, &caches, &input)? == 0 {
                return Err(error);
            }
            if let Some(band_written) =
                write_source_bandstructure_if_supported(work_dir, &caches, &input)?
            {
                written += band_written;
                written += write_or_recover_module_log(&caches.logband_dat, true)?;
                return Ok(written);
            }
            match build_optional_band_pre_solver_setup(work_dir, &caches.phase_bin, &input) {
                Ok(()) => bail!(BAND_SOURCE_REQUIREMENT_ERROR),
                Err(handoff_error) if is_unsupported_band_pre_solver_handoff(&handoff_error) => {
                    return Err(error);
                }
                Err(handoff_error) => return Err(handoff_error),
            }
        }
    };
    if cached_bandstructure_is_stale_against_source_output(work_dir, &caches, &input, &data)?
        && let Some(band_written) =
            write_source_bandstructure_if_supported(work_dir, &caches, &input)?
    {
        written += band_written;
        written += write_or_recover_module_log(&caches.logband_dat, true)?;
        return Ok(written);
    }
    write_bandstructure_dat(&caches.bandstructure_dat, &data)
        .with_context(|| format!("failed to write {}", caches.bandstructure_dat.display()))?;

    let recover_malformed_log =
        can_recover_malformed_module_log(work_dir, &caches, &input, kmesh_source_handoff)?;
    written += 1_usize;
    written += write_or_recover_module_log(&caches.logband_dat, recover_malformed_log)?;
    Ok(written)
}

fn band_enabled(input: &BandInput) -> bool {
    input.mband == 1
}

fn can_use_cached_band_output(work_dir: &Path, caches: &BandCachePaths, input: &BandInput) -> bool {
    prepare_cached_band_output(work_dir, caches, input).is_ok()
}

fn prepare_cached_band_output(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
) -> Result<()> {
    let data = read_bandstructure_dat(&caches.bandstructure_dat)
        .with_context(|| format!("failed to read {}", caches.bandstructure_dat.display()))?;
    ensure_cached_bandstructure_matches_source_output_if_available(work_dir, caches, input, &data)?;
    let kmesh_source_handoff = kmesh::kmesh_needs_generation(work_dir, &caches.kmesh_dat)?;
    prepare_optional_or_generated_kmesh(work_dir, &caches.kmesh_dat)?;
    if caches.logband_dat.is_file() {
        prepare_module_log_cache(work_dir, caches, input, kmesh_source_handoff)?;
    }
    Ok(())
}

fn prepare_module_log_cache(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
    kmesh_source_handoff: bool,
) -> Result<()> {
    match read_module_log_dat(&caches.logband_dat)
        .with_context(|| format!("failed to read {}", caches.logband_dat.display()))
    {
        Ok(_) => Ok(()),
        Err(log_error) => {
            if can_recover_malformed_module_log(work_dir, caches, input, kmesh_source_handoff)? {
                Ok(())
            } else {
                Err(log_error)
            }
        }
    }
}

fn read_input(work_dir: &Path) -> Result<BandInput> {
    let input_path = work_dir.join("band.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    BandInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn supported_pre_solver_handoff_file_count(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
) -> Result<usize> {
    let mut count = usize::from(!input.freeprop && caches.phase_bin.is_file());
    let reciprocal_path = work_dir.join("reciprocal.inp");
    if reciprocal_path.is_file() {
        let reciprocal = kmesh::read_reciprocal_input(&reciprocal_path)?;
        if reciprocal.cell.is_some() {
            count += 1;
        }
    }
    Ok(count)
}

fn is_unsupported_band_pre_solver_handoff(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains("has no FEFF K-path table"))
}

fn build_optional_band_pre_solver_setup(
    work_dir: &Path,
    phase_path: &Path,
    input: &BandInput,
) -> Result<()> {
    let reciprocal_path = work_dir.join("reciprocal.inp");
    if input.freeprop {
        return build_optional_band_k_path_setup(work_dir, input);
    }
    if phase_path.is_file() && reciprocal_path.is_file() {
        let phase = read_phase_bin(phase_path)
            .with_context(|| format!("failed to read {}", phase_path.display()))?;
        let reciprocal = kmesh::read_reciprocal_input(&reciprocal_path)?;
        if let Some(cell) = reciprocal.cell.as_ref() {
            let mut setup =
                build_band_pre_solver_setup_from_handoffs(work_dir, input, &phase, cell)
                    .with_context(|| {
                        format!(
                            "failed to build BAND pre-solver setup from {} and {}",
                            phase_path.display(),
                            reciprocal_path.display()
                        )
                    })?;
            apply_global_spin_selector(work_dir, &mut setup)?;
            return Ok(());
        }

        let _setup = band_search_setup_from_handoffs(input, &phase).with_context(|| {
            format!(
                "failed to build BAND search setup from {}",
                phase_path.display()
            )
        })?;
        return Ok(());
    }

    build_optional_band_search_setup(phase_path, input)?;
    build_optional_band_k_path_setup(work_dir, input)
}

fn write_source_bandstructure_if_supported(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
) -> Result<Option<usize>> {
    let Some(data) = build_source_bandstructure_if_supported(work_dir, &caches.phase_bin, input)?
    else {
        return Ok(None);
    };
    write_bandstructure_dat(&caches.bandstructure_dat, &data)
        .with_context(|| format!("failed to write {}", caches.bandstructure_dat.display()))?;
    Ok(Some(1))
}

fn build_source_bandstructure_if_supported(
    work_dir: &Path,
    phase_path: &Path,
    input: &BandInput,
) -> Result<Option<BandstructureDatData>> {
    let Some(setup) = build_complete_band_source_setup(work_dir, phase_path, input)? else {
        return Ok(None);
    };
    let data = if can_run_band_kspace_rel_source_solve(&setup) && input.freeprop {
        bandstructure_dat_from_kspace_free_propagation_rel_handoffs(&setup)
    } else if can_run_band_kspace_rel_source_solve(&setup) {
        bandstructure_dat_from_kspace_rel_handoffs(&setup)
    } else if can_run_band_kspace_non_rel_source_solve(&setup) && input.freeprop {
        bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs(&setup)
    } else if can_run_band_kspace_non_rel_source_solve(&setup) {
        bandstructure_dat_from_kspace_non_rel_handoffs(&setup)
    } else if can_run_band_kspace_spin_degenerate_source_solve(&setup) && input.freeprop {
        bandstructure_dat_from_kspace_free_propagation_spin_degenerate_handoffs(&setup)
    } else if can_run_band_kspace_spin_degenerate_source_solve(&setup) {
        bandstructure_dat_from_kspace_spin_degenerate_handoffs(&setup)
    } else if can_run_band_kspace_spin_resolved_source_solve(&setup) && input.freeprop {
        bandstructure_dat_from_kspace_free_propagation_spin_resolved_handoffs(&setup)
    } else if can_run_band_kspace_spin_resolved_source_solve(&setup) {
        bandstructure_dat_from_kspace_spin_resolved_handoffs(&setup)
    } else {
        return Ok(None);
    };
    data.map(Some)
        .with_context(|| "failed to solve BAND bandstructure.dat from source handoffs")
}

fn build_complete_band_source_setup(
    work_dir: &Path,
    phase_path: &Path,
    input: &BandInput,
) -> Result<Option<BandPreSolverHandoffSetup>> {
    if !phase_path.is_file() {
        return Ok(None);
    }
    let reciprocal_path = work_dir.join("reciprocal.inp");
    if !reciprocal_path.is_file() {
        return Ok(None);
    }

    let phase = match read_phase_bin(phase_path) {
        Ok(phase) => phase,
        Err(_) if input.freeprop => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", phase_path.display()));
        }
    };
    let reciprocal = kmesh::read_reciprocal_input(&reciprocal_path)?;
    let Some(cell) = reciprocal.cell.as_ref() else {
        return Ok(None);
    };
    let mut setup = build_band_pre_solver_setup_from_handoffs(work_dir, input, &phase, cell)
        .with_context(|| {
            format!(
                "failed to build BAND pre-solver setup from {} and {}",
                phase_path.display(),
                reciprocal_path.display()
            )
        })?;
    apply_global_spin_selector(work_dir, &mut setup)?;
    Ok(Some(setup))
}

fn build_band_pre_solver_setup_from_handoffs(
    work_dir: &Path,
    input: &BandInput,
    phase: &PhaseBinData,
    cell: &ReciprocalCell,
) -> Result<BandPreSolverHandoffSetup> {
    let Some(lmaxph) = read_optional_fms_lmaxph(work_dir, phase.potential_count())? else {
        return band_pre_solver_setup_from_handoffs(input, phase, cell)
            .map_err(anyhow::Error::from);
    };
    band_pre_solver_setup_from_handoffs_with_lmaxph(input, phase, cell, &lmaxph)
        .map_err(anyhow::Error::from)
}

fn read_optional_fms_lmaxph(work_dir: &Path, potential_count: usize) -> Result<Option<Vec<usize>>> {
    let path = work_dir.join("fms.inp");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let input = FmsInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if input.lmaxph.len() < potential_count {
        bail!(
            "{} has {} lmaxph value(s), expected at least {potential_count}",
            path.display(),
            input.lmaxph.len()
        );
    }

    let mut lmaxph = Vec::with_capacity(potential_count);
    for (potential, &value) in input.lmaxph.iter().take(potential_count).enumerate() {
        if value < 0 {
            bail!(
                "{} lmaxph({potential}) must be nonnegative, got {value}",
                path.display()
            );
        }
        lmaxph.push(usize::try_from(value).with_context(|| {
            format!(
                "{} lmaxph({potential}) does not fit in usize",
                path.display()
            )
        })?);
    }
    Ok(Some(lmaxph))
}

fn apply_global_spin_selector(
    work_dir: &Path,
    setup: &mut BandPreSolverHandoffSetup,
) -> Result<()> {
    setup.kspace_solver_basis.spin_selector = read_optional_global_input(work_dir)?
        .as_ref()
        .map_or(0, |global| global.control.ispin);
    Ok(())
}

fn read_optional_global_input(work_dir: &Path) -> Result<Option<GlobalInput>> {
    let path = work_dir.join("global.inp");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    GlobalInput::parse_str(&path, &text)
        .map(Some)
        .with_context(|| {
            format!(
                "failed to parse {} for BAND source spin selector",
                path.display()
            )
        })
}

fn can_run_band_kspace_non_rel_source_solve(setup: &BandPreSolverHandoffSetup) -> bool {
    setup.kspace_solver_basis.spin_channels == 1
        && setup.kspace_energy.spin_count == 1
        && setup.kspace_solver_basis.matrix_order == setup.kspace_angular.matrix_order_non_rel
}

fn can_run_band_kspace_rel_source_solve(setup: &BandPreSolverHandoffSetup) -> bool {
    setup.kspace_solver_basis.spin_channels == 1
        && setup.kspace_energy.spin_count == 1
        && setup.kspace_solver_basis.spin_selector != 0
        && setup.kspace_solver_basis.matrix_order == setup.kspace_angular.matrix_order_rel
}

fn can_run_band_kspace_spin_degenerate_source_solve(setup: &BandPreSolverHandoffSetup) -> bool {
    let spin_channels = setup.kspace_solver_basis.spin_channels;
    if spin_channels <= 1 || setup.kspace_energy.spin_count != spin_channels {
        return false;
    }
    let expected_order = setup.kspace_angular.matrix_order_rel;
    setup.kspace_solver_basis.matrix_order == expected_order
        && setup.search.phase_interpolation.wave_numbers.ncols() >= spin_channels
        && has_degenerate_kspace_energy_columns(setup)
}

fn can_run_band_kspace_spin_resolved_source_solve(setup: &BandPreSolverHandoffSetup) -> bool {
    let spin_channels = setup.kspace_solver_basis.spin_channels;
    if spin_channels <= 1 || setup.kspace_energy.spin_count != spin_channels {
        return false;
    }
    let expected_order = setup.kspace_angular.matrix_order_rel;
    setup.kspace_solver_basis.matrix_order == expected_order
        && setup.search.phase_interpolation.wave_numbers.ncols() >= spin_channels
}

fn has_degenerate_kspace_energy_columns(setup: &BandPreSolverHandoffSetup) -> bool {
    const TOLERANCE: f64 = 1.0e-10;

    for energy_index in 0..setup.kspace_energy.energy_count {
        let base_wave = setup.kspace_energy.wave_numbers[(energy_index, 0)];
        let base_reduced = setup.kspace_energy.reduced_energies[(energy_index, 0)];
        for spin in 1..setup.kspace_energy.spin_count {
            if (setup.kspace_energy.wave_numbers[(energy_index, spin)] - base_wave).norm()
                > TOLERANCE
            {
                return false;
            }
            if (setup.kspace_energy.reduced_energies[(energy_index, spin)] - base_reduced).norm()
                > TOLERANCE
            {
                return false;
            }
        }
    }
    true
}

fn build_optional_band_search_setup(phase_path: &Path, input: &BandInput) -> Result<()> {
    if !phase_path.is_file() {
        return Ok(());
    }

    let phase = read_phase_bin(phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let _setup = band_search_setup_from_handoffs(input, &phase).with_context(|| {
        format!(
            "failed to build BAND search setup from {}",
            phase_path.display()
        )
    })?;
    Ok(())
}

fn build_optional_band_k_path_setup(work_dir: &Path, input: &BandInput) -> Result<()> {
    let reciprocal_path = work_dir.join("reciprocal.inp");
    if !reciprocal_path.is_file() {
        return Ok(());
    }

    let reciprocal = kmesh::read_reciprocal_input(&reciprocal_path)?;
    let Some(cell) = reciprocal.cell.as_ref() else {
        return Ok(());
    };
    let _setup = band_k_path_setup_from_handoffs(input, cell).with_context(|| {
        format!(
            "failed to build BAND k-path setup from {}",
            reciprocal_path.display()
        )
    })?;
    Ok(())
}

fn ensure_cached_bandstructure_matches_source_output_if_available(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
    data: &BandstructureDatData,
) -> Result<()> {
    if cached_bandstructure_is_stale_against_source_output(work_dir, caches, input, data)? {
        bail!("cached bandstructure.dat is stale against BAND source handoffs");
    }
    Ok(())
}

fn cached_bandstructure_is_stale_against_source_output(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
    data: &BandstructureDatData,
) -> Result<bool> {
    validate_declared_band_source_handoffs(work_dir, &caches.phase_bin, input)?;
    let Some(source) =
        supported_source_bandstructure_for_cache(work_dir, &caches.phase_bin, input)?
    else {
        return Ok(false);
    };
    Ok(!cached_bandstructure_matches_source_output(data, &source))
}

fn validate_declared_band_source_handoffs(
    work_dir: &Path,
    phase_path: &Path,
    input: &BandInput,
) -> Result<()> {
    let phase = if !input.freeprop && phase_path.is_file() {
        Some(read_phase_bin(phase_path).with_context(|| {
            format!(
                "failed to read {} for BAND source handoff validation",
                phase_path.display()
            )
        })?)
    } else {
        None
    };

    let reciprocal_path = work_dir.join("reciprocal.inp");
    if reciprocal_path.is_file() {
        kmesh::read_reciprocal_input(&reciprocal_path)?;
    }
    read_optional_global_input(work_dir)?;
    if let Some(phase) = phase.as_ref() {
        read_optional_fms_lmaxph(work_dir, phase.potential_count())?;
    } else {
        validate_declared_fms_source_input(work_dir)?;
    }
    Ok(())
}

fn validate_declared_fms_source_input(work_dir: &Path) -> Result<()> {
    let path = work_dir.join("fms.inp");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    FmsInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(())
}

fn supported_source_bandstructure_for_cache(
    work_dir: &Path,
    phase_path: &Path,
    input: &BandInput,
) -> Result<Option<BandstructureDatData>> {
    match build_source_bandstructure_if_supported(work_dir, phase_path, input) {
        Ok(Some(source)) => Ok(Some(source)),
        Ok(None) => Ok(None),
        Err(_) => Ok(None),
    }
}

fn cached_bandstructure_matches_source_output(
    data: &BandstructureDatData,
    source: &BandstructureDatData,
) -> bool {
    const KPOINT_TOLERANCE: f64 = 5.0e-4;
    const BAND_VALUE_TOLERANCE: f64 = 5.0e-5;

    if !bandstructure_headers_match_source(data, source) || data.rows.len() != source.rows.len() {
        return false;
    }
    data.rows
        .iter()
        .zip(source.rows.iter())
        .enumerate()
        .all(|(index, (row, source_row))| {
            row.index == (index + 1) as i32
                && source_row.index == (index + 1) as i32
                && row.bands.len() == source_row.bands.len()
                && row
                    .k_point
                    .iter()
                    .zip(source_row.k_point.iter())
                    .all(|(cached, source)| (cached - source).abs() <= KPOINT_TOLERANCE)
                && row
                    .bands
                    .iter()
                    .zip(source_row.bands.iter())
                    .all(|(cached, source)| (cached - source).abs() <= BAND_VALUE_TOLERANCE)
        })
}

fn bandstructure_headers_match_source(
    data: &BandstructureDatData,
    source: &BandstructureDatData,
) -> bool {
    data.header_lines.len() == source.header_lines.len()
        && data
            .header_lines
            .iter()
            .zip(source.header_lines.iter())
            .all(|(cached, source)| cached.split_whitespace().eq(source.split_whitespace()))
}

fn can_recover_malformed_module_log(
    work_dir: &Path,
    caches: &BandCachePaths,
    input: &BandInput,
    kmesh_source_handoff: bool,
) -> Result<bool> {
    if !caches.logband_dat.is_file() || read_module_log_dat(&caches.logband_dat).is_ok() {
        return Ok(false);
    }
    if kmesh_source_handoff {
        return Ok(true);
    }
    if supported_pre_solver_handoff_file_count(work_dir, caches, input)? == 0 {
        return Ok(false);
    }

    match build_optional_band_pre_solver_setup(work_dir, &caches.phase_bin, input) {
        Ok(()) => Ok(true),
        Err(error) if is_unsupported_band_pre_solver_handoff(&error) => Ok(false),
        Err(error) => Err(error),
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

fn write_empty_module_log(path: &Path) -> Result<()> {
    let data = ModuleLogData {
        lines: Vec::new(),
        line_terminators: Vec::new(),
    };
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(path: &Path) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_generated_module_log(path)
}

fn write_or_recover_module_log(path: &Path, source_handoff_written: bool) -> Result<usize> {
    if source_handoff_written && path.is_file() && read_module_log_dat(path).is_err() {
        return write_generated_module_log(path);
    }
    write_or_generate_module_log(path)
}

fn recover_existing_module_log_if_malformed(
    path: &Path,
    source_handoff_written: bool,
) -> Result<usize> {
    if !source_handoff_written || !path.is_file() || read_module_log_dat(path).is_ok() {
        return Ok(0);
    }
    write_generated_module_log(path)
}

fn write_generated_module_log(path: &Path) -> Result<usize> {
    write_module_log_dat(path, &generated_band_module_log())
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn generated_band_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating band structure ...".to_string(),
            "Solving band structure.".to_string(),
            " Done with module: band structure.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 3],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BandCachePaths {
    bandstructure_dat: PathBuf,
    kmesh_dat: PathBuf,
    phase_bin: PathBuf,
    logband_dat: PathBuf,
}

impl BandCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            bandstructure_dat: work_dir.join("bandstructure.dat"),
            kmesh_dat: work_dir.join("kmesh.dat"),
            phase_bin: work_dir.join("phase.bin"),
            logband_dat: work_dir.join("logband.dat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BAND_SOURCE_REQUIREMENT_ERROR, build_complete_band_source_setup,
        cached_bandstructure_matches_source_output, can_run_band_kspace_non_rel_source_solve,
        generated_band_module_log, has_cached_band_output, has_supported_pre_solver_handoff,
        read_input, run_in_dir, run_supported_pre_solver_handoff_in_dir,
    };
    use anyhow::{Context, Result, bail};
    use ndarray::{Array1, Array2, Array3, Array4};
    use num_complex::Complex64;
    use refeff_io::phase_bin::PHASE_BIN_DEFAULT_PAD_WIDTH;
    use refeff_io::{
        BandEnergyMesh, BandInput, BandstructureDatData, BandstructureRow, CfAverage,
        GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, KmeshDatData, KmeshMetadata,
        KmeshRow, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars, ReciprocalCell,
        ReciprocalInput, ReciprocalKMesh, band_input_string,
        band_kspace_t_matrix_grid_from_handoffs,
        bandstructure_dat_from_kspace_free_propagation_rel_handoffs,
        bandstructure_dat_from_kspace_rel_handoffs, global_input_string,
        kmesh_dat_from_reciprocal_cell, parse_kmesh_dat, phase_bin_string, read_bandstructure_dat,
        read_kmesh_dat, read_module_log_dat, reciprocal_input_string, write_bandstructure_dat,
        write_kmesh_dat, write_module_log_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn band_module_writes_empty_log_for_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_band_output(temp.path())?);
        assert!(read_module_log_dat(temp.path().join("logband.dat"))?.is_empty());
        Ok(())
    }

    #[test]
    fn band_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled BAND should require complete source state")?;

        assert!(error.to_string().contains(BAND_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn band_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;

        assert!(!has_cached_band_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn band_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("band.inp"), "not a band.inp handoff\n")?;
        let bandstructure = sample_bandstructure_dat();
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &bandstructure)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(!has_supported_pre_solver_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed band.inp should fail through the explicit BAND runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("band.inp"), "{chain}");
        assert_eq!(
            read_bandstructure_dat(temp.path().join("bandstructure.dat"))?,
            bandstructure
        );
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_generates_kmesh_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled BAND should still require complete source state")?;

        assert!(error.to_string().contains(BAND_SOURCE_REQUIREMENT_ERROR));
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        Ok(())
    }

    #[test]
    fn band_module_generates_bandstructure_from_supported_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 5);
        let bandstructure = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_source_bandstructure_generated(&bandstructure);
        assert!(temp.path().join("logband.dat").exists());
        assert!(temp.path().join("kmesh.dat").exists());
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        Ok(())
    }

    #[test]
    fn band_module_generates_bandstructure_from_freeprop_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_freeprop(temp.path(), true, true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let bandstructure = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_source_bandstructure_generated(&bandstructure);
        assert!(temp.path().join("logband.dat").exists());
        assert!(temp.path().join("kmesh.dat").exists());
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        Ok(())
    }

    #[test]
    fn band_module_carries_global_ispin_into_source_solver_basis() -> Result<()> {
        let unpolarized = tempfile::tempdir()?;
        write_band_source_handoffs(unpolarized.path())?;
        let unpolarized_input = read_input(unpolarized.path())?;
        let unpolarized_setup = build_complete_band_source_setup(
            unpolarized.path(),
            &unpolarized.path().join("phase.bin"),
            &unpolarized_input,
        )?
        .context("unpolarized BAND source setup should be complete")?;

        let polarized = tempfile::tempdir()?;
        write_band_source_handoffs(polarized.path())?;
        write_global_input(polarized.path(), 1)?;
        let polarized_input = read_input(polarized.path())?;
        let polarized_setup = build_complete_band_source_setup(
            polarized.path(),
            &polarized.path().join("phase.bin"),
            &polarized_input,
        )?
        .context("polarized BAND source setup should be complete")?;

        assert_eq!(unpolarized_setup.kspace_solver_basis.spin_selector, 0);
        assert_eq!(polarized_setup.kspace_solver_basis.spin_selector, 1);

        let unpolarized_t = band_kspace_t_matrix_grid_from_handoffs(&unpolarized_setup)?;
        let polarized_t = band_kspace_t_matrix_grid_from_handoffs(&polarized_setup)?;
        assert_eq!(unpolarized_t.dim(), polarized_t.dim());
        assert!(
            unpolarized_t
                .iter()
                .zip(polarized_t.iter())
                .any(|(left, right)| (*left - *right).norm() > 1.0e-7),
            "global.inp ispin should change the BAND spin-orbit T-matrix source handoff"
        );

        let count = run_supported_pre_solver_handoff_in_dir(polarized.path())?;

        assert_eq!(count, 5);
        let expected = bandstructure_dat_from_kspace_rel_handoffs(&polarized_setup)?;
        let actual = read_bandstructure_dat(polarized.path().join("bandstructure.dat"))?;
        assert!(
            super::cached_bandstructure_matches_source_output(&actual, &expected),
            "global.inp ispin should route one-spin BAND source output through the rel solve"
        );
        Ok(())
    }

    #[test]
    fn band_module_generates_one_spin_rel_source_handoff_from_global_spin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_source_handoffs(temp.path())?;
        write_global_input(temp.path(), 1)?;
        let input = read_input(temp.path())?;
        let setup =
            build_complete_band_source_setup(temp.path(), &temp.path().join("phase.bin"), &input)?
                .context("BAND source setup should be complete")?;
        let expected = bandstructure_dat_from_kspace_rel_handoffs(&setup)?;

        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 5);
        let actual = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert!(
            super::cached_bandstructure_matches_source_output(&actual, &expected),
            "generated one-spin rel BAND output did not match source solve"
        );
        assert!(temp.path().join("logband.dat").is_file());
        assert!(temp.path().join("kmesh.dat").is_file());
        Ok(())
    }

    #[test]
    fn band_module_generates_one_spin_rel_freeprop_source_handoff_from_global_spin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_freeprop(temp.path(), true, true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;
        write_global_input(temp.path(), 1)?;
        let input = read_input(temp.path())?;
        let setup =
            build_complete_band_source_setup(temp.path(), &temp.path().join("phase.bin"), &input)?
                .context("BAND freeprop source setup should be complete")?;
        let expected = bandstructure_dat_from_kspace_free_propagation_rel_handoffs(&setup)?;

        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let actual = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert!(
            super::cached_bandstructure_matches_source_output(&actual, &expected),
            "generated one-spin rel freeprop BAND output did not match source solve"
        );
        assert!(temp.path().join("logband.dat").is_file());
        assert!(temp.path().join("kmesh.dat").is_file());
        Ok(())
    }

    #[test]
    fn band_module_rejects_malformed_global_spin_selector_before_source_requirement() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_band_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed global.inp should fail before BAND source requirement")?;
        let chain = format!("{error:#}");

        assert!(chain.contains("global.inp"), "{chain}");
        assert!(chain.contains("BAND source spin selector"), "{chain}");
        Ok(())
    }

    #[test]
    fn band_module_does_not_claim_malformed_fms_lmaxph_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

        assert!(!has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed fms.inp should fail before BAND source requirement")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("fms.inp"), "{chain}");
        assert!(!temp.path().join("bandstructure.dat").exists());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_claim_malformed_reciprocal_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(temp.path().join("reciprocal.inp"), "not reciprocal input\n")?;

        assert!(!has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed reciprocal.inp should fail before BAND source requirement")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("reciprocal.inp"), "{chain}");
        assert!(!temp.path().join("bandstructure.dat").exists());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_accept_cached_output_with_malformed_reciprocal_source_handoff()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_kmesh_dat())?;
        std::fs::write(temp.path().join("reciprocal.inp"), "not reciprocal input\n")?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed reciprocal.inp should block cached BAND completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("reciprocal.inp"), "{chain}");
        assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat")).is_ok());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_accept_cached_output_with_malformed_phase_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_kmesh_dat())?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed phase.bin should block cached BAND completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat")).is_ok());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_accept_cached_output_with_malformed_global_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_kmesh_dat())?;
        std::fs::write(temp.path().join("global.inp"), "not a global input\n")?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed global.inp should block cached BAND completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("global.inp"), "{chain}");
        assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat")).is_ok());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_accept_cached_output_with_malformed_fms_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        write_kmesh_dat(temp.path().join("kmesh.dat"), &sample_kmesh_dat())?;
        std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed fms.inp should block cached BAND completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("fms.inp"), "{chain}");
        assert!(read_bandstructure_dat(temp.path().join("bandstructure.dat")).is_ok());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_generates_two_spin_degenerate_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_two_spin_degenerate_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 5);
        let bandstructure = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_source_bandstructure_generated(&bandstructure);
        assert!(temp.path().join("logband.dat").is_file());
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(run_in_dir(temp.path()).is_ok());
        Ok(())
    }

    #[test]
    fn band_module_generates_two_spin_non_degenerate_source_handoff() -> Result<()> {
        assert_band_module_generates_two_spin_non_degenerate_source_handoff(false, 5)
    }

    #[test]
    fn band_module_generates_two_spin_non_degenerate_freeprop_source_handoff() -> Result<()> {
        assert_band_module_generates_two_spin_non_degenerate_source_handoff(true, 4)
    }

    fn assert_band_module_generates_two_spin_non_degenerate_source_handoff(
        freeprop: bool,
        expected_count: usize,
    ) -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_two_spin_non_degenerate_band_source_handoffs(temp.path(), freeprop)?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, expected_count);
        let bandstructure = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_source_bandstructure_generated(&bandstructure);
        assert!(temp.path().join("logband.dat").is_file());
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(run_in_dir(temp.path()).is_ok());
        Ok(())
    }

    #[test]
    fn band_module_recovers_existing_malformed_log_for_supported_pre_solver_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("logband.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert_source_bandstructure_generated(&read_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
        )?);
        assert!(temp.path().join("kmesh.dat").exists());
        assert_eq!(
            read_module_log_dat(temp.path().join("logband.dat"))?,
            generated_band_module_log()
        );
        Ok(())
    }

    #[test]
    fn band_module_ignores_phase_handoff_for_freeprop_reciprocal_setup() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_freeprop(temp.path(), true, true)?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_supported_pre_solver_handoff_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(!temp.path().join("bandstructure.dat").exists());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_does_not_advertise_phase_only_handoff_for_freeprop() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_freeprop(temp.path(), true, true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;

        assert!(!has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 0);
        Ok(())
    }

    #[test]
    fn band_module_skips_supported_pre_solver_handoff_without_handoff_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;

        assert!(!has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 0);
        Ok(())
    }

    #[test]
    fn band_module_validates_k_path_setup_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_nkp(temp.path(), true, 1)?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("invalid BAND k-path setup should fail before source requirement")?;
        let chain = format!("{error:#}");

        assert!(chain.contains("failed to build BAND k-path setup from"));
        assert!(chain.contains("BAND K-path point count"));
        Ok(())
    }

    #[test]
    fn band_module_builds_phase_setup_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled BAND should still require complete source state")?;

        assert!(error.to_string().contains(BAND_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn band_module_validates_phase_setup_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        let phase_text = phase_bin_string(&sample_band_phase_bin())?;
        let (_, rest) = phase_text
            .split_once('\n')
            .context("phase.bin text should contain a header line")?;
        std::fs::write(
            temp.path().join("phase.bin"),
            format!("    1    4    0    0    0    4    1    8    1    1    1\n{rest}"),
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("invalid phase handoff should fail before source requirement")?;
        let chain = format!("{error:#}");

        assert!(chain.contains("failed to build BAND search setup from"));
        assert!(chain.contains("active BAND phase prefix"));
        Ok(())
    }

    #[test]
    fn band_module_does_not_claim_malformed_phase_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;

        assert!(!has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed BAND phase.bin should fail through the explicit BAND runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert!(!temp.path().join("bandstructure.dat").exists());
        assert!(!temp.path().join("logband.dat").exists());
        Ok(())
    }

    #[test]
    fn band_module_validates_combined_pre_solver_setup_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input_with_nkp(temp.path(), true, 1)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let error = run_in_dir(temp.path()).err().context(
            "invalid combined BAND pre-solver setup should fail before source requirement",
        )?;
        let chain = format!("{error:#}");

        assert!(chain.contains("failed to build BAND pre-solver setup from"));
        assert!(chain.contains("BAND K-path point count"));
        Ok(())
    }

    #[test]
    fn band_module_does_not_advertise_malformed_bandstructure_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("bandstructure.dat"),
            "not bandstructure.dat\n",
        )?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed bandstructure.dat should fail through the explicit BAND runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("bandstructure.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn band_module_regenerates_malformed_bandstructure_cache_from_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("bandstructure.dat"),
            "not bandstructure.dat\n",
        )?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_source_bandstructure_generated(&read_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
        )?);
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        assert_eq!(
            read_module_log_dat(temp.path().join("logband.dat"))?,
            generated_band_module_log()
        );
        Ok(())
    }

    #[test]
    fn band_module_regenerates_stale_bandstructure_cache_from_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let bandstructure = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_ne!(
            bandstructure.k_point_count(),
            sample_bandstructure_dat().k_point_count()
        );
        assert_source_bandstructure_generated(&bandstructure);
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        assert_eq!(
            read_module_log_dat(temp.path().join("logband.dat"))?,
            generated_band_module_log()
        );
        Ok(())
    }

    #[test]
    fn band_module_regenerates_bandstructure_cache_with_stale_band_counts_from_source_handoff()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        run_in_dir(temp.path())?;
        let mut stale = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        for row in &mut stale.rows {
            row.bands = Array1::zeros(0);
        }
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let regenerated = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_source_bandstructure_generated(&regenerated);
        Ok(())
    }

    #[test]
    fn band_module_regenerates_bandstructure_cache_with_stale_band_values_from_source_handoff()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_source_handoffs(temp.path())?;
        run_in_dir(temp.path())?;
        let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        let mut stale = expected.clone();
        stale
            .rows
            .iter_mut()
            .find(|row| !row.bands.is_empty())
            .context("source bandstructure should contain at least one band")?
            .bands[0] += 99.0;
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let regenerated = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_eq!(regenerated, expected);
        assert_source_bandstructure_generated(&regenerated);
        Ok(())
    }

    #[test]
    fn band_module_regenerates_stale_two_spin_bandstructure_cache_from_source_handoff() -> Result<()>
    {
        assert_band_module_regenerates_stale_two_spin_bandstructure(false)
    }

    #[test]
    fn band_module_regenerates_stale_two_spin_freeprop_bandstructure_cache_from_source_handoff()
    -> Result<()> {
        assert_band_module_regenerates_stale_two_spin_bandstructure(true)
    }

    #[test]
    fn band_module_regenerates_stale_one_spin_rel_bandstructure_cache_from_source_handoff()
    -> Result<()> {
        assert_band_module_regenerates_stale_one_spin_rel_bandstructure(false)
    }

    #[test]
    fn band_module_regenerates_stale_one_spin_rel_freeprop_bandstructure_cache_from_source_handoff()
    -> Result<()> {
        assert_band_module_regenerates_stale_one_spin_rel_bandstructure(true)
    }

    #[test]
    fn band_module_regenerates_bandstructure_cache_with_stale_header_from_source_handoff()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_source_handoffs(temp.path())?;
        run_in_dir(temp.path())?;
        let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        let mut stale = expected.clone();
        stale.header_lines[1] =
            " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000".to_string();
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let regenerated = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_eq!(regenerated, expected);
        assert_source_bandstructure_generated(&regenerated);
        Ok(())
    }

    #[test]
    fn band_cache_header_match_allows_spacing_but_rejects_metadata_changes() {
        let source = sample_bandstructure_dat();
        let mut spaced = source.clone();
        spaced.header_lines = source
            .header_lines
            .iter()
            .map(|line| line.split_whitespace().collect::<Vec<_>>().join("   "))
            .collect();
        assert!(cached_bandstructure_matches_source_output(&spaced, &source));

        let mut stale = source.clone();
        stale.header_lines[1] = stale.header_lines[1].replacen("4", "5", 1);
        assert!(!cached_bandstructure_matches_source_output(&stale, &source));
    }

    #[test]
    fn band_module_validates_pre_solver_handoff_when_malformed_bandstructure_cache_exists()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("bandstructure.dat"),
            "not bandstructure.dat\n",
        )?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        assert_eq!(run_supported_pre_solver_handoff_in_dir(temp.path())?, 5);
        assert_eq!(run_in_dir(temp.path())?, 3);
        assert_source_bandstructure_generated(&read_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
        )?);
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(temp.path().join("logband.dat").is_file());
        Ok(())
    }

    #[test]
    fn band_module_does_not_advertise_malformed_kmesh_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed kmesh.dat should fail through the explicit BAND runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("kmesh.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn band_module_recovers_malformed_kmesh_from_reciprocal_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

        assert!(has_cached_band_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert_kmesh_row_close(
            data.rows[0],
            KmeshRow {
                index: 1,
                k_point: [0.831_446_454_055_273_6; 3],
                weight: 0.5,
                metadata: data.rows[0].metadata,
            },
            5.0e-4,
        );
        Ok(())
    }

    #[test]
    fn band_module_recovers_malformed_module_log_for_recoverable_kmesh_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("logband.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_band_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        assert_eq!(
            read_module_log_dat(temp.path().join("logband.dat"))?,
            super::generated_band_module_log()
        );
        Ok(())
    }

    #[test]
    fn band_module_recovers_malformed_module_log_from_phase_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        std::fs::write(
            temp.path().join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(temp.path().join("logband.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_band_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_module_log_dat(temp.path().join("logband.dat"))?,
            super::generated_band_module_log()
        );
        Ok(())
    }

    #[test]
    fn band_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        std::fs::write(temp.path().join("logband.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_band_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached logband.dat should fail through explicit BAND runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logband.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn band_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        let band_path = temp.path().join("bandstructure.dat");
        let kmesh_path = temp.path().join("kmesh.dat");
        let log_path = temp.path().join("logband.dat");
        write_bandstructure_dat(&band_path, &sample_bandstructure_dat())?;
        write_kmesh_dat(&kmesh_path, &sample_kmesh_dat())?;
        write_module_log_dat(&log_path, &sample_module_log())?;
        let expected_band = read_bandstructure_dat(&band_path)?;
        let expected_kmesh = read_kmesh_dat(&kmesh_path)?;
        let expected_log = read_module_log_dat(&log_path)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert!(has_cached_band_output(temp.path())?);
        assert_eq!(read_bandstructure_dat(&band_path)?, expected_band);
        assert_eq!(read_kmesh_dat(&kmesh_path)?, expected_kmesh);
        assert_eq!(read_module_log_dat(&log_path)?, expected_log);
        Ok(())
    }

    #[test]
    fn band_module_generates_missing_module_log_from_cached_bandstructure() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        let band_path = temp.path().join("bandstructure.dat");
        write_bandstructure_dat(&band_path, &sample_bandstructure_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let log = read_module_log_dat(temp.path().join("logband.dat"))?;
        assert_eq!(
            log.lines,
            vec![
                "Calculating band structure ...",
                "Solving band structure.",
                " Done with module: band structure.",
            ]
        );
        Ok(())
    }

    #[test]
    fn band_module_generates_missing_kmesh_from_reciprocal_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_band_input(temp.path(), true)?;
        write_bandstructure_dat(
            temp.path().join("bandstructure.dat"),
            &sample_bandstructure_dat(),
        )?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert_kmesh_row_close(
            data.rows[0],
            KmeshRow {
                index: 1,
                k_point: [0.831_446_454_055_273_6; 3],
                weight: 0.5,
                metadata: data.rows[0].metadata,
            },
            5.0e-4,
        );
        Ok(())
    }

    #[test]
    fn band_module_uses_disabled_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_band_dir()? else {
            crate::require_fixture!(
                "BAND reference test; generated KSPACE/Graphite reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("band.inp"), temp.path().join("band.inp"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(read_module_log_dat(temp.path().join("logband.dat"))?.is_empty());
        Ok(())
    }

    #[test]
    fn band_disabled_kmesh_matches_graphite_reference_when_present() -> Result<()> {
        let Some(zip_path) = reference_band_zip()? else {
            crate::require_fixture!(
                "BAND disabled kmesh reference test; generated Graphite zip not found"
            );
        };
        let Some(reference_dir) = reference_band_dir()? else {
            crate::require_fixture!(
                "BAND disabled kmesh reference test; generated Graphite handoff not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("BAND disabled kmesh reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        for name in ["band.inp", "reciprocal.inp"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected = parse_kmesh_dat(&String::from_utf8(unzip_reference_entry(
            &zip_path,
            "REFERENCE/kmesh.dat",
        )?)?)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert!(read_module_log_dat(temp.path().join("logband.dat"))?.is_empty());
        assert_kmesh_close(
            &read_kmesh_dat(temp.path().join("kmesh.dat"))?,
            &expected,
            6.0e-4,
        );
        Ok(())
    }

    #[test]
    fn band_generated_kmesh_matches_graphite_reference_when_present() -> Result<()> {
        let Some(zip_path) = reference_band_zip()? else {
            crate::require_fixture!("BAND kmesh reference test; generated Graphite zip not found");
        };
        let Some(reference_dir) = reference_band_dir()? else {
            crate::require_fixture!(
                "BAND kmesh reference test; generated Graphite handoff not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("BAND kmesh reference test; unzip command not found");
        }

        let reciprocal_text = std::fs::read_to_string(reference_dir.join("reciprocal.inp"))?;
        let reference_text =
            String::from_utf8(unzip_reference_entry(&zip_path, "REFERENCE/kmesh.dat")?)?;
        let reciprocal = ReciprocalInput::parse_str("reciprocal.inp", &reciprocal_text)?;
        let cell = reciprocal
            .cell
            .as_ref()
            .context("Graphite reciprocal.inp should contain a cell")?;
        let actual = kmesh_dat_from_reciprocal_cell(cell)?;
        let expected = parse_kmesh_dat(&reference_text)?;

        assert_kmesh_close(&actual, &expected, 6.0e-4);
        Ok(())
    }

    #[test]
    fn band_cr2gec_reference_handoff_uses_fms_lmaxph_basis_when_present() -> Result<()> {
        let Some((reference_dir, zip_path)) = reference_cr2gec_band_handoff()? else {
            crate::require_fixture!("BAND Cr2GeC basis test; reference handoff not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("BAND Cr2GeC basis test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        std::fs::copy(
            reference_dir.join("reciprocal.inp"),
            temp.path().join("reciprocal.inp"),
        )?;
        std::fs::copy(reference_dir.join("fms.inp"), temp.path().join("fms.inp"))?;
        std::fs::write(
            temp.path().join("phase.bin"),
            unzip_reference_entry(&zip_path, "REFERENCE/phase.bin")?,
        )?;
        write_band_input_custom(temp.path(), true, 6, false)?;
        let input = read_input(temp.path())?;
        let setup =
            build_complete_band_source_setup(temp.path(), &temp.path().join("phase.bin"), &input)?
                .context("Cr2GeC BAND reference setup should be complete")?;

        assert_eq!(setup.kspace_solver_basis.spin_channels, 1);
        assert_eq!(setup.kspace_energy.spin_count, 1);
        assert_eq!(
            setup.kspace_solver_basis.matrix_order,
            setup.kspace_angular.matrix_order_non_rel
        );
        assert_eq!(
            setup.kspace_solver_basis.matrix_order,
            setup.kspace_angular.matrix_order_rel
        );
        assert_eq!(setup.kspace_solver_basis.matrix_order, 128);
        assert_eq!(setup.k_path.mesh.point_count(), 18);
        assert_eq!(setup.search.energy_mesh.point_count(), 61);
        assert!(can_run_band_kspace_non_rel_source_solve(&setup));
        Ok(())
    }

    #[test]
    fn band_cr2gec_generated_bandstructure_matches_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_cr2gec_generated_band_output()? else {
            crate::require_fixture!("BAND Cr2GeC generated-output test; FEFF band run not found");
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "band.inp",
            "reciprocal.inp",
            "fms.inp",
            "global.inp",
            "phase.bin",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))
                .with_context(|| format!("failed to copy Cr2GeC BAND handoff {name}"))?;
        }
        let expected = read_bandstructure_dat(reference_dir.join("bandstructure.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let actual = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert!(
            cached_bandstructure_matches_source_output(&actual, &expected),
            "Cr2GeC source-generated bandstructure.dat did not match FEFF generated output"
        );
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(temp.path().join("logband.dat").is_file());
        Ok(())
    }

    fn write_band_input(work_dir: &Path, enabled: bool) -> Result<()> {
        write_band_input_with_nkp(work_dir, enabled, 2)
    }

    fn write_band_input_with_nkp(work_dir: &Path, enabled: bool, nkp: i32) -> Result<()> {
        write_band_input_custom(work_dir, enabled, nkp, false)
    }

    fn write_band_input_with_freeprop(
        work_dir: &Path,
        enabled: bool,
        freeprop: bool,
    ) -> Result<()> {
        write_band_input_custom(work_dir, enabled, 2, freeprop)
    }

    fn write_band_input_custom(
        work_dir: &Path,
        enabled: bool,
        nkp: i32,
        freeprop: bool,
    ) -> Result<()> {
        let input = BandInput {
            mband: if enabled { 1 } else { 0 },
            energy_mesh: BandEnergyMesh {
                emin: -5.0,
                emax: 10.0,
                estep: 0.25,
            },
            nkp,
            ikpath: 1,
            freeprop,
        };
        std::fs::write(work_dir.join("band.inp"), band_input_string(&input)?)?;
        Ok(())
    }

    fn write_band_source_handoffs(work_dir: &Path) -> Result<()> {
        write_band_input(work_dir, true)?;
        std::fs::write(
            work_dir.join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(
            work_dir.join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;
        Ok(())
    }

    fn write_one_spin_rel_band_source_handoffs(work_dir: &Path, freeprop: bool) -> Result<()> {
        write_band_input_with_freeprop(work_dir, true, freeprop)?;
        std::fs::write(
            work_dir.join("phase.bin"),
            phase_bin_string(&sample_band_phase_bin())?,
        )?;
        std::fs::write(
            work_dir.join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;
        write_global_input(work_dir, 1)?;
        Ok(())
    }

    fn write_two_spin_non_degenerate_band_source_handoffs(
        work_dir: &Path,
        freeprop: bool,
    ) -> Result<()> {
        write_band_input_with_freeprop(work_dir, true, freeprop)?;
        let mut phase = sample_two_spin_degenerate_band_phase_bin();
        phase.reference_energy[(0, 1)] += Complex64::new(0.01, 0.0);
        std::fs::write(work_dir.join("phase.bin"), phase_bin_string(&phase)?)?;
        std::fs::write(
            work_dir.join("reciprocal.inp"),
            reciprocal_input_string(&sample_reciprocal_input(8))?,
        )?;
        Ok(())
    }

    fn assert_band_module_regenerates_stale_two_spin_bandstructure(freeprop: bool) -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_two_spin_non_degenerate_band_source_handoffs(temp.path(), freeprop)?;
        run_in_dir(temp.path())?;
        let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        let mut stale = expected.clone();
        stale
            .rows
            .iter_mut()
            .find(|row| !row.bands.is_empty())
            .context("two-spin source bandstructure should contain at least one band")?
            .bands[0] += 0.5;
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let regenerated = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_eq!(regenerated, expected);
        if regenerated.max_band_count() > 0 {
            assert_source_bandstructure_generated(&regenerated);
        } else {
            assert!(regenerated.k_point_count() > 0);
            assert_eq!(regenerated.header_lines.len(), 3);
            assert!(regenerated.header_lines[0].contains("k-points"));
            assert!(regenerated.header_lines[1].contains("energy points"));
            assert!(regenerated.header_lines[2].contains("number of bands"));
            assert!(
                regenerated
                    .rows
                    .iter()
                    .all(|row| row.k_point.iter().all(|value| value.is_finite()))
            );
        }
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(temp.path().join("logband.dat").is_file());
        Ok(())
    }

    fn assert_band_module_regenerates_stale_one_spin_rel_bandstructure(
        freeprop: bool,
    ) -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_one_spin_rel_band_source_handoffs(temp.path(), freeprop)?;
        let input = read_input(temp.path())?;
        let setup =
            build_complete_band_source_setup(temp.path(), &temp.path().join("phase.bin"), &input)?
                .context("one-spin rel BAND source setup should be complete")?;
        assert_eq!(setup.kspace_solver_basis.spin_selector, 1);

        run_in_dir(temp.path())?;
        let expected = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        let source = if freeprop {
            bandstructure_dat_from_kspace_free_propagation_rel_handoffs(&setup)?
        } else {
            bandstructure_dat_from_kspace_rel_handoffs(&setup)?
        };
        assert!(
            cached_bandstructure_matches_source_output(&expected, &source),
            "generated one-spin rel BAND output did not match source solve"
        );
        let mut stale = expected.clone();
        stale
            .rows
            .first_mut()
            .context("one-spin rel source bandstructure should contain at least one k-point")?
            .k_point[0] += 0.5;
        write_bandstructure_dat(temp.path().join("bandstructure.dat"), &stale)?;

        assert!(!has_cached_band_output(temp.path())?);
        assert!(has_supported_pre_solver_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let regenerated = read_bandstructure_dat(temp.path().join("bandstructure.dat"))?;
        assert_eq!(regenerated, expected);
        if regenerated.max_band_count() > 0 {
            assert_source_bandstructure_generated(&regenerated);
        } else {
            assert!(regenerated.k_point_count() > 0);
            assert_eq!(regenerated.header_lines.len(), 3);
            assert!(regenerated.header_lines[0].contains("k-points"));
            assert!(regenerated.header_lines[1].contains("energy points"));
            assert!(regenerated.header_lines[2].contains("number of bands"));
            assert!(
                regenerated
                    .rows
                    .iter()
                    .all(|row| row.k_point.iter().all(|value| value.is_finite()))
            );
        }
        assert!(temp.path().join("kmesh.dat").is_file());
        assert!(temp.path().join("logband.dat").is_file());
        Ok(())
    }

    fn write_global_input(work_dir: &Path, ispin: i32) -> Result<()> {
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&sample_global_input(ispin))?,
        )?;
        Ok(())
    }

    fn sample_global_input(ispin: i32) -> GlobalInput {
        GlobalInput {
            cfaverage: CfAverage {
                nabs: 1,
                iphabs: 0,
                rclabs: 0.0,
            },
            control: GlobalControl {
                ipol: 0,
                ispin,
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
            polarization_tensor: [[0.0; 6]; 3],
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

    fn sample_bandstructure_dat() -> BandstructureDatData {
        BandstructureDatData {
            header_lines: vec![
                " # grid of            2  k-points.".to_string(),
                " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000".to_string(),
                " # Found between            1  and            2  number of bands.".to_string(),
            ],
            rows: vec![
                BandstructureRow {
                    index: 1,
                    k_point: [0.0, 0.5, 0.25],
                    bands: Array1::from_vec(vec![-5.0, 1.25]),
                },
                BandstructureRow {
                    index: 2,
                    k_point: [0.5, 0.25, 0.0],
                    bands: Array1::from_vec(vec![0.75]),
                },
            ],
        }
    }

    fn assert_source_bandstructure_generated(data: &BandstructureDatData) {
        assert!(data.k_point_count() > 0);
        assert_eq!(data.header_lines.len(), 3);
        assert!(data.header_lines[0].contains("k-points"));
        assert!(data.header_lines[1].contains("energy points"));
        assert!(data.header_lines[2].contains("number of bands"));
        assert!(data.max_band_count() > 0);
        for (index, row) in data.rows.iter().enumerate() {
            assert_eq!(row.index, (index + 1) as i32);
            assert!(row.k_point.iter().all(|value| value.is_finite()));
            assert!(row.bands.iter().all(|value| value.is_finite()));
        }
    }

    fn sample_kmesh_dat() -> KmeshDatData {
        KmeshDatData {
            rows: vec![
                KmeshRow {
                    index: 1,
                    k_point: [0.0, 0.5, 0.25],
                    weight: 0.75,
                    metadata: Some(KmeshMetadata {
                        requested_points: 100,
                        irreducible_points: 2,
                        divisions: [4, 5, 6],
                    }),
                },
                KmeshRow {
                    index: 2,
                    k_point: [0.5, 0.25, 0.0],
                    weight: 0.25,
                    metadata: None,
                },
            ],
        }
    }

    fn sample_reciprocal_input(total_kpoints: i32) -> ReciprocalInput {
        ReciprocalInput {
            ispace: 0,
            cell: Some(ReciprocalCell {
                lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                volume_scale: -1.0,
                imaginary_energy: 0.0,
                core_hole_strength: 1.0,
                lattice_name: "P".to_string(),
                space_group_hm: "Pm-3m".to_string(),
                space_group: 221,
                atom_count: 1,
                absorber: 1,
                core_hole: 1,
                k_mesh: ReciprocalKMesh {
                    total: total_kpoints,
                    x: total_kpoints,
                    y: 0,
                    z: 0,
                    kind: 3,
                    use_symmetry: false,
                },
                positions: vec![[0.0, 0.0, 0.0]],
                potentials: vec![0],
                labels: vec!["Cu".to_string()],
                stretch: [0.0, 0.0, 0.0],
            }),
        }
    }

    fn sample_band_phase_bin() -> PhaseBinData {
        let spin_count = 1;
        let energy_count = 4;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 4,
            fermi_index: 1,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                edge_energy: 0.0,
            },
            energy_grid: Array1::from_vec(vec![
                Complex64::new(-0.2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.4, 0.0),
            ]),
            reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
                Complex64::new(-0.1 + 0.01 * energy as f64, 0.0)
            }),
            potentials: vec![PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, l_slot, _)| {
                        Complex64::new(0.01 * energy as f64 + 0.1 * l_slot as f64, 0.0)
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_two_spin_degenerate_band_phase_bin() -> PhaseBinData {
        let spin_count = 2;
        let energy_count = 4;
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 4,
            fermi_index: 1,
            pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                edge_energy: 0.0,
            },
            energy_grid: Array1::from_vec(vec![
                Complex64::new(-0.2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.4, 0.0),
            ]),
            reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
                Complex64::new(-0.1 + 0.01 * energy as f64, 0.0)
            }),
            potentials: vec![PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, l_slot, _)| {
                        Complex64::new(0.01 * energy as f64 + 0.1 * l_slot as f64, 0.0)
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating band structure ...".to_string(),
                " Done with module: band structure.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\r\n".to_string()],
        }
    }

    fn reference_band_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/KSPACE/Graphite"));
        Ok(reference.filter(|path| path.join("band.inp").is_file()))
    }

    fn reference_band_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/KSPACE/Graphite/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_cr2gec_band_handoff() -> Result<Option<(PathBuf, PathBuf)>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let Some(root) = manifest_dir.parent().and_then(Path::parent) else {
            return Ok(None);
        };
        let reference_dir = root.join("reference-work/golden/KSPACE/Cr2GeC");
        let zip_path = reference_dir.join("REFERENCE.zip");
        Ok(
            (reference_dir.join("reciprocal.inp").is_file() && zip_path.is_file())
                .then_some((reference_dir, zip_path)),
        )
    }

    fn reference_cr2gec_generated_band_output() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let Some(root) = manifest_dir.parent().and_then(Path::parent) else {
            return Ok(None);
        };
        let tmp_dir = root.join("reference-work/tmp");
        if !tmp_dir.is_dir() {
            return Ok(None);
        }

        let required = [
            "band.inp",
            "reciprocal.inp",
            "fms.inp",
            "global.inp",
            "phase.bin",
            "bandstructure.dat",
        ];
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(&tmp_dir)
            .with_context(|| format!("failed to read {}", tmp_dir.display()))?
        {
            let entry = entry.with_context(|| format!("failed to read {}", tmp_dir.display()))?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("feff-band-cr2gec.")
                && required.iter().all(|file| path.join(file).is_file())
            {
                candidates.push(path);
            }
        }
        candidates.sort();
        Ok(candidates.pop())
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to unzip {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            bail!(
                "failed to unzip {entry} from {}: {}",
                zip_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(output.stdout)
    }

    fn assert_kmesh_close(actual: &KmeshDatData, expected: &KmeshDatData, tolerance: f64) {
        assert_eq!(actual.rows.len(), expected.rows.len());
        for (&actual, &expected) in actual.rows.iter().zip(expected.rows.iter()) {
            assert_kmesh_row_close(actual, expected, tolerance);
        }
    }

    fn assert_kmesh_row_close(actual: KmeshRow, expected: KmeshRow, tolerance: f64) {
        assert_eq!(actual.index, expected.index);
        assert_eq!(actual.metadata, expected.metadata);
        assert!((actual.weight - expected.weight).abs() <= tolerance);
        for (actual, expected) in actual.k_point.iter().zip(expected.k_point.iter()) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }
}
