use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
#[cfg(feature = "full")]
use ndarray::Array6;
use ndarray::{
    Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder,
};
use num_complex::Complex32;
use rayon::prelude::*;
use refeff_core::{
    Complex, FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, FmsAtom, FmsDriverSetupInput,
    FmsFullPotentialLuInput, FmsHubbardFullScatteringTransformInput,
    FmsHubbardScatteringTransformInput, FmsHubbardTMatrixTableInput,
    FmsHubbardTMatrixTransformInput, FmsSpinFreePropagatorMatrixInput, FmsYprepClusterInput,
    LdosFmsdosTraceGridInput, MkgtrGreenTraceInput, MkgtrJasGreenTraceInput, MkgtrJasQPairMode,
    MkgtrJasTransition, PolarizationTensorMode, Real, SpringDynamicalMatrix,
    SpringDynamicalMatrixInput, SpringEquationOfMotionInput, SpringInput, SpringRecursionInput,
    SpringRecursionState, TransitionBMatrixInput, classical_debye_waller_factor,
    core_hole_quantum_numbers, dmdw_debye_waller_factors_from_poles, dmdw_lanczos_coefficients,
    dmdw_lanczos_pole_spectrum, dmdw_mass_weighted_dynamical_matrix, dmdw_path_motion,
    dmdw_project_seed_vector, dmdw_rigid_body_projection_modes,
    equation_of_motion_debye_waller_factor, fms_driver_setup, fms_full_potential_lu_scattering,
    fms_hubbard_back_transform_full_scattering, fms_hubbard_back_transform_scattering,
    fms_hubbard_t_matrix_table, fms_hubbard_transform_t_matrix, fms_spin_free_propagator_matrix,
    fms_spin_pair_tables, fms_yprep_cluster, fms_yprep_geometry, ldos_fmsdos_trace_grid,
    legendre_normalization_table, mkgtr_green_trace, mkgtr_jas_green_trace, parse_spring_input,
    quantum_debye_waller_factor, recursion_debye_waller_factor, screen_fms_cluster_green_trace,
    sort_representative_atoms, spin_orbit_coupling_tables, spring_dynamical_matrix,
    transition_b_matrix, update_spring_recursion_state,
};
// `fms::{FmsRealSpaceEnergyPoint, FmsRealSpacePlanInput, fms_real_space_plan,
// fms_real_space_spectrum}` are not yet re-exported from the `refeff_core` crate
// root, so pull them in through the `fms` module path directly.
use refeff_core::fms::{
    FmsRealSpaceEnergyPoint, FmsRealSpacePlanInput, FmsReciprocalAccumulator,
    FmsReciprocalCoreHoleInput, FmsReciprocalPlan, fms_real_space_plan, fms_real_space_spectrum,
    fms_reciprocal_apply_core_hole,
};
use refeff_io::{
    DimensionsDat, DmdwCalculation, DmdwInput, EelsInput, FmsBinData, FmsCluster, FmsControl,
    FmsDebye, FmsInput, FmsKspaceStaticHandoffSetup, FmslBinData, GeomDat, GgDatData, GgDatSection,
    GlobalInput, GtrBinData, GtrDatData, GtrlDatData, HubbardAphaseBinData, HubbardInput,
    HubbardLdosGtrMBinData, HubbardTransformationBinData, LdosInput, ModuleLogData, PhaseBinData,
    PotBinData, PotInput, PotScfFmsSourceGridHandoff, PotScfFmsSourceGridHandoffInput,
    ReciprocalCell, ReciprocalInput, RhorrpGgDiagBinData, RhorrpGgSliceBinData,
    ScreenFmsClusterGreenHandoff, fms_bin_string, fms_kspace_ewald_energy_tables_from_handoff,
    fms_kspace_non_rel_structure_factor, fms_kspace_setup_from_handoffs,
    fms_kspace_setup_from_static_handoffs, fms_kspace_static_setup_from_handoffs,
    fms_kspace_t_matrix, genfmt_jas_q_angles_from_handoffs,
    genfmt_jas_transition_indices_from_handoffs, gg_dat_string, gtr_bin_from_ldos_trace_grid,
    gtr_dat_string, pot_scf_fms_source_grid_handoff, read_aphase_hubbard_bin_inferred, read_dym,
    read_fms_bin, read_fmsl_bin, read_gg_bin, read_gg_dat, read_gtr_bin, read_gtr_dat,
    read_gtrl_dat, read_module_log_dat, read_phase_bin, read_rhorrp_gg_diag_bin,
    read_rhorrp_gg_slice_bin, read_transformation_hubbard_bin_inferred,
    read_v_hubbard_bin_inferred, write_fms_bin, write_fmsl_bin, write_gg_bin, write_gg_dat,
    write_gtr_bin, write_gtr_dat, write_gtrl_dat, write_hubbard_ldos_gtr_m_bin,
    write_module_log_dat, write_rhorrp_gg_diag_bin, write_rhorrp_gg_slice_bin,
    write_transformation_hubbard_bin,
};
#[cfg(feature = "full")]
use refeff_io::{HubbardLdosGtrOffBinData, write_hubbard_ldos_gtr_off_bin};

use crate::work_dir_for_input;

const FMS_DMDW_MATCH_TOLERANCE_BOHR: Real = 0.01;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenFmsSourceGridInput<'a> {
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    pub wave_numbers_bohr: ArrayView1<'a, Complex>,
    pub phase_shifts: ArrayView3<'a, Complex>,
    pub angular_count: usize,
    pub cluster_radius_angstrom: f64,
    pub direct_cutoff_angstrom: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PotScfFmsSourceGridInput<'a> {
    pub pot: &'a PotInput,
    pub energy_grid_hartree: ArrayView1<'a, Complex>,
    pub reference_energies_hartree: ArrayView2<'a, Complex>,
    pub phase_shifts: ArrayView3<'a, Complex>,
    pub angular_count: usize,
}

#[derive(Debug, Default)]
pub(crate) struct PotScfFmsPipelineCache {
    reciprocal: Option<PotScfReciprocalGeometryCache>,
}

#[derive(Debug)]
struct PotScfReciprocalGeometryCache {
    source_path: PathBuf,
    source_bytes: Vec<u8>,
    effective_cell: ReciprocalCell,
    global_lmax: usize,
    max_potential: usize,
    potential_count: usize,
    initial_eta: f64,
    static_setup: Arc<FmsKspaceStaticHandoffSetup>,
}

impl PotScfFmsPipelineCache {
    fn validate_reciprocal_snapshot(
        &self,
        source_path: &Path,
        source_bytes: Option<&[u8]>,
    ) -> Result<()> {
        let Some(cached) = self.reciprocal.as_ref() else {
            return Ok(());
        };
        ensure!(
            cached.source_path == source_path
                && source_bytes.is_some_and(|bytes| bytes == cached.source_bytes),
            "reciprocal.inp changed during active POT SCF pipeline"
        );
        Ok(())
    }

    fn reciprocal_static_setup(
        &mut self,
        source_path: &Path,
        source_bytes: &[u8],
        effective_cell: &ReciprocalCell,
        global_lmax: usize,
        max_potential: usize,
        potential_count: usize,
    ) -> Result<Arc<FmsKspaceStaticHandoffSetup>> {
        if let Some(cached) = self.reciprocal.as_ref() {
            ensure!(
                cached.source_path == source_path && cached.source_bytes == source_bytes,
                "reciprocal.inp changed during active POT SCF pipeline"
            );
            ensure!(
                cached.effective_cell == *effective_cell
                    && cached.global_lmax == global_lmax
                    && cached.max_potential == max_potential
                    && cached.potential_count == potential_count
                    && cached.initial_eta == cached.static_setup.initial_ewald_tables.eta,
                "reciprocal FMS geometry key changed during active POT SCF pipeline"
            );
            return Ok(Arc::clone(&cached.static_setup));
        }

        let probe = ndarray::arr1(&[-3.0_f64, 3.0_f64]);
        let static_setup = Arc::new(
            fms_kspace_static_setup_from_handoffs(effective_cell, probe.view(), global_lmax, 1, 0)
                .context("failed to prepare reusable POT reciprocal FMS KSPACE geometry")?,
        );
        let initial_eta = static_setup.initial_ewald_tables.eta;
        self.reciprocal = Some(PotScfReciprocalGeometryCache {
            source_path: source_path.to_path_buf(),
            source_bytes: source_bytes.to_vec(),
            effective_cell: effective_cell.clone(),
            global_lmax,
            max_potential,
            potential_count,
            initial_eta,
            static_setup: Arc::clone(&static_setup),
        });
        Ok(static_setup)
    }
}

pub(crate) fn screen_fms_source_angular_count(
    work_dir: &Path,
    phase: &PhaseBinData,
) -> Result<Option<usize>> {
    screen_fms_source_angular_count_for_potential_count(work_dir, phase.potential_count())
}

pub(crate) fn screen_fms_source_angular_count_for_potential_count(
    work_dir: &Path,
    potential_count: usize,
) -> Result<Option<usize>> {
    if !work_dir.join("fms.inp").is_file() {
        return Ok(None);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(None);
    };
    let max_potential = potential_count
        .checked_sub(1)
        .context("SCREEN FMS generation requires at least one potential")?;
    let global_lmax = global_fms_lmax(&input, max_potential)?;
    Ok(Some(
        global_lmax
            .checked_add(1)
            .context("SCREEN FMS angular count is too large")?,
    ))
}

/// Run only the FEFF FMS solver beside the requested input.
pub(crate) fn run_fms_for_input(input: &Path) -> Result<usize> {
    run_fms_in_dir(work_dir_for_input(input))
}

/// Run only the FEFF MKGTR trace stage beside the requested input.
pub(crate) fn run_mkgtr_for_input(input: &Path) -> Result<usize> {
    run_mkgtr_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF FMS/MKGTR run can be satisfied from existing caches.
#[cfg(test)]
pub(crate) fn has_cached_fms_output(work_dir: &Path) -> Result<bool> {
    let outputs = cached_output_paths(work_dir)?;
    if !work_dir.join("fms.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !fms_enabled(&input) {
        return Ok(false);
    }
    if !outputs.is_empty() {
        if declared_fms_source_handoff_has_error(work_dir, &input) {
            return Ok(false);
        }
        return Ok(true);
    }
    can_generate_gg_from_source_handoffs(work_dir, &input)
}

/// Whether the FMS solver matrix already exists before running the stage.
pub(crate) fn has_cached_fms_solver_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("fms.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !fms_enabled(&input) || declared_fms_source_handoff_has_error(work_dir, &input) {
        return Ok(false);
    }
    Ok(cached_gg_output(&cached_output_paths(work_dir)?).is_some())
}

/// Whether the FMS matrix solver can run from a cached `gg` matrix or complete
/// phase/geometry source handoffs.
pub(crate) fn has_runnable_fms_solver(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("fms.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !fms_enabled(&input) || declared_fms_source_handoff_has_error(work_dir, &input) {
        return Ok(false);
    }
    if cached_gg_output(&cached_output_paths(work_dir)?).is_some() {
        return Ok(true);
    }
    can_generate_gg_from_source_handoffs(work_dir, &input)
}

/// Whether the MKGTR spectrum metadata and trace outputs already exist.
pub(crate) fn has_cached_mkgtr_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("fms.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !fms_enabled(&input) {
        return Ok(false);
    }
    let fms_path = work_dir.join("fms.bin");
    let gtr_path = work_dir.join("gtr.dat");
    if !fms_path.is_file() || !gtr_path.is_file() {
        return Ok(false);
    }
    if let Some(selectors) = mkgtr_eels_polarization_selectors(work_dir)? {
        let Ok(fms) = read_fms_bin(&fms_path) else {
            return Ok(false);
        };
        if fms.spectrum_count() != selectors.len() || read_gtr_dat(&gtr_path).is_err() {
            return Ok(false);
        }
    }
    let global_path = work_dir.join("global.inp");
    if !global_path.is_file() {
        return Ok(true);
    }
    let global = read_global_input(work_dir)?;
    if global.control.do_nrixs == 1 && global.control.ldecmx >= 0 {
        return Ok(work_dir.join("fmsl.bin").is_file() && work_dir.join("gtrl.dat").is_file());
    }
    Ok(true)
}

/// Whether downstream source-generated stages should wait for an active FMS
/// stage that is not currently satisfiable from caches or Rust handoffs.
#[cfg(feature = "full")]
pub(crate) fn blocks_downstream_source_generation(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("fms.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(true);
    };
    if !fms_enabled(&input) {
        return Ok(false);
    }
    Ok(!has_runnable_fms_solver(work_dir)?)
}

#[cfg(not(feature = "full"))]
pub(crate) fn blocks_downstream_source_generation(_work_dir: &Path) -> Result<bool> {
    Ok(false)
}

/// Run the supported FEFF FMS/MKGTR path from existing handoff files.
///
/// This preserves cached FEFF directories by validating and re-rendering typed
/// `gg.bin`/`gg.dat`, optional `gg_slice.bin`/`gg_diag.bin`, `fms.bin`,
/// `fmsl.bin`, `gtr.dat`, `gtrNN.bin`, `gtrl.dat`, and optional `log3.dat`
/// diagnostic handoffs, plus cached `transformation_hubbard.bin` when
/// `hubbard.inp` and `phase.bin` dimensions are present. When a cached
/// absorber `gg` matrix exists with `phase.bin` and non-NRIXS `global.inp`,
/// the port also folds the Green's functions through MKGTR to generate missing
/// `fms.bin` and `gtr.dat` files. For supported source handoffs, it can also
/// build missing absorber `gg` matrices from the ported real-space FMS solver,
/// including FEFF `SIG2`-style global damping and `idwopt=0` correlated Debye
/// damping, `idwopt=3` classical Debye damping, or `idwopt=4` `sig2.dat`
/// pair damping, or `idwopt=5` dynamical-matrix damping.
#[cfg(test)]
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !fms_enabled(&input) {
        return Ok(0);
    }
    validate_declared_fms_source_handoff_files(work_dir, &input)?;

    let mut outputs = cached_output_paths(work_dir)?;
    let mut generated_source = None;
    if outputs.is_empty() {
        generated_source = generate_gg_outputs_from_source_handoffs(work_dir, &input)?;
        outputs = cached_output_paths(work_dir)?;
        if outputs.is_empty() {
            bail!(
                "FMS Green's-function generation requires cached FMS output or supported phase.bin/geom.dat/global.inp source handoffs"
            );
        }
    } else if let Some(metadata) =
        regenerate_stale_gg_outputs_from_source_handoffs(work_dir, &input, &outputs)?
    {
        generated_source = Some(metadata);
        outputs = cached_output_paths(work_dir)?;
    } else if let Some(metadata) =
        recover_malformed_gg_outputs_from_source_handoffs(work_dir, &input, &outputs)?
    {
        generated_source = Some(metadata);
        outputs = cached_output_paths(work_dir)?;
    }
    if generated_source.is_some() {
        invalidate_derived_mkgtr_outputs(work_dir)?;
        outputs = cached_output_paths(work_dir)?;
    }
    repair_malformed_gg_companion_outputs(&outputs)?;

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
            CachedOutputKind::GgSliceBin => {
                let data = read_rhorrp_gg_slice_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_gg_slice_bin(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
            }
            CachedOutputKind::GgDiagBin => {
                let data = read_rhorrp_gg_diag_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_gg_diag_bin(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
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

    let generated_gg_companion = generate_gg_companion_outputs(work_dir, &outputs)?;
    let hubbard_transformation = write_optional_hubbard_transformation_cache(work_dir)?;
    let generated = generate_mkgtr_outputs_from_cached_gg(work_dir, &input, &outputs)?;
    let log_path = work_dir.join("log3.dat");
    let log_count = if log_path.is_file() {
        write_optional_module_log(&log_path)?
    } else if let Some(metadata) = generated_source {
        write_generated_fms_module_log(&log_path, &input, &metadata, generated)?
    } else {
        write_generated_cached_fms_module_log(&log_path, generated)?
    };

    Ok(outputs.len() + generated_gg_companion + hubbard_transformation + generated + log_count)
}

/// Run the FMS matrix solver without also executing MKGTR.
///
/// FEFF's `fms` executable produces the `gg*` Green-function matrices. The
/// separate [`run_mkgtr_in_dir`] stage folds those matrices into spectrum
/// traces. Keeping this boundary explicit makes the standalone compatibility
/// binaries behave like the upstream executables while [`run_in_dir`] remains
/// the combined scheduler path.
pub(crate) fn run_fms_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !fms_enabled(&input) {
        return Ok(0);
    }
    validate_declared_fms_source_handoff_files(work_dir, &input)?;

    let mut outputs = cached_output_paths(work_dir)?;
    let mut generated_source = None;
    if cached_gg_output(&outputs).is_none() {
        generated_source = generate_gg_outputs_from_source_handoffs(work_dir, &input)?;
        outputs = cached_output_paths(work_dir)?;
        if cached_gg_output(&outputs).is_none() {
            bail!(
                "FMS Green's-function generation requires cached FMS output or supported phase.bin/geom.dat/global.inp source handoffs"
            );
        }
    } else if let Some(metadata) =
        regenerate_stale_gg_outputs_from_source_handoffs(work_dir, &input, &outputs)?
    {
        generated_source = Some(metadata);
        outputs = cached_output_paths(work_dir)?;
    } else if let Some(metadata) =
        recover_malformed_gg_outputs_from_source_handoffs(work_dir, &input, &outputs)?
    {
        generated_source = Some(metadata);
        outputs = cached_output_paths(work_dir)?;
    }
    if generated_source.is_some() {
        invalidate_derived_mkgtr_outputs(work_dir)?;
        outputs = cached_output_paths(work_dir)?;
    }
    repair_malformed_gg_companion_outputs(&outputs)?;

    let solver_outputs: Vec<_> = outputs
        .iter()
        .filter(|output| output.kind.is_fms_solver_output())
        .collect();
    for output in &solver_outputs {
        match output.kind {
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
            CachedOutputKind::GgSliceBin => {
                let data = read_rhorrp_gg_slice_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_gg_slice_bin(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
            }
            CachedOutputKind::GgDiagBin => {
                let data = read_rhorrp_gg_diag_bin(&output.path)
                    .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_rhorrp_gg_diag_bin(&output.path, &data)
                    .with_context(|| format!("failed to write {}", output.path.display()))?;
            }
            _ => bail!("internal FMS error: unexpected trace output in solver set"),
        }
    }

    let generated_companions = generate_gg_companion_outputs(work_dir, &outputs)?;
    let hubbard_transformation = write_optional_hubbard_transformation_cache(work_dir)?;
    let log_path = work_dir.join("log3.dat");
    let log_count = if log_path.is_file() {
        write_optional_module_log(&log_path)?
    } else if let Some(metadata) = generated_source {
        write_generated_fms_module_log(&log_path, &input, &metadata, 0)?
    } else {
        write_generated_cached_fms_module_log(&log_path, 0)?
    };
    Ok(solver_outputs.len() + generated_companions + hubbard_transformation + log_count)
}

/// A regenerated `gg` matrix changes every MKGTR projection derived from it.
///
/// Leaving an older `fms.bin`/`gtr.dat` beside the refreshed matrix makes the
/// scheduler treat the stale projection as complete. Remove only those
/// reproducible derived handoffs so the combined path or the following MKGTR
/// stage rebuilds them from the new matrix.
fn invalidate_derived_mkgtr_outputs(work_dir: &Path) -> Result<()> {
    for name in ["fms.bin", "gtr.dat", "fmsl.bin", "gtrl.dat"] {
        let path = work_dir.join(name);
        if path.is_file() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to invalidate stale {}", path.display()))?;
        }
    }
    Ok(())
}

/// Run MKGTR against an existing FMS Green-function matrix.
pub(crate) fn run_mkgtr_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !fms_enabled(&input) {
        return Ok(0);
    }
    let mut outputs = cached_output_paths(work_dir)?;
    if cached_gg_output(&outputs).is_none() {
        bail!("MKGTR requires gg.bin or gg.dat from the FMS stage");
    }

    // EELS changes the ordinary MKGTR projection from the zero tensor in
    // global.inp to one Cartesian transition tensor per requested selector.
    // Rebuild before validating cached trace files so a stale or truncated
    // single-spectrum cache remains repairable from the authoritative gg
    // matrix and source handoffs.
    generate_mkgtr_outputs_from_cached_gg(work_dir, &input, &outputs)?;
    outputs = cached_output_paths(work_dir)?;
    validate_requested_mkgtr_eels_outputs(work_dir)?;

    let fms_metadata = outputs
        .iter()
        .any(|output| output.kind == CachedOutputKind::FmslBin)
        .then(|| {
            let path = work_dir.join("fms.bin");
            read_fms_bin(&path).with_context(|| format!("failed to read {}", path.display()))
        })
        .transpose()?;
    let trace_outputs: Vec<_> = outputs
        .iter()
        .filter(|output| output.kind.is_mkgtr_output())
        .collect();
    for output in &trace_outputs {
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
                let data = read_fmsl_bin(
                    &output.path,
                    metadata.pad_width,
                    metadata.energy_count,
                    decomposition_channel(&input)?,
                )
                .with_context(|| format!("failed to read {}", output.path.display()))?;
                write_fmsl_cache(&output.path, &data)?;
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
            _ => bail!("internal MKGTR error: unexpected solver output in trace set"),
        }
    }

    let log_count = ensure_mkgtr_module_log(&work_dir.join("log3.dat"))?;
    Ok(trace_outputs.len() + log_count)
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

fn read_global_input(work_dir: &Path) -> Result<GlobalInput> {
    let global_path = work_dir.join("global.inp");
    let global_text = std::fs::read_to_string(&global_path)
        .with_context(|| format!("failed to read {}", global_path.display()))?;
    GlobalInput::parse_str(&global_path, &global_text)
        .with_context(|| format!("failed to parse {}", global_path.display()))
}

fn read_geom_dat(work_dir: &Path) -> Result<GeomDat> {
    let geom_path = work_dir.join("geom.dat");
    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))
}

fn read_hubbard_input(work_dir: &Path) -> Result<HubbardInput> {
    let hubbard_path = work_dir.join("hubbard.inp");
    let hubbard_text = std::fs::read_to_string(&hubbard_path)
        .with_context(|| format!("failed to read {}", hubbard_path.display()))?;
    HubbardInput::parse_str(&hubbard_path, &hubbard_text)
        .with_context(|| format!("failed to parse {}", hubbard_path.display()))
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

fn write_optional_hubbard_transformation_cache(work_dir: &Path) -> Result<usize> {
    let transformation_path = work_dir.join("transformation_hubbard.bin");
    if !transformation_path.is_file() {
        return Ok(0);
    }
    if !work_dir.join("hubbard.inp").is_file() || !work_dir.join("phase.bin").is_file() {
        return Ok(0);
    }

    let hubbard = read_hubbard_input(work_dir)?;
    let hubbard_l = usize::try_from(hubbard.l)
        .context("transformation_hubbard.bin requires nonnegative l_hubbard")?;
    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let data = read_transformation_hubbard_bin_inferred(
        &transformation_path,
        hubbard_l,
        phase.potential_count(),
    )
    .with_context(|| format!("failed to read {}", transformation_path.display()))?;
    write_transformation_hubbard_bin(&transformation_path, &data)
        .with_context(|| format!("failed to write {}", transformation_path.display()))?;
    Ok(1)
}

fn can_generate_gg_from_source_handoffs(work_dir: &Path, input: &FmsInput) -> Result<bool> {
    if !supported_source_fms_controls(input) || !required_fms_source_handoffs_present(work_dir) {
        return Ok(false);
    }
    if input.control.idwopt == 4 && !work_dir.join("sig2.dat").is_file() {
        return Ok(false);
    }
    if matches!(input.control.idwopt, 1 | 2) && !work_dir.join("spring.inp").is_file() {
        return Ok(false);
    }
    if input.control.idwopt == 5 && !can_read_fms_dmdw_handoffs(work_dir)? {
        return Ok(false);
    }

    let phase_path = work_dir.join("phase.bin");
    let Ok(phase) = read_phase_bin(&phase_path) else {
        return Ok(false);
    };
    if !phase_supports_fms_lmax(input, &phase) {
        return Ok(false);
    }
    if active_hubbard_fms_source_requested(work_dir)? {
        return can_generate_active_hubbard_gg_from_source_handoffs(work_dir, input, &phase);
    }
    Ok(true)
}

fn declared_fms_source_handoff_has_error(work_dir: &Path, input: &FmsInput) -> bool {
    validate_declared_fms_source_handoff_files(work_dir, input).is_err()
}

fn validate_declared_fms_source_handoff_files(work_dir: &Path, input: &FmsInput) -> Result<()> {
    if !supported_source_fms_controls(input) || !required_fms_source_handoffs_present(work_dir) {
        return Ok(());
    }

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    read_geom_dat(work_dir)?;
    read_global_input(work_dir)?;
    if let Some(reciprocal) = read_optional_fms_reciprocal_input(work_dir)? {
        match reciprocal.ispace {
            0 => {
                let cell = reciprocal
                    .cell
                    .as_ref()
                    .context("reciprocal.inp ispace=0 requires a reciprocal cell block")?;
                ensure!(
                    !cell.k_mesh.use_symmetry,
                    "reciprocal FMS symmetry reduction is unsupported without FEFF rotation tables"
                );
            }
            1 => {}
            ispace => bail!("reciprocal.inp ispace must be 0 or 1, got {ispace}"),
        }
    }

    if active_hubbard_fms_source_requested(work_dir)? {
        let handoffs = read_active_hubbard_fms_source_handoffs(work_dir, &phase)?;
        validate_active_hubbard_fms_source_handoffs(input, &phase, &handoffs)?;
    }
    Ok(())
}

fn required_fms_source_handoffs_present(work_dir: &Path) -> bool {
    work_dir.join("phase.bin").is_file()
        && work_dir.join("geom.dat").is_file()
        && work_dir.join("global.inp").is_file()
}

fn active_hubbard_fms_source_requested(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("hubbard.inp").is_file() {
        return Ok(false);
    }
    Ok(read_hubbard_input(work_dir)?.mldos_hubb == 2 && work_dir.join("v_hubbard.bin").is_file())
}

fn can_generate_active_hubbard_gg_from_source_handoffs(
    work_dir: &Path,
    input: &FmsInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    if !work_dir.join("aphase_hubbard.bin").is_file()
        || !work_dir.join("transformation_hubbard.bin").is_file()
    {
        return Ok(false);
    }
    let handoffs = read_active_hubbard_fms_source_handoffs(work_dir, phase)?;
    Ok(validate_active_hubbard_fms_source_handoffs(input, phase, &handoffs).is_ok())
}

struct ActiveHubbardFmsSourceHandoffs {
    hubbard_l: usize,
    aphase: HubbardAphaseBinData,
    transformation: HubbardTransformationBinData,
}

fn read_active_hubbard_fms_source_handoffs(
    work_dir: &Path,
    phase: &PhaseBinData,
) -> Result<ActiveHubbardFmsSourceHandoffs> {
    let hubbard = read_hubbard_input(work_dir)?;
    let hubbard_l = usize::try_from(hubbard.l)
        .context("active Hubbard FMS source generation requires nonnegative l_hubbard")?;
    let v_hubbard_path = work_dir.join("v_hubbard.bin");
    let aphase_path = work_dir.join("aphase_hubbard.bin");
    let transformation_path = work_dir.join("transformation_hubbard.bin");
    read_v_hubbard_bin_inferred(&v_hubbard_path, phase.potential_count())
        .with_context(|| format!("failed to read {}", v_hubbard_path.display()))?;
    if !aphase_path.is_file() || !transformation_path.is_file() {
        bail!(
            "active Hubbard FMS source generation requires aphase_hubbard.bin and transformation_hubbard.bin"
        );
    }

    let aphase =
        read_aphase_hubbard_bin_inferred(&aphase_path, phase.energy_count, phase.potential_count())
            .with_context(|| format!("failed to read {}", aphase_path.display()))?;
    let transformation = read_transformation_hubbard_bin_inferred(
        &transformation_path,
        hubbard_l,
        phase.potential_count(),
    )
    .with_context(|| format!("failed to read {}", transformation_path.display()))?;

    Ok(ActiveHubbardFmsSourceHandoffs {
        hubbard_l,
        aphase,
        transformation,
    })
}

fn validate_active_hubbard_fms_source_handoffs(
    input: &FmsInput,
    phase: &PhaseBinData,
    handoffs: &ActiveHubbardFmsSourceHandoffs,
) -> Result<()> {
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for active Hubbard FMS generation")?;
    if max_potential < 1 {
        bail!("active Hubbard FMS source generation requires potential 1 for UseTFrm");
    }
    let global_lmax = global_fms_lmax(input, max_potential)?;
    if handoffs.hubbard_l > global_lmax {
        bail!(
            "l_hubbard {} exceeds FMS global lmax {}",
            handoffs.hubbard_l,
            global_lmax
        );
    }
    if handoffs.aphase.potential_count() != phase.potential_count() {
        bail!(
            "aphase_hubbard.bin has {} potential block(s), expected {}",
            handoffs.aphase.potential_count(),
            phase.potential_count()
        );
    }
    if handoffs.aphase.energy_count() != phase.energy_count {
        bail!(
            "aphase_hubbard.bin has {} energy point(s), expected {}",
            handoffs.aphase.energy_count(),
            phase.energy_count
        );
    }
    if handoffs.aphase.spin_count() < phase.spin_count {
        bail!(
            "aphase_hubbard.bin has {} spin block(s), expected at least {}",
            handoffs.aphase.spin_count(),
            phase.spin_count
        );
    }
    if handoffs.aphase.angular_limit < global_lmax {
        bail!(
            "aphase_hubbard.bin angular limit {} is below FMS global lmax {}",
            handoffs.aphase.angular_limit,
            global_lmax
        );
    }
    if handoffs.transformation.hubbard_l != handoffs.hubbard_l {
        bail!(
            "transformation_hubbard.bin l_hubbard {} does not match hubbard.inp {}",
            handoffs.transformation.hubbard_l,
            handoffs.hubbard_l
        );
    }
    if handoffs.transformation.potential_count() != phase.potential_count() {
        bail!(
            "transformation_hubbard.bin has {} potential block(s), expected {}",
            handoffs.transformation.potential_count(),
            phase.potential_count()
        );
    }
    if handoffs.transformation.spin_count() == 0 {
        bail!("transformation_hubbard.bin requires at least one spin block");
    }
    if handoffs.transformation.spin_count() < phase.spin_count {
        bail!(
            "transformation_hubbard.bin has {} spin block(s), expected at least {}",
            handoffs.transformation.spin_count(),
            phase.spin_count
        );
    }
    if handoffs.transformation.angular_limit < global_lmax {
        bail!(
            "transformation_hubbard.bin angular limit {} is below FMS global lmax {}",
            handoffs.transformation.angular_limit,
            global_lmax
        );
    }
    let expected_block = handoffs
        .hubbard_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("l_hubbard is too large for active Hubbard FMS generation")?;
    if handoffs.transformation.row_count() != expected_block
        || handoffs.transformation.column_count() != expected_block
    {
        bail!(
            "transformation_hubbard.bin has a {}x{} block, expected {}x{} for l_hubbard {}",
            handoffs.transformation.row_count(),
            handoffs.transformation.column_count(),
            expected_block,
            expected_block,
            handoffs.hubbard_l
        );
    }
    if usize::try_from(input.lmaxph[1])
        .ok()
        .is_none_or(|lmax| lmax < handoffs.hubbard_l)
    {
        bail!(
            "active Hubbard FMS source generation requires lmaxph(1) to include l_hubbard {}",
            handoffs.hubbard_l
        );
    }
    Ok(())
}

fn supported_source_fms_controls(input: &FmsInput) -> bool {
    validate_supported_source_fms_controls(input).is_ok()
}

fn supported_source_cluster_controls(input: &FmsInput) -> bool {
    input.cluster.rfms2.is_finite()
        && input.cluster.rdirec.is_finite()
        && (input.do_fms == 0 || (input.cluster.rfms2 >= 0.0 && input.cluster.rdirec >= 0.0))
}

fn validate_supported_source_fms_controls(input: &FmsInput) -> Result<()> {
    validate_supported_source_debye_controls(input)?;
    if !supported_source_cluster_controls(input) {
        bail!("FMS source generation requires finite nonnegative cluster radii for full FMS");
    }
    if !input.debye.sig2g.is_finite() || input.debye.sig2g < 0.0 {
        bail!(
            "FMS source generation requires finite nonnegative global SIG2, got {}",
            input.debye.sig2g
        );
    }
    if !input.cluster.toler1.is_finite() || input.cluster.toler1 <= 0.0 {
        bail!(
            "FMS source generation requires positive finite toler1, got {}",
            input.cluster.toler1
        );
    }
    if !input.cluster.toler2.is_finite() || input.cluster.toler2 < 0.0 {
        bail!(
            "FMS source generation requires finite nonnegative toler2, got {}",
            input.cluster.toler2
        );
    }
    Ok(())
}

fn validate_supported_source_debye_controls(input: &FmsInput) -> Result<()> {
    match input.control.idwopt {
        value if value < 0 => Ok(()),
        0 | 3 => {
            if !input.debye.tk.is_finite() || input.debye.tk < 0.0 {
                bail!(
                    "FMS idwopt={} source generation requires finite nonnegative temperature, got {}",
                    input.control.idwopt,
                    input.debye.tk
                );
            }
            if !input.debye.thetad.is_finite() || input.debye.thetad <= 0.0 {
                bail!(
                    "FMS idwopt={} source generation requires positive finite Debye temperature, got {}",
                    input.control.idwopt,
                    input.debye.thetad
                );
            }
            Ok(())
        }
        1 | 2 => {
            if !input.debye.tk.is_finite() || input.debye.tk < 0.0 {
                bail!(
                    "FMS idwopt={} source generation requires finite nonnegative temperature, got {}",
                    input.control.idwopt,
                    input.debye.tk
                );
            }
            Ok(())
        }
        4 | 5 => Ok(()),
        value => bail!(
            "FMS source generation received unexpected idwopt={} Debye-Waller damping",
            value
        ),
    }
}

fn can_read_fms_dmdw_handoffs(work_dir: &Path) -> Result<bool> {
    let Some(calculation) = read_fms_dmdw_calculation(work_dir)? else {
        return Ok(false);
    };
    validate_fms_dmdw_calculation(&calculation)?;
    let dym_path = work_dir.join(&calculation.dym_file);
    if !dym_path.is_file() {
        return Ok(false);
    }
    Ok(read_dym(&dym_path).is_ok())
}

fn read_fms_dmdw_calculation(work_dir: &Path) -> Result<Option<DmdwCalculation>> {
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

fn validate_fms_dmdw_calculation(calculation: &DmdwCalculation) -> Result<()> {
    if calculation.order <= 0 {
        bail!(
            "FMS idwopt=5 requires a positive DMDW Lanczos order, got {}",
            calculation.order
        );
    }
    if calculation.temperature_flag <= 0 {
        bail!(
            "FMS idwopt=5 requires a positive DMDW temperature count, got {}",
            calculation.temperature_flag
        );
    }
    if !calculation.temperature.is_finite() {
        bail!("FMS idwopt=5 requires a finite DMDW temperature");
    }
    if calculation.temperature_flag > 1 {
        let temperature_max = calculation
            .temperature_max
            .context("FMS idwopt=5 DMDW multi-temperature input requires an upper temperature")?;
        if !temperature_max.is_finite() {
            bail!("FMS idwopt=5 requires a finite DMDW upper temperature");
        }
    }
    if calculation.dym_file.trim().is_empty() {
        bail!("FMS idwopt=5 requires a DMDW dynamical-matrix filename");
    }
    Ok(())
}

fn generate_gg_outputs_from_source_handoffs(
    work_dir: &Path,
    input: &FmsInput,
) -> Result<Option<GeneratedFmsSourceMetadata>> {
    let Some(generated) = build_gg_outputs_from_source_handoffs(work_dir, input)? else {
        return Ok(None);
    };

    write_gg_bin_cache(&work_dir.join("gg.bin"), &generated.gg)?;
    write_gg_dat_cache(&work_dir.join("gg.dat"), &generated.gg)?;
    write_generated_hubbard_gtr_m(work_dir, generated.hubbard_gtr_m.as_ref())?;
    if let Some(slice) = generated.gg_slice {
        write_rhorrp_gg_slice_bin(work_dir.join("gg_slice.bin"), &slice).with_context(|| {
            format!(
                "failed to write {}",
                work_dir.join("gg_slice.bin").display()
            )
        })?;
    }
    if let Some(diag) = generated.gg_diag {
        write_rhorrp_gg_diag_bin(work_dir.join("gg_diag.bin"), &diag).with_context(|| {
            format!("failed to write {}", work_dir.join("gg_diag.bin").display())
        })?;
    }
    Ok(Some(generated.metadata))
}

fn write_generated_hubbard_gtr_m(
    work_dir: &Path,
    data: Option<&HubbardLdosGtrMBinData>,
) -> Result<()> {
    let Some(data) = data else {
        return Ok(());
    };
    let path = work_dir.join("gtr_m00.bin");
    write_hubbard_ldos_gtr_m_bin(&path, data)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn build_gg_outputs_from_source_handoffs(
    work_dir: &Path,
    input: &FmsInput,
) -> Result<Option<GeneratedFmsSourceOutputs>> {
    if !required_fms_source_handoffs_present(work_dir) {
        return Ok(None);
    }
    validate_supported_source_fms_controls(input)?;
    if input.control.idwopt == 4 && !work_dir.join("sig2.dat").is_file() {
        bail!("FMS idwopt=4 source generation requires sig2.dat");
    }
    if matches!(input.control.idwopt, 1 | 2) && !work_dir.join("spring.inp").is_file() {
        bail!(
            "FMS idwopt={} source generation requires spring.inp",
            input.control.idwopt
        );
    }
    if input.control.idwopt == 5 && !can_read_fms_dmdw_handoffs(work_dir)? {
        bail!("FMS idwopt=5 source generation requires enabled dmdw.inp and its dym file");
    }

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    if !phase_supports_fms_lmax(input, &phase) {
        return Ok(None);
    }
    let global = read_global_input(work_dir)?;
    if let Some(reciprocal) = read_optional_fms_reciprocal_input(work_dir)? {
        match reciprocal.ispace {
            0 => {
                if active_hubbard_fms_source_requested(work_dir)? {
                    bail!("reciprocal FMS does not yet support active Hubbard source handoffs");
                }
                let cell = reciprocal
                    .cell
                    .as_ref()
                    .context("reciprocal.inp ispace=0 requires a reciprocal cell block")?;
                let generated =
                    build_reciprocal_fms_source_outputs(work_dir, input, &global, &phase, cell)
                        .context(
                            "failed to generate reciprocal FMS gg cache from phase/cell handoffs",
                        )?;
                return Ok(Some(generated));
            }
            1 => {}
            ispace => bail!("reciprocal.inp ispace must be 0 or 1, got {ispace}"),
        }
    }
    let geom = read_geom_dat(work_dir)?;

    let generated = if global.control.do_nrixs != 1
        && active_hubbard_fms_source_requested(work_dir)?
    {
        let handoffs = read_active_hubbard_fms_source_handoffs(work_dir, &phase)?;
        validate_active_hubbard_fms_source_handoffs(input, &phase, &handoffs)?;
        build_active_hubbard_fms_source_outputs(
            work_dir,
            input,
            &global,
            &phase,
            &geom,
            &handoffs,
            0,
            input.do_fms,
            false,
        )
        .context("failed to generate active Hubbard FMS gg cache from phase/geometry handoffs")?
    } else {
        build_fms_source_outputs(work_dir, input, &global, &phase, &geom)
            .context("failed to generate FMS gg cache from phase/geometry handoffs")?
    };
    Ok(Some(generated))
}

struct GeneratedFmsSourceOutputs {
    gg: GgDatData,
    gg_slice: Option<RhorrpGgSliceBinData>,
    gg_diag: Option<RhorrpGgDiagBinData>,
    hubbard_gtr_m: Option<HubbardLdosGtrMBinData>,
    metadata: GeneratedFmsSourceMetadata,
}

/// Generate LDOS `gtrNN.bin` files from source FMS handoffs when the supported
/// non-spin, non-full-potential `fmsdos` path is available.
pub(crate) fn write_ldos_gtr_bin_source_handoffs(
    work_dir: &Path,
    ldos: &LdosInput,
) -> Result<usize> {
    let Some(outputs) = build_ldos_gtr_bin_source_handoffs(work_dir, ldos)? else {
        return Ok(0);
    };

    let mut written = 0;
    for output in outputs {
        let path = work_dir.join(format!("gtr{:02}.bin", output.central_potential));
        if path.is_file()
            && read_gtr_bin(&path)
                .is_ok_and(|cached| gtr_bin_matches_source_output(&cached, &output.data))
        {
            continue;
        }
        write_gtr_bin_cache(&path, &output.data)?;
        written += 1;
    }
    Ok(written)
}

/// Generate FEFF Hubbard LDOS first-pass `gtr_m00.bin` and
/// `gtr_off00.bin` from ordinary two-spin phase handoffs.
#[cfg(feature = "full")]
pub(crate) fn write_hubbard_ldos_first_pass_traces(
    work_dir: &Path,
    ldos: &LdosInput,
) -> Result<usize> {
    if !work_dir.join("fms.inp").is_file() || !required_fms_source_handoffs_present(work_dir) {
        return Ok(0);
    }
    let hubbard = read_hubbard_input(work_dir)?;
    if hubbard.mldos_hubb != 2 {
        return Ok(0);
    }
    let hubbard_l =
        usize::try_from(hubbard.l).context("hubbard.inp l_hubbard must be nonnegative")?;
    let mut input = read_input(work_dir)?;
    overlay_ldos_fms_controls(&mut input, ldos);
    validate_supported_source_fms_controls(&input)?;

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    if phase.spin_count == 0 {
        bail!(
            "Hubbard LDOS first-pass FMS requires at least one phase spin channel, got {}",
            phase.spin_count
        );
    }
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for Hubbard LDOS first-pass FMS")?;
    let energy_grid = ldos_input_energy_grid_hartree(ldos)?;
    let spin_sources = [
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid_for_spin(
            work_dir,
            energy_grid.clone(),
            1,
        )?,
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid_for_spin(
            work_dir,
            energy_grid.clone(),
            -1,
        )?,
    ];
    let source_angular_count = spin_sources[0]
        .wavefunctions
        .wavefunctions
        .angular_momentum_count();
    let dimensions_lmax = read_optional_dimensions_lmax(work_dir)?;
    input = ldos_source_grid_effective_fms_input(
        &input,
        &phase,
        source_angular_count,
        dimensions_lmax,
        max_potential,
    )?;
    let geom = read_geom_dat(work_dir)?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match phase.bin maximum potential {} for Hubbard LDOS first-pass FMS",
            geom.nph,
            max_potential
        );
    }

    let global_lmax = global_fms_lmax(&input, max_potential)?;
    if hubbard_l > global_lmax {
        bail!(
            "Hubbard l={} exceeds first-pass FMS global lmax {}",
            hubbard_l,
            global_lmax
        );
    }
    // FEFF's `gtr_m`/`gtr_off` arrays use the fixed `DimsMod::lx` capacity,
    // not every angular channel made available by the radial source.  Keep
    // the historical source-width fallback only when `.dimensions.dat` is
    // unavailable.
    let output_lmax = dimensions_lmax.unwrap_or_else(|| global_lmax.max(source_angular_count - 1));
    let magnetic_count = (output_lmax + 1)
        .checked_mul(output_lmax + 1)
        .context("Hubbard LDOS first-pass magnetic dimension is too large")?;
    let offdiag_order = (hubbard_l + 1)
        .checked_mul(hubbard_l + 1)
        .context("Hubbard LDOS first-pass off-diagonal dimension is too large")?;
    let mut gtr_m_values = Array5::<Complex32>::zeros((
        2,
        energy_grid.len(),
        max_potential + 1,
        output_lmax + 1,
        magnetic_count,
    ));
    let mut gtr_off_values = Array6::<Complex32>::zeros((
        output_lmax + 1,
        2,
        energy_grid.len(),
        max_potential + 1,
        offdiag_order,
        offdiag_order,
    ));

    let cluster_radius = effective_fms_cluster_radius(&input)?;
    if cluster_radius > 0.0 {
        let direct_cutoff = effective_fms_direct_cutoff(&input)?;
        let spin_orbit = spin_orbit_coupling_tables(global_lmax)
            .context("failed to build Hubbard LDOS first-pass spin-orbit tables")?;
        let xnlm = legendre_normalization_table(global_lmax)
            .context("failed to build Hubbard LDOS first-pass normalization table")?;
        let calculated_l = vec![true; global_lmax + 1];

        // `fmsdos_h_step1` declares all trace arrays `intent(out)` and clears
        // them on entry.  The outer `lfms2=0` loop therefore leaves only its
        // final (`iph0=nph`) solve alive.  The inner FEFF call nevertheless
        // hard-codes `fms(lfms=1)`, so that final cluster publishes every
        // potential block.  Reproducing the seemingly more useful per-center
        // merge changes the Ni d occupation matrix and creates a spurious
        // crystal-field Hubbard potential.
        let central_potentials = if input.do_fms != 0 {
            vec![0]
        } else {
            vec![max_potential]
        };
        // The current FEFF10 Hubbard driver passes its never-initialized
        // `lmaxphpass` work array to `fms`.  The pinned gfortran reference
        // observes that array as all zeros: only the s channel is solved,
        // higher-l `gtr_m` entries stay zero, and `gtr_off` is byte-zero.
        // Make that binary compatibility behavior deterministic instead of
        // relying on undefined stack contents.
        let solver_lmaxph = vec![0; max_potential + 1];
        for central_potential in central_potentials {
            let central = i32::try_from(central_potential)
                .context("Hubbard LDOS central potential does not fit in i32")?;
            let mut atoms =
                fms_atoms_from_geom(&input, &geom, max_potential, cluster_radius, central)?;
            if input.do_fms != 0 {
                sort_representative_atoms(0, max_potential, &mut atoms)
                    .context("failed to prepare Hubbard LDOS first-pass representative atoms")?;
            }
            let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
                .with_context(|| {
                    format!(
                        "failed to build Hubbard LDOS first-pass rotation geometry for central potential {central_potential}"
                    )
                })?;
            let mean_square_displacements =
                fms_mean_square_displacements(work_dir, &input, &phase, &atoms)?;
            let plan = fms_real_space_plan(FmsRealSpacePlanInput {
                // FEFF hard-codes the inner Hubbard LDOS solve to full
                // potential packing even when the outer LDOS card says
                // `lfms2=0`.
                lfms: 1,
                minv: input.control.minv,
                spin_channels: 1,
                spin_selector: 0,
                atoms: &atoms,
                max_potential,
                global_lmax,
                raw_potential_lmax: &solver_lmaxph,
                state_capacity: None,
                spin_orbit: &spin_orbit,
                direct_cutoff,
                mean_square_displacements: mean_square_displacements.view(),
                xnlm: xnlm.view(),
                rotations: geometry.rotations.view(),
                calculated_l: &calculated_l,
                convergence_tolerance: input.cluster.toler1 as f32,
                zero_tolerance: input.cluster.toler2 as f32,
                full_scattering_matrix_requested: false,
                retain_setup: false,
                retain_pair_tables: false,
                retain_free_propagator: false,
                retain_t_matrix: false,
                retain_system_matrix: false,
            })
            .with_context(|| {
                format!(
                    "failed to prepare Hubbard LDOS first-pass FMS plan for central potential {central_potential}"
                )
            })?;

            for (spin, source) in spin_sources.iter().enumerate() {
                let energy_count = energy_grid.len();
                let mut wave_numbers_by_energy = Vec::with_capacity(energy_count);
                let mut phase_shifts_by_energy = Vec::with_capacity(energy_count);
                for energy in 0..energy_count {
                    wave_numbers_by_energy.push(ldos_source_fms_wave_numbers(
                        source.wavefunctions.wavefunctions.wave_numbers.view(),
                        energy,
                        central_potential,
                    )?);
                    phase_shifts_by_energy.push(ldos_source_fms_phase_shifts_for_energy(
                        source.wavefunctions.wavefunctions.phase_shifts.view(),
                        &input,
                        energy,
                        global_lmax,
                        max_potential,
                        phase.potential_count(),
                    )?);
                }
                let points = wave_numbers_by_energy
                    .iter()
                    .zip(phase_shifts_by_energy.iter())
                    .map(|(wave_numbers, phase_shifts)| FmsRealSpaceEnergyPoint {
                        wave_numbers,
                        phase_shifts: phase_shifts.view(),
                    })
                    .collect::<Vec<_>>();

                for (energy, result) in fms_real_space_spectrum(&plan, &points)
                    .into_iter()
                    .enumerate()
                {
                    let scattering = result
                        .with_context(|| {
                            format!(
                                "failed Hubbard LDOS first-pass FMS for central potential {} spin {} energy {}",
                                central_potential,
                                spin + 1,
                                energy + 1
                            )
                        })?
                        .scattering
                        .scattering;
                    let phase_shifts = &phase_shifts_by_energy[energy];
                    for potential in 0..=max_potential {
                        let potential_lmax =
                            usize::try_from(input.lmaxph[potential])?.min(global_lmax);
                        for angular in 0..=potential_lmax {
                            let magnetic_start = angular * angular;
                            let magnetic_end = (angular + 1) * (angular + 1);
                            let phase_shift = phase_shifts[(0, global_lmax + angular, potential)];
                            for magnetic in magnetic_start..magnetic_end {
                                gtr_m_values[(spin, energy, potential, angular, magnetic)] =
                                    normalize_hubbard_fms_trace(
                                        scattering[(magnetic, magnetic, potential)],
                                        phase_shift,
                                        angular,
                                    );
                            }
                            if angular == hubbard_l {
                                for row in magnetic_start..magnetic_end {
                                    for column in magnetic_start..magnetic_end {
                                        gtr_off_values
                                            [(angular, spin, energy, potential, row, column)] =
                                            normalize_hubbard_fms_trace(
                                                scattering[(row, column, potential)],
                                                phase_shift,
                                                angular,
                                            );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let gtr_m = HubbardLdosGtrMBinData {
        point_count_declared: energy_grid.len(),
        horizontal_count: energy_grid.len(),
        danes_extension_count: 0,
        highest_potential_index: max_potential,
        fms_mode: input.do_fms,
        angular_limit: output_lmax,
        values: gtr_m_values,
    };
    let gtr_off = HubbardLdosGtrOffBinData {
        point_count_declared: energy_grid.len(),
        horizontal_count: energy_grid.len(),
        danes_extension_count: 0,
        highest_potential_index: max_potential,
        fms_mode: input.do_fms,
        hubbard_l,
        angular_limit: output_lmax,
        values: gtr_off_values,
    };
    let gtr_m_path = work_dir.join("gtr_m00.bin");
    let gtr_off_path = work_dir.join("gtr_off00.bin");
    write_hubbard_ldos_gtr_m_bin(&gtr_m_path, &gtr_m)
        .with_context(|| format!("failed to write {}", gtr_m_path.display()))?;
    write_hubbard_ldos_gtr_off_bin(&gtr_off_path, &gtr_off)
        .with_context(|| format!("failed to write {}", gtr_off_path.display()))?;
    Ok(2)
}

/// Regenerate the active-Hubbard magnetic trace using FEFF's final
/// `lfms2=0` central-potential solve.
///
/// The ordinary spectrum FMS input remains absorber-centered and is refreshed
/// separately. `fmsdos_h_step2` clears its `intent(out)` trace on every outer
/// central-potential call and hard-codes full-potential packing internally, so
/// only the final (`iph0=nph`) cluster and all of its potential blocks survive.
pub(crate) fn write_hubbard_ldos_independent_second_pass_trace(
    work_dir: &Path,
    ldos: &LdosInput,
) -> Result<usize> {
    if ldos.control.lfms2 != 0 {
        return Ok(0);
    }
    let mut input = read_input(work_dir)?;
    overlay_ldos_fms_controls(&mut input, ldos);
    validate_supported_source_fms_controls(&input)?;

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for Hubbard LDOS second-pass FMS")?;
    let global = read_global_input(work_dir)?;
    let geom = read_geom_dat(work_dir)?;
    let handoffs = read_active_hubbard_fms_source_handoffs(work_dir, &phase)?;
    validate_active_hubbard_fms_source_handoffs(&input, &phase, &handoffs)?;

    let generated = build_active_hubbard_fms_source_outputs(
        work_dir,
        &input,
        &global,
        &phase,
        &geom,
        &handoffs,
        max_potential,
        1,
        true,
    )
    .with_context(|| {
        format!(
            "failed active Hubbard final-center second-pass solve for central potential {max_potential}"
        )
    })?;
    let source = generated
        .hubbard_gtr_m
        .context("active Hubbard final-center second-pass solve produced no magnetic trace")?;
    let path = work_dir.join("gtr_m00.bin");
    write_hubbard_ldos_gtr_m_bin(&path, &source)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn gtr_bin_matches_source_output(cached: &GtrBinData, source: &GtrBinData) -> bool {
    const GTR_BIN_ABS_TOLERANCE: f64 = 2.0e-6;
    const GTR_BIN_REL_TOLERANCE: f64 = 1.0e-5;

    cached.point_count_declared == source.point_count_declared
        && cached.horizontal_count == source.horizontal_count
        && cached.danes_extension_count == source.danes_extension_count
        && cached.highest_potential_index == source.highest_potential_index
        && cached.fms_mode == source.fms_mode
        && cached.values.dim() == source.values.dim()
        && cached
            .values
            .iter()
            .zip(source.values.iter())
            .all(|(cached, source)| {
                scalar_matches(
                    cached.re,
                    source.re,
                    GTR_BIN_ABS_TOLERANCE,
                    GTR_BIN_REL_TOLERANCE,
                ) && scalar_matches(
                    cached.im,
                    source.im,
                    GTR_BIN_ABS_TOLERANCE,
                    GTR_BIN_REL_TOLERANCE,
                )
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

/// Generate per-potential LDOS `gtrNN.bin` files from source FMS handoffs on
/// the LDOS-card energy grid.
pub(crate) fn write_ldos_gtr_bin_source_grid_handoff(
    work_dir: &Path,
    ldos: &LdosInput,
    energy_grid_hartree: ArrayView1<'_, Complex>,
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    phase_shifts: ArrayView3<'_, Complex>,
) -> Result<usize> {
    let Some(outputs) = build_ldos_gtr_bin_source_grid_handoff(
        work_dir,
        ldos,
        energy_grid_hartree,
        wave_numbers_bohr,
        phase_shifts,
    )?
    else {
        return Ok(0);
    };

    let mut written = 0;
    for output in outputs {
        let path = work_dir.join(format!("gtr{:02}.bin", output.central_potential));
        write_gtr_bin_cache(&path, &output.data)?;
        written += 1;
    }
    Ok(written)
}

pub(crate) fn has_supported_ldos_gtr_bin_source_grid_handoff(
    work_dir: &Path,
    ldos: &LdosInput,
    energy_grid_hartree: ArrayView1<'_, Complex>,
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    phase_shifts: ArrayView3<'_, Complex>,
) -> Result<bool> {
    Ok(ldos_gtr_bin_source_grid_setup(
        work_dir,
        ldos,
        energy_grid_hartree,
        wave_numbers_bohr,
        phase_shifts,
    )?
    .is_some())
}

struct LdosGtrBinSourceOutput {
    central_potential: usize,
    data: GtrBinData,
}

struct LdosGtrBinSourceGridSetup {
    effective_input: FmsInput,
    global: GlobalInput,
    phase: PhaseBinData,
    geom: GeomDat,
    max_potential: usize,
}

fn build_ldos_gtr_bin_source_handoffs(
    work_dir: &Path,
    ldos: &LdosInput,
) -> Result<Option<Vec<LdosGtrBinSourceOutput>>> {
    if !supported_ldos_fmsdos_source_controls(ldos) {
        return Ok(None);
    }
    if !work_dir.join("fms.inp").is_file() || !required_fms_source_handoffs_present(work_dir) {
        return Ok(None);
    }

    let mut input = read_input(work_dir)?;
    overlay_ldos_fms_controls(&mut input, ldos);
    if validate_supported_source_fms_controls(&input).is_err() {
        return Ok(None);
    }
    if input.control.idwopt == 4 && !work_dir.join("sig2.dat").is_file() {
        return Ok(None);
    }
    if matches!(input.control.idwopt, 1 | 2) && !work_dir.join("spring.inp").is_file() {
        return Ok(None);
    }
    if input.control.idwopt == 5 && !can_read_fms_dmdw_handoffs(work_dir)? {
        return Ok(None);
    }
    if active_hubbard_fms_source_requested(work_dir)? {
        return Ok(None);
    }

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    if !ldos_phase_energy_grid_matches_input(ldos, &phase)? {
        return Ok(None);
    }
    if phase.spin_count != 1 || !phase_supports_fms_lmax(&input, &phase) {
        return Ok(None);
    }
    let global = read_global_input(work_dir)?;
    if global.control.do_nrixs != 0 {
        return Ok(None);
    }
    let geom = read_geom_dat(work_dir)?;
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for LDOS FMS generation")?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match phase.bin maximum potential {} for LDOS FMS generation",
            geom.nph,
            max_potential
        );
    }

    let central_potentials = (0..=max_potential).collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(central_potentials.len());
    for central_potential in central_potentials {
        outputs.push(build_ldos_gtr_bin_for_central_potential(
            work_dir,
            &input,
            &global,
            &phase,
            &geom,
            central_potential,
        )?);
    }
    Ok(Some(outputs))
}

fn build_ldos_gtr_bin_source_grid_handoff(
    work_dir: &Path,
    ldos: &LdosInput,
    energy_grid_hartree: ArrayView1<'_, Complex>,
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    phase_shifts: ArrayView3<'_, Complex>,
) -> Result<Option<Vec<LdosGtrBinSourceOutput>>> {
    let Some(setup) = ldos_gtr_bin_source_grid_setup(
        work_dir,
        ldos,
        energy_grid_hartree,
        wave_numbers_bohr,
        phase_shifts,
    )?
    else {
        return Ok(None);
    };

    if setup.effective_input.cluster.rfms2.is_finite() && setup.effective_input.cluster.rfms2 <= 0.0
    {
        let mut outputs = Vec::with_capacity(setup.max_potential + 1);
        for central_potential in 0..=setup.max_potential {
            outputs.push(build_zero_ldos_gtr_bin_source_grid(
                &setup.effective_input,
                energy_grid_hartree.len(),
                setup.max_potential,
                central_potential,
            )?);
        }
        return Ok(Some(outputs));
    }

    let mut outputs = Vec::with_capacity(setup.max_potential + 1);
    for central_potential in 0..=setup.max_potential {
        outputs.push(build_ldos_gtr_bin_for_source_grid_central_potential(
            work_dir,
            &setup.effective_input,
            &setup.global,
            &setup.phase,
            &setup.geom,
            energy_grid_hartree,
            wave_numbers_bohr,
            phase_shifts,
            central_potential,
        )?);
    }
    Ok(Some(outputs))
}

fn ldos_gtr_bin_source_grid_setup(
    work_dir: &Path,
    ldos: &LdosInput,
    energy_grid_hartree: ArrayView1<'_, Complex>,
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    phase_shifts: ArrayView3<'_, Complex>,
) -> Result<Option<LdosGtrBinSourceGridSetup>> {
    if !supported_ldos_fmsdos_source_controls(ldos) {
        return Ok(None);
    }
    if energy_grid_hartree.is_empty() {
        bail!("LDOS FMS source grid requires at least one energy point");
    }
    if !work_dir.join("fms.inp").is_file() || !required_fms_source_handoffs_present(work_dir) {
        return Ok(None);
    }

    let mut input = read_input(work_dir)?;
    overlay_ldos_fms_controls(&mut input, ldos);

    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    if phase.spin_count != 1 {
        return Ok(None);
    }
    let global = read_global_input(work_dir)?;
    if global.control.do_nrixs != 0 || global.control.ispin != 0 {
        return Ok(None);
    }
    let geom = read_geom_dat(work_dir)?;
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for LDOS FMS generation")?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match phase.bin maximum potential {} for LDOS FMS generation",
            geom.nph,
            max_potential
        );
    }
    if phase_shifts.dim().0 != energy_grid_hartree.len() {
        bail!(
            "LDOS RHORRP phase grid has {} energy point(s), expected {}",
            phase_shifts.dim().0,
            energy_grid_hartree.len()
        );
    }
    if wave_numbers_bohr.dim().0 != energy_grid_hartree.len() {
        bail!(
            "LDOS RHORRP wave-number grid has {} energy point(s), expected {}",
            wave_numbers_bohr.dim().0,
            energy_grid_hartree.len()
        );
    }
    if phase_shifts.dim().2 <= max_potential || wave_numbers_bohr.dim().1 <= max_potential {
        return Ok(None);
    }
    let dimensions_lmax = read_optional_dimensions_lmax(work_dir)?;
    let effective_input = ldos_source_grid_effective_fms_input(
        &input,
        &phase,
        phase_shifts.dim().1,
        dimensions_lmax,
        max_potential,
    )?;

    if effective_input.cluster.rfms2.is_finite() && effective_input.cluster.rfms2 <= 0.0 {
        return Ok(Some(LdosGtrBinSourceGridSetup {
            effective_input,
            global,
            phase,
            geom,
            max_potential,
        }));
    }

    if validate_supported_source_fms_controls(&effective_input).is_err() {
        return Ok(None);
    }
    if effective_input.control.idwopt == 4 && !work_dir.join("sig2.dat").is_file() {
        return Ok(None);
    }
    if matches!(effective_input.control.idwopt, 1 | 2) && !work_dir.join("spring.inp").is_file() {
        return Ok(None);
    }
    if effective_input.control.idwopt == 5 && !can_read_fms_dmdw_handoffs(work_dir)? {
        return Ok(None);
    }
    Ok(Some(LdosGtrBinSourceGridSetup {
        effective_input,
        global,
        phase,
        geom,
        max_potential,
    }))
}

fn supported_ldos_fmsdos_source_controls(ldos: &LdosInput) -> bool {
    ldos.control.mldos == 1 && ldos.control.ispin == 0 && ldos.ldostype <= 0
}

fn ldos_phase_energy_grid_matches_input(ldos: &LdosInput, phase: &PhaseBinData) -> Result<bool> {
    let ldos_grid = ldos_input_energy_grid_hartree(ldos)?;
    Ok(phase.energy_grid.len() == ldos_grid.len()
        && phase
            .energy_grid
            .iter()
            .zip(ldos_grid.iter())
            .all(|(phase_energy, ldos_energy)| (phase_energy - ldos_energy).norm() <= 1.0e-8))
}

fn ldos_input_energy_grid_hartree(input: &LdosInput) -> Result<Array1<Complex>> {
    let energy_count = usize::try_from(input.control.neldos)
        .context("ldos.inp neldos must be non-negative and fit in usize")?;
    if energy_count == 0 {
        bail!("ldos.inp neldos must be positive");
    }

    let step_ev = if energy_count > 1 {
        (input.mesh.emax - input.mesh.emin) / (energy_count - 1) as Real
    } else {
        0.0
    };
    Ok(Array1::from_shape_fn(energy_count, |index| {
        Complex::new(
            (input.mesh.emin + step_ev * index as Real) / FEFF_HARTREE_EV,
            input.mesh.eimag / FEFF_HARTREE_EV,
        )
    }))
}

fn overlay_ldos_fms_controls(input: &mut FmsInput, ldos: &LdosInput) {
    input.control.mfms = i32::from(ldos.control.lfms2 != 0);
    input.control.minv = ldos.control.minv;
    input.cluster.rfms2 = ldos.mesh.rfms2;
    input.cluster.rdirec = ldos.fms.rdirec;
    input.cluster.toler1 = ldos.fms.toler1;
    input.cluster.toler2 = ldos.fms.toler2;
    input.lmaxph = ldos.lmaxph.clone();
    input.decomposition_channels = -1;
    input.save_gg_slice = false;
    input.do_fms = ldos.control.lfms2;
}

fn read_optional_dimensions_lmax(work_dir: &Path) -> Result<Option<usize>> {
    let path = work_dir.join(".dimensions.dat");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let dimensions = DimensionsDat::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if dimensions.lx < 0 {
        bail!(
            "{} lx must be nonnegative for LDOS FMS source-grid generation, got {}",
            path.display(),
            dimensions.lx
        );
    }
    usize::try_from(dimensions.lx)
        .map(Some)
        .context("failed to convert .dimensions.dat lx")
}

fn fms_output_spin_capacity(work_dir: &Path, active_spin_count: usize) -> Result<usize> {
    let path = work_dir.join(".dimensions.dat");
    if !path.is_file() {
        return Ok(active_spin_count);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let dimensions = DimensionsDat::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if dimensions.nspu <= 0 {
        bail!(
            "{} nspu must be positive for FMS gg output, got {}",
            path.display(),
            dimensions.nspu
        );
    }
    let output_spin_capacity =
        usize::try_from(dimensions.nspu).context("failed to convert .dimensions.dat nspu")?;
    if output_spin_capacity < active_spin_count {
        bail!(
            "{} nspu {} is smaller than phase.bin active spin count {}",
            path.display(),
            output_spin_capacity,
            active_spin_count
        );
    }
    Ok(output_spin_capacity)
}

fn fms_gg_section_values(
    scattering: ArrayView3<'_, Complex32>,
    absorber_potential: usize,
    active_spin_count: usize,
    output_spin_capacity: usize,
    absorber_lmax: usize,
) -> Result<Array2<Complex>> {
    let angular_count = absorber_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .context("FMS absorber angular channel count is too large")?;
    let active_channel_count = angular_count
        .checked_mul(active_spin_count)
        .context("FMS active gg channel count is too large")?;
    let output_channel_count = angular_count
        .checked_mul(output_spin_capacity)
        .context("FMS output gg channel count is too large")?;
    ensure!(
        absorber_potential < scattering.len_of(Axis(2)),
        "FMS absorber potential {} exceeds scattering potential count {}",
        absorber_potential,
        scattering.len_of(Axis(2))
    );
    ensure!(
        active_channel_count <= scattering.len_of(Axis(0))
            && active_channel_count <= scattering.len_of(Axis(1)),
        "FMS absorber block {} exceeds scattering shape {}x{}",
        active_channel_count,
        scattering.len_of(Axis(0)),
        scattering.len_of(Axis(1))
    );

    // FEFF `fmstot` writes the absorber block with the compiled `nspu`
    // capacity from `.dimensions.dat`, even when the active solve uses fewer
    // spins. Only the active `nsp * (lmaxph(0)+1)^2` corner is populated.
    let mut values = Array2::<Complex>::zeros((output_channel_count, output_channel_count));
    for column in 0..active_channel_count {
        for row in 0..active_channel_count {
            let value = scattering[(row, column, absorber_potential)];
            values[(row, column)] = Complex::new(value.re as f64, value.im as f64);
        }
    }
    Ok(values)
}

fn ldos_source_grid_effective_fms_input(
    input: &FmsInput,
    phase: &PhaseBinData,
    source_angular_count: usize,
    dimensions_lmax: Option<usize>,
    max_potential: usize,
) -> Result<FmsInput> {
    if source_angular_count == 0 {
        bail!("LDOS RHORRP phase-shift table requires at least one angular channel");
    }
    if input.lmaxph.len() <= max_potential {
        bail!(
            "ldos.inp has {} lmaxph value(s), expected at least {} for LDOS FMS generation",
            input.lmaxph.len(),
            max_potential + 1
        );
    }
    if phase.potential_count() <= max_potential {
        bail!(
            "phase.bin has {} potential block(s), expected at least {} for LDOS FMS generation",
            phase.potential_count(),
            max_potential + 1
        );
    }

    let source_lmax = source_angular_count - 1;
    let mut effective = input.clone();
    for potential in 0..=max_potential {
        let raw_lmax = input.lmaxph[potential];
        if raw_lmax < 0 {
            bail!(
                "ldos.inp lmaxph({potential}) must be nonnegative for LDOS FMS source-grid generation, got {raw_lmax}"
            );
        }
        let requested_lmax =
            usize::try_from(raw_lmax).context("failed to convert LDOS FMS lmaxph")?;
        let mut capped_lmax = source_lmax;
        if let Some(dimensions_lmax) = dimensions_lmax {
            capped_lmax = capped_lmax.min(dimensions_lmax);
        }
        effective.lmaxph[potential] = i32::try_from(requested_lmax.min(capped_lmax))
            .context("failed to convert effective LDOS FMS lmaxph")?;
    }
    Ok(effective)
}

fn build_ldos_gtr_bin_for_central_potential(
    work_dir: &Path,
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    geom: &GeomDat,
    central_potential: usize,
) -> Result<LdosGtrBinSourceOutput> {
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for LDOS FMS generation")?;
    if central_potential > max_potential {
        bail!(
            "LDOS central potential {} exceeds maximum potential {}",
            central_potential,
            max_potential
        );
    }

    let global_lmax = global_fms_lmax(input, max_potential)?;
    let cluster_radius = effective_fms_cluster_radius(input)?;
    let direct_cutoff = effective_fms_direct_cutoff(input)?;
    let central =
        i32::try_from(central_potential).context("LDOS central potential does not fit in i32")?;
    // Preserve `yprep` radial order: for an independent `lfms=0` solve the
    // requested central potential must remain in atom slot zero.
    let atoms = fms_atoms_from_geom(input, geom, max_potential, cluster_radius, central)?;
    let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
        .context("failed to build LDOS FMS rotation geometry")?;
    let spin_orbit = spin_orbit_coupling_tables(global_lmax)
        .context("failed to build LDOS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build LDOS FMS normalization table")?;
    let mean_square_displacements = fms_mean_square_displacements(work_dir, input, phase, &atoms)?;
    let calculated_l = vec![true; global_lmax + 1];
    let angular_count = global_lmax
        .checked_add(1)
        .context("LDOS FMS angular count is too large")?;
    let channel_count = angular_count
        .checked_mul(angular_count)
        .context("LDOS FMS channel count is too large")?;
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("LDOS FMS signed-l count is too large")?;
    let mut scattering_matrices = Array4::<Complex32>::zeros((
        phase.energy_count,
        channel_count,
        channel_count,
        phase.potential_count(),
    ));
    let mut phase_shifts = Array4::<Complex32>::zeros((
        phase.energy_count,
        phase.spin_count,
        signed_l_count,
        phase.potential_count(),
    ));

    let plan = fms_real_space_plan(FmsRealSpacePlanInput {
        lfms: input.do_fms,
        minv: input.control.minv,
        spin_channels: phase.spin_count,
        spin_selector: global.control.ispin,
        atoms: &atoms,
        max_potential,
        global_lmax,
        raw_potential_lmax: &input.lmaxph,
        state_capacity: None,
        spin_orbit: &spin_orbit,
        direct_cutoff,
        mean_square_displacements: mean_square_displacements.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: input.cluster.toler1 as f32,
        zero_tolerance: input.cluster.toler2 as f32,
        full_scattering_matrix_requested: false,
        retain_setup: false,
        retain_pair_tables: false,
        retain_free_propagator: false,
        retain_t_matrix: false,
        retain_system_matrix: false,
    })
    .with_context(|| {
        format!(
            "failed to prepare LDOS FMS real-space plan for central potential {central_potential}"
        )
    })?;

    let mut wave_numbers_by_energy = Vec::with_capacity(phase.energy_count);
    let mut phase_shifts_by_energy = Vec::with_capacity(phase.energy_count);
    for energy in 0..phase.energy_count {
        wave_numbers_by_energy.push(fms_wave_numbers(phase, energy)?);
        phase_shifts_by_energy.push(fms_phase_shifts_for_energy(
            phase,
            input,
            energy,
            global_lmax,
            max_potential,
        )?);
    }
    let points: Vec<FmsRealSpaceEnergyPoint<'_>> = wave_numbers_by_energy
        .iter()
        .zip(phase_shifts_by_energy.iter())
        .map(|(wave_numbers, phases)| FmsRealSpaceEnergyPoint {
            wave_numbers,
            phase_shifts: phases.view(),
        })
        .collect();

    for (energy, result) in fms_real_space_spectrum(&plan, &points)
        .into_iter()
        .enumerate()
    {
        let result = result.with_context(|| {
            format!(
                "failed to solve LDOS FMS central potential {} energy section {}",
                central_potential,
                energy + 1
            )
        })?;

        scattering_matrices
            .index_axis_mut(Axis(0), energy)
            .assign(&result.scattering.scattering);
        phase_shifts
            .index_axis_mut(Axis(0), energy)
            .assign(&phase_shifts_by_energy[energy]);
    }

    let mut trace_grid = ldos_fmsdos_trace_grid(LdosFmsdosTraceGridInput {
        scattering_matrices: scattering_matrices.view(),
        phase_shifts: phase_shifts.view(),
        spin_index: 0,
        angular_count,
    })
    .context("failed to project LDOS FMS trace grid")?;
    for potential in 0..phase.potential_count() {
        if potential != central_potential {
            trace_grid
                .index_axis_mut(Axis(1), potential)
                .fill(Complex::new(0.0, 0.0));
        }
    }
    let data = gtr_bin_from_ldos_trace_grid(
        trace_grid.view(),
        phase.main_energy_count,
        phase.auxiliary_energy_count,
        max_potential,
        2,
    )
    .context("failed to package LDOS FMS trace grid as gtrNN.bin")?;
    Ok(LdosGtrBinSourceOutput {
        central_potential,
        data,
    })
}

fn build_zero_ldos_gtr_bin_source_grid(
    input: &FmsInput,
    energy_count: usize,
    max_potential: usize,
    central_potential: usize,
) -> Result<LdosGtrBinSourceOutput> {
    let global_lmax = global_fms_lmax(input, max_potential)?;
    let angular_count = global_lmax
        .checked_add(1)
        .context("LDOS FMS angular count is too large")?;
    let trace_grid = Array3::<Complex>::zeros((energy_count, max_potential + 1, angular_count).f());
    let data = gtr_bin_from_ldos_trace_grid(trace_grid.view(), energy_count, 0, max_potential, 2)
        .context("failed to package zero LDOS FMS source-grid trace as gtrNN.bin")?;
    Ok(LdosGtrBinSourceOutput {
        central_potential,
        data,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_ldos_gtr_bin_for_source_grid_central_potential(
    work_dir: &Path,
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    geom: &GeomDat,
    energy_grid_hartree: ArrayView1<'_, Complex>,
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    source_phase_shifts: ArrayView3<'_, Complex>,
    central_potential: usize,
) -> Result<LdosGtrBinSourceOutput> {
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for LDOS FMS generation")?;
    if central_potential > max_potential {
        bail!(
            "LDOS central potential {} exceeds maximum potential {}",
            central_potential,
            max_potential
        );
    }

    let global_lmax = global_fms_lmax(input, max_potential)?;
    let cluster_radius = effective_fms_cluster_radius(input)?;
    let direct_cutoff = effective_fms_direct_cutoff(input)?;
    let central =
        i32::try_from(central_potential).context("LDOS central potential does not fit in i32")?;
    // Preserve `yprep` radial order so the requested independent center stays
    // in atom slot zero when the FMS driver selects its active potential.
    let atoms = fms_atoms_from_geom(input, geom, max_potential, cluster_radius, central)?;
    let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
        .context("failed to build LDOS FMS rotation geometry")?;
    let spin_orbit = spin_orbit_coupling_tables(global_lmax)
        .context("failed to build LDOS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build LDOS FMS normalization table")?;
    let mean_square_displacements = fms_mean_square_displacements(work_dir, input, phase, &atoms)?;
    let calculated_l = vec![true; global_lmax + 1];
    let angular_count = global_lmax
        .checked_add(1)
        .context("LDOS FMS angular count is too large")?;
    let channel_count = angular_count
        .checked_mul(angular_count)
        .context("LDOS FMS channel count is too large")?;
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("LDOS FMS signed-l count is too large")?;
    let energy_count = energy_grid_hartree.len();
    let mut scattering_matrices = Array4::<Complex32>::zeros((
        energy_count,
        channel_count,
        channel_count,
        phase.potential_count(),
    ));
    let mut phase_shifts = Array4::<Complex32>::zeros((
        energy_count,
        phase.spin_count,
        signed_l_count,
        phase.potential_count(),
    ));

    let plan = fms_real_space_plan(FmsRealSpacePlanInput {
        lfms: input.do_fms,
        minv: input.control.minv,
        spin_channels: phase.spin_count,
        spin_selector: global.control.ispin,
        atoms: &atoms,
        max_potential,
        global_lmax,
        raw_potential_lmax: &input.lmaxph,
        state_capacity: None,
        spin_orbit: &spin_orbit,
        direct_cutoff,
        mean_square_displacements: mean_square_displacements.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: input.cluster.toler1 as f32,
        zero_tolerance: input.cluster.toler2 as f32,
        full_scattering_matrix_requested: false,
        retain_setup: false,
        retain_pair_tables: false,
        retain_free_propagator: false,
        retain_t_matrix: false,
        retain_system_matrix: false,
    })
    .with_context(|| {
        format!(
            "failed to prepare LDOS FMS source-grid real-space plan for central potential {central_potential}"
        )
    })?;

    let mut wave_numbers_by_energy = Vec::with_capacity(energy_count);
    let mut phase_shifts_by_energy = Vec::with_capacity(energy_count);
    for energy in 0..energy_count {
        wave_numbers_by_energy.push(ldos_source_fms_wave_numbers(
            wave_numbers_bohr,
            energy,
            central_potential,
        )?);
        phase_shifts_by_energy.push(ldos_source_fms_phase_shifts_for_energy(
            source_phase_shifts,
            input,
            energy,
            global_lmax,
            max_potential,
            phase.potential_count(),
        )?);
    }
    let points: Vec<FmsRealSpaceEnergyPoint<'_>> = wave_numbers_by_energy
        .iter()
        .zip(phase_shifts_by_energy.iter())
        .map(|(wave_numbers, phases)| FmsRealSpaceEnergyPoint {
            wave_numbers,
            phase_shifts: phases.view(),
        })
        .collect();

    for (energy, result) in fms_real_space_spectrum(&plan, &points)
        .into_iter()
        .enumerate()
    {
        let result = result.with_context(|| {
            format!(
                "failed to solve LDOS FMS source-grid central potential {} energy section {}",
                central_potential,
                energy + 1
            )
        })?;

        scattering_matrices
            .index_axis_mut(Axis(0), energy)
            .assign(&result.scattering.scattering);
        phase_shifts
            .index_axis_mut(Axis(0), energy)
            .assign(&phase_shifts_by_energy[energy]);
    }

    let mut trace_grid = ldos_fmsdos_trace_grid(LdosFmsdosTraceGridInput {
        scattering_matrices: scattering_matrices.view(),
        phase_shifts: phase_shifts.view(),
        spin_index: 0,
        angular_count,
    })
    .context("failed to project LDOS FMS source-grid trace grid")?;
    for potential in 0..phase.potential_count() {
        if potential != central_potential {
            trace_grid
                .index_axis_mut(Axis(1), potential)
                .fill(Complex::new(0.0, 0.0));
        }
    }
    let data = gtr_bin_from_ldos_trace_grid(trace_grid.view(), energy_count, 0, max_potential, 2)
        .context("failed to package LDOS FMS source-grid trace grid as gtrNN.bin")?;
    Ok(LdosGtrBinSourceOutput {
        central_potential,
        data,
    })
}

pub(crate) fn build_screen_fms_source_grid_handoff(
    work_dir: &Path,
    phase: &PhaseBinData,
    grid: ScreenFmsSourceGridInput<'_>,
) -> Result<Option<ScreenFmsClusterGreenHandoff>> {
    build_screen_fms_source_grid_handoff_with_potential_count(
        work_dir,
        phase.potential_count(),
        Some(phase),
        FmsDebyeDampingMetadata::Phase(phase),
        grid,
    )
}

pub(crate) fn build_screen_fms_source_grid_handoff_from_generated_phases(
    work_dir: &Path,
    pot: &PotBinData,
    grid: ScreenFmsSourceGridInput<'_>,
) -> Result<Option<ScreenFmsClusterGreenHandoff>> {
    build_screen_fms_source_grid_handoff_with_potential_count(
        work_dir,
        pot.potential_count(),
        None,
        FmsDebyeDampingMetadata::Pot(pot),
        grid,
    )
}

fn build_screen_fms_source_grid_handoff_with_potential_count(
    work_dir: &Path,
    potential_count: usize,
    phase: Option<&PhaseBinData>,
    damping_metadata: FmsDebyeDampingMetadata<'_>,
    grid: ScreenFmsSourceGridInput<'_>,
) -> Result<Option<ScreenFmsClusterGreenHandoff>> {
    if !work_dir.join("fms.inp").is_file() || !work_dir.join("geom.dat").is_file() {
        return Ok(None);
    }

    let mut input = read_input(work_dir)?;
    input.cluster.rfms2 = grid.cluster_radius_angstrom;
    input.cluster.rdirec = grid.direct_cutoff_angstrom;
    let geom = read_geom_dat(work_dir)?;

    let max_potential = potential_count
        .checked_sub(1)
        .context("SCREEN FMS generation requires at least one potential")?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match maximum potential {} for SCREEN FMS generation",
            geom.nph,
            max_potential
        );
    }

    let energy_count = grid.energy_grid_hartree.len();
    if grid.wave_numbers_bohr.len() < energy_count {
        bail!(
            "SCREEN FMS source wave-number table has {} row(s), expected at least {}",
            grid.wave_numbers_bohr.len(),
            energy_count
        );
    }
    let (phase_energy_count, phase_angular_count, phase_potential_count) = grid.phase_shifts.dim();
    if phase_energy_count < energy_count {
        bail!(
            "SCREEN FMS source phase table has {} energy row(s), expected at least {}",
            phase_energy_count,
            energy_count
        );
    }
    if phase_potential_count != potential_count {
        bail!(
            "SCREEN FMS source phase table has {} potential block(s), expected {}",
            phase_potential_count,
            potential_count
        );
    }

    let global_lmax = global_fms_lmax(&input, max_potential)?;
    let source_angular_count = global_lmax
        .checked_add(1)
        .context("SCREEN FMS angular count is too large")?;
    if grid.angular_count > source_angular_count {
        bail!(
            "screen.inp maxl requests {} angular channel(s), but fms.inp lmaxph supports {}",
            grid.angular_count,
            source_angular_count
        );
    }
    if phase_angular_count < source_angular_count {
        bail!(
            "SCREEN FMS source phase table has {} angular channel(s), expected at least {}",
            phase_angular_count,
            source_angular_count
        );
    }

    let cluster_radius = effective_fms_cluster_radius(&input)?;
    let direct_cutoff = effective_fms_direct_cutoff(&input)?;
    let mut atoms = fms_atoms_from_geom(&input, &geom, max_potential, cluster_radius, 0)?;
    sort_representative_atoms(0, max_potential, &mut atoms)
        .context("failed to prepare SCREEN FMS representative atoms from geom.dat")?;
    let absorber_potential = absorber_potential(&atoms)?;
    let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
        .context("failed to build SCREEN FMS rotation geometry")?;
    let spin_orbit = spin_orbit_coupling_tables(global_lmax)
        .context("failed to build SCREEN FMS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build SCREEN FMS normalization table")?;
    let mean_square_displacements = fms_mean_square_displacements_with_metadata(
        work_dir,
        &input,
        phase,
        damping_metadata,
        &atoms,
    )?;
    let calculated_l = vec![true; source_angular_count];
    let mut cluster_greens = Array2::<Complex>::zeros((energy_count, grid.angular_count));

    let plan = fms_real_space_plan(FmsRealSpacePlanInput {
        // SCREEN consumes only the absorber trace. Keep the full cluster
        // basis for the solve, but pack only the absorber potential's
        // scattering block instead of solving extra RHS blocks for every
        // potential in `fms.inp`.
        lfms: 0,
        minv: 0,
        spin_channels: 1,
        spin_selector: 0,
        atoms: &atoms,
        max_potential,
        global_lmax,
        raw_potential_lmax: &input.lmaxph,
        state_capacity: None,
        spin_orbit: &spin_orbit,
        direct_cutoff,
        mean_square_displacements: mean_square_displacements.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: input.cluster.toler1 as f32,
        zero_tolerance: input.cluster.toler2 as f32,
        full_scattering_matrix_requested: false,
        retain_setup: false,
        retain_pair_tables: false,
        retain_free_propagator: false,
        retain_t_matrix: false,
        retain_system_matrix: false,
    })
    .context("failed to prepare SCREEN FMS real-space plan")?;

    let mut wave_numbers_by_energy = Vec::with_capacity(energy_count);
    let mut phase_shifts_by_energy = Vec::with_capacity(energy_count);
    for energy in 0..energy_count {
        let wave_number = grid.wave_numbers_bohr[energy] / FEFF_BOHR_ANGSTROM;
        wave_numbers_by_energy.push(vec![narrow_complex64_to_complex32(
            wave_number,
            "SCREEN FMS wave number",
        )?]);
        phase_shifts_by_energy.push(
            ldos_source_fms_phase_shifts_for_energy(
                grid.phase_shifts,
                &input,
                energy,
                global_lmax,
                max_potential,
                potential_count,
            )
            .with_context(|| {
                format!(
                    "failed to prepare SCREEN FMS source-grid phase shifts for energy section {}",
                    energy + 1
                )
            })?,
        );
    }
    let points: Vec<FmsRealSpaceEnergyPoint<'_>> = wave_numbers_by_energy
        .iter()
        .zip(phase_shifts_by_energy.iter())
        .map(|(wave_numbers, phases)| FmsRealSpaceEnergyPoint {
            wave_numbers,
            phase_shifts: phases.view(),
        })
        .collect();

    for (energy, result) in fms_real_space_spectrum(&plan, &points)
        .into_iter()
        .enumerate()
    {
        let result = result
            .with_context(|| format!("failed to solve SCREEN FMS energy section {}", energy + 1))?;

        let scattering = result
            .scattering
            .scattering
            .index_axis(Axis(2), absorber_potential);
        for angular in 0..grid.angular_count {
            let phase_shift = grid.phase_shifts[(energy, angular, absorber_potential)];
            cluster_greens[(energy, angular)] =
                screen_fms_cluster_green_trace(scattering, phase_shift, angular).map_err(
                    |source| anyhow::anyhow!("failed to project SCREEN FMS trace: {source}"),
                )?;
        }
    }

    Ok(Some(ScreenFmsClusterGreenHandoff {
        energies_hartree: grid.energy_grid_hartree.to_owned(),
        cluster_greens,
        potential_index: absorber_potential,
        spin_index: 0,
    }))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_pot_scf_fms_source_grid_handoff(
    work_dir: &Path,
    grid: PotScfFmsSourceGridInput<'_>,
) -> Result<PotScfFmsSourceGridHandoff> {
    let mut cache = PotScfFmsPipelineCache::default();
    build_pot_scf_fms_source_grid_handoff_with_cache(work_dir, grid, &mut cache)
}

pub(crate) fn build_pot_scf_fms_source_grid_handoff_with_cache(
    work_dir: &Path,
    grid: PotScfFmsSourceGridInput<'_>,
    cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfFmsSourceGridHandoff> {
    let energy_count = grid.energy_grid_hartree.len();
    if energy_count == 0 {
        bail!("POT SCF FMS source-grid generation requires at least one energy row");
    }
    if grid.angular_count == 0 {
        bail!("POT SCF FMS source-grid generation requires at least one angular channel");
    }

    let potential_count = pot_scf_fms_potential_count(grid.pot)?;
    let max_potential = potential_count
        .checked_sub(1)
        .context("POT SCF FMS generation requires at least one potential")?;
    let (phase_energy_count, phase_angular_count, phase_potential_count) = grid.phase_shifts.dim();
    if phase_energy_count < energy_count {
        bail!(
            "POT SCF FMS phase table has {} energy row(s), expected at least {}",
            phase_energy_count,
            energy_count
        );
    }
    if phase_potential_count != potential_count {
        bail!(
            "POT SCF FMS phase table has {} potential block(s), expected {}",
            phase_potential_count,
            potential_count
        );
    }
    let (reference_energy_count, reference_potential_count) = grid.reference_energies_hartree.dim();
    if reference_energy_count < energy_count || reference_potential_count < potential_count {
        bail!(
            "POT SCF FMS reference-energy table shape {}x{} cannot supply {}x{}",
            reference_energy_count,
            reference_potential_count,
            energy_count,
            potential_count
        );
    }

    let solve_all_potentials = grid.pot.run.lfms1 != 0;
    let fms_input = pot_scf_fms_input_from_pot(grid.pot, if solve_all_potentials { 1 } else { 0 });
    let global_lmax = global_fms_lmax(&fms_input, max_potential)?;
    let source_angular_count = global_lmax
        .checked_add(1)
        .context("POT SCF FMS angular count is too large")?;
    if grid.angular_count > source_angular_count {
        bail!(
            "POT SCF requested {} angular channel(s), but lmaxsc supports {}",
            grid.angular_count,
            source_angular_count
        );
    }
    if phase_angular_count < source_angular_count {
        bail!(
            "POT SCF FMS phase table has {} angular channel(s), expected at least {}",
            phase_angular_count,
            source_angular_count
        );
    }

    let reciprocal_path = work_dir.join("reciprocal.inp");
    let reciprocal_bytes = if reciprocal_path.is_file() {
        Some(
            std::fs::read(&reciprocal_path)
                .with_context(|| format!("failed to read {}", reciprocal_path.display()))?,
        )
    } else {
        None
    };
    cache.validate_reciprocal_snapshot(&reciprocal_path, reciprocal_bytes.as_deref())?;
    let reciprocal = reciprocal_bytes
        .as_deref()
        .map(|bytes| {
            let text = std::str::from_utf8(bytes)
                .with_context(|| format!("{} is not valid UTF-8", reciprocal_path.display()))?;
            ReciprocalInput::parse_str(&reciprocal_path, text)
                .with_context(|| format!("failed to parse {}", reciprocal_path.display()))
        })
        .transpose()?;
    if let Some(reciprocal) = reciprocal.as_ref() {
        ensure!(
            matches!(reciprocal.ispace, 0 | 1),
            "reciprocal.inp ispace must be 0 or 1, got {}",
            reciprocal.ispace
        );
        if reciprocal.ispace == 0 {
            reciprocal
                .cell
                .as_ref()
                .context("reciprocal.inp ispace=0 requires a reciprocal cell block")?;
        }
    }

    if grid.pot.scattering.rfms1 <= 0.0 {
        return build_zero_pot_scf_fms_source_grid_handoff(grid);
    }
    if let Some(reciprocal) = reciprocal.as_ref()
        && reciprocal.ispace == 0
    {
        let cell = reciprocal
            .cell
            .as_ref()
            .context("reciprocal.inp ispace=0 requires a reciprocal cell block")?;
        if work_dir.join("klist.in").is_file() {
            bail!("POT reciprocal FMS klist.in override is not yet supported");
        }
        return build_pot_scf_reciprocal_fms_source_grid_handoff(
            grid,
            &fms_input,
            cell,
            global_lmax,
            max_potential,
            potential_count,
            &reciprocal_path,
            reciprocal_bytes
                .as_deref()
                .context("reciprocal.inp disappeared while preparing POT reciprocal FMS")?,
            cache,
        );
    }

    let geom = read_geom_dat(work_dir)?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match maximum potential {} for POT SCF FMS generation",
            geom.nph,
            max_potential
        );
    }

    let cluster_radius =
        narrow_nonnegative_f64_to_f32(grid.pot.scattering.rfms1, "POT SCF FMS cluster radius")?;
    let direct_cutoff = narrow_nonnegative_f64_to_f32(
        2.0 * grid.pot.scattering.rfms1,
        "POT SCF FMS direct cutoff",
    )?;
    let spin_orbit = spin_orbit_coupling_tables(global_lmax)
        .context("failed to build POT SCF FMS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build POT SCF FMS normalization table")?;
    let calculated_l = vec![true; source_angular_count];
    let channel_count = source_angular_count
        .checked_mul(source_angular_count)
        .context("POT SCF FMS channel count is too large")?;
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("POT SCF FMS signed-l count is too large")?;
    let mut scattering_matrices =
        Array4::<Complex32>::zeros((energy_count, channel_count, channel_count, potential_count));

    let central_potentials: Vec<usize> = if solve_all_potentials {
        vec![0]
    } else {
        (0..potential_count).collect()
    };
    let mut contexts = Vec::with_capacity(central_potentials.len());
    for central_potential in central_potentials {
        let central = i32::try_from(central_potential)
            .context("POT SCF FMS central potential does not fit in i32")?;
        let mut atoms =
            fms_atoms_from_geom(&fms_input, &geom, max_potential, cluster_radius, central)?;
        // FEFF `POT/fmsie` with `lfms1=0` calls `yprep` independently for
        // every central potential. Current FEFF `yprep` leaves its old
        // `sortat` call disabled, so the requested central atom remains in
        // slot zero. Moving potential 0 into that slot makes `fmspack`
        // misidentify every non-absorber context as an absorber solve and
        // leaves the requested `gg(:,:,central)` block empty.
        if solve_all_potentials {
            sort_representative_atoms(central, max_potential, &mut atoms)
                .context("failed to prepare POT SCF FMS representative atoms from geom.dat")?;
        }
        let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
            .context("failed to build POT SCF FMS rotation geometry")?;
        let mean_square_displacements = Array2::<f32>::zeros((atoms.len(), atoms.len()).f());
        contexts.push((
            central_potential,
            atoms,
            geometry,
            mean_square_displacements,
        ));
    }

    // Wave numbers and phase shifts vary per energy but not per central-potential
    // context, so precompute them once and reuse across every context's sweep.
    let mut wave_numbers_by_energy = Vec::with_capacity(energy_count);
    let mut phase_shifts_by_energy = Vec::with_capacity(energy_count);
    for energy in 0..energy_count {
        let reference = grid.reference_energies_hartree[(energy, max_potential)];
        let wave_number = (Complex::new(2.0, 0.0) * (grid.energy_grid_hartree[energy] - reference))
            .sqrt()
            / FEFF_BOHR_ANGSTROM;
        wave_numbers_by_energy.push(vec![narrow_complex64_to_complex32(
            wave_number,
            "POT SCF FMS wave number",
        )?]);
        let phase_shifts = pot_scf_fms_phase_shifts_for_energy(
            grid.phase_shifts,
            &fms_input,
            energy,
            global_lmax,
            max_potential,
            potential_count,
        )
        .with_context(|| {
            format!(
                "failed to prepare POT SCF FMS source-grid phase shifts for energy section {}",
                energy + 1
            )
        })?;
        if phase_shifts.dim() != (1, signed_l_count, potential_count) {
            bail!(
                "POT SCF FMS phase-shift table shape {:?} does not match expected {:?}",
                phase_shifts.dim(),
                (1, signed_l_count, potential_count)
            );
        }
        phase_shifts_by_energy.push(phase_shifts);
    }
    let points: Vec<FmsRealSpaceEnergyPoint<'_>> = wave_numbers_by_energy
        .iter()
        .zip(phase_shifts_by_energy.iter())
        .map(|(wave_numbers, phases)| FmsRealSpaceEnergyPoint {
            wave_numbers,
            phase_shifts: phases.view(),
        })
        .collect();

    for (central_potential, atoms, geometry, mean_square_displacements) in &contexts {
        // FEFF `POT/fmsie` only calls FMS when `inclus.gt.1`; otherwise
        // `gtr` remains zero for this central cluster.
        if atoms.len() <= 1 {
            continue;
        }
        let plan = fms_real_space_plan(FmsRealSpacePlanInput {
            lfms: if solve_all_potentials {
                grid.pot.run.lfms1
            } else {
                0
            },
            minv: 0,
            spin_channels: 1,
            spin_selector: 0,
            atoms,
            max_potential,
            global_lmax,
            raw_potential_lmax: &fms_input.lmaxph,
            state_capacity: None,
            spin_orbit: &spin_orbit,
            direct_cutoff,
            mean_square_displacements: mean_square_displacements.view(),
            xnlm: xnlm.view(),
            rotations: geometry.rotations.view(),
            calculated_l: &calculated_l,
            convergence_tolerance: 0.0,
            zero_tolerance: 0.0,
            full_scattering_matrix_requested: false,
            retain_setup: false,
            retain_pair_tables: false,
            retain_free_propagator: false,
            retain_t_matrix: false,
            retain_system_matrix: false,
        })
        .with_context(|| {
            format!(
                "failed to prepare POT SCF FMS real-space plan for central potential {central_potential}"
            )
        })?;

        for (energy, result) in fms_real_space_spectrum(&plan, &points)
            .into_iter()
            .enumerate()
        {
            let result = result.with_context(|| {
                format!(
                    "failed to solve POT SCF FMS central potential {} energy section {}",
                    central_potential,
                    energy + 1
                )
            })?;

            if solve_all_potentials {
                scattering_matrices
                    .index_axis_mut(Axis(0), energy)
                    .assign(&result.scattering.scattering);
            } else {
                let source = result
                    .scattering
                    .scattering
                    .index_axis(Axis(2), *central_potential);
                let mut scattering_energy = scattering_matrices.index_axis_mut(Axis(0), energy);
                let mut target = scattering_energy.index_axis_mut(Axis(2), *central_potential);
                target.assign(&source);
            }
        }
    }

    pot_scf_fms_source_grid_handoff(PotScfFmsSourceGridHandoffInput {
        energies_hartree: grid.energy_grid_hartree,
        phase_shifts: grid.phase_shifts,
        scattering_matrices: scattering_matrices.view(),
        angular_count: grid.angular_count,
    })
    .context("failed to project POT SCF FMS source-grid traces")
}

fn read_optional_fms_reciprocal_input(work_dir: &Path) -> Result<Option<ReciprocalInput>> {
    let path = work_dir.join("reciprocal.inp");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    ReciprocalInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
        .map(Some)
}

fn integrate_reciprocal_fms_k_points(
    setup: &refeff_io::FmsKspaceHandoffSetup,
    tables: &refeff_core::KSpaceEwaldEnergyTables,
    energy: usize,
    plan: &FmsReciprocalPlan,
    stage: &str,
) -> Result<Array2<Complex32>> {
    let mut accumulator = FmsReciprocalAccumulator::new(plan.order())?;
    let chunk_size = rayon::current_num_threads().max(1);
    for start in (0..setup.k_points.nrows()).step_by(chunk_size) {
        let end = (start + chunk_size).min(setup.k_points.nrows());
        let solved = (start..end)
            .into_par_iter()
            .map(|point| -> Result<Array2<Complex32>> {
                let structure = fms_kspace_non_rel_structure_factor(
                    setup, tables, energy, 0, point,
                )
                .with_context(|| {
                    format!(
                        "failed {stage} reciprocal structure factor at energy {}, k-point {}",
                        energy + 1,
                        point + 1
                    )
                })?;
                plan.solve_k_point(structure.structure_factor.view())
                    .with_context(|| {
                        format!(
                            "failed {stage} reciprocal KKR solve at energy {}, k-point {}",
                            energy + 1,
                            point + 1
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        // FEFF accumulates k-points in mesh order.  Parallelize only the
        // independent solves; apply weights serially in that exact order.
        for (offset, green) in solved.iter().enumerate() {
            accumulator.push(setup.k_weights[start + offset], green.view())?;
        }
    }
    accumulator.finish().with_context(|| {
        format!(
            "failed {stage} reciprocal Brillouin-zone average at energy {}",
            energy + 1
        )
    })
}

fn build_pot_scf_reciprocal_fms_source_grid_handoff(
    grid: PotScfFmsSourceGridInput<'_>,
    fms_input: &FmsInput,
    cell: &ReciprocalCell,
    global_lmax: usize,
    max_potential: usize,
    potential_count: usize,
    reciprocal_path: &Path,
    reciprocal_bytes: &[u8],
    cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfFmsSourceGridHandoff> {
    let energy_count = grid.energy_grid_hartree.len();
    let mut effective_cell = cell.clone();
    if matches!(effective_cell.k_mesh.kind, 2 | 3) {
        effective_cell.k_mesh.total /= 5;
    }
    if effective_cell.k_mesh.total <= 0 {
        bail!(
            "POT reciprocal FMS requires a positive effective k-point request, got {}",
            effective_cell.k_mesh.total
        );
    }
    for (site, &potential) in effective_cell.potentials.iter().enumerate() {
        let valid =
            usize::try_from(potential).is_ok_and(|value| value >= 1 && value <= max_potential);
        if !valid {
            bail!(
                "reciprocal.inp ppot site {} must be in 1..={max_potential}, got {potential}",
                site + 1
            );
        }
    }
    for potential in 1..=max_potential {
        let potential_i32 =
            i32::try_from(potential).context("POT reciprocal potential index is too large")?;
        ensure!(
            effective_cell.potentials.contains(&potential_i32),
            "reciprocal.inp has no unit-cell site for potential {potential}"
        );
    }

    let absorber = usize::try_from(effective_cell.absorber)
        .context("reciprocal.inp absorber index must be positive")?
        .checked_sub(1)
        .context("reciprocal.inp absorber index must be one-based and positive")?;
    if absorber >= effective_cell.atom_count {
        bail!(
            "reciprocal.inp absorber {} exceeds unit-cell atom count {}",
            effective_cell.absorber,
            effective_cell.atom_count
        );
    }

    let mut representative_sites = Vec::with_capacity(potential_count);
    representative_sites.push(absorber);
    for potential in 1..potential_count {
        let potential_i32 =
            i32::try_from(potential).context("POT reciprocal potential index is too large")?;
        let representative = effective_cell
            .potentials
            .iter()
            .enumerate()
            .find_map(|(site, &site_potential)| {
                (site != absorber && site_potential == potential_i32).then_some(site)
            })
            // POT disables the reciprocal core hole, so a unique absorber site
            // is also the neutral representative of its crystallographic
            // potential.  Main FMS must not use this fallback with a core hole.
            .or_else(|| {
                (effective_cell.potentials.get(absorber) == Some(&potential_i32))
                    .then_some(absorber)
            })
            .with_context(|| {
                format!("reciprocal.inp has no unit-cell representative for potential {potential}")
            })?;
        representative_sites.push(representative);
    }

    let mut references = Array2::<Complex>::zeros((energy_count, 1));
    for energy in 0..energy_count {
        references[(energy, 0)] = grid.reference_energies_hartree[(energy, max_potential)];
    }
    let static_setup = cache.reciprocal_static_setup(
        reciprocal_path,
        reciprocal_bytes,
        &effective_cell,
        global_lmax,
        max_potential,
        potential_count,
    )?;
    let setup = fms_kspace_setup_from_static_handoffs(
        static_setup,
        grid.energy_grid_hartree,
        references.view(),
    )
    .context("failed to attach POT reciprocal FMS energy handoffs")?;

    let site_block_order = (global_lmax + 1)
        .checked_mul(global_lmax + 1)
        .context("POT reciprocal FMS site block order is too large")?;
    let energy_matrices = (0..energy_count)
        .into_par_iter()
        .map(|energy| -> Result<Array3<Complex32>> {
            let phases = pot_scf_fms_phase_shifts_for_energy(
                grid.phase_shifts,
                fms_input,
                energy,
                global_lmax,
                max_potential,
                potential_count,
            )
            .with_context(|| {
                format!(
                    "failed to prepare POT reciprocal FMS phase shifts for energy section {}",
                    energy + 1
                )
            })?;
            let t_matrix = fms_kspace_t_matrix(&setup, phases.view()).with_context(|| {
                format!("failed POT reciprocal T matrix at energy {}", energy + 1)
            })?;
            let plan = FmsReciprocalPlan::new(t_matrix.view()).with_context(|| {
                format!(
                    "failed to prepare POT reciprocal KKR plan at energy {}",
                    energy + 1
                )
            })?;
            let tables = fms_kspace_ewald_energy_tables_from_handoff(&setup, energy, 0)
                .with_context(|| {
                    format!("failed POT reciprocal STRCC setup at energy {}", energy + 1)
                })?;
            let integrated =
                integrate_reciprocal_fms_k_points(&setup, &tables, energy, &plan, "POT")?;
            let mut local_blocks =
                Array3::<Complex32>::zeros((site_block_order, site_block_order, potential_count));
            for (potential, &site) in representative_sites.iter().enumerate() {
                let offset = site
                    .checked_mul(site_block_order)
                    .context("POT reciprocal site offset overflowed")?;
                let end = offset
                    .checked_add(site_block_order)
                    .context("POT reciprocal site block overflowed")?;
                ensure!(
                    end <= integrated.nrows(),
                    "POT reciprocal site block [{offset}..{end}) exceeds Green order {}",
                    integrated.nrows()
                );
                for column in 0..site_block_order {
                    for row in 0..site_block_order {
                        local_blocks[(row, column, potential)] =
                            integrated[(offset + row, offset + column)];
                    }
                }
            }
            Ok(local_blocks)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scattering_matrices = Array4::<Complex32>::zeros((
        energy_count,
        site_block_order,
        site_block_order,
        potential_count,
    ));
    for (energy, local_blocks) in energy_matrices.iter().enumerate() {
        scattering_matrices
            .index_axis_mut(Axis(0), energy)
            .assign(local_blocks);
    }

    pot_scf_fms_source_grid_handoff(PotScfFmsSourceGridHandoffInput {
        energies_hartree: grid.energy_grid_hartree,
        phase_shifts: grid.phase_shifts,
        scattering_matrices: scattering_matrices.view(),
        angular_count: grid.angular_count,
    })
    .context("failed to project POT reciprocal FMS source-grid traces")
}

fn build_zero_pot_scf_fms_source_grid_handoff(
    grid: PotScfFmsSourceGridInput<'_>,
) -> Result<PotScfFmsSourceGridHandoff> {
    let energy_count = grid.energy_grid_hartree.len();
    let potential_count = grid.phase_shifts.dim().2;
    let channel_count = grid
        .angular_count
        .checked_mul(grid.angular_count)
        .context("POT SCF zero-FMS channel count is too large")?;
    let scattering_matrices =
        Array4::<Complex32>::zeros((energy_count, channel_count, channel_count, potential_count));
    pot_scf_fms_source_grid_handoff(PotScfFmsSourceGridHandoffInput {
        energies_hartree: grid.energy_grid_hartree,
        phase_shifts: grid.phase_shifts,
        scattering_matrices: scattering_matrices.view(),
        angular_count: grid.angular_count,
    })
    .context("failed to build zero POT SCF FMS source-grid traces")
}

fn pot_scf_fms_potential_count(input: &PotInput) -> Result<usize> {
    if input.control.nph < 0 {
        bail!(
            "POT SCF FMS generation requires nonnegative nph, got {}",
            input.control.nph
        );
    }
    let expected = usize::try_from(input.control.nph)
        .context("POT SCF nph does not fit in usize")?
        .checked_add(1)
        .context("POT SCF potential count is too large")?;
    if input.potentials.len() != expected {
        bail!(
            "POT SCF FMS pot.inp has {} potential row(s), expected {} from nph={}",
            input.potentials.len(),
            expected,
            input.control.nph
        );
    }
    Ok(expected)
}

fn pot_scf_fms_input_from_pot(input: &PotInput, do_fms: i32) -> FmsInput {
    FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: -1,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: input.scattering.rfms1,
            rdirec: 2.0 * input.scattering.rfms1,
            toler1: 0.0,
            toler2: 0.0,
        },
        debye: FmsDebye {
            tk: 0.0,
            thetad: 0.0,
            sig2g: 0.0,
        },
        lmaxph: input
            .potentials
            .iter()
            .map(|potential| potential.lmaxsc)
            .collect(),
        decomposition_channels: 0,
        save_gg_slice: false,
        do_fms,
    }
}

type DebyeWallerFn = for<'a> fn(
    Real,
    Real,
    Real,
    ndarray::ArrayView2<'a, Real>,
    &[usize],
) -> std::result::Result<Real, refeff_core::DebyeError>;

#[derive(Debug, Clone, Copy)]
struct GeneratedFmsSourceMetadata {
    energy_count: usize,
    cluster_atom_count: Option<usize>,
}

fn build_reciprocal_fms_source_outputs(
    work_dir: &Path,
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    cell: &ReciprocalCell,
) -> Result<GeneratedFmsSourceOutputs> {
    if work_dir.join("klist.in").is_file() {
        bail!("reciprocal FMS klist.in override is not yet supported");
    }
    if cell.k_mesh.kind == 3 {
        bail!("reciprocal FMS adaptive ktype=3 integration is not yet supported");
    }
    if input.save_gg_slice {
        bail!("reciprocal FMS save_gg_slice is not yet supported");
    }
    if !matches!(cell.core_hole, 0 | 1) {
        bail!(
            "reciprocal.inp core-hole selector must be 0 or 1, got {}",
            cell.core_hole
        );
    }
    if !cell.core_hole_strength.is_finite() {
        bail!("reciprocal.inp core-hole strength must be finite");
    }
    let absorber = usize::try_from(cell.absorber)
        .context("reciprocal.inp absorber index must be positive")?
        .checked_sub(1)
        .context("reciprocal.inp absorber index must be one-based and positive")?;
    if absorber >= cell.atom_count {
        bail!(
            "reciprocal.inp absorber {} exceeds unit-cell atom count {}",
            cell.absorber,
            cell.atom_count
        );
    }
    let absorber_ground_potential = *cell
        .potentials
        .get(absorber)
        .context("reciprocal.inp absorber has no potential entry")?;
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for reciprocal FMS")?;
    for (site, &potential) in cell.potentials.iter().enumerate() {
        let valid =
            usize::try_from(potential).is_ok_and(|value| value >= 1 && value <= max_potential);
        if !valid {
            bail!(
                "reciprocal.inp ppot site {} must be in 1..={max_potential}, got {potential}",
                site + 1
            );
        }
    }
    for potential in 1..=max_potential {
        let potential_i32 =
            i32::try_from(potential).context("reciprocal FMS potential index is too large")?;
        ensure!(
            cell.potentials.contains(&potential_i32),
            "reciprocal.inp has no unit-cell site for potential {potential}"
        );
    }
    if input.do_fms == 1
        && cell.core_hole == 1
        && !cell
            .potentials
            .iter()
            .enumerate()
            .any(|(site, &potential)| site != absorber && potential == absorber_ground_potential)
    {
        bail!(
            "reciprocal FMS core-hole absorber is the only site with potential {}; FEFF's translated-cell justone correction is not yet supported",
            absorber_ground_potential
        );
    }
    let global_lmax = global_fms_lmax(input, max_potential)?;
    let output_spin_capacity = fms_output_spin_capacity(work_dir, phase.spin_count)?;
    let absorber_lmax = usize::try_from(input.lmaxph[0])
        .context("failed to convert reciprocal FMS absorber lmaxph")?
        .min(global_lmax);
    if input.do_fms != 1 {
        let angular_count = (absorber_lmax + 1)
            .checked_mul(absorber_lmax + 1)
            .context("reciprocal FMS output angular order is too large")?;
        let output_order = angular_count
            .checked_mul(output_spin_capacity)
            .context("reciprocal FMS output spin order is too large")?;
        let sections = (0..phase.energy_count)
            .map(|energy| GgDatSection {
                section_number: energy + 1,
                values: Array2::<Complex>::zeros((output_order, output_order)),
                raw_prefix_lines: None,
            })
            .collect();
        return Ok(GeneratedFmsSourceOutputs {
            gg: GgDatData { sections },
            gg_slice: None,
            gg_diag: None,
            hubbard_gtr_m: None,
            metadata: GeneratedFmsSourceMetadata {
                energy_count: phase.energy_count,
                cluster_atom_count: None,
            },
        });
    }
    let mut energy_probe = Array1::<f64>::zeros(phase.energy_count);
    for energy in 0..phase.energy_count {
        energy_probe[energy] = phase.energy_grid[energy].re;
    }
    let setup = fms_kspace_setup_from_handoffs(
        cell,
        phase.energy_grid.view(),
        phase.reference_energy.view(),
        energy_probe.view(),
        global_lmax,
        phase.spin_count,
        global.control.ispin,
    )
    .context("failed to prepare reciprocal FMS KSPACE handoffs")?;
    let site_block_order = phase
        .spin_count
        .checked_mul(global_lmax + 1)
        .and_then(|value| value.checked_mul(global_lmax + 1))
        .context("reciprocal FMS site block order is too large")?;
    let absorber_offset = absorber
        .checked_mul(site_block_order)
        .context("reciprocal FMS absorber state offset overflowed")?;
    ensure!(
        absorber_offset + site_block_order <= setup.kspace_solver_basis.matrix_order,
        "reciprocal FMS absorber block exceeds state order {}",
        setup.kspace_solver_basis.matrix_order
    );

    let mut sections = Vec::with_capacity(phase.energy_count);
    for energy in 0..phase.energy_count {
        let phases = fms_phase_shifts_for_energy(phase, input, energy, global_lmax, max_potential)
            .with_context(|| {
                format!(
                    "failed to prepare reciprocal FMS phase shifts for energy section {}",
                    energy + 1
                )
            })?;
        let t_matrix = fms_kspace_t_matrix(&setup, phases.view()).with_context(|| {
            format!(
                "failed reciprocal FMS lattice T matrix at energy {}",
                energy + 1
            )
        })?;
        let plan = FmsReciprocalPlan::new(t_matrix.view()).with_context(|| {
            format!(
                "failed to prepare reciprocal KKR plan at energy {}",
                energy + 1
            )
        })?;
        let tables =
            fms_kspace_ewald_energy_tables_from_handoff(&setup, energy, 0).with_context(|| {
                format!("failed reciprocal FMS STRCC setup at energy {}", energy + 1)
            })?;
        let mut integrated =
            integrate_reciprocal_fms_k_points(&setup, &tables, energy, &plan, "FMS")?;

        if cell.core_hole == 1 {
            let mut core_setup = setup.clone();
            Arc::make_mut(&mut core_setup.static_setup)
                .kspace_solver_basis
                .atoms[absorber]
                .potential = 0;
            let core_t = fms_kspace_t_matrix(&core_setup, phases.view()).with_context(|| {
                format!(
                    "failed reciprocal FMS core-hole T matrix at energy {}",
                    energy + 1
                )
            })?;
            let mut difference =
                Array2::<Complex32>::zeros((site_block_order, site_block_order).f());
            let strength = cell.core_hole_strength as f32;
            for column in 0..site_block_order {
                for row in 0..site_block_order {
                    difference[(row, column)] = strength
                        * (t_matrix[(absorber_offset + row, absorber_offset + column)]
                            - core_t[(absorber_offset + row, absorber_offset + column)]);
                }
            }
            integrated = fms_reciprocal_apply_core_hole(FmsReciprocalCoreHoleInput {
                green: integrated.view(),
                absorber_state_offset: absorber_offset,
                site_block_order,
                t_difference: difference.view(),
            })
            .with_context(|| {
                format!(
                    "failed reciprocal FMS core-hole Dyson update at energy {}",
                    energy + 1
                )
            })?;
        }

        if input.debye.sig2g.abs() > 0.001 {
            let wave_number = fms_wave_numbers(phase, energy)?[0];
            let prefactor = reciprocal_sig2_prefactor(input.debye.sig2g, wave_number)?;
            integrated.mapv_inplace(|value| value * prefactor);
        }

        let mut absorber_scattering =
            Array3::<Complex32>::zeros((site_block_order, site_block_order, 1).f());
        for column in 0..site_block_order {
            for row in 0..site_block_order {
                absorber_scattering[(row, column, 0)] =
                    integrated[(absorber_offset + row, absorber_offset + column)];
            }
        }
        let values = fms_gg_section_values(
            absorber_scattering.view(),
            0,
            phase.spin_count,
            output_spin_capacity,
            absorber_lmax,
        )?;
        sections.push(GgDatSection {
            section_number: energy + 1,
            values,
            raw_prefix_lines: None,
        });
    }

    Ok(GeneratedFmsSourceOutputs {
        gg: GgDatData { sections },
        gg_slice: None,
        gg_diag: None,
        hubbard_gtr_m: None,
        metadata: GeneratedFmsSourceMetadata {
            energy_count: phase.energy_count,
            cluster_atom_count: None,
        },
    })
}

fn reciprocal_sig2_prefactor(
    sig2_angstrom_squared: f64,
    wave_number_inv_angstrom: Complex32,
) -> Result<Complex32> {
    let sig2 = narrow_nonnegative_f64_to_f32(sig2_angstrom_squared, "reciprocal FMS SIG2")?;
    if !wave_number_inv_angstrom.re.is_finite() || !wave_number_inv_angstrom.im.is_finite() {
        bail!("reciprocal FMS wave number must be finite");
    }
    // `fms_wave_numbers` has already converted FEFF's atomic-unit momentum to
    // inverse Angstrom.  Thus the native `exp(-sig2 * ck_atomic**2 / bohr**2)`
    // is exactly `exp(-sig2 * ck_angstrom**2)` here.
    let exponent = -(sig2 * wave_number_inv_angstrom * wave_number_inv_angstrom);
    if !exponent.re.is_finite() || !exponent.im.is_finite() {
        bail!("reciprocal FMS SIG2 exponent is not finite");
    }
    let prefactor = exponent.exp();
    if !prefactor.re.is_finite() || !prefactor.im.is_finite() {
        bail!("reciprocal FMS SIG2 prefactor is not finite");
    }
    Ok(prefactor)
}

fn build_fms_source_outputs(
    work_dir: &Path,
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    geom: &GeomDat,
) -> Result<GeneratedFmsSourceOutputs> {
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for FMS generation")?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match phase.bin maximum potential {} for FMS generation",
            geom.nph,
            max_potential
        );
    }
    if input.lmaxph.len() <= max_potential {
        bail!(
            "fms.inp has {} lmaxph value(s), expected at least {} for FMS generation",
            input.lmaxph.len(),
            max_potential + 1
        );
    }

    let global_lmax = global_fms_lmax(input, max_potential)?;
    let cluster_radius = effective_fms_cluster_radius(input)?;
    let direct_cutoff = effective_fms_direct_cutoff(input)?;
    let mut atoms = fms_atoms_from_geom(input, geom, max_potential, cluster_radius, 0)?;
    sort_representative_atoms(0, max_potential, &mut atoms)
        .context("failed to prepare FMS representative atoms from geom.dat")?;
    let absorber_potential = absorber_potential(&atoms)?;
    let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
        .context("failed to build FMS rotation geometry")?;
    let spin_orbit =
        spin_orbit_coupling_tables(global_lmax).context("failed to build FMS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build FMS normalization table")?;
    let output_spin_capacity = fms_output_spin_capacity(work_dir, phase.spin_count)?;
    let mean_square_displacements = fms_mean_square_displacements(work_dir, input, phase, &atoms)?;
    let calculated_l = vec![true; global_lmax + 1];

    let plan = fms_real_space_plan(FmsRealSpacePlanInput {
        lfms: input.do_fms,
        minv: input.control.minv,
        spin_channels: phase.spin_count,
        spin_selector: global.control.ispin,
        atoms: &atoms,
        max_potential,
        global_lmax,
        raw_potential_lmax: &input.lmaxph,
        state_capacity: None,
        spin_orbit: &spin_orbit,
        direct_cutoff,
        mean_square_displacements: mean_square_displacements.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: input.cluster.toler1 as f32,
        zero_tolerance: input.cluster.toler2 as f32,
        full_scattering_matrix_requested: input.save_gg_slice,
        retain_setup: false,
        retain_pair_tables: false,
        retain_free_propagator: false,
        retain_t_matrix: false,
        retain_system_matrix: false,
    })
    .context("failed to prepare FMS real-space plan")?;

    let mut wave_numbers_by_energy = Vec::with_capacity(phase.energy_count);
    let mut phase_shifts_by_energy = Vec::with_capacity(phase.energy_count);
    for energy in 0..phase.energy_count {
        wave_numbers_by_energy.push(fms_wave_numbers(phase, energy)?);
        phase_shifts_by_energy.push(fms_phase_shifts_for_energy(
            phase,
            input,
            energy,
            global_lmax,
            max_potential,
        )?);
    }
    let points: Vec<FmsRealSpaceEnergyPoint<'_>> = wave_numbers_by_energy
        .iter()
        .zip(phase_shifts_by_energy.iter())
        .map(|(wave_numbers, phases)| FmsRealSpaceEnergyPoint {
            wave_numbers,
            phase_shifts: phases.view(),
        })
        .collect();

    let mut sections = Vec::with_capacity(phase.energy_count);
    let mut full_scattering_sections = Vec::new();
    for (energy, result) in fms_real_space_spectrum(&plan, &points)
        .into_iter()
        .enumerate()
    {
        let result =
            result.with_context(|| format!("failed to solve FMS energy section {}", energy + 1))?;

        let absorber_lmax = usize::try_from(input.lmaxph[absorber_potential])
            .context("failed to convert FMS absorber lmaxph")?
            .min(global_lmax);
        let values = fms_gg_section_values(
            result.scattering.scattering.view(),
            absorber_potential,
            phase.spin_count,
            output_spin_capacity,
            absorber_lmax,
        )?;
        if input.save_gg_slice {
            let full_scattering = result
                .scattering
                .full_scattering
                .context("FMS save_gg_slice requires a full LU scattering matrix")?;
            full_scattering_sections.push(full_scattering);
        }
        sections.push(GgDatSection {
            section_number: energy + 1,
            values,
            raw_prefix_lines: None,
        });
    }

    let (gg_slice, gg_diag) = if input.save_gg_slice {
        let block_dimension = fms_saved_scattering_block_dimension(phase.spin_count, global_lmax)?;
        let outputs = build_saved_fms_scattering_outputs(
            &full_scattering_sections,
            atoms.len(),
            block_dimension,
        )?;
        (Some(outputs.0), Some(outputs.1))
    } else {
        (None, None)
    };

    Ok(GeneratedFmsSourceOutputs {
        gg: GgDatData { sections },
        gg_slice,
        gg_diag,
        hubbard_gtr_m: None,
        metadata: GeneratedFmsSourceMetadata {
            energy_count: phase.energy_count,
            cluster_atom_count: (input.do_fms != 0).then_some(atoms.len()),
        },
    })
}

fn build_active_hubbard_fms_source_outputs(
    work_dir: &Path,
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    geom: &GeomDat,
    handoffs: &ActiveHubbardFmsSourceHandoffs,
    central_potential: usize,
    solver_lfms: i32,
    zero_solver_lmax: bool,
) -> Result<GeneratedFmsSourceOutputs> {
    let max_potential = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential for active Hubbard FMS generation")?;
    if geom.nph != max_potential {
        bail!(
            "geom.dat nph {} does not match phase.bin maximum potential {} for active Hubbard FMS generation",
            geom.nph,
            max_potential
        );
    }
    if central_potential > max_potential {
        bail!(
            "active Hubbard FMS central potential {} exceeds maximum potential {}",
            central_potential,
            max_potential
        );
    }
    validate_active_hubbard_fms_source_handoffs(input, phase, handoffs)?;

    let global_lmax = global_fms_lmax(input, max_potential)?;
    let cluster_radius = effective_fms_cluster_radius(input)?;
    let direct_cutoff = effective_fms_direct_cutoff(input)?;
    let central = i32::try_from(central_potential)
        .context("active Hubbard FMS central potential does not fit in i32")?;
    let mut atoms = fms_atoms_from_geom(input, geom, max_potential, cluster_radius, central)?;
    if input.do_fms != 0 {
        sort_representative_atoms(0, max_potential, &mut atoms)
            .context("failed to prepare active Hubbard FMS representative atoms from geom.dat")?;
    }
    let absorber_potential = absorber_potential(&atoms)?;
    if absorber_potential != central_potential {
        bail!(
            "active Hubbard FMS cluster absorber potential {} does not match requested central potential {}",
            absorber_potential,
            central_potential
        );
    }
    let geometry = fms_yprep_geometry(global_lmax, global_lmax, &atoms)
        .context("failed to build active Hubbard FMS rotation geometry")?;
    let spin_orbit = spin_orbit_coupling_tables(global_lmax)
        .context("failed to build active Hubbard FMS spin-orbit tables")?;
    let xnlm = legendre_normalization_table(global_lmax)
        .context("failed to build active Hubbard FMS normalization table")?;
    let output_spin_capacity = fms_output_spin_capacity(work_dir, phase.spin_count)?;
    let mean_square_displacements = fms_mean_square_displacements(work_dir, input, phase, &atoms)?;
    let use_transform = if zero_solver_lmax {
        // FEFF still marks the Hubbard-l transform as enabled, but its
        // all-zero `lmaxphpass` state table contains no such block, so the
        // transform loops never visit it.
        Array2::from_elem((global_lmax + 1, max_potential + 1), false)
    } else {
        active_hubbard_use_transform(handoffs.hubbard_l, global_lmax, max_potential)
    };
    let (transform, inverse) = active_hubbard_transform_tables(&handoffs.transformation, 2)?;
    let zero_lmaxph = vec![0; max_potential + 1];
    let solver_lmaxph = if zero_solver_lmax {
        &zero_lmaxph
    } else {
        &input.lmaxph
    };

    let mut sections = Vec::with_capacity(phase.energy_count);
    let mut full_scattering_sections = Vec::new();
    let output_lmax = global_lmax.max(handoffs.aphase.angular_limit);
    let magnetic_count = output_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .context("active Hubbard FMS magnetic trace dimension is too large")?;
    let mut hubbard_gtr_m_values = Array5::<Complex32>::zeros((
        2,
        phase.energy_count,
        max_potential + 1,
        output_lmax + 1,
        magnetic_count,
    ));
    for energy in 0..phase.energy_count {
        let wave_numbers = fms_wave_numbers(phase, energy)?;
        let magnetic_phase_shifts = fms_hubbard_phase_shifts_for_energy(
            &handoffs.aphase,
            input,
            energy,
            global_lmax,
            max_potential,
            phase.spin_count,
        )?;
        let setup = fms_driver_setup(FmsDriverSetupInput {
            lfms: solver_lfms,
            spin_channels: phase.spin_count,
            atoms: &atoms,
            max_potential,
            global_lmax,
            raw_potential_lmax: solver_lmaxph,
            state_capacity: None,
        })
        .with_context(|| format!("failed to prepare active Hubbard FMS energy {}", energy + 1))?;
        let pair_tables =
            fms_spin_pair_tables(global_lmax, &wave_numbers, &atoms).with_context(|| {
                format!(
                    "failed to build active Hubbard FMS pair tables for energy {}",
                    energy + 1
                )
            })?;
        let free_propagator = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
            states: &setup.state_kets.states,
            atoms: &atoms,
            direct_cutoff,
            rho: pair_tables.rho.view(),
            wave_numbers: &wave_numbers,
            mean_square_displacements: mean_square_displacements.view(),
            xclm: pair_tables.polynomials.view(),
            xnlm: xnlm.view(),
            rotations: geometry.rotations.view(),
        })
        .with_context(|| {
            format!(
                "failed to build active Hubbard FMS free propagator for energy {}",
                energy + 1
            )
        })?;
        let t_matrix = fms_hubbard_t_matrix_table(FmsHubbardTMatrixTableInput {
            states: &setup.state_kets.states,
            atoms: &atoms,
            spin_channels: phase.spin_count,
            spin_selector: global.control.ispin,
            magnetic_phase_shifts: magnetic_phase_shifts.view(),
            spin_orbit: &spin_orbit,
        })
        .with_context(|| {
            format!(
                "failed to build active Hubbard FMS T matrix for energy {}",
                energy + 1
            )
        })?;
        let transformed_t_matrix =
            fms_hubbard_transform_t_matrix(FmsHubbardTMatrixTransformInput {
                states: &setup.state_kets.states,
                atoms: &atoms,
                spin_channels: phase.spin_count,
                use_transform: use_transform.view(),
                transform: transform.view(),
                inverse: inverse.view(),
                t_matrix: t_matrix.view(),
            })
            .with_context(|| {
                format!(
                    "failed to apply active Hubbard FMS T-matrix transform for energy {}",
                    energy + 1
                )
            })?;
        let scattering = fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
            calculate_full_scattering: input.save_gg_slice,
            states: &setup.state_kets.states,
            spin_channels: phase.spin_count,
            global_lmax,
            potential_lmax: &setup.potential_lmax,
            representative_offsets: &setup.state_kets.representative_offsets,
            potential_start: setup.potential_start,
            potential_end: setup.potential_end,
            free_propagator: free_propagator.view(),
            t_matrix: transformed_t_matrix.view(),
        })
        .with_context(|| {
            format!(
                "failed to solve active Hubbard full-potential FMS energy {}",
                energy + 1
            )
        })?;
        if input.save_gg_slice {
            let full_scattering = scattering
                .full_scattering
                .as_ref()
                .context("active Hubbard FMS save_gg_slice requires a full LU scattering matrix")?;
            let full_scattering = fms_hubbard_back_transform_full_scattering(
                FmsHubbardFullScatteringTransformInput {
                    states: &setup.state_kets.states,
                    atoms: &atoms,
                    spin_channels: phase.spin_count,
                    potential_lmax: &setup.potential_lmax,
                    use_transform: use_transform.view(),
                    transform: transform.view(),
                    inverse: inverse.view(),
                    full_scattering: full_scattering.view(),
                },
            )
            .with_context(|| {
                format!(
                    "failed to apply active Hubbard FMS full gg back-transform for energy {}",
                    energy + 1
                )
            })?;
            full_scattering_sections.push(full_scattering);
        }
        let scattering =
            fms_hubbard_back_transform_scattering(FmsHubbardScatteringTransformInput {
                spin_channels: phase.spin_count,
                potential_lmax: &setup.potential_lmax,
                use_transform: use_transform.view(),
                transform: transform.view(),
                inverse: inverse.view(),
                scattering: scattering.scattering.view(),
            })
            .with_context(|| {
                format!(
                    "failed to apply active Hubbard FMS gg back-transform for energy {}",
                    energy + 1
                )
            })?;

        let absorber_lmax = setup.potential_lmax[absorber_potential].min(global_lmax);
        let values = fms_gg_section_values(
            scattering.view(),
            absorber_potential,
            phase.spin_count,
            output_spin_capacity,
            absorber_lmax,
        )?;
        sections.push(GgDatSection {
            section_number: energy + 1,
            values,
            raw_prefix_lines: None,
        });
    }

    // FEFF `fmsdos_h_step2` runs one independent nsp=1 FMS solve for each
    // physical spin. The combined-spin gg above remains the module cache, but
    // the LDOS handoff must follow those two independent solves exactly.
    for spin in 0..2 {
        let source_spin = spin.min(phase.spin_count - 1);
        let spin_transform = transform
            .index_axis(Axis(0), spin)
            .insert_axis(Axis(0))
            .to_owned();
        let spin_inverse = inverse
            .index_axis(Axis(0), spin)
            .insert_axis(Axis(0))
            .to_owned();
        for energy in 0..phase.energy_count {
            let wave_numbers = fms_wave_numbers(phase, energy)?;
            let wave_numbers = [wave_numbers[source_spin]];
            let magnetic_phase_shifts = fms_hubbard_phase_shifts_for_energy(
                &handoffs.aphase,
                input,
                energy,
                global_lmax,
                max_potential,
                2,
            )?;
            let magnetic_phase_shifts = magnetic_phase_shifts
                .index_axis(Axis(0), spin)
                .insert_axis(Axis(0))
                .to_owned();
            let setup = fms_driver_setup(FmsDriverSetupInput {
                lfms: solver_lfms,
                spin_channels: 1,
                atoms: &atoms,
                max_potential,
                global_lmax,
                raw_potential_lmax: solver_lmaxph,
                state_capacity: None,
            })
            .with_context(|| {
                format!(
                    "failed to prepare active Hubbard spin {} FMS energy {}",
                    spin + 1,
                    energy + 1
                )
            })?;
            let pair_tables = fms_spin_pair_tables(global_lmax, &wave_numbers, &atoms)?;
            let free_propagator =
                fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
                    states: &setup.state_kets.states,
                    atoms: &atoms,
                    direct_cutoff,
                    rho: pair_tables.rho.view(),
                    wave_numbers: &wave_numbers,
                    mean_square_displacements: mean_square_displacements.view(),
                    xclm: pair_tables.polynomials.view(),
                    xnlm: xnlm.view(),
                    rotations: geometry.rotations.view(),
                })?;
            let t_matrix = fms_hubbard_t_matrix_table(FmsHubbardTMatrixTableInput {
                states: &setup.state_kets.states,
                atoms: &atoms,
                spin_channels: 1,
                spin_selector: 0,
                magnetic_phase_shifts: magnetic_phase_shifts.view(),
                spin_orbit: &spin_orbit,
            })?;
            let t_matrix = fms_hubbard_transform_t_matrix(FmsHubbardTMatrixTransformInput {
                states: &setup.state_kets.states,
                atoms: &atoms,
                spin_channels: 1,
                use_transform: use_transform.view(),
                transform: spin_transform.view(),
                inverse: spin_inverse.view(),
                t_matrix: t_matrix.view(),
            })?;
            let scattering = fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
                calculate_full_scattering: false,
                states: &setup.state_kets.states,
                spin_channels: 1,
                global_lmax,
                potential_lmax: &setup.potential_lmax,
                representative_offsets: &setup.state_kets.representative_offsets,
                potential_start: setup.potential_start,
                potential_end: setup.potential_end,
                free_propagator: free_propagator.view(),
                t_matrix: t_matrix.view(),
            })?;
            let scattering =
                fms_hubbard_back_transform_scattering(FmsHubbardScatteringTransformInput {
                    spin_channels: 1,
                    potential_lmax: &setup.potential_lmax,
                    use_transform: use_transform.view(),
                    transform: spin_transform.view(),
                    inverse: spin_inverse.view(),
                    scattering: scattering.scattering.view(),
                })?;

            for potential in 0..=max_potential {
                let potential_lmax = setup.potential_lmax[potential].min(global_lmax);
                for angular in 0..=potential_lmax {
                    let magnetic_start = angular * angular;
                    let magnetic_end = (angular + 1) * (angular + 1);
                    for magnetic in magnetic_start..magnetic_end {
                        let phase_shift =
                            handoffs.aphase.values[(potential, spin, energy, angular, magnetic)];
                        let phase_shift = narrow_complex64_to_complex32(
                            phase_shift,
                            "active Hubbard FMS magnetic trace phase shift",
                        )?;
                        hubbard_gtr_m_values[(spin, energy, potential, angular, magnetic)] =
                            normalize_hubbard_fms_trace(
                                scattering[(magnetic, magnetic, potential)],
                                phase_shift,
                                angular,
                            );
                    }
                }
            }
        }
    }

    let (gg_slice, gg_diag) = if input.save_gg_slice {
        let block_dimension = fms_saved_scattering_block_dimension(phase.spin_count, global_lmax)?;
        let outputs = build_saved_fms_scattering_outputs(
            &full_scattering_sections,
            atoms.len(),
            block_dimension,
        )?;
        (Some(outputs.0), Some(outputs.1))
    } else {
        (None, None)
    };

    Ok(GeneratedFmsSourceOutputs {
        gg: GgDatData { sections },
        gg_slice,
        gg_diag,
        hubbard_gtr_m: Some(HubbardLdosGtrMBinData {
            point_count_declared: phase.energy_count,
            horizontal_count: phase.main_energy_count,
            danes_extension_count: phase.auxiliary_energy_count,
            highest_potential_index: max_potential,
            fms_mode: input.do_fms,
            angular_limit: output_lmax,
            values: hubbard_gtr_m_values,
        }),
        metadata: GeneratedFmsSourceMetadata {
            energy_count: phase.energy_count,
            cluster_atom_count: (input.do_fms != 0).then_some(atoms.len()),
        },
    })
}

fn active_hubbard_use_transform(
    hubbard_l: usize,
    global_lmax: usize,
    max_potential: usize,
) -> Array2<bool> {
    let mut values = Array2::<bool>::from_elem((global_lmax + 1, max_potential + 1), false);
    if hubbard_l <= global_lmax && max_potential >= 1 {
        values[(hubbard_l, 1)] = true;
    }
    values
}

fn normalize_hubbard_fms_trace(
    scattering: Complex32,
    phase_shift: Complex32,
    angular: usize,
) -> Complex32 {
    scattering * (Complex32::new(0.0, 2.0) * phase_shift).exp() / (2 * angular + 1) as f32
}

fn active_hubbard_transform_tables(
    data: &HubbardTransformationBinData,
    spin_count: usize,
) -> Result<(Array5<Complex32>, Array5<Complex32>)> {
    if spin_count == 0 || spin_count > data.spin_count() {
        bail!(
            "transformation_hubbard.bin has {} spin block(s), cannot select {spin_count}",
            data.spin_count(),
        );
    }
    let mut transform = Array5::<Complex32>::zeros(
        (
            spin_count,
            data.row_count(),
            data.column_count(),
            data.angular_count(),
            data.potential_count(),
        )
            .f(),
    );
    let mut inverse = Array5::<Complex32>::zeros(transform.raw_dim());

    for spin in 0..spin_count {
        for potential in 0..data.potential_count() {
            for angular in 0..data.angular_count() {
                for column in 0..data.column_count() {
                    for row in 0..data.row_count() {
                        transform[(spin, row, column, angular, potential)] =
                            data.transform[(potential, spin, angular, row, column)];
                        inverse[(spin, row, column, angular, potential)] =
                            data.inverse[(potential, spin, angular, row, column)];
                    }
                }
            }
        }
    }

    Ok((transform, inverse))
}

fn fms_saved_scattering_block_dimension(spin_count: usize, global_lmax: usize) -> Result<usize> {
    let angular_count = global_lmax
        .checked_add(1)
        .and_then(|count| count.checked_mul(count))
        .context("FMS saved scattering angular dimension is too large")?;
    spin_count
        .checked_mul(angular_count)
        .context("FMS saved scattering state block dimension is too large")
}

fn build_saved_fms_scattering_outputs(
    full_scattering_sections: &[Array2<Complex32>],
    atom_count: usize,
    block_dimension: usize,
) -> Result<(RhorrpGgSliceBinData, RhorrpGgDiagBinData)> {
    let first = full_scattering_sections
        .first()
        .context("FMS save_gg_slice requires at least one full scattering section")?;
    let (state_rows, state_columns) = first.dim();
    if state_rows == 0 || state_rows != state_columns {
        bail!("FMS full scattering matrices must be nonempty and square");
    }
    let expected_states = atom_count
        .checked_mul(block_dimension)
        .context("FMS saved scattering state count is too large")?;
    if state_rows != expected_states {
        bail!(
            "FMS save_gg_slice requires fixed-width atom blocks: got {state_rows} state(s), expected {expected_states} from {atom_count} atom(s) and block dimension {block_dimension}"
        );
    }

    let energy_count = full_scattering_sections.len();
    let mut slice_values = Array3::<Complex32>::zeros((energy_count, block_dimension, state_rows));
    let mut diag_values =
        Array4::<Complex32>::zeros((energy_count, atom_count, block_dimension, block_dimension));

    for (energy, full_scattering) in full_scattering_sections.iter().enumerate() {
        if full_scattering.dim() != (state_rows, state_columns) {
            bail!(
                "FMS full scattering section {} shape {:?} does not match first section shape {:?}",
                energy + 1,
                full_scattering.dim(),
                (state_rows, state_columns)
            );
        }

        for row in 0..block_dimension {
            for column in 0..state_rows {
                slice_values[(energy, row, column)] = full_scattering[(row, column)];
            }
        }
        for atom in 0..atom_count {
            let start = atom
                .checked_mul(block_dimension)
                .context("FMS saved scattering atom block offset is too large")?;
            for row in 0..block_dimension {
                for column in 0..block_dimension {
                    diag_values[(energy, atom, row, column)] =
                        full_scattering[(start + row, start + column)];
                }
            }
        }
    }

    Ok((
        RhorrpGgSliceBinData {
            values: slice_values,
        },
        RhorrpGgDiagBinData {
            values: diag_values,
        },
    ))
}

fn global_fms_lmax(input: &FmsInput, max_potential: usize) -> Result<usize> {
    if input.lmaxph.len() <= max_potential {
        bail!(
            "fms.inp has {} lmaxph value(s), expected at least {} for FMS generation",
            input.lmaxph.len(),
            max_potential + 1
        );
    }

    let mut global_lmax = 0;
    for potential in 0..=max_potential {
        let value = input.lmaxph[potential];
        if value < 0 {
            bail!(
                "fms.inp lmaxph({potential}) must be nonnegative for FMS source generation, got {value}"
            );
        }
        let lmax = usize::try_from(value).context("failed to convert FMS lmaxph")?;
        global_lmax = global_lmax.max(lmax);
    }
    Ok(global_lmax)
}

fn phase_supports_fms_lmax(input: &FmsInput, phase: &PhaseBinData) -> bool {
    let Some(max_potential) = phase.potential_count().checked_sub(1) else {
        return false;
    };
    if input.lmaxph.len() <= max_potential {
        return false;
    }

    phase
        .potentials
        .iter()
        .enumerate()
        .all(|(potential, phase_potential)| {
            let raw_lmax = input.lmaxph[potential];
            raw_lmax >= 0
                && usize::try_from(raw_lmax).is_ok_and(|lmax| lmax <= phase_potential.lmax)
        })
}

fn effective_fms_cluster_radius(input: &FmsInput) -> Result<f32> {
    if input.cluster.rfms2 >= 0.0 {
        return narrow_nonnegative_f64_to_f32(input.cluster.rfms2, "FMS cluster radius");
    }
    if input.do_fms == 0 && input.cluster.rfms2.is_finite() {
        return Ok(0.0);
    }
    bail!(
        "FMS source generation requires a nonnegative cluster radius for full FMS, got {}",
        input.cluster.rfms2
    )
}

fn effective_fms_direct_cutoff(input: &FmsInput) -> Result<f32> {
    if input.cluster.rdirec >= 0.0 {
        return narrow_nonnegative_f64_to_f32(input.cluster.rdirec, "FMS direct cutoff");
    }
    if input.do_fms == 0 && input.cluster.rdirec.is_finite() {
        return Ok(0.0);
    }
    bail!(
        "FMS source generation requires a nonnegative direct cutoff for full FMS, got {}",
        input.cluster.rdirec
    )
}

fn fms_atoms_from_geom(
    input: &FmsInput,
    geom: &GeomDat,
    max_potential: usize,
    cluster_radius: f32,
    central_potential: i32,
) -> Result<Vec<FmsAtom>> {
    if geom.atoms.is_empty() {
        bail!("geom.dat requires at least one atom for FMS generation");
    }

    let mut positions = Array2::<f32>::zeros((geom.atoms.len(), 3));
    let mut potentials = Vec::with_capacity(geom.atoms.len());
    for (row, atom) in geom.atoms.iter().enumerate() {
        positions[(row, 0)] = atom.x as f32;
        positions[(row, 1)] = atom.y as f32;
        positions[(row, 2)] = atom.z as f32;
        potentials.push(atom.iph);
    }

    let cluster = fms_yprep_cluster(FmsYprepClusterInput {
        central_potential,
        potentials: &potentials,
        positions: positions.view(),
        cluster_radius,
        cluster_capacity: geom.atoms.len(),
    })
    .context("failed to select FMS cluster from geom.dat")?;
    if input.do_fms != 0 && cluster.atoms.len() <= max_potential {
        bail!(
            "FMS cluster has {} atom(s), but full-potential generation needs representatives through potential {}",
            cluster.atoms.len(),
            max_potential
        );
    }
    Ok(cluster.atoms)
}

fn fms_mean_square_displacements(
    work_dir: &Path,
    input: &FmsInput,
    phase: &PhaseBinData,
    atoms: &[FmsAtom],
) -> Result<Array2<f32>> {
    fms_mean_square_displacements_with_metadata(
        work_dir,
        input,
        Some(phase),
        FmsDebyeDampingMetadata::Phase(phase),
        atoms,
    )
}

#[derive(Debug, Clone, Copy)]
enum FmsDebyeDampingMetadata<'a> {
    Phase(&'a PhaseBinData),
    Pot(&'a PotBinData),
}

fn fms_mean_square_displacements_with_metadata(
    work_dir: &Path,
    input: &FmsInput,
    phase: Option<&PhaseBinData>,
    damping_metadata: FmsDebyeDampingMetadata<'_>,
    atoms: &[FmsAtom],
) -> Result<Array2<f32>> {
    let sig2g = narrow_nonnegative_f64_to_f32(input.debye.sig2g, "FMS SIG2")?;
    if input.control.idwopt == 4 {
        return fms_sig2_dat_mean_square_displacements(
            &work_dir.join("sig2.dat"),
            atoms.len(),
            sig2g,
        );
    }
    if input.control.idwopt == 5 {
        let context = fms_dmdw_context(work_dir)?;
        return fms_dmdw_mean_square_displacements(&context, atoms, sig2g);
    }
    if matches!(input.control.idwopt, 0 | 3) {
        validate_fms_debye_damping_metadata(damping_metadata)?;
    }
    let em_radius_angstrom = if input.control.idwopt == 1 {
        Some(fms_equation_of_motion_radius_angstrom(atoms)?)
    } else {
        None
    };
    let mut spring_context = if matches!(input.control.idwopt, 1 | 2) {
        let phase = phase.context("FMS spring pair damping requires phase.bin metadata")?;
        Some(fms_spring_recursion_context(work_dir, phase)?)
    } else {
        None
    };

    let mut values = Array2::<f32>::zeros((atoms.len(), atoms.len()).f());

    for atom2 in 0..atoms.len() {
        for atom1 in 0..atoms.len() {
            if atom1 == atom2 {
                continue;
            }
            let pair = match input.control.idwopt {
                value if value < 0 => 0.0,
                0 => fms_debye_pair_sigma(
                    input,
                    damping_metadata,
                    atoms[atom1],
                    atoms[atom2],
                    quantum_debye_waller_factor,
                    "FMS correlated Debye-Waller pair damping",
                )?,
                3 => fms_debye_pair_sigma(
                    input,
                    damping_metadata,
                    atoms[atom1],
                    atoms[atom2],
                    classical_debye_waller_factor,
                    "FMS classical Debye-Waller pair damping",
                )?,
                2 => fms_spring_recursion_pair_sigma(
                    spring_context
                        .as_mut()
                        .context("FMS idwopt=2 pair damping requires spring context")?,
                    input,
                    atoms[atom1],
                    atoms[atom2],
                )?,
                1 => fms_spring_equation_of_motion_pair_sigma(
                    spring_context
                        .as_mut()
                        .context("FMS idwopt=1 pair damping requires spring context")?,
                    input,
                    atoms[atom1],
                    atoms[atom2],
                    em_radius_angstrom
                        .context("FMS idwopt=1 pair damping requires an EM radius")?,
                )?,
                value => bail!(
                    "FMS source generation received unexpected idwopt={} Debye-Waller damping",
                    value
                ),
            };
            values[(atom2, atom1)] =
                narrow_nonnegative_f64_to_f32(f64::from(sig2g) + pair, "FMS pair sigsqr")?;
        }
    }

    Ok(values)
}

struct FmsSpringRecursionContext {
    spring: SpringInput,
    matrix: SpringDynamicalMatrix,
    state: SpringRecursionState,
}

fn fms_spring_recursion_context(
    work_dir: &Path,
    phase: &PhaseBinData,
) -> Result<FmsSpringRecursionContext> {
    let spring_path = work_dir.join("spring.inp");
    let spring_text = std::fs::read_to_string(&spring_path)
        .with_context(|| format!("failed to read {}", spring_path.display()))?;
    let spring = parse_spring_input(&spring_text)
        .with_context(|| format!("failed to parse {}", spring_path.display()))?;
    let geom = read_geom_dat(work_dir)?;
    let (positions, atomic_numbers, potential_indices, absorber_index) =
        fms_spring_atom_table(&geom, phase)?;
    let matrix = spring_dynamical_matrix(SpringDynamicalMatrixInput {
        spring: &spring,
        atom_positions_angstrom: positions.view(),
        atomic_numbers: &atomic_numbers,
        potential_indices: &potential_indices,
        absorber_index,
    })
    .context("failed to build FMS idwopt=2 spring dynamical matrix")?;
    Ok(FmsSpringRecursionContext {
        spring,
        matrix,
        state: SpringRecursionState::new(phase.potential_count()),
    })
}

type FmsSpringAtomTable = (Array2<Real>, Vec<usize>, Vec<usize>, usize);

fn fms_spring_atom_table(geom: &GeomDat, phase: &PhaseBinData) -> Result<FmsSpringAtomTable> {
    if geom.atoms.is_empty() {
        bail!("FMS idwopt=2 spring damping requires nonempty geom.dat");
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
        let phase_potential = phase.potentials.get(potential).with_context(|| {
            format!(
                "geom.dat atom {} references missing phase potential {}",
                atom.index, potential
            )
        })?;
        positions[(index, 0)] = atom.x;
        positions[(index, 1)] = atom.y;
        positions[(index, 2)] = atom.z;
        atomic_numbers.push(phase_potential.atomic_number);
        potential_indices.push(potential);
        if potential == 0 {
            absorber_index = Some(index);
        }
    }
    let absorber_index = absorber_index
        .context("FMS idwopt=2 spring damping requires an absorber atom with potential 0")?;
    Ok((positions, atomic_numbers, potential_indices, absorber_index))
}

fn fms_equation_of_motion_radius_angstrom(atoms: &[FmsAtom]) -> Result<Real> {
    if atoms.len() < 2 {
        return Ok(0.0);
    }
    let absorber = atoms
        .iter()
        .position(|atom| atom.potential == 0)
        .unwrap_or(0);
    let mut nearest = Real::INFINITY;
    for (index, atom) in atoms.iter().enumerate() {
        if index == absorber {
            continue;
        }
        let distance = fms_atom_distance_angstrom(atoms[absorber], *atom);
        if distance > 0.0 && distance < nearest {
            nearest = distance;
        }
    }
    if nearest.is_finite() {
        Ok((2.2 * nearest).max(5.0))
    } else {
        Ok(0.0)
    }
}

fn fms_atom_distance_angstrom(first: FmsAtom, second: FmsAtom) -> Real {
    (0..3)
        .map(|axis| (Real::from(first.position[axis]) - Real::from(second.position[axis])).powi(2))
        .sum::<Real>()
        .sqrt()
}

fn fms_spring_pair_path(first: FmsAtom, second: FmsAtom) -> Array2<Real> {
    ndarray::arr2(&[
        [
            Real::from(first.position[0]),
            Real::from(first.position[1]),
            Real::from(first.position[2]),
        ],
        [
            Real::from(second.position[0]),
            Real::from(second.position[1]),
            Real::from(second.position[2]),
        ],
        [
            Real::from(first.position[0]),
            Real::from(first.position[1]),
            Real::from(first.position[2]),
        ],
    ])
}

fn fms_spring_equation_of_motion_pair_sigma(
    context: &mut FmsSpringRecursionContext,
    input: &FmsInput,
    first: FmsAtom,
    second: FmsAtom,
    em_radius_angstrom: Real,
) -> Result<Real> {
    if fms_atom_distance_angstrom(first, second) > em_radius_angstrom {
        return fms_spring_recursion_pair_sigma(context, input, first, second);
    }
    let path = fms_spring_pair_path(first, second);
    let result = equation_of_motion_debye_waller_factor(SpringEquationOfMotionInput {
        matrix: &context.matrix,
        spring: &context.spring,
        temperature: input.debye.tk,
        path_positions_angstrom: path.view(),
    })
    .context("failed to compute FMS idwopt=1 Equation-of-Motion Debye-Waller factor")?;
    update_spring_recursion_state(
        &mut context.state,
        &context.matrix,
        path.view(),
        result.sigma2,
    )
    .context("failed to update FMS idwopt=1 Equation-of-Motion Debye-Waller state")?;
    Ok(result.sigma2)
}

fn fms_spring_recursion_pair_sigma(
    context: &mut FmsSpringRecursionContext,
    input: &FmsInput,
    first: FmsAtom,
    second: FmsAtom,
) -> Result<Real> {
    let path = fms_spring_pair_path(first, second);
    let result = recursion_debye_waller_factor(SpringRecursionInput {
        matrix: &context.matrix,
        temperature: input.debye.tk,
        path_positions_angstrom: path.view(),
        state: Some(&context.state),
    })
    .context("failed to compute FMS idwopt=2 Recursion-method Debye-Waller factor")?;
    update_spring_recursion_state(
        &mut context.state,
        &context.matrix,
        path.view(),
        result.sigma2,
    )
    .context("failed to update FMS idwopt=2 Recursion-method Debye-Waller state")?;
    Ok(result.sigma2)
}

struct FmsDmdwContext {
    atom_positions_bohr: Array2<Real>,
    atom_masses: Array1<Real>,
    mass_weighted_matrix: Array2<Real>,
    rigid_body_modes: Array2<Real>,
    temperatures: Array1<Real>,
    pole_count: usize,
}

fn fms_dmdw_context(work_dir: &Path) -> Result<FmsDmdwContext> {
    let calculation = read_fms_dmdw_calculation(work_dir)?
        .context("FMS idwopt=5 source generation requires enabled dmdw.inp")?;
    validate_fms_dmdw_calculation(&calculation)?;
    let pole_count =
        usize::try_from(calculation.order).context("failed to convert DMDW Lanczos order")?;
    let dym_path = work_dir.join(&calculation.dym_file);
    let dym =
        read_dym(&dym_path).with_context(|| format!("failed to read {}", dym_path.display()))?;
    let atom_positions_bohr = dym.coordinates.cartesian_positions();
    let mass_weighted =
        dmdw_mass_weighted_dynamical_matrix(dym.force_constants.view(), dym.atomic_masses.view())
            .context("failed to build DMDW mass-weighted dynamical matrix for FMS")?;
    let rigid_body_modes =
        dmdw_rigid_body_projection_modes(atom_positions_bohr.view(), dym.atomic_masses.view())
            .context("failed to build DMDW rigid-body modes for FMS")?;

    Ok(FmsDmdwContext {
        atom_positions_bohr,
        atom_masses: dym.atomic_masses,
        mass_weighted_matrix: mass_weighted.matrix,
        rigid_body_modes: rigid_body_modes.projection_modes,
        temperatures: fms_dmdw_temperatures(&calculation)?,
        pole_count,
    })
}

fn fms_dmdw_temperatures(calculation: &DmdwCalculation) -> Result<Array1<Real>> {
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

fn fms_dmdw_mean_square_displacements(
    context: &FmsDmdwContext,
    atoms: &[FmsAtom],
    sig2g: f32,
) -> Result<Array2<f32>> {
    let mut values = Array2::<f32>::zeros((atoms.len(), atoms.len()).f());
    for atom1 in 0..atoms.len().saturating_sub(1) {
        for atom2 in (atom1 + 1)..atoms.len() {
            let pair = fms_dmdw_pair_sigma(context, atoms[atom1], atoms[atom2])?;
            let total =
                narrow_nonnegative_f64_to_f32(f64::from(sig2g) + pair, "FMS DMDW pair sigsqr")?;
            values[(atom2, atom1)] = total;
            values[(atom1, atom2)] = total;
        }
    }
    Ok(values)
}

fn fms_dmdw_pair_sigma(context: &FmsDmdwContext, first: FmsAtom, second: FmsAtom) -> Result<Real> {
    let path_atoms = [
        fms_dmdw_atom_index(context, first)?,
        fms_dmdw_atom_index(context, second)?,
    ];
    let motion = dmdw_path_motion(
        context.atom_positions_bohr.view(),
        context.atom_masses.view(),
        &path_atoms,
    )
    .context("failed to build FMS DMDW pair motion")?;
    let seed = dmdw_project_seed_vector(
        motion.initial_vector.view(),
        context.rigid_body_modes.view(),
    )
    .context("failed to project FMS DMDW pair seed")?;
    let coefficients = dmdw_lanczos_coefficients(
        context.mass_weighted_matrix.view(),
        seed.view(),
        context.pole_count,
    )
    .context("failed to compute FMS DMDW Lanczos coefficients")?;
    let spectrum = dmdw_lanczos_pole_spectrum(
        context.pole_count,
        coefficients.alpha.view(),
        coefficients.beta.view(),
    )
    .context("failed to compute FMS DMDW Lanczos pole spectrum")?;
    let sigma2 = dmdw_debye_waller_factors_from_poles(
        context.temperatures.view(),
        motion.reduced_mass,
        spectrum.angular_frequencies.view(),
        spectrum.weights.view(),
    )
    .context("failed to compute FMS DMDW Debye-Waller factor")?;
    sigma2
        .get(0)
        .copied()
        .context("FMS DMDW Debye-Waller calculation produced no sigma2 values")
}

fn fms_dmdw_atom_index(context: &FmsDmdwContext, atom: FmsAtom) -> Result<usize> {
    let position_bohr = [
        Real::from(atom.position[0]) / FEFF_BOHR_ANGSTROM,
        Real::from(atom.position[1]) / FEFF_BOHR_ANGSTROM,
        Real::from(atom.position[2]) / FEFF_BOHR_ANGSTROM,
    ];
    let mut matched = None;
    for (index, row) in context.atom_positions_bohr.outer_iter().enumerate() {
        let distance = ((row[0] - position_bohr[0]).powi(2)
            + (row[1] - position_bohr[1]).powi(2)
            + (row[2] - position_bohr[2]).powi(2))
        .sqrt();
        if distance < FMS_DMDW_MATCH_TOLERANCE_BOHR {
            if matched.is_some() {
                bail!(
                    "FMS DMDW atom match is ambiguous for position {:?}",
                    atom.position
                );
            }
            matched = Some(index);
        }
    }
    matched.with_context(|| {
        format!(
            "FMS DMDW could not match atom position {:?} to the dynamical matrix",
            atom.position
        )
    })
}

fn fms_sig2_dat_mean_square_displacements(
    path: &Path,
    atom_count: usize,
    sig2g: f32,
) -> Result<Array2<f32>> {
    let mut values = Array2::<f32>::zeros((atom_count, atom_count).f());
    let pair_count = atom_count.saturating_mul(atom_count.saturating_sub(1)) / 2;
    if pair_count == 0 {
        return Ok(values);
    }

    let pair_values = fms_sig2_dat_pair_values(path, pair_count)?;
    let mut index = 0;
    for atom1 in 0..atom_count.saturating_sub(1) {
        for atom2 in (atom1 + 1)..atom_count {
            let pair = pair_values[index];
            index += 1;
            let total =
                narrow_nonnegative_f64_to_f32(f64::from(sig2g) + pair, "FMS sig2.dat pair sigsqr")?;
            values[(atom2, atom1)] = total;
            values[(atom1, atom2)] = total;
        }
    }

    Ok(values)
}

fn fms_sig2_dat_pair_values(path: &Path, pair_count: usize) -> Result<Vec<f64>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut values = Vec::with_capacity(pair_count);
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        if values.len() == pair_count {
            break;
        }

        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 4 {
            bail!(
                "{} line {} has {} field(s), expected at least 4 for sig2.dat pair data",
                path.display(),
                line_index + 1,
                fields.len()
            );
        }
        let _first_index = fields[0].parse::<i32>().with_context(|| {
            format!(
                "failed to parse first sig2.dat atom index in {} line {}",
                path.display(),
                line_index + 1
            )
        })?;
        let _second_index = fields[1].parse::<i32>().with_context(|| {
            format!(
                "failed to parse second sig2.dat atom index in {} line {}",
                path.display(),
                line_index + 1
            )
        })?;
        let pair = fields[2].parse::<f64>().with_context(|| {
            format!(
                "failed to parse sig2.dat pair damping in {} line {}",
                path.display(),
                line_index + 1
            )
        })?;
        let _amplitude = fields[3].parse::<f64>().with_context(|| {
            format!(
                "failed to parse sig2.dat auxiliary value in {} line {}",
                path.display(),
                line_index + 1
            )
        })?;
        if !pair.is_finite() || pair < 0.0 {
            bail!(
                "{} line {} has non-finite or negative sig2.dat pair damping {}",
                path.display(),
                line_index + 1,
                pair
            );
        }
        values.push(pair);
    }
    if values.len() != pair_count {
        bail!(
            "{} contains {} sig2.dat pair row(s), expected {}",
            path.display(),
            values.len(),
            pair_count
        );
    }
    Ok(values)
}

fn fms_debye_pair_sigma(
    input: &FmsInput,
    damping_metadata: FmsDebyeDampingMetadata<'_>,
    first: FmsAtom,
    second: FmsAtom,
    debye_waller: DebyeWallerFn,
    context: &'static str,
) -> Result<Real> {
    let mut positions = Array2::<Real>::zeros((3, 3));
    for (row, atom) in [first, second, first].into_iter().enumerate() {
        positions[(row, 0)] = Real::from(atom.position[0]);
        positions[(row, 1)] = Real::from(atom.position[1]);
        positions[(row, 2)] = Real::from(atom.position[2]);
    }
    let atomic_numbers = [
        fms_atom_atomic_number(damping_metadata, first)?,
        fms_atom_atomic_number(damping_metadata, second)?,
        fms_atom_atomic_number(damping_metadata, first)?,
    ];

    debye_waller(
        input.debye.tk,
        input.debye.thetad,
        fms_average_norman_radius(damping_metadata),
        positions.view(),
        &atomic_numbers,
    )
    .context(context)
}

fn validate_fms_debye_damping_metadata(metadata: FmsDebyeDampingMetadata<'_>) -> Result<()> {
    let average_norman_radius = fms_average_norman_radius(metadata);
    if !average_norman_radius.is_finite() || average_norman_radius <= 0.0 {
        bail!(
            "FMS Debye damping requires a positive finite average Norman radius, got {average_norman_radius}"
        );
    }
    let atomic_numbers: Box<dyn Iterator<Item = (usize, usize)> + '_> = match metadata {
        FmsDebyeDampingMetadata::Phase(phase) => Box::new(
            phase
                .potentials
                .iter()
                .enumerate()
                .map(|(potential, data)| (potential, data.atomic_number)),
        ),
        FmsDebyeDampingMetadata::Pot(pot) => {
            Box::new(pot.atomic_numbers.iter().copied().enumerate())
        }
    };
    for (potential, atomic_number) in atomic_numbers {
        if atomic_number == 0 {
            bail!("FMS Debye damping requires a positive atomic number for potential {potential}");
        }
    }
    Ok(())
}

fn fms_average_norman_radius(metadata: FmsDebyeDampingMetadata<'_>) -> Real {
    match metadata {
        FmsDebyeDampingMetadata::Phase(phase) => phase.scalars.average_norman_radius,
        FmsDebyeDampingMetadata::Pot(pot) => pot.scalars.average_norman_radius,
    }
}

fn fms_atom_atomic_number(metadata: FmsDebyeDampingMetadata<'_>, atom: FmsAtom) -> Result<usize> {
    if atom.potential < 0 {
        bail!(
            "FMS atom potential must be nonnegative for Debye damping, got {}",
            atom.potential
        );
    }
    let potential =
        usize::try_from(atom.potential).context("failed to convert FMS atom potential")?;
    let atomic_number = match metadata {
        FmsDebyeDampingMetadata::Phase(phase) => {
            phase
                .potentials
                .get(potential)
                .with_context(|| {
                    format!(
                        "FMS atom potential {} is outside phase.bin potential table",
                        potential
                    )
                })?
                .atomic_number
        }
        FmsDebyeDampingMetadata::Pot(pot) => {
            *pot.atomic_numbers.get(potential).with_context(|| {
                format!(
                    "FMS atom potential {} is outside pot.bin potential table",
                    potential
                )
            })?
        }
    };
    if atomic_number == 0 {
        bail!("FMS Debye damping requires a positive atomic number for potential {potential}");
    }
    Ok(atomic_number)
}

fn narrow_nonnegative_f64_to_f32(value: f64, name: &'static str) -> Result<f32> {
    let narrowed = value as f32;
    if value.is_finite() && value >= 0.0 && narrowed.is_finite() {
        Ok(narrowed)
    } else {
        bail!("{name} value {value} is negative, non-finite, or out of single-precision range")
    }
}

fn absorber_potential(atoms: &[FmsAtom]) -> Result<usize> {
    let atom = atoms
        .first()
        .context("FMS generation requires a central absorber atom")?;
    if atom.potential < 0 {
        bail!(
            "FMS absorber potential must be nonnegative, got {}",
            atom.potential
        );
    }
    usize::try_from(atom.potential).context("failed to convert FMS absorber potential")
}

fn fms_wave_numbers(phase: &PhaseBinData, energy: usize) -> Result<Vec<Complex32>> {
    let energy_value = *phase
        .energy_grid
        .get(energy)
        .context("phase.bin energy index is out of range for FMS generation")?;
    let mut wave_numbers = Vec::with_capacity(phase.spin_count);
    for spin in 0..phase.spin_count {
        let reference = phase.reference_energy[(energy, spin)];
        let wave =
            (Complex::new(2.0, 0.0) * (energy_value - reference)).sqrt() / FEFF_BOHR_ANGSTROM;
        wave_numbers.push(narrow_complex64_to_complex32(wave, "FMS wave number")?);
    }
    Ok(wave_numbers)
}

fn ldos_source_fms_wave_numbers(
    wave_numbers_bohr: ArrayView2<'_, Complex>,
    energy: usize,
    central_potential: usize,
) -> Result<Vec<Complex32>> {
    let wave_number = *wave_numbers_bohr
        .get((energy, central_potential))
        .context("LDOS RHORRP wave-number index is out of range for FMS generation")?
        / FEFF_BOHR_ANGSTROM;
    Ok(vec![narrow_complex64_to_complex32(
        wave_number,
        "LDOS FMS wave number",
    )?])
}

fn fms_phase_shifts_for_energy(
    phase: &PhaseBinData,
    input: &FmsInput,
    energy: usize,
    global_lmax: usize,
    max_potential: usize,
) -> Result<Array3<Complex32>> {
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("FMS global lmax is too large")?;
    let mut values =
        Array3::<Complex32>::zeros((phase.spin_count, signed_l_count, phase.potential_count()));
    let global_offset = isize::try_from(global_lmax).context("FMS global lmax is too large")?;

    for (potential_index, potential) in phase.potentials.iter().enumerate() {
        if potential_index > max_potential {
            break;
        }
        let raw_lmax = input.lmaxph[potential_index];
        if raw_lmax < 0 {
            bail!(
                "fms.inp lmaxph({potential_index}) must be nonnegative for FMS source generation, got {raw_lmax}"
            );
        }
        let effective_lmax = usize::try_from(raw_lmax)
            .context("failed to convert FMS lmaxph")?
            .min(global_lmax);
        if effective_lmax > potential.lmax {
            bail!(
                "fms.inp lmaxph({potential_index})={} exceeds phase.bin lmax {}",
                effective_lmax,
                potential.lmax
            );
        }
        let potential_offset =
            isize::try_from(potential.lmax).context("FMS potential lmax is too large")?;
        for spin in 0..phase.spin_count {
            let effective_offset =
                isize::try_from(effective_lmax).context("FMS effective lmax is too large")?;
            for signed_l in -effective_offset..=effective_offset {
                let source_l = usize::try_from(signed_l + potential_offset)
                    .context("FMS phase source angular index is out of range")?;
                let target_l = usize::try_from(signed_l + global_offset)
                    .context("FMS phase target angular index is out of range")?;
                let value = potential.phase_shifts[(energy, source_l, spin)];
                values[(spin, target_l, potential_index)] =
                    narrow_complex64_to_complex32(value, "FMS phase shift")?;
            }
        }
    }
    Ok(values)
}

fn ldos_source_fms_phase_shifts_for_energy(
    phase_shifts: ArrayView3<'_, Complex>,
    input: &FmsInput,
    energy: usize,
    global_lmax: usize,
    max_potential: usize,
    phase_potential_count: usize,
) -> Result<Array3<Complex32>> {
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("LDOS FMS global lmax is too large")?;
    let mut values = Array3::<Complex32>::zeros((1, signed_l_count, phase_potential_count));
    let global_offset =
        isize::try_from(global_lmax).context("LDOS FMS global lmax is too large")?;
    let (energy_count, source_angular_count, source_potential_count) = phase_shifts.dim();
    if energy >= energy_count {
        bail!(
            "LDOS RHORRP phase-shift energy index {} exceeds energy count {}",
            energy,
            energy_count
        );
    }
    if source_potential_count <= max_potential {
        bail!(
            "LDOS RHORRP phase-shift table has {} potential block(s), expected at least {}",
            source_potential_count,
            max_potential + 1
        );
    }

    for potential_index in 0..=max_potential {
        let raw_lmax = input.lmaxph[potential_index];
        if raw_lmax < 0 {
            bail!(
                "fms.inp lmaxph({potential_index}) must be nonnegative for LDOS FMS source-grid generation, got {raw_lmax}"
            );
        }
        let effective_lmax = usize::try_from(raw_lmax)
            .context("failed to convert LDOS FMS lmaxph")?
            .min(global_lmax);
        if effective_lmax >= source_angular_count {
            bail!(
                "fms.inp lmaxph({potential_index})={} exceeds LDOS RHORRP phase angular count {}",
                effective_lmax,
                source_angular_count
            );
        }
        let effective_offset =
            isize::try_from(effective_lmax).context("LDOS FMS effective lmax is too large")?;
        for signed_l in -effective_offset..=effective_offset {
            let source_l = signed_l.unsigned_abs();
            let target_l = usize::try_from(signed_l + global_offset)
                .context("LDOS FMS phase target angular index is out of range")?;
            let value = phase_shifts[(energy, source_l, potential_index)];
            values[(0, target_l, potential_index)] =
                narrow_complex64_to_complex32(value, "LDOS FMS phase shift")?;
        }
    }
    Ok(values)
}

fn pot_scf_fms_phase_shifts_for_energy(
    phase_shifts: ArrayView3<'_, Complex>,
    input: &FmsInput,
    energy: usize,
    global_lmax: usize,
    max_potential: usize,
    phase_potential_count: usize,
) -> Result<Array3<Complex32>> {
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("POT SCF FMS global lmax is too large")?;
    let mut values = Array3::<Complex32>::zeros((1, signed_l_count, phase_potential_count));
    let global_offset =
        isize::try_from(global_lmax).context("POT SCF FMS global lmax is too large")?;
    let (energy_count, source_angular_count, source_potential_count) = phase_shifts.dim();
    if energy >= energy_count {
        bail!(
            "POT SCF phase-shift energy index {} exceeds energy count {}",
            energy,
            energy_count
        );
    }
    if source_potential_count <= max_potential {
        bail!(
            "POT SCF phase-shift table has {} potential block(s), expected at least {}",
            source_potential_count,
            max_potential + 1
        );
    }

    for potential_index in 0..=max_potential {
        let raw_lmax = input.lmaxph[potential_index];
        if raw_lmax < 0 {
            bail!(
                "POT SCF lmaxsc({potential_index}) must be nonnegative for FMS source-grid generation, got {raw_lmax}"
            );
        }
        let effective_lmax = usize::try_from(raw_lmax)
            .context("failed to convert POT SCF FMS lmaxsc")?
            .min(global_lmax);
        if effective_lmax >= source_angular_count {
            bail!(
                "POT SCF lmaxsc({potential_index})={} exceeds phase angular count {}",
                effective_lmax,
                source_angular_count
            );
        }
        let effective_offset =
            isize::try_from(effective_lmax).context("POT SCF FMS effective lmax is too large")?;
        for signed_l in -effective_offset..=effective_offset {
            let source_l = signed_l.unsigned_abs();
            let target_l = usize::try_from(signed_l + global_offset)
                .context("POT SCF FMS phase target angular index is out of range")?;
            let value = phase_shifts[(energy, source_l, potential_index)];
            values[(0, target_l, potential_index)] =
                narrow_complex64_to_complex32(value, "POT SCF FMS phase shift")?;
        }
    }
    Ok(values)
}

fn fms_hubbard_phase_shifts_for_energy(
    aphase: &HubbardAphaseBinData,
    input: &FmsInput,
    energy: usize,
    global_lmax: usize,
    max_potential: usize,
    spin_count: usize,
) -> Result<Array4<Complex32>> {
    let signed_l_count = global_lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .context("active Hubbard FMS global lmax is too large")?;
    let magnetic_count = global_lmax
        .checked_add(1)
        .and_then(|value| value.checked_mul(value))
        .context("active Hubbard FMS magnetic dimension is too large")?;
    let mut values = Array4::<Complex32>::zeros((
        spin_count,
        signed_l_count,
        magnetic_count,
        aphase.potential_count(),
    ));
    let global_offset =
        isize::try_from(global_lmax).context("active Hubbard FMS global lmax is too large")?;

    for potential in 0..=max_potential {
        let raw_lmax = input.lmaxph[potential];
        if raw_lmax < 0 {
            bail!(
                "fms.inp lmaxph({potential}) must be nonnegative for active Hubbard FMS source generation, got {raw_lmax}"
            );
        }
        let effective_lmax = usize::try_from(raw_lmax)
            .context("failed to convert active Hubbard FMS lmaxph")?
            .min(global_lmax);
        if effective_lmax > aphase.angular_limit {
            bail!(
                "fms.inp lmaxph({potential})={} exceeds aphase_hubbard.bin angular limit {}",
                effective_lmax,
                aphase.angular_limit
            );
        }
        let effective_offset = isize::try_from(effective_lmax)
            .context("active Hubbard FMS effective lmax is too large")?;
        for spin in 0..spin_count {
            for signed_l in -effective_offset..=effective_offset {
                let source_l = signed_l.unsigned_abs();
                let target_l = usize::try_from(signed_l + global_offset)
                    .context("active Hubbard FMS phase target angular index is out of range")?;
                let first_magnetic = source_l
                    .checked_mul(source_l)
                    .context("active Hubbard FMS magnetic phase lower bound is too large")?;
                let last_magnetic = source_l
                    .checked_add(1)
                    .and_then(|value| value.checked_mul(value))
                    .context("active Hubbard FMS magnetic phase upper bound is too large")?;
                if last_magnetic > aphase.magnetic_count() {
                    bail!(
                        "aphase_hubbard.bin magnetic dimension {} is too small for l={}",
                        aphase.magnetic_count(),
                        source_l
                    );
                }
                for magnetic in first_magnetic..last_magnetic {
                    let value = aphase.values[(potential, spin, energy, source_l, magnetic)];
                    values[(spin, target_l, magnetic, potential)] =
                        narrow_complex64_to_complex32(value, "active Hubbard FMS phase shift")?;
                }
            }
        }
    }

    Ok(values)
}

fn generate_mkgtr_outputs_from_cached_gg(
    work_dir: &Path,
    input: &FmsInput,
    outputs: &[CachedOutputPath],
) -> Result<usize> {
    let fms_path = work_dir.join("fms.bin");
    let gtr_path = work_dir.join("gtr.dat");
    let phase_path = work_dir.join("phase.bin");
    let global_path = work_dir.join("global.inp");
    if !phase_path.is_file() || !global_path.is_file() {
        return Ok(0);
    }
    let global_text = std::fs::read_to_string(&global_path)
        .with_context(|| format!("failed to read {}", global_path.display()))?;
    let global = GlobalInput::parse_str(&global_path, &global_text)
        .with_context(|| format!("failed to parse {}", global_path.display()))?;
    let eels_selectors = mkgtr_eels_polarization_selectors(work_dir)?;
    let needs_fms = !fms_path.is_file()
        || eels_selectors.as_ref().is_some_and(|selectors| {
            read_fms_bin(&fms_path)
                .map(|data| data.spectrum_count() != selectors.len())
                .unwrap_or(true)
        });
    let needs_gtr =
        !gtr_path.is_file() || (eels_selectors.is_some() && read_gtr_dat(&gtr_path).is_err());
    let needs_decomposition = global.control.do_nrixs == 1 && global.control.ldecmx >= 0;
    let fmsl_path = work_dir.join("fmsl.bin");
    let gtrl_path = work_dir.join("gtrl.dat");
    let needs_fmsl = needs_decomposition && !fmsl_path.is_file();
    let needs_gtrl = needs_decomposition && !gtrl_path.is_file();
    if !needs_fms && !needs_gtr && !needs_fmsl && !needs_gtrl && eels_selectors.is_none() {
        return Ok(0);
    }

    let Some(gg_output) = cached_gg_output(outputs) else {
        return Ok(0);
    };
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let gg = read_cached_gg(gg_output)?;
    let generated = build_mkgtr_outputs(input, &global, &phase, &gg, eels_selectors.as_deref())?;
    let mut count = 0;
    let rewrite_fms = needs_fms
        || eels_selectors.is_some()
            && read_fms_bin(&fms_path)
                .and_then(|cached| Ok(fms_bin_string(&cached)? != fms_bin_string(&generated.fms)?))
                .unwrap_or(true);
    let rewrite_gtr = needs_gtr
        || eels_selectors.is_some()
            && read_gtr_dat(&gtr_path)
                .and_then(|cached| Ok(gtr_dat_string(&cached)? != gtr_dat_string(&generated.gtr)?))
                .unwrap_or(true);
    if rewrite_fms {
        write_fms_cache(&fms_path, &generated.fms)?;
        count += 1;
    }
    if rewrite_gtr {
        write_gtr_dat_cache(&gtr_path, &generated.gtr)?;
        count += 1;
    }
    if needs_fmsl {
        let data = generated
            .fmsl
            .as_ref()
            .context("NRIXS/JAS MKGTR did not produce requested fmsl.bin decomposition")?;
        write_fmsl_cache(&fmsl_path, data)?;
        count += 1;
    }
    if needs_gtrl {
        let data = generated
            .gtrl
            .as_ref()
            .context("NRIXS/JAS MKGTR did not produce requested gtrl.dat decomposition")?;
        write_gtrl_dat_cache(&gtrl_path, data)?;
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

fn recover_malformed_gg_outputs_from_source_handoffs(
    work_dir: &Path,
    input: &FmsInput,
    outputs: &[CachedOutputPath],
) -> Result<Option<GeneratedFmsSourceMetadata>> {
    let gg_outputs = outputs
        .iter()
        .filter(|output| {
            matches!(
                output.kind,
                CachedOutputKind::GgBin | CachedOutputKind::GgDat
            )
        })
        .collect::<Vec<_>>();
    if gg_outputs.is_empty() {
        return Ok(None);
    }

    let mut has_malformed_gg = false;
    let mut has_readable_gg = false;
    for output in gg_outputs {
        if read_cached_gg(output).is_ok() {
            has_readable_gg = true;
        } else {
            has_malformed_gg = true;
        }
    }
    if !has_malformed_gg || has_readable_gg {
        return Ok(None);
    }
    if !can_generate_gg_from_source_handoffs(work_dir, input)? {
        return Ok(None);
    }

    generate_gg_outputs_from_source_handoffs(work_dir, input)
}

fn regenerate_stale_gg_outputs_from_source_handoffs(
    work_dir: &Path,
    input: &FmsInput,
    outputs: &[CachedOutputPath],
) -> Result<Option<GeneratedFmsSourceMetadata>> {
    let gg_outputs = outputs
        .iter()
        .filter(|output| {
            matches!(
                output.kind,
                CachedOutputKind::GgBin | CachedOutputKind::GgDat
            )
        })
        .collect::<Vec<_>>();
    if gg_outputs.is_empty() {
        return Ok(None);
    }

    match can_generate_gg_from_source_handoffs(work_dir, input) {
        Ok(true) => {}
        Ok(false) | Err(_) => return Ok(None),
    }

    let Some(generated) = build_gg_outputs_from_source_handoffs(work_dir, input)? else {
        return Ok(None);
    };
    write_generated_hubbard_gtr_m(work_dir, generated.hubbard_gtr_m.as_ref())?;
    let expected = gg_dat_string(&generated.gg)?;
    let mut has_readable_gg = false;
    let mut has_stale_gg = false;
    for output in gg_outputs {
        let Ok(cached) = read_cached_gg(output) else {
            continue;
        };
        has_readable_gg = true;
        if gg_dat_string(&cached)? != expected {
            has_stale_gg = true;
        }
    }
    if !has_readable_gg || !has_stale_gg {
        return Ok(None);
    }

    write_gg_bin_cache(&work_dir.join("gg.bin"), &generated.gg)?;
    write_gg_dat_cache(&work_dir.join("gg.dat"), &generated.gg)?;
    if let Some(slice) = generated.gg_slice {
        write_rhorrp_gg_slice_bin(work_dir.join("gg_slice.bin"), &slice).with_context(|| {
            format!(
                "failed to write {}",
                work_dir.join("gg_slice.bin").display()
            )
        })?;
    }
    if let Some(diag) = generated.gg_diag {
        write_rhorrp_gg_diag_bin(work_dir.join("gg_diag.bin"), &diag).with_context(|| {
            format!("failed to write {}", work_dir.join("gg_diag.bin").display())
        })?;
    }
    Ok(Some(generated.metadata))
}

fn generate_gg_companion_outputs(work_dir: &Path, outputs: &[CachedOutputPath]) -> Result<usize> {
    let has_gg_bin = outputs
        .iter()
        .any(|output| output.kind == CachedOutputKind::GgBin);
    let has_gg_dat = outputs
        .iter()
        .any(|output| output.kind == CachedOutputKind::GgDat);
    if has_gg_bin == has_gg_dat {
        return Ok(0);
    }

    let Some(source) = cached_gg_output(outputs) else {
        return Ok(0);
    };
    let data = read_cached_gg(source)?;
    if has_gg_bin {
        write_gg_dat_cache(&work_dir.join("gg.dat"), &data)?;
    } else {
        write_gg_bin_cache(&work_dir.join("gg.bin"), &data)?;
    }
    Ok(1)
}

fn repair_malformed_gg_companion_outputs(outputs: &[CachedOutputPath]) -> Result<()> {
    let Some(gg_bin) = outputs
        .iter()
        .find(|output| output.kind == CachedOutputKind::GgBin)
    else {
        return Ok(());
    };
    let Some(gg_dat) = outputs
        .iter()
        .find(|output| output.kind == CachedOutputKind::GgDat)
    else {
        return Ok(());
    };

    let bin_data = read_gg_bin(&gg_bin.path)
        .with_context(|| format!("failed to read {}", gg_bin.path.display()));
    let dat_data = read_gg_dat(&gg_dat.path)
        .with_context(|| format!("failed to read {}", gg_dat.path.display()));

    match (bin_data, dat_data) {
        (Ok(_), Ok(_)) => Ok(()),
        (Ok(data), Err(_)) => write_gg_dat_cache(&gg_dat.path, &data),
        (Err(_), Ok(data)) => write_gg_bin_cache(&gg_bin.path, &data),
        (Err(bin_error), Err(dat_error)) => bail!(
            "failed to read paired FMS gg caches from {} and {}: {bin_error}; {dat_error}",
            gg_bin.path.display(),
            gg_dat.path.display()
        ),
    }
}

struct GeneratedMkgtrOutputs {
    fms: FmsBinData,
    gtr: GtrDatData,
    fmsl: Option<FmslBinData>,
    gtrl: Option<GtrlDatData>,
}

fn build_mkgtr_outputs(
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
    eels_selectors: Option<&[usize]>,
) -> Result<GeneratedMkgtrOutputs> {
    if global.control.do_nrixs == 1 {
        build_mkgtr_jas_outputs(input, global, phase, gg)
    } else {
        build_mkgtr_ordinary_outputs(input, global, phase, gg, eels_selectors)
    }
}

fn build_mkgtr_ordinary_outputs(
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
    eels_selectors: Option<&[usize]>,
) -> Result<GeneratedMkgtrOutputs> {
    let absorber_lmax = absorber_lmax(input)?;
    let active_spin_channels = active_spin_channels(global, phase)?;
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .with_context(|| format!("failed to map ihole {} to core-hole kappa", phase.ihole))?;
    let selectors = eels_selectors
        .map(|selectors| selectors.to_vec())
        .unwrap_or_default();
    let transition_tensors = if selectors.is_empty() {
        vec![(global.control.ipol, polarization_tensor(global))]
    } else {
        selectors
            .iter()
            .copied()
            .map(|selector| {
                Ok((
                    1,
                    cartesian_polarization_tensor(selector).with_context(|| {
                        format!("failed to build MKGTR Cartesian polarization tensor {selector}")
                    })?,
                ))
            })
            .collect::<Result<Vec<_>>>()?
    };
    let transition_matrices = transition_tensors
        .into_iter()
        .map(|(polarization, polarization_tensor)| {
            transition_b_matrix(TransitionBMatrixInput {
                lmax: absorber_lmax,
                initial_kappa: core_hole.kappa,
                polarization,
                polarization_tensor,
                multipole: global.control.le2,
                trace_orbital: false,
                spin: global.control.ispin,
                spin_channels: phase.spin_count,
                spin_vector_angle: global.control.angks,
            })
            .context("failed to build MKGTR transition B matrix")
        })
        .collect::<Result<Vec<_>>>()?;
    let green_functions = green_functions_from_gg(gg, phase.energy_count)?;
    let transition_moments = phase.transition_moments.index_axis(Axis(1), 0);
    let trace = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels,
        green_functions: green_functions.view(),
        transition_matrices: &transition_matrices,
        transition_moments,
    })
    .context("failed to fold cached gg matrices into MKGTR trace")?;

    let fms = FmsBinData {
        cluster_radius_angstrom: input.cluster.rfms2,
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
    Ok(GeneratedMkgtrOutputs {
        fms,
        gtr,
        fmsl: None,
        gtrl: None,
    })
}

fn mkgtr_eels_polarization_selectors(work_dir: &Path) -> Result<Option<Vec<usize>>> {
    let path = work_dir.join("eels.inp");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = EelsInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if !input.calculate_elnes {
        return Ok(None);
    }

    let minimum = input.polarization.min;
    let step = input.polarization.step;
    let maximum = input.polarization.max;
    if !(1..=10).contains(&minimum)
        || !(1..=10).contains(&maximum)
        || minimum > maximum
        || step <= 0
    {
        bail!(
            "MKGTR EELS polarization range must satisfy 1 <= min <= max <= 10 and step > 0, got min={minimum}, step={step}, max={maximum}"
        );
    }

    let maximum =
        usize::try_from(maximum).context("failed to convert EELS maximum polarization")?;
    let step = usize::try_from(step).context("failed to convert EELS polarization step")?;
    let mut selector =
        usize::try_from(minimum).context("failed to convert EELS minimum polarization")?;
    let mut selectors = Vec::new();
    while selector <= maximum {
        selectors.push(selector);
        selector = selector
            .checked_add(step)
            .context("EELS polarization selector overflowed")?;
    }
    Ok(Some(selectors))
}

fn validate_requested_mkgtr_eels_outputs(work_dir: &Path) -> Result<()> {
    let Some(selectors) = mkgtr_eels_polarization_selectors(work_dir)? else {
        return Ok(());
    };
    let fms_path = work_dir.join("fms.bin");
    let fms = read_fms_bin(&fms_path)
        .with_context(|| format!("failed to read requested EELS {}", fms_path.display()))?;
    if fms.spectrum_count() != selectors.len() {
        bail!(
            "MKGTR EELS fms.bin has {} spectrum payload(s), expected {} for selectors {:?}",
            fms.spectrum_count(),
            selectors.len(),
            selectors
        );
    }
    let gtr_path = work_dir.join("gtr.dat");
    read_gtr_dat(&gtr_path)
        .with_context(|| format!("failed to read requested EELS {}", gtr_path.display()))?;
    Ok(())
}

fn cartesian_polarization_tensor(selector: usize) -> Result<[[Complex; 3]; 3]> {
    let tensor = refeff_core::polarization_tensor(selector, PolarizationTensorMode::Cartesian)?;
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| tensor[(row, column)])
    }))
}

fn build_mkgtr_jas_outputs(
    input: &FmsInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
) -> Result<GeneratedMkgtrOutputs> {
    let absorber_lmax = absorber_lmax(input)?;
    let active_spin_channels = active_spin_channels(global, phase)?;
    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .with_context(|| format!("failed to map ihole {} to core-hole kappa", phase.ihole))?;
    let indices = genfmt_jas_transition_indices_from_handoffs(global, phase)
        .context("failed to reconstruct MKGTR NRIXS/JAS transition indices")?;
    let transitions = indices
        .transitions
        .iter()
        .map(|transition| {
            Ok(MkgtrJasTransition {
                final_state_kappa: transition.final_state_kappa,
                decomposition_channel: usize::try_from(transition.decomposition_channel)
                    .context("NRIXS/JAS lgind must be nonnegative")?,
                multipole: usize::try_from(transition.total_angular_momentum_channel)
                    .context("NRIXS/JAS ljind must be nonnegative")?,
                orbital_angular_momentum: usize::try_from(transition.orbital_angular_momentum)
                    .context("NRIXS/JAS lind must be nonnegative")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let q_angles = genfmt_jas_q_angles_from_handoffs(global, phase)
        .context("failed to reconstruct MKGTR NRIXS/JAS q angles")?;
    let q_weights = if global.q_control.mixdff {
        q_angles.weights
    } else {
        q_angles.weights.mapv(|weight| weight.sqrt())
    };
    let (q_pair_mode, q_pair_cosines) = mkgtr_jas_q_pair_setup(global, phase.q_count)?;
    let max_decomposition_channel = if global.control.ldecmx >= 0 {
        let maximum = usize::try_from(global.control.ldecmx)
            .context("NRIXS/JAS ldecmx must be nonnegative")?;
        if input.decomposition_channels != global.control.ldecmx {
            bail!(
                "NRIXS/JAS fms.inp decomposition channel {} does not match global.inp ldecmx {}",
                input.decomposition_channels,
                global.control.ldecmx
            );
        }
        Some(maximum)
    } else {
        None
    };
    let green_functions = green_functions_from_gg(gg, phase.energy_count)?;
    let result = mkgtr_jas_green_trace(MkgtrJasGreenTraceInput {
        active_spin_channels,
        max_angular_momentum: absorber_lmax,
        green_functions: green_functions.view(),
        transition_moments: phase.transition_moments.view(),
        initial_kappa: core_hole.kappa,
        initial_j2: indices.initial_j2,
        final_j2_max: indices.final_j2_max,
        final_lj_max: indices.final_lj_max,
        transitions: &transitions,
        q_phases: q_angles.phases.view(),
        q_beta_angles: q_angles.beta_angles.view(),
        q_weights: q_weights.view(),
        q_pair_cosines: q_pair_cosines.view(),
        ellipticity: global.control.elpty,
        q_pair_mode,
        max_decomposition_channel,
    })
    .context("failed to fold cached gg matrices into NRIXS/JAS MKGTR trace")?;

    let highest_potential_index = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin requires at least one potential")?;
    let spectra = result.trace.clone().insert_axis(Axis(0));
    let fms = FmsBinData {
        cluster_radius_angstrom: input.cluster.rfms2,
        energy_count: phase.energy_count,
        main_energy_count: phase.main_energy_count,
        auxiliary_energy_count: phase.auxiliary_energy_count,
        highest_potential_index,
        pad_width: phase.pad_width,
        // `getgtrjas.f90` writes the historical five-field header.
        declared_spectrum_count: None,
        spectra,
    };
    let gtr = GtrDatData {
        energy: phase.energy_grid.clone(),
        trace: result.trace,
    };

    let (fmsl, gtrl) = match (max_decomposition_channel, result.decomposed_traces.as_ref()) {
        (Some(maximum), Some(decomposed)) => {
            let fmsl = FmslBinData {
                pad_width: phase.pad_width,
                max_decomposition_channel: maximum,
                traces: decomposed.clone(),
            };
            // FEFF's readable companion historically emits lg2=0:2 for every
            // lg1=0:ldecmx, even though fmsl.bin retains the complete square.
            let component_count = maximum
                .checked_add(1)
                .and_then(|value| value.checked_mul(3))
                .context("NRIXS/JAS gtrl.dat component count overflowed")?;
            let mut readable = Array2::<Complex>::zeros((phase.energy_count, component_count));
            for energy in 0..phase.energy_count {
                for lg1 in 0..=maximum {
                    for lg2 in 0..=2.min(maximum) {
                        readable[(energy, lg1 * 3 + lg2)] = decomposed[(energy, lg2, lg1)];
                    }
                }
            }
            let gtrl = GtrlDatData {
                energy_index: Array1::from_iter(1..=phase.energy_count),
                energy: phase.energy_grid.mapv(|value| value.re),
                decomposed_trace: readable,
            };
            (Some(fmsl), Some(gtrl))
        }
        (None, None) => (None, None),
        _ => bail!("internal NRIXS/JAS MKGTR decomposition state is inconsistent"),
    };

    Ok(GeneratedMkgtrOutputs {
        fms,
        gtr,
        fmsl,
        gtrl,
    })
}

fn mkgtr_jas_q_pair_setup(
    global: &GlobalInput,
    q_count: usize,
) -> Result<(MkgtrJasQPairMode, Array2<Real>)> {
    if !global.q_control.mixdff {
        return Ok((
            MkgtrJasQPairMode::Diagonal,
            Array2::from_shape_fn(
                (q_count, q_count),
                |(left, right)| {
                    if left == right { 1.0 } else { 0.0 }
                },
            ),
        ));
    }
    let mode = match global.q_control.imdff {
        1 => MkgtrJasQPairMode::AllPairs,
        2 => MkgtrJasQPairMode::FirstToSecond,
        value => bail!("invalid NRIXS/JAS MDFF option imdff={value}"),
    };
    let mdff = global
        .mdff
        .as_ref()
        .context("NRIXS/JAS mixdff requires global.inp MDFF pair cosines")?;
    let expected = q_count
        .checked_mul(q_count)
        .context("NRIXS/JAS q-pair count overflowed")?;
    if mdff.cosines.len() != expected {
        bail!(
            "NRIXS/JAS global.inp has {} MDFF pair cosine(s), expected {}",
            mdff.cosines.len(),
            expected
        );
    }
    let cosines = Array2::from_shape_vec((q_count, q_count), mdff.cosines.clone())
        .context("failed to shape NRIXS/JAS q-pair cosines")?;
    Ok((mode, cosines))
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
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn ensure_mkgtr_module_log(path: &Path) -> Result<usize> {
    let mut data = if path.is_file() {
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        generated_cached_fms_module_log(0)
    };
    if !data
        .lines
        .iter()
        .any(|line| line.contains("Done with module: MKGTR."))
    {
        data.lines.extend([
            String::new(),
            "MKGTR: Tracing over Green's function ...".to_string(),
            "Done with module: MKGTR.".to_string(),
        ]);
        data.line_terminators
            .extend(["\n".to_string(), "\n".to_string(), "\n".to_string()]);
    }
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_generated_cached_fms_module_log(
    path: &Path,
    generated_mkgtr_count: usize,
) -> Result<usize> {
    let data = generated_cached_fms_module_log(generated_mkgtr_count);
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn generated_cached_fms_module_log(generated_mkgtr_count: usize) -> ModuleLogData {
    let mut lines = vec![
        "FMS calculation of full Green's function ...".to_string(),
        "FEFF-serial using 1 thread.".to_string(),
        "Done with module: FMS.".to_string(),
    ];

    if generated_mkgtr_count > 0 {
        lines.push(String::new());
        lines.push("MKGTR: Tracing over Green's function ...".to_string());
        lines.push("Done with module: MKGTR.".to_string());
    }

    let line_terminators = vec!["\n".to_string(); lines.len()];
    ModuleLogData {
        lines,
        line_terminators,
    }
}

fn write_generated_fms_module_log(
    path: &Path,
    input: &FmsInput,
    metadata: &GeneratedFmsSourceMetadata,
    generated_mkgtr_count: usize,
) -> Result<usize> {
    let mut lines = vec![
        "FMS calculation of full Green's function ...".to_string(),
        "FEFF-serial using 1 thread.".to_string(),
        format!("Using {:5} energy points.", metadata.energy_count),
    ];
    if let Some(line) = fms_debye_log_line(input.control.idwopt) {
        lines.push(line.to_string());
    }
    lines.push("xprep done".to_string());
    if let Some(atom_count) = metadata.cluster_atom_count.filter(|count| *count > 1) {
        lines.push(format!("FMS for a cluster of {atom_count:4} atoms"));
        lines.extend(fms_energy_progress_log_lines(metadata.energy_count));
    }
    lines.push("Done with module: FMS.".to_string());

    if generated_mkgtr_count > 0 {
        lines.push(String::new());
        lines.push("MKGTR: Tracing over Green's function ...".to_string());
        lines.push("Done with module: MKGTR.".to_string());
    }

    let line_terminators = vec!["\n".to_string(); lines.len()];
    let data = ModuleLogData {
        lines,
        line_terminators,
    };
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn fms_energy_progress_log_lines(energy_count: usize) -> impl Iterator<Item = String> {
    let mut next_report = 1;
    (1..=energy_count).filter_map(move |energy| {
        if energy < next_report {
            return None;
        }
        let line = format!("Energy point {energy:4}/{energy_count:4}");
        next_report = if next_report == 1 {
            10
        } else {
            next_report + 10
        };
        Some(line)
    })
}

fn fms_debye_log_line(idwopt: i32) -> Option<&'static str> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    FmsBin,
    FmslBin,
    GgBin,
    GgDat,
    GgSliceBin,
    GgDiagBin,
    GtrBin,
    GtrDat,
    GtrlDat,
}

impl CachedOutputKind {
    const fn is_fms_solver_output(self) -> bool {
        matches!(
            self,
            Self::GgBin | Self::GgDat | Self::GgSliceBin | Self::GgDiagBin
        )
    }

    const fn is_mkgtr_output(self) -> bool {
        matches!(
            self,
            Self::FmsBin | Self::FmslBin | Self::GtrBin | Self::GtrDat | Self::GtrlDat
        )
    }
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
        } else if name == "gg_slice.bin" {
            Some(CachedOutputKind::GgSliceBin)
        } else if name == "gg_diag.bin" {
            Some(CachedOutputKind::GgDiagBin)
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

#[cfg(all(test, feature = "full"))]
mod tests;
