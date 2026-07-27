#![allow(clippy::too_many_arguments)]

use std::{
    fmt::Write as _,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{
    Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, Axis, Slice,
};
use num_complex::Complex64;
use refeff_core::xsph::{
    XsphPhaseRadialHeaderInput, XsphPhaseRadialIndicesInput, XsphPhaseRadialOutputInput,
    XsphRhorrpPhaseEnergyMeshInput, XsphTdldaChannelBasis, XsphTdldaChannelBasisInput,
    xsph_phase_radial_header, xsph_phase_radial_indices, xsph_phase_radial_output,
    xsph_rhorrp_phase_energy_mesh, xsph_tdlda_channel_basis,
};
use refeff_core::{
    BroadenedHedinLundqvistTable, DiracSpinorGridInput, ExchangeError, ExcitationPole,
    FEFF_BOHR_ANGSTROM, FEFF_FERMI_MOMENTUM_FACTOR, FEFF_HARTREE_EV, FovrgC3PotentialInput,
    FovrgDiracSolverInput, FovrgInitialPhotoelectronInput, FovrgOrbitalSetupInput, LOUCKS_DELTA,
    ManyPoleSelfEnergyInput, MuffinTinInterstitialParametersInput, MuffinTinOverlapNeighbor,
    PotentialGridInput, XCPOT_MPSE_GRID_POINTS, XcpotFermiCache, XcpotInput,
    XcpotManyPoleDensityGridInput, XcpotManyPoleSelfEnergyInput, XcpotResult, XsphAxafsInput,
    XsphEmptyCellPhaseInput, XsphError, XsphHubbardPhaseAssignmentInput,
    XsphHubbardPhasePotentialInput, XsphJasOrthogonalityCorrectionInput,
    XsphJasPhaseEnergyMeshInput, XsphJasRadialCrossIntegralInput, XsphJasRadialIntegralInput,
    XsphNrixsTransitionIndicesInput, XsphPhaseAngularLimitInput, XsphPhaseChannelPlanInput,
    XsphPhaseCutoffInput, XsphPhaseEnergyDecision, XsphPhaseEnergyMesh84Input,
    XsphPhaseEnergySetupInput, XsphPhaseGridPreparation, XsphPhaseGridPreparationInput,
    XsphPhasePlasmonPoleSetup, XsphPhasePlasmonPoleSetupInput, XsphPhaseUserGridInput,
    XsphPhaseUserGridKind, XsphPhaseUserGridMinimum, XsphPhaseUserGridRecord,
    XsphPhaseUserRegularGrid, XsphRadialIntegralInput, XsphRadialIntegralMode,
    XsphTdldaAngularKernel, XsphTdldaAngularKernelInput, XsphTdldaBroadenedChannelSpectra,
    XsphTdldaChannelBroadeningInput, XsphTdldaChannelMultipliers, XsphTdldaChannelMultipliersInput,
    XsphTdldaChannelSpectraInput, XsphTdldaCoulombFields, XsphTdldaCoulombFieldsInput,
    XsphTdldaDirectKernel, XsphTdldaDirectKernelInput, XsphTdldaEnergyRows,
    XsphTdldaEnergyRowsInput, XsphTdldaNonlocalExchangeInput,
    XsphTdldaProjectorOrthogonalizationInput, XsphTdldaProjectorSelector, XsphTdldaRadialKernel,
    XsphTdldaRadialKernelInput, XsphTdldaRawResponse, XsphTdldaRawResponseInput,
    XsphTdldaResponseConditioningInput, XsphTdldaRowWaveNumbers, XsphTdldaRowWaveNumbersInput,
    XsphTdldaScreenedDipoleInput, XsphTdldaWeightedResponse, XsphTdldaWeightedResponseInput,
    XsphTdldaXmuChannelInput, XsphTdldaXsedgeRowsInput, XsphThermalPhaseEnergyMeshInput,
    XsphTransitionMultipole, XsphXrayBesselTableInput, XsphXsectBcoefNonstandardEnergyRowInput,
    XsphXsectBcoefStandardEnergyRowFieldsInput, XsphXsectBcoefStandardTransitionField,
    XsphXsectBcoefWeightsInput, XsphXsectEnergyDecision, XsphXsectEnergySetupInput,
    XsphXsectHoleNormalizationInput, XsphXsectIrregularChannel, XsphXsectIrregularChannelInput,
    XsphXsectPhiscfContributionPlanInput, XsphXsectPhiscfContributionPlanRow,
    XsphXsectPhiscfLocalFieldInput, XsphXsectPhiscfRadialSolverSetup,
    XsphXsectPhiscfRadialSolverSetupInput, XsphXsectPhiscfWfirdcContributionInput,
    XsphXsectPhiscfWfirdcContributions, XsphXsectPhiscfWfirdcContributionsInput,
    XsphXsectRegularChannel, XsphXsectRegularChannelInput, XsphXsectScreenedFieldInput,
    XsphXsectTransition, XsphXsectTransitionPlanInput, core_hole_quantum_numbers,
    core_hole_width_ev, dirac_hara_exchange_potential, fix_dirac_spinor_grid, fix_potential_grid,
    fovrg_c3_potential, fovrg_orbital_setup, karasiev_sjostrom_dufty_trickey_vxc,
    legendre_polynomials_into, make_excitation_poles, many_pole_self_energy,
    muffin_tin_interstitial_parameters, perdew_zunger_vxc, perrot_dharma_wardana_vxc, somm2, terp,
    von_barth_hedin_potential, wave_number_from_hartree, wigner_3j, xcpot,
    xcpot_many_pole_density_grid, xcpot_with_broadened_table, xsph_axafs, xsph_empty_cell_phase,
    xsph_hubbard_phase_assignments, xsph_hubbard_phase_potential_shifts,
    xsph_hubbard_phase_reference_tail, xsph_jas_orthogonality_correction,
    xsph_jas_phase_energy_mesh, xsph_jas_radial_cross_integral, xsph_jas_radial_integral,
    xsph_lj_needed_flags, xsph_minimize_calculations, xsph_nrixs_transition_indices,
    xsph_nrixs_transition_weights, xsph_phase_angular_limit, xsph_phase_channel_plan,
    xsph_phase_cutoff, xsph_phase_energy_mesh_84, xsph_phase_energy_mesh_user,
    xsph_phase_energy_setup, xsph_phase_grid_preparation, xsph_phase_plasmon_pole_setup,
    xsph_phase_reference_tail, xsph_phase_self_energy_summary, xsph_q_bessel_table,
    xsph_radial_integral, xsph_regular_phase_channel, xsph_tdlda_angular_kernel,
    xsph_tdlda_broaden_channel_spectra, xsph_tdlda_channel_multipliers, xsph_tdlda_channel_spectra,
    xsph_tdlda_condition_response, xsph_tdlda_coulomb_fields, xsph_tdlda_decode_projector_selector,
    xsph_tdlda_direct_kernel, xsph_tdlda_energy_rows, xsph_tdlda_nonlocal_exchange_integrals,
    xsph_tdlda_projector_orthogonalization, xsph_tdlda_radial_kernel_integrals,
    xsph_tdlda_raw_response, xsph_tdlda_row_wave_numbers, xsph_tdlda_screened_dipoles,
    xsph_tdlda_separation_function, xsph_tdlda_weight_response, xsph_tdlda_xsedge_rows,
    xsph_thermal_phase_energy_mesh, xsph_vertical_energy_mesh_84, xsph_xray_bessel_table,
    xsph_xsect_bcoef_nonstandard_energy_row,
    xsph_xsect_bcoef_standard_energy_row_with_transition_fields, xsph_xsect_bcoef_weights,
    xsph_xsect_energy_setup, xsph_xsect_hole_normalization, xsph_xsect_irregular_channel,
    xsph_xsect_phiscf_contribution_plan, xsph_xsect_phiscf_local_field,
    xsph_xsect_phiscf_radial_solver_setup, xsph_xsect_phiscf_wfirdc_contributions,
    xsph_xsect_regular_channel, xsph_xsect_screened_field_setup, xsph_xsect_transition_plan,
};
#[cfg(test)]
use refeff_core::{
    XsphTdldaProjectedKernel, XsphTdldaProjectedKernelInput, xsph_tdlda_projected_kernel,
};
use refeff_io::{
    AxafsDatData, EelsInput, EmeshBinData, EmeshDatData, GeomDat, GlobalInput, GridInput, GridKind,
    GridMinimum, GridRecord, HubbardAphaseBinData, HubbardInput, HubbardVnlmBinData, ModuleLogData,
    MpseDatData, PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData, PotInput,
    RhorrpConfigOrbitalTables, XmuDatData, XseclBinData, XseclBinTransition, XseclDatData,
    XseclDatHeader, XseclFromXsphNrixs, XseclFromXsphNrixsInput, XsectDatData,
    XsectDatFromXsphSpinInput, XsectDatScalars, XsedgeDatData, XsedgeDatFromTdldaRowsInput,
    XsphAdvanced, XsphInput, XsphRlDatData, XsphRlDatRecord, axafs_dat_from_xsph_axafs,
    axafs_dat_string, emesh_bin_from_phase_bin, emesh_dat_from_phase_bin,
    exc_dat_from_excitation_poles, exc_dat_string,
    format::{write_fortran_exp, write_fortran_zero_scaled_exp},
    mpse_dat_string, parse_axafs_dat, parse_mpse_dat,
    phase_bin::{PHASE_BIN_DEFAULT_PAD_WIDTH, PHASE_BIN_DEFAULT_TRANSITION_COUNT},
    read_aphase_hubbard_bin_inferred, read_axafs_dat, read_bphl_dat, read_config_dat,
    read_emesh_bin, read_emesh_dat, read_exc_dat, read_grid_inp, read_loss_dat,
    read_module_log_dat, read_mpse_dat, read_phase_bin, read_pot_bin, read_v_hubbard_bin_inferred,
    read_wscrn_dat, read_xmu_dat, read_xsecl_bin, read_xsecl_dat, read_xsecl2_dat, read_xsect_dat,
    read_xsedge_dat, read_xsph_rl_dat, rhorrp_orbital_tables_from_config_dat,
    write_aphase_hubbard_bin, write_axafs_dat, write_emesh_bin, write_emesh_dat, write_exc_dat,
    write_module_log_dat, write_mpse_dat, write_phase_bin, write_xsecl_bin, write_xsecl_dat,
    write_xsecl2_dat, write_xsect_dat, write_xsedge_dat, write_xsph_rl_dat, xsecl_from_xsph_nrixs,
    xsect_dat_ff2x_handoff, xsect_dat_from_xsph_spin_merge, xsedge_dat_from_tdlda_rows,
};

use crate::{screen, work_dir_for_input};

// Modern FEFF sizes the ordinary phase-energy mesh with `nex=2000`.  Keeping
// the old 120-point work-array limit silently truncated otherwise valid
// ispec=1 meshes (for example, long-k-range XANES and K-space calculations).
const XSPH_DEFAULT_PHASE_MESH_CAPACITY: usize = 2000;
const XSPH_LEGACY_XANES_PHASE_MESH_CAPACITY: usize = 102;
const XSPH_DANES_PHASE_MESH_CAPACITY: usize = 150;
const XSPH_NRIXS_PHASE_MESH_CAPACITY: usize = 179;
const XSPH_FPRIME_PHASE_MESH_CAPACITY: usize = 200;
const XSPH_COMPILED_PHASE_MESH_CAPACITY: usize = 2000;
/// FEFF `DimsMod::nrptx`, the Loucks radial work-grid length used by XSPH.
const XSPH_PHASE_RADIAL_GRID_CAPACITY: usize = 1251;
const XSPH_TDLDA_MESH_HORIZONTAL_COUNT: usize = 100;
const XSPH_TDLDA_MESH_EXTRA_COUNT: usize = 20;
const XSPH_TDLDA_MESH_LEFT_EV: f64 = -20.0;
const XSPH_TDLDA_MESH_RIGHT_EV: f64 = 200.0;
const XSPH_TDLDA_MESH_EXTRA_RIGHT_EV: f64 = 450.0;
const XSPH_TDLDA_GENERATED_BASIS_ENERGIES_EV: [f64; 5] = [36.5, 110.0, 220.0, 370.0, 550.0];
const XSPH_NRIXS_L2LP_SENTINEL: i32 = 30;
const XSPH_NRIXS_MAX_FINAL_ANGULAR_MOMENTUM: usize = 24;
/// FEFF `DimsMod::ltot`, the signed-l phase table capacity used by `phase.f90`.
const XSPH_PHASE_MAX_ANGULAR_MOMENTUM: usize = 24;
/// FEFF Hubbard phase arrays always reserve up/down spin slots.
const XSPH_HUBBARD_SPIN_COUNT: usize = 2;
const XSPH_LOUCKS_GRID_ORIGIN: f64 = 8.8;
const XSPH_SCREENED_CORE_HOLE_SELECTOR: i32 = 2;
const FEFF_WFIRDC_SPEED_OF_LIGHT: f64 = 137.0373;
const XSPH_PHISCF_WFIRDC_COEFFICIENT_COUNT: usize = 3;
const XSPH_BOUND_ORBITAL_COMPONENT_THRESHOLD: f64 = 1.0e-11;
const XSPH_ORBITAL_CORE_COUNT_TOLERANCE: f64 = 1.0e-8;
const XSPH_TDLDA_DEFAULT_PLUS_BASIS_COUNT: i32 = 3;
const XSPH_TDLDA_DEFAULT_MINUS_BASIS_COUNT: i32 = 0;
const XSPH_TDLDA_PRIMARY_CHANNEL_LIMIT: usize = 15;
const XSPH_FINE_STRUCTURE_ALPHA: f64 = 1.0 / 137.035_989_56;
pub(crate) const XSPH_SOURCE_REQUIREMENT_ERROR: &str =
    "XSPH phase generation requires cached phase.bin or supported pot/config source handoffs";
const XSPH_BPHL_REQUIRED_ERROR: &str =
    "XSPH broadened Hedin-Lundqvist exchange requires author-supplied bphl.dat";

fn load_xsph_broadened_table(
    work_dir: &Path,
    exchange_selector: i32,
) -> Result<Option<BroadenedHedinLundqvistTable>> {
    let exchange_branch = exchange_selector % 10;
    let broadened_branch = exchange_selector / 10;
    if broadened_branch != 1 || (exchange_branch != 0 && exchange_branch < 5) {
        return Ok(None);
    }

    let path = work_dir.join("bphl.dat");
    read_bphl_dat(&path)
        .with_context(|| {
            format!(
                "{XSPH_BPHL_REQUIRED_ERROR} for exchange selector {exchange_selector}: {}",
                path.display()
            )
        })
        .map(Some)
}

fn evaluate_xsph_xcpot(
    input: XcpotInput<'_>,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> std::result::Result<XcpotResult, ExchangeError> {
    if let Some(table) = broadened_table {
        xcpot_with_broadened_table(input, table)
    } else {
        xcpot(input)
    }
}

/// Run the supported FEFF XSPH cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_supported_xsph_output(work_dir)? || has_supported_tdlda_xsedge_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_phase_handoff(work_dir)? {
        return run_supported_phase_handoff_in_dir(work_dir);
    }
    if has_supported_phase_text_handoff(work_dir)? {
        return run_supported_phase_text_handoff_in_dir(work_dir);
    }
    if has_supported_phase_mesh_handoff(work_dir)? {
        return run_supported_phase_mesh_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Run XSPH as a required full-run stage.
///
/// Module-level compatibility allows phase-only caches to be refreshed, but a
/// full FEFF run needs the complete ordinary base `phase.bin`/`xsect.dat` pair,
/// including the NRIXS `xsectjas` sidecars when that branch is selected, or the
/// TDLDA `phase.bin`/`xsedge.dat` pair before later stages are allowed to
/// consume XSPH state.
pub(crate) fn run_required_in_dir(work_dir: &Path) -> Result<usize> {
    let written = run_in_dir(work_dir)?;
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) {
        return Ok(written);
    }
    if !has_supported_xsph_output(work_dir)? && !has_supported_tdlda_xsedge_output(work_dir)? {
        bail!(
            "XSPH required stage needs complete phase.bin/xsect.dat caches, complete NRIXS xsecl.dat/xsecl2.dat/xsecl.bin caches when xsectjas is selected, a supported xsect.dat source handoff, or a supported TDLDA xsedge.dat source handoff"
        );
    }
    Ok(written)
}

/// Whether a FEFF XSPH run has the complete base phase/cross-section caches.
#[cfg(test)]
pub(crate) fn has_cached_xsph_output(work_dir: &Path) -> Result<bool> {
    let caches = XsphCachePaths::new(work_dir);
    if !caches.has_complete_base_outputs() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) {
        return Ok(false);
    }
    if tdlda_xsectd_branch_requested(&input) {
        return Ok(false);
    }
    Ok(can_use_cached_xsph_output(&caches, &input))
}

/// Whether a FEFF XSPH run can be satisfied by cached files or supported Rust
/// source handoffs that generate the complete base `phase.bin`/`xsect.dat` pair.
pub(crate) fn has_supported_xsph_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) {
        return Ok(false);
    }
    if tdlda_xsectd_branch_requested(&input) {
        return Ok(false);
    }

    let caches = XsphCachePaths::new(work_dir);
    Ok(can_use_or_generate_base_outputs(&caches, &input)?
        && has_supported_print_rl_output(&caches, &input)?)
}

/// Whether a TDLDA/PMBSE XSPH run can be satisfied by cached files or supported
/// Rust source handoffs that generate the FEFF `phase.bin`/`xsedge.dat` pair.
pub(crate) fn has_supported_tdlda_xsedge_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) || !tdlda_xsectd_branch_requested(&input) {
        return Ok(false);
    }

    let caches = XsphCachePaths::new(work_dir);
    Ok(can_use_or_generate_tdlda_xsedge_outputs(&caches, &input)?
        && has_supported_print_rl_output(&caches, &input)?)
}

/// Whether partial XSPH source state can generate `phase.bin` and phase-derived
/// sidecars without completing the base `xsect.dat` output.
pub(crate) fn has_supported_phase_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(false);
    }
    if has_supported_xsph_output(work_dir)? || has_supported_tdlda_xsedge_output(work_dir)? {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) {
        return Ok(false);
    }

    let caches = XsphCachePaths::new(work_dir);
    if has_readable_phase_cache(&caches) {
        return Ok(false);
    }
    can_generate_source_phase_handoff_for_discovery(&caches, &input)
}

/// Generate only the Rust-backed XSPH phase handoff and sidecars.
///
/// This intentionally does not enter unsupported cross-section branches, so
/// callers can use source-backed `phase.bin` generation in full-run
/// orchestration even when `xsect.dat` cannot yet be generated.
pub(crate) fn run_supported_phase_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) {
        return Ok(0);
    }

    let caches = XsphCachePaths::new(work_dir);
    if has_readable_phase_cache(&caches)
        || has_supported_xsph_output(work_dir)?
        || has_supported_tdlda_xsedge_output(work_dir)?
    {
        return Ok(0);
    }
    let Some(generated) = generate_source_phase_handoff(&caches, &input)? else {
        return Ok(0);
    };
    write_generated_phase_handoff_outputs(&caches, &input, generated)
}

/// Whether partial XSPH source state can generate FEFF phase text sidecars
/// from an existing `phase.bin` without completing the base `xsect.dat` output.
pub(crate) fn has_supported_phase_text_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(false);
    }

    if has_supported_xsph_output(work_dir)? || has_supported_tdlda_xsedge_output(work_dir)? {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) || input.control.ipr2 < 2 {
        return Ok(false);
    }

    let caches = XsphCachePaths::new(work_dir);
    if !caches.has_phase_cache() {
        return Ok(false);
    }
    let Ok(phase) = read_phase_bin(&caches.phase_bin) else {
        return Ok(false);
    };
    Ok(phase_text_sidecar_rewrite_count(&caches, &phase)? > 0
        && prepare_phase_text_sidecars(&input, &phase).is_ok())
}

/// Generate only the Rust-backed XSPH phase text sidecars from `phase.bin`.
///
/// This keeps `PRINT 2` phase diagnostics available in full-run partial-cache
/// paths even when `xsect.dat` cannot yet be generated.
pub(crate) fn run_supported_phase_text_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) || input.control.ipr2 < 2 {
        return Ok(0);
    }

    let caches = XsphCachePaths::new(work_dir);
    if !caches.has_phase_cache()
        || has_supported_xsph_output(work_dir)?
        || has_supported_tdlda_xsedge_output(work_dir)?
    {
        return Ok(0);
    }
    let phase = read_phase_bin(&caches.phase_bin)
        .with_context(|| format!("failed to read {}", caches.phase_bin.display()))?;
    write_stale_or_missing_phase_text_sidecars(&caches, &input, &phase)
}

/// Whether partial XSPH source state can generate the Rust-backed
/// `emesh.dat`/`emesh.bin` handoff without completing the XSPH base stage.
pub(crate) fn has_supported_phase_mesh_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(false);
    }

    if has_supported_xsph_output(work_dir)? || has_supported_tdlda_xsedge_output(work_dir)? {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !xsph_enabled(&input) {
        return Ok(false);
    }

    let caches = XsphCachePaths::new(work_dir);
    if !emesh_sidecars_need_generation(&caches) {
        return Ok(false);
    }
    if caches.has_phase_cache() {
        return match read_phase_bin(&caches.phase_bin) {
            Ok(phase) => Ok(prepare_emesh_sidecars(&caches, &input, &phase).is_ok()),
            Err(_) => can_generate_initial_phase_mesh_handoff(&caches, &input),
        };
    }
    can_generate_initial_phase_mesh_handoff(&caches, &input)
}

/// Generate only the Rust-backed XSPH energy-mesh sidecars.
///
/// This intentionally stays on the source-backed mesh path without requiring
/// complete phase/cross-section outputs, and is used by full-run orchestration
/// for partial XSPH caches.
pub(crate) fn run_supported_phase_mesh_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) {
        return Ok(0);
    }

    let caches = XsphCachePaths::new(work_dir);
    if caches.has_phase_cache() {
        return match read_phase_bin(&caches.phase_bin)
            .with_context(|| format!("failed to read {}", caches.phase_bin.display()))
        {
            Ok(phase) => write_or_generate_emesh_sidecars(&caches, &input, &phase),
            Err(error) => {
                let written = write_initial_phase_mesh_sidecars(&caches, &input)?;
                if written > 0 { Ok(written) } else { Err(error) }
            }
        };
    }
    write_initial_phase_mesh_sidecars(&caches, &input)
}

/// Run the FEFF XSPH cached/source-backed output path.
///
/// This keeps cached FEFF phase directories usable by validating and
/// re-rendering typed `phase.bin`, and uses supported Rust source handoffs
/// when `pot.bin`/`config.dat` carry enough phase and cross-section state.
/// When present, it also preserves `xsect.dat`, optional NRIXS
/// `xsecl.dat`/`xsecl2.dat`/`xsecl.bin`, AXAFS `axafs.dat`, MPSE `mpse.dat`,
/// `aphase_hubbard.bin`, phase-mesh `emesh.dat`/`emesh.bin`, and `log2.dat`
/// diagnostic handoffs. Missing phase-mesh sidecars are regenerated directly
/// from `phase.bin`.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !xsph_enabled(&input) {
        return Ok(0);
    }

    let caches = XsphCachePaths::new(work_dir);
    let generated_phase = if !caches.has_phase_cache() {
        if let Some(phase) = generate_empty_cell_phase_bin(&caches, &input)? {
            Some(GeneratedPhase {
                phase,
                radial: None,
                aphase_hubbard: None,
            })
        } else {
            match generate_normal_potential_phase_bin(&caches, &input) {
                Ok(Some(generated)) => Some(generated),
                Ok(None) => {
                    write_initial_phase_mesh_sidecars(&caches, &input)?;
                    bail!(XSPH_SOURCE_REQUIREMENT_ERROR);
                }
                Err(error) if is_incomplete_source_phase_generation(&error) => {
                    write_initial_phase_mesh_sidecars(&caches, &input)?;
                    bail!(XSPH_SOURCE_REQUIREMENT_ERROR);
                }
                Err(error) => return Err(error),
            }
        }
    } else {
        None
    };
    let (mut phase, mut generated_radial, generated_aphase_hubbard, phase_was_generated) =
        if let Some(generated) = generated_phase {
            (
                generated.phase,
                generated.radial,
                generated.aphase_hubbard,
                true,
            )
        } else if caches.has_phase_cache() {
            match read_phase_bin(&caches.phase_bin)
                .with_context(|| format!("failed to read {}", caches.phase_bin.display()))
            {
                Ok(phase) => {
                    if let Some(generated) =
                        generate_phase_if_stale_against_source(&caches, &input, &phase)?
                    {
                        (
                            generated.phase,
                            generated.radial,
                            generated.aphase_hubbard,
                            true,
                        )
                    } else {
                        (phase, None, None, false)
                    }
                }
                Err(error) => {
                    if let Some(generated) = generate_source_phase_handoff(&caches, &input)? {
                        (
                            generated.phase,
                            generated.radial,
                            generated.aphase_hubbard,
                            true,
                        )
                    } else {
                        return Err(error);
                    }
                }
            }
        } else {
            bail!(XSPH_SOURCE_REQUIREMENT_ERROR)
        };
    write_phase_cache(&caches.phase_bin, &phase)?;
    let mut written = 1_usize;
    let mut rl_was_generated = false;
    if input.print_rl && generated_radial.is_none() {
        if rl_dat_needs_generation(&caches.rl_dat) {
            generated_radial =
                generate_missing_rl_dat_from_normal_potential_handoff(&caches, &input)?;
            rl_was_generated = generated_radial.is_some();
        } else if let Some(generated) = generate_rl_if_stale_against_source(&caches, &input)? {
            generated_radial = Some(generated);
            rl_was_generated = true;
        }
    }
    written += write_or_preserve_rl_dat(&caches, &input, generated_radial.as_ref())?;

    let (mut xsect, xsect_was_generated) = if tdlda_xsectd_branch_requested(&input) {
        (None, false)
    } else if caches.xsect_dat.is_file() {
        if phase_was_generated {
            if let Some(generated) = generate_normal_potential_xsect_dat(&caches, &input, &phase)? {
                phase.transition_moments = generated.transition_moments;
                write_phase_cache(&caches.phase_bin, &phase)?;
                write_xsect_cache(&caches.xsect_dat, &generated.xsect)?;
                (Some(generated.xsect), true)
            } else {
                let data = read_matching_xsect_cache(&caches.xsect_dat, &phase)?;
                write_xsect_cache(&caches.xsect_dat, &data)?;
                (Some(data), false)
            }
        } else {
            match read_matching_xsect_cache(&caches.xsect_dat, &phase) {
                Ok(data) => {
                    if let Some(generated) =
                        generate_xsect_if_stale_against_source(&caches, &input, &phase, &data)?
                    {
                        phase.transition_moments = generated.transition_moments;
                        write_phase_cache(&caches.phase_bin, &phase)?;
                        write_xsect_cache(&caches.xsect_dat, &generated.xsect)?;
                        (Some(generated.xsect), true)
                    } else {
                        write_xsect_cache(&caches.xsect_dat, &data)?;
                        (Some(data), false)
                    }
                }
                Err(error) => {
                    if let Some(generated) =
                        generate_normal_potential_xsect_dat(&caches, &input, &phase)?
                    {
                        phase.transition_moments = generated.transition_moments;
                        write_phase_cache(&caches.phase_bin, &phase)?;
                        write_xsect_cache(&caches.xsect_dat, &generated.xsect)?;
                        (Some(generated.xsect), true)
                    } else {
                        return Err(error);
                    }
                }
            }
        }
    } else if let Some(generated) = generate_normal_potential_xsect_dat(&caches, &input, &phase)? {
        phase.transition_moments = generated.transition_moments;
        write_phase_cache(&caches.phase_bin, &phase)?;
        write_xsect_cache(&caches.xsect_dat, &generated.xsect)?;
        (Some(generated.xsect), true)
    } else {
        (None, false)
    };
    let (nrixs_count, nrixs_was_generated, nrixs_xsect, nrixs_transition_moments) =
        write_or_generate_nrixs_spectrum_sidecars(&caches, &input, &phase)?;
    written += nrixs_count;
    if let Some(generated_xsect) = nrixs_xsect {
        xsect = Some(generated_xsect);
    }
    if let Some(transition_moments) = nrixs_transition_moments {
        phase.transition_moments = transition_moments;
        write_phase_cache(&caches.phase_bin, &phase)?;
    }

    let mut source_handoff_written =
        phase_was_generated || xsect_was_generated || rl_was_generated || nrixs_was_generated;

    written += usize::from(xsect.is_some());
    let (tdlda_xsedge_count, tdlda_xsedge_was_generated) =
        write_or_generate_tdlda_xsedge_cache(&caches, &input, &phase)?;
    written += tdlda_xsedge_count;
    if tdlda_xsedge_was_generated {
        source_handoff_written = true;
    }
    written += write_stale_or_missing_phase_text_sidecars(&caches, &input, &phase)?;
    let (axafs_count, axafs_was_generated) =
        write_or_generate_axafs_cache(&caches, &input, &phase, xsect.as_ref())?;
    written += axafs_count;
    if axafs_was_generated {
        source_handoff_written = true;
    }
    let (aphase_count, aphase_was_generated) =
        write_or_generate_aphase_hubbard_cache(&caches, &input, &phase, generated_aphase_hubbard)?;
    written += aphase_count;
    if aphase_was_generated {
        source_handoff_written = true;
    }
    let (exc_count, exc_was_generated) =
        write_or_generate_xsph_excitation_poles_cache(&caches, &input)?;
    written += exc_count;
    if exc_was_generated {
        source_handoff_written = true;
    }
    let (mpse_count, mpse_was_generated) = write_or_generate_mpse_cache(&caches, &input, &phase)?;
    written += mpse_count;
    if mpse_was_generated {
        source_handoff_written = true;
    }
    written += write_or_generate_emesh_sidecars(&caches, &input, &phase)?;
    written += write_or_recover_module_log(&caches.log2_dat, &phase, source_handoff_written)?;

    Ok(written)
}

/// Generate the active-Hubbard phase table on FEFF LDOS's dedicated energy
/// grid. Unlike the ordinary XSPH mesh, this grid is owned by `ldos.inp`.
pub(crate) fn write_hubbard_phase_on_ldos_grid(
    work_dir: &Path,
    energies: Array1<Complex64>,
) -> Result<usize> {
    ensure!(!energies.is_empty(), "Hubbard LDOS phase grid is empty");
    let input = read_input(work_dir)?;
    let caches = XsphCachePaths::new(work_dir);
    ensure!(
        active_hubbard_phase_requested(&caches)?,
        "Hubbard LDOS phase generation requires active hubbard.inp"
    );
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    let edge = pot.scalars.fermi_level - input.vr0 / FEFF_HARTREE_EV;
    let zero_index = energies
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (left.re - edge).abs().total_cmp(&(right.re - edge).abs()))
        .map_or(0, |(index, _)| index);
    let energy_count = energies.len();
    let mesh = InitialPhaseMesh {
        edge,
        energies,
        horizontal_count: energy_count,
        auxiliary_count: 0,
        fermi_index_1based: zero_index + 1,
        zero_index,
    };
    let generated = generate_normal_potential_phase_bin_with_mesh_and_spin_selectors(
        &caches,
        &input,
        Some(mesh),
        Some(vec![-1, 1]),
    )?
    .context("active Hubbard XSPH source handoffs could not generate the LDOS-grid phase")?;
    let aphase = generated
        .aphase_hubbard
        .context("active Hubbard LDOS-grid phase generation returned no magnetic phases")?;
    write_phase_cache(&caches.phase_bin, &generated.phase)?;
    write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &aphase)
        .with_context(|| format!("failed to write {}", caches.aphase_hubbard_bin.display()))?;
    Ok(2)
}

fn xsph_enabled(input: &XsphInput) -> bool {
    input.control.mphase != 0
}

fn should_generate_axafs(input: &XsphInput) -> bool {
    input.control.ipr2 >= 1
}

fn should_generate_mpse(input: &XsphInput, phase: &PhaseBinData) -> bool {
    input.control.lreal == 0
        && phase.spin_count == 1
        && phase
            .potentials
            .iter()
            .any(|potential| potential.atomic_number != 0)
}

fn has_readable_phase_cache(caches: &XsphCachePaths) -> bool {
    caches.phase_bin.is_file() && read_phase_bin(&caches.phase_bin).is_ok()
}

fn xsph_phase_radial_grid_count(pot: &PotBinData) -> usize {
    pot.total_potential
        .nrows()
        .max(XSPH_PHASE_RADIAL_GRID_CAPACITY)
}

#[derive(Debug, Clone, PartialEq)]
struct XsphInitialSpinorGrid {
    norman_index_1based: usize,
    active_len: usize,
    large: Array1<f64>,
    small: Array1<f64>,
}

fn xsph_initial_spinor_grid(
    pot: &PotBinData,
    prepared: &XsphPhaseGridPreparation,
    potential_index: usize,
    context_label: &str,
) -> Result<XsphInitialSpinorGrid> {
    ensure!(
        potential_index < pot.norman_indices.len()
            && potential_index < pot.norman_radii.len()
            && potential_index < pot.potential_count(),
        "XSPH {context_label} core-hole spinor request for potential {potential_index} exceeds pot.bin potential count {}",
        pot.potential_count()
    );
    let norman_index_1based =
        xsph_xsect_norman_index_1based(pot, potential_index, prepared.radial_dx)?;
    ensure!(
        norman_index_1based <= prepared.radii.len(),
        "XSPH {context_label} Norman index {norman_index_1based} exceeds prepared radial grid length {}",
        prepared.radii.len()
    );
    let spinor = fix_dirac_spinor_grid(DiracSpinorGridInput {
        original_delta: LOUCKS_DELTA,
        new_delta: prepared.radial_dx,
        large_component: pot.initial_large_component.view(),
        small_component: pot.initial_small_component.view(),
        output_len: prepared.radii.len(),
    })
    .with_context(|| {
        format!("failed to interpolate XSPH {context_label} core-hole spinor onto work grid")
    })?;
    Ok(XsphInitialSpinorGrid {
        norman_index_1based,
        active_len: spinor.active_len,
        large: spinor.large_component,
        small: spinor.small_component,
    })
}

fn xsph_normalized_initial_spinor_grid(
    pot: &PotBinData,
    prepared: &XsphPhaseGridPreparation,
    potential_index: usize,
    initial_l: usize,
    context_label: &str,
) -> Result<XsphInitialSpinorGrid> {
    let spinor = xsph_initial_spinor_grid(pot, prepared, potential_index, context_label)?;
    let hole_normalization = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l,
        log_step: prepared.radial_dx,
        radii: prepared.radii.view(),
        initial_large: spinor.large.view(),
        initial_small: spinor.small.view(),
        norman_index_1based: spinor.norman_index_1based,
    })
    .with_context(|| format!("failed to normalize XSPH {context_label} core-hole spinor"))?;
    ensure!(
        hole_normalization.normalization.is_finite() && hole_normalization.normalization > 0.0,
        "XSPH {context_label} core-hole normalization must be positive, got {}",
        hole_normalization.normalization
    );
    let hole_scale = hole_normalization.normalization.sqrt();
    Ok(XsphInitialSpinorGrid {
        norman_index_1based: spinor.norman_index_1based,
        active_len: spinor.active_len,
        large: spinor.large.mapv(|value| value / hole_scale),
        small: spinor.small.mapv(|value| value / hole_scale),
    })
}

fn xsph_xsect_norman_index_1based(
    pot: &PotBinData,
    potential_index: usize,
    target_log_step: f64,
) -> Result<usize> {
    let source_index = pot.norman_indices[potential_index];
    ensure!(
        source_index > 0,
        "pot.bin Norman index must be one-based and positive"
    );
    if (target_log_step - LOUCKS_DELTA).abs() <= 1.0e-12 {
        return Ok(source_index);
    }

    let norman_radius = pot.norman_radii[potential_index];
    ensure!(
        norman_radius.is_finite() && norman_radius > 0.0,
        "pot.bin Norman radius must be positive for XSPH target-grid normalization, got {norman_radius}"
    );
    ensure!(
        target_log_step.is_finite() && target_log_step > 0.0,
        "XSPH target radial log step must be positive, got {target_log_step}"
    );
    let target_index =
        ((norman_radius.ln() + XSPH_LOUCKS_GRID_ORIGIN) / target_log_step + 1.0).trunc();
    ensure!(
        target_index.is_finite() && target_index >= 1.0,
        "XSPH target-grid Norman index must be positive, got {target_index}"
    );
    Ok(target_index as usize)
}

fn can_use_cached_xsph_output(caches: &XsphCachePaths, input: &XsphInput) -> bool {
    prepare_cached_xsph_output(caches, input).is_ok()
}

fn can_use_or_generate_base_outputs(caches: &XsphCachePaths, input: &XsphInput) -> Result<bool> {
    if tdlda_xsectd_branch_requested(input) {
        return Ok(false);
    }

    if caches.has_complete_base_outputs() && can_use_cached_xsph_output(caches, input) {
        return Ok(true);
    }

    if caches.has_complete_base_outputs() && can_use_or_repair_cached_base_outputs(caches, input)? {
        return Ok(true);
    }

    if caches.has_phase_cache()
        && let Ok(phase) = read_phase_bin(&caches.phase_bin)
        && (can_generate_normal_potential_xsect_from_phase_cache(caches, input)?
            || generate_nrixs_xsectjas_sidecars(caches, input, &phase)?.is_some())
    {
        return Ok(true);
    }

    if can_generate_normal_potential_base_outputs(caches, input)? {
        return Ok(true);
    }

    if caches.xsect_dat.is_file() {
        let Ok(xsect) = read_xsect_dat(&caches.xsect_dat) else {
            return Ok(false);
        };
        return can_generate_phase_matching_xsect_cache(caches, input, &xsect);
    }

    Ok(false)
}

fn can_use_or_repair_cached_base_outputs(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    let phase = match read_phase_bin(&caches.phase_bin) {
        Ok(phase) => phase,
        Err(_) => return Ok(false),
    };
    let xsect = match read_xsect_dat(&caches.xsect_dat) {
        Ok(xsect) => xsect,
        Err(_) => return Ok(false),
    };
    if ensure_phase_matches_source_if_available(caches, input, &phase).is_err()
        || ensure_xsect_matches_phase(&phase, &xsect).is_err()
        || ensure_xsect_matches_source_if_available(caches, input, &phase, &xsect).is_err()
    {
        return Ok(false);
    }

    let rl_source_handoff = can_repair_print_rl_cache_from_normal_potential_handoff(caches, input)?;
    if !rl_source_handoff && prepare_print_rl_cache(caches, input).is_err() {
        return Ok(false);
    }
    if prepare_phase_text_sidecars(input, &phase).is_err()
        || !can_use_or_repair_axafs_cache(caches, input, &phase, &xsect)?
        || !can_use_or_repair_optional_spectrum_sidecars(caches, input, &phase)?
        || prepare_emesh_sidecars(caches, input, &phase).is_err()
    {
        return Ok(false);
    }
    if !rl_source_handoff && prepare_module_log_cache(caches).is_err() {
        return Ok(false);
    }
    Ok(true)
}

fn prepare_cached_xsph_output(caches: &XsphCachePaths, input: &XsphInput) -> Result<()> {
    let phase = read_phase_bin(&caches.phase_bin)
        .with_context(|| format!("failed to read {}", caches.phase_bin.display()))?;
    let xsect = read_xsect_dat(&caches.xsect_dat)
        .with_context(|| format!("failed to read {}", caches.xsect_dat.display()))?;
    ensure_phase_matches_source_if_available(caches, input, &phase)?;
    ensure_xsect_matches_phase(&phase, &xsect)?;
    ensure_xsect_matches_source_if_available(caches, input, &phase, &xsect)?;
    let rl_source_handoff =
        can_generate_missing_rl_dat_from_normal_potential_handoff(caches, input)?;

    if !rl_source_handoff {
        prepare_print_rl_cache(caches, input)?;
    }
    prepare_phase_text_sidecars(input, &phase)?;
    prepare_axafs_cache(caches, input, &phase, &xsect)?;
    prepare_optional_spectrum_sidecars(caches, input, &phase)?;
    prepare_emesh_sidecars(caches, input, &phase)?;
    if !rl_source_handoff {
        prepare_module_log_cache(caches)?;
    }
    Ok(())
}

fn prepare_print_rl_cache(caches: &XsphCachePaths, input: &XsphInput) -> Result<()> {
    if input.print_rl && caches.rl_dat.is_file() {
        let data = read_xsph_rl_dat(&caches.rl_dat)
            .with_context(|| format!("failed to read {}", caches.rl_dat.display()))?;
        ensure_rl_matches_source_if_available(caches, input, &data)?;
    }
    Ok(())
}

fn prepare_phase_text_sidecars(input: &XsphInput, phase: &PhaseBinData) -> Result<()> {
    if input.control.ipr2 < 2 {
        return Ok(());
    }
    for potential_index in 0..phase.potential_count() {
        phase_text_dat_string(phase, potential_index)?;
        phmin_text_dat_string(phase, potential_index)?;
    }
    Ok(())
}

fn prepare_axafs_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<()> {
    if caches.axafs_dat.is_file() {
        match read_axafs_dat(&caches.axafs_dat) {
            Ok(data) => {
                if should_generate_axafs(input) {
                    ensure_axafs_matches_source_if_available(
                        &caches.axafs_dat,
                        &data,
                        phase,
                        xsect,
                    )?;
                }
                return Ok(());
            }
            Err(error) => {
                if should_generate_axafs(input) {
                    recover_axafs_dat_from_handoffs(&caches.axafs_dat, phase, xsect)?;
                    return Ok(());
                }
                return Err(error)
                    .with_context(|| format!("failed to read {}", caches.axafs_dat.display()));
            }
        }
    } else if should_generate_axafs(input) {
        generate_axafs_dat(phase, xsect)?;
    }
    Ok(())
}

fn read_matching_xsect_cache(path: &Path, phase: &PhaseBinData) -> Result<XsectDatData> {
    let data =
        read_xsect_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    ensure_xsect_matches_phase(phase, &data)
        .with_context(|| format!("failed to validate {} against phase.bin", path.display()))?;
    Ok(data)
}

fn ensure_xsect_matches_phase(phase: &PhaseBinData, xsect: &XsectDatData) -> Result<()> {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-5;

    ensure!(
        xsect.energy_count() == phase.energy_count,
        "xsect.dat energy count {} does not match phase.bin energy count {}",
        xsect.energy_count(),
        phase.energy_count
    );
    ensure!(
        xsect.main_energy_count == phase.main_energy_count,
        "xsect.dat main energy count {} does not match phase.bin main energy count {}",
        xsect.main_energy_count,
        phase.main_energy_count
    );
    let phase_fermi_index =
        usize::try_from(phase.fermi_index).context("phase.bin fermi index is negative")?;
    ensure!(
        xsect.fermi_index == phase_fermi_index,
        "xsect.dat fermi index {} does not match phase.bin fermi index {}",
        xsect.fermi_index,
        phase.fermi_index
    );
    let expected_energy = phase.energy_grid.mapv(|energy| energy * FEFF_HARTREE_EV);
    ensure!(
        complex_slices_match(
            &xsect.energy_grid_ev,
            &expected_energy,
            ENERGY_TOLERANCE_EV,
            ENERGY_TOLERANCE_EV,
        ),
        "xsect.dat energy grid does not match phase.bin energy grid"
    );
    Ok(())
}

fn generate_phase_if_stale_against_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<GeneratedPhase>> {
    let generated = match generate_source_phase_handoff(caches, input) {
        Ok(Some(generated)) => generated,
        Ok(None) | Err(_) => return Ok(None),
    };
    if phase_matches_source_phase(phase, &generated.phase) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn ensure_phase_matches_source_if_available(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    ensure!(
        generate_phase_if_stale_against_source(caches, input, phase)?.is_none(),
        "cached phase.bin is stale against XSPH source handoffs"
    );
    Ok(())
}

fn ensure_xsect_matches_source_if_available(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<()> {
    ensure!(
        generate_xsect_if_stale_against_source(caches, input, phase, xsect)?.is_none(),
        "cached xsect.dat is stale against XSPH source handoffs"
    );
    Ok(())
}

fn generate_xsect_if_stale_against_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<Option<GeneratedNormalXsect>> {
    let generated = match generate_normal_potential_xsect_dat(caches, input, phase) {
        Ok(Some(generated)) => generated,
        Ok(None) | Err(_) => return Ok(None),
    };
    if xsect_matches_source_xsect(xsect, &generated.xsect)
        && phase_transition_moments_match_source_xsect(phase, &generated)
    {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn xsect_matches_source_xsect(cached: &XsectDatData, source: &XsectDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-7;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    cached.main_energy_count == source.main_energy_count
        && cached.fermi_index == source.fermi_index
        && cached.energy_count() == source.energy_count()
        && real_slices_match(
            &cached.normalized_background,
            &source.normalized_background,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && complex_slices_match(
            &cached.energy_grid_ev,
            &source.energy_grid_ev,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && complex_slices_match(
            &cached.cross_section,
            &source.cross_section,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
}

fn ensure_rl_matches_source_if_available(
    caches: &XsphCachePaths,
    input: &XsphInput,
    cached: &XsphRlDatData,
) -> Result<()> {
    let generated = match generate_rl_dat_from_normal_potential_handoff(caches, input) {
        Ok(Some(generated)) => generated,
        Ok(None) | Err(_) => return Ok(()),
    };
    ensure!(
        rl_dat_matches_source_rl(cached, &generated),
        "cached {} is stale against XSPH normal-potential rl.dat handoff",
        caches.rl_dat.display()
    );
    Ok(())
}

fn generate_rl_if_stale_against_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<XsphRlDatData>> {
    let cached = match read_xsph_rl_dat(&caches.rl_dat) {
        Ok(cached) => cached,
        Err(_) => return Ok(None),
    };
    let generated = match generate_rl_dat_from_normal_potential_handoff(caches, input) {
        Ok(Some(generated)) => generated,
        Ok(None) | Err(_) => return Ok(None),
    };
    if rl_dat_matches_source_rl(&cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn rl_dat_matches_source_rl(cached: &XsphRlDatData, source: &XsphRlDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-7;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    cached.angular_limit == source.angular_limit
        && cached.radial_match_index_1based == source.radial_match_index_1based
        && scalar_matches(
            cached.muffin_tin_radius,
            source.muffin_tin_radius,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && scalar_matches(
            cached.log_step,
            source.log_step,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && scalar_matches(
            cached.grid_origin,
            source.grid_origin,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && cached.records.len() == source.records.len()
        && cached
            .records
            .iter()
            .zip(source.records.iter())
            .all(|(cached, source)| {
                cached.angular_momentum == source.angular_momentum
                    && complex_scalar_matches(
                        cached.energy,
                        source.energy,
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE,
                    )
                    && complex_scalar_matches(
                        cached.phase_shift,
                        source.phase_shift,
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE,
                    )
                    && complex_slices_match(
                        &cached.regular_large,
                        &source.regular_large,
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE,
                    )
                    && complex_slices_match(
                        &cached.regular_small,
                        &source.regular_small,
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE,
                    )
            })
}

fn phase_matches_source_phase(cached: &PhaseBinData, source: &PhaseBinData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-7;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    cached.spin_count == source.spin_count
        && cached.energy_count == source.energy_count
        && cached.main_energy_count == source.main_energy_count
        && cached.auxiliary_energy_count == source.auxiliary_energy_count
        && cached.ihole == source.ihole
        && cached.fermi_index == source.fermi_index
        && cached.pad_width == source.pad_width
        && cached.final_state_count == source.final_state_count
        && cached.transition_count == source.transition_count
        && cached.q_count == source.q_count
        && phase_scalars_match(
            cached.scalars,
            source.scalars,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && complex_slices_match(
            &cached.energy_grid,
            &source.energy_grid,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && complex_arrays2_match(
            &cached.reference_energy,
            &source.reference_energy,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
        && cached.potentials.len() == source.potentials.len()
        && cached
            .potentials
            .iter()
            .zip(source.potentials.iter())
            .all(|(cached, source)| {
                phase_potential_matches_source(
                    cached,
                    source,
                    ABSOLUTE_TOLERANCE,
                    RELATIVE_TOLERANCE,
                )
            })
}

fn phase_scalars_match(
    cached: PhaseBinScalars,
    source: PhaseBinScalars,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached
        .as_array()
        .iter()
        .zip(source.as_array().iter())
        .all(|(cached, source)| {
            scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn phase_potential_matches_source(
    cached: &PhaseBinPotential,
    source: &PhaseBinPotential,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.lmax == source.lmax
        && cached.atomic_number == source.atomic_number
        && cached.label == source.label
        && complex_arrays3_match(
            &cached.phase_shifts,
            &source.phase_shifts,
            absolute_tolerance,
            relative_tolerance,
        )
}

fn phase_transition_moments_match_source_xsect(
    phase: &PhaseBinData,
    source: &GeneratedNormalXsect,
) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-7;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    complex_arrays4_match(
        &phase.transition_moments,
        &source.transition_moments,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
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

fn complex_slices_match(
    cached: &Array1<Complex64>,
    source: &Array1<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.len() == source.len()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            complex_scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_arrays2_match(
    cached: &Array2<Complex64>,
    source: &Array2<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.dim() == source.dim()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            complex_scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_arrays3_match(
    cached: &Array3<Complex64>,
    source: &Array3<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.dim() == source.dim()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            complex_scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_arrays4_match(
    cached: &Array4<Complex64>,
    source: &Array4<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.dim() == source.dim()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            complex_scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_arrays5_match(
    cached: &Array5<Complex64>,
    source: &Array5<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    cached.dim() == source.dim()
        && cached.iter().zip(source.iter()).all(|(cached, source)| {
            complex_scalar_matches(*cached, *source, absolute_tolerance, relative_tolerance)
        })
}

fn complex_scalar_matches(
    cached: Complex64,
    source: Complex64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    scalar_matches(cached.re, source.re, absolute_tolerance, relative_tolerance)
        && scalar_matches(cached.im, source.im, absolute_tolerance, relative_tolerance)
}

fn scalar_matches(
    cached: f64,
    source: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    (cached - source).abs() <= absolute_tolerance + relative_tolerance * source.abs()
}

fn read_matching_xsecl_cache(path: &Path, phase: &PhaseBinData) -> Result<XseclDatData> {
    let data =
        read_xsecl_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    ensure_xsecl_matches_phase(phase, &data)
        .with_context(|| format!("failed to validate {} against phase.bin", path.display()))?;
    Ok(data)
}

fn read_matching_xsecl2_cache(path: &Path, phase: &PhaseBinData) -> Result<XseclDatData> {
    let data =
        read_xsecl2_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    ensure_xsecl_matches_phase(phase, &data)
        .with_context(|| format!("failed to validate {} against phase.bin", path.display()))?;
    Ok(data)
}

fn read_matching_xsecl_bin_cache(path: &Path, phase: &PhaseBinData) -> Result<XseclBinData> {
    let data = read_xsecl_bin(path, phase.pad_width, phase.energy_count)
        .with_context(|| format!("failed to read {}", path.display()))?;
    ensure_xsecl_bin_matches_phase(phase, &data)
        .with_context(|| format!("failed to validate {} against phase.bin", path.display()))?;
    Ok(data)
}

fn ensure_xsecl_matches_phase(phase: &PhaseBinData, xsecl: &XseclDatData) -> Result<()> {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-5;

    ensure!(
        xsecl.row_count() == phase.energy_count,
        "xsecl energy count {} does not match phase.bin energy count {}",
        xsecl.row_count(),
        phase.energy_count
    );
    ensure!(
        xsecl.header.real_energy_count == phase.main_energy_count,
        "xsecl real energy count {} does not match phase.bin main energy count {}",
        xsecl.header.real_energy_count,
        phase.main_energy_count
    );
    let phase_fermi_index =
        usize::try_from(phase.fermi_index).context("phase.bin fermi index is negative")?;
    ensure!(
        xsecl.header.fermi_index == phase_fermi_index,
        "xsecl fermi index {} does not match phase.bin fermi index {}",
        xsecl.header.fermi_index,
        phase.fermi_index
    );
    let expected_energy = xsecl_energy_grid_from_phase(phase, xsecl.header.emu);
    ensure!(
        real_slices_match(&xsecl.energy, &expected_energy, ENERGY_TOLERANCE_EV, 0.0),
        "xsecl energy grid does not match phase.bin shifted energy grid"
    );
    ensure_xsecl_channel_sums_match(xsecl)?;
    Ok(())
}

fn ensure_xsecl_channel_sums_match(xsecl: &XseclDatData) -> Result<()> {
    const SUM_TOLERANCE: f64 = 5.0e-6;

    let expected_sum = xsecl.channel_cross_sections.sum_axis(Axis(1));
    ensure!(
        complex_slices_match(
            &xsecl.channel_sum,
            &expected_sum,
            SUM_TOLERANCE,
            SUM_TOLERANCE
        ),
        "xsecl channel sum does not match channel columns"
    );
    Ok(())
}

fn ensure_xsecl_text_sidecars_match(xsecl: &XseclDatData, xsecl2: &XseclDatData) -> Result<()> {
    const HEADER_TOLERANCE: f64 = 5.0e-10;

    ensure!(
        xsecl.channel_count() == xsecl2.channel_count(),
        "xsecl.dat channel count {} does not match xsecl2.dat channel count {}",
        xsecl.channel_count(),
        xsecl2.channel_count()
    );
    ensure!(
        scalar_matches(
            xsecl.header.edge,
            xsecl2.header.edge,
            HEADER_TOLERANCE,
            HEADER_TOLERANCE
        ),
        "xsecl.dat edge {} does not match xsecl2.dat edge {}",
        xsecl.header.edge,
        xsecl2.header.edge
    );
    ensure!(
        scalar_matches(
            xsecl.header.emu,
            xsecl2.header.emu,
            HEADER_TOLERANCE,
            HEADER_TOLERANCE
        ),
        "xsecl.dat emu {} does not match xsecl2.dat emu {}",
        xsecl.header.emu,
        xsecl2.header.emu
    );
    ensure!(
        scalar_matches(
            xsecl.header.core_hole_width,
            xsecl2.header.core_hole_width,
            HEADER_TOLERANCE,
            HEADER_TOLERANCE
        ),
        "xsecl.dat core-hole width {} does not match xsecl2.dat core-hole width {}",
        xsecl.header.core_hole_width,
        xsecl2.header.core_hole_width
    );
    Ok(())
}

fn ensure_xsecl_bin_matches_phase(phase: &PhaseBinData, xsecl_bin: &XseclBinData) -> Result<()> {
    ensure!(
        xsecl_bin.final_state_count() == phase.final_state_count,
        "xsecl.bin final-state count {} does not match phase.bin final-state count {}",
        xsecl_bin.final_state_count(),
        phase.final_state_count
    );
    ensure!(
        xsecl_bin.transition_index_count() == phase.transition_count,
        "xsecl.bin transition index count {} does not match phase.bin transition count {}",
        xsecl_bin.transition_index_count(),
        phase.transition_count
    );
    Ok(())
}

fn xsecl_energy_grid_from_phase(phase: &PhaseBinData, chemical_potential_ev: f64) -> Array1<f64> {
    phase
        .energy_grid
        .mapv(|energy| energy.re - phase.scalars.edge_energy + chemical_potential_ev)
}

type NrixsSpectrumSidecarWrite = (usize, bool, Option<XsectDatData>, Option<Array4<Complex64>>);

fn write_or_generate_nrixs_spectrum_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<NrixsSpectrumSidecarWrite> {
    if nrixs_xsectjas_requested(caches, input)?
        && let Some(generated) = generate_nrixs_xsectjas_sidecars(caches, input, phase)?
    {
        write_xsect_cache(&caches.xsect_dat, &generated.xsect)?;
        write_xsecl_cache(&caches.xsecl_dat, &generated.handoffs.xsecl)?;
        write_xsecl2_cache(&caches.xsecl2_dat, &generated.handoffs.xsecl2)?;
        write_xsecl_bin_cache(&caches.xsecl_bin, &generated.handoffs.xsecl_bin)?;
        return Ok((
            3,
            true,
            Some(generated.xsect),
            Some(generated.transition_moments),
        ));
    }

    let xsecl_data = if caches.xsecl_dat.is_file() {
        Some(read_matching_xsecl_cache(&caches.xsecl_dat, phase)?)
    } else {
        None
    };
    let xsecl2_data = if caches.xsecl2_dat.is_file() {
        Some(read_matching_xsecl2_cache(&caches.xsecl2_dat, phase)?)
    } else {
        None
    };
    if let (Some(xsecl), Some(xsecl2)) = (&xsecl_data, &xsecl2_data) {
        ensure_xsecl_text_sidecars_match(xsecl, xsecl2)?;
    }

    let mut written = 0_usize;
    if let Some(data) = xsecl_data {
        write_xsecl_cache(&caches.xsecl_dat, &data)?;
        written += 1;
    }
    if let Some(data) = xsecl2_data {
        write_xsecl2_cache(&caches.xsecl2_dat, &data)?;
        written += 1;
    }
    if caches.xsecl_bin.is_file() {
        let data = read_matching_xsecl_bin_cache(&caches.xsecl_bin, phase)?;
        write_xsecl_bin_cache(&caches.xsecl_bin, &data)?;
        written += 1;
    }

    Ok((written, false, None, None))
}

fn write_or_generate_axafs_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    xsect: Option<&XsectDatData>,
) -> Result<(usize, bool)> {
    if caches.axafs_dat.is_file() {
        match read_axafs_dat(&caches.axafs_dat) {
            Ok(data) => {
                if should_generate_axafs(input)
                    && let Some(xsect) = xsect
                    && let Some(generated) =
                        generate_axafs_if_stale_against_source(phase, xsect, &data)?
                {
                    write_axafs_cache(&caches.axafs_dat, &generated)?;
                    return Ok((1, true));
                }
                write_axafs_cache(&caches.axafs_dat, &data)?;
                return Ok((1, false));
            }
            Err(error) => {
                if should_generate_axafs(input)
                    && let Some(xsect) = xsect
                {
                    let data = recover_axafs_dat_from_handoffs(&caches.axafs_dat, phase, xsect)?;
                    write_axafs_cache(&caches.axafs_dat, &data)?;
                    return Ok((1, true));
                }
                return Err(error)
                    .with_context(|| format!("failed to read {}", caches.axafs_dat.display()));
            }
        }
    }

    if should_generate_axafs(input) {
        let Some(xsect) = xsect else {
            bail!("XSPH AXAFS generation requires xsect.dat cross-section handoff");
        };
        if let Some(data) = generate_axafs_dat(phase, xsect)? {
            write_axafs_cache(&caches.axafs_dat, &data)?;
            return Ok((1, true));
        }
    }

    Ok((0, false))
}

fn can_use_or_repair_axafs_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<bool> {
    if caches.axafs_dat.is_file() {
        return match read_axafs_dat(&caches.axafs_dat) {
            Ok(data) => {
                if should_generate_axafs(input) {
                    generate_axafs_if_stale_against_source(phase, xsect, &data)?;
                }
                Ok(true)
            }
            Err(_) if should_generate_axafs(input) => {
                Ok(generate_axafs_dat(phase, xsect)?.is_some())
            }
            Err(_) => Ok(false),
        };
    }

    if should_generate_axafs(input) {
        generate_axafs_dat(phase, xsect)?;
        return Ok(true);
    }

    Ok(true)
}

fn ensure_axafs_matches_source_if_available(
    path: &Path,
    axafs: &AxafsDatData,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<()> {
    ensure!(
        generate_axafs_if_stale_against_source(phase, xsect, axafs)?.is_none(),
        "cached {} is stale against XSPH phase.bin/xsect.dat handoffs",
        path.display()
    );
    Ok(())
}

fn generate_axafs_if_stale_against_source(
    phase: &PhaseBinData,
    xsect: &XsectDatData,
    axafs: &AxafsDatData,
) -> Result<Option<AxafsDatData>> {
    let Some(generated) = generate_axafs_dat(phase, xsect)? else {
        return Ok(None);
    };
    let generated = normalize_axafs_dat(&generated)?;
    let cached = normalize_axafs_dat(axafs)?;
    if axafs_matches_source_axafs(&cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn normalize_axafs_dat(data: &AxafsDatData) -> Result<AxafsDatData> {
    parse_axafs_dat(&axafs_dat_string(data)?).context("failed to normalize axafs.dat table")
}

fn axafs_matches_source_axafs(cached: &AxafsDatData, source: &AxafsDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-3;
    const RELATIVE_TOLERANCE: f64 = 1.0e-4;

    real_slices_match(
        &cached.energy_ev,
        &source.energy_ev,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.edge_relative_energy_ev,
        &source.edge_relative_energy_ev,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.wave_number_inverse_angstrom,
        &source.wave_number_inverse_angstrom,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.atomic_absorption,
        &source.atomic_absorption,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.atomic_background,
        &source.atomic_background,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && real_slices_match(
        &cached.chi_atomic,
        &source.chi_atomic,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
}

fn recover_axafs_dat_from_handoffs(
    path: &Path,
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<AxafsDatData> {
    match generate_axafs_dat(phase, xsect) {
        Ok(Some(data)) => Ok(data),
        Ok(None) => bail!(
            "failed to recover {} from phase.bin and xsect.dat: insufficient AXAFS points",
            path.display()
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to recover {} from phase.bin and xsect.dat",
                path.display()
            )
        }),
    }
}

fn write_or_generate_mpse_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<(usize, bool)> {
    if caches.mpse_dat.is_file() {
        match read_mpse_dat(&caches.mpse_dat) {
            Ok(data) => {
                if should_generate_mpse(input, phase)
                    && let Some(generated) =
                        generate_mpse_if_stale_against_source(caches, input, phase, &data)?
                {
                    write_mpse_cache(&caches.mpse_dat, &generated)?;
                    return Ok((1, true));
                }
                write_mpse_cache(&caches.mpse_dat, &data)?;
                return Ok((1, false));
            }
            Err(error) => {
                if should_generate_mpse(input, phase)
                    && let Some(data) = generate_mpse_dat(caches, input, phase)?
                {
                    write_mpse_cache(&caches.mpse_dat, &data)?;
                    return Ok((1, true));
                }
                return Err(error)
                    .with_context(|| format!("failed to read {}", caches.mpse_dat.display()));
            }
        }
    }

    if should_generate_mpse(input, phase)
        && let Some(data) = generate_mpse_dat(caches, input, phase)?
    {
        write_mpse_cache(&caches.mpse_dat, &data)?;
        return Ok((1, true));
    }

    Ok((0, false))
}

fn can_use_or_repair_optional_spectrum_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    if nrixs_xsectjas_requested(caches, input)? {
        if prepare_required_nrixs_spectrum_sidecars(caches, input, phase).is_err() {
            return Ok(false);
        }
    } else if prepare_available_nrixs_spectrum_sidecars(caches, phase).is_err() {
        return Ok(false);
    }

    if !can_use_or_repair_aphase_hubbard_cache(caches, input, phase)? {
        return Ok(false);
    }

    can_use_or_repair_mpse_cache(caches, input, phase)
}

fn write_or_generate_tdlda_xsedge_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<(usize, bool)> {
    if !tdlda_xsectd_branch_requested(input) {
        return Ok((0, false));
    }

    if let Some(data) = generate_tdlda_pmbse_xsedge_dat(caches, input, phase)? {
        write_xsedge_dat(&caches.xsedge_dat, &data)
            .with_context(|| format!("failed to write {}", caches.xsedge_dat.display()))?;
        return Ok((1, true));
    }

    if caches.xsedge_dat.is_file() {
        let data = read_xsedge_dat(&caches.xsedge_dat)
            .with_context(|| format!("failed to read {}", caches.xsedge_dat.display()))?;
        write_xsedge_dat(&caches.xsedge_dat, &data)
            .with_context(|| format!("failed to write {}", caches.xsedge_dat.display()))?;
        return Ok((1, false));
    }

    Ok((0, false))
}

fn prepare_optional_spectrum_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    if nrixs_xsectjas_requested(caches, input)? {
        prepare_required_nrixs_spectrum_sidecars(caches, input, phase)?;
    } else {
        prepare_available_nrixs_spectrum_sidecars(caches, phase)?;
    }
    prepare_aphase_hubbard_cache(caches, input, phase)?;
    prepare_mpse_cache(caches, input, phase)?;
    Ok(())
}

fn prepare_mpse_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    if !caches.mpse_dat.is_file() {
        return Ok(());
    }

    match read_mpse_dat(&caches.mpse_dat) {
        Ok(data) => {
            if should_generate_mpse(input, phase) {
                ensure_mpse_matches_source_if_available(caches, input, phase, &data)?;
            }
            Ok(())
        }
        Err(error) => {
            if can_generate_mpse_cache(caches, input, phase)? {
                return Ok(());
            }
            Err(error).with_context(|| format!("failed to read {}", caches.mpse_dat.display()))
        }
    }
}

fn can_use_or_repair_mpse_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    if !caches.mpse_dat.is_file() {
        return Ok(true);
    }

    match read_mpse_dat(&caches.mpse_dat) {
        Ok(data) => {
            if should_generate_mpse(input, phase) {
                generate_mpse_if_stale_against_source(caches, input, phase, &data)?;
            }
            Ok(true)
        }
        Err(_) if should_generate_mpse(input, phase) => {
            Ok(generate_mpse_dat(caches, input, phase)?.is_some())
        }
        Err(_) => Ok(false),
    }
}

fn ensure_mpse_matches_source_if_available(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    mpse: &MpseDatData,
) -> Result<()> {
    ensure!(
        generate_mpse_if_stale_against_source(caches, input, phase, mpse)?.is_none(),
        "cached {} is stale against XSPH phase.bin/pot.bin handoffs",
        caches.mpse_dat.display()
    );
    Ok(())
}

fn generate_mpse_if_stale_against_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    mpse: &MpseDatData,
) -> Result<Option<MpseDatData>> {
    let Some(generated) = generate_mpse_dat(caches, input, phase)? else {
        return Ok(None);
    };
    let generated = normalize_mpse_dat(&generated)?;
    let cached = normalize_mpse_dat(mpse)?;
    if mpse_matches_source_mpse(&cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn normalize_mpse_dat(data: &MpseDatData) -> Result<MpseDatData> {
    parse_mpse_dat(&mpse_dat_string(data)?).context("failed to normalize mpse.dat table")
}

fn mpse_matches_source_mpse(cached: &MpseDatData, source: &MpseDatData) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    real_slices_match(
        &cached.energy_ev,
        &source.energy_ev,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && complex_slices_match(
        &cached.self_energy,
        &source.self_energy,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && optional_complex_slices_match(
        &cached.renormalization,
        &source.renormalization,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && optional_real_slices_match(
        &cached.renormalization_magnitude,
        &source.renormalization_magnitude,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && optional_real_slices_match(
        &cached.renormalization_phase,
        &source.renormalization_phase,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    ) && optional_real_slices_match(
        &cached.inelastic_mean_free_path,
        &source.inelastic_mean_free_path,
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE,
    )
}

fn optional_real_slices_match(
    cached: &Option<Array1<f64>>,
    source: &Option<Array1<f64>>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    match (cached, source) {
        (Some(cached), Some(source)) => {
            real_slices_match(cached, source, absolute_tolerance, relative_tolerance)
        }
        (None, None) => true,
        _ => false,
    }
}

fn optional_complex_slices_match(
    cached: &Option<Array1<Complex64>>,
    source: &Option<Array1<Complex64>>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> bool {
    match (cached, source) {
        (Some(cached), Some(source)) => {
            complex_slices_match(cached, source, absolute_tolerance, relative_tolerance)
        }
        (None, None) => true,
        _ => false,
    }
}

fn prepare_required_nrixs_spectrum_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    if generate_nrixs_xsectjas_sidecars(caches, input, phase)?.is_some() {
        return Ok(());
    }
    ensure!(
        caches.xsecl_dat.is_file(),
        "NRIXS XSPH xsectjas output requires xsecl.dat cache"
    );
    ensure!(
        caches.xsecl2_dat.is_file(),
        "NRIXS XSPH xsectjas output requires xsecl2.dat cache"
    );
    ensure!(
        caches.xsecl_bin.is_file(),
        "NRIXS XSPH xsectjas output requires xsecl.bin cache"
    );
    prepare_available_nrixs_spectrum_sidecars(caches, phase)
}

fn prepare_available_nrixs_spectrum_sidecars(
    caches: &XsphCachePaths,
    phase: &PhaseBinData,
) -> Result<()> {
    let xsecl = if caches.xsecl_dat.is_file() {
        Some(read_matching_xsecl_cache(&caches.xsecl_dat, phase)?)
    } else {
        None
    };
    let xsecl2 = if caches.xsecl2_dat.is_file() {
        Some(read_matching_xsecl2_cache(&caches.xsecl2_dat, phase)?)
    } else {
        None
    };
    if let (Some(xsecl), Some(xsecl2)) = (&xsecl, &xsecl2) {
        ensure_xsecl_text_sidecars_match(xsecl, xsecl2)?;
    }
    if caches.xsecl_bin.is_file() {
        read_matching_xsecl_bin_cache(&caches.xsecl_bin, phase)?;
    }
    Ok(())
}

fn prepare_emesh_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    if emesh_dat_needs_generation(&caches.emesh_dat) {
        emesh_dat_from_phase_bin(phase, input.control.ispec)
            .context("failed to generate emesh.dat from phase.bin")?;
    } else {
        read_emesh_dat(&caches.emesh_dat)
            .with_context(|| format!("failed to read {}", caches.emesh_dat.display()))?;
    }
    if emesh_bin_needs_generation(&caches.emesh_bin) {
        emesh_bin_from_phase_bin(phase).context("failed to generate emesh.bin from phase.bin")?;
    } else {
        read_emesh_bin(&caches.emesh_bin)
            .with_context(|| format!("failed to read {}", caches.emesh_bin.display()))?;
    }
    Ok(())
}

fn prepare_module_log_cache(caches: &XsphCachePaths) -> Result<()> {
    if caches.log2_dat.is_file() {
        read_module_log_dat(&caches.log2_dat)
            .with_context(|| format!("failed to read {}", caches.log2_dat.display()))?;
    }
    Ok(())
}

fn prepare_aphase_hubbard_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<()> {
    let active_hubbard = active_hubbard_phase_requested(caches)?;
    if caches.aphase_hubbard_bin.is_file() {
        match read_aphase_hubbard_bin_inferred(
            &caches.aphase_hubbard_bin,
            phase.energy_count,
            phase.potential_count(),
        ) {
            Ok(data) => {
                if active_hubbard {
                    ensure_aphase_hubbard_matches_source_if_available(caches, input, phase, &data)?;
                }
                return Ok(());
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to read {}", caches.aphase_hubbard_bin.display())
                });
            }
        }
    }

    ensure!(
        !active_hubbard,
        "active Hubbard XSPH output requires aphase_hubbard.bin cache or v_hubbard.bin source handoff"
    );
    Ok(())
}

fn can_use_or_repair_aphase_hubbard_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    let active_hubbard = active_hubbard_phase_requested(caches)?;
    if caches.aphase_hubbard_bin.is_file() {
        return match read_aphase_hubbard_bin_inferred(
            &caches.aphase_hubbard_bin,
            phase.energy_count,
            phase.potential_count(),
        ) {
            Ok(data) => {
                if active_hubbard {
                    generate_aphase_hubbard_if_stale_against_source(caches, input, phase, &data)?;
                }
                Ok(true)
            }
            Err(_) if active_hubbard => {
                Ok(generate_active_hubbard_aphase_cache(caches, input, phase)?.is_some())
            }
            Err(_) => Ok(false),
        };
    }

    if active_hubbard {
        return Ok(generate_active_hubbard_aphase_cache(caches, input, phase)?.is_some());
    }
    Ok(true)
}

fn write_or_generate_aphase_hubbard_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    generated: Option<HubbardAphaseBinData>,
) -> Result<(usize, bool)> {
    if let Some(data) = generated {
        write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &data)
            .with_context(|| format!("failed to write {}", caches.aphase_hubbard_bin.display()))?;
        return Ok((1, true));
    }

    if caches.aphase_hubbard_bin.is_file() {
        match read_aphase_hubbard_bin_inferred(
            &caches.aphase_hubbard_bin,
            phase.energy_count,
            phase.potential_count(),
        ) {
            Ok(data) => {
                if let Some(generated) =
                    generate_aphase_hubbard_if_stale_against_source(caches, input, phase, &data)?
                {
                    write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &generated).with_context(
                        || format!("failed to write {}", caches.aphase_hubbard_bin.display()),
                    )?;
                    return Ok((1, true));
                }
                write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &data).with_context(|| {
                    format!("failed to write {}", caches.aphase_hubbard_bin.display())
                })?;
                return Ok((1, false));
            }
            Err(error) => {
                if let Some(data) = generate_active_hubbard_aphase_cache(caches, input, phase)? {
                    write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &data).with_context(
                        || format!("failed to write {}", caches.aphase_hubbard_bin.display()),
                    )?;
                    return Ok((1, true));
                }
                return Err(error).with_context(|| {
                    format!("failed to read {}", caches.aphase_hubbard_bin.display())
                });
            }
        }
    }

    if active_hubbard_phase_requested(caches)? {
        let Some(data) = generate_active_hubbard_aphase_cache(caches, input, phase)? else {
            bail!(
                "active Hubbard XSPH output requires aphase_hubbard.bin cache or v_hubbard.bin source handoff"
            );
        };
        write_aphase_hubbard_bin(&caches.aphase_hubbard_bin, &data)
            .with_context(|| format!("failed to write {}", caches.aphase_hubbard_bin.display()))?;
        return Ok((1, true));
    }

    Ok((0, false))
}

fn ensure_aphase_hubbard_matches_source_if_available(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    cached: &HubbardAphaseBinData,
) -> Result<()> {
    ensure!(
        generate_aphase_hubbard_if_stale_against_source(caches, input, phase, cached)?.is_none(),
        "cached {} is stale against active Hubbard XSPH source handoffs",
        caches.aphase_hubbard_bin.display()
    );
    Ok(())
}

fn generate_aphase_hubbard_if_stale_against_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    cached: &HubbardAphaseBinData,
) -> Result<Option<HubbardAphaseBinData>> {
    let Some(generated) = generate_active_hubbard_aphase_cache(caches, input, phase)? else {
        return Ok(None);
    };
    if aphase_hubbard_matches_source(cached, &generated) {
        Ok(None)
    } else {
        Ok(Some(generated))
    }
}

fn aphase_hubbard_matches_source(
    cached: &HubbardAphaseBinData,
    source: &HubbardAphaseBinData,
) -> bool {
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-7;
    const RELATIVE_TOLERANCE: f64 = 1.0e-6;

    cached.angular_limit == source.angular_limit
        && complex_arrays5_match(
            &cached.values,
            &source.values,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
        )
}

fn active_hubbard_control_requested(caches: &XsphCachePaths) -> Result<bool> {
    if !caches.hubbard_inp.is_file() {
        return Ok(false);
    }

    let text = std::fs::read_to_string(&caches.hubbard_inp)
        .with_context(|| format!("failed to read {}", caches.hubbard_inp.display()))?;
    let input = HubbardInput::parse_str(&caches.hubbard_inp, &text)
        .with_context(|| format!("failed to parse {}", caches.hubbard_inp.display()))?;
    Ok(input.mldos_hubb == 2)
}

/// Whether the active-Hubbard phase source has crossed FEFF's first-pass
/// boundary.
///
/// `hubbard.inp` enables the two-pass LDOS workflow, but the first ordinary
/// XSPH/FMS pass necessarily runs before LDOS has produced `v_hubbard.bin`.
/// Treat that missing file as the bootstrap state. Once it exists, even if it
/// is malformed, the active branch owns the phase and must fail closed rather
/// than falling back to the ordinary calculation.
fn active_hubbard_phase_requested(caches: &XsphCachePaths) -> Result<bool> {
    Ok(active_hubbard_control_requested(caches)? && caches.v_hubbard_bin.is_file())
}

fn generate_active_hubbard_aphase_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<HubbardAphaseBinData>> {
    if !active_hubbard_phase_requested(caches)? {
        return Ok(None);
    }

    let Some(generated) = generate_normal_potential_phase_bin(caches, input)? else {
        return Ok(None);
    };
    let Some(aphase) = generated.aphase_hubbard else {
        return Ok(None);
    };

    ensure!(
        generated.phase.energy_count == phase.energy_count
            && generated.phase.potential_count() == phase.potential_count()
            && generated.phase.spin_count == phase.spin_count,
        "generated active Hubbard aphase_hubbard.bin dimensions do not match cached phase.bin"
    );
    Ok(Some(aphase))
}

fn can_generate_phase_matching_xsect_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    xsect: &XsectDatData,
) -> Result<bool> {
    let Some(generated) = generate_source_phase_handoff_for_discovery(caches, input)? else {
        return Ok(false);
    };
    Ok(ensure_xsect_matches_phase(&generated.phase, xsect).is_ok())
}

fn can_generate_normal_potential_base_outputs(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if !caches.pot_bin.is_file() {
        return Ok(false);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if !can_generate_normal_potential_phase_from_pot(caches, input, &pot)? {
        return Ok(false);
    }
    let can_generate_normal_xsect =
        can_generate_normal_potential_xsect_from_pot(caches, input, &pot, None)?;

    let Some(generated) = generate_source_phase_handoff_for_discovery(caches, input)? else {
        return Ok(false);
    };
    if can_generate_normal_xsect
        && generate_normal_potential_xsect_dat(caches, input, &generated.phase)?.is_some()
    {
        return Ok(true);
    }
    Ok(generate_nrixs_xsectjas_sidecars(caches, input, &generated.phase)?.is_some())
}

fn can_use_or_generate_tdlda_xsedge_outputs(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if !tdlda_xsectd_branch_requested(input) {
        return Ok(false);
    }

    if caches.has_phase_cache() {
        let phase = match read_phase_bin(&caches.phase_bin) {
            Ok(phase) => phase,
            Err(error) => {
                let Some(generated) = generate_source_phase_handoff_for_discovery(caches, input)?
                else {
                    return Ok(false);
                };
                if generate_tdlda_pmbse_xsedge_dat(caches, input, &generated.phase)?.is_some() {
                    return Ok(true);
                }
                return Err(error)
                    .with_context(|| format!("failed to read {}", caches.phase_bin.display()));
            }
        };
        if can_use_tdlda_cached_xsedge_output(caches, input, &phase)? {
            return Ok(true);
        }
        return Ok(generate_tdlda_pmbse_xsedge_dat(caches, input, &phase)?.is_some());
    }

    let Some(generated) = generate_source_phase_handoff_for_discovery(caches, input)? else {
        return Ok(false);
    };
    Ok(generate_tdlda_pmbse_xsedge_dat(caches, input, &generated.phase)?.is_some())
}

fn can_use_tdlda_cached_xsedge_output(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    if !caches.xsedge_dat.is_file() || !matches!(phase.spin_count, 1 | 2) || phase.q_count != 1 {
        return Ok(false);
    }
    let Ok(xsedge) = read_xsedge_dat(&caches.xsedge_dat) else {
        return Ok(false);
    };
    match tdlda_xsedge_shape_from_source_handoff(caches, input, phase)? {
        TdldaXsedgeSourceContractState::Absent => Ok(true),
        TdldaXsedgeSourceContractState::Present(source_shape) => Ok(
            tdlda_xsedge_cache_matches_source_shape(&xsedge, source_shape),
        ),
        TdldaXsedgeSourceContractState::Incompatible => Ok(false),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TdldaXsedgeShape {
    row_count: usize,
    has_branch_columns: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TdldaXsedgeSourceContract {
    shape: TdldaXsedgeShape,
    energy_ev: Array1<f64>,
}

#[derive(Debug, Clone, PartialEq)]
enum TdldaXsedgeSourceContractState {
    Absent,
    Present(TdldaXsedgeSourceContract),
    Incompatible,
}

fn tdlda_xsedge_shape_from_source_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<TdldaXsedgeSourceContractState> {
    let advanced = normal_xsect_effective_advanced(input);
    if !tdlda_xsectd_branch_requested(input)
        || !tdlda_nonlocal_source_is_supported(caches, advanced.nonlocal)
        || !matches!(phase.spin_count, 1 | 2)
        || phase.q_count != 1
        || !caches.pot_bin.is_file()
        || !caches.config_dat.is_file()
    {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    }

    let Ok(pot) = read_pot_bin(&caches.pot_bin) else {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    };
    if pot.potential_count() != phase.potential_count()
        || pot
            .atomic_numbers
            .iter()
            .any(|atomic_number| *atomic_number == 0)
    {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    }

    let Ok(config) = read_config_dat(&caches.config_dat) else {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    };
    let Ok(orbital_tables) = rhorrp_orbital_tables_from_config_dat(&config) else {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    };
    if orbital_tables.bound_orbital_counts.len() != pot.potential_count()
        || !pot_has_complete_bound_orbital_handoffs(&pot, &orbital_tables.bound_orbital_counts)
    {
        return Ok(TdldaXsedgeSourceContractState::Absent);
    }

    let plan = match tdlda_xsectd_source_plan_from_caches(caches, input, &pot, &orbital_tables) {
        Ok(Some(plan)) => plan,
        Ok(None) => return Ok(TdldaXsedgeSourceContractState::Absent),
        Err(_) if caches.listedges_pmbse.is_file() => {
            return Ok(TdldaXsedgeSourceContractState::Incompatible);
        }
        Err(_) => return Ok(TdldaXsedgeSourceContractState::Absent),
    };
    Ok(TdldaXsedgeSourceContractState::Present(
        TdldaXsedgeSourceContract {
            shape: TdldaXsedgeShape {
                row_count: plan.multipliers.energy_hartree.len(),
                has_branch_columns: plan.channel_count != 1,
            },
            energy_ev: plan
                .multipliers
                .energy_hartree
                .mapv(|energy| energy * FEFF_HARTREE_EV),
        },
    ))
}

fn tdlda_xsedge_cache_matches_source_shape(
    xsedge: &XsedgeDatData,
    source: TdldaXsedgeSourceContract,
) -> bool {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-5;

    xsedge.row_count() == source.shape.row_count
        && xsedge.has_branch_columns() == source.shape.has_branch_columns
        && xsedge
            .energy_ev
            .iter()
            .zip(source.energy_ev.iter())
            .all(|(actual, expected)| (actual - expected).abs() <= ENERGY_TOLERANCE_EV)
}

fn can_generate_normal_potential_xsect_from_phase_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if !caches.pot_bin.is_file() {
        return Ok(false);
    }

    let phase = read_phase_bin(&caches.phase_bin)
        .with_context(|| format!("failed to read {}", caches.phase_bin.display()))?;
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    can_generate_normal_potential_xsect_from_pot(caches, input, &pot, Some(&phase))
}

fn can_generate_normal_potential_phase_from_pot(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
) -> Result<bool> {
    if !pot_uses_supported_normal_potentials(pot) {
        return Ok(false);
    }
    if !pot_has_normal_phase_orbital_handoffs(pot) {
        return Ok(false);
    }
    if !screened_core_hole_wscrn_handoff_is_supported(caches, pot) {
        return Ok(false);
    }
    if !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())? {
        return Ok(false);
    }
    if !normal_potential_config_handoff_is_supported(caches, pot)? {
        return Ok(false);
    }
    can_prepare_xsph_excitation_poles(caches, input, input.control.ixc)
}

fn can_generate_normal_potential_xsect_from_pot(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    phase: Option<&PhaseBinData>,
) -> Result<bool> {
    if !normal_xsect_spectroscopy_supported(input.control.ispec) {
        return Ok(false);
    }
    let Some(controls) = xsect_angular_controls(caches, input)? else {
        return Ok(false);
    };
    let advanced = normal_xsect_effective_advanced(input);
    if !normal_xsect_controls_supported(advanced, controls) {
        return Ok(false);
    }
    let spin_count = match phase {
        Some(phase) => phase.spin_count,
        None => phase_spin_selectors(caches, input)?.len(),
    };
    if !xsect_spin_state_supported(spin_count, controls) {
        return Ok(false);
    }
    if let Some(phase) = phase
        && (phase.q_count != 1
            || phase.transition_count == 0
            || phase.transition_count > PHASE_BIN_DEFAULT_TRANSITION_COUNT
            || phase.potential_count() != pot.potential_count())
    {
        return Ok(false);
    }
    if !pot_uses_supported_normal_potentials(pot)
        || pot
            .initial_large_component
            .iter()
            .chain(pot.initial_small_component.iter())
            .all(|value| *value == 0.0)
    {
        return Ok(false);
    }
    if !screened_core_hole_wscrn_handoff_is_supported(caches, pot) {
        return Ok(false);
    }
    if !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())? {
        return Ok(false);
    }
    if !normal_potential_config_handoff_is_supported(caches, pot)? {
        return Ok(false);
    }
    can_prepare_xsph_excitation_poles(caches, input, input.control.ixc0)
}

fn pot_uses_supported_normal_potentials(pot: &PotBinData) -> bool {
    !pot.atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
        && !pot
            .atomic_numbers
            .iter()
            .any(|atomic_number| *atomic_number == 0)
}

fn pot_has_normal_phase_orbital_handoffs(pot: &PotBinData) -> bool {
    pot.large_components.iter().any(|value| value.abs() > 0.0)
        && pot.large_coefficients.iter().any(|value| value.abs() > 0.0)
}

fn can_generate_source_phase_handoff_for_discovery(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    Ok(generate_source_phase_handoff_for_discovery(caches, input)?.is_some())
}

fn normal_potential_config_handoff_is_supported(
    caches: &XsphCachePaths,
    pot: &PotBinData,
) -> Result<bool> {
    let Some(orbital_tables) = normal_potential_orbital_tables(caches, pot)? else {
        return Ok(false);
    };
    Ok(pot_has_bound_orbital_handoffs(pot, &orbital_tables))
}

fn normal_potential_orbital_tables(
    caches: &XsphCachePaths,
    pot: &PotBinData,
) -> Result<Option<RhorrpConfigOrbitalTables>> {
    if !caches.config_dat.is_file() {
        return pot_derived_orbital_tables(pot);
    }
    let config = read_config_dat(&caches.config_dat)
        .with_context(|| format!("failed to read {}", caches.config_dat.display()))?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)
        .with_context(|| format!("failed to prepare {}", caches.config_dat.display()))?;
    Ok(Some(orbital_tables))
}

fn pot_derived_orbital_tables(pot: &PotBinData) -> Result<Option<RhorrpConfigOrbitalTables>> {
    ensure!(
        pot.orbital_occupancy.ncols() == pot.potential_count(),
        "pot.bin orbital occupancy potential count {} does not match potential count {}",
        pot.orbital_occupancy.ncols(),
        pot.potential_count()
    );

    let potential_count = pot.potential_count();
    let orbital_capacity = pot.orbital_occupancy.nrows().min(pot.kappa.len());
    let mut compacted = Vec::with_capacity(potential_count);
    for potential_index in 0..potential_count {
        let mut electron_counts = Vec::new();
        let mut valence_counts = Vec::new();
        let mut kappa = Vec::new();
        let mut orbital_slots = Vec::new();
        for orbital in 0..orbital_capacity {
            let orbital_kappa = pot.kappa[orbital];
            if orbital_kappa == 0 {
                continue;
            }
            let valence_count = pot.orbital_occupancy[(orbital, potential_index)];
            ensure!(
                valence_count.is_finite() && valence_count >= 0.0,
                "pot.bin orbital occupancy for orbital {} potential {potential_index} must be finite and nonnegative, got {valence_count}",
                orbital + 1
            );
            let has_radial = pot_bound_orbital_has_radial_components(pot, potential_index, orbital);
            let has_coefficients =
                pot_bound_orbital_has_coefficients(pot, potential_index, orbital);
            ensure!(
                has_radial == has_coefficients,
                "pot.bin orbital {} for potential {potential_index} has incomplete radial/coefficient handoff",
                orbital + 1
            );
            if valence_count <= 0.0 && !has_radial {
                continue;
            }
            let mut electron_count = if valence_count > 0.0 {
                valence_count
            } else {
                2.0 * f64::from(orbital_kappa.unsigned_abs())
            };
            if potential_index == 0
                && pot.nohole == 0
                && usize::try_from(pot.ihole)
                    .ok()
                    .is_some_and(|hole| hole == orbital + 1)
                && electron_count >= 1.0
            {
                electron_count -= 1.0;
            }
            ensure!(
                electron_count.is_finite() && electron_count >= 0.0,
                "pot.bin derived electron count for orbital {} potential {potential_index} must be finite and nonnegative, got {electron_count}",
                orbital + 1
            );
            electron_counts.push(electron_count);
            valence_counts.push(valence_count);
            kappa.push(orbital_kappa);
            orbital_slots.push(orbital);
        }
        if electron_counts.is_empty() {
            return Ok(None);
        }
        compacted.push((electron_counts, valence_counts, kappa, orbital_slots));
    }

    let bound_orbital_counts = compacted
        .iter()
        .map(|(electron_counts, _, _, _)| electron_counts.len())
        .collect::<Vec<_>>();
    let max_orbitals = bound_orbital_counts.iter().copied().max().unwrap_or(0);
    let mut electron_counts_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut valence_counts_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut kappa_by_potential = Array2::zeros((max_orbitals, potential_count));
    let mut orbital_slots_by_potential = Array2::zeros((max_orbitals, potential_count));

    for (potential_index, (electron_counts, valence_counts, kappa, orbital_slots)) in
        compacted.iter().enumerate()
    {
        for orbital_index in 0..electron_counts.len() {
            electron_counts_by_potential[(orbital_index, potential_index)] =
                electron_counts[orbital_index];
            valence_counts_by_potential[(orbital_index, potential_index)] =
                valence_counts[orbital_index];
            kappa_by_potential[(orbital_index, potential_index)] = kappa[orbital_index];
            orbital_slots_by_potential[(orbital_index, potential_index)] =
                orbital_slots[orbital_index];
        }
    }

    Ok(Some(RhorrpConfigOrbitalTables {
        electron_counts_by_potential,
        valence_counts_by_potential,
        kappa_by_potential,
        orbital_slots_by_potential,
        bound_orbital_counts,
    }))
}

fn normal_potential_hubbard_phase_branch_is_supported(
    caches: &XsphCachePaths,
    potential_count: usize,
) -> Result<bool> {
    if !active_hubbard_control_requested(caches)? {
        return Ok(true);
    }
    if !caches.v_hubbard_bin.is_file() {
        // FEFF's first Hubbard pass is the ordinary phase calculation. LDOS
        // consumes it to produce v_hubbard.bin and the transformation table.
        return Ok(true);
    }
    read_v_hubbard_bin_inferred(&caches.v_hubbard_bin, potential_count)
        .with_context(|| format!("failed to read {}", caches.v_hubbard_bin.display()))?;
    Ok(true)
}

fn read_active_hubbard_phase_handoff(
    caches: &XsphCachePaths,
    potential_count: usize,
) -> Result<Option<HubbardVnlmBinData>> {
    if !active_hubbard_control_requested(caches)? {
        return Ok(None);
    }
    if !caches.v_hubbard_bin.is_file() {
        return Ok(None);
    }
    read_v_hubbard_bin_inferred(&caches.v_hubbard_bin, potential_count)
        .with_context(|| format!("failed to read {}", caches.v_hubbard_bin.display()))
        .map(Some)
}

fn pot_has_bound_orbital_handoffs(
    pot: &PotBinData,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> bool {
    pot_effective_bound_orbital_counts(pot, orbital_tables).is_ok()
}

fn pot_has_complete_bound_orbital_handoffs(
    pot: &PotBinData,
    bound_orbital_counts: &[usize],
) -> bool {
    if bound_orbital_counts.len() != pot.potential_count() {
        return false;
    }

    for (potential_index, &bound_orbital_count) in bound_orbital_counts.iter().enumerate() {
        if bound_orbital_count == 0
            || bound_orbital_count > pot.large_components.len_of(Axis(1))
            || bound_orbital_count > pot.small_components.len_of(Axis(1))
            || bound_orbital_count > pot.large_coefficients.len_of(Axis(1))
            || bound_orbital_count > pot.small_coefficients.len_of(Axis(1))
            || potential_index >= pot.large_components.len_of(Axis(2))
            || potential_index >= pot.small_components.len_of(Axis(2))
            || potential_index >= pot.large_coefficients.len_of(Axis(2))
            || potential_index >= pot.small_coefficients.len_of(Axis(2))
        {
            return false;
        }

        for orbital in 0..bound_orbital_count {
            if !pot_bound_orbital_has_radial_components(pot, potential_index, orbital)
                || !pot_bound_orbital_has_coefficients(pot, potential_index, orbital)
            {
                return false;
            }
        }
    }

    true
}

fn pot_effective_bound_orbital_counts(
    pot: &PotBinData,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<Vec<usize>> {
    let potential_count = pot.potential_count();
    ensure!(
        orbital_tables.bound_orbital_counts.len() == potential_count,
        "config.dat potential count {} does not match pot.bin potential count {potential_count}",
        orbital_tables.bound_orbital_counts.len()
    );
    ensure!(
        pot.orbital_occupancy.ncols() == potential_count,
        "pot.bin orbital occupancy potential count {} does not match potential count {potential_count}",
        pot.orbital_occupancy.ncols()
    );

    let mut effective_counts = Vec::with_capacity(potential_count);
    for (potential_index, &configured_count) in
        orbital_tables.bound_orbital_counts.iter().enumerate()
    {
        ensure!(
            configured_count > 0,
            "config.dat potential {potential_index} has no occupied orbitals"
        );
        ensure!(
            configured_count <= pot.orbital_occupancy.nrows()
                && configured_count <= orbital_tables.electron_counts_by_potential.nrows()
                && configured_count <= pot.large_components.len_of(Axis(1))
                && configured_count <= pot.small_components.len_of(Axis(1))
                && configured_count <= pot.large_coefficients.len_of(Axis(1))
                && configured_count <= pot.small_coefficients.len_of(Axis(1))
                && potential_index < orbital_tables.electron_counts_by_potential.ncols()
                && potential_index < pot.large_components.len_of(Axis(2))
                && potential_index < pot.small_components.len_of(Axis(2))
                && potential_index < pot.large_coefficients.len_of(Axis(2))
                && potential_index < pot.small_coefficients.len_of(Axis(2)),
            "XSPH bound orbital count {configured_count} for potential {potential_index} exceeds pot/config handoff shapes"
        );

        let mut effective_count = 0usize;
        let mut active_by_orbital = Vec::with_capacity(configured_count);
        for orbital in 0..configured_count {
            let has_radial = pot_bound_orbital_has_radial_components(pot, potential_index, orbital);
            let has_coefficients =
                pot_bound_orbital_has_coefficients(pot, potential_index, orbital);
            ensure!(
                has_radial == has_coefficients,
                "XSPH bound orbital {} for potential {potential_index} has incomplete radial/coefficient handoff",
                orbital + 1
            );
            if has_radial {
                effective_count = orbital + 1;
                active_by_orbital.push(true);
                continue;
            }

            let electron_count =
                orbital_tables.electron_counts_by_potential[(orbital, potential_index)];
            let valence_count = pot.orbital_occupancy[(orbital, potential_index)];
            ensure!(
                electron_count.is_finite() && valence_count.is_finite(),
                "XSPH bound orbital {} for potential {potential_index} has non-finite counts",
                orbital + 1
            );
            let core_count = electron_count - valence_count;
            ensure!(
                core_count.abs() <= XSPH_ORBITAL_CORE_COUNT_TOLERANCE,
                "XSPH core orbital {} for potential {potential_index} has no radial handoff but core count {core_count}",
                orbital + 1
            );
            active_by_orbital.push(false);
        }

        ensure!(
            effective_count > 0,
            "XSPH potential {potential_index} has no active bound orbital handoffs"
        );
        for (orbital, &active) in active_by_orbital.iter().take(effective_count).enumerate() {
            ensure!(
                active,
                "XSPH zero-component orbital {} for potential {potential_index} is not trailing and cannot be omitted",
                orbital + 1
            );
        }
        effective_counts.push(effective_count);
    }

    Ok(effective_counts)
}

fn pot_bound_orbital_has_radial_components(
    pot: &PotBinData,
    potential_index: usize,
    orbital: usize,
) -> bool {
    pot.large_components
        .index_axis(Axis(2), potential_index)
        .index_axis(Axis(1), orbital)
        .iter()
        .chain(
            pot.small_components
                .index_axis(Axis(2), potential_index)
                .index_axis(Axis(1), orbital)
                .iter(),
        )
        .any(|value| value.abs() >= XSPH_BOUND_ORBITAL_COMPONENT_THRESHOLD)
}

fn pot_bound_orbital_has_coefficients(
    pot: &PotBinData,
    potential_index: usize,
    orbital: usize,
) -> bool {
    pot.large_coefficients
        .index_axis(Axis(2), potential_index)
        .index_axis(Axis(1), orbital)
        .iter()
        .chain(
            pot.small_coefficients
                .index_axis(Axis(2), potential_index)
                .index_axis(Axis(1), orbital)
                .iter(),
        )
        .any(|value| value.abs() >= XSPH_BOUND_ORBITAL_COMPONENT_THRESHOLD)
}

fn xsph_total_potential_with_screened_core_hole(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
) -> Result<Array2<f64>> {
    let mut total_potential = pot.total_potential.clone();
    apply_xsph_screened_core_hole(caches, input, pot, &mut total_potential)?;
    Ok(total_potential)
}

fn apply_xsph_screened_core_hole(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    total_potential: &mut Array2<f64>,
) -> Result<()> {
    if pot.nohole != XSPH_SCREENED_CORE_HOLE_SELECTOR {
        return Ok(());
    }

    ensure!(
        recover_screened_core_hole_wscrn_handoff_if_needed(caches, pot)?,
        "screened core-hole XSPH handoff requires wscrn.dat cache or vtot.dat/apot.bin source handoff"
    );
    ensure!(
        pot.potential_count() > 0 && total_potential.ncols() > 0,
        "screened core-hole XSPH handoff requires an absorber potential column"
    );
    ensure!(
        pot.initial_large_component.len() >= total_potential.nrows()
            && pot.initial_small_component.len() >= total_potential.nrows()
            && pot.electron_density.nrows() >= total_potential.nrows(),
        "screened core-hole XSPH handoff radial lengths are inconsistent: vtot={}, edens={}, dgc0={}, dpc0={}",
        total_potential.nrows(),
        pot.electron_density.nrows(),
        pot.initial_large_component.len(),
        pot.initial_small_component.len()
    );

    let wscrn = read_wscrn_dat(&caches.wscrn_dat)
        .with_context(|| format!("failed to read {}", caches.wscrn_dat.display()))?;
    let xes_sign = input.control.ispec == XSPH_SCREENED_CORE_HOLE_SELECTOR;
    for row in 0..wscrn.screened_potential.len().min(total_potential.nrows()) {
        if xes_sign {
            total_potential[(row, 0)] += wscrn.screened_potential[row];
        } else {
            total_potential[(row, 0)] -= wscrn.screened_potential[row];
        }
    }

    for row in 0..total_potential.nrows() {
        let radius = (-XSPH_LOUCKS_GRID_ORIGIN + LOUCKS_DELTA * row as f64).exp();
        let density = pot.electron_density[(row, 0)];
        let total_density_radius = xsph_screened_core_hole_density_radius(density);
        let total_xc = von_barth_hedin_potential(total_density_radius, 0.0)
            .context("failed to evaluate XSPH screened core-hole total XC potential")?;

        let core_density = (pot.initial_large_component[row].powi(2)
            + pot.initial_small_component[row].powi(2))
            / radius.powi(2);
        let core_hole_density_radius =
            xsph_screened_core_hole_density_radius(density - core_density);
        let core_hole_xc = von_barth_hedin_potential(core_hole_density_radius, 0.0)
            .context("failed to evaluate XSPH screened core-hole XC correction")?;
        let xc_delta = core_hole_xc - total_xc;
        if xes_sign {
            total_potential[(row, 0)] -= xc_delta;
        } else {
            total_potential[(row, 0)] += xc_delta;
        }
    }

    Ok(())
}

fn xsph_scaled_magnetization(input: &XsphInput, pot: &PotBinData) -> Result<Array2<f64>> {
    ensure!(
        input.spinph.len() >= pot.potential_count(),
        "XSPH spinph has {} value(s), expected at least {}",
        input.spinph.len(),
        pot.potential_count()
    );
    Ok(Array2::from_shape_fn(
        pot.magnetization_density.dim(),
        |(row, potential)| pot.magnetization_density[(row, potential)] * input.spinph[potential],
    ))
}

#[derive(Debug)]
struct XsphSpinGroundState {
    total_potential: Array2<f64>,
    valence_potential: Array2<f64>,
    interstitial_potential: f64,
    interstitial_density: f64,
}

fn xsph_spin_ground_state(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    magnetization: &Array2<f64>,
    spin_selector: i32,
) -> Result<XsphSpinGroundState> {
    if spin_selector == 0 {
        return Ok(XsphSpinGroundState {
            total_potential: pot.total_potential.clone(),
            valence_potential: pot.valence_potential.clone(),
            interstitial_potential: pot.scalars.interstitial_potential,
            interstitial_density: pot.scalars.interstitial_density,
        });
    }
    if let Some(state) =
        xsph_spin_ground_state_from_istprm(caches, input, pot, magnetization, spin_selector)?
    {
        return Ok(state);
    }

    xsph_spin_ground_state_local(input, pot, magnetization, spin_selector)
}

fn xsph_spin_ground_state_local(
    input: &XsphInput,
    pot: &PotBinData,
    magnetization: &Array2<f64>,
    spin_selector: i32,
) -> Result<XsphSpinGroundState> {
    ensure!(
        pot.coulomb_potential.dim() == pot.electron_density.dim()
            && pot.valence_density.dim() == pot.electron_density.dim()
            && magnetization.dim() == pot.electron_density.dim(),
        "XSPH magnetic ground-state potential tables have inconsistent shapes"
    );
    let spin_polarization = spin_selector.signum() as f64;
    let exchange_branch = input.control.ixc.rem_euclid(10);
    let mut total_potential = Array2::<f64>::zeros(pot.electron_density.dim());
    let mut valence_potential = Array2::<f64>::zeros(pot.electron_density.dim());

    for ((row, potential), &density) in pot.electron_density.indexed_iter() {
        let magnetic_fraction = magnetization[(row, potential)];
        let (density_radius, spin_fraction_twice) = if density <= 0.0 {
            (100.0, 1.0)
        } else {
            (
                (density / 3.0).powf(-1.0 / 3.0),
                1.0 + spin_polarization * magnetic_fraction,
            )
        };
        let ground_state_xc = xsph_ground_state_xc(input, density_radius, spin_fraction_twice)?;
        let coulomb = pot.coulomb_potential[(row, potential)];
        total_potential[(row, potential)] = coulomb + ground_state_xc;

        if exchange_branch == 5 {
            let valence_density = pot.valence_density[(row, potential)];
            let valence_radius = if valence_density > 1.0e-5 {
                (valence_density / 3.0).powf(-1.0 / 3.0).min(10.0)
            } else {
                10.0
            };
            let valence_spin_fraction_twice = if valence_density == 0.0 {
                1.0
            } else {
                1.0 + spin_polarization * magnetic_fraction * density / valence_density
            };
            valence_potential[(row, potential)] = coulomb
                + von_barth_hedin_potential(valence_radius, valence_spin_fraction_twice)
                    .context("failed to evaluate XSPH magnetic valence XC potential")?;
        } else if exchange_branch >= 6 {
            let valence_density = pot.valence_density[(row, potential)];
            let core_radius = if density <= valence_density {
                101.0
            } else {
                ((density - valence_density) / 3.0).powf(-1.0 / 3.0)
            };
            let magnetized_density = density * spin_fraction_twice;
            let magnetized_radius = if magnetized_density > 0.0 {
                (magnetized_density / 3.0).powf(-1.0 / 3.0)
            } else {
                100.0
            };
            let dirac_hara = dirac_hara_exchange_potential(
                core_radius,
                FEFF_FERMI_MOMENTUM_FACTOR / magnetized_radius,
            )
            .context("failed to evaluate XSPH magnetic Dirac-Hara potential")?;
            valence_potential[(row, potential)] = coulomb + ground_state_xc - dirac_hara;
        }
    }

    Ok(XsphSpinGroundState {
        total_potential,
        valence_potential,
        interstitial_potential: pot.scalars.interstitial_potential,
        interstitial_density: pot.scalars.interstitial_density,
    })
}

fn xsph_spin_ground_state_from_istprm(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    magnetization: &Array2<f64>,
    spin_selector: i32,
) -> Result<Option<XsphSpinGroundState>> {
    let pot_path = caches.work_dir.join("pot.inp");
    let geom_path = caches.work_dir.join("geom.dat");
    if !pot_path.is_file() || !geom_path.is_file() {
        return Ok(None);
    }

    let pot_text = std::fs::read_to_string(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let pot_input = PotInput::parse_str(&pot_path, &pot_text)
        .with_context(|| format!("failed to parse {}", pot_path.display()))?;
    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    let geom = GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))?;
    let geometry = geom
        .to_rhorrp_handoff()
        .context("failed to convert XSPH geom.dat to Bohr geometry")?;
    ensure!(
        geometry.potential_count() == pot.potential_count()
            && pot_input.potentials.len() == pot.potential_count(),
        "XSPH magnetic istprm source potential counts disagree: geom={}, pot.inp={}, pot.bin={}",
        geometry.potential_count(),
        pot_input.potentials.len(),
        pot.potential_count()
    );

    let explicit_overlaps = pot_input
        .overlap_shells
        .iter()
        .map(|shells| {
            shells
                .iter()
                .map(|shell| {
                    Ok(MuffinTinOverlapNeighbor {
                        source_potential: usize::try_from(shell.iphovr)
                            .context("XSPH overlap potential index is negative")?,
                        multiplicity: usize::try_from(shell.nnovr)
                            .context("XSPH overlap multiplicity is negative")?,
                        distance: shell.rovr / FEFF_BOHR_ANGSTROM,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let explicit_overlap_refs = explicit_overlaps
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let near_neighbor_flags = Array1::<bool>::from_elem(pot.potential_count(), false);
    let atom_potentials = Array1::from_vec(geometry.atom_potentials);
    let representative_atoms = Array1::from_vec(geometry.representative_atoms);
    let total_volume = if pot_input.scattering.totvol > 0.0 {
        pot_input.scattering.totvol / FEFF_BOHR_ANGSTROM.powi(3)
    } else {
        pot_input.scattering.totvol
    };
    let state = muffin_tin_interstitial_parameters(MuffinTinInterstitialParametersInput {
        highest_potential_index: pot.potential_count() - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: geometry.atom_positions_bohr.view(),
        representative_atoms: representative_atoms.view(),
        potential_multiplicities: pot.potential_multiplicities.view(),
        explicit_overlaps: &explicit_overlap_refs,
        electron_density: pot.electron_density.view(),
        valence_density: pot.valence_density.view(),
        magnetization: magnetization.view(),
        coulomb_potential: pot.coulomb_potential.view(),
        muffin_tin_radii: pot.muffin_tin_radii.view(),
        norman_radii: pot.norman_radii.view(),
        near_neighbor_flags: near_neighbor_flags.view(),
        exchange_selector: input.control.ixc,
        scf_exchange_selector: input.control.iscfxc,
        spin_polarization: spin_selector.signum(),
        scf_temperature_hartree: input.electronic_temperature / FEFF_HARTREE_EV,
        total_charge: pot.scalars.total_charge,
        fermi_level: pot.scalars.fermi_level,
        total_volume,
        interstitial_selector: usize::try_from(pot.interstitial_selector)
            .context("XSPH interstitial selector is negative")?,
    })
    .context("failed to run XSPH magnetic istprm refresh")?;

    Ok(Some(XsphSpinGroundState {
        total_potential: state.total_potential,
        valence_potential: state.valence_potential,
        interstitial_potential: state.interstitial_potential,
        interstitial_density: state.interstitial_density,
    }))
}

fn xsph_ground_state_xc(
    input: &XsphInput,
    density_radius: f64,
    spin_fraction_twice: f64,
) -> Result<f64> {
    match input.control.iscfxc {
        11 => von_barth_hedin_potential(density_radius, spin_fraction_twice)
            .context("failed to evaluate XSPH magnetic von Barth-Hedin potential"),
        12 => perdew_zunger_vxc(density_radius)
            .context("failed to evaluate XSPH magnetic Perdew-Zunger potential"),
        21 => perrot_dharma_wardana_vxc(
            density_radius,
            input.electronic_temperature / FEFF_HARTREE_EV,
        )
        .context("failed to evaluate XSPH magnetic Perrot-Dharma-Wardana potential"),
        22 => karasiev_sjostrom_dufty_trickey_vxc(
            density_radius,
            input.electronic_temperature / FEFF_HARTREE_EV,
        )
        .context("failed to evaluate XSPH magnetic KSDT potential"),
        selector => bail!(
            "XSPH magnetic ground-state potential requires iscfxc 11, 12, 21, or 22, got {selector}"
        ),
    }
}

fn screened_core_hole_wscrn_handoff_is_supported(
    caches: &XsphCachePaths,
    pot: &PotBinData,
) -> bool {
    pot.nohole != XSPH_SCREENED_CORE_HOLE_SELECTOR
        || screen::has_usable_wscrn_handoff_in_dir(&caches.work_dir)
        || screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(&caches.work_dir)
}

fn recover_screened_core_hole_wscrn_handoff_if_needed(
    caches: &XsphCachePaths,
    pot: &PotBinData,
) -> Result<bool> {
    if pot.nohole != XSPH_SCREENED_CORE_HOLE_SELECTOR
        || screen::has_usable_wscrn_handoff_in_dir(&caches.work_dir)
    {
        return Ok(true);
    }
    if !screen::has_recoverable_wscrn_from_vtot_and_apot_in_dir(&caches.work_dir) {
        return Ok(false);
    }
    screen::recover_wscrn_from_vtot_and_apot_in_dir(&caches.work_dir)
        .context("failed to recover XSPH wscrn.dat handoff")?;
    Ok(true)
}

fn xsph_screened_core_hole_density_radius(density: f64) -> f64 {
    if density <= 0.0 {
        10.0
    } else {
        (density / 3.0).powf(-1.0 / 3.0)
    }
}

fn can_prepare_xsph_excitation_poles(
    caches: &XsphCachePaths,
    input: &XsphInput,
    exchange_selector: i32,
) -> Result<bool> {
    if input.control.i_plsmn <= 0 || exchange_selector != 0 {
        return Ok(true);
    }
    Ok(xsph_excitation_poles_from_loss(caches, input, exchange_selector)?.is_some())
}

fn has_supported_print_rl_output(caches: &XsphCachePaths, input: &XsphInput) -> Result<bool> {
    if !input.print_rl {
        return Ok(true);
    }
    if caches.rl_dat.is_file() && read_xsph_rl_dat(&caches.rl_dat).is_ok() {
        return Ok(true);
    }
    if !caches.pot_bin.is_file() {
        return Ok(false);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
    {
        return Ok(false);
    }
    can_generate_normal_potential_phase_from_pot(caches, input, &pot)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct XsphXsectAngularControls {
    polarization: i32,
    polarization_tensor: [[Complex64; 3]; 3],
    higher_multipole_selector: i32,
    combined_higher_multipoles: bool,
    spin: i32,
    spin_vector_angle: f64,
    selected_higher_multipole: Option<XsphTransitionMultipole>,
    transition_direction: i32,
}

impl XsphXsectAngularControls {
    fn for_single_higher_multipole(self, higher_multipole_selector: i32) -> Result<Self> {
        let selected_higher_multipole = match higher_multipole_selector {
            0 => None,
            1 => Some(XsphTransitionMultipole::MagneticDipole),
            2 => Some(XsphTransitionMultipole::ElectricQuadrupole),
            _ => bail!(
                "single XSPH multipole pass requires selector 0, 1, or 2, got {higher_multipole_selector}"
            ),
        };
        Ok(Self {
            higher_multipole_selector,
            combined_higher_multipoles: false,
            selected_higher_multipole,
            ..self
        })
    }
}

fn xsect_angular_controls(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<XsphXsectAngularControls>> {
    let (
        polarization,
        mut polarization_tensor,
        higher_multipole_selector,
        spin,
        spin_vector_angle,
        transition_direction,
    ) = if let Some(global) = read_optional_global_input(&caches.global_inp)? {
        (
            global.control.ipol,
            xsph_global_polarization_tensor(&global),
            global.control.le2,
            global.control.ispin,
            global.control.angks,
            global.control.l2lp,
        )
    } else {
        (
            0,
            [[Complex64::new(0.0, 0.0); 3]; 3],
            0,
            0,
            0.0,
            input.control.l2lp,
        )
    };
    if xsph_eels_orientation_average_requested(caches, input)? {
        polarization_tensor = xsph_orientation_averaged_polarization_tensor();
    }
    xsect_angular_controls_from_values(
        polarization,
        polarization_tensor,
        higher_multipole_selector,
        spin,
        spin_vector_angle,
        transition_direction,
    )
}

fn xsph_eels_orientation_average_requested(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if input.control.mphase != 1 || !caches.eels_inp.is_file() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&caches.eels_inp)
        .with_context(|| format!("failed to read {}", caches.eels_inp.display()))?;
    let eels = EelsInput::parse_str(&caches.eels_inp, &text)
        .with_context(|| format!("failed to parse {}", caches.eels_inp.display()))?;
    Ok(eels.calculation_mode == 1)
}

fn xsph_orientation_averaged_polarization_tensor() -> [[Complex64; 3]; 3] {
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for (index, row) in tensor.iter_mut().enumerate() {
        row[index] = Complex64::new(1.0 / 3.0, 0.0);
    }
    tensor
}

fn xsect_angular_controls_from_values(
    polarization: i32,
    polarization_tensor: [[Complex64; 3]; 3],
    higher_multipole_selector: i32,
    spin: i32,
    spin_vector_angle: f64,
    transition_direction: i32,
) -> Result<Option<XsphXsectAngularControls>> {
    let (selected_higher_multipole, combined_higher_multipoles) = match higher_multipole_selector {
        0 => (None, false),
        1 => (Some(XsphTransitionMultipole::MagneticDipole), false),
        2 => (Some(XsphTransitionMultipole::ElectricQuadrupole), false),
        // FEFF RDINP/mkptz defines le2=3 as E1+E2+M1. The underlying
        // XSPH bcoef/radial kernels still operate on one higher multipole
        // at a time, so source generation evaluates and combines those
        // passes below.
        3 => (None, true),
        _ => return Ok(None),
    };
    if !matches!(transition_direction, -1..=1) {
        return Ok(None);
    }
    Ok(Some(XsphXsectAngularControls {
        polarization,
        polarization_tensor,
        higher_multipole_selector,
        combined_higher_multipoles,
        spin,
        spin_vector_angle,
        selected_higher_multipole,
        transition_direction,
    }))
}

fn normal_xsect_positive_izstd_controls_supported(
    advanced: XsphAdvanced,
    controls: XsphXsectAngularControls,
) -> bool {
    advanced.izstd <= 0
        || (!controls.combined_higher_multipoles
            && matches!(
                controls.selected_higher_multipole,
                None | Some(XsphTransitionMultipole::ElectricQuadrupole)
            ))
}

fn normal_xsect_controls_supported(
    advanced: XsphAdvanced,
    controls: XsphXsectAngularControls,
) -> bool {
    advanced.itdlda == 0 && normal_xsect_positive_izstd_controls_supported(advanced, controls)
}

fn normal_xsect_spectroscopy_supported(spectroscopy: i32) -> bool {
    // FEFF xsphsub.f90 routes EXAFS, XANES/SPXAS, XES, DANES, and FPRIME
    // through XSPH/xsect.f90 when the TDLDA/PMBSE xsectd branch is disabled.
    // XMCD/SPXAS handoffs use ispec = -1 for the no-FMS XANES-style grid.
    matches!(spectroscopy, -1..=4)
}

fn tdlda_xsectd_branch_requested(input: &XsphInput) -> bool {
    normal_xsect_effective_advanced(input).itdlda != 0
}

#[derive(Debug, Clone, PartialEq)]
struct XsphTdldaXsectdPlan {
    initial_kappa: i32,
    initial_l: i32,
    channel_count: usize,
    primary_channel_count: usize,
    plus_basis_count: usize,
    minus_basis_count: usize,
    matrix_size: usize,
    basis: XsphTdldaChannelBasis,
    spin_orbit_split: f64,
    plus_broadening: f64,
    minus_broadening: f64,
    reference_shifts: Array1<f64>,
    row_broadenings: Array1<f64>,
    multipliers: XsphTdldaChannelMultipliers,
}

#[derive(Debug, Clone, PartialEq)]
struct XsphTdldaGetchi0Kernel {
    kernel: Array2<Complex64>,
    projected_kernel: Array2<Complex64>,
    direct: XsphTdldaDirectKernel,
    radial: XsphTdldaRadialKernel,
    angular: XsphTdldaAngularKernel,
}

#[derive(Debug, Clone, PartialEq)]
struct XsphTdldaProjectorRows {
    localized_large: Array2<f64>,
    localized_small: Array2<f64>,
    source_rows: Array1<bool>,
    generated_rows: Array1<bool>,
    selector_indices: Array1<usize>,
    norm_integrals: Array1<f64>,
    norm_sqrt: Array1<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct XsphTdldaRawResponseInputs {
    overlaps: Array1<f64>,
    localized_dipoles: Array1<f64>,
    full_dipoles: Array1<f64>,
}

#[derive(Debug, Clone, Copy)]
struct XsphTdldaGeneratedProjectorCandidatesInput<'a> {
    input: &'a XsphInput,
    broadened_table: Option<&'a BroadenedHedinLundqvistTable>,
    plan: &'a XsphTdldaXsectdPlan,
    generated_basis_count: usize,
    active_len: usize,
    generated_target_last_index: usize,
    xcpot_active_len: usize,
    muffin_tin_radius: f64,
    radial_match_index: usize,
    total_potential: ArrayView1<'a, f64>,
    valence_potential: ArrayView1<'a, f64>,
    electron_density: ArrayView1<'a, f64>,
    magnetization: ArrayView1<'a, f64>,
    valence_density: ArrayView1<'a, f64>,
    many_pole_self_energy: Option<&'a XsphManyPoleSelfEnergy>,
    fermi_level: f64,
    radii: ArrayView1<'a, f64>,
    radial_dx: f64,
    bound_large: ArrayView2<'a, f64>,
    bound_small: ArrayView2<'a, f64>,
    bound_large_coefficients: ArrayView2<'a, f64>,
    bound_small_coefficients: ArrayView2<'a, f64>,
    electron_counts: ArrayView1<'a, f64>,
    valence_counts: ArrayView1<'a, f64>,
    kappa: ArrayView1<'a, i32>,
    atomic_number: f64,
    bound_orbital_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct XsphTdldaFileProjectorCandidatesInput<'a> {
    work_dir: &'a Path,
    plan: &'a XsphTdldaXsectdPlan,
    generated_basis_count: usize,
    active_len: usize,
    file_target_last_index: usize,
    radii: ArrayView1<'a, f64>,
}

fn tdlda_xsectd_source_plan_from_caches(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    orbital_tables: &RhorrpConfigOrbitalTables,
) -> Result<Option<XsphTdldaXsectdPlan>> {
    if !tdlda_xsectd_branch_requested(input) {
        return Ok(None);
    }
    let core_hole = core_hole_quantum_numbers(pot.ihole)
        .context("failed to determine XSPH TDLDA initial-state quantum numbers")?;
    ensure!(
        core_hole.kappa < 0,
        "XSPH TDLDA initial-state kappa should be negative, got {}",
        core_hole.kappa
    );
    let Some(multipliers) = tdlda_pmbse_channel_multipliers_from_caches(
        caches,
        core_hole.kappa,
        XSPH_TDLDA_MESH_HORIZONTAL_COUNT + XSPH_TDLDA_MESH_EXTRA_COUNT,
    )?
    else {
        return Ok(None);
    };

    Ok(Some(tdlda_xsectd_source_plan(
        input,
        pot,
        orbital_tables,
        multipliers,
    )?))
}

fn tdlda_xsectd_source_plan(
    input: &XsphInput,
    pot: &PotBinData,
    orbital_tables: &RhorrpConfigOrbitalTables,
    multipliers: XsphTdldaChannelMultipliers,
) -> Result<XsphTdldaXsectdPlan> {
    const ABSORBER_INDEX: usize = 0;

    let core_hole = core_hole_quantum_numbers(pot.ihole)
        .context("failed to determine XSPH TDLDA initial-state quantum numbers")?;
    ensure!(
        core_hole.kappa < 0,
        "XSPH TDLDA initial-state kappa should be negative, got {}",
        core_hole.kappa
    );
    ensure!(
        !pot.atomic_numbers.is_empty(),
        "XSPH TDLDA source plan requires an absorber atomic number"
    );
    ensure!(
        ABSORBER_INDEX < orbital_tables.kappa_by_potential.ncols()
            && ABSORBER_INDEX < orbital_tables.valence_counts_by_potential.ncols()
            && ABSORBER_INDEX < orbital_tables.bound_orbital_counts.len(),
        "XSPH TDLDA source plan requires absorber orbital handoffs"
    );
    let active_orbital_count = orbital_tables.bound_orbital_counts[ABSORBER_INDEX];
    ensure!(
        active_orbital_count > 0
            && active_orbital_count <= orbital_tables.kappa_by_potential.nrows()
            && active_orbital_count <= orbital_tables.valence_counts_by_potential.nrows(),
        "XSPH TDLDA source plan requires active absorber orbital handoffs"
    );

    let mut plus_basis_count = XSPH_TDLDA_DEFAULT_PLUS_BASIS_COUNT;
    let mut minus_basis_count = XSPH_TDLDA_DEFAULT_MINUS_BASIS_COUNT;
    if input.advanced.ibasis == 0 {
        if plus_basis_count > 0 {
            plus_basis_count = 1;
        }
        if minus_basis_count > 0 {
            minus_basis_count = 1;
        }
    }

    let mut channel_count = 2_usize;
    if core_hole.kappa == -1 {
        channel_count = 1;
        minus_basis_count = 0;
    }
    if minus_basis_count > 0 {
        channel_count = 4;
    }

    let orbital_kappa_column = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let orbital_kappas =
        orbital_kappa_column.slice_axis(Axis(0), Slice::from(..active_orbital_count));
    let valence_occupation_column = orbital_tables
        .valence_counts_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let valence_occupations =
        valence_occupation_column.slice_axis(Axis(0), Slice::from(..active_orbital_count));
    let basis = xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
        core_hole_index_1based: usize::try_from(pot.ihole)
            .context("XSPH TDLDA core-hole index is negative")?,
        initial_l: core_hole.angular_momentum,
        plus_basis_count,
        minus_basis_count,
        orbital_kappas,
        valence_occupations,
        basis_selector: input.advanced.ibasis,
    })
    .context("failed to plan XSPH TDLDA channel basis")?;

    let atomic_number = i32::try_from(pot.atomic_numbers[ABSORBER_INDEX])
        .context("XSPH TDLDA absorber atomic number overflow")?;
    ensure!(
        atomic_number > 0,
        "XSPH TDLDA absorber atomic number must be positive, got {atomic_number}"
    );
    let plus_broadening = core_hole_width_ev(atomic_number, pot.ihole)
        .context("failed to prepare TDLDA L3/K width")?
        / FEFF_HARTREE_EV
        / 2.0;
    let minus_broadening = if core_hole.kappa < -1 {
        core_hole_width_ev(atomic_number, pot.ihole - 1)
            .context("failed to prepare TDLDA split-edge width")?
            / FEFF_HARTREE_EV
            / 2.0
    } else {
        plus_broadening
    };

    let mut reference_shifts = Array1::<f64>::zeros(basis.matrix_size);
    let mut row_broadenings = Array1::<f64>::zeros(basis.matrix_size);
    for (row_index, row) in basis.rows.iter().enumerate() {
        if row.initial_kappa > 0 {
            reference_shifts[row_index] = -multipliers.spin_orbit_split;
            row_broadenings[row_index] = minus_broadening;
        } else {
            row_broadenings[row_index] = plus_broadening;
        }
    }

    Ok(XsphTdldaXsectdPlan {
        initial_kappa: core_hole.kappa,
        initial_l: core_hole.angular_momentum,
        channel_count,
        primary_channel_count: XSPH_TDLDA_PRIMARY_CHANNEL_LIMIT.min(basis.matrix_size),
        plus_basis_count: basis.plus_basis_count,
        minus_basis_count: basis.minus_basis_count,
        matrix_size: basis.matrix_size,
        basis,
        spin_orbit_split: multipliers.spin_orbit_split,
        plus_broadening,
        minus_broadening,
        reference_shifts,
        row_broadenings,
        multipliers,
    })
}

fn tdlda_final_l_from_kappa(final_kappa: i32) -> Result<usize> {
    ensure!(
        final_kappa != 0,
        "XSPH TDLDA final-state kappa must be nonzero"
    );
    let final_l = if final_kappa > 0 {
        final_kappa
    } else {
        final_kappa
            .checked_abs()
            .and_then(|value| value.checked_sub(1))
            .context("XSPH TDLDA final-state kappa overflow")?
    };
    usize::try_from(final_l).context("XSPH TDLDA final-state angular momentum is negative")
}

fn tdlda_projector_rows_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    active_len: usize,
    log_step: f64,
    norman_radius: f64,
    radii: ArrayView1<'_, f64>,
    bound_large_components: ArrayView2<'_, f64>,
    bound_small_components: ArrayView2<'_, f64>,
    generated_large_components: Option<ArrayView3<'_, f64>>,
    generated_small_components: Option<ArrayView3<'_, f64>>,
) -> Result<XsphTdldaProjectorRows> {
    ensure!(
        plan.matrix_size == plan.basis.rows.len(),
        "XSPH TDLDA projector plan matrix size {} does not match {} basis rows",
        plan.matrix_size,
        plan.basis.rows.len()
    );
    ensure!(
        radii.len() >= active_len,
        "XSPH TDLDA projector rows require {active_len} radii, got {}",
        radii.len()
    );
    ensure!(
        bound_large_components.nrows() >= active_len,
        "XSPH TDLDA projector rows require bound_large_components shape at least ({active_len}, *), got {:?}",
        bound_large_components.dim()
    );
    ensure!(
        bound_small_components.nrows() >= active_len,
        "XSPH TDLDA projector rows require bound_small_components shape at least ({active_len}, *), got {:?}",
        bound_small_components.dim()
    );
    let generated_components = match (generated_large_components, generated_small_components) {
        (Some(generated_large), Some(generated_small)) => {
            ensure!(
                generated_large.dim() == generated_small.dim(),
                "XSPH TDLDA generated projector large/small shapes must match, got {:?}/{:?}",
                generated_large.dim(),
                generated_small.dim()
            );
            ensure!(
                generated_large.dim().0 >= active_len && generated_large.dim().2 >= 2,
                "XSPH TDLDA generated projectors require shape at least ({active_len}, *, 2), got {:?}",
                generated_large.dim()
            );
            Some((generated_large, generated_small))
        }
        (None, None) => None,
        _ => {
            bail!("XSPH TDLDA generated projector large/small components must be supplied together")
        }
    };

    let mut localized_large = Array2::<f64>::zeros((active_len, plan.matrix_size));
    let mut localized_small = Array2::<f64>::zeros((active_len, plan.matrix_size));
    let mut source_rows = Array1::<bool>::from_elem(plan.matrix_size, false);
    let mut generated_rows = Array1::<bool>::from_elem(plan.matrix_size, false);
    let mut selector_indices = Array1::<usize>::zeros(plan.matrix_size);
    let mut norm_integrals = Array1::<f64>::zeros(plan.matrix_size);
    let mut norm_sqrt = Array1::<f64>::zeros(plan.matrix_size);
    let previous_large = Array2::<f64>::zeros((active_len, 0));
    let previous_small = Array2::<f64>::zeros((active_len, 0));
    let generated_basis_count = generated_components
        .as_ref()
        .map_or(0, |(generated_large, _)| generated_large.dim().1);
    let mut generated_large_projectors =
        Array3::<f64>::zeros((active_len, generated_basis_count, 2));
    let mut generated_small_projectors =
        Array3::<f64>::zeros((active_len, generated_basis_count, 2));
    let mut generated_ready = Array2::<bool>::from_elem((generated_basis_count, 2), false);
    let mut generated_final_l = Array2::<usize>::zeros((generated_basis_count, 2));
    let mut generated_norm_integrals = Array2::<f64>::zeros((generated_basis_count, 2));
    let mut generated_norm_sqrt = Array2::<f64>::zeros((generated_basis_count, 2));

    for (row_index, row) in plan.basis.rows.iter().enumerate() {
        let selector = xsph_tdlda_decode_projector_selector(row.projector_orbital_selector)
            .with_context(|| {
                format!(
                    "failed to decode XSPH TDLDA projector selector {} for row {row_index}",
                    row.projector_orbital_selector
                )
            })?;
        match selector {
            XsphTdldaProjectorSelector::OccupiedOrbital { orbital_index } => {
                ensure!(
                    orbital_index < bound_large_components.ncols()
                        && orbital_index < bound_small_components.ncols(),
                    "XSPH TDLDA projector selector {} requires bound orbital column {}, got large/small shapes {:?}/{:?}",
                    orbital_index + 1,
                    orbital_index,
                    bound_large_components.dim(),
                    bound_small_components.dim()
                );

                let candidate_large = Array1::from_shape_fn(active_len, |radial| {
                    bound_large_components[(radial, orbital_index)]
                });
                let candidate_small = Array1::from_shape_fn(active_len, |radial| {
                    bound_small_components[(radial, orbital_index)]
                });
                let projector = xsph_tdlda_projector_orthogonalization(
                    XsphTdldaProjectorOrthogonalizationInput {
                        active_len,
                        log_step,
                        norman_radius,
                        final_l: tdlda_final_l_from_kappa(row.final_kappa)?,
                        radii,
                        candidate_large: candidate_large.view(),
                        candidate_small: candidate_small.view(),
                        previous_large: previous_large.view(),
                        previous_small: previous_small.view(),
                    },
                )
                .with_context(|| {
                    format!(
                        "failed to assemble XSPH TDLDA projector row {row_index} from orbital selector {}",
                        orbital_index + 1
                    )
                })?;

                for radial in 0..active_len {
                    localized_large[(radial, row_index)] = projector.large[radial];
                    localized_small[(radial, row_index)] = projector.small[radial];
                }
                source_rows[row_index] = true;
                selector_indices[row_index] = orbital_index;
                norm_integrals[row_index] = projector.norm_integral;
                norm_sqrt[row_index] = projector.norm_sqrt;
            }
            XsphTdldaProjectorSelector::GeneratedBasis {
                basis_index,
                positive_final_kappa,
            } => {
                let Some((generated_large_components, generated_small_components)) =
                    generated_components.as_ref()
                else {
                    continue;
                };
                let partner_index = usize::from(positive_final_kappa);
                ensure!(
                    basis_index < generated_basis_count,
                    "XSPH TDLDA generated projector selector requires basis column {}, got shape {:?}",
                    basis_index,
                    generated_large_components.dim()
                );
                let final_l = tdlda_final_l_from_kappa(row.final_kappa)?;
                if generated_ready[(basis_index, partner_index)] {
                    ensure!(
                        generated_final_l[(basis_index, partner_index)] == final_l,
                        "XSPH TDLDA generated projector selector row {row_index} changed final l from {} to {final_l}",
                        generated_final_l[(basis_index, partner_index)]
                    );
                } else {
                    let previous_indices = (0..basis_index)
                        .filter(|previous| {
                            generated_ready[(*previous, partner_index)]
                                && generated_final_l[(*previous, partner_index)] == final_l
                        })
                        .collect::<Vec<_>>();
                    let generated_previous_large = Array2::from_shape_fn(
                        (active_len, previous_indices.len()),
                        |(radial, previous_column)| {
                            generated_large_projectors
                                [(radial, previous_indices[previous_column], partner_index)]
                        },
                    );
                    let generated_previous_small = Array2::from_shape_fn(
                        (active_len, previous_indices.len()),
                        |(radial, previous_column)| {
                            generated_small_projectors
                                [(radial, previous_indices[previous_column], partner_index)]
                        },
                    );
                    let candidate_large = Array1::from_shape_fn(active_len, |radial| {
                        generated_large_components[(radial, basis_index, partner_index)]
                    });
                    let candidate_small = Array1::from_shape_fn(active_len, |radial| {
                        generated_small_components[(radial, basis_index, partner_index)]
                    });
                    let projector = xsph_tdlda_projector_orthogonalization(
                        XsphTdldaProjectorOrthogonalizationInput {
                            active_len,
                            log_step,
                            norman_radius,
                            final_l,
                            radii,
                            candidate_large: candidate_large.view(),
                            candidate_small: candidate_small.view(),
                            previous_large: generated_previous_large.view(),
                            previous_small: generated_previous_small.view(),
                        },
                    )
                    .with_context(|| {
                        format!(
                            "failed to assemble XSPH TDLDA generated projector row {row_index} from basis {basis_index} partner {partner_index}"
                        )
                    })?;

                    for radial in 0..active_len {
                        generated_large_projectors[(radial, basis_index, partner_index)] =
                            projector.large[radial];
                        generated_small_projectors[(radial, basis_index, partner_index)] =
                            projector.small[radial];
                    }
                    generated_ready[(basis_index, partner_index)] = true;
                    generated_final_l[(basis_index, partner_index)] = final_l;
                    generated_norm_integrals[(basis_index, partner_index)] =
                        projector.norm_integral;
                    generated_norm_sqrt[(basis_index, partner_index)] = projector.norm_sqrt;
                }

                for radial in 0..active_len {
                    localized_large[(radial, row_index)] =
                        generated_large_projectors[(radial, basis_index, partner_index)];
                    localized_small[(radial, row_index)] =
                        generated_small_projectors[(radial, basis_index, partner_index)];
                }
                source_rows[row_index] = true;
                generated_rows[row_index] = true;
                selector_indices[row_index] = basis_index;
                norm_integrals[row_index] = generated_norm_integrals[(basis_index, partner_index)];
                norm_sqrt[row_index] = generated_norm_sqrt[(basis_index, partner_index)];
            }
        }
    }
    ensure!(
        source_rows.iter().any(|source_row| *source_row),
        "XSPH TDLDA projector rows require at least one source-backed selector"
    );

    Ok(XsphTdldaProjectorRows {
        localized_large,
        localized_small,
        source_rows,
        generated_rows,
        selector_indices,
        norm_integrals,
        norm_sqrt,
    })
}

fn tdlda_generated_projector_candidates_from_source(
    input: XsphTdldaGeneratedProjectorCandidatesInput<'_>,
) -> Result<(Array3<f64>, Array3<f64>)> {
    ensure!(
        input.generated_basis_count > 0,
        "XSPH TDLDA generated projector candidate generation requires at least one basis"
    );
    ensure!(
        input.active_len > input.generated_target_last_index,
        "XSPH TDLDA generated projector active length {} must cover target index {}",
        input.active_len,
        input.generated_target_last_index
    );
    ensure!(
        input.generated_basis_count <= XSPH_TDLDA_GENERATED_BASIS_ENERGIES_EV.len(),
        "XSPH TDLDA generated basis count {} exceeds FEFF source-backed energy table length {}",
        input.generated_basis_count,
        XSPH_TDLDA_GENERATED_BASIS_ENERGIES_EV.len()
    );

    let mut generated_large =
        Array3::<f64>::zeros((input.active_len, input.generated_basis_count, 2));
    let mut generated_small =
        Array3::<f64>::zeros((input.active_len, input.generated_basis_count, 2));
    let mut ready = Array2::<bool>::from_elem((input.generated_basis_count, 2), false);
    let mut final_kappas = Array2::<i32>::zeros((input.generated_basis_count, 2));
    let mut fermi_cache: Option<Array1<XcpotFermiCache>> = None;

    for row in &input.plan.basis.rows {
        let XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index,
            positive_final_kappa,
        } = xsph_tdlda_decode_projector_selector(row.projector_orbital_selector)?
        else {
            continue;
        };
        let partner_index = usize::from(positive_final_kappa);
        ensure!(
            basis_index < input.generated_basis_count,
            "XSPH TDLDA generated projector selector requires basis column {}, got {}",
            basis_index,
            input.generated_basis_count
        );
        if ready[(basis_index, partner_index)] {
            ensure!(
                final_kappas[(basis_index, partner_index)] == row.final_kappa,
                "XSPH TDLDA generated projector candidate {basis_index}/{partner_index} changed final kappa from {} to {}",
                final_kappas[(basis_index, partner_index)],
                row.final_kappa
            );
            continue;
        }

        let energy_hartree = XSPH_TDLDA_GENERATED_BASIS_ENERGIES_EV[basis_index] / FEFF_HARTREE_EV;
        let xcpot_result = evaluate_xsph_xcpot(
            XcpotInput {
                exchange_selector: input.input.control.ixc,
                lreal: input.input.control.lreal,
                energy: Complex64::new(energy_hartree, 0.0),
                fermi_level: input.fermi_level,
                total_potential: input.total_potential,
                valence_potential: input.valence_potential,
                density: input.electron_density,
                magnetization: input.magnetization,
                valence_density: input.valence_density,
                active_len: input.xcpot_active_len,
                plasmon_selector: input.input.control.i_plsmn,
                many_pole_delta_table: None,
                many_pole_self_energy: input
                    .many_pole_self_energy
                    .map(|poles| poles.as_xcpot_input()),
                fermi_cache: fermi_cache.as_ref().map(|cache| cache.view()),
            },
            input.broadened_table,
        )
        .with_context(|| {
            format!(
                "failed to evaluate XSPH TDLDA generated projector xcpot for basis {}",
                basis_index + 1
            )
        })?;
        if !xcpot_result.fermi_cache.is_empty() {
            fermi_cache = Some(xcpot_result.fermi_cache.clone());
        }

        let momentum_squared = Complex64::new(energy_hartree, 0.0) - xcpot_result.reference_energy;
        let alpha_scaled = momentum_squared * XSPH_FINE_STRUCTURE_ALPHA;
        let wave_number = (2.0 * momentum_squared + alpha_scaled * alpha_scaled).sqrt();
        ensure!(
            wave_number.re.is_finite() && wave_number.im.is_finite(),
            "XSPH TDLDA generated projector wave number must be finite, got {wave_number}"
        );
        let solver_total_potential =
            extend_xcpot_potential(&xcpot_result.total_potential, input.radii.len(), "total")?;
        let solver_valence_potential = if xcpot_result.valence_potential.is_empty() {
            solver_total_potential.clone()
        } else {
            extend_xcpot_potential(
                &xcpot_result.valence_potential,
                input.radii.len(),
                "valence",
            )?
        };
        let solver = FovrgDiracSolverInput {
            exchange_cycle_count: 0,
            target_kappa: row.final_kappa,
            muffin_tin_radius: input.muffin_tin_radius,
            target_last_index: input.generated_target_last_index,
            energy: momentum_squared,
            step: input.radial_dx,
            radii: input.radii,
            exchange_correlation_potential: solver_total_potential.view(),
            valence_exchange_correlation_potential: solver_valence_potential.view(),
            bound_large_components: input.bound_large,
            bound_small_components: input.bound_small,
            bound_large_coefficients: input.bound_large_coefficients,
            bound_small_coefficients: input.bound_small_coefficients,
            electron_counts: input.electron_counts,
            valence_counts: input.valence_counts,
            kappa: input.kappa,
            muffin_tin_large_component: Complex64::new(0.0, 0.0),
            muffin_tin_small_component: Complex64::new(0.0, 0.0),
            atomic_number: input.atomic_number,
            irregular: false,
            c3_scale: 1,
            radial_match_index: input.radial_match_index,
            bound_orbital_count: input.bound_orbital_count,
        };
        let channel = xsph_xsect_regular_channel(XsphXsectRegularChannelInput {
            solver,
            wave_number,
        })
        .with_context(|| {
            format!(
                "failed to solve XSPH TDLDA generated projector basis {} partner {}",
                basis_index + 1,
                partner_index + 1
            )
        })?;
        let candidate_len = input
            .generated_target_last_index
            .checked_add(1)
            .context("XSPH TDLDA generated projector target length overflow")?
            .min(channel.normalized_solution.regular_large.len())
            .min(channel.normalized_solution.regular_small.len());
        let cutoff_len = tdlda_generated_projector_cutoff_len(
            channel.normalized_solution.regular_large.view(),
            candidate_len,
            basis_index,
            input.plan.plus_basis_count,
        );
        for radial in 0..cutoff_len {
            generated_large[(radial, basis_index, partner_index)] =
                channel.normalized_solution.regular_large[radial].re;
            generated_small[(radial, basis_index, partner_index)] =
                channel.normalized_solution.regular_small[radial].re;
        }
        ready[(basis_index, partner_index)] = true;
        final_kappas[(basis_index, partner_index)] = row.final_kappa;
    }

    for selector in input
        .plan
        .basis
        .rows
        .iter()
        .map(|row| xsph_tdlda_decode_projector_selector(row.projector_orbital_selector))
    {
        if let XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index,
            positive_final_kappa,
        } = selector?
        {
            let partner_index = usize::from(positive_final_kappa);
            ensure!(
                ready[(basis_index, partner_index)],
                "XSPH TDLDA generated projector candidate {basis_index}/{partner_index} was not prepared"
            );
        }
    }

    Ok((generated_large, generated_small))
}

fn tdlda_file_projector_candidates_from_source(
    input: XsphTdldaFileProjectorCandidatesInput<'_>,
) -> Result<Option<(Array3<f64>, Array3<f64>)>> {
    ensure!(
        input.generated_basis_count > 0,
        "XSPH TDLDA file projector candidate generation requires at least one basis"
    );
    ensure!(
        input.radii.len() >= input.active_len,
        "XSPH TDLDA file projectors require {active_len} radii, got {}",
        input.radii.len(),
        active_len = input.active_len
    );
    ensure!(
        input.active_len > input.file_target_last_index,
        "XSPH TDLDA file projector active length {} must cover target index {}",
        input.active_len,
        input.file_target_last_index
    );

    let mut orbital_tables = Vec::with_capacity(input.generated_basis_count);
    for basis_index in 0..input.generated_basis_count {
        let path = tdlda_file_projector_orbital_path(input.work_dir, basis_index);
        if !path.is_file() {
            return Ok(None);
        }
        orbital_tables.push(read_tdlda_file_projector_orbital(&path)?);
    }

    let mut generated_large =
        Array3::<f64>::zeros((input.active_len, input.generated_basis_count, 2));
    let generated_small = Array3::<f64>::zeros((input.active_len, input.generated_basis_count, 2));
    let mut ready = Array2::<bool>::from_elem((input.generated_basis_count, 2), false);
    let mut final_kappas = Array2::<i32>::zeros((input.generated_basis_count, 2));
    let interpolation_len = input
        .file_target_last_index
        .checked_add(1)
        .context("XSPH TDLDA file projector target length overflow")?
        .min(input.active_len);

    for row in &input.plan.basis.rows {
        let XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index,
            positive_final_kappa,
        } = xsph_tdlda_decode_projector_selector(row.projector_orbital_selector)?
        else {
            continue;
        };
        let partner_index = usize::from(positive_final_kappa);
        ensure!(
            basis_index < input.generated_basis_count,
            "XSPH TDLDA file projector selector requires basis column {}, got {}",
            basis_index,
            input.generated_basis_count
        );
        if ready[(basis_index, partner_index)] {
            ensure!(
                final_kappas[(basis_index, partner_index)] == row.final_kappa,
                "XSPH TDLDA file projector candidate {basis_index}/{partner_index} changed final kappa from {} to {}",
                final_kappas[(basis_index, partner_index)],
                row.final_kappa
            );
            continue;
        }

        let (source_radii, source_values) = &orbital_tables[basis_index];
        for radial in 0..interpolation_len {
            generated_large[(radial, basis_index, partner_index)] = terp(
                source_radii,
                source_values,
                1,
                input.radii[radial],
            )
            .with_context(|| {
                format!(
                    "failed to interpolate XSPH TDLDA file projector basis {} partner {} at radial row {}",
                    basis_index + 1,
                    partner_index + 1,
                    radial + 1
                )
            })?
            .value;
        }
        ready[(basis_index, partner_index)] = true;
        final_kappas[(basis_index, partner_index)] = row.final_kappa;
    }

    for selector in input
        .plan
        .basis
        .rows
        .iter()
        .map(|row| xsph_tdlda_decode_projector_selector(row.projector_orbital_selector))
    {
        if let XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index,
            positive_final_kappa,
        } = selector?
        {
            let partner_index = usize::from(positive_final_kappa);
            ensure!(
                ready[(basis_index, partner_index)],
                "XSPH TDLDA file projector candidate {basis_index}/{partner_index} was not prepared"
            );
        }
    }

    Ok(Some((generated_large, generated_small)))
}

fn tdlda_file_projector_orbital_path(work_dir: &Path, basis_index: usize) -> PathBuf {
    let file_name = if matches!(basis_index, 1 | 2) {
        "mg.4p.dat"
    } else {
        "mg.3p.dat"
    };
    work_dir.join("Vila").join("Orbs").join(file_name)
}

fn read_tdlda_file_projector_orbital(path: &Path) -> Result<(Vec<f64>, Vec<f64>)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut radii_bohr = Vec::<f64>::new();
    let mut psi_times_radius = Vec::<f64>::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut columns = trimmed.split_whitespace();
        let radius_angstrom = columns
            .next()
            .with_context(|| {
                format!(
                    "{}:{} missing radius column",
                    path.display(),
                    line_index + 1
                )
            })?
            .parse::<f64>()
            .with_context(|| format!("{}:{} has invalid radius", path.display(), line_index + 1))?;
        let orbital_value = columns
            .next()
            .with_context(|| {
                format!(
                    "{}:{} missing orbital column",
                    path.display(),
                    line_index + 1
                )
            })?
            .parse::<f64>()
            .with_context(|| {
                format!(
                    "{}:{} has invalid orbital value",
                    path.display(),
                    line_index + 1
                )
            })?;
        ensure!(
            radius_angstrom.is_finite() && radius_angstrom > 0.0,
            "{}:{} radius must be positive and finite, got {}",
            path.display(),
            line_index + 1,
            radius_angstrom
        );
        ensure!(
            orbital_value.is_finite(),
            "{}:{} orbital value must be finite, got {}",
            path.display(),
            line_index + 1,
            orbital_value
        );
        let radius_bohr = radius_angstrom / FEFF_BOHR_ANGSTROM;
        if let Some(previous) = radii_bohr.last().copied() {
            ensure!(
                radius_bohr > previous,
                "{}:{} radii must be strictly increasing",
                path.display(),
                line_index + 1
            );
        }
        radii_bohr.push(radius_bohr);
        psi_times_radius.push(orbital_value * radius_angstrom);
    }
    ensure!(
        radii_bohr.len() >= 2,
        "{} needs at least two orbital rows for FEFF terp interpolation, got {}",
        path.display(),
        radii_bohr.len()
    );
    Ok((radii_bohr, psi_times_radius))
}

fn tdlda_generated_projector_cutoff_len(
    large: ArrayView1<'_, Complex64>,
    candidate_len: usize,
    basis_index: usize,
    plus_basis_count: usize,
) -> usize {
    let mut cutoff_len = candidate_len;
    let mut remaining_nodes = if basis_index < plus_basis_count {
        basis_index + 2
    } else {
        basis_index - plus_basis_count + 1
    };
    for one_based in 5..=candidate_len {
        let previous = large[one_based - 2].re;
        let current = large[one_based - 1].re;
        if previous * current <= 0.0 {
            cutoff_len = one_based - 1;
            if current != 0.0 {
                remaining_nodes = remaining_nodes.saturating_sub(1);
            }
            if remaining_nodes == 0 {
                break;
            }
        }
    }
    cutoff_len
}

fn tdlda_raw_response_inputs_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    active_len: usize,
    log_step: f64,
    radii: ArrayView1<'_, f64>,
    initial_large: ArrayView1<'_, f64>,
    initial_small: ArrayView1<'_, f64>,
    xray_bessel: ArrayView2<'_, f64>,
    localized_large: ArrayView2<'_, f64>,
    localized_small: ArrayView2<'_, f64>,
    full_large: ArrayView2<'_, f64>,
    full_small: ArrayView2<'_, f64>,
) -> Result<XsphTdldaRawResponseInputs> {
    ensure!(
        active_len >= 4,
        "XSPH TDLDA raw-response inputs require at least four active radial rows, got {active_len}"
    );
    ensure!(
        plan.matrix_size == plan.basis.rows.len(),
        "XSPH TDLDA raw-response input plan matrix size {} does not match {} basis rows",
        plan.matrix_size,
        plan.basis.rows.len()
    );
    ensure!(
        radii.len() >= active_len
            && initial_large.len() >= active_len
            && initial_small.len() >= active_len,
        "XSPH TDLDA raw-response radial inputs cannot supply active length {} (radii {}, initial large {}, initial small {})",
        active_len,
        radii.len(),
        initial_large.len(),
        initial_small.len()
    );
    ensure!(
        xray_bessel.nrows() >= 3 && xray_bessel.ncols() >= active_len,
        "XSPH TDLDA raw-response xray_bessel shape {:?} cannot supply (3, {active_len})",
        xray_bessel.dim()
    );
    for (name, matrix) in [
        ("localized_large", localized_large),
        ("localized_small", localized_small),
        ("full_large", full_large),
        ("full_small", full_small),
    ] {
        ensure!(
            matrix.nrows() >= active_len && matrix.ncols() >= plan.matrix_size,
            "XSPH TDLDA raw-response {name} shape {:?} cannot supply ({active_len}, {})",
            matrix.dim(),
            plan.matrix_size
        );
    }

    let active_radii = radii.iter().take(active_len).copied().collect::<Vec<_>>();
    let mut overlaps = Array1::<f64>::zeros(plan.matrix_size);
    let mut localized_dipoles = Array1::<f64>::zeros(plan.matrix_size);
    let mut full_dipoles = Array1::<f64>::zeros(plan.matrix_size);

    for (row_index, row) in plan.basis.rows.iter().enumerate() {
        let final_l = tdlda_final_l_from_kappa(row.final_kappa)?;
        let overlap_samples = (0..active_len)
            .map(|radial| {
                localized_large[(radial, row_index)] * full_large[(radial, row_index)]
                    + localized_small[(radial, row_index)] * full_small[(radial, row_index)]
            })
            .collect::<Vec<_>>();
        let overlap = somm2(
            &active_radii,
            &overlap_samples,
            log_step,
            2.0 * final_l as f64 + 2.0,
            active_radii[active_len - 1],
            0,
        )
        .with_context(|| {
            format!("failed to assemble XSPH TDLDA raw-response overlap for row {row_index}")
        })?;
        ensure!(
            overlap.is_finite(),
            "XSPH TDLDA raw-response overlap for row {row_index} must be finite, got {overlap}"
        );
        overlaps[row_index] = overlap;

        let localized_final_large = Array1::from_shape_fn(active_len, |radial| {
            Complex64::new(localized_large[(radial, row_index)], 0.0)
        });
        let localized_final_small = Array1::from_shape_fn(active_len, |radial| {
            Complex64::new(localized_small[(radial, row_index)], 0.0)
        });
        let full_final_large = Array1::from_shape_fn(active_len, |radial| {
            Complex64::new(full_large[(radial, row_index)], 0.0)
        });
        let full_final_small = Array1::from_shape_fn(active_len, |radial| {
            Complex64::new(full_small[(radial, row_index)], 0.0)
        });
        let localized = xsph_radial_integral(XsphRadialIntegralInput {
            mode: XsphRadialIntegralMode::RelativisticMatrixElement,
            multipole: XsphTransitionMultipole::ElectricDipole,
            initial_kappa: row.initial_kappa,
            final_kappa: row.final_kappa,
            initial_large,
            initial_small,
            final_large_regular: localized_final_large.view(),
            final_small_regular: localized_final_small.view(),
            xray_bessel,
            radii,
            log_step,
            active_len,
        })
        .with_context(|| {
            format!("failed to assemble XSPH TDLDA localized dipole for row {row_index}")
        })?;
        let full = xsph_radial_integral(XsphRadialIntegralInput {
            mode: XsphRadialIntegralMode::RelativisticMatrixElement,
            multipole: XsphTransitionMultipole::ElectricDipole,
            initial_kappa: row.initial_kappa,
            final_kappa: row.final_kappa,
            initial_large,
            initial_small,
            final_large_regular: full_final_large.view(),
            final_small_regular: full_final_small.view(),
            xray_bessel,
            radii,
            log_step,
            active_len,
        })
        .with_context(|| {
            format!("failed to assemble XSPH TDLDA full dipole for row {row_index}")
        })?;
        ensure!(
            localized.value.re.is_finite() && full.value.re.is_finite(),
            "XSPH TDLDA dipoles for row {row_index} must be finite, got {:?}/{:?}",
            localized.value,
            full.value
        );
        // FEFF getchi0 uses dimag(xirf), then converts the reduced matrix
        // element to this (j,m)->(j',m') row with the dipole 3-j factor.
        let polarization_m2 = row.final_m2 - row.initial_m2;
        let angular = wigner_3j(
            row.final_j2,
            2,
            row.initial_j2,
            -row.final_m2,
            polarization_m2,
            2,
        )?;
        let phase = if ((row.final_j2 - row.final_m2) / 2) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        localized_dipoles[row_index] = localized.value.im * angular * phase;
        full_dipoles[row_index] = full.value.im * angular * phase;
    }

    Ok(XsphTdldaRawResponseInputs {
        overlaps,
        localized_dipoles,
        full_dipoles,
    })
}

fn tdlda_weight_response_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    raw_imaginary_response: ArrayView3<'_, f64>,
) -> Result<XsphTdldaWeightedResponse> {
    let initial_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_kappa));
    let final_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.final_kappa));
    xsph_tdlda_weight_response(XsphTdldaWeightedResponseInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        matrix_size: plan.matrix_size,
        initial_kappas: initial_kappas.view(),
        final_kappas: final_kappas.view(),
        raw_imaginary_response,
        channel_multipliers: plan.multipliers.channel_multipliers.view(),
    })
    .context("failed to apply XSPH TDLDA PMBSE channel multipliers")
}

fn tdlda_energy_rows_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    input: &XsphInput,
    reference_energy: ArrayView1<'_, Complex64>,
    edge_energy: f64,
    chemical_potential: f64,
) -> Result<XsphTdldaEnergyRows> {
    xsph_tdlda_energy_rows(XsphTdldaEnergyRowsInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        energy_hartree: plan.multipliers.energy_hartree.view(),
        reference_energy,
        edge_energy,
        chemical_potential,
        spin_orbit_split: plan.spin_orbit_split,
        ipmbse: normal_xsect_effective_advanced(input).ipmbse,
    })
    .context("failed to prepare XSPH TDLDA xsectd energy rows")
}

fn tdlda_row_wave_numbers_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    energy_hartree: f64,
    reference_energy: Complex64,
) -> Result<XsphTdldaRowWaveNumbers> {
    xsph_tdlda_row_wave_numbers(XsphTdldaRowWaveNumbersInput {
        matrix_size: plan.matrix_size,
        energy_hartree,
        reference_energy,
        reference_shifts: plan.reference_shifts.view(),
    })
    .context("failed to prepare XSPH TDLDA getchi0 row wave numbers")
}

fn tdlda_raw_response_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    energy_hartree: f64,
    reference_energy: Complex64,
    edge_energy: f64,
    overlaps: ArrayView1<'_, f64>,
    localized_dipoles: ArrayView1<'_, f64>,
    full_dipoles: ArrayView1<'_, f64>,
) -> Result<XsphTdldaRawResponse> {
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(plan, energy_hartree, reference_energy)?;
    xsph_tdlda_raw_response(XsphTdldaRawResponseInput {
        matrix_size: plan.matrix_size,
        plus_basis_count: plan.plus_basis_count,
        minus_basis_count: plan.minus_basis_count,
        initial_l: plan.initial_l,
        energy_hartree,
        edge_energy,
        reference_shifts: plan.reference_shifts.view(),
        row_wave_numbers: row_wave_numbers.row_wave_numbers.view(),
        overlaps,
        localized_dipoles,
        full_dipoles,
    })
    .context("failed to assemble XSPH TDLDA getchi0 raw response")
}

#[cfg(test)]
fn tdlda_projected_kernel_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    projected_kernel: ArrayView2<'_, Complex64>,
) -> Result<XsphTdldaProjectedKernel> {
    xsph_tdlda_projected_kernel(XsphTdldaProjectedKernelInput {
        matrix_size: plan.matrix_size,
        plus_basis_count: plan.plus_basis_count,
        minus_basis_count: plan.minus_basis_count,
        initial_l: plan.initial_l,
        projected_kernel,
    })
    .context("failed to fold XSPH TDLDA getchi0 projected kernel")
}

fn tdlda_direct_kernel_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    row_wave_numbers: &XsphTdldaRowWaveNumbers,
    energy_hartree: f64,
    edge_energy: f64,
    separation_function: f64,
    active_len: usize,
    radii: ArrayView1<'_, f64>,
    core_hole_potential: ArrayView1<'_, f64>,
    localized_large: ArrayView2<'_, f64>,
    localized_small: ArrayView2<'_, f64>,
    full_large: ArrayView2<'_, f64>,
    full_small: ArrayView2<'_, f64>,
) -> Result<XsphTdldaDirectKernel> {
    xsph_tdlda_direct_kernel(XsphTdldaDirectKernelInput {
        matrix_size: plan.matrix_size,
        plus_basis_count: plan.plus_basis_count,
        minus_basis_count: plan.minus_basis_count,
        initial_l: plan.initial_l,
        active_len,
        energy_hartree,
        edge_energy,
        separation_function,
        reference_shifts: plan.reference_shifts.view(),
        momentum_squared: row_wave_numbers.momentum_squared.view(),
        radii,
        core_hole_potential,
        localized_large,
        localized_small,
        full_large,
        full_small,
    })
    .context("failed to assemble XSPH TDLDA getchi0 direct kernel")
}

fn tdlda_coulomb_fields_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    active_len: usize,
    source_len: usize,
    coefficient_count: usize,
    step: f64,
    multipole: usize,
    radii: ArrayView1<'_, f64>,
    orbital_large: ArrayView2<'_, f64>,
    orbital_small: ArrayView2<'_, f64>,
    orbital_large_coefficients: ArrayView2<'_, f64>,
    orbital_small_coefficients: ArrayView2<'_, f64>,
    orbital_powers: ArrayView1<'_, f64>,
    orbital_lengths: ArrayView1<'_, usize>,
    target_large: ArrayView2<'_, Complex64>,
    target_small: ArrayView2<'_, Complex64>,
    target_large_coefficients: ArrayView2<'_, Complex64>,
    target_small_coefficients: ArrayView2<'_, Complex64>,
    target_powers: ArrayView1<'_, f64>,
) -> Result<XsphTdldaCoulombFields> {
    ensure!(
        plan.matrix_size > 0,
        "XSPH TDLDA Coulomb field plan requires at least one basis row"
    );
    let core_indices = plan
        .basis
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let index_1based =
                usize::try_from(row.core_orbital_index_1based).with_context(|| {
                    format!(
                        "XSPH TDLDA basis row {row_index} has a negative core orbital index {}",
                        row.core_orbital_index_1based
                    )
                })?;
            index_1based.checked_sub(1).with_context(|| {
                format!("XSPH TDLDA basis row {row_index} has a zero core orbital index")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        core_indices.len() == plan.matrix_size,
        "XSPH TDLDA Coulomb field plan matrix size {} does not match {} basis rows",
        plan.matrix_size,
        core_indices.len()
    );
    let required_orbital_count = core_indices.iter().copied().max().unwrap_or(0) + 1;
    ensure!(
        radii.len() >= active_len,
        "XSPH TDLDA Coulomb fields require {active_len} radii, got {}",
        radii.len()
    );
    ensure!(
        orbital_large.nrows() >= active_len && orbital_large.ncols() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require orbital_large shape at least ({active_len}, {required_orbital_count}), got {:?}",
        orbital_large.dim()
    );
    ensure!(
        orbital_small.nrows() >= active_len && orbital_small.ncols() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require orbital_small shape at least ({active_len}, {required_orbital_count}), got {:?}",
        orbital_small.dim()
    );
    ensure!(
        orbital_large_coefficients.nrows() >= coefficient_count
            && orbital_large_coefficients.ncols() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require orbital_large_coefficients shape at least ({coefficient_count}, {required_orbital_count}), got {:?}",
        orbital_large_coefficients.dim()
    );
    ensure!(
        orbital_small_coefficients.nrows() >= coefficient_count
            && orbital_small_coefficients.ncols() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require orbital_small_coefficients shape at least ({coefficient_count}, {required_orbital_count}), got {:?}",
        orbital_small_coefficients.dim()
    );
    ensure!(
        orbital_powers.len() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require {required_orbital_count} orbital powers, got {}",
        orbital_powers.len()
    );
    ensure!(
        orbital_lengths.len() >= required_orbital_count,
        "XSPH TDLDA Coulomb fields require {required_orbital_count} orbital lengths, got {}",
        orbital_lengths.len()
    );
    ensure!(
        target_large.nrows() >= active_len && target_large.ncols() >= plan.matrix_size,
        "XSPH TDLDA Coulomb fields require target_large shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        target_large.dim()
    );
    ensure!(
        target_small.nrows() >= active_len && target_small.ncols() >= plan.matrix_size,
        "XSPH TDLDA Coulomb fields require target_small shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        target_small.dim()
    );
    ensure!(
        target_large_coefficients.nrows() >= coefficient_count
            && target_large_coefficients.ncols() >= plan.matrix_size,
        "XSPH TDLDA Coulomb fields require target_large_coefficients shape at least ({coefficient_count}, {}), got {:?}",
        plan.matrix_size,
        target_large_coefficients.dim()
    );
    ensure!(
        target_small_coefficients.nrows() >= coefficient_count
            && target_small_coefficients.ncols() >= plan.matrix_size,
        "XSPH TDLDA Coulomb fields require target_small_coefficients shape at least ({coefficient_count}, {}), got {:?}",
        plan.matrix_size,
        target_small_coefficients.dim()
    );
    ensure!(
        target_powers.len() >= plan.matrix_size,
        "XSPH TDLDA Coulomb fields require {} target powers, got {}",
        plan.matrix_size,
        target_powers.len()
    );

    let core_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        orbital_large[(radial, core_indices[row])]
    });
    let core_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        orbital_small[(radial, core_indices[row])]
    });
    let core_large_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| orbital_large_coefficients[(coefficient, core_indices[row])],
    );
    let core_small_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| orbital_small_coefficients[(coefficient, core_indices[row])],
    );
    let core_powers = Array1::from_iter(core_indices.iter().map(|&index| orbital_powers[index]));
    let core_lengths = Array1::from_iter(core_indices.iter().map(|&index| orbital_lengths[index]));

    xsph_tdlda_coulomb_fields(XsphTdldaCoulombFieldsInput {
        matrix_size: plan.matrix_size,
        active_len,
        source_len,
        coefficient_count,
        step,
        multipole,
        radii,
        core_large: core_large.view(),
        core_small: core_small.view(),
        core_large_coefficients: core_large_coefficients.view(),
        core_small_coefficients: core_small_coefficients.view(),
        core_powers: core_powers.view(),
        core_lengths: core_lengths.view(),
        target_large,
        target_small,
        target_large_coefficients,
        target_small_coefficients,
        target_powers,
    })
    .context("failed to assemble XSPH TDLDA yzktd Coulomb fields")
}

fn tdlda_nonlocal_exchange_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    row_wave_numbers: &XsphTdldaRowWaveNumbers,
    active_len: usize,
    source_len: usize,
    coefficient_count: usize,
    step: f64,
    multipole: usize,
    direct_scale: f64,
    radii: ArrayView1<'_, f64>,
    orbital_large: ArrayView2<'_, f64>,
    orbital_small: ArrayView2<'_, f64>,
    orbital_large_coefficients: ArrayView2<'_, f64>,
    orbital_small_coefficients: ArrayView2<'_, f64>,
    orbital_powers: ArrayView1<'_, f64>,
    orbital_lengths: ArrayView1<'_, usize>,
    localized_large: ArrayView2<'_, Complex64>,
    localized_small: ArrayView2<'_, Complex64>,
    full_large: ArrayView2<'_, Complex64>,
    full_small: ArrayView2<'_, Complex64>,
) -> Result<XsphTdldaRadialKernel> {
    ensure!(
        plan.matrix_size > 0,
        "XSPH TDLDA nonlocal exchange plan requires at least one basis row"
    );
    let core_indices = plan
        .basis
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let index_1based =
                usize::try_from(row.core_orbital_index_1based).with_context(|| {
                    format!(
                        "XSPH TDLDA basis row {row_index} has a negative core orbital index {}",
                        row.core_orbital_index_1based
                    )
                })?;
            index_1based.checked_sub(1).with_context(|| {
                format!("XSPH TDLDA basis row {row_index} has a zero core orbital index")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        core_indices.len() == plan.matrix_size,
        "XSPH TDLDA nonlocal exchange plan matrix size {} does not match {} basis rows",
        plan.matrix_size,
        core_indices.len()
    );
    let required_orbital_count = core_indices.iter().copied().max().unwrap_or(0) + 1;
    ensure!(
        radii.len() >= active_len,
        "XSPH TDLDA nonlocal exchange requires {active_len} radii, got {}",
        radii.len()
    );
    ensure!(
        orbital_large.nrows() >= active_len && orbital_large.ncols() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires orbital_large shape at least ({active_len}, {required_orbital_count}), got {:?}",
        orbital_large.dim()
    );
    ensure!(
        orbital_small.nrows() >= active_len && orbital_small.ncols() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires orbital_small shape at least ({active_len}, {required_orbital_count}), got {:?}",
        orbital_small.dim()
    );
    ensure!(
        orbital_large_coefficients.nrows() >= coefficient_count
            && orbital_large_coefficients.ncols() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires orbital_large_coefficients shape at least ({coefficient_count}, {required_orbital_count}), got {:?}",
        orbital_large_coefficients.dim()
    );
    ensure!(
        orbital_small_coefficients.nrows() >= coefficient_count
            && orbital_small_coefficients.ncols() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires orbital_small_coefficients shape at least ({coefficient_count}, {required_orbital_count}), got {:?}",
        orbital_small_coefficients.dim()
    );
    ensure!(
        orbital_powers.len() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires {required_orbital_count} orbital powers, got {}",
        orbital_powers.len()
    );
    ensure!(
        orbital_lengths.len() >= required_orbital_count,
        "XSPH TDLDA nonlocal exchange requires {required_orbital_count} orbital lengths, got {}",
        orbital_lengths.len()
    );
    ensure!(
        localized_large.nrows() >= active_len && localized_large.ncols() >= plan.matrix_size,
        "XSPH TDLDA nonlocal exchange requires localized_large shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        localized_large.dim()
    );
    ensure!(
        localized_small.nrows() >= active_len && localized_small.ncols() >= plan.matrix_size,
        "XSPH TDLDA nonlocal exchange requires localized_small shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        localized_small.dim()
    );
    ensure!(
        full_large.nrows() >= active_len && full_large.ncols() >= plan.matrix_size,
        "XSPH TDLDA nonlocal exchange requires full_large shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        full_large.dim()
    );
    ensure!(
        full_small.nrows() >= active_len && full_small.ncols() >= plan.matrix_size,
        "XSPH TDLDA nonlocal exchange requires full_small shape at least ({active_len}, {}), got {:?}",
        plan.matrix_size,
        full_small.dim()
    );

    let core_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        orbital_large[(radial, core_indices[row])]
    });
    let core_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        orbital_small[(radial, core_indices[row])]
    });
    let core_large_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| orbital_large_coefficients[(coefficient, core_indices[row])],
    );
    let core_small_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| orbital_small_coefficients[(coefficient, core_indices[row])],
    );
    let core_powers = Array1::from_iter(core_indices.iter().map(|&index| orbital_powers[index]));
    let core_lengths = Array1::from_iter(core_indices.iter().map(|&index| orbital_lengths[index]));
    let initial_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_kappa));

    xsph_tdlda_nonlocal_exchange_integrals(XsphTdldaNonlocalExchangeInput {
        matrix_size: plan.matrix_size,
        active_len,
        source_len,
        coefficient_count,
        step,
        multipole,
        direct_scale,
        positive_momentum_rows: row_wave_numbers.positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        radii,
        core_large: core_large.view(),
        core_small: core_small.view(),
        core_large_coefficients: core_large_coefficients.view(),
        core_small_coefficients: core_small_coefficients.view(),
        core_powers: core_powers.view(),
        core_lengths: core_lengths.view(),
        localized_large,
        localized_small,
        full_large,
        full_small,
    })
    .context("failed to assemble XSPH TDLDA getchi0 nonlocal exchange integrals")
}

fn tdlda_radial_kernel_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    row_wave_numbers: &XsphTdldaRowWaveNumbers,
    exchange_correlation_selector: i32,
    direct_scale: f64,
    active_len: usize,
    radii: ArrayView1<'_, f64>,
    exchange_correlation_same_edge: ArrayView1<'_, f64>,
    exchange_correlation_real: ArrayView1<'_, f64>,
    exchange_correlation_imaginary: ArrayView1<'_, f64>,
    response_large: ArrayView2<'_, Complex64>,
    response_small: ArrayView2<'_, Complex64>,
    localized_large: ArrayView2<'_, Complex64>,
    localized_small: ArrayView2<'_, Complex64>,
    full_large: ArrayView2<'_, Complex64>,
    full_small: ArrayView2<'_, Complex64>,
    coulomb_fields: ArrayView2<'_, Complex64>,
) -> Result<XsphTdldaRadialKernel> {
    let initial_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_kappa));

    xsph_tdlda_radial_kernel_integrals(XsphTdldaRadialKernelInput {
        matrix_size: plan.matrix_size,
        active_len,
        positive_momentum_rows: row_wave_numbers.positive_momentum_rows.view(),
        initial_kappas: initial_kappas.view(),
        exchange_correlation_selector,
        direct_scale,
        radii,
        exchange_correlation_same_edge,
        exchange_correlation_real,
        exchange_correlation_imaginary,
        response_large,
        response_small,
        localized_large,
        localized_small,
        full_large,
        full_small,
        coulomb_fields,
    })
    .context("failed to assemble XSPH TDLDA getchi0 radial kernel integrals")
}

fn tdlda_angular_kernel_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    positive_momentum_rows: ArrayView1<'_, bool>,
    radial_integrals: ArrayView2<'_, Complex64>,
    projected_radial_integrals: ArrayView2<'_, Complex64>,
    nonlocal_radial_integrals: Option<ArrayView2<'_, Complex64>>,
    nonlocal_projected_radial_integrals: Option<ArrayView2<'_, Complex64>>,
) -> Result<XsphTdldaAngularKernel> {
    let initial_j2 = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_j2));
    let initial_m2 = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_m2));
    let initial_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_kappa));
    let final_j2 = Array1::from_iter(plan.basis.rows.iter().map(|row| row.final_j2));
    let final_m2 = Array1::from_iter(plan.basis.rows.iter().map(|row| row.final_m2));

    xsph_tdlda_angular_kernel(XsphTdldaAngularKernelInput {
        matrix_size: plan.matrix_size,
        initial_j2: initial_j2.view(),
        initial_m2: initial_m2.view(),
        initial_kappas: initial_kappas.view(),
        final_j2: final_j2.view(),
        final_m2: final_m2.view(),
        positive_momentum_rows,
        radial_integrals,
        projected_radial_integrals,
        nonlocal_radial_integrals,
        nonlocal_projected_radial_integrals,
    })
    .context("failed to assemble XSPH TDLDA getchi0 angular kernel")
}

fn tdlda_getchi0_kernel_from_source_plan(
    plan: &XsphTdldaXsectdPlan,
    row_wave_numbers: &XsphTdldaRowWaveNumbers,
    exchange_correlation_selector: i32,
    direct_scale: f64,
    energy_hartree: f64,
    edge_energy: f64,
    separation_function: f64,
    active_len: usize,
    radii: ArrayView1<'_, f64>,
    core_hole_potential: ArrayView1<'_, f64>,
    exchange_correlation_same_edge: ArrayView1<'_, f64>,
    exchange_correlation_real: ArrayView1<'_, f64>,
    exchange_correlation_imaginary: ArrayView1<'_, f64>,
    direct_localized_large: ArrayView2<'_, f64>,
    direct_localized_small: ArrayView2<'_, f64>,
    direct_full_large: ArrayView2<'_, f64>,
    direct_full_small: ArrayView2<'_, f64>,
    response_large: ArrayView2<'_, Complex64>,
    response_small: ArrayView2<'_, Complex64>,
    localized_large: ArrayView2<'_, Complex64>,
    localized_small: ArrayView2<'_, Complex64>,
    full_large: ArrayView2<'_, Complex64>,
    full_small: ArrayView2<'_, Complex64>,
    coulomb_fields: ArrayView2<'_, Complex64>,
    nonlocal_radial_integrals: Option<ArrayView2<'_, Complex64>>,
    nonlocal_projected_radial_integrals: Option<ArrayView2<'_, Complex64>>,
) -> Result<XsphTdldaGetchi0Kernel> {
    let direct = tdlda_direct_kernel_from_source_plan(
        plan,
        row_wave_numbers,
        energy_hartree,
        edge_energy,
        separation_function,
        active_len,
        radii,
        core_hole_potential,
        direct_localized_large,
        direct_localized_small,
        direct_full_large,
        direct_full_small,
    )?;
    let radial = tdlda_radial_kernel_from_source_plan(
        plan,
        row_wave_numbers,
        exchange_correlation_selector,
        direct_scale,
        active_len,
        radii,
        exchange_correlation_same_edge,
        exchange_correlation_real,
        exchange_correlation_imaginary,
        response_large,
        response_small,
        localized_large,
        localized_small,
        full_large,
        full_small,
        coulomb_fields,
    )?;
    let angular = tdlda_angular_kernel_from_source_plan(
        plan,
        row_wave_numbers.positive_momentum_rows.view(),
        radial.radial_integrals.view(),
        radial.projected_radial_integrals.view(),
        nonlocal_radial_integrals,
        nonlocal_projected_radial_integrals,
    )?;
    let kernel = &direct.kernel + &angular.kernel;
    let projected_kernel = &direct.projected_kernel + &angular.projected_kernel;

    Ok(XsphTdldaGetchi0Kernel {
        kernel,
        projected_kernel,
        direct,
        radial,
        angular,
    })
}

fn tdlda_xsedge_dat_from_raw_source_components(
    plan: &XsphTdldaXsectdPlan,
    energy_rows: &XsphTdldaEnergyRows,
    raw_imaginary_response: ArrayView3<'_, f64>,
    localized_dipole_matrix: ArrayView2<'_, f64>,
    full_dipole_matrix: ArrayView2<'_, f64>,
    kernel: ArrayView3<'_, Complex64>,
    projected_kernel: ArrayView3<'_, Complex64>,
    edge_energy: f64,
    chemical_potential: f64,
) -> Result<XsedgeDatData> {
    let initial_kappas = Array1::from_iter(plan.basis.rows.iter().map(|row| row.initial_kappa));
    let weighted = tdlda_weight_response_from_source_plan(plan, raw_imaginary_response)?;
    let conditioned = xsph_tdlda_condition_response(XsphTdldaResponseConditioningInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        matrix_size: plan.matrix_size,
        energy_hartree: plan.multipliers.energy_hartree.view(),
        chemical_potential,
        edge_energy,
        reference_shifts: plan.reference_shifts.view(),
        row_broadenings: plan.row_broadenings.view(),
        imaginary_response: weighted.imaginary_response.view(),
    })
    .context("failed to condition XSPH TDLDA response")?;
    let screened = xsph_tdlda_screened_dipoles(XsphTdldaScreenedDipoleInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        matrix_size: plan.matrix_size,
        response: conditioned.response.view(),
        kernel,
        dipole_matrix: localized_dipole_matrix,
    })
    .context("failed to solve XSPH TDLDA screened dipoles")?;
    let spectra = xsph_tdlda_channel_spectra(XsphTdldaChannelSpectraInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        matrix_size: plan.matrix_size,
        primary_channel_count: plan.primary_channel_count,
        channel_count: plan.channel_count,
        photon_energy: energy_rows.photon_energy.view(),
        plus_wave_number: energy_rows.plus_wave_number.view(),
        minus_wave_number: energy_rows.minus_wave_number.view(),
        initial_kappas: initial_kappas.view(),
        dipole_matrix: full_dipole_matrix,
        response: conditioned.response.view(),
        projected_kernel,
        screened_dipoles: screened.screened_dipoles.view(),
    })
    .context("failed to accumulate XSPH TDLDA channel spectra")?;
    let broadened = xsph_tdlda_broaden_channel_spectra(XsphTdldaChannelBroadeningInput {
        energy_count: plan.multipliers.energy_hartree.len(),
        channel_count: plan.channel_count,
        energy_hartree: plan.multipliers.energy_hartree.view(),
        edge_energy,
        spin_orbit_split: plan.spin_orbit_split,
        plus_broadening: plan.plus_broadening,
        minus_broadening: plan.minus_broadening,
        single_particle_channels: spectra.single_particle_channels.view(),
        screened_channels: spectra.screened_channels.view(),
    })
    .context("failed to broaden XSPH TDLDA channel spectra")?;

    tdlda_xsedge_dat_from_source_components(plan.channel_count, &broadened, &plan.multipliers)
}

#[cfg(test)]
fn tdlda_pmbse_channel_multipliers_from_source(
    work_dir: &Path,
    initial_kappa: i32,
    energy_capacity: usize,
) -> Result<Option<XsphTdldaChannelMultipliers>> {
    let caches = XsphCachePaths::new(work_dir);
    tdlda_pmbse_channel_multipliers_from_caches(&caches, initial_kappa, energy_capacity)
}

fn tdlda_pmbse_channel_multipliers_from_caches(
    caches: &XsphCachePaths,
    initial_kappa: i32,
    energy_capacity: usize,
) -> Result<Option<XsphTdldaChannelMultipliers>> {
    if !caches.listedges_pmbse.is_file() {
        return Ok(None);
    }

    let entries = tdlda_pmbse_channel_entries(&caches.listedges_pmbse)?;
    let split_edges = initial_kappa < -1;
    let dominant_plus = tdlda_read_pmbse_xmu(caches, &entries, 0, "dominant plus")?;
    let split_plus = if split_edges {
        Some(tdlda_read_pmbse_xmu(caches, &entries, 1, "split plus")?)
    } else {
        None
    };
    let dominant_minus = if split_edges {
        Some(tdlda_read_pmbse_xmu(caches, &entries, 2, "dominant minus")?)
    } else {
        None
    };
    let split_minus = if split_edges {
        Some(tdlda_read_pmbse_xmu(caches, &entries, 3, "split minus")?)
    } else {
        None
    };

    let multipliers = xsph_tdlda_channel_multipliers(XsphTdldaChannelMultipliersInput {
        initial_kappa,
        energy_capacity,
        dominant_plus: tdlda_xmu_channel_input(&dominant_plus),
        split_plus: split_plus.as_ref().map(tdlda_xmu_channel_input),
        dominant_minus: dominant_minus.as_ref().map(tdlda_xmu_channel_input),
        split_minus: split_minus.as_ref().map(tdlda_xmu_channel_input),
    })
    .context("failed to merge PMBSE xmu.dat channel multipliers")?;

    Ok(Some(multipliers))
}

#[cfg(test)]
fn write_tdlda_xsedge_dat_from_source_components(
    work_dir: &Path,
    channel_count: usize,
    spectra: &XsphTdldaBroadenedChannelSpectra,
    multipliers: &XsphTdldaChannelMultipliers,
) -> Result<XsedgeDatData> {
    let caches = XsphCachePaths::new(work_dir);
    let data = tdlda_xsedge_dat_from_source_components(channel_count, spectra, multipliers)?;
    write_xsedge_dat(&caches.xsedge_dat, &data)
        .with_context(|| format!("failed to write {}", caches.xsedge_dat.display()))?;
    Ok(data)
}

fn tdlda_xsedge_dat_from_source_components(
    channel_count: usize,
    spectra: &XsphTdldaBroadenedChannelSpectra,
    multipliers: &XsphTdldaChannelMultipliers,
) -> Result<XsedgeDatData> {
    let rows = xsph_tdlda_xsedge_rows(XsphTdldaXsedgeRowsInput {
        energy_count: multipliers.energy_hartree.len(),
        channel_count,
        energy_hartree: multipliers.energy_hartree.view(),
        single_particle_channels: spectra.single_particle_channels.view(),
        screened_channels: spectra.screened_channels.view(),
        channel_multipliers: multipliers.channel_multipliers.view(),
    })
    .context("failed to assemble TDLDA xsedge.dat rows")?;

    xsedge_dat_from_tdlda_rows(XsedgeDatFromTdldaRowsInput {
        rows: &rows,
        channel_count,
    })
    .context("failed to build TDLDA xsedge.dat table")
}

fn tdlda_pmbse_channel_entries(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect())
}

fn tdlda_read_pmbse_xmu(
    caches: &XsphCachePaths,
    entries: &[String],
    index: usize,
    name: &'static str,
) -> Result<XmuDatData> {
    let entry = entries
        .get(index)
        .with_context(|| format!("PMBSE listedges.pmbse is missing {name} channel entry"))?;
    let path = tdlda_pmbse_xmu_path(caches, entry);
    read_xmu_dat(&path)
        .with_context(|| format!("failed to read PMBSE {name} channel {}", path.display()))
}

fn tdlda_pmbse_xmu_path(caches: &XsphCachePaths, entry: &str) -> PathBuf {
    let path = Path::new(entry);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        caches.work_dir.join(path)
    };
    if path.file_name().is_some_and(|name| name == "xmu.dat") {
        path
    } else {
        path.join("xmu.dat")
    }
}

fn tdlda_xmu_channel_input(data: &XmuDatData) -> XsphTdldaXmuChannelInput<'_> {
    XsphTdldaXmuChannelInput {
        point_count: data.point_count(),
        photon_energy_ev: data.photon_energy_ev.view(),
        relative_energy_ev: data.relative_energy_ev.view(),
        wave_number: data.wave_number.view(),
        background: data.mu0.view(),
        fine_structure: data.chi.view(),
    }
}

fn normal_xsect_effective_advanced(input: &XsphInput) -> XsphAdvanced {
    let mut advanced = input.advanced;
    if advanced.ipmbse <= 0 {
        advanced.itdlda = 0;
    }
    if advanced.izstd > 0 && advanced.itdlda > 0 {
        // FEFF XSPH/xsphsub.f90 ignores PMBSE cards when the TDLDA-style
        // positive-standard branch is already selected, then calls ordinary
        // XSPH/xsect.f90 instead of TDLDA/xsectd.f90.
        advanced.itdlda = 0;
        advanced.ipmbse = 0;
        advanced.nonlocal = 0;
        advanced.ibasis = 0;
    }
    advanced
}

fn normal_xsect_positive_izstd_transitions_supported(
    advanced: XsphAdvanced,
    transitions: &[XsphXsectTransition],
) -> bool {
    advanced.izstd <= 0
        || transitions.iter().all(|transition| {
            matches!(
                transition.multipole,
                XsphTransitionMultipole::ElectricDipole
                    | XsphTransitionMultipole::ElectricQuadrupole
            )
        })
}

fn xsect_spin_state_supported(spin_count: usize, controls: XsphXsectAngularControls) -> bool {
    match spin_count {
        1 => true,
        2 => controls.spin.abs() == 1,
        _ => false,
    }
}

fn validate_xsect_spin_ground_states(
    spin_count: usize,
    prepared_count: usize,
    spin_selectors: &[i32],
    controls: XsphXsectAngularControls,
) -> Result<()> {
    ensure!(
        spin_count > 0 && prepared_count == spin_count && spin_selectors.len() == spin_count,
        "XSPH xsect spin state is incomplete: phase spins={spin_count}, prepared grids={prepared_count}, selectors={}",
        spin_selectors.len()
    );
    for (spin_index, &spin_selector) in spin_selectors.iter().enumerate() {
        let expected_selector = xsect_spin_selector_for_row(controls, spin_count, spin_index);
        ensure!(
            spin_selector == expected_selector,
            "XSPH xsect spin selector {spin_selector} at channel {} disagrees with angular-control selector {expected_selector}",
            spin_index + 1
        );
    }
    Ok(())
}

fn xsect_spin_polarized(spin_count: usize, controls: XsphXsectAngularControls) -> bool {
    spin_count == 2 && controls.spin.abs() == 1
}

fn xsect_spin_polarized_cross_terms(spin_count: usize, controls: XsphXsectAngularControls) -> bool {
    xsect_spin_polarized(spin_count, controls) && controls.transition_direction == 0
}

fn xsect_spin_selector_for_row(
    controls: XsphXsectAngularControls,
    spin_count: usize,
    spin_index: usize,
) -> i32 {
    if xsect_spin_polarized(spin_count, controls) {
        if spin_index == 0 {
            -controls.spin.abs()
        } else {
            controls.spin.abs()
        }
    } else {
        controls.spin
    }
}

fn xsph_global_polarization_tensor(global: &GlobalInput) -> [[Complex64; 3]; 3] {
    let mut tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    for (row_index, row) in global.polarization_tensor.iter().enumerate() {
        tensor[row_index] = [
            Complex64::new(row[0], row[1]),
            Complex64::new(row[2], row[3]),
            Complex64::new(row[4], row[5]),
        ];
    }
    tensor
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedPhase {
    phase: PhaseBinData,
    radial: Option<XsphRlDatData>,
    aphase_hubbard: Option<HubbardAphaseBinData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhaseTransitionDimensions {
    final_state_count: usize,
    transition_count: usize,
    q_count: usize,
}

impl PhaseTransitionDimensions {
    fn ordinary() -> Self {
        Self {
            final_state_count: PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            transition_count: PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            q_count: 1,
        }
    }
}

fn phase_transition_dimensions(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
) -> Result<PhaseTransitionDimensions> {
    if !nrixs_xsectjas_requested(caches, input)? {
        return Ok(PhaseTransitionDimensions::ordinary());
    }

    let global = read_optional_global_input(&caches.global_inp)?.with_context(|| {
        format!(
            "XSPH NRIXS/JAS source phase generation requires {}",
            caches.global_inp.display()
        )
    })?;
    ensure!(
        global.control.do_nrixs == 1 || global.control.l2lp == XSPH_NRIXS_L2LP_SENTINEL,
        "global.inp does not enable NRIXS/JAS source phase generation"
    );
    ensure!(
        global.q_control.nq >= 0,
        "XSPH NRIXS/JAS global.inp nq must be nonnegative, got {}",
        global.q_control.nq
    );
    let q_count =
        usize::try_from(global.q_control.nq).context("XSPH NRIXS/JAS nq conversion failed")?;
    ensure!(
        q_count > 0,
        "XSPH NRIXS/JAS source phase generation requires at least one q-vector"
    );
    ensure!(
        global.q_vectors.len() == q_count,
        "global.inp q-vector count {} does not match nq {q_count}",
        global.q_vectors.len()
    );

    let core_hole = core_hole_quantum_numbers(pot.ihole)
        .context("failed to determine XSPH NRIXS/JAS core-hole quantum numbers")?;
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: core_hole.kappa,
        multipole: global.control.le2,
        max_angular_momentum: XSPH_NRIXS_MAX_FINAL_ANGULAR_MOMENTUM,
    })
    .context("failed to build XSPH NRIXS/JAS phase transition dimensions")?;

    Ok(PhaseTransitionDimensions {
        final_state_count: indices.final_state_capacity,
        transition_count: indices.transitions.len(),
        q_count,
    })
}

fn generate_source_phase_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<GeneratedPhase>> {
    if let Some(phase) = generate_empty_cell_phase_bin(caches, input)? {
        return Ok(Some(GeneratedPhase {
            phase,
            radial: None,
            aphase_hubbard: None,
        }));
    }
    generate_normal_potential_phase_bin(caches, input)
}

fn generate_source_phase_handoff_for_discovery(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<GeneratedPhase>> {
    match generate_source_phase_handoff(caches, input) {
        Ok(generated) => Ok(generated),
        Err(error) if is_unsupported_source_phase_generation(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

fn write_generated_phase_handoff_outputs(
    caches: &XsphCachePaths,
    input: &XsphInput,
    generated: GeneratedPhase,
) -> Result<usize> {
    let GeneratedPhase {
        phase,
        radial,
        aphase_hubbard,
    } = generated;
    write_phase_cache(&caches.phase_bin, &phase)?;
    let mut written = 1_usize;
    written += write_or_preserve_rl_dat(caches, input, radial.as_ref())?;
    written += write_stale_or_missing_phase_text_sidecars(caches, input, &phase)?;
    written += write_or_generate_aphase_hubbard_cache(caches, input, &phase, aphase_hubbard)?.0;
    written += write_or_generate_xsph_excitation_poles_cache(caches, input)?.0;
    written += write_or_generate_mpse_cache(caches, input, &phase)?.0;
    written += write_or_generate_emesh_sidecars(caches, input, &phase)?;
    written += write_or_recover_module_log(&caches.log2_dat, &phase, true)?;
    Ok(written)
}

fn generate_empty_cell_phase_bin(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<PhaseBinData>> {
    if tdlda_xsectd_branch_requested(input) {
        return Ok(None);
    }
    if !caches.pot_bin.is_file() {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if !pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
    {
        return Ok(None);
    }

    let mesh = generate_initial_phase_mesh_from_pot(caches, input, &pot)?;
    let spin_selectors = phase_spin_selectors(caches, input)?;
    let transition_dimensions = phase_transition_dimensions(caches, input, &pot)?;
    generate_empty_cell_phase_bin_from_pot(
        input,
        &pot,
        &mesh,
        &spin_selectors,
        transition_dimensions,
    )
    .map(Some)
}

fn generate_empty_cell_phase_bin_from_pot(
    input: &XsphInput,
    pot: &PotBinData,
    mesh: &InitialPhaseMesh,
    spin_selectors: &[i32],
    transition_dimensions: PhaseTransitionDimensions,
) -> Result<PhaseBinData> {
    let spin_count = spin_selectors.len();
    let energy_count = mesh.energies.len();
    let phase_energy_count = energy_count.saturating_sub(mesh.auxiliary_count);
    let reference_energy = empty_cell_reference_energy(input, pot, mesh, spin_count)?;
    let mut potentials = Vec::with_capacity(pot.potential_count());

    for potential_index in 0..pot.potential_count() {
        let angular_limit = phase_angular_limit_from_pot(input, pot, mesh, potential_index)?;
        let l_count = angular_limit
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .context("XSPH empty-cell phase l-count overflow")?;
        let mut phase_shifts = Array3::<Complex64>::zeros((energy_count, l_count, spin_count));
        let muffin_tin_radius = pot.muffin_tin_radii[potential_index];
        let muffin_tin_potential = pot.total_potential[(0, potential_index)];

        for (spin, &spin_selector) in spin_selectors.iter().enumerate() {
            for energy_index in 0..phase_energy_count {
                let setup = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
                    energy: mesh.energies[energy_index],
                    reference_energy: reference_energy[(energy_index, spin)],
                    muffin_tin_potential,
                    lreal: input.control.lreal,
                    energy_index,
                    real_mesh_count: mesh.horizontal_count,
                    muffin_tin_radius,
                    exchange_selector: input.control.ixc,
                })
                .context("failed to set up XSPH empty-cell phase energy")?;
                if setup.decision != XsphPhaseEnergyDecision::Active {
                    continue;
                }
                let dynamics = setup
                    .dynamics
                    .context("active XSPH empty-cell phase energy missing dynamics")?;
                let initial_cycle_count = setup
                    .cycle_count
                    .context("active XSPH empty-cell phase energy missing cycle count")?;
                let plan = xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
                    angular_limit,
                    log_step: input.grid.rgrd,
                    initial_cycle_count,
                    spin_channels: spin_count as i32,
                    spin: spin_selector,
                })
                .context("failed to plan XSPH empty-cell phase channels")?;

                for (slot, channel) in plan.channels.iter().enumerate() {
                    let phase = xsph_empty_cell_phase(XsphEmptyCellPhaseInput {
                        muffin_tin_radius,
                        wave_number: dynamics.wave_number,
                        empty_cell_wave_number: dynamics.empty_cell_wave_number,
                        kappa: channel.kappa,
                    })
                    .context("failed to evaluate XSPH empty-cell phase")?;
                    let cutoff = xsph_phase_cutoff(XsphPhaseCutoffInput {
                        angular_channel: channel.angular_channel,
                        phase_shift: phase.phase_shift,
                    })
                    .context("failed to apply XSPH empty-cell phase cutoff")?;
                    phase_shifts[(energy_index, slot, spin)] = cutoff.phase_shift;
                    if cutoff.terminate_energy {
                        break;
                    }
                }
            }
        }

        potentials.push(PhaseBinPotential {
            lmax: angular_limit,
            atomic_number: 0,
            label: phase_potential_label(input, potential_index),
            phase_shifts,
        });
    }

    Ok(PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: mesh.horizontal_count,
        auxiliary_energy_count: mesh.auxiliary_count,
        ihole: pot.ihole,
        fermi_index: i32::try_from(mesh.fermi_index_1based).context("XSPH fermi index overflow")?,
        pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: transition_dimensions.final_state_count,
        transition_count: transition_dimensions.transition_count,
        q_count: transition_dimensions.q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: pot.scalars.average_norman_radius,
            fermi_level: pot.scalars.fermi_level,
            edge_energy: mesh.edge,
        },
        energy_grid: mesh.energies.clone(),
        reference_energy,
        potentials,
        transition_moments: Array4::<Complex64>::zeros((
            energy_count,
            transition_dimensions.q_count,
            transition_dimensions.transition_count,
            spin_count,
        )),
        raw_pads: None,
    })
}

fn generate_normal_potential_phase_bin(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<GeneratedPhase>> {
    generate_normal_potential_phase_bin_with_mesh(caches, input, None)
}

fn generate_normal_potential_phase_bin_with_mesh(
    caches: &XsphCachePaths,
    input: &XsphInput,
    mesh_override: Option<InitialPhaseMesh>,
) -> Result<Option<GeneratedPhase>> {
    generate_normal_potential_phase_bin_with_mesh_and_spin_selectors(
        caches,
        input,
        mesh_override,
        None,
    )
}

fn generate_normal_potential_phase_bin_with_mesh_and_spin_selectors(
    caches: &XsphCachePaths,
    input: &XsphInput,
    mesh_override: Option<InitialPhaseMesh>,
    spin_selectors_override: Option<Vec<i32>>,
) -> Result<Option<GeneratedPhase>> {
    if !caches.pot_bin.is_file() {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
        || pot
            .atomic_numbers
            .iter()
            .any(|atomic_number| *atomic_number == 0)
    {
        return Ok(None);
    }
    if !pot_has_normal_phase_orbital_handoffs(&pot) {
        return Ok(None);
    }
    if !screened_core_hole_wscrn_handoff_is_supported(caches, &pot) {
        return Ok(None);
    }
    if !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())? {
        return Ok(None);
    }

    let Some(orbital_tables) = normal_potential_orbital_tables(caches, &pot)? else {
        return Ok(None);
    };
    ensure!(
        orbital_tables.bound_orbital_counts.len() == pot.potential_count(),
        "config.dat potential count {} does not match pot.bin potential count {}",
        orbital_tables.bound_orbital_counts.len(),
        pot.potential_count()
    );
    let bound_orbital_counts = pot_effective_bound_orbital_counts(&pot, &orbital_tables)?;
    if bound_orbital_counts.is_empty() {
        return Ok(None);
    }

    let mesh = match mesh_override {
        Some(mesh) => mesh,
        None => generate_initial_phase_mesh_from_pot(caches, input, &pot)?,
    };
    let muffin_tin_radii = pot
        .muffin_tin_radii
        .as_slice()
        .context("pot.bin muffin-tin radii are not contiguous")?;
    let magnetization = xsph_scaled_magnetization(input, &pot)?;
    let spin_selectors = match spin_selectors_override {
        Some(selectors) => selectors,
        None => phase_spin_selectors(caches, input)?,
    };
    let mut prepared = Vec::with_capacity(spin_selectors.len());
    for &spin_selector in &spin_selectors {
        let mut state = xsph_spin_ground_state(caches, input, &pot, &magnetization, spin_selector)?;
        apply_xsph_screened_core_hole(caches, input, &pot, &mut state.total_potential)?;
        prepared.push(
            xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
                muffin_tin_radii,
                electron_density: pot.electron_density.view(),
                total_potential: state.total_potential.view(),
                valence_density: pot.valence_density.view(),
                valence_potential: state.valence_potential.view(),
                magnetization: magnetization.view(),
                bound_large_components: pot.large_components.view(),
                bound_small_components: pot.small_components.view(),
                interstitial_potential: state.interstitial_potential,
                interstitial_density: state.interstitial_density,
                original_radial_dx: LOUCKS_DELTA,
                target_radial_dx: input.grid.rgrd,
                jump_mode: pot.jump_mode,
                potential_jump: 0.0,
                exchange_selector: input.control.ixc,
                radial_count: xsph_phase_radial_grid_count(&pot),
            })
            .context("failed to prepare XSPH normal-potential radial grids")?,
        );
    }
    let excitation_poles = xsph_excitation_poles_from_loss(caches, input, input.control.ixc)?;
    if input.control.i_plsmn > 0 && input.control.ixc == 0 && excitation_poles.is_none() {
        return Ok(None);
    }
    let hubbard = read_active_hubbard_phase_handoff(caches, pot.potential_count())?;
    let broadened_table = load_xsph_broadened_table(&caches.work_dir, input.control.ixc)?;

    let transition_dimensions = phase_transition_dimensions(caches, input, &pot)?;
    generate_normal_potential_phase_bin_from_pot(
        input,
        &pot,
        &mesh,
        &prepared,
        &orbital_tables,
        &bound_orbital_counts,
        excitation_poles.as_deref(),
        hubbard.as_ref(),
        &spin_selectors,
        transition_dimensions,
        broadened_table.as_ref(),
    )
    .map(Some)
}

fn generate_normal_potential_phase_bin_from_pot(
    input: &XsphInput,
    pot: &PotBinData,
    mesh: &InitialPhaseMesh,
    prepared_by_spin: &[XsphPhaseGridPreparation],
    orbital_tables: &RhorrpConfigOrbitalTables,
    bound_orbital_counts: &[usize],
    excitation_poles: Option<&[ExcitationPole]>,
    hubbard: Option<&HubbardVnlmBinData>,
    spin_selectors: &[i32],
    transition_dimensions: PhaseTransitionDimensions,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<GeneratedPhase> {
    let spin_count = spin_selectors.len();
    ensure!(
        spin_count > 0 && prepared_by_spin.len() == spin_count,
        "XSPH prepared spin-grid count {} does not match active spin count {spin_count}",
        prepared_by_spin.len()
    );
    let prepared = &prepared_by_spin[0];
    let energy_count = mesh.energies.len();
    let phase_energy_count = energy_count.saturating_sub(mesh.auxiliary_count);
    let mut reference_energy = Array2::<Complex64>::zeros((energy_count, spin_count));
    let mut potentials = Vec::with_capacity(pot.potential_count());
    let mut radial = None;
    if hubbard.is_some() {
        ensure!(
            spin_count <= XSPH_HUBBARD_SPIN_COUNT,
            "active Hubbard XSPH phase generation supports at most two spin channels, got {spin_count}"
        );
    }
    let mut hubbard_aphase = hubbard.map(|data| HubbardAphaseBinData {
        angular_limit: data.angular_limit,
        values: Array5::from_elem(
            (
                pot.potential_count(),
                XSPH_HUBBARD_SPIN_COUNT,
                energy_count,
                data.angular_count(),
                data.magnetic_count(),
            ),
            Complex64::new(0.0, 0.0),
        ),
    });
    ensure!(
        bound_orbital_counts.len() == pot.potential_count(),
        "XSPH effective bound-orbital count length {} does not match pot.bin potential count {}",
        bound_orbital_counts.len(),
        pot.potential_count()
    );

    for (potential_index, &bound_orbital_count) in bound_orbital_counts
        .iter()
        .enumerate()
        .take(pot.potential_count())
    {
        let angular_limit = phase_angular_limit_from_pot(input, pot, mesh, potential_index)?;
        let l_count = angular_limit
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .context("XSPH normal-potential phase l-count overflow")?;
        let mut phase_shifts = Array3::<Complex64>::zeros((energy_count, l_count, spin_count));
        let muffin_tin_radius = pot.muffin_tin_radii[potential_index];
        let radial_indices = xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius,
            grid_origin: XSPH_LOUCKS_GRID_ORIGIN,
            log_step: prepared.radial_dx,
            radial_capacity: prepared.radii.len(),
        })
        .with_context(|| {
            format!("failed to compute XSPH radial indices for potential {potential_index}")
        })?;
        let radial_match_index = radial_indices
            .radial_match_index_1based
            .checked_sub(1)
            .context("XSPH radial match index underflow")?;
        let potential_index_i32 =
            i32::try_from(potential_index).context("XSPH potential index overflow")?;
        if let Some(header) = xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
            print_radial: input.print_rl,
            potential_index: potential_index_i32,
            muffin_tin_radius,
            angular_limit,
            radial_match_index_1based: radial_indices.radial_match_index_1based,
            log_step: prepared.radial_dx,
            grid_origin: XSPH_LOUCKS_GRID_ORIGIN,
        })
        .with_context(|| {
            format!("failed to prepare XSPH rl.dat header for potential {potential_index}")
        })? {
            radial = Some(XsphRlDatData {
                muffin_tin_radius: header.muffin_tin_radius,
                angular_limit: header.angular_limit,
                radial_match_index_1based: header.radial_match_index_1based,
                log_step: header.log_step,
                grid_origin: header.grid_origin,
                records: Vec::new(),
            });
        }
        let xcpot_active_len = radial_indices.reference_index_1based;
        ensure!(
            bound_orbital_count > 0,
            "config.dat potential {potential_index} has no occupied orbitals"
        );
        ensure!(
            bound_orbital_count <= pot.orbital_occupancy.nrows(),
            "config.dat potential {potential_index} bound-orbital count {bound_orbital_count} exceeds pot.bin orbital occupancy rows {}",
            pot.orbital_occupancy.nrows()
        );
        ensure!(
            potential_index < prepared.bound_large_components.len_of(Axis(2)),
            "prepared XSPH grid is missing potential {potential_index}"
        );

        let radial_prefix = Slice::from(..xcpot_active_len);
        let orbital_prefix = Slice::from(..bound_orbital_count);
        let electron_density_column = prepared
            .electron_density
            .index_axis(Axis(1), potential_index);
        let electron_density = electron_density_column.slice_axis(Axis(0), radial_prefix);
        let many_pole_self_energy = xsph_many_pole_self_energy_for_potential(
            input,
            excitation_poles,
            electron_density,
            xcpot_active_len,
            input.control.ixc,
        )?;
        let magnetization_column = prepared.magnetization.index_axis(Axis(1), potential_index);
        let magnetization = magnetization_column.slice_axis(Axis(0), radial_prefix);
        let valence_density_column = prepared
            .valence_density
            .index_axis(Axis(1), potential_index);
        let valence_density = valence_density_column.slice_axis(Axis(0), radial_prefix);
        let bound_large_potential = prepared
            .bound_large_components
            .index_axis(Axis(2), potential_index);
        let bound_large = bound_large_potential.slice_axis(Axis(1), orbital_prefix);
        let bound_small_potential = prepared
            .bound_small_components
            .index_axis(Axis(2), potential_index);
        let bound_small = bound_small_potential.slice_axis(Axis(1), orbital_prefix);
        let bound_large_coefficients_potential =
            pot.large_coefficients.index_axis(Axis(2), potential_index);
        let bound_large_coefficients =
            bound_large_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
        let bound_small_coefficients_potential =
            pot.small_coefficients.index_axis(Axis(2), potential_index);
        let bound_small_coefficients =
            bound_small_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
        let electron_counts_potential = orbital_tables
            .electron_counts_by_potential
            .index_axis(Axis(1), potential_index);
        let electron_counts = electron_counts_potential.slice_axis(Axis(0), orbital_prefix);
        let valence_counts_potential = pot.orbital_occupancy.index_axis(Axis(1), potential_index);
        let valence_counts = valence_counts_potential.slice_axis(Axis(0), orbital_prefix);
        let kappa_potential = orbital_tables
            .kappa_by_potential
            .index_axis(Axis(1), potential_index);
        let kappa = kappa_potential.slice_axis(Axis(0), orbital_prefix);
        for (spin, &spin_selector) in spin_selectors.iter().enumerate() {
            let spin_prepared = &prepared_by_spin[spin];
            let total_potential_column = spin_prepared
                .total_potential
                .index_axis(Axis(1), potential_index);
            let total_potential = total_potential_column.slice_axis(Axis(0), radial_prefix);
            let valence_potential_column = spin_prepared
                .valence_potential
                .index_axis(Axis(1), potential_index);
            let valence_potential = valence_potential_column.slice_axis(Axis(0), radial_prefix);
            let mut fermi_cache: Option<Array1<XcpotFermiCache>> = None;
            for energy_index in 0..phase_energy_count {
                let xcpot_result = evaluate_xsph_xcpot(XcpotInput {
                    exchange_selector: input.control.ixc,
                    lreal: input.control.lreal,
                    energy: mesh.energies[energy_index],
                    fermi_level: pot.scalars.fermi_level,
                    total_potential,
                    valence_potential,
                    density: electron_density,
                    magnetization,
                    valence_density,
                    active_len: xcpot_active_len,
                    plasmon_selector: input.control.i_plsmn,
                    many_pole_delta_table: None,
                    many_pole_self_energy: many_pole_self_energy
                        .as_ref()
                        .map(|poles| poles.as_xcpot_input()),
                    fermi_cache: fermi_cache.as_ref().map(|cache| cache.view()),
                }, broadened_table)
                .with_context(|| {
                    format!(
                        "failed to evaluate XSPH xcpot for potential {potential_index}, energy row {}",
                        energy_index + 1
                    )
                })?;
                reference_energy[(energy_index, spin)] = xcpot_result.reference_energy;
                if !xcpot_result.fermi_cache.is_empty() {
                    fermi_cache = Some(xcpot_result.fermi_cache.clone());
                }

                let setup = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
                    energy: mesh.energies[energy_index],
                    reference_energy: xcpot_result.reference_energy,
                    muffin_tin_potential: prepared.total_potential[(0, potential_index)],
                    lreal: input.control.lreal,
                    energy_index,
                    real_mesh_count: mesh.horizontal_count,
                    muffin_tin_radius,
                    exchange_selector: input.control.ixc,
                })
                .context("failed to set up XSPH normal-potential phase energy")?;
                if setup.decision != XsphPhaseEnergyDecision::Active {
                    continue;
                }
                let dynamics = setup
                    .dynamics
                    .context("active XSPH normal-potential phase energy missing dynamics")?;
                let initial_cycle_count = setup
                    .cycle_count
                    .context("active XSPH normal-potential phase energy missing cycle count")?;
                let plan = xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
                    angular_limit,
                    log_step: prepared.radial_dx,
                    initial_cycle_count,
                    spin_channels: spin_count as i32,
                    spin: spin_selector,
                })
                .context("failed to plan XSPH normal-potential phase channels")?;
                let solver_total_potential = extend_xcpot_potential(
                    &xcpot_result.total_potential,
                    prepared.radii.len(),
                    "total",
                )?;
                let solver_valence_potential = if xcpot_result.valence_potential.is_empty() {
                    solver_total_potential.clone()
                } else {
                    extend_xcpot_potential(
                        &xcpot_result.valence_potential,
                        prepared.radii.len(),
                        "valence",
                    )?
                };

                for (slot, channel) in plan.channels.iter().enumerate() {
                    let regular = xsph_regular_phase_channel(
                        FovrgDiracSolverInput {
                            exchange_cycle_count: channel.cycle_count,
                            target_kappa: channel.kappa,
                            muffin_tin_radius,
                            target_last_index: radial_match_index,
                            energy: dynamics.momentum_squared,
                            step: prepared.radial_dx,
                            radii: prepared.radii.view(),
                            exchange_correlation_potential: solver_total_potential.view(),
                            valence_exchange_correlation_potential: solver_valence_potential
                                .view(),
                            bound_large_components: bound_large,
                            bound_small_components: bound_small,
                            bound_large_coefficients,
                            bound_small_coefficients,
                            electron_counts,
                            valence_counts,
                            kappa,
                            muffin_tin_large_component: Complex64::new(0.0, 0.0),
                            muffin_tin_small_component: Complex64::new(0.0, 0.0),
                            atomic_number: pot.atomic_numbers[potential_index] as f64,
                            irregular: false,
                            c3_scale: channel.c3_derivative,
                            radial_match_index,
                            bound_orbital_count,
                        },
                        dynamics.wave_number,
                    )
                    .with_context(|| {
                        format!(
                            "failed to solve XSPH FOVRG channel for potential {potential_index}, energy row {}, slot {}",
                            energy_index + 1,
                            slot + 1
                        )
                    })?;
                    if let Some(output) =
                        xsph_phase_radial_output(XsphPhaseRadialOutputInput {
                            print_radial: input.print_rl,
                            potential_index: potential_index_i32,
                            angular_channel: channel.angular_channel,
                            angular_limit,
                            energy: mesh.energies[energy_index],
                            phase_shift: regular.phase.phase_shift,
                            phase_amplitude: regular.phase.phase_amplitude,
                            regular_large: regular.regular_solution.large_component.view(),
                            regular_small: regular.regular_solution.small_component.view(),
                            active_len: radial_indices.radial_match_index_1based,
                        })
                        .with_context(|| {
                            format!(
                                "failed to prepare XSPH rl.dat row for potential {potential_index}, energy row {}, slot {}",
                                energy_index + 1,
                                slot + 1
                            )
                        })?
                    {
                        let Some(radial) = radial.as_mut() else {
                            bail!("XSPH rl.dat row was generated before its header");
                        };
                        radial.records.push(XsphRlDatRecord {
                            energy: output.energy,
                            angular_momentum: output.output_angular_momentum,
                            phase_shift: output.phase_shift,
                            regular_large: output.regular_large,
                            regular_small: output.regular_small,
                        });
                    }
                    let cutoff = xsph_phase_cutoff(XsphPhaseCutoffInput {
                        angular_channel: channel.angular_channel,
                        phase_shift: regular.phase.phase_shift,
                    })
                    .context("failed to apply XSPH normal-potential phase cutoff")?;
                    phase_shifts[(energy_index, slot, spin)] = cutoff.phase_shift;
                    if let (Some(hubbard), Some(aphase)) = (hubbard, hubbard_aphase.as_mut())
                        && let Ok(hubbard_angular_channel) =
                            usize::try_from(channel.angular_channel)
                        && hubbard_angular_channel <= hubbard.angular_limit
                    {
                        let hubbard_potential_for_potential =
                            hubbard.values.index_axis(Axis(0), potential_index);
                        let hubbard_potential =
                            hubbard_potential_for_potential.index_axis(Axis(0), spin);
                        let hubbard_valence_potential = if xcpot_result.valence_potential.is_empty()
                        {
                            xcpot_result.total_potential.view()
                        } else {
                            xcpot_result.valence_potential.view()
                        };
                        let shifts =
                            xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
                                angular_channel: channel.angular_channel,
                                spin_projection: i32::try_from(spin + 1)
                                    .context("XSPH Hubbard spin index overflow")?,
                                total_potential: xcpot_result.total_potential.view(),
                                valence_potential: hubbard_valence_potential,
                                hubbard_potential,
                                active_len: xcpot_active_len,
                            })
                            .with_context(|| {
                                format!(
                                    "failed to prepare XSPH Hubbard phase shifts for potential {potential_index}, energy row {}, slot {}",
                                    energy_index + 1,
                                    slot + 1
                                )
                            })?;
                        let mut magnetic_phase_shifts = Array1::<Complex64>::zeros(shifts.len());
                        for (magnetic_offset, shift) in shifts.iter().enumerate() {
                            let shifted_total_potential = extend_xcpot_potential(
                                &shift.total_potential,
                                prepared.radii.len(),
                                "Hubbard total",
                            )?;
                            let shifted_valence_potential = extend_xcpot_potential(
                                &shift.valence_potential,
                                prepared.radii.len(),
                                "Hubbard valence",
                            )?;
                            let perturbed = xsph_regular_phase_channel(
                                FovrgDiracSolverInput {
                                    exchange_cycle_count: channel.cycle_count,
                                    target_kappa: channel.kappa,
                                    muffin_tin_radius,
                                    target_last_index: radial_match_index,
                                    energy: dynamics.momentum_squared,
                                    step: prepared.radial_dx,
                                    radii: prepared.radii.view(),
                                    exchange_correlation_potential: shifted_total_potential.view(),
                                    valence_exchange_correlation_potential:
                                        shifted_valence_potential.view(),
                                    bound_large_components: bound_large,
                                    bound_small_components: bound_small,
                                    bound_large_coefficients,
                                    bound_small_coefficients,
                                    electron_counts,
                                    valence_counts,
                                    kappa,
                                    muffin_tin_large_component: Complex64::new(0.0, 0.0),
                                    muffin_tin_small_component: Complex64::new(0.0, 0.0),
                                    atomic_number: pot.atomic_numbers[potential_index] as f64,
                                    irregular: false,
                                    c3_scale: channel.c3_derivative,
                                    radial_match_index,
                                    bound_orbital_count,
                                },
                                dynamics.wave_number,
                            )
                            .with_context(|| {
                                format!(
                                    "failed to solve XSPH Hubbard FOVRG channel for potential {potential_index}, energy row {}, slot {}, magnetic channel {}",
                                    energy_index + 1,
                                    slot + 1,
                                    shift.magnetic_channel + 1
                                )
                            })?;
                            magnetic_phase_shifts[magnetic_offset] = perturbed.phase.phase_shift;
                        }
                        let assignments =
                            xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
                                energy_index,
                                angular_channel: channel.angular_channel,
                                hubbard_angular_limit: hubbard.angular_limit,
                                magnetic_phase_shifts: magnetic_phase_shifts.view(),
                            })
                            .with_context(|| {
                                format!(
                                    "failed to assign XSPH Hubbard phases for potential {potential_index}, energy row {}, slot {}",
                                    energy_index + 1,
                                    slot + 1
                                )
                            })?;
                        for assignment in assignments {
                            aphase.values[(
                                potential_index,
                                spin,
                                assignment.energy_index,
                                assignment.angular_channel,
                                assignment.magnetic_channel,
                            )] = assignment.phase_shift;
                        }
                    }
                    if cutoff.terminate_energy {
                        break;
                    }
                }
            }
        }

        potentials.push(PhaseBinPotential {
            lmax: angular_limit,
            atomic_number: pot.atomic_numbers[potential_index],
            label: phase_potential_label(input, potential_index),
            phase_shifts,
        });
    }

    for spin in 0..spin_count {
        if hubbard.is_some() {
            xsph_hubbard_phase_reference_tail(
                reference_energy.column_mut(spin),
                energy_count,
                mesh.horizontal_count,
                mesh.auxiliary_count,
            )
            .context("failed to finalize XSPH Hubbard reference-energy tail")?;
        } else {
            xsph_phase_reference_tail(
                reference_energy.column_mut(spin),
                energy_count,
                mesh.horizontal_count,
                mesh.auxiliary_count,
            )
            .context("failed to finalize XSPH normal-potential reference-energy tail")?;
        }
    }

    let phase = PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: mesh.horizontal_count,
        auxiliary_energy_count: mesh.auxiliary_count,
        ihole: pot.ihole,
        fermi_index: i32::try_from(mesh.fermi_index_1based).context("XSPH fermi index overflow")?,
        pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: transition_dimensions.final_state_count,
        transition_count: transition_dimensions.transition_count,
        q_count: transition_dimensions.q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: pot.scalars.average_norman_radius,
            fermi_level: pot.scalars.fermi_level,
            edge_energy: mesh.edge,
        },
        energy_grid: mesh.energies.clone(),
        reference_energy,
        potentials,
        transition_moments: Array4::<Complex64>::zeros((
            energy_count,
            transition_dimensions.q_count,
            transition_dimensions.transition_count,
            spin_count,
        )),
        raw_pads: None,
    };

    Ok(GeneratedPhase {
        phase,
        radial,
        aphase_hubbard: hubbard_aphase,
    })
}

fn generate_missing_rl_dat_from_normal_potential_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<XsphRlDatData>> {
    if !input.print_rl || !rl_dat_needs_generation(&caches.rl_dat) {
        return Ok(None);
    }
    generate_rl_dat_from_normal_potential_handoff(caches, input)
}

fn generate_rl_dat_from_normal_potential_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<XsphRlDatData>> {
    if !input.print_rl {
        return Ok(None);
    }
    Ok(generate_normal_potential_phase_bin(caches, input)?.and_then(|generated| generated.radial))
}

fn can_generate_missing_rl_dat_from_normal_potential_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if !input.print_rl || !rl_dat_needs_generation(&caches.rl_dat) || !caches.pot_bin.is_file() {
        return Ok(false);
    }
    can_generate_rl_dat_from_normal_potential_handoff(caches, input)
}

fn can_repair_print_rl_cache_from_normal_potential_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if can_generate_missing_rl_dat_from_normal_potential_handoff(caches, input)? {
        return Ok(true);
    }
    Ok(generate_rl_if_stale_against_source(caches, input)?.is_some())
}

fn can_generate_rl_dat_from_normal_potential_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    if !input.print_rl || !caches.pot_bin.is_file() {
        return Ok(false);
    }
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
    {
        return Ok(false);
    }
    can_generate_normal_potential_phase_from_pot(caches, input, &pot)
}

fn rl_dat_needs_generation(path: &Path) -> bool {
    !path.is_file() || read_xsph_rl_dat(path).is_err()
}

fn can_generate_mpse_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<bool> {
    if !should_generate_mpse(input, phase) {
        return Ok(false);
    }
    Ok(generate_mpse_dat(caches, input, phase)?.is_some())
}

/// Generate an MPSE table from complete XSPH source state without writing it.
///
/// RIXS `ReadSigma` can consume the same self-energy grid that XSPH would
/// normally persist as `mpse.dat`. Keep this read-only so source-handoff
/// predicates can validate the branch without mutating the work directory.
pub(crate) fn generate_mpse_dat_from_source_handoff(
    work_dir: &Path,
) -> Result<Option<MpseDatData>> {
    let caches = XsphCachePaths::new(work_dir);
    if !work_dir.join("xsph.inp").is_file() || !caches.phase_bin.is_file() {
        return Ok(None);
    }

    let input = read_input(work_dir)?;
    let phase = read_phase_bin(&caches.phase_bin)
        .with_context(|| format!("failed to read {}", caches.phase_bin.display()))?;
    generate_mpse_dat(&caches, &input, &phase)
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedNormalXsect {
    xsect: XsectDatData,
    transition_moments: Array4<Complex64>,
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedNrixsSpectrum {
    xsect: XsectDatData,
    handoffs: XseclFromXsphNrixs,
    transition_moments: Array4<Complex64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NrixsSpectrumSourcePlan {
    initial_kappa: i32,
    initial_state_j: i32,
    max_angular_momentum: usize,
    final_lj_max: usize,
    final_state_count: usize,
    active_len: usize,
    kind: Array1<i32>,
    decomposition_l: Array1<i32>,
    final_lj: Array1<i32>,
    orbital_l: Array1<i32>,
    calculation_plan: refeff_core::XsphCalculationPlan,
    lj_needed_by_calculation: Vec<Array1<i32>>,
    transitions: Vec<XseclBinTransition>,
    q_weights: Array1<Complex64>,
    q_cosines: Array2<f64>,
    q_bessel: Option<Array3<f64>>,
    transition_weights: Array3<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NrixsSpectrumRadialSourceContext {
    active_radial_len: usize,
    initial_l: usize,
    log_step: f64,
    hole_normalization: f64,
    initial_large: Array1<f64>,
    initial_small: Array1<f64>,
    radii: Array1<f64>,
    q_bessel: Array3<f64>,
    orthogonality_correction: Array2<Complex64>,
    orthogonality_normalization: Complex64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct NrixsSpectrumRadialChannel<'a> {
    final_kappa: i32,
    phase_shift: Complex64,
    regular_large: ArrayView1<'a, Complex64>,
    regular_small: ArrayView1<'a, Complex64>,
    irregular_large: ArrayView1<'a, Complex64>,
    irregular_small: ArrayView1<'a, Complex64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NrixsSpectrumRowSource {
    decomposition_cross_sections: Array1<Complex64>,
    total_angular_cross_sections: Array1<Complex64>,
    atom_cross_sections: Array1<Complex64>,
    total_spectrum_norm: f64,
}

#[allow(dead_code)]
fn nrixs_spectrum_radial_source_context_from_handoffs(
    plan: &NrixsSpectrumSourcePlan,
    initial_l: usize,
    initial_large: ArrayView1<'_, f64>,
    initial_small: ArrayView1<'_, f64>,
    radii: ArrayView1<'_, f64>,
    log_step: f64,
    norman_index_1based: usize,
    active_radial_len: usize,
) -> Result<NrixsSpectrumRadialSourceContext> {
    ensure!(
        active_radial_len > 0,
        "XSPH NRIXS/JAS radial source context requires a nonempty active radial prefix"
    );
    ensure!(
        active_radial_len <= radii.len()
            && active_radial_len <= initial_large.len()
            && active_radial_len <= initial_small.len(),
        "XSPH NRIXS/JAS active radial length {active_radial_len} exceeds radial/core-hole handoff lengths ({}, {}, {})",
        radii.len(),
        initial_large.len(),
        initial_small.len()
    );
    let q_bessel = plan
        .q_bessel
        .as_ref()
        .context("XSPH NRIXS/JAS radial source context requires q-Bessel tables")?;
    ensure!(
        active_radial_len <= q_bessel.len_of(Axis(0)),
        "XSPH NRIXS/JAS active radial length {active_radial_len} exceeds q-Bessel radial rows {}",
        q_bessel.len_of(Axis(0))
    );
    let initial_j = usize::try_from(plan.initial_state_j)
        .context("XSPH NRIXS/JAS initial-state j is negative")?;
    let hole_normalization = xsph_xsect_hole_normalization(XsphXsectHoleNormalizationInput {
        initial_l,
        log_step,
        radii,
        initial_large,
        initial_small,
        norman_index_1based,
    })
    .context("failed to normalize XSPH NRIXS/JAS core-hole spinor")?;
    ensure!(
        hole_normalization.normalization.is_finite() && hole_normalization.normalization > 0.0,
        "XSPH NRIXS/JAS core-hole normalization must be positive, got {}",
        hole_normalization.normalization
    );
    // xsectjas computes xinorm as a convergence check but keeps the input spinor unchanged.
    let active_initial_large = initial_large
        .slice_axis(Axis(0), Slice::from(..active_radial_len))
        .to_owned();
    let active_initial_small = initial_small
        .slice_axis(Axis(0), Slice::from(..active_radial_len))
        .to_owned();
    let active_radii = radii
        .slice_axis(Axis(0), Slice::from(..active_radial_len))
        .to_owned();
    let active_q_bessel = q_bessel
        .slice_axis(Axis(0), Slice::from(..active_radial_len))
        .to_owned();
    let orthogonality = xsph_jas_orthogonality_correction(XsphJasOrthogonalityCorrectionInput {
        initial_j,
        initial_l,
        large_component: active_initial_large.view(),
        small_component: active_initial_small.view(),
        q_bessel: active_q_bessel.view(),
        radii: active_radii.view(),
        log_step,
        ljmax: plan.final_lj_max,
        active_len: active_radial_len,
    })
    .context("failed to build XSPH NRIXS/JAS orthogonality correction")?;

    Ok(NrixsSpectrumRadialSourceContext {
        active_radial_len,
        initial_l,
        log_step,
        hole_normalization: hole_normalization.normalization,
        initial_large: active_initial_large,
        initial_small: active_initial_small,
        radii: active_radii,
        q_bessel: active_q_bessel,
        orthogonality_correction: orthogonality.corrections,
        orthogonality_normalization: orthogonality.normalization,
    })
}

#[allow(dead_code)]
fn nrixs_spectrum_row_from_radial_channels(
    plan: &NrixsSpectrumSourcePlan,
    context: &NrixsSpectrumRadialSourceContext,
    channels: &[NrixsSpectrumRadialChannel<'_>],
    spin_index: usize,
    mix_dff: bool,
    mdff_mode: i32,
) -> Result<NrixsSpectrumRowSource> {
    let q_count = plan.q_weights.len();
    ensure!(
        q_count > 0,
        "XSPH NRIXS/JAS source row assembly requires at least one q-vector"
    );
    ensure!(
        context.q_bessel.len_of(Axis(2)) == q_count,
        "XSPH NRIXS/JAS q-Bessel table has {} q columns, expected {q_count}",
        context.q_bessel.len_of(Axis(2))
    );
    ensure!(
        context.orthogonality_correction.len_of(Axis(1)) == q_count,
        "XSPH NRIXS/JAS orthogonality table has {} q columns, expected {q_count}",
        context.orthogonality_correction.len_of(Axis(1))
    );
    let calculation_count = plan.calculation_plan.calculations.nrows();
    ensure!(
        channels.len() == calculation_count,
        "XSPH NRIXS/JAS radial channel count {} does not match calculation count {}",
        channels.len(),
        calculation_count
    );
    let channel_count = plan.final_lj_max + 1;
    let q_weights = nrixs_effective_q_weights(plan.q_weights.view(), mix_dff)?;
    let q_pairs = nrixs_q_pairs(mix_dff, mdff_mode, q_count)?;
    let legendre_by_pair = nrixs_legendre_by_q_pair(plan.q_cosines.view(), q_count, channel_count)?;
    let mut decomposition = Array1::<Complex64>::zeros(channel_count);
    let mut total_angular = Array1::<Complex64>::zeros(channel_count);
    let mut atom = Array1::<Complex64>::zeros(plan.final_state_count);
    let mut total_spectrum_norm = 0.0_f64;

    for (calculation_index, channel) in channels.iter().enumerate() {
        let expected_kappa = plan.calculation_plan.calculations[(calculation_index, 0)];
        ensure!(
            channel.final_kappa == expected_kappa,
            "XSPH NRIXS/JAS radial channel {} has kappa {}, expected {}",
            calculation_index + 1,
            channel.final_kappa,
            expected_kappa
        );
        let needed_multipoles = plan
            .lj_needed_by_calculation
            .get(calculation_index)
            .with_context(|| {
                format!(
                    "missing XSPH NRIXS/JAS lj-needed flags for calculation {}",
                    calculation_index + 1
                )
            })?;
        let calculation_index_1based = i32::try_from(calculation_index + 1)
            .context("XSPH NRIXS/JAS calculation index overflow")?;
        let regular_by_q = (0..q_count)
            .map(|q_index| {
                let q_bessel = context.q_bessel.index_axis(Axis(2), q_index);
                let orthogonality = context.orthogonality_correction.index_axis(Axis(1), q_index);
                xsph_jas_radial_integral(XsphJasRadialIntegralInput {
                    initial_kappa: plan.initial_kappa,
                    final_kappa: channel.final_kappa,
                    initial_large: context.initial_large.view(),
                    initial_small: context.initial_small.view(),
                    final_large_regular: channel.regular_large,
                    final_small_regular: channel.regular_small,
                    needed_multipoles: needed_multipoles.view(),
                    q_bessel,
                    orthogonality_correction: orthogonality,
                    radii: context.radii.view(),
                    log_step: context.log_step,
                    ljmax: plan.final_lj_max,
                    active_len: context.active_radial_len,
                })
                .with_context(|| {
                    format!(
                        "failed to evaluate XSPH NRIXS/JAS regular radial integrals for calculation {}, q {}",
                        calculation_index + 1,
                        q_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        for &(iq, iqq) in &q_pairs {
            let q_bessel = context.q_bessel.index_axis(Axis(2), iqq);
            let orthogonality = context.orthogonality_correction.index_axis(Axis(1), iqq);
            let irregular = xsph_jas_radial_cross_integral(XsphJasRadialCrossIntegralInput {
                initial_kappa: plan.initial_kappa,
                final_kappa: channel.final_kappa,
                initial_large: context.initial_large.view(),
                initial_small: context.initial_small.view(),
                final_large_irregular: channel.irregular_large,
                final_small_irregular: channel.irregular_small,
                regular_coupling: regular_by_q[iq].regular_coupling.view(),
                needed_multipoles: needed_multipoles.view(),
                q_bessel,
                orthogonality_correction: orthogonality,
                radii: context.radii.view(),
                log_step: context.log_step,
                ljmax: plan.final_lj_max,
                active_len: context.active_radial_len,
            })
            .with_context(|| {
                format!(
                    "failed to evaluate XSPH NRIXS/JAS irregular radial integrals for calculation {}, q pair ({}, {})",
                    calculation_index + 1,
                    iq + 1,
                    iqq + 1
                )
            })?;
            let q_product = q_weights[iq] * q_weights[iqq];
            nrixs_accumulate_q_pair_spectra(
                plan,
                calculation_index_1based,
                spin_index,
                iq,
                iqq,
                q_count,
                q_product,
                &legendre_by_pair,
                regular_by_q[iq].radial_integrals.view(),
                regular_by_q[iqq].radial_integrals.view(),
                irregular.radial_integrals.view(),
                &mut decomposition,
                &mut total_angular,
                &mut atom,
                &mut total_spectrum_norm,
            )?;
        }
    }

    Ok(NrixsSpectrumRowSource {
        decomposition_cross_sections: decomposition,
        total_angular_cross_sections: total_angular,
        atom_cross_sections: atom,
        total_spectrum_norm,
    })
}

#[allow(clippy::too_many_arguments)]
fn nrixs_accumulate_q_pair_spectra(
    plan: &NrixsSpectrumSourcePlan,
    calculation_index_1based: i32,
    spin_index: usize,
    iq: usize,
    iqq: usize,
    q_count: usize,
    q_product: Complex64,
    legendre_by_pair: &[f64],
    regular_iq: ArrayView1<'_, Complex64>,
    regular_iqq: ArrayView1<'_, Complex64>,
    irregular_pair: ArrayView1<'_, Complex64>,
    decomposition: &mut Array1<Complex64>,
    total_angular: &mut Array1<Complex64>,
    atom: &mut Array1<Complex64>,
    total_spectrum_norm: &mut f64,
) -> Result<()> {
    let channel_count = plan.final_lj_max + 1;
    ensure!(
        regular_iq.len() >= channel_count
            && regular_iqq.len() >= channel_count
            && irregular_pair.len() >= channel_count,
        "XSPH NRIXS/JAS q-pair radial integral length is too short for {} channels",
        channel_count
    );
    for state_index in 0..plan.active_len {
        let mapped = plan.calculation_plan.index_map[state_index]
            .checked_abs()
            .context("XSPH NRIXS/JAS calculation index map overflow")?;
        if mapped != calculation_index_1based {
            continue;
        }

        let final_lj =
            nrixs_nonnegative_index("final_lj", state_index, plan.final_lj[state_index])?;
        ensure!(
            final_lj <= plan.final_lj_max,
            "XSPH NRIXS/JAS final_lj {} at state {} exceeds ljmax {}",
            final_lj,
            state_index + 1,
            plan.final_lj_max
        );
        let orbital_l =
            nrixs_nonnegative_index("orbital_l", state_index, plan.orbital_l[state_index])?;
        let trace = nrixs_transition_trace(
            plan.transition_weights.view(),
            spin_index,
            state_index,
            plan.initial_state_j,
        )?;
        let legendre = legendre_by_pair[(iq * q_count + iqq) * channel_count + final_lj];
        let weighted_trace = trace * q_product;
        let regular_amplitude =
            -Complex64::new(0.0, 1.0) * regular_iq[final_lj] * regular_iqq[final_lj] * legendre;
        let irregular_amplitude = irregular_pair[final_lj] * legendre;

        total_angular[final_lj] -= regular_amplitude * weighted_trace;
        total_angular[final_lj] -= irregular_amplitude * weighted_trace;
        atom[state_index] -= regular_amplitude * weighted_trace;
        atom[state_index] -= irregular_amplitude * weighted_trace;
        if orbital_l <= plan.final_lj_max {
            decomposition[orbital_l] -= regular_amplitude * weighted_trace;
        }
        *total_spectrum_norm += nrixs_regular_norm_increment(
            regular_iq[final_lj],
            regular_iqq[final_lj],
            final_lj,
            q_product,
        );
    }
    Ok(())
}

fn nrixs_effective_q_weights(
    q_weights: ArrayView1<'_, Complex64>,
    mix_dff: bool,
) -> Result<Vec<Complex64>> {
    q_weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| {
            nrixs_ensure_finite_complex("q_weights", index, weight)?;
            let effective = if mix_dff { weight } else { weight.sqrt() };
            nrixs_ensure_finite_complex("effective_q_weight", index, effective)?;
            Ok(effective)
        })
        .collect()
}

fn nrixs_q_pairs(mix_dff: bool, mdff_mode: i32, q_count: usize) -> Result<Vec<(usize, usize)>> {
    if !mix_dff {
        return Ok((0..q_count).map(|index| (index, index)).collect());
    }
    match mdff_mode {
        1 => Ok((0..q_count)
            .flat_map(|iq| (0..q_count).map(move |iqq| (iq, iqq)))
            .collect()),
        2 if q_count >= 2 => Ok(vec![(0, 1)]),
        2 => bail!("XSPH NRIXS/JAS MDFF mode 2 requires at least two q-vectors"),
        _ => bail!("XSPH NRIXS/JAS unsupported MDFF mode {mdff_mode}"),
    }
}

fn nrixs_legendre_by_q_pair(
    q_cosines: ArrayView2<'_, f64>,
    q_count: usize,
    channel_count: usize,
) -> Result<Vec<f64>> {
    ensure!(
        q_cosines.nrows() >= q_count && q_cosines.ncols() >= q_count,
        "XSPH NRIXS/JAS q-cosine matrix shape {:?} is too small for {q_count} q-vectors",
        q_cosines.dim()
    );
    let mut legendre_by_pair = vec![0.0; q_count * q_count * channel_count];
    for iq in 0..q_count {
        for iqq in 0..q_count {
            let cosine = q_cosines[(iq, iqq)];
            ensure!(
                cosine.is_finite(),
                "XSPH NRIXS/JAS q-cosine ({}, {}) is not finite",
                iq + 1,
                iqq + 1
            );
            let offset = (iq * q_count + iqq) * channel_count;
            legendre_polynomials_into(
                cosine,
                &mut legendre_by_pair[offset..offset + channel_count],
            );
        }
    }
    Ok(legendre_by_pair)
}

fn nrixs_nonnegative_index(name: &'static str, index: usize, value: i32) -> Result<usize> {
    ensure!(
        value >= 0,
        "XSPH NRIXS/JAS {name} at state {} must be nonnegative, got {value}",
        index + 1
    );
    usize::try_from(value).with_context(|| {
        format!(
            "XSPH NRIXS/JAS {name} at state {} cannot be converted to usize",
            index + 1
        )
    })
}

fn nrixs_transition_trace(
    transition_weights: ArrayView3<'_, f64>,
    spin_index: usize,
    state_index: usize,
    initial_j2: i32,
) -> Result<f64> {
    ensure!(
        spin_index < transition_weights.len_of(Axis(0)),
        "XSPH NRIXS/JAS spin index {} exceeds transition-weight spin rows {}",
        spin_index,
        transition_weights.len_of(Axis(0))
    );
    ensure!(
        state_index < transition_weights.len_of(Axis(1)),
        "XSPH NRIXS/JAS state index {} exceeds transition-weight rows {}",
        state_index + 1,
        transition_weights.len_of(Axis(1))
    );
    ensure!(
        initial_j2 >= 0,
        "XSPH NRIXS/JAS initial doubled angular momentum is negative: {initial_j2}"
    );
    let mut trace = 0.0;
    let mut magnetic_j2 = -initial_j2;
    while magnetic_j2 <= initial_j2 {
        let magnetic_index =
            usize::try_from((magnetic_j2 + initial_j2) / 2).context("magnetic index overflow")?;
        ensure!(
            magnetic_index < transition_weights.len_of(Axis(2)),
            "XSPH NRIXS/JAS magnetic index {} exceeds transition-weight columns {}",
            magnetic_index,
            transition_weights.len_of(Axis(2))
        );
        let value = transition_weights[(spin_index, state_index, magnetic_index)];
        ensure!(
            value.is_finite(),
            "XSPH NRIXS/JAS transition weight at spin {}, state {}, magnetic column {} is not finite",
            spin_index,
            state_index + 1,
            magnetic_index + 1
        );
        trace += value * value;
        magnetic_j2 += 2;
    }
    Ok(trace)
}

fn nrixs_regular_norm_increment(
    radial_iq: Complex64,
    radial_iqq: Complex64,
    final_lj: usize,
    q_product: Complex64,
) -> f64 {
    let denominator = (2 * final_lj + 1) as f64;
    (radial_iq.conj() * radial_iqq * q_product).re / denominator
}

fn nrixs_ensure_finite_complex(name: &'static str, index: usize, value: Complex64) -> Result<()> {
    ensure!(
        value.re.is_finite() && value.im.is_finite(),
        "XSPH NRIXS/JAS {name} at index {} is not finite: {value}",
        index + 1
    );
    Ok(())
}

#[allow(dead_code)]
fn nrixs_spectrum_handoffs_from_rows(
    phase: &PhaseBinData,
    plan: &NrixsSpectrumSourcePlan,
    rows: &[NrixsSpectrumRowSource],
    chemical_potential_ev: f64,
    core_hole_width_hartree: f64,
) -> Result<XseclFromXsphNrixs> {
    ensure!(
        rows.len() == phase.energy_count,
        "XSPH NRIXS/JAS row count {} does not match phase.bin energy count {}",
        rows.len(),
        phase.energy_count
    );
    ensure!(
        phase.transition_count == plan.active_len,
        "phase.bin transition count {} does not match NRIXS/JAS plan active length {}",
        phase.transition_count,
        plan.active_len
    );
    ensure!(
        phase.final_state_count == plan.final_state_count,
        "phase.bin final-state count {} does not match NRIXS/JAS plan final-state count {}",
        phase.final_state_count,
        plan.final_state_count
    );
    ensure!(
        chemical_potential_ev.is_finite(),
        "XSPH NRIXS/JAS chemical potential must be finite"
    );
    ensure!(
        core_hole_width_hartree.is_finite(),
        "XSPH NRIXS/JAS core-hole width must be finite"
    );
    let channel_count = plan.final_lj_max + 1;
    for (index, row) in rows.iter().enumerate() {
        ensure!(
            row.decomposition_cross_sections.len() == channel_count,
            "XSPH NRIXS/JAS decomposition row {} has {} channels, expected {}",
            index + 1,
            row.decomposition_cross_sections.len(),
            channel_count
        );
        ensure!(
            row.total_angular_cross_sections.len() == channel_count,
            "XSPH NRIXS/JAS total-angular row {} has {} channels, expected {}",
            index + 1,
            row.total_angular_cross_sections.len(),
            channel_count
        );
        ensure!(
            row.atom_cross_sections.len() == phase.final_state_count,
            "XSPH NRIXS/JAS atom row {} has {} final states, expected {}",
            index + 1,
            row.atom_cross_sections.len(),
            phase.final_state_count
        );
    }

    let fermi_index =
        usize::try_from(phase.fermi_index).context("phase.bin fermi index is negative")?;
    let energy = xsecl_energy_grid_from_phase(phase, chemical_potential_ev);
    let decomposition = Array2::from_shape_fn(
        (phase.energy_count, channel_count),
        |(energy_index, channel)| rows[energy_index].decomposition_cross_sections[channel],
    );
    let total_angular = Array2::from_shape_fn(
        (phase.energy_count, channel_count),
        |(energy_index, channel)| rows[energy_index].total_angular_cross_sections[channel],
    );
    let atom = Array2::from_shape_fn(
        (phase.energy_count, phase.final_state_count),
        |(energy_index, final_state)| rows[energy_index].atom_cross_sections[final_state],
    );

    xsecl_from_xsph_nrixs(XseclFromXsphNrixsInput {
        header: XseclDatHeader {
            real_energy_count: phase.main_energy_count,
            fermi_index,
            edge: phase.scalars.edge_energy,
            emu: chemical_potential_ev,
            core_hole_width: core_hole_width_hartree,
        },
        energy: energy.view(),
        decomposition_cross_sections: decomposition.view(),
        total_angular_cross_sections: total_angular.view(),
        atom_cross_sections: atom.view(),
        transitions: &plan.transitions,
        initial_state_j: plan.initial_state_j,
        pad_width: phase.pad_width,
    })
    .context("failed to build XSPH NRIXS/JAS xsecl sidecar payloads from source rows")
}

#[allow(dead_code)]
fn nrixs_spectrum_source_plan_from_handoffs(
    input: &XsphInput,
    global: &GlobalInput,
    phase: &PhaseBinData,
    radii: Option<ArrayView1<'_, f64>>,
) -> Result<NrixsSpectrumSourcePlan> {
    const ABSORBER_INDEX: usize = 0;

    ensure!(
        global.control.do_nrixs == 1,
        "XSPH NRIXS/JAS source plan requires global.inp do_nrixs=1"
    );
    ensure!(
        !phase.potentials.is_empty(),
        "phase.bin contains no potentials for XSPH NRIXS/JAS source planning"
    );
    ensure!(
        input.lmaxph.len() > ABSORBER_INDEX,
        "XSPH lmaxph is missing the absorber entry for NRIXS/JAS source planning"
    );
    ensure!(
        input.lmaxph[ABSORBER_INDEX] >= 0,
        "XSPH absorber lmaxph must be nonnegative for NRIXS/JAS source planning"
    );

    let core_hole = core_hole_quantum_numbers(phase.ihole)
        .context("failed to determine XSPH NRIXS/JAS core-hole quantum numbers")?;
    let max_angular_momentum = XSPH_NRIXS_MAX_FINAL_ANGULAR_MOMENTUM;
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: core_hole.kappa,
        multipole: global.control.le2,
        max_angular_momentum,
    })
    .context("failed to build XSPH NRIXS/JAS transition indices")?;
    ensure!(
        phase.transition_count == indices.transitions.len(),
        "phase.bin transition count {} does not match generated NRIXS/JAS transition count {}",
        phase.transition_count,
        indices.transitions.len()
    );
    ensure!(
        phase.final_state_count >= indices.final_state_capacity,
        "phase.bin final-state count {} cannot supply generated NRIXS/JAS capacity {}",
        phase.final_state_count,
        indices.final_state_capacity
    );

    let active_len = indices.transitions.len();
    let kind = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.final_state_kappa),
    );
    let decomposition_l = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.decomposition_channel),
    );
    let final_lj = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.total_angular_momentum_channel),
    );
    let orbital_l = Array1::from_iter(
        indices
            .transitions
            .iter()
            .map(|transition| transition.orbital_angular_momentum),
    );
    let calculation_plan =
        xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), active_len)
            .context("failed to build XSPH NRIXS/JAS shared calculation plan")?;
    let lj_needed_by_calculation = (0..calculation_plan.calculations.nrows())
        .map(|row| {
            let calculation_index =
                i32::try_from(row + 1).context("XSPH NRIXS/JAS calculation index overflow")?;
            xsph_lj_needed_flags(
                indices.final_lj_max,
                final_lj.view(),
                calculation_plan.index_map.view(),
                active_len,
                calculation_index,
            )
            .with_context(|| {
                format!(
                    "failed to build XSPH NRIXS/JAS lj-needed flags for calculation {}",
                    row + 1
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let transitions = indices
        .transitions
        .iter()
        .map(|transition| XseclBinTransition {
            final_state_kappa: transition.final_state_kappa,
            decomposition_channel: transition.decomposition_channel,
            total_angular_momentum_channel: transition.total_angular_momentum_channel,
            orbital_angular_momentum: transition.orbital_angular_momentum,
        })
        .collect::<Vec<_>>();
    let (q_weights, q_cosines) = nrixs_q_weights_and_cosines(global, phase)?;
    let q_bessel = radii
        .map(|radii| nrixs_q_bessel_tables(global, radii, indices.final_lj_max))
        .transpose()?;
    let transition_weights = nrixs_transition_weight_table(
        core_hole.kappa,
        indices.initial_j2,
        max_angular_momentum,
        indices.final_j2_max,
        indices.final_lj_max,
        decomposition_l.view(),
        final_lj.view(),
        active_len,
    )?;

    Ok(NrixsSpectrumSourcePlan {
        initial_kappa: core_hole.kappa,
        initial_state_j: indices.initial_j2,
        max_angular_momentum,
        final_lj_max: indices.final_lj_max,
        final_state_count: indices.final_state_capacity,
        active_len,
        kind,
        decomposition_l,
        final_lj,
        orbital_l,
        calculation_plan,
        lj_needed_by_calculation,
        transitions,
        q_weights,
        q_cosines,
        q_bessel,
        transition_weights,
    })
}

#[allow(dead_code)]
fn nrixs_q_weights_and_cosines(
    global: &GlobalInput,
    phase: &PhaseBinData,
) -> Result<(Array1<Complex64>, Array2<f64>)> {
    ensure!(
        global.q_control.nq >= 0,
        "XSPH NRIXS/JAS global.inp nq must be nonnegative, got {}",
        global.q_control.nq
    );
    let q_count =
        usize::try_from(global.q_control.nq).context("XSPH NRIXS/JAS nq conversion failed")?;
    ensure!(
        q_count > 0,
        "XSPH NRIXS/JAS source plan requires at least one q-vector"
    );
    ensure!(
        global.q_vectors.len() == q_count,
        "global.inp q-vector count {} does not match nq {q_count}",
        global.q_vectors.len()
    );
    ensure!(
        phase.q_count == q_count,
        "phase.bin q count {} does not match global.inp q count {q_count}",
        phase.q_count
    );

    let q_weights = Array1::from_iter(
        global
            .q_vectors
            .iter()
            .map(|vector| Complex64::new(vector.weight[0], vector.weight[1])),
    );
    let q_cosines = if global.q_control.mixdff {
        let mdff = global
            .mdff
            .as_ref()
            .context("global.inp mixdff is enabled but MDFF cosine data is missing")?;
        let expected = q_count
            .checked_mul(q_count)
            .context("XSPH NRIXS/JAS q cosine matrix size overflow")?;
        ensure!(
            mdff.cosines.len() == expected,
            "global.inp MDFF cosine count {} does not match q-count square {expected}",
            mdff.cosines.len()
        );
        Array2::from_shape_fn((q_count, q_count), |(row, column)| {
            mdff.cosines[row * q_count + column]
        })
    } else {
        Array2::from_shape_fn((q_count, q_count), |(row, column)| {
            normalized_q_cosine(&global.q_vectors[row], &global.q_vectors[column])
        })
    };

    Ok((q_weights, q_cosines))
}

#[allow(dead_code)]
fn normalized_q_cosine(left: &refeff_io::GlobalQVector, right: &refeff_io::GlobalQVector) -> f64 {
    let norm_product = left.norm * right.norm;
    if norm_product <= 0.0 {
        return 0.0;
    }
    let dot = left
        .q
        .iter()
        .zip(right.q)
        .map(|(left, right)| *left * right)
        .sum::<f64>();
    (dot / norm_product).clamp(-1.0, 1.0)
}

#[allow(dead_code)]
fn nrixs_q_bessel_tables(
    global: &GlobalInput,
    radii: ArrayView1<'_, f64>,
    ljmax: usize,
) -> Result<Array3<f64>> {
    let q_count =
        usize::try_from(global.q_control.nq).context("XSPH NRIXS/JAS nq conversion failed")?;
    let mut q_bessel = Array3::<f64>::zeros((radii.len(), ljmax + 1, q_count));
    for (q_index, q_vector) in global.q_vectors.iter().enumerate() {
        let table = xsph_q_bessel_table(q_vector.norm, radii, ljmax).with_context(|| {
            format!(
                "failed to build XSPH NRIXS/JAS q-Bessel table for q-vector {}",
                q_index + 1
            )
        })?;
        for radius_index in 0..radii.len() {
            for angular in 0..=ljmax {
                q_bessel[(radius_index, angular, q_index)] = table[(radius_index, angular)];
            }
        }
    }
    Ok(q_bessel)
}

#[allow(dead_code)]
fn nrixs_transition_weight_table(
    initial_kappa: i32,
    initial_j2: i32,
    max_angular_momentum: usize,
    final_j2_max: i32,
    final_lj_max: usize,
    decomposition_l: ArrayView1<'_, i32>,
    final_lj: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<Array3<f64>> {
    ensure!(
        initial_j2 >= 0,
        "XSPH NRIXS/JAS initial doubled angular momentum must be nonnegative, got {initial_j2}"
    );
    let magnetic_count = usize::try_from(initial_j2)
        .context("XSPH NRIXS/JAS initial-state j conversion failed")?
        .checked_add(1)
        .context("XSPH NRIXS/JAS magnetic column count overflow")?;
    let mut weights = Array3::<f64>::zeros((2, active_len, magnetic_count));
    for magnetic_index in 0..magnetic_count {
        let initial_mj2 = -initial_j2
            + 2 * i32::try_from(magnetic_index)
                .context("XSPH NRIXS/JAS magnetic index conversion failed")?;
        let row = xsph_nrixs_transition_weights(
            initial_kappa,
            initial_mj2,
            max_angular_momentum,
            final_j2_max,
            i32::try_from(final_lj_max).context("XSPH NRIXS/JAS ljmax conversion failed")?,
            decomposition_l,
            final_lj,
            active_len,
        )
        .with_context(|| {
            format!("failed to build XSPH NRIXS/JAS transition weights for mj2={initial_mj2}")
        })?;
        for spin in 0..2 {
            for transition in 0..active_len {
                weights[(spin, transition, magnetic_index)] = row[(spin, transition)];
            }
        }
    }
    Ok(weights)
}

fn generate_nrixs_xsectjas_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<GeneratedNrixsSpectrum>> {
    const ABSORBER_INDEX: usize = 0;

    if !nrixs_xsectjas_requested(caches, input)?
        || tdlda_xsectd_branch_requested(input)
        || !jas_phase_mesh_spectroscopy_supported(input.control.ispec)
        || phase.spin_count != 1
    {
        return Ok(None);
    }
    let Some(global) = read_optional_global_input(&caches.global_inp)? else {
        return Ok(None);
    };
    if global.control.do_nrixs != 1 {
        return Ok(None);
    }
    if !caches.pot_bin.is_file() || !caches.config_dat.is_file() {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if !pot_uses_supported_normal_potentials(&pot)
        || !screened_core_hole_wscrn_handoff_is_supported(caches, &pot)
        || !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())?
        || pot
            .initial_large_component
            .iter()
            .chain(pot.initial_small_component.iter())
            .all(|value| *value == 0.0)
    {
        return Ok(None);
    }
    ensure!(
        pot.potential_count() == phase.potential_count(),
        "pot.bin potential count {} does not match phase.bin potential count {} for NRIXS/JAS xsectjas generation",
        pot.potential_count(),
        phase.potential_count()
    );

    let config = read_config_dat(&caches.config_dat)
        .with_context(|| format!("failed to read {}", caches.config_dat.display()))?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)
        .with_context(|| format!("failed to prepare {}", caches.config_dat.display()))?;
    ensure!(
        orbital_tables.bound_orbital_counts.len() == pot.potential_count(),
        "config.dat potential count {} does not match pot.bin potential count {}",
        orbital_tables.bound_orbital_counts.len(),
        pot.potential_count()
    );
    let bound_orbital_counts = pot_effective_bound_orbital_counts(&pot, &orbital_tables)?;
    if bound_orbital_counts.is_empty() {
        return Ok(None);
    }

    let muffin_tin_radii = pot
        .muffin_tin_radii
        .as_slice()
        .context("pot.bin muffin-tin radii are not contiguous")?;
    let total_potential = xsph_total_potential_with_screened_core_hole(caches, input, &pot)?;
    let prepared = xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
        muffin_tin_radii,
        electron_density: pot.electron_density.view(),
        total_potential: total_potential.view(),
        valence_density: pot.valence_density.view(),
        valence_potential: pot.valence_potential.view(),
        magnetization: pot.magnetization_density.view(),
        bound_large_components: pot.large_components.view(),
        bound_small_components: pot.small_components.view(),
        interstitial_potential: pot.scalars.interstitial_potential,
        interstitial_density: pot.scalars.interstitial_density,
        original_radial_dx: LOUCKS_DELTA,
        target_radial_dx: input.grid.rgrd,
        jump_mode: pot.jump_mode,
        potential_jump: 0.0,
        exchange_selector: input.control.ixc,
        radial_count: xsph_phase_radial_grid_count(&pot),
    })
    .context("failed to prepare XSPH normal-potential radial grids for NRIXS/JAS xsectjas")?;
    let excitation_poles = xsph_excitation_poles_from_loss(caches, input, input.control.ixc0)?;
    if input.control.i_plsmn > 0 && input.control.ixc0 == 0 && excitation_poles.is_none() {
        return Ok(None);
    }
    let broadened_table = load_xsph_broadened_table(&caches.work_dir, input.control.ixc0)?;

    generate_nrixs_xsectjas_sidecars_from_pot(
        input,
        &global,
        &pot,
        phase,
        &prepared,
        &orbital_tables,
        &bound_orbital_counts,
        excitation_poles.as_deref(),
        ABSORBER_INDEX,
        broadened_table.as_ref(),
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn generate_nrixs_xsectjas_sidecars_from_pot(
    input: &XsphInput,
    global: &GlobalInput,
    pot: &PotBinData,
    phase: &PhaseBinData,
    prepared: &XsphPhaseGridPreparation,
    orbital_tables: &RhorrpConfigOrbitalTables,
    bound_orbital_counts: &[usize],
    excitation_poles: Option<&[ExcitationPole]>,
    absorber_index: usize,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<GeneratedNrixsSpectrum> {
    ensure!(
        phase.energy_count == phase.energy_grid.len(),
        "phase.bin declares {} energy rows but stores {}",
        phase.energy_count,
        phase.energy_grid.len()
    );
    ensure!(
        phase.reference_energy.dim() == (phase.energy_count, phase.spin_count),
        "phase.bin reference-energy shape {:?} does not match declared ({}, {})",
        phase.reference_energy.dim(),
        phase.energy_count,
        phase.spin_count
    );
    ensure!(
        absorber_index < pot.potential_count(),
        "XSPH NRIXS/JAS absorber index {} exceeds pot.bin potential count {}",
        absorber_index,
        pot.potential_count()
    );
    ensure!(
        absorber_index < bound_orbital_counts.len(),
        "XSPH NRIXS/JAS effective bound-orbital counts are missing absorber index {}",
        absorber_index
    );
    ensure!(
        phase.spin_count == 1,
        "XSPH NRIXS/JAS source xsectjas generation requires one spin channel, got {}",
        phase.spin_count
    );

    let core_hole = core_hole_quantum_numbers(pot.ihole)
        .context("failed to determine XSPH NRIXS/JAS core-hole quantum numbers")?;
    let core_hole_l = usize::try_from(core_hole.angular_momentum)
        .context("XSPH NRIXS/JAS core-hole angular momentum is negative")?;
    let plan =
        nrixs_spectrum_source_plan_from_handoffs(input, global, phase, Some(prepared.radii.view()))
            .context("failed to plan XSPH NRIXS/JAS xsectjas source rows")?;
    ensure!(
        plan.q_weights.len() == phase.q_count,
        "XSPH NRIXS/JAS source q count {} does not match phase.bin q count {}",
        plan.q_weights.len(),
        phase.q_count
    );

    let muffin_tin_radius = pot.muffin_tin_radii[absorber_index];
    let radial_indices = xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
        muffin_tin_radius,
        grid_origin: XSPH_LOUCKS_GRID_ORIGIN,
        log_step: prepared.radial_dx,
        radial_capacity: prepared.radii.len(),
    })
    .context("failed to compute XSPH absorber radial indices for NRIXS/JAS xsectjas")?;
    let radial_match_index = radial_indices
        .radial_match_index_1based
        .checked_sub(1)
        .context("XSPH NRIXS/JAS radial match index underflow")?;
    let xcpot_active_len = radial_indices.reference_index_1based;
    let initial_spinor = xsph_initial_spinor_grid(pot, prepared, absorber_index, "NRIXS/JAS")?;
    let norman_index_1based = initial_spinor.norman_index_1based;

    let bound_orbital_count = bound_orbital_counts[absorber_index];
    ensure!(
        bound_orbital_count > 0,
        "config.dat absorber potential has no occupied orbitals"
    );
    ensure!(
        bound_orbital_count <= pot.orbital_occupancy.nrows(),
        "config.dat absorber bound-orbital count {bound_orbital_count} exceeds pot.bin orbital occupancy rows {}",
        pot.orbital_occupancy.nrows()
    );

    let radial_prefix = Slice::from(..xcpot_active_len);
    let orbital_prefix = Slice::from(..bound_orbital_count);
    let total_potential_column = prepared.total_potential.index_axis(Axis(1), absorber_index);
    let total_potential = total_potential_column.slice_axis(Axis(0), radial_prefix);
    let valence_potential_column = prepared
        .valence_potential
        .index_axis(Axis(1), absorber_index);
    let valence_potential = valence_potential_column.slice_axis(Axis(0), radial_prefix);
    let electron_density_column = prepared
        .electron_density
        .index_axis(Axis(1), absorber_index);
    let electron_density = electron_density_column.slice_axis(Axis(0), radial_prefix);
    let many_pole_self_energy = xsph_many_pole_self_energy_for_potential(
        input,
        excitation_poles,
        electron_density,
        xcpot_active_len,
        input.control.ixc0,
    )?;
    let magnetization_column = prepared.magnetization.index_axis(Axis(1), absorber_index);
    let magnetization = magnetization_column.slice_axis(Axis(0), radial_prefix);
    let valence_density_column = prepared.valence_density.index_axis(Axis(1), absorber_index);
    let valence_density = valence_density_column.slice_axis(Axis(0), radial_prefix);
    let bound_large_potential = prepared
        .bound_large_components
        .index_axis(Axis(2), absorber_index);
    let bound_large = bound_large_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_potential = prepared
        .bound_small_components
        .index_axis(Axis(2), absorber_index);
    let bound_small = bound_small_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_large_coefficients_potential =
        pot.large_coefficients.index_axis(Axis(2), absorber_index);
    let bound_large_coefficients =
        bound_large_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_coefficients_potential =
        pot.small_coefficients.index_axis(Axis(2), absorber_index);
    let bound_small_coefficients =
        bound_small_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let electron_counts_potential = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), absorber_index);
    let electron_counts = electron_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let valence_counts_potential = pot.orbital_occupancy.index_axis(Axis(1), absorber_index);
    let valence_counts = valence_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let kappa_potential = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), absorber_index);
    let kappa = kappa_potential.slice_axis(Axis(0), orbital_prefix);

    let mut rows = Vec::with_capacity(phase.energy_count);
    let mut transition_moments = Array4::<Complex64>::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    let mut spectrum_norms = Array2::<f64>::zeros((phase.energy_count, phase.spin_count));
    let mut cross_sections = Array2::<Complex64>::zeros((phase.energy_count, phase.spin_count));
    let mut fermi_cache: Option<Array1<XcpotFermiCache>> = None;
    let mut active_rows = 0_usize;
    let photon_energy_offset_hartree = xsph_chemical_potential_hartree(input, pot);

    for energy_index in 0..phase.energy_count {
        let xcpot_result = evaluate_xsph_xcpot(
            XcpotInput {
                exchange_selector: input.control.ixc0,
                lreal: input.control.lreal,
                energy: phase.energy_grid[energy_index],
                fermi_level: pot.scalars.fermi_level,
                total_potential,
                valence_potential,
                density: electron_density,
                magnetization,
                valence_density,
                active_len: xcpot_active_len,
                plasmon_selector: input.control.i_plsmn,
                many_pole_delta_table: None,
                many_pole_self_energy: many_pole_self_energy
                    .as_ref()
                    .map(|poles| poles.as_xcpot_input()),
                fermi_cache: fermi_cache.as_ref().map(|cache| cache.view()),
            },
            broadened_table,
        )
        .with_context(|| {
            format!(
                "failed to evaluate XSPH xcpot for NRIXS/JAS xsectjas row {}",
                energy_index + 1
            )
        })?;
        if !xcpot_result.fermi_cache.is_empty() {
            fermi_cache = Some(xcpot_result.fermi_cache.clone());
        }

        let setup = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
            energy: phase.energy_grid[energy_index],
            reference_energy: xcpot_result.reference_energy,
            edge_energy: phase.scalars.edge_energy,
            chemical_potential: photon_energy_offset_hartree,
            muffin_tin_radius,
            exchange_selector: input.control.ixc0,
            norman_index_1based,
            new_grid_index_1based: initial_spinor.active_len,
            radial_capacity: prepared.radii.len(),
        })
        .context("failed to set up XSPH NRIXS/JAS xsectjas energy row")?;
        if setup.decision != XsphXsectEnergyDecision::Active {
            rows.push(nrixs_zero_spectrum_row(&plan));
            continue;
        }
        ensure!(
            setup.active_radial_len <= initial_spinor.large.len()
                && setup.active_radial_len <= initial_spinor.small.len()
                && setup.active_radial_len <= electron_density_column.len(),
            "XSPH NRIXS/JAS active radial length {} exceeds core-hole spinor/density lengths ({}, {}, {})",
            setup.active_radial_len,
            initial_spinor.large.len(),
            initial_spinor.small.len(),
            electron_density_column.len()
        );
        let solver_total_potential =
            extend_xcpot_potential(&xcpot_result.total_potential, prepared.radii.len(), "total")?;
        let solver_valence_potential = if xcpot_result.valence_potential.is_empty() {
            solver_total_potential.clone()
        } else {
            extend_xcpot_potential(
                &xcpot_result.valence_potential,
                prepared.radii.len(),
                "valence",
            )?
        };
        let target_last_index = setup
            .active_radial_len
            .checked_sub(1)
            .context("XSPH NRIXS/JAS active radial length underflow")?;
        let calculation_count = plan.calculation_plan.calculations.nrows();
        let mut regular_channels = Vec::<XsphXsectRegularChannel>::with_capacity(calculation_count);
        let mut irregular_channels =
            Vec::<XsphXsectIrregularChannel>::with_capacity(calculation_count);

        for calculation_index in 0..calculation_count {
            let final_kappa = plan.calculation_plan.calculations[(calculation_index, 0)];
            let solver = FovrgDiracSolverInput {
                exchange_cycle_count: setup.cycle_count,
                target_kappa: final_kappa,
                muffin_tin_radius,
                target_last_index,
                energy: setup.momentum_squared,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: solver_total_potential.view(),
                valence_exchange_correlation_potential: solver_valence_potential.view(),
                bound_large_components: bound_large,
                bound_small_components: bound_small,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: Complex64::new(0.0, 0.0),
                muffin_tin_small_component: Complex64::new(0.0, 0.0),
                atomic_number: pot.atomic_numbers[absorber_index] as f64,
                irregular: false,
                c3_scale: 0,
                radial_match_index,
                bound_orbital_count,
            };
            let regular = xsph_xsect_regular_channel(XsphXsectRegularChannelInput {
                solver,
                wave_number: setup.wave_number,
            })
            .with_context(|| {
                format!(
                    "failed to solve XSPH NRIXS/JAS regular channel for row {}, calculation {}",
                    energy_index + 1,
                    calculation_index + 1
                )
            })?;
            let irregular = xsph_xsect_irregular_channel(XsphXsectIrregularChannelInput {
                solver,
                wave_number: setup.wave_number,
                regular_channel: &regular,
            })
            .with_context(|| {
                format!(
                    "failed to solve XSPH NRIXS/JAS irregular channel for row {}, calculation {}",
                    energy_index + 1,
                    calculation_index + 1
                )
            })?;
            regular_channels.push(regular);
            irregular_channels.push(irregular);
        }

        let radial_context_len = nrixs_solved_channel_active_len(
            setup.active_radial_len,
            &regular_channels,
            &irregular_channels,
        )?;
        let context = nrixs_spectrum_radial_source_context_from_handoffs(
            &plan,
            core_hole_l,
            initial_spinor.large.view(),
            initial_spinor.small.view(),
            prepared.radii.view(),
            prepared.radial_dx,
            norman_index_1based,
            radial_context_len,
        )
        .with_context(|| {
            format!(
                "failed to build XSPH NRIXS/JAS radial context for row {}",
                energy_index + 1
            )
        })?;

        let radial_channels = (0..calculation_count)
            .map(|calculation_index| NrixsSpectrumRadialChannel {
                final_kappa: plan.calculation_plan.calculations[(calculation_index, 0)],
                phase_shift: regular_channels[calculation_index].phase.phase_shift,
                regular_large: regular_channels[calculation_index]
                    .normalized_solution
                    .regular_large
                    .view(),
                regular_small: regular_channels[calculation_index]
                    .normalized_solution
                    .regular_small
                    .view(),
                irregular_large: irregular_channels[calculation_index]
                    .transformed_solution
                    .irregular_large
                    .view(),
                irregular_small: irregular_channels[calculation_index]
                    .transformed_solution
                    .irregular_small
                    .view(),
            })
            .collect::<Vec<_>>();
        let raw_row = nrixs_spectrum_row_from_radial_channels(
            &plan,
            &context,
            &radial_channels,
            0,
            global.q_control.mixdff,
            global.q_control.imdff,
        )
        .with_context(|| {
            format!(
                "failed to assemble XSPH NRIXS/JAS xsectjas spectrum row {}",
                energy_index + 1
            )
        })?;
        let row_transition_moments = nrixs_transition_moments_from_radial_channels(
            &plan,
            &context,
            &radial_channels,
            setup.wave_number,
            raw_row.total_spectrum_norm,
        )
        .with_context(|| {
            format!(
                "failed to assemble XSPH NRIXS/JAS transition moments for row {}",
                energy_index + 1
            )
        })?;
        for q_index in 0..phase.q_count {
            for transition_index in 0..phase.transition_count {
                transition_moments[(energy_index, q_index, transition_index, 0)] =
                    row_transition_moments[(q_index, transition_index)];
            }
        }
        let scaled_row =
            nrixs_scale_spectrum_row(raw_row, setup.wave_number, plan.initial_state_j)?;
        spectrum_norms[(energy_index, 0)] = scaled_row.total_spectrum_norm;
        cross_sections[(energy_index, 0)] = scaled_row
            .total_angular_cross_sections
            .iter()
            .copied()
            .sum::<Complex64>();
        rows.push(scaled_row);
        active_rows += 1;
    }

    ensure!(
        active_rows > 0,
        "XSPH NRIXS/JAS xsectjas generation produced no active energy rows"
    );
    let chemical_potential_ev = xsph_chemical_potential_hartree(input, pot);
    let handoffs = nrixs_spectrum_handoffs_from_rows(
        phase,
        &plan,
        &rows,
        chemical_potential_ev,
        input.grid.gamach / FEFF_HARTREE_EV,
    )?;
    let fermi_index =
        usize::try_from(phase.fermi_index).context("phase.bin fermi index is negative")?;
    let xsect_titles = xsph_xsect_material_header_titles(input, pot, phase.scalars.edge_energy)?;
    let generated = xsect_dat_from_xsph_spin_merge(XsectDatFromXsphSpinInput {
        titles: &xsect_titles,
        scalars: XsectDatScalars {
            amplitude_reduction: pot.scalars.amplitude_reduction,
            relaxation_energy: pot.scalars.relaxation_energy,
            plasmon_frequency: pot.scalars.plasmon_frequency,
            edge_energy: phase.scalars.edge_energy,
            chemical_potential: chemical_potential_ev,
        },
        core_hole_width_hartree: input.grid.gamach / FEFF_HARTREE_EV,
        main_energy_count: phase.main_energy_count,
        fermi_index,
        energy_grid_hartree: phase.energy_grid.view(),
        spin_polarized: false,
        spectrum_norms: spectrum_norms.view(),
        cross_sections: cross_sections.view(),
        transition_moments: transition_moments.view(),
        q_count: phase.q_count,
        transition_count: phase.transition_count,
    })
    .context("failed to merge XSPH NRIXS/JAS xsect rows")?;

    Ok(GeneratedNrixsSpectrum {
        xsect: generated.xsect,
        handoffs,
        transition_moments: generated.transition_moments,
    })
}

fn nrixs_zero_spectrum_row(plan: &NrixsSpectrumSourcePlan) -> NrixsSpectrumRowSource {
    let channel_count = plan.final_lj_max + 1;
    NrixsSpectrumRowSource {
        decomposition_cross_sections: Array1::zeros(channel_count),
        total_angular_cross_sections: Array1::zeros(channel_count),
        atom_cross_sections: Array1::zeros(plan.final_state_count),
        total_spectrum_norm: 0.0,
    }
}

fn nrixs_solved_channel_active_len(
    requested_active_len: usize,
    regular_channels: &[XsphXsectRegularChannel],
    irregular_channels: &[XsphXsectIrregularChannel],
) -> Result<usize> {
    let active_len = regular_channels
        .iter()
        .map(|channel| {
            channel
                .normalized_solution
                .regular_large
                .len()
                .min(channel.normalized_solution.regular_small.len())
        })
        .chain(irregular_channels.iter().map(|channel| {
            channel
                .transformed_solution
                .irregular_large
                .len()
                .min(channel.transformed_solution.irregular_small.len())
        }))
        .fold(requested_active_len, usize::min);
    ensure!(
        active_len > 0,
        "XSPH NRIXS/JAS solved radial channels produced no active rows"
    );
    Ok(active_len)
}

fn nrixs_scale_spectrum_row(
    mut row: NrixsSpectrumRowSource,
    wave_number: Complex64,
    initial_state_j: i32,
) -> Result<NrixsSpectrumRowSource> {
    let scale = nrixs_jas_output_scale(wave_number, initial_state_j)?;
    row.decomposition_cross_sections
        .iter_mut()
        .for_each(|value| *value *= scale);
    row.total_angular_cross_sections
        .iter_mut()
        .for_each(|value| *value *= scale);
    row.atom_cross_sections
        .iter_mut()
        .for_each(|value| *value *= scale);
    row.total_spectrum_norm *= scale.re;
    ensure!(
        row.total_spectrum_norm.is_finite(),
        "XSPH NRIXS/JAS scaled spectrum norm is not finite"
    );
    Ok(row)
}

fn nrixs_transition_moments_from_radial_channels(
    plan: &NrixsSpectrumSourcePlan,
    context: &NrixsSpectrumRadialSourceContext,
    channels: &[NrixsSpectrumRadialChannel<'_>],
    wave_number: Complex64,
    raw_spectrum_norm: f64,
) -> Result<Array2<Complex64>> {
    let q_count = plan.q_weights.len();
    ensure!(
        q_count > 0,
        "XSPH NRIXS/JAS transition moment assembly requires at least one q-vector"
    );
    ensure!(
        channels.len() == plan.calculation_plan.calculations.nrows(),
        "XSPH NRIXS/JAS transition moment channel count {} does not match calculation count {}",
        channels.len(),
        plan.calculation_plan.calculations.nrows()
    );
    let scale =
        nrixs_transition_moment_scale(wave_number, raw_spectrum_norm, plan.initial_state_j)?;
    let mut transition_moments = Array2::<Complex64>::zeros((q_count, plan.active_len));
    for (calculation_index, channel) in channels.iter().enumerate() {
        let expected_kappa = plan.calculation_plan.calculations[(calculation_index, 0)];
        ensure!(
            channel.final_kappa == expected_kappa,
            "XSPH NRIXS/JAS transition moment channel {} has kappa {}, expected {}",
            calculation_index + 1,
            channel.final_kappa,
            expected_kappa
        );
        let needed_multipoles = plan
            .lj_needed_by_calculation
            .get(calculation_index)
            .with_context(|| {
                format!(
                    "missing XSPH NRIXS/JAS lj-needed flags for calculation {}",
                    calculation_index + 1
                )
            })?;
        let calculation_index_1based = i32::try_from(calculation_index + 1)
            .context("XSPH NRIXS/JAS calculation index overflow")?;
        let regular_by_q = (0..q_count)
            .map(|q_index| {
                let q_bessel = context.q_bessel.index_axis(Axis(2), q_index);
                let orthogonality = context.orthogonality_correction.index_axis(Axis(1), q_index);
                xsph_jas_radial_integral(XsphJasRadialIntegralInput {
                    initial_kappa: plan.initial_kappa,
                    final_kappa: channel.final_kappa,
                    initial_large: context.initial_large.view(),
                    initial_small: context.initial_small.view(),
                    final_large_regular: channel.regular_large,
                    final_small_regular: channel.regular_small,
                    needed_multipoles: needed_multipoles.view(),
                    q_bessel,
                    orthogonality_correction: orthogonality,
                    radii: context.radii.view(),
                    log_step: context.log_step,
                    ljmax: plan.final_lj_max,
                    active_len: context.active_radial_len,
                })
                .with_context(|| {
                    format!(
                        "failed to evaluate XSPH NRIXS/JAS transition radial integrals for calculation {}, q {}",
                        calculation_index + 1,
                        q_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let phase_factor = (Complex64::new(0.0, 1.0) * channel.phase_shift).exp();
        nrixs_ensure_finite_complex("transition_phase_factor", calculation_index, phase_factor)?;
        for state_index in 0..plan.active_len {
            let mapped = plan.calculation_plan.index_map[state_index]
                .checked_abs()
                .context("XSPH NRIXS/JAS calculation index map overflow")?;
            if mapped != calculation_index_1based {
                continue;
            }
            let final_lj =
                nrixs_nonnegative_index("final_lj", state_index, plan.final_lj[state_index])?;
            ensure!(
                final_lj <= plan.final_lj_max,
                "XSPH NRIXS/JAS final_lj {} at state {} exceeds ljmax {}",
                final_lj,
                state_index + 1,
                plan.final_lj_max
            );
            for q_index in 0..q_count {
                transition_moments[(q_index, state_index)] =
                    regular_by_q[q_index].radial_integrals[final_lj] * phase_factor * scale;
            }
        }
    }
    Ok(transition_moments)
}

fn nrixs_transition_moment_scale(
    wave_number: Complex64,
    raw_spectrum_norm: f64,
    initial_state_j: i32,
) -> Result<Complex64> {
    let output_scale = nrixs_jas_output_scale(wave_number, initial_state_j)?;
    let scaled_norm = raw_spectrum_norm * output_scale.re;
    ensure!(
        scaled_norm.is_finite() && scaled_norm > 0.0,
        "XSPH NRIXS/JAS transition moment normalization must be positive, got {scaled_norm}"
    );
    let mut matrix_scale = output_scale.sqrt();
    if matrix_scale.im < 0.0 {
        matrix_scale = -matrix_scale;
    }
    nrixs_ensure_finite_complex("transition_matrix_scale", 0, matrix_scale)?;
    Ok(matrix_scale / scaled_norm.sqrt())
}

fn nrixs_jas_output_scale(wave_number: Complex64, initial_state_j: i32) -> Result<Complex64> {
    let prefactor = nrixs_jas_prefactor(initial_state_j)?;
    let scale = Complex64::new(prefactor * 2.0, 0.0) * wave_number;
    nrixs_ensure_finite_complex("jas_output_scale", 0, scale)?;
    Ok(scale)
}

fn nrixs_jas_prefactor(initial_state_j: i32) -> Result<f64> {
    ensure!(
        initial_state_j >= 0,
        "XSPH NRIXS/JAS initial-state j is negative: {initial_state_j}"
    );
    Ok(2.0 / std::f64::consts::PI / f64::from(initial_state_j + 1))
}

#[derive(Debug, Clone, PartialEq)]
struct XsphManyPoleSelfEnergy {
    pole_frequencies: Array1<f64>,
    pole_widths: Array1<f64>,
    amplitudes: Array1<f64>,
    gap_energy: f64,
}

impl XsphManyPoleSelfEnergy {
    fn from_setup(setup: XsphPhasePlasmonPoleSetup, gap_energy: f64) -> Self {
        let mut pole_frequencies = Vec::with_capacity(setup.poles.len() + 1);
        let mut pole_widths = Vec::with_capacity(setup.poles.len() + 1);
        let mut amplitudes = Vec::with_capacity(setup.poles.len() + 1);
        for pole in setup.poles {
            pole_frequencies.push(pole.energy_over_plasma);
            pole_widths.push(pole.width_hartree);
            amplitudes.push(pole.amplitude);
        }
        pole_frequencies.push(-1.0e30);
        pole_widths.push(0.0);
        amplitudes.push(0.0);
        Self {
            pole_frequencies: Array1::from_vec(pole_frequencies),
            pole_widths: Array1::from_vec(pole_widths),
            amplitudes: Array1::from_vec(amplitudes),
            gap_energy,
        }
    }

    fn as_xcpot_input(&self) -> XcpotManyPoleSelfEnergyInput<'_> {
        XcpotManyPoleSelfEnergyInput {
            pole_frequencies: self.pole_frequencies.view(),
            pole_widths: self.pole_widths.view(),
            amplitudes: self.amplitudes.view(),
            gap_energy: self.gap_energy,
            use_broadened_pole: false,
        }
    }

    fn renormalization(
        &self,
        energy: Complex64,
        fermi_level: f64,
        radius: f64,
    ) -> Result<Complex64> {
        let active_pole_count = self
            .pole_frequencies
            .iter()
            .position(|&frequency| frequency < -1.0)
            .unwrap_or(self.pole_frequencies.len());
        ensure!(
            active_pole_count > 0,
            "XSPH MPSE renormalization requires at least one active pole"
        );
        let frequency_scale = (3.0 / radius.powi(3)).sqrt();
        ensure!(
            frequency_scale.is_finite() && frequency_scale > 0.0,
            "XSPH MPSE renormalization frequency scale is invalid for radius {radius}"
        );
        let scaled_frequencies = Array1::from_iter(
            self.pole_frequencies
                .iter()
                .take(active_pole_count)
                .map(|&frequency| frequency * frequency_scale),
        );
        many_pole_self_energy(ManyPoleSelfEnergyInput {
            energy,
            fermi_level,
            radius,
            pole_frequencies: scaled_frequencies.view(),
            pole_widths: self.pole_widths.view(),
            amplitudes: self.amplitudes.view(),
            gap_energy: self.gap_energy,
            active_pole_count,
            use_broadened_pole: false,
        })
        .map(|sample| sample.renormalization)
        .context("failed to calculate XSPH mpse.dat CSigZ renormalization")
    }
}

fn xsph_excitation_poles_from_loss(
    caches: &XsphCachePaths,
    input: &XsphInput,
    exchange_selector: i32,
) -> Result<Option<Vec<ExcitationPole>>> {
    if input.control.i_plsmn <= 0 {
        return Ok(None);
    }
    if exchange_selector != 0 || !caches.loss_dat.is_file() {
        return Ok(None);
    }
    let requested_poles = usize::try_from(input.control.n_poles)
        .ok()
        .filter(|&count| count > 0)
        .context("XSPH MPSE generation requires a positive NPoles value in xsph.inp")?;
    let loss = read_loss_dat(&caches.loss_dat)
        .with_context(|| format!("failed to read {}", caches.loss_dat.display()))?;
    let poles = make_excitation_poles(
        loss.energy_ev.view(),
        loss.loss.view(),
        input.grid.eps0,
        requested_poles,
    )
    .context("failed to calculate XSPH MPSE excitation poles from loss.dat")?;
    Ok(Some(poles))
}

fn write_or_generate_xsph_excitation_poles_cache(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<(usize, bool)> {
    let Some(poles) = xsph_excitation_poles_from_loss(caches, input, input.control.ixc)? else {
        return Ok((0, false));
    };
    let generated = exc_dat_from_excitation_poles(&poles)
        .context("failed to assemble XSPH MPSE exc.dat handoff")?;
    let generated_text =
        exc_dat_string(&generated).context("failed to render XSPH MPSE exc.dat handoff")?;
    let changed = if caches.exc_dat.is_file() {
        match read_exc_dat(&caches.exc_dat) {
            Ok(cached) => {
                exc_dat_string(&cached)
                    .context("failed to normalize cached XSPH MPSE exc.dat handoff")?
                    != generated_text
            }
            Err(_) => true,
        }
    } else {
        true
    };

    write_exc_dat(&caches.exc_dat, &generated)
        .with_context(|| format!("failed to write {}", caches.exc_dat.display()))?;
    if changed && caches.specfunct_dat.is_file() {
        std::fs::remove_file(&caches.specfunct_dat).with_context(|| {
            format!(
                "failed to invalidate stale {} after XSPH excitation-pole refresh",
                caches.specfunct_dat.display()
            )
        })?;
    }
    Ok((1, changed))
}

fn xsph_many_pole_self_energy_for_potential(
    input: &XsphInput,
    excitation_poles: Option<&[ExcitationPole]>,
    electron_density: ArrayView1<'_, f64>,
    reference_index_1based: usize,
    exchange_selector: i32,
) -> Result<Option<XsphManyPoleSelfEnergy>> {
    let Some(excitation_poles) = excitation_poles else {
        return Ok(None);
    };
    let Some(setup) = xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
        plasmon_selector: input.control.i_plsmn,
        exchange_selector,
        electron_density,
        reference_index_1based,
        excitation_poles,
    })
    .context("failed to set up XSPH MPSE pole data")?
    else {
        return Ok(None);
    };
    Ok(Some(XsphManyPoleSelfEnergy::from_setup(
        setup,
        input.grid.egap / FEFF_HARTREE_EV,
    )))
}

fn generate_normal_potential_xsect_dat(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<GeneratedNormalXsect>> {
    if !caches.pot_bin.is_file() {
        return Ok(None);
    }
    if phase.q_count != 1
        || phase.transition_count == 0
        || phase.transition_count > PHASE_BIN_DEFAULT_TRANSITION_COUNT
        || !normal_xsect_spectroscopy_supported(input.control.ispec)
    {
        return Ok(None);
    }
    let Some(controls) = xsect_angular_controls(caches, input)? else {
        return Ok(None);
    };
    let advanced = normal_xsect_effective_advanced(input);
    if !normal_xsect_controls_supported(advanced, controls) {
        return Ok(None);
    }
    if !xsect_spin_state_supported(phase.spin_count, controls) {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
        || pot
            .atomic_numbers
            .iter()
            .any(|atomic_number| *atomic_number == 0)
    {
        return Ok(None);
    }
    if !screened_core_hole_wscrn_handoff_is_supported(caches, &pot) {
        return Ok(None);
    }
    if !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())? {
        return Ok(None);
    }
    ensure!(
        pot.potential_count() == phase.potential_count(),
        "pot.bin potential count {} does not match phase.bin potential count {} for xsect.dat generation",
        pot.potential_count(),
        phase.potential_count()
    );
    if pot
        .initial_large_component
        .iter()
        .chain(pot.initial_small_component.iter())
        .all(|value| *value == 0.0)
    {
        return Ok(None);
    }

    let Some(orbital_tables) = normal_potential_orbital_tables(caches, &pot)? else {
        return Ok(None);
    };
    ensure!(
        orbital_tables.bound_orbital_counts.len() == pot.potential_count(),
        "config.dat potential count {} does not match pot.bin potential count {}",
        orbital_tables.bound_orbital_counts.len(),
        pot.potential_count()
    );
    let bound_orbital_counts = pot_effective_bound_orbital_counts(&pot, &orbital_tables)?;
    if bound_orbital_counts.is_empty() {
        return Ok(None);
    }

    let muffin_tin_radii = pot
        .muffin_tin_radii
        .as_slice()
        .context("pot.bin muffin-tin radii are not contiguous")?;
    let spin_selectors = phase_spin_selectors(caches, input)?;
    ensure!(
        spin_selectors.len() == phase.spin_count,
        "XSPH xsect spin selector count {} does not match phase.bin spin count {}",
        spin_selectors.len(),
        phase.spin_count
    );
    let scaled_magnetization = xsph_scaled_magnetization(input, &pot)?;
    let ordinary_unpolarized =
        spin_selectors.len() == 1 && spin_selectors.first().copied() == Some(0);
    let mut prepared_by_spin = Vec::with_capacity(spin_selectors.len());
    for &spin_selector in &spin_selectors {
        let mut state = xsph_spin_ground_state(
            caches,
            input,
            &pot,
            &scaled_magnetization,
            spin_selector,
        )
        .with_context(|| {
            format!("failed to prepare XSPH xsect ground state for spin selector {spin_selector}")
        })?;
        apply_xsph_screened_core_hole(caches, input, &pot, &mut state.total_potential)?;
        prepared_by_spin.push(
            xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
                muffin_tin_radii,
                electron_density: pot.electron_density.view(),
                total_potential: state.total_potential.view(),
                valence_density: pot.valence_density.view(),
                valence_potential: state.valence_potential.view(),
                // Preserve the historical ordinary branch byte-for-byte.
                // Magnetic selectors use FEFF's spinph-scaled density.
                magnetization: if ordinary_unpolarized {
                    pot.magnetization_density.view()
                } else {
                    scaled_magnetization.view()
                },
                bound_large_components: pot.large_components.view(),
                bound_small_components: pot.small_components.view(),
                interstitial_potential: state.interstitial_potential,
                interstitial_density: state.interstitial_density,
                original_radial_dx: LOUCKS_DELTA,
                target_radial_dx: input.grid.rgrd,
                jump_mode: pot.jump_mode,
                potential_jump: 0.0,
                exchange_selector: input.control.ixc,
                radial_count: xsph_phase_radial_grid_count(&pot),
            })
            .with_context(|| {
                format!(
                    "failed to prepare XSPH normal-potential radial grids for xsect.dat spin selector {spin_selector}"
                )
            })?,
        );
    }
    let excitation_poles = xsph_excitation_poles_from_loss(caches, input, input.control.ixc0)?;
    if input.control.i_plsmn > 0 && input.control.ixc0 == 0 && excitation_poles.is_none() {
        return Ok(None);
    }
    let broadened_table = load_xsph_broadened_table(&caches.work_dir, input.control.ixc0)?;

    generate_normal_potential_xsect_dat_from_pot(
        input,
        &pot,
        phase,
        &prepared_by_spin,
        &orbital_tables,
        &bound_orbital_counts,
        excitation_poles.as_deref(),
        &spin_selectors,
        controls,
        broadened_table.as_ref(),
    )
    .map(Some)
}

fn generate_tdlda_pmbse_xsedge_dat(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<XsedgeDatData>> {
    if !tdlda_xsectd_branch_requested(input)
        || !matches!(phase.spin_count, 1 | 2)
        || phase.q_count != 1
    {
        return Ok(None);
    }
    let advanced = normal_xsect_effective_advanced(input);
    if !tdlda_nonlocal_source_is_supported(caches, advanced.nonlocal) {
        return Ok(None);
    }
    let spin_selectors = phase_spin_selectors(caches, input)?;
    if spin_selectors.len() != phase.spin_count {
        return Ok(None);
    }

    let mut spin_outputs = Vec::with_capacity(spin_selectors.len());
    for spin_selector in spin_selectors {
        let Some(output) =
            generate_tdlda_pmbse_xsedge_dat_for_spin(caches, input, phase, spin_selector)?
        else {
            return Ok(None);
        };
        spin_outputs.push(output);
    }
    tdlda_merge_spin_xsedge_outputs(spin_outputs).map(Some)
}

fn generate_tdlda_pmbse_xsedge_dat_for_spin(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
    spin_selector: i32,
) -> Result<Option<XsedgeDatData>> {
    const ABSORBER_INDEX: usize = 0;

    let advanced = normal_xsect_effective_advanced(input);
    if !caches.pot_bin.is_file() || !caches.config_dat.is_file() {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    if pot
        .atomic_numbers
        .iter()
        .all(|atomic_number| *atomic_number == 0)
        || pot
            .atomic_numbers
            .iter()
            .any(|atomic_number| *atomic_number == 0)
    {
        return Ok(None);
    }
    if !screened_core_hole_wscrn_handoff_is_supported(caches, &pot) {
        return Ok(None);
    }
    if !normal_potential_hubbard_phase_branch_is_supported(caches, pot.potential_count())? {
        return Ok(None);
    }
    ensure!(
        pot.potential_count() == phase.potential_count(),
        "pot.bin potential count {} does not match phase.bin potential count {} for TDLDA xsedge.dat generation",
        pot.potential_count(),
        phase.potential_count()
    );
    if pot
        .initial_large_component
        .iter()
        .chain(pot.initial_small_component.iter())
        .all(|value| *value == 0.0)
    {
        return Ok(None);
    }

    let config = read_config_dat(&caches.config_dat)
        .with_context(|| format!("failed to read {}", caches.config_dat.display()))?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)
        .with_context(|| format!("failed to prepare {}", caches.config_dat.display()))?;
    ensure!(
        orbital_tables.bound_orbital_counts.len() == pot.potential_count(),
        "config.dat potential count {} does not match pot.bin potential count {}",
        orbital_tables.bound_orbital_counts.len(),
        pot.potential_count()
    );
    if !pot_has_complete_bound_orbital_handoffs(&pot, &orbital_tables.bound_orbital_counts) {
        return Ok(None);
    }

    let Some(plan) = tdlda_xsectd_source_plan_from_caches(caches, input, &pot, &orbital_tables)?
    else {
        return Ok(None);
    };
    let projector_selectors = plan
        .basis
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            xsph_tdlda_decode_projector_selector(row.projector_orbital_selector).with_context(
                || format!("failed to decode XSPH TDLDA projector selector for row {row_index}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let generated_basis_count = projector_selectors
        .iter()
        .filter_map(|selector| match selector {
            XsphTdldaProjectorSelector::GeneratedBasis { basis_index, .. } => Some(basis_index + 1),
            XsphTdldaProjectorSelector::OccupiedOrbital { .. } => None,
        })
        .max()
        .unwrap_or(0);
    let has_generated_projectors = generated_basis_count > 0;

    let muffin_tin_radii = pot
        .muffin_tin_radii
        .as_slice()
        .context("pot.bin muffin-tin radii are not contiguous")?;
    let magnetization = xsph_scaled_magnetization(input, &pot)?;
    let mut spin_state =
        xsph_spin_ground_state(caches, input, &pot, &magnetization, spin_selector)?;
    apply_xsph_screened_core_hole(caches, input, &pot, &mut spin_state.total_potential)?;
    let prepared = xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
        muffin_tin_radii,
        electron_density: pot.electron_density.view(),
        total_potential: spin_state.total_potential.view(),
        valence_density: pot.valence_density.view(),
        valence_potential: spin_state.valence_potential.view(),
        magnetization: magnetization.view(),
        bound_large_components: pot.large_components.view(),
        bound_small_components: pot.small_components.view(),
        interstitial_potential: spin_state.interstitial_potential,
        interstitial_density: spin_state.interstitial_density,
        original_radial_dx: LOUCKS_DELTA,
        target_radial_dx: input.grid.rgrd,
        jump_mode: pot.jump_mode,
        potential_jump: 0.0,
        exchange_selector: input.control.ixc,
        radial_count: xsph_phase_radial_grid_count(&pot),
    })
    .context("failed to prepare XSPH normal-potential radial grids for TDLDA xsedge.dat")?;
    let Some(tdlda_core_hole_potential) =
        tdlda_core_hole_potential_from_source(caches, input, &pot, &prepared, advanced.nonlocal)?
    else {
        return Ok(None);
    };
    let excitation_poles = xsph_excitation_poles_from_loss(caches, input, input.control.ixc0)?;
    if input.control.i_plsmn > 0 && input.control.ixc0 == 0 && excitation_poles.is_none() {
        return Ok(None);
    }

    let muffin_tin_radius = pot.muffin_tin_radii[ABSORBER_INDEX];
    let radial_indices = xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
        muffin_tin_radius,
        grid_origin: XSPH_LOUCKS_GRID_ORIGIN,
        log_step: prepared.radial_dx,
        radial_capacity: prepared.radii.len(),
    })
    .context("failed to compute XSPH absorber radial indices for TDLDA xsedge.dat")?;
    let radial_match_index = radial_indices
        .radial_match_index_1based
        .checked_sub(1)
        .context("XSPH TDLDA radial match index underflow")?;
    let xcpot_active_len = radial_indices.reference_index_1based;
    // FEFF xsectd checks the interpolated core spinor norm but does not
    // rescale dgc0/dpc0 before evaluating dipole matrix elements.
    let initial_spinor =
        xsph_initial_spinor_grid(&pot, &prepared, ABSORBER_INDEX, "TDLDA xsedge.dat")?;
    let norman_index_1based = initial_spinor.norman_index_1based;
    let initial_large = initial_spinor.large;
    let initial_small = initial_spinor.small;

    let bound_orbital_count = orbital_tables.bound_orbital_counts[ABSORBER_INDEX];
    ensure!(
        bound_orbital_count > 0,
        "config.dat absorber potential has no occupied orbitals"
    );
    ensure!(
        bound_orbital_count <= pot.orbital_occupancy.nrows(),
        "config.dat absorber bound-orbital count {bound_orbital_count} exceeds pot.bin orbital occupancy rows {}",
        pot.orbital_occupancy.nrows()
    );

    let radial_prefix = Slice::from(..xcpot_active_len);
    let orbital_prefix = Slice::from(..bound_orbital_count);
    let total_potential_column = prepared.total_potential.index_axis(Axis(1), ABSORBER_INDEX);
    let total_potential = total_potential_column.slice_axis(Axis(0), radial_prefix);
    let valence_potential_column = prepared
        .valence_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let valence_potential = valence_potential_column.slice_axis(Axis(0), radial_prefix);
    let electron_density_column = prepared
        .electron_density
        .index_axis(Axis(1), ABSORBER_INDEX);
    let electron_density = electron_density_column.slice_axis(Axis(0), radial_prefix);
    let many_pole_self_energy = xsph_many_pole_self_energy_for_potential(
        input,
        excitation_poles.as_deref(),
        electron_density,
        xcpot_active_len,
        input.control.ixc0,
    )?;
    let magnetization_column = prepared.magnetization.index_axis(Axis(1), ABSORBER_INDEX);
    let magnetization = magnetization_column.slice_axis(Axis(0), radial_prefix);
    let valence_density_column = prepared.valence_density.index_axis(Axis(1), ABSORBER_INDEX);
    let valence_density = valence_density_column.slice_axis(Axis(0), radial_prefix);
    let bound_large_potential = prepared
        .bound_large_components
        .index_axis(Axis(2), ABSORBER_INDEX);
    let bound_large = bound_large_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_potential = prepared
        .bound_small_components
        .index_axis(Axis(2), ABSORBER_INDEX);
    let bound_small = bound_small_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_large_coefficients_potential =
        pot.large_coefficients.index_axis(Axis(2), ABSORBER_INDEX);
    let bound_large_coefficients =
        bound_large_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_coefficients_potential =
        pot.small_coefficients.index_axis(Axis(2), ABSORBER_INDEX);
    let bound_small_coefficients =
        bound_small_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let electron_counts_potential = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let electron_counts = electron_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let valence_counts_potential = pot.orbital_occupancy.index_axis(Axis(1), ABSORBER_INDEX);
    let valence_counts = valence_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let kappa_potential = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let kappa = kappa_potential.slice_axis(Axis(0), orbital_prefix);

    let energy_count = plan.multipliers.energy_hartree.len();
    let mut reference_energy = Array1::<Complex64>::zeros(energy_count);
    let mut raw_response = Array3::<f64>::zeros((energy_count, plan.matrix_size, plan.matrix_size));
    let mut localized_dipoles = Array2::<f64>::zeros((energy_count, plan.matrix_size));
    let mut full_dipoles = Array2::<f64>::zeros((energy_count, plan.matrix_size));
    let mut kernel = Array3::<Complex64>::zeros((energy_count, plan.matrix_size, plan.matrix_size));
    let mut projected_kernel =
        Array3::<Complex64>::zeros((energy_count, plan.matrix_size, plan.matrix_size));
    let separation_function = xsph_tdlda_separation_function(
        advanced.ipmbse,
        plan.multipliers.energy_hartree.view(),
        energy_count,
    )
    .context("failed to prepare XSPH TDLDA PMBSE separation function")?;
    let mut fermi_cache: Option<Array1<XcpotFermiCache>> = None;
    let mut active_rows = 0_usize;
    let chemical_potential = xsph_chemical_potential_hartree(input, &pot);
    let broadened_table = load_xsph_broadened_table(&caches.work_dir, input.control.ixc0)?;
    let generated_projector_candidates = if has_generated_projectors {
        let candidates = if advanced.ibasis == 1 {
            tdlda_file_projector_candidates_from_source(XsphTdldaFileProjectorCandidatesInput {
                work_dir: &caches.work_dir,
                plan: &plan,
                generated_basis_count,
                active_len: prepared.radii.len(),
                file_target_last_index: norman_index_1based,
                radii: prepared.radii.view(),
            })?
        } else {
            Some(tdlda_generated_projector_candidates_from_source(
                XsphTdldaGeneratedProjectorCandidatesInput {
                    input,
                    broadened_table: broadened_table.as_ref(),
                    plan: &plan,
                    generated_basis_count,
                    active_len: prepared.radii.len(),
                    generated_target_last_index: norman_index_1based,
                    xcpot_active_len,
                    muffin_tin_radius,
                    radial_match_index,
                    total_potential,
                    valence_potential,
                    electron_density,
                    magnetization,
                    valence_density,
                    many_pole_self_energy: many_pole_self_energy.as_ref(),
                    fermi_level: pot.scalars.fermi_level,
                    radii: prepared.radii.view(),
                    radial_dx: prepared.radial_dx,
                    bound_large,
                    bound_small,
                    bound_large_coefficients,
                    bound_small_coefficients,
                    electron_counts,
                    valence_counts,
                    kappa,
                    atomic_number: pot.atomic_numbers[ABSORBER_INDEX] as f64,
                    bound_orbital_count,
                },
            )?)
        };
        let Some(candidates) = candidates else {
            return Ok(None);
        };
        Some(candidates)
    } else {
        None
    };

    for (energy_index, &energy_hartree) in plan.multipliers.energy_hartree.iter().enumerate() {
        let xcpot_result = evaluate_xsph_xcpot(
            XcpotInput {
                exchange_selector: input.control.ixc0,
                lreal: input.control.lreal,
                energy: Complex64::new(energy_hartree, 0.0),
                fermi_level: pot.scalars.fermi_level,
                total_potential,
                valence_potential,
                density: electron_density,
                magnetization,
                valence_density,
                active_len: xcpot_active_len,
                plasmon_selector: input.control.i_plsmn,
                many_pole_delta_table: None,
                many_pole_self_energy: many_pole_self_energy
                    .as_ref()
                    .map(|poles| poles.as_xcpot_input()),
                fermi_cache: fermi_cache.as_ref().map(|cache| cache.view()),
            },
            broadened_table.as_ref(),
        )
        .with_context(|| {
            format!(
                "failed to evaluate XSPH TDLDA xcpot for xsedge.dat row {}",
                energy_index + 1
            )
        })?;
        if !xcpot_result.fermi_cache.is_empty() {
            fermi_cache = Some(xcpot_result.fermi_cache.clone());
        }
        reference_energy[energy_index] = xcpot_result.reference_energy;

        let setup = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
            energy: Complex64::new(energy_hartree, 0.0),
            reference_energy: xcpot_result.reference_energy,
            edge_energy: phase.scalars.edge_energy,
            chemical_potential,
            muffin_tin_radius,
            exchange_selector: input.control.ixc0,
            norman_index_1based,
            new_grid_index_1based: initial_spinor.active_len,
            radial_capacity: prepared.radii.len(),
        })
        .context("failed to set up XSPH TDLDA xsedge.dat energy row")?;
        if setup.decision != XsphXsectEnergyDecision::Active {
            continue;
        }
        ensure!(
            setup.active_radial_len <= initial_large.len()
                && setup.active_radial_len <= initial_small.len(),
            "XSPH TDLDA active radial length {} exceeds core-hole spinor lengths ({}, {})",
            setup.active_radial_len,
            initial_large.len(),
            initial_small.len()
        );
        let solver_total_potential =
            extend_xcpot_potential(&xcpot_result.total_potential, prepared.radii.len(), "total")?;
        let solver_valence_potential = if xcpot_result.valence_potential.is_empty() {
            solver_total_potential.clone()
        } else {
            extend_xcpot_potential(
                &xcpot_result.valence_potential,
                prepared.radii.len(),
                "valence",
            )?
        };
        let target_last_index = setup
            .active_radial_len
            .checked_sub(1)
            .context("XSPH TDLDA active radial length underflow")?;
        let mut regular_channels = Vec::<XsphXsectRegularChannel>::with_capacity(plan.matrix_size);
        for (row_index, row) in plan.basis.rows.iter().enumerate() {
            let solver = FovrgDiracSolverInput {
                exchange_cycle_count: setup.cycle_count,
                target_kappa: row.final_kappa,
                muffin_tin_radius,
                target_last_index,
                energy: setup.momentum_squared,
                step: prepared.radial_dx,
                radii: prepared.radii.view(),
                exchange_correlation_potential: solver_total_potential.view(),
                valence_exchange_correlation_potential: solver_valence_potential.view(),
                bound_large_components: bound_large,
                bound_small_components: bound_small,
                bound_large_coefficients,
                bound_small_coefficients,
                electron_counts,
                valence_counts,
                kappa,
                muffin_tin_large_component: Complex64::new(0.0, 0.0),
                muffin_tin_small_component: Complex64::new(0.0, 0.0),
                atomic_number: pot.atomic_numbers[ABSORBER_INDEX] as f64,
                irregular: false,
                c3_scale: 0,
                radial_match_index,
                bound_orbital_count,
            };
            let regular = xsph_xsect_regular_channel(XsphXsectRegularChannelInput {
                solver,
                wave_number: setup.wave_number,
            })
            .with_context(|| {
                format!(
                    "failed to solve XSPH TDLDA regular row {} for xsedge.dat energy row {}",
                    row_index + 1,
                    energy_index + 1
                )
            })?;
            regular_channels.push(regular);
        }
        let effective_active_len =
            regular_channels
                .iter()
                .fold(setup.active_radial_len, |current, regular| {
                    current
                        .min(regular.normalized_solution.regular_large.len())
                        .min(regular.normalized_solution.regular_small.len())
                });
        ensure!(
            effective_active_len >= 4,
            "XSPH TDLDA xsedge.dat generation requires at least four returned radial rows, got {effective_active_len}"
        );
        let coefficient_count = regular_channels.iter().fold(
            bound_large_coefficients
                .nrows()
                .min(bound_small_coefficients.nrows()),
            |current, regular| {
                current
                    .min(regular.regular_solution.large_coefficients.len())
                    .min(regular.regular_solution.small_coefficients.len())
            },
        );
        ensure!(
            coefficient_count > 0,
            "XSPH TDLDA xsedge.dat generation requires FOVRG origin coefficients"
        );
        let source_len = regular_channels
            .iter()
            .fold(effective_active_len - 1, |current, regular| {
                current.min(regular.regular_solution.retained_len)
            })
            .min(effective_active_len - 1);
        ensure!(
            source_len > 0,
            "XSPH TDLDA xsedge.dat generation requires a nonempty yzktd source length"
        );
        let first_regular = regular_channels
            .first()
            .context("XSPH TDLDA xsedge.dat generation produced no regular rows")?;
        ensure!(
            first_regular.regular_solution.origin_powers.len() > bound_orbital_count
                && first_regular.regular_solution.orbital_lengths.len() > bound_orbital_count,
            "XSPH TDLDA regular solution metadata cannot supply {bound_orbital_count} bound orbitals plus target"
        );
        let orbital_powers = first_regular
            .regular_solution
            .origin_powers
            .slice_axis(Axis(0), Slice::from(..bound_orbital_count))
            .to_owned();
        let orbital_lengths = first_regular
            .regular_solution
            .orbital_lengths
            .slice_axis(Axis(0), Slice::from(..bound_orbital_count))
            .to_owned();
        let active_prefix = Slice::from(..effective_active_len);
        let active_initial_large = initial_large.slice_axis(Axis(0), active_prefix);
        let active_initial_small = initial_small.slice_axis(Axis(0), active_prefix);
        let active_radii = prepared.radii.slice_axis(Axis(0), active_prefix);
        let active_electron_density = electron_density_column.slice_axis(Axis(0), active_prefix);
        let core_hole_potential = tdlda_core_hole_potential
            .slice_axis(Axis(0), active_prefix)
            .to_owned();
        let xray_bessel = xsph_xray_bessel_table(XsphXrayBesselTableInput {
            photon_wave_number: setup.photon_wave_number,
            radii: prepared.radii.view(),
            active_len: effective_active_len,
        })
        .context("failed to build XSPH TDLDA photon Bessel table")?;
        let mut full_large = Array2::<f64>::zeros((effective_active_len, plan.matrix_size));
        let mut full_small = Array2::<f64>::zeros((effective_active_len, plan.matrix_size));
        let mut full_large_complex =
            Array2::<Complex64>::zeros((effective_active_len, plan.matrix_size));
        let mut full_small_complex =
            Array2::<Complex64>::zeros((effective_active_len, plan.matrix_size));
        let mut target_large_coefficients =
            Array2::<Complex64>::zeros((coefficient_count, plan.matrix_size));
        let mut target_small_coefficients =
            Array2::<Complex64>::zeros((coefficient_count, plan.matrix_size));
        let mut target_powers = Array1::<f64>::zeros(plan.matrix_size);
        for row_index in 0..plan.matrix_size {
            let regular = &regular_channels[row_index];
            ensure!(
                regular.regular_solution.origin_powers.len() > bound_orbital_count,
                "XSPH TDLDA regular row {} cannot supply target origin power",
                row_index + 1
            );
            target_powers[row_index] = regular.regular_solution.origin_powers[bound_orbital_count];
            let coefficient_scale = regular.normalized_solution.regular_solution_scale;
            for coefficient in 0..coefficient_count {
                target_large_coefficients[(coefficient, row_index)] =
                    regular.regular_solution.large_coefficients[coefficient] * coefficient_scale;
                target_small_coefficients[(coefficient, row_index)] =
                    regular.regular_solution.small_coefficients[coefficient] * coefficient_scale;
            }
            for radial in 0..effective_active_len {
                let large = regular.normalized_solution.regular_large[radial];
                let small = regular.normalized_solution.regular_small[radial];
                full_large_complex[(radial, row_index)] = large;
                full_small_complex[(radial, row_index)] = small;
                full_large[(radial, row_index)] = large.re;
                full_small[(radial, row_index)] = small.re;
            }
        }
        let norman_radius = prepared.radii[norman_index_1based - 1];
        let active_generated_large = generated_projector_candidates
            .as_ref()
            .map(|(large, _)| large.slice_axis(Axis(0), Slice::from(..effective_active_len)));
        let active_generated_small = generated_projector_candidates
            .as_ref()
            .map(|(_, small)| small.slice_axis(Axis(0), Slice::from(..effective_active_len)));
        let projectors = match tdlda_projector_rows_from_source_plan(
            &plan,
            effective_active_len,
            prepared.radial_dx,
            norman_radius,
            prepared.radii.view(),
            bound_large,
            bound_small,
            active_generated_large,
            active_generated_small,
        ) {
            Ok(projectors) => projectors,
            Err(error)
                if advanced.ibasis == 1 && is_tdlda_file_projector_degenerate_error(&error) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        ensure!(
            projectors.source_rows.iter().all(|source_row| *source_row),
            "XSPH TDLDA xsedge.dat generation requires all projector rows from source-backed candidates"
        );
        let localized_large_complex = projectors
            .localized_large
            .mapv(|value| Complex64::new(value, 0.0));
        let localized_small_complex = projectors
            .localized_small
            .mapv(|value| Complex64::new(value, 0.0));

        let raw_inputs = tdlda_raw_response_inputs_from_source_plan(
            &plan,
            effective_active_len,
            prepared.radial_dx,
            active_radii,
            active_initial_large,
            active_initial_small,
            xray_bessel.values.view(),
            projectors.localized_large.view(),
            projectors.localized_small.view(),
            full_large.view(),
            full_small.view(),
        )?;
        let raw = tdlda_raw_response_from_source_plan(
            &plan,
            energy_hartree,
            xcpot_result.reference_energy,
            phase.scalars.edge_energy,
            raw_inputs.overlaps.view(),
            raw_inputs.localized_dipoles.view(),
            raw_inputs.full_dipoles.view(),
        )?;
        for row in 0..plan.matrix_size {
            localized_dipoles[(energy_index, row)] = raw.localized_dipoles[row];
            full_dipoles[(energy_index, row)] = raw.full_dipoles[row];
            for column in 0..plan.matrix_size {
                raw_response[(energy_index, row, column)] =
                    raw.raw_imaginary_response[(row, column)];
            }
        }
        let row_wave_numbers = tdlda_row_wave_numbers_from_source_plan(
            &plan,
            energy_hartree,
            xcpot_result.reference_energy,
        )?;
        let local_field = xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
            exchange_correlation_selector: advanced.ifxc,
            radii: active_radii,
            electron_density: active_electron_density,
            active_len: effective_active_len,
        })
        .context("failed to prepare XSPH TDLDA getchi0 local exchange field")?;
        let exchange_correlation_imaginary = Array1::<f64>::zeros(effective_active_len);
        let coulomb_fields = tdlda_coulomb_fields_from_source_plan(
            &plan,
            effective_active_len,
            source_len,
            coefficient_count,
            prepared.radial_dx,
            1,
            active_radii,
            bound_large,
            bound_small,
            bound_large_coefficients,
            bound_small_coefficients,
            orbital_powers.view(),
            orbital_lengths.view(),
            full_large_complex.view(),
            full_small_complex.view(),
            target_large_coefficients.view(),
            target_small_coefficients.view(),
            target_powers.view(),
        )?;
        let nonlocal_exchange = if advanced.ifxc == 5 {
            Some(tdlda_nonlocal_exchange_from_source_plan(
                &plan,
                &row_wave_numbers,
                effective_active_len,
                source_len,
                coefficient_count,
                prepared.radial_dx,
                2,
                1.0 - separation_function[energy_index],
                active_radii,
                bound_large,
                bound_small,
                bound_large_coefficients,
                bound_small_coefficients,
                orbital_powers.view(),
                orbital_lengths.view(),
                localized_large_complex.view(),
                localized_small_complex.view(),
                full_large_complex.view(),
                full_small_complex.view(),
            )?)
        } else {
            None
        };
        let nonlocal_radial = nonlocal_exchange
            .as_ref()
            .map(|exchange| exchange.radial_integrals.view());
        let nonlocal_projected = nonlocal_exchange
            .as_ref()
            .map(|exchange| exchange.projected_radial_integrals.view());
        let getchi0 = tdlda_getchi0_kernel_from_source_plan(
            &plan,
            &row_wave_numbers,
            advanced.ifxc,
            1.0 - separation_function[energy_index],
            energy_hartree,
            phase.scalars.edge_energy,
            separation_function[energy_index],
            effective_active_len,
            active_radii,
            core_hole_potential.view(),
            local_field.values.view(),
            local_field.values.view(),
            exchange_correlation_imaginary.view(),
            projectors.localized_large.view(),
            projectors.localized_small.view(),
            full_large.view(),
            full_small.view(),
            full_large_complex.view(),
            full_small_complex.view(),
            localized_large_complex.view(),
            localized_small_complex.view(),
            full_large_complex.view(),
            full_small_complex.view(),
            coulomb_fields.fields.view(),
            nonlocal_radial,
            nonlocal_projected,
        )?;
        for row in 0..plan.matrix_size {
            for column in 0..plan.matrix_size {
                kernel[(energy_index, row, column)] = getchi0.kernel[(row, column)];
                projected_kernel[(energy_index, row, column)] =
                    getchi0.projected_kernel[(row, column)];
            }
        }
        active_rows += 1;
    }

    ensure!(
        active_rows > 0,
        "XSPH TDLDA xsedge.dat generation produced no active energy rows"
    );
    let energy_rows = tdlda_energy_rows_from_source_plan(
        &plan,
        input,
        reference_energy.view(),
        phase.scalars.edge_energy,
        chemical_potential,
    )?;
    let mut xsedge = tdlda_xsedge_dat_from_raw_source_components(
        &plan,
        &energy_rows,
        raw_response.view(),
        localized_dipoles.view(),
        full_dipoles.view(),
        kernel.view(),
        projected_kernel.view(),
        phase.scalars.edge_energy,
        chemical_potential,
    )?;
    xsedge
        .energy_ev
        .mapv_inplace(|energy_ev| energy_ev + chemical_potential * FEFF_HARTREE_EV);
    Ok(Some(xsedge))
}

fn tdlda_nonlocal_source_is_supported(caches: &XsphCachePaths, nonlocal: i32) -> bool {
    match nonlocal {
        0 => true,
        1 => caches.pot_ch.is_file(),
        2 => caches.yoshi_dat.is_file() || caches.wscrn_dat.is_file(),
        _ => false,
    }
}

fn tdlda_core_hole_potential_from_source(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
    prepared: &XsphPhaseGridPreparation,
    nonlocal: i32,
) -> Result<Option<Array1<f64>>> {
    const ABSORBER_INDEX: usize = 0;
    if nonlocal == 0 {
        return Ok(Some(Array1::zeros(prepared.radii.len())));
    }

    let source_values = match nonlocal {
        1 => {
            if !caches.pot_ch.is_file() {
                return Ok(None);
            }
            let screened = read_pot_bin(&caches.pot_ch)
                .with_context(|| format!("failed to read {}", caches.pot_ch.display()))?;
            ensure!(
                screened.potential_count() > ABSORBER_INDEX,
                "XSPH TDLDA pot.ch contains no absorber potential"
            );
            screened.total_potential.column(ABSORBER_INDEX).to_owned()
        }
        2 => {
            let screened = if caches.yoshi_dat.is_file() {
                read_tdlda_yoshi_potential(&caches.yoshi_dat)?
            } else if caches.wscrn_dat.is_file() {
                read_wscrn_dat(&caches.wscrn_dat)
                    .with_context(|| format!("failed to read {}", caches.wscrn_dat.display()))?
                    .screened_potential
            } else {
                return Ok(None);
            };
            screened.mapv(|value| -value)
        }
        _ => return Ok(None),
    };

    let source_len = pot.total_potential.nrows();
    ensure!(
        source_len > 0,
        "XSPH TDLDA nonlocal core-hole source requires a nonempty radial grid"
    );
    let mut source_on_pot_grid = Array1::<f64>::zeros(source_len);
    for (target, source) in source_on_pot_grid.iter_mut().zip(source_values.iter()) {
        *target = *source;
    }
    let fixed = fix_potential_grid(PotentialGridInput {
        muffin_tin_radius: pot.muffin_tin_radii[ABSORBER_INDEX],
        electron_density: pot.electron_density.column(ABSORBER_INDEX),
        total_potential: source_on_pot_grid.view(),
        magnetization: pot.magnetization_density.column(ABSORBER_INDEX),
        interstitial_potential: pot.scalars.interstitial_potential,
        interstitial_density: pot.scalars.interstitial_density,
        original_delta: LOUCKS_DELTA,
        new_delta: input.grid.rgrd,
        jump_mode: pot.jump_mode,
        potential_jump: 0.0,
        output_len: prepared.radii.len(),
    })
    .context("failed to interpolate XSPH TDLDA nonlocal core-hole potential")?;
    let mut result = fixed.total_potential;
    let muffin_tin_radius = pot.muffin_tin_radii[ABSORBER_INDEX];
    for radial in 0..result.len() {
        let radius = prepared.radii[radial];
        if radius < muffin_tin_radius {
            if nonlocal == 1 {
                result[radial] -= prepared.total_potential[(radial, ABSORBER_INDEX)];
            }
        } else if radius < 40.0 && radial > 0 {
            result[radial] = result[radial - 1] * prepared.radii[radial - 1] / radius;
        } else {
            result[radial] = 0.0;
        }
    }
    Ok(Some(result))
}

fn read_tdlda_yoshi_potential(path: &Path) -> Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut values = Vec::new();
    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        ensure!(
            fields.len() >= 2,
            "{} row {} must contain radius and screened potential",
            path.display(),
            line_index + 1
        );
        let value = fields[1]
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .with_context(|| {
                format!(
                    "failed to parse screened potential in {} row {}",
                    path.display(),
                    line_index + 1
                )
            })?;
        ensure!(
            value.is_finite(),
            "{} row {} screened potential is non-finite",
            path.display(),
            line_index + 1
        );
        values.push(value);
    }
    ensure!(
        !values.is_empty(),
        "{} contains no screened-potential rows",
        path.display()
    );
    Ok(Array1::from_vec(values))
}

fn tdlda_merge_spin_xsedge_outputs(mut outputs: Vec<XsedgeDatData>) -> Result<XsedgeDatData> {
    let spin_count = outputs.len();
    ensure!(
        matches!(spin_count, 1 | 2),
        "XSPH TDLDA spin merge requires one or two source rows, got {spin_count}"
    );
    let mut merged = outputs.remove(0);
    for output in outputs {
        ensure!(
            output.row_count() == merged.row_count()
                && output.has_branch_columns() == merged.has_branch_columns(),
            "XSPH TDLDA spin outputs have incompatible xsedge.dat shapes"
        );
        for row in 0..merged.row_count() {
            ensure!(
                (output.energy_ev[row] - merged.energy_ev[row]).abs() <= 1.0e-8,
                "XSPH TDLDA spin output energy grids differ at row {}",
                row + 1
            );
            merged.total_single_particle[row] += output.total_single_particle[row];
            merged.total_screened[row] += output.total_screened[row];
        }
        tdlda_add_optional_spin_column(
            merged.plus_branch_single_particle.as_mut(),
            output.plus_branch_single_particle.as_ref(),
        )?;
        tdlda_add_optional_spin_column(
            merged.minus_branch_single_particle.as_mut(),
            output.minus_branch_single_particle.as_ref(),
        )?;
        tdlda_add_optional_spin_column(
            merged.plus_branch_screened.as_mut(),
            output.plus_branch_screened.as_ref(),
        )?;
        tdlda_add_optional_spin_column(
            merged.minus_branch_screened.as_mut(),
            output.minus_branch_screened.as_ref(),
        )?;
    }
    let scale = 1.0 / spin_count as f64;
    merged
        .total_single_particle
        .mapv_inplace(|value| value * scale);
    merged.total_screened.mapv_inplace(|value| value * scale);
    for column in [
        merged.plus_branch_single_particle.as_mut(),
        merged.minus_branch_single_particle.as_mut(),
        merged.plus_branch_screened.as_mut(),
        merged.minus_branch_screened.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        column.mapv_inplace(|value| value * scale);
    }
    Ok(merged)
}

fn tdlda_add_optional_spin_column(
    target: Option<&mut Array1<f64>>,
    source: Option<&Array1<f64>>,
) -> Result<()> {
    match (target, source) {
        (Some(target), Some(source)) => {
            ensure!(
                target.len() == source.len(),
                "XSPH TDLDA spin output branch lengths differ"
            );
            *target += source;
            Ok(())
        }
        (None, None) => Ok(()),
        _ => bail!("XSPH TDLDA spin outputs disagree on branch columns"),
    }
}

fn is_tdlda_file_projector_degenerate_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<XsphError>(),
            Some(XsphError::InvalidPositiveScalar { name, .. }) if *name == "tdlda_projector_norm"
        )
    })
}

fn generate_normal_potential_xsect_dat_from_pot(
    input: &XsphInput,
    pot: &PotBinData,
    phase: &PhaseBinData,
    prepared_by_spin: &[XsphPhaseGridPreparation],
    orbital_tables: &RhorrpConfigOrbitalTables,
    bound_orbital_counts: &[usize],
    excitation_poles: Option<&[ExcitationPole]>,
    spin_selectors: &[i32],
    controls: XsphXsectAngularControls,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<GeneratedNormalXsect> {
    if !controls.combined_higher_multipoles {
        return generate_normal_potential_xsect_dat_from_pot_single_multipole(
            input,
            pot,
            phase,
            prepared_by_spin,
            orbital_tables,
            bound_orbital_counts,
            excitation_poles,
            spin_selectors,
            controls,
            broadened_table,
        );
    }

    // FEFF's le2=3 contract is the union E1+M1+E2, while bcoef and xsect's
    // radial loop encode only one higher multipole per pass. Accumulate the
    // two supported higher-multipole passes and subtract their shared E1
    // contribution once. This preserves the polarized bcoef result for each
    // branch and keeps the eight-slot phase/xsect handoff layout.
    let dipole = generate_normal_potential_xsect_dat_from_pot_single_multipole(
        input,
        pot,
        phase,
        prepared_by_spin,
        orbital_tables,
        bound_orbital_counts,
        excitation_poles,
        spin_selectors,
        controls.for_single_higher_multipole(0)?,
        broadened_table,
    )?;
    let magnetic_dipole = generate_normal_potential_xsect_dat_from_pot_single_multipole(
        input,
        pot,
        phase,
        prepared_by_spin,
        orbital_tables,
        bound_orbital_counts,
        excitation_poles,
        spin_selectors,
        controls.for_single_higher_multipole(1)?,
        broadened_table,
    )?;
    let electric_quadrupole = generate_normal_potential_xsect_dat_from_pot_single_multipole(
        input,
        pot,
        phase,
        prepared_by_spin,
        orbital_tables,
        bound_orbital_counts,
        excitation_poles,
        spin_selectors,
        controls.for_single_higher_multipole(2)?,
        broadened_table,
    )?;

    combine_normal_xsect_multipole_passes(dipole, magnetic_dipole, electric_quadrupole)
}

fn combine_normal_xsect_multipole_passes(
    dipole: GeneratedNormalXsect,
    magnetic_dipole: GeneratedNormalXsect,
    mut electric_quadrupole: GeneratedNormalXsect,
) -> Result<GeneratedNormalXsect> {
    ensure!(
        dipole.xsect.titles == magnetic_dipole.xsect.titles
            && dipole.xsect.titles == electric_quadrupole.xsect.titles
            && dipole.xsect.scalars == magnetic_dipole.xsect.scalars
            && dipole.xsect.scalars == electric_quadrupole.xsect.scalars
            && dipole.xsect.core_hole_width_ev == magnetic_dipole.xsect.core_hole_width_ev
            && dipole.xsect.core_hole_width_ev == electric_quadrupole.xsect.core_hole_width_ev
            && dipole.xsect.main_energy_count == magnetic_dipole.xsect.main_energy_count
            && dipole.xsect.main_energy_count == electric_quadrupole.xsect.main_energy_count
            && dipole.xsect.fermi_index == magnetic_dipole.xsect.fermi_index
            && dipole.xsect.fermi_index == electric_quadrupole.xsect.fermi_index
            && dipole.xsect.energy_grid_ev == magnetic_dipole.xsect.energy_grid_ev
            && dipole.xsect.energy_grid_ev == electric_quadrupole.xsect.energy_grid_ev
            && dipole.transition_moments.dim() == magnetic_dipole.transition_moments.dim()
            && dipole.transition_moments.dim() == electric_quadrupole.transition_moments.dim(),
        "XSPH combined-multipole source passes produced incompatible handoff shapes"
    );

    electric_quadrupole.xsect.normalized_background =
        &electric_quadrupole.xsect.normalized_background
            + &magnetic_dipole.xsect.normalized_background
            - &dipole.xsect.normalized_background;
    electric_quadrupole.xsect.cross_section = &electric_quadrupole.xsect.cross_section
        + &magnetic_dipole.xsect.cross_section
        - &dipole.xsect.cross_section;
    electric_quadrupole.transition_moments = &electric_quadrupole.transition_moments
        + &magnetic_dipole.transition_moments
        - &dipole.transition_moments;

    ensure!(
        electric_quadrupole
            .xsect
            .normalized_background
            .iter()
            .all(|value| value.is_finite())
            && electric_quadrupole
                .xsect
                .cross_section
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
            && electric_quadrupole
                .transition_moments
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite()),
        "XSPH combined E1+E2+M1 source output contains non-finite values"
    );
    Ok(electric_quadrupole)
}

fn generate_normal_potential_xsect_dat_from_pot_single_multipole(
    input: &XsphInput,
    pot: &PotBinData,
    phase: &PhaseBinData,
    prepared_by_spin: &[XsphPhaseGridPreparation],
    orbital_tables: &RhorrpConfigOrbitalTables,
    bound_orbital_counts: &[usize],
    excitation_poles: Option<&[ExcitationPole]>,
    spin_selectors: &[i32],
    controls: XsphXsectAngularControls,
    broadened_table: Option<&BroadenedHedinLundqvistTable>,
) -> Result<GeneratedNormalXsect> {
    const ABSORBER_INDEX: usize = 0;
    let advanced = normal_xsect_effective_advanced(input);
    ensure!(
        phase.energy_count == phase.energy_grid.len(),
        "phase.bin declares {} energy rows but stores {}",
        phase.energy_count,
        phase.energy_grid.len()
    );
    ensure!(
        phase.reference_energy.dim() == (phase.energy_count, phase.spin_count),
        "phase.bin reference-energy shape {:?} does not match declared ({}, {})",
        phase.reference_energy.dim(),
        phase.energy_count,
        phase.spin_count
    );
    ensure!(
        input.lmaxph.len() > ABSORBER_INDEX,
        "XSPH lmaxph is missing the absorber entry for xsect.dat generation"
    );
    ensure!(
        input.lmaxph[ABSORBER_INDEX] >= 0,
        "XSPH absorber lmaxph must be nonnegative for xsect.dat generation"
    );
    ensure!(
        bound_orbital_counts.len() > ABSORBER_INDEX,
        "XSPH effective bound-orbital counts are missing the absorber entry for xsect.dat generation"
    );

    let energy_count = phase.energy_count;
    let spin_count = phase.spin_count;
    let q_count = phase.q_count;
    let transition_count = phase.transition_count;
    validate_xsect_spin_ground_states(
        spin_count,
        prepared_by_spin.len(),
        spin_selectors,
        controls,
    )?;
    let prepared = &prepared_by_spin[0];
    let active_transition_count = transition_count.min(PHASE_BIN_DEFAULT_TRANSITION_COUNT);
    let fermi_index = usize::try_from(phase.fermi_index)
        .context("phase.bin ik0 is negative for xsect.dat generation")?;
    ensure!(
        fermi_index > 0 && fermi_index <= energy_count,
        "phase.bin ik0 {} must be in 1..={energy_count} for xsect.dat generation",
        phase.fermi_index
    );

    let core_hole = core_hole_quantum_numbers(pot.ihole)
        .context("failed to determine XSPH core-hole quantum numbers")?;
    let core_hole_l = usize::try_from(core_hole.angular_momentum)
        .context("XSPH core-hole angular momentum is negative")?;
    let max_angular_momentum = usize::try_from(input.lmaxph[ABSORBER_INDEX])
        .context("XSPH absorber lmaxph is negative")?;
    let mut bcoef_weights_by_spin = Vec::with_capacity(spin_count);
    for &spin_selector in spin_selectors {
        bcoef_weights_by_spin.push(
            xsph_xsect_bcoef_weights(XsphXsectBcoefWeightsInput {
                max_angular_momentum,
                initial_kappa: core_hole.kappa,
                polarization: controls.polarization,
                polarization_tensor: controls.polarization_tensor,
                higher_multipole_selector: controls.higher_multipole_selector,
                spin: spin_selector,
                spin_channels: spin_count,
                spin_vector_angle: controls.spin_vector_angle,
            })
            .context("failed to build XSPH xsect bcoef weights")?,
        );
    }
    let spin_polarized_cross_terms = xsect_spin_polarized_cross_terms(spin_count, controls);

    let muffin_tin_radius = pot.muffin_tin_radii[ABSORBER_INDEX];
    let radial_indices = xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
        muffin_tin_radius,
        grid_origin: XSPH_LOUCKS_GRID_ORIGIN,
        log_step: prepared.radial_dx,
        radial_capacity: prepared.radii.len(),
    })
    .context("failed to compute XSPH absorber radial indices for xsect.dat")?;
    let radial_match_index = radial_indices
        .radial_match_index_1based
        .checked_sub(1)
        .context("XSPH xsect radial match index underflow")?;
    let xcpot_active_len = radial_indices.reference_index_1based;
    let initial_spinor = xsph_normalized_initial_spinor_grid(
        pot,
        prepared,
        ABSORBER_INDEX,
        core_hole_l,
        "xsect.dat",
    )?;
    let norman_index_1based = initial_spinor.norman_index_1based;
    let initial_large = initial_spinor.large;
    let initial_small = initial_spinor.small;

    let bound_orbital_count = bound_orbital_counts[ABSORBER_INDEX];
    ensure!(
        bound_orbital_count > 0,
        "config.dat absorber potential has no occupied orbitals"
    );
    ensure!(
        bound_orbital_count <= pot.orbital_occupancy.nrows(),
        "config.dat absorber bound-orbital count {bound_orbital_count} exceeds pot.bin orbital occupancy rows {}",
        pot.orbital_occupancy.nrows()
    );

    let radial_prefix = Slice::from(..xcpot_active_len);
    let orbital_prefix = Slice::from(..bound_orbital_count);
    let electron_density_column = prepared
        .electron_density
        .index_axis(Axis(1), ABSORBER_INDEX);
    let electron_density = electron_density_column.slice_axis(Axis(0), radial_prefix);
    let many_pole_self_energy = xsph_many_pole_self_energy_for_potential(
        input,
        excitation_poles,
        electron_density,
        xcpot_active_len,
        input.control.ixc0,
    )?;
    let magnetization_column = prepared.magnetization.index_axis(Axis(1), ABSORBER_INDEX);
    let magnetization = magnetization_column.slice_axis(Axis(0), radial_prefix);
    let valence_density_column = prepared.valence_density.index_axis(Axis(1), ABSORBER_INDEX);
    let valence_density = valence_density_column.slice_axis(Axis(0), radial_prefix);
    let bound_large_potential = prepared
        .bound_large_components
        .index_axis(Axis(2), ABSORBER_INDEX);
    let bound_large = bound_large_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_potential = prepared
        .bound_small_components
        .index_axis(Axis(2), ABSORBER_INDEX);
    let bound_small = bound_small_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_large_coefficients_potential =
        pot.large_coefficients.index_axis(Axis(2), ABSORBER_INDEX);
    let bound_large_coefficients =
        bound_large_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let bound_small_coefficients_potential =
        pot.small_coefficients.index_axis(Axis(2), ABSORBER_INDEX);
    let bound_small_coefficients =
        bound_small_coefficients_potential.slice_axis(Axis(1), orbital_prefix);
    let electron_counts_potential = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let electron_counts = electron_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let valence_counts_potential = pot.orbital_occupancy.index_axis(Axis(1), ABSORBER_INDEX);
    let valence_counts = valence_counts_potential.slice_axis(Axis(0), orbital_prefix);
    let kappa_potential = orbital_tables
        .kappa_by_potential
        .index_axis(Axis(1), ABSORBER_INDEX);
    let kappa = kappa_potential.slice_axis(Axis(0), orbital_prefix);

    let mut spectrum_norms = Array2::<f64>::zeros((energy_count, spin_count));
    let mut cross_sections = Array2::<Complex64>::zeros((energy_count, spin_count));
    let mut transition_moments =
        Array4::<Complex64>::zeros((energy_count, q_count, transition_count, spin_count));
    let mut active_rows_by_spin = vec![0_usize; spin_count];
    let photon_energy_offset_hartree = xsph_chemical_potential_hartree(input, pot);

    for (spin_index, (&spin_selector, bcoef_weights)) in spin_selectors
        .iter()
        .zip(bcoef_weights_by_spin.iter())
        .enumerate()
    {
        let spin_prepared = &prepared_by_spin[spin_index];
        ensure!(
            spin_prepared.radii.len() == prepared.radii.len()
                && spin_prepared.radial_dx == prepared.radial_dx,
            "XSPH xsect spin selector {spin_selector} prepared an incompatible radial grid"
        );
        let total_potential_column = spin_prepared
            .total_potential
            .index_axis(Axis(1), ABSORBER_INDEX);
        let total_potential = total_potential_column.slice_axis(Axis(0), radial_prefix);
        let valence_potential_column = spin_prepared
            .valence_potential
            .index_axis(Axis(1), ABSORBER_INDEX);
        let valence_potential = valence_potential_column.slice_axis(Axis(0), radial_prefix);
        // FEFF's xcpot cache belongs to one XSECT invocation, hence one
        // signed spin selector. Sharing it across spins reuses the first
        // spin's reference potential in the second channel.
        let mut fermi_cache: Option<Array1<XcpotFermiCache>> = None;

        for energy_index in 0..energy_count {
            let xcpot_result = evaluate_xsph_xcpot(
                XcpotInput {
                    exchange_selector: input.control.ixc0,
                    lreal: input.control.lreal,
                    energy: phase.energy_grid[energy_index],
                    fermi_level: pot.scalars.fermi_level,
                    total_potential,
                    valence_potential,
                    density: electron_density,
                    magnetization,
                    valence_density,
                    active_len: xcpot_active_len,
                    plasmon_selector: input.control.i_plsmn,
                    many_pole_delta_table: None,
                    many_pole_self_energy: many_pole_self_energy
                        .as_ref()
                        .map(|poles| poles.as_xcpot_input()),
                    fermi_cache: fermi_cache.as_ref().map(|cache| cache.view()),
                },
                broadened_table,
            )
            .with_context(|| {
                format!(
                    "failed to evaluate XSPH xcpot for xsect.dat energy row {}",
                    energy_index + 1
                )
            })?;
            if !xcpot_result.fermi_cache.is_empty() {
                fermi_cache = Some(xcpot_result.fermi_cache.clone());
            }

            let setup = xsph_xsect_energy_setup(XsphXsectEnergySetupInput {
                energy: phase.energy_grid[energy_index],
                reference_energy: xcpot_result.reference_energy,
                edge_energy: phase.scalars.edge_energy,
                chemical_potential: photon_energy_offset_hartree,
                muffin_tin_radius,
                exchange_selector: input.control.ixc0,
                norman_index_1based,
                new_grid_index_1based: initial_spinor.active_len,
                radial_capacity: prepared.radii.len(),
            })
            .context("failed to set up XSPH xsect energy row")?;
            if setup.decision != XsphXsectEnergyDecision::Active {
                continue;
            }
            ensure!(
                setup.active_radial_len <= initial_large.len()
                    && setup.active_radial_len <= initial_small.len()
                    && setup.active_radial_len <= electron_density_column.len(),
                "XSPH xsect active radial length {} exceeds core-hole spinor/density lengths ({}, {}, {})",
                setup.active_radial_len,
                initial_large.len(),
                initial_small.len(),
                electron_density_column.len()
            );
            let active_prefix = Slice::from(..setup.active_radial_len);
            let active_initial_large = initial_large.slice_axis(Axis(0), active_prefix);
            let active_initial_small = initial_small.slice_axis(Axis(0), active_prefix);
            let active_radii = prepared.radii.slice_axis(Axis(0), active_prefix);
            let active_electron_density =
                electron_density_column.slice_axis(Axis(0), active_prefix);
            let xray_bessel = xsph_xray_bessel_table(XsphXrayBesselTableInput {
                photon_wave_number: setup.photon_wave_number,
                radii: prepared.radii.view(),
                active_len: setup.active_radial_len,
            })
            .context("failed to build XSPH xsect photon Bessel table")?;
            let plan = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
                photon_energy: setup.photon_energy,
                selected_higher_multipole: controls.selected_higher_multipole,
                transition_direction: controls.transition_direction,
                initial_kappa: core_hole.kappa,
                final_kappas: bcoef_weights.final_kappas.view(),
                orbital_l: bcoef_weights.orbital_l.view(),
                active_len: active_transition_count,
            })
            .context("failed to plan XSPH xsect transitions")?;
            if plan.transitions.is_empty() {
                continue;
            }
            ensure!(
                normal_xsect_positive_izstd_transitions_supported(advanced, &plan.transitions),
                "XSPH positive-izstd xsect generation rejects nonrelativistic M1 because FEFF radint stops for mult=1 with ifl < 0"
            );

            let solver_total_potential = extend_xcpot_potential(
                &xcpot_result.total_potential,
                prepared.radii.len(),
                "total",
            )?;
            let solver_valence_potential = if xcpot_result.valence_potential.is_empty() {
                solver_total_potential.clone()
            } else {
                extend_xcpot_potential(
                    &xcpot_result.valence_potential,
                    prepared.radii.len(),
                    "valence",
                )?
            };
            let standard_screened_fields = if advanced.izstd > 0 {
                let unity_fscf = Array1::<Complex64>::from_elem(
                    setup.active_radial_len,
                    Complex64::new(1.0, 0.0),
                );
                let (dipole_field_scale, dipole_fscf) =
                    if plan.transitions.iter().any(|transition| {
                        transition.multipole == XsphTransitionMultipole::ElectricDipole
                    }) {
                        let screened_setup =
                            xsph_xsect_screened_field_setup(XsphXsectScreenedFieldInput {
                                multipole: XsphTransitionMultipole::ElectricDipole,
                                standard_potential: true,
                                orbital_correction_pending: false,
                                momentum_squared: setup.momentum_squared,
                                edge_energy: phase.scalars.edge_energy,
                                chemical_potential: photon_energy_offset_hartree,
                                screened_orbital_energy: normal_xsect_hole_orbital_energy(pot)?,
                            })
                            .context("failed to set up XSPH positive-izstd screened field")?;
                        let phiscf_workspace = screened_setup.phiscf_workspace.context(
                            "XSPH positive-izstd screened field did not prepare phiscf workspace",
                        )?;
                        let local_field =
                            xsph_xsect_phiscf_local_field(XsphXsectPhiscfLocalFieldInput {
                                exchange_correlation_selector: advanced.ifxc,
                                radii: active_radii,
                                electron_density: active_electron_density,
                                active_len: setup.active_radial_len,
                            })
                            .context("failed to build XSPH positive-izstd local field")?;
                        let coarse_count = normal_xsect_phiscf_coarse_count_for_active_len(
                            setup.active_radial_len,
                            prepared.radii.len(),
                        )?;
                        let fine_len = normal_xsect_phiscf_fine_len_for_coarse_count(coarse_count)?;
                        let occupied_table = normal_xsect_phiscf_occupied_table(
                            pot,
                            orbital_tables,
                            ABSORBER_INDEX,
                            bound_orbital_count,
                        )?;
                        let assembly = normal_xsect_phiscf_wfirdc_assembly(
                            NormalXsectPhiscfWfirdcAssemblyInput {
                                momentum_squared: setup.momentum_squared,
                                edge_energy: phase.scalars.edge_energy,
                                chemical_potential: photon_energy_offset_hartree,
                                hole_orbital_energy: normal_xsect_hole_orbital_energy(pot)?,
                                scale_function: phiscf_workspace.scale_function,
                                occupied_table: &occupied_table,
                                orbital_kappas: kappa,
                                radii: prepared.radii.view(),
                                exchange_correlation_potential: solver_total_potential.view(),
                                bound_large_components: bound_large,
                                bound_small_components: bound_small,
                                bound_large_coefficients,
                                bound_small_coefficients,
                                electron_counts,
                                valence_counts,
                                local_field: local_field.values.view(),
                                nuclear_charge: pot.atomic_numbers[ABSORBER_INDEX] as f64,
                                muffin_tin_radius,
                                step: prepared.radial_dx,
                                target_last_index_1based: setup.active_radial_len,
                                active_len: setup.active_radial_len,
                                coarse_count,
                                c3_scale: 1,
                            },
                        )
                        .context("failed to assemble XSPH positive-izstd wfirdc inputs")?;
                        let basis_fields = Array2::<Complex64>::zeros((fine_len, 0));
                        let collected = assembly
                            .collect_wfirdc_contributions(
                                prepared.radii.view(),
                                basis_fields.view(),
                                0,
                            )
                            .context("failed to solve XSPH positive-izstd screened field")?;
                        if normal_xsect_screened_field_collapsed(
                            collected.screened_solution.screened_field.view(),
                        )? {
                            (1.0, Some(unity_fscf.clone()))
                        } else {
                            (
                                screened_setup.field_scale,
                                Some(collected.screened_solution.screened_field),
                            )
                        }
                    } else {
                        (1.0, None)
                    };
                Some(NormalXsectStandardScreenedFields {
                    dipole_field_scale,
                    dipole_fscf,
                    unity_fscf,
                })
            } else {
                None
            };
            let target_last_index = setup
                .active_radial_len
                .checked_sub(1)
                .context("XSPH xsect active radial length underflow")?;
            let mut regular_channels =
                Vec::<XsphXsectRegularChannel>::with_capacity(plan.transitions.len());
            let mut irregular_channels =
                Vec::<XsphXsectIrregularChannel>::with_capacity(plan.transitions.len());
            let mut spin_orbit_removed_regular_channels = if spin_polarized_cross_terms {
                Some(Vec::<XsphXsectRegularChannel>::with_capacity(
                    plan.transitions.len(),
                ))
            } else {
                None
            };
            let mut spin_orbit_removed_irregular_channels = if spin_polarized_cross_terms {
                Some(Vec::<XsphXsectIrregularChannel>::with_capacity(
                    plan.transitions.len(),
                ))
            } else {
                None
            };

            for transition in &plan.transitions {
                let solver = FovrgDiracSolverInput {
                    exchange_cycle_count: setup.cycle_count,
                    target_kappa: transition.final_kappa,
                    muffin_tin_radius,
                    target_last_index,
                    energy: setup.momentum_squared,
                    step: prepared.radial_dx,
                    radii: prepared.radii.view(),
                    exchange_correlation_potential: solver_total_potential.view(),
                    valence_exchange_correlation_potential: solver_valence_potential.view(),
                    bound_large_components: bound_large,
                    bound_small_components: bound_small,
                    bound_large_coefficients,
                    bound_small_coefficients,
                    electron_counts,
                    valence_counts,
                    kappa,
                    muffin_tin_large_component: Complex64::new(0.0, 0.0),
                    muffin_tin_small_component: Complex64::new(0.0, 0.0),
                    atomic_number: pot.atomic_numbers[ABSORBER_INDEX] as f64,
                    irregular: false,
                    c3_scale: 0,
                    radial_match_index,
                    bound_orbital_count,
                };
                let regular = xsph_xsect_regular_channel(XsphXsectRegularChannelInput {
                solver,
                wave_number: setup.wave_number,
            })
            .with_context(|| {
                format!(
                    "failed to solve XSPH xsect regular channel for energy row {}, transition {}",
                    energy_index + 1,
                    transition.transition_index_1based
                )
            })?;
                let irregular = xsph_xsect_irregular_channel(XsphXsectIrregularChannelInput {
                solver,
                wave_number: setup.wave_number,
                regular_channel: &regular,
            })
            .with_context(|| {
                format!(
                    "failed to solve XSPH xsect irregular channel for energy row {}, transition {}",
                    energy_index + 1,
                    transition.transition_index_1based
                )
            })?;
                regular_channels.push(regular);
                irregular_channels.push(irregular);

                if let (Some(retry_regular_channels), Some(retry_irregular_channels)) = (
                    spin_orbit_removed_regular_channels.as_mut(),
                    spin_orbit_removed_irregular_channels.as_mut(),
                ) {
                    let retry_solver = FovrgDiracSolverInput {
                        c3_scale: 1,
                        ..solver
                    };
                    let retry_regular =
                    xsph_xsect_regular_channel(XsphXsectRegularChannelInput {
                        solver: retry_solver,
                        wave_number: setup.wave_number,
                    })
                    .with_context(|| {
                        format!(
                            "failed to solve XSPH xsect spin-orbit-removed regular channel for energy row {}, transition {}",
                            energy_index + 1,
                            transition.transition_index_1based
                        )
                    })?;
                    let retry_irregular =
                    xsph_xsect_irregular_channel(XsphXsectIrregularChannelInput {
                        solver: retry_solver,
                        wave_number: setup.wave_number,
                        regular_channel: &retry_regular,
                    })
                    .with_context(|| {
                        format!(
                            "failed to solve XSPH xsect spin-orbit-removed irregular channel for energy row {}, transition {}",
                            energy_index + 1,
                            transition.transition_index_1based
                        )
                    })?;
                    retry_regular_channels.push(retry_regular);
                    retry_irregular_channels.push(retry_irregular);
                }
            }

            let output_normalization = if let Some(screened_fields) =
                standard_screened_fields.as_ref()
            {
                let transition_fields = screened_fields.transition_fields(&plan.transitions)?;
                xsph_xsect_bcoef_standard_energy_row_with_transition_fields(
                    XsphXsectBcoefStandardEnergyRowFieldsInput {
                        transitions: &plan.transitions,
                        regular_channels: &regular_channels,
                        irregular_channels: &irregular_channels,
                        transition_fields: &transition_fields,
                        selected_higher_multipole: controls.selected_higher_multipole,
                        initial_kappa: core_hole.kappa,
                        initial_large: active_initial_large,
                        initial_small: active_initial_small,
                        xray_bessel: xray_bessel.values.view(),
                        radii: active_radii,
                        log_step: prepared.radial_dx,
                        photon_wave_number: setup.photon_wave_number,
                        diagonal_weights: bcoef_weights.diagonal_weights.view(),
                        spin_polarized_cross_terms,
                        orbital_l: bcoef_weights.orbital_l.view(),
                        trace_weights: bcoef_weights.trace_weights.view(),
                        spin_orbit_removed_regular_channels: spin_orbit_removed_regular_channels
                            .as_deref(),
                        spin_orbit_removed_irregular_channels:
                            spin_orbit_removed_irregular_channels.as_deref(),
                        photon_energy: setup.photon_energy,
                        wave_number: setup.wave_number,
                        active_channel_count: active_transition_count,
                    },
                )
                .with_context(|| {
                    format!(
                        "failed to accumulate XSPH standard xsect row {}",
                        energy_index + 1
                    )
                })?
                .output_normalization
            } else {
                xsph_xsect_bcoef_nonstandard_energy_row(XsphXsectBcoefNonstandardEnergyRowInput {
                    transitions: &plan.transitions,
                    regular_channels: &regular_channels,
                    irregular_channels: &irregular_channels,
                    selected_higher_multipole: controls.selected_higher_multipole,
                    initial_kappa: core_hole.kappa,
                    initial_large: active_initial_large,
                    initial_small: active_initial_small,
                    xray_bessel: xray_bessel.values.view(),
                    radii: active_radii,
                    log_step: prepared.radial_dx,
                    diagonal_weights: bcoef_weights.diagonal_weights.view(),
                    spin_polarized_cross_terms,
                    orbital_l: bcoef_weights.orbital_l.view(),
                    trace_weights: bcoef_weights.trace_weights.view(),
                    spin_orbit_removed_regular_channels: spin_orbit_removed_regular_channels
                        .as_deref(),
                    spin_orbit_removed_irregular_channels: spin_orbit_removed_irregular_channels
                        .as_deref(),
                    photon_energy: setup.photon_energy,
                    wave_number: setup.wave_number,
                    active_channel_count: active_transition_count,
                })
                .with_context(|| {
                    format!(
                        "failed to accumulate XSPH nonstandard xsect row {}",
                        energy_index + 1
                    )
                })?
                .output_normalization
            };
            spectrum_norms[(energy_index, spin_index)] = output_normalization.spectrum_norm;
            cross_sections[(energy_index, spin_index)] = output_normalization.cross_section;
            for transition_index in 0..active_transition_count {
                transition_moments[(energy_index, 0, transition_index, spin_index)] =
                    output_normalization.reduced_matrix_elements[transition_index];
            }
            active_rows_by_spin[spin_index] += 1;
        }
    }

    ensure!(
        active_rows_by_spin.iter().all(|&count| count > 0),
        "XSPH xsect.dat generation produced incomplete spin rows for selectors {spin_selectors:?}: {active_rows_by_spin:?}"
    );
    let xsect_titles = xsph_xsect_material_header_titles(input, pot, phase.scalars.edge_energy)?;
    let generated = xsect_dat_from_xsph_spin_merge(XsectDatFromXsphSpinInput {
        titles: &xsect_titles,
        scalars: XsectDatScalars {
            amplitude_reduction: pot.scalars.amplitude_reduction,
            relaxation_energy: pot.scalars.relaxation_energy,
            plasmon_frequency: pot.scalars.plasmon_frequency,
            edge_energy: phase.scalars.edge_energy,
            chemical_potential: photon_energy_offset_hartree,
        },
        core_hole_width_hartree: input.grid.gamach / FEFF_HARTREE_EV,
        main_energy_count: phase.main_energy_count,
        fermi_index,
        energy_grid_hartree: phase.energy_grid.view(),
        spin_polarized: xsect_spin_polarized(spin_count, controls),
        spectrum_norms: spectrum_norms.view(),
        cross_sections: cross_sections.view(),
        transition_moments: transition_moments.view(),
        q_count,
        transition_count,
    })
    .context("failed to merge XSPH xsect spin rows")?;

    let xsect = normal_xsect_apply_spectroscopy_conventions(input, generated.xsect);

    Ok(GeneratedNormalXsect {
        xsect,
        transition_moments: generated.transition_moments,
    })
}

fn normal_xsect_apply_spectroscopy_conventions(
    input: &XsphInput,
    mut xsect: XsectDatData,
) -> XsectDatData {
    if input.control.ispec == 4 {
        for (cross_section, normalized_background) in xsect
            .cross_section
            .iter_mut()
            .zip(xsect.normalized_background.iter())
        {
            *cross_section = Complex64::new(0.0, *normalized_background);
        }
    }
    xsect
}

fn normal_xsect_screened_field_collapsed(field: ArrayView1<'_, Complex64>) -> Result<bool> {
    ensure!(
        !field.is_empty(),
        "XSPH positive-izstd screened field produced no rows"
    );
    let mut max_norm = 0.0_f64;
    for (index, value) in field.iter().copied().enumerate() {
        ensure!(
            value.re.is_finite() && value.im.is_finite(),
            "XSPH positive-izstd screened field row {index} must be finite, got {value}"
        );
        max_norm = max_norm.max(value.norm());
    }
    Ok(max_norm <= 1.0e-80)
}

#[derive(Debug, Clone, PartialEq)]
struct NormalXsectStandardScreenedFields {
    dipole_field_scale: f64,
    dipole_fscf: Option<Array1<Complex64>>,
    unity_fscf: Array1<Complex64>,
}

impl NormalXsectStandardScreenedFields {
    fn transition_fields<'a>(
        &'a self,
        transitions: &'a [XsphXsectTransition],
    ) -> Result<Vec<XsphXsectBcoefStandardTransitionField<'a>>> {
        transitions
            .iter()
            .map(|transition| {
                if transition.multipole == XsphTransitionMultipole::ElectricDipole {
                    let fscf = self.dipole_fscf.as_ref().context(
                        "XSPH positive-izstd dipole transition is missing screened fscf",
                    )?;
                    Ok(XsphXsectBcoefStandardTransitionField {
                        screened_field_scale: self.dipole_field_scale,
                        fscf: fscf.view(),
                    })
                } else {
                    Ok(XsphXsectBcoefStandardTransitionField {
                        screened_field_scale: 1.0,
                        fscf: self.unity_fscf.view(),
                    })
                }
            })
            .collect()
    }
}

fn normal_xsect_phiscf_fine_len_for_coarse_count(coarse_count: usize) -> Result<usize> {
    coarse_count
        .checked_sub(1)
        .and_then(|intervals| intervals.checked_mul(5))
        .and_then(|offset| offset.checked_add(1))
        .with_context(|| format!("XSPH phiscf coarse count {coarse_count} overflows fine length"))
}

fn normal_xsect_phiscf_coarse_count_for_active_len(
    active_len: usize,
    radial_capacity: usize,
) -> Result<usize> {
    ensure!(
        active_len > 0,
        "XSPH phiscf coarse grid requires at least one active radial row"
    );
    ensure!(
        radial_capacity >= active_len,
        "XSPH phiscf radial capacity {radial_capacity} cannot supply active length {active_len}"
    );

    let intervals = active_len - 1;
    let coarse_count = intervals / 5 + usize::from(!intervals.is_multiple_of(5)) + 1;
    let fine_len = normal_xsect_phiscf_fine_len_for_coarse_count(coarse_count)?;
    ensure!(
        fine_len <= radial_capacity,
        "XSPH phiscf coarse grid fine length {fine_len} exceeds radial capacity {radial_capacity}"
    );
    Ok(coarse_count)
}

fn normal_xsect_hole_orbital_energy(pot: &PotBinData) -> Result<f64> {
    ensure!(
        pot.ihole > 0,
        "XSPH positive-izstd hole index must be positive, got {}",
        pot.ihole
    );
    let hole_slot = usize::try_from(pot.ihole - 1)
        .context("XSPH positive-izstd hole index could not be converted to a slot")?;
    ensure!(
        hole_slot < pot.orbital_energies.len(),
        "XSPH positive-izstd hole orbital slot {hole_slot} exceeds pot.bin eorb length {}",
        pot.orbital_energies.len()
    );
    let energy = pot.orbital_energies[hole_slot];
    ensure!(
        energy.is_finite(),
        "XSPH positive-izstd hole orbital energy for slot {hole_slot} must be finite, got {energy}"
    );
    Ok(energy)
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NormalXsectPhiscfOccupiedTable {
    orbital_energy_counts: Array1<usize>,
    occupied_energies: Array2<f64>,
    occupation_fractions: Array2<f64>,
}

#[allow(dead_code)]
fn normal_xsect_phiscf_occupied_table(
    pot: &PotBinData,
    orbital_tables: &RhorrpConfigOrbitalTables,
    potential_index: usize,
    active_orbital_count: usize,
) -> Result<NormalXsectPhiscfOccupiedTable> {
    ensure!(
        potential_index == 0,
        "XSPH phiscf occupied-energy table currently requires the absorber potential because pot.bin eorb stores absorber orbital energies"
    );
    ensure!(
        potential_index < pot.potential_count(),
        "XSPH phiscf potential index {potential_index} exceeds pot.bin potential count {}",
        pot.potential_count()
    );
    ensure!(
        potential_index < orbital_tables.bound_orbital_counts.len(),
        "XSPH phiscf potential index {potential_index} exceeds config.dat potential count {}",
        orbital_tables.bound_orbital_counts.len()
    );
    ensure!(
        active_orbital_count > 0,
        "XSPH phiscf occupied-energy table requires at least one active orbital"
    );

    let bound_orbital_count = orbital_tables.bound_orbital_counts[potential_index];
    ensure!(
        active_orbital_count <= bound_orbital_count,
        "XSPH phiscf active orbital count {active_orbital_count} exceeds config.dat bound-orbital count {bound_orbital_count}"
    );
    ensure!(
        orbital_tables.electron_counts_by_potential.nrows() >= active_orbital_count
            && orbital_tables.electron_counts_by_potential.ncols() > potential_index,
        "XSPH phiscf electron-count table shape {:?} cannot supply {active_orbital_count} orbitals for potential {potential_index}",
        orbital_tables.electron_counts_by_potential.dim()
    );
    ensure!(
        orbital_tables.valence_counts_by_potential.nrows() >= active_orbital_count
            && orbital_tables.valence_counts_by_potential.ncols() > potential_index,
        "XSPH phiscf valence-count table shape {:?} cannot supply {active_orbital_count} orbitals for potential {potential_index}",
        orbital_tables.valence_counts_by_potential.dim()
    );
    ensure!(
        orbital_tables.kappa_by_potential.nrows() >= active_orbital_count
            && orbital_tables.kappa_by_potential.ncols() > potential_index,
        "XSPH phiscf kappa table shape {:?} cannot supply {active_orbital_count} orbitals for potential {potential_index}",
        orbital_tables.kappa_by_potential.dim()
    );
    ensure!(
        orbital_tables.orbital_slots_by_potential.nrows() >= active_orbital_count
            && orbital_tables.orbital_slots_by_potential.ncols() > potential_index,
        "XSPH phiscf orbital-slot table shape {:?} cannot supply {active_orbital_count} orbitals for potential {potential_index}",
        orbital_tables.orbital_slots_by_potential.dim()
    );

    let mut orbital_energy_counts = Array1::<usize>::zeros(active_orbital_count);
    let mut occupied_energies = Array2::<f64>::zeros((1, active_orbital_count));
    let mut occupation_fractions = Array2::<f64>::zeros((1, active_orbital_count));
    let mut has_valence_response = false;
    for orbital_index in 0..active_orbital_count {
        let valence_count =
            orbital_tables.valence_counts_by_potential[(orbital_index, potential_index)];
        ensure!(
            valence_count.is_finite() && valence_count >= 0.0,
            "XSPH phiscf valence count for compact orbital {} must be finite and nonnegative, got {valence_count}",
            orbital_index + 1
        );
        has_valence_response |= valence_count > 0.0;
    }

    for orbital_index in 0..active_orbital_count {
        let slot = orbital_tables.orbital_slots_by_potential[(orbital_index, potential_index)];
        ensure!(
            slot < pot.orbital_energies.len(),
            "XSPH phiscf orbital slot {slot} for compact orbital {} exceeds pot.bin eorb length {}",
            orbital_index + 1,
            pot.orbital_energies.len()
        );
        let occupied_energy = pot.orbital_energies[slot];
        ensure!(
            occupied_energy.is_finite(),
            "XSPH phiscf occupied orbital energy for compact orbital {} slot {slot} must be finite, got {occupied_energy}",
            orbital_index + 1
        );
        let electron_count =
            orbital_tables.electron_counts_by_potential[(orbital_index, potential_index)];
        ensure!(
            electron_count.is_finite() && electron_count >= 0.0,
            "XSPH phiscf electron count for compact orbital {} must be finite and nonnegative, got {electron_count}",
            orbital_index + 1
        );
        let valence_count =
            orbital_tables.valence_counts_by_potential[(orbital_index, potential_index)];
        let response_count = if has_valence_response {
            valence_count
        } else {
            electron_count
        };
        if response_count <= 0.0 {
            continue;
        }
        let kappa = orbital_tables.kappa_by_potential[(orbital_index, potential_index)];
        ensure!(
            kappa != 0,
            "XSPH phiscf compact orbital {} has zero kappa",
            orbital_index + 1
        );
        let shell_capacity = 2.0 * f64::from(kappa.unsigned_abs());
        let occupation_fraction = response_count / shell_capacity;
        ensure!(
            occupation_fraction.is_finite() && occupation_fraction <= 1.0 + 1.0e-12,
            "XSPH phiscf shell occupation fraction for compact orbital {} must be finite and no more than one, got {occupation_fraction}",
            orbital_index + 1
        );

        orbital_energy_counts[orbital_index] = 1;
        occupied_energies[(0, orbital_index)] = occupied_energy;
        occupation_fractions[(0, orbital_index)] = occupation_fraction;
    }

    Ok(NormalXsectPhiscfOccupiedTable {
        orbital_energy_counts,
        occupied_energies,
        occupation_fractions,
    })
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct NormalXsectPhiscfWfirdcAssemblyInput<'a> {
    momentum_squared: Complex64,
    edge_energy: f64,
    chemical_potential: f64,
    hole_orbital_energy: f64,
    scale_function: f64,
    occupied_table: &'a NormalXsectPhiscfOccupiedTable,
    orbital_kappas: ArrayView1<'a, i32>,
    radii: ArrayView1<'a, f64>,
    exchange_correlation_potential: ArrayView1<'a, Complex64>,
    bound_large_components: ndarray::ArrayView2<'a, f64>,
    bound_small_components: ndarray::ArrayView2<'a, f64>,
    bound_large_coefficients: ndarray::ArrayView2<'a, f64>,
    bound_small_coefficients: ndarray::ArrayView2<'a, f64>,
    electron_counts: ArrayView1<'a, f64>,
    valence_counts: ArrayView1<'a, f64>,
    local_field: ArrayView1<'a, f64>,
    nuclear_charge: f64,
    muffin_tin_radius: f64,
    step: f64,
    target_last_index_1based: usize,
    active_len: usize,
    coarse_count: usize,
    c3_scale: i32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NormalXsectPhiscfWfirdcAssembly {
    rows: Vec<NormalXsectPhiscfWfirdcRow>,
    bound_large_components: Array2<f64>,
    bound_small_components: Array2<f64>,
    bound_large_coefficients: Array2<f64>,
    bound_small_coefficients: Array2<f64>,
    electron_counts: Array1<f64>,
    local_field: Array1<f64>,
    nuclear_charge: f64,
    step: f64,
    coarse_count: usize,
    c3_scale: i32,
    active_len: usize,
    bound_orbital_count: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct NormalXsectPhiscfWfirdcRow {
    plan_row: XsphXsectPhiscfContributionPlanRow,
    radial_setup: XsphXsectPhiscfRadialSolverSetup,
    exchange_correlation_potential: Array1<Complex64>,
    c3_potential: Array1<Complex64>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
}

impl NormalXsectPhiscfWfirdcAssembly {
    #[allow(dead_code)]
    fn contribution_input(
        &self,
        row_index: usize,
    ) -> Result<XsphXsectPhiscfWfirdcContributionInput<'_>> {
        let row = self
            .rows
            .get(row_index)
            .with_context(|| format!("XSPH phiscf contribution row {row_index} is missing"))?;
        let orbital_index = row
            .plan_row
            .orbital_index_1based
            .checked_sub(1)
            .context("XSPH phiscf contribution row has zero orbital index")?;
        ensure!(
            orbital_index < self.bound_orbital_count,
            "XSPH phiscf contribution orbital {} exceeds active bound-orbital count {}",
            row.plan_row.orbital_index_1based,
            self.bound_orbital_count
        );

        Ok(XsphXsectPhiscfWfirdcContributionInput {
            coarse_count: self.coarse_count,
            wave_number: row.radial_setup.wave_number,
            wfirdc_input: FovrgInitialPhotoelectronInput {
                energy: row.plan_row.pole.pole_energy,
                bound_large_coefficients: self.bound_large_coefficients.view(),
                bound_small_coefficients: self.bound_small_coefficients.view(),
                electron_counts: self.electron_counts.view(),
                kappa: row.kappa.view(),
                orbital_lengths: row.orbital_lengths.view(),
                exchange_correlation_potential: row.exchange_correlation_potential.view(),
                c3_potential: row.c3_potential.view(),
                initial_large_coefficient: Complex64::new(0.0, 0.0),
                initial_small_coefficient: Complex64::new(0.0, 0.0),
                nuclear_charge: self.nuclear_charge,
                muffin_tin_radius: row.radial_setup.match_radius,
                step: self.step,
                speed_of_light: FEFF_WFIRDC_SPEED_OF_LIGHT,
                c3_scale: self.c3_scale,
                irregular: false,
                radial_match_index: row.radial_setup.match_index,
                wkb_index: row.radial_setup.wkb_index,
                coefficient_count: XSPH_PHISCF_WFIRDC_COEFFICIENT_COUNT,
                orbital_count: self.bound_orbital_count + 1,
                active_len: self.active_len,
            },
            orbital_large: self
                .bound_large_components
                .index_axis(Axis(1), orbital_index),
            orbital_small: self
                .bound_small_components
                .index_axis(Axis(1), orbital_index),
            local_field: self.local_field.view(),
            response_scale: row.plan_row.rule.scale,
            include_response_imaginary: row.plan_row.rule.include_imaginary,
        })
    }

    #[allow(dead_code)]
    fn contribution_inputs(&self) -> Result<Vec<XsphXsectPhiscfWfirdcContributionInput<'_>>> {
        (0..self.rows.len())
            .map(|row_index| self.contribution_input(row_index))
            .collect()
    }

    #[allow(dead_code)]
    fn collect_wfirdc_contributions(
        &self,
        radii: ArrayView1<'_, f64>,
        basis_fields: ArrayView2<'_, Complex64>,
        basis_count: usize,
    ) -> Result<XsphXsectPhiscfWfirdcContributions> {
        ensure!(
            radii.len() >= self.active_len,
            "XSPH phiscf collector radii length {} cannot supply active length {}",
            radii.len(),
            self.active_len
        );
        ensure!(
            basis_count == 0 || basis_fields.ncols() >= basis_count,
            "XSPH phiscf collector basis field shape {:?} cannot supply {basis_count} columns",
            basis_fields.dim()
        );
        let contribution_inputs = self.contribution_inputs()?;
        xsph_xsect_phiscf_wfirdc_contributions(XsphXsectPhiscfWfirdcContributionsInput {
            coarse_count: self.coarse_count,
            radii,
            contribution_inputs: &contribution_inputs,
            basis_fields,
            basis_count,
        })
        .map_err(|error| self.contextualize_phiscf_collection_error(error))
        .context("failed to collect XSPH phiscf wfirdc contributions")
    }

    fn contextualize_phiscf_collection_error(&self, error: XsphError) -> anyhow::Error {
        if let XsphError::NonFiniteComplex {
            name: "xsect_phiscf_response_contribution",
            index,
            real,
            imaginary,
        } = error
        {
            if let Some(row) = self.rows.get(index) {
                return anyhow::anyhow!(
                    "XSPH phiscf contribution row {index} overflowed while accumulating response: orbital={} energy={} pole={} delta={} initial_kappa={} final_kappa={} pole_energy=({}, {}) wave_number=({}, {}) match_index={} wkb_index={} scale={} include_imaginary={} raw=({real}, {imaginary})",
                    row.plan_row.orbital_index_1based,
                    row.plan_row.energy_index_1based,
                    row.plan_row.pole_index_1based,
                    row.plan_row.dipole_delta,
                    row.plan_row.initial_kappa,
                    row.plan_row.final_kappa,
                    row.plan_row.pole.pole_energy.re,
                    row.plan_row.pole.pole_energy.im,
                    row.radial_setup.wave_number.re,
                    row.radial_setup.wave_number.im,
                    row.radial_setup.match_index_1based,
                    row.radial_setup.wkb_index_1based,
                    row.plan_row.rule.scale,
                    row.plan_row.rule.include_imaginary,
                );
            }
            return anyhow::anyhow!(
                "XSPH phiscf contribution row {index} overflowed while accumulating response, but the row metadata is missing: raw=({real}, {imaginary})"
            );
        }
        error.into()
    }
}

#[allow(dead_code)]
fn normal_xsect_phiscf_wfirdc_assembly(
    input: NormalXsectPhiscfWfirdcAssemblyInput<'_>,
) -> Result<NormalXsectPhiscfWfirdcAssembly> {
    let bound_orbital_count = input.occupied_table.orbital_energy_counts.len();
    ensure!(
        bound_orbital_count > 0,
        "XSPH phiscf wfirdc assembly requires at least one occupied orbital"
    );
    ensure!(
        input.active_len > 1,
        "XSPH phiscf wfirdc assembly requires at least two active radial rows"
    );
    ensure!(
        input.target_last_index_1based > 0 && input.target_last_index_1based <= input.active_len,
        "XSPH phiscf target last index {} must be in 1..={}",
        input.target_last_index_1based,
        input.active_len
    );
    ensure!(
        input.radii.len() >= input.active_len
            && input.exchange_correlation_potential.len() >= input.active_len
            && input.local_field.len() >= input.active_len,
        "XSPH phiscf radial inputs cannot supply active length {} (radii {}, vxc {}, fxc {})",
        input.active_len,
        input.radii.len(),
        input.exchange_correlation_potential.len(),
        input.local_field.len()
    );
    ensure!(
        input.bound_large_components.nrows() >= input.active_len
            && input.bound_small_components.nrows() >= input.active_len
            && input.bound_large_components.ncols() >= bound_orbital_count
            && input.bound_small_components.ncols() >= bound_orbital_count,
        "XSPH phiscf bound spinor shapes {:?}/{:?} cannot supply ({}, {})",
        input.bound_large_components.dim(),
        input.bound_small_components.dim(),
        input.active_len,
        bound_orbital_count
    );
    ensure!(
        input.bound_large_coefficients.ncols() >= bound_orbital_count
            && input.bound_small_coefficients.ncols() >= bound_orbital_count,
        "XSPH phiscf bound coefficient shapes {:?}/{:?} cannot supply {} orbitals",
        input.bound_large_coefficients.dim(),
        input.bound_small_coefficients.dim(),
        bound_orbital_count
    );
    ensure!(
        input.electron_counts.len() >= bound_orbital_count
            && input.valence_counts.len() >= bound_orbital_count
            && input.orbital_kappas.len() >= bound_orbital_count,
        "XSPH phiscf orbital metadata lengths xnel={} xnval={} kap={} cannot supply {} orbitals",
        input.electron_counts.len(),
        input.valence_counts.len(),
        input.orbital_kappas.len(),
        bound_orbital_count
    );

    let plan = xsph_xsect_phiscf_contribution_plan(XsphXsectPhiscfContributionPlanInput {
        momentum_squared: input.momentum_squared,
        edge_energy: input.edge_energy,
        chemical_potential: input.chemical_potential,
        hole_orbital_energy: input.hole_orbital_energy,
        scale_function: input.scale_function,
        orbital_kappas: input.orbital_kappas,
        orbital_energy_counts: input.occupied_table.orbital_energy_counts.view(),
        occupied_energies: input.occupied_table.occupied_energies.view(),
        occupation_fractions: input.occupied_table.occupation_fractions.view(),
        active_orbital_count: bound_orbital_count,
    })
    .context("failed to plan XSPH phiscf occupied-state contributions")?;
    ensure!(
        !plan.rows.is_empty(),
        "XSPH phiscf contribution plan produced no wfirdc rows"
    );

    let radial_prefix = Slice::from(..input.active_len);
    let orbital_prefix = Slice::from(..bound_orbital_count);
    let active_bound_large_rows = input
        .bound_large_components
        .slice_axis(Axis(0), radial_prefix);
    let active_bound_large = active_bound_large_rows.slice_axis(Axis(1), orbital_prefix);
    let active_bound_small_rows = input
        .bound_small_components
        .slice_axis(Axis(0), radial_prefix);
    let active_bound_small = active_bound_small_rows.slice_axis(Axis(1), orbital_prefix);
    let active_bound_large_coefficients = input
        .bound_large_coefficients
        .slice_axis(Axis(1), orbital_prefix);
    let active_bound_small_coefficients = input
        .bound_small_coefficients
        .slice_axis(Axis(1), orbital_prefix);
    let active_electron_counts = input.electron_counts.slice_axis(Axis(0), orbital_prefix);
    let active_valence_counts = input.valence_counts.slice_axis(Axis(0), orbital_prefix);
    let active_kappa = input.orbital_kappas.slice_axis(Axis(0), orbital_prefix);
    let active_radii = input.radii.slice_axis(Axis(0), radial_prefix);

    let mut rows = Vec::with_capacity(plan.rows.len());
    for plan_row in plan.rows {
        let radial_setup =
            xsph_xsect_phiscf_radial_solver_setup(XsphXsectPhiscfRadialSolverSetupInput {
                pole_energy: plan_row.pole.pole_energy,
                muffin_tin_radius: input.muffin_tin_radius,
                radii: active_radii,
                log_step: input.step,
                origin_shift: XSPH_LOUCKS_GRID_ORIGIN,
                active_len: input.active_len,
                target_last_index_1based: input.target_last_index_1based,
            })
            .context("failed to set up XSPH phiscf wfirdc radial indices")?;
        ensure!(
            radial_setup.match_index + 1 < input.active_len,
            "XSPH phiscf wfirdc radial match index {} leaves no following tail row in active length {}",
            radial_setup.match_index,
            input.active_len
        );

        let mut exchange_correlation_potential = input
            .exchange_correlation_potential
            .slice_axis(Axis(0), radial_prefix)
            .to_owned();
        let flat_tail = exchange_correlation_potential[radial_setup.match_index + 1];
        for row in radial_setup.match_index + 1..input.active_len {
            exchange_correlation_potential[row] = flat_tail;
        }
        let c3_potential = fovrg_c3_potential(FovrgC3PotentialInput {
            exchange_correlation_potential: exchange_correlation_potential.view(),
            radii: active_radii,
            target_kappa: plan_row.final_kappa,
            step: input.step,
            radial_match_index: radial_setup.match_index,
            active_len: input.active_len,
        })
        .context("failed to build XSPH phiscf wfirdc C3 potential")?;
        let orbital_setup = fovrg_orbital_setup(FovrgOrbitalSetupInput {
            bound_large_components: active_bound_large,
            bound_small_components: active_bound_small,
            electron_counts: active_electron_counts,
            valence_counts: active_valence_counts,
            kappa: active_kappa,
            target_kappa: plan_row.final_kappa,
            active_len: input.active_len,
            bound_orbital_count,
        })
        .context("failed to set up XSPH phiscf wfirdc orbital metadata")?;
        let mut orbital_lengths = orbital_setup.orbital_lengths;
        orbital_lengths[bound_orbital_count] = input.target_last_index_1based;

        rows.push(NormalXsectPhiscfWfirdcRow {
            plan_row,
            radial_setup,
            exchange_correlation_potential,
            c3_potential,
            kappa: orbital_setup.kappa,
            orbital_lengths,
        });
    }

    Ok(NormalXsectPhiscfWfirdcAssembly {
        rows,
        bound_large_components: active_bound_large.to_owned(),
        bound_small_components: active_bound_small.to_owned(),
        bound_large_coefficients: active_bound_large_coefficients.to_owned(),
        bound_small_coefficients: active_bound_small_coefficients.to_owned(),
        electron_counts: active_electron_counts.to_owned(),
        local_field: input
            .local_field
            .slice_axis(Axis(0), radial_prefix)
            .to_owned(),
        nuclear_charge: input.nuclear_charge,
        step: input.step,
        coarse_count: input.coarse_count,
        c3_scale: input.c3_scale,
        active_len: input.active_len,
        bound_orbital_count,
    })
}

fn extend_xcpot_potential(
    source: &Array1<Complex64>,
    radial_count: usize,
    label: &'static str,
) -> Result<Array1<Complex64>> {
    ensure!(
        !source.is_empty(),
        "XSPH xcpot {label} potential returned no radial rows"
    );
    ensure!(
        radial_count >= source.len(),
        "XSPH xcpot {label} potential length {} exceeds radial grid length {radial_count}",
        source.len()
    );
    let mut extended = Array1::from_elem(radial_count, source[source.len() - 1]);
    for (target, source) in extended.iter_mut().zip(source.iter()) {
        *target = *source;
    }
    Ok(extended)
}

fn empty_cell_reference_energy(
    input: &XsphInput,
    pot: &PotBinData,
    mesh: &InitialPhaseMesh,
    spin_count: usize,
) -> Result<Array2<Complex64>> {
    let energy_count = mesh.energies.len();
    let base_reference = Complex64::new(
        pot.scalars.interstitial_potential + input.vr0 / FEFF_HARTREE_EV,
        -input.vi0 / FEFF_HARTREE_EV,
    );
    let mut reference_energy =
        Array2::<Complex64>::from_elem((energy_count, spin_count), base_reference);
    for spin in 0..spin_count {
        xsph_phase_reference_tail(
            reference_energy.column_mut(spin),
            energy_count,
            mesh.horizontal_count,
            mesh.auxiliary_count,
        )
        .context("failed to finalize XSPH empty-cell reference-energy tail")?;
    }
    Ok(reference_energy)
}

fn phase_angular_limit_from_pot(
    input: &XsphInput,
    pot: &PotBinData,
    mesh: &InitialPhaseMesh,
    potential_index: usize,
) -> Result<usize> {
    ensure!(
        potential_index < input.lmaxph.len(),
        "XSPH lmaxph is shorter than pot.bin potential count"
    );
    let limit = xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
        energies: mesh.energies.view(),
        energy_count: mesh.energies.len(),
        auxiliary_count: mesh.auxiliary_count,
        muffin_tin_radius: pot.muffin_tin_radii[potential_index],
        max_angular_momentum: XSPH_PHASE_MAX_ANGULAR_MOMENTUM,
    })
    .context("failed to plan XSPH angular cutoff")?;
    Ok(limit.angular_limit)
}

fn phase_spin_selectors(caches: &XsphCachePaths, input: &XsphInput) -> Result<Vec<i32>> {
    let magnetized_source = input.spinph.iter().any(|spin| *spin != 0.0);
    let can_generate_two_spins = magnetized_source || active_hubbard_phase_requested(caches)?;
    let requested_spin = read_optional_global_input(&caches.global_inp)?
        .map(|global| global.control.ispin)
        .unwrap_or(if magnetized_source { 1 } else { 0 });

    if requested_spin.abs() == 1 && can_generate_two_spins {
        Ok(vec![-requested_spin.abs(), requested_spin.abs()])
    } else {
        Ok(vec![requested_spin])
    }
}

fn phase_potential_label(input: &XsphInput, potential_index: usize) -> String {
    input
        .pot_labels
        .get(potential_index)
        .filter(|label| label.len() <= 6 && label.is_ascii())
        .cloned()
        .unwrap_or_else(|| format!("E{potential_index:02}"))
}

fn write_initial_phase_mesh_sidecars(caches: &XsphCachePaths, input: &XsphInput) -> Result<usize> {
    if !emesh_sidecars_need_generation(caches) {
        return Ok(0);
    }
    let Some(mesh) = generate_initial_phase_mesh(caches, input)? else {
        return Ok(0);
    };

    let mut written = 0_usize;
    if emesh_dat_needs_generation(&caches.emesh_dat) {
        let data = emesh_dat_from_initial_phase_mesh(input.control.ispec, &mesh)?;
        write_emesh_cache(&caches.emesh_dat, &data)?;
        written += 1;
    }
    if emesh_bin_needs_generation(&caches.emesh_bin) {
        let data = emesh_bin_from_initial_phase_mesh(&mesh)?;
        write_emesh_bin_cache(&caches.emesh_bin, &data)?;
        written += 1;
    }
    Ok(written)
}

fn write_or_generate_emesh_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<usize> {
    let mut written = 0_usize;
    if emesh_dat_needs_generation(&caches.emesh_dat) {
        let data = emesh_dat_from_phase_bin(phase, input.control.ispec)
            .context("failed to generate emesh.dat from phase.bin")?;
        write_emesh_cache(&caches.emesh_dat, &data)?;
        written += 1;
    } else {
        let data = read_emesh_dat(&caches.emesh_dat)
            .with_context(|| format!("failed to read {}", caches.emesh_dat.display()))?;
        write_emesh_cache(&caches.emesh_dat, &data)?;
        written += 1;
    }
    if emesh_bin_needs_generation(&caches.emesh_bin) {
        let data = emesh_bin_from_phase_bin(phase)
            .context("failed to generate emesh.bin from phase.bin")?;
        write_emesh_bin_cache(&caches.emesh_bin, &data)?;
        written += 1;
    } else {
        let data = read_emesh_bin(&caches.emesh_bin)
            .with_context(|| format!("failed to read {}", caches.emesh_bin.display()))?;
        write_emesh_bin_cache(&caches.emesh_bin, &data)?;
        written += 1;
    }
    Ok(written)
}

fn emesh_sidecars_need_generation(caches: &XsphCachePaths) -> bool {
    emesh_dat_needs_generation(&caches.emesh_dat) || emesh_bin_needs_generation(&caches.emesh_bin)
}

fn emesh_dat_needs_generation(path: &Path) -> bool {
    !path.is_file() || read_emesh_dat(path).is_err()
}

fn emesh_bin_needs_generation(path: &Path) -> bool {
    !path.is_file() || read_emesh_bin(path).is_err()
}

fn can_generate_initial_phase_mesh_handoff(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<bool> {
    match generate_initial_phase_mesh(caches, input) {
        Ok(mesh) => Ok(mesh.is_some()),
        Err(error) if is_unsupported_initial_phase_mesh(&error) => Ok(false),
        Err(error) => Err(error),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct InitialPhaseMesh {
    edge: f64,
    energies: Array1<Complex64>,
    horizontal_count: usize,
    auxiliary_count: usize,
    fermi_index_1based: usize,
    zero_index: usize,
}

fn generate_initial_phase_mesh(
    caches: &XsphCachePaths,
    input: &XsphInput,
) -> Result<Option<InitialPhaseMesh>> {
    if !caches.pot_bin.is_file() {
        return Ok(None);
    }

    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    generate_initial_phase_mesh_from_pot(caches, input, &pot).map(Some)
}

fn is_unsupported_initial_phase_mesh(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<XsphError>(),
            Some(XsphError::UnsupportedPhaseMeshSpectroscopy { .. })
        )
    })
}

fn is_unsupported_source_phase_generation(error: &anyhow::Error) -> bool {
    is_unsupported_initial_phase_mesh(error)
        || error.chain().any(|cause| {
            matches!(
                cause.downcast_ref::<ExchangeError>(),
                Some(ExchangeError::NegativeRadicand { .. })
            ) || is_incomplete_source_phase_handoff(cause)
        })
}

fn is_incomplete_source_phase_generation(error: &anyhow::Error) -> bool {
    error.chain().any(is_incomplete_source_phase_handoff)
}

fn is_incomplete_source_phase_handoff(error: &(dyn std::error::Error + 'static)) -> bool {
    let text = error.to_string();
    (text.contains("XSPH bound orbital count")
        && text.contains("exceeds pot/config handoff shapes"))
        || text.contains("has incomplete radial/coefficient handoff")
        || text.contains("has no radial handoff but core count")
        || text.contains("has no active bound orbital handoffs")
        || text.contains("is not trailing and cannot be omitted")
}

/// FEFF shifts both the phase edge and XSECT `emu` by the EXCHANGE `vr0`
/// offset. `pot.bin` stores the unshifted chemical potential in Hartree while
/// `xsph.inp` stores `vr0` in eV.
fn xsph_chemical_potential_hartree(input: &XsphInput, pot: &PotBinData) -> f64 {
    pot.scalars.edge_position - input.vr0 / FEFF_HARTREE_EV
}

/// Restore the material records that FEFF's `wthead` places in `xsect.dat`.
///
/// FF2X forwards these records into `xmu.dat`/`chi.dat`, where SFCONV's
/// `SO2CONV` reader consumes the fixed-width `Gam_ch`, `Rs_int`, `Vint`,
/// `Mu`, and `kf` fields. Keeping them on the typed XSPH handoff avoids
/// weakening the generic spectrum parser when a producer emits an incomplete
/// header.
fn xsph_xsect_material_header_titles(
    input: &XsphInput,
    pot: &PotBinData,
    material_chemical_potential_hartree: f64,
) -> Result<Vec<String>> {
    let core_hole_width_ev = input.grid.gamach;
    let wigner_seitz_radius = pot.scalars.density_radius;
    let interstitial_potential_ev = pot.scalars.interstitial_potential * FEFF_HARTREE_EV;
    let chemical_potential_ev = material_chemical_potential_hartree * FEFF_HARTREE_EV;
    let fermi_wave_number_inverse_angstrom = pot.scalars.fermi_momentum / FEFF_BOHR_ANGSTROM;
    for (label, value) in [
        ("Gam_ch", core_hole_width_ev),
        ("Rs_int", wigner_seitz_radius),
        ("Vint", interstitial_potential_ev),
        ("Mu", chemical_potential_ev),
        ("kf", fermi_wave_number_inverse_angstrom),
    ] {
        ensure!(
            value.is_finite(),
            "XSPH xsect material header field {label} must be finite, got {value}"
        );
    }

    let mut titles = pot.titles.clone();
    titles.push(format!(
        "Gam_ch={core_hole_width_ev:9.3E} exch Vi={:9.3E} Vr={:9.3E}",
        input.vi0, input.vr0
    ));
    titles.push(format!(
        "Mu={chemical_potential_ev:10.3E} kf={fermi_wave_number_inverse_angstrom:9.3E} \
         Vint={interstitial_potential_ev:10.3E} Rs_int={wigner_seitz_radius:6.3}"
    ));
    Ok(titles)
}

fn generate_initial_phase_mesh_from_pot(
    caches: &XsphCachePaths,
    input: &XsphInput,
    pot: &PotBinData,
) -> Result<InitialPhaseMesh> {
    let edge = pot.scalars.fermi_level - input.vr0 / FEFF_HARTREE_EV;
    if tdlda_xsectd_branch_requested(input) {
        return generate_tdlda_phase_mesh(edge);
    }
    if input.electronic_temperature > 0.0 {
        let grid = if input.control.i_grid != 0 {
            Some(read_grid_input(caches)?)
        } else {
            None
        };
        return generate_thermal_phase_mesh(input, pot, edge, grid.as_ref());
    }
    if input.control.i_grid != 0 {
        let grid = read_grid_input(caches)?;
        return generate_user_phase_mesh(input, edge, &grid);
    }

    let nrixs_phase_mesh_capacity = uses_nrixs_phase_mesh_capacity(caches, input)?;
    if nrixs_phase_mesh_capacity
        && input.grid.xkmax <= 0.0
        && jas_phase_mesh_spectroscopy_supported(input.control.ispec)
    {
        return generate_jas_phase_mesh(input, pot, edge);
    }
    generate_default_phase_mesh(input, pot, edge, nrixs_phase_mesh_capacity)
}

fn jas_phase_mesh_spectroscopy_supported(spectroscopy: i32) -> bool {
    spectroscopy != 2 && spectroscopy < 3
}

fn generate_jas_phase_mesh(
    input: &XsphInput,
    pot: &PotBinData,
    edge: f64,
) -> Result<InitialPhaseMesh> {
    let mesh = xsph_jas_phase_energy_mesh(XsphJasPhaseEnergyMeshInput {
        spectroscopy: input.control.ispec,
        edge,
        constant_imaginary: input.vi0 / FEFF_HARTREE_EV,
        core_hole_broadening: input.grid.gamach / FEFF_HARTREE_EV,
        core_valence_separation: pot.scalars.core_valence_energy,
        max_wave_number: input.grid.xkmax.abs() * FEFF_BOHR_ANGSTROM,
        wave_number_step: input.grid.xkstep * FEFF_BOHR_ANGSTROM,
        xanes_energy_step: input.grid.vixan / FEFF_HARTREE_EV,
        horizontal_capacity: XSPH_NRIXS_PHASE_MESH_CAPACITY,
    })
    .context("failed to generate JAS/NRIXS XSPH phase-energy mesh")?;
    Ok(InitialPhaseMesh {
        edge,
        energies: mesh.energies,
        horizontal_count: mesh.horizontal_count,
        auxiliary_count: 0,
        fermi_index_1based: mesh.zero_index + 1,
        zero_index: mesh.zero_index,
    })
}

fn generate_default_phase_mesh(
    input: &XsphInput,
    pot: &PotBinData,
    edge: f64,
    nrixs_phase_mesh_capacity: bool,
) -> Result<InitialPhaseMesh> {
    let capacity = match input.control.ispec {
        2 => XSPH_COMPILED_PHASE_MESH_CAPACITY,
        3 | -3 => XSPH_DANES_PHASE_MESH_CAPACITY,
        4 => XSPH_FPRIME_PHASE_MESH_CAPACITY,
        _ if nrixs_phase_mesh_capacity => XSPH_NRIXS_PHASE_MESH_CAPACITY,
        _ => XSPH_DEFAULT_PHASE_MESH_CAPACITY,
    };
    let max_wave_number = input.grid.xkmax * FEFF_BOHR_ANGSTROM;
    let wave_number_step = input.grid.xkstep * FEFF_BOHR_ANGSTROM;
    let xanes_energy_step = input.grid.vixan / FEFF_HARTREE_EV;
    if input.control.ispec == 5 {
        let mesh = xsph_rhorrp_phase_energy_mesh(XsphRhorrpPhaseEnergyMeshInput {
            edge,
            core_valence_separation: pot.scalars.core_valence_energy,
            scf_temperature: input.electronic_temperature,
            capacity,
        })
        .context("failed to generate NRIXS/RHORRP XSPH phase-energy mesh")?;
        return Ok(InitialPhaseMesh {
            edge,
            energies: mesh.energies,
            horizontal_count: mesh.contour_count,
            auxiliary_count: 0,
            fermi_index_1based: 0,
            zero_index: 0,
        });
    }

    let legacy_fixed_xanes_mesh =
        input.control.ispec == 1 && input.source_format.uses_legacy_fixed_xanes_mesh();
    let xanes_vertical = if input.control.ispec == 1 {
        let xloss = (input.grid.gamach / FEFF_HARTREE_EV / 2.0 + input.vi0 / FEFF_HARTREE_EV)
            .max(0.02 / FEFF_HARTREE_EV);
        Some(
            xsph_vertical_energy_mesh_84(xloss, XSPH_COMPILED_PHASE_MESH_CAPACITY)
                .context("failed to generate default XANES vertical phase-energy mesh")?,
        )
    } else {
        None
    };
    let mesh_capacity = xanes_vertical.as_ref().map_or(capacity, |vertical| {
        if legacy_fixed_xanes_mesh {
            XSPH_LEGACY_XANES_PHASE_MESH_CAPACITY
        } else {
            capacity + vertical.len() + 2
        }
    });
    let mut mesh = xsph_phase_energy_mesh_84(XsphPhaseEnergyMesh84Input {
        spectroscopy: input.control.ispec,
        edge,
        reference_energy: xsph_chemical_potential_hartree(input, pot),
        constant_imaginary: input.vi0 / FEFF_HARTREE_EV,
        core_hole_broadening: input.grid.gamach / FEFF_HARTREE_EV,
        core_valence_separation: pot.scalars.core_valence_energy,
        max_wave_number,
        wave_number_step,
        xanes_energy_step,
        capacity: mesh_capacity,
    })
    .context("failed to generate default XSPH phase-energy mesh")?;
    if let Some(vertical) = xanes_vertical {
        let horizontal_count = mesh.horizontal_count.min(capacity);
        ensure!(
            mesh.zero_index < horizontal_count,
            "default XANES zero index {} exceeds horizontal count {horizontal_count}",
            mesh.zero_index
        );
        ensure!(
            mesh.energies.len() >= horizontal_count,
            "default XANES mesh contains {} rows but needs {horizontal_count} horizontal rows",
            mesh.energies.len()
        );
        let mut energies = mesh
            .energies
            .iter()
            .take(horizontal_count)
            .copied()
            .collect::<Vec<_>>();
        energies.extend(
            vertical
                .iter()
                .map(|&energy| energy + Complex64::new(edge, 0.0)),
        );
        mesh.energies = Array1::from_vec(energies);
        mesh.horizontal_count = horizontal_count;
        mesh.extension_count = 0;
    }
    Ok(InitialPhaseMesh {
        edge,
        energies: mesh.energies,
        horizontal_count: mesh.horizontal_count,
        auxiliary_count: mesh.extension_count,
        fermi_index_1based: mesh.zero_index + 1,
        zero_index: mesh.zero_index,
    })
}

fn generate_tdlda_phase_mesh(edge: f64) -> Result<InitialPhaseMesh> {
    let mut energies =
        Vec::with_capacity(XSPH_TDLDA_MESH_HORIZONTAL_COUNT + XSPH_TDLDA_MESH_EXTRA_COUNT);
    let left = XSPH_TDLDA_MESH_LEFT_EV / FEFF_HARTREE_EV;
    let right = XSPH_TDLDA_MESH_RIGHT_EV / FEFF_HARTREE_EV;
    let extra_right = XSPH_TDLDA_MESH_EXTRA_RIGHT_EV / FEFF_HARTREE_EV;
    let horizontal_step = (right - left) / (XSPH_TDLDA_MESH_HORIZONTAL_COUNT as f64 - 1.0);
    let extra_step = (extra_right - right) / (XSPH_TDLDA_MESH_EXTRA_COUNT as f64 - 1.0);

    for index in 0..XSPH_TDLDA_MESH_HORIZONTAL_COUNT {
        energies.push(Complex64::new(left + horizontal_step * index as f64, 0.0));
    }
    for index in 1..=XSPH_TDLDA_MESH_EXTRA_COUNT {
        energies.push(Complex64::new(right + extra_step * index as f64, 0.0));
    }

    Ok(InitialPhaseMesh {
        edge,
        horizontal_count: energies.len(),
        energies: Array1::from_vec(energies),
        auxiliary_count: 0,
        fermi_index_1based: 0,
        zero_index: 0,
    })
}

fn uses_nrixs_phase_mesh_capacity(caches: &XsphCachePaths, input: &XsphInput) -> Result<bool> {
    nrixs_xsectjas_requested(caches, input)
}

fn nrixs_xsectjas_requested(caches: &XsphCachePaths, input: &XsphInput) -> Result<bool> {
    if input.control.l2lp == XSPH_NRIXS_L2LP_SENTINEL {
        return Ok(true);
    }
    Ok(
        read_optional_global_input(&caches.global_inp)?.is_some_and(|global| {
            global.control.do_nrixs == 1 || global.control.l2lp == XSPH_NRIXS_L2LP_SENTINEL
        }),
    )
}

fn generate_user_phase_mesh(
    input: &XsphInput,
    edge: f64,
    grid: &GridInput,
) -> Result<InitialPhaseMesh> {
    with_user_phase_grid_records(grid, |records| {
        let mesh = xsph_phase_energy_mesh_user(XsphPhaseUserGridInput {
            spectroscopy: input.control.ispec,
            edge,
            constant_imaginary: input.vi0 / FEFF_HARTREE_EV,
            core_hole_broadening: input.grid.gamach / FEFF_HARTREE_EV,
            records,
            capacity: XSPH_COMPILED_PHASE_MESH_CAPACITY,
        })
        .context("failed to generate user-defined XSPH phase-energy mesh")?;
        Ok(InitialPhaseMesh {
            edge,
            energies: mesh.energies,
            horizontal_count: mesh.horizontal_count,
            auxiliary_count: mesh.extension_count,
            fermi_index_1based: mesh.zero_index + 1,
            zero_index: mesh.zero_index,
        })
    })
}

fn generate_thermal_phase_mesh(
    input: &XsphInput,
    pot: &PotBinData,
    edge: f64,
    grid: Option<&GridInput>,
) -> Result<InitialPhaseMesh> {
    if let Some(grid) = grid {
        return with_user_phase_grid_records(grid, |records| {
            generate_thermal_phase_mesh_from_records(input, pot, edge, Some(records))
        });
    }
    generate_thermal_phase_mesh_from_records(input, pot, edge, None)
}

fn generate_thermal_phase_mesh_from_records<'a>(
    input: &XsphInput,
    pot: &PotBinData,
    edge: f64,
    records: Option<&'a [XsphPhaseUserGridRecord<'a>]>,
) -> Result<InitialPhaseMesh> {
    let mesh = xsph_thermal_phase_energy_mesh(XsphThermalPhaseEnergyMeshInput {
        edge,
        constant_imaginary: input.vi0 / FEFF_HARTREE_EV,
        core_hole_broadening: input.grid.gamach / FEFF_HARTREE_EV,
        core_valence_separation: pot.scalars.core_valence_energy,
        electronic_temperature: input.electronic_temperature,
        user_records: records,
        capacity: XSPH_COMPILED_PHASE_MESH_CAPACITY,
    })
    .context("failed to generate finite-temperature XSPH phase-energy mesh")?;
    Ok(InitialPhaseMesh {
        edge,
        energies: mesh.energies,
        horizontal_count: mesh.horizontal_count,
        auxiliary_count: 0,
        fermi_index_1based: mesh.zero_index + 1,
        zero_index: mesh.zero_index,
    })
}

#[derive(Debug, Clone, PartialEq)]
enum OwnedUserPhaseGridRecord {
    Regular(XsphPhaseUserRegularGrid),
    User(Array1<Complex64>),
}

fn read_grid_input(caches: &XsphCachePaths) -> Result<GridInput> {
    read_grid_inp(&caches.grid_inp)
        .with_context(|| format!("failed to read {}", caches.grid_inp.display()))
}

fn with_user_phase_grid_records<T>(
    grid: &GridInput,
    build: impl for<'a> FnOnce(&'a [XsphPhaseUserGridRecord<'a>]) -> Result<T>,
) -> Result<T> {
    let owned = owned_user_phase_grid_records(grid);
    let records = owned
        .iter()
        .map(|record| match record {
            OwnedUserPhaseGridRecord::Regular(record) => XsphPhaseUserGridRecord::Regular(*record),
            OwnedUserPhaseGridRecord::User(points) => XsphPhaseUserGridRecord::User(points.view()),
        })
        .collect::<Vec<_>>();
    build(&records)
}

fn owned_user_phase_grid_records(grid: &GridInput) -> Vec<OwnedUserPhaseGridRecord> {
    grid.records
        .iter()
        .map(|record| match record {
            GridRecord::Regular(record) => {
                OwnedUserPhaseGridRecord::Regular(XsphPhaseUserRegularGrid {
                    kind: user_phase_grid_kind(record.kind),
                    minimum: user_phase_grid_minimum(record.minimum),
                    maximum: record.maximum,
                    step: record.step,
                })
            }
            GridRecord::User(record) => OwnedUserPhaseGridRecord::User(Array1::from_iter(
                record
                    .points
                    .iter()
                    .map(|point| Complex64::new(point.real, point.imaginary)),
            )),
        })
        .collect()
}

fn user_phase_grid_kind(kind: GridKind) -> XsphPhaseUserGridKind {
    match kind {
        GridKind::Energy => XsphPhaseUserGridKind::Energy,
        GridKind::WaveNumber => XsphPhaseUserGridKind::WaveNumber,
        GridKind::Exponential => XsphPhaseUserGridKind::Exponential,
    }
}

fn user_phase_grid_minimum(minimum: GridMinimum) -> XsphPhaseUserGridMinimum {
    match minimum {
        GridMinimum::Value(value) => XsphPhaseUserGridMinimum::Value(value),
        GridMinimum::Last => XsphPhaseUserGridMinimum::Last,
    }
}

fn emesh_dat_from_initial_phase_mesh(
    spectrum: i32,
    mesh: &InitialPhaseMesh,
) -> Result<EmeshDatData> {
    let data = EmeshDatData {
        edge_hartree: mesh.edge,
        bohr_angstrom: FEFF_BOHR_ANGSTROM,
        edge_ev: mesh.edge * FEFF_HARTREE_EV,
        spectrum,
        fermi_index: mesh.fermi_index_1based,
        indices: Array1::from_iter(1..=mesh.energies.len()),
        energy_ev: mesh
            .energies
            .iter()
            .map(|energy| energy.re * FEFF_HARTREE_EV)
            .collect(),
        wave_number_inverse_angstrom: mesh
            .energies
            .iter()
            .map(|energy| wave_number_from_hartree(energy.re - mesh.edge) / FEFF_BOHR_ANGSTROM)
            .collect(),
    };
    Ok(refeff_io::parse_emesh_dat(&refeff_io::emesh_dat_string(
        &data,
    )?)?)
}

fn emesh_bin_from_initial_phase_mesh(mesh: &InitialPhaseMesh) -> Result<EmeshBinData> {
    let data = EmeshBinData {
        point_count_declared: mesh.energies.len(),
        horizontal_count: mesh.horizontal_count,
        danes_extension_count: mesh.auxiliary_count,
        energy_hartree: mesh.energies.clone(),
    };
    refeff_io::parse_emesh_bin(&refeff_io::emesh_bin_bytes(&data)?)
        .context("failed to build emesh.bin from XSPH phase-energy mesh")
}

fn generate_axafs_dat(phase: &PhaseBinData, xsect: &XsectDatData) -> Result<Option<AxafsDatData>> {
    let handoff = xsect_dat_ff2x_handoff(xsect, xsect.scalars.amplitude_reduction, 0)
        .context("failed to prepare xsect.dat data for AXAFS generation")?;
    ensure!(
        phase.main_energy_count == handoff.main_energy_count,
        "phase.bin ne1 {} does not match xsect.dat ne1 {} for AXAFS generation",
        phase.main_energy_count,
        handoff.main_energy_count
    );
    let phase_fermi_index = usize::try_from(phase.fermi_index)
        .context("phase.bin ik0 is negative for AXAFS generation")?;
    ensure!(
        phase_fermi_index == handoff.fermi_index_1based,
        "phase.bin ik0 {} does not match xsect.dat ik0 {} for AXAFS generation",
        phase.fermi_index,
        handoff.fermi_index_1based
    );
    let axafs = match xsph_axafs(XsphAxafsInput {
        energies: phase.energy_grid.view(),
        cross_section: handoff.cross_section.view(),
        fermi_energy: handoff.chemical_potential_hartree,
        horizontal_count: handoff.main_energy_count,
        zero_wave_index: handoff.fermi_index,
    }) {
        Ok(axafs) => axafs,
        Err(XsphError::InsufficientAxafsPoints { .. }) => return Ok(None),
        Err(error) => return Err(error).context("failed to generate XSPH AXAFS table"),
    };
    axafs_dat_from_xsph_axafs(&axafs)
        .map(Some)
        .context("failed to build axafs.dat data")
}

fn generate_mpse_dat(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<Option<MpseDatData>> {
    if !caches.pot_bin.is_file() {
        return Ok(None);
    }
    let Ok(pot) = read_pot_bin(&caches.pot_bin) else {
        return Ok(None);
    };
    let excitation_poles = xsph_excitation_poles_from_loss(caches, input, input.control.ixc)?;
    if input.control.i_plsmn > 0 && input.control.ixc == 0 && excitation_poles.is_none() {
        // A phase/pot cache carries the already-renormalized self energy but
        // not the CSigZ pole data needed to reconstruct Z.  Treat a missing
        // loss-function source as unavailable instead of replacing a valid
        // cached MPSE table with identity renormalization columns.
        return Ok(None);
    }
    Ok(mpse_dat_from_phase_and_pot(input, phase, &pot, excitation_poles.as_deref()).ok())
}

fn mpse_dat_from_phase_and_pot(
    input: &XsphInput,
    phase: &PhaseBinData,
    pot: &PotBinData,
    excitation_poles: Option<&[ExcitationPole]>,
) -> Result<MpseDatData> {
    let potential_index = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin contains no potential blocks for mpse.dat generation")?;
    ensure!(
        potential_index < pot.potential_count(),
        "phase.bin final potential index {} exceeds pot.bin potential count {}",
        potential_index,
        pot.potential_count()
    );
    ensure!(
        pot.jump_mode <= 1,
        "mpse.dat generation does not have FEFF vjump state for pot.bin jumprm={}",
        pot.jump_mode
    );
    let fermi_index = usize::try_from(phase.fermi_index)
        .context("phase.bin ik0 is negative for mpse.dat generation")?;
    ensure!(
        fermi_index < phase.main_energy_count,
        "phase.bin ik0 {} leaves no above-Fermi mpse.dat rows for ne1 {}",
        phase.fermi_index,
        phase.main_energy_count
    );
    ensure!(
        phase.main_energy_count <= phase.energy_count,
        "phase.bin ne1 {} exceeds ne {} for mpse.dat generation",
        phase.main_energy_count,
        phase.energy_count
    );

    let fixed = fix_potential_grid(PotentialGridInput {
        muffin_tin_radius: pot.muffin_tin_radii[potential_index],
        electron_density: pot.electron_density.column(potential_index),
        total_potential: pot.total_potential.column(potential_index),
        magnetization: pot.magnetization_density.column(potential_index),
        interstitial_potential: pot.scalars.interstitial_potential,
        interstitial_density: pot.scalars.interstitial_density,
        original_delta: LOUCKS_DELTA,
        new_delta: input.grid.rgrd,
        jump_mode: pot.jump_mode,
        potential_jump: 0.0,
        output_len: pot.total_potential.nrows(),
    })
    .context("failed to prepare XSPH fixed potential grid for mpse.dat")?;
    let reference_index_1based = fixed.interstitial_index + 1;
    ensure!(
        reference_index_1based <= fixed.total_potential.len(),
        "fixed potential reference index {} exceeds radial length {}",
        reference_index_1based,
        fixed.total_potential.len()
    );
    let reference_potential = fixed.total_potential[reference_index_1based - 1];
    let summary = xsph_phase_self_energy_summary(refeff_core::XsphPhaseSelfEnergySummaryInput {
        electron_density: fixed.charge_density.view(),
        reference_index_1based,
    })
    .context("failed to derive XSPH self-energy density summary")?;
    let many_pole_self_energy = xsph_many_pole_self_energy_for_potential(
        input,
        excitation_poles,
        fixed.charge_density.view(),
        reference_index_1based,
        input.control.ixc,
    )?;
    // FEFF's `xcpot` leaves scalar `ZRnrm` at the final `Rs1` sample after
    // building the MPSE table, then writes that value at the interstitial
    // point.  For radial MPSE (`iPl == 2`) this is `RsMax`, not `RsInt`.
    let renormalization_radius = if many_pole_self_energy.is_some() {
        Some(
            xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
                plasmon_selector: input.control.i_plsmn,
                density: fixed.charge_density.view(),
                radial_match_index_1based: fixed.interstitial_index,
            })
            .context("failed to prepare XSPH mpse.dat CSigZ density grid")?
            .radii[XCPOT_MPSE_GRID_POINTS - 1],
        )
    } else {
        None
    };

    let mut energy_ev = Vec::with_capacity(phase.main_energy_count - fermi_index);
    let mut self_energy = Vec::with_capacity(phase.main_energy_count - fermi_index);
    let mut renormalization = Vec::with_capacity(phase.main_energy_count - fermi_index);
    let mut renormalization_magnitude = Vec::with_capacity(phase.main_energy_count - fermi_index);
    let mut renormalization_phase = Vec::with_capacity(phase.main_energy_count - fermi_index);
    let mut inelastic_mean_free_path = Vec::with_capacity(phase.main_energy_count - fermi_index);

    for energy in fermi_index..phase.main_energy_count {
        let em = phase.energy_grid[energy];
        let energy_above_fermi_hartree = em.re - phase.scalars.fermi_level;
        let delta = phase.reference_energy[(energy, 0)] - Complex64::new(reference_potential, 0.0);
        ensure!(
            energy_above_fermi_hartree > 0.0,
            "phase.bin energy row {} is not above the Fermi level for mpse.dat generation",
            energy + 1
        );
        ensure!(
            delta.im != 0.0,
            "phase.bin reference-energy row {} has zero imaginary self-energy for mpse.dat generation",
            energy + 1
        );

        energy_ev.push(energy_above_fermi_hartree * FEFF_HARTREE_EV);
        self_energy.push(delta * FEFF_HARTREE_EV);
        let row_renormalization = if let Some(many_pole) = &many_pole_self_energy {
            many_pole.renormalization(
                em,
                phase.scalars.fermi_level,
                renormalization_radius
                    .context("active XSPH MPSE is missing its CSigZ density radius")?,
            )?
        } else {
            Complex64::new(1.0, 0.0)
        };
        renormalization.push(row_renormalization);
        renormalization_magnitude.push(row_renormalization.norm());
        renormalization_phase.push(row_renormalization.arg());
        inelastic_mean_free_path
            .push((energy_above_fermi_hartree / 2.0).sqrt() / delta.im.abs() * FEFF_BOHR_ANGSTROM);
    }

    Ok(MpseDatData {
        header_lines: vec![mpse_header_line(
            summary.wigner_seitz_radius,
            summary.plasma_frequency_ev,
        )?],
        energy_ev: Array1::from_vec(energy_ev),
        self_energy: Array1::from_vec(self_energy),
        renormalization: Some(Array1::from_vec(renormalization)),
        renormalization_magnitude: Some(Array1::from_vec(renormalization_magnitude)),
        renormalization_phase: Some(Array1::from_vec(renormalization_phase)),
        inelastic_mean_free_path: Some(Array1::from_vec(inelastic_mean_free_path)),
    })
}

fn mpse_header_line(wigner_seitz_radius: f64, plasma_frequency_ev: f64) -> Result<String> {
    let mut line = "#HD#".to_string();
    write_fortran_zero_scaled_exp(&mut line, wigner_seitz_radius, 21, 10)?;
    write_fortran_zero_scaled_exp(&mut line, plasma_frequency_ev, 21, 10)?;
    line.push(' ');
    Ok(line)
}

fn read_input(work_dir: &Path) -> Result<XsphInput> {
    let input_path = work_dir.join("xsph.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    XsphInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn read_optional_global_input(path: &Path) -> Result<Option<GlobalInput>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    GlobalInput::parse_str(path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
        .map(Some)
}

fn write_phase_cache(path: &Path, data: &PhaseBinData) -> Result<()> {
    write_phase_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_preserve_rl_dat(
    caches: &XsphCachePaths,
    input: &XsphInput,
    generated: Option<&XsphRlDatData>,
) -> Result<usize> {
    if !input.print_rl {
        return Ok(0);
    }

    if let Some(data) = generated {
        write_xsph_rl_cache(&caches.rl_dat, data)?;
        return Ok(1);
    }

    if caches.rl_dat.is_file() {
        let data = read_xsph_rl_dat(&caches.rl_dat)
            .with_context(|| format!("failed to read {}", caches.rl_dat.display()))?;
        write_xsph_rl_cache(&caches.rl_dat, &data)?;
        return Ok(1);
    }

    bail!("XSPH PrintRl output requires rl.dat handoff or source-generated normal-potential phase");
}

fn write_xsph_rl_cache(path: &Path, data: &XsphRlDatData) -> Result<()> {
    write_xsph_rl_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_stale_or_missing_phase_text_sidecars(
    caches: &XsphCachePaths,
    input: &XsphInput,
    phase: &PhaseBinData,
) -> Result<usize> {
    if input.control.ipr2 < 2 {
        return Ok(0);
    }

    let mut written = 0_usize;
    for potential_index in 0..phase.potential_count() {
        let phase_path = caches.phase_dat(potential_index);
        let phase_text = phase_text_dat_string(phase, potential_index)?;
        if phase_text_sidecar_needs_rewrite(&phase_path, &phase_text) {
            write_phase_text_file(&phase_path, &phase_text)?;
            written += 1;
        }

        let phmin_path = caches.phmin_dat(potential_index);
        let phmin_text = phmin_text_dat_string(phase, potential_index)?;
        if phase_text_sidecar_needs_rewrite(&phmin_path, &phmin_text) {
            write_phase_text_file(&phmin_path, &phmin_text)?;
            written += 1;
        }
    }
    Ok(written)
}

fn phase_text_sidecar_rewrite_count(
    caches: &XsphCachePaths,
    phase: &PhaseBinData,
) -> Result<usize> {
    let mut count = 0_usize;
    for potential_index in 0..phase.potential_count() {
        let phase_text = phase_text_dat_string(phase, potential_index)?;
        if phase_text_sidecar_needs_rewrite(&caches.phase_dat(potential_index), &phase_text) {
            count += 1;
        }

        let phmin_text = phmin_text_dat_string(phase, potential_index)?;
        if phase_text_sidecar_needs_rewrite(&caches.phmin_dat(potential_index), &phmin_text) {
            count += 1;
        }
    }
    Ok(count)
}

fn phase_text_sidecar_needs_rewrite(path: &Path, expected: &str) -> bool {
    match std::fs::read_to_string(path) {
        Ok(existing) => existing != expected,
        Err(error) if error.kind() == ErrorKind::NotFound => true,
        Err(_) => true,
    }
}

fn write_phase_text_file(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).with_context(|| format!("failed to write {}", path.display()))
}

fn phase_text_dat_string(phase: &PhaseBinData, potential_index: usize) -> Result<String> {
    let potential = phase
        .potentials
        .get(potential_index)
        .with_context(|| format!("phase.bin missing potential {potential_index}"))?;
    let mut out = String::new();
    writeln!(
        out,
        "#  {:4}{:4}{:4}   unique pot,  lmax, ne",
        potential_index, potential.lmax, phase.energy_count
    )?;
    for energy_index in 0..phase.energy_count {
        let energy = phase.energy_grid[energy_index];
        let reference = phase.reference_energy[(energy_index, 0)];
        let momentum = ((energy - reference) * (2.0 * FEFF_HARTREE_EV)).sqrt();
        validate_phase_text_complex("phase energy", energy)?;
        validate_phase_text_complex("phase reference energy", reference)?;
        validate_phase_text_complex("phase momentum", momentum)?;

        writeln!(
            out,
            "#    ie      energy(eV)     re(eref)(eV)      im(eref)(eV)         re(p)(eV/c)         im(p)(eV/c)"
        )?;
        write!(out, " {:4}", energy_index + 1)?;
        write_phase_field(&mut out, energy.re * FEFF_HARTREE_EV, 14, 6)?;
        write_phase_field(&mut out, reference.re * FEFF_HARTREE_EV, 14, 6)?;
        write_phase_field(&mut out, reference.im * FEFF_HARTREE_EV, 14, 6)?;
        write_phase_field(&mut out, momentum.re, 14, 6)?;
        write_phase_field(&mut out, momentum.im, 14, 6)?;
        out.push('\n');

        out.push(' ');
        for angular_momentum in 0..=potential.lmax {
            let shift = phase_shift_or_zero(potential, energy_index, angular_momentum);
            validate_phase_text_complex("phase shift", shift)?;
            write_phase_field(&mut out, shift.re, 14, 6)?;
            write_phase_field(&mut out, shift.im, 14, 6)?;
        }
        out.push('\n');
    }
    Ok(out)
}

fn phmin_text_dat_string(phase: &PhaseBinData, potential_index: usize) -> Result<String> {
    let potential = phase
        .potentials
        .get(potential_index)
        .with_context(|| format!("phase.bin missing potential {potential_index}"))?;
    let mut out = String::new();
    writeln!(
        out,
        "#  {:4}{:4}{:4}   unique pot,  lmax, ne",
        potential_index, potential.lmax, phase.energy_count
    )?;
    writeln!(
        out,
        "# energy(eV)    re(eref)(eV)   re(p)(eV/c)    phase(0) phase(1) phase(2)"
    )?;
    for energy_index in 0..phase.energy_count {
        let energy = phase.energy_grid[energy_index];
        let reference = phase.reference_energy[(energy_index, 0)];
        let momentum = ((energy - reference) * (2.0 * FEFF_HARTREE_EV)).sqrt();
        validate_phase_text_complex("phmin energy", energy)?;
        validate_phase_text_complex("phmin reference energy", reference)?;
        validate_phase_text_complex("phmin momentum", momentum)?;

        write_phase_field(&mut out, energy.re * FEFF_HARTREE_EV, 13, 5)?;
        write_phase_field(&mut out, reference.re * FEFF_HARTREE_EV, 13, 5)?;
        write_phase_field(&mut out, momentum.re, 13, 5)?;
        for angular_momentum in 0..=2 {
            let shift = phase_shift_or_zero(potential, energy_index, angular_momentum);
            validate_phase_text_complex("phmin phase shift", shift)?;
            write_phase_field(&mut out, shift.re, 13, 5)?;
        }
        out.push('\n');
    }
    Ok(out)
}

fn phase_shift_or_zero(
    potential: &refeff_io::PhaseBinPotential,
    energy_index: usize,
    angular_momentum: usize,
) -> Complex64 {
    if angular_momentum > potential.lmax {
        return Complex64::new(0.0, 0.0);
    }
    let signed_l_slot = potential.lmax + angular_momentum;
    potential.phase_shifts[(energy_index, signed_l_slot, 0)]
}

fn validate_phase_text_complex(name: &'static str, value: Complex64) -> Result<()> {
    ensure!(
        value.re.is_finite() && value.im.is_finite(),
        "XSPH {name} has non-finite value {value:?}"
    );
    Ok(())
}

fn write_phase_field(out: &mut String, value: f64, width: usize, precision: usize) -> Result<()> {
    ensure!(
        value.is_finite(),
        "XSPH phase text output has non-finite value {value}"
    );
    write_fortran_exp(out, value, width, precision)?;
    Ok(())
}

fn write_xsect_cache(path: &Path, data: &XsectDatData) -> Result<()> {
    write_xsect_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_axafs_cache(path: &Path, data: &AxafsDatData) -> Result<()> {
    write_axafs_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl_cache(path: &Path, data: &XseclDatData) -> Result<()> {
    write_xsecl_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl2_cache(path: &Path, data: &XseclDatData) -> Result<()> {
    write_xsecl2_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_xsecl_bin_cache(path: &Path, data: &XseclBinData) -> Result<()> {
    write_xsecl_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_mpse_cache(path: &Path, data: &MpseDatData) -> Result<()> {
    write_mpse_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_emesh_cache(path: &Path, data: &EmeshDatData) -> Result<()> {
    write_emesh_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_emesh_bin_cache(path: &Path, data: &EmeshBinData) -> Result<()> {
    write_emesh_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)?;
    Ok(1)
}

fn write_or_generate_module_log(path: &Path, phase: &PhaseBinData) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_generated_xsph_module_log(path, phase)
}

fn write_or_recover_module_log(
    path: &Path,
    phase: &PhaseBinData,
    source_handoff_written: bool,
) -> Result<usize> {
    if source_handoff_written && path.is_file() && read_module_log_dat(path).is_err() {
        return write_generated_xsph_module_log(path, phase);
    }
    write_or_generate_module_log(path, phase)
}

fn write_generated_xsph_module_log(path: &Path, phase: &PhaseBinData) -> Result<usize> {
    let mut lines = vec![
        "Calculating cross-section and phases ...".to_string(),
        "    absorption cross section".to_string(),
    ];
    lines.extend(
        (0..phase.potential_count())
            .map(|potential| format!("    phase shifts for unique potential{potential:5}")),
    );
    lines.push("Done with module: cross-section and phases (XSPH).".to_string());

    let line_terminators = vec!["\n".to_string(); lines.len()];
    write_module_log(
        path,
        &ModuleLogData {
            lines,
            line_terminators,
        },
    )?;
    Ok(1)
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XsphCachePaths {
    work_dir: PathBuf,
    global_inp: PathBuf,
    eels_inp: PathBuf,
    grid_inp: PathBuf,
    config_dat: PathBuf,
    hubbard_inp: PathBuf,
    v_hubbard_bin: PathBuf,
    loss_dat: PathBuf,
    exc_dat: PathBuf,
    specfunct_dat: PathBuf,
    pot_bin: PathBuf,
    pot_ch: PathBuf,
    yoshi_dat: PathBuf,
    wscrn_dat: PathBuf,
    phase_bin: PathBuf,
    xsect_dat: PathBuf,
    axafs_dat: PathBuf,
    xsecl_dat: PathBuf,
    xsecl2_dat: PathBuf,
    xsecl_bin: PathBuf,
    aphase_hubbard_bin: PathBuf,
    mpse_dat: PathBuf,
    emesh_dat: PathBuf,
    emesh_bin: PathBuf,
    rl_dat: PathBuf,
    log2_dat: PathBuf,
    listedges_pmbse: PathBuf,
    xsedge_dat: PathBuf,
}

impl XsphCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            work_dir: work_dir.to_path_buf(),
            global_inp: work_dir.join("global.inp"),
            eels_inp: work_dir.join("eels.inp"),
            grid_inp: work_dir.join("grid.inp"),
            config_dat: work_dir.join("config.dat"),
            hubbard_inp: work_dir.join("hubbard.inp"),
            v_hubbard_bin: work_dir.join("v_hubbard.bin"),
            loss_dat: work_dir.join("loss.dat"),
            exc_dat: work_dir.join("exc.dat"),
            specfunct_dat: work_dir.join("specfunct.dat"),
            pot_bin: work_dir.join("pot.bin"),
            pot_ch: work_dir.join("pot.ch"),
            yoshi_dat: work_dir.join("yoshi.dat"),
            wscrn_dat: work_dir.join("wscrn.dat"),
            phase_bin: work_dir.join("phase.bin"),
            xsect_dat: work_dir.join("xsect.dat"),
            axafs_dat: work_dir.join("axafs.dat"),
            xsecl_dat: work_dir.join("xsecl.dat"),
            xsecl2_dat: work_dir.join("xsecl2.dat"),
            xsecl_bin: work_dir.join("xsecl.bin"),
            aphase_hubbard_bin: work_dir.join("aphase_hubbard.bin"),
            mpse_dat: work_dir.join("mpse.dat"),
            emesh_dat: work_dir.join("emesh.dat"),
            emesh_bin: work_dir.join("emesh.bin"),
            rl_dat: work_dir.join("rl.dat"),
            log2_dat: work_dir.join("log2.dat"),
            listedges_pmbse: work_dir.join("listedges.pmbse"),
            xsedge_dat: work_dir.join("xsedge.dat"),
        }
    }

    fn has_phase_cache(&self) -> bool {
        self.phase_bin.is_file()
    }

    fn has_complete_base_outputs(&self) -> bool {
        self.phase_bin.is_file() && self.xsect_dat.is_file()
    }

    fn phase_dat(&self, potential_index: usize) -> PathBuf {
        self.work_dir.join(format!("phase{potential_index:02}.dat"))
    }

    fn phmin_dat(&self, potential_index: usize) -> PathBuf {
        self.work_dir.join(format!("phmin{potential_index:02}.dat"))
    }
}

#[cfg(test)]
mod tests;
