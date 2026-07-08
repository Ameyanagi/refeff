use super::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn fms_driver_setup(input: FmsDriverSetupInput<'_>) -> Result<FmsDriverSetup, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.atoms.is_empty() {
        return Err(FmsError::EmptyCluster);
    }
    if input.max_potential >= input.raw_potential_lmax.len() {
        return Err(FmsError::TableIndexOutOfRange {
            table: "lipotx",
            axis: "potential",
            index: input.max_potential,
        });
    }

    let potential_count = input
        .max_potential
        .checked_add(1)
        .ok_or(FmsError::IntegerOverflow {
            field: "max_potential",
            value: input.max_potential,
        })?;
    let potential_lmax = input
        .raw_potential_lmax
        .iter()
        .take(potential_count)
        .map(|&lmax| clamp_fms_lipotx(lmax, input.global_lmax))
        .collect::<Vec<_>>();

    let atom_potentials = input
        .atoms
        .iter()
        .map(|atom| checked_potential(atom.potential, input.max_potential))
        .collect::<Result<Vec<_>, _>>()?;
    let absorber_potential = atom_potentials
        .first()
        .copied()
        .ok_or(FmsError::EmptyCluster)?;
    let (potential_start, potential_end) = if input.lfms == 0 {
        (absorber_potential, absorber_potential)
    } else {
        (0, input.max_potential)
    };

    let state_kets = construct_state_kets_with_limit(
        input.spin_channels,
        &atom_potentials,
        &potential_lmax,
        input.global_lmax,
        input.state_capacity,
    )
    .map_err(fms_state_ket_error)?;

    for potential in potential_start..=potential_end {
        representative_offset(&state_kets.representative_offsets, potential)?;
    }

    Ok(FmsDriverSetup {
        potential_lmax,
        potential_start,
        potential_end,
        state_kets,
    })
}

/// Select the FEFF FMS scattering branch for a raw `minv` value.
///
/// FEFF dispatches `minv=0` to LU, `1` to BiCGStab/VdV, `2` to recursion,
/// `3` to Graves-Morris/Salam, and every other value to TFQMR. When a full
/// scattering matrix is requested, FEFF forces all non-LU choices back to LU.
pub fn fms_scattering_method_selection(
    minv: i32,
    full_scattering_matrix_requested: bool,
) -> FmsScatteringMethodSelection {
    let forced_lu_for_full_scattering = full_scattering_matrix_requested && minv != 0;
    let effective_minv = if forced_lu_for_full_scattering {
        0
    } else {
        minv
    };
    let method = match effective_minv {
        0 => FmsScatteringMethod::Lu,
        1 => FmsScatteringMethod::BiCgStab,
        2 => FmsScatteringMethod::Recursion,
        3 => FmsScatteringMethod::GravesMorris,
        _ => FmsScatteringMethod::Tfqmr,
    };

    FmsScatteringMethodSelection {
        effective_minv,
        method,
        forced_lu_for_full_scattering,
    }
}

/// Build a reusable, `Sync` [`FmsRealSpacePlan`] from every FEFF FMS
/// real-space input that stays fixed across an energy sweep.
///
/// This runs [`fms_driver_setup`] exactly once (state kets, clamped `lipotx`,
/// and the active potential range never depend on energy), so callers that
/// sweep many energy points build the plan a single time and reuse it via
/// [`fms_real_space_energy_with_plan`] or [`fms_real_space_spectrum`] instead
/// of paying `fms_driver_setup`'s cost on every point.
pub fn fms_real_space_plan(
    input: FmsRealSpacePlanInput<'_>,
) -> Result<FmsRealSpacePlan<'_>, FmsError> {
    let setup = fms_driver_setup(FmsDriverSetupInput {
        lfms: input.lfms,
        spin_channels: input.spin_channels,
        atoms: input.atoms,
        max_potential: input.max_potential,
        global_lmax: input.global_lmax,
        raw_potential_lmax: input.raw_potential_lmax,
        state_capacity: input.state_capacity,
    })?;

    Ok(FmsRealSpacePlan {
        setup,
        minv: input.minv,
        spin_channels: input.spin_channels,
        spin_selector: input.spin_selector,
        atoms: input.atoms,
        global_lmax: input.global_lmax,
        spin_orbit: input.spin_orbit,
        direct_cutoff: input.direct_cutoff,
        mean_square_displacements: input.mean_square_displacements,
        xnlm: input.xnlm,
        rotations: input.rotations,
        calculated_l: input.calculated_l,
        convergence_tolerance: input.convergence_tolerance,
        zero_tolerance: input.zero_tolerance,
        full_scattering_matrix_requested: input.full_scattering_matrix_requested,
        retain_setup: input.retain_setup,
        retain_pair_tables: input.retain_pair_tables,
        retain_free_propagator: input.retain_free_propagator,
        retain_t_matrix: input.retain_t_matrix,
        retain_system_matrix: input.retain_system_matrix,
    })
}

/// Assemble and solve one real-space FEFF FMS energy point against a
/// pre-built [`FmsRealSpacePlan`].
///
/// This wires the top-level `fmspack` sequence for real-space FMS after
/// `xprep` has prepared geometry tables and after [`fms_real_space_plan`] has
/// prepared energy-independent state: build spin-resolved `xrho`/`xclm`,
/// assemble `g0`, build the compact T-matrix, normalize `minv`, and dispatch
/// the selected scattering solver. The function is pure in `plan` and
/// `point`, so it is safe to call from any thread, e.g. via
/// [`fms_real_space_spectrum`].
pub fn fms_real_space_energy_with_plan(
    plan: &FmsRealSpacePlan<'_>,
    point: FmsRealSpaceEnergyPoint<'_>,
) -> Result<FmsRealSpaceEnergyResult, FmsError> {
    ensure_spin_channels(plan.spin_channels)?;
    if point.wave_numbers.len() != plan.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "ck",
            expected: plan.spin_channels,
            actual: point.wave_numbers.len(),
        });
    }
    if point.phase_shifts.shape()[0] != plan.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "xphase",
            expected: plan.spin_channels,
            actual: point.phase_shifts.shape()[0],
        });
    }

    let pair_tables = fms_spin_pair_tables(plan.global_lmax, point.wave_numbers, plan.atoms)?;
    let free_propagator = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states: &plan.setup.state_kets.states,
        atoms: plan.atoms,
        direct_cutoff: plan.direct_cutoff,
        rho: pair_tables.rho.view(),
        wave_numbers: point.wave_numbers,
        mean_square_displacements: plan.mean_square_displacements,
        xclm: pair_tables.polynomials.view(),
        xnlm: plan.xnlm,
        rotations: plan.rotations,
    })?;
    let t_matrix = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &plan.setup.state_kets.states,
        atoms: plan.atoms,
        spin_channels: plan.spin_channels,
        spin_selector: plan.spin_selector,
        phase_shifts: point.phase_shifts,
        spin_orbit: plan.spin_orbit,
    })?;
    let method_selection =
        fms_scattering_method_selection(plan.minv, plan.full_scattering_matrix_requested);
    let mut scattering = fms_scattering(FmsScatteringInput {
        method: method_selection.method,
        calculate_full_scattering: plan.full_scattering_matrix_requested,
        states: &plan.setup.state_kets.states,
        spin_channels: plan.spin_channels,
        global_lmax: plan.global_lmax,
        potential_lmax: &plan.setup.potential_lmax,
        representative_offsets: &plan.setup.state_kets.representative_offsets,
        potential_start: plan.setup.potential_start,
        potential_end: plan.setup.potential_end,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: plan.calculated_l,
        convergence_tolerance: plan.convergence_tolerance,
        zero_tolerance: plan.zero_tolerance,
    })?;
    if !plan.retain_system_matrix {
        scattering.system_matrix = None;
    }

    Ok(FmsRealSpaceEnergyResult {
        setup: plan.retain_setup.then(|| plan.setup.clone()),
        method_selection,
        pair_tables: plan.retain_pair_tables.then_some(pair_tables),
        free_propagator: plan.retain_free_propagator.then_some(free_propagator),
        t_matrix: plan.retain_t_matrix.then_some(t_matrix),
        scattering,
    })
}

/// Solve every energy point in `energies` against a shared [`FmsRealSpacePlan`]
/// in parallel, preserving each point's index in the returned `Vec`.
///
/// [`fms_real_space_energy_with_plan`] is a pure function of `plan` and one
/// [`FmsRealSpaceEnergyPoint`], so results do not depend on evaluation order;
/// `rayon`'s indexed `collect` keeps this function's output in the exact
/// order of `energies`, matching a sequential loop bit-for-bit.
pub fn fms_real_space_spectrum(
    plan: &FmsRealSpacePlan<'_>,
    energies: &[FmsRealSpaceEnergyPoint<'_>],
) -> Vec<Result<FmsRealSpaceEnergyResult, FmsError>> {
    energies
        .into_par_iter()
        .map(|&point| fms_real_space_energy_with_plan(plan, point))
        .collect()
}

/// Assemble and solve one real-space FEFF FMS energy point.
///
/// This wires the top-level `fmspack` sequence for real-space FMS after
/// `xprep` has prepared geometry tables: setup state kets, build spin-resolved
/// `xrho`/`xclm`, assemble `g0`, build the compact T-matrix, normalize `minv`,
/// and dispatch the selected scattering solver.
///
/// This is a thin wrapper around [`fms_real_space_plan`] and
/// [`fms_real_space_energy_with_plan`] for callers that only need a single
/// energy point; it always retains every intermediate (matching this
/// function's historical behavior). Callers sweeping many energy points
/// should build a [`FmsRealSpacePlan`] once and call
/// [`fms_real_space_spectrum`] instead.
pub fn fms_real_space_energy(
    input: FmsRealSpaceEnergyInput<'_>,
) -> Result<FmsRealSpaceEnergyResult, FmsError> {
    let plan = fms_real_space_plan(FmsRealSpacePlanInput {
        lfms: input.lfms,
        minv: input.minv,
        spin_channels: input.spin_channels,
        spin_selector: input.spin_selector,
        atoms: input.atoms,
        max_potential: input.max_potential,
        global_lmax: input.global_lmax,
        raw_potential_lmax: input.raw_potential_lmax,
        state_capacity: input.state_capacity,
        spin_orbit: input.spin_orbit,
        direct_cutoff: input.direct_cutoff,
        mean_square_displacements: input.mean_square_displacements,
        xnlm: input.xnlm,
        rotations: input.rotations,
        calculated_l: input.calculated_l,
        convergence_tolerance: input.convergence_tolerance,
        zero_tolerance: input.zero_tolerance,
        full_scattering_matrix_requested: input.full_scattering_matrix_requested,
        retain_setup: true,
        retain_pair_tables: true,
        retain_free_propagator: true,
        retain_t_matrix: true,
        retain_system_matrix: true,
    })?;
    fms_real_space_energy_with_plan(
        &plan,
        FmsRealSpaceEnergyPoint {
            wave_numbers: input.wave_numbers,
            phase_shifts: input.phase_shifts,
        },
    )
}
