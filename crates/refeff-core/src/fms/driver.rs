use super::*;

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

/// Assemble and solve one real-space FEFF FMS energy point.
///
/// This wires the top-level `fmspack` sequence for real-space FMS after
/// `xprep` has prepared geometry tables: setup state kets, build spin-resolved
/// `xrho`/`xclm`, assemble `g0`, build the compact T-matrix, normalize `minv`,
/// and dispatch the selected scattering solver.
pub fn fms_real_space_energy(
    input: FmsRealSpaceEnergyInput<'_>,
) -> Result<FmsRealSpaceEnergyResult, FmsError> {
    ensure_spin_channels(input.spin_channels)?;
    if input.wave_numbers.len() != input.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "ck",
            expected: input.spin_channels,
            actual: input.wave_numbers.len(),
        });
    }
    if input.phase_shifts.shape()[0] != input.spin_channels {
        return Err(FmsError::SpinChannelCountMismatch {
            table: "xphase",
            expected: input.spin_channels,
            actual: input.phase_shifts.shape()[0],
        });
    }

    let setup = fms_driver_setup(FmsDriverSetupInput {
        lfms: input.lfms,
        spin_channels: input.spin_channels,
        atoms: input.atoms,
        max_potential: input.max_potential,
        global_lmax: input.global_lmax,
        raw_potential_lmax: input.raw_potential_lmax,
        state_capacity: input.state_capacity,
    })?;
    let pair_tables = fms_spin_pair_tables(input.global_lmax, input.wave_numbers, input.atoms)?;
    let free_propagator = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states: &setup.state_kets.states,
        atoms: input.atoms,
        direct_cutoff: input.direct_cutoff,
        rho: pair_tables.rho.view(),
        wave_numbers: input.wave_numbers,
        mean_square_displacements: input.mean_square_displacements,
        xclm: pair_tables.polynomials.view(),
        xnlm: input.xnlm,
        rotations: input.rotations,
    })?;
    let t_matrix = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &setup.state_kets.states,
        atoms: input.atoms,
        spin_channels: input.spin_channels,
        spin_selector: input.spin_selector,
        phase_shifts: input.phase_shifts,
        spin_orbit: input.spin_orbit,
    })?;
    let method_selection =
        fms_scattering_method_selection(input.minv, input.full_scattering_matrix_requested);
    let scattering = fms_scattering(FmsScatteringInput {
        method: method_selection.method,
        calculate_full_scattering: input.full_scattering_matrix_requested,
        states: &setup.state_kets.states,
        spin_channels: input.spin_channels,
        global_lmax: input.global_lmax,
        potential_lmax: &setup.potential_lmax,
        representative_offsets: &setup.state_kets.representative_offsets,
        potential_start: setup.potential_start,
        potential_end: setup.potential_end,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: input.calculated_l,
        convergence_tolerance: input.convergence_tolerance,
        zero_tolerance: input.zero_tolerance,
    })?;

    Ok(FmsRealSpaceEnergyResult {
        setup,
        method_selection,
        pair_tables,
        free_propagator,
        t_matrix,
        scattering,
    })
}
