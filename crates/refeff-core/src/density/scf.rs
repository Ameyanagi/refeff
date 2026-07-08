use super::*;

/// Apply one FEFF POT SCF density update after valence-density integration.
///
/// The remaining POT driver computes `integrated_valence_density` through the
/// contour/scattering path. This helper owns the production handoff from that
/// integrated density into FEFF's `broydn` mixer and `coulom` Coulomb-potential
/// correction.
pub fn update_scf_density_potential(
    input: ScfDensityStepInput<'_>,
) -> Result<ScfDensityStep, DensityError> {
    let mixed = mix_broyden_density(BroydenMixInput {
        iteration: input.iteration,
        accelerator: input.accelerator,
        highest_potential_index: input.highest_potential_index,
        valence_occupancy: input.valence_occupancy,
        last_indices: input.last_indices,
        potential_multiplicities: input.potential_multiplicities,
        norman_radii: input.norman_radii,
        norman_charges: input.norman_charges,
        overlapped_valence_density: input.overlapped_valence_density,
        valence_density: input.integrated_valence_density,
        workspace: input.workspace,
    })?;

    let updated = update_coulomb_potential(CoulombPotentialUpdateInput {
        mode: input.coulomb_mode,
        highest_potential_index: input.highest_potential_index,
        last_indices: input.last_indices,
        valence_density: mixed.valence_density.view(),
        overlapped_valence_density: input.overlapped_valence_density,
        overlapped_density: input.overlapped_density,
        atom_positions: input.atom_positions,
        representative_atoms: input.representative_atoms,
        atom_potentials: input.atom_potentials,
        norman_radii: input.norman_radii,
        charge_deltas: mixed.charge_deltas.view(),
        atomic_numbers: input.atomic_numbers,
        coulomb_potential: input.coulomb_potential,
    })?;

    Ok(ScfDensityStep {
        valence_density: mixed.valence_density,
        charge_deltas: mixed.charge_deltas,
        norman_charges: mixed.norman_charges,
        coulomb_potential: updated.coulomb_potential,
        workspace: mixed.workspace,
    })
}

/// Run one source-backed FEFF POT SCF iteration after radial/FMS source rows exist.
///
/// This composes the Rust-backed SCMT contour integration with FEFF's
/// occupation-count repeat check, `broydn` density mixing, `coulom` potential
/// correction, and final `edens`/`edenvl` table update.
pub fn run_pot_scf_iteration(
    input: PotScfIterationInput<'_>,
) -> Result<PotScfIteration, DensityError> {
    let contour_input = input.contour;
    let mut contour = run_pot_scf_contour(contour_input)?;
    if contour.status == PotScfContourRunStatus::NeedsMoreSourcePoints {
        return Ok(PotScfIteration {
            status: PotScfIterationStatus::NeedsMoreSourcePoints,
            contour,
            density_step: None,
            bad_occupation_count: 0,
            overlapped_density: input.overlapped_density.to_owned(),
            overlapped_valence_density: input.overlapped_valence_density.to_owned(),
        });
    }
    contour.occupancy_by_l = pot_scf_padded_occupancy_by_l(
        contour.occupancy_by_l.view(),
        input.expected_valence_occupancy,
        contour_input.highest_potential_index,
    )?;

    let bad_occupation_count = pot_scf_bad_occupation_count(
        contour.occupancy_by_l.view(),
        input.expected_valence_occupancy,
        contour_input.highest_potential_index,
    )?;
    if bad_occupation_count > 0 && input.repeat_on_bad_counts {
        return Ok(PotScfIteration {
            status: PotScfIterationStatus::RepeatRequired,
            contour,
            density_step: None,
            bad_occupation_count,
            overlapped_density: input.overlapped_density.to_owned(),
            overlapped_valence_density: input.overlapped_valence_density.to_owned(),
        });
    }

    let density_step = update_scf_density_potential(ScfDensityStepInput {
        iteration: input.iteration,
        accelerator: input.accelerator,
        coulomb_mode: input.coulomb_mode,
        highest_potential_index: contour_input.highest_potential_index,
        valence_occupancy: input.expected_valence_occupancy,
        last_indices: contour_input.last_indices,
        potential_multiplicities: contour_input.potential_multiplicities,
        norman_radii: input.norman_radii,
        norman_charges: input.norman_charges,
        overlapped_valence_density: input.overlapped_valence_density,
        integrated_valence_density: contour.valence_density.view(),
        workspace: input.workspace,
        overlapped_density: input.overlapped_density,
        atom_positions: input.atom_positions,
        representative_atoms: input.representative_atoms,
        atom_potentials: input.atom_potentials,
        atomic_numbers: input.atomic_numbers,
        coulomb_potential: input.coulomb_potential,
    })?;
    let (overlapped_density, overlapped_valence_density) = pot_scf_update_overlapped_density(
        input.overlapped_density,
        input.overlapped_valence_density,
        density_step.valence_density.view(),
        contour_input.last_indices,
        contour_input.highest_potential_index,
    )?;

    Ok(PotScfIteration {
        status: PotScfIterationStatus::Updated,
        contour,
        density_step: Some(density_step),
        bad_occupation_count,
        overlapped_density,
        overlapped_valence_density,
    })
}

/// Map a non-`Updated` [`PotScfIterationStatus`] to its outer-loop status,
/// returning `None` for `Updated` (handled separately by the caller).
fn non_updated_outer_status(status: PotScfIterationStatus) -> Option<PotScfOuterIterationStatus> {
    match status {
        PotScfIterationStatus::NeedsMoreSourcePoints => {
            Some(PotScfOuterIterationStatus::NeedsMoreSourcePoints)
        }
        PotScfIterationStatus::RepeatRequired => Some(PotScfOuterIterationStatus::RepeatRequired),
        PotScfIterationStatus::Updated => None,
    }
}

/// Apply FEFF `POT/potsub.f90` outer-loop convergence and density transition.
///
/// `scmt` leaves `edens` in the working form used by Coulomb correction. The
/// outer POT loop then checks Fermi/charge convergence, either restores the
/// pre-`scmt` density/potential for final output or copies mixed `rhoval` into
/// `edenvl` before the next `istprm` pass.
pub fn finish_pot_scf_outer_iteration(
    input: PotScfOuterIterationInput<'_>,
) -> Result<PotScfOuterIteration, DensityError> {
    validate_pot_scf_outer_iteration_input(input)?;

    let iteration_result = input.iteration_result;
    if let Some(status) = non_updated_outer_status(iteration_result.status) {
        return Ok(PotScfOuterIteration {
            status,
            fermi_energy: input.previous_fermi_energy,
            charge_distance: 0.0,
            partial_charge_distance: 0.0,
            norman_charge_reference: input.previous_norman_charges.to_owned(),
            reported_charge_transfer: Array1::zeros(input.previous_norman_charges.len()),
            overlapped_density: iteration_result.overlapped_density.clone(),
            overlapped_valence_density: iteration_result.overlapped_valence_density.clone(),
            coulomb_potential: input.previous_coulomb_potential.to_owned(),
        });
    }

    let Some(density_step) = iteration_result.density_step.as_ref() else {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_outer_iteration_density_step",
            index: 0,
        });
    };
    let fermi_energy = iteration_result
        .contour
        .fermi_energy
        .ok_or(DensityError::InvalidIndex {
            name: "pot_scf_outer_iteration_fermi_energy",
            index: 0,
        })?;
    validate_real_scalar("pot_scf_outer_iteration_fermi_energy", fermi_energy)?;

    let potential_count =
        potential_count_from_highest(iteration_result.contour.embedded_ldos.ncols() - 1)?;
    let mut passes = true;
    if input.iteration < input.max_iterations && input.iteration <= input.minimum_iterations {
        passes = false;
    }
    if (fermi_energy - input.previous_fermi_energy).abs() > input.fermi_tolerance {
        passes = false;
    }

    let mut charge_distance: Real = 0.0;
    let mut partial_charge_distance: Real = 0.0;
    let mut norman_charge_reference = input.previous_norman_charges.to_owned();
    let mut reported_charge_transfer = Array1::<Real>::zeros(potential_count);
    for potential in 0..potential_count {
        let norman_charge = density_step.norman_charges[potential];
        let delta = (norman_charge - input.previous_norman_charges[potential]).abs();
        if delta > input.charge_tolerance {
            passes = false;
        }
        charge_distance = charge_distance.max(delta);
        norman_charge_reference[potential] = norman_charge;
        reported_charge_transfer[potential] = -norman_charge + input.ion_charges[potential];

        let mut charge_sum = -norman_charge;
        for angular in 0..iteration_result.contour.occupancy_by_l.nrows() {
            let partial_delta = (iteration_result.contour.occupancy_by_l[(angular, potential)]
                - input.previous_occupancy_by_l[(angular, potential)])
                .abs();
            if partial_delta > input.partial_charge_tolerance {
                passes = false;
            }
            partial_charge_distance = partial_charge_distance.max(partial_delta);
            charge_sum += iteration_result.contour.occupancy_by_l[(angular, potential)]
                - input.expected_valence_occupancy[(angular, potential)];
        }
        if charge_sum.abs() > input.charge_sum_tolerance {
            passes = false;
        }
    }
    validate_real_scalar("pot_scf_outer_iteration_charge_distance", charge_distance)?;
    validate_real_scalar(
        "pot_scf_outer_iteration_partial_charge_distance",
        partial_charge_distance,
    )?;
    validate_real_values(
        "pot_scf_outer_iteration_reported_charge_transfer",
        reported_charge_transfer.view(),
    )?;

    let reached_limit = input.iteration == input.max_iterations;
    let finalize = passes || reached_limit;
    let status = if passes {
        PotScfOuterIterationStatus::Converged
    } else if reached_limit {
        PotScfOuterIterationStatus::ReachedIterationLimit
    } else {
        PotScfOuterIterationStatus::NeedsNextIteration
    };

    let (overlapped_density, overlapped_valence_density, coulomb_potential) = if finalize {
        pot_scf_restore_outer_density_state(
            iteration_result.overlapped_density.view(),
            iteration_result.overlapped_valence_density.view(),
            density_step.valence_density.view(),
            input.previous_coulomb_potential,
            potential_count,
        )?
    } else {
        pot_scf_prepare_next_outer_density_state(
            iteration_result.overlapped_density.view(),
            density_step.valence_density.view(),
            density_step.coulomb_potential.view(),
            potential_count,
        )?
    };

    Ok(PotScfOuterIteration {
        status,
        fermi_energy,
        charge_distance,
        partial_charge_distance,
        norman_charge_reference,
        reported_charge_transfer,
        overlapped_density,
        overlapped_valence_density,
        coulomb_potential,
    })
}

/// Advance one supplied source-backed POT SCF iteration from an explicit state.
///
/// This helper owns the state plumbing around [`run_pot_scf_iteration`] and
/// [`finish_pot_scf_outer_iteration`]. When the returned outer status is
/// `NeedsNextIteration`, the caller still owns the FEFF `istprm` pass and
/// generation of the next iteration's radial/FMS source rows.
pub fn advance_pot_scf_state(
    input: PotScfStateAdvanceInput<'_>,
) -> Result<PotScfStateAdvance, DensityError> {
    let iteration = run_pot_scf_iteration(PotScfIterationInput {
        contour: input.contour,
        iteration: input.iteration,
        accelerator: input.accelerator,
        coulomb_mode: input.coulomb_mode,
        repeat_on_bad_counts: input.repeat_on_bad_counts,
        expected_valence_occupancy: input.expected_valence_occupancy,
        norman_radii: input.norman_radii,
        norman_charges: input.state.norman_charges.view(),
        overlapped_valence_density: input.state.overlapped_valence_density.view(),
        workspace: &input.state.workspace,
        overlapped_density: input.state.overlapped_density.view(),
        atom_positions: input.atom_positions,
        representative_atoms: input.representative_atoms,
        atom_potentials: input.atom_potentials,
        atomic_numbers: input.atomic_numbers,
        coulomb_potential: input.state.coulomb_potential.view(),
    })?;

    let outer = finish_pot_scf_outer_iteration(PotScfOuterIterationInput {
        iteration_result: &iteration,
        iteration: input.iteration,
        max_iterations: input.max_iterations,
        minimum_iterations: input.minimum_iterations,
        previous_fermi_energy: input.state.fermi_energy,
        previous_norman_charges: input.state.norman_charge_reference.view(),
        previous_occupancy_by_l: input.state.occupancy_by_l.view(),
        expected_valence_occupancy: input.expected_valence_occupancy,
        ion_charges: input.ion_charges,
        previous_coulomb_potential: input.state.coulomb_potential.view(),
        fermi_tolerance: input.fermi_tolerance,
        charge_tolerance: input.charge_tolerance,
        charge_sum_tolerance: input.charge_sum_tolerance,
        partial_charge_tolerance: input.partial_charge_tolerance,
    })?;

    let (norman_charges, occupancy_by_l, workspace) =
        if let Some(density_step) = iteration.density_step.as_ref() {
            (
                density_step.norman_charges.clone(),
                iteration.contour.occupancy_by_l.clone(),
                density_step.workspace.clone(),
            )
        } else {
            (
                input.state.norman_charges.clone(),
                input.state.occupancy_by_l.clone(),
                input.state.workspace.clone(),
            )
        };

    let state = PotScfState {
        fermi_energy: outer.fermi_energy,
        norman_charges,
        norman_charge_reference: outer.norman_charge_reference.clone(),
        occupancy_by_l,
        overlapped_density: outer.overlapped_density.clone(),
        overlapped_valence_density: outer.overlapped_valence_density.clone(),
        coulomb_potential: outer.coulomb_potential.clone(),
        workspace,
    };

    Ok(PotScfStateAdvance {
        iteration,
        outer,
        state,
    })
}

fn pot_scf_padded_occupancy_by_l(
    actual: ArrayView2<'_, Real>,
    expected: ArrayView2<'_, Real>,
    highest_potential_index: usize,
) -> Result<Array2<Real>, DensityError> {
    let potential_count = potential_count_from_highest(highest_potential_index)?;
    ensure_shape(
        "pot_scf_iteration_actual_occupancy",
        actual.shape(),
        1,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_iteration_expected_occupancy",
        expected.shape(),
        actual.nrows(),
        potential_count,
    )?;
    validate_real_table_values("pot_scf_iteration_actual_occupancy", actual)?;
    validate_real_table_values("pot_scf_iteration_expected_occupancy", expected)?;

    let mut padded = Array2::<Real>::zeros((expected.nrows(), potential_count));
    for potential in 0..potential_count {
        for angular in 0..actual.nrows() {
            padded[(angular, potential)] = actual[(angular, potential)];
        }
    }
    Ok(padded)
}

fn validate_pot_scf_outer_iteration_input(
    input: PotScfOuterIterationInput<'_>,
) -> Result<(), DensityError> {
    if input.iteration == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_outer_iteration",
            index: input.iteration,
        });
    }
    if input.max_iterations == 0 {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_outer_max_iterations",
            index: input.max_iterations,
        });
    }
    if input.iteration > input.max_iterations {
        return Err(DensityError::InvalidIndex {
            name: "pot_scf_outer_iteration_over_max",
            index: input.iteration,
        });
    }
    validate_real_scalar(
        "pot_scf_outer_previous_fermi_energy",
        input.previous_fermi_energy,
    )?;
    validate_positive_real_scalar("pot_scf_outer_fermi_tolerance", input.fermi_tolerance)?;
    validate_positive_real_scalar("pot_scf_outer_charge_tolerance", input.charge_tolerance)?;
    validate_positive_real_scalar(
        "pot_scf_outer_charge_sum_tolerance",
        input.charge_sum_tolerance,
    )?;
    validate_positive_real_scalar(
        "pot_scf_outer_partial_charge_tolerance",
        input.partial_charge_tolerance,
    )?;

    let potential_count = input.iteration_result.contour.embedded_ldos.ncols();
    ensure_len("pot_scf_outer_potential_count", potential_count, 1)?;
    ensure_len(
        "pot_scf_outer_previous_norman_charges",
        input.previous_norman_charges.len(),
        potential_count,
    )?;
    ensure_len(
        "pot_scf_outer_ion_charges",
        input.ion_charges.len(),
        potential_count,
    )?;
    validate_real_values(
        "pot_scf_outer_previous_norman_charges",
        input.previous_norman_charges,
    )?;
    validate_real_values("pot_scf_outer_ion_charges", input.ion_charges)?;

    let angular_count = input.iteration_result.contour.occupancy_by_l.nrows();
    ensure_shape(
        "pot_scf_outer_previous_occupancy_by_l",
        input.previous_occupancy_by_l.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_outer_expected_valence_occupancy",
        input.expected_valence_occupancy.shape(),
        angular_count,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_outer_previous_coulomb_potential",
        input.previous_coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values(
        "pot_scf_outer_previous_occupancy_by_l",
        input.previous_occupancy_by_l,
    )?;
    validate_real_table_values(
        "pot_scf_outer_expected_valence_occupancy",
        input.expected_valence_occupancy,
    )?;
    validate_real_table_values(
        "pot_scf_outer_previous_coulomb_potential",
        input.previous_coulomb_potential,
    )
}

type PotScfOuterDensityState = (Array2<Real>, Array2<Real>, Array2<Real>);

fn pot_scf_restore_outer_density_state(
    overlapped_density: ArrayView2<'_, Real>,
    overlapped_valence_density: ArrayView2<'_, Real>,
    valence_density: ArrayView2<'_, Real>,
    previous_coulomb_potential: ArrayView2<'_, Real>,
    potential_count: usize,
) -> Result<PotScfOuterDensityState, DensityError> {
    validate_outer_density_tables(
        "pot_scf_outer_restore",
        overlapped_density,
        overlapped_valence_density,
        valence_density,
        previous_coulomb_potential,
        potential_count,
    )?;

    let mut restored_density = overlapped_density.to_owned();
    for potential in 0..potential_count {
        for radial in 0..OVRLP_DENSITY_POINTS {
            restored_density[(radial, potential)] = overlapped_density[(radial, potential)]
                - valence_density[(radial, potential)]
                + overlapped_valence_density[(radial, potential)];
        }
    }
    validate_real_table_values("pot_scf_outer_restored_density", restored_density.view())?;
    Ok((
        restored_density,
        overlapped_valence_density.to_owned(),
        previous_coulomb_potential.to_owned(),
    ))
}

fn pot_scf_prepare_next_outer_density_state(
    overlapped_density: ArrayView2<'_, Real>,
    valence_density: ArrayView2<'_, Real>,
    coulomb_potential: ArrayView2<'_, Real>,
    potential_count: usize,
) -> Result<PotScfOuterDensityState, DensityError> {
    ensure_shape(
        "pot_scf_outer_next_overlapped_density",
        overlapped_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_outer_next_valence_density",
        valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_outer_next_coulomb_potential",
        coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("pot_scf_outer_next_overlapped_density", overlapped_density)?;
    validate_real_table_values("pot_scf_outer_next_valence_density", valence_density)?;
    validate_real_table_values("pot_scf_outer_next_coulomb_potential", coulomb_potential)?;
    Ok((
        overlapped_density.to_owned(),
        valence_density.to_owned(),
        coulomb_potential.to_owned(),
    ))
}

fn validate_outer_density_tables(
    prefix: &'static str,
    overlapped_density: ArrayView2<'_, Real>,
    overlapped_valence_density: ArrayView2<'_, Real>,
    valence_density: ArrayView2<'_, Real>,
    coulomb_potential: ArrayView2<'_, Real>,
    potential_count: usize,
) -> Result<(), DensityError> {
    ensure_shape(
        prefix,
        overlapped_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        prefix,
        overlapped_valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        prefix,
        valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        prefix,
        coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values(prefix, overlapped_density)?;
    validate_real_table_values(prefix, overlapped_valence_density)?;
    validate_real_table_values(prefix, valence_density)?;
    validate_real_table_values(prefix, coulomb_potential)
}

fn pot_scf_bad_occupation_count(
    actual: ArrayView2<'_, Real>,
    expected: ArrayView2<'_, Real>,
    highest_potential_index: usize,
) -> Result<usize, DensityError> {
    let potential_count = potential_count_from_highest(highest_potential_index)?;
    ensure_shape(
        "pot_scf_iteration_actual_occupancy",
        actual.shape(),
        1,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_iteration_expected_occupancy",
        expected.shape(),
        actual.nrows(),
        potential_count,
    )?;
    validate_real_table_values("pot_scf_iteration_actual_occupancy", actual)?;
    validate_real_table_values("pot_scf_iteration_expected_occupancy", expected)?;

    let mut bad_count = 0;
    for potential in 0..potential_count {
        for angular in 0..actual.nrows() {
            let threshold = match angular {
                0 => 1.95,
                1 => 5.1,
                2 => 9.1,
                _ => 13.1,
            };
            if (actual[(angular, potential)] - expected[(angular, potential)]).abs() > threshold {
                bad_count += 1;
            }
        }
    }
    Ok(bad_count)
}

fn pot_scf_update_overlapped_density(
    overlapped_density: ArrayView2<'_, Real>,
    overlapped_valence_density: ArrayView2<'_, Real>,
    valence_density: ArrayView2<'_, Real>,
    last_indices: ArrayView1<'_, usize>,
    highest_potential_index: usize,
) -> Result<(Array2<Real>, Array2<Real>), DensityError> {
    let potential_count = potential_count_from_highest(highest_potential_index)?;
    ensure_len(
        "pot_scf_iteration_last_indices",
        last_indices.len(),
        potential_count,
    )?;
    validate_usize_radial_indices("pot_scf_iteration_last_indices", last_indices)?;
    ensure_shape(
        "pot_scf_iteration_overlapped_density",
        overlapped_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_iteration_overlapped_valence_density",
        overlapped_valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "pot_scf_iteration_valence_density",
        valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("pot_scf_iteration_overlapped_density", overlapped_density)?;
    validate_real_table_values(
        "pot_scf_iteration_overlapped_valence_density",
        overlapped_valence_density,
    )?;
    validate_real_table_values("pot_scf_iteration_valence_density", valence_density)?;

    let mut updated_density = overlapped_density.to_owned();
    let mut updated_valence = overlapped_valence_density.to_owned();
    for potential in 0..potential_count {
        let active_len = last_indices[potential];
        for radial in 0..active_len {
            updated_density[(radial, potential)] = overlapped_density[(radial, potential)]
                - overlapped_valence_density[(radial, potential)]
                + valence_density[(radial, potential)];
        }
        for radial in active_len..OVRLP_DENSITY_POINTS {
            updated_density[(radial, potential)] = 0.0;
            updated_valence[(radial, potential)] = 0.0;
        }
    }

    Ok((updated_density, updated_valence))
}
