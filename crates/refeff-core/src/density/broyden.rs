use super::*;

/// Mix FEFF valence densities with the Broyden SCF accelerator.
///
/// This ports `POT/broydn.f90`. The Fortran routine mutates module-level
/// Broyden history; this Rust API takes an explicit [`BroydenWorkspace`] and
/// returns the updated workspace together with the mixed density and Norman
/// charge deltas.
pub fn mix_broyden_density(input: BroydenMixInput<'_>) -> Result<BroydenMix, DensityError> {
    validate_broyden_mix_input(input)?;

    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    let iteration = input.iteration - 1;
    let mut workspace = input.workspace.clone();
    if input.iteration == 1 {
        for index in 0..OVRLP_DENSITY_POINTS {
            let radius = broydn_radius(index);
            workspace.radii[index] = radius;
            workspace.weights[index] = radius.powi(3);
        }
    }

    for potential in 0..potential_count {
        for radial in 0..input.last_indices[potential] {
            workspace.residuals[(radial, potential, iteration)] = input.valence_density
                [(radial, potential)]
                * workspace.radii[radial]
                - input.overlapped_valence_density[(radial, potential)] * workspace.weights[radial];
        }
    }

    let valence_counts = broydn_valence_counts(input, potential_count)?;
    let total_fermi_count = broydn_total_fermi_count(&valence_counts, input, potential_count)?;

    if input.iteration > 1 {
        let previous_iteration = iteration - 1;
        workspace.norms[iteration] = broydn_residual_norm(input, &workspace, iteration)?;
        validate_nonzero_real_scalar("broyden_norm", workspace.norms[iteration])?;

        for history_1based in 2..=input.iteration {
            let history = history_1based - 1;
            let numerator = broydn_history_projection(input, &workspace, iteration, history)?;
            validate_nonzero_real_scalar("broyden_history_norm", workspace.norms[history])?;
            workspace.coefficients[(iteration, history)] = numerator / workspace.norms[history];
        }

        for potential in 0..potential_count {
            for radial in 0..input.last_indices[potential] {
                workspace.multipliers[(radial, potential, iteration)] = input.accelerator
                    * (workspace.residuals[(radial, potential, iteration)]
                        - workspace.residuals[(radial, potential, previous_iteration)])
                    + (input.overlapped_valence_density[(radial, potential)]
                        - workspace.previous_density[(radial, potential)])
                        * workspace.weights[radial];
            }
        }

        for history_1based in 2..input.iteration {
            let history = history_1based - 1;
            let correction = workspace.coefficients[(iteration, history)]
                - workspace.coefficients[(previous_iteration, history)];
            for potential in 0..potential_count {
                for radial in 0..input.last_indices[potential] {
                    workspace.multipliers[(radial, potential, iteration)] -=
                        workspace.multipliers[(radial, potential, history)] * correction;
                }
            }
        }
    }

    let mut valence_density = input.valence_density.to_owned();
    for potential in 0..potential_count {
        for radial in 0..input.last_indices[potential] {
            workspace.previous_density[(radial, potential)] =
                input.overlapped_valence_density[(radial, potential)];
            valence_density[(radial, potential)] = input.overlapped_valence_density
                [(radial, potential)]
                + input.accelerator * workspace.residuals[(radial, potential, iteration)]
                    / workspace.weights[radial];
            for history_1based in 2..=input.iteration {
                let history = history_1based - 1;
                valence_density[(radial, potential)] -= workspace.coefficients
                    [(iteration, history)]
                    * workspace.multipliers[(radial, potential, history)]
                    / workspace.weights[radial];
            }
        }
    }

    let mut charge_deltas = Array1::<Real>::zeros(input.norman_charges.len());
    let mut norman_charges = input.norman_charges.to_owned();
    let mut average_delta = 0.0;
    let mut atom_count = 0.0;
    for potential in 0..potential_count {
        let integrated_charge =
            broydn_integrated_charge(input, &workspace, &valence_density, potential)?;
        charge_deltas[potential] =
            integrated_charge - input.norman_charges[potential] - valence_counts[potential];
        average_delta += input.potential_multiplicities[potential] * charge_deltas[potential];
        atom_count += input.potential_multiplicities[potential];
    }
    validate_nonzero_real_scalar("broyden_atom_count", atom_count)?;
    let density_scale = average_delta / total_fermi_count;
    average_delta /= atom_count;

    for potential in 0..potential_count {
        charge_deltas[potential] -= average_delta;
        norman_charges[potential] += charge_deltas[potential];
        for radial in 0..input.last_indices[potential] {
            valence_density[(radial, potential)] -=
                density_scale * input.overlapped_valence_density[(radial, potential)];
        }
    }

    Ok(BroydenMix {
        valence_density,
        charge_deltas,
        norman_charges,
        workspace,
    })
}
fn validate_broyden_mix_input(input: BroydenMixInput<'_>) -> Result<(), DensityError> {
    if input.iteration == 0 {
        return Err(DensityError::InvalidIndex {
            name: "iteration",
            index: input.iteration,
        });
    }
    validate_real_scalar("accelerator", input.accelerator)?;
    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    ensure_len("last_indices", input.last_indices.len(), potential_count)?;
    ensure_len(
        "potential_multiplicities",
        input.potential_multiplicities.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len(
        "norman_charges",
        input.norman_charges.len(),
        potential_count,
    )?;
    validate_usize_radial_indices("last_indices", input.last_indices)?;
    validate_real_values("potential_multiplicities", input.potential_multiplicities)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_real_values("norman_charges", input.norman_charges)?;
    for &multiplicity in input.potential_multiplicities.iter().take(potential_count) {
        validate_positive_real_scalar("potential_multiplicities", multiplicity)?;
    }
    for &radius in input.norman_radii.iter().take(potential_count) {
        validate_positive_real_scalar("norman_radii", radius)?;
    }

    ensure_shape(
        "valence_occupancy",
        input.valence_occupancy.shape(),
        1,
        potential_count,
    )?;
    ensure_shape(
        "overlapped_valence_density",
        input.overlapped_valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("valence_occupancy", input.valence_occupancy)?;
    validate_real_table_values(
        "overlapped_valence_density",
        input.overlapped_valence_density,
    )?;
    validate_real_table_values("valence_density", input.valence_density)?;

    ensure_shape(
        "workspace.coefficients",
        input.workspace.coefficients.shape(),
        input.iteration,
        input.iteration,
    )?;
    ensure_shape3(
        "workspace.residuals",
        input.workspace.residuals.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
        input.iteration,
    )?;
    ensure_shape3(
        "workspace.multipliers",
        input.workspace.multipliers.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
        input.iteration,
    )?;
    ensure_len(
        "workspace.norms",
        input.workspace.norms.len(),
        input.iteration,
    )?;
    ensure_len(
        "workspace.weights",
        input.workspace.weights.len(),
        OVRLP_DENSITY_POINTS,
    )?;
    ensure_len(
        "workspace.radii",
        input.workspace.radii.len(),
        OVRLP_DENSITY_POINTS,
    )?;
    ensure_shape(
        "workspace.previous_density",
        input.workspace.previous_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values(
        "workspace.coefficients",
        input.workspace.coefficients.view(),
    )?;
    validate_real_cube_values("workspace.residuals", &input.workspace.residuals)?;
    validate_real_cube_values("workspace.multipliers", &input.workspace.multipliers)?;
    validate_real_values("workspace.norms", input.workspace.norms.view())?;
    validate_real_table_values(
        "workspace.previous_density",
        input.workspace.previous_density.view(),
    )?;
    if input.iteration > 1 {
        validate_positive_real_values("workspace.weights", input.workspace.weights.view())?;
        validate_positive_real_values("workspace.radii", input.workspace.radii.view())?;
    }
    Ok(())
}

fn broydn_valence_counts(
    input: BroydenMixInput<'_>,
    potential_count: usize,
) -> Result<Array1<Real>, DensityError> {
    let mut counts = Array1::<Real>::zeros(potential_count);
    for potential in 0..potential_count {
        let count = input
            .valence_occupancy
            .column(potential)
            .iter()
            .copied()
            .sum::<Real>();
        validate_real_scalar("valence_count", count)?;
        counts[potential] = count;
    }
    Ok(counts)
}

fn broydn_total_fermi_count(
    valence_counts: &Array1<Real>,
    input: BroydenMixInput<'_>,
    potential_count: usize,
) -> Result<Real, DensityError> {
    let total = (0..potential_count)
        .map(|potential| valence_counts[potential] * input.potential_multiplicities[potential])
        .sum::<Real>();
    validate_nonzero_real_scalar("broyden_total_fermi_count", total)?;
    Ok(total)
}

fn broydn_residual_norm(
    input: BroydenMixInput<'_>,
    workspace: &BroydenWorkspace,
    iteration: usize,
) -> Result<Real, DensityError> {
    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    let previous = iteration - 1;
    let mut norm = 0.0;
    for potential in 0..potential_count {
        for radial in 0..input.last_indices[potential] {
            let delta = workspace.residuals[(radial, potential, iteration)]
                - workspace.residuals[(radial, potential, previous)];
            norm += delta * delta;
        }
    }
    validate_real_scalar("broyden_norm", norm)?;
    Ok(norm)
}

fn broydn_history_projection(
    input: BroydenMixInput<'_>,
    workspace: &BroydenWorkspace,
    iteration: usize,
    history: usize,
) -> Result<Real, DensityError> {
    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    let mut projection = 0.0;
    for potential in 0..potential_count {
        for radial in 0..input.last_indices[potential] {
            projection += workspace.residuals[(radial, potential, iteration)]
                * (workspace.residuals[(radial, potential, history)]
                    - workspace.residuals[(radial, potential, history - 1)]);
        }
    }
    validate_real_scalar("broyden_projection", projection)?;
    Ok(projection)
}

fn broydn_integrated_charge(
    input: BroydenMixInput<'_>,
    workspace: &BroydenWorkspace,
    valence_density: &Array2<Real>,
    potential: usize,
) -> Result<Real, DensityError> {
    let norman_radius = input.norman_radii[potential];
    let norman_index = ((norman_radius.ln() + 8.8) / BROYDN_DELTA + 2.0).trunc() as usize;
    let active_len = norman_index
        .checked_add(1)
        .ok_or(DensityError::InvalidIndex {
            name: "norman_index",
            index: norman_index,
        })?;
    if active_len > OVRLP_DENSITY_POINTS {
        return Err(DensityError::LengthTooShort {
            name: "broyden_radii",
            required: active_len,
            actual: OVRLP_DENSITY_POINTS,
        });
    }
    let radii = workspace
        .radii
        .iter()
        .take(active_len)
        .copied()
        .collect::<Vec<_>>();
    let density_moments = (0..active_len)
        .map(|radial| valence_density[(radial, potential)] * workspace.radii[radial].powi(2))
        .collect::<Vec<_>>();
    somm2(
        &radii,
        &density_moments,
        BROYDN_DELTA,
        2.0,
        norman_radius,
        0,
    )
    .map_err(DensityError::from)
}

fn broydn_radius(radial: usize) -> Real {
    (-BROYDN_LITERAL_OFFSET + BROYDN_DELTA * radial as Real).exp()
}
