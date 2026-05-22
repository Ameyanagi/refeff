use super::*;

/// Correct FEFF Coulomb potentials after a valence-density update.
///
/// This ports `POT/coulom.f90`. FEFF builds a radial Coulomb correction from
/// `rhoval - edenvl`, fixes its additive constant either from cluster charge
/// deltas (`icoul == 1`) or from Norman-sphere charge normalization, then
/// zero-fills the inactive radial tail for each potential.
pub fn update_coulomb_potential(
    input: CoulombPotentialUpdateInput<'_>,
) -> Result<CoulombPotentialUpdate, DensityError> {
    validate_coulomb_update_input(input)?;

    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    let radii = coulom_radii();
    let mut updated = input.coulomb_potential.to_owned();
    for potential in 0..potential_count {
        let active_len = input.last_indices[potential];
        let mut density_delta = Array1::<Real>::zeros(OVRLP_DENSITY_POINTS);
        for radial in 0..active_len {
            density_delta[radial] = (input.valence_density[(radial, potential)]
                - input.overlapped_valence_density[(radial, potential)])
                * radii[radial].powi(2);
        }
        let correction = coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density_delta.view(),
            radii: radii.view(),
            delta: COULOM_DELTA,
            active_len,
        })?;
        let constant_shift = match input.mode {
            CoulombUpdateMode::LongRange => coulom_long_range_shift(
                input,
                potential,
                &radii,
                &density_delta,
                &correction.potential,
            )?,
            CoulombUpdateMode::Norman => {
                coulom_norman_shift(input, potential, &radii, &correction.potential)?
            }
        };

        for radial in 0..active_len {
            updated[(radial, potential)] += correction.potential[radial] + constant_shift;
        }
        for radial in active_len..OVRLP_DENSITY_POINTS {
            updated[(radial, potential)] = 0.0;
        }
    }

    Ok(CoulombPotentialUpdate {
        coulomb_potential: updated,
    })
}
fn validate_coulomb_update_input(
    input: CoulombPotentialUpdateInput<'_>,
) -> Result<(), DensityError> {
    let potential_count = potential_count_from_highest(input.highest_potential_index)?;
    ensure_len("last_indices", input.last_indices.len(), potential_count)?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        potential_count,
    )?;
    ensure_len("norman_radii", input.norman_radii.len(), potential_count)?;
    ensure_len("charge_deltas", input.charge_deltas.len(), potential_count)?;
    ensure_len(
        "atomic_numbers",
        input.atomic_numbers.len(),
        potential_count,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(DensityError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values("atom_potentials", input.atom_potentials, potential_count)?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;
    validate_usize_positive_values("atomic_numbers", input.atomic_numbers)?;
    validate_usize_radial_indices("last_indices", input.last_indices)?;
    validate_real_values("norman_radii", input.norman_radii)?;
    validate_real_values("charge_deltas", input.charge_deltas)?;
    for &radius in input.norman_radii.iter().take(potential_count) {
        validate_positive_real_scalar("norman_radii", radius)?;
    }

    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "overlapped_valence_density",
        input.overlapped_valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "overlapped_density",
        input.overlapped_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("valence_density", input.valence_density)?;
    validate_real_table_values(
        "overlapped_valence_density",
        input.overlapped_valence_density,
    )?;
    validate_real_table_values("overlapped_density", input.overlapped_density)?;
    validate_real_table_values("coulomb_potential", input.coulomb_potential)
}

fn coulom_long_range_shift(
    input: CoulombPotentialUpdateInput<'_>,
    potential: usize,
    radii: &Array1<Real>,
    density_delta: &Array1<Real>,
    correction: &Array1<Real>,
) -> Result<Real, DensityError> {
    let norman_radius = input.norman_radii[potential];
    let norman_index = coulom_index_for_radius(norman_radius, 2.0)?;
    ensure_coulom_grid_index("norman_radius", norman_index, 2)?;

    let representative = input.representative_atoms[potential];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut boundary_value = input.charge_deltas[potential] / norman_radius;
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        let distance = distance_between(position, center).max(norman_radius);
        boundary_value += input.charge_deltas[input.atom_potentials[atom]] / distance;
    }

    let row = norman_index - 1;
    let dr = radii[row] - norman_radius;
    let slope = (density_delta[row] - density_delta[row - 1]) / (radii[row] - radii[row - 1]);
    boundary_value -= dr / 2.0
        * (input.charge_deltas[potential] / norman_radius.powi(2)
            + (input.charge_deltas[potential] + density_delta[row] * dr
                - slope / 2.0 * dr.powi(2))
                / radii[row].powi(2));
    validate_real_scalar("long_range_shift", boundary_value - correction[row])?;
    Ok(boundary_value - correction[row])
}

fn coulom_norman_shift(
    input: CoulombPotentialUpdateInput<'_>,
    potential: usize,
    radii: &Array1<Real>,
    correction: &Array1<Real>,
) -> Result<Real, DensityError> {
    let density = table_column_prefix(input.overlapped_density, potential);
    let mut combined = Array1::<Real>::zeros(OVRLP_DENSITY_POINTS);
    for radial in 0..OVRLP_DENSITY_POINTS {
        combined[radial] = input.overlapped_density[(radial, potential)]
            - input.overlapped_valence_density[(radial, potential)]
            + input.valence_density[(radial, potential)];
    }
    let radius_original = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: input.atomic_numbers[potential],
    })?
    .radius;
    let radius_updated = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: combined.view(),
        atomic_number: input.atomic_numbers[potential],
    })?
    .radius;

    let rmin = radius_original.min(radius_updated);
    let inrm = coulom_index_for_radius(rmin, 1.0)?;
    ensure_coulom_grid_index("norman_min", inrm, 1)?;
    ensure_coulom_grid_index("norman_min_next", inrm + 1, 1)?;
    let row = inrm - 1;
    let r0 = radii[row];
    let mut delta = 0.0;
    if radius_updated > radius_original {
        let slope = (combined[row + 1] - combined[row]) / (radii[row + 1] - radii[row]);
        let intercept = combined[row] - slope * radii[row];
        delta -= coulom_fab(slope, intercept, r0, radius_original, radius_updated);
    } else {
        let slope = (input.overlapped_density[(row, potential)]
            - input.overlapped_density[(row + 1, potential)])
            / (radii[row + 1] - radii[row]);
        let intercept = -input.overlapped_density[(row, potential)] - slope * radii[row];
        delta -= coulom_fab(slope, intercept, r0, radius_updated, radius_original);
    }
    let slope = (combined[row + 1] - combined[row] + input.overlapped_density[(row, potential)]
        - input.overlapped_density[(row + 1, potential)])
        / (radii[row + 1] - radii[row]);
    let intercept = combined[row] - input.overlapped_density[(row, potential)] - slope * radii[row];
    delta -= coulom_fab(slope, intercept, r0, r0, rmin);
    validate_real_scalar("norman_shift", delta - correction[row])?;
    Ok(delta - correction[row])
}

fn coulom_fab(slope: Real, intercept: Real, r0: Real, r1: Real, r2: Real) -> Real {
    let a2 = (r2.powi(2) - r1.powi(2)) / 2.0;
    let a3 = (r2.powi(3) - r1.powi(3)) / 3.0;
    let a4 = (r2.powi(4) - r1.powi(4)) / 4.0;
    slope * (a4 / r0 - a3) + intercept * (a3 / r0 - a2)
}

fn coulom_index_for_radius(radius: Real, offset: Real) -> Result<usize, DensityError> {
    validate_positive_real_scalar("radius", radius)?;
    Ok(((radius.ln() + COULOM_LITERAL_OFFSET) / COULOM_LITERAL_DELTA + offset).trunc() as usize)
}
fn ensure_coulom_grid_index(
    name: &'static str,
    index: usize,
    minimum: usize,
) -> Result<(), DensityError> {
    if index < minimum || index > OVRLP_DENSITY_POINTS {
        return Err(DensityError::InvalidIndex { name, index });
    }
    Ok(())
}

fn coulom_radii() -> Array1<Real> {
    (1..=OVRLP_DENSITY_POINTS)
        .map(|index| (-COULOM_LITERAL_OFFSET + COULOM_DELTA * (index - 1) as Real).exp())
        .collect::<Array1<_>>()
}
