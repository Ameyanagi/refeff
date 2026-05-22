use super::*;

/// Overlap one FEFF potential's free-atom Coulomb potential and densities.
///
/// This ports `POT/ovrlp.f90` for a single `iph`. The routine starts from the
/// current free-atom columns, applies either explicit `OVERLAP`-style neighbor
/// specifications or all geometry neighbors within 12 Bohr, computes the Norman
/// radius from the overlapped electron density, and converts the current
/// spin-density column into FEFF's `dmag / edens` ratio. FEFF initializes
/// `edenvl` from `rhoval(:,iph)` but adds source `rho` during overlaps; this
/// port preserves that behavior.
pub fn overlap_potential_density(
    input: PotentialOverlapInput<'_>,
) -> Result<PotentialOverlap, DensityError> {
    validate_potential_overlap_input(input)?;

    let potential = input.potential_index;
    let mut coulomb_potential = table_column_prefix(input.coulomb_potential, potential);
    let mut electron_density = table_column_prefix(input.electron_density, potential);
    let mut valence_density = table_column_prefix(input.valence_density, potential);

    let neighbors = overlap_neighbors(input)?;
    for neighbor in neighbors {
        let source_coulomb =
            table_column_prefix(input.coulomb_potential, neighbor.source_potential);
        let source_density = table_column_prefix(input.electron_density, neighbor.source_potential);
        coulomb_potential = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_coulomb.view(),
            accumulated: coulomb_potential.view(),
        })?
        .accumulated;
        electron_density = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_density.view(),
            accumulated: electron_density.view(),
        })?
        .accumulated;
        valence_density = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: neighbor.distance,
            multiplicity: neighbor.multiplicity,
            source: source_density.view(),
            accumulated: valence_density.view(),
        })?
        .accumulated;
    }

    let norman_atomic_number = input.atomic_numbers[potential].max(1);
    let norman_radius = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: electron_density.view(),
        atomic_number: norman_atomic_number,
    })?;
    let spin_density_ratio = table_column_prefix(input.spin_density, potential)
        .iter()
        .zip(electron_density.iter())
        .map(|(&spin, &density)| if density > 0.0 { spin / density } else { 0.0 })
        .collect::<Array1<_>>();

    Ok(PotentialOverlap {
        coulomb_potential,
        electron_density,
        valence_density,
        spin_density_ratio,
        norman_radius,
    })
}
fn validate_potential_overlap_input(input: PotentialOverlapInput<'_>) -> Result<(), DensityError> {
    ensure_len(
        "atomic_numbers",
        input.atomic_numbers.len(),
        input.potential_index + 1,
    )?;
    ensure_len(
        "representative_atoms",
        input.representative_atoms.len(),
        input.potential_index + 1,
    )?;
    validate_position_table(input.atom_positions)?;
    if input.atom_potentials.len() != input.atom_positions.nrows() {
        return Err(DensityError::AtomPotentialLengthMismatch {
            potentials: input.atom_potentials.len(),
            positions: input.atom_positions.nrows(),
        });
    }
    validate_usize_potential_values(
        "atom_potentials",
        input.atom_potentials,
        input.atomic_numbers.len(),
    )?;
    validate_usize_potential_values(
        "representative_atoms",
        input.representative_atoms,
        input.atom_positions.nrows(),
    )?;

    let potential_count = input.atomic_numbers.len();
    ensure_shape(
        "electron_density",
        input.electron_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "spin_density",
        input.spin_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "valence_density",
        input.valence_density.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    ensure_shape(
        "coulomb_potential",
        input.coulomb_potential.shape(),
        OVRLP_DENSITY_POINTS,
        potential_count,
    )?;
    validate_real_table_values("electron_density", input.electron_density)?;
    validate_real_table_values("spin_density", input.spin_density)?;
    validate_real_table_values("valence_density", input.valence_density)?;
    validate_real_table_values("coulomb_potential", input.coulomb_potential)?;

    for neighbor in input.explicit_overlaps {
        if neighbor.source_potential >= potential_count {
            return Err(DensityError::InvalidPotentialIndex {
                name: "explicit_overlaps.source_potential",
                index: neighbor.source_potential,
                available: potential_count,
            });
        }
        validate_positive_real_scalar("explicit_overlaps.distance", neighbor.distance)?;
        validate_real_scalar("explicit_overlaps.multiplicity", neighbor.multiplicity)?;
    }

    Ok(())
}
fn overlap_neighbors(
    input: PotentialOverlapInput<'_>,
) -> Result<Vec<PotentialOverlapNeighbor>, DensityError> {
    if !input.explicit_overlaps.is_empty() {
        return Ok(input.explicit_overlaps.to_vec());
    }

    let representative = input.representative_atoms[input.potential_index];
    let center = [
        input.atom_positions[(representative, 0)],
        input.atom_positions[(representative, 1)],
        input.atom_positions[(representative, 2)],
    ];
    let mut neighbors = Vec::new();
    for atom in 0..input.atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let position = [
            input.atom_positions[(atom, 0)],
            input.atom_positions[(atom, 1)],
            input.atom_positions[(atom, 2)],
        ];
        let distance = distance_between(position, center);
        if distance <= OVRLP_GEOMETRY_CUTOFF {
            neighbors.push(PotentialOverlapNeighbor {
                source_potential: input.atom_potentials[atom],
                multiplicity: 1.0,
                distance,
            });
        }
    }
    Ok(neighbors)
}
