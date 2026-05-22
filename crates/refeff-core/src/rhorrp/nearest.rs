use ndarray::{Array2, ArrayView2};

use crate::{Real, Vector3};

use super::types::{
    RhorrpError, RhorrpFmsInclusionInput, RhorrpNearestAtom, RhorrpNearestAtomInput,
    RhorrpNearestAtomTable, RhorrpNearestAtomTableInput,
};
use super::validation::{
    validate_atom_search_input, validate_fms_inclusion_input, validate_nearest_atom_table_input,
    validate_vector,
};

/// Port of FEFF `init_inclus` FMS-radius atom counting.
///
/// For each potential representative atom, FEFF counts all atoms with
/// `|r_atom - r_representative|^2 <= rfms2^2`. Coordinates and `fms_radius`
/// are already in Bohr, matching RHORRP after input-file unit conversion.
pub fn rhorrp_fms_inclusion_counts(
    input: RhorrpFmsInclusionInput<'_>,
) -> Result<Vec<usize>, RhorrpError> {
    validate_fms_inclusion_input(input)?;
    let radius_squared = input.fms_radius * input.fms_radius;

    Ok(input
        .representative_atoms
        .iter()
        .map(|&representative| {
            let center = [
                input.atom_positions[(representative, 0)],
                input.atom_positions[(representative, 1)],
                input.atom_positions[(representative, 2)],
            ];
            (0..input.atom_positions.nrows())
                .filter(|&atom| {
                    let dx = input.atom_positions[(atom, 0)] - center[0];
                    let dy = input.atom_positions[(atom, 1)] - center[1];
                    let dz = input.atom_positions[(atom, 2)] - center[2];
                    dx * dx + dy * dy + dz * dz <= radius_squared
                })
                .count()
        })
        .collect())
}

/// Port of FEFF `nearest_atom`.
///
/// When `fms_atom_count` is set, only the leading `fms_atom_count` atoms are
/// considered, matching the `fmsF` branch in FEFF. Ties keep the first atom,
/// because the FEFF routine updates only on strictly smaller distance.
pub fn rhorrp_nearest_atom(
    input: RhorrpNearestAtomInput<'_>,
) -> Result<RhorrpNearestAtom, RhorrpError> {
    validate_vector("point", input.point)?;
    let atoms_to_search = validate_atom_search_input(
        input.atom_positions,
        input.atom_potentials,
        input.fms_atom_count,
    )?;
    nearest_atom_unchecked(
        input.point,
        input.atom_positions,
        input.atom_potentials,
        atoms_to_search,
    )
}

/// Evaluate nearest-atom diagnostics for every point in a FEFF-order table.
///
/// The point table uses the same `(xyz, point)` layout as
/// [`rhorrp_density_grid_points`](crate::rhorrp::rhorrp_density_grid_points).
/// The returned displacement table is
/// `(point, xyz)`, matching RHORRP text diagnostic rows.
pub fn rhorrp_nearest_atom_table(
    input: RhorrpNearestAtomTableInput<'_>,
) -> Result<RhorrpNearestAtomTable, RhorrpError> {
    let atoms_to_search = validate_nearest_atom_table_input(input)?;
    let point_count = input.points.ncols();
    let mut displacement_bohr = Array2::zeros((point_count, 3));
    let mut atom_indices = Vec::with_capacity(point_count);
    let mut atom_indices_1based = Vec::with_capacity(point_count);
    let mut potential_indices = Vec::with_capacity(point_count);

    for point_index in 0..point_count {
        let point = [
            input.points[(0, point_index)],
            input.points[(1, point_index)],
            input.points[(2, point_index)],
        ];
        let nearest = nearest_atom_unchecked(
            point,
            input.atom_positions,
            input.atom_potentials,
            atoms_to_search,
        )?;
        for axis in 0..3 {
            displacement_bohr[(point_index, axis)] = nearest.displacement[axis];
        }
        atom_indices.push(nearest.atom_index);
        atom_indices_1based.push(nearest.atom_index_1based);
        potential_indices.push(nearest.potential_index);
    }

    Ok(RhorrpNearestAtomTable {
        displacement_bohr,
        atom_indices,
        atom_indices_1based,
        potential_indices,
    })
}

fn nearest_atom_unchecked(
    point: Vector3,
    atom_positions: ArrayView2<'_, Real>,
    atom_potentials: &[usize],
    atoms_to_search: usize,
) -> Result<RhorrpNearestAtom, RhorrpError> {
    let mut best: Option<RhorrpNearestAtom> = None;
    for atom_index in 0..atoms_to_search {
        let displacement = [
            point[0] - atom_positions[(atom_index, 0)],
            point[1] - atom_positions[(atom_index, 1)],
            point[2] - atom_positions[(atom_index, 2)],
        ];
        let squared_distance = displacement.iter().map(|value| value * value).sum();
        let candidate = RhorrpNearestAtom {
            atom_index,
            atom_index_1based: atom_index + 1,
            potential_index: atom_potentials[atom_index],
            displacement,
            squared_distance,
        };
        if best
            .as_ref()
            .is_none_or(|current| candidate.squared_distance < current.squared_distance)
        {
            best = Some(candidate);
        }
    }

    best.ok_or(RhorrpError::NoAtoms)
}
