use super::*;

pub(super) fn validate_nonempty_path(path_indices: &[usize]) -> Result<usize, PathError> {
    if path_indices.is_empty() {
        Err(PathError::EmptyPathCriteria)
    } else {
        Ok(path_indices.len())
    }
}

pub(super) fn validate_position_shape(
    atom_positions: ArrayView2<'_, Real>,
) -> Result<(), PathError> {
    if atom_positions.nrows() == 0 || atom_positions.ncols() != 3 {
        Err(PathError::InvalidAtomPositionShape {
            rows: atom_positions.nrows(),
            columns: atom_positions.ncols(),
        })
    } else {
        Ok(())
    }
}

pub(super) fn validate_atom_index(
    atom_positions: ArrayView2<'_, Real>,
    position: usize,
    atom_index: usize,
) -> Result<(), PathError> {
    if atom_index >= atom_positions.nrows() {
        Err(PathError::AtomIndexOutOfRange {
            position,
            atom_index,
            atoms: atom_positions.nrows(),
        })
    } else {
        Ok(())
    }
}

pub(super) fn atom_position(
    atom_positions: ArrayView2<'_, Real>,
    position: usize,
    atom_index: usize,
) -> Result<[f32; 3], PathError> {
    validate_atom_index(atom_positions, position, atom_index)?;
    let mut point = [0.0_f32; 3];
    for component in 0..3 {
        let value = atom_positions[(atom_index, component)];
        if !value.is_finite() {
            return Err(PathError::NonFiniteAtomPosition {
                atom_index,
                component,
                value,
            });
        }
        point[component] = value as f32;
    }
    Ok(point)
}
