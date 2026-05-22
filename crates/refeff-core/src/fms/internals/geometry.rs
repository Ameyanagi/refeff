use super::*;

pub(in crate::fms) fn fms_atom_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    fms_atom_distance_squared(left, right).sqrt()
}

pub(in crate::fms) fn fms_atom_distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

pub(in crate::fms) fn fms_free_propagator_prefactor(
    rho: Complex32,
    wave_number: Complex32,
    mean_square_displacement: f32,
) -> Complex32 {
    const BOHR: f32 = 0.529_177_25;
    let phase = (Complex32::new(0.0, 1.0) * rho).exp() / rho;
    let damping_factor = Complex32::new(-mean_square_displacement / (BOHR * BOHR), 0.0);
    let damping = (damping_factor * wave_number * wave_number).exp();
    phase * damping
}

pub(in crate::fms) fn sort_radius_key(index: usize, atom: FmsAtom) -> Result<f64, FmsError> {
    ensure_finite_position(index, atom.position)?;
    Ok(f64::from(atom.position[0]) * f64::from(atom.position[0])
        + f64::from(atom.position[1]) * f64::from(atom.position[1])
        + f64::from(atom.position[2]) * f64::from(atom.position[2])
        + (index as f64 + 1.0) * 1.0e-6)
}

pub(in crate::fms) fn checked_position(
    positions: &[[f32; 3]],
    index: usize,
) -> Result<[f32; 3], FmsError> {
    let position = positions
        .get(index)
        .copied()
        .ok_or(FmsError::AtomIndexOutOfRange {
            index,
            len: positions.len(),
        })?;
    ensure_finite_position(index, position)?;
    Ok(position)
}

pub(in crate::fms) fn ensure_finite_position(
    atom: usize,
    position: [f32; 3],
) -> Result<(), FmsError> {
    for (axis, value) in position.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(FmsError::NonFiniteCoordinate { atom, axis });
        }
    }
    Ok(())
}

pub(in crate::fms) fn checked_atom_index(atom: usize) -> Result<usize, FmsError> {
    atom.checked_sub(1)
        .ok_or(FmsError::InvalidStateAtom { atom })
}

pub(in crate::fms) fn ensure_atom_table_index(index: usize, len: usize) -> Result<(), FmsError> {
    if index < len {
        Ok(())
    } else {
        Err(FmsError::AtomIndexOutOfRange { index, len })
    }
}
