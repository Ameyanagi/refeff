use super::*;

/// Port FEFF DMDW `Calc_R_CM`: mass-weighted center of mass.
///
/// `atom_positions` is an `(atom, xyz)` table in any consistent distance unit,
/// and `atom_masses` is FEFF `dym_In%am`.
pub fn dmdw_center_of_mass(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<[Real; 3], DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let total_mass = atom_masses.iter().copied().sum::<Real>();
    let mut center = [0.0; 3];
    for (component, value) in center.iter_mut().enumerate() {
        *value = atom_positions
            .column(component)
            .iter()
            .zip(atom_masses.iter())
            .map(|(&coordinate, &mass)| coordinate * mass)
            .sum::<Real>()
            / total_mass;
    }
    Ok(center)
}

/// Port FEFF DMDW `Calc_ToI`: tensor of inertia about the supplied origin.
///
/// FEFF calls this after shifting coordinates to the center of mass. This
/// function preserves that explicit calling convention: pass centered
/// coordinates when a center-of-mass tensor is required.
pub fn dmdw_inertia_tensor(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<Array2<Real>, DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let mut tensor = Array2::<Real>::zeros((3, 3));
    for (atom, row) in atom_positions.rows().into_iter().enumerate() {
        let mass = atom_masses[atom];
        let x = row[0];
        let y = row[1];
        let z = row[2];
        tensor[(0, 0)] += mass * (y * y + z * z);
        tensor[(1, 1)] += mass * (x * x + z * z);
        tensor[(2, 2)] += mass * (x * x + y * y);
        tensor[(1, 0)] -= mass * y * x;
        tensor[(2, 0)] -= mass * z * x;
        tensor[(2, 1)] -= mass * z * y;
    }
    tensor[(0, 1)] = tensor[(1, 0)];
    tensor[(0, 2)] = tensor[(2, 0)];
    tensor[(1, 2)] = tensor[(2, 1)];
    Ok(tensor)
}

/// Port FEFF DMDW `Make_TrfD` rigid translation/rotation basis.
///
/// The returned `projection_modes` matrix contains the first six normalized
/// `TrfD` columns used by FEFF to project translations and rotations out of a
/// DMDW Lanczos seed. Rows use FEFF's component-major coordinate order: all x
/// atom components, then all y, then all z.
pub fn dmdw_rigid_body_projection_modes(
    atom_positions: ArrayView2<'_, Real>,
    atom_masses: ArrayView1<'_, Real>,
) -> Result<DmdwRigidBodyModes, DebyeError> {
    validate_dmdw_atoms(atom_positions, atom_masses)?;
    let atom_count = atom_masses.len();
    if atom_count < 2 {
        return Err(DebyeError::TooFewDmdwRigidBodyAtoms { atoms: atom_count });
    }

    let center_of_mass = dmdw_center_of_mass(atom_positions, atom_masses)?;
    let centered_positions = Array2::from_shape_fn((atom_count, 3), |(atom, component)| {
        atom_positions[(atom, component)] - center_of_mass[component]
    });
    let inertia_tensor = dmdw_inertia_tensor(centered_positions.view(), atom_masses)?;
    let eigensystem = real64_symmetric_eigen(inertia_tensor.view(), SymmetricTriangle::Lower)
        .map_err(|_| DebyeError::DmdwRigidBodyEigenDidNotConverge)?;
    let moments_of_inertia = eigensystem.eigenvalues().to_owned();
    let principal_axes = eigensystem.eigenvectors().to_owned();
    let mut projection_modes = Array2::<Real>::zeros((atom_count * 3, 6));

    for atom in 0..atom_count {
        let mass_root = atom_masses[atom].sqrt();
        projection_modes[(atom, 0)] = mass_root;
        projection_modes[(atom_count + atom, 1)] = mass_root;
        projection_modes[(2 * atom_count + atom, 2)] = mass_root;
    }

    for axis_index in 0..3 {
        let axis = [
            principal_axes[(0, axis_index)],
            principal_axes[(1, axis_index)],
            principal_axes[(2, axis_index)],
        ];
        for atom in 0..atom_count {
            let position = [
                centered_positions[(atom, 0)],
                centered_positions[(atom, 1)],
                centered_positions[(atom, 2)],
            ];
            let rotation = cross(axis, position);
            let mass_root = atom_masses[atom].sqrt();
            for component in 0..3 {
                projection_modes[(component * atom_count + atom, 3 + axis_index)] =
                    rotation[component] * mass_root;
            }
        }
    }

    normalize_dmdw_projection_modes(&mut projection_modes)?;

    Ok(DmdwRigidBodyModes {
        center_of_mass,
        centered_positions,
        inertia_tensor,
        moments_of_inertia,
        principal_axes,
        projection_modes,
    })
}

/// Project a DMDW seed vector out of rigid-body modes and normalize it.
///
/// This ports the FEFF `qj0 = qj0 - sum(qj0*TrfD(:,i))*TrfD(:,i)` loop used
/// before Lanczos recursion. `projection_modes` uses FEFF's `TrfD` orientation:
/// rows are seed-vector components and columns are modes to remove. The modes
/// are expected to be pre-normalized, matching `Make_TrfD`.
pub fn dmdw_project_seed_vector(
    seed: ArrayView1<'_, Real>,
    projection_modes: ArrayView2<'_, Real>,
) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_seed_projection(seed, projection_modes)?;
    let mut projected = seed.to_owned();
    for mode in projection_modes.columns() {
        let projection = projected
            .iter()
            .zip(mode.iter())
            .map(|(&seed_value, &mode_value)| seed_value * mode_value)
            .sum::<Real>();
        for (value, &mode_value) in projected.iter_mut().zip(mode.iter()) {
            *value -= projection * mode_value;
        }
    }
    dmdw_normalize_seed_vector(projected.view())
}

/// Normalize a DMDW Lanczos seed vector with FEFF's Euclidean norm.
pub fn dmdw_normalize_seed_vector(seed: ArrayView1<'_, Real>) -> Result<Array1<Real>, DebyeError> {
    validate_dmdw_seed(seed)?;
    let norm = seed.iter().map(|value| value * value).sum::<Real>().sqrt();
    ensure_finite_output("DMDW seed norm", norm)?;
    if norm == 0.0 {
        return Err(DebyeError::ZeroDmdwSeedNorm);
    }
    Ok(Array1::from_iter(seed.iter().map(|value| value / norm)))
}
