use super::*;

pub(super) fn validate_grid_for_jzzp(grid: &ComptonGrid) -> Result<(), ComptonError> {
    validate_grid_count("ns", grid.ns())?;
    validate_grid_count("nphi", grid.nphi())?;
    validate_grid_count("nz", grid.nz())?;
    validate_grid_count("nzp", grid.nzp())?;
    validate_real_vec("s", &grid.s)?;
    validate_real_vec("phi", &grid.phi)?;
    validate_real_vec("z", &grid.z)?;
    validate_real_vec("zp", &grid.zp)?;
    validate_rotation_shape(grid.rotation_matrix.view())?;
    validate_matrix_finite("rotation_matrix", grid.rotation_matrix.view())
}

pub(super) fn validate_grid_for_rhozzp(grid: &ComptonGrid) -> Result<(), ComptonError> {
    validate_grid_count("nzp", grid.nzp())?;
    validate_real_vec("zp", &grid.zp)?;
    validate_rotation_shape(grid.rotation_matrix.view())?;
    validate_matrix_finite("rotation_matrix", grid.rotation_matrix.view())
}

pub(super) fn validate_grid_for_profile(
    grid: &ComptonGrid,
    jzzp: ArrayView2<'_, Real>,
) -> Result<(), ComptonError> {
    validate_grid_count("nz", grid.nz())?;
    validate_grid_count("nzp", grid.nzp())?;
    validate_real_vec("z", &grid.z)?;
    validate_real_vec("zp", &grid.zp)?;
    let (rows, columns) = jzzp.dim();
    if rows != grid.nz() || columns != grid.nzp() {
        return Err(ComptonError::InvalidJzzpShape {
            rows,
            columns,
            expected_rows: grid.nz(),
            expected_columns: grid.nzp(),
        });
    }
    Ok(())
}

pub(super) fn default_extent(
    value: Real,
    default: Real,
    value_name: &'static str,
    default_name: &'static str,
) -> Result<Real, ComptonError> {
    if value == 0.0 {
        validate_extent(default_name, default)?;
        if default == 0.0 {
            return Err(ComptonError::InvalidGridExtent {
                name: default_name,
                value: default,
            });
        }
        Ok(default)
    } else {
        validate_extent(value_name, value)?;
        Ok(value)
    }
}

pub(super) fn linspace(start: Real, end: Real, count: usize) -> RealVec {
    let step = (end - start) / (count as Real - 1.0);
    Array1::from_iter((0..count).map(|index| start + step * index as Real))
}

pub(super) fn identity_rotation_matrix() -> RealMat {
    let mut matrix = Array2::zeros((3, 3).f());
    for axis in 0..3 {
        matrix[(axis, axis)] = 1.0;
    }
    matrix
}

pub(super) fn validate_real_vec(name: &'static str, values: &RealVec) -> Result<(), ComptonError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteVector {
                name,
                axis: index,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_vector(name: &'static str, vector: Vector3) -> Result<(), ComptonError> {
    for (axis, value) in vector.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteVector { name, axis, value });
        }
    }
    Ok(())
}

pub(super) fn validate_matrix_finite(
    name: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), ComptonError> {
    for &value in &matrix {
        if !value.is_finite() {
            return Err(ComptonError::NonFiniteResult { name, value });
        }
    }
    Ok(())
}

pub(super) fn validate_rotation_shape(rotation: ArrayView2<'_, Real>) -> Result<(), ComptonError> {
    let (rows, columns) = rotation.dim();
    if rows != 3 || columns != 3 {
        return Err(ComptonError::InvalidRotationMatrixShape { rows, columns });
    }
    Ok(())
}

pub(super) fn rotate_vector_checked_shape(
    rotation: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Vector3 {
    let mut rotated = [0.0; 3];
    for row in 0..3 {
        rotated[row] = (0..3)
            .map(|column| rotation[(row, column)] * vector[column])
            .sum();
    }
    rotated
}

pub(super) fn validate_finite(name: &'static str, value: Real) -> Result<(), ComptonError> {
    if !value.is_finite() {
        return Err(ComptonError::NonFiniteInput { name, value });
    }
    Ok(())
}

pub(super) fn validate_grid_count(name: &'static str, value: usize) -> Result<(), ComptonError> {
    if value < 2 {
        return Err(ComptonError::InvalidGridCount { name, value });
    }
    Ok(())
}

pub(super) fn validate_extent(name: &'static str, value: Real) -> Result<(), ComptonError> {
    if value < 0.0 || !value.is_finite() {
        return Err(ComptonError::InvalidGridExtent { name, value });
    }
    Ok(())
}

pub(super) fn vector_norm2(vector: Vector3) -> Real {
    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
}
