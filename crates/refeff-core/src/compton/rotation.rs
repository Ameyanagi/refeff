use super::*;

/// FEFF `cross`: return `a x b` for two 3-vectors.
#[must_use]
pub fn compton_cross_product(a: Vector3, b: Vector3) -> Vector3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Port of FEFF `rotation_axis_angle`.
///
/// FEFF obtains the axis from `a x b` and the angle from
/// `asin(|a x b| / (|a| |b|))`. This intentionally preserves that convention,
/// including the zero angle for parallel and antiparallel vectors.
pub fn compton_rotation_axis_angle(
    a: Vector3,
    b: Vector3,
) -> Result<ComptonRotationAxisAngle, ComptonError> {
    validate_vector("a", a)?;
    validate_vector("b", b)?;
    let a_norm2 = vector_norm2(a);
    let b_norm2 = vector_norm2(b);
    if a_norm2 == 0.0 {
        return Err(ComptonError::ZeroVector { name: "a" });
    }
    if b_norm2 == 0.0 {
        return Err(ComptonError::ZeroVector { name: "b" });
    }

    let axis = compton_cross_product(a, b);
    let ratio = vector_norm2(axis) / (a_norm2 * b_norm2);
    if !ratio.is_finite() {
        return Err(ComptonError::InvalidRotationRatio { value: ratio });
    }
    let clamped_ratio = if ratio > 1.0 && ratio <= 1.0 + ROTATION_RATIO_TOLERANCE {
        1.0
    } else {
        ratio
    };
    if !(0.0..=1.0).contains(&clamped_ratio) {
        return Err(ComptonError::InvalidRotationRatio { value: ratio });
    }

    let theta = clamped_ratio.sqrt().asin();
    if !theta.is_finite() {
        return Err(ComptonError::NonFiniteResult {
            name: "theta",
            value: theta,
        });
    }
    Ok(ComptonRotationAxisAngle { axis, theta })
}

/// Port of FEFF `rotation_matrix`: build a 3x3 axis-angle rotation matrix.
///
/// The returned `ndarray` uses Fortran-order storage to match FEFF matrix
/// traversal while retaining normal Rust `(row, column)` indexing.
pub fn compton_rotation_matrix(axis: Vector3, theta: Real) -> Result<RealMat, ComptonError> {
    validate_vector("axis", axis)?;
    validate_finite("theta", theta)?;
    let axis_norm = vector_norm2(axis).sqrt();
    if axis_norm == 0.0 {
        return Err(ComptonError::ZeroVector { name: "axis" });
    }

    let u = [
        axis[0] / axis_norm,
        axis[1] / axis_norm,
        axis[2] / axis_norm,
    ];
    let (sine, cosine) = theta.sin_cos();
    let delta = 1.0 - cosine;
    let mut rotation = Array2::zeros((3, 3).f());
    rotation[(0, 0)] = cosine + u[0] * u[0] * delta;
    rotation[(0, 1)] = u[0] * u[1] * delta - u[2] * sine;
    rotation[(0, 2)] = u[0] * u[2] * delta + u[1] * sine;
    rotation[(1, 0)] = u[1] * u[0] * delta + u[2] * sine;
    rotation[(1, 1)] = cosine + u[1] * u[1] * delta;
    rotation[(1, 2)] = u[1] * u[2] * delta - u[0] * sine;
    rotation[(2, 0)] = u[2] * u[0] * delta - u[1] * sine;
    rotation[(2, 1)] = u[2] * u[1] * delta + u[0] * sine;
    rotation[(2, 2)] = cosine + u[2] * u[2] * delta;
    validate_matrix_finite("rotation_matrix", rotation.view())?;
    Ok(rotation)
}

/// Port of FEFF `rotate`: multiply a 3x3 matrix by a 3-vector.
pub fn compton_rotate_vector(
    rotation: ArrayView2<'_, Real>,
    vector: Vector3,
) -> Result<Vector3, ComptonError> {
    validate_rotation_shape(rotation)?;
    validate_matrix_finite("rotation", rotation)?;
    validate_vector("vector", vector)?;
    let rotated = rotate_vector_checked_shape(rotation, vector);
    validate_vector("rotated", rotated)?;
    Ok(rotated)
}

/// Port of FEFF `rotate_in_place`.
pub fn compton_rotate_vector_in_place(
    rotation: ArrayView2<'_, Real>,
    vector: &mut Vector3,
) -> Result<(), ComptonError> {
    *vector = compton_rotate_vector(rotation, *vector)?;
    Ok(())
}
