use super::*;

pub(super) fn validate_finite(name: &'static str, value: Real) -> Result<(), EelsError> {
    if !value.is_finite() {
        return Err(EelsError::NonFiniteInput { name, value });
    }
    Ok(())
}

pub(super) fn validate_finite_array(
    name: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_finite_matrix(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_finite_tensor(
    name: &'static str,
    values: ArrayView3<'_, Real>,
) -> Result<(), EelsError> {
    for &value in &values {
        validate_finite(name, value)?;
    }
    Ok(())
}

pub(super) fn validate_qmesh_input(input: EelsQMeshInput<'_>) -> Result<(), EelsError> {
    if input.theta_x.len() != input.theta_y.len() {
        return Err(EelsError::QMeshLengthMismatch {
            theta_x_len: input.theta_x.len(),
            theta_y_len: input.theta_y.len(),
        });
    }
    for &value in &input.beam_direction {
        validate_finite("beam_direction", value)?;
    }
    validate_finite_array("theta_x", input.theta_x)?;
    validate_finite_array("theta_y", input.theta_y)?;
    Ok(())
}
