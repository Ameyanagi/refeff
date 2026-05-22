use super::*;

pub(super) fn validate_positive_limit(name: &'static str, value: usize) -> Result<(), GenfmtError> {
    if value == 0 || isize::try_from(value).is_err() {
        Err(GenfmtError::InvalidAngularLimit { name, value })
    } else {
        Ok(())
    }
}

pub(super) fn checked_double_plus_one(
    name: &'static str,
    value: usize,
) -> Result<usize, GenfmtError> {
    value
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(GenfmtError::InvalidAngularLimit { name, value })
}

pub(super) fn checked_odd_factor(
    value: usize,
    name: &'static str,
    limit: usize,
) -> Result<Real, GenfmtError> {
    let factor = value.checked_mul(2).and_then(|value| value.checked_sub(1));
    factor
        .map(|value| value as Real)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value: limit })
}

pub(super) fn checked_count(name: &'static str, value: usize) -> Result<usize, GenfmtError> {
    value
        .checked_add(1)
        .ok_or(GenfmtError::InvalidAngularLimit { name, value })
}

pub(super) fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), GenfmtError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteScalar { field, value })
    }
}

pub(super) fn validate_finite_complex(
    field: &'static str,
    value: Complex,
) -> Result<(), GenfmtError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(GenfmtError::NonFiniteComplex {
            field,
            real: value.re,
            imaginary: value.im,
        })
    }
}
