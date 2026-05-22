//! Shared FULLSPECTRUM validation helpers.

use ndarray::ArrayView1;

use crate::Real;

use super::types::FullSpectrumError;

pub(super) fn validate_positive(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if !value.is_finite() {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FullSpectrumError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

pub(super) fn validate_finite(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    }
}

pub(super) fn validate_transform_len(
    name: &'static str,
    len: usize,
) -> Result<(), FullSpectrumError> {
    if len >= 2 {
        Ok(())
    } else {
        Err(FullSpectrumError::TooFewRows { name, len })
    }
}

pub(super) fn validate_strictly_increasing_grid(
    omega: ArrayView1<'_, Real>,
) -> Result<(), FullSpectrumError> {
    for (row, value) in omega.iter().copied().enumerate() {
        validate_finite_value("omega", row, value)?;
        if row > 0 && value <= omega[row - 1] {
            return Err(FullSpectrumError::NonIncreasingOmega {
                row,
                previous: omega[row - 1],
                current: value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_matching_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), FullSpectrumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FullSpectrumError::LengthMismatch {
            field,
            actual,
            expected,
        })
    }
}

pub(super) fn validate_segment_len(
    field: &'static str,
    segment: usize,
    actual: usize,
    expected: usize,
) -> Result<(), FullSpectrumError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FullSpectrumError::SegmentLengthMismatch {
            field,
            segment,
            actual,
            expected,
        })
    }
}

pub(super) fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FullSpectrumError> {
    if active_len > len {
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

pub(super) fn validate_finite_value(
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteValue { field, row, value })
    }
}

pub(super) fn push_sum_rule_value(
    values: &mut Vec<Real>,
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        values.push(value);
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteSumRule { field, row, value })
    }
}
