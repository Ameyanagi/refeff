//! Shared SCREEN validation helpers.

use num_complex::Complex32;

use crate::{Complex, Real};

use super::types::ScreenError;

pub(super) fn validate_active_count(active_count: usize, len: usize) -> Result<(), ScreenError> {
    if active_count > len {
        Err(ScreenError::ActiveCountOutOfRange { active_count, len })
    } else {
        Ok(())
    }
}

pub(super) fn checked_radial_add(
    name: &'static str,
    value: isize,
    increment: isize,
) -> Result<isize, ScreenError> {
    value
        .checked_add(increment)
        .ok_or(ScreenError::IndexSizeOverflow { name })
}

pub(super) fn positive_radial_bound(
    name: &'static str,
    value: isize,
) -> Result<usize, ScreenError> {
    if value > 0 {
        Ok(value as usize)
    } else {
        Err(ScreenError::NonPositiveRadialBound { name, value })
    }
}

pub(super) fn complex32_result(
    name: &'static str,
    value: Complex,
) -> Result<Complex32, ScreenError> {
    let single = Complex32::new(value.re as f32, value.im as f32);
    if single.re.is_finite() && single.im.is_finite() {
        Ok(single)
    } else {
        Err(ScreenError::NonFiniteComplexResult {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), ScreenError> {
    if actual < minimum {
        Err(ScreenError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

pub(super) fn validate_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteInput { name, value })
    }
}

pub(super) fn validate_finite_complex_input(
    name: &'static str,
    value: Complex,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexInput {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_positive(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveInput { name, value })
    }
}

pub(super) fn validate_result_finite_complex(
    name: &'static str,
    value: Complex,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexResult {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_increasing(
    lower_name: &'static str,
    lower: Real,
    upper_name: &'static str,
    upper: Real,
) -> Result<(), ScreenError> {
    if upper > lower {
        Ok(())
    } else {
        Err(ScreenError::NonIncreasingInput {
            lower_name,
            upper_name,
            lower,
            upper,
        })
    }
}

pub(super) fn validate_result_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteResult { name, value })
    }
}

pub(super) fn validate_positive_result(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_result_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveResult { name, value })
    }
}

pub(super) fn validate_active_matrix_shape(
    name: &'static str,
    rows: usize,
    columns: usize,
    active_count: usize,
) -> Result<(), ScreenError> {
    if rows < active_count || columns < active_count {
        Err(ScreenError::MatrixTooSmall {
            name,
            rows,
            columns,
            active_count,
        })
    } else {
        Ok(())
    }
}

pub(super) fn validate_finite_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Real,
) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteMatrixInput {
            name,
            row,
            column,
            value,
        })
    }
}

pub(super) fn validate_finite_complex_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Complex,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name,
            row,
            column,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_finite_complex32_matrix(
    name: &'static str,
    row: usize,
    column: usize,
    value: Complex32,
) -> Result<(), ScreenError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name,
            row,
            column,
            real: value.re as Real,
            imaginary: value.im as Real,
        })
    }
}
