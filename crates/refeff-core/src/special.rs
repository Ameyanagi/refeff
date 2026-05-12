//! Small scalar special-function helpers from FEFF.
//!
//! This module currently ports `MATH/xlogx.f90`, used by the self-energy
//! routines. The FEFF routine is only called with nonnegative arguments; the
//! Rust API validates that contract and returns a structured error for invalid
//! inputs.

use thiserror::Error;

use crate::{Complex, Real};

const X_LOG_X_TOLERANCE: Real = 1.0e-10;

/// Error returned by scalar special-function helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SpecialFunctionError {
    /// The argument must be finite.
    #[error("xlogx argument must be finite, got {value}")]
    NonFiniteArgument { value: Real },
    /// FEFF only uses `xLogx` for nonnegative real arguments.
    #[error("xlogx argument must be nonnegative, got {value}")]
    NegativeArgument { value: Real },
}

/// Port of FEFF `xLogx` for nonnegative real arguments.
///
/// For `x > 1e-10`, this returns `x * ln(x)`. For small positive values FEFF
/// linearly extrapolates toward zero with slope `ln(1e-10)`. FEFF leaves the
/// exact `x = 0` result undefined; the Rust port returns the limiting value
/// `0 + 0i` instead of propagating uninitialized data.
pub fn x_log_x(value: Real) -> Result<Complex, SpecialFunctionError> {
    if !value.is_finite() {
        return Err(SpecialFunctionError::NonFiniteArgument { value });
    }
    if value < 0.0 {
        return Err(SpecialFunctionError::NegativeArgument { value });
    }
    if value == 0.0 {
        return Ok(Complex::new(0.0, 0.0));
    }
    if value > X_LOG_X_TOLERANCE {
        return Ok(Complex::new(value * value.ln(), 0.0));
    }
    Ok(Complex::new(value * X_LOG_X_TOLERANCE.ln(), 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-18,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    #[test]
    fn x_log_x_matches_feff_reference_values() -> Result<(), SpecialFunctionError> {
        assert_complex_close(x_log_x(2.5)?, Complex::new(2.2907268296853878, 0.0));
        assert_complex_close(
            x_log_x(1.0e-12)?,
            Complex::new(-2.3025850929940458e-11, 0.0),
        );
        Ok(())
    }

    #[test]
    fn x_log_x_returns_limiting_value_at_zero() -> Result<(), SpecialFunctionError> {
        assert_eq!(x_log_x(0.0)?, Complex::new(0.0, 0.0));
        Ok(())
    }

    #[test]
    fn x_log_x_rejects_invalid_inputs() {
        assert!(matches!(
            x_log_x(-1.0e-12),
            Err(SpecialFunctionError::NegativeArgument { .. })
        ));
        assert!(matches!(
            x_log_x(Real::NAN),
            Err(SpecialFunctionError::NonFiniteArgument { .. })
        ));
    }
}
