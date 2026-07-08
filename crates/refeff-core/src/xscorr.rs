//! FF2X `xscorr` contour-convolution primitives.
//!
//! FEFF's `FF2X/xscorr.f90` driver is a larger contour integration routine.
//! This module ports its small standalone `lorenz` and `astep` helpers so the
//! contour terms can be assembled from checked Rust functions.

use thiserror::Error;

use crate::{Complex, Real};

const XSCORR_PI: Real = std::f64::consts::PI;

/// Error returned by FF2X xscorr helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum XscorrError {
    /// Loss widths appear in denominators and must be positive.
    #[error("xscorr loss width must be positive and finite, got {value}")]
    InvalidLoss { value: Real },
    /// Scalar inputs must be finite.
    #[error("xscorr {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// The Lorentzian denominator is singular.
    #[error("xscorr Lorentzian denominator is singular")]
    SingularDenominator,
    /// The final value must remain finite.
    #[error("xscorr {field} result is non-finite: ({real}, {imaginary})")]
    NonFiniteOutput {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
}

/// Port of FEFF `lorenz` from `FF2X/xscorr.f90`.
///
/// FEFF carries an unused `ifp` argument; the active expression depends only on
/// `xloss`, vertical contour coordinate `w`, and energy displacement `dele`.
pub fn xscorr_lorentz_kernel(
    loss: Real,
    vertical_frequency: Real,
    delta: Real,
) -> Result<Complex, XscorrError> {
    validate_loss(loss)?;
    validate_scalar("vertical_frequency", vertical_frequency)?;
    validate_scalar("delta", delta)?;

    let pole = Complex::new(-delta, vertical_frequency);
    let denominator = Complex::new(loss * loss, 0.0) + pole.powi(2);
    if denominator.re == 0.0 && denominator.im == 0.0 {
        return Err(XscorrError::SingularDenominator);
    }
    let value = loss / XSCORR_PI / denominator;
    ensure_finite_output("lorenz", value)?;
    Ok(value)
}

/// Port of FEFF `astep`: `1 - fermi*cauchy` arctangent step.
pub fn xscorr_arctangent_step(loss: Real, delta: Real) -> Result<Real, XscorrError> {
    validate_loss(loss)?;
    validate_scalar("delta", delta)?;

    let value = (0.5 + (delta / loss).atan() / XSCORR_PI).clamp(0.0, 1.0);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(XscorrError::NonFiniteScalar {
            field: "astep",
            value,
        })
    }
}

fn validate_loss(value: Real) -> Result<(), XscorrError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(XscorrError::InvalidLoss { value })
    }
}

fn validate_scalar(field: &'static str, value: Real) -> Result<(), XscorrError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(XscorrError::NonFiniteScalar { field, value })
    }
}

fn ensure_finite_output(field: &'static str, value: Complex) -> Result<(), XscorrError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(XscorrError::NonFiniteOutput {
            field,
            real: value.re,
            imaginary: value.im,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() <= 1.0e-12 * expected.norm().max(1.0),
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    #[test]
    fn xscorr_lorentz_kernel_matches_feff_lorenz_reference() -> Result<(), XscorrError> {
        let cases = [
            (
                0.08,
                0.02,
                -0.11,
                Complex::new(1.328_393_564_844_594_4, -0.322_924_402_503_658_3),
            ),
            (0.08, 0.13, 0.0, Complex::new(-2.425_218_180_447_928_7, 0.0)),
            (
                0.15,
                0.05,
                0.19,
                Complex::new(0.763_516_919_522_092, 0.258_588_618_019_959_9),
            ),
        ];

        for (loss, vertical_frequency, delta, expected) in cases {
            assert_complex_close(
                xscorr_lorentz_kernel(loss, vertical_frequency, delta)?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn xscorr_arctangent_step_matches_feff_astep_reference() -> Result<(), XscorrError> {
        assert_close(
            xscorr_arctangent_step(0.08, -0.11)?,
            0.200_152_074_361_686_7,
        );
        assert_close(xscorr_arctangent_step(0.08, 0.0)?, 0.5);
        assert_close(xscorr_arctangent_step(0.08, 0.22)?, 0.888_982_741_545_000_1);
        Ok(())
    }

    #[test]
    fn xscorr_helpers_reject_invalid_inputs() {
        assert!(matches!(
            xscorr_lorentz_kernel(0.0, 0.02, -0.11),
            Err(XscorrError::InvalidLoss { .. })
        ));
        assert!(matches!(
            xscorr_arctangent_step(0.08, Real::NAN),
            Err(XscorrError::NonFiniteScalar { field: "delta", .. })
        ));
    }
}
