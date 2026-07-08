//! Small scalar special-function helpers from FEFF.
//!
//! This module currently ports `MATH/xlogx.f90`, used by the self-energy
//! routines, and the SLATEC complex digamma/cotangent helpers embedded in
//! `DMDW/m_dmdw.f90`. The Rust API validates FEFF's implicit contracts and
//! returns structured errors for invalid inputs.

use thiserror::Error;

use crate::{Complex, Real};

const X_LOG_X_TOLERANCE: Real = 1.0e-10;
#[allow(clippy::excessive_precision)]
const CPSI_BERN: [Real; 13] = [
    0.833_333_333_333_333_3e-1,
    -0.833_333_333_333_333_3e-2,
    0.396_825_396_825_396_83e-2,
    -0.416_666_666_666_666_67e-2,
    0.757_575_757_575_757_6e-2,
    -0.210_927_960_927_960_93e-1,
    0.833_333_333_333_333_3e-1,
    -0.443_259_803_921_568_63,
    0.305_395_433_027_011_97e1,
    -0.264_562_121_212_121_21e2,
    0.281_460_144_927_536_23e3,
    -0.345_488_539_377_289_38e4,
    0.548_275_833_333_333_3e5,
];
const CPSI_PI: Real = std::f64::consts::PI;
const R1MACH_1: Real = f32::MIN_POSITIVE as Real;
const R1MACH_2: Real = f32::MAX as Real;
const R1MACH_3: Real = (f32::EPSILON as Real) * 0.5;
const R1MACH_4: Real = f32::EPSILON as Real;

/// Error returned by scalar special-function helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum SpecialFunctionError {
    /// The argument must be finite.
    #[error("xlogx argument must be finite, got {value}")]
    NonFiniteArgument { value: Real },
    /// FEFF only uses `xLogx` for nonnegative real arguments.
    #[error("xlogx argument must be nonnegative, got {value}")]
    NegativeArgument { value: Real },
    /// Complex special-function arguments must be finite.
    #[error("complex special-function argument {name} must be finite, got {value:?}")]
    NonFiniteComplexArgument { name: &'static str, value: Complex },
    /// SLATEC `CPSI` overflows near zero.
    #[error("complex digamma argument is too near zero: {value:?}")]
    ComplexDigammaOverflow { value: Complex },
    /// SLATEC `CPSI` loses precision near negative integer poles.
    #[error("complex digamma argument is too near a negative integer pole: {value:?}")]
    ComplexDigammaNearPole { value: Complex },
    /// SLATEC `CCOT` is singular.
    #[error("complex cotangent is singular for input {value:?}")]
    ComplexCotangentSingular { value: Complex },
    /// SLATEC `CCOT` loses precision near a singular point.
    #[error("complex cotangent loses precision for input {value:?}")]
    ComplexCotangentPoorPrecision { value: Complex },
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

/// Port of the SLATEC `CPSI` complex digamma helper embedded in
/// `DMDW/m_dmdw.f90`.
///
/// FEFF uses this routine in the DMDW type-2 phonon self-energy kernel. The
/// constants intentionally follow the single-precision `R1MACH` values used by
/// the original SLATEC routine, while the Rust arithmetic remains `f64`.
pub fn complex_digamma(value: Complex) -> Result<Complex, SpecialFunctionError> {
    validate_complex_argument("CPSI", value)?;
    let (nterm, bound, dxrel, rmin, rbig) = cpsi_constants();
    let mut z = value;
    let original_imaginary = z.im;
    if z.im < 0.0 {
        z = z.conj();
    }

    let mut correction = Complex::new(0.0, 0.0);
    let cabsz = z.norm();
    let x = z.re;
    let y = z.im;
    if !((x >= 0.0 && cabsz > bound) || (x < 0.0 && y.abs() > bound)) {
        if cabsz >= bound {
            correction = -CPSI_PI * complex_cot(CPSI_PI * z)?;
            z = Complex::new(1.0, 0.0) - z;
        } else {
            if cabsz < rmin {
                return Err(SpecialFunctionError::ComplexDigammaOverflow { value });
            }
            if x < -0.5 && y.abs() <= dxrel {
                let nearest = (x - 0.5).trunc();
                if ((z - Complex::new(nearest, 0.0)) / x).norm() < dxrel {
                    return Err(SpecialFunctionError::ComplexDigammaNearPole { value });
                }
            }
            if y == 0.0 && x <= 0.0 && x == x.trunc() {
                return Err(SpecialFunctionError::ComplexDigammaNearPole { value });
            }

            let steps = (bound.mul_add(bound, -(y * y))).sqrt() - x + 1.0;
            let steps = steps.trunc().max(0.0) as usize;
            for _ in 0..steps {
                correction -= Complex::new(1.0, 0.0) / z;
                z += Complex::new(1.0, 0.0);
            }
        }
    }

    let mut result = if cabsz > rbig {
        z.ln() + correction
    } else {
        let z2inv = Complex::new(1.0, 0.0) / (z * z);
        let series = CPSI_BERN
            .iter()
            .take(nterm)
            .rev()
            .fold(Complex::new(0.0, 0.0), |acc, &bernoulli| {
                Complex::new(bernoulli, 0.0) + z2inv * acc
            });
        z.ln() - Complex::new(0.5, 0.0) / z - series * z2inv + correction
    };
    if original_imaginary < 0.0 {
        result = result.conj();
    }
    Ok(result)
}

fn complex_cot(value: Complex) -> Result<Complex, SpecialFunctionError> {
    validate_complex_argument("CCOT", value)?;
    let sqeps = R1MACH_4.sqrt();
    let x2 = 2.0 * value.re;
    let y2 = 2.0 * value.im;
    let denominator = y2.cosh() - x2.cos();
    if denominator == 0.0 {
        return Err(SpecialFunctionError::ComplexCotangentSingular { value });
    }
    if denominator.abs() <= x2.abs().max(1.0) * sqeps {
        return Err(SpecialFunctionError::ComplexCotangentPoorPrecision { value });
    }
    Ok(Complex::new(
        x2.sin() / denominator,
        -y2.sinh() / denominator,
    ))
}

fn validate_complex_argument(
    name: &'static str,
    value: Complex,
) -> Result<(), SpecialFunctionError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(SpecialFunctionError::NonFiniteComplexArgument { name, value })
    }
}

fn cpsi_constants() -> (usize, Real, Real, Real, Real) {
    let nterm = (-0.30 * R1MACH_3.ln()).trunc().max(1.0) as usize;
    let bound =
        0.1171 * (nterm as Real) * (0.1 * R1MACH_3).powf(-1.0 / (2.0 * nterm as Real - 1.0));
    let dxrel = R1MACH_4.sqrt();
    let rmin = (R1MACH_1.ln().max(-R1MACH_2.ln()) + 0.011).exp();
    let rbig = 1.0 / R1MACH_3;
    (nterm, bound, dxrel, rmin, rbig)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EULER_MASCHERONI: Real = 0.577_215_664_901_532_9;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-18,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
        assert!(
            (actual - expected).norm() < tolerance,
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

    #[test]
    fn complex_digamma_matches_known_real_values() -> Result<(), SpecialFunctionError> {
        assert_complex_close_tol(
            complex_digamma(Complex::new(1.0, 0.0))?,
            Complex::new(-EULER_MASCHERONI, 0.0),
            1.0e-10,
        );
        assert_complex_close_tol(
            complex_digamma(Complex::new(0.5, 0.0))?,
            Complex::new(-EULER_MASCHERONI - 2.0 * 2.0_f64.ln(), 0.0),
            1.0e-10,
        );
        Ok(())
    }

    #[test]
    fn complex_digamma_obeys_recurrence_and_conjugation() -> Result<(), SpecialFunctionError> {
        let z = Complex::new(0.7, 1.3);
        let recurrence = complex_digamma(z + Complex::new(1.0, 0.0))?;
        let expected = complex_digamma(z)? + Complex::new(1.0, 0.0) / z;
        assert_complex_close_tol(recurrence, expected, 1.0e-12);

        let lower = complex_digamma(z.conj())?;
        assert_complex_close_tol(lower, complex_digamma(z)?.conj(), 1.0e-12);
        Ok(())
    }

    #[test]
    fn complex_digamma_rejects_invalid_inputs() {
        assert!(matches!(
            complex_digamma(Complex::new(0.0, 0.0)),
            Err(SpecialFunctionError::ComplexDigammaOverflow { .. })
        ));
        assert!(matches!(
            complex_digamma(Complex::new(-1.0, 0.0)),
            Err(SpecialFunctionError::ComplexDigammaNearPole { .. })
        ));
        assert!(matches!(
            complex_digamma(Complex::new(Real::NAN, 0.0)),
            Err(SpecialFunctionError::NonFiniteComplexArgument { .. })
        ));
    }
}
