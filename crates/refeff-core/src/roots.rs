//! Polynomial root helpers ported from FEFF.
//!
//! FEFF `MATH/quartc.f90` solves the depressed quartic
//! `a*x^4 + b*x^2 + c*x + d = 0` and stores its four roots back into the
//! coefficient array. The Rust API keeps the same coefficient order while
//! returning a new root array and reporting singular inputs explicitly.

use thiserror::Error;

use crate::Complex;

const TWO_TO_ONE_THIRD: f64 = 1.259_921_1_f32 as f64;
const TWO_TO_TWO_THIRDS: f64 = 1.587_401_f32 as f64;
const ROOT_SIX: f64 = 2.449_489_8_f32 as f64;

/// Error returned by FEFF polynomial-root helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RootError {
    /// Coefficients must have finite real and imaginary parts.
    #[error("quartic coefficient {index} must be finite, got {value:?}")]
    NonFiniteCoefficient { index: usize, value: Complex },
    /// FEFF `quartc` divides by the leading quartic coefficient.
    #[error("quartic leading coefficient must be nonzero")]
    ZeroLeadingCoefficient,
    /// A zero intermediate would make FEFF `quartc` divide by zero.
    #[error("quartic formula is singular at intermediate {name}")]
    SingularIntermediate { name: &'static str },
    /// A root became non-finite after the FEFF formula was evaluated.
    #[error("quartic root {index} is non-finite: {value:?}")]
    NonFiniteRoot { index: usize, value: Complex },
}

/// Solve FEFF's depressed quartic `a*x^4 + b*x^2 + c*x + d = 0`.
///
/// Coefficients use the same order as `quartc.f90`: `[a, b, c, d]`. The four
/// returned roots preserve FEFF's output ordering.
pub fn depressed_quartic_roots(coefficients: [Complex; 4]) -> Result<[Complex; 4], RootError> {
    for (index, &value) in coefficients.iter().enumerate() {
        ensure_finite_coefficient(index, value)?;
    }

    let [a, b, c, d] = coefficients;
    if a.norm() == 0.0 {
        return Err(RootError::ZeroLeadingCoefficient);
    }

    let f = b * b + a * d * 12.0;
    let g = b * b * b * 2.0 + a * c * c * 27.0 - a * b * d * 72.0;
    let a1 = (g + (-4.0 * f * f * f + g * g).sqrt()).powf(1.0 / 3.0);
    if a1.norm() == 0.0 {
        return Err(RootError::SingularIntermediate { name: "A1" });
    }

    let b1 = f * (2.0 * TWO_TO_ONE_THIRD);
    let p = ((-4.0 * b + b1 / a1 + a1 * TWO_TO_TWO_THIRDS) / a).sqrt();
    if p.norm() == 0.0 {
        return Err(RootError::SingularIntermediate { name: "P" });
    }

    let d1 = 8.0 * b + b1 / a1 + a1 * TWO_TO_TWO_THIRDS;
    let d2 = c * (12.0 * ROOT_SIX) / p;
    let q_plus = (-(d1 + d2) / a).sqrt();
    let q_minus = (-(d1 - d2) / a).sqrt();
    let amplitude = 1.0 / (2.0 * ROOT_SIX);

    let roots = [
        (p - q_plus) * amplitude,
        (p + q_plus) * amplitude,
        -(p + q_minus) * amplitude,
        (-p + q_minus) * amplitude,
    ];
    for (index, &value) in roots.iter().enumerate() {
        if !is_finite(value) {
            return Err(RootError::NonFiniteRoot { index, value });
        }
    }
    Ok(roots)
}

fn ensure_finite_coefficient(index: usize, value: Complex) -> Result<(), RootError> {
    if !is_finite(value) {
        return Err(RootError::NonFiniteCoefficient { index, value });
    }
    Ok(())
}

fn is_finite(value: Complex) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depressed_quartic_roots_match_feff_real_even_case() -> Result<(), RootError> {
        let roots = depressed_quartic_roots([
            Complex::new(1.0, 0.0),
            Complex::new(-5.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
        ])?;

        assert_complex_close(
            roots[0],
            Complex::new(0.9999999699988967, -3.017687356679965e-9),
        );
        assert_complex_close(
            roots[1],
            Complex::new(1.999999924021137, 1.5088436903931213e-9),
        );
        assert_complex_close(
            roots[2],
            Complex::new(-1.999999924021137, -1.5088436903931213e-9),
        );
        assert_complex_close(
            roots[3],
            Complex::new(-0.9999999699988967, 3.017687356679965e-9),
        );
        Ok(())
    }

    #[test]
    fn depressed_quartic_roots_match_feff_mixed_case() -> Result<(), RootError> {
        let roots = depressed_quartic_roots([
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ])?;

        assert_complex_close(
            roots[0],
            Complex::new(0.7806386155810884, 1.603168234229653),
        );
        assert_complex_close(
            roots[1],
            Complex::new(0.7806386171247728, -1.6031682192559897),
        );
        assert_complex_close(
            roots[2],
            Complex::new(-0.7806386293301134, 0.8053848631669775),
        );
        assert_complex_close(
            roots[3],
            Complex::new(-0.7806386033757476, -0.8053848781406405),
        );
        Ok(())
    }

    #[test]
    fn depressed_quartic_roots_match_feff_complex_case() -> Result<(), RootError> {
        let roots = depressed_quartic_roots([
            Complex::new(0.75, -0.2),
            Complex::new(-1.5, 0.6),
            Complex::new(0.3, 0.4),
            Complex::new(2.2, -0.7),
        ])?;

        assert_complex_close(
            roots[0],
            Complex::new(1.0391579368656545, 0.605001022806941),
        );
        assert_complex_close(
            roots[1],
            Complex::new(1.3083833758671544, -0.6739852646311872),
        );
        assert_complex_close(
            roots[2],
            Complex::new(-1.2512579751357262, -0.5149432179101516),
        );
        assert_complex_close(
            roots[3],
            Complex::new(-1.0962833375970826, 0.5839274597343977),
        );
        Ok(())
    }

    #[test]
    fn depressed_quartic_rejects_invalid_inputs() {
        assert_eq!(
            depressed_quartic_roots([
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
            ]),
            Err(RootError::ZeroLeadingCoefficient)
        );
        assert!(matches!(
            depressed_quartic_roots([
                Complex::new(1.0, 0.0),
                Complex::new(f64::NAN, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
            ]),
            Err(RootError::NonFiniteCoefficient { index: 1, .. })
        ));
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }
}
