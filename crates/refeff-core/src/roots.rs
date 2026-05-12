//! Polynomial root helpers ported from FEFF.
//!
//! FEFF `MATH/quartc.f90` solves the depressed quartic
//! `a*x^4 + b*x^2 + c*x + d = 0` and stores its four roots back into the
//! coefficient array. The Rust API keeps the same coefficient order while
//! returning a new root array and reporting singular inputs explicitly.
//!
//! FEFF `MATH/czeros.f90` also provides quadratic and cubic helpers used by
//! the self-energy singularity search. Those routines report a root count in
//! `NSol`; Rust wraps the same count plus fixed-capacity storage in
//! [`ComplexRoots`].

use thiserror::Error;

use crate::Complex;

const TWO_TO_ONE_THIRD: f64 = 1.259_921_1_f32 as f64;
const TWO_TO_TWO_THIRDS: f64 = 1.587_401_f32 as f64;
const ROOT_SIX: f64 = 2.449_489_8_f32 as f64;

/// Error returned by FEFF polynomial-root helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RootError {
    /// Coefficients must have finite real and imaginary parts.
    #[error("polynomial coefficient {index} must be finite, got {value:?}")]
    NonFiniteCoefficient { index: usize, value: Complex },
    /// FEFF `quartc` divides by the leading quartic coefficient.
    #[error("quartic leading coefficient must be nonzero")]
    ZeroLeadingCoefficient,
    /// FEFF `czeros` reports `NSol = -1` when all active coefficients vanish.
    #[error("degree-{degree} polynomial is degenerate")]
    DegeneratePolynomial { degree: usize },
    /// A zero intermediate would make the FEFF formula divide by zero.
    #[error("polynomial formula is singular at intermediate {name}")]
    SingularIntermediate { name: &'static str },
    /// A root became non-finite after the FEFF formula was evaluated.
    #[error("polynomial root {index} is non-finite: {value:?}")]
    NonFiniteRoot { index: usize, value: Complex },
}

/// FEFF-compatible roots with `NSol`-style count and fixed-capacity storage.
///
/// `roots()` returns only the first `count()` values, matching the Fortran
/// convention that unused slots in the output array are not meaningful.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexRoots<const N: usize> {
    roots: [Complex; N],
    count: usize,
}

impl<const N: usize> ComplexRoots<N> {
    /// Return the number of meaningful roots, equivalent to FEFF `NSol`.
    #[must_use]
    pub fn count(self) -> usize {
        self.count
    }

    /// Return the meaningful root slice in FEFF output order.
    #[must_use]
    pub fn roots(&self) -> &[Complex] {
        &self.roots[..self.count]
    }

    /// Return the full fixed-capacity storage, including unused trailing slots.
    #[must_use]
    pub fn into_inner(self) -> [Complex; N] {
        self.roots
    }
}

/// Port of FEFF `CQdrtc`: zeros of a quadratic polynomial.
///
/// Coefficients use FEFF order `[a, b, c]` for `a*x^2 + b*x + c = 0`.
/// If `a` is zero, the routine falls back to the linear solution. If both
/// active coefficients vanish, FEFF sets `NSol = -1`; Rust reports
/// [`RootError::DegeneratePolynomial`] instead.
pub fn quadratic_zeros(coefficients: [Complex; 3]) -> Result<ComplexRoots<2>, RootError> {
    ensure_finite_coefficients(&coefficients)?;

    let [a, b, c] = coefficients;
    if a == Complex::new(0.0, 0.0) {
        if b == Complex::new(0.0, 0.0) {
            return Err(RootError::DegeneratePolynomial { degree: 2 });
        }
        return checked_roots([-(c / b), Complex::new(0.0, 0.0)], 1);
    }

    let discriminant = b * b - a * c * 4.0;
    // FEFF leaves `Root` undeclared, so it is a default real. Reproduce the
    // single-precision real conversion before the stable-deflation step.
    let root = discriminant.sqrt().re as f32 as f64;
    let sign = (b.conj() * root).re.abs();
    let q = -0.5 * (b + root * sign);
    if q == Complex::new(0.0, 0.0) {
        return Err(RootError::SingularIntermediate { name: "q" });
    }

    checked_roots([q / a, c / q], 2)
}

/// Port of FEFF `CCubic`: zeros of a cubic polynomial.
///
/// Coefficients use FEFF order `[a, b, c, d]` for
/// `a*x^3 + b*x^2 + c*x + d = 0`. A zero leading coefficient falls back to
/// [`quadratic_zeros`], preserving FEFF's `NSol` behavior.
pub fn cubic_zeros(coefficients: [Complex; 4]) -> Result<ComplexRoots<3>, RootError> {
    ensure_finite_coefficients(&coefficients)?;

    let [leading, b, c, d] = coefficients;
    if leading == Complex::new(0.0, 0.0) {
        let roots = quadratic_zeros([b, c, d])?;
        let mut values = [Complex::new(0.0, 0.0); 3];
        for (target, &source) in values.iter_mut().zip(roots.roots()) {
            *target = source;
        }
        return checked_roots(values, roots.count());
    }

    let a = b / leading;
    let b = c / leading;
    let c = d / leading;
    let q = (a * a - b * 3.0) / 9.0;
    let r = (a * a * a * 2.0 - a * b * 9.0 + c * 27.0) / 54.0;
    let q_cubed = q * q * q;
    let r_squared = r * r;

    let roots = if q.im == 0.0 && r.im == 0.0 && r_squared.im < q_cubed.im {
        let theta = (r / q_cubed.sqrt()).re.acos();
        let q_sqrt = q.sqrt();
        [
            -2.0 * q_sqrt * (theta / 3.0).cos() - a / 3.0,
            -2.0 * q_sqrt * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0,
            -2.0 * q_sqrt * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() - a / 3.0,
        ]
    } else {
        let radical = (r_squared - q_cubed).sqrt();
        let sign = signed_unit((r.conj() * radical).re);
        let p1 = -(r + radical * sign).powf(1.0 / 3.0);
        let p2 = if p1 == Complex::new(0.0, 0.0) {
            Complex::new(0.0, 0.0)
        } else {
            q / p1
        };
        let sum = p1 + p2;
        let difference = p1 - p2;
        let imaginary_factor = Complex::new(0.0, 3.0_f64.sqrt() / 2.0);
        [
            sum - a / 3.0,
            -0.5 * sum - a / 3.0 + imaginary_factor * difference,
            -0.5 * sum - a / 3.0 - imaginary_factor * difference,
        ]
    };

    checked_roots(roots, 3)
}

/// Solve FEFF's depressed quartic `a*x^4 + b*x^2 + c*x + d = 0`.
///
/// Coefficients use the same order as `quartc.f90`: `[a, b, c, d]`. The four
/// returned roots preserve FEFF's output ordering.
pub fn depressed_quartic_roots(coefficients: [Complex; 4]) -> Result<[Complex; 4], RootError> {
    ensure_finite_coefficients(&coefficients)?;

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

fn ensure_finite_coefficients(coefficients: &[Complex]) -> Result<(), RootError> {
    for (index, &value) in coefficients.iter().enumerate() {
        ensure_finite_coefficient(index, value)?;
    }
    Ok(())
}

fn checked_roots<const N: usize>(
    roots: [Complex; N],
    count: usize,
) -> Result<ComplexRoots<N>, RootError> {
    for (index, &value) in roots.iter().take(count).enumerate() {
        if !is_finite(value) {
            return Err(RootError::NonFiniteRoot { index, value });
        }
    }
    Ok(ComplexRoots { roots, count })
}

fn is_finite(value: Complex) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn signed_unit(value: f64) -> f64 {
    if value.is_sign_negative() { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_zeros_match_feff_complex_case() -> Result<(), RootError> {
        let roots = quadratic_zeros([
            Complex::new(1.0, 0.5),
            Complex::new(-2.0, 1.0),
            Complex::new(0.25, -0.75),
        ])?;

        assert_eq!(roots.count(), 2);
        assert_complex_slice_close(
            roots.roots(),
            &[
                Complex::new(-0.232455575436245, -0.3837722122818775),
                Complex::new(1.449885172012256, 0.6176421439353421),
            ],
        );
        Ok(())
    }

    #[test]
    fn quadratic_zeros_match_feff_linear_case() -> Result<(), RootError> {
        let roots = quadratic_zeros([
            Complex::new(0.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(-3.0, 4.0),
        ])?;

        assert_eq!(roots.count(), 1);
        assert_complex_slice_close(roots.roots(), &[Complex::new(2.0, -1.0)]);
        Ok(())
    }

    #[test]
    fn quadratic_zeros_reject_degenerate_and_singular_feff_cases() {
        assert_eq!(
            quadratic_zeros([
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(5.0, 0.0),
            ]),
            Err(RootError::DegeneratePolynomial { degree: 2 })
        );
        assert_eq!(
            quadratic_zeros([
                Complex::new(1.0, 0.0),
                Complex::new(-5.0, 0.0),
                Complex::new(6.0, 0.0),
            ]),
            Err(RootError::SingularIntermediate { name: "q" })
        );
    }

    #[test]
    fn cubic_zeros_match_feff_real_and_complex_cases() -> Result<(), RootError> {
        let real_roots = cubic_zeros([
            Complex::new(1.0, 0.0),
            Complex::new(-6.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(-6.0, 0.0),
        ])?;
        assert_eq!(real_roots.count(), 3);
        assert_complex_slice_close(
            real_roots.roots(),
            &[
                Complex::new(1.0, -1.1102230246251565e-16),
                Complex::new(3.0, -8.871105024750947e-17),
                Complex::new(2.0, 1.9973335271002513e-16),
            ],
        );

        let complex_roots = cubic_zeros([
            Complex::new(0.75, -0.2),
            Complex::new(-1.5, 0.6),
            Complex::new(0.3, 0.4),
            Complex::new(2.2, -0.7),
        ])?;
        assert_eq!(complex_roots.count(), 3);
        assert_complex_slice_close(
            complex_roots.roots(),
            &[
                Complex::new(-0.95518641575802, 0.07021087926138053),
                Complex::new(1.8021704048322575, -1.125150679988228),
                Complex::new(1.219406052419538, 0.8059771451251877),
            ],
        );
        Ok(())
    }

    #[test]
    fn cubic_zeros_match_feff_quadratic_fallback() -> Result<(), RootError> {
        let roots = cubic_zeros([
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.5),
            Complex::new(-2.0, 1.0),
            Complex::new(0.25, -0.75),
        ])?;

        assert_eq!(roots.count(), 2);
        assert_complex_slice_close(
            roots.roots(),
            &[
                Complex::new(-0.232455575436245, -0.3837722122818775),
                Complex::new(1.449885172012256, 0.6176421439353421),
            ],
        );
        Ok(())
    }

    #[test]
    fn cubic_zeros_reject_degenerate_polynomial() {
        assert_eq!(
            cubic_zeros([
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ]),
            Err(RootError::DegeneratePolynomial { degree: 2 })
        );
    }

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

    fn assert_complex_slice_close(actual: &[Complex], expected: &[Complex]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }
}
