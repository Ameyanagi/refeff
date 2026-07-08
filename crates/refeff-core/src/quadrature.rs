//! FEFF trapezoid and radial Simpson quadrature helpers.
//!
//! The routines in this module port `MATH/trap.f90`, `MATH/strap.f90`,
//! `MATH/somm.f90`, `MATH/somm2.f90`, `MATH/csomm.f90`, `MATH/csomm2.f90`,
//! `XSPH/csommjas.f90`, and `BAND/gauleg.f90`. They keep FEFF's endpoint
//! corrections for logarithmic radial grids while replacing process
//! termination with typed validation errors.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::{Complex, Real};

const GAUSS_LEGENDRE_EPSILON: Real = 3.0e-14;
const GAUSS_LEGENDRE_MAX_ITERATIONS: usize = 64;

/// Nodes and weights returned by FEFF `GAULEG`.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussLegendreQuadrature {
    nodes: Array1<Real>,
    weights: Array1<Real>,
}

impl GaussLegendreQuadrature {
    /// Gauss-Legendre abscissae on the caller's interval.
    #[must_use]
    pub fn nodes(&self) -> ArrayView1<'_, Real> {
        self.nodes.view()
    }

    /// Weights corresponding to [`Self::nodes`].
    #[must_use]
    pub fn weights(&self) -> ArrayView1<'_, Real> {
        self.weights.view()
    }
}

/// Error returned by quadrature helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum QuadratureError {
    /// Arrays that are integrated together must have identical lengths.
    #[error("{routine} length mismatch: {left_name} has {left_len}, {right_name} has {right_len}")]
    LengthMismatch {
        routine: &'static str,
        left_name: &'static str,
        left_len: usize,
        right_name: &'static str,
        right_len: usize,
    },
    /// A quadrature rule needs more input points than the caller supplied.
    #[error("{routine} requires at least {required} points but received {available}")]
    InsufficientPoints {
        routine: &'static str,
        required: usize,
        available: usize,
    },
    /// Log-grid spacing must be positive and finite.
    #[error("{routine} requires positive finite log-grid step, got {step}")]
    InvalidStep { routine: &'static str, step: Real },
    /// A finite integration bound is required.
    #[error("{routine} requires finite {name} bound, got {value}")]
    NonFiniteBound {
        routine: &'static str,
        name: &'static str,
        value: Real,
    },
    /// Radial grid values must be positive and finite.
    #[error("{routine} requires positive finite {name}[{index}], got {radius}")]
    InvalidRadius {
        routine: &'static str,
        name: &'static str,
        index: usize,
        radius: Real,
    },
    /// The requested integer power cannot be represented for `powi`.
    #[error("{routine} radial power {power} is too large")]
    PowerTooLarge { routine: &'static str, power: usize },
    /// The analytic small-r correction became singular.
    #[error("{routine} has singular endpoint correction for near-origin power {near_origin_power}")]
    SingularCorrection {
        routine: &'static str,
        near_origin_power: Real,
    },
    /// Newton iteration did not converge while finding a Gauss-Legendre node.
    #[error("{routine} root {root_index} did not converge after {iterations} iterations")]
    QuadratureRootDidNotConverge {
        routine: &'static str,
        root_index: usize,
        iterations: usize,
    },
}

/// FEFF `trap`: signed trapezoidal integration of `y(x)`.
pub fn trap(xs: &[Real], ys: &[Real]) -> Result<Real, QuadratureError> {
    const ROUTINE: &str = "trap";
    ensure_pair_lengths(ROUTINE, "x", xs.len(), "y", ys.len())?;
    ensure_min_points(ROUTINE, xs.len(), 2)?;

    let mut sum = ys[0] * (xs[1] - xs[0]);
    for index in 1..(xs.len() - 1) {
        sum += ys[index] * (xs[index + 1] - xs[index - 1]);
    }
    sum += ys[xs.len() - 1] * (xs[xs.len() - 1] - xs[xs.len() - 2]);
    Ok(sum / 2.0)
}

/// FEFF `strap`: trapezoidal integration using absolute interval widths.
///
/// FEFF uses this single-precision helper for spectra that can be tabulated in
/// descending energy order. The Rust port uses [`Real`] but preserves the
/// positive-width rule.
pub fn strap(xs: &[Real], ys: &[Real]) -> Result<Real, QuadratureError> {
    const ROUTINE: &str = "strap";
    ensure_pair_lengths(ROUTINE, "x", xs.len(), "y", ys.len())?;
    ensure_min_points(ROUTINE, xs.len(), 2)?;

    let mut sum = ys[0] * (xs[1] - xs[0]).abs();
    for index in 1..(xs.len() - 1) {
        sum += ys[index] * (xs[index + 1] - xs[index - 1]).abs();
    }
    sum += ys[xs.len() - 1] * (xs[xs.len() - 1] - xs[xs.len() - 2]).abs();
    Ok(sum / 2.0)
}

/// Port of FEFF `BAND/gauleg.f90`: Gauss-Legendre nodes and weights.
///
/// The returned abscissae and weights are mapped from `[-1, 1]` onto
/// `[lower, upper]`, preserving FEFF's Newton iteration and symmetric fill
/// order. A zero-length interval is accepted and produces zero weights.
pub fn gauss_legendre_quadrature(
    lower: Real,
    upper: Real,
    points: usize,
) -> Result<GaussLegendreQuadrature, QuadratureError> {
    const ROUTINE: &str = "gauleg";
    ensure_min_points(ROUTINE, points, 1)?;
    ensure_bound(ROUTINE, "lower", lower)?;
    ensure_bound(ROUTINE, "upper", upper)?;

    let midpoint = 0.5 * (upper + lower);
    let half_width = 0.5 * (upper - lower);
    let roots_to_compute = points.div_ceil(2);
    let mut nodes = Array1::zeros(points);
    let mut weights = Array1::zeros(points);

    for root in 1..=roots_to_compute {
        let mut z = (std::f64::consts::PI * (root as Real - 0.25) / (points as Real + 0.5)).cos();
        let mut converged = false;
        for _ in 0..GAUSS_LEGENDRE_MAX_ITERATIONS {
            let (polynomial, derivative) = legendre_value_and_derivative(points, z);
            let previous_z = z;
            z = previous_z - polynomial / derivative;
            if (z - previous_z).abs() <= GAUSS_LEGENDRE_EPSILON {
                converged = true;
                let left = root - 1;
                let right = points - root;
                nodes[left] = midpoint - half_width * z;
                nodes[right] = midpoint + half_width * z;
                let weight = 2.0 * half_width / ((1.0 - z * z) * derivative * derivative);
                weights[left] = weight;
                weights[right] = weight;
                break;
            }
        }
        if !converged {
            return Err(QuadratureError::QuadratureRootDidNotConverge {
                routine: ROUTINE,
                root_index: root,
                iterations: GAUSS_LEGENDRE_MAX_ITERATIONS,
            });
        }
    }

    Ok(GaussLegendreQuadrature { nodes, weights })
}

/// FEFF `somm`: Simpson integration of `(dp + dq) * r^m` on a log radial grid.
///
/// `near_origin_power` is FEFF's input `da` before the routine overwrites it
/// with the integral result.
pub fn somm(
    radii: &[Real],
    dp: &[Real],
    dq: &[Real],
    step: Real,
    near_origin_power: Real,
    radial_power: usize,
) -> Result<Real, QuadratureError> {
    const ROUTINE: &str = "somm";
    ensure_radial_inputs(ROUTINE, radii, dp)?;
    ensure_pair_lengths(ROUTINE, "radii", radii.len(), "dq", dq.len())?;
    ensure_min_points(ROUTINE, radii.len(), 2)?;
    ensure_step(ROUTINE, step)?;
    ensure_positive_radii(ROUTINE, "radii", radii)?;

    let exponent = checked_power(ROUTINE, next_radial_power(ROUTINE, radial_power)?)?;
    let mut positive = 0.0;
    let mut negative = 0.0;
    for (index, (&radius, (&p_value, &q_value))) in
        radii.iter().zip(dp.iter().zip(dq.iter())).enumerate()
    {
        let weight = simpson_weight(index, radii.len()) * radius.powi(exponent);
        accumulate_signed(weight * p_value, &mut positive, &mut negative);
        accumulate_signed(weight * q_value, &mut positive, &mut negative);
    }

    let mut result = step * (positive + negative) / 3.0;
    let (first_coefficient, second_coefficient) =
        initial_correction_coefficients(ROUTINE, radii, step, near_origin_power, radial_power)?;
    result += first_coefficient * (dp[0] + dq[0]) - second_coefficient * (dp[1] + dq[1]);
    Ok(result)
}

/// FEFF `somm2`: corrected Simpson integration of `dp * r^m` to `rnrm`.
pub fn somm2(
    radii: &[Real],
    values: &[Real],
    step: Real,
    near_origin_power: Real,
    rnrm: Real,
    radial_power: usize,
) -> Result<Real, QuadratureError> {
    const ROUTINE: &str = "somm2";
    ensure_radial_inputs(ROUTINE, radii, values)?;
    ensure_min_points(ROUTINE, radii.len(), 4)?;
    ensure_step(ROUTINE, step)?;
    ensure_positive_radii(ROUTINE, "radii", radii)?;
    ensure_radius(ROUTINE, "rnrm", 0, rnrm)?;

    let exponent = checked_power(ROUTINE, next_radial_power(ROUTINE, radial_power)?)?;
    let (a1, a2, a3) = corrected_endpoint_factors(ROUTINE, radii, step, rnrm)?;
    let mut result = radii
        .iter()
        .zip(values.iter())
        .enumerate()
        .map(|(index, (&radius, &value))| {
            value * radius.powi(exponent) * corrected_simpson_weight(index, radii.len(), a1, a2, a3)
        })
        .sum::<Real>();
    result *= step;

    let (first_coefficient, second_coefficient) =
        initial_correction_coefficients(ROUTINE, radii, step, near_origin_power, radial_power)?;
    result += first_coefficient * values[0] - second_coefficient * values[1];
    Ok(result)
}

/// FEFF `csomm`: complex Simpson integration of `(dp + dq) * r^m`.
pub fn csomm(
    radii: &[Real],
    dp: &[Complex],
    dq: &[Complex],
    step: Real,
    near_origin_power: Real,
    radial_power: usize,
) -> Result<Complex, QuadratureError> {
    const ROUTINE: &str = "csomm";
    ensure_radial_inputs(ROUTINE, radii, dp)?;
    ensure_pair_lengths(ROUTINE, "radii", radii.len(), "dq", dq.len())?;
    ensure_min_points(ROUTINE, radii.len(), 2)?;
    ensure_step(ROUTINE, step)?;
    ensure_positive_radii(ROUTINE, "radii", radii)?;

    let exponent = checked_power(ROUTINE, next_radial_power(ROUTINE, radial_power)?)?;
    let mut result = radii
        .iter()
        .zip(dp.iter().zip(dq.iter()))
        .enumerate()
        .map(|(index, (&radius, (&p_value, &q_value)))| {
            (p_value + q_value) * (simpson_weight(index, radii.len()) * radius.powi(exponent))
        })
        .sum::<Complex>();
    result *= step / 3.0;

    let (first_coefficient, second_coefficient) =
        initial_correction_coefficients(ROUTINE, radii, step, near_origin_power, radial_power)?;
    result += (dp[0] + dq[0]) * first_coefficient - (dp[1] + dq[1]) * second_coefficient;
    Ok(result)
}

/// Port of FEFF `XSPH/csommjas.f90`.
///
/// This is the four-step complex Simpson variant used by the NRIXS radial
/// integral path in `radjas`. It integrates `(dp + dq) * r^m` over FEFF's
/// logarithmic radial grid and applies the same near-origin correction as
/// [`csomm`].
pub fn csommjas(
    radii: &[Real],
    dp: &[Complex],
    dq: &[Complex],
    step: Real,
    near_origin_power: Real,
    radial_power: usize,
) -> Result<Complex, QuadratureError> {
    const ROUTINE: &str = "csommjas";
    ensure_radial_inputs(ROUTINE, radii, dp)?;
    ensure_pair_lengths(ROUTINE, "radii", radii.len(), "dq", dq.len())?;
    ensure_min_points(ROUTINE, radii.len(), 2)?;
    ensure_step(ROUTINE, step)?;
    ensure_positive_radii(ROUTINE, "radii", radii)?;

    let exponent = checked_power(ROUTINE, next_radial_power(ROUTINE, radial_power)?)?;
    let mut result = Complex::new(0.0, 0.0);
    let mut k = radii.len();
    loop {
        let index = k - 1;
        let weight = if k == radii.len() || k < 5 {
            14.0
        } else {
            28.0
        };
        result += (dp[index] + dq[index]) * (weight * radii[index].powi(exponent));
        if k <= 4 {
            break;
        }
        k -= 4;
    }
    let lower_boundary = k;

    let mut j = radii.len() - 1;
    while j > lower_boundary {
        let index = j - 1;
        result += (dp[index] + dq[index]) * (64.0 * radii[index].powi(exponent));
        j -= 2;
    }

    let mut l = radii.len() - 2;
    while l > lower_boundary {
        let index = l - 1;
        result += (dp[index] + dq[index]) * (24.0 * radii[index].powi(exponent));
        if l <= 4 {
            break;
        }
        l -= 4;
    }

    result *= step / 45.0;

    let (first_coefficient, second_coefficient) =
        initial_correction_coefficients(ROUTINE, radii, step, near_origin_power, radial_power)?;
    result += (dp[0] + dq[0]) * first_coefficient - (dp[1] + dq[1]) * second_coefficient;
    Ok(result)
}

/// FEFF `csomm2`: corrected complex Simpson integration of `dp * r` to `rnrm`.
pub fn csomm2(
    radii: &[Real],
    values: &[Complex],
    step: Real,
    near_origin_power: Real,
    rnrm: Real,
) -> Result<Complex, QuadratureError> {
    const ROUTINE: &str = "csomm2";
    ensure_radial_inputs(ROUTINE, radii, values)?;
    ensure_min_points(ROUTINE, radii.len(), 4)?;
    ensure_step(ROUTINE, step)?;
    ensure_positive_radii(ROUTINE, "radii", radii)?;
    ensure_radius(ROUTINE, "rnrm", 0, rnrm)?;

    let (a1, a2, a3) = corrected_endpoint_factors(ROUTINE, radii, step, rnrm)?;
    let mut result = radii
        .iter()
        .zip(values.iter())
        .enumerate()
        .map(|(index, (&radius, &value))| {
            value * (radius * corrected_simpson_weight(index, radii.len(), a1, a2, a3))
        })
        .sum::<Complex>();
    result *= step;

    let (first_coefficient, second_coefficient) =
        initial_correction_coefficients(ROUTINE, radii, step, near_origin_power, 0)?;
    result += values[0] * first_coefficient - values[1] * second_coefficient;
    Ok(result)
}

fn legendre_value_and_derivative(points: usize, value: Real) -> (Real, Real) {
    let mut p1 = 1.0;
    let mut p2 = 0.0;
    for order in 1..=points {
        let p3 = p2;
        p2 = p1;
        let order_real = order as Real;
        p1 = ((2.0 * order_real - 1.0) * value * p2 - (order_real - 1.0) * p3) / order_real;
    }
    let derivative = points as Real * (value * p1 - p2) / (value * value - 1.0);
    (p1, derivative)
}

fn simpson_weight(index: usize, len: usize) -> Real {
    if index == 0 || index + 1 == len {
        1.0
    } else if (index + 1).is_multiple_of(2) {
        4.0
    } else {
        2.0
    }
}

fn corrected_simpson_weight(index: usize, len: usize, a1: Real, a2: Real, a3: Real) -> Real {
    let position = index + 1;
    if position == 1 {
        9.0 / 24.0
    } else if position == 2 {
        28.0 / 24.0
    } else if position == 3 {
        23.0 / 24.0
    } else if position == len - 3 {
        25.0 / 24.0 - a2 + a3
    } else if position == len - 2 {
        0.5 + a1 - 3.0 * a2 - a3
    } else if position == len - 1 {
        -1.0 / 24.0 + 5.0 * a2 - a3
    } else if position == len {
        -a2 + a3
    } else {
        1.0
    }
}

fn corrected_endpoint_factors(
    routine: &'static str,
    radii: &[Real],
    step: Real,
    rnrm: Real,
) -> Result<(Real, Real, Real), QuadratureError> {
    let reference_radius = radii[radii.len() - 3];
    let a1 = (rnrm / reference_radius).ln() / step;
    if !a1.is_finite() {
        return Err(QuadratureError::InvalidRadius {
            routine,
            name: "rnrm",
            index: 0,
            radius: rnrm,
        });
    }
    let a2 = a1.powi(2) / 8.0;
    let a3 = a1.powi(3) / 12.0;
    Ok((a1, a2, a3))
}

fn initial_correction_coefficients(
    routine: &'static str,
    radii: &[Real],
    step: Real,
    near_origin_power: Real,
    radial_power: usize,
) -> Result<(Real, Real), QuadratureError> {
    let mm = next_radial_power(routine, radial_power)?;
    let mm_real = mm as Real;
    let d1 = near_origin_power + mm_real;
    let expm1 = step.exp() - 1.0;
    let denominator = d1 * (d1 + 1.0) * expm1 * ((d1 - 1.0) * step).exp();
    let secondary_denominator = expm1 * (d1 + 1.0);
    if denominator == 0.0
        || secondary_denominator == 0.0
        || d1 == 0.0
        || !denominator.is_finite()
        || !secondary_denominator.is_finite()
    {
        return Err(QuadratureError::SingularCorrection {
            routine,
            near_origin_power,
        });
    }

    let radial_power_i32 = checked_power(routine, radial_power)?;
    let mm_i32 = checked_power(routine, mm)?;
    let second_coefficient = radii[0] * radii[1].powi(radial_power_i32) / denominator;
    let first_coefficient = radii[0].powi(mm_i32) * (1.0 + 1.0 / secondary_denominator) / d1;
    Ok((first_coefficient, second_coefficient))
}

fn accumulate_signed(value: Real, positive: &mut Real, negative: &mut Real) {
    if value < 0.0 {
        *negative += value;
    } else if value > 0.0 {
        *positive += value;
    }
}

fn ensure_radial_inputs<T>(
    routine: &'static str,
    radii: &[Real],
    values: &[T],
) -> Result<(), QuadratureError> {
    ensure_pair_lengths(routine, "radii", radii.len(), "values", values.len())
}

fn ensure_pair_lengths(
    routine: &'static str,
    left_name: &'static str,
    left_len: usize,
    right_name: &'static str,
    right_len: usize,
) -> Result<(), QuadratureError> {
    if left_len != right_len {
        return Err(QuadratureError::LengthMismatch {
            routine,
            left_name,
            left_len,
            right_name,
            right_len,
        });
    }
    Ok(())
}

fn ensure_min_points(
    routine: &'static str,
    available: usize,
    required: usize,
) -> Result<(), QuadratureError> {
    if available < required {
        return Err(QuadratureError::InsufficientPoints {
            routine,
            required,
            available,
        });
    }
    Ok(())
}

fn ensure_step(routine: &'static str, step: Real) -> Result<(), QuadratureError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(QuadratureError::InvalidStep { routine, step });
    }
    Ok(())
}

fn ensure_bound(
    routine: &'static str,
    name: &'static str,
    value: Real,
) -> Result<(), QuadratureError> {
    if !value.is_finite() {
        return Err(QuadratureError::NonFiniteBound {
            routine,
            name,
            value,
        });
    }
    Ok(())
}

fn ensure_positive_radii(
    routine: &'static str,
    name: &'static str,
    radii: &[Real],
) -> Result<(), QuadratureError> {
    for (index, &radius) in radii.iter().enumerate() {
        ensure_radius(routine, name, index, radius)?;
    }
    Ok(())
}

fn ensure_radius(
    routine: &'static str,
    name: &'static str,
    index: usize,
    radius: Real,
) -> Result<(), QuadratureError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(QuadratureError::InvalidRadius {
            routine,
            name,
            index,
            radius,
        });
    }
    Ok(())
}

fn checked_power(routine: &'static str, power: usize) -> Result<i32, QuadratureError> {
    i32::try_from(power).map_err(|_| QuadratureError::PowerTooLarge { routine, power })
}

fn next_radial_power(routine: &'static str, power: usize) -> Result<usize, QuadratureError> {
    power
        .checked_add(1)
        .ok_or(QuadratureError::PowerTooLarge { routine, power })
}

#[cfg(test)]
mod tests {
    use super::*;

    type ReferenceRadialInputs = (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Complex>, Vec<Complex>);

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-14,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }

    fn reference_radial_inputs() -> ReferenceRadialInputs {
        let mut radii = Vec::new();
        let mut dp = Vec::new();
        let mut dq = Vec::new();
        let mut cdp = Vec::new();
        let mut cdq = Vec::new();

        for index in 1..=6 {
            let index_real = index as Real;
            let radius = (-0.3 + (index_real - 1.0) * 0.1).exp();
            let p_value = 0.2 + 0.03 * index_real + 0.01 * index_real * index_real;
            let q_value = -0.05 + 0.02 * index_real;
            radii.push(radius);
            dp.push(p_value);
            dq.push(q_value);
            cdp.push(Complex::new(p_value, 0.1 * index_real - 0.2));
            cdq.push(Complex::new(q_value, -0.04 * index_real + 0.3));
        }

        (radii, dp, dq, cdp, cdq)
    }

    #[test]
    fn trap_matches_feff_reference() -> Result<(), QuadratureError> {
        let xs = [0.0, 0.25, 0.5, 1.0];
        let ys = [1.0, 1.5, 2.25, 4.0];

        assert_close(trap(&xs, &ys)?, 2.34375);
        Ok(())
    }

    #[test]
    fn strap_uses_absolute_widths() -> Result<(), QuadratureError> {
        let xs = [1.0, 0.5, 0.25, 0.0];
        let ys = [4.0, 2.25, 1.5, 1.0];

        assert_close(strap(&xs, &ys)?, 2.34375);
        Ok(())
    }

    #[test]
    fn gauss_legendre_quadrature_matches_feff_reference() -> Result<(), QuadratureError> {
        let table = gauss_legendre_quadrature(-1.0, 2.0, 5)?;
        let expected_nodes = [
            -0.859269768907996,
            -0.3077039651585247,
            0.5,
            1.3077039651585247,
            1.859269768907996,
        ];
        let expected_weights = [
            0.3553903275842726,
            0.7179430057490497,
            0.8533333333333334,
            0.7179430057490497,
            0.3553903275842726,
        ];

        for ((&node, &weight), (&expected_node, &expected_weight)) in table
            .nodes()
            .iter()
            .zip(table.weights().iter())
            .zip(expected_nodes.iter().zip(expected_weights.iter()))
        {
            assert_close(node, expected_node);
            assert_close(weight, expected_weight);
        }
        Ok(())
    }

    #[test]
    fn gauss_legendre_quadrature_handles_even_order() -> Result<(), QuadratureError> {
        let table = gauss_legendre_quadrature(0.25, 1.75, 4)?;
        let expected_nodes = [
            0.35414776630446054,
            0.7450142173113578,
            1.2549857826886424,
            1.6458522336955395,
        ];
        let expected_weights = [
            0.2608911338530857,
            0.4891088661469098,
            0.4891088661469098,
            0.2608911338530857,
        ];

        for ((&node, &weight), (&expected_node, &expected_weight)) in table
            .nodes()
            .iter()
            .zip(table.weights().iter())
            .zip(expected_nodes.iter().zip(expected_weights.iter()))
        {
            assert_close(node, expected_node);
            assert_close(weight, expected_weight);
        }
        Ok(())
    }

    #[test]
    fn somm_matches_feff_reference() -> Result<(), QuadratureError> {
        let (radii, dp, dq, _, _) = reference_radial_inputs();

        assert_close(somm(&radii, &dp, &dq, 0.1, 0.5, 1)?, 0.21907941524351088);
        Ok(())
    }

    #[test]
    fn somm2_matches_feff_reference() -> Result<(), QuadratureError> {
        let (radii, dp, _, _, _) = reference_radial_inputs();
        let rnrm = radii[3] * 0.037_f64.exp();

        assert_close(somm2(&radii, &dp, 0.1, 0.5, rnrm, 1)?, 0.12442259439192405);
        Ok(())
    }

    #[test]
    fn csomm_matches_feff_reference() -> Result<(), QuadratureError> {
        let (radii, _, _, cdp, cdq) = reference_radial_inputs();

        assert_complex_close(
            csomm(&radii, &cdp, &cdq, 0.1, 0.5, 1)?,
            Complex::new(0.21907941524351088, 0.1443916648140805),
        );
        Ok(())
    }

    #[test]
    fn csommjas_matches_feff_reference() -> Result<(), QuadratureError> {
        let mut radii = Vec::new();
        let mut dp = Vec::new();
        let mut dq = Vec::new();
        for index in 1..=9 {
            let index_real = index as Real;
            radii.push((-0.7 + 0.08 * (index_real - 1.0)).exp());
            dp.push(Complex::new(
                0.18 * index_real + 0.011 * index_real * index_real,
                -0.04 * index_real + 0.003 * index_real * index_real,
            ));
            dq.push(Complex::new(
                -0.025 * index_real + 0.002 * index_real * index_real,
                0.07 * index_real - 0.004 * index_real * index_real,
            ));
        }
        assert_complex_close(
            csommjas(&radii, &dp, &dq, 0.08, 0.5, 0)?,
            Complex::new(3.444_824_289_178_648e-1, 2.717_724_571_355_891e-2),
        );

        radii.clear();
        dp.clear();
        dq.clear();
        for index in 1..=8 {
            let index_real = index as Real;
            radii.push((-1.2 + 0.11 * (index_real - 1.0)).exp());
            dp.push(Complex::new(
                (-1.0_f64).powi(index) * (0.07 * index_real + 0.006 * index_real * index_real),
                0.025 * index_real - 0.002 * index_real * index_real,
            ));
            dq.push(Complex::new(
                0.015 * index_real - 0.001 * index_real * index_real,
                (-1.0_f64).powi(index + 1) * 0.035 * index_real,
            ));
        }
        assert_complex_close(
            csommjas(&radii, &dp, &dq, 0.11, 2.75, 2)?,
            Complex::new(-1.502_328_394_263_977e-2, 1.147_925_975_546_005e-2),
        );

        radii.clear();
        dp.clear();
        dq.clear();
        for index in 1..=5 {
            let index_real = index as Real;
            radii.push((-0.35 + 0.2 * (index_real - 1.0)).exp());
            dp.push(Complex::new(
                0.05 * index_real * index_real,
                -0.013 * index_real,
            ));
            dq.push(Complex::new(
                -0.017 * index_real,
                0.009 * index_real * index_real,
            ));
        }
        assert_complex_close(
            csommjas(&radii, &dp, &dq, 0.2, 1.2, 1)?,
            Complex::new(5.934_517_428_804_738e-1, 7.196_569_719_927_988e-2),
        );
        Ok(())
    }

    #[test]
    fn csomm2_matches_feff_reference() -> Result<(), QuadratureError> {
        let (radii, _, _, cdp, _) = reference_radial_inputs();
        let rnrm = radii[3] * 0.037_f64.exp();

        assert_complex_close(
            csomm2(&radii, &cdp, 0.1, 0.5, rnrm)?,
            Complex::new(0.14214372723206203, -0.21435871394141595),
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            trap(&[0.0], &[1.0]),
            Err(QuadratureError::InsufficientPoints {
                routine: "trap",
                ..
            })
        ));
        assert!(matches!(
            somm(&[1.0, 2.0], &[1.0, 2.0], &[1.0], 0.1, 0.5, 0),
            Err(QuadratureError::LengthMismatch {
                routine: "somm",
                ..
            })
        ));
        assert!(matches!(
            somm2(
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 2.0, 3.0, 4.0],
                0.0,
                0.5,
                3.5,
                0
            ),
            Err(QuadratureError::InvalidStep {
                routine: "somm2",
                ..
            })
        ));
        assert!(matches!(
            csommjas(
                &[1.0],
                &[Complex::new(1.0, 0.0)],
                &[Complex::new(0.0, 0.0)],
                0.1,
                0.5,
                0
            ),
            Err(QuadratureError::InsufficientPoints {
                routine: "csommjas",
                ..
            })
        ));
        assert!(matches!(
            gauss_legendre_quadrature(0.0, 1.0, 0),
            Err(QuadratureError::InsufficientPoints {
                routine: "gauleg",
                required: 1,
                available: 0,
            })
        ));
        assert!(matches!(
            gauss_legendre_quadrature(Real::NAN, 1.0, 2),
            Err(QuadratureError::NonFiniteBound {
                routine: "gauleg",
                name: "lower",
                ..
            })
        ));
    }
}
