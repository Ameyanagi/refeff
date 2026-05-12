//! FEFF trapezoid and radial Simpson quadrature helpers.
//!
//! The routines in this module port `MATH/trap.f90`, `MATH/strap.f90`,
//! `MATH/somm.f90`, `MATH/somm2.f90`, `MATH/csomm.f90`, and
//! `MATH/csomm2.f90`. They keep FEFF's endpoint corrections for logarithmic
//! radial grids while replacing process termination with typed validation
//! errors.

use thiserror::Error;

use crate::{Complex, Real};

/// Error returned by quadrature helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
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
    }
}
