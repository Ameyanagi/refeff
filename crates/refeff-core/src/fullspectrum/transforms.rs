//! FULLSPECTRUM Kramers-Kronig and Hamaker transforms.

use ndarray::{Array1, ArrayView1};

use crate::interpolation::{LintCache, lint_with_cache};
use crate::{Complex, Real};

use super::types::*;
use super::validation::{
    validate_finite_value, validate_matching_len, validate_strictly_increasing_grid,
    validate_transform_len,
};

/// Port of `FULLSPECTRUM/kk.f90`: Kramers-Kronig transform of `eps2`.
///
/// FEFF evaluates the principal-value integral at interval midpoints, using an
/// analytic linear-`eps2` contribution near the singularity and trapezoid
/// contributions elsewhere, then linearly interpolates the midpoint result back
/// to the input grid. The first and last output values copy their neighbors,
/// matching the Fortran endpoint convention.
pub fn full_spectrum_kramers_kronig(
    input: FullSpectrumKramersKronigInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_transform_len("kramers_kronig", input.omega.len())?;
    validate_matching_len("epsilon2", input.epsilon2.len(), input.omega.len())?;
    validate_strictly_increasing_grid(input.omega)?;
    for (row, value) in input.epsilon2.iter().copied().enumerate() {
        validate_finite_value("epsilon2", row, value)?;
    }

    let midpoints = midpoint_grid(input.omega);
    let midpoint_values = midpoints
        .iter()
        .copied()
        .map(|midpoint| kk_midpoint_value(input.omega, input.epsilon2, midpoint))
        .collect::<Result<Vec<_>, _>>()?;
    interpolate_midpoints_to_grid(input.omega, &midpoints, &midpoint_values)
}

/// Port of `FULLSPECTRUM/hamaker.f90`: dielectric transform on the imaginary axis.
///
/// The FEFF routine integrates `omega' * eps2(omega') / (omega'^2 + omega^2)`
/// at interval midpoints, applies the `2/pi` factor, and interpolates back to
/// the input grid. The real part of the input epsilon is ignored by FEFF.
pub fn full_spectrum_hamaker_transform(
    input: FullSpectrumHamakerInput<'_>,
) -> Result<Array1<Real>, FullSpectrumError> {
    validate_transform_len("hamaker", input.omega.len())?;
    validate_matching_len("epsilon", input.epsilon.len(), input.omega.len())?;
    validate_strictly_increasing_grid(input.omega)?;
    for row in 0..input.epsilon.len() {
        validate_finite_value("epsilon real", row, input.epsilon[row].re)?;
        validate_finite_value("epsilon imaginary", row, input.epsilon[row].im)?;
    }

    let midpoints = midpoint_grid(input.omega);
    let midpoint_values = midpoints
        .iter()
        .copied()
        .map(|midpoint| hamaker_midpoint_value(input.omega, input.epsilon, midpoint))
        .collect::<Result<Vec<_>, _>>()?;
    interpolate_midpoints_to_grid(input.omega, &midpoints, &midpoint_values)
}

fn midpoint_grid(omega: ArrayView1<'_, Real>) -> Vec<Real> {
    (0..omega.len() - 1)
        .map(|row| 0.5 * (omega[row + 1] + omega[row]))
        .collect()
}

fn kk_midpoint_value(
    omega: ArrayView1<'_, Real>,
    epsilon2: ArrayView1<'_, Real>,
    midpoint: Real,
) -> Result<Real, FullSpectrumError> {
    const ANALYTIC_WINDOW: Real = 25.0;
    const TOO_BIG: Real = 1.0e8;

    let mut integral = 0.0;
    for row in (0..omega.len() - 1).rev() {
        let left = omega[row];
        let right = omega[row + 1];
        let contribution =
            if (left - midpoint).abs().max((right - midpoint).abs()) < ANALYTIC_WINDOW {
                let delta = right - left;
                let mut value = epsilon2[row + 1] - epsilon2[row];
                let slope = value / delta;
                let intercept = epsilon2[row] - slope * left;

                let plus_factor = (right + midpoint) / (left + midpoint);
                if plus_factor <= 0.0 || plus_factor >= TOO_BIG {
                    continue;
                }
                value += plus_factor.ln() * (intercept - slope * midpoint) / 2.0;

                let minus_factor = (right - midpoint) / (left - midpoint);
                if minus_factor == 0.0 {
                    continue;
                }
                value += minus_factor.abs().ln() * (intercept + midpoint * slope) / 2.0;
                if value.abs() < TOO_BIG {
                    value
                } else {
                    continue;
                }
            } else {
                let left_denominator = left.powi(2) - midpoint.powi(2);
                let right_denominator = right.powi(2) - midpoint.powi(2);
                let value = right * epsilon2[row + 1] / right_denominator
                    + left * epsilon2[row] / left_denominator;
                value * (right - left) / 2.0
            };
        integral += contribution;
    }

    let value = 2.0 * integral / std::f64::consts::PI;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

pub(super) fn hamaker_midpoint_value(
    omega: ArrayView1<'_, Real>,
    epsilon: ArrayView1<'_, Complex>,
    midpoint: Real,
) -> Result<Real, FullSpectrumError> {
    let integral = (0..omega.len() - 1)
        .rev()
        .map(|row| {
            let left = omega[row];
            let right = omega[row + 1];
            let left_value = left * epsilon[row].im / (left.powi(2) + midpoint.powi(2));
            let right_value = right * epsilon[row + 1].im / (right.powi(2) + midpoint.powi(2));
            (left_value + right_value) * (right - left) / 2.0
        })
        .sum::<Real>();
    let value = 2.0 * integral / std::f64::consts::PI;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value })
    }
}

pub(super) fn interpolate_midpoints_to_grid(
    omega: ArrayView1<'_, Real>,
    midpoints: &[Real],
    midpoint_values: &[Real],
) -> Result<Array1<Real>, FullSpectrumError> {
    let mut output = vec![0.0; omega.len()];
    if omega.len() == 2 {
        output[0] = midpoint_values[0];
        output[1] = midpoint_values[0];
        return Ok(Array1::from_vec(output));
    }

    let mut cache = LintCache::new();
    for row in 1..omega.len() - 1 {
        output[row] = lint_with_cache(midpoints, midpoint_values, omega[row], &mut cache)
            .map_err(|source| FullSpectrumError::Interpolation { source })?;
    }
    output[0] = output[1];
    output[omega.len() - 1] = output[omega.len() - 2];
    Ok(Array1::from_vec(output))
}
