//! FEFF `GENFMT` helper routines.
//!
//! This module ports small, self-contained setup routines used by FEFF's
//! curved-wave multiple-scattering formatter. `lambda_indices` is the Rust
//! equivalent of `GENFMT/setlam.f90`: it builds the Rehr-Albers lambda index
//! arrays `(m, n)` from FEFF's `icalc` mode, path order, and dimension limits.

use ndarray::Array1;
use thiserror::Error;

use crate::Real;

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;

/// Inputs for FEFF `GENFMT/setlam.f90` lambda-index selection.
#[derive(Debug, Clone, Copy)]
pub struct LambdaIndexInput<'a> {
    /// FEFF `icalc` selector: `0..=9` for exact order, `10` for the cute
    /// heuristic, or a negative encoded `(nmax, mmax, iord)` request.
    pub calculation: i32,
    /// FEFF one-based energy index `ie`; the cute heuristic raises `nmax` for
    /// `ie >= 42`.
    pub energy_index: usize,
    /// FEFF `nsc`, used to detect single-scattering paths.
    pub scattering_count: usize,
    /// FEFF `ilinit`, the initial-state angular momentum.
    pub initial_l: usize,
    /// FEFF `beta(1:nleg)` path scattering angles in radians.
    pub beta_angles: &'a [Real],
    /// FEFF `lamtot`, the capacity of `mlam` and `nlam`.
    pub lambda_capacity: usize,
    /// FEFF `mtot`, the maximum magnetic index dimension.
    pub max_m: usize,
    /// FEFF `ntot`, the maximum order index dimension.
    pub max_n: usize,
}

/// FEFF lambda index arrays and associated `setlam` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaIndexSet {
    /// FEFF `mlam(1:lamx)` magnetic indices.
    pub m_indices: Array1<i32>,
    /// FEFF `nlam(1:lamx)` order indices.
    pub n_indices: Array1<i32>,
    /// FEFF `laml0x`: prefix count whose entries are within `ilinit`.
    pub initial_l_prefix_len: usize,
    /// FEFF `mmaxp1`, computed after capacity truncation and ordering.
    pub max_m_plus_one: usize,
    /// FEFF final `nmax`, computed after capacity truncation and ordering.
    pub max_n: usize,
    /// FEFF `iord`, the requested Rehr-Albers order.
    pub order: i32,
    /// Requested `nmax` before lambda-capacity truncation.
    pub requested_n_max: usize,
    /// Requested `mmax` before lambda-capacity truncation.
    pub requested_m_max: usize,
    /// Whether FEFF would have logged `Lambda array filled, some order lost`.
    pub truncated: bool,
}

/// Error returned by FEFF `GENFMT` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GenfmtError {
    /// FEFF only defines nonnegative `icalc` values through `10`.
    #[error("undefined FEFF lambda calculation {calculation}")]
    UndefinedLambdaCalculation { calculation: i32 },
    /// A negative `icalc` could not be decoded safely.
    #[error("lambda calculation code {calculation} cannot be decoded safely")]
    LambdaCodeOverflow { calculation: i32 },
    /// The cute heuristic needs finite beta angles.
    #[error("beta angle at index {index} must be finite, got {value}")]
    NonFiniteBetaAngle { index: usize, value: Real },
    /// A generated FEFF integer field would overflow.
    #[error("lambda field {field}={value} does not fit in i32")]
    IntegerOverflow { field: &'static str, value: usize },
    /// Generated lambda indices exceed the caller's FEFF dimensions.
    #[error(
        "lambda selection exceeded dimensions: mmaxp1={max_m_plus_one}, nmax={max_n}, mtot={max_m}, ntot={max_n_limit}"
    )]
    DimensionExceeded {
        max_m_plus_one: usize,
        max_n: usize,
        max_m: usize,
        max_n_limit: usize,
    },
}

/// Build FEFF `mlam` and `nlam` arrays from `GENFMT/setlam.f90` rules.
///
/// The returned arrays preserve FEFF's insertion order, including `-m` before
/// `+m`, and then apply FEFF's second pass that moves entries satisfying
/// `n <= ilinit && abs(m) <= ilinit` to the front to minimize `laml0x`.
/// Capacity handling also follows FEFF: if `lamtot` fills, the result is
/// truncated and flagged instead of failing.
pub fn lambda_indices(input: LambdaIndexInput<'_>) -> Result<LambdaIndexSet, GenfmtError> {
    let request = lambda_request(input)?;
    let mut raw = Vec::with_capacity(input.lambda_capacity.min(128));
    let mut truncated = false;

    if request.order >= 0 {
        let order = usize::try_from(request.order).map_err(|_| GenfmtError::IntegerOverflow {
            field: "iord",
            value: request.order.unsigned_abs() as usize,
        })?;
        let valid_n_max = request.n_max.min(order / 2);

        'outer: for n in 0..=valid_n_max {
            let valid_m_max = request.m_max.min(order - 2 * n);
            for m in 0..=valid_m_max {
                if raw.len() >= input.lambda_capacity {
                    truncated = true;
                    break 'outer;
                }
                raw.push((-checked_i32("m", m)?, checked_i32("n", n)?));

                if m != 0 {
                    if raw.len() >= input.lambda_capacity {
                        truncated = true;
                        break 'outer;
                    }
                    raw.push((checked_i32("m", m)?, checked_i32("n", n)?));
                }
            }
        }
    }

    let mut pairs = Vec::with_capacity(raw.len());
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| within_initial_l(m, n, input.initial_l)),
    );
    let initial_l_prefix_len = pairs.len();
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| !within_initial_l(m, n, input.initial_l)),
    );

    let max_m_plus_one = pairs
        .iter()
        .filter_map(|&(m, _)| usize::try_from(m.saturating_add(1)).ok())
        .max()
        .unwrap_or(0);
    let max_n = pairs
        .iter()
        .filter_map(|&(_, n)| usize::try_from(n).ok())
        .max()
        .unwrap_or(0);

    if max_n > input.max_n || max_m_plus_one > input.max_m.saturating_add(1) {
        return Err(GenfmtError::DimensionExceeded {
            max_m_plus_one,
            max_n,
            max_m: input.max_m,
            max_n_limit: input.max_n,
        });
    }

    let (m_values, n_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok(LambdaIndexSet {
        m_indices: Array1::from_vec(m_values),
        n_indices: Array1::from_vec(n_values),
        initial_l_prefix_len,
        max_m_plus_one,
        max_n,
        order: request.order,
        requested_n_max: request.n_max,
        requested_m_max: request.m_max,
        truncated,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LambdaRequest {
    order: i32,
    n_max: usize,
    m_max: usize,
}

fn lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    if input.calculation < 0 {
        return decode_lambda_request(input.calculation);
    }

    if input.scattering_count == 1 {
        return Ok(LambdaRequest {
            order: checked_order(input.initial_l, input.initial_l)?,
            n_max: input.initial_l,
            m_max: input.initial_l,
        });
    }

    if input.calculation < 10 {
        let order = input.calculation;
        return Ok(LambdaRequest {
            order,
            n_max: usize::try_from(order / 2).map_err(|_| GenfmtError::IntegerOverflow {
                field: "nmax",
                value: order.unsigned_abs() as usize,
            })?,
            m_max: usize::try_from(order).map_err(|_| GenfmtError::IntegerOverflow {
                field: "mmax",
                value: order.unsigned_abs() as usize,
            })?,
        });
    }

    if input.calculation == 10 {
        return cute_lambda_request(input);
    }

    Err(GenfmtError::UndefinedLambdaCalculation {
        calculation: input.calculation,
    })
}

fn decode_lambda_request(calculation: i32) -> Result<LambdaRequest, GenfmtError> {
    let code = calculation
        .checked_neg()
        .ok_or(GenfmtError::LambdaCodeOverflow { calculation })?;
    let order = (code / 10_000) - 1;
    Ok(LambdaRequest {
        order,
        n_max: usize::try_from(code % 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
        m_max: usize::try_from((code % 10_000) / 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
    })
}

fn cute_lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    let mut m_max = input.initial_l;
    for (index, &angle) in input.beta_angles.iter().enumerate() {
        if !angle.is_finite() {
            return Err(GenfmtError::NonFiniteBetaAngle {
                index,
                value: angle,
            });
        }
        let magnitude = angle.abs();
        let pi_distance = (magnitude - std::f64::consts::PI).abs();
        if magnitude > ONE_DEGREE_RADIANS && pi_distance > ONE_DEGREE_RADIANS {
            m_max = 3;
        }
    }

    let n_max = if input.energy_index >= 42 {
        9
    } else {
        input.initial_l
    };

    Ok(LambdaRequest {
        order: checked_order(n_max, m_max)?,
        n_max,
        m_max,
    })
}

fn checked_order(n_max: usize, m_max: usize) -> Result<i32, GenfmtError> {
    let order = n_max
        .checked_mul(2)
        .and_then(|value| value.checked_add(m_max))
        .ok_or(GenfmtError::IntegerOverflow {
            field: "iord",
            value: n_max,
        })?;
    checked_i32("iord", order)
}

fn checked_i32(field: &'static str, value: usize) -> Result<i32, GenfmtError> {
    i32::try_from(value).map_err(|_| GenfmtError::IntegerOverflow { field, value })
}

fn within_initial_l(m: i32, n: i32, initial_l: usize) -> bool {
    let abs_m = m.unsigned_abs() as usize;
    let Ok(n) = usize::try_from(n) else {
        return false;
    };
    n <= initial_l && abs_m <= initial_l
}

#[cfg(test)]
mod tests {
    use super::{GenfmtError, LambdaIndexInput, lambda_indices};

    fn input<'a>(
        calculation: i32,
        energy_index: usize,
        scattering_count: usize,
        initial_l: usize,
        beta_angles: &'a [f64],
        lambda_capacity: usize,
    ) -> LambdaIndexInput<'a> {
        LambdaIndexInput {
            calculation,
            energy_index,
            scattering_count,
            initial_l,
            beta_angles,
            lambda_capacity,
            max_m: 10,
            max_n: 10,
        }
    }

    #[test]
    fn exact_order_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, std::f64::consts::PI, 0.5, 2.8];
        let lambda = lambda_indices(input(2, 10, 2, 3, &beta, 40))?;

        assert_eq!(lambda.order, 2);
        assert_eq!(lambda.requested_n_max, 1);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 6);
        assert_eq!(lambda.max_n, 1);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert!(!lambda.truncated);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2, 0]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0, 1]);
        Ok(())
    }

    #[test]
    fn single_scattering_uses_initial_l_exact_reference() -> Result<(), GenfmtError> {
        let beta = [0.3, 1.2];
        let lambda = lambda_indices(input(10, 8, 1, 2, &beta, 40))?;

        assert_eq!(lambda.order, 6);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 2);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        );
        Ok(())
    }

    #[test]
    fn cute_linear_low_energy_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [
            0.0,
            std::f64::consts::PI,
            0.010,
            std::f64::consts::PI - 0.010,
        ];
        let lambda = lambda_indices(input(10, 41, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 12);
        assert_eq!(lambda.requested_n_max, 4);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 45);
        assert_eq!(lambda.max_n, 4);
        assert_eq!(lambda.max_m_plus_one, 5);
        assert_eq!(lambda.m_indices.len(), 45);
        assert_eq!(
            &lambda.m_indices.to_vec()[..9],
            &[0, -1, 1, -2, 2, -3, 3, -4, 4]
        );
        assert_eq!(
            &lambda.n_indices.to_vec()[36..],
            &[4, 4, 4, 4, 4, 4, 4, 4, 4]
        );
        Ok(())
    }

    #[test]
    fn cute_nonlinear_high_energy_sorts_initial_l_prefix() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.25, std::f64::consts::PI];
        let lambda = lambda_indices(input(10, 42, 2, 4, &beta, 80))?;

        assert_eq!(lambda.order, 21);
        assert_eq!(lambda.requested_n_max, 9);
        assert_eq!(lambda.requested_m_max, 3);
        assert_eq!(lambda.m_indices.len(), 70);
        assert_eq!(lambda.initial_l_prefix_len, 35);
        assert_eq!(lambda.max_n, 9);
        assert_eq!(lambda.max_m_plus_one, 4);
        assert_eq!(&lambda.n_indices.to_vec()[..7], &[0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&lambda.n_indices.to_vec()[28..35], &[4, 4, 4, 4, 4, 4, 4]);
        assert_eq!(&lambda.n_indices.to_vec()[35..42], &[5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(&lambda.n_indices.to_vec()[63..], &[9, 9, 9, 9, 9, 9, 9]);
        Ok(())
    }

    #[test]
    fn negative_calculation_decodes_requested_limits() -> Result<(), GenfmtError> {
        let beta = [0.0, 0.5];
        let lambda = lambda_indices(input(-80_205, 12, 2, 2, &beta, 80))?;

        assert_eq!(lambda.order, 7);
        assert_eq!(lambda.requested_n_max, 5);
        assert_eq!(lambda.requested_m_max, 2);
        assert_eq!(lambda.initial_l_prefix_len, 15);
        assert_eq!(lambda.max_n, 3);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(
            lambda.m_indices.to_vec(),
            vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1]
        );
        assert_eq!(
            lambda.n_indices.to_vec(),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        );
        Ok(())
    }

    #[test]
    fn capacity_truncation_matches_feff_reference() -> Result<(), GenfmtError> {
        let beta = [0.0, 1.0];
        let lambda = lambda_indices(input(4, 10, 2, 1, &beta, 5))?;

        assert!(lambda.truncated);
        assert_eq!(lambda.order, 4);
        assert_eq!(lambda.requested_n_max, 2);
        assert_eq!(lambda.requested_m_max, 4);
        assert_eq!(lambda.initial_l_prefix_len, 3);
        assert_eq!(lambda.max_n, 0);
        assert_eq!(lambda.max_m_plus_one, 3);
        assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2]);
        assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn cute_calculation_rejects_nonfinite_beta() {
        let beta = [f64::NAN];

        assert!(matches!(
            lambda_indices(input(10, 42, 2, 4, &beta, 80)),
            Err(GenfmtError::NonFiniteBetaAngle { index: 0, .. })
        ));
    }

    #[test]
    fn undefined_calculation_is_an_error_for_multiple_scattering() {
        assert_eq!(
            lambda_indices(input(11, 1, 2, 0, &[], 10)),
            Err(GenfmtError::UndefinedLambdaCalculation { calculation: 11 })
        );
    }

    #[test]
    fn dimension_overflow_is_reported() {
        let mut bad = input(10, 42, 2, 4, &[0.25], 80);
        bad.max_n = 8;

        assert!(matches!(
            lambda_indices(bad),
            Err(GenfmtError::DimensionExceeded {
                max_n: 9,
                max_n_limit: 8,
                ..
            })
        ));
    }
}
