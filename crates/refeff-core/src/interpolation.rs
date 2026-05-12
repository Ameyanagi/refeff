//! FEFF polynomial and linear interpolation helpers.
//!
//! This module ports the small interpolation routines from `MATH/terp.f90`,
//! `MATH/terpc.f90`, `MATH/polint.f90`, and `MATH/lint.f90`. FEFF chooses a
//! local window with `locat`, then evaluates an order-`m` polynomial with the
//! Numerical Recipes `polint` recurrence. The Rust API preserves that behavior
//! while returning structured errors instead of terminating the process.

use std::ops::{Add, Div, Mul, Sub};

use thiserror::Error;

use crate::{Complex, Real};

const MAX_POLYNOMIAL_ORDER: usize = 3;
const MAX_POLYNOMIAL_POINTS: usize = MAX_POLYNOMIAL_ORDER + 1;

/// Value returned by polynomial interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interpolation<T> {
    /// Interpolated or extrapolated value at the requested abscissa.
    pub value: T,
    /// Last correction term from the `polint` recurrence.
    ///
    /// FEFF carries this as `dy`; it is a local estimate of interpolation error,
    /// not a rigorous bound.
    pub error_estimate: T,
}

/// Reusable search state for FEFF `lint` linear interpolation.
///
/// FEFF carries `flag`, `klo`, and `khi` between monotonic calls. This cache
/// stores the same adjacent interval with zero-based Rust indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LintCache {
    lower: usize,
    upper: usize,
    needs_search: bool,
}

impl Default for LintCache {
    fn default() -> Self {
        Self {
            lower: 0,
            upper: 1,
            needs_search: true,
        }
    }
}

impl LintCache {
    /// Return a cache that performs a fresh binary search on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the cache so the next interpolation performs a fresh search.
    pub fn reset(&mut self) {
        self.needs_search = true;
    }

    /// Return FEFF-style one-based interval bounds if a search has occurred.
    #[must_use]
    pub fn fortran_bounds(self) -> Option<(usize, usize)> {
        (!self.needs_search).then_some((self.lower + 1, self.upper + 1))
    }
}

/// Error returned by FEFF interpolation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InterpolationError {
    /// At least one point is required for interpolation.
    #[error("interpolation input is empty")]
    EmptyInput,
    /// Abscissa and ordinate arrays must have the same length.
    #[error("x/y length mismatch: x has {x_len} values but y has {y_len}")]
    LengthMismatch { x_len: usize, y_len: usize },
    /// FEFF's `terp` and `terpc` routines support only cubic or lower order.
    #[error("polynomial order {order} exceeds FEFF maximum {max_order}")]
    OrderTooLarge { order: usize, max_order: usize },
    /// There are not enough points to form the requested interpolation order.
    #[error("order {order} requires {required} points but only {available} are available")]
    InsufficientPoints {
        order: usize,
        required: usize,
        available: usize,
    },
    /// Direct `polint` accepts at most four points, matching FEFF `nmax=4`.
    #[error("polynomial interpolation received {points} points; maximum is {max_points}")]
    TooManyPoints { points: usize, max_points: usize },
    /// Duplicate abscissae make the interpolation denominator zero.
    #[error("duplicate interpolation abscissae at window positions {left} and {right}")]
    DuplicateAbscissa { left: usize, right: usize },
}

/// Return FEFF's `locat` result: the number of grid points `<= x`.
///
/// For a monotonic increasing grid this is the 1-based index of the grid point
/// immediately below `x`, with `0` below the first point and `xs.len()` at or
/// above the last point. This intentionally mirrors FEFF's boundary behavior.
#[must_use]
pub fn locate_below(x: Real, xs: &[Real]) -> usize {
    let mut lower = 0;
    let mut upper = xs.len() + 1;

    while upper - lower > 1 {
        let middle = (upper + lower) / 2;
        let middle_value = xs[middle - 1];
        if x < middle_value {
            upper = middle;
        } else {
            lower = middle;
        }
    }

    lower
}

/// Interpolate real values with FEFF's `polint` recurrence.
///
/// `xs` and `ys` must contain one to four values. Use [`terp`] when FEFF's
/// local window selection from a larger table is desired.
pub fn polynomial_interpolate(
    xs: &[Real],
    ys: &[Real],
    x: Real,
) -> Result<Interpolation<Real>, InterpolationError> {
    polynomial_interpolate_values(xs, ys, x)
}

/// Interpolate complex values with FEFF's `polinc` recurrence.
///
/// `xs` and `ys` must contain one to four values. Use [`terpc`] when FEFF's
/// local window selection from a larger table is desired.
pub fn polynomial_interpolate_complex(
    xs: &[Real],
    ys: &[Complex],
    x: Real,
) -> Result<Interpolation<Complex>, InterpolationError> {
    polynomial_interpolate_values(xs, ys, x)
}

/// FEFF-compatible real interpolation/extrapolation by an order-`m` polynomial.
///
/// This ports `terp`: locate the point immediately below `x`, select the same
/// local window FEFF would use, then evaluate `polint` on `order + 1` points.
pub fn terp(
    xs: &[Real],
    ys: &[Real],
    order: usize,
    x: Real,
) -> Result<Interpolation<Real>, InterpolationError> {
    interpolate_window(xs, ys, order, x)
}

/// FEFF-compatible complex interpolation/extrapolation by an order-`m` polynomial.
///
/// This ports `terpc` and uses the same window selection as [`terp`].
pub fn terpc(
    xs: &[Real],
    ys: &[Complex],
    order: usize,
    x: Real,
) -> Result<Interpolation<Complex>, InterpolationError> {
    interpolate_window(xs, ys, order, x)
}

/// FEFF `terp1`-style clamped linear interpolation for real arrays.
///
/// Requests outside the table use the first or last interval for extrapolation,
/// matching the `i = max(i, 1); i = min(i, n - 1)` logic in FEFF.
pub fn terp1(xs: &[Real], ys: &[Real], x: Real) -> Result<Real, InterpolationError> {
    ensure_matching_nonempty(xs, ys)?;
    if xs.len() < 2 {
        return Err(InterpolationError::InsufficientPoints {
            order: 1,
            required: 2,
            available: xs.len(),
        });
    }

    let located = locate_below(x, xs);
    let lower = located.saturating_sub(1).min(xs.len() - 2);
    let upper = lower + 1;
    let x_lower = xs[lower];
    let x_upper = xs[upper];
    let denominator = x_upper - x_lower;
    if denominator == 0.0 {
        return Err(InterpolationError::DuplicateAbscissa {
            left: lower,
            right: upper,
        });
    }

    Ok(ys[lower] + (x - x_lower) * (ys[upper] - ys[lower]) / denominator)
}

/// FEFF `lint` linear interpolation with a fresh interval search.
///
/// Unlike [`terp1`], values at or below the first abscissa return exactly zero.
/// Values above the final abscissa extrapolate from the last interval, matching
/// FEFF's defined binary-search path.
pub fn lint(xs: &[Real], ys: &[Real], x: Real) -> Result<Real, InterpolationError> {
    let mut cache = LintCache::new();
    lint_with_cache(xs, ys, x, &mut cache)
}

/// FEFF `lint` linear interpolation using reusable interval state.
///
/// This mirrors FEFF's `flag/klo/khi` optimization for nondecreasing `x`
/// sequences while guarding against out-of-bounds cache movement.
pub fn lint_with_cache(
    xs: &[Real],
    ys: &[Real],
    x: Real,
    cache: &mut LintCache,
) -> Result<Real, InterpolationError> {
    ensure_matching_nonempty(xs, ys)?;
    if xs.len() < 2 {
        return Err(InterpolationError::InsufficientPoints {
            order: 1,
            required: 2,
            available: xs.len(),
        });
    }

    if x <= xs[0] {
        return Ok(0.0);
    }

    if cache.needs_search || cache.upper >= xs.len() || x < xs[cache.lower] {
        cache.lower = 0;
        cache.upper = xs.len() - 1;
        while cache.upper - cache.lower > 1 {
            let middle = (cache.upper + cache.lower) / 2;
            if xs[middle] > x {
                cache.upper = middle;
            } else {
                cache.lower = middle;
            }
        }
        cache.needs_search = false;
    } else {
        while xs[cache.upper] < x {
            if cache.upper + 1 >= xs.len() {
                cache.lower = xs.len() - 2;
                cache.upper = xs.len() - 1;
                break;
            }
            cache.lower = cache.upper;
            cache.upper += 1;
        }
    }

    let x_lower = xs[cache.lower];
    let x_upper = xs[cache.upper];
    let denominator = x_upper - x_lower;
    if denominator == 0.0 {
        return Err(InterpolationError::DuplicateAbscissa {
            left: cache.lower,
            right: cache.upper,
        });
    }

    let a = (x_upper - x) / denominator;
    let b = (x - x_lower) / denominator;
    Ok(a * ys[cache.lower] + b * ys[cache.upper])
}

fn interpolate_window<T>(
    xs: &[Real],
    ys: &[T],
    order: usize,
    x: Real,
) -> Result<Interpolation<T>, InterpolationError>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Real, Output = T>
        + Div<Real, Output = T>,
{
    ensure_order(order)?;
    ensure_matching_nonempty(xs, ys)?;

    let points = order + 1;
    if xs.len() < points {
        return Err(InterpolationError::InsufficientPoints {
            order,
            required: points,
            available: xs.len(),
        });
    }

    let located = locate_below(x, xs);
    let first_fortran_index = located
        .saturating_sub(order / 2)
        .max(1)
        .min(xs.len() - order);
    let start = first_fortran_index - 1;
    let end = start + points;
    let x_window = &xs[start..end];
    let y_window = &ys[start..end];

    polynomial_interpolate_values(x_window, y_window, x)
}

fn polynomial_interpolate_values<T>(
    xs: &[Real],
    ys: &[T],
    x: Real,
) -> Result<Interpolation<T>, InterpolationError>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Real, Output = T>
        + Div<Real, Output = T>,
{
    ensure_matching_nonempty(xs, ys)?;
    if xs.len() > MAX_POLYNOMIAL_POINTS {
        return Err(InterpolationError::TooManyPoints {
            points: xs.len(),
            max_points: MAX_POLYNOMIAL_POINTS,
        });
    }

    let first_x = xs[0];
    let mut nearest = 0;
    let mut nearest_distance = (x - first_x).abs();
    let mut c = Vec::with_capacity(xs.len());
    let mut d = Vec::with_capacity(xs.len());

    for (index, (&x_value, &y_value)) in xs.iter().zip(ys.iter()).enumerate() {
        let distance = (x - x_value).abs();
        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
        c.push(y_value);
        d.push(y_value);
    }

    let mut value = ys[nearest];
    let mut error_estimate = T::default();
    let mut nearest_offset = nearest;
    for order in 1..xs.len() {
        for index in 0..(xs.len() - order) {
            let lower_delta = xs[index] - x;
            let upper_delta = xs[index + order] - x;
            let denominator = lower_delta - upper_delta;
            if denominator == 0.0 {
                return Err(InterpolationError::DuplicateAbscissa {
                    left: index,
                    right: index + order,
                });
            }

            let divided_difference = (c[index + 1] - d[index]) / denominator;
            d[index] = divided_difference * upper_delta;
            c[index] = divided_difference * lower_delta;
        }

        if 2 * nearest_offset < xs.len() - order {
            error_estimate = c[nearest_offset];
        } else {
            error_estimate = d[nearest_offset - 1];
            nearest_offset -= 1;
        }
        value = value + error_estimate;
    }

    Ok(Interpolation {
        value,
        error_estimate,
    })
}

fn ensure_order(order: usize) -> Result<(), InterpolationError> {
    if order > MAX_POLYNOMIAL_ORDER {
        return Err(InterpolationError::OrderTooLarge {
            order,
            max_order: MAX_POLYNOMIAL_ORDER,
        });
    }
    Ok(())
}

fn ensure_matching_nonempty<T>(xs: &[Real], ys: &[T]) -> Result<(), InterpolationError> {
    if xs.len() != ys.len() {
        return Err(InterpolationError::LengthMismatch {
            x_len: xs.len(),
            y_len: ys.len(),
        });
    }
    if xs.is_empty() {
        return Err(InterpolationError::EmptyInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: Real, right: Real) {
        assert!(
            (left - right).abs() < 1.0e-12,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    fn assert_complex_close(left: Complex, right: Complex) {
        assert_close(left.re, right.re);
        assert_close(left.im, right.im);
    }

    #[test]
    fn locate_below_matches_feff_boundary_semantics() {
        let xs = [1.0, 2.0, 4.0];

        assert_eq!(locate_below(0.5, &xs), 0);
        assert_eq!(locate_below(1.0, &xs), 1);
        assert_eq!(locate_below(3.0, &xs), 2);
        assert_eq!(locate_below(4.0, &xs), 3);
        assert_eq!(locate_below(5.0, &xs), 3);
    }

    #[test]
    fn polynomial_interpolate_reproduces_quadratic() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 4.0, 9.0];

        let interpolated = polynomial_interpolate(&xs, &ys, 1.5)?;

        assert_close(interpolated.value, 6.25);
        assert_close(interpolated.error_estimate, 0.75);
        Ok(())
    }

    #[test]
    fn polynomial_interpolate_reproduces_cubic_extrapolation() -> Result<(), InterpolationError> {
        let xs = [-1.0, 0.0, 1.0, 2.0];
        let ys = xs.map(|x| x * x * x - 2.0 * x + 1.0);

        let interpolated = polynomial_interpolate(&xs, &ys, 3.0)?;

        assert_close(interpolated.value, 22.0);
        Ok(())
    }

    #[test]
    fn terp_uses_feff_window_selection() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = xs.map(|x| x * x * x);

        let interpolated = terp(&xs, &ys, 3, 2.5)?;

        assert_close(interpolated.value, 15.625);
        Ok(())
    }

    #[test]
    fn terp_extrapolates_from_clamped_edge_window() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = xs.map(|x| 2.0 * x + 1.0);

        let low = terp(&xs, &ys, 1, -1.0)?;
        let high = terp(&xs, &ys, 1, 4.0)?;

        assert_close(low.value, -1.0);
        assert_close(high.value, 9.0);
        Ok(())
    }

    #[test]
    fn terpc_interpolates_real_and_imaginary_parts() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.0];
        let ys = [
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 3.0),
            Complex::new(4.0, 5.0),
        ];

        let interpolated = terpc(&xs, &ys, 2, 1.5)?;

        assert_complex_close(interpolated.value, Complex::new(2.25, 4.0));
        Ok(())
    }

    #[test]
    fn duplicate_abscissae_return_error() {
        let xs = [0.0, 1.0, 1.0];
        let ys = [0.0, 1.0, 2.0];

        assert!(matches!(
            polynomial_interpolate(&xs, &ys, 1.0),
            Err(InterpolationError::DuplicateAbscissa { .. })
        ));
    }

    #[test]
    fn terp_rejects_orders_above_feff_limit() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = xs;

        assert!(matches!(
            terp(&xs, &ys, 4, 2.0),
            Err(InterpolationError::OrderTooLarge { order: 4, .. })
        ));
    }

    #[test]
    fn terp1_clamps_to_boundary_intervals() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 5.0];

        assert_close(terp1(&xs, &ys, -1.0)?, -1.0);
        assert_close(terp1(&xs, &ys, 3.0)?, 7.0);
        Ok(())
    }

    #[test]
    fn lint_matches_feff_reference_and_cache_state() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.5, 4.0, 7.0];
        let ys = [0.0, 2.0, 1.0, 5.0, 11.0];
        let mut cache = LintCache::new();

        assert_close(lint_with_cache(&xs, &ys, -1.0, &mut cache)?, 0.0);
        assert_eq!(cache.fortran_bounds(), None);
        assert_close(lint_with_cache(&xs, &ys, 0.0, &mut cache)?, 0.0);
        assert_eq!(cache.fortran_bounds(), None);
        assert_close(lint_with_cache(&xs, &ys, 0.5, &mut cache)?, 1.0);
        assert_eq!(cache.fortran_bounds(), Some((1, 2)));
        assert_close(lint_with_cache(&xs, &ys, 1.75, &mut cache)?, 1.5);
        assert_eq!(cache.fortran_bounds(), Some((2, 3)));
        assert_close(lint_with_cache(&xs, &ys, 6.0, &mut cache)?, 9.0);
        assert_eq!(cache.fortran_bounds(), Some((4, 5)));

        cache.reset();
        assert_close(lint_with_cache(&xs, &ys, 8.0, &mut cache)?, 13.0);
        assert_eq!(cache.fortran_bounds(), Some((4, 5)));
        Ok(())
    }

    #[test]
    fn lint_stateless_matches_feff_boundary_semantics() -> Result<(), InterpolationError> {
        let xs = [0.0, 1.0, 2.5, 4.0, 7.0];
        let ys = [0.0, 2.0, 1.0, 5.0, 11.0];

        assert_close(lint(&xs, &ys, -1.0)?, 0.0);
        assert_close(lint(&xs, &ys, 0.5)?, 1.0);
        assert_close(lint(&xs, &ys, 8.0)?, 13.0);
        Ok(())
    }

    #[test]
    fn lint_rejects_duplicate_interval() {
        let xs = [0.0, 1.0, 1.0];
        let ys = [0.0, 1.0, 2.0];

        assert!(matches!(
            lint(&xs, &ys, 1.5),
            Err(InterpolationError::DuplicateAbscissa { left: 1, right: 2 })
        ));
    }
}
