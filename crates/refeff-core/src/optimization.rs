//! FEFF one-dimensional table minimization helpers.
//!
//! This module ports the table-backed BAND routines `mnbrak.f90` and
//! `brent.f90`. FEFF stores the objective as `fitx`/`fity` module state and
//! evaluates it through cubic `terp`; the Rust API takes slices explicitly and
//! returns structured errors instead of pausing the process.

use thiserror::Error;

use crate::{InterpolationError, Real, terp};

const MINIMUM_BRACKET_GOLD: Real = 1.618_034;
const MINIMUM_BRACKET_G_LIMIT: Real = 100.0;
const MINIMUM_BRACKET_TINY: Real = 1.0e-20;
const MINIMUM_BRACKET_MAX_ITERATIONS: usize = 256;
const BRENT_MAX_ITERATIONS: usize = 100;
const BRENT_CGOLD: Real = 0.381_966_f32 as Real;
const BRENT_ZEPS: Real = 1.0e-10_f32 as Real;

/// Bracketing triplet returned by FEFF `mnbrak`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumBracket {
    /// First abscissa.
    pub ax: Real,
    /// Middle abscissa, with the lowest sampled value.
    pub bx: Real,
    /// Third abscissa.
    pub cx: Real,
    /// Objective value at [`Self::ax`].
    pub fa: Real,
    /// Objective value at [`Self::bx`].
    pub fb: Real,
    /// Objective value at [`Self::cx`].
    pub fc: Real,
}

/// Minimum isolated by FEFF `brent`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableMinimum {
    /// Abscissa of the isolated minimum.
    pub x: Real,
    /// Interpolated objective value at [`Self::x`].
    pub value: Real,
    /// Number of Brent iterations performed before convergence.
    pub iterations: usize,
}

/// Error returned by one-dimensional FEFF minimization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum OptimizationError {
    /// Abscissae and tolerances must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// FEFF `mnbrak` requires distinct starting abscissae.
    #[error("mnbrak starting abscissae must be distinct")]
    CoincidentInitialPoints,
    /// FEFF `brent` requires a positive finite tolerance.
    #[error("brent tolerance must be positive and finite, got {tolerance}")]
    InvalidTolerance { tolerance: Real },
    /// A table-interpolated objective value must be finite.
    #[error("objective value at x={x} must be finite, got {value}")]
    NonFiniteObjective { x: Real, value: Real },
    /// The Rust port guards FEFF's unbounded bracketing loop.
    #[error("mnbrak did not bracket a minimum after {iterations} iterations")]
    BracketDidNotConverge { iterations: usize },
    /// FEFF `brent` exceeded its iteration limit.
    #[error("brent did not converge after {iterations} iterations")]
    BrentDidNotConverge { iterations: usize },
    /// FEFF table interpolation failed while evaluating the objective.
    #[error(transparent)]
    Interpolation(#[from] InterpolationError),
}

/// Port of FEFF `BAND/mnbrak.f90` for a `terp`-interpolated table.
///
/// `first` and `second` are the two initial abscissae. The routine evaluates
/// the downhill direction and returns an enclosing triplet suitable for
/// [`brent_table_minimum`].
pub fn bracket_table_minimum(
    xs: &[Real],
    ys: &[Real],
    order: usize,
    first: Real,
    second: Real,
) -> Result<MinimumBracket, OptimizationError> {
    validate_finite("first", first)?;
    validate_finite("second", second)?;
    if first == second {
        return Err(OptimizationError::CoincidentInitialPoints);
    }

    let mut ax = first;
    let mut bx = second;
    let mut fa = table_value(xs, ys, order, ax)?;
    let mut fb = table_value(xs, ys, order, bx)?;
    if fb > fa {
        std::mem::swap(&mut ax, &mut bx);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut cx = bx + MINIMUM_BRACKET_GOLD * (bx - ax);
    let mut fc = table_value(xs, ys, order, cx)?;
    for _ in 0..MINIMUM_BRACKET_MAX_ITERATIONS {
        if fb < fc {
            return Ok(MinimumBracket {
                ax,
                bx,
                cx,
                fa,
                fb,
                fc,
            });
        }

        let r = (bx - ax) * (fb - fc);
        let q = (bx - cx) * (fb - fa);
        let denominator = 2.0 * feff_sign((q - r).abs().max(MINIMUM_BRACKET_TINY), q - r);
        let mut u = bx - ((bx - cx) * q - (bx - ax) * r) / denominator;
        let ulim = bx + MINIMUM_BRACKET_G_LIMIT * (cx - bx);
        let fu;

        if (bx - u) * (u - cx) > 0.0 {
            let candidate = table_value(xs, ys, order, u)?;
            if candidate < fc {
                return Ok(MinimumBracket {
                    ax: bx,
                    bx: u,
                    cx,
                    fa: fb,
                    fb: candidate,
                    fc,
                });
            }
            if candidate > fb {
                return Ok(MinimumBracket {
                    ax,
                    bx,
                    cx: u,
                    fa,
                    fb,
                    fc: candidate,
                });
            }
            u = cx + MINIMUM_BRACKET_GOLD * (cx - bx);
            fu = table_value(xs, ys, order, u)?;
        } else if (cx - u) * (u - ulim) > 0.0 {
            let candidate = table_value(xs, ys, order, u)?;
            if candidate < fc {
                bx = cx;
                cx = u;
                u = cx + MINIMUM_BRACKET_GOLD * (cx - bx);
                fb = fc;
                fc = candidate;
                fu = table_value(xs, ys, order, u)?;
            } else {
                fu = candidate;
            }
        } else if (u - ulim) * (ulim - cx) >= 0.0 {
            u = ulim;
            fu = table_value(xs, ys, order, u)?;
        } else {
            u = cx + MINIMUM_BRACKET_GOLD * (cx - bx);
            fu = table_value(xs, ys, order, u)?;
        }

        ax = bx;
        bx = cx;
        cx = u;
        fa = fb;
        fb = fc;
        fc = fu;
    }

    Err(OptimizationError::BracketDidNotConverge {
        iterations: MINIMUM_BRACKET_MAX_ITERATIONS,
    })
}

/// Port of FEFF `BAND/brent.f90` for a `terp`-interpolated table.
///
/// `bracket` should normally come from [`bracket_table_minimum`]. The result
/// preserves FEFF's single-precision `CGOLD` and `ZEPS` constants.
pub fn brent_table_minimum(
    xs: &[Real],
    ys: &[Real],
    order: usize,
    bracket: MinimumBracket,
    tolerance: Real,
) -> Result<TableMinimum, OptimizationError> {
    validate_brent_inputs(bracket, tolerance)?;

    let mut a = bracket.ax.min(bracket.cx);
    let mut b = bracket.ax.max(bracket.cx);
    let mut v = bracket.bx;
    let mut w = v;
    let mut x = v;
    let mut e: Real = 0.0;
    let mut d: Real = 0.0;
    let mut fx = table_value(xs, ys, order, x)?;
    let mut fv = fx;
    let mut fw = fx;

    for iteration in 1..=BRENT_MAX_ITERATIONS {
        let xm = 0.5 * (a + b);
        let tol1 = tolerance * x.abs() + BRENT_ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(TableMinimum {
                x,
                value: fx,
                iterations: iteration - 1,
            });
        }

        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = feff_sign(tol1, xm - x);
                }
            } else {
                if x >= xm {
                    e = a - x;
                } else {
                    e = b - x;
                }
                d = BRENT_CGOLD * e;
            }
        } else {
            if x >= xm {
                e = a - x;
            } else {
                e = b - x;
            }
            d = BRENT_CGOLD * e;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + feff_sign(tol1, d)
        };
        let fu = table_value(xs, ys, order, u)?;
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    Err(OptimizationError::BrentDidNotConverge {
        iterations: BRENT_MAX_ITERATIONS,
    })
}

fn validate_brent_inputs(
    bracket: MinimumBracket,
    tolerance: Real,
) -> Result<(), OptimizationError> {
    validate_finite("ax", bracket.ax)?;
    validate_finite("bx", bracket.bx)?;
    validate_finite("cx", bracket.cx)?;
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(OptimizationError::InvalidTolerance { tolerance });
    }
    Ok(())
}

fn table_value(xs: &[Real], ys: &[Real], order: usize, x: Real) -> Result<Real, OptimizationError> {
    validate_finite("x", x)?;
    let value = terp(xs, ys, order, x)?.value;
    if !value.is_finite() {
        return Err(OptimizationError::NonFiniteObjective { x, value });
    }
    Ok(value)
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), OptimizationError> {
    if !value.is_finite() {
        return Err(OptimizationError::NonFiniteInput { name, value });
    }
    Ok(())
}

fn feff_sign(magnitude: Real, sign: Real) -> Real {
    if sign.is_sign_negative() {
        -magnitude.abs()
    } else {
        magnitude.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn minimization_fixture() -> (Vec<Real>, Vec<Real>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for index in 1..=13 {
            let x = -1.0 + 0.5 * (index as Real - 1.0);
            xs.push(x);
            ys.push((x - 2.15).powi(2) + 0.02 * (x - 2.15).powi(4) + 0.1);
        }
        (xs, ys)
    }

    #[test]
    fn mnbrak_and_brent_match_feff_reference() -> Result<(), OptimizationError> {
        let (xs, ys) = minimization_fixture();
        let bracket = bracket_table_minimum(&xs, &ys, 3, 0.0, 0.75)?;

        assert_close(bracket.ax, 2.0402909982490796);
        assert_close(bracket.bx, 2.1524292715796376);
        assert_close(bracket.cx, 2.1645001844430305);
        assert_close(bracket.fa, 0.11184687966719077);
        assert_close(bracket.fb, 0.0994199643232456);
        assert_close(bracket.fc, 0.09959744017790287);

        let minimum = brent_table_minimum(&xs, &ys, 3, bracket, 1.0e-5)?;
        assert_close(minimum.x, 2.1511986762610844);
        assert_close(minimum.value, 0.09941842727812539);
        Ok(())
    }

    #[test]
    fn mnbrak_swaps_to_downhill_direction_like_feff() -> Result<(), OptimizationError> {
        let (xs, ys) = minimization_fixture();
        let bracket = bracket_table_minimum(&xs, &ys, 3, 4.75, 3.75)?;

        assert_close(bracket.ax, 3.75);
        assert_close(bracket.bx, 2.1319660000000002);
        assert_close(bracket.cx, -0.48606802515599945);
        assert_close(bracket.fa, 2.790368875);
        assert_close(bracket.fb, 0.0997923705593791);
        assert_close(bracket.fc, 8.014517611486413);

        let minimum = brent_table_minimum(&xs, &ys, 3, bracket, 1.0e-6)?;
        assert_close(minimum.x, 2.1511963582519695);
        assert_close(minimum.value, 0.09941842727318148);
        Ok(())
    }

    #[test]
    fn minimization_rejects_invalid_inputs() {
        let (xs, ys) = minimization_fixture();

        assert!(matches!(
            bracket_table_minimum(&xs, &ys, 3, 1.0, 1.0),
            Err(OptimizationError::CoincidentInitialPoints)
        ));
        assert!(matches!(
            bracket_table_minimum(&xs, &ys, 3, Real::NAN, 1.0),
            Err(OptimizationError::NonFiniteInput { name: "first", .. })
        ));
        assert!(matches!(
            brent_table_minimum(
                &xs,
                &ys,
                3,
                MinimumBracket {
                    ax: 0.0,
                    bx: 1.0,
                    cx: 2.0,
                    fa: 0.0,
                    fb: 0.0,
                    fc: 0.0,
                },
                0.0,
            ),
            Err(OptimizationError::InvalidTolerance { .. })
        ));
    }
}
