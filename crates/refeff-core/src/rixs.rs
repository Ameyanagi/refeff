//! FEFF RIXS numerical helpers.
//!
//! This module ports the small analytic and interpolation kernels from
//! `RIXS/kkint.f90`, `RIXS/doublelorentz.f90`, and `RIXS/blinterp2d.f90`. The Rust API uses
//! `ndarray` views for table inputs and reports structured errors instead of
//! terminating the process with Fortran `STOP`.

use ndarray::{ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Complex, Real};

const BL_INTERP_TOLERANCE: Real = 1.0e-5;
const KKINT_PI: Real = 3_141_592_653.0 / 1_000_000_000.0;

/// Error returned by FEFF RIXS helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RixsError {
    /// Scalar inputs must be finite real values.
    #[error("RIXS input {name} must be finite, got {value}")]
    NonFiniteReal { name: &'static str, value: Real },
    /// Complex table entries must have finite real and imaginary parts.
    #[error("RIXS complex input {name} must be finite, got ({real}, {imaginary})")]
    NonFiniteComplex {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// Lorentzian widths must be positive and finite.
    #[error("RIXS width {name} must be positive and finite, got {value}")]
    InvalidWidth { name: &'static str, value: Real },
    /// Analytic integration bounds must be finite and increasing.
    #[error("RIXS integration interval must increase: lower={lower}, upper={upper}")]
    InvalidInterval { lower: Real, upper: Real },
    /// Bilinear interpolation needs at least two points on each axis.
    #[error("RIXS interpolation axis {axis} requires at least 2 points, got {len}")]
    InsufficientGrid { axis: &'static str, len: usize },
    /// The interpolation value table must cover the requested x/y grids.
    #[error(
        "RIXS interpolation table has shape ({rows}, {cols}) but needs at least ({required_rows}, {required_cols})"
    )]
    MatrixTooSmall {
        rows: usize,
        cols: usize,
        required_rows: usize,
        required_cols: usize,
    },
    /// FEFF assumes strictly increasing interpolation grids.
    #[error(
        "RIXS interpolation axis {axis} must increase at index {index}: previous={previous}, current={current}"
    )]
    NonIncreasingGrid {
        axis: &'static str,
        index: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF `BLInterp2D` allows only a small tolerance outside the table.
    #[error(
        "RIXS interpolation coordinate {axis}={value} is outside [{min}, {max}] with tolerance {tolerance}"
    )]
    OutOfRange {
        axis: &'static str,
        value: Real,
        min: Real,
        max: Real,
        tolerance: Real,
    },
    /// Duplicate adjacent grid points make the bilinear denominator zero.
    #[error("RIXS interpolation axis {axis} has a zero-width interval at index {index}")]
    ZeroInterval { axis: &'static str, index: usize },
}

/// Port of FEFF `KKInt`.
///
/// This evaluates the analytic integral of `(slope * x' + intercept) /
/// (x' - x + i * width)` from `x0` to `x1`. FEFF uses separate expressions
/// for interior/off-interval points and the two exact endpoint cases; this
/// function preserves those branches.
pub fn kk_integral(
    slope: Complex,
    intercept: Complex,
    x0: Real,
    x1: Real,
    width: Real,
    x: Real,
) -> Result<Complex, RixsError> {
    validate_complex("a", slope)?;
    validate_complex("b", intercept)?;
    validate_finite("x0", x0)?;
    validate_finite("x1", x1)?;
    validate_width("gam", width)?;
    validate_finite("x", x)?;
    if x0 >= x1 {
        return Err(RixsError::InvalidInterval {
            lower: x0,
            upper: x1,
        });
    }

    let i = Complex::new(0.0, 1.0);
    let width_at_x = Complex::new(width, x);
    if x != x0 && x != x1 {
        let left = x - x0;
        let right = x - x1;
        let log_ratio = ((width * width + left * left) / (width * width + right * right)).ln();
        let bracket = Complex::new(
            (width / left).atan() - (width / right).atan() - KKINT_PI,
            0.5 * log_ratio,
        );
        let mut value = slope * (Complex::new(x1 - x0, 0.0) + width_at_x * bracket);
        if x < x0 || x > x1 {
            value += slope * KKINT_PI * width_at_x;
        }
        Ok(value + intercept * (Complex::new(x1 - x, width) / Complex::new(x0 - x, width)).ln())
    } else if x == x0 {
        let span = x1 - x0;
        let bracket = Complex::new(
            (width / (x0 - x1)).atan() + 0.5 * KKINT_PI,
            0.5 * ((width * width + span * span) / (width * width)).ln(),
        );
        Ok(
            slope * (Complex::new(span, 0.0) - Complex::new(width, x0) * bracket)
                + intercept
                    * (Complex::new(width, 0.0) / (Complex::new(width, 0.0) - i * span)).ln(),
        )
    } else {
        let span = x1 - x0;
        let bracket = Complex::new(
            (width / (x0 - x1)).atan() + 0.5 * KKINT_PI,
            0.5 * ((width * width) / (width * width + span * span)).ln(),
        );
        Ok(
            slope * (Complex::new(span, 0.0) - Complex::new(width, x1) * bracket)
                + intercept * ((Complex::new(width, 0.0) - i * span) / width).ln(),
        )
    }
}

/// Port of FEFF `IntDoubleLorentz`.
///
/// `omega = Some(value)` corresponds to FEFF `iinf >= 0`, where the analytic
/// antiderivative is evaluated at a finite upper limit. `omega = None`
/// corresponds to FEFF `iinf < 0`, the simplified infinite-limit branch.
pub fn integrated_double_lorentz(
    rem1: Real,
    rem2: Real,
    core_width: Real,
    width: Real,
    intercept: Real,
    slope: Real,
    omega: Option<Real>,
) -> Result<Real, RixsError> {
    validate_finite("rem1", rem1)?;
    validate_finite("rem2", rem2)?;
    validate_width("gamch", core_width)?;
    validate_width("gam", width)?;
    validate_finite("a", intercept)?;
    validate_finite("b", slope)?;

    let delta = rem1 - rem2;
    let value = if let Some(omega) = omega {
        validate_finite("omega", omega)?;
        let gamch2 = core_width * core_width;
        let gam2 = width * width;
        let delta2 = delta * delta;
        let first = 2.0
            * width
            * (intercept * (gam2 - gamch2 + delta2)
                + slope * (gam2 * rem1 + gamch2 * (rem1 - 2.0 * rem2) + rem1 * delta2))
            * ((omega - rem1) / core_width).atan();
        let second = core_width
            * (2.0
                * (intercept * (-gam2 + gamch2 + delta2)
                    + slope * ((gamch2 + delta2) * rem2 + gam2 * (-2.0 * rem1 + rem2)))
                * ((omega - rem2) / width).atan()
                + width
                    * (2.0 * intercept * (-rem1 + rem2)
                        + slope * (gam2 - gamch2 - rem1 * rem1 + rem2 * rem2))
                    * ((gamch2 + (omega - rem1) * (omega - rem1)).ln()
                        - (gam2 + (omega - rem2) * (omega - rem2)).ln()));
        let denominator = 2.0
            * width
            * core_width
            * ((width - core_width) * (width - core_width) + delta2)
            * ((width + core_width) * (width + core_width) + delta2);
        (first + second) / denominator
    } else {
        intercept * (width + core_width) * std::f64::consts::PI
            / (delta * delta + (width + core_width) * (width + core_width))
            / (2.0 * width * core_width)
    };

    Ok(value * width / std::f64::consts::PI)
}

/// Port of FEFF `BLInterp2D`: bilinear interpolation of a complex 2-D table.
///
/// `x` and `y` are strictly increasing coordinate grids. `values` is indexed as
/// `values[(x_index, y_index)]`, matching FEFF `A(ix, iy)`. Coordinates within
/// `1e-5` outside either endpoint use FEFF's endpoint interval and therefore
/// may extrapolate slightly, including FEFF's sentinel-order behavior above the
/// upper endpoint.
pub fn bilinear_interpolate_complex(
    x: ArrayView1<'_, Real>,
    y: ArrayView1<'_, Real>,
    values: ArrayView2<'_, Complex>,
    x0: Real,
    y0: Real,
) -> Result<Complex, RixsError> {
    validate_bilinear_inputs(x, y, values, x0, y0)?;

    let (x_lower, x_upper) = interpolation_interval(x, x0);
    let (y_lower, y_upper) = interpolation_interval(y, y0);
    let dx = x[x_upper] - x[x_lower];
    let dy = y[y_upper] - y[y_lower];
    if dx == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "x",
            index: x_lower,
        });
    }
    if dy == 0.0 {
        return Err(RixsError::ZeroInterval {
            axis: "y",
            index: y_lower,
        });
    }

    let lower_lower = matrix_value(values, x_lower, y_lower)?;
    let upper_lower = matrix_value(values, x_upper, y_lower)?;
    let lower_upper = matrix_value(values, x_lower, y_upper)?;
    let upper_upper = matrix_value(values, x_upper, y_upper)?;
    let dxdy = dx * dy;
    Ok((lower_lower * (x[x_upper] - x0) * (y[y_upper] - y0)
        + upper_lower * (x0 - x[x_lower]) * (y[y_upper] - y0)
        + lower_upper * (x[x_upper] - x0) * (y0 - y[y_lower])
        + upper_upper * (x0 - x[x_lower]) * (y0 - y[y_lower]))
        / dxdy)
}

fn validate_bilinear_inputs(
    x: ArrayView1<'_, Real>,
    y: ArrayView1<'_, Real>,
    values: ArrayView2<'_, Complex>,
    x0: Real,
    y0: Real,
) -> Result<(), RixsError> {
    validate_finite("x0", x0)?;
    validate_finite("y0", y0)?;
    validate_grid("x", x)?;
    validate_grid("y", y)?;
    if values.nrows() < x.len() || values.ncols() < y.len() {
        return Err(RixsError::MatrixTooSmall {
            rows: values.nrows(),
            cols: values.ncols(),
            required_rows: x.len(),
            required_cols: y.len(),
        });
    }
    validate_range("x", x0, x[0], x[x.len() - 1])?;
    validate_range("y", y0, y[0], y[y.len() - 1])?;
    for row in 0..x.len() {
        for col in 0..y.len() {
            validate_complex("values", matrix_value(values, row, col)?)?;
        }
    }
    Ok(())
}

fn validate_grid(axis: &'static str, values: ArrayView1<'_, Real>) -> Result<(), RixsError> {
    if values.len() < 2 {
        return Err(RixsError::InsufficientGrid {
            axis,
            len: values.len(),
        });
    }

    let mut previous = values[0];
    validate_finite(axis, previous)?;
    for (index, &current) in values.iter().enumerate().skip(1) {
        validate_finite(axis, current)?;
        if current <= previous {
            return Err(RixsError::NonIncreasingGrid {
                axis,
                index,
                previous,
                current,
            });
        }
        previous = current;
    }
    Ok(())
}

fn validate_range(axis: &'static str, value: Real, min: Real, max: Real) -> Result<(), RixsError> {
    if value < min - BL_INTERP_TOLERANCE || value > max + BL_INTERP_TOLERANCE {
        Err(RixsError::OutOfRange {
            axis,
            value,
            min,
            max,
            tolerance: BL_INTERP_TOLERANCE,
        })
    } else {
        Ok(())
    }
}

fn interpolation_interval(values: ArrayView1<'_, Real>, target: Real) -> (usize, usize) {
    let (mut lower, mut upper) = values
        .iter()
        .position(|&value| value >= target)
        .map_or((-1, (values.len() * 2) as isize), |index| {
            (index as isize - 1, index as isize)
        });

    if lower < 0 {
        lower = 0;
        upper = 1;
    }
    if upper > values.len() as isize - 1 {
        upper = values.len() as isize - 1;
        lower = values.len() as isize - 2;
    }
    (lower as usize, upper as usize)
}

fn matrix_value(
    values: ArrayView2<'_, Complex>,
    row: usize,
    col: usize,
) -> Result<Complex, RixsError> {
    values
        .get((row, col))
        .copied()
        .ok_or(RixsError::MatrixTooSmall {
            rows: values.nrows(),
            cols: values.ncols(),
            required_rows: row + 1,
            required_cols: col + 1,
        })
}

fn validate_width(name: &'static str, value: Real) -> Result<(), RixsError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(RixsError::InvalidWidth { name, value })
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), RixsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RixsError::NonFiniteReal { name, value })
    }
}

fn validate_complex(name: &'static str, value: Complex) -> Result<(), RixsError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(RixsError::NonFiniteComplex {
            name,
            real: value.re,
            imaginary: value.im,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ShapeBuilder;

    use super::*;

    fn assert_real_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn interpolation_fixture() -> (
        ndarray::Array1<Real>,
        ndarray::Array1<Real>,
        ndarray::Array2<Complex>,
    ) {
        let x = ndarray::arr1(&[0.0, 1.0, 2.5]);
        let y = ndarray::arr1(&[-1.0, 0.5, 2.0, 4.0]);
        let mut values = ndarray::Array2::zeros((x.len(), y.len()).f());
        for col in 0..y.len() {
            for row in 0..x.len() {
                let fortran_row = row as Real + 1.0;
                let fortran_col = col as Real + 1.0;
                values[(row, col)] = Complex::new(
                    10.0 * fortran_row + fortran_col,
                    -1.5 * fortran_row + 0.25 * fortran_col,
                );
            }
        }
        (x, y, values)
    }

    #[test]
    fn kk_integral_matches_feff_reference() -> Result<(), RixsError> {
        let slope = Complex::new(0.7, -0.2);
        let intercept = Complex::new(1.1, 0.3);
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, 0.4)?,
            Complex::new(2.399_207_722_391_849_5, -4.331_304_425_751_682),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, -2.5)?,
            Complex::new(1.408_013_813_215_639, 0.155_788_642_294_960_7),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, -1.0)?,
            Complex::new(-2.912_583_862_845_679, 1.467_861_035_772_940_7),
        );
        assert_complex_close(
            kk_integral(slope, intercept, -1.0, 2.0, 0.25, 2.0)?,
            Complex::new(1.068_803_131_267_718_9, -2.067_433_969_813_704_3),
        );
        Ok(())
    }

    #[test]
    fn double_lorentz_matches_feff_reference() -> Result<(), RixsError> {
        assert_real_close(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, Some(5.0))?,
            1.117_803_997_544_239_5,
        );
        assert_real_close(
            integrated_double_lorentz(1.4, 2.2, 0.25, 0.65, -0.7, 0.18, Some(1.9))?,
            -0.348_748_408_558_602_4,
        );
        assert_real_close(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, None)?,
            1.384_083_044_982_698_6,
        );
        Ok(())
    }

    #[test]
    fn bilinear_interpolation_matches_feff_reference() -> Result<(), RixsError> {
        let (x, y, values) = interpolation_fixture();
        assert_complex_close(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), 0.4, 1.1)?,
            Complex::new(16.400_000_000_000_002, -1.5),
        );
        assert_complex_close(
            bilinear_interpolate_complex(
                x.view(),
                y.view(),
                values.view(),
                -0.000_004,
                -1.000_003,
            )?,
            Complex::new(10.999_958, -1.249_994_5),
        );
        assert_complex_close(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), 2.500_004, 4.000_002)?,
            Complex::new(39.333_374_666_666_68, -4.166_672_333_333_331),
        );
        Ok(())
    }

    #[test]
    fn rixs_helpers_reject_invalid_inputs() {
        assert!(matches!(
            integrated_double_lorentz(3.1, 2.7, 0.0, 0.3, 1.2, -0.08, Some(5.0)),
            Err(RixsError::InvalidWidth { name: "gamch", .. })
        ));
        assert!(matches!(
            integrated_double_lorentz(3.1, 2.7, 0.45, 0.3, 1.2, -0.08, Some(Real::NAN)),
            Err(RixsError::NonFiniteReal { name: "omega", .. })
        ));
        assert!(matches!(
            kk_integral(
                Complex::new(0.7, -0.2),
                Complex::new(1.1, 0.3),
                2.0,
                -1.0,
                0.25,
                0.4,
            ),
            Err(RixsError::InvalidInterval { .. })
        ));

        let (x, y, values) = interpolation_fixture();
        assert!(matches!(
            bilinear_interpolate_complex(x.view(), y.view(), values.view(), -0.1, 1.0),
            Err(RixsError::OutOfRange { axis: "x", .. })
        ));
        assert!(matches!(
            bilinear_interpolate_complex(
                ndarray::arr1(&[0.0, 0.0]).view(),
                y.view(),
                values.view(),
                0.0,
                1.0,
            ),
            Err(RixsError::NonIncreasingGrid { axis: "x", .. })
        ));
        assert!(matches!(
            bilinear_interpolate_complex(
                x.view(),
                y.view(),
                ndarray::Array2::zeros((2, 2)).view(),
                0.4,
                1.1,
            ),
            Err(RixsError::MatrixTooSmall { .. })
        ));
    }
}
