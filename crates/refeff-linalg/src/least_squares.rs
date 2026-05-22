use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::error::LinalgError;
use crate::lu::complex_solve;

/// Port of FEFF `leastsq`: complex least squares by normal equations.
///
/// FEFF forms `transpose(F) * F` and `transpose(F) * y`, then solves the
/// resulting complex system with LAPACK `zgetrf/zgetrs`. Note that this is a
/// plain transpose, not a conjugate transpose; this function preserves that
/// legacy behavior for compatibility.
pub fn complex_least_squares_normal_eq(
    design: ArrayView2<'_, Complex64>,
    values: ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LinalgError> {
    let rows = design.nrows();
    let cols = design.ncols();
    if cols == 0 {
        return Err(LinalgError::EmptyDesign);
    }
    if values.len() != rows {
        return Err(LinalgError::LengthMismatch {
            left_name: "values",
            left: values.len(),
            right_name: "design rows",
            right: rows,
        });
    }

    let normal = Array2::from_shape_fn((cols, cols), |(row, col)| {
        (0..rows)
            .map(|index| design[(index, row)] * design[(index, col)])
            .sum()
    });
    let right_hand_side = Array1::from_shape_fn(cols, |row| {
        (0..rows)
            .map(|index| design[(index, row)] * values[index])
            .sum()
    });

    complex_solve(normal.view(), right_hand_side.view())
}

/// Port of FEFF `polyfit`: fit complex polynomial coefficients.
///
/// Coefficients are returned in ascending order, so `coefficients[0]` is the
/// constant term and `coefficients[order]` multiplies `x^order`.
pub fn complex_polyfit(
    x: &[f64],
    y: ArrayView1<'_, Complex64>,
    order: usize,
) -> Result<Array1<Complex64>, LinalgError> {
    if y.len() != x.len() {
        return Err(LinalgError::LengthMismatch {
            left_name: "y",
            left: y.len(),
            right_name: "x",
            right: x.len(),
        });
    }
    let design = Array2::from_shape_fn((x.len(), order + 1), |(row, col)| {
        Complex64::new(x[row].powi(col as i32), 0.0)
    });
    complex_least_squares_normal_eq(design.view(), y)
}

/// Port of FEFF `polyval`: evaluate complex polynomial coefficients.
///
/// Coefficients are interpreted in ascending order, matching [`complex_polyfit`].
#[must_use]
pub fn complex_polyval(coefficients: ArrayView1<'_, Complex64>, x: &[f64]) -> Array1<Complex64> {
    Array1::from_shape_fn(x.len(), |row| {
        coefficients
            .iter()
            .rev()
            .fold(Complex64::new(0.0, 0.0), |value, &coefficient| {
                value * x[row] + coefficient
            })
    })
}
