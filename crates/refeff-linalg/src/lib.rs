#![forbid(unsafe_code)]

//! Linear algebra bridge for the FEFF10 Rust port.
//!
//! FEFF module state is stored in `ndarray`; performance-critical matrix
//! operations are delegated to pure-Rust `faer` through this small adapter layer.

use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use thiserror::Error;

/// Error returned by FEFF linear-algebra helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LinalgError {
    /// FEFF determinant and inversion helpers require square matrices.
    #[error("matrix must be square, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    /// Matrix/vector dimensions must match.
    #[error("length mismatch: {left_name} has {left} values but {right_name} has {right}")]
    LengthMismatch {
        left_name: &'static str,
        left: usize,
        right_name: &'static str,
        right: usize,
    },
    /// FEFF `invertmatrix` stops when a pivot is exactly zero.
    #[error("matrix is singular at pivot {pivot}")]
    SingularMatrix { pivot: usize },
    /// Least-squares design matrices must contain at least one column.
    #[error("design matrix must have at least one column")]
    EmptyDesign,
}

/// Copy a real `ndarray` matrix view into a `faer` matrix.
pub fn real_to_faer(view: ArrayView2<'_, f64>) -> Mat<f64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

/// Copy a complex `ndarray` matrix view into a `faer` matrix.
pub fn complex_to_faer(view: ArrayView2<'_, Complex64>) -> Mat<Complex64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

/// Copy a real `faer` matrix into row-indexed `ndarray` storage.
pub fn real_from_faer(matrix: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

/// Copy a complex `faer` matrix into row-indexed `ndarray` storage.
pub fn complex_from_faer(matrix: &Mat<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

/// Multiply two real matrices through the pure-Rust `faer` backend.
pub fn real_matmul(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>) -> Array2<f64> {
    let lhs = real_to_faer(lhs);
    let rhs = real_to_faer(rhs);
    real_from_faer(&(&lhs * &rhs))
}

/// Multiply two complex matrices through the pure-Rust `faer` backend.
pub fn complex_matmul(
    lhs: ArrayView2<'_, Complex64>,
    rhs: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let lhs = complex_to_faer(lhs);
    let rhs = complex_to_faer(rhs);
    complex_from_faer(&(&lhs * &rhs))
}

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

/// Port of FEFF `determ`: determinant by Bevington-style elimination.
///
/// The original routine mutates its input work array and swaps columns when a
/// diagonal pivot is zero. This wrapper copies the input view and returns the
/// determinant without modifying caller-owned `ndarray` storage.
pub fn feff_determinant(matrix: ArrayView2<'_, f64>) -> Result<f64, LinalgError> {
    ensure_square(matrix)?;
    let order = matrix.nrows();
    let mut work = matrix.to_owned();
    let mut determinant = 1.0;

    for pivot in 0..order {
        if work[(pivot, pivot)] == 0.0 {
            let Some(swap_col) = (pivot..order).find(|&col| work[(pivot, col)] != 0.0) else {
                return Ok(0.0);
            };
            for row in pivot..order {
                let saved = work[(row, swap_col)];
                work[(row, swap_col)] = work[(row, pivot)];
                work[(row, pivot)] = saved;
            }
            determinant = -determinant;
        }

        determinant *= work[(pivot, pivot)];
        if pivot + 1 < order {
            for row in (pivot + 1)..order {
                for col in (pivot + 1)..order {
                    work[(row, col)] -=
                        work[(row, pivot)] * work[(pivot, col)] / work[(pivot, pivot)];
                }
            }
        }
    }

    Ok(determinant)
}

/// Port of FEFF `invertmatrix`: inverse by pivoted Gaussian elimination.
///
/// FEFF aborts on an exactly zero pivot; this Rust wrapper returns
/// [`LinalgError::SingularMatrix`] instead.
pub fn feff_inverse(matrix: ArrayView2<'_, f64>) -> Result<Array2<f64>, LinalgError> {
    ensure_square(matrix)?;
    let order = matrix.nrows();
    let mut work = matrix.to_owned();
    let mut inverse = Array2::zeros((order, order));
    for index in 0..order {
        inverse[(index, index)] = 1.0;
    }

    for pivot in 0..order {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..order {
            if work[(row, pivot)].abs() > work[(pivot_row, pivot)].abs() {
                pivot_row = row;
            }
        }

        if pivot_row > pivot {
            for col in pivot..order {
                let saved = work[(pivot, col)];
                work[(pivot, col)] = work[(pivot_row, col)];
                work[(pivot_row, col)] = saved;
            }
            for col in 0..order {
                let saved = inverse[(pivot, col)];
                inverse[(pivot, col)] = inverse[(pivot_row, col)];
                inverse[(pivot_row, col)] = saved;
            }
        }

        if work[(pivot, pivot)] == 0.0 {
            return Err(LinalgError::SingularMatrix { pivot });
        }

        for row in 0..order {
            let pivot_value = work[(pivot, pivot)];
            let ratio = if row == pivot {
                1.0 / pivot_value - 1.0
            } else {
                -work[(row, pivot)] / pivot_value
            };
            for col in pivot..order {
                work[(row, col)] += ratio * work[(pivot, col)];
            }
            for col in 0..order {
                inverse[(row, col)] += ratio * inverse[(pivot, col)];
            }
        }
    }

    Ok(inverse)
}

fn ensure_square(matrix: ArrayView2<'_, f64>) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

fn ensure_complex_square(matrix: ArrayView2<'_, Complex64>) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

fn complex_solve(
    matrix: ArrayView2<'_, Complex64>,
    right_hand_side: ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LinalgError> {
    ensure_complex_square(matrix)?;
    let order = matrix.nrows();
    if right_hand_side.len() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side",
            left: right_hand_side.len(),
            right_name: "matrix rows",
            right: order,
        });
    }

    let mut work = matrix.to_owned();
    let mut rhs = right_hand_side.to_owned();
    for pivot in 0..order {
        let pivot_row = (pivot..order)
            .max_by(|&left, &right| {
                work[(left, pivot)]
                    .norm_sqr()
                    .total_cmp(&work[(right, pivot)].norm_sqr())
            })
            .ok_or(LinalgError::SingularMatrix { pivot })?;

        if work[(pivot_row, pivot)] == Complex64::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            for col in 0..order {
                let saved = work[(pivot, col)];
                work[(pivot, col)] = work[(pivot_row, col)];
                work[(pivot_row, col)] = saved;
            }
            let saved = rhs[pivot];
            rhs[pivot] = rhs[pivot_row];
            rhs[pivot_row] = saved;
        }

        let pivot_value = work[(pivot, pivot)];
        for row in (pivot + 1)..order {
            let factor = work[(row, pivot)] / pivot_value;
            work[(row, pivot)] = Complex64::new(0.0, 0.0);
            for col in (pivot + 1)..order {
                let pivot_col = work[(pivot, col)];
                work[(row, col)] -= factor * pivot_col;
            }
            let pivot_rhs = rhs[pivot];
            rhs[row] -= factor * pivot_rhs;
        }
    }

    let mut solution = Array1::zeros(order);
    for row in (0..order).rev() {
        let mut sum = rhs[row];
        for col in (row + 1)..order {
            sum -= work[(row, col)] * solution[col];
        }
        let diagonal = work[(row, row)];
        if diagonal == Complex64::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot: row });
        }
        solution[row] = sum / diagonal;
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn multiplies_real_matrices_through_faer() {
        let lhs = array![[1.0, 2.0], [3.0, 4.0]];
        let rhs = array![[5.0], [6.0]];
        let out = real_matmul(lhs.view(), rhs.view());
        assert_eq!(out, array![[17.0], [39.0]]);
    }

    #[test]
    fn roundtrips_complex_matrix_layout() {
        let input = array![[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]];
        let matrix = complex_to_faer(input.view());
        let out = complex_from_faer(&matrix);
        assert_eq!(out, input);
    }

    #[test]
    fn determinant_matches_feff_reference_with_column_swap() -> Result<(), LinalgError> {
        let matrix = array![[0.0, 2.0, -1.0], [1.0, -3.0, 2.0], [4.0, 1.0, 0.5]];

        assert_close(feff_determinant(matrix.view())?, 2.0);
        Ok(())
    }

    #[test]
    fn determinant_returns_zero_for_singular_matrix() -> Result<(), LinalgError> {
        let matrix = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, 0.0, 1.0]];

        assert_eq!(feff_determinant(matrix.view())?, 0.0);
        Ok(())
    }

    #[test]
    fn determinant_rejects_non_square_matrix() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        assert_eq!(
            feff_determinant(matrix.view()),
            Err(LinalgError::NonSquare { rows: 2, cols: 3 })
        );
    }

    #[test]
    fn inverse_matches_feff_reference() -> Result<(), LinalgError> {
        let matrix = array![[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [0.25, -0.5, 1.5]];
        let inverse = feff_inverse(matrix.view())?;
        let expected = array![
            [
                0.41791044776119407,
                0.14925373134328357,
                0.05970149253731341
            ],
            [
                -0.23880597014925373,
                0.34328358208955223,
                0.5373134328358209
            ],
            [-0.14925373134328357, 0.08955223880597014, 0.835820895522388]
        ];

        assert_matrix_close(inverse.view(), expected.view());
        assert_matrix_close(
            real_matmul(matrix.view(), inverse.view()).view(),
            Array2::eye(3).view(),
        );
        Ok(())
    }

    #[test]
    fn inverse_rejects_singular_matrix() {
        let matrix = array![[1.0, 2.0], [2.0, 4.0]];

        assert_eq!(
            feff_inverse(matrix.view()),
            Err(LinalgError::SingularMatrix { pivot: 1 })
        );
    }

    #[test]
    fn inverse_rejects_non_square_matrix() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        assert_eq!(
            feff_inverse(matrix.view()),
            Err(LinalgError::NonSquare { rows: 2, cols: 3 })
        );
    }

    #[test]
    fn complex_polyfit_matches_feff_reference_order2() -> Result<(), LinalgError> {
        let x = [-1.0, 0.0, 1.5, 2.0, 3.5];
        let y = array![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(3.0, -0.5),
            Complex64::new(4.2, 1.1),
            Complex64::new(10.0, 2.0)
        ];

        let coefficients = complex_polyfit(&x, y.view(), 2)?;
        assert_complex_vec_close(
            coefficients.view(),
            &[
                Complex64::new(0.7367647058823517, -0.4195433436532505),
                Complex64::new(0.47303921568627294, 0.3809984520123841),
                Complex64::new(0.6245098039215693, 0.08521671826625377),
            ],
        );

        let evaluated = complex_polyval(coefficients.view(), &[-0.5, 0.25, 2.25, 4.0]);
        assert_complex_vec_close(
            evaluated.view(),
            &[
                Complex64::new(0.6563725490196075, -0.5887383900928791),
                Complex64::new(0.8940563725490179, -0.31896768575851364),
                Complex64::new(4.96268382352941, 0.8691128095975234),
                Complex64::new(12.621078431372553, 2.467917956656346),
            ],
        );
        Ok(())
    }

    #[test]
    fn complex_polyfit_matches_feff_reference_order3() -> Result<(), LinalgError> {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = array![
            Complex64::new(-1.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(2.5, -0.25),
            Complex64::new(5.0, 1.0)
        ];

        let coefficients = complex_polyfit(&x, y.view(), 3)?;
        assert_complex_vec_close(
            coefficients.view(),
            &[
                Complex64::new(1.0, -0.4428571428571429),
                Complex64::new(1.1666666666666663, 0.5833333333333331),
                Complex64::new(0.25, 0.44642857142857145),
                Complex64::new(0.08333333333333344, -0.208_333_333_333_333_3),
            ],
        );

        let evaluated = complex_polyval(coefficients.view(), &[-1.5, 0.5, 2.5]);
        assert_complex_vec_close(
            evaluated.view(),
            &[
                Complex64::new(-0.4687499999999998, 0.389732142857143),
                Complex64::new(1.6562499999999998, -0.06562500000000011),
                Complex64::new(6.781250000000001, 0.550446428571429),
            ],
        );
        Ok(())
    }

    #[test]
    fn complex_least_squares_rejects_invalid_inputs() {
        let design = array![[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]];
        let y = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        assert!(matches!(
            complex_least_squares_normal_eq(design.view(), y.view()),
            Err(LinalgError::LengthMismatch { .. })
        ));

        let singular_y = array![Complex64::new(1.0, 0.0)];
        assert!(matches!(
            complex_least_squares_normal_eq(design.view(), singular_y.view()),
            Err(LinalgError::SingularMatrix { .. })
        ));
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_matrix_close(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, col), &value) in actual.indexed_iter() {
            assert_close(value, expected[(row, col)]);
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_complex_vec_close(actual: ArrayView1<'_, Complex64>, expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }
}
