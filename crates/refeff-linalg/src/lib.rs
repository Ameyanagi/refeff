#![forbid(unsafe_code)]

//! Linear algebra bridge for the FEFF10 Rust port.
//!
//! FEFF module state is stored in `ndarray`; performance-critical matrix
//! operations are delegated to pure-Rust `faer` through this small adapter layer.

use faer::Mat;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use thiserror::Error;

/// Error returned by FEFF linear-algebra helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LinalgError {
    /// FEFF determinant and inversion helpers require square matrices.
    #[error("matrix must be square, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    /// FEFF `invertmatrix` stops when a pivot is exactly zero.
    #[error("matrix is singular at pivot {pivot}")]
    SingularMatrix { pivot: usize },
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
}
