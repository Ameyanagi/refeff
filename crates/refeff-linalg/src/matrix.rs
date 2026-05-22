use ndarray::{Array2, ArrayView2};

use crate::error::LinalgError;
use crate::validation::ensure_square;

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
