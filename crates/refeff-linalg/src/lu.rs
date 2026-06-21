use faer::linalg::solvers::{PartialPivLu, Solve};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::{Complex32, Complex64};

use crate::convert::{complex32_from_faer, complex32_to_faer};
use crate::error::LinalgError;
use crate::validation::{ensure_complex_square, ensure_complex32_square, ensure_square};

/// LU factors for a real square matrix, matching FEFF's `dgetrf` layout.
///
/// The factor matrix stores the unit-lower `L` multipliers below the diagonal
/// and the upper-triangular `U` factors on and above the diagonal. Pivot
/// indices are one-based, matching LAPACK and the FEFF Fortran sources.
#[derive(Debug, Clone, PartialEq)]
pub struct RealLu {
    factors: Array2<f64>,
    pivots: Vec<usize>,
}

impl RealLu {
    /// Return the packed `L`/`U` factor matrix.
    #[must_use]
    pub fn factors(&self) -> ArrayView2<'_, f64> {
        self.factors.view()
    }

    /// Return one-based row-pivot indices in LAPACK `IPIV` order.
    #[must_use]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
}

/// LU factors for a complex square matrix, matching FEFF's `zgetrf` layout.
///
/// Complex pivot selection uses LAPACK's `dcabs1` norm, `abs(re) + abs(im)`,
/// so tie-breaking and row swaps remain compatible with the FEFF reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexLu {
    factors: Array2<Complex64>,
    pivots: Vec<usize>,
}

impl ComplexLu {
    /// Return the packed `L`/`U` factor matrix.
    #[must_use]
    pub fn factors(&self) -> ArrayView2<'_, Complex64> {
        self.factors.view()
    }

    /// Return one-based row-pivot indices in LAPACK `IPIV` order.
    #[must_use]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
}

/// LU factors for a single-precision complex square matrix.
///
/// This mirrors FEFF's `cgetrf` storage and pivot convention for legacy
/// modules that still operate on Fortran `complex` arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct Complex32Lu {
    factors: Array2<Complex32>,
    pivots: Vec<usize>,
}

impl Complex32Lu {
    /// Return the packed `L`/`U` factor matrix.
    #[must_use]
    pub fn factors(&self) -> ArrayView2<'_, Complex32> {
        self.factors.view()
    }

    /// Return one-based row-pivot indices in LAPACK `IPIV` order.
    #[must_use]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
}

/// `faer` LU factors for a single-precision complex square matrix.
///
/// This backend is intended for hot solve paths that do not inspect FEFF's
/// packed LAPACK-compatible LU storage or one-based pivot list.
#[derive(Debug, Clone)]
pub struct Complex32FaerLu {
    lu: PartialPivLu<Complex32>,
    order: usize,
}

impl Complex32FaerLu {
    /// Return the matrix order factored by `faer`.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }
}

/// Factor a real square matrix using FEFF-compatible partial pivoting.
///
/// This is the square-system subset of LAPACK `dgetrf` used by FEFF call
/// sites. The returned factors can be passed to [`real_lu_solve`].
pub fn real_lu_factor(matrix: ArrayView2<'_, f64>) -> Result<RealLu, LinalgError> {
    ensure_square(matrix)?;
    let order = matrix.nrows();
    let mut factors = matrix.to_owned();
    let mut pivots = Vec::with_capacity(order);

    for pivot in 0..order {
        let mut pivot_row = pivot;
        let mut pivot_norm = factors[(pivot, pivot)].abs();
        for row in (pivot + 1)..order {
            let candidate = factors[(row, pivot)].abs();
            if candidate > pivot_norm {
                pivot_row = row;
                pivot_norm = candidate;
            }
        }
        pivots.push(pivot_row + 1);

        if factors[(pivot_row, pivot)] == 0.0 {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            swap_real_rows(&mut factors, pivot, pivot_row);
        }

        let pivot_value = factors[(pivot, pivot)];
        for row in (pivot + 1)..order {
            factors[(row, pivot)] /= pivot_value;
            let factor = factors[(row, pivot)];
            for col in (pivot + 1)..order {
                let pivot_col = factors[(pivot, col)];
                factors[(row, col)] -= factor * pivot_col;
            }
        }
    }

    Ok(RealLu { factors, pivots })
}

/// Solve `A * X = B` from FEFF-compatible real LU factors.
///
/// The solve path mirrors LAPACK `dgetrs` with `TRANS = 'N'`: row pivots are
/// applied first, followed by unit-lower and upper-triangular substitution.
pub fn real_lu_solve(
    lu: &RealLu,
    right_hand_side: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, LinalgError> {
    let order = lu.factors.nrows();
    if right_hand_side.nrows() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side rows",
            left: right_hand_side.nrows(),
            right_name: "factor rows",
            right: order,
        });
    }

    let mut solution = right_hand_side.to_owned();
    for (pivot, &pivot_row) in lu.pivots.iter().enumerate() {
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            swap_real_rows(&mut solution, pivot, swap_row);
        }
    }

    let columns = solution.ncols();
    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = lu.factors[(pivot, pivot)];
        if diagonal == 0.0 {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        for col in 0..columns {
            solution[(pivot, col)] /= diagonal;
        }
        for row in 0..pivot {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    Ok(solution)
}

/// Solve `A * x = b` from FEFF-compatible real LU factors.
pub fn real_lu_solve_vector(
    lu: &RealLu,
    right_hand_side: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, LinalgError> {
    let matrix_rhs =
        Array2::from_shape_fn((right_hand_side.len(), 1), |(row, _)| right_hand_side[row]);
    let solution = real_lu_solve(lu, matrix_rhs.view())?;
    Ok(Array1::from_shape_fn(solution.nrows(), |row| {
        solution[(row, 0)]
    }))
}

/// Factor a complex square matrix using FEFF-compatible partial pivoting.
///
/// This is the square-system subset of LAPACK `zgetrf` used by FEFF call
/// sites. The returned factors can be passed to [`complex_lu_solve`].
pub fn complex_lu_factor(matrix: ArrayView2<'_, Complex64>) -> Result<ComplexLu, LinalgError> {
    ensure_complex_square(matrix)?;
    let order = matrix.nrows();
    let mut factors = matrix.to_owned();
    let mut pivots = Vec::with_capacity(order);

    for pivot in 0..order {
        let mut pivot_row = pivot;
        let mut pivot_norm = complex_abs1(factors[(pivot, pivot)]);
        for row in (pivot + 1)..order {
            let candidate = complex_abs1(factors[(row, pivot)]);
            if candidate > pivot_norm {
                pivot_row = row;
                pivot_norm = candidate;
            }
        }
        pivots.push(pivot_row + 1);

        if factors[(pivot_row, pivot)] == Complex64::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            swap_complex_rows(&mut factors, pivot, pivot_row);
        }

        let pivot_value = factors[(pivot, pivot)];
        for row in (pivot + 1)..order {
            factors[(row, pivot)] /= pivot_value;
            let factor = factors[(row, pivot)];
            for col in (pivot + 1)..order {
                let pivot_col = factors[(pivot, col)];
                factors[(row, col)] -= factor * pivot_col;
            }
        }
    }

    Ok(ComplexLu { factors, pivots })
}

/// Solve `A * X = B` from FEFF-compatible complex LU factors.
///
/// The solve path mirrors LAPACK `zgetrs` with `TRANS = 'N'`: row pivots are
/// applied first, followed by unit-lower and upper-triangular substitution.
pub fn complex_lu_solve(
    lu: &ComplexLu,
    right_hand_side: ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LinalgError> {
    let order = lu.factors.nrows();
    if right_hand_side.nrows() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side rows",
            left: right_hand_side.nrows(),
            right_name: "factor rows",
            right: order,
        });
    }

    let mut solution = right_hand_side.to_owned();
    for (pivot, &pivot_row) in lu.pivots.iter().enumerate() {
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            swap_complex_rows(&mut solution, pivot, swap_row);
        }
    }

    let columns = solution.ncols();
    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = lu.factors[(pivot, pivot)];
        if diagonal == Complex64::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        for col in 0..columns {
            solution[(pivot, col)] /= diagonal;
        }
        for row in 0..pivot {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    Ok(solution)
}

/// Solve `A * x = b` from FEFF-compatible complex LU factors.
pub fn complex_lu_solve_vector(
    lu: &ComplexLu,
    right_hand_side: ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LinalgError> {
    let matrix_rhs =
        Array2::from_shape_fn((right_hand_side.len(), 1), |(row, _)| right_hand_side[row]);
    let solution = complex_lu_solve(lu, matrix_rhs.view())?;
    Ok(Array1::from_shape_fn(solution.nrows(), |row| {
        solution[(row, 0)]
    }))
}

/// Factor a single-precision complex square matrix like FEFF `cgetrf`.
pub fn complex32_lu_factor(matrix: ArrayView2<'_, Complex32>) -> Result<Complex32Lu, LinalgError> {
    ensure_complex32_square(matrix)?;
    let order = matrix.nrows();
    let mut factors = matrix.to_owned();
    let mut pivots = Vec::with_capacity(order);

    for pivot in 0..order {
        let mut pivot_row = pivot;
        let mut pivot_norm = complex32_abs1(factors[(pivot, pivot)]);
        for row in (pivot + 1)..order {
            let candidate = complex32_abs1(factors[(row, pivot)]);
            if candidate > pivot_norm {
                pivot_row = row;
                pivot_norm = candidate;
            }
        }
        pivots.push(pivot_row + 1);

        if factors[(pivot_row, pivot)] == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            swap_complex32_rows(&mut factors, pivot, pivot_row);
        }

        let pivot_value = factors[(pivot, pivot)];
        for row in (pivot + 1)..order {
            factors[(row, pivot)] /= pivot_value;
            let factor = factors[(row, pivot)];
            for col in (pivot + 1)..order {
                let pivot_col = factors[(pivot, col)];
                factors[(row, col)] -= factor * pivot_col;
            }
        }
    }

    Ok(Complex32Lu { factors, pivots })
}

/// Factor a single-precision complex square matrix using `faer` partial pivoting.
pub fn complex32_faer_lu_factor(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<Complex32FaerLu, LinalgError> {
    ensure_complex32_square(matrix)?;
    let order = matrix.nrows();
    let faer_matrix = complex32_to_faer(matrix);
    let lu = faer_matrix.partial_piv_lu();
    for pivot in 0..order {
        if lu.U()[(pivot, pivot)] == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
    }
    Ok(Complex32FaerLu { lu, order })
}

/// Solve `A * X = B` from FEFF-compatible single-complex LU factors.
pub fn complex32_lu_solve(
    lu: &Complex32Lu,
    right_hand_side: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, LinalgError> {
    let order = lu.factors.nrows();
    if right_hand_side.nrows() != order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side rows",
            left: right_hand_side.nrows(),
            right_name: "factor rows",
            right: order,
        });
    }

    let mut solution = right_hand_side.to_owned();
    for (pivot, &pivot_row) in lu.pivots.iter().enumerate() {
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            swap_complex32_rows(&mut solution, pivot, swap_row);
        }
    }

    let columns = solution.ncols();
    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = lu.factors[(pivot, pivot)];
        if diagonal == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        for col in 0..columns {
            solution[(pivot, col)] /= diagonal;
        }
        for row in 0..pivot {
            let factor = lu.factors[(row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[(pivot, col)];
                solution[(row, col)] -= factor * pivot_value;
            }
        }
    }

    Ok(solution)
}

/// Solve `A * X = B` from `faer` single-complex LU factors.
pub fn complex32_faer_lu_solve(
    lu: &Complex32FaerLu,
    right_hand_side: ArrayView2<'_, Complex32>,
) -> Result<Array2<Complex32>, LinalgError> {
    if right_hand_side.nrows() != lu.order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side rows",
            left: right_hand_side.nrows(),
            right_name: "factor rows",
            right: lu.order,
        });
    }

    let rhs = complex32_to_faer(right_hand_side);
    let solution = lu.lu.solve(rhs.as_ref());
    Ok(complex32_from_faer(&solution))
}

/// Solve `A * x = b` from FEFF-compatible single-complex LU factors.
pub fn complex32_lu_solve_vector(
    lu: &Complex32Lu,
    right_hand_side: ArrayView1<'_, Complex32>,
) -> Result<Array1<Complex32>, LinalgError> {
    let matrix_rhs =
        Array2::from_shape_fn((right_hand_side.len(), 1), |(row, _)| right_hand_side[row]);
    let solution = complex32_lu_solve(lu, matrix_rhs.view())?;
    Ok(Array1::from_shape_fn(solution.nrows(), |row| {
        solution[(row, 0)]
    }))
}

pub(crate) fn complex_solve(
    matrix: ArrayView2<'_, Complex64>,
    right_hand_side: ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LinalgError> {
    let lu = complex_lu_factor(matrix)?;
    complex_lu_solve_vector(&lu, right_hand_side)
}

fn swap_real_rows(matrix: &mut Array2<f64>, left: usize, right: usize) {
    for col in 0..matrix.ncols() {
        let saved = matrix[(left, col)];
        matrix[(left, col)] = matrix[(right, col)];
        matrix[(right, col)] = saved;
    }
}

fn swap_complex_rows(matrix: &mut Array2<Complex64>, left: usize, right: usize) {
    for col in 0..matrix.ncols() {
        let saved = matrix[(left, col)];
        matrix[(left, col)] = matrix[(right, col)];
        matrix[(right, col)] = saved;
    }
}

fn swap_complex32_rows(matrix: &mut Array2<Complex32>, left: usize, right: usize) {
    for col in 0..matrix.ncols() {
        let saved = matrix[(left, col)];
        matrix[(left, col)] = matrix[(right, col)];
        matrix[(right, col)] = saved;
    }
}

fn complex_abs1(value: Complex64) -> f64 {
    value.re.abs() + value.im.abs()
}

fn complex32_abs1(value: Complex32) -> f32 {
    value.re.abs() + value.im.abs()
}
