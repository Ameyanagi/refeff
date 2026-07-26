use faer::MatMut;
use faer::linalg::solvers::{PartialPivLu, Solve};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_complex::{Complex32, Complex64};

use crate::convert::{column_major_slice_mut, complex32_to_faer, complex32_view};
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
    let mut factors = complex32_row_major_values(matrix);
    let mut pivots = Vec::with_capacity(order);

    for pivot in 0..order {
        let mut pivot_row = pivot;
        let pivot_index = complex32_lu_index(order, pivot, pivot);
        let mut pivot_norm = complex32_abs1(factors[pivot_index]);
        for row in (pivot + 1)..order {
            let candidate = complex32_abs1(factors[complex32_lu_index(order, row, pivot)]);
            if candidate > pivot_norm {
                pivot_row = row;
                pivot_norm = candidate;
            }
        }
        pivots.push(pivot_row + 1);

        if factors[complex32_lu_index(order, pivot_row, pivot)] == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        if pivot_row != pivot {
            swap_complex32_flat_rows(&mut factors, order, pivot, pivot_row);
        }

        let pivot_value = factors[pivot_index];
        for row in (pivot + 1)..order {
            let row_pivot_index = complex32_lu_index(order, row, pivot);
            factors[row_pivot_index] = complex32_div(factors[row_pivot_index], pivot_value);
            let factor = factors[row_pivot_index];
            for col in (pivot + 1)..order {
                let pivot_col = factors[complex32_lu_index(order, pivot, col)];
                let row_col_index = complex32_lu_index(order, row, col);
                factors[row_col_index] =
                    complex32_sub_mul(factors[row_col_index], factor, pivot_col);
            }
        }
    }

    let len = factors.len();
    let factors = Array2::from_shape_vec((order, order), factors).map_err(|_| {
        LinalgError::InvalidOwnedShape {
            rows: order,
            cols: order,
            len,
        }
    })?;
    Ok(Complex32Lu { factors, pivots })
}

/// Factor a single-precision complex square matrix using `faer` partial pivoting.
///
/// Borrows `matrix` directly into `faer` without an element-wise copy when its
/// storage is already column-major (as the FMS system and scattering
/// matrices are, being built with `ndarray`'s `.f()` shape), falling back to
/// a copy otherwise.
pub fn complex32_faer_lu_factor(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<Complex32FaerLu, LinalgError> {
    ensure_complex32_square(matrix)?;
    let order = matrix.nrows();
    let lu = complex32_view(matrix).as_ref().partial_piv_lu();
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

    let columns = right_hand_side.ncols();
    let factors = lu
        .factors
        .as_slice()
        .ok_or(LinalgError::NonContiguousLuFactors)?;
    let mut solution = complex32_row_major_values(right_hand_side);
    for (pivot, &pivot_row) in lu.pivots.iter().enumerate() {
        let swap_row = pivot_row - 1;
        if swap_row != pivot {
            swap_complex32_flat_rows(&mut solution, columns, pivot, swap_row);
        }
    }

    for pivot in 0..order {
        for row in (pivot + 1)..order {
            let factor = factors[complex32_lu_index(order, row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[complex32_lu_index(columns, pivot, col)];
                let row_col_index = complex32_lu_index(columns, row, col);
                solution[row_col_index] =
                    complex32_sub_mul(solution[row_col_index], factor, pivot_value);
            }
        }
    }

    for pivot in (0..order).rev() {
        let diagonal = factors[complex32_lu_index(order, pivot, pivot)];
        if diagonal == Complex32::new(0.0, 0.0) {
            return Err(LinalgError::SingularMatrix { pivot });
        }
        for col in 0..columns {
            let index = complex32_lu_index(columns, pivot, col);
            solution[index] = complex32_div(solution[index], diagonal);
        }
        for row in 0..pivot {
            let factor = factors[complex32_lu_index(order, row, pivot)];
            for col in 0..columns {
                let pivot_value = solution[complex32_lu_index(columns, pivot, col)];
                let row_col_index = complex32_lu_index(columns, row, col);
                solution[row_col_index] =
                    complex32_sub_mul(solution[row_col_index], factor, pivot_value);
            }
        }
    }

    let len = solution.len();
    Array2::from_shape_vec((order, columns), solution).map_err(|_| LinalgError::InvalidOwnedShape {
        rows: order,
        cols: columns,
        len,
    })
}

/// Solve `A * X = B` from `faer` single-complex LU factors.
///
/// Copies `right_hand_side` into the owned result once, then solves in place
/// through [`complex32_faer_lu_solve_in_place`] so the common column-major
/// case (FMS system matrices built with `ndarray`'s `.f()` shape) avoids the
/// extra `faer`-side copy the naive `Mat::from_fn` + `solve` pipeline used to
/// take.
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

    let mut solution = right_hand_side.to_owned();
    complex32_faer_lu_solve_in_place(lu, solution.view_mut())?;
    Ok(solution)
}

/// Solve `A * X = B` from `faer` single-complex LU factors, writing the
/// solution directly into the caller-owned `right_hand_side` buffer.
///
/// When `right_hand_side`'s storage is already column-major (as `ndarray`'s
/// `.f()`-shaped FMS matrices are), the solve runs entirely against a `faer`
/// `MatMut` borrowed from that buffer with no intervening copy. Otherwise the
/// values are copied through a temporary `faer` matrix and written back,
/// matching [`complex32_faer_lu_solve`]'s result exactly.
pub fn complex32_faer_lu_solve_in_place(
    lu: &Complex32FaerLu,
    mut right_hand_side: ArrayViewMut2<'_, Complex32>,
) -> Result<(), LinalgError> {
    if right_hand_side.nrows() != lu.order {
        return Err(LinalgError::LengthMismatch {
            left_name: "right hand side rows",
            left: right_hand_side.nrows(),
            right_name: "factor rows",
            right: lu.order,
        });
    }

    let rows = right_hand_side.nrows();
    let cols = right_hand_side.ncols();
    if let Some(slice) = column_major_slice_mut(right_hand_side.view_mut()) {
        lu.lu
            .solve_in_place(MatMut::from_column_major_slice_mut(slice, rows, cols));
        return Ok(());
    }

    let rhs = complex32_to_faer(right_hand_side.view());
    let solution = lu.lu.solve(rhs.as_ref());
    for row in 0..rows {
        for col in 0..cols {
            right_hand_side[(row, col)] = solution[(row, col)];
        }
    }
    Ok(())
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

fn complex32_row_major_values(matrix: ArrayView2<'_, Complex32>) -> Vec<Complex32> {
    let mut values = Vec::with_capacity(matrix.len());
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

#[inline(always)]
fn complex32_lu_index(stride: usize, row: usize, col: usize) -> usize {
    row * stride + col
}

fn swap_complex32_flat_rows(values: &mut [Complex32], stride: usize, left: usize, right: usize) {
    for col in 0..stride {
        values.swap(
            complex32_lu_index(stride, left, col),
            complex32_lu_index(stride, right, col),
        );
    }
}

#[inline(always)]
fn complex32_div(value: Complex32, divisor: Complex32) -> Complex32 {
    let norm = divisor.re * divisor.re + divisor.im * divisor.im;
    Complex32::new(
        (value.re * divisor.re + value.im * divisor.im) / norm,
        (value.im * divisor.re - value.re * divisor.im) / norm,
    )
}

#[inline(always)]
fn complex32_sub_mul(value: Complex32, left: Complex32, right: Complex32) -> Complex32 {
    let product_re = left.re * right.re - left.im * right.im;
    let product_im = left.re * right.im + left.im * right.re;
    Complex32::new(value.re - product_re, value.im - product_im)
}

fn complex_abs1(value: Complex64) -> f64 {
    value.re.abs() + value.im.abs()
}

fn complex32_abs1(value: Complex32) -> f32 {
    value.re.abs() + value.im.abs()
}
