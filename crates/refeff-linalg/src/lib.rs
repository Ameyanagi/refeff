#![forbid(unsafe_code)]

//! Linear algebra bridge for the FEFF10 Rust port.
//!
//! FEFF module state is stored in `ndarray`; performance-critical matrix
//! operations are delegated to pure-Rust `faer` through this small adapter layer.

use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::{Complex32, Complex64};
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
    /// Selected matrix entries must be finite before calling the eigensolver.
    #[error("matrix entry ({row},{col}) must be finite")]
    NonFiniteMatrixEntry { row: usize, col: usize },
    /// Scalar inputs must be finite before evaluating FEFF helper formulas.
    #[error("{name} must be finite")]
    NonFiniteScalar { name: &'static str },
    /// FEFF's `SSYEV` reports positive `INFO` when eigenvalue iteration fails.
    #[error("symmetric eigensolver did not converge")]
    EigenDidNotConverge,
}

/// FEFF `UPLO` selector for real symmetric eigensolvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetricTriangle {
    /// Read the lower triangle, matching FEFF `UPLO = 'L'`.
    Lower,
    /// Read the upper triangle, matching FEFF `UPLO = 'U'`.
    Upper,
}

/// Single-precision symmetric eigensystem from FEFF `SSYEV`.
#[derive(Debug, Clone, PartialEq)]
pub struct Real32SymmetricEigen {
    eigenvalues: Array1<f32>,
    eigenvectors: Array2<f32>,
}

/// Analytic single-precision 2x2 symmetric eigensystem from FEFF `SLAEV2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Real32Symmetric2x2Eigen {
    larger_abs_eigenvalue: f32,
    smaller_abs_eigenvalue: f32,
    larger_abs_eigenvector: [f32; 2],
}

impl Real32SymmetricEigen {
    /// Eigenvalues sorted in nondecreasing order, matching LAPACK `SSYEV`.
    #[must_use]
    pub fn eigenvalues(&self) -> ArrayView1<'_, f32> {
        self.eigenvalues.view()
    }

    /// Orthonormal eigenvectors stored column-wise.
    #[must_use]
    pub fn eigenvectors(&self) -> ArrayView2<'_, f32> {
        self.eigenvectors.view()
    }
}

impl Real32Symmetric2x2Eigen {
    /// Eigenvalue with larger absolute value, matching FEFF `RT1`.
    #[must_use]
    pub fn larger_abs_eigenvalue(self) -> f32 {
        self.larger_abs_eigenvalue
    }

    /// Eigenvalue with smaller absolute value, matching FEFF `RT2`.
    #[must_use]
    pub fn smaller_abs_eigenvalue(self) -> f32 {
        self.smaller_abs_eigenvalue
    }

    /// Unit right eigenvector for [`Self::larger_abs_eigenvalue`].
    #[must_use]
    pub fn larger_abs_eigenvector(self) -> [f32; 2] {
        self.larger_abs_eigenvector
    }
}

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

/// Port of FEFF `SLAE2`: analytic eigenvalues for a real symmetric 2x2 matrix.
///
/// The matrix is `[a b; b c]`. Results follow LAPACK/FEFF ordering: `RT1` is
/// the eigenvalue with larger absolute value and `RT2` is the one with smaller
/// absolute value. Use [`real32_symmetric_eigenvalues`] when `SSYEV` ascending
/// eigenvalue order is required.
pub fn real32_symmetric_2x2_eigenvalues(
    diagonal_a: f32,
    off_diagonal: f32,
    diagonal_c: f32,
) -> Result<[f32; 2], LinalgError> {
    ensure_finite_f32("a", diagonal_a)?;
    ensure_finite_f32("b", off_diagonal)?;
    ensure_finite_f32("c", diagonal_c)?;
    Ok(slae2_values(diagonal_a, off_diagonal, diagonal_c).values())
}

/// Port of FEFF `SLAEV2`: analytic eigensystem for a real symmetric 2x2 matrix.
///
/// The matrix is `[a b; b c]`. The eigenvector is returned for the eigenvalue
/// with larger absolute value, matching FEFF `CS1,SN1`.
pub fn real32_symmetric_2x2_eigen(
    diagonal_a: f32,
    off_diagonal: f32,
    diagonal_c: f32,
) -> Result<Real32Symmetric2x2Eigen, LinalgError> {
    ensure_finite_f32("a", diagonal_a)?;
    ensure_finite_f32("b", off_diagonal)?;
    ensure_finite_f32("c", diagonal_c)?;

    let values = slae2_values(diagonal_a, off_diagonal, diagonal_c);
    let (mut cosine, mut sine, vector_sign) = eigenvector_for_larger_abs_eigenvalue(
        values.difference,
        values.double_off_diagonal_abs,
        values.double_off_diagonal,
        values.radical,
    );
    if values.sign == vector_sign {
        let saved = cosine;
        cosine = -sine;
        sine = saved;
    }

    Ok(Real32Symmetric2x2Eigen {
        larger_abs_eigenvalue: values.larger_abs_eigenvalue,
        smaller_abs_eigenvalue: values.smaller_abs_eigenvalue,
        larger_abs_eigenvector: [cosine, sine],
    })
}

/// Port of FEFF `SSYEV` for single-precision symmetric eigenvalues.
///
/// FEFF passes either the lower or upper triangle through `UPLO`; entries in
/// the opposite triangle are ignored. The returned eigenvalues are sorted in
/// nondecreasing order.
pub fn real32_symmetric_eigenvalues(
    matrix: ArrayView2<'_, f32>,
    triangle: SymmetricTriangle,
) -> Result<Array1<f32>, LinalgError> {
    ensure_real32_symmetric_input(matrix, triangle)?;
    if matrix.nrows() == 0 {
        return Ok(Array1::zeros(0));
    }

    let faer_matrix = real32_symmetric_to_faer(matrix, triangle);
    faer_matrix
        .self_adjoint_eigenvalues(Side::Lower)
        .map(Array1::from_vec)
        .map_err(|_| LinalgError::EigenDidNotConverge)
}

/// Port of FEFF `SSYEV` for single-precision symmetric eigensystems.
///
/// The input triangle is mirrored into a full self-adjoint matrix before
/// calling `faer`, preserving FEFF's `UPLO` behavior while leaving caller-owned
/// `ndarray` storage untouched. Eigenvectors are returned column-wise.
pub fn real32_symmetric_eigen(
    matrix: ArrayView2<'_, f32>,
    triangle: SymmetricTriangle,
) -> Result<Real32SymmetricEigen, LinalgError> {
    ensure_real32_symmetric_input(matrix, triangle)?;
    if matrix.nrows() == 0 {
        return Ok(Real32SymmetricEigen {
            eigenvalues: Array1::zeros(0),
            eigenvectors: Array2::zeros((0, 0)),
        });
    }

    let faer_matrix = real32_symmetric_to_faer(matrix, triangle);
    let decomposition = faer_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| LinalgError::EigenDidNotConverge)?;
    let eigenvalues = Array1::from_iter(decomposition.S().column_vector().iter().copied());
    let eigenvectors_ref = decomposition.U();
    let eigenvectors = Array2::from_shape_fn(
        (eigenvectors_ref.nrows(), eigenvectors_ref.ncols()),
        |(row, col)| eigenvectors_ref[(row, col)],
    );

    Ok(Real32SymmetricEigen {
        eigenvalues,
        eigenvectors,
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

fn ensure_complex32_square(matrix: ArrayView2<'_, Complex32>) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

fn ensure_real32_symmetric_input(
    matrix: ArrayView2<'_, f32>,
    triangle: SymmetricTriangle,
) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }

    for row in 0..rows {
        for col in 0..cols {
            if triangle.includes(row, col) && !matrix[(row, col)].is_finite() {
                return Err(LinalgError::NonFiniteMatrixEntry { row, col });
            }
        }
    }
    Ok(())
}

fn ensure_finite_f32(name: &'static str, value: f32) -> Result<(), LinalgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LinalgError::NonFiniteScalar { name })
    }
}

fn real32_symmetric_to_faer(matrix: ArrayView2<'_, f32>, triangle: SymmetricTriangle) -> Mat<f32> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        triangle.selected_entry(matrix, row, col)
    })
}

#[derive(Debug, Clone, Copy)]
struct Slae2Values {
    larger_abs_eigenvalue: f32,
    smaller_abs_eigenvalue: f32,
    difference: f32,
    double_off_diagonal: f32,
    double_off_diagonal_abs: f32,
    radical: f32,
    sign: i8,
}

impl Slae2Values {
    fn values(self) -> [f32; 2] {
        [self.larger_abs_eigenvalue, self.smaller_abs_eigenvalue]
    }
}

fn slae2_values(diagonal_a: f32, off_diagonal: f32, diagonal_c: f32) -> Slae2Values {
    let sum = diagonal_a + diagonal_c;
    let difference = diagonal_a - diagonal_c;
    let difference_abs = difference.abs();
    let double_off_diagonal = off_diagonal + off_diagonal;
    let double_off_diagonal_abs = double_off_diagonal.abs();
    let (larger_abs_diagonal, smaller_abs_diagonal) = if diagonal_a.abs() > diagonal_c.abs() {
        (diagonal_a, diagonal_c)
    } else {
        (diagonal_c, diagonal_a)
    };
    let radical = if difference_abs > double_off_diagonal_abs {
        difference_abs * (1.0 + (double_off_diagonal_abs / difference_abs).powi(2)).sqrt()
    } else if difference_abs < double_off_diagonal_abs {
        double_off_diagonal_abs * (1.0 + (difference_abs / double_off_diagonal_abs).powi(2)).sqrt()
    } else {
        double_off_diagonal_abs * 2.0_f32.sqrt()
    };
    let (larger_abs_eigenvalue, smaller_abs_eigenvalue, sign) = if sum < 0.0 {
        let larger = 0.5 * (sum - radical);
        let smaller = (larger_abs_diagonal / larger) * smaller_abs_diagonal
            - (off_diagonal / larger) * off_diagonal;
        (larger, smaller, -1)
    } else if sum > 0.0 {
        let larger = 0.5 * (sum + radical);
        let smaller = (larger_abs_diagonal / larger) * smaller_abs_diagonal
            - (off_diagonal / larger) * off_diagonal;
        (larger, smaller, 1)
    } else {
        (0.5 * radical, -0.5 * radical, 1)
    };

    Slae2Values {
        larger_abs_eigenvalue,
        smaller_abs_eigenvalue,
        difference,
        double_off_diagonal,
        double_off_diagonal_abs,
        radical,
        sign,
    }
}

fn eigenvector_for_larger_abs_eigenvalue(
    difference: f32,
    double_off_diagonal_abs: f32,
    double_off_diagonal: f32,
    radical: f32,
) -> (f32, f32, i8) {
    let (vector_component, vector_sign) = if difference >= 0.0 {
        (difference + radical, 1)
    } else {
        (difference - radical, -1)
    };
    let vector_component_abs = vector_component.abs();
    if vector_component_abs > double_off_diagonal_abs {
        let tangent = -double_off_diagonal / vector_component;
        let sine = 1.0 / (1.0 + tangent * tangent).sqrt();
        (tangent * sine, sine, vector_sign)
    } else if double_off_diagonal_abs == 0.0 {
        (1.0, 0.0, vector_sign)
    } else {
        let tangent = -vector_component / double_off_diagonal;
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        (cosine, tangent * cosine, vector_sign)
    }
}

impl SymmetricTriangle {
    fn includes(self, row: usize, col: usize) -> bool {
        match self {
            Self::Lower => row >= col,
            Self::Upper => row <= col,
        }
    }

    fn selected_entry(self, matrix: ArrayView2<'_, f32>, row: usize, col: usize) -> f32 {
        match self {
            Self::Lower if row >= col => matrix[(row, col)],
            Self::Lower => matrix[(col, row)],
            Self::Upper if row <= col => matrix[(row, col)],
            Self::Upper => matrix[(col, row)],
        }
    }
}

fn complex_solve(
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
    fn real_lu_matches_feff_dgetrf_dgetrs_reference() -> Result<(), LinalgError> {
        let matrix = array![[0.0, 2.0, -1.0], [3.0, -1.0, 4.0], [1.0, 0.5, 2.0]];
        let right_hand_side = array![[1.0, -2.0], [0.0, 3.0], [2.0, -1.0]];

        let lu = real_lu_factor(matrix.view())?;
        assert_eq!(lu.pivots(), &[2, 2, 3]);
        assert_matrix_close(
            lu.factors(),
            array![
                [3.0, -1.0, 4.0],
                [0.0, 2.0, -1.0],
                [
                    0.333_333_333_333_333_3,
                    0.41666666666666663,
                    1.0833333333333335
                ]
            ]
            .view(),
        );

        let solution = real_lu_solve(&lu, right_hand_side.view())?;
        assert_matrix_close(
            solution.view(),
            array![
                [-1.5384615384615383, 1.9230769230769231],
                [1.2307692307692308, -1.5384615384615383],
                [1.4615384615384615, -1.0769230769230769]
            ]
            .view(),
        );

        let vector_solution = real_lu_solve_vector(&lu, right_hand_side.column(0))?;
        assert_eq!(vector_solution.len(), 3);
        assert_close(vector_solution[0], -1.5384615384615383);
        Ok(())
    }

    #[test]
    fn complex_lu_matches_feff_zgetrf_zgetrs_reference() -> Result<(), LinalgError> {
        let matrix = array![
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5)
            ],
            [
                Complex64::new(3.0, 2.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(4.0, -1.0)
            ],
            [
                Complex64::new(1.0, -3.0),
                Complex64::new(0.5, 2.0),
                Complex64::new(2.0, 0.0)
            ]
        ];
        let right_hand_side = array![
            [Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(3.0, 0.0)],
            [Complex64::new(2.0, 2.0), Complex64::new(-1.0, -0.5)]
        ];

        let lu = complex_lu_factor(matrix.view())?;
        assert_eq!(lu.pivots(), &[2, 2, 3]);
        assert_complex_matrix_close(
            lu.factors(),
            array![
                [
                    Complex64::new(3.0, 2.0),
                    Complex64::new(-1.0, 0.0),
                    Complex64::new(4.0, -1.0)
                ],
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(2.0, -1.0),
                    Complex64::new(-1.0, 0.5)
                ],
                [
                    Complex64::new(-0.23076923076923078, -0.846_153_846_153_846_1),
                    Complex64::new(-0.12307692307692306, 0.515_384_615_384_615_3),
                    Complex64::new(3.9038461538461537, 3.730_769_230_769_231)
                ]
            ]
            .view(),
        );

        let solution = complex_lu_solve(&lu, right_hand_side.view())?;
        assert_complex_matrix_close(
            solution.view(),
            array![
                [
                    Complex64::new(-0.233_470_733_718_054_4, 0.43198680956306673),
                    Complex64::new(-0.13470733718054406, -0.280_131_904_369_332_2)
                ],
                [
                    Complex64::new(0.600_164_880_461_665_2, 0.281_615_828_524_319_9),
                    Complex64::new(-0.798_351_195_383_347_1, 0.216_158_285_243_198_7)
                ],
                [
                    Complex64::new(0.600_329_760_923_330_6, -0.236_768_342_951_360_3),
                    Complex64::new(0.403_297_609_233_305_9, 0.432_316_570_486_397_3)
                ]
            ]
            .view(),
        );

        let vector_solution = complex_lu_solve_vector(&lu, right_hand_side.column(0))?;
        assert_complex_close(
            vector_solution[0],
            Complex64::new(-0.233_470_733_718_054_4, 0.43198680956306673),
        );
        Ok(())
    }

    #[test]
    fn complex32_lu_matches_feff_cgetrf_cgetrs_reference() -> Result<(), LinalgError> {
        let matrix = array![
            [
                Complex32::new(0.0, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(-1.0, 0.5)
            ],
            [
                Complex32::new(3.0, 2.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(4.0, -1.0)
            ],
            [
                Complex32::new(1.0, -3.0),
                Complex32::new(0.5, 2.0),
                Complex32::new(2.0, 0.0)
            ]
        ];
        let right_hand_side = array![
            [Complex32::new(1.0, 0.5), Complex32::new(-2.0, 1.0)],
            [Complex32::new(0.0, -1.0), Complex32::new(3.0, 0.0)],
            [Complex32::new(2.0, 2.0), Complex32::new(-1.0, -0.5)]
        ];

        let lu = complex32_lu_factor(matrix.view())?;
        assert_eq!(lu.pivots(), &[2, 2, 3]);
        assert_complex32_matrix_close(
            lu.factors(),
            array![
                [
                    Complex32::new(3.0, 2.0),
                    Complex32::new(-1.0, 0.0),
                    Complex32::new(4.0, -1.0)
                ],
                [
                    Complex32::new(0.0, 0.0),
                    Complex32::new(2.0, -1.0),
                    Complex32::new(-1.0, 0.5)
                ],
                [
                    Complex32::new(-0.23076928, -0.8461538),
                    Complex32::new(-0.12307697, 0.515_384_7),
                    Complex32::new(3.9038463, 3.730_769)
                ]
            ]
            .view(),
        );

        let solution = complex32_lu_solve(&lu, right_hand_side.view())?;
        assert_complex32_matrix_close(
            solution.view(),
            array![
                [
                    Complex32::new(-0.23347072, 0.43198672),
                    Complex32::new(-0.13470735, -0.280_131_9)
                ],
                [
                    Complex32::new(0.60016483, 0.28161582),
                    Complex32::new(-0.798_351_2, 0.2161583)
                ],
                [
                    Complex32::new(0.6003297, -0.236_768_3),
                    Complex32::new(0.4032976, 0.43231657)
                ]
            ]
            .view(),
        );

        let vector_solution = complex32_lu_solve_vector(&lu, right_hand_side.column(0))?;
        assert_complex32_close(vector_solution[0], Complex32::new(-0.23347072, 0.43198672));
        Ok(())
    }

    #[test]
    fn lu_rejects_singular_and_mismatched_inputs() {
        let singular = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]];
        assert_eq!(
            real_lu_factor(singular.view()),
            Err(LinalgError::SingularMatrix { pivot: 2 })
        );

        let non_square = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(
            real_lu_factor(non_square.view()),
            Err(LinalgError::NonSquare { rows: 2, cols: 3 })
        );

        let lu = real_lu_factor(array![[1.0, 0.0], [0.0, 1.0]].view());
        let rhs = array![[1.0], [2.0], [3.0]];
        assert!(matches!(
            lu.and_then(|lu| real_lu_solve(&lu, rhs.view())),
            Err(LinalgError::LengthMismatch { .. })
        ));

        let complex32_non_square = array![
            [Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
            [Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
            [Complex32::new(5.0, 0.0), Complex32::new(6.0, 0.0)]
        ];
        assert_eq!(
            complex32_lu_factor(complex32_non_square.view()),
            Err(LinalgError::NonSquare { rows: 3, cols: 2 })
        );
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

    #[test]
    fn real32_symmetric_2x2_helpers_match_feff_slaev2_reference() -> Result<(), LinalgError> {
        let cases = [
            (
                4.0,
                1.25,
                1.0,
                [4.452_562_3, 0.547_437_6],
                [-0.940_271_56, -0.340_425_25],
            ),
            (
                -5.0,
                2.0,
                -1.0,
                [-5.828_427_3, -0.171_572_86],
                [-0.923_879_5, 0.382_683_4],
            ),
            (2.0, 0.0, -3.0, [-3.0, 2.0], [-0.0, 1.0]),
            (0.0, 0.0, 0.0, [0.0, -0.0], [-0.0, 1.0]),
        ];

        for (diagonal_a, off_diagonal, diagonal_c, expected_values, expected_vector) in cases {
            let values = real32_symmetric_2x2_eigenvalues(diagonal_a, off_diagonal, diagonal_c)?;
            assert_f32_slice_close(&values, &expected_values);
            let eigensystem = real32_symmetric_2x2_eigen(diagonal_a, off_diagonal, diagonal_c)?;
            assert_f32_close(eigensystem.larger_abs_eigenvalue(), expected_values[0]);
            assert_f32_close(eigensystem.smaller_abs_eigenvalue(), expected_values[1]);
            assert_f32_slice_close(&eigensystem.larger_abs_eigenvector(), &expected_vector);
        }

        Ok(())
    }

    #[test]
    fn real32_symmetric_2x2_helpers_reject_non_finite_inputs() {
        assert_eq!(
            real32_symmetric_2x2_eigenvalues(f32::NAN, 0.0, 1.0),
            Err(LinalgError::NonFiniteScalar { name: "a" })
        );
        assert_eq!(
            real32_symmetric_2x2_eigen(1.0, f32::INFINITY, 0.0),
            Err(LinalgError::NonFiniteScalar { name: "b" })
        );
    }

    #[test]
    fn real32_symmetric_eigen_matches_feff_ssyev_reference() -> Result<(), LinalgError> {
        let matrix = array![
            [4.0_f32, 99.0, 99.0, 99.0],
            [-1.0, 3.0, 99.0, 99.0],
            [0.5, 1.25, 2.5, 99.0],
            [0.75, -0.5, 1.5, 1.0],
        ];

        let eigensystem = real32_symmetric_eigen(matrix.view(), SymmetricTriangle::Lower)?;
        assert_f32_vec_close(
            eigensystem.eigenvalues(),
            &[-0.301_708_43, 1.816_907, 4.142_762_7, 4.842_039],
        );
        assert_f32_matrix_abs_close(
            eigensystem.eigenvectors(),
            array![
                [0.008_082_926, -0.523_904_4, -0.127_553_43, 0.842_133_64],
                [0.328_703_9, -0.589_783_8, -0.578_473_57, -0.457_686_87],
                [-0.556_576_4, 0.343_050_42, -0.749_305_3, 0.105_265_945],
                [0.762_962_2, 0.509_897_6, -0.296_040_98, 0.265_052_56],
            ]
            .view(),
        );

        let values_only = real32_symmetric_eigenvalues(matrix.view(), SymmetricTriangle::Lower)?;
        assert_f32_vec_close(values_only.view(), &eigensystem.eigenvalues().to_vec());
        Ok(())
    }

    #[test]
    fn real32_symmetric_eigen_uses_selected_triangle_only() -> Result<(), LinalgError> {
        let lower = array![[2.0_f32, f32::NAN], [0.5, 3.0]];
        let upper = array![[2.0_f32, 0.5], [f32::NAN, 3.0]];

        let lower_values = real32_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Lower)?;
        let upper_values = real32_symmetric_eigenvalues(upper.view(), SymmetricTriangle::Upper)?;
        assert_f32_vec_close(lower_values.view(), &upper_values.to_vec());

        assert_eq!(
            real32_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Upper),
            Err(LinalgError::NonFiniteMatrixEntry { row: 0, col: 1 })
        );
        Ok(())
    }

    #[test]
    fn real32_symmetric_eigen_rejects_invalid_inputs() {
        let non_square = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(
            real32_symmetric_eigen(non_square.view(), SymmetricTriangle::Lower),
            Err(LinalgError::NonSquare { rows: 2, cols: 3 })
        );

        let non_finite = array![[1.0_f32, 0.0], [f32::INFINITY, 2.0]];
        assert_eq!(
            real32_symmetric_eigenvalues(non_finite.view(), SymmetricTriangle::Lower),
            Err(LinalgError::NonFiniteMatrixEntry { row: 1, col: 0 })
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

    fn assert_f32_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 2.0e-5,
            "actual={actual} expected={expected}"
        );
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2.0e-5,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    fn assert_f32_vec_close(actual: ArrayView1<'_, f32>, expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2.0e-5,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    fn assert_f32_matrix_abs_close(actual: ArrayView2<'_, f32>, expected: ArrayView2<'_, f32>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());
        for ((row, col), &actual) in actual.indexed_iter() {
            let expected = expected[(row, col)];
            assert!(
                (actual.abs() - expected.abs()).abs() < 2.0e-5,
                "({row},{col}) actual={actual} expected={expected}"
            );
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        assert!(
            (actual - expected).norm() < 5.0e-6,
            "actual={actual:?} expected={expected:?}"
        );
    }

    fn assert_complex_matrix_close(
        actual: ArrayView2<'_, Complex64>,
        expected: ArrayView2<'_, Complex64>,
    ) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, col), &value) in actual.indexed_iter() {
            assert_complex_close(value, expected[(row, col)]);
        }
    }

    fn assert_complex32_matrix_close(
        actual: ArrayView2<'_, Complex32>,
        expected: ArrayView2<'_, Complex32>,
    ) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, col), &value) in actual.indexed_iter() {
            assert_complex32_close(value, expected[(row, col)]);
        }
    }

    fn assert_complex_vec_close(actual: ArrayView1<'_, Complex64>, expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }
}
