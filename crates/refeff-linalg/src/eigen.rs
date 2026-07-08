use faer::{Mat, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;

use crate::error::LinalgError;
use crate::types::SymmetricTriangle;
use crate::validation::{
    ensure_complex32_finite_square, ensure_finite_f32, ensure_real32_symmetric_input,
    ensure_real64_symmetric_input,
};

/// Single-precision symmetric eigensystem from FEFF `SSYEV`.
#[derive(Debug, Clone, PartialEq)]
pub struct Real32SymmetricEigen {
    eigenvalues: Array1<f32>,
    eigenvectors: Array2<f32>,
}

/// Double-precision symmetric eigensystem.
#[derive(Debug, Clone, PartialEq)]
pub struct Real64SymmetricEigen {
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
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

impl Real64SymmetricEigen {
    /// Eigenvalues sorted in nondecreasing order.
    #[must_use]
    pub fn eigenvalues(&self) -> ArrayView1<'_, f64> {
        self.eigenvalues.view()
    }

    /// Orthonormal eigenvectors stored column-wise.
    #[must_use]
    pub fn eigenvectors(&self) -> ArrayView2<'_, f64> {
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

/// General single-complex eigenvalues for FEFF `CGEES`-style call sites.
///
/// FEFF's BAND path requests eigenvalues only (`JOBVS='N'`, `SORT='N'`) from
/// LAPACK `CGEES`. This wrapper performs the same no-eigenvector calculation
/// through `faer`; callers that need FEFF-specific ordering should sort the
/// returned values explicitly.
pub fn complex32_general_eigenvalues(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<Array1<Complex32>, LinalgError> {
    ensure_complex32_finite_square(matrix)?;
    if matrix.nrows() == 0 {
        return Ok(Array1::zeros(0));
    }

    let faer_matrix = Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    });
    faer_matrix
        .eigenvalues()
        .map(Array1::from_vec)
        .map_err(|_| LinalgError::EigenDidNotConverge)
}

/// Symmetric double-precision eigenvalues through the pure-Rust `faer` backend.
///
/// The selected triangle is mirrored into a full self-adjoint matrix before
/// calling `faer`, matching the triangle semantics used by the FEFF LAPACK
/// call sites.
pub fn real64_symmetric_eigenvalues(
    matrix: ArrayView2<'_, f64>,
    triangle: SymmetricTriangle,
) -> Result<Array1<f64>, LinalgError> {
    ensure_real64_symmetric_input(matrix, triangle)?;
    if matrix.nrows() == 0 {
        return Ok(Array1::zeros(0));
    }

    let faer_matrix = real64_symmetric_to_faer(matrix, triangle);
    faer_matrix
        .self_adjoint_eigenvalues(Side::Lower)
        .map(Array1::from_vec)
        .map_err(|_| LinalgError::EigenDidNotConverge)
}

/// Symmetric double-precision eigensystem through the pure-Rust `faer` backend.
///
/// Eigenvectors are returned column-wise and eigenvalues are sorted in
/// nondecreasing order.
pub fn real64_symmetric_eigen(
    matrix: ArrayView2<'_, f64>,
    triangle: SymmetricTriangle,
) -> Result<Real64SymmetricEigen, LinalgError> {
    ensure_real64_symmetric_input(matrix, triangle)?;
    if matrix.nrows() == 0 {
        return Ok(Real64SymmetricEigen {
            eigenvalues: Array1::zeros(0),
            eigenvectors: Array2::zeros((0, 0)),
        });
    }

    let faer_matrix = real64_symmetric_to_faer(matrix, triangle);
    let decomposition = faer_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| LinalgError::EigenDidNotConverge)?;
    let eigenvalues = Array1::from_iter(decomposition.S().column_vector().iter().copied());
    let eigenvectors_ref = decomposition.U();
    let eigenvectors = Array2::from_shape_fn(
        (eigenvectors_ref.nrows(), eigenvectors_ref.ncols()),
        |(row, col)| eigenvectors_ref[(row, col)],
    );

    Ok(Real64SymmetricEigen {
        eigenvalues,
        eigenvectors,
    })
}

fn real32_symmetric_to_faer(matrix: ArrayView2<'_, f32>, triangle: SymmetricTriangle) -> Mat<f32> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        triangle.selected_entry(matrix, row, col)
    })
}

fn real64_symmetric_to_faer(matrix: ArrayView2<'_, f64>, triangle: SymmetricTriangle) -> Mat<f64> {
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
