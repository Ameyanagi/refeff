use ndarray::ArrayView2;
use num_complex::{Complex32, Complex64};

use crate::error::LinalgError;
use crate::types::SymmetricTriangle;

pub(crate) fn ensure_square(matrix: ArrayView2<'_, f64>) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

pub(crate) fn ensure_complex_square(matrix: ArrayView2<'_, Complex64>) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

pub(crate) fn ensure_complex32_square(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<(), LinalgError> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows != cols {
        return Err(LinalgError::NonSquare { rows, cols });
    }
    Ok(())
}

pub(crate) fn ensure_complex32_finite_square(
    matrix: ArrayView2<'_, Complex32>,
) -> Result<(), LinalgError> {
    ensure_complex32_square(matrix)?;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(LinalgError::NonFiniteMatrixEntry { row, col });
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_real32_symmetric_input(
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

pub(crate) fn ensure_real64_symmetric_input(
    matrix: ArrayView2<'_, f64>,
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

pub(crate) fn ensure_finite_f32(name: &'static str, value: f32) -> Result<(), LinalgError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LinalgError::NonFiniteScalar { name })
    }
}
