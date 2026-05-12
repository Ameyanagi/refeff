#![forbid(unsafe_code)]

//! Linear algebra bridge for the FEFF10 Rust port.
//!
//! FEFF module state is stored in `ndarray`; performance-critical matrix
//! operations are delegated to pure-Rust `faer` through this small adapter layer.

use faer::Mat;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

pub fn real_to_faer(view: ArrayView2<'_, f64>) -> Mat<f64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

pub fn complex_to_faer(view: ArrayView2<'_, Complex64>) -> Mat<Complex64> {
    Mat::from_fn(view.nrows(), view.ncols(), |row, col| view[(row, col)])
}

pub fn real_from_faer(matrix: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

pub fn complex_from_faer(matrix: &Mat<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |(row, col)| {
        matrix[(row, col)]
    })
}

pub fn real_matmul(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>) -> Array2<f64> {
    let lhs = real_to_faer(lhs);
    let rhs = real_to_faer(rhs);
    real_from_faer(&(&lhs * &rhs))
}

pub fn complex_matmul(
    lhs: ArrayView2<'_, Complex64>,
    rhs: ArrayView2<'_, Complex64>,
) -> Array2<Complex64> {
    let lhs = complex_to_faer(lhs);
    let rhs = complex_to_faer(rhs);
    complex_from_faer(&(&lhs * &rhs))
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
}
