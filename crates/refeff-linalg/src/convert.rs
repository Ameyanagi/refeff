use faer::Mat;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

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
