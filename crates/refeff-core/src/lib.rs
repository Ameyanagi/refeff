#![forbid(unsafe_code)]

//! Core numerical types for the FEFF10 Rust port.
//!
//! `ndarray` is the primary storage and view API. Helpers in this crate create
//! arrays with FEFF-friendly Fortran-order layout where the original algorithms
//! and file formats depend on column-major traversal.

use ndarray::{Array1, Array2, Array3, Array4, ShapeBuilder};
use num_complex::Complex64;

pub type Real = f64;
pub type Complex = Complex64;

pub type RealVec = Array1<Real>;
pub type RealMat = Array2<Real>;
pub type RealCube = Array3<Real>;
pub type RealArray4 = Array4<Real>;

pub type ComplexVec = Array1<Complex>;
pub type ComplexMat = Array2<Complex>;
pub type ComplexCube = Array3<Complex>;
pub type ComplexArray4 = Array4<Complex>;

pub fn zeros_real_vec(len: usize) -> RealVec {
    Array1::zeros(len)
}

pub fn zeros_complex_vec(len: usize) -> ComplexVec {
    Array1::zeros(len)
}

pub fn zeros_real_mat_fortran(rows: usize, cols: usize) -> RealMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_complex_mat_fortran(rows: usize, cols: usize) -> ComplexMat {
    Array2::zeros((rows, cols).f())
}

pub fn zeros_real_cube_fortran(a: usize, b: usize, c: usize) -> RealCube {
    Array3::zeros((a, b, c).f())
}

pub fn zeros_complex_cube_fortran(a: usize, b: usize, c: usize) -> ComplexCube {
    Array3::zeros((a, b, c).f())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeffDimensions {
    pub nph: usize,
    pub ne: usize,
    pub nsp: usize,
    pub lmax: usize,
}

impl FeffDimensions {
    pub fn phase_shape(self) -> (usize, usize, usize, usize) {
        (self.ne, 2 * self.lmax + 1, self.nsp, self.nph + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_fortran_order_matrix_storage() {
        let mat = zeros_real_mat_fortran(3, 2);
        assert_eq!(mat.shape(), &[3, 2]);
        assert_eq!(mat.strides(), &[1, 3]);
    }

    #[test]
    fn computes_phase_shape_with_absorber_potential() {
        let dims = FeffDimensions {
            nph: 2,
            ne: 10,
            nsp: 1,
            lmax: 3,
        };
        assert_eq!(dims.phase_shape(), (10, 7, 1, 3));
    }
}
