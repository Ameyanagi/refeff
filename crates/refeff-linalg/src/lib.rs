#![forbid(unsafe_code)]

//! Linear algebra bridge for the FEFF10 Rust port.
//!
//! FEFF module state is stored in `ndarray`; performance-critical matrix
//! operations are delegated to pure-Rust `faer` through this adapter layer.

mod convert;
mod eigen;
mod error;
mod least_squares;
mod lu;
mod matrix;
mod types;
mod validation;

pub use convert::{
    complex_from_faer, complex_matmul, complex_to_faer, complex32_from_faer, complex32_to_faer,
    real_from_faer, real_matmul, real_to_faer,
};
pub use eigen::{
    Real32Symmetric2x2Eigen, Real32SymmetricEigen, Real64SymmetricEigen,
    complex32_general_eigenvalues, real32_symmetric_2x2_eigen, real32_symmetric_2x2_eigenvalues,
    real32_symmetric_eigen, real32_symmetric_eigenvalues, real64_symmetric_eigen,
    real64_symmetric_eigenvalues,
};
pub use error::LinalgError;
pub use least_squares::{complex_least_squares_normal_eq, complex_polyfit, complex_polyval};
pub use lu::{
    Complex32FaerLu, Complex32Lu, ComplexLu, RealLu, complex_lu_factor, complex_lu_solve,
    complex_lu_solve_vector, complex32_faer_lu_factor, complex32_faer_lu_solve,
    complex32_faer_lu_solve_in_place, complex32_lu_factor, complex32_lu_solve,
    complex32_lu_solve_vector, real_lu_factor, real_lu_solve, real_lu_solve_vector,
};
pub use matrix::{feff_determinant, feff_inverse};
pub use types::SymmetricTriangle;

/// Bound the number of threads `faer`'s solvers and matrix products use.
///
/// `Some(1)` selects `faer::Par::Seq` for a fully deterministic, single-threaded
/// run (used for golden-output validation and reproducible HPC batch jobs).
/// `Some(n)` for `n > 1` bounds the global `rayon` pool `faer` draws from to
/// `n` threads. `None` (or `Some(0)`) restores `faer`'s default of drawing on
/// however many threads `rayon` reports as available.
///
/// This sets `faer`'s *global* parallelism setting and therefore affects every
/// `faer`-backed call in the process, matching how FEFF10's own MPI/OpenMP
/// thread controls are process-wide.
pub fn set_parallelism(threads: Option<usize>) {
    let par = match threads {
        Some(1) => faer::Par::Seq,
        Some(n) => faer::Par::rayon(n),
        None => faer::Par::rayon(0),
    };
    faer::set_global_parallelism(par);
}

#[cfg(test)]
mod tests;
