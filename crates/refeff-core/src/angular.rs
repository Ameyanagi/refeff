//! Angular-momentum normalization helpers.
//!
//! FEFF stores associated-Legendre normalization factors in `xnlm`; FMS uses
//! `xnlm(m,l)` while GENFMT carries the same values in a one-based table. The
//! helpers here compute the shared value
//! `sqrt((2l+1) * (l-m)! / (l+m)!)`.

use ndarray::{Array2, Array3, Array4, Array6, ArrayView2, ShapeBuilder};
use refeff_linalg::complex_matmul;

use crate::{Complex, ComplexMat, ComplexVec, Real, RealMat, RealVec};

mod basis;
mod coupling;
mod harmonics;
mod polarization;
mod support;
mod transition;
mod types;
mod wigner;

pub use basis::{basis_transform_matrices, change_basis_representation};
pub use coupling::{
    mkgtr_clebsch_gordan_coefficients, relativistic_clebsch_gordan_coefficients,
    relativistic_state_index_1based, spin_orbit_coupling_tables,
};
pub use harmonics::{
    legendre_normalization, legendre_normalization_table, legendre_polynomials,
    legendre_polynomials_into, spherical_harmonics,
};
pub use polarization::polarization_tensor;
pub use transition::transition_b_matrix;
pub use types::{
    AngularError, BasisTransformMatrices, BasisTransformMode, PolarizationTensorMode,
    RelativisticClebschGordanCoefficients, SpinOrbitCouplingTables, TransitionBMatrix,
    TransitionBMatrixInput,
};
pub use wigner::{wigner_3j, wigner_rotation};

use support::*;

#[cfg(test)]
mod tests;
