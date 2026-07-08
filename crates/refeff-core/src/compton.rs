//! FEFF COMPTON grid, rotation, and profile helpers.
//!
//! This module ports the compact numerical kernels from
//! `COMPTON/m_rotation.f90` and the `compton_build_grid`/`jpq` routines in
//! `COMPTON/m_compton.f90`. The routines preserve FEFF's grid and Fourier
//! transform formulas while replacing implicit NaN/Inf behavior with typed
//! validation errors.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ShapeBuilder};
use num_complex::Complex64;
use thiserror::Error;

use crate::{Real, RealMat, RealVec, Vector3};

const COMPTON_ROTATION_TOLERANCE: Real = 1.0e-10;
const ROTATION_RATIO_TOLERANCE: Real = 1.0e-12;

mod density;
mod grid;
mod profile;
mod rotation;
mod support;
mod types;

pub use density::{
    compton_jzzp, compton_jzzp_from_rhorrp, compton_rhozzp_slice, compton_rhozzp_slice_from_rhorrp,
};
pub use grid::compton_build_grid;
pub use profile::{compton_profile, compton_profiles};
pub use rotation::{
    compton_cross_product, compton_rotate_vector, compton_rotate_vector_in_place,
    compton_rotation_axis_angle, compton_rotation_matrix,
};
pub(in crate::compton) use support::*;
pub use types::*;

#[cfg(test)]
mod tests;
