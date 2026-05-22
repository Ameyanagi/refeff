//! FEFF `GENFMT` helper routines.
//!
//! This module ports small, self-contained setup routines used by FEFF's
//! curved-wave multiple-scattering formatter. `lambda_indices` is the Rust
//! equivalent of `GENFMT/setlam.f90`: it builds the Rehr-Albers lambda index
//! arrays `(m, n)` from FEFF's `icalc` mode, path order, and dimension limits.

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView6,
    ShapeBuilder,
};

use crate::{Complex, Real};

mod lambda;
mod matrices;
mod path;
mod polynomial;
mod rotation;
mod types;
mod validation;
mod xstar;

pub use lambda::lambda_indices;
pub use matrices::{
    energy_independent_transition_matrix, polarized_scattering_amplitude_matrix,
    scattering_amplitude_matrix,
};
pub use path::path_rotation_angles;
pub use polynomial::{curved_wave_polynomials, genfmt_legendre_normalization_table};
pub use rotation::initial_state_rotation;
pub use types::*;
pub use xstar::xstar;

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;
const RDPATH_EPSILON: Real = 1.0e-6;
const SNLM_AFAC: Real = 1.0 / 64.0;
const SNLM_FACTORIAL_LIMIT: usize = 210;
const SNLM_FACTORIAL_COUNT: usize = SNLM_FACTORIAL_LIMIT + 1;

#[cfg(test)]
mod tests;
