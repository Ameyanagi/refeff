//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels for final-state planning,
//! angular coefficient tables, NRIXS transition weights, q-Bessel tables,
//! initial-state occupation normalization, and angular-decomposition spectrum
//! updates, plus phase-mesh primitive, FEFF84 grid construction, and
//! `grid.inp` user-grid phase mesh composition.

use ndarray::{Array1, Array2, Array5, ArrayView1, ShapeBuilder};
use refeff_linalg::feff_determinant;

use crate::{
    Complex, Real, spherical_bessel_j_y, terp, wigner_3j,
    xsph_occ_norm::{
        XSPH_OCC_NORM_ATOMIC_NUMBER_MAX, XSPH_OCC_NORM_HOLE_COUNT, xsph_occ_norm_denominator,
        xsph_occ_norm_numerator,
    },
};

const QBESSEL_MAX_LJ: usize = 39;
const QBESSEL_ZERO_CUTOFF: Real = 1.0e8;
const CWIG3J_MAX_DOUBLED_ARGUMENT: i32 = 116;
const XSPH_MAX_LX: usize = 20;
const XSPH_HARTREE_EV: Real = 27.211_396;
const XSPH_BOHR_ANGSTROM: Real = 0.529_177_249;
const XSPH_HOLE_ORBITAL_X0: Real = 8.80;
const XSPH_HOLE_ORBITAL_TAIL_CUTOFF: Real = 1.0e-11;
const XSPH_PHASE_SORT_TOLERANCE: Real = 0.001;
const XSPH_USER_PHASE_GRID_MAX_RECORDS: usize = 10;

mod angular;
mod axafs;
mod mesh;
mod orbital;
mod planning;
mod spectrum;
mod support;
mod types;

pub use angular::{
    xsph_angular_density_coefficients, xsph_longitudinal_multipole_factor,
    xsph_relativistic_multipole_factors,
};
pub use axafs::xsph_axafs;
pub use mesh::*;
pub use orbital::xsph_initial_hole_orbital;
pub use planning::{
    xsph_lj_needed_flags, xsph_minimize_calculations, xsph_occupation_normalization,
    xsph_q_bessel_table,
};
pub use spectrum::*;
pub(in crate::xsph) use support::*;
pub use types::*;

#[cfg(test)]
mod tests;
