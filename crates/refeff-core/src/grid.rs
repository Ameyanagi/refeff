//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`,
//! `m_ifuns.f90`, radial resampling helpers from `COMMON/`, and the ATOM
//! `FixAtomicQuantities` resampling helper from `ATOM/scfdat.f90`. FEFF uses
//! a 1-based logarithmic radial grid with `x = -8.8 + (j - 1) * delta` and
//! `r = exp(x)`.

use std::f64::consts::PI;

use crate::interpolation::terp;
use crate::quadrature::somm2;
use crate::vector::distance_between;
use crate::{Complex, Real};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;
use refeff_linalg::{Complex32Lu, LinalgError, complex32_lu_factor};

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

/// FEFF Hartree constant in eV, from `COMMON/m_constants.f90`.
pub const FEFF_HARTREE_EV: Real = 27.211_396;

/// FEFF Fermi-momentum factor `(9*pi/4)^(1/3)`, from `COMMON/m_constants.f90`.
pub const FEFF_FERMI_MOMENTUM_FACTOR: Real = 1.919_158_292_677_512_8;

mod density;
mod overlap;
mod potential;
mod radial;
mod resample;
mod types;
mod validation;

pub use density::*;
pub use overlap::*;
pub use potential::*;
pub use radial::{
    loucks_index_below, loucks_radius, loucks_x, radial_index_below, radial_radius, radial_x,
    wave_number_from_hartree,
};
pub use resample::*;
pub use types::*;

#[cfg(test)]
use radial::{feff_legacy_loucks_radius, feff_legacy_loucks_x};

const SPINOR_ZERO_THRESHOLD: Real = 1.0e-11;
const SUMAX_WIGNER_SEITZ_RADIUS: Real = 15.0;
const SUMAX_LITERAL_DELTA: Real = 0.05_f32 as Real;
const SUMAX_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const SIDX_DENSITY_CUTOFF: Real = 1.0e-5;
const FRNRM_DENSITY_POINTS: usize = 251;
const FRNRM_NRPTX: usize = 1251;
const FRNRM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const FRNRM_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const FRNRM_CORRECTION_THRESHOLD: Real = 0.0001_f32 as Real;
const MOVRLP_NOVP: usize = 50;

#[cfg(test)]
mod tests;
