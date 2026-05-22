//! FEFF density and LDOS accumulation helpers.
//!
//! This module ports compact numerical routines that update radial valence
//! densities and angular-momentum-resolved density of states after scattering
//! terms have been computed.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use num_complex::Complex32;

use crate::grid::{
    CoulombPotentialSlwInput, GridError, LoucksSphericalOverlapInput, NormanRadius,
    NormanRadiusInput, coulomb_potential_slw, norman_radius_from_density,
    sum_loucks_spherical_overlap,
};
use crate::quadrature::{QuadratureError, somm2};
use crate::vector::distance_between;
use crate::{Complex, Real};

const OVRLP_DENSITY_POINTS: usize = 251;
const OVRLP_GEOMETRY_CUTOFF: Real = 12.0;
const COULOM_DELTA: Real = 0.05;
const COULOM_LITERAL_DELTA: Real = 0.05_f32 as Real;
const COULOM_LITERAL_OFFSET: Real = 8.8_f32 as Real;
const BROYDN_DELTA: Real = 0.05;
const BROYDN_LITERAL_OFFSET: Real = 8.8_f32 as Real;

mod broyden;
mod coulomb;
mod overlap;
mod support;
mod types;
mod valence;

pub use broyden::mix_broyden_density;
pub use coulomb::update_coulomb_potential;
pub use overlap::overlap_potential_density;
pub use types::*;
pub use valence::update_valence_density;

use support::*;

#[cfg(test)]
mod tests;
