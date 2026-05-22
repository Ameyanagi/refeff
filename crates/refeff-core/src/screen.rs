//! FEFF SCREEN helper kernels.
//!
//! These routines cover small, self-contained pieces from `SCREEN/frgrid.f90`,
//! `SCREEN/fegrid.f90`, `SCREEN/fxc.f90`, and the response setup blocks in
//! `SCREEN/screensub.f90` and `CRPA/chi_crpa.f90`, plus the compact CRPA radial
//! density setup block. The full SCREEN/CRPA drivers also depend on phase,
//! potential, and FMS handoff state; keeping these kernels separate makes them
//! usable and testable while those drivers are ported incrementally.

mod constants;
mod grids;
mod potentials;
mod radial_solution;
mod response;
mod setup;
mod types;
mod validation;

pub use constants::*;
pub use grids::*;
pub use potentials::*;
pub use radial_solution::*;
pub use response::*;
pub use setup::*;
pub use types::*;

#[cfg(test)]
use crate::RealVec;

#[cfg(test)]
mod tests;
