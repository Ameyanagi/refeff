//! FEFF RHORRP grid and atom-localization helpers.
//!
//! The full `RHORRP/m_rhorrp.f90` density-matrix calculation depends on the
//! potential, phase, and FMS handoff data. This module starts with the compact
//! support routines used by that calculation and by `RHORRP/rhorrp.f90` output:
//! FEFF-order density-grid traversal, nearest-atom selection, radial
//! wavefunction interpolation, and contour Fermi occupations.

mod constants;
mod density;
mod greens;
mod grid;
mod integration;
mod nearest;
mod radial;
mod types;
mod validation;

pub use density::*;
pub use greens::*;
pub use grid::*;
pub use integration::*;
pub use nearest::*;
pub use radial::*;
pub use types::*;

#[cfg(test)]
mod tests;
