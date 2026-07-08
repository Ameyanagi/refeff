//! SCREEN constants mirrored from FEFF.

use crate::Real;

/// FEFF inverse fine-structure constant from `COMMON/m_constants.f90`.
pub const SCREEN_ALPHA_INVERSE: Real = 137.035_989_56;
/// FEFF fine-structure constant `alphfs`.
pub const SCREEN_FINE_STRUCTURE_ALPHA: Real = 1.0 / SCREEN_ALPHA_INVERSE;
/// FEFF Bohr radius in Angstrom, `bohr` from `COMMON/m_constants.f90`.
pub use crate::constants::BOHR_ANGSTROM as SCREEN_BOHR_ANGSTROM;
/// FEFF Hartree energy in eV, `hart` from `COMMON/m_constants.f90`.
pub use crate::constants::HARTREE_EV as SCREEN_HARTREE_EV;
