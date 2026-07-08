//! FULLSPECTRUM constants mirrored from FEFF.

use crate::Real;

/// FEFF Bohr radius in Angstrom, matching `COMMON/m_constants.f90`.
pub use crate::constants::BOHR_ANGSTROM as FEFF_BOHR_ANGSTROM;
/// FEFF Hartree energy in eV, matching `COMMON/m_constants.f90`.
pub use crate::constants::HARTREE_EV as FEFF_HARTREE_EV;
/// Inverse fine-structure constant used by FEFF optical sum rules.
pub const FEFF_ALPHA_INV: Real = 137.035_989_56;
/// Reduced Planck constant in eV seconds used by `FULLSPECTRUM/drdtrm.f90`.
pub const FEFF_HBAR_EV_SECONDS: Real = 6.58E-16;
/// FEFF lower energy floor for `FULLSPECTRUM/egrid_lin.f90`, in Hartree.
pub const FEFF_FULLSPECTRUM_MIN_LINEAR_ENERGY: Real = 0.01 / FEFF_HARTREE_EV;
/// FEFF lower energy floor for `FULLSPECTRUM/egrid.f90`, in Hartree.
pub const FEFF_FULLSPECTRUM_MIN_EDGE_GRID_ENERGY: Real = 0.001 / FEFF_HARTREE_EV;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` default k-space step.
pub const FEFF_FULLSPECTRUM_XK_STEP: Real = 0.005;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` energy-grid capacity.
pub const FEFF_FULLSPECTRUM_GRID_CAPACITY: usize = 200_001;
/// FEFF `FULLSPECTRUM/gtedgs.f90` DOS-convolution edge threshold in Hartree.
pub const FEFF_FULLSPECTRUM_CONVOLUTION_EDGE_HARTREE: Real = 1.837_465_5;
/// Number of core-hole slots scanned by FEFF `FULLSPECTRUM/gtedgs.f90`.
pub const FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT: usize = 40;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` lower sum-rule grid bound.
pub const FEFF_FULLSPECTRUM_BACKGROUND_SUM_MIN: Real = 0.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` upper sum-rule grid bound.
pub const FEFF_FULLSPECTRUM_BACKGROUND_SUM_MAX: Real = 18_383.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` lower path-expansion transition k.
pub const FEFF_FULLSPECTRUM_FINE_STRUCTURE_LOW_K: Real = 3.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` upper FMS transition k.
pub const FEFF_FULLSPECTRUM_FINE_STRUCTURE_HIGH_K: Real = 4.0;
/// FEFF `FULLSPECTRUM/HEADERS/params.h` entry transition size, in Hartree.
pub const FEFF_FULLSPECTRUM_EDGE_TRANSITION_SIZE: Real = 0.05;
/// FEFF `FULLSPECTRUM/addedg.f90` multiplier for the imaginary exit transition.
pub const FEFF_FULLSPECTRUM_IMAGINARY_EXIT_MULTIPLIER: usize = 10;
/// FEFF `FULLSPECTRUM/rdop.f90` lower padding before selected edges, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_LOW_PADDING_EV: Real = 50.0;
/// FEFF `FULLSPECTRUM/rdop.f90` upper padding after selected edges, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_HIGH_PADDING_EV: Real = 1000.0;
/// FEFF `FULLSPECTRUM/rdop.f90` lower default energy-grid floor, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_MIN_EV: Real = 0.1;
/// FEFF `FULLSPECTRUM/rdop.f90` default point-count spacing, in eV.
pub const FEFF_FULLSPECTRUM_DEFAULT_EDGE_STEP_EV: Real = 0.5;

pub(super) const FEFF_FULLSPECTRUM_EDGE_LABELS: [&str; FEFF_FULLSPECTRUM_EDGE_SLOT_COUNT] = [
    "K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5", "N1", "N2", "N3", "N4", "N5", "N6", "N7",
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3",
];
