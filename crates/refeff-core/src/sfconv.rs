//! FEFF SFCONV numerical helpers.
//!
//! These kernels support spectral-function convolution. The full SFCONV driver
//! also depends on spectrum file orchestration, so this module keeps the
//! reusable numerical transforms independent and directly testable.

use crate::{Real, RealVec, real_polynomial_roots};

const SFCONV_GRATER_MAX_REGIONS: usize = 1_500;
const SFCONV_GRATER_MAX_SINGULARITIES: usize = 20;
/// Legacy Bohr/Angstrom conversion used inside FEFF `SFCONV/so2conv.f90`.
pub use crate::constants::BOHR_ANGSTROM_SFCONV_LEGACY as SFCONV_SO2CONV_BOHR_ANGSTROM;
/// Legacy Hartree/eV conversion used inside FEFF `SFCONV/so2conv.f90`.
pub use crate::constants::HARTREE_EV_SFCONV_LEGACY as SFCONV_SO2CONV_HARTREE_EV;
/// Number of energy rows in FEFF `SFCONV/mkspectf.f90` spectral functions.
pub const SFCONV_MKSPECTF_GRID_LEN: usize = 112;
/// Number of FEFF `SFCONV/so2conv.f90` minimal momentum-grid rows.
pub const SFCONV_SO2CONV_MOMENTUM_GRID_LEN: usize = 66;
const SFCONV_GRATER_DX: [Real; 3] = [
    0.112_701_66_f32 as Real,
    0.5_f32 as Real,
    0.887_298_35_f32 as Real,
];
const SFCONV_GRATER_WT: [Real; 3] = [
    0.277_777_8_f32 as Real,
    0.444_444_45_f32 as Real,
    0.277_777_8_f32 as Real,
];
const SFCONV_GRATER_WT9: [Real; 9] = [
    0.061_693_88_f32 as Real,
    0.108_384_23_f32 as Real,
    0.039_846_36_f32 as Real,
    0.175_209_03_f32 as Real,
    0.229_732_99_f32 as Real,
    0.175_209_03_f32 as Real,
    0.039_846_36_f32 as Real,
    0.108_384_23_f32 as Real,
    0.061_693_88_f32 as Real,
];

mod convolution;
mod plasma;
mod quadrature;
mod self_energy;
mod so2conv;
mod spectral;
mod support;
mod types;

pub use convolution::*;
pub use plasma::*;
pub use quadrature::*;
pub use self_energy::*;
pub use so2conv::*;
pub use spectral::*;
pub use types::*;

#[cfg(test)]
mod tests;
