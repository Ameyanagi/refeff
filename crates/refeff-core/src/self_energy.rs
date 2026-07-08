//! Self-energy singularity helpers ported from FEFF.
//!
//! `SELF/fndsng.f90` finds real singularities of the Hedin-Lundqvist
//! self-energy integrands by solving the FEFF cubic and quadratic equations,
//! filtering roots to the integration window, and sorting the accepted values.
//! This module also ports the small `SELF/omegaq.f90` dispersion helpers and
//! the `SELF/logi.f90` logarithm branch helper used by the BPR integrands.

use ndarray::ArrayView1;

use crate::{
    Complex, InterpolationError, Real, RootError, SpecialFunctionError, cubic_zeros,
    quadratic_zeros, terp, x_log_x,
};

const SINGULARITY_TOLERANCE: Real = 1.0e-4;
const MKEXC_FINE_POINTS: usize = 50_000;
const MKEXC_WIDTH_EV: Real = 0.1;
const SELF_ENERGY_LOG_SHIFT: Real = -1.0e-10;
const SELF_ENERGY_ZERO_PL: Real = 1.0e-5;
const SELF_ENERGY_INF: Real = 1.0e2;
const SELF_ENERGY_ABS_ERR: Real = 1.0e-5;
const SELF_ENERGY_REL_ERR: Real = 1.0e-4;
const SELF_ENERGY_FERMI_MOMENTUM_FACTOR: Real = 1.919_158_292_677_512_8;
const CGRATR_MAX_REGIONS: usize = 1_500;
const CGRATR_MAX_SINGULARITIES: usize = 20;
const CGRATR_DX: [Real; 3] = [
    0.112_701_66_f32 as Real,
    0.5_f32 as Real,
    0.887_298_35_f32 as Real,
];
const CGRATR_WT: [Real; 3] = [
    0.277_777_8_f32 as Real,
    0.444_444_45_f32 as Real,
    0.277_777_8_f32 as Real,
];
const CGRATR_WT9: [Real; 9] = [
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

mod bpr;
mod cgratr;
mod csigz;
mod kernels;
mod poles;
mod singularities;
mod support;
mod types;

pub use bpr::{self_energy_bpr1_integrand, self_energy_bpr2_integrand, self_energy_bpr3_integrand};
pub use cgratr::cgratr;
pub use csigz::{
    many_pole_self_energy, self_energy_single_pole, self_energy_single_pole_derivative,
};
pub use kernels::{
    gamma_q, hartree_fock_exchange, log_i, omega_q, self_energy_dr1_integrand,
    self_energy_dr2_integrand, self_energy_dr3_integrand, self_energy_pole_dispersion,
    self_energy_r1_integrand, self_energy_r2_integrand, self_energy_r3_integrand,
};
pub use poles::make_excitation_poles;
pub use singularities::find_self_energy_singularities;
pub use types::*;

use support::*;

#[cfg(test)]
mod tests;
