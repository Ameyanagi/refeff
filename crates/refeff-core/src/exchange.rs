//! Exchange and self-energy scalar helpers from FEFF.
//!
//! This module ports small routines from `EXCH/`: the Dirac-Hara
//! energy-dependent exchange potential (`edp`), the Von Barth-Hedin spin
//! potential (`vbh`), Perdew-Zunger and Perrot-Dharma-Wardana LDA potentials,
//! Karasiev-Sjostrom-Dufty-Trickey finite-temperature LDA potentials, and the
//! Hedin-Lundqvist helper function `ffq`.

use thiserror::Error;

use crate::{Complex, Real};

const FEFF_FA: Real = 1.919_158_292_677_512_8;
const FEFF_PI: Real = std::f64::consts::PI;

mod hedin;
mod potentials;
mod support;
mod thermal;
mod types;
mod xcpot;

pub use hedin::{
    hedin_lundqvist_ffq, hedin_lundqvist_imaginary_self_energy, hedin_lundqvist_self_energy,
    quinn_imaginary_self_energy,
};
pub use potentials::{
    dirac_hara_exchange_potential, perdew_zunger_exchange_correlation, perdew_zunger_vxc,
    von_barth_hedin_potential,
};
pub(in crate::exchange) use support::*;
pub use thermal::{
    karasiev_sjostrom_dufty_trickey_free_energy, karasiev_sjostrom_dufty_trickey_internal_energy,
    karasiev_sjostrom_dufty_trickey_vxc, perrot_dharma_wardana_reduced_vxc,
    perrot_dharma_wardana_vxc,
};
pub use types::*;
pub use xcpot::{
    xcpot, xcpot_apply_self_energy_deltas, xcpot_fermi_cache, xcpot_ground_state_branch,
    xcpot_local_scales, xcpot_many_pole_control, xcpot_many_pole_delta_table,
    xcpot_many_pole_density_grid, xcpot_many_pole_row_delta,
    xcpot_many_pole_self_energy_delta_table, xcpot_reference_shift, xcpot_self_energy_correction,
    xcpot_sigma,
};

#[cfg(test)]
mod tests;
