//! FEFF `phase.bin` text/PAD phase-shift codec.
//!
//! `XSPH/wrxsph.f90` writes this handoff file for downstream FMS and FF2X
//! stages. The file is formatted text: a fixed-width integer header, a small
//! real PAD block, and several complex PAD blocks. This module preserves that
//! order while exposing phase shifts and transition moments as `ndarray`
//! values.

mod band;
mod common;
mod genfmt;
mod parse;
mod path;
mod render;
mod rhorrp;
mod rixs;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use band::{
    PhaseBinBandData, PhaseBinBandSearchSetup, band_search_setup_from_handoffs,
    band_search_setup_from_handoffs_with_lmaxph, phase_bin_band_data,
};
pub use genfmt::{
    PhaseBinGenfmtData, genfmt_core_legendre_normalization_from_feff_dims,
    genfmt_driver_setup_from_handoffs, genfmt_edge_start_index_from_phase,
    genfmt_jas_driver_setup_from_handoffs, genfmt_jas_path_setups_from_handoffs,
    genfmt_jas_q_angles_from_handoffs, genfmt_jas_transition_indices_from_handoffs,
    genfmt_jas_transition_setups_from_handoff_setups, genfmt_legendre_normalization_from_feff_dims,
    genfmt_nstar_driver_input_from_handoffs, genfmt_nstar_rows_from_handoffs,
    genfmt_ordinary_path_setups_from_handoffs, genfmt_ordinary_spin_radial_factors_from_phase,
    genfmt_ordinary_transition_b_matrix_from_handoffs,
    genfmt_ordinary_transition_matrices_from_handoff_setups, phase_bin_genfmt_data,
};
pub use parse::parse_phase_bin;
pub use path::{
    PhaseBinPathHandoff, path_phase_criteria_tables_from_phase_bin,
    phase_bin_path_handoff_from_phase_bin,
};
pub use render::{phase_bin_string, read_phase_bin, write_phase_bin};
pub use rhorrp::{
    RhorrpPhaseBinHandoff, rhorrp_phase_handoff_from_phase_bin, rhorrp_phase_table_from_phase_bin,
};
pub use rixs::{
    PhaseBinRixsHandoff, phase_bin_rixs_handoff_from_phase_bin,
    phase_bin_rixs_transition_phase_shifts_from_handoff,
    phase_bin_rixs_transition_setup_from_handoffs, rixs_angular_limits_from_phase_bin,
    rixs_transition_moments_from_phase_bin,
};
pub use types::{
    PHASE_BIN_DEFAULT_PAD_WIDTH, PHASE_BIN_DEFAULT_TRANSITION_COUNT, PHASE_BIN_SCALARS,
    PhaseBinData, PhaseBinPotential, PhaseBinRawPads, PhaseBinScalars,
};
