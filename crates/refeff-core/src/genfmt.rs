//! FEFF `GENFMT` helper routines.
//!
//! This module ports small, self-contained setup routines used by FEFF's
//! curved-wave multiple-scattering formatter. `lambda_indices` is the Rust
//! equivalent of `GENFMT/setlam.f90`: it builds the Rehr-Albers lambda index
//! arrays `(m, n)` from FEFF's `icalc` mode, path order, and dimension limits.

use ndarray::{
    Array1, Array2, Array3, Array4, Array5, ArrayView1, ArrayView2, ArrayView3, ArrayView4,
    ArrayView5, ArrayView6, ShapeBuilder,
};

use crate::{Complex, Real};

mod lambda;
mod matrices;
mod path;
mod polynomial;
mod q;
mod rotation;
mod trace;
mod types;
mod validation;
mod xstar;

pub use lambda::lambda_indices;
pub use matrices::{
    energy_independent_transition_matrix, genfmt_jas_transition_matrices,
    genfmt_ordinary_transition_matrices, jas_left_right_amplitude_matrices,
    jas_one_sided_transition_matrices, jas_scattering_amplitude_matrices,
    jas_spin_transition_matrix, polarized_scattering_amplitude_matrix, scattering_amplitude_matrix,
};
pub use path::{genfmt_jas_path_setup, genfmt_ordinary_path_setup, path_rotation_angles};
pub use polynomial::{curved_wave_polynomials, genfmt_legendre_normalization_table};
pub use q::jas_q_angles;
pub use rotation::{genfmt_path_rotation_tables, initial_state_rotation};
pub use trace::{
    genfmt_central_phase_shifts, genfmt_chi_amplitude_phase, genfmt_curved_wave_leg_limits,
    genfmt_curved_wave_path_factor, genfmt_curved_wave_polynomial_tables,
    genfmt_decomposed_chi_amplitude_phase, genfmt_driver_setup, genfmt_feff_bin_header,
    genfmt_jas_driver_output, genfmt_jas_driver_setup, genfmt_jas_effective_initial_j,
    genfmt_jas_energy_grid_branch_from_transition_setup, genfmt_jas_left_right_path_trace,
    genfmt_jas_path_energy_grid, genfmt_jas_path_energy_grid_finalization,
    genfmt_jas_path_energy_grid_from_setup, genfmt_jas_path_energy_point,
    genfmt_jas_path_evaluation, genfmt_jas_path_evaluation_from_driver_setup,
    genfmt_jas_path_evaluation_from_setup, genfmt_jas_path_finalization, genfmt_jas_path_outputs,
    genfmt_jas_path_sequence, genfmt_jas_path_sequence_from_driver_setup,
    genfmt_jas_path_sequence_from_setup, genfmt_jas_path_signal, genfmt_jas_path_signals,
    genfmt_jas_path_trace, genfmt_jas_spherical_path_trace, genfmt_jas_spin_radial_factors,
    genfmt_jas_spin_selection, genfmt_jas_transition_count, genfmt_jas_transition_setup,
    genfmt_momentum_grid, genfmt_nstar_row, genfmt_nstar_rows, genfmt_ordinary_driver_output,
    genfmt_ordinary_path_energy_grid, genfmt_ordinary_path_energy_grid_finalization,
    genfmt_ordinary_path_energy_grid_from_driver_setup,
    genfmt_ordinary_path_energy_grid_from_setup, genfmt_ordinary_path_energy_point,
    genfmt_ordinary_path_evaluation, genfmt_ordinary_path_evaluation_from_driver_setup,
    genfmt_ordinary_path_evaluation_from_setup, genfmt_ordinary_path_finalization,
    genfmt_ordinary_path_outputs, genfmt_ordinary_path_sequence,
    genfmt_ordinary_path_sequence_from_driver_setup, genfmt_ordinary_path_sequence_from_setup,
    genfmt_ordinary_path_trace, genfmt_ordinary_spin_momentum_grid, genfmt_path_geometry,
    genfmt_path_importance, genfmt_path_matrix_product, genfmt_path_matrix_trace,
    genfmt_path_output_decision, genfmt_path_retention, genfmt_path_signal_contribution,
    genfmt_path_signals, genfmt_retained_path_output, genfmt_scattering_matrix_plan,
    genfmt_scattering_path_product, genfmt_spin_channel_count, genfmt_spin_phase_shifts,
    genfmt_spin_radial_factors, genfmt_spin_reference_energies,
};
pub use types::*;
pub use xstar::xstar;

const ONE_DEGREE_RADIANS: Real = 0.017_453_292_52;
const RDPATH_EPSILON: Real = 1.0e-6;
const SNLM_AFAC: Real = 1.0 / 64.0;
const SNLM_FACTORIAL_LIMIT: usize = 210;
const SNLM_FACTORIAL_COUNT: usize = SNLM_FACTORIAL_LIMIT + 1;

#[cfg(test)]
mod tests;
