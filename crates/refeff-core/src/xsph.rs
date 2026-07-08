//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels for final-state planning,
//! angular coefficient tables, NRIXS transition weights, q-Bessel tables,
//! reduced and cross-section radial matrix elements, X-ray Bessel tables,
//! initial-state occupation normalization, and angular-decomposition spectrum
//! updates, plus phase-mesh primitive, FEFF84 grid construction, and `grid.inp`
//! user-grid phase mesh composition.

use ndarray::{Array1, Array2, Array5, ArrayView1, ShapeBuilder};
use refeff_linalg::feff_determinant;

use crate::{
    BesselError, Complex, Real, SphericalBessel,
    constants::{BOHR_ANGSTROM as XSPH_BOHR_ANGSTROM, HARTREE_EV as XSPH_HARTREE_EV},
    spherical_bessel_j_y, terp, wigner_3j,
    xsph_occ_norm::{
        XSPH_OCC_NORM_ATOMIC_NUMBER_MAX, XSPH_OCC_NORM_HOLE_COUNT, xsph_occ_norm_denominator,
        xsph_occ_norm_numerator,
    },
};

const QBESSEL_MAX_LJ: usize = 39;
const QBESSEL_ZERO_CUTOFF: Real = 1.0e8;
const CWIG3J_MAX_DOUBLED_ARGUMENT: i32 = 116;
const XSPH_MAX_LX: usize = 20;
const XSPH_FINE_STRUCTURE_ALPHA: Real = 1.0 / 137.035_989_56;
const XSPH_HOLE_ORBITAL_X0: Real = 8.80;
const XSPH_HOLE_ORBITAL_TAIL_CUTOFF: Real = 1.0e-11;
const XSPH_PHASE_SORT_TOLERANCE: Real = 0.001;
const XSPH_USER_PHASE_GRID_MAX_RECORDS: usize = 10;

mod angular;
mod axafs;
mod mesh;
mod orbital;
mod phase;
mod planning;
mod radial;
mod spectrum;
mod support;
mod types;

pub use angular::{
    xsph_angular_density_coefficients, xsph_longitudinal_multipole_factor,
    xsph_relativistic_multipole_factors,
};
pub use axafs::xsph_axafs;
pub use mesh::*;
pub use orbital::xsph_initial_hole_orbital;
pub use phase::{
    xsph_empty_cell_phase, xsph_hubbard_phase_assignments, xsph_hubbard_phase_potential_shifts,
    xsph_hubbard_phase_reference_tail, xsph_phase_angular_limit, xsph_phase_channel_plan,
    xsph_phase_cutoff, xsph_phase_energy_setup, xsph_phase_grid_preparation,
    xsph_phase_plasmon_pole_setup, xsph_phase_radial_header, xsph_phase_radial_indices,
    xsph_phase_radial_output, xsph_phase_reference_tail, xsph_phase_self_energy_summary,
    xsph_regular_phase, xsph_regular_phase_channel,
};
pub use planning::{
    xsph_bcoef_transition_indices, xsph_jas_bessel_functions, xsph_lj_needed_flags,
    xsph_minimize_calculations, xsph_nrixs_transition_indices, xsph_occupation_normalization,
    xsph_q_bessel_table, xsph_tdlda_channel_basis, xsph_tdlda_decode_projector_selector,
    xsph_xsect_bcoef_weights,
};
pub use radial::{
    xsph_jas_orthogonality_correction, xsph_jas_overlap, xsph_jas_radial_cross_integral,
    xsph_jas_radial_integral, xsph_radial_cross_integral, xsph_radial_integral,
    xsph_xray_bessel_table, xsph_xsect_bcoef_central_cross_section,
    xsph_xsect_bcoef_cross_term_accumulation, xsph_xsect_bcoef_cross_term_state_accumulation,
    xsph_xsect_bcoef_direct_transition, xsph_xsect_bcoef_direct_transition_update,
    xsph_xsect_bcoef_nonstandard_channel_row, xsph_xsect_bcoef_nonstandard_energy_row,
    xsph_xsect_bcoef_ordinary_row, xsph_xsect_bcoef_standard_channel_row,
    xsph_xsect_bcoef_standard_energy_row,
    xsph_xsect_bcoef_standard_energy_row_with_transition_fields, xsph_xsect_central_cross_section,
    xsph_xsect_cross_term_accumulation, xsph_xsect_cross_term_plan,
    xsph_xsect_cross_term_state_reuse, xsph_xsect_cross_term_state_save, xsph_xsect_density_branch,
    xsph_xsect_direct_transition, xsph_xsect_embedded_density, xsph_xsect_energy_setup,
    xsph_xsect_fscf_integral, xsph_xsect_fscf_weights, xsph_xsect_hole_normalization,
    xsph_xsect_irregular_channel, xsph_xsect_irregular_initial_condition,
    xsph_xsect_irregular_transform, xsph_xsect_output_normalization,
    xsph_xsect_phiscf_accumulated_response, xsph_xsect_phiscf_angular_channels,
    xsph_xsect_phiscf_contribution_plan, xsph_xsect_phiscf_contribution_rule,
    xsph_xsect_phiscf_field_assembly, xsph_xsect_phiscf_irregular_seed,
    xsph_xsect_phiscf_linear_solve, xsph_xsect_phiscf_lipman_response,
    xsph_xsect_phiscf_local_field, xsph_xsect_phiscf_pole_energy,
    xsph_xsect_phiscf_radial_contribution, xsph_xsect_phiscf_radial_solver_setup,
    xsph_xsect_phiscf_screened_contributions, xsph_xsect_phiscf_screened_solution,
    xsph_xsect_phiscf_wfirdc_contribution, xsph_xsect_phiscf_wfirdc_contributions,
    xsph_xsect_projected_density, xsph_xsect_radial_pass, xsph_xsect_regular_channel,
    xsph_xsect_regular_solution, xsph_xsect_screened_field_setup, xsph_xsect_transition_plan,
    xsph_xsect_weighted_radial_cross_integral, xsph_xsect_weighted_radial_integral,
};
pub use spectrum::*;
pub(in crate::xsph) use support::*;
pub use types::*;

#[cfg(test)]
mod tests;
