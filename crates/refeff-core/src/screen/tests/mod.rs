use super::{
    RealVec, ScreenContourEnergyGridInput, ScreenCrpaProjectionWindow,
    ScreenCrpaResponseSliceInput, ScreenEnergyStateInput, ScreenError,
    ScreenExactRadialContinuationInput, ScreenFmsResponseSliceInput, ScreenGetphRadialBoundsInput,
    ScreenIrregularInitialConditionInput, ScreenIrregularWronskianScaleInput,
    ScreenPhasePotentialInput, ScreenRadialBoundsInput, ScreenRdgeomAtomicUnitsInput,
    ScreenSolutionNormalizationInput, screen_atomic_response_slice,
    screen_bare_core_hole_potential, screen_contour_energy_grid, screen_coulomb_kernel_matrix,
    screen_crpa_density_weights, screen_crpa_hubbard_summary, screen_crpa_orbital_density,
    screen_crpa_response_slice, screen_energy_integration_delta, screen_energy_state,
    screen_exact_radial_continuation, screen_exponential_energy_grid,
    screen_fms_cluster_green_trace, screen_fms_response_slice, screen_getph_lmax,
    screen_getph_radial_bounds, screen_integrate_response_step, screen_irregular_initial_condition,
    screen_irregular_wronskian_scale, screen_lda_exchange_correlation_kernel,
    screen_phase_potential_reference_shift, screen_radial_bounds, screen_radial_coulomb_potential,
    screen_radial_grid, screen_radial_index_1based, screen_rdgeom_atomic_units,
    screen_response_system_matrix, screen_solution_normalization, screen_solve_response_potential,
    screen_symmetrize_response_upper,
};
use ndarray::array;
use num_complex::Complex32;
use refeff_linalg::LinalgError;

use crate::Complex;

mod grids_setup;
mod invalid_inputs;
mod potentials;
mod radial_solution;
mod response;
mod support;
