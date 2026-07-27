//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{
    Array2, Array3, Array4, Array5, Array6, ArrayView2, ArrayView3, ArrayView4, ArrayView5,
    ArrayView6, Axis, ShapeBuilder,
};
use num_complex::Complex32;
use refeff_linalg::{complex32_faer_lu_factor, complex32_faer_lu_solve};

use crate::{
    Real,
    angular::SpinOrbitCouplingTables,
    state::{StateKet, StateKetError, construct_state_kets_with_limit},
};

const FMS_ROTATION_LMAX: usize = 24;

mod cluster;
mod driver;
mod internals;
mod mkgtr;
mod pairs;
mod polynomial;
mod reciprocal;
mod rotation;
mod solvers;
mod t_matrix;
mod types;
mod xgllm;

pub use cluster::{
    fms_yprep_cluster, pair_polar_angles, sort_atoms_by_radius, sort_representative_atoms,
};
pub use driver::{
    fms_driver_setup, fms_real_space_energy, fms_real_space_energy_with_plan, fms_real_space_plan,
    fms_real_space_spectrum, fms_scattering_method_selection,
};
use internals::*;
pub use mkgtr::{
    MkgtrGreenTraceInput, MkgtrGreenTraceResult, MkgtrJasGreenTraceInput, MkgtrJasGreenTraceResult,
    MkgtrJasQPairMode, MkgtrJasTransition, mkgtr_green_trace, mkgtr_jas_green_trace,
};
pub use pairs::{
    fms_free_propagator_element, fms_free_propagator_matrix, fms_pair_tables,
    fms_spin_free_propagator_matrix, fms_spin_pair_tables,
};
pub use polynomial::rehr_albers_polynomials;
pub use reciprocal::{
    FmsReciprocalAccumulator, FmsReciprocalCoreHoleInput, FmsReciprocalError, FmsReciprocalPlan,
    fms_reciprocal_apply_core_hole,
};
pub use rotation::{fms_rotation_matrix, fms_yprep_geometry};
pub use solvers::{
    fms_bicgstab_scattering, fms_full_potential_lu_scattering, fms_graves_morris_scattering,
    fms_iterative_system_matrix, fms_lu_scattering, fms_recursion_scattering, fms_scattering,
    fms_tfqmr_scattering,
};
pub use t_matrix::{
    fms_hubbard_back_transform_full_scattering, fms_hubbard_back_transform_scattering,
    fms_hubbard_t_matrix_element, fms_hubbard_t_matrix_table, fms_hubbard_transform_t_matrix,
    fms_t_matrix_element, fms_t_matrix_table,
};
pub use types::*;
pub use xgllm::rehr_albers_z_axis_propagator;

#[cfg(test)]
mod tests;
