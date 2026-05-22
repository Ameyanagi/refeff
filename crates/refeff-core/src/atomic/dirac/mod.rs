mod integration;
mod matching;
mod setup;
mod state;
mod types;

pub(in crate::atomic) use integration::atom_intdir_decay;
pub use integration::atomic_dirac_integration;
pub use matching::{
    atomic_dirac_energy_disagreement_correction, atomic_dirac_energy_disagreement_match,
    atomic_dirac_energy_disagreement_source, atomic_dirac_homogeneous_match,
    atomic_dirac_homogeneous_pass_setup, atomic_dirac_homogeneous_seed,
    atomic_dirac_inhomogeneous_branch, atomic_dirac_inhomogeneous_seed,
    atomic_dirac_large_component_match, atomic_dirac_matching_point_update,
    atomic_dirac_shooting_pass_setup, atomic_dirac_two_component_match,
};
pub use setup::atomic_dirac_solver_setup;
pub use state::{
    atomic_dirac_abnormal_exit_recovery, atomic_dirac_energy_step, atomic_dirac_entry_state,
    atomic_dirac_iteration_reset, atomic_dirac_method_one_energy_correction,
    atomic_dirac_node_count, atomic_dirac_node_energy_search, atomic_dirac_normalization,
    atomic_dirac_rematch_attempt, atomic_dirac_solution_normalization,
};
pub use types::*;

use ndarray::{Array1, ArrayView1};

use crate::Real;

use super::types::*;
use super::validation::*;

pub(super) const ATOM_INTDIR_HISTORY: usize = 5;
const ATOM_INTDIR_PREDICTOR: [Real; ATOM_INTDIR_HISTORY] =
    [251.0, -1274.0, 2616.0, -2774.0, 1901.0];
const ATOM_INTDIR_CORRECTOR_RAW: [Real; ATOM_INTDIR_HISTORY] = [-19.0, 106.0, -264.0, 646.0, 251.0];
const ATOM_INTDIR_MIX_NUMERATOR: Real = 473.0;
const ATOM_INTDIR_MIX_DENOMINATOR: Real = 502.0;
const ATOM_INTDIR_STEP_DIVISOR: Real = 720.0;
const ATOM_INTDIR_INWARD_THRESHOLD: Real = 700.0;
const ATOM_INTDIR_EXPONENT_FLOOR: Real = -170.0;
const ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN: usize = 10;
pub(super) const ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET: usize = 12;
const ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY: Real = 1.0e-1;
