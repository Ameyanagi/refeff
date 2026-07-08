//! FEFF path-finder packing helpers.
//!
//! `PATH/ipack.f90` stores up to eight path atom indices in three signed
//! integers by treating each packed field as base 1290. These helpers preserve
//! that representation for compatibility with FEFF path-degeneracy logic.
//! This module also ports the small companion-index min-heap maintenance
//! routines from `PATH/heap.f90`, the path distance/angle builder from
//! `PATH/mrb.f90`, the path-pruning criteria from `PATH/mcrith.f90` and
//! `PATH/mcritk.f90`, the `PATH/paths.f90` heap search, and the `pathsd`
//! degeneracy reduction. Errors are structured instead of calling `par_stop`.

use ndarray::{Array2, ArrayView2, ArrayView3};

use crate::{Real, quadrature::strap, vector::single_precision_distance_between};

const PATH_PACK_BASE: i32 = 1_290;
const PATH_PACK_BASE_SQUARED: i32 = PATH_PACK_BASE * PATH_PACK_BASE;
const MAX_PACKED_PATH_INDICES: usize = 8;
const MAX_PACKED_PATH_VALUE: i32 = PATH_PACK_BASE - 1;
const DOT_COSINE_EPSILON: f32 = 1.0e-8;
const PATH_HASH_SCALE: f32 = 1_000.0;
const PATH_HASH_FACTOR: f32 = 16.123_457;
const PATH_HASH_POTENTIAL_FACTOR: f32 = 8.576_543;
const PATH_HASH_Y_WEIGHT: f32 = 0.894_375;
const PATH_HASH_Z_WEIGHT: f32 = 0.573_498;
const PATH_HASH_LENGTH_OFFSET: Real = 40_000_000.0;
const PATH_OUTPUT_MIN_ABS_ANGLE_COSINE: f32 = 0.3;
const PATH_STANDARD_EPSILON: Real = 1.0e-4;

mod criteria;
mod degeneracy;
mod error;
mod finder;
mod geometry;
mod heap;
mod packing;
mod phase;
mod standard;
mod support;
mod types;

pub use criteria::{
    path_beta_indices, path_criteria_decision, path_heap_criterion, path_output_criterion,
    path_output_importance,
};
pub use degeneracy::{
    path_degeneracy_groups, path_degeneracy_range, path_degeneracy_reduction,
    path_degeneracy_retention,
};
pub use error::PathError;
pub use finder::{pathfinder_preparation, pathfinder_reduction, pathfinder_search};
pub use geometry::{path_geometry, path_output_parameters};
pub use heap::{path_heap_bubble_down, path_heap_bubble_up};
pub use packing::{pack_path_indices, unpack_path_indices};
pub use phase::path_phase_criteria_tables;
pub use standard::{
    path_canonical_representation, path_degeneracy_hash, path_standard_coordinates,
};
pub use types::*;

#[cfg(test)]
pub(crate) use phase::single_precision_path_value;

use support::*;

#[cfg(test)]
mod tests;
