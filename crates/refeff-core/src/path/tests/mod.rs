use super::*;
use crate::Complex;
use ndarray::{Array2, Array3, ShapeBuilder, arr2};

const CRITERION_TOLERANCE: Real = 1.0e-6;
const HASH_TOLERANCE: Real = 1.0e-3;
const MRB_TOLERANCE: Real = 1.0e-7;
const OUTPUT_PARAMETER_TOLERANCE: Real = 1.0e-7;
const STANDARD_TOLERANCE: Real = 1.0e-7;
const PHASE_CRITERIA_TOLERANCE: Real = 1.0e-6;

mod criteria;
mod geometry;
mod packing_phase;
mod standard;
mod support;
