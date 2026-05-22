//! FEFF FOVRG numerical helpers.
//!
//! These routines cover small pieces of the relativistic radial solver that can
//! be validated independently of the full `dfovrg` integration path.

use ndarray::{Array1, Array2, ArrayView1};

use crate::{
    Complex, ComplexVec, Real, RealMat,
    angular::wigner_3j,
    bessel::{besjh, besjn},
};

// `diff.f90` uses unsuffixed Fortran real literals in these stencils. Preserve
// their default-real rounding before widening to the Rust `Real` type.
const F77_REAL_HALF: Real = 0.5_f32 as Real;
const F77_REAL_ONE_POINT_TWO: Real = 1.2_f32 as Real;
const F77_REAL_ONE_POINT_FIVE: Real = 1.5_f32 as Real;
const F77_REAL_TWO: Real = 2.0_f32 as Real;
const F77_REAL_TWO_POINT_FOUR_FIVE: Real = 2.45_f32 as Real;
const F77_REAL_THREE_POINT_THREE: Real = 3.3_f32 as Real;
const F77_REAL_THREE_POINT_SEVEN_FIVE: Real = 3.75_f32 as Real;
const F77_REAL_FOUR_POINT_TWO: Real = 4.2_f32 as Real;
const F77_REAL_SIX: Real = 6.0_f32 as Real;
const F77_REAL_SIX_AND_TWO_THIRDS: Real = 6.666_666_5_f32 as Real;
const F77_REAL_SEVEN_POINT_FIVE: Real = 7.5_f32 as Real;
const F77_REAL_SEVEN_POINT_EIGHT: Real = 7.8_f32 as Real;
const F77_REAL_EIGHT: Real = 8.0_f32 as Real;
const F77_REAL_TWELVE: Real = 12.0_f32 as Real;
const F77_REAL_ONE_SIXTH: Real = 0.166_666_67_f32 as Real;
const F77_REAL_FOURTEEN_OVER_FORTY_FIVE: Real = (14.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_TWENTY_FOUR_OVER_FORTY_FIVE: Real = (24.0_f32 as Real) / (45.0_f32 as Real);
const F77_REAL_SIXTY_FOUR_OVER_FORTY_FIVE: Real = (64.0_f32 as Real) / (45.0_f32 as Real);
const FOVRG_INT_OUT_HISTORY: usize = 6;
const FOVRG_INT_OUT_TEST: Real = 1.0e5;
const FOVRG_ANGULAR_COEFFICIENT_SLOTS: usize = 5;
const FOVRG_ORIGIN_COEFFICIENTS: usize = 10;
const FEFF_ALPHA_INVERSE: Real = 137.03598956;
const FEFF_FINE_STRUCTURE_ALPHA: Real = 1.0 / FEFF_ALPHA_INVERSE;
const FEFF_WFIRDC_SPEED_OF_LIGHT: Real = 137.0373;
const FOVRG_BOUND_ORBITAL_THRESHOLD: Real = 1.0e-11;
const FOVRG_WKB_MINIMUM_COUNT: usize = 10;

mod derivatives;
mod internals;
mod overlap_exchange;
mod potential;
mod solutions;
mod types;

use internals::*;

pub use derivatives::*;
pub use overlap_exchange::*;
pub use potential::*;
pub use solutions::*;
pub use types::*;

#[cfg(test)]
mod tests;
