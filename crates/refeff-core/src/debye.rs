//! Debye and Einstein-model cumulant helpers ported from FEFF.
//!
//! This module starts with `DEBYE/sigm3.f90`, the correlated Einstein model
//! with a Morse potential used for first and third cumulant estimates.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayView4};
use refeff_linalg::{SymmetricTriangle, real64_symmetric_eigen};

use crate::{
    Complex, Real,
    atomic::atomic_weight as feff_atomic_weight,
    constants::{
        BOHR_ANGSTROM, HARTREE_EV as DMDW_COUPLING_NORM_HARTREE_EV,
        HARTREE_EV_DMDW_COUPLING_LEGACY as DMDW_COUPLING_ENERGY_HARTREE_EV,
    },
    special::complex_digamma,
};

/// FEFF DMDW conversion factor from Angstrom to Bohr.
pub const DMDW_ANGSTROM_TO_BOHR: Real = 1.889_726_663_510_319_2;
const HBAR: Real = 1.054_572_7e-34_f32 as Real;
const ATOMIC_MASS_UNIT: Real = 1.660_54e-27_f32 as Real;
const BOLTZMANN: Real = 1.380_658e-23_f32 as Real;
const DEBYE_CORRELATION_FACTOR: Real = 48.508_46_f32 as Real;
const DEBYE_ROMBERG_TOLERANCE: Real = 1.0e-5;
const DEBYE_ROMBERG_MAX_ITERATIONS: usize = 10;
const AU_FORCE_TO_NEWTON_PER_METER: Real = 1_556.892_791_61;
const NEWTON_PER_METER_TO_AMU_PER_PS2: Real = 602.214_198_280;
const DMDW_DYNAMICAL_MATRIX_SCALE: Real =
    AU_FORCE_TO_NEWTON_PER_METER * NEWTON_PER_METER_TO_AMU_PER_PS2;
const DMDW_AMU_EV: Real = 9.314_78e8;
const DMDW_LIGHT_SPEED_ANGSTROM_PER_PS: Real = 2.997_924_58e6;
const DMDW_BOLTZMANN_EV_PER_K: Real = 8.617_385e-5;
const DMDW_HBAR_EV_PS: Real = 6.582_122e-4;
const DMDW_HBARC_EV_ANGSTROM: Real = 1_973.27;
const DMDW_GAS_CONSTANT_J_PER_MOL_K: Real = 8.314_713_470;
const DMDW_THZ_TO_KELVIN: Real = 47.990_874_194_2;
const DMDW_AMU_THZ2_TO_NEWTON_PER_METER: Real = 0.001_660_538_730_00;
const DMDW_LANCZOS_POLE_SEARCH_LIMIT: Real = 810_000.0;
const DMDW_LANCZOS_DEFAULT_SAMPLES_PER_POLE: usize = 100_000;
const DMDW_IMAGINARY_POLE_SMALL_WEIGHT: Real = 0.01;
const DMDW_IMAGINARY_POLE_LARGE_WEIGHT: Real = 0.05;
const DMDW_COUPLING_GRID_TOLERANCE: Real = 1.0e-10;
// FEFF uses the rounded literal 6.28 in the dmdw_a2f.info diagnostic.
#[allow(clippy::approx_constant)]
const DMDW_A2F_DIAGNOSTIC_TWO_PI: Real = 6.28;
const DMDW_A2F_PLANCK_EV_PS: Real = 4.135_667_516;
const DMDW_A2F_POLE_ANGULAR_TO_EV: Real = 6.582_119_28e-4;
const DMDW_SELF_ENERGY_BOLTZMANN_EV_PER_K: Real = 8.617_334_2e-5;
const DMDW_SELF_ENERGY_TWO_PI: Real = 6.283_185_307_179_584;
const DMDW_SPECTRAL_DENOMINATOR_SHIFT: Real = 1.0e-30;

mod dmdw;
mod models;
mod path;
mod spring;
mod types;

pub use dmdw::*;
pub use models::*;
pub use path::*;
pub use spring::*;
pub use types::*;

use dmdw::{dmdw_director_sum, validate_dmdw_atom_positions, validate_dmdw_atoms};

fn vector_norm(vector: [Real; 3]) -> Real {
    dot(vector, vector).sqrt()
}

fn dot(left: [Real; 3], right: [Real; 3]) -> Real {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

fn cross(left: [Real; 3], right: [Real; 3]) -> [Real; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn atomic_weight(atomic_number: usize) -> Result<Real, DebyeError> {
    feff_atomic_weight(atomic_number)
        .map_err(|_| DebyeError::InvalidAtomicNumber { z: atomic_number })
}

fn ensure_nonnegative(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(DebyeError::Negative { name, value })
    }
}

fn to_feff_real(value: Real) -> Real {
    (value as f32) as Real
}

fn ensure_finite(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFinite { name, value })
    }
}

fn ensure_finite_complex(name: &'static str, value: Complex) -> Result<(), DebyeError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteComplex { name, value })
    }
}

fn ensure_positive(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DebyeError::NonPositive { name, value })
    }
}

fn ensure_finite_output(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteOutput { name, value })
    }
}

fn ensure_finite_complex_output(name: &'static str, value: Complex) -> Result<(), DebyeError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteComplex { name, value })
    }
}

#[cfg(test)]
mod tests;
