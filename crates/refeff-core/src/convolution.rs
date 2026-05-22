//! FEFF analytic convolution helpers.
//!
//! This module ports `MATH/conv.f90`. FEFF linearly interpolates a complex
//! spectrum on each energy interval and integrates the Lorentzian kernel
//! analytically with `conv1`; `conv` applies that segment integral to every
//! requested output energy and adds one extrapolated endpoint interval. It also
//! contains the `FF2X/exconv.f90` excitation-spectrum convolution and
//! `FF2X/xscorratan.f90` arctangent correction used by the final spectrum
//! assembly path.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::interpolation::{InterpolationError, locate_below, terp, terpc};
use crate::{Complex, ComplexVec, Real, RealVec};

const FEFF_REAL_PI: Real = std::f32::consts::PI as Real;
const XSCORR_ATAN_PI: Real = std::f64::consts::PI;
const XSCORR_EPS4: Real = 1.0e-4;

mod atan;
mod excitation;
mod lorentzian;
mod support;
mod types;

pub use atan::ff2x_atan_correction;
pub use excitation::ff2x_excitation_convolve;
pub use lorentzian::{conv, conv_in_place, conv1};
pub(in crate::convolution) use support::*;
pub use types::*;

#[cfg(test)]
mod tests;
