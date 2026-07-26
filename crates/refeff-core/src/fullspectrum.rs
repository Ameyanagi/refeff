//! FEFF FULLSPECTRUM numerical helpers.
//!
//! This module contains the numerical kernels used by the Rust FULLSPECTRUM
//! driver, including edge selection and grids, FPRIME/FMS/path assembly,
//! dielectric conversion, optical constants, and sum-rule transforms.

mod assembly;
mod background;
mod constants;
mod edges;
mod fine_structure;
mod grids;
mod optics;
mod sum_rules;
mod transforms;
mod types;
mod validation;

pub use assembly::*;
pub use background::*;
pub use constants::*;
pub use edges::*;
pub use fine_structure::*;
pub use grids::*;
pub use optics::*;
pub use sum_rules::*;
pub use transforms::*;
pub use types::*;

#[cfg(test)]
mod tests;
