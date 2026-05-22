//! Typed reader for FEFF `dmdw.inp` module handoff files.
//!
//! `dmdw.inp` is either disabled with FEFF's `-999` sentinel or contains a
//! dynamical-matrix Debye-Waller calculation request. Most run types carry
//! selected path rows; FEFF run type 2 carries self-energy input files instead.

mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use render::dmdw_input_string;
pub use types::{DmdwCalculation, DmdwInput, DmdwPath, DmdwPdosOptions, DmdwSelfEnergyOptions};
