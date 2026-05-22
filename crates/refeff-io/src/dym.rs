//! FEFF DMDW `.dym` dynamical-matrix codec.
//!
//! `DMDW/m_dmdw.f90` reads `.dym` files as a dynamical-matrix type flag,
//! atom metadata, coordinates, and one 3x3 force-constant block for every
//! atom pair. Type 1 files store Cartesian coordinates directly. Type 2 files
//! add unique-atom metadata for self-energy runs. Type 3 files add
//! Gaussian-style dipole derivatives for DMDW IR runs. Type 4 files store
//! reduced coordinates followed by three cell vectors.

mod common;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use parse::{parse_dym, read_dym};
pub use render::{dym_string, write_dym};
pub use types::{DymCoordinates, DymData, DymType2Metadata, DymUniqueAtom};
