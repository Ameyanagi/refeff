//! FEFF `feff.bin` text/PAD path-data codec.
//!
//! `GENFMT/genfmtsub.f90` writes this printable handoff file and FF2X reads it
//! via `FF2X/rdfbin.f90`. The format uses tagged text records for metadata and
//! Packed ASCII Data (PAD) blocks for shared energy arrays and per-path data.

mod common;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use parse::parse_feff_bin;
pub use render::{feff_bin_string, read_feff_bin, write_feff_bin};
pub use types::{
    FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH, FeffBinData, FeffBinPath, FeffBinPotential,
};
