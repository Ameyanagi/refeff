//! FEFF `phase.bin` text/PAD phase-shift codec.
//!
//! `XSPH/wrxsph.f90` writes this handoff file for downstream FMS and FF2X
//! stages. The file is formatted text: a fixed-width integer header, a small
//! real PAD block, and several complex PAD blocks. This module preserves that
//! order while exposing phase shifts and transition moments as `ndarray`
//! values.

mod common;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use parse::parse_phase_bin;
pub use render::{phase_bin_string, read_phase_bin, write_phase_bin};
pub use types::{
    PHASE_BIN_DEFAULT_PAD_WIDTH, PHASE_BIN_DEFAULT_TRANSITION_COUNT, PHASE_BIN_SCALARS,
    PhaseBinData, PhaseBinPotential, PhaseBinRawPads, PhaseBinScalars,
};
