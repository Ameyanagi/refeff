//! FEFF `pot.bin` text/PAD potential-state codec.
//!
//! FEFF10 writes `pot.bin` from `POT/wrpot.f90` as a formatted text file with
//! fixed-width integer records and PAD-encoded real arrays. This module keeps
//! the same field order and Fortran column-major traversal while exposing the
//! data as typed `ndarray` arrays.

mod common;
mod fullspectrum;
mod parse;
mod render;
mod rhorrp;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use fullspectrum::{
    fullspectrum_number_density_from_pot_bin, fullspectrum_potential_state_from_pot_bin,
};
pub use parse::{parse_pot_bin, read_pot_bin};
pub use render::{pot_bin_string, write_pot_bin};
pub use rhorrp::{
    RHORRP_POT_BIN_RADIAL_DX, RHORRP_WAVEFUNCTION_RADIAL_COUNT, RHORRP_WAVEFUNCTION_RADIAL_X0,
    RhorrpPotBinWavefunctionHandoff, RhorrpWavefunctionTablesHandoff,
    RhorrpWavefunctionTablesHandoffInput, rhorrp_wavefunction_handoff_from_pot_bin,
    rhorrp_wavefunction_tables_from_handoffs,
};
pub use types::{
    FullSpectrumPotentialState, POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH,
    POT_BIN_IORB_SLOTS, POT_BIN_MISC_SCALARS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS, PotBinData,
    PotBinScalars,
};
