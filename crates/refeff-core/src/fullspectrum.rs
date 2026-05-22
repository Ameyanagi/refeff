//! FEFF FULLSPECTRUM numerical helpers.
//!
//! This module covers small kernels from `FULLSPECTRUM/` that can be tested
//! independently of the full driver. Larger spectrum assembly remains in the
//! module runner layer until the surrounding FEFF state is ported.

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
