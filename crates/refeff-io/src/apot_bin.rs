//! FEFF `apot.bin` TXT section-stream codec.
//!
//! The atomic-potential stage writes `apot.bin` through FEFF's generic
//! `WriteData`, `WriteArrayData`, and `Write2D` helpers. The generated file is
//! a text stream of `#SN#` sections even though the suffix is `.bin`. This
//! module keeps the stream typed while preserving section headers and FEFF's
//! column-major matrix shapes in [`ndarray::Array2`] values.

mod common;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use parse::parse_apot_bin;
pub use render::{apot_bin_string, read_apot_bin, write_apot_bin};
pub use types::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords,
    ApotBinSection, ApotBinType, ApotBinValue,
};
