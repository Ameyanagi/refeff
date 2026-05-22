//! FEFF `dmdw.out` Debye-Waller diagnostic output support.
//!
//! The FEFF DMDW stage writes a human-readable report containing Lanczos
//! settings, projected-density-of-states poles, Einstein-frequency summaries,
//! moment-derived summaries, and the run-type-specific result block. This
//! parser keeps those pieces structured so generated FEFF10 DMDW reports can
//! be validated without string matching.

mod common;
mod parse;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use parse::parse_dmdw_out;
pub use render::{dmdw_out_string, read_dmdw_out, write_dmdw_out};
pub use types::{
    DmdwOutData, DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment, DmdwOutPole, DmdwOutSection,
    DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue,
};
