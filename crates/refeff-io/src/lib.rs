#![forbid(unsafe_code)]

//! Input/output compatibility support for the FEFF10 Rust port.
//!
//! This crate owns FEFF text parsing, FEFF-style intermediate writers, and
//! file-format codecs such as Packed ASCII Data (PAD). Numerical modules should
//! depend on these typed structures rather than re-parsing FEFF text ad hoc.

pub mod error;
pub mod format;
pub mod input;
pub mod model;
pub mod pad;
pub mod pot_input;
pub mod rdinp;

pub use error::{IoError, Result};
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use model::{Atom, FeffDocument, Potential};
pub use pot_input::{PotControl, PotInput, PotPotential, PotRun, PotScattering};
