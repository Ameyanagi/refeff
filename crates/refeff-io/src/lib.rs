#![forbid(unsafe_code)]

//! Input/output compatibility support for the FEFF10 Rust port.
//!
//! This crate owns FEFF text parsing, FEFF-style intermediate writers, and
//! file-format codecs such as Packed ASCII Data (PAD). Numerical modules should
//! depend on these typed structures rather than re-parsing FEFF text ad hoc.

pub mod eels_input;
pub mod error;
pub mod ff2x_input;
pub mod fms_input;
pub mod format;
pub mod genfmt_input;
pub mod global_input;
pub mod input;
pub mod ldos_input;
pub mod model;
pub mod pad;
pub mod paths_input;
pub mod pot_input;
pub mod rdinp;
pub mod rixs_input;
pub mod xsph_input;

pub use eels_input::{EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh};
pub use error::{IoError, Result};
pub use ff2x_input::{Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput};
pub use fms_input::{FmsCluster, FmsControl, FmsDebye, FmsInput};
pub use genfmt_input::{GenfmtControl, GenfmtInput};
pub use global_input::{
    CfAverage, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GlobalQVector,
};
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use ldos_input::{LdosControl, LdosFms, LdosInput, LdosMesh};
pub use model::{Atom, FeffDocument, Potential};
pub use paths_input::{PathsControl, PathsCriteria, PathsInput};
pub use pot_input::{PotControl, PotInput, PotPotential, PotRun, PotScattering};
pub use rixs_input::{RixsBroadening, RixsEnergyWindow, RixsInput, RixsSwitches};
pub use xsph_input::{XsphAdvanced, XsphControl, XsphGrid, XsphInput};
