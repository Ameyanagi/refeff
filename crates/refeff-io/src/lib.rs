pub mod error;
pub mod format;
pub mod input;
pub mod model;
pub mod pad;
pub mod rdinp;

pub use error::{IoError, Result};
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use model::{Atom, FeffDocument, Potential};
