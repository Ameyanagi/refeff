//! FEFF module handoff input writers emitted by `rdinp`.

use crate::Result;
use crate::model::FeffDocument;
use crate::sfconv_input::sfconv_input_string;

mod dmdw;
mod fms;
mod ldos;
mod paths;
mod potential;
mod rixs;
mod xsph;

pub use dmdw::dmdw_inp_string;
pub use fms::fms_inp_string;
pub use ldos::ldos_inp_string;
pub use paths::{
    ff2x_inp_string, genfmt_inp_string, paths_inp_string, single_scattering_paths_dat_string,
};
pub use potential::pot_inp_string;
pub use rixs::rixs_inp_string;
pub use xsph::xsph_inp_string;

/// Render FEFF-compatible `sfconv.inp` content from an [`FeffDocument`].
pub fn sfconv_inp_string(document: &FeffDocument) -> Result<String> {
    sfconv_input_string(&document.sfconv_input)
}
