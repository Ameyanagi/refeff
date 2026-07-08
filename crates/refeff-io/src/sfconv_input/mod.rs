//! Typed reader for FEFF `sfconv.inp` module handoff files.
//!
//! `sfconv.inp` controls spectral-function convolution after spectrum
//! assembly.

mod header;
mod input_file;
mod target_data;
mod targets;
mod types;

#[cfg(test)]
mod tests;

pub use header::{sfconv_so2conv_header_from_text, sfconv_so2conv_material_input_from_header};
pub use input_file::sfconv_input_string;
pub use target_data::{
    sfconv_so2conv_chi_data_from_convolution_rows, sfconv_so2conv_convoluted_target_data_string,
    sfconv_so2conv_feff_path_data_from_averages, sfconv_so2conv_feff_path_data_string,
    sfconv_so2conv_target_data_from_text, sfconv_so2conv_target_data_string,
    sfconv_so2conv_xmu_data_from_convolution_rows, write_sfconv_so2conv_convoluted_target_data,
    write_sfconv_so2conv_feff_path_data, write_sfconv_so2conv_target_data,
};
pub use targets::sfconv_so2conv_targets;
pub use types::{
    SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvControl, SfconvInput, SfconvSo2convFeffPathData,
    SfconvSo2convHeader, SfconvSo2convTarget, SfconvSo2convTargetData, SfconvSo2convTargetKind,
    SfconvSpectrum, SfconvWindow,
};
