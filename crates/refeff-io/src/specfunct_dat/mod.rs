//! FEFF `specfunct.dat` spectral-function cache codec.
//!
//! `SFCONV/so2conv.f90` stores this cache as thirteen Fortran sequential
//! unformatted records. The payloads are scalar material settings, pole arrays,
//! two eight-column momentum tables, and seven spectral-function tables. The
//! parser accepts little-endian and big-endian record markers and payloads; the
//! writer emits the little-endian layout produced by the generated FEFF10
//! reference suite.

mod codec;
mod rows;
mod spectral;
mod support;
mod targets;
mod types;
mod validation;

#[cfg(test)]
mod tests;

pub use codec::{
    parse_specfunct_dat, read_specfunct_dat, specfunct_dat_bytes, write_specfunct_dat,
};
pub use rows::{sfconv_specfunct_exafs_convolution_rows, sfconv_specfunct_xanes_convolution_rows};
pub use spectral::{
    sfconv_specfunct_data_from_spectral_rows, sfconv_specfunct_interpolate_momentum,
    sfconv_specfunct_matches_so2conv_inputs, sfconv_specfunct_momentum_interpolation_input,
};
pub use targets::{
    sfconv_specfunct_chi_data_from_cache, sfconv_specfunct_feff_path_data_from_cache,
    sfconv_specfunct_target_data_from_cache, sfconv_specfunct_xmu_data_from_cache,
};
pub use types::{
    SPECFUNCT_DAT_INFO_COLUMNS, SfconvSpecfunctChiDataInput, SfconvSpecfunctCompatibilityInput,
    SfconvSpecfunctData, SfconvSpecfunctExafsRowsInput, SfconvSpecfunctFeffPathDataInput,
    SfconvSpecfunctSpectralRowsInput, SfconvSpecfunctTargetDataInput,
    SfconvSpecfunctXanesRowsInput, SfconvSpecfunctXmuDataInput,
};
