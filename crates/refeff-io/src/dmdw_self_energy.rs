//! FEFF DMDW run-type 2 self-energy and spectral-function sidecar codecs.
//!
//! `DMDW/m_dmdw.f90` writes these files after constructing the pole-weight
//! `a2f` representation: `dmdw_a2f.info`, `dmdw_spectral.info`,
//! `dmdw_Egrid.info`, `dmdw_reSE_a2F.dat`, `dmdw_imSE_a2F.dat`, and
//! `dmdw_Akw.dat`. This module keeps those text boundaries typed so the solver
//! port can target FEFF-compatible artifacts.

mod a2f_info;
mod akw_dat;
mod common;
mod egrid_info;
mod self_energy_dat;
mod spectral_info;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use a2f_info::{
    dmdw_a2f_info_from_pole_weighted, dmdw_a2f_info_string, parse_dmdw_a2f_info,
    read_dmdw_a2f_info, write_dmdw_a2f_info,
};
pub use akw_dat::{dmdw_akw_dat_string, parse_dmdw_akw_dat, read_dmdw_akw_dat, write_dmdw_akw_dat};
pub use egrid_info::{
    dmdw_egrid_info_string, parse_dmdw_egrid_info, read_dmdw_egrid_info, write_dmdw_egrid_info,
};
pub use self_energy_dat::{
    dmdw_self_energy_dat_string, parse_dmdw_self_energy_dat, read_dmdw_self_energy_dat,
    write_dmdw_self_energy_dat,
};
pub use spectral_info::{
    dmdw_spectral_info_string, parse_dmdw_spectral_info, read_dmdw_spectral_info,
    write_dmdw_spectral_info,
};
pub use types::{
    DmdwA2fInfoData, DmdwAkwDatData, DmdwEnergyGridInfo, DmdwSelfEnergyDatData,
    DmdwSpectralInfoData,
};
