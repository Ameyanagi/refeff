use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

use super::common::{
    DMDW_SPECTRAL_INFO_PATH, parse_complex_value, parse_error, parse_error_value,
    parse_single_value, write_fortran_complex_tuple,
};
use super::types::DmdwSpectralInfoData;
use super::validate::validate_dmdw_spectral_info;

/// Parse FEFF `dmdw_spectral.info` text.
pub fn parse_dmdw_spectral_info(text: &str) -> Result<DmdwSpectralInfoData> {
    let mut gamma = None;
    let mut effective_electron_energy = None;
    let mut total_cumulant_derivative = None;
    let mut quasiparticle_weight = None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with("Gamma_k") {
            gamma = Some(parse_single_value(
                DMDW_SPECTRAL_INFO_PATH,
                line_number,
                "Gamma_k",
                line,
            )?);
        } else if line.starts_with("epk") {
            effective_electron_energy = Some(parse_single_value(
                DMDW_SPECTRAL_INFO_PATH,
                line_number,
                "epk",
                line,
            )?);
        } else if line.starts_with("atot") {
            total_cumulant_derivative = Some(parse_complex_value(
                DMDW_SPECTRAL_INFO_PATH,
                line_number,
                "atot",
                line,
            )?);
        } else if line.starts_with("Zk") {
            quasiparticle_weight = Some(parse_complex_value(
                DMDW_SPECTRAL_INFO_PATH,
                line_number,
                "Zk",
                line,
            )?);
        } else {
            return parse_error(
                DMDW_SPECTRAL_INFO_PATH,
                line_number,
                format!("unrecognized dmdw_spectral.info line {line:?}"),
            );
        }
    }

    let data = DmdwSpectralInfoData {
        gamma: gamma
            .ok_or_else(|| parse_error_value(DMDW_SPECTRAL_INFO_PATH, 0, "missing Gamma_k"))?,
        effective_electron_energy: effective_electron_energy
            .ok_or_else(|| parse_error_value(DMDW_SPECTRAL_INFO_PATH, 0, "missing epk"))?,
        total_cumulant_derivative: total_cumulant_derivative
            .ok_or_else(|| parse_error_value(DMDW_SPECTRAL_INFO_PATH, 0, "missing atot"))?,
        quasiparticle_weight: quasiparticle_weight
            .ok_or_else(|| parse_error_value(DMDW_SPECTRAL_INFO_PATH, 0, "missing Zk"))?,
    };
    validate_dmdw_spectral_info(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_spectral.info` text.
pub fn dmdw_spectral_info_string(data: &DmdwSpectralInfoData) -> Result<String> {
    validate_dmdw_spectral_info(data)?;

    let mut out = String::new();
    write!(out, "Gamma_k =")?;
    write_fortran_zero_scaled_exp(&mut out, data.gamma, 20, 10)?;
    out.push('\n');
    write!(out, "epk = E_k - ReSE(E_k) =")?;
    write_fortran_zero_scaled_exp(&mut out, data.effective_electron_energy, 20, 10)?;
    out.push_str("\n\n");
    write!(out, "atot    = ")?;
    write_fortran_complex_tuple(&mut out, data.total_cumulant_derivative)?;
    out.push('\n');
    write!(out, "Zk      = ")?;
    write_fortran_complex_tuple(&mut out, data.quasiparticle_weight)?;
    out.push('\n');
    Ok(out)
}

/// Read FEFF `dmdw_spectral.info` from disk.
pub fn read_dmdw_spectral_info(path: impl AsRef<Path>) -> Result<DmdwSpectralInfoData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_spectral_info(&text)
}

/// Write FEFF `dmdw_spectral.info` text to disk.
pub fn write_dmdw_spectral_info(path: impl AsRef<Path>, data: &DmdwSpectralInfoData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_spectral_info_string(data)?)
        .map_err(|source| IoError::io(path, source))
}
