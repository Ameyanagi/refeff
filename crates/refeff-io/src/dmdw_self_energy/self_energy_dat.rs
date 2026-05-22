use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

use super::common::{
    DMDW_SELF_ENERGY_DAT_PATH, DMDW_SELF_ENERGY_ROW_WIDTH, is_numeric_token, parse_error, parse_f64,
};
use super::types::DmdwSelfEnergyDatData;
use super::validate::validate_dmdw_self_energy_dat;

/// Parse FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` text.
pub fn parse_dmdw_self_energy_dat(text: &str) -> Result<DmdwSelfEnergyDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut value_ev = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != DMDW_SELF_ENERGY_ROW_WIDTH {
                return parse_error(
                    DMDW_SELF_ENERGY_DAT_PATH,
                    line_number,
                    format!(
                        "DMDW self-energy row has {} token(s), expected {DMDW_SELF_ENERGY_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            energy_ev.push(parse_f64(
                DMDW_SELF_ENERGY_DAT_PATH,
                line_number,
                "energy",
                tokens[0],
            )?);
            value_ev.push(parse_f64(
                DMDW_SELF_ENERGY_DAT_PATH,
                line_number,
                "self-energy value",
                tokens[1],
            )?);
        } else if !line.trim().is_empty() {
            header_lines.push(raw.to_string());
        }
    }

    let data = DmdwSelfEnergyDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        value_ev: Array1::from_vec(value_ev),
    };
    validate_dmdw_self_energy_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible self-energy table text.
pub fn dmdw_self_energy_dat_string(data: &DmdwSelfEnergyDatData) -> Result<String> {
    validate_dmdw_self_energy_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (&energy, &value) in data.energy_ev.iter().zip(data.value_ev.iter()) {
        write_fortran_zero_scaled_exp(&mut out, energy, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, value, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` text from disk.
pub fn read_dmdw_self_energy_dat(path: impl AsRef<Path>) -> Result<DmdwSelfEnergyDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_self_energy_dat(&text)
}

/// Write FEFF self-energy table text to disk.
pub fn write_dmdw_self_energy_dat(
    path: impl AsRef<Path>,
    data: &DmdwSelfEnergyDatData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_self_energy_dat_string(data)?)
        .map_err(|source| IoError::io(path, source))
}
