use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

use super::common::{
    DMDW_AKW_DAT_PATH, DMDW_AKW_ROW_WIDTH, parse_error, parse_f64, parse_single_value,
};
use super::types::DmdwAkwDatData;
use super::validate::validate_dmdw_akw_dat;

/// Parse FEFF `dmdw_Akw.dat` text.
pub fn parse_dmdw_akw_dat(text: &str) -> Result<DmdwAkwDatData> {
    let mut normalization = None;
    let mut energy_mev = Vec::new();
    let mut magnitude = Vec::new();
    let mut phase = Vec::new();
    let mut real = Vec::new();
    let mut imaginary = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("# norm") {
            normalization = Some(parse_single_value(
                DMDW_AKW_DAT_PATH,
                line_number,
                "normalization",
                line,
            )?);
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != DMDW_AKW_ROW_WIDTH {
            return parse_error(
                DMDW_AKW_DAT_PATH,
                line_number,
                format!(
                    "dmdw_Akw.dat row has {} token(s), expected {DMDW_AKW_ROW_WIDTH}",
                    tokens.len()
                ),
            );
        }
        energy_mev.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "energy",
            tokens[0],
        )?);
        magnitude.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "magnitude",
            tokens[1],
        )?);
        phase.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "phase",
            tokens[2],
        )?);
        real.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "real",
            tokens[3],
        )?);
        imaginary.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "imaginary",
            tokens[4],
        )?);
    }

    let data = DmdwAkwDatData {
        normalization,
        energy_mev: Array1::from_vec(energy_mev),
        magnitude: Array1::from_vec(magnitude),
        phase: Array1::from_vec(phase),
        real: Array1::from_vec(real),
        imaginary: Array1::from_vec(imaginary),
    };
    validate_dmdw_akw_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_Akw.dat` text.
pub fn dmdw_akw_dat_string(data: &DmdwAkwDatData) -> Result<String> {
    validate_dmdw_akw_dat(data)?;

    let mut out = String::new();
    if let Some(normalization) = data.normalization {
        write!(out, "# norm =")?;
        write_fortran_zero_scaled_exp(&mut out, normalization, 20, 10)?;
        out.push('\n');
        writeln!(out, "# w [meV], mag, ph, re, im")?;
    }
    for ((((&energy, &magnitude), &phase), &real), &imaginary) in data
        .energy_mev
        .iter()
        .zip(data.magnitude.iter())
        .zip(data.phase.iter())
        .zip(data.real.iter())
        .zip(data.imaginary.iter())
    {
        writeln!(
            out,
            "{energy:20.10}{magnitude:20.10}{phase:20.10}{real:20.10}{imaginary:20.10}"
        )?;
    }
    Ok(out)
}

/// Read FEFF `dmdw_Akw.dat` text from disk.
pub fn read_dmdw_akw_dat(path: impl AsRef<Path>) -> Result<DmdwAkwDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_akw_dat(&text)
}

/// Write FEFF `dmdw_Akw.dat` text to disk.
pub fn write_dmdw_akw_dat(path: impl AsRef<Path>, data: &DmdwAkwDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_akw_dat_string(data)?).map_err(|source| IoError::io(path, source))
}
