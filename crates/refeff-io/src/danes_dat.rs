//! FEFF `danes.dat` anomalous-scattering text codec.
//!
//! The FEFF `fprime` writer emits `danes.dat` as a single header line followed
//! by seven numeric columns: energy, Matsubara pole contribution, Sommerfeld
//! correction, anomalous contribution, tail contribution, total, and
//! difference.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const DANES_DAT_ROW_WIDTH: usize = 7;

/// Parsed FEFF `danes.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct DanesDatData {
    /// Header and comment lines before and around the numeric table.
    pub header_lines: Vec<String>,
    /// Energy relative to the edge in eV.
    pub energy_ev: Array1<f64>,
    /// Matsubara pole contribution.
    pub matsubara: Array1<f64>,
    /// Sommerfeld correction.
    pub sommerfeld: Array1<f64>,
    /// Anomalous contribution.
    pub anomalous: Array1<f64>,
    /// Tail contribution. The FEFF header historically spells this as `tale`.
    pub tail: Array1<f64>,
    /// Total anomalous scattering factor.
    pub total: Array1<f64>,
    /// Difference term.
    pub difference: Array1<f64>,
}

impl DanesDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Render FEFF-compatible `danes.dat` text.
pub fn danes_dat_string(data: &DanesDatData) -> Result<String> {
    validate_danes_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for ((((((energy, matsubara), sommerfeld), anomalous), tail), total), difference) in data
        .energy_ev
        .iter()
        .zip(data.matsubara.iter())
        .zip(data.sommerfeld.iter())
        .zip(data.anomalous.iter())
        .zip(data.tail.iter())
        .zip(data.total.iter())
        .zip(data.difference.iter())
    {
        out.push(' ');
        write_fortran_exp(&mut out, *energy, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *matsubara, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *sommerfeld, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *anomalous, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *tail, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *total, 11, 4)?;
        out.push(' ');
        write_fortran_exp(&mut out, *difference, 11, 4)?;
        out.push('\n');
    }
    Ok(out)
}

/// Parse FEFF `danes.dat` text.
pub fn parse_danes_dat(text: &str) -> Result<DanesDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut matsubara = Vec::new();
    let mut sommerfeld = Vec::new();
    let mut anomalous = Vec::new();
    let mut tail = Vec::new();
    let mut total = Vec::new();
    let mut difference = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != DANES_DAT_ROW_WIDTH {
                return Err(IoError::DanesDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: DANES_DAT_ROW_WIDTH,
                });
            }
            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            matsubara.push(parse_f64(line_number, "matsubara", tokens[1])?);
            sommerfeld.push(parse_f64(line_number, "sommerfeld", tokens[2])?);
            anomalous.push(parse_f64(line_number, "anomalous", tokens[3])?);
            tail.push(parse_f64(line_number, "tail", tokens[4])?);
            total.push(parse_f64(line_number, "total", tokens[5])?);
            difference.push(parse_f64(line_number, "difference", tokens[6])?);
        } else {
            header_lines.push(line.to_string());
        }
    }

    let data = DanesDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        matsubara: Array1::from_vec(matsubara),
        sommerfeld: Array1::from_vec(sommerfeld),
        anomalous: Array1::from_vec(anomalous),
        tail: Array1::from_vec(tail),
        total: Array1::from_vec(total),
        difference: Array1::from_vec(difference),
    };
    validate_danes_dat(&data)?;
    Ok(data)
}

/// Write FEFF `danes.dat` text to a file.
pub fn write_danes_dat(path: impl AsRef<Path>, data: &DanesDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, danes_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `danes.dat` text from a file.
pub fn read_danes_dat(path: impl AsRef<Path>) -> Result<DanesDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_danes_dat(&text)
}

fn validate_danes_dat(data: &DanesDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_danes_dat(
            "rows",
            "at least one spectrum row is required",
        ));
    }
    validate_len("matsubara", data.matsubara.len(), point_count)?;
    validate_len("sommerfeld", data.sommerfeld.len(), point_count)?;
    validate_len("anomalous", data.anomalous.len(), point_count)?;
    validate_len("tail", data.tail.len(), point_count)?;
    validate_len("total", data.total.len(), point_count)?;
    validate_len("difference", data.difference.len(), point_count)?;

    for (row, ((((((energy, matsubara), sommerfeld), anomalous), tail), total), difference)) in data
        .energy_ev
        .iter()
        .zip(data.matsubara.iter())
        .zip(data.sommerfeld.iter())
        .zip(data.anomalous.iter())
        .zip(data.tail.iter())
        .zip(data.total.iter())
        .zip(data.difference.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("energy", *energy, row)?;
        validate_finite_row("matsubara", *matsubara, row)?;
        validate_finite_row("sommerfeld", *sommerfeld, row)?;
        validate_finite_row("anomalous", *anomalous, row)?;
        validate_finite_row("tail", *tail, row)?;
        validate_finite_row("total", *total, row)?;
        validate_finite_row("difference", *difference, row)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::DanesDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::DanesDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidDanesDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_danes_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidDanesDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_danes_reference_shape() -> Result<()> {
        let data = parse_danes_dat(DANES_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.energy_ev[0], -18.690);
        assert_eq!(data.matsubara[1], 0.0);
        assert_eq!(data.sommerfeld[2], 0.0);
        assert_eq!(data.anomalous[0], 10.097);
        assert_eq!(data.tail[1], 4.9442);
        assert_eq!(data.total[2], 5.2935);
        assert_eq!(data.difference[0], -5.4576);
        Ok(())
    }

    #[test]
    fn roundtrips_danes_text() -> Result<()> {
        let data = parse_danes_dat(DANES_DAT)?;
        let rendered = danes_dat_string(&data)?;
        assert_eq!(parse_danes_dat(&rendered)?, data);
        assert_eq!(rendered, DANES_DAT);
        Ok(())
    }

    #[test]
    fn rejects_bad_danes_inputs() {
        assert!(parse_danes_dat("# no data\n").is_err());
        assert!(parse_danes_dat("1 2 3 4 5 6\n").is_err());
        assert!(parse_danes_dat("1 2 3 NaN 5 6 7\n").is_err());
    }

    const DANES_DAT: &str = r#"# E  matsub. sommerf. anomal. tale, total, differ.
 -1.8690E+01  0.0000E+00  0.0000E+00  1.0097E+01  4.6396E+00  4.6396E+00 -5.4576E+00
 -1.7122E+01  0.0000E+00  0.0000E+00  1.0603E+01  4.9442E+00  4.9442E+00 -5.6591E+00
 -1.5703E+01  0.0000E+00  0.0000E+00  1.1159E+01  5.2935E+00  5.2935E+00 -5.8651E+00
"#;
}
