//! FEFF `exc.dat` excitation-pole table codec.
//!
//! The SELF many-pole path writes `exc.dat` through FEFF's generic
//! `WriteData` helper with three required double columns and one auxiliary
//! weight column. SFCONV's `rdeps` reader consumes the first three columns as
//! pole energy, pole broadening, and oscillator strength in eV, and can also
//! create a three-column fallback file when no `exc.dat` exists.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const EXC_DAT_PATH: &str = "exc.dat";
const EXC_DAT_REQUIRED_COLUMNS: usize = 3;
const EXC_DAT_AUXILIARY_COLUMNS: usize = 4;

/// Parsed FEFF `exc.dat` excitation-pole table.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcDatData {
    /// Header and comment lines before and around the numeric pole table.
    pub header_lines: Vec<String>,
    /// Pole energy in eV.
    pub energy_ev: Array1<f64>,
    /// Pole broadening in eV.
    pub broadening_ev: Array1<f64>,
    /// Oscillator strength for each pole.
    pub oscillator_strength: Array1<f64>,
    /// Optional fourth `WriteData` column from SELF's many-pole generator.
    pub auxiliary_weight: Option<Array1<f64>>,
}

impl ExcDatData {
    /// Number of excitation poles.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether this table carries SELF's fourth auxiliary column.
    #[must_use]
    pub fn has_auxiliary_weight(&self) -> bool {
        self.auxiliary_weight.is_some()
    }
}

/// Render FEFF-compatible `exc.dat` text.
pub fn exc_dat_string(data: &ExcDatData) -> Result<String> {
    validate_exc_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        for (((energy, broadening), strength), auxiliary) in data
            .energy_ev
            .iter()
            .zip(data.broadening_ev.iter())
            .zip(data.oscillator_strength.iter())
            .zip(auxiliary_weight.iter())
        {
            write_exc_row(&mut out, [*energy, *broadening, *strength, *auxiliary])?;
        }
    } else {
        for ((energy, broadening), strength) in data
            .energy_ev
            .iter()
            .zip(data.broadening_ev.iter())
            .zip(data.oscillator_strength.iter())
        {
            write_exc_row(&mut out, [*energy, *broadening, *strength])?;
        }
    }
    Ok(out)
}

fn write_exc_row<const N: usize>(out: &mut String, fields: [f64; N]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 20, 10)?;
        out.push(' ');
    }
    out.push('\n');
    Ok(())
}

/// Parse FEFF `exc.dat` text.
pub fn parse_exc_dat(text: &str) -> Result<ExcDatData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut broadening_ev = Vec::new();
    let mut oscillator_strength = Vec::new();
    let mut auxiliary_weight = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(width, EXC_DAT_REQUIRED_COLUMNS | EXC_DAT_AUXILIARY_COLUMNS) {
                return parse_error(
                    line_number,
                    format!("exc.dat row has {width} token(s), expected 3 or 4 numeric columns"),
                );
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return parse_error(
                        line_number,
                        format!(
                            "exc.dat row has {width} token(s), expected {expected} to match previous rows"
                        ),
                    );
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            broadening_ev.push(parse_f64(line_number, "broadening", tokens[1])?);
            oscillator_strength.push(parse_f64(line_number, "oscillator strength", tokens[2])?);
            if width == EXC_DAT_AUXILIARY_COLUMNS {
                auxiliary_weight.push(parse_f64(line_number, "auxiliary weight", tokens[3])?);
            }
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let auxiliary_weight = if row_width == Some(EXC_DAT_AUXILIARY_COLUMNS) {
        Some(Array1::from_vec(auxiliary_weight))
    } else {
        None
    };
    let data = ExcDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        broadening_ev: Array1::from_vec(broadening_ev),
        oscillator_strength: Array1::from_vec(oscillator_strength),
        auxiliary_weight,
    };
    validate_exc_dat(&data)?;
    Ok(data)
}

/// Write FEFF `exc.dat` text to a file.
pub fn write_exc_dat(path: impl AsRef<Path>, data: &ExcDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, exc_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `exc.dat` text from a file.
pub fn read_exc_dat(path: impl AsRef<Path>) -> Result<ExcDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_exc_dat(&text)
}

fn validate_exc_dat(data: &ExcDatData) -> Result<()> {
    let pole_count = data.pole_count();
    if pole_count == 0 {
        return invalid_exc_dat("rows", "at least one excitation-pole row is required");
    }
    validate_len("broadening_ev", data.broadening_ev.len(), pole_count)?;
    validate_len(
        "oscillator_strength",
        data.oscillator_strength.len(),
        pole_count,
    )?;
    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        validate_len("auxiliary_weight", auxiliary_weight.len(), pole_count)?;
    }

    for (row, ((energy, broadening), strength)) in data
        .energy_ev
        .iter()
        .zip(data.broadening_ev.iter())
        .zip(data.oscillator_strength.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite("energy", *energy, row)?;
        validate_finite("broadening", *broadening, row)?;
        validate_finite("oscillator strength", *strength, row)?;
    }
    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        for (row, value) in auxiliary_weight.iter().enumerate() {
            validate_finite("auxiliary weight", *value, row + 1)?;
        }
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_exc_dat(field, format!("got {actual} value(s), expected {expected}"))
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("invalid {field} value {token:?}")))
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_exc_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_exc_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: EXC_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: EXC_DAT_PATH.into(),
        line,
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
    fn parses_self_write_data_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(EXC_DAT)?;

        assert_eq!(parsed.header_lines.len(), 5);
        assert_eq!(parsed.pole_count(), 2);
        assert!(parsed.has_auxiliary_weight());
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.broadening_ev[1], 0.2);
        assert_eq!(parsed.oscillator_strength[0], 0.25);
        assert_eq!(
            parsed.auxiliary_weight.as_ref().map(|values| values[1]),
            Some(2.5)
        );
        Ok(())
    }

    #[test]
    fn parses_three_column_rdeps_fallback_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(RDEPS_FALLBACK_EXC_DAT)?;

        assert_eq!(parsed.pole_count(), 1);
        assert!(!parsed.has_auxiliary_weight());
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.broadening_ev[0], 0.01);
        assert_eq!(parsed.oscillator_strength[0], 1.0);
        Ok(())
    }

    #[test]
    fn roundtrips_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(EXC_DAT)?;
        let rendered = exc_dat_string(&parsed)?;

        assert_eq!(rendered, EXC_DAT);
        assert_eq!(parse_exc_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_exc_dat_inputs() {
        assert!(parse_exc_dat("# only a header\n").is_err());
        assert!(parse_exc_dat("1 2\n").is_err());
        assert!(parse_exc_dat("1 2 3 4 5\n").is_err());
        assert!(parse_exc_dat("1 2 3\n4 5 6 7\n").is_err());
        assert!(parse_exc_dat("1 NaN 3\n").is_err());

        let bad = ExcDatData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            broadening_ev: Array1::from_vec(vec![0.1]),
            oscillator_strength: Array1::from_vec(vec![1.0, 1.0]),
            auxiliary_weight: None,
        };
        assert!(exc_dat_string(&bad).is_err());
    }

    const EXC_DAT: &str = r#"#SN#   Section:    1
#DF# This section written in TXT.
#H#
#H# The following data types are written in this section.
#DT#  Double Double Double Double
    0.1000000000E+02     0.1000000000E+00     0.2500000000E+00     0.1250000000E+01 
    0.2000000000E+02     0.2000000000E+00     0.5000000000E+00     0.2500000000E+01 
"#;

    const RDEPS_FALLBACK_EXC_DAT: &str = "      10.00000      0.01000      1.00000\n";
}
