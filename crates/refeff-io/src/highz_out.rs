//! FEFF `HighZ.out` reference-script output support.
//!
//! The HIGHZ example sweeps atomic numbers and records whether FEFF's high-Z
//! atomic path produced `atom00.dat`, plus optional 1s binding energies from
//! HIGHZ and the standard atomic run. This file is produced by the example's
//! shell script rather than by the `rdinp` handoff stage.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

/// Parsed contents of FEFF's `HighZ.out` example summary.
#[derive(Debug, Clone, PartialEq)]
pub struct HighZOut {
    /// One row per attempted atomic number.
    pub rows: Vec<HighZOutRow>,
}

impl HighZOut {
    /// Number of atomic-number rows in the summary.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// One row from FEFF's `HighZ.out` example summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighZOutRow {
    /// Atomic number tested by the HIGHZ example script.
    pub atomic_number: usize,
    /// Whether the HIGHZ atomic run produced `atom00.dat`.
    pub passed: bool,
    /// HIGHZ 1s binding energy, when present.
    pub highz_energy_ev: Option<f64>,
    /// Standard FEFF atomic 1s binding energy, available for rows 1 through 99.
    pub reference_energy_ev: Option<f64>,
    /// Absolute relative difference in percent between HIGHZ and standard FEFF.
    pub relative_difference_percent: Option<f64>,
}

/// Parse FEFF `HighZ.out` text.
pub fn parse_highz_out(text: &str) -> Result<HighZOut> {
    parse_highz_out_with_source(PathBuf::from("HighZ.out"), text)
}

/// Render FEFF-compatible `HighZ.out` text.
pub fn highz_out_string(data: &HighZOut) -> Result<String> {
    validate_highz_out(data)?;
    let mut out = String::new();
    for row in &data.rows {
        write_highz_row(&mut out, row)?;
        out.push('\n');
    }
    Ok(out)
}

/// Read and parse a FEFF `HighZ.out` file.
pub fn read_highz_out(path: impl AsRef<Path>) -> Result<HighZOut> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_highz_out_with_source(path.to_path_buf(), &text)
}

/// Write FEFF-compatible `HighZ.out` text to a file.
pub fn write_highz_out(path: impl AsRef<Path>, data: &HighZOut) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, highz_out_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_highz_out_with_source(source: PathBuf, text: &str) -> Result<HighZOut> {
    let rows = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| parse_highz_row(&source, index + 1, line))
        .collect::<Result<Vec<_>>>()?;

    if rows.is_empty() {
        return Err(parse_error(
            &source,
            0,
            "HighZ.out requires at least one row",
        ));
    }

    Ok(HighZOut { rows })
}

fn validate_highz_out(data: &HighZOut) -> Result<()> {
    if data.rows.is_empty() {
        return Err(parse_error(
            Path::new("HighZ.out"),
            0,
            "HighZ.out requires at least one row",
        ));
    }
    for (index, row) in data.rows.iter().enumerate() {
        let line = index + 1;
        if row.atomic_number == 0 {
            return Err(parse_error(
                Path::new("HighZ.out"),
                line,
                "atomic number must be positive",
            ));
        }
        validate_optional_finite("HIGHZ energy", line, row.highz_energy_ev)?;
        validate_optional_finite("reference energy", line, row.reference_energy_ev)?;
        validate_optional_finite("relative difference", line, row.relative_difference_percent)?;
        match (
            row.highz_energy_ev,
            row.reference_energy_ev,
            row.relative_difference_percent,
        ) {
            (Some(_), Some(_), Some(_)) | (Some(_), None, None) | (None, None, None) => {}
            _ => {
                return Err(parse_error(
                    Path::new("HighZ.out"),
                    line,
                    "HIGHZ row must contain either 0, 1, or 3 numeric values",
                ));
            }
        }
    }
    Ok(())
}

fn validate_optional_finite(field: &str, line: usize, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) if !value.is_finite() => Err(parse_error(
            Path::new("HighZ.out"),
            line,
            format!("{field} must be finite"),
        )),
        _ => Ok(()),
    }
}

fn write_highz_row(out: &mut String, row: &HighZOutRow) -> Result<()> {
    let status = if row.passed { "pass" } else { "fail" };
    write!(out, "{}: {status}", row.atomic_number)?;
    match (
        row.highz_energy_ev,
        row.reference_energy_ev,
        row.relative_difference_percent,
    ) {
        (Some(highz), Some(reference), Some(relative)) => {
            out.push_str("  ");
            write_fortran_exp(out, highz, 0, 6)?;
            out.push(' ');
            write_fortran_exp(out, reference, 0, 6)?;
            out.push(' ');
            out.push_str(&format_shell_g6(relative));
        }
        (Some(highz), None, None) => {
            out.push_str("  ");
            write_fortran_exp(out, highz, 0, 6)?;
            out.push_str("  ");
        }
        (None, None, None) => out.push_str("    "),
        _ => {
            return Err(parse_error(
                Path::new("HighZ.out"),
                0,
                "HIGHZ row must contain either 0, 1, or 3 numeric values",
            ));
        }
    }
    Ok(())
}

fn format_shell_g6(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let precision = 6_usize;
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let decimals = precision.saturating_sub(1);
        let raw = format!("{value:.decimals$e}");
        format_lower_exp(&raw)
    } else {
        let decimals = (precision as i32 - exponent - 1).max(0) as usize;
        trim_fraction(format!("{value:.decimals$}"))
    }
}

fn format_lower_exp(raw: &str) -> String {
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return trim_fraction(raw.to_string());
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'-') => ('-', &exponent[1..]),
        Some(b'+') => ('+', &exponent[1..]),
        _ => ('+', exponent),
    };
    let trimmed_digits = digits.trim_start_matches('0');
    let exponent_digits = if trimmed_digits.is_empty() {
        "0"
    } else {
        trimmed_digits
    };
    format!(
        "{}e{}{:0>2}",
        trim_fraction(mantissa.to_string()),
        sign,
        exponent_digits
    )
}

fn trim_fraction(mut text: String) -> String {
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

fn parse_highz_row(source: &Path, line_number: usize, line: &str) -> Result<HighZOutRow> {
    let Some((atomic_number, payload)) = line.split_once(':') else {
        return Err(parse_error(
            source,
            line_number,
            "HighZ.out row requires an atomic number followed by ':'",
        ));
    };
    let atomic_number = atomic_number
        .trim()
        .parse::<usize>()
        .map_err(|_| parse_error(source, line_number, "invalid atomic number"))?;
    if atomic_number == 0 {
        return Err(parse_error(
            source,
            line_number,
            "atomic number must be positive",
        ));
    }

    let tokens = payload.split_whitespace().collect::<Vec<_>>();
    let Some(status) = tokens.first() else {
        return Err(parse_error(source, line_number, "missing HIGHZ status"));
    };
    let passed = match *status {
        "pass" => true,
        "fail" => false,
        _ => {
            return Err(parse_error(
                source,
                line_number,
                format!("invalid HIGHZ status {status:?}"),
            ));
        }
    };

    let values = tokens
        .iter()
        .skip(1)
        .map(|token| parse_f64(source, line_number, token))
        .collect::<Result<Vec<_>>>()?;
    if !matches!(values.len(), 0 | 1 | 3) {
        return Err(parse_error(
            source,
            line_number,
            format!(
                "expected 0, 1, or 3 numeric HIGHZ values, got {}",
                values.len()
            ),
        ));
    }

    Ok(HighZOutRow {
        atomic_number,
        passed,
        highz_energy_ev: values.first().copied(),
        reference_energy_ev: values.get(1).copied(),
        relative_difference_percent: values.get(2).copied(),
    })
}

fn parse_f64(source: &Path, line_number: usize, token: &str) -> Result<f64> {
    let value = token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error(
            source,
            line_number,
            format!("invalid numeric value {token:?}"),
        )
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error(
            source,
            line_number,
            format!("numeric value {token:?} must be finite"),
        ))
    }
}

fn parse_error(source: &Path, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_highz_summary_rows() -> Result<()> {
        let text = "1: pass  1.360588E+01 1.360588E+01 0\n\
                    2: pass  2.497982E+01 2.497981E+01 4.00323e-05\n\
                    100: pass  1.430273E+05  \n\
                    118: fail    \n";
        let parsed = parse_highz_out(text)?;

        assert_eq!(parsed.row_count(), 4);
        assert_eq!(parsed.rows[0].atomic_number, 1);
        assert!(parsed.rows[0].passed);
        assert_eq!(parsed.rows[0].highz_energy_ev, Some(13.60588));
        assert_eq!(parsed.rows[0].reference_energy_ev, Some(13.60588));
        assert_eq!(parsed.rows[0].relative_difference_percent, Some(0.0));
        assert_eq!(parsed.rows[1].relative_difference_percent, Some(4.00323e-5));
        assert_eq!(parsed.rows[2].atomic_number, 100);
        assert_eq!(parsed.rows[2].reference_energy_ev, None);
        assert_eq!(parsed.rows[3].atomic_number, 118);
        assert!(!parsed.rows[3].passed);
        assert_eq!(parsed.rows[3].highz_energy_ev, None);
        assert_eq!(highz_out_string(&parsed)?, text);
        Ok(())
    }

    #[test]
    fn rejects_bad_highz_rows() {
        assert!(matches!(
            parse_highz_out("1 pass 10.0"),
            Err(IoError::Parse { .. })
        ));
        assert!(matches!(
            parse_highz_out("1: maybe 10.0"),
            Err(IoError::Parse { .. })
        ));
        assert!(matches!(
            parse_highz_out("1: pass 10.0 11.0"),
            Err(IoError::Parse { .. })
        ));
        assert!(matches!(
            highz_out_string(&HighZOut {
                rows: vec![HighZOutRow {
                    atomic_number: 1,
                    passed: true,
                    highz_energy_ev: Some(10.0),
                    reference_energy_ev: Some(11.0),
                    relative_difference_percent: None,
                }],
            }),
            Err(IoError::Parse { .. })
        ));
    }
}
