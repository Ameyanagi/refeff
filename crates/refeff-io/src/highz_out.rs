//! FEFF `HighZ.out` reference-script output parser.
//!
//! The HIGHZ example sweeps atomic numbers and records whether FEFF's high-Z
//! atomic path produced `atom00.dat`, plus optional 1s binding energies from
//! HIGHZ and the standard atomic run. This file is produced by the example's
//! shell script rather than by the `rdinp` handoff stage.

use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};

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

/// Read and parse a FEFF `HighZ.out` file.
pub fn read_highz_out(path: impl AsRef<Path>) -> Result<HighZOut> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_highz_out_with_source(path.to_path_buf(), &text)
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
        let parsed = parse_highz_out(
            "1: pass  1.360588E+01 1.360588E+01 0\n\
             100: pass  1.430273E+05\n\
             118: fail\n",
        )?;

        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.rows[0].atomic_number, 1);
        assert!(parsed.rows[0].passed);
        assert_eq!(parsed.rows[0].highz_energy_ev, Some(13.60588));
        assert_eq!(parsed.rows[0].reference_energy_ev, Some(13.60588));
        assert_eq!(parsed.rows[0].relative_difference_percent, Some(0.0));
        assert_eq!(parsed.rows[1].atomic_number, 100);
        assert_eq!(parsed.rows[1].reference_energy_ev, None);
        assert_eq!(parsed.rows[2].atomic_number, 118);
        assert!(!parsed.rows[2].passed);
        assert_eq!(parsed.rows[2].highz_energy_ev, None);
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
    }
}
