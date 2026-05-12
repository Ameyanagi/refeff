//! FEFF `crpa.dat` constrained random phase approximation output codec.
//!
//! The CRPA module writes a short text table containing the screened Hubbard
//! `U`, occupation `n`, and unscreened bare `U` values.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

const CRPA_DAT_ROW_WIDTH: usize = 3;

/// Parsed FEFF `crpa.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct CrpaDatData {
    /// Header and comment lines before the numeric CRPA result row.
    pub header_lines: Vec<String>,
    /// Screened Hubbard interaction `U`.
    pub hubbard_u: f64,
    /// Occupation number `n`.
    pub occupation: f64,
    /// Unscreened bare interaction `U_Bare`.
    pub bare_u: f64,
}

/// Render FEFF-compatible `crpa.dat` text.
pub fn crpa_dat_string(data: &CrpaDatData) -> Result<String> {
    validate_crpa_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    writeln!(
        out,
        "{hubbard_u:24.17E} {occupation:24.17E} {bare_u:24.17E}",
        hubbard_u = data.hubbard_u,
        occupation = data.occupation,
        bare_u = data.bare_u
    )?;
    Ok(out)
}

/// Parse FEFF `crpa.dat` text.
pub fn parse_crpa_dat(text: &str) -> Result<CrpaDatData> {
    let mut header_lines = Vec::new();
    let mut row = None;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if row.is_some() {
                return Err(invalid_crpa_dat(
                    "rows",
                    "crpa.dat must contain exactly one numeric row",
                ));
            }
            if tokens.len() != CRPA_DAT_ROW_WIDTH {
                return Err(IoError::CrpaDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: CRPA_DAT_ROW_WIDTH,
                });
            }
            row = Some((
                parse_f64(line_number, "U", tokens[0])?,
                parse_f64(line_number, "n", tokens[1])?,
                parse_f64(line_number, "U_Bare", tokens[2])?,
            ));
        } else {
            header_lines.push(line.to_string());
        }
    }

    let (hubbard_u, occupation, bare_u) = row.ok_or(IoError::CrpaDatMissing { field: "row" })?;
    let data = CrpaDatData {
        header_lines,
        hubbard_u,
        occupation,
        bare_u,
    };
    validate_crpa_dat(&data)?;
    Ok(data)
}

/// Write FEFF `crpa.dat` text to a file.
pub fn write_crpa_dat(path: impl AsRef<Path>, data: &CrpaDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, crpa_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `crpa.dat` text from a file.
pub fn read_crpa_dat(path: impl AsRef<Path>) -> Result<CrpaDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_crpa_dat(&text)
}

fn validate_crpa_dat(data: &CrpaDatData) -> Result<()> {
    validate_finite("U", data.hubbard_u)?;
    validate_finite("n", data.occupation)?;
    validate_finite("U_Bare", data.bare_u)?;
    Ok(())
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::CrpaDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_crpa_dat(field, "value must be finite"))
    }
}

fn invalid_crpa_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidCrpaDat {
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
    fn parses_feff_crpa_reference_values() -> Result<()> {
        let data = parse_crpa_dat(CRPA_DAT)?;
        assert_eq!(data.header_lines, vec!["U, n, U_Bare"]);
        assert_eq!(data.hubbard_u, 0.197879035252010);
        assert_eq!(data.occupation, 1.0);
        assert_eq!(data.bare_u, 0.694283422651496);
        Ok(())
    }

    #[test]
    fn roundtrips_crpa_text() -> Result<()> {
        let data = parse_crpa_dat(CRPA_DAT)?;
        let rendered = crpa_dat_string(&data)?;
        assert_eq!(parse_crpa_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_crpa_inputs() {
        assert!(parse_crpa_dat("U, n, U_Bare\n").is_err());
        assert!(parse_crpa_dat("1 2\n").is_err());
        assert!(parse_crpa_dat("1 NaN 2\n").is_err());
        assert!(parse_crpa_dat("1 2 3\n4 5 6\n").is_err());

        let bad = CrpaDatData {
            header_lines: Vec::new(),
            hubbard_u: f64::NAN,
            occupation: 1.0,
            bare_u: 2.0,
        };
        assert!(crpa_dat_string(&bad).is_err());
    }

    const CRPA_DAT: &str = r#"U, n, U_Bare
  0.197879035252010        1.00000000000000       0.694283422651496
"#;
}
