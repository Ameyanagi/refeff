//! FEFF `loss.dat` loss-function table codec.
//!
//! The MPSE and OPCONS paths use `loss.dat` as a two-column optical loss
//! function table. FEFF reads the first column as energy in eV and the second
//! column as `-Im(epsilon^-1)`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::fortran_exp;

const LOSS_DAT_ROW_WIDTH: usize = 2;

/// Parsed FEFF `loss.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct LossDatData {
    /// Header and comment lines before and around the numeric loss table.
    pub header_lines: Vec<String>,
    /// Loss-function energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// Optical loss function, normally `-Im(epsilon^-1)`.
    pub loss: Array1<f64>,
}

impl LossDatData {
    /// Number of loss-function samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Render FEFF-compatible `loss.dat` text.
pub fn loss_dat_string(data: &LossDatData) -> Result<String> {
    validate_loss_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, loss) in data.energy_ev.iter().zip(data.loss.iter()) {
        writeln!(
            out,
            "{} {}",
            fortran_exp(*energy, 14, 6),
            fortran_exp(*loss, 14, 6)
        )?;
    }
    Ok(out)
}

/// Parse FEFF `loss.dat` text.
pub fn parse_loss_dat(text: &str) -> Result<LossDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut loss = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != LOSS_DAT_ROW_WIDTH {
                return Err(IoError::LossDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: LOSS_DAT_ROW_WIDTH,
                });
            }
            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            loss.push(parse_f64(line_number, "loss", tokens[1])?);
        } else {
            header_lines.push(line.to_string());
        }
    }

    let data = LossDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        loss: Array1::from_vec(loss),
    };
    validate_loss_dat(&data)?;
    Ok(data)
}

/// Write FEFF `loss.dat` text to a file.
pub fn write_loss_dat(path: impl AsRef<Path>, data: &LossDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, loss_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `loss.dat` text from a file.
pub fn read_loss_dat(path: impl AsRef<Path>) -> Result<LossDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_loss_dat(&text)
}

fn validate_loss_dat(data: &LossDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_loss_dat(
            "rows",
            "at least one loss-function row is required",
        ));
    }
    validate_len("loss", data.loss.len(), point_count)?;

    for (row, (energy, loss)) in data.energy_ev.iter().zip(data.loss.iter()).enumerate() {
        validate_finite("energy", *energy, row + 1)?;
        validate_finite("loss", *loss, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::LossDatShape {
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
        .map_err(|_| IoError::LossDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_loss_dat(
            field,
            format!("row {row} value must be finite"),
        ))
    }
}

fn invalid_loss_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidLossDat {
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
    fn parses_feff_loss_reference_shape_and_header() -> Result<()> {
        let data = parse_loss_dat(LOSS_DAT)?;
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.point_count(), 4);
        assert_eq!(data.energy_ev[0], 0.0100000);
        assert_eq!(data.loss[0], 0.200377e-05);
        assert_eq!(data.energy_ev[3], 0.0443767);
        assert_eq!(data.loss[3], 0.889400e-05);
        Ok(())
    }

    #[test]
    fn roundtrips_loss_text() -> Result<()> {
        let data = parse_loss_dat(LOSS_DAT)?;
        let rendered = loss_dat_string(&data)?;
        assert_eq!(parse_loss_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_loss_inputs() {
        assert!(parse_loss_dat("# only a header\n").is_err());
        assert!(parse_loss_dat("1\n").is_err());
        assert!(parse_loss_dat("1 2 3\n").is_err());
        assert!(parse_loss_dat("1 NaN\n").is_err());

        let bad = LossDatData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            loss: Array1::from_vec(vec![1.0]),
        };
        assert!(loss_dat_string(&bad).is_err());
    }

    const LOSS_DAT: &str = r#"# E(eV)    Loss
#
# omega	-Im[eps**{-1}]
0.100000E-01 0.200377E-05
0.187377E-01 0.375479E-05
0.301966E-01 0.605133E-05
0.443767E-01 0.889400E-05
"#;
}
