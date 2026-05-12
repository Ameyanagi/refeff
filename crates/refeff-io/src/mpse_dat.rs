//! FEFF `mpse.dat` many-pole self-energy table codec.
//!
//! FEFF writes `mpse.dat` from the MPSE self-energy path and reads it from RIXS
//! as an energy grid plus complex self-energy. Some files also carry a complex
//! renormalization factor in columns four and five.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::fortran_exp;

const MPSE_DAT_MIN_ROW_WIDTH: usize = 3;
const MPSE_DAT_FULL_ROW_WIDTH: usize = 5;
const MPSE_DAT_ALLOWED_ROW_WIDTHS: &str = "3 or 5";

/// Parsed FEFF `mpse.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct MpseDatData {
    /// Header and comment lines before and around the numeric self-energy table.
    pub header_lines: Vec<String>,
    /// Photoelectron energy relative to the Fermi energy in eV.
    pub energy_ev: Array1<f64>,
    /// Complex self-energy in eV.
    pub self_energy: Array1<Complex64>,
    /// Optional complex renormalization factor `Z`.
    pub renormalization: Option<Array1<Complex64>>,
}

impl MpseDatData {
    /// Number of self-energy samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether this table includes the complex renormalization factor.
    #[must_use]
    pub fn has_renormalization(&self) -> bool {
        self.renormalization.is_some()
    }
}

/// Render FEFF-compatible `mpse.dat` text.
pub fn mpse_dat_string(data: &MpseDatData) -> Result<String> {
    validate_mpse_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    if let Some(renormalization) = &data.renormalization {
        for ((energy, sigma), z) in data
            .energy_ev
            .iter()
            .zip(data.self_energy.iter())
            .zip(renormalization.iter())
        {
            writeln!(
                out,
                "{} {} {} {} {}",
                fortran_exp(*energy, 20, 10),
                fortran_exp(sigma.re, 20, 10),
                fortran_exp(sigma.im, 20, 10),
                fortran_exp(z.re, 20, 10),
                fortran_exp(z.im, 20, 10)
            )?;
        }
    } else {
        for (energy, sigma) in data.energy_ev.iter().zip(data.self_energy.iter()) {
            writeln!(
                out,
                "{} {} {}",
                fortran_exp(*energy, 20, 10),
                fortran_exp(sigma.re, 20, 10),
                fortran_exp(sigma.im, 20, 10)
            )?;
        }
    }
    Ok(out)
}

/// Parse FEFF `mpse.dat` text.
pub fn parse_mpse_dat(text: &str) -> Result<MpseDatData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut self_energy = Vec::new();
    let mut renormalization = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(width, MPSE_DAT_MIN_ROW_WIDTH | MPSE_DAT_FULL_ROW_WIDTH) {
                return Err(IoError::MpseDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: MPSE_DAT_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::MpseDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: row_width_label(expected),
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            self_energy.push(Complex64::new(
                parse_f64(line_number, "self energy real", tokens[1])?,
                parse_f64(line_number, "self energy imaginary", tokens[2])?,
            ));
            if width == MPSE_DAT_FULL_ROW_WIDTH {
                renormalization.push(Complex64::new(
                    parse_f64(line_number, "renormalization real", tokens[3])?,
                    parse_f64(line_number, "renormalization imaginary", tokens[4])?,
                ));
            }
        } else {
            header_lines.push(line.to_string());
        }
    }

    let renormalization = if row_width == Some(MPSE_DAT_FULL_ROW_WIDTH) {
        Some(Array1::from_vec(renormalization))
    } else {
        None
    };
    let data = MpseDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        self_energy: Array1::from_vec(self_energy),
        renormalization,
    };
    validate_mpse_dat(&data)?;
    Ok(data)
}

/// Write FEFF `mpse.dat` text to a file.
pub fn write_mpse_dat(path: impl AsRef<Path>, data: &MpseDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, mpse_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `mpse.dat` text from a file.
pub fn read_mpse_dat(path: impl AsRef<Path>) -> Result<MpseDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_mpse_dat(&text)
}

fn validate_mpse_dat(data: &MpseDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_mpse_dat(
            "rows",
            "at least one self-energy row is required",
        ));
    }
    validate_len("self_energy", data.self_energy.len(), point_count)?;
    if let Some(renormalization) = &data.renormalization {
        validate_len("renormalization", renormalization.len(), point_count)?;
    }

    for (row, (energy, sigma)) in data
        .energy_ev
        .iter()
        .zip(data.self_energy.iter())
        .enumerate()
    {
        validate_finite("energy", *energy, row + 1)?;
        validate_complex_finite("self_energy", *sigma, row + 1)?;
    }
    if let Some(renormalization) = &data.renormalization {
        for (row, value) in renormalization.iter().enumerate() {
            validate_complex_finite("renormalization", *value, row + 1)?;
        }
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::MpseDatShape {
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
        .map_err(|_| IoError::MpseDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_mpse_dat(
            field,
            format!("row {row} value must be finite"),
        ))
    }
}

fn validate_complex_finite(field: &'static str, value: Complex64, row: usize) -> Result<()> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(invalid_mpse_dat(
            field,
            format!("row {row} complex value must be finite"),
        ))
    }
}

fn invalid_mpse_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidMpseDat {
        field,
        message: message.into(),
    }
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        MPSE_DAT_MIN_ROW_WIDTH => "3",
        MPSE_DAT_FULL_ROW_WIDTH => "5",
        _ => MPSE_DAT_ALLOWED_ROW_WIDTHS,
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_mpse_reference_shape() -> Result<()> {
        let data = parse_mpse_dat(MPSE_DAT)?;
        assert_eq!(data.header_lines.len(), 2);
        assert_eq!(data.point_count(), 4);
        assert!(data.has_renormalization());
        assert_eq!(data.energy_ev[0], 0.05);
        assert_eq!(
            data.self_energy[0],
            Complex64::new(0.0055477980, 0.0000486100)
        );
        assert_eq!(
            data.renormalization.as_ref().map(|values| values[0]),
            Some(Complex64::new(0.7774233564, -0.0000445267))
        );
        Ok(())
    }

    #[test]
    fn parses_three_column_mpse_data() -> Result<()> {
        let data = parse_mpse_dat("0.05 0.1 -0.2\n0.20 0.3 -0.4\n")?;
        assert_eq!(data.point_count(), 2);
        assert!(!data.has_renormalization());
        assert_eq!(data.self_energy[1], Complex64::new(0.3, -0.4));
        Ok(())
    }

    #[test]
    fn roundtrips_mpse_text() -> Result<()> {
        let data = parse_mpse_dat(MPSE_DAT)?;
        let rendered = mpse_dat_string(&data)?;
        assert_eq!(parse_mpse_dat(&rendered)?, data);

        let minimal = parse_mpse_dat("0.05 0.1 -0.2\n")?;
        assert_eq!(parse_mpse_dat(&mpse_dat_string(&minimal)?)?, minimal);
        Ok(())
    }

    #[test]
    fn rejects_bad_mpse_inputs() {
        assert!(parse_mpse_dat("# only a header\n").is_err());
        assert!(parse_mpse_dat("1 2\n").is_err());
        assert!(parse_mpse_dat("1 2 3 4\n").is_err());
        assert!(parse_mpse_dat("1 2 3\n4 5 6 7 8\n").is_err());
        assert!(parse_mpse_dat("1 NaN 3\n").is_err());

        let bad = MpseDatData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            self_energy: Array1::from_vec(vec![Complex64::new(1.0, 0.0)]),
            renormalization: None,
        };
        assert!(mpse_dat_string(&bad).is_err());
    }

    const MPSE_DAT: &str = r#"# This file contains information about the self-energy.
# E-EFermi (eV) Re[Sigma(E)] (eV) Im[Sigma(E)] (eV) Re[Z] Im[Z]
        0.0500000000        0.0055477980        0.0000486100        0.7774233564       -0.0000445267
        0.2000000000        0.0382056454        0.0211745226        0.7718384336        0.0124524474
        0.4500000000        0.0694975161        0.0240130827        0.7684244346        0.0123280681
        0.8000000000        0.1100989099        0.0277103596        0.7667449537        0.0108823568
"#;
}
