//! FEFF `hamaker.dat` FULLSPECTRUM imaginary-axis dielectric codec.
//!
//! `FULLSPECTRUM/fullspectrum.f90` can write `hamaker.dat` from the dormant
//! `dohamaker` branch as `omega` plus a complex dielectric transform in
//! Fortran `e20.10` fields. The transform is currently disabled in FEFF10 by a
//! compile-time flag, but the file boundary is kept here for orchestration and
//! reference testing.

use std::path::Path;

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use refeff_core::{FullSpectrumHamakerInput, full_spectrum_hamaker_transform};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const HAMAKER_DAT_PATH: &str = "hamaker.dat";
const HAMAKER_DAT_ROW_WIDTH: usize = 3;

/// Parsed FEFF `hamaker.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct HamakerDatData {
    /// Header or comment lines before and around the numeric table.
    pub header_lines: Vec<String>,
    /// FULLSPECTRUM energy grid `omega`, in Hartree.
    pub omega: Array1<f64>,
    /// Dielectric transform evaluated on the imaginary axis.
    pub imaginary_axis_epsilon: Array1<Complex64>,
}

impl HamakerDatData {
    /// Number of imaginary-axis dielectric samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

/// Compute FEFF-compatible `hamaker.dat` rows from `eps - 1` on a Hartree grid.
pub fn hamaker_dat_from_fullspectrum_epsilon(
    header_lines: Vec<String>,
    omega: ArrayView1<'_, f64>,
    epsilon_minus_one: ArrayView1<'_, Complex64>,
) -> Result<HamakerDatData> {
    let transform = full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
        omega,
        epsilon: epsilon_minus_one,
    })
    .map_err(|source| invalid_hamaker_dat_error("hamaker_transform", source.to_string()))?;
    let data = HamakerDatData {
        header_lines,
        omega: omega.to_owned(),
        imaginary_axis_epsilon: transform.mapv(|value| Complex64::new(value, 0.0)),
    };
    validate_hamaker_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `hamaker.dat` text.
pub fn hamaker_dat_string(data: &HamakerDatData) -> Result<String> {
    validate_hamaker_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        out.push_str(line);
        out.push('\n');
    }
    for (omega, epsilon) in data.omega.iter().zip(data.imaginary_axis_epsilon.iter()) {
        write_hamaker_row(&mut out, [*omega, epsilon.re, epsilon.im])?;
    }
    Ok(out)
}

/// Parse FEFF `hamaker.dat` text.
pub fn parse_hamaker_dat(text: &str) -> Result<HamakerDatData> {
    let mut header_lines = Vec::new();
    let mut omega = Vec::new();
    let mut imaginary_axis_epsilon = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != HAMAKER_DAT_ROW_WIDTH {
                return parse_error(
                    line_number,
                    format!(
                        "hamaker.dat row has {} token(s), expected {HAMAKER_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            omega.push(parse_f64(line_number, "omega", tokens[0])?);
            imaginary_axis_epsilon.push(Complex64::new(
                parse_f64(line_number, "imaginary-axis epsilon real", tokens[1])?,
                parse_f64(line_number, "imaginary-axis epsilon imaginary", tokens[2])?,
            ));
        } else if !line.trim().is_empty() {
            header_lines.push(line.to_string());
        }
    }

    let data = HamakerDatData {
        header_lines,
        omega: Array1::from_vec(omega),
        imaginary_axis_epsilon: Array1::from_vec(imaginary_axis_epsilon),
    };
    validate_hamaker_dat(&data)?;
    Ok(data)
}

/// Write FEFF `hamaker.dat` text to a file.
pub fn write_hamaker_dat(path: impl AsRef<Path>, data: &HamakerDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, hamaker_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `hamaker.dat` text from a file.
pub fn read_hamaker_dat(path: impl AsRef<Path>) -> Result<HamakerDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_hamaker_dat(&text)
}

fn write_hamaker_row(out: &mut String, fields: [f64; HAMAKER_DAT_ROW_WIDTH]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 20, 10)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_hamaker_dat(data: &HamakerDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return invalid_hamaker_dat("rows", "at least one Hamaker row is required");
    }
    validate_len(
        "imaginary_axis_epsilon",
        data.imaginary_axis_epsilon.len(),
        point_count,
    )?;

    for (row, omega) in data.omega.iter().enumerate() {
        validate_positive("omega", *omega, row + 1)?;
    }
    for (row, epsilon) in data.imaginary_axis_epsilon.iter().enumerate() {
        validate_finite("imaginary-axis epsilon real", epsilon.re, row + 1)?;
        validate_finite("imaginary-axis epsilon imaginary", epsilon.im, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_hamaker_dat(field, format!("got {actual} value(s), expected {expected}"))
    }
}

fn validate_positive(field: &'static str, value: f64, row: usize) -> Result<()> {
    if !value.is_finite() {
        invalid_hamaker_dat(field, format!("row {row} value must be finite"))
    } else if value <= 0.0 {
        invalid_hamaker_dat(field, format!("row {row} value must be positive"))
    } else {
        Ok(())
    }
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_hamaker_dat(field, format!("row {row} value must be finite"))
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("invalid {field} value {token:?}")))
}

fn invalid_hamaker_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_hamaker_dat_error(field, message))
}

fn invalid_hamaker_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: HAMAKER_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: HAMAKER_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn parses_feff_fullspectrum_hamaker_dat() -> Result<()> {
        let parsed = parse_hamaker_dat(HAMAKER_DAT)?;

        assert_eq!(parsed.point_count(), 5);
        assert_eq!(parsed.omega[0], 1.0);
        assert_eq!(
            parsed.imaginary_axis_epsilon[2],
            Complex64::new(0.223_104_490_2, 0.0)
        );
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_hamaker_dat() -> Result<()> {
        let parsed = parse_hamaker_dat(HAMAKER_DAT)?;
        let rendered = hamaker_dat_string(&parsed)?;

        assert_eq!(rendered, HAMAKER_DAT);
        assert_eq!(parse_hamaker_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn derives_hamaker_dat_from_fullspectrum_epsilon() -> Result<()> {
        let omega = array![1.0, 2.0, 4.0, 7.0, 11.0];
        let epsilon = array![
            Complex64::new(0.1, 0.2),
            Complex64::new(0.3, 0.5),
            Complex64::new(0.2, 0.25),
            Complex64::new(0.4, 0.4),
            Complex64::new(0.1, 0.15),
        ];

        let rendered = hamaker_dat_string(&hamaker_dat_from_fullspectrum_epsilon(
            Vec::new(),
            omega.view(),
            epsilon.view(),
        )?)?;

        assert_eq!(rendered, HAMAKER_DAT);
        Ok(())
    }

    #[test]
    fn rejects_bad_hamaker_dat_inputs() {
        assert!(parse_hamaker_dat("").is_err());
        assert!(parse_hamaker_dat("# only header\n").is_err());
        assert!(parse_hamaker_dat("1 2\n").is_err());
        assert!(parse_hamaker_dat("1 2 3 4\n").is_err());
        assert!(parse_hamaker_dat("0 1 0\n").is_err());
        assert!(parse_hamaker_dat("1 NaN 0\n").is_err());

        let bad = HamakerDatData {
            header_lines: Vec::new(),
            omega: array![1.0, 2.0],
            imaginary_axis_epsilon: array![Complex64::new(1.0, 0.0)],
        };
        assert!(hamaker_dat_string(&bad).is_err());
    }

    const HAMAKER_DAT: &str = "    0.1000000000E+01    0.3546469825E+00    0.0000000000E+00\n    0.2000000000E+01    0.3546469825E+00    0.0000000000E+00\n    0.4000000000E+01    0.2231044902E+00    0.0000000000E+00\n    0.7000000000E+01    0.1268866601E+00    0.0000000000E+00\n    0.1100000000E+02    0.1268866601E+00    0.0000000000E+00\n";
}
