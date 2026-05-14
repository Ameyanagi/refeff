//! FEFF `drude.dat` FULLSPECTRUM Drude-term codec.
//!
//! `FULLSPECTRUM/drdtrm.f90` writes the free-electron dielectric contribution
//! as two scalar header records followed by `omega`, real epsilon, and
//! imaginary epsilon columns in Fortran `e20.10` fields.

use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{FullSpectrumDrudeInput, FullSpectrumDrudeTerm, full_spectrum_drude_term};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const DRUDE_DAT_PATH: &str = "drude.dat";
const DRUDE_DAT_ROW_WIDTH: usize = 3;

/// Parsed FEFF `drude.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct DrudeDatData {
    /// Drude width in eV.
    pub gamma_ev: f64,
    /// Plasma frequency in eV.
    pub plasma_frequency_ev: f64,
    /// Energy grid `omega` in Hartree.
    pub omega: Array1<f64>,
    /// Complex Drude dielectric contribution.
    pub epsilon: Array1<Complex64>,
}

impl DrudeDatData {
    /// Number of Drude rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

impl From<FullSpectrumDrudeTerm> for DrudeDatData {
    fn from(term: FullSpectrumDrudeTerm) -> Self {
        Self {
            gamma_ev: term.gamma_ev,
            plasma_frequency_ev: term.plasma_frequency_ev,
            omega: term.omega,
            epsilon: term.epsilon,
        }
    }
}

/// Compute FEFF-compatible `drude.dat` contents from an energy grid.
pub fn drude_dat_from_grid(
    omega: ndarray::ArrayView1<'_, f64>,
    lifetime_seconds: f64,
    number_density: f64,
) -> Result<DrudeDatData> {
    full_spectrum_drude_term(FullSpectrumDrudeInput {
        omega,
        lifetime_seconds,
        number_density,
    })
    .map(DrudeDatData::from)
    .map_err(|error| invalid_drude_dat_error("drude_term", error.to_string()))
}

/// Render FEFF-compatible `drude.dat` text.
pub fn drude_dat_string(data: &DrudeDatData) -> Result<String> {
    validate_drude_dat(data)?;

    let mut out = String::new();
    out.push_str("# gam (eV):");
    write_fortran_zero_scaled_exp(&mut out, data.gamma_ev, 20, 10)?;
    out.push('\n');
    out.push_str("# wp (eV):");
    write_fortran_zero_scaled_exp(&mut out, data.plasma_frequency_ev, 20, 10)?;
    out.push('\n');
    for (omega, epsilon) in data.omega.iter().zip(data.epsilon.iter()) {
        write_drude_row(&mut out, [*omega, epsilon.re, epsilon.im])?;
    }
    Ok(out)
}

/// Parse FEFF `drude.dat` text.
pub fn parse_drude_dat(text: &str) -> Result<DrudeDatData> {
    let mut gamma_ev = None;
    let mut plasma_frequency_ev = None;
    let mut omega = Vec::new();
    let mut epsilon = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if let Some(value) = parse_header_value(line, "# gam (eV):", line_number)? {
            gamma_ev = Some(value);
        } else if let Some(value) = parse_header_value(line, "# wp (eV):", line_number)? {
            plasma_frequency_ev = Some(value);
        } else if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != DRUDE_DAT_ROW_WIDTH {
                return parse_error(
                    line_number,
                    format!(
                        "drude.dat row has {} token(s), expected {DRUDE_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            omega.push(parse_f64(line_number, "omega", tokens[0])?);
            epsilon.push(Complex64::new(
                parse_f64(line_number, "epsilon real", tokens[1])?,
                parse_f64(line_number, "epsilon imaginary", tokens[2])?,
            ));
        } else if !line.trim().is_empty() && !line.trim_start().starts_with('#') {
            return parse_error(line_number, format!("unexpected drude.dat line {line:?}"));
        }
    }

    let data = DrudeDatData {
        gamma_ev: gamma_ev.ok_or_else(|| missing_drude_dat("gamma_ev"))?,
        plasma_frequency_ev: plasma_frequency_ev
            .ok_or_else(|| missing_drude_dat("plasma_frequency_ev"))?,
        omega: Array1::from_vec(omega),
        epsilon: Array1::from_vec(epsilon),
    };
    validate_drude_dat(&data)?;
    Ok(data)
}

/// Write FEFF `drude.dat` text to a file.
pub fn write_drude_dat(path: impl AsRef<Path>, data: &DrudeDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, drude_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `drude.dat` text from a file.
pub fn read_drude_dat(path: impl AsRef<Path>) -> Result<DrudeDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_drude_dat(&text)
}

fn write_drude_row(out: &mut String, fields: [f64; DRUDE_DAT_ROW_WIDTH]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 20, 10)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_drude_dat(data: &DrudeDatData) -> Result<()> {
    validate_positive("gamma_ev", data.gamma_ev, 0)?;
    validate_positive("plasma_frequency_ev", data.plasma_frequency_ev, 0)?;
    let point_count = data.point_count();
    if point_count == 0 {
        return invalid_drude_dat("rows", "at least one Drude row is required");
    }
    validate_len("epsilon", data.epsilon.len(), point_count)?;

    for (row, omega) in data.omega.iter().enumerate() {
        validate_positive("omega", *omega, row + 1)?;
    }
    for (row, epsilon) in data.epsilon.iter().enumerate() {
        validate_finite("epsilon real", epsilon.re, row + 1)?;
        validate_finite("epsilon imaginary", epsilon.im, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_drude_dat(field, format!("got {actual} value(s), expected {expected}"))
    }
}

fn validate_positive(field: &'static str, value: f64, row: usize) -> Result<()> {
    if !value.is_finite() {
        invalid_drude_dat(field, format!("row {row} value must be finite"))
    } else if value <= 0.0 {
        invalid_drude_dat(field, format!("row {row} value must be positive"))
    } else {
        Ok(())
    }
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_drude_dat(field, format!("row {row} value must be finite"))
    }
}

fn parse_header_value(line: &str, prefix: &'static str, line_number: usize) -> Result<Option<f64>> {
    if !line.starts_with(prefix) {
        return Ok(None);
    }
    let value = line[prefix.len()..].trim();
    if value.is_empty() {
        return parse_error(line_number, format!("missing {prefix} value"));
    }
    parse_f64(line_number, prefix, value).map(Some)
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("invalid {field} value {token:?}")))
}

fn missing_drude_dat(field: &'static str) -> IoError {
    invalid_drude_dat_error(field, "missing required header")
}

fn invalid_drude_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_drude_dat_error(field, message))
}

fn invalid_drude_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: DRUDE_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: DRUDE_DAT_PATH.into(),
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
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn parses_feff_fullspectrum_drude_dat() -> Result<()> {
        let parsed = parse_drude_dat(DRUDE_DAT)?;

        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.gamma_ev, 0.658);
        assert_eq!(parsed.plasma_frequency_ev, 26.417_175_80);
        assert_eq!(parsed.omega[0], 0.1);
        assert_eq!(
            parsed.epsilon[1],
            Complex64::new(-23.222_477_02, 2.807_718_847)
        );
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_drude_dat() -> Result<()> {
        let parsed = parse_drude_dat(DRUDE_DAT)?;
        let rendered = drude_dat_string(&parsed)?;

        assert_eq!(rendered, DRUDE_DAT);
        assert_eq!(parse_drude_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn derives_drude_dat_from_grid() -> Result<()> {
        let omega = array![0.1, 0.2, 0.5];
        let rendered = drude_dat_string(&drude_dat_from_grid(omega.view(), 1.0e-15, 0.075)?)?;

        assert_eq!(rendered, DRUDE_DAT);
        Ok(())
    }

    #[test]
    fn rejects_bad_drude_dat_inputs() {
        assert!(parse_drude_dat("").is_err());
        assert!(parse_drude_dat("# gam (eV): 1\n").is_err());
        assert!(parse_drude_dat("# gam (eV): 1\n# wp (eV): 2\n").is_err());
        assert!(parse_drude_dat("# gam (eV): 1\n# wp (eV): 2\n1 2\n").is_err());
        assert!(parse_drude_dat("# gam (eV): 1\n# wp (eV): 2\n1 2 3 4\n").is_err());
        assert!(parse_drude_dat("# gam (eV): 1\n# wp (eV): 2\n0 2 3\n").is_err());

        let bad = DrudeDatData {
            gamma_ev: 1.0,
            plasma_frequency_ev: 2.0,
            omega: array![1.0, 2.0],
            epsilon: array![Complex64::new(1.0, 0.0)],
        };
        assert!(drude_dat_string(&bad).is_err());
    }

    const DRUDE_DAT: &str = "# gam (eV):    0.6580000000E+00\n# wp (eV):    0.2641717580E+02\n    0.1000000000E+00   -0.8904132874E+02    0.2153112406E+02\n    0.2000000000E+00   -0.2322247702E+02    0.2807718847E+01\n    0.5000000000E+00   -0.3761114345E+01    0.1818953529E+00\n";
}
