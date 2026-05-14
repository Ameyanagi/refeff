//! FEFF `eps.dat` dielectric-function table codec.
//!
//! `FULLSPECTRUM/fullspectrum.f90` writes `eps.dat` as a six-column table:
//! omega, complex total dielectric response, complex atomic-background
//! response, and the scalar conductivity-like `sigma` column. The complex
//! Fortran values are expanded as real/imaginary adjacent columns.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use refeff_core::{
    FullSpectrumScatteringDielectric, FullSpectrumScatteringDielectricInput,
    full_spectrum_scattering_to_dielectric,
};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const EPS_DAT_PATH: &str = "eps.dat";
const EPS_DAT_ROW_WIDTH: usize = 6;

/// Parsed FEFF `eps.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EpsDatData {
    /// Header or comment lines before and around the numeric table.
    pub header_lines: Vec<String>,
    /// FULLSPECTRUM energy grid `omega`, in Hartree.
    pub omega: Array1<f64>,
    /// Total bound-charge dielectric response, stored as `eps - 1`.
    pub epsilon: Array1<Complex64>,
    /// Atomic-background dielectric response, stored as `eps0 - 1`.
    pub background_epsilon: Array1<Complex64>,
    /// FEFF `sigma` column.
    pub sigma: Array1<f64>,
}

impl EpsDatData {
    /// Number of dielectric-function samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.omega.len()
    }
}

/// Render FEFF-compatible `eps.dat` text.
pub fn eps_dat_string(data: &EpsDatData) -> Result<String> {
    validate_eps_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (((omega, epsilon), background), sigma) in data
        .omega
        .iter()
        .zip(data.epsilon.iter())
        .zip(data.background_epsilon.iter())
        .zip(data.sigma.iter())
    {
        write_eps_row(
            &mut out,
            [
                *omega,
                epsilon.re,
                epsilon.im,
                background.re,
                background.im,
                *sigma,
            ],
        )?;
    }
    Ok(out)
}

/// Parse FEFF `eps.dat` text.
pub fn parse_eps_dat(text: &str) -> Result<EpsDatData> {
    let mut header_lines = Vec::new();
    let mut omega = Vec::new();
    let mut epsilon = Vec::new();
    let mut background_epsilon = Vec::new();
    let mut sigma = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != EPS_DAT_ROW_WIDTH {
                return parse_error(
                    line_number,
                    format!(
                        "eps.dat row has {} token(s), expected {EPS_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            omega.push(parse_f64(line_number, "omega", tokens[0])?);
            epsilon.push(Complex64::new(
                parse_f64(line_number, "epsilon real", tokens[1])?,
                parse_f64(line_number, "epsilon imaginary", tokens[2])?,
            ));
            background_epsilon.push(Complex64::new(
                parse_f64(line_number, "background epsilon real", tokens[3])?,
                parse_f64(line_number, "background epsilon imaginary", tokens[4])?,
            ));
            sigma.push(parse_f64(line_number, "sigma", tokens[5])?);
        } else if !line.trim().is_empty() {
            header_lines.push(line.to_string());
        }
    }

    let data = EpsDatData {
        header_lines,
        omega: Array1::from_vec(omega),
        epsilon: Array1::from_vec(epsilon),
        background_epsilon: Array1::from_vec(background_epsilon),
        sigma: Array1::from_vec(sigma),
    };
    validate_eps_dat(&data)?;
    Ok(data)
}

/// Write FEFF `eps.dat` text to a file.
pub fn write_eps_dat(path: impl AsRef<Path>, data: &EpsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, eps_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `eps.dat` text from a file.
pub fn read_eps_dat(path: impl AsRef<Path>) -> Result<EpsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_eps_dat(&text)
}

/// Convert a core FULLSPECTRUM dielectric contribution to `eps.dat` rows.
pub fn eps_dat_from_fullspectrum_scattering_dielectric(
    header_lines: Vec<String>,
    dielectric: &FullSpectrumScatteringDielectric,
) -> Result<EpsDatData> {
    let data = EpsDatData {
        header_lines,
        omega: dielectric.omega.clone(),
        epsilon: dielectric.epsilon_minus_one.clone(),
        background_epsilon: dielectric.background_epsilon_minus_one.clone(),
        sigma: dielectric.sigma.clone(),
    };
    validate_eps_dat(&data)?;
    Ok(data)
}

/// Generate FEFF-compatible `eps.dat` rows from assembled scattering factors.
pub fn eps_dat_from_fullspectrum_scattering_factors(
    header_lines: Vec<String>,
    number_density: f64,
    omega: ArrayView1<'_, f64>,
    scattering_factor: ArrayView1<'_, Complex64>,
    background_scattering_factor: ArrayView1<'_, Complex64>,
) -> Result<EpsDatData> {
    let dielectric =
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density,
            omega,
            scattering_factor,
            background_scattering_factor,
        })
        .map_err(|source| invalid_eps_dat_error("scattering_to_dielectric", source.to_string()))?;
    eps_dat_from_fullspectrum_scattering_dielectric(header_lines, &dielectric)
}

fn write_eps_row(out: &mut String, fields: [f64; EPS_DAT_ROW_WIDTH]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 20, 10)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_eps_dat(data: &EpsDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return invalid_eps_dat("rows", "at least one dielectric-function row is required");
    }
    validate_len("epsilon", data.epsilon.len(), point_count)?;
    validate_len(
        "background epsilon",
        data.background_epsilon.len(),
        point_count,
    )?;
    validate_len("sigma", data.sigma.len(), point_count)?;

    for (row, value) in data.omega.iter().enumerate() {
        validate_finite("omega", *value, row + 1)?;
    }
    for (row, value) in data.epsilon.iter().enumerate() {
        validate_finite("epsilon real", value.re, row + 1)?;
        validate_finite("epsilon imaginary", value.im, row + 1)?;
    }
    for (row, value) in data.background_epsilon.iter().enumerate() {
        validate_finite("background epsilon real", value.re, row + 1)?;
        validate_finite("background epsilon imaginary", value.im, row + 1)?;
    }
    for (row, value) in data.sigma.iter().enumerate() {
        validate_finite("sigma", *value, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_eps_dat(field, format!("got {actual} value(s), expected {expected}"))
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
        invalid_eps_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_eps_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_eps_dat_error(field, message))
}

fn invalid_eps_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: EPS_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: EPS_DAT_PATH.into(),
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
    fn parses_feff_fullspectrum_eps_dat() -> Result<()> {
        let parsed = parse_eps_dat(EPS_DAT)?;

        assert_eq!(parsed.header_lines.len(), 1);
        assert_eq!(parsed.point_count(), 2);
        assert_eq!(parsed.omega[0], 1.0);
        assert_eq!(parsed.epsilon[0], Complex64::new(1.1, -0.25));
        assert_eq!(parsed.background_epsilon[1], Complex64::new(0.95, -0.01));
        assert_eq!(parsed.sigma[1], 0.006);
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_eps_dat() -> Result<()> {
        let parsed = parse_eps_dat(EPS_DAT)?;
        let rendered = eps_dat_string(&parsed)?;

        assert_eq!(rendered, EPS_DAT);
        assert_eq!(parse_eps_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn renders_eps_dat_from_arrays() -> Result<()> {
        let data = EpsDatData {
            header_lines: Vec::new(),
            omega: array![0.1],
            epsilon: array![Complex64::new(1.0, 2.0)],
            background_epsilon: array![Complex64::new(0.5, -0.25)],
            sigma: array![0.0125],
        };

        assert_eq!(
            eps_dat_string(&data)?,
            "    0.1000000000E+00    0.1000000000E+01    0.2000000000E+01    0.5000000000E+00   -0.2500000000E+00    0.1250000000E-01\n"
        );
        Ok(())
    }

    #[test]
    fn converts_fullspectrum_scattering_factors_to_eps_dat() -> Result<()> {
        let omega = array![1.0, 2.0];
        let scattering_factor = array![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)];
        let background_scattering_factor =
            array![Complex64::new(0.25, 0.5), Complex64::new(0.1, 0.05)];
        let header_lines = vec!["# generated by FULLSPECTRUM/fullspectrum.f90".to_string()];

        let data = eps_dat_from_fullspectrum_scattering_factors(
            header_lines.clone(),
            0.01,
            omega.view(),
            scattering_factor.view(),
            background_scattering_factor.view(),
        )?;

        assert_eq!(data.header_lines, header_lines);
        assert_eq!(data.point_count(), 2);
        assert_eq!(data.omega[1], omega[1]);
        assert_close(data.epsilon[0].re, -0.125_663_706_143_591_74, 1.0e-15);
        assert_close(data.epsilon[0].im, -0.251_327_412_287_183_47, 1.0e-15);
        assert_close(
            data.background_epsilon[0].re,
            -0.031_415_926_535_897_934,
            1.0e-15,
        );
        assert_close(
            data.background_epsilon[0].im,
            -0.062_831_853_071_795_87,
            1.0e-15,
        );
        assert_close(data.sigma[0], -0.052_118_634_285_441_2, 1.0e-15);

        let rendered = eps_dat_string(&data)?;
        let parsed = parse_eps_dat(&rendered)?;
        assert_eq!(parsed.header_lines, data.header_lines);
        assert_eq!(parsed.point_count(), data.point_count());
        assert_close(parsed.epsilon[0].re, data.epsilon[0].re, 1.0e-10);
        assert_close(parsed.sigma[0], data.sigma[0], 1.0e-10);
        Ok(())
    }

    #[test]
    fn rejects_bad_fullspectrum_eps_inputs() {
        let omega = array![0.0];
        let scattering_factor = array![Complex64::new(0.1, 0.2)];
        let background_scattering_factor = array![Complex64::new(0.1, 0.2)];

        assert!(matches!(
            eps_dat_from_fullspectrum_scattering_factors(
                Vec::new(),
                0.01,
                omega.view(),
                scattering_factor.view(),
                background_scattering_factor.view(),
            ),
            Err(IoError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_bad_eps_dat_inputs() {
        assert!(parse_eps_dat("").is_err());
        assert!(parse_eps_dat("# only header\n").is_err());
        assert!(parse_eps_dat("1 2 3 4 5\n").is_err());
        assert!(parse_eps_dat("1 2 3 4 5 6 7\n").is_err());
        assert!(parse_eps_dat("1 2 NaN 4 5 6\n").is_err());

        let bad = EpsDatData {
            header_lines: Vec::new(),
            omega: array![1.0, 2.0],
            epsilon: array![Complex64::new(1.0, 0.0)],
            background_epsilon: array![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            sigma: array![0.0, 0.0],
        };
        assert!(eps_dat_string(&bad).is_err());
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    const EPS_DAT: &str = "# FULLSPECTRUM eps.dat\n    0.1000000000E+01    0.1100000000E+01   -0.2500000000E+00    0.1000000000E+01    0.3000000000E-01    0.4000000000E-02\n    0.2000000000E+01    0.1200000000E+01   -0.5000000000E+00    0.9500000000E+00   -0.1000000000E-01    0.6000000000E-02\n";
}
