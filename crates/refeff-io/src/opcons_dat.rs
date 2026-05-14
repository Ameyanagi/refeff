//! FEFF FULLSPECTRUM optical-constants table codec.
//!
//! `FULLSPECTRUM/opcons.f90` writes `opcons.dat`, `opconsKK.dat`, and
//! `opcons0.dat` as eight-column optical-constant tables. EELS can also read
//! `opconsKK*.dat` as an alternate spectrum source.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use refeff_core::{
    FEFF_HARTREE_EV, FullSpectrumOpticalConstants, FullSpectrumOpticalConstantsInput,
    full_spectrum_optical_constants,
};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const OPCONS_DAT_PATH: &str = "opcons.dat";
const OPCONS_DAT_ROW_WIDTH: usize = 8;

/// Parsed FEFF `opcons.dat`/`opconsKK.dat`/`opcons0.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct OpconsDatData {
    /// Header and comment lines before and around optical-constant rows.
    pub header_lines: Vec<String>,
    /// Photon energy in eV.
    pub energy_ev: Array1<f64>,
    /// Complex dielectric response minus one, written by FEFF as columns 2-3.
    pub epsilon_minus_one: Array1<Complex64>,
    /// Complex refractive index minus one, written by FEFF as columns 4-5.
    pub refractive_index_minus_one: Array1<Complex64>,
    /// FEFF `mu` absorption-coefficient column.
    pub absorption_coefficient: Array1<f64>,
    /// Normal-incidence reflectivity column.
    pub reflectivity: Array1<f64>,
    /// Energy-loss function column.
    pub loss: Array1<f64>,
}

impl OpconsDatData {
    /// Number of optical-constant rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Render FEFF-compatible optical-constants table text.
pub fn opcons_dat_string(data: &OpconsDatData) -> Result<String> {
    validate_opcons_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for ((((energy, epsilon), refractive_index), absorption), (reflectivity, loss)) in data
        .energy_ev
        .iter()
        .zip(data.epsilon_minus_one.iter())
        .zip(data.refractive_index_minus_one.iter())
        .zip(data.absorption_coefficient.iter())
        .zip(data.reflectivity.iter().zip(data.loss.iter()))
    {
        write_opcons_row(
            &mut out,
            [
                *energy,
                epsilon.re,
                epsilon.im,
                refractive_index.re,
                refractive_index.im,
                *absorption,
                *reflectivity,
                *loss,
            ],
        )?;
    }
    Ok(out)
}

/// Parse FEFF optical-constants table text.
pub fn parse_opcons_dat(text: &str) -> Result<OpconsDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut epsilon_minus_one = Vec::new();
    let mut refractive_index_minus_one = Vec::new();
    let mut absorption_coefficient = Vec::new();
    let mut reflectivity = Vec::new();
    let mut loss = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != OPCONS_DAT_ROW_WIDTH {
                return parse_error(
                    line_number,
                    format!(
                        "opcons row has {} token(s), expected {OPCONS_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            epsilon_minus_one.push(Complex64::new(
                parse_f64(line_number, "epsilon real", tokens[1])?,
                parse_f64(line_number, "epsilon imaginary", tokens[2])?,
            ));
            refractive_index_minus_one.push(Complex64::new(
                parse_f64(line_number, "refractive index real", tokens[3])?,
                parse_f64(line_number, "refractive index imaginary", tokens[4])?,
            ));
            absorption_coefficient.push(parse_f64(
                line_number,
                "absorption coefficient",
                tokens[5],
            )?);
            reflectivity.push(parse_f64(line_number, "reflectivity", tokens[6])?);
            loss.push(parse_f64(line_number, "loss", tokens[7])?);
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let data = OpconsDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        epsilon_minus_one: Array1::from_vec(epsilon_minus_one),
        refractive_index_minus_one: Array1::from_vec(refractive_index_minus_one),
        absorption_coefficient: Array1::from_vec(absorption_coefficient),
        reflectivity: Array1::from_vec(reflectivity),
        loss: Array1::from_vec(loss),
    };
    validate_opcons_dat(&data)?;
    Ok(data)
}

/// Write FEFF optical-constants table text to a file.
pub fn write_opcons_dat(path: impl AsRef<Path>, data: &OpconsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, opcons_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF optical-constants table text from a file.
pub fn read_opcons_dat(path: impl AsRef<Path>) -> Result<OpconsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_opcons_dat(&text)
}

/// Convert a core FULLSPECTRUM optical-constants result to `opcons.dat` rows.
///
/// FEFF writes photon energy in eV even though `FULLSPECTRUM/opcons.f90`
/// performs the calculation on the Hartree grid. The remaining columns are the
/// direct outputs of the optical-constants kernel.
pub fn opcons_dat_from_fullspectrum_optical_constants(
    header_lines: Vec<String>,
    constants: &FullSpectrumOpticalConstants,
) -> Result<OpconsDatData> {
    let data = OpconsDatData {
        header_lines,
        energy_ev: constants.omega.mapv(|omega| omega * FEFF_HARTREE_EV),
        epsilon_minus_one: constants.epsilon_minus_one.clone(),
        refractive_index_minus_one: constants.refractive_index_minus_one.clone(),
        absorption_coefficient: constants.absorption_coefficient.clone(),
        reflectivity: constants.reflectivity.clone(),
        loss: constants.loss.clone(),
    };
    validate_opcons_dat(&data)?;
    Ok(data)
}

/// Generate FEFF-compatible `opcons.dat` rows from `eps - 1` on a Hartree grid.
pub fn opcons_dat_from_fullspectrum_epsilon_minus_one(
    header_lines: Vec<String>,
    omega: ArrayView1<'_, f64>,
    epsilon_minus_one: ArrayView1<'_, Complex64>,
) -> Result<OpconsDatData> {
    let constants = full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
        omega,
        epsilon_minus_one,
    })
    .map_err(|source| invalid_opcons_dat_error("optical_constants", source.to_string()))?;
    opcons_dat_from_fullspectrum_optical_constants(header_lines, &constants)
}

fn write_opcons_row(out: &mut String, fields: [f64; OPCONS_DAT_ROW_WIDTH]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 16, 6)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_opcons_dat(data: &OpconsDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return invalid_opcons_dat("rows", "at least one optical-constant row is required");
    }
    validate_len("epsilon", data.epsilon_minus_one.len(), point_count)?;
    validate_len(
        "refractive index",
        data.refractive_index_minus_one.len(),
        point_count,
    )?;
    validate_len(
        "absorption coefficient",
        data.absorption_coefficient.len(),
        point_count,
    )?;
    validate_len("reflectivity", data.reflectivity.len(), point_count)?;
    validate_len("loss", data.loss.len(), point_count)?;

    for (row, value) in data.energy_ev.iter().enumerate() {
        validate_finite("energy", *value, row + 1)?;
    }
    for (row, value) in data.epsilon_minus_one.iter().enumerate() {
        validate_finite("epsilon real", value.re, row + 1)?;
        validate_finite("epsilon imaginary", value.im, row + 1)?;
    }
    for (row, value) in data.refractive_index_minus_one.iter().enumerate() {
        validate_finite("refractive index real", value.re, row + 1)?;
        validate_finite("refractive index imaginary", value.im, row + 1)?;
    }
    for (row, value) in data.absorption_coefficient.iter().enumerate() {
        validate_finite("absorption coefficient", *value, row + 1)?;
    }
    for (row, value) in data.reflectivity.iter().enumerate() {
        validate_finite("reflectivity", *value, row + 1)?;
    }
    for (row, value) in data.loss.iter().enumerate() {
        validate_finite("loss", *value, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_opcons_dat(field, format!("got {actual} value(s), expected {expected}"))
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
        invalid_opcons_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_opcons_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_opcons_dat_error(field, message))
}

fn invalid_opcons_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: OPCONS_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: OPCONS_DAT_PATH.into(),
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
    fn parses_feff_fullspectrum_opcons_dat() -> Result<()> {
        let parsed = parse_opcons_dat(OPCONS_DAT)?;

        assert_eq!(parsed.header_lines.len(), 2);
        assert_eq!(parsed.point_count(), 2);
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.epsilon_minus_one[0], Complex64::new(0.25, 0.5));
        assert_eq!(
            parsed.refractive_index_minus_one[1],
            Complex64::new(0.15, 0.25)
        );
        assert_eq!(parsed.absorption_coefficient[1], 2.0e4);
        assert_eq!(parsed.reflectivity[0], 0.012);
        assert_eq!(parsed.loss[1], 0.04);
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_opcons_dat() -> Result<()> {
        let parsed = parse_opcons_dat(OPCONS_DAT)?;
        let rendered = opcons_dat_string(&parsed)?;

        assert_eq!(rendered, OPCONS_DAT);
        assert_eq!(parse_opcons_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn renders_opcons_dat_from_arrays() -> Result<()> {
        let data = OpconsDatData {
            header_lines: Vec::new(),
            energy_ev: array![1.0],
            epsilon_minus_one: array![Complex64::new(0.2, -0.1)],
            refractive_index_minus_one: array![Complex64::new(0.08, 0.03)],
            absorption_coefficient: array![123.0],
            reflectivity: array![0.004],
            loss: array![0.05],
        };

        assert_eq!(
            opcons_dat_string(&data)?,
            "    0.100000E+01    0.200000E+00   -0.100000E+00    0.800000E-01    0.300000E-01    0.123000E+03    0.400000E-02    0.500000E-01\n"
        );
        Ok(())
    }

    #[test]
    fn converts_fullspectrum_epsilon_to_opcons_dat() -> Result<()> {
        let omega = array![0.5, 1.0];
        let epsilon_minus_one = array![Complex64::new(3.0, 4.0), Complex64::new(-0.5, 0.25)];
        let header_lines = vec!["# generated by FULLSPECTRUM/opcons.f90".to_string()];

        let data = opcons_dat_from_fullspectrum_epsilon_minus_one(
            header_lines.clone(),
            omega.view(),
            epsilon_minus_one.view(),
        )?;

        assert_eq!(data.header_lines, header_lines);
        assert_eq!(data.point_count(), 2);
        assert_close(data.energy_ev[0], 0.5 * FEFF_HARTREE_EV, 1.0e-12);
        assert_eq!(data.epsilon_minus_one[0], epsilon_minus_one[0]);
        assert_close(
            data.refractive_index_minus_one[0].re,
            1.197_368_226_935_62,
            1.0e-14,
        );
        assert_close(
            data.refractive_index_minus_one[0].im,
            0.910_179_721_124_454_7,
            1.0e-14,
        );
        assert_close(
            data.absorption_coefficient[0],
            12.551_376_312_230_127,
            1.0e-14,
        );
        assert_close(data.reflectivity[1], 0.034_392_102_279_900_92, 1.0e-14);
        assert_close(data.loss[1], 0.8, 1.0e-14);

        let rendered = opcons_dat_string(&data)?;
        let parsed = parse_opcons_dat(&rendered)?;
        assert_eq!(parsed.header_lines, data.header_lines);
        assert_eq!(parsed.point_count(), data.point_count());
        assert_close(parsed.energy_ev[0], data.energy_ev[0], 1.0e-5);
        assert_close(
            parsed.absorption_coefficient[0],
            data.absorption_coefficient[0],
            1.0e-4,
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_fullspectrum_opcons_inputs() {
        let omega = array![0.0];
        let epsilon_minus_one = array![Complex64::new(0.0, 0.0)];

        assert!(matches!(
            opcons_dat_from_fullspectrum_epsilon_minus_one(
                Vec::new(),
                omega.view(),
                epsilon_minus_one.view(),
            ),
            Err(IoError::Parse { .. })
        ));
    }

    #[test]
    fn rejects_bad_opcons_dat_inputs() {
        assert!(parse_opcons_dat("").is_err());
        assert!(parse_opcons_dat("# only header\n").is_err());
        assert!(parse_opcons_dat("1 2 3 4 5 6 7\n").is_err());
        assert!(parse_opcons_dat("1 2 3 4 5 6 7 8 9\n").is_err());
        assert!(parse_opcons_dat("1 2 3 4 5 6 7 NaN\n").is_err());

        let bad = OpconsDatData {
            header_lines: Vec::new(),
            energy_ev: array![1.0, 2.0],
            epsilon_minus_one: array![Complex64::new(1.0, 0.0)],
            refractive_index_minus_one: array![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            absorption_coefficient: array![0.0, 0.0],
            reflectivity: array![0.0, 0.0],
            loss: array![0.0, 0.0],
        };
        assert!(opcons_dat_string(&bad).is_err());
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    const OPCONS_DAT: &str = "# Cu K\n#   omega (eV)      epsilon_1       epsilon_2       n               kappa           mu (cm^(-1))    R               epsinv\n    0.100000E+02    0.250000E+00    0.500000E+00    0.100000E+00    0.200000E+00    0.100000E+04    0.120000E-01    0.200000E-01\n    0.200000E+02    0.350000E+00    0.600000E+00    0.150000E+00    0.250000E+00    0.200000E+05    0.140000E-01    0.400000E-01\n";
}
