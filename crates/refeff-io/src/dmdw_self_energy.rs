//! FEFF DMDW run-type 2 self-energy and spectral-function sidecar codecs.
//!
//! `DMDW/m_dmdw.f90` writes these files after constructing the pole-weight
//! `a2f` representation: `dmdw_Egrid.info`, `dmdw_reSE_a2F.dat`,
//! `dmdw_imSE_a2F.dat`, and `dmdw_Akw.dat`. This module keeps those text
//! boundaries typed so the solver port can target FEFF-compatible artifacts.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const DMDW_EGRID_INFO_PATH: &str = "dmdw_Egrid.info";
const DMDW_SELF_ENERGY_DAT_PATH: &str = "dmdw_*SE_a2F.dat";
const DMDW_AKW_DAT_PATH: &str = "dmdw_Akw.dat";
const DMDW_SELF_ENERGY_ROW_WIDTH: usize = 2;
const DMDW_AKW_ROW_WIDTH: usize = 5;

/// Parsed FEFF `dmdw_Egrid.info` energy-window metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdwEnergyGridInfo {
    /// Lowest printed spectral energy in meV.
    pub low_energy_mev: f64,
    /// Highest printed spectral energy in meV.
    pub high_energy_mev: f64,
    /// Spectral energy step in meV.
    pub step_mev: f64,
    /// Characteristic phonon energy `w0` in meV.
    pub characteristic_energy_mev: f64,
    /// Requested electron energy `E_k` in meV.
    pub electron_energy_mev: f64,
    /// Nearest grid energy selected for `E_k`, in meV.
    pub selected_energy_mev: f64,
}

/// Parsed FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwSelfEnergyDatData {
    /// Header/comment lines preserved before and around numeric rows.
    pub header_lines: Vec<String>,
    /// Self-energy sample energy in eV.
    pub energy_ev: Array1<f64>,
    /// Real or imaginary self-energy value in eV.
    pub value_ev: Array1<f64>,
}

/// Parsed FEFF `dmdw_Akw.dat` spectral-function table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwAkwDatData {
    /// Spectral-function energy in meV.
    pub energy_mev: Array1<f64>,
    /// Spectral-function magnitude.
    pub magnitude: Array1<f64>,
    /// Spectral-function phase in radians.
    pub phase: Array1<f64>,
    /// Real spectral-function component.
    pub real: Array1<f64>,
    /// Imaginary spectral-function component.
    pub imaginary: Array1<f64>,
}

impl DmdwSelfEnergyDatData {
    /// Number of self-energy samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

impl DmdwAkwDatData {
    /// Number of spectral-function samples.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_mev.len()
    }
}

/// Parse FEFF `dmdw_Egrid.info` text.
pub fn parse_dmdw_egrid_info(text: &str) -> Result<DmdwEnergyGridInfo> {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.len() < 4 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            lines.len() + 1,
            format!(
                "dmdw_Egrid.info has {} line(s), expected at least 4",
                lines.len()
            ),
        );
    }

    let low_high = numeric_values(DMDW_EGRID_INFO_PATH, 2, lines[1])?;
    let step_w0 = numeric_values(DMDW_EGRID_INFO_PATH, 3, lines[2])?;
    let electron_selected = numeric_values(DMDW_EGRID_INFO_PATH, 4, lines[3])?;
    if low_high.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            2,
            format!(
                "low/high line has {} numeric value(s), expected 2",
                low_high.len()
            ),
        );
    }
    if step_w0.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            3,
            format!(
                "step/w0 line has {} numeric value(s), expected 2",
                step_w0.len()
            ),
        );
    }
    if electron_selected.len() != 2 {
        return parse_error(
            DMDW_EGRID_INFO_PATH,
            4,
            format!(
                "electron/grid line has {} numeric value(s), expected 2",
                electron_selected.len()
            ),
        );
    }

    let data = DmdwEnergyGridInfo {
        low_energy_mev: low_high[0],
        high_energy_mev: low_high[1],
        step_mev: step_w0[0],
        characteristic_energy_mev: step_w0[1],
        electron_energy_mev: electron_selected[0],
        selected_energy_mev: electron_selected[1],
    };
    validate_dmdw_egrid_info(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_Egrid.info` text.
pub fn dmdw_egrid_info_string(data: &DmdwEnergyGridInfo) -> Result<String> {
    validate_dmdw_egrid_info(data)?;

    let mut out = String::new();
    writeln!(out, "#  Energies printed in meV")?;
    writeln!(
        out,
        "#  lowE {:10.3} highE {:10.3}",
        data.low_energy_mev, data.high_energy_mev
    )?;
    writeln!(
        out,
        "#  dE  = {:10.3} w0 = {:10.3}",
        data.step_mev, data.characteristic_energy_mev
    )?;
    writeln!(
        out,
        "#  Ek  = {:10.3} --> E = {:10.3}",
        data.electron_energy_mev, data.selected_energy_mev
    )?;
    Ok(out)
}

/// Read FEFF `dmdw_Egrid.info` from disk.
pub fn read_dmdw_egrid_info(path: impl AsRef<Path>) -> Result<DmdwEnergyGridInfo> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_egrid_info(&text)
}

/// Write FEFF `dmdw_Egrid.info` text to disk.
pub fn write_dmdw_egrid_info(path: impl AsRef<Path>, data: &DmdwEnergyGridInfo) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_egrid_info_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` text.
pub fn parse_dmdw_self_energy_dat(text: &str) -> Result<DmdwSelfEnergyDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut value_ev = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != DMDW_SELF_ENERGY_ROW_WIDTH {
                return parse_error(
                    DMDW_SELF_ENERGY_DAT_PATH,
                    line_number,
                    format!(
                        "DMDW self-energy row has {} token(s), expected {DMDW_SELF_ENERGY_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            energy_ev.push(parse_f64(
                DMDW_SELF_ENERGY_DAT_PATH,
                line_number,
                "energy",
                tokens[0],
            )?);
            value_ev.push(parse_f64(
                DMDW_SELF_ENERGY_DAT_PATH,
                line_number,
                "self-energy value",
                tokens[1],
            )?);
        } else if !line.trim().is_empty() {
            header_lines.push(raw.to_string());
        }
    }

    let data = DmdwSelfEnergyDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        value_ev: Array1::from_vec(value_ev),
    };
    validate_dmdw_self_energy_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible self-energy table text.
pub fn dmdw_self_energy_dat_string(data: &DmdwSelfEnergyDatData) -> Result<String> {
    validate_dmdw_self_energy_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (&energy, &value) in data.energy_ev.iter().zip(data.value_ev.iter()) {
        write_fortran_zero_scaled_exp(&mut out, energy, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, value, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF `dmdw_reSE_a2F.dat` or `dmdw_imSE_a2F.dat` text from disk.
pub fn read_dmdw_self_energy_dat(path: impl AsRef<Path>) -> Result<DmdwSelfEnergyDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_self_energy_dat(&text)
}

/// Write FEFF self-energy table text to disk.
pub fn write_dmdw_self_energy_dat(
    path: impl AsRef<Path>,
    data: &DmdwSelfEnergyDatData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_self_energy_dat_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `dmdw_Akw.dat` text.
pub fn parse_dmdw_akw_dat(text: &str) -> Result<DmdwAkwDatData> {
    let mut energy_mev = Vec::new();
    let mut magnitude = Vec::new();
    let mut phase = Vec::new();
    let mut real = Vec::new();
    let mut imaginary = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != DMDW_AKW_ROW_WIDTH {
            return parse_error(
                DMDW_AKW_DAT_PATH,
                line_number,
                format!(
                    "dmdw_Akw.dat row has {} token(s), expected {DMDW_AKW_ROW_WIDTH}",
                    tokens.len()
                ),
            );
        }
        energy_mev.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "energy",
            tokens[0],
        )?);
        magnitude.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "magnitude",
            tokens[1],
        )?);
        phase.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "phase",
            tokens[2],
        )?);
        real.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "real",
            tokens[3],
        )?);
        imaginary.push(parse_f64(
            DMDW_AKW_DAT_PATH,
            line_number,
            "imaginary",
            tokens[4],
        )?);
    }

    let data = DmdwAkwDatData {
        energy_mev: Array1::from_vec(energy_mev),
        magnitude: Array1::from_vec(magnitude),
        phase: Array1::from_vec(phase),
        real: Array1::from_vec(real),
        imaginary: Array1::from_vec(imaginary),
    };
    validate_dmdw_akw_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_Akw.dat` text.
pub fn dmdw_akw_dat_string(data: &DmdwAkwDatData) -> Result<String> {
    validate_dmdw_akw_dat(data)?;

    let mut out = String::new();
    for ((((&energy, &magnitude), &phase), &real), &imaginary) in data
        .energy_mev
        .iter()
        .zip(data.magnitude.iter())
        .zip(data.phase.iter())
        .zip(data.real.iter())
        .zip(data.imaginary.iter())
    {
        writeln!(
            out,
            "{energy:20.10}{magnitude:20.10}{phase:20.10}{real:20.10}{imaginary:20.10}"
        )?;
    }
    Ok(out)
}

/// Read FEFF `dmdw_Akw.dat` text from disk.
pub fn read_dmdw_akw_dat(path: impl AsRef<Path>) -> Result<DmdwAkwDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_akw_dat(&text)
}

/// Write FEFF `dmdw_Akw.dat` text to disk.
pub fn write_dmdw_akw_dat(path: impl AsRef<Path>, data: &DmdwAkwDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_akw_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_dmdw_egrid_info(data: &DmdwEnergyGridInfo) -> Result<()> {
    validate_finite_field(DMDW_EGRID_INFO_PATH, "low_energy_mev", data.low_energy_mev)?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "high_energy_mev",
        data.high_energy_mev,
    )?;
    validate_finite_field(DMDW_EGRID_INFO_PATH, "step_mev", data.step_mev)?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "characteristic_energy_mev",
        data.characteristic_energy_mev,
    )?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "electron_energy_mev",
        data.electron_energy_mev,
    )?;
    validate_finite_field(
        DMDW_EGRID_INFO_PATH,
        "selected_energy_mev",
        data.selected_energy_mev,
    )?;
    if data.high_energy_mev < data.low_energy_mev {
        return invalid_data(
            DMDW_EGRID_INFO_PATH,
            "high_energy_mev",
            "high energy must be greater than or equal to low energy",
        );
    }
    if data.step_mev <= 0.0 {
        return invalid_data(DMDW_EGRID_INFO_PATH, "step_mev", "step must be positive");
    }
    if data.characteristic_energy_mev <= 0.0 {
        return invalid_data(
            DMDW_EGRID_INFO_PATH,
            "characteristic_energy_mev",
            "w0 must be positive",
        );
    }
    Ok(())
}

fn validate_dmdw_self_energy_dat(data: &DmdwSelfEnergyDatData) -> Result<()> {
    validate_header_lines(DMDW_SELF_ENERGY_DAT_PATH, &data.header_lines)?;
    if data.point_count() == 0 {
        return invalid_data(
            DMDW_SELF_ENERGY_DAT_PATH,
            "rows",
            "at least one self-energy row is required",
        );
    }
    if data.value_ev.len() != data.point_count() {
        return invalid_data(
            DMDW_SELF_ENERGY_DAT_PATH,
            "value_ev",
            format!(
                "got {} value(s), expected {}",
                data.value_ev.len(),
                data.point_count()
            ),
        );
    }
    validate_array(DMDW_SELF_ENERGY_DAT_PATH, "energy", &data.energy_ev)?;
    validate_array(
        DMDW_SELF_ENERGY_DAT_PATH,
        "self-energy value",
        &data.value_ev,
    )
}

fn validate_dmdw_akw_dat(data: &DmdwAkwDatData) -> Result<()> {
    if data.point_count() == 0 {
        return invalid_data(
            DMDW_AKW_DAT_PATH,
            "rows",
            "at least one spectral-function row is required",
        );
    }
    validate_length(
        DMDW_AKW_DAT_PATH,
        "magnitude",
        data.magnitude.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "phase",
        data.phase.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "real",
        data.real.len(),
        data.point_count(),
    )?;
    validate_length(
        DMDW_AKW_DAT_PATH,
        "imaginary",
        data.imaginary.len(),
        data.point_count(),
    )?;
    validate_array(DMDW_AKW_DAT_PATH, "energy", &data.energy_mev)?;
    validate_array(DMDW_AKW_DAT_PATH, "magnitude", &data.magnitude)?;
    validate_array(DMDW_AKW_DAT_PATH, "phase", &data.phase)?;
    validate_array(DMDW_AKW_DAT_PATH, "real", &data.real)?;
    validate_array(DMDW_AKW_DAT_PATH, "imaginary", &data.imaginary)
}

fn validate_header_lines(path: &'static str, lines: &[String]) -> Result<()> {
    for (index, line) in lines.iter().enumerate() {
        if line.contains(['\n', '\r']) {
            return invalid_data(
                path,
                "header_lines",
                format!("header line {} contains an embedded newline", index + 1),
            );
        }
    }
    Ok(())
}

fn validate_length(
    path: &'static str,
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_data(
            path,
            field,
            format!("got {actual} value(s), expected {expected}"),
        )
    }
}

fn validate_array(path: &'static str, field: &'static str, values: &Array1<f64>) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_data(
                path,
                field,
                format!("row {} value must be finite", index + 1),
            );
        }
    }
    Ok(())
}

fn validate_finite_field(path: &'static str, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_data(path, field, "value must be finite")
    }
}

fn numeric_values(path: &'static str, line: usize, text: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for token in text.split_whitespace() {
        if let Some(value) = parse_numeric_token(path, line, token) {
            values.push(value?);
        }
    }
    Ok(values)
}

fn is_numeric_token(token: &str) -> bool {
    parse_numeric_token(DMDW_SELF_ENERGY_DAT_PATH, 0, token).is_some_and(|value| value.is_ok())
}

fn parse_numeric_token(path: &'static str, line: usize, token: &str) -> Option<Result<f64>> {
    let normalized = token.replace(['D', 'd'], "E");
    normalized
        .parse::<f64>()
        .map(|value| {
            if value.is_finite() {
                Ok(value)
            } else {
                parse_error(path, line, "numeric value must be finite")
            }
        })
        .ok()
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error_value(
            path,
            line,
            format!("{field} value must be finite"),
        ))
    }
}

fn invalid_data<T>(
    path: &'static str,
    field: &'static str,
    message: impl Into<String>,
) -> Result<T> {
    Err(IoError::Parse {
        path: path.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn parse_error<T>(path: &'static str, line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(path, line, message))
}

fn parse_error_value(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const DMDW_EGRID_INFO: &str = "\
#  Energies printed in meV
#  lowE   -125.000 highE    175.000
#  dE  =      0.030 w0 =     15.000
#  Ek  =      5.000 --> E =      4.990
";

    const DMDW_RESE_DAT: &str = "\
#  Real part of the Self-energy
 -1.5000000000D-01  2.5000000000D-03
  0.0000000000D+00  0.0000000000D+00
  1.5000000000D-01 -2.5000000000D-03
";

    const DMDW_AKW_DAT: &str = "\
      -150.0000000000        0.0100000000       -1.5700000000        0.0000000000       -0.0100000000
         0.0000000000        0.5000000000        0.0000000000        0.5000000000        0.0000000000
       150.0000000000        0.0100000000        1.5700000000        0.0000000000        0.0100000000
";

    #[test]
    fn parses_and_renders_dmdw_egrid_info() -> Result<()> {
        let parsed = parse_dmdw_egrid_info(DMDW_EGRID_INFO)?;
        assert_eq!(
            parsed,
            DmdwEnergyGridInfo {
                low_energy_mev: -125.0,
                high_energy_mev: 175.0,
                step_mev: 0.03,
                characteristic_energy_mev: 15.0,
                electron_energy_mev: 5.0,
                selected_energy_mev: 4.99,
            }
        );

        let rendered = dmdw_egrid_info_string(&parsed)?;
        let reparsed = parse_dmdw_egrid_info(&rendered)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn parses_and_renders_dmdw_self_energy_dat() -> Result<()> {
        let parsed = parse_dmdw_self_energy_dat(DMDW_RESE_DAT)?;
        assert_eq!(parsed.header_lines, vec!["#  Real part of the Self-energy"]);
        assert_eq!(parsed.energy_ev, array![-0.15, 0.0, 0.15]);
        assert_eq!(parsed.value_ev, array![0.0025, 0.0, -0.0025]);

        let rendered = dmdw_self_energy_dat_string(&parsed)?;
        let reparsed = parse_dmdw_self_energy_dat(&rendered)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn parses_and_renders_dmdw_akw_dat() -> Result<()> {
        let parsed = parse_dmdw_akw_dat(DMDW_AKW_DAT)?;
        assert_eq!(parsed.energy_mev, array![-150.0, 0.0, 150.0]);
        assert_eq!(parsed.magnitude, array![0.01, 0.5, 0.01]);
        assert_eq!(parsed.phase, array![-1.57, 0.0, 1.57]);
        assert_eq!(parsed.real, array![0.0, 0.5, 0.0]);
        assert_eq!(parsed.imaginary, array![-0.01, 0.0, 0.01]);

        let rendered = dmdw_akw_dat_string(&parsed)?;
        let reparsed = parse_dmdw_akw_dat(&rendered)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn rejects_invalid_dmdw_self_energy_sidecars() {
        assert!(parse_dmdw_egrid_info("# too short\n").is_err());
        assert!(
            dmdw_egrid_info_string(&DmdwEnergyGridInfo {
                low_energy_mev: 1.0,
                high_energy_mev: 0.0,
                step_mev: 0.1,
                characteristic_energy_mev: 1.0,
                electron_energy_mev: 0.0,
                selected_energy_mev: 0.0,
            })
            .is_err()
        );
        assert!(parse_dmdw_self_energy_dat("1.0 2.0 3.0\n").is_err());
        assert!(parse_dmdw_self_energy_dat("1.0 NaN\n").is_err());
        assert!(parse_dmdw_akw_dat("1.0 2.0\n").is_err());
        assert!(parse_dmdw_akw_dat("1.0 2.0 3.0 4.0 inf\n").is_err());

        let bad = DmdwAkwDatData {
            energy_mev: array![0.0],
            magnitude: array![1.0, 2.0],
            phase: array![0.0],
            real: array![1.0],
            imaginary: array![0.0],
        };
        assert!(dmdw_akw_dat_string(&bad).is_err());
    }
}
