//! FEFF `sumrules.dat` optical sum-rule table codec.
//!
//! `FULLSPECTRUM/sumrules.f90` reads `opconsKK.dat` and writes a seven-column
//! cumulative table with epsilon, absorption, loss, and refractive-index sum
//! rules. The helper in this module preserves the fixed-width `7e24.10` text
//! layout and can derive the table directly from parsed optical constants.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use refeff_core::{FullSpectrumSumRules, FullSpectrumSumRulesInput, full_spectrum_sum_rules};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;
use crate::opcons_dat::OpconsDatData;

const SUMRULES_DAT_PATH: &str = "sumrules.dat";
const SUMRULES_DAT_ROW_WIDTH: usize = 7;

/// Parsed FEFF `sumrules.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct SumRulesDatData {
    /// Header or comment lines before and around the numeric table.
    pub header_lines: Vec<String>,
    /// Photon energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// Cumulative `epsilon_2` sum-rule effective electron count.
    pub epsilon2_effective_electrons: Array1<f64>,
    /// Cumulative absorption-coefficient sum-rule effective electron count.
    pub absorption_effective_electrons: Array1<f64>,
    /// Cumulative loss-function sum-rule effective electron count.
    pub loss_effective_electrons: Array1<f64>,
    /// Cumulative `mu * (n - 1)` sum-rule column.
    pub absorption_refractive_sum: Array1<f64>,
    /// Cumulative `(n - 1)` signed-to-absolute integral ratio.
    pub refractive_index_sum_ratio: Array1<f64>,
    /// Cumulative logarithmic loss-function moment ratio.
    pub log_loss_moment_ratio: Array1<f64>,
}

impl SumRulesDatData {
    /// Number of cumulative sum-rule rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

impl From<FullSpectrumSumRules> for SumRulesDatData {
    fn from(rules: FullSpectrumSumRules) -> Self {
        Self {
            header_lines: Vec::new(),
            energy_ev: rules.energy_ev,
            epsilon2_effective_electrons: rules.epsilon2_effective_electrons,
            absorption_effective_electrons: rules.absorption_effective_electrons,
            loss_effective_electrons: rules.loss_effective_electrons,
            absorption_refractive_sum: rules.absorption_refractive_sum,
            refractive_index_sum_ratio: rules.refractive_index_sum_ratio,
            log_loss_moment_ratio: rules.log_loss_moment_ratio,
        }
    }
}

/// Compute FEFF-compatible `sumrules.dat` contents from `opconsKK.dat` data.
pub fn sumrules_dat_from_opcons(
    number_density: f64,
    opcons: &OpconsDatData,
) -> Result<SumRulesDatData> {
    full_spectrum_sum_rules(FullSpectrumSumRulesInput {
        number_density,
        energy_ev: opcons.energy_ev.view(),
        epsilon_minus_one: opcons.epsilon_minus_one.view(),
        refractive_index_minus_one: opcons.refractive_index_minus_one.view(),
        absorption_coefficient: opcons.absorption_coefficient.view(),
    })
    .map(SumRulesDatData::from)
    .map_err(|error| invalid_sumrules_dat_error("sum_rules", error.to_string()))
}

/// Render FEFF-compatible `sumrules.dat` text.
pub fn sumrules_dat_string(data: &SumRulesDatData) -> Result<String> {
    validate_sumrules_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (
        (((((energy, epsilon2), absorption), loss), absorption_refractive), refractive_ratio),
        log_loss_ratio,
    ) in data
        .energy_ev
        .iter()
        .zip(data.epsilon2_effective_electrons.iter())
        .zip(data.absorption_effective_electrons.iter())
        .zip(data.loss_effective_electrons.iter())
        .zip(data.absorption_refractive_sum.iter())
        .zip(data.refractive_index_sum_ratio.iter())
        .zip(data.log_loss_moment_ratio.iter())
    {
        write_sumrules_row(
            &mut out,
            [
                *energy,
                *epsilon2,
                *absorption,
                *loss,
                *absorption_refractive,
                *refractive_ratio,
                *log_loss_ratio,
            ],
        )?;
    }
    Ok(out)
}

/// Parse FEFF `sumrules.dat` text.
pub fn parse_sumrules_dat(text: &str) -> Result<SumRulesDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut epsilon2_effective_electrons = Vec::new();
    let mut absorption_effective_electrons = Vec::new();
    let mut loss_effective_electrons = Vec::new();
    let mut absorption_refractive_sum = Vec::new();
    let mut refractive_index_sum_ratio = Vec::new();
    let mut log_loss_moment_ratio = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != SUMRULES_DAT_ROW_WIDTH {
                return parse_error(
                    line_number,
                    format!(
                        "sumrules.dat row has {} token(s), expected {SUMRULES_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                );
            }
            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            epsilon2_effective_electrons.push(parse_f64(
                line_number,
                "epsilon2 effective electrons",
                tokens[1],
            )?);
            absorption_effective_electrons.push(parse_f64(
                line_number,
                "absorption effective electrons",
                tokens[2],
            )?);
            loss_effective_electrons.push(parse_f64(
                line_number,
                "loss effective electrons",
                tokens[3],
            )?);
            absorption_refractive_sum.push(parse_f64(
                line_number,
                "absorption refractive sum",
                tokens[4],
            )?);
            refractive_index_sum_ratio.push(parse_f64(
                line_number,
                "refractive index sum ratio",
                tokens[5],
            )?);
            log_loss_moment_ratio.push(parse_f64(line_number, "log loss moment ratio", tokens[6])?);
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let data = SumRulesDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        epsilon2_effective_electrons: Array1::from_vec(epsilon2_effective_electrons),
        absorption_effective_electrons: Array1::from_vec(absorption_effective_electrons),
        loss_effective_electrons: Array1::from_vec(loss_effective_electrons),
        absorption_refractive_sum: Array1::from_vec(absorption_refractive_sum),
        refractive_index_sum_ratio: Array1::from_vec(refractive_index_sum_ratio),
        log_loss_moment_ratio: Array1::from_vec(log_loss_moment_ratio),
    };
    validate_sumrules_dat(&data)?;
    Ok(data)
}

/// Write FEFF `sumrules.dat` text to a file.
pub fn write_sumrules_dat(path: impl AsRef<Path>, data: &SumRulesDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, sumrules_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `sumrules.dat` text from a file.
pub fn read_sumrules_dat(path: impl AsRef<Path>) -> Result<SumRulesDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_sumrules_dat(&text)
}

fn write_sumrules_row(out: &mut String, fields: [f64; SUMRULES_DAT_ROW_WIDTH]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 24, 10)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_sumrules_dat(data: &SumRulesDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return invalid_sumrules_dat("rows", "at least one sum-rule row is required");
    }
    validate_len(
        "epsilon2 effective electrons",
        data.epsilon2_effective_electrons.len(),
        point_count,
    )?;
    validate_len(
        "absorption effective electrons",
        data.absorption_effective_electrons.len(),
        point_count,
    )?;
    validate_len(
        "loss effective electrons",
        data.loss_effective_electrons.len(),
        point_count,
    )?;
    validate_len(
        "absorption refractive sum",
        data.absorption_refractive_sum.len(),
        point_count,
    )?;
    validate_len(
        "refractive index sum ratio",
        data.refractive_index_sum_ratio.len(),
        point_count,
    )?;
    validate_len(
        "log loss moment ratio",
        data.log_loss_moment_ratio.len(),
        point_count,
    )?;

    for (row, value) in data.energy_ev.iter().enumerate() {
        validate_finite("energy", *value, row + 1)?;
    }
    for (row, value) in data.epsilon2_effective_electrons.iter().enumerate() {
        validate_finite("epsilon2 effective electrons", *value, row + 1)?;
    }
    for (row, value) in data.absorption_effective_electrons.iter().enumerate() {
        validate_finite("absorption effective electrons", *value, row + 1)?;
    }
    for (row, value) in data.loss_effective_electrons.iter().enumerate() {
        validate_finite("loss effective electrons", *value, row + 1)?;
    }
    for (row, value) in data.absorption_refractive_sum.iter().enumerate() {
        validate_finite("absorption refractive sum", *value, row + 1)?;
    }
    for (row, value) in data.refractive_index_sum_ratio.iter().enumerate() {
        validate_finite("refractive index sum ratio", *value, row + 1)?;
    }
    for (row, value) in data.log_loss_moment_ratio.iter().enumerate() {
        validate_finite("log loss moment ratio", *value, row + 1)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_sumrules_dat(field, format!("got {actual} value(s), expected {expected}"))
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
        invalid_sumrules_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_sumrules_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_sumrules_dat_error(field, message))
}

fn invalid_sumrules_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: SUMRULES_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: SUMRULES_DAT_PATH.into(),
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
    fn parses_feff_fullspectrum_sumrules_dat() -> Result<()> {
        let parsed = parse_sumrules_dat(SUMRULES_DAT)?;

        assert_eq!(parsed.header_lines.len(), 1);
        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.epsilon2_effective_electrons[2], 0.214_375_530_8);
        assert_eq!(parsed.absorption_effective_electrons[1], 54.002_669_93);
        assert_eq!(parsed.loss_effective_electrons[0], 0.007_297_890_411);
        assert_eq!(parsed.absorption_refractive_sum[2], 4.140_204_695);
        assert_eq!(parsed.refractive_index_sum_ratio[2], 1.0);
        assert_eq!(parsed.log_loss_moment_ratio[1], -0.868_798_069);
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_sumrules_dat() -> Result<()> {
        let parsed = parse_sumrules_dat(SUMRULES_DAT)?;
        let rendered = sumrules_dat_string(&parsed)?;

        assert_eq!(rendered, SUMRULES_DAT);
        assert_eq!(parse_sumrules_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn derives_sumrules_dat_from_opcons_arrays() -> Result<()> {
        let opcons = OpconsDatData {
            header_lines: Vec::new(),
            energy_ev: array![10.0, 20.0, 40.0],
            epsilon_minus_one: array![
                Complex64::new(0.10, 0.20),
                Complex64::new(0.15, 0.25),
                Complex64::new(0.20, 0.35),
            ],
            refractive_index_minus_one: array![
                Complex64::new(0.01, 0.02),
                Complex64::new(0.02, 0.03),
                Complex64::new(0.03, 0.04),
            ],
            absorption_coefficient: array![1000.0, 2000.0, 3000.0],
            reflectivity: array![0.0, 0.0, 0.0],
            loss: array![0.0, 0.0, 0.0],
        };

        let rendered = sumrules_dat_string(&sumrules_dat_from_opcons(0.075, &opcons)?)?;
        assert_eq!(
            rendered,
            &SUMRULES_DAT["# FULLSPECTRUM sumrules.dat\n".len()..]
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_sumrules_dat_inputs() {
        assert!(parse_sumrules_dat("").is_err());
        assert!(parse_sumrules_dat("# only header\n").is_err());
        assert!(parse_sumrules_dat("1 2 3 4 5 6\n").is_err());
        assert!(parse_sumrules_dat("1 2 3 4 5 6 7 8\n").is_err());
        assert!(parse_sumrules_dat("1 2 3 4 NaN 6 7\n").is_err());

        let bad = SumRulesDatData {
            header_lines: Vec::new(),
            energy_ev: array![1.0, 2.0],
            epsilon2_effective_electrons: array![1.0],
            absorption_effective_electrons: array![1.0, 2.0],
            loss_effective_electrons: array![1.0, 2.0],
            absorption_refractive_sum: array![1.0, 2.0],
            refractive_index_sum_ratio: array![1.0, 2.0],
            log_loss_moment_ratio: array![1.0, 2.0],
        };
        assert!(sumrules_dat_string(&bad).is_err());
    }

    const SUMRULES_DAT: &str = "# FULLSPECTRUM sumrules.dat\n        0.1000000000E+02        0.9122363013E-02        0.1800088998E+02        0.7297890411E-02        0.1800088998E+00        0.1000000000E+01       -0.2002101526E+01\n        0.2000000000E+02        0.4105063356E-01        0.5400266993E+02        0.3106214005E-01        0.9000444989E+00        0.1000000000E+01       -0.8687980690E+00\n        0.4000000000E+02        0.2143755308E+00        0.1620080098E+03        0.1457312311E+00        0.4140204695E+01        0.1000000000E+01       -0.3869050570E-01\n";
}
