//! FEFF `exc.dat` excitation-pole table codec.
//!
//! The SELF many-pole path writes `exc.dat` through FEFF's generic
//! `WriteData` helper with three required double columns and one auxiliary
//! weight column. SFCONV's `rdeps` reader consumes the first three columns as
//! pole energy, pole broadening, and oscillator strength in eV, and can also
//! create a three-column fallback file when no `exc.dat` exists.

use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::Path;

use ndarray::Array1;
use refeff_core::FEFF_HARTREE_EV;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const EXC_DAT_PATH: &str = "exc.dat";
const EXC_DAT_REQUIRED_COLUMNS: usize = 3;
const EXC_DAT_AUXILIARY_COLUMNS: usize = 4;
const SFCONV_RDEPS_FALLBACK_BROADENING_FRACTION: f64 = 0.001;

/// Parsed FEFF `exc.dat` excitation-pole table.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcDatData {
    /// Header and comment lines before and around the numeric pole table.
    pub header_lines: Vec<String>,
    /// Pole energy in eV.
    pub energy_ev: Array1<f64>,
    /// Pole broadening in eV.
    pub broadening_ev: Array1<f64>,
    /// Oscillator strength for each pole.
    pub oscillator_strength: Array1<f64>,
    /// Optional fourth `WriteData` column from SELF's many-pole generator.
    pub auxiliary_weight: Option<Array1<f64>>,
}

/// FEFF `SFCONV/rdeps.f90` pole table after conversion from eV to Hartree.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvRdepsPoleTable {
    /// Pole energy in Hartree, FEFF `plengy`.
    pub energy_hartree: Array1<f64>,
    /// Pole broadening in Hartree, FEFF `plbrd`.
    pub broadening_hartree: Array1<f64>,
    /// Oscillator strength, FEFF `oscstr`.
    pub oscillator_strength: Array1<f64>,
}

impl SfconvRdepsPoleTable {
    /// Number of excitation poles read by FEFF `rdeps`.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

impl ExcDatData {
    /// Number of excitation poles.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether this table carries SELF's fourth auxiliary column.
    #[must_use]
    pub fn has_auxiliary_weight(&self) -> bool {
        self.auxiliary_weight.is_some()
    }
}

/// Port of FEFF `SFCONV/rdeps.f90` for an already parsed `exc.dat`.
///
/// The on-disk `exc.dat` energies and broadenings are in eV. FEFF converts the
/// first two columns to Hartree and keeps the oscillator strength unchanged.
/// `max_poles` corresponds to FEFF `nplmax`; Rust reports an error instead of
/// overflowing the caller's arrays.
pub fn sfconv_rdeps_from_exc_dat(
    data: &ExcDatData,
    max_poles: usize,
) -> Result<SfconvRdepsPoleTable> {
    validate_exc_dat(data)?;
    validate_rdeps_max_poles(max_poles)?;
    if data.pole_count() > max_poles {
        return invalid_exc_dat(
            "rows",
            format!(
                "got {} excitation pole(s), maximum is {max_poles}",
                data.pole_count()
            ),
        );
    }

    Ok(SfconvRdepsPoleTable {
        energy_hartree: data.energy_ev.mapv(|energy| energy / FEFF_HARTREE_EV),
        broadening_hartree: data
            .broadening_ev
            .mapv(|broadening| broadening / FEFF_HARTREE_EV),
        oscillator_strength: data.oscillator_strength.clone(),
    })
}

/// FEFF `SFCONV/rdeps.f90` fallback table for a missing `exc.dat`.
///
/// FEFF uses one pole at the plasma frequency, broadens it by `0.001 * omp`,
/// and gives it unit oscillator strength.
pub fn sfconv_rdeps_fallback_poles(
    plasma_frequency_hartree: f64,
    max_poles: usize,
) -> Result<SfconvRdepsPoleTable> {
    validate_rdeps_plasma_frequency(plasma_frequency_hartree)?;
    validate_rdeps_max_poles(max_poles)?;

    Ok(SfconvRdepsPoleTable {
        energy_hartree: Array1::from_vec(vec![plasma_frequency_hartree]),
        broadening_hartree: Array1::from_vec(vec![
            SFCONV_RDEPS_FALLBACK_BROADENING_FRACTION * plasma_frequency_hartree,
        ]),
        oscillator_strength: Array1::from_vec(vec![1.0]),
    })
}

/// Build the FEFF `SFCONV/rdeps.f90` missing-file fallback as `ExcDatData`.
pub fn sfconv_rdeps_fallback_exc_dat(plasma_frequency_hartree: f64) -> Result<ExcDatData> {
    validate_rdeps_plasma_frequency(plasma_frequency_hartree)?;
    Ok(ExcDatData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_vec(vec![plasma_frequency_hartree * FEFF_HARTREE_EV]),
        broadening_ev: Array1::from_vec(vec![
            SFCONV_RDEPS_FALLBACK_BROADENING_FRACTION * plasma_frequency_hartree * FEFF_HARTREE_EV,
        ]),
        oscillator_strength: Array1::from_vec(vec![1.0]),
        auxiliary_weight: None,
    })
}

/// Render the exact fixed-width fallback row written by FEFF `rdeps`.
pub fn sfconv_rdeps_fallback_exc_dat_string(plasma_frequency_hartree: f64) -> Result<String> {
    let data = sfconv_rdeps_fallback_exc_dat(plasma_frequency_hartree)?;
    let mut out = String::new();
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        data.energy_ev[0], data.broadening_ev[0], data.oscillator_strength[0]
    )?;
    Ok(out)
}

/// Read FEFF `exc.dat` like `SFCONV/rdeps.f90`, creating the fallback if absent.
///
/// When `path` is missing, this writes FEFF's fixed-width fallback row and
/// returns the exact in-memory Hartree values that FEFF uses in the same call.
pub fn read_or_create_sfconv_rdeps(
    path: impl AsRef<Path>,
    plasma_frequency_hartree: f64,
    max_poles: usize,
) -> Result<SfconvRdepsPoleTable> {
    let path = path.as_ref();
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let data = parse_exc_dat(&text)?;
            sfconv_rdeps_from_exc_dat(&data, max_poles)
        }
        Err(source) if source.kind() == ErrorKind::NotFound => {
            let text = sfconv_rdeps_fallback_exc_dat_string(plasma_frequency_hartree)?;
            std::fs::write(path, text).map_err(|source| IoError::io(path, source))?;
            sfconv_rdeps_fallback_poles(plasma_frequency_hartree, max_poles)
        }
        Err(source) => Err(IoError::io(path, source)),
    }
}

/// Render FEFF-compatible `exc.dat` text.
pub fn exc_dat_string(data: &ExcDatData) -> Result<String> {
    validate_exc_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        for (((energy, broadening), strength), auxiliary) in data
            .energy_ev
            .iter()
            .zip(data.broadening_ev.iter())
            .zip(data.oscillator_strength.iter())
            .zip(auxiliary_weight.iter())
        {
            write_exc_row(&mut out, [*energy, *broadening, *strength, *auxiliary])?;
        }
    } else {
        for ((energy, broadening), strength) in data
            .energy_ev
            .iter()
            .zip(data.broadening_ev.iter())
            .zip(data.oscillator_strength.iter())
        {
            write_exc_row(&mut out, [*energy, *broadening, *strength])?;
        }
    }
    Ok(out)
}

fn write_exc_row<const N: usize>(out: &mut String, fields: [f64; N]) -> Result<()> {
    for value in fields {
        write_fortran_zero_scaled_exp(out, value, 20, 10)?;
        out.push(' ');
    }
    out.push('\n');
    Ok(())
}

/// Parse FEFF `exc.dat` text.
pub fn parse_exc_dat(text: &str) -> Result<ExcDatData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut broadening_ev = Vec::new();
    let mut oscillator_strength = Vec::new();
    let mut auxiliary_weight = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(width, EXC_DAT_REQUIRED_COLUMNS | EXC_DAT_AUXILIARY_COLUMNS) {
                return parse_error(
                    line_number,
                    format!("exc.dat row has {width} token(s), expected 3 or 4 numeric columns"),
                );
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return parse_error(
                        line_number,
                        format!(
                            "exc.dat row has {width} token(s), expected {expected} to match previous rows"
                        ),
                    );
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            broadening_ev.push(parse_f64(line_number, "broadening", tokens[1])?);
            oscillator_strength.push(parse_f64(line_number, "oscillator strength", tokens[2])?);
            if width == EXC_DAT_AUXILIARY_COLUMNS {
                auxiliary_weight.push(parse_f64(line_number, "auxiliary weight", tokens[3])?);
            }
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let auxiliary_weight = if row_width == Some(EXC_DAT_AUXILIARY_COLUMNS) {
        Some(Array1::from_vec(auxiliary_weight))
    } else {
        None
    };
    let data = ExcDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        broadening_ev: Array1::from_vec(broadening_ev),
        oscillator_strength: Array1::from_vec(oscillator_strength),
        auxiliary_weight,
    };
    validate_exc_dat(&data)?;
    Ok(data)
}

/// Write FEFF `exc.dat` text to a file.
pub fn write_exc_dat(path: impl AsRef<Path>, data: &ExcDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, exc_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `exc.dat` text from a file.
pub fn read_exc_dat(path: impl AsRef<Path>) -> Result<ExcDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_exc_dat(&text)
}

fn validate_exc_dat(data: &ExcDatData) -> Result<()> {
    let pole_count = data.pole_count();
    if pole_count == 0 {
        return invalid_exc_dat("rows", "at least one excitation-pole row is required");
    }
    validate_len("broadening_ev", data.broadening_ev.len(), pole_count)?;
    validate_len(
        "oscillator_strength",
        data.oscillator_strength.len(),
        pole_count,
    )?;
    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        validate_len("auxiliary_weight", auxiliary_weight.len(), pole_count)?;
    }

    for (row, ((energy, broadening), strength)) in data
        .energy_ev
        .iter()
        .zip(data.broadening_ev.iter())
        .zip(data.oscillator_strength.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite("energy", *energy, row)?;
        validate_finite("broadening", *broadening, row)?;
        validate_finite("oscillator strength", *strength, row)?;
    }
    if let Some(auxiliary_weight) = &data.auxiliary_weight {
        for (row, value) in auxiliary_weight.iter().enumerate() {
            validate_finite("auxiliary weight", *value, row + 1)?;
        }
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_exc_dat(field, format!("got {actual} value(s), expected {expected}"))
    }
}

fn validate_rdeps_max_poles(max_poles: usize) -> Result<()> {
    if max_poles > 0 {
        Ok(())
    } else {
        invalid_exc_dat("nplmax", "maximum pole count must be positive")
    }
}

fn validate_rdeps_plasma_frequency(plasma_frequency_hartree: f64) -> Result<()> {
    if plasma_frequency_hartree.is_finite() && plasma_frequency_hartree > 0.0 {
        Ok(())
    } else {
        invalid_exc_dat(
            "plasma_frequency_hartree",
            format!("value must be positive and finite, got {plasma_frequency_hartree}"),
        )
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
        invalid_exc_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_exc_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: EXC_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: EXC_DAT_PATH.into(),
        line,
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
    fn parses_self_write_data_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(EXC_DAT)?;

        assert_eq!(parsed.header_lines.len(), 5);
        assert_eq!(parsed.pole_count(), 2);
        assert!(parsed.has_auxiliary_weight());
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.broadening_ev[1], 0.2);
        assert_eq!(parsed.oscillator_strength[0], 0.25);
        assert_eq!(
            parsed.auxiliary_weight.as_ref().map(|values| values[1]),
            Some(2.5)
        );
        Ok(())
    }

    #[test]
    fn parses_three_column_rdeps_fallback_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(RDEPS_FALLBACK_EXC_DAT)?;

        assert_eq!(parsed.pole_count(), 1);
        assert!(!parsed.has_auxiliary_weight());
        assert_eq!(parsed.energy_ev[0], 10.0);
        assert_eq!(parsed.broadening_ev[0], 0.01);
        assert_eq!(parsed.oscillator_strength[0], 1.0);
        Ok(())
    }

    #[test]
    fn roundtrips_exc_dat() -> Result<()> {
        let parsed = parse_exc_dat(EXC_DAT)?;
        let rendered = exc_dat_string(&parsed)?;

        assert_eq!(rendered, EXC_DAT);
        assert_eq!(parse_exc_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_exc_dat_inputs() {
        assert!(parse_exc_dat("# only a header\n").is_err());
        assert!(parse_exc_dat("1 2\n").is_err());
        assert!(parse_exc_dat("1 2 3 4 5\n").is_err());
        assert!(parse_exc_dat("1 2 3\n4 5 6 7\n").is_err());
        assert!(parse_exc_dat("1 NaN 3\n").is_err());

        let bad = ExcDatData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            broadening_ev: Array1::from_vec(vec![0.1]),
            oscillator_strength: Array1::from_vec(vec![1.0, 1.0]),
            auxiliary_weight: None,
        };
        assert!(exc_dat_string(&bad).is_err());
    }

    #[test]
    fn sfconv_rdeps_existing_exc_dat_matches_feff_reference() -> Result<()> {
        let data = parse_exc_dat(RDEPS_EXISTING_EXC_DAT)?;
        let poles = sfconv_rdeps_from_exc_dat(&data, 5)?;

        assert_eq!(poles.pole_count(), 2);
        assert_close(poles.energy_hartree[0], 0.5);
        assert_close(poles.broadening_hartree[0], 0.001);
        assert_close(poles.oscillator_strength[0], 0.25);
        assert_close(poles.energy_hartree[1], 1.0);
        assert_close(poles.broadening_hartree[1], 0.002);
        assert_close(poles.oscillator_strength[1], 0.75);
        Ok(())
    }

    #[test]
    fn sfconv_rdeps_fallback_matches_feff_reference() -> Result<()> {
        let poles = sfconv_rdeps_fallback_poles(0.47, 5)?;
        let text = sfconv_rdeps_fallback_exc_dat_string(0.47)?;

        assert_eq!(poles.pole_count(), 1);
        assert_close(poles.energy_hartree[0], 0.47);
        assert_close(poles.broadening_hartree[0], 0.000_47);
        assert_close(poles.oscillator_strength[0], 1.0);
        assert_eq!(text, "     12.78936      0.01279      1.00000\n");
        Ok(())
    }

    #[test]
    fn read_or_create_sfconv_rdeps_creates_feff_fallback_when_missing() -> Result<()> {
        let temp = tempfile::tempdir().map_err(|source| IoError::io("tempdir", source))?;
        let path = temp.path().join("exc.dat");

        let poles = read_or_create_sfconv_rdeps(&path, 0.47, 5)?;

        assert_close(poles.energy_hartree[0], 0.47);
        assert_eq!(
            std::fs::read_to_string(path).map_err(|source| IoError::io("exc.dat", source))?,
            "     12.78936      0.01279      1.00000\n"
        );
        Ok(())
    }

    #[test]
    fn sfconv_rdeps_rejects_invalid_inputs() -> Result<()> {
        let data = parse_exc_dat(RDEPS_EXISTING_EXC_DAT)?;

        assert!(sfconv_rdeps_from_exc_dat(&data, 1).is_err());
        assert!(sfconv_rdeps_from_exc_dat(&data, 0).is_err());
        assert!(sfconv_rdeps_fallback_poles(0.0, 5).is_err());
        assert!(sfconv_rdeps_fallback_exc_dat_string(f64::NAN).is_err());
        Ok(())
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual} expected={expected}"
        );
    }

    const EXC_DAT: &str = r#"#SN#   Section:    1
#DF# This section written in TXT.
#H#
#H# The following data types are written in this section.
#DT#  Double Double Double Double
    0.1000000000E+02     0.1000000000E+00     0.2500000000E+00     0.1250000000E+01 
    0.2000000000E+02     0.2000000000E+00     0.5000000000E+00     0.2500000000E+01 
"#;

    const RDEPS_FALLBACK_EXC_DAT: &str = "      10.00000      0.01000      1.00000\n";
    const RDEPS_EXISTING_EXC_DAT: &str = concat!(
        "# comment row\n",
        "  13.605698D0  0.027211396D0  0.25D0\n",
        "  27.211396D0  0.054422792D0  0.75D0\n",
    );
}
