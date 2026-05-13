//! FEFF `gtrl.dat` NRIXS/LDEC FMS trace-decomposition support.
//!
//! The NRIXS variant of MKGTR writes `gtrl.dat` as a readable companion to
//! `fmsl.bin`. Each row contains the one-based energy index, the real energy,
//! then the real and imaginary parts of the decomposed FMS trace components.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};

const GTRL_DAT_PATH: &str = "gtrl.dat";
const FINAL_COMPONENTS_PER_CHANNEL: usize = 3;

/// Parsed contents of FEFF `gtrl.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GtrlDatData {
    /// One-based FEFF energy row index.
    pub energy_index: Array1<usize>,
    /// Real energy grid written by MKGTR.
    pub energy: Array1<f64>,
    /// Complex decomposed trace components, shaped `(energy, component)`.
    pub decomposed_trace: Array2<Complex64>,
}

impl GtrlDatData {
    /// Number of energy rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy_index.len()
    }

    /// Number of complex decomposition components per energy row.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.decomposed_trace.len_of(Axis(1))
    }

    /// Number of initial decomposition channels represented per row.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.component_count() / FINAL_COMPONENTS_PER_CHANNEL
    }
}

/// Parse FEFF `gtrl.dat` text.
pub fn parse_gtrl_dat(text: &str) -> Result<GtrlDatData> {
    let mut energy_index = Vec::new();
    let mut energy = Vec::new();
    let mut decomposed_trace = Vec::new();
    let mut component_count = None;

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let row = parse_gtrl_row(line_number, line)?;
        match component_count {
            Some(expected) if row.components.len() != expected => {
                return parse_error(
                    line_number,
                    format!(
                        "row has {} component(s), expected {expected}",
                        row.components.len()
                    ),
                );
            }
            Some(_) => {}
            None => component_count = Some(row.components.len()),
        }
        energy_index.push(row.energy_index);
        energy.push(row.energy);
        decomposed_trace.extend(row.components);
    }

    let row_count = energy_index.len();
    let component_count = component_count
        .ok_or_else(|| parse_error_value(0, "at least one gtrl decomposition row is required"))?;
    let decomposed_trace = Array2::from_shape_vec((row_count, component_count), decomposed_trace)
        .map_err(|source| {
        parse_error_value(0, format!("invalid decomposed trace shape: {source}"))
    })?;
    let data = GtrlDatData {
        energy_index: Array1::from_vec(energy_index),
        energy: Array1::from_vec(energy),
        decomposed_trace,
    };
    validate_gtrl_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `gtrl.dat` text.
pub fn gtrl_dat_string(data: &GtrlDatData) -> Result<String> {
    validate_gtrl_dat(data)?;
    let mut out = String::new();
    for row in 0..data.row_count() {
        write!(out, "{:5}", data.energy_index[row])?;
        write!(out, "{:18.8E}", data.energy[row])?;
        for value in data.decomposed_trace.row(row) {
            write!(out, "{:18.8E}", value.re)?;
        }
        for value in data.decomposed_trace.row(row) {
            write!(out, "{:18.8E}", value.im)?;
        }
        writeln!(out)?;
    }
    Ok(out)
}

/// Read FEFF `gtrl.dat` text from a file.
pub fn read_gtrl_dat(path: impl AsRef<Path>) -> Result<GtrlDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_gtrl_dat(&text)
}

/// Write FEFF `gtrl.dat` text to a file.
pub fn write_gtrl_dat(path: impl AsRef<Path>, data: &GtrlDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, gtrl_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

struct GtrlRow {
    energy_index: usize,
    energy: f64,
    components: Vec<Complex64>,
}

fn parse_gtrl_row(line_number: usize, line: &str) -> Result<GtrlRow> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 4 {
        return parse_error(
            line_number,
            format!("row has {} token(s), expected at least 4", tokens.len()),
        );
    }
    let component_tokens = tokens.len() - 2;
    if component_tokens % 2 != 0 {
        return parse_error(
            line_number,
            format!("row has {component_tokens} decomposition token(s), expected an even count"),
        );
    }
    let component_count = component_tokens / 2;
    if component_count == 0 || !component_count.is_multiple_of(FINAL_COMPONENTS_PER_CHANNEL) {
        return parse_error(
            line_number,
            format!(
                "row has {component_count} complex component(s), expected a positive multiple of {FINAL_COMPONENTS_PER_CHANNEL}"
            ),
        );
    }

    let energy_index = parse_usize(line_number, "energy_index", tokens[0])?;
    let energy = parse_f64(line_number, "energy", tokens[1])?;
    let real_values = tokens[2..2 + component_count]
        .iter()
        .map(|token| parse_f64(line_number, "component_real", token))
        .collect::<Result<Vec<_>>>()?;
    let imag_values = tokens[2 + component_count..]
        .iter()
        .map(|token| parse_f64(line_number, "component_imag", token))
        .collect::<Result<Vec<_>>>()?;
    let components = real_values
        .into_iter()
        .zip(imag_values)
        .map(|(real, imag)| Complex64::new(real, imag))
        .collect();

    Ok(GtrlRow {
        energy_index,
        energy,
        components,
    })
}

fn validate_gtrl_dat(data: &GtrlDatData) -> Result<()> {
    if data.row_count() == 0 {
        return parse_error(0, "at least one gtrl decomposition row is required");
    }
    validate_len("energy", data.energy.len(), data.row_count())?;
    if data.decomposed_trace.len_of(Axis(0)) != data.row_count() {
        return parse_error(
            0,
            format!(
                "decomposed trace row count {} does not match energy count {}",
                data.decomposed_trace.len_of(Axis(0)),
                data.row_count()
            ),
        );
    }
    if data.component_count() == 0
        || !data
            .component_count()
            .is_multiple_of(FINAL_COMPONENTS_PER_CHANNEL)
    {
        return parse_error(
            0,
            format!(
                "component count {} must be a positive multiple of {FINAL_COMPONENTS_PER_CHANNEL}",
                data.component_count()
            ),
        );
    }
    for (index, (energy_index, energy)) in
        data.energy_index.iter().zip(data.energy.iter()).enumerate()
    {
        let row = index + 1;
        if *energy_index == 0 {
            return parse_error(row, "energy index must be positive");
        }
        validate_finite("energy", *energy, row)?;
        for value in data.decomposed_trace.row(index) {
            validate_complex("decomposed_trace", *value, row)?;
        }
    }
    Ok(())
}

fn validate_len(field: &'static str, len: usize, expected: usize) -> Result<()> {
    if len == expected {
        Ok(())
    } else {
        parse_error(0, format!("{field} has {len} row(s), expected {expected}"))
    }
}

fn validate_complex(field: &'static str, value: Complex64, row: usize) -> Result<()> {
    validate_finite(field, value.re, row)?;
    validate_finite(field, value.im, row)
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(row, format!("{field} must be finite"))
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: GTRL_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gtrl_dat() -> Result<()> {
        let parsed = parse_gtrl_dat(GTRL_DAT)?;
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.component_count(), 9);
        assert_eq!(parsed.channel_count(), 3);
        assert_eq!(parsed.energy_index[0], 1);
        assert_eq!(parsed.energy[0], -0.433_093_63);
        assert_eq!(
            parsed.decomposed_trace[(0, 0)],
            Complex64::new(0.875_934_54, -0.382_255_02)
        );
        assert_eq!(
            parsed.decomposed_trace[(0, 4)],
            Complex64::new(-2.203_646_7, 1.919_603_5)
        );
        assert_eq!(
            parsed.decomposed_trace[(1, 8)],
            Complex64::new(-0.003_525_367_7, 0.024_426_693)
        );

        let rendered = gtrl_dat_string(&parsed)?;
        assert_eq!(parse_gtrl_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_gtrl_dat("1 -1.0D+00 1D+00 2D+00 3D+00 4D+00 5D+00 6D+00\n")?;
        assert_eq!(parsed.row_count(), 1);
        assert_eq!(parsed.component_count(), 3);
        assert_eq!(parsed.decomposed_trace[(0, 2)], Complex64::new(3.0, 6.0));
        Ok(())
    }

    #[test]
    fn rejects_bad_gtrl_dat_inputs() {
        assert!(parse_gtrl_dat("").is_err());
        assert!(parse_gtrl_dat("1 2 3\n").is_err());
        assert!(parse_gtrl_dat("1 2 3 4 5\n").is_err());
        assert!(parse_gtrl_dat("1 2 3 4\n").is_err());
        assert!(parse_gtrl_dat(&GTRL_DAT.replace("   2", "   0")).is_err());
        assert!(parse_gtrl_dat(&GTRL_DAT.replace("0.87593454E+00", "NaN")).is_err());
        assert!(parse_gtrl_dat(&format!("{GTRL_DAT}   3 1 1 2 3 4 5 6\n")).is_err());
    }

    const GTRL_DAT: &str = r#"    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01
    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01
"#;
}
