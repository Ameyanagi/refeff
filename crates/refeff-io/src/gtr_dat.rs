//! FEFF `gtr.dat` FMS trace diagnostic support.
//!
//! The MKGTR stage writes `gtr.dat` as a four-column formatted table: the
//! complex energy grid followed by the complex trace of the multiple-scattering
//! Green's function. Keeping the table typed lets the Rust port compare
//! generated FMS diagnostics without ad-hoc parsing.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;

use crate::error::{IoError, Result};

const GTR_DAT_PATH: &str = "gtr.dat";

/// Parsed contents of FEFF `gtr.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GtrDatData {
    /// Complex energy grid written by MKGTR.
    pub energy: Array1<Complex64>,
    /// Complex Green's-function trace for each energy point.
    pub trace: Array1<Complex64>,
}

impl GtrDatData {
    /// Number of trace rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy.len()
    }
}

/// Parse FEFF `gtr.dat` text.
pub fn parse_gtr_dat(text: &str) -> Result<GtrDatData> {
    let mut energy = Vec::new();
    let mut trace = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let row = parse_numeric_row(line_number, line)?;
        energy.push(Complex64::new(row[0], row[1]));
        trace.push(Complex64::new(row[2], row[3]));
    }

    let data = GtrDatData {
        energy: Array1::from_vec(energy),
        trace: Array1::from_vec(trace),
    };
    validate_gtr_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `gtr.dat` text.
pub fn gtr_dat_string(data: &GtrDatData) -> Result<String> {
    validate_gtr_dat(data)?;
    let mut out = String::new();
    for (energy, trace) in data.energy.iter().zip(data.trace.iter()) {
        writeln!(
            out,
            "{:13.6}{:13.6}{:13.6}{:13.6}",
            energy.re, energy.im, trace.re, trace.im
        )?;
    }
    Ok(out)
}

/// Read FEFF `gtr.dat` text from a file.
pub fn read_gtr_dat(path: impl AsRef<Path>) -> Result<GtrDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_gtr_dat(&text)
}

/// Write FEFF `gtr.dat` text to a file.
pub fn write_gtr_dat(path: impl AsRef<Path>, data: &GtrDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, gtr_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_numeric_row(line_number: usize, line: &str) -> Result<[f64; 4]> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 4 {
        return parse_error(
            line_number,
            format!("row has {} token(s), expected 4", tokens.len()),
        );
    }

    Ok([
        parse_f64(line_number, "energy_real", tokens[0])?,
        parse_f64(line_number, "energy_imag", tokens[1])?,
        parse_f64(line_number, "trace_real", tokens[2])?,
        parse_f64(line_number, "trace_imag", tokens[3])?,
    ])
}

fn validate_gtr_dat(data: &GtrDatData) -> Result<()> {
    if data.row_count() == 0 {
        return parse_error(0, "at least one gtr row is required");
    }
    if data.trace.len() != data.row_count() {
        return parse_error(
            0,
            format!(
                "trace count {} does not match energy count {}",
                data.trace.len(),
                data.row_count()
            ),
        );
    }
    for (index, (energy, trace)) in data.energy.iter().zip(data.trace.iter()).enumerate() {
        let row = index + 1;
        validate_complex("energy", *energy, row)?;
        validate_complex("trace", *trace, row)?;
    }
    Ok(())
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

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: GTR_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gtr_dat() -> Result<()> {
        let parsed = parse_gtr_dat(GTR_DAT)?;
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.energy[0], Complex64::new(-0.138_801, 0.031_773));
        assert_eq!(parsed.trace[0], Complex64::new(0.0, 0.0));
        assert_eq!(parsed.energy[2], Complex64::new(55.866_911, 0.031_773));
        assert_eq!(parsed.trace[2], Complex64::new(1.624_106, 1.081_113));

        let rendered = gtr_dat_string(&parsed)?;
        assert_eq!(rendered, GTR_DAT);
        assert_eq!(parse_gtr_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_gtr_dat(" -1.38801D-01 3.1773D-02 0.0D+00 1.0D+00\n")?;
        assert_eq!(parsed.row_count(), 1);
        assert_eq!(parsed.energy[0], Complex64::new(-0.138_801, 0.031_773));
        assert_eq!(parsed.trace[0], Complex64::new(0.0, 1.0));
        Ok(())
    }

    #[test]
    fn rejects_bad_gtr_dat_inputs() {
        assert!(parse_gtr_dat("").is_err());
        assert!(parse_gtr_dat("1 2 3\n").is_err());
        assert!(parse_gtr_dat("1 2 3 4 5\n").is_err());
        assert!(parse_gtr_dat("1 2 NaN 4\n").is_err());
        assert!(parse_gtr_dat("1 bad 3 4\n").is_err());
    }

    const GTR_DAT: &str = r#"    -0.138801     0.031773     0.000000     0.000000
    -0.137401     0.031773     0.000000     0.000000
    55.866911     0.031773     1.624106     1.081113
"#;
}
