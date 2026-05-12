//! FEFF `compton.dat` profile text codec.
//!
//! The COMPTON module writes a comment header describing the integration grid
//! followed by two numeric columns: projected momentum `pq` and the Compton
//! profile `J(pq)`.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};

const COMPTON_DAT_ROW_WIDTH: usize = 2;
const RHOZZP_DAT_ROW_WIDTH: usize = 2;

#[derive(Default)]
struct ComptonDatHeader {
    ns: Option<usize>,
    nphi: Option<usize>,
    nz: Option<usize>,
    nzp: Option<usize>,
    zpmax: Option<f64>,
    temperature_ev: Option<f64>,
}

/// Parsed FEFF `compton.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ComptonDatData {
    /// Header and comment lines before the numeric Compton profile table.
    pub header_lines: Vec<String>,
    /// Number of radial integration points in cylindrical radius.
    pub ns: Option<usize>,
    /// Number of azimuthal integration points.
    pub nphi: Option<usize>,
    /// Number of `z` integration points.
    pub nz: Option<usize>,
    /// Number of `z'` integration points.
    pub nzp: Option<usize>,
    /// Maximum `z'` integration coordinate from the header.
    pub zpmax: Option<f64>,
    /// Electronic temperature in eV from the header.
    pub temperature_ev: Option<f64>,
    /// Projected momentum grid `pq`.
    pub momentum: Array1<f64>,
    /// Compton profile values `J(pq)`.
    pub profile: Array1<f64>,
}

impl ComptonDatData {
    /// Number of profile rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.momentum.len()
    }
}

/// Parsed FEFF `rhozzp.dat` diagnostic density slice.
#[derive(Debug, Clone, PartialEq)]
pub struct RhozzpDatData {
    /// Optional comment lines before or around the numeric diagnostic table.
    pub header_lines: Vec<String>,
    /// `z'` coordinate grid.
    pub z_prime: Array1<f64>,
    /// Density matrix slice values `rho(z,z')`.
    pub density: Array1<f64>,
}

impl RhozzpDatData {
    /// Number of diagnostic rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.z_prime.len()
    }
}

/// Render FEFF-compatible `compton.dat` text.
pub fn compton_dat_string(data: &ComptonDatData) -> Result<String> {
    validate_compton_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (momentum, profile) in data.momentum.iter().zip(data.profile.iter()) {
        writeln!(out, "{momentum:24.17E} {profile:24.17E}")?;
    }
    Ok(out)
}

/// Parse FEFF `compton.dat` text.
pub fn parse_compton_dat(text: &str) -> Result<ComptonDatData> {
    let mut header_lines = Vec::new();
    let mut header = ComptonDatHeader::default();
    let mut momentum = Vec::new();
    let mut profile = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != COMPTON_DAT_ROW_WIDTH {
                return Err(IoError::ComptonDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: COMPTON_DAT_ROW_WIDTH,
                });
            }
            momentum.push(parse_f64(line_number, "momentum", tokens[0])?);
            profile.push(parse_f64(line_number, "profile", tokens[1])?);
        } else {
            parse_header_metadata(line, line_number, &mut header)?;
            header_lines.push(line.to_string());
        }
    }

    let data = ComptonDatData {
        header_lines,
        ns: header.ns,
        nphi: header.nphi,
        nz: header.nz,
        nzp: header.nzp,
        zpmax: header.zpmax,
        temperature_ev: header.temperature_ev,
        momentum: Array1::from_vec(momentum),
        profile: Array1::from_vec(profile),
    };
    validate_compton_dat(&data)?;
    Ok(data)
}

/// Write FEFF `compton.dat` text to a file.
pub fn write_compton_dat(path: impl AsRef<Path>, data: &ComptonDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, compton_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `compton.dat` text from a file.
pub fn read_compton_dat(path: impl AsRef<Path>) -> Result<ComptonDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_compton_dat(&text)
}

/// Render FEFF-compatible `rhozzp.dat` text.
pub fn rhozzp_dat_string(data: &RhozzpDatData) -> Result<String> {
    validate_rhozzp_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (z_prime, density) in data.z_prime.iter().zip(data.density.iter()) {
        writeln!(out, "{z_prime:24.17E} {density:24.17E}")?;
    }
    Ok(out)
}

/// Parse FEFF `rhozzp.dat` diagnostic text.
pub fn parse_rhozzp_dat(text: &str) -> Result<RhozzpDatData> {
    let mut header_lines = Vec::new();
    let mut z_prime = Vec::new();
    let mut density = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != RHOZZP_DAT_ROW_WIDTH {
                return Err(IoError::RhozzpDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: RHOZZP_DAT_ROW_WIDTH,
                });
            }
            z_prime.push(parse_rhozzp_f64(line_number, "z prime", tokens[0])?);
            density.push(parse_rhozzp_f64(line_number, "density", tokens[1])?);
        } else {
            header_lines.push(line.to_string());
        }
    }

    let data = RhozzpDatData {
        header_lines,
        z_prime: Array1::from_vec(z_prime),
        density: Array1::from_vec(density),
    };
    validate_rhozzp_dat(&data)?;
    Ok(data)
}

/// Write FEFF `rhozzp.dat` text to a file.
pub fn write_rhozzp_dat(path: impl AsRef<Path>, data: &RhozzpDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhozzp_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `rhozzp.dat` text from a file.
pub fn read_rhozzp_dat(path: impl AsRef<Path>) -> Result<RhozzpDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rhozzp_dat(&text)
}

fn parse_header_metadata(
    line: &str,
    line_number: usize,
    header: &mut ComptonDatHeader,
) -> Result<()> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("nphi:") {
        header.nphi = Some(parse_usize_header(line_number, "nphi", line)?);
    } else if lower.contains("nzp:") {
        header.nzp = Some(parse_usize_header(line_number, "nzp", line)?);
    } else if lower.contains("ns:") {
        header.ns = Some(parse_usize_header(line_number, "ns", line)?);
    } else if lower.contains("nz:") {
        header.nz = Some(parse_usize_header(line_number, "nz", line)?);
    } else if lower.contains("zpmax:") {
        header.zpmax = Some(parse_f64_header(line_number, "zpmax", line)?);
    } else if lower.contains("temperature") {
        header.temperature_ev = Some(parse_f64_header(line_number, "temperature", line)?);
    }
    Ok(())
}

fn validate_compton_dat(data: &ComptonDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_compton_dat(
            "rows",
            "at least one Compton profile row is required",
        ));
    }
    validate_len("profile", data.profile.len(), point_count)?;

    validate_positive_header("ns", data.ns)?;
    validate_positive_header("nphi", data.nphi)?;
    validate_positive_header("nz", data.nz)?;
    validate_positive_header("nzp", data.nzp)?;

    if let Some(value) = data.zpmax {
        validate_finite("zpmax", value)?;
    }
    if let Some(value) = data.temperature_ev {
        validate_finite("temperature", value)?;
    }

    for (row, (momentum, profile)) in data.momentum.iter().zip(data.profile.iter()).enumerate() {
        let row = row + 1;
        validate_finite_row("momentum", *momentum, row)?;
        validate_finite_row("profile", *profile, row)?;
    }

    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::ComptonDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn validate_rhozzp_dat(data: &RhozzpDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_rhozzp_dat(
            "rows",
            "at least one rhozzp diagnostic row is required",
        ));
    }
    validate_rhozzp_len("density", data.density.len(), point_count)?;

    for (row, (z_prime, density)) in data.z_prime.iter().zip(data.density.iter()).enumerate() {
        let row = row + 1;
        validate_rhozzp_finite_row("z prime", *z_prime, row)?;
        validate_rhozzp_finite_row("density", *density, row)?;
    }

    Ok(())
}

fn validate_rhozzp_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::RhozzpDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn validate_positive_header(field: &'static str, value: Option<usize>) -> Result<()> {
    if value.is_some_and(|value| value == 0) {
        Err(invalid_compton_dat(field, "value must be positive"))
    } else {
        Ok(())
    }
}

fn parse_usize_header(line: usize, field: &'static str, text: &str) -> Result<usize> {
    parse_usize(
        line,
        field,
        last_numeric_token(text)
            .ok_or_else(|| invalid_compton_dat(field, "missing numeric header value"))?,
    )
}

fn parse_f64_header(line: usize, field: &'static str, text: &str) -> Result<f64> {
    parse_f64(
        line,
        field,
        last_numeric_token(text)
            .ok_or_else(|| invalid_compton_dat(field, "missing numeric header value"))?,
    )
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::ComptonDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| IoError::ComptonDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn parse_rhozzp_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::RhozzpDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_compton_dat(field, "value must be finite"))
    }
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidComptonDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn validate_rhozzp_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidRhozzpDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_compton_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidComptonDat {
        field,
        message: message.into(),
    }
}

fn invalid_rhozzp_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidRhozzpDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

fn last_numeric_token(line: &str) -> Option<&str> {
    line.split_whitespace()
        .rev()
        .find(|token| is_numeric_token(token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_compton_reference_shape_and_metadata() -> Result<()> {
        let data = parse_compton_dat(COMPTON_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.ns, Some(32));
        assert_eq!(data.nphi, Some(32));
        assert_eq!(data.nz, Some(32));
        assert_eq!(data.nzp, Some(120));
        assert_eq!(data.zpmax, Some(10.0));
        assert_eq!(data.temperature_ev, Some(0.0));
        assert_eq!(data.momentum[1], 5.005004815757275e-3);
        assert_eq!(data.profile[2], 2.74462341659279);
        Ok(())
    }

    #[test]
    fn roundtrips_compton_text() -> Result<()> {
        let data = parse_compton_dat(COMPTON_DAT)?;
        let rendered = compton_dat_string(&data)?;
        assert_eq!(parse_compton_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_compton_inputs() {
        assert!(parse_compton_dat("# no data\n").is_err());
        assert!(parse_compton_dat("1 2 3\n").is_err());
        assert!(parse_compton_dat("1 NaN\n").is_err());
        assert!(parse_compton_dat("# ns: 0\n1 2\n").is_err());

        let bad_shape = ComptonDatData {
            header_lines: Vec::new(),
            ns: None,
            nphi: None,
            nz: None,
            nzp: None,
            zpmax: None,
            temperature_ev: None,
            momentum: Array1::from_vec(vec![1.0, 2.0]),
            profile: Array1::from_vec(vec![1.0]),
        };
        assert!(compton_dat_string(&bad_shape).is_err());
    }

    #[test]
    fn parses_feff_rhozzp_reference_shape() -> Result<()> {
        let data = parse_rhozzp_dat(RHOZZP_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.z_prime[0], 9.999999776482582e-3);
        assert_eq!(data.density[1], 2.66921255682004);
        assert_eq!(data.density[2], 1.84694165446344);
        Ok(())
    }

    #[test]
    fn roundtrips_rhozzp_text() -> Result<()> {
        let data = parse_rhozzp_dat(RHOZZP_DAT)?;
        let rendered = rhozzp_dat_string(&data)?;
        assert_eq!(parse_rhozzp_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_rhozzp_inputs() {
        assert!(parse_rhozzp_dat("# no data\n").is_err());
        assert!(parse_rhozzp_dat("1 2 3\n").is_err());
        assert!(parse_rhozzp_dat("1 NaN\n").is_err());

        let bad_shape = RhozzpDatData {
            header_lines: Vec::new(),
            z_prime: Array1::from_vec(vec![1.0, 2.0]),
            density: Array1::from_vec(vec![1.0]),
        };
        assert!(rhozzp_dat_string(&bad_shape).is_err());
    }

    const COMPTON_DAT: &str = r#" # Compton profile, J(pq)
 # ns:            32
 # nphi:          32
 # nz:            32
 # nzp:          120
 # zpmax:   10.0000000000000     
 # temperature (eV):  0.0000000E+00
 #----------------------------
 # pq               J
  0.000000000000000E+000   2.74476734850343     
  5.005004815757275E-003   2.74473136578831     
  1.001000963151455E-002   2.74462341659279     
"#;

    const RHOZZP_DAT: &str = r#"  9.999999776482582E-003   3.71096344005271
  2.001000978649259E-002   2.66921255682004
  3.002001979650260E-002   1.84694165446344
"#;
}
