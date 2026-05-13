//! FEFF `xmu.dat` spectrum text codec.
//!
//! FEFF writes `xmu.dat` as a comment-rich header followed by six numeric
//! columns: photon energy, edge-relative energy, photoelectron wave number,
//! normalized total absorption, normalized atomic background, and chi. The
//! full-spectrum reader in `FULLSPECTRUM/rdxmu.f90` uses the `xsedge+ 50`
//! header scalar to convert normalized `mu` and `mu0` to absolute cross
//! sections.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};
use crate::format::{write_fortran_exp, write_fortran_zero_scaled_exp};

const XMU_DAT_ROW_WIDTH: usize = 6;
const COMPACT_FIXED_PRECISION: i32 = 3;
const COLUMN_EQUALITY_TOLERANCE: f64 = 1.0e-12;

/// Parsed FEFF `xmu.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct XmuDatData {
    /// Header lines before and around the numeric data table.
    pub header_lines: Vec<String>,
    /// `xsedge+ 50` normalization scalar when the header provides it.
    pub normalization: Option<f64>,
    /// Photon energy in eV.
    pub photon_energy_ev: Array1<f64>,
    /// Photoelectron energy relative to the edge in eV.
    pub relative_energy_ev: Array1<f64>,
    /// Photoelectron wave number in inverse Angstrom.
    pub wave_number: Array1<f64>,
    /// Normalized total absorption coefficient.
    pub mu: Array1<f64>,
    /// Normalized atomic background absorption coefficient.
    pub mu0: Array1<f64>,
    /// Fine structure, `chi = mu - mu0` in normalized units.
    pub chi: Array1<f64>,
}

impl XmuDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.photon_energy_ev.len()
    }

    /// Total absorption converted to absolute units when normalization exists.
    #[must_use]
    pub fn absolute_mu(&self) -> Option<Array1<f64>> {
        self.normalization
            .map(|normalization| self.mu.mapv(|value| value * normalization))
    }

    /// Atomic background converted to absolute units when normalization exists.
    #[must_use]
    pub fn absolute_mu0(&self) -> Option<Array1<f64>> {
        self.normalization
            .map(|normalization| self.mu0.mapv(|value| value * normalization))
    }
}

/// Render FEFF-compatible `xmu.dat` text.
pub fn xmu_dat_string(data: &XmuDatData) -> Result<String> {
    validate_xmu_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    let format = xmu_render_format(data);
    for (((((omega, edge), k), mu), mu0), chi) in data
        .photon_energy_ev
        .iter()
        .zip(data.relative_energy_ev.iter())
        .zip(data.wave_number.iter())
        .zip(data.mu.iter())
        .zip(data.mu0.iter())
        .zip(data.chi.iter())
    {
        match format {
            XmuRenderFormat::Compact => {
                write!(out, "{omega:12.3}{edge:11.3}{k:8.3}")?;
                write_fortran_exp(&mut out, *mu, 13, 5)?;
                write_fortran_exp(&mut out, *mu0, 13, 5)?;
                write_fortran_exp(&mut out, *chi, 13, 5)?;
            }
            XmuRenderFormat::FPrime => {
                write!(out, "{omega:12.3}{edge:11.3}")?;
                write_fortran_zero_scaled_exp(&mut out, *k, 13, 5)?;
                write_fortran_zero_scaled_exp(&mut out, *mu, 13, 5)?;
                write_fortran_zero_scaled_exp(&mut out, *mu0, 13, 5)?;
                write_fortran_zero_scaled_exp(&mut out, *chi, 13, 5)?;
            }
            XmuRenderFormat::Wide => {
                write!(out, "{omega:21.10}{edge:20.10}{k:20.10}")?;
                write_fortran_exp(&mut out, *mu, 20, 10)?;
                write_fortran_exp(&mut out, *mu0, 20, 10)?;
                write_fortran_exp(&mut out, *chi, 20, 10)?;
            }
        }
        out.push('\n');
    }
    Ok(out)
}

/// Parse FEFF `xmu.dat` text.
pub fn parse_xmu_dat(text: &str) -> Result<XmuDatData> {
    let mut header_lines = Vec::new();
    let mut normalization = None;
    let mut photon_energy_ev = Vec::new();
    let mut relative_energy_ev = Vec::new();
    let mut wave_number = Vec::new();
    let mut mu = Vec::new();
    let mut mu0 = Vec::new();
    let mut chi = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != XMU_DAT_ROW_WIDTH {
                return Err(IoError::XmuDatRowWidth {
                    line: line_number,
                    actual: tokens.len(),
                    expected: XMU_DAT_ROW_WIDTH,
                });
            }
            photon_energy_ev.push(parse_f64(line_number, "omega", tokens[0])?);
            relative_energy_ev.push(parse_f64(line_number, "edge-relative energy", tokens[1])?);
            wave_number.push(parse_f64(line_number, "wave number", tokens[2])?);
            mu.push(parse_f64(line_number, "mu", tokens[3])?);
            mu0.push(parse_f64(line_number, "mu0", tokens[4])?);
            chi.push(parse_f64(line_number, "chi", tokens[5])?);
        } else {
            if let Some(value) = parse_normalization(line, line_number)? {
                normalization = Some(value);
            }
            header_lines.push(raw.to_string());
        }
    }

    let data = XmuDatData {
        header_lines,
        normalization,
        photon_energy_ev: Array1::from_vec(photon_energy_ev),
        relative_energy_ev: Array1::from_vec(relative_energy_ev),
        wave_number: Array1::from_vec(wave_number),
        mu: Array1::from_vec(mu),
        mu0: Array1::from_vec(mu0),
        chi: Array1::from_vec(chi),
    };
    validate_xmu_dat(&data)?;
    Ok(data)
}

/// Write FEFF `xmu.dat` text to a file.
pub fn write_xmu_dat(path: impl AsRef<Path>, data: &XmuDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xmu_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `xmu.dat` text from a file.
pub fn read_xmu_dat(path: impl AsRef<Path>) -> Result<XmuDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xmu_dat(&text)
}

fn parse_normalization(line: &str, line_number: usize) -> Result<Option<f64>> {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("xsedge") && lower.contains("normalize")) {
        return Ok(None);
    }
    let Some(token) = line.split_whitespace().last() else {
        return Ok(None);
    };
    Ok(Some(parse_f64(line_number, "xsedge normalization", token)?))
}

fn validate_xmu_dat(data: &XmuDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_xmu_dat(
            "rows",
            "at least one spectrum row is required",
        ));
    }
    validate_len(
        "relative_energy_ev",
        data.relative_energy_ev.len(),
        point_count,
    )?;
    validate_len("wave_number", data.wave_number.len(), point_count)?;
    validate_len("mu", data.mu.len(), point_count)?;
    validate_len("mu0", data.mu0.len(), point_count)?;
    validate_len("chi", data.chi.len(), point_count)?;

    if let Some(normalization) = data.normalization {
        validate_finite("xsedge normalization", normalization)?;
    }
    for (row, (((((omega, edge), k), mu), mu0), chi)) in data
        .photon_energy_ev
        .iter()
        .zip(data.relative_energy_ev.iter())
        .zip(data.wave_number.iter())
        .zip(data.mu.iter())
        .zip(data.mu0.iter())
        .zip(data.chi.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("omega", *omega, row)?;
        validate_finite_row("edge-relative energy", *edge, row)?;
        validate_finite_row("wave number", *k, row)?;
        validate_finite_row("mu", *mu, row)?;
        validate_finite_row("mu0", *mu0, row)?;
        validate_finite_row("chi", *chi, row)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::XmuDatShape {
            field,
            actual,
            expected,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XmuRenderFormat {
    Compact,
    FPrime,
    Wide,
}

fn xmu_render_format(data: &XmuDatData) -> XmuRenderFormat {
    if looks_like_fprime_xmu(data) {
        XmuRenderFormat::FPrime
    } else if needs_wide_xmu_format(data) {
        XmuRenderFormat::Wide
    } else {
        XmuRenderFormat::Compact
    }
}

fn looks_like_fprime_xmu(data: &XmuDatData) -> bool {
    data.wave_number
        .iter()
        .zip(data.mu.iter())
        .all(|(wave_number, mu)| (*wave_number - *mu).abs() <= COLUMN_EQUALITY_TOLERANCE)
        && data
            .mu0
            .iter()
            .zip(data.chi.iter())
            .all(|(mu0, chi)| (*mu0 - *chi).abs() <= COLUMN_EQUALITY_TOLERANCE)
}

fn needs_wide_xmu_format(data: &XmuDatData) -> bool {
    data.photon_energy_ev
        .iter()
        .chain(data.relative_energy_ev.iter())
        .chain(data.wave_number.iter())
        .any(|value| has_more_decimal_precision(*value, COMPACT_FIXED_PRECISION))
}

fn has_more_decimal_precision(value: f64, precision: i32) -> bool {
    let scale = 10.0_f64.powi(precision);
    let rounded = (value * scale).round() / scale;
    (value - rounded).abs() > 1.0e-9
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::XmuDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_xmu_dat(field, "value must be finite"))
    }
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidXmuDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_xmu_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXmuDat {
        field,
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
    fn parses_feff_xmu_reference_shape() -> Result<()> {
        let data = parse_xmu_dat(XMU_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.normalization, Some(1.2667e-4));
        assert_eq!(data.photon_energy_ev[0], 11076.317);
        assert_eq!(data.relative_energy_ev[1], -39.429);
        assert_eq!(data.wave_number[2], -2.965);
        assert_eq!(data.mu[0], 9.93209e-3);
        assert_eq!(data.mu0[1], 8.38540e-3);
        assert_eq!(data.chi[2], 3.54700e-4);
        let absolute = data
            .absolute_mu()
            .ok_or_else(|| invalid_xmu_dat("mu", "missing norm"))?;
        assert!((absolute[0] - 9.93209e-3 * 1.2667e-4).abs() < 1.0e-14);
        Ok(())
    }

    #[test]
    fn roundtrips_xmu_text() -> Result<()> {
        let data = parse_xmu_dat(XMU_DAT)?;
        let rendered = xmu_dat_string(&data)?;
        assert_eq!(rendered, XMU_DAT);
        assert_eq!(parse_xmu_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_xmu_inputs() {
        assert!(parse_xmu_dat("# no data\n").is_err());
        assert!(parse_xmu_dat("1 2 3\n").is_err());
        assert!(parse_xmu_dat("1 2 3 NaN 5 6\n").is_err());
        assert!(parse_xmu_dat("# xsedge+ 50, used to normalize mu nope\n1 2 3 4 5 6\n").is_err());
    }

    const XMU_DAT: &str = r#"# # Cu                                                           FEFF 10.0.0
#  S02=1.000  Temp=   0.00  Debye_temp=   0.00  Global_sig2= 0.00000
#     0/   0 paths used
#  xsedge+ 50, used to normalize mu           1.2667E-04
#  -----------------------------------------------------------------------
#  omega    e    k    mu    mu0     chi     @#
   11076.317    -40.000  -3.016  9.93209E-03  9.60242E-03  3.29662E-04
   11076.888    -39.429  -2.991  8.72601E-03  8.38540E-03  3.40613E-04
   11077.459    -38.858  -2.965  7.66539E-03  7.31069E-03  3.54700E-04
"#;
}
