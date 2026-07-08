//! FEFF `cum.dat` first/third-cumulant output codec.
//!
//! FF2X writes this file from `dwadd` when `SIG3`/`alphat` is active. Values in
//! the table are emitted in Angstrom units even though the internal damping
//! calculation uses FEFF code units.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::{IoError, Result};

/// One `cum.dat` path row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CumDatEntry {
    /// FEFF path index.
    pub path_index: usize,
    /// First cumulant, `sig1`, in Angstrom.
    pub first_cumulant_angstrom: f64,
    /// Total Debye-Waller factor, `sig2`, in Angstrom squared.
    pub sigma2_angstrom2: f64,
    /// Third cumulant, `sig3`, in Angstrom cubed.
    pub third_cumulant_angstrom3: f64,
}

/// FEFF `cum.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct CumDatData {
    /// Einstein temperature from `ff2x.inp`, FEFF `thetae`.
    pub einstein_temperature: f64,
    /// Thermal expansion coefficient from `SIG3`, FEFF `alphat`.
    pub thermal_expansion: f64,
    /// Per-path cumulant rows.
    pub entries: Vec<CumDatEntry>,
}

/// Render FEFF-compatible `cum.dat` text.
pub fn cum_dat_string(data: &CumDatData) -> Result<String> {
    validate_cum_dat(data)?;

    let mut out = String::new();
    writeln!(
        out,
        "# first and third icumulant for single scattering paths"
    )?;
    writeln!(
        out,
        "# Einstein-Temp. ={thetae:9.2}   alpha={alphat:9.5}",
        thetae = data.einstein_temperature,
        alphat = data.thermal_expansion
    )?;
    writeln!(out, "#       file   sig1    sig2    sig3 ")?;
    for entry in &data.entries {
        writeln!(
            out,
            "{path_index:>10}{sig1:>9.5}{sig2:>9.5} {sig3:>9.7}",
            path_index = entry.path_index,
            sig1 = entry.first_cumulant_angstrom,
            sig2 = entry.sigma2_angstrom2,
            sig3 = entry.third_cumulant_angstrom3
        )?;
    }
    Ok(out)
}

/// Parse FEFF `cum.dat` text.
pub fn parse_cum_dat(text: &str) -> Result<CumDatData> {
    parse_cum_dat_with_path(PathBuf::from("cum.dat"), text)
}

/// Write FEFF `cum.dat` text.
pub fn write_cum_dat(path: impl AsRef<Path>, data: &CumDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, cum_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `cum.dat` text.
pub fn read_cum_dat(path: impl AsRef<Path>) -> Result<CumDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_cum_dat_with_path(path.to_path_buf(), &text)
}

fn parse_cum_dat_with_path(path: PathBuf, text: &str) -> Result<CumDatData> {
    let mut einstein_temperature = None;
    let mut thermal_expansion = None;
    let mut entries = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("# Einstein-Temp.") {
            let values = trimmed
                .split_whitespace()
                .filter_map(|token| token.parse::<f64>().ok())
                .collect::<Vec<_>>();
            if values.len() != 2 {
                return Err(parse_error(
                    &path,
                    line_number,
                    "expected thetae and alphat on Einstein-Temp. header",
                ));
            }
            einstein_temperature = Some(values[0]);
            thermal_expansion = Some(values[1]);
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }

        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 4 {
            return Err(parse_error(
                &path,
                line_number,
                format!("expected 4 table columns, got {}", tokens.len()),
            ));
        }
        entries.push(CumDatEntry {
            path_index: parse_usize(&path, line_number, tokens[0], "path_index")?,
            first_cumulant_angstrom: parse_f64(&path, line_number, tokens[1], "sig1")?,
            sigma2_angstrom2: parse_f64(&path, line_number, tokens[2], "sig2")?,
            third_cumulant_angstrom3: parse_f64(&path, line_number, tokens[3], "sig3")?,
        });
    }

    let data = CumDatData {
        einstein_temperature: einstein_temperature.ok_or_else(|| {
            parse_error(
                &path,
                1,
                "missing Einstein-Temp. header with thetae and alphat",
            )
        })?,
        thermal_expansion: thermal_expansion.ok_or_else(|| {
            parse_error(
                &path,
                1,
                "missing Einstein-Temp. header with thetae and alphat",
            )
        })?,
        entries,
    };
    validate_cum_dat(&data)?;
    Ok(data)
}

fn validate_cum_dat(data: &CumDatData) -> Result<()> {
    validate_finite("thetae", data.einstein_temperature)?;
    validate_finite("alphat", data.thermal_expansion)?;
    for entry in &data.entries {
        if entry.path_index == 0 {
            return Err(invalid_cum_dat("path_index", "path index must be positive"));
        }
        validate_finite("sig1", entry.first_cumulant_angstrom)?;
        validate_finite("sig2", entry.sigma2_angstrom2)?;
        validate_finite("sig3", entry.third_cumulant_angstrom3)?;
    }
    Ok(())
}

fn parse_usize(path: &Path, line: usize, token: &str, field: &'static str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| {
        parse_error(
            path,
            line,
            format!("could not parse {field} from token {token:?}"),
        )
    })
}

fn parse_f64(path: &Path, line: usize, token: &str, field: &'static str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| {
        parse_error(
            path,
            line,
            format!("could not parse {field} from token {token:?}"),
        )
    })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_cum_dat(field, "value must be finite"))
    }
}

fn invalid_cum_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: PathBuf::from("cum.dat"),
        line: 0,
        message: format!("invalid {field}: {}", message.into()),
    }
}

fn parse_error(path: &Path, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.to_path_buf(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cum_dat_roundtrips_feff_format() -> Result<()> {
        let data = CumDatData {
            einstein_temperature: 250.0,
            thermal_expansion: 0.034,
            entries: vec![
                CumDatEntry {
                    path_index: 1,
                    first_cumulant_angstrom: 0.00001,
                    sigma2_angstrom2: 0.00610,
                    third_cumulant_angstrom3: 0.0000007,
                },
                CumDatEntry {
                    path_index: 12,
                    first_cumulant_angstrom: -0.00002,
                    sigma2_angstrom2: 0.02442,
                    third_cumulant_angstrom3: -0.0000013,
                },
            ],
        };

        let text = cum_dat_string(&data)?;

        assert!(text.contains("# Einstein-Temp. =   250.00   alpha=  0.03400"));
        assert!(text.contains("         1  0.00001  0.00610 0.0000007"));
        assert_eq!(parse_cum_dat(&text)?, data);
        Ok(())
    }

    #[test]
    fn cum_dat_rejects_missing_header() {
        assert!(matches!(
            parse_cum_dat("         1  0.00001  0.00610 0.0000007\n"),
            Err(IoError::Parse { .. })
        ));
    }
}
