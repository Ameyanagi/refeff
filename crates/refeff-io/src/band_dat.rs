//! FEFF `BAND` band-structure output support.
//!
//! `BAND/bandtot.f90` writes `bandstructure.dat` as one row per k-point with a
//! variable number of band energies. `KSPACE/kmesh.f90` can also write
//! `kmesh.dat`, where the first row carries mesh metadata and later rows only
//! carry k-point coordinates and weights.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};

const BANDSTRUCTURE_DAT_PATH: &str = "bandstructure.dat";
const KMESH_DAT_PATH: &str = "kmesh.dat";

/// One k-point row from FEFF `bandstructure.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct BandstructureRow {
    /// One-based k-point index written by FEFF.
    pub index: i32,
    /// Cartesian k-point coordinates.
    pub k_point: [f64; 3],
    /// Band energies at this k-point.
    pub bands: Array1<f64>,
}

/// Parsed FEFF `bandstructure.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct BandstructureDatData {
    /// Comment/header lines before the k-point rows.
    pub header_lines: Vec<String>,
    /// K-point rows in FEFF file order.
    pub rows: Vec<BandstructureRow>,
}

/// Optional mesh metadata written on the first FEFF `kmesh.dat` row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KmeshMetadata {
    /// Requested k-point count.
    pub requested_points: i32,
    /// Irreducible k-point count.
    pub irreducible_points: i32,
    /// K-mesh subdivisions.
    pub divisions: [i32; 3],
}

/// One FEFF `kmesh.dat` row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KmeshRow {
    /// One-based k-point index written by FEFF.
    pub index: i32,
    /// Irreducible Brillouin-zone k-point coordinates.
    pub k_point: [f64; 3],
    /// Integration weight.
    pub weight: f64,
    /// Mesh metadata, usually present only on the first row.
    pub metadata: Option<KmeshMetadata>,
}

/// Parsed FEFF `kmesh.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct KmeshDatData {
    /// K-point rows in FEFF file order.
    pub rows: Vec<KmeshRow>,
}

impl BandstructureDatData {
    /// Number of k-point rows.
    #[must_use]
    pub fn k_point_count(&self) -> usize {
        self.rows.len()
    }

    /// Minimum number of bands found on any k-point row.
    #[must_use]
    pub fn min_band_count(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.bands.len())
            .min()
            .unwrap_or(0)
    }

    /// Maximum number of bands found on any k-point row.
    #[must_use]
    pub fn max_band_count(&self) -> usize {
        self.rows
            .iter()
            .map(|row| row.bands.len())
            .max()
            .unwrap_or(0)
    }
}

impl KmeshDatData {
    /// Number of k-point rows.
    #[must_use]
    pub fn k_point_count(&self) -> usize {
        self.rows.len()
    }
}

/// Parse FEFF `bandstructure.dat` text.
pub fn parse_bandstructure_dat(text: &str) -> Result<BandstructureDatData> {
    let mut header_lines = Vec::new();
    let mut rows = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            header_lines.push(raw.trim_end().to_string());
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 5 {
            return bandstructure_parse_error(
                line_number,
                format!("row has {} token(s), expected at least 5", tokens.len()),
            );
        }
        let band_count = parse_usize(BANDSTRUCTURE_DAT_PATH, line_number, "band count", tokens[4])?;
        if tokens.len() != 5 + band_count {
            return bandstructure_parse_error(
                line_number,
                format!(
                    "row declares {band_count} band(s) but has {} band value token(s)",
                    tokens.len().saturating_sub(5)
                ),
            );
        }
        let bands = tokens[5..]
            .iter()
            .map(|token| parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "band energy", token))
            .collect::<Result<Vec<_>>>()?;
        rows.push(BandstructureRow {
            index: parse_i32(
                BANDSTRUCTURE_DAT_PATH,
                line_number,
                "k-point index",
                tokens[0],
            )?,
            k_point: [
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "kx", tokens[1])?,
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "ky", tokens[2])?,
                parse_f64(BANDSTRUCTURE_DAT_PATH, line_number, "kz", tokens[3])?,
            ],
            bands: Array1::from_vec(bands),
        });
    }

    let data = BandstructureDatData { header_lines, rows };
    validate_bandstructure_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `bandstructure.dat` text.
pub fn bandstructure_dat_string(data: &BandstructureDatData) -> Result<String> {
    validate_bandstructure_dat(data)?;
    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for row in &data.rows {
        write!(
            out,
            "{:5} {:8.4} {:8.4} {:8.4} {:4}",
            row.index,
            row.k_point[0],
            row.k_point[1],
            row.k_point[2],
            row.bands.len()
        )?;
        for band in &row.bands {
            write!(out, " {band:8.4}")?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF `bandstructure.dat` text from a file.
pub fn read_bandstructure_dat(path: impl AsRef<Path>) -> Result<BandstructureDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_bandstructure_dat(&text)
}

/// Write FEFF `bandstructure.dat` text to a file.
pub fn write_bandstructure_dat(path: impl AsRef<Path>, data: &BandstructureDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, bandstructure_dat_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `kmesh.dat` text.
pub fn parse_kmesh_dat(text: &str) -> Result<KmeshDatData> {
    let mut rows = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 5 && tokens.len() != 10 {
            return kmesh_parse_error(
                line_number,
                format!("row has {} token(s), expected 5 or 10", tokens.len()),
            );
        }
        let metadata = if tokens.len() == 10 {
            Some(KmeshMetadata {
                requested_points: parse_i32(
                    KMESH_DAT_PATH,
                    line_number,
                    "requested k-points",
                    tokens[5],
                )?,
                irreducible_points: parse_i32(
                    KMESH_DAT_PATH,
                    line_number,
                    "irreducible k-points",
                    tokens[6],
                )?,
                divisions: [
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division x", tokens[7])?,
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division y", tokens[8])?,
                    parse_i32(KMESH_DAT_PATH, line_number, "k-division z", tokens[9])?,
                ],
            })
        } else {
            None
        };
        rows.push(KmeshRow {
            index: parse_i32(KMESH_DAT_PATH, line_number, "k-point index", tokens[0])?,
            k_point: [
                parse_f64(KMESH_DAT_PATH, line_number, "kx", tokens[1])?,
                parse_f64(KMESH_DAT_PATH, line_number, "ky", tokens[2])?,
                parse_f64(KMESH_DAT_PATH, line_number, "kz", tokens[3])?,
            ],
            weight: parse_f64(KMESH_DAT_PATH, line_number, "weight", tokens[4])?,
            metadata,
        });
    }
    let data = KmeshDatData { rows };
    validate_kmesh_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `kmesh.dat` text.
pub fn kmesh_dat_string(data: &KmeshDatData) -> Result<String> {
    validate_kmesh_dat(data)?;
    let mut out = String::new();
    for row in &data.rows {
        write!(
            out,
            "{:10}{:9.4}{:9.4}{:9.4}{:9.4}",
            row.index, row.k_point[0], row.k_point[1], row.k_point[2], row.weight
        )?;
        if let Some(metadata) = row.metadata {
            write!(
                out,
                "{:7}{:7}{:7}{:7}{:7}",
                metadata.requested_points,
                metadata.irreducible_points,
                metadata.divisions[0],
                metadata.divisions[1],
                metadata.divisions[2]
            )?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF `kmesh.dat` text from a file.
pub fn read_kmesh_dat(path: impl AsRef<Path>) -> Result<KmeshDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_kmesh_dat(&text)
}

/// Write FEFF `kmesh.dat` text to a file.
pub fn write_kmesh_dat(path: impl AsRef<Path>, data: &KmeshDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, kmesh_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_bandstructure_dat(data: &BandstructureDatData) -> Result<()> {
    if data.rows.is_empty() {
        return invalid_bandstructure_dat("rows", "at least one k-point row is required");
    }
    for (row_index, row) in data.rows.iter().enumerate() {
        let row_number = row_index + 1;
        validate_positive_i32(
            BANDSTRUCTURE_DAT_PATH,
            "k-point index",
            row.index,
            row_number,
        )?;
        validate_finite_array(BANDSTRUCTURE_DAT_PATH, "k-point", &row.k_point, row_number)?;
        for value in &row.bands {
            validate_finite_value(BANDSTRUCTURE_DAT_PATH, "band energy", *value, row_number)?;
        }
    }
    Ok(())
}

fn validate_kmesh_dat(data: &KmeshDatData) -> Result<()> {
    if data.rows.is_empty() {
        return invalid_kmesh_dat("rows", "at least one k-point row is required");
    }
    for (row_index, row) in data.rows.iter().enumerate() {
        let row_number = row_index + 1;
        validate_positive_i32(KMESH_DAT_PATH, "k-point index", row.index, row_number)?;
        validate_finite_array(KMESH_DAT_PATH, "k-point", &row.k_point, row_number)?;
        validate_finite_value(KMESH_DAT_PATH, "weight", row.weight, row_number)?;
        if let Some(metadata) = row.metadata {
            validate_positive_i32(
                KMESH_DAT_PATH,
                "requested k-points",
                metadata.requested_points,
                row_number,
            )?;
            validate_positive_i32(
                KMESH_DAT_PATH,
                "irreducible k-points",
                metadata.irreducible_points,
                row_number,
            )?;
            for division in metadata.divisions {
                validate_positive_i32(KMESH_DAT_PATH, "k-division", division, row_number)?;
            }
        }
    }
    Ok(())
}

fn validate_positive_i32(
    path: &'static str,
    field: &'static str,
    value: i32,
    row: usize,
) -> Result<()> {
    if value > 0 {
        Ok(())
    } else {
        invalid_dat(path, field, format!("row {row} value must be positive"))
    }
}

fn validate_finite_array(
    path: &'static str,
    field: &'static str,
    values: &[f64],
    row: usize,
) -> Result<()> {
    for value in values {
        validate_finite_value(path, field, *value, row)?;
    }
    Ok(())
}

fn validate_finite_value(
    path: &'static str,
    field: &'static str,
    value: f64,
    row: usize,
) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_dat(path, field, format!("row {row} value must be finite"))
    }
}

fn parse_i32(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))
}

fn invalid_bandstructure_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    invalid_dat(BANDSTRUCTURE_DAT_PATH, field, message)
}

fn invalid_kmesh_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    invalid_dat(KMESH_DAT_PATH, field, message)
}

fn invalid_dat<T>(
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

fn bandstructure_parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(BANDSTRUCTURE_DAT_PATH, line, message))
}

fn kmesh_parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(KMESH_DAT_PATH, line, message))
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
    use super::*;

    #[test]
    fn parses_bandstructure_dat() -> Result<()> {
        let data = parse_bandstructure_dat(SAMPLE_BANDSTRUCTURE)?;

        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.k_point_count(), 2);
        assert_eq!(data.min_band_count(), 1);
        assert_eq!(data.max_band_count(), 2);
        assert_eq!(data.rows[0].index, 1);
        assert_eq!(data.rows[0].k_point, [0.0, 0.5, 0.25]);
        assert_eq!(data.rows[0].bands.as_slice(), Some(&[-5.0, 1.25][..]));
        assert_eq!(data.rows[1].bands.as_slice(), Some(&[0.75][..]));
        Ok(())
    }

    #[test]
    fn roundtrips_bandstructure_dat() -> Result<()> {
        let data = parse_bandstructure_dat(SAMPLE_BANDSTRUCTURE)?;
        let rendered = bandstructure_dat_string(&data)?;
        let reparsed = parse_bandstructure_dat(&rendered)?;

        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_bandstructure_dat() {
        assert!(parse_bandstructure_dat("    1 0.0 0.0 0.0 2 1.0\n").is_err());
        assert!(parse_bandstructure_dat("# header only\n").is_err());
    }

    #[test]
    fn parses_kmesh_dat() -> Result<()> {
        let data = parse_kmesh_dat(SAMPLE_KMESH)?;

        assert_eq!(data.k_point_count(), 2);
        assert_eq!(data.rows[0].index, 1);
        assert_eq!(data.rows[0].k_point, [0.0, 0.5, 0.25]);
        assert_eq!(data.rows[0].weight, 0.75);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 100,
                irreducible_points: 2,
                divisions: [4, 5, 6],
            })
        );
        assert_eq!(data.rows[1].metadata, None);
        Ok(())
    }

    #[test]
    fn roundtrips_kmesh_dat() -> Result<()> {
        let data = parse_kmesh_dat(SAMPLE_KMESH)?;
        let rendered = kmesh_dat_string(&data)?;
        let reparsed = parse_kmesh_dat(&rendered)?;

        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_kmesh_dat() {
        assert!(parse_kmesh_dat("1 0.0 0.0\n").is_err());
        assert!(parse_kmesh_dat("").is_err());
    }

    const SAMPLE_BANDSTRUCTURE: &str = concat!(
        " # grid of            2  k-points.\n",
        " # grid of            4  energy points  emin=   -5.0000000000000000       , emax=    10.000000000000000       , estep=   0.25000000000000000\n",
        " # Found between            1  and            2  number of bands.\n",
        "    1   0.0000   0.5000   0.2500    2  -5.0000   1.2500\n",
        "    2   0.5000   0.2500   0.0000    1   0.7500\n",
    );

    const SAMPLE_KMESH: &str = concat!(
        "         1   0.0000   0.5000   0.2500   0.7500    100      2      4      5      6\n",
        "         2   0.5000   0.2500   0.0000   0.2500\n",
    );
}
