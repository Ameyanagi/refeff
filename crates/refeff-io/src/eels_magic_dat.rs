//! FEFF `magic.dat` EELS collection-angle table codec.
//!
//! `EELS/writeangulardependence2.f90` writes this sidecar when the `MAGIC`
//! card is enabled. Rows contain the collection semiangle, pi/total ratio,
//! pi contribution, sigma-dipole contribution, total contribution, and the
//! q-mesh point count used for that collection angle.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};

use crate::error::{IoError, Result};

const EELS_MAGIC_DAT_ROW_WIDTH: usize = 6;
const EELS_MAGIC_FLOAT_COLUMNS: usize = 5;
const EELS_MAGIC_HEADER: &str = "#    beta        sp2        pi        sigmadip        total";

/// Parsed FEFF `magic.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsMagicDatData {
    /// Header or comment lines before and around the numeric rows.
    pub header_lines: Vec<String>,
    /// `(collection, column)` rows: `beta`, `sp2`, `pi`, `sigmadip`, `total`.
    pub rows: Array2<f64>,
    /// FEFF `npos` value written at the end of each row.
    pub point_counts: Array1<usize>,
}

impl EelsMagicDatData {
    /// Number of collection-angle rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.rows.nrows()
    }
}

/// Build FEFF-compatible `magic.dat` contents from the core EELS collection table.
#[must_use]
pub fn eels_magic_dat_from_collection_table(
    rows: Array2<f64>,
    point_counts: Array1<usize>,
) -> EelsMagicDatData {
    EelsMagicDatData {
        header_lines: vec![EELS_MAGIC_HEADER.to_string()],
        rows,
        point_counts,
    }
}

/// Render FEFF-compatible `magic.dat` text.
pub fn eels_magic_dat_string(data: &EelsMagicDatData) -> Result<String> {
    validate_eels_magic_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (row, &point_count) in data.rows.axis_iter(Axis(0)).zip(data.point_counts.iter()) {
        for value in row {
            write!(out, "{value:14.9} ")?;
        }
        writeln!(out, "{point_count:7}")?;
    }
    Ok(out)
}

/// Parse FEFF `magic.dat` text.
pub fn parse_eels_magic_dat(text: &str) -> Result<EelsMagicDatData> {
    let mut header_lines = Vec::new();
    let mut rows = Vec::new();
    let mut point_counts = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != EELS_MAGIC_DAT_ROW_WIDTH {
                return Err(parse_error(
                    line_number,
                    format!(
                        "magic.dat row has {} token(s), expected {EELS_MAGIC_DAT_ROW_WIDTH}",
                        tokens.len()
                    ),
                ));
            }
            for (column, token) in tokens.iter().take(EELS_MAGIC_FLOAT_COLUMNS).enumerate() {
                rows.push(parse_f64(line_number, magic_column_name(column), token)?);
            }
            point_counts.push(parse_usize(line_number, "point_count", tokens[5])?);
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let row_count = point_counts.len();
    let rows =
        Array2::from_shape_vec((row_count, EELS_MAGIC_FLOAT_COLUMNS), rows).map_err(|_| {
            IoError::Parse {
                path: "magic.dat".into(),
                line: 0,
                message: "numeric payload did not match magic.dat row count".to_string(),
            }
        })?;
    let data = EelsMagicDatData {
        header_lines,
        rows,
        point_counts: Array1::from_vec(point_counts),
    };
    validate_eels_magic_dat(&data)?;
    Ok(data)
}

/// Write FEFF `magic.dat` text to a file.
pub fn write_eels_magic_dat(path: impl AsRef<Path>, data: &EelsMagicDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, eels_magic_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `magic.dat` text from a file.
pub fn read_eels_magic_dat(path: impl AsRef<Path>) -> Result<EelsMagicDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_eels_magic_dat(&text)
}

fn validate_eels_magic_dat(data: &EelsMagicDatData) -> Result<()> {
    let shape = data.rows.shape();
    if shape[1] != EELS_MAGIC_FLOAT_COLUMNS {
        return invalid_eels_magic_dat(
            "rows",
            format!(
                "magic.dat rows have {} column(s), expected {EELS_MAGIC_FLOAT_COLUMNS}",
                shape[1]
            ),
        );
    }
    if shape[0] == 0 {
        return invalid_eels_magic_dat("rows", "at least one collection row is required");
    }
    if data.point_counts.len() != shape[0] {
        return invalid_eels_magic_dat(
            "point_counts",
            format!(
                "got {} point count(s), expected {}",
                data.point_counts.len(),
                shape[0]
            ),
        );
    }
    for (row, values) in data.rows.axis_iter(Axis(0)).enumerate() {
        for (column, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return invalid_eels_magic_dat(
                    magic_column_name(column),
                    format!("row {} is not finite: {value}", row + 1),
                );
            }
        }
    }
    for (row, &point_count) in data.point_counts.iter().enumerate() {
        if point_count == 0 {
            return invalid_eels_magic_dat(
                "point_count",
                format!("row {} has zero q-mesh point count", row + 1),
            );
        }
    }
    Ok(())
}

fn is_numeric_token(token: &str) -> bool {
    token
        .as_bytes()
        .first()
        .is_some_and(|byte| matches!(byte, b'+' | b'-' | b'.' | b'0'..=b'9'))
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|source| IoError::Parse {
            path: "magic.dat".into(),
            line,
            message: format!("could not parse {field} from {token:?}: {source}"),
        })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|source| IoError::Parse {
        path: "magic.dat".into(),
        line,
        message: format!("could not parse {field} from {token:?}: {source}"),
    })
}

fn magic_column_name(column: usize) -> &'static str {
    match column {
        0 => "beta",
        1 => "sp2",
        2 => "pi",
        3 => "sigmadip",
        4 => "total",
        _ => "unknown",
    }
}

fn parse_error(line: usize, message: String) -> IoError {
    IoError::Parse {
        path: "magic.dat".into(),
        line,
        message,
    }
}

fn invalid_eels_magic_dat(
    field: &'static str,
    message: impl Into<String>,
) -> std::result::Result<(), IoError> {
    Err(IoError::Parse {
        path: "magic.dat".into(),
        line: 0,
        message: format!("invalid {field}: {}", message.into()),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        EelsMagicDatData, eels_magic_dat_from_collection_table, eels_magic_dat_string,
        parse_eels_magic_dat,
    };
    use ndarray::{arr1, arr2};

    #[test]
    fn eels_magic_dat_roundtrips_feff_shape() -> crate::Result<()> {
        let text = concat!(
            "#    beta        sp2        pi        sigmadip        total\n",
            "   0.001000000    0.046371981    0.000000001    0.000000014    0.000000015       2\n",
            "   0.002258101    0.031779589    0.000000137    0.000004159    0.000004295       8\n",
        );

        let data = parse_eels_magic_dat(text)?;

        assert_eq!(
            data.header_lines,
            vec!["#    beta        sp2        pi        sigmadip        total"]
        );
        assert_eq!(data.point_counts.to_vec(), vec![2, 8]);
        assert_eq!(
            data.rows,
            arr2(&[
                [
                    0.001,
                    0.046_371_981,
                    0.000_000_001,
                    0.000_000_014,
                    0.000_000_015
                ],
                [
                    0.002_258_101,
                    0.031_779_589,
                    0.000_000_137,
                    0.000_004_159,
                    0.000_004_295
                ],
            ])
        );
        assert_eq!(parse_eels_magic_dat(&eels_magic_dat_string(&data)?)?, data);
        Ok(())
    }

    #[test]
    fn eels_magic_dat_builds_from_collection_table() {
        let data =
            eels_magic_dat_from_collection_table(arr2(&[[0.001, 0.25, 1.0, 3.0, 4.0]]), arr1(&[3]));

        assert_eq!(
            data,
            EelsMagicDatData {
                header_lines: vec![
                    "#    beta        sp2        pi        sigmadip        total".to_string()
                ],
                rows: arr2(&[[0.001, 0.25, 1.0, 3.0, 4.0]]),
                point_counts: arr1(&[3]),
            }
        );
    }

    #[test]
    fn eels_magic_dat_rejects_bad_rows() {
        let empty = EelsMagicDatData {
            header_lines: Vec::new(),
            rows: arr2::<f64, _>(&[[]]),
            point_counts: arr1(&[]),
        };
        assert!(eels_magic_dat_string(&empty).is_err());
        assert!(parse_eels_magic_dat("0.001 0.2\n").is_err());
        assert!(parse_eels_magic_dat("0.001 0.2 1.0 2.0 3.0 0\n").is_err());
    }
}
