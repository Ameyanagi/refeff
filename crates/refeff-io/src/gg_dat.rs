//! FEFF `gg.dat`/`gg.bin` 2-D complex Green's-function support.
//!
//! FMS writes `gg.bin` and MKGTR can write `gg.dat` through FEFF's generic
//! `Write2D` path. Despite the `.bin` suffix, FEFF's generated files are
//! sectioned text records with a `#SN#` marker, a `#DT#` shape line, and one
//! formatted complex matrix per section. The Rust parser keeps the sections
//! typed so generated FMS Green's-function handoff files can be checked
//! directly.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};

const GG_DAT_PATH: &str = "gg.dat";

/// One `Write2D` section from FEFF `gg.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GgDatSection {
    /// One-based section number from the `#SN#` marker.
    pub section_number: usize,
    /// Complex Green's-function matrix for this section.
    pub values: Array2<Complex64>,
}

impl GgDatSection {
    /// Matrix shape as `(rows, columns)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.values.dim()
    }

    /// Number of matrix rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of matrix columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }
}

/// Parsed contents of FEFF `gg.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GgDatData {
    /// `Write2D` sections in file order.
    pub sections: Vec<GgDatSection>,
}

impl GgDatData {
    /// Number of matrix sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

/// Parse FEFF `gg.dat` text.
pub fn parse_gg_dat(text: &str) -> Result<GgDatData> {
    let lines = text.lines().enumerate().collect::<Vec<_>>();
    let mut sections = Vec::new();
    let mut position = 0;

    while position < lines.len() {
        let (line_number, raw) = lines[position];
        let line = raw.trim();
        if line.is_empty() {
            position += 1;
            continue;
        }
        if !line.starts_with("#SN#") {
            return parse_error(
                line_number + 1,
                format!("expected #SN# section marker, found {line:?}"),
            );
        }
        let section_number = parse_section_number(line_number + 1, line)?;
        position += 1;

        let (rows, columns) = find_section_shape(&lines, &mut position)?;
        let values = parse_section_matrix(&lines, &mut position, rows, columns)?;
        sections.push(GgDatSection {
            section_number,
            values,
        });
    }

    let data = GgDatData { sections };
    validate_gg_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `gg.bin` text.
pub fn parse_gg_bin(text: &str) -> Result<GgDatData> {
    parse_gg_dat(text)
}

/// Render FEFF-compatible `gg.dat` text.
pub fn gg_dat_string(data: &GgDatData) -> Result<String> {
    validate_gg_dat(data)?;
    let mut out = String::new();
    for section in &data.sections {
        let (rows, columns) = section.shape();
        writeln!(out, "#SN#   Section: {:4}", section.section_number)?;
        writeln!(out, "#DF# This section written in txt.")?;
        writeln!(out, "#H#")?;
        writeln!(
            out,
            "#DT# 2D complex array with sizes {:4}{:4}",
            rows, columns
        )?;
        writeln!(
            out,
            "#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . ."
        )?;
        writeln!(out, "#H#                                Array(2,i)")?;
        writeln!(out, "#H#                                     .")?;
        writeln!(out, "#H#                                     .")?;
        writeln!(out, "#H#                                     .")?;
        for row in section.values.rows() {
            for value in row {
                write!(out, "{:20.10E} {:20.10E}", value.re, value.im)?;
            }
            writeln!(out)?;
        }
    }
    Ok(out)
}

/// Render FEFF-compatible `gg.bin` text.
pub fn gg_bin_string(data: &GgDatData) -> Result<String> {
    gg_dat_string(data)
}

/// Read FEFF `gg.dat` from a file.
///
/// FEFF can emit non-UTF-8 bytes in the descriptive `#DF#` line. This reader
/// decodes the file lossily because the parser only relies on ASCII section
/// markers and numeric rows.
pub fn read_gg_dat(path: impl AsRef<Path>) -> Result<GgDatData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_gg_dat(&text)
}

/// Read FEFF `gg.bin` from a file.
///
/// FEFF's `gg.bin` uses the same sectioned text format as `gg.dat` in the
/// generated reference suite.
pub fn read_gg_bin(path: impl AsRef<Path>) -> Result<GgDatData> {
    read_gg_dat(path)
}

/// Write FEFF `gg.dat` text to a file.
pub fn write_gg_dat(path: impl AsRef<Path>, data: &GgDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, gg_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `gg.bin` text to a file.
pub fn write_gg_bin(path: impl AsRef<Path>, data: &GgDatData) -> Result<()> {
    write_gg_dat(path, data)
}

fn find_section_shape(lines: &[(usize, &str)], position: &mut usize) -> Result<(usize, usize)> {
    let mut shape = None;
    while *position < lines.len() {
        let (index, raw) = lines[*position];
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || is_header_line(line) {
            *position += 1;
            continue;
        }
        if line.starts_with("#SN#") {
            return parse_error(line_number, "section is missing #DT# shape line");
        }
        if line.starts_with("#DT#") {
            shape = Some(parse_shape_line(line_number, line)?);
            *position += 1;
            continue;
        }
        if let Some(shape) = shape {
            return Ok(shape);
        }
        return parse_error(
            line_number,
            format!("expected #DT# shape line before data, found {line:?}"),
        );
    }
    shape.ok_or_else(|| parse_error_value(0, "section is missing #DT# shape line"))
}

fn parse_section_matrix(
    lines: &[(usize, &str)],
    position: &mut usize,
    rows: usize,
    columns: usize,
) -> Result<Array2<Complex64>> {
    let mut values = Vec::with_capacity(checked_product(rows, columns)?);
    for row_index in 0..rows {
        let (line_number, line) = next_data_line(lines, position, row_index + 1)?;
        let row = parse_complex_row(line_number, line, columns)?;
        values.extend(row);
    }
    Array2::from_shape_vec((rows, columns), values)
        .map_err(|source| parse_error_value(0, format!("invalid section matrix shape: {source}")))
}

fn next_data_line<'a>(
    lines: &'a [(usize, &'a str)],
    position: &mut usize,
    row: usize,
) -> Result<(usize, &'a str)> {
    while *position < lines.len() {
        let (index, raw) = lines[*position];
        *position += 1;
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || is_header_line(line) {
            continue;
        }
        if line.starts_with("#SN#") {
            return parse_error(
                line_number,
                format!("section ended before matrix row {row} was read"),
            );
        }
        if let Some(data) = line.strip_prefix("#HD#") {
            return Ok((line_number, data.trim()));
        }
        if line.starts_with('#') {
            return parse_error(line_number, format!("unexpected marker {line:?} in data"));
        }
        return Ok((line_number, line));
    }
    parse_error(0, format!("missing matrix row {row}"))
}

fn parse_complex_row(line_number: usize, line: &str, columns: usize) -> Result<Vec<Complex64>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let expected = checked_product(columns, 2)?;
    if tokens.len() != expected {
        return parse_error(
            line_number,
            format!(
                "matrix row has {} token(s), expected {expected}",
                tokens.len()
            ),
        );
    }

    tokens
        .chunks_exact(2)
        .map(|pair| {
            Ok(Complex64::new(
                parse_f64(line_number, "real", pair[0])?,
                parse_f64(line_number, "imaginary", pair[1])?,
            ))
        })
        .collect()
}

fn parse_section_number(line_number: usize, line: &str) -> Result<usize> {
    let token = line
        .split_whitespace()
        .last()
        .ok_or_else(|| parse_error_value(line_number, "missing section number"))?;
    parse_usize(line_number, "section_number", token)
}

fn parse_shape_line(line_number: usize, line: &str) -> Result<(usize, usize)> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let has_supported_type = tokens.iter().any(|token| token.eq_ignore_ascii_case("2D"))
        && tokens
            .iter()
            .any(|token| token.eq_ignore_ascii_case("complex"))
        && tokens
            .iter()
            .any(|token| token.eq_ignore_ascii_case("array"));
    if !has_supported_type {
        return parse_error(line_number, "expected 2D complex array shape line");
    }
    let rows_token = tokens
        .get(tokens.len().saturating_sub(2))
        .ok_or_else(|| parse_error_value(line_number, "missing matrix row count"))?;
    let columns_token = tokens
        .last()
        .ok_or_else(|| parse_error_value(line_number, "missing matrix column count"))?;
    let rows = parse_usize(line_number, "rows", rows_token)?;
    let columns = parse_usize(line_number, "columns", columns_token)?;
    if rows == 0 || columns == 0 {
        return parse_error(line_number, "matrix dimensions must be positive");
    }
    Ok((rows, columns))
}

fn validate_gg_dat(data: &GgDatData) -> Result<()> {
    if data.section_count() == 0 {
        return parse_error(0, "at least one section is required");
    }
    for (index, section) in data.sections.iter().enumerate() {
        let row = index + 1;
        if section.section_number == 0 {
            return parse_error(row, "section number must be positive");
        }
        if section.row_count() == 0 || section.column_count() == 0 {
            return parse_error(row, "section matrix dimensions must be positive");
        }
        for value in &section.values {
            validate_complex("value", *value, row)?;
        }
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

fn checked_product(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| parse_error_value(0, "matrix shape overflows usize"))
}

fn is_header_line(line: &str) -> bool {
    line.starts_with("#DF#") || line.starts_with("#H#")
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
        path: GG_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gg_dat_sections() -> Result<()> {
        let parsed = parse_gg_dat(GG_DAT)?;
        assert_eq!(parsed.section_count(), 2);
        assert_eq!(parsed.sections[0].section_number, 1);
        assert_eq!(parsed.sections[0].shape(), (2, 2));
        assert_eq!(parsed.sections[0].values[(0, 0)], Complex64::new(1.0, -0.5));
        assert_eq!(
            parsed.sections[0].values[(1, 1)],
            Complex64::new(-4.0, 0.75)
        );
        assert_eq!(parsed.sections[1].shape(), (1, 2));
        assert_eq!(parsed.sections[1].values[(0, 1)], Complex64::new(6.0, -6.5));

        let rendered = gg_dat_string(&parsed)?;
        assert_eq!(parse_gg_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents_and_header_data_prefix() -> Result<()> {
        let parsed = parse_gg_dat(
            r#"#SN#   Section:    1
#DT# 2D double complex array with sizes    1   1
#DT# 2D complex array with sizes    1   1
#HD# 1.0D+00 -2.5D+00
"#,
        )?;
        assert_eq!(parsed.section_count(), 1);
        assert_eq!(parsed.sections[0].values[(0, 0)], Complex64::new(1.0, -2.5));
        Ok(())
    }

    #[test]
    fn rejects_bad_gg_dat_inputs() {
        assert!(parse_gg_dat("").is_err());
        assert!(
            parse_gg_dat("#SN# Section: 0\n#DT# 2D complex array with sizes 1 1\n1 2\n").is_err()
        );
        assert!(parse_gg_dat("#SN# Section: 1\n1 2\n").is_err());
        assert!(parse_gg_dat("#SN# Section: 1\n#DT# 2D real array with sizes 1 1\n1 2\n").is_err());
        assert!(
            parse_gg_dat("#SN# Section: 1\n#DT# 2D complex array with sizes 1 1\n1\n").is_err()
        );
        assert!(
            parse_gg_dat("#SN# Section: 1\n#DT# 2D complex array with sizes 1 1\nNaN 1\n").is_err()
        );
    }

    const GG_DAT: &str = r#"#SN#   Section:    1
#DF# This section written in txt.
#H#
#DT# 2D complex array with sizes    2   2
#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . .
#H#                                Array(2,i)
    1.0000000000E+00    -5.0000000000E-01    2.0000000000E+00     2.5000000000E+00
    3.0000000000E+00     0.0000000000E+00   -4.0000000000E+00     7.5000000000E-01
#SN#   Section:    2
#DT# 2D complex array with sizes    1   2
    5.0000000000E+00    -5.5000000000E+00    6.0000000000E+00    -6.5000000000E+00
"#;
}
