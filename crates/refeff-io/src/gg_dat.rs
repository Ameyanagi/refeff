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

use ndarray::{Array2, Array3, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const GG_DAT_PATH: &str = "gg.dat";

/// One `Write2D` section from FEFF `gg.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct GgDatSection {
    /// One-based section number from the `#SN#` marker.
    pub section_number: usize,
    /// Complex Green's-function matrix for this section.
    pub values: Array2<Complex64>,
    /// Raw section prefix lines through the `#DT#`/`#H#` boilerplate.
    ///
    /// Generated FEFF `gg.dat` and `gg.bin` files can contain non-UTF bytes in
    /// `#DF#` descriptor lines. Byte readers preserve those lines here so
    /// byte-level roundtrips do not lose the original descriptor payload.
    pub raw_prefix_lines: Option<Vec<Vec<u8>>>,
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

    /// Assemble this FEFF Green-function stream for RIXS core kernels.
    pub fn to_rixs_handoff(&self) -> Result<GgDatRixsHandoff> {
        gg_dat_rixs_handoff(self)
    }
}

/// RIXS-ready view of a FEFF `gg.dat`/`gg.bin` Green-function stream.
#[derive(Debug, Clone, PartialEq)]
pub struct GgDatRixsHandoff {
    /// Square Green-function matrix order, FEFF angular-channel count.
    pub angular_count: usize,
    /// Number of Green-function energy sections.
    pub energy_count: usize,
    /// Green functions in RIXS `(L1, L2, energy)` order.
    pub green: Array3<Complex64>,
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
            raw_prefix_lines: None,
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

/// Parse FEFF `gg.dat` bytes while preserving raw section prefix lines.
pub fn parse_gg_dat_bytes(bytes: &[u8]) -> Result<GgDatData> {
    let text = String::from_utf8_lossy(bytes);
    let mut data = parse_gg_dat(&text)?;
    let prefixes = collect_raw_prefix_lines(bytes)?;
    if prefixes.len() != data.sections.len() {
        return parse_error(
            0,
            format!(
                "found {} raw section prefix block(s), expected {}",
                prefixes.len(),
                data.sections.len()
            ),
        );
    }
    for (section, prefix) in data.sections.iter_mut().zip(prefixes) {
        section.raw_prefix_lines = Some(prefix);
    }
    Ok(data)
}

/// Parse FEFF `gg.bin` bytes while preserving raw section prefix lines.
pub fn parse_gg_bin_bytes(bytes: &[u8]) -> Result<GgDatData> {
    parse_gg_dat_bytes(bytes)
}

/// Render FEFF-compatible `gg.dat` text.
pub fn gg_dat_string(data: &GgDatData) -> Result<String> {
    validate_gg_dat(data)?;
    let mut out = String::new();
    for section in &data.sections {
        write_canonical_section_prefix(&mut out, section)?;
        for row in section.values.rows() {
            for value in row {
                write_fortran_zero_scaled_exp(&mut out, value.re, 20, 10)?;
                out.push(' ');
                write_fortran_zero_scaled_exp(&mut out, value.im, 20, 10)?;
            }
            writeln!(out)?;
        }
    }
    Ok(out)
}

/// Render FEFF-compatible `gg.dat` bytes, preserving raw non-UTF section
/// descriptor lines when they came from [`parse_gg_dat_bytes`].
pub fn gg_dat_bytes(data: &GgDatData) -> Result<Vec<u8>> {
    validate_gg_dat(data)?;
    let mut out = Vec::new();
    for section in &data.sections {
        if let Some(prefix_lines) = &section.raw_prefix_lines {
            for line in prefix_lines {
                out.extend_from_slice(line);
                out.push(b'\n');
            }
        } else {
            let mut prefix = String::new();
            write_canonical_section_prefix(&mut prefix, section)?;
            out.extend_from_slice(prefix.as_bytes());
        }

        let mut row_text = String::new();
        for row in section.values.rows() {
            row_text.clear();
            for value in row {
                write_fortran_zero_scaled_exp(&mut row_text, value.re, 20, 10)?;
                row_text.push(' ');
                write_fortran_zero_scaled_exp(&mut row_text, value.im, 20, 10)?;
            }
            row_text.push('\n');
            out.extend_from_slice(row_text.as_bytes());
        }
    }
    Ok(out)
}

/// Render FEFF-compatible `gg.bin` text.
pub fn gg_bin_string(data: &GgDatData) -> Result<String> {
    gg_dat_string(data)
}

/// Render FEFF-compatible `gg.bin` bytes.
pub fn gg_bin_bytes(data: &GgDatData) -> Result<Vec<u8>> {
    gg_dat_bytes(data)
}

fn write_canonical_section_prefix(out: &mut String, section: &GgDatSection) -> Result<()> {
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
    Ok(())
}

/// Read FEFF `gg.dat` from a file.
///
/// FEFF can emit non-UTF-8 bytes in the descriptive `#DF#` line. This reader
/// decodes the file lossily because the parser only relies on ASCII section
/// markers and numeric rows.
pub fn read_gg_dat(path: impl AsRef<Path>) -> Result<GgDatData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_gg_dat_bytes(&bytes)
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
    std::fs::write(path, gg_dat_bytes(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `gg.bin` text to a file.
pub fn write_gg_bin(path: impl AsRef<Path>, data: &GgDatData) -> Result<()> {
    write_gg_dat(path, data)
}

/// Build the FEFF RIXS Green-function handoff from parsed `gg.dat`/`gg.bin`.
///
/// FEFF writes one `gg(L1,L2)` matrix per energy section. RIXS core kernels
/// consume the same values as a single `(L1, L2, energy)` tensor.
pub fn gg_dat_rixs_handoff(data: &GgDatData) -> Result<GgDatRixsHandoff> {
    validate_gg_dat(data)?;
    let first = data
        .sections
        .first()
        .ok_or_else(|| parse_error_value(0, "at least one section is required"))?;
    let (rows, columns) = first.shape();
    if rows != columns {
        return parse_error(
            first.section_number,
            format!("RIXS Green-function matrix must be square, got {rows}x{columns}"),
        );
    }

    let energy_count = data.section_count();
    let mut green = Array3::zeros((rows, columns, energy_count));
    for (energy, section) in data.sections.iter().enumerate() {
        if section.shape() != (rows, columns) {
            return parse_error(
                section.section_number,
                format!(
                    "RIXS Green-function section shape {:?} does not match first section shape {:?}",
                    section.shape(),
                    (rows, columns)
                ),
            );
        }
        for row in 0..rows {
            for column in 0..columns {
                green[(row, column, energy)] = section.values[(row, column)];
            }
        }
    }

    Ok(GgDatRixsHandoff {
        angular_count: rows,
        energy_count,
        green,
    })
}

fn collect_raw_prefix_lines(bytes: &[u8]) -> Result<Vec<Vec<Vec<u8>>>> {
    let mut prefixes = Vec::new();
    let mut current = Vec::new();
    let mut in_prefix = false;

    for raw in bytes.split(|byte| *byte == b'\n') {
        let line = strip_trailing_cr(raw);
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() {
            if in_prefix {
                current.push(line.to_vec());
            }
            continue;
        }

        if trimmed.starts_with(b"#SN#") {
            if in_prefix && !current.is_empty() {
                return parse_error(0, "section prefix ended before a data row");
            }
            current.clear();
            current.push(line.to_vec());
            in_prefix = true;
            continue;
        }

        if !in_prefix {
            continue;
        }

        if is_prefix_line(trimmed) {
            current.push(line.to_vec());
            continue;
        }

        if current.is_empty() {
            return parse_error(0, "data row appeared before a #SN# section marker");
        }
        prefixes.push(std::mem::take(&mut current));
        in_prefix = false;
    }

    if in_prefix && !current.is_empty() {
        return parse_error(0, "section prefix ended before a data row");
    }

    Ok(prefixes)
}

fn is_prefix_line(line: &[u8]) -> bool {
    line.starts_with(b"#DF#") || line.starts_with(b"#H#") || line.starts_with(b"#DT#")
}

fn strip_trailing_cr(line: &[u8]) -> &[u8] {
    if let Some(rest) = line.strip_suffix(b"\r") {
        rest
    } else {
        line
    }
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = match bytes.iter().position(|byte| !byte.is_ascii_whitespace()) {
        Some(index) => index,
        None => bytes.len(),
    };
    let end = match bytes.iter().rposition(|byte| !byte.is_ascii_whitespace()) {
        Some(index) => index + 1,
        None => start,
    };
    &bytes[start..end]
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
    fn preserves_non_utf_descriptor_bytes() -> Result<()> {
        let bytes = b"#SN#   Section:    1\n#DF# This section written in \0\xc0\xc2v.\n#H#\n#DT# 2D complex array with sizes    1   1\n#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . .\n#H#                                Array(2,i)\n#H#                                     .\n#H#                                     .\n#H#                                     .\n    0.1000000000E+01    -0.2500000000E+01\n";
        let parsed = parse_gg_dat_bytes(bytes)?;
        assert_eq!(parsed.section_count(), 1);
        assert_eq!(parsed.sections[0].values[(0, 0)], Complex64::new(1.0, -2.5));
        assert_eq!(gg_dat_bytes(&parsed)?, bytes);
        Ok(())
    }

    #[test]
    fn gg_dat_builds_rixs_green_handoff() -> Result<()> {
        let data = sample_square_gg_dat();
        let handoff = data.to_rixs_handoff()?;

        assert_eq!(handoff.angular_count, 2);
        assert_eq!(handoff.energy_count, 2);
        assert_eq!(handoff.green.dim(), (2, 2, 2));
        assert_eq!(handoff.green[(0, 0, 0)], Complex64::new(1.0, -1.0));
        assert_eq!(handoff.green[(1, 0, 1)], Complex64::new(7.0, -7.0));
        Ok(())
    }

    #[test]
    fn gg_dat_rixs_handoff_rejects_non_square_or_mismatched_sections() {
        let non_square = GgDatData {
            sections: vec![GgDatSection {
                section_number: 1,
                values: Array2::from_shape_vec(
                    (1, 2),
                    vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                )
                .expect("sample shape"),
                raw_prefix_lines: None,
            }],
        };
        assert!(gg_dat_rixs_handoff(&non_square).is_err());

        let mut mismatched = sample_square_gg_dat();
        mismatched.sections.push(GgDatSection {
            section_number: 3,
            values: Array2::from_elem((3, 3), Complex64::new(0.0, 0.0)),
            raw_prefix_lines: None,
        });
        assert!(gg_dat_rixs_handoff(&mismatched).is_err());
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

    fn sample_square_gg_dat() -> GgDatData {
        GgDatData {
            sections: vec![
                GgDatSection {
                    section_number: 1,
                    values: Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(1.0, -1.0),
                            Complex64::new(2.0, -2.0),
                            Complex64::new(3.0, -3.0),
                            Complex64::new(4.0, -4.0),
                        ],
                    )
                    .expect("sample shape"),
                    raw_prefix_lines: None,
                },
                GgDatSection {
                    section_number: 2,
                    values: Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            Complex64::new(5.0, -5.0),
                            Complex64::new(6.0, -6.0),
                            Complex64::new(7.0, -7.0),
                            Complex64::new(8.0, -8.0),
                        ],
                    )
                    .expect("sample shape"),
                    raw_prefix_lines: None,
                },
            ],
        }
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
