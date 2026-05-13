//! FEFF `apot.bin` TXT section-stream codec.
//!
//! The atomic-potential stage writes `apot.bin` through FEFF's generic
//! `WriteData`, `WriteArrayData`, and `Write2D` helpers. The generated file is
//! a text stream of `#SN#` sections even though the suffix is `.bin`. This
//! module keeps the stream typed while preserving section headers and FEFF's
//! column-major matrix shapes in [`ndarray::Array2`] values.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

/// Scalar type marker written in a FEFF `#DT#` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApotBinType {
    /// Fortran `INTEGER`.
    Int,
    /// Fortran `REAL`.
    Real,
    /// Fortran `DOUBLE PRECISION`.
    Double,
    /// Fortran `COMPLEX`.
    Complex,
    /// Fortran `COMPLEX*16`.
    DComplex,
    /// Fixed-width Fortran character payload.
    Text,
}

impl ApotBinType {
    fn record_name(self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::Real => "Real",
            Self::Double => "Double",
            Self::Complex => "Complex",
            Self::DComplex => "DComplex",
            Self::Text => "String",
        }
    }

    fn matrix_name(self) -> &'static str {
        match self {
            Self::Int => "integer",
            Self::Real => "real",
            Self::Double => "double",
            Self::Complex => "complex",
            Self::DComplex => "double complex",
            Self::Text => "string",
        }
    }

    fn token_width(self) -> usize {
        match self {
            Self::Complex | Self::DComplex => 2,
            Self::Int | Self::Real | Self::Double | Self::Text => 1,
        }
    }

    fn is_real(self) -> bool {
        matches!(self, Self::Real | Self::Double)
    }

    fn is_complex(self) -> bool {
        matches!(self, Self::Complex | Self::DComplex)
    }
}

/// One typed scalar value from a FEFF `WriteData` or `WriteArrayData` section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinValue {
    /// Integer value.
    Int(i64),
    /// Real or double-precision value.
    Real(f64),
    /// Complex or double-complex value.
    Complex(Complex64),
    /// Character value.
    Text(String),
}

/// Typed row data from a FEFF `WriteData` or `WriteArrayData` section.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinRecords {
    /// Column types from the section `#DT#` line.
    pub column_types: Vec<ApotBinType>,
    /// Parsed rows. Scalar `WriteData` sections have one row; `WriteArrayData`
    /// sections have one row per array element.
    pub rows: Vec<Vec<ApotBinValue>>,
}

impl ApotBinRecords {
    /// Number of parsed data rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of typed values per row.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.column_types.len()
    }
}

/// Storage for a FEFF `Write2D` matrix section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinMatrixValues {
    /// Integer matrix.
    Int(Array2<i64>),
    /// Real or double-precision matrix.
    Real(Array2<f64>),
    /// Complex or double-complex matrix.
    Complex(Array2<Complex64>),
    /// Character matrix.
    Text(Array2<String>),
}

impl ApotBinMatrixValues {
    fn dim(&self) -> (usize, usize) {
        match self {
            Self::Int(values) => values.dim(),
            Self::Real(values) => values.dim(),
            Self::Complex(values) => values.dim(),
            Self::Text(values) => values.dim(),
        }
    }
}

/// Typed payload from a FEFF `Write2D` section.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinMatrix {
    /// Matrix element type from the section `#DT#` line.
    pub value_type: ApotBinType,
    /// Matrix values in FEFF row/column order.
    pub values: ApotBinMatrixValues,
}

impl ApotBinMatrix {
    /// Matrix shape as `(rows, columns)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.values.dim()
    }

    /// Number of matrix rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.shape().0
    }

    /// Number of matrix columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.shape().1
    }
}

/// Parsed payload variant for one FEFF `apot.bin` section.
#[derive(Debug, Clone, PartialEq)]
pub enum ApotBinPayload {
    /// Header-only section or header lines attached outside a data section.
    HeadersOnly,
    /// `WriteData` or `WriteArrayData` row payload.
    Records(ApotBinRecords),
    /// `Write2D` matrix payload.
    Matrix(ApotBinMatrix),
}

/// One `#SN#` section from FEFF `apot.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinSection {
    /// One-based section number from the `#SN#` marker.
    pub section_number: usize,
    /// Semantic `#H#` lines before the payload.
    pub headers: Vec<String>,
    /// Raw payloads after `#H#` for [`Self::headers`], preserving FEFF padding.
    pub header_texts: Vec<String>,
    /// `#CL#` column labels when FEFF supplied them.
    pub column_labels: Vec<String>,
    /// Raw `#CL#` payload after the marker, preserving FEFF field padding.
    pub column_label_text: Option<String>,
    /// Parsed payload.
    pub payload: ApotBinPayload,
    /// Semantic `#H#` lines emitted after the payload, before the next section.
    ///
    /// FEFF sometimes uses a header-only `WriteData` call to annotate the next
    /// group of matrix sections. Because that call does not force a new section,
    /// those headers are attached to the previous section in the raw file.
    pub trailing_headers: Vec<String>,
    /// Raw payloads after `#H#` for [`Self::trailing_headers`].
    pub trailing_header_texts: Vec<String>,
}

impl ApotBinSection {
    /// Return row-record payload when this is a `WriteData` or
    /// `WriteArrayData` section.
    #[must_use]
    pub fn records(&self) -> Option<&ApotBinRecords> {
        match &self.payload {
            ApotBinPayload::Records(records) => Some(records),
            ApotBinPayload::HeadersOnly | ApotBinPayload::Matrix(_) => None,
        }
    }

    /// Return matrix payload when this is a `Write2D` section.
    #[must_use]
    pub fn matrix(&self) -> Option<&ApotBinMatrix> {
        match &self.payload {
            ApotBinPayload::Matrix(matrix) => Some(matrix),
            ApotBinPayload::HeadersOnly | ApotBinPayload::Records(_) => None,
        }
    }
}

/// Parsed FEFF `apot.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ApotBinData {
    /// Sections in file order.
    pub sections: Vec<ApotBinSection>,
}

impl ApotBinData {
    /// Number of parsed sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of parsed matrix sections.
    #[must_use]
    pub fn matrix_count(&self) -> usize {
        self.sections
            .iter()
            .filter(|section| section.matrix().is_some())
            .count()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ApotBinDescriptor {
    Records(Vec<ApotBinType>),
    Matrix {
        value_type: ApotBinType,
        rows: usize,
        columns: usize,
    },
}

/// Parse FEFF `apot.bin` TXT section-stream text.
pub fn parse_apot_bin(text: &str) -> Result<ApotBinData> {
    let lines = text.lines().enumerate().collect::<Vec<_>>();
    let mut sections = Vec::new();
    let mut position = 0;

    while position < lines.len() {
        let (line_index, raw) = lines[position];
        let line = raw.trim();
        if line.is_empty() {
            position += 1;
            continue;
        }
        if !line.starts_with("#SN#") {
            return invalid_apot_bin(
                line_index + 1,
                format!("expected #SN# section marker, found {line:?}"),
            );
        }

        let start = position;
        position += 1;
        while position < lines.len() && !lines[position].1.trim_start().starts_with("#SN#") {
            position += 1;
        }
        sections.push(parse_section(&lines[start..position])?);
    }

    if sections.is_empty() {
        return invalid_apot_bin(0, "at least one apot.bin section is required");
    }
    validate_apot_bin(&sections)?;
    Ok(ApotBinData { sections })
}

/// Render FEFF-compatible `apot.bin` TXT section-stream text.
pub fn apot_bin_string(data: &ApotBinData) -> Result<String> {
    validate_apot_bin(&data.sections)?;
    let mut out = String::new();
    for section in &data.sections {
        writeln!(out, "#SN#   Section: {:4}", section.section_number)?;
        writeln!(out, "#DF# This section written in TXT .")?;
        writeln!(out, "#H#")?;
        match &section.payload {
            ApotBinPayload::HeadersOnly => {}
            ApotBinPayload::Records(records) => write_records_section(&mut out, section, records)?,
            ApotBinPayload::Matrix(matrix) => write_matrix_section(&mut out, section, matrix)?,
        }
        write_headers(
            &mut out,
            &section.trailing_headers,
            &section.trailing_header_texts,
        )?;
    }
    Ok(out)
}

/// Read FEFF `apot.bin` from a file.
pub fn read_apot_bin(path: impl AsRef<Path>) -> Result<ApotBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_apot_bin(&text)
}

/// Write FEFF `apot.bin` TXT section-stream text to a file.
pub fn write_apot_bin(path: impl AsRef<Path>, data: &ApotBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, apot_bin_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_section(lines: &[(usize, &str)]) -> Result<ApotBinSection> {
    let Some((line_index, first)) = lines.first() else {
        return invalid_apot_bin(0, "empty apot.bin section");
    };
    let section_number = parse_section_number(*line_index + 1, first.trim())?;
    let mut descriptor = None;
    let mut headers = Vec::new();
    let mut header_texts = Vec::new();
    let mut trailing_headers = Vec::new();
    let mut trailing_header_texts = Vec::new();
    let mut column_labels = Vec::new();
    let mut column_label_text = None;
    let mut data_lines = Vec::new();
    let mut seen_data = false;

    for (index, raw) in &lines[1..] {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with("#DF#") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#DT#") {
            descriptor = Some(parse_descriptor(line_number, rest.trim())?);
            continue;
        }
        if let Some(rest) = line.strip_prefix("#CL#") {
            let raw_label_text = if let Some(raw_rest) = raw.strip_prefix("#CL#") {
                raw_rest
            } else {
                rest
            };
            column_label_text = Some(raw_label_text.to_string());
            column_labels = rest
                .split_whitespace()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            continue;
        }
        if let Some(rest) = line.strip_prefix("#HD#") {
            let data = rest.trim();
            if !data.is_empty() {
                seen_data = true;
                data_lines.push((line_number, data.to_string()));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("#H#") {
            let raw_header_text = if let Some(raw_rest) = raw.strip_prefix("#H#") {
                raw_rest
            } else {
                rest
            };
            let header = rest.trim();
            if is_boilerplate_header(header) {
                continue;
            }
            if seen_data {
                trailing_headers.push(header.to_string());
                trailing_header_texts.push(raw_header_text.to_string());
            } else {
                headers.push(header.to_string());
                header_texts.push(raw_header_text.to_string());
            }
            continue;
        }
        if line.starts_with('#') {
            return invalid_apot_bin(line_number, format!("unexpected marker {line:?}"));
        }

        seen_data = true;
        data_lines.push((line_number, line.to_string()));
    }

    let payload = match descriptor {
        Some(ApotBinDescriptor::Records(column_types)) => {
            ApotBinPayload::Records(parse_records(column_types, &data_lines)?)
        }
        Some(ApotBinDescriptor::Matrix {
            value_type,
            rows,
            columns,
        }) => ApotBinPayload::Matrix(parse_matrix(value_type, rows, columns, &data_lines)?),
        None => {
            if !data_lines.is_empty() {
                return invalid_apot_bin(
                    data_lines[0].0,
                    "data rows appeared before an apot.bin #DT# line",
                );
            }
            ApotBinPayload::HeadersOnly
        }
    };

    Ok(ApotBinSection {
        section_number,
        headers,
        header_texts,
        column_labels,
        column_label_text,
        payload,
        trailing_headers,
        trailing_header_texts,
    })
}

fn parse_section_number(line_number: usize, line: &str) -> Result<usize> {
    let Some(number) = line.split_whitespace().last() else {
        return invalid_apot_bin(line_number, "missing section number");
    };
    parse_usize(line_number, "section number", number)
}

fn parse_descriptor(line_number: usize, descriptor: &str) -> Result<ApotBinDescriptor> {
    if descriptor
        .split_whitespace()
        .next()
        .is_some_and(|token| token.eq_ignore_ascii_case("2d"))
    {
        parse_matrix_descriptor(line_number, descriptor)
    } else {
        let column_types = descriptor
            .split_whitespace()
            .map(|token| parse_type(line_number, token))
            .collect::<Result<Vec<_>>>()?;
        if column_types.is_empty() {
            return invalid_apot_bin(line_number, "record section is missing data types");
        }
        Ok(ApotBinDescriptor::Records(column_types))
    }
}

fn parse_matrix_descriptor(line_number: usize, descriptor: &str) -> Result<ApotBinDescriptor> {
    let tokens = descriptor.split_whitespace().collect::<Vec<_>>();
    let Some(first) = tokens.first() else {
        return invalid_apot_bin(line_number, "matrix descriptor is empty");
    };
    if !first.eq_ignore_ascii_case("2d") {
        return invalid_apot_bin(line_number, "matrix descriptor must start with 2D");
    }

    let (value_type, array_index) = parse_matrix_type(line_number, &tokens)?;
    if tokens
        .get(array_index)
        .is_none_or(|token| !token.eq_ignore_ascii_case("array"))
    {
        return invalid_apot_bin(line_number, "matrix descriptor is missing array keyword");
    }
    let sizes_index = tokens
        .iter()
        .position(|token| token.eq_ignore_ascii_case("sizes"))
        .ok_or_else(|| invalid_apot_bin_value(line_number, "matrix descriptor is missing sizes"))?;
    if sizes_index <= array_index {
        return invalid_apot_bin(line_number, "matrix descriptor has invalid keyword order");
    }

    let Some(sizes_offset) = descriptor.to_ascii_lowercase().find("sizes") else {
        return invalid_apot_bin(line_number, "matrix descriptor is missing sizes");
    };
    let shape_text = descriptor
        .get(sizes_offset + "sizes".len()..)
        .ok_or_else(|| invalid_apot_bin_value(line_number, "matrix shape is missing"))?;
    let (rows, columns) = parse_matrix_shape(line_number, shape_text)?;
    Ok(ApotBinDescriptor::Matrix {
        value_type,
        rows,
        columns,
    })
}

fn parse_matrix_type(line_number: usize, tokens: &[&str]) -> Result<(ApotBinType, usize)> {
    let Some(first_type) = tokens.get(1) else {
        return invalid_apot_bin(line_number, "matrix descriptor is missing value type");
    };
    if first_type.eq_ignore_ascii_case("double")
        && tokens
            .get(2)
            .is_some_and(|token| token.eq_ignore_ascii_case("complex"))
    {
        return Ok((ApotBinType::DComplex, 3));
    }
    Ok((parse_type(line_number, first_type)?, 2))
}

fn parse_matrix_shape(line_number: usize, shape_text: &str) -> Result<(usize, usize)> {
    let parts = shape_text.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        [rows, columns] => Ok((
            parse_usize(line_number, "matrix rows", rows)?,
            parse_usize(line_number, "matrix columns", columns)?,
        )),
        [_compact] => parse_fixed_width_shape(line_number, shape_text),
        _ => invalid_apot_bin(
            line_number,
            format!("matrix shape has {} token(s), expected 2", parts.len()),
        ),
    }
}

fn parse_fixed_width_shape(line_number: usize, shape_text: &str) -> Result<(usize, usize)> {
    let field = shape_text.trim_end();
    if field.len() < 8 {
        return invalid_apot_bin(
            line_number,
            "compact matrix shape is shorter than two I4 fields",
        );
    }
    let start = field.len() - 8;
    let first = field
        .get(start..start + 4)
        .ok_or_else(|| invalid_apot_bin_value(line_number, "invalid compact matrix row field"))?;
    let second = field.get(start + 4..start + 8).ok_or_else(|| {
        invalid_apot_bin_value(line_number, "invalid compact matrix column field")
    })?;
    Ok((
        parse_usize(line_number, "matrix rows", first.trim())?,
        parse_usize(line_number, "matrix columns", second.trim())?,
    ))
}

fn parse_records(
    column_types: Vec<ApotBinType>,
    data_lines: &[(usize, String)],
) -> Result<ApotBinRecords> {
    if data_lines.is_empty() {
        return invalid_apot_bin(0, "record section is missing data rows");
    }
    let mut rows = Vec::with_capacity(data_lines.len());
    for (line_number, line) in data_lines {
        rows.push(parse_record_row(*line_number, line, &column_types)?);
    }
    Ok(ApotBinRecords { column_types, rows })
}

fn parse_record_row(
    line_number: usize,
    line: &str,
    column_types: &[ApotBinType],
) -> Result<Vec<ApotBinValue>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let expected = column_types
        .iter()
        .map(|value_type| value_type.token_width())
        .sum::<usize>();
    if tokens.len() != expected {
        return invalid_apot_bin(
            line_number,
            format!(
                "record row has {} token(s), expected {expected}",
                tokens.len()
            ),
        );
    }

    let mut cursor = 0;
    let mut row = Vec::with_capacity(column_types.len());
    for value_type in column_types {
        row.push(parse_value(line_number, *value_type, &tokens, &mut cursor)?);
    }
    Ok(row)
}

fn parse_value(
    line_number: usize,
    value_type: ApotBinType,
    tokens: &[&str],
    cursor: &mut usize,
) -> Result<ApotBinValue> {
    match value_type {
        ApotBinType::Int => {
            let token = token_at(line_number, tokens, *cursor)?;
            *cursor += 1;
            Ok(ApotBinValue::Int(parse_i64(
                line_number,
                "integer value",
                token,
            )?))
        }
        ApotBinType::Real | ApotBinType::Double => {
            let token = token_at(line_number, tokens, *cursor)?;
            *cursor += 1;
            Ok(ApotBinValue::Real(parse_f64(
                line_number,
                "real value",
                token,
            )?))
        }
        ApotBinType::Complex | ApotBinType::DComplex => {
            let real = token_at(line_number, tokens, *cursor)?;
            let imag = token_at(line_number, tokens, *cursor + 1)?;
            *cursor += 2;
            Ok(ApotBinValue::Complex(Complex64::new(
                parse_f64(line_number, "complex real", real)?,
                parse_f64(line_number, "complex imaginary", imag)?,
            )))
        }
        ApotBinType::Text => {
            let token = token_at(line_number, tokens, *cursor)?;
            *cursor += 1;
            Ok(ApotBinValue::Text(token.to_string()))
        }
    }
}

fn parse_matrix(
    value_type: ApotBinType,
    rows: usize,
    columns: usize,
    data_lines: &[(usize, String)],
) -> Result<ApotBinMatrix> {
    if rows == 0 || columns == 0 {
        return invalid_apot_bin(0, "matrix shape must be non-empty");
    }
    if data_lines.len() != rows {
        return invalid_apot_bin(
            data_lines
                .first()
                .map_or(0, |(line_number, _)| *line_number),
            format!("matrix has {} row(s), expected {rows}", data_lines.len()),
        );
    }

    let values = match value_type {
        ApotBinType::Int => {
            let mut values = Vec::with_capacity(checked_product(rows, columns)?);
            for (line_number, line) in data_lines {
                values.extend(parse_i64_matrix_row(*line_number, line, columns)?);
            }
            ApotBinMatrixValues::Int(Array2::from_shape_vec((rows, columns), values).map_err(
                |source| {
                    invalid_apot_bin_value(0, format!("invalid integer matrix shape: {source}"))
                },
            )?)
        }
        ApotBinType::Real | ApotBinType::Double => {
            let mut values = Vec::with_capacity(checked_product(rows, columns)?);
            for (line_number, line) in data_lines {
                values.extend(parse_f64_matrix_row(*line_number, line, columns)?);
            }
            ApotBinMatrixValues::Real(Array2::from_shape_vec((rows, columns), values).map_err(
                |source| invalid_apot_bin_value(0, format!("invalid real matrix shape: {source}")),
            )?)
        }
        ApotBinType::Complex | ApotBinType::DComplex => {
            let mut values = Vec::with_capacity(checked_product(rows, columns)?);
            for (line_number, line) in data_lines {
                values.extend(parse_complex_matrix_row(*line_number, line, columns)?);
            }
            ApotBinMatrixValues::Complex(Array2::from_shape_vec((rows, columns), values).map_err(
                |source| {
                    invalid_apot_bin_value(0, format!("invalid complex matrix shape: {source}"))
                },
            )?)
        }
        ApotBinType::Text => {
            let mut values = Vec::with_capacity(checked_product(rows, columns)?);
            for (line_number, line) in data_lines {
                values.extend(parse_text_matrix_row(*line_number, line, columns)?);
            }
            ApotBinMatrixValues::Text(Array2::from_shape_vec((rows, columns), values).map_err(
                |source| invalid_apot_bin_value(0, format!("invalid text matrix shape: {source}")),
            )?)
        }
    };

    Ok(ApotBinMatrix { value_type, values })
}

fn parse_i64_matrix_row(line_number: usize, line: &str, columns: usize) -> Result<Vec<i64>> {
    parse_matrix_tokens(line_number, line, columns, |token| {
        parse_i64(line_number, "integer matrix", token)
    })
}

fn parse_f64_matrix_row(line_number: usize, line: &str, columns: usize) -> Result<Vec<f64>> {
    parse_matrix_tokens(line_number, line, columns, |token| {
        parse_f64(line_number, "real matrix", token)
    })
}

fn parse_text_matrix_row(line_number: usize, line: &str, columns: usize) -> Result<Vec<String>> {
    parse_matrix_tokens(line_number, line, columns, |token| Ok(token.to_string()))
}

fn parse_matrix_tokens<T>(
    line_number: usize,
    line: &str,
    columns: usize,
    parse: impl Fn(&str) -> Result<T>,
) -> Result<Vec<T>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != columns {
        return invalid_apot_bin(
            line_number,
            format!(
                "matrix row has {} token(s), expected {columns}",
                tokens.len()
            ),
        );
    }
    tokens.into_iter().map(parse).collect()
}

fn parse_complex_matrix_row(
    line_number: usize,
    line: &str,
    columns: usize,
) -> Result<Vec<Complex64>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let expected = checked_product(columns, 2)?;
    if tokens.len() != expected {
        return invalid_apot_bin(
            line_number,
            format!(
                "complex matrix row has {} token(s), expected {expected}",
                tokens.len()
            ),
        );
    }

    let mut values = Vec::with_capacity(columns);
    for pair in tokens.chunks(2) {
        let Some(real) = pair.first() else {
            return invalid_apot_bin(line_number, "missing complex real value");
        };
        let Some(imag) = pair.get(1) else {
            return invalid_apot_bin(line_number, "missing complex imaginary value");
        };
        values.push(Complex64::new(
            parse_f64(line_number, "complex matrix real", real)?,
            parse_f64(line_number, "complex matrix imaginary", imag)?,
        ));
    }
    Ok(values)
}

fn parse_type(line_number: usize, token: &str) -> Result<ApotBinType> {
    match token.to_ascii_lowercase().as_str() {
        "int" | "integer" => Ok(ApotBinType::Int),
        "real" => Ok(ApotBinType::Real),
        "double" => Ok(ApotBinType::Double),
        "complex" => Ok(ApotBinType::Complex),
        "dcomplex" => Ok(ApotBinType::DComplex),
        "string" | "character" | "text" => Ok(ApotBinType::Text),
        _ => invalid_apot_bin(line_number, format!("unknown data type {token:?}")),
    }
}

fn write_records_section(
    out: &mut String,
    section: &ApotBinSection,
    records: &ApotBinRecords,
) -> Result<()> {
    writeln!(
        out,
        "#H# The following data types are written in this section."
    )?;
    write!(out, "#DT#  ")?;
    for (index, value_type) in records.column_types.iter().enumerate() {
        if index > 0 {
            write!(out, " ")?;
        }
        write!(out, "{}", value_type.record_name())?;
    }
    writeln!(out)?;
    write_headers(out, &section.headers, &section.header_texts)?;
    write_column_labels(
        out,
        &section.column_labels,
        section.column_label_text.as_deref(),
    )?;
    for row in &records.rows {
        write_record_row(out, &records.column_types, row)?;
    }
    Ok(())
}

fn write_matrix_section(
    out: &mut String,
    section: &ApotBinSection,
    matrix: &ApotBinMatrix,
) -> Result<()> {
    let (rows, columns) = matrix.shape();
    writeln!(
        out,
        "#DT# 2D {} array with sizes {:4}{:4}",
        matrix.value_type.matrix_name(),
        rows,
        columns
    )?;
    writeln!(
        out,
        "#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . ."
    )?;
    writeln!(out, "#H#                                Array(2,i)")?;
    writeln!(out, "#H#                                     .")?;
    writeln!(out, "#H#                                     .")?;
    writeln!(out, "#H#                                     .")?;
    write_headers(out, &section.headers, &section.header_texts)?;
    write_column_labels(
        out,
        &section.column_labels,
        section.column_label_text.as_deref(),
    )?;
    write_matrix_rows(out, matrix)?;
    Ok(())
}

fn write_header(out: &mut String, header: &str) -> Result<()> {
    if header.is_empty() {
        writeln!(out, "#H#")?;
    } else {
        writeln!(out, "#H# {header}")?;
    }
    Ok(())
}

fn write_headers(out: &mut String, headers: &[String], header_texts: &[String]) -> Result<()> {
    if header_texts.len() == headers.len() {
        for header_text in header_texts {
            writeln!(out, "#H#{header_text}")?;
        }
        return Ok(());
    }
    for header in headers {
        write_header(out, header)?;
    }
    Ok(())
}

fn write_column_labels(
    out: &mut String,
    labels: &[String],
    label_text: Option<&str>,
) -> Result<()> {
    if let Some(label_text) = label_text {
        writeln!(out, "#CL#{label_text}")?;
        return Ok(());
    }
    if labels.is_empty() {
        return Ok(());
    }
    write!(out, "#CL#")?;
    for label in labels {
        write!(out, " {label}")?;
    }
    writeln!(out)?;
    Ok(())
}

fn write_record_row(
    out: &mut String,
    column_types: &[ApotBinType],
    row: &[ApotBinValue],
) -> Result<()> {
    if column_types.len() != row.len() {
        return invalid_apot_bin(
            0,
            format!(
                "record row has {} value(s), expected {}",
                row.len(),
                column_types.len()
            ),
        );
    }
    for (value_type, value) in column_types.iter().zip(row) {
        write_record_value(out, *value_type, value)?;
    }
    writeln!(out)?;
    Ok(())
}

fn write_record_value(
    out: &mut String,
    value_type: ApotBinType,
    value: &ApotBinValue,
) -> Result<()> {
    match (value_type, value) {
        (ApotBinType::Int, ApotBinValue::Int(value)) => write!(out, "{value:10} ")?,
        (value_type, ApotBinValue::Real(value)) if value_type.is_real() => {
            write_fortran_zero_scaled_exp(out, *value, 20, 10)?;
            write!(out, " ")?
        }
        (value_type, ApotBinValue::Complex(value)) if value_type.is_complex() => {
            write_fortran_zero_scaled_exp(out, value.re, 20, 10)?;
            write!(out, " ")?;
            write_fortran_zero_scaled_exp(out, value.im, 20, 10)?;
            write!(out, " ")?
        }
        (ApotBinType::Text, ApotBinValue::Text(value)) => write!(out, "{value} ")?,
        _ => {
            return invalid_apot_bin(
                0,
                format!(
                    "record value does not match declared type {}",
                    value_type.record_name()
                ),
            );
        }
    }
    Ok(())
}

fn write_matrix_rows(out: &mut String, matrix: &ApotBinMatrix) -> Result<()> {
    match (matrix.value_type, &matrix.values) {
        (ApotBinType::Int, ApotBinMatrixValues::Int(values)) => {
            for row in values.axis_iter(Axis(0)) {
                for value in row {
                    write!(out, "{value:10}")?;
                }
                writeln!(out)?;
            }
        }
        (value_type, ApotBinMatrixValues::Real(values)) if value_type.is_real() => {
            for row in values.axis_iter(Axis(0)) {
                for value in row {
                    write_fortran_zero_scaled_exp(out, *value, 20, 10)?;
                }
                writeln!(out)?;
            }
        }
        (value_type, ApotBinMatrixValues::Complex(values)) if value_type.is_complex() => {
            for row in values.axis_iter(Axis(0)) {
                for value in row {
                    write_fortran_zero_scaled_exp(out, value.re, 20, 10)?;
                    write_fortran_zero_scaled_exp(out, value.im, 20, 10)?;
                }
                writeln!(out)?;
            }
        }
        (ApotBinType::Text, ApotBinMatrixValues::Text(values)) => {
            for row in values.axis_iter(Axis(0)) {
                for value in row {
                    write!(out, "{value} ")?;
                }
                writeln!(out)?;
            }
        }
        _ => {
            return invalid_apot_bin(
                0,
                format!(
                    "matrix values do not match declared type {}",
                    matrix.value_type.record_name()
                ),
            );
        }
    }
    Ok(())
}

fn validate_apot_bin(sections: &[ApotBinSection]) -> Result<()> {
    if sections.is_empty() {
        return invalid_apot_bin(0, "at least one apot.bin section is required");
    }
    for section in sections {
        if section.section_number == 0 {
            return invalid_apot_bin(0, "section number must be positive");
        }
        match &section.payload {
            ApotBinPayload::HeadersOnly => {}
            ApotBinPayload::Records(records) => validate_records(records)?,
            ApotBinPayload::Matrix(matrix) => validate_matrix(matrix)?,
        }
    }
    Ok(())
}

fn validate_records(records: &ApotBinRecords) -> Result<()> {
    if records.column_types.is_empty() {
        return invalid_apot_bin(0, "record section must declare at least one column type");
    }
    if records.rows.is_empty() {
        return invalid_apot_bin(0, "record section must contain at least one row");
    }
    for row in &records.rows {
        if row.len() != records.column_types.len() {
            return invalid_apot_bin(
                0,
                format!(
                    "record row has {} value(s), expected {}",
                    row.len(),
                    records.column_types.len()
                ),
            );
        }
        for (value_type, value) in records.column_types.iter().zip(row) {
            validate_value(*value_type, value)?;
        }
    }
    Ok(())
}

fn validate_value(value_type: ApotBinType, value: &ApotBinValue) -> Result<()> {
    match (value_type, value) {
        (ApotBinType::Int, ApotBinValue::Int(_)) | (ApotBinType::Text, ApotBinValue::Text(_)) => {
            Ok(())
        }
        (value_type, ApotBinValue::Real(value)) if value_type.is_real() => {
            validate_finite(0, "real value", *value)
        }
        (value_type, ApotBinValue::Complex(value)) if value_type.is_complex() => {
            validate_finite(0, "complex real", value.re)?;
            validate_finite(0, "complex imaginary", value.im)
        }
        _ => invalid_apot_bin(
            0,
            format!(
                "record value does not match declared type {}",
                value_type.record_name()
            ),
        ),
    }
}

fn validate_matrix(matrix: &ApotBinMatrix) -> Result<()> {
    let (rows, columns) = matrix.shape();
    if rows == 0 || columns == 0 {
        return invalid_apot_bin(0, "matrix shape must be non-empty");
    }
    match (matrix.value_type, &matrix.values) {
        (ApotBinType::Int, ApotBinMatrixValues::Int(_))
        | (ApotBinType::Text, ApotBinMatrixValues::Text(_)) => Ok(()),
        (value_type, ApotBinMatrixValues::Real(values)) if value_type.is_real() => {
            for value in values {
                validate_finite(0, "real matrix", *value)?;
            }
            Ok(())
        }
        (value_type, ApotBinMatrixValues::Complex(values)) if value_type.is_complex() => {
            for value in values {
                validate_finite(0, "complex matrix real", value.re)?;
                validate_finite(0, "complex matrix imaginary", value.im)?;
            }
            Ok(())
        }
        _ => invalid_apot_bin(
            0,
            format!(
                "matrix values do not match declared type {}",
                matrix.value_type.record_name()
            ),
        ),
    }
}

fn is_boilerplate_header(header: &str) -> bool {
    header.is_empty()
        || header == "The following data types are written in this section."
        || header.starts_with("File is organized as follows:")
        || header.starts_with("Array(2,i)")
        || header == "."
}

fn token_at<'a>(line_number: usize, tokens: &'a [&str], index: usize) -> Result<&'a str> {
    tokens
        .get(index)
        .copied()
        .ok_or_else(|| invalid_apot_bin_value(line_number, "missing data token"))
}

fn parse_i64(line: usize, field: &'static str, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| IoError::ApotBinParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::ApotBinParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value =
        token
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .map_err(|_| IoError::ApotBinParse {
                field,
                line,
                token: token.to_string(),
            })?;
    validate_finite(line, field, value)?;
    Ok(value)
}

fn validate_finite(line: usize, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_apot_bin(line, format!("{field} must be finite"))
    }
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_apot_bin_value(0, "matrix shape overflows usize"))
}

fn invalid_apot_bin<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(invalid_apot_bin_value(line, message))
}

fn invalid_apot_bin_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::InvalidApotBin {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_records_and_matrix_sections() -> Result<()> {
        let data = parse_apot_bin(APOT_BIN)?;
        assert_eq!(data.section_count(), 3);
        assert_eq!(data.matrix_count(), 1);

        let first = &data.sections[0];
        assert_eq!(first.section_number, 1);
        assert_eq!(first.column_labels, ["nph", "nat", "ihole", "s02"]);
        let Some(records) = first.records() else {
            return invalid_apot_bin(0, "first section should contain records");
        };
        assert_eq!(records.row_count(), 1);
        assert_eq!(records.column_count(), 4);
        assert_eq!(records.rows[0][0], ApotBinValue::Int(1));
        assert_eq!(records.rows[0][3], ApotBinValue::Real(0.95));

        let Some(matrix) = data.sections[2].matrix() else {
            return invalid_apot_bin(0, "third section should contain a matrix");
        };
        assert_eq!(matrix.shape(), (2, 3));
        match &matrix.values {
            ApotBinMatrixValues::Real(values) => assert_eq!(values[[1, 2]], 6.0),
            _ => return invalid_apot_bin(0, "third section should contain real matrix values"),
        }
        assert_eq!(data.sections[2].trailing_headers, ["next block"]);

        Ok(())
    }

    #[test]
    fn parses_compact_i4_shape_fields() -> Result<()> {
        assert_eq!(parse_matrix_shape(1, "   34000")?, (3, 4000));
        Ok(())
    }

    #[test]
    fn roundtrips_apot_bin_data() -> Result<()> {
        let data = parse_apot_bin(APOT_BIN)?;
        let rendered = apot_bin_string(&data)?;
        let reparsed = parse_apot_bin(&rendered)?;
        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_apot_bin_data() {
        assert!(parse_apot_bin("").is_err());
        assert!(parse_apot_bin("1 2 3\n").is_err());
        assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\n").is_err());
        assert!(
            parse_apot_bin("#SN#   Section:    1\n#DT# 2D double array with sizes    2   2\n1\n")
                .is_err()
        );
        assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\nNaN\n").is_err());
    }

    const APOT_BIN: &str = r#"#SN#   Section:    1
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Int Int Double
#H# first section
#CL# nph nat ihole s02
         1         79          1     0.9500000000E+00
#SN#   Section:    2
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Double
        29     0.2838535628D+01
        30     0.2632330371E+01
#SN#   Section:    3
#DF# This section written in TXT .
#H#
#DT# 2D double array with sizes    2   3
#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . .
#H#                                Array(2,i)
#H#                                     .
#H#                                     .
#H#                                     .
#H# matrix
    1.0000000000E+00    2.0000000000E+00    3.0000000000E+00
    4.0000000000E+00    5.0000000000E+00    6.0000000000E+00
#H# next block
"#;
}
