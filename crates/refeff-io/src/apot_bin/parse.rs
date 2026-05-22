//! Parser for FEFF `apot.bin` TXT section streams.

use ndarray::Array2;
use num_complex::Complex64;

use crate::error::Result;

use super::common::{
    checked_product, invalid_apot_bin, invalid_apot_bin_value, parse_f64, parse_i64, parse_usize,
    token_at,
};
use super::types::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords,
    ApotBinSection, ApotBinType, ApotBinValue,
};
use super::validate::validate_apot_bin;

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

pub(super) fn parse_matrix_shape(line_number: usize, shape_text: &str) -> Result<(usize, usize)> {
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

fn is_boilerplate_header(header: &str) -> bool {
    header.is_empty()
        || header == "The following data types are written in this section."
        || header.starts_with("File is organized as follows:")
        || header.starts_with("Array(2,i)")
        || header == "."
}
