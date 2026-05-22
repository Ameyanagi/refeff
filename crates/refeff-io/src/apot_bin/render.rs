//! Renderer and file I/O for FEFF `apot.bin` TXT section streams.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Axis;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

use super::common::invalid_apot_bin;
use super::parse::parse_apot_bin;
use super::types::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords,
    ApotBinSection, ApotBinType, ApotBinValue,
};
use super::validate::validate_apot_bin;

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
