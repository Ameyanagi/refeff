//! Validation for structured FEFF `apot.bin` section streams.

use crate::error::Result;

use super::common::{invalid_apot_bin, validate_finite};
use super::types::{
    ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords, ApotBinSection,
    ApotBinType, ApotBinValue,
};

pub(super) fn validate_apot_bin(sections: &[ApotBinSection]) -> Result<()> {
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
