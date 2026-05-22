//! Shared helpers for FEFF `apot.bin` parsing and validation.

use crate::error::{IoError, Result};

pub(super) fn token_at<'a>(
    line_number: usize,
    tokens: &'a [&str],
    index: usize,
) -> Result<&'a str> {
    tokens
        .get(index)
        .copied()
        .ok_or_else(|| invalid_apot_bin_value(line_number, "missing data token"))
}

pub(super) fn parse_i64(line: usize, field: &'static str, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| IoError::ApotBinParse {
        field,
        line,
        token: token.to_string(),
    })
}

pub(super) fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::ApotBinParse {
        field,
        line,
        token: token.to_string(),
    })
}

pub(super) fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
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

pub(super) fn validate_finite(line: usize, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_apot_bin(line, format!("{field} must be finite"))
    }
}

pub(super) fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_apot_bin_value(0, "matrix shape overflows usize"))
}

pub(super) fn invalid_apot_bin<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(invalid_apot_bin_value(line, message))
}

pub(super) fn invalid_apot_bin_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::InvalidApotBin {
        line,
        message: message.into(),
    }
}
