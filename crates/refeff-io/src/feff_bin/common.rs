use crate::error::{IoError, Result};

pub(super) fn validate_label(label: &str) -> Result<()> {
    if !label.is_empty() && label.len() <= 6 && label.is_ascii() {
        Ok(())
    } else {
        Err(invalid_feff_bin(
            "potlbl",
            format!("label {label:?} must be non-empty and at most 6 ASCII bytes"),
        ))
    }
}

pub(super) fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_feff_bin(field, "array element count overflowed"))
}

pub(super) fn parse_i64(field: &'static str, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| IoError::FeffBinParse {
        field,
        token: token.to_string(),
    })
}

pub(super) fn parse_f64(field: &'static str, token: &str) -> Result<f64> {
    let value = token.parse::<f64>().map_err(|_| IoError::FeffBinParse {
        field,
        token: token.to_string(),
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_feff_bin(
            field,
            format!("value must be finite, got {value}"),
        ))
    }
}

pub(super) fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in i64")))
}

pub(super) fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_feff_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in usize")))
}

pub(super) fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_feff_bin(field, format!("value {value} does not fit in i32")))
}

pub(super) fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_feff_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_feff_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn invalid_feff_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidFeffBin {
        field,
        message: message.into(),
    }
}
