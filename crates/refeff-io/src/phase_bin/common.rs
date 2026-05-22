use crate::error::{IoError, Result};

pub(super) fn validate_label(label: &str) -> Result<()> {
    if label.len() <= 6 && label.is_ascii() {
        Ok(())
    } else {
        Err(invalid_phase_bin(
            "potlbl",
            format!("label {label:?} must be at most 6 ASCII bytes"),
        ))
    }
}

pub(super) fn checked_l_count(lmax: usize) -> Result<usize> {
    lmax.checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_phase_bin("lmax", "l channel count overflowed"))
}

pub(super) fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_phase_bin(field, "array element count overflowed"))
}

pub(super) fn checked_count3(
    field: &'static str,
    rows: usize,
    cols: usize,
    planes: usize,
) -> Result<usize> {
    checked_count2(field, rows, cols)?
        .checked_mul(planes)
        .ok_or_else(|| invalid_phase_bin(field, "array element count overflowed"))
}

pub(super) fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in i64")))
}

pub(super) fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_phase_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in usize")))
}

pub(super) fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_phase_bin(field, format!("value {value} does not fit in i32")))
}

pub(super) fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_phase_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_phase_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn invalid_phase_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPhaseBin {
        field,
        message: message.into(),
    }
}
