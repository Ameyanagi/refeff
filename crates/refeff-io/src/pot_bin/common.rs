use ndarray::{Array2, Array3, ShapeBuilder};

use crate::error::{IoError, Result};
use crate::pad::decode_f64;

pub(super) fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual],
            expected: vec![expected],
        })
    }
}

pub(super) fn validate_shape2(
    field: &'static str,
    actual: (usize, usize),
    expected: (usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual.0, actual.1],
            expected: vec![expected.0, expected.1],
        })
    }
}

pub(super) fn validate_shape3(
    field: &'static str,
    actual: (usize, usize, usize),
    expected: (usize, usize, usize),
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::PotBinShape {
            field,
            actual: vec![actual.0, actual.1, actual.2],
            expected: vec![expected.0, expected.1, expected.2],
        })
    }
}

pub(super) fn validate_finite_values(
    field: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(invalid_pot_bin(
                field,
                format!("value must be finite, got {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn array2_from_fortran(
    field: &'static str,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>> {
    Array2::from_shape_vec((rows, cols).f(), values).map_err(|_| IoError::PotBinShape {
        field,
        actual: vec![rows, cols],
        expected: vec![rows, cols],
    })
}

pub(super) fn array3_from_fortran(
    field: &'static str,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    planes: usize,
) -> Result<Array3<f64>> {
    Array3::from_shape_vec((rows, cols, planes).f(), values).map_err(|_| IoError::PotBinShape {
        field,
        actual: vec![rows, cols, planes],
        expected: vec![rows, cols, planes],
    })
}

pub(super) fn decode_pad_line(
    field: &'static str,
    line: &str,
    pad_width: usize,
) -> Result<Vec<f64>> {
    if pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(pad_width));
    }
    let trimmed = line.trim_start().trim_end();
    let Some(found) = trimmed.chars().next() else {
        return Err(IoError::PotBinMissing { field });
    };
    if found != '!' {
        return Err(IoError::PadMarker {
            expected: '!',
            found,
        });
    }
    let payload = &trimmed[found.len_utf8()..];
    if payload.is_empty() || !payload.len().is_multiple_of(pad_width) {
        return Err(IoError::PadPayload {
            payload_len: payload.len(),
            unit_len: pad_width,
        });
    }

    let mut values = Vec::with_capacity(payload.len() / pad_width);
    for chunk in payload.as_bytes().chunks(pad_width) {
        let chunk =
            std::str::from_utf8(chunk).map_err(|source| IoError::PadChunkUtf8 { source })?;
        values.push(decode_f64(chunk, pad_width)?);
    }
    Ok(values)
}

pub(super) fn checked_count2(field: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_pot_bin(field, "array element count overflowed"))
}

pub(super) fn checked_count3(
    field: &'static str,
    rows: usize,
    cols: usize,
    planes: usize,
) -> Result<usize> {
    checked_count2(field, rows, cols)?
        .checked_mul(planes)
        .ok_or_else(|| invalid_pot_bin(field, "array element count overflowed"))
}

pub(super) fn i64_from_usize(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in i64")))
}

pub(super) fn usize_from_i64(value: i64, field: &'static str) -> Result<usize> {
    if value < 0 {
        return Err(invalid_pot_bin(
            field,
            format!("value {value} must be non-negative"),
        ));
    }
    usize::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in usize")))
}

pub(super) fn i32_from_i64(value: i64, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| invalid_pot_bin(field, format!("value {value} does not fit in i32")))
}

pub(super) fn check_i4(value: i64, field: &'static str) -> Result<()> {
    check_fixed_int(value, 4, field)
}

pub(super) fn check_i2(value: i64, field: &'static str) -> Result<()> {
    check_fixed_int(value, 2, field)
}

pub(super) fn check_fixed_int(value: i64, width: usize, field: &'static str) -> Result<()> {
    let limit = 10_i64.pow(u32::try_from(width).map_err(|_| {
        invalid_pot_bin(field, format!("integer field width {width} is too large"))
    })?);
    let negative_limit = -(limit / 10 - 1);
    if value < negative_limit || value >= limit {
        Err(invalid_pot_bin(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn invalid_pot_bin(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPotBin {
        field,
        message: message.into(),
    }
}
