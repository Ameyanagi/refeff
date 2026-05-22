use crate::error::{IoError, Result};

pub(super) fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_specfunct_dat_value("byte offset overflows usize"))
}

pub(super) fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_specfunct_dat_value("record length overflows usize"))
}

pub(super) fn invalid_specfunct_dat<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_specfunct_dat_value(message))
}

pub(super) fn invalid_specfunct_dat_value(message: impl Into<String>) -> IoError {
    IoError::InvalidSpecfunctDat {
        message: message.into(),
    }
}
