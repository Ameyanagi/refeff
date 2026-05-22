//! Shared DMDW output parsing helpers.

use crate::error::{IoError, Result};

pub(in crate::dmdw_out) const DMDW_OUT_PATH: &str = "dmdw.out";

pub(in crate::dmdw_out) fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(line, message))
}

pub(in crate::dmdw_out) fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: DMDW_OUT_PATH.into(),
        line,
        message: message.into(),
    }
}
