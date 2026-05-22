use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

pub(super) const DMDW_EGRID_INFO_PATH: &str = "dmdw_Egrid.info";
pub(super) const DMDW_A2F_INFO_PATH: &str = "dmdw_a2f.info";
pub(super) const DMDW_SPECTRAL_INFO_PATH: &str = "dmdw_spectral.info";
pub(super) const DMDW_SELF_ENERGY_DAT_PATH: &str = "dmdw_*SE_a2F.dat";
pub(super) const DMDW_AKW_DAT_PATH: &str = "dmdw_Akw.dat";
pub(super) const DMDW_A2F_INFO_ROW_WIDTH: usize = 2;
pub(super) const DMDW_SELF_ENERGY_ROW_WIDTH: usize = 2;
pub(super) const DMDW_AKW_ROW_WIDTH: usize = 5;

pub(super) fn numeric_values(path: &'static str, line: usize, text: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for token in text.split_whitespace() {
        if let Some(value) = parse_numeric_token(path, line, token) {
            values.push(value?);
        }
    }
    Ok(values)
}

pub(super) fn parse_single_value(
    path: &'static str,
    line: usize,
    field: &'static str,
    text: &str,
) -> Result<f64> {
    let values = numeric_values(path, line, text)?;
    if values.len() == 1 {
        Ok(values[0])
    } else {
        parse_error(
            path,
            line,
            format!(
                "{field} line has {} numeric value(s), expected 1",
                values.len()
            ),
        )
    }
}

pub(super) fn parse_complex_value(
    path: &'static str,
    line: usize,
    field: &'static str,
    text: &str,
) -> Result<Complex64> {
    let value_text = text
        .rsplit_once('=')
        .map(|(_, value)| value.trim())
        .unwrap_or_else(|| text.trim());
    let normalized = value_text
        .trim_start_matches('(')
        .trim_end_matches(')')
        .replace(',', " ");
    let values = normalized
        .split_whitespace()
        .map(|token| parse_f64(path, line, field, token))
        .collect::<Result<Vec<_>>>()?;

    match values.as_slice() {
        [real] => Ok(Complex64::new(*real, 0.0)),
        [real, imaginary] => Ok(Complex64::new(*real, *imaginary)),
        _ => parse_error(
            path,
            line,
            format!(
                "{field} line has {} numeric value(s), expected 1 or 2",
                values.len()
            ),
        ),
    }
}

pub(super) fn parse_i32_field(
    path: &'static str,
    line: usize,
    field: &'static str,
    text: &str,
) -> Result<i32> {
    let value = parse_single_value(path, line, field, text)?;
    if value.fract() != 0.0 {
        return parse_error(path, line, format!("{field} must be an integer"));
    }
    if value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        return parse_error(path, line, format!("{field} is out of range"));
    }
    Ok(value as i32)
}

pub(super) fn parse_two_column_row(
    path: &'static str,
    line: usize,
    text: &str,
    expected_width: usize,
    row_name: &'static str,
) -> Result<(f64, f64)> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != expected_width {
        return parse_error(
            path,
            line,
            format!(
                "{row_name} row has {} token(s), expected {expected_width}",
                tokens.len()
            ),
        );
    }
    Ok((
        parse_f64(path, line, "first column", tokens[0])?,
        parse_f64(path, line, "second column", tokens[1])?,
    ))
}

pub(super) fn write_fortran_complex_tuple(out: &mut String, value: Complex64) -> std::fmt::Result {
    out.push('(');
    write_fortran_zero_scaled_exp(out, value.re, 20, 10)?;
    out.push(',');
    write_fortran_zero_scaled_exp(out, value.im, 20, 10)?;
    out.push(')');
    Ok(())
}

pub(super) fn is_numeric_token(token: &str) -> bool {
    parse_numeric_token(DMDW_SELF_ENERGY_DAT_PATH, 0, token).is_some_and(|value| value.is_ok())
}

fn parse_numeric_token(path: &'static str, line: usize, token: &str) -> Option<Result<f64>> {
    let normalized = token.replace(['D', 'd'], "E");
    normalized
        .parse::<f64>()
        .map(|value| {
            if value.is_finite() {
                Ok(value)
            } else {
                parse_error(path, line, "numeric value must be finite")
            }
        })
        .ok()
}

pub(super) fn parse_f64(
    path: &'static str,
    line: usize,
    field: &'static str,
    token: &str,
) -> Result<f64> {
    let value = token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error_value(
            path,
            line,
            format!("{field} value must be finite"),
        ))
    }
}

pub(super) fn invalid_data<T>(
    path: &'static str,
    field: &'static str,
    message: impl Into<String>,
) -> Result<T> {
    Err(IoError::Parse {
        path: path.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

pub(super) fn parse_error<T>(
    path: &'static str,
    line: usize,
    message: impl Into<String>,
) -> Result<T> {
    Err(parse_error_value(path, line, message))
}

pub(super) fn parse_error_value(
    path: &'static str,
    line: usize,
    message: impl Into<String>,
) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}
