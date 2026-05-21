use crate::error::{IoError, Result};
use crate::input::{FeffLine, LineKind};

pub(super) fn card_args(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::Card { args, .. } => Ok(args),
        LineKind::SectionData { .. } => Err(parse_error(line, "expected card line")),
    }
}

pub(super) fn section_fields(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::SectionData { fields, .. } => Ok(fields),
        LineKind::Card { .. } => Err(parse_error(line, "expected section data line")),
    }
}

pub(super) fn section_fields_before_star(line: &FeffLine) -> Result<Vec<&String>> {
    Ok(section_fields(line)?
        .iter()
        .take_while(|field| field.as_str() != "*")
        .collect())
}

pub(super) fn parse_i32(line: &FeffLine, value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .map_err(|_| parse_error(line, format!("invalid integer {value:?}")))
}

pub(super) fn parse_optional_i32(line: &FeffLine, value: Option<&String>) -> Result<Option<i32>> {
    value.map(|value| parse_i32(line, value)).transpose()
}

pub(super) fn parse_logical(line: &FeffLine, value: &str) -> Result<bool> {
    let normalized = value.trim_matches('.').to_ascii_uppercase();
    match normalized.as_str() {
        "T" | "TRUE" | "1" => Ok(true),
        "F" | "FALSE" | "0" => Ok(false),
        _ => Err(parse_error(line, format!("invalid logical {value:?}"))),
    }
}

pub(super) fn parse_f64(line: &FeffLine, value: &str) -> Result<f64> {
    value
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error(line, format!("invalid float {value:?}")))
}

pub(super) fn parse_optional_f64(line: &FeffLine, value: Option<&String>) -> Result<Option<f64>> {
    value.map(|value| parse_f64(line, value)).transpose()
}

pub(super) fn parse_error(line: &FeffLine, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: line.location.path.clone(),
        line: line.location.line,
        message: message.into(),
    }
}
