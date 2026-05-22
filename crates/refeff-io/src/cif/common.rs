use crate::{IoError, Result};

pub(super) const CIF_FRACTION_TOLERANCE: f64 = 0.0002;
pub(super) const CIF_POSITION_TOLERANCE: f64 = 1.0e-5;

pub(super) fn required_f64(value: Option<f64>, field: &str) -> Result<f64> {
    value.ok_or_else(|| invalid_cif(field, "missing required cell field"))
}

pub(super) fn strip_element_label(label: &str) -> String {
    label
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
}

pub(super) fn invalid_cif(field: &str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "cif".into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}
