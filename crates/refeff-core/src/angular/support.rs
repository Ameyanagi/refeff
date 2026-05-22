use super::*;

pub(super) fn usize_to_real(value: usize) -> Result<Real, AngularError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| AngularError::IndexTooLarge { value })
}

pub(super) fn usize_to_i32(value: usize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::IndexTooLarge { value })
}
