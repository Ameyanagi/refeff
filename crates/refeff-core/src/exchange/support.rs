use super::*;

pub(super) fn ensure_finite(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ExchangeError::NonFiniteInput { name, value })
    }
}

pub(super) fn ensure_positive(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ExchangeError::NonPositiveInput { name, value })
    }
}

pub(super) fn ensure_nonnegative(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    ensure_finite(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ExchangeError::NegativeInput { name, value })
    }
}
