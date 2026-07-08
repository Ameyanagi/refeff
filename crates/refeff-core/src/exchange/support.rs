use super::*;

pub(super) fn ensure_finite(name: &'static str, value: Real) -> Result<(), ExchangeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ExchangeError::NonFiniteInput { name, value })
    }
}

pub(super) fn ensure_finite_complex(
    name: &'static str,
    value: Complex,
) -> Result<(), ExchangeError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ExchangeError::NonFiniteComplex { name, value })
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
