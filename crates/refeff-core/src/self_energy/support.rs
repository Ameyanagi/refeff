use super::*;

pub(super) fn ensure_finite_complex(
    name: &'static str,
    value: Complex,
) -> Result<(), SelfEnergyError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(SelfEnergyError::NonFiniteComplex { name, value })
    }
}

pub(super) fn ensure_finite_real(name: &'static str, value: Real) -> Result<(), SelfEnergyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SelfEnergyError::NonFiniteReal { name, value })
    }
}

pub(super) fn ensure_positive_real(name: &'static str, value: Real) -> Result<(), SelfEnergyError> {
    ensure_finite_real(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SelfEnergyError::NonPositiveReal { name, value })
    }
}

pub(super) fn ensure_nonnegative_real(
    name: &'static str,
    value: Real,
) -> Result<(), SelfEnergyError> {
    ensure_finite_real(name, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(SelfEnergyError::NegativeReal { name, value })
    }
}

pub(super) fn dp_name(index: usize) -> &'static str {
    match index {
        0 => "DPPar(1)",
        1 => "DPPar(2)",
        2 => "DPPar(3)",
        _ => "DPPar(4)",
    }
}
