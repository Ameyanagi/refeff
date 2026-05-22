use super::*;

pub(super) fn validate_inputs(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<(), ConvolutionError> {
    if omega.len() != spectrum.len() {
        return Err(ConvolutionError::LengthMismatch {
            omega_len: omega.len(),
            spectrum_len: spectrum.len(),
        });
    }
    if omega.len() < 2 {
        return Err(ConvolutionError::InsufficientPoints {
            points: omega.len(),
        });
    }
    validate_width(width)?;
    for &energy in omega {
        validate_energy("omega", energy)?;
    }
    for &value in spectrum {
        validate_spectrum("spectrum", value)?;
    }
    Ok(())
}

pub(super) fn validate_width(width: Real) -> Result<(), ConvolutionError> {
    if !(width.is_finite() && width > 0.0) {
        return Err(ConvolutionError::InvalidWidth { width });
    }
    Ok(())
}

pub(super) fn validate_energy(name: &'static str, value: Real) -> Result<(), ConvolutionError> {
    if !value.is_finite() {
        return Err(ConvolutionError::NonFiniteEnergy { name, value });
    }
    Ok(())
}

pub(super) fn validate_excitation_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), ConvolutionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::ExcitationNonFiniteScalar { field, value })
    }
}

pub(super) fn validate_excitation_width(value: Real) -> Result<(), ConvolutionError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(ConvolutionError::ExcitationInvalidDistributionWidth { value })
    }
}

pub(super) fn validate_atan_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), ConvolutionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::AtanNonFiniteScalar { field, value })
    }
}

pub(super) fn validate_atan_complex(
    field: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), ConvolutionError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::AtanNonFiniteSpectrum {
            field,
            row,
            real: value.re,
            imaginary: value.im,
        })
    }
}

pub(super) fn validate_spectrum(
    name: &'static str,
    value: Complex,
) -> Result<(), ConvolutionError> {
    if !(value.re.is_finite() && value.im.is_finite()) {
        return Err(ConvolutionError::NonFiniteSpectrum {
            name,
            real: value.re,
            imaginary: value.im,
        });
    }
    Ok(())
}
