//! Debye and Einstein-model cumulant helpers ported from FEFF.
//!
//! This module starts with `DEBYE/sigm3.f90`, the correlated Einstein model
//! with a Morse potential used for first and third cumulant estimates.

use crate::Real;

const BOHR_ANGSTROM: Real = 0.529_177_249;

/// First and third cumulants from FEFF `sigm3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorseCumulants {
    /// FEFF `sig1`: first cumulant.
    pub first: Real,
    /// FEFF `sig3`: third cumulant.
    pub third: Real,
    /// FEFF mutates `alphat` from inverse angstrom to inverse bohr; Rust returns
    /// the scaled value explicitly.
    pub scaled_thermal_expansion: Real,
}

/// Error returned by Debye/Einstein cumulant helpers.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DebyeError {
    /// Inputs must be finite real values.
    #[error("Debye input {name} must be finite, got {value}")]
    NonFinite { name: &'static str, value: Real },
    /// Inputs used as scales must be strictly positive.
    #[error("Debye input {name} must be positive, got {value}")]
    NonPositive { name: &'static str, value: Real },
    /// A computed output became non-finite.
    #[error("Debye output {name} must be finite, got {value}")]
    NonFiniteOutput { name: &'static str, value: Real },
}

/// Port of FEFF `sigm3`: correlated Einstein-model Morse cumulants.
///
/// `mean_square_relative_displacement` is FEFF `sig2`, `temperature` is `tk`,
/// `thermal_expansion` is `alphat` in inverse angstrom, and
/// `einstein_temperature` is `thetae`. FEFF stores several intermediates as
/// single precision `real`; this port keeps those roundings to match the
/// reference values.
pub fn morse_einstein_cumulants(
    mean_square_relative_displacement: Real,
    temperature: Real,
    thermal_expansion: Real,
    einstein_temperature: Real,
) -> Result<MorseCumulants, DebyeError> {
    ensure_positive("sig2", mean_square_relative_displacement)?;
    ensure_positive("tk", temperature)?;
    ensure_finite("alphat", thermal_expansion)?;
    ensure_positive("thetae", einstein_temperature)?;

    let scaled_thermal_expansion = thermal_expansion * BOHR_ANGSTROM;
    let z = to_feff_real((-einstein_temperature / temperature).exp());
    let occupation_ratio = to_feff_real(((1.0_f32 - z as f32) / (1.0_f32 + z as f32)) as Real);
    let sig02 = to_feff_real(occupation_ratio * mean_square_relative_displacement);
    let sig01 = to_feff_real(scaled_thermal_expansion * sig02 * 0.75);
    let first = sig01 * mean_square_relative_displacement / sig02;
    let third = (2.0
        - ((4.0_f32 / 3.0_f32) as Real) * (sig02 / mean_square_relative_displacement).powi(2))
        * first
        * mean_square_relative_displacement;

    ensure_finite_output("sig1", first)?;
    ensure_finite_output("sig3", third)?;
    ensure_finite_output("alphat", scaled_thermal_expansion)?;

    Ok(MorseCumulants {
        first,
        third,
        scaled_thermal_expansion,
    })
}

fn to_feff_real(value: Real) -> Real {
    (value as f32) as Real
}

fn ensure_finite(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFinite { name, value })
    }
}

fn ensure_positive(name: &'static str, value: Real) -> Result<(), DebyeError> {
    ensure_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(DebyeError::NonPositive { name, value })
    }
}

fn ensure_finite_output(name: &'static str, value: Real) -> Result<(), DebyeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DebyeError::NonFiniteOutput { name, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morse_einstein_cumulants_match_feff_reference() -> Result<(), DebyeError> {
        let first = morse_einstein_cumulants(0.003, 300.0, 1.0e-5, 400.0)?;
        assert_close(first.first, 1.190_648_842_682_321_3e-8);
        assert_close(first.third, 5.526_344_214_607_83e-11);
        assert_close(first.scaled_thermal_expansion, 5.291_772_49e-6);

        let second = morse_einstein_cumulants(0.0075, 800.0, 2.5e-5, 250.0)?;
        assert_close(second.first, 7.441_554_786_684_262e-8);
        assert_close(second.third, 1.098_357_016_560_439_2e-9);
        assert_close(second.scaled_thermal_expansion, 1.322_943_122_5e-5);

        let negative_alpha = morse_einstein_cumulants(0.0012, 120.0, -7.0e-6, 350.0)?;
        assert_close(negative_alpha.first, -3.333_816_545_419_16e-9);
        assert_close(negative_alpha.third, -3.706_146_208_663_239e-12);
        assert_close(negative_alpha.scaled_thermal_expansion, -3.704_240_743e-6);
        Ok(())
    }

    #[test]
    fn morse_einstein_cumulants_reject_invalid_inputs() {
        assert!(matches!(
            morse_einstein_cumulants(0.0, 300.0, 1.0e-5, 400.0),
            Err(DebyeError::NonPositive { name: "sig2", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, Real::NAN, 1.0e-5, 400.0),
            Err(DebyeError::NonFinite { name: "tk", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, Real::INFINITY, 400.0),
            Err(DebyeError::NonFinite { name: "alphat", .. })
        ));
        assert!(matches!(
            morse_einstein_cumulants(0.003, 300.0, 1.0e-5, -1.0),
            Err(DebyeError::NonPositive { name: "thetae", .. })
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-18,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }
}
