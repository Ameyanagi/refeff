//! FEFF FULLSPECTRUM numerical helpers.
//!
//! This module covers small kernels from `FULLSPECTRUM/` that can be tested
//! independently of the full driver. Larger spectrum assembly remains in the
//! module runner layer until the surrounding FEFF state is ported.

use ndarray::ArrayView1;
use thiserror::Error;

use crate::Real;

/// Inputs for FEFF `FULLSPECTRUM/qsum.f90`.
#[derive(Debug, Clone, Copy)]
pub struct FullSpectrumQSumInput<'a> {
    /// Number density `numden` used in the oscillator-strength sum rule.
    pub number_density: Real,
    /// Imaginary dielectric function `eps2`.
    pub epsilon2: ArrayView1<'a, Real>,
    /// Energy grid `omega`.
    pub omega: ArrayView1<'a, Real>,
    /// Number of active rows, equivalent to FEFF `iepts`.
    pub active_len: usize,
}

/// Error returned by FULLSPECTRUM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FullSpectrumError {
    /// Number density must be positive.
    #[error("FULLSPECTRUM {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    /// Scalar inputs must be finite.
    #[error("FULLSPECTRUM {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// Active rows must fit in both input arrays.
    #[error("FULLSPECTRUM active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Array values must be finite.
    #[error("FULLSPECTRUM {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// Energy rows are expected in nondecreasing order for the trapezoid rule.
    #[error("FULLSPECTRUM omega row {row} must not decrease, got {current} after {previous}")]
    DecreasingOmega {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// The final sum-rule value must be finite.
    #[error("FULLSPECTRUM neff must be finite, got {value}")]
    NonFiniteResult { value: Real },
}

/// Port of `FULLSPECTRUM/qsum.f90`: compute the effective electron count.
///
/// FEFF applies a trapezoid integral to `omega * eps2`, then scales it by
/// `1 / (2*pi^2*numden)`. An active length of zero or one follows the Fortran
/// loop semantics and returns zero.
pub fn full_spectrum_effective_electron_count(
    input: FullSpectrumQSumInput<'_>,
) -> Result<Real, FullSpectrumError> {
    validate_positive("number_density", input.number_density)?;
    validate_active_len("epsilon2", input.active_len, input.epsilon2.len())?;
    validate_active_len("omega", input.active_len, input.omega.len())?;

    for row in 0..input.active_len {
        validate_finite_value("epsilon2", row, input.epsilon2[row])?;
        validate_finite_value("omega", row, input.omega[row])?;
        if row > 0 && input.omega[row] < input.omega[row - 1] {
            return Err(FullSpectrumError::DecreasingOmega {
                row,
                previous: input.omega[row - 1],
                current: input.omega[row],
            });
        }
    }

    let integral = (0..input.active_len.saturating_sub(1))
        .map(|row| {
            let left = input.omega[row] * input.epsilon2[row];
            let right = input.omega[row + 1] * input.epsilon2[row + 1];
            0.5 * (left + right) * (input.omega[row + 1] - input.omega[row])
        })
        .sum::<Real>();
    let result = integral / (2.0 * std::f64::consts::PI.powi(2) * input.number_density);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(FullSpectrumError::NonFiniteResult { value: result })
    }
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), FullSpectrumError> {
    if !value.is_finite() {
        Err(FullSpectrumError::NonFiniteInput { name, value })
    } else if value <= 0.0 {
        Err(FullSpectrumError::NonPositiveInput { name, value })
    } else {
        Ok(())
    }
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), FullSpectrumError> {
    if active_len > len {
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_value(
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), FullSpectrumError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FullSpectrumError::NonFiniteValue { field, row, value })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::Real;

    use super::{FullSpectrumError, FullSpectrumQSumInput, full_spectrum_effective_electron_count};

    #[test]
    fn effective_electron_count_matches_feff_qsum_reference() -> Result<(), FullSpectrumError> {
        let omega = array![0.0, 0.1, 0.2, 0.5, 1.0, 1.8];
        let epsilon2 = array![0.0, 0.5, 1.0, 0.25, 0.75, 0.1];

        let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 6,
        })?;

        assert_close(neff, 0.442_098_097_959_400_5, 1.0e-14);
        Ok(())
    }

    #[test]
    fn effective_electron_count_matches_feff_single_point_reference()
    -> Result<(), FullSpectrumError> {
        let omega = array![0.0, 0.1];
        let epsilon2 = array![1.0, 2.0];

        let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 1,
        })?;

        assert_eq!(neff, 0.0);
        Ok(())
    }

    #[test]
    fn effective_electron_count_rejects_invalid_inputs() {
        let omega = array![0.0, 0.1, 0.2];
        let epsilon2 = array![0.0, 0.5, 1.0];

        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.0,
                epsilon2: epsilon2.view(),
                omega: omega.view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::NonPositiveInput {
                name: "number_density",
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: epsilon2.view(),
                omega: omega.view(),
                active_len: 4,
            }),
            Err(FullSpectrumError::ActiveCountOutOfRange {
                field: "epsilon2",
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: array![0.0, f64::NAN, 1.0].view(),
                omega: omega.view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::NonFiniteValue {
                field: "epsilon2",
                row: 1,
                ..
            })
        ));
        assert!(matches!(
            full_spectrum_effective_electron_count(FullSpectrumQSumInput {
                number_density: 0.075,
                epsilon2: epsilon2.view(),
                omega: array![0.0, 0.2, 0.1].view(),
                active_len: 3,
            }),
            Err(FullSpectrumError::DecreasingOmega { row: 2, .. })
        ));
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
