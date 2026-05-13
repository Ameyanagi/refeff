//! FEFF EELS numerical helpers.
//!
//! This module ports the small kernels from `EELS/wavelength.f90` and
//! `EELS/euler.f90`. The functions keep FEFF's constants and matrix convention
//! while validating inputs instead of producing NaN/Inf outputs.

use ndarray::{Array2, ShapeBuilder};
use thiserror::Error;

use crate::{Real, RealMat};

/// FEFF electron rest energy `m_e c^2` in eV, from `COMMON/m_constants.f90`.
pub const FEFF_ELECTRON_REST_ENERGY_EV: Real = 511_004.0;
/// FEFF `HOnSqrtTwoMe` constant for electron wavelengths in atomic units.
pub const FEFF_H_ON_SQRT_TWO_ME: Real = 23.1761;

/// Error returned by FEFF EELS helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum EelsError {
    /// Scalar EELS inputs must be finite real values.
    #[error("EELS input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    /// FEFF EELS wavelength calculation requires a positive beam energy.
    #[error("EELS beam energy must be positive, got {value}")]
    InvalidBeamEnergy { value: Real },
    /// A result became non-finite after evaluating the FEFF formula.
    #[error("EELS result {name} must be finite, got {value}")]
    NonFiniteResult { name: &'static str, value: Real },
}

/// Return FEFF's relativistic electron wavelength in atomic units.
///
/// This ports `EELS/wavelength.f90`:
/// `HOnSqrtTwoMe / sqrt(E + E**2 / (2 * MeC2))`, with `E` in eV.
pub fn electron_wavelength_atomic_units(energy_ev: Real) -> Result<Real, EelsError> {
    validate_finite("energy_ev", energy_ev)?;
    if energy_ev <= 0.0 {
        return Err(EelsError::InvalidBeamEnergy { value: energy_ev });
    }

    let denominator =
        (energy_ev + energy_ev * energy_ev / (2.0 * FEFF_ELECTRON_REST_ENERGY_EV)).sqrt();
    let wavelength = FEFF_H_ON_SQRT_TWO_ME / denominator;
    if !wavelength.is_finite() {
        return Err(EelsError::NonFiniteResult {
            name: "wavelength",
            value: wavelength,
        });
    }
    Ok(wavelength)
}

/// Build FEFF's EELS Euler rotation matrix.
///
/// The three angles correspond to FEFF `a`, `b`, and `g`. The returned matrix
/// is shaped `(3, 3)` in Fortran-order `ndarray` storage and preserves the
/// `E(row,column)` assignments in `EELS/euler.f90`.
pub fn eels_euler_rotation_matrix(
    alpha: Real,
    beta: Real,
    gamma: Real,
) -> Result<RealMat, EelsError> {
    validate_finite("alpha", alpha)?;
    validate_finite("beta", beta)?;
    validate_finite("gamma", gamma)?;

    let (sin_alpha, cos_alpha) = alpha.sin_cos();
    let (sin_beta, cos_beta) = beta.sin_cos();
    let (sin_gamma, cos_gamma) = gamma.sin_cos();

    let mut matrix = Array2::zeros((3, 3).f());
    matrix[(0, 0)] = cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma;
    matrix[(1, 0)] = sin_alpha * cos_beta * cos_gamma + cos_alpha * sin_gamma;
    matrix[(0, 1)] = -cos_alpha * cos_beta * sin_gamma - sin_alpha * cos_gamma;
    matrix[(1, 1)] = -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma;
    matrix[(0, 2)] = cos_alpha * sin_beta;
    matrix[(1, 2)] = sin_alpha * sin_beta;
    matrix[(2, 2)] = cos_beta;
    matrix[(2, 0)] = -sin_beta * cos_gamma;
    matrix[(2, 1)] = sin_beta * sin_gamma;

    for &value in &matrix {
        if !value.is_finite() {
            return Err(EelsError::NonFiniteResult {
                name: "euler_matrix",
                value,
            });
        }
    }
    Ok(matrix)
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), EelsError> {
    if !value.is_finite() {
        return Err(EelsError::NonFiniteInput { name, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView2, arr2};

    #[test]
    fn electron_wavelength_matches_feff_reference() -> Result<(), EelsError> {
        assert_close(
            electron_wavelength_atomic_units(1_000.0)?,
            0.732_534_340_476_640,
        );
        assert_close(
            electron_wavelength_atomic_units(100_000.0)?,
            0.069_947_069_983_283,
        );
        assert_close(
            electron_wavelength_atomic_units(300_000.0)?,
            0.037_204_017_054_112,
        );
        Ok(())
    }

    #[test]
    fn eels_euler_rotation_matrix_matches_feff_reference() -> Result<(), EelsError> {
        assert_matrix_close(
            eels_euler_rotation_matrix(0.3, 0.4, -0.2)?.view(),
            arr2(&[
                [
                    0.921_094_097_834_994,
                    -0.114_815_729_042_654,
                    0.372_025_551_942_260,
                ],
                [
                    0.076_970_353_575_606,
                    0.990_369_592_951_021,
                    0.115_080_988_996_769,
                ],
                [
                    -0.381_655_902_095_048,
                    -0.077_365_481_465_782,
                    0.921_060_994_002_885,
                ],
            ])
            .view(),
        );
        assert_matrix_close(
            eels_euler_rotation_matrix(-1.1, 0.75, 1.4)?.view(),
            arr2(&[
                [
                    0.934_650_656_964_861,
                    -0.175_586_157_235_345,
                    0.309_188_697_759_924,
                ],
                [
                    0.336_162_895_167_387,
                    0.719_694_907_282_947,
                    -0.607_481_479_835_946,
                ],
                [
                    -0.115_856_192_531_229,
                    0.671_720_732_014_663,
                    0.731_688_868_873_821,
                ],
            ])
            .view(),
        );
        Ok(())
    }

    #[test]
    fn eels_euler_rotation_matrix_uses_fortran_order_storage() -> Result<(), EelsError> {
        let matrix = eels_euler_rotation_matrix(0.3, 0.4, -0.2)?;
        let mut expected = Vec::new();
        for column in 0..3 {
            for row in 0..3 {
                expected.push(matrix[(row, column)]);
            }
        }
        assert_eq!(matrix.as_slice_memory_order(), Some(expected.as_slice()));
        Ok(())
    }

    #[test]
    fn eels_helpers_reject_invalid_inputs() {
        assert_eq!(
            electron_wavelength_atomic_units(0.0),
            Err(EelsError::InvalidBeamEnergy { value: 0.0 })
        );
        assert!(matches!(
            electron_wavelength_atomic_units(f64::NAN),
            Err(EelsError::NonFiniteInput {
                name: "energy_ev",
                ..
            })
        ));
        assert!(matches!(
            eels_euler_rotation_matrix(0.0, f64::INFINITY, 0.0),
            Err(EelsError::NonFiniteInput { name: "beta", .. })
        ));
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-14,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, column), &actual) in actual.indexed_iter() {
            assert_close(actual, expected[(row, column)]);
        }
        assert_close(determinant_3x3(actual), 1.0);
    }

    fn determinant_3x3(matrix: ArrayView2<'_, Real>) -> Real {
        matrix[(0, 0)] * matrix[(1, 1)] * matrix[(2, 2)]
            + matrix[(0, 1)] * matrix[(1, 2)] * matrix[(2, 0)]
            + matrix[(1, 0)] * matrix[(2, 1)] * matrix[(0, 2)]
            - matrix[(2, 0)] * matrix[(1, 1)] * matrix[(0, 2)]
            - matrix[(1, 0)] * matrix[(0, 1)] * matrix[(2, 2)]
            - matrix[(0, 0)] * matrix[(2, 1)] * matrix[(1, 2)]
    }
}
