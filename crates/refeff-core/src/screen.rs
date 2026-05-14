//! FEFF SCREEN helper kernels.
//!
//! These routines cover small, self-contained pieces from `SCREEN/frgrid.f90`
//! and `SCREEN/fxc.f90`. The full SCREEN/CRPA drivers also depend on phase,
//! potential, and FMS handoff state; keeping these kernels separate makes them
//! usable and testable while those drivers are ported incrementally.

use ndarray::Array1;
use thiserror::Error;

use crate::{Real, RealVec};

/// Error returned by FEFF SCREEN helper kernels.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ScreenError {
    #[error("SCREEN input {name} must be finite, got {value}")]
    NonFiniteInput { name: &'static str, value: Real },
    #[error("SCREEN input {name} must be positive, got {value}")]
    NonPositiveInput { name: &'static str, value: Real },
    #[error("SCREEN radial count must be positive")]
    EmptyRadialGrid,
    #[error("SCREEN active radial count {active_count} exceeds input length {len}")]
    ActiveCountOutOfRange { active_count: usize, len: usize },
    #[error("SCREEN radial index is outside isize range: {value}")]
    RadialIndexOutOfRange { value: Real },
}

/// Port of SCREEN `setri`: build the logarithmic radial grid.
///
/// FEFF stores radial samples as `ri(i) = exp(-x0 + (i-1)*dx)` using 1-based
/// loop bounds. This helper returns the same values in Rust's zero-based
/// [`ndarray::Array1`] layout.
pub fn screen_radial_grid(dx: Real, x0: Real, count: usize) -> Result<RealVec, ScreenError> {
    validate_positive("dx", dx)?;
    validate_finite("x0", x0)?;
    if count == 0 {
        return Err(ScreenError::EmptyRadialGrid);
    }

    Ok(Array1::from_iter(
        (0..count).map(|index| (-x0 + index as Real * dx).exp()),
    ))
}

/// Port of SCREEN `getiat`: map a radius to FEFF's 1-based radial index.
///
/// Fortran assigns the floating-point expression to an integer, which truncates
/// toward zero. Returning an `isize` preserves that behavior for callers that
/// need to handle out-of-grid locations explicitly. Values reconstructed from
/// the same logarithmic grid are snapped back to exact integer boundaries when
/// roundoff alone would move them just below the FEFF index.
pub fn screen_radial_index_1based(x0: Real, dx: Real, radius: Real) -> Result<isize, ScreenError> {
    validate_finite("x0", x0)?;
    validate_positive("dx", dx)?;
    validate_positive("radius", radius)?;

    let value = (radius.ln() + x0) / dx + 1.0;
    if value < isize::MIN as Real || value > isize::MAX as Real {
        return Err(ScreenError::RadialIndexOutOfRange { value });
    }
    Ok(feff_truncated_index(value))
}

fn feff_truncated_index(value: Real) -> isize {
    let nearest = value.round();
    let tolerance = 1.0e-12 * nearest.abs().max(1.0);
    if value >= 0.0 && (value - nearest).abs() <= tolerance {
        nearest as isize
    } else {
        value.trunc() as isize
    }
}

/// Port of SCREEN `ldafxc`: local-density exchange-correlation kernel.
///
/// FEFF evaluates only the first `active_count` rows, sets non-positive
/// electron-density rows to zero, and uses a pure-exchange branch when
/// `exchange_selector == 2`.
pub fn screen_lda_exchange_correlation_kernel(
    radii: &[Real],
    electron_density: &[Real],
    exchange_selector: i32,
    active_count: usize,
) -> Result<RealVec, ScreenError> {
    validate_active_count(active_count, radii.len())?;
    validate_active_count(active_count, electron_density.len())?;

    let mut output = Array1::zeros(active_count);
    for index in 0..active_count {
        let radius = radii[index];
        let density = electron_density[index];
        validate_positive("radius", radius)?;
        validate_finite("electron_density", density)?;
        if density <= 0.0 {
            continue;
        }

        let rs = (density / 3.0).powf(-1.0 / 3.0);
        let exchange = -1.222 / rs;
        let correlation = if exchange_selector == 2 {
            0.0
        } else {
            -0.75924 / (11.4 + rs)
        };
        output[index] = rs.powi(3) / radius.powi(2) / 6.0 * (exchange + correlation);
    }
    Ok(output)
}

fn validate_active_count(active_count: usize, len: usize) -> Result<(), ScreenError> {
    if active_count > len {
        Err(ScreenError::ActiveCountOutOfRange { active_count, len })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: Real) -> Result<(), ScreenError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreenError::NonFiniteInput { name, value })
    }
}

fn validate_positive(name: &'static str, value: Real) -> Result<(), ScreenError> {
    validate_finite(name, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreenError::NonPositiveInput { name, value })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ScreenError, screen_lda_exchange_correlation_kernel, screen_radial_grid,
        screen_radial_index_1based,
    };

    #[test]
    fn radial_grid_matches_feff_setri_reference() -> Result<(), ScreenError> {
        let grid = screen_radial_grid(0.05, 8.8, 5)?;

        assert_close(grid[0], 0.000_150_733_075_095_476_5, 1.0e-15);
        assert_close(grid[1], 0.000_158_461_325_115_751_26, 1.0e-15);
        assert_close(grid[2], 0.000_166_585_810_987_633_24, 1.0e-15);
        assert_close(grid[3], 0.000_175_126_848_157_658_42, 1.0e-15);
        assert_close(grid[4], 0.000_184_105_793_667_578_87, 1.0e-15);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, grid[2])?, 3);
        assert_eq!(screen_radial_index_1based(8.8, 0.05, 1.0)?, 177);
        assert_eq!(screen_radial_index_1based(0.0, 1.0, 0.01)?, -3);
        Ok(())
    }

    #[test]
    fn lda_exchange_correlation_kernel_matches_feff_ldafxc_reference() -> Result<(), ScreenError> {
        let radii = [0.5, 0.75, 1.0, 1.5, 2.0];
        let density = [0.04, 0.10, 0.0, -1.0, 0.25];

        let full = screen_lda_exchange_correlation_kernel(&radii, &density, 0, radii.len())?;
        assert_close(full[0], -16.919_199_214_545_813, 1.0e-13);
        assert_close(full[1], -3.960_989_192_391_738_6, 1.0e-13);
        assert_close(full[2], 0.0, 1.0e-15);
        assert_close(full[3], 0.0, 1.0e-15);
        assert_close(full[4], -0.294_609_719_384_913, 1.0e-13);

        let exchange_only =
            screen_lda_exchange_correlation_kernel(&radii, &density, 2, radii.len())?;
        assert_close(exchange_only[0], -14.488_412_060_289_518, 1.0e-13);
        assert_close(exchange_only[1], -3.495_786_749_594_309_6, 1.0e-13);
        assert_close(exchange_only[4], -0.266_878_831_976_939_35, 1.0e-13);
        Ok(())
    }

    #[test]
    fn screen_helpers_reject_invalid_inputs() {
        assert!(matches!(
            screen_radial_grid(0.0, 8.8, 5),
            Err(ScreenError::NonPositiveInput { name: "dx", .. })
        ));
        assert!(matches!(
            screen_radial_grid(0.05, 8.8, 0),
            Err(ScreenError::EmptyRadialGrid)
        ));
        assert!(matches!(
            screen_radial_index_1based(8.8, 0.05, -1.0),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[0.1], 0, 2),
            Err(ScreenError::ActiveCountOutOfRange { .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[0.0], &[0.1], 0, 1),
            Err(ScreenError::NonPositiveInput { name: "radius", .. })
        ));
        assert!(matches!(
            screen_lda_exchange_correlation_kernel(&[1.0], &[f64::NAN], 0, 1),
            Err(ScreenError::NonFiniteInput {
                name: "electron_density",
                ..
            })
        ));
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}
