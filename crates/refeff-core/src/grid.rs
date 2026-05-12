//! FEFF common energy and radial-grid helpers.
//!
//! These functions port the small common routines `getxk.f90`, `xx.f90`, and
//! `m_ifuns.f90`. FEFF uses a 1-based logarithmic radial grid with
//! `x = -8.8 + (j - 1) * delta` and `r = exp(x)`.

use thiserror::Error;

use crate::Real;

/// Default FEFF logarithmic radial-grid spacing.
pub const LOUCKS_DELTA: Real = 0.05;

/// Offset used by FEFF's Loucks radial grid.
pub const LOUCKS_X_OFFSET: Real = 8.8;

/// Error returned by radial-grid indexing helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GridError {
    /// The radius must be positive and finite before `ln(r)` is meaningful.
    #[error("radius must be positive and finite, got {radius}")]
    InvalidRadius { radius: Real },
    /// The logarithmic grid spacing must be positive and finite.
    #[error("grid delta must be positive and finite, got {delta}")]
    InvalidDelta { delta: Real },
}

/// Convert energy in Hartrees to FEFF's signed photoelectron wave number.
///
/// This ports `getxk`: `sqrt(2E)` above the edge and `-sqrt(-2E)` below it.
#[must_use]
pub fn wave_number_from_hartree(energy: Real) -> Real {
    let magnitude = (2.0 * energy).abs().sqrt();
    if energy < 0.0 { -magnitude } else { magnitude }
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a 1-based index.
#[must_use]
pub fn loucks_x(index_1based: usize) -> Real {
    radial_x(index_1based, LOUCKS_DELTA)
}

/// Return the radial coordinate for a 1-based Loucks grid index.
#[must_use]
pub fn loucks_radius(index_1based: usize) -> Real {
    loucks_x(index_1based).exp()
}

/// Return the 1-based Loucks grid index immediately below `radius`.
pub fn loucks_index_below(radius: Real) -> Result<usize, GridError> {
    radial_index_below(radius, LOUCKS_DELTA)
}

/// Return the logarithmic `x = ln(r)` grid coordinate for a custom spacing.
#[must_use]
pub fn radial_x(index_1based: usize, delta: Real) -> Real {
    -LOUCKS_X_OFFSET + (index_1based as Real - 1.0) * delta
}

/// Return the radial coordinate for a custom logarithmic spacing.
#[must_use]
pub fn radial_radius(index_1based: usize, delta: Real) -> Real {
    radial_x(index_1based, delta).exp()
}

/// Return the 1-based grid index immediately below `radius` for a custom spacing.
pub fn radial_index_below(radius: Real, delta: Real) -> Result<usize, GridError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(GridError::InvalidRadius { radius });
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(GridError::InvalidDelta { delta });
    }
    let index = ((radius.ln() + LOUCKS_X_OFFSET) / delta + 1.0).trunc();
    Ok(index as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_energy_to_signed_wave_number() {
        assert_eq!(wave_number_from_hartree(2.0), 2.0);
        assert_eq!(wave_number_from_hartree(-2.0), -2.0);
        assert_eq!(wave_number_from_hartree(0.0), 0.0);
    }

    #[test]
    fn reproduces_loucks_log_grid_points() {
        assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
        assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
        assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
    }

    #[test]
    fn maps_radius_to_index_below() -> Result<(), GridError> {
        let radius = loucks_radius(42);
        assert_eq!(loucks_index_below(radius)?, 42);

        let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
        assert_eq!(loucks_index_below(midpoint)?, 42);
        Ok(())
    }

    #[test]
    fn rejects_invalid_radius_or_delta() {
        assert!(matches!(
            loucks_index_below(0.0),
            Err(GridError::InvalidRadius { .. })
        ));
        assert!(matches!(
            radial_index_below(1.0, 0.0),
            Err(GridError::InvalidDelta { .. })
        ));
    }
}
