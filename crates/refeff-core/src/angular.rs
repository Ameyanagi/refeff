//! Angular-momentum normalization helpers.
//!
//! FEFF stores associated-Legendre normalization factors in `xnlm`; FMS uses
//! `xnlm(m,l)` while GENFMT carries the same values in a one-based table. The
//! helpers here compute the shared value
//! `sqrt((2l+1) * (l-m)! / (l+m)!)`.

use ndarray::{Array2, ShapeBuilder};

use crate::Real;

/// Error returned by angular normalization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AngularError {
    /// Integer indices must fit in `u32` before conversion to `f64`.
    #[error("angular index {value} is too large for stable floating-point conversion")]
    IndexTooLarge { value: usize },
}

/// Return FEFF's associated-Legendre normalization factor for nonnegative `m`.
///
/// Values with `m > l` are zero, matching FEFF's explicitly zeroed `xnlm`
/// table outside the physically valid triangular region.
pub fn legendre_normalization(l: usize, m: usize) -> Result<Real, AngularError> {
    if m > l {
        return Ok(0.0);
    }

    let numerator = usize_to_real(2 * l + 1)?;
    let denominator = ((l - m + 1)..=(l + m))
        .map(usize_to_real)
        .try_fold(1.0, |accumulator, value| {
            value.map(|value| accumulator * value)
        })?;

    Ok((numerator / denominator).sqrt())
}

/// Build FEFF's `xnlm(m,l)` normalization table in Fortran order.
pub fn legendre_normalization_table(lmax: usize) -> Result<Array2<Real>, AngularError> {
    let mut table = Array2::zeros((lmax + 1, lmax + 1).f());
    for l in 0..=lmax {
        for m in 0..=l {
            table[[m, l]] = legendre_normalization(l, m)?;
        }
    }
    Ok(table)
}

fn usize_to_real(value: usize) -> Result<Real, AngularError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| AngularError::IndexTooLarge { value })
}

#[cfg(test)]
mod tests {
    use super::{AngularError, legendre_normalization, legendre_normalization_table};

    #[test]
    fn computes_snlm_values() -> Result<(), AngularError> {
        assert_close(legendre_normalization(0, 0)?, 1.0);
        assert_close(legendre_normalization(1, 0)?, 3.0_f64.sqrt());
        assert_close(legendre_normalization(1, 1)?, (3.0_f64 / 2.0).sqrt());
        assert_close(legendre_normalization(2, 2)?, (5.0_f64 / 24.0).sqrt());
        assert_eq!(legendre_normalization(1, 2)?, 0.0);
        Ok(())
    }

    #[test]
    fn builds_fortran_order_xnlm_table() -> Result<(), AngularError> {
        let table = legendre_normalization_table(3)?;

        assert_eq!(table.shape(), &[4, 4]);
        assert_eq!(table.strides(), &[1, 4]);
        assert_close(table[[0, 2]], 5.0_f64.sqrt());
        assert_close(table[[2, 2]], (5.0_f64 / 24.0).sqrt());
        assert_eq!(table[[3, 2]], 0.0);
        Ok(())
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
