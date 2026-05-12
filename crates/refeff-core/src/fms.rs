//! Full multiple-scattering helpers.
//!
//! FEFF's FMS routines use Rehr-Albers polynomial tables when building
//! multiple-scattering propagators. The helpers here keep the legacy table
//! layout explicit while returning Rust-owned `ndarray` storage.

use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex32;
use thiserror::Error;

/// Error returned by FEFF FMS helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FmsError {
    /// FEFF angular limits must fit the allocated `clm(lx+2, 2*lx+3)` table.
    #[error("{name}={value} is invalid for lx={lx}")]
    InvalidAngularLimit {
        name: &'static str,
        value: usize,
        lx: usize,
    },
    /// `rho` appears in the denominator of FEFF `xclmz`.
    #[error("rho must be nonzero")]
    ZeroRho,
    /// `rho` must contain finite real and imaginary parts.
    #[error("rho must be finite")]
    NonFiniteRho,
}

/// Port of FEFF `xclmz`: Rehr-Albers Hankel-like polynomial table.
///
/// The returned matrix has FEFF's work shape `clm(lx+2, 2*lx+3)` and
/// Fortran-order strides. Rust indices are zero-based, so FEFF `clm(il, im)`
/// is `table[(il - 1, im - 1)]`.
pub fn rehr_albers_polynomials(
    lx: usize,
    lmaxp1: usize,
    mmaxp1: usize,
    rho: Complex32,
) -> Result<Array2<Complex32>, FmsError> {
    let max_lmaxp1 = lx.checked_add(1).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    if lmaxp1 == 0 || lmaxp1 > max_lmaxp1 {
        return Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: lmaxp1,
            lx,
        });
    }
    if mmaxp1 == 0 {
        return Err(FmsError::InvalidAngularLimit {
            name: "mmaxp1",
            value: mmaxp1,
            lx,
        });
    }
    if !(rho.re.is_finite() && rho.im.is_finite()) {
        return Err(FmsError::NonFiniteRho);
    }
    if rho == Complex32::new(0.0, 0.0) {
        return Err(FmsError::ZeroRho);
    }

    let rows = lx.checked_add(2).ok_or(FmsError::InvalidAngularLimit {
        name: "lx",
        value: lx,
        lx,
    })?;
    let cols = lx
        .checked_mul(2)
        .and_then(|value| value.checked_add(3))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    let mut clm = Array2::zeros((rows, cols).f());

    let one = Complex32::new(1.0, 0.0);
    let z = Complex32::new(0.0, -1.0) / rho;
    clm[(0, 0)] = one;
    clm[(1, 0)] = one - z;

    let lmax = lmaxp1 - 1;
    for il in 2..=lmax {
        let factor = odd_factor(il, lx)? * z;
        clm[(il, 0)] = clm[(il - 2, 0)] - factor * clm[(il - 1, 0)];
    }

    let mut cmm = one;
    let mmxp1 = lmaxp1.min(mmaxp1);
    for im in 2..=mmxp1 {
        let m = im - 1;
        let cmm_factor = odd_factor(m, lx)? * z;
        cmm = -cmm * cmm_factor;
        clm[(im - 1, im - 1)] = cmm;
        clm[(im, im - 1)] = cmm * odd_factor(im, lx)? * (one - Complex32::new(im as f32, 0.0) * z);

        for il in (im + 1)..=lmax {
            let factor = odd_factor(il, lx)? * z;
            clm[(il, im - 1)] =
                clm[(il - 2, im - 1)] - factor * (clm[(il - 1, im - 1)] + clm[(il - 1, im - 2)]);
        }
    }

    Ok(clm)
}

fn odd_factor(index: usize, lx: usize) -> Result<Complex32, FmsError> {
    let value = index
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .ok_or(FmsError::InvalidAngularLimit {
            name: "lx",
            value: lx,
            lx,
        })?;
    Ok(Complex32::new(value as f32, 0.0))
}

#[cfg(test)]
mod tests {
    use super::{FmsError, rehr_albers_polynomials};
    use ndarray::ArrayView2;
    use num_complex::Complex32;

    #[test]
    fn xclmz_matches_feff_reference_lx3() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;

        assert_eq!(table.shape(), &[5, 9]);
        assert_eq!(table.strides(), &[1, 5]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.2322206, 0.725_689_4));
        assert_complex32_close(table[(3, 0)], Complex32::new(-10.012509, 5.438_266));
        assert_complex32_close(table[(2, 1)], Complex32::new(-2.1395304, 4.1993084));
        assert_complex32_close(table[(3, 2)], Complex32::new(-23.036537, -6.8588142));
        assert_complex32_close(table[(4, 3)], Complex32::new(8.928_719, -161.62775));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(-58.983994, -154.61885),
        );
        assert_eq!(nonzero_count(table.view()), 11);
        Ok(())
    }

    #[test]
    fn xclmz_matches_feff_reference_with_limited_m() -> Result<(), FmsError> {
        let table = rehr_albers_polynomials(4, 3, 2, Complex32::new(-0.8, 1.1))?;

        assert_eq!(table.shape(), &[6, 11]);
        assert_eq!(table.strides(), &[1, 6]);
        assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
        assert_complex32_close(table[(1, 0)], Complex32::new(1.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 0)], Complex32::new(3.2834187, -2.840029));
        assert_complex32_close(table[(1, 1)], Complex32::new(0.5945946, -0.432_432_4));
        assert_complex32_close(table[(2, 1)], Complex32::new(2.7830534, -4.382761));
        assert_complex32_close(
            matrix_sum(table.view()),
            Complex32::new(9.255661, -8.087655),
        );
        assert_eq!(nonzero_count(table.view()), 5);
        Ok(())
    }

    #[test]
    fn xclmz_rejects_invalid_inputs() {
        assert_eq!(
            rehr_albers_polynomials(3, 0, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 0,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 5, 1, Complex32::new(1.0, 0.0)),
            Err(FmsError::InvalidAngularLimit {
                name: "lmaxp1",
                value: 5,
                lx: 3,
            })
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(0.0, 0.0)),
            Err(FmsError::ZeroRho)
        );
        assert_eq!(
            rehr_albers_polynomials(3, 1, 1, Complex32::new(f32::NAN, 0.0)),
            Err(FmsError::NonFiniteRho)
        );
    }

    fn matrix_sum(matrix: ArrayView2<'_, Complex32>) -> Complex32 {
        matrix
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
    }

    fn nonzero_count(matrix: ArrayView2<'_, Complex32>) -> usize {
        matrix
            .iter()
            .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
            .count()
    }

    fn assert_complex32_close(actual: Complex32, expected: Complex32) {
        assert!(
            (actual - expected).norm() < 2.0e-4,
            "actual={actual:?} expected={expected:?}"
        );
    }
}
