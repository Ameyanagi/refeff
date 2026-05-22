use super::*;

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
