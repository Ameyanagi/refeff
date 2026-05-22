use super::*;

// FEFF `ylm.f90` stores an unsuffixed PI literal in a double variable, so
// gfortran rounds it as single precision before widening.
#[allow(clippy::approx_constant)]
const YLM_PI: Real = 3.141_592_7_f32 as Real;

/// Compute ordinary Legendre polynomials `P_l(x)` for `l = 0..=lmax`.
///
/// This ports FEFF `cpl0`, which fills `pl0(l + 1)` by the three-term
/// recurrence. The returned vector uses zero-based Rust indexing, so `out[l]`
/// contains `P_l(x)`.
#[must_use]
pub fn legendre_polynomials(x: Real, lmax: usize) -> RealVec {
    let mut values = RealVec::zeros(lmax + 1);
    if let Some(slice) = values.as_slice_mut() {
        legendre_polynomials_into(x, slice);
    }
    values
}

/// Fill a slice with ordinary Legendre polynomials `P_l(x)`.
///
/// `values[0]` receives `P_0(x)`, `values[1]` receives `P_1(x)`, and so on.
/// An empty slice is accepted and left unchanged, matching FEFF's defensive
/// bounds checks around short `pl0` arrays.
pub fn legendre_polynomials_into(x: Real, values: &mut [Real]) {
    if values.is_empty() {
        return;
    }
    values[0] = 1.0;
    if values.len() < 2 {
        return;
    }
    values[1] = x;
    if values.len() < 3 {
        return;
    }

    for ell in 1..(values.len() - 1) {
        let ell_real = ell as Real;
        values[ell + 1] = ((2.0 * ell_real + 1.0) * x * values[ell] - ell_real * values[ell - 1])
            / (ell_real + 1.0);
    }
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

/// Port of FEFF `ylm`: complex spherical harmonics for a Cartesian vector.
///
/// Values are returned in FEFF order:
/// `Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...`.
/// The input vector is normalized internally; the zero vector follows FEFF's
/// convention of using `theta = 0` and `phi = 0`.
pub fn spherical_harmonics(vector: [Real; 3], lmax: usize) -> Result<ComplexVec, AngularError> {
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(AngularError::NonFiniteVector);
    }

    let count = lmax
        .checked_add(1)
        .and_then(|size| size.checked_mul(size))
        .ok_or(AngularError::IndexTooLarge { value: lmax })?;
    let mut values = vec![Complex::new(0.0, 0.0); count];
    let y00 = 1.0 / (4.0 * YLM_PI).sqrt();
    values[0] = Complex::new(y00, 0.0);
    if lmax == 0 {
        return Ok(ComplexVec::from_vec(values));
    }

    let abmax = vector[0].abs().max(vector[1].abs());
    let (cos_phi, sin_phi) = if abmax > 0.0 {
        let a = vector[0] / abmax;
        let b = vector[1] / abmax;
        let ab = (a * a + b * b).sqrt();
        (a / ab, b / ab)
    } else {
        (1.0, 0.0)
    };

    let abcmax = abmax.max(vector[2].abs());
    let (cos_theta, sin_theta) = if abcmax > 0.0 {
        let a = vector[0] / abcmax;
        let b = vector[1] / abcmax;
        let c = vector[2] / abcmax;
        let ab = a * a + b * b;
        let abc = (ab + c * c).sqrt();
        (c / abc, ab.sqrt() / abc)
    } else {
        (1.0, 0.0)
    };

    values[spherical_harmonic_index(1, 0)] = Complex::new(3.0_f64.sqrt() * y00 * cos_theta, 0.0);
    let temp = -(1.5_f64).sqrt() * y00 * sin_theta;
    values[spherical_harmonic_index(1, 1)] = Complex::new(temp * cos_phi, temp * sin_phi);
    values[spherical_harmonic_index(1, -1)] = -values[spherical_harmonic_index(1, 1)].conj();

    for l in 2..=lmax {
        let l_real = l as Real;
        let sign_l = alternating_sign_usize(l);
        let previous_diagonal = values[spherical_harmonic_index(l - 1, (l - 1) as isize)];

        let diagonal_factor = -((2.0 * l_real + 1.0) / (2.0 * l_real)).sqrt() * sin_theta;
        let diagonal = Complex::new(
            diagonal_factor * (cos_phi * previous_diagonal.re - sin_phi * previous_diagonal.im),
            diagonal_factor * (cos_phi * previous_diagonal.im + sin_phi * previous_diagonal.re),
        );
        values[spherical_harmonic_index(l, l as isize)] = diagonal;
        values[spherical_harmonic_index(l, -(l as isize))] = diagonal.conj() * sign_l;

        let near_diagonal = previous_diagonal * ((2.0 * l_real + 1.0).sqrt() * cos_theta);
        values[spherical_harmonic_index(l, (l - 1) as isize)] = near_diagonal;
        values[spherical_harmonic_index(l, -((l - 1) as isize))] =
            near_diagonal.conj() * alternating_sign_usize(l - 1);

        let first_factor = cos_theta * (4.0 * l_real * l_real - 1.0).sqrt();
        let second_factor_base = -((2.0 * l_real + 1.0) / (2.0 * l_real - 3.0)).sqrt();
        for m in (0..=(l - 2)).rev() {
            let m_real = m as Real;
            let denominator = ((l_real + m_real) * (l_real - m_real)).sqrt();
            let first_factor = first_factor / denominator;
            let second_factor = second_factor_base
                * ((l_real + m_real - 1.0) * (l_real - m_real - 1.0)).sqrt()
                / denominator;
            let harmonic = values[spherical_harmonic_index(l - 1, m as isize)] * first_factor
                + values[spherical_harmonic_index(l - 2, m as isize)] * second_factor;
            values[spherical_harmonic_index(l, m as isize)] = harmonic;
            values[spherical_harmonic_index(l, -(m as isize))] =
                harmonic.conj() * alternating_sign_usize(m);
        }
    }

    Ok(ComplexVec::from_vec(values))
}

fn alternating_sign_usize(exponent: usize) -> Real {
    if exponent.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

fn spherical_harmonic_index(l: usize, m: isize) -> usize {
    let base = l * l + l;
    if m < 0 {
        base - m.unsigned_abs()
    } else {
        base + m as usize
    }
}
