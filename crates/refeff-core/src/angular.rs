//! Angular-momentum normalization helpers.
//!
//! FEFF stores associated-Legendre normalization factors in `xnlm`; FMS uses
//! `xnlm(m,l)` while GENFMT carries the same values in a one-based table. The
//! helpers here compute the shared value
//! `sqrt((2l+1) * (l-m)! / (l+m)!)`.

use ndarray::{Array2, Array3, ShapeBuilder};

use crate::{Real, RealVec};

/// Error returned by angular normalization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AngularError {
    /// Integer indices must fit in `u32` before conversion to `f64`.
    #[error("angular index {value} is too large for stable floating-point conversion")]
    IndexTooLarge { value: usize },
    /// FEFF angular helpers accept only integer (`1`) and half-integer (`2`) scales.
    #[error("invalid angular momentum scale {scale}; expected 1 or 2")]
    InvalidWignerScale { scale: i32 },
    /// A Wigner 3j argument did not divide evenly by the selected scale.
    #[error("Wigner 3j argument {argument} is not divisible by scale {scale}")]
    InvalidWignerParity { argument: i32, scale: i32 },
    /// FEFF's common `cwig3j` table is limited to factorial arguments up to 58.
    #[error("Wigner 3j factorial argument {argument} exceeds FEFF limit {limit}")]
    WignerFactorialOutOfRange { argument: i32, limit: i32 },
    /// The requested magnetic index does not fit the allocated table.
    #[error("magnetic index {magnetic} is outside table range for lmax {lmax}")]
    MagneticIndexOutOfRange { magnetic: isize, lmax: usize },
    /// FEFF Wigner rotations require a finite angle.
    #[error("Wigner rotation angle must be finite")]
    NonFiniteRotationAngle,
}

/// Spin-orbit Clebsch-Gordon tables used by FEFF's FMS and POT paths.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinOrbitCouplingTables {
    /// `j = l + 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub plus: Array3<Real>,
    /// `j = l - 1/2` coefficients, indexed as `[l, m + m_offset, spin - 1]`.
    pub minus: Array3<Real>,
    /// Offset added to signed `m` before indexing the second axis.
    pub m_offset: usize,
}

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

/// Port of FEFF `cwig3j`.
///
/// Inputs are scaled by `scale`: use `scale = 1` for integer angular momenta
/// and `scale = 2` for half-integers represented as doubled integers.
pub fn wigner_3j(
    j1: i32,
    j2: i32,
    j3: i32,
    m1: i32,
    m2: i32,
    scale: i32,
) -> Result<Real, AngularError> {
    const FACTORIAL_LIMIT: i32 = 58;

    if scale != 1 && scale != 2 {
        return Err(AngularError::InvalidWignerScale { scale });
    }

    let double_scale = scale + scale;
    let m3 = -m1 - m2;
    if m1.abs() + m2.abs() == 0 && (j1 + j2 + j3) % double_scale != 0 {
        return Ok(0.0);
    }

    let mut values = [
        j1 + j2 - j3,
        j2 + j3 - j1,
        j3 + j1 - j2,
        j1 + m1,
        j1 - m1,
        j2 + m2,
        j2 - m2,
        j3 + m3,
        j3 - m3,
        j1 + j2 + j3 + scale,
        j2 - j3 - m1,
        j1 - j3 + m2,
    ];

    for (index, value) in values.iter_mut().enumerate() {
        if index < 10 && *value < 0 {
            return Ok(0.0);
        }
        if *value % scale != 0 {
            return Err(AngularError::InvalidWignerParity {
                argument: *value,
                scale,
            });
        }
        *value /= scale;
        if *value > FACTORIAL_LIMIT {
            return Err(AngularError::WignerFactorialOutOfRange {
                argument: *value,
                limit: FACTORIAL_LIMIT,
            });
        }
    }

    let log_factorial = log_factorials(FACTORIAL_LIMIT)?;
    let k_min = values[10].max(values[11]).max(0);
    let k_max = values[0].min(values[4]).min(values[5]);
    if k_min > k_max {
        return Ok(0.0);
    }

    let mut sign = if k_min % 2 == 0 { 1.0 } else { -1.0 };
    let c = values[..9].iter().try_fold(
        -log_factorial_value(&log_factorial, values[9], FACTORIAL_LIMIT)?,
        |accumulator, value| {
            Ok::<_, AngularError>(
                accumulator + log_factorial_value(&log_factorial, *value, FACTORIAL_LIMIT)?,
            )
        },
    )? / 2.0;

    let mut coefficient = 0.0;
    for k in k_min..=k_max {
        let b = log_factorial_value(&log_factorial, k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[0] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[4] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, values[5] - k, FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, k - values[10], FACTORIAL_LIMIT)?
            + log_factorial_value(&log_factorial, k - values[11], FACTORIAL_LIMIT)?;
        coefficient += sign * (c - b).exp();
        sign = -sign;
    }

    if (j1 - j2 - m3) % double_scale != 0 {
        coefficient = -coefficient;
    }
    Ok(coefficient)
}

/// Port of FEFF `rotwig`: Wigner small-d rotation matrix element.
///
/// `jj`, `m1`, and `m2` are scaled by `scale`, matching FEFF's `ient`: use
/// `scale = 1` for integer angular momenta and `scale = 2` for half-integers
/// represented as doubled integers.
pub fn wigner_rotation(
    beta: Real,
    jj: i32,
    m1: i32,
    m2: i32,
    scale: i32,
) -> Result<Real, AngularError> {
    const FACTORIAL_LIMIT: i32 = 58;

    if scale != 1 && scale != 2 {
        return Err(AngularError::InvalidWignerScale { scale });
    }
    if !beta.is_finite() {
        return Err(AngularError::NonFiniteRotationAngle);
    }

    let (m1p, m2p, beta, sign) = if m1 >= 0 && m1.abs() >= m2.abs() {
        (m1, m2, beta, 1.0)
    } else if m2 >= 0 && m2.abs() >= m1.abs() {
        (m2, m1, -beta, 1.0)
    } else if m1 <= 0 && m1.abs() >= m2.abs() {
        (
            -m1,
            -m2,
            beta,
            alternating_sign(checked_scaled_argument(m1 - m2, scale)?),
        )
    } else {
        (
            -m2,
            -m1,
            -beta,
            alternating_sign(checked_scaled_argument(m2 - m1, scale)?),
        )
    };

    let log_factorial = log_factorials(FACTORIAL_LIMIT)?;
    let zeta = (beta / 2.0).cos();
    let eta = (beta / 2.0).sin();
    let mut total = 0.0;
    let mut term_index = m1p - m2p;
    let last = jj - m2p;
    while term_index <= last {
        let factorial_arguments = [
            checked_scaled_argument(jj + m1p, scale)?,
            checked_scaled_argument(jj - m1p, scale)?,
            checked_scaled_argument(jj + m2p, scale)?,
            checked_scaled_argument(jj - m2p, scale)?,
            checked_scaled_argument(jj + m1p - term_index, scale)?,
            checked_scaled_argument(jj - m2p - term_index, scale)?,
            checked_scaled_argument(term_index, scale)?,
            checked_scaled_argument(m2p - m1p + term_index, scale)?,
        ];
        let zeta_power = checked_scaled_argument(2 * jj + m1p - m2p - 2 * term_index, scale)?;
        let eta_power = checked_scaled_argument(2 * term_index - m1p + m2p, scale)?;
        if zeta_power < 0 || eta_power < 0 {
            return Err(AngularError::WignerFactorialOutOfRange {
                argument: zeta_power.min(eta_power),
                limit: FACTORIAL_LIMIT,
            });
        }

        let mut factor = 0.0;
        for &argument in &factorial_arguments[..4] {
            factor += log_factorial_value(&log_factorial, argument, FACTORIAL_LIMIT)? / 2.0;
        }
        for &argument in &factorial_arguments[4..] {
            factor -= log_factorial_value(&log_factorial, argument, FACTORIAL_LIMIT)?;
        }

        let coefficient =
            alternating_sign(checked_scaled_argument(term_index, scale)?) * factor.exp();
        let term = match (zeta_power, eta_power) {
            (0, 0) => coefficient,
            (_, 0) => coefficient * zeta.powi(zeta_power),
            (0, _) => coefficient * eta.powi(eta_power),
            _ => coefficient * zeta.powi(zeta_power) * eta.powi(eta_power),
        };
        total += term;
        term_index += scale;
    }

    Ok(sign * total)
}

/// Build FEFF `t3jp` and `t3jm` spin-orbit coupling tables.
pub fn spin_orbit_coupling_tables(lmax: usize) -> Result<SpinOrbitCouplingTables, AngularError> {
    let mut plus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let mut minus = Array3::zeros((lmax + 1, 2 * lmax + 1, 2).f());
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;

    for l in 0..=lmax {
        let l_i32 = usize_to_i32(l)?;
        let l_isize = isize::try_from(l).map_err(|_| AngularError::IndexTooLarge { value: l })?;
        for magnetic in -l_isize..=l_isize {
            let magnetic_i32 = isize_to_i32(magnetic)?;
            let magnetic_index = magnetic_table_index(magnetic, lmax)?;
            for spin_index in 0..2 {
                let spin_i32 = usize_to_i32(spin_index + 1)?;
                let j1 = 2 * l_i32;
                let j2 = 1;
                let j3p = j1 + 1;
                let j3m = j1 - 1;
                let m1 = 2 * magnetic_i32;
                let m2 = 2 * spin_i32 - 3;
                let sign = feff_spin_coupling_sign(j2, j1, m1, m2);

                plus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3p + 1).sqrt() * wigner_3j(j1, j2, j3p, m1, m2, 2)?;
                minus[[l, magnetic_index, spin_index]] =
                    sign * f64::from(j3m + 1).sqrt() * wigner_3j(j1, j2, j3m, m1, m2, 2)?;
            }
        }
    }

    Ok(SpinOrbitCouplingTables {
        plus,
        minus,
        m_offset: usize::try_from(lmax_isize)
            .map_err(|_| AngularError::IndexTooLarge { value: lmax })?,
    })
}

fn usize_to_real(value: usize) -> Result<Real, AngularError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| AngularError::IndexTooLarge { value })
}

fn usize_to_i32(value: usize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::IndexTooLarge { value })
}

fn isize_to_i32(value: isize) -> Result<i32, AngularError> {
    i32::try_from(value).map_err(|_| AngularError::MagneticIndexOutOfRange {
        magnetic: value,
        lmax: usize::MAX,
    })
}

fn magnetic_table_index(magnetic: isize, lmax: usize) -> Result<usize, AngularError> {
    let lmax_isize =
        isize::try_from(lmax).map_err(|_| AngularError::IndexTooLarge { value: lmax })?;
    let shifted = magnetic + lmax_isize;
    usize::try_from(shifted).map_err(|_| AngularError::MagneticIndexOutOfRange { magnetic, lmax })
}

fn feff_spin_coupling_sign(j2: i32, j1: i32, m1: i32, m2: i32) -> Real {
    let phase = (j2 - j1 - m1 - m2) / 2;
    if phase % 2 == 0 { 1.0 } else { -1.0 }
}

fn checked_scaled_argument(argument: i32, scale: i32) -> Result<i32, AngularError> {
    if argument % scale != 0 {
        return Err(AngularError::InvalidWignerParity { argument, scale });
    }
    Ok(argument / scale)
}

fn alternating_sign(exponent: i32) -> Real {
    if exponent % 2 == 0 { 1.0 } else { -1.0 }
}

fn log_factorials(limit: i32) -> Result<Vec<Real>, AngularError> {
    let limit = usize::try_from(limit).map_err(|_| AngularError::WignerFactorialOutOfRange {
        argument: limit,
        limit,
    })?;
    let mut values = Vec::with_capacity(limit + 1);
    let mut previous = 0.0;
    values.push(previous);
    for index in 1..=limit {
        let index = usize_to_real(index)?;
        previous += index.ln();
        values.push(previous);
    }
    Ok(values)
}

fn log_factorial_value(
    log_factorials: &[Real],
    argument: i32,
    limit: i32,
) -> Result<Real, AngularError> {
    if argument < 0 || argument > limit {
        return Err(AngularError::WignerFactorialOutOfRange { argument, limit });
    }
    let index = usize::try_from(argument)
        .map_err(|_| AngularError::WignerFactorialOutOfRange { argument, limit })?;
    log_factorials
        .get(index)
        .copied()
        .ok_or(AngularError::WignerFactorialOutOfRange { argument, limit })
}

#[cfg(test)]
mod tests {
    use super::{
        AngularError, legendre_normalization, legendre_normalization_table, legendre_polynomials,
        spin_orbit_coupling_tables, wigner_3j, wigner_rotation,
    };

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
    fn computes_cpl0_legendre_polynomials() {
        let values = legendre_polynomials(0.25, 4);
        let expected = [1.0, 0.25, -0.40625, -0.3359375, 0.15771484375];

        for (&actual, expected) in values.iter().zip(expected) {
            assert_close(actual, expected);
        }
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

    #[test]
    fn computes_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
        assert_close(wigner_3j(1, 1, 0, 0, 0, 1)?, -1.0 / 3.0_f64.sqrt());
        assert_eq!(wigner_3j(1, 1, 3, 0, 0, 1)?, 0.0);
        Ok(())
    }

    #[test]
    fn computes_half_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
        assert_close(wigner_3j(0, 1, 1, 0, -1, 2)?, -1.0 / 2.0_f64.sqrt());
        assert_eq!(
            wigner_3j(2, 2, 2, 1, 0, 2),
            Err(AngularError::InvalidWignerParity {
                argument: 3,
                scale: 2,
            })
        );
        Ok(())
    }

    #[test]
    fn computes_wigner_rotation_elements() -> Result<(), AngularError> {
        assert_close(wigner_rotation(0.7, 2, 1, -1, 1)?, 0.2974375221921237);
        assert_close(wigner_rotation(1.1, 3, -2, 1, 1)?, 0.4544222701103565);
        assert_close(wigner_rotation(0.7, 3, 1, -1, 2)?, -0.5648429673316498);
        assert_close(wigner_rotation(-0.9, 5, -3, 1, 2)?, 0.494867123375203);
        assert_close(wigner_rotation(0.4, 4, -2, -4, 2)?, -0.3740481938792059);
        Ok(())
    }

    #[test]
    fn rejects_invalid_wigner_rotation_inputs() {
        assert_eq!(
            wigner_rotation(0.1, 1, 0, 0, 3),
            Err(AngularError::InvalidWignerScale { scale: 3 })
        );
        assert_eq!(
            wigner_rotation(f64::NAN, 1, 0, 0, 1),
            Err(AngularError::NonFiniteRotationAngle)
        );
        assert!(matches!(
            wigner_rotation(0.1, 3, 1, 0, 2),
            Err(AngularError::InvalidWignerParity { .. })
        ));
    }

    #[test]
    fn builds_spin_orbit_coupling_tables() -> Result<(), AngularError> {
        let tables = spin_orbit_coupling_tables(1)?;

        assert_eq!(tables.plus.shape(), &[2, 3, 2]);
        assert_eq!(tables.plus.strides(), &[1, 2, 6]);
        assert_eq!(tables.m_offset, 1);
        assert_close(tables.plus[[0, 1, 0]], 1.0);
        assert_close(tables.plus[[0, 1, 1]], 1.0);
        assert_close(tables.minus[[0, 1, 0]], 0.0);
        Ok(())
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
    }
}
