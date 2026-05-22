use super::*;

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
