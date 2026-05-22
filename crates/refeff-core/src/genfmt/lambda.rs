use super::*;

/// Build FEFF `mlam` and `nlam` arrays from `GENFMT/setlam.f90` rules.
///
/// The returned arrays preserve FEFF's insertion order, including `-m` before
/// `+m`, and then apply FEFF's second pass that moves entries satisfying
/// `n <= ilinit && abs(m) <= ilinit` to the front to minimize `laml0x`.
/// Capacity handling also follows FEFF: if `lamtot` fills, the result is
/// truncated and flagged instead of failing.
pub fn lambda_indices(input: LambdaIndexInput<'_>) -> Result<LambdaIndexSet, GenfmtError> {
    let request = lambda_request(input)?;
    let mut raw = Vec::with_capacity(input.lambda_capacity.min(128));
    let mut truncated = false;

    if request.order >= 0 {
        let order = usize::try_from(request.order).map_err(|_| GenfmtError::IntegerOverflow {
            field: "iord",
            value: request.order.unsigned_abs() as usize,
        })?;
        let valid_n_max = request.n_max.min(order / 2);

        'outer: for n in 0..=valid_n_max {
            let valid_m_max = request.m_max.min(order - 2 * n);
            for m in 0..=valid_m_max {
                if raw.len() >= input.lambda_capacity {
                    truncated = true;
                    break 'outer;
                }
                raw.push((-checked_i32("m", m)?, checked_i32("n", n)?));

                if m != 0 {
                    if raw.len() >= input.lambda_capacity {
                        truncated = true;
                        break 'outer;
                    }
                    raw.push((checked_i32("m", m)?, checked_i32("n", n)?));
                }
            }
        }
    }

    let mut pairs = Vec::with_capacity(raw.len());
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| within_initial_l(m, n, input.initial_l)),
    );
    let initial_l_prefix_len = pairs.len();
    pairs.extend(
        raw.iter()
            .copied()
            .filter(|&(m, n)| !within_initial_l(m, n, input.initial_l)),
    );

    let max_m_plus_one = max_lambda_m_plus_one(&pairs)?;
    let max_n = max_lambda_n(&pairs)?;

    if max_n > input.max_n || max_m_plus_one > input.max_m.saturating_add(1) {
        return Err(GenfmtError::DimensionExceeded {
            max_m_plus_one,
            max_n,
            max_m: input.max_m,
            max_n_limit: input.max_n,
        });
    }

    let (m_values, n_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok(LambdaIndexSet {
        m_indices: Array1::from_vec(m_values),
        n_indices: Array1::from_vec(n_values),
        initial_l_prefix_len,
        max_m_plus_one,
        max_n,
        order: request.order,
        requested_n_max: request.n_max,
        requested_m_max: request.m_max,
        truncated,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LambdaRequest {
    order: i32,
    n_max: usize,
    m_max: usize,
}

fn lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    if input.calculation < 0 {
        return decode_lambda_request(input.calculation);
    }

    if input.scattering_count == 1 {
        return Ok(LambdaRequest {
            order: checked_order(input.initial_l, input.initial_l)?,
            n_max: input.initial_l,
            m_max: input.initial_l,
        });
    }

    if input.calculation < 10 {
        let order = input.calculation;
        return Ok(LambdaRequest {
            order,
            n_max: usize::try_from(order / 2).map_err(|_| GenfmtError::IntegerOverflow {
                field: "nmax",
                value: order.unsigned_abs() as usize,
            })?,
            m_max: usize::try_from(order).map_err(|_| GenfmtError::IntegerOverflow {
                field: "mmax",
                value: order.unsigned_abs() as usize,
            })?,
        });
    }

    if input.calculation == 10 {
        return cute_lambda_request(input);
    }

    Err(GenfmtError::UndefinedLambdaCalculation {
        calculation: input.calculation,
    })
}

fn decode_lambda_request(calculation: i32) -> Result<LambdaRequest, GenfmtError> {
    let code = calculation
        .checked_neg()
        .ok_or(GenfmtError::LambdaCodeOverflow { calculation })?;
    let order = (code / 10_000) - 1;
    Ok(LambdaRequest {
        order,
        n_max: usize::try_from(code % 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
        m_max: usize::try_from((code % 10_000) / 100)
            .map_err(|_| GenfmtError::LambdaCodeOverflow { calculation })?,
    })
}

fn cute_lambda_request(input: LambdaIndexInput<'_>) -> Result<LambdaRequest, GenfmtError> {
    let mut m_max = input.initial_l;
    for (index, &angle) in input.beta_angles.iter().enumerate() {
        if !angle.is_finite() {
            return Err(GenfmtError::NonFiniteBetaAngle {
                index,
                value: angle,
            });
        }
        let magnitude = angle.abs();
        let pi_distance = (magnitude - std::f64::consts::PI).abs();
        if magnitude > ONE_DEGREE_RADIANS && pi_distance > ONE_DEGREE_RADIANS {
            m_max = 3;
        }
    }

    let n_max = if input.energy_index >= 42 {
        9
    } else {
        input.initial_l
    };

    Ok(LambdaRequest {
        order: checked_order(n_max, m_max)?,
        n_max,
        m_max,
    })
}

fn checked_order(n_max: usize, m_max: usize) -> Result<i32, GenfmtError> {
    let order = n_max
        .checked_mul(2)
        .and_then(|value| value.checked_add(m_max))
        .ok_or(GenfmtError::IntegerOverflow {
            field: "iord",
            value: n_max,
        })?;
    checked_i32("iord", order)
}

pub(super) fn checked_i32(field: &'static str, value: usize) -> Result<i32, GenfmtError> {
    i32::try_from(value).map_err(|_| GenfmtError::IntegerOverflow { field, value })
}

fn within_initial_l(m: i32, n: i32, initial_l: usize) -> bool {
    let abs_m = m.unsigned_abs() as usize;
    let Ok(n) = usize::try_from(n) else {
        return false;
    };
    n <= initial_l && abs_m <= initial_l
}

fn max_lambda_m_plus_one(pairs: &[(i32, i32)]) -> Result<usize, GenfmtError> {
    pairs.iter().try_fold(0, |maximum, &(m, _)| {
        if m < 0 {
            return Ok(maximum);
        }
        let plus_one = m.checked_add(1).ok_or(GenfmtError::IntegerOverflow {
            field: "mmaxp1",
            value: m.unsigned_abs() as usize,
        })?;
        let value = usize::try_from(plus_one).map_err(|_| GenfmtError::IntegerOverflow {
            field: "mmaxp1",
            value: m.unsigned_abs() as usize,
        })?;
        Ok(maximum.max(value))
    })
}

fn max_lambda_n(pairs: &[(i32, i32)]) -> Result<usize, GenfmtError> {
    pairs.iter().try_fold(0, |maximum, &(_, n)| {
        if n < 0 {
            return Ok(maximum);
        }
        let value = usize::try_from(n).map_err(|_| GenfmtError::IntegerOverflow {
            field: "nmax",
            value: n.unsigned_abs() as usize,
        })?;
        Ok(maximum.max(value))
    })
}
