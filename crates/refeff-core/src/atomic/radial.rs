use super::*;

/// Port of FEFF `ATOM/dsordf.f90`.
///
/// This evaluates FEFF's Simpson-rule radial integral and adds the analytic
/// origin-development correction from `aprdev`. The supported modes cover the
/// `jnd = -2, -1, 1, 2, 3, 4` branches used by FEFF10 `ATOM`.
pub fn atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    validate_differential_integral_input(&input)?;
    calculate_atomic_differential_integral(input)
}

/// Port of FEFF `ATOM/yzkteg.f90`.
///
/// This builds the radial `yk` and `zk` Coulomb kernels using FEFF's four-point
/// point-to-point integration stencil and origin-development correction.
pub fn atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_transform_input(&input)?;
    calculate_atomic_yk_zk_transform(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for positive orbital indices.
///
/// This constructs the source and origin coefficients from ATOM orbital
/// component tables, then delegates to [`atomic_yk_zk_transform`].
pub fn atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_exchange_input(&input)?;
    calculate_atomic_yk_zk_exchange(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for caller-prepared sources.
///
/// This mirrors the `i <= 0` branch, where the caller supplies the source and
/// coefficients directly and FEFF uses `k + 2` as the first origin power before
/// delegating to `yzkteg`.
pub fn atomic_yk_zk_prepared_source(
    input: AtomicYkZkPreparedSourceInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let initial_power = input.angular_momentum.checked_add(2).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )? as Real;

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: input.source,
        source_coefficients: input.source_coefficients,
        radii: input.radii,
        initial_power,
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len: input.source_len,
        active_len: input.active_len,
    })
}

/// Port of FEFF `ATOM/fdrirk.f90`.
///
/// This composes the `yzkrdf` first-factor construction with the `dsordf`
/// radial integration branch. When the first pair in the request is zero, the
/// caller must pass the previous first factor to mirror FEFF's common-block
/// sentinel path.
pub fn atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    validate_radial_integral_input(&input)?;
    calculate_atomic_radial_integral(input)
}

/// Port of FEFF `ATOM/potslw.f90`.
///
/// FEFF uses this four-point integration stencil to convert a logarithmic-grid
/// radial source into a Coulomb potential, for example when building the
/// transition-state core-hole potential in `ATOM/apot.f90`.
pub fn atomic_four_point_coulomb_potential(
    input: AtomicFourPointCoulombPotentialInput<'_>,
) -> Result<Array1<Real>, AtomMathError> {
    validate_four_point_coulomb_potential_input(input)?;

    let count = input.active_len;
    let step_weight = input.step / 24.0;
    let step_exp = input.step.exp();
    let step_exp_squared = step_exp * step_exp;

    let mut potential =
        Array1::from_shape_fn(count, |index| input.density[index] * input.radii[index]);
    let mut work = Array1::<Real>::zeros(count);

    work[1] = input.radii[0] * (input.density[1] - input.density[0] * step_exp_squared)
        / (12.0 * (step_exp - 1.0));
    work[0] = potential[0] / 3.0 - work[1] / step_exp_squared;
    work[1] = potential[1] / 3.0 - work[1] * step_exp_squared;

    let penultimate = count - 2;
    if penultimate >= 2 {
        for index in 2..=penultimate {
            work[index] = work[index - 1]
                + step_weight
                    * (13.0 * (potential[index] + potential[index - 1])
                        - (potential[index - 2] + potential[index + 1]));
        }
    }

    let tail = work[penultimate];
    work[count - 1] = tail;
    potential[penultimate] = tail;
    potential[count - 1] = tail;

    for index_1based in 3..=penultimate + 1 {
        let row = count - index_1based;
        potential[row] = potential[row + 1] / step_exp
            + step_weight
                * (13.0 * (work[row + 1] / step_exp + work[row])
                    - (work[row + 2] / step_exp_squared + work[row - 1] * step_exp));
    }

    potential[0] = potential[2] / step_exp_squared
        + input.step * (work[0] + 4.0 * work[1] / step_exp + work[2] / step_exp_squared) / 3.0;
    for index in 0..count {
        potential[index] /= input.radii[index];
        validate_finite_scalar("potslw_potential", potential[index])?;
    }

    Ok(potential)
}

pub(super) struct AtomicDifferentialIntegralWork {
    values: Vec<Real>,
    coefficients: Vec<Real>,
    origin_power: Real,
}

fn validate_four_point_coulomb_potential_input(
    input: AtomicFourPointCoulombPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("potslw_step", input.step)?;
    if input.active_len < 3 {
        return Err(AtomMathError::InvalidCount {
            field: "potslw_active_len",
            minimum: 3,
            actual: input.active_len,
        });
    }
    if input.density.len() < input.active_len {
        return Err(AtomMathError::RadialTableLengthMismatch {
            table: "potslw_density",
            expected_len: input.active_len,
            actual_len: input.density.len(),
        });
    }
    if input.radii.len() < input.active_len {
        return Err(AtomMathError::RadialTableLengthMismatch {
            table: "potslw_radii",
            expected_len: input.active_len,
            actual_len: input.radii.len(),
        });
    }
    for index in 0..input.active_len {
        validate_finite_scalar("potslw_density", input.density[index])?;
        let radius = input.radii[index];
        validate_finite_scalar("radius", radius)?;
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
    }
    Ok(())
}

pub(super) fn calculate_atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    let work = atomic_differential_integral_work(&input)?;
    integrate_atomic_differential_work(&input, work)
}

pub(super) fn atomic_differential_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            false,
        ),
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            true,
        ),
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => atomic_derivative_projection_integral_work(
            input,
            large_orbital_1based,
            small_orbital_1based,
        ),
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            atomic_derivative_norm_integral_work(input, active_len)
        }
    }
}

pub(super) fn atomic_component_overlap_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    left_orbital_1based: usize,
    right_orbital_1based: usize,
    multiply_by_derivative: bool,
    large_small: bool,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let left = one_based_atomic_orbital_index(left_orbital_1based, input.active_lengths.len())?;
    let right = one_based_atomic_orbital_index(right_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[left].min(input.active_lengths[right]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            let base = if large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            };
            if multiply_by_derivative {
                base * input.derivative_large[row]
            } else {
                base
            }
        })
        .collect::<Vec<_>>();

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let mut coefficients = (1..=coefficient_count)
        .map(|term| {
            if large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut origin_power = input.orbital_powers[left] + input.orbital_powers[right];
    if multiply_by_derivative {
        origin_power += input.origin_power;
        coefficients = (1..=coefficient_count)
            .map(|term| {
                polynomial_product_coefficient_slice_view(
                    &coefficients,
                    input.derivative_large_coefficients,
                    term,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

pub(super) fn atomic_derivative_projection_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    large_orbital_1based: usize,
    small_orbital_1based: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let large_orbital =
        one_based_atomic_orbital_index(large_orbital_1based, input.active_lengths.len())?;
    let small_orbital =
        one_based_atomic_orbital_index(small_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[large_orbital].min(input.active_lengths[small_orbital]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.large_components[(row, large_orbital)]
                + input.derivative_small[row] * input.small_components[(row, small_orbital)]
        })
        .collect::<Vec<_>>();
    let large_coefficients = input.large_coefficients.index_axis(Axis(1), large_orbital);
    let small_coefficients = input.small_coefficients.index_axis(Axis(1), small_orbital);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.orbital_powers[large_orbital];
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

pub(super) fn atomic_derivative_norm_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    active_len: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.derivative_large[row]
                + input.derivative_small[row] * input.derivative_small[row]
        })
        .collect::<Vec<_>>();
    let coefficient_count = input.derivative_large_coefficients.len();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                input.derivative_large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                input.derivative_small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.origin_power;
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

pub(super) fn integrate_atomic_differential_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    work: AtomicDifferentialIntegralWork,
) -> Result<Real, AtomMathError> {
    let radial_power = input
        .power
        .checked_add(1)
        .ok_or(AtomMathError::DifferentialIntegralPowerOutOfRange { power: input.power })?;
    let scaled = work
        .values
        .iter()
        .enumerate()
        .map(|(row, &value)| value * input.radii[row].powi(radial_power))
        .collect::<Vec<_>>();
    validate_finite_slice("dsordf_integrand", &scaled)?;

    let active_len = scaled.len();
    let mut integral = 0.0;
    let mut row_1based = 2;
    while row_1based < active_len {
        let row = row_1based - 1;
        integral += scaled[row] + scaled[row] + scaled[row + 1];
        row_1based += 2;
    }
    integral = input.step * (integral + integral + scaled[0] - scaled[active_len - 1]) / 3.0;

    let mut origin_exponent = work.origin_power + Real::from(input.power);
    validate_finite_scalar("dsordf_origin_exponent", origin_exponent)?;
    for coefficient in work.coefficients {
        origin_exponent += 1.0;
        if origin_exponent == 0.0 {
            return Err(AtomMathError::ZeroDifferentialIntegralOriginExponent);
        }
        let correction = coefficient * input.radii[0].powf(origin_exponent) / origin_exponent;
        validate_finite_scalar("dsordf_origin_correction", correction)?;
        integral += correction;
    }
    validate_finite_scalar("dsordf_integral", integral)?;
    Ok(integral)
}

pub(super) fn calculate_atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let source_len = input.source_len.min(input.active_len - 2);
    let k_i32 = i32::try_from(input.angular_momentum).map_err(|_| {
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        }
    })?;
    let k_plus_one = input.angular_momentum.checked_add(1).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )?;
    let k_plus_one_i32 =
        i32::try_from(k_plus_one).map_err(|_| AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        })?;
    let k_real = input.angular_momentum as Real;
    let order = (input.angular_momentum + input.angular_momentum + 1) as Real;

    let mut yk = Array1::<Real>::zeros(input.active_len);
    let mut zk = Array1::<Real>::zeros(input.active_len);
    let mut yk_coefficients = Array1::from_iter(
        (0..input.coefficient_count).map(|coefficient| input.source_coefficients[coefficient]),
    );
    let mut zk_coefficients = Array1::<Real>::zeros(input.coefficient_count);

    let mut power = input.initial_power;
    let mut origin_constant = 0.0;
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_yk_zk_denominator("zk_origin", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != 0.0 {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;
            zk[1] += zk_coefficients[coefficient] * input.radii[1].powf(power);

            let yk_denominator = power - k_real - 1.0;
            validate_yk_zk_denominator("yk_origin", yk_denominator)?;
            yk_coefficients[coefficient] = order * zk_coefficients[coefficient] / yk_denominator;
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] = input.source[row] * input.radii[row];
    }
    yk[source_len] = 0.0;
    yk[source_len + 1] = 0.0;

    let step_exp = input.step.exp();
    let attenuation = step_exp.powi(-k_i32);
    let base_weight = input.step / 24.0;
    let middle_weight = 13.0 * base_weight;
    let leading_weight = attenuation * attenuation * base_weight;
    let trailing_weight = base_weight / attenuation;

    for row_1based in 3..=(source_len + 1) {
        let row = row_1based - 1;
        zk[row] = zk[row - 1] * attenuation
            + (middle_weight * (yk[row] + yk[row - 1] * attenuation)
                - (yk[row - 2] * leading_weight + yk[row + 1] * trailing_weight));
    }

    yk[source_len - 1] = zk[source_len - 1];
    for row in source_len..input.active_len {
        yk[row] = yk[row - 1] * attenuation;
    }

    let backward_trailing_weight = order * trailing_weight * step_exp;
    let backward_leading_weight = order * leading_weight / (step_exp * step_exp);
    let backward_attenuation = attenuation / step_exp;
    let backward_middle_weight = order * middle_weight;
    for row_1based in (2..source_len).rev() {
        let row = row_1based - 1;
        yk[row] = yk[row + 1] * backward_attenuation
            + (backward_middle_weight * (zk[row] + zk[row + 1] * backward_attenuation)
                - (zk[row + 2] * backward_leading_weight + zk[row - 1] * backward_trailing_weight));
    }

    let attenuation_squared = backward_attenuation * backward_attenuation;
    let first_weight = 8.0 * backward_middle_weight / 13.0;
    yk[0] = yk[2] * attenuation_squared
        + first_weight * (zk[2] * attenuation_squared + 4.0 * backward_attenuation * zk[1] + zk[0]);
    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);

    validate_finite_scalar("yk_zk_origin_constant", origin_constant)?;
    for row in 0..input.active_len {
        validate_finite_scalar("yk", yk[row])?;
        validate_finite_scalar("zk", zk[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_finite_scalar("yk_coefficient", yk_coefficients[coefficient])?;
        validate_finite_scalar("zk_coefficient", zk_coefficients[coefficient])?;
    }

    Ok(AtomicYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_source_len: source_len,
    })
}

pub(super) fn validate_yk_zk_denominator(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value == 0.0 {
        Err(AtomMathError::ZeroYkZkDenominator { field })
    } else {
        validate_finite_scalar(field, value)
    }
}

pub(super) fn calculate_atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let left =
        one_based_atomic_orbital_index(input.left_orbital_1based, input.active_lengths.len())?;
    let right =
        one_based_atomic_orbital_index(input.right_orbital_1based, input.active_lengths.len())?;
    let source_len = input.active_lengths[left].min(input.active_lengths[right]);
    let source = Array1::from_shape_fn(input.radii.len(), |row| {
        if row < source_len {
            if input.large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            }
        } else {
            0.0
        }
    });

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            if input.large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_coefficients = Array1::from_vec(coefficients);

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: input.orbital_powers[left] + input.orbital_powers[right],
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count,
        source_len,
        active_len: input.radii.len(),
    })
}

pub(super) fn calculate_atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    let first_factor = if input.request.first_left > 0 && input.request.first_right > 0 {
        Some(calculate_atomic_radial_first_factor(&input)?)
    } else {
        None
    };

    if input.request.second_left == 0 || input.request.second_right == 0 {
        return Ok(AtomicRadialIntegral {
            value: 0.0,
            first_factor,
        });
    }

    let value = match first_factor.as_ref() {
        Some(first_factor) => {
            calculate_atomic_radial_second_integral(&input, first_factor.as_view())?
        }
        None => calculate_atomic_radial_second_integral(
            &input,
            input
                .previous_first_factor
                .ok_or(AtomMathError::MissingRadialFirstFactor)?,
        )?,
    };

    Ok(AtomicRadialIntegral {
        value,
        first_factor,
    })
}

pub(super) fn calculate_atomic_radial_second_integral(
    input: &AtomicRadialIntegralInput<'_>,
    first_factor: AtomicRadialFirstFactorView<'_>,
) -> Result<Real, AtomMathError> {
    let zero_derivative = Array1::<Real>::zeros(input.radii.len());
    let zero_coefficients = Array1::<Real>::zeros(input.large_coefficients.nrows());
    let kind = if input.large_small {
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    } else {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    };
    atomic_differential_integral(AtomicDifferentialIntegralInput {
        kind,
        power: -1,
        origin_power: first_factor.origin_power,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
        derivative_large: first_factor.values,
        derivative_small: zero_derivative.view(),
        derivative_large_coefficients: first_factor.coefficients,
        derivative_small_coefficients: zero_coefficients.view(),
    })
}

pub(super) fn calculate_atomic_radial_first_factor(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialFirstFactor, AtomMathError> {
    let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
        left_orbital_1based: input.request.first_left,
        right_orbital_1based: input.request.first_right,
        large_small: input.large_small,
        angular_momentum: input.request.rank,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
    })?;

    let left = one_based_atomic_orbital_index(input.request.first_left, input.kappas.len())?;
    let right = one_based_atomic_orbital_index(input.request.first_right, input.kappas.len())?;
    let left_abs = abs_kappa_i32(input.kappas[left])? as usize;
    let right_abs = abs_kappa_i32(input.kappas[right])? as usize;
    let abs_sum = left_abs
        .checked_add(right_abs)
        .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
    let shifted_start = abs_sum.saturating_sub(input.request.rank).max(1);

    let coefficient_count = transform.yk_coefficients.len();
    let mut coefficients = Array1::<Real>::zeros(coefficient_count);
    for (source, coefficient) in transform.yk_coefficients.iter().copied().enumerate() {
        let target_1based = shifted_start
            .checked_add(source)
            .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
        if target_1based <= coefficient_count {
            coefficients[target_1based - 1] = -coefficient;
        }
    }
    coefficients[0] += transform.origin_constant;

    let origin_power = (input.request.rank as Real) + 1.0;
    validate_finite_scalar("radial_first_factor_origin_power", origin_power)?;
    validate_finite_vector("radial_first_factor", transform.yk.view())?;
    validate_finite_vector("radial_first_factor_coefficient", coefficients.view())?;

    Ok(AtomicRadialFirstFactor {
        values: transform.yk,
        coefficients,
        origin_power,
    })
}
