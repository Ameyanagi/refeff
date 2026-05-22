use super::*;

/// Port of `FOVRG/yzktec.f90`: build the radial `yk` and `zk` exchange kernels.
///
/// FEFF evaluates
/// `zk(r) = r^-k * integral(0..r, f(u) * u^k du)` and then
/// `yk(r) = zk(r) + r^(k+1) * integral(r..infinity, f(u) * u^(-k-1) du)`.
/// The first integration runs outward on the logarithmic radial mesh and the
/// second runs backward from FEFF's clamped `np + 1` endpoint.
pub fn fovrg_yk_zk_transform(
    input: FovrgYkZkTransformInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len("source", input.active_len, input.source.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_complex_input("initial_power", 0, input.initial_power)?;
    validate_complex_input("tail_correction", 0, input.tail_correction)?;
    if input.angular_momentum > i32::MAX as usize - 1 {
        return Err(FovrgError::CountTooLarge {
            name: "angular_momentum",
            actual: input.angular_momentum,
            maximum: i32::MAX as usize - 1,
        });
    }

    let source_len = input.source_len.min(input.active_len - 1);
    let computed_len = source_len + 1;
    for row in 0..source_len {
        validate_complex_input("source", row, input.source[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "source_coefficients",
            coefficient,
            input.source_coefficients[coefficient],
        )?;
    }

    let k = input.angular_momentum;
    let k_real = k as Real;
    let k_i32 = k as i32;
    let k_plus_one_i32 = (k + 1) as i32;
    let singular_tolerance = 1.0e-5_f32 as Real;
    let mut yk = Array1::<Complex>::zeros(input.active_len);
    let mut zk = Array1::<Complex>::zeros(input.active_len);
    let mut yk_coefficients = Array1::<Complex>::from_iter(
        (0..input.coefficient_count).map(|row| input.source_coefficients[row]),
    );
    let mut zk_coefficients = Array1::<Complex>::zeros(input.coefficient_count);
    for row in 0..source_len {
        yk[row] = input.source[row];
    }

    let mut power = input.initial_power.re;
    let mut origin_constant = Complex::new(0.0, 0.0);
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_nonzero_denominator("yk_zk_origin_zk", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != Complex::new(0.0, 0.0) {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;

            let yk_denominator = power - k_real - 1.0;
            if yk_denominator.abs() <= singular_tolerance {
                yk_coefficients[coefficient] = Complex::new(0.0, 0.0);
                power -= 1.0;
            } else {
                yk_coefficients[coefficient] =
                    ((k + k + 1) as Real) * zk_coefficients[coefficient] / yk_denominator;
            }
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] *= input.radii[row];
    }

    let hk = input.step * k_real;
    let e = (-input.step).exp();
    let ehk = e.powi(k_i32);
    let b1 = if k == 0 {
        input.step / 2.0
    } else {
        (ehk - 1.0 + hk) / (hk * k_real)
    };
    let b0 = input.step - (1.0 + hk) * b1;
    for row in 0..source_len {
        zk[row + 1] = zk[row] * ehk + b0 * yk[row] + b1 * yk[row + 1];
    }

    yk[source_len] = zk[source_len] + input.tail_correction;
    let backward_ehk = ehk * e;
    let backward_hk = hk + input.step;
    let backward_order = (k + k + 1) as Real;
    let backward_b1 =
        backward_order * (backward_ehk - 1.0 + backward_hk) / (backward_hk * (k_real + 1.0));
    let backward_b0 = backward_order * input.step - (1.0 + backward_hk) * backward_b1;
    for row in (0..source_len).rev() {
        yk[row] = yk[row + 1] * backward_ehk + backward_b0 * zk[row + 1] + backward_b1 * zk[row];
    }

    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);
    validate_complex_result("origin_constant", 0, origin_constant)?;
    for row in 0..computed_len {
        validate_complex_result("yk", row, yk[row])?;
        validate_complex_result("zk", row, zk[row])?;
    }
    for row in 0..input.coefficient_count {
        validate_complex_result("yk_coefficients", row, yk_coefficients[row])?;
        validate_complex_result("zk_coefficients", row, zk_coefficients[row])?;
    }

    Ok(FovrgYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_len,
    })
}

/// Port of `FOVRG/yzkrdc.f90`: construct exchange source terms and `yk/zk`.
///
/// FEFF forms `f = cg_i * ps + cp_i * qs`, builds origin coefficients from the
/// products of the large/small development polynomials, and then delegates the
/// radial integrations to `yzktec`.
pub fn fovrg_yk_zk_exchange(
    input: FovrgYkZkExchangeInput<'_>,
) -> Result<FovrgYkZkTransform, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("source_len", input.source_len, 1)?;
    validate_count_at_least("orbital_len", input.orbital_len, 1)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len(
        "partner_large_component",
        input.active_len,
        input.partner_large_component.len(),
    )?;
    validate_active_len(
        "partner_small_component",
        input.active_len,
        input.partner_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_active_len(
        "partner_large_coefficients",
        input.coefficient_count,
        input.partner_large_coefficients.len(),
    )?;
    validate_active_len(
        "partner_small_coefficients",
        input.coefficient_count,
        input.partner_small_coefficients.len(),
    )?;
    validate_positive_finite("step", input.step)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_finite("partner_power", input.partner_power)?;

    let source_len = input
        .orbital_len
        .min(input.source_len)
        .min(input.active_len - 1);
    for row in 0..source_len {
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_complex_input(
            "partner_large_component",
            row,
            input.partner_large_component[row],
        )?;
        validate_complex_input(
            "partner_small_component",
            row,
            input.partner_small_component[row],
        )?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_large_coefficients",
            coefficient,
            input.partner_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "partner_small_coefficients",
            coefficient,
            input.partner_small_coefficients[coefficient],
        )?;
    }

    let source = Array1::from_iter((0..input.active_len).map(|row| {
        if row < source_len {
            input.large_component[row] * input.partner_large_component[row]
                + input.small_component[row] * input.partner_small_component[row]
        } else {
            Complex::new(0.0, 0.0)
        }
    }));
    let source_coefficients = Array1::from_iter((1..=input.coefficient_count).map(|count| {
        complex_real_product_coefficient(
            input.partner_large_coefficients,
            input.large_coefficients,
            count,
        ) + complex_real_product_coefficient(
            input.partner_small_coefficients,
            input.small_coefficients,
            count,
        )
    }));

    fovrg_yk_zk_transform(FovrgYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: Complex::new(input.orbital_power + input.partner_power, 0.0),
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len,
        active_len: input.active_len,
        tail_correction: Complex::new(0.0, 0.0),
    })
}

/// Port of `FOVRG/dsordc.f90`: complex radial overlap integral.
///
/// FEFF forms `hg = dg * cg_j + dp * cp_j`, integrates `hg(r) * r` over the
/// logarithmic radial mesh with its Simpson stencil, and adds the analytic
/// origin contribution from the product of the large/small development
/// coefficients.
pub fn fovrg_overlap_integral(input: FovrgOverlapIntegralInput<'_>) -> Result<Complex, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "large_integrand",
        input.active_len,
        input.large_integrand.len(),
    )?;
    validate_active_len(
        "small_integrand",
        input.active_len,
        input.small_integrand.len(),
    )?;
    validate_active_len(
        "large_component",
        input.active_len,
        input.large_component.len(),
    )?;
    validate_active_len(
        "small_component",
        input.active_len,
        input.small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "large_integrand_coefficients",
        input.coefficient_count,
        input.large_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "small_integrand_coefficients",
        input.coefficient_count,
        input.small_integrand_coefficients.len(),
    )?;
    validate_active_len(
        "large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_active_len(
        "small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite("integrand_power", input.integrand_power)?;
    validate_finite("orbital_power", input.orbital_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input("large_integrand", row, input.large_integrand[row])?;
        validate_complex_input("small_integrand", row, input.small_integrand[row])?;
        validate_real_input("large_component", row, input.large_component[row])?;
        validate_real_input("small_component", row, input.small_component[row])?;
        validate_radius(row, input.radii[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "large_integrand_coefficients",
            coefficient,
            input.large_integrand_coefficients[coefficient],
        )?;
        validate_complex_input(
            "small_integrand_coefficients",
            coefficient,
            input.small_integrand_coefficients[coefficient],
        )?;
        validate_real_input(
            "large_coefficients",
            coefficient,
            input.large_coefficients[coefficient],
        )?;
        validate_real_input(
            "small_coefficients",
            coefficient,
            input.small_coefficients[coefficient],
        )?;
    }

    let mixed_integrand = Array1::from_iter((0..input.active_len).map(|row| {
        (input.large_integrand[row] * input.large_component[row]
            + input.small_integrand[row] * input.small_component[row])
            * input.radii[row]
    }));

    let simpson_sum = (1..input.active_len - 1)
        .step_by(2)
        .fold(Complex::new(0.0, 0.0), |sum, row| {
            sum + mixed_integrand[row] + mixed_integrand[row] + mixed_integrand[row + 1]
        });
    let mut integral = input.step
        * (simpson_sum + simpson_sum + mixed_integrand[0] - mixed_integrand[input.active_len - 1])
        / 3.0;

    let mut origin_power = input.integrand_power + input.orbital_power;
    for coefficient in 1..=input.coefficient_count {
        origin_power += 1.0;
        validate_nonzero_denominator("overlap_origin_power", origin_power)?;
        let origin_coefficient = complex_real_product_coefficient(
            input.large_integrand_coefficients,
            input.large_coefficients,
            coefficient,
        ) + complex_real_product_coefficient(
            input.small_integrand_coefficients,
            input.small_coefficients,
            coefficient,
        );
        integral += origin_coefficient * input.radii[0].powf(origin_power) / origin_power;
    }
    validate_complex_result("overlap_integral", 0, integral)?;
    Ok(integral)
}

/// Port of `FOVRG/ortdac.f90`: Schmidt orthogonalization against bound orbitals.
///
/// FEFF walks the bound orbitals in order, skips orbitals whose kappa differs
/// from `ikap` or whose occupation is not positive, computes the current
/// overlap with `dsordc`, and subtracts that overlap from both the radial
/// target arrays and their origin development coefficients.
pub fn fovrg_schmidt_orthogonalize(
    input: FovrgOrthogonalizationInput<'_>,
) -> Result<FovrgOrthogonalization, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 3)?;
    validate_count_at_least("coefficient_count", input.coefficient_count, 1)?;
    if input.active_len.is_multiple_of(2) {
        return Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            actual: input.active_len,
        });
    }
    validate_active_len(
        "target_large_component",
        input.active_len,
        input.target_large_component.len(),
    )?;
    validate_active_len(
        "target_small_component",
        input.active_len,
        input.target_small_component.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_active_len(
        "target_large_coefficients",
        input.coefficient_count,
        input.target_large_coefficients.len(),
    )?;
    validate_active_len(
        "target_small_coefficients",
        input.coefficient_count,
        input.target_small_coefficients.len(),
    )?;
    validate_matrix_rows(
        "bound_large_components",
        input.active_len,
        input.bound_large_components.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_components",
        input.active_len,
        input.bound_small_components.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_components",
        input.bound_orbital_count,
        input.bound_large_components.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_components",
        input.bound_orbital_count,
        input.bound_small_components.shape()[1],
    )?;
    validate_matrix_rows(
        "bound_large_coefficients",
        input.coefficient_count,
        input.bound_large_coefficients.shape()[0],
    )?;
    validate_matrix_rows(
        "bound_small_coefficients",
        input.coefficient_count,
        input.bound_small_coefficients.shape()[0],
    )?;
    validate_matrix_cols(
        "bound_large_coefficients",
        input.bound_orbital_count,
        input.bound_large_coefficients.shape()[1],
    )?;
    validate_matrix_cols(
        "bound_small_coefficients",
        input.bound_orbital_count,
        input.bound_small_coefficients.shape()[1],
    )?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_active_len(
        "orbital_powers",
        input.bound_orbital_count,
        input.orbital_powers.len(),
    )?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;
    validate_finite("target_power", input.target_power)?;
    validate_positive_finite("step", input.step)?;

    for row in 0..input.active_len {
        validate_complex_input(
            "target_large_component",
            row,
            input.target_large_component[row],
        )?;
        validate_complex_input(
            "target_small_component",
            row,
            input.target_small_component[row],
        )?;
        validate_radius(row, input.radii[row])?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_components",
                row,
                input.bound_large_components[(row, orbital)],
            )?;
            validate_real_input(
                "bound_small_components",
                row,
                input.bound_small_components[(row, orbital)],
            )?;
        }
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_input(
            "target_large_coefficients",
            coefficient,
            input.target_large_coefficients[coefficient],
        )?;
        validate_complex_input(
            "target_small_coefficients",
            coefficient,
            input.target_small_coefficients[coefficient],
        )?;
        for orbital in 0..input.bound_orbital_count {
            validate_real_input(
                "bound_large_coefficients",
                coefficient,
                input.bound_large_coefficients[(coefficient, orbital)],
            )?;
            validate_real_input(
                "bound_small_coefficients",
                coefficient,
                input.bound_small_coefficients[(coefficient, orbital)],
            )?;
        }
    }
    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
        validate_real_input("orbital_powers", orbital, input.orbital_powers[orbital])?;
    }

    let mut large_component = input.target_large_component.to_owned();
    let mut small_component = input.target_small_component.to_owned();
    let mut large_coefficients = input.target_large_coefficients.to_owned();
    let mut small_coefficients = input.target_small_coefficients.to_owned();
    let mut overlaps = Array1::<Complex>::zeros(input.bound_orbital_count);

    for orbital in 0..input.bound_orbital_count {
        if input.kappa[orbital] != input.target_kappa || input.electron_counts[orbital] <= 0.0 {
            continue;
        }

        let overlap = fovrg_overlap_integral(FovrgOverlapIntegralInput {
            large_integrand: large_component.view(),
            small_integrand: small_component.view(),
            large_integrand_coefficients: large_coefficients.view(),
            small_integrand_coefficients: small_coefficients.view(),
            large_component: input.bound_large_components.column(orbital),
            small_component: input.bound_small_components.column(orbital),
            large_coefficients: input.bound_large_coefficients.column(orbital),
            small_coefficients: input.bound_small_coefficients.column(orbital),
            radii: input.radii,
            integrand_power: input.target_power,
            orbital_power: input.orbital_powers[orbital],
            step: input.step,
            coefficient_count: input.coefficient_count,
            active_len: input.active_len,
        })?;
        overlaps[orbital] = overlap;

        for row in 0..input.active_len {
            large_component[row] -= overlap * input.bound_large_components[(row, orbital)];
            small_component[row] -= overlap * input.bound_small_components[(row, orbital)];
        }
        for coefficient in 0..input.coefficient_count {
            large_coefficients[coefficient] -=
                overlap * input.bound_large_coefficients[(coefficient, orbital)];
            small_coefficients[coefficient] -=
                overlap * input.bound_small_coefficients[(coefficient, orbital)];
        }
    }

    for row in 0..input.active_len {
        validate_complex_result("large_component", row, large_component[row])?;
        validate_complex_result("small_component", row, small_component[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_complex_result(
            "large_coefficients",
            coefficient,
            large_coefficients[coefficient],
        )?;
        validate_complex_result(
            "small_coefficients",
            coefficient,
            small_coefficients[coefficient],
        )?;
    }
    for orbital in 0..input.bound_orbital_count {
        validate_complex_result("overlaps", orbital, overlaps[orbital])?;
    }

    Ok(FovrgOrthogonalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlaps,
    })
}

/// Port of `FOVRG/muatcc.f90`: angular coefficients for exchange coupling.
///
/// FEFF builds `afgkc(ikap, orbital, index)` for every target kappa. This
/// helper returns the single target-kappa row consumed by `potex`, indexed as
/// `(orbital, index)` with FEFF's fixed five coefficient slots.
pub fn fovrg_angular_coefficients(
    input: FovrgAngularCoefficientsInput<'_>,
) -> Result<RealMat, FovrgError> {
    validate_count_at_least("bound_orbital_count", input.bound_orbital_count, 1)?;
    validate_active_len(
        "electron_counts",
        input.bound_orbital_count,
        input.electron_counts.len(),
    )?;
    validate_active_len(
        "valence_counts",
        input.bound_orbital_count,
        input.valence_counts.len(),
    )?;
    validate_active_len("kappa", input.bound_orbital_count, input.kappa.len())?;
    validate_nonzero_kappa("target_kappa", 0, input.target_kappa)?;

    for orbital in 0..input.bound_orbital_count {
        validate_real_input("electron_counts", orbital, input.electron_counts[orbital])?;
        validate_real_input("valence_counts", orbital, input.valence_counts[orbital])?;
        validate_nonzero_kappa("kappa", orbital, input.kappa[orbital])?;
    }

    let target_j = target_j_value(input.target_kappa);
    let target_j_i32 = fovrg_usize_to_i32("target_j", target_j)?;
    let mut coefficients =
        Array2::<Real>::zeros((input.bound_orbital_count, FOVRG_ANGULAR_COEFFICIENT_SLOTS));

    for orbital in 0..input.bound_orbital_count {
        let bound_j = target_j_value(input.kappa[orbital]);
        let max_multipole = (target_j + bound_j) / 2;
        let mut min_multipole = target_j.abs_diff(bound_j) / 2;
        if (input.target_kappa < 0) != (input.kappa[orbital] < 0) {
            min_multipole += 1;
        }
        let required_slots = (max_multipole - min_multipole) / 2 + 1;
        if required_slots > FOVRG_ANGULAR_COEFFICIENT_SLOTS {
            return Err(FovrgError::CountTooLarge {
                name: "angular_coefficient_slots",
                actual: required_slots,
                maximum: FOVRG_ANGULAR_COEFFICIENT_SLOTS,
            });
        }
        if input.valence_counts[orbital] > 0.0 {
            continue;
        }

        let bound_j_i32 = fovrg_usize_to_i32("bound_j", bound_j)?;
        let mut multipole = min_multipole;
        while multipole <= max_multipole {
            let angular_index = (multipole - min_multipole) / 2;
            let doubled_multipole = multipole.checked_mul(2).ok_or(FovrgError::CountTooLarge {
                name: "doubled_multipole",
                actual: multipole,
                maximum: usize::MAX / 2,
            })?;
            let wigner = wigner_3j(
                target_j_i32,
                fovrg_usize_to_i32("doubled_multipole", doubled_multipole)?,
                bound_j_i32,
                1,
                0,
                2,
            )
            .map_err(|source| FovrgError::AngularCoefficient { source })?;
            let coefficient = input.electron_counts[orbital] * wigner * wigner;
            validate_real_result("angular_coefficients", orbital, coefficient)?;
            coefficients[(orbital, angular_index)] = coefficient;
            multipole += 2;
        }
    }

    Ok(coefficients)
}
