use super::*;

/// Port of `FOVRG/diff.f90`: C3 radial derivative term.
///
/// FEFF first differentiates `v(r) * r^2` with one-sided boundary stencils and
/// a centered fourth-order interior stencil, then returns
/// `(d(v*r^2)/dx - 2*v*r^2) / r * (kap+1) / cl`.
pub fn fovrg_c3_derivative(input: FovrgC3DerivativeInput<'_>) -> Result<ComplexVec, FovrgError> {
    validate_count_at_least("active_len", input.active_len, 8)?;
    validate_active_len("potential", input.active_len, input.potential.len())?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    validate_nonzero_finite("delta", input.delta)?;
    validate_nonzero_finite("speed_of_light", input.speed_of_light)?;

    for row in 0..input.active_len {
        validate_radius(row, input.radii[row])?;
        validate_potential(row, input.potential[row])?;
    }

    let vt = Array1::from_iter(
        (0..input.active_len).map(|row| input.potential[row] * input.radii[row].powi(2)),
    );
    let mut derivative = Array1::<Complex>::zeros(input.active_len);

    derivative[0] = ((F77_REAL_SIX * vt[1]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[3]
        + F77_REAL_ONE_POINT_TWO * vt[5])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[0]
            + F77_REAL_SEVEN_POINT_FIVE * vt[2]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[4]
            + F77_REAL_ONE_SIXTH * vt[6]))
        / input.delta;
    derivative[1] = ((F77_REAL_SIX * vt[2]
        + F77_REAL_SIX_AND_TWO_THIRDS * vt[4]
        + F77_REAL_ONE_POINT_TWO * vt[6])
        - (F77_REAL_TWO_POINT_FOUR_FIVE * vt[1]
            + F77_REAL_SEVEN_POINT_FIVE * vt[3]
            + F77_REAL_THREE_POINT_SEVEN_FIVE * vt[5]
            + F77_REAL_ONE_SIXTH * vt[7]))
        / input.delta;

    for row in 2..input.active_len - 2 {
        derivative[row] = ((vt[row - 2] + F77_REAL_EIGHT * vt[row + 1])
            - (F77_REAL_EIGHT * vt[row - 1] + vt[row + 2]))
            / F77_REAL_TWELVE
            / input.delta;
    }

    let last = input.active_len - 1;
    derivative[last - 1] = (vt[last] - vt[last - 2]) / (F77_REAL_TWO * input.delta);
    derivative[last] = (F77_REAL_HALF * vt[last - 2] - F77_REAL_TWO * vt[last - 1]
        + F77_REAL_ONE_POINT_FIVE * vt[last])
        / input.delta;

    let scale = ((input.kappa as f32 + 1.0_f32) as Real) / input.speed_of_light;
    let mut output = Array1::<Complex>::zeros(input.active_len);
    for row in 0..input.active_len {
        let value = (derivative[row] - F77_REAL_TWO * vt[row]) / input.radii[row] * scale;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(FovrgError::NonFiniteResult { row, value });
        }
        output[row] = value;
    }
    Ok(output)
}

/// Build the FEFF `dfovrg` C3 correction potential `vm`.
///
/// FEFF computes `diff` through the muffin-tin matching row and copies only the
/// rows before `jri` into `vm`; the matching row and interstitial tail stay
/// zero. This public helper exposes that source-backed array for callers that
/// need to drive `wfirdc` directly, such as XSPH `phiscf`.
pub fn fovrg_c3_potential(input: FovrgC3PotentialInput<'_>) -> Result<ComplexVec, FovrgError> {
    validate_active_len(
        "exchange_correlation_potential",
        input.active_len,
        input.exchange_correlation_potential.len(),
    )?;
    validate_active_len("radii", input.active_len, input.radii.len())?;
    let derivative_len =
        input
            .radial_match_index
            .checked_add(1)
            .ok_or(FovrgError::CountTooLarge {
                name: "radial_match_index",
                actual: input.radial_match_index,
                maximum: usize::MAX - 1,
            })?;
    if derivative_len > input.active_len {
        return Err(FovrgError::ActiveCountOutOfRange {
            field: "active_len",
            active_len: derivative_len,
            len: input.active_len,
        });
    }

    let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
        potential: input.exchange_correlation_potential,
        radii: input.radii,
        kappa: input.target_kappa,
        speed_of_light: FEFF_ALPHA_INVERSE,
        delta: input.step,
        active_len: derivative_len,
    })?;
    let mut c3_potential = Array1::<Complex>::zeros(input.active_len);
    for row in 0..input.radial_match_index {
        c3_potential[row] = derivative[row];
    }
    Ok(c3_potential)
}

/// Port of `FOVRG/aprdep.f90`: real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// two real origin-development polynomials.
pub fn fovrg_real_product_coefficient(
    left_coefficients: ArrayView1<'_, Real>,
    right_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Real, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "left_coefficients",
        coefficient_count,
        left_coefficients.len(),
    )?;
    validate_active_len(
        "right_coefficients",
        coefficient_count,
        right_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_real_input(
            "left_coefficients",
            coefficient,
            left_coefficients[coefficient],
        )?;
        validate_real_input(
            "right_coefficients",
            coefficient,
            right_coefficients[coefficient],
        )?;
    }

    let coefficient =
        real_product_coefficient(left_coefficients, right_coefficients, coefficient_count);
    validate_real_result("real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
}

/// Port of `FOVRG/aprdec.f90`: complex-real polynomial product coefficient.
///
/// Returns the coefficient for power `coefficient_count - 1` in the product of
/// a complex origin-development polynomial and a real one.
pub fn fovrg_complex_real_product_coefficient(
    complex_coefficients: ArrayView1<'_, Complex>,
    real_coefficients: ArrayView1<'_, Real>,
    coefficient_count: usize,
) -> Result<Complex, FovrgError> {
    validate_count_at_least("coefficient_count", coefficient_count, 1)?;
    validate_active_len(
        "complex_coefficients",
        coefficient_count,
        complex_coefficients.len(),
    )?;
    validate_active_len(
        "real_coefficients",
        coefficient_count,
        real_coefficients.len(),
    )?;
    for coefficient in 0..coefficient_count {
        validate_complex_input(
            "complex_coefficients",
            coefficient,
            complex_coefficients[coefficient],
        )?;
        validate_real_input(
            "real_coefficients",
            coefficient,
            real_coefficients[coefficient],
        )?;
    }

    let coefficient = complex_real_product_coefficient(
        complex_coefficients,
        real_coefficients,
        coefficient_count,
    );
    validate_complex_result("complex_real_product_coefficient", 0, coefficient)?;
    Ok(coefficient)
}
