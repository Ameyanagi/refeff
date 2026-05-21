use super::*;

/// Port FEFF DMDW `Lanczos` tridiagonal-recursion coefficients.
///
/// FEFF applies the dynamical matrix by taking dot products with matrix
/// columns. For symmetric DMDW matrices this is equivalent to the usual
/// matrix-vector product, but this helper preserves the exact column
/// convention. The seed is normalized with the same Euclidean norm used by FEFF
/// before recursion.
pub fn dmdw_lanczos_coefficients(
    dynamical_matrix: ArrayView2<'_, Real>,
    seed: ArrayView1<'_, Real>,
    pole_count: usize,
) -> Result<DmdwLanczosCoefficients, DebyeError> {
    validate_dmdw_lanczos_inputs(dynamical_matrix, seed, pole_count)?;

    let mut alpha = Array1::<Real>::zeros(pole_count + 1);
    let mut beta = Array1::<Real>::zeros(pole_count + 2);
    let mut qj = dmdw_normalize_seed_vector(seed)?;

    let applied = dmdw_apply_dynamical_matrix(dynamical_matrix, qj.view());
    let alpha0 = dot_array_views(qj.view(), applied.view());
    alpha[0] = alpha0;
    ensure_finite_output("DMDW Lanczos alpha", alpha0)?;
    let single_pole_frequency = alpha0.sqrt() / (2.0 * std::f64::consts::PI);
    ensure_finite_output("DMDW single-pole frequency", single_pole_frequency)?;

    let mut qp = lanczos_residual(applied, qj.view(), alpha0, None);
    beta[1] = array_vector_norm(qp.view());
    qp = normalize_lanczos_vector(qp, beta[1], 1)?;

    for iteration in 1..=pole_count {
        let qm = qj;
        qj = qp;
        let applied = dmdw_apply_dynamical_matrix(dynamical_matrix, qj.view());
        let alpha_i = dot_array_views(qj.view(), applied.view());
        alpha[iteration] = alpha_i;
        ensure_finite_output("DMDW Lanczos alpha", alpha_i)?;

        qp = lanczos_residual(
            applied,
            qj.view(),
            alpha_i,
            Some((beta[iteration], qm.view())),
        );
        beta[iteration + 1] = array_vector_norm(qp.view());
        qp = normalize_lanczos_vector(qp, beta[iteration + 1], iteration + 1)?;
    }

    Ok(DmdwLanczosCoefficients {
        alpha,
        beta,
        single_pole_frequency,
    })
}

/// Port FEFF DMDW `Poly_Y('S', ...)` for Lanczos pole locations.
pub fn dmdw_lanczos_s_polynomial(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut previous = 1.0;
    let mut value = x - alpha[0];
    for n in 2..=order {
        let older = previous;
        previous = value;
        value = (x - alpha[n - 1]) * previous - beta[n - 1].powi(2) * older;
    }
    ensure_finite_output("DMDW Lanczos S polynomial", value)?;
    Ok(value)
}

/// Port FEFF DMDW `Poly_Y('R', ...)`.
///
/// FEFF's `'P'` branch is identical to `'R'`; callers can use this function
/// for both recurrence variants.
pub fn dmdw_lanczos_r_polynomial(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut previous = 0.0;
    let mut value = 1.0;
    for n in 2..=order {
        let older = previous;
        previous = value;
        value = (x - alpha[n - 1]) * previous - beta[n - 1].powi(2) * older;
    }
    ensure_finite_output("DMDW Lanczos R polynomial", value)?;
    Ok(value)
}

/// Port FEFF DMDW `PolyD_Y('S', ...)`.
pub fn dmdw_lanczos_s_polynomial_derivative(
    order: usize,
    x: Real,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<Real, DebyeError> {
    validate_dmdw_lanczos_polynomial_inputs(order, x, alpha, beta)?;
    let mut y_previous_2 = 0.0;
    let mut y_previous_1 = 1.0;
    let mut derivative_previous_1 = 0.0;
    let mut derivative = 1.0;
    for n in 2..=order {
        let y_previous_3 = y_previous_2;
        y_previous_2 = y_previous_1;
        y_previous_1 = (x - alpha[n - 2]) * y_previous_2 - beta[n - 2].powi(2) * y_previous_3;
        let derivative_previous_2 = derivative_previous_1;
        derivative_previous_1 = derivative;
        derivative = y_previous_1 + (x - alpha[n - 1]) * derivative_previous_1
            - beta[n - 1].powi(2) * derivative_previous_2;
    }
    ensure_finite_output("DMDW Lanczos S polynomial derivative", derivative)?;
    Ok(derivative)
}

/// Port FEFF DMDW `Lanczos` pole search and `wil` weight calculation.
///
/// This uses FEFF's default scan range, `[-810000, 810000]`, and 100000 scan
/// samples per requested pole. The returned angular frequencies match FEFF
/// `w_pole`; the `frequencies` field is the `DW_Out%Poles_Frq` value after
/// division by `2*pi`.
pub fn dmdw_lanczos_pole_spectrum(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
) -> Result<DmdwLanczosPoleSpectrum, DebyeError> {
    dmdw_lanczos_pole_spectrum_with_search(
        order,
        alpha,
        beta,
        DMDW_LANCZOS_POLE_SEARCH_LIMIT,
        DMDW_LANCZOS_DEFAULT_SAMPLES_PER_POLE,
    )
}

/// Port FEFF DMDW `Lanczos` pole search with a configurable scan grid.
///
/// FEFF uses linear interpolation inside sign-changing grid intervals. This
/// helper keeps that behavior while exposing the grid for focused tests and
/// benchmarks.
pub fn dmdw_lanczos_pole_spectrum_with_search(
    order: usize,
    alpha: ArrayView1<'_, Real>,
    beta: ArrayView1<'_, Real>,
    search_limit: Real,
    samples_per_pole: usize,
) -> Result<DmdwLanczosPoleSpectrum, DebyeError> {
    validate_dmdw_lanczos_pole_search_inputs(order, alpha, beta, search_limit, samples_per_pole)?;
    let total_steps =
        order
            .checked_mul(samples_per_pole)
            .ok_or(DebyeError::DmdwLanczosPoleSearchTooLarge {
                order,
                samples_per_pole,
            })?;
    let step = 2.0 * search_limit / total_steps as Real;
    let mut roots = Vec::new();
    let mut previous_sample: Option<(Real, Real)> = None;

    for step_index in 1..=total_steps {
        let x = -search_limit + step * step_index as Real;
        let value = dmdw_lanczos_s_polynomial(order, x, alpha, beta)?;
        if let Some((previous_x, previous_value)) = previous_sample {
            if value == 0.0 {
                roots.push(x);
            } else if value * previous_value < 0.0 {
                let ratio = previous_value.abs() / (previous_value.abs() + value.abs());
                roots.push(ratio * (x - previous_x) + previous_x);
            }
        }
        previous_sample = Some((x, value));
    }

    let mut angular_frequencies = Vec::with_capacity(roots.len());
    let mut frequencies = Vec::with_capacity(roots.len());
    let mut weights = Vec::with_capacity(roots.len());
    let mut imaginary_warnings = Vec::new();

    for (pole_index, &root) in roots.iter().enumerate() {
        ensure_finite_output("DMDW Lanczos pole root", root)?;
        let angular_frequency = if root < 0.0 {
            -(-root).sqrt()
        } else {
            root.sqrt()
        };
        let frequency = angular_frequency / (2.0 * std::f64::consts::PI);
        let derivative = dmdw_lanczos_s_polynomial_derivative(order, root, alpha, beta)?;
        if derivative == 0.0 {
            return Err(DebyeError::ZeroDmdwLanczosPoleDerivative {
                pole_index,
                x: root,
            });
        }
        let weight = dmdw_lanczos_r_polynomial(order, root, alpha, beta)? / derivative;
        ensure_finite_output("DMDW Lanczos pole angular frequency", angular_frequency)?;
        ensure_finite_output("DMDW Lanczos pole frequency", frequency)?;
        ensure_finite_output("DMDW Lanczos pole weight", weight)?;

        if root < 0.0 {
            let severity = if weight >= DMDW_IMAGINARY_POLE_LARGE_WEIGHT {
                Some(DmdwImaginaryPoleSeverity::LargeWeight)
            } else if (DMDW_IMAGINARY_POLE_SMALL_WEIGHT..=DMDW_IMAGINARY_POLE_LARGE_WEIGHT)
                .contains(&weight)
            {
                Some(DmdwImaginaryPoleSeverity::SmallWeight)
            } else {
                None
            };
            if let Some(severity) = severity {
                imaginary_warnings.push(DmdwImaginaryPoleWarning {
                    pole_index,
                    squared_angular_frequency: root,
                    angular_frequency,
                    frequency,
                    weight,
                    severity,
                });
            }
        }

        angular_frequencies.push(angular_frequency);
        frequencies.push(frequency);
        weights.push(weight);
    }

    Ok(DmdwLanczosPoleSpectrum {
        expected_poles: order,
        squared_angular_frequencies: Array1::from_vec(roots),
        angular_frequencies: Array1::from_vec(angular_frequencies),
        frequencies: Array1::from_vec(frequencies),
        weights: Array1::from_vec(weights),
        imaginary_warnings,
    })
}
