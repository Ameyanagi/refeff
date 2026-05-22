use super::*;

pub(in crate::atomic) fn validate_total_energy_input(
    input: &AtomicTotalEnergyInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_coefficient_table(
        input.coulomb_coefficients,
        orbital_count - 1,
        orbital_count - 1,
        0,
    )
}

pub(in crate::atomic) fn validate_differential_integral_input(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("dsordf_step", input.step)?;
    validate_finite_scalar("dsordf_origin_power", input.origin_power)?;
    if input.radii.is_empty() {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: 0,
            radial_count: 0,
        });
    }
    validate_positive_finite_radii(input.radii)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;

    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_radial_table_len(
        "derivative_large",
        input.radii.len(),
        input.derivative_large.len(),
    )?;
    validate_radial_table_len(
        "derivative_small",
        input.radii.len(),
        input.derivative_small.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_large_coefficients",
        coefficient_count,
        input.derivative_large_coefficients.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_small_coefficients",
        coefficient_count,
        input.derivative_small_coefficients.len(),
    )?;

    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("derivative_large", input.derivative_large)?;
    validate_finite_vector("derivative_small", input.derivative_small)?;
    validate_finite_vector(
        "derivative_large_coefficient",
        input.derivative_large_coefficients,
    )?;
    validate_finite_vector(
        "derivative_small_coefficient",
        input.derivative_small_coefficients,
    )?;

    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        }
        | AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        } => {
            let left = one_based_atomic_orbital_index(left_orbital_1based, orbital_count)?;
            let right = one_based_atomic_orbital_index(right_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[left].min(input.active_lengths[right]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => {
            let large = one_based_atomic_orbital_index(large_orbital_1based, orbital_count)?;
            let small = one_based_atomic_orbital_index(small_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[large].min(input.active_lengths[small]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            validate_differential_active_len(active_len, input.radii.len())?;
        }
    }
    Ok(())
}

pub(in crate::atomic) fn validate_yk_zk_transform_input(
    input: &AtomicYkZkTransformInput<'_>,
) -> Result<(), AtomMathError> {
    if input.active_len < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.active_len,
            radial_count: input.active_len,
        });
    }
    if input.source_len < 2 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.source_len,
            radial_count: input.active_len,
        });
    }
    validate_coefficient_count("source_coefficients", input.coefficient_count)?;
    validate_radial_table_len("source", input.active_len, input.source.len())?;
    validate_radial_table_len("radii", input.active_len, input.radii.len())?;
    validate_coefficient_vector_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_finite_scalar("yk_zk_initial_power", input.initial_power)?;
    validate_finite_scalar("yk_zk_step", input.step)?;
    if input.step == 0.0 {
        return Err(AtomMathError::ZeroYkZkDenominator { field: "step" });
    }
    if input.angular_momentum > i32::MAX as usize {
        return Err(AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        });
    }
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("yk_zk_source", input.source)?;
    validate_finite_vector("yk_zk_source_coefficient", input.source_coefficients)?;
    Ok(())
}

pub(in crate::atomic) fn validate_yk_zk_exchange_input(
    input: &AtomicYkZkExchangeInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("yk_zk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    let left = one_based_atomic_orbital_index(input.left_orbital_1based, orbital_count)?;
    let right = one_based_atomic_orbital_index(input.right_orbital_1based, orbital_count)?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_differential_active_len(
        input.active_lengths[left].min(input.active_lengths[right]),
        input.radii.len(),
    )?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

pub(in crate::atomic) fn validate_radial_integral_input(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("fdrirk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;

    if input.request.first_left > 0 && input.request.first_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.first_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.first_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
    }
    if input.request.second_left > 0 && input.request.second_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.second_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.second_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
        if (input.request.first_left == 0 || input.request.first_right == 0)
            && input.previous_first_factor.is_none()
        {
            return Err(AtomMathError::MissingRadialFirstFactor);
        }
    }
    if let Some(first_factor) = input.previous_first_factor {
        validate_radial_table_len(
            "previous_first_factor",
            input.radii.len(),
            first_factor.values.len(),
        )?;
        validate_coefficient_vector_len(
            "previous_first_factor_coefficients",
            coefficient_count,
            first_factor.coefficients.len(),
        )?;
        validate_finite_vector("previous_first_factor", first_factor.values)?;
        validate_finite_vector(
            "previous_first_factor_coefficient",
            first_factor.coefficients,
        )?;
        validate_finite_scalar(
            "previous_first_factor_origin_power",
            first_factor.origin_power,
        )?;
    }
    Ok(())
}

pub(in crate::atomic) fn validate_schmidt_orthogonalization_input(
    input: &AtomicSchmidtOrthogonalizationInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("active_lengths", orbital_count, input.active_lengths.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    if let Some(active_orbital_1based) = input.active_orbital_1based
        && !(1..=orbital_count).contains(&active_orbital_1based)
    {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based,
            orbital_count,
        });
    }

    let radial_rows = input.large_components.nrows();
    let coefficient_rows = input.large_coefficients.nrows();
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_rows,
        orbital_count,
    )?;

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_rows {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_rows,
            });
        }
    }
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

pub(in crate::atomic) fn validate_coulomb_coefficient_input(
    input: &AtomicCoulombCoefficientInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    Ok(())
}
