use super::*;

pub(super) fn validate_total_energy_input(
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

pub(super) fn validate_differential_integral_input(
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

pub(super) fn validate_yk_zk_transform_input(
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

pub(super) fn validate_yk_zk_exchange_input(
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

pub(super) fn validate_radial_integral_input(
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

pub(super) fn validate_schmidt_orthogonalization_input(
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

pub(super) fn validate_coulomb_coefficient_input(
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

pub(super) fn validate_orbital_initialization_input(
    input: &AtomicOrbitalInitializationInput<'_>,
) -> Result<(), AtomMathError> {
    if input.atomic_number == 0 {
        return Err(AtomMathError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    validate_finite_scalar("inmuat_ionicity", input.ionicity)?;
    let orbital_count = input.occupations.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_finite_slice("occupation", input.occupations)?;
    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_abs_usize(kappa)?;
    }
    let actual = input.occupations.iter().copied().sum::<Real>();
    let expected = input.atomic_number as Real - input.ionicity;
    if (expected - actual).abs() > ATOM_INMUAT_ELECTRON_TOLERANCE {
        return Err(AtomMathError::ElectronCountMismatch {
            atomic_number: input.atomic_number,
            ionicity: input.ionicity,
            expected,
            actual,
            tolerance: ATOM_INMUAT_ELECTRON_TOLERANCE,
        });
    }
    Ok(())
}

pub(super) fn validate_dirac_entry_state_input(
    input: &AtomicDiracEntryStateInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_entry_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_normalization_input(
    input: &AtomicDiracNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_step", input.step)?;
    validate_finite_scalar(
        "soldir_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar("soldir_origin_power", input.origin_power)?;
    validate_dirac_normalization_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_large_component", input.large_component)?;
    validate_finite_vector("soldir_small_component", input.small_component)?;
    validate_coefficient_count("soldir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("soldir_small_coefficient", input.small_coefficients)?;
    if input.method == 1
        && (input.matching_index_1based == 0 || input.matching_index_1based > input.active_len)
    {
        return Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange {
            matching_index_1based: input.matching_index_1based,
            active_len: input.active_len,
        });
    }
    Ok(())
}

pub(super) fn validate_dirac_solution_normalization_input(
    input: &AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_solution_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_solution_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_solution_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_dirac_solution_normalization_active_len(
        input.active_len,
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_solution_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_solution_large_component", input.large_component)?;
    validate_finite_vector("soldir_solution_small_component", input.small_component)?;
    validate_coefficient_count(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_solution_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_solution_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_node_count_input(
    input: &AtomicDiracNodeCountInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector("soldir_node_large_component", input.large_component)?;
    validate_dirac_node_count_index(
        "matching",
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    validate_dirac_node_count_index("scan", input.scan_index_1based, input.large_component.len())?;
    Ok(())
}

pub(super) fn validate_dirac_node_energy_search_input(
    input: &AtomicDiracNodeEnergySearchInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_node_search_energy", input.energy)?;
    validate_finite_scalar("soldir_node_search_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_node_search_einf", input.energy_inf)?;
    validate_finite_scalar("soldir_node_search_emin", input.energy_floor)?;
    validate_positive_finite_scalar("soldir_node_search_precision", input.energy_precision)?;
    Ok(())
}

pub(super) fn validate_dirac_iteration_reset_input(
    input: &AtomicDiracIterationResetInput,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar(
        "soldir_iteration_primary_precision",
        input.primary_matching_precision,
    )?;
    validate_positive_finite_scalar(
        "soldir_iteration_secondary_precision",
        input.secondary_matching_precision,
    )?;
    validate_finite_scalar("soldir_iteration_energy_floor", input.energy_floor)?;
    validate_finite_scalar("soldir_iteration_reference_energy", input.reference_energy)?;
    Ok(())
}

pub(super) fn validate_dirac_method_one_energy_correction_input(
    input: &AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("soldir_energy_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_matching_small_component",
        input.matching_small_component,
    )?;
    validate_radial_table_len(
        "soldir_energy_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_energy_large_component", input.large_component)?;
    validate_finite_vector("soldir_energy_small_component", input.small_component)?;
    validate_dirac_energy_correction_matching_index(
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    Ok(())
}

pub(super) fn validate_dirac_energy_step_input(
    input: &AtomicDiracEnergyStepInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_energy_step_energy", input.energy)?;
    validate_finite_scalar("soldir_energy_step_correction", input.correction)?;
    validate_finite_scalar("soldir_energy_step_mismatch", input.mismatch)?;
    validate_finite_scalar("soldir_energy_step_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_energy_step_einf", input.energy_inf)?;
    validate_positive_finite_scalar(
        "soldir_energy_step_mismatch_precision",
        input.mismatch_precision,
    )?;
    validate_positive_finite_scalar(
        "soldir_energy_step_zero_precision",
        input.zero_energy_precision,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_rematch_attempt_input(
    input: &AtomicDiracRematchAttemptInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_rematch_mismatch", input.mismatch)?;
    validate_positive_finite_scalar(
        "soldir_rematch_mismatch_precision",
        input.mismatch_precision,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_large_component_match_input(
    input: &AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_finite_vector("soldir_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_homogeneous_match_input(
    input: &AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_homogeneous_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_homogeneous_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_small_component",
        input.small_component,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_two_component_match_input(
    input: &AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_two_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_large_component",
        input.homogeneous_matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_small_component",
        input.homogeneous_matching_small_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_two_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_coefficient_count(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_large_coefficients",
        input.coefficient_count,
        input.homogeneous_large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_small_coefficients",
        input.coefficient_count,
        input.homogeneous_small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_two_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_two_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_coefficient",
        input.homogeneous_large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_coefficient",
        input.homogeneous_small_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_energy_disagreement_source_input(
    input: &AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar(
        "soldir_energy_disagreement_speed_of_light",
        input.speed_of_light,
    )?;
    validate_dirac_energy_disagreement_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_energy_disagreement_correction_input(
    input: &AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_disagreement_correction_step", input.step)?;
    validate_finite_scalar("soldir_energy_disagreement_correction_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_disagreement_correction_origin_power",
        input.origin_power,
    )?;
    validate_dirac_energy_disagreement_correction_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_derivative",
        input.radii.len(),
        input.large_derivative.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_derivative",
        input.radii.len(),
        input.small_derivative.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_derivative_coefficients",
        input.coefficient_count,
        input.large_derivative_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_derivative_coefficients",
        input.coefficient_count,
        input.small_derivative_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative",
        input.large_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative",
        input.small_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative_coefficient",
        input.large_derivative_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative_coefficient",
        input.small_derivative_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_matching_point_update_input(
    input: &AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector(
        "soldir_matching_point_large_component",
        input.large_component,
    )?;
    if input.active_len > ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET
        && input.active_len <= input.large_component.len()
    {
        validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len: input.active_len,
            radial_count: input.large_component.len(),
        })
    }
}

pub(super) fn validate_dirac_inhomogeneous_seed_input(
    input: &AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<(), AtomMathError> {
    validate_radial_table_len(
        "soldir_seed_small_source",
        input.large_source.len(),
        input.small_source.len(),
    )?;
    validate_coefficient_count("soldir_seed_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_seed_large_coefficients",
        input.coefficient_count,
        input.large_source_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_seed_small_coefficients",
        input.coefficient_count,
        input.small_source_coefficients.len(),
    )?;
    validate_finite_vector("soldir_seed_large_source", input.large_source)?;
    validate_finite_vector("soldir_seed_small_source", input.small_source)?;
    validate_finite_vector(
        "soldir_seed_large_source_coefficient",
        input.large_source_coefficients,
    )?;
    validate_finite_vector(
        "soldir_seed_small_source_coefficient",
        input.small_source_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_homogeneous_seed_input(
    input: &AtomicDiracHomogeneousSeedInput,
) -> Result<(), AtomMathError> {
    if input.radial_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_radial_len",
            minimum: 1,
            actual: input.radial_len,
        });
    }
    if input.coefficient_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_coefficient_len",
            minimum: 1,
            actual: input.coefficient_len,
        });
    }
    Ok(())
}

pub(super) fn validate_dirac_shooting_pass_setup_input(
    input: &AtomicDiracShootingPassSetupInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_shooting_pass_energy", input.energy)?;
    validate_finite_scalar(
        "soldir_shooting_pass_previous_energy",
        input.previous_energy,
    )?;
    Ok(())
}

pub(super) fn validate_dirac_integration_input(
    input: &AtomicDiracIntegrationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("intdir_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("intdir_step", input.step)?;
    validate_positive_finite_scalar("intdir_matching_precision", input.matching_precision)?;
    validate_finite_scalar("intdir_energy", input.energy)?;
    validate_finite_scalar("intdir_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "intdir_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    validate_dirac_integration_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "intdir_large_source",
        input.radii.len(),
        input.large_source.len(),
    )?;
    validate_radial_table_len(
        "intdir_small_source",
        input.radii.len(),
        input.small_source.len(),
    )?;
    validate_radial_table_len("intdir_potential", input.radii.len(), input.potential.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("intdir_large_source", input.large_source)?;
    validate_finite_vector("intdir_small_source", input.small_source)?;
    validate_finite_vector("intdir_potential", input.potential)?;
    validate_coefficient_count("intdir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "intdir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_potential_coefficients",
        input.coefficient_count,
        input.potential_coefficients.len(),
    )?;
    validate_finite_vector("intdir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("intdir_small_coefficient", input.small_coefficients)?;
    validate_finite_vector("intdir_potential_coefficient", input.potential_coefficients)?;
    atom_intdir_decay(input.energy, input.speed_of_light)?;

    match input.mode {
        AtomicDiracIntegrationMode::SearchMatchingPoint => Ok(()),
        AtomicDiracIntegrationMode::FixedMatchingPoint | AtomicDiracIntegrationMode::InwardOnly => {
            validate_dirac_integration_matching_index(
                input.matching_index_1based,
                input.active_len,
            )?;
            validate_dirac_integration_max_index(
                input.max_index_1based,
                input.matching_index_1based,
                input.active_len,
            )
        }
    }
}

pub(super) fn validate_dirac_solver_setup_input(
    input: &AtomicDiracSolverSetupInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_setup_energy", input.energy)?;
    validate_finite_scalar("soldir_setup_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "soldir_setup_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_positive_finite_scalar("soldir_setup_speed_of_light", input.speed_of_light)?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    if input.principal_quantum_number == 0 {
        return Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        });
    }
    validate_dirac_solver_setup_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_setup_potential",
        input.radii.len(),
        input.potential.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_setup_potential_coefficients",
        1,
        input.potential_coefficients.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_setup_potential", input.potential)?;
    validate_finite_vector(
        "soldir_setup_potential_coefficient",
        input.potential_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_local_density_potential_input(
    input: &AtomicLocalDensityPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "vlda_radii",
            minimum: 1,
            actual: radial_count,
        });
    }
    validate_positive_finite_scalar("vlda_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "initial_potential",
        radial_count,
        input.initial_potential.len(),
    )?;
    validate_radial_table_len(
        "initial_energy_density",
        radial_count,
        input.initial_energy_density.len(),
    )?;
    if input.initial_development_coefficients.len() < 2 {
        return Err(AtomMathError::InvalidCount {
            field: "initial_development_coefficients",
            minimum: 2,
            actual: input.initial_development_coefficients.len(),
        });
    }

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }

    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_vector("initial_potential", input.initial_potential)?;
    validate_finite_vector(
        "initial_development_coefficient",
        input.initial_development_coefficients,
    )?;
    validate_finite_vector("initial_energy_density", input.initial_energy_density)?;
    Ok(())
}

pub(super) fn validate_orbital_potential_input(
    input: &AtomicOrbitalPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: radial_count,
            radial_count,
        });
    }
    validate_positive_finite_scalar("potrdf_speed_of_light", input.speed_of_light)?;
    validate_finite_scalar("potrdf_step", input.step)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "nuclear_potential",
        radial_count,
        input.nuclear_potential.len(),
    )?;

    let coefficient_count = input.nuclear_development_coefficients.len();
    validate_coefficient_count("nuclear_development_coefficients", coefficient_count)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    if input.self_consistent_count > orbital_count {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: input.self_consistent_count,
            orbital_count,
        });
    }
    one_based_atomic_orbital_index(input.active_orbital_1based, orbital_count)?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len("shell_markers", orbital_count, input.shell_markers.len())?;
    validate_orbital_table_len("origin_scales", orbital_count, input.origin_scales.len())?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
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

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_angular_rank(kappa)?;
    }
    for &origin_scale in input.origin_scales {
        validate_positive_finite_scalar("origin_scale", origin_scale)?;
    }
    validate_positive_occupation(
        "potrdf_active_orbital",
        input.active_orbital_1based - 1,
        input.occupations,
    )?;

    if input.include_lagrange {
        let expected_pairs = orbital_pair_count(orbital_count)?;
        validate_coefficient_vector_len(
            "lagrange_parameters",
            expected_pairs,
            input.lagrange_parameters.len(),
        )?;
        validate_finite_vector("lagrange_parameter", input.lagrange_parameters)?;
    }

    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("nuclear_potential", input.nuclear_potential)?;
    validate_finite_vector(
        "nuclear_development_coefficient",
        input.nuclear_development_coefficients,
    )?;
    Ok(())
}

pub(super) fn validate_positive_finite_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::NonPositiveScalar { field, value })
    }
}

pub(super) fn validate_positive_finite_nuclear_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialScalar { field, value })
    }
}

pub(super) fn validate_nuclear_count(
    field: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), AtomMathError> {
    if actual >= minimum {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialCount {
            field,
            minimum,
            actual,
        })
    }
}

pub(super) fn validate_differential_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNormalizationActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_integration_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > ATOM_INTDIR_HISTORY + 12 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_solver_setup_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_solution_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracSolutionNormalizationActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

pub(super) fn validate_dirac_match_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_energy_disagreement_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength {
            active_len,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_energy_disagreement_correction_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len % 2 == 1 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

pub(super) fn validate_dirac_match_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

pub(super) fn validate_dirac_node_count_index(
    field: &'static str,
    index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if index_1based > 0 && index_1based <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNodeCountIndex {
            field,
            index_1based,
            radial_count,
        })
    }
}

pub(super) fn validate_dirac_energy_correction_matching_index(
    matching_index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange {
                matching_index_1based,
                radial_count,
            },
        )
    }
}

pub(super) fn validate_dirac_integration_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > ATOM_INTDIR_HISTORY && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

pub(super) fn validate_dirac_integration_max_index(
    max_index_1based: usize,
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if max_index_1based <= active_len
        && max_index_1based > matching_index_1based + ATOM_INTDIR_HISTORY
    {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange {
            max_index_1based,
            matching_index_1based,
            active_len,
        })
    }
}

pub(super) fn validate_coefficient_count(
    table: &'static str,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len > 0 {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: 1,
            actual_len,
        })
    }
}

pub(super) fn validate_coefficient_vector_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(super) fn validate_coefficient_vector_capacity(
    table: &'static str,
    required_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len >= required_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: required_len,
            actual_len,
        })
    }
}

pub(super) fn validate_matrix_shape(
    table: &'static str,
    matrix: ArrayView2<'_, Real>,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), AtomMathError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows == expected_rows && columns == expected_columns {
        Ok(())
    } else {
        Err(AtomMathError::MatrixShape {
            table,
            expected_rows,
            expected_columns,
            rows,
            columns,
        })
    }
}

pub(super) fn validate_orbital_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(super) fn validate_radial_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::RadialTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

pub(super) fn validate_occupation_tables(
    occupations: &[Real],
    kappas: &[i32],
) -> Result<(), AtomMathError> {
    if occupations.len() != kappas.len() {
        return Err(AtomMathError::OccupationKappaLengthMismatch {
            occupation_len: occupations.len(),
            kappa_len: kappas.len(),
        });
    }
    validate_finite_slice("occupation", occupations)
}

pub(super) fn validate_positive_occupation(
    context: &'static str,
    orbital: usize,
    occupations: &[Real],
) -> Result<Real, AtomMathError> {
    let occupation = occupations[orbital];
    if occupation > 0.0 {
        Ok(occupation)
    } else {
        Err(AtomMathError::NonPositiveOccupation {
            context,
            orbital_1based: orbital + 1,
            occupation,
        })
    }
}

pub(super) fn validate_orbital_index(index: usize, len: usize) -> Result<(), AtomMathError> {
    if index < len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalIndexOutOfRange { index, len })
    }
}

pub(super) fn validate_coefficient_table(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<(), AtomMathError> {
    let shape = coefficients.shape();
    let rows = shape[0];
    let columns = shape[1];
    let channels = shape[2];
    if rows == 0 || columns == 0 || rows != columns || channels == 0 {
        return Err(AtomMathError::CoefficientTableShape {
            rows,
            columns,
            channels,
        });
    }
    if left >= rows {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: left,
            len: rows,
        });
    }
    if right >= columns {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: right,
            len: columns,
        });
    }
    let channel = rank / 2;
    if channel >= channels {
        return Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        });
    }
    for value in coefficients.iter().copied() {
        if !value.is_finite() {
            return Err(AtomMathError::NonFiniteScalar {
                field: "coefficient",
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_finite_slice(
    field: &'static str,
    values: &[Real],
) -> Result<(), AtomMathError> {
    for &value in values {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(super) fn validate_finite_vector(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in values.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(super) fn validate_finite_matrix(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in matrix.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

pub(super) fn validate_positive_finite_radii(
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for &radius in values {
        validate_finite_scalar("radius", radius)?;
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
    }
    Ok(())
}

pub(super) fn validate_finite_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AtomMathError::NonFiniteScalar { field, value })
    }
}
