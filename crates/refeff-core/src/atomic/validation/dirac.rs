use super::*;

pub(in crate::atomic) fn validate_dirac_entry_state_input(
    input: &AtomicDiracEntryStateInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_entry_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    Ok(())
}

pub(in crate::atomic) fn validate_dirac_normalization_input(
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

pub(in crate::atomic) fn validate_dirac_solution_normalization_input(
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

pub(in crate::atomic) fn validate_dirac_node_count_input(
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

pub(in crate::atomic) fn validate_dirac_node_energy_search_input(
    input: &AtomicDiracNodeEnergySearchInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_node_search_energy", input.energy)?;
    validate_finite_scalar("soldir_node_search_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_node_search_einf", input.energy_inf)?;
    validate_finite_scalar("soldir_node_search_emin", input.energy_floor)?;
    validate_positive_finite_scalar("soldir_node_search_precision", input.energy_precision)?;
    Ok(())
}

pub(in crate::atomic) fn validate_dirac_iteration_reset_input(
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

pub(in crate::atomic) fn validate_dirac_method_one_energy_correction_input(
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

pub(in crate::atomic) fn validate_dirac_energy_step_input(
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

pub(in crate::atomic) fn validate_dirac_rematch_attempt_input(
    input: &AtomicDiracRematchAttemptInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_rematch_mismatch", input.mismatch)?;
    validate_positive_finite_scalar(
        "soldir_rematch_mismatch_precision",
        input.mismatch_precision,
    )?;
    Ok(())
}

pub(in crate::atomic) fn validate_dirac_large_component_match_input(
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

pub(in crate::atomic) fn validate_dirac_homogeneous_match_input(
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

pub(in crate::atomic) fn validate_dirac_two_component_match_input(
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

pub(in crate::atomic) fn validate_dirac_energy_disagreement_source_input(
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

pub(in crate::atomic) fn validate_dirac_energy_disagreement_correction_input(
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

pub(in crate::atomic) fn validate_dirac_matching_point_update_input(
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

pub(in crate::atomic) fn validate_dirac_inhomogeneous_seed_input(
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

pub(in crate::atomic) fn validate_dirac_homogeneous_seed_input(
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

pub(in crate::atomic) fn validate_dirac_shooting_pass_setup_input(
    input: &AtomicDiracShootingPassSetupInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_shooting_pass_energy", input.energy)?;
    validate_finite_scalar(
        "soldir_shooting_pass_previous_energy",
        input.previous_energy,
    )?;
    Ok(())
}

pub(in crate::atomic) fn validate_dirac_integration_input(
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

pub(in crate::atomic) fn validate_dirac_solver_setup_input(
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
