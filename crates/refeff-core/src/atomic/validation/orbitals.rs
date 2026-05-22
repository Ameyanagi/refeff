use super::*;

pub(in crate::atomic) fn validate_orbital_initialization_input(
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

pub(in crate::atomic) fn validate_local_density_potential_input(
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

pub(in crate::atomic) fn validate_orbital_potential_input(
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
