use super::*;

/// Port of FEFF `ATOM/wfirdf.f90`, the Thomas-Fermi starting-orbital driver.
///
/// This builds the logarithmic radial mesh and point/finite nuclear potential,
/// adds FEFF's Thomas-Fermi starting potential, computes origin powers and
/// starting coefficients, then calls the composed `soldir` driver once per
/// orbital with `method = 0`.
pub fn atomic_initial_orbitals(
    input: AtomicInitialOrbitalsInput<'_>,
) -> Result<AtomicInitialOrbitals, AtomMathError> {
    validate_initial_orbitals_input(&input)?;
    calculate_atomic_initial_orbitals(input)
}

fn calculate_atomic_initial_orbitals(
    input: AtomicInitialOrbitalsInput<'_>,
) -> Result<AtomicInitialOrbitals, AtomMathError> {
    let orbital_count = input.principal_quantum_numbers.len();
    let nuclear = atomic_nuclear_potential(AtomicNuclearPotentialInput {
        nuclear_charge: input.nuclear_charge,
        step: input.step,
        requested_nucleus_index: input.requested_nucleus_index,
        radial_count: input.radial_count,
        coefficient_count: input.coefficient_count,
        first_radius_times_charge: input.first_radius_times_charge,
    })?;

    let mut orbital_powers = Array1::<Real>::zeros(orbital_count);
    let mut origin_scales = Array1::<Real>::zeros(orbital_count);
    let finite_nucleus = nuclear.nucleus_index > 1;
    let point_charge_shift = if finite_nucleus {
        0.0
    } else {
        (input.nuclear_charge / input.speed_of_light).powi(2)
    };
    for orbital in 0..orbital_count {
        let kappa_abs = kappa_abs_usize(input.kappas[orbital])?;
        let power_squared = (input.kappas[orbital] as Real).powi(2) - point_charge_shift;
        if power_squared < 0.0 {
            return Err(AtomMathError::InvalidKappa {
                kappa: input.kappas[orbital],
            });
        }
        let power = power_squared.sqrt();
        validate_finite_scalar("wfirdf_origin_power", power)?;
        orbital_powers[orbital] = power;
        origin_scales[orbital] = nuclear.radii[0].powf(power - kappa_abs as Real);
        validate_finite_scalar("wfirdf_origin_scale", origin_scales[orbital])?;
    }

    let mut potential = Array1::<Real>::zeros(input.radial_count);
    for row in 0..input.radial_count {
        potential[row] = (thomas_fermi_density_potential(
            nuclear.radii[row],
            input.nuclear_charge,
            input.thomas_fermi_ionicity,
        )? + nuclear.potential[row])
            / input.speed_of_light;
    }
    let mut potential_coefficients = nuclear
        .development_coefficients
        .mapv(|value| value / input.speed_of_light);
    let nucleus_row = nuclear.nucleus_index - 1;
    potential_coefficients[1] += thomas_fermi_density_potential(
        nuclear.radii[nucleus_row],
        input.nuclear_charge,
        input.thomas_fermi_ionicity,
    )? / input.speed_of_light;

    let source = Array1::<Real>::zeros(input.radial_count);
    let source_coefficients = Array1::<Real>::zeros(input.coefficient_count);
    let mut orbital_energies = Array1::<Real>::zeros(orbital_count);
    let mut active_lengths = Array1::<usize>::zeros(orbital_count);
    let mut large_components = Array2::<Real>::zeros((input.radial_count, orbital_count));
    let mut small_components = Array2::<Real>::zeros((input.radial_count, orbital_count));
    let mut large_coefficients = Array2::<Real>::zeros((input.coefficient_count, orbital_count));
    let mut small_coefficients = Array2::<Real>::zeros((input.coefficient_count, orbital_count));
    let mut attempts_exhausted = Vec::with_capacity(orbital_count);

    for orbital in 0..orbital_count {
        let (large_initial, small_initial) = initial_origin_coefficients(
            input.nuclear_charge,
            input.speed_of_light,
            input.principal_quantum_numbers[orbital],
            input.kappas[orbital],
            orbital_powers[orbital],
            finite_nucleus,
        )?;
        let principal = input.principal_quantum_numbers[orbital] as Real;
        let energy = -input.nuclear_charge * input.nuclear_charge / principal * principal;
        let solution = atomic_dirac_bound_orbital(AtomicDiracBoundOrbitalInput {
            large_source: source.view(),
            small_source: source.view(),
            large_source_coefficients: source_coefficients.view(),
            small_source_coefficients: source_coefficients.view(),
            radii: nuclear.radii.view(),
            potential: potential.view(),
            potential_coefficients: potential_coefficients.view(),
            energy,
            origin_power: orbital_powers[orbital],
            initial_large_coefficient: large_initial,
            initial_small_coefficient: small_initial,
            asymptotic_large_component: input.primary_matching_precision,
            principal_quantum_number: input.principal_quantum_numbers[orbital],
            kappa: input.kappas[orbital],
            speed_of_light: input.speed_of_light,
            step: input.step,
            primary_matching_precision: input.primary_matching_precision,
            secondary_matching_precision: input.secondary_matching_precision,
            coefficient_count: input.coefficient_count,
            active_len: input.radial_count,
            initial_max_index_1based: input.active_lengths[orbital],
            max_attempt_count: input.max_attempt_count,
            method: 0,
        })?;

        orbital_energies[orbital] = solution.energy;
        active_lengths[orbital] = solution.active_len;
        attempts_exhausted.push(solution.attempts_exhausted);
        large_components
            .column_mut(orbital)
            .assign(&solution.large_component);
        small_components
            .column_mut(orbital)
            .assign(&solution.small_component);
        large_coefficients
            .column_mut(orbital)
            .assign(&solution.large_coefficients);
        small_coefficients
            .column_mut(orbital)
            .assign(&solution.small_coefficients);
    }

    Ok(AtomicInitialOrbitals {
        radii: nuclear.radii,
        nuclear_potential: nuclear.potential,
        nuclear_development_coefficients: nuclear.development_coefficients,
        potential,
        potential_coefficients,
        nucleus_index: nuclear.nucleus_index,
        orbital_powers,
        origin_scales,
        orbital_energies,
        active_lengths,
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        attempts_exhausted,
    })
}

fn initial_origin_coefficients(
    nuclear_charge: Real,
    speed_of_light: Real,
    principal_quantum_number: usize,
    kappa: i32,
    origin_power: Real,
    finite_nucleus: bool,
) -> Result<(Real, Real), AtomMathError> {
    let kappa_abs = abs_kappa_i32(kappa)?;
    let principal = i32::try_from(principal_quantum_number).map_err(|_| {
        AtomMathError::InvalidPrincipalQuantumNumber {
            orbital_1based: 1,
            principal_quantum_number,
        }
    })?;
    let mut parity = principal - kappa_abs;
    if kappa < 0 {
        parity -= 1;
    }

    let mut large = if parity % 2 == 0 { -1.0 } else { 1.0 };
    let mut small;
    if kappa < 0 {
        let denominator = speed_of_light * (kappa as Real - origin_power);
        if denominator == 0.0 {
            return Err(AtomMathError::ZeroDiracSolverInitialCoefficientDenominator);
        }
        small = large * nuclear_charge / denominator;
        if finite_nucleus {
            small = 0.0;
        }
    } else {
        small = large * speed_of_light * (kappa as Real + origin_power) / nuclear_charge;
        if finite_nucleus {
            large = 0.0;
        }
    }
    validate_finite_scalar("wfirdf_initial_large_coefficient", large)?;
    validate_finite_scalar("wfirdf_initial_small_coefficient", small)?;
    Ok((large, small))
}

fn validate_initial_orbitals_input(
    input: &AtomicInitialOrbitalsInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("wfirdf_nuclear_charge", input.nuclear_charge)?;
    validate_finite_scalar("wfirdf_thomas_fermi_ionicity", input.thomas_fermi_ionicity)?;
    validate_positive_finite_scalar("wfirdf_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("wfirdf_step", input.step)?;
    validate_positive_finite_scalar(
        "wfirdf_first_radius_times_charge",
        input.first_radius_times_charge,
    )?;
    validate_positive_finite_scalar(
        "wfirdf_primary_matching_precision",
        input.primary_matching_precision,
    )?;
    validate_positive_finite_scalar(
        "wfirdf_secondary_matching_precision",
        input.secondary_matching_precision,
    )?;
    validate_nuclear_count("wfirdf_radial_count", input.radial_count, 1)?;
    validate_nuclear_count("wfirdf_coefficient_count", input.coefficient_count, 5)?;

    let orbital_count = input.principal_quantum_numbers.len();
    if orbital_count == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "wfirdf_orbital_count",
            minimum: 1,
            actual: orbital_count,
        });
    }
    validate_orbital_table_len("wfirdf_kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len(
        "wfirdf_active_lengths",
        orbital_count,
        input.active_lengths.len(),
    )?;
    for orbital in 0..orbital_count {
        if input.principal_quantum_numbers[orbital] == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number: input.principal_quantum_numbers[orbital],
            });
        }
        if input.kappas[orbital] == 0 {
            return Err(AtomMathError::InvalidKappa {
                kappa: input.kappas[orbital],
            });
        }
        if input.active_lengths[orbital] == 0 || input.active_lengths[orbital] > input.radial_count
        {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len: input.active_lengths[orbital],
                row_count: input.radial_count,
            });
        }
    }
    Ok(())
}
