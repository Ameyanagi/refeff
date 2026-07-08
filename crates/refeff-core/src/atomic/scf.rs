use super::*;

/// Port the positive-`niter` FEFF `ATOM/scfdat.f90` SCF orbital loop.
pub fn atomic_self_consistent_orbitals(
    input: AtomicScfInput<'_>,
) -> Result<AtomicScf, AtomMathError> {
    validate_scf_input(&input)?;

    let orbital_count = input.active_lengths.len();
    let iteration_limit = input
        .max_orbital_iterations
        .checked_mul(orbital_count)
        .ok_or(AtomMathError::ScfIterationLimitOverflow {
            max_orbital_iterations: input.max_orbital_iterations,
            orbital_count,
        })?;

    let mut active_lengths = input.active_lengths.to_vec();
    let mut large_components = input.large_components.to_owned();
    let mut small_components = input.small_components.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();
    let mut orbital_energies = input.orbital_energies.to_vec();
    let mut convergence_acceleration = input.convergence_acceleration.to_vec();
    let mut wavefunction_errors = input.wavefunction_errors.to_vec();
    let mut energy_errors = input.energy_errors.to_vec();
    let mut lagrange_parameters = input.lagrange_parameters.to_owned();
    let mut attempts_exhausted = vec![false; orbital_count];

    let mut iteration_count = 0usize;
    let mut active = 0usize;
    let mut direction = 1_i32;
    let mut unresolved_failed_orbital = None;

    loop {
        if iteration_count == iteration_limit {
            return Err(AtomMathError::ScfIterationLimitExceeded {
                iteration_count,
                iteration_limit,
            });
        }
        iteration_count += 1;

        if input.include_lagrange && input.shell_markers[active] > 0 {
            refresh_active_lagrange_parameters(
                &input,
                active,
                &active_lengths,
                large_components.view(),
                small_components.view(),
                large_coefficients.view(),
                small_coefficients.view(),
                &mut lagrange_parameters,
            )?;
        }

        let iteration = atomic_scf_orbital_iteration(AtomicScfOrbitalIterationInput {
            active_orbital_1based: active + 1,
            exchange_mode: input.exchange_mode,
            include_lagrange: input.include_lagrange,
            self_consistent_count: input.self_consistent_count,
            speed_of_light: input.speed_of_light,
            step: input.step,
            radii: input.radii,
            active_lengths: &active_lengths,
            principal_quantum_numbers: input.principal_quantum_numbers,
            kappas: input.kappas,
            orbital_powers: input.orbital_powers,
            occupations: input.occupations,
            valence_occupations: input.valence_occupations,
            shell_markers: input.shell_markers,
            origin_scales: input.origin_scales,
            coulomb_coefficients: input.coulomb_coefficients,
            lagrange_parameters: lagrange_parameters.view(),
            nuclear_potential: input.nuclear_potential,
            nuclear_development_coefficients: input.nuclear_development_coefficients,
            large_components: large_components.view(),
            small_components: small_components.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            orbital_energies: &orbital_energies,
            convergence_acceleration: &convergence_acceleration,
            wavefunction_errors: &wavefunction_errors,
            primary_matching_precision: input.primary_matching_precision,
            secondary_matching_precision: input.secondary_matching_precision,
            max_attempt_count: input.max_attempt_count,
        })?;

        orbital_energies[active] = iteration.orbital_energy;
        active_lengths[active] = iteration.active_len;
        large_components
            .column_mut(active)
            .assign(&iteration.large_component);
        small_components
            .column_mut(active)
            .assign(&iteration.small_component);
        large_coefficients
            .column_mut(active)
            .assign(&iteration.large_coefficients);
        small_coefficients
            .column_mut(active)
            .assign(&iteration.small_coefficients);
        convergence_acceleration[active] = iteration.convergence_acceleration;
        wavefunction_errors[active] = iteration.wavefunction_error;
        energy_errors[active] = iteration.energy_error;
        attempts_exhausted[active] = iteration.attempts_exhausted;

        if iteration.attempts_exhausted && unresolved_failed_orbital.is_none() {
            unresolved_failed_orbital = Some(active + 1);
        } else if unresolved_failed_orbital == Some(active + 1) && !iteration.attempts_exhausted {
            unresolved_failed_orbital = None;
        }

        if iteration_count < input.self_consistent_count
            || (direction < 0 && active + 1 < input.self_consistent_count)
        {
            active += 1;
            continue;
        }

        let (largest_wave_orbital, largest_wave_error) =
            largest_abs_error(&wavefunction_errors, input.self_consistent_count);
        if largest_wave_error > input.wavefunction_precision {
            active = largest_wave_orbital;
            direction = 1;
            continue;
        }

        let (largest_energy_orbital, largest_energy_error) =
            largest_abs_error(&energy_errors, input.self_consistent_count);
        if largest_energy_error >= input.energy_precision {
            active = largest_energy_orbital;
            direction = 1;
            continue;
        }

        if direction < 0 {
            break;
        }

        direction = -1;
        active = 0;
    }

    if let Some(orbital_1based) = unresolved_failed_orbital {
        return Err(AtomMathError::ScfDiracAttemptFailed { orbital_1based });
    }

    let final_density = scf_final_density(
        &input,
        &active_lengths,
        large_components.view(),
        small_components.view(),
    )?;
    let density_4pi =
        scf_density_per_radius_squared(input.radii, final_density.total_density.view())?;
    let valence_density_4pi =
        scf_density_per_radius_squared(input.radii, final_density.valence_density.view())?;
    let coulomb_potential = scf_coulomb_potential(
        input.nuclear_charge,
        input.radii,
        input.step,
        final_density.total_density.view(),
    )?;

    Ok(AtomicScf {
        iteration_count,
        orbital_energies: Array1::from_vec(orbital_energies),
        active_lengths: Array1::from_vec(active_lengths),
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        convergence_acceleration: Array1::from_vec(convergence_acceleration),
        wavefunction_errors: Array1::from_vec(wavefunction_errors),
        energy_errors: Array1::from_vec(energy_errors),
        lagrange_parameters,
        attempts_exhausted,
        total_density: final_density.total_density,
        valence_density: final_density.valence_density,
        energy_density: final_density.energy_density,
        density_4pi,
        valence_density_4pi,
        coulomb_potential,
    })
}

/// Compose one FEFF `ATOM/scfdat.f90` orbital iteration body.
pub fn atomic_scf_orbital_iteration(
    input: AtomicScfOrbitalIterationInput<'_>,
) -> Result<AtomicScfOrbitalIteration, AtomMathError> {
    let active = validate_scf_orbital_iteration_input(&input)?;

    let orbital_potential = atomic_orbital_potential(AtomicOrbitalPotentialInput {
        active_orbital_1based: input.active_orbital_1based,
        include_exchange: true,
        include_lagrange: input.include_lagrange,
        self_consistent_count: input.self_consistent_count,
        speed_of_light: input.speed_of_light,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        kappas: input.kappas,
        orbital_powers: input.orbital_powers,
        occupations: input.occupations,
        shell_markers: input.shell_markers,
        origin_scales: input.origin_scales,
        coulomb_coefficients: input.coulomb_coefficients,
        lagrange_parameters: input.lagrange_parameters,
        nuclear_potential: input.nuclear_potential,
        nuclear_development_coefficients: input.nuclear_development_coefficients,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
    })?;

    let zero_energy_density = Array1::<Real>::zeros(input.radii.len());
    let local_density = atomic_local_density_potential(AtomicLocalDensityPotentialInput {
        mode: input.exchange_mode,
        accumulate_energy_density: false,
        speed_of_light: input.speed_of_light,
        radii: input.radii,
        active_lengths: input.active_lengths,
        occupations: input.occupations,
        valence_occupations: input.valence_occupations,
        large_components: input.large_components,
        small_components: input.small_components,
        initial_potential: orbital_potential.central_potential.view(),
        initial_development_coefficients: orbital_potential.central_development_coefficients.view(),
        initial_energy_density: zero_energy_density.view(),
    })?;

    let previous_energy = input.orbital_energies[active];
    let previous_active_len = input.active_lengths[active];
    let asymptotic_large_component = input.large_components[(previous_active_len - 1, active)];
    let solution = atomic_dirac_bound_orbital(AtomicDiracBoundOrbitalInput {
        large_source: orbital_potential.exchange_large.view(),
        small_source: orbital_potential.exchange_small.view(),
        large_source_coefficients: orbital_potential.exchange_large_coefficients.view(),
        small_source_coefficients: orbital_potential.exchange_small_coefficients.view(),
        radii: input.radii,
        potential: local_density.potential.view(),
        potential_coefficients: local_density.development_coefficients.view(),
        energy: previous_energy,
        origin_power: input.orbital_powers[active],
        initial_large_coefficient: input.large_coefficients[(0, active)],
        initial_small_coefficient: input.small_coefficients[(0, active)],
        asymptotic_large_component,
        principal_quantum_number: input.principal_quantum_numbers[active],
        kappa: input.kappas[active],
        speed_of_light: input.speed_of_light,
        step: input.step,
        primary_matching_precision: input.primary_matching_precision,
        secondary_matching_precision: input.secondary_matching_precision,
        coefficient_count: input.large_coefficients.nrows(),
        active_len: input.radii.len(),
        initial_max_index_1based: previous_active_len,
        max_attempt_count: input.max_attempt_count,
        method: 1,
    })?;

    if solution.energy == 0.0 {
        return Err(AtomMathError::ZeroScfOrbitalEnergy {
            orbital_1based: input.active_orbital_1based,
        });
    }
    let energy_error = ((previous_energy - solution.energy) / solution.energy).abs();
    validate_finite_scalar("scfdat_energy_error", energy_error)?;

    let wavefunction_error = scf_wavefunction_error(input, active, &solution);
    let mix = atomic_convergence_mix(
        input.convergence_acceleration[active],
        wavefunction_error,
        input.wavefunction_errors[active],
    )?;

    let mut mixed_large = Array1::<Real>::zeros(input.radii.len());
    let mut mixed_small = Array1::<Real>::zeros(input.radii.len());
    for row in 0..solution.active_len {
        mixed_large[row] = mix.final_weight * solution.large_component[row]
            + mix.initial_weight * input.large_components[(row, active)];
        mixed_small[row] = mix.final_weight * solution.small_component[row]
            + mix.initial_weight * input.small_components[(row, active)];
    }

    let coefficient_count = input.large_coefficients.nrows();
    let mut mixed_large_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut mixed_small_coefficients = Array1::<Real>::zeros(coefficient_count);
    for coefficient in 0..coefficient_count {
        mixed_large_coefficients[coefficient] = mix.final_weight
            * solution.large_coefficients[coefficient]
            + mix.initial_weight * input.large_coefficients[(coefficient, active)];
        mixed_small_coefficients[coefficient] = mix.final_weight
            * solution.small_coefficients[coefficient]
            + mix.initial_weight * input.small_coefficients[(coefficient, active)];
    }

    let normalization = atomic_differential_integral(AtomicDifferentialIntegralInput {
        kind: AtomicDifferentialIntegralKind::DerivativeNorm {
            active_len: solution.active_len,
        },
        power: 0,
        origin_power: input.orbital_powers[active],
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
        derivative_large: mixed_large.view(),
        derivative_small: mixed_small.view(),
        derivative_large_coefficients: mixed_large_coefficients.view(),
        derivative_small_coefficients: mixed_small_coefficients.view(),
    })?;
    if normalization <= 0.0 {
        return Err(AtomMathError::NonPositiveScalar {
            field: "scfdat_normalization",
            value: normalization,
        });
    }
    let normalization_root = normalization.sqrt();
    let large_component = mixed_large.mapv(|value| value / normalization_root);
    let small_component = mixed_small.mapv(|value| value / normalization_root);
    let large_coefficients = mixed_large_coefficients.mapv(|value| value / normalization_root);
    let small_coefficients = mixed_small_coefficients.mapv(|value| value / normalization_root);

    Ok(AtomicScfOrbitalIteration {
        active_orbital_1based: input.active_orbital_1based,
        orbital_energy: solution.energy,
        active_len: solution.active_len,
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        convergence_acceleration: mix.final_weight,
        wavefunction_error: mix.previous_error,
        energy_error,
        attempts_exhausted: solution.attempts_exhausted,
        total_density: local_density.total_density,
        valence_density: local_density.valence_density,
        potential: local_density.potential,
        potential_coefficients: local_density.development_coefficients,
        normalization,
    })
}

fn validate_scf_input(input: &AtomicScfInput<'_>) -> Result<(), AtomMathError> {
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_positive_finite_scalar("scfdat_nuclear_charge", input.nuclear_charge)?;
    if input.self_consistent_count == 0 || input.self_consistent_count > orbital_count {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: input.self_consistent_count,
            orbital_count,
        });
    }
    if input.max_orbital_iterations == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "scfdat_max_orbital_iterations",
            minimum: 1,
            actual: input.max_orbital_iterations,
        });
    }
    validate_positive_finite_scalar(
        "scfdat_wavefunction_precision",
        input.wavefunction_precision,
    )?;
    validate_positive_finite_scalar("scfdat_energy_precision", input.energy_precision)?;
    validate_orbital_table_len("energy_errors", orbital_count, input.energy_errors.len())?;
    validate_finite_slice("energy_error", input.energy_errors)?;
    if input.include_lagrange {
        let expected_pairs = orbital_pair_count(orbital_count)?;
        if input.lagrange_parameters.len() < expected_pairs {
            validate_coefficient_vector_len(
                "lagrange_parameters",
                expected_pairs,
                input.lagrange_parameters.len(),
            )?;
        }
        validate_finite_vector("lagrange_parameter", input.lagrange_parameters)?;
    }
    validate_scf_orbital_iteration_input(&AtomicScfOrbitalIterationInput {
        active_orbital_1based: 1,
        exchange_mode: input.exchange_mode,
        include_lagrange: input.include_lagrange,
        self_consistent_count: input.self_consistent_count,
        speed_of_light: input.speed_of_light,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        principal_quantum_numbers: input.principal_quantum_numbers,
        kappas: input.kappas,
        orbital_powers: input.orbital_powers,
        occupations: input.occupations,
        valence_occupations: input.valence_occupations,
        shell_markers: input.shell_markers,
        origin_scales: input.origin_scales,
        coulomb_coefficients: input.coulomb_coefficients,
        lagrange_parameters: input.lagrange_parameters,
        nuclear_potential: input.nuclear_potential,
        nuclear_development_coefficients: input.nuclear_development_coefficients,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
        orbital_energies: input.orbital_energies,
        convergence_acceleration: input.convergence_acceleration,
        wavefunction_errors: input.wavefunction_errors,
        primary_matching_precision: input.primary_matching_precision,
        secondary_matching_precision: input.secondary_matching_precision,
        max_attempt_count: input.max_attempt_count,
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn refresh_active_lagrange_parameters(
    input: &AtomicScfInput<'_>,
    active: usize,
    active_lengths: &[usize],
    large_components: ArrayView2<'_, Real>,
    small_components: ArrayView2<'_, Real>,
    large_coefficients: ArrayView2<'_, Real>,
    small_coefficients: ArrayView2<'_, Real>,
    lagrange_parameters: &mut Array1<Real>,
) -> Result<(), AtomMathError> {
    let mut previous_first_factor = None;
    let updated = atomic_lagrange_parameters(
        AtomicLagrangeParametersInput {
            active_orbital_1based: Some(active + 1),
            include_exchange: true,
            kappas: input.kappas,
            occupations: input.occupations,
            shell_markers: input.shell_markers,
            coulomb_coefficients: input.coulomb_coefficients,
        },
        |request| {
            let previous_first_factor_view = previous_first_factor
                .as_ref()
                .map(AtomicRadialFirstFactor::as_view);
            let integral = atomic_radial_integral(AtomicRadialIntegralInput {
                request,
                large_small: false,
                previous_first_factor: previous_first_factor_view,
                kappas: input.kappas,
                step: input.step,
                radii: input.radii,
                active_lengths,
                orbital_powers: input.orbital_powers,
                large_components,
                small_components,
                large_coefficients,
                small_coefficients,
            })?;
            if let Some(first_factor) = integral.first_factor {
                previous_first_factor = Some(first_factor);
            }
            Ok(integral.value)
        },
    )?;

    for other in 0..input.self_consistent_count {
        if !scf_uses_lagrange_pair(input, active, other) {
            continue;
        }
        let packed = packed_orbital_pair_index(active, other)?;
        lagrange_parameters[packed] = updated[packed];
    }
    Ok(())
}

fn scf_uses_lagrange_pair(input: &AtomicScfInput<'_>, active: usize, other: usize) -> bool {
    active != other
        && input.kappas[active] == input.kappas[other]
        && (input.shell_markers[active] >= 0 || input.shell_markers[other] >= 0)
        && input.occupations[active] != input.occupations[other]
}

fn largest_abs_error(values: &[Real], count: usize) -> (usize, Real) {
    let mut selected = 0usize;
    let mut largest = 0.0;
    for (index, value) in values.iter().copied().take(count).enumerate() {
        let value = value.abs();
        if value > largest {
            selected = index;
            largest = value;
        }
    }
    (selected, largest)
}

fn scf_final_density(
    input: &AtomicScfInput<'_>,
    active_lengths: &[usize],
    large_components: ArrayView2<'_, Real>,
    small_components: ArrayView2<'_, Real>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    let zero_potential = Array1::<Real>::zeros(input.radii.len());
    let zero_coefficients = Array1::<Real>::zeros(input.nuclear_development_coefficients.len());
    let zero_energy_density = Array1::<Real>::zeros(input.radii.len());
    atomic_local_density_potential(AtomicLocalDensityPotentialInput {
        mode: input.exchange_mode,
        accumulate_energy_density: true,
        speed_of_light: input.speed_of_light,
        radii: input.radii,
        active_lengths,
        occupations: input.occupations,
        valence_occupations: input.valence_occupations,
        large_components,
        small_components,
        initial_potential: zero_potential.view(),
        initial_development_coefficients: zero_coefficients.view(),
        initial_energy_density: zero_energy_density.view(),
    })
}

fn scf_density_per_radius_squared(
    radii: ArrayView1<'_, Real>,
    density: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, AtomMathError> {
    validate_radial_table_len("scfdat_density", radii.len(), density.len())?;
    let mut values = Array1::<Real>::zeros(radii.len());
    for row in 0..radii.len() {
        let radius_squared = radii[row] * radii[row];
        if radius_squared <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius: radii[row] });
        }
        values[row] = density[row] / radius_squared;
        validate_finite_scalar("scfdat_density_4pi", values[row])?;
    }
    Ok(values)
}

fn scf_coulomb_potential(
    nuclear_charge: Real,
    radii: ArrayView1<'_, Real>,
    step: Real,
    total_density: ArrayView1<'_, Real>,
) -> Result<Array1<Real>, AtomMathError> {
    let mut potential =
        atomic_four_point_coulomb_potential(AtomicFourPointCoulombPotentialInput {
            density: total_density,
            radii,
            step,
            active_len: radii.len(),
        })?;
    for row in 0..potential.len() {
        potential[row] -= nuclear_charge / radii[row];
        validate_finite_scalar("scfdat_coulomb_potential", potential[row])?;
    }
    Ok(potential)
}

fn validate_scf_orbital_iteration_input(
    input: &AtomicScfOrbitalIterationInput<'_>,
) -> Result<usize, AtomMathError> {
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }

    let active = one_based_atomic_orbital_index(input.active_orbital_1based, orbital_count)?;
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    validate_orbital_table_len(
        "convergence_acceleration",
        orbital_count,
        input.convergence_acceleration.len(),
    )?;
    validate_orbital_table_len(
        "wavefunction_errors",
        orbital_count,
        input.wavefunction_errors.len(),
    )?;

    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }

    let active_len = input.active_lengths[active];
    if active_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "scfdat_active_len",
            minimum: 1,
            actual: active_len,
        });
    }

    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    validate_finite_slice("convergence_acceleration", input.convergence_acceleration)?;
    validate_finite_slice("wavefunction_error", input.wavefunction_errors)?;
    Ok(active)
}

fn scf_wavefunction_error(
    input: AtomicScfOrbitalIterationInput<'_>,
    active: usize,
    solution: &AtomicDiracBoundOrbital,
) -> Real {
    let mut error: Real = 0.0;
    for row in 0..solution.active_len {
        let large = input.large_components[(row, active)] - solution.large_component[row];
        if large.abs() > error.abs() {
            error = large;
        }
        let small = input.small_components[(row, active)] - solution.small_component[row];
        if small.abs() > error.abs() {
            error = small;
        }
    }
    error
}
