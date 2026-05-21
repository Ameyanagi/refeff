use super::*;

/// Port of FEFF `ATOM/inmuat.f90` after the `getorb` occupation compaction.
///
/// The caller supplies compacted orbital quantum numbers and occupations, for
/// example from [`crate::orbital_configuration`]. This routine mirrors the
/// deterministic ATOM setup that follows `getorb`: electron-count checking,
/// convergence defaults, active lengths, open-shell flags, and the Lagrange
/// pair count.
pub fn atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    validate_orbital_initialization_input(&input)?;
    calculate_atomic_orbital_initialization(input)
}

/// Port of FEFF `ATOM/vlda.f90`, local-density exchange potential.
///
/// The routine builds FEFF's total and valence spherical densities from orbital
/// components, evaluates the selected `idfock` exchange branch, and returns the
/// updated potential/development/energy arrays.
pub fn atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    validate_local_density_potential_input(&input)?;
    calculate_atomic_local_density_potential(input)
}

/// Port of FEFF `ATOM/potrdf.f90`, orbital potential/source assembly.
pub fn atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    validate_orbital_potential_input(&input)?;
    calculate_atomic_orbital_potential(input)
}

pub(super) fn calculate_atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    let orbital_count = input.occupations.len();
    let active_lengths =
        Array1::<usize>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_RADIAL_COUNT);
    let mut shell_markers = Array1::<i32>::from_elem(orbital_count, -1);
    let mut convergence_acceleration =
        Array1::<Real>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_CONVERGENCE);
    let mut lagrange_pair_count = 0usize;

    for orbital in 0..orbital_count {
        let kappa_abs = kappa_abs_usize(input.kappas[orbital])?;
        let angular_momentum = if input.kappas[orbital] < 0 {
            kappa_abs
                .checked_sub(1)
                .ok_or(AtomMathError::InvalidKappa {
                    kappa: input.kappas[orbital],
                })?
        } else {
            kappa_abs
        };
        let principal_quantum_number = input.principal_quantum_numbers[orbital];
        if angular_momentum >= principal_quantum_number || angular_momentum > 4 {
            return Err(AtomMathError::OrbitalAngularMomentumOutOfRange {
                orbital_1based: orbital + 1,
                principal_quantum_number,
                kappa: input.kappas[orbital],
                angular_momentum,
            });
        }

        let closed_shell_capacity = 2.0 * kappa_abs as Real;
        if input.occupations[orbital] < closed_shell_capacity {
            shell_markers[orbital] = 1;
        }
        if input.occupations[orbital] < 0.5 {
            convergence_acceleration[orbital] = 1.0;
        }
        for previous in 0..orbital {
            if input.kappas[previous] == input.kappas[orbital]
                && (shell_markers[previous] > 0 || shell_markers[orbital] > 0)
            {
                lagrange_pair_count = lagrange_pair_count
                    .checked_add(1)
                    .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })?;
            }
        }
    }

    for value in convergence_acceleration.iter().copied() {
        validate_finite_scalar("inmuat_convergence_acceleration", value)?;
    }

    Ok(AtomicOrbitalInitialization {
        orbital_count,
        self_consistent_count: orbital_count,
        wavefunction_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION,
        energy_precision: ATOM_INMUAT_ENERGY_PRECISION,
        precision_ratios: [ATOM_INMUAT_PRIMARY_RATIO, ATOM_INMUAT_SECONDARY_RATIO],
        primary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION / ATOM_INMUAT_PRIMARY_RATIO,
        secondary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION
            / ATOM_INMUAT_SECONDARY_RATIO,
        development_order: ATOM_INMUAT_DEVELOPMENT_ORDER,
        attempt_count: ATOM_INMUAT_ATTEMPT_COUNT,
        nucleus_index: ATOM_INMUAT_NUCLEUS_INDEX,
        radial_count: ATOM_INMUAT_DEFAULT_RADIAL_COUNT,
        orbital_energies: Array1::<Real>::zeros(orbital_count),
        convergence_acceleration,
        wavefunction_errors: Array1::<Real>::zeros(orbital_count),
        energy_errors: Array1::<Real>::zeros(orbital_count),
        active_lengths,
        shell_markers,
        lagrange_pair_count,
        lagrange_parameters: Array1::<Real>::zeros(ATOM_INMUAT_LAGRANGE_CAPACITY),
    })
}

pub(super) fn calculate_atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    let radial_count = input.radii.len();
    let orbital_count = input.active_lengths.len();
    let mut total_density = Array1::<Real>::zeros(radial_count);
    let mut valence_density = Array1::<Real>::zeros(radial_count);

    for orbital in 0..orbital_count {
        for row in 0..input.active_lengths[orbital] {
            let component_density = input.large_components[(row, orbital)].powi(2)
                + input.small_components[(row, orbital)].powi(2);
            total_density[row] += input.occupations[orbital] * component_density;
            valence_density[row] += input.valence_occupations[orbital] * component_density;
        }
    }

    let mut potential = input.initial_potential.to_owned();
    let mut development_coefficients = input.initial_development_coefficients.to_owned();
    let mut energy_density = input.initial_energy_density.to_owned();

    for row in 0..radial_count {
        let radius_squared = input.radii[row] * input.radii[row];
        let density = total_density[row] / radius_squared;
        if density <= 0.0 {
            continue;
        }

        let comparison_density = match input.mode {
            AtomicLocalDensityExchangeMode::ValenceDensity => valence_density[row] / radius_squared,
            AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
                (total_density[row] - valence_density[row]) / radius_squared
            }
            AtomicLocalDensityExchangeMode::DiracFockOnly => 0.0,
            AtomicLocalDensityExchangeMode::TotalDensity => density,
        };
        let vxc = atomic_local_density_vxc(input.mode, density, comparison_density)?;
        if input.accumulate_energy_density {
            energy_density[row] += vxc * total_density[row];
        }
        let scaled_vxc = vxc / input.speed_of_light;
        if row == 0 {
            development_coefficients[1] += scaled_vxc;
        }
        potential[row] += scaled_vxc;
    }

    validate_finite_vector("vlda_total_density", total_density.view())?;
    validate_finite_vector("vlda_valence_density", valence_density.view())?;
    validate_finite_vector("vlda_potential", potential.view())?;
    validate_finite_vector(
        "vlda_development_coefficient",
        development_coefficients.view(),
    )?;
    validate_finite_vector("vlda_energy_density", energy_density.view())?;

    Ok(AtomicLocalDensityPotential {
        total_density,
        valence_density,
        potential,
        development_coefficients,
        energy_density,
    })
}

pub(super) fn atomic_local_density_vxc(
    mode: AtomicLocalDensityExchangeMode,
    density: Real,
    comparison_density: Real,
) -> Result<Real, AtomMathError> {
    let density_parameter = (density / 3.0).powf(-1.0 / 3.0);
    let comparison_parameter = if comparison_density > 0.0 {
        (comparison_density / 3.0).powf(-1.0 / 3.0)
    } else {
        101.0
    };

    match mode {
        AtomicLocalDensityExchangeMode::DiracFockOnly => Ok(0.0),
        AtomicLocalDensityExchangeMode::TotalDensity
        | AtomicLocalDensityExchangeMode::ValenceDensity => {
            Ok(von_barth_hedin_potential(comparison_parameter, 1.0)?)
        }
        AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
            let total_vbh = von_barth_hedin_potential(density_parameter, 1.0)?;
            let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
            let dirac_hara = dirac_hara_exchange_potential(comparison_parameter, fermi_momentum)?;
            Ok(total_vbh - dirac_hara)
        }
    }
}

pub(super) fn calculate_atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    let active_orbital =
        one_based_atomic_orbital_index(input.active_orbital_1based, input.active_lengths.len())?;
    let active_occupation =
        validate_positive_occupation("potrdf_active_orbital", active_orbital, input.occupations)?;
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let active_j2 = kappa_angular_rank(input.kappas[active_orbital])?;

    let mut central_development_coefficients = input.nuclear_development_coefficients.to_owned();
    let mut central_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_small_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_large_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_small_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut exchange_small_coefficients = Array1::<Real>::zeros(coefficient_count);

    let mut rank = 0usize;
    loop {
        let (source, source_coefficients, source_len) =
            direct_orbital_potential_source(input, active_orbital, active_occupation, rank)?;
        let transform = atomic_yk_zk_prepared_source(AtomicYkZkPreparedSourceInput {
            source: source.view(),
            source_coefficients: source_coefficients.view(),
            radii: input.radii,
            step: input.step,
            angular_momentum: rank,
            coefficient_count,
            source_len,
            active_len: radial_count,
        })?;

        for (coefficient, &value) in transform.yk_coefficients.iter().enumerate() {
            let target = rank
                .checked_add(coefficient)
                .and_then(|value| value.checked_add(3))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            if target < coefficient_count {
                central_development_coefficients[target] -= value;
            }
        }
        for row in 0..radial_count {
            central_work[row] += transform.yk[row];
        }

        rank = rank
            .checked_add(2)
            .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                angular_momentum: rank,
            })?;
        if rank <= coefficient_count {
            central_development_coefficients[rank - 1] += transform.origin_constant;
        }
        if rank >= active_j2 {
            break;
        }
    }

    if input.include_exchange {
        accumulate_orbital_exchange_terms(
            input,
            active_orbital,
            active_occupation,
            active_j2,
            exchange_large_work.view_mut(),
            exchange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }
    if input.include_lagrange {
        accumulate_orbital_lagrange_terms(
            input,
            active_orbital,
            lagrange_large_work.view_mut(),
            lagrange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }

    for coefficient in 0..coefficient_count {
        central_development_coefficients[coefficient] /= input.speed_of_light;
        exchange_large_coefficients[coefficient] /= input.speed_of_light;
        exchange_small_coefficients[coefficient] /= input.speed_of_light;
    }

    let central_potential = Array1::from_shape_fn(radial_count, |row| {
        (central_work[row] / input.radii[row] + input.nuclear_potential[row]) / input.speed_of_light
    });
    let exchange_large = Array1::from_shape_fn(radial_count, |row| {
        (exchange_large_work[row] + lagrange_large_work[row] * input.radii[row])
            / input.speed_of_light
    });
    let exchange_small = Array1::from_shape_fn(radial_count, |row| {
        (exchange_small_work[row] + lagrange_small_work[row] * input.radii[row])
            / input.speed_of_light
    });

    validate_finite_vector("potrdf_central_potential", central_potential.view())?;
    validate_finite_vector(
        "potrdf_central_development_coefficient",
        central_development_coefficients.view(),
    )?;
    validate_finite_vector("potrdf_exchange_large", exchange_large.view())?;
    validate_finite_vector("potrdf_exchange_small", exchange_small.view())?;
    validate_finite_vector(
        "potrdf_exchange_large_coefficient",
        exchange_large_coefficients.view(),
    )?;
    validate_finite_vector(
        "potrdf_exchange_small_coefficient",
        exchange_small_coefficients.view(),
    )?;

    Ok(AtomicOrbitalPotential {
        central_potential,
        central_development_coefficients,
        exchange_large,
        exchange_small,
        exchange_large_coefficients,
        exchange_small_coefficients,
    })
}

pub(super) fn direct_orbital_potential_source(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    rank: usize,
) -> Result<(Array1<Real>, Array1<Real>, usize), AtomMathError> {
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let mut source = Array1::<Real>::zeros(radial_count);
    let mut source_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut source_len = 0usize;

    for orbital in 0..input.active_lengths.len() {
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        if rank > orbital_j2 {
            source_len = source_len.max(input.active_lengths[orbital]);
            continue;
        }

        let scale = atomic_direct_coulomb_coefficient(
            input.coulomb_coefficients,
            active_orbital,
            orbital,
            rank,
        )? / active_occupation;
        if scale != 0.0 {
            for row in 0..input.active_lengths[orbital] {
                source[row] += scale
                    * (input.large_components[(row, orbital)].powi(2)
                        + input.small_components[(row, orbital)].powi(2));
            }

            let origin_power_start = kappa_angular_rank(input.kappas[orbital])?
                .checked_add(1)
                .and_then(|value| value.checked_sub(rank))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            let max_terms = coefficient_count
                .checked_add(2)
                .and_then(|value| value.checked_sub(origin_power_start))
                .unwrap_or(0);
            if max_terms > 0 {
                let coefficient_scale = scale * input.origin_scales[orbital].powi(2);
                let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                for term in 1..=max_terms {
                    let target_1based = origin_power_start
                        .checked_add(term)
                        .and_then(|value| value.checked_sub(2))
                        .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                            angular_momentum: rank,
                        })?;
                    if target_1based == 0 || target_1based > coefficient_count {
                        continue;
                    }
                    source_coefficients[target_1based - 1] += coefficient_scale
                        * (polynomial_product_coefficient_view(
                            large_coefficients,
                            large_coefficients,
                            term,
                        )? + polynomial_product_coefficient_view(
                            small_coefficients,
                            small_coefficients,
                            term,
                        )?);
                }
            }
        }
        source_len = source_len.max(input.active_lengths[orbital]);
    }

    Ok((source, source_coefficients, source_len))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_orbital_exchange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    active_j2: usize,
    mut exchange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    let coefficient_count = input.nuclear_development_coefficients.len();
    for orbital in 0..input.active_lengths.len() {
        if orbital == active_orbital {
            continue;
        }
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        let maximum_rank = (orbital_j2 + active_j2) / 2;
        let mut rank = orbital_j2.abs_diff(maximum_rank);
        if input.kappas[orbital].signum() * input.kappas[active_orbital].signum() < 0 {
            rank = rank
                .checked_add(1)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }

        while rank <= maximum_rank {
            let scale = atomic_exchange_coulomb_coefficient(
                input.coulomb_coefficients,
                orbital,
                active_orbital,
                rank,
            )? / active_occupation;
            if scale != 0.0 {
                let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
                    left_orbital_1based: orbital + 1,
                    right_orbital_1based: active_orbital + 1,
                    large_small: false,
                    angular_momentum: rank,
                    step: input.step,
                    radii: input.radii,
                    active_lengths: input.active_lengths,
                    orbital_powers: input.orbital_powers,
                    large_components: input.large_components,
                    small_components: input.small_components,
                    large_coefficients: input.large_coefficients,
                    small_coefficients: input.small_coefficients,
                })?;
                for row in 0..input.active_lengths[orbital] {
                    exchange_large_work[row] +=
                        scale * transform.yk[row] * input.large_components[(row, orbital)];
                    exchange_small_work[row] +=
                        scale * transform.yk[row] * input.small_components[(row, orbital)];
                }

                let orbital_abs = kappa_abs_usize(input.kappas[orbital])?;
                let active_abs = kappa_abs_usize(input.kappas[active_orbital])?;
                let origin_shift = rank
                    .checked_add(1)
                    .and_then(|value| value.checked_add(orbital_abs))
                    .and_then(|value| value.checked_sub(active_abs))
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if origin_shift <= coefficient_count {
                    let origin_scale =
                        scale * transform.origin_constant * input.origin_scales[orbital]
                            / input.origin_scales[active_orbital];
                    for coefficient_1based in origin_shift..=coefficient_count {
                        let source_index = coefficient_1based - origin_shift;
                        exchange_large_coefficients[coefficient_1based - 1] +=
                            input.large_coefficients[(source_index, orbital)] * origin_scale;
                        exchange_small_coefficients[coefficient_1based - 1] +=
                            input.small_coefficients[(source_index, orbital)] * origin_scale;
                    }
                }

                let exchange_start = kappa_angular_rank(input.kappas[orbital])?
                    .checked_add(2)
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if exchange_start <= coefficient_count {
                    let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                    let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                    let source_scale = scale * input.origin_scales[orbital].powi(2);
                    for coefficient_1based in exchange_start..=coefficient_count {
                        let term = coefficient_1based + 1 - exchange_start;
                        exchange_large_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                large_coefficients,
                                term,
                            )?;
                        exchange_small_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                small_coefficients,
                                term,
                            )?;
                    }
                }
            }

            rank = rank
                .checked_add(2)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }
    }
    Ok(())
}

pub(super) fn accumulate_orbital_lagrange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    mut lagrange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut lagrange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    for orbital in 0..input.self_consistent_count {
        if input.kappas[orbital] != input.kappas[active_orbital] || orbital == active_orbital {
            continue;
        }
        if input.shell_markers[orbital] < 0 && input.shell_markers[active_orbital] < 0 {
            continue;
        }

        let packed = packed_orbital_pair_index(orbital, active_orbital)?;
        let scale = input.lagrange_parameters[packed] * input.occupations[orbital];
        for row in 0..input.active_lengths[orbital] {
            lagrange_large_work[row] += scale * input.large_components[(row, orbital)];
            lagrange_small_work[row] += scale * input.small_components[(row, orbital)];
        }
        for coefficient in 0..input.nuclear_development_coefficients.len() {
            exchange_large_coefficients[coefficient] +=
                input.large_coefficients[(coefficient, orbital)] * scale;
            exchange_small_coefficients[coefficient] +=
                input.small_coefficients[(coefficient, orbital)] * scale;
        }
    }
    Ok(())
}
