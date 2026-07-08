use super::*;

/// Compose FEFF `ATOM/inmuat.f90`, `wfirdf.f90`, `muatco.f90`, and the
/// positive-`niter` `scfdat.f90` scheduler for one atomic state.
pub fn atomic_scf_state_from_configuration(
    input: AtomicScfStateInput<'_>,
) -> Result<AtomicScfState, AtomMathError> {
    validate_finite_scalar("atomic_scf_state_ionicity", input.ionicity)?;
    validate_finite_scalar(
        "atomic_scf_state_thomas_fermi_ionicity",
        input.thomas_fermi_ionicity,
    )?;
    validate_positive_finite_scalar("atomic_scf_state_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("atomic_scf_state_step", input.step)?;
    validate_positive_finite_scalar(
        "atomic_scf_state_first_radius_times_charge",
        input.first_radius_times_charge,
    )?;

    let principal_quantum_numbers = input
        .configuration
        .principal_quantum_numbers
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value <= 0 {
                return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                    orbital_1based: index + 1,
                    principal_quantum_number: 0,
                });
            }
            usize::try_from(value).map_err(|_| AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: index + 1,
                principal_quantum_number: 0,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let principal_quantum_numbers = Array1::from_vec(principal_quantum_numbers);
    let kappas = input.configuration.kappa.clone();
    let occupations = input.configuration.electron_counts.clone();
    let valence_occupations = input.configuration.valence_counts.clone();
    let coefficient_valence_occupations = scf_coefficient_valence_occupations(
        input.exchange_mode,
        occupations.view(),
        valence_occupations.view(),
    );

    let orbital_initialization = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number: input.atomic_number,
        ionicity: input.ionicity,
        principal_quantum_numbers: principal_quantum_numbers.as_slice().unwrap_or(&[]),
        kappas: kappas.as_slice().unwrap_or(&[]),
        occupations: occupations.as_slice().unwrap_or(&[]),
    })?;

    let initial_orbitals = atomic_initial_orbitals(AtomicInitialOrbitalsInput {
        nuclear_charge: input.atomic_number as Real,
        thomas_fermi_ionicity: input.thomas_fermi_ionicity,
        principal_quantum_numbers: principal_quantum_numbers.as_slice().unwrap_or(&[]),
        kappas: kappas.as_slice().unwrap_or(&[]),
        active_lengths: orbital_initialization
            .active_lengths
            .as_slice()
            .unwrap_or(&[]),
        speed_of_light: input.speed_of_light,
        step: input.step,
        requested_nucleus_index: input.requested_nucleus_index,
        radial_count: orbital_initialization.radial_count,
        coefficient_count: orbital_initialization.development_order,
        first_radius_times_charge: input.first_radius_times_charge,
        primary_matching_precision: orbital_initialization.primary_matching_precision,
        secondary_matching_precision: orbital_initialization.secondary_matching_precision,
        max_attempt_count: orbital_initialization.attempt_count,
    })?;

    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: kappas.as_slice().unwrap_or(&[]),
        occupations: occupations.as_slice().unwrap_or(&[]),
        valence_occupations: coefficient_valence_occupations.as_slice().unwrap_or(&[]),
    })?;

    let scf = atomic_self_consistent_orbitals(AtomicScfInput {
        nuclear_charge: input.atomic_number as Real,
        exchange_mode: input.exchange_mode,
        include_lagrange: orbital_initialization.lagrange_pair_count != 0,
        self_consistent_count: orbital_initialization.self_consistent_count,
        max_orbital_iterations: input.max_orbital_iterations,
        wavefunction_precision: orbital_initialization.wavefunction_precision,
        energy_precision: orbital_initialization.energy_precision,
        speed_of_light: input.speed_of_light,
        step: input.step,
        radii: initial_orbitals.radii.view(),
        active_lengths: initial_orbitals.active_lengths.as_slice().unwrap_or(&[]),
        principal_quantum_numbers: principal_quantum_numbers.as_slice().unwrap_or(&[]),
        kappas: kappas.as_slice().unwrap_or(&[]),
        orbital_powers: initial_orbitals.orbital_powers.as_slice().unwrap_or(&[]),
        occupations: occupations.as_slice().unwrap_or(&[]),
        valence_occupations: valence_occupations.as_slice().unwrap_or(&[]),
        shell_markers: orbital_initialization
            .shell_markers
            .as_slice()
            .unwrap_or(&[]),
        origin_scales: initial_orbitals.origin_scales.as_slice().unwrap_or(&[]),
        coulomb_coefficients: coulomb_coefficients.view(),
        lagrange_parameters: orbital_initialization.lagrange_parameters.view(),
        nuclear_potential: initial_orbitals.nuclear_potential.view(),
        nuclear_development_coefficients: initial_orbitals.nuclear_development_coefficients.view(),
        large_components: initial_orbitals.large_components.view(),
        small_components: initial_orbitals.small_components.view(),
        large_coefficients: initial_orbitals.large_coefficients.view(),
        small_coefficients: initial_orbitals.small_coefficients.view(),
        orbital_energies: initial_orbitals.orbital_energies.as_slice().unwrap_or(&[]),
        convergence_acceleration: orbital_initialization
            .convergence_acceleration
            .as_slice()
            .unwrap_or(&[]),
        wavefunction_errors: orbital_initialization
            .wavefunction_errors
            .as_slice()
            .unwrap_or(&[]),
        energy_errors: orbital_initialization
            .energy_errors
            .as_slice()
            .unwrap_or(&[]),
        primary_matching_precision: orbital_initialization.primary_matching_precision,
        secondary_matching_precision: orbital_initialization.secondary_matching_precision,
        max_attempt_count: orbital_initialization.attempt_count,
    })?;

    Ok(AtomicScfState {
        principal_quantum_numbers,
        kappas,
        occupations,
        valence_occupations,
        orbital_initialization,
        initial_orbitals,
        scf,
    })
}

fn scf_coefficient_valence_occupations(
    exchange_mode: AtomicLocalDensityExchangeMode,
    occupations: ArrayView1<'_, Real>,
    valence_occupations: ArrayView1<'_, Real>,
) -> Array1<Real> {
    match exchange_mode {
        AtomicLocalDensityExchangeMode::DiracFockOnly => Array1::zeros(occupations.len()),
        AtomicLocalDensityExchangeMode::TotalDensity => occupations.to_owned(),
        AtomicLocalDensityExchangeMode::ValenceDensity
        | AtomicLocalDensityExchangeMode::CoreDensitySeparated => valence_occupations.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scf_coefficient_valence_occupations_match_scfdat_idfock_branches() {
        let occupations = Array1::from_vec(vec![2.0, 1.5]);
        let valence = Array1::from_vec(vec![0.0, 1.5]);

        assert_eq!(
            scf_coefficient_valence_occupations(
                AtomicLocalDensityExchangeMode::DiracFockOnly,
                occupations.view(),
                valence.view(),
            )
            .to_vec(),
            vec![0.0, 0.0]
        );
        assert_eq!(
            scf_coefficient_valence_occupations(
                AtomicLocalDensityExchangeMode::TotalDensity,
                occupations.view(),
                valence.view(),
            ),
            occupations
        );
        assert_eq!(
            scf_coefficient_valence_occupations(
                AtomicLocalDensityExchangeMode::CoreDensitySeparated,
                occupations.view(),
                valence.view(),
            ),
            valence
        );
    }
}
