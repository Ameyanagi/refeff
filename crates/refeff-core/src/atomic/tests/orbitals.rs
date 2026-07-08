#![allow(clippy::excessive_precision)]

use super::*;

#[allow(clippy::excessive_precision)]
#[test]
fn atom_orbital_initialization_matches_feff_inmuat_reference() -> Result<(), AtomMathError> {
    let open_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number: 4,
        ionicity: 0.0,
        principal_quantum_numbers: &[2, 3, 1],
        kappas: &[1, 1, -1],
        occupations: &[0.4, 1.6, 2.0],
    })?;

    assert_eq!(open_shell.orbital_count, 3);
    assert_eq!(open_shell.self_consistent_count, 3);
    assert_eq!(open_shell.lagrange_pair_count, 1);
    assert_eq!(open_shell.radial_count, 251);
    assert_eq!(open_shell.development_order, 10);
    assert_eq!(open_shell.attempt_count, 50);
    assert_eq!(open_shell.nucleus_index, 11);
    assert_close_with(
        open_shell.wavefunction_precision,
        1.000_000_000_000_000_08e-5,
        1.0e-20,
    );
    assert_close_with(
        open_shell.energy_precision,
        5.000_000_000_000_000_41e-6,
        1.0e-20,
    );
    assert_close(open_shell.precision_ratios[0], 100.0);
    assert_close(open_shell.precision_ratios[1], 10.0);
    assert_close_with(open_shell.primary_matching_precision, 1.0e-7, 1.0e-20);
    assert_close_with(open_shell.secondary_matching_precision, 1.0e-6, 1.0e-20);
    assert_eq!(open_shell.shell_markers.to_vec(), vec![1, 1, -1]);
    assert_eq!(open_shell.active_lengths.to_vec(), vec![251, 251, 251]);
    assert_close_with(open_shell.convergence_acceleration[0], 1.0, 1.0e-16);
    assert_close_with(
        open_shell.convergence_acceleration[1],
        3.000_000_119_209_289_55e-1,
        1.0e-16,
    );
    assert_close_with(
        open_shell.convergence_acceleration[2],
        3.000_000_119_209_289_55e-1,
        1.0e-16,
    );
    assert!(
        open_shell
            .orbital_energies
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        open_shell
            .wavefunction_errors
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(open_shell.energy_errors.iter().all(|&value| value == 0.0));
    assert_eq!(open_shell.lagrange_parameters.len(), 820);
    assert!(
        open_shell
            .lagrange_parameters
            .iter()
            .all(|&value| value == 0.0)
    );

    let closed_shell = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number: 10,
        ionicity: 0.0,
        principal_quantum_numbers: &[1, 2, 2, 2],
        kappas: &[-1, -1, 1, -2],
        occupations: &[2.0, 2.0, 2.0, 4.0],
    })?;
    assert_eq!(closed_shell.orbital_count, 4);
    assert_eq!(closed_shell.self_consistent_count, 4);
    assert_eq!(closed_shell.lagrange_pair_count, 0);
    assert_eq!(closed_shell.shell_markers.to_vec(), vec![-1, -1, -1, -1]);
    assert_eq!(
        closed_shell.active_lengths.to_vec(),
        vec![251, 251, 251, 251]
    );
    for value in closed_shell.convergence_acceleration {
        assert_close_with(value, 3.000_000_119_209_289_55e-1, 1.0e-16);
    }
    Ok(())
}

fn sample_oxygen_initial_orbitals() -> Result<AtomicInitialOrbitals, AtomMathError> {
    atomic_initial_orbitals(AtomicInitialOrbitalsInput {
        nuclear_charge: 8.0,
        thomas_fermi_ionicity: -1.0,
        principal_quantum_numbers: &[1, 2],
        kappas: &[-1, 1],
        active_lengths: &[251, 251],
        speed_of_light: 137.0373,
        step: 0.05,
        requested_nucleus_index: 11,
        radial_count: 251,
        coefficient_count: 10,
        first_radius_times_charge: 8.0 * (-8.8_f64).exp(),
        primary_matching_precision: 1.0e-7,
        secondary_matching_precision: 1.0e-6,
        max_attempt_count: 50,
    })
}

#[test]
fn atom_initial_orbitals_composes_feff_wfirdf_driver() -> Result<(), AtomMathError> {
    let initial = sample_oxygen_initial_orbitals()?;

    assert_eq!(initial.nucleus_index, 1);
    assert_eq!(initial.radii.len(), 251);
    assert_eq!(initial.active_lengths.to_vec(), vec![213, 243]);
    assert_eq!(initial.large_components.dim(), (251, 2));
    assert_eq!(initial.large_coefficients.dim(), (10, 2));
    assert_eq!(initial.attempts_exhausted, vec![false, false]);
    assert_close_with(initial.radii[0], 1.507_330_750_954_765_0e-4, 1.0e-18);
    assert_close_with(initial.radii[176], 1.0, 1.0e-15);
    assert_close_with(
        initial.nuclear_potential[0],
        -5.307_395_205_022_312_0e4,
        1.0e-8,
    );
    assert_close_with(initial.nuclear_potential[176], -8.0, 1.0e-12);
    assert_close_with(initial.potential[0], -3.871_230_377_714_645_7e2, 1.0e-10);
    assert_close_with(
        initial.potential_coefficients[0],
        -5.837_826_635_521_862_0e-2,
        1.0e-16,
    );
    assert_close_with(
        initial.potential_coefficients[1],
        1.726_258_925_387_370_9e-1,
        1.0e-15,
    );
    assert_close_with(
        initial.orbital_powers[0],
        9.982_945_347_027_394_0e-1,
        1.0e-16,
    );
    assert_close_with(initial.origin_scales[0], 1.015_121_281_599_321_6, 1.0e-15);
    assert_close_with(
        initial.orbital_energies[0],
        -1.878_937_053_082_044_0e1,
        1.0e-12,
    );
    assert_close_with(
        initial.orbital_energies[1],
        -9.575_411_173_389_776_0e-1,
        1.0e-14,
    );
    assert_close_with(
        initial.large_components[(0, 0)],
        6.369_982_540_733_882_0e-3,
        1.0e-17,
    );
    assert_close_with(
        initial.small_components[(0, 0)],
        -1.860_685_608_607_028_0e-4,
        1.0e-18,
    );
    assert_close_with(
        initial.large_components[(0, 1)],
        1.158_490_630_013_305_7e-6,
        1.0e-20,
    );
    assert_close_with(
        initial.small_components[(0, 1)],
        2.694_531_046_713_704_0e-5,
        1.0e-19,
    );
    assert_close_with(initial.large_components[(213, 0)], 0.0, 1.0e-18);
    assert_close_with(initial.small_components[(213, 0)], 0.0, 1.0e-18);
    assert_close_with(initial.large_components[(243, 1)], 0.0, 1.0e-18);
    assert_close_with(initial.small_components[(243, 1)], 0.0, 1.0e-18);
    assert_close_with(
        initial.large_coefficients[(0, 0)],
        4.168_071_627_670_580_0e1,
        1.0e-12,
    );
    assert_close_with(
        initial.small_coefficients[(0, 0)],
        -1.217_662_318_753_121_5,
        1.0e-14,
    );
    assert_close_with(
        initial.large_coefficients[(1, 1)],
        1.608_131_964_517_360_0e1,
        1.0e-12,
    );
    assert_close_with(
        initial.small_coefficients[(1, 1)],
        -9.394_770_324_558_013_0e-1,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn atom_scf_orbital_iteration_composes_feff_scfdat_body() -> Result<(), AtomMathError> {
    let initial = sample_oxygen_initial_orbitals()?;
    let principal_quantum_numbers = [1, 2];
    let kappas = [-1, 1];
    let occupations = [2.0, 2.0];
    let valence_occupations = [0.0, 0.0];
    let shell_markers = [-1, -1];
    let active_lengths = initial.active_lengths.to_vec();
    let orbital_powers = initial.orbital_powers.to_vec();
    let origin_scales = initial.origin_scales.to_vec();
    let orbital_energies = initial.orbital_energies.to_vec();
    let convergence_acceleration = vec![3.000_000_119_209_289_55e-1; 2];
    let wavefunction_errors = vec![0.0; 2];
    let lagrange_parameters = ndarray::Array1::<Real>::zeros(1);
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let input = AtomicScfOrbitalIterationInput {
        active_orbital_1based: 1,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        include_lagrange: false,
        self_consistent_count: 2,
        speed_of_light: 137.0373,
        step: 0.05,
        radii: initial.radii.view(),
        active_lengths: &active_lengths,
        principal_quantum_numbers: &principal_quantum_numbers,
        kappas: &kappas,
        orbital_powers: &orbital_powers,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
        shell_markers: &shell_markers,
        origin_scales: &origin_scales,
        coulomb_coefficients: coulomb_coefficients.view(),
        lagrange_parameters: lagrange_parameters.view(),
        nuclear_potential: initial.nuclear_potential.view(),
        nuclear_development_coefficients: initial.nuclear_development_coefficients.view(),
        large_components: initial.large_components.view(),
        small_components: initial.small_components.view(),
        large_coefficients: initial.large_coefficients.view(),
        small_coefficients: initial.small_coefficients.view(),
        orbital_energies: &orbital_energies,
        convergence_acceleration: &convergence_acceleration,
        wavefunction_errors: &wavefunction_errors,
        primary_matching_precision: 1.0e-7,
        secondary_matching_precision: 1.0e-6,
        max_attempt_count: 50,
    };
    assert!(matches!(
        atomic_scf_orbital_iteration(AtomicScfOrbitalIterationInput {
            orbital_energies: &orbital_energies[..1],
            ..input
        }),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));

    let iteration = atomic_scf_orbital_iteration(input)?;

    assert_eq!(iteration.active_orbital_1based, 1);
    assert_eq!(iteration.active_len, 211);
    assert_eq!(iteration.large_component.len(), 251);
    assert_eq!(iteration.large_coefficients.len(), 10);
    assert!(!iteration.attempts_exhausted);
    assert_close_with(
        iteration.orbital_energy,
        -2.493_645_853_400_927_2e1,
        1.0e-12,
    );
    assert_close_with(
        iteration.large_component[0],
        6.439_905_248_791_872_0e-3,
        1.0e-17,
    );
    assert_close_with(
        iteration.small_component[0],
        -1.881_165_810_048_878_4e-4,
        1.0e-18,
    );
    assert_close_with(iteration.large_component[211], 0.0, 1.0e-18);
    assert_close_with(iteration.small_component[211], 0.0, 1.0e-18);
    assert_close_with(
        iteration.large_coefficients[0],
        4.213_824_769_371_194_6e1,
        1.0e-12,
    );
    assert_close_with(
        iteration.large_coefficients[1],
        -3.369_851_268_521_291_6e2,
        1.0e-10,
    );
    assert_close_with(
        iteration.small_coefficients[0],
        -1.231_028_662_134_448_2,
        1.0e-14,
    );
    assert_close_with(
        iteration.convergence_acceleration,
        3.000_000_119_209_289_6e-1,
        1.0e-16,
    );
    assert_close_with(
        iteration.wavefunction_error,
        -5.040_643_523_308_796_5e-2,
        1.0e-16,
    );
    assert_close_with(iteration.energy_error, 2.465_100_645_629_050_6e-1, 1.0e-15);
    assert_close_with(iteration.normalization, 9.997_874_033_473_430_0e-1, 1.0e-15);
    assert_close_with(
        iteration.total_density[0],
        8.122_405_294_090_392_0e-5,
        1.0e-18,
    );
    assert_close_with(iteration.valence_density[0], 0.0, 1.0e-18);
    assert_close_with(iteration.potential[0], -3.872_236_540_539_925_0e2, 1.0e-10);
    assert_close_with(
        iteration.potential_coefficients[0],
        -5.837_826_635_521_862_0e-2,
        1.0e-16,
    );
    assert_close_with(
        iteration.potential_coefficients[1],
        7.200_965_946_419_544_0e-2,
        1.0e-16,
    );
    Ok(())
}

#[test]
fn atom_self_consistent_orbitals_composes_feff_scfdat_loop() -> Result<(), AtomMathError> {
    let initial = sample_oxygen_initial_orbitals()?;
    let principal_quantum_numbers = [1, 2];
    let kappas = [-1, 1];
    let occupations = [2.0, 2.0];
    let valence_occupations = [0.0, 0.0];
    let shell_markers = [-1, -1];
    let active_lengths = initial.active_lengths.to_vec();
    let orbital_powers = initial.orbital_powers.to_vec();
    let origin_scales = initial.origin_scales.to_vec();
    let orbital_energies = initial.orbital_energies.to_vec();
    let convergence_acceleration = vec![3.000_000_119_209_289_55e-1; 2];
    let wavefunction_errors = vec![0.0; 2];
    let energy_errors = vec![0.0; 2];
    let lagrange_parameters = ndarray::Array1::<Real>::zeros(1);
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let input = AtomicScfInput {
        nuclear_charge: 8.0,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        include_lagrange: false,
        self_consistent_count: 2,
        max_orbital_iterations: 40,
        wavefunction_precision: 1.0e-5,
        energy_precision: 5.0e-6,
        speed_of_light: 137.0373,
        step: 0.05,
        radii: initial.radii.view(),
        active_lengths: &active_lengths,
        principal_quantum_numbers: &principal_quantum_numbers,
        kappas: &kappas,
        orbital_powers: &orbital_powers,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
        shell_markers: &shell_markers,
        origin_scales: &origin_scales,
        coulomb_coefficients: coulomb_coefficients.view(),
        lagrange_parameters: lagrange_parameters.view(),
        nuclear_potential: initial.nuclear_potential.view(),
        nuclear_development_coefficients: initial.nuclear_development_coefficients.view(),
        large_components: initial.large_components.view(),
        small_components: initial.small_components.view(),
        large_coefficients: initial.large_coefficients.view(),
        small_coefficients: initial.small_coefficients.view(),
        orbital_energies: &orbital_energies,
        convergence_acceleration: &convergence_acceleration,
        wavefunction_errors: &wavefunction_errors,
        energy_errors: &energy_errors,
        primary_matching_precision: 1.0e-7,
        secondary_matching_precision: 1.0e-6,
        max_attempt_count: 50,
    };
    assert!(matches!(
        atomic_self_consistent_orbitals(AtomicScfInput {
            max_orbital_iterations: 1,
            ..input
        }),
        Err(AtomMathError::ScfIterationLimitExceeded { .. })
    ));

    let scf = atomic_self_consistent_orbitals(input)?;

    assert_eq!(scf.iteration_count, 22);
    assert_eq!(scf.active_lengths.to_vec(), vec![209, 229]);
    assert_eq!(scf.large_components.dim(), (251, 2));
    assert_eq!(scf.large_coefficients.dim(), (10, 2));
    assert_eq!(scf.attempts_exhausted, vec![false, false]);
    assert_close_with(scf.orbital_energies[0], -2.437_299_436_914_503_3e1, 1.0e-12);
    assert_close_with(scf.orbital_energies[1], -3.557_783_055_677_064_0, 1.0e-13);
    assert_close_with(
        scf.large_components[(0, 0)],
        6.579_957_095_720_304_0e-3,
        1.0e-17,
    );
    assert_close_with(
        scf.small_components[(0, 0)],
        -1.922_200_061_528_998_3e-4,
        1.0e-18,
    );
    assert_close_with(
        scf.large_components[(0, 1)],
        1.525_059_102_844_499_6e-6,
        1.0e-20,
    );
    assert_close_with(
        scf.small_components[(0, 1)],
        3.547_002_899_034_184_0e-5,
        1.0e-19,
    );
    assert_close_with(scf.large_components[(209, 0)], 0.0, 1.0e-18);
    assert_close_with(scf.small_components[(209, 0)], 0.0, 1.0e-18);
    assert_close_with(scf.large_components[(229, 1)], 0.0, 1.0e-18);
    assert_close_with(scf.small_components[(229, 1)], 0.0, 1.0e-18);
    assert_close_with(
        scf.large_coefficients[(0, 0)],
        4.305_466_274_205_934_0e1,
        1.0e-12,
    );
    assert_close_with(
        scf.large_coefficients[(1, 1)],
        2.117_138_523_095_522_7e1,
        1.0e-12,
    );
    assert_close_with(
        scf.small_coefficients[(0, 0)],
        -1.257_800_852_547_466,
        1.0e-14,
    );
    assert_close_with(
        scf.small_coefficients[(0, 1)],
        2.319_979_207_186_439_7e-1,
        1.0e-15,
    );
    assert_close_with(
        scf.convergence_acceleration[0],
        8.000_000_119_209_288_0e-1,
        1.0e-16,
    );
    assert_close_with(
        scf.convergence_acceleration[1],
        8.000_000_119_209_288_0e-1,
        1.0e-16,
    );
    assert_close_with(
        scf.wavefunction_errors[0],
        1.213_701_739_555_261_8e-6,
        1.0e-18,
    );
    assert_close_with(
        scf.wavefunction_errors[1],
        1.851_259_518_598_214_0e-6,
        1.0e-18,
    );
    assert_close_with(scf.energy_errors[0], 3.806_087_630_124_143_0e-7, 1.0e-19);
    assert_close_with(scf.energy_errors[1], 4.638_313_913_425_985_0e-7, 1.0e-19);
    assert_close_with(scf.lagrange_parameters[0], 0.0, 1.0e-18);
    assert_close_with(scf.total_density[0], 8.666_808_872_209_448_0e-5, 1.0e-18);
    assert_close_with(scf.valence_density[0], 0.0, 1.0e-18);
    assert_close_with(scf.energy_density[0], 0.0, 1.0e-18);
    assert_close_with(scf.density_4pi[0], 3.814_539_362_219_717_0e3, 1.0e-9);
    assert_close_with(scf.valence_density_4pi[0], 0.0, 1.0e-18);
    assert_close_with(scf.coulomb_potential[0], -5.305_567_405_624_433_0e4, 1.0e-8);
    Ok(())
}

#[test]
fn atom_scf_state_from_configuration_composes_atom_state_driver() -> Result<(), AtomMathError> {
    let configuration = crate::OrbitalConfiguration {
        orbital_count: 2,
        core_orbital_count: 2,
        projection_orbitals: ndarray::Array1::zeros(10),
        hole_position: 0,
        principal_quantum_numbers: ndarray::Array1::from_vec(vec![1, 2]),
        kappa: ndarray::Array1::from_vec(vec![-1, -1]),
        electron_counts: ndarray::Array1::from_vec(vec![2.0, 2.0]),
        valence_counts: ndarray::Array1::from_vec(vec![0.0, 0.0]),
        spin_magnetization: ndarray::Array1::zeros(2),
        ionization_orbital: 0,
        screening_orbital: 0,
        last_occupied_orbital: 2,
        template_atomic_number: 4,
        ionicity_delta: 0.0,
    };
    let atomic_number = 4;
    let speed_of_light = 137.0373;
    let step = 0.05;
    let first_radius_times_charge = atomic_number as Real * (-8.8_f64).exp();

    let state = atomic_scf_state_from_configuration(AtomicScfStateInput {
        atomic_number,
        ionicity: 0.0,
        thomas_fermi_ionicity: -1.0,
        configuration: &configuration,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        max_orbital_iterations: 40,
        speed_of_light,
        step,
        requested_nucleus_index: 11,
        first_radius_times_charge,
    })?;

    assert_eq!(state.principal_quantum_numbers.to_vec(), vec![1, 2]);
    assert_eq!(state.kappas.to_vec(), vec![-1, -1]);
    assert_eq!(state.occupations.to_vec(), vec![2.0, 2.0]);
    assert_eq!(state.valence_occupations.to_vec(), vec![0.0, 0.0]);

    let setup = atomic_orbital_initialization(AtomicOrbitalInitializationInput {
        atomic_number,
        ionicity: 0.0,
        principal_quantum_numbers: &[1, 2],
        kappas: &[-1, -1],
        occupations: &[2.0, 2.0],
    })?;
    let initial = atomic_initial_orbitals(AtomicInitialOrbitalsInput {
        nuclear_charge: atomic_number as Real,
        thomas_fermi_ionicity: -1.0,
        principal_quantum_numbers: &[1, 2],
        kappas: &[-1, -1],
        active_lengths: setup.active_lengths.as_slice().unwrap_or(&[]),
        speed_of_light,
        step,
        requested_nucleus_index: setup.nucleus_index as isize,
        radial_count: setup.radial_count,
        coefficient_count: setup.development_order,
        first_radius_times_charge,
        primary_matching_precision: setup.primary_matching_precision,
        secondary_matching_precision: setup.secondary_matching_precision,
        max_attempt_count: setup.attempt_count,
    })?;
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &[-1, -1],
        occupations: &[2.0, 2.0],
        valence_occupations: &[0.0, 0.0],
    })?;
    let expected = atomic_self_consistent_orbitals(AtomicScfInput {
        nuclear_charge: atomic_number as Real,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        include_lagrange: setup.lagrange_pair_count != 0,
        self_consistent_count: setup.self_consistent_count,
        max_orbital_iterations: 40,
        wavefunction_precision: setup.wavefunction_precision,
        energy_precision: setup.energy_precision,
        speed_of_light,
        step,
        radii: initial.radii.view(),
        active_lengths: initial.active_lengths.as_slice().unwrap_or(&[]),
        principal_quantum_numbers: &[1, 2],
        kappas: &[-1, -1],
        orbital_powers: initial.orbital_powers.as_slice().unwrap_or(&[]),
        occupations: &[2.0, 2.0],
        valence_occupations: &[0.0, 0.0],
        shell_markers: setup.shell_markers.as_slice().unwrap_or(&[]),
        origin_scales: initial.origin_scales.as_slice().unwrap_or(&[]),
        coulomb_coefficients: coulomb_coefficients.view(),
        lagrange_parameters: setup.lagrange_parameters.view(),
        nuclear_potential: initial.nuclear_potential.view(),
        nuclear_development_coefficients: initial.nuclear_development_coefficients.view(),
        large_components: initial.large_components.view(),
        small_components: initial.small_components.view(),
        large_coefficients: initial.large_coefficients.view(),
        small_coefficients: initial.small_coefficients.view(),
        orbital_energies: initial.orbital_energies.as_slice().unwrap_or(&[]),
        convergence_acceleration: setup.convergence_acceleration.as_slice().unwrap_or(&[]),
        wavefunction_errors: setup.wavefunction_errors.as_slice().unwrap_or(&[]),
        energy_errors: setup.energy_errors.as_slice().unwrap_or(&[]),
        primary_matching_precision: setup.primary_matching_precision,
        secondary_matching_precision: setup.secondary_matching_precision,
        max_attempt_count: setup.attempt_count,
    })?;

    assert_eq!(state.orbital_initialization.orbital_count, 2);
    assert_eq!(state.orbital_initialization.self_consistent_count, 2);
    assert_eq!(
        state.initial_orbitals.active_lengths,
        initial.active_lengths
    );
    assert_eq!(state.scf.iteration_count, expected.iteration_count);
    assert_eq!(state.scf.active_lengths, expected.active_lengths);
    assert_close_with(
        state.scf.orbital_energies[0],
        expected.orbital_energies[0],
        1.0e-12,
    );
    assert_close_with(
        state.scf.orbital_energies[1],
        expected.orbital_energies[1],
        1.0e-12,
    );
    assert_close_with(
        state.scf.large_components[(0, 0)],
        expected.large_components[(0, 0)],
        1.0e-17,
    );
    assert_close_with(
        state.scf.large_coefficients[(1, 1)],
        expected.large_coefficients[(1, 1)],
        1.0e-12,
    );
    assert_close_with(state.scf.density_4pi[0], expected.density_4pi[0], 1.0e-9);
    assert_close_with(
        state.scf.coulomb_potential[0],
        expected.coulomb_potential[0],
        1.0e-8,
    );
    Ok(())
}

#[test]
fn atom_scf_state_from_configuration_uses_finite_nucleus_request() -> Result<(), AtomMathError> {
    let configuration = crate::OrbitalConfiguration {
        orbital_count: 2,
        core_orbital_count: 2,
        projection_orbitals: ndarray::Array1::zeros(10),
        hole_position: 0,
        principal_quantum_numbers: ndarray::Array1::from_vec(vec![1, 2]),
        kappa: ndarray::Array1::from_vec(vec![-1, -1]),
        electron_counts: ndarray::Array1::from_vec(vec![2.0, 2.0]),
        valence_counts: ndarray::Array1::from_vec(vec![0.0, 0.0]),
        spin_magnetization: ndarray::Array1::zeros(2),
        ionization_orbital: 0,
        screening_orbital: 0,
        last_occupied_orbital: 2,
        template_atomic_number: 4,
        ionicity_delta: 0.0,
    };
    let atomic_number = 4;
    let speed_of_light = 137.0373;
    let step = 0.05;
    let first_radius_times_charge = atomic_number as Real * (-8.8_f64).exp();

    let point = atomic_scf_state_from_configuration(AtomicScfStateInput {
        atomic_number,
        ionicity: 0.0,
        thomas_fermi_ionicity: -1.0,
        configuration: &configuration,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        max_orbital_iterations: 40,
        speed_of_light,
        step,
        requested_nucleus_index: 11,
        first_radius_times_charge,
    })?;
    let finite = atomic_scf_state_from_configuration(AtomicScfStateInput {
        atomic_number,
        ionicity: 0.0,
        thomas_fermi_ionicity: -1.0,
        configuration: &configuration,
        exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
        max_orbital_iterations: 40,
        speed_of_light,
        step,
        requested_nucleus_index: -11,
        first_radius_times_charge,
    })?;

    assert_eq!(point.initial_orbitals.nucleus_index, 1);
    assert!(finite.initial_orbitals.nucleus_index > 1);
    assert_ne!(
        finite.initial_orbitals.radii[0],
        point.initial_orbitals.radii[0]
    );
    Ok(())
}
