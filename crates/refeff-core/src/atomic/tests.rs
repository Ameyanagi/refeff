use ndarray::{Array1, Array2, Array3};

use super::*;

fn assert_close(actual: Real, expected: Real) {
    assert_close_with(actual, expected, 1.0e-12);
}

fn assert_close_with(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

fn assert_some_close(actual: Option<Real>, expected: Real, tolerance: Real) {
    match actual {
        Some(value) => assert_close_with(value, expected, tolerance),
        None => assert_eq!(actual, Some(expected)),
    }
}

#[test]
fn atomic_weight_matches_feff_pertab_reference() -> Result<(), AtomicError> {
    assert_close(atomic_weight(1)?, 1.007_899_999_618_530_3);
    assert_close(atomic_weight(2)?, 4.002_600_193_023_682);
    assert_close(atomic_weight(26)?, 55.849_998_474_121_094);
    assert_close(atomic_weight(75)?, 186.199_996_948_242_2);
    assert_close(atomic_weight(92)?, 238.029_998_779_296_88);
    assert_close(atomic_weight(118)?, 294.0);
    assert_close(atomic_weight(121)?, 330.0);
    assert_close(atomic_weight(139)?, 392.0);
    Ok(())
}

#[test]
fn atomic_symbol_matches_feff_pertab_reference() -> Result<(), AtomicError> {
    assert_eq!(atomic_symbol(1)?, "H");
    assert_eq!(atomic_symbol(2)?, "He");
    assert_eq!(atomic_symbol(26)?, "Fe");
    assert_eq!(atomic_symbol(75)?, "Te");
    assert_eq!(atomic_symbol(92)?, "U");
    assert_eq!(atomic_symbol(118)?, "Uuo");
    assert_eq!(atomic_symbol(121)?, "Ubu");
    assert_eq!(atomic_symbol(139)?, "Ute");
    Ok(())
}

#[test]
fn nuclear_mass_matches_feff_reference() -> Result<(), AtomicError> {
    assert_close(nuclear_mass(1)?, 1.007_940_053_939_819_3);
    assert_close(nuclear_mass(6)?, 12.010_700_225_830_078);
    assert_close(nuclear_mass(29)?, 63.546_001_434_326_17);
    assert_close(nuclear_mass(57)?, 138.905_471_801_757_8);
    assert_close(nuclear_mass(92)?, 238.028_915_405_273_44);
    assert_close(nuclear_mass(118)?, 294.0);
    assert_close(nuclear_mass(121)?, 330.0);
    assert_close(nuclear_mass(138)?, 388.0);
    Ok(())
}

#[test]
fn nuclear_mass_rejects_invalid_atomic_numbers() {
    assert_eq!(
        nuclear_mass(0),
        Err(AtomicError::InvalidAtomicNumber { z: 0 })
    );
    assert_eq!(
        nuclear_mass(139),
        Err(AtomicError::InvalidAtomicNumber { z: 139 })
    );
    assert_eq!(
        atomic_weight(140),
        Err(AtomicError::InvalidAtomicNumber { z: 140 })
    );
    assert_eq!(
        atomic_symbol(0),
        Err(AtomicError::InvalidAtomicNumber { z: 0 })
    );
}

#[test]
fn atom_nuclear_potential_matches_feff_nucdev_reference() -> Result<(), AtomMathError> {
    let point = atomic_nuclear_potential(AtomicNuclearPotentialInput {
        nuclear_charge: 26.0,
        step: 0.05,
        requested_nucleus_index: 1,
        radial_count: 251,
        coefficient_count: 10,
        first_radius_times_charge: 26.0 * (-8.8_f64).exp(),
    })?;
    assert_eq!(point.nucleus_index, 1);
    assert_close_with(
        point.first_radius_times_charge,
        3.919_059_952_482e-3,
        5.0e-16,
    );
    assert_close(point.development_coefficients[0], -26.0);
    assert_close_with(point.radii[0], 1.507_330_750_955e-4, 5.0e-16);
    assert_close_with(point.potential[0], -1.724_903_441_632e5, 5.0e-8);
    assert_close_with(point.radii[4], 1.841_057_936_676e-4, 5.0e-16);
    assert_close_with(point.potential[4], -1.412_231_493_754e5, 5.0e-8);

    let finite = atomic_nuclear_potential(AtomicNuclearPotentialInput {
        nuclear_charge: 92.0,
        step: 0.05,
        requested_nucleus_index: -11,
        radial_count: 251,
        coefficient_count: 10,
        first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
    })?;
    assert_eq!(finite.nucleus_index, 11);
    assert_close_with(
        finite.first_radius_times_charge,
        7.842_167_533_588e-3,
        5.0e-15,
    );
    assert_close_with(
        finite.development_coefficients[1],
        -9.819_368_462_521e5,
        5.0e-7,
    );
    assert_close_with(
        finite.development_coefficients[3],
        1.657_185_951_350e13,
        10.0,
    );
    assert_close_with(finite.radii[0], 8.524_095_145_204e-5, 5.0e-16);
    assert_close_with(finite.potential[0], -8.615_253_868_304e5, 5.0e-7);
    assert_close_with(finite.radii[10], 1.405_385_697_937e-4, 5.0e-16);
    assert_close_with(finite.potential[10], -6.546_245_641_680e5, 5.0e-7);
    Ok(())
}

#[test]
fn atom_helper_kernels_match_feff_reference() -> Result<(), AtomMathError> {
    let left = (1..=10)
        .map(|index| 0.1 * index as Real + 0.03)
        .collect::<Vec<_>>();
    let right = (1..=10)
        .map(|index| -0.04 * index as Real + 0.25)
        .collect::<Vec<_>>();
    assert_close(
        atomic_polynomial_product_coefficient(&left, &right, 3)?,
        0.125_300_000_000_000_02,
    );
    assert_close(
        atomic_polynomial_product_coefficient(&left, &right, 7)?,
        0.382_9,
    );

    let mixed = atomic_convergence_mix(0.5, 0.3, 0.2)?;
    assert_close(mixed.initial_weight, 0.4);
    assert_close(mixed.final_weight, 0.6);
    assert_close(mixed.previous_error, 0.3);

    let mixed = atomic_convergence_mix(0.2, 0.5, -0.4)?;
    assert_close(mixed.initial_weight, 0.9);
    assert_close(mixed.final_weight, 0.1);
    assert_close(mixed.previous_error, 0.5);

    let mixed = atomic_convergence_mix(0.9, 0.5, 0.4)?;
    assert_close(mixed.initial_weight, 0.099_999_999_999_999_98);
    assert_close(mixed.final_weight, 0.9);
    assert_close(mixed.previous_error, 0.5);

    assert_close(
        thomas_fermi_density_potential(0.45, 29.0, -1.0)?,
        43.097_863_212_551_05,
    );
    assert_close(
        thomas_fermi_density_potential(1.25, 8.0, -2.5)?,
        3.548_014_948_104_207,
    );
    assert_close(thomas_fermi_density_potential(1.25, 0.0, 0.0)?, 0.0);

    let mut occupations = vec![0.0; 41];
    let mut kappas = vec![1; 41];
    occupations[1] = 1.5;
    occupations[4] = 3.0;
    kappas[1] = -1;
    kappas[4] = -3;
    assert_close(atomic_occupation_product(&occupations, &kappas, 4, 4)?, 7.2);
    assert_close(atomic_occupation_product(&occupations, &kappas, 1, 4)?, 4.5);
    Ok(())
}

#[test]
fn atom_coulomb_coefficient_lookups_match_feff_reference() -> Result<(), AtomMathError> {
    let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
        1000.0 * (row + 1) as Real + 10.0 * (column + 1) as Real + channel as Real
    });

    assert_close(
        atomic_direct_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
        2052.0,
    );
    assert_close(
        atomic_direct_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
        2052.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
        5022.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
        5022.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 4, 4)?,
        0.0,
    );
    Ok(())
}

#[test]
fn atom_coulomb_coefficients_match_feff_muatco_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, 1, -2, 2, -3];
    let occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
    let valence_occupations = [0.0, 0.5, 0.0, 0.25, 0.0];
    let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let expected = [
        [
            [2.0, 3.0, 6.0, 1.0, 8.0],
            [0.5, 2.25, 4.5, 0.75, 6.0],
            [1.000_000_000_000_000_7, 0.0, 6.0, 1.5, 12.0],
            [0.0, 0.0, 0.025_000_000_000_000_026, 0.25, 2.0],
            [0.0, 0.0, 1.199_999_999_999_999_3, 0.0, 12.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.450_000_000_000_000_2,
                -0.400_000_000_000_000_3,
                0.0,
                0.0,
            ],
            [
                0.100_000_000_000_000_03,
                0.0,
                0.096_428_571_428_571_31,
                0.0,
                0.0,
            ],
            [
                0.799_999_999_999_999_5,
                0.428_571_428_571_428_2,
                0.342_857_142_857_142_47,
                0.028_571_428_571_428_536,
                -0.548_571_428_571_427_9,
            ],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.095_238_095_238_094_86,
                -0.228_571_428_571_427_8,
            ],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ];

    for (channel, rows) in expected.iter().enumerate() {
        for (row, columns) in rows.iter().enumerate() {
            for (column, &expected) in columns.iter().enumerate() {
                assert_close_with(coefficients[(row, column, channel)], expected, 1.0e-12);
            }
        }
    }

    Ok(())
}

#[test]
fn atom_breit_coefficients_match_feff_bkmrdf_reference() -> Result<(), AtomMathError> {
    let cases = [
        (
            -1,
            -1,
            1,
            [0.5, 0.333_333_333_333_333_2, 0.5],
            [
                -0.166_666_666_666_666_69,
                0.333_333_333_333_333_37,
                -0.166_666_666_666_666_69,
            ],
        ),
        (
            -1,
            1,
            1,
            [
                1.500_000_000_000_000_4,
                1.000_000_000_000_000_2,
                0.166_666_666_666_666_7,
            ],
            [
                1.500_000_000_000_000_4,
                3.000_000_000_000_001,
                0.833_333_333_333_333_6,
            ],
        ),
        (
            1,
            -2,
            1,
            [
                0.500_000_000_000_000_2,
                0.333_333_333_333_334_8,
                0.100_000_000_000_000_06,
            ],
            [
                -0.166_666_666_666_667_4,
                -0.666_666_666_666_669_6,
                -0.126_666_666_666_667_1,
            ],
        ),
        (
            -2,
            2,
            3,
            [
                0.116_666_666_666_666_78,
                0.033_333_333_333_333_36,
                0.002_380_952_380_952_383,
            ],
            [
                0.070_000_000_000_000_05,
                0.420_000_000_000_000_3,
                0.058_571_428_571_428_62,
            ],
        ),
        (
            -3,
            -3,
            5,
            [
                0.050_505_050_505_050_37,
                0.072_150_072_150_071_99,
                0.050_505_050_505_050_37,
            ],
            [
                -0.039_281_705_948_372_45,
                0.078_563_411_896_744_9,
                -0.039_281_705_948_372_45,
            ],
        ),
        (
            2,
            -4,
            3,
            [
                0.102_380_952_380_952_13,
                0.201_587_301_587_301_2,
                0.254_761_904_761_904_3,
            ],
            [
                0.238_500_881_834_214_8,
                0.721_305_114_638_447_2,
                0.264_320_987_654_320_66,
            ],
        ),
    ];

    for (left, right, rank, magnetic, retarded) in cases {
        let actual = atomic_breit_angular_coefficients(left, right, rank)?;
        for index in 0..3 {
            assert_close(actual.magnetic[index], magnetic[index]);
            assert_close(actual.retarded[index], retarded[index]);
        }
    }
    Ok(())
}

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

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_entry_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method0_negative_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: -0.25,
        method: 0,
    })?;
    assert_close_with(method0_negative_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(
        method0_negative_tail.asymptotic_large_component,
        2.5e-1,
        1.0e-18,
    );
    assert_eq!(method0_negative_tail.requested_method, 0);
    assert_eq!(method0_negative_tail.method, 1);

    let method2_positive_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: 0.4,
        method: 2,
    })?;
    assert_close_with(method2_positive_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(
        method2_positive_tail.asymptotic_large_component,
        4.000_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_eq!(method2_positive_tail.requested_method, 2);
    assert_eq!(method2_positive_tail.method, 2);

    let negative_method = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: -0.75,
        method: -3,
    })?;
    assert_close_with(negative_method.previous_energy, 1.0, 1.0e-18);
    assert_close_with(negative_method.asymptotic_large_component, 7.5e-1, 1.0e-18);
    assert_eq!(negative_method.requested_method, -3);
    assert_eq!(negative_method.method, 1);

    let method1_zero_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: 0.0,
        method: 1,
    })?;
    assert_close_with(method1_zero_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(method1_zero_tail.asymptotic_large_component, 0.0, 1.0e-18);
    assert_eq!(method1_zero_tail.requested_method, 1);
    assert_eq!(method1_zero_tail.method, 1);
    Ok(())
}

#[test]
fn atom_dirac_normalization_matches_feff_soldir_norm_reference() -> Result<(), AtomMathError> {
    let fixture = sample_soldir_norm_fixture();

    let method_one = atomic_dirac_normalization(fixture.input(1, 6, 0.177, 0.82, 11, 5))?;
    assert_close_with(method_one.norm, 5.408_474_263_575_392e-6, 1.0e-18);

    let method_two = atomic_dirac_normalization(fixture.input(2, 8, 0.0, 1.35, 13, 7))?;
    assert_close_with(method_two.norm, 9.499_334_208_495_336e-6, 1.0e-18);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_solution_normalization_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let kept_fixture = sample_soldir_solution_normalization_fixture(false, false);
    let kept = atomic_dirac_solution_normalization(kept_fixture.input(6.25, 0.8, -0.4))?;
    assert_close_with(kept.component_divisor, 2.5, 1.0e-18);
    assert_close_with(kept.coefficient_divisor, 2.5, 1.0e-18);
    assert_close_with(kept.large_coefficients[0], 8.4e-2, 1.0e-18);
    assert_close_with(kept.small_coefficients[0], -4.28e-2, 1.0e-18);
    assert_close_with(kept.large_coefficients[3], 3.84e-1, 1.0e-18);
    assert_close_with(kept.small_coefficients[3], -1.568e-1, 1.0e-18);
    assert_close_with(kept.large_component[0], 1.64e-2, 1.0e-18);
    assert_close_with(kept.small_component[0], -1.18e-2, 1.0e-18);
    assert_close_with(kept.large_component[6], 1.316e-1, 1.0e-18);
    assert_close_with(kept.large_component[7], 0.0, 1.0e-18);
    assert_close_with(kept.small_component[8], 0.0, 1.0e-18);

    let flipped_fixture = sample_soldir_solution_normalization_fixture(true, true);
    let flipped = atomic_dirac_solution_normalization(flipped_fixture.input(1.44, 0.75, -0.25))?;
    assert_close_with(flipped.component_divisor, -1.2, 1.0e-18);
    assert_close_with(flipped.coefficient_divisor, -1.2, 1.0e-18);
    assert_close_with(
        flipped.large_coefficients[0],
        1.750_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_coefficients[0],
        8.916_666_666_666_667_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_coefficients[3],
        -8.000_000_000_000_000_4e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_coefficients[3],
        3.266_666_666_666_667_2e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_component[0],
        3.416_666_666_666_667_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_component[0],
        2.458_333_333_333_333_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_component[6],
        -2.741_666_666_666_666_7e-1,
        1.0e-18,
    );
    assert_close_with(flipped.large_component[7], 0.0, 1.0e-18);
    assert_close_with(flipped.small_component[8], 0.0, 1.0e-18);
    Ok(())
}

#[test]
fn atom_dirac_node_count_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = sample_soldir_node_count_component();

    let limited = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 4,
        scan_index_1based: 7,
    })?;
    assert_eq!(limited.scan_index_1based, 7);
    assert_eq!(limited.node_count, 4);

    let matching_extends = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 8,
        scan_index_1based: 3,
    })?;
    assert_eq!(matching_extends.scan_index_1based, 8);
    assert_eq!(matching_extends.node_count, 4);

    let full = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 1,
        scan_index_1based: 9,
    })?;
    assert_eq!(full.scan_index_1based, 9);
    assert_eq!(full.node_count, 5);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_node_energy_search_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let too_few_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 0,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_few_scale.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(too_few_scale.energy_sup, -5.0e-1, 1.0e-18);
    assert_close_with(too_few_scale.energy_inf, 1.0, 1.0e-18);
    assert_eq!(too_few_scale.search_attempt_count, 1);
    assert!(too_few_scale.needs_reintegration);
    assert!(!too_few_scale.attempts_exhausted);

    let too_few_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.6,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: -0.2,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 4,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_few_bisect.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(
        too_few_bisect.energy_sup,
        -5.999_999_999_999_999_8e-1,
        1.0e-18,
    );
    assert_close_with(
        too_few_bisect.energy_inf,
        -2.000_000_000_000_000_1e-1,
        1.0e-18,
    );
    assert_eq!(too_few_bisect.search_attempt_count, 5);
    assert!(too_few_bisect.needs_reintegration);
    assert!(!too_few_bisect.attempts_exhausted);

    let too_many_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 5,
        target_node_count: 3,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 7,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_many_scale.energy, -5.999_999_999_999_999_8e-1, 1.0e-18);
    assert_close_with(too_many_scale.energy_sup, -5.0, 1.0e-18);
    assert_close_with(too_many_scale.energy_inf, -5.0e-1, 1.0e-18);
    assert_eq!(too_many_scale.search_attempt_count, 8);
    assert!(too_many_scale.needs_reintegration);
    assert!(!too_many_scale.attempts_exhausted);

    let too_many_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.4,
        node_count: 5,
        target_node_count: 3,
        energy_sup: -0.7,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 2,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_many_bisect.energy, -5.500_000_000_000_000_4e-1, 1.0e-18);
    assert_close_with(
        too_many_bisect.energy_sup,
        -6.999_999_999_999_999_6e-1,
        1.0e-18,
    );
    assert_close_with(
        too_many_bisect.energy_inf,
        -4.000_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_eq!(too_many_bisect.search_attempt_count, 3);
    assert!(too_many_bisect.needs_reintegration);
    assert!(!too_many_bisect.attempts_exhausted);

    let matched = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.4,
        node_count: 3,
        target_node_count: 3,
        energy_sup: -0.7,
        energy_inf: -0.2,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 2,
        max_attempt_count: 50,
    })?;
    assert_close_with(matched.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(matched.energy_sup, -6.999_999_999_999_999_6e-1, 1.0e-18);
    assert_close_with(matched.energy_inf, -2.000_000_000_000_000_1e-1, 1.0e-18);
    assert_eq!(matched.search_attempt_count, 2);
    assert!(!matched.needs_reintegration);
    assert!(!matched.attempts_exhausted);

    let exhausted = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 1,
        max_attempt_count: 1,
    })?;
    assert_close_with(exhausted.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(exhausted.energy_sup, -5.0e-1, 1.0e-18);
    assert_close_with(exhausted.energy_inf, 1.0, 1.0e-18);
    assert_eq!(exhausted.search_attempt_count, 2);
    assert!(!exhausted.needs_reintegration);
    assert!(exhausted.attempts_exhausted);

    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -1.0e-8,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyTooSmall { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -5.0,
            node_count: 5,
            target_node_count: 3,
            energy_sup: -5.5,
            energy_inf: 1.0,
            energy_floor: -5.5,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyBelowPotentialFloor { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: -0.500_000_05,
            energy_floor: -5.0,
            energy_precision: 1.0e-6,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyBracketCollapsed { .. })
    ));
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_correction_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = Array1::from_vec(vec![0.12, -0.22, 0.31, 0.27, -0.18]);
    let small_component = Array1::from_vec(vec![-0.011, 0.024, 0.047, -0.018, 0.009]);

    let scaled =
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 2.6,
            large_component: large_component.view(),
            small_component: small_component.view(),
            matching_small_component: 0.052,
            matching_index_1based: 3,
        })?;
    assert_close_with(scaled.correction, 8.169_531_346_153_841_0e-2, 1.0e-16);
    assert_close_with(scaled.mismatch, 9.615_384_615_384_610_4e-2, 1.0e-16);

    let zero_matching =
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 1.9,
            large_component: large_component.view(),
            small_component: small_component.view(),
            matching_small_component: 0.0,
            matching_index_1based: 4,
        })?;
    assert_close_with(
        zero_matching.correction,
        3.505_269_884_210_525_7e-1,
        1.0e-16,
    );
    assert_close_with(zero_matching.mismatch, 1.8e-2, 1.0e-18);

    let accepted = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -0.5,
        correction: -0.02,
        mismatch: 0.001,
        energy_sup: -0.8,
        energy_inf: -0.2,
        mismatch_precision: 0.01,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(accepted.energy, -5.2e-1, 1.0e-18);
    assert_close_with(accepted.correction, -2.0e-2, 1.0e-18);
    assert_close_with(accepted.relative_step, 4.0e-2, 1.0e-18);
    assert!(!accepted.needs_rematch);

    let positive_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -0.05,
        correction: 0.08,
        mismatch: 0.001,
        energy_sup: -0.8,
        energy_inf: -0.02,
        mismatch_precision: 0.01,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(positive_halved.energy, -4.0e-2, 1.0e-18);
    assert_close_with(positive_halved.correction, 1.0e-2, 1.0e-18);
    assert_close_with(
        positive_halved.relative_step,
        1.999_999_999_999_999_8e-1,
        1.0e-18,
    );
    assert!(!positive_halved.needs_rematch);

    let bound_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -1.0,
        correction: 0.30,
        mismatch: 0.4,
        energy_sup: -1.2,
        energy_inf: -0.8,
        mismatch_precision: 0.1,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(bound_halved.energy, -8.5e-1, 1.0e-18);
    assert_close_with(bound_halved.correction, 1.5e-1, 1.0e-18);
    assert_close_with(bound_halved.relative_step, 1.5e-1, 1.0e-18);
    assert!(bound_halved.needs_rematch);

    let too_small = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -1.0,
        correction: 1.0e-9,
        mismatch: 1.0,
        energy_sup: -0.5,
        energy_inf: -0.6,
        mismatch_precision: 0.1,
        zero_energy_precision: 1.0e-7,
    });
    let Err(AtomMathError::DiracEnergyCorrectionTooSmall { relative_step }) = too_small else {
        return Err(AtomMathError::NonFiniteScalar {
            field: "soldir_energy_too_small_reference",
            value: 0.0,
        });
    };
    assert_close_with(relative_step, 5.0e-10, 1.0e-24);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_iteration_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method1 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 1,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method1.mismatch_precision,
        1.000_000_000_000_000_1e-5,
        1.0e-20,
    );
    assert_close_with(method1.energy_inf, 1.0, 1.0e-18);
    assert_close_with(method1.energy_sup, -0.75, 1.0e-18);
    assert_close_with(method1.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_eq!(method1.match_attempt_count, 0);
    assert_eq!(method1.node_count, 0);
    assert_eq!(method1.search_attempt_count, 0);

    let method2 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 2,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method2.mismatch_precision,
        2.000_000_000_000_000_2e-5,
        1.0e-20,
    );
    assert_close_with(method2.energy_inf, 1.0, 1.0e-18);
    assert_close_with(method2.energy_sup, -0.75, 1.0e-18);
    assert_close_with(method2.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);

    let method0 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 0,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method0.mismatch_precision,
        1.000_000_000_000_000_1e-5,
        1.0e-20,
    );

    let homogeneous = atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
        requested_method: 0,
        method: 1,
    })?;
    assert_eq!(homogeneous.method, 1);
    assert!(!homogeneous.needs_restart);

    let method1_retry =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 1,
            method: 1,
        })?;
    assert_eq!(method1_retry.method, 2);
    assert!(method1_retry.needs_restart);

    let method2_stop = atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
        requested_method: 1,
        method: 2,
    })?;
    assert_eq!(method2_stop.method, 2);
    assert!(!method2_stop.needs_restart);

    let negative_retry =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: -1,
            method: 1,
        })?;
    assert_eq!(negative_retry.method, 2);
    assert!(negative_retry.needs_restart);

    let requested2_current1 =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 2,
            method: 1,
        })?;
    assert_eq!(requested2_current1.method, 2);
    assert!(requested2_current1.needs_restart);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_loop_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let far_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: 1.0,
    })?;
    assert_eq!(
        far_energy.integration_mode,
        AtomicDiracIntegrationMode::SearchMatchingPoint
    );
    assert!(!far_energy.relocated);
    assert_close_with(far_energy.reference_energy, -5.0e-1, 1.0e-18);
    assert_close_with(far_energy.relative_energy_change, 3.0, 1.0e-18);

    let near_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: -0.54,
    })?;
    assert_eq!(
        near_energy.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert!(!near_energy.relocated);
    assert_close_with(near_energy.reference_energy, -5.0e-1, 1.0e-18);
    assert_close_with(
        near_energy.relative_energy_change,
        8.000_000_000_000_007_1e-2,
        1.0e-17,
    );

    let far_negative = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: -0.42,
    })?;
    assert_eq!(
        far_negative.integration_mode,
        AtomicDiracIntegrationMode::SearchMatchingPoint
    );
    assert_close_with(
        far_negative.relative_energy_change,
        1.600_000_000_000_000_3e-1,
        1.0e-17,
    );

    let below_test = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.005,
        mismatch_precision: 0.01,
        match_attempt_count: 3,
        max_attempt_count: 5,
    })?;
    assert_eq!(below_test.match_attempt_count, 3);
    assert!(!below_test.needs_rematch);
    assert!(!below_test.attempts_exhausted);

    let retry_left = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.02,
        mismatch_precision: 0.01,
        match_attempt_count: 4,
        max_attempt_count: 5,
    })?;
    assert_eq!(retry_left.match_attempt_count, 5);
    assert!(retry_left.needs_rematch);
    assert!(!retry_left.attempts_exhausted);

    let exhausted = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.02,
        mismatch_precision: 0.01,
        match_attempt_count: 5,
        max_attempt_count: 5,
    })?;
    assert_eq!(exhausted.match_attempt_count, 6);
    assert!(!exhausted.needs_rematch);
    assert!(exhausted.attempts_exhausted);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_integration_seeds_match_feff_soldir_reference() -> Result<(), AtomMathError> {
    let radial_count = 8;
    let coefficient_count = 5;
    let large_source = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.05 * index + 0.003 * index * index
    });
    let small_source = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.04 * index + 0.002 * index * index
    });
    let large_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.11 * index - 0.004 * index * index
    });
    let small_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.09 * index + 0.005 * index * index
    });

    let inhomogeneous = atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
        large_source: large_source.view(),
        small_source: small_source.view(),
        large_source_coefficients: large_source_coefficients.view(),
        small_source_coefficients: small_source_coefficients.view(),
        coefficient_count,
    })?;
    assert_close_with(
        inhomogeneous.large_source[0],
        5.300_000_000_000_000_5e-2,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.large_source[7],
        5.920_000_000_000_000_8e-1,
        1.0e-17,
    );
    assert_close_with(
        inhomogeneous.small_source[4],
        -1.500_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_close_with(inhomogeneous.large_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        inhomogeneous.large_coefficients[1],
        1.060_000_000_000_000_0e-1,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.large_coefficients[4],
        3.760_000_000_000_000_0e-1,
        1.0e-18,
    );
    assert_close_with(inhomogeneous.small_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        inhomogeneous.small_coefficients[1],
        -8.499_999_999_999_999_2e-2,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.small_coefficients[4],
        -2.799_999_999_999_999_7e-1,
        1.0e-18,
    );

    let homogeneous = atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
        radial_len: radial_count,
        coefficient_len: coefficient_count,
    })?;
    assert_eq!(homogeneous.large_source.len(), radial_count);
    assert_eq!(homogeneous.small_source.len(), radial_count);
    assert_eq!(homogeneous.large_coefficients.len(), coefficient_count);
    assert_eq!(homogeneous.small_coefficients.len(), coefficient_count);
    assert!(homogeneous.large_source.iter().all(|&value| value == 0.0));
    assert!(homogeneous.small_source.iter().all(|&value| value == 0.0));
    assert!(
        homogeneous
            .large_coefficients
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        homogeneous
            .small_coefficients
            .iter()
            .all(|&value| value == 0.0)
    );
    Ok(())
}

#[test]
fn atom_dirac_inhomogeneous_branch_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let homogeneous_request =
        atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
            requested_method: 0,
        })?;
    assert_eq!(
        homogeneous_request.action,
        AtomicDiracInhomogeneousBranchAction::MatchHomogeneousTail
    );

    let method1_request = atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
        requested_method: 1,
    })?;
    assert_eq!(
        method1_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );

    let method2_request = atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
        requested_method: 2,
    })?;
    assert_eq!(
        method2_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );

    let negative_request =
        atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
            requested_method: -1,
        })?;
    assert_eq!(
        negative_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );
    Ok(())
}

#[test]
fn atom_dirac_homogeneous_pass_setup_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method1 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 1 })?;
    assert_eq!(
        method1.integration_mode,
        AtomicDiracIntegrationMode::InwardOnly
    );
    assert_eq!(method1.raw_integration_flag, -1);

    let method2 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 2 })?;
    assert_eq!(
        method2.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(method2.raw_integration_flag, 1);

    let method3 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 3 })?;
    assert_eq!(
        method3.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(method3.raw_integration_flag, 1);

    let negative =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: -2 })?;
    assert_eq!(
        negative.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(negative.raw_integration_flag, 1);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_matching_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        0.08 * index - 0.006 * index * index
    });
    let small_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        -0.025 * index + 0.0015 * index * index
    });
    let homogeneous_large_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        0.018 * index + 0.0007 * index * index
    });
    let homogeneous_small_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        -0.012 * index + 0.0004 * index * index
    });

    let homogeneous_match = atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        matching_large_component: 0.240,
        active_len: 7,
        matching_index_1based: 4,
    })?;
    assert_close_with(
        homogeneous_match.tail_scale,
        1.071_428_571_428_571_4,
        1.0e-16,
    );
    assert_eq!(homogeneous_match.scan_index_1based, 4);
    assert_close_with(homogeneous_match.large_component[0], 7.4e-2, 1.0e-18);
    assert_close_with(
        homogeneous_match.large_component[3],
        2.399_999_999_999_999_9e-1,
        1.0e-18,
    );
    assert_close_with(
        homogeneous_match.large_component[6],
        2.850_000_000_000_000_3e-1,
        1.0e-16,
    );
    assert_close_with(
        homogeneous_match.large_component[7],
        large_component[7],
        1.0e-18,
    );
    assert_close_with(
        homogeneous_match.small_component[3],
        -8.142_857_142_857_143_3e-2,
        1.0e-17,
    );
    assert_close_with(
        homogeneous_match.small_component[6],
        -1.087_500_000_000_000_0e-1,
        1.0e-17,
    );

    let large_match = atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        homogeneous_large_component: homogeneous_large_component.view(),
        homogeneous_small_component: homogeneous_small_component.view(),
        matching_large_component: 0.240,
        active_len: 7,
        matching_index_1based: 4,
    })?;
    assert_close_with(large_match.tail_scale, 1.923_076_923_076_921_5e-1, 1.0e-16);
    assert_close_with(large_match.large_mismatch, -1.6e-2, 1.0e-16);
    assert_close_with(large_match.large_component[3], 2.4e-1, 1.0e-18);
    assert_close_with(
        large_match.large_component[6],
        2.968_269_230_769_230_4e-1,
        1.0e-16,
    );
    assert_close_with(
        large_match.small_component[6],
        -1.138_846_153_846_153_9e-1,
        1.0e-16,
    );
    assert_close_with(large_match.large_component[7], large_component[7], 1.0e-18);

    let large_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        0.11 * index - 0.004 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        -0.07 * index + 0.003 * index * index
    });
    let homogeneous_large_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        0.012 * index + 0.0005 * index * index
    });
    let homogeneous_small_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        -0.009 * index + 0.0003 * index * index
    });

    let two_match = atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        large_coefficients: large_coefficients.view(),
        small_coefficients: small_coefficients.view(),
        homogeneous_large_component: homogeneous_large_component.view(),
        homogeneous_small_component: homogeneous_small_component.view(),
        homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
        homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
        matching_large_component: 0.285,
        matching_small_component: -0.068,
        homogeneous_matching_large_component: 0.087,
        homogeneous_matching_small_component: -0.047,
        coefficient_count: 4,
        active_len: 8,
        matching_index_1based: 5,
    })?;
    assert_close_with(two_match.determinant, -7.025e-4, 1.0e-18);
    assert_close_with(two_match.tail_scale, 4.756_583_629_893_235_4, 1.0e-15);
    assert_close_with(two_match.prefix_scale, 5.475_088_967_971_526_4, 1.0e-15);
    assert_close_with(two_match.large_mismatch, -3.5e-2, 1.0e-16);
    assert_close_with(two_match.small_mismatch, -1.95e-2, 1.0e-16);
    assert_close_with(
        two_match.large_component[0],
        1.763_841_637_010_675_2e-1,
        1.0e-16,
    );
    assert_close_with(
        two_match.large_component[4],
        7.613_327_402_135_228_2e-1,
        1.0e-15,
    );
    assert_close_with(
        two_match.small_component[4],
        -3.253_291_814_946_617_2e-1,
        1.0e-15,
    );
    assert_close_with(
        two_match.large_coefficients[3],
        6.826_049_822_064_055_3e-1,
        1.0e-15,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_match_matches_feff_soldir_reference() -> Result<(), AtomMathError>
{
    let radial_count = 8;
    let coefficient_count = 5;
    let large_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.004 * index + 0.0005 * index * index
    });
    let small_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.003 * index + 0.0002 * index * index
    });
    let homogeneous_large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.018 * index + 0.0007 * index * index
    });
    let homogeneous_small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.012 * index + 0.0004 * index * index
    });
    let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.0008 * index + 0.00007 * index * index
    });
    let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.0006 * index + 0.00005 * index * index
    });
    let homogeneous_large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.012 * index + 0.0005 * index * index
    });
    let homogeneous_small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.009 * index + 0.0003 * index * index
    });

    let matched =
        atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
            large_derivative: large_derivative.view(),
            small_derivative: small_derivative.view(),
            large_derivative_coefficients: large_derivative_coefficients.view(),
            small_derivative_coefficients: small_derivative_coefficients.view(),
            homogeneous_large_component: homogeneous_large_component.view(),
            homogeneous_small_component: homogeneous_small_component.view(),
            homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
            homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
            matching_large_derivative: 0.037,
            matching_small_derivative: -0.011,
            homogeneous_matching_large_component: 0.087,
            homogeneous_matching_small_component: -0.047,
            coefficient_count,
            active_len: radial_count,
            matching_index_1based: 5,
        })?;

    assert_close_with(matched.determinant, -7.025e-4, 1.0e-18);
    assert_close_with(matched.prefix_scale, 1.672_597_864_768_679_6e-1, 1.0e-16);
    assert_close_with(matched.tail_scale, 1.772_241_992_882_559_2e-1, 1.0e-16);
    assert_close_with(matched.large_mismatch, -4.499_999_999_999_997_1e-3, 1.0e-18);
    assert_close_with(matched.small_mismatch, 1.000_000_000_000_000_9e-3, 1.0e-18);
    assert_close_with(
        matched.large_derivative[0],
        7.627_758_007_117_430_9e-3,
        1.0e-18,
    );
    assert_close_with(
        matched.large_derivative[4],
        5.155_160_142_348_751_1e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.small_derivative[4],
        -1.886_120_996_441_279_3e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.large_derivative_coefficients[4],
        1.787_633_451_957_292_7e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.small_derivative_coefficients[4],
        -8.022_241_992_882_548_1e-3,
        1.0e-18,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_source_matches_feff_soldir_reference() -> Result<(), AtomMathError>
{
    let radial_count = 8;
    let coefficient_count = 5;
    let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
    let large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.06 * index - 0.002 * index * index
    });
    let small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.015 * index + 0.0007 * index * index
    });
    let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.12 * index - 0.004 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.08 * index + 0.003 * index * index
    });

    let source =
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: large_component.view(),
            small_component: small_component.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            radii: radii.view(),
            speed_of_light: 137.0373,
            coefficient_count,
            active_len: 7,
        })?;

    assert_close_with(source.large_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        source.large_coefficients[1],
        8.464_848_621_506_699_7e-4,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[2],
        1.634_591_457_946_121_3e-3,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[3],
        2.364_319_787_386_353_7e-3,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[4],
        3.035_669_850_471_368_2e-3,
        1.0e-18,
    );
    assert_close_with(source.small_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        source.small_coefficients[1],
        -5.618_908_136_689_792_0e-4,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[2],
        -1.079_997_927_571_544_4e-3,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[3],
        -1.554_321_341_707_695_8e-3,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[4],
        -1.984_861_056_077_433_2e-3,
        1.0e-18,
    );
    assert_close_with(source.large_source[0], 3.385_939_448_602_680_0e-5, 1.0e-19);
    assert_close_with(source.large_source[3], 1.689_008_004_217_633_1e-4, 1.0e-18);
    assert_close_with(source.large_source[6], 3.636_984_276_120_176_0e-4, 1.0e-18);
    assert_close_with(source.large_source[7], 0.0, 1.0e-18);
    assert_close_with(source.small_source[0], -8.348_092_088_796_263_3e-6, 1.0e-20);
    assert_close_with(source.small_source[3], -3.962_672_625_279_831_4e-5, 1.0e-19);
    assert_close_with(source.small_source[6], -7.985_552_432_350_822_2e-5, 1.0e-19);
    assert_close_with(source.small_source[7], 0.0, 1.0e-18);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_correction_matches_feff_soldir_reference()
-> Result<(), AtomMathError> {
    let radial_count = 8;
    let coefficient_count = 5;
    let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
    let large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.08 * index - 0.003 * index * index
    });
    let small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.018 * index + 0.0008 * index * index
    });
    let large_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.002 * index + 0.0003 * index * index
    });
    let small_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.0014 * index + 0.0001 * index * index
    });
    let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.13 * index - 0.005 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.09 * index + 0.0035 * index * index
    });
    let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.0007 * index + 0.00004 * index * index
    });
    let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.0005 * index + 0.00003 * index * index
    });

    let correction = atomic_dirac_energy_disagreement_correction(
        AtomicDiracEnergyDisagreementCorrectionInput {
            radii: radii.view(),
            large_component: large_component.view(),
            small_component: small_component.view(),
            large_derivative: large_derivative.view(),
            small_derivative: small_derivative.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            large_derivative_coefficients: large_derivative_coefficients.view(),
            small_derivative_coefficients: small_derivative_coefficients.view(),
            norm: 0.913,
            step: 0.11,
            origin_power: 1.30,
            coefficient_count,
            active_len: 7,
        },
    )?;

    assert_close_with(
        correction.overlap_integral,
        3.960_742_076_990_347_3e-4,
        1.0e-18,
    );
    assert_close_with(correction.correction, 1.098_279_038_483_979_4e2, 1.0e-12);
    assert_close_with(
        correction.normalization_mismatch,
        8.699_999_999_999_996_6e-2,
        1.0e-18,
    );
    assert_close_with(
        correction.large_component[0],
        3.296_041_788_513_152_7e-1,
        1.0e-16,
    );
    assert_close_with(
        correction.large_component[3],
        1.677_797_169_259_493_5,
        1.0e-15,
    );
    assert_close_with(
        correction.large_component[6],
        3.565_060_840_449_020_5,
        1.0e-15,
    );
    assert_close_with(correction.large_component[7], 4.48e-1, 1.0e-18);
    assert_close_with(
        correction.small_component[0],
        -1.599_762_750_029_173_1e-1,
        1.0e-16,
    );
    assert_close_with(
        correction.small_component[6],
        -6.249_567_288_571_498_1e-1,
        1.0e-16,
    );
    assert_close_with(correction.small_component[7], -9.28e-2, 1.0e-18);
    assert_close_with(
        correction.large_coefficients[4],
        1.019_225_567_317_790_8,
        1.0e-15,
    );
    assert_close_with(
        correction.small_coefficients[4],
        -5.546_988_317_346_963_6e-1,
        1.0e-16,
    );
    Ok(())
}

#[test]
fn atom_dirac_matching_point_update_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let mut large_component = Array1::<Real>::zeros(25);
    large_component[2] = 0.60;
    large_component[4] = 0.40;
    let no_update = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 13,
        matching_index_1based: 5,
        already_relocated: false,
    })?;
    assert_eq!(no_update.matching_index_1based, 5);
    assert_eq!(no_update.peak_index_1based, 3);
    assert_eq!(no_update.scan_index_1based, 5);
    assert!(!no_update.relocated);
    assert!(!no_update.needs_reintegration);

    large_component.fill(0.0);
    large_component[5] = 0.90;
    let reintegrate_even =
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: large_component.view(),
            active_len: 21,
            matching_index_1based: 3,
            already_relocated: false,
        })?;
    assert_eq!(reintegrate_even.matching_index_1based, 7);
    assert_eq!(reintegrate_even.peak_index_1based, 6);
    assert_eq!(reintegrate_even.scan_index_1based, 7);
    assert!(reintegrate_even.relocated);
    assert!(reintegrate_even.needs_reintegration);

    large_component.fill(0.0);
    large_component[17] = 0.90;
    let fallback_tail = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 21,
        matching_index_1based: 5,
        already_relocated: false,
    })?;
    assert_eq!(fallback_tail.matching_index_1based, 9);
    assert_eq!(fallback_tail.peak_index_1based, 18);
    assert_eq!(fallback_tail.scan_index_1based, 9);
    assert!(fallback_tail.relocated);
    assert!(!fallback_tail.needs_reintegration);

    let already_moved = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 21,
        matching_index_1based: 5,
        already_relocated: true,
    })?;
    assert_eq!(already_moved.matching_index_1based, 5);
    assert_eq!(already_moved.peak_index_1based, 18);
    assert_eq!(already_moved.scan_index_1based, 18);
    assert!(already_moved.relocated);
    assert!(!already_moved.needs_reintegration);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_solver_setup_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let fixture = sample_soldir_setup_fixture();

    let clamped = atomic_dirac_solver_setup(fixture.input(-8.0, 0, -2, 2, true))?;
    assert_eq!(clamped.requested_method, 0);
    assert_eq!(clamped.method, 1);
    assert_eq!(clamped.target_nodes, 1);
    assert_close_with(clamped.energy, -5.963_839_259_330_666_4, 1.0e-14);
    assert_close_with(clamped.energy_floor, -6.626_488_065_922_962_4, 1.0e-14);
    assert_close_with(
        clamped.initial_small_coefficient,
        -1.472_928_410_311_296_5e-2,
        1.0e-16,
    );
    assert_close_with(clamped.angular_term, 7.297_283_294_402_327_9e-3, 1.0e-18);
    assert_close_with(clamped.doubled_speed_of_light, 274.0746, 1.0e-12);

    let positive_kappa = atomic_dirac_solver_setup(fixture.input(-0.2, 2, 1, 3, true))?;
    assert_eq!(positive_kappa.requested_method, 2);
    assert_eq!(positive_kappa.method, 2);
    assert_eq!(positive_kappa.target_nodes, 2);
    assert_close_with(positive_kappa.energy, -0.2, 1.0e-18);
    assert_close_with(
        positive_kappa.energy_floor,
        -6.626_488_065_922_962_4,
        1.0e-14,
    );
    assert_close_with(
        positive_kappa.initial_small_coefficient,
        3.160_423_066_381_816_8e1,
        1.0e-13,
    );
    assert_close_with(
        positive_kappa.angular_term,
        7.297_283_294_402_327_9e-3,
        1.0e-18,
    );

    let no_adjust = atomic_dirac_solver_setup(fixture.input(-0.1, -1, -1, 1, false))?;
    assert_eq!(no_adjust.requested_method, -1);
    assert_eq!(no_adjust.method, 1);
    assert_eq!(no_adjust.target_nodes, 1);
    assert_close_with(no_adjust.energy, -0.1, 1.0e-18);
    assert_close_with(no_adjust.energy_floor, -5.619_077_423_139_916_0e1, 1.0e-13);
    assert_close_with(no_adjust.initial_small_coefficient, -6.0e-3, 1.0e-18);
    assert_close_with(no_adjust.angular_term, 0.0, 1.0e-18);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_integration_matches_feff_intdir_reference() -> Result<(), AtomMathError> {
    let fixture = sample_intdir_fixture();

    let searched = atomic_dirac_integration(fixture.input(
        AtomicDiracIntegrationMode::SearchMatchingPoint,
        0,
        0,
    ))?;
    assert_eq!(searched.matching_index_1based, 127);
    assert_eq!(searched.max_index_1based, 151);
    assert_some_close(
        searched.matching_large_component,
        7.844_180_279_031_651_7e-1,
        1.0e-12,
    );
    assert_some_close(
        searched.matching_small_component,
        6.433_852_518_326_962_0e-4,
        1.0e-15,
    );
    assert_close_with(
        searched.large_component[126],
        3.946_584_591_497_206_1e2,
        1.0e-9,
    );
    assert_close_with(
        searched.small_component[126],
        -5.380_100_169_329_787_9e-1,
        1.0e-12,
    );
    assert_close_with(
        searched.large_coefficients[1],
        -1.096_438_489_149_803_4,
        1.0e-12,
    );
    assert_close_with(
        searched.small_coefficients[1],
        2.146_028_457_009_671_9e-2,
        1.0e-14,
    );
    assert_close_with(
        searched.large_component[150],
        7.844_180_279_031_651_3e-8,
        1.0e-20,
    );
    assert_close_with(
        searched.small_component[150],
        -1.144_825_333_416_651_0e-10,
        1.0e-22,
    );

    let fixed = atomic_dirac_integration(fixture.input(
        AtomicDiracIntegrationMode::FixedMatchingPoint,
        65,
        139,
    ))?;
    assert_eq!(fixed.matching_index_1based, 65);
    assert_eq!(fixed.max_index_1based, 139);
    assert_some_close(
        fixed.matching_large_component,
        -4.787_017_896_869_409_0e-2,
        1.0e-13,
    );
    assert_some_close(
        fixed.matching_small_component,
        2.893_471_976_931_037_7e-3,
        1.0e-15,
    );
    assert_close_with(fixed.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
    assert_close_with(
        fixed.small_component[64],
        1.444_514_204_264_709_7e-2,
        1.0e-15,
    );
    assert_close_with(
        fixed.large_coefficients[1],
        -1.096_438_489_149_803_4,
        1.0e-12,
    );
    assert_close_with(
        fixed.small_coefficients[1],
        2.146_028_457_009_671_9e-2,
        1.0e-14,
    );
    assert_close_with(fixed.large_component[138], 2.0e-2, 1.0e-20);
    assert_close_with(
        fixed.small_component[138],
        -2.918_916_426_428_632_8e-5,
        1.0e-22,
    );

    let inward =
        atomic_dirac_integration(fixture.input(AtomicDiracIntegrationMode::InwardOnly, 65, 139))?;
    assert_eq!(inward.matching_large_component, None);
    assert_eq!(inward.matching_small_component, None);
    assert_eq!(inward.matching_index_1based, 65);
    assert_eq!(inward.max_index_1based, 139);
    assert_close_with(inward.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
    assert_close_with(
        inward.small_component[64],
        1.444_514_204_264_709_7e-2,
        1.0e-15,
    );
    assert_close_with(inward.large_coefficients[1], 4.0e-4, 1.0e-18);
    assert_close_with(inward.small_coefficients[1], -3.0e-4, 1.0e-18);
    assert_close_with(inward.large_component[138], 2.0e-2, 1.0e-20);
    assert_close_with(
        inward.small_component[138],
        -2.918_916_426_428_632_8e-5,
        1.0e-22,
    );
    Ok(())
}

#[test]
fn atom_total_energy_matches_feff_etotal_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, 1, -2, 2];
    let occupations = [2.0, 1.5, 3.0, 0.5];
    let valence_occupations = [0.0, 0.0, 1.0, 0.0];
    let orbital_energies = [-0.7, -0.3, -0.12, -0.05];
    let coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
        0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as Real
    });

    let energy = atomic_total_energy(
        AtomicTotalEnergyInput {
            kappas: &kappas,
            occupations: &occupations,
            valence_occupations: &valence_occupations,
            orbital_energies: &orbital_energies,
            coulomb_coefficients: coefficients.view(),
        },
        |request| {
            Ok(0.0001 * (request.rank + 1) as Real
                + 0.001 * request.first_left as Real
                + 0.0002 * request.first_right as Real
                + 0.00003 * request.second_left as Real
                + 0.000004 * request.second_right as Real)
        },
    )?;

    assert_close(energy.total, -2.230_065_144_829_932);
    assert_close_with(energy.direct_coulomb, 0.109_629, 1.0e-6);
    assert_close_with(energy.exchange_coulomb, -0.055_702_8, 1.0e-6);
    assert_close_with(energy.magnetic_breit, 0.075_902_3, 1.0e-6);
    assert_close_with(energy.retarded_breit, -0.017_041_4, 1.0e-6);
    Ok(())
}

#[test]
fn atom_lagrange_parameters_match_feff_lagdat_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, -1, 1, 1, -2];
    let occupations = [2.0, 1.0, 1.5, 0.5, 3.0];
    let valence_occupations = [0.0, 0.0, 0.25, 0.0, 0.0];
    let shell_markers = [-1, 1, 1, 1, -1];
    let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let all_parameters = atomic_lagrange_parameters(
        AtomicLagrangeParametersInput {
            active_orbital_1based: None,
            include_exchange: true,
            kappas: &kappas,
            occupations: &occupations,
            shell_markers: &shell_markers,
            coulomb_coefficients: coefficients.view(),
        },
        sample_atomic_radial_integral,
    )?;
    let expected_all = [
        -1.780_000_000_000_000_1e-3,
        0.0,
        0.0,
        0.0,
        0.0,
        -6.871_000_000_000_001e-3,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (&actual, expected) in all_parameters.iter().zip(expected_all) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    let active_parameters = atomic_lagrange_parameters(
        AtomicLagrangeParametersInput {
            active_orbital_1based: Some(2),
            include_exchange: false,
            kappas: &kappas,
            occupations: &occupations,
            shell_markers: &shell_markers,
            coulomb_coefficients: coefficients.view(),
        },
        sample_atomic_radial_integral,
    )?;
    let expected_active = [
        -1.200_000_000_000_000_1e-3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (&actual, expected) in active_parameters.iter().zip(expected_active) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    Ok(())
}

#[test]
fn atom_tabulation_matches_feff_tabrat_reference() -> Result<(), AtomMathError> {
    let principal_quantum_numbers = [1, 2, 2, 3, 3];
    let kappas = [-1, -1, 1, -2, 1];
    let occupations = [2.0, 1.5, 0.5, 3.0, 0.25];
    let orbital_energies = [-0.70, -0.25, -0.18, -0.09, -0.04];
    let tabulation = atomic_tabulation(
        AtomicTabulationInput {
            principal_quantum_numbers: &principal_quantum_numbers,
            kappas: &kappas,
            occupations: &occupations,
            orbital_energies: &orbital_energies,
        },
        sample_atomic_tabrat_integral,
    )?;

    let expected = [
        (
            1,
            "s",
            2.0,
            19.047_977_2,
            [0.136, 0.134, 0.132, 0.131, 0.129, 0.128, 0.0],
            6,
        ),
        (
            2,
            "s",
            1.5,
            6.802_849,
            [0.166, 0.164, 0.162, 0.161, 0.159, 0.158, 0.0],
            6,
        ),
        (
            2,
            "p*",
            0.5,
            4.898_051_28,
            [0.196, 0.194, 0.192, 0.191, 0.189, 0.188, 0.0],
            6,
        ),
        (
            3,
            "p",
            3.0,
            2.449_025_64,
            [0.226, 0.224, 0.222, 0.221, 0.219, 0.218, 0.217],
            7,
        ),
        (
            3,
            "p*",
            0.25,
            1.088_455_84,
            [0.256, 0.254, 0.252, 0.251, 0.249, 0.248, 0.0],
            6,
        ),
    ];
    for (orbital, (nq, label, occupation, binding_energy_ev, moments, moment_count)) in
        tabulation.orbitals.iter().zip(expected)
    {
        assert_eq!(orbital.principal_quantum_number, nq);
        assert_eq!(orbital.orbital_label, label);
        assert_close(orbital.occupation, occupation);
        assert_close_with(orbital.binding_energy_ev, binding_energy_ev, 1.0e-10);
        assert_eq!(orbital.moments.len(), moment_count);
        for ((moment, &expected_value), &expected_power) in orbital
            .moments
            .iter()
            .zip(moments.iter())
            .zip(ATOM_TABRAT_MOMENT_POWERS.iter())
        {
            assert_eq!(moment.power, expected_power);
            assert_close(moment.value, expected_value);
        }
    }
    assert_eq!(tabulation.overlaps.len(), 2);
    assert_eq!(tabulation.overlaps[0].left, 0);
    assert_eq!(tabulation.overlaps[0].right, 1);
    assert_eq!(tabulation.overlaps[0].left_orbital_label, "s");
    assert_eq!(tabulation.overlaps[0].right_orbital_label, "s");
    assert_close(tabulation.overlaps[0].value, 0.15);
    assert_eq!(tabulation.overlaps[1].left, 2);
    assert_eq!(tabulation.overlaps[1].right, 4);
    assert_eq!(tabulation.overlaps[1].left_orbital_label, "p*");
    assert_eq!(tabulation.overlaps[1].right_orbital_label, "p*");
    assert_close(tabulation.overlaps[1].value, 0.23);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_differential_integral_matches_feff_dsordf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_dsordf_fixture();
    let cases = [
        (
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 1,
                right_orbital_1based: 2,
                multiply_by_derivative: false,
            },
            2,
            0.0,
            4.983_995_991_889_760_16e-9,
        ),
        (
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 1,
                right_orbital_1based: 3,
                multiply_by_derivative: true,
            },
            -1,
            0.4,
            4.174_834_158_519_188_87e-5,
        ),
        (
            AtomicDifferentialIntegralKind::LargeSmallOverlap {
                left_orbital_1based: 2,
                right_orbital_1based: 3,
                multiply_by_derivative: false,
            },
            1,
            0.0,
            -5.798_475_020_316_198_31e-8,
        ),
        (
            AtomicDifferentialIntegralKind::LargeSmallOverlap {
                left_orbital_1based: 2,
                right_orbital_1based: 1,
                multiply_by_derivative: true,
            },
            0,
            0.3,
            -4.232_100_062_570_746_56e-8,
        ),
        (
            AtomicDifferentialIntegralKind::DerivativeProjection {
                large_orbital_1based: 2,
                small_orbital_1based: 3,
            },
            0,
            0.45,
            1.816_237_327_192_537_93e-5,
        ),
        (
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
            0,
            0.45,
            5.411_954_636_180_096_36e-5,
        ),
    ];

    for (kind, power, origin_power, expected) in cases {
        let actual = atomic_differential_integral(fixture.input(kind, power, origin_power))?;
        assert_close_with(actual, expected, 1.0e-17);
    }
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_local_density_potential_matches_feff_vlda_reference() -> Result<(), AtomMathError> {
    let fixture = sample_vlda_fixture();

    let valence = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::ValenceDensity, true),
    )?;
    assert_close_with(
        valence.total_density[0],
        6.809_505_899_999_999_42e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.total_density[4],
        8.670_367_500_000_001_48e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.total_density[9],
        4.974_400_000_000_001_22e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[0],
        2.049_973_999_999_999_98e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[4],
        2.672_390_000_000_000_62e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[9],
        1.243_600_000_000_000_30e-3,
        1.0e-17,
    );
    assert_close_with(valence.potential[0], -7.054_707_605_385_910_14e-3, 2.0e-10);
    assert_close_with(valence.potential[4], -6.362_810_615_972_094_51e-3, 2.0e-10);
    assert_close_with(valence.potential[9], -3.663_730_681_720_983_07e-3, 2.0e-10);
    assert_close_with(valence.potential[12], 1.300_000_000_000_000_16e-3, 1.0e-18);
    assert_close_with(
        valence.development_coefficients[1],
        1.284_529_239_461_408_91e-2,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[0],
        -4.676_397_112_407_516_99e-3,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[4],
        1.845_934_601_355_665_38e-3,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[9],
        1.682_086_596_903_880_49e-2,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[12],
        2.600_000_000_000_000_23e-2,
        1.0e-18,
    );

    let core = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::CoreDensitySeparated, true),
    )?;
    assert_close_with(core.potential[0], -4.639_483_986_312_321_55e-3, 2.0e-10);
    assert_close_with(core.potential[4], -4.094_974_008_849_363_44e-3, 2.0e-10);
    assert_close_with(core.potential[9], -1.989_064_683_335_639_83e-3, 2.0e-10);
    assert_close_with(
        core.development_coefficients[1],
        1.526_051_601_368_767_77e-2,
        2.0e-10,
    );
    assert_close_with(
        core.energy_density[0],
        -2.422_637_366_298_145_52e-3,
        2.0e-10,
    );
    assert_close_with(core.energy_density[4], 4.540_470_272_335_868_71e-3, 2.0e-10);
    assert_close_with(core.energy_density[9], 1.796_243_867_752_029_75e-2, 2.0e-10);

    let total = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::TotalDensity, false),
    )?;
    assert_close_with(total.potential[0], -1.030_418_779_316_292_88e-2, 2.0e-10);
    assert_close_with(total.potential[4], -9.399_113_926_789_406_24e-3, 2.0e-10);
    assert_close_with(total.potential[9], -6.124_858_580_930_082_20e-3, 2.0e-10);
    assert_close_with(
        total.energy_density[0],
        2.000_000_000_000_000_04e-3,
        1.0e-18,
    );
    assert_close_with(
        total.energy_density[4],
        1.000_000_000_000_000_02e-2,
        1.0e-18,
    );
    assert_close_with(
        total.energy_density[9],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );

    let dirac = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::DiracFockOnly, true),
    )?;
    assert_close_with(dirac.potential[0], 1.000_000_000_000_000_05e-4, 1.0e-19);
    assert_close_with(dirac.potential[4], 5.000_000_000_000_000_10e-4, 1.0e-19);
    assert_close_with(dirac.potential[9], 1.000_000_000_000_000_02e-3, 1.0e-18);
    assert_close_with(
        dirac.development_coefficients[1],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[0],
        2.000_000_000_000_000_04e-3,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[4],
        1.000_000_000_000_000_02e-2,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[9],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_orbital_potential_matches_feff_potrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_potrdf_fixture();

    let full = atomic_orbital_potential(fixture.input(true, true))?;
    for (index, expected) in [
        (0, -1.451_464_734_879_546_50e-3),
        (4, -1.422_294_851_220_632_99e-3),
        (9, -1.385_920_785_309_911_19e-3),
        (12, -1.364_108_165_051_381_58e-3),
    ] {
        assert_close_with(full.central_potential[index], expected, 1.0e-15);
    }
    for (index, expected) in [
        (0, -2.189_205_772_127_074_25e-4),
        (1, -4.371_323_520_763_144_61e-4),
        (3, -8.773_080_991_906_762_18e-4),
        (5, -1.317_263_492_825_318_69e-3),
    ] {
        assert_close_with(
            full.central_development_coefficients[index],
            expected,
            1.0e-15,
        );
    }
    for (index, expected) in [
        (0, 1.702_743_222_291_228_72e-7),
        (4, 2.294_031_531_020_954_80e-7),
        (9, 0.0),
    ] {
        assert_close_with(full.exchange_large[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, -4.763_258_868_894_551_20e-8),
        (4, -4.776_069_610_481_555_18e-8),
        (9, 0.0),
    ] {
        assert_close_with(full.exchange_small[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, 2.307_477_389_651_008_40e-5),
        (2, 4.794_137_619_410_912_88e-5),
        (5, 7.832_202_463_049_932_73e-5),
    ] {
        assert_close_with(full.exchange_large_coefficients[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, 1.845_981_911_720_806_31e-6),
        (2, -4.841_519_661_027_267_90e-6),
        (5, -1.331_336_809_940_218_72e-5),
    ] {
        assert_close_with(full.exchange_small_coefficients[index], expected, 1.0e-16);
    }

    let direct = atomic_orbital_potential(fixture.input(false, false))?;
    for (actual, expected) in direct
        .central_potential
        .iter()
        .zip(full.central_potential.iter())
    {
        assert_close_with(*actual, *expected, 1.0e-16);
    }
    for (actual, expected) in direct
        .central_development_coefficients
        .iter()
        .zip(full.central_development_coefficients.iter())
    {
        assert_close_with(*actual, *expected, 1.0e-16);
    }
    for value in direct
        .exchange_large
        .iter()
        .chain(direct.exchange_small.iter())
        .chain(direct.exchange_large_coefficients.iter())
        .chain(direct.exchange_small_coefficients.iter())
    {
        assert_close_with(*value, 0.0, 1.0e-20);
    }
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_transform_matches_feff_yzkteg_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkteg_fixture();
    let transform = atomic_yk_zk_transform(fixture.input())?;

    assert_eq!(transform.computed_source_len, 9);
    assert_close_with(
        transform.origin_constant,
        1.024_939_588_738_283_48e2,
        1.0e-11,
    );
    assert_close_with(transform.yk[0], 3.871_202_667_947_041_34e-4, 1.0e-16);
    assert_close_with(transform.yk[1], 4.476_978_947_879_065_22e-4, 1.0e-16);
    assert_close_with(transform.yk[4], 6.350_731_526_853_801_77e-4, 1.0e-16);
    assert_close_with(transform.yk[8], 6.665_230_606_586_294_07e-4, 1.0e-16);
    assert_close_with(transform.yk[12], 4.467_837_687_045_075_67e-4, 1.0e-16);
    assert_close_with(transform.zk[0], 1.055_350_291_449_006_03e-5, 1.0e-17);
    assert_close_with(transform.zk[1], 1.147_457_094_885_342_41e-5, 1.0e-17);
    assert_close_with(transform.zk[4], 1.675_242_796_907_188_86e-4, 1.0e-16);
    assert_close_with(transform.zk[9], 7.118_915_805_710_559_43e-4, 1.0e-16);
    assert_close_with(
        transform.yk_coefficients[0],
        -3.906_646_372_399_797_53e-2,
        1.0e-16,
    );
    assert_close_with(
        transform.yk_coefficients[3],
        6.197_311_460_469_354_11e-2,
        1.0e-16,
    );
    assert_close_with(
        transform.zk_coefficients[0],
        1.054_794_520_547_945_24e-2,
        1.0e-17,
    );
    assert_close_with(
        transform.zk_coefficients[3],
        2.045_112_781_954_887_27e-2,
        1.0e-17,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_prepared_source_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkteg_fixture();
    let rank_two = atomic_yk_zk_prepared_source(fixture.prepared_input(9, 2))?;
    assert_eq!(rank_two.computed_source_len, 9);
    assert_close_with(
        rank_two.origin_constant,
        1.110_957_296_725_969_88e2,
        1.0e-11,
    );
    assert_close_with(rank_two.yk[0], 3.746_164_822_999_324_47e-4, 1.0e-16);
    assert_close_with(rank_two.yk[1], 4.361_981_443_957_904_09e-4, 1.0e-16);
    assert_close_with(rank_two.yk[4], 6.265_729_070_725_439_66e-4, 1.0e-16);
    assert_close_with(rank_two.yk[8], 6.608_249_600_892_370_22e-4, 1.0e-16);
    assert_close_with(rank_two.yk[12], 4.429_642_176_685_166_84e-4, 1.0e-16);
    assert_close_with(rank_two.zk[0], 4.277_638_252_436_042_60e-12, 1.0e-22);
    assert_close_with(rank_two.zk[1], 5.499_800_258_296_022_76e-12, 1.0e-22);
    assert_close_with(rank_two.zk[4], 1.590_237_125_316_554_21e-4, 1.0e-16);
    assert_close_with(rank_two.zk[9], 7.067_357_259_641_375_48e-4, 1.0e-16);
    assert_close_with(
        rank_two.yk_coefficients[0],
        1.374_999_999_999_999_83e-2,
        1.0e-17,
    );
    assert_close_with(
        rank_two.yk_coefficients[3],
        1.360_000_000_000_000_10e-2,
        1.0e-17,
    );

    let rank_one = atomic_yk_zk_prepared_source(fixture.prepared_input(7, 1))?;
    assert_eq!(rank_one.computed_source_len, 7);
    assert_close_with(rank_one.origin_constant, 1.293_492_132_385_440_25, 1.0e-13);
    assert_close_with(rank_one.yk[0], 2.908_635_211_432_032_27e-4, 1.0e-16);
    assert_close_with(rank_one.yk[1], 3.220_388_501_435_997_46e-4, 1.0e-16);
    assert_close_with(rank_one.yk[4], 4.003_521_683_966_694_17e-4, 1.0e-16);
    assert_close_with(rank_one.yk[8], 3.610_570_331_017_010_91e-4, 1.0e-16);
    assert_close_with(rank_one.yk[12], 2.956_084_966_154_574_63e-4, 1.0e-16);
    assert_close_with(rank_one.zk[0], 3.988_806_776_811_954_55e-10, 1.0e-20);
    assert_close_with(rank_one.zk[1], 4.878_024_038_015_732_55e-10, 1.0e-20);
    assert_close_with(rank_one.zk[4], 1.686_537_565_491_518_30e-4, 1.0e-16);
    assert_close_with(rank_one.zk[9], 0.0, 1.0e-20);
    assert_close_with(
        rank_one.yk_coefficients[0],
        1.155_000_000_000_000_12e-2,
        1.0e-17,
    );
    assert_close_with(
        rank_one.yk_coefficients[3],
        1.020_000_000_000_000_07e-2,
        1.0e-17,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_exchange_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkrdf_fixture();
    let overlap = atomic_yk_zk_exchange(fixture.yzkrdf_input(1, 2, 2, false))?;
    assert_eq!(overlap.computed_source_len, 9);
    assert_close_with(overlap.origin_constant, -2.571_240_643_442_588_96, 1.0e-12);
    assert_close_with(overlap.yk[0], 1.109_878_400_538_443_00e-5, 1.0e-17);
    assert_close_with(overlap.yk[1], 1.135_633_080_766_094_54e-5, 1.0e-17);
    assert_close_with(overlap.yk[4], 1.178_867_152_957_986_59e-5, 1.0e-17);
    assert_close_with(overlap.yk[8], 1.017_973_162_090_520_64e-5, 1.0e-17);
    assert_close_with(overlap.yk[12], 6.823_678_168_755_628_77e-6, 1.0e-18);
    assert_close_with(overlap.zk[0], 5.468_221_372_334_369_25e-6, 1.0e-18);
    assert_close_with(overlap.zk[1], 5.909_940_448_128_294_29e-6, 1.0e-18);
    assert_close_with(overlap.zk[4], 7.024_129_238_136_815_07e-6, 1.0e-18);
    assert_close_with(overlap.zk[9], 1.014_708_699_883_866_62e-5, 1.0e-17);
    assert_close_with(
        overlap.yk_coefficients[0],
        -9.990_630_795_999_924_30e-3,
        1.0e-17,
    );
    assert_close_with(
        overlap.yk_coefficients[3],
        8.575_701_162_755_210_16e-2,
        1.0e-16,
    );

    let large_small = atomic_yk_zk_exchange(fixture.yzkrdf_input(2, 3, 1, true))?;
    assert_eq!(large_small.computed_source_len, 7);
    assert_close_with(
        large_small.origin_constant,
        -2.237_401_842_533_894_71e-2,
        1.0e-14,
    );
    assert_close_with(large_small.yk[0], -1.770_958_131_971_287_30e-6, 1.0e-18);
    assert_close_with(large_small.yk[1], -2.024_241_049_179_754_12e-6, 1.0e-18);
    assert_close_with(large_small.yk[4], -2.505_938_578_653_440_58e-6, 1.0e-18);
    assert_close_with(large_small.yk[8], -2.208_316_861_767_755_49e-6, 1.0e-18);
    assert_close_with(large_small.yk[12], -1.808_016_927_269_919_53e-6, 1.0e-18);
    assert_close_with(large_small.zk[0], 3.406_624_777_460_352_47e-7, 1.0e-19);
    assert_close_with(large_small.zk[1], 3.708_373_404_554_750_70e-7, 1.0e-19);
    assert_close_with(large_small.zk[4], -1.328_125_640_689_300_04e-6, 1.0e-18);
    assert_close_with(large_small.zk[9], 0.0, 1.0e-19);
    assert_close_with(
        large_small.yk_coefficients[0],
        -3.957_309_029_859_694_58e-3,
        1.0e-17,
    );
    assert_close_with(
        large_small.yk_coefficients[3],
        -2.038_402_989_657_719_41e-3,
        1.0e-17,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_radial_integral_matches_feff_fdrirk_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkrdf_fixture();
    let kappas = [-1, 1, -2];

    let overlap = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 1,
            first_right: 2,
            second_left: 1,
            second_right: 3,
            rank: 2,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close_with(overlap.value, 3.844_030_024_958_072_30e-9, 1.0e-20);
    let overlap_factor = overlap
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    assert_close_with(
        overlap_factor.values[0],
        1.109_878_400_538_443_00e-5,
        1.0e-17,
    );
    assert_close_with(
        overlap_factor.values[3],
        1.171_927_755_618_356_82e-5,
        1.0e-17,
    );
    assert_close_with(
        overlap_factor.coefficients[0],
        -2.561_250_012_646_588_91,
        1.0e-12,
    );
    assert_close_with(
        overlap_factor.coefficients[3],
        -8.575_701_162_755_210_16e-2,
        1.0e-16,
    );

    let large_small = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 2,
            first_right: 3,
            second_left: 1,
            second_right: 2,
            rank: 1,
        },
        &kappas,
        true,
        None,
    ))?;
    assert_close_with(large_small.value, 2.056_815_682_976_472_25e-10, 1.0e-21);
    let large_small_factor = large_small
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    assert_close_with(
        large_small_factor.coefficients[0],
        -2.237_401_842_533_894_71e-2,
        1.0e-14,
    );
    assert_close_with(
        large_small_factor.coefficients[3],
        9.462_409_003_166_756_97e-4,
        1.0e-17,
    );

    let first = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 2,
            first_right: 1,
            second_left: 2,
            second_right: 1,
            rank: 1,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close_with(first.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
    let previous = first
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    let sentinel = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 0,
            first_right: 0,
            second_left: 1,
            second_right: 2,
            rank: 1,
        },
        &kappas,
        false,
        Some(previous.as_view()),
    ))?;
    assert_close_with(sentinel.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
    assert!(sentinel.first_factor.is_none());

    let no_second = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 1,
            first_right: 2,
            second_left: 0,
            second_right: 0,
            rank: 2,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close(no_second.value, 0.0);
    assert!(no_second.first_factor.is_some());
    Ok(())
}

#[test]
fn atom_form_factor_matches_feff_fpf0_reference() -> Result<(), AtomMathError> {
    let radial_count = 251;
    let orbital_count = 5;
    let radial_step = 0.05;
    let radii = Array1::from_shape_fn(radial_count, |index| {
        (-8.8 + radial_step * index as Real).exp()
    });
    let density_4pi = Array1::from_shape_fn(radial_count, |index| {
        0.3 * (-0.7 * radii[index]).exp() + 0.01 * (index + 1).rem_euclid(7) as Real
    });
    let initial_large_component = Array1::from_shape_fn(radial_count, |index| {
        0.2 * (-0.4 * radii[index]).exp() + 0.001 * (index + 1) as Real
    });
    let initial_small_component = Array1::from_shape_fn(radial_count, |index| {
        -0.05 * (-0.3 * radii[index]).exp() + 0.0002 * (index + 1) as Real
    });
    let large_components = Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
        let orbital = (col + 1) as Real;
        (0.03 * orbital + 0.0007 * (row + 1) as Real) * (-0.05 * orbital * radii[row]).exp()
    });
    let small_components = Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
        let orbital = (col + 1) as Real;
        (-0.01 * orbital + 0.0003 * (row + 1) as Real) * (-0.03 * orbital * radii[row]).exp()
    });
    let occupations = [2.0, 2.0, 1.5, 0.5, 0.0];
    let orbital_energies = [-0.85, -0.55, -0.21, -0.08, 0.04];
    let kappas = [-1, 1, -2, 2, -1];

    let form_factor = atomic_form_factor(AtomicFormFactorInput {
        atomic_number: 26,
        hole_orbital_1based: 2,
        radial_step,
        total_energy: -2.345,
        radii: radii.view(),
        density_4pi: density_4pi.view(),
        initial_large_component: initial_large_component.view(),
        initial_small_component: initial_small_component.view(),
        large_components: large_components.view(),
        small_components: small_components.view(),
        occupations: &occupations,
        orbital_energies: &orbital_energies,
        kappas: &kappas,
    })?;

    assert_eq!(form_factor.atomic_number, 26);
    assert_close_with(form_factor.total_energy_fprime, -2.081_24e-4, 5.0e-10);
    assert_close_with(form_factor.relativistic_correction, -6.478_75e-2, 5.0e-8);
    assert_eq!(form_factor.oscillators.len(), 3);
    let expected_oscillators = [(2.0, -0.55, 2), (0.104_07, -0.85, 1), (0.003_60, -0.08, 4)];
    for (actual, (strength, energy, index)) in
        form_factor.oscillators.iter().zip(expected_oscillators)
    {
        assert_close_with(actual.oscillator_strength, strength, 5.0e-6);
        assert_close_with(actual.excitation_energy, energy, 5.0e-13);
        assert_eq!(actual.orbital_index_1based, index);
    }
    assert_eq!(form_factor.form_factor.len(), 81);
    let expected_rows = [
        (0, 0.0, 760.5215),
        (1, 0.5, -4.0195),
        (2, 1.0, 16.7054),
        (3, 1.5, -1.1065),
        (4, 2.0, -0.5452),
        (10, 5.0, 1.4707),
        (20, 10.0, -0.1129),
        (40, 20.0, -0.6736),
        (80, 40.0, 0.1214),
    ];
    for (index, momentum, value) in expected_rows {
        assert_close_with(form_factor.form_factor_momentum[index], momentum, 1.0e-13);
        assert_close_with(form_factor.form_factor[index], value, 5.5e-5);
    }
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_schmidt_orthogonalization_matches_feff_ortdat_reference() -> Result<(), AtomMathError> {
    let fixture = sample_schmidt_fixture();
    let all_orbitals =
        atomic_schmidt_orthogonalization(fixture.as_input(None), sample_schmidt_integral)?;
    assert_eq!(all_orbitals.active_lengths, vec![3, 4, 3, 5]);
    assert_columns_close(
        &all_orbitals.large_components,
        &[
            [0.18, 0.25, 0.32, 0.39, 0.46],
            [
                0.333_475_933_348_347_96,
                0.403_443_338_654_020_99,
                0.473_410_743_959_694_18,
                0.697_998_855_802_804_52,
                0.57,
            ],
            [
                0.487_117_140_335_587_17,
                0.572_362_639_894_314_91,
                0.657_608_139_453_042_64,
                0.61,
                0.68,
            ],
            [
                0.086_758_208_000_696_446,
                0.041_346_281_239_887_581,
                -0.004_065_645_520_921_706_5,
                -0.041_673_823_238_614_134,
                0.979_213_171_940_273_24,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.small_components,
        &[
            [0.01, 0.04, 0.07, 0.1, 0.13],
            [
                -0.017_924_610_617_016_022,
                0.012_061_420_228_272_458,
                0.042_047_451_073_560_942,
                0.111_679_816_928_448_71,
                0.11,
            ],
            [
                -0.036_533_785_525_169_032,
                0.0,
                0.036_533_785_525_169_032,
                0.06,
                0.09,
            ],
            [
                -0.043_493_187_919_062_107,
                -0.062_955_442_245_123_172,
                -0.082_417_696_571_184_237,
                -0.099_878_989_604_138_421,
                0.086_765_724_095_973_565,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.large_coefficients,
        &[
            [0.25, 0.45, 0.65, 0.85],
            [
                0.319_683_475_957_684_54,
                0.519_590_348_259_607_70,
                0.719_497_220_561_530_87,
                0.919_404_092_863_454_04,
            ],
            [
                0.426_227_497_793_638_78,
                0.669_786_067_961_432_36,
                0.913_344_638_129_225_95,
                1.156_903_208_297_019_4,
            ],
            [
                -0.069_671_028_191_237_896,
                -0.199_419_390_364_978_30,
                -0.329_167_752_538_718_77,
                -0.458_916_114_712_459_13,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.small_coefficients,
        &[
            [0.01, -0.02, -0.05, -0.08],
            [
                0.065_835_252_079_320_519,
                0.035_849_221_234_032_044,
                0.005_863_190_388_743_565_9,
                -0.024_122_840_456_544_916,
            ],
            [
                0.109_601_356_575_507_10,
                0.073_067_571_050_338_065,
                0.036_533_785_525_169_032,
                0.0,
            ],
            [
                0.067_524_121_512_063_162,
                0.086_986_375_838_124_214,
                0.106_448_630_164_185_28,
                0.125_910_884_490_246_34,
            ],
        ],
        1.0e-12,
    );

    let active_two =
        atomic_schmidt_orthogonalization(fixture.as_input(Some(2)), sample_schmidt_integral)?;
    assert_eq!(active_two.active_lengths, vec![3, 5, 3, 5]);
    assert_columns_close(
        &active_two.large_components,
        &[
            [0.18, 0.25, 0.32, 0.39, 0.46],
            [
                -0.257_731_473_167_008_73,
                -0.271_503_234_760_490_32,
                -0.285_274_996_353_971_69,
                -0.160_996_405_265_147_69,
                -0.860_433_208_548_678_89,
            ],
            [0.4, 0.47, 0.54, 0.61, 0.68],
            [0.51, 0.58, 0.65, 0.72, 0.79],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.small_components,
        &[
            [0.01, 0.04, 0.07, 0.1, 0.13],
            [
                0.038_454_127_655_123_280,
                0.032_551_944_115_059_794,
                0.026_649_760_574_996_302,
                0.056_145_103_363_729_076,
                -0.076_240_917_213_174_053,
            ],
            [-0.03, 0.0, 0.03, 0.06, 0.09],
            [-0.05, -0.02, 0.01, 0.04, 0.07],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.large_coefficients,
        &[
            [0.25, 0.45, 0.65, 0.85],
            [
                -0.150_238_668_255_056_88,
                -0.189_586_558_522_146_90,
                -0.228_934_448_789_236_94,
                -0.268_282_339_056_326_81,
            ],
            [0.35, 0.55, 0.75, 0.95],
            [0.4, 0.6, 0.8, 1.0],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.small_coefficients,
        &[
            [0.01, -0.02, -0.05, -0.08],
            [
                -0.082_810_438_850_310_059,
                -0.076_908_255_310_246_559,
                -0.071_006_071_770_183_060,
                -0.065_103_888_230_119_589,
            ],
            [0.09, 0.06, 0.03, 0.0],
            [0.13, 0.10, 0.07, 0.04],
        ],
        1.0e-12,
    );

    Ok(())
}

#[test]
fn atom_overlap_amplitude_reduction_matches_feff_s02at_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, -1, 1, 1, -2, -3];
    let occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
    let overlaps = sample_s02at_overlaps();
    let cases = [
        (None, 9.680_452_235_999_996e-3),
        (Some(1), 9.680_452_235_999_996e-3),
        (Some(2), 0.327_600_000_000_000_1),
        (Some(3), 9.680_452_235_999_996e-3),
        (Some(4), 9.020_027_472_527_463e-2),
        (Some(5), 9.680_452_235_999_996e-3),
        (Some(6), 9.680_452_235_999_996e-3),
    ];

    for (hole_orbital_1based, expected) in cases {
        let actual = atomic_overlap_amplitude_reduction(AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based,
            kappas: &kappas,
            occupations: &occupations,
            overlap_integrals: overlaps.view(),
        })?;
        assert_close(actual, expected);
    }
    Ok(())
}

#[test]
fn atom_helper_kernels_reject_invalid_inputs() {
    assert!(matches!(
        atomic_polynomial_product_coefficient(&[1.0], &[2.0], 2),
        Err(AtomMathError::InvalidPolynomialTerm { .. })
    ));
    assert!(matches!(
        atomic_convergence_mix(0.5, Real::INFINITY, 1.0),
        Err(AtomMathError::NonFiniteScalar {
            field: "current_error",
            ..
        })
    ));
    assert!(matches!(
        thomas_fermi_density_potential(0.0, 1.0, 0.0),
        Err(AtomMathError::NonPositiveRadius { .. })
    ));
    assert!(matches!(
        atomic_occupation_product(&[1.0], &[], 0, 0),
        Err(AtomMathError::OccupationKappaLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_occupation_product(&[1.0], &[0], 0, 0),
        Err(AtomMathError::ZeroKappa)
    ));

    let coefficients = Array3::zeros((2, 2, 1));
    assert!(matches!(
        atomic_direct_coulomb_coefficient(coefficients.view(), 0, 0, 4),
        Err(AtomMathError::CoefficientChannelOutOfRange { .. })
    ));

    let coefficients = Array3::zeros((2, 3, 1));
    assert!(matches!(
        atomic_direct_coulomb_coefficient(coefficients.view(), 1, 2, 0),
        Err(AtomMathError::CoefficientTableShape { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[1],
            occupations: &[],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[0],
            occupations: &[1.0],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[1],
            occupations: &[Real::NAN],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::NonFiniteScalar {
            field: "occupation",
            ..
        })
    ));
    assert!(matches!(
        atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &[6],
            occupations: &[2.0],
            valence_occupations: &[0.0],
        }),
        Err(AtomMathError::CoefficientChannelOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 0.0,
            step: 0.05,
            requested_nucleus_index: 1,
            radial_count: 251,
            coefficient_count: 10,
            first_radius_times_charge: 1.0,
        }),
        Err(AtomMathError::InvalidNuclearPotentialScalar { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 92.0,
            step: 0.05,
            requested_nucleus_index: -11,
            radial_count: 5,
            coefficient_count: 10,
            first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
        }),
        Err(AtomMathError::NuclearRadiusOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: 26.0,
            step: 0.05,
            requested_nucleus_index: 1,
            radial_count: 251,
            coefficient_count: 4,
            first_radius_times_charge: 26.0 * (-8.8_f64).exp(),
        }),
        Err(AtomMathError::InvalidNuclearPotentialCount { .. })
    ));
    let dsordf = sample_dsordf_fixture();
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 8 },
            0,
            0.45,
        )),
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
    ));
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 0,
                right_orbital_1based: 1,
                multiply_by_derivative: false,
            },
            0,
            0.0,
        )),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_differential_integral(dsordf.input(
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
            0,
            -1.0,
        )),
        Err(AtomMathError::ZeroDifferentialIntegralOriginExponent)
    ));
    let bad_radii = Array1::from_vec(vec![0.0; 11]);
    assert!(matches!(
        atomic_differential_integral(AtomicDifferentialIntegralInput {
            radii: bad_radii.view(),
            ..dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                0.45,
            )
        }),
        Err(AtomMathError::NonPositiveRadius { .. })
    ));
    let bad_derivative_coefficients = Array1::from_vec(vec![0.1; 5]);
    assert!(matches!(
        atomic_differential_integral(AtomicDifferentialIntegralInput {
            derivative_large_coefficients: bad_derivative_coefficients.view(),
            ..dsordf.input(
                AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
                0,
                0.45,
            )
        }),
        Err(AtomMathError::CoefficientTableLengthMismatch { .. })
    ));
    let yzkteg = sample_yzkteg_fixture();
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            active_len: 3,
            ..yzkteg.input()
        }),
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength { .. })
    ));
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            step: 0.0,
            ..yzkteg.input()
        }),
        Err(AtomMathError::ZeroYkZkDenominator { field: "step" })
    ));
    assert!(matches!(
        atomic_yk_zk_transform(AtomicYkZkTransformInput {
            initial_power: 2.0,
            ..yzkteg.input()
        }),
        Err(AtomMathError::ZeroYkZkDenominator { field: "yk_origin" })
    ));
    let yzkrdf = sample_yzkrdf_fixture();
    assert!(matches!(
        atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
            left_orbital_1based: 0,
            ..yzkrdf.yzkrdf_input(1, 2, 2, false)
        }),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        AtomicLocalDensityExchangeMode::try_from(4),
        Err(AtomMathError::InvalidExchangeMode { idfock: 4 })
    ));
    let vlda = sample_vlda_fixture();
    assert!(matches!(
        atomic_local_density_potential(AtomicLocalDensityPotentialInput {
            speed_of_light: 0.0,
            ..vlda.input(AtomicLocalDensityExchangeMode::TotalDensity, false)
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 4,
            ionicity: 0.0,
            principal_quantum_numbers: &[2],
            kappas: &[1],
            occupations: &[1.0],
        }),
        Err(AtomMathError::ElectronCountMismatch { .. })
    ));
    assert!(matches!(
        atomic_orbital_initialization(AtomicOrbitalInitializationInput {
            atomic_number: 1,
            ionicity: 0.0,
            principal_quantum_numbers: &[1],
            kappas: &[1],
            occupations: &[1.0],
        }),
        Err(AtomMathError::OrbitalAngularMomentumOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_entry_state(AtomicDiracEntryStateInput {
            asymptotic_large_component: Real::NAN,
            method: 1,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    let soldir_norm = sample_soldir_norm_fixture();
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 10, 5)),
        Err(AtomMathError::InvalidDiracNormalizationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(1, 6, 0.177, 0.82, 11, 0)),
        Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_normalization(soldir_norm.input(2, 6, 0.177, -0.5, 11, 5)),
        Err(AtomMathError::ZeroDiracNormalizationOriginExponent)
    ));
    let soldir_solution_norm = sample_soldir_solution_normalization_fixture(false, false);
    assert!(matches!(
        atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
            active_len: 0,
            ..soldir_solution_norm.input(6.25, 0.8, -0.4)
        }),
        Err(AtomMathError::InvalidDiracSolutionNormalizationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_solution_normalization(AtomicDiracSolutionNormalizationInput {
            norm: 0.0,
            ..soldir_solution_norm.input(6.25, 0.8, -0.4)
        }),
        Err(AtomMathError::NonPositiveScalar {
            field: "soldir_solution_norm",
            ..
        })
    ));
    let soldir_nodes = sample_soldir_node_count_component();
    assert!(matches!(
        atomic_dirac_node_count(AtomicDiracNodeCountInput {
            large_component: soldir_nodes.view(),
            matching_index_1based: 0,
            scan_index_1based: 3,
        }),
        Err(AtomMathError::InvalidDiracNodeCountIndex { .. })
    ));
    assert!(matches!(
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 1.0,
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_small_component: 0.0,
            matching_index_1based: 0,
        },),
        Err(AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
            energy: 0.0,
            correction: 0.0,
            mismatch: 0.0,
            energy_sup: -1.0,
            energy_inf: -0.1,
            mismatch_precision: 0.1,
            zero_energy_precision: 1.0e-7,
        }),
        Err(AtomMathError::ZeroDiracEnergyCorrectionDenominator)
    ));
    assert!(matches!(
        atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: Real::NAN,
            mismatch_precision: 0.1,
            match_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
            mismatch: 1.0,
            mismatch_precision: 0.1,
            match_attempt_count: usize::MAX,
            max_attempt_count: usize::MAX,
        }),
        Err(AtomMathError::DiracRematchAttemptCountOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: 0.0,
            previous_energy: -0.1,
        }),
        Err(AtomMathError::ZeroDiracShootingPassEnergy)
    ));
    assert!(matches!(
        atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
            energy: Real::NAN,
            previous_energy: -0.1,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: Real::NAN,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 0.0,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: usize::MAX,
            max_attempt_count: usize::MAX,
        }),
        Err(AtomMathError::DiracNodeEnergyAttemptCountOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
            method: 1,
            primary_matching_precision: 0.0,
            secondary_matching_precision: 1.0e-5,
            energy_floor: -5.0,
            reference_energy: -0.5,
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
            method: 1,
            primary_matching_precision: 1.0e-5,
            secondary_matching_precision: 2.0e-5,
            energy_floor: Real::NAN,
            reference_energy: -0.5,
        }),
        Err(AtomMathError::NonFiniteScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 1,
            method: i32::MAX,
        }),
        Err(AtomMathError::DiracMethodOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
            large_source: soldir_nodes.view(),
            small_source: soldir_nodes
                .view()
                .slice_axis(Axis(0), Slice::from(..soldir_nodes.len() - 1)),
            large_source_coefficients: soldir_nodes.view(),
            small_source_coefficients: soldir_nodes.view(),
            coefficient_count: 1,
        }),
        Err(AtomMathError::RadialTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
            large_source: soldir_nodes.view(),
            small_source: soldir_nodes.view(),
            large_source_coefficients: soldir_nodes.view().slice_axis(Axis(0), Slice::from(..1)),
            small_source_coefficients: soldir_nodes.view(),
            coefficient_count: 2,
        }),
        Err(AtomMathError::CoefficientTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
            radial_len: 0,
            coefficient_len: 1,
        }),
        Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_radial_len",
            ..
        })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
            radial_len: 1,
            coefficient_len: 0,
        }),
        Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_coefficient_len",
            ..
        })
    ));
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 0,
        }),
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange { .. })
    ));
    let zero_match_denominator = Array1::<Real>::zeros(soldir_nodes.len());
    assert!(matches!(
        atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            homogeneous_large_component: zero_match_denominator.view(),
            homogeneous_small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 1,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            matching_large_component: 0.0,
            active_len: soldir_nodes.len(),
            matching_index_1based: 4,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            homogeneous_large_coefficients: soldir_nodes.view(),
            homogeneous_small_coefficients: soldir_nodes.view(),
            matching_large_component: 0.0,
            matching_small_component: 0.0,
            homogeneous_matching_large_component: 1.0,
            homogeneous_matching_small_component: 1.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
            matching_index_1based: 1,
        }),
        Err(AtomMathError::ZeroDiracMatchDenominator { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            homogeneous_large_component: soldir_nodes.view(),
            homogeneous_small_component: soldir_nodes.view(),
            homogeneous_large_coefficients: soldir_nodes.view(),
            homogeneous_small_coefficients: soldir_nodes.view(),
            matching_large_derivative: 0.0,
            matching_small_derivative: 0.0,
            homogeneous_matching_large_component: 1.0,
            homogeneous_matching_small_component: 1.0,
            coefficient_count: 1,
            active_len: 0,
            matching_index_1based: 1,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    let positive_radii = Array1::from_elem(soldir_nodes.len(), 0.1);
    assert!(matches!(
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            radii: positive_radii.view(),
            speed_of_light: 137.0373,
            coefficient_count: 1,
            active_len: 0,
        }),
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            radii: positive_radii.view(),
            speed_of_light: 0.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        }),
        Err(AtomMathError::NonPositiveScalar { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 1.0,
            step: 0.11,
            origin_power: 1.0,
            coefficient_count: 1,
            active_len: 8,
        },),
        Err(AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: soldir_nodes.view(),
            small_derivative: soldir_nodes.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 1.0,
            step: 0.11,
            origin_power: -0.5,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        },),
        Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionOriginExponent)
    ));
    let zero_derivative = Array1::<Real>::zeros(soldir_nodes.len());
    assert!(matches!(
        atomic_dirac_energy_disagreement_correction(AtomicDiracEnergyDisagreementCorrectionInput {
            radii: positive_radii.view(),
            large_component: soldir_nodes.view(),
            small_component: soldir_nodes.view(),
            large_derivative: zero_derivative.view(),
            small_derivative: zero_derivative.view(),
            large_coefficients: soldir_nodes.view(),
            small_coefficients: soldir_nodes.view(),
            large_derivative_coefficients: soldir_nodes.view(),
            small_derivative_coefficients: soldir_nodes.view(),
            norm: 0.9,
            step: 0.11,
            origin_power: 1.0,
            coefficient_count: 1,
            active_len: soldir_nodes.len(),
        },),
        Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionIntegral)
    ));
    assert!(matches!(
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: soldir_nodes.view(),
            active_len: 9,
            matching_index_1based: 5,
            already_relocated: false,
        }),
        Err(AtomMathError::InvalidDiracMatchActiveLength { .. })
    ));
    let zero_matching_component = Array1::<Real>::zeros(13);
    assert!(matches!(
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: zero_matching_component.view(),
            active_len: 13,
            matching_index_1based: 5,
            already_relocated: false,
        }),
        Err(AtomMathError::DiracMatchingPointNotFound { .. })
    ));
    let intdir = sample_intdir_fixture();
    assert!(matches!(
        atomic_dirac_integration(AtomicDiracIntegrationInput {
            active_len: 12,
            ..intdir.input(AtomicDiracIntegrationMode::SearchMatchingPoint, 0, 0)
        }),
        Err(AtomMathError::InvalidDiracIntegrationActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(intdir.input(
            AtomicDiracIntegrationMode::FixedMatchingPoint,
            5,
            139
        )),
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 68)),
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_dirac_integration(AtomicDiracIntegrationInput {
            energy: 0.01,
            ..intdir.input(AtomicDiracIntegrationMode::InwardOnly, 65, 139)
        }),
        Err(AtomMathError::InvalidDiracIntegrationEnergy { .. })
    ));
    let soldir_setup = sample_soldir_setup_fixture();
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            active_len: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength { .. })
    ));
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            kappa: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            principal_quantum_number: 0,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber { .. })
    ));
    let positive_potential = Array1::from_vec(vec![0.25; 7]);
    assert!(matches!(
        atomic_dirac_solver_setup(AtomicDiracSolverSetupInput {
            potential: positive_potential.view(),
            kappa: 2,
            ..soldir_setup.input(-0.2, 2, 2, 4, true)
        }),
        Err(AtomMathError::DiracSolverPotentialNotAttractive { .. })
    ));
    let potrdf = sample_potrdf_fixture();
    assert!(matches!(
        atomic_orbital_potential(AtomicOrbitalPotentialInput {
            active_orbital_1based: 0,
            ..potrdf.input(true, true)
        }),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    let kappas = [-1, 1, -2];
    assert!(matches!(
        atomic_radial_integral(yzkrdf.fdrirk_input(
            AtomicRadialIntegralRequest {
                first_left: 0,
                first_right: 0,
                second_left: 1,
                second_right: 2,
                rank: 1,
            },
            &kappas,
            false,
            None,
        )),
        Err(AtomMathError::MissingRadialFirstFactor)
    ));
    let coefficients = Array3::zeros((2, 2, 1));
    assert!(matches!(
        atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: Some(0),
                kappas: &[1, 1],
                occupations: &[1.0, 2.0],
                shell_markers: &[1, 1],
                include_exchange: true,
                coulomb_coefficients: coefficients.view(),
            },
            |_| Ok(0.0),
        ),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_lagrange_parameters(
            AtomicLagrangeParametersInput {
                active_orbital_1based: None,
                kappas: &[1, 1],
                occupations: &[0.0, 2.0],
                shell_markers: &[1, 1],
                include_exchange: true,
                coulomb_coefficients: coefficients.view(),
            },
            |_| Ok(0.0),
        ),
        Err(AtomMathError::NonPositiveOccupation { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[1, 1],
                occupations: &[1.0, 2.0],
                orbital_energies: &[-0.1, -0.2],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[0],
                kappas: &[1],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::InvalidPrincipalQuantumNumber { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[5],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            sample_atomic_tabrat_integral,
        ),
        Err(AtomMathError::OrbitalLabelKappaOutOfRange { .. })
    ));
    assert!(matches!(
        atomic_tabulation(
            AtomicTabulationInput {
                principal_quantum_numbers: &[1],
                kappas: &[1],
                occupations: &[1.0],
                orbital_energies: &[-0.1],
            },
            |_| Ok(Real::NAN),
        ),
        Err(AtomMathError::NonFiniteScalar {
            field: "tabrat_integral",
            ..
        })
    ));
    let fpf0_radii = Array1::from_vec(vec![1.0, 1.2]);
    let fpf0_values = Array1::from_vec(vec![0.1, 0.2]);
    let fpf0_components = Array2::zeros((2, 1));
    let fpf0_input = AtomicFormFactorInput {
        atomic_number: 26,
        hole_orbital_1based: 1,
        radial_step: 0.05,
        total_energy: -1.0,
        radii: fpf0_radii.view(),
        density_4pi: fpf0_values.view(),
        initial_large_component: fpf0_values.view(),
        initial_small_component: fpf0_values.view(),
        large_components: fpf0_components.view(),
        small_components: fpf0_components.view(),
        occupations: &[1.0],
        orbital_energies: &[-0.2],
        kappas: &[1],
    };
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            atomic_number: 0,
            ..fpf0_input
        }),
        Err(AtomMathError::InvalidFormFactorAtomicNumber { .. })
    ));
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            hole_orbital_1based: 2,
            ..fpf0_input
        }),
        Err(AtomMathError::HoleOrbitalOutOfRange { .. })
    ));
    let bad_fpf0_density = Array1::from_vec(vec![0.1]);
    assert!(matches!(
        atomic_form_factor(AtomicFormFactorInput {
            density_4pi: bad_fpf0_density.view(),
            ..fpf0_input
        }),
        Err(AtomMathError::RadialTableLengthMismatch { .. })
    ));
    let schmidt = sample_schmidt_fixture();
    assert!(matches!(
        atomic_schmidt_orthogonalization(schmidt.as_input(Some(5)), sample_schmidt_integral),
        Err(AtomMathError::ActiveOrbitalOutOfRange { .. })
    ));

    let bad_small_components = Array2::<Real>::zeros((4, 4));
    assert!(matches!(
        atomic_schmidt_orthogonalization(
            AtomicSchmidtOrthogonalizationInput {
                small_components: bad_small_components.view(),
                ..schmidt.as_input(None)
            },
            sample_schmidt_integral,
        ),
        Err(AtomMathError::MatrixShape { .. })
    ));

    let bad_active_lengths = [6, 4, 3, 5];
    assert!(matches!(
        atomic_schmidt_orthogonalization(
            AtomicSchmidtOrthogonalizationInput {
                active_lengths: &bad_active_lengths,
                ..schmidt.as_input(None)
            },
            sample_schmidt_integral,
        ),
        Err(AtomMathError::ActiveLengthOutOfRange { .. })
    ));

    assert!(matches!(
        atomic_schmidt_orthogonalization(schmidt.as_input(Some(1)), |request| match request {
            AtomicSchmidtIntegralRequest::Projection(_) => Ok(0.0),
            AtomicSchmidtIntegralRequest::Norm(_) => Ok(0.0),
        }),
        Err(AtomMathError::NonPositiveNorm { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(0, -1, 1),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(i32::MIN, -1, 1),
        Err(AtomMathError::InvalidKappa { .. })
    ));
    assert!(matches!(
        atomic_breit_angular_coefficients(1, -1, usize::MAX),
        Err(AtomMathError::BreitRankOutOfRange { .. })
    ));

    let coefficients = Array3::zeros((2, 2, 1));
    let input = AtomicTotalEnergyInput {
        kappas: &[1],
        occupations: &[],
        valence_occupations: &[0.0],
        orbital_energies: &[0.0],
        coulomb_coefficients: coefficients.view(),
    };
    assert!(matches!(
        atomic_total_energy(input, |_| Ok(0.0)),
        Err(AtomMathError::OrbitalTableLengthMismatch { .. })
    ));

    let input = AtomicTotalEnergyInput {
        kappas: &[1],
        occupations: &[1.0],
        valence_occupations: &[0.0],
        orbital_energies: &[0.0],
        coulomb_coefficients: coefficients.view(),
    };
    assert!(matches!(
        atomic_total_energy(input, |_| Ok(Real::NAN)),
        Err(AtomMathError::NonFiniteScalar {
            field: "radial_integral",
            ..
        })
    ));

    let single_overlap = Array2::from_elem((1, 1), 1.0);
    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: Some(0),
        kappas: &[1],
        occupations: &[1.0],
        overlap_integrals: single_overlap.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::HoleOrbitalOutOfRange { .. })
    ));

    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: None,
        kappas: &[1, 1],
        occupations: &[1.0, 1.0],
        overlap_integrals: single_overlap.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::OverlapMatrixShape { .. })
    ));

    let too_many_kappas = [1; 9];
    let too_many_occupations = [1.0; 9];
    let too_many_overlaps = Array2::from_diag(&Array1::from_elem(9, 1.0));
    let input = AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: None,
        kappas: &too_many_kappas,
        occupations: &too_many_occupations,
        overlap_integrals: too_many_overlaps.view(),
    };
    assert!(matches!(
        atomic_overlap_amplitude_reduction(input),
        Err(AtomMathError::KappaGroupTooLarge { .. })
    ));
}

struct SchmidtFixture {
    kappas: Vec<i32>,
    active_lengths: Vec<usize>,
    orbital_powers: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
}

struct DsordfFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    orbital_powers: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
    derivative_large: Array1<Real>,
    derivative_small: Array1<Real>,
    derivative_large_coefficients: Array1<Real>,
    derivative_small_coefficients: Array1<Real>,
}

struct YzktegFixture {
    source: Array1<Real>,
    source_coefficients: Array1<Real>,
    radii: Array1<Real>,
}

struct VldaFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    occupations: Vec<Real>,
    valence_occupations: Vec<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    initial_potential: Array1<Real>,
    initial_development_coefficients: Array1<Real>,
    initial_energy_density: Array1<Real>,
}

struct PotrdfFixture {
    radii: Array1<Real>,
    active_lengths: Vec<usize>,
    kappas: Vec<i32>,
    orbital_powers: Vec<Real>,
    occupations: Vec<Real>,
    shell_markers: Vec<i32>,
    origin_scales: Vec<Real>,
    coulomb_coefficients: Array3<Real>,
    lagrange_parameters: Array1<Real>,
    nuclear_potential: Array1<Real>,
    nuclear_development_coefficients: Array1<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
}

struct SoldirNormFixture {
    radii: Array1<Real>,
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

struct SoldirSolutionNormalizationFixture {
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

struct SoldirSetupFixture {
    radii: Array1<Real>,
    potential: Array1<Real>,
    potential_coefficients: Array1<Real>,
    positive_origin_coefficients: Array1<Real>,
}

struct IntdirFixture {
    radii: Array1<Real>,
    potential: Array1<Real>,
    potential_coefficients: Array1<Real>,
    large_source: Array1<Real>,
    small_source: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
}

impl SoldirNormFixture {
    fn input(
        &self,
        method: i32,
        coefficient_count: usize,
        matching_small_component: Real,
        origin_power: Real,
        active_len: usize,
        matching_index_1based: usize,
    ) -> AtomicDiracNormalizationInput<'_> {
        AtomicDiracNormalizationInput {
            radii: self.radii.view(),
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            method,
            step: 0.05,
            coefficient_count,
            matching_small_component,
            origin_power,
            active_len,
            matching_index_1based,
        }
    }
}

impl SoldirSolutionNormalizationFixture {
    fn input(
        &self,
        norm: Real,
        initial_large_coefficient: Real,
        initial_small_coefficient: Real,
    ) -> AtomicDiracSolutionNormalizationInput<'_> {
        AtomicDiracSolutionNormalizationInput {
            norm,
            initial_large_coefficient,
            initial_small_coefficient,
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            coefficient_count: 4,
            active_len: 7,
        }
    }
}

impl SoldirSetupFixture {
    fn input(
        &self,
        energy: Real,
        method: i32,
        kappa: i32,
        principal_quantum_number: usize,
        negative_origin: bool,
    ) -> AtomicDiracSolverSetupInput<'_> {
        AtomicDiracSolverSetupInput {
            energy,
            origin_power: 1.25,
            initial_large_coefficient: 0.82,
            initial_small_coefficient: -0.006,
            principal_quantum_number,
            kappa,
            speed_of_light: 137.0373,
            method,
            radii: self.radii.view(),
            potential: self.potential.view(),
            potential_coefficients: if negative_origin {
                self.potential_coefficients.view()
            } else {
                self.positive_origin_coefficients.view()
            },
            active_len: 7,
        }
    }
}

impl IntdirFixture {
    fn input(
        &self,
        mode: AtomicDiracIntegrationMode,
        matching_index_1based: usize,
        max_index_1based: usize,
    ) -> AtomicDiracIntegrationInput<'_> {
        AtomicDiracIntegrationInput {
            large_source: self.large_source.view(),
            small_source: self.small_source.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            radii: self.radii.view(),
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            energy: -0.08,
            origin_power: 0.999,
            initial_large_coefficient: 0.85,
            initial_small_coefficient: -0.004,
            asymptotic_large_component: 0.02,
            kappa: -1,
            speed_of_light: 137.0373,
            step: 0.05,
            matching_precision: 1.0e-7,
            coefficient_count: 6,
            active_len: 151,
            mode,
            matching_index_1based,
            max_index_1based,
        }
    }
}

impl DsordfFixture {
    fn input(
        &self,
        kind: AtomicDifferentialIntegralKind,
        power: i32,
        origin_power: Real,
    ) -> AtomicDifferentialIntegralInput<'_> {
        AtomicDifferentialIntegralInput {
            kind,
            power,
            origin_power,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            derivative_large: self.derivative_large.view(),
            derivative_small: self.derivative_small.view(),
            derivative_large_coefficients: self.derivative_large_coefficients.view(),
            derivative_small_coefficients: self.derivative_small_coefficients.view(),
        }
    }

    fn yzkrdf_input(
        &self,
        left_orbital_1based: usize,
        right_orbital_1based: usize,
        angular_momentum: usize,
        large_small: bool,
    ) -> AtomicYkZkExchangeInput<'_> {
        AtomicYkZkExchangeInput {
            left_orbital_1based,
            right_orbital_1based,
            large_small,
            angular_momentum,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }

    fn fdrirk_input<'a>(
        &'a self,
        request: AtomicRadialIntegralRequest,
        kappas: &'a [i32],
        large_small: bool,
        previous_first_factor: Option<AtomicRadialFirstFactorView<'a>>,
    ) -> AtomicRadialIntegralInput<'a> {
        AtomicRadialIntegralInput {
            request,
            large_small,
            previous_first_factor,
            kappas,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

impl YzktegFixture {
    fn input(&self) -> AtomicYkZkTransformInput<'_> {
        AtomicYkZkTransformInput {
            source: self.source.view(),
            source_coefficients: self.source_coefficients.view(),
            radii: self.radii.view(),
            initial_power: 0.65,
            step: 0.05,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 13,
        }
    }

    fn prepared_input(
        &self,
        source_len: usize,
        angular_momentum: usize,
    ) -> AtomicYkZkPreparedSourceInput<'_> {
        AtomicYkZkPreparedSourceInput {
            source: self.source.view(),
            source_coefficients: self.source_coefficients.view(),
            radii: self.radii.view(),
            step: 0.05,
            angular_momentum,
            coefficient_count: 6,
            source_len,
            active_len: 13,
        }
    }
}

impl VldaFixture {
    fn input(
        &self,
        mode: AtomicLocalDensityExchangeMode,
        accumulate_energy_density: bool,
    ) -> AtomicLocalDensityPotentialInput<'_> {
        AtomicLocalDensityPotentialInput {
            mode,
            accumulate_energy_density,
            speed_of_light: 137.035_999,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            occupations: &self.occupations,
            valence_occupations: &self.valence_occupations,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            initial_potential: self.initial_potential.view(),
            initial_development_coefficients: self.initial_development_coefficients.view(),
            initial_energy_density: self.initial_energy_density.view(),
        }
    }
}

impl PotrdfFixture {
    fn input(
        &self,
        include_exchange: bool,
        include_lagrange: bool,
    ) -> AtomicOrbitalPotentialInput<'_> {
        AtomicOrbitalPotentialInput {
            active_orbital_1based: 2,
            include_exchange,
            include_lagrange,
            self_consistent_count: 3,
            speed_of_light: 137.035_999,
            step: 0.05,
            radii: self.radii.view(),
            active_lengths: &self.active_lengths,
            kappas: &self.kappas,
            orbital_powers: &self.orbital_powers,
            occupations: &self.occupations,
            shell_markers: &self.shell_markers,
            origin_scales: &self.origin_scales,
            coulomb_coefficients: self.coulomb_coefficients.view(),
            lagrange_parameters: self.lagrange_parameters.view(),
            nuclear_potential: self.nuclear_potential.view(),
            nuclear_development_coefficients: self.nuclear_development_coefficients.view(),
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

fn sample_dsordf_fixture() -> DsordfFixture {
    sample_atomic_radial_fixture(11)
}

fn sample_yzkrdf_fixture() -> DsordfFixture {
    sample_atomic_radial_fixture(13)
}

fn sample_soldir_norm_fixture() -> SoldirNormFixture {
    SoldirNormFixture {
        radii: Array1::from_shape_fn(251, |row| (-8.8 + 0.05 * row as Real).exp()),
        large_component: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.03 * index + 0.002 * (0.17 * index).sin()
        }),
        small_component: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            -0.014 * index + 0.003 * (0.11 * index).cos()
        }),
        large_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            0.021 * index - 0.0007 * index * index
        }),
        small_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            -0.013 * index + 0.0004 * index * index
        }),
    }
}

fn sample_soldir_solution_normalization_fixture(
    flip_coefficients: bool,
    flip_components: bool,
) -> SoldirSolutionNormalizationFixture {
    let mut large_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as Real;
        0.2 * index + 0.01 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as Real;
        -0.11 * index + 0.003 * index * index
    });
    let mut large_component = Array1::from_shape_fn(9, |row| {
        let index = (row + 1) as Real;
        0.04 * index + 0.001 * index * index
    });
    let small_component = Array1::from_shape_fn(9, |row| {
        let index = (row + 1) as Real;
        -0.03 * index + 0.0005 * index * index
    });

    if flip_coefficients {
        large_coefficients[0] = -large_coefficients[0];
    }
    if flip_components {
        large_component[0] = -large_component[0];
    }

    SoldirSolutionNormalizationFixture {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
    }
}

fn sample_soldir_node_count_component() -> Array1<Real> {
    Array1::from_vec(vec![0.2, 0.1, -0.05, 0.0, 0.0, 0.03, -0.02, -0.01, 0.01])
}

fn sample_soldir_setup_fixture() -> SoldirSetupFixture {
    SoldirSetupFixture {
        radii: Array1::from_shape_fn(7, |row| 0.08 * (0.11 * row as Real).exp()),
        potential: Array1::from_shape_fn(7, |row| {
            let radius = 0.08 * (0.11 * row as Real).exp();
            -0.42 * (-0.30 * radius).exp() + 0.008 * row as Real
        }),
        potential_coefficients: Array1::from_vec(vec![-0.058_378_260_164_777, 0.0006, -0.0003]),
        positive_origin_coefficients: Array1::from_vec(vec![0.021, 0.0006, -0.0003]),
    }
}

fn sample_intdir_fixture() -> IntdirFixture {
    let speed_of_light = 137.0373;
    let step = 0.05;
    let nuclear_charge = 8.0;
    IntdirFixture {
        radii: Array1::from_shape_fn(251, |row| 0.03 * (step * row as Real).exp()),
        potential: Array1::from_shape_fn(251, |row| {
            let radius = 0.03 * (step * row as Real).exp();
            -0.25 * (-0.40 * radius).exp()
        }),
        potential_coefficients: Array1::from_shape_fn(10, |row| {
            if row == 0 {
                -nuclear_charge / speed_of_light
            } else {
                0.0003 * row as Real * (-1.0_f64).powi((row + 1) as i32)
            }
        }),
        large_source: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.001 * (0.17 * index).sin() + 0.0002 * (0.03 * index).cos()
        }),
        small_source: Array1::from_shape_fn(251, |row| {
            let index = (row + 1) as Real;
            0.0007 * (0.11 * index).cos() - 0.0001 * (0.05 * index).sin()
        }),
        large_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            0.0002 * index * (-1.0_f64).powi((row + 1) as i32)
        }),
        small_coefficients: Array1::from_shape_fn(10, |row| {
            let index = (row + 1) as Real;
            -0.00015 * index * (-1.0_f64).powi((row + 1) as i32)
        }),
    }
}

fn sample_vlda_fixture() -> VldaFixture {
    let radial_count = 13;
    let orbital_count = 3;
    VldaFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        occupations: vec![2.0, 1.6, 0.7],
        valence_occupations: vec![1.0, 0.4, 0.2],
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        initial_potential: Array1::from_shape_fn(radial_count, |row| 0.0001 * (row + 1) as Real),
        initial_development_coefficients: Array1::from_shape_fn(6, |row| 0.01 * (row + 1) as Real),
        initial_energy_density: Array1::from_shape_fn(radial_count, |row| {
            0.002 * (row + 1) as Real
        }),
    }
}

fn sample_potrdf_fixture() -> PotrdfFixture {
    let radial_count = 13;
    let orbital_count = 3;
    let coefficient_count = 6;
    PotrdfFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        kappas: vec![-1, 1, 1],
        orbital_powers: (1..=orbital_count)
            .map(|orbital| 0.12 + 0.09 * orbital as Real)
            .collect(),
        occupations: vec![2.0, 1.6, 0.7],
        shell_markers: vec![-1, 1, 1],
        origin_scales: vec![1.05, 0.95, 1.10],
        coulomb_coefficients: Array3::from_shape_fn(
            (orbital_count, orbital_count, 5),
            |(left, right, rank)| {
                0.015 * (left + 1) as Real + 0.011 * (right + 1) as Real + 0.003 * rank as Real
            },
        ),
        lagrange_parameters: Array1::from_shape_fn(3, |row| 0.012 * (row + 1) as Real),
        nuclear_potential: Array1::from_shape_fn(radial_count, |row| {
            -0.2 + 0.001 * (row + 1) as Real
        }),
        nuclear_development_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            -0.03 * (row + 1) as Real
        }),
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        large_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| 0.08 * (row + 1) as Real + 0.015 * (col + 1) as Real,
        ),
        small_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| -0.02 * (row + 1) as Real + 0.01 * (col + 1) as Real,
        ),
    }
}

fn sample_atomic_radial_fixture(radial_count: usize) -> DsordfFixture {
    let orbital_count = 3;
    let coefficient_count = 6;
    DsordfFixture {
        radii: Array1::from_shape_fn(radial_count, |row| (-4.2 + 0.05 * row as Real).exp()),
        active_lengths: vec![9, 11, 7],
        orbital_powers: (1..=orbital_count)
            .map(|orbital| 0.12 + 0.09 * orbital as Real)
            .collect(),
        large_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        }),
        small_components: Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
            let radial = (row + 1) as Real;
            let orbital = (col + 1) as Real;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        }),
        large_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| {
                let coefficient = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                0.08 * coefficient + 0.015 * orbital
            },
        ),
        small_coefficients: Array2::from_shape_fn(
            (coefficient_count, orbital_count),
            |(row, col)| {
                let coefficient = (row + 1) as Real;
                let orbital = (col + 1) as Real;
                -0.02 * coefficient + 0.01 * orbital
            },
        ),
        derivative_large: Array1::from_shape_fn(radial_count, |row| {
            let radial = (row + 1) as Real;
            0.015 * radial - 0.00007 * radial * radial
        }),
        derivative_small: Array1::from_shape_fn(radial_count, |row| {
            let radial = (row + 1) as Real;
            -0.004 * radial + 0.00013 * radial * radial
        }),
        derivative_large_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let coefficient = (row + 1) as Real;
            0.05 * coefficient - 0.003
        }),
        derivative_small_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let coefficient = (row + 1) as Real;
            -0.015 * coefficient + 0.004
        }),
    }
}

fn sample_yzkteg_fixture() -> YzktegFixture {
    let active_len = 13;
    let coefficient_count = 6;
    YzktegFixture {
        source: Array1::from_shape_fn(active_len, |row| {
            let row = (row + 1) as Real;
            0.017 * row + 0.0008 * row * row - 0.00001 * row * row * row
        }),
        source_coefficients: Array1::from_shape_fn(coefficient_count, |row| {
            let row = (row + 1) as Real;
            0.04 * row - 0.0015 * row * row
        }),
        radii: Array1::from_shape_fn(active_len, |row| (-4.2 + 0.05 * row as Real).exp()),
    }
}

impl SchmidtFixture {
    fn as_input(
        &self,
        active_orbital_1based: Option<usize>,
    ) -> AtomicSchmidtOrthogonalizationInput<'_> {
        AtomicSchmidtOrthogonalizationInput {
            active_orbital_1based,
            kappas: &self.kappas,
            active_lengths: &self.active_lengths,
            orbital_powers: &self.orbital_powers,
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

fn sample_schmidt_fixture() -> SchmidtFixture {
    SchmidtFixture {
        kappas: vec![-1, -1, 1, -1],
        active_lengths: vec![3, 4, 3, 5],
        orbital_powers: (1..=4).map(|orbital| 0.1 * orbital as Real).collect(),
        large_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
            0.07 * (row + 1) as Real + 0.11 * (orbital + 1) as Real
        }),
        small_components: Array2::from_shape_fn((5, 4), |(row, orbital)| {
            0.03 * (row + 1) as Real - 0.02 * (orbital + 1) as Real
        }),
        large_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
            0.2 * (row + 1) as Real + 0.05 * (orbital + 1) as Real
        }),
        small_coefficients: Array2::from_shape_fn((4, 4), |(row, orbital)| {
            -0.03 * (row + 1) as Real + 0.04 * (orbital + 1) as Real
        }),
    }
}

fn sample_schmidt_integral(
    request: AtomicSchmidtIntegralRequest<'_>,
) -> Result<Real, AtomMathError> {
    match request {
        AtomicSchmidtIntegralRequest::Projection(request) => Ok(request
            .target_large
            .iter()
            .zip(request.reference_large.iter())
            .map(|(&target, &reference)| target * reference)
            .sum::<Real>()
            + request
                .target_small
                .iter()
                .zip(request.reference_small.iter())
                .map(|(&target, &reference)| target * reference)
                .sum::<Real>()),
        AtomicSchmidtIntegralRequest::Norm(request) => Ok(request
            .target_large
            .iter()
            .map(|&value| value * value)
            .sum::<Real>()
            + request
                .target_small
                .iter()
                .map(|&value| value * value)
                .sum::<Real>()),
    }
}

fn assert_columns_close<const ROWS: usize, const COLUMNS: usize>(
    actual: &Array2<Real>,
    expected_columns: &[[Real; ROWS]; COLUMNS],
    tolerance: Real,
) {
    assert_eq!(actual.nrows(), ROWS);
    assert_eq!(actual.ncols(), COLUMNS);
    for (column, expected_column) in expected_columns.iter().enumerate() {
        for (row, &expected) in expected_column.iter().enumerate() {
            assert_close_with(actual[(row, column)], expected, tolerance);
        }
    }
}

fn sample_s02at_overlaps() -> Array2<Real> {
    let mut overlaps =
        Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as Real);
    for index in 0..6 {
        overlaps[(index, index)] = 1.0;
    }
    overlaps[(0, 1)] = 0.91;
    overlaps[(1, 0)] = 0.91;
    overlaps[(2, 3)] = 0.82;
    overlaps[(3, 2)] = 0.82;
    overlaps
}

fn sample_atomic_radial_integral(
    request: AtomicRadialIntegralRequest,
) -> Result<Real, AtomMathError> {
    Ok(0.0001 * (request.rank + 1) as Real
        + 0.001 * request.first_left as Real
        + 0.0002 * request.first_right as Real
        + 0.00003 * request.second_left as Real
        + 0.000004 * request.second_right as Real)
}

fn sample_atomic_tabrat_integral(
    request: AtomicTabulationIntegralRequest,
) -> Result<Real, AtomMathError> {
    Ok(0.01 * (request.left + 1) as Real
        + 0.02 * (request.right + 1) as Real
        + 0.001 * request.power as Real
        + 0.1)
}
