#![allow(clippy::excessive_precision)]

use super::*;

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
fn finite_nucleus_data_covers_every_supported_highz_atomic_number() -> Result<(), AtomMathError> {
    for atomic_number in 1..=138 {
        let finite = atomic_nuclear_potential(AtomicNuclearPotentialInput {
            nuclear_charge: atomic_number as Real,
            step: 0.05,
            requested_nucleus_index: -5,
            radial_count: 251,
            coefficient_count: 10,
            first_radius_times_charge: atomic_number as Real * (-8.8_f64).exp(),
        })?;
        assert!(
            finite.nucleus_index > 1,
            "HIGHZ Z={atomic_number} selected a point nucleus"
        );
        assert!(
            finite
                .development_coefficients
                .iter()
                .chain(finite.radii.iter())
                .chain(finite.potential.iter())
                .all(|value| value.is_finite()),
            "HIGHZ Z={atomic_number} produced non-finite nuclear data"
        );

        let boundary = finite.nucleus_index - 1;
        let expected_boundary = -(atomic_number as Real) / finite.radii[boundary];
        assert_close_with(
            finite.potential[boundary],
            expected_boundary,
            expected_boundary.abs() * 2.0e-15,
        );
    }
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
