use super::{support::*, *};

#[test]
fn effective_electron_count_matches_feff_qsum_reference() -> Result<(), FullSpectrumError> {
    let omega = array![0.0, 0.1, 0.2, 0.5, 1.0, 1.8];
    let epsilon2 = array![0.0, 0.5, 1.0, 0.25, 0.75, 0.1];

    let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 0.075,
        epsilon2: epsilon2.view(),
        omega: omega.view(),
        active_len: 6,
    })?;

    assert_close(neff, 0.442_098_097_959_400_5, 1.0e-14);
    Ok(())
}

#[test]
fn effective_electron_count_matches_feff_single_point_reference() -> Result<(), FullSpectrumError> {
    let omega = array![0.0, 0.1];
    let epsilon2 = array![1.0, 2.0];

    let neff = full_spectrum_effective_electron_count(FullSpectrumQSumInput {
        number_density: 0.075,
        epsilon2: epsilon2.view(),
        omega: omega.view(),
        active_len: 1,
    })?;

    assert_eq!(neff, 0.0);
    Ok(())
}

#[test]
fn effective_electron_count_rejects_invalid_inputs() {
    let omega = array![0.0, 0.1, 0.2];
    let epsilon2 = array![0.0, 0.5, 1.0];

    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.0,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: omega.view(),
            active_len: 4,
        }),
        Err(FullSpectrumError::ActiveCountOutOfRange {
            field: "epsilon2",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: array![0.0, f64::NAN, 1.0].view(),
            omega: omega.view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon2",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_effective_electron_count(FullSpectrumQSumInput {
            number_density: 0.075,
            epsilon2: epsilon2.view(),
            omega: array![0.0, 0.2, 0.1].view(),
            active_len: 3,
        }),
        Err(FullSpectrumError::DecreasingOmega { row: 2, .. })
    ));
}

#[test]
fn drude_term_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![0.1, 0.2, 0.5];

    let drude = full_spectrum_drude_term(FullSpectrumDrudeInput {
        omega: omega.view(),
        lifetime_seconds: 1.0e-15,
        number_density: 0.075,
    })?;

    assert_eq!(drude.point_count(), 3);
    assert_close(drude.gamma_ev, 0.658, 1.0e-14);
    assert_close(drude.plasma_frequency_ev, 26.417_175_795_207_253, 1.0e-14);
    assert_close(drude.epsilon[0].re, -89.041_328_740_125_08, 1.0e-14);
    assert_close(drude.epsilon[0].im, 21.531_124_059_567_656, 1.0e-14);
    assert_close(drude.epsilon[2].re, -3.761_114_344_763_512, 1.0e-14);
    assert_close(drude.epsilon[2].im, 0.181_895_352_877_477_57, 1.0e-14);
    Ok(())
}

#[test]
fn drude_term_rejects_invalid_inputs() {
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: array![0.1].view(),
            lifetime_seconds: 0.0,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "lifetime_seconds",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: Array1::<Real>::zeros(0).view(),
            lifetime_seconds: 1.0e-15,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::EmptyTable { name: "drude_term" })
    ));
    assert!(matches!(
        full_spectrum_drude_term(FullSpectrumDrudeInput {
            omega: array![0.0].view(),
            lifetime_seconds: 1.0e-15,
            number_density: 0.075,
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
}

#[test]
fn valence_epsilon2_matches_feff_rdval_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![
        5.0 / FEFF_HARTREE_EV,
        10.0 / FEFF_HARTREE_EV,
        15.0 / FEFF_HARTREE_EV,
        25.0 / FEFF_HARTREE_EV,
        40.0 / FEFF_HARTREE_EV,
        50.0 / FEFF_HARTREE_EV,
    ];
    let source_energy_ev = array![10.0, 20.0, 40.0];
    let source_absorption = array![1.0, 3.0, 7.0];

    let epsilon2 = full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
        number_density: 0.075,
        omega: omega.view(),
        source_energy_ev: source_energy_ev.view(),
        source_absorption_angstrom2: source_absorption.view(),
    })?;

    assert_eq!(epsilon2.len(), omega.len());
    assert_close(epsilon2[0], 0.0, 0.0);
    assert_close(epsilon2[1], 0.0, 0.0);
    assert_close(epsilon2[2], 131.219_281_455_964_96, 1.0e-12);
    assert_close(epsilon2[3], 157.463_137_747_157_93, 1.0e-12);
    assert_close(epsilon2[4], 0.0, 0.0);
    assert_close(epsilon2[5], 0.0, 0.0);
    Ok(())
}

#[test]
fn valence_epsilon2_rejects_invalid_inputs() {
    let omega = array![0.1, 0.2, 0.3];
    let source_energy_ev = array![10.0, 20.0, 40.0];
    let source_absorption = array![1.0, 3.0, 7.0];

    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.0,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: array![0.1, 0.0, 0.3].view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: array![10.0, 10.0, 40.0].view(),
            source_absorption_angstrom2: source_absorption.view(),
        }),
        Err(FullSpectrumError::NonIncreasingOmega { row: 1, .. })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: array![1.0, 3.0].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "source_absorption_angstrom2",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_valence_epsilon2(FullSpectrumValenceInput {
            number_density: 0.075,
            omega: omega.view(),
            source_energy_ev: source_energy_ev.view(),
            source_absorption_angstrom2: array![1.0, f64::NAN, 7.0].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "source_absorption_angstrom2",
            row: 1,
            ..
        })
    ));
}

#[test]
fn number_density_matches_feff_rddens_reference_algorithm() -> Result<(), FullSpectrumError> {
    let atomic_numbers = array![29_usize, 8, 29];
    let multiplicities = array![0.01, 2.0, 3.0];
    let norman_radii = array![2.0, 1.5, 2.5];

    let copper_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 29,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;
    let oxygen_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 8,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;
    let missing_density = full_spectrum_number_density(FullSpectrumNumberDensityInput {
        target_atomic_number: 26,
        atomic_numbers: atomic_numbers.view(),
        potential_multiplicities: multiplicities.view(),
        norman_radii: norman_radii.view(),
    })?;

    assert_close(copper_density, 0.013_380_217_262_078_158, 1.0e-16);
    assert_close(oxygen_density, 0.008_890_509_808_689_806, 1.0e-16);
    assert_close(missing_density, 0.0, 0.0);
    Ok(())
}

#[test]
fn number_density_rejects_invalid_inputs() {
    let atomic_numbers = array![29_usize, 8];
    let multiplicities = array![0.01, 2.0];
    let norman_radii = array![2.0, 1.5];

    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 0,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::InvalidAtomicNumber { atomic_number: 0 })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: array![29_usize, 0].view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::InvalidAtomicNumber { atomic_number: 0 })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "potential_multiplicities",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01, f64::NAN].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "potential_multiplicities",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: array![0.01, 0.0].view(),
            norman_radii: norman_radii.view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "potential_multiplicities",
            row: 1,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_number_density(FullSpectrumNumberDensityInput {
            target_atomic_number: 29,
            atomic_numbers: atomic_numbers.view(),
            potential_multiplicities: multiplicities.view(),
            norman_radii: array![2.0, -1.5].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "norman_radii",
            row: 1,
            ..
        })
    ));
}
