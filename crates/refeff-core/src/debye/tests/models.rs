use super::{support::*, *};

#[test]
fn morse_einstein_cumulants_match_feff_reference() -> Result<(), DebyeError> {
    let first = morse_einstein_cumulants(0.003, 300.0, 1.0e-5, 400.0)?;
    assert_close(first.first, 1.190_648_842_682_321_3e-8);
    assert_close(first.third, 5.526_344_214_607_83e-11);
    assert_close(first.scaled_thermal_expansion, 5.291_772_49e-6);

    let second = morse_einstein_cumulants(0.0075, 800.0, 2.5e-5, 250.0)?;
    assert_close(second.first, 7.441_554_786_684_262e-8);
    assert_close(second.third, 1.098_357_016_560_439_2e-9);
    assert_close(second.scaled_thermal_expansion, 1.322_943_122_5e-5);

    let negative_alpha = morse_einstein_cumulants(0.0012, 120.0, -7.0e-6, 350.0)?;
    assert_close(negative_alpha.first, -3.333_816_545_419_16e-9);
    assert_close(negative_alpha.third, -3.706_146_208_663_239e-12);
    assert_close(negative_alpha.scaled_thermal_expansion, -3.704_240_743e-6);
    Ok(())
}

#[test]
fn morse_einstein_cumulants_reject_invalid_inputs() {
    assert!(matches!(
        morse_einstein_cumulants(0.0, 300.0, 1.0e-5, 400.0),
        Err(DebyeError::NonPositive { name: "sig2", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, Real::NAN, 1.0e-5, 400.0),
        Err(DebyeError::NonFinite { name: "tk", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, 300.0, Real::INFINITY, 400.0),
        Err(DebyeError::NonFinite { name: "alphat", .. })
    ));
    assert!(matches!(
        morse_einstein_cumulants(0.003, 300.0, 1.0e-5, -1.0),
        Err(DebyeError::NonPositive { name: "thetae", .. })
    ));
}

#[test]
fn thermal_expansion_cumulants_match_feff_reference() -> Result<(), DebyeError> {
    let copper = thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, 2.55)?;
    assert_relative_close(copper.first, -3.563_418_839_026_406e17);
    assert_relative_close(copper.third, -2.138_051_303_415_843_5e15);

    let copper_oxygen = thermal_expansion_cumulants(29, 8, 0.0042, 1.8e-5, 650.0, 1.91)?;
    assert_relative_close(copper_oxygen.first, -7.144_230_125_822_932e17);
    assert_relative_close(copper_oxygen.third, -6.001_153_305_691_263e15);

    let carbon_hydrogen = thermal_expansion_cumulants(6, 1, 0.0015, -6.0e-6, 300.0, 1.09)?;
    assert_relative_close(carbon_hydrogen.first, 7.521_958_969_413_031e14);
    assert_relative_close(carbon_hydrogen.third, 2.256_587_690_823_909e12);
    Ok(())
}

#[test]
fn thermal_expansion_cumulants_reject_invalid_inputs() {
    assert!(matches!(
        thermal_expansion_cumulants(0, 29, 0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::InvalidAtomicNumber { z: 0 })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 140, 0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::InvalidAtomicNumber { z: 140 })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, -0.003, 1.0e-5, 400.0, 2.55),
        Err(DebyeError::NonPositive { name: "sig2", .. })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 0.0, 2.55),
        Err(DebyeError::NonPositive { name: "thetad", .. })
    ));
    assert!(matches!(
        thermal_expansion_cumulants(29, 29, 0.003, 1.0e-5, 400.0, Real::NAN),
        Err(DebyeError::NonFinite { name: "reff", .. })
    ));
}

#[test]
fn debye_correlations_match_feff_reference() -> Result<(), DebyeError> {
    let zero = quantum_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(zero.value, 4.501_999_849_393_054e-3);
    let copper = quantum_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(copper.value, 1.691_640_883_386_128e-3);
    let copper_oxygen = quantum_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
    assert_close(copper_oxygen.value, 7.447_746_368_694_431e-4);

    let classical_zero = classical_debye_correlation(0.0, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(classical_zero.value, 4.293_628_582_101_32e-3);
    let classical_copper = classical_debye_correlation(2.55, 400.0, 300.0, 29, 29, 2.7)?;
    assert_close(classical_copper.value, 1.685_437_153_407_153e-3);
    let classical_copper_oxygen = classical_debye_correlation(1.91, 650.0, 120.0, 29, 8, 2.3)?;
    assert_close(classical_copper_oxygen.value, 6.129_399_740_209_465e-4);
    Ok(())
}

#[test]
fn debye_correlations_reject_invalid_inputs() {
    assert!(matches!(
        quantum_debye_correlation(-1.0, 400.0, 300.0, 29, 29, 2.7),
        Err(DebyeError::Negative { name: "rij", .. })
    ));
    assert!(matches!(
        quantum_debye_correlation(1.0, 400.0, -1.0, 29, 29, 2.7),
        Err(DebyeError::Negative { name: "tk", .. })
    ));
    assert!(matches!(
        classical_debye_correlation(1.0, 400.0, 300.0, 29, 0, 2.7),
        Err(DebyeError::InvalidAtomicNumber { z: 0 })
    ));
}

#[test]
fn debye_waller_factors_match_feff_reference() -> Result<(), DebyeError> {
    let copper_path = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.55, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let copper_atomic_numbers = [29, 29, 29];
    assert_close(
        quantum_debye_waller_factor(
            300.0,
            400.0,
            2.7,
            copper_path.view(),
            &copper_atomic_numbers,
        )?,
        5.620_717_932_013_852e-3,
    );
    assert_close(
        classical_debye_waller_factor(
            300.0,
            400.0,
            2.7,
            copper_path.view(),
            &copper_atomic_numbers,
        )?,
        5.216_382_857_388_334e-3,
    );

    let triangle_path = ndarray::arr2(&[
        [0.0, 0.0, 0.0],
        [1.91, 0.25, 0.10],
        [2.60, 1.40, -0.20],
        [0.0, 0.0, 0.0],
    ]);
    let triangle_atomic_numbers = [29, 8, 29, 29];
    assert_close(
        quantum_debye_waller_factor(
            180.0,
            650.0,
            2.3,
            triangle_path.view(),
            &triangle_atomic_numbers,
        )?,
        2.623_124_881_997_499_5e-3,
    );
    assert_close(
        classical_debye_waller_factor(
            180.0,
            650.0,
            2.3,
            triangle_path.view(),
            &triangle_atomic_numbers,
        )?,
        1.796_449_763_322_294e-3,
    );
    Ok(())
}

#[test]
fn debye_waller_factors_reject_invalid_inputs() {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29, 29]),
        Err(DebyeError::ZeroLengthPathLeg { leg: 1 })
    ));
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, positions.view(), &[29]),
        Err(DebyeError::InvalidAtomicNumberCount { .. })
    ));
    let bad_shape = ndarray::Array2::<Real>::zeros((1, 3));
    assert!(matches!(
        quantum_debye_waller_factor(300.0, 400.0, 2.7, bad_shape.view(), &[29]),
        Err(DebyeError::InvalidPathShape { .. })
    ));
}
