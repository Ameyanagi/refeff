use ndarray::{Array1, Array2};

use crate::{Complex, Real};

use super::{
    FovrgAngularCoefficientsInput, FovrgC3DerivativeInput, FovrgDiracSolverInput, FovrgError,
    FovrgExchangePotentialInput, FovrgFlatPotentialInput, FovrgInitialPhotoelectronInput,
    FovrgInwardSolutionInput, FovrgNuclearPotentialInput, FovrgOrbitalSetupInput,
    FovrgOrthogonalizationInput, FovrgOutgoingSolutionInput, FovrgOutwardIntegrationInput,
    FovrgOverlapIntegralInput, FovrgPotentialDevelopmentInput, FovrgYkZkExchangeInput,
    FovrgYkZkTransformInput, fovrg_angular_coefficients, fovrg_c3_derivative,
    fovrg_complex_real_product_coefficient, fovrg_dirac_solver, fovrg_exchange_potential,
    fovrg_flat_potential_propagate, fovrg_initial_photoelectron, fovrg_inward_solution,
    fovrg_nuclear_potential, fovrg_orbital_setup, fovrg_outgoing_solution, fovrg_outward_integrate,
    fovrg_overlap_integral, fovrg_potential_development, fovrg_real_product_coefficient,
    fovrg_schmidt_orthogonalize, fovrg_yk_zk_exchange, fovrg_yk_zk_transform,
};

#[test]
fn c3_derivative_matches_feff_diff_reference() -> Result<(), FovrgError> {
    let (potential, radii) = diff_reference_inputs(10);

    let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
        potential: potential.view(),
        radii: radii.view(),
        kappa: -2,
        speed_of_light: 137.035_999_084,
        delta: 0.0375,
        active_len: 10,
    })?;

    let expected = [
        (-0.011_975_827_006_405_27, -0.011_279_195_671_167_455),
        (-0.016_505_394_195_758_99, -0.008_884_114_730_822_418),
        (-0.020_242_542_448_345_43, -0.005_647_908_958_998_54),
        (-0.022_839_291_155_546_27, -0.001_659_964_058_354_706_8),
        (-0.024_047_315_082_090_202, 0.002_950_607_669_371_263_3),
        (-0.023_683_648_659_231_31, 0.008_014_885_042_325_136),
        (-0.021_663_526_338_827_583, 0.013_330_188_602_550_464),
        (-0.018_012_853_921_219_218, 0.018_667_556_473_840_063),
        (-0.012_457_714_462_626_513, 0.023_984_332_127_499_31),
        (-0.007_300_598_102_380_937, 0.028_056_048_903_698_883),
    ];
    for (actual, (expected_re, expected_im)) in derivative.iter().zip(expected) {
        assert_complex_close(*actual, expected_re, expected_im, 1.0e-13);
    }
    Ok(())
}

#[test]
fn c3_derivative_rejects_invalid_inputs() {
    let (potential, radii) = diff_reference_inputs(8);

    assert!(matches!(
        fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 7,
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 9,
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "potential",
            ..
        })
    ));
    assert!(matches!(
        fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0,
            active_len: 8,
        }),
        Err(FovrgError::ZeroInput { name: "delta" })
    ));

    let mut bad_radii = radii.clone();
    bad_radii[3] = 0.0;
    assert!(matches!(
        fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: potential.view(),
            radii: bad_radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 8,
        }),
        Err(FovrgError::NonPositiveRadius { row: 3, .. })
    ));

    let mut bad_potential = potential.clone();
    bad_potential[2] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        fovrg_c3_derivative(FovrgC3DerivativeInput {
            potential: bad_potential.view(),
            radii: radii.view(),
            kappa: -2,
            speed_of_light: 137.035_999_084,
            delta: 0.0375,
            active_len: 8,
        }),
        Err(FovrgError::NonFinitePotential { row: 2, .. })
    ));
}

#[test]
fn polynomial_product_coefficients_match_feff_aprd_reference() -> Result<(), FovrgError> {
    let (real_left, real_right, complex_left) = aprd_reference_inputs(10);

    assert_close(
        fovrg_real_product_coefficient(real_left.view(), real_right.view(), 4)?,
        0.611_437_708_836_968_1,
        1.0e-14,
    );
    assert_close(
        fovrg_real_product_coefficient(real_left.view(), real_right.view(), 7)?,
        1.688_549_807_000_237_2,
        1.0e-14,
    );
    assert_complex_close(
        fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 4)?,
        0.615_721_272_049_818_1,
        0.159_539_410_440_073_47,
        1.0e-14,
    );
    assert_complex_close(
        fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 7)?,
        1.660_658_325_254_387,
        0.615_717_443_886_918,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn polynomial_product_coefficients_reject_invalid_inputs() {
    let (real_left, real_right, complex_left) = aprd_reference_inputs(10);

    assert!(matches!(
        fovrg_real_product_coefficient(real_left.view(), real_right.view(), 0),
        Err(FovrgError::CountTooSmall {
            name: "coefficient_count",
            ..
        })
    ));
    assert!(matches!(
        fovrg_real_product_coefficient(real_left.view(), real_right.view(), 11),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "left_coefficients",
            ..
        })
    ));
    assert!(matches!(
        fovrg_complex_real_product_coefficient(complex_left.view(), real_right.view(), 11),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "complex_coefficients",
            ..
        })
    ));

    let mut bad_real_right = real_right.clone();
    bad_real_right[2] = Real::NAN;
    assert!(matches!(
        fovrg_real_product_coefficient(real_left.view(), bad_real_right.view(), 4),
        Err(FovrgError::NonFiniteRealInput {
            name: "right_coefficients",
            row: 2,
            ..
        })
    ));

    let mut bad_complex_left = complex_left.clone();
    bad_complex_left[1] = Complex::new(0.0, Real::NAN);
    assert!(matches!(
        fovrg_complex_real_product_coefficient(bad_complex_left.view(), real_right.view(), 4),
        Err(FovrgError::NonFiniteComplexInput {
            name: "complex_coefficients",
            row: 1,
            ..
        })
    ));
}

#[test]
fn angular_coefficients_match_feff_muatcc_reference() -> Result<(), FovrgError> {
    let (electron_counts, valence_counts, kappa) = muatcc_reference_inputs();

    let target_negative = fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
        electron_counts: electron_counts.view(),
        valence_counts: valence_counts.view(),
        kappa: kappa.view(),
        target_kappa: -2,
        bound_orbital_count: 5,
    })?;
    let expected_negative = [
        [0.333_333_333_333_333_54, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.625_000_000_000_000_2,
            0.125_000_000_000_000_08,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.016_666_666_666_666_684,
            0.064_285_714_285_714_21,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.299_999_999_999_999_9,
            0.085_714_285_714_285_62,
            0.0,
            0.0,
            0.0,
        ],
    ];
    assert_real_matrix_close(&target_negative, &expected_negative, 1.0e-14);

    let target_positive = fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
        electron_counts: electron_counts.view(),
        valence_counts: valence_counts.view(),
        kappa: kappa.view(),
        target_kappa: 3,
        bound_orbital_count: 5,
    })?;
    let expected_positive = [
        [0.142_857_142_857_142_74, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.035_714_285_714_285_67,
            0.119_047_619_047_618_57,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.099_999_999_999_999_94,
            0.028_571_428_571_428_54,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.014_285_714_285_714_28,
            0.038_095_238_095_237_96,
            0.108_225_108_225_107_97,
            0.0,
            0.0,
        ],
    ];
    assert_real_matrix_close(&target_positive, &expected_positive, 1.0e-14);

    Ok(())
}

#[test]
fn angular_coefficients_reject_invalid_inputs() {
    let (electron_counts, valence_counts, kappa) = muatcc_reference_inputs();

    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: -2,
            bound_orbital_count: 0,
        }),
        Err(FovrgError::CountTooSmall {
            name: "bound_orbital_count",
            ..
        })
    ));
    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: 0,
            bound_orbital_count: 5,
        }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "target_kappa",
            row: 0,
            ..
        })
    ));

    let mut bad_kappa = kappa.clone();
    bad_kappa[1] = 0;
    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: bad_kappa.view(),
            target_kappa: -2,
            bound_orbital_count: 5,
        }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "kappa",
            row: 1,
            ..
        })
    ));

    let mut bad_electron_counts = electron_counts.clone();
    bad_electron_counts[3] = Real::NAN;
    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: bad_electron_counts.view(),
            valence_counts: valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: -2,
            bound_orbital_count: 5,
        }),
        Err(FovrgError::NonFiniteRealInput {
            name: "electron_counts",
            row: 3,
            ..
        })
    ));

    let mut bad_valence_counts = valence_counts.clone();
    bad_valence_counts[2] = Real::NAN;
    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: electron_counts.view(),
            valence_counts: bad_valence_counts.view(),
            kappa: kappa.view(),
            target_kappa: -2,
            bound_orbital_count: 5,
        }),
        Err(FovrgError::NonFiniteRealInput {
            name: "valence_counts",
            row: 2,
            ..
        })
    ));

    assert!(matches!(
        fovrg_angular_coefficients(FovrgAngularCoefficientsInput {
            electron_counts: Array1::from_vec(vec![1.0]).view(),
            valence_counts: Array1::from_vec(vec![0.0]).view(),
            kappa: Array1::from_vec(vec![-6]).view(),
            target_kappa: -6,
            bound_orbital_count: 1,
        }),
        Err(FovrgError::CountTooLarge {
            name: "angular_coefficient_slots",
            actual: 6,
            maximum: 5,
        })
    ));
}

#[test]
fn flat_potential_propagation_matches_feff_flatv_reference() -> Result<(), FovrgError> {
    let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
        start_radius: 0.8,
        end_radius: 1.35,
        large_component: Complex::new(0.32, -0.11),
        small_component: Complex::new(-0.08, 0.045),
        energy: Complex::new(0.85, 0.12),
        average_potential: Complex::new(-0.18, 0.025),
        kappa: -2,
    })?;
    assert_complex_close(
        propagated.large_component,
        -11.083_037_894_089_62,
        6.535_303_549_398_971_5,
        1.0e-12,
    );
    assert_complex_close(
        propagated.small_component,
        -0.009_973_201_918_406_406,
        0.007_263_491_015_424_047,
        1.0e-12,
    );

    let propagated = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
        start_radius: 1.2,
        end_radius: 0.9,
        large_component: Complex::new(-0.14, 0.27),
        small_component: Complex::new(0.19, -0.06),
        energy: Complex::new(1.6, -0.05),
        average_potential: Complex::new(0.2, 0.01),
        kappa: 3,
    })?;
    assert_complex_close(
        propagated.large_component,
        -17.939_760_805_034_215,
        6.125_209_917_887_357,
        1.0e-12,
    );
    assert_complex_close(
        propagated.small_component,
        0.060_863_298_623_451_69,
        -0.017_588_891_061_652_855,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn flat_potential_propagation_rejects_invalid_inputs() {
    let input = FovrgFlatPotentialInput {
        start_radius: 0.8,
        end_radius: 1.35,
        large_component: Complex::new(0.32, -0.11),
        small_component: Complex::new(-0.08, 0.045),
        energy: Complex::new(0.85, 0.12),
        average_potential: Complex::new(-0.18, 0.025),
        kappa: -2,
    };

    assert!(matches!(
        fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            start_radius: 0.0,
            ..input
        }),
        Err(FovrgError::NonPositiveInput {
            name: "start_radius",
            ..
        })
    ));
    assert!(matches!(
        fovrg_flat_potential_propagate(FovrgFlatPotentialInput { kappa: 0, ..input }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "kappa",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            energy: Complex::new(Real::NAN, 0.0),
            ..input
        }),
        Err(FovrgError::NonFiniteComplexInput {
            name: "energy",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
            energy: Complex::new(0.85, 0.12),
            average_potential: Complex::new(0.85, 0.12),
            ..input
        }),
        Err(FovrgError::ZeroDenominator {
            name: "flat_potential_factor"
        })
    ));
}

#[test]
fn outward_integration_matches_feff_intout_reference() -> Result<(), FovrgError> {
    let tolerance = 2.0e-5;
    let case1 = intout_reference_inputs(1);
    let integrated = fovrg_outward_integrate(case1.to_input())?;
    assert_eq!(integrated.difficult_iterations, 0);
    assert_complex_close(
        integrated.large_component[1],
        0.017_463_805_053_776_79,
        -0.003_797_490_517_828_303_4,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[1],
        -0.008_406_932_541_754_055,
        0.003_782_923_630_515_854,
        tolerance,
    );
    assert_complex_close(
        integrated.large_component[5],
        -0.066_689_204_945_806_59,
        0.035_296_711_329_772_11,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[5],
        -0.006_557_738_302_850_965,
        0.002_754_941_000_224_632,
        tolerance,
    );
    assert_complex_close(
        integrated.large_component[11],
        -0.233_694_640_455_667_15,
        0.106_985_257_008_756_33,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[11],
        -0.002_738_717_599_426_290_6,
        0.000_956_279_014_355_418_1,
        tolerance,
    );
    assert_complex_close(integrated.large_component[12], 0.0, 0.0, 0.0);
    assert_complex_close(integrated.small_component[12], 0.0, 0.0, 0.0);

    let case2 = intout_reference_inputs(2);
    let integrated = fovrg_outward_integrate(case2.to_input())?;
    assert_eq!(integrated.difficult_iterations, 0);
    assert_complex_close(
        integrated.large_component[1],
        0.009_943_048_888_859_825,
        0.010_956_870_736_304_185,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[1],
        0.012_429_931_740_178_783,
        -0.006_764_974_141_679_289,
        tolerance,
    );
    assert_complex_close(
        integrated.large_component[12],
        0.586_154_840_562_188_5,
        -0.333_526_150_681_263_4,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[12],
        0.049_643_797_589_848_5,
        -0.028_715_924_595_140_7,
        tolerance,
    );
    assert_complex_close(integrated.large_component[13], 0.0, 0.0, 0.0);
    assert_complex_close(integrated.small_component[13], 0.0, 0.0, 0.0);

    let case3 = intout_reference_inputs(3);
    let integrated = fovrg_outward_integrate(case3.to_input())?;
    assert_eq!(integrated.difficult_iterations, 0);
    assert_complex_close(integrated.large_component[0], 0.0, 0.0, 0.0);
    assert_complex_close(integrated.small_component[0], 0.0, 0.0, 0.0);
    assert_complex_close(integrated.large_component[3], 0.026, 0.014, 1.0e-15);
    assert_complex_close(integrated.small_component[3], -0.008, 0.017, 1.0e-15);
    assert_complex_close(
        integrated.large_component[4],
        0.005_997_786_087_037_459,
        0.058_938_915_396_690_67,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[4],
        -0.007_911_306_018_542_903,
        0.016_250_490_408_196_89,
        tolerance,
    );
    assert_complex_close(
        integrated.large_component[10],
        -0.136_499_070_289_096_64,
        0.363_725_975_734_245_37,
        tolerance,
    );
    assert_complex_close(
        integrated.small_component[10],
        -0.005_440_281_025_366_89,
        0.011_164_144_227_578_115,
        tolerance,
    );
    assert_complex_close(integrated.large_component[11], 0.0, 0.0, 0.0);
    assert_complex_close(integrated.small_component[11], 0.0, 0.0, 0.0);

    Ok(())
}

#[test]
fn outward_integration_rejects_invalid_inputs() {
    let mut input = intout_reference_inputs(1);

    assert!(matches!(
        fovrg_outward_integrate(FovrgOutwardIntegrationInput {
            active_len: 0,
            ..input.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));

    assert!(matches!(
        fovrg_outward_integrate(FovrgOutwardIntegrationInput {
            start_index: 5,
            last_index: 4,
            ..input.to_input()
        }),
        Err(FovrgError::InvalidRange {
            name: "outward_integration",
            ..
        })
    ));

    assert!(matches!(
        fovrg_outward_integrate(FovrgOutwardIntegrationInput {
            kappa: 0,
            ..input.to_input()
        }),
        Err(FovrgError::InvalidQuantumNumber { name: "kappa", .. })
    ));

    assert!(matches!(
        fovrg_outward_integrate(FovrgOutwardIntegrationInput {
            step: 0.0,
            ..input.to_input()
        }),
        Err(FovrgError::ZeroInput { name: "step" })
    ));

    input.potential[2] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_outward_integrate(input.to_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "potential",
            row: 2,
            ..
        })
    ));
}

#[test]
fn outgoing_solution_matches_feff_solout_reference() -> Result<(), FovrgError> {
    let tolerance = 5.0e-5;

    let case1 = solout_reference_inputs(1);
    let solution = fovrg_outgoing_solution(case1.to_input())?;
    assert_eq!(solution.difficult_iterations, 0);
    assert_complex_close(solution.large_coefficients[0], 0.85, -0.13, 1.0e-14);
    assert_complex_close(
        solution.small_coefficients[0],
        -0.044_826_720_241_084_875,
        0.006_855_851_330_989_452,
        1.0e-13,
    );
    assert_complex_close(
        solution.large_coefficients[1],
        -12.399_643_967_721_534,
        1.897_735_483_431_073_6,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[5],
        18.506_377_176_380_628,
        -3.169_720_164_560_293,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[6],
        -0.013_570_707_559_336_159,
        0.004_971_932_036_426_257,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[6],
        -0.000_934_704_975_260_855_8,
        0.000_196_765_336_633_890_28,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[11],
        -0.036_169_233_625_015_3,
        0.010_896_837_643_968_734,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[11],
        -0.000_538_567_588_404_036_5,
        0.000_111_061_211_719_815_06,
        tolerance,
    );
    assert_complex_close(solution.large_component[12], 0.0, 0.0, 0.0);

    let case2 = solout_reference_inputs(2);
    let solution = fovrg_outgoing_solution(case2.to_input())?;
    assert_complex_close(solution.large_coefficients[0], -0.72, 0.21, 1.0e-14);
    assert_complex_close(
        solution.small_coefficients[0],
        0.000_037_001_955_937_810_44,
        -0.000_013_870_807_763_694_648,
        tolerance,
    );
    assert_complex_close(
        solution.large_coefficients[3],
        0.008_048_689_860_581_586,
        -0.004_087_598_669_440_412,
        1.0e-14,
    );
    assert_complex_close(
        solution.large_component[12],
        -0.046_463_364_356_391_396,
        0.013_612_530_210_438_293,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[12],
        -0.003_952_067_939_391_788,
        0.001_101_615_849_936_607_4,
        tolerance,
    );
    assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

    let case3 = solout_reference_inputs(3);
    let solution = fovrg_outgoing_solution(case3.to_input())?;
    assert_complex_close(solution.large_coefficients[0], 0.64, 0.08, 1.0e-14);
    assert_complex_close(
        solution.small_coefficients[0],
        -0.124_444_435_795_557_43,
        -0.015_555_554_474_444_679,
        tolerance,
    );
    assert_complex_close(
        solution.large_coefficients[2],
        710_464.572_458_431_2,
        88_808.071_557_303_9,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        4_432.600_877_072_657,
        554.075_109_634_082_1,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[0],
        -488.363_646_550_350_4,
        -61.045_455_818_793_8,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[10],
        -6_977.844_764_917_198,
        -863.171_547_536_417_2,
        tolerance,
    );
    assert_complex_close(solution.large_component[11], 0.0, 0.0, 0.0);

    Ok(())
}

#[test]
fn outgoing_solution_rejects_invalid_inputs() {
    let mut input = solout_reference_inputs(1);

    assert!(matches!(
        fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
            active_len: 0,
            ..input.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));

    assert!(matches!(
        fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
            coefficient_count: 0,
            ..input.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "coefficient_count",
            ..
        })
    ));

    assert!(matches!(
        fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
            radial_match_index: 14,
            wkb_index: 13,
            last_index: 12,
            ..input.to_input()
        }),
        Err(FovrgError::InvalidRange {
            name: "outgoing_solution",
            ..
        })
    ));

    let case2 = solout_reference_inputs(2);
    assert!(matches!(
        fovrg_outgoing_solution(FovrgOutgoingSolutionInput {
            coefficient_count: 2,
            ..case2.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "coefficient_count",
            ..
        })
    ));

    input.large_exchange_coefficients[1] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_outgoing_solution(input.to_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "large_exchange_coefficients",
            row: 1,
            ..
        })
    ));
}

#[test]
fn inward_solution_matches_feff_solin_reference() -> Result<(), FovrgError> {
    let tolerance = 5.0e-5;

    let case1 = solin_reference_inputs(1);
    let solution = fovrg_inward_solution(case1.to_input())?;
    assert_eq!(solution.difficult_iterations, 0);
    assert_complex_close(
        solution.large_coefficients[0],
        13.035_518_197_636_561,
        0.349_850_489_417_380_23,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[0],
        -0.628_236_070_374_292_4,
        0.041_526_389_085_050_74,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        0.435_590_505_925_380_57,
        0.011_690_486_666_743_206,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[0],
        -0.020_992_925_911_033_328,
        0.001_387_631_895_914_275,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[11],
        0.333_382_471_777_223,
        0.079_399_312_578_217_63,
        tolerance,
    );
    assert_complex_close(solution.large_component[12], 0.0, 0.0, 0.0);
    assert_complex_close(solution.large_coefficients[1], 0.0, 0.0, 0.0);

    let case2 = solin_reference_inputs(2);
    let solution = fovrg_inward_solution(case2.to_input())?;
    assert_complex_close(
        solution.large_coefficients[0],
        3_881.336_079_768_998_4,
        -425.269_741_140_675_76,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[0],
        17.080_439_990_261_297,
        -1.894_813_979_341_339,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        21.686_056_118_456_456,
        -2.376_095_056_526_779,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[0],
        0.095_432_957_245_686_26,
        -0.010_586_829_237_543_801,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[9],
        6.955_715_550_949_966,
        -0.740_042_518_825_175_1,
        tolerance,
    );
    assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

    let case3 = solin_reference_inputs(3);
    let solution = fovrg_inward_solution(case3.to_input())?;
    assert_complex_close(
        solution.large_coefficients[0],
        1.010_225_566_356_747_7,
        0.771_472_351_197_525_2,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[0],
        -0.021_374_479_295_957_21,
        0.001_988_824_772_918_297,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        0.193_088_461_728_290_2,
        0.147_454_602_733_775_3,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[7],
        -0.005_174_328_864_862_752,
        -0.000_796_579_407_773_487_3,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[12],
        0.109_417_857_489_330_1,
        0.230_010_707_501_99,
        tolerance,
    );
    assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);

    Ok(())
}

#[test]
fn inward_solution_rejects_invalid_inputs() {
    let mut input = solin_reference_inputs(1);

    assert!(matches!(
        fovrg_inward_solution(FovrgInwardSolutionInput {
            active_len: 0,
            ..input.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));

    assert!(matches!(
        fovrg_inward_solution(FovrgInwardSolutionInput {
            radial_match_index: 12,
            last_index: 11,
            ..input.to_input()
        }),
        Err(FovrgError::InvalidRange {
            name: "inward_solution",
            ..
        })
    ));

    assert!(matches!(
        fovrg_inward_solution(FovrgInwardSolutionInput {
            last_index: 8,
            ..input.to_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "inward_history_rows",
            ..
        })
    ));

    assert!(matches!(
        fovrg_inward_solution(FovrgInwardSolutionInput {
            kappa: 0,
            ..input.to_input()
        }),
        Err(FovrgError::InvalidQuantumNumber { name: "kappa", .. })
    ));

    input.potential[2] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_inward_solution(input.to_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "potential",
            row: 2,
            ..
        })
    ));
}

#[test]
fn initial_photoelectron_matches_feff_wfirdc_reference() -> Result<(), FovrgError> {
    let tolerance = 5.0e-5;

    let case1 = wfirdc_reference_inputs(1);
    let solution = fovrg_initial_photoelectron(case1.to_input())?;
    assert_eq!(solution.retained_len, 15);
    assert_eq!(solution.target_last_index, 11);
    assert_eq!(solution.orbital_lengths[2], 12);
    assert_close(solution.origin_powers[2], 1.988_772_601_665_248_3, 1.0e-13);
    assert_close(solution.normalization[2], 1.103_846_730_630_056, 1.0e-8);
    assert_complex_close(solution.large_coefficients[0], 1.0, 0.0, 1.0e-13);
    assert_complex_close(
        solution.small_coefficients[0],
        -0.053_054_219_097_202_746,
        0.0,
        1.0e-13,
    );
    assert_complex_close(
        solution.large_coefficients[1],
        119.768_469_441_685_45,
        -0.000_010_615_814_710_931_2,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[2],
        33_498.897_252_995_04,
        -0.005_668_144_020_686_9,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        0.000_000_025_445_236_858_674_065,
        -0.000_000_000_000_000_024_637_979_365_338_104,
        1.0e-10,
    );
    assert_complex_close(
        solution.small_component[0],
        -0.000_000_000_266_838_905_572_576_97,
        -0.000_000_000_000_000_085_595_651_872_462_98,
        1.0e-10,
    );
    assert_complex_close(
        solution.large_component[11],
        0.000_000_068_471_452_282_948_73,
        -0.000_000_000_000_000_079_393_996_781_681_8,
        1.0e-10,
    );
    assert_complex_close(solution.large_component[12], 0.0, 0.0, 0.0);

    let case2 = wfirdc_reference_inputs(2);
    let solution = fovrg_initial_photoelectron(case2.to_input())?;
    assert_eq!(solution.retained_len, 15);
    assert_eq!(solution.target_last_index, 12);
    assert_eq!(solution.orbital_lengths[2], 13);
    assert_close(solution.origin_powers[2], -0.977_351_759_160_620_8, 1.0e-13);
    assert_close(solution.normalization[2], 0.819_300_359_324_134_3, 1.0e-8);
    assert_complex_close(
        solution.large_coefficients[0],
        0.000_239_564_127_222_562_78,
        -0.000_017_857_911_377_344_37,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[0],
        -0.005_798_875_319_195_739,
        0.000_432_945_475_533_893_36,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        1.302_136_302_585_460_4,
        -0.097_065_595_598_087_33,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[0],
        -31.519_435_546_694_215,
        2.353_248_907_792_702,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[12],
        -18.367_896_149_255_156,
        1.371_351_666_142_547,
        tolerance,
    );
    assert_complex_close(solution.large_component[13], 0.0, 0.0, 0.0);
    Ok(())
}

#[test]
fn initial_photoelectron_rejects_invalid_inputs() {
    let mut input = wfirdc_reference_inputs(1);
    input.kappa[2] = 0;
    assert!(matches!(
        fovrg_initial_photoelectron(input.to_input()),
        Err(FovrgError::InvalidQuantumNumber {
            name: "kappa",
            row: 2,
            ..
        })
    ));

    let mut input = wfirdc_reference_inputs(1);
    input.radial_match_index = 14;
    assert!(matches!(
        fovrg_initial_photoelectron(input.to_input()),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            ..
        })
    ));

    let mut input = wfirdc_reference_inputs(1);
    input.orbital_lengths[2] = 0;
    assert!(matches!(
        fovrg_initial_photoelectron(input.to_input()),
        Err(FovrgError::CountTooSmall {
            name: "target_orbital_length",
            ..
        })
    ));

    let mut input = wfirdc_reference_inputs(1);
    input.exchange_correlation_potential[3] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_initial_photoelectron(input.to_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "exchange_correlation_potential",
            row: 3,
            ..
        })
    ));
}

#[test]
fn dirac_solver_matches_feff_dfovrg_reference() -> Result<(), FovrgError> {
    let tolerance = 5.0e-5;

    let regular = dfovrg_reference_inputs(false);
    let solution = fovrg_dirac_solver(regular.to_input())?;
    assert_eq!(solution.active_len, 29);
    assert_eq!(solution.retained_len, 25);
    assert_eq!(solution.wkb_index, 28);
    assert_eq!(solution.target_last_index, 15);
    assert_eq!(solution.iteration_count, 2);
    assert_eq!(solution.orbital_lengths[3], 16);
    assert_close(solution.origin_powers[3], 1.988_772_601_665_248_3, 1.0e-13);
    assert_close(solution.normalization[3], 1.103_846_730_630_056, 1.0e-8);
    assert_complex_close(
        solution.potential_coefficients[1],
        1_403.946_174_404_389_7,
        0.000_001_946_414_857_041_34,
        tolerance,
    );
    assert_complex_close(
        solution.muffin_tin_large_component,
        1.842_027_614_533_388_3,
        -0.017_263_266_231_816_614,
        tolerance,
    );
    assert_complex_close(
        solution.muffin_tin_small_component,
        -0.001_918_179_398_245_837_9,
        -0.000_076_212_881_816_988_88,
        tolerance,
    );
    assert_complex_close(solution.large_coefficients[0], 1.0, 0.0, 1.0e-13);
    assert_complex_close(
        solution.small_coefficients[0],
        -0.053_054_219_097_202_746,
        0.0,
        1.0e-13,
    );
    assert_complex_close(
        solution.large_coefficients[1],
        119.768_559_495_787_56,
        -0.000_013_779_152_967_198_376,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[2],
        33_498.945_342_645_8,
        -0.007_357_163_254_477_731,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[0],
        0.000_000_025_445_237_067_646_95,
        -0.000_000_000_000_000_031_979_676_075_592_074,
        1.0e-10,
    );
    assert_complex_close(
        solution.small_component[9],
        0.000_000_008_488_451_233_913_252,
        -0.000_000_000_020_539_698_699_887_03,
        1.0e-10,
    );
    assert_complex_close(
        solution.large_component[15],
        0.018_402_062_384_959_8,
        -0.000_001_336_412_443_131_888_3,
        tolerance,
    );
    assert_complex_close(solution.large_component[16], 0.0, 0.0, 0.0);
    assert_complex_close(
        solution.exchange_correlation_potential[10],
        -0.094,
        -0.001_928_388_969_284_731_3,
        1.0e-13,
    );
    assert_complex_close(
        solution.valence_exchange_correlation_potential[10],
        -0.094,
        -0.001_928_388_969_284_731_3,
        1.0e-13,
    );

    let irregular = dfovrg_reference_inputs(true);
    let solution = fovrg_dirac_solver(irregular.to_input())?;
    assert_eq!(solution.active_len, 29);
    assert_eq!(solution.retained_len, 25);
    assert_eq!(solution.wkb_index, 28);
    assert_eq!(solution.target_last_index, 16);
    assert_eq!(solution.iteration_count, 0);
    assert_eq!(solution.orbital_lengths[3], 17);
    assert_close(solution.origin_powers[3], -0.977_351_759_160_620_8, 1.0e-13);
    assert_close(solution.normalization[3], 0.819_300_359_324_134_3, 1.0e-8);
    assert_complex_close(solution.muffin_tin_large_component, 0.48, 0.06, 1.0e-13);
    assert_complex_close(solution.muffin_tin_small_component, 0.018, -0.009, 1.0e-13);
    assert_complex_close(
        solution.large_coefficients[0],
        0.000_224_089_936_316_388_8,
        -0.000_012_323_091_642_224_378,
        tolerance,
    );
    assert_complex_close(
        solution.small_coefficients[0],
        -0.005_424_255_616_431_365,
        0.000_298_963_577_133_433_4,
        tolerance,
    );
    assert_complex_close(solution.large_coefficients[2], 0.0, 0.0, 0.0);
    assert_complex_close(
        solution.large_component[0],
        1.218_027_275_221_161_4,
        -0.066_981_418_184_201_97,
        tolerance,
    );
    assert_complex_close(
        solution.small_component[9],
        -0.513_654_790_320_891_5,
        0.028_309_148_435_548_014,
        tolerance,
    );
    assert_complex_close(
        solution.large_component[16],
        1.201_407_383_643_303_7,
        0.132_939_846_114_703_22,
        tolerance,
    );
    assert_complex_close(solution.large_component[17], 0.0, 0.0, 0.0);
    assert_complex_close(
        solution.valence_exchange_correlation_potential[11],
        -0.094,
        -0.001_928_388_969_284_731_3,
        1.0e-13,
    );
    Ok(())
}

#[test]
fn dirac_solver_rejects_invalid_inputs() {
    let mut input = dfovrg_reference_inputs(false);
    input.target_kappa = 0;
    assert!(matches!(
        fovrg_dirac_solver(input.to_input()),
        Err(FovrgError::InvalidQuantumNumber {
            name: "target_kappa",
            ..
        })
    ));

    let mut input = dfovrg_reference_inputs(false);
    input.radial_match_index = 28;
    assert!(matches!(
        fovrg_dirac_solver(input.to_input()),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_match_index",
            ..
        })
    ));

    let mut input = dfovrg_reference_inputs(false);
    input.exchange_correlation_potential[2] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_dirac_solver(input.to_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "exchange_correlation_potential",
            row: 2,
            ..
        })
    ));

    let mut input = dfovrg_reference_inputs(false);
    input.bound_large_components[(3, 1)] = Real::NAN;
    assert!(matches!(
        fovrg_dirac_solver(input.to_input()),
        Err(FovrgError::NonFiniteRealInput {
            name: "bound_large_components",
            row: 3,
            ..
        })
    ));
}

#[test]
fn orbital_setup_matches_feff_inmuac_bookkeeping() -> Result<(), FovrgError> {
    let input = dfovrg_reference_inputs(false);
    let setup = fovrg_orbital_setup(FovrgOrbitalSetupInput {
        bound_large_components: input.bound_large_components.view(),
        bound_small_components: input.bound_small_components.view(),
        electron_counts: input.electron_counts.view(),
        valence_counts: input.valence_counts.view(),
        kappa: input.kappa.view(),
        target_kappa: -2,
        active_len: 29,
        bound_orbital_count: 3,
    })?;

    assert_eq!(setup.orbital_lengths.to_vec(), vec![29, 29, 29, 0]);
    assert_eq!(setup.kappa.to_vec(), vec![-1, 1, -2, -2]);
    assert_eq!(setup.open_shell.to_vec(), vec![true, true, true]);
    assert_eq!(setup.matching_kappa_count, 1);
    assert_close(setup.core_counts[0], 1.80, 1.0e-13);
    assert_close(setup.core_counts[1], 0.80, 1.0e-13);
    assert_close(setup.core_counts[2], 0.70, 1.0e-13);
    Ok(())
}

#[test]
fn orbital_setup_rejects_invalid_inputs() {
    let mut input = dfovrg_reference_inputs(false);
    input.kappa[1] = 0;
    assert!(matches!(
        fovrg_orbital_setup(FovrgOrbitalSetupInput {
            bound_large_components: input.bound_large_components.view(),
            bound_small_components: input.bound_small_components.view(),
            electron_counts: input.electron_counts.view(),
            valence_counts: input.valence_counts.view(),
            kappa: input.kappa.view(),
            target_kappa: -2,
            active_len: 29,
            bound_orbital_count: 3,
        }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "kappa",
            row: 1,
            ..
        })
    ));

    let mut input = dfovrg_reference_inputs(false);
    input.bound_small_components[(28, 2)] = Real::NAN;
    assert!(matches!(
        fovrg_orbital_setup(FovrgOrbitalSetupInput {
            bound_large_components: input.bound_large_components.view(),
            bound_small_components: input.bound_small_components.view(),
            electron_counts: input.electron_counts.view(),
            valence_counts: input.valence_counts.view(),
            kappa: input.kappa.view(),
            target_kappa: -2,
            active_len: 29,
            bound_orbital_count: 3,
        }),
        Err(FovrgError::NonFiniteRealInput {
            name: "bound_small_components",
            row: 28,
            ..
        })
    ));

    let mut input = dfovrg_reference_inputs(false);
    input.bound_large_components.column_mut(0).fill(0.0);
    input.bound_small_components.column_mut(0).fill(0.0);
    assert!(matches!(
        fovrg_orbital_setup(FovrgOrbitalSetupInput {
            bound_large_components: input.bound_large_components.view(),
            bound_small_components: input.bound_small_components.view(),
            electron_counts: input.electron_counts.view(),
            valence_counts: input.valence_counts.view(),
            kappa: input.kappa.view(),
            target_kappa: -2,
            active_len: 29,
            bound_orbital_count: 3,
        }),
        Err(FovrgError::CountTooSmall {
            name: "orbital_length",
            ..
        })
    ));
}

struct DfovrgReferenceInputs {
    exchange_cycle_count: usize,
    target_kappa: i32,
    muffin_tin_radius: Real,
    target_last_index: usize,
    energy: Complex,
    radii: Array1<Real>,
    exchange_correlation_potential: Array1<Complex>,
    valence_exchange_correlation_potential: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    valence_counts: Array1<Real>,
    kappa: Array1<i32>,
    muffin_tin_large_component: Complex,
    muffin_tin_small_component: Complex,
    irregular: bool,
    radial_match_index: usize,
    bound_orbital_count: usize,
}

impl DfovrgReferenceInputs {
    fn to_input(&self) -> FovrgDiracSolverInput<'_> {
        FovrgDiracSolverInput {
            exchange_cycle_count: self.exchange_cycle_count,
            target_kappa: self.target_kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            target_last_index: self.target_last_index,
            energy: self.energy,
            step: 0.45,
            radii: self.radii.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            valence_exchange_correlation_potential: self
                .valence_exchange_correlation_potential
                .view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            valence_counts: self.valence_counts.view(),
            kappa: self.kappa.view(),
            muffin_tin_large_component: self.muffin_tin_large_component,
            muffin_tin_small_component: self.muffin_tin_small_component,
            atomic_number: 29.0,
            irregular: self.irregular,
            c3_scale: 0,
            radial_match_index: self.radial_match_index,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn dfovrg_reference_inputs(irregular: bool) -> DfovrgReferenceInputs {
    let count = 40;
    let bound_orbitals = 3;
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (-8.8 + 0.45 * (row - 1.0)).exp()
    }));
    let exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.16 + 0.006 * row, 0.002 * (0.31 * row).cos())
    }));
    let valence_exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.12 + 0.004 * row, 0.001 * (0.27 * row).sin())
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.012 * orbital * (0.08 * row * orbital).sin() * (-0.010 * row).exp()
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.009 * orbital * (0.07 * row * orbital).cos() * (-0.012 * row).exp()
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.008 * row + 0.0011 * orbital * (0.19 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.005 * row + 0.0008 * orbital * (0.16 * row * orbital).sin()
    });

    if irregular {
        DfovrgReferenceInputs {
            exchange_cycle_count: 0,
            target_kappa: -1,
            muffin_tin_radius: 1.35,
            target_last_index: 16,
            energy: Complex::new(0.24, 0.035),
            radii,
            exchange_correlation_potential,
            valence_exchange_correlation_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.80, 1.00, 0.70]),
            valence_counts: Array1::from_vec(vec![0.0, 0.20, 0.0]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            muffin_tin_large_component: Complex::new(0.48, 0.06),
            muffin_tin_small_component: Complex::new(0.018, -0.009),
            irregular,
            radial_match_index: 9,
            bound_orbital_count: bound_orbitals,
        }
    } else {
        DfovrgReferenceInputs {
            exchange_cycle_count: 1,
            target_kappa: -2,
            muffin_tin_radius: 1.42,
            target_last_index: 15,
            energy: Complex::new(0.38, 0.020),
            radii,
            exchange_correlation_potential,
            valence_exchange_correlation_potential,
            bound_large_components,
            bound_small_components,
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.80, 1.00, 0.70]),
            valence_counts: Array1::from_vec(vec![0.0, 0.20, 0.0]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            muffin_tin_large_component: Complex::new(0.0, 0.0),
            muffin_tin_small_component: Complex::new(0.0, 0.0),
            irregular,
            radial_match_index: 9,
            bound_orbital_count: bound_orbitals,
        }
    }
}

struct SoloutReferenceInputs {
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    energy: Complex,
    origin_power: Real,
    kappa: i32,
    muffin_tin_radius: Real,
    potential: Array1<Complex>,
    potential_coefficients: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    large_exchange_coefficients: Array1<Complex>,
    small_exchange_coefficients: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    c3_scale: i32,
    radial_match_index: usize,
    last_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl SoloutReferenceInputs {
    fn to_input(&self) -> FovrgOutgoingSolutionInput<'_> {
        FovrgOutgoingSolutionInput {
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            energy: self.energy,
            origin_power: self.origin_power,
            kappa: self.kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            large_exchange_coefficients: self.large_exchange_coefficients.view(),
            small_exchange_coefficients: self.small_exchange_coefficients.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            c3_scale: self.c3_scale,
            radial_match_index: self.radial_match_index,
            last_index: self.last_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn solout_reference_inputs(case_id: usize) -> SoloutReferenceInputs {
    let active_len = 15;
    let coefficient_count = 6;
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));
    let large_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
        let row = row as Real;
        Complex::new(
            0.0025 * row + 0.001 * (0.33 * row).cos(),
            -0.0015 * row + 0.0007 * (0.21 * row).sin(),
        )
    }));
    let small_exchange_coefficients = Array1::from_iter((1..=coefficient_count).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.0018 * row + 0.0008 * (0.27 * row).sin(),
            0.0012 * row + 0.0005 * (0.19 * row).cos(),
        )
    }));

    let mut potential_coefficients = Array1::<Complex>::zeros(coefficient_count);
    match case_id {
        1 => {
            potential_coefficients[0] = Complex::new(-0.21, 0.0);
            potential_coefficients[1] = Complex::new(0.013, -0.002);
            potential_coefficients[2] = Complex::new(-0.004, 0.001);
            potential_coefficients[3] = Complex::new(0.002, 0.0005);
            potential_coefficients[4] = Complex::new(-0.001, 0.0002);
            potential_coefficients[5] = Complex::new(0.0006, -0.0001);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(0.85, -0.13),
                initial_small_coefficient: Complex::new(-0.045, 0.018),
                energy: Complex::new(-0.42, 0.018),
                origin_power: 1.982,
                kappa: -2,
                muffin_tin_radius: 1.35,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 0,
                radial_match_index: 8,
                last_index: 11,
                wkb_index: 6,
                coefficient_count,
                active_len,
            }
        }
        2 => {
            potential_coefficients[0] = Complex::new(0.11, 0.0);
            potential_coefficients[1] = Complex::new(-0.009, 0.002);
            potential_coefficients[2] = Complex::new(0.003, -0.001);
            potential_coefficients[3] = Complex::new(0.018, -0.004);
            potential_coefficients[4] = Complex::new(0.001, 0.0003);
            potential_coefficients[5] = Complex::new(-0.0004, 0.0002);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(-0.72, 0.21),
                initial_small_coefficient: Complex::new(0.037, -0.015),
                energy: Complex::new(0.36, -0.027),
                origin_power: 3.025,
                kappa: 3,
                muffin_tin_radius: 1.20,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 1,
                radial_match_index: 9,
                last_index: 12,
                wkb_index: 7,
                coefficient_count,
                active_len,
            }
        }
        _ => {
            potential_coefficients[0] = Complex::new(-0.18, 0.0);
            potential_coefficients[1] = Complex::new(0.010, 0.001);
            potential_coefficients[2] = Complex::new(-0.003, 0.0008);
            potential_coefficients[3] = Complex::new(-0.015, 0.003);
            potential_coefficients[4] = Complex::new(0.0008, -0.0002);
            potential_coefficients[5] = Complex::new(-0.0003, 0.0001);
            SoloutReferenceInputs {
                initial_large_coefficient: Complex::new(0.64, 0.08),
                initial_small_coefficient: Complex::new(0.025, -0.011),
                energy: Complex::new(0.22, 0.041),
                origin_power: 0.965,
                kappa: -1,
                muffin_tin_radius: 1.40,
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                large_exchange_coefficients,
                small_exchange_coefficients,
                c3_potential,
                radii,
                c3_scale: 1,
                radial_match_index: 8,
                last_index: 10,
                wkb_index: 7,
                coefficient_count,
                active_len,
            }
        }
    }
}

struct SolinReferenceInputs {
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    energy: Complex,
    origin_power: Real,
    kappa: i32,
    muffin_tin_radius: Real,
    potential: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    c3_scale: i32,
    radial_match_index: usize,
    last_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl SolinReferenceInputs {
    fn to_input(&self) -> FovrgInwardSolutionInput<'_> {
        FovrgInwardSolutionInput {
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            energy: self.energy,
            origin_power: self.origin_power,
            kappa: self.kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            potential: self.potential.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            c3_scale: self.c3_scale,
            radial_match_index: self.radial_match_index,
            last_index: self.last_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn solin_reference_inputs(case_id: usize) -> SolinReferenceInputs {
    let active_len = 15;
    let coefficient_count = 6;
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));

    match case_id {
        1 => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(0.85, -0.13),
            initial_small_coefficient: Complex::new(-0.045, 0.018),
            energy: Complex::new(0.42, 0.018),
            origin_power: 1.982,
            kappa: -2,
            muffin_tin_radius: 1.35,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 8,
            last_index: 11,
            wkb_index: 6,
            coefficient_count,
            active_len,
        },
        2 => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(-0.72, 0.21),
            initial_small_coefficient: Complex::new(0.037, -0.015),
            energy: Complex::new(0.36, -0.027),
            origin_power: 3.025,
            kappa: 3,
            muffin_tin_radius: 1.20,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 9,
            last_index: 12,
            wkb_index: 7,
            coefficient_count,
            active_len,
        },
        _ => SolinReferenceInputs {
            initial_large_coefficient: Complex::new(0.64, 0.08),
            initial_small_coefficient: Complex::new(0.025, -0.011),
            energy: Complex::new(0.22, 0.041),
            origin_power: 0.965,
            kappa: -1,
            muffin_tin_radius: 1.40,
            potential,
            large_exchange,
            small_exchange,
            c3_potential,
            radii,
            c3_scale: 0,
            radial_match_index: 8,
            last_index: 12,
            wkb_index: 7,
            coefficient_count,
            active_len,
        },
    }
}

struct WfirdcReferenceInputs {
    energy: Complex,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
    exchange_correlation_potential: Array1<Complex>,
    c3_potential: Array1<Complex>,
    initial_large_coefficient: Complex,
    initial_small_coefficient: Complex,
    muffin_tin_radius: Real,
    c3_scale: i32,
    irregular: bool,
    radial_match_index: usize,
    wkb_index: usize,
    coefficient_count: usize,
    active_len: usize,
}

impl WfirdcReferenceInputs {
    fn to_input(&self) -> FovrgInitialPhotoelectronInput<'_> {
        FovrgInitialPhotoelectronInput {
            energy: self.energy,
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            orbital_lengths: self.orbital_lengths.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            c3_potential: self.c3_potential.view(),
            initial_large_coefficient: self.initial_large_coefficient,
            initial_small_coefficient: self.initial_small_coefficient,
            nuclear_charge: 29.0,
            muffin_tin_radius: self.muffin_tin_radius,
            step: 0.045,
            speed_of_light: 137.0373,
            c3_scale: self.c3_scale,
            irregular: self.irregular,
            radial_match_index: self.radial_match_index,
            wkb_index: self.wkb_index,
            coefficient_count: self.coefficient_count,
            orbital_count: 3,
            active_len: self.active_len,
        }
    }
}

fn wfirdc_reference_inputs(case_id: usize) -> WfirdcReferenceInputs {
    let active_len = 15;
    let bound_orbitals = 2;
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.01 * row + 0.0015 * orbital * (0.25 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.007 * row + 0.001 * orbital * (0.18 * row * orbital).sin()
    });
    let exchange_correlation_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.22 + 0.015 * row, 0.003 * (0.37 * row).cos())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));

    if case_id == 1 {
        WfirdcReferenceInputs {
            energy: Complex::new(0.42, 0.018),
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.25, 0.65]),
            kappa: Array1::from_vec(vec![-1, 1, -2]),
            orbital_lengths: Array1::from_vec(vec![0, 0, 12]),
            exchange_correlation_potential,
            c3_potential,
            initial_large_coefficient: Complex::new(0.0, 0.0),
            initial_small_coefficient: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.35,
            c3_scale: 0,
            irregular: false,
            radial_match_index: 8,
            wkb_index: 6,
            coefficient_count: 3,
            active_len,
        }
    } else {
        WfirdcReferenceInputs {
            energy: Complex::new(0.22, 0.041),
            bound_large_coefficients,
            bound_small_coefficients,
            electron_counts: Array1::from_vec(vec![1.25, 0.65]),
            kappa: Array1::from_vec(vec![-1, 1, -1]),
            orbital_lengths: Array1::from_vec(vec![0, 0, 13]),
            exchange_correlation_potential,
            c3_potential,
            initial_large_coefficient: Complex::new(0.64, 0.08),
            initial_small_coefficient: Complex::new(0.025, -0.011),
            muffin_tin_radius: 1.40,
            c3_scale: 0,
            irregular: true,
            radial_match_index: 8,
            wkb_index: 7,
            coefficient_count: 2,
            active_len,
        }
    }
}

#[test]
fn nuclear_potential_matches_feff_nucdec_point_reference() -> Result<(), FovrgError> {
    let potential = fovrg_nuclear_potential(FovrgNuclearPotentialInput {
        nuclear_charge: 29.0,
        step: 0.0725,
        first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
        radial_count: 8,
        coefficient_count: 6,
    })?;

    assert_eq!(potential.nucleus_index, 1);
    assert_close(
        potential.first_radius_times_charge,
        0.004_371_259_177_768_818_5,
        1.0e-15,
    );
    let expected_coefficients = [-29.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (actual, expected) in potential
        .development_coefficients
        .iter()
        .zip(expected_coefficients)
    {
        assert_close(*actual, expected, 1.0e-13);
    }

    let expected_rows = [
        (0.000_150_733_075_095_476_5, -192_393.076_182_058_78),
        (0.000_162_067_117_982_503_44, -178_938.210_051_534_35),
        (0.000_174_253_399_358_552_06, -166_424.299_937_634_06),
        (0.000_187_356_001_427_070_04, -154_785.540_783_909_73),
        (0.000_201_443_824_912_202_5, -143_960.729_561_402),
        (0.000_216_590_951_376_884_9, -133_892.943_429_283_77),
        (0.000_232_877_032_784_649_17, -124_529.240_403_099_25),
        (0.000_250_387_710_353_676, -115_820.380_956_545_8),
    ];
    for (row, (expected_radius, expected_potential)) in expected_rows.into_iter().enumerate() {
        assert_close(potential.radii[row], expected_radius, 1.0e-13);
        assert_close(potential.potential[row], expected_potential, 1.0e-13);
    }
    Ok(())
}

#[test]
fn nuclear_potential_rejects_invalid_inputs() {
    assert!(matches!(
        fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0725,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 0,
            coefficient_count: 6,
        }),
        Err(FovrgError::CountTooSmall {
            name: "radial_count",
            ..
        })
    ));
    assert!(matches!(
        fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0725,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 8,
            coefficient_count: 4,
        }),
        Err(FovrgError::CountTooSmall {
            name: "coefficient_count",
            ..
        })
    ));
    assert!(matches!(
        fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 0.0,
            step: 0.0725,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 8,
            coefficient_count: 6,
        }),
        Err(FovrgError::NonPositiveInput {
            name: "nuclear_charge",
            ..
        })
    ));
    assert!(matches!(
        fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0,
            first_radius_times_charge: 29.0 * (-8.8_f64).exp(),
            radial_count: 8,
            coefficient_count: 6,
        }),
        Err(FovrgError::NonPositiveInput { name: "step", .. })
    ));
    assert!(matches!(
        fovrg_nuclear_potential(FovrgNuclearPotentialInput {
            nuclear_charge: 29.0,
            step: 0.0725,
            first_radius_times_charge: Real::NAN,
            radial_count: 8,
            coefficient_count: 6,
        }),
        Err(FovrgError::NonFiniteInput {
            name: "first_radius_times_charge",
            ..
        })
    ));
}

#[test]
fn yk_zk_transform_matches_feff_yzktec_reference() -> Result<(), FovrgError> {
    let (source, coefficients, radii) = yzktec_reference_inputs(12);

    let transform = fovrg_yk_zk_transform(FovrgYkZkTransformInput {
        source: source.view(),
        source_coefficients: coefficients.view(),
        radii: radii.view(),
        initial_power: Complex::new(1.35, -0.25),
        step: 0.0725,
        angular_momentum: 2,
        coefficient_count: 6,
        source_len: 9,
        active_len: 12,
        tail_correction: Complex::new(0.011, -0.006),
    })?;

    assert_eq!(transform.computed_len, 10);
    assert_complex_close(
        transform.origin_constant,
        1_069.293_326_934_643,
        639.337_203_837_502_8,
        1.0e-12,
    );

    let expected_rows = [
        (
            0.006_376_970_423_953_328,
            0.003_747_109_936_537_645_4,
            0.000_019_115_876_398_023_115,
            0.000_002_615_603_860_575_636,
        ),
        (
            0.007_841_326_927_116_237,
            0.004_425_503_213_295_339,
            0.000_415_175_421_810_819_7,
            0.001_186_221_311_123_577_3,
        ),
        (
            0.009_454_062_278_996_728,
            0.004_817_609_696_528_203,
            0.001_052_233_690_642_138_8,
            0.002_225_420_005_754_274_7,
        ),
        (
            0.011_156_498_748_891_856,
            0.004_912_703_002_968_925,
            0.001_915_624_422_266_479,
            0.003_118_393_016_633_964_7,
        ),
        (
            0.012_883_154_525_001_68,
            0.004_698_896_965_378_377,
            0.002_982_924_829_137_327,
            0.003_859_683_726_441_837_7,
        ),
        (
            0.014_563_357_943_144_598,
            0.004_164_285_902_400_606,
            0.004_223_307_649_668_978,
            0.004_440_459_445_284_031,
        ),
        (
            0.016_123_447_951_845_19,
            0.003_298_256_791_962_156,
            0.005_597_243_449_987_172_5,
            0.004_848_768_666_445_236,
        ),
        (
            0.017_489_549_856_229_16,
            0.002_093_015_402_338_084,
            0.007_056_612_425_756_153,
            0.005_069_801_813_782_371,
        ),
        (
            0.018_590_890_204_374_55,
            0.000_545_375_511_912_808,
            0.008_545_277_387_035_53,
            0.005_086_162_856_970_115_5,
        ),
        (
            0.019_630_800_153_888_66,
            -0.001_305_325_902_639_564_2,
            0.008_630_800_153_888_66,
            0.004_694_674_097_360_436,
        ),
    ];
    for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
        assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
        assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
    }

    let expected_coefficients = [
        (
            -1.824_158_963_244_542,
            -0.246_122_633_186_553_6,
            0.237_140_665_221_790_4,
            0.031_995_942_314_251_964,
        ),
        (
            2.794_098_740_012_050_3,
            0.730_272_609_187_755_4,
            0.195_586_911_800_843_6,
            0.051_119_082_643_142_88,
        ),
        (
            0.609_454_103_153_871_8,
            0.232_241_030_552_876_95,
            0.164_552_607_851_545_35,
            0.062_705_078_249_276_76,
        ),
        (
            0.297_530_519_518_787_1,
            0.147_284_129_112_308_2,
            0.139_839_344_173_829_93,
            0.069_223_540_682_784_84,
        ),
        (
            0.178_046_974_447_949_95,
            0.107_477_058_743_461_04,
            0.119_291_472_880_126_45,
            0.072_009_629_358_118_89,
        ),
        (
            0.116_898_830_661_045_85,
            0.082_624_380_349_051_94,
            0.101_701_982_675_109_88,
            0.071_883_210_903_675_18,
        ),
    ];
    for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_coefficients.into_iter().enumerate() {
        assert_complex_close(transform.yk_coefficients[row], yk_re, yk_im, 1.0e-13);
        assert_complex_close(transform.zk_coefficients[row], zk_re, zk_im, 1.0e-13);
    }
    Ok(())
}

#[test]
fn yk_zk_transform_rejects_invalid_inputs() {
    let (source, coefficients, radii) = yzktec_reference_inputs(12);

    assert!(matches!(
        fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, 0.0),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 1,
            tail_correction: Complex::new(0.0, 0.0),
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, 0.0),
            step: 0.0,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.0, 0.0),
        }),
        Err(FovrgError::NonPositiveInput { name: "step", .. })
    ));
    assert!(matches!(
        fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, 0.0),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 11,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.0, 0.0),
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "source_coefficients",
            ..
        })
    ));

    let mut bad_source = source.clone();
    bad_source[3] = Complex::new(0.0, Real::NAN);
    assert!(matches!(
        fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: bad_source.view(),
            source_coefficients: coefficients.view(),
            radii: radii.view(),
            initial_power: Complex::new(1.35, 0.0),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.0, 0.0),
        }),
        Err(FovrgError::NonFiniteComplexInput {
            name: "source",
            row: 3,
            ..
        })
    ));

    let mut bad_radii = radii.clone();
    bad_radii[0] = -1.0;
    assert!(matches!(
        fovrg_yk_zk_transform(FovrgYkZkTransformInput {
            source: source.view(),
            source_coefficients: coefficients.view(),
            radii: bad_radii.view(),
            initial_power: Complex::new(1.35, 0.0),
            step: 0.0725,
            angular_momentum: 2,
            coefficient_count: 6,
            source_len: 9,
            active_len: 12,
            tail_correction: Complex::new(0.0, 0.0),
        }),
        Err(FovrgError::NonPositiveRadius { row: 0, .. })
    ));
}

#[test]
fn yk_zk_exchange_matches_feff_yzkrdc_reference() -> Result<(), FovrgError> {
    let input = yzkrdc_reference_inputs(12);

    let transform = fovrg_yk_zk_exchange(input.as_exchange_input())?;

    assert_eq!(transform.computed_len, 10);
    assert_complex_close(
        transform.origin_constant,
        1_321.269_761_542_853_5,
        1_058.551_269_340_285_2,
        1.0e-12,
    );

    let expected_rows = [
        (
            0.007_686_009_135_817_749,
            0.006_170_157_063_400_744,
            0.000_000_645_317_783_462_879_7,
            0.000_000_110_270_749_084_274_43,
        ),
        (
            0.009_300_746_624_727_518,
            0.007_544_419_441_270_886,
            0.001_294_275_945_600_778,
            0.000_639_802_281_166_626_1,
        ),
        (
            0.010_786_770_527_864_456,
            0.008_925_139_869_295_514,
            0.002_573_522_373_652_341,
            0.001_630_025_738_506_313,
        ),
        (
            0.012_109_032_230_448_815,
            0.010_184_928_348_947_297,
            0.003_887_582_939_633_221_6,
            0.002_904_232_874_818_797,
        ),
        (
            0.013_206_275_901_284_993,
            0.011_197_011_268_772_228,
            0.005_274_223_639_134_339,
            0.004_372_201_622_443_519_5,
        ),
        (
            0.013_990_089_034_721_83,
            0.011_844_365_308_609_77,
            0.006_755_737_128_168_105,
            0.005_923_123_611_939_633,
        ),
        (
            0.014_345_254_715_897_94,
            0.012_029_374_779_196_415,
            0.008_335_434_490_732_629,
            0.007_430_974_286_925_581,
        ),
        (
            0.014_131_414_050_294_111,
            0.011_683_264_170_573_946,
            0.009_995_141_713_724_128,
            0.008_761_915_452_378_507,
        ),
        (
            0.013_185_862_903_069_551,
            0.010_774_522_262_808_485,
            0.011_694_148_402_953_802,
            0.009_783_248_603_735_934,
        ),
        (
            0.011_660_651_859_152_821,
            0.009_488_268_367_479_21,
            0.011_660_651_859_152_821,
            0.009_488_268_367_479_21,
        ),
    ];
    for (row, (yk_re, yk_im, zk_re, zk_im)) in expected_rows.into_iter().enumerate() {
        assert_complex_close(transform.yk[row], yk_re, yk_im, 1.0e-13);
        assert_complex_close(transform.zk[row], zk_re, zk_im, 1.0e-13);
    }

    let expected_coefficients = [
        (6.375_854_958_562_043, 1.073_871_387_646_292_4),
        (1.497_833_540_772_686, 0.370_169_086_848_655_57),
        (1.049_320_795_997_538_8, 0.338_218_568_996_506_2),
        (0.843_625_047_557_286_8, 0.332_360_660_760_794_2),
        (0.713_658_559_831_859_2, 0.329_689_349_293_459_3),
        (0.619_406_204_717_043, 0.325_898_372_715_123_5),
    ];
    for (row, (expected_re, expected_im)) in expected_coefficients.into_iter().enumerate() {
        assert_complex_close(
            transform.yk_coefficients[row],
            expected_re,
            expected_im,
            1.0e-13,
        );
    }
    Ok(())
}

#[test]
fn yk_zk_exchange_rejects_invalid_inputs() {
    let mut input = yzkrdc_reference_inputs(12);
    input.large_component[2] = Real::NAN;

    assert!(matches!(
        fovrg_yk_zk_exchange(input.as_exchange_input()),
        Err(FovrgError::NonFiniteRealInput {
            name: "large_component",
            row: 2,
            ..
        })
    ));

    let mut input = yzkrdc_reference_inputs(12);
    input.partner_small_coefficients[1] = Complex::new(0.0, Real::INFINITY);
    assert!(matches!(
        fovrg_yk_zk_exchange(input.as_exchange_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "partner_small_coefficients",
            row: 1,
            ..
        })
    ));

    let input = yzkrdc_reference_inputs(4);
    assert!(matches!(
        fovrg_yk_zk_exchange(FovrgYkZkExchangeInput {
            active_len: 5,
            ..input.as_exchange_input()
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "large_component",
            ..
        })
    ));
}

#[test]
fn overlap_integral_matches_feff_dsordc_reference() -> Result<(), FovrgError> {
    let input = dsordc_reference_inputs(9);

    let integral = fovrg_overlap_integral(input.as_overlap_input())?;

    assert_complex_close(
        integral,
        0.018_257_373_605_649_284,
        0.014_647_428_406_545_006,
        1.0e-13,
    );
    Ok(())
}

#[test]
fn overlap_integral_rejects_invalid_inputs() {
    let input = dsordc_reference_inputs(9);

    assert!(matches!(
        fovrg_overlap_integral(FovrgOverlapIntegralInput {
            active_len: 8,
            ..input.as_overlap_input()
        }),
        Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_overlap_integral(FovrgOverlapIntegralInput {
            active_len: 2,
            ..input.as_overlap_input()
        }),
        Err(FovrgError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_overlap_integral(FovrgOverlapIntegralInput {
            active_len: 11,
            ..input.as_overlap_input()
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "large_integrand",
            ..
        })
    ));
    assert!(matches!(
        fovrg_overlap_integral(FovrgOverlapIntegralInput {
            step: 0.0,
            ..input.as_overlap_input()
        }),
        Err(FovrgError::NonPositiveInput { name: "step", .. })
    ));

    let mut input = dsordc_reference_inputs(9);
    input.radii[2] = 0.0;
    assert!(matches!(
        fovrg_overlap_integral(input.as_overlap_input()),
        Err(FovrgError::NonPositiveRadius { row: 2, .. })
    ));

    let mut input = dsordc_reference_inputs(9);
    input.large_integrand_coefficients[3] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        fovrg_overlap_integral(input.as_overlap_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "large_integrand_coefficients",
            row: 3,
            ..
        })
    ));
}

#[test]
fn schmidt_orthogonalization_matches_feff_ortdac_reference() -> Result<(), FovrgError> {
    let input = ortdac_reference_inputs(9);

    let orthogonalized = fovrg_schmidt_orthogonalize(input.as_orthogonalization_input())?;

    assert_ne!(orthogonalized.overlaps[0], Complex::new(0.0, 0.0));
    assert_eq!(orthogonalized.overlaps[1], Complex::new(0.0, 0.0));
    assert_eq!(orthogonalized.overlaps[2], Complex::new(0.0, 0.0));
    assert_ne!(orthogonalized.overlaps[3], Complex::new(0.0, 0.0));

    let expected_rows = [
        (
            0.184_796_621_476_688_8,
            0.960_525_659_674_847_8,
            0.953_489_591_844_743_2,
            0.196_175_227_984_495_05,
        ),
        (
            0.364_943_848_030_108_26,
            0.909_210_209_413_431_4,
            0.932_155_457_421_994,
            0.411_067_158_250_690_3,
        ),
        (
            0.535_755_652_121_730_3,
            0.846_307_238_142_853_7,
            0.903_311_497_295_576_5,
            0.608_386_237_505_285_2,
        ),
        (
            0.692_898_032_849_261_3,
            0.772_271_807_926_033_1,
            0.867_091_885_664_384_1,
            0.780_141_644_613_100_9,
        ),
        (
            0.832_426_823_226_043_9,
            0.687_685_325_115_306_8,
            0.823_683_636_514_673_7,
            0.919_472_245_213_980_9,
        ),
        (
            0.950_900_258_284_096_4,
            0.593_246_291_500_718_2,
            0.773_325_760_496_256_5,
            1.020_947_524_294_445_2,
        ),
        (
            1.045_477_281_469_491_5,
            0.489_760_058_904_124,
            0.716_308_162_735_437_1,
            1.080_805_531_074_559_2,
        ),
        (
            1.113_998_793_619_971_6,
            0.378_127_783_523_524,
            0.652_970_269_213_538_8,
            1.097_117_398_093_875_3,
        ),
        (
            1.155_049_530_148_776_2,
            0.259_334_766_409_572_26,
            0.583_699_367_941_233_6,
            1.069_871_225_404_380_5,
        ),
    ];
    for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate() {
        assert_complex_close(
            orthogonalized.large_component[row],
            large_re,
            large_im,
            1.0e-13,
        );
        assert_complex_close(
            orthogonalized.small_component[row],
            small_re,
            small_im,
            1.0e-13,
        );
    }

    let expected_coefficients = [
        (
            0.998_449_079_711_476_6,
            0.111_350_550_939_607_72,
            0.068_224_857_711_253_53,
            1.016_544_722_930_134,
        ),
        (
            1.013_026_259_606_995_7,
            0.245_410_754_538_065_13,
            0.135_740_121_285_759_1,
            1.018_823_567_734_752_8,
        ),
        (
            1.011_555_658_693_028_5,
            0.370_053_923_878_711_44,
            0.201_841_908_576_728_68,
            1.007_158_627_823_294_7,
        ),
        (
            0.994_728_243_334_449_6,
            0.480_813_496_862_218_4,
            0.265_837_715_515_805,
            0.982_072_667_678_714_5,
        ),
        (
            0.963_494_952_966_650_6,
            0.573_626_109_357_632,
            0.327_051_990_766_753_06,
            0.944_281_663_334_709_4,
        ),
        (
            0.919_050_620_050_682_9,
            0.644_948_608_319_962_8,
            0.384_831_574_144_536_1,
            0.894_684_562_223_998_2,
        ),
    ];
    for (coefficient, (large_re, large_im, small_re, small_im)) in
        expected_coefficients.into_iter().enumerate()
    {
        assert_complex_close(
            orthogonalized.large_coefficients[coefficient],
            large_re,
            large_im,
            1.0e-13,
        );
        assert_complex_close(
            orthogonalized.small_coefficients[coefficient],
            small_re,
            small_im,
            1.0e-13,
        );
    }
    Ok(())
}

#[test]
fn schmidt_orthogonalization_rejects_invalid_inputs() {
    let input = ortdac_reference_inputs(9);

    assert!(matches!(
        fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
            target_kappa: 0,
            ..input.as_orthogonalization_input()
        }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "target_kappa",
            value: 0,
            ..
        })
    ));
    assert!(matches!(
        fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
            active_len: 8,
            ..input.as_orthogonalization_input()
        }),
        Err(FovrgError::CountMustBeOdd {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_schmidt_orthogonalize(FovrgOrthogonalizationInput {
            bound_orbital_count: 5,
            ..input.as_orthogonalization_input()
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "bound_large_components",
            ..
        })
    ));

    let mut input = ortdac_reference_inputs(9);
    input.electron_counts[0] = Real::NAN;
    assert!(matches!(
        fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
        Err(FovrgError::NonFiniteRealInput {
            name: "electron_counts",
            row: 0,
            ..
        })
    ));

    let mut input = ortdac_reference_inputs(9);
    input.target_large_component[1] = Complex::new(0.0, Real::INFINITY);
    assert!(matches!(
        fovrg_schmidt_orthogonalize(input.as_orthogonalization_input()),
        Err(FovrgError::NonFiniteComplexInput {
            name: "target_large_component",
            row: 1,
            ..
        })
    ));
}

#[test]
fn exchange_potential_matches_feff_potex_reference() -> Result<(), FovrgError> {
    let input = potex_reference_inputs(9);

    let potential = fovrg_exchange_potential(input.as_exchange_potential_input())?;

    let expected_rows = [
        (
            0.000_005_554_864_571_582_592,
            0.000_004_589_245_040_261_105,
            0.000_039_609_278_623_293_83,
            0.000_033_434_207_770_074_9,
        ),
        (
            0.000_011_841_826_673_104_183,
            0.000_009_794_866_583_804_325,
            0.000_042_053_026_297_685_21,
            0.000_035_634_329_767_400_91,
        ),
        (
            0.000_018_578_019_491_824_635,
            0.000_015_404_560_990_245_302,
            0.000_043_309_970_634_588_22,
            0.000_036_974_047_816_590_13,
        ),
        (
            0.000_025_293_374_225_649_448,
            0.000_020_974_401_383_246_206,
            0.000_043_220_277_722_069_02,
            0.000_037_209_351_789_692_02,
        ),
        (
            0.000_031_463_027_867_695_25,
            0.000_026_005_262_272_416_62,
            0.000_041_711_066_672_227_27,
            0.000_036_233_682_295_299_64,
        ),
        (
            0.000_036_572_642_292_251_735,
            0.000_030_066_055_448_048_974,
            0.000_038_831_760_336_099_916,
            0.000_034_098_776_520_822_48,
        ),
        (
            0.000_040_212_198_293_964_37,
            0.000_032_909_100_990_340_72,
            0.000_034_779_450_472_774_91,
            0.000_030_991_961_623_472_31,
        ),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ];
    for (row, (large_re, large_im, small_re, small_im)) in expected_rows.into_iter().enumerate() {
        assert_complex_close(potential.large_potential[row], large_re, large_im, 1.0e-13);
        assert_complex_close(potential.small_potential[row], small_re, small_im, 1.0e-13);
    }

    let expected_coefficients = [
        (
            0.056_004_531_605_744_41,
            0.046_663_043_007_772_96,
            0.000_603_453_997_835_984_7,
            0.000_503_278_831_814_845_3,
        ),
        (
            -1.349_603_830_126_038,
            -1.128_877_944_124_393_2,
            -0.045_179_344_996_730_146,
            -0.037_885_484_599_471_386,
        ),
        (
            -2.231_032_417_788_144_4,
            -1.854_555_246_757_578,
            -0.141_157_217_260_626_58,
            -0.117_915_434_665_414_93,
        ),
        (
            24.781_460_626_354_06,
            19.753_480_329_995_963,
            2.027_705_895_254_902,
            1.613_953_852_227_001_6,
        ),
        (
            24.726_993_882_200_276,
            19.712_367_835_956_03,
            4.170_773_455_085_641,
            3.325_408_244_119_067_5,
        ),
        (
            24.319_899_966_071_53,
            19.384_360_683_910_17,
            6.262_813_264_194_341,
            4.995_924_412_893_355,
        ),
    ];
    for (coefficient, (large_re, large_im, small_re, small_im)) in
        expected_coefficients.into_iter().enumerate()
    {
        assert_complex_close(
            potential.large_coefficients[coefficient],
            large_re,
            large_im,
            1.0e-13,
        );
        assert_complex_close(
            potential.small_coefficients[coefficient],
            small_re,
            small_im,
            1.0e-13,
        );
    }
    Ok(())
}

#[test]
fn exchange_potential_rejects_invalid_inputs() {
    let input = potex_reference_inputs(9);

    assert!(matches!(
        fovrg_exchange_potential(FovrgExchangePotentialInput {
            target_kappa: 0,
            ..input.as_exchange_potential_input()
        }),
        Err(FovrgError::InvalidQuantumNumber {
            name: "target_kappa",
            value: 0,
            ..
        })
    ));
    assert!(matches!(
        fovrg_exchange_potential(FovrgExchangePotentialInput {
            radial_output_count: 10,
            ..input.as_exchange_potential_input()
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "radial_output_count",
            ..
        })
    ));
    assert!(matches!(
        fovrg_exchange_potential(FovrgExchangePotentialInput {
            speed_of_light: 0.0,
            ..input.as_exchange_potential_input()
        }),
        Err(FovrgError::ZeroInput {
            name: "speed_of_light"
        })
    ));
    assert!(matches!(
        fovrg_exchange_potential(FovrgExchangePotentialInput {
            bound_orbital_count: 5,
            ..input.as_exchange_potential_input()
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "bound_large_components",
            ..
        })
    ));

    let mut input = potex_reference_inputs(9);
    input.orbital_lengths[2] = 0;
    assert!(matches!(
        fovrg_exchange_potential(input.as_exchange_potential_input()),
        Err(FovrgError::CountTooSmall {
            name: "orbital_length",
            ..
        })
    ));

    let mut input = potex_reference_inputs(9);
    input.angular_coefficients[(1, 0)] = Real::NAN;
    assert!(matches!(
        fovrg_exchange_potential(input.as_exchange_potential_input()),
        Err(FovrgError::NonFiniteRealInput {
            name: "angular_coefficients",
            row: 1,
            ..
        })
    ));
}

#[test]
fn potential_development_matches_feff_potdvp_reference() -> Result<(), FovrgError> {
    let input = potdvp_reference_inputs(12);

    let development = fovrg_potential_development(input.as_potential_input())?;

    assert_close(
        development.origin_correction,
        0.000_092_381_409_682_418_76,
        1.0e-13,
    );
    let expected_potential = [
        -0.002_211_097_828_492_991_6,
        -0.001_838_258_707_742_217_9,
        -0.001_437_578_456_148_908_5,
        -0.003_049_520_002_144_625,
        -0.002_623_511_736_279_590_5,
        -0.002_546_330_557_249_715,
        -0.002_045_957_521_005_020_5,
        -0.001_773_999_888_200_908_3,
        0.001_583_525_507_534_584_8,
        0.002_189_205_770_785_14,
    ];
    for (actual, expected) in development
        .potential_coefficients
        .iter()
        .zip(expected_potential)
    {
        assert_complex_close(*actual, expected, 0.0, 1.0e-13);
    }

    let expected_density = [
        0.279_894_020_220_530_5,
        0.284_515_551_889_673_2,
        0.340_938_951_910_833_2,
        0.343_369_832_974_347,
        0.381_101_847_054_515_8,
        0.388_553_939_183_866_9,
        0.381_768_833_467_862,
        0.368_012_415_945_436_16,
    ];
    for (actual, expected) in development
        .density_coefficients
        .iter()
        .zip(expected_density)
    {
        assert_close(*actual, expected, 1.0e-13);
    }
    Ok(())
}

#[test]
fn potential_development_rejects_invalid_inputs() {
    let mut input = potdvp_reference_inputs(12);
    input.nuclear_coefficients[0] = Real::NAN;
    assert!(matches!(
        fovrg_potential_development(input.as_potential_input()),
        Err(FovrgError::NonFiniteRealInput {
            name: "nuclear_coefficients",
            row: 0,
            ..
        })
    ));

    let mut input = potdvp_reference_inputs(12);
    input.kappa[1] = 0;
    assert!(matches!(
        fovrg_potential_development(input.as_potential_input()),
        Err(FovrgError::InvalidQuantumNumber {
            name: "kappa",
            row: 1,
            value: 0,
        })
    ));

    let mut input = potdvp_reference_inputs(12);
    input.large_coefficients = Array2::zeros((7, 4));
    assert!(matches!(
        fovrg_potential_development(input.as_potential_input()),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "large_coefficients",
            ..
        })
    ));

    let input = potdvp_reference_inputs(12);
    assert!(matches!(
        fovrg_potential_development(FovrgPotentialDevelopmentInput {
            speed_of_light: 0.0,
            ..input.as_potential_input()
        }),
        Err(FovrgError::ZeroInput {
            name: "speed_of_light"
        })
    ));
}

struct IntoutReferenceInputs {
    initial_large_component: Complex,
    initial_small_component: Complex,
    energy: Complex,
    potential: Array1<Complex>,
    potential_coefficients: Array1<Complex>,
    large_exchange: Array1<Complex>,
    small_exchange: Array1<Complex>,
    c3_potential: Array1<Complex>,
    radii: Array1<Real>,
    kappa: i32,
    c3_scale: i32,
    start_index: usize,
    last_index: usize,
    active_len: usize,
}

impl IntoutReferenceInputs {
    fn to_input(&self) -> FovrgOutwardIntegrationInput<'_> {
        FovrgOutwardIntegrationInput {
            initial_large_component: self.initial_large_component,
            initial_small_component: self.initial_small_component,
            energy: self.energy,
            potential: self.potential.view(),
            potential_coefficients: self.potential_coefficients.view(),
            large_exchange: self.large_exchange.view(),
            small_exchange: self.small_exchange.view(),
            c3_potential: self.c3_potential.view(),
            radii: self.radii.view(),
            speed_of_light: 137.035_999_084,
            step: 0.045,
            kappa: self.kappa,
            c3_scale: self.c3_scale,
            start_index: self.start_index,
            last_index: self.last_index,
            active_len: self.active_len,
        }
    }
}

fn intout_reference_inputs(case_id: usize) -> IntoutReferenceInputs {
    let active_len = 15;
    let mut potential_coefficients = Array1::<Complex>::zeros(10);
    let radii = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        0.18 * ((row - 1.0) * 0.045).exp()
    }));
    let potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.18 + 0.013 * row, 0.004 * (0.37 * row).cos())
    }));
    let large_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.006 * (0.42 * row).sin(), -0.003 * (0.28 * row).cos())
    }));
    let small_exchange = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(-0.004 * (0.31 * row).cos(), 0.0025 * (0.53 * row).sin())
    }));
    let c3_potential = Array1::from_iter((1..=active_len).map(|row| {
        let row = row as Real;
        Complex::new(0.021 + 0.002 * row, -0.001 * (0.19 * row).sin())
    }));

    match case_id {
        1 => {
            potential_coefficients[0] = Complex::new(-0.21, 0.0);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(0.035, -0.012),
                initial_small_component: Complex::new(-0.009, 0.004),
                energy: Complex::new(-0.42, 0.018),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: -2,
                c3_scale: 0,
                start_index: 0,
                last_index: 11,
                active_len,
            }
        }
        2 => {
            potential_coefficients[0] = Complex::new(0.11, 0.0);
            potential_coefficients[3] = Complex::new(0.018, -0.004);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(-0.017, 0.028),
                initial_small_component: Complex::new(0.011, -0.006),
                energy: Complex::new(0.36, -0.027),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: 3,
                c3_scale: 1,
                start_index: 0,
                last_index: 12,
                active_len,
            }
        }
        _ => {
            potential_coefficients[0] = Complex::new(0.09, 0.0);
            potential_coefficients[3] = Complex::new(-0.015, 0.003);
            IntoutReferenceInputs {
                initial_large_component: Complex::new(0.026, 0.014),
                initial_small_component: Complex::new(-0.008, 0.017),
                energy: Complex::new(0.22, 0.041),
                potential,
                potential_coefficients,
                large_exchange,
                small_exchange,
                c3_potential,
                radii,
                kappa: -1,
                c3_scale: 1,
                start_index: 3,
                last_index: 10,
                active_len,
            }
        }
    }
}

fn diff_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Real>) {
    let potential = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        Complex::new(
            (0.21 * index).sin() + 0.03 * index,
            (0.17 * index).cos() - 0.02 * index,
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        0.15 + 0.04 * index + 0.001 * index * index
    }));
    (potential, radii)
}

fn aprd_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Complex>) {
    let real_left = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * 2.0).cos()
    }));
    let real_right = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * 3.0).sin()
    }));
    let complex_left = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    (real_left, real_right, complex_left)
}

fn muatcc_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<i32>) {
    (
        Array1::from_vec(vec![2.0, 1.5, 2.5, 1.0, 3.0]),
        Array1::from_vec(vec![0.0, 0.25, -0.10, 0.0, -0.20]),
        Array1::from_vec(vec![-1, 1, -2, 2, -3]),
    )
}

fn yzktec_reference_inputs(count: usize) -> (Array1<Complex>, Array1<Complex>, Array1<Real>) {
    let step = 0.0725;
    let source = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        Complex::new(
            (0.19 * index).sin() + 0.02 * index,
            (0.11 * index).cos() - 0.03 * index,
        )
    }));
    let coefficients = Array1::from_iter((1..=10).map(|index| {
        let index = index as Real;
        Complex::new(
            0.04 * index + (0.13 * index).cos(),
            -0.03 * index + (0.17 * index).sin(),
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|index| {
        let index = index as Real;
        0.018 * (step * (index - 1.0)).exp()
    }));
    (source, coefficients, radii)
}

struct YzkrdcReferenceInputs {
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
    partner_large_component: Array1<Complex>,
    partner_small_component: Array1<Complex>,
    partner_large_coefficients: Array1<Complex>,
    partner_small_coefficients: Array1<Complex>,
    radii: Array1<Real>,
    orbital_power: Real,
    partner_power: Real,
    step: Real,
    angular_momentum: usize,
    coefficient_count: usize,
    orbital_len: usize,
    source_len: usize,
    active_len: usize,
}

impl YzkrdcReferenceInputs {
    fn as_exchange_input(&self) -> FovrgYkZkExchangeInput<'_> {
        FovrgYkZkExchangeInput {
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            partner_large_component: self.partner_large_component.view(),
            partner_small_component: self.partner_small_component.view(),
            partner_large_coefficients: self.partner_large_coefficients.view(),
            partner_small_coefficients: self.partner_small_coefficients.view(),
            radii: self.radii.view(),
            orbital_power: self.orbital_power,
            partner_power: self.partner_power,
            step: self.step,
            angular_momentum: self.angular_momentum,
            coefficient_count: self.coefficient_count,
            orbital_len: self.orbital_len,
            source_len: self.source_len,
            active_len: self.active_len,
        }
    }
}

fn yzkrdc_reference_inputs(count: usize) -> YzkrdcReferenceInputs {
    let step = 0.0725;
    let orbital_column = 2.0;
    let large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.05 * row * orbital_column).sin() + 0.001 * (row + orbital_column)
    }));
    let small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.04 * row * orbital_column).cos() - 0.002 * (row - orbital_column)
    }));
    let large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * orbital_column).cos()
    }));
    let small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * orbital_column).sin()
    }));
    let partner_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.19 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let partner_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.07 * row).cos() - 0.01 * row,
            (0.23 * row).sin() + 0.015 * row,
        )
    }));
    let partner_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let partner_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    YzkrdcReferenceInputs {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        partner_large_component,
        partner_small_component,
        partner_large_coefficients,
        partner_small_coefficients,
        radii,
        orbital_power: 0.65 + 0.08 * orbital_column,
        partner_power: 1.35,
        step,
        angular_momentum: 2,
        coefficient_count: 6,
        orbital_len: 9,
        source_len: 9,
        active_len: count,
    }
}

struct DsordcReferenceInputs {
    large_integrand: Array1<Complex>,
    small_integrand: Array1<Complex>,
    large_integrand_coefficients: Array1<Complex>,
    small_integrand_coefficients: Array1<Complex>,
    large_component: Array1<Real>,
    small_component: Array1<Real>,
    large_coefficients: Array1<Real>,
    small_coefficients: Array1<Real>,
    radii: Array1<Real>,
    integrand_power: Real,
    orbital_power: Real,
    step: Real,
    coefficient_count: usize,
    active_len: usize,
}

impl DsordcReferenceInputs {
    fn as_overlap_input(&self) -> FovrgOverlapIntegralInput<'_> {
        FovrgOverlapIntegralInput {
            large_integrand: self.large_integrand.view(),
            small_integrand: self.small_integrand.view(),
            large_integrand_coefficients: self.large_integrand_coefficients.view(),
            small_integrand_coefficients: self.small_integrand_coefficients.view(),
            large_component: self.large_component.view(),
            small_component: self.small_component.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            radii: self.radii.view(),
            integrand_power: self.integrand_power,
            orbital_power: self.orbital_power,
            step: self.step,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
        }
    }
}

fn dsordc_reference_inputs(count: usize) -> DsordcReferenceInputs {
    let step = 0.0725;
    let orbital = 3.0;
    let large_integrand = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let small_integrand = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let large_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let small_integrand_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
    }));
    let small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
    }));
    let large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    }));
    let small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    DsordcReferenceInputs {
        large_integrand,
        small_integrand,
        large_integrand_coefficients,
        small_integrand_coefficients,
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        radii,
        integrand_power: 1.35,
        orbital_power: 0.45 + 0.06 * orbital,
        step,
        coefficient_count: 6,
        active_len: count,
    }
}

struct OrtdacReferenceInputs {
    target_large_component: Array1<Complex>,
    target_small_component: Array1<Complex>,
    target_large_coefficients: Array1<Complex>,
    target_small_coefficients: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    orbital_powers: Array1<Real>,
    radii: Array1<Real>,
    target_power: Real,
    target_kappa: i32,
    step: Real,
    coefficient_count: usize,
    active_len: usize,
    bound_orbital_count: usize,
}

impl OrtdacReferenceInputs {
    fn as_orthogonalization_input(&self) -> FovrgOrthogonalizationInput<'_> {
        FovrgOrthogonalizationInput {
            target_large_component: self.target_large_component.view(),
            target_small_component: self.target_small_component.view(),
            target_large_coefficients: self.target_large_coefficients.view(),
            target_small_coefficients: self.target_small_coefficients.view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            orbital_powers: self.orbital_powers.view(),
            radii: self.radii.view(),
            target_power: self.target_power,
            target_kappa: self.target_kappa,
            step: self.step,
            coefficient_count: self.coefficient_count,
            active_len: self.active_len,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn ortdac_reference_inputs(count: usize) -> OrtdacReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let target_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let target_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    OrtdacReferenceInputs {
        target_large_component,
        target_small_component,
        target_large_coefficients,
        target_small_coefficients,
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        electron_counts: Array1::from_vec(vec![1.2, 1.4, 0.0, 2.0]),
        kappa: Array1::from_vec(vec![-2, 1, -2, -2]),
        orbital_powers: Array1::from_iter((1..=bound_orbitals).map(|orbital| {
            let orbital = orbital as Real;
            0.45 + 0.06 * orbital
        })),
        radii,
        target_power: 0.45 + 0.06 * 5.0,
        target_kappa: -2,
        step,
        coefficient_count: 6,
        active_len: count,
        bound_orbital_count: bound_orbitals,
    }
}

struct PotexReferenceInputs {
    target_large_component: Array1<Complex>,
    target_small_component: Array1<Complex>,
    target_large_coefficients: Array1<Complex>,
    target_small_coefficients: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    angular_coefficients: Array2<Real>,
    orbital_powers: Array1<Real>,
    kappa: Array1<i32>,
    orbital_lengths: Array1<usize>,
    normalization: Array1<Real>,
    radii: Array1<Real>,
    target_power: Real,
    target_kappa: i32,
    target_normalization: Real,
    speed_of_light: Real,
    step: Real,
    coefficient_count: usize,
    source_len: usize,
    active_len: usize,
    radial_output_count: usize,
    bound_orbital_count: usize,
}

impl PotexReferenceInputs {
    fn as_exchange_potential_input(&self) -> FovrgExchangePotentialInput<'_> {
        FovrgExchangePotentialInput {
            target_large_component: self.target_large_component.view(),
            target_small_component: self.target_small_component.view(),
            target_large_coefficients: self.target_large_coefficients.view(),
            target_small_coefficients: self.target_small_coefficients.view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            angular_coefficients: self.angular_coefficients.view(),
            orbital_powers: self.orbital_powers.view(),
            kappa: self.kappa.view(),
            orbital_lengths: self.orbital_lengths.view(),
            normalization: self.normalization.view(),
            radii: self.radii.view(),
            target_power: self.target_power,
            target_kappa: self.target_kappa,
            target_normalization: self.target_normalization,
            speed_of_light: self.speed_of_light,
            step: self.step,
            coefficient_count: self.coefficient_count,
            source_len: self.source_len,
            active_len: self.active_len,
            radial_output_count: self.radial_output_count,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn potex_reference_inputs(count: usize) -> PotexReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let target_large_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.17 * row).sin() + 0.02 * row,
            (0.11 * row).cos() - 0.03 * row,
        )
    }));
    let target_small_component = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(
            (0.09 * row).cos() - 0.01 * row,
            (0.21 * row).sin() + 0.015 * row,
        )
    }));
    let target_large_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            0.04 * row + (0.13 * row).cos(),
            -0.03 * row + (0.17 * row).sin(),
        )
    }));
    let target_small_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        Complex::new(
            -0.02 * row + (0.09 * row).sin(),
            0.025 * row + (0.12 * row).cos(),
        )
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.05 * row * orbital).sin() + 0.001 * (row + orbital)
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            (0.04 * row * orbital).cos() - 0.002 * (row - orbital)
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let mut angular_coefficients = Array2::zeros((bound_orbitals, 5));
    angular_coefficients[(0, 0)] = 0.31;
    angular_coefficients[(1, 0)] = -0.18;
    angular_coefficients[(2, 0)] = 0.27;
    angular_coefficients[(2, 1)] = -0.11;
    angular_coefficients[(3, 0)] = 0.19;
    angular_coefficients[(3, 1)] = 0.07;
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    PotexReferenceInputs {
        target_large_component,
        target_small_component,
        target_large_coefficients,
        target_small_coefficients,
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        angular_coefficients,
        orbital_powers: Array1::from_vec(vec![0.51, 0.57, 0.63, 0.69]),
        kappa: Array1::from_vec(vec![-1, 1, -2, 2]),
        orbital_lengths: Array1::from_vec(vec![9, 8, 7, 9]),
        normalization: Array1::from_vec(vec![1.01, 1.02, 1.03, 1.04]),
        radii,
        target_power: 0.75,
        target_kappa: -2,
        target_normalization: 1.08,
        speed_of_light: 137.035_999_084,
        step,
        coefficient_count: 6,
        source_len: 9,
        active_len: count,
        radial_output_count: 7,
        bound_orbital_count: bound_orbitals,
    }
}

struct PotdvpReferenceInputs {
    nuclear_coefficients: Array1<Real>,
    large_coefficients: Array2<Real>,
    small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    kappa: Array1<i32>,
    normalization: Array1<Real>,
    radii: Array1<Real>,
    speed_of_light: Real,
    coefficient_count: usize,
    orbital_count: usize,
}

impl PotdvpReferenceInputs {
    fn as_potential_input(&self) -> FovrgPotentialDevelopmentInput<'_> {
        FovrgPotentialDevelopmentInput {
            nuclear_coefficients: self.nuclear_coefficients.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            kappa: self.kappa.view(),
            normalization: self.normalization.view(),
            radii: self.radii.view(),
            speed_of_light: self.speed_of_light,
            coefficient_count: self.coefficient_count,
            orbital_count: self.orbital_count,
        }
    }
}

fn potdvp_reference_inputs(count: usize) -> PotdvpReferenceInputs {
    let step = 0.0725;
    let bound_orbitals = 4;
    let large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.02 * row + (0.03 * row * orbital).cos()
    });
    let small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.015 * row + (0.025 * row * orbital).sin()
    });
    let nuclear_coefficients = Array1::from_iter((1..=10).map(|row| {
        let row = row as Real;
        -0.35 + 0.045 * row + 0.002 * row * row
    }));
    let electron_counts = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
        let orbital = orbital as Real;
        0.45 * orbital + 0.1
    }));
    let kappa = Array1::from_vec(vec![-1, 1, -2, 3]);
    let normalization = Array1::from_iter((1..=bound_orbitals).map(|orbital| {
        let orbital = orbital as Real;
        1.0 + 0.013 * orbital
    }));
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        0.018 * (step * (row - 1.0)).exp()
    }));

    PotdvpReferenceInputs {
        nuclear_coefficients,
        large_coefficients,
        small_coefficients,
        electron_counts,
        kappa,
        normalization,
        radii,
        speed_of_light: 137.035_999_084,
        coefficient_count: 8,
        orbital_count: 5,
    }
}

fn assert_complex_close(actual: Complex, expected_re: Real, expected_im: Real, tolerance: Real) {
    assert_close(actual.re, expected_re, tolerance);
    assert_close(actual.im, expected_im, tolerance);
}

fn assert_real_matrix_close<const ROWS: usize, const COLS: usize>(
    actual: &Array2<Real>,
    expected: &[[Real; COLS]; ROWS],
    tolerance: Real,
) {
    assert_eq!(actual.shape(), &[ROWS, COLS]);
    for row in 0..ROWS {
        for column in 0..COLS {
            assert_close(actual[(row, column)], expected[row][column], tolerance);
        }
    }
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}
