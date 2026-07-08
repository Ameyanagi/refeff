use super::*;

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
fn c3_potential_matches_dfovrg_vm_setup() -> Result<(), FovrgError> {
    let (potential, radii) = diff_reference_inputs(12);

    let result = fovrg_c3_potential(FovrgC3PotentialInput {
        exchange_correlation_potential: potential.view(),
        radii: radii.view(),
        target_kappa: -2,
        step: 0.0375,
        radial_match_index: 9,
        active_len: 12,
    })?;
    let derivative = fovrg_c3_derivative(FovrgC3DerivativeInput {
        potential: potential.view(),
        radii: radii.view(),
        kappa: -2,
        speed_of_light: 137.035_989_56,
        delta: 0.0375,
        active_len: 10,
    })?;

    assert_eq!(result.len(), 12);
    for row in 0..9 {
        assert_complex_close(result[row], derivative[row].re, derivative[row].im, 1.0e-14);
    }
    for row in 9..12 {
        assert_eq!(result[row], Complex::new(0.0, 0.0));
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
fn c3_potential_rejects_invalid_inputs() {
    let (potential, radii) = diff_reference_inputs(8);

    assert!(matches!(
        fovrg_c3_potential(FovrgC3PotentialInput {
            exchange_correlation_potential: potential.view(),
            radii: radii.view(),
            target_kappa: -2,
            step: 0.0375,
            radial_match_index: 8,
            active_len: 8,
        }),
        Err(FovrgError::ActiveCountOutOfRange {
            field: "active_len",
            ..
        })
    ));
    assert!(matches!(
        fovrg_c3_potential(FovrgC3PotentialInput {
            exchange_correlation_potential: potential.view(),
            radii: radii.view(),
            target_kappa: -2,
            step: 0.0,
            radial_match_index: 7,
            active_len: 8,
        }),
        Err(FovrgError::ZeroInput { name: "delta" })
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
