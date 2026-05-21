use super::*;

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
