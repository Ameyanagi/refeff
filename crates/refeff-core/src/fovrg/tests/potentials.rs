use super::*;

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
