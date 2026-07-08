use super::*;

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
fn flat_potential_propagation_uses_stable_hankel_basis_for_large_imaginary_argument()
-> Result<(), FovrgError> {
    let input = FovrgFlatPotentialInput {
        start_radius: 7.6,
        end_radius: 8.2,
        large_component: Complex::new(0.43, -0.17),
        small_component: Complex::new(-0.013, 0.021),
        energy: Complex::new(-32.0, 0.10),
        average_potential: Complex::new(0.0, 0.0),
        kappa: -1,
    };

    let direct = fovrg_flat_potential_propagate(input)?;
    let midpoint = 7.9;
    let first = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
        end_radius: midpoint,
        ..input
    })?;
    let second = fovrg_flat_potential_propagate(FovrgFlatPotentialInput {
        start_radius: midpoint,
        large_component: first.large_component,
        small_component: first.small_component,
        ..input
    })?;

    assert!(direct.large_component.re.is_finite());
    assert!(direct.large_component.im.is_finite());
    assert!(direct.small_component.re.is_finite());
    assert!(direct.small_component.im.is_finite());
    assert!(direct.large_component.norm() > 0.0);
    assert!(direct.small_component.norm() > 0.0);
    assert_complex_close(
        second.large_component,
        direct.large_component.re,
        direct.large_component.im,
        2.0e-7 * direct.large_component.norm().max(1.0),
    );
    assert_complex_close(
        second.small_component,
        direct.small_component.re,
        direct.small_component.im,
        2.0e-7 * direct.small_component.norm().max(1.0),
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
fn photoelectron_retained_len_extends_to_cover_match_history() -> Result<(), FovrgError> {
    assert_eq!(fovrg_photoelectron_retained_len(0.05, 251, 4)?, 223);
    assert_eq!(fovrg_photoelectron_retained_len(0.05, 251, 220)?, 226);
    assert_eq!(fovrg_photoelectron_retained_len(0.05, 224, 220)?, 224);
    Ok(())
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
fn dirac_solver_skips_c3_potential_when_c3_scale_is_zero() -> Result<(), FovrgError> {
    let input = dfovrg_reference_inputs(false);

    let disabled = fovrg_dirac_solver(input.to_input())?;
    assert_eq!(disabled.c3_potential.len(), disabled.active_len);
    assert!(
        disabled
            .c3_potential
            .iter()
            .all(|value| *value == Complex::new(0.0, 0.0))
    );

    let enabled = fovrg_dirac_solver(FovrgDiracSolverInput {
        c3_scale: 1,
        ..input.to_input()
    })?;
    assert_eq!(enabled.c3_potential.len(), enabled.active_len);
    assert!(
        enabled
            .c3_potential
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(enabled.c3_potential.iter().any(|value| value.norm() > 0.0));

    let prepared_c3 = fovrg_dirac_solver_c3_potential(FovrgDiracSolverInput {
        c3_scale: 1,
        ..input.to_input()
    })?;
    for row in 0..enabled.active_len {
        assert_complex_close(
            prepared_c3[row],
            enabled.c3_potential[row].re,
            enabled.c3_potential[row].im,
            1.0e-13,
        );
    }

    let prepared_solution = fovrg_dirac_solver_with_c3_potential(
        FovrgDiracSolverInput {
            c3_scale: 1,
            ..input.to_input()
        },
        prepared_c3.view(),
    )?;
    assert_complex_close(
        prepared_solution.muffin_tin_large_component,
        enabled.muffin_tin_large_component.re,
        enabled.muffin_tin_large_component.im,
        1.0e-13,
    );
    assert_complex_close(
        prepared_solution.muffin_tin_small_component,
        enabled.muffin_tin_small_component.re,
        enabled.muffin_tin_small_component.im,
        1.0e-13,
    );
    Ok(())
}

#[test]
fn dirac_solver_feeds_xsph_regular_phase_channel() -> Result<(), crate::XsphError> {
    let input = dfovrg_reference_inputs(false);
    let wave_number = Complex::new(0.89, 0.035);
    let channel = crate::xsph_regular_phase_channel(input.to_input(), wave_number)?;
    let regular_solution = fovrg_dirac_solver(input.to_input())?;
    let expected = crate::xsph_regular_phase(crate::XsphRegularPhaseInput {
        muffin_tin_radius: input.muffin_tin_radius,
        wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: input.target_kappa,
    })?;

    assert_eq!(
        channel.regular_solution.active_len,
        regular_solution.active_len
    );
    assert_complex_close(
        channel.regular_solution.muffin_tin_large_component,
        regular_solution.muffin_tin_large_component.re,
        regular_solution.muffin_tin_large_component.im,
        1.0e-12,
    );
    assert_complex_close(
        channel.regular_solution.muffin_tin_small_component,
        regular_solution.muffin_tin_small_component.re,
        regular_solution.muffin_tin_small_component.im,
        1.0e-12,
    );
    assert_complex_close(
        channel.phase.phase_shift,
        expected.phase_shift.re,
        expected.phase_shift.im,
        1.0e-12,
    );
    assert_complex_close(
        channel.phase.phase_amplitude,
        expected.phase_amplitude.re,
        expected.phase_amplitude.im,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn dirac_solver_feeds_xsph_xsect_regular_channel() -> Result<(), crate::XsphError> {
    let input = dfovrg_reference_inputs(false);
    let wave_number = Complex::new(0.89, 0.035);
    let channel = crate::xsph_xsect_regular_channel(crate::XsphXsectRegularChannelInput {
        solver: input.to_input(),
        wave_number,
    })?;
    let regular_solution = fovrg_dirac_solver(input.to_input())?;
    let phase = crate::xsph_regular_phase(crate::XsphRegularPhaseInput {
        muffin_tin_radius: input.muffin_tin_radius,
        wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: input.target_kappa,
    })?;
    let active_len = regular_solution.target_last_index + 1;
    let normalized = crate::xsph_xsect_regular_solution(crate::XsphXsectRegularSolutionInput {
        wave_number,
        phase_amplitude: phase.phase_amplitude,
        final_kappa: input.target_kappa,
        regular_large: regular_solution.large_component.view(),
        regular_small: regular_solution.small_component.view(),
        active_len,
    })?;

    assert_eq!(
        channel.regular_solution.target_last_index,
        regular_solution.target_last_index
    );
    assert_eq!(channel.normalized_solution.regular_large.len(), active_len);
    assert_eq!(channel.normalized_solution.regular_small.len(), active_len);
    assert_complex_close(
        channel.phase.phase_shift,
        phase.phase_shift.re,
        phase.phase_shift.im,
        1.0e-12,
    );
    assert_complex_close(
        channel.phase.phase_amplitude,
        phase.phase_amplitude.re,
        phase.phase_amplitude.im,
        1.0e-12,
    );
    assert_complex_close(
        channel.normalized_solution.regular_solution_scale,
        normalized.regular_solution_scale.re,
        normalized.regular_solution_scale.im,
        1.0e-12,
    );
    for index in 0..active_len {
        assert_complex_close(
            channel.normalized_solution.regular_large[index],
            normalized.regular_large[index].re,
            normalized.regular_large[index].im,
            1.0e-12,
        );
        assert_complex_close(
            channel.normalized_solution.regular_small[index],
            normalized.regular_small[index].re,
            normalized.regular_small[index].im,
            1.0e-12,
        );
    }
    Ok(())
}

#[test]
fn dirac_solver_feeds_xsph_xsect_irregular_channel() -> Result<(), crate::XsphError> {
    let input = dfovrg_reference_inputs(false);
    let wave_number = Complex::new(0.89, 0.035);
    let regular_channel = crate::xsph_xsect_regular_channel(crate::XsphXsectRegularChannelInput {
        solver: input.to_input(),
        wave_number,
    })?;
    let irregular_channel =
        crate::xsph_xsect_irregular_channel(crate::XsphXsectIrregularChannelInput {
            solver: input.to_input(),
            wave_number,
            regular_channel: &regular_channel,
        })?;
    let expected_initial = crate::xsph_xsect_irregular_initial_condition(
        crate::XsphXsectIrregularInitialConditionInput {
            muffin_tin_radius: input.muffin_tin_radius,
            phase_shift: regular_channel.phase.phase_shift,
            wave_number,
            final_kappa: input.target_kappa,
            bessel_j_l: regular_channel.phase.bessel_j_large,
            neumann_l: regular_channel.phase.neumann_large,
            bessel_j_l_plus_1: regular_channel.phase.bessel_j_small,
            neumann_l_plus_1: regular_channel.phase.neumann_small,
        },
    )?;
    let irregular_input = FovrgDiracSolverInput {
        irregular: true,
        muffin_tin_large_component: expected_initial.large_component,
        muffin_tin_small_component: expected_initial.small_component,
        ..input.to_input()
    };
    let expected_irregular_solution = fovrg_dirac_solver(irregular_input)?;
    let active_len = regular_channel.normalized_solution.regular_large.len();
    let expected_transform =
        crate::xsph_xsect_irregular_transform(crate::XsphXsectIrregularTransformInput {
            phase_shift: regular_channel.phase.phase_shift,
            regular_large: regular_channel.normalized_solution.regular_large.view(),
            regular_small: regular_channel.normalized_solution.regular_small.view(),
            irregular_large: expected_irregular_solution.large_component.view(),
            irregular_small: expected_irregular_solution.small_component.view(),
            active_len,
        })?;

    assert_complex_close(
        irregular_channel.initial_condition.large_component,
        expected_initial.large_component.re,
        expected_initial.large_component.im,
        1.0e-12,
    );
    assert_complex_close(
        irregular_channel.initial_condition.small_component,
        expected_initial.small_component.re,
        expected_initial.small_component.im,
        1.0e-12,
    );
    assert_eq!(
        irregular_channel.irregular_solution.target_last_index,
        expected_irregular_solution.target_last_index
    );
    assert_eq!(
        irregular_channel.transformed_solution.irregular_large.len(),
        active_len
    );
    assert_eq!(
        irregular_channel.transformed_solution.irregular_small.len(),
        active_len
    );
    assert_complex_close(
        irregular_channel.transformed_solution.phase_factor,
        expected_transform.phase_factor.re,
        expected_transform.phase_factor.im,
        1.0e-12,
    );
    for index in 0..active_len {
        assert_complex_close(
            irregular_channel.transformed_solution.irregular_large[index],
            expected_transform.irregular_large[index].re,
            expected_transform.irregular_large[index].im,
            1.0e-12,
        );
        assert_complex_close(
            irregular_channel.transformed_solution.irregular_small[index],
            expected_transform.irregular_small[index].re,
            expected_transform.irregular_small[index].im,
            1.0e-12,
        );
    }
    Ok(())
}

#[test]
fn dirac_solver_feeds_xsph_xsect_bcoef_nonstandard_channel_row() -> Result<(), crate::XsphError> {
    let input = dfovrg_reference_inputs(false);
    let wave_number = Complex::new(0.89, 0.035);
    let regular_channel = crate::xsph_xsect_regular_channel(crate::XsphXsectRegularChannelInput {
        solver: input.to_input(),
        wave_number,
    })?;
    let irregular_channel =
        crate::xsph_xsect_irregular_channel(crate::XsphXsectIrregularChannelInput {
            solver: input.to_input(),
            wave_number,
            regular_channel: &regular_channel,
        })?;
    let active_len = regular_channel.normalized_solution.regular_large.len();
    let initial_large = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        0.021 * (0.17 * row).sin() * (-0.004 * row).exp()
    }));
    let initial_small = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        -0.014 * (0.13 * row).cos() * (-0.005 * row).exp()
    }));
    let xray_bessel = crate::xsph_xray_bessel_table(crate::XsphXrayBesselTableInput {
        photon_wave_number: 0.023,
        radii: input.radii.view(),
        active_len,
    })?;
    let transition = crate::XsphXsectTransition {
        multipole: crate::XsphTransitionMultipole::ElectricDipole,
        transition_delta: -1,
        transition_index_1based: 1,
        final_kappa: input.target_kappa,
        final_l: 1,
        multipole_order: 1,
    };
    let diagonal_weights = Array1::from_vec(vec![Complex::new(-1.0 / 3.0, 0.0); 8]);
    let reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let phase_shifts = Array1::<Complex>::zeros(8);

    let result = crate::xsph_xsect_bcoef_nonstandard_channel_row(
        crate::XsphXsectBcoefNonstandardChannelRowInput {
            transition,
            selected_higher_multipole: None,
            initial_kappa: -1,
            initial_large: initial_large.view(),
            initial_small: initial_small.view(),
            regular_channel: &regular_channel,
            irregular_channel: &irregular_channel,
            xray_bessel: xray_bessel.values.view(),
            radii: input.radii.view(),
            log_step: input.to_input().step,
            diagonal_weights: diagonal_weights.view(),
            spectrum_norm: 0.42,
            cross_section: Complex::new(0.08, -0.03),
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        },
    )?;

    let expected_reduced = crate::xsph_radial_integral(crate::XsphRadialIntegralInput {
        mode: crate::XsphRadialIntegralMode::RelativisticMatrixElement,
        multipole: transition.multipole,
        initial_kappa: -1,
        final_kappa: transition.final_kappa,
        initial_large: initial_large.view(),
        initial_small: initial_small.view(),
        final_large_regular: regular_channel.normalized_solution.regular_large.view(),
        final_small_regular: regular_channel.normalized_solution.regular_small.view(),
        xray_bessel: xray_bessel.values.view(),
        radii: input.radii.view(),
        log_step: input.to_input().step,
        active_len,
    })?;
    let expected_central =
        crate::xsph_radial_cross_integral(crate::XsphRadialCrossIntegralInput {
            mode: crate::XsphRadialIntegralMode::RelativisticMatrixElement,
            branch: crate::XsphRadialCrossIntegralBranch::CurrentRegularAndIrregular,
            multipole: transition.multipole,
            initial_kappa: -1,
            final_kappa: transition.final_kappa,
            initial_large: initial_large.view(),
            initial_small: initial_small.view(),
            final_large_regular: regular_channel.normalized_solution.regular_large.view(),
            final_small_regular: regular_channel.normalized_solution.regular_small.view(),
            final_large_irregular: irregular_channel
                .transformed_solution
                .irregular_large
                .view(),
            final_small_irregular: irregular_channel
                .transformed_solution
                .irregular_small
                .view(),
            xray_bessel: xray_bessel.values.view(),
            radii: input.radii.view(),
            log_step: input.to_input().step,
            active_len,
        })?;
    let expected_row =
        crate::xsph_xsect_bcoef_ordinary_row(crate::XsphXsectBcoefOrdinaryRowInput {
            multipole: transition.multipole,
            selected_higher_multipole: None,
            transition_index_1based: transition.transition_index_1based,
            diagonal_weights: diagonal_weights.view(),
            reduced_matrix_integral: expected_reduced.value,
            central_cross_integral: expected_central.value,
            phase_shift: regular_channel.phase.phase_shift,
            spectrum_norm: 0.42,
            cross_section: Complex::new(0.08, -0.03),
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
        })?;

    assert_eq!(result.reduced_radial_pass.feff_ifl, 1);
    assert_eq!(result.central_radial_pass.feff_ifl, 2);
    assert_complex_close(
        result.reduced_radial_integral.value,
        expected_reduced.value.re,
        expected_reduced.value.im,
        1.0e-12,
    );
    assert_complex_close(
        result.central_cross_integral.value,
        expected_central.value.re,
        expected_central.value.im,
        1.0e-12,
    );
    assert_complex_close(
        result.row.cross_section,
        expected_row.cross_section.re,
        expected_row.cross_section.im,
        1.0e-12,
    );
    assert_close(
        result.row.spectrum_norm,
        expected_row.spectrum_norm,
        1.0e-12,
    );
    assert_complex_close(
        result.row.reduced_matrix_elements[0],
        expected_row.reduced_matrix_elements[0].re,
        expected_row.reduced_matrix_elements[0].im,
        1.0e-12,
    );
    assert_complex_close(
        result.row.phase_shifts[0],
        expected_row.phase_shifts[0].re,
        expected_row.phase_shifts[0].im,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn dirac_solver_feeds_xsph_xsect_bcoef_nonstandard_energy_row() -> Result<(), crate::XsphError> {
    let input = dfovrg_reference_inputs(false);
    let wave_number = Complex::new(0.89, 0.035);
    let regular_channel = crate::xsph_xsect_regular_channel(crate::XsphXsectRegularChannelInput {
        solver: input.to_input(),
        wave_number,
    })?;
    let irregular_channel =
        crate::xsph_xsect_irregular_channel(crate::XsphXsectIrregularChannelInput {
            solver: input.to_input(),
            wave_number,
            regular_channel: &regular_channel,
        })?;
    let active_len = regular_channel.normalized_solution.regular_large.len();
    let initial_large = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        0.021 * (0.17 * row).sin() * (-0.004 * row).exp()
    }));
    let initial_small = Array1::from_iter((0..active_len).map(|index| {
        let row = (index + 1) as Real;
        -0.014 * (0.13 * row).cos() * (-0.005 * row).exp()
    }));
    let xray_bessel = crate::xsph_xray_bessel_table(crate::XsphXrayBesselTableInput {
        photon_wave_number: 0.023,
        radii: input.radii.view(),
        active_len,
    })?;
    let transitions = vec![
        crate::XsphXsectTransition {
            multipole: crate::XsphTransitionMultipole::ElectricDipole,
            transition_delta: -1,
            transition_index_1based: 1,
            final_kappa: input.target_kappa,
            final_l: 1,
            multipole_order: 1,
        },
        crate::XsphXsectTransition {
            multipole: crate::XsphTransitionMultipole::ElectricDipole,
            transition_delta: 0,
            transition_index_1based: 2,
            final_kappa: input.target_kappa,
            final_l: 1,
            multipole_order: 1,
        },
    ];
    let regular_channels = vec![regular_channel.clone(), regular_channel.clone()];
    let irregular_channels = vec![irregular_channel.clone(), irregular_channel.clone()];
    let diagonal_weights = Array1::from_vec(vec![Complex::new(-1.0 / 3.0, 0.0); 8]);
    let orbital_l = Array1::from_vec(vec![1, 1, 0, 0, 0, 0, 0, 0]);
    let trace_weights = Array2::<Complex>::zeros((8, 8));

    let result = crate::xsph_xsect_bcoef_nonstandard_energy_row(
        crate::XsphXsectBcoefNonstandardEnergyRowInput {
            transitions: &transitions,
            regular_channels: &regular_channels,
            irregular_channels: &irregular_channels,
            selected_higher_multipole: None,
            initial_kappa: -1,
            initial_large: initial_large.view(),
            initial_small: initial_small.view(),
            xray_bessel: xray_bessel.values.view(),
            radii: input.radii.view(),
            log_step: input.to_input().step,
            diagonal_weights: diagonal_weights.view(),
            spin_polarized_cross_terms: false,
            orbital_l: orbital_l.view(),
            trace_weights: trace_weights.view(),
            spin_orbit_removed_regular_channels: None,
            spin_orbit_removed_irregular_channels: None,
            photon_energy: 0.31,
            wave_number,
            active_channel_count: 2,
        },
    )?;

    let mut spectrum_norm = 0.0;
    let mut cross_section = Complex::new(0.0, 0.0);
    let mut reduced_matrix_elements = Array1::<Complex>::zeros(8);
    let mut phase_shifts = Array1::<Complex>::zeros(8);
    for transition in transitions.iter().copied() {
        let row = crate::xsph_xsect_bcoef_nonstandard_channel_row(
            crate::XsphXsectBcoefNonstandardChannelRowInput {
                transition,
                selected_higher_multipole: None,
                initial_kappa: -1,
                initial_large: initial_large.view(),
                initial_small: initial_small.view(),
                regular_channel: &regular_channel,
                irregular_channel: &irregular_channel,
                xray_bessel: xray_bessel.values.view(),
                radii: input.radii.view(),
                log_step: input.to_input().step,
                diagonal_weights: diagonal_weights.view(),
                spectrum_norm,
                cross_section,
                reduced_matrix_elements: reduced_matrix_elements.view(),
                phase_shifts: phase_shifts.view(),
            },
        )?;
        spectrum_norm = row.row.spectrum_norm;
        cross_section = row.row.cross_section;
        reduced_matrix_elements.assign(&row.row.reduced_matrix_elements);
        phase_shifts.assign(&row.row.phase_shifts);
    }
    let expected_output =
        crate::xsph_xsect_output_normalization(crate::XsphXsectOutputNormalizationInput {
            photon_energy: 0.31,
            wave_number,
            spectrum_norm,
            cross_section,
            reduced_matrix_elements: reduced_matrix_elements.view(),
            phase_shifts: phase_shifts.view(),
            active_channel_count: 2,
        })?;

    assert_eq!(result.transition_rows.len(), 2);
    assert!(result.cross_term_updates.is_empty());
    assert_close(result.unnormalized_spectrum_norm, spectrum_norm, 1.0e-12);
    assert_complex_close(
        result.unnormalized_cross_section,
        cross_section.re,
        cross_section.im,
        1.0e-12,
    );
    assert_close(
        result.output_normalization.spectrum_norm,
        expected_output.spectrum_norm,
        1.0e-12,
    );
    assert_complex_close(
        result.output_normalization.cross_section,
        expected_output.cross_section.re,
        expected_output.cross_section.im,
        1.0e-12,
    );
    for index in 0..2 {
        assert_complex_close(
            result.output_normalization.reduced_matrix_elements[index],
            expected_output.reduced_matrix_elements[index].re,
            expected_output.reduced_matrix_elements[index].im,
            1.0e-12,
        );
    }

    let retry_solver = FovrgDiracSolverInput {
        c3_scale: 1,
        ..input.to_input()
    };
    let retry_regular_channel =
        crate::xsph_xsect_regular_channel(crate::XsphXsectRegularChannelInput {
            solver: retry_solver,
            wave_number,
        })?;
    let retry_irregular_channel =
        crate::xsph_xsect_irregular_channel(crate::XsphXsectIrregularChannelInput {
            solver: retry_solver,
            wave_number,
            regular_channel: &retry_regular_channel,
        })?;
    let retry_regular_channels = vec![retry_regular_channel.clone(), retry_regular_channel];
    let retry_irregular_channels = vec![retry_irregular_channel.clone(), retry_irregular_channel];
    let mut off_diagonal_trace_weights = trace_weights;
    off_diagonal_trace_weights[(0, 1)] = Complex::new(0.6, -0.1);
    off_diagonal_trace_weights[(1, 0)] = Complex::new(-0.2, 0.3);
    let cross_term_result = crate::xsph_xsect_bcoef_nonstandard_energy_row(
        crate::XsphXsectBcoefNonstandardEnergyRowInput {
            transitions: &transitions,
            regular_channels: &regular_channels,
            irregular_channels: &irregular_channels,
            selected_higher_multipole: None,
            initial_kappa: -1,
            initial_large: initial_large.view(),
            initial_small: initial_small.view(),
            xray_bessel: xray_bessel.values.view(),
            radii: input.radii.view(),
            log_step: input.to_input().step,
            diagonal_weights: diagonal_weights.view(),
            spin_polarized_cross_terms: true,
            orbital_l: orbital_l.view(),
            trace_weights: off_diagonal_trace_weights.view(),
            spin_orbit_removed_regular_channels: Some(&retry_regular_channels),
            spin_orbit_removed_irregular_channels: Some(&retry_irregular_channels),
            photon_energy: 0.31,
            wave_number,
            active_channel_count: 2,
        },
    )?;

    assert_eq!(cross_term_result.transition_rows.len(), 2);
    assert_eq!(cross_term_result.cross_term_updates.len(), 1);
    assert_close(
        cross_term_result.unnormalized_spectrum_norm,
        spectrum_norm,
        1.0e-12,
    );
    assert_complex_close(
        cross_term_result.unnormalized_cross_section,
        cross_term_result.cross_term_updates[0].cross_section.re,
        cross_term_result.cross_term_updates[0].cross_section.im,
        1.0e-12,
    );
    assert!(
        (cross_term_result.unnormalized_cross_section - result.unnormalized_cross_section).norm()
            > 1.0e-14
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
