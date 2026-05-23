use super::{support::*, *};

#[test]
fn solution_normalization_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(1.25, -0.4),
    })?;

    assert_complex_close(
        normalization.small_component_factor,
        -0.001_459_482_078_780_620_7,
        -0.001_824_332_682_938_356_4,
        1.0e-16,
    );
    assert_complex_close(
        normalization.relativistic_scale,
        1.000_000_599_040_804_3,
        -0.000_002_662_585_641_506_650_3,
        1.0e-16,
    );
    assert_complex_close(
        normalization.regular_solution_scale,
        0.725_690_457_959_513_5,
        0.232_218_816_478_531_07,
        1.0e-16,
    );

    let zero_amplitude = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(0.0, 0.0),
    })?;
    assert_complex_close(zero_amplitude.regular_solution_scale, 0.0, 0.0, 1.0e-16);
    Ok(())
}

#[test]
fn irregular_initial_condition_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let input = ScreenIrregularInitialConditionInput {
        muffin_tin_radius: 1.7,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.8, 0.1),
        neumann_l: Complex::new(-0.3, 0.05),
        bessel_j_l_plus_1: Complex::new(0.25, -0.03),
        neumann_l_plus_1: Complex::new(-0.6, 0.2),
        hankel_l: Complex::new(0.1, 0.7),
        hankel_l_plus_1: Complex::new(-0.2, 0.3),
        use_hankel_boundary: false,
    };

    let standing = screen_irregular_initial_condition(input)?;
    assert_complex_close(
        standing.large_component,
        -0.215_795_629_731_268_06,
        -0.025_994_455_746_676_352,
        1.0e-16,
    );
    assert_complex_close(
        standing.small_component,
        0.001_838_866_245_442_668,
        0.001_316_132_001_240_697_2,
        1.0e-17,
    );

    let hankel = screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
        use_hankel_boundary: true,
        ..input
    })?;
    assert_complex_close(
        hankel.large_component,
        -0.077_143_175_772_786_6,
        1.326_264_690_969_657_8,
        1.0e-15,
    );
    assert_complex_close(
        hankel.small_component,
        0.001_572_486_508_374_408_2,
        0.000_178_855_217_613_778_5,
        1.0e-17,
    );
    Ok(())
}

#[test]
fn irregular_wronskian_scale_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let scale = screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        regular_large_at_match: Complex::new(0.3, 0.2),
        regular_small_at_match: Complex::new(-0.01, 0.04),
        irregular_large_at_match: Complex::new(0.7, -0.2),
        irregular_small_at_match: Complex::new(0.02, 0.03),
    })?;

    assert_complex_close(
        scale.phase_factor,
        1.083_141_079_608_063_2,
        0.219_563_566_708_252_36,
        1.0e-15,
    );
    assert_complex_close(
        scale.denominator,
        -0.726_137_142_242_051_2,
        5.106_772_750_294_418,
        1.0e-14,
    );
    assert_complex_close(
        scale.reciprocal_wave_scale,
        -0.260_696_573_980_254_4,
        -0.153_973_620_782_305_84,
        1.0e-15,
    );
    assert_complex_close(
        scale.irregular_solution_scale,
        -0.248_564_171_233_149_1,
        -0.224_014_623_457_035_68,
        1.0e-15,
    );

    let zero = screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.0, 0.0),
        regular_large_at_match: Complex::new(0.0, 0.0),
        regular_small_at_match: Complex::new(0.0, 0.0),
        irregular_large_at_match: Complex::new(0.0, 0.0),
        irregular_small_at_match: Complex::new(0.0, 0.0),
    })?;
    assert_complex_close(zero.reciprocal_wave_scale, 0.0, 0.0, 1.0e-16);
    assert_complex_close(zero.irregular_solution_scale, 0.0, 0.0, 1.0e-16);
    Ok(())
}

#[test]
fn exact_radial_continuation_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let continued = screen_exact_radial_continuation(ScreenExactRadialContinuationInput {
        radius: 2.0,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.6, 0.2),
        neumann_l: Complex::new(-0.4, 0.1),
        bessel_j_l_plus_1: Complex::new(0.3, 0.05),
        neumann_l_plus_1: Complex::new(-0.2, 0.2),
        hankel_l: Complex::new(0.1, 0.7),
        hankel_l_plus_1: Complex::new(-0.2, 0.3),
    })?;

    assert_complex_close(
        continued.regular_large_component,
        1.314_103_542_373_494,
        0.299_396_383_930_798,
        1.0e-15,
    );
    assert_complex_close(
        continued.regular_small_component,
        -0.000_934_743_791_234_705_6,
        -0.001_135_887_639_152_749_7,
        1.0e-17,
    );
    assert_complex_close(
        continued.irregular_large_component,
        -0.090_756_677_379_748_95,
        1.560_311_401_140_773_7,
        1.0e-15,
    );
    assert_complex_close(
        continued.irregular_small_component,
        0.001_849_984_127_499_303_5,
        0.000_210_417_903_075_033_55,
        1.0e-17,
    );
    Ok(())
}
