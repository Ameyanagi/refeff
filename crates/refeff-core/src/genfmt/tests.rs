use super::{
    CurvedWavePolynomialInput, EnergyIndependentMatrixInput, GenfmtError,
    GenfmtLegendreNormalizationInput, InitialStateRotation, InitialStateRotationInput,
    LambdaIndexInput, PathRotationInput, PolarizedScatteringAmplitudeInput,
    ScatteringAmplitudeMatrixInput, TransitionRotationInput, XStarInput, curved_wave_polynomials,
    energy_independent_transition_matrix, genfmt_legendre_normalization_table,
    initial_state_rotation, lambda_indices, path_rotation_angles,
    polarized_scattering_amplitude_matrix, scattering_amplitude_matrix, xstar,
};
use crate::{Complex, Real, legendre_normalization_table};
use ndarray::{Array1, Array2, Array3, Array4, Array6, ShapeBuilder, arr2};

fn input<'a>(
    calculation: i32,
    energy_index: usize,
    scattering_count: usize,
    initial_l: usize,
    beta_angles: &'a [f64],
    lambda_capacity: usize,
) -> LambdaIndexInput<'a> {
    LambdaIndexInput {
        calculation,
        energy_index,
        scattering_count,
        initial_l,
        beta_angles,
        lambda_capacity,
        max_m: 10,
        max_n: 10,
    }
}

#[test]
fn exact_order_matches_feff_reference() -> Result<(), GenfmtError> {
    let beta = [0.0, std::f64::consts::PI, 0.5, 2.8];
    let lambda = lambda_indices(input(2, 10, 2, 3, &beta, 40))?;

    assert_eq!(lambda.order, 2);
    assert_eq!(lambda.requested_n_max, 1);
    assert_eq!(lambda.requested_m_max, 2);
    assert_eq!(lambda.initial_l_prefix_len, 6);
    assert_eq!(lambda.max_n, 1);
    assert_eq!(lambda.max_m_plus_one, 3);
    assert!(!lambda.truncated);
    assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2, 0]);
    assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0, 1]);
    Ok(())
}

#[test]
fn single_scattering_uses_initial_l_exact_reference() -> Result<(), GenfmtError> {
    let beta = [0.3, 1.2];
    let lambda = lambda_indices(input(10, 8, 1, 2, &beta, 40))?;

    assert_eq!(lambda.order, 6);
    assert_eq!(lambda.requested_n_max, 2);
    assert_eq!(lambda.requested_m_max, 2);
    assert_eq!(lambda.initial_l_prefix_len, 15);
    assert_eq!(lambda.max_n, 2);
    assert_eq!(lambda.max_m_plus_one, 3);
    assert_eq!(
        lambda.m_indices.to_vec(),
        vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2]
    );
    assert_eq!(
        lambda.n_indices.to_vec(),
        vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn cute_linear_low_energy_matches_feff_reference() -> Result<(), GenfmtError> {
    let beta = [
        0.0,
        std::f64::consts::PI,
        0.010,
        std::f64::consts::PI - 0.010,
    ];
    let lambda = lambda_indices(input(10, 41, 2, 4, &beta, 80))?;

    assert_eq!(lambda.order, 12);
    assert_eq!(lambda.requested_n_max, 4);
    assert_eq!(lambda.requested_m_max, 4);
    assert_eq!(lambda.initial_l_prefix_len, 45);
    assert_eq!(lambda.max_n, 4);
    assert_eq!(lambda.max_m_plus_one, 5);
    assert_eq!(lambda.m_indices.len(), 45);
    assert_eq!(
        &lambda.m_indices.to_vec()[..9],
        &[0, -1, 1, -2, 2, -3, 3, -4, 4]
    );
    assert_eq!(
        &lambda.n_indices.to_vec()[36..],
        &[4, 4, 4, 4, 4, 4, 4, 4, 4]
    );
    Ok(())
}

#[test]
fn cute_nonlinear_high_energy_sorts_initial_l_prefix() -> Result<(), GenfmtError> {
    let beta = [0.0, 0.25, std::f64::consts::PI];
    let lambda = lambda_indices(input(10, 42, 2, 4, &beta, 80))?;

    assert_eq!(lambda.order, 21);
    assert_eq!(lambda.requested_n_max, 9);
    assert_eq!(lambda.requested_m_max, 3);
    assert_eq!(lambda.m_indices.len(), 70);
    assert_eq!(lambda.initial_l_prefix_len, 35);
    assert_eq!(lambda.max_n, 9);
    assert_eq!(lambda.max_m_plus_one, 4);
    assert_eq!(&lambda.n_indices.to_vec()[..7], &[0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&lambda.n_indices.to_vec()[28..35], &[4, 4, 4, 4, 4, 4, 4]);
    assert_eq!(&lambda.n_indices.to_vec()[35..42], &[5, 5, 5, 5, 5, 5, 5]);
    assert_eq!(&lambda.n_indices.to_vec()[63..], &[9, 9, 9, 9, 9, 9, 9]);
    Ok(())
}

#[test]
fn negative_calculation_decodes_requested_limits() -> Result<(), GenfmtError> {
    let beta = [0.0, 0.5];
    let lambda = lambda_indices(input(-80_205, 12, 2, 2, &beta, 80))?;

    assert_eq!(lambda.order, 7);
    assert_eq!(lambda.requested_n_max, 5);
    assert_eq!(lambda.requested_m_max, 2);
    assert_eq!(lambda.initial_l_prefix_len, 15);
    assert_eq!(lambda.max_n, 3);
    assert_eq!(lambda.max_m_plus_one, 3);
    assert_eq!(
        lambda.m_indices.to_vec(),
        vec![0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1, -2, 2, 0, -1, 1]
    );
    assert_eq!(
        lambda.n_indices.to_vec(),
        vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    );
    Ok(())
}

#[test]
fn capacity_truncation_matches_feff_reference() -> Result<(), GenfmtError> {
    let beta = [0.0, 1.0];
    let lambda = lambda_indices(input(4, 10, 2, 1, &beta, 5))?;

    assert!(lambda.truncated);
    assert_eq!(lambda.order, 4);
    assert_eq!(lambda.requested_n_max, 2);
    assert_eq!(lambda.requested_m_max, 4);
    assert_eq!(lambda.initial_l_prefix_len, 3);
    assert_eq!(lambda.max_n, 0);
    assert_eq!(lambda.max_m_plus_one, 3);
    assert_eq!(lambda.m_indices.to_vec(), vec![0, -1, 1, -2, 2]);
    assert_eq!(lambda.n_indices.to_vec(), vec![0, 0, 0, 0, 0]);
    Ok(())
}

#[test]
fn cute_calculation_rejects_nonfinite_beta() {
    let beta = [f64::NAN];

    assert!(matches!(
        lambda_indices(input(10, 42, 2, 4, &beta, 80)),
        Err(GenfmtError::NonFiniteBetaAngle { index: 0, .. })
    ));
}

#[test]
fn undefined_calculation_is_an_error_for_multiple_scattering() {
    assert_eq!(
        lambda_indices(input(11, 1, 2, 0, &[], 10)),
        Err(GenfmtError::UndefinedLambdaCalculation { calculation: 11 })
    );
}

#[test]
fn dimension_overflow_is_reported() {
    let mut bad = input(10, 42, 2, 4, &[0.25], 80);
    bad.max_n = 8;

    assert!(matches!(
        lambda_indices(bad),
        Err(GenfmtError::DimensionExceeded {
            max_n: 9,
            max_n_limit: 8,
            ..
        })
    ));
}

#[test]
fn initial_state_rotation_matches_feff_full_reference() -> Result<(), GenfmtError> {
    let rotation = initial_state_rotation(InitialStateRotationInput {
        lmaxp1: 4,
        mmaxp1: 4,
        beta_angle: 0.7,
    })?;

    assert_eq!(rotation.matrix.shape(), &[4, 7, 7]);
    assert_eq!(rotation.matrix.strides(), &[1, 4, 28]);
    assert_eq!(rotation.magnetic_offset, 3);
    assert_close(rotation_sum(&rotation), 14.508_147_433_950_487);
    assert_eq!(rotation_nonzero_count(&rotation), 84);
    assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
    assert_close(
        rotation_value(&rotation, 2, -1, -1),
        0.882_421_093_642_244_2,
    );
    assert_close(
        rotation_value(&rotation, 2, -1, 0),
        0.455_530_695_206_085_63,
    );
    assert_close(rotation_value(&rotation, 2, 0, 1), 0.455_530_695_206_085_63);
    assert_close(
        rotation_value(&rotation, 3, -2, 1),
        0.075_746_411_121_730_47,
    );
    assert_close(
        rotation_value(&rotation, 4, -3, 3),
        0.001_625_504_772_936_771_3,
    );
    assert_close(
        rotation_value(&rotation, 4, 0, 0),
        -0.028_712_995_143_227_615,
    );
    Ok(())
}

#[test]
fn initial_state_rotation_matches_feff_limited_m_reference() -> Result<(), GenfmtError> {
    let rotation = initial_state_rotation(InitialStateRotationInput {
        lmaxp1: 5,
        mmaxp1: 2,
        beta_angle: -0.4,
    })?;

    assert_eq!(rotation.matrix.shape(), &[5, 3, 3]);
    assert_eq!(rotation.matrix.strides(), &[1, 5, 15]);
    assert_eq!(rotation.magnetic_offset, 1);
    assert_close(rotation_sum(&rotation), 10.424_101_881_334_796);
    assert_eq!(rotation_nonzero_count(&rotation), 37);
    assert_close(rotation_value(&rotation, 1, 0, 0), 1.0);
    assert_close(
        rotation_value(&rotation, 2, -1, -1),
        0.960_530_497_001_442_6,
    );
    assert_close(rotation_value(&rotation, 2, -1, 0), -0.275_360_350_564_871);
    assert_close(rotation_value(&rotation, 2, 0, 1), -0.275_360_350_564_871);
    assert_close(
        rotation_value(&rotation, 3, -1, 1),
        0.112_177_142_327_859_86,
    );
    assert_close(rotation_value(&rotation, 5, -1, 1), 0.307_544_785_027_699_8);
    assert_close(rotation_value(&rotation, 5, 0, 0), 0.342_377_357_912_471_87);
    Ok(())
}

#[test]
fn initial_state_rotation_rejects_invalid_inputs() {
    assert_eq!(
        initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 0,
            mmaxp1: 1,
            beta_angle: 0.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 1,
            mmaxp1: 0,
            beta_angle: 0.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 1,
            mmaxp1: 1,
            beta_angle: f64::NAN,
        }),
        Err(GenfmtError::NonFiniteRotationAngle)
    );
}

#[test]
fn path_rotation_angles_match_polarized_rdpath_reference() -> Result<(), GenfmtError> {
    let positions = arr2(&[
        [1.2, -0.4, 0.7],
        [-0.3, 1.1, 1.5],
        [0.5, 0.2, -0.6],
        [0.0, 0.0, 0.0],
    ]);
    let angles = path_rotation_angles(PathRotationInput {
        positions: positions.view(),
        polarized: true,
    })?;

    assert_array_close(
        &angles.beta_angles,
        &[
            2.166_858_401_769_925_3,
            2.450_803_939_009_357,
            2.431_538_373_717_806,
            0.731_447_381_254_918_5,
            1.065_347_578_436_332_9,
        ],
    );
    assert_array_close(
        &angles.eta_values,
        &[
            3.463_343_207_986_435_3,
            3.671_719_781_241_285,
            6.729_824_761_627_887,
            11.178_806_101_438_672,
            0.800_671_291_800_303_8,
            3.522_099_030_702_158,
        ],
    );
    assert_array_close(
        &angles.leg_lengths,
        &[
            1.445_683_229_480_096,
            2.267_156_809_750_926_7,
            2.420_743_687_382_041,
            0.806_225_774_829_855,
        ],
    );
    Ok(())
}

#[test]
fn path_rotation_angles_match_unpolarized_rdpath_reference() -> Result<(), GenfmtError> {
    let positions = arr2(&[[-0.2, 0.8, -1.0], [1.4, -0.5, 0.3], [0.0, 0.0, 0.0]]);
    let angles = path_rotation_angles(PathRotationInput {
        positions: positions.view(),
        polarized: false,
    })?;

    assert_array_close(
        &angles.beta_angles,
        &[
            2.571_854_110_984_37,
            2.662_458_542_799_463,
            1.048_872_653_395_752_4,
        ],
    );
    assert_array_close(
        &angles.eta_values,
        &[
            0.0,
            std::f64::consts::TAU,
            6.283_185_307_179_585,
            std::f64::consts::TAU,
            0.0,
        ],
    );
    assert_array_close(
        &angles.leg_lengths,
        &[
            1.296_148_139_681_572_2,
            2.437_211_521_390_788_3,
            1.516_575_088_810_31,
        ],
    );
    Ok(())
}

#[test]
fn path_rotation_angles_rejects_invalid_inputs() {
    let empty = Array2::<Real>::zeros((0, 3));
    assert_eq!(
        path_rotation_angles(PathRotationInput {
            positions: empty.view(),
            polarized: false,
        }),
        Err(GenfmtError::EmptyPath)
    );

    let bad_columns = Array2::<Real>::zeros((1, 2));
    assert_eq!(
        path_rotation_angles(PathRotationInput {
            positions: bad_columns.view(),
            polarized: false,
        }),
        Err(GenfmtError::InvalidPathCoordinateColumns { columns: 2 })
    );

    let nonfinite = arr2(&[[0.0, f64::NAN, 0.0]]);
    assert!(matches!(
        path_rotation_angles(PathRotationInput {
            positions: nonfinite.view(),
            polarized: false,
        }),
        Err(GenfmtError::NonFinitePathCoordinate {
            leg_index: 0,
            component: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_legendre_normalization_matches_snlm_reference() -> Result<(), GenfmtError> {
    let table = genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
        lmaxp1: 5,
        mmaxp1: 4,
    })?;

    assert_eq!(table.shape(), &[5, 4]);
    assert_eq!(table.strides(), &[1, 5]);
    assert_table_close(
        &table,
        &[
            [1.0, 0.0, 0.0, 0.0],
            [1.732_050_807_568_877_2, 1.224_744_871_391_589, 0.0, 0.0],
            [
                2.236_067_977_499_79,
                0.912_870_929_175_276_9,
                0.456_435_464_587_638_45,
                0.0,
            ],
            [
                2.645_751_311_064_590_7,
                0.763_762_615_825_973_4,
                0.241_522_945_769_823_97,
                0.098_601_329_718_326_94,
            ],
            [
                3.0,
                0.670_820_393_249_936_9,
                0.158_113_883_008_418_97,
                0.042_257_712_736_425_826,
            ],
        ],
    );
    Ok(())
}

#[test]
fn genfmt_legendre_normalization_matches_limited_m_reference() -> Result<(), GenfmtError> {
    let table = genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
        lmaxp1: 7,
        mmaxp1: 3,
    })?;

    assert_eq!(table.shape(), &[7, 3]);
    assert_eq!(table.strides(), &[1, 7]);
    assert_table_close(
        &table,
        &[
            [1.0, 0.0, 0.0],
            [1.732_050_807_568_877_2, 1.224_744_871_391_589, 0.0],
            [
                2.236_067_977_499_79,
                0.912_870_929_175_276_9,
                0.456_435_464_587_638_45,
            ],
            [
                2.645_751_311_064_590_7,
                0.763_762_615_825_973_4,
                0.241_522_945_769_823_97,
            ],
            [3.0, 0.670_820_393_249_936_9, 0.158_113_883_008_418_97],
            [
                3.316_624_790_355_4,
                0.605_530_070_819_498_3,
                0.114_434_427_054_265_86,
            ],
            [
                3.605_551_275_463_989,
                0.556_348_640_264_186_8,
                0.087_966_443_818_624_6,
            ],
        ],
    );
    Ok(())
}

#[test]
fn genfmt_legendre_normalization_rejects_invalid_inputs() {
    assert_eq!(
        genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
            lmaxp1: 0,
            mmaxp1: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
            lmaxp1: 1,
            mmaxp1: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
            lmaxp1: 107,
            mmaxp1: 107,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1+mmaxp1",
            value: 212,
        })
    );
}

#[test]
fn curved_wave_polynomials_match_feff_sclmz_reference() -> Result<(), GenfmtError> {
    let table = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 4,
        rho: Complex::new(1.25, 0.4),
    })?;

    assert_eq!(table.shape(), &[5, 4]);
    assert_eq!(table.strides(), &[1, 5]);
    assert_eq!(complex_nonzero_count(&table), 11);
    assert_complex_close(table[(0, 0)], Complex::new(1.0, 0.0));
    assert_complex_close(
        table[(1, 0)],
        Complex::new(1.232_220_609_579_100_2, 0.725_689_404_934_687_9),
    );
    assert_complex_close(
        table[(2, 0)],
        Complex::new(0.278_565_725_973_782_6, 3.188_188_430_678_23),
    );
    assert_complex_close(
        table[(3, 1)],
        Complex::new(-28.733_692_908_170_283, 2.550_923_127_350_68),
    );
    assert_complex_close(table[(4, 2)], Complex::new(0.0, 0.0));
    assert_complex_close(
        complex_sum(&table),
        Complex::new(-58.983_990_231_020_26, -154.618_863_530_600_9),
    );
    Ok(())
}

#[test]
fn curved_wave_polynomials_match_limited_m_reference() -> Result<(), GenfmtError> {
    let table = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 5,
        mmaxp1: 3,
        rho: Complex::new(-0.8, 1.1),
    })?;

    assert_eq!(table.shape(), &[6, 3]);
    assert_eq!(table.strides(), &[1, 6]);
    assert_eq!(complex_nonzero_count(&table), 12);
    assert_complex_close(
        table[(1, 0)],
        Complex::new(1.594_594_594_594_594_5, -0.432_432_432_432_432_35),
    );
    assert_complex_close(
        table[(2, 0)],
        Complex::new(3.283_418_553_688_824, -2.840_029_218_407_596),
    );
    assert_complex_close(
        table[(3, 1)],
        Complex::new(3.013_207_509_920_446_7, -35.022_288_906_876_184),
    );
    assert_complex_close(
        table[(4, 2)],
        Complex::new(-180.487_514_146_329_86, -250.055_955_704_979_3),
    );
    assert_complex_close(
        complex_sum(&table),
        Complex::new(-306.259_756_232_255_1, -662.066_424_389_366_5),
    );
    Ok(())
}

#[test]
fn curved_wave_polynomials_retain_requested_zero_columns() -> Result<(), GenfmtError> {
    let table = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 2,
        mmaxp1: 4,
        rho: Complex::new(1.0, 0.25),
    })?;

    assert_eq!(table.shape(), &[3, 4]);
    assert!(
        table
            .column(2)
            .iter()
            .all(|&value| value == Complex::new(0.0, 0.0))
    );
    assert!(
        table
            .column(3)
            .iter()
            .all(|&value| value == Complex::new(0.0, 0.0))
    );
    Ok(())
}

#[test]
fn curved_wave_polynomials_reject_invalid_inputs() {
    assert_eq!(
        curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 0,
            mmaxp1: 1,
            rho: Complex::new(1.0, 0.0),
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 1,
            mmaxp1: 0,
            rho: Complex::new(1.0, 0.0),
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "mmaxp1",
            value: 0,
        })
    );
    assert_eq!(
        curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 1,
            mmaxp1: 1,
            rho: Complex::new(0.0, 0.0),
        }),
        Err(GenfmtError::ZeroComplex { field: "rho" })
    );
    assert!(matches!(
        curved_wave_polynomials(CurvedWavePolynomialInput {
            lmaxp1: 1,
            mmaxp1: 1,
            rho: Complex::new(f64::NAN, 0.0),
        }),
        Err(GenfmtError::NonFiniteComplex { field: "rho", .. })
    ));
}

#[test]
fn scattering_amplitude_matrix_matches_feff_fmtrxi_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = fmtrxi_reference_data()?;
    let matrix = scattering_amplitude_matrix(data.input())?;

    assert_eq!(matrix.shape(), &[6, 5]);
    assert_eq!(matrix.strides(), &[1, 6]);
    assert_complex_close(
        matrix[(0, 0)],
        Complex::new(-38.563_289_559_671_01, 28.084_721_411_987_896),
    );
    assert_complex_close(
        matrix[(0, 1)],
        Complex::new(-129.565_304_116_042_23, 92.125_635_892_089_4),
    );
    assert_complex_close(
        matrix[(1, 2)],
        Complex::new(122.713_265_094_310_16, 21.039_927_424_360_677),
    );
    assert_complex_close(
        matrix[(3, 4)],
        Complex::new(-63.332_044_984_118_596, -84.365_936_676_961_67),
    );
    assert_complex_close(
        matrix[(5, 4)],
        Complex::new(-1_309.182_568_320_504, 255.082_893_344_668_2),
    );
    assert_complex_close(
        complex_sum(&matrix),
        Complex::new(-3_078.729_163_920_782_4, 1_027.554_784_760_136),
    );
    Ok(())
}

#[test]
fn scattering_amplitude_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let data = fmtrxi_reference_data()?;
    assert!(matches!(
        scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            left_lambda_count: 9,
            ..data.input()
        }),
        Err(GenfmtError::LambdaCountOutOfRange {
            name: "left_lambda_count",
            requested: 9,
            available: 8,
        })
    ));

    let bad_phase = Array1::from_vec(vec![Complex::new(0.0, 0.0); 4]);
    assert_eq!(
        scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            phase_shifts: bad_phase.view(),
            ..data.input()
        }),
        Err(GenfmtError::InvalidSignedPhaseShape { length: 4 })
    );

    let mut nonfinite_phase = data.phase_shifts.clone();
    nonfinite_phase[4] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            phase_shifts: nonfinite_phase.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "phase_shifts",
            row: 4,
            ..
        })
    ));

    let mut zero_xnlm = data.xnlm.clone();
    zero_xnlm[(1, 1)] = 0.0;
    assert_eq!(
        scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            xnlm: zero_xnlm.view(),
            ..data.input()
        }),
        Err(GenfmtError::ZeroLegendreNormalization {
            angular_momentum: 1,
            magnetic: 1,
        })
    );

    let short_polynomials = Array2::zeros((4, 1).f());
    assert!(matches!(
        scattering_amplitude_matrix(ScatteringAmplitudeMatrixInput {
            first_leg_polynomials: short_polynomials.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "first_leg_polynomials",
            axis: "column",
            ..
        })
    ));
    Ok(())
}

#[test]
fn polarized_scattering_amplitude_matrix_matches_feff_mmtrxi_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = mmtrxi_reference_data()?;
    let matrix = polarized_scattering_amplitude_matrix(data.input())?;

    assert_eq!(matrix.shape(), &[6, 6]);
    assert_eq!(matrix.strides(), &[1, 6]);
    assert_complex_close(
        matrix[(0, 0)],
        Complex::new(-2_845.112_371_916_357, 2_888.147_341_052_974),
    );
    assert_complex_close(
        matrix[(0, 1)],
        Complex::new(-10_079.776_065_551_37, 9_413.994_100_845_948),
    );
    assert_complex_close(
        matrix[(1, 2)],
        Complex::new(8_697.313_993_828_167, -374.375_986_576_882_5),
    );
    assert_complex_close(
        matrix[(3, 4)],
        Complex::new(-4_714.438_045_315_254, -3_615.819_287_952_961_5),
    );
    assert_complex_close(
        matrix[(5, 5)],
        Complex::new(-16_490.015_276_258_873, 9_905.708_935_168_93),
    );
    assert_complex_close(
        complex_sum(&matrix),
        Complex::new(-235_884.893_264_593_76, 120_845.446_342_197_36),
    );
    Ok(())
}

#[test]
fn polarized_scattering_amplitude_matrix_rejects_invalid_inputs()
-> Result<(), Box<dyn std::error::Error>> {
    let data = mmtrxi_reference_data()?;
    assert!(matches!(
        polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
            lambda_count: 9,
            ..data.input()
        }),
        Err(GenfmtError::LambdaCountOutOfRange {
            name: "lambda_count",
            requested: 9,
            available: 8,
        })
    ));

    let mut bad_radial = data.radial_factors.clone();
    bad_radial[1] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
            radial_factors: bad_radial.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "radial_factors",
            row: 1,
            ..
        })
    ));

    let mut bad_transition = data.transition_matrix.clone();
    bad_transition[(4, 0, 4, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
            transition_matrix: bad_transition.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTensorComplex {
            table: "transition_matrix",
            i0: 4,
            i1: 0,
            i2: 4,
            i3: 0,
            ..
        })
    ));

    let short_transition_matrix = Array4::zeros((8, 8, 9, 8).f());
    assert!(matches!(
        polarized_scattering_amplitude_matrix(PolarizedScatteringAmplitudeInput {
            transition_matrix: short_transition_matrix.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "transition_matrix",
            axis: "m1",
            ..
        })
    ));
    Ok(())
}

#[test]
fn energy_independent_transition_matrix_matches_feff_mmtr_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = mmtr_reference_data();
    let polarized = energy_independent_transition_matrix(data.polarized_input())?;

    assert_eq!(polarized.shape(), &[7, 8, 7, 8]);
    assert_eq!(polarized.strides(), &[1, 7, 56, 392]);
    assert_complex_close(
        polarized[(3, 0, 3, 0)],
        Complex::new(0.021_453_694_254_769_01, 0.071_512_314_182_563_39),
    );
    assert_complex_close(
        polarized[(2, 1, 4, 2)],
        Complex::new(0.002_111_873_685_701_496, 1.236_234_538_760_950_8),
    );
    assert_complex_close(
        polarized[(5, 3, 3, 4)],
        Complex::new(0.628_672_134_559_167_4, 1.917_320_183_093_828_2),
    );
    assert_complex_close(
        polarized[(1, 5, 1, 5)],
        Complex::new(0.581_425_567_184_014_2, 3.044_675_502_642_624),
    );
    assert_complex_close(
        active_bmati_sum(&polarized),
        Complex::new(286.229_896_462_046_5, 1_632.094_116_299_501_8),
    );

    let averaged = energy_independent_transition_matrix(data.unpolarized_input())?;
    assert_complex_close(
        averaged[(3, 0, 3, 0)],
        Complex::new(0.014_330_047_336_884_089, 0.047_766_824_456_280_305),
    );
    assert_complex_close(
        averaged[(2, 1, 4, 1)],
        Complex::new(0.028_570_007_096_571_4, 0.095_233_356_988_571_34),
    );
    assert_complex_close(
        averaged[(5, 3, 3, 3)],
        Complex::new(0.040_492_545_604_276_02, 0.134_975_152_014_253_42),
    );
    assert_complex_close(
        averaged[(1, 5, 1, 5)],
        Complex::new(0.078_103_726_170_988_49, 0.260_345_753_903_294_95),
    );
    assert_complex_close(
        active_bmati_sum(&averaged),
        Complex::new(7.154_567_773_293_091, 23.848_559_244_310_298),
    );
    Ok(())
}

#[test]
fn energy_independent_transition_matrix_rejects_invalid_inputs() {
    let data = mmtr_reference_data();
    assert!(matches!(
        energy_independent_transition_matrix(EnergyIndependentMatrixInput {
            spin_index: 2,
            ..data.polarized_input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "transition_b_matrix",
            axis: "spin1",
            ..
        })
    ));

    let mut bad_bmat = data.transition_b_matrix.clone();
    bad_bmat[(3, 1, 0, 3, 1, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        energy_independent_transition_matrix(EnergyIndependentMatrixInput {
            transition_b_matrix: bad_bmat.view(),
            ..data.polarized_input()
        }),
        Err(GenfmtError::NonFiniteTensor6Complex {
            table: "transition_b_matrix",
            i0: 3,
            i1: 1,
            i2: 0,
            i3: 3,
            i4: 1,
            i5: 0,
            ..
        })
    ));

    let short_rotation = Array3::zeros((3, 7, 7).f());
    assert!(matches!(
        energy_independent_transition_matrix(EnergyIndependentMatrixInput {
            rotations: TransitionRotationInput::Unpolarized {
                combined_rotation: short_rotation.view(),
            },
            ..data.unpolarized_input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "combined_rotation",
            axis: "l",
            ..
        })
    ));
}

#[test]
fn xstar_matches_feff_linear_references() -> Result<(), GenfmtError> {
    assert_close(
        xstar(XStarInput {
            primary_polarization: [1.0, 0.0, 0.0],
            secondary_polarization: [0.0, 1.0, 0.0],
            first_leg: [2.0, 0.0, 0.0],
            last_leg: [0.0, 3.0, 0.0],
            degeneracy: 3.5,
            initial_l: 1,
            ellipticity: 0.0,
        })?,
        0.0,
    );
    assert_close(
        xstar(XStarInput {
            primary_polarization: [0.2, 0.9, 0.4],
            secondary_polarization: [0.0, 1.0, 0.0],
            first_leg: [1.0, 0.5, -0.25],
            last_leg: [0.4, -0.3, 1.2],
            degeneracy: 1.75,
            initial_l: 1,
            ellipticity: 0.0,
        })?,
        0.185_559_995_771_885_34,
    );
    Ok(())
}

#[test]
fn xstar_matches_feff_elliptic_references() -> Result<(), GenfmtError> {
    assert_close(
        xstar(XStarInput {
            primary_polarization: [0.3, 1.0, -0.2],
            secondary_polarization: [-0.4, 0.2, 1.5],
            first_leg: [1.2, -0.5, 0.8],
            last_leg: [-0.7, 1.4, 0.6],
            degeneracy: 2.25,
            initial_l: 2,
            ellipticity: 0.7,
        })?,
        -0.014_836_343_260_557_886,
    );
    assert_close(
        xstar(XStarInput {
            primary_polarization: [1.0, 2.0, 3.0],
            secondary_polarization: [2.0, -1.0, 0.5],
            first_leg: [-0.25, 0.75, 1.50],
            last_leg: [1.1, -0.9, 0.4],
            degeneracy: 5.0,
            initial_l: 4,
            ellipticity: -0.35,
        })?,
        0.254_890_323_398_489_77,
    );
    Ok(())
}

#[test]
fn xstar_rejects_invalid_inputs() {
    assert_eq!(
        xstar(XStarInput {
            primary_polarization: [1.0, 0.0, 0.0],
            secondary_polarization: [0.0, 1.0, 0.0],
            first_leg: [1.0, 0.0, 0.0],
            last_leg: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 5,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::InvalidInitialAngularMomentum { initial_l: 5 })
    );
    assert!(matches!(
        xstar(XStarInput {
            primary_polarization: [f64::NAN, 0.0, 0.0],
            secondary_polarization: [0.0, 1.0, 0.0],
            first_leg: [1.0, 0.0, 0.0],
            last_leg: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "primary_polarization",
            index: 0,
            ..
        })
    ));
    assert_eq!(
        xstar(XStarInput {
            primary_polarization: [1.0, 0.0, 0.0],
            secondary_polarization: [0.0, 1.0, 0.0],
            first_leg: [0.0, 0.0, 0.0],
            last_leg: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::ZeroVector { field: "first_leg" })
    );
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

fn assert_array_close(actual: &Array1<Real>, expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected}"
        );
    }
}

fn assert_table_close<const ROWS: usize, const COLUMNS: usize>(
    actual: &Array2<Real>,
    expected: &[[Real; COLUMNS]; ROWS],
) {
    assert_eq!(actual.shape(), &[ROWS, COLUMNS]);
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            assert_close(actual[(row, column)], expected[row][column]);
        }
    }
}

struct FmtrxiReferenceData {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    phase_shifts: Array1<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    rotation: Array3<Real>,
    xnlm: Array2<Real>,
}

impl FmtrxiReferenceData {
    fn input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
        ScatteringAmplitudeMatrixInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            left_lambda_count: 6,
            right_lambda_count: 5,
            phase_shifts: self.phase_shifts.view(),
            angular_limit: 3,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            rotation: self.rotation.view(),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn fmtrxi_reference_data() -> Result<FmtrxiReferenceData, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let phase_shifts = Array1::from_iter((-4..=4).map(|l| {
        let l = l as Real;
        Complex::new(0.015 * l + 0.02, -0.01 * l + 0.03)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut rotation = Array3::zeros((5, 9, 9).f());
    for l in 0..=4 {
        let il = (l + 1) as Real;
        for m1 in -4..=4 {
            for m2 in -4..=4 {
                if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                    let row = (m1 + 4) as usize;
                    let column = (m2 + 4) as usize;
                    rotation[(l, row, column)] =
                        (0.11 * il + 0.07 * (m1 as Real) - 0.05 * (m2 as Real)).cos();
                }
            }
        }
    }
    let xnlm = legendre_normalization_table(4)?;

    Ok(FmtrxiReferenceData {
        m_indices,
        n_indices,
        phase_shifts,
        first_polynomials,
        second_polynomials,
        rotation,
        xnlm,
    })
}

struct MmtrxiReferenceData {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array1<Complex>,
    transition_matrix: Array4<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    xnlm: Array2<Real>,
}

impl MmtrxiReferenceData {
    fn input(&self) -> PolarizedScatteringAmplitudeInput<'_> {
        PolarizedScatteringAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 6,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            transition_matrix: self.transition_matrix.view(),
            transition_magnetic_offset: 4,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn mmtrxi_reference_data() -> Result<MmtrxiReferenceData, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let radial_factors = Array1::from_iter((1..=8).map(|k| {
        let k = k as Real;
        Complex::new(0.9 + 0.07 * k, -0.02 * k)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut transition_matrix = Array4::zeros((9, 8, 9, 8).f());
    for k2 in 1..=8 {
        for m2 in -4_i32..=4 {
            for k1 in 1..=8 {
                for m1 in -4_i32..=4 {
                    let first_m = (m1 + 4) as usize;
                    let second_m = (m2 + 4) as usize;
                    transition_matrix[(first_m, k1 - 1, second_m, k2 - 1)] = Complex::new(
                        0.01 * (m1 as Real) + 0.02 * (m2 as Real) + 0.03 * (k1 as Real)
                            - 0.015 * (k2 as Real),
                        0.02 * ((m1 - m2) as Real) + 0.01 * (k1 as Real) + 0.04 * (k2 as Real),
                    );
                }
            }
        }
    }
    let xnlm = legendre_normalization_table(4)?;

    Ok(MmtrxiReferenceData {
        m_indices,
        n_indices,
        transition_angular_momenta,
        radial_factors,
        transition_matrix,
        first_polynomials,
        second_polynomials,
        xnlm,
    })
}

struct MmtrReferenceData {
    transition_angular_momenta: Array1<i32>,
    transition_b_matrix: Array6<Complex>,
    combined_rotation: Array3<Real>,
    first_rotation: Array3<Real>,
    last_rotation: Array3<Real>,
}

impl MmtrReferenceData {
    fn polarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 1,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Polarized {
                first_rotation: self.first_rotation.view(),
                last_rotation: self.last_rotation.view(),
                first_eta: 0.23,
                last_eta: 0.41,
            },
        }
    }

    fn unpolarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 0,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Unpolarized {
                combined_rotation: self.combined_rotation.view(),
            },
        }
    }
}

fn mmtr_reference_data() -> MmtrReferenceData {
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let mut transition_b_matrix = Array6::zeros((7, 2, 8, 7, 2, 8).f());
    for k2 in 1..=8 {
        for s2 in 0..=1 {
            for m2 in -3_i32..=3 {
                for k1 in 1..=8 {
                    for s1 in 0..=1 {
                        for m1 in -3_i32..=3 {
                            let first_m = (m1 + 3) as usize;
                            let second_m = (m2 + 3) as usize;
                            transition_b_matrix[(first_m, s1, k1 - 1, second_m, s2, k2 - 1)] =
                                Complex::new(
                                    0.01 * (m1 as Real) + 0.02 * (m2 as Real) + 0.03 * (k1 as Real)
                                        - 0.015 * (k2 as Real)
                                        + 0.04 * (s1 as Real)
                                        - 0.025 * (s2 as Real),
                                    0.02 * ((m1 - m2) as Real)
                                        + 0.01 * (k1 as Real)
                                        + 0.04 * (k2 as Real)
                                        + 0.03 * (s1 as Real)
                                        + 0.02 * (s2 as Real),
                                );
                        }
                    }
                }
            }
        }
    }

    let combined_rotation = mmtr_rotation_table(1);
    let first_rotation = mmtr_rotation_table(2);
    let last_rotation = mmtr_rotation_table(3);
    MmtrReferenceData {
        transition_angular_momenta,
        transition_b_matrix,
        combined_rotation,
        first_rotation,
        last_rotation,
    }
}

fn mmtr_rotation_table(leg: usize) -> Array3<Real> {
    let mut rotation = Array3::zeros((4, 7, 7).f());
    for l in 0..=3 {
        let il = (l + 1) as Real;
        for m1 in -3_i32..=3 {
            for m2 in -3_i32..=3 {
                if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                    let row = (m1 + 3) as usize;
                    let column = (m2 + 3) as usize;
                    rotation[(l, row, column)] = (0.13 * il + 0.07 * (m1 as Real)
                        - 0.05 * (m2 as Real)
                        + 0.17 * (leg as Real))
                        .cos();
                }
            }
        }
    }
    rotation
}

fn i32_abs_usize(value: i32) -> usize {
    value.unsigned_abs() as usize
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

fn complex_sum(table: &ndarray::Array2<Complex>) -> Complex {
    table
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
}

fn active_bmati_sum(table: &Array4<Complex>) -> Complex {
    let mut sum = Complex::new(0.0, 0.0);
    for mu1 in 1..=5 {
        for k1 in 0..8 {
            for mu2 in 1..=5 {
                for k2 in 0..8 {
                    sum += table[(mu1, k1, mu2, k2)];
                }
            }
        }
    }
    sum
}

fn complex_nonzero_count(table: &ndarray::Array2<Complex>) -> usize {
    table
        .iter()
        .filter(|&&value| value.re.abs() > 1.0e-14 || value.im.abs() > 1.0e-14)
        .count()
}

fn rotation_value(rotation: &InitialStateRotation, il: usize, m1: isize, m2: isize) -> f64 {
    let row = (m1 + rotation.magnetic_offset as isize) as usize;
    let column = (m2 + rotation.magnetic_offset as isize) as usize;
    rotation.matrix[(il - 1, row, column)]
}

fn rotation_sum(rotation: &InitialStateRotation) -> f64 {
    rotation.matrix.iter().sum()
}

fn rotation_nonzero_count(rotation: &InitialStateRotation) -> usize {
    rotation
        .matrix
        .iter()
        .filter(|&&value| value.abs() > 1.0e-14)
        .count()
}
