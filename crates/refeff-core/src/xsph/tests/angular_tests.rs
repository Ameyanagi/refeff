use super::{support::*, *};

#[test]
fn xsph_longitudinal_multipole_factor_matches_feff_reference() -> Result<(), XsphError> {
    let cases = [
        (-1, -1, 0, -std::f64::consts::SQRT_2),
        (-1, 1, 1, 2.449_489_742_783_178),
        (1, -1, 1, 2.449_489_742_783_178),
        (-2, 1, 1, 0.0),
        (2, -1, 2, -4.472_135_954_999_58),
        (-3, 2, 3, 0.0),
        (3, -2, 2, 2.927_700_218_845_598),
        (-2, -2, 5, 0.0),
    ];

    for (kappa, kappa_prime, multipole_l, expected) in cases {
        let value = xsph_longitudinal_multipole_factor(kappa, kappa_prime, multipole_l)?;
        assert_close(value.re, expected);
        assert_close(value.im, 0.0);
    }
    Ok(())
}

#[test]
fn xsph_relativistic_multipole_factors_match_feff_reference() -> Result<(), XsphError> {
    let cases = [
        (-1, -1, 0, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            1,
            -1,
            0,
            1,
            Complex::new(0.0, -8.164_965_809_277_261e-1),
            Complex::new(0.0, -2.449_489_742_783_178),
        ),
        (
            -2,
            -1,
            0,
            1,
            Complex::new(0.0, -2.309_401_076_758_503_4),
            Complex::new(0.0, 0.0),
        ),
        (2, -1, 2, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            -2,
            1,
            1,
            2,
            Complex::new(-3.872_983_346_207_417_5, 0.0),
            Complex::new(-7.745_966_692_414_837e-1, 0.0),
        ),
        (
            3,
            -2,
            1,
            1,
            Complex::new(2.323_790_007_724_448_4, 0.0),
            Complex::new(2.323_790_007_724_45, 0.0),
        ),
        (
            -3,
            2,
            1,
            2,
            Complex::new(-3.549_647_869_859_77, 0.0),
            Complex::new(-1.521_277_658_511_329_2, 0.0),
        ),
        (2, -3, 3, 1, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        (
            1,
            1,
            1,
            1,
            Complex::new(-2.449_489_742_783_178, 0.0),
            Complex::new(-2.449_489_742_783_178, 0.0),
        ),
        (
            -2,
            -2,
            1,
            1,
            Complex::new(3.098_386_676_965_933_6, 0.0),
            Complex::new(3.098_386_676_965_934, 0.0),
        ),
    ];

    for (kappa, kappa_prime, bessel_l, multipole_l, expected_pq, expected_qp) in cases {
        let factors =
            xsph_relativistic_multipole_factors(kappa, kappa_prime, bessel_l, multipole_l)?;
        assert_close(factors.p_q_prime.re, expected_pq.re);
        assert_close(factors.p_q_prime.im, expected_pq.im);
        assert_close(factors.q_p_prime.re, expected_qp.re);
        assert_close(factors.q_p_prime.im, expected_qp.im);
    }
    Ok(())
}

#[test]
fn xsph_relativistic_multipole_factors_return_zero_for_unmatched_orders() -> Result<(), XsphError> {
    let factors = xsph_relativistic_multipole_factors(-1, 1, 4, 1)?;

    assert_close(factors.p_q_prime.re, 0.0);
    assert_close(factors.p_q_prime.im, 0.0);
    assert_close(factors.q_p_prime.re, 0.0);
    assert_close(factors.q_p_prime.im, 0.0);
    Ok(())
}

#[test]
fn xsph_angular_density_coefficients_match_feff_acoef_reference() -> Result<(), XsphError> {
    let cases = [
        (
            0,
            [
                3.199_999_994_039_535_5e1,
                3.199_999_994_039_535_5e1,
                3.199_999_991_059_303_3e1,
            ],
            [
                (-3, 1, 1, 1, 3, 1.714_285_731_315_612_8),
                (0, 2, 2, 1, 0, 1.999_999_523_162_841_8),
                (-1, 1, 1, 3, 1, 1.333_333_134_651_184),
                (-2, 2, 2, 2, 3, 5.714_284_777_641_296e-1),
                (3, 1, 2, 1, 3, 0.0),
            ],
        ),
        (
            1,
            [
                7.999_999_996_274_71,
                -6.109_476_089_477_539e-7,
                -1.369_044_184_684_753_4e-7,
            ],
            [
                (-2, 1, 2, 3, 2, 2.285_714_261_233_806_6e-2),
                (-1, 2, 1, 2, 2, -2.400_000_095_367_431_6e-1),
                (0, 1, 2, 3, 1, -4.444_444_924_592_972e-2),
                (2, 2, 2, 3, 3, -4.081_631_451_845_169e-2),
                (-1, 1, 1, 3, 1, 1.777_777_522_802_353e-1),
            ],
        ),
        (
            -1,
            [
                -7.999_999_996_274_71,
                6.109_476_089_477_539e-7,
                1.406_297_087_669_372_6e-7,
            ],
            [
                (-1, 2, 1, 2, 2, 1.599_999_964_237_213e-1),
                (0, 1, 2, 3, 1, 4.444_444_924_592_972e-2),
                (1, 2, 1, 3, 1, 4.444_444_179_534_912e-2),
                (3, 1, 2, 1, 3, -1.224_489_733_576_774_6e-1),
                (2, 1, 1, 1, 2, -2.399_999_946_355_819_7e-1),
            ],
        ),
        (
            2,
            [
                7.999_999_996_274_71,
                1.599_999_997_019_767_8e1,
                9.999_999_787_658_453,
            ],
            [
                (-3, 1, 1, 1, 3, 3.061_224_520_206_451_4e-1),
                (-2, 1, 2, 3, 2, -3.725_290_298_461_914e-9),
                (1, 1, 1, 2, 2, 1.999_999_880_790_710_4e-1),
                (2, 2, 2, 3, 3, 8.571_425_676_345_825e-1),
                (-2, 2, 2, 2, 3, 2.857_142_388_820_648e-1),
            ],
        ),
        (
            -2,
            [
                -7.999_999_996_274_71,
                1.599_999_997_019_767_8e1,
                9.999_999_674_037_099,
            ],
            [
                (0, 2, 2, 1, 0, -4.999_998_807_907_104_5e-1),
                (1, 1, 1, 2, 2, 6.000_000_834_465_027e-1),
                (2, 2, 2, 3, 3, 2.857_142_388_820_648e-1),
                (3, 1, 2, 1, 3, -1.224_489_733_576_774_6e-1),
                (-2, 2, 2, 2, 3, 8.571_426_868_438_721e-1),
            ],
        ),
    ];

    for (spin_selector, expected_sums, expected_entries) in cases {
        let coefficients = xsph_angular_density_coefficients(spin_selector, 3)?;
        assert_eq!(coefficients.shape(), &[7, 2, 2, 3, 4]);
        assert_eq!(coefficients.strides(), &[1, 7, 14, 28, 84]);
        for (operator, &expected_sum) in expected_sums.iter().enumerate() {
            assert_close_tol(acoef_sum(&coefficients, operator, 3), expected_sum, 1.0e-6);
        }
        for (magnetic_l, branch_1, branch_2, operator, l, expected) in expected_entries {
            assert_close_tol(
                acoef_entry(
                    &coefficients,
                    3,
                    magnetic_l,
                    branch_1,
                    branch_2,
                    operator,
                    l,
                ),
                expected,
                1.0e-7,
            );
        }
    }
    Ok(())
}

#[test]
fn xsph_angular_density_coefficients_reject_invalid_inputs() {
    assert!(matches!(
        xsph_angular_density_coefficients(1, XSPH_MAX_LX + 1),
        Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum,
            ljmax
        }) if angular_momentum == XSPH_MAX_LX + 1 && ljmax == XSPH_MAX_LX
    ));
    assert!(matches!(
        xsph_angular_density_coefficients(i32::MIN, 1),
        Err(XsphError::IntegerOutOfRange {
            name: "spin_selector",
            value: i32::MIN
        })
    ));
}

#[test]
fn xsph_nrixs_transition_weights_match_feff_reference() -> Result<(), XsphError> {
    let lgind = arr1(&[0, 1, 2, 1, 3, 2, 4]);
    let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3]);
    let weights = xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 7)?;
    assert_eq!(weights.shape(), &[2, 7]);
    assert_eq!(weights.strides(), &[1, 2]);
    let expected = arr2(&[
        [
            0.0,
            -3.333_333_333_333_333_7e-1,
            3.162_277_660_168_380_5e-1,
            1.825_741_858_350_554_4e-1,
            -2.390_457_218_668_785e-1,
            -1.690_308_509_457_032e-1,
            1.992_047_682_223_989_4e-1,
        ],
        [
            -7.071_067_811_865_477e-1,
            2.357_022_603_955_158_7e-1,
            -2.581_988_897_471_612_6e-1,
            2.581_988_897_471_612_6e-1,
            2.070_196_678_027_061_4e-1,
            -2.070_196_678_027_061_4e-1,
            -1.781_741_612_749_495_3e-1,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }

    let lgind = arr1(&[1, 2, 1, 3, 2, 4, 3, 4]);
    let ljind = arr1(&[0, 1, 1, 2, 2, 3, 3, 4]);
    let weights = xsph_nrixs_transition_weights(2, -1, 4, 11, 4, lgind.view(), ljind.view(), 8)?;
    let expected = arr2(&[
        [
            4.082_482_904_638_632_4e-1,
            0.0,
            -1.054_092_553_389_460_6e-1,
            7.824_607_964_359_512e-2,
            0.0,
            0.0,
            1.106_566_670_344_975_2e-1,
            -9.390_602_830_316_835e-2,
        ],
        [
            2.886_751_345_948_13e-1,
            0.0,
            -7.453_559_924_999_303e-2,
            -9.035_079_029_052_508e-2,
            0.0,
            0.0,
            -1.277_753_129_999_878_7e-1,
            1.049_901_313_914_518_7e-1,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }

    let lgind = arr1(&[0, 1, 2, 2, 3]);
    let ljind = arr1(&[0, 1, 2, 2, 3]);
    let weights = xsph_nrixs_transition_weights(-2, 3, 4, 9, 3, lgind.view(), ljind.view(), 5)?;
    let expected = arr2(&[
        [0.0, 0.0, 2.0e-1, -1.309_307_341_415_953e-1, 0.0],
        [
            0.0,
            0.0,
            -1.000_000_000_000_000_2e-1,
            -2.618_614_682_831_905e-1,
            0.0,
        ],
    ]);
    for ((spin, channel), &expected_value) in expected.indexed_iter() {
        assert_close(weights[(spin, channel)], expected_value);
    }
    Ok(())
}
