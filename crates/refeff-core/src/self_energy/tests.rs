use super::*;

#[test]
fn finds_no_singularities_for_feff_empty_window_case() -> Result<(), SelfEnergyError> {
    let values = find_self_energy_singularities(
        [Complex::new(0.0, 0.0), Complex::new(4.0, 0.0)],
        [0.5, 0.01, 1.0, 0.0],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        SingularityFunction::Second,
    )?;

    assert!(values.is_empty());
    Ok(())
}

#[test]
fn omega_q_and_gamma_q_match_feff_reference() -> Result<(), SelfEnergyError> {
    assert_real_close(omega_q(0.7, 0.2)?, 0.709_685_489_669_779_9);
    assert_real_close(omega_q(0.35, 1.4)?, 2.066_779_772_595_223_3);
    assert_real_close(omega_q(1.1, 2.0)?, 3.446_254_004_954_751_4);
    assert_real_close(gamma_q(0.08, 0.2)?, 0.320_000_005_960_464_46);
    assert_real_close(gamma_q(0.12, 1.4)?, 2.172_187_880_207_457);
    Ok(())
}

#[test]
fn omega_q_and_gamma_q_reject_invalid_inputs() {
    assert!(matches!(
        omega_q(0.0, 0.2),
        Err(SelfEnergyError::NonPositiveReal { name: "Wp", .. })
    ));
    assert!(matches!(
        omega_q(0.7, 0.0),
        Err(SelfEnergyError::NonPositiveReal { name: "q", .. })
    ));
    assert!(matches!(
        gamma_q(-0.1, 0.2),
        Err(SelfEnergyError::NegativeReal { name: "gam0", .. })
    ));
    assert!(matches!(
        gamma_q(0.1, Real::NAN),
        Err(SelfEnergyError::NonFiniteReal { name: "q", .. })
    ));
}

#[test]
fn log_i_matches_feff_reference() -> Result<(), SelfEnergyError> {
    assert_complex_close(
        log_i(Complex::new(-1.0, 0.5), -1)?,
        Complex::new(-0.0, -std::f64::consts::PI),
    );
    assert_complex_close(
        log_i(Complex::new(-1.0, 0.5), 1)?,
        Complex::new(0.0, std::f64::consts::PI),
    );
    assert_complex_close(log_i(Complex::new(0.0, -2.0), 1)?, Complex::new(0.0, 0.0));
    assert_complex_close(log_i(Complex::new(2.0, -1.0), -1)?, Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn log_i_rejects_invalid_inputs() {
    assert!(matches!(
        log_i(Complex::new(Real::NAN, 0.0), 1),
        Err(SelfEnergyError::NonFiniteComplex {
            name: "Logi argument",
            ..
        })
    ));
}

#[test]
fn hartree_fock_exchange_matches_feff_reference() -> Result<(), SelfEnergyError> {
    assert_complex_close(
        hartree_fock_exchange(Complex::new(1.2, 0.0), 0.72, 1.2)?,
        Complex::new(-0.381_971_863_420_548_8, 0.0),
    );
    assert_complex_close(
        hartree_fock_exchange(Complex::new(1.200_006, 0.0), 0.72, 1.2)?,
        Complex::new(-0.381_971_863_420_548_8, 0.0),
    );
    assert_complex_close(
        hartree_fock_exchange(Complex::new(1.6, 0.2), 0.8, 1.1)?,
        Complex::new(-0.153_371_750_648_922_64, 0.044_419_065_094_958_084),
    );
    assert_complex_close(
        hartree_fock_exchange(Complex::new(0.7, 0.05), 0.5, 1.3)?,
        Complex::new(-0.374_033_589_717_366_3, 0.511_283_530_382_516_3),
    );
    Ok(())
}

#[test]
fn hartree_fock_exchange_rejects_invalid_inputs() {
    assert!(matches!(
        hartree_fock_exchange(Complex::new(Real::NAN, 0.0), 0.72, 1.2),
        Err(SelfEnergyError::NonFiniteComplex { name: "ckIn", .. })
    ));
    assert!(matches!(
        hartree_fock_exchange(Complex::new(1.2, 0.0), 0.0, 1.2),
        Err(SelfEnergyError::NonPositiveReal { name: "EFermi", .. })
    ));
    assert!(matches!(
        hartree_fock_exchange(Complex::new(0.0, 0.0), 0.72, 1.2),
        Err(SelfEnergyError::ZeroDenominator {
            name: "HFExc normalized momentum"
        })
    ));
    assert!(matches!(
        hartree_fock_exchange(Complex::new(-1.2, 0.0), 0.72, 1.2),
        Err(SelfEnergyError::ZeroDenominator {
            name: "HFExc logarithm argument"
        })
    ));
}

#[test]
fn self_energy_integrands_match_feff_reference() -> Result<(), SelfEnergyError> {
    let case_a = SelfEnergyIntegrandInput {
        q: Complex::new(0.8, 0.0),
        normalized_momentum: Complex::new(0.7, 0.0),
        normalized_energy: Complex::new(0.9, 0.02),
        plasmon_over_fermi: 0.35,
        width_over_fermi: 0.02,
        gap_energy: 0.0,
        on_shell: true,
    };
    assert_complex_close(
        self_energy_pole_dispersion(case_a)?,
        Complex::new(1.176_889_421_585_4, -0.005_947_882_504_178_027),
    );
    assert_complex_close(
        self_energy_r1_integrand(case_a)?,
        Complex::new(0.032_308_875_112_581_06, 0.017_475_355_906_951_158),
    );
    assert_complex_close(
        self_energy_r2_integrand(case_a)?,
        Complex::new(2.306_044_211_646_646_4, 0.096_551_854_156_049_07),
    );
    assert_complex_close(
        self_energy_r3_integrand(case_a)?,
        Complex::new(-2.646_785_111_313_805_2, 3.230_128_477_780_93),
    );
    assert_complex_close(
        self_energy_dr1_integrand(case_a)?,
        Complex::new(0.883_406_652_847_007_7, 0.007_670_970_224_101_967),
    );
    assert_complex_close(
        self_energy_dr2_integrand(case_a)?,
        Complex::new(3.250_136_513_836_996_4, 0.344_240_751_348_657_2),
    );
    assert_complex_close(
        self_energy_dr3_integrand(case_a)?,
        Complex::new(-6.606_546_420_326_403, -0.524_680_437_136_778_1),
    );

    let case_b = SelfEnergyIntegrandInput {
        q: Complex::new(1.1, 0.2),
        normalized_momentum: Complex::new(1.2, 0.15),
        normalized_energy: Complex::new(1.6, -0.03),
        plasmon_over_fermi: 0.65,
        width_over_fermi: 0.05,
        gap_energy: 0.1,
        on_shell: false,
    };
    assert_complex_close(
        self_energy_pole_dispersion(case_b)?,
        Complex::new(1.826_377_952_391_546_9, 0.424_683_911_847_316_2),
    );
    assert_complex_close(
        self_energy_r1_integrand(case_b)?,
        Complex::new(0.503_261_695_846_839, -0.180_058_409_154_909_64),
    );
    assert_complex_close(
        self_energy_r2_integrand(case_b)?,
        Complex::new(0.932_472_508_833_754_9, -0.778_164_945_934_748_8),
    );
    assert_complex_close(
        self_energy_r3_integrand(case_b)?,
        Complex::new(0.476_553_873_697_922_57, 1.682_084_842_690_467),
    );
    assert_complex_close(
        self_energy_dr1_integrand(case_b)?,
        Complex::new(0.230_393_654_783_199_54, -0.201_333_716_007_401_4),
    );
    assert_complex_close(
        self_energy_dr2_integrand(case_b)?,
        Complex::new(0.012_886_461_912_132_44, -0.888_921_147_324_259_4),
    );
    assert_complex_close(
        self_energy_dr3_integrand(case_b)?,
        Complex::new(-0.237_825_344_903_220_74, 0.260_736_139_569_914_5),
    );
    Ok(())
}

#[test]
fn self_energy_bpr_integrands_match_feff_reference() -> Result<(), SelfEnergyError> {
    let small_on_shell = SelfEnergyIntegrandInput {
        q: Complex::new(0.1, 0.0),
        normalized_momentum: Complex::new(0.7, 0.0),
        normalized_energy: Complex::new(0.9, 0.0),
        plasmon_over_fermi: 0.7,
        width_over_fermi: 0.01,
        gap_energy: 0.0,
        on_shell: true,
    };
    assert_complex_close_tol(
        self_energy_bpr1_integrand(small_on_shell)?,
        Complex::new(-3.730_415_838_277_395_8, 0.0),
        1.0e-12,
    );
    assert_complex_close_tol(
        self_energy_bpr2_integrand(small_on_shell)?,
        Complex::new(5.649_059_314_516_695, 0.0),
        1.0e-12,
    );
    assert_complex_close_tol(
        self_energy_bpr3_integrand(small_on_shell)?,
        Complex::new(-5.818_492_494_926_515, 0.0),
        1.0e-12,
    );

    let large_on_shell = SelfEnergyIntegrandInput {
        q: Complex::new(1.2, 0.0),
        normalized_momentum: Complex::new(0.9, 0.0),
        normalized_energy: Complex::new(1.2, 0.0),
        plasmon_over_fermi: 0.4,
        width_over_fermi: 0.05,
        gap_energy: 0.0,
        on_shell: true,
    };
    assert_complex_close_tol(
        self_energy_bpr1_integrand(large_on_shell)?,
        Complex::new(0.428_711_438_643_630_15, 9.394_291_973_239_055e-4),
        1.0e-12,
    );
    assert_complex_close_tol(
        self_energy_bpr2_integrand(large_on_shell)?,
        Complex::new(0.730_806_408_402_361_9, 3.946_709_191_415_367e-2),
        1.0e-12,
    );
    assert_complex_close_tol(
        self_energy_bpr3_integrand(large_on_shell)?,
        Complex::new(-0.215_062_330_652_128_18, -1.320_856_511_909_857_6),
        1.0e-12,
    );

    let off_shell_complex = SelfEnergyIntegrandInput {
        q: Complex::new(0.45, 0.0),
        normalized_momentum: Complex::new(1.1, 0.15),
        normalized_energy: Complex::new(1.6, -0.03),
        plasmon_over_fermi: 0.65,
        width_over_fermi: 0.03,
        gap_energy: 0.0,
        on_shell: false,
    };
    assert_complex_close_tol(
        self_energy_bpr1_integrand(off_shell_complex)?,
        Complex::new(2.900_082_890_815_600_2, 2.231_685_771_427_218),
        1.0e-12,
    );
    assert_complex_close_tol(
        self_energy_bpr2_integrand(off_shell_complex)?,
        Complex::new(9.320_463_915_139_964, 15.354_411_218_075_196),
        1.0e-6,
    );
    assert_complex_close_tol(
        self_energy_bpr3_integrand(off_shell_complex)?,
        Complex::new(-4.614_749_612_587_786_5, -6.296_322_966_191_412),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn self_energy_integrands_reject_invalid_inputs() {
    let input = SelfEnergyIntegrandInput {
        q: Complex::new(0.0, 0.0),
        normalized_momentum: Complex::new(0.7, 0.0),
        normalized_energy: Complex::new(0.9, 0.02),
        plasmon_over_fermi: 0.35,
        width_over_fermi: 0.02,
        gap_energy: 0.0,
        on_shell: true,
    };
    assert!(matches!(
        self_energy_r1_integrand(input),
        Err(SelfEnergyError::ZeroDenominator {
            name: "self-energy q*fq"
        })
    ));

    assert!(matches!(
        self_energy_pole_dispersion(SelfEnergyIntegrandInput {
            plasmon_over_fermi: 0.0,
            q: Complex::new(0.8, 0.0),
            ..input
        }),
        Err(SelfEnergyError::NonPositiveReal {
            name: "DPPar(1)",
            ..
        })
    ));

    assert!(matches!(
        self_energy_r2_integrand(SelfEnergyIntegrandInput {
            normalized_energy: Complex::new(Real::NAN, 0.0),
            q: Complex::new(0.8, 0.0),
            ..input
        }),
        Err(SelfEnergyError::NonFiniteComplex {
            name: "CPar(2)",
            ..
        })
    ));
}

#[test]
fn self_energy_single_pole_matches_feff_csigz_reference() -> Result<(), SelfEnergyError> {
    let case_a = SelfEnergySinglePoleInput {
        momentum: Complex::new(0.9, 0.0),
        energy: Complex::new(0.55, 0.0),
        pole_energy: 0.25,
        width: 0.0,
        amplitude: 0.7,
        fermi_momentum: 1.2,
        fermi_energy: 0.72,
        on_shell: true,
        use_broadened_pole: false,
    };
    assert_complex_close_tol(
        self_energy_single_pole(case_a)?,
        Complex::new(5.353_189_540_301_497e-2, -1.233_737_687_764_275e-11),
        1.0e-13,
    );
    assert_complex_close_tol(
        self_energy_single_pole_derivative(case_a)?,
        Complex::new(-1.712_321_807_255_59e-1, 6.569_697_968_466_269e-11),
        1.0e-13,
    );

    let case_b = SelfEnergySinglePoleInput {
        momentum: Complex::new(1.6, 0.0),
        energy: Complex::new(1.15, 0.0),
        pole_energy: 0.4,
        width: 0.0,
        amplitude: 0.45,
        fermi_momentum: 1.1,
        fermi_energy: 0.605,
        on_shell: true,
        use_broadened_pole: false,
    };
    assert_complex_close_tol(
        self_energy_single_pole(case_b)?,
        Complex::new(-9.199_882_535_049_86e-2, -5.732_140_380_313_636e-3),
        1.0e-13,
    );
    assert_complex_close_tol(
        self_energy_single_pole_derivative(case_b)?,
        Complex::new(-2.187_035_922_040_086_5e-1, -2.850_028_456_822_25e-1),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn self_energy_bpr_single_and_many_pole_match_feff_reference() -> Result<(), SelfEnergyError> {
    let case_a = SelfEnergySinglePoleInput {
        momentum: Complex::new(0.9, 0.0),
        energy: Complex::new(0.55, 0.0),
        pole_energy: 0.25,
        width: 0.02,
        amplitude: 0.7,
        fermi_momentum: 1.2,
        fermi_energy: 0.72,
        on_shell: true,
        use_broadened_pole: true,
    };
    assert_complex_close_tol(
        self_energy_single_pole(case_a)?,
        Complex::new(8.263_180_923_660_31e-2, 1.482_655_742_716_911_5e-2),
        1.0e-10,
    );

    let case_b = SelfEnergySinglePoleInput {
        momentum: Complex::new(1.6, 0.0),
        energy: Complex::new(1.15, 0.0),
        pole_energy: 0.4,
        width: 0.03,
        amplitude: 0.45,
        fermi_momentum: 1.1,
        fermi_energy: 0.605,
        on_shell: true,
        use_broadened_pole: true,
    };
    assert_complex_close_tol(
        self_energy_single_pole(case_b)?,
        Complex::new(-9.702_874_582_903_437e-2, -3.103_634_769_514_895e-2),
        1.0e-10,
    );

    let frequencies = ndarray::arr1(&[0.35, 0.8, 1.6]);
    let widths = ndarray::arr1(&[0.02, 0.03, 0.04]);
    let amplitudes = ndarray::arr1(&[0.25, 0.4, 0.15]);
    let many = many_pole_self_energy(ManyPoleSelfEnergyInput {
        energy: Complex::new(0.4, 0.0),
        fermi_level: 0.05,
        radius: 2.1,
        pole_frequencies: frequencies.view(),
        pole_widths: widths.view(),
        amplitudes: amplitudes.view(),
        gap_energy: 0.03,
        active_pole_count: 3,
        use_broadened_pole: true,
    })?;
    assert_complex_close_tol(
        many.self_energy,
        Complex::new(-3.503_264_131_341_754e-1, -8.818_033_789_294_872e-3),
        1.0e-9,
    );
    assert_complex_close_tol(
        many.renormalization,
        Complex::new(7.575_684_793_566_081e-1, -6.470_504_032_909_914e-2),
        1.0e-9,
    );
    Ok(())
}

#[test]
fn many_pole_self_energy_matches_feff_csigz_reference() -> Result<(), SelfEnergyError> {
    let frequencies_a = ndarray::arr1(&[0.35, 0.8, 1.6]);
    let widths_a = ndarray::arr1(&[0.02, 0.03, 0.04]);
    let amplitudes_a = ndarray::arr1(&[0.25, 0.4, 0.15]);
    let case_a = many_pole_self_energy(ManyPoleSelfEnergyInput {
        energy: Complex::new(0.4, 0.0),
        fermi_level: 0.05,
        radius: 2.1,
        pole_frequencies: frequencies_a.view(),
        pole_widths: widths_a.view(),
        amplitudes: amplitudes_a.view(),
        gap_energy: 0.03,
        active_pole_count: 3,
        use_broadened_pole: false,
    })?;
    assert_complex_close_tol(
        case_a.self_energy,
        Complex::new(-3.374_360_348_369_058e-1, -1.541_476_936_320_443e-11),
        1.0e-13,
    );
    assert_complex_close_tol(
        case_a.renormalization,
        Complex::new(7.312_717_445_229_927e-1, -7.815_881_662_038_596e-11),
        1.0e-13,
    );

    let frequencies_b = ndarray::arr1(&[0.22, 0.55]);
    let widths_b = ndarray::arr1(&[0.01, 0.02]);
    let amplitudes_b = ndarray::arr1(&[0.6, 0.3]);
    let case_b = many_pole_self_energy(ManyPoleSelfEnergyInput {
        energy: Complex::new(0.18, 0.0),
        fermi_level: -0.02,
        radius: 1.65,
        pole_frequencies: frequencies_b.view(),
        pole_widths: widths_b.view(),
        amplitudes: amplitudes_b.view(),
        gap_energy: 0.0,
        active_pole_count: 2,
        use_broadened_pole: false,
    })?;
    assert_complex_close_tol(
        case_b.self_energy,
        Complex::new(-3.476_417_064_397_695e-1, -2.839_141_932_096_666e-11),
        1.0e-13,
    );
    assert_complex_close_tol(
        case_b.renormalization,
        Complex::new(7.044_615_536_261_144e-1, -1.649_926_001_414_770_6e-10),
        1.0e-13,
    );
    Ok(())
}

#[test]
fn many_pole_self_energy_rejects_invalid_inputs() {
    let frequencies = ndarray::arr1(&[0.35]);
    let widths = ndarray::arr1(&[0.02]);
    let amplitudes = ndarray::arr1(&[0.25]);
    assert!(matches!(
        many_pole_self_energy(ManyPoleSelfEnergyInput {
            energy: Complex::new(0.4, 0.0),
            fermi_level: 0.05,
            radius: 2.1,
            pole_frequencies: frequencies.view(),
            pole_widths: widths.view(),
            amplitudes: amplitudes.view(),
            gap_energy: 0.03,
            active_pole_count: 0,
            use_broadened_pole: false,
        }),
        Err(SelfEnergyError::InvalidPoleCount)
    ));
    assert!(matches!(
        many_pole_self_energy(ManyPoleSelfEnergyInput {
            energy: Complex::new(0.4, 0.0),
            fermi_level: 0.05,
            radius: 2.1,
            pole_frequencies: frequencies.view(),
            pole_widths: widths.view(),
            amplitudes: amplitudes.view(),
            gap_energy: 0.03,
            active_pole_count: 2,
            use_broadened_pole: false,
        }),
        Err(SelfEnergyError::LengthTooShort { name: "WpScl", .. })
    ));
    assert!(matches!(
        many_pole_self_energy(ManyPoleSelfEnergyInput {
            energy: Complex::new(-10.0, 0.0),
            fermi_level: 0.05,
            radius: 2.1,
            pole_frequencies: frequencies.view(),
            pole_widths: widths.view(),
            amplitudes: amplitudes.view(),
            gap_energy: 0.03,
            active_pole_count: 1,
            use_broadened_pole: false,
        }),
        Err(SelfEnergyError::NegativeRadicand {
            name: "CSigZ ck0",
            ..
        })
    ));
}

#[test]
fn cgratr_matches_feff_reference() -> Result<(), SelfEnergyError> {
    let polynomial = cgratr(
        |q| Ok(q * q + Complex::new(0.0, 1.0) * q),
        Complex::new(0.0, 0.0),
        Complex::new(2.0, 0.0),
        1.0e-5,
        1.0e-4,
        &[],
    )?;
    assert_complex_close(
        polynomial.value,
        Complex::new(2.666_666_687_848_671, 1.999_999_991_878_729_5),
    );
    assert_real_close(polynomial.estimated_error, 1.108_565_221_527_852_5e-7);
    assert_eq!(polynomial.evaluations, 9);
    assert_eq!(polynomial.max_regions, 1);

    let oscillatory = cgratr(
        |q| Ok((Complex::new(0.0, 3.0) * q).exp() / (Complex::new(1.0, 0.0) + q * q)),
        Complex::new(0.0, 0.0),
        Complex::new(4.0, 0.0),
        1.0e-5,
        1.0e-4,
        &[],
    )?;
    assert_complex_close(
        oscillatory.value,
        Complex::new(0.065_590_780_560_835_28, 0.363_877_022_438_551_47),
    );
    assert_real_close(oscillatory.estimated_error, 1.647_949_536_992_779e-5);
    assert_eq!(oscillatory.evaluations, 45);
    assert_eq!(oscillatory.max_regions, 4);

    let split = cgratr(
        |q| Ok(q + Complex::new(0.0, 1.0) * q * q),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        1.0e-5,
        1.0e-4,
        &[0.35, 0.70],
    )?;
    assert_complex_close(
        split.value,
        Complex::new(0.499_999_996_842_525_5, 0.333_333_331_787_268_4),
    );
    assert_real_close(split.estimated_error, 2.251_264_286_488_908_6e-8);
    assert_eq!(split.evaluations, 27);
    assert_eq!(split.max_regions, 3);
    Ok(())
}

#[test]
fn cgratr_rejects_invalid_inputs() {
    assert!(matches!(
        cgratr(
            Ok::<Complex, SelfEnergyError>,
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            1.0e-5,
            1.0e-4,
            &[],
        ),
        Err(SelfEnergyError::InvalidIntegrationInterval { .. })
    ));
    assert!(matches!(
        cgratr(
            Ok::<Complex, SelfEnergyError>,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            0.0,
            1.0e-4,
            &[],
        ),
        Err(SelfEnergyError::NonPositiveTolerance { name: "abr", .. })
    ));
    assert!(matches!(
        cgratr(
            Ok::<Complex, SelfEnergyError>,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            1.0e-5,
            1.0e-4,
            &[0.7, 0.3],
        ),
        Err(SelfEnergyError::InvalidSingularity { index: 1, .. })
    ));
}

#[test]
fn make_excitation_poles_matches_feff_reference() -> Result<(), SelfEnergyError> {
    let energy_a = ndarray::arr1(&[5.0, 12.0, 25.0, 60.0, 120.0, 250.0, 500.0]);
    let loss_a = ndarray::arr1(&[0.18, 0.45, 0.32, 0.20, 0.11, 0.05, 0.02]);
    let case_a = make_excitation_poles(energy_a.view(), loss_a.view(), 12.0, 4)?;
    assert_poles_close(
        &case_a,
        &[
            (
                4.165_512_009_431_169,
                0.229_183_902_782_623_38,
                0.187_235_701_641_844_33,
            ),
            (
                11.025_700_801_851_542,
                0.229_299_271_755_288_3,
                0.456_876_736_589_728_7,
            ),
            (
                24.194_364_348_638_345,
                0.229_170_783_294_228_82,
                0.364_889_139_851_398,
            ),
            (
                128.605_604_607_891_82,
                0.229_012_708_834_526_18,
                0.101_808_025_452_920_42,
            ),
        ],
    );

    let energy_b = ndarray::arr1(&[2.5, 8.0, 18.0, 45.0, 90.0, 160.0]);
    let loss_b = ndarray::arr1(&[0.05, 0.16, 0.40, 0.24, 0.08, 0.03]);
    let case_b = make_excitation_poles(energy_b.view(), loss_b.view(), -1.0, 3)?;
    assert_poles_close(
        &case_b,
        &[
            (
                5.741_019_682_660_089,
                0.333_408_628_654_138_84,
                0.222_079_659_617_889_17,
            ),
            (
                15.153_714_460_887_954,
                0.333_333_726_361_256_63,
                0.508_556_667_140_174_8,
            ),
            (
                45.301_797_610_056_88,
                0.333_257_644_984_604_6,
                0.184_756_013_234_317_15,
            ),
        ],
    );

    let case_c = make_excitation_poles(energy_a.view(), loss_a.view(), -2.0, 5)?;
    assert_poles_close(
        &case_c,
        &[
            (
                3.720_822_084_245_034_6,
                0.147_763_245_532_865_18,
                0.134_207_458_306_101_28,
            ),
            (
                9.737_854_565_244_156,
                0.147_704_464_410_052_1,
                0.360_527_069_673_214_5,
            ),
            (
                17.277_436_390_886_596,
                0.147_715_268_479_413_24,
                0.398_169_894_810_244,
            ),
            (
                35.184_447_871_741_21,
                0.147_712_703_984_404_6,
                0.287_023_624_934_877_54,
            ),
            (
                158.585_550_889_797_47,
                0.147_547_484_169_155,
                0.082_822_840_220_122_9,
            ),
        ],
    );
    Ok(())
}

#[test]
fn make_excitation_poles_rejects_invalid_inputs() {
    let energy = ndarray::arr1(&[1.0, 2.0]);
    let loss = ndarray::arr1(&[0.1, 0.2]);
    assert!(matches!(
        make_excitation_poles(energy.view(), loss.view(), 1.0, 0),
        Err(SelfEnergyError::InvalidPoleCount)
    ));

    let short_loss = ndarray::arr1(&[0.1]);
    assert!(matches!(
        make_excitation_poles(energy.view(), short_loss.view(), 1.0, 1),
        Err(SelfEnergyError::LossGridLengthMismatch { .. })
    ));

    let non_increasing = ndarray::arr1(&[2.0, 1.0]);
    assert!(matches!(
        make_excitation_poles(non_increasing.view(), loss.view(), 1.0, 1),
        Err(SelfEnergyError::NonIncreasingLossEnergy { index: 1, .. })
    ));

    let negative_loss = ndarray::arr1(&[0.1, -0.2]);
    assert!(matches!(
        make_excitation_poles(energy.view(), negative_loss.view(), 1.0, 1),
        Err(SelfEnergyError::NegativeReal {
            name: "loss value",
            ..
        })
    ));
}

#[test]
fn finds_feff_cubic_singularities() -> Result<(), SelfEnergyError> {
    let values = find_self_energy_singularities(
        [Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)],
        [0.35, 0.02, 0.8, 0.0],
        [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
        SingularityFunction::Second,
    )?;

    assert_real_slice_close(
        &values,
        &[
            -0.5702425768022866,
            -0.5421244103394355,
            -0.03049911884380313,
        ],
    );
    Ok(())
}

#[test]
fn finds_feff_first_function_singularities() -> Result<(), SelfEnergyError> {
    let values = find_self_energy_singularities(
        [Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)],
        [0.35, 0.02, 0.8, 0.0],
        [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
        SingularityFunction::First,
    )?;

    assert_real_slice_close(
        &values,
        &[
            -0.5702425768022866,
            -0.5421244103394355,
            -0.03049911884380313,
            0.0,
            -0.0,
            0.0,
            -0.0,
        ],
    );
    Ok(())
}

#[test]
fn rejects_invalid_singularity_inputs() {
    assert_eq!(
        SingularityFunction::try_from(3),
        Err(SelfEnergyError::InvalidFunction { value: 3 })
    );
    assert!(matches!(
        find_self_energy_singularities(
            [Complex::new(f64::NAN, 0.0), Complex::new(2.0, 0.0)],
            [0.35, 0.02, 0.8, 0.0],
            [Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)],
            SingularityFunction::Second,
        ),
        Err(SelfEnergyError::NonFiniteComplex {
            name: "lower limit",
            ..
        })
    ));
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).norm() < 1.0e-14,
        "actual={actual:?} expected={expected:?} diff={}",
        (actual - expected).norm()
    );
}

fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert!(
        (actual - expected).norm() < tolerance,
        "actual={actual:?} expected={expected:?} diff={}",
        (actual - expected).norm()
    );
}

fn assert_real_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

fn assert_poles_close(actual: &[ExcitationPole], expected: &[(Real, Real, Real)]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, &(energy, amplitude, loss_height)) in actual.iter().zip(expected.iter()) {
        assert_real_close(actual.energy, energy);
        assert_real_close(actual.width, 0.1);
        assert_real_close(actual.amplitude, amplitude);
        assert_real_close(actual.loss_height, loss_height);
    }
}

fn assert_real_slice_close(actual: &[Real], expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual} expected={expected}"
        );
        if expected == 0.0 {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}
