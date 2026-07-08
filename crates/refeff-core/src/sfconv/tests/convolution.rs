use super::*;

#[test]
fn so2conv_path_average_matches_feff_reference() -> Result<(), SfconvError> {
    let (source_momentum, amplitude_reduction, phase_shift) = so2conv_path_average_inputs();

    let no_exact = sfconv_path_average(SfconvPathAverageInput {
        source_momentum: source_momentum.view(),
        amplitude_reduction: amplitude_reduction.view(),
        phase_shift: phase_shift.view(),
        previous_momentum: 1.00,
        center_momentum: 1.60,
        next_momentum: 2.30,
        momentum_step: 0.05,
    })?;
    assert_close(
        no_exact.amplitude_reduction,
        0.888_169_014_084_507_1,
        1.0e-15,
    );
    assert_close(no_exact.phase_shift, 0.136_384_976_525_821_6, 1.0e-15);
    assert_close(no_exact.normalization, 0.126_785_714_285_714_28, 1.0e-15);

    let exact = sfconv_path_average(SfconvPathAverageInput {
        source_momentum: source_momentum.view(),
        amplitude_reduction: amplitude_reduction.view(),
        phase_shift: phase_shift.view(),
        previous_momentum: 1.00,
        center_momentum: 1.50,
        next_momentum: 2.00,
        momentum_step: 0.05,
    })?;
    assert_close(exact.amplitude_reduction, 0.897_5, 1.0e-15);
    assert_close(exact.phase_shift, 0.152_5, 1.0e-15);
    assert_close(exact.normalization, 0.1, 1.0e-15);
    Ok(())
}

#[test]
fn so2conv_path_average_rejects_invalid_inputs() {
    let (source_momentum, amplitude_reduction, phase_shift) = so2conv_path_average_inputs();
    let input = SfconvPathAverageInput {
        source_momentum: source_momentum.view(),
        amplitude_reduction: amplitude_reduction.view(),
        phase_shift: phase_shift.view(),
        previous_momentum: 1.00,
        center_momentum: 1.60,
        next_momentum: 2.30,
        momentum_step: 0.05,
    };

    assert_eq!(
        sfconv_path_average(SfconvPathAverageInput {
            amplitude_reduction: array![0.1].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "source_momentum",
            left_len: 7,
            right: "amplitude_reduction",
            right_len: 1,
        })
    );
    assert_eq!(
        sfconv_path_average(SfconvPathAverageInput {
            source_momentum: array![0.75, 1.00, 0.90, 1.50, 1.75, 2.00, 2.25].view(),
            ..input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "source_momentum",
            row: 2,
            previous: 1.00,
            current: 0.90,
        })
    );
    assert_eq!(
        sfconv_path_average(SfconvPathAverageInput {
            previous_momentum: 2.00,
            center_momentum: 1.50,
            next_momentum: 2.30,
            ..input
        }),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: 2.00,
            upper: 2.30,
        })
    );
    assert_eq!(
        sfconv_path_average(SfconvPathAverageInput {
            previous_momentum: 3.00,
            center_momentum: 3.20,
            next_momentum: 3.40,
            ..input
        }),
        Err(SfconvError::ZeroDenominator {
            field: "path average normalization",
        })
    );
    assert_eq!(
        sfconv_path_average(SfconvPathAverageInput {
            momentum_step: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "momentum_step",
            value: 0.0,
        })
    );
}

#[test]
fn finds_senergies_split_points_like_feff() -> Result<(), SfconvError> {
    let candidates = array![0.90, 0.20, 1.40, 0.70, -0.10];

    let forward = sfconv_find_singularities(0.15, 1.00, candidates.view())?;
    assert_real_slice_close(&forward, &[0.20, 0.70, 0.90], 0.0);

    let reverse = sfconv_find_singularities(1.00, 0.15, candidates.view())?;
    assert_real_slice_close(&reverse, &[0.20, 0.70, 0.90], 0.0);

    let empty = sfconv_find_singularities(0.15, 0.15, candidates.view())?;
    assert!(empty.is_empty());

    let duplicate_candidates = array![0.90, 0.20, 0.20, 0.70, 0.70];
    let unique = sfconv_find_singularities(0.15, 1.00, duplicate_candidates.view())?;
    assert_real_slice_close(&unique, &[0.20, 0.70, 0.90], 0.0);
    Ok(())
}

#[test]
fn senergies_helpers_reject_invalid_inputs() {
    let context = senergies_reference_context(false);
    assert_eq!(
        sfconv_free_electron_exchange(0.0, 1.0),
        Err(SfconvError::NonPositiveScalar {
            field: "momentum",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_extrinsic_beta(
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "photoelectron_momentum",
            ..
        })
    ));
    assert!(matches!(
        sfconv_real_self_energy_derivative(
            0.36,
            SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            ..
        })
    ));
    assert!(matches!(
        sfconv_find_singularities(0.0, 1.0, array![0.2, f64::NAN].view()),
        Err(SfconvError::NonFiniteValue {
            field: "singularity candidate",
            row: 1,
            ..
        })
    ));
}

#[test]
fn grater_integrate_matches_feff_reference() -> Result<(), SfconvError> {
    assert_integral_close(
        sfconv_grater_integrate(
            |x| Ok(x.powi(4) - 2.0 * x + 1.0),
            -0.25,
            1.75,
            1.0e-6,
            1.0e-6,
            &[],
        )?,
        SfconvAdaptiveIntegral {
            value: 2.282_812_623_992_166_7,
            estimated_error: 1.651_258_862_978_011_2e-8,
            evaluations: 9,
            max_regions: 1,
        },
        1.0e-14,
    );

    assert_integral_close(
        sfconv_grater_integrate(
            |x| Ok((5.0 * x).sin() / (1.0 + x * x)),
            0.0,
            4.0,
            1.0e-6,
            1.0e-6,
            &[],
        )?,
        SfconvAdaptiveIntegral {
            value: 0.214_866_405_696_591,
            estimated_error: 2.960_202_197_766_978_5e-7,
            evaluations: 135,
            max_regions: 6,
        },
        1.0e-13,
    );

    assert_integral_close(
        sfconv_grater_integrate(
            |x| Ok((x - 0.3).abs() + 0.25 * (x - 0.8).abs()),
            -1.0,
            2.0,
            1.0e-6,
            1.0e-6,
            &[0.3, 0.8],
        )?,
        SfconvAdaptiveIntegral {
            value: 2.874_999_978_367_709_4,
            estimated_error: 1.071_163_531_207_730_6e-7,
            evaluations: 27,
            max_regions: 3,
        },
        1.0e-14,
    );
    Ok(())
}

#[test]
fn grater_integrate_rejects_invalid_inputs() {
    assert_eq!(
        sfconv_grater_integrate(Ok, 1.0, 1.0, 1.0e-6, 1.0e-6, &[]),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: 1.0,
            upper: 1.0,
        })
    );
    assert_eq!(
        sfconv_grater_integrate(Ok, 0.0, 1.0, 0.0, 1.0e-6, &[]),
        Err(SfconvError::NonPositiveTolerance {
            field: "abr",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_grater_integrate(Ok, 0.0, 1.0, 1.0e-6, 1.0e-6, &[0.5, 0.4]),
        Err(SfconvError::InvalidSingularity {
            index: 1,
            value: 0.4,
        })
    );
    assert!(matches!(
        sfconv_grater_integrate(|_| Ok(f64::NAN), 0.0, 1.0, 1.0e-6, 1.0e-6, &[]),
        Err(SfconvError::NonFiniteValue {
            field: "grater integrand",
            ..
        })
    ));
}

#[test]
fn mksat_helpers_match_feff_reference() -> Result<(), SfconvError> {
    let context = mksat_reference_context();
    let self_energy = mksat_reference_self_energy();

    assert_close(
        sfconv_extrinsic_satellite_debroadened(0.36, context, self_energy)?,
        -0.044_294_665_346_589_21,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_satellite_broadened(0.36, self_energy)?,
        0.039_176_601_376_466_56,
        1.0e-14,
    );
    assert_close(
        sfconv_interference_satellite_integrand(0.55, 0.32, 0.045, context)?,
        4.656_810_436_207_971,
        1.0e-13,
    );
    assert_close(
        sfconv_intrinsic_satellite_integrand(0.55, 0.32, 0.045, context)?,
        2.780_182_754_299_514_3,
        1.0e-13,
    );
    assert_close(
        sfconv_interference_satellite_integrand(0.55, 0.95, 0.045, context)?,
        1.568_981_693_763_851_9,
        1.0e-13,
    );

    let interference = sfconv_interference_satellite(0.75, 0.045, context)?;
    assert_close(interference.value, 0.742_287_519_666_663_1, 1.0e-12);
    assert!(interference.evaluations > 0);
    assert!(interference.max_regions > 0);

    let intrinsic = sfconv_intrinsic_satellite(0.75, 0.045, context)?;
    assert_close(intrinsic.value, 0.496_852_311_955_514_77, 1.0e-12);
    assert!(intrinsic.evaluations > 0);
    assert!(intrinsic.max_regions > 0);

    let quasiparticle = sfconv_interference_quasiparticle(0.35, 2.40, context)?;
    assert_close(quasiparticle.value, 0.882_200_373_088_965_2, 1.0e-12);
    assert!(quasiparticle.evaluations > 0);
    assert!(quasiparticle.max_regions > 0);

    assert_close(
        sfconv_interference_quasiparticle(-0.01, 2.40, context)?.value,
        0.0,
        0.0,
    );
    assert_close(
        sfconv_interference_quasiparticle_integrand(0.55, (2.0_f64 * 0.85).sqrt(), context)?,
        0.886_179_631_715_177_2,
        1.0e-13,
    );
    Ok(())
}

#[test]
fn mksat_helpers_reject_invalid_inputs() {
    let context = mksat_reference_context();
    let self_energy = mksat_reference_self_energy();
    assert_eq!(
        sfconv_extrinsic_satellite_debroadened(0.0, context, self_energy),
        Err(SfconvError::ZeroDenominator {
            field: "satellite energy",
        })
    );
    assert!(matches!(
        sfconv_interference_satellite_integrand(0.0, 0.32, 0.045, context),
        Err(SfconvError::NonPositiveScalar {
            field: "momentum",
            ..
        })
    ));
    assert!(matches!(
        sfconv_intrinsic_satellite(0.75, 0.0, context),
        Err(SfconvError::NonPositiveScalar {
            field: "satellite width",
            ..
        })
    ));
    assert!(matches!(
        sfconv_interference_quasiparticle(0.35, -1.0, context),
        Err(SfconvError::NegativeRadicand { .. })
    ));
}

#[test]
fn interpolates_spectral_function_matches_feff_interpsf_reference() -> Result<(), SfconvError> {
    let (energy, spectral_function) = interpsf_reference_inputs();
    let interpolation = sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
        energy: energy.view(),
        spectral_function: spectral_function.view(),
        output_len: 13,
    })?;

    let expected_energy = [
        -2.0,
        -1.727_590_833_333_333_4,
        -1.455_181_666_666_666_8,
        -1.182_772_5,
        -0.910_363_333_333_333_4,
        -0.637_954_166_666_666_8,
        -0.365_545,
        -0.093_135_833_333_333_42,
        0.179_273_333_333_333_17,
        0.451_682_5,
        0.724_091_666_666_666_4,
        0.996_500_833_333_333_2,
        1.268_91,
    ];
    let expected_spectral_function = [
        -0.03,
        -0.035_578_048_005_086_65,
        -0.040_441_264_512_519_18,
        -0.044_809_714_285_714_24,
        -0.048_809_091_974_223_85,
        -0.052_519_432_577_500_3,
        -0.055_996_334_265_299_72,
        -0.059_278_128_963_028_02,
        -0.062_395_108_746_383_016,
        -0.065_369_121_964_238_19,
        -0.068_218_832_777_920_12,
        -0.070_958_429_921_906_93,
        -0.073_599_999_999_999_89,
    ];

    assert_real_slice_close(&interpolation.energy, &expected_energy, 1.0e-15);
    assert_real_slice_close(
        &interpolation.spectral_function,
        &expected_spectral_function,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn interpolates_spectral_function_rejects_invalid_inputs() {
    let (energy, spectral_function) = interpsf_reference_inputs();

    assert!(matches!(
        sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
            energy: energy.view(),
            spectral_function: spectral_function.view(),
            output_len: 1,
        }),
        Err(SfconvError::CountTooSmall {
            name: "output_len",
            ..
        })
    ));

    let short_rows = Array2::from_shape_fn((7, spectral_function.ncols()).f(), |(row, column)| {
        spectral_function[(row, column)]
    });
    assert!(matches!(
        sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
            energy: energy.view(),
            spectral_function: short_rows.view(),
            output_len: 13,
        }),
        Err(SfconvError::CountMismatch {
            field: "spectral_function rows",
            actual: 7,
            expected: 8,
        })
    ));

    let short_energy = Array1::from_iter(energy.iter().copied().take(100));
    assert!(matches!(
        sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
            energy: short_energy.view(),
            spectral_function: spectral_function.view(),
            output_len: 13,
        }),
        Err(SfconvError::LengthMismatch {
            left: "energy",
            right: "spectral_function columns",
            ..
        })
    ));

    let mut bad_energy = energy.clone();
    bad_energy[10] = bad_energy[9];
    assert!(matches!(
        sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
            energy: bad_energy.view(),
            spectral_function: spectral_function.view(),
            output_len: 13,
        }),
        Err(SfconvError::NonIncreasingEnergy { row: 10, .. })
    ));
}

#[test]
fn convolve_matches_feff_sfconvsub_reference() -> Result<(), SfconvError> {
    let reference = sfconvsub_reference_inputs();

    let cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
        photoelectron_energy: 1.35,
        chemical_potential: 0.15,
        core_hole_lifetime: 0.08,
        signal_energy: reference.signal_energy.view(),
        signal: reference.signal.view(),
        spectral_energy: reference.spectral_energy.view(),
        spectral_function: reference.spectral_function.view(),
        weights: reference.weights.view(),
        asymmetric_phase: false,
        cutoff: true,
        plasma_frequency: 0.55,
    })?;
    assert_close(cutoff_phase.amplitude, 0.404_768_834_000_475_8, 1.0e-14);
    assert_close(cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

    let no_cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
        cutoff: false,
        ..sfconv_reference_input(
            reference.signal_energy.view(),
            reference.signal.view(),
            reference.spectral_energy.view(),
            reference.spectral_function.view(),
            reference.weights.view(),
        )
    })?;
    assert_close(no_cutoff_phase.amplitude, 0.405_036_447_280_840_4, 1.0e-14);
    assert_close(no_cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

    let asym_cutoff = sfconv_convolve(SfconvConvolutionInput {
        asymmetric_phase: true,
        ..sfconv_reference_input(
            reference.signal_energy.view(),
            reference.signal.view(),
            reference.spectral_energy.view(),
            reference.spectral_function.view(),
            reference.weights.view(),
        )
    })?;
    assert_close(asym_cutoff.amplitude, 0.394_308_834_584_619_57, 1.0e-14);
    assert_close(asym_cutoff.phase, 0.0, 1.0e-14);
    Ok(())
}

#[test]
fn convolve_rejects_invalid_inputs() {
    let reference = sfconvsub_reference_inputs();

    let short_signal = array![0.62, 0.82, 0.74, 0.48, 0.22];
    assert!(matches!(
        sfconv_convolve(SfconvConvolutionInput {
            signal: short_signal.view(),
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        }),
        Err(SfconvError::LengthMismatch {
            left: "signal_energy",
            ..
        })
    ));

    let bad_spectral_energy = array![-0.18, -0.04, 0.0, 0.0, 0.31, 0.55, 0.82];
    assert!(matches!(
        sfconv_convolve(SfconvConvolutionInput {
            spectral_energy: bad_spectral_energy.view(),
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "spectral_energy",
            row: 3,
            ..
        })
    ));

    let zero_asym_weight = array![0.0, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(matches!(
        sfconv_convolve(SfconvConvolutionInput {
            weights: zero_asym_weight.view(),
            asymmetric_phase: true,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        }),
        Err(SfconvError::ZeroAsymmetricWeight)
    ));
}
