use ndarray::{Array1, Array2, ShapeBuilder, array};

use crate::Real;

use super::{
    SFCONV_MKSPECTF_GRID_LEN, SFCONV_SO2CONV_MOMENTUM_GRID_LEN, SfconvAdaptiveIntegral,
    SfconvBroadenedSelfEnergyBranch, SfconvBroadenedSelfEnergyDerivativeIntegrands,
    SfconvBroadenedSelfEnergyIntegrandInput, SfconvBroadenedSelfEnergyIntegrands,
    SfconvConvolutionInput, SfconvError, SfconvExafsConvolutionInput,
    SfconvExponentialReductionInput, SfconvExtrinsicSatelliteInput, SfconvExtrinsicSatelliteMode,
    SfconvExtrinsicSatelliteSplitInput, SfconvFeffPathInterpolationInput,
    SfconvFeffPathSignalInput, SfconvKramersKronigInput, SfconvMomentumSpectralInterpolation,
    SfconvMomentumSpectralInterpolationInput, SfconvPathAverageInput,
    SfconvPhotoelectronMomentumInput, SfconvPole, SfconvQLimits,
    SfconvQuasiparticleInterferenceInput, SfconvQuasiparticlePeakInput,
    SfconvQuasiparticlePoleInput, SfconvQuasiparticleTableInput, SfconvRenormalization,
    SfconvSatelliteContext, SfconvSatelliteCorrectionInput, SfconvSatellitePoleContributionsInput,
    SfconvSatelliteSelfEnergy, SfconvSatelliteTableInput, SfconvSelfEnergyContext,
    SfconvSo2convExafsEnergyPaddingInput, SfconvSo2convExafsPreparationInput,
    SfconvSo2convMaterialInput, SfconvSo2convMaterialParameters, SfconvSo2convSelfEnergyGridInput,
    SfconvSo2convSelfEnergySampleInput, SfconvSo2convXanesPreparationInput,
    SfconvSpectralCellInput, SfconvSpectralEnergyGrid, SfconvSpectralFinalizationInput,
    SfconvSpectralInterpolationInput, SfconvSpectralTableInput, SfconvSpectralWeightsInput,
    SfconvXanesConvolutionInput, sfconv_broadened_self_energy,
    sfconv_broadened_self_energy_derivative, sfconv_broadened_self_energy_derivative_integrands,
    sfconv_broadened_self_energy_integrands, sfconv_convolve, sfconv_correct_satellite_weights,
    sfconv_coupling_potential_squared, sfconv_exafs_convolution, sfconv_exponential_reduction,
    sfconv_extrinsic_beta, sfconv_extrinsic_satellite, sfconv_extrinsic_satellite_broadened,
    sfconv_extrinsic_satellite_debroadened, sfconv_feff_path_signal,
    sfconv_finalize_spectral_table, sfconv_find_singularities, sfconv_free_electron_exchange,
    sfconv_grater_integrate, sfconv_imaginary_self_energy, sfconv_imaginary_self_energy_derivative,
    sfconv_interference_quasiparticle, sfconv_interference_quasiparticle_integrand,
    sfconv_interference_satellite, sfconv_interference_satellite_integrand,
    sfconv_interpolate_feff_path, sfconv_interpolate_momentum_spectral_function,
    sfconv_interpolate_spectral_function, sfconv_intrinsic_satellite,
    sfconv_intrinsic_satellite_integrand, sfconv_inverse_pole_dispersion,
    sfconv_kramers_kronig_real_part, sfconv_path_average, sfconv_plasma_parameters,
    sfconv_plasmon_threshold_momentum, sfconv_pole_dispersion, sfconv_pole_dispersion_derivative,
    sfconv_pole_dispersion_second_derivative, sfconv_q_limits,
    sfconv_quasiparticle_interference_amplitude, sfconv_quasiparticle_main_peak,
    sfconv_quasiparticle_pole, sfconv_quasiparticle_table, sfconv_real_self_energy,
    sfconv_real_self_energy_derivative, sfconv_real_self_energy_derivative_integrand_lower,
    sfconv_real_self_energy_derivative_integrand_middle,
    sfconv_real_self_energy_derivative_integrand_upper, sfconv_real_self_energy_integrand_lower,
    sfconv_real_self_energy_integrand_middle, sfconv_real_self_energy_integrand_upper,
    sfconv_satellite_pole_contributions, sfconv_satellite_table, sfconv_select_pole,
    sfconv_self_energy_renormalization, sfconv_so2conv_broadened_self_energy_grid,
    sfconv_so2conv_broadened_self_energy_sample, sfconv_so2conv_material_parameters,
    sfconv_so2conv_momentum_grid, sfconv_so2conv_pad_exafs_energy_grid,
    sfconv_so2conv_photoelectron_momentum, sfconv_so2conv_prepare_exafs_signal,
    sfconv_so2conv_prepare_xanes_signal, sfconv_so2conv_unbroadened_self_energy_grid,
    sfconv_so2conv_unbroadened_self_energy_sample, sfconv_spectral_cell,
    sfconv_spectral_energy_grid, sfconv_spectral_table, sfconv_spectral_weights,
    sfconv_split_extrinsic_satellite, sfconv_xanes_convolution,
};

#[test]
fn kramers_kronig_real_part_matches_feff_mkrmu_reference() -> Result<(), SfconvError> {
    let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(25);

    let real_part = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
        imaginary: imaginary.view(),
        reference_imaginary: reference_imaginary.view(),
        energy: energy.view(),
        active_len: 25,
    })?;

    let expected = [
        0.653_321_127_749_770_8,
        0.750_003_058_275_569_8,
        0.770_088_761_144_957_1,
        0.744_953_602_096_770_5,
        0.685_875_097_053_667_7,
        0.599_956_814_602_449_9,
        0.492_993_575_338_788_3,
        0.370_329_818_936_448_6,
        0.237_144_234_118_930_07,
        0.098_519_596_973_469_21,
        -0.040_581_567_325_286_456,
        -0.175_385_521_001_154_32,
        -0.301_395_336_623_902_3,
        -0.414_483_981_972_534_94,
        -0.510_982_552_336_513_5,
        -0.587_755_578_520_523_2,
        -0.642_255_441_484_044_2,
        -0.672_546_008_587_787_2,
        -0.677_279_884_911_601_4,
        -0.631_242_351_812_862_9,
        -0.631_242_351_812_862_9,
        -0.530_174_264_181_443_8,
        -0.422_544_809_832_420_15,
        -0.273_383_187_221_121_7,
        -0.036_668_636_491_773_95,
    ];
    for (actual, expected) in real_part.iter().zip(expected) {
        assert_close(*actual, expected, 1.0e-13);
    }
    Ok(())
}

#[test]
fn kramers_kronig_real_part_rejects_invalid_inputs() {
    let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(21);

    assert!(matches!(
        sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 20,
        }),
        Err(SfconvError::CountTooSmall {
            name: "active_len",
            ..
        })
    ));
    assert!(matches!(
        sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 22,
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "imaginary",
            ..
        })
    ));

    let mut bad_imaginary = imaginary.clone();
    bad_imaginary[3] = f64::NAN;
    assert!(matches!(
        sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: bad_imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 21,
        }),
        Err(SfconvError::NonFiniteValue {
            field: "imaginary",
            row: 3,
            ..
        })
    ));

    let mut bad_energy = energy.clone();
    bad_energy[5] = bad_energy[4];
    assert!(matches!(
        sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: bad_energy.view(),
            active_len: 21,
        }),
        Err(SfconvError::NonIncreasingEnergy { row: 5, .. })
    ));
}

#[test]
fn selects_pole_parameters_matches_feff_plset_reference() -> Result<(), SfconvError> {
    let (energy, weight, broadening) = plset_reference_inputs();

    assert_pole_close(
        sfconv_select_pole(3, energy.view(), weight.view(), broadening.view())?,
        SfconvPole {
            energy: 0.495,
            weight: 0.46,
            broadening: 0.048,
        },
    );
    assert_pole_close(
        sfconv_select_pole(5, energy.view(), weight.view(), broadening.view())?,
        SfconvPole {
            energy: 0.975,
            weight: 0.600_000_000_000_000_1,
            broadening: 0.1,
        },
    );
    Ok(())
}

#[test]
fn selects_pole_parameters_rejects_invalid_inputs() {
    let (energy, weight, broadening) = plset_reference_inputs();

    assert!(matches!(
        sfconv_select_pole(0, energy.view(), weight.view(), broadening.view()),
        Err(SfconvError::IndexOutOfRange {
            field: "pole",
            index: 0,
            len: 5,
        })
    ));
    assert!(matches!(
        sfconv_select_pole(6, energy.view(), weight.view(), broadening.view()),
        Err(SfconvError::IndexOutOfRange {
            field: "pole",
            index: 6,
            len: 5,
        })
    ));

    let short_weight = Array1::from_iter(weight.iter().copied().take(4));
    assert!(matches!(
        sfconv_select_pole(1, energy.view(), short_weight.view(), broadening.view()),
        Err(SfconvError::LengthMismatch {
            left: "energy",
            right: "weight",
            ..
        })
    ));

    let mut bad_energy = energy.clone();
    bad_energy[2] = f64::NAN;
    assert!(matches!(
        sfconv_select_pole(3, bad_energy.view(), weight.view(), broadening.view()),
        Err(SfconvError::NonFiniteValue {
            field: "energy",
            row: 2,
            ..
        })
    ));
}

#[test]
fn plasma_parameters_match_feff_ppset_reference() -> Result<(), SfconvError> {
    let first = sfconv_plasma_parameters(2.35)?;
    assert_close(first.fermi_momentum, 0.816_663_103_267_026_7, 1.0e-15);
    assert_close(first.fermi_energy, 0.333_469_312_118_865_2, 1.0e-15);
    assert_close(first.plasma_frequency, 0.480_793_772_651_942_2, 1.0e-15);

    let second = sfconv_plasma_parameters(0.95)?;
    assert_close(second.fermi_momentum, 2.020_166_623_871_066, 1.0e-15);
    assert_close(second.fermi_energy, 2.040_536_594_101_310_7, 1.0e-15);
    assert_close(second.plasma_frequency, 1.870_575_403_449_765_5, 1.0e-15);
    Ok(())
}

#[test]
fn plasma_parameters_reject_invalid_radius() {
    assert_eq!(
        sfconv_plasma_parameters(0.0),
        Err(SfconvError::NonPositiveScalar {
            field: "wigner_seitz_radius",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_plasma_parameters(f64::NAN),
        Err(SfconvError::NonFiniteScalar {
            field: "wigner_seitz_radius",
            ..
        })
    ));
}

#[test]
fn so2conv_material_parameters_match_feff_reference() -> Result<(), SfconvError> {
    assert_so2conv_material_close(
        sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
            core_hole_width_ev: 1.729,
            wigner_seitz_radius: 2.05,
            interstitial_potential_ev: 12.34,
            chemical_potential_ev: 18.76,
            fermi_wave_number_inv_angstrom: 1.23,
        })?,
        SfconvSo2convMaterialParameters {
            core_hole_lifetime: 0.031_769_539_461_112_17,
            interstitial_potential: 0.453_483_073_395_169_7,
            chemical_potential_offset: 0.235_928_795_072_689_63,
            fermi_wave_number: 0.650_887_783_8,
            fermi_momentum: 0.936_174_776_915_860,
            fermi_energy: 0.438_211_606_466_730_13,
            electron_concentration: 0.027_710_847_450_018_78,
            plasma_frequency: 0.590_105_735_521_106_2,
            dispersion_parameter: 0.292_141_070_977_820_1,
            initial_photoelectron_energy: 0.438_211_606_466_730_13,
            initial_photoelectron_momentum: 0.936_174_776_915_860,
            accuracy: 1.0e-4,
        },
        1.0e-15,
    );

    assert_so2conv_material_close(
        sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
            core_hole_width_ev: 5.533,
            wigner_seitz_radius: 1.42,
            interstitial_potential_ev: -3.25,
            chemical_potential_ev: 0.80,
            fermi_wave_number_inv_angstrom: 0.78,
        })?,
        SfconvSo2convMaterialParameters {
            core_hole_lifetime: 0.101_666_201_178_909,
            interstitial_potential: -0.119_434_358_876_361_54,
            chemical_potential_offset: 0.148_833_585_676_696_7,
            fermi_wave_number: 0.412_758_106_8,
            fermi_momentum: 1.351_519_924_420_783_8,
            fermi_energy: 0.913_303_053_053_180_6,
            electron_concentration: 0.083_377_017_833_289_21,
            plasma_frequency: 1.023_594_893_897_554_8,
            dispersion_parameter: 0.608_868_702_035_453_7,
            initial_photoelectron_energy: 0.913_303_053_053_180_6,
            initial_photoelectron_momentum: 1.351_519_924_420_783_8,
            accuracy: 1.0e-4,
        },
        1.0e-15,
    );
    Ok(())
}

#[test]
fn so2conv_material_parameters_reject_invalid_inputs() {
    let valid = SfconvSo2convMaterialInput {
        core_hole_width_ev: 1.729,
        wigner_seitz_radius: 2.05,
        interstitial_potential_ev: 12.34,
        chemical_potential_ev: 18.76,
        fermi_wave_number_inv_angstrom: 1.23,
    };

    assert!(matches!(
        sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
            core_hole_width_ev: f64::NAN,
            ..valid
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "core_hole_width_ev",
            ..
        })
    ));
    assert_eq!(
        sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
            wigner_seitz_radius: 0.0,
            ..valid
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "wigner_seitz_radius",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_so2conv_material_parameters(SfconvSo2convMaterialInput {
            fermi_wave_number_inv_angstrom: f64::NAN,
            ..valid
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "fermi_wave_number_inv_angstrom",
            ..
        })
    ));
}

#[test]
fn pole_dispersion_helpers_match_feff_ppole_reference() -> Result<(), SfconvError> {
    let pole_energy = 0.47;
    let dispersion_parameter = 0.28;
    let plasma_frequency = 0.62;

    assert_close(
        sfconv_pole_dispersion(0.35, pole_energy, dispersion_parameter)?,
        0.508_872_835_293_848_2,
        1.0e-15,
    );
    assert_close(
        sfconv_pole_dispersion_derivative(0.35, pole_energy, dispersion_parameter)?,
        0.234_709_915_161_871_29,
        1.0e-15,
    );
    assert_close(
        sfconv_pole_dispersion_second_derivative(0.35, pole_energy, dispersion_parameter)?,
        0.803_071_469_689_919_9,
        1.0e-15,
    );
    assert_close(
        sfconv_inverse_pole_dispersion(0.80, pole_energy, dispersion_parameter)?,
        0.922_319_683_172_048_9,
        1.0e-15,
    );
    assert_close(
        sfconv_coupling_potential_squared(
            0.35,
            plasma_frequency,
            pole_energy,
            dispersion_parameter,
        )?,
        38.745_198_544_546_376,
        1.0e-14,
    );

    assert_close(
        sfconv_pole_dispersion(1.70, pole_energy, dispersion_parameter)?,
        1.765_821_338_641_030_2,
        1.0e-15,
    );
    assert_close(
        sfconv_pole_dispersion_derivative(1.70, pole_energy, dispersion_parameter)?,
        1.660_700_284_807_318_7,
        1.0e-15,
    );
    assert_close(
        sfconv_pole_dispersion_second_derivative(1.70, pole_energy, dispersion_parameter)?,
        1.051_677_496_133_378_6,
        1.0e-15,
    );
    assert_close(
        sfconv_inverse_pole_dispersion(0.30, pole_energy, dispersion_parameter)?,
        0.0,
        0.0,
    );
    assert_close(
        sfconv_coupling_potential_squared(
            1.70,
            plasma_frequency,
            pole_energy,
            dispersion_parameter,
        )?,
        0.473_280_535_773_200_1,
        1.0e-15,
    );
    Ok(())
}

#[test]
fn pole_dispersion_helpers_reject_invalid_inputs() {
    assert!(matches!(
        sfconv_pole_dispersion(f64::NAN, 0.47, 0.28),
        Err(SfconvError::NonFiniteScalar {
            field: "momentum",
            ..
        })
    ));
    assert_eq!(
        sfconv_pole_dispersion(0.35, 0.0, 0.28),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_energy",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_coupling_potential_squared(0.0, 0.62, 0.47, 0.28),
        Err(SfconvError::NonPositiveScalar {
            field: "momentum",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_pole_dispersion(1.0, 0.47, -10.0),
        Err(SfconvError::NegativeRadicand {
            field: "pole_dispersion",
            ..
        })
    ));
}

#[test]
fn q_limits_match_feff_qlimits_reference() -> Result<(), SfconvError> {
    assert_q_limits_close(
        sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 12.0)?,
        SfconvQLimits {
            count: 3,
            q1: 0.112_905_963_336_969_05,
            q2: 1.252_615_998_981_518,
            q3: 0.926_614_797_549_310_8,
        },
        1.0e-14,
    );
    assert_q_limits_close(
        sfconv_q_limits(0.55, 0.92, 0.47, 0.28, 3.0)?,
        SfconvQLimits {
            count: 1,
            q1: 0.0,
            q2: 0.0,
            q3: 0.590_402_885_211_133_4,
        },
        1.0e-14,
    );
    assert_q_limits_close(
        sfconv_q_limits(2.40, 0.60, 0.47, 0.28, 0.75)?,
        SfconvQLimits {
            count: 3,
            q1: 0.75,
            q2: 0.75,
            q3: 4.179_832_657_474_71,
        },
        1.0e-14,
    );
    Ok(())
}

#[test]
fn q_limits_reject_invalid_inputs() {
    assert!(matches!(
        sfconv_q_limits(1.15, f64::NAN, 0.47, 0.28, 12.0),
        Err(SfconvError::NonFiniteScalar {
            field: "photoelectron_momentum",
            ..
        })
    ));
    assert_eq!(
        sfconv_q_limits(1.15, 0.0, 0.47, 0.28, 12.0),
        Err(SfconvError::NonPositiveScalar {
            field: "photoelectron_momentum",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_q_limits(1.15, 1.05, 0.47, 0.28, 0.0),
        Err(SfconvError::NonPositiveScalar {
            field: "upper_limit",
            value: 0.0,
        })
    );
}

#[test]
fn plasmon_threshold_momentum_matches_feff_qthresh_reference() -> Result<(), SfconvError> {
    assert_close(
        sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.42, 0.88)?,
        0.972_154_268_542_323_2,
        1.0e-14,
    );
    assert_close(
        sfconv_plasmon_threshold_momentum(0.75, 0.31, 0.55, 1.05)?,
        1.230_338_193_805_480_7,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn plasmon_threshold_momentum_rejects_invalid_inputs() {
    assert_eq!(
        sfconv_plasmon_threshold_momentum(0.0, 0.28, 0.42, 0.88),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_energy",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_plasmon_threshold_momentum(0.47, 0.28, 0.0, 0.88),
        Err(SfconvError::NonPositiveScalar {
            field: "fermi_energy",
            value: 0.0,
        })
    );
}

#[test]
fn so2conv_momentum_grid_matches_feff_reference() -> Result<(), SfconvError> {
    let grid = sfconv_so2conv_momentum_grid(0.816_663_103_267_026_7, 1.733_25)?;
    assert_eq!(grid.len(), SFCONV_SO2CONV_MOMENTUM_GRID_LEN);

    let expected = [
        (0, 0.908_321_792_940_324),
        (4, 1.274_956_551_633_513_3),
        (9, 1.733_25),
        (10, 1.747_693_75),
        (39, 2.166_562_5),
        (40, 2.296_556_25),
        (49, 3.466_5),
        (50, 3.813_15),
        (59, 6.933),
        (60, 8.666_25),
        (61, 12.132_75),
        (62, 17.332_5),
        (63, 51.997_5),
        (64, 173.325),
        (65, 519.975),
    ];
    for (index, expected) in expected {
        assert_close(grid[index], expected, 1.0e-15);
    }
    assert_close(grid.sum(), 937.896_733_964_701_5, 1.0e-15);
    Ok(())
}

#[test]
fn so2conv_momentum_grid_rejects_invalid_inputs() {
    assert_eq!(
        sfconv_so2conv_momentum_grid(0.0, 1.73),
        Err(SfconvError::NonPositiveScalar {
            field: "fermi_momentum",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_so2conv_momentum_grid(0.82, 0.0),
        Err(SfconvError::NonPositiveScalar {
            field: "threshold_momentum",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_so2conv_momentum_grid(0.82, 0.82),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: 0.82,
            upper: 0.82,
        })
    );
    assert!(matches!(
        sfconv_so2conv_momentum_grid(f64::NAN, 1.73),
        Err(SfconvError::NonFiniteScalar {
            field: "fermi_momentum",
            ..
        })
    ));
}

#[test]
fn so2conv_momentum_spectral_interpolation_matches_feff_reference() -> Result<(), SfconvError> {
    let inputs = so2conv_momentum_spectral_inputs();

    let below = sfconv_interpolate_momentum_spectral_function(so2conv_momentum_spectral_input(
        &inputs, 0.25,
    ))?;
    assert_momentum_spectral_close(
        &below,
        &[0.41, 0.42, 0.43, 0.44],
        &[
            [1.11, 1.12, 1.13, 1.14],
            [2.22, 2.24, 2.26, 2.28],
            [3.33, 3.36, 3.39, 3.42],
            [0.444, 0.448, 0.452, 0.456],
            [0.555, 0.560, 0.565, 0.570],
            [1.887, 1.904, 1.921, 1.938],
            [1.554, 1.568, 1.582, 1.596],
            [0.666, 0.672, 0.678, 0.684],
        ],
        &[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
        &[41.0, 51.0, 61.0, 71.0, 81.0],
    );

    let interior = sfconv_interpolate_momentum_spectral_function(so2conv_momentum_spectral_input(
        &inputs, 0.75,
    ))?;
    assert_momentum_spectral_close(
        &interior,
        &[0.16, 0.17, 0.18, 0.19],
        &[
            [1.16, 1.17, 1.18, 1.19],
            [2.32, 2.34, 2.36, 2.38],
            [3.48, 3.51, 3.54, 3.57],
            [0.464, 0.468, 0.472, 0.476],
            [0.580, 0.585, 0.590, 0.595],
            [1.972, 1.989, 2.006, 2.023],
            [1.624, 1.638, 1.652, 1.666],
            [0.696, 0.702, 0.708, 0.714],
        ],
        &[0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23],
        &[41.5, 51.5, 61.5, 71.5, 81.5],
    );

    let exact = sfconv_interpolate_momentum_spectral_function(so2conv_momentum_spectral_input(
        &inputs, 2.0,
    ))?;
    assert_momentum_spectral_close(
        &exact,
        &[0.31, 0.32, 0.33, 0.34],
        &[
            [1.31, 1.32, 1.33, 1.34],
            [2.62, 2.64, 2.66, 2.68],
            [3.93, 3.96, 3.99, 4.02],
            [0.524, 0.528, 0.532, 0.536],
            [0.655, 0.660, 0.665, 0.670],
            [2.227, 2.244, 2.261, 2.278],
            [1.834, 1.848, 1.862, 1.876],
            [0.786, 0.792, 0.798, 0.804],
        ],
        &[0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
        &[43.0, 53.0, 63.0, 73.0, 83.0],
    );

    let above = sfconv_interpolate_momentum_spectral_function(so2conv_momentum_spectral_input(
        &inputs, 4.5,
    ))?;
    assert_momentum_spectral_close(
        &above,
        &[0.41, 0.42, 0.43, 0.44],
        &[
            [1.41, 1.42, 1.43, 1.44],
            [2.82, 2.84, 2.86, 2.88],
            [4.23, 4.26, 4.29, 4.32],
            [0.564, 0.568, 0.572, 0.576],
            [0.705, 0.710, 0.715, 0.720],
            [2.397, 2.414, 2.431, 2.448],
            [1.974, 1.988, 2.002, 2.016],
            [0.846, 0.852, 0.858, 0.864],
        ],
        &[0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
        &[44.0, 54.0, 64.0, 74.0, 84.0],
    );
    Ok(())
}

#[test]
fn so2conv_momentum_spectral_interpolation_rejects_invalid_inputs() {
    let inputs = so2conv_momentum_spectral_inputs();
    let input = so2conv_momentum_spectral_input(&inputs, 0.75);

    assert_eq!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            momentum_grid: array![0.50].view(),
            energy_grid: array![[0.11, 0.12, 0.13, 0.14]].view(),
            extrinsic_quasiparticle: array![[1.11, 1.12, 1.13, 1.14]].view(),
            extrinsic_satellite: array![[2.22, 2.24, 2.26, 2.28]].view(),
            interference_quasiparticle: array![[3.33, 3.36, 3.39, 3.42]].view(),
            interference_satellite: array![[0.444, 0.448, 0.452, 0.456]].view(),
            intrinsic_satellite: array![[0.555, 0.560, 0.565, 0.570]].view(),
            clipped_extrinsic_satellite: array![[0.666, 0.672, 0.678, 0.684]].view(),
            weights: array![[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]].view(),
            self_energy_real: array![41.0].view(),
            energy_correction: array![51.0].view(),
            width: array![61.0].view(),
            renormalization_real: array![71.0].view(),
            renormalization_imag: array![81.0].view(),
            ..input
        },),
        Err(SfconvError::CountTooSmall {
            name: "momentum_grid",
            actual: 1,
            minimum: 2,
        })
    );
    assert_eq!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            energy_grid: array![[0.11, 0.12, 0.13, 0.14]].view(),
            ..input
        },),
        Err(SfconvError::CountMismatch {
            field: "energy_grid",
            actual: 1,
            expected: 4,
        })
    );
    assert_eq!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            weights: array![
                [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
                [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37],
                [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47],
            ]
            .view(),
            ..input
        },),
        Err(SfconvError::CountMismatch {
            field: "weights",
            actual: 7,
            expected: 8,
        })
    );
    assert_eq!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            self_energy_real: array![41.0, 42.0].view(),
            ..input
        },),
        Err(SfconvError::LengthMismatch {
            left: "momentum_grid",
            left_len: 4,
            right: "self_energy_real",
            right_len: 2,
        })
    );
    assert_eq!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            momentum_grid: array![0.50, 1.00, 0.75, 4.00].view(),
            ..input
        },),
        Err(SfconvError::NonIncreasingEnergy {
            field: "momentum_grid",
            row: 2,
            previous: 1.00,
            current: 0.75,
        })
    );
    assert!(matches!(
        sfconv_interpolate_momentum_spectral_function(SfconvMomentumSpectralInterpolationInput {
            intrinsic_satellite: array![
                [0.555, 0.560, 0.565, 0.570],
                [0.605, f64::NAN, 0.615, 0.620],
                [0.655, 0.660, 0.665, 0.670],
                [0.705, 0.710, 0.715, 0.720],
            ]
            .view(),
            ..input
        },),
        Err(SfconvError::NonFiniteValue {
            field: "intrinsic_satellite",
            row: 5,
            ..
        })
    ));
}

#[test]
fn so2conv_photoelectron_momentum_matches_feff_reference() -> Result<(), SfconvError> {
    let (momentum, self_energy) = so2conv_photoelectron_momentum_inputs();

    let output = sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
        momentum: momentum.view(),
        chemical_potential: 0.47,
        fermi_momentum: 0.92,
        fermi_level: 0.36,
        fermi_self_energy: 0.115,
        self_energy: self_energy.view(),
    })?;

    assert_real_slice_close(
        &output.kinetic_energy,
        &[
            0.47,
            0.531_25,
            0.389_999_999_999_999_96,
            0.806_199_999_999_999_9,
            1.075_000_000_000_000_2,
            1.521_25,
        ],
        1.0e-15,
    );
    assert_real_slice_close(
        &output.zero_order_momentum,
        &[
            1.032_666_451_474_047,
            1.090_366_910_723_174_8,
            0.952_050_418_832_952_4,
            1.318_635_658_550_154_6,
            1.508_774_337_003_384,
            1.780_140_443_897_615_6,
        ],
        1.0e-15,
    );
    assert_real_slice_close(
        &output.renormalization,
        &[
            0.803_278_688_524_59,
            1.600_000_000_000_000_5,
            0.859_353_023_909_986,
            0.907_284_768_211_920_6,
            0.877_308_140_604_871,
            0.881_481_481_481_481_3,
        ],
        1.0e-15,
    );
    assert_real_slice_close(
        &output.photoelectron_momentum,
        &[
            1.051_933_426_803_345_6,
            1.104_943_437_466_371,
            0.947_526_500_822_483_8,
            1.294_329_968_062_690_5,
            1.464_514_861_279_758_5,
            1.711_987_149_484_481_4,
        ],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn so2conv_photoelectron_momentum_rejects_invalid_inputs() {
    let (momentum, self_energy) = so2conv_photoelectron_momentum_inputs();
    let input = SfconvPhotoelectronMomentumInput {
        momentum: momentum.view(),
        chemical_potential: 0.47,
        fermi_momentum: 0.92,
        fermi_level: 0.36,
        fermi_self_energy: 0.115,
        self_energy: self_energy.view(),
    };

    assert_eq!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            momentum: array![0.0].view(),
            self_energy: array![0.09].view(),
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "momentum",
            actual: 1,
            minimum: 2,
        })
    );
    assert_eq!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            self_energy: array![0.09, 0.105].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "momentum",
            left_len: 6,
            right: "self_energy",
            right_len: 2,
        })
    );
    assert_eq!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            fermi_momentum: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "fermi_momentum",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            momentum: array![0.0, f64::NAN, 0.35, 0.82, 1.10, 1.45].view(),
            ..input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "momentum",
            row: 1,
            ..
        })
    ));
    assert_eq!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            momentum: array![0.0, 0.0].view(),
            self_energy: array![0.09, 0.105].view(),
            ..input
        }),
        Err(SfconvError::ZeroDenominator {
            field: "photoelectron momentum finite difference",
        })
    );
    assert!(matches!(
        sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
            self_energy: array![0.09, 0.105, 4.00, 0.150, 0.190, 0.250].view(),
            ..input
        }),
        Err(SfconvError::NegativeRadicand {
            field: "photoelectron momentum",
            ..
        })
    ));
}

#[test]
fn so2conv_unbroadened_self_energy_sample_matches_weighted_poles() -> Result<(), SfconvError> {
    let material = so2conv_self_energy_material();
    let pole_energy = array![0.35, 0.57];
    let pole_weight = array![0.30, 0.70];
    let pole_broadening = array![0.01, 0.02];
    let input = SfconvSo2convSelfEnergySampleInput {
        material,
        energy: 0.0,
        quasiparticle_energy: 0.85,
        photoelectron_momentum: 1.15,
        pole_count: 2,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        include_below_fermi: false,
    };

    let actual = sfconv_so2conv_unbroadened_self_energy_sample(input)?;
    let expected_poles =
        pole_weight
            .iter()
            .enumerate()
            .try_fold(0.0, |accumulator, (index, &weight)| {
                let context = SfconvSelfEnergyContext {
                    fermi_energy: material.fermi_energy,
                    fermi_momentum: material.fermi_momentum,
                    plasma_frequency: material.plasma_frequency,
                    pole_energy: pole_energy[index],
                    quasiparticle_energy: input.quasiparticle_energy,
                    photoelectron_momentum: input.photoelectron_momentum,
                    accuracy: material.accuracy,
                    pole_broadening: pole_broadening[index],
                    dispersion_parameter: material.dispersion_parameter,
                    include_below_fermi: input.include_below_fermi,
                };
                let value = sfconv_real_self_energy(input.energy, context)?.value;
                Ok::<_, SfconvError>(accumulator + weight * value)
            })?;
    let expected = expected_poles
        + sfconv_free_electron_exchange(input.photoelectron_momentum, material.fermi_momentum)?;
    assert_close(actual, expected, 1.0e-12);
    Ok(())
}

#[test]
fn so2conv_unbroadened_self_energy_grid_builds_momentum_inputs() -> Result<(), SfconvError> {
    let material = so2conv_self_energy_material();
    let pole_energy = array![0.42];
    let pole_weight = array![1.0];
    let pole_broadening = array![0.02];
    let momentum = array![0.25, 0.50];
    let input = SfconvSo2convSelfEnergyGridInput {
        momentum: momentum.view(),
        chemical_potential: 0.80,
        fermi_level: 0.45,
        material,
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        include_below_fermi: false,
    };

    let grid = sfconv_so2conv_unbroadened_self_energy_grid(input)?;
    assert_real_slice_close(&grid.kinetic_energy, &[0.831_25, 0.925], 1.0e-15);
    assert_real_slice_close(
        &grid.zero_order_momentum,
        &[
            (material.fermi_momentum.powi(2) + 2.0 * (0.831_25 - input.fermi_level)).sqrt(),
            (material.fermi_momentum.powi(2) + 2.0 * (0.925 - input.fermi_level)).sqrt(),
        ],
        1.0e-15,
    );

    let expected_fermi =
        sfconv_so2conv_unbroadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: material.fermi_energy,
            photoelectron_momentum: material.fermi_momentum,
            pole_count: input.pole_count,
            pole_energy: input.pole_energy,
            pole_weight: input.pole_weight,
            pole_broadening: input.pole_broadening,
            include_below_fermi: input.include_below_fermi,
        })?;
    assert_close(grid.fermi_self_energy, expected_fermi, 1.0e-12);

    for row in 0..momentum.len() {
        let expected =
            sfconv_so2conv_unbroadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: grid.kinetic_energy[row],
                photoelectron_momentum: grid.zero_order_momentum[row],
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        assert_close(grid.self_energy[row], expected, 1.0e-12);
    }
    Ok(())
}

#[test]
fn so2conv_broadened_self_energy_sample_matches_weighted_poles() -> Result<(), SfconvError> {
    let material = so2conv_self_energy_material();
    let pole_energy = array![0.35, 0.57];
    let pole_weight = array![0.30, 0.70];
    let pole_broadening = array![0.01, 0.02];
    let input = SfconvSo2convSelfEnergySampleInput {
        material,
        energy: 0.0,
        quasiparticle_energy: 0.85,
        photoelectron_momentum: 1.15,
        pole_count: 2,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        include_below_fermi: false,
    };

    let actual = sfconv_so2conv_broadened_self_energy_sample(input)?;
    let expected_poles =
        pole_weight
            .iter()
            .enumerate()
            .try_fold(0.0, |accumulator, (index, &weight)| {
                let context = SfconvSelfEnergyContext {
                    fermi_energy: material.fermi_energy,
                    fermi_momentum: material.fermi_momentum,
                    plasma_frequency: material.plasma_frequency,
                    pole_energy: pole_energy[index],
                    quasiparticle_energy: input.quasiparticle_energy,
                    photoelectron_momentum: input.photoelectron_momentum,
                    accuracy: material.accuracy,
                    pole_broadening: pole_broadening[index],
                    dispersion_parameter: material.dispersion_parameter,
                    include_below_fermi: input.include_below_fermi,
                };
                let value = sfconv_broadened_self_energy(input.energy, context)?.real;
                Ok::<_, SfconvError>(accumulator + weight * value)
            })?;
    let expected = expected_poles
        + sfconv_free_electron_exchange(input.photoelectron_momentum, material.fermi_momentum)?;
    assert_close(actual, expected, 1.0e-12);
    Ok(())
}

#[test]
fn so2conv_broadened_self_energy_grid_builds_momentum_inputs() -> Result<(), SfconvError> {
    let material = so2conv_self_energy_material();
    let pole_energy = array![0.42];
    let pole_weight = array![1.0];
    let pole_broadening = array![0.02];
    let momentum = array![0.25, 0.50];
    let input = SfconvSo2convSelfEnergyGridInput {
        momentum: momentum.view(),
        chemical_potential: 0.80,
        fermi_level: 0.45,
        material,
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        include_below_fermi: false,
    };

    let grid = sfconv_so2conv_broadened_self_energy_grid(input)?;
    assert_real_slice_close(&grid.kinetic_energy, &[0.831_25, 0.925], 1.0e-15);
    assert_real_slice_close(
        &grid.zero_order_momentum,
        &[
            (material.fermi_momentum.powi(2) + 2.0 * (0.831_25 - input.fermi_level)).sqrt(),
            (material.fermi_momentum.powi(2) + 2.0 * (0.925 - input.fermi_level)).sqrt(),
        ],
        1.0e-15,
    );

    let expected_fermi =
        sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: material.fermi_energy,
            photoelectron_momentum: material.fermi_momentum,
            pole_count: input.pole_count,
            pole_energy: input.pole_energy,
            pole_weight: input.pole_weight,
            pole_broadening: input.pole_broadening,
            include_below_fermi: input.include_below_fermi,
        })?;
    assert_close(grid.fermi_self_energy, expected_fermi, 1.0e-12);

    for row in 0..momentum.len() {
        let expected =
            sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
                material,
                energy: 0.0,
                quasiparticle_energy: grid.kinetic_energy[row],
                photoelectron_momentum: grid.zero_order_momentum[row],
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        assert_close(grid.self_energy[row], expected, 1.0e-12);
    }
    Ok(())
}

#[test]
fn so2conv_unbroadened_self_energy_rejects_invalid_inputs() {
    let material = so2conv_self_energy_material();
    let pole_energy = array![0.42];
    let pole_weight = array![1.0];
    let pole_broadening = array![0.02];

    assert_eq!(
        sfconv_so2conv_unbroadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: 0.85,
            photoelectron_momentum: 1.15,
            pole_count: 0,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: 0.85,
            photoelectron_momentum: 1.15,
            pole_count: 0,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        sfconv_so2conv_unbroadened_self_energy_grid(SfconvSo2convSelfEnergyGridInput {
            momentum: array![0.25].view(),
            chemical_potential: 0.80,
            fermi_level: 0.45,
            material,
            pole_count: 2,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "pole_energy",
            active_len: 2,
            len: 1,
        })
    );
    assert_eq!(
        sfconv_so2conv_broadened_self_energy_grid(SfconvSo2convSelfEnergyGridInput {
            momentum: array![0.25].view(),
            chemical_potential: 0.80,
            fermi_level: 0.45,
            material,
            pole_count: 2,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
            include_below_fermi: false,
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "pole_energy",
            active_len: 2,
            len: 1,
        })
    );
}

#[test]
fn brsigma_broadened_integrands_match_feff_formulas() -> Result<(), SfconvError> {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    let expected = [
        (
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: -7.011_705_793_259_941,
                log_imag: -0.369_504_267_018_922_97,
                atan_real: 0.325_185_453_673_107_86,
                atan_imag: 6.170_712_852_111_19,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: -13.797_429_315_272_487,
                log_imag: -0.727_099_675_343_745_2,
                atan_real: 0.070_287_942_851_676_6,
                atan_imag: 1.333_782_638_196_666_4,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HoleFermi,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: 0.953_851_591_976_764_5,
                log_imag: 0.050_266_260_982_741_846,
                atan_real: 0.000_611_333_609_869_711_8,
                atan_imag: 0.011_600_654_705_615_22,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HolePair,
            SfconvBroadenedSelfEnergyIntegrands {
                log_real: 7.129_828_634_229_525,
                log_imag: 0.375_729_127_995_351,
                atan_real: 0.324_874_938_869_935_85,
                atan_imag: 6.164_820_529_238_007,
            },
        ),
    ];

    for (branch, expected_integrands) in expected {
        let actual = sfconv_broadened_self_energy_integrands(branch, input)?;
        assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
        assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
        assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
        assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
    }
    Ok(())
}

#[test]
fn dbrsigma_broadened_derivative_integrands_match_feff_formulas() -> Result<(), SfconvError> {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    let expected = [
        (
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: 8.237_536_803_919_268,
                log_imag: 0.434_103_353_523_399_8,
                atan_real: 0.042_642_154_194_309_74,
                atan_imag: 0.809_176_689_659_211_2,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: -27.330_804_124_143_9,
                log_imag: -1.440_284_153_769_992_4,
                atan_real: 1.193_324_638_256_228,
                atan_imag: 22.644_505_154_990_576,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HoleFermi,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: -0.331_054_683_886_457,
                log_imag: -0.017_445_985_601_711,
                atan_real: -0.000_853_016_831_865_627_2,
                atan_imag: -0.016_186_830_831_466_846,
            },
        ),
        (
            SfconvBroadenedSelfEnergyBranch::HolePair,
            SfconvBroadenedSelfEnergyDerivativeIntegrands {
                log_real: 8.400_819_219_599_3,
                log_imag: 0.442_708_042_753_362_3,
                atan_real: -0.044_853_381_319_184_02,
                atan_imag: -0.851_136_892_627_315_1,
            },
        ),
    ];

    for (branch, expected_integrands) in expected {
        let actual = sfconv_broadened_self_energy_derivative_integrands(branch, input)?;
        assert_close(actual.log_real, expected_integrands.log_real, 1.0e-13);
        assert_close(actual.log_imag, expected_integrands.log_imag, 1.0e-13);
        assert_close(actual.atan_real, expected_integrands.atan_real, 1.0e-13);
        assert_close(actual.atan_imag, expected_integrands.atan_imag, 1.0e-13);
    }
    Ok(())
}

#[test]
fn brsigma_broadened_integrands_reject_invalid_inputs() {
    let input = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: -0.10,
        energy: 0.21,
        context: senergies_reference_context(false),
    };
    assert_eq!(
        sfconv_broadened_self_energy_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            input,
        ),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: -0.10,
            upper: 0.0,
        })
    );
    assert_eq!(
        sfconv_broadened_self_energy_derivative_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            input,
        ),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: -0.10,
            upper: 0.0,
        })
    );

    let zero_broadening = SfconvBroadenedSelfEnergyIntegrandInput {
        momentum: 0.73,
        energy: 0.21,
        context: SfconvSelfEnergyContext {
            pole_broadening: 0.0,
            ..senergies_reference_context(false)
        },
    };
    assert_eq!(
        sfconv_broadened_self_energy_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            zero_broadening,
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_broadened_self_energy_derivative_integrands(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            zero_broadening,
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}

#[test]
fn brsigma_broadened_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
    let cases = [
        (
            0.36,
            senergies_reference_context(false),
            -0.518_548_796_704_916_7,
            -0.820_845_165_208_279_3,
        ),
        (
            -0.20,
            senergies_reference_context(true),
            -0.276_438_440_404_569,
            -0.012_356_840_692_487_325,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                ..senergies_reference_context(false)
            },
            -0.090_781_303_269_171_75,
            -0.280_887_927_239_661_94,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                include_below_fermi: true,
                ..senergies_reference_context(false)
            },
            0.008_365_301_760_209_81,
            -0.284_132_323_784_229_7,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 1.0,
                ..senergies_reference_context(false)
            },
            0.013_728_093_655_548_983,
            -0.412_629_377_510_605_5,
        ),
    ];

    for (energy, context, expected_real, expected_imaginary) in cases {
        let actual = sfconv_broadened_self_energy(energy, context)?;
        assert_close(actual.real, expected_real, 1.0e-12);
        assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
        assert!(actual.real_estimated_error >= 0.0);
        assert!(actual.real_estimated_error < 1.0e-6);
        assert!(actual.imaginary_estimated_error >= 0.0);
        assert!(actual.imaginary_estimated_error < 1.0e-6);
        assert!(actual.evaluations > 0);
        assert!(actual.max_regions > 0);
    }
    Ok(())
}

#[test]
fn brsigma_broadened_self_energy_rejects_invalid_inputs() {
    let context = senergies_reference_context(false);
    assert!(matches!(
        sfconv_broadened_self_energy(f64::NAN, context),
        Err(SfconvError::NonFiniteScalar {
            field: "self-energy energy",
            ..
        })
    ));
    assert_eq!(
        sfconv_broadened_self_energy(
            0.36,
            SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}

#[test]
fn dbrsigma_broadened_self_energy_derivative_matches_feff_reference() -> Result<(), SfconvError> {
    let cases = [
        (
            0.36,
            senergies_reference_context(false),
            2.953_632_555_240_584,
            -4.153_776_392_437_791,
        ),
        (
            -0.20,
            senergies_reference_context(true),
            -0.453_145_835_952_415_03,
            -0.046_313_231_462_640_74,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                ..senergies_reference_context(false)
            },
            0.533_248_980_604_782_1,
            0.196_090_288_785_958_72,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 0.82,
                include_below_fermi: true,
                ..senergies_reference_context(false)
            },
            0.467_087_536_743_928_44,
            0.199_768_325_815_296_63,
        ),
        (
            0.36,
            SfconvSelfEnergyContext {
                photoelectron_momentum: 1.0,
                ..senergies_reference_context(false)
            },
            0.462_197_179_911_945_2,
            0.647_423_140_545_274,
        ),
    ];

    for (energy, context, expected_real, expected_imaginary) in cases {
        let actual = sfconv_broadened_self_energy_derivative(energy, context)?;
        assert_close(actual.real, expected_real, 1.0e-12);
        assert_close(actual.imaginary, expected_imaginary, 1.0e-12);
        assert!(actual.real_estimated_error >= 0.0);
        assert!(actual.real_estimated_error < 1.0e-6);
        assert!(actual.imaginary_estimated_error >= 0.0);
        assert!(actual.imaginary_estimated_error < 1.0e-6);
        assert!(actual.evaluations > 0);
        assert!(actual.max_regions > 0);
    }
    Ok(())
}

#[test]
fn dbrsigma_broadened_self_energy_derivative_rejects_invalid_inputs() {
    let context = senergies_reference_context(false);
    assert!(matches!(
        sfconv_broadened_self_energy_derivative(f64::NAN, context),
        Err(SfconvError::NonFiniteScalar {
            field: "self-energy energy",
            ..
        })
    ));
    assert_eq!(
        sfconv_broadened_self_energy_derivative(
            0.36,
            SfconvSelfEnergyContext {
                pole_broadening: 0.0,
                ..context
            },
        ),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_broadening",
            value: 0.0,
        })
    );
}

#[test]
fn so2conv_signal_preparation_matches_feff_reference() -> Result<(), SfconvError> {
    let exafs_energy = array![0.10, 0.22, 0.37, 0.55];
    let padded_exafs =
        sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
            energy: exafs_energy.view(),
            active_len: 4,
            output_len: 7,
        })?;
    assert_real_slice_close(
        &padded_exafs,
        &[0.10, 0.22, 0.37, 0.55, 0.73, 0.91, 1.09],
        1.0e-14,
    );
    let exafs_momentum = array![0.0, 0.1, 0.2, 0.3];
    let exafs_magnitude = array![1.0, 2.0, 3.0, 4.0];
    let exafs_phase = array![
        0.0,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        -std::f64::consts::FRAC_PI_2,
    ];
    let exafs_phase_minus_2kr = array![0.1, 0.2, 0.3, 0.4];
    let prepared_exafs = sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
        momentum: exafs_momentum.view(),
        magnitude: exafs_magnitude.view(),
        phase: exafs_phase.view(),
        phase_minus_2kr: Some(exafs_phase_minus_2kr.view()),
        chemical_potential: 0.5,
        active_len: 4,
        output_len: 6,
    })?;
    assert_real_slice_close(
        &prepared_exafs.signal_energy,
        &[0.5, 0.505, 0.52, 0.545, 0.57, 0.595],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared_exafs.real_signal,
        &[1.0, 0.0, -3.0, 0.0, 0.0, 0.0],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared_exafs.imaginary_signal,
        &[0.0, 2.0, 0.0, -4.0, 0.0, 0.0],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared_exafs.original_magnitude,
        &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared_exafs.original_phase,
        &[
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            0.0,
            0.0,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared_exafs.phase_minus_2kr,
        &[0.1, 0.2, 0.3, 0.4, 0.0, 0.0],
        1.0e-14,
    );

    let prepared_exafs_default_phase =
        sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
            phase_minus_2kr: None,
            ..SfconvSo2convExafsPreparationInput {
                momentum: exafs_momentum.view(),
                magnitude: exafs_magnitude.view(),
                phase: exafs_phase.view(),
                phase_minus_2kr: Some(exafs_phase_minus_2kr.view()),
                chemical_potential: 0.5,
                active_len: 4,
                output_len: 6,
            }
        })?;
    assert_real_slice_close(
        &prepared_exafs_default_phase.phase_minus_2kr,
        &[0.0; 6],
        1.0e-14,
    );

    let (incident_energy, excitation_energy, absorption, embedded_background) =
        so2conv_xanes_preparation_inputs();
    let prepared = sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
        incident_energy: incident_energy.view(),
        excitation_energy: excitation_energy.view(),
        absorption: absorption.view(),
        embedded_background: embedded_background.view(),
        active_len: 22,
        output_len: 25,
    })?;

    assert_real_slice_close(
        &prepared.incident_energy,
        &[
            0.202, 0.334, 0.460, 0.592, 0.724, 0.850, 0.982, 1.114, 1.240, 1.372, 1.504, 1.630,
            1.762, 1.894, 2.020, 2.152, 2.284, 2.410, 2.542, 2.674, 2.800, 2.911, 3.022, 3.133,
            3.244,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared.excitation_energy,
        &[
            -0.399, -0.288, -0.177, -0.070, 0.041, 0.152, 0.263, 0.370, 0.481, 0.592, 0.703, 0.810,
            0.921, 1.032, 1.143, 1.250, 1.361, 1.472, 1.583, 1.690, 1.801, 1.912, 2.023, 2.134,
            2.245,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared.absorption,
        &[
            1.013_002_345_457_738,
            1.040_241_406_421_492,
            1.066_864_797_635_351,
            1.088_831_359_977_982,
            1.108_791_350_567_574,
            1.123_338_851_323_157,
            1.135_831_399_724_224,
            1.143_574_970_312_228,
            1.150_575_738_690_336,
            1.154_663_226_497_332,
            1.160_192_154_165_129,
            1.165_132_358_117_229,
            1.173_757_266_916_467,
            1.183_741_568_249_87,
            1.198_877_822_449_645,
            1.216_219_972_021_848,
            1.238_859_132_580_952,
            1.263_133_973_109_753,
            1.291_474_695_964_75,
            1.319_676_423_887_3,
            1.349_794_997_826_861,
            1.315,
            1.315,
            1.315,
            1.315,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared.embedded_background,
        &[
            1.0008, 1.015, 1.0308, 1.045, 1.0608, 1.075, 1.0908, 1.105, 1.1208, 1.135, 1.1508,
            1.165, 1.1808, 1.195, 1.2108, 1.225, 1.2408, 1.255, 1.2708, 1.285, 1.3008, 1.315,
            1.315, 1.315, 1.315,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared.imaginary_fine_structure,
        &[
            0.012_202_345_457_738,
            0.025_241_406_421_492,
            0.036_064_797_635_351,
            0.043_831_359_977_982,
            0.047_991_350_567_574,
            0.048_338_851_323_157,
            0.045_031_399_724_224,
            0.038_574_970_312_228,
            0.029_775_738_690_336,
            0.019_663_226_497_332,
            0.009_392_154_165_129,
            0.000_132_358_117_229,
            -0.007_042_733_083_533,
            -0.011_258_431_750_130,
            -0.011_922_177_550_355,
            -0.008_780_027_978_152,
            -0.001_940_867_419_048,
            0.008_133_973_109_753,
            0.020_674_695_964_750,
            0.034_676_423_887_300,
            0.048_994_997_826_861,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &prepared.real_fine_structure,
        &[
            0.032_463_374_088_541,
            0.031_708_281_403_956,
            0.027_054_881_272_691,
            0.017_990_328_415_378,
            0.008_527_437_386_775,
            -0.002_125_087_125_751,
            -0.011_497_227_273_338,
            -0.020_683_431_261_378,
            -0.025_978_917_059_008,
            -0.029_016_022_387_064,
            -0.028_834_004_298_412,
            -0.025_910_106_145_618,
            -0.020_120_578_606_356,
            -0.012_652_748_322_213,
            -0.004_600_832_388_766,
            0.003_191_694_845_944,
            0.009_092_681_421_030,
            0.012_096_380_083_534,
            0.010_920_250_201_848,
            -0.009_338_141_883_948,
            -0.009_338_141_883_948,
            -0.029_208_468_871_716,
            -0.018_711_184_393_096,
            -0.014_581_157_747_772,
            -0.012_254_476_090_090,
        ],
        1.0e-14,
    );
    Ok(())
}

#[test]
fn so2conv_signal_preparation_rejects_invalid_inputs() {
    let exafs_energy = array![0.10, 0.22, 0.37, 0.55];
    assert_eq!(
        sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
            energy: exafs_energy.view(),
            active_len: 1,
            output_len: 7,
        }),
        Err(SfconvError::CountTooSmall {
            name: "active_len",
            actual: 1,
            minimum: 2,
        })
    );
    assert_eq!(
        sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
            energy: array![0.10, 0.22, 0.20].view(),
            active_len: 3,
            output_len: 5,
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "energy",
            row: 2,
            previous: 0.22,
            current: 0.20,
        })
    );
    let exafs_momentum = array![0.0, 0.1, 0.2, 0.3];
    let exafs_magnitude = array![1.0, 2.0, 3.0, 4.0];
    let exafs_phase = array![0.0, 0.1, 0.2, 0.3];
    let exafs_input = SfconvSo2convExafsPreparationInput {
        momentum: exafs_momentum.view(),
        magnitude: exafs_magnitude.view(),
        phase: exafs_phase.view(),
        phase_minus_2kr: None,
        chemical_potential: 0.5,
        active_len: 4,
        output_len: 6,
    };
    assert_eq!(
        sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
            active_len: 1,
            ..exafs_input
        }),
        Err(SfconvError::CountTooSmall {
            name: "active_len",
            actual: 1,
            minimum: 2,
        })
    );
    assert_eq!(
        sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
            momentum: array![0.3, 0.2, 0.1, 0.0].view(),
            ..exafs_input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "energy",
            row: 1,
            previous: 0.545,
            current: 0.52,
        })
    );
    assert_eq!(
        sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
            magnitude: array![1.0, 0.0, 3.0, 4.0].view(),
            ..exafs_input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "magnitude",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
            phase: array![0.0, f64::NAN, 0.2, 0.3].view(),
            ..exafs_input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "phase",
            row: 1,
            ..
        })
    ));

    let (incident_energy, excitation_energy, absorption, embedded_background) =
        so2conv_xanes_preparation_inputs();
    let input = SfconvSo2convXanesPreparationInput {
        incident_energy: incident_energy.view(),
        excitation_energy: excitation_energy.view(),
        absorption: absorption.view(),
        embedded_background: embedded_background.view(),
        active_len: 22,
        output_len: 25,
    };
    assert_eq!(
        sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
            output_len: 20,
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "output_len",
            actual: 20,
            minimum: 21,
        })
    );
    assert_eq!(
        sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
            output_len: 21,
            ..input
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "output_len",
            active_len: 22,
            len: 21,
        })
    );
    assert!(matches!(
        sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
            absorption: array![1.0, f64::NAN, 1.1, 1.2].view(),
            active_len: 4,
            output_len: 25,
            ..input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "absorption",
            row: 1,
            ..
        })
    ));
    assert_eq!(
        sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
            excitation_energy: array![0.0, 0.2, 0.1, 0.4].view(),
            active_len: 4,
            output_len: 25,
            ..input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "excitation_energy",
            row: 2,
            previous: 0.2,
            current: 0.1,
        })
    );
}

#[test]
fn so2conv_feff_path_interpolation_matches_feff_reference() -> Result<(), SfconvError> {
    let inputs = so2conv_feff_path_interpolation_inputs();

    let interpolated = sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
        source_momentum: inputs.source_momentum.view(),
        path_momentum: inputs.path_momentum.view(),
        central_phase: inputs.central_phase.view(),
        effective_amplitude: inputs.effective_amplitude.view(),
        effective_phase: inputs.effective_phase.view(),
        reduction_factor: inputs.reduction_factor.view(),
        mean_free_path: inputs.mean_free_path.view(),
    })?;

    assert_real_slice_close(
        &interpolated.central_phase,
        &[0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &interpolated.effective_amplitude,
        &[0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &interpolated.effective_phase,
        &[0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &interpolated.reduction_factor,
        &[0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &interpolated.mean_free_path,
        &[0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn so2conv_feff_path_interpolation_rejects_invalid_inputs() {
    let inputs = so2conv_feff_path_interpolation_inputs();
    let input = SfconvFeffPathInterpolationInput {
        source_momentum: inputs.source_momentum.view(),
        path_momentum: inputs.path_momentum.view(),
        central_phase: inputs.central_phase.view(),
        effective_amplitude: inputs.effective_amplitude.view(),
        effective_phase: inputs.effective_phase.view(),
        reduction_factor: inputs.reduction_factor.view(),
        mean_free_path: inputs.mean_free_path.view(),
    };

    assert_eq!(
        sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            path_momentum: array![0.25].view(),
            central_phase: array![0.10].view(),
            effective_amplitude: array![1.00].view(),
            effective_phase: array![0.50].view(),
            reduction_factor: array![0.80].view(),
            mean_free_path: array![6.00].view(),
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "path_momentum",
            actual: 1,
            minimum: 2,
        })
    );
    assert_eq!(
        sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            central_phase: array![0.10, 0.20].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "path_momentum",
            left_len: 4,
            right: "central_phase",
            right_len: 2,
        })
    );
    assert_eq!(
        sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            source_momentum: array![0.0, 0.50, 0.25].view(),
            ..input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "source_momentum",
            row: 2,
            previous: 0.50,
            current: 0.25,
        })
    );
    assert_eq!(
        sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            path_momentum: array![0.25, 0.75, 0.70, 1.75].view(),
            ..input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "path_momentum",
            row: 2,
            previous: 0.75,
            current: 0.70,
        })
    );
    assert!(matches!(
        sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
            effective_phase: array![0.50, f64::NAN, 0.60, 1.00].view(),
            ..input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "effective_phase",
            row: 1,
            ..
        })
    ));
}

#[test]
fn so2conv_feff_path_signal_matches_feff_reference() -> Result<(), SfconvError> {
    let inputs = so2conv_feff_path_interpolation_inputs();
    let signal = sfconv_feff_path_signal(SfconvFeffPathSignalInput {
        momentum: inputs.source_momentum.view(),
        central_phase: inputs.interpolated_central_phase.view(),
        effective_amplitude: inputs.interpolated_effective_amplitude.view(),
        effective_phase: inputs.interpolated_effective_phase.view(),
        reduction_factor: inputs.interpolated_reduction_factor.view(),
        mean_free_path: inputs.interpolated_mean_free_path.view(),
        degeneracy: 4.0,
        half_path_length: 3.25,
    })?;

    assert_real_slice_close(
        &signal.magnitude,
        &[
            0.536_124_841_919_397_1,
            0.410_164_018_117_519_6,
            0.284_203_194_315_642_06,
            0.251_379_063_300_987_75,
            0.174_109_626_719_572_4,
            0.125_698_646_320_718_7,
            0.153_357_484_762_483_76,
            0.179_719_087_666_981_03,
            0.0,
        ],
        1.0e-15,
    );
    assert_real_slice_close(
        &signal.phase_minus_2kr,
        &[0.0, 0.60, 0.75, 0.90, 0.80, 0.70, 1.00, 1.30, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &signal.phase,
        &[0.0, 2.225, 4.0, 5.775, 7.30, 8.825, 10.75, 12.675, 13.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &signal.real,
        &[
            0.536_124_841_919_397_1,
            -0.249_596_094_763_011_48,
            -0.185_767_604_993_480_97,
            0.219_612_030_110_783_8,
            0.091_595_160_176_783_6,
            -0.103_759_326_185_720_1,
            -0.037_283_262_995_958_43,
            0.178_659_756_510_624_74,
            0.0,
        ],
        1.0e-15,
    );
    assert_real_slice_close(
        &signal.imaginary,
        &[
            0.0,
            0.325_478_587_986_003_7,
            -0.215_085_686_632_561_95,
            -0.122_319_212_295_952_02,
            0.148_069_202_566_293_94,
            0.070_951_757_669_183,
            -0.148_756_433_249_287_2,
            0.019_484_400_822_614_257,
            0.0,
        ],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn so2conv_feff_path_signal_rejects_invalid_inputs() {
    let inputs = so2conv_feff_path_interpolation_inputs();
    let input = SfconvFeffPathSignalInput {
        momentum: inputs.source_momentum.view(),
        central_phase: inputs.interpolated_central_phase.view(),
        effective_amplitude: inputs.interpolated_effective_amplitude.view(),
        effective_phase: inputs.interpolated_effective_phase.view(),
        reduction_factor: inputs.interpolated_reduction_factor.view(),
        mean_free_path: inputs.interpolated_mean_free_path.view(),
        degeneracy: 4.0,
        half_path_length: 3.25,
    };

    assert_eq!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            momentum: array![0.0, 0.25].view(),
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "momentum",
            actual: 2,
            minimum: 3,
        })
    );
    assert_eq!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            central_phase: array![0.0, 0.10].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "momentum",
            left_len: 9,
            right: "central_phase",
            right_len: 2,
        })
    );
    assert_eq!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            momentum: array![0.0, 0.50, 0.25, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0].view(),
            ..input
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "momentum",
            row: 2,
            previous: 0.50,
            current: 0.25,
        })
    );
    assert!(matches!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            effective_amplitude: array![0.0, f64::NAN, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0]
                .view(),
            ..input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "effective_amplitude",
            row: 1,
            ..
        })
    ));
    assert_eq!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            half_path_length: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "half_path_length",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_feff_path_signal(SfconvFeffPathSignalInput {
            mean_free_path: array![0.0, 0.0, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0].view(),
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "mean_free_path",
            value: 0.0,
        })
    );
}

#[test]
fn so2conv_exafs_convolution_matches_feff_reference() -> Result<(), SfconvError> {
    let real_channel = [
        1.960_133_155_682_483_3,
        -1.493_739_884_954_432_7,
        -1.494_388_190_129_498_7,
        -1.942_505_586_276_729,
        -1.979_984_993_200_890_8,
    ];
    let imaginary_channel = [
        0.397_338_661_590_122_43,
        0.137_168_698_409_705_4,
        -0.137_577_673_742_690_1,
        -0.478_498_658_427_964_87,
        0.282_240_016_119_734_4,
    ];
    let original_magnitude = [2.4, 1.8, 1.7, 2.3, 2.6];
    let original_phase = [0.10, 0.20, 0.25, 0.30, 0.35];
    let phase_minus_2kr = [0.01, 0.02, 0.03, 0.04, 0.05];
    let expected = [
        (
            0,
            1.960_133_155_682_483_3,
            0.397_338_661_590_122_43,
            2.000_000_000_000_000_0,
            0.2,
            0.110_000_000_000_000_01,
            0.833_333_333_333_333_4,
            0.1,
            0.2,
        ),
        (
            0,
            -1.493_739_884_954_432_8,
            0.137_168_698_409_705_4,
            1.500_024_698_372_361_7,
            3.050_020_434_612_271,
            2.870_020_434_612_271,
            0.833_347_054_651_312_1,
            2.850_020_434_612_271,
            3.050_020_434_612_271,
        ),
        (
            -2,
            -1.494_388_190_129_498_6,
            -0.137_577_673_742_690_1,
            1.500_707_726_078_255_8,
            3.233_396_748_497_55,
            3.013_396_748_497_55,
            0.882_769_250_634_268_1,
            2.983_396_748_497_55,
            -3.049_788_558_682_036,
        ),
        (
            -2,
            -1.942_505_586_276_729,
            -0.478_498_658_427_964_85,
            2.000_572_147_870_119,
            3.383_114_837_790_301_5,
            3.123_114_837_790_301_7,
            0.869_813_977_334_834_3,
            3.083_114_837_790_301_7,
            -2.900_070_469_389_284_7,
        ),
        (
            0,
            -1.979_984_993_200_890_8,
            0.282_240_016_119_734_4,
            1.999_999_999_999_999_8,
            3.000_000_000_000_000_0,
            2.699_999_999_999_999_7,
            0.769_230_769_230_769_2,
            2.65,
            3.000_000_000_000_000_0,
        ),
    ];

    let mut previous_phase = 0.0;
    let mut phase_jump_count = 0;
    for row in 0..real_channel.len() {
        let actual = sfconv_exafs_convolution(SfconvExafsConvolutionInput {
            real_convolution_amplitude: real_channel[row],
            real_convolution_phase: 0.0,
            imaginary_convolution_amplitude: imaginary_channel[row],
            imaginary_convolution_phase: 0.0,
            original_magnitude: original_magnitude[row],
            original_phase: original_phase[row],
            phase_minus_2kr: phase_minus_2kr[row],
            previous_phase,
            phase_jump_count,
        })?;
        let expected_row = expected[row];

        assert_eq!(actual.phase_jump_count, expected_row.0);
        assert_close(actual.real, expected_row.1, 1.0e-15);
        assert_close(actual.imaginary, expected_row.2, 1.0e-15);
        assert_close(actual.magnitude, expected_row.3, 1.0e-15);
        assert_close(actual.output_phase, expected_row.4, 1.0e-15);
        assert_close(actual.output_phase_minus_original, expected_row.5, 1.0e-15);
        assert_close(actual.amplitude_reduction, expected_row.6, 1.0e-15);
        assert_close(actual.phase_shift, expected_row.7, 1.0e-15);
        assert_close(actual.previous_phase, expected_row.8, 1.0e-15);

        previous_phase = actual.previous_phase;
        phase_jump_count = actual.phase_jump_count;
    }

    Ok(())
}

#[test]
fn so2conv_exafs_convolution_rejects_invalid_inputs() {
    let input = SfconvExafsConvolutionInput {
        real_convolution_amplitude: 1.0,
        real_convolution_phase: 0.0,
        imaginary_convolution_amplitude: 0.2,
        imaginary_convolution_phase: 0.0,
        original_magnitude: 2.0,
        original_phase: 0.1,
        phase_minus_2kr: 0.05,
        previous_phase: 0.0,
        phase_jump_count: 0,
    };

    assert_eq!(
        sfconv_exafs_convolution(SfconvExafsConvolutionInput {
            original_magnitude: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "original_magnitude",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_exafs_convolution(SfconvExafsConvolutionInput {
            real_convolution_phase: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "real_convolution_phase",
            ..
        })
    ));
    assert_eq!(
        sfconv_exafs_convolution(SfconvExafsConvolutionInput {
            real_convolution_amplitude: -1.0,
            imaginary_convolution_amplitude: 0.0,
            previous_phase: -3.0,
            phase_jump_count: i32::MAX,
            ..input
        }),
        Err(SfconvError::PhaseJumpOverflow {
            value: i32::MAX,
            delta: 2,
        })
    );
}

#[test]
fn so2conv_xanes_convolution_matches_feff_reference() -> Result<(), SfconvError> {
    let inputs = [
        SfconvXanesConvolutionInput {
            asymmetric_phase: false,
            absorption_convolution: f64::NAN,
            embedded_background: 3.40,
            fine_structure_imaginary_amplitude: 1.80,
            fine_structure_imaginary_phase: 0.20,
            fine_structure_real_amplitude: 0.70,
            fine_structure_real_phase: 0.90,
        },
        SfconvXanesConvolutionInput {
            asymmetric_phase: false,
            absorption_convolution: f64::NAN,
            embedded_background: 2.10,
            fine_structure_imaginary_amplitude: -0.55,
            fine_structure_imaginary_phase: 2.40,
            fine_structure_real_amplitude: 1.25,
            fine_structure_real_phase: -0.35,
        },
        SfconvXanesConvolutionInput {
            asymmetric_phase: true,
            absorption_convolution: 5.25,
            embedded_background: 4.90,
            fine_structure_imaginary_amplitude: f64::NAN,
            fine_structure_imaginary_phase: f64::NAN,
            fine_structure_real_amplitude: f64::NAN,
            fine_structure_real_phase: f64::NAN,
        },
        SfconvXanesConvolutionInput {
            asymmetric_phase: true,
            absorption_convolution: -0.75,
            embedded_background: -1.10,
            fine_structure_imaginary_amplitude: f64::NAN,
            fine_structure_imaginary_phase: f64::NAN,
            fine_structure_real_amplitude: f64::NAN,
            fine_structure_real_phase: f64::NAN,
        },
    ];
    let expected = [
        (5.712_448_676_853_473, 3.40, 2.312_448_676_853_473),
        (2.076_944_284_228_370_7, 2.10, -0.023_055_715_771_629_348),
        (5.25, 4.90, 0.349_999_999_999_999_64),
        (-0.75, -1.10, 0.350_000_000_000_000_1),
    ];

    for (input, expected_row) in inputs.into_iter().zip(expected) {
        let actual = sfconv_xanes_convolution(input)?;
        assert_close(actual.absorption, expected_row.0, 1.0e-14);
        assert_close(actual.embedded_background, expected_row.1, 1.0e-14);
        assert_close(actual.fine_structure, expected_row.2, 1.0e-14);
    }

    Ok(())
}

#[test]
fn so2conv_xanes_convolution_rejects_invalid_inputs() {
    let input = SfconvXanesConvolutionInput {
        asymmetric_phase: false,
        absorption_convolution: 0.0,
        embedded_background: 3.40,
        fine_structure_imaginary_amplitude: 1.80,
        fine_structure_imaginary_phase: 0.20,
        fine_structure_real_amplitude: 0.70,
        fine_structure_real_phase: 0.90,
    };

    assert!(matches!(
        sfconv_xanes_convolution(SfconvXanesConvolutionInput {
            embedded_background: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "embedded_background",
            ..
        })
    ));
    assert!(matches!(
        sfconv_xanes_convolution(SfconvXanesConvolutionInput {
            fine_structure_real_phase: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "fine_structure_real_phase",
            ..
        })
    ));
    assert!(matches!(
        sfconv_xanes_convolution(SfconvXanesConvolutionInput {
            asymmetric_phase: true,
            absorption_convolution: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "absorption_convolution",
            ..
        })
    ));
}

#[test]
fn senergies_beta_helpers_match_feff_reference() -> Result<(), SfconvError> {
    let lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_free_electron_exchange(1.0, lowq0_context.fermi_momentum)?,
        -std::f64::consts::FRAC_1_PI,
        1.0e-15,
    );
    assert_close(
        sfconv_free_electron_exchange(1.35, lowq0_context.fermi_momentum)?,
        -0.133_662_411_513_184_28,
        1.0e-15,
    );
    assert_close(
        sfconv_extrinsic_beta(0.36, lowq0_context)?,
        0.287_008_463_933_952_74,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_beta(0.95, lowq0_context)?,
        0.099_242_494_271_372_31,
        1.0e-14,
    );
    assert_close(
        sfconv_imaginary_self_energy(0.36, lowq0_context)?,
        -0.901_663_681_812_997,
        1.0e-14,
    );

    let lowq1_context = senergies_reference_context(true);
    assert_close(sfconv_extrinsic_beta(-0.20, lowq1_context)?, 0.0, 0.0);
    assert_close(
        sfconv_extrinsic_beta(0.36, lowq1_context)?,
        0.287_008_463_933_952_74,
        1.0e-14,
    );
    assert_close(
        sfconv_imaginary_self_energy(-0.20, lowq1_context)?,
        0.0,
        0.0,
    );
    Ok(())
}

#[test]
fn senergies_real_self_energy_matches_feff_reference() -> Result<(), SfconvError> {
    let pkgt_lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_real_self_energy_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
        2.874_639_111_469_788_7,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
        5.222_817_359_927_24,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
        -8.010_746_486_392_092,
        1.0e-14,
    );
    let real_pkgt = sfconv_real_self_energy(0.36, pkgt_lowq0_context)?;
    assert_close(real_pkgt.value, -0.707_783_970_737_988_9, 1.0e-12);
    assert!(real_pkgt.evaluations > 0);
    assert!(real_pkgt.max_regions > 0);
    assert_close(
        sfconv_real_self_energy(0.95, pkgt_lowq0_context)?.value,
        0.196_748_431_942_598_25,
        1.0e-12,
    );

    let pkgt_lowq1_context = senergies_reference_context(true);
    assert_close(
        sfconv_real_self_energy(-0.20, pkgt_lowq1_context)?.value,
        -0.277_039_230_882_649,
        1.0e-12,
    );

    let pklt_lowq0_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 0.82,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
        -3.190_158_193_028_965_5,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
        0.649_162_805_914_428_2,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
        -2.194_055_192_971_564_6,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy(0.36, pklt_lowq0_context)?.value,
        -0.077_377_126_607_744_2,
        1.0e-12,
    );

    let pklt_lowq1_context = SfconvSelfEnergyContext {
        include_below_fermi: true,
        ..pklt_lowq0_context
    };
    assert_close(
        sfconv_real_self_energy_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
        -0.291_337_926_232_215_6,
        1.0e-14,
    );
    assert_close(
        sfconv_real_self_energy(0.36, pklt_lowq1_context)?.value,
        0.021_796_867_569_840_478,
        1.0e-12,
    );

    let pkeq_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 1.0,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy(0.36, pkeq_context)?.value,
        0.043_101_938_251_358_85,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn senergies_self_energy_derivatives_match_feff_reference() -> Result<(), SfconvError> {
    let pkgt_lowq0_context = senergies_reference_context(false);
    assert_close(
        sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pkgt_lowq0_context)?,
        10.732_547_867_812_46,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq0_context)?,
        18.720_823_012_355_86,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pkgt_lowq0_context)?,
        -20.398_432_856_236_53,
        1.0e-13,
    );
    let real_derivative = sfconv_real_self_energy_derivative(0.36, pkgt_lowq0_context)?;
    assert_close(real_derivative.value, 2.961_445_535_932_464, 1.0e-12);
    assert!(real_derivative.evaluations > 0);
    assert!(real_derivative.max_regions > 0);
    assert_close(
        sfconv_real_self_energy_derivative(0.95, pkgt_lowq0_context)?.value,
        -0.034_316_545_918_129_96,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq0_context)?,
        -6.610_090_947_687_186,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.95, pkgt_lowq0_context)?,
        0.400_030_527_250_079_1,
        1.0e-12,
    );

    let pkgt_lowq1_context = senergies_reference_context(true);
    assert_close(
        sfconv_real_self_energy_derivative(-0.20, pkgt_lowq1_context)?.value,
        -0.452_613_488_967_939_7,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(-0.20, pkgt_lowq1_context)?,
        0.0,
        0.0,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pkgt_lowq1_context)?,
        18.394_386_251_356_508,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pkgt_lowq1_context)?.value,
        2.951_013_422_721_360_7,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkgt_lowq1_context)?,
        -6.610_090_947_687_186,
        1.0e-12,
    );

    let pklt_lowq0_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 0.82,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_derivative_integrand_upper(0.55, 0.36, pklt_lowq0_context)?,
        17.650_634_174_439_2,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq0_context)?,
        28.479_960_013_795_644,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative_integrand_lower(0.55, 0.36, pklt_lowq0_context)?,
        -0.329_585_793_363_548_93,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pklt_lowq0_context)?.value,
        0.540_035_967_831_518_6,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq0_context)?,
        0.295_448_827_556_208_1,
        1.0e-12,
    );

    let pklt_lowq1_context = SfconvSelfEnergyContext {
        include_below_fermi: true,
        ..pklt_lowq0_context
    };
    assert_close(
        sfconv_real_self_energy_derivative_integrand_middle(0.55, 0.36, pklt_lowq1_context)?,
        27.874_936_402_523_584,
        1.0e-13,
    );
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pklt_lowq1_context)?.value,
        0.664_935_465_444_516_1,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pklt_lowq1_context)?,
        0.295_448_827_556_208_1,
        1.0e-12,
    );

    let pkeq_context = SfconvSelfEnergyContext {
        photoelectron_momentum: 1.0,
        ..senergies_reference_context(false)
    };
    assert_close(
        sfconv_real_self_energy_derivative(0.36, pkeq_context)?.value,
        0.468_906_060_872_854_14,
        1.0e-12,
    );
    assert_close(
        sfconv_imaginary_self_energy_derivative(0.36, pkeq_context)?,
        0.750_887_782_735_307_7,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn mkspectf_energy_grid_matches_feff_reference() -> Result<(), SfconvError> {
    let grid = sfconv_spectral_energy_grid(0.62)?;
    assert_eq!(grid.energy.len(), SFCONV_MKSPECTF_GRID_LEN);
    assert_eq!(grid.boundaries.len(), SFCONV_MKSPECTF_GRID_LEN + 1);

    let expected_energy = [
        (0, -3.389_333_333_333_333),
        (12, -0.992),
        (21, -0.62),
        (51, -0.000_413_333_333_333_333_3),
        (52, -0.000_206_666_666_666_666_66),
        (53, 0.000_206_666_666_666_666_66),
        (54, 0.000_413_333_333_333_333_3),
        (84, 0.62),
        (93, 1.053_999_999_999_999_8),
        (105, 3.534),
        (111, 7.253_999_999_999_6),
    ];
    for (index, expected) in expected_energy {
        assert_close(grid.energy[index], expected, 1.0e-12);
    }

    let expected_boundaries = [
        (0, -3.595_999_999_999_999),
        (1, -3.286),
        (52, -0.000_31),
        (53, 0.0),
        (54, 0.000_31),
        (111, 6.944),
        (112, 7.873_999_999_999_999),
    ];
    for (index, expected) in expected_boundaries {
        assert_close(grid.boundaries[index], expected, 1.0e-12);
    }
    assert_close(grid.boundaries[1] - grid.boundaries[0], 0.31, 1.0e-14);
    assert_close(grid.boundaries[53] - grid.boundaries[52], 0.000_31, 1.0e-16);
    assert_close(grid.boundaries[112] - grid.boundaries[111], 0.93, 1.0e-14);
    Ok(())
}

#[test]
fn mkspectf_energy_grid_rejects_invalid_inputs() {
    assert_eq!(
        sfconv_spectral_energy_grid(0.0),
        Err(SfconvError::NonPositiveScalar {
            field: "plasma_frequency",
            value: 0.0,
        })
    );
    assert!(matches!(
        sfconv_spectral_energy_grid(f64::NAN),
        Err(SfconvError::NonFiniteScalar {
            field: "plasma_frequency",
            ..
        })
    ));
}

#[test]
fn mkspectf_self_energy_renormalization_matches_feff_formula() -> Result<(), SfconvError> {
    let renormalization = sfconv_self_energy_renormalization(0.18, 0.06)?;

    assert_close(renormalization.real, 1.213_017_751_479_289_7, 1.0e-15);
    assert_close(renormalization.imaginary, 0.088_757_396_449_704_12, 1.0e-15);
    assert_close(renormalization.magnitude, 1.216_260_638_526_299_5, 1.0e-15);
    Ok(())
}

#[test]
fn mkspectf_self_energy_renormalization_rejects_invalid_inputs() {
    assert_eq!(
        sfconv_self_energy_renormalization(1.0, 0.0),
        Err(SfconvError::ZeroDenominator {
            field: "self-energy renormalization",
        })
    );
    assert!(matches!(
        sfconv_self_energy_renormalization(f64::NAN, 0.0),
        Err(SfconvError::NonFiniteScalar {
            field: "self-energy real derivative",
            ..
        })
    ));
}

#[test]
fn mkspectf_exponential_reduction_matches_feff_formula() -> Result<(), SfconvError> {
    let pole_energy = array![0.5, 0.9, 1.4, 9.0];
    let pole_weight = array![0.42, 0.36, 0.22, 0.99];

    let reduction = sfconv_exponential_reduction(SfconvExponentialReductionInput {
        plasma_frequency: 0.62,
        pole_count: 3,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
    })?;

    assert_close(reduction, 0.741_119_102_598_755_9, 1.0e-15);
    Ok(())
}

#[test]
fn mkspectf_exponential_reduction_rejects_invalid_inputs() {
    let pole_energy = array![0.5, 0.9, 1.4];
    let pole_weight = array![0.42, 0.36, 0.22];
    let input = SfconvExponentialReductionInput {
        plasma_frequency: 0.62,
        pole_count: 3,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
    };

    assert_eq!(
        sfconv_exponential_reduction(SfconvExponentialReductionInput {
            pole_count: 0,
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        sfconv_exponential_reduction(SfconvExponentialReductionInput {
            pole_count: 4,
            ..input
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "pole_energy",
            active_len: 4,
            len: 3,
        })
    );
    assert_eq!(
        sfconv_exponential_reduction(SfconvExponentialReductionInput {
            pole_energy: array![0.5, 0.0, 1.4].view(),
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "pole_energy",
            value: 0.0,
        })
    );
}

#[test]
fn mkspectf_quasiparticle_pole_matches_feff_formula() -> Result<(), SfconvError> {
    let pole = sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
        photoelectron_energy: 0.944,
        width: 0.073,
        renormalization: SfconvRenormalization {
            real: 0.82,
            imaginary: 0.06,
            magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        },
    })?;

    assert_close(pole.energy, 0.948_38, 1.0e-15);
    assert_close(pole.width, 0.059_86, 1.0e-15);
    Ok(())
}

#[test]
fn mkspectf_quasiparticle_pole_rejects_invalid_inputs() {
    let input = SfconvQuasiparticlePoleInput {
        photoelectron_energy: 0.944,
        width: 0.073,
        renormalization: SfconvRenormalization {
            real: 0.82,
            imaginary: 0.06,
            magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        },
    };

    assert_eq!(
        sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
            width: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "width",
            value: 0.0,
        })
    );
    let negative_width = sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
        renormalization: SfconvRenormalization {
            real: -0.82,
            ..input.renormalization
        },
        ..input
    });
    assert!(matches!(
        negative_width,
        Err(SfconvError::NonPositiveScalar {
            field: "quasiparticle width",
            value,
        }) if (value + 0.059_86).abs() <= 1.0e-15
    ));
    assert!(matches!(
        sfconv_quasiparticle_pole(SfconvQuasiparticlePoleInput {
            renormalization: SfconvRenormalization {
                imaginary: f64::NAN,
                ..input.renormalization
            },
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "renormalization_imag",
            ..
        })
    ));
}

#[test]
fn mkspectf_quasiparticle_interference_matches_feff_loop() -> Result<(), SfconvError> {
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];

    let interference =
        sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
            quasiparticle_energy: 0.35,
            upper_energy: 2.40,
            bare_photoelectron_energy: 0.85,
            plasma_frequency: 0.62,
            dispersion_parameter: 0.28,
            accuracy: 1.0e-4,
            interference_reduction: 0.43,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
        })?;

    assert_close(interference.amplitude, 0.132_771_156_149_889_24, 1.0e-13);
    assert!(interference.estimated_error >= 0.0);
    assert!(interference.evaluations > 0);
    assert!(interference.max_regions > 0);
    Ok(())
}

#[test]
fn mkspectf_quasiparticle_interference_rejects_invalid_inputs() {
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let input = SfconvQuasiparticleInterferenceInput {
        quasiparticle_energy: 0.35,
        upper_energy: 2.40,
        bare_photoelectron_energy: 0.85,
        plasma_frequency: 0.62,
        dispersion_parameter: 0.28,
        accuracy: 1.0e-4,
        interference_reduction: 0.43,
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
    };

    assert_eq!(
        sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
            pole_count: 0,
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
            bare_photoelectron_energy: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "bare_photoelectron_energy",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_quasiparticle_interference_amplitude(SfconvQuasiparticleInterferenceInput {
            pole_count: 3,
            ..input
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "pole_energy",
            active_len: 3,
            len: 2,
        })
    );
}

#[test]
fn mkspectf_quasiparticle_peak_matches_feff_reference() -> Result<(), SfconvError> {
    let grid = sfconv_spectral_energy_grid(0.62)?;
    let base = mkspectf_quasiparticle_peak_input(&grid, 53);

    let expected = [
        (1, 1.447_562_484_485_791_4e-3),
        (53, 3.978_159_860_663_877_3),
        (54, 3.979_528_363_928_183),
        (85, 2.074_480_177_474_116_4e-2),
        (112, 3.135_403_407_459_253_6e-4),
    ];
    for (index, expected_peak) in expected {
        let input = mkspectf_quasiparticle_peak_input(&grid, index);
        assert_close(
            sfconv_quasiparticle_main_peak(input)?,
            expected_peak,
            1.0e-13,
        );
    }
    assert_close(
        sfconv_quasiparticle_main_peak(base)?,
        3.978_159_860_663_877_3,
        1.0e-13,
    );
    Ok(())
}

#[test]
fn mkspectf_quasiparticle_peak_rejects_invalid_inputs() {
    let input = SfconvQuasiparticlePeakInput {
        center_energy: 0.0,
        lower_boundary: -0.1,
        upper_boundary: 0.1,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.9348,
        quasiparticle_width: 0.0656,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
    };

    assert_eq!(
        sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
            upper_boundary: -0.1,
            ..input
        }),
        Err(SfconvError::InvalidIntegrationInterval {
            lower: -0.1,
            upper: -0.1,
        })
    );
    assert_eq!(
        sfconv_quasiparticle_main_peak(SfconvQuasiparticlePeakInput {
            quasiparticle_width: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "quasiparticle_width",
            value: 0.0,
        })
    );
}

#[test]
fn mkspectf_quasiparticle_table_matches_feff_reference() -> Result<(), SfconvError> {
    let (energy, boundaries) = mkspectf_quasiparticle_table_grid();

    let table = sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        endpoint_width: 0.073,
        quasiparticle_width: 0.073 * 0.82,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        exponential_reduction: 0.74,
    })?;

    assert_close(table.integrated_main_weight, 0.611_144_694_397_008, 1.0e-14);
    assert_close(
        table.integrated_interference_weight,
        0.139_028_009_901_435_63,
        1.0e-14,
    );
    assert_real_slice_close(
        &table.main_peak,
        &[
            0.144_118_631_068_914_32,
            0.796_854_020_052_775_2,
            3.306_037_878_829_96,
            2.944_827_731_705_054,
            0.351_606_691_790_681_77,
            0.027_414_131_538_569_52,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &table.interference_peak,
        &[
            0.031_993_167_546_517_99,
            0.176_895_131_355_183_62,
            0.733_913_602_898_189_5,
            0.653_727_879_020_868,
            0.078_053_834_660_399_79,
            0.006_085_714_920_760_973,
        ],
        1.0e-14,
    );
    Ok(())
}

#[test]
fn mkspectf_quasiparticle_table_rejects_invalid_inputs() {
    let (energy, boundaries) = mkspectf_quasiparticle_table_grid();
    let input = SfconvQuasiparticleTableInput {
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        endpoint_width: 0.073,
        quasiparticle_width: 0.073 * 0.82,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        exponential_reduction: 0.74,
    };

    assert_eq!(
        sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
            boundaries: array![-0.55, -0.25, -0.05].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "boundaries",
            left_len: 3,
            right: "energy plus endpoints",
            right_len: 7,
        })
    );
    assert_eq!(
        sfconv_quasiparticle_table(SfconvQuasiparticleTableInput {
            endpoint_width: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "endpoint_width",
            value: 0.0,
        })
    );
}

#[test]
fn mkspectf_satellite_pole_contributions_match_feff_loop() -> Result<(), SfconvError> {
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let pole_broadening = array![0.045, 0.060];

    let contributions =
        sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
            energy: 0.75,
            uniform_width: 0.009,
            quasiparticle_width: 0.02,
            plasma_frequency: 0.62,
            bare_photoelectron_energy: 0.85,
            dispersion_parameter: 0.28,
            accuracy: 1.0e-4,
            interference_reduction: 0.43,
            include_full_broadening: false,
            pole_count: 1,
            pole_energy: pole_energy.view(),
            pole_weight: pole_weight.view(),
            pole_broadening: pole_broadening.view(),
        })?;

    assert_close(
        contributions.interference_satellite,
        0.111_714_271_709_832_78,
        1.0e-12,
    );
    assert_close(
        contributions.intrinsic_satellite,
        0.173_898_309_184_430_17,
        1.0e-12,
    );
    assert!(contributions.interference_estimated_error >= 0.0);
    assert!(contributions.intrinsic_estimated_error >= 0.0);
    assert!(contributions.evaluations > 0);
    assert!(contributions.max_regions > 0);
    Ok(())
}

#[test]
fn mkspectf_satellite_pole_contributions_rejects_invalid_inputs() {
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let pole_broadening = array![0.045, 0.060];
    let short_broadening = array![0.045];
    let input = SfconvSatellitePoleContributionsInput {
        energy: 0.75,
        uniform_width: 0.009,
        quasiparticle_width: 0.02,
        plasma_frequency: 0.62,
        bare_photoelectron_energy: 0.85,
        dispersion_parameter: 0.28,
        accuracy: 1.0e-4,
        interference_reduction: 0.43,
        include_full_broadening: false,
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
    };

    assert_eq!(
        sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
            pole_count: 0,
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
            uniform_width: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "uniform_width",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_satellite_pole_contributions(SfconvSatellitePoleContributionsInput {
            pole_count: 2,
            pole_broadening: short_broadening.view(),
            ..input
        }),
        Err(SfconvError::ActiveCountOutOfRange {
            field: "pole_broadening",
            active_len: 2,
            len: 1,
        })
    );
}

#[test]
fn mkspectf_extrinsic_satellite_modes_match_feff_branches() -> Result<(), SfconvError> {
    let input = SfconvExtrinsicSatelliteInput {
        energy: 0.36,
        main_peak: 0.0123,
        imaginary_derivative: -0.015,
        mode: SfconvExtrinsicSatelliteMode::Debroadened,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
    };

    assert_close(
        sfconv_extrinsic_satellite(input)?,
        -0.044_294_665_346_589_21,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
            mode: SfconvExtrinsicSatelliteMode::FullBroadening,
            ..input
        })?,
        0.039_176_601_376_466_56,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
            mode: SfconvExtrinsicSatelliteMode::BroadenedMinusMain,
            ..input
        })?,
        0.026_876_601_376_466_56,
        1.0e-14,
    );
    assert_close(
        sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
            mode: SfconvExtrinsicSatelliteMode::DerivativeExpansion,
            ..input
        })?,
        -0.121_822_302_119_722_35,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn mkspectf_extrinsic_satellite_rejects_invalid_inputs() {
    let input = SfconvExtrinsicSatelliteInput {
        energy: 0.36,
        main_peak: 0.0123,
        imaginary_derivative: -0.015,
        mode: SfconvExtrinsicSatelliteMode::Debroadened,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
    };

    assert!(matches!(
        sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
            main_peak: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "main_peak",
            ..
        })
    ));
    assert_eq!(
        sfconv_extrinsic_satellite(SfconvExtrinsicSatelliteInput {
            energy: 0.0,
            mode: SfconvExtrinsicSatelliteMode::DerivativeExpansion,
            ..input
        }),
        Err(SfconvError::ZeroDenominator {
            field: "derivative extrinsic satellite energy",
        })
    );
}

#[test]
fn mkspectf_spectral_cell_matches_feff_loop() -> Result<(), SfconvError> {
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let pole_broadening = array![0.045, 0.060];

    let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
        center_energy: 0.75,
        lower_boundary: 0.70,
        upper_boundary: 0.80,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        quasiparticle_width: 0.073 * 0.82,
        interference_amplitude: 0.135,
        extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
        imaginary_derivative: -0.015,
        uniform_width: 0.009,
        interference_reduction: 0.43,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
    })?;

    assert_close(cell.main_peak, 0.010_633_354_619_341_801, 1.0e-14);
    assert_close(
        cell.quasiparticle_interference,
        0.002_360_518_507_530_576,
        1.0e-14,
    );
    assert_close(
        cell.extrinsic_satellite,
        -0.008_565_813_402_423_753,
        1.0e-14,
    );
    assert_close(
        cell.interference_satellite,
        0.111_714_271_709_832_78,
        1.0e-12,
    );
    assert_close(cell.intrinsic_satellite, 0.173_898_309_184_430_17, 1.0e-12);
    assert_close(cell.combined_satellite, -0.058_096_047_637_659_13, 1.0e-12);
    assert!(cell.evaluations > 0);
    assert!(cell.max_regions > 0);
    Ok(())
}

#[test]
fn mkspectf_spectral_cell_adds_quasiparticle_for_full_broadening() -> Result<(), SfconvError> {
    let pole_energy = array![0.47];
    let pole_weight = array![1.0];
    let pole_broadening = array![0.045];

    let cell = sfconv_spectral_cell(SfconvSpectralCellInput {
        center_energy: 0.75,
        lower_boundary: 0.70,
        upper_boundary: 0.80,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        quasiparticle_width: 0.073 * 0.82,
        interference_amplitude: 0.135,
        extrinsic_mode: SfconvExtrinsicSatelliteMode::FullBroadening,
        imaginary_derivative: -0.015,
        uniform_width: 0.009,
        interference_reduction: 0.43,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
    })?;

    assert_close(
        cell.combined_satellite,
        cell.extrinsic_satellite + cell.intrinsic_satellite - 2.0 * cell.interference_satellite
            + cell.quasiparticle_interference,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn mkspectf_spectral_cell_rejects_invalid_inputs() {
    let pole_energy = array![0.47];
    let pole_weight = array![1.0];
    let pole_broadening = array![0.045];
    let input = SfconvSpectralCellInput {
        center_energy: 0.75,
        lower_boundary: 0.70,
        upper_boundary: 0.80,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        quasiparticle_width: 0.073 * 0.82,
        interference_amplitude: 0.135,
        extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
        imaginary_derivative: -0.015,
        uniform_width: 0.009,
        interference_reduction: 0.43,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
    };

    assert!(matches!(
        sfconv_spectral_cell(SfconvSpectralCellInput {
            interference_amplitude: f64::NAN,
            ..input
        }),
        Err(SfconvError::NonFiniteScalar {
            field: "interference_amplitude",
            ..
        })
    ));
    assert_eq!(
        sfconv_spectral_cell(SfconvSpectralCellInput {
            pole_count: 0,
            ..input
        }),
        Err(SfconvError::CountTooSmall {
            name: "pole_count",
            actual: 0,
            minimum: 1,
        })
    );
}

#[test]
fn mkspectf_spectral_table_matches_feff_loop() -> Result<(), SfconvError> {
    let energy = array![0.70, 0.75, 0.80];
    let boundaries = array![0.66, 0.72, 0.78, 0.84];
    let off_shell_real = array![0.028, 0.030, 0.034];
    let off_shell_imag = array![0.024, 0.025, 0.026];
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let pole_broadening = array![0.045, 0.060];
    let table = sfconv_spectral_table(SfconvSpectralTableInput {
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        quasiparticle_width: 0.073 * 0.82,
        interference_amplitude: 0.135,
        extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
        imaginary_derivative: -0.015,
        uniform_width: 0.009,
        interference_reduction: 0.43,
        exponential_reduction: 0.74,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
        off_shell_real: off_shell_real.view(),
        off_shell_imag: off_shell_imag.view(),
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        quasiparticle_lower_column_1based: 1,
        quasiparticle_upper_column_1based: 2,
    })?;

    assert_real_slice_close(
        &table.spectral_function.row(0).to_owned(),
        &[
            0.013_335_746_983_781_725,
            0.010_566_822_656_071_801,
            0.008_583_303_711_206_833,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &table.spectral_function.row(1).to_owned(),
        &[
            -0.009_414_324_593_076_924,
            -0.009_414_324_593_076_924,
            -0.007_383_327_650_441_56,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &table.spectral_function.row(2).to_owned(),
        &[
            0.002_960_427_700_746_65,
            0.002_345_748_951_142_838,
            0.001_905_423_828_262_557_8,
        ],
        1.0e-14,
    );
    assert_real_slice_close(
        &table.spectral_function.row(3).to_owned(),
        &[
            0.136_581_893_015_150_34,
            0.111_714_271_709_832_8,
            0.094_552_375_187_062_4,
        ],
        1.0e-12,
    );
    assert_real_slice_close(
        &table.spectral_function.row(4).to_owned(),
        &[
            0.233_741_598_732_050_86,
            0.173_898_309_184_430_17,
            0.133_074_376_438_518_66,
        ],
        1.0e-12,
    );
    assert_real_slice_close(
        &table.spectral_function.row(5).to_owned(),
        &[
            -0.049_685_023_081_979_92,
            -0.058_096_047_637_659_16,
            -0.063_413_701_586_047_7,
        ],
        1.0e-12,
    );
    assert_close(
        table.integrated_main_weight,
        0.603_283_822_002_286_8,
        1.0e-14,
    );
    assert_close(
        table.integrated_quasiparticle_interference_weight,
        0.180_866_035_390_252_56,
        1.0e-14,
    );
    assert_close(
        table.integrated_extrinsic_weight,
        -0.001_163_811_771_544_835_7,
        1.0e-14,
    );
    assert_close(
        table.integrated_interference_weight,
        0.015_222_475_172_094_817,
        1.0e-14,
    );
    assert_close(
        table.integrated_intrinsic_weight,
        0.024_007_714_225_361_98,
        1.0e-14,
    );
    assert_close(
        table.interference_estimated_error,
        1.045_764_905_935_126e-5,
        1.0e-16,
    );
    assert_close(
        table.intrinsic_estimated_error,
        1.406_441_733_587_394_4e-5,
        1.0e-16,
    );
    assert_eq!(table.evaluations, 396);
    assert_eq!(table.max_regions, 4);
    Ok(())
}

#[test]
fn mkspectf_spectral_table_rejects_invalid_inputs() {
    let energy = array![0.70, 0.75, 0.80];
    let boundaries = array![0.66, 0.72, 0.78, 0.84];
    let off_shell_real = array![0.028, 0.030, 0.034];
    let off_shell_imag = array![0.024, 0.025, 0.026];
    let pole_energy = array![0.47, 0.91];
    let pole_weight = array![0.35, 0.65];
    let pole_broadening = array![0.045, 0.060];
    let input = SfconvSpectralTableInput {
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.944,
        quasiparticle_width: 0.073 * 0.82,
        interference_amplitude: 0.135,
        extrinsic_mode: SfconvExtrinsicSatelliteMode::Debroadened,
        imaginary_derivative: -0.015,
        uniform_width: 0.009,
        interference_reduction: 0.43,
        exponential_reduction: 0.74,
        context: mksat_reference_context(),
        self_energy: mksat_reference_self_energy(),
        off_shell_real: off_shell_real.view(),
        off_shell_imag: off_shell_imag.view(),
        pole_count: 1,
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
        quasiparticle_lower_column_1based: 1,
        quasiparticle_upper_column_1based: 2,
    };

    assert_eq!(
        sfconv_spectral_table(SfconvSpectralTableInput {
            off_shell_imag: array![0.024, 0.025].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "off_shell_imag",
            left_len: 2,
            right: "energy",
            right_len: 3,
        })
    );
    assert!(matches!(
        sfconv_spectral_table(SfconvSpectralTableInput {
            off_shell_real: array![0.028, f64::NAN, 0.034].view(),
            ..input
        }),
        Err(SfconvError::NonFiniteValue {
            field: "off_shell_real",
            row: 1,
            ..
        })
    ));
    assert_eq!(
        sfconv_spectral_table(SfconvSpectralTableInput {
            exponential_reduction: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "exponential_reduction",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_spectral_table(SfconvSpectralTableInput {
            quasiparticle_lower_column_1based: 0,
            ..input
        }),
        Err(SfconvError::IndexOutOfRange {
            field: "quasiparticle_lower_column",
            index: 0,
            len: 3,
        })
    );
}

#[test]
fn mkspectf_satellite_table_matches_feff_reference() -> Result<(), SfconvError> {
    let inputs = mkspectf_satellite_table_inputs();

    let table = sfconv_satellite_table(SfconvSatelliteTableInput {
        main_peak: inputs.main_peak.view(),
        quasiparticle_interference: inputs.quasiparticle_interference.view(),
        extrinsic_satellite: inputs.extrinsic.view(),
        interference_satellite: inputs.interference.view(),
        intrinsic_satellite: inputs.intrinsic.view(),
        boundaries: inputs.boundaries.view(),
        quasiparticle_lower_column_1based: 3,
        quasiparticle_upper_column_1based: 4,
        include_full_broadening_quasiparticle: true,
        exponential_reduction: 0.74,
    })?;

    assert_close(
        table.integrated_extrinsic_weight,
        0.081_844_000_000_000_01,
        1.0e-15,
    );
    assert_close(table.integrated_interference_weight, 0.022_610_7, 1.0e-15);
    assert_close(
        table.integrated_intrinsic_weight,
        0.036_378_400_000_000_005,
        1.0e-15,
    );
    assert_real_slice_close(
        &table.spectral_function.row(1).to_owned(),
        &[0.04, 0.09, 0.08, 0.08, 0.13, 0.07],
        1.0e-15,
    );
    assert_real_slice_close(
        &table.spectral_function.row(3).to_owned(),
        &[0.01, 0.025, 0.006, 0.055, 0.04, 0.015],
        1.0e-15,
    );
    assert_real_slice_close(
        &table.spectral_function.row(4).to_owned(),
        &[0.02, 0.035, 0.012, 0.08, 0.065, 0.025],
        1.0e-15,
    );
    assert_real_slice_close(
        &table.spectral_function.row(5).to_owned(),
        &[
            0.071_993_167_546_518,
            0.251_895_131_355_183_6,
            0.713_913_602_898_189_5,
            0.803_727_879_020_868_1,
            0.193_053_834_660_399_8,
            0.071_085_714_920_760_98,
        ],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn mkspectf_satellite_table_rejects_invalid_inputs() {
    let inputs = mkspectf_satellite_table_inputs();
    let input = SfconvSatelliteTableInput {
        main_peak: inputs.main_peak.view(),
        quasiparticle_interference: inputs.quasiparticle_interference.view(),
        extrinsic_satellite: inputs.extrinsic.view(),
        interference_satellite: inputs.interference.view(),
        intrinsic_satellite: inputs.intrinsic.view(),
        boundaries: inputs.boundaries.view(),
        quasiparticle_lower_column_1based: 3,
        quasiparticle_upper_column_1based: 4,
        include_full_broadening_quasiparticle: true,
        exponential_reduction: 0.74,
    };

    assert_eq!(
        sfconv_satellite_table(SfconvSatelliteTableInput {
            main_peak: array![0.1, 0.2].view(),
            ..input
        }),
        Err(SfconvError::LengthMismatch {
            left: "main_peak",
            left_len: 2,
            right: "satellite columns",
            right_len: 6,
        })
    );
    assert_eq!(
        sfconv_satellite_table(SfconvSatelliteTableInput {
            quasiparticle_lower_column_1based: 0,
            ..input
        }),
        Err(SfconvError::IndexOutOfRange {
            field: "quasiparticle_lower_column",
            index: 0,
            len: 6,
        })
    );
    assert_eq!(
        sfconv_satellite_table(SfconvSatelliteTableInput {
            exponential_reduction: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "exponential_reduction",
            value: 0.0,
        })
    );
}

#[test]
fn mkspectf_extrinsic_split_matches_feff_reference() -> Result<(), SfconvError> {
    let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();

    let split = sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
        spectral_function: spectral_function.view(),
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.05,
        beta_zero: 1.0,
    })?;

    assert_eq!(split.switch_column, 5);
    assert!(split.derivative_triggered);
    assert_close(split.switch_energy, 0.35, 1.0e-15);
    assert_real_slice_close(
        &split.spectral_function.row(6).to_owned(),
        &[0.10, 0.18, 0.35, 0.30, 0.22, 0.0, 0.0, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &split.spectral_function.row(7).to_owned(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.25, 0.20],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn mkspectf_extrinsic_split_rejects_invalid_inputs() {
    let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();
    assert_eq!(
        sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
            spectral_function: Array2::<Real>::zeros((7, energy.len()).f()).view(),
            energy: energy.view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.05,
            beta_zero: 1.0,
        }),
        Err(SfconvError::CountMismatch {
            field: "spectral_function rows",
            actual: 7,
            expected: 8,
        })
    );
    assert_eq!(
        sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
            spectral_function: spectral_function.view(),
            energy: array![-0.6, -0.3, -0.4, 0.0, 0.1, 0.3, 0.6, 1.0].view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.05,
            beta_zero: 1.0,
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "energy",
            row: 2,
            previous: -0.3,
            current: -0.4,
        })
    );

    let mut flat = spectral_function.clone();
    flat.row_mut(1).fill(0.1);
    assert_eq!(
        sfconv_split_extrinsic_satellite(SfconvExtrinsicSatelliteSplitInput {
            spectral_function: flat.view(),
            energy: energy.view(),
            boundaries: boundaries.view(),
            photoelectron_energy: 0.05,
            beta_zero: 1.0,
        }),
        Err(SfconvError::MissingTrigger {
            field: "extrinsic satellite split",
        })
    );
}

#[test]
fn mkspectf_satellite_correction_matches_feff_reference() -> Result<(), SfconvError> {
    let (spectral_function, boundaries) = mkspectf_satellite_correction_inputs();

    let correction = sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
        spectral_function: spectral_function.view(),
        boundaries: boundaries.view(),
        uniform_width: 0.2,
        exponential_reduction: 0.73,
    })?;

    assert_close(correction.uncorrected_satellite_weight, 0.267, 1.0e-15);
    assert_close(
        correction.clipped_negative_weight,
        -0.053_999_999_999_999_99,
        1.0e-15,
    );
    assert_close(
        correction.correction_factor,
        0.831_775_700_934_579_4,
        1.0e-15,
    );
    assert_real_slice_close(
        &correction.weights,
        &[
            0.259_15,
            0.119_355_000_000_000_03,
            0.174_470_000_000_000_01,
            0.036_5,
            0.054_02,
        ],
        1.0e-14,
    );

    let expected_rows = [
        (0, 0.121_028_037_383_177_58, 0.25),
        (1, 0.11, 0.0),
        (2, 0.088_411_214_953_271_03, 0.1),
        (3, 0.265, 0.0),
        (4, 0.090_373_831_775_700_99, 0.48),
        (5, 0.048_504_672_897_196_26, 0.220_000_000_000_000_03),
    ];
    for (column, expected_interference, expected_combined) in expected_rows {
        assert_close(
            correction.spectral_function[(3, column)],
            expected_interference,
            1.0e-14,
        );
        assert_close(
            correction.spectral_function[(5, column)],
            expected_combined,
            1.0e-14,
        );
    }
    Ok(())
}

#[test]
fn mkspectf_satellite_correction_rejects_invalid_inputs() {
    let (spectral_function, boundaries) = mkspectf_satellite_correction_inputs();
    assert_eq!(
        sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
            spectral_function: Array2::<Real>::zeros((7, spectral_function.ncols()).f()).view(),
            boundaries: boundaries.view(),
            uniform_width: 0.2,
            exponential_reduction: 0.73,
        }),
        Err(SfconvError::CountMismatch {
            field: "spectral_function rows",
            actual: 7,
            expected: 8,
        })
    );
    assert_eq!(
        sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
            spectral_function: spectral_function.view(),
            boundaries: array![0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6].view(),
            uniform_width: 0.2,
            exponential_reduction: 0.73,
        }),
        Err(SfconvError::NonIncreasingEnergy {
            field: "boundaries",
            row: 2,
            previous: 0.2,
            current: 0.1,
        })
    );
    assert_eq!(
        sfconv_correct_satellite_weights(SfconvSatelliteCorrectionInput {
            spectral_function: Array2::<Real>::zeros((8, 2).f()).view(),
            boundaries: array![0.0, 0.2, 0.4].view(),
            uniform_width: 0.2,
            exponential_reduction: 0.73,
        }),
        Err(SfconvError::ZeroDenominator {
            field: "satellite correction",
        })
    );
}

#[test]
fn mkspectf_spectral_finalization_matches_feff_sequence() -> Result<(), SfconvError> {
    let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();

    let finalization = sfconv_finalize_spectral_table(SfconvSpectralFinalizationInput {
        spectral_function: spectral_function.view(),
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.05,
        beta_zero: 1.0,
        uniform_width: 0.2,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        interference_reduction: 0.43,
        exponential_reduction: 0.73,
    })?;

    assert_eq!(finalization.switch_column, 5);
    assert!(finalization.derivative_triggered);
    assert_close(finalization.switch_energy, 0.35, 1.0e-15);
    assert_close(finalization.uncorrected_satellite_weight, 0.558_5, 1.0e-15);
    assert_close(finalization.clipped_negative_weight, 0.0, 1.0e-15);
    assert_close(finalization.correction_factor, 1.0, 1.0e-15);
    assert_real_slice_close(
        &finalization.spectral_function.row(5).to_owned(),
        &[0.12, 0.23, 0.46, 0.46, 0.35, 0.24, 0.37, 0.27],
        1.0e-15,
    );
    assert_real_slice_close(
        &finalization.spectral_function.row(6).to_owned(),
        &[0.10, 0.18, 0.35, 0.30, 0.22, 0.0, 0.0, 0.0],
        1.0e-15,
    );
    assert_real_slice_close(
        &finalization.spectral_function.row(7).to_owned(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.25, 0.20],
        1.0e-15,
    );
    assert_real_slice_close(
        &finalization.weights,
        &[
            0.598_599_999_999_999_9,
            0.043_8,
            0.057_140_268_951_075_84,
            0.288_715,
            0.0,
            0.118_99,
            0.087_6,
            0.167_900_000_000_000_02,
        ],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn mkspectf_spectral_finalization_rejects_invalid_inputs() {
    let (spectral_function, energy, boundaries) = mkspectf_extrinsic_split_inputs();
    let input = SfconvSpectralFinalizationInput {
        spectral_function: spectral_function.view(),
        energy: energy.view(),
        boundaries: boundaries.view(),
        photoelectron_energy: 0.05,
        beta_zero: 1.0,
        uniform_width: 0.2,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        interference_reduction: 0.43,
        exponential_reduction: 0.73,
    };

    assert_eq!(
        sfconv_finalize_spectral_table(SfconvSpectralFinalizationInput {
            spectral_function: Array2::<Real>::zeros((7, energy.len()).f()).view(),
            ..input
        }),
        Err(SfconvError::CountMismatch {
            field: "spectral_function rows",
            actual: 7,
            expected: 8,
        })
    );
    assert_eq!(
        sfconv_finalize_spectral_table(SfconvSpectralFinalizationInput {
            renormalization_magnitude: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "renormalization_magnitude",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_finalize_spectral_table(SfconvSpectralFinalizationInput {
            uniform_width: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "uniform_width",
            value: 0.0,
        })
    );
}

#[test]
fn mkspectf_spectral_weights_match_feff_reference() -> Result<(), SfconvError> {
    let satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];

    let weights = sfconv_spectral_weights(SfconvSpectralWeightsInput {
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        interference_reduction: 0.43,
        exponential_reduction: 0.74,
        satellite_weights: satellite_weights.view(),
    })?;

    assert_real_slice_close(
        &weights,
        &[
            0.606_8,
            0.044_4,
            0.057_923_012_361_364_55,
            0.259_15,
            0.119_355,
            0.174_47,
            0.036_5,
            0.054_02,
        ],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn mkspectf_spectral_weights_rejects_invalid_inputs() {
    let satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];
    let input = SfconvSpectralWeightsInput {
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
        interference_amplitude: 0.135,
        interference_reduction: 0.43,
        exponential_reduction: 0.74,
        satellite_weights: satellite_weights.view(),
    };

    assert_eq!(
        sfconv_spectral_weights(SfconvSpectralWeightsInput {
            satellite_weights: array![0.1, 0.2].view(),
            ..input
        }),
        Err(SfconvError::CountMismatch {
            field: "satellite_weights",
            actual: 2,
            expected: 5,
        })
    );
    assert_eq!(
        sfconv_spectral_weights(SfconvSpectralWeightsInput {
            renormalization_magnitude: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "renormalization_magnitude",
            value: 0.0,
        })
    );
    assert_eq!(
        sfconv_spectral_weights(SfconvSpectralWeightsInput {
            exponential_reduction: 0.0,
            ..input
        }),
        Err(SfconvError::NonPositiveScalar {
            field: "exponential_reduction",
            value: 0.0,
        })
    );
}

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

fn mkrmu_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let indices = (1..=count).map(|index| index as Real);
    let imaginary = Array1::from_iter(
        indices
            .clone()
            .map(|index| (0.17 * index).sin() + 0.01 * index),
    );
    let reference_imaginary =
        Array1::from_iter(indices.clone().map(|index| 0.2 * (0.11 * index).cos()));
    let energy = Array1::from_iter((0..count).map(|index| {
        let index = index as Real;
        0.05 * index + 0.002 * index * index
    }));
    (imaginary, reference_imaginary, energy)
}

fn plset_reference_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let energy = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.12 * i + 0.015 * i * i
    });
    let weight = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.25 + 0.07 * i
    });
    let broadening = Array1::from_shape_fn(5, |index| {
        let i = index as Real + 1.0;
        0.01 * i + 0.002 * i * i
    });
    (energy, weight, broadening)
}

fn interpsf_reference_inputs() -> (Array1<Real>, Array2<Real>) {
    let count = 110usize;
    let energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real;
        -2.0 + 0.018 * i + 0.000_11 * i * i
    });
    let spectral_function = Array2::from_shape_fn((8, count).f(), |(row, column)| {
        let fortran_row = row as Real + 1.0;
        let i = column as Real;
        0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
    });
    (energy, spectral_function)
}

struct SfconvSubReference {
    spectral_energy: Array1<Real>,
    spectral_function: Array1<Real>,
    signal_energy: Array1<Real>,
    signal: Array1<Real>,
    weights: Array1<Real>,
}

fn sfconvsub_reference_inputs() -> SfconvSubReference {
    SfconvSubReference {
        spectral_energy: array![-0.18, -0.04, 0.0, 0.12, 0.31, 0.55, 0.82],
        spectral_function: array![0.05, 0.18, 0.30, 0.23, 0.14, 0.07, 0.02],
        signal_energy: array![0.40, 0.72, 0.95, 1.22, 1.58, 1.95],
        signal: array![0.62, 0.82, 0.74, 0.48, 0.22, 0.12],
        weights: array![0.72, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

fn sfconv_reference_input<'a>(
    signal_energy: ndarray::ArrayView1<'a, Real>,
    signal: ndarray::ArrayView1<'a, Real>,
    spectral_energy: ndarray::ArrayView1<'a, Real>,
    spectral_function: ndarray::ArrayView1<'a, Real>,
    weights: ndarray::ArrayView1<'a, Real>,
) -> SfconvConvolutionInput<'a> {
    SfconvConvolutionInput {
        photoelectron_energy: 1.35,
        chemical_potential: 0.15,
        core_hole_lifetime: 0.08,
        signal_energy,
        signal,
        spectral_energy,
        spectral_function,
        weights,
        asymmetric_phase: false,
        cutoff: true,
        plasma_frequency: 0.55,
    }
}

fn mkspectf_quasiparticle_peak_input(
    grid: &SfconvSpectralEnergyGrid,
    index_1based: usize,
) -> SfconvQuasiparticlePeakInput {
    let index = index_1based - 1;
    SfconvQuasiparticlePeakInput {
        center_energy: grid.energy[index],
        lower_boundary: grid.boundaries[index],
        upper_boundary: grid.boundaries[index + 1],
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.93 + 0.08 * 0.06,
        quasiparticle_width: 0.08 * 0.82,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
    }
}

fn mkspectf_quasiparticle_table_grid() -> (Array1<Real>, Array1<Real>) {
    let energy = array![-0.40, -0.12, -0.01, 0.02, 0.20, 0.55];
    let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    (energy, boundaries)
}

struct MkspectfSatelliteTableInputs {
    main_peak: Array1<Real>,
    quasiparticle_interference: Array1<Real>,
    extrinsic: Array1<Real>,
    interference: Array1<Real>,
    intrinsic: Array1<Real>,
    boundaries: Array1<Real>,
}

fn mkspectf_satellite_table_inputs() -> MkspectfSatelliteTableInputs {
    let main_peak = array![
        0.144_118_631_068_914_32,
        0.796_854_020_052_775_2,
        3.306_037_878_829_96,
        2.944_827_731_705_054,
        0.351_606_691_790_681_77,
        0.027_414_131_538_569_52,
    ];
    let quasiparticle_interference = array![
        0.031_993_167_546_517_99,
        0.176_895_131_355_183_62,
        0.733_913_602_898_189_5,
        0.653_727_879_020_868,
        0.078_053_834_660_399_79,
        0.006_085_714_920_760_973,
    ];
    let extrinsic = array![0.04, 0.09, -0.02, 0.18, 0.13, 0.07];
    let interference = array![0.01, 0.025, 0.006, 0.055, 0.04, 0.015];
    let intrinsic = array![0.02, 0.035, 0.012, 0.08, 0.065, 0.025];
    let boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    MkspectfSatelliteTableInputs {
        main_peak,
        quasiparticle_interference,
        extrinsic,
        interference,
        intrinsic,
        boundaries,
    }
}

fn mkspectf_extrinsic_split_inputs() -> (Array2<Real>, Array1<Real>, Array1<Real>) {
    let mut spectral_function = Array2::<Real>::zeros((8, 8).f());
    for (row, values) in [
        (1, [0.10, 0.18, 0.35, 0.30, 0.22, 0.15, 0.25, 0.20]),
        (4, [0.02, 0.05, 0.11, 0.16, 0.13, 0.09, 0.12, 0.07]),
        (6, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
        (7, [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            spectral_function[(row, column)] = value;
        }
    }
    let energy = array![-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0];
    let boundaries = array![-0.75, -0.45, -0.20, -0.05, 0.05, 0.20, 0.45, 0.80, 1.20];
    (spectral_function, energy, boundaries)
}

fn mkspectf_satellite_correction_inputs() -> (Array2<Real>, Array1<Real>) {
    let mut spectral_function = Array2::<Real>::zeros((8, 6).f());
    for (row, values) in [
        (1, [0.40, 0.18, 0.06, 0.50, 0.28, 0.08]),
        (3, [0.10, 0.16, 0.08, 0.35, 0.05, 0.03]),
        (4, [0.05, 0.04, 0.20, 0.03, 0.30, 0.20]),
        (6, [0.08, 0.05, 0.03, 0.12, 0.07, 0.02]),
        (7, [0.04, 0.02, 0.01, 0.06, 0.09, 0.03]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            spectral_function[(row, column)] = value;
        }
    }
    let boundaries = array![-0.4, -0.2, 0.0, 0.15, 0.35, 0.7, 1.1];
    (spectral_function, boundaries)
}

struct So2convMomentumSpectralInputs {
    momentum_grid: Array1<Real>,
    energy_grid: Array2<Real>,
    extrinsic_quasiparticle: Array2<Real>,
    extrinsic_satellite: Array2<Real>,
    interference_quasiparticle: Array2<Real>,
    interference_satellite: Array2<Real>,
    intrinsic_satellite: Array2<Real>,
    clipped_extrinsic_satellite: Array2<Real>,
    weights: Array2<Real>,
    self_energy_real: Array1<Real>,
    energy_correction: Array1<Real>,
    width: Array1<Real>,
    renormalization_real: Array1<Real>,
    renormalization_imag: Array1<Real>,
}

fn so2conv_momentum_spectral_inputs() -> So2convMomentumSpectralInputs {
    So2convMomentumSpectralInputs {
        momentum_grid: array![0.50, 1.00, 2.00, 4.00],
        energy_grid: array![
            [0.11, 0.12, 0.13, 0.14],
            [0.21, 0.22, 0.23, 0.24],
            [0.31, 0.32, 0.33, 0.34],
            [0.41, 0.42, 0.43, 0.44],
        ],
        extrinsic_quasiparticle: array![
            [1.11, 1.12, 1.13, 1.14],
            [1.21, 1.22, 1.23, 1.24],
            [1.31, 1.32, 1.33, 1.34],
            [1.41, 1.42, 1.43, 1.44],
        ],
        extrinsic_satellite: array![
            [2.22, 2.24, 2.26, 2.28],
            [2.42, 2.44, 2.46, 2.48],
            [2.62, 2.64, 2.66, 2.68],
            [2.82, 2.84, 2.86, 2.88],
        ],
        interference_quasiparticle: array![
            [3.33, 3.36, 3.39, 3.42],
            [3.63, 3.66, 3.69, 3.72],
            [3.93, 3.96, 3.99, 4.02],
            [4.23, 4.26, 4.29, 4.32],
        ],
        interference_satellite: array![
            [0.444, 0.448, 0.452, 0.456],
            [0.484, 0.488, 0.492, 0.496],
            [0.524, 0.528, 0.532, 0.536],
            [0.564, 0.568, 0.572, 0.576],
        ],
        intrinsic_satellite: array![
            [0.555, 0.560, 0.565, 0.570],
            [0.605, 0.610, 0.615, 0.620],
            [0.655, 0.660, 0.665, 0.670],
            [0.705, 0.710, 0.715, 0.720],
        ],
        clipped_extrinsic_satellite: array![
            [0.666, 0.672, 0.678, 0.684],
            [0.726, 0.732, 0.738, 0.744],
            [0.786, 0.792, 0.798, 0.804],
            [0.846, 0.852, 0.858, 0.864],
        ],
        weights: array![
            [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
            [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
            [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
            [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
        ],
        self_energy_real: array![41.0, 42.0, 43.0, 44.0],
        energy_correction: array![51.0, 52.0, 53.0, 54.0],
        width: array![61.0, 62.0, 63.0, 64.0],
        renormalization_real: array![71.0, 72.0, 73.0, 74.0],
        renormalization_imag: array![81.0, 82.0, 83.0, 84.0],
    }
}

fn so2conv_momentum_spectral_input<'a>(
    inputs: &'a So2convMomentumSpectralInputs,
    photoelectron_momentum: Real,
) -> SfconvMomentumSpectralInterpolationInput<'a> {
    SfconvMomentumSpectralInterpolationInput {
        photoelectron_momentum,
        momentum_grid: inputs.momentum_grid.view(),
        energy_grid: inputs.energy_grid.view(),
        extrinsic_quasiparticle: inputs.extrinsic_quasiparticle.view(),
        extrinsic_satellite: inputs.extrinsic_satellite.view(),
        interference_quasiparticle: inputs.interference_quasiparticle.view(),
        interference_satellite: inputs.interference_satellite.view(),
        intrinsic_satellite: inputs.intrinsic_satellite.view(),
        clipped_extrinsic_satellite: inputs.clipped_extrinsic_satellite.view(),
        weights: inputs.weights.view(),
        self_energy_real: inputs.self_energy_real.view(),
        energy_correction: inputs.energy_correction.view(),
        width: inputs.width.view(),
        renormalization_real: inputs.renormalization_real.view(),
        renormalization_imag: inputs.renormalization_imag.view(),
    }
}

fn so2conv_photoelectron_momentum_inputs() -> (Array1<Real>, Array1<Real>) {
    let momentum = array![0.0, 0.35, -0.40, 0.82, 1.10, 1.45];
    let self_energy = array![0.090, 0.105, 0.120, 0.150, 0.190, 0.250];
    (momentum, self_energy)
}

fn so2conv_self_energy_material() -> SfconvSo2convMaterialParameters {
    SfconvSo2convMaterialParameters {
        core_hole_lifetime: 0.03,
        interstitial_potential: 0.0,
        chemical_potential_offset: 0.20,
        fermi_wave_number: 1.0,
        fermi_momentum: 1.0,
        fermi_energy: 0.50,
        electron_concentration: 0.08,
        plasma_frequency: 0.70,
        dispersion_parameter: 0.33,
        initial_photoelectron_energy: 0.50,
        initial_photoelectron_momentum: 1.0,
        accuracy: 1.0e-4,
    }
}

fn so2conv_xanes_preparation_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>, Array1<Real>) {
    let count = 22;
    let incident_energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        0.2 + 0.13 * (i - 1.0) + 0.002 * ((i as usize) % 3) as Real
    });
    let excitation_energy = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        -0.4 + 0.11 * (i - 1.0) + 0.001 * ((i as usize) % 4) as Real
    });
    let embedded_background = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        1.0 + 0.015 * (i - 1.0) + 0.0008 * ((i as usize) % 2) as Real
    });
    let absorption = Array1::from_shape_fn(count, |index| {
        let i = index as Real + 1.0;
        embedded_background[index] + 0.04 * (0.31 * i).sin() + 0.002 * (i - 1.0)
    });
    (
        incident_energy,
        excitation_energy,
        absorption,
        embedded_background,
    )
}

struct So2convFeffPathInterpolationInputs {
    source_momentum: Array1<Real>,
    path_momentum: Array1<Real>,
    central_phase: Array1<Real>,
    effective_amplitude: Array1<Real>,
    effective_phase: Array1<Real>,
    reduction_factor: Array1<Real>,
    mean_free_path: Array1<Real>,
    interpolated_central_phase: Array1<Real>,
    interpolated_effective_amplitude: Array1<Real>,
    interpolated_effective_phase: Array1<Real>,
    interpolated_reduction_factor: Array1<Real>,
    interpolated_mean_free_path: Array1<Real>,
}

fn so2conv_feff_path_interpolation_inputs() -> So2convFeffPathInterpolationInputs {
    So2convFeffPathInterpolationInputs {
        source_momentum: array![0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
        path_momentum: array![0.25, 0.75, 1.25, 1.75],
        central_phase: array![0.10, 0.20, 0.10, 0.30],
        effective_amplitude: array![1.00, 1.40, 1.10, 1.80],
        effective_phase: array![0.50, 0.70, 0.60, 1.00],
        reduction_factor: array![0.80, 0.90, 0.85, 0.95],
        mean_free_path: array![6.00, 7.00, 8.00, 9.00],
        interpolated_central_phase: array![0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0],
        interpolated_effective_amplitude: array![
            0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0
        ],
        interpolated_effective_phase: array![0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0],
        interpolated_reduction_factor: array![0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0],
        interpolated_mean_free_path: array![0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0],
    }
}

fn so2conv_path_average_inputs() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let source_momentum = array![0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25];
    let amplitude_reduction = array![0.82, 0.84, 0.88, 0.91, 0.89, 0.86, 0.83];
    let phase_shift = array![0.05, 0.08, 0.13, 0.17, 0.14, 0.09, 0.02];
    (source_momentum, amplitude_reduction, phase_shift)
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn assert_real_slice_close(actual: &Array1<Real>, expected: &[Real], tolerance: Real) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}

fn assert_so2conv_material_close(
    actual: SfconvSo2convMaterialParameters,
    expected: SfconvSo2convMaterialParameters,
    tolerance: Real,
) {
    assert_close(
        actual.core_hole_lifetime,
        expected.core_hole_lifetime,
        tolerance,
    );
    assert_close(
        actual.interstitial_potential,
        expected.interstitial_potential,
        tolerance,
    );
    assert_close(
        actual.chemical_potential_offset,
        expected.chemical_potential_offset,
        tolerance,
    );
    assert_close(
        actual.fermi_wave_number,
        expected.fermi_wave_number,
        tolerance,
    );
    assert_close(actual.fermi_momentum, expected.fermi_momentum, tolerance);
    assert_close(actual.fermi_energy, expected.fermi_energy, tolerance);
    assert_close(
        actual.electron_concentration,
        expected.electron_concentration,
        tolerance,
    );
    assert_close(
        actual.plasma_frequency,
        expected.plasma_frequency,
        tolerance,
    );
    assert_close(
        actual.dispersion_parameter,
        expected.dispersion_parameter,
        tolerance,
    );
    assert_close(
        actual.initial_photoelectron_energy,
        expected.initial_photoelectron_energy,
        tolerance,
    );
    assert_close(
        actual.initial_photoelectron_momentum,
        expected.initial_photoelectron_momentum,
        tolerance,
    );
    assert_close(actual.accuracy, expected.accuracy, tolerance);
}

fn assert_momentum_spectral_close(
    actual: &SfconvMomentumSpectralInterpolation,
    expected_energy: &[Real; 4],
    expected_rows: &[[Real; 4]; 8],
    expected_weights: &[Real; 8],
    expected_self_energy: &[Real; 5],
) {
    assert_real_slice_close(&actual.energy, expected_energy, 1.0e-15);
    for (row, expected) in expected_rows.iter().enumerate() {
        assert_real_slice_close(
            &actual.spectral_function.row(row).to_owned(),
            expected,
            1.0e-15,
        );
    }
    assert_real_slice_close(&actual.weights, expected_weights, 1.0e-15);
    assert_close(actual.self_energy_real, expected_self_energy[0], 1.0e-15);
    assert_close(actual.energy_correction, expected_self_energy[1], 1.0e-15);
    assert_close(actual.width, expected_self_energy[2], 1.0e-15);
    assert_close(
        actual.renormalization_real,
        expected_self_energy[3],
        1.0e-15,
    );
    assert_close(
        actual.renormalization_imag,
        expected_self_energy[4],
        1.0e-15,
    );
}

fn assert_pole_close(actual: SfconvPole, expected: SfconvPole) {
    assert_close(actual.energy, expected.energy, 1.0e-15);
    assert_close(actual.weight, expected.weight, 1.0e-15);
    assert_close(actual.broadening, expected.broadening, 1.0e-15);
}

fn assert_q_limits_close(actual: SfconvQLimits, expected: SfconvQLimits, tolerance: Real) {
    assert_eq!(actual.count, expected.count);
    assert_close(actual.q1, expected.q1, tolerance);
    assert_close(actual.q2, expected.q2, tolerance);
    assert_close(actual.q3, expected.q3, tolerance);
}

fn assert_integral_close(
    actual: SfconvAdaptiveIntegral,
    expected: SfconvAdaptiveIntegral,
    tolerance: Real,
) {
    assert_close(actual.value, expected.value, tolerance);
    assert_close(
        actual.estimated_error,
        expected.estimated_error,
        tolerance.max(1.0e-12),
    );
    assert_eq!(actual.evaluations, expected.evaluations);
    assert_eq!(actual.max_regions, expected.max_regions);
}

fn mksat_reference_context() -> SfconvSatelliteContext {
    SfconvSatelliteContext {
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        dispersion_parameter: 0.28,
        photoelectron_energy: 0.85,
        accuracy: 1.0e-4,
    }
}

fn mksat_reference_self_energy() -> SfconvSatelliteSelfEnergy {
    SfconvSatelliteSelfEnergy {
        on_shell_real: 0.12,
        width: 0.08,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
        off_shell_real: 0.03,
        off_shell_imag: 0.025,
    }
}

fn senergies_reference_context(include_below_fermi: bool) -> SfconvSelfEnergyContext {
    SfconvSelfEnergyContext {
        fermi_energy: 0.50,
        fermi_momentum: 1.00,
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        quasiparticle_energy: 0.91,
        photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
        accuracy: 1.0e-4,
        pole_broadening: 0.035,
        dispersion_parameter: 0.28,
        include_below_fermi,
    }
}
