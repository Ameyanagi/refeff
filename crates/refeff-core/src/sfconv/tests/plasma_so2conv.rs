use super::*;

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
