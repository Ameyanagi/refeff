use super::*;

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
