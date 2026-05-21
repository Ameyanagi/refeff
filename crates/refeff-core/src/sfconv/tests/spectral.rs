use super::*;

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
