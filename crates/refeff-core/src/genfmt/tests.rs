use super::{
    CurvedWavePolynomialInput, EnergyIndependentMatrixInput, GenfmtCentralPhaseShiftInput,
    GenfmtChiAmplitudePhaseInput, GenfmtCurvedWaveLegLimit, GenfmtCurvedWaveLegLimitsInput,
    GenfmtCurvedWavePathFactorInput, GenfmtCurvedWavePolynomialTablesInput,
    GenfmtDecomposedChiAmplitudePhase, GenfmtDecomposedChiAmplitudePhaseInput,
    GenfmtDriverSetupInput, GenfmtError, GenfmtFeffBinHeaderInput, GenfmtJasDriverOutputInput,
    GenfmtJasDriverSetupInput, GenfmtJasEffectiveInitialJInput,
    GenfmtJasEnergyGridBranchFromTransitionSetupInput, GenfmtJasLeftRightPathTraceInput,
    GenfmtJasPathEnergyBranchInput, GenfmtJasPathEnergyGridBranchInput,
    GenfmtJasPathEnergyGridFinalizationInput, GenfmtJasPathEnergyGridFromSetupInput,
    GenfmtJasPathEnergyGridInput, GenfmtJasPathEnergyPointInput,
    GenfmtJasPathEvaluationFromDriverSetupInput, GenfmtJasPathEvaluationFromSetupInput,
    GenfmtJasPathEvaluationInput, GenfmtJasPathFinalization, GenfmtJasPathFinalizationInput,
    GenfmtJasPathOutputsInput, GenfmtJasPathSequenceFromDriverSetupInput,
    GenfmtJasPathSequenceFromSetupInput, GenfmtJasPathSequenceInput, GenfmtJasPathSetupInput,
    GenfmtJasPathSignalInput, GenfmtJasPathSignals, GenfmtJasPathSignalsInput, GenfmtJasPathTrace,
    GenfmtJasPathTraceInput, GenfmtJasSphericalPathTraceInput, GenfmtJasSpinRadialFactorInput,
    GenfmtJasSpinSelectionInput, GenfmtJasTransitionCountInput, GenfmtJasTransitionMatrices,
    GenfmtJasTransitionMatricesInput, GenfmtJasTransitionSetupInput,
    GenfmtLegendreNormalizationInput, GenfmtMomentumGridInput, GenfmtNStarDriverInput,
    GenfmtNStarInput, GenfmtNStarPathInput, GenfmtNStarRowsInput, GenfmtOrdinaryDriverOutputInput,
    GenfmtOrdinaryPathEnergyGridFinalizationInput,
    GenfmtOrdinaryPathEnergyGridFromDriverSetupInput, GenfmtOrdinaryPathEnergyGridFromSetupInput,
    GenfmtOrdinaryPathEnergyGridInput, GenfmtOrdinaryPathEnergyPointInput,
    GenfmtOrdinaryPathEvaluationFromDriverSetupInput, GenfmtOrdinaryPathEvaluationFromSetupInput,
    GenfmtOrdinaryPathEvaluationInput, GenfmtOrdinaryPathFinalization,
    GenfmtOrdinaryPathFinalizationInput, GenfmtOrdinaryPathOutputsInput,
    GenfmtOrdinaryPathSequenceFromDriverSetupInput, GenfmtOrdinaryPathSequenceFromSetupInput,
    GenfmtOrdinaryPathSequenceInput, GenfmtOrdinaryPathSetupInput, GenfmtOrdinaryPathTraceInput,
    GenfmtOrdinarySpinMomentumGridInput, GenfmtOrdinaryTransitionMatricesInput,
    GenfmtPathGeometryInput, GenfmtPathImportance, GenfmtPathImportanceInput,
    GenfmtPathMatrixProductInput, GenfmtPathMatrixTraceInput, GenfmtPathOutputDecision,
    GenfmtPathOutputDecisionInput, GenfmtPathOutputSummary, GenfmtPathRetention,
    GenfmtPathRetentionInput, GenfmtPathRotationTables, GenfmtPathRotationTablesInput,
    GenfmtPathSetup, GenfmtPathSignalContributionInput, GenfmtPathSignals, GenfmtPathSignalsInput,
    GenfmtReferenceEnergyMode, GenfmtRetainedPathOutput, GenfmtRetainedPathOutputInput,
    GenfmtScatteringMatrixPlanInput, GenfmtScatteringMatrixRole, GenfmtScatteringMatrixTask,
    GenfmtScatteringPathProductInput, GenfmtSpinChannelCountInput, GenfmtSpinPhaseShiftInput,
    GenfmtSpinRadialFactorInput, GenfmtSpinReferenceEnergyInput, InitialStateRotation,
    InitialStateRotationInput, JasLeftRightAmplitudeInput, JasOneSidedTransitionInput,
    JasQAngleInput, JasScatteringAmplitudeInput, JasSpinTransitionInput, LambdaIndexInput,
    LambdaIndexSet, PathRotationAngles, PathRotationInput, PolarizedScatteringAmplitudeInput,
    ScatteringAmplitudeMatrixInput, TransitionRotationInput, XStarInput, curved_wave_polynomials,
    energy_independent_transition_matrix, genfmt_central_phase_shifts, genfmt_chi_amplitude_phase,
    genfmt_curved_wave_leg_limits, genfmt_curved_wave_path_factor,
    genfmt_curved_wave_polynomial_tables, genfmt_decomposed_chi_amplitude_phase,
    genfmt_driver_setup, genfmt_feff_bin_header, genfmt_jas_driver_output, genfmt_jas_driver_setup,
    genfmt_jas_effective_initial_j, genfmt_jas_energy_grid_branch_from_transition_setup,
    genfmt_jas_left_right_path_trace, genfmt_jas_path_energy_grid,
    genfmt_jas_path_energy_grid_finalization, genfmt_jas_path_energy_grid_from_setup,
    genfmt_jas_path_energy_point, genfmt_jas_path_evaluation,
    genfmt_jas_path_evaluation_from_driver_setup, genfmt_jas_path_evaluation_from_setup,
    genfmt_jas_path_finalization, genfmt_jas_path_outputs, genfmt_jas_path_sequence,
    genfmt_jas_path_sequence_from_driver_setup, genfmt_jas_path_sequence_from_setup,
    genfmt_jas_path_setup, genfmt_jas_path_signal, genfmt_jas_path_signals, genfmt_jas_path_trace,
    genfmt_jas_spherical_path_trace, genfmt_jas_spin_radial_factors, genfmt_jas_spin_selection,
    genfmt_jas_transition_count, genfmt_jas_transition_matrices, genfmt_jas_transition_setup,
    genfmt_legendre_normalization_table, genfmt_momentum_grid, genfmt_nstar_row, genfmt_nstar_rows,
    genfmt_ordinary_driver_output, genfmt_ordinary_path_energy_grid,
    genfmt_ordinary_path_energy_grid_finalization,
    genfmt_ordinary_path_energy_grid_from_driver_setup,
    genfmt_ordinary_path_energy_grid_from_setup, genfmt_ordinary_path_energy_point,
    genfmt_ordinary_path_evaluation, genfmt_ordinary_path_evaluation_from_driver_setup,
    genfmt_ordinary_path_evaluation_from_setup, genfmt_ordinary_path_finalization,
    genfmt_ordinary_path_outputs, genfmt_ordinary_path_sequence,
    genfmt_ordinary_path_sequence_from_driver_setup, genfmt_ordinary_path_sequence_from_setup,
    genfmt_ordinary_path_setup, genfmt_ordinary_path_trace, genfmt_ordinary_spin_momentum_grid,
    genfmt_ordinary_transition_matrices, genfmt_path_geometry, genfmt_path_importance,
    genfmt_path_matrix_product, genfmt_path_matrix_trace, genfmt_path_output_decision,
    genfmt_path_retention, genfmt_path_rotation_tables, genfmt_path_signal_contribution,
    genfmt_path_signals, genfmt_retained_path_output, genfmt_scattering_matrix_plan,
    genfmt_scattering_path_product, genfmt_spin_channel_count, genfmt_spin_phase_shifts,
    genfmt_spin_radial_factors, genfmt_spin_reference_energies, initial_state_rotation,
    jas_left_right_amplitude_matrices, jas_one_sided_transition_matrices, jas_q_angles,
    jas_scattering_amplitude_matrices, jas_spin_transition_matrix, lambda_indices,
    path_rotation_angles, polarized_scattering_amplitude_matrix, scattering_amplitude_matrix,
    xstar,
};
use crate::{Complex, Real, legendre_normalization_table};
use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, Axis, ShapeBuilder, Slice, arr2};

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
fn genfmt_path_rotation_tables_matches_driver_rot3i_loop() -> Result<(), GenfmtError> {
    let beta_angles = Array1::from_vec(vec![0.25, 1.1, 2.4, 0.7]);
    let tables = genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
        beta_angles: beta_angles.view(),
        leg_count: 3,
        lmaxp1: 4,
        mmaxp1: 3,
        polarized_extra: Some((2, 2)),
    })?;

    assert_eq!(tables.len(), 4);
    assert_eq!(tables.real_leg_count, 3);
    assert_eq!(tables.rotation_magnetic_offset, 2);
    assert_eq!(tables.rotations.shape(), &[4, 4, 5, 5]);
    assert_eq!(tables.rotations.strides(), &[1, 4, 16, 80]);

    let real_leg_rotations = tables.real_leg_rotations()?;
    assert_eq!(real_leg_rotations.shape(), &[3, 4, 5, 5]);
    for leg_index in 0..3 {
        let expected = initial_state_rotation(InitialStateRotationInput {
            lmaxp1: 4,
            mmaxp1: 3,
            beta_angle: beta_angles[leg_index],
        })?;
        for l in 0..expected.matrix.shape()[0] {
            for row in 0..expected.matrix.shape()[1] {
                for column in 0..expected.matrix.shape()[2] {
                    assert_close(
                        real_leg_rotations[(leg_index, l, row, column)],
                        expected.matrix[(l, row, column)],
                    );
                }
            }
        }
    }

    let expected_extra = initial_state_rotation(InitialStateRotationInput {
        lmaxp1: 2,
        mmaxp1: 2,
        beta_angle: beta_angles[3],
    })?;
    let extra = tables
        .polarized_extra_rotation()?
        .expect("polarized extra rotation");
    assert_eq!(extra.shape(), &[4, 5, 5]);
    for l in 0..expected_extra.matrix.shape()[0] {
        for m1 in -1..=1 {
            for m2 in -1..=1 {
                assert_close(
                    padded_rotation_value(extra, 2, l + 1, m1, m2),
                    rotation_value(&expected_extra, l + 1, m1, m2),
                );
            }
        }
    }
    assert_close(padded_rotation_value(extra, 2, 3, 0, 0), 0.0);
    assert_close(padded_rotation_value(extra, 2, 1, -2, -2), 0.0);
    Ok(())
}

#[test]
fn genfmt_path_rotation_tables_selects_mmtr_rotations() -> Result<(), GenfmtError> {
    let beta_angles = Array1::from_vec(vec![0.25, 1.1, 2.4, 0.7]);
    let eta_values = Array1::from_vec(vec![0.3, 1.0, 1.5, 2.0, 2.7]);
    let tables = genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
        beta_angles: beta_angles.view(),
        leg_count: 3,
        lmaxp1: 4,
        mmaxp1: 3,
        polarized_extra: Some((2, 2)),
    })?;

    match tables.transition_rotations(eta_values.view(), false)? {
        TransitionRotationInput::Unpolarized { combined_rotation } => {
            assert_eq!(combined_rotation.shape(), &[4, 5, 5]);
            assert_close(
                padded_rotation_value(combined_rotation, 2, 2, 0, 1),
                tables.rotations[(2, 1, 2, 3)],
            );
        }
        TransitionRotationInput::Polarized { .. } => panic!("expected unpolarized rotations"),
    }

    match tables.transition_rotations(eta_values.view(), true)? {
        TransitionRotationInput::Polarized {
            first_rotation,
            last_rotation,
            first_eta,
            last_eta,
        } => {
            assert_close(first_eta, eta_values[0]);
            assert_close(last_eta, eta_values[4]);
            assert_close(
                padded_rotation_value(first_rotation, 2, 2, -1, 1),
                tables.rotations[(3, 1, 1, 3)],
            );
            assert_close(
                padded_rotation_value(last_rotation, 2, 2, -1, 1),
                tables.rotations[(2, 1, 1, 3)],
            );
        }
        TransitionRotationInput::Unpolarized { .. } => panic!("expected polarized rotations"),
    }
    Ok(())
}

#[test]
fn genfmt_path_rotation_tables_rejects_invalid_inputs() {
    let beta_angles = Array1::from_vec(vec![0.1, 0.2]);

    assert_eq!(
        genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
            beta_angles: beta_angles.view(),
            leg_count: 0,
            lmaxp1: 2,
            mmaxp1: 2,
            polarized_extra: None,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
            beta_angles: beta_angles.view(),
            leg_count: 2,
            lmaxp1: 2,
            mmaxp1: 2,
            polarized_extra: Some((1, 1)),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "beta_angles",
            axis: "leg",
            length: 2,
            required: 3,
        })
    );

    let beta_angles = Array1::from_vec(vec![0.1, f64::NAN]);
    assert_eq!(
        genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
            beta_angles: beta_angles.view(),
            leg_count: 2,
            lmaxp1: 2,
            mmaxp1: 2,
            polarized_extra: None,
        }),
        Err(GenfmtError::NonFiniteRotationAngle)
    );

    let beta_angles = Array1::from_vec(vec![0.1, 0.2]);
    let tables = genfmt_path_rotation_tables(GenfmtPathRotationTablesInput {
        beta_angles: beta_angles.view(),
        leg_count: 2,
        lmaxp1: 2,
        mmaxp1: 2,
        polarized_extra: None,
    })
    .expect("rotation tables");
    let eta_values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    assert!(matches!(
        tables.transition_rotations(eta_values.view(), true),
        Err(GenfmtError::TableAxisTooShort {
            table: "rotations",
            axis: "leg",
            length: 2,
            required: 3,
        })
    ));
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
fn genfmt_ordinary_path_setup_composes_rdpath_setlam_rot3i_reference() -> Result<(), GenfmtError> {
    let positions = arr2(&[
        [1.2, -0.4, 0.7],
        [-0.3, 1.1, 1.5],
        [0.5, 0.2, -0.6],
        [0.0, 0.0, 0.0],
    ]);
    let setup = genfmt_ordinary_path_setup(GenfmtOrdinaryPathSetupInput {
        positions: positions.view(),
        polarized: true,
        calculation: 10,
        initial_l: 1,
        lambda_capacity: 80,
        max_m: 10,
        max_n: 10,
        lmaxp1: 4,
    })?;

    let angles = path_rotation_angles(PathRotationInput {
        positions: positions.view(),
        polarized: true,
    })?;
    assert_array_close(
        &setup.angles.beta_angles,
        angles.beta_angles.as_slice().unwrap(),
    );
    assert_array_close(
        &setup.angles.eta_values,
        angles.eta_values.as_slice().unwrap(),
    );
    assert_array_close(
        &setup.angles.leg_lengths,
        angles.leg_lengths.as_slice().unwrap(),
    );
    assert_close(
        setup.effective_half_path_length,
        angles.leg_lengths.iter().sum::<Real>() / 2.0,
    );

    let beta = angles.beta_angles.as_slice().unwrap();
    let lambda = lambda_indices(LambdaIndexInput {
        calculation: 10,
        energy_index: 1,
        scattering_count: 3,
        initial_l: 1,
        beta_angles: &beta[..4],
        lambda_capacity: 80,
        max_m: 10,
        max_n: 10,
    })?;
    assert_eq!(setup.lambda, lambda);
    assert_eq!(setup.lambda.requested_n_max, 1);
    assert_eq!(setup.lambda.requested_m_max, 3);

    assert_eq!(setup.rotations.real_leg_count, 4);
    assert_eq!(setup.lambda.max_m_plus_one, 4);
    assert_eq!(setup.rotations.rotation_magnetic_offset, 3);
    assert_eq!(setup.rotations.rotations.shape(), &[5, 4, 7, 7]);
    let ordinary_extra = setup
        .rotations
        .polarized_extra_rotation()?
        .expect("ordinary polarized pseudo-leg");
    assert_close(
        padded_rotation_value(
            ordinary_extra,
            setup.rotations.rotation_magnetic_offset,
            4,
            0,
            0,
        ),
        0.0,
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_setup_uses_full_polarized_rot3i_dimensions() -> Result<(), GenfmtError> {
    let positions = arr2(&[
        [1.2, -0.4, 0.7],
        [-0.3, 1.1, 1.5],
        [0.5, 0.2, -0.6],
        [0.0, 0.0, 0.0],
    ]);
    let setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: true,
        calculation: 10,
        initial_l: 1,
        lambda_capacity: 80,
        max_m: 10,
        max_n: 10,
        lmaxp1: 4,
    })?;

    let jas_extra = setup
        .rotations
        .polarized_extra_rotation()?
        .expect("JAS polarized pseudo-leg");
    let expected_extra = initial_state_rotation(InitialStateRotationInput {
        lmaxp1: 4,
        mmaxp1: setup.lambda.max_m_plus_one,
        beta_angle: setup.angles.beta_angles[4],
    })?;

    assert_close(
        padded_rotation_value(jas_extra, setup.rotations.rotation_magnetic_offset, 4, 0, 0),
        rotation_value(&expected_extra, 4, 0, 0),
    );
    assert_ne!(
        padded_rotation_value(jas_extra, setup.rotations.rotation_magnetic_offset, 4, 0, 0,),
        0.0
    );
    Ok(())
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
fn jas_spin_transition_matrix_matches_feff_mmtrjas0_reference() -> Result<(), GenfmtError> {
    let data = mmtrjas0_reference_data();
    let transition = jas_spin_transition_matrix(data.input())?;

    assert_eq!(transition.generated_final_j2, vec![1, 1, 3, 3]);
    assert_eq!(transition.matrix.shape(), &[4, 2, 7, 7, 4]);
    assert_eq!(transition.matrix.strides(), &[1, 4, 8, 56, 392]);
    assert_complex_close(
        complex5_sum(&transition.matrix),
        Complex::new(10.035_783_076_654_589, -0.621_416_020_216_338_2),
    );
    assert_complex_close(transition.matrix[(0, 0, 3, 3, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(
        transition.matrix[(2, 1, 4, 2, 1)],
        Complex::new(-0.059_197_030_354_702_53, 0.048_412_270_136_146_775),
    );
    assert_complex_close(
        transition.matrix[(3, 0, 1, 5, 3)],
        Complex::new(0.007_457_080_409_986_383, -0.086_088_690_538_654_55),
    );
    assert_complex_close(transition.matrix[(1, 1, 6, 0, 2)], Complex::new(0.0, 0.0));
    assert_complex_close(
        transition.matrix[(1, 0, 2, 4, 1)],
        Complex::new(-0.048_301_690_482_561_03, -0.044_803_840_424_248_634),
    );
    assert_complex_close(
        transition.matrix[(3, 1, 5, 1, 3)],
        Complex::new(-0.020_429_273_563_469_852, 0.038_351_352_489_888_524),
    );
    Ok(())
}

#[test]
fn jas_spin_transition_matrix_rejects_invalid_inputs() {
    let data = mmtrjas0_reference_data();
    assert_eq!(
        jas_spin_transition_matrix(JasSpinTransitionInput {
            initial_kappa: 0,
            ..data.input()
        }),
        Err(GenfmtError::InvalidInitialKappa { kappa: 0 })
    );
    assert!(matches!(
        jas_spin_transition_matrix(JasSpinTransitionInput {
            spin_channels: 0,
            ..data.input()
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channels",
            value: 0,
        })
    ));
    let too_many_transitions = Array1::from_vec(vec![0, 1, 1, 2, 2, 3]);
    assert!(matches!(
        jas_spin_transition_matrix(JasSpinTransitionInput {
            transition_angular_momenta: too_many_transitions.view(),
            ..data.input()
        }),
        Err(GenfmtError::InsufficientGeneratedTransitions {
            required: 6,
            generated: 5,
        })
    ));

    let mut bad_first_rotation = data.first_rotation.clone();
    bad_first_rotation[(0, 3, 3)] = f64::NAN;
    assert!(matches!(
        jas_spin_transition_matrix(JasSpinTransitionInput {
            first_rotation: bad_first_rotation.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableScalar {
            table: "rotation",
            row: 0,
            column: 3,
            ..
        })
    ));

    let short_last_rotation = Array3::zeros((3, 7, 6).f());
    assert!(matches!(
        jas_spin_transition_matrix(JasSpinTransitionInput {
            last_rotation: short_last_rotation.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "last_rotation",
            axis: "m2",
            ..
        })
    ));
}

#[test]
fn genfmt_jas_transition_matrices_selects_genfmtjas_branch() -> Result<(), GenfmtError> {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();

    let left_right = genfmt_jas_transition_matrices(GenfmtJasTransitionMatricesInput {
        ellipticity: 0.0,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let expected_left_right = jas_one_sided_transition_matrices(left_right_data.input())?;
    match left_right {
        GenfmtJasTransitionMatrices::LeftRight(matrices) => {
            assert_eq!(
                matrices.generated_final_j2,
                expected_left_right.generated_final_j2
            );
            assert_complex_close(
                complex4_sum(&matrices.left_matrix),
                complex4_sum(&expected_left_right.left_matrix),
            );
            assert_complex_close(
                matrices.right_matrix[(2, 2, 1, 2)],
                expected_left_right.right_matrix[(2, 2, 1, 2)],
            );
        }
        GenfmtJasTransitionMatrices::Spherical(_) => panic!("expected left/right JAS branch"),
    }

    let spherical = genfmt_jas_transition_matrices(GenfmtJasTransitionMatricesInput {
        ellipticity: -1.0,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let expected_spherical = jas_spin_transition_matrix(spherical_data.input())?;
    match spherical {
        GenfmtJasTransitionMatrices::Spherical(matrix) => {
            assert_eq!(
                matrix.generated_final_j2,
                expected_spherical.generated_final_j2
            );
            assert_complex_close(
                complex5_sum(&matrix.matrix),
                complex5_sum(&expected_spherical.matrix),
            );
            assert_complex_close(
                matrix.matrix[(2, 1, 4, 2, 1)],
                expected_spherical.matrix[(2, 1, 4, 2, 1)],
            );
        }
        GenfmtJasTransitionMatrices::LeftRight(_) => panic!("expected spherical JAS branch"),
    }

    Ok(())
}

#[test]
fn genfmt_jas_transition_matrices_rejects_nonfinite_ellipticity() {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    assert!(matches!(
        genfmt_jas_transition_matrices(GenfmtJasTransitionMatricesInput {
            ellipticity: f64::NAN,
            left_right: left_right_data.input(),
            spherical: spherical_data.input(),
        }),
        Err(GenfmtError::NonFiniteScalar {
            field: "ellipticity",
            ..
        })
    ));
}

#[test]
fn genfmt_jas_transition_setup_composes_left_right_branch() -> Result<(), GenfmtError> {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();

    let setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let expected_left_right = jas_one_sided_transition_matrices(left_right_data.input())?;

    assert_eq!(setup.effective_initial_j.initial_j2, 3);
    assert!(!setup.effective_initial_j.promoted_to_final_j2_max);
    assert_eq!(setup.transition_count.transition_count, transition_count);
    match setup.matrices {
        GenfmtJasTransitionMatrices::LeftRight(matrices) => {
            assert_eq!(
                matrices.generated_final_j2,
                expected_left_right.generated_final_j2
            );
            assert_complex_close(
                complex4_sum(&matrices.left_matrix),
                complex4_sum(&expected_left_right.left_matrix),
            );
            assert_complex_close(
                matrices.right_matrix[(2, 2, 1, 2)],
                expected_left_right.right_matrix[(2, 2, 1, 2)],
            );
        }
        GenfmtJasTransitionMatrices::Spherical(_) => panic!("expected left/right JAS branch"),
    }
    Ok(())
}

#[test]
fn genfmt_jas_transition_setup_promotes_spherical_initial_j() -> Result<(), GenfmtError> {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = spherical_data.transition_angular_momenta.len();

    let setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: -1.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let mut expected_input = spherical_data.input();
    expected_input.initial_j2 = expected_input.final_j2_max;
    let expected_spherical = jas_spin_transition_matrix(expected_input)?;

    assert_eq!(
        setup.effective_initial_j.initial_j2,
        expected_input.final_j2_max
    );
    assert!(setup.effective_initial_j.promoted_to_final_j2_max);
    assert_eq!(setup.transition_count.transition_count, transition_count);
    match setup.matrices {
        GenfmtJasTransitionMatrices::Spherical(matrix) => {
            assert_eq!(
                matrix.generated_final_j2,
                expected_spherical.generated_final_j2
            );
            assert_eq!(
                matrix.matrix.shape()[0],
                (expected_input.final_j2_max + 1) as usize
            );
            assert_complex_close(
                complex5_sum(&matrix.matrix),
                complex5_sum(&expected_spherical.matrix),
            );
        }
        GenfmtJasTransitionMatrices::LeftRight(_) => panic!("expected spherical JAS branch"),
    }
    Ok(())
}

#[test]
fn genfmt_jas_transition_setup_rejects_transition_count_mismatch() {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    assert_eq!(
        genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
            ellipticity: 0.0,
            phase_transition_count: 2,
            requested_transition_count: 3,
            left_right: left_right_data.input(),
            spherical: spherical_data.input(),
        }),
        Err(GenfmtError::MismatchedJasTransitionCount {
            phase_transition_count: 2,
            requested_transition_count: 3,
        })
    );
}

#[test]
fn genfmt_jas_energy_grid_branch_from_transition_setup_builds_left_right_branch()
-> Result<(), GenfmtError> {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let mut transition_angular_momenta = left_right_data.transition_angular_momenta.to_vec();
    transition_angular_momenta.push(99);
    let transition_angular_momenta = Array1::from_vec(transition_angular_momenta);
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count + 1);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);

    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &setup,
            transition_angular_momenta: transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: Some(2),
        },
    )?;

    match (&setup.matrices, branch) {
        (
            GenfmtJasTransitionMatrices::LeftRight(setup_matrices),
            GenfmtJasPathEnergyGridBranchInput::LeftRight {
                transition_angular_momenta,
                radial_factors,
                q_weights,
                left_transition_matrix,
                right_transition_matrix,
                initial_j2,
                transition_magnetic_offset,
                max_angular_momentum,
                decomposition_l_max,
            },
        ) => {
            assert_eq!(
                transition_angular_momenta.to_vec(),
                left_right_data.transition_angular_momenta.to_vec()
            );
            assert_eq!(radial_factors.shape(), &[2, 2, transition_count]);
            assert_eq!(
                q_weights.to_vec(),
                vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]
            );
            assert_eq!(left_transition_matrix.shape()[2], 2);
            assert_eq!(left_transition_matrix.shape()[3], transition_count);
            assert_eq!(right_transition_matrix.shape()[2], 2);
            assert_eq!(right_transition_matrix.shape()[3], transition_count);
            assert_eq!(initial_j2, setup.effective_initial_j.initial_j2);
            assert_eq!(
                transition_magnetic_offset,
                left_right_data.input().rotation_magnetic_offset
            );
            assert_eq!(
                max_angular_momentum,
                left_right_data.input().max_angular_momentum
            );
            assert_eq!(decomposition_l_max, Some(2));
            let expected_left = setup_matrices
                .left_matrix
                .view()
                .slice_axis_move(Axis(2), Slice::from(..2))
                .slice_axis_move(Axis(3), Slice::from(..transition_count))
                .to_owned();
            assert_complex_close(
                complex4_sum(&left_transition_matrix.to_owned()),
                complex4_sum(&expected_left),
            );
        }
        _ => panic!("expected left/right JAS energy-grid branch"),
    }
    Ok(())
}

#[test]
fn genfmt_jas_energy_grid_branch_from_transition_setup_builds_spherical_branch()
-> Result<(), GenfmtError> {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = spherical_data.transition_angular_momenta.len();
    let setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: -1.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 3, transition_count);
    let q_weights = Array1::from_vec(vec![
        Complex::new(0.2, 0.0),
        Complex::new(0.3, 0.0),
        Complex::new(0.5, 0.0),
    ]);

    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &setup,
            transition_angular_momenta: spherical_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: spherical_data.input().rotation_magnetic_offset,
            max_angular_momentum: spherical_data.input().max_angular_momentum,
            decomposition_l_max: Some(1),
        },
    )?;

    match (&setup.matrices, branch) {
        (
            GenfmtJasTransitionMatrices::Spherical(setup_matrix),
            GenfmtJasPathEnergyGridBranchInput::Spherical {
                transition_angular_momenta,
                radial_factors,
                q_weights,
                transition_matrix,
                initial_j2,
                transition_magnetic_offset,
                max_angular_momentum,
                decomposition_l_max,
            },
        ) => {
            assert_eq!(
                transition_angular_momenta.to_vec(),
                spherical_data.transition_angular_momenta.to_vec()
            );
            assert_eq!(radial_factors.shape(), &[2, 3, transition_count]);
            assert_eq!(q_weights.len(), 3);
            assert_eq!(
                transition_matrix.shape()[0],
                (spherical_data.input().final_j2_max + 1) as usize
            );
            assert_eq!(transition_matrix.shape()[4], transition_count);
            assert_eq!(initial_j2, spherical_data.input().final_j2_max);
            assert_eq!(
                transition_magnetic_offset,
                spherical_data.input().rotation_magnetic_offset
            );
            assert_eq!(
                max_angular_momentum,
                spherical_data.input().max_angular_momentum
            );
            assert_eq!(decomposition_l_max, Some(1));
            assert_complex_close(
                complex5_sum(&transition_matrix.to_owned()),
                complex5_sum(&setup_matrix.matrix),
            );
        }
        _ => panic!("expected spherical JAS energy-grid branch"),
    }
    Ok(())
}

#[test]
fn genfmt_jas_energy_grid_branch_from_transition_setup_rejects_short_radial_transitions() {
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })
    .expect("transition setup");
    let radial_factors = jas_setup_radial_grid(1, 3, transition_count - 1);
    let q_weights = Array1::from_vec(vec![
        Complex::new(0.2, 0.0),
        Complex::new(0.3, 0.0),
        Complex::new(0.5, 0.0),
    ]);

    assert!(matches!(
        genfmt_jas_energy_grid_branch_from_transition_setup(
            GenfmtJasEnergyGridBranchFromTransitionSetupInput {
                transition_setup: &setup,
                transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
                radial_factors: radial_factors.view(),
                q_weights: q_weights.view(),
                transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
                max_angular_momentum: left_right_data.input().max_angular_momentum,
                decomposition_l_max: None,
            },
        ),
        Err(GenfmtError::TableAxisTooShort {
            table: "radial_factors",
            axis: "transition",
            length,
            required,
        }) if length == transition_count - 1 && required == transition_count
    ));
}

#[test]
fn genfmt_jas_path_energy_grid_from_setup_threads_driver_and_path_setup() -> Result<(), GenfmtError>
{
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: false,
        calculation: 0,
        initial_l: 1,
        lambda_capacity: 24,
        max_m: 6,
        max_n: 6,
        lmaxp1: 3,
    })?;
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let transition_setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);
    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &transition_setup,
            transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: None,
        },
    )?;
    let path_potential_indices = Array1::from_vec(vec![1, 0]);
    let mut xnlm = Array2::zeros((5, 5).f());
    for row in 0..xnlm.shape()[0] {
        for column in 0..xnlm.shape()[1] {
            xnlm[(row, column)] = 0.8 + 0.07 * (row as Real) + 0.03 * (column as Real);
        }
    }

    let energy_input =
        genfmt_jas_path_energy_grid_from_setup(GenfmtJasPathEnergyGridFromSetupInput {
            driver_setup: &driver_setup,
            path_setup: &path_setup,
            path_potential_indices: path_potential_indices.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            xnlm: xnlm.view(),
            branch,
            momentum_zero_epsilon: 1.0e-16,
        })?;

    assert_eq!(
        energy_input.m_indices.to_vec(),
        path_setup.lambda.m_indices.to_vec()
    );
    assert_eq!(
        energy_input.n_indices.to_vec(),
        path_setup.lambda.n_indices.to_vec()
    );
    assert_eq!(
        energy_input.full_lambda_count,
        path_setup.lambda.m_indices.len()
    );
    assert_eq!(
        energy_input.initial_lambda_count,
        path_setup.lambda.initial_l_prefix_len
    );
    assert_eq!(energy_input.path_potential_indices.to_vec(), vec![1, 0]);
    assert_complex_array3_close(
        &energy_input.phase_shifts.to_owned(),
        &driver_setup.phase_shifts.phase_shifts,
    );
    assert_eq!(
        energy_input.complex_momenta.to_vec(),
        driver_setup.momentum_grid.complex_momenta.to_vec()
    );
    assert_array_close(
        &energy_input.wave_numbers.to_owned(),
        driver_setup.momentum_grid.wave_numbers.as_slice().unwrap(),
    );
    assert_array_close(
        &energy_input.leg_lengths.to_owned(),
        path_setup.angles.leg_lengths.as_slice().unwrap(),
    );
    assert_eq!(
        energy_input.max_m_plus_one,
        path_setup.lambda.max_m_plus_one
    );
    assert_eq!(energy_input.max_n, path_setup.lambda.max_n);
    assert_eq!(
        energy_input.rotation_magnetic_offset,
        path_setup.rotations.rotation_magnetic_offset
    );
    match energy_input.branch {
        GenfmtJasPathEnergyGridBranchInput::LeftRight { initial_j2, .. } => {
            assert_eq!(initial_j2, transition_setup.effective_initial_j.initial_j2);
        }
        GenfmtJasPathEnergyGridBranchInput::Spherical { .. } => {
            panic!("expected left/right branch")
        }
    }

    let grid = genfmt_jas_path_energy_grid(energy_input)?;
    assert_eq!(grid.active.len(), energies.len());

    let short_path_potential_indices = Array1::from_vec(vec![1]);
    assert!(matches!(
        genfmt_jas_path_energy_grid_from_setup(GenfmtJasPathEnergyGridFromSetupInput {
            driver_setup: &driver_setup,
            path_setup: &path_setup,
            path_potential_indices: short_path_potential_indices.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            xnlm: xnlm.view(),
            branch,
            momentum_zero_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_potential_indices",
            axis: "leg",
            length: 1,
            required: 2,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_jas_path_evaluation_from_setup_matches_manual_driver_wiring() -> Result<(), GenfmtError> {
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: false,
        calculation: 0,
        initial_l: 1,
        lambda_capacity: 24,
        max_m: 6,
        max_n: 6,
        lmaxp1: 3,
    })?;
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let transition_setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);
    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &transition_setup,
            transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: None,
        },
    )?;
    let path_potential_indices = Array1::from_vec(vec![1, 0]);
    let mut xnlm = Array2::zeros((5, 5).f());
    for row in 0..xnlm.shape()[0] {
        for column in 0..xnlm.shape()[1] {
            xnlm[(row, column)] = 0.8 + 0.07 * (row as Real) + 0.03 * (column as Real);
        }
    }
    let energy_setup = GenfmtJasPathEnergyGridFromSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        xnlm: xnlm.view(),
        branch,
        momentum_zero_epsilon: 1.0e-16,
    };

    let evaluation =
        genfmt_jas_path_evaluation_from_setup(GenfmtJasPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: driver_setup
                .momentum_grid
                .complex_momentum_magnitudes
                .view(),
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: -1.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        })?;
    let expected_energy_input = genfmt_jas_path_energy_grid_from_setup(energy_setup)?;
    let expected = genfmt_jas_path_evaluation(GenfmtJasPathEvaluationInput {
        energy_grid: expected_energy_input,
        path_index: 23,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: driver_setup
            .momentum_grid
            .complex_momentum_magnitudes
            .view(),
        edge_start_index: 0,
        active_energy_count: energies.len(),
        degeneracy: 1.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;
    assert_eq!(evaluation, expected);

    let short_positions = arr2(&[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        genfmt_jas_path_evaluation_from_setup(GenfmtJasPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: driver_setup
                .momentum_grid
                .complex_momentum_magnitudes
                .view(),
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: -1.0,
            positions: short_positions.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "positions",
            axis: "leg",
            length: 1,
            required: 2,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_jas_path_evaluation_from_driver_setup_matches_manual_driver_wiring()
-> Result<(), GenfmtError> {
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: false,
        calculation: 0,
        initial_l: 1,
        lambda_capacity: 24,
        max_m: 6,
        max_n: 6,
        lmaxp1: 3,
    })?;
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let transition_setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);
    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &transition_setup,
            transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: None,
        },
    )?;
    let path_potential_indices = Array1::from_vec(vec![1, 0]);
    let mut xnlm = Array2::zeros((5, 5).f());
    for row in 0..xnlm.shape()[0] {
        for column in 0..xnlm.shape()[1] {
            xnlm[(row, column)] = 0.8 + 0.07 * (row as Real) + 0.03 * (column as Real);
        }
    }
    let energy_setup = GenfmtJasPathEnergyGridFromSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        xnlm: xnlm.view(),
        branch,
        momentum_zero_epsilon: 1.0e-16,
    };

    let evaluation = genfmt_jas_path_evaluation_from_driver_setup(
        GenfmtJasPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: -1.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    )?;
    let expected_energy_input = genfmt_jas_path_energy_grid_from_setup(energy_setup)?;
    let expected = genfmt_jas_path_evaluation(GenfmtJasPathEvaluationInput {
        energy_grid: expected_energy_input,
        path_index: 23,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: driver_setup
            .momentum_grid
            .complex_momentum_magnitudes
            .view(),
        edge_start_index: 0,
        active_energy_count: energies.len(),
        degeneracy: 1.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;
    assert_eq!(evaluation, expected);

    let short_positions = arr2(&[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        genfmt_jas_path_evaluation_from_driver_setup(GenfmtJasPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: -1.0,
            positions: short_positions.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "positions",
            axis: "leg",
            length: 1,
            required: 2,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_jas_path_sequence_from_setup_threads_normalization_reference() -> Result<(), GenfmtError>
{
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: false,
        calculation: 0,
        initial_l: 1,
        lambda_capacity: 24,
        max_m: 6,
        max_n: 6,
        lmaxp1: 3,
    })?;
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let transition_setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);
    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &transition_setup,
            transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: None,
        },
    )?;
    let path_potential_indices = Array1::from_vec(vec![1, 0]);
    let mut xnlm = Array2::zeros((5, 5).f());
    for row in 0..xnlm.shape()[0] {
        for column in 0..xnlm.shape()[1] {
            xnlm[(row, column)] = 0.8 + 0.07 * (row as Real) + 0.03 * (column as Real);
        }
    }
    let energy_setup = GenfmtJasPathEnergyGridFromSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        xnlm: xnlm.view(),
        branch,
        momentum_zero_epsilon: 1.0e-16,
    };
    let path_inputs = [
        GenfmtJasPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: driver_setup
                .momentum_grid
                .complex_momentum_magnitudes
                .view(),
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtJasPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 24,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: driver_setup
                .momentum_grid
                .complex_momentum_magnitudes
                .view(),
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 0.90,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence = genfmt_jas_path_sequence_from_setup(GenfmtJasPathSequenceFromSetupInput {
        path_inputs: &path_inputs,
        initial_normalization: -1.0,
    })?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation_from_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &expected_finalizations,
    })?;

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    Ok(())
}

#[test]
fn genfmt_jas_path_sequence_from_driver_setup_threads_normalization_reference()
-> Result<(), GenfmtError> {
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_setup = genfmt_jas_path_setup(GenfmtJasPathSetupInput {
        positions: positions.view(),
        polarized: false,
        calculation: 0,
        initial_l: 1,
        lambda_capacity: 24,
        max_m: 6,
        max_n: 6,
        lmaxp1: 3,
    })?;
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let left_right_data = mmtrjas_reference_data();
    let spherical_data = mmtrjas0_reference_data();
    let transition_count = left_right_data.transition_angular_momenta.len();
    let transition_setup = genfmt_jas_transition_setup(GenfmtJasTransitionSetupInput {
        ellipticity: 0.0,
        phase_transition_count: transition_count,
        requested_transition_count: transition_count,
        left_right: left_right_data.input(),
        spherical: spherical_data.input(),
    })?;
    let radial_factors = jas_setup_radial_grid(2, 2, transition_count);
    let q_weights = Array1::from_vec(vec![Complex::new(0.4, 0.1), Complex::new(0.6, -0.1)]);
    let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
        GenfmtJasEnergyGridBranchFromTransitionSetupInput {
            transition_setup: &transition_setup,
            transition_angular_momenta: left_right_data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            q_weights: q_weights.view(),
            transition_magnetic_offset: left_right_data.input().rotation_magnetic_offset,
            max_angular_momentum: left_right_data.input().max_angular_momentum,
            decomposition_l_max: None,
        },
    )?;
    let path_potential_indices = Array1::from_vec(vec![1, 0]);
    let mut xnlm = Array2::zeros((5, 5).f());
    for row in 0..xnlm.shape()[0] {
        for column in 0..xnlm.shape()[1] {
            xnlm[(row, column)] = 0.8 + 0.07 * (row as Real) + 0.03 * (column as Real);
        }
    }
    let energy_setup = GenfmtJasPathEnergyGridFromSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        xnlm: xnlm.view(),
        branch,
        momentum_zero_epsilon: 1.0e-16,
    };
    let path_inputs = [
        GenfmtJasPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 1.75,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtJasPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 24,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: energies.len(),
            degeneracy: 0.90,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence =
        genfmt_jas_path_sequence_from_driver_setup(GenfmtJasPathSequenceFromDriverSetupInput {
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
        })?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation_from_driver_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &expected_finalizations,
    })?;

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    Ok(())
}

#[test]
fn genfmt_jas_driver_output_assembles_empty_sequence_reference() -> Result<(), GenfmtError> {
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let driver_setup = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let path_inputs: [GenfmtJasPathEvaluationFromDriverSetupInput<'_>; 0] = [];

    let output = genfmt_jas_driver_output(GenfmtJasDriverOutputInput {
        driver_setup: &driver_setup,
        path_inputs: &path_inputs,
        initial_normalization: -1.0,
        nstar: None,
    })?;
    let expected_sequence =
        genfmt_jas_path_sequence_from_driver_setup(GenfmtJasPathSequenceFromDriverSetupInput {
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
        })?;

    assert_eq!(output.header, driver_setup.header);
    assert_eq!(output.path_sequence, expected_sequence);
    assert!(output.nstar_rows.is_none());
    assert_eq!(output.path_sequence.outputs.examined_path_count, 0);
    assert_eq!(output.path_sequence.outputs.retained_path_count, 0);
    assert_eq!(output.path_sequence.outputs.final_normalization, None);
    Ok(())
}

#[test]
fn jas_q_angles_match_feff_genfmtjas_reference() -> Result<(), GenfmtError> {
    let theta: [Real; 3] = [0.25, 1.2, 2.7];
    let phi: [Real; 3] = [0.4, -0.9, 1.6];
    let mut q_trig = Array2::zeros((3, 4).f());
    for q in 0..3 {
        q_trig[(q, 0)] = theta[q].cos();
        q_trig[(q, 1)] = theta[q].sin();
        q_trig[(q, 2)] = phi[q].cos();
        q_trig[(q, 3)] = phi[q].sin();
    }
    let q_weights = Array1::from_vec(vec![
        Complex::new(0.40, 0.10),
        Complex::new(0.25, -0.05),
        Complex::new(0.35, 0.02),
    ]);

    let angles = jas_q_angles(JasQAngleInput {
        qaverage: false,
        q_trig: q_trig.view(),
        q_weights: q_weights.view(),
    })?;

    assert_eq!(angles.phases.len(), 3);
    assert_eq!(angles.beta_angles.len(), 3);
    assert_eq!(angles.weights.len(), 3);
    for q in 0..3 {
        assert_complex_close(angles.phases[q], Complex::new(phi[q].cos(), -phi[q].sin()));
        assert_close(angles.beta_angles[q], theta[q]);
        assert_complex_close(angles.weights[q], q_weights[q]);
    }

    let averaged = jas_q_angles(JasQAngleInput {
        qaverage: true,
        q_trig: q_trig.view(),
        q_weights: q_weights.view(),
    })?;
    assert_eq!(averaged.phases.to_vec(), vec![Complex::new(1.0, 0.0)]);
    assert_eq!(averaged.beta_angles.to_vec(), vec![0.0]);
    assert_eq!(averaged.weights.to_vec(), vec![Complex::new(1.0, 0.0)]);
    Ok(())
}

#[test]
fn jas_q_angles_reject_invalid_inputs() {
    let q_trig = arr2(&[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]);
    let q_weights = Array1::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.1)]);
    let empty_weights = Array1::from_vec(Vec::<Complex>::new());
    assert_eq!(
        jas_q_angles(JasQAngleInput {
            qaverage: false,
            q_trig: q_trig.view(),
            q_weights: empty_weights.view(),
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "q_count",
            value: 0,
        })
    );

    let short_trig = arr2(&[[1.0, 0.0, 1.0]]);
    assert!(matches!(
        jas_q_angles(JasQAngleInput {
            qaverage: false,
            q_trig: short_trig.view(),
            q_weights: Array1::from_vec(vec![Complex::new(1.0, 0.0)]).view(),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "q_trig",
            axis: "component",
            ..
        })
    ));

    let mut bad_trig = q_trig.clone();
    bad_trig[(1, 2)] = f64::NAN;
    assert!(matches!(
        jas_q_angles(JasQAngleInput {
            qaverage: false,
            q_trig: bad_trig.view(),
            q_weights: q_weights.view(),
        }),
        Err(GenfmtError::NonFiniteTableScalar {
            table: "q_trig",
            row: 1,
            column: 2,
            ..
        })
    ));

    let mut bad_weights = q_weights.clone();
    bad_weights[0] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        jas_q_angles(JasQAngleInput {
            qaverage: false,
            q_trig: q_trig.view(),
            q_weights: bad_weights.view(),
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "q_weights",
            row: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_scattering_matrix_plan_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let plan = genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
        leg_count: 5,
        full_lambda_count: 7,
        initial_lambda_count: 3,
    })?;

    assert_eq!(plan.scattering_count, 4);
    assert_eq!(
        plan.tasks,
        vec![
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::First,
                current_leg_index: 1,
                previous_leg_index: 0,
                matrix_slot_index: 0,
                left_lambda_count: 7,
                right_lambda_count: 3,
            },
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::LastOrdinary,
                current_leg_index: 4,
                previous_leg_index: 3,
                matrix_slot_index: 3,
                left_lambda_count: 3,
                right_lambda_count: 7,
            },
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::Intermediate,
                current_leg_index: 2,
                previous_leg_index: 1,
                matrix_slot_index: 1,
                left_lambda_count: 7,
                right_lambda_count: 7,
            },
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::Intermediate,
                current_leg_index: 3,
                previous_leg_index: 2,
                matrix_slot_index: 2,
                left_lambda_count: 7,
                right_lambda_count: 7,
            },
        ]
    );
    Ok(())
}

#[test]
fn genfmt_scattering_matrix_plan_handles_short_paths() -> Result<(), GenfmtError> {
    let single = genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
        leg_count: 2,
        full_lambda_count: 4,
        initial_lambda_count: 2,
    })?;

    assert_eq!(single.scattering_count, 1);
    assert_eq!(
        single.tasks,
        vec![GenfmtScatteringMatrixTask {
            role: GenfmtScatteringMatrixRole::First,
            current_leg_index: 1,
            previous_leg_index: 0,
            matrix_slot_index: 0,
            left_lambda_count: 4,
            right_lambda_count: 2,
        }]
    );

    let three_leg = genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
        leg_count: 3,
        full_lambda_count: 5,
        initial_lambda_count: 2,
    })?;

    assert_eq!(three_leg.scattering_count, 2);
    assert_eq!(
        three_leg.tasks,
        vec![
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::First,
                current_leg_index: 1,
                previous_leg_index: 0,
                matrix_slot_index: 0,
                left_lambda_count: 5,
                right_lambda_count: 2,
            },
            GenfmtScatteringMatrixTask {
                role: GenfmtScatteringMatrixRole::LastOrdinary,
                current_leg_index: 2,
                previous_leg_index: 1,
                matrix_slot_index: 1,
                left_lambda_count: 2,
                right_lambda_count: 5,
            },
        ]
    );
    Ok(())
}

#[test]
fn genfmt_scattering_matrix_plan_rejects_invalid_inputs() {
    assert_eq!(
        genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
            leg_count: 1,
            full_lambda_count: 4,
            initial_lambda_count: 2,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: 1,
        })
    );
    assert_eq!(
        genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
            leg_count: 2,
            full_lambda_count: 0,
            initial_lambda_count: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "full_lambda_count",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
            leg_count: 2,
            full_lambda_count: 4,
            initial_lambda_count: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "initial_lambda_count",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_scattering_matrix_plan(GenfmtScatteringMatrixPlanInput {
            leg_count: 2,
            full_lambda_count: 2,
            initial_lambda_count: 3,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "scattering_matrix_plan",
            axis: "full_lambda",
            length: 2,
            required: 3,
        })
    );
}

#[test]
fn genfmt_path_matrix_trace_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let data = genfmt_path_trace_reference_data();
    let trace = genfmt_path_matrix_trace(data.input())?;

    assert_eq!(trace.product_matrix.shape(), &[4, 3]);
    assert_eq!(trace.product_matrix.strides(), &[1, 4]);
    assert_complex_close(trace.trace, Complex::new(-0.025_118_64, 0.051_806_58));
    assert_complex_close(
        complex_sum(&trace.product_matrix),
        Complex::new(-0.311_106, 0.437_922),
    );
    assert_complex_close(
        trace.product_matrix[(0, 0)],
        Complex::new(-0.008_713, 0.017_231),
    );
    assert_complex_close(
        trace.product_matrix[(1, 2)],
        Complex::new(-0.030_676, 0.030_262),
    );
    assert_complex_close(
        trace.product_matrix[(2, 1)],
        Complex::new(-0.029_937, 0.042_969),
    );
    assert_complex_close(
        trace.product_matrix[(3, 0)],
        Complex::new(-0.022_03, 0.054_86),
    );
    Ok(())
}

#[test]
fn genfmt_path_matrix_product_matches_shared_pmati_reference() -> Result<(), GenfmtError> {
    let data = genfmt_path_trace_reference_data();
    let product = genfmt_path_matrix_product(data.product_input())?;
    let trace = genfmt_path_matrix_trace(data.input())?;

    assert_eq!(product.product_matrix, trace.product_matrix);
    assert_eq!(product.product_matrix.shape(), &[4, 3]);
    assert_eq!(product.product_matrix.strides(), &[1, 4]);
    assert_complex_close(
        complex_sum(&product.product_matrix),
        Complex::new(-0.311_106, 0.437_922),
    );
    assert_complex_close(
        product.product_matrix[(1, 2)],
        Complex::new(-0.030_676, 0.030_262),
    );
    Ok(())
}

#[test]
fn genfmt_path_matrix_trace_handles_single_scattering_reference() -> Result<(), GenfmtError> {
    let first = genfmt_single_scattering_first_matrix();
    let intermediate = Array3::zeros((0, 3, 3).f());
    let termination = genfmt_single_scattering_termination_matrix();

    let trace = genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
        first_scattering: first.view(),
        intermediate_scattering: intermediate.view(),
        termination_matrix: termination.view(),
        full_lambda_count: 3,
        initial_lambda_count: 2,
    })?;

    assert_eq!(trace.product_matrix.shape(), &[3, 2]);
    assert_complex_close(trace.trace, Complex::new(0.053_5, 0.093_3));
    assert_complex_close(complex_sum(&trace.product_matrix), Complex::new(1.47, 0.21));
    Ok(())
}

#[test]
fn genfmt_scattering_path_product_matches_genfmtjas_pmati_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let product = genfmt_scattering_path_product(GenfmtScatteringPathProductInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: data.angular_limits.view(),
        phase_shifts: data.phase_shifts.view(),
        signed_angular_offset: 4,
        curved_wave_polynomials: data.curved_wave_polynomials.view(),
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
    })?;
    let ordinary_trace = genfmt_ordinary_path_trace(data.input())?;

    assert_eq!(
        product.scattering_matrices,
        ordinary_trace.scattering_matrices
    );
    assert_eq!(
        product.matrix_product.product_matrix,
        ordinary_trace.matrix_trace.product_matrix
    );
    assert_eq!(product.scattering_matrices.len(), 2);
    assert_eq!(product.matrix_product.product_matrix.shape(), &[4, 3]);
    assert_complex_close(
        complex_sum(&product.matrix_product.product_matrix),
        Complex::new(88_985.182_040_018_19, 57_942.831_138_064_736),
    );
    assert_complex_close(
        product.matrix_product.product_matrix[(2, 1)],
        Complex::new(39_785.854_879_542_83, 26_177.402_973_140_3),
    );
    Ok(())
}

#[test]
fn genfmt_scattering_path_product_rejects_invalid_inputs() -> Result<(), Box<dyn std::error::Error>>
{
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let one_leg = Array1::from_vec(vec![0]);

    assert_eq!(
        genfmt_scattering_path_product(GenfmtScatteringPathProductInput {
            path_potential_indices: one_leg.view(),
            ..data.scattering_product_input()
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_potential_indices",
            value: 1,
        })
    );
    Ok(())
}

#[test]
fn genfmt_path_matrix_trace_rejects_invalid_inputs() {
    let data = genfmt_path_trace_reference_data();
    assert!(matches!(
        genfmt_path_matrix_product(GenfmtPathMatrixProductInput {
            initial_lambda_count: 5,
            ..data.product_input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_product",
            axis: "full_lambda",
            length: 4,
            required: 5,
        })
    ));

    assert!(matches!(
        genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
            initial_lambda_count: 5,
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_product",
            axis: "full_lambda",
            length: 4,
            required: 5,
        })
    ));

    let short_first = Array2::zeros((3, 3).f());
    assert!(matches!(
        genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
            first_scattering: short_first.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "first_scattering",
            axis: "lambda",
            ..
        })
    ));

    let mut bad_intermediate = data.intermediate_scattering.clone();
    bad_intermediate[(1, 2, 3)] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
            intermediate_scattering: bad_intermediate.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: "intermediate_scattering",
            i0: 1,
            i1: 2,
            i2: 3,
            ..
        })
    ));

    let mut bad_termination = data.termination_matrix.clone();
    bad_termination[(1, 2)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
            termination_matrix: bad_termination.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "termination_matrix",
            row: 1,
            column: 2,
            ..
        })
    ));
}

#[test]
fn genfmt_ordinary_path_trace_matches_genfmtsub_call_order_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let trace = genfmt_ordinary_path_trace(data.input())?;

    assert_eq!(trace.scattering_matrices.len(), 2);
    assert_eq!(trace.scattering_matrices[0].shape(), &[4, 3]);
    assert_eq!(trace.scattering_matrices[1].shape(), &[3, 4]);
    assert_eq!(trace.termination_matrix.shape(), &[3, 3]);

    let first_expected = scattering_amplitude_matrix(data.first_scattering_input())?;
    let last_expected = scattering_amplitude_matrix(data.last_scattering_input())?;
    let termination_expected = polarized_scattering_amplitude_matrix(data.termination_input())?;
    assert_eq!(trace.scattering_matrices[0], first_expected);
    assert_eq!(trace.scattering_matrices[1], last_expected);
    assert_eq!(trace.termination_matrix, termination_expected);

    let mut padded_last = Array3::zeros((1, 4, 4).f());
    for row in 0..last_expected.shape()[0] {
        for column in 0..last_expected.shape()[1] {
            padded_last[(0, row, column)] = last_expected[(row, column)];
        }
    }
    let expected_trace = genfmt_path_matrix_trace(GenfmtPathMatrixTraceInput {
        first_scattering: first_expected.view(),
        intermediate_scattering: padded_last.view(),
        termination_matrix: termination_expected.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
    })?;

    assert_eq!(trace.matrix_trace, expected_trace);
    assert_complex_close(
        trace.matrix_trace.trace,
        Complex::new(3_345_293_043.485_528, 1_565_109_212.399_629_6),
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_trace_rejects_invalid_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let one_leg = Array1::from_vec(vec![0]);
    assert_eq!(
        genfmt_ordinary_path_trace(GenfmtOrdinaryPathTraceInput {
            path_potential_indices: one_leg.view(),
            ..data.input()
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_potential_indices",
            value: 1,
        })
    );

    let bad_potentials = Array1::from_vec(vec![0, 4, 0]);
    assert!(matches!(
        genfmt_ordinary_path_trace(GenfmtOrdinaryPathTraceInput {
            path_potential_indices: bad_potentials.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "angular_limits",
            axis: "potential",
            length: 2,
            required: 5,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_point_matches_genfmtsub_loop_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let (angular_limits, phase_shifts, radial_factors) =
        genfmt_ordinary_path_energy_point_tables(&data);
    let leg_lengths = Array1::from_vec(vec![1.25, 1.75, 0.95]);
    let complex_momentum = Complex::new(0.85, 0.20);
    let wave_number = 0.64;
    let accumulated_chi = Complex::new(0.25, -0.15);

    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: leg_lengths.view(),
        complex_momentum,
        momentum_zero_epsilon: 1.0e-16,
    })?;
    let leg_limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        energy_index: 0,
        max_m_plus_one: 3,
        max_n: 1,
    })?;
    let curved_wave_polynomials =
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: geometry.leg_rhos.view(),
            leg_limits: &leg_limits.limits,
            mixed_order_capacity: leg_limits.mixed_order_capacity,
        })?;
    let path_trace = genfmt_ordinary_path_trace(GenfmtOrdinaryPathTraceInput {
        curved_wave_polynomials: curved_wave_polynomials.tables.view(),
        angular_limits: angular_limits.index_axis(Axis(0), 0),
        phase_shifts: phase_shifts.index_axis(Axis(0), 0),
        radial_factors: radial_factors.index_axis(Axis(0), 0),
        ..data.input()
    })?;
    let path_factor = genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
        leg_rhos: geometry.leg_rhos.view(),
        wave_number,
        effective_path_length: geometry.effective_path_length,
    })?;
    let signal = genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
        accumulated_chi,
        path_trace: path_trace.matrix_trace.trace,
        path_factor: path_factor.factor,
        spin_channel_count: 2,
        spin_index: 0,
    })?;

    let energy_point = genfmt_ordinary_path_energy_point(GenfmtOrdinaryPathEnergyPointInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        energy_index: 0,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        phase_shifts: phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: leg_lengths.view(),
        complex_momentum,
        wave_number,
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        radial_factors: radial_factors.view(),
        transition_matrix: data.transition_matrix.view(),
        transition_magnetic_offset: 4,
        accumulated_chi,
        spin_channel_count: 2,
        spin_index: 0,
    })?;

    assert_eq!(energy_point.geometry, geometry);
    assert_eq!(energy_point.leg_limits, Some(leg_limits));
    assert_eq!(
        energy_point.curved_wave_polynomials,
        Some(curved_wave_polynomials)
    );
    assert_eq!(energy_point.path_trace, Some(path_trace));
    assert_eq!(energy_point.path_factor, Some(path_factor));
    assert_eq!(energy_point.signal, Some(signal));
    let actual_trace = energy_point
        .path_trace
        .as_ref()
        .expect("active energy has a trace")
        .matrix_trace
        .trace;
    let actual_factor = energy_point
        .path_factor
        .as_ref()
        .expect("active energy has a path factor")
        .factor;
    let actual_signal = energy_point.signal.expect("active energy has a signal");

    assert_complex_close(
        actual_trace,
        Complex::new(586_903_439.491_290_3, -636_084_850.220_311_9),
    );
    assert_complex_close(
        actual_factor,
        Complex::new(0.324_962_734_637_752_6, 0.044_544_920_300_945_284),
    );
    assert_complex_close(
        actual_signal.contribution,
        Complex::new(-219_056_095.623_094_98, 180_560_305.452_747_64),
    );
    assert_complex_close(
        actual_signal.accumulated_chi,
        Complex::new(-219_056_095.373_094_98, 180_560_305.302_747_64),
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_point_skips_zero_momentum_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let (angular_limits, phase_shifts, radial_factors) =
        genfmt_ordinary_path_energy_point_tables(&data);
    let leg_lengths = Array1::from_vec(vec![1.25, 1.75, 0.95]);

    let energy_point = genfmt_ordinary_path_energy_point(GenfmtOrdinaryPathEnergyPointInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        energy_index: 0,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: angular_limits.view(),
        phase_shifts: phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: leg_lengths.view(),
        complex_momentum: Complex::new(1.0e-18, 0.0),
        wave_number: 0.64,
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        radial_factors: radial_factors.view(),
        transition_matrix: data.transition_matrix.view(),
        transition_magnetic_offset: 4,
        accumulated_chi: Complex::new(f64::NAN, f64::NAN),
        spin_channel_count: 2,
        spin_index: 0,
    })?;

    assert!(!energy_point.geometry.active);
    assert_eq!(
        energy_point.geometry.leg_rhos.to_vec(),
        vec![
            Complex::new(1.25e-18, 0.0),
            Complex::new(1.75e-18, 0.0),
            Complex::new(0.95e-18, 0.0),
        ]
    );
    assert!(energy_point.leg_limits.is_none());
    assert!(energy_point.curved_wave_polynomials.is_none());
    assert!(energy_point.path_trace.is_none());
    assert!(energy_point.path_factor.is_none());
    assert!(energy_point.signal.is_none());
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_point_rejects_invalid_inputs()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let (angular_limits, _phase_shifts, radial_factors) =
        genfmt_ordinary_path_energy_point_tables(&data);
    let short_phase_shifts = Array3::zeros((0, 9, 2).f());
    let leg_lengths = Array1::from_vec(vec![1.25, 1.75, 0.95]);

    assert!(matches!(
        genfmt_ordinary_path_energy_point(GenfmtOrdinaryPathEnergyPointInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
            energy_index: 0,
            path_potential_indices: data.path_potential_indices.view(),
            angular_limits: angular_limits.view(),
            phase_shifts: short_phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: leg_lengths.view(),
            complex_momentum: Complex::new(0.85, 0.20),
            wave_number: 0.64,
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: data.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: data.xnlm.view(),
            eta_angles: data.eta_angles.view(),
            transition_angular_momenta: data.transition_angular_momenta.view(),
            radial_factors: radial_factors.view(),
            transition_matrix: data.transition_matrix.view(),
            transition_magnetic_offset: 4,
            accumulated_chi: Complex::new(0.25, -0.15),
            spin_channel_count: 2,
            spin_index: 0,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "phase_shifts",
            axis: "energy",
            length: 0,
            required: 1,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_grid_matches_genfmtsub_spin_loop_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let grid = genfmt_ordinary_path_energy_grid(GenfmtOrdinaryPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: grid_data.leg_lengths.view(),
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    })?;

    let mut expected_active = Array2::<bool>::from_elem((2, 3).f(), false);
    let mut expected_traces = Array2::<Complex>::zeros((2, 3).f());
    let mut expected_factors = Array2::<Complex>::zeros((2, 3).f());
    let mut expected_contributions = Array2::<Complex>::zeros((2, 3).f());
    let mut expected_chi = Array1::<Complex>::zeros(3);
    for spin in 0..2 {
        let phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: grid_data.spin_phase_shifts.view(),
            angular_limits: grid_data.angular_limits.view(),
            signed_angular_offset: 4,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: spin },
        })?;
        let radial_factors = genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: grid_data.spin_radial_factors.view(),
            spin_channel_count: 2,
            spin_index: spin,
        })?;
        for energy in 0..3 {
            let point = genfmt_ordinary_path_energy_point(GenfmtOrdinaryPathEnergyPointInput {
                m_indices: data.m_indices.view(),
                n_indices: data.n_indices.view(),
                full_lambda_count: 4,
                initial_lambda_count: 3,
                energy_index: energy,
                path_potential_indices: data.path_potential_indices.view(),
                angular_limits: grid_data.angular_limits.view(),
                phase_shifts: phase_shifts.phase_shifts.view(),
                signed_angular_offset: 4,
                leg_lengths: grid_data.leg_lengths.view(),
                complex_momentum: grid_data.complex_momenta[(energy, spin)],
                wave_number: grid_data.wave_numbers[energy],
                momentum_zero_epsilon: 1.0e-16,
                max_m_plus_one: 3,
                max_n: 1,
                rotations: data.rotations.view(),
                rotation_magnetic_offset: 4,
                xnlm: data.xnlm.view(),
                eta_angles: data.eta_angles.view(),
                transition_angular_momenta: data.transition_angular_momenta.view(),
                radial_factors: radial_factors.radial_factors.view(),
                transition_matrix: grid_data.transition_matrices.index_axis(Axis(0), spin),
                transition_magnetic_offset: 4,
                accumulated_chi: expected_chi[energy],
                spin_channel_count: 2,
                spin_index: spin,
            })?;

            expected_active[(spin, energy)] = point.geometry.active;
            if let Some(path_trace) = point.path_trace.as_ref() {
                expected_traces[(spin, energy)] = path_trace.matrix_trace.trace;
            }
            if let Some(path_factor) = point.path_factor.as_ref() {
                expected_factors[(spin, energy)] = path_factor.factor;
            }
            if let Some(signal) = point.signal {
                expected_contributions[(spin, energy)] = signal.contribution;
                expected_chi[energy] = signal.accumulated_chi;
            }
        }
    }

    assert_eq!(grid.active, expected_active);
    assert_eq!(grid.active.row(0).to_vec(), vec![true, false, true]);
    assert_eq!(grid.active.row(1).to_vec(), vec![true, true, true]);
    assert_complex_array2_close(&grid.path_traces, &expected_traces);
    assert_complex_array2_close(&grid.path_factors, &expected_factors);
    assert_complex_array2_close(&grid.signals.contributions, &expected_contributions);
    assert_complex_array1_close(&grid.signals.chi, &expected_chi);
    assert_complex_close(grid.signals.contributions[(0, 1)], Complex::new(0.0, 0.0));
    assert_complex_close(
        grid.signals.chi[0],
        grid.signals.contributions[(0, 0)] + grid.signals.contributions[(1, 0)],
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_grid_from_setup_threads_path_setup_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let setup_input = GenfmtOrdinaryPathEnergyGridFromSetupInput {
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    };

    let energy_input = genfmt_ordinary_path_energy_grid_from_setup(setup_input)?;

    assert_eq!(
        energy_input.m_indices.to_vec(),
        path_setup.lambda.m_indices.to_vec()
    );
    assert_eq!(
        energy_input.n_indices.to_vec(),
        path_setup.lambda.n_indices.to_vec()
    );
    assert_eq!(energy_input.full_lambda_count, 4);
    assert_eq!(energy_input.initial_lambda_count, 3);
    assert_eq!(energy_input.max_m_plus_one, 3);
    assert_eq!(energy_input.max_n, 1);
    assert_eq!(
        energy_input.path_potential_indices.to_vec(),
        data.path_potential_indices.to_vec()
    );
    assert_array_close(
        &energy_input.leg_lengths.to_owned(),
        grid_data
            .leg_lengths
            .as_slice()
            .expect("contiguous fixture"),
    );

    let grid = genfmt_ordinary_path_energy_grid(energy_input)?;
    let expected = genfmt_ordinary_path_energy_grid(GenfmtOrdinaryPathEnergyGridInput {
        m_indices: path_setup.lambda.m_indices.view(),
        n_indices: path_setup.lambda.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: grid_data.leg_lengths.view(),
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    })?;
    assert_eq!(grid.active, expected.active);
    assert_complex_array2_close(&grid.path_traces, &expected.path_traces);
    assert_complex_array2_close(&grid.path_factors, &expected.path_factors);
    assert_complex_array2_close(&grid.signals.contributions, &expected.signals.contributions);
    assert_complex_array1_close(&grid.signals.chi, &expected.signals.chi);

    let short_potential_indices = Array1::from_vec(vec![0, 1]);
    assert!(matches!(
        genfmt_ordinary_path_energy_grid_from_setup(GenfmtOrdinaryPathEnergyGridFromSetupInput {
            path_potential_indices: short_potential_indices.view(),
            ..setup_input
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_potential_indices",
            axis: "leg",
            length: 2,
            required: 3,
        })
    ));

    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_grid_from_driver_setup_threads_spin_momentum_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.05),
        Complex::new(0.425, 0.07),
        Complex::new(0.525, 0.09),
    ]);
    let spin_reference_energies =
        genfmt_reference_energies_for_complex_momenta(&energies, &grid_data.complex_momenta);
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", "O"];
    let driver_setup = genfmt_driver_setup(GenfmtDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        angular_limits: grid_data.angular_limits.view(),
        signed_angular_offset: 4,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;

    let driver_input = genfmt_ordinary_path_energy_grid_from_driver_setup(
        GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
            driver_setup: &driver_setup,
            path_setup: &path_setup,
            path_potential_indices: data.path_potential_indices.view(),
            angular_limits: grid_data.angular_limits.view(),
            spin_phase_shifts: grid_data.spin_phase_shifts.view(),
            signed_angular_offset: 4,
            momentum_zero_epsilon: 1.0e-16,
            xnlm: data.xnlm.view(),
            transition_angular_momenta: data.transition_angular_momenta.view(),
            spin_radial_factors: grid_data.spin_radial_factors.view(),
            transition_matrices: grid_data.transition_matrices.view(),
            transition_magnetic_offset: 4,
        },
    )?;
    let setup_input =
        genfmt_ordinary_path_energy_grid_from_setup(GenfmtOrdinaryPathEnergyGridFromSetupInput {
            path_setup: &path_setup,
            path_potential_indices: data.path_potential_indices.view(),
            angular_limits: grid_data.angular_limits.view(),
            spin_phase_shifts: grid_data.spin_phase_shifts.view(),
            signed_angular_offset: 4,
            complex_momenta: driver_setup.spin_momentum_grid.complex_momenta.view(),
            wave_numbers: driver_setup.spin_momentum_grid.wave_numbers.view(),
            momentum_zero_epsilon: 1.0e-16,
            xnlm: data.xnlm.view(),
            transition_angular_momenta: data.transition_angular_momenta.view(),
            spin_radial_factors: grid_data.spin_radial_factors.view(),
            transition_matrices: grid_data.transition_matrices.view(),
            transition_magnetic_offset: 4,
            spin_channel_count: driver_setup.spin_channel_count,
        })?;

    assert_eq!(
        driver_input.complex_momenta.to_owned(),
        driver_setup.spin_momentum_grid.complex_momenta
    );
    assert_array_close(
        &driver_input.wave_numbers.to_owned(),
        driver_setup
            .spin_momentum_grid
            .wave_numbers
            .as_slice()
            .expect("contiguous fixture"),
    );
    assert_eq!(driver_input.spin_channel_count, 2);
    assert_eq!(
        driver_input.full_lambda_count,
        setup_input.full_lambda_count
    );
    assert_eq!(
        driver_input.initial_lambda_count,
        setup_input.initial_lambda_count
    );

    let grid = genfmt_ordinary_path_energy_grid(driver_input)?;
    let expected = genfmt_ordinary_path_energy_grid(setup_input)?;
    assert_eq!(grid, expected);
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_grid_rejects_invalid_inputs()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let short_complex_momenta = Array2::<Complex>::zeros((3, 1).f());

    assert!(matches!(
        genfmt_ordinary_path_energy_grid(GenfmtOrdinaryPathEnergyGridInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
            path_potential_indices: data.path_potential_indices.view(),
            angular_limits: grid_data.angular_limits.view(),
            spin_phase_shifts: grid_data.spin_phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: grid_data.leg_lengths.view(),
            complex_momenta: short_complex_momenta.view(),
            wave_numbers: grid_data.wave_numbers.view(),
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: data.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: data.xnlm.view(),
            eta_angles: data.eta_angles.view(),
            transition_angular_momenta: data.transition_angular_momenta.view(),
            spin_radial_factors: grid_data.spin_radial_factors.view(),
            transition_matrices: grid_data.transition_matrices.view(),
            transition_magnetic_offset: 4,
            spin_channel_count: 2,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "complex_momenta",
            axis: "spin",
            length: 1,
            required: 2,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_ordinary_path_energy_grid_finalization_uses_accumulated_grid_signal_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let energy_grid = genfmt_ordinary_path_energy_grid(GenfmtOrdinaryPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: grid_data.leg_lengths.view(),
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    })?;
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let effective_half_path_length = grid_data.leg_lengths.iter().sum::<Real>() / 2.0;

    let finalization = genfmt_ordinary_path_energy_grid_finalization(
        GenfmtOrdinaryPathEnergyGridFinalizationInput {
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            energy_grid: &energy_grid,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: -1.0,
            effective_half_path_length_bohr: effective_half_path_length,
            potential_indices: data.path_potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: data.eta_angles.view(),
            leg_lengths: grid_data.leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        },
    )?;
    let expected_decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: 19,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        chi: energy_grid.signals.chi.view(),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 2.75,
        current_normalization: -1.0,
        effective_half_path_length_bohr: effective_half_path_length,
        potential_indices: data.path_potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: data.eta_angles.view(),
        leg_lengths: grid_data.leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(finalization.signals, energy_grid.signals);
    assert_eq!(finalization.output_decision, expected_decision);
    assert!(finalization.output_decision.retention.keep);
    assert!(finalization.output_decision.retained_output.is_some());
    Ok(())
}

#[test]
fn genfmt_ordinary_path_evaluation_composes_energy_grid_and_finalization()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let energy_input = GenfmtOrdinaryPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: grid_data.leg_lengths.view(),
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let effective_half_path_length = grid_data.leg_lengths.iter().sum::<Real>() / 2.0;

    let evaluation = genfmt_ordinary_path_evaluation(GenfmtOrdinaryPathEvaluationInput {
        energy_grid: energy_input,
        path_index: 19,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 2.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;

    let expected_grid = genfmt_ordinary_path_energy_grid(energy_input)?;
    let expected_finalization = genfmt_ordinary_path_energy_grid_finalization(
        GenfmtOrdinaryPathEnergyGridFinalizationInput {
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            energy_grid: &expected_grid,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: -1.0,
            effective_half_path_length_bohr: effective_half_path_length,
            potential_indices: data.path_potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: data.eta_angles.view(),
            leg_lengths: grid_data.leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        },
    )?;

    assert_eq!(evaluation.energy_grid, expected_grid);
    assert_eq!(evaluation.finalization, expected_finalization);
    Ok(())
}

#[test]
fn genfmt_ordinary_path_evaluation_from_setup_matches_manual_driver_wiring()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energy_setup = GenfmtOrdinaryPathEnergyGridFromSetupInput {
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);

    let evaluation =
        genfmt_ordinary_path_evaluation_from_setup(GenfmtOrdinaryPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: -1.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        })?;
    let expected_energy_input = genfmt_ordinary_path_energy_grid_from_setup(energy_setup)?;
    let expected = genfmt_ordinary_path_evaluation(GenfmtOrdinaryPathEvaluationInput {
        energy_grid: expected_energy_input,
        path_index: 19,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 2.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;
    assert_eq!(evaluation, expected);

    let short_positions = arr2(&[[0.0, 0.0, 0.0], [0.4, -0.3, 1.2]]);
    assert!(matches!(
        genfmt_ordinary_path_evaluation_from_setup(GenfmtOrdinaryPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: -1.0,
            positions: short_positions.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "positions",
            axis: "leg",
            length: 2,
            required: 3,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_ordinary_path_evaluation_from_driver_setup_matches_manual_driver_wiring()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.05),
        Complex::new(0.425, 0.07),
        Complex::new(0.525, 0.09),
    ]);
    let spin_reference_energies =
        genfmt_reference_energies_for_complex_momenta(&energies, &grid_data.complex_momenta);
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", "O"];
    let driver_setup = genfmt_driver_setup(GenfmtDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        angular_limits: grid_data.angular_limits.view(),
        signed_angular_offset: 4,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let energy_setup = GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
    };
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);

    let evaluation = genfmt_ordinary_path_evaluation_from_driver_setup(
        GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: -1.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    )?;
    let expected_energy_input = genfmt_ordinary_path_energy_grid_from_driver_setup(energy_setup)?;
    let expected = genfmt_ordinary_path_evaluation(GenfmtOrdinaryPathEvaluationInput {
        energy_grid: expected_energy_input,
        path_index: 19,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: driver_setup
            .momentum_grid
            .complex_momentum_magnitudes
            .view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 2.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: path_setup.angles.beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;
    assert_eq!(evaluation, expected);

    let short_positions = arr2(&[[0.0, 0.0, 0.0], [0.4, -0.3, 1.2]]);
    assert!(matches!(
        genfmt_ordinary_path_evaluation_from_driver_setup(
            GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
                energy_grid: energy_setup,
                path_index: 19,
                print_level: 1,
                curved_wave_criterion_percent: 120.0,
                edge_start_index: 0,
                active_energy_count: 3,
                degeneracy: 2.75,
                current_normalization: -1.0,
                positions: short_positions.view(),
                phase_epsilon: 1.0e-16,
            }
        ),
        Err(GenfmtError::TableAxisTooShort {
            table: "positions",
            axis: "leg",
            length: 2,
            required: 3,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_ordinary_path_sequence_threads_normalization_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let energy_input = GenfmtOrdinaryPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 3,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: grid_data.leg_lengths.view(),
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: data.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: data.xnlm.view(),
        eta_angles: data.eta_angles.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let path_inputs = [
        GenfmtOrdinaryPathEvaluationInput {
            energy_grid: energy_input,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: 999.0,
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtOrdinaryPathEvaluationInput {
            energy_grid: energy_input,
            path_index: 20,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.50,
            current_normalization: 999.0,
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence = genfmt_ordinary_path_sequence(GenfmtOrdinaryPathSequenceInput {
        path_inputs: &path_inputs,
        initial_normalization: -1.0,
    })?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &expected_finalizations,
    });

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    assert_eq!(
        sequence.evaluations[0]
            .finalization
            .output_decision
            .importance
            .normalization,
        sequence.evaluations[1]
            .finalization
            .output_decision
            .importance
            .normalization
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_sequence_from_setup_threads_normalization_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energy_setup = GenfmtOrdinaryPathEnergyGridFromSetupInput {
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        complex_momenta: grid_data.complex_momenta.view(),
        wave_numbers: grid_data.wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
        spin_channel_count: 2,
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_inputs = [
        GenfmtOrdinaryPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtOrdinaryPathEvaluationFromSetupInput {
            energy_grid: energy_setup,
            path_index: 20,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.50,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence =
        genfmt_ordinary_path_sequence_from_setup(GenfmtOrdinaryPathSequenceFromSetupInput {
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
        })?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation_from_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &expected_finalizations,
    });

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    assert_eq!(
        sequence.evaluations[0]
            .finalization
            .output_decision
            .importance
            .normalization,
        sequence.evaluations[1]
            .finalization
            .output_decision
            .importance
            .normalization
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_sequence_from_driver_setup_threads_normalization_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.05),
        Complex::new(0.425, 0.07),
        Complex::new(0.525, 0.09),
    ]);
    let spin_reference_energies =
        genfmt_reference_energies_for_complex_momenta(&energies, &grid_data.complex_momenta);
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", "O"];
    let driver_setup = genfmt_driver_setup(GenfmtDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        angular_limits: grid_data.angular_limits.view(),
        signed_angular_offset: 4,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let energy_setup = GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
    };
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_inputs = [
        GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 20,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.50,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence = genfmt_ordinary_path_sequence_from_driver_setup(
        GenfmtOrdinaryPathSequenceFromDriverSetupInput {
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
        },
    )?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_ordinary_path_evaluation_from_driver_setup(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &expected_finalizations,
    });

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    assert_eq!(
        sequence.evaluations[0]
            .finalization
            .output_decision
            .importance
            .normalization,
        sequence.evaluations[1]
            .finalization
            .output_decision
            .importance
            .normalization
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_driver_output_assembles_header_sequence_and_nstar_reference()
-> Result<(), Box<dyn std::error::Error>> {
    let data = genfmt_ordinary_path_trace_reference_data()?;
    let grid_data = genfmt_ordinary_energy_grid_reference_data(&data, 3, 2);
    let path_setup = genfmt_ordinary_reference_path_setup(&data, &grid_data);
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.05),
        Complex::new(0.425, 0.07),
        Complex::new(0.525, 0.09),
    ]);
    let spin_reference_energies =
        genfmt_reference_energies_for_complex_momenta(&energies, &grid_data.complex_momenta);
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", "O"];
    let driver_setup = genfmt_driver_setup(GenfmtDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        angular_limits: grid_data.angular_limits.view(),
        signed_angular_offset: 4,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    })?;
    let energy_setup = GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
        driver_setup: &driver_setup,
        path_setup: &path_setup,
        path_potential_indices: data.path_potential_indices.view(),
        angular_limits: grid_data.angular_limits.view(),
        spin_phase_shifts: grid_data.spin_phase_shifts.view(),
        signed_angular_offset: 4,
        momentum_zero_epsilon: 1.0e-16,
        xnlm: data.xnlm.view(),
        transition_angular_momenta: data.transition_angular_momenta.view(),
        spin_radial_factors: grid_data.spin_radial_factors.view(),
        transition_matrices: grid_data.transition_matrices.view(),
        transition_magnetic_offset: 4,
    };
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let path_inputs = [
        GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 19,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 2.75,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
            energy_grid: energy_setup,
            path_index: 20,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.50,
            current_normalization: 999.0,
            positions: positions.view(),
            phase_epsilon: 1.0e-16,
        },
    ];
    let nstar = GenfmtNStarDriverInput {
        primary_polarization: [0.0, 0.0, 1.0],
        ellipticity_vector: [1.0, 0.0, 0.0],
        initial_l: 2,
        ellipticity: 0.0,
    };

    let output = genfmt_ordinary_driver_output(GenfmtOrdinaryDriverOutputInput {
        driver_setup: &driver_setup,
        path_inputs: &path_inputs,
        initial_normalization: -1.0,
        nstar: Some(nstar),
    })?;
    let expected_sequence = genfmt_ordinary_path_sequence_from_driver_setup(
        GenfmtOrdinaryPathSequenceFromDriverSetupInput {
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
        },
    )?;
    let nstar_path_inputs = [
        GenfmtNStarPathInput {
            positions: positions.view(),
            degeneracy: 2.75,
        },
        GenfmtNStarPathInput {
            positions: positions.view(),
            degeneracy: 1.50,
        },
    ];
    let expected_nstar = genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: nstar.primary_polarization,
        ellipticity_vector: nstar.ellipticity_vector,
        initial_l: nstar.initial_l,
        ellipticity: nstar.ellipticity,
        path_inputs: &nstar_path_inputs,
    })?;

    assert_eq!(output.header, driver_setup.header);
    assert_eq!(output.path_sequence, expected_sequence);
    assert_eq!(output.nstar_rows, Some(expected_nstar));
    assert_eq!(
        output.path_sequence.outputs.examined_path_count,
        output.nstar_rows.as_ref().expect("nstar rows").rows.len()
    );
    Ok(())
}

#[test]
fn genfmt_jas_left_right_path_trace_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    let path_product = genfmt_jas_path_product();
    let left_amplitudes = genfmt_jas_left_amplitudes();
    let right_amplitudes = genfmt_jas_right_amplitudes();
    let decomposed_left = genfmt_jas_decomposed_left_amplitudes();
    let decomposed_right = genfmt_jas_decomposed_right_amplitudes();

    let trace = genfmt_jas_left_right_path_trace(GenfmtJasLeftRightPathTraceInput {
        path_product: path_product.view(),
        left_amplitudes: left_amplitudes.view(),
        right_amplitudes: right_amplitudes.view(),
        lambda_count: 2,
        decomposed_left_amplitudes: Some(decomposed_left.view()),
        decomposed_right_amplitudes: Some(decomposed_right.view()),
    })?;
    let decomposed = trace
        .decomposed_traces
        .as_ref()
        .expect("decomposition was requested");

    assert_complex_close(trace.trace, Complex::new(2.386_2, 0.992_5));
    assert_eq!(decomposed.shape(), &[2, 2]);
    assert_eq!(decomposed.strides(), &[1, 2]);
    assert_complex_close(decomposed[(0, 0)], Complex::new(0.145_003, 0.142_085));
    assert_complex_close(decomposed[(0, 1)], Complex::new(0.149_958, 0.172_098));
    assert_complex_close(decomposed[(1, 0)], Complex::new(0.189_555, 0.151_789));
    assert_complex_close(decomposed[(1, 1)], Complex::new(0.199_174, 0.186_93));
    Ok(())
}

#[test]
fn genfmt_jas_spherical_path_trace_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    let path_product = genfmt_jas_path_product();
    let amplitudes = genfmt_jas_spherical_amplitudes();
    let decomposed_amplitudes = genfmt_jas_spherical_decomposed_amplitudes();

    let trace = genfmt_jas_spherical_path_trace(GenfmtJasSphericalPathTraceInput {
        path_product: path_product.view(),
        amplitudes: amplitudes.view(),
        lambda_count: 2,
        decomposed_amplitudes: Some(decomposed_amplitudes.view()),
    })?;
    let decomposed = trace
        .decomposed_traces
        .as_ref()
        .expect("decomposition was requested");

    assert_complex_close(trace.trace, Complex::new(0.877, 0.509));
    assert_eq!(decomposed.shape(), &[2, 2]);
    assert_eq!(decomposed.strides(), &[1, 2]);
    assert_complex_close(decomposed[(0, 0)], Complex::new(0.379, 0.576_6));
    assert_complex_close(decomposed[(0, 1)], Complex::new(0.0, 0.0));
    assert_complex_close(decomposed[(1, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(decomposed[(1, 1)], Complex::new(0.617, 0.753));
    Ok(())
}

#[test]
fn genfmt_jas_path_trace_matches_left_right_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let path_product = genfmt_jas_path_product_for(5);
    let expected_amplitudes = jas_left_right_amplitude_matrices(data.input())?;
    let expected_trace = genfmt_jas_left_right_path_trace(GenfmtJasLeftRightPathTraceInput {
        path_product: path_product.view(),
        left_amplitudes: expected_amplitudes.left_amplitudes.view(),
        right_amplitudes: expected_amplitudes.right_amplitudes.view(),
        lambda_count: 5,
        decomposed_left_amplitudes: expected_amplitudes
            .decomposed_left_amplitudes
            .as_ref()
            .map(|table| table.view()),
        decomposed_right_amplitudes: expected_amplitudes
            .decomposed_right_amplitudes
            .as_ref()
            .map(|table| table.view()),
    })?;

    let GenfmtJasPathTrace::LeftRight { amplitudes, trace } =
        genfmt_jas_path_trace(GenfmtJasPathTraceInput::LeftRight {
            path_product: path_product.view(),
            amplitude_input: data.input(),
        })?
    else {
        panic!("expected left/right JAS path-trace branch");
    };
    let decomposed = trace
        .decomposed_traces
        .as_ref()
        .expect("angular decomposition was requested");

    assert_eq!(amplitudes, expected_amplitudes);
    assert_eq!(trace, expected_trace);
    assert_complex_close(
        trace.trace,
        Complex::new(-0.000_566_505_310_560_807, 0.002_845_101_659_357_188),
    );
    assert_eq!(decomposed.shape(), &[3, 3]);
    assert_eq!(decomposed.strides(), &[1, 3]);
    assert_complex_close(
        decomposed[(2, 1)],
        Complex::new(
            0.000_045_167_019_158_626_565,
            -0.000_001_965_670_328_312_428_5,
        ),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_trace_matches_spherical_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas0_reference_data();
    let path_product = genfmt_jas_path_product_for(4);
    let expected_amplitudes = jas_scattering_amplitude_matrices(data.input())?;
    let expected_trace = genfmt_jas_spherical_path_trace(GenfmtJasSphericalPathTraceInput {
        path_product: path_product.view(),
        amplitudes: expected_amplitudes.amplitudes.view(),
        lambda_count: 4,
        decomposed_amplitudes: expected_amplitudes
            .decomposed_amplitudes
            .as_ref()
            .map(|table| table.view()),
    })?;

    let GenfmtJasPathTrace::Spherical { amplitudes, trace } =
        genfmt_jas_path_trace(GenfmtJasPathTraceInput::Spherical {
            path_product: path_product.view(),
            amplitude_input: data.input(),
        })?
    else {
        panic!("expected spherical JAS path-trace branch");
    };
    let decomposed = trace
        .decomposed_traces
        .as_ref()
        .expect("angular decomposition was requested");

    assert_eq!(amplitudes, expected_amplitudes);
    assert_eq!(trace, expected_trace);
    assert_complex_close(
        trace.trace,
        Complex::new(-0.000_609_334_447_579_917_8, -0.001_561_699_649_918_816_8),
    );
    assert_eq!(decomposed.shape(), &[3, 3]);
    assert_eq!(decomposed.strides(), &[1, 3]);
    assert_complex_close(decomposed[(0, 1)], Complex::new(0.0, 0.0));
    assert_complex_close(
        decomposed[(2, 2)],
        Complex::new(-0.000_272_555_093_610_689_4, -0.000_933_121_089_174_582_7),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_trace_rejects_invalid_inputs() {
    let data = mmtrxijas_reference_data();
    let short_product = Array2::zeros((4, 5).f());

    assert!(matches!(
        genfmt_jas_path_trace(GenfmtJasPathTraceInput::LeftRight {
            path_product: short_product.view(),
            amplitude_input: data.input(),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_product",
            axis: "lambda",
            length: 4,
            required: 5,
        })
    ));
}

#[test]
fn genfmt_jas_path_energy_point_matches_left_right_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let common = genfmt_jas_energy_point_common_data(5);
    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: common.leg_lengths.view(),
        complex_momentum: common.complex_momentum,
        momentum_zero_epsilon: 1.0e-16,
    })?;
    let leg_limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        energy_index: 0,
        max_m_plus_one: 3,
        max_n: 1,
    })?;
    let curved_wave_polynomials =
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: geometry.leg_rhos.view(),
            leg_limits: &leg_limits.limits,
            mixed_order_capacity: leg_limits.mixed_order_capacity,
        })?;
    let scattering_product = genfmt_scattering_path_product(
        data.jas_scattering_product_input(&common, &curved_wave_polynomials.tables),
    )?;
    let expected_trace = genfmt_jas_path_trace(GenfmtJasPathTraceInput::LeftRight {
        path_product: scattering_product.matrix_product.product_matrix.view(),
        amplitude_input: data
            .energy_point_amplitude_input(&curved_wave_polynomials.tables, common.xnlm.view()),
    })?;
    let expected_factor = genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
        leg_rhos: geometry.leg_rhos.view(),
        wave_number: common.wave_number,
        effective_path_length: geometry.effective_path_length,
    })?;
    let expected_signal = genfmt_jas_path_signal(GenfmtJasPathSignalInput {
        path_trace: genfmt_jas_trace_value(&expected_trace),
        path_factor: expected_factor.factor,
        decomposed_traces: genfmt_jas_decomposed_trace_view(&expected_trace),
    })?;

    let energy_point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        energy_index: 0,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momentum: common.complex_momentum,
        wave_number: common.wave_number,
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_point_branch_input(),
    })?;

    assert_eq!(energy_point.geometry, geometry);
    assert_eq!(energy_point.leg_limits, Some(leg_limits));
    assert_eq!(
        energy_point.curved_wave_polynomials,
        Some(curved_wave_polynomials)
    );
    assert_eq!(energy_point.scattering_product, Some(scattering_product));
    assert_eq!(energy_point.path_trace, Some(expected_trace));
    assert_eq!(energy_point.path_factor, Some(expected_factor));
    assert_eq!(energy_point.signal, Some(expected_signal));
    let signal = energy_point.signal.expect("active JAS energy has a signal");
    assert_complex_close(
        genfmt_jas_trace_value(
            energy_point
                .path_trace
                .as_ref()
                .expect("active JAS energy has a trace"),
        ),
        Complex::new(-64_226.696_460_605_02, -511_230.213_491_839_5),
    );
    assert_complex_close(
        signal.chi,
        Complex::new(-142_402.260_630_861_33, -652_304.909_775_097_6),
    );
    assert_complex_close(
        signal.decomposed_sum.expect("lgcchi"),
        Complex::new(25_874.868_171_894_763, 32_951.397_262_588_15),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_energy_point_matches_spherical_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas0_reference_data();
    let common = genfmt_jas_energy_point_common_data(4);
    let energy_point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 4,
        energy_index: 0,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momentum: common.complex_momentum,
        wave_number: common.wave_number,
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_point_branch_input(),
    })?;
    let trace = energy_point
        .path_trace
        .as_ref()
        .expect("active JAS energy has a trace");
    let signal = energy_point.signal.expect("active JAS energy has a signal");

    assert!(matches!(trace, GenfmtJasPathTrace::Spherical { .. }));
    assert_eq!(
        energy_point
            .scattering_product
            .as_ref()
            .expect("scattering product")
            .matrix_product
            .product_matrix
            .shape(),
        &[4, 4]
    );
    assert_complex_close(
        genfmt_jas_trace_value(trace),
        Complex::new(5_620.672_505_128_577, -1_714.844_807_417_113_3),
    );
    assert_complex_close(
        signal.chi,
        Complex::new(7_054.280_021_153_908, -2_867.446_050_766_692_7),
    );
    assert_complex_close(
        signal.decomposed_sum.expect("lgcchi"),
        Complex::new(6_083.575_821_763_184, -2_715.284_584_314_135),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_energy_point_skips_zero_momentum_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let mut common = genfmt_jas_energy_point_common_data(5);
    common.complex_momentum = Complex::new(1.0e-18, 0.0);

    let energy_point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        energy_index: 0,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momentum: common.complex_momentum,
        wave_number: common.wave_number,
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_point_branch_input(),
    })?;

    assert!(!energy_point.geometry.active);
    assert!(energy_point.leg_limits.is_none());
    assert!(energy_point.curved_wave_polynomials.is_none());
    assert!(energy_point.scattering_product.is_none());
    assert!(energy_point.path_trace.is_none());
    assert!(energy_point.path_factor.is_none());
    assert!(energy_point.signal.is_none());
    Ok(())
}

#[test]
fn genfmt_jas_path_energy_point_rejects_invalid_inputs() {
    let data = mmtrxijas_reference_data();
    let common = genfmt_jas_energy_point_common_data(5);
    let short_phase_shifts = Array3::zeros((0, 9, 1).f());

    assert!(matches!(
        genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 5,
            initial_lambda_count: 5,
            energy_index: 0,
            path_potential_indices: common.path_potential_indices.view(),
            angular_limits: common.angular_limits.view(),
            phase_shifts: short_phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: common.leg_lengths.view(),
            complex_momentum: common.complex_momentum,
            wave_number: common.wave_number,
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: common.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: common.xnlm.view(),
            eta_angles: common.eta_angles.view(),
            branch: data.energy_point_branch_input(),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "phase_shifts",
            axis: "energy",
            length: 0,
            required: 1,
        })
    ));
}

#[test]
fn genfmt_jas_path_energy_grid_matches_left_right_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(5, 3);
    let radial_factors = jas_radial_factor_energy_grid(data.radial_factors.view(), 3);
    let grid = genfmt_jas_path_energy_grid(GenfmtJasPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_grid_branch_input(radial_factors.view()),
    })?;

    let mut expected_active = Array1::<bool>::from_elem(3, false);
    let mut expected_traces = Array1::<Complex>::zeros(3);
    let mut expected_factors = Array1::<Complex>::zeros(3);
    let mut expected_decomposed = Array3::<Complex>::zeros((3, 3, 3).f());
    for energy in 0..3 {
        let point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 5,
            initial_lambda_count: 5,
            energy_index: energy,
            path_potential_indices: common.path_potential_indices.view(),
            angular_limits: common.angular_limits.view(),
            phase_shifts: common.phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: common.leg_lengths.view(),
            complex_momentum: complex_momenta[energy],
            wave_number: wave_numbers[energy],
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: common.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: common.xnlm.view(),
            eta_angles: common.eta_angles.view(),
            branch: data
                .energy_point_branch_input_with_radial(radial_factors.index_axis(Axis(0), energy)),
        })?;
        expected_active[energy] = point.geometry.active;
        if let Some(path_factor) = point.path_factor {
            expected_factors[energy] = path_factor.factor;
        }
        if let Some(path_trace) = point.path_trace {
            expected_traces[energy] = genfmt_jas_trace_value(&path_trace);
            let decomposed = genfmt_jas_decomposed_trace_view(&path_trace).expect("pgtrl traces");
            for row in 0..3 {
                for column in 0..3 {
                    expected_decomposed[(row, column, energy)] = decomposed[(row, column)];
                }
            }
        }
    }
    let expected_signals = genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
        path_traces: expected_traces.view(),
        path_factors: expected_factors.view(),
        active: expected_active.view(),
        decomposed_traces: Some(expected_decomposed.view()),
    })?;

    assert_eq!(grid.active, expected_active);
    assert_complex_array1_close(&grid.path_traces, &expected_traces);
    assert_complex_array1_close(&grid.path_factors, &expected_factors);
    assert_complex_array3_close(
        grid.decomposed_traces.as_ref().expect("pgtrl traces"),
        &expected_decomposed,
    );
    assert_eq!(grid.signals, expected_signals);
    assert_eq!(grid.active.to_vec(), vec![true, false, true]);
    assert_complex_close(
        grid.signals.chi[0],
        Complex::new(-142_402.260_630_861_33, -652_304.909_775_097_6),
    );
    assert_complex_close(grid.signals.chi[1], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn genfmt_jas_path_energy_grid_matches_spherical_branch_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas0_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(4, 2);
    let radial_factors = jas_radial_factor_energy_grid(data.radial_factors.view(), 2);
    let grid = genfmt_jas_path_energy_grid(GenfmtJasPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 4,
        initial_lambda_count: 4,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_grid_branch_input(radial_factors.view()),
    })?;

    let mut expected_traces = Array1::<Complex>::zeros(2);
    let mut expected_factors = Array1::<Complex>::zeros(2);
    let mut expected_decomposed = Array3::<Complex>::zeros((3, 3, 2).f());
    for energy in 0..2 {
        let point = genfmt_jas_path_energy_point(GenfmtJasPathEnergyPointInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 4,
            initial_lambda_count: 4,
            energy_index: energy,
            path_potential_indices: common.path_potential_indices.view(),
            angular_limits: common.angular_limits.view(),
            phase_shifts: common.phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: common.leg_lengths.view(),
            complex_momentum: complex_momenta[energy],
            wave_number: wave_numbers[energy],
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: common.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: common.xnlm.view(),
            eta_angles: common.eta_angles.view(),
            branch: data
                .energy_point_branch_input_with_radial(radial_factors.index_axis(Axis(0), energy)),
        })?;
        expected_factors[energy] = point.path_factor.expect("cfac").factor;
        let trace = point.path_trace.expect("active JAS trace");
        expected_traces[energy] = genfmt_jas_trace_value(&trace);
        let decomposed = genfmt_jas_decomposed_trace_view(&trace).expect("pgtrl traces");
        for row in 0..3 {
            for column in 0..3 {
                expected_decomposed[(row, column, energy)] = decomposed[(row, column)];
            }
        }
    }

    assert_eq!(grid.active.to_vec(), vec![true, true]);
    assert_complex_array1_close(&grid.path_traces, &expected_traces);
    assert_complex_array1_close(&grid.path_factors, &expected_factors);
    assert_complex_array3_close(
        grid.decomposed_traces.as_ref().expect("pgtrl traces"),
        &expected_decomposed,
    );
    assert_complex_close(
        grid.signals.chi[0],
        Complex::new(7_054.280_021_153_908, -2_867.446_050_766_692_7),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_energy_grid_rejects_invalid_inputs() {
    let data = mmtrxijas_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(5, 1);
    let radial_factors = Array3::zeros((0, 3, 4).f());

    assert!(matches!(
        genfmt_jas_path_energy_grid(GenfmtJasPathEnergyGridInput {
            m_indices: data.m_indices.view(),
            n_indices: data.n_indices.view(),
            full_lambda_count: 5,
            initial_lambda_count: 5,
            path_potential_indices: common.path_potential_indices.view(),
            angular_limits: common.angular_limits.view(),
            phase_shifts: common.phase_shifts.view(),
            signed_angular_offset: 4,
            leg_lengths: common.leg_lengths.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: wave_numbers.view(),
            momentum_zero_epsilon: 1.0e-16,
            max_m_plus_one: 3,
            max_n: 1,
            rotations: common.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: common.xnlm.view(),
            eta_angles: common.eta_angles.view(),
            branch: data.energy_grid_branch_input(radial_factors.view()),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "radial_factors",
            axis: "energy",
            length: 0,
            required: 1,
        })
    ));
}

#[test]
fn genfmt_jas_path_energy_grid_finalization_matches_post_loop_reference() -> Result<(), GenfmtError>
{
    let data = mmtrxijas_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(5, 3);
    let radial_factors = jas_radial_factor_energy_grid(data.radial_factors.view(), 3);
    let energy_grid = genfmt_jas_path_energy_grid(GenfmtJasPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_grid_branch_input(radial_factors.view()),
    })?;
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.20, 0.30]);
    let effective_half_path_length = common.leg_lengths.iter().sum::<Real>() / 2.0;

    let finalization =
        genfmt_jas_path_energy_grid_finalization(GenfmtJasPathEnergyGridFinalizationInput {
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            energy_grid: &energy_grid,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.75,
            current_normalization: -1.0,
            effective_half_path_length_bohr: effective_half_path_length,
            potential_indices: common.path_potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: common.eta_angles.view(),
            leg_lengths: common.leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        })?;
    let expected = genfmt_jas_path_finalization(GenfmtJasPathFinalizationInput {
        path_index: 23,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        path_traces: energy_grid.path_traces.view(),
        path_factors: energy_grid.path_factors.view(),
        active: energy_grid.active.view(),
        decomposed_traces: energy_grid
            .decomposed_traces
            .as_ref()
            .map(|traces| traces.view()),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 1.75,
        current_normalization: -1.0,
        effective_half_path_length_bohr: effective_half_path_length,
        potential_indices: common.path_potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: common.eta_angles.view(),
        leg_lengths: common.leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(finalization, expected);
    assert!(finalization.output_decision.retention.keep);
    assert!(finalization.decomposed_output.is_some());
    Ok(())
}

#[test]
fn genfmt_jas_path_evaluation_composes_energy_grid_and_finalization() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(5, 3);
    let radial_factors = jas_radial_factor_energy_grid(data.radial_factors.view(), 3);
    let energy_input = GenfmtJasPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_grid_branch_input(radial_factors.view()),
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.20, 0.30]);
    let effective_half_path_length = common.leg_lengths.iter().sum::<Real>() / 2.0;

    let evaluation = genfmt_jas_path_evaluation(GenfmtJasPathEvaluationInput {
        energy_grid: energy_input,
        path_index: 23,
        print_level: 1,
        curved_wave_criterion_percent: 120.0,
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 1.75,
        current_normalization: -1.0,
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        phase_epsilon: 1.0e-16,
    })?;

    let expected_grid = genfmt_jas_path_energy_grid(energy_input)?;
    let expected_finalization =
        genfmt_jas_path_energy_grid_finalization(GenfmtJasPathEnergyGridFinalizationInput {
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            energy_grid: &expected_grid,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.75,
            current_normalization: -1.0,
            effective_half_path_length_bohr: effective_half_path_length,
            potential_indices: common.path_potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: common.eta_angles.view(),
            leg_lengths: common.leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        })?;

    assert_eq!(evaluation.energy_grid, expected_grid);
    assert_eq!(evaluation.finalization, expected_finalization);
    Ok(())
}

#[test]
fn genfmt_jas_path_sequence_threads_normalization_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let (common, complex_momenta, wave_numbers) = genfmt_jas_energy_grid_common_data(5, 3);
    let radial_factors = jas_radial_factor_energy_grid(data.radial_factors.view(), 3);
    let energy_input = GenfmtJasPathEnergyGridInput {
        m_indices: data.m_indices.view(),
        n_indices: data.n_indices.view(),
        full_lambda_count: 5,
        initial_lambda_count: 5,
        path_potential_indices: common.path_potential_indices.view(),
        angular_limits: common.angular_limits.view(),
        phase_shifts: common.phase_shifts.view(),
        signed_angular_offset: 4,
        leg_lengths: common.leg_lengths.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
        momentum_zero_epsilon: 1.0e-16,
        max_m_plus_one: 3,
        max_n: 1,
        rotations: common.rotations.view(),
        rotation_magnetic_offset: 4,
        xnlm: common.xnlm.view(),
        eta_angles: common.eta_angles.view(),
        branch: data.energy_grid_branch_input(radial_factors.view()),
    };
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let positions = arr2(&[[0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.20, 0.30]);
    let path_inputs = [
        GenfmtJasPathEvaluationInput {
            energy_grid: energy_input,
            path_index: 23,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 1.75,
            current_normalization: 999.0,
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            phase_epsilon: 1.0e-16,
        },
        GenfmtJasPathEvaluationInput {
            energy_grid: energy_input,
            path_index: 24,
            print_level: 1,
            curved_wave_criterion_percent: 120.0,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 0.90,
            current_normalization: 999.0,
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            phase_epsilon: 1.0e-16,
        },
    ];

    let sequence = genfmt_jas_path_sequence(GenfmtJasPathSequenceInput {
        path_inputs: &path_inputs,
        initial_normalization: -1.0,
    })?;

    let mut expected_evaluations = Vec::new();
    let mut current_normalization = -1.0;
    for path_input in path_inputs {
        let mut path_input = path_input;
        path_input.current_normalization = current_normalization;
        let evaluation = genfmt_jas_path_evaluation(path_input)?;
        current_normalization = evaluation
            .finalization
            .output_decision
            .importance
            .normalization;
        expected_evaluations.push(evaluation);
    }
    let expected_finalizations = expected_evaluations
        .iter()
        .map(|path| path.finalization.clone())
        .collect::<Vec<_>>();
    let expected_outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &expected_finalizations,
    })?;

    assert_eq!(sequence.evaluations, expected_evaluations);
    assert_eq!(sequence.outputs, expected_outputs);
    assert_eq!(sequence.outputs.examined_path_count, 2);
    assert_eq!(sequence.outputs.retained_path_count, 2);
    assert!(sequence.outputs.decomposed_paths.is_some());
    assert_eq!(
        sequence.evaluations[0]
            .finalization
            .output_decision
            .importance
            .normalization,
        sequence.evaluations[1]
            .finalization
            .output_decision
            .importance
            .normalization
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_traces_reject_invalid_inputs() {
    let path_product = genfmt_jas_path_product();
    let left_amplitudes = genfmt_jas_left_amplitudes();
    let right_amplitudes = genfmt_jas_right_amplitudes();
    let decomposed_left = genfmt_jas_decomposed_left_amplitudes();
    assert_eq!(
        genfmt_jas_left_right_path_trace(GenfmtJasLeftRightPathTraceInput {
            path_product: path_product.view(),
            left_amplitudes: left_amplitudes.view(),
            right_amplitudes: right_amplitudes.view(),
            lambda_count: 2,
            decomposed_left_amplitudes: Some(decomposed_left.view()),
            decomposed_right_amplitudes: None,
        }),
        Err(GenfmtError::MismatchedJasDecompositionTables)
    );

    let mut bad_right = right_amplitudes.clone();
    bad_right[(1, 1, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_jas_left_right_path_trace(GenfmtJasLeftRightPathTraceInput {
            path_product: path_product.view(),
            left_amplitudes: left_amplitudes.view(),
            right_amplitudes: bad_right.view(),
            lambda_count: 2,
            decomposed_left_amplitudes: None,
            decomposed_right_amplitudes: None,
        }),
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: "right_amplitudes",
            i0: 1,
            i1: 1,
            i2: 0,
            ..
        })
    ));

    let amplitudes = genfmt_jas_spherical_amplitudes();
    let short_product = Array2::zeros((1, 2).f());
    assert!(matches!(
        genfmt_jas_spherical_path_trace(GenfmtJasSphericalPathTraceInput {
            path_product: short_product.view(),
            amplitudes: amplitudes.view(),
            lambda_count: 2,
            decomposed_amplitudes: None,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_product",
            axis: "lambda",
            length: 1,
            required: 2,
        })
    ));

    let mut bad_decomposed = genfmt_jas_spherical_decomposed_amplitudes();
    bad_decomposed[(0, 1, 1, 0, 1)] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        genfmt_jas_spherical_path_trace(GenfmtJasSphericalPathTraceInput {
            path_product: path_product.view(),
            amplitudes: amplitudes.view(),
            lambda_count: 2,
            decomposed_amplitudes: Some(bad_decomposed.view()),
        }),
        Err(GenfmtError::NonFiniteTensor5Complex {
            table: "decomposed_amplitudes",
            i0: 0,
            i1: 1,
            i2: 1,
            i3: 0,
            i4: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_spin_channel_count_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    assert_eq!(
        genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
            spin_selector: 1,
            available_spin_channels: 2,
        })?,
        2
    );
    assert_eq!(
        genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
            spin_selector: 0,
            available_spin_channels: 2,
        })?,
        1
    );
    assert_eq!(
        genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
            spin_selector: 2,
            available_spin_channels: 2,
        })?,
        1
    );
    Ok(())
}

#[test]
fn genfmt_spin_channel_count_rejects_invalid_inputs() {
    assert_eq!(
        genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
            spin_selector: 1,
            available_spin_channels: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_spin_channel_count(GenfmtSpinChannelCountInput {
            spin_selector: 1,
            available_spin_channels: 3,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: 3,
        })
    );
}

#[test]
fn genfmt_jas_spin_selection_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 1,
            available_spin_channels: 2,
        })?
        .spin_index,
        1
    );
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 2,
            available_spin_channels: 2,
        })?
        .spin_index,
        0
    );
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 0,
            available_spin_channels: 2,
        })?
        .spin_index,
        0
    );
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 1,
            available_spin_channels: 1,
        })?
        .spin_index,
        0
    );
    Ok(())
}

#[test]
fn genfmt_jas_spin_selection_rejects_invalid_inputs() {
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 1,
            available_spin_channels: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_jas_spin_selection(GenfmtJasSpinSelectionInput {
            spin_selector: 1,
            available_spin_channels: 3,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: 3,
        })
    );
}

#[test]
fn genfmt_spin_reference_energies_match_genfmtsub_reference() -> Result<(), GenfmtError> {
    let references = genfmt_spin_reference_table();

    let averaged = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
        spin_reference_energies: references.view(),
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    assert_array_complex_close(
        &averaged.reference_energies,
        &[
            Complex::new(0.20, 0.30),
            Complex::new(-0.15, 0.20),
            Complex::new(0.60, -0.15),
        ],
    );

    let single = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
        spin_reference_energies: references.view(),
        spin_channel_count: 1,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    assert_array_complex_close(
        &single.reference_energies,
        &[
            Complex::new(0.10, 0.20),
            Complex::new(-0.20, 0.10),
            Complex::new(0.50, -0.20),
        ],
    );

    let spin_two = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
        spin_reference_energies: references.view(),
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 1 },
    })?;
    assert_array_complex_close(
        &spin_two.reference_energies,
        &[
            Complex::new(0.30, 0.40),
            Complex::new(-0.10, 0.30),
            Complex::new(0.70, -0.10),
        ],
    );
    Ok(())
}

#[test]
fn genfmt_spin_reference_energies_reject_invalid_inputs() {
    let empty = Array2::from_shape_vec((0, 2).f(), Vec::<Complex>::new()).expect("empty table");
    assert_eq!(
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: empty.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energy_count",
            value: 0,
        })
    );

    let references = genfmt_spin_reference_table();
    let short = Array2::from_elem((3, 1).f(), Complex::new(0.0, 0.0));
    assert_eq!(
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: short.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "spin_reference_energies",
            axis: "spin",
            length: 1,
            required: 2,
        })
    );
    assert_eq!(
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: references.view(),
            spin_channel_count: 0,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channel_count",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: references.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 2 },
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: 2,
        })
    );

    let mut bad = references.clone();
    bad[(1, 1)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: bad.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "spin_reference_energies",
            row: 1,
            column: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_spin_phase_shifts_match_genfmtsub_reference() -> Result<(), GenfmtError> {
    let phase_shifts = genfmt_spin_phase_shift_table();
    let angular_limits = genfmt_spin_angular_limits();

    let averaged = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;

    assert_eq!(averaged.phase_shifts.shape(), &[2, 5, 2]);
    assert_complex_close(averaged.phase_shifts[(0, 1, 0)], Complex::new(10.5, 4.0));
    assert_complex_close(averaged.phase_shifts[(0, 0, 1)], Complex::new(0.6, -1.1));
    assert_complex_close(averaged.phase_shifts[(0, 0, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(averaged.phase_shifts[(1, 1, 0)], Complex::new(0.0, 0.0));

    let spin_two = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 1 },
    })?;
    assert_complex_close(spin_two.phase_shifts[(1, 3, 1)], Complex::new(131.1, -37.1));

    let single = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        spin_channel_count: 1,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    assert_complex_close(single.phase_shifts[(0, 4, 1)], Complex::new(40.1, 19.9));
    Ok(())
}

#[test]
fn genfmt_spin_phase_shifts_reject_invalid_inputs() {
    let phase_shifts = genfmt_spin_phase_shift_table();
    let angular_limits = genfmt_spin_angular_limits();
    let empty =
        Array4::from_shape_vec((0, 5, 2, 2).f(), Vec::<Complex>::new()).expect("empty table");
    assert_eq!(
        genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: empty.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energy_count",
            value: 0,
        })
    );

    let short_spin = Array4::from_elem((2, 5, 1, 2).f(), Complex::new(0.0, 0.0));
    assert_eq!(
        genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: short_spin.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "spin_phase_shifts",
            axis: "spin",
            length: 1,
            required: 2,
        })
    );

    assert_eq!(
        genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: phase_shifts.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 1,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "signed_angular_offset",
            value: 1,
        })
    );

    assert_eq!(
        genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: phase_shifts.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 2 },
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: 2,
        })
    );

    let mut bad = phase_shifts.clone();
    bad[(0, 1, 1, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
            spin_phase_shifts: bad.view(),
            angular_limits: angular_limits.view(),
            signed_angular_offset: 2,
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        }),
        Err(GenfmtError::NonFiniteTensorComplex {
            table: "spin_phase_shifts",
            i0: 0,
            i1: 1,
            i2: 1,
            i3: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_central_phase_shifts_match_genfmt_header_reference() -> Result<(), GenfmtError> {
    let phase_shifts = genfmt_central_phase_shift_table();

    let negative_kappa = genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
        phase_shifts: phase_shifts.view(),
        signed_angular_offset: 3,
        initial_orbital_l: 1,
        initial_kappa: -2,
    })?;
    assert_eq!(negative_kappa.signed_angular_momentum, -2);
    assert_complex_close(negative_kappa.phase_shifts[0], phase_shifts[(0, 1, 0)]);
    assert_complex_close(negative_kappa.phase_shifts[1], phase_shifts[(1, 1, 0)]);
    assert_complex_close(negative_kappa.phase_shifts[2], phase_shifts[(2, 1, 0)]);

    let positive_kappa = genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
        phase_shifts: phase_shifts.view(),
        signed_angular_offset: 3,
        initial_orbital_l: 1,
        initial_kappa: 1,
    })?;
    assert_eq!(positive_kappa.signed_angular_momentum, 2);
    assert_complex_close(positive_kappa.phase_shifts[0], phase_shifts[(0, 5, 0)]);
    assert_complex_close(positive_kappa.phase_shifts[2], phase_shifts[(2, 5, 0)]);
    Ok(())
}

#[test]
fn genfmt_central_phase_shifts_reject_invalid_inputs() {
    let phase_shifts = genfmt_central_phase_shift_table();

    assert_eq!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: phase_shifts.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 1,
            initial_kappa: 0,
        }),
        Err(GenfmtError::InvalidInitialKappa { kappa: 0 })
    );

    let empty = Array3::from_shape_vec((0, 7, 1).f(), Vec::<Complex>::new()).expect("empty table");
    assert_eq!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: empty.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 1,
            initial_kappa: -2,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energy_count",
            value: 0,
        })
    );

    let no_potentials =
        Array3::from_shape_vec((2, 7, 0).f(), Vec::<Complex>::new()).expect("empty potential axis");
    assert_eq!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: no_potentials.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 1,
            initial_kappa: -2,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "phase_shifts",
            axis: "potential",
            length: 0,
            required: 1,
        })
    );

    assert_eq!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: phase_shifts.view(),
            signed_angular_offset: 1,
            initial_orbital_l: 2,
            initial_kappa: -3,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "signed_angular_offset",
            value: 1,
        })
    );

    assert_eq!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: phase_shifts.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 4,
            initial_kappa: 4,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "phase_shifts",
            axis: "signed_angular_momentum",
            length: 7,
            required: 9,
        })
    );

    let mut bad = phase_shifts.clone();
    bad[(1, 1, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: bad.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 1,
            initial_kappa: -2,
        }),
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: "phase_shifts",
            i0: 1,
            i1: 1,
            i2: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_spin_radial_factors_match_genfmtsub_reference() -> Result<(), GenfmtError> {
    let radial = genfmt_spin_radial_factor_table();

    let spin_two = genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
        spin_radial_factors: radial.view(),
        spin_channel_count: 2,
        spin_index: 1,
    })?;
    assert_eq!(spin_two.radial_factors.shape(), &[2, 3]);
    assert_complex_close(spin_two.radial_factors[(0, 0)], Complex::new(1.0, -2.0));
    assert_complex_close(spin_two.radial_factors[(1, 2)], Complex::new(121.0, -42.0));

    let single = genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
        spin_radial_factors: radial.view(),
        spin_channel_count: 1,
        spin_index: 0,
    })?;
    assert_complex_close(single.radial_factors[(1, 2)], Complex::new(120.0, -40.0));
    Ok(())
}

#[test]
fn genfmt_spin_radial_factors_reject_invalid_inputs() {
    let radial = genfmt_spin_radial_factor_table();
    let empty_energy =
        Array3::from_shape_vec((0, 3, 2).f(), Vec::<Complex>::new()).expect("empty table");
    assert_eq!(
        genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: empty_energy.view(),
            spin_channel_count: 2,
            spin_index: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energy_count",
            value: 0,
        })
    );

    let empty_transition = Array3::zeros((2, 0, 2).f());
    assert_eq!(
        genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: empty_transition.view(),
            spin_channel_count: 2,
            spin_index: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "transition_count",
            value: 0,
        })
    );

    let short_spin = Array3::zeros((2, 3, 1).f());
    assert_eq!(
        genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: short_spin.view(),
            spin_channel_count: 2,
            spin_index: 0,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "spin_radial_factors",
            axis: "spin",
            length: 1,
            required: 2,
        })
    );

    assert_eq!(
        genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: radial.view(),
            spin_channel_count: 2,
            spin_index: 2,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: 2,
        })
    );

    let mut bad = radial.clone();
    bad[(1, 2, 1)] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        genfmt_spin_radial_factors(GenfmtSpinRadialFactorInput {
            spin_radial_factors: bad.view(),
            spin_channel_count: 2,
            spin_index: 1,
        }),
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: "spin_radial_factors",
            i0: 1,
            i1: 2,
            i2: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_jas_spin_radial_factors_match_genfmtjas_reference() -> Result<(), GenfmtError> {
    let radial = genfmt_jas_spin_radial_factor_table();

    let spin_two = genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
        spin_radial_factors: radial.view(),
        spin_index: 1,
    })?;
    assert_eq!(spin_two.radial_factors.shape(), &[2, 2, 3]);
    assert_complex_close(
        spin_two.radial_factors[(0, 1, 2)],
        Complex::new(121.0, 58.0),
    );
    assert_complex_close(
        spin_two.radial_factors[(1, 0, 1)],
        Complex::new(1011.0, -497.0),
    );

    let spin_one = genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
        spin_radial_factors: radial.view(),
        spin_index: 0,
    })?;
    assert_complex_close(
        spin_one.radial_factors[(1, 0, 1)],
        Complex::new(1010.0, -495.0),
    );
    Ok(())
}

#[test]
fn genfmt_jas_spin_radial_factors_reject_invalid_inputs() {
    let radial = genfmt_jas_spin_radial_factor_table();
    let empty_energy = Array4::zeros((0, 2, 3, 2).f());
    assert_eq!(
        genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
            spin_radial_factors: empty_energy.view(),
            spin_index: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energy_count",
            value: 0,
        })
    );

    let empty_q = Array4::zeros((2, 0, 3, 2).f());
    assert_eq!(
        genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
            spin_radial_factors: empty_q.view(),
            spin_index: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "q_count",
            value: 0,
        })
    );

    let empty_transition = Array4::zeros((2, 2, 0, 2).f());
    assert_eq!(
        genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
            spin_radial_factors: empty_transition.view(),
            spin_index: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "transition_count",
            value: 0,
        })
    );

    let short_spin = Array4::zeros((2, 2, 3, 1).f());
    assert_eq!(
        genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
            spin_radial_factors: short_spin.view(),
            spin_index: 1,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "spin_radial_factors",
            axis: "spin",
            length: 1,
            required: 2,
        })
    );

    let mut bad = radial.clone();
    bad[(1, 1, 2, 1)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
            spin_radial_factors: bad.view(),
            spin_index: 1,
        }),
        Err(GenfmtError::NonFiniteTensorComplex {
            table: "spin_radial_factors",
            i0: 1,
            i1: 1,
            i2: 2,
            i3: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_jas_effective_initial_j_matches_regenf_reference() -> Result<(), GenfmtError> {
    let left_right = genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
        ellipticity: 0.0,
        initial_j2: 3,
        final_j2_max: 9,
    })?;
    assert_eq!(left_right.initial_j2, 3);
    assert!(!left_right.promoted_to_final_j2_max);

    let spherical = genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
        ellipticity: -1.0,
        initial_j2: 3,
        final_j2_max: 9,
    })?;
    assert_eq!(spherical.initial_j2, 9);
    assert!(spherical.promoted_to_final_j2_max);
    Ok(())
}

#[test]
fn genfmt_jas_effective_initial_j_rejects_invalid_inputs() {
    assert!(matches!(
        genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
            ellipticity: f64::NAN,
            initial_j2: 3,
            final_j2_max: 9,
        }),
        Err(GenfmtError::NonFiniteScalar {
            field: "ellipticity",
            ..
        })
    ));
    assert_eq!(
        genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
            ellipticity: 0.0,
            initial_j2: -1,
            final_j2_max: 9,
        }),
        Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jinit",
            value: -1,
        })
    );
    assert_eq!(
        genfmt_jas_effective_initial_j(GenfmtJasEffectiveInitialJInput {
            ellipticity: -1.0,
            initial_j2: 3,
            final_j2_max: -1,
        }),
        Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jmax",
            value: -1,
        })
    );
}

#[test]
fn genfmt_jas_transition_count_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    let count = genfmt_jas_transition_count(GenfmtJasTransitionCountInput {
        phase_transition_count: 4,
        requested_transition_count: 4,
    })?;
    assert_eq!(count.transition_count, 4);
    Ok(())
}

#[test]
fn genfmt_jas_transition_count_rejects_mismatch() {
    assert_eq!(
        genfmt_jas_transition_count(GenfmtJasTransitionCountInput {
            phase_transition_count: 3,
            requested_transition_count: 4,
        }),
        Err(GenfmtError::MismatchedJasTransitionCount {
            phase_transition_count: 3,
            requested_transition_count: 4,
        })
    );
}

#[test]
fn genfmt_momentum_grid_matches_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.25),
        Complex::new(0.045, 0.12),
        Complex::new(0.22, 0.0),
    ]);
    let reference_energies = Array1::from_vec(vec![
        Complex::new(0.10, 0.01),
        Complex::new(0.15, 0.02),
        Complex::new(0.30, 0.0),
    ]);

    let grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
        energies: energies.view(),
        reference_energies: reference_energies.view(),
        edge: 0.175,
    })?;

    assert_close(grid.wave_numbers[0], 0.632_455_532_033_675_9);
    assert_close(grid.wave_numbers[1], -0.509_901_951_359_278_5);
    assert_close(grid.wave_numbers[2], 0.3);
    assert_complex_close(grid.complex_momenta[0], Complex::new(0.8, 0.3));
    assert_complex_close(grid.complex_momenta[1], Complex::new(0.2, 0.5));
    assert_complex_close(grid.complex_momenta[2], Complex::new(0.0, 0.4));
    assert_close(grid.complex_momentum_magnitudes[0], 0.854_400_374_531_753_2);
    assert_close(grid.complex_momentum_magnitudes[1], 0.538_516_480_713_450_5);
    assert_close(grid.complex_momentum_magnitudes[2], 0.4);
    assert_eq!(grid.output_wave_numbers, grid.wave_numbers);
    Ok(())
}

#[test]
fn genfmt_ordinary_spin_momentum_grid_matches_genfmtsub_spin_loop_reference()
-> Result<(), GenfmtError> {
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.25),
        Complex::new(0.045, 0.12),
        Complex::new(0.22, 0.0),
    ]);
    let expected_complex_momenta = arr2(&[
        [Complex::new(0.80, 0.30), Complex::new(0.45, 0.20)],
        [Complex::new(0.20, 0.50), Complex::new(0.70, 0.10)],
        [Complex::new(0.05, 0.40), Complex::new(0.60, 0.35)],
    ]);
    let spin_reference_energies =
        genfmt_reference_energies_for_complex_momenta(&energies, &expected_complex_momenta);

    let grid = genfmt_ordinary_spin_momentum_grid(GenfmtOrdinarySpinMomentumGridInput {
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        edge: 0.175,
        spin_channel_count: 2,
    })?;

    assert_eq!(grid.complex_momenta.shape(), &[3, 2]);
    assert_eq!(grid.complex_momentum_magnitudes.shape(), &[3, 2]);
    for spin in 0..2 {
        let reference_energies = genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: spin_reference_energies.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: spin },
        })?;
        let expected_grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: energies.view(),
            reference_energies: reference_energies.reference_energies.view(),
            edge: 0.175,
        })?;
        assert_array_close(
            &grid.wave_numbers,
            expected_grid
                .wave_numbers
                .as_slice()
                .expect("contiguous fixture"),
        );
        for energy in 0..energies.len() {
            assert_complex_close(
                grid.complex_momenta[(energy, spin)],
                expected_grid.complex_momenta[energy],
            );
            assert_close(
                grid.complex_momentum_magnitudes[(energy, spin)],
                expected_grid.complex_momentum_magnitudes[energy],
            );
        }
    }

    let short_spin_reference_energies =
        Array2::from_elem((energies.len(), 1).f(), Complex::new(0.0, 0.0));
    assert!(matches!(
        genfmt_ordinary_spin_momentum_grid(GenfmtOrdinarySpinMomentumGridInput {
            energies: energies.view(),
            spin_reference_energies: short_spin_reference_energies.view(),
            edge: 0.175,
            spin_channel_count: 2,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "spin_reference_energies",
            axis: "spin",
            length: 1,
            required: 2,
        })
    ));
    Ok(())
}

#[test]
fn genfmt_momentum_grid_rejects_invalid_inputs() {
    let empty = Array1::from_vec(Vec::<Complex>::new());
    let references = Array1::from_vec(vec![Complex::new(0.0, 0.0)]);
    assert_eq!(
        genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: empty.view(),
            reference_energies: references.view(),
            edge: 0.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "energies",
            value: 0,
        })
    );

    let energies = Array1::from_vec(vec![Complex::new(0.3, 0.1), Complex::new(0.4, 0.1)]);
    let short_references = Array1::from_vec(vec![Complex::new(0.0, 0.0)]);
    assert_eq!(
        genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: energies.view(),
            reference_energies: short_references.view(),
            edge: 0.0,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "reference_energies",
            axis: "energy",
            length: 1,
            required: 2,
        })
    );

    let bad_energies = Array1::from_vec(vec![Complex::new(f64::NAN, 0.0)]);
    assert!(matches!(
        genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: bad_energies.view(),
            reference_energies: references.view(),
            edge: 0.0,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "energies",
            row: 0,
            ..
        })
    ));

    let bad_references = Array1::from_vec(vec![Complex::new(0.0, f64::INFINITY)]);
    assert!(matches!(
        genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: references.view(),
            reference_energies: bad_references.view(),
            edge: 0.0,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "reference_energies",
            row: 0,
            ..
        })
    ));

    assert!(matches!(
        genfmt_momentum_grid(GenfmtMomentumGridInput {
            energies: references.view(),
            reference_energies: references.view(),
            edge: f64::NAN,
        }),
        Err(GenfmtError::NonFiniteScalar { field: "edge", .. })
    ));
}

#[test]
fn genfmt_feff_bin_header_matches_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let atomic_numbers = Array1::from_vec(vec![29, 8, 6]);
    let potential_labels = ["Cu1", " ", " C "];
    let central_phase_shifts = Array1::from_vec(vec![
        Complex::new(0.10, -0.01),
        Complex::new(0.20, -0.02),
        Complex::new(0.30, -0.03),
    ]);
    let complex_momenta = Array1::from_vec(vec![
        Complex::new(1.0, 0.1),
        Complex::new(1.1, 0.2),
        Complex::new(1.2, 0.3),
    ]);
    let wave_numbers = Array1::from_vec(vec![0.5, 0.6, 0.7]);

    let header = genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 1,
        order: 2,
        initial_angular_momentum: 1,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 9.1,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
        central_phase_shifts: central_phase_shifts.view(),
        complex_momenta: complex_momenta.view(),
        wave_numbers: wave_numbers.view(),
    })?;

    assert_eq!(header.version, " 9.6.4");
    assert_eq!(header.pad_width, 8);
    assert_eq!(header.core_hole, 1);
    assert_eq!(header.order, 2);
    assert_eq!(header.initial_angular_momentum, 1);
    assert_close(header.average_norman_radius, 1.25);
    assert_close(header.fermi_level, -0.4);
    assert_close(header.edge_energy, 9.1);
    assert_eq!(header.potentials.len(), 3);
    assert_eq!(header.potentials[0].label, "Cu1");
    assert_eq!(header.potentials[0].atomic_number, 29);
    assert_eq!(header.potentials[1].label, "O");
    assert_eq!(header.potentials[1].atomic_number, 8);
    assert_eq!(header.potentials[2].label, "C");
    assert_eq!(header.central_phase_shifts, central_phase_shifts);
    assert_eq!(header.complex_momenta, complex_momenta);
    assert_eq!(header.wave_numbers, wave_numbers);
    Ok(())
}

#[test]
fn genfmt_driver_setup_composes_common_header_branch() -> Result<(), GenfmtError> {
    let energies = Array1::from_vec(vec![
        Complex::new(0.375, 0.25),
        Complex::new(0.045, 0.12),
        Complex::new(0.22, 0.0),
    ]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
        [Complex::new(0.50, -0.20), Complex::new(0.70, -0.10)],
    ]);
    let angular_limits = arr2(&[[2, 1, 0], [2, 0, 1], [2, 1, 1]]);
    let mut spin_phase_shifts = Array4::zeros((3, 7, 2, 3).f());
    for energy in 0..3 {
        let ef = energy as Real;
        for signed_l in 0..7 {
            let lf = signed_l as Real;
            for spin in 0..2 {
                let sf = spin as Real;
                for potential in 0..3 {
                    let pf = potential as Real;
                    spin_phase_shifts[(energy, signed_l, spin, potential)] = Complex::new(
                        100.0 * ef + 10.0 * lf + sf + 0.1 * pf,
                        -50.0 * ef + 5.0 * lf - 2.0 * sf - 0.1 * pf,
                    );
                }
            }
        }
    }
    let atomic_numbers = Array1::from_vec(vec![29, 8, 6]);
    let potential_labels = ["Cu1", " ", " C "];
    let input = GenfmtDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 3,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    };

    let setup = genfmt_driver_setup(input)?;
    let expected_reference_energies =
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: spin_reference_energies.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::Header,
        })?;
    let expected_phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: spin_phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 3,
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::Header,
    })?;
    let expected_central_phase_shifts =
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: expected_phase_shifts.phase_shifts.view(),
            signed_angular_offset: 3,
            initial_orbital_l: 1,
            initial_kappa: -2,
        })?;
    let expected_spin_momentum_grid =
        genfmt_ordinary_spin_momentum_grid(GenfmtOrdinarySpinMomentumGridInput {
            energies: energies.view(),
            spin_reference_energies: spin_reference_energies.view(),
            edge: 0.175,
            spin_channel_count: 2,
        })?;
    let expected_momentum_grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
        energies: energies.view(),
        reference_energies: expected_reference_energies.reference_energies.view(),
        edge: 0.175,
    })?;
    let expected_header = genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        initial_angular_momentum: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
        central_phase_shifts: expected_central_phase_shifts.phase_shifts.view(),
        complex_momenta: expected_momentum_grid.complex_momenta.view(),
        wave_numbers: expected_momentum_grid.output_wave_numbers.view(),
    })?;

    assert_eq!(setup.spin_channel_count, 2);
    assert_eq!(setup.reference_energies, expected_reference_energies);
    assert_eq!(setup.phase_shifts, expected_phase_shifts);
    assert_eq!(setup.central_phase_shifts, expected_central_phase_shifts);
    assert_eq!(setup.spin_momentum_grid, expected_spin_momentum_grid);
    assert_eq!(setup.momentum_grid, expected_momentum_grid);
    assert_eq!(setup.header, expected_header);
    Ok(())
}

#[test]
fn genfmt_jas_driver_setup_selects_single_spin_reference() -> Result<(), GenfmtError> {
    let energies = Array1::from_vec(vec![Complex::new(0.375, 0.25), Complex::new(0.045, 0.12)]);
    let spin_reference_energies = arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
    ]);
    let angular_limits = arr2(&[[2, 1], [2, 1]]);
    let spin_phase_shifts = genfmt_spin_phase_shift_table();
    let spin_radial_factors = genfmt_jas_spin_radial_factor_table();
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu1", " "];
    let input = GenfmtJasDriverSetupInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        spin_selector: 1,
        available_spin_channels: 2,
        energies: energies.view(),
        spin_reference_energies: spin_reference_energies.view(),
        spin_phase_shifts: spin_phase_shifts.view(),
        spin_radial_factors: spin_radial_factors.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        initial_orbital_l: 1,
        initial_kappa: -2,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
    };

    let setup = genfmt_jas_driver_setup(input)?;
    let expected_reference_energies =
        genfmt_spin_reference_energies(GenfmtSpinReferenceEnergyInput {
            spin_reference_energies: spin_reference_energies.view(),
            spin_channel_count: 2,
            mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 1 },
        })?;
    let expected_phase_shifts = genfmt_spin_phase_shifts(GenfmtSpinPhaseShiftInput {
        spin_phase_shifts: spin_phase_shifts.view(),
        angular_limits: angular_limits.view(),
        signed_angular_offset: 2,
        spin_channel_count: 2,
        mode: GenfmtReferenceEnergyMode::SpinChannel { spin_index: 1 },
    })?;
    let expected_radial_factors = genfmt_jas_spin_radial_factors(GenfmtJasSpinRadialFactorInput {
        spin_radial_factors: spin_radial_factors.view(),
        spin_index: 1,
    })?;
    let expected_central_phase_shifts =
        genfmt_central_phase_shifts(GenfmtCentralPhaseShiftInput {
            phase_shifts: expected_phase_shifts.phase_shifts.view(),
            signed_angular_offset: 2,
            initial_orbital_l: 1,
            initial_kappa: -2,
        })?;
    let expected_momentum_grid = genfmt_momentum_grid(GenfmtMomentumGridInput {
        energies: energies.view(),
        reference_energies: expected_reference_energies.reference_energies.view(),
        edge: 0.175,
    })?;
    let expected_header = genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
        version: " 9.6.4 ",
        pad_width: 8,
        core_hole: 4,
        order: 2,
        initial_angular_momentum: 2,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 0.175,
        potential_labels: &potential_labels,
        atomic_numbers: atomic_numbers.view(),
        central_phase_shifts: expected_central_phase_shifts.phase_shifts.view(),
        complex_momenta: expected_momentum_grid.complex_momenta.view(),
        wave_numbers: expected_momentum_grid.output_wave_numbers.view(),
    })?;

    assert_eq!(setup.spin_selection.spin_index, 1);
    assert_eq!(setup.reference_energies, expected_reference_energies);
    assert_eq!(setup.phase_shifts, expected_phase_shifts);
    assert_eq!(setup.radial_factors, expected_radial_factors);
    assert_eq!(setup.central_phase_shifts, expected_central_phase_shifts);
    assert_eq!(setup.momentum_grid, expected_momentum_grid);
    assert_eq!(setup.header, expected_header);
    assert_complex_close(
        setup.reference_energies.reference_energies[0],
        Complex::new(0.30, 0.40),
    );
    assert_complex_close(
        setup.phase_shifts.phase_shifts[(0, 0, 0)],
        spin_phase_shifts[(0, 0, 1, 0)],
    );
    assert_complex_close(
        setup.radial_factors.radial_factors[(1, 0, 1)],
        spin_radial_factors[(1, 0, 1, 1)],
    );
    Ok(())
}

#[test]
fn genfmt_feff_bin_header_rejects_invalid_inputs() {
    let atomic_numbers = Array1::from_vec(vec![29, 8]);
    let potential_labels = ["Cu", "O"];
    let central_phase_shifts = Array1::from_vec(vec![Complex::new(0.10, -0.01)]);
    let complex_momenta = Array1::from_vec(vec![Complex::new(1.0, 0.1)]);
    let wave_numbers = Array1::from_vec(vec![0.5]);

    assert_eq!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "",
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &potential_labels,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: wave_numbers.view(),
        }),
        Err(GenfmtError::InvalidTextField { field: "version" })
    );

    assert_eq!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "refeff",
            pad_width: 2,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &potential_labels,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: wave_numbers.view(),
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "pad_width",
            value: 2,
        })
    );

    let short_labels = ["Cu"];
    assert_eq!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "refeff",
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &short_labels,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: wave_numbers.view(),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "potential_labels",
            axis: "potential",
            length: 1,
            required: 2,
        })
    );

    let bad_label = ["too-long", "O"];
    assert_eq!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "refeff",
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &bad_label,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: wave_numbers.view(),
        }),
        Err(GenfmtError::InvalidPotentialLabel { index: 0 })
    );

    let short_momenta = Array1::from_vec(Vec::<Complex>::new());
    assert_eq!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "refeff",
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &potential_labels,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: short_momenta.view(),
            wave_numbers: wave_numbers.view(),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "complex_momenta",
            axis: "energy",
            length: 0,
            required: 1,
        })
    );

    let bad_wave_numbers = Array1::from_vec(vec![f64::NAN]);
    assert!(matches!(
        genfmt_feff_bin_header(GenfmtFeffBinHeaderInput {
            version: "refeff",
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 1,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potential_labels: &potential_labels,
            atomic_numbers: atomic_numbers.view(),
            central_phase_shifts: central_phase_shifts.view(),
            complex_momenta: complex_momenta.view(),
            wave_numbers: bad_wave_numbers.view(),
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "wave_numbers",
            index: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_curved_wave_leg_limits_match_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let path_potentials = Array1::from_vec(vec![2, 1, 0]);
    let angular_limits = arr2(&[[1, 2, 0], [2, 0, 3]]);

    let limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: path_potentials.view(),
        angular_limits: angular_limits.view(),
        energy_index: 1,
        max_m_plus_one: 2,
        max_n: 1,
    })?;

    assert_eq!(limits.mixed_order_capacity, 3);
    assert_eq!(
        limits.limits,
        vec![
            GenfmtCurvedWaveLegLimit {
                previous_potential_index: 0,
                current_potential_index: 2,
                angular_count: 4,
                mixed_order_count: 3,
            },
            GenfmtCurvedWaveLegLimit {
                previous_potential_index: 2,
                current_potential_index: 1,
                angular_count: 4,
                mixed_order_count: 3,
            },
            GenfmtCurvedWaveLegLimit {
                previous_potential_index: 1,
                current_potential_index: 0,
                angular_count: 3,
                mixed_order_count: 3,
            },
        ]
    );
    Ok(())
}

#[test]
fn genfmt_curved_wave_leg_limits_clamp_mixed_order_reference() -> Result<(), GenfmtError> {
    let path_potentials = Array1::from_vec(vec![0, 2]);
    let angular_limits = arr2(&[[4, 0, 2]]);

    let limits = genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
        path_potential_indices: path_potentials.view(),
        angular_limits: angular_limits.view(),
        energy_index: 0,
        max_m_plus_one: 1,
        max_n: 1,
    })?;

    assert_eq!(limits.mixed_order_capacity, 2);
    assert_eq!(
        limits.limits,
        vec![
            GenfmtCurvedWaveLegLimit {
                previous_potential_index: 2,
                current_potential_index: 0,
                angular_count: 5,
                mixed_order_count: 2,
            },
            GenfmtCurvedWaveLegLimit {
                previous_potential_index: 0,
                current_potential_index: 2,
                angular_count: 5,
                mixed_order_count: 2,
            },
        ]
    );
    Ok(())
}

#[test]
fn genfmt_curved_wave_leg_limits_rejects_invalid_inputs() {
    let empty = Array1::from_vec(Vec::<usize>::new());
    let angular_limits = arr2(&[[1, 2, 0], [2, 0, 3]]);
    assert_eq!(
        genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
            path_potential_indices: empty.view(),
            angular_limits: angular_limits.view(),
            energy_index: 0,
            max_m_plus_one: 2,
            max_n: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_potential_indices",
            value: 0,
        })
    );

    let path_potentials = Array1::from_vec(vec![0, 2]);
    assert_eq!(
        genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
            path_potential_indices: path_potentials.view(),
            angular_limits: angular_limits.view(),
            energy_index: 2,
            max_m_plus_one: 2,
            max_n: 1,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "angular_limits",
            axis: "energy",
            length: 2,
            required: 3,
        })
    );

    let bad_potentials = Array1::from_vec(vec![4]);
    assert_eq!(
        genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
            path_potential_indices: bad_potentials.view(),
            angular_limits: angular_limits.view(),
            energy_index: 0,
            max_m_plus_one: 2,
            max_n: 1,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "angular_limits",
            axis: "potential",
            length: 3,
            required: 5,
        })
    );

    assert_eq!(
        genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
            path_potential_indices: path_potentials.view(),
            angular_limits: angular_limits.view(),
            energy_index: 0,
            max_m_plus_one: 0,
            max_n: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "max_m_plus_one",
            value: 0,
        })
    );

    let overflowing_limit = arr2(&[[usize::MAX]]);
    let absorber = Array1::from_vec(vec![0]);
    assert_eq!(
        genfmt_curved_wave_leg_limits(GenfmtCurvedWaveLegLimitsInput {
            path_potential_indices: absorber.view(),
            angular_limits: overflowing_limit.view(),
            energy_index: 0,
            max_m_plus_one: 2,
            max_n: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "angular_limit",
            value: usize::MAX,
        })
    );
}

#[test]
fn genfmt_curved_wave_polynomial_tables_match_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let leg_rhos = Array1::from_vec(vec![Complex::new(1.25, 0.4), Complex::new(-0.8, 1.1)]);
    let leg_limits = vec![
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 0,
            current_potential_index: 2,
            angular_count: 4,
            mixed_order_count: 3,
        },
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 2,
            current_potential_index: 1,
            angular_count: 2,
            mixed_order_count: 2,
        },
    ];

    let tables = genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
        leg_rhos: leg_rhos.view(),
        leg_limits: &leg_limits,
        mixed_order_capacity: 4,
    })?;

    assert_eq!(tables.tables.shape(), &[5, 4, 2]);
    assert_eq!(tables.tables.strides(), &[1, 5, 20]);
    assert_complex_close(
        tables.tables[(1, 0, 0)],
        Complex::new(1.232_220_609_579_100_2, 0.725_689_404_934_687_9),
    );
    assert_complex_close(
        tables.tables[(3, 1, 0)],
        Complex::new(-28.733_692_908_170_283, 2.550_923_127_350_68),
    );
    assert_complex_close(tables.tables[(1, 3, 0)], Complex::new(0.0, 0.0));
    assert_complex_close(tables.tables[(3, 0, 1)], Complex::new(0.0, 0.0));
    assert_complex_close(tables.tables[(0, 0, 1)], Complex::new(1.0, 0.0));
    assert_complex_close(
        tables.tables[(1, 0, 1)],
        Complex::new(1.594_594_594_594_594_5, -0.432_432_432_432_432_35),
    );
    Ok(())
}

#[test]
fn genfmt_curved_wave_polynomial_tables_reject_invalid_inputs() {
    let empty_rhos = Array1::from_vec(Vec::<Complex>::new());
    let limits = vec![GenfmtCurvedWaveLegLimit {
        previous_potential_index: 0,
        current_potential_index: 0,
        angular_count: 1,
        mixed_order_count: 1,
    }];
    assert_eq!(
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: empty_rhos.view(),
            leg_limits: &limits,
            mixed_order_capacity: 1,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_rhos",
            value: 0,
        })
    );

    let leg_rhos = Array1::from_vec(vec![Complex::new(1.0, 0.0)]);
    let two_limits = vec![
        limits[0],
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 0,
            current_potential_index: 1,
            angular_count: 1,
            mixed_order_count: 1,
        },
    ];
    assert_eq!(
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: leg_rhos.view(),
            leg_limits: &two_limits,
            mixed_order_capacity: 1,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "leg_rhos",
            axis: "leg",
            length: 1,
            required: 2,
        })
    );

    assert_eq!(
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: leg_rhos.view(),
            leg_limits: &limits,
            mixed_order_capacity: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "mixed_order_capacity",
            value: 0,
        })
    );

    let too_wide = vec![GenfmtCurvedWaveLegLimit {
        previous_potential_index: 0,
        current_potential_index: 0,
        angular_count: 2,
        mixed_order_count: 3,
    }];
    assert_eq!(
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: leg_rhos.view(),
            leg_limits: &too_wide,
            mixed_order_capacity: 2,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "curved_wave_polynomial_tables",
            axis: "mixed_order",
            length: 2,
            required: 3,
        })
    );

    let zero_rho = Array1::from_vec(vec![Complex::new(0.0, 0.0)]);
    assert_eq!(
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: zero_rho.view(),
            leg_limits: &limits,
            mixed_order_capacity: 1,
        }),
        Err(GenfmtError::ZeroComplex { field: "rho" })
    );
}

#[test]
fn genfmt_path_geometry_matches_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let leg_lengths = Array1::from_vec(vec![1.5, 2.0, 0.75]);

    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: leg_lengths.view(),
        complex_momentum: Complex::new(1.2, 0.4),
        momentum_zero_epsilon: 1.0e-16,
    })?;

    assert_array_complex_close(
        &geometry.leg_rhos,
        &[
            Complex::new(1.8, 0.6),
            Complex::new(2.4, 0.8),
            Complex::new(0.9, 0.3),
        ],
    );
    assert_close(geometry.effective_path_length, 2.125);
    assert!(geometry.active);
    Ok(())
}

#[test]
fn genfmt_path_geometry_flags_zero_momentum_reference() -> Result<(), GenfmtError> {
    let leg_lengths = Array1::from_vec(vec![1.5, 2.0]);

    let geometry = genfmt_path_geometry(GenfmtPathGeometryInput {
        leg_lengths: leg_lengths.view(),
        complex_momentum: Complex::new(1.0e-18, 0.0),
        momentum_zero_epsilon: 1.0e-16,
    })?;

    assert_array_complex_close(
        &geometry.leg_rhos,
        &[Complex::new(1.5e-18, 0.0), Complex::new(2.0e-18, 0.0)],
    );
    assert_close(geometry.effective_path_length, 1.75);
    assert!(!geometry.active);
    Ok(())
}

#[test]
fn genfmt_path_geometry_rejects_invalid_inputs() {
    let empty = Array1::from_vec(Vec::<Real>::new());
    assert_eq!(
        genfmt_path_geometry(GenfmtPathGeometryInput {
            leg_lengths: empty.view(),
            complex_momentum: Complex::new(1.0, 0.0),
            momentum_zero_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_lengths",
            value: 0,
        })
    );

    let leg_lengths = Array1::from_vec(vec![1.5, 2.0]);
    assert!(matches!(
        genfmt_path_geometry(GenfmtPathGeometryInput {
            leg_lengths: leg_lengths.view(),
            complex_momentum: Complex::new(f64::NAN, 0.0),
            momentum_zero_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NonFiniteComplex {
            field: "complex_momentum",
            ..
        })
    ));

    assert_eq!(
        genfmt_path_geometry(GenfmtPathGeometryInput {
            leg_lengths: leg_lengths.view(),
            complex_momentum: Complex::new(1.0, 0.0),
            momentum_zero_epsilon: -1.0,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "momentum_zero_epsilon",
            value: -1.0,
        })
    );

    let bad_lengths = Array1::from_vec(vec![1.5, f64::NAN]);
    assert!(matches!(
        genfmt_path_geometry(GenfmtPathGeometryInput {
            leg_lengths: bad_lengths.view(),
            complex_momentum: Complex::new(1.0, 0.0),
            momentum_zero_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "leg_lengths",
            index: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_path_signal_contribution_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let input = GenfmtPathSignalContributionInput {
        accumulated_chi: Complex::new(0.10, -0.20),
        path_trace: Complex::new(0.40, 0.30),
        path_factor: Complex::new(2.0, -0.5),
        spin_channel_count: 1,
        spin_index: 0,
    };

    let contribution = genfmt_path_signal_contribution(input)?;
    assert_complex_close(contribution.contribution, Complex::new(0.95, 0.40));
    assert_complex_close(contribution.accumulated_chi, Complex::new(1.05, 0.20));
    Ok(())
}

#[test]
fn genfmt_path_signal_contribution_applies_two_spin_sign_reference() -> Result<(), GenfmtError> {
    let first_spin = genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
        accumulated_chi: Complex::new(0.10, -0.20),
        path_trace: Complex::new(0.40, 0.30),
        path_factor: Complex::new(2.0, -0.5),
        spin_channel_count: 2,
        spin_index: 0,
    })?;

    assert_complex_close(first_spin.contribution, Complex::new(-0.95, -0.40));
    assert_complex_close(first_spin.accumulated_chi, Complex::new(-0.85, -0.60));

    let second_spin = genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
        accumulated_chi: first_spin.accumulated_chi,
        path_trace: Complex::new(0.40, 0.30),
        path_factor: Complex::new(2.0, -0.5),
        spin_channel_count: 2,
        spin_index: 1,
    })?;

    assert_complex_close(second_spin.contribution, Complex::new(0.95, 0.40));
    assert_complex_close(second_spin.accumulated_chi, Complex::new(0.10, -0.20));
    Ok(())
}

#[test]
fn genfmt_path_signal_contribution_rejects_invalid_inputs() {
    let input = GenfmtPathSignalContributionInput {
        accumulated_chi: Complex::new(0.10, -0.20),
        path_trace: Complex::new(0.40, 0.30),
        path_factor: Complex::new(2.0, -0.5),
        spin_channel_count: 1,
        spin_index: 0,
    };

    assert_eq!(
        genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
            spin_channel_count: 0,
            ..input
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channel_count",
            value: 0,
        })
    );
    assert_eq!(
        genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
            spin_channel_count: 2,
            spin_index: 2,
            ..input
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_index",
            value: 2,
        })
    );
    assert!(matches!(
        genfmt_path_signal_contribution(GenfmtPathSignalContributionInput {
            path_trace: Complex::new(f64::NAN, 0.0),
            ..input
        }),
        Err(GenfmtError::NonFiniteComplex {
            field: "path_trace",
            ..
        })
    ));
}

#[test]
fn genfmt_path_signals_match_genfmtsub_spin_loop_reference() -> Result<(), GenfmtError> {
    let path_traces = arr2(&[
        [
            Complex::new(0.4, 0.3),
            Complex::new(-0.2, 0.5),
            Complex::new(f64::NAN, f64::NAN),
        ],
        [
            Complex::new(0.1, -0.2),
            Complex::new(0.3, -0.4),
            Complex::new(f64::NAN, f64::NAN),
        ],
    ]);
    let path_factors = Array1::from_vec(vec![
        Complex::new(2.0, -0.5),
        Complex::new(0.25, 1.5),
        Complex::new(f64::NAN, f64::NAN),
    ]);
    let active = Array1::from_vec(vec![true, true, false]);

    let signals = genfmt_path_signals(GenfmtPathSignalsInput {
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        spin_channel_count: 2,
    })?;

    assert_eq!(signals.contributions.shape(), &[2, 3]);
    assert_eq!(signals.contributions.strides(), &[1, 2]);
    assert_complex_close(signals.contributions[(0, 0)], Complex::new(-0.95, -0.40));
    assert_complex_close(signals.contributions[(1, 0)], Complex::new(0.10, -0.45));
    assert_complex_close(signals.contributions[(0, 1)], Complex::new(0.80, 0.175));
    assert_complex_close(signals.contributions[(1, 1)], Complex::new(0.675, 0.35));
    assert_complex_close(signals.contributions[(0, 2)], Complex::new(0.0, 0.0));
    assert_complex_close(signals.contributions[(1, 2)], Complex::new(0.0, 0.0));

    assert_complex_close(signals.chi[0], Complex::new(-0.85, -0.85));
    assert_complex_close(signals.chi[1], Complex::new(1.475, 0.525));
    assert_complex_close(signals.chi[2], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn genfmt_path_signals_handles_single_spin_reference() -> Result<(), GenfmtError> {
    let path_traces = arr2(&[[Complex::new(0.4, 0.3), Complex::new(-0.2, 0.5)]]);
    let path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.25, 1.5)]);
    let active = Array1::from_vec(vec![true, false]);

    let signals = genfmt_path_signals(GenfmtPathSignalsInput {
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        spin_channel_count: 1,
    })?;

    assert_complex_close(signals.contributions[(0, 0)], Complex::new(0.95, 0.40));
    assert_complex_close(signals.contributions[(0, 1)], Complex::new(0.0, 0.0));
    assert_complex_close(signals.chi[0], Complex::new(0.95, 0.40));
    assert_complex_close(signals.chi[1], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn genfmt_path_signals_rejects_invalid_inputs() {
    let path_traces = arr2(&[[Complex::new(0.4, 0.3), Complex::new(-0.2, 0.5)]]);
    let path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.25, 1.5)]);
    let active = Array1::from_vec(vec![true, true]);

    assert_eq!(
        genfmt_path_signals(GenfmtPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: active.view(),
            spin_channel_count: 0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "spin_channel_count",
            value: 0,
        })
    );

    assert!(matches!(
        genfmt_path_signals(GenfmtPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: active.view(),
            spin_channel_count: 2,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_traces",
            axis: "spin",
            length: 1,
            required: 2,
        })
    ));

    let short_active = Array1::from_vec(vec![true]);
    assert!(matches!(
        genfmt_path_signals(GenfmtPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: short_active.view(),
            spin_channel_count: 1,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "active",
            axis: "energy",
            length: 1,
            required: 2,
        })
    ));

    let bad_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(f64::NAN, 0.0)]);
    assert!(matches!(
        genfmt_path_signals(GenfmtPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: bad_factors.view(),
            active: active.view(),
            spin_channel_count: 1,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "path_factors",
            row: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_ordinary_path_finalization_matches_genfmtsub_post_trace_reference()
-> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();
    let path_traces = arr2(&[[
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
        Complex::new(0.30, -0.40),
        Complex::new(0.55, 0.05),
    ]]);
    let path_factors = Array1::from_vec(vec![Complex::new(1.0, 0.0); 5]);
    let active = Array1::from_vec(vec![true; 5]);
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let finalization = genfmt_ordinary_path_finalization(GenfmtOrdinaryPathFinalizationInput {
        path_index: 17,
        print_level: 0,
        curved_wave_criterion_percent: 120.0,
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        spin_channel_count: 1,
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: 1.75,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(finalization.signals.contributions.shape(), &[1, 5]);
    assert_eq!(finalization.signals.chi, chi);
    assert_close(
        finalization.output_decision.importance.percent,
        85.326_445_362_965_63,
    );
    assert!(finalization.output_decision.retention.keep);

    let output = finalization
        .output_decision
        .retained_output
        .expect("retained output");
    assert_eq!(output.path_index, 17);
    assert_close(output.degeneracy, 3.25);
    assert_close(output.criterion_percent, 85.326_445_362_965_63);
    assert_eq!(output.potential_indices, potential_indices);
    assert_array_close(
        &output.amplitudes,
        &[
            0.223_606_797_749_979,
            0.364_005_494_464_025_86,
            0.474_341_649_025_256_9,
            0.5,
            0.552_268_050_859_363,
        ],
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_path_finalization_discards_below_threshold_reference() -> Result<(), GenfmtError>
{
    let path_traces = arr2(&[[
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
        Complex::new(0.30, -0.40),
        Complex::new(0.55, 0.05),
    ]]);
    let path_factors = Array1::from_vec(vec![Complex::new(1.0, 0.0); 5]);
    let active = Array1::from_vec(vec![true; 5]);
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let finalization = genfmt_ordinary_path_finalization(GenfmtOrdinaryPathFinalizationInput {
        path_index: 17,
        print_level: 0,
        curved_wave_criterion_percent: 150.0,
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        spin_channel_count: 1,
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: 1.75,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_close(
        finalization.output_decision.importance.percent,
        85.326_445_362_965_63,
    );
    assert!(!finalization.output_decision.retention.keep);
    assert_eq!(finalization.output_decision.retained_output, None);
    Ok(())
}

#[test]
fn genfmt_ordinary_path_finalization_rejects_signal_inputs() {
    let path_traces = arr2(&[[Complex::new(0.4, 0.3), Complex::new(-0.2, 0.5)]]);
    let path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.25, 1.5)]);
    let active = Array1::from_vec(vec![true, true]);
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30]);
    let potential_indices = Array1::from_vec(vec![1, 0]);
    let positions = arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1]);

    assert!(matches!(
        genfmt_ordinary_path_finalization(GenfmtOrdinaryPathFinalizationInput {
            path_index: 17,
            print_level: 1,
            curved_wave_criterion_percent: 150.0,
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: active.view(),
            spin_channel_count: 2,
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 2,
            degeneracy: 3.25,
            current_normalization: 1.75,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_traces",
            axis: "spin",
            length: 1,
            required: 2,
        })
    ));
}

#[test]
fn genfmt_ordinary_path_outputs_collect_retained_paths_reference() {
    let skipped = genfmt_ordinary_finalization_fixture(1, false, 1.25);
    let kept_first = genfmt_ordinary_finalization_fixture(2, true, 1.50);
    let kept_second = genfmt_ordinary_finalization_fixture(3, true, 2.00);

    let outputs = genfmt_ordinary_path_outputs(GenfmtOrdinaryPathOutputsInput {
        path_finalizations: &[skipped, kept_first, kept_second],
    });

    assert_eq!(outputs.examined_path_count, 3);
    assert_eq!(outputs.retained_path_count, 2);
    assert_eq!(outputs.final_normalization, Some(2.00));
    assert_eq!(outputs.path_summaries.len(), 3);
    assert_eq!(outputs.path_summaries[0].path_index, 1);
    assert!(!outputs.path_summaries[0].retained);
    assert_eq!(outputs.path_summaries[1].path_index, 2);
    assert!(outputs.path_summaries[1].retained);
    assert_eq!(outputs.retained_paths.len(), 2);
    assert_eq!(outputs.retained_paths[0].path_index, 2);
    assert_eq!(outputs.retained_paths[1].path_index, 3);
}

#[test]
fn genfmt_jas_path_signal_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    let decomposed_traces = arr2(&[
        [Complex::new(0.1, 0.2), Complex::new(-0.3, 0.4)],
        [Complex::new(0.5, -0.1), Complex::new(-0.2, -0.6)],
    ]);

    let signal = genfmt_jas_path_signal(GenfmtJasPathSignalInput {
        path_trace: Complex::new(0.4, 0.3),
        path_factor: Complex::new(2.0, -0.5),
        decomposed_traces: Some(decomposed_traces.view()),
    })?;

    assert_complex_close(signal.chi, Complex::new(0.95, 0.40));
    let decomposed = signal
        .decomposed_chi
        .as_ref()
        .expect("decomposed traces should produce pgtrl values");
    assert_eq!(decomposed.shape(), &[2, 2]);
    assert_eq!(decomposed.strides(), &[1, 2]);
    assert_complex_close(decomposed[(0, 0)], Complex::new(0.30, 0.35));
    assert_complex_close(decomposed[(0, 1)], Complex::new(-0.40, 0.95));
    assert_complex_close(decomposed[(1, 0)], Complex::new(0.95, -0.45));
    assert_complex_close(decomposed[(1, 1)], Complex::new(-0.70, -1.10));
    assert_complex_close(
        signal.decomposed_sum.expect("lgcchi"),
        Complex::new(0.15, -0.25),
    );
    Ok(())
}

#[test]
fn genfmt_jas_path_signal_allows_no_decomposition() -> Result<(), GenfmtError> {
    let signal = genfmt_jas_path_signal(GenfmtJasPathSignalInput {
        path_trace: Complex::new(-0.2, 0.5),
        path_factor: Complex::new(0.25, 1.5),
        decomposed_traces: None,
    })?;

    assert_complex_close(signal.chi, Complex::new(-0.80, -0.175));
    assert_eq!(signal.decomposed_chi, None);
    assert_eq!(signal.decomposed_sum, None);
    Ok(())
}

#[test]
fn genfmt_jas_path_signal_rejects_invalid_inputs() {
    let decomposed_traces = arr2(&[[Complex::new(0.1, 0.2)]]);

    assert!(matches!(
        genfmt_jas_path_signal(GenfmtJasPathSignalInput {
            path_trace: Complex::new(f64::NAN, 0.0),
            path_factor: Complex::new(1.0, 0.0),
            decomposed_traces: Some(decomposed_traces.view()),
        }),
        Err(GenfmtError::NonFiniteComplex {
            field: "path_trace",
            ..
        })
    ));

    let empty = Array2::zeros((0, 1).f());
    assert_eq!(
        genfmt_jas_path_signal(GenfmtJasPathSignalInput {
            path_trace: Complex::new(1.0, 0.0),
            path_factor: Complex::new(1.0, 0.0),
            decomposed_traces: Some(empty.view()),
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "decomposition_rows",
            value: 0,
        })
    );

    let bad_decomposed = arr2(&[[Complex::new(0.0, f64::NAN)]]);
    assert!(matches!(
        genfmt_jas_path_signal(GenfmtJasPathSignalInput {
            path_trace: Complex::new(1.0, 0.0),
            path_factor: Complex::new(1.0, 0.0),
            decomposed_traces: Some(bad_decomposed.view()),
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "decomposed_traces",
            row: 0,
            column: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_jas_path_signals_match_genfmtjas_energy_loop_reference() -> Result<(), GenfmtError> {
    let path_traces = Array1::from_vec(vec![
        Complex::new(0.4, 0.3),
        Complex::new(-0.2, 0.5),
        Complex::new(f64::NAN, f64::NAN),
    ]);
    let path_factors = Array1::from_vec(vec![
        Complex::new(2.0, -0.5),
        Complex::new(0.25, 1.5),
        Complex::new(f64::NAN, f64::NAN),
    ]);
    let active = Array1::from_vec(vec![true, true, false]);
    let mut decomposed_traces = Array3::<Complex>::zeros((2, 2, 3).f());
    decomposed_traces[(0, 0, 0)] = Complex::new(0.1, 0.2);
    decomposed_traces[(0, 1, 0)] = Complex::new(-0.3, 0.4);
    decomposed_traces[(1, 0, 0)] = Complex::new(0.5, -0.1);
    decomposed_traces[(1, 1, 0)] = Complex::new(-0.2, -0.6);
    decomposed_traces[(0, 0, 1)] = Complex::new(1.0, 0.0);
    decomposed_traces[(0, 1, 1)] = Complex::new(0.0, 1.0);
    decomposed_traces[(1, 0, 1)] = Complex::new(-1.0, 0.5);
    decomposed_traces[(1, 1, 1)] = Complex::new(0.2, -0.4);
    decomposed_traces[(0, 0, 2)] = Complex::new(f64::NAN, f64::NAN);

    let signals = genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        decomposed_traces: Some(decomposed_traces.view()),
    })?;

    assert_eq!(signals.chi.len(), 3);
    assert_complex_close(signals.chi[0], Complex::new(0.95, 0.40));
    assert_complex_close(signals.chi[1], Complex::new(-0.80, -0.175));
    assert_complex_close(signals.chi[2], Complex::new(0.0, 0.0));

    let decomposed = signals
        .decomposed_chi
        .as_ref()
        .expect("decomposed traces should produce pgtrl values");
    assert_eq!(decomposed.shape(), &[2, 2, 3]);
    assert_eq!(decomposed.strides(), &[1, 2, 4]);
    assert_complex_close(decomposed[(0, 0, 0)], Complex::new(0.30, 0.35));
    assert_complex_close(decomposed[(1, 1, 0)], Complex::new(-0.70, -1.10));
    assert_complex_close(decomposed[(0, 0, 1)], Complex::new(0.25, 1.50));
    assert_complex_close(decomposed[(0, 1, 1)], Complex::new(-1.50, 0.25));
    assert_complex_close(decomposed[(1, 0, 1)], Complex::new(-1.00, -1.375));
    assert_complex_close(decomposed[(1, 1, 1)], Complex::new(0.65, 0.20));
    assert_complex_close(decomposed[(0, 0, 2)], Complex::new(0.0, 0.0));

    let sums = signals.decomposed_sums.expect("lgcchi");
    assert_complex_close(sums[0], Complex::new(0.15, -0.25));
    assert_complex_close(sums[1], Complex::new(-1.60, 0.575));
    assert_complex_close(sums[2], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn genfmt_jas_path_signals_allows_no_decomposition() -> Result<(), GenfmtError> {
    let path_traces = Array1::from_vec(vec![Complex::new(0.4, 0.3), Complex::new(-0.2, 0.5)]);
    let path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.25, 1.5)]);
    let active = Array1::from_vec(vec![true, false]);

    let signals = genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        decomposed_traces: None,
    })?;

    assert_complex_close(signals.chi[0], Complex::new(0.95, 0.40));
    assert_complex_close(signals.chi[1], Complex::new(0.0, 0.0));
    assert_eq!(signals.decomposed_chi, None);
    assert_eq!(signals.decomposed_sums, None);
    Ok(())
}

#[test]
fn genfmt_jas_path_signals_rejects_invalid_inputs() {
    let path_traces = Array1::from_vec(vec![Complex::new(0.4, 0.3), Complex::new(-0.2, 0.5)]);
    let short_path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5)]);
    let active = Array1::from_vec(vec![true, true]);

    assert!(matches!(
        genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: short_path_factors.view(),
            active: active.view(),
            decomposed_traces: None,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "path_factors",
            axis: "energy",
            length: 1,
            required: 2,
        })
    ));

    let bad_path_factors =
        Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.0, f64::NAN)]);
    assert!(matches!(
        genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: bad_path_factors.view(),
            active: active.view(),
            decomposed_traces: None,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "path_factors",
            row: 1,
            ..
        })
    ));

    let short_decomposed = Array3::zeros((1, 1, 1).f());
    let path_factors = Array1::from_vec(vec![Complex::new(2.0, -0.5), Complex::new(0.25, 1.5)]);
    assert!(matches!(
        genfmt_jas_path_signals(GenfmtJasPathSignalsInput {
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: active.view(),
            decomposed_traces: Some(short_decomposed.view()),
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "decomposed_traces",
            axis: "energy",
            length: 1,
            required: 2,
        })
    ));
}

#[test]
fn genfmt_jas_path_finalization_keeps_decomposition_reference() -> Result<(), GenfmtError> {
    let path_traces = Array1::from_vec(vec![
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
    ]);
    let path_factors = Array1::from_vec(vec![Complex::new(1.0, 0.0); 3]);
    let active = Array1::from_vec(vec![true; 3]);
    let decomposed_traces = genfmt_decomposed_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let finalization = genfmt_jas_path_finalization(GenfmtJasPathFinalizationInput {
        path_index: 21,
        print_level: 0,
        curved_wave_criterion_percent: 120.0,
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        decomposed_traces: Some(decomposed_traces.view()),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 3.25,
        current_normalization: 0.0,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(finalization.signals.chi, path_traces);
    assert_close(finalization.output_decision.importance.percent, 100.0);
    assert!(finalization.output_decision.retention.keep);
    assert!(finalization.output_decision.retained_output.is_some());

    let decomposed_signal = finalization
        .signals
        .decomposed_chi
        .as_ref()
        .expect("decomposed signal");
    assert_eq!(decomposed_signal, &decomposed_traces);

    let decomposed_output = finalization
        .decomposed_output
        .expect("retained decomposition output");
    assert_eq!(decomposed_output.amplitudes.shape(), &[2, 2, 3]);
    assert_close(decomposed_output.amplitudes[(0, 0, 0)], 1.0);
    assert_close(decomposed_output.amplitudes[(0, 1, 1)], 1.1);
    assert_close(decomposed_output.amplitudes[(1, 1, 2)], 1.0);
    assert_close(decomposed_output.phases[(0, 1, 1)], 3.233_185_307_179_586_4);
    assert_close(decomposed_output.phases[(1, 1, 2)], 3.283_185_307_179_586_2);
    Ok(())
}

#[test]
fn genfmt_jas_path_finalization_discards_decomposition_output_reference() -> Result<(), GenfmtError>
{
    let path_traces = Array1::from_vec(vec![
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
    ]);
    let path_factors = Array1::from_vec(vec![Complex::new(1.0, 0.0); 3]);
    let active = Array1::from_vec(vec![true; 3]);
    let decomposed_traces = genfmt_decomposed_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let finalization = genfmt_jas_path_finalization(GenfmtJasPathFinalizationInput {
        path_index: 21,
        print_level: 0,
        curved_wave_criterion_percent: 180.0,
        path_traces: path_traces.view(),
        path_factors: path_factors.view(),
        active: active.view(),
        decomposed_traces: Some(decomposed_traces.view()),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 0,
        active_energy_count: 3,
        degeneracy: 3.25,
        current_normalization: 0.0,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_close(finalization.output_decision.importance.percent, 100.0);
    assert!(!finalization.output_decision.retention.keep);
    assert!(finalization.signals.decomposed_chi.is_some());
    assert_eq!(finalization.output_decision.retained_output, None);
    assert_eq!(finalization.decomposed_output, None);
    Ok(())
}

#[test]
fn genfmt_jas_path_finalization_rejects_signal_inputs() {
    let path_traces = Array1::from_vec(vec![
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
    ]);
    let path_factors = Array1::from_vec(vec![Complex::new(1.0, 0.0); 3]);
    let active = Array1::from_vec(vec![true; 3]);
    let short_decomposed = Array3::zeros((1, 1, 2).f());
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    assert!(matches!(
        genfmt_jas_path_finalization(GenfmtJasPathFinalizationInput {
            path_index: 21,
            print_level: 0,
            curved_wave_criterion_percent: 120.0,
            path_traces: path_traces.view(),
            path_factors: path_factors.view(),
            active: active.view(),
            decomposed_traces: Some(short_decomposed.view()),
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 0,
            active_energy_count: 3,
            degeneracy: 3.25,
            current_normalization: 0.0,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "decomposed_traces",
            axis: "energy",
            length: 2,
            required: 3,
        })
    ));
}

#[test]
fn genfmt_jas_path_outputs_collect_retained_decomposition_reference() -> Result<(), GenfmtError> {
    let retained = genfmt_jas_finalization_fixture(4, true, 1.50, true);
    let skipped = genfmt_jas_finalization_fixture(5, false, 2.00, true);

    let outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &[retained, skipped],
    })?;

    assert_eq!(outputs.examined_path_count, 2);
    assert_eq!(outputs.retained_path_count, 1);
    assert_eq!(outputs.final_normalization, Some(2.00));
    assert_eq!(outputs.path_summaries.len(), 2);
    assert_eq!(outputs.path_summaries[0].path_index, 4);
    assert!(outputs.path_summaries[0].retained);
    assert_eq!(outputs.path_summaries[1].path_index, 5);
    assert!(!outputs.path_summaries[1].retained);
    assert_eq!(outputs.retained_paths.len(), 1);
    assert_eq!(outputs.retained_paths[0].path_index, 4);
    let decomposed = outputs
        .decomposed_paths
        .expect("retained decomposition output");
    assert_eq!(decomposed.len(), 1);
    assert_close(decomposed[0].amplitudes[(0, 0, 0)], 4.0);
    Ok(())
}

#[test]
fn genfmt_jas_path_outputs_allow_no_decomposition_reference() -> Result<(), GenfmtError> {
    let retained = genfmt_jas_finalization_fixture(4, true, 1.50, false);
    let skipped = genfmt_jas_finalization_fixture(5, false, 2.00, false);

    let outputs = genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
        path_finalizations: &[retained, skipped],
    })?;

    assert_eq!(outputs.examined_path_count, 2);
    assert_eq!(outputs.retained_path_count, 1);
    assert_eq!(outputs.decomposed_paths, None);
    Ok(())
}

#[test]
fn genfmt_jas_path_outputs_reject_mixed_decomposition_reference() {
    let retained_decomposed = genfmt_jas_finalization_fixture(4, true, 1.50, true);
    let retained_total_only = genfmt_jas_finalization_fixture(5, true, 2.00, false);

    assert_eq!(
        genfmt_jas_path_outputs(GenfmtJasPathOutputsInput {
            path_finalizations: &[retained_decomposed, retained_total_only],
        }),
        Err(GenfmtError::MismatchedJasFinalizationDecomposition)
    );
}

#[test]
fn genfmt_curved_wave_path_factor_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let leg_rhos = Array1::from_vec(vec![
        Complex::new(1.2, 0.3),
        Complex::new(-0.4, 0.8),
        Complex::new(0.7, -0.2),
    ]);

    let factor = genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
        leg_rhos: leg_rhos.view(),
        wave_number: 1.15,
        effective_path_length: 2.4,
    })?;

    assert_complex_close(factor.rho_sum, Complex::new(1.5, 0.9));
    assert_complex_close(factor.rho_product, Complex::new(-0.336, 0.732));
    assert_complex_close(
        factor.factor,
        Complex::new(0.487_550_360_581_460_1, 0.130_776_156_281_947_6),
    );
    Ok(())
}

#[test]
fn genfmt_curved_wave_path_factor_rejects_invalid_inputs() {
    let empty = Array1::from_vec(Vec::<Complex>::new());
    assert_eq!(
        genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
            leg_rhos: empty.view(),
            wave_number: 1.0,
            effective_path_length: 2.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_rhos",
            value: 0,
        })
    );

    let zero_product = Array1::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
    assert_eq!(
        genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
            leg_rhos: zero_product.view(),
            wave_number: 1.0,
            effective_path_length: 2.0,
        }),
        Err(GenfmtError::ZeroComplex {
            field: "rho_product",
        })
    );

    let nonfinite_rhos = Array1::from_vec(vec![Complex::new(1.0, f64::NAN)]);
    assert!(matches!(
        genfmt_curved_wave_path_factor(GenfmtCurvedWavePathFactorInput {
            leg_rhos: nonfinite_rhos.view(),
            wave_number: 1.0,
            effective_path_length: 2.0,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "leg_rhos",
            row: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_path_importance_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);

    let importance = genfmt_path_importance(GenfmtPathImportanceInput {
        chi: chi.view(),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: 1.75,
    })?;

    assert_array_close(
        &importance.magnitudes,
        &[
            0.223_606_797_749_979,
            0.364_005_494_464_025_86,
            0.474_341_649_025_256_9,
            0.5,
            0.552_268_050_859_363,
        ],
    );
    assert_close(importance.raw_importance, 1.493_212_793_851_898_6);
    assert_close(importance.normalization, 1.75);
    assert_close(importance.percent, 85.326_445_362_965_63);
    Ok(())
}

#[test]
fn genfmt_path_importance_initializes_normalization_reference() -> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);

    let importance = genfmt_path_importance(GenfmtPathImportanceInput {
        chi: chi.view(),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: -1.0,
    })?;

    assert_close(importance.normalization, importance.raw_importance);
    assert_close(importance.percent, 100.0);
    Ok(())
}

#[test]
fn genfmt_path_importance_rejects_invalid_inputs() {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    assert!(matches!(
        genfmt_path_importance(GenfmtPathImportanceInput {
            chi: chi.view(),
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 4,
            active_energy_count: 5,
            degeneracy: 3.25,
            current_normalization: 1.75,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "active_energy_count",
            value: 5,
        })
    ));

    let zero_chi = Array1::from_vec(vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)]);
    let two_momenta = Array1::from_vec(vec![0.10, 0.30]);
    assert_eq!(
        genfmt_path_importance(GenfmtPathImportanceInput {
            chi: zero_chi.view(),
            momentum_magnitudes: two_momenta.view(),
            edge_start_index: 0,
            active_energy_count: 2,
            degeneracy: 1.0,
            current_normalization: 0.0,
        }),
        Err(GenfmtError::ZeroScalar {
            field: "path_importance_normalization",
        })
    );

    let mut bad_chi = chi.clone();
    bad_chi[2] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_path_importance(GenfmtPathImportanceInput {
            chi: bad_chi.view(),
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 1,
            active_energy_count: 5,
            degeneracy: 3.25,
            current_normalization: 1.75,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "chi",
            row: 2,
            ..
        })
    ));
}

#[test]
fn genfmt_path_retention_matches_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let below = genfmt_path_retention(GenfmtPathRetentionInput {
        print_level: 0,
        curved_wave_criterion_percent: 4.5,
        path_importance_percent: 2.999,
    })?;
    assert_close(below.discard_threshold_percent.expect("threshold"), 3.0);
    assert!(!below.keep);

    let at_threshold = genfmt_path_retention(GenfmtPathRetentionInput {
        print_level: -2,
        curved_wave_criterion_percent: 4.5,
        path_importance_percent: 3.0,
    })?;
    assert_close(
        at_threshold.discard_threshold_percent.expect("threshold"),
        3.0,
    );
    assert!(at_threshold.keep);

    let forced = genfmt_path_retention(GenfmtPathRetentionInput {
        print_level: 1,
        curved_wave_criterion_percent: 4.5,
        path_importance_percent: 0.0,
    })?;
    assert_eq!(forced.discard_threshold_percent, None);
    assert!(forced.keep);
    Ok(())
}

#[test]
fn genfmt_path_retention_rejects_invalid_inputs() {
    assert_eq!(
        genfmt_path_retention(GenfmtPathRetentionInput {
            print_level: 0,
            curved_wave_criterion_percent: -1.0,
            path_importance_percent: 3.0,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "curved_wave_criterion_percent",
            value: -1.0,
        })
    );

    assert_eq!(
        genfmt_path_retention(GenfmtPathRetentionInput {
            print_level: 0,
            curved_wave_criterion_percent: 4.5,
            path_importance_percent: -0.01,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "path_importance_percent",
            value: -0.01,
        })
    );

    assert!(matches!(
        genfmt_path_retention(GenfmtPathRetentionInput {
            print_level: 0,
            curved_wave_criterion_percent: f64::NAN,
            path_importance_percent: 3.0,
        }),
        Err(GenfmtError::NonFiniteScalar {
            field: "curved_wave_criterion_percent",
            ..
        })
    ));
}

#[test]
fn genfmt_path_output_decision_keeps_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: 17,
        print_level: 0,
        curved_wave_criterion_percent: 120.0,
        chi: chi.view(),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: 1.75,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_close(decision.importance.raw_importance, 1.493_212_793_851_898_6);
    assert_close(decision.importance.normalization, 1.75);
    assert_close(decision.importance.percent, 85.326_445_362_965_63);
    assert_close(
        decision
            .retention
            .discard_threshold_percent
            .expect("threshold"),
        80.0,
    );
    assert!(decision.retention.keep);
    assert_eq!(decision.summary.path_index, 17);
    assert!(decision.summary.retained);
    assert_close(decision.summary.criterion_percent, 85.326_445_362_965_63);
    assert_close(decision.summary.degeneracy, 3.25);
    assert_eq!(decision.summary.leg_count, 3);
    assert_close(decision.summary.effective_half_path_length_bohr, 2.4);
    assert_close(
        decision.summary.effective_half_path_length_angstrom,
        2.4 * 0.529_177_249,
    );

    let output = decision.retained_output.expect("retained output");
    assert_eq!(output.path_index, 17);
    assert_close(output.degeneracy, 3.25);
    assert_close(output.criterion_percent, 85.326_445_362_965_63);
    assert_eq!(output.potential_indices, potential_indices);
    assert_eq!(output.positions, positions);
    assert_array_close(
        &output.amplitudes,
        &[
            0.223_606_797_749_979,
            0.364_005_494_464_025_86,
            0.474_341_649_025_256_9,
            0.5,
            0.552_268_050_859_363,
        ],
    );
    assert_close(output.phases[4], 6.373_845_194_380_332);
    Ok(())
}

#[test]
fn genfmt_path_output_decision_discards_below_threshold_reference() -> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    let decision = genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
        path_index: 17,
        print_level: 0,
        curved_wave_criterion_percent: 150.0,
        chi: chi.view(),
        momentum_magnitudes: momentum_magnitudes.view(),
        edge_start_index: 1,
        active_energy_count: 5,
        degeneracy: 3.25,
        current_normalization: 1.75,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_close(decision.importance.percent, 85.326_445_362_965_63);
    assert_close(
        decision
            .retention
            .discard_threshold_percent
            .expect("threshold"),
        100.0,
    );
    assert!(!decision.retention.keep);
    assert_eq!(decision.summary.path_index, 17);
    assert!(!decision.summary.retained);
    assert_eq!(decision.summary.leg_count, 3);
    assert_close(decision.summary.criterion_percent, 85.326_445_362_965_63);
    assert_eq!(decision.retained_output, None);
    Ok(())
}

#[test]
fn genfmt_path_output_decision_rejects_retained_output_inputs() {
    let chi = genfmt_reference_chi();
    let momentum_magnitudes = Array1::from_vec(vec![0.10, 0.30, 0.55, 0.90, 1.25]);
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);

    assert_eq!(
        genfmt_path_output_decision(GenfmtPathOutputDecisionInput {
            path_index: 0,
            print_level: 1,
            curved_wave_criterion_percent: 150.0,
            chi: chi.view(),
            momentum_magnitudes: momentum_magnitudes.view(),
            edge_start_index: 1,
            active_energy_count: 5,
            degeneracy: 3.25,
            current_normalization: 1.75,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_index",
            value: 0,
        })
    );
}

#[test]
fn genfmt_chi_amplitude_phase_matches_genfmtsub_reference() -> Result<(), GenfmtError> {
    let chi = genfmt_reference_chi();

    let output = genfmt_chi_amplitude_phase(GenfmtChiAmplitudePhaseInput {
        chi: chi.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_array_close(
        &output.amplitudes,
        &[
            0.223_606_797_749_979,
            0.364_005_494_464_025_86,
            0.474_341_649_025_256_9,
            0.5,
            0.552_268_050_859_363,
        ],
    );
    assert_array_close(
        &output.phases,
        &[
            0.463_647_609_000_806_1,
            1.849_095_985_800_008,
            3.463_343_207_986_435_2,
            5.355_890_089_177_974,
            6.373_845_194_380_332,
        ],
    );
    Ok(())
}

#[test]
fn genfmt_chi_amplitude_phase_unwraps_pijump_reference() -> Result<(), GenfmtError> {
    let chi = Array1::from_vec(vec![
        Complex::new(3.0_f64.cos(), 3.0_f64.sin()),
        Complex::new((-3.05_f64).cos(), (-3.05_f64).sin()),
        Complex::new((-2.9_f64).cos(), (-2.9_f64).sin()),
    ]);

    let output = genfmt_chi_amplitude_phase(GenfmtChiAmplitudePhaseInput {
        chi: chi.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_array_close(
        &output.phases,
        &[3.0, 3.233_185_307_179_586_4, 3.383_185_307_179_586_8],
    );
    Ok(())
}

#[test]
fn genfmt_chi_amplitude_phase_rejects_invalid_inputs() {
    let chi = genfmt_reference_chi();
    assert_eq!(
        genfmt_chi_amplitude_phase(GenfmtChiAmplitudePhaseInput {
            chi: chi.view(),
            phase_epsilon: -1.0,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "phase_epsilon",
            value: -1.0,
        })
    );

    let mut bad_chi = chi.clone();
    bad_chi[0] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        genfmt_chi_amplitude_phase(GenfmtChiAmplitudePhaseInput {
            chi: bad_chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "chi",
            row: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_retained_path_output_matches_genfmt_output_block_reference() -> Result<(), GenfmtError> {
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);
    let chi = genfmt_reference_chi();

    let output = genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
        path_index: 17,
        degeneracy: 4.0,
        criterion_percent: 12.5,
        effective_half_path_length_bohr: 2.4,
        potential_indices: potential_indices.view(),
        positions: positions.view(),
        beta_angles: beta_angles.view(),
        eta_angles: eta_angles.view(),
        leg_lengths: leg_lengths.view(),
        chi: chi.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(output.path_index, 17);
    assert_close(output.degeneracy, 4.0);
    assert_close(output.criterion_percent, 12.5);
    assert_close(output.effective_half_path_length_bohr, 2.4);
    assert_close(
        output.effective_half_path_length_angstrom,
        2.4 * 0.529_177_249,
    );
    assert_close(output.list_sigma2, 0.0);
    assert_eq!(output.potential_indices, potential_indices);
    assert_eq!(output.positions, positions);
    assert_eq!(output.beta_angles, beta_angles);
    assert_eq!(output.eta_angles, eta_angles);
    assert_eq!(output.leg_lengths, leg_lengths);
    assert_array_close(
        &output.amplitudes,
        &[
            0.223_606_797_749_979,
            0.364_005_494_464_025_86,
            0.474_341_649_025_256_9,
            0.5,
            0.552_268_050_859_363,
        ],
    );
    assert_array_close(
        &output.phases,
        &[
            0.463_647_609_000_806_1,
            1.849_095_985_800_008,
            3.463_343_207_986_435_2,
            5.355_890_089_177_974,
            6.373_845_194_380_332,
        ],
    );
    Ok(())
}

#[test]
fn genfmt_retained_path_output_rejects_invalid_inputs() {
    let potential_indices = Array1::from_vec(vec![1, 2, 0]);
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let beta_angles = Array1::from_vec(vec![0.10, 0.20, 0.30]);
    let eta_angles = Array1::from_vec(vec![0.40, 0.50, 0.60]);
    let leg_lengths = Array1::from_vec(vec![1.0, 1.1, 1.2]);
    let chi = genfmt_reference_chi();

    assert_eq!(
        genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
            path_index: 0,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            chi: chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_index",
            value: 0,
        })
    );

    let bad_positions = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
    assert_eq!(
        genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
            path_index: 17,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: bad_positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            chi: chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::InvalidPathCoordinateColumns { columns: 2 })
    );

    let short_beta_angles = Array1::from_vec(vec![0.10, 0.20]);
    assert_eq!(
        genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
            path_index: 17,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: short_beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            chi: chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "beta_angles",
            axis: "leg",
            length: 2,
            required: 3,
        })
    );

    let mut bad_leg_lengths = leg_lengths.clone();
    bad_leg_lengths[1] = -1.0;
    assert_eq!(
        genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
            path_index: 17,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: bad_leg_lengths.view(),
            chi: chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "leg_lengths",
            value: -1.0,
        })
    );

    let mut bad_chi = chi.clone();
    bad_chi[0] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_retained_path_output(GenfmtRetainedPathOutputInput {
            path_index: 17,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            potential_indices: potential_indices.view(),
            positions: positions.view(),
            beta_angles: beta_angles.view(),
            eta_angles: eta_angles.view(),
            leg_lengths: leg_lengths.view(),
            chi: bad_chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "chi",
            row: 0,
            ..
        })
    ));
}

#[test]
fn genfmt_decomposed_chi_amplitude_phase_matches_genfmtjas_reference() -> Result<(), GenfmtError> {
    let decomposed_chi = genfmt_decomposed_reference_chi();

    let output = genfmt_decomposed_chi_amplitude_phase(GenfmtDecomposedChiAmplitudePhaseInput {
        decomposed_chi: decomposed_chi.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_eq!(output.amplitudes.shape(), &[2, 2, 3]);
    assert_eq!(output.amplitudes.strides(), &[1, 2, 4]);
    assert_close(output.amplitudes[(0, 0, 0)], 1.0);
    assert_close(output.amplitudes[(0, 0, 2)], 1.5);
    assert_close(output.amplitudes[(0, 1, 1)], 1.1);
    assert_close(output.amplitudes[(1, 0, 2)], 0.9);
    assert_close(output.amplitudes[(1, 1, 0)], 1.4);

    assert_close(output.phases[(0, 0, 0)], 0.2);
    assert_close(output.phases[(0, 0, 2)], 0.6);
    assert_close(output.phases[(0, 1, 0)], 3.0);
    assert_close(output.phases[(0, 1, 1)], 3.233_185_307_179_586_4);
    assert_close(output.phases[(0, 1, 2)], 3.383_185_307_179_586_8);
    assert_close(output.phases[(1, 0, 2)], -0.9);
    assert_close(output.phases[(1, 1, 2)], 3.283_185_307_179_586_2);
    Ok(())
}

#[test]
fn genfmt_decomposed_chi_amplitude_phase_resets_phase_per_channel() -> Result<(), GenfmtError> {
    let decomposed_chi = genfmt_decomposed_reference_chi();
    let output = genfmt_decomposed_chi_amplitude_phase(GenfmtDecomposedChiAmplitudePhaseInput {
        decomposed_chi: decomposed_chi.view(),
        phase_epsilon: 1.0e-16,
    })?;

    assert_close(output.phases[(0, 1, 0)], 3.0);
    assert_close(output.phases[(1, 0, 0)], -0.5);
    Ok(())
}

#[test]
fn genfmt_decomposed_chi_amplitude_phase_rejects_invalid_inputs() {
    let empty = Array3::zeros((0, 2, 3).f());
    assert_eq!(
        genfmt_decomposed_chi_amplitude_phase(GenfmtDecomposedChiAmplitudePhaseInput {
            decomposed_chi: empty.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "decomposition_rows",
            value: 0,
        })
    );

    let decomposed_chi = genfmt_decomposed_reference_chi();
    assert_eq!(
        genfmt_decomposed_chi_amplitude_phase(GenfmtDecomposedChiAmplitudePhaseInput {
            decomposed_chi: decomposed_chi.view(),
            phase_epsilon: -1.0,
        }),
        Err(GenfmtError::NegativeScalar {
            field: "phase_epsilon",
            value: -1.0,
        })
    );

    let mut bad_chi = decomposed_chi.clone();
    bad_chi[(1, 0, 2)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        genfmt_decomposed_chi_amplitude_phase(GenfmtDecomposedChiAmplitudePhaseInput {
            decomposed_chi: bad_chi.view(),
            phase_epsilon: 1.0e-16,
        }),
        Err(GenfmtError::NonFiniteTensor3Complex {
            table: "decomposed_chi",
            i0: 1,
            i1: 0,
            i2: 2,
            ..
        })
    ));
}

#[test]
fn jas_one_sided_transition_matrices_match_feff_mmtrjas_reference() -> Result<(), GenfmtError> {
    let data = mmtrjas_reference_data();
    let matrices = jas_one_sided_transition_matrices(data.input())?;

    assert_eq!(matrices.generated_final_j2, vec![1, 1, 3, 3]);
    assert_eq!(matrices.left_matrix.shape(), &[4, 7, 3, 4]);
    assert_eq!(matrices.left_matrix.strides(), &[1, 4, 28, 84]);
    assert_eq!(matrices.right_matrix.shape(), &[4, 7, 3, 4]);
    assert_eq!(matrices.right_matrix.strides(), &[1, 4, 28, 84]);
    assert_complex_close(
        complex4_sum(&matrices.left_matrix),
        Complex::new(-8.277_797_958_733_443, -0.934_599_979_033_657_9),
    );
    assert_complex_close(
        complex4_sum(&matrices.right_matrix),
        Complex::new(-7.941_602_117_420_824, -3.804_547_793_137_926),
    );
    assert_complex_close(
        matrices.left_matrix[(1, 4, 0, 1)],
        Complex::new(0.034_677_463_413_266_635, 0.006_083_393_800_935_682),
    );
    assert_complex_close(
        matrices.left_matrix[(2, 2, 1, 2)],
        Complex::new(-0.510_969_182_083_186_8, -0.086_831_937_996_326_96),
    );
    assert_complex_close(
        matrices.left_matrix[(2, 5, 2, 3)],
        Complex::new(0.005_044_093_305_638_375, -0.107_270_006_137_426_38),
    );
    assert_complex_close(
        matrices.left_matrix[(1, 1, 1, 3)],
        Complex::new(-0.048_257_962_177_175_73, 0.018_649_823_650_482_26),
    );
    assert_complex_close(
        matrices.right_matrix[(1, 4, 0, 1)],
        Complex::new(-0.015_506_753_109_301_552, 0.127_763_441_142_026_48),
    );
    assert_complex_close(
        matrices.right_matrix[(2, 2, 1, 2)],
        Complex::new(-0.572_544_057_067_629, -0.010_075_578_494_005_052),
    );
    assert_complex_close(
        matrices.right_matrix[(2, 5, 2, 3)],
        Complex::new(0.271_168_086_362_318_15, 0.172_206_576_708_800_02),
    );
    assert_complex_close(
        matrices.right_matrix[(1, 1, 1, 3)],
        Complex::new(-0.028_571_087_180_750_965, 0.005_408_483_016_787_932),
    );
    assert_complex_close(matrices.left_matrix[(3, 5, 2, 3)], Complex::new(0.0, 0.0));
    assert_complex_close(matrices.right_matrix[(3, 5, 2, 3)], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn jas_one_sided_transition_matrices_reject_invalid_inputs() {
    let data = mmtrjas_reference_data();
    assert_eq!(
        jas_one_sided_transition_matrices(JasOneSidedTransitionInput {
            initial_kappa: 0,
            ..data.input()
        }),
        Err(GenfmtError::InvalidInitialKappa { kappa: 0 })
    );

    let mut zero_phase = data.q_phases.clone();
    zero_phase[0] = Complex::new(0.0, 0.0);
    assert_eq!(
        jas_one_sided_transition_matrices(JasOneSidedTransitionInput {
            q_phases: zero_phase.view(),
            ..data.input()
        }),
        Err(GenfmtError::ZeroComplex { field: "q_phases" })
    );

    let mut bad_beta = data.q_beta_angles.clone();
    bad_beta[1] = f64::NAN;
    assert!(matches!(
        jas_one_sided_transition_matrices(JasOneSidedTransitionInput {
            q_beta_angles: bad_beta.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableScalar {
            table: "q_beta_angles",
            row: 1,
            ..
        })
    ));

    let short_lj = Array1::from_vec(vec![0, 1, 1]);
    assert!(matches!(
        jas_one_sided_transition_matrices(JasOneSidedTransitionInput {
            final_lj_momenta: short_lj.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "final_lj_momenta",
            axis: "transition",
            ..
        })
    ));

    let short_last_rotation = Array3::zeros((3, 7, 6).f());
    assert!(matches!(
        jas_one_sided_transition_matrices(JasOneSidedTransitionInput {
            last_rotation: short_last_rotation.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "last_rotation",
            axis: "m2",
            ..
        })
    ));
}

#[test]
fn jas_scattering_amplitude_matrices_match_feff_mmtrxijas0_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas0_reference_data();
    let matrices = jas_scattering_amplitude_matrices(data.input())?;
    let decomposed = matrices
        .decomposed_amplitudes
        .as_ref()
        .expect("angular decomposition was requested");

    assert_eq!(matrices.amplitudes.shape(), &[4, 2, 4, 4]);
    assert_eq!(matrices.amplitudes.strides(), &[1, 4, 8, 32]);
    assert_eq!(decomposed.shape(), &[4, 2, 3, 4, 4]);
    assert_eq!(decomposed.strides(), &[1, 4, 8, 24, 96]);

    assert_complex_close(
        complex4_sum(&matrices.amplitudes),
        Complex::new(-0.006_921_578_555_211_599, 0.000_935_987_085_663_069_2),
    );
    assert_complex_close(
        complex5_sum(decomposed),
        Complex::new(-0.005_166_184_563_092_493, 0.001_078_099_621_085_962_3),
    );
    assert_complex_close(
        matrices.amplitudes[(0, 0, 0, 0)],
        Complex::new(0.000_180_346_480_22, 0.000_034_592_309_56),
    );
    assert_complex_close(
        matrices.amplitudes[(2, 1, 1, 0)],
        Complex::new(0.000_195_860_680_820_122_3, -0.000_066_229_968_760_970_32),
    );
    assert_complex_close(
        matrices.amplitudes[(3, 0, 3, 1)],
        Complex::new(-0.000_130_386_880_322_040_22, 0.000_069_626_338_242_270_14),
    );
    assert_complex_close(matrices.amplitudes[(1, 1, 2, 3)], Complex::new(0.0, 0.0));

    assert_complex_close(
        decomposed[(0, 0, 0, 0, 0)],
        Complex::new(0.000_012_491_541_04, 0.000_012_401_316_18),
    );
    assert_complex_close(
        decomposed[(2, 1, 1, 1, 0)],
        Complex::new(0.000_059_276_703_677_054_82, -0.000_021_171_960_280_479_47),
    );
    assert_complex_close(
        decomposed[(3, 0, 2, 3, 1)],
        Complex::new(-0.000_096_385_295_652_882_38, 0.000_112_064_447_167_902_03),
    );
    assert_complex_close(decomposed[(1, 1, 2, 2, 3)], Complex::new(0.0, 0.0));
    Ok(())
}

#[test]
fn jas_scattering_amplitude_matrices_reject_invalid_inputs() {
    let data = mmtrxijas0_reference_data();
    assert!(matches!(
        jas_scattering_amplitude_matrices(JasScatteringAmplitudeInput {
            lambda_count: 5,
            ..data.input()
        }),
        Err(GenfmtError::LambdaCountOutOfRange {
            name: "lambda_count",
            requested: 5,
            available: 4,
        })
    ));

    assert_eq!(
        jas_scattering_amplitude_matrices(JasScatteringAmplitudeInput {
            initial_j2: -1,
            ..data.input()
        }),
        Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jinit",
            value: -1,
        })
    );

    let mut bad_weights = data.q_weights.clone();
    bad_weights[1] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        jas_scattering_amplitude_matrices(JasScatteringAmplitudeInput {
            q_weights: bad_weights.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "q_weights",
            row: 1,
            ..
        })
    ));

    let mut bad_transition = data.transition_matrix.clone();
    bad_transition[(0, 0, 2, 2, 0)] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        jas_scattering_amplitude_matrices(JasScatteringAmplitudeInput {
            transition_matrix: bad_transition.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTensor5Complex {
            table: "transition_matrix",
            i0: 0,
            i1: 0,
            i2: 2,
            i3: 2,
            i4: 0,
            ..
        })
    ));

    let short_transition_matrix = Array5::zeros((3, 2, 5, 5, 3).f());
    assert!(matches!(
        jas_scattering_amplitude_matrices(JasScatteringAmplitudeInput {
            transition_matrix: short_transition_matrix.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "transition_matrix",
            axis: "mj",
            ..
        })
    ));
}

#[test]
fn jas_left_right_amplitude_matrices_match_feff_mmtrxijas_reference() -> Result<(), GenfmtError> {
    let data = mmtrxijas_reference_data();
    let matrices = jas_left_right_amplitude_matrices(data.input())?;
    let decomposed_left = matrices
        .decomposed_left_amplitudes
        .as_ref()
        .expect("left angular decomposition was requested");
    let decomposed_right = matrices
        .decomposed_right_amplitudes
        .as_ref()
        .expect("right angular decomposition was requested");

    assert_eq!(matrices.left_amplitudes.shape(), &[4, 3, 5]);
    assert_eq!(matrices.left_amplitudes.strides(), &[1, 4, 12]);
    assert_eq!(matrices.right_amplitudes.shape(), &[4, 3, 5]);
    assert_eq!(matrices.right_amplitudes.strides(), &[1, 4, 12]);
    assert_eq!(decomposed_left.shape(), &[4, 3, 3, 5]);
    assert_eq!(decomposed_left.strides(), &[1, 4, 12, 36]);
    assert_eq!(decomposed_right.shape(), &[4, 3, 3, 5]);
    assert_eq!(decomposed_right.strides(), &[1, 4, 12, 36]);

    assert_complex_close(
        complex3_sum(&matrices.left_amplitudes),
        Complex::new(0.034_786_338_721_399_09, -0.033_022_851_886_168_43),
    );
    assert_complex_close(
        complex3_sum(&matrices.right_amplitudes),
        Complex::new(1.077_206_789_254_615_8, 1.912_849_363_494_874_7),
    );
    assert_complex_close(
        complex4_sum(decomposed_left),
        Complex::new(-0.007_428_352_400_339_67, -0.009_090_019_299_865_67),
    );
    assert_complex_close(
        complex4_sum(decomposed_right),
        Complex::new(0.517_253_853_554_418_3, 0.632_544_743_507_885_8),
    );

    assert_complex_close(
        matrices.left_amplitudes[(0, 0, 0)],
        Complex::new(0.004_432_801_220_000_001, 0.003_701_889_540_000_001),
    );
    assert_complex_close(
        matrices.right_amplitudes[(0, 0, 0)],
        Complex::new(0.005_354_596_733_541_892, 0.014_712_895_976_469_187),
    );
    assert_complex_close(
        matrices.left_amplitudes[(2, 1, 2)],
        Complex::new(-0.007_538_061_735_417_915, 0.001_941_738_744_337_205),
    );
    assert_complex_close(
        matrices.right_amplitudes[(2, 1, 2)],
        Complex::new(0.003_173_042_609_089_469, 0.033_701_134_089_792_79),
    );
    assert_complex_close(
        matrices.left_amplitudes[(3, 2, 3)],
        Complex::new(0.004_937_346_089_914_192, -0.005_358_501_973_003_61),
    );
    assert_complex_close(
        matrices.right_amplitudes[(3, 2, 3)],
        Complex::new(0.039_874_161_094_060_3, 0.053_578_134_928_085_68),
    );
    assert_complex_close(
        matrices.left_amplitudes[(1, 2, 4)],
        Complex::new(0.010_471_100_353_799_999, -0.001_751_919_003_000_001_5),
    );
    assert_complex_close(
        matrices.right_amplitudes[(1, 2, 4)],
        Complex::new(0.036_511_345_486_706_77, 0.033_084_428_236_362_054),
    );

    assert_complex_close(
        decomposed_left[(0, 0, 0, 0)],
        Complex::new(0.000_067_301_300_000_000_03, 0.000_160_419_820_000_000_08),
    );
    assert_complex_close(
        decomposed_right[(0, 0, 0, 0)],
        Complex::new(0.000_265_622_033_898_305_1, -0.000_143_876_271_186_440_66),
    );
    assert_complex_close(
        decomposed_left[(2, 1, 1, 2)],
        Complex::new(-0.001_041_848_981_396_156_8, 0.000_446_028_374_593_973_25),
    );
    assert_complex_close(
        decomposed_right[(2, 1, 1, 2)],
        Complex::new(0.000_871_508_219_178_082_6, 0.002_648_034_246_575_343),
    );
    assert_complex_close(decomposed_left[(3, 2, 2, 3)], Complex::new(0.0, 0.0));
    assert_complex_close(
        decomposed_right[(3, 2, 2, 3)],
        Complex::new(0.015_188_327_586_206_897, 0.017_083_821_839_080_457),
    );
    Ok(())
}

#[test]
fn jas_left_right_amplitude_matrices_reject_invalid_inputs() {
    let data = mmtrxijas_reference_data();
    assert!(matches!(
        jas_left_right_amplitude_matrices(JasLeftRightAmplitudeInput {
            lambda_count: 6,
            ..data.input()
        }),
        Err(GenfmtError::LambdaCountOutOfRange {
            name: "lambda_count",
            requested: 6,
            available: 5,
        })
    ));

    assert_eq!(
        jas_left_right_amplitude_matrices(JasLeftRightAmplitudeInput {
            initial_j2: -1,
            ..data.input()
        }),
        Err(GenfmtError::InvalidDoubledAngularMomentum {
            field: "jinit",
            value: -1,
        })
    );

    let mut bad_weights = data.q_weights.clone();
    bad_weights[0] = Complex::new(0.0, f64::NAN);
    assert!(matches!(
        jas_left_right_amplitude_matrices(JasLeftRightAmplitudeInput {
            q_weights: bad_weights.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTableComplex {
            table: "q_weights",
            row: 0,
            ..
        })
    ));

    let mut bad_transition = data.left_transition_matrix.clone();
    bad_transition[(0, 2, 0, 0)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        jas_left_right_amplitude_matrices(JasLeftRightAmplitudeInput {
            left_transition_matrix: bad_transition.view(),
            ..data.input()
        }),
        Err(GenfmtError::NonFiniteTensorComplex {
            table: "left_transition_matrix",
            i0: 0,
            i1: 2,
            i2: 0,
            i3: 0,
            ..
        })
    ));

    let short_transition_matrix = Array4::zeros((4, 4, 3, 4).f());
    assert!(matches!(
        jas_left_right_amplitude_matrices(JasLeftRightAmplitudeInput {
            right_transition_matrix: short_transition_matrix.view(),
            ..data.input()
        }),
        Err(GenfmtError::TableAxisTooShort {
            table: "right_transition_matrix",
            axis: "mu",
            ..
        })
    ));
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
fn genfmt_ordinary_transition_matrices_match_genfmtsub_spin_loop_reference()
-> Result<(), GenfmtError> {
    let data = mmtr_reference_data();
    let matrices = genfmt_ordinary_transition_matrices(GenfmtOrdinaryTransitionMatricesInput {
        spin_selector: 1,
        active_spin_channel_count: 2,
        available_spin_channels: 2,
        transition_angular_momenta: data.transition_angular_momenta.view(),
        transition_b_matrix: data.transition_b_matrix.view(),
        transition_magnetic_offset: 3,
        initial_l: 2,
        magnetic_limit: 3,
        rotation_magnetic_offset: 3,
        rotations: data.polarized_input().rotations,
    })?;

    assert_eq!(matrices.matrices.shape(), &[2, 7, 8, 7, 8]);
    assert_eq!(matrices.matrices.strides(), &[1, 2, 14, 112, 784]);
    assert_eq!(matrices.b_matrix_spin_indices, vec![1, 0]);

    let first_expected = energy_independent_transition_matrix(data.polarized_input())?;
    let second_expected = energy_independent_transition_matrix(EnergyIndependentMatrixInput {
        spin_index: 0,
        ..data.polarized_input()
    })?;
    assert_complex_close(
        matrices.matrices[(0, 2, 1, 4, 2)],
        first_expected[(2, 1, 4, 2)],
    );
    assert_complex_close(
        matrices.matrices[(1, 2, 1, 4, 2)],
        second_expected[(2, 1, 4, 2)],
    );
    assert_complex_close(
        complex4_sum(&matrices.matrices.index_axis(Axis(0), 0).to_owned()),
        complex4_sum(&first_expected),
    );
    assert_complex_close(
        complex4_sum(&matrices.matrices.index_axis(Axis(0), 1).to_owned()),
        complex4_sum(&second_expected),
    );

    let single = genfmt_ordinary_transition_matrices(GenfmtOrdinaryTransitionMatricesInput {
        spin_selector: 2,
        active_spin_channel_count: 1,
        available_spin_channels: 2,
        transition_angular_momenta: data.transition_angular_momenta.view(),
        transition_b_matrix: data.transition_b_matrix.view(),
        transition_magnetic_offset: 3,
        initial_l: 2,
        magnetic_limit: 3,
        rotation_magnetic_offset: 3,
        rotations: data.unpolarized_input().rotations,
    })?;
    let single_expected = energy_independent_transition_matrix(data.unpolarized_input())?;
    assert_eq!(single.b_matrix_spin_indices, vec![0]);
    assert_complex_close(
        single.matrices[(0, 5, 3, 3, 3)],
        single_expected[(5, 3, 3, 3)],
    );
    Ok(())
}

#[test]
fn genfmt_ordinary_transition_matrices_reject_invalid_spin_setup() {
    let data = mmtr_reference_data();
    assert_eq!(
        genfmt_ordinary_transition_matrices(GenfmtOrdinaryTransitionMatricesInput {
            spin_selector: 1,
            active_spin_channel_count: 1,
            available_spin_channels: 2,
            transition_angular_momenta: data.transition_angular_momenta.view(),
            transition_b_matrix: data.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: data.polarized_input().rotations,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "active_spin_channel_count",
            value: 1,
        })
    );
    assert_eq!(
        genfmt_ordinary_transition_matrices(GenfmtOrdinaryTransitionMatricesInput {
            spin_selector: 1,
            active_spin_channel_count: 2,
            available_spin_channels: 0,
            transition_angular_momenta: data.transition_angular_momenta.view(),
            transition_b_matrix: data.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: data.polarized_input().rotations,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "available_spin_channels",
            value: 0,
        })
    );
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
fn genfmt_nstar_row_matches_genfmt_driver_reference() -> Result<(), GenfmtError> {
    let positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let row = genfmt_nstar_row(GenfmtNStarInput {
        path_number: 7,
        positions: positions.view(),
        primary_polarization: [0.2, 0.9, 0.4],
        ellipticity_vector: [f64::NAN, 0.0, 0.0],
        degeneracy: 1.75,
        initial_l: 1,
        ellipticity: 0.0,
    })?;

    assert_eq!(row.path_number, 7);
    assert_close(row.nstar, 0.212_068_566_596_440_38);
    Ok(())
}

#[test]
fn genfmt_nstar_row_applies_elliptic_cross_polarization() -> Result<(), GenfmtError> {
    let positions = arr2(&[[1.2, -0.5, 0.8], [-0.7, 1.4, 0.6], [0.25, -0.5, 0.1]]);
    let primary = [0.0, 0.0, 1.0];
    let ellipticity_vector = [0.25, -0.5, 3.0];
    let secondary = [-0.5, -0.25, 0.0];

    let row = genfmt_nstar_row(GenfmtNStarInput {
        path_number: 2,
        positions: positions.view(),
        primary_polarization: primary,
        ellipticity_vector,
        degeneracy: 3.51,
        initial_l: 2,
        ellipticity: 0.7,
    })?;

    assert_eq!(row.path_number, 2);
    assert_close(
        row.nstar,
        xstar(XStarInput {
            primary_polarization: primary,
            secondary_polarization: secondary,
            first_leg: [0.95, 0.0, 0.7],
            last_leg: [-0.95, 1.9, 0.5],
            degeneracy: 4.0,
            initial_l: 2,
            ellipticity: 0.7,
        })?,
    );
    Ok(())
}

#[test]
fn genfmt_nstar_row_rejects_invalid_inputs() {
    let positions = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);

    assert_eq!(
        genfmt_nstar_row(GenfmtNStarInput {
            path_number: 0,
            positions: positions.view(),
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "path_number",
            value: 0,
        })
    );

    let short_path = arr2(&[[0.0, 0.0, 0.0]]);
    assert_eq!(
        genfmt_nstar_row(GenfmtNStarInput {
            path_number: 1,
            positions: short_path.view(),
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::InvalidAngularLimit {
            name: "leg_count",
            value: 1,
        })
    );

    let short_coordinates = arr2(&[[1.0, 0.0], [0.0, 0.0]]);
    assert_eq!(
        genfmt_nstar_row(GenfmtNStarInput {
            path_number: 1,
            positions: short_coordinates.view(),
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::InvalidPathCoordinateColumns { columns: 2 })
    );

    let mut bad_position = positions.clone();
    bad_position[(1, 2)] = f64::NAN;
    assert!(matches!(
        genfmt_nstar_row(GenfmtNStarInput {
            path_number: 1,
            positions: bad_position.view(),
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.0,
        }),
        Err(GenfmtError::NonFinitePathCoordinate {
            leg_index: 1,
            component: 2,
            ..
        })
    ));

    assert!(matches!(
        genfmt_nstar_row(GenfmtNStarInput {
            path_number: 1,
            positions: positions.view(),
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [f64::NAN, 1.0, 0.0],
            degeneracy: 1.0,
            initial_l: 1,
            ellipticity: 0.5,
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "secondary_polarization",
            index: 1,
            ..
        })
    ));
}

#[test]
fn genfmt_nstar_rows_match_genfmt_driver_bookkeeping_reference() -> Result<(), GenfmtError> {
    let first_positions = arr2(&[[1.0, 0.5, -0.25], [0.4, -0.3, 1.2], [0.0, 0.0, 0.0]]);
    let second_positions = arr2(&[[1.2, -0.2, 0.7], [0.3, 0.6, -0.4], [0.0, 0.0, 0.0]]);
    let primary_polarization = [0.2, 0.9, 0.4];
    let ellipticity_vector = [0.1, -0.3, 0.8];
    let path_inputs = [
        GenfmtNStarPathInput {
            positions: first_positions.view(),
            degeneracy: 2.6,
        },
        GenfmtNStarPathInput {
            positions: second_positions.view(),
            degeneracy: 1.4,
        },
    ];

    let rows = genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization,
        ellipticity_vector,
        initial_l: 1,
        ellipticity: 0.2,
        path_inputs: &path_inputs,
    })?;

    let expected_first = genfmt_nstar_row(GenfmtNStarInput {
        path_number: 1,
        positions: first_positions.view(),
        primary_polarization,
        ellipticity_vector,
        degeneracy: 2.6,
        initial_l: 1,
        ellipticity: 0.2,
    })?;
    let expected_second = genfmt_nstar_row(GenfmtNStarInput {
        path_number: 2,
        positions: second_positions.view(),
        primary_polarization,
        ellipticity_vector,
        degeneracy: 1.4,
        initial_l: 1,
        ellipticity: 0.2,
    })?;

    assert_eq!(rows.primary_polarization, primary_polarization);
    assert_eq!(rows.rows, vec![expected_first, expected_second]);
    Ok(())
}

#[test]
fn genfmt_nstar_rows_reject_invalid_header_inputs() {
    let path_inputs: [GenfmtNStarPathInput<'_>; 0] = [];

    assert!(matches!(
        genfmt_nstar_rows(GenfmtNStarRowsInput {
            primary_polarization: [1.0, f64::NAN, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            initial_l: 1,
            ellipticity: 0.0,
            path_inputs: &path_inputs,
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "primary_polarization",
            index: 1,
            ..
        })
    ));
    assert!(matches!(
        genfmt_nstar_rows(GenfmtNStarRowsInput {
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, f64::NAN, 0.0],
            initial_l: 1,
            ellipticity: 0.0,
            path_inputs: &path_inputs,
        }),
        Err(GenfmtError::NonFiniteVector {
            field: "ellipticity_vector",
            index: 1,
            ..
        })
    ));
    assert!(matches!(
        genfmt_nstar_rows(GenfmtNStarRowsInput {
            primary_polarization: [1.0, 0.0, 0.0],
            ellipticity_vector: [0.0, 1.0, 0.0],
            initial_l: 1,
            ellipticity: f64::NAN,
            path_inputs: &path_inputs,
        }),
        Err(GenfmtError::NonFiniteScalar {
            field: "ellipticity",
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

fn assert_array_complex_close(actual: &Array1<Complex>, expected: &[Complex]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let real_tolerance = 1.0e-12_f64.max(expected.re.abs() * 1.0e-12);
        let imaginary_tolerance = 1.0e-12_f64.max(expected.im.abs() * 1.0e-12);
        assert!(
            (actual.re - expected.re).abs() <= real_tolerance
                && (actual.im - expected.im).abs() <= imaginary_tolerance,
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

struct GenfmtPathTraceReferenceData {
    first_scattering: Array2<Complex>,
    intermediate_scattering: Array3<Complex>,
    termination_matrix: Array2<Complex>,
}

impl GenfmtPathTraceReferenceData {
    fn input(&self) -> GenfmtPathMatrixTraceInput<'_> {
        GenfmtPathMatrixTraceInput {
            first_scattering: self.first_scattering.view(),
            intermediate_scattering: self.intermediate_scattering.view(),
            termination_matrix: self.termination_matrix.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
        }
    }

    fn product_input(&self) -> GenfmtPathMatrixProductInput<'_> {
        GenfmtPathMatrixProductInput {
            first_scattering: self.first_scattering.view(),
            intermediate_scattering: self.intermediate_scattering.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
        }
    }
}

fn genfmt_path_trace_reference_data() -> GenfmtPathTraceReferenceData {
    let mut first_scattering = Array2::zeros((4, 3).f());
    for lambda in 0..4 {
        let lam = (lambda + 1) as Real;
        for initial_lambda in 0..3 {
            let init = (initial_lambda + 1) as Real;
            first_scattering[(lambda, initial_lambda)] =
                Complex::new(0.05 * lam - 0.02 * init, 0.03 * lam + 0.04 * init);
        }
    }

    let mut intermediate_scattering = Array3::zeros((2, 4, 4).f());
    for intermediate_leg in 0..2 {
        let leg = (intermediate_leg + 1) as Real;
        for lambda in 0..4 {
            let lam = (lambda + 1) as Real;
            for inner_lambda in 0..4 {
                let inner = (inner_lambda + 1) as Real;
                intermediate_scattering[(intermediate_leg, lambda, inner_lambda)] = Complex::new(
                    0.02 * leg + 0.04 * lam - 0.01 * inner,
                    -0.03 * leg + 0.015 * lam + 0.025 * inner,
                );
            }
        }
    }

    let mut termination_matrix = Array2::zeros((3, 3).f());
    for lambda in 0..3 {
        let lam = (lambda + 1) as Real;
        for initial_lambda in 0..3 {
            let init = (initial_lambda + 1) as Real;
            termination_matrix[(lambda, initial_lambda)] =
                Complex::new(-0.01 + 0.06 * lam + 0.025 * init, 0.02 * lam - 0.035 * init);
        }
    }

    GenfmtPathTraceReferenceData {
        first_scattering,
        intermediate_scattering,
        termination_matrix,
    }
}

struct GenfmtOrdinaryPathTraceReferenceData {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    path_potential_indices: Array1<usize>,
    angular_limits: Array1<usize>,
    phase_shifts: Array2<Complex>,
    curved_wave_polynomials: Array3<Complex>,
    rotations: Array4<Real>,
    xnlm: Array2<Real>,
    eta_angles: Array1<Real>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array1<Complex>,
    transition_matrix: Array4<Complex>,
}

impl GenfmtOrdinaryPathTraceReferenceData {
    fn input(&self) -> GenfmtOrdinaryPathTraceInput<'_> {
        GenfmtOrdinaryPathTraceInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
            path_potential_indices: self.path_potential_indices.view(),
            angular_limits: self.angular_limits.view(),
            phase_shifts: self.phase_shifts.view(),
            signed_angular_offset: 4,
            curved_wave_polynomials: self.curved_wave_polynomials.view(),
            rotations: self.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta_angles: self.eta_angles.view(),
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            transition_matrix: self.transition_matrix.view(),
            transition_magnetic_offset: 4,
        }
    }

    fn scattering_product_input(&self) -> GenfmtScatteringPathProductInput<'_> {
        GenfmtScatteringPathProductInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            full_lambda_count: 4,
            initial_lambda_count: 3,
            path_potential_indices: self.path_potential_indices.view(),
            angular_limits: self.angular_limits.view(),
            phase_shifts: self.phase_shifts.view(),
            signed_angular_offset: 4,
            curved_wave_polynomials: self.curved_wave_polynomials.view(),
            rotations: self.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta_angles: self.eta_angles.view(),
        }
    }

    fn first_scattering_input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
        ScatteringAmplitudeMatrixInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            left_lambda_count: 4,
            right_lambda_count: 3,
            phase_shifts: self.phase_shifts.index_axis(Axis(1), 0),
            angular_limit: self.angular_limits[0],
            first_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 1),
            second_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 0),
            rotation: self.rotations.index_axis(Axis(0), 0),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta: self.eta_angles[1],
        }
    }

    fn last_scattering_input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
        ScatteringAmplitudeMatrixInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            left_lambda_count: 3,
            right_lambda_count: 4,
            phase_shifts: self.phase_shifts.index_axis(Axis(1), 1),
            angular_limit: self.angular_limits[1],
            first_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 2),
            second_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 1),
            rotation: self.rotations.index_axis(Axis(0), 1),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta: self.eta_angles[2],
        }
    }

    fn termination_input(&self) -> PolarizedScatteringAmplitudeInput<'_> {
        PolarizedScatteringAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 3,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            transition_matrix: self.transition_matrix.view(),
            transition_magnetic_offset: 4,
            first_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 0),
            second_leg_polynomials: self.curved_wave_polynomials.index_axis(Axis(2), 2),
            xnlm: self.xnlm.view(),
            eta: self.eta_angles[0],
        }
    }
}

fn genfmt_ordinary_path_trace_reference_data()
-> Result<GenfmtOrdinaryPathTraceReferenceData, Box<dyn std::error::Error>> {
    let matrix_data = mmtrxi_reference_data()?;
    let path_potential_indices = Array1::from_vec(vec![0, 1, 0]);
    let angular_limits = Array1::from_vec(vec![3, 3]);

    let mut phase_shifts = Array2::zeros((9, 2).f());
    for potential in 0..2 {
        let potential_value = potential as Real;
        for signed_l in -4..=4 {
            let row = (signed_l + 4) as usize;
            let angular_value = signed_l as Real;
            phase_shifts[(row, potential)] = Complex::new(
                0.015 * angular_value + 0.02 * potential_value + 0.02,
                -0.01 * angular_value + 0.03 * potential_value + 0.01,
            );
        }
    }

    let leg_rhos = Array1::from_vec(vec![
        Complex::new(1.25, 0.4),
        Complex::new(-0.8, 1.1),
        Complex::new(0.55, -0.25),
    ]);
    let leg_limits = vec![
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 0,
            current_potential_index: 0,
            angular_count: 4,
            mixed_order_count: 4,
        },
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 0,
            current_potential_index: 1,
            angular_count: 4,
            mixed_order_count: 4,
        },
        GenfmtCurvedWaveLegLimit {
            previous_potential_index: 1,
            current_potential_index: 0,
            angular_count: 4,
            mixed_order_count: 4,
        },
    ];
    let curved_wave_polynomials =
        genfmt_curved_wave_polynomial_tables(GenfmtCurvedWavePolynomialTablesInput {
            leg_rhos: leg_rhos.view(),
            leg_limits: &leg_limits,
            mixed_order_capacity: 4,
        })?
        .tables;

    let mut rotations = Array4::zeros((3, 5, 9, 9).f());
    for leg in 0..3 {
        let leg_value = (leg + 1) as Real;
        for l in 0..=4 {
            let angular_value = (l + 1) as Real;
            for m1 in -4_i32..=4 {
                for m2 in -4_i32..=4 {
                    if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                        rotations[(leg, l, (m1 + 4) as usize, (m2 + 4) as usize)] =
                            (0.07 * leg_value + 0.11 * angular_value + 0.03 * (m1 as Real)
                                - 0.04 * (m2 as Real))
                                .cos();
                    }
                }
            }
        }
    }

    Ok(GenfmtOrdinaryPathTraceReferenceData {
        m_indices: matrix_data.m_indices,
        n_indices: matrix_data.n_indices,
        path_potential_indices,
        angular_limits,
        phase_shifts,
        curved_wave_polynomials,
        rotations,
        xnlm: matrix_data.xnlm,
        eta_angles: Array1::from_vec(vec![0.19, 0.37, -0.21]),
        transition_angular_momenta: matrix_data.transition_angular_momenta,
        radial_factors: matrix_data.radial_factors,
        transition_matrix: matrix_data.transition_matrix,
    })
}

fn genfmt_ordinary_path_energy_point_tables(
    data: &GenfmtOrdinaryPathTraceReferenceData,
) -> (Array2<usize>, Array3<Complex>, Array2<Complex>) {
    let mut angular_limits = Array2::zeros((1, data.angular_limits.len()).f());
    for potential in 0..data.angular_limits.len() {
        angular_limits[(0, potential)] = data.angular_limits[potential];
    }

    let mut phase_shifts = Array3::zeros(
        (
            1,
            data.phase_shifts.shape()[0],
            data.phase_shifts.shape()[1],
        )
            .f(),
    );
    for signed_l in 0..data.phase_shifts.shape()[0] {
        for potential in 0..data.phase_shifts.shape()[1] {
            phase_shifts[(0, signed_l, potential)] = data.phase_shifts[(signed_l, potential)];
        }
    }

    let mut radial_factors = Array2::zeros((1, data.radial_factors.len()).f());
    for transition in 0..data.radial_factors.len() {
        radial_factors[(0, transition)] = data.radial_factors[transition];
    }

    (angular_limits, phase_shifts, radial_factors)
}

struct GenfmtOrdinaryEnergyGridReferenceData {
    angular_limits: Array2<usize>,
    spin_phase_shifts: Array4<Complex>,
    spin_radial_factors: Array3<Complex>,
    transition_matrices: Array5<Complex>,
    leg_lengths: Array1<Real>,
    complex_momenta: Array2<Complex>,
    wave_numbers: Array1<Real>,
}

fn genfmt_ordinary_energy_grid_reference_data(
    data: &GenfmtOrdinaryPathTraceReferenceData,
    energy_count: usize,
    spin_count: usize,
) -> GenfmtOrdinaryEnergyGridReferenceData {
    let mut angular_limits = Array2::zeros((energy_count, data.angular_limits.len()).f());
    let mut spin_phase_shifts = Array4::zeros(
        (
            energy_count,
            data.phase_shifts.shape()[0],
            spin_count,
            data.phase_shifts.shape()[1],
        )
            .f(),
    );
    let mut spin_radial_factors =
        Array3::zeros((energy_count, data.radial_factors.len(), spin_count).f());
    let transition_shape = data.transition_matrix.shape();
    let mut transition_matrices = Array5::zeros(
        (
            spin_count,
            transition_shape[0],
            transition_shape[1],
            transition_shape[2],
            transition_shape[3],
        )
            .f(),
    );
    let mut complex_momenta = Array2::<Complex>::zeros((energy_count, spin_count).f());
    let mut wave_numbers = Array1::<Real>::zeros(energy_count);

    for energy in 0..energy_count {
        let energy_value = energy as Real;
        for potential in 0..data.angular_limits.len() {
            angular_limits[(energy, potential)] = data.angular_limits[potential];
        }
        for signed_l in 0..data.phase_shifts.shape()[0] {
            for spin in 0..spin_count {
                let spin_value = spin as Real;
                for potential in 0..data.phase_shifts.shape()[1] {
                    spin_phase_shifts[(energy, signed_l, spin, potential)] = data.phase_shifts
                        [(signed_l, potential)]
                        + Complex::new(
                            0.004 * energy_value + 0.006 * spin_value,
                            -0.003 * energy_value + 0.002 * spin_value,
                        );
                }
            }
        }
        for transition in 0..data.radial_factors.len() {
            for spin in 0..spin_count {
                let spin_value = spin as Real;
                spin_radial_factors[(energy, transition, spin)] = data.radial_factors[transition]
                    * Complex::new(1.0 + 0.07 * energy_value, -0.02 * spin_value)
                    + Complex::new(0.01 * spin_value, -0.004 * energy_value);
            }
        }
        for spin in 0..spin_count {
            let spin_value = spin as Real;
            complex_momenta[(energy, spin)] = if energy == 1 && spin == 0 {
                Complex::new(1.0e-18, 0.0)
            } else {
                Complex::new(
                    0.85 - 0.09 * energy_value + 0.03 * spin_value,
                    0.20 + 0.04 * energy_value + 0.02 * spin_value,
                )
            };
        }
        wave_numbers[energy] = 0.64 + 0.05 * energy_value;
    }

    for spin in 0..spin_count {
        let spin_value = spin as Real;
        let scale = Complex::new(1.0 + 0.05 * spin_value, -0.03 * spin_value);
        for m1 in 0..transition_shape[0] {
            for k1 in 0..transition_shape[1] {
                for m2 in 0..transition_shape[2] {
                    for k2 in 0..transition_shape[3] {
                        transition_matrices[(spin, m1, k1, m2, k2)] =
                            data.transition_matrix[(m1, k1, m2, k2)] * scale;
                    }
                }
            }
        }
    }

    GenfmtOrdinaryEnergyGridReferenceData {
        angular_limits,
        spin_phase_shifts,
        spin_radial_factors,
        transition_matrices,
        leg_lengths: Array1::from_vec(vec![1.25, 1.75, 0.95]),
        complex_momenta,
        wave_numbers,
    }
}

fn genfmt_ordinary_reference_path_setup(
    data: &GenfmtOrdinaryPathTraceReferenceData,
    grid_data: &GenfmtOrdinaryEnergyGridReferenceData,
) -> GenfmtPathSetup {
    let active_m_indices = data.m_indices.iter().take(4).copied().collect();
    let active_n_indices = data.n_indices.iter().take(4).copied().collect();

    GenfmtPathSetup {
        angles: PathRotationAngles {
            beta_angles: Array1::from_vec(vec![0.10, 0.20, 0.30]),
            eta_values: data.eta_angles.clone(),
            leg_lengths: grid_data.leg_lengths.clone(),
        },
        effective_half_path_length: grid_data.leg_lengths.iter().sum::<Real>() / 2.0,
        lambda: LambdaIndexSet {
            m_indices: Array1::from_vec(active_m_indices),
            n_indices: Array1::from_vec(active_n_indices),
            initial_l_prefix_len: 3,
            max_m_plus_one: 3,
            max_n: 1,
            order: 0,
            requested_n_max: 1,
            requested_m_max: 2,
            truncated: false,
        },
        rotations: GenfmtPathRotationTables {
            rotations: data.rotations.clone(),
            real_leg_count: data.path_potential_indices.len(),
            rotation_magnetic_offset: 4,
        },
    }
}

fn genfmt_reference_energies_for_complex_momenta(
    energies: &Array1<Complex>,
    complex_momenta: &Array2<Complex>,
) -> Array2<Complex> {
    let mut reference_energies = Array2::zeros((energies.len(), complex_momenta.shape()[1]).f());
    for energy in 0..energies.len() {
        for spin in 0..complex_momenta.shape()[1] {
            let momentum = complex_momenta[(energy, spin)];
            reference_energies[(energy, spin)] = energies[energy] - momentum * momentum * 0.5;
        }
    }
    reference_energies
}

fn genfmt_single_scattering_first_matrix() -> Array2<Complex> {
    let mut first = Array2::zeros((3, 2).f());
    for lambda in 0..3 {
        let lam = (lambda + 1) as Real;
        for initial_lambda in 0..2 {
            let init = (initial_lambda + 1) as Real;
            first[(lambda, initial_lambda)] =
                Complex::new(0.1 * lam + 0.03 * init, -0.02 * lam + 0.05 * init);
        }
    }
    first
}

fn genfmt_single_scattering_termination_matrix() -> Array2<Complex> {
    let mut termination = Array2::zeros((2, 2).f());
    for lambda in 0..2 {
        let lam = (lambda + 1) as Real;
        for initial_lambda in 0..2 {
            let init = (initial_lambda + 1) as Real;
            termination[(lambda, initial_lambda)] =
                Complex::new(0.07 * lam - 0.01 * init, 0.04 * lam + 0.02 * init);
        }
    }
    termination
}

fn genfmt_jas_path_product() -> Array2<Complex> {
    let mut product = Array2::zeros((2, 2).f());
    product[(0, 0)] = Complex::new(1.0, 0.1);
    product[(0, 1)] = Complex::new(-0.2, 0.3);
    product[(1, 0)] = Complex::new(0.4, -0.1);
    product[(1, 1)] = Complex::new(0.7, 0.2);
    product
}

fn genfmt_jas_path_product_for(lambda_count: usize) -> Array2<Complex> {
    let mut product = Array2::zeros((lambda_count, lambda_count).f());
    for lambda in 0..lambda_count {
        let lam = (lambda + 1) as Real;
        for initial_lambda in 0..lambda_count {
            let init = (initial_lambda + 1) as Real;
            product[(lambda, initial_lambda)] =
                Complex::new(0.07 * lam - 0.03 * init, 0.02 * lam + 0.05 * init);
        }
    }
    product
}

struct GenfmtJasEnergyPointCommonData {
    path_potential_indices: Array1<usize>,
    angular_limits: Array2<usize>,
    phase_shifts: Array3<Complex>,
    leg_lengths: Array1<Real>,
    complex_momentum: Complex,
    wave_number: Real,
    rotations: Array4<Real>,
    xnlm: Array2<Real>,
    eta_angles: Array1<Real>,
}

fn genfmt_jas_energy_point_common_data(_lambda_count: usize) -> GenfmtJasEnergyPointCommonData {
    let path_potential_indices = Array1::from_vec(vec![0, 0]);
    let angular_limits = arr2(&[[3]]);
    let mut phase_shifts = Array3::zeros((1, 9, 1).f());
    for signed_l in -4..=4 {
        let row = (signed_l + 4) as usize;
        let angular_value = signed_l as Real;
        phase_shifts[(0, row, 0)] =
            Complex::new(0.02 + 0.015 * angular_value, 0.01 - 0.01 * angular_value);
    }

    let mut rotations = Array4::zeros((2, 4, 9, 9).f());
    for leg in 0..2 {
        let leg_value = (leg + 1) as Real;
        for l in 0..=3 {
            let angular_value = (l + 1) as Real;
            for m1 in -4_i32..=4 {
                for m2 in -4_i32..=4 {
                    if i32_abs_usize(m1) <= l && i32_abs_usize(m2) <= l {
                        rotations[(leg, l, (m1 + 4) as usize, (m2 + 4) as usize)] =
                            (0.09 * leg_value + 0.13 * angular_value + 0.02 * (m1 as Real)
                                - 0.03 * (m2 as Real))
                                .cos();
                    }
                }
            }
        }
    }

    let mut xnlm = Array2::zeros((4, 4).f());
    for l in 0..4 {
        let il = (l + 1) as Real;
        for m in 0..4 {
            let im = (m + 1) as Real;
            xnlm[(m, l)] = 0.9 + 0.17 * il + 0.11 * im;
        }
    }

    GenfmtJasEnergyPointCommonData {
        path_potential_indices,
        angular_limits,
        phase_shifts,
        leg_lengths: Array1::from_vec(vec![1.15, 0.85]),
        complex_momentum: Complex::new(0.72, 0.18),
        wave_number: 0.52,
        rotations,
        xnlm,
        eta_angles: Array1::from_vec(vec![0.31, -0.17]),
    }
}

fn genfmt_jas_energy_grid_common_data(
    lambda_count: usize,
    energy_count: usize,
) -> (
    GenfmtJasEnergyPointCommonData,
    Array1<Complex>,
    Array1<Real>,
) {
    let mut common = genfmt_jas_energy_point_common_data(lambda_count);
    common.angular_limits = Array2::zeros((energy_count, 1).f());
    common.phase_shifts = Array3::zeros((energy_count, 9, 1).f());
    let mut complex_momenta = Array1::<Complex>::zeros(energy_count);
    let mut wave_numbers = Array1::<Real>::zeros(energy_count);

    for energy in 0..energy_count {
        common.angular_limits[(energy, 0)] = 3;
        let energy_value = energy as Real;
        for signed_l in -4..=4 {
            let row = (signed_l + 4) as usize;
            let angular_value = signed_l as Real;
            common.phase_shifts[(energy, row, 0)] = Complex::new(
                0.02 + 0.015 * angular_value + 0.004 * energy_value,
                0.01 - 0.01 * angular_value - 0.003 * energy_value,
            );
        }
        complex_momenta[energy] = if energy == 1 && energy_count > 2 {
            Complex::new(1.0e-18, 0.0)
        } else {
            Complex::new(0.72 - 0.08 * energy_value, 0.18 + 0.03 * energy_value)
        };
        wave_numbers[energy] = 0.52 + 0.04 * energy_value;
    }

    (common, complex_momenta, wave_numbers)
}

fn jas_radial_factor_energy_grid(
    radial_factors: ndarray::ArrayView2<'_, Complex>,
    energy_count: usize,
) -> Array3<Complex> {
    let mut grid = Array3::<Complex>::zeros(
        (
            energy_count,
            radial_factors.shape()[0],
            radial_factors.shape()[1],
        )
            .f(),
    );
    for energy in 0..energy_count {
        let energy_value = energy as Real;
        let scale = Complex::new(1.0 + 0.08 * energy_value, -0.03 * energy_value);
        for q in 0..radial_factors.shape()[0] {
            for transition in 0..radial_factors.shape()[1] {
                grid[(energy, q, transition)] = radial_factors[(q, transition)] * scale
                    + Complex::new(0.005 * energy_value, -0.002 * energy_value);
            }
        }
    }
    grid
}

fn genfmt_jas_left_amplitudes() -> Array3<Complex> {
    let mut amplitudes = Array3::zeros((2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for q in 0..2 {
            let qf = (q + 1) as Real;
            for lambda in 0..2 {
                let lam = (lambda + 1) as Real;
                amplitudes[(mj, q, lambda)] = Complex::new(
                    0.1 * mjf + 0.2 * qf + 0.3 * lam,
                    0.05 * mjf - 0.04 * qf + 0.02 * lam,
                );
            }
        }
    }
    amplitudes
}

fn genfmt_jas_right_amplitudes() -> Array3<Complex> {
    let mut amplitudes = Array3::zeros((2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for q in 0..2 {
            let qf = (q + 1) as Real;
            for lambda in 0..2 {
                let lam = (lambda + 1) as Real;
                amplitudes[(mj, q, lambda)] = Complex::new(
                    -0.03 * mjf + 0.17 * qf + 0.11 * lam,
                    0.07 * mjf + 0.01 * qf - 0.06 * lam,
                );
            }
        }
    }
    amplitudes
}

fn genfmt_jas_decomposed_left_amplitudes() -> Array4<Complex> {
    let mut amplitudes = Array4::zeros((2, 2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for q in 0..2 {
            let qf = (q + 1) as Real;
            for angular in 0..2 {
                let lf = (angular + 1) as Real;
                for lambda in 0..2 {
                    let lam = (lambda + 1) as Real;
                    amplitudes[(mj, q, angular, lambda)] = Complex::new(
                        0.02 * mjf + 0.03 * qf + 0.04 * lf + 0.05 * lam,
                        -0.01 * mjf + 0.015 * qf - 0.02 * lf + 0.025 * lam,
                    );
                }
            }
        }
    }
    amplitudes
}

fn genfmt_jas_decomposed_right_amplitudes() -> Array4<Complex> {
    let mut amplitudes = Array4::zeros((2, 2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for q in 0..2 {
            let qf = (q + 1) as Real;
            for angular in 0..2 {
                let lf = (angular + 1) as Real;
                for lambda in 0..2 {
                    let lam = (lambda + 1) as Real;
                    amplitudes[(mj, q, angular, lambda)] = Complex::new(
                        0.06 * mjf - 0.02 * qf + 0.01 * lf + 0.035 * lam,
                        0.012 * mjf + 0.02 * qf + 0.017 * lf - 0.014 * lam,
                    );
                }
            }
        }
    }
    amplitudes
}

fn genfmt_jas_spherical_amplitudes() -> Array4<Complex> {
    let mut amplitudes = Array4::zeros((2, 2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for spin in 0..2 {
            let sf = (spin + 1) as Real;
            for lambda2 in 0..2 {
                let l2 = (lambda2 + 1) as Real;
                for lambda1 in 0..2 {
                    let l1 = (lambda1 + 1) as Real;
                    amplitudes[(mj, spin, lambda2, lambda1)] = Complex::new(
                        0.04 * mjf + 0.03 * sf + 0.02 * l2 - 0.01 * l1,
                        -0.02 * mjf + 0.015 * sf + 0.025 * l2 + 0.005 * l1,
                    );
                }
            }
        }
    }
    amplitudes
}

fn genfmt_jas_spherical_decomposed_amplitudes() -> Array5<Complex> {
    let mut amplitudes = Array5::zeros((2, 2, 2, 2, 2).f());
    for mj in 0..2 {
        let mjf = (mj + 1) as Real;
        for spin in 0..2 {
            let sf = (spin + 1) as Real;
            for angular in 0..2 {
                let lf = (angular + 1) as Real;
                for lambda2 in 0..2 {
                    let l2 = (lambda2 + 1) as Real;
                    for lambda1 in 0..2 {
                        let l1 = (lambda1 + 1) as Real;
                        amplitudes[(mj, spin, angular, lambda2, lambda1)] = Complex::new(
                            -0.015 * mjf + 0.025 * sf + 0.035 * lf + 0.02 * l2 - 0.012 * l1,
                            0.018 * mjf - 0.011 * sf + 0.014 * lf + 0.017 * l2 + 0.009 * l1,
                        );
                    }
                }
            }
        }
    }
    amplitudes
}

fn genfmt_reference_chi() -> Array1<Complex> {
    Array1::from_vec(vec![
        Complex::new(0.20, 0.10),
        Complex::new(-0.10, 0.35),
        Complex::new(-0.45, -0.15),
        Complex::new(0.30, -0.40),
        Complex::new(0.55, 0.05),
    ])
}

fn genfmt_spin_reference_table() -> Array2<Complex> {
    arr2(&[
        [Complex::new(0.10, 0.20), Complex::new(0.30, 0.40)],
        [Complex::new(-0.20, 0.10), Complex::new(-0.10, 0.30)],
        [Complex::new(0.50, -0.20), Complex::new(0.70, -0.10)],
    ])
}

fn genfmt_spin_angular_limits() -> Array2<usize> {
    arr2(&[[1, 2], [0, 1]])
}

fn genfmt_spin_phase_shift_table() -> Array4<Complex> {
    let mut phase_shifts = Array4::zeros((2, 5, 2, 2).f());
    for energy in 0..2 {
        let ef = energy as Real;
        for signed_l in 0..5 {
            let lf = signed_l as Real;
            for spin in 0..2 {
                let sf = spin as Real;
                for potential in 0..2 {
                    let pf = potential as Real;
                    phase_shifts[(energy, signed_l, spin, potential)] = Complex::new(
                        100.0 * ef + 10.0 * lf + sf + 0.1 * pf,
                        -50.0 * ef + 5.0 * lf - 2.0 * sf - 0.1 * pf,
                    );
                }
            }
        }
    }
    phase_shifts
}

fn genfmt_central_phase_shift_table() -> Array3<Complex> {
    let mut phase_shifts = Array3::zeros((3, 7, 2).f());
    for energy in 0..3 {
        let ef = energy as Real;
        for signed_l in 0..7 {
            let lf = signed_l as Real;
            for potential in 0..2 {
                let pf = potential as Real;
                phase_shifts[(energy, signed_l, potential)] =
                    Complex::new(100.0 * ef + 10.0 * lf + pf, -100.0 * ef - 10.0 * lf - pf);
            }
        }
    }
    phase_shifts
}

fn genfmt_spin_radial_factor_table() -> Array3<Complex> {
    let mut radial_factors = Array3::zeros((2, 3, 2).f());
    for energy in 0..2 {
        let ef = energy as Real;
        for transition in 0..3 {
            let tf = transition as Real;
            for spin in 0..2 {
                let sf = spin as Real;
                radial_factors[(energy, transition, spin)] = Complex::new(
                    100.0 * ef + 10.0 * tf + sf,
                    -50.0 * ef + 5.0 * tf - 2.0 * sf,
                );
            }
        }
    }
    radial_factors
}

fn genfmt_jas_spin_radial_factor_table() -> Array4<Complex> {
    let mut radial_factors = Array4::zeros((2, 2, 3, 2).f());
    for energy in 0..2 {
        let ef = energy as Real;
        for q_index in 0..2 {
            let qf = q_index as Real;
            for transition in 0..3 {
                let tf = transition as Real;
                for spin in 0..2 {
                    let sf = spin as Real;
                    radial_factors[(energy, q_index, transition, spin)] = Complex::new(
                        1000.0 * ef + 100.0 * qf + 10.0 * tf + sf,
                        -500.0 * ef + 50.0 * qf + 5.0 * tf - 2.0 * sf,
                    );
                }
            }
        }
    }
    radial_factors
}

fn genfmt_decomposed_reference_chi() -> Array3<Complex> {
    let channels: [([Real; 3], [Real; 3]); 4] = [
        ([1.0, 1.25, 1.5], [0.2, 0.4, 0.6]),
        ([0.9, 1.1, 1.3], [3.0, -3.05, -2.9]),
        ([0.7, 0.8, 0.9], [-0.5, -0.7, -0.9]),
        ([1.4, 1.2, 1.0], [1.0, 2.0, -3.0]),
    ];
    let mut table = Array3::zeros((2, 2, 3).f());
    for row in 0..2 {
        for column in 0..2 {
            let (amplitudes, phases) = channels[row * 2 + column];
            for energy in 0..3 {
                table[(row, column, energy)] = Complex::new(
                    amplitudes[energy] * phases[energy].cos(),
                    amplitudes[energy] * phases[energy].sin(),
                );
            }
        }
    }
    table
}

fn genfmt_ordinary_finalization_fixture(
    path_index: usize,
    keep: bool,
    normalization: Real,
) -> GenfmtOrdinaryPathFinalization {
    GenfmtOrdinaryPathFinalization {
        signals: GenfmtPathSignals {
            contributions: Array2::zeros((1, 1).f()),
            chi: Array1::from_vec(vec![Complex::new(path_index as Real, 0.0)]),
        },
        output_decision: genfmt_output_decision_fixture(path_index, keep, normalization),
    }
}

fn genfmt_jas_finalization_fixture(
    path_index: usize,
    keep: bool,
    normalization: Real,
    decomposed: bool,
) -> GenfmtJasPathFinalization {
    GenfmtJasPathFinalization {
        signals: GenfmtJasPathSignals {
            chi: Array1::from_vec(vec![Complex::new(path_index as Real, 0.0)]),
            decomposed_chi: decomposed.then(genfmt_output_decomposition_chi_fixture),
            decomposed_sums: decomposed
                .then(|| Array1::from_vec(vec![Complex::new(path_index as Real, 0.0)])),
        },
        output_decision: genfmt_output_decision_fixture(path_index, keep, normalization),
        decomposed_output: (keep && decomposed)
            .then(|| genfmt_decomposed_output_fixture(path_index)),
    }
}

fn genfmt_output_decision_fixture(
    path_index: usize,
    keep: bool,
    normalization: Real,
) -> GenfmtPathOutputDecision {
    GenfmtPathOutputDecision {
        summary: GenfmtPathOutputSummary {
            path_index,
            retained: keep,
            criterion_percent: if keep { 100.0 } else { 1.0 },
            degeneracy: 1.0,
            leg_count: 1,
            effective_half_path_length_bohr: path_index as Real,
            effective_half_path_length_angstrom: path_index as Real * 0.529_177_249,
        },
        importance: GenfmtPathImportance {
            magnitudes: Array1::from_vec(vec![path_index as Real]),
            raw_importance: path_index as Real,
            normalization,
            percent: if keep { 100.0 } else { 1.0 },
        },
        retention: GenfmtPathRetention {
            discard_threshold_percent: Some(2.0),
            keep,
        },
        retained_output: keep.then(|| genfmt_retained_output_fixture(path_index)),
    }
}

fn genfmt_retained_output_fixture(path_index: usize) -> GenfmtRetainedPathOutput {
    GenfmtRetainedPathOutput {
        path_index,
        degeneracy: 1.0,
        criterion_percent: 100.0,
        effective_half_path_length_bohr: path_index as Real,
        effective_half_path_length_angstrom: path_index as Real * 0.529_177_249,
        list_sigma2: 0.0,
        potential_indices: Array1::from_vec(vec![0]),
        positions: arr2(&[[0.0, 0.0, 0.0]]),
        beta_angles: Array1::from_vec(vec![0.0]),
        eta_angles: Array1::from_vec(vec![0.0]),
        leg_lengths: Array1::from_vec(vec![0.0]),
        amplitudes: Array1::from_vec(vec![path_index as Real]),
        phases: Array1::from_vec(vec![0.0]),
    }
}

fn genfmt_decomposed_output_fixture(path_index: usize) -> GenfmtDecomposedChiAmplitudePhase {
    GenfmtDecomposedChiAmplitudePhase {
        amplitudes: Array3::from_elem((1, 1, 1).f(), path_index as Real),
        phases: Array3::from_elem((1, 1, 1).f(), 0.0),
    }
}

fn genfmt_output_decomposition_chi_fixture() -> Array3<Complex> {
    Array3::from_elem((1, 1, 1).f(), Complex::new(1.0, 0.0))
}

struct Mmtrjas0ReferenceData {
    transition_angular_momenta: Array1<i32>,
    first_rotation: Array3<Real>,
    last_rotation: Array3<Real>,
}

impl Mmtrjas0ReferenceData {
    fn input(&self) -> JasSpinTransitionInput<'_> {
        JasSpinTransitionInput {
            initial_kappa: -1,
            initial_j2: 3,
            spin_channels: 2,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            final_lj_max: 3,
            final_j2_max: 5,
            max_angular_momentum: 2,
            first_rotation: self.first_rotation.view(),
            last_rotation: self.last_rotation.view(),
            rotation_magnetic_offset: 3,
            first_eta: 0.23,
            last_eta: 0.41,
        }
    }
}

fn mmtrjas0_reference_data() -> Mmtrjas0ReferenceData {
    Mmtrjas0ReferenceData {
        transition_angular_momenta: Array1::from_vec(vec![0, 1, 1, 2]),
        first_rotation: mmtrjas0_first_rotation_table(),
        last_rotation: mmtrjas0_last_rotation_table(),
    }
}

fn mmtrjas0_first_rotation_table() -> Array3<Real> {
    let mut rotation = Array3::zeros((3, 7, 7).f());
    for l in 0..=2 {
        let il = (l + 1) as Real;
        for mu1 in -3_i32..=3 {
            for m1 in -3_i32..=3 {
                if i32_abs_usize(mu1) <= l && i32_abs_usize(m1) <= l {
                    rotation[(l, (mu1 + 3) as usize, (m1 + 3) as usize)] =
                        (0.13 * il + 0.07 * (mu1 as Real) - 0.05 * (m1 as Real) + 0.17).cos();
                }
            }
        }
    }
    rotation
}

fn mmtrjas0_last_rotation_table() -> Array3<Real> {
    let mut rotation = Array3::zeros((3, 7, 7).f());
    for l in 0..=2 {
        let il = (l + 1) as Real;
        for m2 in -3_i32..=3 {
            for mu2 in -3_i32..=3 {
                if i32_abs_usize(m2) <= l && i32_abs_usize(mu2) <= l {
                    rotation[(l, (m2 + 3) as usize, (mu2 + 3) as usize)] =
                        (0.19 * il - 0.04 * (m2 as Real) + 0.06 * (mu2 as Real) - 0.11).cos();
                }
            }
        }
    }
    rotation
}

struct MmtrjasReferenceData {
    transition_angular_momenta: Array1<i32>,
    final_lg_momenta: Array1<i32>,
    final_lj_momenta: Array1<i32>,
    q_phases: Array1<Complex>,
    q_beta_angles: Array1<Real>,
    first_rotation: Array3<Real>,
    last_rotation: Array3<Real>,
}

impl MmtrjasReferenceData {
    fn input(&self) -> JasOneSidedTransitionInput<'_> {
        JasOneSidedTransitionInput {
            initial_kappa: -1,
            initial_j2: 3,
            spin_channels: 2,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            final_lg_momenta: self.final_lg_momenta.view(),
            final_lj_momenta: self.final_lj_momenta.view(),
            final_lj_max: 3,
            final_j2_max: 5,
            max_angular_momentum: 2,
            q_phases: self.q_phases.view(),
            q_beta_angles: self.q_beta_angles.view(),
            first_rotation: self.first_rotation.view(),
            last_rotation: self.last_rotation.view(),
            rotation_magnetic_offset: 3,
            first_eta: 0.23,
            last_eta: 0.41,
        }
    }
}

fn mmtrjas_reference_data() -> MmtrjasReferenceData {
    MmtrjasReferenceData {
        transition_angular_momenta: Array1::from_vec(vec![0, 1, 1, 2]),
        final_lg_momenta: Array1::from_vec(vec![0, 1, 1, 2]),
        final_lj_momenta: Array1::from_vec(vec![0, 1, 1, 2]),
        q_phases: Array1::from_vec(vec![
            (Complex::new(0.0, 1.0) * 0.20).exp(),
            (Complex::new(0.0, 1.0) * -0.35).exp(),
            (Complex::new(0.0, 1.0) * 0.55).exp(),
        ]),
        q_beta_angles: Array1::from_vec(vec![0.25, 0.80, 1.15]),
        first_rotation: mmtrjas0_first_rotation_table(),
        last_rotation: mmtrjas0_last_rotation_table(),
    }
}

fn jas_setup_radial_grid(
    energy_count: usize,
    q_count: usize,
    transition_count: usize,
) -> Array3<Complex> {
    let mut radial_factors = Array3::zeros((energy_count, q_count, transition_count).f());
    for energy in 0..energy_count {
        let ie = (energy + 1) as Real;
        for q in 0..q_count {
            let iq = (q + 1) as Real;
            for transition in 0..transition_count {
                let k = (transition + 1) as Real;
                radial_factors[(energy, q, transition)] = Complex::new(
                    0.08 * ie + 0.03 * iq - 0.01 * k,
                    -0.04 * ie + 0.02 * iq + 0.015 * k,
                );
            }
        }
    }
    radial_factors
}

struct Mmtrxijas0ReferenceData {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array2<Complex>,
    q_weights: Array1<Complex>,
    transition_matrix: Array5<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    xnlm: Array2<Real>,
}

impl Mmtrxijas0ReferenceData {
    fn input(&self) -> JasScatteringAmplitudeInput<'_> {
        JasScatteringAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 4,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            q_weights: self.q_weights.view(),
            transition_matrix: self.transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            xnlm: self.xnlm.view(),
            eta: 0.37,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn energy_point_branch_input(&self) -> GenfmtJasPathEnergyBranchInput<'_> {
        self.energy_point_branch_input_with_radial(self.radial_factors.view())
    }

    fn energy_point_branch_input_with_radial<'a>(
        &'a self,
        radial_factors: ndarray::ArrayView2<'a, Complex>,
    ) -> GenfmtJasPathEnergyBranchInput<'a> {
        GenfmtJasPathEnergyBranchInput::Spherical {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors,
            q_weights: self.q_weights.view(),
            transition_matrix: self.transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn energy_grid_branch_input<'a>(
        &'a self,
        radial_factors: ndarray::ArrayView3<'a, Complex>,
    ) -> GenfmtJasPathEnergyGridBranchInput<'a> {
        GenfmtJasPathEnergyGridBranchInput::Spherical {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors,
            q_weights: self.q_weights.view(),
            transition_matrix: self.transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }
}

fn mmtrxijas0_reference_data() -> Mmtrxijas0ReferenceData {
    let m_indices = Array1::from_vec(vec![0, 1, -1, 2]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 1]);
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2]);
    let q_weights = Array1::from_vec(vec![
        Complex::new(0.2, 0.0),
        Complex::new(0.3, 0.0),
        Complex::new(0.5, 0.0),
    ]);

    let mut radial_factors = Array2::zeros((3, 3).f());
    for q in 0..3 {
        let iq = (q + 1) as Real;
        for transition in 0..3 {
            let k = (transition + 1) as Real;
            radial_factors[(q, transition)] = Complex::new(
                0.08 * 2.0 + 0.03 * iq - 0.01 * k,
                -0.04 * 2.0 + 0.02 * iq + 0.015 * k,
            );
        }
    }

    let mut transition_matrix = Array5::zeros((4, 2, 5, 5, 3).f());
    for transition in 0..3 {
        let k = (transition + 1) as Real;
        for mu1 in -2_i32..=2 {
            for mu2 in -2_i32..=2 {
                for spin in 0..=1 {
                    for mj_row in 0..4 {
                        let mj = -3 + 2 * (mj_row as i32);
                        transition_matrix[(
                            mj_row,
                            spin,
                            (mu2 + 2) as usize,
                            (mu1 + 2) as usize,
                            transition,
                        )] = Complex::new(
                            0.01 * (mj as Real) + 0.02 * (spin as Real) - 0.015 * (mu2 as Real)
                                + 0.017 * (mu1 as Real)
                                + 0.03 * k,
                            -0.012 * (mj as Real) + 0.018 * (spin as Real) + 0.013 * (mu2 as Real)
                                - 0.011 * (mu1 as Real)
                                + 0.01 * k,
                        );
                    }
                }
            }
        }
    }

    let mut xnlm = Array2::zeros((3, 4).f());
    for l in 0..4 {
        let il = (l + 1) as Real;
        for m in 0..3 {
            let im = (m + 1) as Real;
            xnlm[(m, l)] = 0.9 + 0.17 * il + 0.11 * im;
        }
    }

    Mmtrxijas0ReferenceData {
        m_indices,
        n_indices,
        transition_angular_momenta,
        radial_factors,
        q_weights,
        transition_matrix,
        first_polynomials: jas_polynomial_table(1),
        second_polynomials: jas_polynomial_table(2),
        xnlm,
    }
}

fn jas_polynomial_table(leg: usize) -> Array2<Complex> {
    let mut table = Array2::zeros((4, 6).f());
    let leg = leg as Real;
    for l in 0..4 {
        let il = (l + 1) as Real;
        for column in 0..6 {
            let icol = (column + 1) as Real;
            table[(l, column)] = Complex::new(
                0.04 * il + 0.015 * icol + 0.02 * leg,
                -0.025 * il + 0.012 * icol - 0.01 * leg,
            );
        }
    }
    table
}

struct MmtrxijasReferenceData {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array2<Complex>,
    q_weights: Array1<Complex>,
    left_transition_matrix: Array4<Complex>,
    right_transition_matrix: Array4<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    xnlm: Array2<Real>,
}

impl MmtrxijasReferenceData {
    fn input(&self) -> JasLeftRightAmplitudeInput<'_> {
        JasLeftRightAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 5,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            q_weights: self.q_weights.view(),
            left_transition_matrix: self.left_transition_matrix.view(),
            right_transition_matrix: self.right_transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            xnlm: self.xnlm.view(),
            eta: 0.37,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn energy_point_branch_input(&self) -> GenfmtJasPathEnergyBranchInput<'_> {
        self.energy_point_branch_input_with_radial(self.radial_factors.view())
    }

    fn energy_point_branch_input_with_radial<'a>(
        &'a self,
        radial_factors: ndarray::ArrayView2<'a, Complex>,
    ) -> GenfmtJasPathEnergyBranchInput<'a> {
        GenfmtJasPathEnergyBranchInput::LeftRight {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors,
            q_weights: self.q_weights.view(),
            left_transition_matrix: self.left_transition_matrix.view(),
            right_transition_matrix: self.right_transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn energy_grid_branch_input<'a>(
        &'a self,
        radial_factors: ndarray::ArrayView3<'a, Complex>,
    ) -> GenfmtJasPathEnergyGridBranchInput<'a> {
        GenfmtJasPathEnergyGridBranchInput::LeftRight {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors,
            q_weights: self.q_weights.view(),
            left_transition_matrix: self.left_transition_matrix.view(),
            right_transition_matrix: self.right_transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn energy_point_amplitude_input<'a>(
        &'a self,
        curved_wave_polynomials: &'a Array3<Complex>,
        xnlm: ndarray::ArrayView2<'a, Real>,
    ) -> JasLeftRightAmplitudeInput<'a> {
        JasLeftRightAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 5,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            q_weights: self.q_weights.view(),
            left_transition_matrix: self.left_transition_matrix.view(),
            right_transition_matrix: self.right_transition_matrix.view(),
            initial_j2: 3,
            transition_magnetic_offset: 2,
            first_leg_polynomials: curved_wave_polynomials.index_axis(Axis(2), 0),
            second_leg_polynomials: curved_wave_polynomials.index_axis(Axis(2), 1),
            xnlm,
            eta: 0.31,
            max_angular_momentum: 3,
            decomposition_l_max: Some(2),
        }
    }

    fn jas_scattering_product_input<'a>(
        &'a self,
        common: &'a GenfmtJasEnergyPointCommonData,
        curved_wave_polynomials: &'a Array3<Complex>,
    ) -> GenfmtScatteringPathProductInput<'a> {
        GenfmtScatteringPathProductInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            full_lambda_count: 5,
            initial_lambda_count: 5,
            path_potential_indices: common.path_potential_indices.view(),
            angular_limits: common.angular_limits.index_axis(Axis(0), 0),
            phase_shifts: common.phase_shifts.index_axis(Axis(0), 0),
            signed_angular_offset: 4,
            curved_wave_polynomials: curved_wave_polynomials.view(),
            rotations: common.rotations.view(),
            rotation_magnetic_offset: 4,
            xnlm: common.xnlm.view(),
            eta_angles: common.eta_angles.view(),
        }
    }
}

fn mmtrxijas_reference_data() -> MmtrxijasReferenceData {
    let m_indices = Array1::from_vec(vec![0, 1, -1, 2, 0]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 1, 1]);
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3]);
    let q_weights = Array1::from_vec(vec![
        Complex::new(0.40, 0.10),
        Complex::new(0.25, -0.05),
        Complex::new(0.35, 0.02),
    ]);

    let mut radial_factors = Array2::zeros((3, 4).f());
    for q in 0..3 {
        let iq = (q + 1) as Real;
        for transition in 0..4 {
            let k = (transition + 1) as Real;
            radial_factors[(q, transition)] =
                Complex::new(0.02 + 0.06 * iq + 0.03 * k, -0.025 * iq + 0.014 * k);
        }
    }

    let mut left_transition_matrix = Array4::zeros((4, 5, 3, 4).f());
    let mut right_transition_matrix = Array4::zeros((4, 5, 3, 4).f());
    for transition in 0..4 {
        let k = (transition + 1) as Real;
        for q in 0..3 {
            let iq = (q + 1) as Real;
            for mu in -2_i32..=2 {
                for mj_row in 0..4 {
                    let mj = -3 + 2 * (mj_row as i32);
                    left_transition_matrix[(mj_row, (mu + 2) as usize, q, transition)] =
                        Complex::new(
                            0.01 * (mj as Real) - 0.02 * (mu as Real) + 0.015 * iq + 0.025 * k,
                            -0.012 * (mj as Real) + 0.018 * (mu as Real) - 0.007 * iq + 0.011 * k,
                        );
                    right_transition_matrix[(mj_row, (mu + 2) as usize, q, transition)] =
                        Complex::new(
                            -0.008 * (mj as Real) + 0.017 * (mu as Real) + 0.021 * iq - 0.013 * k,
                            0.009 * (mj as Real) + 0.014 * (mu as Real) + 0.006 * iq + 0.019 * k,
                        );
                }
            }
        }
    }

    let mut xnlm = Array2::zeros((3, 4).f());
    for l in 0..4 {
        let il = (l + 1) as Real;
        for m in 0..3 {
            let im = (m + 1) as Real;
            xnlm[(m, l)] = 0.9 + 0.17 * il + 0.11 * im;
        }
    }

    MmtrxijasReferenceData {
        m_indices,
        n_indices,
        transition_angular_momenta,
        radial_factors,
        q_weights,
        left_transition_matrix,
        right_transition_matrix,
        first_polynomials: jas_polynomial_table(1),
        second_polynomials: jas_polynomial_table(2),
        xnlm,
    }
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

fn assert_complex_array1_close(actual: &Array1<Complex>, expected: &Array1<Complex>) {
    assert_eq!(actual.len(), expected.len());
    for index in 0..actual.len() {
        assert_complex_close(actual[index], expected[index]);
    }
}

fn assert_complex_array2_close(actual: &Array2<Complex>, expected: &Array2<Complex>) {
    assert_eq!(actual.shape(), expected.shape());
    for index in actual.indexed_iter().map(|(index, _)| index) {
        assert_complex_close(actual[index], expected[index]);
    }
}

fn assert_complex_array3_close(actual: &Array3<Complex>, expected: &Array3<Complex>) {
    assert_eq!(actual.shape(), expected.shape());
    for index in actual.indexed_iter().map(|(index, _)| index) {
        assert_complex_close(actual[index], expected[index]);
    }
}

fn genfmt_jas_trace_value(trace: &GenfmtJasPathTrace) -> Complex {
    match trace {
        GenfmtJasPathTrace::LeftRight { trace, .. } => trace.trace,
        GenfmtJasPathTrace::Spherical { trace, .. } => trace.trace,
    }
}

fn genfmt_jas_decomposed_trace_view<'a>(
    trace: &'a GenfmtJasPathTrace,
) -> Option<ndarray::ArrayView2<'a, Complex>> {
    match trace {
        GenfmtJasPathTrace::LeftRight { trace, .. } => {
            trace.decomposed_traces.as_ref().map(|table| table.view())
        }
        GenfmtJasPathTrace::Spherical { trace, .. } => {
            trace.decomposed_traces.as_ref().map(|table| table.view())
        }
    }
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

fn complex3_sum(table: &Array3<Complex>) -> Complex {
    table
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
}

fn complex4_sum(table: &Array4<Complex>) -> Complex {
    table
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
}

fn complex5_sum(table: &Array5<Complex>) -> Complex {
    table
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), |sum, value| sum + value)
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

fn padded_rotation_value(
    rotation: ndarray::ArrayView3<'_, Real>,
    magnetic_offset: usize,
    il: usize,
    m1: isize,
    m2: isize,
) -> f64 {
    let row = (m1 + magnetic_offset as isize) as usize;
    let column = (m2 + magnetic_offset as isize) as usize;
    rotation[(il - 1, row, column)]
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
