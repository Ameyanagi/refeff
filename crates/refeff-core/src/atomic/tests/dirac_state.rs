#![allow(clippy::excessive_precision)]

use super::*;

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_entry_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method0_negative_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: -0.25,
        method: 0,
    })?;
    assert_close_with(method0_negative_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(
        method0_negative_tail.asymptotic_large_component,
        2.5e-1,
        1.0e-18,
    );
    assert_eq!(method0_negative_tail.requested_method, 0);
    assert_eq!(method0_negative_tail.method, 1);

    let method2_positive_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: 0.4,
        method: 2,
    })?;
    assert_close_with(method2_positive_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(
        method2_positive_tail.asymptotic_large_component,
        4.000_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_eq!(method2_positive_tail.requested_method, 2);
    assert_eq!(method2_positive_tail.method, 2);

    let negative_method = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: -0.75,
        method: -3,
    })?;
    assert_close_with(negative_method.previous_energy, 1.0, 1.0e-18);
    assert_close_with(negative_method.asymptotic_large_component, 7.5e-1, 1.0e-18);
    assert_eq!(negative_method.requested_method, -3);
    assert_eq!(negative_method.method, 1);

    let method1_zero_tail = atomic_dirac_entry_state(AtomicDiracEntryStateInput {
        asymptotic_large_component: 0.0,
        method: 1,
    })?;
    assert_close_with(method1_zero_tail.previous_energy, 1.0, 1.0e-18);
    assert_close_with(method1_zero_tail.asymptotic_large_component, 0.0, 1.0e-18);
    assert_eq!(method1_zero_tail.requested_method, 1);
    assert_eq!(method1_zero_tail.method, 1);
    Ok(())
}

#[test]
fn atom_dirac_normalization_matches_feff_soldir_norm_reference() -> Result<(), AtomMathError> {
    let fixture = sample_soldir_norm_fixture();

    let method_one = atomic_dirac_normalization(fixture.input(1, 6, 0.177, 0.82, 11, 5))?;
    assert_close_with(method_one.norm, 5.408_474_263_575_392e-6, 1.0e-18);

    let method_two = atomic_dirac_normalization(fixture.input(2, 8, 0.0, 1.35, 13, 7))?;
    assert_close_with(method_two.norm, 9.499_334_208_495_336e-6, 1.0e-18);
    Ok(())
}
#[test]
fn atom_dirac_solution_normalization_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let kept_fixture = sample_soldir_solution_normalization_fixture(false, false);
    let kept = atomic_dirac_solution_normalization(kept_fixture.input(6.25, 0.8, -0.4))?;
    assert_close_with(kept.component_divisor, 2.5, 1.0e-18);
    assert_close_with(kept.coefficient_divisor, 2.5, 1.0e-18);
    assert_close_with(kept.large_coefficients[0], 8.4e-2, 1.0e-18);
    assert_close_with(kept.small_coefficients[0], -4.28e-2, 1.0e-18);
    assert_close_with(kept.large_coefficients[3], 3.84e-1, 1.0e-18);
    assert_close_with(kept.small_coefficients[3], -1.568e-1, 1.0e-18);
    assert_close_with(kept.large_component[0], 1.64e-2, 1.0e-18);
    assert_close_with(kept.small_component[0], -1.18e-2, 1.0e-18);
    assert_close_with(kept.large_component[6], 1.316e-1, 1.0e-18);
    assert_close_with(kept.large_component[7], 0.0, 1.0e-18);
    assert_close_with(kept.small_component[8], 0.0, 1.0e-18);

    let flipped_fixture = sample_soldir_solution_normalization_fixture(true, true);
    let flipped = atomic_dirac_solution_normalization(flipped_fixture.input(1.44, 0.75, -0.25))?;
    assert_close_with(flipped.component_divisor, -1.2, 1.0e-18);
    assert_close_with(flipped.coefficient_divisor, -1.2, 1.0e-18);
    assert_close_with(
        flipped.large_coefficients[0],
        1.750_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_coefficients[0],
        8.916_666_666_666_667_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_coefficients[3],
        -8.000_000_000_000_000_4e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_coefficients[3],
        3.266_666_666_666_667_2e-1,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_component[0],
        3.416_666_666_666_667_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.small_component[0],
        2.458_333_333_333_333_2e-2,
        1.0e-18,
    );
    assert_close_with(
        flipped.large_component[6],
        -2.741_666_666_666_666_7e-1,
        1.0e-18,
    );
    assert_close_with(flipped.large_component[7], 0.0, 1.0e-18);
    assert_close_with(flipped.small_component[8], 0.0, 1.0e-18);
    Ok(())
}

#[test]
fn atom_dirac_node_count_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = sample_soldir_node_count_component();

    let limited = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 4,
        scan_index_1based: 7,
    })?;
    assert_eq!(limited.scan_index_1based, 7);
    assert_eq!(limited.node_count, 4);

    let matching_extends = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 8,
        scan_index_1based: 3,
    })?;
    assert_eq!(matching_extends.scan_index_1based, 8);
    assert_eq!(matching_extends.node_count, 4);

    let full = atomic_dirac_node_count(AtomicDiracNodeCountInput {
        large_component: large_component.view(),
        matching_index_1based: 1,
        scan_index_1based: 9,
    })?;
    assert_eq!(full.scan_index_1based, 9);
    assert_eq!(full.node_count, 5);
    Ok(())
}
#[test]
fn atom_dirac_node_energy_search_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let too_few_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 0,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_few_scale.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(too_few_scale.energy_sup, -5.0e-1, 1.0e-18);
    assert_close_with(too_few_scale.energy_inf, 1.0, 1.0e-18);
    assert_eq!(too_few_scale.search_attempt_count, 1);
    assert!(too_few_scale.needs_reintegration);
    assert!(!too_few_scale.attempts_exhausted);

    let too_few_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.6,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: -0.2,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 4,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_few_bisect.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(
        too_few_bisect.energy_sup,
        -5.999_999_999_999_999_8e-1,
        1.0e-18,
    );
    assert_close_with(
        too_few_bisect.energy_inf,
        -2.000_000_000_000_000_1e-1,
        1.0e-18,
    );
    assert_eq!(too_few_bisect.search_attempt_count, 5);
    assert!(too_few_bisect.needs_reintegration);
    assert!(!too_few_bisect.attempts_exhausted);

    let too_many_scale = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 5,
        target_node_count: 3,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 7,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_many_scale.energy, -5.999_999_999_999_999_8e-1, 1.0e-18);
    assert_close_with(too_many_scale.energy_sup, -5.0, 1.0e-18);
    assert_close_with(too_many_scale.energy_inf, -5.0e-1, 1.0e-18);
    assert_eq!(too_many_scale.search_attempt_count, 8);
    assert!(too_many_scale.needs_reintegration);
    assert!(!too_many_scale.attempts_exhausted);

    let too_many_bisect = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.4,
        node_count: 5,
        target_node_count: 3,
        energy_sup: -0.7,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 2,
        max_attempt_count: 50,
    })?;
    assert_close_with(too_many_bisect.energy, -5.500_000_000_000_000_4e-1, 1.0e-18);
    assert_close_with(
        too_many_bisect.energy_sup,
        -6.999_999_999_999_999_6e-1,
        1.0e-18,
    );
    assert_close_with(
        too_many_bisect.energy_inf,
        -4.000_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_eq!(too_many_bisect.search_attempt_count, 3);
    assert!(too_many_bisect.needs_reintegration);
    assert!(!too_many_bisect.attempts_exhausted);

    let matched = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.4,
        node_count: 3,
        target_node_count: 3,
        energy_sup: -0.7,
        energy_inf: -0.2,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 2,
        max_attempt_count: 50,
    })?;
    assert_close_with(matched.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(matched.energy_sup, -6.999_999_999_999_999_6e-1, 1.0e-18);
    assert_close_with(matched.energy_inf, -2.000_000_000_000_000_1e-1, 1.0e-18);
    assert_eq!(matched.search_attempt_count, 2);
    assert!(!matched.needs_reintegration);
    assert!(!matched.attempts_exhausted);

    let exhausted = atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
        energy: -0.5,
        node_count: 2,
        target_node_count: 4,
        energy_sup: -5.0,
        energy_inf: 1.0,
        energy_floor: -5.0,
        energy_precision: 1.0e-7,
        search_attempt_count: 1,
        max_attempt_count: 1,
    })?;
    assert_close_with(exhausted.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_close_with(exhausted.energy_sup, -5.0e-1, 1.0e-18);
    assert_close_with(exhausted.energy_inf, 1.0, 1.0e-18);
    assert_eq!(exhausted.search_attempt_count, 2);
    assert!(!exhausted.needs_reintegration);
    assert!(exhausted.attempts_exhausted);

    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -1.0e-8,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: 1.0,
            energy_floor: -5.0,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyTooSmall { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -5.0,
            node_count: 5,
            target_node_count: 3,
            energy_sup: -5.5,
            energy_inf: 1.0,
            energy_floor: -5.5,
            energy_precision: 1.0e-7,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyBelowPotentialFloor { .. })
    ));
    assert!(matches!(
        atomic_dirac_node_energy_search(AtomicDiracNodeEnergySearchInput {
            energy: -0.5,
            node_count: 2,
            target_node_count: 4,
            energy_sup: -5.0,
            energy_inf: -0.500_000_05,
            energy_floor: -5.0,
            energy_precision: 1.0e-6,
            search_attempt_count: 0,
            max_attempt_count: 50,
        }),
        Err(AtomMathError::DiracNodeEnergyBracketCollapsed { .. })
    ));
    Ok(())
}
#[test]
fn atom_dirac_energy_correction_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = Array1::from_vec(vec![0.12, -0.22, 0.31, 0.27, -0.18]);
    let small_component = Array1::from_vec(vec![-0.011, 0.024, 0.047, -0.018, 0.009]);

    let scaled =
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 2.6,
            large_component: large_component.view(),
            small_component: small_component.view(),
            matching_small_component: 0.052,
            matching_index_1based: 3,
        })?;
    assert_close_with(scaled.correction, 8.169_531_346_153_841_0e-2, 1.0e-16);
    assert_close_with(scaled.mismatch, 9.615_384_615_384_610_4e-2, 1.0e-16);

    let zero_matching =
        atomic_dirac_method_one_energy_correction(AtomicDiracMethodOneEnergyCorrectionInput {
            speed_of_light: 137.0373,
            norm: 1.9,
            large_component: large_component.view(),
            small_component: small_component.view(),
            matching_small_component: 0.0,
            matching_index_1based: 4,
        })?;
    assert_close_with(
        zero_matching.correction,
        3.505_269_884_210_525_7e-1,
        1.0e-16,
    );
    assert_close_with(zero_matching.mismatch, 1.8e-2, 1.0e-18);

    let accepted = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -0.5,
        correction: -0.02,
        mismatch: 0.001,
        energy_sup: -0.8,
        energy_inf: -0.2,
        mismatch_precision: 0.01,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(accepted.energy, -5.2e-1, 1.0e-18);
    assert_close_with(accepted.correction, -2.0e-2, 1.0e-18);
    assert_close_with(accepted.relative_step, 4.0e-2, 1.0e-18);
    assert!(!accepted.needs_rematch);

    let positive_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -0.05,
        correction: 0.08,
        mismatch: 0.001,
        energy_sup: -0.8,
        energy_inf: -0.02,
        mismatch_precision: 0.01,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(positive_halved.energy, -4.0e-2, 1.0e-18);
    assert_close_with(positive_halved.correction, 1.0e-2, 1.0e-18);
    assert_close_with(
        positive_halved.relative_step,
        1.999_999_999_999_999_8e-1,
        1.0e-18,
    );
    assert!(!positive_halved.needs_rematch);

    let bound_halved = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -1.0,
        correction: 0.30,
        mismatch: 0.4,
        energy_sup: -1.2,
        energy_inf: -0.8,
        mismatch_precision: 0.1,
        zero_energy_precision: 1.0e-7,
    })?;
    assert_close_with(bound_halved.energy, -8.5e-1, 1.0e-18);
    assert_close_with(bound_halved.correction, 1.5e-1, 1.0e-18);
    assert_close_with(bound_halved.relative_step, 1.5e-1, 1.0e-18);
    assert!(bound_halved.needs_rematch);

    let too_small = atomic_dirac_energy_step(AtomicDiracEnergyStepInput {
        energy: -1.0,
        correction: 1.0e-9,
        mismatch: 1.0,
        energy_sup: -0.5,
        energy_inf: -0.6,
        mismatch_precision: 0.1,
        zero_energy_precision: 1.0e-7,
    });
    let Err(AtomMathError::DiracEnergyCorrectionTooSmall { relative_step }) = too_small else {
        return Err(AtomMathError::NonFiniteScalar {
            field: "soldir_energy_too_small_reference",
            value: 0.0,
        });
    };
    assert_close_with(relative_step, 5.0e-10, 1.0e-24);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_iteration_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method1 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 1,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method1.mismatch_precision,
        1.000_000_000_000_000_1e-5,
        1.0e-20,
    );
    assert_close_with(method1.energy_inf, 1.0, 1.0e-18);
    assert_close_with(method1.energy_sup, -0.75, 1.0e-18);
    assert_close_with(method1.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);
    assert_eq!(method1.match_attempt_count, 0);
    assert_eq!(method1.node_count, 0);
    assert_eq!(method1.search_attempt_count, 0);

    let method2 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 2,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method2.mismatch_precision,
        2.000_000_000_000_000_2e-5,
        1.0e-20,
    );
    assert_close_with(method2.energy_inf, 1.0, 1.0e-18);
    assert_close_with(method2.energy_sup, -0.75, 1.0e-18);
    assert_close_with(method2.energy, -4.000_000_000_000_000_2e-1, 1.0e-18);

    let method0 = atomic_dirac_iteration_reset(AtomicDiracIterationResetInput {
        method: 0,
        primary_matching_precision: 1.0e-5,
        secondary_matching_precision: 2.0e-5,
        energy_floor: -0.75,
        reference_energy: -0.4,
    })?;
    assert_close_with(
        method0.mismatch_precision,
        1.000_000_000_000_000_1e-5,
        1.0e-20,
    );

    let homogeneous = atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
        requested_method: 0,
        method: 1,
    })?;
    assert_eq!(homogeneous.method, 1);
    assert!(!homogeneous.needs_restart);

    let method1_retry =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 1,
            method: 1,
        })?;
    assert_eq!(method1_retry.method, 2);
    assert!(method1_retry.needs_restart);

    let method2_stop = atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
        requested_method: 1,
        method: 2,
    })?;
    assert_eq!(method2_stop.method, 2);
    assert!(!method2_stop.needs_restart);

    let negative_retry =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: -1,
            method: 1,
        })?;
    assert_eq!(negative_retry.method, 2);
    assert!(negative_retry.needs_restart);

    let requested2_current1 =
        atomic_dirac_abnormal_exit_recovery(AtomicDiracAbnormalExitRecoveryInput {
            requested_method: 2,
            method: 1,
        })?;
    assert_eq!(requested2_current1.method, 2);
    assert!(requested2_current1.needs_restart);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_loop_state_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let far_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: 1.0,
    })?;
    assert_eq!(
        far_energy.integration_mode,
        AtomicDiracIntegrationMode::SearchMatchingPoint
    );
    assert!(!far_energy.relocated);
    assert_close_with(far_energy.reference_energy, -5.0e-1, 1.0e-18);
    assert_close_with(far_energy.relative_energy_change, 3.0, 1.0e-18);

    let near_energy = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: -0.54,
    })?;
    assert_eq!(
        near_energy.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert!(!near_energy.relocated);
    assert_close_with(near_energy.reference_energy, -5.0e-1, 1.0e-18);
    assert_close_with(
        near_energy.relative_energy_change,
        8.000_000_000_000_007_1e-2,
        1.0e-17,
    );

    let far_negative = atomic_dirac_shooting_pass_setup(AtomicDiracShootingPassSetupInput {
        energy: -0.5,
        previous_energy: -0.42,
    })?;
    assert_eq!(
        far_negative.integration_mode,
        AtomicDiracIntegrationMode::SearchMatchingPoint
    );
    assert_close_with(
        far_negative.relative_energy_change,
        1.600_000_000_000_000_3e-1,
        1.0e-17,
    );

    let below_test = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.005,
        mismatch_precision: 0.01,
        match_attempt_count: 3,
        max_attempt_count: 5,
    })?;
    assert_eq!(below_test.match_attempt_count, 3);
    assert!(!below_test.needs_rematch);
    assert!(!below_test.attempts_exhausted);

    let retry_left = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.02,
        mismatch_precision: 0.01,
        match_attempt_count: 4,
        max_attempt_count: 5,
    })?;
    assert_eq!(retry_left.match_attempt_count, 5);
    assert!(retry_left.needs_rematch);
    assert!(!retry_left.attempts_exhausted);

    let exhausted = atomic_dirac_rematch_attempt(AtomicDiracRematchAttemptInput {
        mismatch: 0.02,
        mismatch_precision: 0.01,
        match_attempt_count: 5,
        max_attempt_count: 5,
    })?;
    assert_eq!(exhausted.match_attempt_count, 6);
    assert!(!exhausted.needs_rematch);
    assert!(exhausted.attempts_exhausted);
    Ok(())
}
