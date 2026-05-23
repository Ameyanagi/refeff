use super::{support::*, *};

#[test]
fn path_criteria_match_feff_references() -> Result<(), PathError> {
    let path_indices = [1, 2, 3, 4];
    let leg_distances = [1.10, 1.25, 1.40, 1.60, 1.20];
    let angle_cosines = [0.80, -0.35, 0.55, -0.10, 0.25];
    let beta_indices = [-3, 4, 10, -2, 0];
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_table();
    let wave_numbers = [2.0, 3.5, 5.0];
    let mean_free_paths = [7.5, 10.0, 12.0];

    assert_option_close(
        path_heap_criterion(
            &path_indices,
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        )?,
        Some(4.501_763_821),
    );

    let initialized = path_output_criterion(PathOutputCriterionInput {
        path_indices: &path_indices,
        leg_distances: &leg_distances,
        angle_cosines: &angle_cosines,
        beta_indices: &beta_indices,
        atom_potentials: &atom_potentials,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        current_normalization: -1.0,
    })?;
    assert_option_close(initialized.output_importance, Some(100.0));
    assert_close(initialized.normalization, 9.197_448_526E-05);

    let fixed_angle_cosines = [0.80, -0.35, 0.55, -0.10, -0.80];
    let fixed = path_output_criterion(PathOutputCriterionInput {
        path_indices: &path_indices,
        leg_distances: &leg_distances,
        angle_cosines: &fixed_angle_cosines,
        beta_indices: &beta_indices,
        atom_potentials: &atom_potentials,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        current_normalization: 0.004,
    })?;
    assert_option_close(fixed.output_importance, Some(6.131_631_851));
    assert_close(fixed.normalization, 4.000_000_190E-03);

    let central_path = [1, 2, 3, 0];
    assert_eq!(
        path_heap_criterion(
            &central_path,
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        )?,
        None
    );
    let central = path_output_criterion(PathOutputCriterionInput {
        path_indices: &central_path,
        leg_distances: &leg_distances,
        angle_cosines: &fixed_angle_cosines,
        beta_indices: &beta_indices,
        atom_potentials: &atom_potentials,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        current_normalization: 0.004,
    })?;
    assert_eq!(central.output_importance, None);
    assert_close(central.normalization, 4.000_000_190E-03);

    let short_path = [1, 2];
    assert_eq!(
        path_heap_criterion(
            &short_path,
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        )?,
        None
    );
    Ok(())
}

#[test]
fn path_criteria_reject_invalid_inputs() {
    let path_indices = [1, 2, 3, 4];
    let leg_distances = [1.10, 1.25, 1.40, 1.60, 1.20];
    let angle_cosines = [0.80, -0.35, 0.55, -0.10, 0.25];
    let beta_indices = [-3, 4, 10, -2, 0];
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_table();
    let wave_numbers = [2.0, 3.5, 5.0];
    let mean_free_paths = [7.5, 10.0, 12.0];

    assert!(matches!(
        path_heap_criterion(
            &[],
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        ),
        Err(PathError::EmptyPathCriteria)
    ));
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &leg_distances[..4],
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        ),
        Err(PathError::PathCriteriaLengthMismatch {
            expected: 5,
            leg_distances: 4,
            beta_entries: 5
        })
    ));

    let bad_table = Array3::zeros((80, 4, 3));
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            bad_table.view(),
            &wave_numbers,
        ),
        Err(PathError::InvalidPathCriteriaTableShape { beta_rows: 80, .. })
    ));
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &leg_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &[],
        ),
        Err(PathError::PathCriteriaWaveCountMismatch {
            wave_numbers: 0,
            ..
        })
    ));

    let bad_beta_indices = [-41, 4, 10, -2, 0];
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &leg_distances,
            &bad_beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        ),
        Err(PathError::PathCriteriaBetaIndexOutOfRange {
            position: 0,
            beta_index: -41,
            ..
        })
    ));

    let short_potentials = [0, 1, 2];
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &leg_distances,
            &beta_indices,
            &short_potentials,
            fbeta.view(),
            &wave_numbers,
        ),
        Err(PathError::PathCriteriaAtomIndexOutOfRange {
            position: 2,
            atom_index: 3,
            atoms: 3
        })
    ));

    let bad_distances = [1.10, 0.0, 1.40, 1.60, 1.20];
    assert!(matches!(
        path_heap_criterion(
            &path_indices,
            &bad_distances,
            &beta_indices,
            &atom_potentials,
            fbeta.view(),
            &wave_numbers,
        ),
        Err(PathError::NonPositivePathCriteriaValue {
            quantity: "leg distance",
            index: 1,
            ..
        })
    ));

    assert!(matches!(
        path_output_criterion(PathOutputCriterionInput {
            path_indices: &path_indices,
            leg_distances: &leg_distances,
            angle_cosines: &angle_cosines,
            beta_indices: &beta_indices,
            atom_potentials: &atom_potentials,
            fbeta_critical: fbeta.view(),
            mean_free_paths: &mean_free_paths[..2],
            wave_numbers: &wave_numbers,
            current_normalization: 1.0,
        }),
        Err(PathError::PathCriteriaMeanFreePathCountMismatch {
            wave_numbers: 3,
            mean_free_paths: 2
        })
    ));
}

#[test]
fn path_criteria_decision_matches_feff_ccrit_references() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let path_indices = [1, 2, 3, 4];
    let atom_potentials = reference_atom_potentials();
    let mut cluster_outside = vec![false; atom_potentials.len()];
    cluster_outside[4] = true;
    let fbeta = reference_fbeta_table();
    let wave_numbers = [2.0, 3.5, 5.0];
    let mean_free_paths = [7.5, 10.0, 12.0];

    let keep = path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        cluster_outside: &cluster_outside,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        max_path_length: 20.0,
        heap_cutoff: 0.0,
        output_cutoff: 50.0,
        current_normalization: -1.0,
    })?;
    assert_close(keep.total_path_length, 9.766_139_984);
    assert!(keep.add_to_heap);
    assert!(keep.keep_for_output);
    assert_eq!(keep.heap_importance, None);
    assert_option_close(keep.output_importance, Some(100.0));
    assert_close(keep.normalization, 1.964_455_259E-05);

    let heap_reject = path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        cluster_outside: &cluster_outside,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        max_path_length: 20.0,
        heap_cutoff: 999.0,
        output_cutoff: 50.0,
        current_normalization: 0.004,
    })?;
    assert_close(heap_reject.total_path_length, 9.766_139_984);
    assert!(!heap_reject.add_to_heap);
    assert!(!heap_reject.keep_for_output);
    assert!(heap_reject.heap_importance.is_some());
    assert_eq!(heap_reject.output_importance, None);
    assert_close(heap_reject.normalization, 4.000_000_190E-03);

    let rmax_reject = path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        cluster_outside: &cluster_outside,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        max_path_length: 1.0,
        heap_cutoff: 0.0,
        output_cutoff: 50.0,
        current_normalization: 0.004,
    })?;
    assert_close(rmax_reject.total_path_length, 9.766_139_984);
    assert!(!rmax_reject.add_to_heap);
    assert!(!rmax_reject.keep_for_output);
    assert_close(rmax_reject.normalization, 4.000_000_190E-03);

    let central_path = [1, 2, 0];
    let central = path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: atom_positions.view(),
        path_indices: &central_path,
        atom_potentials: &atom_potentials,
        cluster_outside: &cluster_outside,
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        max_path_length: 20.0,
        heap_cutoff: 0.0,
        output_cutoff: 50.0,
        current_normalization: 0.004,
    })?;
    assert_close(central.total_path_length, 4.658_454_895);
    assert!(central.add_to_heap);
    assert!(!central.keep_for_output);
    assert_close(central.normalization, 4.000_000_190E-03);

    let cluster_block = path_criteria_decision(PathCriteriaDecisionInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        cluster_outside: &[false; 9],
        fbeta_critical: fbeta.view(),
        mean_free_paths: &mean_free_paths,
        wave_numbers: &wave_numbers,
        max_path_length: 20.0,
        heap_cutoff: 0.0,
        output_cutoff: -1.0,
        current_normalization: 0.004,
    })?;
    assert_close(cluster_block.total_path_length, 9.766_139_984);
    assert!(cluster_block.add_to_heap);
    assert!(!cluster_block.keep_for_output);
    assert_close(cluster_block.normalization, 4.000_000_190E-03);
    Ok(())
}

#[test]
fn path_beta_indices_match_ccrit_grid_quantization() -> Result<(), PathError> {
    assert_eq!(
        path_beta_indices(&[0.0, 0.0125, 0.0126, -0.0376, -0.999])?,
        vec![0, 0, 1, -2, -40]
    );
    assert!(matches!(
        path_beta_indices(&[Real::NAN]),
        Err(PathError::NonFinitePathCriteriaValue {
            quantity: "angle cosine",
            index: 0,
            ..
        })
    ));
    Ok(())
}

#[test]
fn path_criteria_decision_rejects_missing_cluster_flags() {
    let atom_positions = mrb_reference_positions();
    let path_indices = [1, 2, 3, 4];
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_table();
    assert!(matches!(
        path_criteria_decision(PathCriteriaDecisionInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            atom_potentials: &atom_potentials,
            cluster_outside: &[false; 4],
            fbeta_critical: fbeta.view(),
            mean_free_paths: &[7.5, 10.0, 12.0],
            wave_numbers: &[2.0, 3.5, 5.0],
            max_path_length: 20.0,
            heap_cutoff: 0.0,
            output_cutoff: -1.0,
            current_normalization: 0.004,
        }),
        Err(PathError::PathCriteriaClusterIndexOutOfRange {
            position: 3,
            atom_index: 4,
            atoms: 4
        })
    ));
}

#[test]
fn path_output_importance_matches_feff_outcrt_references() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let path_indices = [1, 2, 3, 4];
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    let wave_numbers = [1.2, 2.0, 3.25, 4.5, 6.0];
    let mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    let critical_wave_numbers = [2.0, 3.5, 5.0];
    let critical_mean_free_paths = [7.5, 10.0, 12.0];

    let initialized = path_output_importance(PathOutputImportanceInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        current_normalization: -1.0,
    })?;
    assert_close(initialized.port_importance, 1.117_176_271E-05);
    assert_option_close(initialized.heap_importance, Some(1.036_497_688E1));
    assert_option_close(initialized.reversed_heap_importance, Some(2.983_642_340));
    assert_option_close(initialized.output_importance, Some(100.0));
    assert_close(initialized.normalization, 1.964_455_259E-05);

    let fixed = path_output_importance(PathOutputImportanceInput {
        atom_positions: atom_positions.view(),
        path_indices: &path_indices,
        atom_potentials: &atom_potentials,
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        current_normalization: 0.004,
    })?;
    assert_close(fixed.port_importance, 1.117_176_271E-05);
    assert_option_close(fixed.output_importance, Some(4.911_137_819E-1));
    assert_close(fixed.normalization, 4.000_000_190E-03);

    let two_leg = path_output_importance(PathOutputImportanceInput {
        atom_positions: atom_positions.view(),
        path_indices: &[1, 2],
        atom_potentials: &atom_potentials,
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        current_normalization: 0.004,
    })?;
    assert_close(two_leg.port_importance, 7.728_009_950E-03);
    assert_eq!(two_leg.heap_importance, None);
    assert_eq!(two_leg.reversed_heap_importance, None);
    assert_option_close(two_leg.output_importance, Some(2.475_754_242E2));
    assert_close(two_leg.normalization, 4.000_000_190E-03);
    Ok(())
}

#[test]
fn path_output_importance_rejects_invalid_start_energy() {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    assert!(matches!(
        path_output_importance(PathOutputImportanceInput {
            atom_positions: atom_positions.view(),
            path_indices: &[1, 2, 3, 4],
            atom_potentials: &atom_potentials,
            fbeta: fbeta.view(),
            wave_numbers: &[1.2, 2.0, 3.25],
            mean_free_paths: &[6.0, 7.5, 9.0],
            start_energy_index: 2,
            fbeta_critical: fbetac.view(),
            critical_wave_numbers: &[2.0, 3.5, 5.0],
            critical_mean_free_paths: &[7.5, 10.0, 12.0],
            current_normalization: 0.004,
        }),
        Err(PathError::PathImportanceStartOutOfRange {
            start: 2,
            remaining: 1
        })
    ));
}
