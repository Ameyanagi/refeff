use super::{support::*, *};

#[test]
fn path_degeneracy_groups_match_feff_pathsd_hash_range() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let first_bounce_degeneracies = [0, 2, 3, 4, 5, 1, 7];
    let candidates = [
        PathDegeneracyCandidate {
            path_indices: &[1, 2, 3, 4],
        },
        PathDegeneracyCandidate {
            path_indices: &[4, 3, 2, 1],
        },
    ];

    let groups = path_degeneracy_groups(PathDegeneracyGroupsInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        first_bounce_degeneracies: &first_bounce_degeneracies,
        candidates: &candidates,
        polarization: 0,
        spin: 0,
        electric_vector: [0.0, 0.0, 1.0],
        incident_vector: [0.0, 0.0, 0.0],
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].path_indices, vec![1, 2, 3, 4]);
    assert_eq!(groups[0].degeneracy, 7);
    assert_eq!(groups[0].member_count, 2);
    assert_hash_close(groups[0].degeneracy_hash, 1.540_019_626_331_394E8);
    assert_standard_coordinates(
        PathStandardCoordinates {
            coordinates: groups[0].coordinates.clone(),
            symmetry_case: groups[0].symmetry_case,
        },
        1,
        &[
            [1.054_711_873E-15, -1.387_778_781E-17, 1.118_034_005],
            [7.429_670_095E-1, -0.0, 2.146_625_280],
            [1.646_371_841, 6.958_302_259E-1, -1.878_297_031E-1],
            [-6.697_471_142E-1, -1.377_214_193, 4.740_463_793E-1],
        ],
    );
    Ok(())
}

#[test]
fn path_degeneracy_groups_keep_spin_polarized_directions_separate() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let first_bounce_degeneracies = [0, 2, 3, 4, 5, 1, 7];
    let candidates = [
        PathDegeneracyCandidate {
            path_indices: &[1, 2, 3, 4],
        },
        PathDegeneracyCandidate {
            path_indices: &[4, 3, 2, 1],
        },
    ];

    let groups = path_degeneracy_groups(PathDegeneracyGroupsInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        first_bounce_degeneracies: &first_bounce_degeneracies,
        candidates: &candidates,
        polarization: 1,
        spin: 1,
        electric_vector: [0.0, 0.0, 1.0],
        incident_vector: [0.0, 0.0, 0.0],
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;

    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].path_indices, vec![1, 2, 3, 4]);
    assert_eq!(groups[0].degeneracy, 2);
    assert_eq!(groups[0].member_count, 1);
    assert_eq!(groups[1].path_indices, vec![4, 3, 2, 1]);
    assert_eq!(groups[1].degeneracy, 5);
    assert_eq!(groups[1].member_count, 1);
    assert_hash_close(groups[0].degeneracy_hash, 1.597_067_314_650_102E8);
    assert_hash_close(groups[1].degeneracy_hash, 1.669_436_988_304_592E8);
    Ok(())
}

#[test]
fn path_degeneracy_groups_reject_missing_first_bounce_degeneracy() {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let candidates = [PathDegeneracyCandidate { path_indices: &[4] }];

    assert!(matches!(
        path_degeneracy_groups(PathDegeneracyGroupsInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            first_bounce_degeneracies: &[0, 1],
            candidates: &candidates,
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
            force_no_symmetry: false,
        }),
        Err(PathError::PathDegeneracyFirstBounceOutOfRange {
            candidate: 0,
            atom_index: 4,
            atoms: 2,
        })
    ));
}

#[test]
fn path_degeneracy_retention_matches_feff_pathsd_fraction_filter() -> Result<(), PathError> {
    let groups = [sample_group(4), sample_group(2), sample_group(6)];

    let retention = path_degeneracy_retention(PathDegeneracyRetentionInput {
        groups: &groups,
        port_importances: &[2.0, 0.05, 0.5],
        criterion_percent: 10.0,
        initial_reference: None,
    })?;

    assert_eq!(retention.retained_unique_count, 2);
    assert_eq!(retention.retained_total_degeneracy, 10);
    assert_eq!(retention.reference_group_index, Some(0));
    assert_eq!(retention.reference_degeneracy, Some(4));
    assert_eq!(retention.reference_port_importance, Some(2.0));
    assert_eq!(
        retention.reference,
        Some(PathDegeneracyRetentionReference {
            port_importance: 2.0,
            degeneracy: 4,
        })
    );
    assert_eq!(
        retention.decisions,
        vec![
            PathDegeneracyRetentionDecision {
                group_index: 0,
                port_importance: 2.0,
                fraction_percent: 100.0,
                retained: true,
            },
            PathDegeneracyRetentionDecision {
                group_index: 1,
                port_importance: 0.05,
                fraction_percent: 1.25,
                retained: false,
            },
            PathDegeneracyRetentionDecision {
                group_index: 2,
                port_importance: 0.5,
                fraction_percent: 37.5,
                retained: true,
            },
        ]
    );
    Ok(())
}

#[test]
fn path_degeneracy_retention_rejects_invalid_inputs() {
    let groups = [sample_group(4)];

    assert!(matches!(
        path_degeneracy_retention(PathDegeneracyRetentionInput {
            groups: &groups,
            port_importances: &[],
            criterion_percent: 10.0,
            initial_reference: None,
        }),
        Err(PathError::PathDegeneracyRetentionLengthMismatch {
            groups: 1,
            port_importances: 0,
        })
    ));

    assert!(matches!(
        path_degeneracy_retention(PathDegeneracyRetentionInput {
            groups: &groups,
            port_importances: &[Real::NAN],
            criterion_percent: 10.0,
            initial_reference: None,
        }),
        Err(PathError::NonFinitePathDegeneracyRetentionValue {
            quantity: "port_importance",
            index: 0,
            ..
        })
    ));

    assert!(matches!(
        path_degeneracy_retention(PathDegeneracyRetentionInput {
            groups: &[sample_group(0)],
            port_importances: &[1.0],
            criterion_percent: 10.0,
            initial_reference: None,
        }),
        Err(PathError::NonPositivePathDegeneracy {
            index: 0,
            degeneracy: 0,
        })
    ));

    assert!(matches!(
        path_degeneracy_retention(PathDegeneracyRetentionInput {
            groups: &groups,
            port_importances: &[0.0],
            criterion_percent: 10.0,
            initial_reference: None,
        }),
        Err(PathError::ZeroPathDegeneracyRetentionReference { index: 0 })
    ));

    assert!(matches!(
        path_degeneracy_retention(PathDegeneracyRetentionInput {
            groups: &groups,
            port_importances: &[1.0],
            criterion_percent: 10.0,
            initial_reference: Some(PathDegeneracyRetentionReference {
                port_importance: 0.0,
                degeneracy: 4,
            }),
        }),
        Err(PathError::InvalidPathDegeneracyRetentionReference {
            quantity: "port_importance",
            ..
        })
    ));
}

#[test]
fn path_degeneracy_range_composes_pathsd_grouping_outcrt_and_retention() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let first_bounce_degeneracies = [0, 2, 3, 4, 5, 1, 7];
    let candidates = [
        PathDegeneracyCandidate {
            path_indices: &[1, 2, 3, 4],
        },
        PathDegeneracyCandidate {
            path_indices: &[4, 3, 2, 1],
        },
    ];
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    let wave_numbers = [1.2, 2.0, 3.25, 4.5, 6.0];
    let mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    let critical_wave_numbers = [2.0, 3.5, 5.0];
    let critical_mean_free_paths = [7.5, 10.0, 12.0];

    let range = path_degeneracy_range(PathDegeneracyRangeInput {
        grouping: PathDegeneracyGroupsInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            first_bounce_degeneracies: &first_bounce_degeneracies,
            candidates: &candidates,
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
            force_no_symmetry: false,
        },
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        current_normalization: -1.0,
        criterion_percent: 50.0,
        retention_reference: None,
    })?;

    assert_eq!(range.groups.len(), 1);
    assert_eq!(range.groups[0].path_indices, vec![1, 2, 3, 4]);
    assert_eq!(range.groups[0].degeneracy, 7);
    assert_close(range.importances[0].port_importance, 1.117_176_271E-05);
    assert_option_close(range.importances[0].heap_importance, Some(1.036_497_688E1));
    assert_option_close(
        range.importances[0].reversed_heap_importance,
        Some(2.983_642_340),
    );
    assert_option_close(range.importances[0].output_importance, Some(100.0));
    assert_close(range.normalization, 1.964_455_259E-05);
    assert_eq!(range.retention.retained_unique_count, 1);
    assert_eq!(range.retention.retained_total_degeneracy, 7);
    assert_eq!(range.retention.decisions[0].fraction_percent, 100.0);
    assert!(range.retention.decisions[0].retained);
    assert_eq!(
        range.retention.reference,
        Some(PathDegeneracyRetentionReference {
            port_importance: range.importances[0].port_importance,
            degeneracy: 7,
        })
    );
    Ok(())
}

#[test]
fn path_degeneracy_reduction_partitions_ranges_and_carries_state() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let first_bounce_degeneracies = [0, 2, 3, 4, 5, 1, 7];
    let records = [
        PathDegeneracyRecord {
            total_path_length: 9.766_0,
            path_indices: &[1, 2, 3, 4],
        },
        PathDegeneracyRecord {
            total_path_length: 9.766_5,
            path_indices: &[4, 3, 2, 1],
        },
        PathDegeneracyRecord {
            total_path_length: 4.5,
            path_indices: &[1, 2],
        },
    ];
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    let wave_numbers = [1.2, 2.0, 3.25, 4.5, 6.0];
    let mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    let critical_wave_numbers = [2.0, 3.5, 5.0];
    let critical_mean_free_paths = [7.5, 10.0, 12.0];

    let reduction = path_degeneracy_reduction(PathDegeneracyReductionInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        first_bounce_degeneracies: &first_bounce_degeneracies,
        records: &records,
        polarization: 0,
        spin: 0,
        electric_vector: [0.0, 0.0, 1.0],
        incident_vector: [0.0, 0.0, 0.0],
        symmetry_case_override: None,
        force_no_symmetry: false,
        fbeta: fbeta.view(),
        wave_numbers: &wave_numbers,
        mean_free_paths: &mean_free_paths,
        start_energy_index: 1,
        fbeta_critical: fbetac.view(),
        critical_wave_numbers: &critical_wave_numbers,
        critical_mean_free_paths: &critical_mean_free_paths,
        current_normalization: -1.0,
        criterion_percent: 50.0,
        retention_reference: None,
    })?;

    assert_eq!(reduction.ranges.len(), 2);
    assert_eq!(reduction.retained_unique_count, 2);
    assert_eq!(reduction.retained_total_degeneracy, 9);
    assert_eq!(reduction.ranges[0].representative_total_path_length, 9.766);
    assert_eq!(reduction.ranges[0].range.groups[0].degeneracy, 7);
    assert_eq!(reduction.ranges[1].representative_total_path_length, 4.5);
    assert_eq!(reduction.ranges[1].range.groups[0].path_indices, vec![2, 1]);
    assert_close(
        reduction.ranges[1].range.importances[0].port_importance,
        7.728_009_950E-03,
    );
    assert_eq!(
        reduction.ranges[1].range.retention.reference,
        reduction.ranges[0].range.retention.reference
    );
    assert!(reduction.ranges[1].range.retention.decisions[0].retained);
    assert_close(reduction.normalization, 1.964_455_259E-05);
    assert_eq!(
        reduction.retention_reference,
        Some(PathDegeneracyRetentionReference {
            port_importance: reduction.ranges[0].range.importances[0].port_importance,
            degeneracy: 7,
        })
    );
    Ok(())
}

#[test]
fn path_degeneracy_reduction_rejects_nonfinite_record_lengths() {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let fbeta = reference_fbeta_output_table();
    let fbetac = reference_fbeta_table();
    let records = [PathDegeneracyRecord {
        total_path_length: Real::NAN,
        path_indices: &[1],
    }];

    assert!(matches!(
        path_degeneracy_reduction(PathDegeneracyReductionInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            first_bounce_degeneracies: &[0, 1],
            records: &records,
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
            force_no_symmetry: false,
            fbeta: fbeta.view(),
            wave_numbers: &[1.0, 2.0],
            mean_free_paths: &[5.0, 6.0],
            start_energy_index: 0,
            fbeta_critical: fbetac.view(),
            critical_wave_numbers: &[1.0, 2.0, 3.0],
            critical_mean_free_paths: &[5.0, 6.0, 7.0],
            current_normalization: -1.0,
            criterion_percent: 1.0,
            retention_reference: None,
        }),
        Err(PathError::NonFinitePathDegeneracyRecordLength { record: 0, .. })
    ));
}

fn sample_group(degeneracy: usize) -> PathDegeneracyGroup {
    PathDegeneracyGroup {
        path_indices: vec![1],
        degeneracy,
        degeneracy_hash: degeneracy as Real,
        member_count: 1,
        coordinates: arr2(&[[0.0, 0.0, 1.0]]),
        reversed: false,
        symmetry_case: 1,
    }
}
