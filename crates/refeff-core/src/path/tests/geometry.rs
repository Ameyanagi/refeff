use super::{support::*, *};

#[test]
fn path_heap_helpers_match_feff_reference() -> Result<(), PathError> {
    let mut keys = [1.0, 3.0, 2.0, 5.0, 4.0, 0.5];
    let mut indices = [10, 30, 20, 50, 40, 5];
    path_heap_bubble_up(&mut keys, &mut indices)?;
    assert_eq!(keys, [0.5, 3.0, 1.0, 5.0, 4.0, 2.0]);
    assert_eq!(indices, [5, 30, 10, 50, 40, 20]);

    let mut keys = [6.0, 2.0, 3.0, 4.0, 5.0];
    let mut indices = [60, 20, 30, 40, 50];
    path_heap_bubble_down(&mut keys, &mut indices)?;
    assert_eq!(keys, [2.0, 4.0, 3.0, 6.0, 5.0]);
    assert_eq!(indices, [20, 40, 30, 60, 50]);

    let mut keys = [0.2, 0.4, 0.3, 0.8, 0.7, 0.5, 0.1];
    let mut indices = [2, 4, 3, 8, 7, 5, 1];
    path_heap_bubble_up(&mut keys, &mut indices)?;
    assert_eq!(keys, [0.1, 0.4, 0.2, 0.8, 0.7, 0.5, 0.3]);
    assert_eq!(indices, [1, 4, 2, 8, 7, 5, 3]);
    Ok(())
}

#[test]
fn path_heap_helpers_reject_invalid_inputs() {
    assert!(matches!(
        path_heap_bubble_up(&mut [1.0, 2.0], &mut [1]),
        Err(PathError::HeapLengthMismatch {
            keys_len: 2,
            indices_len: 1
        })
    ));
    assert!(matches!(
        path_heap_bubble_down(&mut [1.0, Real::NAN], &mut [1, 2]),
        Err(PathError::NonFiniteHeapKey { index: 1, .. })
    ));
}

#[test]
fn pathfinder_preparation_matches_feff_neighbor_table_reference() -> Result<(), PathError> {
    let atom_positions = arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 0.0],
    ]);

    let preparation = pathfinder_preparation(PathfinderPreparationInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &[0, 1, 1, 2, 2],
        first_bounce_degeneracies: &[0, 2, 0, 3, 1],
        fms_radius: 2.1,
    })?;

    assert_eq!(preparation.absorber_source_index, 0);
    assert_eq!(preparation.first_bounce_count, 3);
    assert_eq!(
        preparation.cluster_outside,
        vec![false, false, false, true, false]
    );
    assert_eq!(preparation.first_bounce_degeneracies, vec![0, 2, 0, 3, 1]);
    assert_eq!(preparation.first_bounce_neighbors, vec![1, 4, 3, 2, 0]);
    assert_eq!(
        preparation.neighbor_rows,
        vec![
            vec![1, 4, 2, 3, 0],
            vec![0, 4, 2, 3, 1],
            vec![0, 4, 1, 3, 2],
            vec![0, 1, 4, 2, 3],
            vec![0, 1, 2, 3, 4],
        ]
    );
    Ok(())
}

#[test]
fn pathfinder_preparation_moves_absorber_like_feff_paths() -> Result<(), PathError> {
    let atom_positions = arr2(&[
        [5.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ]);

    let preparation = pathfinder_preparation(PathfinderPreparationInput {
        atom_positions: atom_positions.view(),
        atom_potentials: &[1, 2, 0, 3],
        first_bounce_degeneracies: &[0, 1, 0, 2],
        fms_radius: 2.0,
    })?;

    assert_eq!(preparation.absorber_source_index, 2);
    assert_eq!(preparation.atom_potentials, vec![0, 2, 1, 3]);
    assert_eq!(preparation.first_bounce_degeneracies, vec![0, 1, 0, 2]);
    assert_eq!(preparation.cluster_outside, vec![false, false, true, true]);
    assert_eq!(preparation.first_bounce_count, 2);
    assert_eq!(preparation.first_bounce_neighbors, vec![1, 3, 2, 0]);
    assert_eq!(
        preparation.atom_positions.row(0).to_vec(),
        vec![0.0, 0.0, 0.0]
    );
    assert_eq!(
        preparation.atom_positions.row(2).to_vec(),
        vec![5.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
fn pathfinder_preparation_rejects_invalid_inputs() {
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

    assert!(matches!(
        pathfinder_preparation(PathfinderPreparationInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &[0],
            first_bounce_degeneracies: &[0, 1],
            fms_radius: 2.0,
        }),
        Err(PathError::PathfinderPreparationLengthMismatch {
            positions: 2,
            potentials: 1,
            first_bounce_degeneracies: 2,
        })
    ));
    assert!(matches!(
        pathfinder_preparation(PathfinderPreparationInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &[1, 2],
            first_bounce_degeneracies: &[0, 1],
            fms_radius: 2.0,
        }),
        Err(PathError::PathfinderMissingAbsorber)
    ));
    assert!(matches!(
        pathfinder_preparation(PathfinderPreparationInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &[0, 1],
            first_bounce_degeneracies: &[9, 0],
            fms_radius: 2.0,
        }),
        Err(PathError::PathfinderMissingFirstBounce)
    ));
    assert!(matches!(
        pathfinder_preparation(PathfinderPreparationInput {
            atom_positions: atom_positions.view(),
            atom_potentials: &[0, 1],
            first_bounce_degeneracies: &[0, 1],
            fms_radius: Real::NAN,
        }),
        Err(PathError::NonFinitePathfinderFmsRadius { .. })
    ));
}

#[test]
fn path_geometry_matches_feff_mrb_reference() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();

    let case_a = path_geometry(atom_positions.view(), &[1, 2, 3])?;
    assert_path_geometry_close(
        &case_a,
        &[1.118_034_005, 1.268_857_718, 2.598_076_344, 1.797_220_111],
        &[
            0.810_643_494_1,
            -0.524_784_803_4,
            -0.516_135_692_6,
            0.104_511_246_1,
        ],
    );

    let case_b = path_geometry(atom_positions.view(), &[5, 1])?;
    assert_path_geometry_close(
        &case_b,
        &[0.0, 1.118_034_005, 1.118_034_005],
        &[0.0, -1.0, 0.0],
    );

    let case_c = path_geometry(atom_positions.view(), &[4, 1, 6, 2])?;
    assert_path_geometry_close(
        &case_c,
        &[
            1.603_121_996,
            1.661_324_859,
            0.0,
            1.268_857_718,
            2.271_563_292,
        ],
        &[
            -0.765_965_223_3,
            0.0,
            0.0,
            -0.957_571_744_9,
            -0.142_794_638_9,
        ],
    );

    Ok(())
}

#[test]
fn path_geometry_rejects_invalid_inputs() {
    let bad_shape = arr2(&[[0.0, 0.0]]);
    assert!(matches!(
        path_geometry(bad_shape.view(), &[]),
        Err(PathError::InvalidAtomPositionShape {
            rows: 1,
            columns: 2
        })
    ));

    let atom_positions = mrb_reference_positions();
    assert!(matches!(
        path_geometry(atom_positions.view(), &[7]),
        Err(PathError::AtomIndexOutOfRange {
            position: 0,
            atom_index: 7,
            atoms: 7
        })
    ));

    let with_nan = arr2(&[[0.0, 0.0, 0.0], [1.0, Real::NAN, 0.0]]);
    assert!(matches!(
        path_geometry(with_nan.view(), &[1]),
        Err(PathError::NonFiniteAtomPosition {
            atom_index: 1,
            component: 1,
            ..
        })
    ));
}

#[test]
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn path_output_parameters_match_feff_mpprmd_references() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();

    let case_four = path_output_parameters(atom_positions.view(), &[1, 2, 3, 4])?;
    assert_output_parameters_close(
        &case_four,
        &[
            1.118_034_005_165_100,
            1.268_857_717_514_038,
            2.598_076_343_536_377,
            3.178_049_802_780_151,
            1.603_121_995_925_903,
        ],
        &[
            6.255_461_105_722_411E-1,
            2.123_258_617_461_206,
            2.233_498_530_404_556,
            2.755_624_947_274_060,
            1.870_986_633_782_862,
        ],
        &[
            3.200_682_415_730_146E-1,
            -5.247_101_484_388_530E-1,
            1.812_598_918_451_830,
            -1.387_118_191_789_391,
            2.023_428_608_484_946,
        ],
    );

    let zero_leg = path_output_parameters(atom_positions.view(), &[5, 1])?;
    assert_output_parameters_close(
        &zero_leg,
        &[0.0, 1.118_034_005_165_100, 1.118_034_005_165_100],
        &[
            1.570_796_326_794_897,
            3.141_592_632_516_369,
            1.570_796_326_794_897,
        ],
        &[0.0, 0.0, 0.0],
    );

    let repeat = path_output_parameters(atom_positions.view(), &[4, 1, 6, 2])?;
    assert_output_parameters_close(
        &repeat,
        &[
            1.603_121_995_925_903,
            1.661_324_858_665_466,
            0.0,
            1.268_857_717_514_038,
            2.271_563_291_549_683,
        ],
        &[
            2.443_337_762_597_217,
            2.073_211_203_214_494,
            1.250_082_395_713_183,
            2.849_251_090_931_223,
            1.714_080_733_266_083,
        ],
        &[
            -1.803_951_023_146_609,
            -2.575_738_352_658_139,
            -2.048_662_038_325_701,
            -1.051_114_679_587_665,
            3.017_397_755_512_722E-1,
        ],
    );
    Ok(())
}

#[test]
fn path_output_parameters_reject_invalid_inputs() {
    let atom_positions = mrb_reference_positions();
    assert!(matches!(
        path_output_parameters(atom_positions.view(), &[]),
        Err(PathError::EmptyPathCriteria)
    ));
    assert!(matches!(
        path_output_parameters(atom_positions.view(), &[99]),
        Err(PathError::AtomIndexOutOfRange {
            position: 0,
            atom_index: 99,
            atoms: 7
        })
    ));
}
