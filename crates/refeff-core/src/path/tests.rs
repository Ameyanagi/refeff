use super::*;
use ndarray::{Array2, Array3, ShapeBuilder, arr2};

const CRITERION_TOLERANCE: Real = 1.0e-6;
const HASH_TOLERANCE: Real = 1.0e-3;
const MRB_TOLERANCE: Real = 1.0e-7;
const OUTPUT_PARAMETER_TOLERANCE: Real = 1.0e-7;
const STANDARD_TOLERANCE: Real = 1.0e-7;
const PHASE_CRITERIA_TOLERANCE: Real = 1.0e-6;

#[test]
fn pack_path_indices_matches_feff_reference() -> Result<(), PathError> {
    let packed = pack_path_indices(&[0, 12, 1289])?;
    assert_eq!(packed, [19_969_203, 1_289, 0]);
    assert_eq!(unpack_path_indices(packed, 8)?, vec![0, 12, 1289]);

    let packed = pack_path_indices(&[1, 2, 3, 4, 5, 6, 7, 8])?;
    assert_eq!(packed, [3_329_498, 8_325_663, 13_321_836]);
    assert_eq!(
        unpack_path_indices(packed, 8)?,
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );

    let packed = pack_path_indices(&[1289, 0, 17, 300, 42])?;
    assert_eq!(packed, [1_662_815, 70_279_217, 0]);
    assert_eq!(unpack_path_indices(packed, 8)?, vec![1289, 0, 17, 300, 42]);
    Ok(())
}

#[test]
fn pack_path_indices_rejects_invalid_inputs() {
    assert!(matches!(
        pack_path_indices(&[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        Err(PathError::TooManyIndices { count: 9, .. })
    ));
    assert!(matches!(
        pack_path_indices(&[1290]),
        Err(PathError::IndexOutOfRange {
            position: 0,
            value: 1290,
            ..
        })
    ));
    assert!(matches!(
        unpack_path_indices([3_329_498, 8_325_663, 13_321_836], 7),
        Err(PathError::UnpackCapacityTooSmall {
            packed_count: 8,
            capacity: 7
        })
    ));
    assert!(matches!(
        unpack_path_indices([-1, 0, 0], 8),
        Err(PathError::NegativePackedValue {
            position: 0,
            value: -1
        })
    ));
}

#[test]
fn path_phase_criteria_tables_match_feff_prcrit_reference() -> Result<(), PathError> {
    let (energies, references, phase_shifts, angular_limits) = prcrit_reference_inputs();
    let tables = path_phase_criteria_tables(PathPhaseCriteriaInput {
        energies: &energies,
        reference_energies: &references,
        phase_shifts: phase_shifts.view(),
        angular_limits: angular_limits.view(),
        output_energy_count: 38,
        zero_wave_energy_index: 1,
    })?;

    assert_eq!(tables.output_energy_count, 38);
    assert_eq!(tables.zero_wave_energy_index, 1);
    assert_eq!(tables.fbeta.shape(), &[81, 3, 43]);
    assert_eq!(tables.fbeta.strides(), &[1, 81, 243]);
    assert_eq!(
        tables.critical_energy_indices,
        vec![1, 6, 11, 16, 21, 31, 35, 39, 41]
    );
    assert_eq!(tables.fbeta_critical.shape(), &[81, 3, 9]);
    assert_phase_close(
        tables.wave_numbers[1],
        0.346_975_892_782_211_3,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.wave_numbers[41],
        2.472_743_988_037_109_4,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.mean_free_paths[1],
        12.784_625_053_405_762,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.mean_free_paths[41],
        35.328_521_728_515_625,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.fbeta[(0, 0, 1)],
        1.187_224_864_959_716_8,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.fbeta[(40, 1, 6)],
        0.270_274_966_955_184_94,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.fbeta[(80, 2, 41)],
        0.729_954_421_520_233_2,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.fbeta[(30, 2, 11)],
        1.127_693_772_315_979,
        PHASE_CRITERIA_TOLERANCE,
    );

    let mut fbeta_zero_sum = 0.0;
    for energy in 0..43 {
        for potential in 0..3 {
            fbeta_zero_sum += tables.fbeta[(40, potential, energy)];
        }
    }
    assert_phase_close(
        single_precision_path_value(fbeta_zero_sum),
        103.829_582_214_355_47,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.critical_wave_numbers[8],
        2.472_743_988_037_109_4,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.critical_mean_free_paths[8],
        35.328_521_728_515_625,
        PHASE_CRITERIA_TOLERANCE,
    );
    assert_phase_close(
        tables.fbeta_critical[(40, 2, 8)],
        0.729_954_421_520_233_2,
        PHASE_CRITERIA_TOLERANCE,
    );
    Ok(())
}

#[test]
fn path_phase_criteria_tables_truncate_critical_indices() -> Result<(), PathError> {
    let (energies, references, phase_shifts, angular_limits) = prcrit_reference_inputs();
    let tables = path_phase_criteria_tables(PathPhaseCriteriaInput {
        energies: &energies,
        reference_energies: &references,
        phase_shifts: phase_shifts.view(),
        angular_limits: angular_limits.view(),
        output_energy_count: 10,
        zero_wave_energy_index: 39,
    })?;

    assert_eq!(tables.critical_energy_indices, vec![39]);
    assert_eq!(tables.critical_wave_numbers, vec![tables.wave_numbers[39]]);
    assert_eq!(
        tables.critical_mean_free_paths,
        vec![tables.mean_free_paths[39]]
    );
    assert_eq!(tables.fbeta_critical.shape(), &[81, 3, 1]);
    Ok(())
}

#[test]
fn path_phase_criteria_tables_reject_invalid_inputs() {
    let (energies, references, phase_shifts, angular_limits) = prcrit_reference_inputs();
    assert!(matches!(
        path_phase_criteria_tables(PathPhaseCriteriaInput {
            energies: &energies[..0],
            reference_energies: &references,
            phase_shifts: phase_shifts.view(),
            angular_limits: angular_limits.view(),
            output_energy_count: 1,
            zero_wave_energy_index: 0,
        }),
        Err(PathError::InvalidPathPhaseCriteriaShape { .. })
    ));

    let mut bad_limits = angular_limits.clone();
    bad_limits[(2, 1)] = 4;
    assert_eq!(
        path_phase_criteria_tables(PathPhaseCriteriaInput {
            energies: &energies,
            reference_energies: &references,
            phase_shifts: phase_shifts.view(),
            angular_limits: bad_limits.view(),
            output_energy_count: 38,
            zero_wave_energy_index: 1,
        }),
        Err(PathError::PathPhaseAngularLimitOutOfRange {
            energy: 2,
            potential: 1,
            angular_limit: 4,
            angular_channels: 4,
        })
    );

    let mut bad_phase = phase_shifts.clone();
    bad_phase[(3, 0, 2)] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        path_phase_criteria_tables(PathPhaseCriteriaInput {
            energies: &energies,
            reference_energies: &references,
            phase_shifts: bad_phase.view(),
            angular_limits: angular_limits.view(),
            output_energy_count: 38,
            zero_wave_energy_index: 1,
        }),
        Err(PathError::NonFinitePathPhaseComplex {
            quantity: "phase shift",
            energy: 3,
            angular: 0,
            potential: 2,
            ..
        })
    ));

    assert_eq!(
        path_phase_criteria_tables(PathPhaseCriteriaInput {
            energies: &energies,
            reference_energies: &references,
            phase_shifts: phase_shifts.view(),
            angular_limits: angular_limits.view(),
            output_energy_count: 44,
            zero_wave_energy_index: 1,
        }),
        Err(PathError::PathPhaseOutputEnergyOutOfRange {
            output_energy_count: 44,
            energies: 43,
        })
    );
}

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

#[test]
fn path_degeneracy_hash_matches_feff_phash_reference() -> Result<(), PathError> {
    let case_a_positions = arr2(&[
        [1.23456, -0.34567, 0.12549],
        [-2.25, 1.5004, -0.9995],
        [0.0, 2.4996, 3.3333],
    ]);
    assert_hash_close(
        path_degeneracy_hash(case_a_positions.view(), &[1, 3, 0])?,
        1.210_820_169_326_026E8,
    );

    let case_b_positions = arr2(&[[-0.0005, 0.0005, -1.2345]]);
    assert_hash_close(
        path_degeneracy_hash(case_b_positions.view(), &[2])?,
        4.000_129_162_432_861E7,
    );

    let case_c_positions = arr2(&[
        [4.4444, -3.3333, 2.2222],
        [-1.1111, 0.0, 1.1111],
        [0.75, -0.25, 0.5],
        [-0.5, -0.75, 1.25],
    ]);
    assert_hash_close(
        path_degeneracy_hash(case_c_positions.view(), &[1, 2, 3, 0])?,
        1.585_427_338_452_837E8,
    );

    Ok(())
}

#[test]
fn path_degeneracy_hash_rejects_invalid_inputs() {
    let bad_shape = arr2(&[[0.0, 0.0]]);
    assert!(matches!(
        path_degeneracy_hash(bad_shape.view(), &[1]),
        Err(PathError::InvalidPathHashShape {
            rows: 1,
            columns: 2,
            potentials: 1
        })
    ));

    let positions = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    assert!(matches!(
        path_degeneracy_hash(positions.view(), &[1]),
        Err(PathError::InvalidPathHashShape {
            rows: 2,
            columns: 3,
            potentials: 1
        })
    ));

    let positions = arr2(&[[0.0, 0.0, 0.0]]);
    assert!(matches!(
        path_degeneracy_hash(positions.view(), &[-1]),
        Err(PathError::NegativePathPotential {
            position: 0,
            value: -1
        })
    ));

    let positions = arr2(&[[Real::INFINITY, 0.0, 0.0]]);
    assert!(matches!(
        path_degeneracy_hash(positions.view(), &[1]),
        Err(PathError::PathHashCoordinateOutOfRange {
            position: 0,
            component: 0,
            ..
        })
    ));
}

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

#[test]
fn path_standard_coordinates_match_feff_mpprmp_references() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let path_indices = [1, 2, 3, 4];
    let z_vector = [0.0, 0.0, 1.0];
    let zero_vector = [0.0, 0.0, 0.0];

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 0,
            spin: 0,
            electric_vector: z_vector,
            incident_vector: zero_vector,
            symmetry_case_override: None,
        })?,
        1,
        &[
            [1.054_711_873E-15, -1.387_778_781E-17, 1.118_034_013],
            [7.429_670_302E-1, -0.0, 2.146_625_258],
            [1.646_371_899, 6.958_302_540E-1, -1.878_297_037E-1],
            [-6.697_471_102E-1, -1.377_214_196, 4.740_463_925E-1],
        ],
    );

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 1,
            spin: 0,
            electric_vector: z_vector,
            incident_vector: zero_vector,
            symmetry_case_override: None,
        })?,
        2,
        &[
            [1.118_034_013, -2.775_557_562E-17, 0.0],
            [2.146_625_258, 6.260_990_363E-1, 4.000_000_060E-1],
            [-1.878_297_037E-1, 1.762_021_613, 3.000_000_119E-1],
            [4.740_463_925E-1, -1.305_863_743, 8.000_000_119E-1],
        ],
    );

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 1,
            spin: 0,
            electric_vector: z_vector,
            incident_vector: [1.0, 0.0, 0.0],
            symmetry_case_override: None,
        })?,
        3,
        &[
            [1.100_000_024, 2.000_000_030E-1, 0.0],
            [2.0, 1.0, 4.000_000_060E-1],
            [-5.0E-1, 1.700_000_048, 3.000_000_119E-1],
            [6.999_999_881E-1, -1.200_000_048, 8.000_000_119E-1],
        ],
    );

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 2,
            spin: 0,
            electric_vector: z_vector,
            incident_vector: z_vector,
            symmetry_case_override: None,
        })?,
        4,
        &[
            [1.118_034_013, -2.775_557_562E-17, 0.0],
            [2.146_625_258, 6.260_990_363E-1, 4.000_000_060E-1],
            [-1.878_297_037E-1, 1.762_021_613, 3.000_000_119E-1],
            [4.740_463_925E-1, -1.305_863_743, 8.000_000_119E-1],
        ],
    );

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 1,
            spin: 1,
            electric_vector: [1.0, 0.0, 0.0],
            incident_vector: z_vector,
            symmetry_case_override: None,
        })?,
        6,
        &[
            [1.100_000_024, 2.000_000_030E-1, 0.0],
            [2.0, 1.0, 4.000_000_060E-1],
            [-5.0E-1, 1.700_000_048, 3.000_000_119E-1],
            [6.999_999_881E-1, -1.200_000_048, 8.000_000_119E-1],
        ],
    );

    assert_standard_coordinates(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &path_indices,
            polarization: 0,
            spin: 0,
            electric_vector: z_vector,
            incident_vector: zero_vector,
            symmetry_case_override: Some(7),
        })?,
        7,
        &[
            [1.100_000_024, 2.000_000_030E-1, 0.0],
            [2.0, 1.0, 4.000_000_060E-1],
            [-5.0E-1, 1.700_000_048, 3.000_000_119E-1],
            [6.999_999_881E-1, -1.200_000_048, 8.000_000_119E-1],
        ],
    );
    Ok(())
}

#[test]
fn path_standard_coordinates_reject_invalid_inputs() {
    let atom_positions = mrb_reference_positions();
    assert!(matches!(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &[1, 2],
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: Some(8),
        }),
        Err(PathError::InvalidPathSymmetryCase { symmetry_case: 8 })
    ));
    assert!(matches!(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: atom_positions.view(),
            path_indices: &[1, 2],
            polarization: 1,
            spin: 0,
            electric_vector: [Real::NAN, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
        }),
        Err(PathError::NonFinitePathStandardVector {
            vector: "electric vector",
            component: 0,
            ..
        })
    ));

    let degenerate = arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert!(matches!(
        path_standard_coordinates(PathStandardCoordinatesInput {
            atom_positions: degenerate.view(),
            path_indices: &[1],
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
        }),
        Err(PathError::DegeneratePathStandardAxis { symmetry_case: 1 })
    ));
}

#[test]
fn path_canonical_representation_matches_feff_timrep_references() -> Result<(), PathError> {
    let atom_positions = mrb_reference_positions();
    let atom_potentials = reference_atom_potentials();
    let z_vector = [0.0, 0.0, 1.0];
    let zero_vector = [0.0, 0.0, 0.0];

    let forward = path_canonical_representation(PathCanonicalRepresentationInput {
        atom_positions: atom_positions.view(),
        path_indices: &[1, 2, 3, 4],
        atom_potentials: &atom_potentials,
        polarization: 0,
        spin: 0,
        electric_vector: z_vector,
        incident_vector: zero_vector,
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;
    assert_canonical_representation(
        forward,
        &[1, 2, 3, 4],
        1,
        false,
        1.540_019_626_331_394E8,
        &[
            [1.054_711_873E-15, -1.387_778_781E-17, 1.118_034_005],
            [7.429_670_095E-1, -0.0, 2.146_625_280],
            [1.646_371_841, 6.958_302_259E-1, -1.878_297_031E-1],
            [-6.697_471_142E-1, -1.377_214_193, 4.740_463_793E-1],
        ],
    );

    let reversed = path_canonical_representation(PathCanonicalRepresentationInput {
        atom_positions: atom_positions.view(),
        path_indices: &[4, 3, 2, 1],
        atom_potentials: &atom_potentials,
        polarization: 0,
        spin: 0,
        electric_vector: z_vector,
        incident_vector: zero_vector,
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;
    assert_canonical_representation(
        reversed,
        &[1, 2, 3, 4],
        1,
        true,
        1.540_019_626_331_394E8,
        &[
            [1.054_711_873E-15, -1.387_778_781E-17, 1.118_034_005],
            [7.429_670_095E-1, -0.0, 2.146_625_280],
            [1.646_371_841, 6.958_302_259E-1, -1.878_297_031E-1],
            [-6.697_471_142E-1, -1.377_214_193, 4.740_463_793E-1],
        ],
    );

    let spin_block = path_canonical_representation(PathCanonicalRepresentationInput {
        atom_positions: atom_positions.view(),
        path_indices: &[4, 3, 2, 1],
        atom_potentials: &atom_potentials,
        polarization: 1,
        spin: 1,
        electric_vector: z_vector,
        incident_vector: zero_vector,
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;
    assert_canonical_representation(
        spin_block,
        &[4, 3, 2, 1],
        5,
        false,
        1.669_436_988_304_592E8,
        &[
            [1.389_244_437, 0.0, 8.000_000_119E-1],
            [-1.720_359_683, 4.246_912_599E-1, 3.000_000_119E-1],
            [1.439_630_985E-1, 2.231_428_862, 4.000_000_060E-1],
            [3.815_023_303E-1, 1.050_930_977, 0.0],
        ],
    );

    let forced = path_canonical_representation(PathCanonicalRepresentationInput {
        atom_positions: atom_positions.view(),
        path_indices: &[4, 3, 2, 1],
        atom_potentials: &atom_potentials,
        polarization: 0,
        spin: 0,
        electric_vector: z_vector,
        incident_vector: zero_vector,
        symmetry_case_override: None,
        force_no_symmetry: true,
    })?;
    assert_canonical_representation(
        forced,
        &[1, 2, 3, 4],
        7,
        true,
        1.609_590_554_105_973E8,
        &[
            [1.100_000_024, 2.000_000_030E-1, 0.0],
            [2.0, 1.0, 4.000_000_060E-1],
            [-5.0E-1, 1.700_000_048, 3.000_000_119E-1],
            [6.999_999_881E-1, -1.200_000_048, 8.000_000_119E-1],
        ],
    );

    let single = path_canonical_representation(PathCanonicalRepresentationInput {
        atom_positions: atom_positions.view(),
        path_indices: &[4],
        atom_potentials: &atom_potentials,
        polarization: 0,
        spin: 0,
        electric_vector: z_vector,
        incident_vector: zero_vector,
        symmetry_case_override: None,
        force_no_symmetry: false,
    })?;
    assert_canonical_representation(
        single,
        &[4],
        1,
        false,
        4.000_091_931_732_178E7,
        &[[0.0, 0.0, 1.603_121_996]],
    );
    Ok(())
}

#[test]
fn path_canonical_representation_rejects_missing_potentials() {
    let atom_positions = mrb_reference_positions();
    assert!(matches!(
        path_canonical_representation(PathCanonicalRepresentationInput {
            atom_positions: atom_positions.view(),
            path_indices: &[1, 2, 3],
            atom_potentials: &[0, 1],
            polarization: 0,
            spin: 0,
            electric_vector: [0.0, 0.0, 1.0],
            incident_vector: [0.0, 0.0, 0.0],
            symmetry_case_override: None,
            force_no_symmetry: false,
        }),
        Err(PathError::PathCriteriaAtomIndexOutOfRange {
            position: 1,
            atom_index: 2,
            atoms: 2
        })
    ));
}

fn mrb_reference_positions() -> ndarray::Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
        [2.0, 1.0, 0.4],
        [-0.5, 1.7, 0.3],
        [0.7, -1.2, 0.8],
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
    ])
}

fn assert_path_geometry_close(
    actual: &PathGeometry,
    expected_distances: &[Real],
    expected_cosines: &[Real],
) {
    assert_eq!(actual.leg_distances.len(), expected_distances.len());
    assert_eq!(actual.angle_cosines.len(), expected_cosines.len());

    for (&actual, &expected) in actual.leg_distances.iter().zip(expected_distances) {
        assert!(
            (actual - expected).abs() <= MRB_TOLERANCE,
            "leg distance {actual} != {expected}"
        );
    }
    for (&actual, &expected) in actual.angle_cosines.iter().zip(expected_cosines) {
        assert!(
            (actual - expected).abs() <= MRB_TOLERANCE,
            "angle cosine {actual} != {expected}"
        );
    }

    let expected_total = expected_distances
        .iter()
        .fold(0.0_f32, |sum, &distance| sum + distance as f32);
    assert!(
        (actual.total_path_length - Real::from(expected_total)).abs() <= MRB_TOLERANCE,
        "total path length {} != {}",
        actual.total_path_length,
        expected_total
    );
}

fn assert_output_parameters_close(
    actual: &PathOutputParameters,
    expected_distances: &[Real],
    expected_angles: &[Real],
    expected_eta: &[Real],
) {
    assert_real_slice_close(
        &actual.leg_distances,
        expected_distances,
        "output leg distance",
        OUTPUT_PARAMETER_TOLERANCE,
    );
    assert_real_slice_close(
        &actual.scattering_angles,
        expected_angles,
        "output scattering angle",
        OUTPUT_PARAMETER_TOLERANCE,
    );
    assert_real_slice_close(
        &actual.eta_angles,
        expected_eta,
        "output eta angle",
        OUTPUT_PARAMETER_TOLERANCE,
    );
}

fn assert_real_slice_close(actual: &[Real], expected: &[Real], label: &str, tolerance: Real) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} {actual} != {expected}"
        );
    }
}

fn assert_hash_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() <= HASH_TOLERANCE,
        "path hash {actual} != {expected}"
    );
}

fn assert_standard_coordinates(
    actual: PathStandardCoordinates,
    expected_case: u8,
    expected: &[[Real; 3]],
) {
    assert_eq!(actual.symmetry_case, expected_case);
    assert_eq!(actual.coordinates.nrows(), expected.len());
    assert_eq!(actual.coordinates.ncols(), 3);
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_value) in expected_row.iter().enumerate() {
            let actual_value = actual.coordinates[(row, column)];
            assert!(
                (actual_value - expected_value).abs() <= STANDARD_TOLERANCE,
                "standard coordinate ({row}, {column}) {actual_value} != {expected_value}"
            );
        }
    }
}

fn assert_canonical_representation(
    actual: PathCanonicalRepresentation,
    expected_path: &[usize],
    expected_case: u8,
    expected_reversed: bool,
    expected_hash: Real,
    expected_coordinates: &[[Real; 3]],
) {
    assert_eq!(actual.path_indices, expected_path);
    assert_eq!(actual.reversed, expected_reversed);
    assert_hash_close(actual.degeneracy_hash, expected_hash);
    assert_standard_coordinates(
        PathStandardCoordinates {
            coordinates: actual.coordinates,
            symmetry_case: actual.symmetry_case,
        },
        expected_case,
        expected_coordinates,
    );
}

fn reference_atom_potentials() -> Vec<usize> {
    (0..=8).map(|index| index % 4).collect()
}

fn prcrit_reference_inputs() -> (Vec<Complex>, Vec<Complex>, Array3<Complex>, Array2<usize>) {
    let energy_count = 43;
    let potential_count = 3;
    let angular_channels = 4;
    let energies = (0..energy_count)
        .map(|index| {
            let ie = (index + 1) as Real;
            Complex::new(0.02 * (ie - 2.0) + 0.001 * (ie - 1.0), 0.005 + 0.0003 * ie)
        })
        .collect::<Vec<_>>();
    let references = vec![Complex::new(-0.015, -0.002); energy_count];
    let phase_shifts = Array3::from_shape_fn(
        (energy_count, angular_channels, potential_count).f(),
        |(energy, angular, potential)| {
            let ie = (energy + 1) as Real;
            let il = (angular + 1) as Real;
            let iph = potential as Real;
            Complex::new(
                0.02 * ie + 0.11 * il + 0.03 * iph,
                0.004 * ie - 0.002 * il + 0.001 * iph,
            )
        },
    );
    let angular_limits = Array2::from_shape_fn(
        (energy_count, potential_count).f(),
        |(energy, potential)| (energy + 1 + potential) % angular_channels,
    );
    (energies, references, phase_shifts, angular_limits)
}

fn reference_fbeta_table() -> Array3<Real> {
    Array3::from_shape_fn((81, 4, 3), |(beta_row, potential, criterion)| {
        let beta_index = beta_row as i32 - 40;
        Real::from(
            0.5_f32
                + 0.01_f32 * potential as f32
                + 0.002_f32 * (criterion + 1) as f32
                + 0.003_f32 * beta_index.abs() as f32
                + 0.0001_f32 * beta_index as f32,
        )
    })
}

fn reference_fbeta_output_table() -> Array3<Real> {
    Array3::from_shape_fn((81, 4, 5), |(beta_row, potential, energy)| {
        let beta_index = beta_row as i32 - 40;
        Real::from(
            0.45_f32
                + 0.008_f32 * potential as f32
                + 0.015_f32 * (energy + 1) as f32
                + 0.0025_f32 * beta_index.abs() as f32
                + 0.0002_f32 * beta_index as f32,
        )
    })
}

fn assert_option_close(actual: Option<Real>, expected: Option<Real>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_close(actual, expected),
        (actual, expected) => assert_eq!(actual, expected),
    }
}

fn assert_phase_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "path phase criteria {actual} != {expected}"
    );
}

fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() <= CRITERION_TOLERANCE,
        "path criterion {actual} != {expected}"
    );
}
