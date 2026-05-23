use super::{support::*, *};

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
