use super::{support::*, *};

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
