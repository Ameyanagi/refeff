use super::{support::*, *};

#[test]
fn density_grid_points_match_feff_reference() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let points = rhorrp_density_grid_points(input)?;

    assert_eq!(points.points.dim(), (3, 24));
    assert_vector_close(column(&points.points, 0), [0.1, -0.2, 0.3]);
    assert_vector_close(column(&points.points, 1), [0.7, -0.4, 0.4]);
    assert_vector_close(column(&points.points, 3), [-0.2, 0.7, 0.8]);
    assert_vector_close(
        column(&points.points, 6),
        [0.233333333333333, -0.166666666666667, 0.666666666666667],
    );
    assert_vector_close(column(&points.points, 23), [1.4, 0.4, 2.1]);
    Ok(())
}

#[test]
fn evaluate_density_grid_matches_feff_reference() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let evaluated = rhorrp_evaluate_density_grid(input, |point| Ok(sample_density(point)))?;

    assert_eq!(evaluated.point_count(), 24);
    assert_eq!(evaluated.points.dim(), (3, 24));
    assert_vector_close(column(&evaluated.points, 0), [0.1, -0.2, 0.3]);
    assert_real_close(evaluated.density_per_bohr3[0], -0.470_000_000_000_000_1);
    assert_vector_close(column(&evaluated.points, 1), [0.7, -0.4, 0.4]);
    assert_real_close(evaluated.density_per_bohr3[1], -0.580_000_000_000_000_1);
    assert_vector_close(column(&evaluated.points, 3), [-0.2, 0.7, 0.8]);
    assert_real_close(evaluated.density_per_bohr3[3], 0.659_999_999_999_999_9);
    assert_vector_close(
        column(&evaluated.points, 6),
        [0.233333333333333, -0.166666666666667, 0.666666666666667],
    );
    assert_real_close(evaluated.density_per_bohr3[6], -0.472_222_222_222_222_27);
    assert_vector_close(column(&evaluated.points, 23), [1.4, 0.4, 2.1]);
    assert_real_close(evaluated.density_per_bohr3[23], 1.709_999_999_999_999_5);
    Ok(())
}

#[test]
fn point_and_next_index_match_feff_order() -> Result<(), RhorrpError> {
    let axes = reference_axes();
    let input = reference_grid_input(&axes);
    let mut index = vec![1, 1, 1];
    assert_vector_close(rhorrp_point_at_index(input, &index)?, [0.1, -0.2, 0.3]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![2, 1, 1]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![3, 1, 1]);
    rhorrp_next_index_1based(&[3, 2, 4], &mut index)?;
    assert_eq!(index, vec![1, 2, 1]);
    Ok(())
}

#[test]
fn process_ranges_match_feff_reference() -> Result<(), RhorrpError> {
    assert_eq!(
        rhorrp_process_ranges(10, 3)?,
        vec![
            RhorrpProcessRange {
                process: 0,
                start_1based: 1,
                end_1based: 4,
            },
            RhorrpProcessRange {
                process: 1,
                start_1based: 5,
                end_1based: 7,
            },
            RhorrpProcessRange {
                process: 2,
                start_1based: 8,
                end_1based: 10,
            },
        ]
    );
    assert_eq!(
        rhorrp_process_ranges(3, 5)?,
        vec![
            RhorrpProcessRange {
                process: 0,
                start_1based: 1,
                end_1based: 1,
            },
            RhorrpProcessRange {
                process: 1,
                start_1based: 2,
                end_1based: 2,
            },
            RhorrpProcessRange {
                process: 2,
                start_1based: 3,
                end_1based: 3,
            },
            RhorrpProcessRange {
                process: 3,
                start_1based: 4,
                end_1based: 3,
            },
            RhorrpProcessRange {
                process: 4,
                start_1based: 4,
                end_1based: 3,
            },
        ]
    );
    assert_eq!(rhorrp_process_ranges(24, 4)?[3].len(), 6);
    assert!(rhorrp_process_ranges(3, 5)?[3].is_empty());
    assert!(matches!(
        rhorrp_process_ranges(10, 0),
        Err(RhorrpError::InvalidProcessCount)
    ));
    Ok(())
}

#[test]
fn fms_inclusion_counts_match_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_inclusion_positions();
    let counts = rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
        atom_positions: positions.view(),
        representative_atoms: &[0, 1, 3, 5],
        fms_radius: 1.25,
    })?;

    assert_eq!(counts, vec![4, 2, 2, 4]);
    Ok(())
}

#[test]
fn nearest_atom_matches_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_positions();
    let potentials = [0, 2, 1, 3];
    let first = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.7, 0.2, 0.1],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: Some(3),
    })?;
    assert_eq!(first.atom_index_1based, 2);
    assert_eq!(first.potential_index, 2);
    assert_vector_close(first.displacement, [-0.3, 0.2, 0.1]);

    let z_limited = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.0, 0.1, 0.8],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: Some(3),
    })?;
    assert_eq!(z_limited.atom_index_1based, 1);
    assert_eq!(z_limited.potential_index, 0);
    assert_vector_close(z_limited.displacement, [0.0, 0.1, 0.8]);

    let z_all = rhorrp_nearest_atom(RhorrpNearestAtomInput {
        point: [0.0, 0.1, 0.8],
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: None,
    })?;
    assert_eq!(z_all.atom_index_1based, 4);
    assert_eq!(z_all.potential_index, 3);
    assert_vector_close(z_all.displacement, [0.0, 0.1, -0.2]);
    Ok(())
}

#[test]
fn nearest_atom_table_matches_feff_reference() -> Result<(), RhorrpError> {
    let positions = reference_positions();
    let potentials = [0, 2, 1, 3];
    let points = reference_nearest_points();
    let table = rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
        points: points.view(),
        atom_positions: positions.view(),
        atom_potentials: &potentials,
        fms_atom_count: None,
    })?;

    assert_eq!(table.point_count(), 4);
    assert_vector_close(
        row(&table.displacement_bohr, 0),
        [-0.300_000_000_000_000_04, 0.2, 0.1],
    );
    assert_vector_close(row(&table.displacement_bohr, 1), [0.0, 0.1, -0.2]);
    assert_vector_close(row(&table.displacement_bohr, 2), [0.2, -0.1, 0.1]);
    assert_vector_close(row(&table.displacement_bohr, 3), [0.0, 0.5, 0.5]);
    assert_eq!(table.atom_indices, vec![1, 3, 2, 0]);
    assert_eq!(table.atom_indices_1based, vec![2, 4, 3, 1]);
    assert_eq!(table.potential_indices, vec![2, 3, 1, 0]);
    Ok(())
}

#[test]
fn rhorrp_helpers_reject_invalid_inputs() {
    let axes = arr2(&[[1.0], [0.0], [0.0]]);
    assert!(matches!(
        rhorrp_density_grid_points(RhorrpDensityGridInput {
            origin: [0.0; 3],
            axes: axes.view(),
            points_per_axis: &[1],
        }),
        Err(RhorrpError::InvalidPointCount { axis: 0, value: 1 })
    ));
    assert!(matches!(
        rhorrp_point_at_index(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            &[3],
        ),
        Err(RhorrpError::InvalidGridIndex {
            axis: 0,
            index: 3,
            limit: 2,
        })
    ));
    assert!(matches!(
        rhorrp_evaluate_density_grid(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            |_| Ok(f64::NAN),
        ),
        Err(RhorrpError::NonFiniteDensityValue { point: 0, .. })
    ));
    assert!(matches!(
        rhorrp_evaluate_density_grid(
            RhorrpDensityGridInput {
                origin: [0.0; 3],
                axes: axes.view(),
                points_per_axis: &[2],
            },
            |_| Err(RhorrpError::InvalidProcessCount),
        ),
        Err(RhorrpError::InvalidProcessCount)
    ));

    let positions = reference_positions();
    assert!(matches!(
        rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0; 3],
            atom_positions: positions.view(),
            atom_potentials: &[0, 1],
            fms_atom_count: None,
        }),
        Err(RhorrpError::AtomPotentialLengthMismatch {
            potentials: 2,
            atoms: 4,
        })
    ));
    assert!(matches!(
        rhorrp_nearest_atom(RhorrpNearestAtomInput {
            point: [0.0; 3],
            atom_positions: positions.view(),
            atom_potentials: &[0, 1, 2, 3],
            fms_atom_count: Some(5),
        }),
        Err(RhorrpError::InvalidFmsAtomCount {
            fms_atom_count: 5,
            atoms: 4,
        })
    ));
    let bad_points = arr2(&[[0.0, 1.0]]);
    assert!(matches!(
        rhorrp_nearest_atom_table(RhorrpNearestAtomTableInput {
            points: bad_points.view(),
            atom_positions: positions.view(),
            atom_potentials: &[0, 1, 2, 3],
            fms_atom_count: None,
        }),
        Err(RhorrpError::InvalidPointTableShape {
            rows: 1,
            columns: 2,
        })
    ));
    assert!(matches!(
        rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
            atom_positions: positions.view(),
            representative_atoms: &[0, 4],
            fms_radius: 1.0,
        }),
        Err(RhorrpError::InvalidRepresentativeAtom {
            potential: 1,
            representative: 4,
            atoms: 4,
        })
    ));
    assert!(matches!(
        rhorrp_fms_inclusion_counts(RhorrpFmsInclusionInput {
            atom_positions: positions.view(),
            representative_atoms: &[0],
            fms_radius: f64::NAN,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "fms_radius",
            ..
        })
    ));
}
