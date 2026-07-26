use super::*;

#[test]
fn atheap_matches_feff_reference_sort_order() -> Result<(), FmsError> {
    let mut atoms = vec![
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [0.0, 2.0, 0.0],
            potential: 4,
        },
    ];

    let keys = sort_atoms_by_radius(&mut atoms)?;

    assert_eq!(
        atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
        vec![0, 2, 3, 1, 4]
    );
    assert_eq!(atoms[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(atoms[1].position, [-1.0, 0.0, 0.0]);
    assert_close_f64(keys[0], 2.0e-6);
    assert_close_f64(keys[1], 1.000_003);
    assert_close_f64(keys[2], 1.000_004);
    assert_close_f64(keys[3], 4.000_001);
    assert_close_f64(keys[4], 4.000_005);
    Ok(())
}

#[test]
fn getang_matches_feff_reference_angles() -> Result<(), FmsError> {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0],
        [0.0, 5.0e-8, 2.0e-7],
        [0.0, 2.0e-7, 0.0],
    ];

    let (theta, phi) = pair_polar_angles(&positions, 1, 0)?;
    assert_close_f32(theta, 0.841_068_6);
    assert_close_f32(phi, 1.107_148_8);

    let (theta, phi) = pair_polar_angles(&positions, 3, 2)?;
    assert_close_f32(theta, 2.498_091_5);
    assert_close_f32(phi, 1.570_796_4);

    assert_eq!(pair_polar_angles(&positions, 0, 0)?, (0.0, 0.0));
    Ok(())
}

#[test]
fn sortat_matches_feff_reference_representative_order() -> Result<(), FmsError> {
    let mut atoms = vec![
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [3.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [4.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [5.0, 0.0, 0.0],
            potential: 1,
        },
    ];

    let representatives = sort_representative_atoms(0, 3, &mut atoms)?;

    assert_eq!(representatives, vec![Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(
        atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 2, 1]
    );
    assert_eq!(atoms[1].position, [2.0, 0.0, 0.0]);
    assert_eq!(atoms[2].position, [1.0, 0.0, 0.0]);
    assert_eq!(atoms[3].position, [3.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn yprep_cluster_matches_feff_radius_prefix_reference() -> Result<(), FmsError> {
    let positions = array![
        [2.0_f32, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let potentials = [1, 0, 2, 1, 2];

    let cluster = fms_yprep_cluster(FmsYprepClusterInput {
        central_potential: 0,
        potentials: &potentials,
        positions: positions.view(),
        cluster_radius: 2.1,
        cluster_capacity: 3,
    })?;

    assert_eq!(cluster.central_atom, 1);
    assert_eq!(cluster.untruncated_count, 4);
    assert_eq!(cluster.atoms.len(), 3);
    assert_eq!(
        cluster
            .atoms
            .iter()
            .map(|atom| atom.potential)
            .collect::<Vec<_>>(),
        vec![0, 2, 1]
    );
    assert_eq!(cluster.atoms[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(cluster.atoms[1].position, [0.0, 0.0, 1.0]);
    assert_eq!(cluster.atoms[2].position, [1.0, -1.0, 0.0]);
    Ok(())
}

#[test]
fn yprep_geometry_matches_feff_pair_rotation_sequence() -> Result<(), FmsError> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [0.0, 0.0, 1.0],
            potential: 2,
        },
        FmsAtom {
            position: [1.0, -1.0, 0.0],
            potential: 1,
        },
    ];

    let geometry = fms_yprep_geometry(2, 2, &atoms)?;

    assert_eq!(geometry.phi.shape(), &[3, 3]);
    assert_eq!(geometry.rotations.shape(), &[5, 5, 3, 2, 3, 3]);
    assert_close_f32(geometry.phi[(1, 0)], 0.0);
    assert_close_f32(geometry.phi[(2, 0)], -std::f32::consts::FRAC_PI_4);
    assert_close_f32(geometry.phi[(0, 2)], 3.0 * std::f32::consts::FRAC_PI_4);
    assert_complex32_close(
        geometry.rotations[(2, 2, 0, 0, 0, 0)],
        Complex32::new(0.0, 0.0),
    );

    let expected_forward = fms_rotation_matrix(
        2,
        2,
        std::f32::consts::FRAC_PI_2,
        3.0 * std::f32::consts::FRAC_PI_4,
        FmsRotationDirection::Forward,
    )?;
    let expected_backward = fms_rotation_matrix(
        2,
        2,
        -std::f32::consts::FRAC_PI_2,
        3.0 * std::f32::consts::FRAC_PI_4,
        FmsRotationDirection::Backward,
    )?;
    assert_complex32_close(
        geometry.rotations[(3, 1, 1, 0, 2, 0)],
        expected_forward[(3, 1, 1)],
    );
    assert_complex32_close(
        geometry.rotations[(1, 3, 2, 1, 2, 0)],
        expected_backward[(1, 3, 2)],
    );
    Ok(())
}

#[test]
fn fms_cluster_helpers_reject_invalid_inputs() {
    let positions = [[0.0, 0.0, 0.0]];
    assert_eq!(
        pair_polar_angles(&positions, 1, 0),
        Err(FmsError::AtomIndexOutOfRange { index: 1, len: 1 })
    );

    let mut atoms = [FmsAtom {
        position: [f32::NAN, 0.0, 0.0],
        potential: 0,
    }];
    assert_eq!(
        sort_atoms_by_radius(&mut atoms),
        Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
    );

    let mut atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    assert_eq!(
        sort_representative_atoms(0, 1, &mut atoms),
        Err(FmsError::CentralAtomMismatch {
            expected: 0,
            actual: 1,
        })
    );
    assert_eq!(
        sort_representative_atoms(-1, 1, &mut atoms),
        Err(FmsError::PotentialOutOfRange {
            potential: -1,
            max_potential: 1,
        })
    );

    let yprep_positions = array![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 0,
            potentials: &[0, 0],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::DuplicateAbsorber)
    );
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 2,
            potentials: &[0, 1],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::MissingCentralAtom { potential: 2 })
    );
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 0,
            potentials: &[0],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::AtomCountMismatch {
            potentials: 1,
            positions: 2,
        })
    );
    assert_eq!(
        fms_yprep_geometry(2, 2, &[]),
        Err(FmsError::AtomIndexOutOfRange { index: 0, len: 0 })
    );
    assert_eq!(
        fms_yprep_geometry(
            2,
            2,
            &[FmsAtom {
                position: [f32::NAN, 0.0, 0.0],
                potential: 0,
            }],
        ),
        Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
    );
}
