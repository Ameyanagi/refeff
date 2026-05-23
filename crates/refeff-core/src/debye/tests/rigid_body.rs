use super::{support::*, *};

#[test]
fn dmdw_center_of_mass_and_inertia_match_feff_reference_formulas() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]);
    let masses = ndarray::arr1(&[2.0, 3.0, 5.0]);

    let center = dmdw_center_of_mass(positions.view(), masses.view())?;
    assert_slice_close(&center, &[0.6, 1.5, 0.0]);

    let centered = ndarray::arr2(&[[-0.6, -1.5, 0.0], [1.4, -1.5, 0.0], [-0.6, 1.5, 0.0]]);
    let tensor = dmdw_inertia_tensor(centered.view(), masses.view())?;
    assert_matrix_close(
        tensor.view(),
        &[[22.5, 9.0, 0.0], [9.0, 8.4, 0.0], [0.0, 0.0, 30.9]],
    );
    Ok(())
}

#[test]
fn dmdw_rigid_body_projection_modes_match_feff_make_trfd_formulas() -> Result<(), DebyeError> {
    let positions = ndarray::arr2(&[
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0],
    ]);
    let masses = ndarray::arr1(&[1.0; 6]);
    let modes = dmdw_rigid_body_projection_modes(positions.view(), masses.view())?;

    assert_slice_close(&modes.center_of_mass, &[0.0, 0.0, 0.0]);
    assert_vector_close(&modes.moments_of_inertia, &[10.0, 20.0, 26.0]);
    assert_matrix_abs_close(
        modes.principal_axes.view(),
        &[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    );

    let projection = modes.projection_modes;
    assert_eq!(projection.shape(), &[18, 6]);
    for left in 0..6 {
        assert_dmdw_close(column_dot(projection.view(), left, left), 1.0);
        for right in (left + 1)..6 {
            assert_dmdw_close(column_dot(projection.view(), left, right), 0.0);
        }
    }

    let translation_scale = 1.0 / 6.0_f64.sqrt();
    for atom in 0..6 {
        assert_dmdw_close(projection[(atom, 0)], translation_scale);
        assert_dmdw_close(projection[(6 + atom, 1)], translation_scale);
        assert_dmdw_close(projection[(12 + atom, 2)], translation_scale);
    }

    let rotation_z = ndarray::arr1(&[
        0.0,
        0.0,
        -2.0 / 10.0_f64.sqrt(),
        2.0 / 10.0_f64.sqrt(),
        0.0,
        0.0,
        1.0 / 10.0_f64.sqrt(),
        -1.0 / 10.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    let rotation_y = ndarray::arr1(&[
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 20.0_f64.sqrt(),
        -3.0 / 20.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0 / 20.0_f64.sqrt(),
        1.0 / 20.0_f64.sqrt(),
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    let rotation_x = ndarray::arr1(&[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -3.0 / 26.0_f64.sqrt(),
        3.0 / 26.0_f64.sqrt(),
        0.0,
        0.0,
        2.0 / 26.0_f64.sqrt(),
        -2.0 / 26.0_f64.sqrt(),
        0.0,
        0.0,
    ]);
    assert_dmdw_close(projection.column(3).dot(&rotation_z).abs(), 1.0);
    assert_dmdw_close(projection.column(4).dot(&rotation_y).abs(), 1.0);
    assert_dmdw_close(projection.column(5).dot(&rotation_x).abs(), 1.0);
    Ok(())
}

#[test]
fn dmdw_seed_projection_matches_feff_qj0_loop() -> Result<(), DebyeError> {
    let seed = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
    let inv_sqrt_two = 0.5_f64.sqrt();
    let modes = ndarray::arr2(&[
        [1.0, 0.0],
        [0.0, inv_sqrt_two],
        [0.0, inv_sqrt_two],
        [0.0, 0.0],
    ]);

    let projected = dmdw_project_seed_vector(seed.view(), modes.view())?;
    assert_vector_close(
        &projected,
        &[
            0.0,
            -0.123_091_490_979_332_72,
            0.123_091_490_979_332_72,
            0.984_731_927_834_661_8,
        ],
    );
    assert_dmdw_close(projected.iter().map(|value| value * value).sum(), 1.0);

    let normalized = dmdw_normalize_seed_vector(seed.view())?;
    assert_vector_close(
        &normalized,
        &[
            0.182_574_185_835_055_36,
            0.365_148_371_670_110_7,
            0.547_722_557_505_166_1,
            0.730_296_743_340_221_4,
        ],
    );
    Ok(())
}

#[test]
fn dmdw_rigid_body_helpers_reject_invalid_inputs() {
    let empty_positions = ndarray::Array2::<Real>::zeros((0, 3));
    let empty_masses = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_center_of_mass(empty_positions.view(), empty_masses.view()),
        Err(DebyeError::EmptyDmdwAtomTable)
    ));

    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
    let bad_masses = ndarray::arr1(&[-1.0]);
    assert!(matches!(
        dmdw_inertia_tensor(positions.view(), bad_masses.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW atom mass",
            ..
        })
    ));

    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0]]);
    let masses = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_rigid_body_projection_modes(positions.view(), masses.view()),
        Err(DebyeError::TooFewDmdwRigidBodyAtoms { atoms: 1 })
    ));

    let collinear_positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let collinear_masses = ndarray::arr1(&[1.0, 1.0]);
    assert!(matches!(
        dmdw_rigid_body_projection_modes(collinear_positions.view(), collinear_masses.view()),
        Err(DebyeError::ZeroDmdwProjectionModeNorm { .. })
    ));
}

#[test]
fn dmdw_seed_projection_rejects_invalid_inputs() {
    let seed = ndarray::arr1(&[1.0, 2.0]);
    let bad_modes = ndarray::Array2::<Real>::zeros((3, 1));
    assert!(matches!(
        dmdw_project_seed_vector(seed.view(), bad_modes.view()),
        Err(DebyeError::InvalidDmdwProjectionShape { .. })
    ));

    let empty_seed = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_normalize_seed_vector(empty_seed.view()),
        Err(DebyeError::EmptyDmdwSeed)
    ));

    let zero_seed = ndarray::arr1(&[0.0, 0.0]);
    assert!(matches!(
        dmdw_normalize_seed_vector(zero_seed.view()),
        Err(DebyeError::ZeroDmdwSeedNorm)
    ));
}
