use ndarray::ShapeBuilder;

use super::*;

#[test]
fn compton_rotation_matches_feff_reference() -> Result<(), ComptonError> {
    let axis_angle = compton_rotation_axis_angle([0.0, 0.0, 1.0], [0.35, -0.25, 0.92])?;
    assert_vector_close(axis_angle.axis, [0.25, 0.35, -0.0], 1.0e-15);
    assert_close(axis_angle.theta, 0.437325754687105, 1.0e-15);

    let rotation = compton_rotation_matrix(axis_angle.axis, axis_angle.theta)?;
    assert_close(rotation[(0, 0)], 0.937682259061576, 1.0e-15);
    assert_close(rotation[(1, 0)], 0.044512672098874, 1.0e-15);
    assert_close(rotation[(2, 0)], -0.344631111572645, 1.0e-15);
    assert_close(rotation[(0, 1)], 0.044512672098874, 1.0e-15);
    assert_close(rotation[(1, 1)], 0.968205234215090, 1.0e-15);
    assert_close(rotation[(2, 1)], 0.246165079694746, 1.0e-15);
    assert_close(rotation[(0, 2)], 0.344631111572645, 1.0e-15);
    assert_close(rotation[(1, 2)], -0.246165079694746, 1.0e-15);
    assert_close(rotation[(2, 2)], 0.905887493276667, 1.0e-15);

    let rotated = compton_rotate_vector(rotation.view(), [0.7, -0.2, 1.1])?;
    assert_vector_close(
        rotated,
        [1.026569269653238, -0.433263764038027, 0.706001448564533],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn compton_grid_matches_feff_reference() -> Result<(), ComptonError> {
    let grid = reference_grid()?;
    assert_eq!(grid.ns(), 4);
    assert_eq!(grid.nphi(), 5);
    assert_eq!(grid.nz(), 4);
    assert_eq!(grid.nzp(), 5);
    assert!(grid.rotate);
    assert_slice_close(
        grid.s.as_slice().unwrap_or(&[]),
        &[0.0, 0.75, 1.5, 2.25],
        1.0e-15,
    );
    assert_slice_close(
        grid.phi.as_slice().unwrap_or(&[]),
        &[
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
            3.0 * std::f64::consts::FRAC_PI_4,
            std::f64::consts::PI,
        ],
        1.0e-15,
    );
    assert_slice_close(
        grid.z.as_slice().unwrap_or(&[]),
        &[-1.2, -0.4, 0.4, 1.2],
        1.0e-15,
    );
    assert_slice_close(
        grid.zp.as_slice().unwrap_or(&[]),
        &[-1.5, -0.75, 0.0, 0.75, 1.5],
        1.0e-15,
    );
    assert_close(grid.rotation_matrix[(0, 0)], 0.937682259061576, 1.0e-15);
    assert_close(grid.rotation_matrix[(2, 2)], 0.905887493276667, 1.0e-15);
    Ok(())
}

#[test]
fn compton_profile_matches_feff_reference() -> Result<(), ComptonError> {
    let grid = reference_grid()?;
    let jzzp = reference_jzzp();

    assert_close(
        compton_profile(
            &grid,
            jzzp.view(),
            ComptonProfileInput {
                pq: 0.0,
                window: ComptonWindow::Rectangular,
                window_cutoff: 0.0,
            },
        )?,
        4.481999999999999,
        1.0e-14,
    );
    assert_close(
        compton_profile(
            &grid,
            jzzp.view(),
            ComptonProfileInput {
                pq: 1.35,
                window: ComptonWindow::Rectangular,
                window_cutoff: 0.0,
            },
        )?,
        1.284270509089719,
        1.0e-14,
    );
    assert_close(
        compton_profile(
            &grid,
            jzzp.view(),
            ComptonProfileInput {
                pq: 1.35,
                window: ComptonWindow::CosineSquared,
                window_cutoff: 1.0,
            },
        )?,
        0.541117104926063,
        1.0e-14,
    );
    assert_close(
        compton_profile(
            &grid,
            jzzp.view(),
            ComptonProfileInput {
                pq: 0.65,
                window: ComptonWindow::Unwindowed,
                window_cutoff: 0.0,
            },
        )?,
        3.454016879329959,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn compton_profiles_match_scalar_profile_results() -> Result<(), ComptonError> {
    let grid = reference_grid()?;
    let jzzp = reference_jzzp();
    let momentum = Array1::from_vec(vec![0.0, 0.35, 1.35, 2.0]);

    let profiles = compton_profiles(
        &grid,
        jzzp.view(),
        momentum.view(),
        ComptonWindow::CosineSquared,
        1.0,
    )?;

    assert_eq!(profiles.len(), momentum.len());
    for (&pq, &profile) in momentum.iter().zip(profiles.iter()) {
        let scalar = compton_profile(
            &grid,
            jzzp.view(),
            ComptonProfileInput {
                pq,
                window: ComptonWindow::CosineSquared,
                window_cutoff: 1.0,
            },
        )?;
        assert_close(profile, scalar, 1.0e-14);
    }
    Ok(())
}

#[test]
fn compton_jzzp_matches_feff_reference() -> Result<(), ComptonError> {
    let grid = reference_grid()?;
    let jzzp = compton_jzzp(&grid, reference_density)?;

    assert_slice_close(
        &jzzp.row(0).iter().copied().collect::<Vec<_>>(),
        &[
            0.594570850449979,
            0.494977100812839,
            0.389227436640316,
            0.267164600812839,
            0.138945850449979,
        ],
        1.0e-14,
    );
    assert_slice_close(
        &jzzp.row(2).iter().copied().collect::<Vec<_>>(),
        &[
            0.318867984973198,
            0.408313642350487,
            0.475618692962606,
            0.484251142350487,
            0.470742984973198,
        ],
        1.0e-14,
    );
    assert_close(jzzp.sum(), 8.085360573551853, 1.0e-14);
    Ok(())
}

#[test]
fn compton_rhozzp_slice_matches_feff_reference() -> Result<(), ComptonError> {
    let grid = reference_grid()?;
    let slice = compton_rhozzp_slice(
        &grid,
        ComptonRhoZzpInput {
            sample_count: 7,
            base_z: 0.01,
        },
        reference_density,
    )?;

    assert_slice_close(
        &slice.z_prime.iter().copied().collect::<Vec<_>>(),
        &[0.01, 0.26, 0.51, 0.76, 1.01, 1.26, 1.51],
        1.0e-14,
    );
    assert_slice_close(
        &slice.rho.iter().copied().collect::<Vec<_>>(),
        &[
            0.999860011249437,
            0.966928166620989,
            0.878473726484003,
            0.749847110368395,
            0.601415511230681,
            0.453338247587858,
            0.321281052560816,
        ],
        1.0e-14,
    );
    Ok(())
}

#[test]
fn compton_helpers_reject_invalid_inputs() -> Result<(), ComptonError> {
    assert!(matches!(
        compton_rotation_axis_angle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        Err(ComptonError::ZeroVector { name: "a" })
    ));
    assert!(matches!(
        compton_rotation_matrix([0.0, 0.0, 0.0], 0.3),
        Err(ComptonError::ZeroVector { name: "axis" })
    ));
    assert!(matches!(
        compton_build_grid(ComptonGridInput {
            ns: 1,
            ..reference_grid_input()
        }),
        Err(ComptonError::InvalidGridCount {
            name: "ns",
            value: 1
        })
    ));

    let grid = reference_grid()?;
    let bad = Array2::zeros((grid.nz(), grid.nzp() + 1).f());
    assert!(matches!(
        compton_profile(
            &grid,
            bad.view(),
            ComptonProfileInput {
                pq: 0.0,
                window: ComptonWindow::Rectangular,
                window_cutoff: 0.0,
            },
        ),
        Err(ComptonError::InvalidJzzpShape { .. })
    ));
    assert!(matches!(
        compton_profile(
            &grid,
            reference_jzzp().view(),
            ComptonProfileInput {
                pq: 1.0,
                window: ComptonWindow::CosineSquared,
                window_cutoff: -1.0,
            },
        ),
        Err(ComptonError::InvalidWindowCutoff { value: -1.0 })
    ));
    assert!(matches!(
        compton_jzzp(&grid, |_, _| Ok(Real::NAN)),
        Err(ComptonError::NonFiniteDensity { .. })
    ));
    assert!(matches!(
        compton_rhozzp_slice(
            &grid,
            ComptonRhoZzpInput {
                sample_count: 1,
                base_z: 0.01,
            },
            reference_density,
        ),
        Err(ComptonError::InvalidGridCount {
            name: "rhozzp_samples",
            value: 1,
        })
    ));
    Ok(())
}

fn reference_grid_input() -> ComptonGridInput {
    ComptonGridInput {
        ns: 4,
        nphi: 5,
        nz: 4,
        nzp: 5,
        smax: 0.0,
        phimax: std::f64::consts::PI,
        zmax: 1.2,
        zpmax: 1.5,
        norman_radius: 2.25,
        qhat: [0.35, -0.25, 0.92],
    }
}

fn reference_grid() -> Result<ComptonGrid, ComptonError> {
    compton_build_grid(reference_grid_input())
}

fn reference_jzzp() -> Array2<Real> {
    Array2::from_shape_fn((4, 5).f(), |(iz, izp)| {
        let iz = iz as Real + 1.0;
        let izp = izp as Real + 1.0;
        0.12 * iz + 0.07 * izp + 0.015 * iz * izp
    })
}

fn reference_density(r: Vector3, rp: Vector3) -> Result<Real, ComptonError> {
    let r2 = r.iter().map(|value| value * value).sum::<Real>();
    let rp2 = rp.iter().map(|value| value * value).sum::<Real>();
    let dot = r
        .iter()
        .zip(rp)
        .map(|(left, right)| *left * right)
        .sum::<Real>();
    Ok((-r2 - 0.5 * rp2).exp() + 0.1 * dot)
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
        (actual - expected).abs()
    );
}

fn assert_vector_close(actual: Vector3, expected: Vector3, tolerance: Real) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}

fn assert_slice_close(actual: &[Real], expected: &[Real], tolerance: Real) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected, tolerance);
    }
}
