use super::{support::*, *};

#[test]
fn xsph_axafs_matches_feff_reference_output() -> Result<(), XsphError> {
    let energies = Array1::from_shape_fn(12, |index| {
        let i = index as Real + 1.0;
        Complex::new(0.015 * (i - 3.0).powi(2) + 0.012 * (i - 1.0), 0.002 * i)
    });
    let cross_section = Array1::from_shape_fn(12, |index| {
        let i = index as Real + 1.0;
        Complex::new(
            -0.03 * i,
            0.42 + 0.021 * i + 0.004 * i * i + 0.025 * (0.7 * i).sin(),
        )
    });

    let axafs = xsph_axafs(XsphAxafsInput {
        energies: energies.view(),
        cross_section: cross_section.view(),
        fermi_energy: 0.37,
        horizontal_count: 10,
        zero_wave_index: 2,
    })?;

    let expected = arr2(&[
        [10.803, 0.735, 0.439, 2.861_61e-1, 2.853_50e-1, 2.839_94e-3],
        [12.354, 2.286, 0.775, 3.059_48e-1, 3.037_03e-1, 7.391_87e-3],
        [14.721, 4.653, 1.105, 3.317_56e-1, 3.312_74e-1, 1.452_30e-3],
        [17.905, 7.837, 1.434, 3.666_23e-1, 3.675_12e-1, -2.418_88e-3],
        [
            21.905,
            11.837,
            1.763,
            4.111_97e-1,
            4.116_72e-1,
            -1.155_65e-3,
        ],
        [26.722, 16.653, 2.091, 4.634_28e-1, 4.628_25e-1, 1.303_51e-3],
        [
            32.354,
            22.286,
            2.419,
            5.195_33e-1,
            5.198_45e-1,
            -6.003_98e-4,
        ],
    ]);
    assert_eq!(axafs.rows.dim(), (7, XSPH_AXAFS_COLUMN_COUNT));
    for ((row, column), &expected_value) in expected.indexed_iter() {
        let tolerance = if column < 3 { 5.0e-4 } else { 5.0e-7 };
        assert_close_tol(axafs.rows[(row, column)], expected_value, tolerance);
    }
    Ok(())
}

#[test]
fn xsph_axafs_rejects_invalid_inputs() {
    let energies = Array1::from_vec(vec![Complex::new(0.0, 0.0); 5]);
    let cross_section = Array1::from_vec(vec![Complex::new(0.0, 1.0); 5]);

    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 5,
        }),
        Err(XsphError::InvalidAxafsGridIndex { .. })
    ));
    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 4,
            zero_wave_index: 1,
        }),
        Err(XsphError::InsufficientAxafsPoints { point_count: 2 })
    ));

    let mut bad_energy = energies.clone();
    bad_energy[2] = Complex::new(Real::NAN, 0.0);
    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: bad_energy.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 1,
        }),
        Err(XsphError::NonFiniteComplex {
            name: "energies",
            index: 2,
            ..
        })
    ));

    assert!(matches!(
        xsph_axafs(XsphAxafsInput {
            energies: energies.view(),
            cross_section: cross_section.view(),
            fermi_energy: 0.1,
            horizontal_count: 5,
            zero_wave_index: 1,
        }),
        Err(XsphError::SingularAxafsFit)
    ));
}

#[test]
fn xsph_planning_helpers_reject_invalid_inputs() {
    let kind = arr1(&[2]);
    let orbital_l = arr1(&[1]);
    let final_lj = arr1(&[2]);

    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 0),
        Err(XsphError::EmptyIndexSet)
    ));
    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 2),
        Err(XsphError::LengthTooShort { name: "kind", .. })
    ));

    let bad_lj = arr1(&[-1]);
    assert!(matches!(
        xsph_minimize_calculations(kind.view(), orbital_l.view(), bad_lj.view(), 1),
        Err(XsphError::NegativeAngularMomentum { .. })
    ));

    let index_map = arr1(&[1]);
    assert!(matches!(
        xsph_lj_needed_flags(1, final_lj.view(), index_map.view(), 1, 1),
        Err(XsphError::AngularMomentumOutOfRange { .. })
    ));
    assert!(matches!(
        xsph_lj_needed_flags(2, final_lj.view(), index_map.view(), 1, 0),
        Err(XsphError::NonPositiveCalculationIndex { .. })
    ));

    let overflow_map = arr1(&[i32::MIN]);
    assert!(matches!(
        xsph_lj_needed_flags(2, final_lj.view(), overflow_map.view(), 1, 1),
        Err(XsphError::IndexMapOverflow { .. })
    ));

    let radii = arr1(&[1.0]);
    assert!(matches!(
        xsph_q_bessel_table(Real::NAN, radii.view(), 4),
        Err(XsphError::NonFiniteScalar { name: "qtrans", .. })
    ));
    let bad_radii = arr1(&[Real::INFINITY]);
    assert!(matches!(
        xsph_q_bessel_table(1.0, bad_radii.view(), 4),
        Err(XsphError::NonFiniteScalar { name: "radius", .. })
    ));
    assert!(matches!(
        xsph_q_bessel_table(1.0, radii.view(), 40),
        Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: 40,
            ljmax: 39,
        })
    ));
    assert!(matches!(
        xsph_q_bessel_table(0.0, radii.view(), 4),
        Err(XsphError::Bessel(
            BesselError::NonPositiveRealArgument { .. }
        ))
    ));

    assert!(matches!(
        xsph_longitudinal_multipole_factor(0, 1, 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(1, 1, -1),
        Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(i32::MIN, 1, 1),
        Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
    ));
    assert!(matches!(
        xsph_longitudinal_multipole_factor(60, -60, 1),
        Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(0, 1, 0, 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, -1, 1),
        Err(XsphError::NegativeAngularMomentum {
            name: "bessel_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, 0, -1),
        Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_relativistic_multipole_factors(1, 1, 59, 59),
        Err(XsphError::IntegerOutOfRange {
            name: "bessel_l",
            ..
        })
    ));

    let lgind = arr1(&[0]);
    let ljind = arr1(&[0]);
    assert!(matches!(
        xsph_nrixs_transition_weights(0, 1, 4, 9, 3, lgind.view(), ljind.view(), 1),
        Err(XsphError::ZeroKappa)
    ));
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, -1, 3, lgind.view(), ljind.view(), 1),
        Err(XsphError::NegativeAngularMomentum { name: "jmax", .. })
    ));
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, lgind.view(), ljind.view(), 2),
        Err(XsphError::LengthTooShort { name: "lgind", .. })
    ));
    let bad_lgind = arr1(&[-1]);
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 4, 9, 3, bad_lgind.view(), ljind.view(), 1),
        Err(XsphError::NegativeAngularMomentum { name: "lgind", .. })
    ));
    let two_lgind = arr1(&[0, 0]);
    let two_ljind = arr1(&[0, 0]);
    assert!(matches!(
        xsph_nrixs_transition_weights(-1, 1, 0, 1, 0, two_lgind.view(), two_ljind.view(), 2),
        Err(XsphError::InsufficientGeneratedStates { .. })
    ));
}
