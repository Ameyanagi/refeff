use super::{support::*, *};

#[test]
fn xsph_nrixs_transition_indices_match_feff_nrixs_init_reference() -> Result<(), XsphError> {
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: -1,
        multipole: 3,
        max_angular_momentum: 24,
    })?;

    assert_eq!(indices.initial_j2, 1);
    assert_eq!(indices.final_j2_max, 7);
    assert_eq!(indices.final_lj_max, 3);
    assert_eq!(indices.final_state_capacity, 48);
    assert_eq!(
        indices.transitions,
        vec![
            XsphNrixsTransitionIndex {
                final_state_kappa: -1,
                decomposition_channel: 0,
                total_angular_momentum_channel: 0,
                orbital_angular_momentum: 0,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: 1,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: -2,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: 2,
                decomposition_channel: 2,
                total_angular_momentum_channel: 2,
                orbital_angular_momentum: 2,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: -3,
                decomposition_channel: 2,
                total_angular_momentum_channel: 2,
                orbital_angular_momentum: 2,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: 3,
                decomposition_channel: 3,
                total_angular_momentum_channel: 3,
                orbital_angular_momentum: 3,
            },
            XsphNrixsTransitionIndex {
                final_state_kappa: -4,
                decomposition_channel: 3,
                total_angular_momentum_channel: 3,
                orbital_angular_momentum: 3,
            },
        ]
    );
    Ok(())
}

#[test]
fn xsph_nrixs_transition_indices_honor_feff_angular_capacity() -> Result<(), XsphError> {
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: -1,
        multipole: 3,
        max_angular_momentum: 1,
    })?;

    assert_eq!(indices.final_state_capacity, 48);
    assert_eq!(
        indices
            .transitions
            .iter()
            .map(|transition| transition.orbital_angular_momentum)
            .collect::<Vec<_>>(),
        vec![0, 1, 1]
    );

    let l_shell = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: -2,
        multipole: 0,
        max_angular_momentum: 24,
    })?;
    assert_eq!(l_shell.final_state_capacity, 40);
    assert_eq!(l_shell.transitions.len(), 1);
    assert_eq!(l_shell.transitions[0].final_state_kappa, -2);
    assert_eq!(l_shell.transitions[0].orbital_angular_momentum, 1);
    Ok(())
}

#[test]
fn xsph_nrixs_transition_indices_reject_invalid_inputs() {
    assert_eq!(
        xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
            initial_kappa: 0,
            multipole: 1,
            max_angular_momentum: 2,
        }),
        Err(XsphError::ZeroKappa)
    );
    assert!(matches!(
        xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
            initial_kappa: i32::MIN,
            multipole: 1,
            max_angular_momentum: 2,
        }),
        Err(XsphError::IntegerOutOfRange {
            name: "initial_kappa",
            ..
        })
    ));
}

#[test]
fn xsph_bcoef_transition_indices_match_feff_dipole_and_e2_rows() -> Result<(), XsphError> {
    let dipole_only = xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
        initial_kappa: -1,
        higher_multipole_selector: 0,
        max_angular_momentum: 4,
    })?;
    assert_eq!(dipole_only.final_kappas, arr1(&[-2, 1, 0, 0, 0, 0, 0, 0]));
    assert_eq!(dipole_only.j_indices, arr1(&[2, 1, 0, 0, 0, 0, 0, 0]));
    assert_eq!(dipole_only.orbital_l, arr1(&[1, 1, -1, -1, -1, -1, -1, -1]));
    assert_eq!(dipole_only.transitions[0].slot_1based, 1);
    assert_eq!(dipole_only.transitions[0].final_kappa, -2);
    assert_eq!(dipole_only.transitions[2].orbital_l, -1);

    let e2 = xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
        initial_kappa: -1,
        higher_multipole_selector: 2,
        max_angular_momentum: 4,
    })?;
    assert_eq!(e2.final_kappas, arr1(&[-2, 1, 0, 0, 0, -1, 2, -3]));
    assert_eq!(e2.j_indices, arr1(&[2, 1, 0, 0, 0, 1, 2, 3]));
    assert_eq!(e2.orbital_l, arr1(&[1, 1, -1, -1, -1, 0, 2, 2]));

    let plan = xsph_xsect_transition_plan(XsphXsectTransitionPlanInput {
        photon_energy: 1.0,
        selected_higher_multipole: Some(XsphTransitionMultipole::ElectricQuadrupole),
        transition_direction: 0,
        initial_kappa: -1,
        final_kappas: e2.final_kappas.view(),
        orbital_l: e2.orbital_l.view(),
        active_len: e2.final_kappas.len(),
    })?;
    assert_eq!(
        plan.transitions
            .iter()
            .map(|transition| transition.transition_index_1based)
            .collect::<Vec<_>>(),
        vec![1, 2, 6, 7, 8]
    );

    Ok(())
}

#[test]
fn xsph_bcoef_transition_indices_preserve_feff_m1_and_cap_rules() -> Result<(), XsphError> {
    let m1 = xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
        initial_kappa: -1,
        higher_multipole_selector: 1,
        max_angular_momentum: 4,
    })?;
    assert_eq!(m1.final_kappas, arr1(&[-2, 1, 0, 0, 0, -1, 2, 0]));
    assert_eq!(m1.j_indices, arr1(&[2, 1, 0, 0, 0, 1, 2, 0]));
    assert_eq!(m1.orbital_l, arr1(&[1, 1, -1, -1, -1, 0, 2, -1]));

    let capped = xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
        initial_kappa: -1,
        higher_multipole_selector: 2,
        max_angular_momentum: 1,
    })?;
    assert_eq!(capped.final_kappas, arr1(&[-2, 1, 0, 0, 0, -1, 0, 0]));
    assert_eq!(capped.j_indices, arr1(&[2, 1, 0, 0, 0, 1, 0, 0]));
    assert_eq!(capped.orbital_l, arr1(&[1, 1, -1, -1, -1, 0, -1, -1]));

    Ok(())
}

#[test]
fn xsph_bcoef_transition_indices_reject_invalid_inputs() {
    assert_eq!(
        xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
            initial_kappa: 0,
            higher_multipole_selector: 0,
            max_angular_momentum: 4,
        }),
        Err(XsphError::ZeroKappa)
    );
    assert!(matches!(
        xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
            initial_kappa: -1,
            higher_multipole_selector: 3,
            max_angular_momentum: 4,
        }),
        Err(XsphError::IntegerOutOfRange {
            name: "le2",
            value: 3
        })
    ));
    assert!(matches!(
        xsph_bcoef_transition_indices(XsphBcoefTransitionIndicesInput {
            initial_kappa: i32::MIN,
            higher_multipole_selector: 0,
            max_angular_momentum: 4,
        }),
        Err(XsphError::IntegerOutOfRange {
            name: "bcoef_dipole_kappa",
            ..
        })
    ));
}

#[test]
fn xsph_tdlda_channel_basis_matches_feff_getmat_k_edge_reference() -> Result<(), XsphError> {
    let basis = xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
        core_hole_index_1based: 1,
        initial_l: 0,
        plus_basis_count: 3,
        minus_basis_count: 0,
        orbital_kappas: arr1(&[]).view(),
        valence_occupations: arr1(&[]).view(),
        basis_selector: 2,
    })?;

    assert_eq!(basis.plus_basis_count, 3);
    assert_eq!(basis.minus_basis_count, 0);
    assert_eq!(basis.matrix_size, 9);
    assert_eq!(
        basis.rows,
        vec![
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 1,
                final_m2: 1,
                final_kappa: 1,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -2,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 1,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -1,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: 1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 3,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -1,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 1,
                final_m2: 1,
                final_kappa: 1,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -4,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 1,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -3,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: 1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 3,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -3,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 1,
                final_m2: 1,
                final_kappa: 1,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -6,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: -1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 1,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -5,
            },
            XsphTdldaChannelBasisRow {
                initial_j2: 1,
                initial_m2: 1,
                initial_kappa: -1,
                final_j2: 3,
                final_m2: 3,
                final_kappa: -2,
                core_orbital_index_1based: 1,
                projector_orbital_selector: -5,
            },
        ]
    );
    Ok(())
}

#[test]
fn xsph_tdlda_channel_basis_maps_occupied_projectors_like_feff_ibasis_zero() -> Result<(), XsphError>
{
    let basis = xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
        core_hole_index_1based: 4,
        initial_l: 1,
        plus_basis_count: 1,
        minus_basis_count: 1,
        orbital_kappas: arr1(&[2, -3, -1]).view(),
        valence_occupations: arr1(&[1.0, 2.0, 2.0]).view(),
        basis_selector: 0,
    })?;

    assert_eq!(basis.plus_basis_count, 1);
    assert_eq!(basis.minus_basis_count, 1);
    assert_eq!(basis.matrix_size, 12);
    assert_eq!(
        basis
            .rows
            .iter()
            .map(|row| row.projector_orbital_selector)
            .collect::<Vec<_>>(),
        vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    );
    assert_eq!(
        basis
            .rows
            .iter()
            .map(|row| row.core_orbital_index_1based)
            .collect::<Vec<_>>(),
        vec![3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4]
    );
    assert_eq!(
        basis
            .rows
            .iter()
            .map(|row| (row.initial_kappa, row.final_kappa))
            .collect::<Vec<_>>(),
        vec![
            (1, 2),
            (1, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, -3),
            (-2, -3),
            (-2, -3),
            (-2, -3),
            (1, -1),
            (-2, -1),
            (-2, -1),
        ]
    );
    Ok(())
}

#[test]
fn xsph_tdlda_projector_selector_decodes_feff_getmat_slots() -> Result<(), XsphError> {
    assert_eq!(
        xsph_tdlda_decode_projector_selector(3)?,
        XsphTdldaProjectorSelector::OccupiedOrbital { orbital_index: 2 }
    );
    assert_eq!(
        xsph_tdlda_decode_projector_selector(-1)?,
        XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index: 0,
            positive_final_kappa: false,
        }
    );
    assert_eq!(
        xsph_tdlda_decode_projector_selector(-2)?,
        XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index: 0,
            positive_final_kappa: true,
        }
    );
    assert_eq!(
        xsph_tdlda_decode_projector_selector(-5)?,
        XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index: 2,
            positive_final_kappa: false,
        }
    );
    assert_eq!(
        xsph_tdlda_decode_projector_selector(-6)?,
        XsphTdldaProjectorSelector::GeneratedBasis {
            basis_index: 2,
            positive_final_kappa: true,
        }
    );

    assert!(matches!(
        xsph_tdlda_decode_projector_selector(0),
        Err(XsphError::IntegerOutOfRange {
            name: "tdlda_projector_selector",
            value: 0,
        })
    ));
    Ok(())
}

#[test]
fn xsph_tdlda_channel_basis_clamps_counts_and_rejects_invalid_inputs() -> Result<(), XsphError> {
    let clamped = xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
        core_hole_index_1based: 1,
        initial_l: 0,
        plus_basis_count: 0,
        minus_basis_count: -2,
        orbital_kappas: arr1(&[]).view(),
        valence_occupations: arr1(&[]).view(),
        basis_selector: 2,
    })?;
    assert_eq!(clamped.plus_basis_count, 1);
    assert_eq!(clamped.minus_basis_count, 0);
    assert_eq!(clamped.matrix_size, 3);

    assert!(matches!(
        xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
            core_hole_index_1based: 0,
            initial_l: 0,
            plus_basis_count: 1,
            minus_basis_count: 0,
            orbital_kappas: arr1(&[]).view(),
            valence_occupations: arr1(&[]).view(),
            basis_selector: 2,
        }),
        Err(XsphError::InvalidOneBasedIndex {
            name: "tdlda_core_hole_index",
            ..
        })
    ));
    assert!(matches!(
        xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
            core_hole_index_1based: 1,
            initial_l: -1,
            plus_basis_count: 1,
            minus_basis_count: 0,
            orbital_kappas: arr1(&[]).view(),
            valence_occupations: arr1(&[]).view(),
            basis_selector: 2,
        }),
        Err(XsphError::NegativeAngularMomentum {
            name: "tdlda_initial_l",
            ..
        })
    ));
    assert!(matches!(
        xsph_tdlda_channel_basis(XsphTdldaChannelBasisInput {
            core_hole_index_1based: 1,
            initial_l: 4,
            plus_basis_count: 4,
            minus_basis_count: 2,
            orbital_kappas: arr1(&[]).view(),
            valence_occupations: arr1(&[]).view(),
            basis_selector: 2,
        }),
        Err(XsphError::SizeOutOfRange {
            name: "tdlda_matrix_size",
            ..
        })
    ));
    Ok(())
}

#[test]
fn xsph_xsect_bcoef_weights_match_feff_average_trace() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 2))?;

    assert_eq!(weights.selected_spin_index, 0);
    assert_eq!(weights.final_kappas, arr1(&[-2, 1, 0, 0, 0, -1, 2, -3]));
    assert_eq!(weights.orbital_l, arr1(&[1, 1, -1, -1, -1, 0, 2, 2]));
    assert_xsect_bcoef_diagonal(
        &weights,
        &[
            Complex::new(-1.0 / 3.0, 0.0),
            Complex::new(-1.0 / 3.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.2, 0.0),
            Complex::new(0.2, 0.0),
            Complex::new(0.2, 0.0),
        ],
    );
    assert_xsect_bcoef_off_diagonal_zero(&weights);
    Ok(())
}

#[test]
fn xsph_xsect_bcoef_weights_select_feff_spin_column() -> Result<(), XsphError> {
    let weights = xsph_xsect_bcoef_weights(xsect_bcoef_average_input(1, 2))?;

    assert_eq!(weights.selected_spin_index, 1);
    assert_xsect_bcoef_diagonal(
        &weights,
        &[
            Complex::new(-1.0 / 6.0, 0.0),
            Complex::new(-1.0 / 6.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.1, 0.0),
            Complex::new(0.1, 0.0),
            Complex::new(0.1, 0.0),
        ],
    );
    assert_xsect_bcoef_off_diagonal_zero(&weights);
    Ok(())
}

#[test]
fn xsph_xsect_bcoef_weights_reject_invalid_inputs() {
    let mut zero_kappa = xsect_bcoef_average_input(0, 2);
    zero_kappa.initial_kappa = 0;
    assert_eq!(
        xsph_xsect_bcoef_weights(zero_kappa),
        Err(XsphError::ZeroKappa)
    );

    let mut bad_multipole = xsect_bcoef_average_input(0, 2);
    bad_multipole.higher_multipole_selector = 3;
    assert_eq!(
        xsph_xsect_bcoef_weights(bad_multipole),
        Err(XsphError::IntegerOutOfRange {
            name: "le2",
            value: 3,
        })
    );

    assert_eq!(
        xsph_xsect_bcoef_weights(xsect_bcoef_average_input(0, 0)),
        Err(XsphError::Angular(
            crate::AngularError::InvalidSpinChannelCount { value: 0 }
        ))
    );

    let mut nonfinite = xsect_bcoef_average_input(0, 2);
    nonfinite.polarization_tensor[2][0] = Complex::new(0.0, f64::INFINITY);
    assert_eq!(
        xsph_xsect_bcoef_weights(nonfinite),
        Err(XsphError::Angular(
            crate::AngularError::NonFinitePolarizationTensor { row: 1, column: -1 }
        ))
    );

    assert_eq!(
        xsph_xsect_bcoef_weights(xsect_bcoef_average_input(1, 3)),
        Err(XsphError::SizeOutOfRange {
            name: "xsect_bcoef_spin_index",
            value: 2,
        })
    );
}

fn xsect_bcoef_average_input(spin: i32, spin_channels: usize) -> XsphXsectBcoefWeightsInput {
    XsphXsectBcoefWeightsInput {
        max_angular_momentum: 4,
        initial_kappa: -1,
        polarization: 0,
        polarization_tensor: [[Complex::new(0.0, 0.0); 3]; 3],
        higher_multipole_selector: 2,
        spin,
        spin_channels,
        spin_vector_angle: 0.0,
    }
}

fn assert_xsect_bcoef_diagonal(weights: &XsphXsectBcoefWeights, expected: &[Complex; 8]) {
    assert_eq!(weights.diagonal_weights.len(), expected.len());
    for (transition, &expected_value) in expected.iter().enumerate() {
        assert_complex_close(weights.diagonal_weights[transition], expected_value);
        assert_complex_close(
            weights.trace_weights[(transition, transition)],
            expected_value,
        );
    }
}

fn assert_xsect_bcoef_off_diagonal_zero(weights: &XsphXsectBcoefWeights) {
    assert_eq!(weights.trace_weights.shape(), &[8, 8]);
    for transition2 in 0..8 {
        for transition1 in 0..8 {
            if transition1 != transition2 {
                assert_complex_close(
                    weights.trace_weights[(transition2, transition1)],
                    Complex::new(0.0, 0.0),
                );
            }
        }
    }
}

#[test]
fn xsph_jas_bessel_functions_match_feff_besjnjas_reference() -> Result<(), XsphError> {
    let small = xsph_jas_bessel_functions(Complex::new(0.7, 0.2), 4)?;
    assert_eq!(small.j.len(), 6);
    assert_complex_close(
        small.j[0],
        Complex::new(9.260_369_547_275_705e-1, -4.459_588_911_838_349e-2),
    );
    assert_complex_close(
        small.y[0],
        Complex::new(-9.814_947_532_033_563e-1, 4.657_188_065_076_81e-1),
    );
    assert_complex_close(
        small.j[2],
        Complex::new(2.940_719_684_553_348_4e-2, 1.748_613_789_282_074e-2),
    );
    assert_complex_close(
        small.y[5],
        Complex::new(5.417_504_841_403_511e2, 6.485_978_713_442_445e3),
    );

    let medium = xsph_jas_bessel_functions(Complex::new(3.5, 0.4), 5)?;
    assert_eq!(medium.j.len(), 7);
    assert_complex_close(
        medium.j[0],
        Complex::new(-1.193_503_546_160_998e-1, -9.626_046_299_195_844e-2),
    );
    assert_complex_close(
        medium.y[2],
        Complex::new(-1.084_354_979_749_215_2e-1, 1.105_971_099_923_874_2e-1),
    );
    assert_complex_close(
        medium.j[6],
        Complex::new(7.807_752_079_869_525e-3, 5.166_725_723_488_525e-3),
    );

    let large = xsph_jas_bessel_functions(Complex::new(12.0, 0.5), 12)?;
    assert_eq!(large.j.len(), 14);
    assert_complex_close(
        large.j[0],
        Complex::new(-4.880_955_623_263_242e-2, 3.867_775_954_451_103e-2),
    );
    assert_complex_close(
        large.y[6],
        Complex::new(3.226_385_529_939_757e-2, -3.862_116_086_289_696e-2),
    );
    assert_complex_close(
        large.j[13],
        Complex::new(3.175_663_597_984_340_6e-2, 9.325_892_014_613_632e-3),
    );
    assert_complex_close(
        large.y[13],
        Complex::new(-1.988_938_334_052_988_7e-1, 4.602_641_317_092_328e-2),
    );

    Ok(())
}

#[test]
fn xsph_jas_bessel_functions_reject_invalid_inputs() {
    assert!(matches!(
        xsph_jas_bessel_functions(Complex::new(1.0, 0.0), 40),
        Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: 40,
            ljmax: 39,
        })
    ));
    assert!(matches!(
        xsph_jas_bessel_functions(Complex::new(0.0, 0.1), 2),
        Err(XsphError::Bessel(BesselError::NonPositiveRealArgument {
            real: 0.0
        }))
    ));
}

#[test]
fn xsph_minimize_calculations_matches_feff_reference() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 8)?;

    assert_eq!(plan.max_lj, 6);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2, 4, -3, -1]));
    assert_eq!(
        plan.calculations,
        arr2(&[[2, 5, 1], [4, 4, 2], [-3, 6, 1], [5, 0, 0]])
    );
    Ok(())
}

#[test]
fn xsph_minimize_calculations_honors_active_prefix() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 5)?;

    assert_eq!(plan.max_lj, 5);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2]));
    assert_eq!(plan.calculations, arr2(&[[2, 5, 1], [4, 4, 2], [-3, 3, 1]]));
    Ok(())
}

#[test]
fn xsph_lj_needed_flags_match_feff_reference() -> Result<(), XsphError> {
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);
    let index_map = arr1(&[1, 2, -1, 3, -2, 4, -3, -1]);

    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 1)?,
        arr1(&[0, 1, 1, 0, 0, 1, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 2)?,
        arr1(&[0, 1, 0, 0, 1, 0, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 3)?,
        arr1(&[0, 0, 0, 1, 0, 0, 1])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 4)?,
        arr1(&[1, 0, 0, 0, 0, 0, 0])
    );
    Ok(())
}

#[test]
fn xsph_q_bessel_table_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(0.35, radii.view(), 4)?;

    assert_eq!(table.shape(), &[4, 5]);
    assert_eq!(table.strides(), &[1, 4]);
    let expected = arr2(&[
        [
            9.997_958_458_381_769e-1,
            1.166_523_756_252_462e-2,
            8.165_952_107_648_562e-5,
            4.083_055_447_551_5e-7,
            1.587_874_544_380_937_5e-9,
        ],
        [
            9.797_080_213_012_896e-1,
            1.152_437_384_397_447_3e-1,
            8.095_451_039_379_387e-3,
            4.055_621_228_179_726_3e-4,
            1.579_141_698_006_595_3e-5,
        ],
        [
            8.261_173_577_085_878e-1,
            3.129_012_474_446_291e-1,
            6.788_620_641_892_411e-2,
            1.036_640_216_929_531_6e-2,
            1.223_141_376_378_009_1e-3,
        ],
        [
            9.385_522_838_839_835e-2,
            -9.429_243_227_927_261e-2,
            -1.342_662_707_938_009e-1,
            -1.612_046_859_156_612_8e-3,
            1.326_542_239_346_443e-1,
        ],
    ]);
    for ((row, column), &expected_value) in expected.indexed_iter() {
        assert_close(table[(row, column)], expected_value);
    }
    Ok(())
}

#[test]
fn xsph_q_bessel_table_applies_feff_large_argument_cutoff() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(1.0e8, radii.view(), 4)?;

    let expected_first_row = [
        4.205_477_931_907_825e-8,
        9.072_704_282_365_188e-8,
        -4.205_475_210_096_54e-8,
        -9.072_706_385_102_794e-8,
        4.205_468_859_202_071e-8,
    ];
    for (column, &expected_value) in expected_first_row.iter().enumerate() {
        assert_close(table[(0, column)], expected_value);
    }
    for row in 1..4 {
        for column in 0..5 {
            assert_close(table[(row, column)], 0.0);
        }
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_matches_feff_getoccnorm_reference() -> Result<(), XsphError> {
    let cases = [
        (1, 1, 0.5),
        (6, 4, 0.25),
        (8, 4, 0.5),
        (26, 9, 1.0 / 3.0),
        (29, 10, 0.5),
        (47, 17, 0.5),
        (58, 15, 1.0 / 6.0),
        (79, 24, 0.5),
        (80, 24, 1.0),
        (92, 22, 0.5),
        (100, 16, 1.0),
        (100, 29, 1.0),
    ];

    for (atomic_number, hole_index, expected) in cases {
        let actual = xsph_occupation_normalization(atomic_number, hole_index)?;
        assert_close_tol(actual, expected, 5.0e-13);
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_rejects_invalid_inputs() {
    assert_eq!(
        xsph_occupation_normalization(0, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 0,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(101, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 101,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 0),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 0,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 30),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 30,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(92, 27),
        Err(XsphError::ZeroOccupationNormDenominator { hole_index: 27 })
    );
}
