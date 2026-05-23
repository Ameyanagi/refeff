use super::{support::*, *};

#[test]
fn dmdw_phonon_coupling_matches_feff_phonon_coupling_formulas() -> Result<(), DebyeError> {
    let pds_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0, 30.0]);
    let a2f_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0, 1.5]);

    let coupling = dmdw_phonon_coupling(
        pds_energy.view(),
        phonon_dos.view(),
        a2f_energy.view(),
        eliashberg.view(),
    )?;

    assert_eq!(coupling.point_count(), 3);
    assert_vector_close(&coupling.energy_hartree, &[0.001, 0.002, 0.004]);
    assert_vector_close(
        &coupling.energy_ev,
        &[
            0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.002 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.004 * DMDW_COUPLING_ENERGY_HARTREE_EV,
        ],
    );
    assert_vector_close(&coupling.eliashberg, &[0.5, 1.0, 1.5]);
    assert_vector_close(&coupling.matrix_element, &[0.05, 0.05, 0.05]);
    assert_dmdw_close(
        coupling.normalization,
        (10.0 * 0.001 + 20.0 * 0.001 + 30.0 * 0.002) * DMDW_COUPLING_NORM_HARTREE_EV,
    );
    Ok(())
}

#[test]
fn dmdw_pole_weighted_a2f_matches_feff_diagnostic_formulas() -> Result<(), DebyeError> {
    let pds_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0, 30.0]);
    let a2f_energy = ndarray::arr1(&[0.001, 0.002, 0.004]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0, 1.5]);
    let coupling = dmdw_phonon_coupling(
        pds_energy.view(),
        phonon_dos.view(),
        a2f_energy.view(),
        eliashberg.view(),
    )?;
    let spectrum = DmdwLanczosPoleSpectrum {
        expected_poles: 3,
        squared_angular_frequencies: ndarray::arr1(&[
            (5.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
            (10.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
            (20.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2),
        ]),
        angular_frequencies: ndarray::arr1(&[
            5.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
            10.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
            20.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI,
        ]),
        frequencies: ndarray::arr1(&[5.0, 10.0, 20.0]),
        weights: ndarray::arr1(&[0.2, 0.3, 0.5]),
        imaginary_warnings: Vec::new(),
    };

    let diagnostic = dmdw_pole_weighted_a2f(&spectrum, &coupling)?;
    let expected_pole_weights = [
        0.05 * 0.2 * coupling.normalization,
        0.05 * 0.3 * coupling.normalization,
        0.05 * 0.5 * coupling.normalization,
    ];
    let expected_energies = [
        5.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
        10.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
        20.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI * DMDW_A2F_POLE_ANGULAR_TO_EV,
    ];
    let expected_total = expected_pole_weights.iter().sum::<Real>();
    let expected_w0 = expected_energies
        .iter()
        .zip(expected_pole_weights.iter())
        .map(|(&energy, &weight)| energy * weight)
        .sum::<Real>()
        / expected_total;
    let expected_lambda = expected_energies
        .iter()
        .zip(expected_pole_weights.iter())
        .map(|(&energy, &weight)| 2.0 * weight / energy)
        .sum::<Real>();

    assert_vector_close(&diagnostic.lanczos_frequency_thz, &[5.0, 10.0, 20.0]);
    assert_vector_close(&diagnostic.lanczos_weight, &[0.2, 0.3, 0.5]);
    assert_dmdw_close(diagnostic.normalization, coupling.normalization);
    assert_vector_close(&diagnostic.pole_energy_ev, &expected_energies);
    assert_vector_close(&diagnostic.pole_weight, &expected_pole_weights);
    assert_dmdw_close(diagnostic.mass_enhancement, expected_lambda);
    assert_dmdw_close(diagnostic.characteristic_energy_ev, expected_w0);
    Ok(())
}

#[test]
fn dmdw_pole_weighted_a2f_rejects_unmatched_and_zero_weight_inputs() {
    let coupling = DmdwPhononCoupling {
        energy_hartree: ndarray::arr1(&[0.001]),
        energy_ev: ndarray::arr1(&[0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV]),
        eliashberg: ndarray::arr1(&[0.5]),
        matrix_element: ndarray::arr1(&[0.05]),
        normalization: 1.0,
    };
    let unmatched = DmdwLanczosPoleSpectrum {
        expected_poles: 1,
        squared_angular_frequencies: ndarray::arr1(&[
            (100.0_f64 * DMDW_A2F_DIAGNOSTIC_TWO_PI).powi(2)
        ]),
        angular_frequencies: ndarray::arr1(&[100.0 * DMDW_A2F_DIAGNOSTIC_TWO_PI]),
        frequencies: ndarray::arr1(&[100.0]),
        weights: ndarray::arr1(&[1.0]),
        imaginary_warnings: Vec::new(),
    };
    assert!(matches!(
        dmdw_pole_weighted_a2f(&unmatched, &coupling),
        Err(DebyeError::UnmatchedDmdwA2fPole { pole_index: 0, .. })
    ));

    let imaginary = DmdwLanczosPoleSpectrum {
        expected_poles: 1,
        squared_angular_frequencies: ndarray::arr1(&[-1.0]),
        angular_frequencies: ndarray::arr1(&[-1.0]),
        frequencies: ndarray::arr1(&[-1.0 / (2.0 * std::f64::consts::PI)]),
        weights: ndarray::arr1(&[1.0]),
        imaginary_warnings: Vec::new(),
    };
    assert!(matches!(
        dmdw_pole_weighted_a2f(&imaginary, &coupling),
        Err(DebyeError::NonPositive {
            name: "DMDW a2f total pole weight",
            ..
        })
    ));
}

#[test]
fn dmdw_type2_pole_weighted_a2f_matches_feff_unit_seed_accumulation() -> Result<(), DebyeError> {
    let (force_blocks, masses) = sample_dmdw_type2_blocks();
    let coupling = sample_dmdw_type2_coupling();
    let groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![0],
    }];

    let all_displacements =
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 0, 1, &coupling)?;
    let selected_y =
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 2, 1, &coupling)?;

    let scale = DMDW_DYNAMICAL_MATRIX_SCALE / masses[0];
    let expected_all_angular = (0..3)
        .map(|component| (force_blocks[(0, 0, component, component)] * scale).sqrt())
        .sum::<Real>()
        / 3.0;
    let expected_y_angular = (force_blocks[(0, 0, 1, 1)] * scale).sqrt();
    let expected_all_energy = expected_all_angular * DMDW_A2F_POLE_ANGULAR_TO_EV;
    let expected_y_energy = expected_y_angular * DMDW_A2F_POLE_ANGULAR_TO_EV;
    let expected_weight = coupling.matrix_element[0] * coupling.normalization;

    assert_vector_close(
        &all_displacements.lanczos_frequency_thz,
        &[expected_all_angular / DMDW_A2F_DIAGNOSTIC_TWO_PI],
    );
    assert_vector_close(&all_displacements.lanczos_weight, &[1.0]);
    assert_vector_close(&all_displacements.pole_energy_ev, &[expected_all_energy]);
    assert_vector_close(&all_displacements.pole_weight, &[expected_weight]);
    assert_dmdw_close(
        all_displacements.mass_enhancement,
        2.0 * expected_weight / expected_all_energy,
    );
    assert_dmdw_close(
        all_displacements.characteristic_energy_ev,
        expected_all_energy,
    );

    assert_vector_close(
        &selected_y.lanczos_frequency_thz,
        &[expected_y_angular / DMDW_A2F_DIAGNOSTIC_TWO_PI],
    );
    assert_vector_close(&selected_y.lanczos_weight, &[1.0]);
    assert_vector_close(&selected_y.pole_energy_ev, &[expected_y_energy]);
    assert_vector_close(&selected_y.pole_weight, &[expected_weight]);
    assert_dmdw_close(selected_y.characteristic_energy_ev, expected_y_energy);
    Ok(())
}

#[test]
fn dmdw_type2_pole_weighted_a2f_rejects_invalid_metadata() {
    let (force_blocks, masses) = sample_dmdw_type2_blocks();
    let coupling = sample_dmdw_type2_coupling();
    let groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![masses.len()],
    }];

    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &[], 0, 1, &coupling),
        Err(DebyeError::EmptyDmdwType2UniqueAtomTable)
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: Vec::new()
            }],
            0,
            1,
            &coupling
        ),
        Err(DebyeError::EmptyDmdwType2CenterAtomGroup { group: 0 })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(force_blocks.view(), masses.view(), &groups, 0, 1, &coupling),
        Err(DebyeError::InvalidDmdwType2CenterAtomIndex { group: 0, .. })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: vec![0]
            }],
            4,
            1,
            &coupling
        ),
        Err(DebyeError::InvalidDmdwType2DisplacementOption { option: 4 })
    ));
    assert!(matches!(
        dmdw_type2_pole_weighted_a2f(
            force_blocks.view(),
            masses.view(),
            &[DmdwType2AtomGroup {
                center_atom_indices: vec![0]
            }],
            0,
            0,
            &coupling
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW type 2 Lanczos pole count",
            ..
        })
    ));
}

#[test]
fn dmdw_phonon_coupling_rejects_invalid_inputs() {
    let energy = ndarray::arr1(&[0.001, 0.002]);
    let short_energy = ndarray::arr1(&[0.001]);
    let phonon_dos = ndarray::arr1(&[10.0, 20.0]);
    let eliashberg = ndarray::arr1(&[0.5, 1.0]);

    assert!(matches!(
        dmdw_phonon_coupling(
            short_energy.view(),
            phonon_dos.view(),
            energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::InvalidDmdwCouplingTableShape { .. })
    ));

    let empty = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_phonon_coupling(empty.view(), empty.view(), empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwCouplingTable)
    ));

    let bad_dos = ndarray::arr1(&[10.0, 0.0]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            bad_dos.view(),
            energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::NonPositiveDmdwPhononDensity { row: 2, .. })
    ));

    let shifted_energy = ndarray::arr1(&[0.001, 0.002_1]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            phonon_dos.view(),
            shifted_energy.view(),
            eliashberg.view()
        ),
        Err(DebyeError::MismatchedDmdwCouplingEnergyGrid { row: 2, .. })
    ));

    let nonfinite = ndarray::arr1(&[0.5, Real::NAN]);
    assert!(matches!(
        dmdw_phonon_coupling(
            energy.view(),
            phonon_dos.view(),
            energy.view(),
            nonfinite.view()
        ),
        Err(DebyeError::NonFinite {
            name: "DMDW Eliashberg coupling",
            ..
        })
    ));
}
