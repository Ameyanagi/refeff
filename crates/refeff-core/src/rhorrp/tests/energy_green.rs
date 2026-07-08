use super::{support::*, *};

#[test]
fn radial_interpolation_location_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference = [
        (0.0, 1, 0.0),
        (3.011_942_119_122_021_4e-1, 1, 0.0),
        (5.220_457_767_610_16e-1, 1, 2.499_999_999_999_995_6e-1),
        (1.061_836_546_545_359_6, 4, 7.999_999_999_999_998e-1),
        (1.0_f64.exp(), 6, 0.0),
    ];

    for (radius, index, fraction) in reference {
        let location = rhorrp_radial_interpolation_location(RhorrpRadialInterpolationInput {
            radius,
            x0: 0.7,
            dx: 0.2,
            radial_count: 6,
        })?;
        assert_eq!(location.index_below_1based, index);
        assert_real_close(location.fraction, fraction);
    }
    Ok(())
}

#[test]
fn energy_prefactor_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference_energy = Complex::new(0.03, -0.01);
    let reference = [
        (
            Complex::new(0.2, 0.05),
            Complex::new(7.535_556_933_393_025e-1, 1.290_779_920_176_434_2e-1),
        ),
        (
            Complex::new(-0.1, 0.0),
            Complex::new(2.495_199_004_442_948e-2, 6.497_077_620_608_896e-1),
        ),
        (
            Complex::new(1.5, -0.2),
            Complex::new(2.187_643_962_932_14, -1.407_872_105_075_902e-1),
        ),
    ];

    for (energy, expected) in reference {
        let actual = rhorrp_energy_prefactor(RhorrpEnergyPrefactorInput {
            energy_hartree: energy,
            reference_energy_hartree: reference_energy,
        })?;
        assert_complex_close(actual, expected);
    }
    Ok(())
}

#[test]
fn energy_density_finish_matches_feff_reference() -> Result<(), RhorrpError> {
    let energies = Array1::from_vec(vec![
        Complex::new(0.2, 0.05),
        Complex::new(-0.1, 0.0),
        Complex::new(1.5, -0.2),
    ]);
    let green = Array1::from_vec(vec![
        Complex::new(0.002_385_790_539_293_98, -0.001_327_985_363_644_39),
        Complex::new(0.004_561_327_948_938_82, -0.002_352_045_463_805_32),
        Complex::new(0.007_803_700_836_496_53, -0.003_398_486_727_838_54),
    ]);
    let density = rhorrp_finish_energy_density(RhorrpEnergyDensityInput {
        energies_hartree: energies.view(),
        green_function: green.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        radius: 0.85,
        prime_radius: 1.25,
    })?;

    assert_complex_close(
        density[0],
        Complex::new(1.853_402_097_099_352_3e-3, -6.520_074_157_729_285e-4),
    );
    assert_complex_close(
        density[1],
        Complex::new(1.545_370_733_294_796_5e-3, 2.733_968_902_337_800_6e-3),
    );
    assert_complex_close(
        density[2],
        Complex::new(1.561_718_170_082_886_1e-2, -8.031_379_054_745_488e-3),
    );
    Ok(())
}

#[test]
fn pair_energy_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_pair_energy_tables();
    let energies = reference_pair_energies();
    let density = rhorrp_pair_energy_density(RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: [0.22, -0.18, 0.44],
        second_displacement: [-0.31, 0.28, 0.36],
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    })?;

    assert_complex_close_tol(
        density[0],
        Complex::new(4.920_268_421_420_252e-3, 1.431_508_175_602_251_9e-3),
        5.0e-11,
    );
    assert_complex_close_tol(
        density[1],
        Complex::new(-2.189_314_228_048_695e-4, 1.597_291_659_336_723_4e-2),
        5.0e-11,
    );
    assert_complex_close_tol(
        density[2],
        Complex::new(1.170_398_112_832_168_2e-1, -4.062_425_109_588_009e-3),
        5.0e-11,
    );
    Ok(())
}

#[test]
fn pair_density_matches_composed_feff_rhorrp_flow() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let pair_energy = RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: [0.22, -0.18, 0.44],
        second_displacement: [-0.31, 0.28, 0.36],
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    };
    let energy_density = rhorrp_pair_energy_density(pair_energy)?;
    let expected = rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: energies.view(),
        energy_density: energy_density.view(),
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;
    let actual = rhorrp_pair_density(RhorrpPairDensityInput {
        pair_energy,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    assert_real_close(actual, expected);
    Ok(())
}

#[test]
fn pair_density_applies_feff_temperature_floor() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let pair_energy = RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: [0.22, -0.18, 0.44],
        second_displacement: [-0.31, 0.28, 0.36],
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    };
    let energy_density = rhorrp_pair_energy_density(pair_energy)?;
    let expected = rhorrp_integrate_density(RhorrpDensityIntegrationInput {
        energies_hartree: energies.view(),
        energy_density: energy_density.view(),
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.001,
        chemical_potential_override_hartree: None,
    })?;
    let actual = rhorrp_pair_density(RhorrpPairDensityInput {
        pair_energy,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 1.0e-6,
        chemical_potential_override_hartree: None,
    })?;

    assert_real_close(actual, expected);
    assert_real_close(rhorrp_effective_temperature_hartree(0.0035)?, 0.0035);
    assert_real_close(rhorrp_effective_temperature_hartree(-0.4)?, 0.001);
    Ok(())
}

#[test]
fn point_density_selects_nearest_handoff_tables() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 1];
    let point = [1.85, 0.10, -0.05];
    let displacement = [-0.15, 0.10, -0.05];

    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let scattering = diagonal_scattering_matrices(&tables.scattering_matrix);
    let wavefunctions = wavefunction_tables_from_handoff(
        &phase,
        &regular_large,
        &irregular_large,
        &regular_small,
        &irregular_small,
    );

    let expected_pair_energy = RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.second_regular_large.view(),
        first_irregular_large: tables.first_regular_large.view(),
        first_regular_small: tables.second_regular_small.view(),
        first_irregular_small: tables.first_regular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.second_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: true,
        first_displacement: displacement,
        second_displacement: displacement,
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    };
    let expected_energy_density = rhorrp_pair_energy_density(expected_pair_energy)?;
    let expected = rhorrp_pair_density(RhorrpPairDensityInput {
        pair_energy: expected_pair_energy,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    let point_input = RhorrpPointDensityInput {
        point,
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(scattering.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    };
    let actual_energy_density = rhorrp_point_energy_density(point_input.energy_density_input())?;
    let actual_energy_density_from_tables =
        rhorrp_point_energy_density_from_tables(RhorrpPointEnergyDensityFromTablesInput {
            point,
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            fms_atom_count: Some(2),
            energies_hartree: energies.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            wavefunctions: &wavefunctions,
            diagonal_scattering_matrices: Some(scattering.view()),
            radial_x0: 0.7,
            radial_dx: 0.2,
        })?;
    let actual_from_tables = rhorrp_point_density_from_tables(RhorrpPointDensityFromTablesInput {
        point,
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        wavefunctions: &wavefunctions,
        diagonal_scattering_matrices: Some(scattering.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;
    let actual = rhorrp_point_density(point_input)?;

    for (actual, expected) in actual_energy_density
        .iter()
        .zip(expected_energy_density.iter())
    {
        assert_complex_close_tol(*actual, *expected, 5.0e-11);
    }
    for (actual, expected) in actual_energy_density_from_tables
        .iter()
        .zip(actual_energy_density.iter())
    {
        assert_complex_close_tol(*actual, *expected, 5.0e-11);
    }
    assert_real_close(actual, expected);
    assert_real_close(actual_from_tables, actual);
    Ok(())
}

#[test]
fn point_density_rejects_invalid_handoff_shapes() {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 2];
    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let short_scattering = Array4::zeros((energies.len(), 1, 4, 4));

    assert!(matches!(
        rhorrp_point_density(RhorrpPointDensityInput {
            point: [1.85, 0.10, -0.05],
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            fms_atom_count: Some(2),
            energies_hartree: energies.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            regular_large: regular_large.view(),
            irregular_large: irregular_large.view(),
            regular_small: regular_small.view(),
            irregular_small: irregular_small.view(),
            phase: phase.view(),
            diagonal_scattering_matrices: None,
            radial_x0: 0.7,
            radial_dx: 0.2,
            radial_count: 6,
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::InvalidPointDensityPotential {
            atom_index_1based: 2,
            potential: 2,
            max_potential: 1,
        })
    ));

    assert!(matches!(
        rhorrp_point_density(RhorrpPointDensityInput {
            point: [1.85, 0.10, -0.05],
            atom_positions: atom_positions.view(),
            atom_potentials: &[0, 1],
            fms_atom_count: Some(2),
            energies_hartree: energies.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            regular_large: regular_large.view(),
            irregular_large: irregular_large.view(),
            regular_small: regular_small.view(),
            irregular_small: irregular_small.view(),
            phase: phase.view(),
            diagonal_scattering_matrices: Some(short_scattering.view()),
            radial_x0: 0.7,
            radial_dx: 0.2,
            radial_count: 6,
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::PointDensityDiagonalScatteringShapeMismatch {
            expected_atoms: 2,
            actual_atoms: 1,
            ..
        })
    ));
}

#[test]
fn scattering_matrix_selection_matches_feff_saved_handoff() -> Result<(), RhorrpError> {
    let diagonal = Array4::from_shape_fn((2, 3, 2, 2), |(energy, atom, row, column)| {
        Complex::new(
            100.0 * energy as Real + 10.0 * atom as Real + row as Real,
            column as Real,
        )
    });
    let central = Array4::from_shape_fn((2, 3, 2, 2), |(energy, atom, row, column)| {
        Complex::new(
            -100.0 * energy as Real - 10.0 * atom as Real - row as Real,
            -(column as Real),
        )
    });

    let same_site = rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
        first_atom_index: 2,
        second_atom_index: 2,
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
    })?
    .expect("same-site gg_diag block");
    assert_complex_close(same_site[(1, 0, 1)], diagonal[(1, 2, 0, 1)]);

    let central_to_other = rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
        first_atom_index: 0,
        second_atom_index: 1,
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
    })?
    .expect("central gg_slice block");
    assert_complex_close(central_to_other[(1, 0, 1)], central[(1, 1, 0, 1)]);

    assert!(
        rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
            first_atom_index: 2,
            second_atom_index: 1,
            diagonal_scattering_matrices: Some(diagonal.view()),
            central_scattering_matrices: Some(central.view()),
        })?
        .is_none()
    );

    assert!(matches!(
        rhorrp_select_scattering_matrix(RhorrpScatteringMatrixSelectionInput {
            first_atom_index: 0,
            second_atom_index: 3,
            diagonal_scattering_matrices: Some(diagonal.view()),
            central_scattering_matrices: Some(central.view()),
        }),
        Err(RhorrpError::ScatteringMatrixAtomOutOfRange {
            matrix: "central",
            atom_index: 3,
            atom_count: 3,
        })
    ));
    Ok(())
}

#[test]
fn point_pair_density_selects_central_slice_handoff_tables() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 1];
    let first_point = [0.20, -0.10, 0.30];
    let second_point = [1.85, 0.10, -0.05];
    let second_displacement = [-0.15, 0.10, -0.05];

    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let diagonal = diagonal_scattering_matrices(&tables.scattering_matrix);
    let central = central_scattering_matrices(&tables.scattering_matrix);
    let wavefunctions = wavefunction_tables_from_handoff(
        &phase,
        &regular_large,
        &irregular_large,
        &regular_small,
        &irregular_small,
    );

    let expected_pair_energy = RhorrpPairEnergyDensityInput {
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        first_regular_large: tables.first_regular_large.view(),
        first_irregular_large: tables.first_irregular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        first_irregular_small: tables.first_irregular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: Some(tables.scattering_matrix.view()),
        same_atom: false,
        first_displacement: first_point,
        second_displacement,
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
    };
    let expected_energy_density = rhorrp_pair_energy_density(expected_pair_energy)?;
    let expected = rhorrp_pair_density(RhorrpPairDensityInput {
        pair_energy: expected_pair_energy,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    let point_pair_input = RhorrpPointPairDensityInput {
        first_point,
        second_point,
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        restrict_first_point_to_central_voronoi: false,
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    };
    let actual_energy_density =
        rhorrp_point_pair_energy_density(point_pair_input.energy_density_input())?;
    let actual_energy_density_from_tables = rhorrp_point_pair_energy_density_from_tables(
        RhorrpPointPairEnergyDensityFromTablesInput {
            first_point,
            second_point,
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            fms_atom_count: Some(2),
            restrict_first_point_to_central_voronoi: false,
            energies_hartree: energies.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            wavefunctions: &wavefunctions,
            diagonal_scattering_matrices: Some(diagonal.view()),
            central_scattering_matrices: Some(central.view()),
            radial_x0: 0.7,
            radial_dx: 0.2,
        },
    )?;
    let actual_from_tables =
        rhorrp_point_pair_density_from_tables(RhorrpPointPairDensityFromTablesInput {
            first_point,
            second_point,
            atom_positions: atom_positions.view(),
            atom_potentials: &atom_potentials,
            fms_atom_count: Some(2),
            restrict_first_point_to_central_voronoi: false,
            energies_hartree: energies.view(),
            reference_energy_hartree: Complex::new(0.03, -0.01),
            wavefunctions: &wavefunctions,
            diagonal_scattering_matrices: Some(diagonal.view()),
            central_scattering_matrices: Some(central.view()),
            radial_x0: 0.7,
            radial_dx: 0.2,
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        })?;
    let actual = rhorrp_point_pair_density(point_pair_input)?;

    for (actual, expected) in actual_energy_density
        .iter()
        .zip(expected_energy_density.iter())
    {
        assert_complex_close_tol(*actual, *expected, 5.0e-11);
    }
    for (actual, expected) in actual_energy_density_from_tables
        .iter()
        .zip(actual_energy_density.iter())
    {
        assert_complex_close_tol(*actual, *expected, 5.0e-11);
    }
    assert_real_close(actual, expected);
    assert_real_close(actual_from_tables, actual);
    Ok(())
}

#[test]
fn point_pair_density_matches_point_density_for_same_point() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 1];
    let point = [1.85, 0.10, -0.05];

    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let diagonal = diagonal_scattering_matrices(&tables.scattering_matrix);
    let central = central_scattering_matrices(&tables.scattering_matrix);

    let point_density = rhorrp_point_density(RhorrpPointDensityInput {
        point,
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(diagonal.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    let point_pair_density = rhorrp_point_pair_density(RhorrpPointPairDensityInput {
        first_point: point,
        second_point: point,
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        restrict_first_point_to_central_voronoi: false,
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    assert_real_close(point_pair_density, point_density);
    Ok(())
}

#[test]
fn point_pair_density_zeroes_unsupported_saved_matrix_pair() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 1];
    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let diagonal = diagonal_scattering_matrices(&tables.scattering_matrix);
    let central = central_scattering_matrices(&tables.scattering_matrix);

    let input = RhorrpPointPairDensityInput {
        first_point: [1.85, 0.10, -0.05],
        second_point: [0.20, -0.10, 0.30],
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        restrict_first_point_to_central_voronoi: false,
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    };
    let energy_density = rhorrp_point_pair_energy_density(input.energy_density_input())?;
    let density = rhorrp_point_pair_density(input)?;

    assert!(energy_density.iter().all(|value| value.norm() == 0.0));
    assert_real_close(density, 0.0);
    Ok(())
}

#[test]
fn point_pair_density_restricts_first_point_to_central_voronoi() -> Result<(), RhorrpError> {
    let (energies, _) = reference_density_integration_inputs();
    let tables = reference_pair_energy_tables_with_energy_count(energies.len());
    let atom_positions = arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let atom_potentials = [0, 1];
    let regular_large =
        potential_wavefunctions(&tables.first_regular_large, &tables.second_regular_large);
    let irregular_large =
        potential_wavefunctions(&tables.first_irregular_large, &tables.first_regular_large);
    let regular_small =
        potential_wavefunctions(&tables.first_regular_small, &tables.second_regular_small);
    let irregular_small =
        potential_wavefunctions(&tables.first_irregular_small, &tables.first_regular_small);
    let phase = potential_phase(&tables.first_phase, &tables.second_phase);
    let diagonal = diagonal_scattering_matrices(&tables.scattering_matrix);
    let central = central_scattering_matrices(&tables.scattering_matrix);

    let density = rhorrp_point_pair_density(RhorrpPointPairDensityInput {
        first_point: [1.85, 0.10, -0.05],
        second_point: [0.20, -0.10, 0.30],
        atom_positions: atom_positions.view(),
        atom_potentials: &atom_potentials,
        fms_atom_count: Some(2),
        restrict_first_point_to_central_voronoi: true,
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.03, -0.01),
        regular_large: regular_large.view(),
        irregular_large: irregular_large.view(),
        regular_small: regular_small.view(),
        irregular_small: irregular_small.view(),
        phase: phase.view(),
        diagonal_scattering_matrices: Some(diagonal.view()),
        central_scattering_matrices: Some(central.view()),
        radial_x0: 0.7,
        radial_dx: 0.2,
        radial_count: 6,
        real_axis_count: 6,
        chemical_potential_hartree: 0.045,
        temperature_hartree: 0.0035,
        chemical_potential_override_hartree: None,
    })?;

    assert_real_close(density, 0.0);
    Ok(())
}

#[test]
fn same_site_green_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_same_site_wavefunctions();
    let same = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
        regular_large: tables.regular_large.view(),
        irregular_large: tables.irregular_large.view(),
        regular_small: tables.regular_small.view(),
        irregular_small: tables.irregular_small.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.25,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 3,
            fraction: 0.60,
        },
        cosine_between: 0.35,
    })?;
    assert_complex_close(
        same[0],
        Complex::new(2.385_790_539_293_98e-3, -1.327_985_363_644_393_7e-3),
    );
    assert_complex_close(
        same[1],
        Complex::new(4.561_327_948_938_822e-3, -2.352_045_463_805_323e-3),
    );
    assert_complex_close(
        same[2],
        Complex::new(7.803_700_836_496_527e-3, -3.398_486_727_838_544e-3),
    );

    let swapped = rhorrp_same_site_green(RhorrpSameSiteGreenInput {
        regular_large: tables.regular_large.view(),
        irregular_large: tables.irregular_large.view(),
        regular_small: tables.regular_small.view(),
        irregular_small: tables.irregular_small.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 3,
            fraction: 0.70,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.20,
        },
        cosine_between: -0.40,
    })?;
    assert_complex_close(
        swapped[0],
        Complex::new(5.306_066_647_740_74e-5, -2.530_189_581_044_869_3e-3),
    );
    assert_complex_close(
        swapped[1],
        Complex::new(-5.026_528_497_243_523e-3, -4.721_792_936_156_043e-3),
    );
    assert_complex_close(
        swapped[2],
        Complex::new(-1.351_999_119_028_561e-2, -6.841_776_566_875_865e-3),
    );
    Ok(())
}

#[test]
fn scattering_green_matches_feff_reference() -> Result<(), RhorrpError> {
    let tables = reference_scattering_green_tables();
    let scattering = rhorrp_scattering_green(RhorrpScatteringGreenInput {
        first_regular_large: tables.first_regular_large.view(),
        first_regular_small: tables.first_regular_small.view(),
        second_regular_large: tables.second_regular_large.view(),
        second_regular_small: tables.second_regular_small.view(),
        first_phase: tables.first_phase.view(),
        second_phase: tables.second_phase.view(),
        scattering_matrix: tables.scattering_matrix.view(),
        first_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 1,
            fraction: 0.25,
        },
        second_location: RhorrpRadialInterpolationLocation {
            index_below_1based: 2,
            fraction: 0.40,
        },
        first_displacement: [0.4, -0.2, 0.6],
        second_displacement: [-0.3, 0.5, 0.7],
    })?;

    assert_complex_close_tol(
        scattering[0],
        Complex::new(-3.113_692_815_836_969e-5, -1.736_972_186_353_566e-5),
        5.0e-12,
    );
    assert_complex_close_tol(
        scattering[1],
        Complex::new(-8.550_104_470_011_752e-5, -4.571_303_454_320_471e-5),
        5.0e-12,
    );
    assert_complex_close_tol(
        scattering[2],
        Complex::new(-1.728_024_650_639_756_4e-4, -5.819_987_091_037_537_5e-5),
        5.0e-12,
    );
    Ok(())
}
