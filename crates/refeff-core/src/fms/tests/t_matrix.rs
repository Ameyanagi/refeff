use super::*;

#[test]
fn fms_t_matrix_matches_feff_reference_branches() -> Result<(), Box<dyn Error>> {
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };

    let non_spin = fms_t_matrix_element(FmsTMatrixInput {
        first,
        second: first,
        spin_channels: 1,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;
    let spin_orbit_diagonal = fms_t_matrix_element(FmsTMatrixInput {
        first,
        second: first,
        spin_channels: 2,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;
    let spin_mixing = fms_t_matrix_element(FmsTMatrixInput {
        first,
        second: StateKet {
            magnetic: 0,
            spin: 2,
            ..first
        },
        spin_channels: 2,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_complex32_close(non_spin, Complex32::new(0.176_180_14, 0.083_294_78));
    assert_complex32_close(
        spin_orbit_diagonal,
        Complex32::new(0.068_288_13, 0.065_378_49),
    );
    assert_complex32_close(spin_mixing, Complex32::new(-0.087_964_38, -0.001_144_098_1));
    Ok(())
}

#[test]
fn fms_t_matrix_returns_zero_for_disallowed_pairs() -> Result<(), Box<dyn Error>> {
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };

    let different_atom = fms_t_matrix_element(FmsTMatrixInput {
        second: StateKet { atom: 2, ..first },
        first,
        spin_channels: 2,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;
    let disallowed_spin_mix = fms_t_matrix_element(FmsTMatrixInput {
        second: StateKet {
            magnetic: -1,
            spin: 2,
            ..first
        },
        first,
        spin_channels: 2,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_complex32_close(different_atom, Complex32::new(0.0, 0.0));
    assert_complex32_close(disallowed_spin_mix, Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_t_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
    let mut phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };

    let invalid_spin_count = fms_t_matrix_element(FmsTMatrixInput {
        first,
        second: first,
        spin_channels: 3,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    });
    assert!(matches!(
        invalid_spin_count,
        Err(FmsError::InvalidSpinChannelCount { value: 3 })
    ));

    let invalid_state_spin = fms_t_matrix_element(FmsTMatrixInput {
        first: StateKet { spin: 2, ..first },
        second: StateKet { spin: 2, ..first },
        spin_channels: 1,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    });
    assert!(matches!(
        invalid_state_spin,
        Err(FmsError::InvalidStateSpin {
            spin: 2,
            spin_channels: 1,
        })
    ));

    phases[(0, 4, 1)] = Complex32::new(f32::NAN, 0.0);
    let nonfinite_phase = fms_t_matrix_element(FmsTMatrixInput {
        first,
        second: first,
        spin_channels: 1,
        spin_selector: 0,
        potential: 1,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    });
    assert!(matches!(
        nonfinite_phase,
        Err(FmsError::NonFinitePhaseShift {
            spin: 1,
            angular_momentum: 2,
            potential: 1,
        })
    ));
    Ok(())
}

#[test]
fn fms_t_matrix_table_matches_feff_compact_layout() -> Result<(), Box<dyn Error>> {
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };
    let states = [
        first,
        StateKet {
            magnetic: 0,
            spin: 2,
            ..first
        },
    ];

    let table = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 2,
        spin_selector: 0,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_eq!(table.shape(), &[2, 2]);
    assert_eq!(table.strides(), &[1, 2]);
    assert_complex32_close(table[(0, 0)], Complex32::new(0.068_288_13, 0.065_378_49));
    assert_complex32_close(
        table[(1, 0)],
        Complex32::new(-0.087_964_38, -0.001_144_098_1),
    );
    Ok(())
}

#[test]
fn fms_t_matrix_table_handles_non_spin_branch() -> Result<(), Box<dyn Error>> {
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    let states = [StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    }];

    let table = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 1,
        spin_selector: 0,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_eq!(table.shape(), &[1, 1]);
    assert_complex32_close(table[(0, 0)], Complex32::new(0.176_180_14, 0.083_294_78));
    Ok(())
}

#[test]
fn fms_t_matrix_table_rejects_invalid_potential() -> Result<(), Box<dyn Error>> {
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 2,
    }];
    let states = [StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    }];

    let result = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 1,
        spin_selector: 0,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    });

    assert!(matches!(
        result,
        Err(FmsError::PotentialOutOfRange {
            potential: 2,
            max_potential: 1,
        })
    ));
    Ok(())
}

#[test]
fn fms_hubbard_t_matrix_uses_feff_magnetic_phase_slot() -> Result<(), Box<dyn Error>> {
    let mut phases = Array4::zeros((1, 5, 9, 1).f());
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let state = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };
    let selected_phase = Complex32::new(0.2, 0.05);
    let other_m_phase = Complex32::new(0.7, 0.1);
    phases[(0, 4, 7, 0)] = selected_phase;
    phases[(0, 4, 6, 0)] = other_m_phase;

    let value = fms_hubbard_t_matrix_element(FmsHubbardTMatrixInput {
        first: state,
        second: state,
        spin_channels: 1,
        spin_selector: 0,
        potential: 0,
        magnetic_phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_complex32_close(value, expected_t_matrix_phase(selected_phase));
    assert_ne!(value, expected_t_matrix_phase(other_m_phase));
    Ok(())
}

#[test]
fn fms_hubbard_t_matrix_table_builds_full_same_site_matrix() -> Result<(), Box<dyn Error>> {
    let mut phases = Array4::zeros((1, 3, 4, 1).f());
    let spin_orbit = spin_orbit_coupling_tables(1)?;
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let states = construct_state_kets(1, &[0], &[1], 1)?.states;
    phases[(0, 2, 1, 0)] = Complex32::new(0.11, 0.01);
    phases[(0, 2, 2, 0)] = Complex32::new(0.22, 0.02);
    phases[(0, 2, 3, 0)] = Complex32::new(0.33, 0.03);

    let table = fms_hubbard_t_matrix_table(FmsHubbardTMatrixTableInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 1,
        spin_selector: 0,
        magnetic_phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;

    assert_eq!(table.shape(), &[4, 4]);
    assert_complex32_close(table[(1, 1)], expected_t_matrix_phase(phases[(0, 2, 1, 0)]));
    assert_complex32_close(table[(2, 2)], expected_t_matrix_phase(phases[(0, 2, 2, 0)]));
    assert_complex32_close(table[(3, 3)], expected_t_matrix_phase(phases[(0, 2, 3, 0)]));
    assert_complex32_close(table[(1, 2)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_transform_t_matrix_matches_feff_block_order() -> Result<(), Box<dyn Error>> {
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let states = construct_state_kets(1, &[0], &[1], 1)?.states;
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut t_matrix = Array2::zeros((4, 4).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            t_matrix[(1 + row, 1 + column)] =
                Complex32::new(block_values[row][column], -block_values[row][column]);
        }
    }

    let transformed = fms_hubbard_transform_t_matrix(FmsHubbardTMatrixTransformInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 1,
        use_transform: use_transform.view(),
        transform: transform.view(),
        inverse: inverse.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_complex32_close(transformed[(1, 1)], t_matrix[(2, 2)]);
    assert_complex32_close(transformed[(1, 2)], t_matrix[(2, 1)]);
    assert_complex32_close(transformed[(2, 1)], t_matrix[(1, 2)]);
    assert_complex32_close(transformed[(3, 3)], t_matrix[(3, 3)]);
    assert_complex32_close(transformed[(0, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_transform_t_matrix_allows_two_spin_feff_block_start() -> Result<(), Box<dyn Error>> {
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let states = construct_state_kets(2, &[0], &[1], 1)?.states;
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut t_matrix = Array2::zeros((8, 8).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            t_matrix[(3 + row, 3 + column)] =
                Complex32::new(block_values[row][column], block_values[row][column] * 0.25);
        }
    }

    let transformed = fms_hubbard_transform_t_matrix(FmsHubbardTMatrixTransformInput {
        states: &states,
        atoms: &atoms,
        spin_channels: 2,
        use_transform: use_transform.view(),
        transform: transform.view(),
        inverse: inverse.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_complex32_close(transformed[(3, 3)], t_matrix[(4, 4)]);
    assert_complex32_close(transformed[(3, 4)], t_matrix[(4, 3)]);
    assert_complex32_close(transformed[(4, 3)], t_matrix[(3, 4)]);
    assert_complex32_close(transformed[(5, 5)], t_matrix[(5, 5)]);
    assert_complex32_close(transformed[(2, 2)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_back_transform_scattering_matches_feff_block_order() -> Result<(), Box<dyn Error>> {
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut scattering = Array3::zeros((4, 4, 1).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            scattering[(1 + row, 1 + column, 0)] =
                Complex32::new(block_values[row][column], block_values[row][column] * 0.5);
        }
    }

    let transformed = fms_hubbard_back_transform_scattering(FmsHubbardScatteringTransformInput {
        spin_channels: 1,
        potential_lmax: &[1],
        use_transform: use_transform.view(),
        transform: transform.view(),
        inverse: inverse.view(),
        scattering: scattering.view(),
    })?;

    assert_complex32_close(transformed[(1, 1, 0)], scattering[(2, 2, 0)]);
    assert_complex32_close(transformed[(1, 2, 0)], scattering[(2, 1, 0)]);
    assert_complex32_close(transformed[(2, 1, 0)], scattering[(1, 2, 0)]);
    assert_complex32_close(transformed[(3, 3, 0)], scattering[(3, 3, 0)]);
    assert_complex32_close(transformed[(0, 0, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_back_transform_scattering_allows_two_spin_feff_block_start()
-> Result<(), Box<dyn Error>> {
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut scattering = Array3::zeros((8, 8, 1).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            scattering[(3 + row, 3 + column, 0)] =
                Complex32::new(block_values[row][column], -block_values[row][column] * 0.5);
        }
    }

    let transformed = fms_hubbard_back_transform_scattering(FmsHubbardScatteringTransformInput {
        spin_channels: 2,
        potential_lmax: &[1],
        use_transform: use_transform.view(),
        transform: transform.view(),
        inverse: inverse.view(),
        scattering: scattering.view(),
    })?;

    assert_complex32_close(transformed[(3, 3, 0)], scattering[(4, 4, 0)]);
    assert_complex32_close(transformed[(3, 4, 0)], scattering[(4, 3, 0)]);
    assert_complex32_close(transformed[(4, 3, 0)], scattering[(3, 4, 0)]);
    assert_complex32_close(transformed[(5, 5, 0)], scattering[(5, 5, 0)]);
    assert_complex32_close(transformed[(2, 2, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_back_transform_full_scattering_matches_feff_block_order()
-> Result<(), Box<dyn Error>> {
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let states = construct_state_kets(1, &[0], &[1], 1)?.states;
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut full_scattering = Array2::zeros((4, 4).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            full_scattering[(1 + row, 1 + column)] =
                Complex32::new(block_values[row][column], -block_values[row][column] * 0.25);
        }
    }

    let transformed =
        fms_hubbard_back_transform_full_scattering(FmsHubbardFullScatteringTransformInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 1,
            potential_lmax: &[1],
            use_transform: use_transform.view(),
            transform: transform.view(),
            inverse: inverse.view(),
            full_scattering: full_scattering.view(),
        })?;

    assert_complex32_close(transformed[(1, 1)], full_scattering[(2, 2)]);
    assert_complex32_close(transformed[(1, 2)], full_scattering[(2, 1)]);
    assert_complex32_close(transformed[(2, 1)], full_scattering[(1, 2)]);
    assert_complex32_close(transformed[(3, 3)], full_scattering[(3, 3)]);
    assert_complex32_close(transformed[(0, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_hubbard_back_transform_full_scattering_allows_two_spin_feff_block_start()
-> Result<(), Box<dyn Error>> {
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let states = construct_state_kets(2, &[0], &[1], 1)?.states;
    let mut use_transform = Array2::from_elem((2, 1), false);
    use_transform[(1, 0)] = true;
    let (transform, inverse) = swap_transform_tables();
    let mut full_scattering = Array2::zeros((8, 8).f());
    let block_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for row in 0..3 {
        for column in 0..3 {
            full_scattering[(3 + row, 3 + column)] =
                Complex32::new(block_values[row][column], -block_values[row][column] * 0.75);
        }
    }

    let transformed =
        fms_hubbard_back_transform_full_scattering(FmsHubbardFullScatteringTransformInput {
            states: &states,
            atoms: &atoms,
            spin_channels: 2,
            potential_lmax: &[1],
            use_transform: use_transform.view(),
            transform: transform.view(),
            inverse: inverse.view(),
            full_scattering: full_scattering.view(),
        })?;

    assert_complex32_close(transformed[(3, 3)], full_scattering[(4, 4)]);
    assert_complex32_close(transformed[(3, 4)], full_scattering[(4, 3)]);
    assert_complex32_close(transformed[(4, 3)], full_scattering[(3, 4)]);
    assert_complex32_close(transformed[(5, 5)], full_scattering[(5, 5)]);
    assert_complex32_close(transformed[(2, 2)], Complex32::new(0.0, 0.0));
    Ok(())
}

fn expected_t_matrix_phase(phase: Complex32) -> Complex32 {
    let two_i = Complex32::new(0.0, 2.0);
    ((two_i * phase).exp() - Complex32::new(1.0, 0.0)) / two_i
}

fn swap_transform_tables() -> (Array4<Complex32>, Array4<Complex32>) {
    let mut transform = Array4::zeros((3, 3, 2, 1).f());
    transform[(0, 0, 0, 0)] = Complex32::new(1.0, 0.0);
    transform[(0, 1, 1, 0)] = Complex32::new(1.0, 0.0);
    transform[(1, 0, 1, 0)] = Complex32::new(1.0, 0.0);
    transform[(2, 2, 1, 0)] = Complex32::new(1.0, 0.0);
    (transform.clone(), transform)
}
