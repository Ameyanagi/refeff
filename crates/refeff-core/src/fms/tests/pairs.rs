use super::*;
use crate::fms::fms_free_propagator_prefactor;

#[test]
fn fms_pair_tables_match_feff_reference() -> Result<(), FmsError> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.5],
            potential: 2,
        },
    ];

    let tables = fms_pair_tables(2, Complex32::new(1.2, 0.3), &atoms)?;

    assert_eq!(tables.rho.shape(), &[3, 3]);
    assert_eq!(tables.rho.strides(), &[1, 3]);
    assert_eq!(tables.polynomials.shape(), &[3, 3, 3, 3]);
    assert_eq!(tables.polynomials.strides(), &[1, 3, 9, 27]);
    assert_complex32_close(
        tables.rho[(0, 1)],
        Complex32::new(3.600_000_1, 0.900_000_04),
    );
    assert_complex32_close(tables.rho[(0, 2)], Complex32::new(1.341_640_8, 0.335_410_2));
    assert_complex32_close(tables.rho[(1, 2)], Complex32::new(3.841_874_8, 0.960_468_7));
    assert_complex32_close(
        pair_table_sum(tables.polynomials.view()),
        Complex32::new(8.870_853, 26.772_633),
    );
    assert_eq!(pair_table_nonzero_count(tables.polynomials.view()), 36);
    assert_complex32_close(tables.polynomials[(0, 0, 1, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(
        tables.polynomials[(1, 1, 1, 0)],
        Complex32::new(0.065_359_47, 0.261_437_9),
    );
    assert_complex32_close(
        tables.polynomials[(2, 2, 2, 0)],
        Complex32::new(-1.384_083, 0.738_177_6),
    );
    assert_complex32_close(
        tables.polynomials[(1, 2, 2, 1)],
        Complex32::new(-0.153_847_35, 0.914_978_6),
    );
    assert_complex32_close(tables.polynomials[(1, 1, 0, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_pair_tables_reject_invalid_inputs() {
    assert_eq!(
        fms_pair_tables(
            1,
            Complex32::new(f32::NAN, 0.0),
            &[FmsAtom {
                position: [0.0, 0.0, 0.0],
                potential: 0,
            }],
        ),
        Err(FmsError::NonFiniteWaveNumber)
    );
    assert_eq!(
        fms_pair_tables(
            1,
            Complex32::new(1.0, 0.0),
            &[FmsAtom {
                position: [0.0, f32::INFINITY, 0.0],
                potential: 0,
            }],
        ),
        Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 1 })
    );
}

#[test]
fn fms_spin_pair_tables_match_feff_spin_axis_layout() -> Result<(), FmsError> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.5],
            potential: 2,
        },
    ];
    let wave_numbers = [Complex32::new(1.2, 0.3), Complex32::new(0.8, 0.15)];
    let spin_tables = fms_spin_pair_tables(2, &wave_numbers, &atoms)?;
    let first_spin = fms_pair_tables(2, wave_numbers[0], &atoms)?;
    let second_spin = fms_pair_tables(2, wave_numbers[1], &atoms)?;

    assert_eq!(spin_tables.rho.shape(), &[3, 3, 2]);
    assert_eq!(spin_tables.rho.strides(), &[1, 3, 9]);
    assert_eq!(spin_tables.polynomials.shape(), &[3, 3, 3, 3, 2]);
    assert_eq!(spin_tables.polynomials.strides(), &[1, 3, 9, 27, 81]);
    assert_complex32_close(spin_tables.rho[(1, 0, 0)], first_spin.rho[(1, 0)]);
    assert_complex32_close(spin_tables.rho[(1, 0, 1)], second_spin.rho[(1, 0)]);
    assert_complex32_close(
        pair_table_sum(spin_tables.polynomials.index_axis(Axis(4), 0)),
        pair_table_sum(first_spin.polynomials.view()),
    );
    assert_complex32_close(
        pair_table_sum(spin_tables.polynomials.index_axis(Axis(4), 1)),
        pair_table_sum(second_spin.polynomials.view()),
    );
    Ok(())
}

#[test]
fn fms_free_propagator_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
    let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 2,
        magnetic: -1,
        spin: 1,
    };

    let value = fms_free_propagator_element(FmsFreePropagatorInput {
        first,
        second,
        rho: tables.rho[(0, 1)],
        wave_number,
        mean_square_displacement: 0.05,
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        backward_rotation: backward.view(),
        forward_rotation: forward.view(),
    })?;

    assert_complex32_close(value, Complex32::new(-0.134_080_86, 0.113_885_55));
    Ok(())
}

#[test]
fn fms_free_propagator_damping_uses_angstrom_units_and_handles_zero_wave_number() {
    let rho = Complex32::new(1.5, 0.0);
    let undamped = fms_free_propagator_prefactor(rho, Complex32::new(2.0, 0.0), 0.0);
    let damped = fms_free_propagator_prefactor(rho, Complex32::new(2.0, 0.0), 0.05);
    let damping_ratio = damped / undamped;

    // sigma^2 = 0.05 Angstrom^2 and k = 2 Angstrom^-1 give
    // exp(-sigma^2 k^2) = exp(-0.2).
    assert_close_f32(damping_ratio.re, (-0.2_f32).exp());
    assert_close_f32(damping_ratio.im, 0.0);

    let zero_wave_undamped = fms_free_propagator_prefactor(rho, Complex32::new(0.0, 0.0), 0.0);
    let zero_wave_damped = fms_free_propagator_prefactor(rho, Complex32::new(0.0, 0.0), 10.0);
    assert_complex32_close(zero_wave_damped, zero_wave_undamped);
}

#[test]
fn fms_free_propagator_returns_zero_for_excluded_state_pairs() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
    let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };

    let same_atom = fms_free_propagator_element(FmsFreePropagatorInput {
        second: StateKet { atom: 1, ..second },
        first,
        rho: Complex32::new(0.0, 0.0),
        wave_number,
        mean_square_displacement: 0.05,
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        backward_rotation: backward.view(),
        forward_rotation: forward.view(),
    })?;
    let spin_mismatch = fms_free_propagator_element(FmsFreePropagatorInput {
        second: StateKet { spin: 2, ..second },
        first,
        rho: tables.rho[(0, 1)],
        wave_number,
        mean_square_displacement: 0.05,
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        backward_rotation: backward.view(),
        forward_rotation: forward.view(),
    })?;

    assert_complex32_close(same_atom, Complex32::new(0.0, 0.0));
    assert_complex32_close(spin_mismatch, Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_free_propagator_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
    let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };
    let input = |rho, wave_number, mean_square_displacement| FmsFreePropagatorInput {
        first,
        second,
        rho,
        wave_number,
        mean_square_displacement,
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        backward_rotation: backward.view(),
        forward_rotation: forward.view(),
    };

    assert_eq!(
        fms_free_propagator_element(input(tables.rho[(0, 1)], wave_number, f32::INFINITY,)),
        Err(FmsError::NonFiniteMeanSquareDisplacement)
    );
    assert_eq!(
        fms_free_propagator_element(input(Complex32::new(0.0, 0.0), wave_number, 0.05)),
        Err(FmsError::ZeroRho)
    );
    assert_eq!(
        fms_free_propagator_element(input(
            tables.rho[(0, 1)],
            Complex32::new(f32::NAN, 0.0),
            0.05,
        )),
        Err(FmsError::NonFiniteWaveNumber)
    );
    Ok(())
}

#[test]
fn fms_free_propagator_matrix_matches_feff_reference_element() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
    let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let mut rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
    copy_rotation_pair(
        &mut rotations,
        1,
        0,
        FmsRotationDirection::Backward,
        &backward,
    );
    copy_rotation_pair(
        &mut rotations,
        1,
        0,
        FmsRotationDirection::Forward,
        &forward,
    );
    let mut sigsqr = Array2::zeros((2, 2).f());
    sigsqr[(1, 0)] = 0.05;
    sigsqr[(0, 1)] = 0.05;
    let states = [
        StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        },
        StateKet {
            atom: 2,
            angular_momentum: 2,
            magnetic: -1,
            spin: 1,
        },
    ];

    let matrix = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: tables.rho.view(),
        wave_number,
        mean_square_displacements: sigsqr.view(),
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    })?;

    assert_eq!(matrix.shape(), &[2, 2]);
    assert_eq!(matrix.strides(), &[1, 2]);
    assert_complex32_close(matrix[(0, 0)], Complex32::new(0.0, 0.0));
    assert_complex32_close(matrix[(0, 1)], Complex32::new(-0.134_080_86, 0.113_885_55));
    assert_complex32_close(matrix[(1, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn yprep_pair_direction_removes_legacy_odd_even_channel_parity() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 0,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let sigsqr = Array2::zeros((2, 2).f());
    let states = construct_state_kets(1, &[0, 0], &[2], 2)?.states;
    let geometry = fms_yprep_geometry(2, 2, &atoms)?;
    let positions = [atoms[0].position, atoms[1].position];

    let corrected = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: tables.rho.view(),
        wave_number,
        mean_square_displacements: sigsqr.view(),
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
    })?;

    let mut reversed_pair_rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
    for atom2 in 0..atoms.len() {
        for atom1 in 0..atoms.len() {
            if atom2 == atom1 {
                continue;
            }
            let (beta, phi) = pair_polar_angles(&positions, atom2, atom1)?;
            let forward = fms_rotation_matrix(2, 2, beta, phi, FmsRotationDirection::Forward)?;
            let backward = fms_rotation_matrix(2, 2, -beta, phi, FmsRotationDirection::Backward)?;
            copy_rotation_pair(
                &mut reversed_pair_rotations,
                atom2,
                atom1,
                FmsRotationDirection::Forward,
                &forward,
            );
            copy_rotation_pair(
                &mut reversed_pair_rotations,
                atom2,
                atom1,
                FmsRotationDirection::Backward,
                &backward,
            );
        }
    }
    let reversed = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: tables.rho.view(),
        wave_number,
        mean_square_displacements: sigsqr.view(),
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: reversed_pair_rotations.view(),
    })?;

    let mut checked_same_parity = false;
    let mut checked_opposite_parity = false;
    for (row, first) in states.iter().enumerate() {
        for (column, second) in states.iter().enumerate() {
            let legacy = reversed[(row, column)];
            if legacy.norm() < 1.0e-5 {
                continue;
            }
            let parity = if (first.angular_momentum + second.angular_momentum).is_multiple_of(2) {
                checked_same_parity = true;
                1.0
            } else {
                checked_opposite_parity = true;
                -1.0
            };
            let expected = legacy * parity;
            assert!(
                (corrected[(row, column)] - expected).norm() < 2.0e-5,
                "row={row} column={column} corrected={:?} expected={expected:?}",
                corrected[(row, column)]
            );
        }
    }
    assert!(checked_same_parity);
    assert!(checked_opposite_parity);
    Ok(())
}

#[test]
fn fms_spin_free_propagator_matrix_uses_spin_specific_tables() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_numbers = [Complex32::new(1.2, 0.3), Complex32::new(0.8, 0.15)];
    let spin_tables = fms_spin_pair_tables(2, &wave_numbers, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let backward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)?;
    let forward = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let mut rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
    copy_rotation_pair(
        &mut rotations,
        1,
        0,
        FmsRotationDirection::Backward,
        &backward,
    );
    copy_rotation_pair(
        &mut rotations,
        1,
        0,
        FmsRotationDirection::Forward,
        &forward,
    );
    let mut sigsqr = Array2::zeros((2, 2).f());
    sigsqr[(1, 0)] = 0.05;
    let spin1_states = [
        StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        },
        StateKet {
            atom: 2,
            angular_momentum: 2,
            magnetic: -1,
            spin: 1,
        },
    ];
    let spin2_states = [
        StateKet {
            spin: 2,
            ..spin1_states[0]
        },
        StateKet {
            spin: 2,
            ..spin1_states[1]
        },
    ];
    let states = [
        spin1_states[0],
        spin1_states[1],
        spin2_states[0],
        spin2_states[1],
    ];

    let matrix = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: spin_tables.rho.view(),
        wave_numbers: &wave_numbers,
        mean_square_displacements: sigsqr.view(),
        xclm: spin_tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    })?;
    let spin1_tables = fms_pair_tables(2, wave_numbers[0], &atoms)?;
    let spin2_tables = fms_pair_tables(2, wave_numbers[1], &atoms)?;
    let spin1_reference = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &spin1_states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: spin1_tables.rho.view(),
        wave_number: wave_numbers[0],
        mean_square_displacements: sigsqr.view(),
        xclm: spin1_tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    })?;
    let spin2_reference = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &spin2_states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: spin2_tables.rho.view(),
        wave_number: wave_numbers[1],
        mean_square_displacements: sigsqr.view(),
        xclm: spin2_tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    })?;

    assert_eq!(matrix.shape(), &[4, 4]);
    assert_eq!(matrix.strides(), &[1, 4]);
    assert_complex32_close(matrix[(0, 1)], spin1_reference[(0, 1)]);
    assert_complex32_close(matrix[(2, 3)], spin2_reference[(0, 1)]);
    assert_complex32_close(matrix[(0, 3)], Complex32::new(0.0, 0.0));
    assert_complex32_close(matrix[(2, 1)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_free_propagator_matrix_applies_direct_cutoff() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
    let sigsqr = Array2::zeros((2, 2).f());
    let states = [
        StateKet {
            atom: 1,
            angular_momentum: 2,
            magnetic: 1,
            spin: 1,
        },
        StateKet {
            atom: 2,
            angular_momentum: 2,
            magnetic: -1,
            spin: 1,
        },
    ];

    let matrix = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: 2.99,
        rho: tables.rho.view(),
        wave_number,
        mean_square_displacements: sigsqr.view(),
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    })?;

    assert_complex32_close(matrix[(0, 1)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_free_propagator_matrix_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    let wave_number = Complex32::new(1.2, 0.3);
    let tables = fms_pair_tables(2, wave_number, &atoms)?;
    let xnlm = legendre_normalization_table(2)?;
    let rotations = Array6::zeros((5, 5, 3, 2, 2, 2).f());
    let sigsqr = Array2::zeros((2, 2).f());
    let states = [
        StateKet {
            atom: 1,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        },
        StateKet {
            atom: 2,
            angular_momentum: 1,
            magnetic: 0,
            spin: 1,
        },
    ];

    let result = fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
        states: &states,
        atoms: &atoms,
        direct_cutoff: f32::NAN,
        rho: tables.rho.view(),
        wave_number,
        mean_square_displacements: sigsqr.view(),
        xclm: tables.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: rotations.view(),
    });

    assert!(matches!(result, Err(FmsError::InvalidDirectCutoff)));
    Ok(())
}
