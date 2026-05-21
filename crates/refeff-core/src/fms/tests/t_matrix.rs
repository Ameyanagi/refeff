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
