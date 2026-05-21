use super::*;

#[test]
fn fms_driver_setup_matches_feff_fmspack_prelude() -> Result<(), FmsError> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 2,
        },
    ];

    let setup = fms_driver_setup(FmsDriverSetupInput {
        lfms: 0,
        spin_channels: 1,
        atoms: &atoms,
        max_potential: 2,
        global_lmax: 2,
        raw_potential_lmax: &[-1, 5, 1],
        state_capacity: None,
    })?;

    assert_eq!(setup.potential_lmax, vec![2, 2, 1]);
    assert_eq!(setup.potential_start, 1);
    assert_eq!(setup.potential_end, 1);
    assert_eq!(
        setup.state_kets.representative_offsets,
        vec![Some(9), Some(0), Some(18)]
    );
    assert_eq!(setup.state_kets.states.len(), 22);
    assert_eq!(
        setup.state_kets.states[0],
        StateKet {
            atom: 1,
            angular_momentum: 0,
            magnetic: 0,
            spin: 1,
        }
    );
    assert_eq!(
        setup.state_kets.states[9],
        StateKet {
            atom: 2,
            angular_momentum: 0,
            magnetic: 0,
            spin: 1,
        }
    );
    Ok(())
}

#[test]
fn fms_driver_setup_requires_representatives_for_active_range() {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 2,
        },
    ];

    assert_eq!(
        fms_driver_setup(FmsDriverSetupInput {
            lfms: 1,
            spin_channels: 1,
            atoms: &atoms,
            max_potential: 2,
            global_lmax: 1,
            raw_potential_lmax: &[1, 1, 1],
            state_capacity: None,
        }),
        Err(FmsError::MissingRepresentativePotential { potential: 1 })
    );
}

#[test]
fn fms_driver_setup_rejects_invalid_inputs() {
    let atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 0,
    }];
    let base = FmsDriverSetupInput {
        lfms: 0,
        spin_channels: 1,
        atoms: &atoms,
        max_potential: 0,
        global_lmax: 1,
        raw_potential_lmax: &[1],
        state_capacity: None,
    };

    assert_eq!(
        fms_driver_setup(FmsDriverSetupInput {
            atoms: &[],
            ..base.clone()
        }),
        Err(FmsError::EmptyCluster)
    );
    assert_eq!(
        fms_driver_setup(FmsDriverSetupInput {
            spin_channels: 3,
            ..base.clone()
        }),
        Err(FmsError::InvalidSpinChannelCount { value: 3 })
    );
    assert_eq!(
        fms_driver_setup(FmsDriverSetupInput {
            max_potential: 2,
            raw_potential_lmax: &[1, 1],
            ..base.clone()
        }),
        Err(FmsError::TableIndexOutOfRange {
            table: "lipotx",
            axis: "potential",
            index: 2,
        })
    );
    assert_eq!(
        fms_driver_setup(FmsDriverSetupInput {
            state_capacity: Some(2),
            ..base
        }),
        Err(FmsError::StateCapacityExceeded { capacity: 2 })
    );
}

#[test]
fn fms_real_space_energy_matches_manual_fmspack_sequence() -> Result<(), Box<dyn Error>> {
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
    let raw_lmax = [1, 1];
    let wave_numbers = [Complex32::new(1.2, 0.3), Complex32::new(0.8, 0.15)];
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let xnlm = legendre_normalization_table(2)?;
    let geometry = fms_yprep_geometry(2, 2, &atoms)?;
    let mut sigsqr = Array2::zeros((2, 2).f());
    sigsqr[(1, 0)] = 0.05;
    sigsqr[(0, 1)] = 0.05;
    let calculated_l = [true, true, true];

    let result = fms_real_space_energy(FmsRealSpaceEnergyInput {
        lfms: 1,
        minv: 0,
        spin_channels: 2,
        spin_selector: 0,
        atoms: &atoms,
        max_potential: 1,
        global_lmax: 2,
        raw_potential_lmax: &raw_lmax,
        state_capacity: None,
        wave_numbers: &wave_numbers,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
        direct_cutoff: 3.0,
        mean_square_displacements: sigsqr.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
        full_scattering_matrix_requested: false,
    })?;
    let manual_setup = fms_driver_setup(FmsDriverSetupInput {
        lfms: 1,
        spin_channels: 2,
        atoms: &atoms,
        max_potential: 1,
        global_lmax: 2,
        raw_potential_lmax: &raw_lmax,
        state_capacity: None,
    })?;
    let manual_pairs = fms_spin_pair_tables(2, &wave_numbers, &atoms)?;
    let manual_g0 = fms_spin_free_propagator_matrix(FmsSpinFreePropagatorMatrixInput {
        states: &manual_setup.state_kets.states,
        atoms: &atoms,
        direct_cutoff: 3.0,
        rho: manual_pairs.rho.view(),
        wave_numbers: &wave_numbers,
        mean_square_displacements: sigsqr.view(),
        xclm: manual_pairs.polynomials.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
    })?;
    let manual_t = fms_t_matrix_table(FmsTMatrixTableInput {
        states: &manual_setup.state_kets.states,
        atoms: &atoms,
        spin_channels: 2,
        spin_selector: 0,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
    })?;
    let manual_scattering = fms_scattering(FmsScatteringInput {
        method: FmsScatteringMethod::Lu,
        calculate_full_scattering: false,
        states: &manual_setup.state_kets.states,
        spin_channels: 2,
        global_lmax: 2,
        potential_lmax: &manual_setup.potential_lmax,
        representative_offsets: &manual_setup.state_kets.representative_offsets,
        potential_start: manual_setup.potential_start,
        potential_end: manual_setup.potential_end,
        free_propagator: manual_g0.view(),
        t_matrix: manual_t.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.setup, manual_setup);
    assert_eq!(result.method_selection.method, FmsScatteringMethod::Lu);
    assert_eq!(result.pair_tables, manual_pairs);
    assert_eq!(result.free_propagator, manual_g0);
    assert_eq!(result.t_matrix, manual_t);
    assert_eq!(result.scattering, manual_scattering);
    Ok(())
}

#[test]
fn fms_real_space_energy_forces_lu_for_full_scattering() -> Result<(), Box<dyn Error>> {
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
    let raw_lmax = [1, 1];
    let wave_numbers = [Complex32::new(1.2, 0.3), Complex32::new(0.8, 0.15)];
    let phases = reference_phase_shifts();
    let spin_orbit = spin_orbit_coupling_tables(2)?;
    let xnlm = legendre_normalization_table(2)?;
    let geometry = fms_yprep_geometry(2, 2, &atoms)?;
    let sigsqr = Array2::zeros((2, 2).f());
    let calculated_l = [true, true, true];

    let result = fms_real_space_energy(FmsRealSpaceEnergyInput {
        lfms: 1,
        minv: 3,
        spin_channels: 2,
        spin_selector: 0,
        atoms: &atoms,
        max_potential: 1,
        global_lmax: 2,
        raw_potential_lmax: &raw_lmax,
        state_capacity: None,
        wave_numbers: &wave_numbers,
        phase_shifts: phases.view(),
        spin_orbit: &spin_orbit,
        direct_cutoff: 3.0,
        mean_square_displacements: sigsqr.view(),
        xnlm: xnlm.view(),
        rotations: geometry.rotations.view(),
        calculated_l: &calculated_l,
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
        full_scattering_matrix_requested: true,
    })?;

    assert_eq!(
        result.method_selection,
        FmsScatteringMethodSelection {
            effective_minv: 0,
            method: FmsScatteringMethod::Lu,
            forced_lu_for_full_scattering: true,
        }
    );
    assert_eq!(result.scattering.method, FmsScatteringMethod::Lu);
    let Some(full_scattering) = result.scattering.full_scattering.as_ref() else {
        return Err("missing full scattering matrix".into());
    };
    assert_eq!(
        full_scattering.shape(),
        [
            result.setup.state_kets.states.len(),
            result.setup.state_kets.states.len(),
        ]
    );
    Ok(())
}
