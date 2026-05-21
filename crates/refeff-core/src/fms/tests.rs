use super::{
    FmsAtom, FmsBiCgStabInput, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput,
    FmsFullPotentialLuInput, FmsGravesMorrisInput, FmsIterativeSystemInput, FmsLuInput,
    FmsRealSpaceEnergyInput, FmsRecursionInput, FmsRotationDirection, FmsScatteringInput,
    FmsScatteringMethod, FmsScatteringMethodSelection, FmsSpinFreePropagatorMatrixInput,
    FmsTMatrixInput, FmsTMatrixTableInput, FmsTfqmrInput, FmsYprepClusterInput,
    MkgtrGreenTraceInput, fms_bicgstab_scattering, fms_driver_setup, fms_free_propagator_element,
    fms_free_propagator_matrix, fms_full_potential_lu_scattering, fms_graves_morris_scattering,
    fms_iterative_system_matrix, fms_lu_scattering, fms_pair_tables, fms_real_space_energy,
    fms_recursion_scattering, fms_rotation_matrix, fms_scattering, fms_scattering_method_selection,
    fms_spin_free_propagator_matrix, fms_spin_pair_tables, fms_t_matrix_element,
    fms_t_matrix_table, fms_tfqmr_scattering, fms_yprep_cluster, fms_yprep_geometry,
    mkgtr_green_trace, pair_polar_angles, sort_atoms_by_radius, sort_representative_atoms,
};
use super::{
    FmsDriverSetupInput, FmsError, rehr_albers_polynomials, rehr_albers_z_axis_propagator,
};
use crate::{
    Complex, Real,
    angular::{TransitionBMatrix, legendre_normalization_table, spin_orbit_coupling_tables},
    state::{StateKet, construct_state_kets},
};
use ndarray::{
    Array2, Array3, Array4, Array6, ArrayView2, ArrayView3, ArrayView4, Axis, ShapeBuilder, array,
};
use num_complex::Complex32;
use std::error::Error;

const REFERENCE_LCALC: [bool; 2] = [true, true];
const REFERENCE_POTENTIAL_LMAX: [usize; 1] = [1];

#[test]
fn fms_scattering_method_selection_matches_feff_minv_rules() {
    assert_eq!(
        fms_scattering_method_selection(0, false),
        FmsScatteringMethodSelection {
            effective_minv: 0,
            method: FmsScatteringMethod::Lu,
            forced_lu_for_full_scattering: false,
        }
    );
    assert_eq!(
        fms_scattering_method_selection(1, false).method,
        FmsScatteringMethod::BiCgStab
    );
    assert_eq!(
        fms_scattering_method_selection(2, false).method,
        FmsScatteringMethod::Recursion
    );
    assert_eq!(
        fms_scattering_method_selection(3, false).method,
        FmsScatteringMethod::GravesMorris
    );
    assert_eq!(
        fms_scattering_method_selection(4, false),
        FmsScatteringMethodSelection {
            effective_minv: 4,
            method: FmsScatteringMethod::Tfqmr,
            forced_lu_for_full_scattering: false,
        }
    );
    assert_eq!(
        fms_scattering_method_selection(-1, false).method,
        FmsScatteringMethod::Tfqmr
    );
    assert_eq!(
        fms_scattering_method_selection(3, true),
        FmsScatteringMethodSelection {
            effective_minv: 0,
            method: FmsScatteringMethod::Lu,
            forced_lu_for_full_scattering: true,
        }
    );
    assert_eq!(FmsScatteringMethod::Lu.feff_label(), "LUD");
    assert_eq!(FmsScatteringMethod::BiCgStab.feff_label(), "VdV");
    assert_eq!(FmsScatteringMethod::Recursion.feff_label(), "LLU");
    assert_eq!(FmsScatteringMethod::GravesMorris.feff_label(), "GMS");
    assert_eq!(FmsScatteringMethod::Tfqmr.feff_label(), "TF");
}

#[test]
fn fms_scattering_dispatches_lu_branch() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_scattering(reference_scattering_input(
        FmsScatteringMethod::Lu,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    ))?;

    assert_eq!(result.method, FmsScatteringMethod::Lu);
    assert_eq!(result.multiple_scattering_order, None);
    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.full_scattering, None);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(8.107_28, -0.542_959_87),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    Ok(())
}

#[test]
fn fms_scattering_dispatches_lu_full_matrix_request() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());
    let mut input = reference_scattering_input(
        FmsScatteringMethod::Lu,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    );
    input.calculate_full_scattering = true;

    let result = fms_scattering(input)?;

    assert_eq!(result.method, FmsScatteringMethod::Lu);
    assert_eq!(result.multiple_scattering_order, None);
    let Some(full_scattering) = result.full_scattering else {
        return Err("missing full scattering matrix".into());
    };
    assert_eq!(full_scattering.shape(), &[8, 8]);
    assert_complex32_close(
        matrix_sum(full_scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    Ok(())
}

#[test]
fn fms_scattering_dispatches_iterative_branches() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let cases = [
        (
            FmsScatteringMethod::BiCgStab,
            2,
            Complex32::new(-2.949_217_6, 4.806_942),
        ),
        (
            FmsScatteringMethod::Recursion,
            3,
            Complex32::new(-2.944_324, 4.799_402),
        ),
        (
            FmsScatteringMethod::GravesMorris,
            4,
            Complex32::new(-2.944_321_6, 4.799_405),
        ),
        (
            FmsScatteringMethod::Tfqmr,
            4,
            Complex32::new(-2.944_320_7, 4.799_402_7),
        ),
    ];

    for (method, order, scattering_reference) in cases {
        let result = fms_scattering(reference_scattering_input(
            method,
            &state_set.states,
            &state_set.representative_offsets,
            free_propagator.view(),
            t_matrix.view(),
        ))?;

        assert_eq!(result.method, method);
        assert_eq!(result.multiple_scattering_order, Some(order));
        assert_eq!(result.system_matrix.shape(), &[8, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.full_scattering, None);
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            scattering_reference,
        );
    }
    Ok(())
}

#[test]
fn fms_scattering_rejects_iterative_full_matrix_request() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());
    let mut input = reference_scattering_input(
        FmsScatteringMethod::BiCgStab,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    );
    input.calculate_full_scattering = true;

    assert!(matches!(
        fms_scattering(input),
        Err(FmsError::FullScatteringRequiresLu {
            method: FmsScatteringMethod::BiCgStab,
        })
    ));
    Ok(())
}

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

#[test]
fn mkgtr_green_trace_matches_feff_getgtr_loop() -> Result<(), Box<dyn Error>> {
    let mut first_matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    first_matrix.matrix[(0, 0, 0, 0, 0, 0)] = Complex::new(2.0, 0.5);
    let mut second_matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    second_matrix.matrix[(0, 0, 0, 0, 0, 0)] = Complex::new(-1.0, 0.25);
    let matrices = [first_matrix, second_matrix];
    let mut green = Array3::zeros((2, 1, 1).f());
    green[(0, 0, 0)] = Complex32::new(1.0, 2.0);
    green[(1, 0, 0)] = Complex32::new(-0.5, 0.75);
    let mut rkk = Array3::zeros((2, 8, 1).f());
    rkk[(0, 0, 0)] = Complex::new(3.0, -1.0);
    rkk[(1, 0, 0)] = Complex::new(0.5, 2.0);

    let result = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 1,
        green_functions: green.view(),
        transition_matrices: &matrices,
        transition_moments: rkk.view(),
    })?;

    assert_eq!(result.traces.shape(), &[2, 2]);
    assert_complex_close(
        result.traces[(0, 0)],
        widen_complex32_for_test(green[(0, 0, 0)])
            * matrices[0].matrix[(0, 0, 0, 0, 0, 0)]
            * rkk[(0, 0, 0)]
            * rkk[(0, 0, 0)],
    );
    assert_complex_close(
        result.traces[(1, 1)],
        widen_complex32_for_test(green[(1, 0, 0)])
            * matrices[1].matrix[(0, 0, 0, 0, 0, 0)]
            * rkk[(1, 0, 0)]
            * rkk[(1, 0, 0)],
    );
    Ok(())
}

#[test]
fn mkgtr_green_trace_uses_feff_spin_channel_indexing() -> Result<(), Box<dyn Error>> {
    let mut matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    matrix.matrix[(0, 1, 0, 0, 0, 0)] = Complex::new(1.5, -0.25);
    let matrices = [matrix];
    let mut green = Array3::zeros((1, 2, 2).f());
    green[(0, 0, 1)] = Complex32::new(0.5, -0.25);
    let mut rkk = Array3::zeros((1, 8, 2).f());
    rkk[(0, 0, 0)] = Complex::new(2.0, 0.0);
    rkk[(0, 0, 1)] = Complex::new(3.0, 0.5);

    let result = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 2,
        green_functions: green.view(),
        transition_matrices: &matrices,
        transition_moments: rkk.view(),
    })?;

    assert_complex_close(
        result.traces[(0, 0)],
        widen_complex32_for_test(green[(0, 0, 1)])
            * matrices[0].matrix[(0, 1, 0, 0, 0, 0)]
            * rkk[(0, 0, 0)]
            * rkk[(0, 0, 1)],
    );
    Ok(())
}

#[test]
fn mkgtr_green_trace_rejects_invalid_inputs() {
    let matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    let matrices = [matrix];
    let green = Array3::from_elem((1, 1, 1).f(), Complex32::new(f32::NAN, 0.0));
    let rkk = Array3::from_elem((1, 8, 1).f(), Complex::new(1.0, 0.0));

    assert_eq!(
        mkgtr_green_trace(MkgtrGreenTraceInput {
            active_spin_channels: 1,
            green_functions: green.view(),
            transition_matrices: &matrices,
            transition_moments: rkk.view(),
        }),
        Err(FmsError::NonFiniteComplexValue {
            table: "gg",
            index: 0,
        })
    );

    let short_rkk = Array3::zeros((1, 8, 0).f());
    assert_eq!(
        mkgtr_green_trace(MkgtrGreenTraceInput {
            active_spin_channels: 1,
            green_functions: Array3::zeros((1, 1, 1).f()).view(),
            transition_matrices: &matrices,
            transition_moments: short_rkk.view(),
        }),
        Err(FmsError::SpinChannelCountMismatch {
            table: "rkk",
            expected: 1,
            actual: 0,
        })
    );
}

#[test]
fn xclmz_matches_feff_reference_lx3() -> Result<(), FmsError> {
    let table = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;

    assert_eq!(table.shape(), &[5, 9]);
    assert_eq!(table.strides(), &[1, 5]);
    assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(table[(1, 0)], Complex32::new(1.2322206, 0.725_689_4));
    assert_complex32_close(table[(3, 0)], Complex32::new(-10.012509, 5.438_266));
    assert_complex32_close(table[(2, 1)], Complex32::new(-2.1395304, 4.1993084));
    assert_complex32_close(table[(3, 2)], Complex32::new(-23.036537, -6.8588142));
    assert_complex32_close(table[(4, 3)], Complex32::new(8.928_719, -161.62775));
    assert_complex32_close(
        matrix_sum(table.view()),
        Complex32::new(-58.983994, -154.61885),
    );
    assert_eq!(nonzero_count(table.view()), 11);
    Ok(())
}

#[test]
fn xclmz_matches_feff_reference_with_limited_m() -> Result<(), FmsError> {
    let table = rehr_albers_polynomials(4, 3, 2, Complex32::new(-0.8, 1.1))?;

    assert_eq!(table.shape(), &[6, 11]);
    assert_eq!(table.strides(), &[1, 6]);
    assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(table[(1, 0)], Complex32::new(1.5945946, -0.432_432_4));
    assert_complex32_close(table[(2, 0)], Complex32::new(3.2834187, -2.840029));
    assert_complex32_close(table[(1, 1)], Complex32::new(0.5945946, -0.432_432_4));
    assert_complex32_close(table[(2, 1)], Complex32::new(2.7830534, -4.382761));
    assert_complex32_close(
        matrix_sum(table.view()),
        Complex32::new(9.255661, -8.087655),
    );
    assert_eq!(nonzero_count(table.view()), 5);
    Ok(())
}

#[test]
fn xclmz_rejects_invalid_inputs() {
    assert_eq!(
        rehr_albers_polynomials(3, 0, 1, Complex32::new(1.0, 0.0)),
        Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 0,
            lx: 3,
        })
    );
    assert_eq!(
        rehr_albers_polynomials(3, 5, 1, Complex32::new(1.0, 0.0)),
        Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 5,
            lx: 3,
        })
    );
    assert_eq!(
        rehr_albers_polynomials(3, 1, 1, Complex32::new(0.0, 0.0)),
        Err(FmsError::ZeroRho)
    );
    assert_eq!(
        rehr_albers_polynomials(3, 1, 1, Complex32::new(f32::NAN, 0.0)),
        Err(FmsError::NonFiniteRho)
    );
}

#[test]
fn rotxan_matches_feff_reference_forward_and_backward() -> Result<(), FmsError> {
    let forward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let backward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Backward)?;

    assert_eq!(forward.shape(), &[7, 7, 4]);
    assert_eq!(forward.strides(), &[1, 7, 49]);
    assert_complex32_close(
        rotation_sum(forward.view()),
        Complex32::new(1.159_583_6, 0.288_981_8),
    );
    assert_complex32_close(
        rotation_sum(backward.view()),
        Complex32::new(1.159_583_1, 0.288_981_74),
    );
    assert_eq!(rotation_nonzero_count(forward.view()), 84);
    assert_eq!(rotation_nonzero_count(backward.view()), 84);

    assert_complex32_close(rotation_value(&forward, 0, 0, 0), Complex32::new(1.0, 0.0));
    assert_complex32_close(
        rotation_value(&forward, 1, -1, 1),
        Complex32::new(-0.053_333_33, -0.104_787_19),
    );
    assert_complex32_close(
        rotation_value(&forward, -1, 1, 1),
        Complex32::new(-0.053_333_33, 0.104_787_19),
    );
    assert_complex32_close(
        rotation_value(&forward, 2, -1, 2),
        Complex32::new(-0.044_576_85, 0.061_240_695),
    );
    assert_complex32_close(
        rotation_value(&forward, -2, 1, 3),
        Complex32::new(0.116_102_73, 0.159_504_58),
    );
    assert_complex32_close(
        rotation_value(&forward, 3, 3, 3),
        Complex32::new(0.678_509_35, 0.108_389_09),
    );

    assert_complex32_close(
        rotation_value(&backward, 2, -1, 2),
        Complex32::new(-0.034_358_274, -0.067_505_76),
    );
    assert_complex32_close(
        rotation_value(&backward, -2, 1, 3),
        Complex32::new(0.089_487_91, -0.175_822_26),
    );
    assert_complex32_close(
        rotation_value(&backward, 3, 3, 3),
        Complex32::new(0.678_509_35, -0.108_389_09),
    );
    Ok(())
}

#[test]
fn rotxan_rejects_invalid_inputs() {
    assert_eq!(
        fms_rotation_matrix(25, 1, 0.0, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::InvalidAngularLimit {
            name: "lmax",
            value: 25,
            lx: 24,
        })
    );
    assert_eq!(
        fms_rotation_matrix(3, 4, 0.0, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::InvalidAngularLimit {
            name: "mmax",
            value: 4,
            lx: 3,
        })
    );
    assert_eq!(
        fms_rotation_matrix(3, 3, f32::NAN, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::NonFiniteRotationAngle { name: "beta" })
    );
}

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

    assert_complex32_close(value, Complex32::new(-0.103_387_31, 0.105_749_39));
    Ok(())
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
    assert_complex32_close(matrix[(0, 1)], Complex32::new(-0.103_387_31, 0.105_749_39));
    assert_complex32_close(matrix[(1, 0)], Complex32::new(0.0, 0.0));
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
fn fms_iterative_system_matrix_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let system = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: 0.0,
    })?;

    assert_eq!(system.shape(), &[8, 8]);
    assert_eq!(system.strides(), &[1, 8]);
    assert_complex32_close(
        matrix_sum(system.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(system[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(system[(1, 3)], Complex32::new(0.001_4, -0.003_199_999_7));
    assert_complex32_close(
        system[(4, 5)],
        Complex32::new(0.001_230_000_3, -0.011_239_999),
    );
    assert_complex32_close(system[(6, 7)], Complex32::new(0.001_789_999_7, -0.020_9));

    let cutoff_system = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: 0.09,
    })?;

    assert_complex32_close(
        matrix_sum(cutoff_system.view()),
        Complex32::new(7.922_833_4, -0.471_125_07),
    );
    assert_complex32_close(cutoff_system[(1, 3)], Complex32::new(0.0, 0.0));
    assert_complex32_close(
        cutoff_system[(4, 5)],
        Complex32::new(0.001_230_000_3, -0.011_239_999),
    );
    Ok(())
}

#[test]
fn fms_iterative_system_matrix_rejects_invalid_tolerance() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: -1.0,
    });

    assert!(matches!(
        result,
        Err(FmsError::InvalidTolerance {
            name: "toler2",
            value: -1.0,
        })
    ));
    Ok(())
}

#[test]
fn fms_bicgstab_scattering_matches_feff_ggbi_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_bicgstab_scattering(FmsBiCgStabInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 2);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.949_217_6, 4.806_942),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_855_818, -0.003_201_462_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.066_029_795, 0.044_123_195),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_492_656, 0.140_840_8),
    );
    Ok(())
}

#[test]
fn fms_bicgstab_scattering_respects_lcalc_mask() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_bicgstab_scattering(FmsBiCgStabInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, false],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_855_818, -0.003_201_462_3),
    );
    assert_complex32_close(result.scattering[(2, 2, 0)], Complex32::new(0.0, 0.0));
    assert_complex32_close(result.scattering[(7, 7, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_recursion_scattering_matches_feff_ggrm_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_recursion_scattering(FmsRecursionInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 3);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_324, 4.799_402),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_021, -0.003_244_287_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_52, 0.044_093_154),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_72, 0.140_520_17),
    );
    Ok(())
}

#[test]
fn fms_graves_morris_scattering_matches_feff_gggm_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_graves_morris_scattering(FmsGravesMorrisInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 4);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(0.090_419_99, 0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_321_6, 4.799_405),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_049_4, -0.003_244_209),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_47, 0.044_093_188),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_895, 0.140_520_08),
    );
    Ok(())
}

#[test]
fn fms_tfqmr_scattering_matches_feff_ggtf_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_tfqmr_scattering(FmsTfqmrInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 4);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_7, 4.799_402_7),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_021_4, -0.003_244_287_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_43, 0.044_093_173),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_91, 0.140_520_1),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_matches_feff_gglu_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: false,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.full_scattering, None);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(8.107_28, -0.542_959_87),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_020_5, -0.003_244_286_6),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_42, 0.044_093_15),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_9, 0.140_520_07),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_returns_feff_gg_full_when_requested() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0, 1], &[1, 0], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: true,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1, 0],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 1,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    let Some(full_scattering) = result.full_scattering else {
        return Err("missing full scattering matrix".into());
    };
    assert_eq!(full_scattering.shape(), &[10, 10]);
    assert_eq!(result.scattering.shape(), &[8, 8, 2]);
    assert_complex32_close(
        matrix_sum(full_scattering.view()),
        Complex32::new(-6.616_672_5, 8.779_471),
    );
    assert_complex32_close(
        full_scattering[(0, 9)],
        Complex32::new(-0.189_542, 0.041_967_187),
    );
    assert_complex32_close(
        full_scattering[(9, 0)],
        Complex32::new(0.063_354_82, 0.163_031_2),
    );

    for potential in 0..=1 {
        let lmax = [1, 0][potential];
        let ipart = 2 * (lmax + 1) * (lmax + 1);
        let offset = match state_set.representative_offsets[potential] {
            Some(offset) => offset,
            None => return Err("missing representative offset".into()),
        };
        for column in 0..ipart {
            for row in 0..ipart {
                assert_complex32_close(
                    result.scattering[(row, column, potential)],
                    full_scattering[(offset + row, offset + column)],
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fms_full_potential_lu_scattering_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, _) = reference_gglu_inputs(state_set.states.len());
    let t_matrix = reference_full_potential_t_matrix(state_set.states.len());

    let result = fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(8.191_353, -0.610_848),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.843_191_9, 4.688_064),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.006_074_232, -0.004_277_690_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.063_446_34, 0.043_493_286),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_970_54, 0.136_094_53),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_rejects_missing_representative() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: false,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &[None],
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    });

    assert!(matches!(
        result,
        Err(FmsError::MissingRepresentativePotential { potential: 0 })
    ));
    Ok(())
}

#[test]
fn atheap_matches_feff_reference_sort_order() -> Result<(), FmsError> {
    let mut atoms = vec![
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [-1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [0.0, 2.0, 0.0],
            potential: 4,
        },
    ];

    let keys = sort_atoms_by_radius(&mut atoms)?;

    assert_eq!(
        atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
        vec![0, 2, 3, 1, 4]
    );
    assert_eq!(atoms[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(atoms[1].position, [-1.0, 0.0, 0.0]);
    assert_close_f64(keys[0], 2.0e-6);
    assert_close_f64(keys[1], 1.000_003);
    assert_close_f64(keys[2], 1.000_004);
    assert_close_f64(keys[3], 4.000_001);
    assert_close_f64(keys[4], 4.000_005);
    Ok(())
}

#[test]
fn getang_matches_feff_reference_angles() -> Result<(), FmsError> {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0],
        [0.0, 5.0e-8, 2.0e-7],
        [0.0, 2.0e-7, 0.0],
    ];

    let (theta, phi) = pair_polar_angles(&positions, 1, 0)?;
    assert_close_f32(theta, 0.841_068_6);
    assert_close_f32(phi, 1.107_148_8);

    let (theta, phi) = pair_polar_angles(&positions, 3, 2)?;
    assert_close_f32(theta, 2.498_091_5);
    assert_close_f32(phi, 1.570_796_4);

    assert_eq!(pair_polar_angles(&positions, 0, 0)?, (0.0, 0.0));
    Ok(())
}

#[test]
fn sortat_matches_feff_reference_representative_order() -> Result<(), FmsError> {
    let mut atoms = vec![
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [2.0, 0.0, 0.0],
            potential: 1,
        },
        FmsAtom {
            position: [3.0, 0.0, 0.0],
            potential: 3,
        },
        FmsAtom {
            position: [4.0, 0.0, 0.0],
            potential: 2,
        },
        FmsAtom {
            position: [5.0, 0.0, 0.0],
            potential: 1,
        },
    ];

    let representatives = sort_representative_atoms(0, 3, &mut atoms)?;

    assert_eq!(representatives, vec![Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(
        atoms.iter().map(|atom| atom.potential).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 2, 1]
    );
    assert_eq!(atoms[1].position, [2.0, 0.0, 0.0]);
    assert_eq!(atoms[2].position, [1.0, 0.0, 0.0]);
    assert_eq!(atoms[3].position, [3.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn yprep_cluster_matches_feff_radius_prefix_reference() -> Result<(), FmsError> {
    let positions = array![
        [2.0_f32, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let potentials = [1, 0, 2, 1, 2];

    let cluster = fms_yprep_cluster(FmsYprepClusterInput {
        central_potential: 0,
        potentials: &potentials,
        positions: positions.view(),
        cluster_radius: 2.1,
        cluster_capacity: 3,
    })?;

    assert_eq!(cluster.central_atom, 1);
    assert_eq!(cluster.untruncated_count, 4);
    assert_eq!(cluster.atoms.len(), 3);
    assert_eq!(
        cluster
            .atoms
            .iter()
            .map(|atom| atom.potential)
            .collect::<Vec<_>>(),
        vec![0, 2, 1]
    );
    assert_eq!(cluster.atoms[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(cluster.atoms[1].position, [0.0, 0.0, 1.0]);
    assert_eq!(cluster.atoms[2].position, [1.0, -1.0, 0.0]);
    Ok(())
}

#[test]
fn yprep_geometry_matches_feff_pair_rotation_sequence() -> Result<(), FmsError> {
    let atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [0.0, 0.0, 1.0],
            potential: 2,
        },
        FmsAtom {
            position: [1.0, -1.0, 0.0],
            potential: 1,
        },
    ];

    let geometry = fms_yprep_geometry(2, 2, &atoms)?;

    assert_eq!(geometry.phi.shape(), &[3, 3]);
    assert_eq!(geometry.rotations.shape(), &[5, 5, 3, 2, 3, 3]);
    assert_close_f32(geometry.phi[(1, 0)], 0.0);
    assert_close_f32(geometry.phi[(2, 0)], -std::f32::consts::FRAC_PI_4);
    assert_close_f32(geometry.phi[(0, 2)], 3.0 * std::f32::consts::FRAC_PI_4);
    assert_complex32_close(
        geometry.rotations[(2, 2, 0, 0, 0, 0)],
        Complex32::new(0.0, 0.0),
    );

    let expected_forward = fms_rotation_matrix(
        2,
        2,
        std::f32::consts::FRAC_PI_2,
        -std::f32::consts::FRAC_PI_4,
        FmsRotationDirection::Forward,
    )?;
    let expected_backward = fms_rotation_matrix(
        2,
        2,
        -std::f32::consts::FRAC_PI_2,
        -std::f32::consts::FRAC_PI_4,
        FmsRotationDirection::Backward,
    )?;
    assert_complex32_close(
        geometry.rotations[(3, 1, 1, 0, 2, 0)],
        expected_forward[(3, 1, 1)],
    );
    assert_complex32_close(
        geometry.rotations[(1, 3, 2, 1, 2, 0)],
        expected_backward[(1, 3, 2)],
    );
    Ok(())
}

#[test]
fn fms_cluster_helpers_reject_invalid_inputs() {
    let positions = [[0.0, 0.0, 0.0]];
    assert_eq!(
        pair_polar_angles(&positions, 1, 0),
        Err(FmsError::AtomIndexOutOfRange { index: 1, len: 1 })
    );

    let mut atoms = [FmsAtom {
        position: [f32::NAN, 0.0, 0.0],
        potential: 0,
    }];
    assert_eq!(
        sort_atoms_by_radius(&mut atoms),
        Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
    );

    let mut atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    assert_eq!(
        sort_representative_atoms(0, 1, &mut atoms),
        Err(FmsError::CentralAtomMismatch {
            expected: 0,
            actual: 1,
        })
    );
    assert_eq!(
        sort_representative_atoms(-1, 1, &mut atoms),
        Err(FmsError::PotentialOutOfRange {
            potential: -1,
            max_potential: 1,
        })
    );

    let yprep_positions = array![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 0,
            potentials: &[0, 0],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::DuplicateAbsorber)
    );
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 2,
            potentials: &[0, 1],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::MissingCentralAtom { potential: 2 })
    );
    assert_eq!(
        fms_yprep_cluster(FmsYprepClusterInput {
            central_potential: 0,
            potentials: &[0],
            positions: yprep_positions.view(),
            cluster_radius: 1.0,
            cluster_capacity: 2,
        }),
        Err(FmsError::AtomCountMismatch {
            potentials: 1,
            positions: 2,
        })
    );
    assert_eq!(
        fms_yprep_geometry(2, 2, &[]),
        Err(FmsError::AtomIndexOutOfRange { index: 0, len: 0 })
    );
    assert_eq!(
        fms_yprep_geometry(
            2,
            2,
            &[FmsAtom {
                position: [f32::NAN, 0.0, 0.0],
                potential: 0,
            }],
        ),
        Err(FmsError::NonFiniteCoordinate { atom: 0, axis: 0 })
    );
}

#[test]
fn xgllm_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };

    assert_complex32_close(
        rehr_albers_z_axis_propagator(0, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(415.546_9, -1006.2809),
    );
    assert_complex32_close(
        rehr_albers_z_axis_propagator(1, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(-307.497_3, 722.469_5),
    );
    assert_complex32_close(
        rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(115.08963, -235.94589),
    );
    Ok(())
}

#[test]
fn xgllm_matches_feff_empty_sum_case() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };

    assert_complex32_close(
        rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(0.0, 0.0),
    );
    Ok(())
}

#[test]
fn xgllm_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };

    assert_eq!(
        rehr_albers_z_axis_propagator(3, first, second, xclm.view(), xnlm.view()),
        Err(FmsError::MuOutOfRange {
            mu: 3,
            angular_momentum: 2,
        })
    );
    assert_eq!(
        rehr_albers_z_axis_propagator(
            0,
            StateKet { atom: 0, ..first },
            second,
            xclm.view(),
            xnlm.view(),
        ),
        Err(FmsError::InvalidStateAtom { atom: 0 })
    );

    let mut bad_xnlm = xnlm.clone();
    bad_xnlm[(0, 2)] = 0.0;
    assert_eq!(
        rehr_albers_z_axis_propagator(0, first, second, xclm.view(), bad_xnlm.view()),
        Err(FmsError::InvalidNormalization {
            mu: 0,
            angular_momentum: 2,
        })
    );
    Ok(())
}

fn reference_xgllm_tables() -> Result<(Array4<Complex32>, Array2<Real>), Box<dyn Error>> {
    let clm = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;
    let mut xclm = Array4::zeros((4, 4, 2, 2).f());
    for l in 0..=3 {
        for m in 0..=3 {
            xclm[(m, l, 1, 0)] = clm[(l, m)];
            xclm[(m, l, 0, 1)] = clm[(l, m)];
        }
    }
    Ok((xclm, legendre_normalization_table(3)?))
}

fn reference_phase_shifts() -> Array3<Complex32> {
    let mut phases = Array3::zeros((2, 5, 2).f());
    phases[(0, 4, 1)] = Complex32::new(0.2, 0.05);
    phases[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
    phases[(1, 4, 1)] = Complex32::new(0.15, -0.02);
    phases[(1, 0, 1)] = Complex32::new(0.07, 0.04);
    phases
}

fn reference_gglu_inputs(state_count: usize) -> (Array2<Complex32>, Array2<Complex32>) {
    let mut free_propagator = Array2::zeros((state_count, state_count).f());
    let mut t_matrix = Array2::zeros((2, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            if row != column {
                free_propagator[(row, column)] = Complex32::new(
                    0.01 * row_feff - 0.02 * column_feff,
                    0.015 * row_feff + 0.005 * column_feff,
                );
            }
        }
        let column_feff = column as f32 + 1.0;
        t_matrix[(0, column)] = Complex32::new(0.02 * column_feff, -0.01 * column_feff);
        t_matrix[(1, column)] = Complex32::new(-0.005 * column_feff, 0.003 * column_feff);
    }
    (free_propagator, t_matrix)
}

fn reference_scattering_input<'a>(
    method: FmsScatteringMethod,
    states: &'a [StateKet],
    representative_offsets: &'a [Option<usize>],
    free_propagator: ArrayView2<'a, Complex32>,
    t_matrix: ArrayView2<'a, Complex32>,
) -> FmsScatteringInput<'a> {
    FmsScatteringInput {
        method,
        calculate_full_scattering: false,
        states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &REFERENCE_POTENTIAL_LMAX,
        representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator,
        t_matrix,
        calculated_l: &REFERENCE_LCALC,
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    }
}

fn reference_full_potential_t_matrix(state_count: usize) -> Array2<Complex32> {
    let mut t_matrix = Array2::zeros((state_count, state_count).f());
    for column in 0..state_count {
        for row in 0..state_count {
            let row_feff = row as f32 + 1.0;
            let column_feff = column as f32 + 1.0;
            t_matrix[(row, column)] = Complex32::new(
                0.002 * row_feff + 0.001 * column_feff,
                -0.0015 * row_feff + 0.0007 * column_feff,
            );
        }
    }
    t_matrix
}

fn matrix_sum(matrix: ArrayView2<'_, Complex32>) -> Complex32 {
    matrix
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn nonzero_count(matrix: ArrayView2<'_, Complex32>) -> usize {
    matrix
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn rotation_sum(matrix: ArrayView3<'_, Complex32>) -> Complex32 {
    matrix
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn rotation_nonzero_count(matrix: ArrayView3<'_, Complex32>) -> usize {
    matrix
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn scattering_sum(table: ArrayView3<'_, Complex32>) -> Complex32 {
    table
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn pair_table_sum(table: ArrayView4<'_, Complex32>) -> Complex32 {
    table
        .iter()
        .copied()
        .fold(Complex32::new(0.0, 0.0), |sum, value| sum + value)
}

fn pair_table_nonzero_count(table: ArrayView4<'_, Complex32>) -> usize {
    table
        .iter()
        .filter(|value| value.re.abs() + value.im.abs() > 1.0e-6)
        .count()
}

fn rotation_value(
    matrix: &Array3<Complex32>,
    m2: isize,
    m1: isize,
    angular_momentum: usize,
) -> Complex32 {
    let offset = 3_isize;
    matrix[(
        (m2 + offset) as usize,
        (m1 + offset) as usize,
        angular_momentum,
    )]
}

fn copy_rotation_pair(
    rotations: &mut Array6<Complex32>,
    atom2: usize,
    atom1: usize,
    direction: FmsRotationDirection,
    table: &Array3<Complex32>,
) {
    let branch = match direction {
        FmsRotationDirection::Forward => 0,
        FmsRotationDirection::Backward => 1,
    };
    for l in 0..table.shape()[2] {
        for m1 in 0..table.shape()[1] {
            for m2 in 0..table.shape()[0] {
                rotations[(m2, m1, l, branch, atom2, atom1)] = table[(m2, m1, l)];
            }
        }
    }
}

fn sample_mkgtr_transition_matrix(orbital_momenta: [i32; 8]) -> TransitionBMatrix {
    TransitionBMatrix {
        kappa_indices: [0; 8],
        orbital_momenta,
        matrix: Array6::zeros((1, 2, 8, 1, 2, 8).f()),
        l_offset: 0,
    }
}

fn widen_complex32_for_test(value: Complex32) -> Complex {
    Complex::new(value.re as Real, value.im as Real)
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).norm() < 1.0e-11,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert!(
        (actual - expected).norm() < 2.0e-4,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_close_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 2.0e-6,
        "actual={actual} expected={expected}"
    );
}

fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected}"
    );
}
