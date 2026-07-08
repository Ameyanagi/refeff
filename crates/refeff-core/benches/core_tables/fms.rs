use super::*;

pub(super) fn bench_fms(c: &mut Criterion) {
    let driver_setup_atoms = sample_pair_table_atoms();
    c.bench_function("fms_driver_setup_atoms3_l2", |b| {
        b.iter(|| {
            black_box(fms_driver_setup(black_box(FmsDriverSetupInput {
                lfms: black_box(1),
                spin_channels: black_box(2),
                atoms: black_box(&driver_setup_atoms),
                max_potential: black_box(2),
                global_lmax: black_box(2),
                raw_potential_lmax: black_box(&[-1, 2, 1]),
                state_capacity: black_box(None),
            })))
        });
    });

    c.bench_function("rehr_albers_polynomials_lx3", |b| {
        b.iter(|| {
            black_box(rehr_albers_polynomials(
                black_box(3),
                black_box(4),
                black_box(4),
                black_box(Complex32::new(1.25, 0.4)),
            ))
        });
    });

    let Ok(clm) = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4)) else {
        return;
    };
    let Ok(xnlm) = legendre_normalization_table(3) else {
        return;
    };
    let mut xclm = Array4::zeros((4, 4, 2, 2).f());
    for l in 0..=3 {
        for m in 0..=3 {
            xclm[(m, l, 1, 0)] = clm[(l, m)];
        }
    }
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
    c.bench_function("rehr_albers_z_axis_propagator_mu1", |b| {
        b.iter(|| {
            black_box(rehr_albers_z_axis_propagator(
                black_box(1),
                black_box(first),
                black_box(second),
                black_box(xclm.view()),
                black_box(xnlm.view()),
            ))
        });
    });

    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 2.0],
        [0.0, 5.0e-8, 2.0e-7],
        [0.0, 2.0e-7, 0.0],
    ];
    c.bench_function("fms_pair_polar_angles", |b| {
        b.iter(|| {
            black_box(pair_polar_angles(
                black_box(&positions),
                black_box(1),
                black_box(0),
            ))
        });
    });

    c.bench_function("fms_sort_atoms_by_radius", |b| {
        b.iter(|| {
            let mut atoms = sample_fms_atoms();
            black_box(sort_atoms_by_radius(black_box(&mut atoms[..])))
        });
    });
    let yprep_positions = array![
        [2.0_f32, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let yprep_potentials = [1, 0, 2, 1, 2];
    c.bench_function("fms_yprep_cluster_atoms5", |b| {
        b.iter(|| {
            black_box(fms_yprep_cluster(black_box(FmsYprepClusterInput {
                central_potential: 0,
                potentials: &yprep_potentials,
                positions: yprep_positions.view(),
                cluster_radius: 2.1,
                cluster_capacity: 4,
            })))
        });
    });
    let yprep_geometry_atoms = [
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
    c.bench_function("fms_yprep_geometry_l2_atoms3", |b| {
        b.iter(|| {
            black_box(fms_yprep_geometry(
                black_box(2),
                black_box(2),
                black_box(&yprep_geometry_atoms),
            ))
        });
    });
    c.bench_function("fms_sort_representative_atoms", |b| {
        b.iter(|| {
            let mut atoms = sample_representative_atoms();
            black_box(sort_representative_atoms(
                black_box(0),
                black_box(3),
                black_box(&mut atoms[..]),
            ))
        });
    });
    c.bench_function("fms_rotation_matrix_l3", |b| {
        b.iter(|| {
            black_box(fms_rotation_matrix(
                black_box(3),
                black_box(3),
                black_box(0.7),
                black_box(1.1),
                black_box(FmsRotationDirection::Forward),
            ))
        });
    });
    c.bench_function("fms_pair_tables_l2_atoms3", |b| {
        b.iter(|| {
            black_box(fms_pair_tables(
                black_box(2),
                black_box(Complex32::new(1.2, 0.3)),
                black_box(&sample_pair_table_atoms()),
            ))
        });
    });
    c.bench_function("fms_spin_pair_tables_l2_atoms3_nsp2", |b| {
        b.iter(|| {
            black_box(fms_spin_pair_tables(
                black_box(2),
                black_box(&[Complex32::new(1.2, 0.3), Complex32::new(1.05, 0.45)]),
                black_box(&sample_pair_table_atoms()),
            ))
        });
    });

    let pair_atoms = sample_pair_table_atoms();
    let free_wave_number = Complex32::new(1.2, 0.3);
    let spin_wave_numbers = [Complex32::new(1.2, 0.3), Complex32::new(1.05, 0.45)];
    let Ok(pair_tables) = fms_pair_tables(2, free_wave_number, &pair_atoms) else {
        return;
    };
    let Ok(spin_pair_tables) = fms_spin_pair_tables(2, &spin_wave_numbers, &pair_atoms) else {
        return;
    };
    let Ok(free_xnlm) = legendre_normalization_table(2) else {
        return;
    };
    let Ok(backward_rotation) = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Backward)
    else {
        return;
    };
    let Ok(forward_rotation) = fms_rotation_matrix(2, 2, 0.7, 1.1, FmsRotationDirection::Forward)
    else {
        return;
    };
    let free_first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 1,
        spin: 1,
    };
    let free_second = StateKet {
        atom: 2,
        angular_momentum: 2,
        magnetic: -1,
        spin: 1,
    };
    c.bench_function("fms_free_propagator_element_l2", |b| {
        b.iter(|| {
            black_box(fms_free_propagator_element(FmsFreePropagatorInput {
                first: black_box(free_first),
                second: black_box(free_second),
                rho: black_box(pair_tables.rho[(0, 1)]),
                wave_number: black_box(free_wave_number),
                mean_square_displacement: black_box(0.05),
                xclm: black_box(pair_tables.polynomials.view()),
                xnlm: black_box(free_xnlm.view()),
                backward_rotation: black_box(backward_rotation.view()),
                forward_rotation: black_box(forward_rotation.view()),
            }))
        });
    });

    let mut free_rotations = Array6::zeros((5, 5, 3, 2, 3, 3).f());
    copy_rotation_pair(
        &mut free_rotations,
        1,
        0,
        FmsRotationDirection::Backward,
        &backward_rotation,
    );
    copy_rotation_pair(
        &mut free_rotations,
        1,
        0,
        FmsRotationDirection::Forward,
        &forward_rotation,
    );
    let mut free_sigsqr = Array2::zeros((3, 3).f());
    free_sigsqr[(1, 0)] = 0.05;
    let free_states = [free_first, free_second];
    c.bench_function("fms_free_propagator_matrix_states2", |b| {
        b.iter(|| {
            black_box(fms_free_propagator_matrix(FmsFreePropagatorMatrixInput {
                states: black_box(&free_states),
                atoms: black_box(&pair_atoms),
                direct_cutoff: black_box(3.0),
                rho: black_box(pair_tables.rho.view()),
                wave_number: black_box(free_wave_number),
                mean_square_displacements: black_box(free_sigsqr.view()),
                xclm: black_box(pair_tables.polynomials.view()),
                xnlm: black_box(free_xnlm.view()),
                rotations: black_box(free_rotations.view()),
            }))
        });
    });
    let spin_free_states = [
        free_first,
        free_second,
        StateKet {
            spin: 2,
            ..free_first
        },
        StateKet {
            spin: 2,
            ..free_second
        },
    ];
    c.bench_function("fms_spin_free_propagator_matrix_states4", |b| {
        b.iter(|| {
            black_box(fms_spin_free_propagator_matrix(
                FmsSpinFreePropagatorMatrixInput {
                    states: black_box(&spin_free_states),
                    atoms: black_box(&pair_atoms),
                    direct_cutoff: black_box(3.0),
                    rho: black_box(spin_pair_tables.rho.view()),
                    wave_numbers: black_box(&spin_wave_numbers),
                    mean_square_displacements: black_box(free_sigsqr.view()),
                    xclm: black_box(spin_pair_tables.polynomials.view()),
                    xnlm: black_box(free_xnlm.view()),
                    rotations: black_box(free_rotations.view()),
                },
            ))
        });
    });

    let mut phase_shifts = Array3::zeros((2, 5, 2).f());
    phase_shifts[(0, 4, 1)] = Complex32::new(0.2, 0.05);
    phase_shifts[(0, 0, 1)] = Complex32::new(-0.1, 0.03);
    phase_shifts[(1, 4, 1)] = Complex32::new(0.15, -0.02);
    phase_shifts[(1, 0, 1)] = Complex32::new(0.07, 0.04);
    let Ok(t_matrix_spin_orbit) = spin_orbit_coupling_tables(2) else {
        return;
    };
    c.bench_function("fms_t_matrix_element_spin_mix_l2", |b| {
        b.iter(|| {
            black_box(fms_t_matrix_element(FmsTMatrixInput {
                first: black_box(free_first),
                second: black_box(StateKet {
                    magnetic: 0,
                    spin: 2,
                    ..free_first
                }),
                spin_channels: black_box(2),
                spin_selector: black_box(0),
                potential: black_box(1),
                phase_shifts: black_box(phase_shifts.view()),
                spin_orbit: black_box(&t_matrix_spin_orbit),
            }))
        });
    });

    let t_matrix_atoms = [FmsAtom {
        position: [0.0, 0.0, 0.0],
        potential: 1,
    }];
    let t_matrix_states = [
        free_first,
        StateKet {
            magnetic: 0,
            spin: 2,
            ..free_first
        },
    ];
    c.bench_function("fms_t_matrix_table_states2", |b| {
        b.iter(|| {
            black_box(fms_t_matrix_table(FmsTMatrixTableInput {
                states: black_box(&t_matrix_states),
                atoms: black_box(&t_matrix_atoms),
                spin_channels: black_box(2),
                spin_selector: black_box(0),
                phase_shifts: black_box(phase_shifts.view()),
                spin_orbit: black_box(&t_matrix_spin_orbit),
            }))
        });
    });
    let real_space_atoms = [
        FmsAtom {
            position: [0.0, 0.0, 0.0],
            potential: 0,
        },
        FmsAtom {
            position: [1.0, 2.0, 2.0],
            potential: 1,
        },
    ];
    c.bench_function("fms_real_space_energy_atoms2_l2_lu", |b| {
        b.iter(|| {
            black_box(fms_real_space_energy(FmsRealSpaceEnergyInput {
                lfms: black_box(1),
                minv: black_box(0),
                spin_channels: black_box(2),
                spin_selector: black_box(0),
                atoms: black_box(&real_space_atoms),
                max_potential: black_box(1),
                global_lmax: black_box(2),
                raw_potential_lmax: black_box(&[1, 1]),
                state_capacity: black_box(None),
                wave_numbers: black_box(&spin_wave_numbers),
                phase_shifts: black_box(phase_shifts.view()),
                spin_orbit: black_box(&t_matrix_spin_orbit),
                direct_cutoff: black_box(3.0),
                mean_square_displacements: black_box(free_sigsqr.view()),
                xnlm: black_box(free_xnlm.view()),
                rotations: black_box(free_rotations.view()),
                calculated_l: black_box(&[true, true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
                full_scattering_matrix_requested: black_box(false),
            }))
        });
    });

    let mut mkgtr_matrix = TransitionBMatrix {
        kappa_indices: [0; 8],
        orbital_momenta: [0, -1, -1, -1, -1, -1, -1, -1],
        matrix: Array6::zeros((1, 2, 8, 1, 2, 8).f()),
        l_offset: 0,
    };
    mkgtr_matrix.matrix[(0, 0, 0, 0, 0, 0)] = Complex::new(2.0, 0.5);
    let mkgtr_matrices = [mkgtr_matrix];
    let mkgtr_green = Array3::from_shape_fn((32, 1, 1).f(), |(energy, _, _)| {
        Complex32::new(1.0 + energy as f32 * 0.01, 0.5 - energy as f32 * 0.002)
    });
    let mkgtr_rkk = Array3::from_shape_fn((32, 8, 1).f(), |(energy, transition, _)| {
        if transition == 0 {
            Complex::new(1.0 + energy as f64 * 0.005, -0.25)
        } else {
            Complex::new(0.0, 0.0)
        }
    });
    c.bench_function("mkgtr_green_trace_ne32_spectra1_l0", |b| {
        b.iter(|| {
            black_box(mkgtr_green_trace(MkgtrGreenTraceInput {
                active_spin_channels: black_box(1),
                green_functions: black_box(mkgtr_green.view()),
                transition_matrices: black_box(&mkgtr_matrices),
                transition_moments: black_box(mkgtr_rkk.view()),
            }))
        });
    });

    let Ok(lu_states) = construct_state_kets(2, &[0], &[1], 1) else {
        return;
    };
    let (lu_g0, lu_t) = reference_gglu_inputs(lu_states.states.len());
    c.bench_function("fms_scattering_dispatch_lu_states8", |b| {
        b.iter(|| {
            black_box(fms_scattering(FmsScatteringInput {
                method: black_box(FmsScatteringMethod::Lu),
                calculate_full_scattering: black_box(false),
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_iterative_system_states8", |b| {
        b.iter(|| {
            black_box(fms_iterative_system_matrix(FmsIterativeSystemInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_bicgstab_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_bicgstab_scattering(FmsBiCgStabInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_tfqmr_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_tfqmr_scattering(FmsTfqmrInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_recursion_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_recursion_scattering(FmsRecursionInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_graves_morris_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_graves_morris_scattering(FmsGravesMorrisInput {
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
                calculated_l: black_box(&[true, true]),
                convergence_tolerance: black_box(1.0e-5),
                zero_tolerance: black_box(0.0),
            }))
        });
    });
    c.bench_function("fms_lu_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_lu_scattering(FmsLuInput {
                states: black_box(&lu_states.states),
                calculate_full_scattering: black_box(false),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
            }))
        });
    });
    c.bench_function("fms_lu_scattering_full_matrix_states8", |b| {
        b.iter(|| {
            black_box(fms_lu_scattering(FmsLuInput {
                states: black_box(&lu_states.states),
                calculate_full_scattering: black_box(true),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t.view()),
            }))
        });
    });
    let lu_t_full = reference_full_potential_t_matrix(lu_states.states.len());
    c.bench_function("fms_full_potential_lu_scattering_states8", |b| {
        b.iter(|| {
            black_box(fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
                calculate_full_scattering: black_box(false),
                states: black_box(&lu_states.states),
                spin_channels: black_box(2),
                global_lmax: black_box(1),
                potential_lmax: black_box(&[1]),
                representative_offsets: black_box(&lu_states.representative_offsets),
                potential_start: black_box(0),
                potential_end: black_box(0),
                free_propagator: black_box(lu_g0.view()),
                t_matrix: black_box(lu_t_full.view()),
            }))
        });
    });
}

fn sample_fms_atoms() -> [FmsAtom; 5] {
    [
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
    ]
}

fn sample_representative_atoms() -> [FmsAtom; 6] {
    [
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
    ]
}

fn sample_pair_table_atoms() -> [FmsAtom; 3] {
    [
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
    ]
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
