use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array2, Array3, Array4, Array6, ShapeBuilder};
use num_complex::Complex32;
use refeff_core::{
    Complex, FmsAtom, FmsFreePropagatorInput, FmsFreePropagatorMatrixInput, FmsRotationDirection,
    PolarizationTensorMode, SingularityFunction, StateKet, TransitionBMatrixInput, besjh, besjn,
    construct_state_kets, conv, cubic_zeros, depressed_quartic_roots, distance_between, exjlnl,
    find_self_energy_singularities, fms_free_propagator_element, fms_free_propagator_matrix,
    fms_pair_tables, fms_rotation_matrix, legendre_normalization_table, legendre_polynomials, lint,
    muffin_tin_phase_amplitude, pair_polar_angles, polarization_tensor, qsortd_order_1based,
    quadratic_zeros, rehr_albers_polynomials, rehr_albers_z_axis_propagator, somm2,
    sort_atoms_by_radius, sort_representative_atoms, spherical_harmonics,
    spin_orbit_coupling_tables, terp, terpc, transition_b_matrix, trap, wigner_rotation, x_log_x,
};

fn bench_angular_tables(c: &mut Criterion) {
    c.bench_function("build_legendre_xnlm_lmax8", |b| {
        b.iter(|| black_box(legendre_normalization_table(black_box(8))));
    });
    c.bench_function("build_spin_orbit_tables_lmax8", |b| {
        b.iter(|| black_box(spin_orbit_coupling_tables(black_box(8))));
    });
    c.bench_function("build_legendre_polynomials_lmax32", |b| {
        b.iter(|| black_box(legendre_polynomials(black_box(0.25), black_box(32))));
    });
    c.bench_function("wigner_rotation_half_integer", |b| {
        b.iter(|| {
            black_box(wigner_rotation(
                black_box(0.7),
                black_box(3),
                black_box(1),
                black_box(-1),
                black_box(2),
            ))
        });
    });
    c.bench_function("spherical_harmonics_l8", |b| {
        b.iter(|| {
            black_box(spherical_harmonics(
                black_box([1.0, 2.0, 3.0]),
                black_box(8),
            ))
        });
    });
    c.bench_function("polarization_tensor_cartesian", |b| {
        b.iter(|| {
            black_box(polarization_tensor(
                black_box(5),
                black_box(PolarizationTensorMode::Cartesian),
            ))
        });
    });
    c.bench_function("transition_b_matrix_l3", |b| {
        b.iter(|| {
            black_box(transition_b_matrix(black_box(TransitionBMatrixInput {
                lmax: 3,
                initial_kappa: -1,
                polarization: 1,
                polarization_tensor: sample_polarization_tensor(),
                multipole: 2,
                trace_orbital: false,
                spin: 1,
                spin_channels: 1,
                spin_vector_angle: 0.3,
            })))
        });
    });
}

fn sample_polarization_tensor() -> [[Complex; 3]; 3] {
    [
        [
            Complex::new(0.20, -0.05),
            Complex::new(-0.10, 0.04),
            Complex::new(0.03, 0.02),
        ],
        [
            Complex::new(0.11, -0.07),
            Complex::new(0.50, 0.00),
            Complex::new(-0.08, 0.09),
        ],
        [
            Complex::new(0.06, 0.01),
            Complex::new(0.13, -0.02),
            Complex::new(0.17, 0.03),
        ],
    ]
}

fn bench_state_kets(c: &mut Criterion) {
    let atom_potentials = vec![0, 1, 1, 2, 2, 2, 1, 0, 3, 3, 2, 1];
    let potential_lmax = vec![0, 2, 3, 1];

    c.bench_function("construct_state_kets_small_cluster", |b| {
        b.iter(|| {
            black_box(construct_state_kets(
                black_box(2),
                black_box(&atom_potentials),
                black_box(&potential_lmax),
                black_box(3),
            ))
        });
    });
}

fn bench_interpolation(c: &mut Criterion) {
    let xs: Vec<_> = (0..128).map(|index| index as f64 * 0.05).collect();
    let ys: Vec<_> = xs
        .iter()
        .map(|&x| (x * x * x) - (0.5 * x * x) + (2.0 * x) + 1.0)
        .collect();
    let complex_ys: Vec<_> = xs.iter().map(|&x| Complex::new(x.sin(), x.cos())).collect();

    c.bench_function("terp_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terp(
                black_box(&xs),
                black_box(&ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("terpc_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terpc(
                black_box(&xs),
                black_box(&complex_ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("lint_128_points", |b| {
        b.iter(|| black_box(lint(black_box(&xs), black_box(&ys), black_box(2.75))));
    });
}

fn bench_quadrature(c: &mut Criterion) {
    let xs: Vec<_> = (0..1024).map(|index| index as f64 * 0.01).collect();
    let ys: Vec<_> = xs.iter().map(|&x| x.sin() * x.exp()).collect();
    c.bench_function("trap_1024_points", |b| {
        b.iter(|| black_box(trap(black_box(&xs), black_box(&ys))));
    });

    let radii: Vec<_> = (0..128)
        .map(|index| (-8.8 + index as f64 * 0.05).exp())
        .collect();
    let values: Vec<_> = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| radius * (1.0 + index as f64 * 0.001))
        .collect();
    let rnrm = radii[100] * 0.02_f64.exp();
    c.bench_function("somm2_128_points", |b| {
        b.iter(|| {
            black_box(somm2(
                black_box(&radii),
                black_box(&values),
                black_box(0.05),
                black_box(0.5),
                black_box(rnrm),
                black_box(0),
            ))
        });
    });
}

fn bench_bessel(c: &mut Criterion) {
    c.bench_function("besjn_medium_l17", |b| {
        b.iter(|| black_box(besjn(black_box(Complex::new(3.5, 0.4)), black_box(17))));
    });
    c.bench_function("besjh_large_l8", |b| {
        b.iter(|| black_box(besjh(black_box(Complex::new(12.0, 0.5)), black_box(8))));
    });
    c.bench_function("exjlnl_l9", |b| {
        b.iter(|| black_box(exjlnl(black_box(Complex::new(6.1, 0.8)), black_box(9))));
    });
}

fn bench_convolution(c: &mut Criterion) {
    let omega: Vec<_> = (0..128).map(|index| -5.0 + index as f64 * 0.1).collect();
    let spectrum: Vec<_> = omega
        .iter()
        .map(|&energy| Complex::new((energy * 0.7).sin(), (energy * 0.4).cos()))
        .collect();

    c.bench_function("conv_128_points", |b| {
        b.iter(|| {
            black_box(conv(
                black_box(&omega),
                black_box(&spectrum),
                black_box(0.2),
            ))
        });
    });
}

fn bench_fms(c: &mut Criterion) {
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

    let pair_atoms = sample_pair_table_atoms();
    let free_wave_number = Complex32::new(1.2, 0.3);
    let Ok(pair_tables) = fms_pair_tables(2, free_wave_number, &pair_atoms) else {
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

fn bench_scalar_helpers(c: &mut Criterion) {
    c.bench_function("distance_between", |b| {
        b.iter(|| {
            black_box(distance_between(
                black_box([1.0, -2.0, 0.5]),
                black_box([-3.0, 4.0, 2.5]),
            ))
        });
    });
    c.bench_function("x_log_x", |b| {
        b.iter(|| black_box(x_log_x(black_box(2.5))));
    });
    c.bench_function("muffin_tin_phase_amplitude", |b| {
        b.iter(|| {
            black_box(muffin_tin_phase_amplitude(
                black_box(1.7),
                black_box(Complex::new(0.8, 0.2)),
                black_box(Complex::new(-0.3, 0.4)),
                black_box(Complex::new(1.1, 0.15)),
                black_box(Complex::new(0.9, -0.1)),
                black_box(Complex::new(-0.2, 0.7)),
                black_box(Complex::new(0.4, 0.3)),
                black_box(Complex::new(-0.6, 0.25)),
                black_box(-2),
            ))
        });
    });
    c.bench_function("depressed_quartic_roots", |b| {
        b.iter(|| {
            black_box(depressed_quartic_roots(black_box([
                Complex::new(0.75, -0.2),
                Complex::new(-1.5, 0.6),
                Complex::new(0.3, 0.4),
                Complex::new(2.2, -0.7),
            ])))
        });
    });
    c.bench_function("quadratic_zeros", |b| {
        b.iter(|| {
            black_box(quadratic_zeros(black_box([
                Complex::new(1.0, 0.5),
                Complex::new(-2.0, 1.0),
                Complex::new(0.25, -0.75),
            ])))
        });
    });
    c.bench_function("cubic_zeros", |b| {
        b.iter(|| {
            black_box(cubic_zeros(black_box([
                Complex::new(0.75, -0.2),
                Complex::new(-1.5, 0.6),
                Complex::new(0.3, 0.4),
                Complex::new(2.2, -0.7),
            ])))
        });
    });
    c.bench_function("find_self_energy_singularities", |b| {
        b.iter(|| {
            black_box(find_self_energy_singularities(
                black_box([Complex::new(-2.0, 0.0), Complex::new(2.0, 0.0)]),
                black_box([0.35, 0.02, 0.8, 0.0]),
                black_box([Complex::new(0.7, 0.0), Complex::new(0.0, 0.0)]),
                black_box(SingularityFunction::First),
            ))
        });
    });
}

fn bench_sort_helpers(c: &mut Criterion) {
    let values: Vec<_> = (0..256)
        .map(|index| ((index * 37) % 256) as f64 - 128.0)
        .collect();
    c.bench_function("qsortd_order_256", |b| {
        b.iter(|| black_box(qsortd_order_1based(black_box(&values))));
    });
}

criterion_group!(
    benches,
    bench_angular_tables,
    bench_state_kets,
    bench_interpolation,
    bench_quadrature,
    bench_bessel,
    bench_convolution,
    bench_fms,
    bench_scalar_helpers,
    bench_sort_helpers
);
criterion_main!(benches);
