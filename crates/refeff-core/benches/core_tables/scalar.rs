use super::*;

pub(super) fn bench_scalar_helpers(c: &mut Criterion) {
    c.bench_function("nuclear_mass", |b| {
        b.iter(|| black_box(nuclear_mass(black_box(92))));
    });
    c.bench_function("atom_nucdev_finite_nucleus", |b| {
        b.iter(|| {
            black_box(atomic_nuclear_potential(black_box(
                AtomicNuclearPotentialInput {
                    nuclear_charge: 92.0,
                    step: 0.05,
                    requested_nucleus_index: -11,
                    radial_count: 251,
                    coefficient_count: 10,
                    first_radius_times_charge: 92.0 * (-8.8_f64).exp(),
                },
            )))
        });
    });
    let dsordf_radial_count = 11;
    let dsordf_orbital_count = 3;
    let dsordf_coefficient_count = 6;
    let dsordf_radii =
        Array1::from_shape_fn(dsordf_radial_count, |row| (-4.2 + 0.05 * row as f64).exp());
    let dsordf_active_lengths = [9, 11, 7];
    let dsordf_powers = [0.21, 0.30, 0.39];
    let dsordf_large =
        Array2::from_shape_fn((dsordf_radial_count, dsordf_orbital_count), |(row, col)| {
            let radial = (row + 1) as f64;
            let orbital = (col + 1) as f64;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        });
    let dsordf_small =
        Array2::from_shape_fn((dsordf_radial_count, dsordf_orbital_count), |(row, col)| {
            let radial = (row + 1) as f64;
            let orbital = (col + 1) as f64;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        });
    let dsordf_large_coefficients = Array2::from_shape_fn(
        (dsordf_coefficient_count, dsordf_orbital_count),
        |(row, col)| 0.08 * (row + 1) as f64 + 0.015 * (col + 1) as f64,
    );
    let dsordf_small_coefficients = Array2::from_shape_fn(
        (dsordf_coefficient_count, dsordf_orbital_count),
        |(row, col)| -0.02 * (row + 1) as f64 + 0.01 * (col + 1) as f64,
    );
    let dsordf_derivative_large = Array1::from_shape_fn(dsordf_radial_count, |row| {
        let radial = (row + 1) as f64;
        0.015 * radial - 0.00007 * radial * radial
    });
    let dsordf_derivative_small = Array1::from_shape_fn(dsordf_radial_count, |row| {
        let radial = (row + 1) as f64;
        -0.004 * radial + 0.00013 * radial * radial
    });
    let dsordf_derivative_large_coefficients =
        Array1::from_shape_fn(dsordf_coefficient_count, |row| {
            0.05 * (row + 1) as f64 - 0.003
        });
    let dsordf_derivative_small_coefficients =
        Array1::from_shape_fn(dsordf_coefficient_count, |row| {
            -0.015 * (row + 1) as f64 + 0.004
        });
    c.bench_function("atom_dsordf_derivative_projection", |b| {
        b.iter(|| {
            black_box(atomic_differential_integral(black_box(
                AtomicDifferentialIntegralInput {
                    kind: AtomicDifferentialIntegralKind::DerivativeProjection {
                        large_orbital_1based: 2,
                        small_orbital_1based: 3,
                    },
                    power: 0,
                    origin_power: 0.45,
                    step: 0.05,
                    radii: dsordf_radii.view(),
                    active_lengths: &dsordf_active_lengths,
                    orbital_powers: &dsordf_powers,
                    large_components: dsordf_large.view(),
                    small_components: dsordf_small.view(),
                    large_coefficients: dsordf_large_coefficients.view(),
                    small_coefficients: dsordf_small_coefficients.view(),
                    derivative_large: dsordf_derivative_large.view(),
                    derivative_small: dsordf_derivative_small.view(),
                    derivative_large_coefficients: dsordf_derivative_large_coefficients.view(),
                    derivative_small_coefficients: dsordf_derivative_small_coefficients.view(),
                },
            )))
        });
    });
    let vlda_occupations = [2.0, 1.6, 0.7];
    let vlda_valence_occupations = [1.0, 0.4, 0.2];
    let vlda_initial_potential =
        Array1::from_shape_fn(dsordf_radial_count, |row| 0.0001 * (row + 1) as f64);
    let vlda_initial_coefficients =
        Array1::from_shape_fn(dsordf_coefficient_count, |row| 0.01 * (row + 1) as f64);
    let vlda_initial_energy =
        Array1::from_shape_fn(dsordf_radial_count, |row| 0.002 * (row + 1) as f64);
    c.bench_function("atom_vlda_local_density_potential", |b| {
        b.iter(|| {
            black_box(atomic_local_density_potential(black_box(
                AtomicLocalDensityPotentialInput {
                    mode: AtomicLocalDensityExchangeMode::CoreDensitySeparated,
                    accumulate_energy_density: true,
                    speed_of_light: 137.035_999,
                    radii: dsordf_radii.view(),
                    active_lengths: &dsordf_active_lengths,
                    occupations: &vlda_occupations,
                    valence_occupations: &vlda_valence_occupations,
                    large_components: dsordf_large.view(),
                    small_components: dsordf_small.view(),
                    initial_potential: vlda_initial_potential.view(),
                    initial_development_coefficients: vlda_initial_coefficients.view(),
                    initial_energy_density: vlda_initial_energy.view(),
                },
            )))
        });
    });
    let screen_radii = screen_radial_grid(0.05, 8.8, 251).unwrap_or_else(|_| Array1::zeros(251));
    let screen_positions_angstrom = Array2::from_shape_fn((128, 3).f(), |(row, column)| {
        (row as f64 * 0.05) - column as f64 * 0.25
    });
    c.bench_function("screen_rdgeom_atomic_units_128_atoms", |b| {
        b.iter(|| {
            black_box(screen_rdgeom_atomic_units(black_box(
                ScreenRdgeomAtomicUnitsInput {
                    atom_positions_angstrom: screen_positions_angstrom.view(),
                    rfms2_angstrom: 5.0,
                    direct_radius_angstrom: 10.0,
                    min_real_energy_ev: -40.0,
                    max_real_energy_ev: 0.0,
                    max_imaginary_energy_ev: 2.0,
                    screen_rfms_angstrom: 4.0,
                    min_imaginary_energy_ev: 0.001,
                    max_l: 4,
                    angular_capacity_lx: 2,
                },
            )))
        });
    });
    c.bench_function("screen_radial_bounds", |b| {
        b.iter(|| {
            black_box(screen_radial_bounds(black_box(ScreenRadialBoundsInput {
                x0: 8.8,
                dx: 0.05,
                muffin_tin_radius: 0.5,
                norman_radius: 1.2,
                tail_extension: 3,
                radial_capacity: 251,
                response_capacity: 251,
            })))
        });
    });
    c.bench_function("screen_getph_radial_bounds", |b| {
        b.iter(|| {
            black_box(screen_getph_radial_bounds(black_box(
                ScreenGetphRadialBoundsInput {
                    x0: 8.8,
                    dx: 0.05,
                    muffin_tin_radius: 0.5,
                    norman_radius: 1.2,
                    radial_capacity: 251,
                },
            )))
        });
    });
    c.bench_function("screen_energy_state", |b| {
        b.iter(|| {
            black_box(screen_energy_state(black_box(ScreenEnergyStateInput {
                energy: Complex::new(0.4, 0.5),
                reference_energy: Complex::new(0.1, 0.05),
                muffin_tin_radius: 1.7,
                exchange_selector: 7,
            })))
        });
    });
    c.bench_function("screen_getph_lmax", |b| {
        b.iter(|| black_box(screen_getph_lmax(black_box(29), black_box(5), black_box(3))));
    });
    c.bench_function("screen_solution_normalization", |b| {
        b.iter(|| {
            black_box(screen_solution_normalization(black_box(
                ScreenSolutionNormalizationInput {
                    wave_number: Complex::new(0.4, 0.5),
                    phase_amplitude: Complex::new(1.25, -0.4),
                },
            )))
        });
    });
    c.bench_function("screen_irregular_initial_condition", |b| {
        b.iter(|| {
            black_box(screen_irregular_initial_condition(black_box(
                ScreenIrregularInitialConditionInput {
                    muffin_tin_radius: 1.7,
                    phase_shift: Complex::new(0.2, -0.1),
                    wave_number: Complex::new(0.4, 0.5),
                    bessel_j_l: Complex::new(0.8, 0.1),
                    neumann_l: Complex::new(-0.3, 0.05),
                    bessel_j_l_plus_1: Complex::new(0.25, -0.03),
                    neumann_l_plus_1: Complex::new(-0.6, 0.2),
                    hankel_l: Complex::new(0.1, 0.7),
                    hankel_l_plus_1: Complex::new(-0.2, 0.3),
                    use_hankel_boundary: true,
                },
            )))
        });
    });
    c.bench_function("screen_irregular_wronskian_scale", |b| {
        b.iter(|| {
            black_box(screen_irregular_wronskian_scale(black_box(
                ScreenIrregularWronskianScaleInput {
                    phase_shift: Complex::new(0.2, -0.1),
                    wave_number: Complex::new(0.4, 0.5),
                    regular_large_at_match: Complex::new(0.3, 0.2),
                    regular_small_at_match: Complex::new(-0.01, 0.04),
                    irregular_large_at_match: Complex::new(0.7, -0.2),
                    irregular_small_at_match: Complex::new(0.02, 0.03),
                },
            )))
        });
    });
    c.bench_function("screen_exact_radial_continuation", |b| {
        b.iter(|| {
            black_box(screen_exact_radial_continuation(black_box(
                ScreenExactRadialContinuationInput {
                    radius: 2.0,
                    phase_shift: Complex::new(0.2, -0.1),
                    wave_number: Complex::new(0.4, 0.5),
                    bessel_j_l: Complex::new(0.6, 0.2),
                    neumann_l: Complex::new(-0.4, 0.1),
                    bessel_j_l_plus_1: Complex::new(0.3, 0.05),
                    neumann_l_plus_1: Complex::new(-0.2, 0.2),
                    hankel_l: Complex::new(0.1, 0.7),
                    hankel_l_plus_1: Complex::new(-0.2, 0.3),
                },
            )))
        });
    });
    let screen_phase_total = Array1::from_shape_fn(251, |row| -3.0 + 0.02 * row as f64);
    let screen_phase_valence = Array1::from_shape_fn(251, |row| -2.5 + 0.015 * row as f64);
    c.bench_function("screen_phase_potential_reference_shift_251", |b| {
        b.iter(|| {
            black_box(screen_phase_potential_reference_shift(black_box(
                ScreenPhasePotentialInput {
                    total_potential: screen_phase_total.view(),
                    valence_potential: screen_phase_valence.view(),
                    muffin_tin_next_index_1based: 165,
                    exchange_selector: 5,
                },
            )))
        });
    });
    let screen_density = Array1::from_shape_fn(screen_radii.len(), |row| {
        0.2 * (-0.04 * row as f64).exp() + 0.001
    });
    let screen_local_kernel = screen_lda_exchange_correlation_kernel(
        screen_radii.as_slice().unwrap_or(&[]),
        screen_density.as_slice().unwrap_or(&[]),
        0,
        screen_radii.len(),
    )
    .unwrap_or_else(|_| Array1::zeros(screen_radii.len()));
    let screen_large_component = Array1::from_shape_fn(screen_radii.len(), |row| {
        let x = row as f64 * 0.015;
        (-x).exp()
    });
    let screen_small_component = Array1::from_shape_fn(screen_radii.len(), |row| {
        0.01 * (-(row as f64) * 0.02).exp()
    });
    c.bench_function("screen_coulomb_kernel_matrix_251", |b| {
        b.iter(|| {
            black_box(screen_coulomb_kernel_matrix(
                black_box(screen_radii.as_slice().unwrap_or(&[])),
                black_box(screen_radii.len()),
                black_box(Some(screen_local_kernel.as_slice().unwrap_or(&[]))),
            ))
        });
    });
    c.bench_function("screen_bare_core_hole_potential_251", |b| {
        b.iter(|| {
            black_box(screen_bare_core_hole_potential(
                black_box(screen_radii.as_slice().unwrap_or(&[])),
                black_box(screen_large_component.as_slice().unwrap_or(&[])),
                black_box(screen_small_component.as_slice().unwrap_or(&[])),
                black_box(0.05),
                black_box(screen_radii.len()),
            ))
        });
    });
    let screen_shell_weights =
        Array1::from_shape_fn(screen_radii.len(), |row| 0.001 * (-0.02 * row as f64).exp());
    c.bench_function("screen_radial_coulomb_potential_251", |b| {
        b.iter(|| {
            black_box(screen_radial_coulomb_potential(
                black_box(screen_radii.as_slice().unwrap_or(&[])),
                black_box(screen_shell_weights.as_slice().unwrap_or(&[])),
                black_box(screen_radii.len()),
            ))
        });
    });
    let screen_crpa_density =
        Array1::from_shape_fn(screen_radii.len(), |row| 0.2 * (-0.025 * row as f64).exp());
    let screen_crpa_window = ScreenCrpaProjectionWindow {
        inner_radius: screen_radii.get(50).copied().unwrap_or(0.01),
        outer_radius: screen_radii.get(180).copied().unwrap_or(1.0),
    };
    c.bench_function("screen_crpa_density_weights_251", |b| {
        b.iter(|| {
            black_box(screen_crpa_density_weights(
                black_box(screen_radii.as_slice().unwrap_or(&[])),
                black_box(screen_crpa_density.as_slice().unwrap_or(&[])),
                black_box(0.05),
                black_box(screen_radii.len()),
                black_box(245),
                black_box(Some(screen_crpa_window)),
            ))
        });
    });
    let screen_radii_slice = screen_radii.as_slice().unwrap_or(&[]);
    let screen_response_order = screen_radii_slice.len().min(64);
    let screen_response_kernel =
        screen_coulomb_kernel_matrix(screen_radii_slice, screen_response_order, None)
            .unwrap_or_else(|_| Array2::zeros((screen_response_order, screen_response_order).f()));
    let screen_response_susceptibility = Array2::from_shape_fn(
        (screen_response_order, screen_response_order).f(),
        |(row, col)| {
            let scaled_index = 1.0 + (row + col) as f64;
            Complex::new(0.0, 1.0e-9 / scaled_index)
        },
    );
    let screen_response_bare = Array1::from_shape_fn(screen_response_order, |row| {
        0.1 * (-0.02 * row as f64).exp()
    });
    let screen_response_total_density = Array1::from_shape_fn(screen_response_order, |row| {
        0.05 * (-0.01 * row as f64).exp()
    });
    let screen_response_orbital_density = Array1::from_shape_fn(screen_response_order, |row| {
        0.02 * (-0.015 * row as f64).exp()
    });
    let screen_response_energies = Array1::from_shape_fn(8, |index| {
        Complex::new(-0.4 + 0.12 * index as f64, 0.05 + 0.03 * index as f64)
    });
    let screen_response_delta = screen_energy_integration_delta(screen_response_energies.view(), 3)
        .unwrap_or_else(|_| Complex::new(0.12, 0.03));
    let screen_response_step =
        screen_response_susceptibility.mapv(|value| value * Complex::new(0.8, -0.15));
    let screen_response_integrated = screen_integrate_response_step(
        screen_response_susceptibility.view(),
        screen_response_step.view(),
        screen_response_delta,
        screen_response_order,
    )
    .unwrap_or_else(|_| Array2::zeros((screen_response_order, screen_response_order).f()));
    let screen_crpa_regular = Array1::from_shape_fn(screen_response_order, |row| {
        let radius = screen_radii_slice.get(row).copied().unwrap_or(1.0);
        Complex::new((-0.04 * row as f64).exp(), 0.01 * radius)
    });
    let screen_crpa_irregular = Array1::from_shape_fn(screen_response_order, |row| {
        let scaled = 1.0 / (1.0 + row as f64);
        Complex::new(0.5 * scaled, -0.2 * scaled)
    });
    let screen_fms_scattering = Array2::from_shape_fn((9, 9).f(), |(row, column)| {
        if row == column {
            let state = row as f32 + 1.0;
            Complex32::new(0.01 * state, -0.004 * state)
        } else {
            Complex32::new(0.0, 0.0)
        }
    });
    c.bench_function("screen_atomic_response_slice_64", |b| {
        b.iter(|| {
            black_box(screen_atomic_response_slice(
                black_box(screen_radii_slice),
                black_box(screen_crpa_regular.view()),
                black_box(screen_crpa_irregular.view()),
                black_box(Complex::new(0.7, 0.3)),
                black_box(0.05),
                black_box(2),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_fms_response_slice_64", |b| {
        b.iter(|| {
            black_box(screen_fms_response_slice(black_box(
                ScreenFmsResponseSliceInput {
                    radii: screen_radii_slice,
                    regular_solution: screen_crpa_regular.view(),
                    irregular_solution: screen_crpa_irregular.view(),
                    cluster_green: Complex::new(0.1, 0.2),
                    wave_number: Complex::new(0.7, 0.3),
                    dx: 0.05,
                    angular_momentum: 2,
                    active_count: screen_response_order,
                    fms_count: screen_response_order.saturating_sub(4).max(1),
                },
            )))
        });
    });
    c.bench_function("screen_fms_cluster_green_trace_l2", |b| {
        b.iter(|| {
            black_box(screen_fms_cluster_green_trace(
                black_box(screen_fms_scattering.view()),
                black_box(Complex::new(0.2, 0.05)),
                black_box(2),
            ))
        });
    });
    c.bench_function("screen_crpa_response_slice_64", |b| {
        b.iter(|| {
            black_box(screen_crpa_response_slice(black_box(
                ScreenCrpaResponseSliceInput {
                    radii: screen_radii_slice,
                    regular_solution: screen_crpa_regular.view(),
                    irregular_solution: screen_crpa_irregular.view(),
                    cluster_green: Complex::new(0.1, 0.2),
                    wave_number: Complex::new(0.7, 0.3),
                    dx: 0.05,
                    angular_momentum: 2,
                    crpa_angular_momentum: 2,
                    projection_window: Some(screen_crpa_window),
                    active_count: screen_response_order,
                },
            )))
        });
    });
    c.bench_function("screen_energy_integration_delta_8", |b| {
        b.iter(|| {
            black_box(screen_energy_integration_delta(
                black_box(screen_response_energies.view()),
                black_box(3),
            ))
        });
    });
    c.bench_function("screen_response_system_matrix_64", |b| {
        b.iter(|| {
            black_box(screen_response_system_matrix(
                black_box(screen_response_kernel.view()),
                black_box(screen_response_susceptibility.view()),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_solve_response_potential_64", |b| {
        b.iter(|| {
            black_box(screen_solve_response_potential(
                black_box(screen_response_kernel.view()),
                black_box(screen_response_susceptibility.view()),
                black_box(screen_response_bare.view()),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_integrate_response_step_64", |b| {
        b.iter(|| {
            black_box(screen_integrate_response_step(
                black_box(screen_response_susceptibility.view()),
                black_box(screen_response_step.view()),
                black_box(screen_response_delta),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_symmetrize_response_upper_64", |b| {
        b.iter(|| {
            black_box(screen_symmetrize_response_upper(
                black_box(screen_response_integrated.view()),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_crpa_orbital_density_64", |b| {
        b.iter(|| {
            black_box(screen_crpa_orbital_density(
                black_box(screen_crpa_regular.view()),
                black_box(screen_crpa_irregular.view()),
                black_box(Complex::new(0.1, 0.2)),
                black_box(Complex::new(0.7, 0.3)),
                black_box(2),
                black_box(screen_response_order),
            ))
        });
    });
    c.bench_function("screen_crpa_hubbard_summary_64", |b| {
        b.iter(|| {
            black_box(screen_crpa_hubbard_summary(
                black_box(screen_radii_slice),
                black_box(screen_response_bare.as_slice().unwrap_or(&[])),
                black_box(screen_shell_weights.as_slice().unwrap_or(&[])),
                black_box(screen_response_total_density.as_slice().unwrap_or(&[])),
                black_box(screen_response_orbital_density.as_slice().unwrap_or(&[])),
                black_box(0.05),
                black_box(screen_response_order),
            ))
        });
    });
    let yzkrdf_radial_count = 13;
    let yzkrdf_radii =
        Array1::from_shape_fn(yzkrdf_radial_count, |row| (-4.2 + 0.05 * row as f64).exp());
    let yzkrdf_large =
        Array2::from_shape_fn((yzkrdf_radial_count, dsordf_orbital_count), |(row, col)| {
            let radial = (row + 1) as f64;
            let orbital = (col + 1) as f64;
            0.02 * orbital + 0.0015 * radial + 0.00003 * radial * orbital
        });
    let yzkrdf_small =
        Array2::from_shape_fn((yzkrdf_radial_count, dsordf_orbital_count), |(row, col)| {
            let radial = (row + 1) as f64;
            let orbital = (col + 1) as f64;
            -0.006 * orbital + 0.0008 * radial - 0.00001 * radial * orbital
        });
    c.bench_function("atom_yzkrdf_overlap_source", |b| {
        b.iter(|| {
            black_box(atomic_yk_zk_exchange(black_box(AtomicYkZkExchangeInput {
                left_orbital_1based: 1,
                right_orbital_1based: 2,
                large_small: false,
                angular_momentum: 2,
                step: 0.05,
                radii: yzkrdf_radii.view(),
                active_lengths: &dsordf_active_lengths,
                orbital_powers: &dsordf_powers,
                large_components: yzkrdf_large.view(),
                small_components: yzkrdf_small.view(),
                large_coefficients: dsordf_large_coefficients.view(),
                small_coefficients: dsordf_small_coefficients.view(),
            })))
        });
    });
    let potrdf_kappas = [-1, 1, 1];
    let potrdf_occupations = [2.0, 1.6, 0.7];
    let potrdf_shell_markers = [-1, 1, 1];
    let potrdf_origin_scales = [1.05, 0.95, 1.10];
    let potrdf_coulomb = Array3::from_shape_fn(
        (dsordf_orbital_count, dsordf_orbital_count, 5),
        |(left, right, rank)| {
            0.015 * (left + 1) as f64 + 0.011 * (right + 1) as f64 + 0.003 * rank as f64
        },
    );
    let potrdf_lagrange = Array1::from_shape_fn(3, |row| 0.012 * (row + 1) as f64);
    let potrdf_nuclear_potential =
        Array1::from_shape_fn(yzkrdf_radial_count, |row| -0.2 + 0.001 * (row + 1) as f64);
    let potrdf_nuclear_coefficients =
        Array1::from_shape_fn(dsordf_coefficient_count, |row| -0.03 * (row + 1) as f64);
    c.bench_function("atom_potrdf_orbital_potential", |b| {
        b.iter(|| {
            black_box(atomic_orbital_potential(black_box(
                AtomicOrbitalPotentialInput {
                    active_orbital_1based: 2,
                    include_exchange: true,
                    include_lagrange: true,
                    self_consistent_count: 3,
                    speed_of_light: 137.035_999,
                    step: 0.05,
                    radii: yzkrdf_radii.view(),
                    active_lengths: &dsordf_active_lengths,
                    kappas: &potrdf_kappas,
                    orbital_powers: &dsordf_powers,
                    occupations: &potrdf_occupations,
                    shell_markers: &potrdf_shell_markers,
                    origin_scales: &potrdf_origin_scales,
                    coulomb_coefficients: potrdf_coulomb.view(),
                    lagrange_parameters: potrdf_lagrange.view(),
                    nuclear_potential: potrdf_nuclear_potential.view(),
                    nuclear_development_coefficients: potrdf_nuclear_coefficients.view(),
                    large_components: yzkrdf_large.view(),
                    small_components: yzkrdf_small.view(),
                    large_coefficients: dsordf_large_coefficients.view(),
                    small_coefficients: dsordf_small_coefficients.view(),
                },
            )))
        });
    });
    let fdrirk_kappas = [-1, 1, -2];
    c.bench_function("atom_fdrirk_radial_integral", |b| {
        b.iter(|| {
            black_box(atomic_radial_integral(black_box(
                AtomicRadialIntegralInput {
                    request: AtomicRadialIntegralRequest {
                        first_left: 1,
                        first_right: 2,
                        second_left: 1,
                        second_right: 3,
                        rank: 2,
                    },
                    large_small: false,
                    previous_first_factor: None,
                    kappas: &fdrirk_kappas,
                    step: 0.05,
                    radii: yzkrdf_radii.view(),
                    active_lengths: &dsordf_active_lengths,
                    orbital_powers: &dsordf_powers,
                    large_components: yzkrdf_large.view(),
                    small_components: yzkrdf_small.view(),
                    large_coefficients: dsordf_large_coefficients.view(),
                    small_coefficients: dsordf_small_coefficients.view(),
                },
            )))
        });
    });
    let yzkteg_active_len = 13;
    let yzkteg_coefficient_count = 6;
    let yzkteg_source = Array1::from_shape_fn(yzkteg_active_len, |row| {
        let row = (row + 1) as f64;
        0.017 * row + 0.0008 * row * row - 0.00001 * row * row * row
    });
    let yzkteg_coefficients = Array1::from_shape_fn(yzkteg_coefficient_count, |row| {
        let row = (row + 1) as f64;
        0.04 * row - 0.0015 * row * row
    });
    let yzkteg_radii =
        Array1::from_shape_fn(yzkteg_active_len, |row| (-4.2 + 0.05 * row as f64).exp());
    c.bench_function("atom_yzkteg_transform", |b| {
        b.iter(|| {
            black_box(atomic_yk_zk_transform(black_box(
                AtomicYkZkTransformInput {
                    source: yzkteg_source.view(),
                    source_coefficients: yzkteg_coefficients.view(),
                    radii: yzkteg_radii.view(),
                    initial_power: 0.65,
                    step: 0.05,
                    angular_momentum: 2,
                    coefficient_count: yzkteg_coefficient_count,
                    source_len: 9,
                    active_len: yzkteg_active_len,
                },
            )))
        });
    });
    c.bench_function("atom_yzkrdf_prepared_source", |b| {
        b.iter(|| {
            black_box(atomic_yk_zk_prepared_source(black_box(
                AtomicYkZkPreparedSourceInput {
                    source: yzkteg_source.view(),
                    source_coefficients: yzkteg_coefficients.view(),
                    radii: yzkteg_radii.view(),
                    step: 0.05,
                    angular_momentum: 2,
                    coefficient_count: yzkteg_coefficient_count,
                    source_len: 9,
                    active_len: yzkteg_active_len,
                },
            )))
        });
    });
    let elam_components = [29, 8, 79];
    c.bench_function("elam_edge_lookup", |b| {
        b.iter(|| {
            black_box((
                elam_edge_energy_hartree(black_box(29), black_box(1)),
                previous_elam_edge_hartree(black_box(35.0), black_box(&elam_components)),
                next_elam_edge_hartree(black_box(35.0), black_box(&elam_components)),
            ))
        });
    });

    let left = (1..=10)
        .map(|index| 0.1 * index as f64 + 0.03)
        .collect::<Vec<_>>();
    let right = (1..=10)
        .map(|index| -0.04 * index as f64 + 0.25)
        .collect::<Vec<_>>();
    c.bench_function("atom_aprdev_product_l7", |b| {
        b.iter(|| {
            black_box(atomic_polynomial_product_coefficient(
                black_box(&left),
                black_box(&right),
                black_box(7),
            ))
        });
    });
    c.bench_function("atom_cofcon_mix", |b| {
        b.iter(|| {
            black_box(atomic_convergence_mix(
                black_box(0.5),
                black_box(0.3),
                black_box(0.2),
            ))
        });
    });
    c.bench_function("atom_dentfa_density", |b| {
        b.iter(|| {
            black_box(thomas_fermi_density_potential(
                black_box(0.45),
                black_box(29.0),
                black_box(-1.0),
            ))
        });
    });

    let mut occupations = vec![0.0; 41];
    let mut kappas = vec![1; 41];
    occupations[1] = 1.5;
    occupations[4] = 3.0;
    kappas[1] = -1;
    kappas[4] = -3;
    c.bench_function("atom_fdmocc_same_orbital", |b| {
        b.iter(|| {
            black_box(atomic_occupation_product(
                black_box(&occupations),
                black_box(&kappas),
                black_box(4),
                black_box(4),
            ))
        });
    });

    let muatco_kappas = [-1, 1, -2, 2, -3];
    let muatco_occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
    let muatco_valence = [0.0, 0.5, 0.0, 0.25, 0.0];
    c.bench_function("atom_muatco_coefficients_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_coulomb_coefficients(black_box(
                AtomicCoulombCoefficientInput {
                    kappas: &muatco_kappas,
                    occupations: &muatco_occupations,
                    valence_occupations: &muatco_valence,
                },
            )))
        });
    });
    let inmuat_principal = [2, 3, 1];
    let inmuat_kappas = [1, 1, -1];
    let inmuat_occupations = [0.4, 1.6, 2.0];
    c.bench_function("atom_inmuat_orbital_initialization", |b| {
        b.iter(|| {
            black_box(atomic_orbital_initialization(black_box(
                AtomicOrbitalInitializationInput {
                    atomic_number: 4,
                    ionicity: 0.0,
                    principal_quantum_numbers: &inmuat_principal,
                    kappas: &inmuat_kappas,
                    occupations: &inmuat_occupations,
                },
            )))
        });
    });
    let soldir_radii = (1..=251)
        .map(|index| (-8.8 + 0.05 * (index - 1) as f64).exp())
        .collect::<Array1<_>>();
    let soldir_large = (1..=251)
        .map(|index| 0.03 * index as f64 + 0.002 * (0.17 * index as f64).sin())
        .collect::<Array1<_>>();
    let soldir_small = (1..=251)
        .map(|index| -0.014 * index as f64 + 0.003 * (0.11 * index as f64).cos())
        .collect::<Array1<_>>();
    let soldir_large_coefficients = (1..=10)
        .map(|index| 0.021 * index as f64 - 0.0007 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_small_coefficients = (1..=10)
        .map(|index| -0.013 * index as f64 + 0.0004 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_homogeneous_large = (1..=251)
        .map(|index| 0.018 * index as f64 + 0.0007 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_homogeneous_small = (1..=251)
        .map(|index| -0.012 * index as f64 + 0.0004 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_homogeneous_large_coefficients = (1..=10)
        .map(|index| 0.012 * index as f64 + 0.0005 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_homogeneous_small_coefficients = (1..=10)
        .map(|index| -0.009 * index as f64 + 0.0003 * (index * index) as f64)
        .collect::<Array1<_>>();
    let soldir_setup_radii = Array1::from_shape_fn(7, |row| 0.08 * (0.11 * row as f64).exp());
    let soldir_setup_potential = Array1::from_shape_fn(7, |row| {
        let radius = 0.08 * (0.11 * row as f64).exp();
        -0.42 * (-0.30 * radius).exp() + 0.008 * row as f64
    });
    let soldir_setup_coefficients = Array1::from_vec(vec![-0.058_378_260_164_777, 0.0006, -0.0003]);
    c.bench_function("atom_soldir_setup_7", |b| {
        b.iter(|| {
            black_box(atomic_dirac_solver_setup(black_box(
                AtomicDiracSolverSetupInput {
                    energy: -8.0,
                    origin_power: 1.25,
                    initial_large_coefficient: 0.82,
                    initial_small_coefficient: -0.006,
                    principal_quantum_number: 2,
                    kappa: -2,
                    speed_of_light: 137.0373,
                    method: 0,
                    radii: soldir_setup_radii.view(),
                    potential: soldir_setup_potential.view(),
                    potential_coefficients: soldir_setup_coefficients.view(),
                    active_len: 7,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_inhomogeneous_seed_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_inhomogeneous_seed(black_box(
                AtomicDiracInhomogeneousSeedInput {
                    large_source: soldir_large.view(),
                    small_source: soldir_small.view(),
                    large_source_coefficients: soldir_large_coefficients.view(),
                    small_source_coefficients: soldir_small_coefficients.view(),
                    coefficient_count: 10,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_inhomogeneous_branch", |b| {
        b.iter(|| {
            black_box(atomic_dirac_inhomogeneous_branch(black_box(
                AtomicDiracInhomogeneousBranchInput {
                    requested_method: 1,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_homogeneous_seed_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_homogeneous_seed(black_box(
                AtomicDiracHomogeneousSeedInput {
                    radial_len: 251,
                    coefficient_len: 10,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_homogeneous_pass_setup", |b| {
        b.iter(|| {
            black_box(atomic_dirac_homogeneous_pass_setup(black_box(
                AtomicDiracHomogeneousPassSetupInput { method: 1 },
            )))
        });
    });
    c.bench_function("atom_soldir_entry_state", |b| {
        b.iter(|| {
            black_box(atomic_dirac_entry_state(black_box(
                AtomicDiracEntryStateInput {
                    asymptotic_large_component: -0.25,
                    method: 0,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_norm_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_normalization(black_box(
                AtomicDiracNormalizationInput {
                    radii: soldir_radii.view(),
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    large_coefficients: soldir_large_coefficients.view(),
                    small_coefficients: soldir_small_coefficients.view(),
                    method: 1,
                    step: 0.05,
                    coefficient_count: 6,
                    matching_small_component: 0.177,
                    origin_power: 0.82,
                    active_len: 11,
                    matching_index_1based: 5,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_solution_normalization_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_solution_normalization(black_box(
                AtomicDiracSolutionNormalizationInput {
                    norm: 9.499_334_208_495_336e-6,
                    initial_large_coefficient: 0.82,
                    initial_small_coefficient: -0.006,
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    large_coefficients: soldir_large_coefficients.view(),
                    small_coefficients: soldir_small_coefficients.view(),
                    coefficient_count: 6,
                    active_len: 151,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_node_count_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_node_count(black_box(
                AtomicDiracNodeCountInput {
                    large_component: soldir_large.view(),
                    matching_index_1based: 127,
                    scan_index_1based: 151,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_node_energy_search", |b| {
        b.iter(|| {
            black_box(atomic_dirac_node_energy_search(black_box(
                AtomicDiracNodeEnergySearchInput {
                    energy: -0.5,
                    node_count: 2,
                    target_node_count: 4,
                    energy_sup: -5.0,
                    energy_inf: 1.0,
                    energy_floor: -5.0,
                    energy_precision: 1.0e-7,
                    search_attempt_count: 0,
                    max_attempt_count: 50,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_matching_point_update_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_matching_point_update(black_box(
                AtomicDiracMatchingPointUpdateInput {
                    large_component: soldir_large.view(),
                    active_len: 151,
                    matching_index_1based: 127,
                    already_relocated: false,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_method1_energy_correction", |b| {
        b.iter(|| {
            black_box(atomic_dirac_method_one_energy_correction(black_box(
                AtomicDiracMethodOneEnergyCorrectionInput {
                    speed_of_light: 137.0373,
                    norm: 2.6,
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    matching_small_component: 0.052,
                    matching_index_1based: 127,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_energy_step", |b| {
        b.iter(|| {
            black_box(atomic_dirac_energy_step(black_box(
                AtomicDiracEnergyStepInput {
                    energy: -1.0,
                    correction: 0.30,
                    mismatch: 0.4,
                    energy_sup: -1.2,
                    energy_inf: -0.8,
                    mismatch_precision: 0.1,
                    zero_energy_precision: 1.0e-7,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_iteration_reset", |b| {
        b.iter(|| {
            black_box(atomic_dirac_iteration_reset(black_box(
                AtomicDiracIterationResetInput {
                    method: 2,
                    primary_matching_precision: 1.0e-5,
                    secondary_matching_precision: 2.0e-5,
                    energy_floor: -0.75,
                    reference_energy: -0.4,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_abnormal_exit_recovery", |b| {
        b.iter(|| {
            black_box(atomic_dirac_abnormal_exit_recovery(black_box(
                AtomicDiracAbnormalExitRecoveryInput {
                    requested_method: 1,
                    method: 1,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_rematch_attempt", |b| {
        b.iter(|| {
            black_box(atomic_dirac_rematch_attempt(black_box(
                AtomicDiracRematchAttemptInput {
                    mismatch: 0.4,
                    mismatch_precision: 0.1,
                    match_attempt_count: 4,
                    max_attempt_count: 50,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_shooting_pass_setup", |b| {
        b.iter(|| {
            black_box(atomic_dirac_shooting_pass_setup(black_box(
                AtomicDiracShootingPassSetupInput {
                    energy: -0.5,
                    previous_energy: -0.54,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_homogeneous_match_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_homogeneous_match(black_box(
                AtomicDiracHomogeneousMatchInput {
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    matching_large_component: 0.24,
                    active_len: 151,
                    matching_index_1based: 127,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_large_component_match_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_large_component_match(black_box(
                AtomicDiracLargeComponentMatchInput {
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    homogeneous_large_component: soldir_homogeneous_large.view(),
                    homogeneous_small_component: soldir_homogeneous_small.view(),
                    matching_large_component: 0.24,
                    active_len: 151,
                    matching_index_1based: 127,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_two_component_match_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_two_component_match(black_box(
                AtomicDiracTwoComponentMatchInput {
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    large_coefficients: soldir_large_coefficients.view(),
                    small_coefficients: soldir_small_coefficients.view(),
                    homogeneous_large_component: soldir_homogeneous_large.view(),
                    homogeneous_small_component: soldir_homogeneous_small.view(),
                    homogeneous_large_coefficients: soldir_homogeneous_large_coefficients.view(),
                    homogeneous_small_coefficients: soldir_homogeneous_small_coefficients.view(),
                    matching_large_component: 0.285,
                    matching_small_component: -0.068,
                    homogeneous_matching_large_component: 0.087,
                    homogeneous_matching_small_component: -0.047,
                    coefficient_count: 6,
                    active_len: 151,
                    matching_index_1based: 127,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_energy_disagreement_match_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_energy_disagreement_match(black_box(
                AtomicDiracEnergyDisagreementMatchInput {
                    large_derivative: soldir_large.view(),
                    small_derivative: soldir_small.view(),
                    large_derivative_coefficients: soldir_large_coefficients.view(),
                    small_derivative_coefficients: soldir_small_coefficients.view(),
                    homogeneous_large_component: soldir_homogeneous_large.view(),
                    homogeneous_small_component: soldir_homogeneous_small.view(),
                    homogeneous_large_coefficients: soldir_homogeneous_large_coefficients.view(),
                    homogeneous_small_coefficients: soldir_homogeneous_small_coefficients.view(),
                    matching_large_derivative: 0.285,
                    matching_small_derivative: -0.068,
                    homogeneous_matching_large_component: 0.087,
                    homogeneous_matching_small_component: -0.047,
                    coefficient_count: 6,
                    active_len: 151,
                    matching_index_1based: 127,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_energy_disagreement_source_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_energy_disagreement_source(black_box(
                AtomicDiracEnergyDisagreementSourceInput {
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    large_coefficients: soldir_large_coefficients.view(),
                    small_coefficients: soldir_small_coefficients.view(),
                    radii: soldir_radii.view(),
                    speed_of_light: 137.0373,
                    coefficient_count: 10,
                    active_len: 151,
                },
            )))
        });
    });
    c.bench_function("atom_soldir_energy_disagreement_correction_251", |b| {
        b.iter(|| {
            black_box(atomic_dirac_energy_disagreement_correction(black_box(
                AtomicDiracEnergyDisagreementCorrectionInput {
                    radii: soldir_radii.view(),
                    large_component: soldir_large.view(),
                    small_component: soldir_small.view(),
                    large_derivative: soldir_homogeneous_large.view(),
                    small_derivative: soldir_homogeneous_small.view(),
                    large_coefficients: soldir_large_coefficients.view(),
                    small_coefficients: soldir_small_coefficients.view(),
                    large_derivative_coefficients: soldir_homogeneous_large_coefficients.view(),
                    small_derivative_coefficients: soldir_homogeneous_small_coefficients.view(),
                    norm: 0.913,
                    step: 0.05,
                    origin_power: 0.82,
                    coefficient_count: 10,
                    active_len: 151,
                },
            )))
        });
    });
    let intdir_speed_of_light = 137.0373;
    let intdir_step = 0.05;
    let intdir_radii = Array1::from_shape_fn(251, |row| 0.03 * (intdir_step * row as f64).exp());
    let intdir_potential = Array1::from_shape_fn(251, |row| {
        let radius = 0.03 * (intdir_step * row as f64).exp();
        -0.25 * (-0.40 * radius).exp()
    });
    let intdir_potential_coefficients = Array1::from_shape_fn(10, |row| {
        if row == 0 {
            -8.0 / intdir_speed_of_light
        } else {
            0.0003 * row as f64 * (-1.0_f64).powi((row + 1) as i32)
        }
    });
    let intdir_large_source = Array1::from_shape_fn(251, |row| {
        let index = (row + 1) as f64;
        0.001 * (0.17 * index).sin() + 0.0002 * (0.03 * index).cos()
    });
    let intdir_small_source = Array1::from_shape_fn(251, |row| {
        let index = (row + 1) as f64;
        0.0007 * (0.11 * index).cos() - 0.0001 * (0.05 * index).sin()
    });
    let intdir_large_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as f64;
        0.0002 * index * (-1.0_f64).powi((row + 1) as i32)
    });
    let intdir_small_coefficients = Array1::from_shape_fn(10, |row| {
        let index = (row + 1) as f64;
        -0.00015 * index * (-1.0_f64).powi((row + 1) as i32)
    });
    c.bench_function("atom_intdir_search_151", |b| {
        b.iter(|| {
            black_box(atomic_dirac_integration(black_box(
                AtomicDiracIntegrationInput {
                    large_source: intdir_large_source.view(),
                    small_source: intdir_small_source.view(),
                    large_coefficients: intdir_large_coefficients.view(),
                    small_coefficients: intdir_small_coefficients.view(),
                    radii: intdir_radii.view(),
                    potential: intdir_potential.view(),
                    potential_coefficients: intdir_potential_coefficients.view(),
                    energy: -0.08,
                    origin_power: 0.999,
                    initial_large_coefficient: 0.85,
                    initial_small_coefficient: -0.004,
                    asymptotic_large_component: 0.02,
                    kappa: -1,
                    speed_of_light: intdir_speed_of_light,
                    step: intdir_step,
                    matching_precision: 1.0e-7,
                    coefficient_count: 6,
                    active_len: 151,
                    mode: AtomicDiracIntegrationMode::SearchMatchingPoint,
                    matching_index_1based: 0,
                    max_index_1based: 0,
                },
            )))
        });
    });
    let muatco_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &muatco_kappas,
        occupations: &muatco_occupations,
        valence_occupations: &muatco_valence,
    });
    if let Ok(muatco_coefficients) = muatco_coefficients {
        let lagdat_shell_markers = [-1, 1, 1, 1, -1];
        c.bench_function("atom_lagdat_parameters_5_orbitals", |b| {
            b.iter(|| {
                black_box(atomic_lagrange_parameters(
                    black_box(AtomicLagrangeParametersInput {
                        active_orbital_1based: None,
                        include_exchange: true,
                        kappas: &muatco_kappas,
                        occupations: &muatco_occupations,
                        shell_markers: &lagdat_shell_markers,
                        coulomb_coefficients: muatco_coefficients.view(),
                    }),
                    |request: AtomicRadialIntegralRequest| {
                        Ok(0.0001 * (request.rank + 1) as f64
                            + 0.001 * request.first_left as f64
                            + 0.0002 * request.first_right as f64
                            + 0.00003 * request.second_left as f64
                            + 0.000004 * request.second_right as f64)
                    },
                ))
            });
        });
    }

    let tabrat_principal = [1, 2, 2, 3, 3];
    let tabrat_kappas = [-1, -1, 1, -2, 1];
    let tabrat_occupations = [2.0, 1.5, 0.5, 3.0, 0.25];
    let tabrat_energies = [-0.70, -0.25, -0.18, -0.09, -0.04];
    c.bench_function("atom_tabrat_tabulation_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_tabulation(
                black_box(AtomicTabulationInput {
                    principal_quantum_numbers: &tabrat_principal,
                    kappas: &tabrat_kappas,
                    occupations: &tabrat_occupations,
                    orbital_energies: &tabrat_energies,
                }),
                |request: AtomicTabulationIntegralRequest| {
                    Ok(0.01 * (request.left + 1) as f64
                        + 0.02 * (request.right + 1) as f64
                        + 0.001 * request.power as f64
                        + 0.1)
                },
            ))
        });
    });

    let fpf0_radial_count = 251;
    let fpf0_orbital_count = 5;
    let fpf0_radial_step = 0.05;
    let fpf0_radii = Array1::from_shape_fn(fpf0_radial_count, |index| {
        (-8.8 + fpf0_radial_step * index as f64).exp()
    });
    let fpf0_density = Array1::from_shape_fn(fpf0_radial_count, |index| {
        0.3 * (-0.7 * fpf0_radii[index]).exp() + 0.01 * (index + 1).rem_euclid(7) as f64
    });
    let fpf0_initial_large = Array1::from_shape_fn(fpf0_radial_count, |index| {
        0.2 * (-0.4 * fpf0_radii[index]).exp() + 0.001 * (index + 1) as f64
    });
    let fpf0_initial_small = Array1::from_shape_fn(fpf0_radial_count, |index| {
        -0.05 * (-0.3 * fpf0_radii[index]).exp() + 0.0002 * (index + 1) as f64
    });
    let fpf0_large =
        Array2::from_shape_fn((fpf0_radial_count, fpf0_orbital_count), |(row, col)| {
            let orbital = (col + 1) as f64;
            (0.03 * orbital + 0.0007 * (row + 1) as f64) * (-0.05 * orbital * fpf0_radii[row]).exp()
        });
    let fpf0_small =
        Array2::from_shape_fn((fpf0_radial_count, fpf0_orbital_count), |(row, col)| {
            let orbital = (col + 1) as f64;
            (-0.01 * orbital + 0.0003 * (row + 1) as f64)
                * (-0.03 * orbital * fpf0_radii[row]).exp()
        });
    let fpf0_occupations = [2.0, 2.0, 1.5, 0.5, 0.0];
    let fpf0_energies = [-0.85, -0.55, -0.21, -0.08, 0.04];
    let fpf0_kappas = [-1, 1, -2, 2, -1];
    c.bench_function("atom_fpf0_form_factor_5_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_form_factor(black_box(AtomicFormFactorInput {
                atomic_number: 26,
                hole_orbital_1based: 2,
                radial_step: fpf0_radial_step,
                total_energy: -2.345,
                radii: fpf0_radii.view(),
                density_4pi: fpf0_density.view(),
                initial_large_component: fpf0_initial_large.view(),
                initial_small_component: fpf0_initial_small.view(),
                large_components: fpf0_large.view(),
                small_components: fpf0_small.view(),
                occupations: &fpf0_occupations,
                orbital_energies: &fpf0_energies,
                kappas: &fpf0_kappas,
            })))
        });
    });

    let ortdat_kappas = [-1, -1, 1, -1];
    let ortdat_active_lengths = [3, 4, 3, 5];
    let ortdat_powers = [0.1, 0.2, 0.3, 0.4];
    let ortdat_large_components = Array2::from_shape_fn((5, 4), |(row, orbital)| {
        0.07 * (row + 1) as f64 + 0.11 * (orbital + 1) as f64
    });
    let ortdat_small_components = Array2::from_shape_fn((5, 4), |(row, orbital)| {
        0.03 * (row + 1) as f64 - 0.02 * (orbital + 1) as f64
    });
    let ortdat_large_coefficients = Array2::from_shape_fn((4, 4), |(row, orbital)| {
        0.2 * (row + 1) as f64 + 0.05 * (orbital + 1) as f64
    });
    let ortdat_small_coefficients = Array2::from_shape_fn((4, 4), |(row, orbital)| {
        -0.03 * (row + 1) as f64 + 0.04 * (orbital + 1) as f64
    });
    c.bench_function("atom_ortdat_schmidt_4_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_schmidt_orthogonalization(
                black_box(AtomicSchmidtOrthogonalizationInput {
                    active_orbital_1based: None,
                    kappas: &ortdat_kappas,
                    active_lengths: &ortdat_active_lengths,
                    orbital_powers: &ortdat_powers,
                    large_components: ortdat_large_components.view(),
                    small_components: ortdat_small_components.view(),
                    large_coefficients: ortdat_large_coefficients.view(),
                    small_coefficients: ortdat_small_coefficients.view(),
                }),
                |request| match request {
                    AtomicSchmidtIntegralRequest::Projection(request) => Ok(request
                        .target_large
                        .iter()
                        .zip(request.reference_large.iter())
                        .map(|(&target, &reference)| target * reference)
                        .sum::<f64>()
                        + request
                            .target_small
                            .iter()
                            .zip(request.reference_small.iter())
                            .map(|(&target, &reference)| target * reference)
                            .sum::<f64>()),
                    AtomicSchmidtIntegralRequest::Norm(request) => Ok(request
                        .target_large
                        .iter()
                        .map(|&value| value * value)
                        .sum::<f64>()
                        + request
                            .target_small
                            .iter()
                            .map(|&value| value * value)
                            .sum::<f64>()),
                },
            ))
        });
    });

    let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
        1000.0 * (row + 1) as f64 + 10.0 * (column + 1) as f64 + channel as f64
    });
    c.bench_function("atom_akeato_lookup", |b| {
        b.iter(|| {
            black_box(atomic_direct_coulomb_coefficient(
                black_box(coefficients.view()),
                black_box(4),
                black_box(1),
                black_box(4),
            ))
        });
    });
    c.bench_function("atom_bkeato_lookup", |b| {
        b.iter(|| {
            black_box(atomic_exchange_coulomb_coefficient(
                black_box(coefficients.view()),
                black_box(1),
                black_box(4),
                black_box(4),
            ))
        });
    });
    c.bench_function("atom_bkmrdf_coefficients", |b| {
        b.iter(|| {
            black_box(atomic_breit_angular_coefficients(
                black_box(2),
                black_box(-4),
                black_box(3),
            ))
        });
    });
    let total_kappas = [-1, 1, -2, 2];
    let total_occupations = [2.0, 1.5, 3.0, 0.5];
    let total_valence = [0.0, 0.0, 1.0, 0.0];
    let total_energies = [-0.7, -0.3, -0.12, -0.05];
    let total_coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
        0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as f64
    });
    c.bench_function("atom_etotal_accumulation_4_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_total_energy(
                black_box(AtomicTotalEnergyInput {
                    kappas: &total_kappas,
                    occupations: &total_occupations,
                    valence_occupations: &total_valence,
                    orbital_energies: &total_energies,
                    coulomb_coefficients: total_coefficients.view(),
                }),
                |request: AtomicRadialIntegralRequest| {
                    Ok(0.0001 * (request.rank + 1) as f64
                        + 0.001 * request.first_left as f64
                        + 0.0002 * request.first_right as f64
                        + 0.00003 * request.second_left as f64
                        + 0.000004 * request.second_right as f64)
                },
            ))
        });
    });
    let s02_kappas = [-1, -1, 1, 1, -2, -3];
    let s02_occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
    let mut s02_overlaps =
        Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as f64);
    for index in 0..6 {
        s02_overlaps[(index, index)] = 1.0;
    }
    s02_overlaps[(0, 1)] = 0.91;
    s02_overlaps[(1, 0)] = 0.91;
    s02_overlaps[(2, 3)] = 0.82;
    s02_overlaps[(3, 2)] = 0.82;
    c.bench_function("atom_s02at_overlap_reduction_6_orbitals", |b| {
        b.iter(|| {
            black_box(atomic_overlap_amplitude_reduction(black_box(
                AtomicOverlapAmplitudeReductionInput {
                    hole_orbital_1based: Some(4),
                    kappas: &s02_kappas,
                    occupations: &s02_occupations,
                    overlap_integrals: s02_overlaps.view(),
                },
            )))
        });
    });

    let xsph_kind = array![2, 4, 2, -3, 4, 5, -3, 2];
    let xsph_orbital_l = array![1, 2, 3, 1, 4, 0, 5, 6];
    let xsph_final_lj = array![2, 1, 5, 3, 4, 0, 6, 1];
    let xsph_index_map = array![1, 2, -1, 3, -2, 4, -3, -1];
    c.bench_function("xsph_mincalc_plan", |b| {
        b.iter(|| {
            black_box(xsph_minimize_calculations(
                black_box(xsph_kind.view()),
                black_box(xsph_orbital_l.view()),
                black_box(xsph_final_lj.view()),
                black_box(8),
            ))
        });
    });
    c.bench_function("xsph_ljneeded0_flags", |b| {
        b.iter(|| {
            black_box(xsph_lj_needed_flags(
                black_box(6),
                black_box(xsph_final_lj.view()),
                black_box(xsph_index_map.view()),
                black_box(8),
                black_box(1),
            ))
        });
    });
    c.bench_function("xsph_xmultjas_factor", |b| {
        b.iter(|| {
            black_box(xsph_longitudinal_multipole_factor(
                black_box(3),
                black_box(-2),
                black_box(2),
            ))
        });
    });
    c.bench_function("xsph_xmult_factors", |b| {
        b.iter(|| {
            black_box(xsph_relativistic_multipole_factors(
                black_box(-3),
                black_box(2),
                black_box(1),
                black_box(2),
            ))
        });
    });
    c.bench_function("xsph_acoef_lx4", |b| {
        b.iter(|| {
            black_box(xsph_angular_density_coefficients(
                black_box(1),
                black_box(4),
            ))
        });
    });
    let xsph_lgind = array![0, 1, 2, 1, 3, 2, 4];
    let xsph_ljind = array![0, 1, 1, 2, 2, 3, 3];
    c.bench_function("xsph_bcoefjas_weights", |b| {
        b.iter(|| {
            black_box(xsph_nrixs_transition_weights(
                black_box(-1),
                black_box(1),
                black_box(4),
                black_box(9),
                black_box(3),
                black_box(xsph_lgind.view()),
                black_box(xsph_ljind.view()),
                black_box(7),
            ))
        });
    });
    let xsph_spec_index_map = array![1, -1, 2, 1, -2];
    let xsph_spec_lind = array![0, 1, 2, 3, 4];
    let xsph_spec_ljind = array![0, 1, 2, 3, 1];
    let xsph_spec_radial = array![
        Complex::new(0.12, -0.03),
        Complex::new(-0.08, 0.19),
        Complex::new(0.31, 0.07),
        Complex::new(-0.22, -0.11)
    ];
    let xsph_spec_qweights = array![Complex::new(1.0, 0.0), Complex::new(0.49, 0.12)];
    let xsph_spec_cosines = arr2(&[[0.25, -0.35], [0.60, -0.40]]);
    let xsph_spec_hbmat = Array3::from_shape_fn((2, 5, 4).f(), |(spin, state, magnetic)| {
        let state_feff = state as f64 + 1.0;
        let magnetic_j2 = [-3.0, -1.0, 1.0, 3.0][magnetic];
        0.05 * state_feff
            + 0.11 * spin as f64
            + 0.017 * magnetic_j2
            + 0.003 * state_feff * magnetic_j2
    });
    c.bench_function("xsph_specupdlg_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
            black_box(xsph_update_nrixs_lg_spectrum(
                XsphLgSpectrumUpdateInput {
                    calculation_index: black_box(1),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    orbital_l: black_box(xsph_spec_lind.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
            ))
        });
    });
    c.bench_function("xsph_specupd_lj_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(4, Complex::new(0.01, -0.02));
            let mut spectrum_norm = 0.02;
            black_box(xsph_update_nrixs_lj_spectrum(
                XsphLjSpectrumUpdateInput {
                    calculation_index: black_box(1),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
                black_box(&mut spectrum_norm),
            ))
        });
    });
    c.bench_function("xsph_specupdatom_update", |b| {
        b.iter(|| {
            let mut spectrum = Array1::from_elem(5, Complex::new(0.02, 0.01));
            let mut spectrum_norm = 0.005;
            black_box(xsph_update_nrixs_atom_spectrum(
                XsphLjSpectrumUpdateInput {
                    calculation_index: black_box(2),
                    spin_index: black_box(1),
                    index_map: black_box(xsph_spec_index_map.view()),
                    final_lj: black_box(xsph_spec_ljind.view()),
                    initial_j2: black_box(3),
                    transition_weights: black_box(xsph_spec_hbmat.view()),
                    radial_integrals: black_box(xsph_spec_radial.view()),
                    q_weights: black_box(xsph_spec_qweights.view()),
                    q_cosines: black_box(xsph_spec_cosines.view()),
                    mix_dff: black_box(false),
                    mdff_mode: black_box(0),
                    ljmax: black_box(3),
                    active_len: black_box(5),
                    mode: XsphSpectrumUpdateMode::Regular,
                },
                black_box(spectrum.view_mut()),
                black_box(&mut spectrum_norm),
            ))
        });
    });
    let xsph_axafs_energies = Array1::from_shape_fn(64, |index| {
        let i = index as f64 + 1.0;
        Complex::new(0.015 * (i - 3.0).powi(2) + 0.012 * (i - 1.0), 0.002 * i)
    });
    let xsph_axafs_xsec = Array1::from_shape_fn(64, |index| {
        let i = index as f64 + 1.0;
        Complex::new(
            -0.03 * i,
            0.42 + 0.021 * i + 0.004 * i * i + 0.025 * (0.7 * i).sin(),
        )
    });
    c.bench_function("xsph_axafs_table", |b| {
        b.iter(|| {
            black_box(xsph_axafs(black_box(XsphAxafsInput {
                energies: black_box(xsph_axafs_energies.view()),
                cross_section: black_box(xsph_axafs_xsec.view()),
                fermi_energy: black_box(0.37),
                horizontal_count: black_box(48),
                zero_wave_index: black_box(2),
            })))
        });
    });
    c.bench_function("xsph_getoccnorm", |b| {
        b.iter(|| black_box(xsph_occupation_normalization(black_box(92), black_box(22))));
    });
    let mut xsph_hole_large = Array1::<f64>::zeros(251);
    let mut xsph_hole_small = Array1::<f64>::zeros(251);
    for index in 0..15 {
        let i = index as f64 + 1.0;
        xsph_hole_large[index] = 0.1 + 0.017 * i + 0.0009 * i * i + 0.002 * (0.3 * i).sin();
        xsph_hole_small[index] = -0.04 + 0.011 * i - 0.0004 * i * i + 0.001 * (0.25 * i).cos();
    }
    c.bench_function("xsph_getholeorb0", |b| {
        b.iter(|| {
            black_box(xsph_initial_hole_orbital(black_box(XsphHoleOrbitalInput {
                large_component: black_box(xsph_hole_large.view()),
                small_component: black_box(xsph_hole_small.view()),
                original_step: black_box(0.05),
                new_step: black_box(0.035),
                output_count: black_box(64),
                output_capacity: black_box(96),
            })))
        });
    });
    let xsph_phase_sort_input = arr1(&[
        Complex::new(0.002, 9.0),
        Complex::new(-0.004, 8.0),
        Complex::new(0.0004, 7.0),
        Complex::new(0.0012, 6.0),
        Complex::new(-0.0036, 5.0),
        Complex::new(0.25, 4.0),
    ]);
    let xsph_user_phase_points = arr1(&[
        Complex::new(-5.0, 0.2),
        Complex::new(0.0004, 0.0),
        Complex::new(12.0, -0.1),
    ]);
    let xsph_user_phase_records = [
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::Energy,
            minimum: XsphPhaseUserGridMinimum::Value(-2.0),
            maximum: 2.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::Regular(XsphPhaseUserRegularGrid {
            kind: XsphPhaseUserGridKind::WaveNumber,
            minimum: XsphPhaseUserGridMinimum::Last,
            maximum: 3.0,
            step: 1.0,
        }),
        XsphPhaseUserGridRecord::User(xsph_user_phase_points.view()),
    ];
    c.bench_function("xsph_phmesh2_primitives", |b| {
        b.iter(|| {
            let even = black_box(xsph_even_energy_mesh(
                black_box(-0.2),
                black_box(0.35),
                black_box(0.11),
                black_box(64),
            ));
            let k_mesh = black_box(xsph_k_energy_mesh(
                black_box(-1.2),
                black_box(-0.2),
                black_box(0.25),
                black_box(64),
            ));
            let exp_mesh = black_box(xsph_exponential_energy_mesh(
                black_box(0.02),
                black_box(0.5),
                black_box(0.4),
                black_box(64),
            ));
            let vertical = black_box(xsph_vertical_energy_mesh_84(black_box(0.05), black_box(64)));
            let exafs84 = black_box(xsph_exafs_energy_grid_84(
                black_box(18.0 * FEFF_BOHR_ANGSTROM),
                black_box(100),
            ));
            let xanes84 = black_box(xsph_xanes_energy_grid_84(
                black_box(4.0 * FEFF_BOHR_ANGSTROM),
                black_box(0.5 * FEFF_BOHR_ANGSTROM),
                black_box(0.02),
                black_box(80),
            ));
            let fprime84 = black_box(xsph_fprime_energy_grid_84(
                black_box(-5.0),
                black_box(10.0),
                black_box(0.25),
                black_box(9.0),
                black_box(-0.4),
                black_box(64),
            ));
            let xes84 = black_box(xsph_xes_energy_grid_84(
                black_box(-5.0),
                black_box(10.0),
                black_box(0.25),
                black_box(-0.4),
                black_box(64),
            ));
            let phase84 = black_box(xsph_phase_energy_mesh_84(black_box(
                XsphPhaseEnergyMesh84Input {
                    spectroscopy: 1,
                    edge: -0.4,
                    reference_energy: 9.0,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    max_wave_number: 18.0 * FEFF_BOHR_ANGSTROM,
                    wave_number_step: 0.5 * FEFF_BOHR_ANGSTROM,
                    xanes_energy_step: 0.02,
                    capacity: 120,
                },
            )));
            let no_fms_phase84 = black_box(xsph_phase_energy_mesh_84(black_box(
                XsphPhaseEnergyMesh84Input {
                    spectroscopy: -1,
                    edge: -0.4,
                    reference_energy: 9.0,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    max_wave_number: 18.0 * FEFF_BOHR_ANGSTROM,
                    wave_number_step: 0.5 * FEFF_BOHR_ANGSTROM,
                    xanes_energy_step: 0.02,
                    capacity: 120,
                },
            )));
            let user_phase = black_box(xsph_phase_energy_mesh_user(black_box(
                XsphPhaseUserGridInput {
                    spectroscopy: 1,
                    edge: -0.4,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    records: &xsph_user_phase_records,
                    capacity: 120,
                },
            )));
            let thermal_phase = black_box(xsph_thermal_phase_energy_mesh(black_box(
                XsphThermalPhaseEnergyMeshInput {
                    edge: -0.4,
                    constant_imaginary: 0.01,
                    core_hole_broadening: 0.08,
                    core_valence_separation: -1.5,
                    electronic_temperature: 5.0,
                    user_records: Some(&xsph_user_phase_records),
                    capacity: 240,
                },
            )));
            let reversed = black_box(xsph_reverse_energy_grid(
                black_box(xsph_phase_sort_input.view()),
                black_box(0.25),
            ));
            let sorted = black_box(xsph_sort_energy_grid(black_box(
                xsph_phase_sort_input.view(),
            )));
            black_box((
                even,
                k_mesh,
                exp_mesh,
                vertical,
                exafs84,
                xanes84,
                fprime84,
                xes84,
                phase84,
                no_fms_phase84,
                user_phase,
                thermal_phase,
                reversed,
                sorted,
            ))
        });
    });
    let xsph_radii = array![0.1, 1.0, 3.0, 20.0, 40.0, 80.0];
    c.bench_function("xsph_qbesselget_table", |b| {
        b.iter(|| {
            black_box(xsph_q_bessel_table(
                black_box(0.35),
                black_box(xsph_radii.view()),
                black_box(6),
            ))
        });
    });

    c.bench_function("distance_between", |b| {
        b.iter(|| {
            black_box(distance_between(
                black_box([1.0, -2.0, 0.5]),
                black_box([-3.0, 4.0, 2.5]),
            ))
        });
    });
    c.bench_function("eels_electron_wavelength", |b| {
        b.iter(|| black_box(electron_wavelength_atomic_units(black_box(300_000.0))));
    });
    c.bench_function("eels_euler_rotation_matrix", |b| {
        b.iter(|| {
            black_box(eels_euler_rotation_matrix(
                black_box(0.3),
                black_box(0.4),
                black_box(-0.2),
            ))
        });
    });
    let eels_matrix = arr2(&[[1.25, 2.0, -0.25], [-0.5, 0.125, 3.0], [0.75, -1.5, 0.5]]);
    let eels_vector = arr1(&[0.2, -1.5, 4.0]);
    c.bench_function("eels_product_matrix_vector", |b| {
        b.iter(|| {
            black_box(eels_product_matrix_vector(
                black_box(eels_matrix.view()),
                black_box(eels_vector.view()),
            ))
        });
    });
    let qmesh_theta_x = arr1(&[0.0, 0.0015, -0.002, -0.001, 0.0025, -0.0035, 0.004, 0.005]);
    let qmesh_theta_y = arr1(&[0.0, -0.0025, 0.001, -0.003, 0.002, 0.0015, -0.004, 0.003]);
    c.bench_function("eels_qmesh_8pos", |b| {
        b.iter(|| {
            black_box(eels_qmesh(black_box(EelsQMeshInput {
                incident_energy_ev: 300_000.0,
                scattered_energy_ev: 299_880.0,
                beam_direction: [0.2, 0.3, 0.9],
                theta_x: qmesh_theta_x.view(),
                theta_y: qmesh_theta_y.view(),
                relativistic: true,
            })))
        });
    });
    c.bench_function("eels_integration_mesh_log", |b| {
        b.iter(|| {
            black_box(eels_integration_mesh(black_box(EelsMeshInput {
                collection_angle: 0.015,
                convergence_angle: 0.008,
                theta0: 0.001,
                theta_x_center: -0.0015,
                theta_y_center: 0.0005,
                radial_count: 3,
                angular_count: 2,
                mode: EelsMeshMode::Logarithmic,
            })))
        });
    });
    let eels_losses = arr1(&[12.5, 28.0, 64.0, 92.0]);
    let eels_tensor = Array3::from_shape_fn((4, 3, 3), |(energy, row, column)| {
        let i = (energy + 1) as f64;
        let j1 = (row + 1) as f64;
        let j2 = (column + 1) as f64;
        0.015 * i + 0.11 * j1 - 0.045 * j2 + 0.002 * i * j1 * j2
    });
    let eels_background = arr1(&[0.092, 0.104, 0.116, 0.128]);
    c.bench_function("eels_spectrum_4e_8pos", |b| {
        b.iter(|| {
            black_box(eels_spectrum(black_box(EelsSpectrumInput {
                incident_energy_ev: 200_000.0,
                beam_direction: [0.25, -0.15, 0.95],
                mesh: EelsMeshInput {
                    collection_angle: 0.014,
                    convergence_angle: 0.006,
                    theta0: 0.0007,
                    theta_x_center: 0.0012,
                    theta_y_center: -0.0008,
                    radial_count: 2,
                    angular_count: 2,
                    mode: EelsMeshMode::Uniform,
                },
                energy_loss_ev: eels_losses.view(),
                transition_tensor: eels_tensor.view(),
                atomic_background: eels_background.view(),
                relativistic: true,
            })))
        });
    });
    let eels_readsp_owned = (1..=10)
        .map(|polarization_index| {
            let energy_loss = Array1::from_shape_fn(4, |energy| 10.0 * (energy + 1) as f64 + 0.25);
            let spectrum = Array1::from_shape_fn(4, |energy| {
                let ip = polarization_index as f64;
                let row = (energy + 1) as f64;
                0.1 * ip + 0.01 * row + 0.001 * ip * row
            });
            let background = Array1::from_shape_fn(4, |energy| {
                let ip = polarization_index as f64;
                let row = (energy + 1) as f64;
                1.0 + 0.2 * ip + 0.03 * row
            });
            (energy_loss, spectrum, background)
        })
        .collect::<Vec<_>>();
    let eels_readsp_sources = eels_readsp_owned
        .iter()
        .enumerate()
        .map(
            |(index, (energy_loss, spectrum, background))| EelsReadSpectrumSource {
                polarization_index: index + 1,
                energy_loss_ev: energy_loss.view(),
                selected_spectrum: spectrum.view(),
                atomic_background: background.view(),
            },
        )
        .collect::<Vec<_>>();
    c.bench_function("eels_readsp_reduce_10x4", |b| {
        b.iter(|| {
            black_box(eels_read_spectrum(black_box(EelsReadSpectrumInput {
                sources: &eels_readsp_sources,
                orientation_averaged: false,
                cross_terms: true,
                polarization_min: 1,
                polarization_step: 1,
                polarization_max: 9,
            })))
        });
    });
    let eels_averaged = arr1(&[0.0045, 0.0062, 0.0087, 0.011]);
    c.bench_function("eels_gos_20q_4e", |b| {
        b.iter(|| {
            black_box(eels_generalized_oscillator_strength(black_box(
                EelsGosInput {
                    incident_energy_ev: 200_000.0,
                    energy_loss_ev: eels_losses.view(),
                    averaged_spectrum: eels_averaged.view(),
                    relativistic: true,
                },
            )))
        });
    });
    let eels_angular_q = arr2(&[
        [0.145, 0.310, 0.720, 1.350],
        [0.010, 0.045, 0.115, 0.210],
        [0.25, -0.40, 0.90, -1.20],
    ]);
    let eels_angular_weights = arr1(&[0.185, 0.295, 0.470, 0.815]);
    let eels_angular_partials = Array2::from_shape_fn((10, 4), |(partial, position)| {
        let l = (partial + 1) as f64;
        let k = (position + 1) as f64;
        0.003 * l.powi(2) + 0.017 * k + 0.0009 * l * k
    });
    c.bench_function("eels_angular_dependence_4pos", |b| {
        b.iter(|| {
            black_box(eels_angular_dependence(black_box(
                EelsAngularDependenceInput {
                    q_vectors_spherical: eels_angular_q.view(),
                    weights: eels_angular_weights.view(),
                    partial_spectra: eels_angular_partials.view(),
                    incident_wave_number: 82.75,
                },
            )))
        });
    });
    let eels_collection_sigma_x = arr1(&[0.0045, 0.0062, 0.0087]);
    let eels_collection_sigma_y = arr1(&[0.0051, 0.0068, 0.0094]);
    let eels_collection_pi = arr1(&[0.0060, 0.0077, 0.0102]);
    c.bench_function("eels_collection_dependence_uniform_3rows", |b| {
        b.iter(|| {
            black_box(eels_collection_angle_dependence(black_box(
                EelsCollectionDependenceInput {
                    incident_energy_ev: 200_000.0,
                    beam_direction: [0.25, -0.15, 0.95],
                    mesh: EelsMeshInput {
                        collection_angle: 0.020,
                        convergence_angle: 0.006,
                        theta0: 0.001,
                        theta_x_center: 0.0012,
                        theta_y_center: -0.0008,
                        radial_count: 5,
                        angular_count: 2,
                        mode: EelsMeshMode::Uniform,
                    },
                    magic_energy_ev: 10.0,
                    energy_loss_ev: eels_losses.slice(ndarray::s![..3]),
                    sigma_x_spectrum: eels_collection_sigma_x.view(),
                    sigma_y_spectrum: eels_collection_sigma_y.view(),
                    pi_spectrum: eels_collection_pi.view(),
                    relativistic: true,
                },
            )))
        });
    });
    let epsilon_tables = sample_epsilon_tables();
    let epsilon_weights = [1.0, 0.35, 0.15];
    c.bench_function("opcons_combine_epsilon_tables_3x160", |b| {
        b.iter(|| {
            black_box(combine_epsilon_tables(
                black_box(&epsilon_tables),
                black_box(&epsilon_weights),
            ))
        });
    });
    let fullspectrum_energy = Array1::from_shape_fn(4096, |index| 5.0 + 0.05 * index as f64);
    let fullspectrum_epsilon2 = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        0.2 + 0.05 * x.sin().abs()
    });
    let fullspectrum_epsilon = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(0.1 + 0.02 * x.cos(), fullspectrum_epsilon2[index])
    });
    let fullspectrum_refractive = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(0.02 + 0.005 * x.sin(), 0.01 + 0.002 * x.cos())
    });
    let fullspectrum_absorption = Array1::from_shape_fn(4096, |index| {
        1000.0 + 5.0 * index as f64 + 20.0 * (index as f64 * 0.005).sin()
    });
    let fullspectrum_valence_energy_ev =
        Array1::from_shape_fn(4096, |index| (5.0 + 0.05 * index as f64) * 27.211_396);
    let fullspectrum_valence_absorption = Array1::from_shape_fn(4096, |index| {
        0.5 + 0.01 * index as f64 + 0.05 * (index as f64 * 0.01).cos()
    });
    let fullspectrum_background_energy_ev =
        Array1::from_shape_fn(2048, |index| (5.0 + 0.2 * index as f64) * 27.211_396);
    let fullspectrum_background_f_prime = Array1::from_shape_fn(2048, |index| {
        1.0 + 0.002 * index as f64 + 0.05 * (index as f64 * 0.01).sin()
    });
    let fullspectrum_background_f_double_prime = Array1::from_shape_fn(2048, |index| {
        0.05 + 0.0005 * index as f64 + 0.01 * (index as f64 * 0.02).cos().abs()
    });
    let fullspectrum_background_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: fullspectrum_background_energy_ev.view(),
        f_prime: fullspectrum_background_f_prime.view(),
        f_double_prime: fullspectrum_background_f_double_prime.view(),
    }];
    let fullspectrum_epsilon_minus_one = Array1::from_shape_fn(4096, |index| {
        Complex::new(
            0.2 + 0.01 * (index as f64 * 0.01).sin(),
            0.4 + 0.05 * (index as f64 * 0.02).cos().abs(),
        )
    });
    let fullspectrum_scattering_factor = Array1::from_shape_fn(4096, |index| {
        let x = index as f64 * 0.01;
        Complex::new(1.0 + 0.02 * x.sin(), 0.3 + 0.04 * x.cos().abs())
    });
    let fullspectrum_background_scattering_factor =
        fullspectrum_scattering_factor.mapv(|value| value * Complex::new(0.85, 0.02));
    let fullspectrum_fine_fms_energy_ev =
        Array1::from_shape_fn(2048, |index| (5.0 + 0.05 * index as f64) * 27.211_396);
    let fullspectrum_fine_path_energy_ev =
        Array1::from_shape_fn(2048, |index| (8.0 + 0.08 * index as f64) * 27.211_396);
    let fullspectrum_fine_fms_wave =
        Array1::from_shape_fn(2048, |index| 0.25 + 0.003 * index as f64);
    let fullspectrum_fine_path_wave =
        Array1::from_shape_fn(2048, |index| 3.0 + 0.004 * index as f64);
    let fullspectrum_fine_real_fms = Array1::from_shape_fn(2048, |index| {
        1.0 + 0.002 * index as f64 + 0.02 * (index as f64 * 0.01).sin()
    });
    let fullspectrum_fine_real_path = Array1::from_shape_fn(2048, |index| {
        1.5 + 0.0025 * index as f64 + 0.03 * (index as f64 * 0.015).cos()
    });
    let fullspectrum_fine_imag_fms = Array1::from_shape_fn(2048, |index| {
        0.05 + 0.001 * index as f64 + 0.01 * (index as f64 * 0.02).sin().abs()
    });
    let fullspectrum_fine_imag_path = Array1::from_shape_fn(2048, |index| {
        0.08 + 0.0012 * index as f64 + 0.01 * (index as f64 * 0.02).cos().abs()
    });
    let fullspectrum_fine_real_fms_background =
        fullspectrum_fine_real_fms.mapv(|value| value * 0.85);
    let fullspectrum_fine_real_path_background =
        fullspectrum_fine_real_path.mapv(|value| value * 0.9);
    let fullspectrum_fine_imag_fms_background =
        fullspectrum_fine_imag_fms.mapv(|value| value * 0.8);
    let fullspectrum_fine_imag_path_background =
        fullspectrum_fine_imag_path.mapv(|value| value * 0.82);
    let fullspectrum_atomic_numbers = array![29_usize, 8, 29, 14, 8, 29];
    let fullspectrum_multiplicities = array![0.01, 2.0, 3.0, 1.0, 4.0, 2.0];
    let fullspectrum_norman_radii = array![2.0, 1.5, 2.5, 1.8, 1.6, 2.2];
    let fullspectrum_occupations = Array1::from_shape_fn(40, |index| match index {
        0 => 2.0,
        1 => 1.0,
        2 => 2.0,
        3 => 1.0,
        4 => 0.5,
        _ => 0.0,
    });
    let fullspectrum_edge_onsets = Array1::from_shape_fn(40, |index| 0.2 + 0.1 * index as f64);
    let fullspectrum_edges = array![25.0, 75.0, 140.0, 210.0];
    let fullspectrum_elam_components = [29, 8, 79];
    let fullspectrum_default_edges = [
        FullSpectrumDefaultGridEdge {
            atomic_number: 8,
            hole_index: 1,
            fine_structure: false,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 1,
            fine_structure: true,
        },
        FullSpectrumDefaultGridEdge {
            atomic_number: 29,
            hole_index: 4,
            fine_structure: false,
        },
    ];
    c.bench_function("fullspectrum_elam_edge_adapter", |b| {
        b.iter(|| {
            black_box(full_spectrum_elam_edge_energies(black_box(
                &fullspectrum_elam_components,
            )))
        });
    });
    c.bench_function("fullspectrum_rdop_default_grid", |b| {
        b.iter(|| {
            black_box(full_spectrum_default_energy_grid(black_box(
                &fullspectrum_default_edges,
            )))
        });
    });
    c.bench_function("fullspectrum_edge_grid_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_edge_energy_grid(black_box(
                FullSpectrumEdgeGridInput {
                    min_energy: 0.0,
                    max_energy: 250.0,
                    edge_energies: fullspectrum_edges.view(),
                    wave_number_step: 0.2,
                    max_points: 4096,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_linear_grid_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_linear_energy_grid(black_box(
                FullSpectrumLinearGridInput {
                    point_count: 4096,
                    min_energy: 0.0,
                    max_energy: 250.0,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_qsum_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_effective_electron_count(black_box(
                FullSpectrumQSumInput {
                    number_density: 0.075,
                    epsilon2: fullspectrum_epsilon2.view(),
                    omega: fullspectrum_energy.view(),
                    active_len: fullspectrum_energy.len(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_valence_epsilon2_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_valence_epsilon2(black_box(
                FullSpectrumValenceInput {
                    number_density: 0.075,
                    omega: fullspectrum_energy.view(),
                    source_energy_ev: fullspectrum_valence_energy_ev.view(),
                    source_absorption_angstrom2: fullspectrum_valence_absorption.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_background_from_fprime_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_background_from_fprime(black_box(
                FullSpectrumBackgroundInput {
                    omega: fullspectrum_energy.view(),
                    segments: &fullspectrum_background_segments,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_optical_constants_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_optical_constants(black_box(
                FullSpectrumOpticalConstantsInput {
                    omega: fullspectrum_energy.view(),
                    epsilon_minus_one: fullspectrum_epsilon_minus_one.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_scattering_to_dielectric_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_scattering_to_dielectric(black_box(
                FullSpectrumScatteringDielectricInput {
                    number_density: 0.075,
                    omega: fullspectrum_energy.view(),
                    scattering_factor: fullspectrum_scattering_factor.view(),
                    background_scattering_factor: fullspectrum_background_scattering_factor.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_fine_structure_from_segments_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_fine_structure_from_segments(black_box(
                FullSpectrumFineStructureInput {
                    omega: fullspectrum_energy.view(),
                    real_fms: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                        scattering_factor: fullspectrum_fine_real_fms.view(),
                        background: fullspectrum_fine_real_fms_background.view(),
                    },
                    real_path: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                        scattering_factor: fullspectrum_fine_real_path.view(),
                        background: fullspectrum_fine_real_path_background.view(),
                    },
                    imaginary_fms: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                        scattering_factor: fullspectrum_fine_imag_fms.view(),
                        background: fullspectrum_fine_imag_fms_background.view(),
                    },
                    imaginary_path: FullSpectrumFineStructureSegmentInput {
                        photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                        wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                        scattering_factor: fullspectrum_fine_imag_path.view(),
                        background: fullspectrum_fine_imag_path_background.view(),
                    },
                    low_wave_number: 3.0,
                    high_wave_number: 4.0,
                },
            )))
        });
    });
    let fullspectrum_background_result =
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: fullspectrum_energy.view(),
            segments: &fullspectrum_background_segments,
        });
    let fullspectrum_fine_structure_result =
        full_spectrum_fine_structure_from_segments(FullSpectrumFineStructureInput {
            omega: fullspectrum_energy.view(),
            real_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                scattering_factor: fullspectrum_fine_real_fms.view(),
                background: fullspectrum_fine_real_fms_background.view(),
            },
            real_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                scattering_factor: fullspectrum_fine_real_path.view(),
                background: fullspectrum_fine_real_path_background.view(),
            },
            imaginary_fms: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_fms_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_fms_wave.view(),
                scattering_factor: fullspectrum_fine_imag_fms.view(),
                background: fullspectrum_fine_imag_fms_background.view(),
            },
            imaginary_path: FullSpectrumFineStructureSegmentInput {
                photon_energy_ev: fullspectrum_fine_path_energy_ev.view(),
                wave_number_inverse_angstrom: fullspectrum_fine_path_wave.view(),
                scattering_factor: fullspectrum_fine_imag_path.view(),
                background: fullspectrum_fine_imag_path_background.view(),
            },
            low_wave_number: 3.0,
            high_wave_number: 4.0,
        });
    if let (Ok(background_result), Ok(fine_structure_result)) = (
        fullspectrum_background_result,
        fullspectrum_fine_structure_result,
    ) {
        c.bench_function("fullspectrum_assemble_edge_4096", |b| {
            b.iter(|| {
                black_box(full_spectrum_assemble_edge(black_box(
                    FullSpectrumEdgeAssemblyInput {
                        omega: fullspectrum_energy.view(),
                        background: &background_result,
                        fine_structure: &fine_structure_result,
                        transition_size: 0.05,
                    },
                )))
            });
        });
    }
    c.bench_function("fullspectrum_number_density", |b| {
        b.iter(|| {
            black_box(full_spectrum_number_density(black_box(
                FullSpectrumNumberDensityInput {
                    target_atomic_number: 29,
                    atomic_numbers: fullspectrum_atomic_numbers.view(),
                    potential_multiplicities: fullspectrum_multiplicities.view(),
                    norman_radii: fullspectrum_norman_radii.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_edge_selection", |b| {
        b.iter(|| {
            black_box(full_spectrum_edges_from_occupations(black_box(
                FullSpectrumEdgeSelectionInput {
                    occupations: fullspectrum_occupations.view(),
                    edge_onsets_hartree: fullspectrum_edge_onsets.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_drude_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_drude_term(black_box(
                FullSpectrumDrudeInput {
                    omega: fullspectrum_energy.view(),
                    lifetime_seconds: 1.0e-15,
                    number_density: 0.075,
                },
            )))
        });
    });
    c.bench_function("fullspectrum_kk_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_kramers_kronig(black_box(
                FullSpectrumKramersKronigInput {
                    omega: fullspectrum_energy.view(),
                    epsilon2: fullspectrum_epsilon2.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_hamaker_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_hamaker_transform(black_box(
                FullSpectrumHamakerInput {
                    omega: fullspectrum_energy.view(),
                    epsilon: fullspectrum_epsilon.view(),
                },
            )))
        });
    });
    c.bench_function("fullspectrum_sumrules_4096", |b| {
        b.iter(|| {
            black_box(full_spectrum_sum_rules(black_box(
                FullSpectrumSumRulesInput {
                    number_density: 0.075,
                    energy_ev: fullspectrum_energy.view(),
                    epsilon_minus_one: fullspectrum_epsilon.view(),
                    refractive_index_minus_one: fullspectrum_refractive.view(),
                    absorption_coefficient: fullspectrum_absorption.view(),
                },
            )))
        });
    });
    let hydrogen_potentials = Array1::from_vec(vec![0, 1, 0]);
    let potential_atomic_numbers = Array1::from_vec(vec![8, 1]);
    let hydrogen_positions = arr2(&[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    c.bench_function("adjust_hydrogen_bonds_moveh", |b| {
        b.iter(|| {
            black_box(adjust_hydrogen_bonds(black_box(
                HydrogenBondAdjustmentInput {
                    atom_potentials: hydrogen_potentials.view(),
                    potential_atomic_numbers: potential_atomic_numbers.view(),
                    positions: hydrogen_positions.view(),
                },
            )))
        });
    });
    c.bench_function("x_log_x", |b| {
        b.iter(|| black_box(x_log_x(black_box(2.5))));
    });
    c.bench_function("dirac_hara_exchange_potential", |b| {
        b.iter(|| {
            black_box(dirac_hara_exchange_potential(
                black_box(2.0),
                black_box(1.3),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_ffq", |b| {
        b.iter(|| {
            black_box(hedin_lundqvist_ffq(
                black_box(0.8),
                black_box(0.42),
                black_box(1.2),
                black_box(0.7),
                black_box(4.0 / 3.0),
            ))
        });
    });
    c.bench_function("von_barth_hedin_potential", |b| {
        b.iter(|| black_box(von_barth_hedin_potential(black_box(2.5), black_box(1.2))));
    });
    c.bench_function("perdew_zunger_vxc", |b| {
        b.iter(|| black_box(perdew_zunger_vxc(black_box(2.0))));
    });
    c.bench_function("perrot_dharma_wardana_vxc", |b| {
        b.iter(|| black_box(perrot_dharma_wardana_vxc(black_box(2.0), black_box(0.05))));
    });
    c.bench_function("karasiev_sjostrom_dufty_trickey_vxc", |b| {
        b.iter(|| {
            black_box(karasiev_sjostrom_dufty_trickey_vxc(
                black_box(2.0),
                black_box(0.05),
            ))
        });
    });
    c.bench_function("quinn_imaginary_self_energy", |b| {
        b.iter(|| {
            black_box(quinn_imaginary_self_energy(
                black_box(1.15),
                black_box(2.0),
                black_box(0.65),
                black_box(0.42),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_imaginary_self_energy", |b| {
        b.iter(|| {
            black_box(hedin_lundqvist_imaginary_self_energy(
                black_box(2.0),
                black_box(1.3),
            ))
        });
    });
    c.bench_function("hedin_lundqvist_self_energy", |b| {
        b.iter(|| black_box(hedin_lundqvist_self_energy(black_box(2.0), black_box(1.3))));
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
    c.bench_function("real_polynomial_roots_croots", |b| {
        b.iter(|| black_box(real_polynomial_roots(black_box([1.0, 0.0, -1.0, 1.0]))));
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
    c.bench_function("omega_q", |b| {
        b.iter(|| black_box(omega_q(black_box(0.7), black_box(0.2))));
    });
    c.bench_function("gamma_q", |b| {
        b.iter(|| black_box(gamma_q(black_box(0.08), black_box(0.2))));
    });
    c.bench_function("log_i", |b| {
        b.iter(|| black_box(log_i(black_box(Complex::new(-1.0, 0.5)), black_box(-1))));
    });
    c.bench_function("hartree_fock_exchange", |b| {
        b.iter(|| {
            black_box(hartree_fock_exchange(
                black_box(Complex::new(1.6, 0.2)),
                black_box(0.8),
                black_box(1.1),
            ))
        });
    });
    let integrand_input = SelfEnergyIntegrandInput {
        q: Complex::new(0.8, 0.0),
        normalized_momentum: Complex::new(0.7, 0.0),
        normalized_energy: Complex::new(0.9, 0.02),
        plasmon_over_fermi: 0.35,
        width_over_fermi: 0.02,
        gap_energy: 0.0,
    };
    c.bench_function("self_energy_r1_integrand", |b| {
        b.iter(|| black_box(self_energy_r1_integrand(black_box(integrand_input))));
    });
    c.bench_function("cgratr_oscillatory", |b| {
        b.iter(|| {
            black_box(cgratr(
                |q| Ok((Complex::new(0.0, 3.0) * q).exp() / (Complex::new(1.0, 0.0) + q * q)),
                black_box(Complex::new(0.0, 0.0)),
                black_box(Complex::new(4.0, 0.0)),
                black_box(1.0e-5),
                black_box(1.0e-4),
                black_box(&[]),
            ))
        });
    });
    let mkexc_energy = ndarray::arr1(&[5.0, 12.0, 25.0, 60.0, 120.0, 250.0, 500.0]);
    let mkexc_loss = ndarray::arr1(&[0.18, 0.45, 0.32, 0.20, 0.11, 0.05, 0.02]);
    c.bench_function("make_excitation_poles_4", |b| {
        b.iter(|| {
            black_box(make_excitation_poles(
                black_box(mkexc_energy.view()),
                black_box(mkexc_loss.view()),
                black_box(12.0),
                black_box(4),
            ))
        });
    });
    let sfconv_energy = Array1::from_shape_fn(110, |index| {
        let i = index as f64;
        -2.0 + 0.018 * i + 0.000_11 * i * i
    });
    let sfconv_spectral = Array2::from_shape_fn((8, 110).f(), |(row, column)| {
        let fortran_row = row as f64 + 1.0;
        let i = column as f64;
        0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
    });
    let sfconv_pole_energy = Array1::from_shape_fn(5, |index| {
        let i = index as f64 + 1.0;
        0.12 * i + 0.015 * i * i
    });
    let sfconv_pole_weight = Array1::from_shape_fn(5, |index| 0.25 + 0.07 * (index as f64 + 1.0));
    let sfconv_pole_broadening = Array1::from_shape_fn(5, |index| {
        let i = index as f64 + 1.0;
        0.01 * i + 0.002 * i * i
    });
    c.bench_function("sfconv_plset_ppset", |b| {
        b.iter(|| {
            let pole = black_box(sfconv_select_pole(
                black_box(3),
                black_box(sfconv_pole_energy.view()),
                black_box(sfconv_pole_weight.view()),
                black_box(sfconv_pole_broadening.view()),
            ));
            let plasma = black_box(sfconv_plasma_parameters(black_box(2.35)));
            black_box((pole, plasma))
        });
    });
    c.bench_function("sfconv_ppole_qlimits", |b| {
        b.iter(|| {
            let dispersion = black_box(sfconv_pole_dispersion(
                black_box(0.35),
                black_box(0.47),
                black_box(0.28),
            ));
            let limits = black_box(sfconv_q_limits(
                black_box(1.15),
                black_box(1.05),
                black_box(0.47),
                black_box(0.28),
                black_box(12.0),
            ));
            let threshold = black_box(sfconv_plasmon_threshold_momentum(
                black_box(0.47),
                black_box(0.28),
                black_box(0.42),
                black_box(0.88),
            ));
            black_box((dispersion, limits, threshold))
        });
    });
    c.bench_function("sfconv_so2conv_momentum_grid", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_momentum_grid(
                black_box(0.816_663_103_267_026_7),
                black_box(1.733_25),
            ))
        });
    });
    c.bench_function("sfconv_so2conv_material_parameters", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_material_parameters(black_box(
                SfconvSo2convMaterialInput {
                    core_hole_width_ev: 1.729,
                    wigner_seitz_radius: 2.05,
                    interstitial_potential_ev: 12.34,
                    chemical_potential_ev: 18.76,
                    fermi_wave_number_inv_angstrom: 1.23,
                },
            )))
        });
    });
    let photoelectron_momentum_grid = array![0.0, 0.35, -0.40, 0.82, 1.10, 1.45];
    let photoelectron_self_energy = array![0.090, 0.105, 0.120, 0.150, 0.190, 0.250];
    c.bench_function("sfconv_so2conv_photoelectron_momentum", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_photoelectron_momentum(black_box(
                SfconvPhotoelectronMomentumInput {
                    momentum: photoelectron_momentum_grid.view(),
                    chemical_potential: 0.47,
                    fermi_momentum: 0.92,
                    fermi_level: 0.36,
                    fermi_self_energy: 0.115,
                    self_energy: photoelectron_self_energy.view(),
                },
            )))
        });
    });
    let exafs_padding_energy = array![0.10, 0.22, 0.37, 0.55];
    c.bench_function("sfconv_so2conv_pad_exafs_energy_grid", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_pad_exafs_energy_grid(black_box(
                SfconvSo2convExafsEnergyPaddingInput {
                    energy: exafs_padding_energy.view(),
                    active_len: 4,
                    output_len: 401,
                },
            )))
        });
    });
    let exafs_prep_count = 112;
    let exafs_prep_momentum = Array1::from_shape_fn(exafs_prep_count, |index| 0.02 * index as f64);
    let exafs_prep_magnitude =
        Array1::from_shape_fn(exafs_prep_count, |index| 1.0 + 0.001 * index as f64);
    let exafs_prep_phase = Array1::from_shape_fn(exafs_prep_count, |index| 0.02 * index as f64);
    c.bench_function("sfconv_so2conv_prepare_exafs_signal", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_prepare_exafs_signal(black_box(
                SfconvSo2convExafsPreparationInput {
                    momentum: exafs_prep_momentum.view(),
                    magnitude: exafs_prep_magnitude.view(),
                    phase: exafs_prep_phase.view(),
                    phase_minus_2kr: None,
                    chemical_potential: 0.5,
                    active_len: exafs_prep_count,
                    output_len: 401,
                },
            )))
        });
    });
    let xanes_prep_count = 112;
    let xanes_prep_incident = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        0.2 + 0.13 * (i - 1.0) + 0.002 * ((i as usize) % 3) as f64
    });
    let xanes_prep_energy = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        -0.4 + 0.11 * (i - 1.0) + 0.001 * ((i as usize) % 4) as f64
    });
    let xanes_prep_background = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        1.0 + 0.015 * (i - 1.0) + 0.0008 * ((i as usize) % 2) as f64
    });
    let xanes_prep_absorption = Array1::from_shape_fn(xanes_prep_count, |index| {
        let i = index as f64 + 1.0;
        xanes_prep_background[index] + 0.04 * (0.31 * i).sin() + 0.002 * (i - 1.0)
    });
    c.bench_function("sfconv_so2conv_prepare_xanes_signal", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_prepare_xanes_signal(black_box(
                SfconvSo2convXanesPreparationInput {
                    incident_energy: xanes_prep_incident.view(),
                    excitation_energy: xanes_prep_energy.view(),
                    absorption: xanes_prep_absorption.view(),
                    embedded_background: xanes_prep_background.view(),
                    active_len: xanes_prep_count,
                    output_len: 401,
                },
            )))
        });
    });
    let momentum_spectral_grid = array![0.50, 1.00, 2.00, 4.00];
    let momentum_spectral_energy = array![
        [0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22, 0.23, 0.24],
        [0.31, 0.32, 0.33, 0.34],
        [0.41, 0.42, 0.43, 0.44],
    ];
    let momentum_spectral_emsf = array![
        [1.11, 1.12, 1.13, 1.14],
        [1.21, 1.22, 1.23, 1.24],
        [1.31, 1.32, 1.33, 1.34],
        [1.41, 1.42, 1.43, 1.44],
    ];
    let momentum_spectral_essf = array![
        [2.22, 2.24, 2.26, 2.28],
        [2.42, 2.44, 2.46, 2.48],
        [2.62, 2.64, 2.66, 2.68],
        [2.82, 2.84, 2.86, 2.88],
    ];
    let momentum_spectral_xmsf = array![
        [3.33, 3.36, 3.39, 3.42],
        [3.63, 3.66, 3.69, 3.72],
        [3.93, 3.96, 3.99, 4.02],
        [4.23, 4.26, 4.29, 4.32],
    ];
    let momentum_spectral_xssf = array![
        [0.444, 0.448, 0.452, 0.456],
        [0.484, 0.488, 0.492, 0.496],
        [0.524, 0.528, 0.532, 0.536],
        [0.564, 0.568, 0.572, 0.576],
    ];
    let momentum_spectral_xissf = array![
        [0.555, 0.560, 0.565, 0.570],
        [0.605, 0.610, 0.615, 0.620],
        [0.655, 0.660, 0.665, 0.670],
        [0.705, 0.710, 0.715, 0.720],
    ];
    let momentum_spectral_escsf = array![
        [0.666, 0.672, 0.678, 0.684],
        [0.726, 0.732, 0.738, 0.744],
        [0.786, 0.792, 0.798, 0.804],
        [0.846, 0.852, 0.858, 0.864],
    ];
    let momentum_spectral_weights = array![
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
    ];
    let momentum_spectral_self = array![41.0, 42.0, 43.0, 44.0];
    let momentum_spectral_correction = array![51.0, 52.0, 53.0, 54.0];
    let momentum_spectral_width = array![61.0, 62.0, 63.0, 64.0];
    let momentum_spectral_z1 = array![71.0, 72.0, 73.0, 74.0];
    let momentum_spectral_z1i = array![81.0, 82.0, 83.0, 84.0];
    c.bench_function("sfconv_so2conv_momentum_spectral_interpolation", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_momentum_spectral_function(black_box(
                SfconvMomentumSpectralInterpolationInput {
                    photoelectron_momentum: 0.75,
                    momentum_grid: momentum_spectral_grid.view(),
                    energy_grid: momentum_spectral_energy.view(),
                    extrinsic_quasiparticle: momentum_spectral_emsf.view(),
                    extrinsic_satellite: momentum_spectral_essf.view(),
                    interference_quasiparticle: momentum_spectral_xmsf.view(),
                    interference_satellite: momentum_spectral_xssf.view(),
                    intrinsic_satellite: momentum_spectral_xissf.view(),
                    clipped_extrinsic_satellite: momentum_spectral_escsf.view(),
                    weights: momentum_spectral_weights.view(),
                    self_energy_real: momentum_spectral_self.view(),
                    energy_correction: momentum_spectral_correction.view(),
                    width: momentum_spectral_width.view(),
                    renormalization_real: momentum_spectral_z1.view(),
                    renormalization_imag: momentum_spectral_z1i.view(),
                },
            )))
        });
    });
    let path_interp_source = array![0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00];
    let path_interp_momentum = array![0.25, 0.75, 1.25, 1.75];
    let path_interp_central_phase = array![0.10, 0.20, 0.10, 0.30];
    let path_interp_amplitude = array![1.00, 1.40, 1.10, 1.80];
    let path_interp_phase = array![0.50, 0.70, 0.60, 1.00];
    let path_interp_reduction = array![0.80, 0.90, 0.85, 0.95];
    let path_interp_lambda = array![6.00, 7.00, 8.00, 9.00];
    c.bench_function("sfconv_so2conv_path_interpolation", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_feff_path(black_box(
                SfconvFeffPathInterpolationInput {
                    source_momentum: path_interp_source.view(),
                    path_momentum: path_interp_momentum.view(),
                    central_phase: path_interp_central_phase.view(),
                    effective_amplitude: path_interp_amplitude.view(),
                    effective_phase: path_interp_phase.view(),
                    reduction_factor: path_interp_reduction.view(),
                    mean_free_path: path_interp_lambda.view(),
                },
            )))
        });
    });
    let path_signal_central_phase = array![0.0, 0.10, 0.15, 0.20, 0.15, 0.10, 0.20, 0.30, 0.0];
    let path_signal_amplitude = array![0.0, 1.00, 1.20, 1.40, 1.25, 1.10, 1.45, 1.80, 0.0];
    let path_signal_phase = array![0.0, 0.50, 0.60, 0.70, 0.65, 0.60, 0.80, 1.00, 0.0];
    let path_signal_reduction = array![0.0, 0.80, 0.85, 0.90, 0.875, 0.85, 0.90, 0.95, 0.0];
    let path_signal_lambda = array![0.0, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 0.0];
    c.bench_function("sfconv_so2conv_path_signal", |b| {
        b.iter(|| {
            black_box(sfconv_feff_path_signal(black_box(
                SfconvFeffPathSignalInput {
                    momentum: path_interp_source.view(),
                    central_phase: path_signal_central_phase.view(),
                    effective_amplitude: path_signal_amplitude.view(),
                    effective_phase: path_signal_phase.view(),
                    reduction_factor: path_signal_reduction.view(),
                    mean_free_path: path_signal_lambda.view(),
                    degeneracy: 4.0,
                    half_path_length: 3.25,
                },
            )))
        });
    });
    c.bench_function("sfconv_so2conv_exafs_convolution", |b| {
        b.iter(|| {
            black_box(sfconv_exafs_convolution(black_box(
                SfconvExafsConvolutionInput {
                    real_convolution_amplitude: -1.494_388_190_129_498_7,
                    real_convolution_phase: 0.0,
                    imaginary_convolution_amplitude: -0.137_577_673_742_690_1,
                    imaginary_convolution_phase: 0.0,
                    original_magnitude: 1.7,
                    original_phase: 0.25,
                    phase_minus_2kr: 0.03,
                    previous_phase: 3.050_020_434_612_271,
                    phase_jump_count: 0,
                },
            )))
        });
    });
    c.bench_function("sfconv_so2conv_xanes_convolution", |b| {
        b.iter(|| {
            black_box(sfconv_xanes_convolution(black_box(
                SfconvXanesConvolutionInput {
                    asymmetric_phase: false,
                    absorption_convolution: 0.0,
                    embedded_background: 3.40,
                    fine_structure_imaginary_amplitude: 1.80,
                    fine_structure_imaginary_phase: 0.20,
                    fine_structure_real_amplitude: 0.70,
                    fine_structure_real_phase: 0.90,
                },
            )))
        });
    });
    c.bench_function("sfconv_grater_oscillatory", |b| {
        b.iter(|| {
            black_box(sfconv_grater_integrate(
                |x| Ok((5.0 * x).sin() / (1.0 + x * x)),
                black_box(0.0),
                black_box(4.0),
                black_box(1.0e-6),
                black_box(1.0e-6),
                black_box(&[]),
            ))
        });
    });
    c.bench_function("sfconv_mkspectf_energy_grid", |b| {
        b.iter(|| black_box(sfconv_spectral_energy_grid(black_box(0.62))));
    });
    let quasiparticle_peak_input = SfconvQuasiparticlePeakInput {
        center_energy: -0.000_206_666_666_666_666_66,
        lower_boundary: -0.000_31,
        upper_boundary: 0.0,
        photoelectron_energy: 0.93,
        quasiparticle_energy: 0.9348,
        quasiparticle_width: 0.0656,
        plasma_frequency: 0.62,
        renormalization_real: 0.82,
        renormalization_imag: 0.06,
    };
    c.bench_function("sfconv_mkspectf_quasiparticle_peak", |b| {
        b.iter(|| {
            black_box(sfconv_quasiparticle_main_peak(black_box(
                quasiparticle_peak_input,
            )))
        });
    });
    let quasiparticle_table_energy = array![-0.40, -0.12, -0.01, 0.02, 0.20, 0.55];
    let quasiparticle_table_boundaries = array![-0.55, -0.25, -0.05, 0.005, 0.10, 0.36, 0.80];
    c.bench_function("sfconv_mkspectf_quasiparticle_table", |b| {
        b.iter(|| {
            black_box(sfconv_quasiparticle_table(black_box(
                SfconvQuasiparticleTableInput {
                    energy: quasiparticle_table_energy.view(),
                    boundaries: quasiparticle_table_boundaries.view(),
                    photoelectron_energy: 0.93,
                    quasiparticle_energy: 0.944,
                    endpoint_width: 0.073,
                    quasiparticle_width: 0.073 * 0.82,
                    plasma_frequency: 0.62,
                    renormalization_real: 0.82,
                    renormalization_imag: 0.06,
                    renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
                    interference_amplitude: 0.135,
                    exponential_reduction: 0.74,
                },
            )))
        });
    });
    let satellite_main_peak = array![
        0.144_118_631_068_914_32,
        0.796_854_020_052_775_2,
        3.306_037_878_829_96,
        2.944_827_731_705_054,
        0.351_606_691_790_681_77,
        0.027_414_131_538_569_52,
    ];
    let satellite_quasiparticle_interference = array![
        0.031_993_167_546_517_99,
        0.176_895_131_355_183_62,
        0.733_913_602_898_189_5,
        0.653_727_879_020_868,
        0.078_053_834_660_399_79,
        0.006_085_714_920_760_973,
    ];
    let satellite_extrinsic = array![0.04, 0.09, -0.02, 0.18, 0.13, 0.07];
    let satellite_interference = array![0.01, 0.025, 0.006, 0.055, 0.04, 0.015];
    let satellite_intrinsic = array![0.02, 0.035, 0.012, 0.08, 0.065, 0.025];
    c.bench_function("sfconv_mkspectf_satellite_table", |b| {
        b.iter(|| {
            black_box(sfconv_satellite_table(black_box(
                SfconvSatelliteTableInput {
                    main_peak: satellite_main_peak.view(),
                    quasiparticle_interference: satellite_quasiparticle_interference.view(),
                    extrinsic_satellite: satellite_extrinsic.view(),
                    interference_satellite: satellite_interference.view(),
                    intrinsic_satellite: satellite_intrinsic.view(),
                    boundaries: quasiparticle_table_boundaries.view(),
                    quasiparticle_lower_column_1based: 3,
                    quasiparticle_upper_column_1based: 4,
                    include_full_broadening_quasiparticle: true,
                    exponential_reduction: 0.74,
                },
            )))
        });
    });
    let mut split_table = Array2::<f64>::zeros((8, 8).f());
    for (row, values) in [
        (1, [0.10, 0.18, 0.35, 0.30, 0.22, 0.15, 0.25, 0.20]),
        (4, [0.02, 0.05, 0.11, 0.16, 0.13, 0.09, 0.12, 0.07]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            split_table[(row, column)] = value;
        }
    }
    let split_energy = array![-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0];
    let split_boundaries = array![-0.75, -0.45, -0.20, -0.05, 0.05, 0.20, 0.45, 0.80, 1.20];
    c.bench_function("sfconv_mkspectf_extrinsic_satellite_split", |b| {
        b.iter(|| {
            black_box(sfconv_split_extrinsic_satellite(black_box(
                SfconvExtrinsicSatelliteSplitInput {
                    spectral_function: split_table.view(),
                    energy: split_energy.view(),
                    boundaries: split_boundaries.view(),
                    photoelectron_energy: 0.05,
                    beta_zero: 1.0,
                },
            )))
        });
    });
    let mut satellite_table = Array2::<f64>::zeros((8, 6).f());
    for (row, values) in [
        (1, [0.40, 0.18, 0.06, 0.50, 0.28, 0.08]),
        (3, [0.10, 0.16, 0.08, 0.35, 0.05, 0.03]),
        (4, [0.05, 0.04, 0.20, 0.03, 0.30, 0.20]),
        (6, [0.08, 0.05, 0.03, 0.12, 0.07, 0.02]),
        (7, [0.04, 0.02, 0.01, 0.06, 0.09, 0.03]),
    ] {
        for (column, value) in values.into_iter().enumerate() {
            satellite_table[(row, column)] = value;
        }
    }
    let satellite_boundaries = array![-0.4, -0.2, 0.0, 0.15, 0.35, 0.7, 1.1];
    c.bench_function("sfconv_mkspectf_satellite_correction", |b| {
        b.iter(|| {
            black_box(sfconv_correct_satellite_weights(black_box(
                SfconvSatelliteCorrectionInput {
                    spectral_function: satellite_table.view(),
                    boundaries: satellite_boundaries.view(),
                    uniform_width: 0.2,
                    exponential_reduction: 0.73,
                },
            )))
        });
    });
    let spectral_satellite_weights = array![0.259_15, 0.119_355, 0.174_47, 0.036_5, 0.054_02];
    c.bench_function("sfconv_mkspectf_spectral_weights", |b| {
        b.iter(|| {
            black_box(sfconv_spectral_weights(black_box(
                SfconvSpectralWeightsInput {
                    renormalization_real: 0.82,
                    renormalization_imag: 0.06,
                    renormalization_magnitude: (0.82_f64.powi(2) + 0.06_f64.powi(2)).sqrt(),
                    interference_amplitude: 0.135,
                    interference_reduction: 0.43,
                    exponential_reduction: 0.74,
                    satellite_weights: spectral_satellite_weights.view(),
                },
            )))
        });
    });
    let path_average_source = array![0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25];
    let path_average_amplitude = array![0.82, 0.84, 0.88, 0.91, 0.89, 0.86, 0.83];
    let path_average_phase = array![0.05, 0.08, 0.13, 0.17, 0.14, 0.09, 0.02];
    c.bench_function("sfconv_so2conv_path_average", |b| {
        b.iter(|| {
            black_box(sfconv_path_average(black_box(SfconvPathAverageInput {
                source_momentum: path_average_source.view(),
                amplitude_reduction: path_average_amplitude.view(),
                phase_shift: path_average_phase.view(),
                previous_momentum: 1.00,
                center_momentum: 1.60,
                next_momentum: 2.30,
                momentum_step: 0.05,
            })))
        });
    });
    let senergies_context = SfconvSelfEnergyContext {
        fermi_energy: 0.50,
        fermi_momentum: 1.00,
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        quasiparticle_energy: 0.91,
        photoelectron_momentum: (2.0_f64 * 0.85).sqrt(),
        accuracy: 1.0e-4,
        pole_broadening: 0.035,
        dispersion_parameter: 0.28,
        include_below_fermi: false,
    };
    c.bench_function("sfconv_senergies_beta", |b| {
        b.iter(|| {
            let beta = black_box(sfconv_extrinsic_beta(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let imaginary = black_box(sfconv_imaginary_self_energy(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let real = black_box(sfconv_real_self_energy(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let real_derivative = black_box(sfconv_real_self_energy_derivative(
                black_box(0.36),
                black_box(senergies_context),
            ));
            let imaginary_derivative = black_box(sfconv_imaginary_self_energy_derivative(
                black_box(0.36),
                black_box(senergies_context),
            ));
            black_box((beta, imaginary, real, real_derivative, imaginary_derivative))
        });
    });
    let satellite_context = SfconvSatelliteContext {
        plasma_frequency: 0.62,
        pole_energy: 0.47,
        dispersion_parameter: 0.28,
        photoelectron_energy: 0.85,
        accuracy: 1.0e-4,
    };
    c.bench_function("sfconv_mksat_satellites", |b| {
        b.iter(|| {
            let interference = black_box(sfconv_interference_satellite(
                black_box(0.75),
                black_box(0.045),
                black_box(satellite_context),
            ));
            let intrinsic = black_box(sfconv_intrinsic_satellite(
                black_box(0.75),
                black_box(0.045),
                black_box(satellite_context),
            ));
            black_box((interference, intrinsic))
        });
    });
    c.bench_function("sfconv_interpsf_512_points", |b| {
        b.iter(|| {
            black_box(sfconv_interpolate_spectral_function(black_box(
                SfconvSpectralInterpolationInput {
                    energy: sfconv_energy.view(),
                    spectral_function: sfconv_spectral.view(),
                    output_len: black_box(512),
                },
            )))
        });
    });
    c.bench_function("integrated_double_lorentz", |b| {
        b.iter(|| {
            black_box(integrated_double_lorentz(
                black_box(3.1),
                black_box(2.7),
                black_box(0.45),
                black_box(0.3),
                black_box(1.2),
                black_box(-0.08),
                black_box(Some(5.0)),
            ))
        });
    });
    c.bench_function("kk_integral", |b| {
        b.iter(|| {
            black_box(kk_integral(
                black_box(Complex::new(0.7, -0.2)),
                black_box(Complex::new(1.1, 0.3)),
                black_box(-1.0),
                black_box(2.0),
                black_box(0.25),
                black_box(0.4),
            ))
        });
    });
    let rixs_x = ndarray::arr1(&[0.0, 1.0, 2.5]);
    let rixs_y = ndarray::arr1(&[-1.0, 0.5, 2.0, 4.0]);
    let rixs_values = Array2::from_shape_fn((rixs_x.len(), rixs_y.len()).f(), |(row, col)| {
        let fortran_row = row as f64 + 1.0;
        let fortran_col = col as f64 + 1.0;
        Complex::new(
            10.0 * fortran_row + fortran_col,
            -1.5 * fortran_row + 0.25 * fortran_col,
        )
    });
    c.bench_function("bilinear_interpolate_complex", |b| {
        b.iter(|| {
            black_box(bilinear_interpolate_complex(
                black_box(rixs_x.view()),
                black_box(rixs_y.view()),
                black_box(rixs_values.view()),
                black_box(0.4),
                black_box(1.1),
            ))
        });
    });
    c.bench_function("morse_einstein_cumulants", |b| {
        b.iter(|| {
            black_box(morse_einstein_cumulants(
                black_box(0.003),
                black_box(300.0),
                black_box(1.0e-5),
                black_box(400.0),
            ))
        });
    });
    c.bench_function("thermal_expansion_cumulants", |b| {
        b.iter(|| {
            black_box(thermal_expansion_cumulants(
                black_box(29),
                black_box(29),
                black_box(0.003),
                black_box(1.0e-5),
                black_box(400.0),
                black_box(2.55),
            ))
        });
    });
    c.bench_function("quantum_debye_correlation", |b| {
        b.iter(|| {
            black_box(quantum_debye_correlation(
                black_box(2.55),
                black_box(400.0),
                black_box(300.0),
                black_box(29),
                black_box(29),
                black_box(2.7),
            ))
        });
    });
    c.bench_function("classical_debye_correlation", |b| {
        b.iter(|| {
            black_box(classical_debye_correlation(
                black_box(2.55),
                black_box(400.0),
                black_box(300.0),
                black_box(29),
                black_box(29),
                black_box(2.7),
            ))
        });
    });
    let debye_path = Array2::from_shape_fn((3, 3), |(row, col)| match (row, col) {
        (1, 0) => 2.55,
        _ => 0.0,
    });
    let debye_atomic_numbers = [29, 29, 29];
    c.bench_function("quantum_debye_waller_factor", |b| {
        b.iter(|| {
            black_box(quantum_debye_waller_factor(
                black_box(300.0),
                black_box(400.0),
                black_box(2.7),
                black_box(debye_path.view()),
                black_box(&debye_atomic_numbers),
            ))
        });
    });
    let dmdw_positions = Array2::from_shape_fn((48, 3), |(atom, component)| {
        let shell = atom / 12;
        let slot = atom % 12;
        match component {
            0 => shell as f64 * 1.7 + (slot % 3) as f64 * 0.9,
            1 => (slot / 3) as f64 * 1.1 - shell as f64 * 0.2,
            _ => (slot % 2) as f64 * 1.3 + shell as f64 * 0.4,
        }
    });
    let dmdw_descriptor = DmdwPathDescriptor {
        selectors: vec![1, 0, 0],
        max_effective_length: 7.0,
    };
    c.bench_function("dmdw_expand_path_descriptor_triple_wildcards", |b| {
        b.iter(|| {
            black_box(dmdw_expand_path_descriptor(
                black_box(dmdw_positions.view()),
                black_box(&dmdw_descriptor),
            ))
        });
    });
    let dmdw_pole_frequencies = Array1::from_shape_fn(64, |index| {
        if index % 17 == 0 {
            -0.25 * (index as f64 + 1.0)
        } else {
            1.0 + index as f64 * 0.125
        }
    });
    let dmdw_pole_weights = Array1::from_shape_fn(64, |index| 1.0 / (64.0 + index as f64));
    c.bench_function("dmdw_moment_summaries_from_poles_64", |b| {
        b.iter(|| {
            black_box(dmdw_moment_summaries_from_poles(
                black_box(31.773),
                black_box(dmdw_pole_frequencies.view()),
                black_box(dmdw_pole_weights.view()),
            ))
        });
    });
    let dmdw_type2_masses = arr1(&[63.546, 63.546, 63.546]);
    let mut dmdw_type2_force_blocks = Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            dmdw_type2_force_blocks[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as f64 + 0.001 * component as f64;
        }
    }
    for component in 0..3 {
        dmdw_type2_force_blocks[(0, 1, component, component)] = -0.004;
        dmdw_type2_force_blocks[(1, 0, component, component)] = -0.004;
        dmdw_type2_force_blocks[(1, 2, component, component)] = -0.003;
        dmdw_type2_force_blocks[(2, 1, component, component)] = -0.003;
    }
    let dmdw_type2_groups = vec![DmdwType2AtomGroup {
        center_atom_indices: vec![0],
    }];
    let dmdw_type2_coupling = DmdwPhononCoupling {
        energy_hartree: arr1(&[0.001, 0.002, 0.004]),
        energy_ev: arr1(&[0.027_211_396_132, 0.054_422_792_264, 0.108_845_584_528]),
        eliashberg: arr1(&[0.5, 1.0, 1.5]),
        matrix_element: arr1(&[0.05, 0.05, 0.05]),
        normalization: 1.0,
    };
    c.bench_function("dmdw_type2_a2f_single_group_order1", |b| {
        b.iter(|| {
            black_box(dmdw_type2_pole_weighted_a2f(
                black_box(dmdw_type2_force_blocks.view()),
                black_box(dmdw_type2_masses.view()),
                black_box(&dmdw_type2_groups),
                black_box(0),
                black_box(1),
                black_box(&dmdw_type2_coupling),
            ))
        });
    });
    let dmdw_a2f = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: Array1::from_shape_fn(32, |index| 2.0 + index as f64 * 0.5),
        lanczos_weight: Array1::from_shape_fn(32, |index| 1.0 / (32.0 + index as f64)),
        normalization: 1.0,
        pole_energy_ev: Array1::from_shape_fn(32, |index| 0.008 + index as f64 * 0.0015),
        pole_weight: Array1::from_shape_fn(32, |index| 0.02 / (1.0 + index as f64 * 0.1)),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.024,
    };
    let dmdw_self_energy_grid = Array1::from_shape_fn(257, |index| -0.18 + index as f64 * 0.0015);
    c.bench_function("dmdw_self_energy_grid_a2f_257x32", |b| {
        b.iter(|| {
            black_box(dmdw_self_energy_grid_from_a2f_poles(
                black_box(300.0),
                black_box(dmdw_self_energy_grid.view()),
                black_box(&dmdw_a2f),
            ))
        });
    });
    let dmdw_spectral_grid = Array1::from_shape_fn(65, |index| -2.0 + index as f64 * 0.0625);
    c.bench_function("dmdw_spectral_function_a2f_65x65x32", |b| {
        b.iter(|| {
            black_box(dmdw_spectral_function_from_a2f_poles(
                black_box(300.0),
                black_box(dmdw_spectral_grid.view()),
                black_box(0.0),
                black_box(dmdw_a2f.characteristic_energy_ev),
                black_box(&dmdw_a2f),
                black_box(20.0),
                black_box(65),
            ))
        });
    });
}

fn sample_epsilon_tables() -> Vec<EpsilonTable> {
    (0..3)
        .map(|table| {
            let table_f = table as f64;
            let energy_ev =
                Array1::from_shape_fn(160, |index| 0.02 + 0.018 * index as f64 + table_f * 0.003);
            let epsilon1_minus_one = energy_ev
                .mapv(|energy| (0.2 + 0.03 * table_f) * (-0.4 * energy).exp() + 0.01 * energy);
            let epsilon2 =
                energy_ev.mapv(|energy| (0.08 + 0.01 * table_f) * (1.0 + energy).recip());
            EpsilonTable {
                energy_ev,
                epsilon1_minus_one,
                epsilon2,
            }
        })
        .collect()
}
