use super::*;

#[path = "scalar/debye_rixs.rs"]
mod debye_rixs;
#[path = "scalar/eels.rs"]
mod eels;
#[path = "scalar/fullspectrum.rs"]
mod fullspectrum;
#[path = "scalar/physics.rs"]
mod physics;
#[path = "scalar/sfconv.rs"]
mod sfconv;

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

    eels::bench_eels_helpers(c);
    fullspectrum::bench_fullspectrum_helpers(c);
    physics::bench_physics_helpers(c);
    sfconv::bench_sfconv_helpers(c);
    debye_rixs::bench_debye_rixs_helpers(c);
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
