use super::*;

pub(super) fn bench_rhorrp_helpers(c: &mut Criterion) {
    let axes = arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]]);
    let points_per_axis = [40, 30, 20];
    c.bench_function("rhorrp_density_grid_points_40x30x20", |b| {
        b.iter(|| {
            black_box(rhorrp_density_grid_points(black_box(
                RhorrpDensityGridInput {
                    origin: [0.1, -0.2, 0.3],
                    axes: axes.view(),
                    points_per_axis: &points_per_axis,
                },
            )))
        });
    });
    c.bench_function("rhorrp_evaluate_density_grid_40x30x20", |b| {
        b.iter(|| {
            black_box(rhorrp_evaluate_density_grid(
                black_box(RhorrpDensityGridInput {
                    origin: [0.1, -0.2, 0.3],
                    axes: axes.view(),
                    points_per_axis: &points_per_axis,
                }),
                |point| Ok(point[0] + 2.0 * point[1] - 0.5 * point[2] + point[0] * point[1]),
            ))
        });
    });
    c.bench_function("rhorrp_process_ranges_1000000x64", |b| {
        b.iter(|| black_box(rhorrp_process_ranges(black_box(1_000_000), black_box(64))));
    });

    let positions = Array2::from_shape_fn((128, 3), |(atom, axis)| {
        let atom = atom as f64;
        match axis {
            0 => (atom * 0.37).sin(),
            1 => (atom * 0.23).cos(),
            _ => atom * 0.015,
        }
    });
    let potentials = (0..128).map(|atom| atom % 5).collect::<Vec<_>>();
    let representative_atoms = [0, 7, 31, 63, 95, 127];
    c.bench_function("rhorrp_fms_inclusion_counts_6x128", |b| {
        b.iter(|| {
            black_box(rhorrp_fms_inclusion_counts(black_box(
                RhorrpFmsInclusionInput {
                    atom_positions: positions.view(),
                    representative_atoms: &representative_atoms,
                    fms_radius: 1.25,
                },
            )))
        });
    });
    c.bench_function("rhorrp_nearest_atom_128", |b| {
        b.iter(|| {
            black_box(rhorrp_nearest_atom(black_box(RhorrpNearestAtomInput {
                point: [0.25, -0.15, 0.75],
                atom_positions: positions.view(),
                atom_potentials: &potentials,
                fms_atom_count: None,
            })))
        });
    });
    let nearest_points = Array2::from_shape_fn((3, 4096), |(axis, point)| {
        let point = point as f64;
        match axis {
            0 => 0.001 * point,
            1 => (0.003 * point).sin(),
            _ => (0.002 * point).cos(),
        }
    });
    c.bench_function("rhorrp_nearest_atom_table_4096x128", |b| {
        b.iter(|| {
            black_box(rhorrp_nearest_atom_table(black_box(
                RhorrpNearestAtomTableInput {
                    points: nearest_points.view(),
                    atom_positions: positions.view(),
                    atom_potentials: &potentials,
                    fms_atom_count: None,
                },
            )))
        });
    });

    c.bench_function("rhorrp_radial_interpolation_location", |b| {
        b.iter(|| {
            black_box(rhorrp_radial_interpolation_location(black_box(
                RhorrpRadialInterpolationInput {
                    radius: 0.522_045_776_761_016,
                    x0: 0.7,
                    dx: 0.2,
                    radial_count: 251,
                },
            )))
        });
    });

    c.bench_function("rhorrp_energy_prefactor", |b| {
        b.iter(|| {
            black_box(rhorrp_energy_prefactor(black_box(
                RhorrpEnergyPrefactorInput {
                    energy_hartree: Complex::new(0.2, 0.05),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                },
            )))
        });
    });

    let finish_energies = Array1::from_shape_fn(256, |index| {
        let index = index as f64;
        Complex::new(-0.1 + 0.006 * index, 0.02 * (0.05 * index).sin())
    });
    let finish_green = Array1::from_shape_fn(256, |index| {
        let index = index as f64;
        Complex::new(0.001 * (0.03 * index).cos(), -0.0007 * (0.02 * index).sin())
    });
    c.bench_function("rhorrp_finish_energy_density_256", |b| {
        b.iter(|| {
            black_box(rhorrp_finish_energy_density(black_box(
                RhorrpEnergyDensityInput {
                    energies_hartree: finish_energies.view(),
                    green_function: finish_green.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    radius: 0.85,
                    prime_radius: 1.25,
                },
            )))
        });
    });

    let same_regular_large = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            0.001 * energy + 0.03 * angular + (0.01 * radial).sin(),
            -0.0007 * energy + 0.02 * angular - (0.008 * radial).cos(),
        )
    });
    let same_irregular_large = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            -0.0008 * energy + 0.04 * angular + (0.012 * radial).cos(),
            0.0005 * energy - 0.01 * angular + (0.009 * radial).sin(),
        )
    });
    let same_regular_small = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            0.0007 * energy - 0.02 * angular + (0.006 * radial).sin(),
            0.0004 * energy + 0.015 * angular - (0.011 * radial).cos(),
        )
    });
    let same_irregular_small = Array3::from_shape_fn((64, 4, 96), |(energy, angular, radial)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        let radial = (radial + 1) as f64;
        Complex::new(
            -0.0003 * energy + 0.025 * angular - (0.007 * radial).cos(),
            0.0002 * energy + 0.018 * angular + (0.005 * radial).sin(),
        )
    });
    c.bench_function("rhorrp_same_site_green_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_same_site_green(black_box(
                RhorrpSameSiteGreenInput {
                    regular_large: same_regular_large.view(),
                    irregular_large: same_irregular_large.view(),
                    regular_small: same_regular_small.view(),
                    irregular_small: same_irregular_small.view(),
                    first_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 34,
                        fraction: 0.25,
                    },
                    second_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 61,
                        fraction: 0.60,
                    },
                    cosine_between: 0.35,
                },
            )))
        });
    });

    let scattering_phase = Array2::from_shape_fn((64, 4), |(energy, angular)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        Complex::new(
            0.00015 * energy + 0.04 * angular,
            -0.00006 * energy + 0.02 * angular,
        )
    });
    let scattering_phase_prime = Array2::from_shape_fn((64, 4), |(energy, angular)| {
        let energy = (energy + 1) as f64;
        let angular = angular as f64;
        Complex::new(
            -0.00011 * energy + 0.03 * angular,
            0.00007 * energy - 0.015 * angular,
        )
    });
    let scattering_matrix = Array3::from_shape_fn((64, 16, 16), |(energy, row, column)| {
        let energy = (energy + 1) as f64;
        let row = (row + 1) as f64;
        let column = (column + 1) as f64;
        Complex::new(
            0.00002 * energy + 0.0004 * row - 0.0003 * column,
            -0.000015 * energy + 0.00025 * row + 0.0001 * column,
        )
    });
    c.bench_function("rhorrp_scattering_green_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_scattering_green(black_box(
                RhorrpScatteringGreenInput {
                    first_regular_large: same_regular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: scattering_matrix.view(),
                    first_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 34,
                        fraction: 0.25,
                    },
                    second_location: RhorrpRadialInterpolationLocation {
                        index_below_1based: 61,
                        fraction: 0.60,
                    },
                    first_displacement: [0.4, -0.2, 0.6],
                    second_displacement: [-0.3, 0.5, 0.7],
                },
            )))
        });
    });

    let pair_energies = Array1::from_shape_fn(64, |index| {
        let index = index as f64;
        Complex::new(-0.08 + 0.004 * index, 0.015 * (0.05 * index).cos())
    });
    c.bench_function("rhorrp_pair_energy_density_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_pair_energy_density(black_box(
                RhorrpPairEnergyDensityInput {
                    energies_hartree: pair_energies.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    first_regular_large: same_regular_large.view(),
                    first_irregular_large: same_irregular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    first_irregular_small: same_irregular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: Some(scattering_matrix.view()),
                    same_atom: true,
                    first_displacement: [0.22, -0.18, 0.44],
                    second_displacement: [-0.31, 0.28, 0.36],
                    radial_x0: 0.7,
                    radial_dx: 0.2,
                    radial_count: 96,
                },
            )))
        });
    });

    let wavefunctions = Array3::from_shape_fn((192, 5, 251), |(energy, angular, radial)| {
        let real = 0.001 * energy as f64 + 0.01 * angular as f64 + (0.002 * radial as f64).sin();
        let imag = -0.0005 * energy as f64 + 0.005 * angular as f64 - (0.001 * radial as f64).cos();
        Complex::new(real, imag)
    });
    c.bench_function("rhorrp_interpolate_wavefunction_192x5", |b| {
        b.iter(|| {
            black_box(rhorrp_interpolate_wavefunction(black_box(
                RhorrpWavefunctionInterpolationInput {
                    wavefunctions: wavefunctions.view(),
                    index_below_1based: 120,
                    fraction: 0.375,
                },
            )))
        });
    });

    c.bench_function("rhorrp_fermi_distribution_complex", |b| {
        b.iter(|| {
            black_box(rhorrp_fermi_distribution(black_box(
                RhorrpFermiDistributionInput {
                    energy_hartree: Complex::new(0.2, 0.05),
                    chemical_potential_hartree: 0.1,
                    temperature_hartree: 0.025,
                    chemical_potential_override_hartree: Some(0.22),
                },
            )))
        });
    });

    let irregular_radii = (1..=251)
        .map(|index| {
            let index = index as f64;
            0.02 * index + 0.0001 * index * index
        })
        .collect::<Vec<_>>();
    let irregular_values = Array1::from_shape_fn(251, |index| {
        let one_based = (index + 1) as f64;
        Complex::new(
            (0.07 * one_based).sin() + 0.002 * one_based,
            (0.05 * one_based).cos() - 0.001 * one_based,
        )
    });
    c.bench_function("rhorrp_fix_irregular_origin_251", |b| {
        b.iter(|| {
            black_box(rhorrp_fix_irregular_origin(black_box(
                RhorrpIrregularFixInput {
                    radii: &irregular_radii,
                    values: irregular_values.view(),
                },
            )))
        });
    });

    let atomic_radii = (1..=251)
        .map(|index| {
            let index = index as f64;
            0.015 + 0.035 * index + 0.0002 * (index - 1.0) * (index - 1.0)
        })
        .collect::<Vec<_>>();
    let atomic_positions = Array2::from_shape_fn((128, 3), |(atom, axis)| {
        let atom = atom as f64;
        match axis {
            0 => 1.25 * (0.17 * atom).sin(),
            1 => 1.10 * (0.13 * atom).cos(),
            _ => -0.85 + 0.013 * atom,
        }
    });
    let atomic_potentials = (0..128).map(|atom| atom % 5).collect::<Vec<_>>();
    let atomic_large = Array3::from_shape_fn((251, 4, 5), |(radial, orbital, potential)| {
        let index = (radial + 1) as f64;
        (0.017 * index).sin()
            + 0.021 * (orbital + 1) as f64
            + 0.012 * potential as f64
            + 0.03 * atomic_radii[radial]
    });
    let atomic_small = Array3::from_shape_fn((251, 4, 5), |(radial, orbital, potential)| {
        let index = (radial + 1) as f64;
        (0.011 * index).cos() - 0.014 * (orbital + 1) as f64 + 0.009 * potential as f64
            - 0.02 * atomic_radii[radial]
    });
    c.bench_function("rhorrp_atomic_density_128_atoms", |b| {
        b.iter(|| {
            black_box(rhorrp_atomic_density(black_box(RhorrpAtomicDensityInput {
                point: [0.22, -0.15, 0.18],
                orbital_index_1based: 2,
                atom_positions: atomic_positions.view(),
                atom_potentials: &atomic_potentials,
                radii: &atomic_radii,
                large_components: atomic_large.view(),
                small_components: atomic_small.view(),
            })))
        });
    });

    let contour_energies = Array1::from_shape_fn(64, |index| {
        if index < 16 {
            Complex::new(-0.025, 0.075 - 0.005 * index as f64)
        } else if index < 56 {
            Complex::new(-0.025 + 0.006 * (index - 15) as f64, 0.0)
        } else {
            Complex::new(0.045, 0.0035 * std::f64::consts::TAU * (index - 55) as f64)
        }
    });
    let contour_density = Array1::from_shape_fn(64, |index| {
        let energy = contour_energies[index];
        let one_based = (index + 1) as f64;
        Complex::new(
            0.40 + 0.003 * one_based + 0.04 * energy.re - 0.05 * energy.im,
            -0.20 + 0.002 * one_based + 0.03 * energy.re + 0.04 * energy.im,
        )
    });
    c.bench_function("rhorrp_integrate_density_64_energy", |b| {
        b.iter(|| {
            black_box(rhorrp_integrate_density(black_box(
                RhorrpDensityIntegrationInput {
                    energies_hartree: contour_energies.view(),
                    energy_density: contour_density.view(),
                    real_axis_count: 56,
                    chemical_potential_hartree: 0.045,
                    temperature_hartree: 0.0035,
                    chemical_potential_override_hartree: None,
                },
            )))
        });
    });
    c.bench_function("rhorrp_pair_density_64x4", |b| {
        b.iter(|| {
            black_box(rhorrp_pair_density(black_box(RhorrpPairDensityInput {
                pair_energy: RhorrpPairEnergyDensityInput {
                    energies_hartree: contour_energies.view(),
                    reference_energy_hartree: Complex::new(0.03, -0.01),
                    first_regular_large: same_regular_large.view(),
                    first_irregular_large: same_irregular_large.view(),
                    first_regular_small: same_regular_small.view(),
                    first_irregular_small: same_irregular_small.view(),
                    second_regular_large: same_irregular_large.view(),
                    second_regular_small: same_irregular_small.view(),
                    first_phase: scattering_phase.view(),
                    second_phase: scattering_phase_prime.view(),
                    scattering_matrix: Some(scattering_matrix.view()),
                    same_atom: true,
                    first_displacement: [0.22, -0.18, 0.44],
                    second_displacement: [-0.31, 0.28, 0.36],
                    radial_x0: 0.7,
                    radial_dx: 0.2,
                    radial_count: 96,
                },
                real_axis_count: 56,
                chemical_potential_hartree: 0.045,
                temperature_hartree: 0.0035,
                chemical_potential_override_hartree: None,
            })))
        });
    });
}
