use super::*;

pub(super) fn bench_density_helpers(c: &mut Criterion) {
    let l_count = 4;
    let potential_count = 3;
    let radial_count = 251;
    let scattering_trace = (0..l_count)
        .map(|angular| {
            let l = angular as f64;
            Complex32::new(
                ((0.05_f32 as f64) * l + 0.11_f32 as f64) as f32,
                ((-0.03_f32 as f64) * l + 0.07_f32 as f64) as f32,
            )
        })
        .collect::<Array1<_>>();
    let scattering_ldos = (0..l_count)
        .map(|angular| {
            let l = angular as f64;
            Complex::new(0.2 + 0.04 * l, -0.13 + 0.02 * l)
        })
        .collect::<Array1<_>>();
    let embedded_ldos =
        Array2::from_shape_fn((l_count, potential_count), |(angular, potential)| {
            let l = angular as f64;
            let p = potential as f64;
            Complex::new(0.4 + 0.03 * l + 0.02 * p, -0.2 + 0.01 * l - 0.015 * p)
        });
    let previous_ldos =
        Array2::from_shape_fn((l_count, potential_count), |(angular, potential)| {
            let l = angular as f64;
            let p = potential as f64;
            Complex::new(-0.1 + 0.025 * l + 0.01 * p, 0.08 - 0.02 * l + 0.005 * p)
        });
    let scattering_density = Array2::from_shape_fn((radial_count, l_count), |(radial, angular)| {
        let r = (radial + 1) as f64;
        let l = angular as f64;
        Complex::new(0.006 * r + 0.02 * l, -0.004 * r + 0.015 * l)
    });
    let embedded_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as f64;
            Complex::new(0.05 * r, -0.02 * r)
        })
        .collect::<Array1<_>>();
    let previous_density = (1..=radial_count)
        .map(|radial| {
            let r = radial as f64;
            Complex::new(-0.03 * r, 0.04 * r)
        })
        .collect::<Array1<_>>();
    let valence_density = (1..=radial_count)
        .map(|radial| 0.01 * radial as f64)
        .collect::<Array1<_>>();
    let occupancy_by_l = (0..l_count)
        .map(|angular| -0.03 + 0.015 * angular as f64)
        .collect::<Array1<_>>();

    c.bench_function("density_update_ff2g_251_l4", |b| {
        b.iter(|| {
            black_box(update_valence_density(black_box(
                ValenceDensityUpdateInput {
                    scattering_trace: scattering_trace.view(),
                    potential_index: 1,
                    energy_index: 1,
                    last_radial_index: radial_count,
                    scattering_ldos: scattering_ldos.view(),
                    embedded_ldos: embedded_ldos.view(),
                    previous_ldos: previous_ldos.view(),
                    scattering_density: scattering_density.view(),
                    embedded_density: embedded_density.view(),
                    previous_density: previous_density.view(),
                    valence_density: valence_density.view(),
                    occupancy_by_l: occupancy_by_l.view(),
                    current_energy: Complex::new(0.72, 0.11),
                    previous_energy: Complex::new(0.61, -0.04),
                    potential_multiplicity: 2.5,
                    current_floor: 1,
                    previous_floor: 0,
                    left_sum: Complex::new(0.2, -0.1),
                    right_sum: Complex::new(-0.3, 0.25),
                    total_electron_count: 1.25,
                    include_high_l: false,
                },
            )))
        });
    });

    let atom_potentials = Array1::from_vec(vec![0, 1, 2, 1]);
    let atom_positions = arr2(&[
        [0.0, 0.0, 0.0],
        [1.35, 0.2, -0.15],
        [3.10, -0.4, 0.25],
        [13.5, 0.0, 0.0],
    ]);
    let representative_atoms = Array1::from_vec(vec![0, 1, 2]);
    let atomic_numbers = Array1::from_vec(vec![6, 8, 14]);
    let explicit_overlaps = [
        PotentialOverlapNeighbor {
            source_potential: 0,
            multiplicity: 2.0,
            distance: 1.6,
        },
        PotentialOverlapNeighbor {
            source_potential: 2,
            multiplicity: 1.0,
            distance: 2.4,
        },
    ];
    let electron_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let radius = ((0.05_f32 as f64) * radial as f64 - 8.8_f32 as f64).exp();
            let i = (radial + 1) as f64;
            let p = potential as f64;
            (45.0 + 18.0 * p) * (-(1.0 + 0.08 * p) * radius).exp() + 0.05 * (i + p)
        });
    let spin_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            0.02 + 0.0003 * (radial + 1) as f64 + 0.005 * potential as f64
        });
    let valence_density =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let density = electron_density[(radial, potential)];
            0.65 * density + 0.01 * potential as f64 + 0.0002 * (radial + 1) as f64
        });
    let coulomb_potential =
        Array2::from_shape_fn((radial_count, potential_count), |(radial, potential)| {
            let i = (radial + 1) as f64;
            let p = potential as f64;
            -2.0 - 0.12 * p + 0.004 * i + 0.03 * (0.05 * i + p).cos()
        });

    c.bench_function("density_overlap_ovrlp_251_explicit", |b| {
        b.iter(|| {
            black_box(overlap_potential_density(black_box(
                PotentialOverlapInput {
                    potential_index: 1,
                    atom_potentials: atom_potentials.view(),
                    atom_positions: atom_positions.view(),
                    representative_atoms: representative_atoms.view(),
                    atomic_numbers: atomic_numbers.view(),
                    explicit_overlaps: &explicit_overlaps,
                    electron_density: electron_density.view(),
                    spin_density: spin_density.view(),
                    valence_density: valence_density.view(),
                    coulomb_potential: coulomb_potential.view(),
                },
            )))
        });
    });

    let last_indices = Array1::from_vec(vec![140, 132]);
    let coulom_atom_potentials = Array1::from_vec(vec![0, 1, 1]);
    let coulom_representatives = Array1::from_vec(vec![0, 1]);
    let coulom_atomic_numbers = Array1::from_vec(vec![8, 14]);
    let coulom_norman_radii = Array1::from_vec(vec![0.65, 0.82]);
    let coulom_charge_deltas = Array1::from_vec(vec![0.15, -0.07]);
    let coulom_atom_positions = arr2(&[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 2.1, 0.0]]);
    let coulom_density = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        let radius = (-8.8 + 0.05 * radial as f64).exp();
        (80.0 + 15.0 * potential as f64) * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
    });
    let coulom_edenvl = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        (0.42 + 0.03 * potential as f64) * coulom_density[(radial, potential)]
    });
    let coulom_rhoval = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        (0.36 + 0.02 * potential as f64) * coulom_density[(radial, potential)]
    });
    let coulom_vclap = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        -1.7 - 0.25 * potential as f64 + 0.004 * (radial + 1) as f64
    });
    c.bench_function("density_coulom_update_251x2", |b| {
        b.iter(|| {
            black_box(update_coulomb_potential(black_box(
                CoulombPotentialUpdateInput {
                    mode: CoulombUpdateMode::Norman,
                    highest_potential_index: 1,
                    last_indices: last_indices.view(),
                    valence_density: coulom_rhoval.view(),
                    overlapped_valence_density: coulom_edenvl.view(),
                    overlapped_density: coulom_density.view(),
                    atom_positions: coulom_atom_positions.view(),
                    representative_atoms: coulom_representatives.view(),
                    atom_potentials: coulom_atom_potentials.view(),
                    norman_radii: coulom_norman_radii.view(),
                    charge_deltas: coulom_charge_deltas.view(),
                    atomic_numbers: coulom_atomic_numbers.view(),
                    coulomb_potential: coulom_vclap.view(),
                },
            )))
        });
    });

    let broydn_last_indices = Array1::from_vec(vec![190, 196]);
    let broydn_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let broydn_norman_radii = Array1::from_vec(vec![0.72, 0.88]);
    let broydn_initial_charges = Array1::from_vec(vec![1.40, 2.10]);
    let mut broydn_occupancy = Array2::<f64>::zeros((3, 2));
    broydn_occupancy[(0, 0)] = 1.10;
    broydn_occupancy[(1, 0)] = 0.60;
    broydn_occupancy[(0, 1)] = 1.45;
    broydn_occupancy[(1, 1)] = 0.80;
    broydn_occupancy[(2, 1)] = 0.30;
    let broydn_edenvl = Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
        let radius = (-8.8 + 0.05 * radial as f64).exp();
        (45.0 + 8.0 * potential as f64) * (-0.92 * radius).exp() / (1.0 + 0.10 * radius)
    });
    let broydn_density_for_iteration = |iteration: usize| {
        Array2::from_shape_fn((radial_count, 2), |(radial, potential)| {
            let radius = (-8.8 + 0.05 * radial as f64).exp();
            broydn_edenvl[(radial, potential)]
                * (0.97 + 0.018 * iteration as f64 + 0.004 * potential as f64)
                + (0.015 * iteration as f64 + 0.003 * potential as f64) * (-0.35 * radius).exp()
        })
    };
    let broydn_rhoval1 = broydn_density_for_iteration(1);
    let broydn_rhoval2 = broydn_density_for_iteration(2);
    let broydn_rhoval3 = broydn_density_for_iteration(3);
    let broydn_workspace0 = BroydenWorkspace::zeros(4, 2);
    let broydn_iter2_setup = match mix_broyden_density(BroydenMixInput {
        iteration: 1,
        accelerator: 0.35,
        highest_potential_index: 1,
        valence_occupancy: broydn_occupancy.view(),
        last_indices: broydn_last_indices.view(),
        potential_multiplicities: broydn_multiplicities.view(),
        norman_radii: broydn_norman_radii.view(),
        norman_charges: broydn_initial_charges.view(),
        overlapped_valence_density: broydn_edenvl.view(),
        valence_density: broydn_rhoval1.view(),
        workspace: &broydn_workspace0,
    }) {
        Ok(first_mix) => mix_broyden_density(BroydenMixInput {
            iteration: 2,
            accelerator: 0.35,
            highest_potential_index: 1,
            valence_occupancy: broydn_occupancy.view(),
            last_indices: broydn_last_indices.view(),
            potential_multiplicities: broydn_multiplicities.view(),
            norman_radii: broydn_norman_radii.view(),
            norman_charges: first_mix.norman_charges.view(),
            overlapped_valence_density: broydn_edenvl.view(),
            valence_density: broydn_rhoval2.view(),
            workspace: &first_mix.workspace,
        }),
        Err(error) => Err(error),
    };
    if let Ok(second_mix) = broydn_iter2_setup {
        c.bench_function("density_broydn_mix_251x2_iter3", |b| {
            b.iter(|| {
                black_box(mix_broyden_density(black_box(BroydenMixInput {
                    iteration: 3,
                    accelerator: 0.35,
                    highest_potential_index: 1,
                    valence_occupancy: broydn_occupancy.view(),
                    last_indices: broydn_last_indices.view(),
                    potential_multiplicities: broydn_multiplicities.view(),
                    norman_radii: broydn_norman_radii.view(),
                    norman_charges: second_mix.norman_charges.view(),
                    overlapped_valence_density: broydn_edenvl.view(),
                    valence_density: broydn_rhoval3.view(),
                    workspace: &second_mix.workspace,
                })))
            });
        });
    }
}
