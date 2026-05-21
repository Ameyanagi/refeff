use super::*;

pub(super) fn bench_path_helpers(c: &mut Criterion) {
    let path = [1, 2, 3, 4, 5, 6, 7, 8];
    c.bench_function("pack_path_indices_8", |b| {
        b.iter(|| black_box(pack_path_indices(black_box(&path))));
    });
    let packed = [3_329_498, 8_325_663, 13_321_836];
    c.bench_function("unpack_path_indices_8", |b| {
        b.iter(|| black_box(unpack_path_indices(black_box(packed), black_box(8))));
    });
    let (phase_energies, reference_energies, phase_shifts, angular_limits) =
        sample_path_phase_criteria_inputs();
    c.bench_function("path_phase_criteria_tables_43", |b| {
        b.iter(|| {
            black_box(path_phase_criteria_tables(black_box(
                PathPhaseCriteriaInput {
                    energies: &phase_energies,
                    reference_energies: &reference_energies,
                    phase_shifts: phase_shifts.view(),
                    angular_limits: angular_limits.view(),
                    output_energy_count: 38,
                    zero_wave_energy_index: 1,
                },
            )))
        });
    });
    c.bench_function("path_heap_bubble_up", |b| {
        b.iter(|| {
            let mut keys = black_box([1.0, 3.0, 2.0, 5.0, 4.0, 0.5]);
            let mut indices = black_box([10, 30, 20, 50, 40, 5]);
            black_box(path_heap_bubble_up(&mut keys, &mut indices))
        });
    });
    c.bench_function("path_heap_bubble_down", |b| {
        b.iter(|| {
            let mut keys = black_box([6.0, 2.0, 3.0, 4.0, 5.0]);
            let mut indices = black_box([60, 20, 30, 40, 50]);
            black_box(path_heap_bubble_down(&mut keys, &mut indices))
        });
    });

    let atom_positions = ndarray::arr2(&[
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
        [2.0, 1.0, 0.4],
        [-0.5, 1.7, 0.3],
        [0.7, -1.2, 0.8],
    ]);
    let path_indices = [1_usize, 2, 3, 4];
    c.bench_function("path_geometry_4_scatterers", |b| {
        b.iter(|| {
            black_box(path_geometry(
                black_box(atom_positions.view()),
                black_box(&path_indices),
            ))
        });
    });
    c.bench_function("path_output_parameters_4", |b| {
        b.iter(|| {
            black_box(path_output_parameters(
                black_box(atom_positions.view()),
                black_box(&path_indices),
            ))
        });
    });
    c.bench_function("path_standard_coordinates_4", |b| {
        b.iter(|| {
            black_box(path_standard_coordinates(black_box(
                PathStandardCoordinatesInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    polarization: 0,
                    spin: 0,
                    electric_vector: [0.0, 0.0, 1.0],
                    incident_vector: [0.0, 0.0, 0.0],
                    symmetry_case_override: None,
                },
            )))
        });
    });

    let atom_potentials: Vec<_> = (0..=8).map(|index| index % 4).collect();
    c.bench_function("path_canonical_representation_4", |b| {
        b.iter(|| {
            black_box(path_canonical_representation(black_box(
                PathCanonicalRepresentationInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    polarization: 0,
                    spin: 0,
                    electric_vector: [0.0, 0.0, 1.0],
                    incident_vector: [0.0, 0.0, 0.0],
                    symmetry_case_override: None,
                    force_no_symmetry: false,
                },
            )))
        });
    });

    let hash_positions = ndarray::arr2(&[
        [1.23456, -0.34567, 0.12549],
        [-2.25, 1.5004, -0.9995],
        [0.0, 2.4996, 3.3333],
        [0.75, -0.25, 0.5],
    ]);
    let potential_indices = [1, 3, 0, 2];
    c.bench_function("path_degeneracy_hash_4", |b| {
        b.iter(|| {
            black_box(path_degeneracy_hash(
                black_box(hash_positions.view()),
                black_box(&potential_indices),
            ))
        });
    });

    let criteria_distances = [1.10, 1.25, 1.40, 1.60, 1.20];
    let criteria_angles = [0.80, -0.35, 0.55, -0.10, 0.25];
    let criteria_beta = [-3, 4, 10, -2, 0];
    let fbeta = Array3::from_shape_fn((81, 4, 3), |(beta_row, potential, criterion)| {
        let beta_index = beta_row as i32 - 40;
        f64::from(
            0.5_f32
                + 0.01_f32 * potential as f32
                + 0.002_f32 * (criterion + 1) as f32
                + 0.003_f32 * beta_index.abs() as f32
                + 0.0001_f32 * beta_index as f32,
        )
    });
    let criteria_waves = [2.0, 3.5, 5.0];
    let mean_free_paths = [7.5, 10.0, 12.0];
    c.bench_function("path_heap_criterion_4", |b| {
        b.iter(|| {
            black_box(path_heap_criterion(
                black_box(&path_indices),
                black_box(&criteria_distances),
                black_box(&criteria_beta),
                black_box(&atom_potentials),
                black_box(fbeta.view()),
                black_box(&criteria_waves),
            ))
        });
    });
    c.bench_function("path_output_criterion_4", |b| {
        b.iter(|| {
            black_box(path_output_criterion(black_box(PathOutputCriterionInput {
                path_indices: &path_indices,
                leg_distances: &criteria_distances,
                angle_cosines: &criteria_angles,
                beta_indices: &criteria_beta,
                atom_potentials: &atom_potentials,
                fbeta_critical: fbeta.view(),
                mean_free_paths: &mean_free_paths,
                wave_numbers: &criteria_waves,
                current_normalization: 0.004,
            })))
        });
    });

    let mut cluster_outside = vec![false; atom_potentials.len()];
    cluster_outside[4] = true;
    c.bench_function("path_criteria_decision_4", |b| {
        b.iter(|| {
            black_box(path_criteria_decision(black_box(
                PathCriteriaDecisionInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    cluster_outside: &cluster_outside,
                    fbeta_critical: fbeta.view(),
                    mean_free_paths: &mean_free_paths,
                    wave_numbers: &criteria_waves,
                    max_path_length: 20.0,
                    heap_cutoff: 0.0,
                    output_cutoff: 50.0,
                    current_normalization: -1.0,
                },
            )))
        });
    });

    let fbeta_output = Array3::from_shape_fn((81, 4, 5), |(beta_row, potential, energy)| {
        let beta_index = beta_row as i32 - 40;
        f64::from(
            0.45_f32
                + 0.008_f32 * potential as f32
                + 0.015_f32 * (energy + 1) as f32
                + 0.0025_f32 * beta_index.abs() as f32
                + 0.0002_f32 * beta_index as f32,
        )
    });
    let output_waves = [1.2, 2.0, 3.25, 4.5, 6.0];
    let output_mean_free_paths = [6.0, 7.5, 9.0, 11.0, 14.0];
    c.bench_function("path_output_importance_4", |b| {
        b.iter(|| {
            black_box(path_output_importance(black_box(
                PathOutputImportanceInput {
                    atom_positions: atom_positions.view(),
                    path_indices: &path_indices,
                    atom_potentials: &atom_potentials,
                    fbeta: fbeta_output.view(),
                    wave_numbers: &output_waves,
                    mean_free_paths: &output_mean_free_paths,
                    start_energy_index: 1,
                    fbeta_critical: fbeta.view(),
                    critical_wave_numbers: &criteria_waves,
                    critical_mean_free_paths: &mean_free_paths,
                    current_normalization: 0.004,
                },
            )))
        });
    });
}

fn sample_path_phase_criteria_inputs()
-> (Vec<Complex>, Vec<Complex>, Array3<Complex>, Array2<usize>) {
    let energy_count = 43;
    let potential_count = 3;
    let angular_channels = 4;
    let energies = (0..energy_count)
        .map(|index| {
            let ie = (index + 1) as f64;
            Complex::new(0.02 * (ie - 2.0) + 0.001 * (ie - 1.0), 0.005 + 0.0003 * ie)
        })
        .collect::<Vec<_>>();
    let references = vec![Complex::new(-0.015, -0.002); energy_count];
    let phase_shifts = Array3::from_shape_fn(
        (energy_count, angular_channels, potential_count).f(),
        |(energy, angular, potential)| {
            let ie = (energy + 1) as f64;
            let il = (angular + 1) as f64;
            let iph = potential as f64;
            Complex::new(
                0.02 * ie + 0.11 * il + 0.03 * iph,
                0.004 * ie - 0.002 * il + 0.001 * iph,
            )
        },
    );
    let angular_limits = Array2::from_shape_fn(
        (energy_count, potential_count).f(),
        |(energy, potential)| (energy + 1 + potential) % angular_channels,
    );
    (energies, references, phase_shifts, angular_limits)
}
