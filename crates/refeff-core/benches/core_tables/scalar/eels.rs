use super::*;

pub(super) fn bench_eels_helpers(c: &mut Criterion) {
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
}
