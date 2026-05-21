use super::*;

pub(super) fn bench_fullspectrum_helpers(c: &mut Criterion) {
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
}
