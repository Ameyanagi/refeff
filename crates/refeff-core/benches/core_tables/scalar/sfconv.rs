use super::*;

pub(super) fn bench_sfconv_helpers(c: &mut Criterion) {
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
}
