use super::*;

pub(super) fn bench_genfmt_helpers(c: &mut Criterion) {
    let beta_angles = [0.0, 0.25, std::f64::consts::PI];
    c.bench_function("genfmt_lambda_indices_cute_high", |b| {
        b.iter(|| {
            black_box(lambda_indices(black_box(LambdaIndexInput {
                calculation: 10,
                energy_index: 42,
                scattering_count: 2,
                initial_l: 4,
                beta_angles: &beta_angles,
                lambda_capacity: 80,
                max_m: 10,
                max_n: 10,
            })))
        });
    });
    c.bench_function("genfmt_xstar_elliptic", |b| {
        b.iter(|| {
            black_box(xstar(black_box(XStarInput {
                primary_polarization: [0.3, 1.0, -0.2],
                secondary_polarization: [-0.4, 0.2, 1.5],
                first_leg: [1.2, -0.5, 0.8],
                last_leg: [-0.7, 1.4, 0.6],
                degeneracy: 2.25,
                initial_l: 2,
                ellipticity: 0.7,
            })))
        });
    });
    c.bench_function("genfmt_initial_state_rotation_l3", |b| {
        b.iter(|| {
            black_box(initial_state_rotation(black_box(
                InitialStateRotationInput {
                    lmaxp1: 4,
                    mmaxp1: 4,
                    beta_angle: 0.7,
                },
            )))
        });
    });
    let path_positions = arr2(&[
        [1.2, -0.4, 0.7],
        [-0.3, 1.1, 1.5],
        [0.5, 0.2, -0.6],
        [0.0, 0.0, 0.0],
    ]);
    c.bench_function("genfmt_path_rotation_angles_polarized", |b| {
        b.iter(|| {
            black_box(path_rotation_angles(black_box(PathRotationInput {
                positions: path_positions.view(),
                polarized: true,
            })))
        });
    });
    c.bench_function("genfmt_legendre_normalization_l16_m8", |b| {
        b.iter(|| {
            black_box(genfmt_legendre_normalization_table(black_box(
                GenfmtLegendreNormalizationInput {
                    lmaxp1: 17,
                    mmaxp1: 9,
                },
            )))
        });
    });
    c.bench_function("genfmt_curved_wave_polynomials_l4", |b| {
        b.iter(|| {
            black_box(curved_wave_polynomials(black_box(
                CurvedWavePolynomialInput {
                    lmaxp1: 5,
                    mmaxp1: 4,
                    rho: Complex::new(1.25, 0.4),
                },
            )))
        });
    });

    let Ok(scattering) = sample_scattering_amplitude_inputs() else {
        return;
    };
    c.bench_function("genfmt_scattering_amplitude_matrix_6x5", |b| {
        b.iter(|| black_box(scattering_amplitude_matrix(black_box(scattering.input()))));
    });

    let Ok(polarized) = sample_polarized_scattering_amplitude_inputs() else {
        return;
    };
    c.bench_function("genfmt_polarized_scattering_amplitude_matrix_6", |b| {
        b.iter(|| {
            black_box(polarized_scattering_amplitude_matrix(black_box(
                polarized.input(),
            )))
        });
    });

    let transition = sample_energy_independent_transition_inputs();
    c.bench_function("genfmt_energy_independent_transition_matrix", |b| {
        b.iter(|| {
            black_box(energy_independent_transition_matrix(black_box(
                transition.input(),
            )))
        });
    });
    c.bench_function("genfmt_energy_independent_transition_matrix_avg", |b| {
        b.iter(|| {
            black_box(energy_independent_transition_matrix(black_box(
                transition.unpolarized_input(),
            )))
        });
    });
}

struct SampleScatteringAmplitude {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    phase_shifts: Array1<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    rotation: Array3<f64>,
    xnlm: Array2<f64>,
}

impl SampleScatteringAmplitude {
    fn input(&self) -> ScatteringAmplitudeMatrixInput<'_> {
        ScatteringAmplitudeMatrixInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            left_lambda_count: 6,
            right_lambda_count: 5,
            phase_shifts: self.phase_shifts.view(),
            angular_limit: 3,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            rotation: self.rotation.view(),
            rotation_magnetic_offset: 4,
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn sample_scattering_amplitude_inputs()
-> Result<SampleScatteringAmplitude, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let phase_shifts = Array1::from_iter((-4..=4).map(|l| {
        let l = l as f64;
        Complex::new(0.015 * l + 0.02, -0.01 * l + 0.03)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut rotation = Array3::zeros((5, 9, 9).f());
    for l in 0..=4 {
        let il = (l + 1) as f64;
        for m1 in -4_i32..=4 {
            for m2 in -4_i32..=4 {
                if (m1.unsigned_abs() as usize) <= l && (m2.unsigned_abs() as usize) <= l {
                    let row = (m1 + 4) as usize;
                    let column = (m2 + 4) as usize;
                    rotation[(l, row, column)] =
                        (0.11 * il + 0.07 * (m1 as f64) - 0.05 * (m2 as f64)).cos();
                }
            }
        }
    }

    Ok(SampleScatteringAmplitude {
        m_indices,
        n_indices,
        phase_shifts,
        first_polynomials,
        second_polynomials,
        rotation,
        xnlm: legendre_normalization_table(4)?,
    })
}

struct SamplePolarizedScatteringAmplitude {
    m_indices: Array1<i32>,
    n_indices: Array1<i32>,
    transition_angular_momenta: Array1<i32>,
    radial_factors: Array1<Complex>,
    transition_matrix: Array4<Complex>,
    first_polynomials: Array2<Complex>,
    second_polynomials: Array2<Complex>,
    xnlm: Array2<f64>,
}

impl SamplePolarizedScatteringAmplitude {
    fn input(&self) -> PolarizedScatteringAmplitudeInput<'_> {
        PolarizedScatteringAmplitudeInput {
            m_indices: self.m_indices.view(),
            n_indices: self.n_indices.view(),
            lambda_count: 6,
            transition_angular_momenta: self.transition_angular_momenta.view(),
            radial_factors: self.radial_factors.view(),
            transition_matrix: self.transition_matrix.view(),
            transition_magnetic_offset: 4,
            first_leg_polynomials: self.first_polynomials.view(),
            second_leg_polynomials: self.second_polynomials.view(),
            xnlm: self.xnlm.view(),
            eta: 0.37,
        }
    }
}

fn sample_polarized_scattering_amplitude_inputs()
-> Result<SamplePolarizedScatteringAmplitude, Box<dyn std::error::Error>> {
    let m_indices = Array1::from_vec(vec![0, -1, 1, -2, 2, 0, -1, 1]);
    let n_indices = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let radial_factors = Array1::from_iter((1..=8).map(|k| {
        let k = k as f64;
        Complex::new(0.9 + 0.07 * k, -0.02 * k)
    }));
    let first_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(1.25, 0.4),
    })?;
    let second_polynomials = curved_wave_polynomials(CurvedWavePolynomialInput {
        lmaxp1: 4,
        mmaxp1: 9,
        rho: Complex::new(-0.8, 1.1),
    })?;
    let mut transition_matrix = Array4::zeros((9, 8, 9, 8).f());
    for k2 in 1..=8 {
        for m2 in -4_i32..=4 {
            for k1 in 1..=8 {
                for m1 in -4_i32..=4 {
                    let first_m = (m1 + 4) as usize;
                    let second_m = (m2 + 4) as usize;
                    transition_matrix[(first_m, k1 - 1, second_m, k2 - 1)] = Complex::new(
                        0.01 * (m1 as f64) + 0.02 * (m2 as f64) + 0.03 * (k1 as f64)
                            - 0.015 * (k2 as f64),
                        0.02 * ((m1 - m2) as f64) + 0.01 * (k1 as f64) + 0.04 * (k2 as f64),
                    );
                }
            }
        }
    }

    Ok(SamplePolarizedScatteringAmplitude {
        m_indices,
        n_indices,
        transition_angular_momenta,
        radial_factors,
        transition_matrix,
        first_polynomials,
        second_polynomials,
        xnlm: legendre_normalization_table(4)?,
    })
}

struct SampleEnergyIndependentTransition {
    transition_angular_momenta: Array1<i32>,
    transition_b_matrix: Array6<Complex>,
    combined_rotation: Array3<f64>,
    first_rotation: Array3<f64>,
    last_rotation: Array3<f64>,
}

impl SampleEnergyIndependentTransition {
    fn input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 1,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Polarized {
                first_rotation: self.first_rotation.view(),
                last_rotation: self.last_rotation.view(),
                first_eta: 0.23,
                last_eta: 0.41,
            },
        }
    }

    fn unpolarized_input(&self) -> EnergyIndependentMatrixInput<'_> {
        EnergyIndependentMatrixInput {
            transition_angular_momenta: self.transition_angular_momenta.view(),
            transition_b_matrix: self.transition_b_matrix.view(),
            transition_magnetic_offset: 3,
            spin_index: 0,
            initial_l: 2,
            magnetic_limit: 3,
            rotation_magnetic_offset: 3,
            rotations: TransitionRotationInput::Unpolarized {
                combined_rotation: self.combined_rotation.view(),
            },
        }
    }
}

fn sample_energy_independent_transition_inputs() -> SampleEnergyIndependentTransition {
    let transition_angular_momenta = Array1::from_vec(vec![0, 1, 2, 3, 1, 2, -1, 3]);
    let mut transition_b_matrix = Array6::zeros((7, 2, 8, 7, 2, 8).f());
    for k2 in 1..=8 {
        for s2 in 0..=1 {
            for m2 in -3_i32..=3 {
                for k1 in 1..=8 {
                    for s1 in 0..=1 {
                        for m1 in -3_i32..=3 {
                            let first_m = (m1 + 3) as usize;
                            let second_m = (m2 + 3) as usize;
                            transition_b_matrix[(first_m, s1, k1 - 1, second_m, s2, k2 - 1)] =
                                Complex::new(
                                    0.01 * (m1 as f64) + 0.02 * (m2 as f64) + 0.03 * (k1 as f64)
                                        - 0.015 * (k2 as f64)
                                        + 0.04 * (s1 as f64)
                                        - 0.025 * (s2 as f64),
                                    0.02 * ((m1 - m2) as f64)
                                        + 0.01 * (k1 as f64)
                                        + 0.04 * (k2 as f64)
                                        + 0.03 * (s1 as f64)
                                        + 0.02 * (s2 as f64),
                                );
                        }
                    }
                }
            }
        }
    }

    SampleEnergyIndependentTransition {
        transition_angular_momenta,
        transition_b_matrix,
        combined_rotation: sample_mmtr_rotation(1),
        first_rotation: sample_mmtr_rotation(2),
        last_rotation: sample_mmtr_rotation(3),
    }
}

fn sample_mmtr_rotation(leg: usize) -> Array3<f64> {
    let mut rotation = Array3::zeros((4, 7, 7).f());
    for l in 0..=3 {
        let il = (l + 1) as f64;
        for m1 in -3_i32..=3 {
            for m2 in -3_i32..=3 {
                if (m1.unsigned_abs() as usize) <= l && (m2.unsigned_abs() as usize) <= l {
                    let row = (m1 + 3) as usize;
                    let column = (m2 + 3) as usize;
                    rotation[(l, row, column)] =
                        (0.13 * il + 0.07 * (m1 as f64) - 0.05 * (m2 as f64) + 0.17 * (leg as f64))
                            .cos();
                }
            }
        }
    }
    rotation
}
