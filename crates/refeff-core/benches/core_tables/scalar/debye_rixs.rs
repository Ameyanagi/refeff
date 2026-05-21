use super::*;

pub(super) fn bench_debye_rixs_helpers(c: &mut Criterion) {
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
