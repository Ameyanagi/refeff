use super::*;

pub(super) fn bench_angular_tables(c: &mut Criterion) {
    c.bench_function("build_legendre_xnlm_lmax8", |b| {
        b.iter(|| black_box(legendre_normalization_table(black_box(8))));
    });
    c.bench_function("build_spin_orbit_tables_lmax8", |b| {
        b.iter(|| black_box(spin_orbit_coupling_tables(black_box(8))));
    });
    c.bench_function("build_relativistic_cgc_lmax8", |b| {
        b.iter(|| black_box(relativistic_clebsch_gordan_coefficients(black_box(8))));
    });
    c.bench_function("build_mkgtr_clbcoef_lmax8", |b| {
        b.iter(|| {
            black_box(mkgtr_clebsch_gordan_coefficients(
                black_box(8),
                black_box(9),
                black_box(18),
            ))
        });
    });
    c.bench_function("relativistic_state_index_kappa_grid", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for kappa in black_box([-1, 1, -2, 2, -3, 3, -4, 4]) {
                let jp05 = i32::abs(kappa);
                for mu_minus_half in -jp05..jp05 {
                    if let Ok(index) =
                        relativistic_state_index_1based(black_box(kappa), black_box(mu_minus_half))
                    {
                        total = total.saturating_add(index);
                    }
                }
            }
            black_box(total)
        });
    });
    c.bench_function("build_basis_transform_lmax4", |b| {
        b.iter(|| black_box(basis_transform_matrices(black_box(4))));
    });
    let Ok(basis_transforms) = basis_transform_matrices(3) else {
        return;
    };
    let basis_input = Array2::from_shape_fn(
        (basis_transforms.order, basis_transforms.order).f(),
        |(row, column)| {
            Complex::new(
                0.01 * (row as f64 + 1.0) + 0.003 * (column as f64 + 1.0),
                -0.002 * (row as f64 + 1.0) + 0.007 * (column as f64 + 1.0),
            )
        },
    );
    c.bench_function("change_basis_representation_lmax3_rel_to_real", |b| {
        b.iter(|| {
            black_box(change_basis_representation(
                black_box(basis_input.view()),
                black_box(BasisTransformMode::RelativisticToReal),
                black_box(&basis_transforms),
            ))
        });
    });
    c.bench_function("build_legendre_polynomials_lmax32", |b| {
        b.iter(|| black_box(legendre_polynomials(black_box(0.25), black_box(32))));
    });
    c.bench_function("wigner_rotation_half_integer", |b| {
        b.iter(|| {
            black_box(wigner_rotation(
                black_box(0.7),
                black_box(3),
                black_box(1),
                black_box(-1),
                black_box(2),
            ))
        });
    });
    c.bench_function("spherical_harmonics_l8", |b| {
        b.iter(|| {
            black_box(spherical_harmonics(
                black_box([1.0, 2.0, 3.0]),
                black_box(8),
            ))
        });
    });
    c.bench_function("polarization_tensor_cartesian", |b| {
        b.iter(|| {
            black_box(polarization_tensor(
                black_box(5),
                black_box(PolarizationTensorMode::Cartesian),
            ))
        });
    });
    c.bench_function("transition_b_matrix_l3", |b| {
        b.iter(|| {
            black_box(transition_b_matrix(black_box(TransitionBMatrixInput {
                lmax: 3,
                initial_kappa: -1,
                polarization: 1,
                polarization_tensor: sample_polarization_tensor(),
                multipole: 2,
                trace_orbital: false,
                spin: 1,
                spin_channels: 1,
                spin_vector_angle: 0.3,
            })))
        });
    });
}

fn sample_polarization_tensor() -> [[Complex; 3]; 3] {
    [
        [
            Complex::new(0.20, -0.05),
            Complex::new(-0.10, 0.04),
            Complex::new(0.03, 0.02),
        ],
        [
            Complex::new(0.11, -0.07),
            Complex::new(0.50, 0.00),
            Complex::new(-0.08, 0.09),
        ],
        [
            Complex::new(0.06, 0.01),
            Complex::new(0.13, -0.02),
            Complex::new(0.17, 0.03),
        ],
    ]
}

pub(super) fn bench_state_kets(c: &mut Criterion) {
    let atom_potentials = vec![0, 1, 1, 2, 2, 2, 1, 0, 3, 3, 2, 1];
    let potential_lmax = vec![0, 2, 3, 1];

    c.bench_function("construct_state_kets_small_cluster", |b| {
        b.iter(|| {
            black_box(construct_state_kets(
                black_box(2),
                black_box(&atom_potentials),
                black_box(&potential_lmax),
                black_box(3),
            ))
        });
    });
}
