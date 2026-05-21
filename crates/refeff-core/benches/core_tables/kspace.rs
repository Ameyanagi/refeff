use super::*;

pub(super) fn bench_kspace_helpers(c: &mut Criterion) {
    let basis = [[1.1, 0.2, 0.05], [-0.1, 1.3, 0.04], [0.03, 0.2, 0.9]];
    c.bench_function("define_k_path_fcc_default", |b| {
        b.iter(|| {
            black_box(define_k_path(
                black_box(BravaisLattice::CubicFaceCentered),
                black_box(0),
                black_box(basis),
            ))
        });
    });
    c.bench_function("define_k_path_orthorhombic_full", |b| {
        b.iter(|| {
            black_box(define_k_path(
                black_box(BravaisLattice::OrthorhombicPrimitive),
                black_box(1),
                black_box(basis),
            ))
        });
    });

    let direct = arr2(&[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
    let reciprocal = arr2(&[
        [std::f64::consts::TAU / 2.0, 0.0, 0.0],
        [0.0, std::f64::consts::TAU / 3.0, 0.0],
        [0.0, 0.0, std::f64::consts::TAU / 4.0],
    ]);
    let operation = arr2(&[[1, -2, 0], [3, 0, 1], [-1, 2, 1]]);
    let vector = [3.2, -1.55, 8.2];
    let skew_lattice = arr2(&[[2.0, 0.3, -0.2], [0.1, 3.0, 0.5], [0.2, 0.4, 4.0]]);
    c.bench_function("reciprocal_lattice_vectors_skew_3x3", |b| {
        b.iter(|| black_box(reciprocal_lattice_vectors(black_box(skew_lattice.view()))));
    });
    let bravais_right_angle = 1_570_796.0 / 1_000_000.0;
    c.bench_function("kmesh_bravais_basis_cxz", |b| {
        b.iter(|| {
            black_box(kmesh_bravais_basis(
                black_box("CXZ"),
                black_box([2.0, 3.0, 4.0]),
                black_box([bravais_right_angle; 3]),
            ))
        });
    });
    let Ok(skew_reciprocal) = reciprocal_lattice_vectors(skew_lattice.view()) else {
        return;
    };
    c.bench_function("kmesh_basis_divisions_skew_120", |b| {
        b.iter(|| {
            black_box(kmesh_basis_divisions(
                black_box(skew_reciprocal.view()),
                black_box(120),
                black_box([false, false, false]),
            ))
        });
    });
    let tetdiv_reciprocal = arr2(&[[2.0, 0.5, 0.0], [0.0, 3.0, 0.25], [0.1, 0.0, 4.0]]);
    c.bench_function("kmesh_tetrahedron_division_skew", |b| {
        b.iter(|| {
            black_box(kmesh_tetrahedron_division(
                black_box([2, 3, 4]),
                black_box(tetdiv_reciprocal.view()),
            ))
        });
    });
    let Ok(tetdiv_offsets) = kmesh_tetrahedron_division([2, 3, 4], tetdiv_reciprocal.view()) else {
        return;
    };
    let tetdiv_links = (1..=60).collect::<Vec<_>>();
    c.bench_function("kmesh_tetrahedron_records_2x3x4_identity", |b| {
        b.iter(|| {
            black_box(kmesh_tetrahedron_records(
                black_box(tetdiv_offsets.view()),
                black_box([2, 3, 4]),
                black_box(&tetdiv_links),
                black_box(60),
            ))
        });
    });
    let reduz_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ];
    let reduz_reciprocal = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    c.bench_function("reduce_kmesh_irreducible_points_2x1x1_sign", |b| {
        b.iter(|| {
            black_box(reduce_kmesh_irreducible_points(
                black_box([2, 1, 1]),
                black_box(reduz_operations.view()),
                black_box(reduz_reciprocal.view()),
            ))
        });
    });
    c.bench_function("kmesh_arbitrary_mesh_4_sign_tetrahedra", |b| {
        b.iter(|| {
            black_box(kmesh_arbitrary_mesh(
                black_box(tetdiv_reciprocal.view()),
                black_box(reduz_operations.view()),
                black_box(4),
                black_box([false, false, false]),
                black_box(true),
            ))
        });
    });
    let klist = arr2(&[[6, 12, 18], [24, 30, 36]]);
    c.bench_function("reduce_kmesh_common_divisor_2x3", |b| {
        b.iter(|| {
            black_box(reduce_kmesh_common_divisor(
                black_box(klist.view()),
                black_box(12),
            ))
        });
    });
    let sdef_operations = array![
        [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
        [[211, 212, 213], [221, 222, 223], [231, 232, 233]]
    ];
    c.bench_function("redefine_lattice_symmetry_cxz_2", |b| {
        b.iter(|| {
            black_box(redefine_lattice_symmetry_operations(
                black_box(sdef_operations.view()),
                black_box("CXZ"),
            ))
        });
    });
    let sdefl_direct = arr2(&[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let Ok(sdefl_reciprocal) = reciprocal_lattice_vectors(sdefl_direct.view()) else {
        return;
    };
    let sdefl_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ];
    c.bench_function("transform_lapw_symmetry_shear_2", |b| {
        b.iter(|| {
            black_box(transform_lapw_symmetry_operations(
                black_box(sdefl_direct.view()),
                black_box(sdefl_reciprocal.view()),
                black_box(sdefl_operations.view()),
                black_box("P  "),
                black_box(true),
            ))
        });
    });
    c.bench_function("subtract_lattice_translation_3d", |b| {
        b.iter(|| {
            black_box(subtract_lattice_translation(
                black_box(reciprocal.view()),
                vector,
            ))
        });
    });
    c.bench_function("reduce_to_lattice_cell_3d", |b| {
        b.iter(|| {
            black_box(reduce_to_lattice_cell(
                black_box(direct.view()),
                black_box(reciprocal.view()),
                black_box(vector),
            ))
        });
    });
    c.bench_function("change_cartesian_basis_3x3", |b| {
        b.iter(|| {
            black_box(change_cartesian_basis(
                black_box(reciprocal.view()),
                black_box(direct.view()),
                black_box(operation.view()),
            ))
        });
    });

    let cubic = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let Ok(cubic_metric) = reciprocal_metric(cubic.view()) else {
        return;
    };
    c.bench_function("point_group_cubic_48", |b| {
        b.iter(|| {
            black_box(point_group_operations(
                black_box(cubic.view()),
                black_box(cubic_metric.view()),
                black_box(64),
            ))
        });
    });

    let sign_operations = array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ];
    let sign_translations = Array2::<f64>::zeros((4, 3));
    c.bench_function("symmetry_check_sign_group_4", |b| {
        b.iter(|| {
            black_box(symmetry_check(
                black_box(sign_operations.view()),
                black_box(sign_translations.view()),
            ))
        });
    });
}
