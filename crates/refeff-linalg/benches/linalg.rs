use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::array;
use refeff_linalg::{feff_determinant, feff_inverse, real_matmul};

fn bench_matrix_helpers(c: &mut Criterion) {
    let matrix = array![[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [0.25, -0.5, 1.5]];
    c.bench_function("feff_determinant_3x3", |b| {
        b.iter(|| black_box(feff_determinant(black_box(matrix.view()))));
    });
    c.bench_function("feff_inverse_3x3", |b| {
        b.iter(|| black_box(feff_inverse(black_box(matrix.view()))));
    });
}

fn bench_faer_bridge(c: &mut Criterion) {
    let lhs = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let rhs = array![[3.0, 2.0], [1.0, 0.5], [4.0, -1.0]];
    c.bench_function("real_matmul_3x3_3x2", |b| {
        b.iter(|| black_box(real_matmul(black_box(lhs.view()), black_box(rhs.view()))));
    });
}

criterion_group!(benches, bench_matrix_helpers, bench_faer_bridge);
criterion_main!(benches);
