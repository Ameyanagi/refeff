use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::array;
use num_complex::Complex64;
use refeff_linalg::{
    complex_polyfit, complex_polyval, feff_determinant, feff_inverse, real_matmul,
};

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

fn bench_polyfit(c: &mut Criterion) {
    let x = [-1.0, 0.0, 1.5, 2.0, 3.5];
    let y = array![
        Complex64::new(1.0, -1.0),
        Complex64::new(0.5, 0.25),
        Complex64::new(3.0, -0.5),
        Complex64::new(4.2, 1.1),
        Complex64::new(10.0, 2.0)
    ];
    let coefficients = complex_polyfit(&x, y.view(), 2);

    c.bench_function("complex_polyfit_order2_5_points", |b| {
        b.iter(|| {
            black_box(complex_polyfit(
                black_box(&x),
                black_box(y.view()),
                black_box(2),
            ))
        });
    });
    if let Ok(coefficients) = coefficients {
        c.bench_function("complex_polyval_order2_4_points", |b| {
            b.iter(|| {
                black_box(complex_polyval(
                    black_box(coefficients.view()),
                    black_box(&[-0.5, 0.25, 2.25, 4.0]),
                ))
            });
        });
    }
}

criterion_group!(
    benches,
    bench_matrix_helpers,
    bench_faer_bridge,
    bench_polyfit
);
criterion_main!(benches);
