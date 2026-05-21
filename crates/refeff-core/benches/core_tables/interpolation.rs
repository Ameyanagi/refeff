use super::*;

pub(super) fn bench_interpolation(c: &mut Criterion) {
    let xs: Vec<_> = (0..128).map(|index| index as f64 * 0.05).collect();
    let ys: Vec<_> = xs
        .iter()
        .map(|&x| (x * x * x) - (0.5 * x * x) + (2.0 * x) + 1.0)
        .collect();
    let complex_ys: Vec<_> = xs.iter().map(|&x| Complex::new(x.sin(), x.cos())).collect();

    c.bench_function("terp_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terp(
                black_box(&xs),
                black_box(&ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("terpc_cubic_128_points", |b| {
        b.iter(|| {
            black_box(terpc(
                black_box(&xs),
                black_box(&complex_ys),
                black_box(3),
                black_box(2.75),
            ))
        });
    });
    c.bench_function("lint_128_points", |b| {
        b.iter(|| black_box(lint(black_box(&xs), black_box(&ys), black_box(2.75))));
    });
    c.bench_function("polcoe_15_points", |b| {
        b.iter(|| {
            black_box(interpolation_polynomial_coefficients(
                black_box(&xs[..15]),
                black_box(&ys[..15]),
            ))
        });
    });

    let min_xs: Vec<_> = (1..=13)
        .map(|index| -1.0 + 0.5 * (index as f64 - 1.0))
        .collect();
    let min_ys: Vec<_> = min_xs
        .iter()
        .map(|&x| (x - 2.15).powi(2) + 0.02 * (x - 2.15).powi(4) + 0.1)
        .collect();
    let bracket = match bracket_table_minimum(&min_xs, &min_ys, 3, 0.0, 0.75) {
        Ok(bracket) => bracket,
        Err(error) => {
            eprintln!("skipping table minimization benches: {error}");
            return;
        }
    };
    c.bench_function("mnbrak_table_cubic_13_points", |b| {
        b.iter(|| {
            black_box(bracket_table_minimum(
                black_box(&min_xs),
                black_box(&min_ys),
                black_box(3),
                black_box(0.0),
                black_box(0.75),
            ))
        });
    });
    c.bench_function("brent_table_cubic_13_points", |b| {
        b.iter(|| {
            black_box(brent_table_minimum(
                black_box(&min_xs),
                black_box(&min_ys),
                black_box(3),
                black_box(bracket),
                black_box(1.0e-5),
            ))
        });
    });
    c.bench_function("dbrent_quartic", |b| {
        b.iter(|| {
            black_box(brent_derivative_minimum(
                black_box(MinimumBracket {
                    ax: -1.0,
                    bx: 1.0,
                    cx: 3.0,
                    fa: 0.0,
                    fb: 0.0,
                    fc: 0.0,
                }),
                black_box(1.0e-8),
                |x| (x - 1.35).powi(2) + 0.05 * (x + 0.5).powi(4) + 0.25,
                |x| 2.0 * (x - 1.35) + 0.2 * (x + 0.5).powi(3),
            ))
        });
    });
}
