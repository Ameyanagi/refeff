use super::*;

pub(super) fn bench_quadrature(c: &mut Criterion) {
    let xs: Vec<_> = (0..1024).map(|index| index as f64 * 0.01).collect();
    let ys: Vec<_> = xs.iter().map(|&x| x.sin() * x.exp()).collect();
    c.bench_function("trap_1024_points", |b| {
        b.iter(|| black_box(trap(black_box(&xs), black_box(&ys))));
    });
    c.bench_function("gauleg_64_points", |b| {
        b.iter(|| {
            black_box(gauss_legendre_quadrature(
                black_box(-1.0),
                black_box(1.0),
                black_box(64),
            ))
        });
    });

    let radii: Vec<_> = (0..128)
        .map(|index| (-8.8 + index as f64 * 0.05).exp())
        .collect();
    let values: Vec<_> = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| radius * (1.0 + index as f64 * 0.001))
        .collect();
    let rnrm = radii[100] * 0.02_f64.exp();
    c.bench_function("somm2_128_points", |b| {
        b.iter(|| {
            black_box(somm2(
                black_box(&radii),
                black_box(&values),
                black_box(0.05),
                black_box(0.5),
                black_box(rnrm),
                black_box(0),
            ))
        });
    });
    let complex_dp = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| {
            let row = index as f64 + 1.0;
            Complex::new(radius * (0.3 + 0.002 * row), -0.02 * row)
        })
        .collect::<Vec<_>>();
    let complex_dq = radii
        .iter()
        .enumerate()
        .map(|(index, &radius)| {
            let row = index as f64 + 1.0;
            Complex::new(-0.05 * radius * row, 0.004 * row * row)
        })
        .collect::<Vec<_>>();
    c.bench_function("csommjas_128_points", |b| {
        b.iter(|| {
            black_box(csommjas(
                black_box(&radii),
                black_box(&complex_dp),
                black_box(&complex_dq),
                black_box(0.05),
                black_box(0.5),
                black_box(0),
            ))
        });
    });
}
