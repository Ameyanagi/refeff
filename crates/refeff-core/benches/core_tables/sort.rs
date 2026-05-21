use super::*;

pub(super) fn bench_sort_helpers(c: &mut Criterion) {
    let values: Vec<_> = (0..256)
        .map(|index| ((index * 37) % 256) as f64 - 128.0)
        .collect();
    c.bench_function("qsortd_order_256", |b| {
        b.iter(|| black_box(qsortd_order_1based(black_box(&values))));
    });
    c.bench_function("sortid_order_256", |b| {
        b.iter(|| black_box(sortid_order_1based(black_box(&values))));
    });
    c.bench_function("sortir_order_256", |b| {
        b.iter(|| black_box(sortir_order_1based(black_box(&values))));
    });

    let int_values: Vec<_> = (0..256).map(|index| ((index * 37) % 256) - 128).collect();
    c.bench_function("sortii_order_256", |b| {
        b.iter(|| black_box(sortii_order_1based(black_box(&int_values))));
    });
}
