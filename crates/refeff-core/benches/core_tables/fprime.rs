use super::*;

pub(super) fn bench_fprime_helpers(c: &mut Criterion) {
    c.bench_function("fprime_funlog_real_branch", |b| {
        b.iter(|| {
            black_box(fprime_log_correction(
                black_box(FprimeLogCase::RealFrequency),
                black_box(0.08),
                black_box(0.50),
                black_box(0.21),
            ))
        });
    });
    c.bench_function("xscorr_lorentz_kernel", |b| {
        b.iter(|| {
            black_box(xscorr_lorentz_kernel(
                black_box(0.08),
                black_box(0.13),
                black_box(0.02),
            ))
        });
    });
    c.bench_function("xscorr_arctangent_step", |b| {
        b.iter(|| black_box(xscorr_arctangent_step(black_box(0.08), black_box(-0.11))));
    });

    let contour_energy = Array1::from_iter(
        (0..96)
            .map(|index| Complex::new(-0.05 + 0.012 * index as f64, 0.02 + 0.0015 * index as f64)),
    );
    let contour_xmu = Array1::from_iter((0..96).map(|index| {
        let row = index as f64 + 1.0;
        Complex::new(0.7 + 0.006 * row + 0.0001 * row * row, -0.04 + 0.001 * row)
    }));
    c.bench_function("fprime_fpint_96_points", |b| {
        b.iter(|| {
            black_box(fprime_contour_integral(black_box(
                FprimeContourIntegralInput {
                    energy: contour_energy.view(),
                    xmu: contour_xmu.view(),
                    start_index: 1,
                    end_index: 95,
                    delta: 0.11,
                    loss: 0.08,
                    epsilon: 1.0e-4,
                    fermi_energy: 0.03,
                },
            )))
        });
    });

    let axis_energy = Array1::from_iter((0..96).map(|index| 0.03 + 0.025 * index as f64));
    let axis_xmu = Array1::from_iter((0..96).map(|index| {
        let row = index as f64 + 1.0;
        Complex::new(0.5 + 0.008 * row + 0.00015 * row * row, 0.01 + 0.002 * row)
    }));
    c.bench_function("fprime_fpintp_96_points", |b| {
        b.iter(|| {
            black_box(fprime_positive_axis_integral(black_box(
                FprimePositiveAxisIntegralInput {
                    energy: axis_energy.view(),
                    xmu: axis_xmu.view(),
                    delta: 0.09,
                    loss: 0.08,
                    fermi_energy: 0.03,
                },
            )))
        });
    });
}
