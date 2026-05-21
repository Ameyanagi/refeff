use super::*;

pub(super) fn bench_convolution(c: &mut Criterion) {
    let omega: Vec<_> = (0..128).map(|index| -5.0 + index as f64 * 0.1).collect();
    let spectrum: Vec<_> = omega
        .iter()
        .map(|&energy| Complex::new((energy * 0.7).sin(), (energy * 0.4).cos()))
        .collect();

    c.bench_function("conv_128_points", |b| {
        b.iter(|| {
            black_box(conv(
                black_box(&omega),
                black_box(&spectrum),
                black_box(0.2),
            ))
        });
    });

    let excitation_energy = Array1::from_iter((0..256).map(|index| -1.0 + index as f64 * 0.02));
    let excitation_xmu = Array1::from_iter(
        excitation_energy
            .iter()
            .map(|&energy| 0.8 + (energy * 0.9).sin() * 0.2 + (energy * 0.25).cos() * 0.08),
    );
    c.bench_function("ff2x_exconv_256_points", |b| {
        b.iter(|| {
            black_box(ff2x_excitation_convolve(black_box(
                Ff2xExcitationConvolutionInput {
                    energy: excitation_energy.view(),
                    xmu: excitation_xmu.view(),
                    fermi_energy: 0.05,
                    amplitude_reduction: 0.72,
                    relaxation_energy: 0.18,
                    plasmon_frequency: 0.55,
                },
            )))
        });
    });

    let atan_energy = Array1::from_iter((0..320).map(|index| {
        if index < 256 {
            Complex::new(-1.2 + index as f64 * 0.01, 0.08)
        } else {
            Complex::new(1.35, 0.001 + (index - 256) as f64 * 0.002)
        }
    }));
    let atan_xsec = Array1::from_iter(
        atan_energy
            .iter()
            .map(|energy| Complex::new(0.9 + (energy.re * 0.6).sin() * 0.2, energy.re * 0.01)),
    );
    let atan_xsnorm = Array1::from_iter((0..320).map(|index| 0.85 + index as f64 * 0.0005));
    let atan_chia = Array1::from_iter(atan_energy.iter().map(|energy| {
        Complex::new(
            (energy.re * 1.3).cos() * 0.08,
            (energy.re * 0.9).sin() * 0.03,
        )
    }));
    c.bench_function("ff2x_xscorratan_256_horizontal_points", |b| {
        b.iter(|| {
            black_box(ff2x_atan_correction(black_box(Ff2xAtanCorrectionInput {
                spectroscopy: 1,
                energy: atan_energy.view(),
                horizontal_len: 256,
                fermi_index: 120,
                xsec: atan_xsec.view(),
                xsnorm: atan_xsnorm.view(),
                chia: atan_chia.view(),
                real_correction: 0.03,
                imaginary_correction: 0.0,
            })))
        });
    });
}
