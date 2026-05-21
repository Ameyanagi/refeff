use super::*;

pub(super) fn bench_xsect_dat(c: &mut Criterion) {
    let data = xsect_dat_bench_data();
    let text = match xsect_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xsect.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xsect_dat_text", |b| {
        b.iter(|| black_box(xsect_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xsect_dat_text", |b| {
        b.iter(|| black_box(parse_xsect_dat(black_box(&text))));
    });
    c.bench_function("xsect_dat_ff2x_handoff_256", |b| {
        b.iter(|| {
            black_box(xsect_dat_ff2x_handoff(
                black_box(&data),
                black_box(0.05),
                black_box(1),
            ))
        });
    });
}

pub(super) fn bench_xmu_dat(c: &mut Criterion) {
    let data = xmu_dat_bench_data();
    let text = match xmu_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xmu.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xmu_dat_text", |b| {
        b.iter(|| black_box(xmu_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xmu_dat_text", |b| {
        b.iter(|| black_box(parse_xmu_dat(black_box(&text))));
    });
    c.bench_function("fullspectrum_absolute_xmu_from_xmu_dat", |b| {
        b.iter(|| black_box(fullspectrum_absolute_xmu_from_xmu_dat(black_box(&data))));
    });
    c.bench_function("fullspectrum_normalized_xmu_from_xmu_dat", |b| {
        b.iter(|| black_box(fullspectrum_normalized_xmu_from_xmu_dat(black_box(&data))));
    });
    c.bench_function("fullspectrum_background_segment_from_fprime_xmu_dat", |b| {
        b.iter(|| {
            black_box(fullspectrum_background_segment_from_fprime_xmu_dat(
                black_box(&data),
            ))
        });
    });
    c.bench_function(
        "fullspectrum_real_fine_structure_segment_from_xmu_dat",
        |b| {
            b.iter(|| {
                black_box(fullspectrum_real_fine_structure_segment_from_xmu_dat(
                    black_box(&data),
                ))
            });
        },
    );
    c.bench_function(
        "fullspectrum_imaginary_fine_structure_segment_from_xmu_dat",
        |b| {
            b.iter(|| {
                black_box(fullspectrum_imaginary_fine_structure_segment_from_xmu_dat(
                    black_box(&data),
                ))
            });
        },
    );
}

pub(super) fn bench_opcons_dat(c: &mut Criterion) {
    let data = opcons_dat_bench_data();
    let text = match opcons_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping opcons.dat benchmarks: {err}");
            return;
        }
    };
    let point_count = 4096;
    let omega = Array1::from_shape_fn(point_count, |index| {
        0.01 + 10.0 * index as f64 / (point_count - 1) as f64
    });
    let epsilon_minus_one = Array1::from_shape_fn(point_count, |index| {
        let x = index as f64 / point_count as f64;
        Complex64::new(0.2 + 0.03 * x.sin(), 0.1 + 0.02 * x.cos())
    });

    c.bench_function("render_opcons_dat_text", |b| {
        b.iter(|| black_box(opcons_dat_string(black_box(&data))));
    });
    c.bench_function("parse_opcons_dat_text", |b| {
        b.iter(|| black_box(parse_opcons_dat(black_box(&text))));
    });
    c.bench_function("opcons_dat_from_fullspectrum_epsilon_minus_one_4096", |b| {
        b.iter(|| {
            black_box(opcons_dat_from_fullspectrum_epsilon_minus_one(
                Vec::new(),
                black_box(omega.view()),
                black_box(epsilon_minus_one.view()),
            ))
        });
    });
}

pub(super) fn bench_eps_dat(c: &mut Criterion) {
    let data = eps_dat_bench_data();
    let text = match eps_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping eps.dat benchmarks: {err}");
            return;
        }
    };
    let point_count = 4096;
    let omega = Array1::from_shape_fn(point_count, |index| {
        0.01 + 10.0 * index as f64 / (point_count - 1) as f64
    });
    let scattering_factor = Array1::from_shape_fn(point_count, |index| {
        let x = index as f64 * 0.01;
        Complex64::new(1.0 + 0.02 * x.sin(), 0.3 + 0.04 * x.cos().abs())
    });
    let background_scattering_factor =
        scattering_factor.mapv(|value| value * Complex64::new(0.85, 0.02));

    c.bench_function("render_eps_dat_text", |b| {
        b.iter(|| black_box(eps_dat_string(black_box(&data))));
    });
    c.bench_function("parse_eps_dat_text", |b| {
        b.iter(|| black_box(parse_eps_dat(black_box(&text))));
    });
    c.bench_function("eps_dat_from_fullspectrum_scattering_factors_4096", |b| {
        b.iter(|| {
            black_box(eps_dat_from_fullspectrum_scattering_factors(
                Vec::new(),
                black_box(0.075),
                black_box(omega.view()),
                black_box(scattering_factor.view()),
                black_box(background_scattering_factor.view()),
            ))
        });
    });
}

pub(super) fn bench_xmul_dat(c: &mut Criterion) {
    let data = xmul_dat_bench_data();
    let text = match xmul_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xmul.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xmul_dat_text", |b| {
        b.iter(|| black_box(xmul_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xmul_dat_text", |b| {
        b.iter(|| black_box(parse_xmul_dat(black_box(&text))));
    });
}

pub(super) fn bench_xscorr_raw_dat(c: &mut Criterion) {
    let data = match parse_xscorr_raw_dat(XSCORR_RAW_DAT_BENCH) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("skipping XSCORR raw.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_xscorr_raw_dat_text", |b| {
        b.iter(|| black_box(parse_xscorr_raw_dat(black_box(XSCORR_RAW_DAT_BENCH))));
    });
    c.bench_function("render_xscorr_raw_dat_text", |b| {
        b.iter(|| black_box(xscorr_raw_dat_string(black_box(&data))));
    });
}

pub(super) fn bench_chi_dat(c: &mut Criterion) {
    let data = chi_dat_bench_data();
    let text = match chi_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping chi.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_chi_dat_text", |b| {
        b.iter(|| black_box(chi_dat_string(black_box(&data))));
    });
    c.bench_function("parse_chi_dat_text", |b| {
        b.iter(|| black_box(parse_chi_dat(black_box(&text))));
    });
}

pub(super) fn bench_eels_dat(c: &mut Criterion) {
    let data = eels_dat_bench_data();
    let text = match eels_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping eels.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_eels_dat_text", |b| {
        b.iter(|| black_box(eels_dat_string(black_box(&data))));
    });
    c.bench_function("parse_eels_dat_text", |b| {
        b.iter(|| black_box(parse_eels_dat(black_box(&text))));
    });
}

pub(super) fn bench_danes_dat(c: &mut Criterion) {
    let data = danes_dat_bench_data();
    let text = match danes_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping danes.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_danes_dat_text", |b| {
        b.iter(|| black_box(danes_dat_string(black_box(&data))));
    });
    c.bench_function("parse_danes_dat_text", |b| {
        b.iter(|| black_box(parse_danes_dat(black_box(&text))));
    });
}

pub(super) fn bench_ldos_dat(c: &mut Criterion) {
    let data = ldos_dat_bench_data();
    let text = match ldos_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping ldosNN.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_ldos_dat_text", |b| {
        b.iter(|| black_box(ldos_dat_string(black_box(&data))));
    });
    c.bench_function("parse_ldos_dat_text", |b| {
        b.iter(|| black_box(parse_ldos_dat(black_box(&text))));
    });
    c.bench_function("fullspectrum_ldos_from_ldos_dat", |b| {
        b.iter(|| black_box(fullspectrum_ldos_from_ldos_dat(black_box(&data))));
    });
}

pub(super) fn bench_compton_dat(c: &mut Criterion) {
    let data = compton_dat_bench_data();
    let text = match compton_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping compton.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_compton_dat_text", |b| {
        b.iter(|| black_box(compton_dat_string(black_box(&data))));
    });
    c.bench_function("parse_compton_dat_text", |b| {
        b.iter(|| black_box(parse_compton_dat(black_box(&text))));
    });
}
