use super::*;

pub(super) fn bench_pot_bin(c: &mut Criterion) {
    let data = pot_bin_bench_data();
    let text = match pot_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping pot.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_pot_bin_text", |b| {
        b.iter(|| black_box(pot_bin_string(black_box(&data))));
    });
    c.bench_function("parse_pot_bin_text", |b| {
        b.iter(|| black_box(parse_pot_bin(black_box(&text))));
    });
    c.bench_function("fullspectrum_pot_bin_rdpotp_view", |b| {
        b.iter(|| black_box(fullspectrum_potential_state_from_pot_bin(black_box(&data))));
    });
}

pub(super) fn bench_phase_bin(c: &mut Criterion) {
    let data = phase_bin_bench_data();
    let text = match phase_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping phase.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_phase_bin_text", |b| {
        b.iter(|| black_box(phase_bin_string(black_box(&data))));
    });
    c.bench_function("parse_phase_bin_text", |b| {
        b.iter(|| black_box(parse_phase_bin(black_box(&text))));
    });
}

pub(super) fn bench_feff_bin(c: &mut Criterion) {
    let data = feff_bin_bench_data();
    let text = match feff_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping feff.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_feff_bin_text", |b| {
        b.iter(|| black_box(feff_bin_string(black_box(&data))));
    });
    c.bench_function("parse_feff_bin_text", |b| {
        b.iter(|| black_box(parse_feff_bin(black_box(&text))));
    });
}

pub(super) fn bench_fms_bin(c: &mut Criterion) {
    let data = fms_bin_bench_data();
    let text = match fms_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping fms.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_fms_bin_text", |b| {
        b.iter(|| black_box(fms_bin_string(black_box(&data))));
    });
    c.bench_function("parse_fms_bin_text", |b| {
        b.iter(|| black_box(parse_fms_bin(black_box(&text))));
    });
}

pub(super) fn bench_gtr_dat(c: &mut Criterion) {
    let gtr = match parse_gtr_dat(GTR_DAT_BENCH) {
        Ok(gtr) => gtr,
        Err(err) => {
            eprintln!("skipping FMS trace text benchmarks: {err}");
            return;
        }
    };
    let gtrl = match parse_gtrl_dat(GTRL_DAT_BENCH) {
        Ok(gtrl) => gtrl,
        Err(err) => {
            eprintln!("skipping FMS trace text benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_gtr_dat_text", |b| {
        b.iter(|| black_box(parse_gtr_dat(black_box(GTR_DAT_BENCH))));
    });
    c.bench_function("render_gtr_dat_text", |b| {
        b.iter(|| black_box(gtr_dat_string(black_box(&gtr))));
    });
    c.bench_function("parse_gtrl_dat_text", |b| {
        b.iter(|| black_box(parse_gtrl_dat(black_box(GTRL_DAT_BENCH))));
    });
    c.bench_function("render_gtrl_dat_text", |b| {
        b.iter(|| black_box(gtrl_dat_string(black_box(&gtrl))));
    });
}

pub(super) fn bench_gtr_bin(c: &mut Criterion) {
    let data = gtr_bin_bench_data();
    let bytes = match gtr_bin_bytes(&data) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("skipping gtrNN.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_gtr_bin_bytes", |b| {
        b.iter(|| black_box(gtr_bin_bytes(black_box(&data))));
    });
    c.bench_function("parse_gtr_bin_bytes", |b| {
        b.iter(|| black_box(parse_gtr_bin(black_box(&bytes))));
    });
}

pub(super) fn bench_fmsl_bin(c: &mut Criterion) {
    let data = fmsl_bin_bench_data();
    let text = match fmsl_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping fmsl.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_fmsl_bin_text", |b| {
        b.iter(|| black_box(fmsl_bin_string(black_box(&data))));
    });
    c.bench_function("parse_fmsl_bin_text", |b| {
        b.iter(|| {
            black_box(parse_fmsl_bin(
                black_box(&text),
                black_box(data.pad_width),
                black_box(data.energy_count()),
                black_box(data.max_decomposition_channel),
            ))
        });
    });
}

pub(super) fn bench_xsecl_dat(c: &mut Criterion) {
    let data = xsecl_dat_bench_data();
    let text = match xsecl_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xsecl.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xsecl_dat_text", |b| {
        b.iter(|| black_box(xsecl_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xsecl_dat_text", |b| {
        b.iter(|| black_box(parse_xsecl_dat(black_box(&text))));
    });
}

pub(super) fn bench_xsecl_bin(c: &mut Criterion) {
    let data = xsecl_bin_bench_data();
    let text = match xsecl_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xsecl.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xsecl_bin_text", |b| {
        b.iter(|| black_box(xsecl_bin_string(black_box(&data))));
    });
    c.bench_function("parse_xsecl_bin_text", |b| {
        b.iter(|| {
            black_box(parse_xsecl_bin(
                black_box(&text),
                black_box(data.pad_width),
                black_box(data.energy_count()),
            ))
        });
    });
}

pub(super) fn bench_feffl_bin(c: &mut Criterion) {
    let data = feffl_bin_bench_data();
    let text = match feffl_bin_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping feffl.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_feffl_bin_text", |b| {
        b.iter(|| black_box(feffl_bin_string(black_box(&data))));
    });
    c.bench_function("parse_feffl_bin_text", |b| {
        b.iter(|| {
            black_box(parse_feffl_bin(
                black_box(&text),
                black_box(data.pad_width),
                black_box(data.path_count()),
                black_box(data.energy_count()),
                black_box(data.max_decomposition_channel),
            ))
        });
    });
}
