use super::*;

pub(super) fn bench_crpa_dat(c: &mut Criterion) {
    let data = crpa_dat_bench_data();
    let text = match crpa_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping crpa.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_crpa_dat_text", |b| {
        b.iter(|| black_box(crpa_dat_string(black_box(&data))));
    });
    c.bench_function("parse_crpa_dat_text", |b| {
        b.iter(|| black_box(parse_crpa_dat(black_box(&text))));
    });
}

pub(super) fn bench_loss_dat(c: &mut Criterion) {
    let data = loss_dat_bench_data();
    let text = match loss_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping loss.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_loss_dat_text", |b| {
        b.iter(|| black_box(loss_dat_string(black_box(&data))));
    });
    c.bench_function("parse_loss_dat_text", |b| {
        b.iter(|| black_box(parse_loss_dat(black_box(&text))));
    });
}

pub(super) fn bench_osc_str_dat(c: &mut Criterion) {
    let data = osc_str_dat_bench_data();
    let text = match osc_str_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping osc_str.dat benchmarks: {err}");
            return;
        }
    };
    let edge = fullspectrum_edge_assembly_bench_data(5.1234);
    c.bench_function("render_osc_str_dat_text", |b| {
        b.iter(|| black_box(osc_str_dat_string(black_box(&data))));
    });
    c.bench_function("parse_osc_str_dat_text", |b| {
        b.iter(|| black_box(parse_osc_str_dat(black_box(&text))));
    });
    c.bench_function("osc_str_row_from_fullspectrum_edge", |b| {
        b.iter(|| {
            black_box(osc_str_row_from_fullspectrum_edge(
                black_box("Cu"),
                black_box("K"),
                black_box(&edge),
            ))
        });
    });
}

pub(super) fn bench_sumrules_dat(c: &mut Criterion) {
    let data = sumrules_dat_bench_data();
    let text = match sumrules_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping sumrules.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_sumrules_dat_text", |b| {
        b.iter(|| black_box(sumrules_dat_string(black_box(&data))));
    });
    c.bench_function("parse_sumrules_dat_text", |b| {
        b.iter(|| black_box(parse_sumrules_dat(black_box(&text))));
    });
}

pub(super) fn bench_drude_dat(c: &mut Criterion) {
    let data = drude_dat_bench_data();
    let text = match drude_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping drude.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_drude_dat_text", |b| {
        b.iter(|| black_box(drude_dat_string(black_box(&data))));
    });
    c.bench_function("parse_drude_dat_text", |b| {
        b.iter(|| black_box(parse_drude_dat(black_box(&text))));
    });
}

pub(super) fn bench_hamaker_dat(c: &mut Criterion) {
    let data = hamaker_dat_bench_data();
    let text = match hamaker_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping hamaker.dat benchmarks: {err}");
            return;
        }
    };
    let epsilon_minus_one = Array1::from_shape_fn(data.point_count(), |index| {
        let phase = index as f64 * 0.001;
        Complex64::new(0.1 + 0.02 * phase.sin(), 0.08 + 0.01 * phase.cos())
    });
    c.bench_function("render_hamaker_dat_text", |b| {
        b.iter(|| black_box(hamaker_dat_string(black_box(&data))));
    });
    c.bench_function("parse_hamaker_dat_text", |b| {
        b.iter(|| black_box(parse_hamaker_dat(black_box(&text))));
    });
    c.bench_function("hamaker_dat_from_fullspectrum_epsilon_8192", |b| {
        b.iter(|| {
            black_box(hamaker_dat_from_fullspectrum_epsilon(
                Vec::new(),
                black_box(data.omega.view()),
                black_box(epsilon_minus_one.view()),
            ))
        });
    });
}

pub(super) fn bench_exc_dat(c: &mut Criterion) {
    let data = exc_dat_bench_data();
    let text = match exc_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping exc.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_exc_dat_text", |b| {
        b.iter(|| black_box(exc_dat_string(black_box(&data))));
    });
    c.bench_function("parse_exc_dat_text", |b| {
        b.iter(|| black_box(parse_exc_dat(black_box(&text))));
    });
    c.bench_function("sfconv_rdeps_from_exc_dat_128", |b| {
        b.iter(|| black_box(sfconv_rdeps_from_exc_dat(black_box(&data), black_box(256))));
    });
    c.bench_function("sfconv_rdeps_fallback_exc_dat_string", |b| {
        b.iter(|| black_box(sfconv_rdeps_fallback_exc_dat_string(black_box(0.47))));
    });
}

pub(super) fn bench_mpse_dat(c: &mut Criterion) {
    let data = mpse_dat_bench_data();
    let text = match mpse_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping mpse.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_mpse_dat_text", |b| {
        b.iter(|| black_box(mpse_dat_string(black_box(&data))));
    });
    c.bench_function("parse_mpse_dat_text", |b| {
        b.iter(|| black_box(parse_mpse_dat(black_box(&text))));
    });
}

pub(super) fn bench_rixs_map(c: &mut Criterion) {
    let data = rixs_map_bench_data();
    let text = match rixs_map_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping RIXS map benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rixs_map_text", |b| {
        b.iter(|| black_box(rixs_map_string(black_box(&data))));
    });
    c.bench_function("parse_rixs_map_text", |b| {
        b.iter(|| black_box(parse_rixs_map(black_box(&text))));
    });
}

pub(super) fn bench_rixs_line(c: &mut Criterion) {
    let data = rixs_line_bench_data();
    let text = match rixs_line_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping RIXS line benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rixs_line_text", |b| {
        b.iter(|| black_box(rixs_line_string(black_box(&data))));
    });
    c.bench_function("parse_rixs_line_text", |b| {
        b.iter(|| black_box(parse_rixs_line(black_box(&text))));
    });
}
