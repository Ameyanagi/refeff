use super::*;

pub(super) fn bench_parse(c: &mut Criterion) {
    let input = bench_input();
    if let Err(err) = FeffInput::parse_str("bench.inp", &input) {
        eprintln!("skipping parse_cu_feff_input benchmark: {err}");
        return;
    }
    c.bench_function("parse_cu_feff_input", |b| {
        b.iter(|| black_box(FeffInput::parse_str("bench.inp", black_box(&input))));
    });
}

pub(super) fn bench_rdinp_outputs(c: &mut Criterion) {
    let input = bench_input();
    let parsed = match FeffInput::parse_str("bench.inp", &input) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("skipping render_rdinp_text_outputs benchmark: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping render_rdinp_text_outputs benchmark: {err}");
            return;
        }
    };

    c.bench_function("render_rdinp_text_outputs", |b| {
        b.iter(|| black_box(rdinp::text_outputs(black_box(&document))));
    });
    c.bench_function("render_rdinp_log_dat", |b| {
        b.iter(|| black_box(rdinp::rdinp_log_dat_string(black_box(&document))));
    });
}
