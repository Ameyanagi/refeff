use super::*;

pub(super) fn bench_energy_outputs(c: &mut Criterion) {
    let edges = match parse_edges_dat(EDGES_DAT_BENCH) {
        Ok(edges) => edges,
        Err(err) => {
            eprintln!("skipping energy output benchmarks: {err}");
            return;
        }
    };
    let chemical = match parse_chemical_dat(CHEMICAL_DAT_BENCH) {
        Ok(chemical) => chemical,
        Err(err) => {
            eprintln!("skipping energy output benchmarks: {err}");
            return;
        }
    };
    let emesh = match parse_emesh_dat(EMESH_DAT_BENCH) {
        Ok(emesh) => emesh,
        Err(err) => {
            eprintln!("skipping energy output benchmarks: {err}");
            return;
        }
    };
    let fpf0 = match parse_fpf0_dat(FPF0_DAT_BENCH) {
        Ok(fpf0) => fpf0,
        Err(err) => {
            eprintln!("skipping energy output benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_edges_dat", |b| {
        b.iter(|| black_box(parse_edges_dat(black_box(EDGES_DAT_BENCH))));
    });
    c.bench_function("render_edges_dat", |b| {
        b.iter(|| black_box(edges_dat_string(black_box(&edges))));
    });
    c.bench_function("parse_chemical_dat", |b| {
        b.iter(|| black_box(parse_chemical_dat(black_box(CHEMICAL_DAT_BENCH))));
    });
    c.bench_function("render_chemical_dat", |b| {
        b.iter(|| black_box(chemical_dat_string(black_box(&chemical))));
    });
    c.bench_function("parse_emesh_dat", |b| {
        b.iter(|| black_box(parse_emesh_dat(black_box(EMESH_DAT_BENCH))));
    });
    c.bench_function("render_emesh_dat", |b| {
        b.iter(|| black_box(emesh_dat_string(black_box(&emesh))));
    });
    c.bench_function("parse_fpf0_dat", |b| {
        b.iter(|| black_box(parse_fpf0_dat(black_box(FPF0_DAT_BENCH))));
    });
    c.bench_function("render_fpf0_dat", |b| {
        b.iter(|| black_box(fpf0_dat_string(black_box(&fpf0))));
    });
}
