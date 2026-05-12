use std::path::Path;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use refeff_io::{FeffDocument, FeffInput, rdinp};

const FALLBACK_INPUT: &str = r#"
TITLE Cu crystal
EDGE K
SCF 5.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.805 1.805 0.0 1 Cu1 2.55266 1
-1.805 1.805 0.0 1 Cu1 2.55266 2
1.805 -1.805 0.0 1 Cu1 2.55266 3
-1.805 -1.805 0.0 1 Cu1 2.55266 4
END
"#;

fn bench_parse(c: &mut Criterion) {
    let input = bench_input();
    if let Err(err) = FeffInput::parse_str("bench.inp", &input) {
        eprintln!("skipping parse_cu_feff_input benchmark: {err}");
        return;
    }
    c.bench_function("parse_cu_feff_input", |b| {
        b.iter(|| black_box(FeffInput::parse_str("bench.inp", black_box(&input))));
    });
}

fn bench_rdinp_outputs(c: &mut Criterion) {
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
}

fn bench_input() -> String {
    let local_cu =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples/EXAFS/Cu/feff.inp");
    std::fs::read_to_string(local_cu).unwrap_or_else(|_| FALLBACK_INPUT.to_string())
}

criterion_group!(benches, bench_parse, bench_rdinp_outputs);
criterion_main!(benches);
