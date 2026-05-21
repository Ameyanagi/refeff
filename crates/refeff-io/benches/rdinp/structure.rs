use super::*;

pub(super) fn bench_structure_outputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let dimensions_text = match rdinp::dimensions_dat_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let atoms_text = match rdinp::atoms_dat_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let geom_text = match rdinp::geom_dat_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let dimensions = match DimensionsDat::parse_str(".dimensions.dat", &dimensions_text) {
        Ok(dimensions) => dimensions,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let atoms = match AtomsDat::parse_str("atoms.dat", &atoms_text) {
        Ok(atoms) => atoms,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };
    let geom = match GeomDat::parse_str("geom.dat", &geom_text) {
        Ok(geom) => geom,
        Err(err) => {
            eprintln!("skipping structure output benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_dimensions_dat", |b| {
        b.iter(|| {
            black_box(DimensionsDat::parse_str(
                ".dimensions.dat",
                black_box(&dimensions_text),
            ))
        });
    });
    c.bench_function("render_dimensions_dat", |b| {
        b.iter(|| black_box(dimensions_dat_string(black_box(&dimensions))));
    });
    c.bench_function("parse_atoms_dat", |b| {
        b.iter(|| black_box(AtomsDat::parse_str("atoms.dat", black_box(&atoms_text))));
    });
    c.bench_function("render_atoms_dat", |b| {
        b.iter(|| black_box(atoms_dat_string(black_box(&atoms))));
    });
    c.bench_function("parse_geom_dat", |b| {
        b.iter(|| black_box(GeomDat::parse_str("geom.dat", black_box(&geom_text))));
    });
    c.bench_function("render_geom_dat", |b| {
        b.iter(|| black_box(geom_dat_string(black_box(&geom))));
    });
}

pub(super) fn bench_cif(c: &mut Criterion) {
    let text = cif_bench_text();
    let document = match parse_cif(&text) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping CIF benchmarks: {err}");
            return;
        }
    };
    if let Err(err) = expand_cif_structure(&document, 1) {
        eprintln!("skipping CIF expansion benchmarks: {err}");
        return;
    }
    if let Err(err) = expand_cif_cluster(&document, 1, 7.0) {
        eprintln!("skipping CIF cluster benchmarks: {err}");
        return;
    }

    c.bench_function("parse_cif_first_data_block", |b| {
        b.iter(|| black_box(parse_cif(black_box(&text))));
    });
    c.bench_function("expand_cif_structure", |b| {
        b.iter(|| black_box(expand_cif_structure(black_box(&document), 1)));
    });
    c.bench_function("expand_cif_cluster_rmax7", |b| {
        b.iter(|| black_box(expand_cif_cluster(black_box(&document), 1, 7.0)));
    });
}
