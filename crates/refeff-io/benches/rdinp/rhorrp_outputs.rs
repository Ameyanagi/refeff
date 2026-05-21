use super::*;

pub(super) fn bench_rhozzp_dat(c: &mut Criterion) {
    let data = rhozzp_dat_bench_data();
    let text = match rhozzp_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping rhozzp.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rhozzp_dat_text", |b| {
        b.iter(|| black_box(rhozzp_dat_string(black_box(&data))));
    });
    c.bench_function("parse_rhozzp_dat_text", |b| {
        b.iter(|| black_box(parse_rhozzp_dat(black_box(&text))));
    });
}

pub(super) fn bench_rhorrp_density_text(c: &mut Criterion) {
    let data = rhorrp_density_bench_data();
    let text = match rhorrp_density_text_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping RHORRP density text benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rhorrp_density_text", |b| {
        b.iter(|| black_box(rhorrp_density_text_string(black_box(&data))));
    });
    c.bench_function("parse_rhorrp_density_text", |b| {
        b.iter(|| black_box(parse_rhorrp_density_text(black_box(&text))));
    });

    let points_bohr = Array2::from_shape_fn((3, data.point_count()), |(axis, point)| {
        0.005 * point as f64 + 0.25 * axis as f64
    });
    let density_per_bohr3 = Array1::from_shape_fn(data.point_count(), |point| {
        let scaled = point as f64 / data.point_count() as f64;
        0.5 * (-2.0 * scaled).exp()
    });
    c.bench_function("convert_rhorrp_density_text_from_bohr", |b| {
        b.iter(|| {
            black_box(rhorrp_density_text_from_bohr(RhorrpDensityTextBohrInput {
                points_bohr: points_bohr.view(),
                density_per_bohr3: density_per_bohr3.view(),
                nearest: None,
            }))
        });
    });
    let text_axes_bohr = Array2::zeros((3, 1));
    c.bench_function("select_rhorrp_density_output_text_from_bohr", |b| {
        b.iter(|| {
            black_box(rhorrp_density_output_from_bohr(
                "density.dat",
                RhorrpDensityOutputBohrInput {
                    origin_bohr: [0.1, -0.2, 0.3],
                    axes_bohr: text_axes_bohr.view(),
                    points_per_axis: &[data.point_count()],
                    points_bohr: points_bohr.view(),
                    density_per_bohr3: density_per_bohr3.view(),
                    nearest: None,
                },
            ))
        });
    });
}

pub(super) fn bench_rhorrp_density_bin(c: &mut Criterion) {
    let data = rhorrp_density_bin_bench_data();
    let bytes = match rhorrp_density_bin_bytes(&data) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("skipping RHORRP density binary benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rhorrp_density_bin", |b| {
        b.iter(|| black_box(rhorrp_density_bin_bytes(black_box(&data))));
    });
    c.bench_function("parse_rhorrp_density_bin", |b| {
        b.iter(|| black_box(parse_rhorrp_density_bin(black_box(&bytes))));
    });

    let axes_bohr = Array2::from_shape_fn(data.axes_angstrom.dim(), |(axis, dimension)| {
        0.1 + 0.3 * axis as f64 + 0.05 * dimension as f64
    });
    let density_per_bohr3 = Array1::from_shape_fn(data.point_count(), |point| {
        let scaled = point as f64 / data.point_count() as f64;
        0.2 * (-2.0 * scaled).exp()
    });
    c.bench_function("convert_rhorrp_density_bin_from_bohr", |b| {
        b.iter(|| {
            black_box(rhorrp_density_bin_from_bohr(RhorrpDensityBinBohrInput {
                origin_bohr: [0.1, -0.2, 0.3],
                axes_bohr: axes_bohr.view(),
                points_per_axis: &data.points_per_axis,
                density_per_bohr3: density_per_bohr3.view(),
            }))
        });
    });
    let points_bohr = Array2::zeros((3, data.point_count()));
    c.bench_function("select_rhorrp_density_output_bin_from_bohr", |b| {
        b.iter(|| {
            black_box(rhorrp_density_output_from_bohr(
                "density.bin",
                RhorrpDensityOutputBohrInput {
                    origin_bohr: [0.1, -0.2, 0.3],
                    axes_bohr: axes_bohr.view(),
                    points_per_axis: &data.points_per_axis,
                    points_bohr: points_bohr.view(),
                    density_per_bohr3: density_per_bohr3.view(),
                    nearest: None,
                },
            ))
        });
    });

    let filenames = [
        "density.bin",
        "density.BIN",
        "density.bin1",
        "archive.tar.bin",
        "density",
        ".bin",
        "density.",
        "density.b",
        "density.binary",
        "density.bin   ",
    ];
    c.bench_function("classify_rhorrp_density_filename", |b| {
        b.iter(|| {
            black_box(
                filenames
                    .iter()
                    .filter(|filename| rhorrp_density_filename_is_binary(black_box(filename)))
                    .count(),
            );
        });
    });
}

pub(super) fn bench_rhorrp_gg_bin(c: &mut Criterion) {
    let slice = RhorrpGgSliceBinData {
        values: Array3::from_shape_fn((64, 48, 48), |(energy, row, column)| {
            let value = 0.0001 * energy as f32 + 0.001 * row as f32 - 0.0007 * column as f32;
            Complex32::new(value, -0.5 * value)
        }),
    };
    let slice_bytes = match rhorrp_gg_slice_bin_bytes(&slice) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("skipping RHORRP gg_slice.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rhorrp_gg_slice_bin", |b| {
        b.iter(|| black_box(rhorrp_gg_slice_bin_bytes(black_box(&slice))));
    });
    c.bench_function("parse_rhorrp_gg_slice_bin", |b| {
        b.iter(|| black_box(parse_rhorrp_gg_slice_bin(black_box(&slice_bytes))));
    });
    c.bench_function("extract_rhorrp_gg_slice_block", |b| {
        b.iter(|| black_box(rhorrp_gg_slice_block(black_box(&slice), 1, 2, 24)));
    });

    let diag = RhorrpGgDiagBinData {
        values: Array4::from_shape_fn((32, 8, 24, 24), |(energy, atom, row, column)| {
            let value = 0.0002 * energy as f32 + 0.002 * atom as f32 + 0.0005 * row as f32
                - 0.0003 * column as f32;
            Complex32::new(value, -0.25 * value)
        }),
    };
    let diag_bytes = match rhorrp_gg_diag_bin_bytes(&diag) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("skipping RHORRP gg_diag.bin benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_rhorrp_gg_diag_bin", |b| {
        b.iter(|| black_box(rhorrp_gg_diag_bin_bytes(black_box(&diag))));
    });
    c.bench_function("parse_rhorrp_gg_diag_bin", |b| {
        b.iter(|| black_box(parse_rhorrp_gg_diag_bin(black_box(&diag_bytes))));
    });
    c.bench_function("extract_rhorrp_gg_diag_matrix", |b| {
        b.iter(|| black_box(rhorrp_gg_diag_matrix(black_box(&diag), 3)));
    });
    c.bench_function("select_rhorrp_gg_pair_matrix", |b| {
        b.iter(|| {
            black_box(rhorrp_gg_pair_matrix(
                black_box(&diag),
                black_box(&slice),
                1,
                2,
                24,
            ))
        });
    });
}

pub(super) fn bench_jzzp_dat(c: &mut Criterion) {
    let data = jzzp_dat_bench_data();
    let text = match jzzp_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping jzzp.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_jzzp_dat_text", |b| {
        b.iter(|| black_box(jzzp_dat_string(black_box(&data))));
    });
    c.bench_function("parse_jzzp_dat_text", |b| {
        b.iter(|| black_box(parse_jzzp_dat(black_box(&text))));
    });
}
