use super::*;

pub(super) fn bench_compton_helpers(c: &mut Criterion) {
    c.bench_function("compton_rotation_axis_angle", |b| {
        b.iter(|| {
            black_box(compton_rotation_axis_angle(
                black_box([0.0, 0.0, 1.0]),
                black_box([0.35, -0.25, 0.92]),
            ))
        });
    });

    let grid_input = ComptonGridInput {
        ns: 16,
        nphi: 17,
        nz: 32,
        nzp: 33,
        smax: 0.0,
        phimax: std::f64::consts::PI,
        zmax: 1.2,
        zpmax: 1.5,
        norman_radius: 2.25,
        qhat: [0.35, -0.25, 0.92],
    };
    c.bench_function("compton_build_grid_16_17_32_33", |b| {
        b.iter(|| black_box(compton_build_grid(black_box(grid_input))));
    });

    let Ok(grid) = compton_build_grid(grid_input) else {
        return;
    };
    let jzzp = Array2::from_shape_fn((grid.nz(), grid.nzp()).f(), |(iz, izp)| {
        let iz = iz as f64 + 1.0;
        let izp = izp as f64 + 1.0;
        0.12 * iz + 0.07 * izp + 0.015 * iz * izp
    });
    c.bench_function("compton_profile_cosine_32x33", |b| {
        b.iter(|| {
            black_box(compton_profile(
                black_box(&grid),
                black_box(jzzp.view()),
                black_box(ComptonProfileInput {
                    pq: 1.35,
                    window: ComptonWindow::CosineSquared,
                    window_cutoff: 1.0,
                }),
            ))
        });
    });

    let Ok(profile_grid) = compton_build_grid(ComptonGridInput {
        ns: 32,
        nphi: 32,
        nz: 32,
        nzp: 120,
        smax: 2.642_889_499_664_3,
        phimax: std::f64::consts::TAU,
        zmax: 2.642_889_499_664_3,
        zpmax: 10.0,
        norman_radius: 0.0,
        qhat: [0.0, 0.0, 1.0],
    }) else {
        return;
    };
    let profile_jzzp =
        Array2::from_shape_fn((profile_grid.nz(), profile_grid.nzp()).f(), |(iz, izp)| {
            let z = profile_grid.z[iz];
            let zp = profile_grid.zp[izp];
            (-(z * z) * 0.18 - (zp * zp) * 0.025).exp() * (1.0 + 0.01 * izp as f64)
        });
    let momentum = Array1::from_iter((0..1000).map(|index| 5.0 * index as f64 / 999.0));
    c.bench_function("compton_profile_loop_cosine_1000x32x120", |b| {
        b.iter(|| {
            black_box(
                momentum
                    .iter()
                    .map(|&pq| {
                        compton_profile(
                            black_box(&profile_grid),
                            black_box(profile_jzzp.view()),
                            black_box(ComptonProfileInput {
                                pq,
                                window: ComptonWindow::CosineSquared,
                                window_cutoff: 0.0,
                            }),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>(),
            )
        });
    });
    c.bench_function("compton_profiles_batch_cosine_1000x32x120", |b| {
        b.iter(|| {
            black_box(compton_profiles(
                black_box(&profile_grid),
                black_box(profile_jzzp.view()),
                black_box(momentum.view()),
                black_box(ComptonWindow::CosineSquared),
                black_box(0.0),
            ))
        });
    });

    let Ok(jzzp_grid) = compton_build_grid(ComptonGridInput {
        ns: 8,
        nphi: 9,
        nz: 8,
        nzp: 9,
        ..grid_input
    }) else {
        return;
    };
    c.bench_function("compton_jzzp_stub_8_9_8_9", |b| {
        b.iter(|| black_box(compton_jzzp(black_box(&jzzp_grid), sample_compton_density)));
    });
    c.bench_function("compton_rhozzp_slice_stub_1000", |b| {
        b.iter(|| {
            black_box(compton_rhozzp_slice(
                black_box(&jzzp_grid),
                black_box(ComptonRhoZzpInput {
                    sample_count: 1000,
                    base_z: 0.01,
                }),
                sample_compton_density,
            ))
        });
    });
}

fn sample_compton_density(r: [f64; 3], rp: [f64; 3]) -> Result<f64, refeff_core::ComptonError> {
    let r2 = r.iter().map(|value| value * value).sum::<f64>();
    let rp2 = rp.iter().map(|value| value * value).sum::<f64>();
    let dot = r
        .iter()
        .zip(rp)
        .map(|(left, right)| *left * right)
        .sum::<f64>();
    Ok((-r2 - 0.5 * rp2).exp() + 0.1 * dot)
}
