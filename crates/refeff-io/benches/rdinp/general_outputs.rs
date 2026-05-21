use super::*;

pub(super) fn bench_potential_outputs(c: &mut Criterion) {
    let state = PotOutputBenchState::new();
    c.bench_function("render_wpot_potential_dat_outputs", |b| {
        b.iter(|| black_box(potential_dat_outputs(black_box(state.input()))));
    });

    let pot = pot_bin_bench_data();
    let apot = apot_bin_wpot_bench_data(&pot);
    c.bench_function("render_wpot_from_pot_apot_bins", |b| {
        b.iter(|| {
            black_box(potential_dat_outputs_from_bins(
                black_box(&pot),
                black_box(&apot),
            ))
        });
    });
}

pub(super) fn bench_mtdp(c: &mut Criterion) {
    let data = mtdp_bench_data();
    let text = match mtdp_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping mtdp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_mtdp_text", |b| {
        b.iter(|| black_box(mtdp_string(black_box(&data))));
    });
    c.bench_function("parse_mtdp_text", |b| {
        b.iter(|| black_box(parse_mtdp(black_box(&text))));
    });
}

pub(super) fn bench_list_dat(c: &mut Criterion) {
    let data = list_dat_bench_data();
    let text = match list_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping list.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_list_dat_text", |b| {
        b.iter(|| black_box(list_dat_string(black_box(&data))));
    });
    c.bench_function("parse_list_dat_text", |b| {
        b.iter(|| black_box(parse_list_dat(black_box(&text))));
    });
}

pub(super) fn bench_log_dat(c: &mut Criterion) {
    let data = log_dat_bench_data();
    let text = match log_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping log.dat benchmarks: {err}");
            return;
        }
    };
    let module_log = match parse_module_log_dat(MODULE_LOG_BENCH) {
        Ok(module_log) => module_log,
        Err(err) => {
            eprintln!("skipping module log benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_log_dat_text", |b| {
        b.iter(|| black_box(log_dat_string(black_box(&data))));
    });
    c.bench_function("parse_log_dat_text", |b| {
        b.iter(|| black_box(parse_log_dat(black_box(&text))));
    });
    c.bench_function("render_module_log_dat_text", |b| {
        b.iter(|| black_box(module_log_dat_string(black_box(&module_log))));
    });
    c.bench_function("parse_module_log_dat_text", |b| {
        b.iter(|| black_box(parse_module_log_dat(black_box(MODULE_LOG_BENCH))));
    });
}

pub(super) fn bench_run_output(c: &mut Criterion) {
    let stdout = run_stdout_bench_data();
    let stdout_text = match run_stdout_string(&stdout) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping run stdout benchmarks: {err}");
            return;
        }
    };
    let stderr = run_stderr_bench_data();
    let stderr_text = match run_stderr_string(&stderr) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping run stderr benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_run_stdout_text", |b| {
        b.iter(|| black_box(parse_run_stdout(black_box(&stdout_text))));
    });
    c.bench_function("render_run_stdout_text", |b| {
        b.iter(|| black_box(run_stdout_string(black_box(&stdout))));
    });
    c.bench_function("parse_run_stderr_text", |b| {
        b.iter(|| black_box(parse_run_stderr(black_box(&stderr_text))));
    });
    c.bench_function("render_run_stderr_text", |b| {
        b.iter(|| black_box(run_stderr_string(black_box(&stderr))));
    });
}

pub(super) fn bench_paths_dat(c: &mut Criterion) {
    let data = paths_dat_bench_data();
    let text = match paths_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping paths.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_paths_dat_text", |b| {
        b.iter(|| black_box(paths_dat_string(black_box(&data))));
    });
    c.bench_function("parse_paths_dat_text", |b| {
        b.iter(|| black_box(parse_paths_dat(black_box(&text))));
    });
}

pub(super) fn bench_dym(c: &mut Criterion) {
    let data = dym_bench_data();
    let text = match dym_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping .dym benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_dym_text", |b| {
        b.iter(|| black_box(dym_string(black_box(&data))));
    });
    c.bench_function("parse_dym_text", |b| {
        b.iter(|| black_box(parse_dym(black_box(&text))));
    });
    c.bench_function("mass_weight_dym_matrix", |b| {
        b.iter(|| black_box(data.mass_weighted_dynamical_matrix()));
    });
}

pub(super) fn bench_grid_inp(c: &mut Criterion) {
    let data = grid_inp_bench_data();
    let text = match grid_inp_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping grid.inp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_grid_inp_text", |b| {
        b.iter(|| black_box(grid_inp_string(black_box(&data))));
    });
    c.bench_function("parse_grid_inp_text", |b| {
        b.iter(|| black_box(parse_grid_inp(black_box(&text))));
    });
}

pub(super) fn bench_config_inp(c: &mut Criterion) {
    let data = config_inp_bench_data();
    let text = match config_inp_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping config.inp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_config_inp_text", |b| {
        b.iter(|| black_box(config_inp_string(black_box(&data))));
    });
    c.bench_function("parse_config_inp_text", |b| {
        b.iter(|| black_box(parse_config_inp(black_box(&text))));
    });
}

pub(super) fn bench_spring_inp(c: &mut Criterion) {
    let data = spring_inp_bench_data();
    let text = match spring_inp_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping spring.inp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_spring_inp_text", |b| {
        b.iter(|| black_box(spring_inp_string(black_box(&data))));
    });
    c.bench_function("parse_spring_inp_text", |b| {
        b.iter(|| black_box(parse_spring_inp(black_box(&text))));
    });
}
