use super::*;

pub(super) fn bench_control_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };
    let band_text = rdinp::band_inp_string();
    let fullspectrum_text = rdinp::fullspectrum_inp_string();
    let opcons_text = match rdinp::opcons_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };
    let band = match BandInput::parse_str("band.inp", &band_text) {
        Ok(band) => band,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };
    let fullspectrum = match FullSpectrumInput::parse_str("fullspectrum.inp", &fullspectrum_text) {
        Ok(fullspectrum) => fullspectrum,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };
    let opcons = match OpconsInput::parse_str("opcons.inp", &opcons_text) {
        Ok(opcons) => opcons,
        Err(err) => {
            eprintln!("skipping control input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_band_inp", |b| {
        b.iter(|| black_box(BandInput::parse_str("band.inp", black_box(&band_text))));
    });
    c.bench_function("render_band_inp", |b| {
        b.iter(|| black_box(band_input_string(black_box(&band))));
    });
    c.bench_function("parse_fullspectrum_inp", |b| {
        b.iter(|| {
            black_box(FullSpectrumInput::parse_str(
                "fullspectrum.inp",
                black_box(&fullspectrum_text),
            ))
        });
    });
    c.bench_function("parse_fullspectrum_rdop_options", |b| {
        b.iter(|| {
            black_box(parse_fullspectrum_options(black_box(
                FULLSPECTRUM_OPTIONS_BENCH,
            )))
        });
    });
    c.bench_function("render_fullspectrum_inp", |b| {
        b.iter(|| black_box(fullspectrum_input_string(black_box(&fullspectrum))));
    });
    c.bench_function("parse_opcons_inp", |b| {
        b.iter(|| {
            black_box(OpconsInput::parse_str(
                "opcons.inp",
                black_box(&opcons_text),
            ))
        });
    });
    c.bench_function("render_opcons_inp", |b| {
        b.iter(|| black_box(opcons_input_string(black_box(&opcons))));
    });
}

pub(super) fn bench_shared_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let global_text = match rdinp::global_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let compton_text = match rdinp::compton_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let eels_text = match rdinp::eels_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let global = match GlobalInput::parse_str("global.inp", &global_text) {
        Ok(global) => global,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let compton = match ComptonInput::parse_str("compton.inp", &compton_text) {
        Ok(compton) => compton,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };
    let eels = match EelsInput::parse_str("eels.inp", &eels_text) {
        Ok(eels) => eels,
        Err(err) => {
            eprintln!("skipping shared module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_global_inp", |b| {
        b.iter(|| {
            black_box(GlobalInput::parse_str(
                "global.inp",
                black_box(&global_text),
            ))
        });
    });
    c.bench_function("render_global_inp", |b| {
        b.iter(|| black_box(global_input_string(black_box(&global))));
    });
    c.bench_function("parse_compton_inp", |b| {
        b.iter(|| {
            black_box(ComptonInput::parse_str(
                "compton.inp",
                black_box(&compton_text),
            ))
        });
    });
    c.bench_function("render_compton_inp", |b| {
        b.iter(|| black_box(compton_input_string(black_box(&compton))));
    });
    c.bench_function("parse_eels_inp", |b| {
        b.iter(|| black_box(EelsInput::parse_str("eels.inp", black_box(&eels_text))));
    });
    c.bench_function("render_eels_inp", |b| {
        b.iter(|| black_box(eels_input_string(black_box(&eels))));
    });
}

pub(super) fn bench_phase_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping phase module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping phase module input benchmarks: {err}");
            return;
        }
    };
    let xsph_text = match rdinp::xsph_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping phase module input benchmarks: {err}");
            return;
        }
    };
    let xsph = match XsphInput::parse_str("xsph.inp", &xsph_text) {
        Ok(xsph) => xsph,
        Err(err) => {
            eprintln!("skipping phase module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_xsph_inp", |b| {
        b.iter(|| black_box(XsphInput::parse_str("xsph.inp", black_box(&xsph_text))));
    });
    c.bench_function("render_xsph_inp", |b| {
        b.iter(|| black_box(xsph_input_string(black_box(&xsph))));
    });
}

pub(super) fn bench_potential_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping potential module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping potential module input benchmarks: {err}");
            return;
        }
    };
    let pot_text = match rdinp::pot_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping potential module input benchmarks: {err}");
            return;
        }
    };
    let pot = match PotInput::parse_str("pot.inp", &pot_text) {
        Ok(pot) => pot,
        Err(err) => {
            eprintln!("skipping potential module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_pot_inp", |b| {
        b.iter(|| black_box(PotInput::parse_str("pot.inp", black_box(&pot_text))));
    });
    c.bench_function("render_pot_inp", |b| {
        b.iter(|| black_box(pot_input_string(black_box(&pot))));
    });
}

pub(super) fn bench_scalar_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", "CRPA 2 3.5\nHUBBARD 4.0 0.5 1.5 2\nEND\n")
    {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping scalar module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping scalar module input benchmarks: {err}");
            return;
        }
    };
    let crpa_text = rdinp::crpa_inp_string(&document);
    let hubbard_text = rdinp::hubbard_inp_string(&document);
    let screen_text = rdinp::screen_inp_string();
    let crpa = match CrpaInput::parse_str("crpa.inp", &crpa_text) {
        Ok(crpa) => crpa,
        Err(err) => {
            eprintln!("skipping scalar module input benchmarks: {err}");
            return;
        }
    };
    let hubbard = match HubbardInput::parse_str("hubbard.inp", &hubbard_text) {
        Ok(hubbard) => hubbard,
        Err(err) => {
            eprintln!("skipping scalar module input benchmarks: {err}");
            return;
        }
    };
    let screen = match ScreenInput::parse_str("screen.inp", &screen_text) {
        Ok(screen) => screen,
        Err(err) => {
            eprintln!("skipping scalar module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_crpa_inp", |b| {
        b.iter(|| black_box(CrpaInput::parse_str("crpa.inp", black_box(&crpa_text))));
    });
    c.bench_function("render_crpa_inp", |b| {
        b.iter(|| black_box(crpa_input_string(black_box(&crpa))));
    });
    c.bench_function("parse_hubbard_inp", |b| {
        b.iter(|| {
            black_box(HubbardInput::parse_str(
                "hubbard.inp",
                black_box(&hubbard_text),
            ))
        });
    });
    c.bench_function("render_hubbard_inp", |b| {
        b.iter(|| black_box(hubbard_input_string(black_box(&hubbard))));
    });
    c.bench_function("parse_screen_inp", |b| {
        b.iter(|| {
            black_box(ScreenInput::parse_str(
                "screen.inp",
                black_box(&screen_text),
            ))
        });
    });
    c.bench_function("render_screen_inp", |b| {
        b.iter(|| black_box(screen_input_string(black_box(&screen))));
    });
}

pub(super) fn bench_path_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str("bench.inp", FALLBACK_INPUT) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let paths_text = match rdinp::paths_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let sfconv_text = match rdinp::sfconv_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let dmdw_text = match rdinp::dmdw_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let fms_text = match rdinp::fms_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let genfmt_text = match rdinp::genfmt_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let paths = match PathsInput::parse_str("paths.inp", &paths_text) {
        Ok(paths) => paths,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let sfconv = match SfconvInput::parse_str("sfconv.inp", &sfconv_text) {
        Ok(sfconv) => sfconv,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let dmdw = match DmdwInput::parse_str("dmdw.inp", &dmdw_text) {
        Ok(dmdw) => dmdw,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let fms = match FmsInput::parse_str("fms.inp", &fms_text) {
        Ok(fms) => fms,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let genfmt = match GenfmtInput::parse_str("genfmt.inp", &genfmt_text) {
        Ok(genfmt) => genfmt,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };
    let enabled_dmdw = match DmdwInput::parse_str("dmdw.inp", DMDW_ENABLED_INPUT_BENCH) {
        Ok(dmdw) => dmdw,
        Err(err) => {
            eprintln!("skipping path module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_paths_inp", |b| {
        b.iter(|| black_box(PathsInput::parse_str("paths.inp", black_box(&paths_text))));
    });
    c.bench_function("render_paths_inp", |b| {
        b.iter(|| black_box(paths_input_string(black_box(&paths))));
    });
    c.bench_function("parse_sfconv_inp", |b| {
        b.iter(|| {
            black_box(SfconvInput::parse_str(
                "sfconv.inp",
                black_box(&sfconv_text),
            ))
        });
    });
    let so2conv_header = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
    );
    c.bench_function("parse_so2conv_material_header", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_material_input_from_header(
                "xmu.dat",
                black_box(so2conv_header),
            ))
        });
    });
    c.bench_function("scan_so2conv_header_status", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_header_from_text(
                "xmu.dat",
                black_box(so2conv_header),
            ))
        });
    });
    let so2conv_xmu_target = SfconvSo2convTarget {
        file_name: "xmu.dat".to_string(),
        kind: SfconvSo2convTargetKind::Xmu,
    };
    let so2conv_xmu_text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "  100.000 0.000 0.000 1.00000E+00 9.00000E-01 1.00000E-01\n",
    );
    c.bench_function("parse_so2conv_xmu_target_data", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_target_data_from_text(
                "xmu.dat",
                black_box(&so2conv_xmu_target),
                black_box(so2conv_xmu_text),
            ))
        });
    });
    let so2conv_feff_path_target = SfconvSo2convTarget {
        file_name: "feff0001.dat".to_string(),
        kind: SfconvSo2convTargetKind::FeffPath,
    };
    let so2conv_feff_path_text = concat!(
        "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
        "Mu= 18.76000 kf= 1.230000\n",
        " ------------------------------------------------------------------------------\n",
        "#    3   4.250   2.7500 reff path metadata\n",
        "#       k          phase @#\n",
        "  0.500  1.10000E-01  2.20000E+00 -3.30000E-01  8.80000E-01  4.40000E+00  6.60000E-01\n",
    );
    c.bench_function("parse_so2conv_feff_path_target_data", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_target_data_from_text(
                "feff0001.dat",
                black_box(&so2conv_feff_path_target),
                black_box(so2conv_feff_path_text),
            ))
        });
    });
    let so2conv_feff_path_data = match sfconv_so2conv_target_data_from_text(
        "feff0001.dat",
        &so2conv_feff_path_target,
        so2conv_feff_path_text,
    ) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("skipping render_so2conv_feff_path_target_data benchmark: {err}");
            return;
        }
    };
    c.bench_function("render_so2conv_feff_path_target_data", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_target_data_string(black_box(
                &so2conv_feff_path_data,
            )))
        });
    });
    let so2conv_feff_path_body = match &so2conv_feff_path_data {
        SfconvSo2convTargetData::FeffPath { data, .. } => data,
        _ => {
            eprintln!("skipping apply_so2conv_feff_path_averages benchmark: wrong target kind");
            return;
        }
    };
    let so2conv_path_averages = vec![
        SfconvPathAverage {
            amplitude_reduction: 0.92,
            phase_shift: 0.015,
            normalization: 0.05,
        };
        so2conv_feff_path_body.point_count()
    ];
    c.bench_function("apply_so2conv_feff_path_averages", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_feff_path_data_from_averages(
                black_box(so2conv_feff_path_body),
                black_box(&so2conv_path_averages),
            ))
        });
    });
    let specfunct = so2conv_specfunct_bench_data();
    let specfunct_bytes = match specfunct_dat_bytes(&specfunct) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("skipping specfunct.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_specfunct_dat_bytes", |b| {
        b.iter(|| specfunct_dat_bytes(black_box(&specfunct)));
    });
    c.bench_function("parse_specfunct_dat_bytes", |b| {
        b.iter(|| parse_specfunct_dat(black_box(&specfunct_bytes)));
    });
    c.bench_function("interpolate_specfunct_dat_momentum", |b| {
        b.iter(|| {
            sfconv_specfunct_interpolate_momentum(
                black_box(&specfunct),
                black_box(specfunct.spectral_info[[16, 0]]),
            )
        });
    });
    let mut specfunct_exafs = specfunct.clone();
    specfunct_exafs.asymmetric_phase = 0;
    let specfunct_exafs_input = so2conv_exafs_bench_data(128);
    let specfunct_exafs_momentum =
        Array1::from_shape_fn(64, |row| specfunct.spectral_info[[row % 32, 0]]);
    c.bench_function("convolve_specfunct_exafs_rows_64", |b| {
        b.iter(|| {
            sfconv_specfunct_exafs_convolution_rows(black_box(SfconvSpecfunctExafsRowsInput {
                cache: &specfunct_exafs,
                signal_energy: specfunct_exafs_input.signal_energy.view(),
                real_signal: specfunct_exafs_input.real_signal.view(),
                imaginary_signal: specfunct_exafs_input.imaginary_signal.view(),
                original_magnitude: specfunct_exafs_input.original_magnitude.view(),
                original_phase: specfunct_exafs_input.original_phase.view(),
                phase_minus_2kr: specfunct_exafs_input.phase_minus_2kr.view(),
                photoelectron_momentum: specfunct_exafs_momentum.view(),
                active_len: specfunct_exafs_momentum.len(),
                chemical_potential: 0.0,
                cutoff: true,
                plasma_frequency: 1.0,
            }))
        });
    });
    let specfunct_xanes_preparation = so2conv_xanes_preparation_bench_data(128);
    let specfunct_xanes_momentum =
        Array1::from_shape_fn(64, |row| specfunct.spectral_info[[row % 32, 0]]);
    c.bench_function("convolve_specfunct_xanes_rows_64", |b| {
        b.iter(|| {
            sfconv_specfunct_xanes_convolution_rows(black_box(SfconvSpecfunctXanesRowsInput {
                cache: &specfunct,
                prepared: &specfunct_xanes_preparation,
                photoelectron_momentum: specfunct_xanes_momentum.view(),
                active_len: specfunct_xanes_momentum.len(),
                chemical_potential: 0.0,
                cutoff: false,
                plasma_frequency: 1.0,
            }))
        });
    });
    let so2conv_target_input = SfconvInput {
        control: refeff_io::SfconvControl {
            msfconv: 1,
            ipse: 0,
            ipsk: 0,
        },
        window: refeff_io::SfconvWindow {
            wsigk: 0.0,
            cen: 0.0,
        },
        spectrum: refeff_io::SfconvSpectrum { ispec: 0, ipr6: 3 },
        cfname: "NULL".to_string(),
    };
    let so2conv_target_list = ListDatData {
        titles: vec!["bench".to_string()],
        entries: (1..=64)
            .map(|path_index| ListDatEntry {
                path_index,
                sigma2: 0.0,
                amplitude_ratio: 1.0,
                degeneracy: 4.0,
                leg_count: 2,
                effective_half_path_length_angstrom: 2.5,
            })
            .collect(),
    };
    c.bench_function("select_so2conv_targets_64_paths", |b| {
        b.iter(|| {
            black_box(sfconv_so2conv_targets(
                black_box(&so2conv_target_input),
                Some(black_box(&so2conv_target_list)),
                None,
            ))
        });
    });
    c.bench_function("render_sfconv_inp", |b| {
        b.iter(|| black_box(sfconv_input_string(black_box(&sfconv))));
    });
    c.bench_function("parse_dmdw_inp_disabled", |b| {
        b.iter(|| black_box(DmdwInput::parse_str("dmdw.inp", black_box(&dmdw_text))));
    });
    c.bench_function("render_dmdw_inp_disabled", |b| {
        b.iter(|| black_box(dmdw_input_string(black_box(&dmdw))));
    });
    c.bench_function("parse_dmdw_inp_enabled", |b| {
        b.iter(|| {
            black_box(DmdwInput::parse_str(
                "dmdw.inp",
                black_box(DMDW_ENABLED_INPUT_BENCH),
            ))
        });
    });
    c.bench_function("render_dmdw_inp_enabled", |b| {
        b.iter(|| black_box(dmdw_input_string(black_box(&enabled_dmdw))));
    });
    c.bench_function("parse_fms_inp", |b| {
        b.iter(|| black_box(FmsInput::parse_str("fms.inp", black_box(&fms_text))));
    });
    c.bench_function("render_fms_inp", |b| {
        b.iter(|| black_box(fms_input_string(black_box(&fms))));
    });
    c.bench_function("parse_genfmt_inp", |b| {
        b.iter(|| {
            black_box(GenfmtInput::parse_str(
                "genfmt.inp",
                black_box(&genfmt_text),
            ))
        });
    });
    c.bench_function("render_genfmt_inp", |b| {
        b.iter(|| black_box(genfmt_input_string(black_box(&genfmt))));
    });
}

pub(super) fn bench_dmdw_out(c: &mut Criterion) {
    let data = match parse_dmdw_out(DMDW_OUT_BENCH) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("skipping dmdw.out benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_dmdw_out_text", |b| {
        b.iter(|| black_box(parse_dmdw_out(black_box(DMDW_OUT_BENCH))));
    });
    c.bench_function("render_dmdw_out_text", |b| {
        b.iter(|| black_box(dmdw_out_string(black_box(&data))));
    });
}

pub(super) fn bench_spectrum_module_inputs(c: &mut Criterion) {
    let input = match FeffInput::parse_str(
        "bench.inp",
        r#"
EDGE K L1 VAL
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 2
EXAFS 20
S02 0.8
CORRECTIONS 1.1 2.2
CRITERIA 3.3 4.4
DEBYE 300.0 400.0
ABSOLUTE
NRIXS 1 1.0 2.0 -3.0
LDEC 5
TEMP 0.25
RIXS 1.0 2.0 3.0
LDOS -30.0 20.0 0.1 151 2
FMS 4.5 1 2 0.002 0.003 8.0
EXCHANGE 5
SPIN 1 0.0 0.0 1.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
    ) {
        Ok(input) => input,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let document = match FeffDocument::from_input(&input) {
        Ok(document) => document,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let ff2x_text = match rdinp::ff2x_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let ldos_text = match rdinp::ldos_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let rixs_text = match rdinp::rixs_inp_string(&document) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let ff2x = match Ff2xInput::parse_str("ff2x.inp", &ff2x_text) {
        Ok(ff2x) => ff2x,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let ldos = match LdosInput::parse_str("ldos.inp", &ldos_text) {
        Ok(ldos) => ldos,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };
    let rixs = match RixsInput::parse_str("rixs.inp", &rixs_text) {
        Ok(rixs) => rixs,
        Err(err) => {
            eprintln!("skipping spectrum module input benchmarks: {err}");
            return;
        }
    };

    c.bench_function("parse_ff2x_inp", |b| {
        b.iter(|| black_box(Ff2xInput::parse_str("ff2x.inp", black_box(&ff2x_text))));
    });
    c.bench_function("render_ff2x_inp", |b| {
        b.iter(|| black_box(ff2x_input_string(black_box(&ff2x))));
    });
    c.bench_function("parse_ldos_inp", |b| {
        b.iter(|| black_box(LdosInput::parse_str("ldos.inp", black_box(&ldos_text))));
    });
    c.bench_function("render_ldos_inp", |b| {
        b.iter(|| black_box(ldos_input_string(black_box(&ldos))));
    });
    c.bench_function("parse_rixs_inp", |b| {
        b.iter(|| black_box(RixsInput::parse_str("rixs.inp", black_box(&rixs_text))));
    });
    c.bench_function("render_rixs_inp", |b| {
        b.iter(|| black_box(rixs_input_string(black_box(&rixs))));
    });
}

pub(super) fn bench_density_input(c: &mut Criterion) {
    let density = match DensityInput::parse_str("density.inp", DENSITY_INPUT_BENCH) {
        Ok(density) => density,
        Err(err) => {
            eprintln!("skipping density.inp benchmarks: {err}");
            return;
        }
    };
    let grids = match density.to_bohr_grids() {
        Ok(grids) => grids,
        Err(err) => {
            eprintln!("skipping density.inp benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_density_inp_bwords", |b| {
        b.iter(|| {
            black_box(DensityInput::parse_str(
                "density.inp",
                black_box(DENSITY_INPUT_BENCH),
            ))
        });
    });
    c.bench_function("render_density_inp_text", |b| {
        b.iter(|| black_box(density_input_string(black_box(&density))));
    });
    c.bench_function("convert_density_inp_bohr_grids", |b| {
        b.iter(|| black_box(density.to_bohr_grids()));
    });
    let Some(line_grid) = grids.first() else {
        return;
    };
    c.bench_function("evaluate_density_grid_output_text_from_density_inp", |b| {
        b.iter(|| {
            black_box(rhorrp_density_output_from_grid(
                RhorrpDensityGridOutputInput {
                    grid: black_box(line_grid),
                    nearest: None,
                },
                |point| Ok(0.5 * (-0.1 * point[0]).exp()),
            ))
        });
    });
    let nearest_atom_positions = Array2::from_shape_fn((64, 3), |(atom, axis)| {
        let atom = atom as f64;
        match axis {
            0 => 0.25 * atom,
            1 => (0.13 * atom).sin(),
            _ => (0.07 * atom).cos(),
        }
    });
    let nearest_atom_potentials = (0..64).map(|atom| atom % 8).collect::<Vec<_>>();
    c.bench_function(
        "evaluate_density_grid_output_text_nearest_from_density_inp",
        |b| {
            b.iter(|| {
                black_box(rhorrp_density_output_from_grid_with_nearest(
                    RhorrpDensityGridNearestOutputInput {
                        grid: black_box(line_grid),
                        atom_positions_bohr: nearest_atom_positions.view(),
                        atom_potentials: &nearest_atom_potentials,
                        fms_atom_count: None,
                    },
                    |point| Ok(0.5 * (-0.1 * point[0]).exp()),
                ))
            });
        },
    );
    let Some(volume_grid) = grids.get(2) else {
        return;
    };
    c.bench_function("evaluate_density_grid_output_bin_from_density_inp", |b| {
        b.iter(|| {
            black_box(rhorrp_density_output_from_grid(
                RhorrpDensityGridOutputInput {
                    grid: black_box(volume_grid),
                    nearest: None,
                },
                |point| {
                    let radius_squared = point.iter().map(|value| value * value).sum::<f64>();
                    Ok((-0.05 * radius_squared).exp())
                },
            ))
        });
    });
}
