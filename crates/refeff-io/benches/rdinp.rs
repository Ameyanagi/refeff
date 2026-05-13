use std::path::Path;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::{Complex32, Complex64};
use refeff_io::phase_bin::{PHASE_BIN_DEFAULT_PAD_WIDTH, PHASE_BIN_DEFAULT_TRANSITION_COUNT};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    AtomsDat, BandInput, ComptonInput, ConfigInput, ConfigOccupation, ConfigRecord, ConfigState,
    CrpaInput, DensityInput, DimensionsDat, DmdwInput, DymCoordinates, DymData, EelsInput,
    Ff2xInput, FmsInput, FullSpectrumInput, GenfmtInput, GeomDat, GlobalInput, GridInput, GridKind,
    GridMinimum, GridPoint, GridRecord, GridRegularRecord, GridUserRecord, HubbardInput, LdosInput,
    OpconsInput, PathsInput, PotInput, RixsInput, ScreenInput, SfconvInput, SpringAngle,
    SpringInput, SpringStretch, SpringVdos, XsphInput,
};
use refeff_io::{
    ChiDatData, ComptonDatData, CrpaDatData, DanesDatData, EELS_TENSOR_LABELS, EelsDatData,
    FMS_BIN_DEFAULT_PAD_WIDTH, FeffBinData, FeffBinPath, FeffBinPotential, FeffDocument, FeffInput,
    FefflBinData, FmsBinData, FmslBinData, GtrBinData, JzzpDatData, LdosDatData, LdosElectronCount,
    ListDatData, ListDatEntry, LogDatData, LossDatData, MpseDatData, MtdpData, PathsDatAtom,
    PathsDatData, PathsDatPath, PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData,
    PotBinScalars, PotentialDatSetInput, RhorrpDensityBinBohrInput, RhorrpDensityBinData,
    RhorrpDensityGridNearestOutputInput, RhorrpDensityGridOutputInput,
    RhorrpDensityOutputBohrInput, RhorrpDensityTextBohrInput, RhorrpDensityTextData,
    RhorrpGgDiagBinData, RhorrpGgSliceBinData, RhorrpNearestAtomColumns, RhozzpDatData,
    RixsLineData, RixsMapData, RunStderrData, RunStdoutData, XmuDatData, XmulDatData, XseclBinData,
    XseclBinTransition, XseclDatData, XseclDatHeader, XsectDatData, XsectDatScalars,
    atoms_dat_string, band_input_string, chemical_dat_string, chi_dat_string, compton_dat_string,
    compton_input_string, config_inp_string, crpa_dat_string, crpa_input_string, danes_dat_string,
    density_input_string, dimensions_dat_string, dmdw_input_string, dmdw_out_string, dym_string,
    edges_dat_string, eels_dat_string, eels_input_string, emesh_dat_string, feff_bin_string,
    feffl_bin_string, ff2x_input_string, fms_bin_string, fms_input_string, fmsl_bin_string,
    fpf0_dat_string, fullspectrum_input_string, genfmt_input_string, geom_dat_string,
    global_input_string, grid_inp_string, gtr_bin_bytes, gtr_dat_string, gtrl_dat_string,
    hubbard_input_string, jzzp_dat_string, ldos_dat_string, ldos_input_string, list_dat_string,
    log_dat_string, loss_dat_string, module_log_dat_string, mpse_dat_string, mtdp_string,
    opcons_input_string, parse_chemical_dat, parse_chi_dat, parse_compton_dat, parse_config_inp,
    parse_crpa_dat, parse_danes_dat, parse_dmdw_out, parse_dym, parse_edges_dat, parse_eels_dat,
    parse_emesh_dat, parse_feff_bin, parse_feffl_bin, parse_fms_bin, parse_fmsl_bin,
    parse_fpf0_dat, parse_grid_inp, parse_gtr_bin, parse_gtr_dat, parse_gtrl_dat, parse_jzzp_dat,
    parse_ldos_dat, parse_list_dat, parse_log_dat, parse_loss_dat, parse_module_log_dat,
    parse_mpse_dat, parse_mtdp, parse_paths_dat, parse_phase_bin, parse_pot_bin,
    parse_rhorrp_density_bin, parse_rhorrp_density_text, parse_rhorrp_gg_diag_bin,
    parse_rhorrp_gg_slice_bin, parse_rhozzp_dat, parse_rixs_line, parse_rixs_map, parse_run_stderr,
    parse_run_stdout, parse_spring_inp, parse_xmu_dat, parse_xmul_dat, parse_xscorr_raw_dat,
    parse_xsecl_bin, parse_xsecl_dat, parse_xsect_dat, paths_dat_string, paths_input_string,
    phase_bin_string, pot_bin_string, pot_input_string, potential_dat_outputs, rdinp,
    rhorrp_density_bin_bytes, rhorrp_density_bin_from_bohr, rhorrp_density_filename_is_binary,
    rhorrp_density_output_from_bohr, rhorrp_density_output_from_grid,
    rhorrp_density_output_from_grid_with_nearest, rhorrp_density_text_from_bohr,
    rhorrp_density_text_string, rhorrp_gg_diag_bin_bytes, rhorrp_gg_diag_matrix,
    rhorrp_gg_pair_matrix, rhorrp_gg_slice_bin_bytes, rhorrp_gg_slice_block, rhozzp_dat_string,
    rixs_input_string, rixs_line_string, rixs_map_string, run_stderr_string, run_stdout_string,
    screen_input_string, sfconv_input_string, spring_inp_string, xmu_dat_string, xmul_dat_string,
    xscorr_raw_dat_string, xsecl_bin_string, xsecl_dat_string, xsect_dat_string, xsph_input_string,
};

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

const DENSITY_INPUT_BENCH: &str = r#"
line,line.dat,0.0,1.0,2.0,core
1.0,0.0,0.0,251
plane plane.dat 0.0, 1.0 2.0
1.0,0.0,0.0,101
0.0,1.0,0.0,101
volume volume.bin 0.0 0.0 0.0
1.0,0.0,0.0,41
0.0,1.0,0.0,41
0.0,0.0,1.0,41
"#;

const DMDW_ENABLED_INPUT_BENCH: &str = concat!(
    "   1\n",
    "   6\n",
    "   1    450.000\n",
    "   0\n",
    "feff.dym\n",
    "   1\n",
    "   2   1   0          29.78\n",
);

const DMDW_OUT_BENCH: &str = concat!(
    "# Lanczos recursion order:    6\n",
    "# Temperature:  450.00\n",
    "# Dynamical matrix file: feff.dym\n",
    "\n",
    "--------------------------------------------------------------\n",
    " Path Indices:    1   2\n",
    " PDOS Poles:\n",
    "     Freq. (THz)    Weight\n",
    "        2.860       0.039469598\n",
    "        3.854       0.182890396\n",
    "        4.940       0.220041663\n",
    "        6.026       0.159715119\n",
    "        6.812       0.284980130\n",
    "        7.306       0.112876736\n",
    "\n",
    " PDOS Einstein freq (single pole), associated temp and eff. force constant: \n",
    " Freq (THz)   Temp (K)   Eff. FC (N/m)\n",
    "   5.784       277.60      69.6914\n",
    "\n",
    " pDOS n Moments, associated Einstein freqs, temps and eff. force constants:\n",
    "  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)\n",
    " -2       0.03881       5.07607       243.60      53.6688\n",
    " -1       0.18959       5.27461       253.13      57.9492\n",
    "  0       0.99997     ---------     --------\n",
    "  1       5.63317       5.63317       270.34      66.0957\n",
    "  2      33.45823       5.78431       277.59      69.6899\n",
    "\n",
    " Path Red. Mass (AMU):   31.773000\n",
    " Path Length (Ang), s^2 (1e-3 Ang^2):  2.5323  11.8576\n",
    "--------------------------------------------------------------\n",
);

const EDGES_DAT_BENCH: &str = concat!(
    " # emu, M_kk, gam\n",
    "   330.31915602984373        1.0000000000000000        6.3546470930994858E-002\n",
);

const CHEMICAL_DAT_BENCH: &str =
    "   0.0000000000000000        0.0000000000000000       -7.7292787791436899     \n";

const EMESH_DAT_BENCH: &str = concat!(
    "# edge, bohr, edge*hart      -0.13880      0.52918     -3.77698\n",
    "# ispec, ik0      0     1\n",
    " # ie, em(ie)*hart, xk(ie)\n",
    "    1            -3.77698             0.00000\n",
    "    2            -3.73888             0.10000\n",
    "    3            -3.62458             0.20000\n",
    "    4            -3.43408             0.30000\n",
    "    5            -3.16738             0.40000\n",
);

const FPF0_DAT_BENCH: &str = concat!(
    "  atom Z =           29\n",
    "       -1.46689E-01       -8.39242E-02 total energy part of fprime - 5/3*E_tot/mc**2\n",
    "           5\n",
    "  2.00000    -332.657   1\n",
    "  0.00162     -36.320   3\n",
    "  0.00317     -35.556   4\n",
    "  0.00017      -3.431   6\n",
    "  0.00033      -3.329   7\n",
    "  0.0   29.0000\n",
    "  0.5   28.6430\n",
    "  1.0   27.7057\n",
    "  1.5   26.4437\n",
    "  2.0   25.0396\n",
    "  2.5   23.5793\n",
);

const MODULE_LOG_BENCH: &str = concat!(
    "Calculating SCF potentials ...\n",
    "FEFF-serial using 1 thread.\n",
    "Done with module: potentials.\n",
);

const GTR_DAT_BENCH: &str = concat!(
    "    -0.616104     0.031773     1.624106     1.081113\n",
    "    -0.558474     0.031773     0.550420     1.190721\n",
    "    -0.506332     0.031773     0.087675     0.846187\n",
    "    -0.459680     0.031773    -0.391425     0.869742\n",
);

const GTRL_DAT_BENCH: &str = concat!(
    "    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01\n",
    "    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01\n",
);

const XSCORR_RAW_DAT_BENCH: &str = concat!(
    " Temperature (Hatree) = 0\n",
    " Electronic Temperature (eV) = 0\n",
    " xloss =   0.86458999999999986       eV\n",
    " efermi =   -3.7769771800000003       eV\n",
    " Number of poles = 0\n",
    " Omega(Hart)    Re CCHI     Im CCHI   1-Fermi   Re xmu0    Im xmu0\n",
    "  -0.1388013015E+000  -0.1629950000E-004   0.1152400000E-003   0.5000000000E+000  -0.3259900000E-004   0.2304800000E-003\n",
    "  -0.1374011587E+000  -0.1689833765E-004   0.1185582229E-003   0.5140178752E+000  -0.3287500000E-004   0.2306500000E-003\n",
);

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
    c.bench_function("render_rdinp_log_dat", |b| {
        b.iter(|| black_box(rdinp::rdinp_log_dat_string(black_box(&document))));
    });
}

fn bench_structure_outputs(c: &mut Criterion) {
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

fn bench_energy_outputs(c: &mut Criterion) {
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

fn bench_control_inputs(c: &mut Criterion) {
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
    let opcons_text = rdinp::opcons_inp_string(&document);
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

fn bench_shared_module_inputs(c: &mut Criterion) {
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

fn bench_phase_module_inputs(c: &mut Criterion) {
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

fn bench_potential_module_inputs(c: &mut Criterion) {
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

fn bench_scalar_module_inputs(c: &mut Criterion) {
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

fn bench_path_module_inputs(c: &mut Criterion) {
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

fn bench_dmdw_out(c: &mut Criterion) {
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

fn bench_spectrum_module_inputs(c: &mut Criterion) {
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

fn bench_density_input(c: &mut Criterion) {
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

fn bench_potential_outputs(c: &mut Criterion) {
    let state = PotOutputBenchState::new();
    c.bench_function("render_wpot_potential_dat_outputs", |b| {
        b.iter(|| black_box(potential_dat_outputs(black_box(state.input()))));
    });
}

fn bench_mtdp(c: &mut Criterion) {
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

fn bench_pot_bin(c: &mut Criterion) {
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
}

fn bench_phase_bin(c: &mut Criterion) {
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

fn bench_feff_bin(c: &mut Criterion) {
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

fn bench_list_dat(c: &mut Criterion) {
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

fn bench_log_dat(c: &mut Criterion) {
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

fn bench_run_output(c: &mut Criterion) {
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

fn bench_paths_dat(c: &mut Criterion) {
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

fn bench_dym(c: &mut Criterion) {
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

fn bench_grid_inp(c: &mut Criterion) {
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

fn bench_config_inp(c: &mut Criterion) {
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

fn bench_spring_inp(c: &mut Criterion) {
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

fn bench_xsect_dat(c: &mut Criterion) {
    let data = xsect_dat_bench_data();
    let text = match xsect_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xsect.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xsect_dat_text", |b| {
        b.iter(|| black_box(xsect_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xsect_dat_text", |b| {
        b.iter(|| black_box(parse_xsect_dat(black_box(&text))));
    });
}

fn bench_xmu_dat(c: &mut Criterion) {
    let data = xmu_dat_bench_data();
    let text = match xmu_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xmu.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xmu_dat_text", |b| {
        b.iter(|| black_box(xmu_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xmu_dat_text", |b| {
        b.iter(|| black_box(parse_xmu_dat(black_box(&text))));
    });
}

fn bench_xmul_dat(c: &mut Criterion) {
    let data = xmul_dat_bench_data();
    let text = match xmul_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping xmul.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_xmul_dat_text", |b| {
        b.iter(|| black_box(xmul_dat_string(black_box(&data))));
    });
    c.bench_function("parse_xmul_dat_text", |b| {
        b.iter(|| black_box(parse_xmul_dat(black_box(&text))));
    });
}

fn bench_xscorr_raw_dat(c: &mut Criterion) {
    let data = match parse_xscorr_raw_dat(XSCORR_RAW_DAT_BENCH) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("skipping XSCORR raw.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("parse_xscorr_raw_dat_text", |b| {
        b.iter(|| black_box(parse_xscorr_raw_dat(black_box(XSCORR_RAW_DAT_BENCH))));
    });
    c.bench_function("render_xscorr_raw_dat_text", |b| {
        b.iter(|| black_box(xscorr_raw_dat_string(black_box(&data))));
    });
}

fn bench_chi_dat(c: &mut Criterion) {
    let data = chi_dat_bench_data();
    let text = match chi_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping chi.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_chi_dat_text", |b| {
        b.iter(|| black_box(chi_dat_string(black_box(&data))));
    });
    c.bench_function("parse_chi_dat_text", |b| {
        b.iter(|| black_box(parse_chi_dat(black_box(&text))));
    });
}

fn bench_eels_dat(c: &mut Criterion) {
    let data = eels_dat_bench_data();
    let text = match eels_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping eels.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_eels_dat_text", |b| {
        b.iter(|| black_box(eels_dat_string(black_box(&data))));
    });
    c.bench_function("parse_eels_dat_text", |b| {
        b.iter(|| black_box(parse_eels_dat(black_box(&text))));
    });
}

fn bench_danes_dat(c: &mut Criterion) {
    let data = danes_dat_bench_data();
    let text = match danes_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping danes.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_danes_dat_text", |b| {
        b.iter(|| black_box(danes_dat_string(black_box(&data))));
    });
    c.bench_function("parse_danes_dat_text", |b| {
        b.iter(|| black_box(parse_danes_dat(black_box(&text))));
    });
}

fn bench_ldos_dat(c: &mut Criterion) {
    let data = ldos_dat_bench_data();
    let text = match ldos_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping ldosNN.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_ldos_dat_text", |b| {
        b.iter(|| black_box(ldos_dat_string(black_box(&data))));
    });
    c.bench_function("parse_ldos_dat_text", |b| {
        b.iter(|| black_box(parse_ldos_dat(black_box(&text))));
    });
}

fn bench_compton_dat(c: &mut Criterion) {
    let data = compton_dat_bench_data();
    let text = match compton_dat_string(&data) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping compton.dat benchmarks: {err}");
            return;
        }
    };
    c.bench_function("render_compton_dat_text", |b| {
        b.iter(|| black_box(compton_dat_string(black_box(&data))));
    });
    c.bench_function("parse_compton_dat_text", |b| {
        b.iter(|| black_box(parse_compton_dat(black_box(&text))));
    });
}

fn bench_rhozzp_dat(c: &mut Criterion) {
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

fn bench_rhorrp_density_text(c: &mut Criterion) {
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

fn bench_rhorrp_density_bin(c: &mut Criterion) {
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

fn bench_rhorrp_gg_bin(c: &mut Criterion) {
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

fn bench_jzzp_dat(c: &mut Criterion) {
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

fn bench_crpa_dat(c: &mut Criterion) {
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

fn bench_loss_dat(c: &mut Criterion) {
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

fn bench_mpse_dat(c: &mut Criterion) {
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

fn bench_rixs_map(c: &mut Criterion) {
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

fn bench_rixs_line(c: &mut Criterion) {
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

fn bench_fms_bin(c: &mut Criterion) {
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

fn bench_gtr_dat(c: &mut Criterion) {
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

fn bench_gtr_bin(c: &mut Criterion) {
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

fn bench_fmsl_bin(c: &mut Criterion) {
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

fn bench_xsecl_dat(c: &mut Criterion) {
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

fn bench_xsecl_bin(c: &mut Criterion) {
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

fn bench_feffl_bin(c: &mut Criterion) {
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

fn bench_input() -> String {
    let local_cu =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples/EXAFS/Cu/feff.inp");
    std::fs::read_to_string(local_cu).unwrap_or_else(|_| FALLBACK_INPUT.to_string())
}

struct PotOutputBenchState {
    muffin_tin_indices: Vec<usize>,
    norman_indices: Vec<usize>,
    titles: Vec<String>,
    electron_density: Array2<f64>,
    free_density: Array2<f64>,
    overlapped_coulomb: Array2<f64>,
    free_coulomb: Array2<f64>,
    total_potential: Array2<f64>,
}

impl PotOutputBenchState {
    fn new() -> Self {
        let rows = 251;
        let potentials = 6;
        Self {
            muffin_tin_indices: (0..potentials).map(|potential| 12 + potential).collect(),
            norman_indices: (0..potentials)
                .map(|potential| 40 + 2 * potential)
                .collect(),
            titles: vec![
                "Cu crystal".to_string(),
                "Gam_ch=1.000E+00 H-L exch Vi=0.000E+00 Vr=0.000E+00".to_string(),
            ],
            electron_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.035 * (row + 1) as f64 + 0.125 * potential as f64
            }),
            free_density: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                0.015 * (row + 1) as f64 + 0.25 * potential as f64
            }),
            overlapped_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -1.2 * (potential + 1) as f64 - 0.02 * (row + 1) as f64
            }),
            free_coulomb: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
            }),
            total_potential: Array2::from_shape_fn((rows, potentials), |(row, potential)| {
                -0.45 * (potential + 1) as f64 + 0.003 * (row + 1) as f64
            }),
        }
    }

    fn input(&self) -> PotentialDatSetInput<'_> {
        PotentialDatSetInput {
            highest_potential_index: self.muffin_tin_indices.len() - 1,
            muffin_tin_indices: &self.muffin_tin_indices,
            norman_indices: &self.norman_indices,
            titles: &self.titles,
            electron_density: self.electron_density.view(),
            free_density: self.free_density.view(),
            overlapped_coulomb: self.overlapped_coulomb.view(),
            free_coulomb: self.free_coulomb.view(),
            total_potential: self.total_potential.view(),
        }
    }
}

fn mtdp_bench_data() -> MtdpData {
    let radial_count = 251;
    let atom_count = 12;
    let empty_count = 4;
    MtdpData {
        radial_count,
        atomic_numbers: Array1::from_shape_fn(
            atom_count,
            |atom| if atom % 3 == 0 { 29 } else { 8 },
        ),
        atom_coordinates: Array2::from_shape_fn((atom_count, 3), |(atom, axis)| {
            atom as f64 * 0.25 + axis as f64 * 0.125
        }),
        atom_radii: Array1::from_shape_fn(atom_count, |atom| 0.4 + atom as f64 * 0.01),
        atom_radius_indices: Array1::from_shape_fn(atom_count, |atom| 40 + atom),
        atom_density: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            0.001 * (radial + 1) as f64 + 0.0001 * atom as f64
        }),
        atom_potential: Array2::from_shape_fn((radial_count, atom_count), |(radial, atom)| {
            -1.0 - 0.01 * radial as f64 - 0.05 * atom as f64
        }),
        empty_sphere_coordinates: Array2::from_shape_fn((empty_count, 3), |(sphere, axis)| {
            sphere as f64 * 0.5 + axis as f64 * 0.2
        }),
        empty_sphere_radii: Array1::from_shape_fn(empty_count, |sphere| 0.2 + sphere as f64 * 0.02),
        empty_sphere_radius_indices: Array1::from_shape_fn(empty_count, |sphere| 25 + sphere),
        empty_sphere_density: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| 0.0005 * (radial + 1) as f64 + 0.0002 * sphere as f64,
        ),
        empty_sphere_potential: Array2::from_shape_fn(
            (radial_count, empty_count),
            |(radial, sphere)| -0.5 - 0.006 * radial as f64 - 0.025 * sphere as f64,
        ),
        interstitial_potential: -0.75,
        homo_energy: -0.12,
        lumo_energy: 0.34,
    }
}

fn pot_bin_bench_data() -> PotBinData {
    let potentials = 6;
    let angular_count = 5;
    PotBinData {
        titles: vec![
            "Cu crystal".to_string(),
            "Gam_ch=1.000E+00 H-L exch Vi=0.000E+00 Vr=0.000E+00".to_string(),
        ],
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 1,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            interstitial_potential: -1.2,
            interstitial_density: 0.03,
            edge_position: 9.1,
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            core_valence_energy: -3.0,
            density_radius: 1.7,
            fermi_momentum: 0.9,
            total_charge: 42.0,
            total_volume: 11.0,
        },
        muffin_tin_indices: Array1::from_shape_fn(potentials, |potential| 12 + potential),
        muffin_tin_radii: Array1::from_shape_fn(potentials, |potential| {
            1.1 + potential as f64 * 0.02
        }),
        norman_indices: Array1::from_shape_fn(potentials, |potential| 30 + potential),
        atomic_numbers: Array1::from_shape_fn(
            potentials,
            |potential| {
                if potential % 2 == 0 { 29 } else { 8 }
            },
        ),
        kappa: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| orbital as i32 - 20),
        norman_radii: Array1::from_shape_fn(potentials, |potential| 2.0 + potential as f64 * 0.03),
        overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            0.85 + potential as f64 * 0.01
        }),
        max_overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            1.15 + potential as f64 * 0.01
        }),
        potential_multiplicities: Array1::from_shape_fn(potentials, |potential| {
            1.0 + potential as f64
        }),
        ionization: Array1::from_shape_fn(potentials, |potential| potential as f64 * 0.25),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            0.001 * (row + 1) as f64
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            -0.001 * (row + 1) as f64
        }),
        large_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        large_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        electron_density: pot_bin_radial_matrix(potentials, 0.01),
        coulomb_potential: pot_bin_radial_matrix(potentials, -0.02),
        total_potential: pot_bin_radial_matrix(potentials, -0.03),
        valence_density: pot_bin_radial_matrix(potentials, 0.004),
        valence_potential: pot_bin_radial_matrix(potentials, -0.005),
        magnetization_density: pot_bin_radial_matrix(potentials, 0.0002),
        orbital_occupancy: Array2::from_shape_fn(
            (POT_BIN_ORBITALS, potentials),
            |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
        ),
        orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
            -10.0 + orbital as f64 * 0.25
        }),
        occupied_orbital_indices: Array2::from_shape_fn(
            (POT_BIN_IORB_SLOTS, potentials),
            |(slot, _)| slot as i32 - 5,
        ),
        norman_charges: Array1::from_shape_fn(potentials, |potential| 8.0 + potential as f64 * 0.5),
        valence_occupancy: Array2::from_shape_fn(
            (angular_count, potentials),
            |(angular, potential)| 0.5 * angular as f64 + potential as f64,
        ),
    }
}

fn pot_bin_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

fn phase_bin_bench_data() -> PhaseBinData {
    let spin_count = 2;
    let energy_count = 64;
    let potentials = 6;
    let q_count = 1;
    let transition_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 48,
        auxiliary_energy_count: 8,
        ihole: 1,
        fermi_index: 24,
        pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: transition_count,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.1 * energy as f64, 0.01 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
            Complex64::new(-1.0 + energy as f64 * 0.05, 0.02 * spin as f64)
        }),
        potentials: (0..potentials)
            .map(|potential| {
                phase_bin_bench_potential(
                    3,
                    if potential % 2 == 0 { 29 } else { 8 },
                    if potential % 2 == 0 { "Cu" } else { "O" },
                    energy_count,
                    spin_count,
                    potential as f64 * 0.01,
                )
            })
            .collect(),
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.1 * q_index as f64 + 0.01 * transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

fn phase_bin_bench_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    offset: f64,
) -> PhaseBinPotential {
    let l_count = 2 * lmax + 1;
    PhaseBinPotential {
        lmax,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::from_shape_fn(
            (energy_count, l_count, spin_count),
            |(energy, l_slot, spin)| {
                Complex64::new(
                    offset + 0.0005 * energy as f64 + 0.01 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
    }
}

fn feff_bin_bench_data() -> FeffBinData {
    let energy_count = 64;
    let path_count = 24;
    FeffBinData {
        version: "refeff-bench".to_string(),
        pad_width: 8,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 9.1,
        potentials: vec![
            FeffBinPotential {
                label: "Cu".to_string(),
                atomic_number: 29,
            },
            FeffBinPotential {
                label: "O".to_string(),
                atomic_number: 8,
            },
        ],
        central_phase_shift: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.01 * energy as f64, -0.001 * energy as f64)
        }),
        complex_momentum: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + 0.02 * energy as f64, 0.01 * energy as f64)
        }),
        real_momentum: Array1::from_shape_fn(energy_count, |energy| 0.1 + 0.02 * energy as f64),
        paths: (0..path_count)
            .map(|path| feff_bin_bench_path(path, energy_count))
            .collect(),
    }
}

fn feff_bin_bench_path(path: usize, energy_count: usize) -> FeffBinPath {
    let leg_count = 3 + path % 4;
    FeffBinPath {
        index: path + 1,
        degeneracy: 2.0 + path as f64 * 0.25,
        effective_half_path_length_bohr: 3.0 + path as f64 * 0.05,
        criterion: 100.0 / (path + 1) as f64,
        potential_indices: Array1::from_shape_fn(leg_count, |leg| leg % 2),
        positions: Array2::from_shape_fn((leg_count, 3), |(leg, axis)| {
            leg as f64 * 0.4 + axis as f64 * 0.125 + path as f64 * 0.01
        }),
        beta: Array1::from_shape_fn(leg_count, |leg| 0.1 * leg as f64),
        eta: Array1::from_shape_fn(leg_count, |leg| 0.2 * leg as f64),
        leg_distances: Array1::from_shape_fn(leg_count, |leg| 1.0 + 0.05 * leg as f64),
        amplitude: Array1::from_shape_fn(energy_count, |energy| {
            0.001 * (energy + 1) as f64 + path as f64 * 0.0001
        }),
        phase: Array1::from_shape_fn(energy_count, |energy| -0.01 * energy as f64),
    }
}

fn list_dat_bench_data() -> ListDatData {
    ListDatData {
        titles: vec![
            "PATH  Rmax= 6.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 2.50%".to_string(),
        ],
        entries: (0..256)
            .map(|path| ListDatEntry {
                path_index: path + 1,
                sigma2: 0.0,
                amplitude_ratio: 100.0 / (path + 1) as f64,
                degeneracy: 2.0 + (path % 8) as f64,
                leg_count: 2 + path % 6,
                effective_half_path_length_angstrom: 1.5 + path as f64 * 0.015,
            })
            .collect(),
    }
}

fn log_dat_bench_data() -> LogDatData {
    LogDatData {
        version: "FEFF 10.0.0".to_string(),
        preamble_lines: vec![
            "Resetting lmaxsc to 2 for iph =    0.  Use  UNFREEZE to prevent this.".to_string(),
            "Resetting lmaxsc to 2 for iph =    1.  Use  UNFREEZE to prevent this.".to_string(),
        ],
        core_hole_lifetime_ev: Some(1.729),
        post_core_lines: Vec::new(),
        titles: vec![" Cu crystal".to_string()],
        calculation_summary: Some("Cu K edge XANES using RPA corehole.".to_string()),
        features: vec![
            "Debye-Waller factors".to_string(),
            "Many-Pole Self-Energy".to_string(),
            "Self-Consistent Field potentials".to_string(),
        ],
        cards: [
            "ATOMS",
            "CONTROL",
            "EXCHANGE",
            "TITLE",
            "DEBYE",
            "POTENTIALS",
            "XANES",
            "CORRECTIONS",
            "SCF",
            "FMS",
            "MPSE",
            "SFCONV",
            "COREHOLE",
            "OPCONS",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        trailing_lines: Vec::new(),
    }
}

fn run_stdout_bench_data() -> RunStdoutData {
    let mut lines = Vec::new();
    for cycle in 0..128 {
        lines.push(format!("Calculating synthetic module {cycle} ..."));
        lines.push("FEFF-serial using 1 thread.".to_string());
        lines.push(format!("Done with module: synthetic module {cycle}."));
    }
    let text = lines.join("\n");
    match parse_run_stdout(&text) {
        Ok(data) => data,
        Err(_) => RunStdoutData {
            lines,
            line_endings: Vec::new(),
            module_events: Vec::new(),
        },
    }
}

fn run_stderr_bench_data() -> RunStderrData {
    let lines = (0..128)
        .map(|index| {
            if index % 7 == 0 {
                "Note: The following floating-point exceptions are signalling: IEEE_INVALID_FLAG"
                    .to_string()
            } else {
                "Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG"
                    .to_string()
            }
        })
        .collect::<Vec<_>>();
    let text = lines.join("\n");
    match parse_run_stderr(&text) {
        Ok(data) => data,
        Err(_) => RunStderrData {
            lines,
            line_endings: Vec::new(),
            floating_point_notes: Vec::new(),
        },
    }
}

fn paths_dat_bench_data() -> PathsDatData {
    let paths = (0..256)
        .map(|path| PathsDatPath {
            index: path + 1,
            degeneracy: 4.0 + (path % 8) as f64,
            effective_half_path_length_angstrom: 2.0 + 0.01 * path as f64,
            row_header:
                "      x           y           z     ipot  label      rleg      beta        eta"
                    .to_string(),
            atoms: vec![
                PathsDatAtom {
                    position_angstrom: [1.0 + 0.01 * path as f64, 0.5, 0.0],
                    potential_index: 1,
                    label: "Cu1".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(45.0),
                },
                PathsDatAtom {
                    position_angstrom: [-1.0 - 0.01 * path as f64, 0.5, 0.0],
                    potential_index: 1,
                    label: "Cu1".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(135.0),
                },
                PathsDatAtom {
                    position_angstrom: [0.0, 0.0, 0.0],
                    potential_index: 0,
                    label: "Cu0".to_string(),
                    leg_distance_angstrom: Some(2.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(225.0),
                },
            ],
        })
        .collect();
    PathsDatData {
        titles: vec!["TITLE Cu crystal".to_string()],
        paths,
    }
}

fn dym_bench_data() -> DymData {
    let atom_count = 32_usize;
    let atomic_numbers =
        Array1::from_iter((0..atom_count).map(|index| if index % 2 == 0 { 29 } else { 8 }));
    let atomic_masses = Array1::from_iter(
        (0..atom_count).map(|index| if index % 2 == 0 { 63.546 } else { 15.999 }),
    );
    let positions = Array2::from_shape_fn((atom_count, 3), |(atom, axis)| match axis {
        0 => atom as f64 * 0.25,
        1 => (atom % 7) as f64 * 0.1,
        _ => (atom % 5) as f64 * 0.05,
    });
    let mut force_constants = Array4::zeros((atom_count, atom_count, 3, 3));
    for i_atom in 0..atom_count {
        for j_atom in 0..atom_count {
            let diagonal = if i_atom == j_atom { 0.2 } else { -0.002 };
            for row in 0..3 {
                for column in 0..3 {
                    force_constants[[i_atom, j_atom, row, column]] = if row == column {
                        diagonal
                    } else {
                        0.0001 * (i_atom as f64 - j_atom as f64)
                    };
                }
            }
        }
    }

    DymData {
        dym_type: 1,
        atomic_numbers,
        atomic_masses,
        coordinates: DymCoordinates::Cartesian(positions),
        force_constants,
    }
}

fn grid_inp_bench_data() -> GridInput {
    let mut records = (0..8)
        .map(|index| {
            GridRecord::Regular(GridRegularRecord {
                kind: if index % 2 == 0 {
                    GridKind::Energy
                } else {
                    GridKind::WaveNumber
                },
                minimum: if index == 0 {
                    GridMinimum::Value(-15.0)
                } else {
                    GridMinimum::Last
                },
                maximum: 5.0 + index as f64,
                step: 0.05 + 0.01 * index as f64,
            })
        })
        .collect::<Vec<_>>();
    records.push(GridRecord::User(GridUserRecord {
        points: (0..64)
            .map(|index| GridPoint {
                real: -2.0 + 0.1 * index as f64,
                imaginary: if index % 3 == 0 { 0.05 } else { 0.0 },
            })
            .collect(),
    }));
    GridInput { records }
}

fn config_inp_bench_data() -> ConfigInput {
    ConfigInput {
        records: (0..16)
            .map(|index| ConfigRecord {
                potential_index: index,
                element: if index % 2 == 0 {
                    "Cu".to_string()
                } else {
                    "Ge".to_string()
                },
                noble_gas: (index % 3 == 0).then(|| "Ar".to_string()),
                states: vec![
                    ConfigState {
                        orbital: "3d".to_string(),
                        occupations: vec![
                            ConfigOccupation {
                                occupation: 4.0 + (index % 3) as f64,
                                spin: None,
                            },
                            ConfigOccupation {
                                occupation: 6.0,
                                spin: None,
                            },
                        ],
                    },
                    ConfigState {
                        orbital: "4s".to_string(),
                        occupations: vec![ConfigOccupation {
                            occupation: 1.0,
                            spin: Some((index % 2) as f64),
                        }],
                    },
                    ConfigState {
                        orbital: "4p".to_string(),
                        occupations: vec![
                            ConfigOccupation {
                                occupation: 0.0,
                                spin: Some(1.0),
                            },
                            ConfigOccupation {
                                occupation: 0.0,
                                spin: Some(0.0),
                            },
                        ],
                    },
                ],
            })
            .collect(),
    }
}

fn spring_inp_bench_data() -> SpringInput {
    SpringInput {
        vdos: Some(SpringVdos {
            resolution: 0.02,
            wmax: 20.0,
            dosfit: 0.1,
            acut: 3.0,
        }),
        print_projected: Some(8),
        stretches: (0..64)
            .map(|index| SpringStretch {
                first_atom: index,
                second_atom: index + 1,
                force_constant: 25.0 + index as f64,
                distance_tolerance_percent: 2.0 + (index % 4) as f64,
            })
            .collect(),
        angles: (0..64)
            .map(|index| SpringAngle {
                first_atom: index,
                center_atom: index + 1,
                third_atom: index + 2,
                force_constant: 40.0 + 3.0 * index as f64,
                angle_tolerance_percent: 5.0 + (index % 5) as f64,
            })
            .collect(),
    }
}

fn xsect_dat_bench_data() -> XsectDatData {
    let energy_count = 256;
    XsectDatData {
        titles: vec!["Cu crystal".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            edge_energy: 9.1,
            chemical_potential: -0.4,
        },
        core_hole_width_ev: 1.23,
        main_energy_count: 192,
        fermi_index: 24,
        energy_grid_ev: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.25 * energy as f64, 0.01 * energy as f64)
        }),
        normalized_background: Array1::from_shape_fn(energy_count, |energy| {
            1.0 + 0.002 * energy as f64
        }),
        cross_section: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + 0.001 * energy as f64, -0.1 - 0.0005 * energy as f64)
        }),
    }
}

fn xmu_dat_bench_data() -> XmuDatData {
    let point_count = 512;
    XmuDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0.0"
                .to_string(),
            "#     0/   0 paths used".to_string(),
            "#  xsedge+ 50, used to normalize mu           1.2667E-04".to_string(),
            "#  -----------------------------------------------------------------------"
                .to_string(),
            "#  omega    e    k    mu    mu0     chi     @#".to_string(),
        ],
        normalization: Some(1.2667e-4),
        photon_energy_ev: Array1::from_shape_fn(point_count, |index| 8979.0 + 0.5 * index as f64),
        relative_energy_ev: Array1::from_shape_fn(point_count, |index| -40.0 + 0.5 * index as f64),
        wave_number: Array1::from_shape_fn(point_count, |index| -3.0 + 0.02 * index as f64),
        mu: Array1::from_shape_fn(point_count, |index| 0.01 + 0.0001 * index as f64),
        mu0: Array1::from_shape_fn(point_count, |index| 0.009 + 0.00008 * index as f64),
        chi: Array1::from_shape_fn(point_count, |index| 0.001 + 0.00002 * index as f64),
    }
}

fn xmul_dat_bench_data() -> XmulDatData {
    let point_count = 512;
    let max_decomposition_channel = 2;
    let channel_count = max_decomposition_channel + 1;
    XmulDatData {
        header_lines: vec![
            "#  Decomposition of S(q,w) for a single electron".to_string(),
            "#  omega    k   S^0(qw)  S_{l=0,...,ldecmx}^0(qw)       chi^q_{l=0,..ldecmx,l^*=0,...,ldecmx}".to_string(),
            "# and ldecmx=     2".to_string(),
        ],
        max_decomposition_channel,
        photon_energy_ev: Array1::from_shape_fn(point_count, |index| {
            11_100.0 + 0.571 * index as f64
        }),
        wave_number: Array1::from_shape_fn(point_count, |index| -1.3 + 0.05 * index as f64),
        total_single_electron: Array1::from_shape_fn(point_count, |index| {
            2.0e-6 * (1.0 + 0.01 * index as f64)
        }),
        channel_background: Array2::from_shape_fn((point_count, channel_count), |(row, channel)| {
            1.0e-7 * (channel + 1) as f64 * (1.0 + 0.005 * row as f64)
        }),
        normalized_fine_structure: Array3::from_shape_fn(
            (point_count, channel_count, channel_count),
            |(row, l_star, channel)| {
                0.05 * (channel + 1) as f64 / (l_star + 1) as f64
                    * (1.0 + 0.001 * row as f64)
            },
        ),
    }
}

fn chi_dat_bench_data() -> ChiDatData {
    let point_count = 512;
    ChiDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0.0"
                .to_string(),
            "#     0/   0 paths used".to_string(),
            "#  -----------------------------------------------------------------------"
                .to_string(),
            "#       k          chi          mag           phase @#".to_string(),
        ],
        wave_number: Array1::from_shape_fn(point_count, |index| 0.05 * index as f64),
        chi: Array1::from_shape_fn(point_count, |index| {
            (0.04 * index as f64).sin() * (-0.001 * index as f64).exp()
        }),
        magnitude: Array1::from_shape_fn(point_count, |index| {
            0.25 * (-0.0005 * index as f64).exp()
        }),
        phase: Array1::from_shape_fn(point_count, |index| -2.7 + 0.01 * index as f64),
        phase_minus_2kr: None,
        ckp_real: None,
        ckp_imag: None,
    }
}

fn eels_dat_bench_data() -> EelsDatData {
    let point_count = 512;
    EelsDatData {
        header_lines: vec![
            "# Orientation sensitive EELS calculation - beam energy =   300.keV".to_string(),
            "# Units are a_0^2 / eV.  Multiply by 28.00 10^-18  to get cm^-2 / eV.".to_string(),
            format!(
                "#  Energy       total         atomic-bg     fine-struct   {}",
                EELS_TENSOR_LABELS.join("            ")
            ),
        ],
        energy_loss_ev: Array1::from_shape_fn(point_count, |index| 8979.0 + 0.25 * index as f64),
        total: Array1::from_shape_fn(point_count, |index| 1.0e-12 + 1.0e-15 * index as f64),
        atomic_background: Array1::from_shape_fn(point_count, |index| {
            1.2e-12 + 0.8e-15 * index as f64
        }),
        fine_structure: Array1::from_shape_fn(point_count, |index| {
            -0.2e-12 + 0.2e-15 * index as f64
        }),
        tensor: Some(Array2::from_shape_fn(
            (point_count, EELS_TENSOR_LABELS.len()),
            |(row, column)| 1.0e-14 * (column + 1) as f64 + 1.0e-18 * row as f64,
        )),
    }
}

fn danes_dat_bench_data() -> DanesDatData {
    let point_count = 512;
    DanesDatData {
        header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| -100.0 + 0.5 * index as f64),
        matsubara: Array1::from_shape_fn(point_count, |_| 0.0),
        sommerfeld: Array1::from_shape_fn(point_count, |index| 1.0e-4 * index as f64),
        anomalous: Array1::from_shape_fn(point_count, |index| 8.0 + (0.01 * index as f64).sin()),
        tail: Array1::from_shape_fn(point_count, |index| 4.0 + 0.001 * index as f64),
        total: Array1::from_shape_fn(point_count, |index| 4.5 + 0.0015 * index as f64),
        difference: Array1::from_shape_fn(point_count, |index| -5.0 + 0.002 * index as f64),
    }
}

fn ldos_dat_bench_data() -> LdosDatData {
    let point_count = 512;
    LdosDatData {
        header_lines: vec![
            "#  Fermi level (eV): -14.683".to_string(),
            "#  Charge transfer :   0.711".to_string(),
            "#    Electron counts for each orbital momentum:".to_string(),
            "#       0      1.428".to_string(),
            "#       1      1.637".to_string(),
            "#       2     10.223".to_string(),
            "#       3      0.000".to_string(),
            "#  Number of atoms in cluster:   0".to_string(),
            "#  Lorentzian broadening with HWHH     0.0100 eV".to_string(),
            "# -----------------------------------------------------------------------".to_string(),
            "#      e        sDOS           pDOS          dDOS          fDOS    @#".to_string(),
        ],
        fermi_level_ev: Some(-14.683),
        charge_transfer: Some(0.711),
        electron_counts: vec![
            LdosElectronCount {
                angular_momentum: 0,
                count: 1.428,
            },
            LdosElectronCount {
                angular_momentum: 1,
                count: 1.637,
            },
            LdosElectronCount {
                angular_momentum: 2,
                count: 10.223,
            },
            LdosElectronCount {
                angular_momentum: 3,
                count: 0.0,
            },
        ],
        atom_count: Some(0),
        lorentzian_hwhh_ev: Some(0.0100),
        energy_ev: Array1::from_shape_fn(point_count, |index| -30.0 + 0.45 * index as f64),
        density: Array2::from_shape_fn((point_count, 4), |(row, column)| {
            1.0e-4 * (column + 1) as f64 * (1.0 + 0.01 * row as f64)
        }),
    }
}

fn compton_dat_bench_data() -> ComptonDatData {
    let point_count = 1000;
    ComptonDatData {
        header_lines: vec![
            " # Compton profile, J(pq)".to_string(),
            " # ns:            32".to_string(),
            " # nphi:          32".to_string(),
            " # nz:            32".to_string(),
            " # nzp:          120".to_string(),
            " # zpmax:   10.0000000000000".to_string(),
            " # temperature (eV):  0.0000000E+00".to_string(),
            " #----------------------------".to_string(),
            " # pq               J".to_string(),
        ],
        ns: Some(32),
        nphi: Some(32),
        nz: Some(32),
        nzp: Some(120),
        zpmax: Some(10.0),
        temperature_ev: Some(0.0),
        momentum: Array1::from_shape_fn(point_count, |index| 5.0 * index as f64 / 999.0),
        profile: Array1::from_shape_fn(point_count, |index| {
            let momentum = 5.0 * index as f64 / 999.0;
            2.75 * (-0.6 * momentum).exp() + 0.02 * (2.0 * momentum).cos()
        }),
    }
}

fn rhozzp_dat_bench_data() -> RhozzpDatData {
    let point_count = 1000;
    RhozzpDatData {
        header_lines: Vec::new(),
        z_prime: Array1::from_shape_fn(point_count, |index| 0.01 + 10.0 * index as f64 / 999.0),
        density: Array1::from_shape_fn(point_count, |index| {
            let z_prime = 0.01 + 10.0 * index as f64 / 999.0;
            3.7 * (-4.0 * z_prime).exp() - 0.55 * (-0.9 * z_prime).exp()
        }),
    }
}

fn rhorrp_density_bench_data() -> RhorrpDensityTextData {
    let point_count = 10_000;
    RhorrpDensityTextData {
        points_angstrom: Array2::from_shape_fn((point_count, 3), |(point, axis)| {
            let point = point as f64;
            match axis {
                0 => 0.01 * point,
                1 => (0.001 * point).sin(),
                _ => (0.0015 * point).cos(),
            }
        }),
        density_per_angstrom3: Array1::from_shape_fn(point_count, |point| {
            let x = point as f64 / point_count as f64;
            0.25 * (-2.5 * x).exp()
        }),
        nearest: Some(RhorrpNearestAtomColumns {
            displacement_bohr: Array2::from_shape_fn((point_count, 3), |(point, axis)| {
                0.001 * (point % 97) as f64 - 0.02 * axis as f64
            }),
            atom_indices: Array1::from_shape_fn(point_count, |point| point % 64),
            potential_indices: Array1::from_shape_fn(point_count, |point| point % 8),
        }),
    }
}

fn rhorrp_density_bin_bench_data() -> RhorrpDensityBinData {
    let points_per_axis = vec![100, 50, 20];
    let point_count = points_per_axis.iter().product::<usize>();
    RhorrpDensityBinData {
        origin_angstrom: [0.1, -0.2, 0.3],
        axes_angstrom: ndarray::arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]]),
        points_per_axis,
        density_per_angstrom3: Array1::from_shape_fn(point_count, |point| {
            let scaled = point as f64 / point_count as f64;
            0.15 * (-3.0 * scaled).exp() + 0.01 * (13.0 * scaled).sin()
        }),
    }
}

fn jzzp_dat_bench_data() -> JzzpDatData {
    let nz = 64;
    let nzp = 120;
    JzzpDatData {
        ns: 32,
        nphi: 32,
        nz,
        nzp,
        smax: 4.5,
        phimax: std::f64::consts::PI,
        zmax: 4.5,
        zpmax: 10.0,
        values: Array2::from_shape_fn((nz, nzp), |(z, zp)| {
            let z_coord = z as f64 / (nz - 1) as f64;
            let zp_coord = zp as f64 / (nzp - 1) as f64;
            (1.0 + z_coord).exp() * (-2.0 * zp_coord).exp()
        }),
    }
}

fn crpa_dat_bench_data() -> CrpaDatData {
    CrpaDatData {
        header_lines: vec!["U, n, U_Bare".to_string()],
        hubbard_u: 0.197879035252010,
        occupation: 1.0,
        bare_u: 0.694283422651496,
    }
}

fn loss_dat_bench_data() -> LossDatData {
    let point_count = 8192;
    LossDatData {
        header_lines: vec!["# E(eV)    Loss".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            0.01 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        loss: Array1::from_shape_fn(point_count, |index| {
            let energy = 0.01 + 50_000.0 * index as f64 / (point_count - 1) as f64;
            2.0e-6 * (-energy / 25_000.0).exp() + 5.0e-5 / (1.0 + (energy / 25.0).powi(2))
        }),
    }
}

fn mpse_dat_bench_data() -> MpseDatData {
    let point_count = 1024;
    MpseDatData {
        header_lines: vec!["# E-EFermi Re[Sigma] Im[Sigma] Re[Z] Im[Z]".to_string()],
        energy_ev: Array1::from_shape_fn(point_count, |index| 0.05 + 0.15 * index as f64),
        self_energy: Array1::from_shape_fn(point_count, |index| {
            let energy = 0.05 + 0.15 * index as f64;
            Complex64::new(0.02 * energy.sqrt(), -0.01 * (1.0 + energy).ln())
        }),
        renormalization: Some(Array1::from_shape_fn(point_count, |index| {
            let scale = 1.0 + index as f64 / point_count as f64;
            Complex64::new(1.0 - 0.05 / scale, -0.02 / scale)
        })),
        renormalization_magnitude: None,
        renormalization_phase: None,
        inelastic_mean_free_path: None,
    }
}

fn rixs_map_bench_data() -> RixsMapData {
    let block_count = 64;
    let rows_per_block = 64;
    let point_count = block_count * rows_per_block;
    RixsMapData {
        header_lines: Vec::new(),
        block_lengths: vec![rows_per_block; block_count],
        first_energy_ev: Array1::from_shape_fn(point_count, |index| {
            11_540.0 + (index % rows_per_block) as f64
        }),
        second_energy_ev: Array1::from_shape_fn(point_count, |index| {
            -15.0 + (index / rows_per_block) as f64 * 0.5
        }),
        channels: Array2::from_shape_fn((point_count, 4), |(row, channel)| {
            let local = (row % rows_per_block) as f64;
            let block = (row / rows_per_block) as f64;
            1.0e-6 * (channel + 1) as f64 * (1.0 + 0.01 * local) * (1.0 + 0.005 * block)
        }),
    }
}

fn rixs_line_bench_data() -> RixsLineData {
    let point_count = 512;
    RixsLineData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_shape_fn(point_count, |index| 11_540.0 + index as f64),
        channels: Array2::from_shape_fn((point_count, 4), |(row, channel)| {
            1.0e-5 * (channel + 1) as f64 * (1.0 + 0.01 * row as f64).ln()
        }),
    }
}

fn fms_bin_bench_data() -> FmsBinData {
    let energy_count = 256;
    let spectrum_count = 4;
    FmsBinData {
        cluster_radius_angstrom: 6.25,
        energy_count,
        main_energy_count: 192,
        auxiliary_energy_count: 16,
        highest_potential_index: 5,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(spectrum_count),
        spectra: Array2::from_shape_fn((spectrum_count, energy_count), |(spectrum, energy)| {
            Complex64::new(
                0.001 * (energy + 1) as f64 + spectrum as f64 * 0.01,
                -0.0005 * (energy + 1) as f64 - spectrum as f64 * 0.005,
            )
        }),
    }
}

fn gtr_bin_bench_data() -> GtrBinData {
    let energy_count = 256;
    let potential_count = 4;
    let angular_channel_count = 4;
    GtrBinData {
        point_count_declared: energy_count,
        horizontal_count: 192,
        danes_extension_count: 0,
        highest_potential_index: potential_count - 1,
        fms_mode: 2,
        values: Array3::from_shape_fn(
            (energy_count, potential_count, angular_channel_count),
            |(energy, potential, angular)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.01 * potential as f64 + 0.02 * angular as f64,
                    -0.0005 * (energy + 1) as f64
                        - 0.005 * potential as f64
                        - 0.01 * angular as f64,
                )
            },
        ),
    }
}

fn fmsl_bin_bench_data() -> FmslBinData {
    let energy_count = 256;
    let max_decomposition_channel = 4;
    let channel_count = max_decomposition_channel + 1;
    FmslBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        traces: Array3::from_shape_fn(
            (energy_count, channel_count, channel_count),
            |(energy, lg2, lg1)| {
                Complex64::new(
                    0.001 * (energy + 1) as f64 + 0.01 * lg2 as f64 + 0.02 * lg1 as f64,
                    -0.0005 * (energy + 1) as f64 - 0.005 * lg2 as f64 - 0.01 * lg1 as f64,
                )
            },
        ),
    }
}

fn xsecl_dat_bench_data() -> XseclDatData {
    let energy_count = 192;
    let channel_count = 11;
    let channel_cross_sections =
        Array2::from_shape_fn((energy_count, channel_count), |(energy, channel)| {
            let scale = (energy + 1) as f64;
            Complex64::new(
                1.0e-4 * scale / (channel + 1) as f64,
                -8.0e-5 * scale / (channel + 2) as f64,
            )
        });
    let channel_sum = Array1::from_shape_fn(energy_count, |energy| {
        channel_cross_sections.row(energy).iter().copied().sum()
    });
    XseclDatData {
        header: XseclDatHeader {
            real_energy_count: 157,
            fermi_index: 11,
            edge: -0.196_469_493_817_166_7,
            emu: 408.320_206_199_998_44,
            core_hole_width: 8.394_938_649_968_564e-2,
        },
        energy: Array1::from_shape_fn(energy_count, |energy| 408.083_58 + 0.003_5 * energy as f64),
        channel_cross_sections,
        channel_sum,
    }
}

fn xsecl_bin_bench_data() -> XseclBinData {
    let energy_count = 256;
    let final_state_count = 12;
    XseclBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        initial_state_j: 1,
        transitions: (0..8)
            .map(|index| XseclBinTransition {
                final_state_kappa: if index % 2 == 0 {
                    -((index / 2) + 1)
                } else {
                    (index / 2) + 1
                },
                decomposition_channel: index % 4,
                total_angular_momentum_channel: index % 5,
                orbital_angular_momentum: index % 4,
            })
            .collect(),
        atom_cross_sections: Array2::from_shape_fn(
            (energy_count, final_state_count),
            |(energy, final_state)| {
                Complex64::new(
                    0.002 * (energy + 1) as f64 + 0.01 * final_state as f64,
                    -0.001 * (energy + 1) as f64 - 0.005 * final_state as f64,
                )
            },
        ),
        raw_atom_cross_section_pad: None,
    }
}

fn feffl_bin_bench_data() -> FefflBinData {
    let path_count = 64;
    let energy_count = 128;
    let max_decomposition_channel = 2;
    let channel_count = max_decomposition_channel + 1;
    FefflBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        amplitudes: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| {
                0.01 * (path + 1) as f64
                    + 0.001 * lg2 as f64
                    + 0.002 * lg1 as f64
                    + 0.0001 * energy as f64
            },
        ),
        phases: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| {
                -0.005 * (path + 1) as f64
                    - 0.0005 * lg2 as f64
                    - 0.001 * lg1 as f64
                    - 0.00005 * energy as f64
            },
        ),
    }
}

criterion_group!(
    benches,
    bench_parse,
    bench_rdinp_outputs,
    bench_structure_outputs,
    bench_energy_outputs,
    bench_control_inputs,
    bench_shared_module_inputs,
    bench_phase_module_inputs,
    bench_potential_module_inputs,
    bench_scalar_module_inputs,
    bench_path_module_inputs,
    bench_dmdw_out,
    bench_spectrum_module_inputs,
    bench_density_input,
    bench_potential_outputs,
    bench_mtdp,
    bench_pot_bin,
    bench_phase_bin,
    bench_feff_bin,
    bench_list_dat,
    bench_log_dat,
    bench_run_output,
    bench_paths_dat,
    bench_dym,
    bench_grid_inp,
    bench_config_inp,
    bench_spring_inp,
    bench_xsect_dat,
    bench_xmu_dat,
    bench_xmul_dat,
    bench_xscorr_raw_dat,
    bench_chi_dat,
    bench_eels_dat,
    bench_danes_dat,
    bench_ldos_dat,
    bench_compton_dat,
    bench_rhozzp_dat,
    bench_rhorrp_density_text,
    bench_rhorrp_density_bin,
    bench_rhorrp_gg_bin,
    bench_jzzp_dat,
    bench_crpa_dat,
    bench_loss_dat,
    bench_mpse_dat,
    bench_rixs_map,
    bench_rixs_line,
    bench_fms_bin,
    bench_gtr_dat,
    bench_gtr_bin,
    bench_fmsl_bin,
    bench_xsecl_dat,
    bench_xsecl_bin,
    bench_feffl_bin
);
criterion_main!(benches);
