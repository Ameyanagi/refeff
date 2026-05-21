use std::path::Path;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::{Complex32, Complex64};
use refeff_core::{
    FullSpectrumEdgeAssembly, SFCONV_MKSPECTF_GRID_LEN, SFCONV_SO2CONV_MOMENTUM_GRID_LEN,
    SfconvPathAverage, SfconvSo2convXanesPreparation,
};
use refeff_io::phase_bin::{PHASE_BIN_DEFAULT_PAD_WIDTH, PHASE_BIN_DEFAULT_TRANSITION_COUNT};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection, ApotBinType,
    ChiDatData, ComptonDatData, CrpaDatData, DanesDatData, DrudeDatData, EELS_TENSOR_LABELS,
    EelsDatData, EpsDatData, ExcDatData, FMS_BIN_DEFAULT_PAD_WIDTH, FeffBinData, FeffBinPath,
    FeffBinPotential, FeffDocument, FeffInput, FefflBinData, FmsBinData, FmslBinData, GtrBinData,
    HamakerDatData, JzzpDatData, LdosDatData, LdosElectronCount, ListDatData, ListDatEntry,
    LogDatData, LossDatData, MpseDatData, MtdpData, OpconsDatData, OscStrDatData, OscStrRow,
    PathsDatAtom, PathsDatData, PathsDatPath, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
    PotBinData, PotBinScalars, PotentialDatSetInput, RhorrpDensityBinBohrInput,
    RhorrpDensityBinData, RhorrpDensityGridNearestOutputInput, RhorrpDensityGridOutputInput,
    RhorrpDensityOutputBohrInput, RhorrpDensityTextBohrInput, RhorrpDensityTextData,
    RhorrpGgDiagBinData, RhorrpGgSliceBinData, RhorrpNearestAtomColumns, RhozzpDatData,
    RixsLineData, RixsMapData, RunStderrData, RunStdoutData, SfconvSpecfunctData, SumRulesDatData,
    XmuDatData, XmulDatData, XseclBinData, XseclBinTransition, XseclDatData, XseclDatHeader,
    XsectDatData, XsectDatScalars, atoms_dat_string, band_input_string, chemical_dat_string,
    chi_dat_string, compton_dat_string, compton_input_string, config_inp_string, crpa_dat_string,
    crpa_input_string, danes_dat_string, density_input_string, dimensions_dat_string,
    dmdw_input_string, dmdw_out_string, drude_dat_string, dym_string, edges_dat_string,
    eels_dat_string, eels_input_string, emesh_dat_string,
    eps_dat_from_fullspectrum_scattering_factors, eps_dat_string, exc_dat_string,
    expand_cif_cluster, expand_cif_structure, feff_bin_string, feffl_bin_string, ff2x_input_string,
    fms_bin_string, fms_input_string, fmsl_bin_string, fpf0_dat_string,
    fullspectrum_absolute_xmu_from_xmu_dat, fullspectrum_background_segment_from_fprime_xmu_dat,
    fullspectrum_imaginary_fine_structure_segment_from_xmu_dat, fullspectrum_input_string,
    fullspectrum_ldos_from_ldos_dat, fullspectrum_normalized_xmu_from_xmu_dat,
    fullspectrum_potential_state_from_pot_bin,
    fullspectrum_real_fine_structure_segment_from_xmu_dat, genfmt_input_string, geom_dat_string,
    global_input_string, grid_inp_string, gtr_bin_bytes, gtr_dat_string, gtrl_dat_string,
    hamaker_dat_from_fullspectrum_epsilon, hamaker_dat_string, hubbard_input_string,
    jzzp_dat_string, ldos_dat_string, ldos_input_string, list_dat_string, log_dat_string,
    loss_dat_string, module_log_dat_string, mpse_dat_string, mtdp_string,
    opcons_dat_from_fullspectrum_epsilon_minus_one, opcons_dat_string, opcons_input_string,
    osc_str_dat_string, osc_str_row_from_fullspectrum_edge, parse_chemical_dat, parse_chi_dat,
    parse_cif, parse_compton_dat, parse_config_inp, parse_crpa_dat, parse_danes_dat,
    parse_dmdw_out, parse_drude_dat, parse_dym, parse_edges_dat, parse_eels_dat, parse_emesh_dat,
    parse_eps_dat, parse_exc_dat, parse_feff_bin, parse_feffl_bin, parse_fms_bin, parse_fmsl_bin,
    parse_fpf0_dat, parse_fullspectrum_options, parse_grid_inp, parse_gtr_bin, parse_gtr_dat,
    parse_gtrl_dat, parse_hamaker_dat, parse_jzzp_dat, parse_ldos_dat, parse_list_dat,
    parse_log_dat, parse_loss_dat, parse_module_log_dat, parse_mpse_dat, parse_mtdp,
    parse_opcons_dat, parse_osc_str_dat, parse_paths_dat, parse_phase_bin, parse_pot_bin,
    parse_rhorrp_density_bin, parse_rhorrp_density_text, parse_rhorrp_gg_diag_bin,
    parse_rhorrp_gg_slice_bin, parse_rhozzp_dat, parse_rixs_line, parse_rixs_map, parse_run_stderr,
    parse_run_stdout, parse_specfunct_dat, parse_spring_inp, parse_sumrules_dat, parse_xmu_dat,
    parse_xmul_dat, parse_xscorr_raw_dat, parse_xsecl_bin, parse_xsecl_dat, parse_xsect_dat,
    paths_dat_string, paths_input_string, phase_bin_string, pot_bin_string, pot_input_string,
    potential_dat_outputs, potential_dat_outputs_from_bins, rdinp, rhorrp_density_bin_bytes,
    rhorrp_density_bin_from_bohr, rhorrp_density_filename_is_binary,
    rhorrp_density_output_from_bohr, rhorrp_density_output_from_grid,
    rhorrp_density_output_from_grid_with_nearest, rhorrp_density_text_from_bohr,
    rhorrp_density_text_string, rhorrp_gg_diag_bin_bytes, rhorrp_gg_diag_matrix,
    rhorrp_gg_pair_matrix, rhorrp_gg_slice_bin_bytes, rhorrp_gg_slice_block, rhozzp_dat_string,
    rixs_input_string, rixs_line_string, rixs_map_string, run_stderr_string, run_stdout_string,
    screen_input_string, sfconv_input_string, sfconv_rdeps_fallback_exc_dat_string,
    sfconv_rdeps_from_exc_dat, sfconv_so2conv_feff_path_data_from_averages,
    sfconv_so2conv_header_from_text, sfconv_so2conv_material_input_from_header,
    sfconv_so2conv_target_data_from_text, sfconv_so2conv_target_data_string,
    sfconv_so2conv_targets, sfconv_specfunct_exafs_convolution_rows,
    sfconv_specfunct_interpolate_momentum, sfconv_specfunct_xanes_convolution_rows,
    specfunct_dat_bytes, spring_inp_string, sumrules_dat_string, xmu_dat_string, xmul_dat_string,
    xscorr_raw_dat_string, xsecl_bin_string, xsecl_dat_string, xsect_dat_ff2x_handoff,
    xsect_dat_string, xsph_input_string,
};
use refeff_io::{
    AtomsDat, BandInput, ComptonInput, ConfigInput, ConfigOccupation, ConfigRecord, ConfigState,
    CrpaInput, DensityInput, DimensionsDat, DmdwInput, DymCoordinates, DymData, EelsInput,
    Ff2xInput, FmsInput, FullSpectrumInput, GenfmtInput, GeomDat, GlobalInput, GridInput, GridKind,
    GridMinimum, GridPoint, GridRecord, GridRegularRecord, GridUserRecord, HubbardInput, LdosInput,
    OpconsInput, PathsInput, PotInput, RixsInput, ScreenInput, SfconvInput, SfconvSo2convTarget,
    SfconvSo2convTargetData, SfconvSo2convTargetKind, SfconvSpecfunctExafsRowsInput,
    SfconvSpecfunctXanesRowsInput, SpringAngle, SpringInput, SpringStretch, SpringVdos, XsphInput,
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

const FULLSPECTRUM_OPTIONS_BENCH: &str = r#"
CONTROL 1 0 1 0 1 0
EGRID 5.0 120.0 230
DRUDE 1.5E-15 0.025
VALENCE
EELS
DETAIL
COMPONENT Cu2 29 0.0847 EDGES
K CONV
4 DETAIL
M1 BACKGROUND
COMPONENT O1 8 DETAIL
1
L1
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

fn exc_dat_bench_data() -> ExcDatData {
    let count = 128;
    ExcDatData {
        header_lines: vec![
            "#SN#   Section:    1".to_string(),
            "#DT#  Double Double Double Double".to_string(),
        ],
        energy_ev: Array1::from_shape_fn(count, |index| 5.0 + index as f64 * 0.25),
        broadening_ev: Array1::from_shape_fn(count, |index| 0.05 + index as f64 * 0.0005),
        oscillator_strength: Array1::from_shape_fn(count, |index| {
            0.1 + (index as f64 * 0.01).sin().abs()
        }),
        auxiliary_weight: Some(Array1::from_shape_fn(count, |index| {
            0.2 + index as f64 * 0.02
        })),
    }
}

fn so2conv_specfunct_bench_data() -> SfconvSpecfunctData {
    let momentum_count = SFCONV_SO2CONV_MOMENTUM_GRID_LEN;
    let spectral_count = SFCONV_MKSPECTF_GRID_LEN;
    let pole_capacity = 5_000;
    let mut spectral_info = Array2::from_shape_fn((momentum_count, 8), |(row, col)| {
        0.01 * row as f64 + 0.001 * col as f64
    });
    for row in 0..momentum_count {
        spectral_info[[row, 0]] = 0.05 + 0.02 * row as f64;
    }

    SfconvSpecfunctData {
        wigner_seitz_radius: 2.05,
        core_hole_lifetime: 0.03125,
        asymmetric_phase: 1,
        satellite_type: 0,
        low_q_mode: 0,
        pole_count: 8,
        pole_energy: Array1::from_shape_fn(pole_capacity, |index| 0.25 + 0.01 * index as f64),
        pole_broadening: Array1::from_shape_fn(pole_capacity, |index| 0.02 + 0.0001 * index as f64),
        pole_weight: Array1::from_shape_fn(pole_capacity, |index| 1.0 / (1.0 + index as f64)),
        spectral_info,
        weights: Array2::from_shape_fn((momentum_count, 8), |(row, col)| {
            0.1 + 0.001 * row as f64 + 0.01 * col as f64
        }),
        extrinsic_quasiparticle: so2conv_specfunct_table(momentum_count, spectral_count, 0.1),
        extrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.2),
        interference_quasiparticle: so2conv_specfunct_table(momentum_count, spectral_count, 0.3),
        interference_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.4),
        intrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.5),
        clipped_extrinsic_satellite: so2conv_specfunct_table(momentum_count, spectral_count, 0.6),
        energy_grid: Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
            -2.0 + 0.05 * col as f64 + 0.001 * row as f64
        }),
    }
}

fn so2conv_specfunct_table(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        scale + 0.0001 * row as f64 + 0.0002 * col as f64
    })
}

struct So2convExafsBenchData {
    signal_energy: Array1<f64>,
    real_signal: Array1<f64>,
    imaginary_signal: Array1<f64>,
    original_magnitude: Array1<f64>,
    original_phase: Array1<f64>,
    phase_minus_2kr: Array1<f64>,
}

fn so2conv_exafs_bench_data(len: usize) -> So2convExafsBenchData {
    let signal_energy = Array1::from_shape_fn(len, |row| row as f64 * 0.02);
    let real_signal = Array1::from_shape_fn(len, |row| 1.0 + 0.001 * row as f64);
    let imaginary_signal = Array1::from_shape_fn(len, |row| 0.35 + 0.0005 * row as f64);
    let original_magnitude = Array1::from_shape_fn(len, |row| {
        let real = real_signal[row];
        let imaginary = imaginary_signal[row];
        (real * real + imaginary * imaginary).sqrt()
    });
    let original_phase =
        Array1::from_shape_fn(len, |row| imaginary_signal[row].atan2(real_signal[row]));
    let phase_minus_2kr =
        Array1::from_shape_fn(len, |row| original_phase[row] - 0.005 * row as f64);

    So2convExafsBenchData {
        signal_energy,
        real_signal,
        imaginary_signal,
        original_magnitude,
        original_phase,
        phase_minus_2kr,
    }
}

fn so2conv_xanes_preparation_bench_data(len: usize) -> SfconvSo2convXanesPreparation {
    let excitation_energy = Array1::from_shape_fn(len, |row| row as f64 * 2.0);
    let absorption = Array1::from_shape_fn(len, |row| 1.0 + 0.001 * row as f64);
    let embedded_background = Array1::from_shape_fn(len, |row| 0.8 + 0.0005 * row as f64);
    let imaginary_fine_structure = &absorption - &embedded_background;

    SfconvSo2convXanesPreparation {
        incident_energy: Array1::from_shape_fn(len, |row| 100.0 + row as f64 * 2.0),
        excitation_energy,
        absorption,
        embedded_background,
        imaginary_fine_structure,
        real_fine_structure: Array1::from_shape_fn(len, |row| 0.1 + 0.0002 * row as f64),
    }
}

fn bench_input() -> String {
    let local_cu =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../feff10/examples/EXAFS/Cu/feff.inp");
    std::fs::read_to_string(local_cu).unwrap_or_else(|_| FALLBACK_INPUT.to_string())
}

fn cif_bench_text() -> String {
    let mut text = String::from(
        r#"
data_primary
_cell_length_a 4.2000
_cell_length_b 4.4000
_cell_length_c 5.1000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_symmetry_space_group_name_H-M 'P 1'
_publ_section_comment
;
"#,
    );
    text.push_str(&"x".repeat(9000));
    text.push_str(
        r#"
;
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
"#,
    );
    for index in 0..128 {
        let x = (index % 8) as f64 / 8.0;
        let y = ((index / 8) % 4) as f64 / 4.0;
        let z = (index / 32) as f64 / 4.0;
        text.push_str(&format!("C{index} {x:.6} {y:.6} {z:.6}\n"));
    }
    text.push_str(
        r#"
data_ignored
_cell_length_a 20.0
_cell_length_b 20.0
_cell_length_c 20.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_atom_site_label _atom_site_fract_x _atom_site_fract_y _atom_site_fract_z
O1 0.5 0.5 0.5
"#,
    );
    text
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

fn apot_bin_wpot_bench_data(pot: &PotBinData) -> ApotBinData {
    let rows = POT_BIN_RADIAL_POINTS;
    let columns = pot.potential_count() + 1;
    ApotBinData {
        sections: vec![
            apot_bin_wpot_matrix_section(
                8,
                "rho(r,0:nphx+1) - atomic density for each unique potential",
                Array2::from_shape_fn((rows, columns), |(row, potential)| {
                    0.015 * (row + 1) as f64 + 0.25 * potential as f64
                }),
            ),
            apot_bin_wpot_matrix_section(
                11,
                "vcoul(r,nph) - coulomb potential for each unique potential.",
                Array2::from_shape_fn((rows, columns), |(row, potential)| {
                    -0.75 * (potential + 1) as f64 - 0.0125 * (row + 1) as f64
                }),
            ),
        ],
    }
}

fn apot_bin_wpot_matrix_section(
    section_number: usize,
    header: &str,
    values: Array2<f64>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: vec![header.to_string()],
        header_texts: vec![format!(" {header}")],
        column_labels: vec![],
        column_label_text: None,
        payload: ApotBinPayload::Matrix(ApotBinMatrix {
            value_type: ApotBinType::Double,
            values: ApotBinMatrixValues::Real(values),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
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
        raw_text: None,
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
        raw_text: None,
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
        type2_metadata: None,
        dipole_derivatives: None,
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

fn opcons_dat_bench_data() -> OpconsDatData {
    let point_count = 4096;
    OpconsDatData {
        header_lines: vec![
            "# Cu K".to_string(),
            "#   omega (eV)      epsilon_1       epsilon_2       n               kappa           mu (cm^(-1))    R               epsinv".to_string(),
        ],
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            10.0 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon_minus_one: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.2 + 0.03 * phase.sin(), 0.1 + 0.02 * phase.cos())
        }),
        refractive_index_minus_one: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.05 + 0.005 * phase.cos(), 0.02 + 0.004 * phase.sin())
        }),
        absorption_coefficient: Array1::from_shape_fn(point_count, |index| {
            1000.0 + 5.0 * index as f64
        }),
        reflectivity: Array1::from_shape_fn(point_count, |index| 0.01 + 0.000001 * index as f64),
        loss: Array1::from_shape_fn(point_count, |index| 0.02 + 0.000002 * index as f64),
    }
}

fn eps_dat_bench_data() -> EpsDatData {
    let point_count = 4096;
    EpsDatData {
        header_lines: vec!["# FULLSPECTRUM eps.dat".to_string()],
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.2 + 0.03 * phase.sin(), 0.1 + 0.02 * phase.cos())
        }),
        background_epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.15 + 0.02 * phase.cos(), 0.08 + 0.015 * phase.sin())
        }),
        sigma: Array1::from_shape_fn(point_count, |index| 0.001 + 0.000001 * index as f64),
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

fn osc_str_dat_bench_data() -> OscStrDatData {
    let edges = ["K", "L1", "L2", "L3"];
    OscStrDatData {
        header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
        rows: (0..256)
            .map(|index| OscStrRow {
                component: if index % 2 == 0 {
                    "Cu".to_string()
                } else {
                    "O".to_string()
                },
                edge: edges[index % edges.len()].to_string(),
                core_hole_index: (index % edges.len() + 1) as i32,
                effective_electron_count: 0.5 + 0.01 * index as f64,
            })
            .collect(),
    }
}

fn fullspectrum_edge_assembly_bench_data(
    effective_electron_count: f64,
) -> FullSpectrumEdgeAssembly {
    FullSpectrumEdgeAssembly {
        scattering_factor: Array1::from_elem(2, Complex64::new(0.0, 0.0)),
        background: Array1::from_elem(2, Complex64::new(0.0, 0.0)),
        effective_electron_count,
        zero_energy_fprime: 0.0,
        overlap_points: 1,
    }
}

fn sumrules_dat_bench_data() -> SumRulesDatData {
    let point_count = 8192;
    SumRulesDatData {
        header_lines: Vec::new(),
        energy_ev: Array1::from_shape_fn(point_count, |index| {
            10.0 + 50_000.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon2_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.0001 * index as f64 + 0.05 * (index as f64 * 0.001).sin().abs()
        }),
        absorption_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.05 * index as f64 + 0.1 * (index as f64 * 0.002).cos().abs()
        }),
        loss_effective_electrons: Array1::from_shape_fn(point_count, |index| {
            0.00008 * index as f64 + 0.02 * (index as f64 * 0.003).sin().abs()
        }),
        absorption_refractive_sum: Array1::from_shape_fn(point_count, |index| {
            0.001 * index as f64 + 0.005 * (index as f64 * 0.004).cos()
        }),
        refractive_index_sum_ratio: Array1::from_shape_fn(point_count, |index| {
            0.8 + 0.2 * (index as f64 * 0.005).sin().abs()
        }),
        log_loss_moment_ratio: Array1::from_shape_fn(point_count, |index| {
            -2.0 + 0.0005 * index as f64
        }),
    }
}

fn drude_dat_bench_data() -> DrudeDatData {
    let point_count = 8192;
    DrudeDatData {
        gamma_ev: 0.658,
        plasma_frequency_ev: 26.417_175_795_207_253,
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        epsilon: Array1::from_shape_fn(point_count, |index| {
            let omega = 0.01 + 10.0 * index as f64 / (point_count - 1) as f64;
            Complex64::new(-1.0 / (1.0 + omega * omega), 0.2 / (omega + 0.1))
        }),
    }
}

fn hamaker_dat_bench_data() -> HamakerDatData {
    let point_count = 8192;
    HamakerDatData {
        header_lines: Vec::new(),
        omega: Array1::from_shape_fn(point_count, |index| {
            0.01 + 10.0 * index as f64 / (point_count - 1) as f64
        }),
        imaginary_axis_epsilon: Array1::from_shape_fn(point_count, |index| {
            let phase = index as f64 * 0.001;
            Complex64::new(0.1 + 0.02 * phase.sin(), 0.0)
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

#[path = "rdinp/binary_outputs.rs"]
mod binary_outputs;
#[path = "rdinp/energy.rs"]
mod energy;
#[path = "rdinp/general_outputs.rs"]
mod general_outputs;
#[path = "rdinp/many_body_outputs.rs"]
mod many_body_outputs;
#[path = "rdinp/module_inputs.rs"]
mod module_inputs;
#[path = "rdinp/parse.rs"]
mod parse;
#[path = "rdinp/rhorrp_outputs.rs"]
mod rhorrp_outputs;
#[path = "rdinp/spectra_outputs.rs"]
mod spectra_outputs;
#[path = "rdinp/structure.rs"]
mod structure;

criterion_group!(
    benches,
    parse::bench_parse,
    parse::bench_rdinp_outputs,
    structure::bench_structure_outputs,
    structure::bench_cif,
    energy::bench_energy_outputs,
    module_inputs::bench_control_inputs,
    module_inputs::bench_shared_module_inputs,
    module_inputs::bench_phase_module_inputs,
    module_inputs::bench_potential_module_inputs,
    module_inputs::bench_scalar_module_inputs,
    module_inputs::bench_path_module_inputs,
    module_inputs::bench_dmdw_out,
    module_inputs::bench_spectrum_module_inputs,
    module_inputs::bench_density_input,
    general_outputs::bench_potential_outputs,
    general_outputs::bench_mtdp,
    general_outputs::bench_list_dat,
    general_outputs::bench_log_dat,
    general_outputs::bench_run_output,
    general_outputs::bench_paths_dat,
    general_outputs::bench_dym,
    general_outputs::bench_grid_inp,
    general_outputs::bench_config_inp,
    general_outputs::bench_spring_inp,
    binary_outputs::bench_pot_bin,
    binary_outputs::bench_phase_bin,
    binary_outputs::bench_feff_bin,
    binary_outputs::bench_fms_bin,
    binary_outputs::bench_gtr_dat,
    binary_outputs::bench_gtr_bin,
    binary_outputs::bench_fmsl_bin,
    binary_outputs::bench_xsecl_dat,
    binary_outputs::bench_xsecl_bin,
    binary_outputs::bench_feffl_bin,
    spectra_outputs::bench_xsect_dat,
    spectra_outputs::bench_xmu_dat,
    spectra_outputs::bench_opcons_dat,
    spectra_outputs::bench_eps_dat,
    spectra_outputs::bench_xmul_dat,
    spectra_outputs::bench_xscorr_raw_dat,
    spectra_outputs::bench_chi_dat,
    spectra_outputs::bench_eels_dat,
    spectra_outputs::bench_danes_dat,
    spectra_outputs::bench_ldos_dat,
    spectra_outputs::bench_compton_dat,
    rhorrp_outputs::bench_rhozzp_dat,
    rhorrp_outputs::bench_rhorrp_density_text,
    rhorrp_outputs::bench_rhorrp_density_bin,
    rhorrp_outputs::bench_rhorrp_gg_bin,
    rhorrp_outputs::bench_jzzp_dat,
    many_body_outputs::bench_crpa_dat,
    many_body_outputs::bench_loss_dat,
    many_body_outputs::bench_osc_str_dat,
    many_body_outputs::bench_sumrules_dat,
    many_body_outputs::bench_drude_dat,
    many_body_outputs::bench_hamaker_dat,
    many_body_outputs::bench_exc_dat,
    many_body_outputs::bench_mpse_dat,
    many_body_outputs::bench_rixs_map,
    many_body_outputs::bench_rixs_line,
);
criterion_main!(benches);
