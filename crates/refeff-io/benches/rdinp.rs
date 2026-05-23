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

#[path = "rdinp/fixtures/mod.rs"]
mod fixtures;
use fixtures::*;

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
