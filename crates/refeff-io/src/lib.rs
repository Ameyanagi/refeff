#![forbid(unsafe_code)]

//! Input/output compatibility support for the FEFF10 Rust port.
//!
//! This crate owns FEFF text parsing, FEFF-style intermediate writers, and
//! file-format codecs such as Packed ASCII Data (PAD). Numerical modules should
//! depend on these typed structures rather than re-parsing FEFF text ad hoc.

pub mod apot_bin;
pub mod chi_dat;
pub mod cif;
pub mod compton_dat;
pub mod compton_input;
pub mod config_dat;
pub mod config_input;
pub mod control_input;
pub mod crpa_dat;
pub mod crpa_input;
pub mod danes_dat;
pub mod dmdw_input;
pub mod dmdw_out;
pub mod dym;
pub mod eels_dat;
pub mod eels_input;
pub mod emesh_bin;
pub mod energy_output;
pub mod error;
pub mod feff_bin;
pub mod feffl_bin;
pub mod ff2x_input;
pub mod fms_bin;
pub mod fms_input;
pub mod fmsl_bin;
pub mod format;
pub mod fpf0_dat;
pub mod genfmt_input;
pub mod gg_dat;
pub mod global_input;
pub mod grid_input;
pub mod gtr_bin;
pub mod gtr_dat;
pub mod gtrl_dat;
pub mod highz_out;
pub mod hubbard_input;
pub mod input;
pub mod ldos_dat;
pub mod ldos_input;
pub mod list_dat;
pub mod log_dat;
pub mod loss_dat;
pub mod misc_dat;
pub mod model;
pub mod mpse_dat;
pub mod mtdp;
pub mod pad;
pub mod paths_dat;
pub mod paths_input;
pub mod phase_bin;
pub mod pot_bin;
pub mod pot_diagnostics;
pub mod pot_input;
pub mod pot_output;
pub mod rdinp;
pub mod rhorrp_density;
pub mod rhorrp_density_bin;
pub mod rhorrp_density_output;
pub mod rixs_dat;
pub mod rixs_input;
pub mod run_output;
pub mod screen_dat;
pub mod screen_input;
pub mod sfconv_input;
pub mod spring_input;
pub mod structure_output;
pub mod xmu_dat;
pub mod xmul_dat;
pub mod xscorr_dat;
pub mod xsecl_bin;
pub mod xsecl_dat;
pub mod xsect_dat;
pub mod xsph_input;

pub use apot_bin::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinRecords,
    ApotBinSection, ApotBinType, ApotBinValue, apot_bin_string, parse_apot_bin, read_apot_bin,
    write_apot_bin,
};
pub use chi_dat::{ChiDatData, chi_dat_string, parse_chi_dat, read_chi_dat, write_chi_dat};
pub use cif::{
    CifAtomSite, CifCell, CifCluster, CifClusterAtom, CifDocument, CifExpandedStructure,
    CifPotential, expand_cif_cluster, expand_cif_structure, parse_cif, read_cif,
};
pub use compton_dat::{
    ComptonDatData, JzzpDatData, RhozzpDatData, compton_dat_string, jzzp_dat_string,
    parse_compton_dat, parse_jzzp_dat, parse_rhozzp_dat, read_compton_dat, read_jzzp_dat,
    read_rhozzp_dat, rhozzp_dat_string, write_compton_dat, write_jzzp_dat, write_rhozzp_dat,
};
pub use compton_input::{
    ComptonChemicalPotential, ComptonDensityOutputs, ComptonGrid, ComptonInput, ComptonLimits,
    ComptonMomentum, ComptonSwitches, ComptonWindow,
};
pub use config_dat::{
    CONFIG_DAT_ORBITAL_COUNT, ConfigDatData, ConfigDatPotential, config_dat_string,
    parse_config_dat, read_config_dat, write_config_dat,
};
pub use config_input::{
    CONFIG_RECORD_WIDTH, ConfigInput, ConfigOccupation, ConfigRecord, ConfigState,
    config_inp_lines_string, config_inp_string, parse_config_inp, read_config_inp,
    write_config_inp,
};
pub use control_input::{
    BandEnergyMesh, BandInput, DensityAxis, DensityGrid, DensityGridBohr, DensityGridKind,
    DensityInput, FEFF_BOHR_ANGSTROM, FullSpectrumInput, OpconsInput, ReciprocalCell,
    ReciprocalInput, ReciprocalKMesh, reciprocal_input_string,
};
pub use crpa_dat::{CrpaDatData, crpa_dat_string, parse_crpa_dat, read_crpa_dat, write_crpa_dat};
pub use crpa_input::CrpaInput;
pub use danes_dat::{
    DanesDatData, danes_dat_string, parse_danes_dat, read_danes_dat, write_danes_dat,
};
pub use dmdw_input::{DmdwCalculation, DmdwInput, DmdwPath};
pub use dmdw_out::{
    DmdwOutData, DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment, DmdwOutPole, DmdwOutSection,
    DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue, dmdw_out_string, parse_dmdw_out,
    read_dmdw_out, write_dmdw_out,
};
pub use dym::{DymCoordinates, DymData, dym_string, parse_dym, read_dym, write_dym};
pub use eels_dat::{
    EELS_TENSOR_LABELS, EelsDatData, eels_dat_string, parse_eels_dat, read_eels_dat, write_eels_dat,
};
pub use eels_input::{EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh};
pub use emesh_bin::{
    EmeshBinData, emesh_bin_bytes, parse_emesh_bin, read_emesh_bin, write_emesh_bin,
};
pub use energy_output::{
    ChemicalDatData, EdgesDatData, EdgesDatRow, EmeshDatData, chemical_dat_string,
    edges_dat_string, emesh_dat_string, parse_chemical_dat, parse_edges_dat, parse_emesh_dat,
    read_chemical_dat, read_edges_dat, read_emesh_dat, write_chemical_dat, write_edges_dat,
    write_emesh_dat,
};
pub use error::{IoError, Result};
pub use feff_bin::{
    FeffBinData, FeffBinPath, FeffBinPotential, feff_bin_string, parse_feff_bin, read_feff_bin,
    write_feff_bin,
};
pub use feffl_bin::{
    FefflBinData, feffl_bin_string, parse_feffl_bin, read_feffl_bin, write_feffl_bin,
};
pub use ff2x_input::{Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput};
pub use fms_bin::{
    FMS_BIN_DEFAULT_PAD_WIDTH, FmsBinData, fms_bin_string, parse_fms_bin, read_fms_bin,
    write_fms_bin,
};
pub use fms_input::{FmsCluster, FmsControl, FmsDebye, FmsInput};
pub use fmsl_bin::{FmslBinData, fmsl_bin_string, parse_fmsl_bin, read_fmsl_bin, write_fmsl_bin};
pub use fpf0_dat::{
    Fpf0DatData, Fpf0Oscillator, fpf0_dat_string, parse_fpf0_dat, read_fpf0_dat, write_fpf0_dat,
};
pub use genfmt_input::{GenfmtControl, GenfmtInput};
pub use gg_dat::{
    GgDatData, GgDatSection, gg_bin_string, gg_dat_string, parse_gg_bin, parse_gg_dat, read_gg_bin,
    read_gg_dat, write_gg_bin, write_gg_dat,
};
pub use global_input::{
    CfAverage, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GlobalQVector,
};
pub use grid_input::{
    GridInput, GridKind, GridMinimum, GridPoint, GridRecord, GridRegularRecord, GridUserRecord,
    grid_inp_string, parse_grid_inp, read_grid_inp, write_grid_inp,
};
pub use gtr_bin::{GtrBinData, gtr_bin_bytes, parse_gtr_bin, read_gtr_bin, write_gtr_bin};
pub use gtr_dat::{GtrDatData, gtr_dat_string, parse_gtr_dat, read_gtr_dat, write_gtr_dat};
pub use gtrl_dat::{GtrlDatData, gtrl_dat_string, parse_gtrl_dat, read_gtrl_dat, write_gtrl_dat};
pub use highz_out::{HighZOut, HighZOutRow, parse_highz_out, read_highz_out};
pub use hubbard_input::HubbardInput;
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use ldos_dat::{
    LDOS_ORBITAL_LABELS, LDOS_SPIN_ORBITAL_LABELS, LdosDatData, LdosElectronCount, RhocDatData,
    ldos_dat_string, parse_ldos_dat, parse_rhoc_dat, read_ldos_dat, read_rhoc_dat, rhoc_dat_string,
    write_ldos_dat, write_rhoc_dat,
};
pub use ldos_input::{LdosControl, LdosFms, LdosInput, LdosMesh};
pub use list_dat::{
    ListDatData, ListDatEntry, list_dat_string, parse_list_dat, read_list_dat, write_list_dat,
};
pub use log_dat::{
    LogDatData, ModuleLogData, log_dat_string, module_log_dat_string, parse_log_dat,
    parse_module_log_dat, read_log_dat, read_module_log_dat, write_log_dat, write_module_log_dat,
};
pub use loss_dat::{LossDatData, loss_dat_string, parse_loss_dat, read_loss_dat, write_loss_dat};
pub use misc_dat::{MiscDatData, misc_dat_string, parse_misc_dat, read_misc_dat, write_misc_dat};
pub use model::{Atom, FeffDocument, Potential};
pub use mpse_dat::{MpseDatData, mpse_dat_string, parse_mpse_dat, read_mpse_dat, write_mpse_dat};
pub use mtdp::{MtdpData, mtdp_string, parse_mtdp, read_mtdp, write_mtdp};
pub use paths_dat::{
    PathsDatAtom, PathsDatData, PathsDatPath, parse_paths_dat, paths_dat_string, read_paths_dat,
    write_paths_dat,
};
pub use paths_input::{PathsControl, PathsCriteria, PathsInput};
pub use phase_bin::{
    PhaseBinData, PhaseBinPotential, PhaseBinScalars, parse_phase_bin, phase_bin_string,
    read_phase_bin, write_phase_bin,
};
pub use pot_bin::{
    PotBinData, PotBinScalars, parse_pot_bin, pot_bin_string, read_pot_bin, write_pot_bin,
};
pub use pot_diagnostics::{
    Fort16Data, ScfConvergenceData, ScfConvergenceRow, convergence_scf_fine_string,
    convergence_scf_string, fort16_string, parse_convergence_scf, parse_convergence_scf_fine,
    parse_fort16, read_convergence_scf, read_convergence_scf_fine, read_fort16,
    write_convergence_scf, write_convergence_scf_fine, write_fort16,
};
pub use pot_input::{PotControl, PotInput, PotPotential, PotRun, PotScattering};
pub use pot_output::{
    PotentialDatInput, PotentialDatSetInput, pot_dat_string, potential_dat_filename,
    potential_dat_outputs, write_potential_dat,
};
pub use rhorrp_density::{
    RhorrpDensityTextBohrInput, RhorrpDensityTextData, RhorrpNearestAtomColumns,
    parse_rhorrp_density_text, read_rhorrp_density_text, rhorrp_density_text_from_bohr,
    rhorrp_density_text_string, write_rhorrp_density_text,
};
pub use rhorrp_density_bin::{
    RhorrpDensityBinBohrInput, RhorrpDensityBinData, parse_rhorrp_density_bin,
    read_rhorrp_density_bin, rhorrp_density_bin_bytes, rhorrp_density_bin_from_bohr,
    rhorrp_density_filename_is_binary, write_rhorrp_density_bin,
};
pub use rhorrp_density_output::{
    RhorrpDensityGridNearestOutputInput, RhorrpDensityGridOutputInput,
    RhorrpDensityOutputBohrInput, RhorrpDensityOutputData, RhorrpNearestAtomColumnsBohrInput,
    rhorrp_density_output_from_bohr, rhorrp_density_output_from_grid,
    rhorrp_density_output_from_grid_with_nearest, rhorrp_nearest_atom_columns_from_bohr,
    write_rhorrp_density_grid_output, write_rhorrp_density_grid_output_with_nearest,
    write_rhorrp_density_output_from_bohr,
};
pub use rixs_dat::{
    RixsLineData, RixsMapData, parse_rixs_line, parse_rixs_map, read_rixs_line, read_rixs_map,
    rixs_line_string, rixs_map_string, write_rixs_line, write_rixs_map,
};
pub use rixs_input::{RixsBroadening, RixsEnergyWindow, RixsInput, RixsSwitches};
pub use run_output::{
    FloatingPointNote, RunModuleEvent, RunModuleEventKind, RunStderrData, RunStdoutData,
    parse_fort11, parse_run_stderr, parse_run_stdout, read_fort11, read_run_stderr,
    read_run_stdout, run_stderr_string, run_stdout_string, write_run_stderr, write_run_stdout,
};
pub use screen_dat::{
    VtotDatData, WscrnDatData, parse_vtot_dat, parse_wscrn_dat, read_vtot_dat, read_wscrn_dat,
    vtot_dat_string, write_vtot_dat, write_wscrn_dat, wscrn_dat_string,
};
pub use screen_input::ScreenInput;
pub use sfconv_input::{SfconvControl, SfconvInput, SfconvSpectrum, SfconvWindow};
pub use spring_input::{
    SPRING_DEFAULT_ACUT, SPRING_DEFAULT_DOSFIT, SPRING_DEFAULT_RESOLUTION, SPRING_DEFAULT_WMAX,
    SpringAngle, SpringInput, SpringStretch, SpringVdos, parse_spring_inp, read_spring_inp,
    spring_inp_string, write_spring_inp,
};
pub use structure_output::{AtomsDat, AtomsDatRow, DimensionsDat, GeomDat, GeomDatRow};
pub use xmu_dat::{XmuDatData, parse_xmu_dat, read_xmu_dat, write_xmu_dat, xmu_dat_string};
pub use xmul_dat::{XmulDatData, parse_xmul_dat, read_xmul_dat, write_xmul_dat, xmul_dat_string};
pub use xscorr_dat::{
    XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData, contour_dat_string, curve_dat_string,
    parse_contour_dat, parse_curve_dat, parse_prexmu_dat, parse_residue_dat, parse_xscorr_raw_dat,
    prexmu_dat_string, read_contour_dat, read_curve_dat, read_prexmu_dat, read_residue_dat,
    read_xscorr_raw_dat, residue_dat_string, write_contour_dat, write_curve_dat, write_prexmu_dat,
    write_residue_dat, write_xscorr_raw_dat, xscorr_raw_dat_string,
};
pub use xsecl_bin::{
    XseclBinData, XseclBinTransition, parse_xsecl_bin, read_xsecl_bin, write_xsecl_bin,
    xsecl_bin_string,
};
pub use xsecl_dat::{
    XseclDatData, XseclDatHeader, parse_xsecl_dat, parse_xsecl2_dat, read_xsecl_dat,
    read_xsecl2_dat, write_xsecl_dat, write_xsecl2_dat, xsecl_dat_string, xsecl2_dat_string,
};
pub use xsect_dat::{
    XsectDatData, XsectDatScalars, parse_xsect_dat, read_xsect_dat, write_xsect_dat,
    xsect_dat_string,
};
pub use xsph_input::{XsphAdvanced, XsphControl, XsphGrid, XsphInput};
