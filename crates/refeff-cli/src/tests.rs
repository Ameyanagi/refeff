use super::{
    Cli, atomic, band, eelsmdff, execute_rdinp, opcons, paths, run_cli, run_feff_to_dir,
    run_module, run_supported_cached_modules, wpot, xsph,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
};
use refeff_io::rdinp;
use refeff_io::{
    ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection, ApotBinType,
    BandstructureDatData, BandstructureRow, ChiDatData, CrpaDatData, DanesDatData, DmdwOutData,
    DmdwOutHeader, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature, DymCoordinates, DymData,
    EdgesDatData, EdgesDatRow, EelsDatData, EmeshBinData, EmeshDatData, EpsDatData, ExcDatData,
    FeffBinData, FeffBinPath, FeffBinPotential, FeffDocument, FeffInput, FmsBinData, Fort16Data,
    Fpf0DatData, HamakerDatData, HubbardVnlmBinData, JzzpDatData, LdosDatData, ListDatData,
    ListDatEntry, MdffDatData, MiscDatData, ModuleLogData, MpseDatData, OscStrDatData, OscStrRow,
    PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData, PotBinScalars,
    RhorrpDensityTextData, RhorrpNearestAtomColumns, RhozzpDatData, RixsMapData,
    SFCONV_SO2CONV_CONVOLUTED_MARKER, ScfConvergenceData, ScfConvergenceLine, ScfConvergenceRow,
    VtotDatData, WscrnDatData, XmuDatData, XscorrComplexTable, XscorrCurveDatData,
    XscorrRawDatData, XsectDatData, XsectDatScalars, XsphRlDatData, XsphRlDatRecord,
    parse_loss_dat, read_aphase_hubbard_bin_inferred, read_apot_bin, read_bandstructure_dat,
    read_chi_dat, read_compton_dat, read_contour_dat, read_convergence_scf,
    read_convergence_scf_fine, read_crpa_dat, read_curve_dat, read_danes_dat, read_dmdw_out,
    read_eels_dat, read_emesh_bin, read_emesh_dat, read_exc_dat, read_feff_bin, read_feffl_bin,
    read_fms_bin, read_fort16, read_fpf0_dat, read_gg_bin, read_gg_dat, read_gtr_dat,
    read_hamaker_dat, read_jzzp_dat, read_ldos_dat, read_list_dat, read_mdff_dat, read_misc_dat,
    read_module_log_dat, read_mpse_dat, read_nstar_dat, read_opcons_dat, read_osc_str_dat,
    read_paths_dat, read_pot_bin, read_prexmu_dat, read_residue_dat, read_rhoc_dat,
    read_rhorrp_density_text, read_rhozzp_dat, read_rixs_line, read_rixs_map, read_sumrules_dat,
    read_vtot_dat, read_wscrn_dat, read_xmu_dat, read_xscorr_raw_dat, read_xsect_dat,
    read_xsedge_dat, read_xsph_rl_dat, write_apot_bin, write_bandstructure_dat, write_chi_dat,
    write_contour_dat, write_convergence_scf, write_convergence_scf_fine, write_crpa_dat,
    write_curve_dat, write_danes_dat, write_dmdw_out, write_dym, write_edges_dat, write_eels_dat,
    write_emesh_bin, write_emesh_dat, write_eps_dat, write_exc_dat, write_feff_bin,
    write_feffl_bin, write_fms_bin, write_fort16, write_hamaker_dat, write_jzzp_dat,
    write_ldos_dat, write_list_dat, write_mdff_dat, write_misc_dat, write_module_log_dat,
    write_mpse_dat, write_nstar_dat, write_osc_str_dat, write_paths_dat, write_phase_bin,
    write_pot_bin, write_prexmu_dat, write_residue_dat, write_rhoc_dat, write_rhorrp_density_text,
    write_rhozzp_dat, write_rixs_map, write_v_hubbard_bin, write_vtot_dat, write_wscrn_dat,
    write_xmu_dat, write_xscorr_raw_dat, write_xsect_dat, write_xsedge_dat, write_xsph_rl_dat,
};
use refeff_io::{PathsDatAtom, PathsDatData, PathsDatPath};
use std::path::{Path, PathBuf};
use std::process::Command;

mod fixtures;
use fixtures::*;

mod error_logs;
mod full_run_core_cache;
mod full_run_spectrum_cache;
mod module_aliases;
mod rdinp_stage;
