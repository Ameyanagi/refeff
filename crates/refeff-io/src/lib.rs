#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

//! Input/output compatibility support for the FEFF10 Rust port.
//!
//! This crate owns FEFF text parsing, FEFF-style intermediate writers, and
//! file-format codecs such as Packed ASCII Data (PAD). Numerical modules should
//! depend on these typed structures rather than re-parsing FEFF text ad hoc.

pub mod apot_bin;
pub mod atom_dat;
pub mod axafs_dat;
pub mod band_dat;
pub mod bphl_dat;
pub mod chi_dat;
pub mod chia_bin;
pub mod cif;
pub mod codec;
pub mod compton_dat;
pub mod compton_input;
pub mod config_dat;
pub mod config_input;
pub mod control_input;
pub mod crpa_dat;
pub mod crpa_input;
pub mod cum_dat;
pub mod danes_dat;
pub mod dmdw_coupling;
pub mod dmdw_input;
pub mod dmdw_out;
pub mod dmdw_self_energy;
pub mod drude_dat;
pub mod dym;
pub mod eels_dat;
pub mod eels_gos_dat;
pub mod eels_input;
pub mod eels_magic_dat;
pub mod emesh_bin;
pub mod energy_output;
pub mod eps_dat;
pub mod error;
pub mod exc_dat;
pub mod feff_bin;
pub mod feffl_bin;
pub mod ff2x_input;
pub mod fms_bin;
pub mod fms_input;
pub mod fmsl_bin;
pub mod format;
pub mod fpf0_dat;
pub mod fullspectrum_options;
pub mod genfmt_input;
pub mod genfmt_output;
pub mod gg_dat;
pub mod global_input;
pub mod grid_input;
pub mod gtr_bin;
pub mod gtr_dat;
pub mod gtrl_dat;
pub mod hamaker_dat;
pub mod highz_out;
pub mod hubbard_bin;
pub mod hubbard_input;
pub mod input;
pub mod ldos_dat;
pub mod ldos_input;
pub mod list_dat;
pub mod log_dat;
pub mod loss_dat;
pub mod mdff_dat;
pub mod mdff_input;
pub mod misc_dat;
pub mod model;
pub mod mpse_dat;
pub mod mtdp;
pub mod nstar_dat;
pub mod opcons_dat;
pub mod osc_str_dat;
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
pub mod rhorrp_gg_bin;
pub mod rixs_dat;
pub mod rixs_input;
pub mod run_output;
pub mod screen_dat;
pub mod screen_input;
pub mod sfconv_input;
pub mod specfunct_dat;
pub mod spring_input;
pub mod structure_output;
pub mod sumrules_dat;
pub mod xmu_dat;
pub mod xmul_dat;
pub mod xscorr_dat;
pub mod xsecl_bin;
pub mod xsecl_dat;
pub mod xsect_dat;
pub mod xsedge_dat;
pub mod xsph_input;
pub mod xsph_rl_dat;

pub use apot_bin::{
    APOT_ATOMIC_COEFFICIENTS, APOT_ATOMIC_COULOMB_SECTION_NUMBER,
    APOT_ATOMIC_DENSITY_SECTION_NUMBER, APOT_ATOMIC_KAPPA_SECTION_NUMBER,
    APOT_ATOMIC_NORB_SECTION_NUMBER, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER,
    APOT_ATOMIC_ORBITAL_SECTION_START, APOT_ATOMIC_ORBITAL_SLOTS, APOT_ATOMIC_RADIAL_POINTS,
    APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
    APOT_CORE_HOLE_GRID_ORIGIN, APOT_CORE_HOLE_GRID_STEP, APOT_CORE_HOLE_RADIAL_POINTS,
    APOT_CORE_HOLE_SECTION_NUMBER, APOT_CORE_HOLE_TOLERANCE, ApotAtomicPotsSectionsInput,
    ApotAtomicScfStateRef, ApotAtomicScfStateSectionsInput, ApotBinData, ApotBinMatrix,
    ApotBinMatrixValues, ApotBinPayload, ApotBinRecords, ApotBinSection, ApotBinType, ApotBinValue,
    ApotCoreHoleColumns, apot_atomic_pots_sections, apot_atomic_scf_sections,
    apot_atomic_scf_sections_from_states, apot_atomic_scf_state_sections,
    apot_atomic_scf_state_sections_from_state, apot_bin_string, apot_core_hole_columns,
    apot_core_hole_coulomb_from_density, apot_core_hole_radii, parse_apot_bin, read_apot_bin,
    refresh_apot_core_hole_coulomb_payload, write_apot_bin,
};
pub use atom_dat::{AtomDatData, atom_dat_string, write_atom_dat};
pub use axafs_dat::{
    AxafsDatData, axafs_dat_from_rows, axafs_dat_from_xsph_axafs, axafs_dat_string,
    parse_axafs_dat, read_axafs_dat, write_axafs_dat,
};
pub use band_dat::{
    BAND_KSPACE_J22MAX, BandKPathHandoffSetup, BandKspaceAngularHandoffSetup,
    BandKspaceEnergyHandoffSetup, BandKspaceFreePropagationNonRelSolveHandoffResult,
    BandKspaceFreePropagationRelSolveHandoffResult,
    BandKspaceFreePropagationSpinDegenerateSolveHandoffResult,
    BandKspaceFreePropagationSpinResolvedSolveHandoffResult, BandKspaceLatticeHandoffSetup,
    BandKspaceNonRelSolveHandoffResult, BandKspaceRelComponentHandoffSetup,
    BandKspaceRelSolveHandoffResult, BandKspaceSolverBasisHandoffSetup,
    BandKspaceSpinDegenerateSolveHandoffResult, BandKspaceSpinResolvedSolveHandoffResult,
    BandPreSolverHandoffSetup, BandstructureDatData, BandstructureDatFromBandResultInput,
    BandstructureDatFromEigenvaluesInput, BandstructureRow, KmeshDatData, KmeshMetadata, KmeshRow,
    band_k_path_setup_from_handoffs, band_kspace_angular_setup_from_handoffs,
    band_kspace_angular_setup_from_lmaxph, band_kspace_energy_setup_from_handoffs,
    band_kspace_ewald_energy_tables_from_handoff,
    band_kspace_free_propagation_non_rel_solve_from_handoffs,
    band_kspace_free_propagation_rel_solve_from_handoffs,
    band_kspace_free_propagation_spin_degenerate_solve_from_handoffs,
    band_kspace_free_propagation_spin_resolved_solve_from_handoffs,
    band_kspace_lattice_setup_from_handoffs, band_kspace_non_rel_solve_from_handoffs,
    band_kspace_non_rel_structure_factor_input,
    band_kspace_rel_component_setup_from_basis_transforms, band_kspace_rel_solve_from_handoffs,
    band_kspace_rel_structure_factor_input, band_kspace_solver_basis_setup_from_handoffs,
    band_kspace_solver_basis_setup_from_lmaxph, band_kspace_spin_degenerate_solve_from_handoffs,
    band_kspace_spin_resolved_solve_from_handoffs, band_kspace_t_matrix_grid_from_handoffs,
    band_pre_solver_setup_from_handoffs, band_pre_solver_setup_from_handoffs_with_lmaxph,
    bandstructure_dat_from_band_result, bandstructure_dat_from_eigenvalues,
    bandstructure_dat_from_kspace_free_propagation_non_rel_handoffs,
    bandstructure_dat_from_kspace_free_propagation_rel_handoffs,
    bandstructure_dat_from_kspace_free_propagation_spin_degenerate_handoffs,
    bandstructure_dat_from_kspace_free_propagation_spin_resolved_handoffs,
    bandstructure_dat_from_kspace_non_rel_handoffs, bandstructure_dat_from_kspace_rel_handoffs,
    bandstructure_dat_from_kspace_spin_degenerate_handoffs,
    bandstructure_dat_from_kspace_spin_resolved_handoffs, bandstructure_dat_string,
    kmesh_dat_from_reciprocal_cell, kmesh_dat_from_reciprocal_cell_with_operations,
    kmesh_dat_string, parse_bandstructure_dat, parse_kmesh_dat, read_bandstructure_dat,
    read_kmesh_dat, write_bandstructure_dat, write_kmesh_dat,
};
pub use bphl_dat::{parse_bphl_dat, read_bphl_dat};
pub use chi_dat::{ChiDatData, chi_dat_string, parse_chi_dat, read_chi_dat, write_chi_dat};
pub use chia_bin::{ChiaBinData, chia_bin_bytes, parse_chia_bin, read_chia_bin, write_chia_bin};
pub use cif::{
    CifAtomSite, CifCell, CifCluster, CifClusterAtom, CifDocument, CifEquivalence,
    CifExpandedStructure, CifPotential, expand_cif_cluster, expand_cif_cluster_with_equivalence,
    expand_cif_structure, expand_cif_structure_with_equivalence, parse_cif, read_cif,
};
pub use compton_dat::{
    ComptonDatData, JzzpDatData, RhozzpDatData, compton_dat_string, jzzp_dat_string,
    parse_compton_dat, parse_jzzp_dat, parse_rhozzp_dat, read_compton_dat, read_jzzp_dat,
    read_rhozzp_dat, rhozzp_dat_string, write_compton_dat, write_jzzp_dat, write_rhozzp_dat,
};
pub use compton_input::{
    ComptonChemicalPotential, ComptonDensityOutputs, ComptonGrid, ComptonInput, ComptonLimits,
    ComptonMomentum, ComptonSwitches, ComptonWindow, compton_input_string,
};
pub use config_dat::{
    CONFIG_DAT_ORBITAL_COUNT, ConfigDatData, ConfigDatPotential, RhorrpConfigOrbitalTables,
    config_dat_from_orbital_configurations, config_dat_string, parse_config_dat, read_config_dat,
    rhorrp_orbital_tables_from_config_dat, write_config_dat,
};
pub use config_input::{
    CONFIG_RECORD_WIDTH, ConfigInput, ConfigOccupation, ConfigRecord, ConfigSlotRows,
    ConfigSlotTable, ConfigState, config_inp_lines_string, config_inp_string,
    config_input_slot_table, config_noble_gas_slot_rows, config_orbital_slot,
    config_record_slot_rows, parse_config_inp, read_config_inp, write_config_inp,
};
pub use control_input::{
    BandEnergyMesh, BandInput, DensityAxis, DensityGrid, DensityGridBohr, DensityGridKind,
    DensityInput, FEFF_BOHR_ANGSTROM, FullSpectrumInput, OpconsInput, ReciprocalCell,
    ReciprocalInput, ReciprocalKMesh, band_input_string, density_input_string,
    fullspectrum_input_string, opcons_input_string, reciprocal_input_string,
};
pub use crpa_dat::{
    CrpaDatAndWscrnDatFromScreenedHubbard, CrpaDatData, CrpaDatFromHubbardSummary,
    CrpaDatFromHubbardSummaryInput, CrpaDatFromScreenedHubbard, CrpaDatFromScreenedHubbardInput,
    CrpaDatFromScreenedHubbardResponseSlicesInput, CrpaResponseAssemblyHandoff,
    CrpaResponseAssemblyHandoffInput, crpa_dat_and_wscrn_dat_from_screened_hubbard,
    crpa_dat_and_wscrn_dat_from_screened_hubbard_response_slices, crpa_dat_from_hubbard_summary,
    crpa_dat_from_screened_hubbard, crpa_dat_string, crpa_response_assembly_handoff,
    parse_crpa_dat, read_crpa_dat, write_crpa_dat,
};
pub use crpa_input::{CrpaInput, crpa_input_string};
pub use cum_dat::{
    CumDatData, CumDatEntry, cum_dat_string, parse_cum_dat, read_cum_dat, write_cum_dat,
};
pub use danes_dat::{
    DanesDatData, danes_dat_string, parse_danes_dat, read_danes_dat, write_danes_dat,
};
pub use dmdw_coupling::{
    DmdwA2DatData, DmdwCouplingTable, dmdw_a2_dat_from_coupling, dmdw_a2_dat_string,
    dmdw_coupling_table_string, dmdw_phonon_coupling_from_tables, parse_dmdw_a2_dat,
    parse_dmdw_coupling_table, read_dmdw_a2_dat, read_dmdw_coupling_table, write_dmdw_a2_dat,
    write_dmdw_coupling_table,
};
pub use dmdw_input::{
    DmdwCalculation, DmdwInput, DmdwPath, DmdwPdosOptions, DmdwSelfEnergyOptions, dmdw_input_string,
};
pub use dmdw_out::{
    DmdwOutData, DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment, DmdwOutPole, DmdwOutSection,
    DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue, dmdw_out_string, parse_dmdw_out,
    read_dmdw_out, write_dmdw_out,
};
pub use dmdw_self_energy::{
    DmdwA2fInfoData, DmdwAkwDatData, DmdwEnergyGridInfo, DmdwSelfEnergyDatData,
    DmdwSpectralInfoData, dmdw_a2f_info_from_pole_weighted, dmdw_a2f_info_string,
    dmdw_akw_dat_string, dmdw_egrid_info_string, dmdw_self_energy_dat_string,
    dmdw_spectral_info_string, parse_dmdw_a2f_info, parse_dmdw_akw_dat, parse_dmdw_egrid_info,
    parse_dmdw_self_energy_dat, parse_dmdw_spectral_info, read_dmdw_a2f_info, read_dmdw_akw_dat,
    read_dmdw_egrid_info, read_dmdw_self_energy_dat, read_dmdw_spectral_info, write_dmdw_a2f_info,
    write_dmdw_akw_dat, write_dmdw_egrid_info, write_dmdw_self_energy_dat,
    write_dmdw_spectral_info,
};
pub use drude_dat::{
    DrudeDatData, drude_dat_from_grid, drude_dat_string, parse_drude_dat, read_drude_dat,
    write_drude_dat,
};
pub use dym::{
    DymCoordinates, DymData, DymFeffConversion, DymSpectrum, DymToFeffOptions, DymType2Metadata,
    DymUniqueAtom, convert_dym_to_feff, dym_feff_inp_string, dym_string, parse_dym, read_dym,
    write_dym, write_dym_feff_outputs,
};
pub use eels_dat::{
    EELS_TENSOR_LABELS, EelsDatData, eels_dat_string, parse_eels_dat, read_eels_dat, write_eels_dat,
};
pub use eels_gos_dat::{
    EelsGos1DatData, EelsGos2DatData, eels_gos_dat_from_table, eels_gos1_dat_string,
    eels_gos2_dat_string, parse_eels_gos1_dat, parse_eels_gos2_dat, read_eels_gos1_dat,
    read_eels_gos2_dat, write_eels_gos1_dat, write_eels_gos2_dat,
};
pub use eels_input::{
    EelsAngles, EelsControl, EelsInput, EelsPolarization, EelsQMesh, eels_input_string,
};
pub use eels_magic_dat::{
    EelsMagicDatData, eels_magic_dat_from_collection_table, eels_magic_dat_string,
    parse_eels_magic_dat, read_eels_magic_dat, write_eels_magic_dat,
};
pub use emesh_bin::{
    EmeshBinData, emesh_bin_bytes, emesh_bin_from_phase_bin, parse_emesh_bin, read_emesh_bin,
    write_emesh_bin,
};
pub use energy_output::{
    ChemicalDatData, EdgesDatData, EdgesDatRow, EmeshDatData, chemical_dat_string,
    edges_dat_string, emesh_dat_from_phase_bin, emesh_dat_string, parse_chemical_dat,
    parse_edges_dat, parse_emesh_dat, read_chemical_dat, read_edges_dat, read_emesh_dat,
    write_chemical_dat, write_edges_dat, write_emesh_dat,
};
pub use eps_dat::{
    EpsDatData, eps_dat_from_fullspectrum_scattering_dielectric,
    eps_dat_from_fullspectrum_scattering_factors, eps_dat_string, parse_eps_dat, read_eps_dat,
    write_eps_dat,
};
pub use error::{IoError, Result};
pub use exc_dat::{
    ExcDatData, SfconvRdepsPoleTable, exc_dat_from_excitation_poles, exc_dat_string, parse_exc_dat,
    read_exc_dat, read_or_create_sfconv_rdeps, sfconv_apl_dat_string,
    sfconv_rdeps_fallback_exc_dat, sfconv_rdeps_fallback_exc_dat_string,
    sfconv_rdeps_fallback_poles, sfconv_rdeps_from_exc_dat, write_exc_dat, write_sfconv_apl_dat,
};
pub use feff_bin::{
    FeffBinData, FeffBinPath, FeffBinPotential, feff_bin_string, parse_feff_bin, read_feff_bin,
    write_feff_bin,
};
pub use feffl_bin::{
    FefflBinData, feffl_bin_string, parse_feffl_bin, read_feffl_bin, write_feffl_bin,
};
pub use ff2x_input::{Ff2xControl, Ff2xCorrections, Ff2xDebye, Ff2xInput, ff2x_input_string};
pub use fms_bin::{
    FMS_BIN_DEFAULT_PAD_WIDTH, FmsBinData, fms_bin_string, parse_fms_bin, read_fms_bin,
    write_fms_bin,
};
pub use fms_input::{
    FmsCluster, FmsControl, FmsDebye, FmsInput, RhorrpFmsInputHandoff, fms_input_string,
    rhorrp_handoff_from_fms_input,
};
pub use fmsl_bin::{FmslBinData, fmsl_bin_string, parse_fmsl_bin, read_fmsl_bin, write_fmsl_bin};
pub use fpf0_dat::{
    Fpf0DatData, Fpf0Oscillator, fpf0_dat_string, parse_fpf0_dat, read_fpf0_dat, write_fpf0_dat,
};
pub use fullspectrum_options::{
    FullSpectrumAutomaticFineStructure, FullSpectrumComponent, FullSpectrumComponentEdge,
    FullSpectrumComponentEdgeSource, FullSpectrumDrudeOptions, FullSpectrumOptions,
    FullSpectrumOptionsEnergyGrid, parse_fullspectrum_options, read_fullspectrum_options,
};
pub use genfmt_input::{GenfmtControl, GenfmtInput, genfmt_input_string};
pub use genfmt_output::{GenfmtOutputData, GenfmtOutputPaths, write_genfmt_output_files};
pub use gg_dat::{
    GgDatData, GgDatRixsHandoff, GgDatSection, gg_bin_bytes, gg_bin_string, gg_dat_bytes,
    gg_dat_rixs_handoff, gg_dat_string, parse_gg_bin, parse_gg_bin_bytes, parse_gg_dat,
    parse_gg_dat_bytes, read_gg_bin, read_gg_dat, write_gg_bin, write_gg_dat,
};
pub use global_input::{
    CfAverage, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, GlobalQVector,
    global_input_string,
};
pub use grid_input::{
    GridInput, GridKind, GridMinimum, GridPoint, GridRecord, GridRegularRecord, GridUserRecord,
    grid_inp_string, parse_grid_inp, read_grid_inp, write_grid_inp,
};
pub use gtr_bin::{
    GtrBinData, GtrBinLdosTraceHandoff, gtr_bin_bytes, gtr_bin_from_ldos_trace_grid,
    gtr_bin_ldos_trace_handoff, parse_gtr_bin, read_gtr_bin, write_gtr_bin,
};
pub use gtr_dat::{GtrDatData, gtr_dat_string, parse_gtr_dat, read_gtr_dat, write_gtr_dat};
pub use gtrl_dat::{GtrlDatData, gtrl_dat_string, parse_gtrl_dat, read_gtrl_dat, write_gtrl_dat};
pub use hamaker_dat::{
    HamakerDatData, hamaker_dat_from_fullspectrum_epsilon, hamaker_dat_string, parse_hamaker_dat,
    read_hamaker_dat, write_hamaker_dat,
};
pub use highz_out::{
    HighZOut, HighZOutRow, highz_out_string, parse_highz_out, read_highz_out, write_highz_out,
};
pub use hubbard_bin::{
    HubbardAphaseBinData, HubbardLdosGtrBinData, HubbardLdosGtrMBinData,
    HubbardLdosGtrMTraceHandoff, HubbardLdosGtrOffBinData, HubbardTransformationBinData,
    HubbardVnlmBinData, aphase_hubbard_bin_bytes, hubbard_ldos_gtr_bin_bytes,
    hubbard_ldos_gtr_m_bin_bytes, hubbard_ldos_gtr_m_trace_handoff, hubbard_ldos_gtr_off_bin_bytes,
    parse_aphase_hubbard_bin, parse_aphase_hubbard_bin_inferred, parse_hubbard_ldos_gtr_bin,
    parse_hubbard_ldos_gtr_bin_inferred, parse_hubbard_ldos_gtr_m_bin,
    parse_hubbard_ldos_gtr_m_bin_inferred, parse_hubbard_ldos_gtr_off_bin,
    parse_transformation_hubbard_bin, parse_transformation_hubbard_bin_inferred,
    parse_v_hubbard_bin, parse_v_hubbard_bin_inferred, read_aphase_hubbard_bin,
    read_aphase_hubbard_bin_inferred, read_hubbard_ldos_gtr_bin,
    read_hubbard_ldos_gtr_bin_inferred, read_hubbard_ldos_gtr_m_bin,
    read_hubbard_ldos_gtr_m_bin_inferred, read_hubbard_ldos_gtr_off_bin,
    read_transformation_hubbard_bin, read_transformation_hubbard_bin_inferred, read_v_hubbard_bin,
    read_v_hubbard_bin_inferred, transformation_hubbard_bin_bytes, v_hubbard_bin_bytes,
    write_aphase_hubbard_bin, write_hubbard_ldos_gtr_bin, write_hubbard_ldos_gtr_m_bin,
    write_hubbard_ldos_gtr_off_bin, write_transformation_hubbard_bin, write_v_hubbard_bin,
};
pub use hubbard_input::{HubbardInput, hubbard_input_string};
pub use input::{FeffInput, FeffLine, LineKind, SourceLocation};
pub use ldos_dat::{
    FullSpectrumLdosData, LDOS_ORBITAL_LABELS, LDOS_SPIN_ORBITAL_LABELS, LdosDatData,
    LdosDatFromFf2rho, LdosDatFromFf2rhoInput, LdosElectronCount, LdosMagneticDatData,
    LdosMagneticDatFromFf2rho, LdosMagneticDatFromFf2rhoInput, LdosSpinDatFromFf2rhoInput,
    LmdosDatData, RhocDatData, RhocmDatData, fullspectrum_ldos_from_ldos_dat, ldos_dat_from_ff2rho,
    ldos_dat_string, ldos_magnetic_dat_from_ff2rho, ldos_magnetic_dat_string,
    ldos_spin_dat_from_ff2rho, lmdos_dat_string, parse_ldos_dat, parse_ldos_magnetic_dat,
    parse_lmdos_dat, parse_rhoc_dat, parse_rhocm_dat, read_ldos_dat, read_lmdos_dat, read_rhoc_dat,
    read_rhocm_dat, rhoc_dat_string, rhocm_dat_string, write_ldos_dat, write_lmdos_dat,
    write_rhoc_dat, write_rhocm_dat,
};
pub use ldos_input::{LdosControl, LdosFms, LdosInput, LdosMesh, ldos_input_string};
pub use list_dat::{
    ListDatData, ListDatEntry, list_dat_string, parse_list_dat, read_list_dat, write_list_dat,
};
pub use log_dat::{
    GenfmtPathLogEntry, GenfmtPathLogInput, GenfmtPathLogMode, LogDatData, ModuleLogData,
    cached_pot_stage_module_log, genfmt_jas_path_outputs_module_log,
    genfmt_ordinary_path_outputs_module_log, genfmt_path_module_log,
    is_atomic_potential_module_log, log_dat_string, module_log_dat_string, parse_log_dat,
    parse_module_log_dat, pot_module_log, read_log_dat, read_module_log_dat, write_log_dat,
    write_module_log_dat,
};
pub use loss_dat::{LossDatData, loss_dat_string, parse_loss_dat, read_loss_dat, write_loss_dat};
pub use mdff_dat::{MdffDatData, mdff_dat_string, parse_mdff_dat, read_mdff_dat, write_mdff_dat};
pub use mdff_input::{MdffInput, mdff_input_string};
pub use misc_dat::{MiscDatData, misc_dat_string, parse_misc_dat, read_misc_dat, write_misc_dat};
pub use model::{Atom, FeffDocument, Potential};
pub use mpse_dat::{MpseDatData, mpse_dat_string, parse_mpse_dat, read_mpse_dat, write_mpse_dat};
pub use mtdp::{MtdpData, mtdp_string, parse_mtdp, read_mtdp, write_mtdp};
pub use nstar_dat::{
    NStarDatData, NStarDatEntry, nstar_dat_string, parse_nstar_dat, read_nstar_dat, write_nstar_dat,
};
pub use opcons_dat::{
    OpconsDatData, opcons_dat_from_fullspectrum_epsilon_minus_one,
    opcons_dat_from_fullspectrum_optical_constants, opcons_dat_string, parse_opcons_dat,
    read_opcons_dat, write_opcons_dat,
};
pub use osc_str_dat::{
    OscStrDatData, OscStrRow, osc_str_dat_string, osc_str_row_from_fullspectrum_edge,
    parse_osc_str_dat, read_osc_str_dat, write_osc_str_dat,
};
pub use paths_dat::{
    PathsDatAtom, PathsDatData, PathsDatGenfmtPath, PathsDatPath, PathsdPathsDatDataInput,
    PathsdPathsDatPathInput, genfmt_nstar_path_inputs, genfmt_path_rotation_inputs,
    parse_paths_dat, paths_dat_string, pathsd_paths_dat_data, pathsd_paths_dat_path,
    read_paths_dat, write_paths_dat,
};
pub use paths_input::{PathsControl, PathsCriteria, PathsInput, paths_input_string};
pub use phase_bin::{
    PhaseBinBandData, PhaseBinBandSearchSetup, PhaseBinData, PhaseBinGenfmtData,
    PhaseBinPathHandoff, PhaseBinPotential, PhaseBinRawPads, PhaseBinRixsHandoff, PhaseBinScalars,
    RhorrpPhaseBinHandoff, band_search_setup_from_handoffs,
    band_search_setup_from_handoffs_with_lmaxph, genfmt_core_legendre_normalization_from_feff_dims,
    genfmt_driver_setup_from_handoffs, genfmt_edge_start_index_from_phase,
    genfmt_jas_driver_setup_from_handoffs, genfmt_jas_path_setups_from_handoffs,
    genfmt_jas_q_angles_from_handoffs, genfmt_jas_transition_indices_from_handoffs,
    genfmt_jas_transition_setups_from_handoff_setups, genfmt_legendre_normalization_from_feff_dims,
    genfmt_nstar_driver_input_from_handoffs, genfmt_nstar_rows_from_handoffs,
    genfmt_ordinary_path_setups_from_handoffs, genfmt_ordinary_spin_radial_factors_from_phase,
    genfmt_ordinary_transition_b_matrix_from_handoffs,
    genfmt_ordinary_transition_matrices_from_handoff_setups, parse_phase_bin,
    path_phase_criteria_tables_from_phase_bin, phase_bin_band_data, phase_bin_genfmt_data,
    phase_bin_path_handoff_from_phase_bin, phase_bin_rixs_handoff_from_phase_bin,
    phase_bin_rixs_transition_phase_shifts_from_handoff,
    phase_bin_rixs_transition_setup_from_handoffs, phase_bin_string, read_phase_bin,
    rhorrp_phase_handoff_from_phase_bin, rhorrp_phase_table_from_phase_bin,
    rixs_angular_limits_from_phase_bin, rixs_transition_moments_from_phase_bin, write_phase_bin,
};
pub use pot_bin::{
    FullSpectrumPotentialState, PotBinData, PotBinScalars, RHORRP_POT_BIN_RADIAL_DX,
    RHORRP_WAVEFUNCTION_RADIAL_COUNT, RHORRP_WAVEFUNCTION_RADIAL_X0,
    RhorrpPotBinWavefunctionHandoff, RhorrpWavefunctionTablesHandoff,
    RhorrpWavefunctionTablesHandoffInput, fullspectrum_number_density_from_pot_bin,
    fullspectrum_potential_state_from_pot_bin, parse_pot_bin, pot_bin_string, read_pot_bin,
    rhorrp_wavefunction_handoff_from_pot_bin, rhorrp_wavefunction_tables_from_handoffs,
    write_pot_bin,
};
pub use pot_diagnostics::{
    Fort16Data, ScfConvergenceData, ScfConvergenceLine, ScfConvergenceRow,
    convergence_scf_fine_string, convergence_scf_string, fort16_string, parse_convergence_scf,
    parse_convergence_scf_fine, parse_fort16, read_convergence_scf, read_convergence_scf_fine,
    read_fort16, write_convergence_scf, write_convergence_scf_fine, write_fort16,
};
pub use pot_input::{
    PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp, PotRun, PotScattering,
    PotThermal, PotTolerances, RhorrpPotInputControls, pot_input_string,
    rhorrp_controls_from_pot_input,
};
pub use pot_output::{
    PotentialDatInput, PotentialDatSetInput, pot_dat_string, potential_dat_filename,
    potential_dat_outputs, potential_dat_outputs_from_bins, write_potential_dat,
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
    RhorrpDensityGridTablesHandoffInput, RhorrpDensityGridTablesOutputInput,
    RhorrpDensityOutputBohrInput, RhorrpDensityOutputData, RhorrpNearestAtomColumnsBohrInput,
    rhorrp_density_grid_tables_input_from_handoffs, rhorrp_density_output_from_bohr,
    rhorrp_density_output_from_grid, rhorrp_density_output_from_grid_tables,
    rhorrp_density_output_from_grid_with_nearest, rhorrp_nearest_atom_columns_from_bohr,
    write_rhorrp_density_grid_output, write_rhorrp_density_grid_output_from_tables,
    write_rhorrp_density_grid_output_with_nearest, write_rhorrp_density_output_from_bohr,
};
pub use rhorrp_gg_bin::{
    RhorrpGgDiagBinData, RhorrpGgPairMatrix, RhorrpGgSliceBinData, parse_rhorrp_gg_diag_bin,
    parse_rhorrp_gg_slice_bin, read_rhorrp_gg_diag_bin, read_rhorrp_gg_slice_bin,
    rhorrp_gg_diag_bin_bytes, rhorrp_gg_diag_matrices, rhorrp_gg_diag_matrix,
    rhorrp_gg_pair_matrix, rhorrp_gg_slice_bin_bytes, rhorrp_gg_slice_block,
    rhorrp_gg_slice_central_matrices, write_rhorrp_gg_diag_bin, write_rhorrp_gg_slice_bin,
};
pub use rixs_dat::{
    RixsDatFromFinalSpectrum, RixsLineData, RixsMapData, RixsSkipCalcEdgeOffsets, parse_rixs_line,
    parse_rixs_map, read_rixs_line, read_rixs_map, rixs_dat_from_final_spectrum,
    rixs_line_from_map_diagonal, rixs_line_string, rixs_map_string,
    rixs_skip_calc_edge_offsets_from_edges_dat, rixs_skip_calc_herfd_from_rixs_et_map,
    rixs_skip_calc_outputs_from_rixs_et_map, rixs_skip_calc_satellite_outputs, write_rixs_line,
    write_rixs_map,
};
pub use rixs_input::{
    RixsBroadening, RixsEnergyWindow, RixsInput, RixsSwitches, rixs_input_string,
};
pub use run_output::{
    FloatingPointNote, RunLineEnding, RunModuleEvent, RunModuleEventKind, RunStderrData,
    RunStdoutData, parse_fort11, parse_run_stderr, parse_run_stdout, read_fort11, read_run_stderr,
    read_run_stdout, run_stderr_string, run_stdout_string, write_run_stderr, write_run_stdout,
};
pub use screen_dat::{
    PotScfCorvalLdosHandoff, PotScfCorvalLdosHandoffInput, PotScfFmsSourceGridHandoff,
    PotScfFmsSourceGridHandoffInput, PotScfFovrgSourceGridFromPlanInput,
    PotScfFovrgSourceGridHandoff, PotScfFovrgSourceGridHandoffInput, PotScfFovrgSourceGridPlan,
    PotScfFovrgSourceGridPlanInput, ScreenFmsClusterGreenHandoff,
    ScreenFmsClusterGreenHandoffInput, ScreenFovrgPhaseGridHandoff,
    ScreenFovrgPhaseGridHandoffInput, ScreenFovrgPhaseHandoff, ScreenFovrgRadialHandoff,
    ScreenFovrgRadialHandoffInput, ScreenPotentialKernelHandoff, ScreenPotentialKernelHandoffInput,
    ScreenResponseAssemblyHandoff, ScreenResponseAssemblyHandoffInput, VtotDatData, WscrnDatData,
    WscrnDatFromCoreHoleResponseInput, WscrnDatFromScreenResponseInput,
    WscrnDatFromScreenResponseSlicesInput, parse_vtot_dat, parse_wscrn_dat,
    pot_scf_corval_ldos_handoff, pot_scf_fms_source_grid_handoff,
    pot_scf_fovrg_source_grid_handoff, pot_scf_fovrg_source_grid_handoff_from_plan,
    pot_scf_fovrg_source_grid_plan, read_vtot_dat, read_wscrn_dat,
    screen_fms_cluster_green_handoff, screen_fovrg_phase_grid_handoff, screen_fovrg_phase_handoff,
    screen_fovrg_radial_handoff, screen_potential_kernel_handoff, screen_response_assembly_handoff,
    vtot_dat_from_wscrn_and_pot_bin, vtot_dat_from_wscrn_and_total_potential, vtot_dat_string,
    write_vtot_dat, write_wscrn_dat, wscrn_dat_from_core_hole_response,
    wscrn_dat_from_screen_response, wscrn_dat_from_screen_response_slices, wscrn_dat_string,
};
pub use screen_input::{ScreenInput, screen_input_string};
pub use sfconv_input::{
    SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvControl, SfconvInput, SfconvSo2convFeffPathData,
    SfconvSo2convHeader, SfconvSo2convTarget, SfconvSo2convTargetData, SfconvSo2convTargetKind,
    SfconvSpectrum, SfconvWindow, sfconv_input_string,
    sfconv_so2conv_chi_data_from_convolution_rows, sfconv_so2conv_convoluted_target_data_string,
    sfconv_so2conv_feff_path_data_from_averages, sfconv_so2conv_feff_path_data_string,
    sfconv_so2conv_header_from_text, sfconv_so2conv_material_input_from_header,
    sfconv_so2conv_target_data_from_text, sfconv_so2conv_target_data_string,
    sfconv_so2conv_targets, sfconv_so2conv_xmu_data_from_convolution_rows,
    write_sfconv_so2conv_convoluted_target_data, write_sfconv_so2conv_feff_path_data,
    write_sfconv_so2conv_target_data,
};
pub use specfunct_dat::{
    SPECFUNCT_DAT_INFO_COLUMNS, SfconvSpecfunctChiDataInput, SfconvSpecfunctCompatibilityInput,
    SfconvSpecfunctData, SfconvSpecfunctExafsRowsInput, SfconvSpecfunctFeffPathDataInput,
    SfconvSpecfunctSpectralRowsInput, SfconvSpecfunctTargetDataInput,
    SfconvSpecfunctXanesRowsInput, SfconvSpecfunctXmuDataInput, parse_specfunct_dat,
    read_specfunct_dat, sfconv_specfunct_chi_data_from_cache,
    sfconv_specfunct_data_from_spectral_rows, sfconv_specfunct_exafs_convolution_rows,
    sfconv_specfunct_feff_path_data_from_cache, sfconv_specfunct_interpolate_momentum,
    sfconv_specfunct_matches_so2conv_inputs, sfconv_specfunct_momentum_interpolation_input,
    sfconv_specfunct_target_data_from_cache, sfconv_specfunct_xanes_convolution_rows,
    sfconv_specfunct_xmu_data_from_cache, specfunct_dat_bytes, write_specfunct_dat,
};
pub use spring_input::{
    SPRING_DEFAULT_ACUT, SPRING_DEFAULT_DOSFIT, SPRING_DEFAULT_RESOLUTION, SPRING_DEFAULT_WMAX,
    SpringAngle, SpringInput, SpringStretch, SpringVdos, parse_spring_inp, read_spring_inp,
    spring_inp_string, write_spring_inp,
};
pub use structure_output::{
    AtomsDat, AtomsDatRow, DimensionsDat, GeomDat, GeomDatRow, PathfinderGeomHandoff,
    RhorrpGeomHandoff, atoms_dat_string, dimensions_dat_string, geom_dat_string,
    pathfinder_geom_handoff_from_geom_dat, rhorrp_fms_inclusion_counts_from_geom_handoff,
    rhorrp_geom_handoff_from_geom_dat,
};
pub use sumrules_dat::{
    SumRulesDatData, parse_sumrules_dat, read_sumrules_dat, sumrules_dat_from_opcons,
    sumrules_dat_string, write_sumrules_dat,
};
pub use xmu_dat::{
    FullSpectrumBackgroundSegmentData, FullSpectrumFineStructureSegmentData, FullSpectrumXmuData,
    FullSpectrumXmuUnits, XmuDatData, fullspectrum_absolute_xmu_from_xmu_dat,
    fullspectrum_background_segment_from_fprime_xmu_dat,
    fullspectrum_imaginary_fine_structure_segment_from_xmu_dat,
    fullspectrum_normalized_xmu_from_xmu_dat,
    fullspectrum_real_fine_structure_segment_from_xmu_dat, parse_xmu_dat, read_xmu_dat,
    valence_epsilon2_from_xmu_dat, write_xmu_dat, xmu_dat_string,
};
pub use xmul_dat::{
    XmulDatData, XmulDatFromNrixsDecompositionInput, parse_xmul_dat, read_xmul_dat, write_xmul_dat,
    xmul_dat_from_nrixs_decomposition, xmul_dat_string,
};
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
    XseclDatData, XseclDatHeader, XseclFromXsphNrixs, XseclFromXsphNrixsInput, parse_xsecl_dat,
    parse_xsecl2_dat, read_xsecl_dat, read_xsecl2_dat, write_xsecl_dat, write_xsecl2_dat,
    xsecl_dat_string, xsecl_from_xsph_nrixs, xsecl2_dat_string,
};
pub use xsect_dat::{
    XsectDatData, XsectDatFromXsphSpin, XsectDatFromXsphSpinInput, XsectDatRixsHandoff,
    XsectDatScalars, XsectFf2xHandoff, parse_xsect_dat, read_xsect_dat, write_xsect_dat,
    xsect_dat_ff2x_handoff, xsect_dat_from_xsph_spin_merge, xsect_dat_rixs_handoff,
    xsect_dat_string,
};
pub use xsedge_dat::{
    XsedgeDatData, XsedgeDatFromTdldaRowsInput, parse_xsedge_dat, read_xsedge_dat,
    write_xsedge_dat, xsedge_dat_from_tdlda_rows, xsedge_dat_string,
};
pub use xsph_input::{
    XsphAdvanced, XsphControl, XsphControlHeaderFormat, XsphGrid, XsphGridHeaderFormat, XsphInput,
    XsphInputSourceFormat, xsph_input_string,
};
pub use xsph_rl_dat::{
    XsphRlDatData, XsphRlDatRecord, XsphRlDatRixsHandoff, parse_xsph_rl_dat, read_xsph_rl_dat,
    write_xsph_rl_dat, xsph_rl_dat_rixs_handoff, xsph_rl_dat_string,
};
