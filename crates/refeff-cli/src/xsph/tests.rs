use super::{
    NormalXsectPhiscfWfirdcAssemblyInput, NrixsSpectrumRadialChannel, XSPH_BPHL_REQUIRED_ERROR,
    XSPH_SOURCE_REQUIREMENT_ERROR, XsphCachePaths, XsphTdldaFileProjectorCandidatesInput,
    has_cached_xsph_output, has_supported_phase_handoff, has_supported_phase_mesh_handoff,
    has_supported_phase_text_handoff, has_supported_tdlda_xsedge_output, has_supported_xsph_output,
    load_xsph_broadened_table, normal_potential_orbital_tables, normal_xsect_hole_orbital_energy,
    normal_xsect_phiscf_coarse_count_for_active_len, normal_xsect_phiscf_fine_len_for_coarse_count,
    normal_xsect_phiscf_occupied_table, normal_xsect_phiscf_wfirdc_assembly,
    nrixs_spectrum_handoffs_from_rows, nrixs_spectrum_radial_source_context_from_handoffs,
    nrixs_spectrum_row_from_radial_channels, nrixs_spectrum_source_plan_from_handoffs,
    phase_transition_dimensions, pot_has_bound_orbital_handoffs, run_for_input, run_in_dir,
    run_required_in_dir, run_supported_phase_handoff_in_dir,
    run_supported_phase_mesh_handoff_in_dir, run_supported_phase_text_handoff_in_dir,
    tdlda_angular_kernel_from_source_plan, tdlda_coulomb_fields_from_source_plan,
    tdlda_direct_kernel_from_source_plan, tdlda_energy_rows_from_source_plan,
    tdlda_file_projector_candidates_from_source, tdlda_getchi0_kernel_from_source_plan,
    tdlda_nonlocal_exchange_from_source_plan, tdlda_pmbse_channel_multipliers_from_source,
    tdlda_projected_kernel_from_source_plan, tdlda_projector_rows_from_source_plan,
    tdlda_radial_kernel_from_source_plan, tdlda_raw_response_from_source_plan,
    tdlda_raw_response_inputs_from_source_plan, tdlda_row_wave_numbers_from_source_plan,
    tdlda_weight_response_from_source_plan, tdlda_xsectd_source_plan_from_caches,
    tdlda_xsedge_dat_from_raw_source_components, validate_xsect_spin_ground_states,
    write_tdlda_xsedge_dat_from_source_components, xsect_angular_controls,
    xsect_angular_controls_from_values,
};
use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, Array4, Array5, Axis};
use num_complex::Complex64;
use refeff_core::{
    BPHL_RADIUS_COUNT, BPHL_REDUCED_ENERGY_COUNT, FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV,
    XsphAxafsInput, XsphRadialIntegralInput, XsphRadialIntegralMode,
    XsphTdldaBroadenedChannelSpectra, XsphTdldaProjectorSelector, XsphTransitionMultipole,
    core_hole_width_ev, somm2, xsph_axafs, xsph_radial_integral,
    xsph_tdlda_decode_projector_selector,
};
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    ApotBinData, ApotBinPayload, ApotBinSection, ApotBinType, ApotBinValue, AxafsDatData,
    CONFIG_DAT_ORBITAL_COUNT, CfAverage, ConfigDatData, ConfigDatPotential, EelsAngles,
    EelsControl, EelsInput, EelsPolarization, EelsQMesh, EmeshBinData, EmeshDatData, GlobalControl,
    GlobalInput, GlobalNorms, GlobalQControl, GridInput, GridKind, GridMinimum, GridPoint,
    GridRecord, GridRegularRecord, GridUserRecord, HubbardAphaseBinData, HubbardInput,
    HubbardVnlmBinData, LossDatData, ModuleLogData, MpseDatData, PhaseBinData, PhaseBinPotential,
    PhaseBinScalars, PotBinData, PotBinScalars, VtotDatData, XmuDatData, XseclBinData,
    XseclBinTransition, XseclDatData, XseclDatHeader, XsectDatData, XsectDatScalars, XsedgeDatData,
    XsphAdvanced, XsphControl, XsphGrid, XsphInput, XsphInputSourceFormat,
    axafs_dat_from_xsph_axafs, axafs_dat_string, eels_input_string, emesh_bin_from_phase_bin,
    emesh_dat_from_phase_bin, emesh_dat_string, global_input_string, hubbard_input_string,
    parse_axafs_dat, parse_emesh_dat, read_aphase_hubbard_bin_inferred, read_axafs_dat,
    read_emesh_bin, read_emesh_dat, read_exc_dat, read_module_log_dat, read_mpse_dat,
    read_phase_bin, read_pot_bin, read_wscrn_dat, read_xsecl_bin, read_xsecl_dat, read_xsecl2_dat,
    read_xsect_dat, read_xsedge_dat, read_xsph_rl_dat, rhorrp_orbital_tables_from_config_dat,
    sfconv_so2conv_header_from_text, write_aphase_hubbard_bin, write_apot_bin, write_axafs_dat,
    write_config_dat, write_emesh_bin, write_emesh_dat, write_grid_inp, write_loss_dat,
    write_module_log_dat, write_mpse_dat, write_phase_bin, write_pot_bin, write_v_hubbard_bin,
    write_vtot_dat, write_xmu_dat, write_xsecl_bin, write_xsecl_dat, write_xsecl2_dat,
    write_xsect_dat, write_xsedge_dat, write_xsph_rl_dat, xsect_dat_ff2x_handoff, xsect_dat_string,
    xsph_input_string,
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[test]
fn xsph_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_xsph_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_source_broadened_exchange_requires_work_dir_bphl_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;

    let error =
        load_xsph_broadened_table(temp.path(), 10).expect_err("selector 10 requires bphl.dat");
    let message = format!("{error:#}");
    assert!(message.contains(XSPH_BPHL_REQUIRED_ERROR));
    assert!(message.contains("bphl.dat"));

    assert!(load_xsph_broadened_table(temp.path(), 0)?.is_none());
    assert!(load_xsph_broadened_table(temp.path(), 13)?.is_none());

    std::fs::write(temp.path().join("bphl.dat"), synthetic_bphl_dat())?;
    let table = load_xsph_broadened_table(temp.path(), 15)?.expect("selector 15 loads bphl.dat");
    assert_eq!(table.radius_mesh().len(), BPHL_RADIUS_COUNT);
    assert_eq!(table.reduced_energy_mesh().len(), BPHL_REDUCED_ENERGY_COUNT);
    Ok(())
}

#[test]
fn xsph_normal_source_handoff_threads_work_dir_bphl_table() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 10;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    let error = run_in_dir(temp.path()).expect_err("selector 10 requires work-dir bphl.dat");
    assert!(
        format!("{error:#}").contains(XSPH_BPHL_REQUIRED_ERROR),
        "{error:?}"
    );

    std::fs::write(temp.path().join("bphl.dat"), synthetic_bphl_dat())?;
    let written = run_in_dir(temp.path())?;
    assert!(written >= 5);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled XSPH should require cached or source-backed phase state")?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    Ok(())
}

#[test]
fn xsph_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("xsph.inp"), "not an xsph.inp handoff\n")?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);
    assert!(!has_supported_phase_handoff(temp.path())?);
    assert!(!has_supported_phase_text_handoff(temp.path())?);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed XSPH input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("xsph.inp"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);
    assert!(!has_supported_phase_handoff(temp.path())?);
    assert!(!has_supported_phase_text_handoff(temp.path())?);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);
    assert_eq!(read_phase_bin(temp.path().join("phase.bin"))?, phase);
    assert_eq!(read_xsect_dat(temp.path().join("xsect.dat"))?, xsect);
    Ok(())
}

#[test]
fn xsph_module_generates_supported_initial_emesh_handoff_from_pot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_mesh_handoff(temp.path())?);

    let count = run_supported_phase_mesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_input_runner_generates_initial_emesh_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;

    let count = run_for_input(&temp.path().join("feff.inp"))?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(!temp.path().join("log2.dat").exists());
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_initial_emesh_handoff_from_pot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    std::fs::write(temp.path().join("emesh.dat"), "not emesh.dat\n")?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_mesh_handoff(temp.path())?);

    let count = run_supported_phase_mesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("phase.bin").exists());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_initial_emesh_handoff_when_phase_cache_is_malformed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_mesh_handoff(temp.path())?);

    let count = run_supported_phase_mesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(read_phase_bin(temp.path().join("phase.bin")).is_err());
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_rhorrp_initial_emesh_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.ispec = 5;
    })?;
    let mut pot = sample_mpse_pot_bin();
    pot.scalars.core_valence_energy = -1.5;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_mesh_handoff(temp.path())?);
    let written = run_supported_phase_mesh_handoff_in_dir(temp.path())?;

    assert_eq!(written, 2);
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.point_count(), 119);
    assert_eq!(emesh.fermi_index, 0);
    assert_eq!(emesh_bin.point_count(), 119);
    assert_eq!(emesh_bin.horizontal_count, 111);
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_rhorrp_empty_cell_phase_from_pot() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.ispec = 5;
        input.control.i_core_state = 1;
        input.lmaxph = vec![1, 2];
        input.pot_labels = vec!["E0".to_string(), "E1".to_string()];
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_empty_cell_pot_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert!(written > 0);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.fermi_index, 0);
    assert_eq!(phase.energy_count, 19);
    assert_eq!(phase.main_energy_count, 11);
    assert_eq!(phase.auxiliary_energy_count, 0);
    assert_eq!(phase.potential_count(), 2);
    assert!(!has_supported_xsph_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_empty_cell_phase_from_pot_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_empty_cell_xsph_input(temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_empty_cell_pot_bin())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 2);
    assert_eq!(phase.potentials[0].atomic_number, 0);
    assert_eq!(phase.potentials[1].atomic_number, 0);
    assert_eq!(phase.potentials[0].label, "E0");
    assert_eq!(phase.potentials[1].label, "E1");
    assert_phase_lmax_within_compiled_capacity(&phase);
    assert!(phase.potentials.iter().any(|potential| {
        potential
            .phase_shifts
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    }));
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_preserves_matching_cached_xsect_when_empty_cell_phase_is_regenerated() -> Result<()>
{
    let seed = tempfile::tempdir()?;
    write_empty_cell_xsph_input(seed.path())?;
    write_pot_bin(seed.path().join("pot.bin"), &sample_empty_cell_pot_bin())?;
    run_in_dir(seed.path())?;
    let expected_xsect =
        sample_xsect_dat_for_phase(&read_phase_bin(seed.path().join("phase.bin"))?);

    let temp = tempfile::tempdir()?;
    write_empty_cell_xsph_input(temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_empty_cell_pot_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &expected_xsect)?;
    let expected_xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect, expected_xsect);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_does_not_advertise_generated_phase_with_mismatched_cached_xsect_as_complete()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_empty_cell_xsph_input(temp.path())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_empty_cell_pot_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("mismatched generated phase/xsect pair should not complete XSPH")?;
    let chain = format!("{error:?}");
    assert!(
        chain.contains("xsect.dat energy count 2 does not match phase.bin energy count"),
        "{chain}"
    );
    Ok(())
}

#[test]
fn xsph_module_generates_normal_phase_from_pot_and_config_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
        input.print_rl = true;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let source_pot = sample_normal_phase_pot_bin();
    write_pot_bin(temp.path().join("pot.bin"), &source_pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written > 0);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(phase.potentials[0].label, "Cu");
    assert_phase_lmax_within_compiled_capacity(&phase);
    assert!(
        phase
            .reference_energy
            .iter()
            .any(|energy| energy.norm() > 0.0)
    );
    assert!(
        phase.potentials[0]
            .phase_shifts
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    let radial = read_xsph_rl_dat(temp.path().join("rl.dat"))?;
    assert_eq!(radial.angular_limit, phase.potentials[0].lmax);
    assert_eq!(radial.record_count(), radial.records.len());
    assert!(radial.record_count() > 0);
    assert!(radial.records.iter().all(|record| {
        record.regular_large.len() == radial.radial_count()
            && record.regular_small.len() == radial.radial_count()
            && record.angular_momentum <= radial.angular_limit
    }));
    assert!(
        radial
            .records
            .iter()
            .flat_map(|record| record.regular_large.iter())
            .any(|value| value.norm() > 0.0)
    );
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(temp.path().join("xsect.dat").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    assert!(!temp.path().join("mpse.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_normal_phase_from_pot_without_config_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.kappa[0] = -1;
    pot.kappa[1] = -1;
    pot.kappa[2] = 1;
    pot.kappa[3] = -2;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;

    let caches = XsphCachePaths::new(temp.path());
    let pot = read_pot_bin(temp.path().join("pot.bin"))?;
    let orbital_tables = normal_potential_orbital_tables(&caches, &pot)?
        .context("expected pot.bin-derived orbital tables")?;
    assert!(pot_has_bound_orbital_handoffs(&pot, &orbital_tables));

    assert!(!temp.path().join("config.dat").exists());
    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert!(
        phase.potentials[0]
            .phase_shifts
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_module_log_for_normal_potential_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    std::fs::write(temp.path().join("log2.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_for_input(&temp.path().join("feff.inp"))?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.potential_count(), 1);
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    let log = read_module_log_dat(temp.path().join("log2.dat"))?;
    assert_log_contains(&log, "Calculating cross-section and phases ...");
    assert_log_contains(&log, "Done with module: cross-section and phases (XSPH).");
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_phase_cache_from_pot_and_config_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_phase_cache_from_pot_and_config_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let mut stale_phase = expected_phase.clone();
    stale_phase.potentials[0].phase_shifts[(0, 0, 0)] += Complex64::new(0.25, -0.125);
    write_phase_bin(&phase_path, &stale_phase)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_phase_transition_moments_from_xsect_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let mut stale_phase = expected_phase.clone();
    stale_phase.transition_moments[(0, 0, 0, 0)] += Complex64::new(0.125, -0.25);
    write_phase_bin(&phase_path, &stale_phase)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let actual_phase = read_phase_bin(&phase_path)?;
    assert_phase_cache_sentinel_preserved(&actual_phase, &expected_phase);
    assert_phase_transition_moments_close(&actual_phase, &expected_phase, 1.0e-8);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_xsect_cache_from_phase_and_pot_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;
    std::fs::write(temp.path().join("xsect.dat"), "not xsect.dat\n")?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    Ok(())
}

#[test]
fn xsph_module_bootstraps_hubbard_active_normal_potential_without_v_hubbard() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(temp.path(), 2)?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 4);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("xsect.dat").is_file());
    assert!(!temp.path().join("aphase_hubbard.bin").exists());
    Ok(())
}

#[test]
fn xsph_module_generates_active_hubbard_phase_handoff_when_xsect_branch_is_unsupported()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 1, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(temp.path(), 2)?;
    write_v_hubbard_bin(temp.path().join("v_hubbard.bin"), &sample_v_hubbard_bin(1))?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let aphase = read_aphase_hubbard_bin_inferred(
        temp.path().join("aphase_hubbard.bin"),
        phase.energy_count,
        phase.potential_count(),
    )?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(aphase.potential_count(), phase.potential_count());
    assert_eq!(aphase.energy_count(), phase.energy_count);
    assert_eq!(aphase.angular_limit, 1);
    assert!(
        aphase
            .values
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_active_hubbard_cached_outputs_from_v_hubbard() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(temp.path(), 2)?;
    write_v_hubbard_bin(temp.path().join("v_hubbard.bin"), &sample_v_hubbard_bin(1))?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(temp.path().join("xsect.dat").is_file());
    assert!(
        read_aphase_hubbard_bin_inferred(
            temp.path().join("aphase_hubbard.bin"),
            phase.energy_count,
            phase.potential_count(),
        )?
        .values
        .iter()
        .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_generates_missing_active_hubbard_aphase_from_cached_base_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_source_handoff_inputs(temp.path())?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let aphase_path = temp.path().join("aphase_hubbard.bin");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_aphase = read_aphase_hubbard_bin_inferred(
        &aphase_path,
        expected_phase.energy_count,
        expected_phase.potential_count(),
    )?;
    std::fs::remove_file(&aphase_path)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    assert_eq!(
        read_aphase_hubbard_bin_inferred(
            &aphase_path,
            expected_phase.energy_count,
            expected_phase.potential_count(),
        )?,
        expected_aphase
    );
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_active_hubbard_aphase_from_cached_base_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_source_handoff_inputs(temp.path())?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let aphase_path = temp.path().join("aphase_hubbard.bin");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_aphase = read_aphase_hubbard_bin_inferred(
        &aphase_path,
        expected_phase.energy_count,
        expected_phase.potential_count(),
    )?;
    std::fs::write(&aphase_path, "not aphase_hubbard.bin\n")?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    assert_eq!(
        read_aphase_hubbard_bin_inferred(
            &aphase_path,
            expected_phase.energy_count,
            expected_phase.potential_count(),
        )?,
        expected_aphase
    );
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_active_hubbard_aphase_from_cached_base_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_active_hubbard_source_handoff_inputs(temp.path())?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let aphase_path = temp.path().join("aphase_hubbard.bin");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_aphase = read_aphase_hubbard_bin_inferred(
        &aphase_path,
        expected_phase.energy_count,
        expected_phase.potential_count(),
    )?;
    let mut stale_aphase = expected_aphase.clone();
    stale_aphase.values[(0, 0, 0, 0, 0)] += Complex64::new(0.25, -0.125);
    write_aphase_hubbard_bin(&aphase_path, &stale_aphase)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    assert_eq!(
        read_aphase_hubbard_bin_inferred(
            &aphase_path,
            expected_phase.energy_count,
            expected_phase.potential_count(),
        )?,
        expected_aphase
    );
    Ok(())
}

#[test]
fn xsph_module_accepts_active_hubbard_bootstrap_base_cache_without_v_hubbard() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_hubbard_input(temp.path(), 2)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    assert!(run_in_dir(temp.path())? >= 2);
    assert!(!temp.path().join("aphase_hubbard.bin").exists());
    Ok(())
}

#[test]
fn xsph_module_rejects_malformed_active_hubbard_v_source_without_ordinary_fallback() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(temp.path(), 2)?;
    std::fs::write(
        temp.path().join("v_hubbard.bin"),
        b"not a Hubbard V source\n",
    )?;

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed active Hubbard V source must fail closed")?;
    assert!(format!("{error:#}").contains("v_hubbard.bin"), "{error:?}");
    assert!(!temp.path().join("phase.bin").exists());
    Ok(())
}

#[test]
fn xsph_module_allows_ordinary_hubbard_input_on_normal_potential_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(temp.path(), 1)?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_negative_izstd_source_xsect_from_pot_and_config() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = -1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_generates_positive_izstd_source_xsect_from_pot_and_config() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("xsect.dat").is_file());
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_generates_positive_izstd_xsect_when_pmbse_is_ignored_like_feff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.advanced.ifxc = 0;
        input.advanced.ipmbse = 3;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 2;
        input.advanced.ibasis = 6;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_generates_positive_izstd_e2_xsect_from_pot_and_config() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 2, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        (3..phase.transition_count).any(|transition| {
            (0..phase.energy_count)
                .any(|energy| phase.transition_moments[(energy, 0, transition, 0)].norm() > 0.0)
        }),
        "expected positive-izstd E2 transition slots to be populated"
    );
    Ok(())
}

#[test]
fn xsph_module_rejects_feff_nonrelativistic_m1_positive_izstd() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 1, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert!(written > 0);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_pmbse_phase_handoff_without_xmu_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    assert!(has_supported_phase_mesh_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert_eq!(written, 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.fermi_index, 0);
    assert_eq!(phase.energy_count, 120);
    assert_eq!(phase.main_energy_count, 120);
    assert_eq!(phase.auxiliary_energy_count, 0);
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.fermi_index, 0);
    assert_eq!(emesh.point_count(), 120);
    assert_eq!(emesh_bin.point_count(), 120);
    assert_eq!(emesh_bin.horizontal_count, 120);
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert!((emesh.energy_ev[0] + 20.0).abs() < 5.0e-5);
    assert!((emesh.energy_ev[99] - 200.0).abs() < 5.0e-5);
    assert!((emesh.energy_ev[118] - 450.0).abs() < 5.0e-5);
    assert!((emesh.energy_ev[119] - (200.0 + 20.0 * (450.0 - 200.0) / 19.0)).abs() < 5.0e-5);
    assert!(!temp.path().join("xsect.dat").is_file());

    let error = run_required_in_dir(temp.path())
        .err()
        .context("PMBSE xsectd branch without xmu sources should stay phase-only")?;
    assert!(
        error
            .to_string()
            .contains("supported TDLDA xsedge.dat source handoff"),
        "{error:?}"
    );
    assert!(!temp.path().join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn xsph_tdlda_required_stage_rejects_stale_ordinary_xsect_cache_without_xsedge() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("TDLDA xsectd branch should not be completed by stale ordinary xsect.dat")?;
    assert!(
        error
            .to_string()
            .contains("supported TDLDA xsedge.dat source handoff"),
        "{error:?}"
    );
    assert!(!temp.path().join("xsedge.dat").is_file());
    Ok(())
}

#[test]
fn xsph_tdlda_rejects_stale_cached_xsedge_shape_when_source_shape_is_known() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(run_supported_phase_handoff_in_dir(temp.path())? > 0);
    write_split_pmbse_xmu_sources(temp.path())?;
    std::fs::write(temp.path().join("xsedge.dat"), "0.0 1.0 2.0\n")?;

    let caches = super::XsphCachePaths::new(temp.path());
    let input = super::read_input(temp.path())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(!super::can_use_tdlda_cached_xsedge_output(
        &caches, &input, &phase
    )?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);

    assert!(run_in_dir(temp.path())? > 0);
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    Ok(())
}

#[test]
fn xsph_tdlda_rejects_stale_cached_xsedge_energy_when_source_grid_is_known() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(run_supported_phase_handoff_in_dir(temp.path())? > 0);
    write_split_pmbse_xmu_sources(temp.path())?;
    std::fs::write(
        temp.path().join("xsedge.dat"),
        "\
  9.00000  1 2 3 4 5 6
 10.00000  1 2 3 4 5 6
 11.00000  1 2 3 4 5 6
 12.00000  1 2 3 4 5 6
 13.00000  1 2 3 4 5 6
 14.00000  1 2 3 4 5 6
",
    )?;

    let caches = super::XsphCachePaths::new(temp.path());
    let input = super::read_input(temp.path())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(!super::can_use_tdlda_cached_xsedge_output(
        &caches, &input, &phase
    )?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);

    assert!(run_in_dir(temp.path())? > 0);
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!((xsedge.energy_ev[0] - 0.0).abs() < 5.0e-5);
    assert!((xsedge.energy_ev[5] - 3.5).abs() < 5.0e-5);
    Ok(())
}

#[test]
fn xsph_tdlda_rejects_cached_xsedge_when_declared_pmbse_source_is_malformed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(run_supported_phase_handoff_in_dir(temp.path())? > 0);
    std::fs::write(temp.path().join("listedges.pmbse"), "Oddp1\n")?;
    std::fs::write(
        temp.path().join("xsedge.dat"),
        "\
  0.00000  1 2 3 4 5 6
  1.00000  1 2 3 4 5 6
  2.00000  1 2 3 4 5 6
  2.50000  1 2 3 4 5 6
  3.00000  1 2 3 4 5 6
  3.50000  1 2 3 4 5 6
",
    )?;

    let caches = super::XsphCachePaths::new(temp.path());
    let input = super::read_input(temp.path())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(!super::can_use_tdlda_cached_xsedge_output(
        &caches, &input, &phase
    )?);

    let error = has_supported_tdlda_xsedge_output(temp.path())
        .err()
        .context("declared malformed PMBSE source should not advertise cached xsedge")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("PMBSE"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_writes_tdlda_xsedge_from_pmbse_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written > 0);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("xsedge.dat").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);

    assert!(run_required_in_dir(temp.path())? > 0);
    Ok(())
}

#[test]
fn xsph_module_writes_tdlda_xsedge_from_nonlocal_pot_ch_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_tdlda_no_cache_source_case(temp.path(), 1, false)?;
    let mut screened = sample_normal_phase_pot_bin();
    screened.total_potential.mapv_inplace(|value| value + 0.075);
    write_pot_bin(temp.path().join("pot.ch"), &screened)?;

    assert!(!temp.path().join("phase.bin").is_file());
    assert!(!temp.path().join("xsedge.dat").is_file());
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    assert!(run_in_dir(temp.path())? > 0);

    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn xsph_module_writes_tdlda_xsedge_from_nonlocal_yoshi_or_wscrn_source() -> Result<()> {
    for file_name in ["yoshi.dat", "wscrn.dat"] {
        let temp = tempfile::tempdir()?;
        write_tdlda_no_cache_source_case(temp.path(), 2, false)?;
        write_tdlda_screened_potential_source(temp.path().join(file_name))?;

        assert!(!temp.path().join("phase.bin").is_file());
        assert!(!temp.path().join("xsedge.dat").is_file());
        assert!(has_supported_tdlda_xsedge_output(temp.path())?);
        assert!(run_in_dir(temp.path())? > 0);

        let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
        assert_eq!(xsedge.row_count(), 6, "source {file_name}");
        assert!(
            xsedge.total_screened.iter().all(|value| value.is_finite()),
            "source {file_name}"
        );
    }
    Ok(())
}

#[test]
fn xsph_module_writes_two_spin_tdlda_xsedge_from_sources_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_tdlda_no_cache_source_case(temp.path(), 0, true)?;

    assert!(!temp.path().join("phase.bin").is_file());
    assert!(!temp.path().join("xsedge.dat").is_file());
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    assert!(run_in_dir(temp.path())? > 0);

    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_tdlda_spin_merge_averages_matching_source_outputs() -> Result<()> {
    let first = XsedgeDatData {
        energy_ev: Array1::from_vec(vec![1.0, 2.0]),
        total_single_particle: Array1::from_vec(vec![3.0, 4.0]),
        total_screened: Array1::from_vec(vec![5.0, 6.0]),
        plus_branch_single_particle: None,
        minus_branch_single_particle: None,
        plus_branch_screened: None,
        minus_branch_screened: None,
    };
    let mut second = first.clone();
    second
        .total_single_particle
        .mapv_inplace(|value| value + 2.0);
    second.total_screened.mapv_inplace(|value| value + 4.0);

    let merged = super::tdlda_merge_spin_xsedge_outputs(vec![first.clone(), second.clone()])?;
    assert_eq!(merged.energy_ev, first.energy_ev);
    for row in 0..merged.row_count() {
        assert_eq!(
            merged.total_single_particle[row],
            (first.total_single_particle[row] + second.total_single_particle[row]) / 2.0
        );
        assert_eq!(
            merged.total_screened[row],
            (first.total_screened[row] + second.total_screened[row]) / 2.0
        );
    }
    Ok(())
}

#[test]
fn xsph_module_ignores_malformed_ordinary_xsect_for_tdlda_xsedge_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;
    std::fs::write(
        temp.path().join("xsect.dat"),
        "not an ordinary xsect.dat cache\n",
    )?;

    assert!(run_in_dir(temp.path())? > 0);

    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("xsedge.dat").is_file());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("xsect.dat"))?,
        "not an ordinary xsect.dat cache\n"
    );
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_ignores_readable_stale_ordinary_xsect_for_tdlda_xsedge_source_handoff() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let stale_xsect = sample_xsect_dat();
    write_xsect_dat(temp.path().join("xsect.dat"), &stale_xsect)?;

    assert!(run_in_dir(temp.path())? > 0);

    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("xsedge.dat").is_file());
    assert_xsect_table_preserved(
        &read_xsect_dat(temp.path().join("xsect.dat"))?,
        &stale_xsect,
    );
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_tdlda_xsedge_from_pmbse_source_handoffs() -> Result<()> {
    assert_xsph_module_regenerates_stale_tdlda_xsedge_from_pmbse_source_handoffs(0, false)
}

#[test]
fn xsph_module_regenerates_stale_file_basis_tdlda_xsedge_from_pmbse_source_handoffs() -> Result<()>
{
    assert_xsph_module_regenerates_stale_tdlda_xsedge_from_pmbse_source_handoffs(1, true)
}

#[test]
fn xsph_module_regenerates_stale_generated_basis_tdlda_xsedge_from_pmbse_source_handoffs()
-> Result<()> {
    assert_xsph_module_regenerates_stale_tdlda_xsedge_from_pmbse_source_handoffs(2, false)
}

fn assert_xsph_module_regenerates_stale_tdlda_xsedge_from_pmbse_source_handoffs(
    ibasis: i32,
    write_file_basis_orbitals: bool,
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = ibasis;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;
    if write_file_basis_orbitals {
        write_tdlda_file_basis_orbitals(temp.path())?;
    }

    assert!(run_in_dir(temp.path())? > 0);
    let expected = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    let mut stale = expected.clone();
    stale.total_single_particle[0] += 25.0;
    stale.total_screened[0] += 12.5;
    if let Some(plus) = stale.plus_branch_single_particle.as_mut() {
        plus[0] += 7.0;
    }
    write_xsedge_dat(temp.path().join("xsedge.dat"), &stale)?;

    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    assert_ne!(read_xsedge_dat(temp.path().join("xsedge.dat"))?, expected);

    let count = run_in_dir(temp.path())?;

    assert!(count > 0);
    assert_eq!(read_xsedge_dat(temp.path().join("xsedge.dat"))?, expected);
    assert!(!temp.path().join("xsect.dat").is_file());
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_keeps_file_basis_pmbse_projectors_guarded() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written > 0);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(!temp.path().join("xsedge.dat").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_tdlda_xsedge_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("file-basis TDLDA xsedge bridge should stay guarded")?;
    assert!(
        error
            .to_string()
            .contains("supported TDLDA xsedge.dat source handoff"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn xsph_module_writes_tdlda_xsedge_from_file_basis_pmbse_projectors() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;
    write_tdlda_file_basis_orbitals(temp.path())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written > 0);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("xsedge.dat").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);

    assert!(run_required_in_dir(temp.path())? > 0);
    Ok(())
}

#[test]
fn xsph_reads_file_basis_pmbse_projectors_from_vila_orbs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("file-basis PMBSE source handoff should produce a TDLDA plan")?;
    let generated_basis_count = plan
        .basis
        .rows
        .iter()
        .filter_map(|row| {
            match xsph_tdlda_decode_projector_selector(row.projector_orbital_selector) {
                Ok(XsphTdldaProjectorSelector::GeneratedBasis { basis_index, .. }) => {
                    Some(basis_index + 1)
                }
                _ => None,
            }
        })
        .max()
        .unwrap_or(0);
    let radii = Array1::from_vec(
        (1..=8)
            .map(|index| 0.05 * index as f64 / FEFF_BOHR_ANGSTROM)
            .collect(),
    );

    assert!(
        tdlda_file_projector_candidates_from_source(XsphTdldaFileProjectorCandidatesInput {
            work_dir: temp.path(),
            plan: &plan,
            generated_basis_count,
            active_len: radii.len(),
            file_target_last_index: radii.len() - 1,
            radii: radii.view(),
        })?
        .is_none()
    );

    write_tdlda_file_basis_orbitals(temp.path())?;
    let (large, small) =
        tdlda_file_projector_candidates_from_source(XsphTdldaFileProjectorCandidatesInput {
            work_dir: temp.path(),
            plan: &plan,
            generated_basis_count,
            active_len: radii.len(),
            file_target_last_index: radii.len() - 1,
            radii: radii.view(),
        })?
        .context("Vila/Orbs file basis should produce projector candidates")?;

    assert_eq!(large.dim(), (radii.len(), generated_basis_count, 2));
    assert_eq!(small.dim(), large.dim());
    assert!(small.iter().all(|value| *value == 0.0));
    assert!((large[(0, 0, 0)] - 0.10).abs() < 1.0e-12);
    assert!((large[(0, 0, 1)] - 0.10).abs() < 1.0e-12);
    assert!((large[(0, 1, 0)] - 0.15).abs() < 1.0e-12);
    assert!((large[(0, 2, 1)] - 0.15).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn xsph_module_writes_tdlda_xsedge_from_calculated_pmbse_projectors() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = 0;
        input.advanced.ibasis = 2;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(temp.path())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written > 0);
    assert!(temp.path().join("phase.bin").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("xsedge.dat").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    let xsedge = read_xsedge_dat(temp.path().join("xsedge.dat"))?;
    assert_eq!(xsedge.row_count(), 6);
    assert!(xsedge.has_branch_columns());
    assert!(
        xsedge
            .total_single_particle
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(xsedge.total_screened.iter().all(|value| value.is_finite()));
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_tdlda_xsedge_output(temp.path())?);

    assert!(run_required_in_dir(temp.path())? > 0);
    Ok(())
}

#[test]
fn xsph_reads_pmbse_listedges_channel_multipliers_from_xmu_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;

    let multipliers = tdlda_pmbse_channel_multipliers_from_source(temp.path(), -2, 10)?
        .context("PMBSE listedges.pmbse source should produce channel multipliers")?;

    let expected_energy_ev = [0.0, 1.0, 2.0, 2.5, 3.0, 3.5];
    assert_eq!(multipliers.energy_hartree.len(), expected_energy_ev.len());
    for (actual, expected_ev) in multipliers
        .energy_hartree
        .iter()
        .zip(expected_energy_ev.iter().copied())
    {
        assert!((actual - expected_ev / FEFF_HARTREE_EV).abs() < 1.0e-14);
    }
    assert!((multipliers.spin_orbit_split - 2.0 / FEFF_HARTREE_EV).abs() < 1.0e-14);

    let expected = Array2::from_shape_vec(
        (6, 4),
        vec![
            2.0, 10.0, 6.0, 50.0, 2.0, 10.0, 6.0, 50.0, 3.0, 10.0, 7.0, 50.0, 3.5, 20.0, 7.5, 60.0,
            4.0, 30.0, 8.0, 70.0, 1.0, 40.0, 1.0, 80.0,
        ],
    )?;
    assert_eq!(multipliers.channel_multipliers.dim(), expected.dim());
    for row in 0..expected.nrows() {
        for channel in 0..expected.ncols() {
            assert!(
                (multipliers.channel_multipliers[(row, channel)] - expected[(row, channel)]).abs()
                    < 2.0e-12
            );
        }
    }

    let empty = tempfile::tempdir()?;
    assert!(tdlda_pmbse_channel_multipliers_from_source(empty.path(), -2, 10)?.is_none());
    Ok(())
}

#[test]
fn xsph_plans_tdlda_xsectd_channels_from_pmbse_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 0;
    let pot = sample_normal_phase_pot_bin();
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;

    let k_plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("K-edge PMBSE source handoff should produce a TDLDA plan")?;

    assert_eq!(k_plan.initial_kappa, -1);
    assert_eq!(k_plan.initial_l, 0);
    assert_eq!(k_plan.channel_count, 1);
    assert_eq!(k_plan.plus_basis_count, 1);
    assert_eq!(k_plan.minus_basis_count, 0);
    assert_eq!(k_plan.matrix_size, 3);
    assert_eq!(k_plan.primary_channel_count, 3);
    assert_eq!(k_plan.spin_orbit_split, 0.0);
    assert!(k_plan.reference_shifts.iter().all(|value| *value == 0.0));
    let k_width = core_hole_width_ev(29, 1)? / FEFF_HARTREE_EV / 2.0;
    for width in k_plan.row_broadenings.iter().copied() {
        assert!((width - k_width).abs() < 1.0e-14);
    }

    let mut l3_pot = pot.clone();
    l3_pot.ihole = 4;
    let l3_plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &l3_pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a split-edge TDLDA plan")?;

    assert_eq!(l3_plan.initial_kappa, -2);
    assert_eq!(l3_plan.initial_l, 1);
    assert_eq!(l3_plan.channel_count, 2);
    assert_eq!(l3_plan.plus_basis_count, 1);
    assert_eq!(l3_plan.minus_basis_count, 0);
    assert_eq!(l3_plan.matrix_size, 9);
    assert_eq!(l3_plan.primary_channel_count, 9);
    assert!((l3_plan.spin_orbit_split - 2.0 / FEFF_HARTREE_EV).abs() < 1.0e-14);
    assert!(l3_plan.reference_shifts.iter().any(|value| *value < 0.0));
    assert!(l3_plan.reference_shifts.iter().any(|value| *value == 0.0));
    let l3_width = core_hole_width_ev(29, 4)? / FEFF_HARTREE_EV / 2.0;
    let l2_width = core_hole_width_ev(29, 3)? / FEFF_HARTREE_EV / 2.0;
    for (basis_row, width) in l3_plan
        .basis
        .rows
        .iter()
        .zip(l3_plan.row_broadenings.iter().copied())
    {
        let expected = if basis_row.initial_kappa > 0 {
            l2_width
        } else {
            l3_width
        };
        assert!((width - expected).abs() < 1.0e-14);
    }

    let reference_energy = Array1::from_elem(
        l3_plan.multipliers.energy_hartree.len(),
        Complex64::new(0.0, 0.0),
    );
    let energy_rows =
        tdlda_energy_rows_from_source_plan(&l3_plan, &input, reference_energy.view(), 0.0, 0.0)?;
    assert_eq!(
        energy_rows.photon_energy.len(),
        l3_plan.multipliers.energy_hartree.len()
    );
    assert!(
        energy_rows
            .separation_function
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(energy_rows.active_rows.iter().all(|value| *value));
    assert!((energy_rows.photon_energy[0] - 0.1 / FEFF_HARTREE_EV).abs() < 1.0e-14);
    assert!((energy_rows.photon_energy[3] - l3_plan.multipliers.energy_hartree[3]).abs() < 1.0e-14);
    assert!(energy_rows.plus_wave_number[3] > 0.0);
    assert!(energy_rows.minus_wave_number[0].abs() < 1.0e-14);

    let raw_response = Array3::from_shape_fn(
        (
            l3_plan.multipliers.energy_hartree.len(),
            l3_plan.matrix_size,
            l3_plan.matrix_size,
        ),
        |(energy, row, column)| 100.0 * energy as f64 + 10.0 * row as f64 + column as f64 + 1.0,
    );
    let weighted = tdlda_weight_response_from_source_plan(&l3_plan, raw_response.view())?;
    assert_eq!(
        weighted.imaginary_response.dim(),
        (
            l3_plan.multipliers.energy_hartree.len(),
            l3_plan.matrix_size,
            l3_plan.matrix_size
        )
    );
    for (row_index, basis_row) in l3_plan.basis.rows.iter().enumerate() {
        let expected_channel = match (
            basis_row.initial_kappa > 0,
            (basis_row.final_kappa + 1).abs() > (basis_row.initial_kappa + 1).abs(),
        ) {
            (true, true) => 1,
            (true, false) => 3,
            (false, true) => 0,
            (false, false) => 2,
        };
        assert_eq!(weighted.row_channels[row_index], expected_channel);
        for energy in 0..l3_plan.multipliers.energy_hartree.len() {
            let multiplier = l3_plan.multipliers.channel_multipliers[(energy, expected_channel)];
            for column in 0..l3_plan.matrix_size {
                assert!(
                    (weighted.imaginary_response[(energy, row_index, column)]
                        - raw_response[(energy, row_index, column)] * multiplier)
                        .abs()
                        < 1.0e-14
                );
            }
        }
    }
    Ok(())
}

#[test]
fn xsph_assembles_tdlda_ibasis_zero_projectors_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 0;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut valence_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    valence_occupations[8] = 1.0;
    let projector_config = ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![ConfigDatPotential {
            potential_index: 0,
            atomic_number: 29,
            element: "Cu".to_string(),
            occupations,
            valence_occupations,
            spin_occupations: None,
        }],
    };
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&projector_config)?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;

    assert!(
        plan.basis
            .rows
            .iter()
            .any(|row| row.projector_orbital_selector > 0)
    );
    assert!(
        plan.basis
            .rows
            .iter()
            .any(|row| row.projector_orbital_selector < 0)
    );
    let active_len = 220;
    let step = 0.05;
    let radii = Array1::from_shape_fn(active_len, |row| (-8.8 + step * row as f64).exp());
    let bound_large_components = pot.large_components.index_axis(Axis(2), 0);
    let bound_small_components = pot.small_components.index_axis(Axis(2), 0);

    let projectors = tdlda_projector_rows_from_source_plan(
        &plan,
        active_len,
        step,
        pot.norman_radii[0],
        radii.view(),
        bound_large_components,
        bound_small_components,
        None,
        None,
    )?;

    assert_eq!(
        projectors.localized_large.dim(),
        (active_len, plan.matrix_size)
    );
    assert_eq!(
        projectors.localized_small.dim(),
        (active_len, plan.matrix_size)
    );
    assert_eq!(projectors.source_rows.len(), plan.matrix_size);
    assert_eq!(projectors.generated_rows.len(), plan.matrix_size);
    assert_eq!(projectors.selector_indices.len(), plan.matrix_size);
    let radius_values = radii.iter().copied().collect::<Vec<_>>();
    for (row_index, basis_row) in plan.basis.rows.iter().enumerate() {
        let decoded = xsph_tdlda_decode_projector_selector(basis_row.projector_orbital_selector)?;
        if basis_row.projector_orbital_selector < 0 {
            assert!(matches!(
                decoded,
                XsphTdldaProjectorSelector::GeneratedBasis { .. }
            ));
            assert!(!projectors.source_rows[row_index]);
            assert!(!projectors.generated_rows[row_index]);
            assert_eq!(projectors.norm_integrals[row_index], 0.0);
            assert_eq!(projectors.norm_sqrt[row_index], 0.0);
            assert!(
                projectors
                    .localized_large
                    .index_axis(Axis(1), row_index)
                    .iter()
                    .all(|value| *value == 0.0)
            );
            assert!(
                projectors
                    .localized_small
                    .index_axis(Axis(1), row_index)
                    .iter()
                    .all(|value| *value == 0.0)
            );
            continue;
        }

        assert!(projectors.source_rows[row_index]);
        assert!(!projectors.generated_rows[row_index]);
        let XsphTdldaProjectorSelector::OccupiedOrbital { orbital_index } = decoded else {
            bail!("positive TDLDA selector did not decode as occupied orbital");
        };
        assert_eq!(projectors.selector_indices[row_index], orbital_index);
        assert!(projectors.norm_integrals[row_index].is_finite());
        assert!(projectors.norm_integrals[row_index] > 0.0);
        assert!(projectors.norm_sqrt[row_index].is_finite());
        assert!(projectors.norm_sqrt[row_index] > 0.0);

        let large_column = projectors.localized_large.index_axis(Axis(1), row_index);
        let small_column = projectors.localized_small.index_axis(Axis(1), row_index);
        assert!(
            large_column
                .iter()
                .chain(small_column.iter())
                .all(|value| value.is_finite())
        );
        assert!(
            large_column.iter().any(|value| value.abs() > 0.0)
                || small_column.iter().any(|value| value.abs() > 0.0)
        );

        let final_l = if basis_row.final_kappa > 0 {
            basis_row.final_kappa
        } else {
            basis_row.final_kappa.abs() - 1
        } as f64;
        let samples = (0..active_len)
            .map(|radial| large_column[radial].powi(2) + small_column[radial].powi(2))
            .collect::<Vec<_>>();
        let normalized = somm2(
            &radius_values,
            &samples,
            step,
            2.0 * final_l + 2.0,
            pot.norman_radii[0],
            0,
        )?;
        assert!(
            (normalized - 1.0).abs() < 1.0e-10,
            "projector row {row_index} normalized to {normalized}"
        );
    }

    let mut duplicate_selector_checked = false;
    for left in 0..plan.matrix_size {
        for right in (left + 1)..plan.matrix_size {
            if !projectors.source_rows[left] || !projectors.source_rows[right] {
                continue;
            }
            if projectors.selector_indices[left] != projectors.selector_indices[right] {
                continue;
            }
            duplicate_selector_checked = true;
            for radial in 0..active_len {
                assert!(
                    (projectors.localized_large[(radial, left)]
                        - projectors.localized_large[(radial, right)])
                        .abs()
                        < 1.0e-14
                );
                assert!(
                    (projectors.localized_small[(radial, left)]
                        - projectors.localized_small[(radial, right)])
                        .abs()
                        < 1.0e-14
                );
            }
        }
    }
    assert!(duplicate_selector_checked);
    Ok(())
}

#[test]
fn xsph_assembles_tdlda_generated_basis_projectors_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a generated-basis TDLDA plan")?;

    let mut generated_basis_count = 0;
    for basis_row in &plan.basis.rows {
        if let XsphTdldaProjectorSelector::GeneratedBasis { basis_index, .. } =
            xsph_tdlda_decode_projector_selector(basis_row.projector_orbital_selector)?
        {
            generated_basis_count = generated_basis_count.max(basis_index + 1);
        }
    }
    assert!(generated_basis_count >= 2);

    let active_len = 18;
    let step = 0.08;
    let radii = Array1::from_shape_fn(active_len, |row| 0.35 * (step * row as f64).exp());
    let norman_radius = radii[active_len - 1];
    let generated_large = Array3::from_shape_fn(
        (active_len, generated_basis_count, 2),
        |(radial, basis, partner)| {
            let r = (radial + 1) as f64;
            let b = (basis + 1) as f64;
            let p = (partner + 1) as f64;
            0.20 + 0.018 * b + 0.006 * p + 0.035 * (0.13 * r * (b + p)).sin()
        },
    );
    let generated_small = Array3::from_shape_fn(
        (active_len, generated_basis_count, 2),
        |(radial, basis, partner)| {
            let r = (radial + 1) as f64;
            let b = (basis + 1) as f64;
            let p = (partner + 1) as f64;
            0.05 + 0.010 * b - 0.004 * p + 0.018 * (0.11 * r * (b + 2.0 * p)).cos()
        },
    );
    let bound_large_components = pot.large_components.index_axis(Axis(2), 0);
    let bound_small_components = pot.small_components.index_axis(Axis(2), 0);

    let projectors = tdlda_projector_rows_from_source_plan(
        &plan,
        active_len,
        step,
        norman_radius,
        radii.view(),
        bound_large_components,
        bound_small_components,
        Some(generated_large.view()),
        Some(generated_small.view()),
    )?;

    assert!(projectors.source_rows.iter().all(|value| *value));
    assert!(projectors.generated_rows.iter().all(|value| *value));
    assert!(projectors.norm_sqrt.iter().all(|value| *value > 0.0));

    let radius_values = radii.iter().copied().collect::<Vec<_>>();
    for (row_index, basis_row) in plan.basis.rows.iter().enumerate() {
        let XsphTdldaProjectorSelector::GeneratedBasis { basis_index, .. } =
            xsph_tdlda_decode_projector_selector(basis_row.projector_orbital_selector)?
        else {
            bail!("ibasis=1 TDLDA plan should use generated projectors");
        };
        assert_eq!(projectors.selector_indices[row_index], basis_index);
        let large_column = projectors.localized_large.index_axis(Axis(1), row_index);
        let small_column = projectors.localized_small.index_axis(Axis(1), row_index);
        let final_l = if basis_row.final_kappa > 0 {
            basis_row.final_kappa
        } else {
            basis_row.final_kappa.abs() - 1
        } as f64;
        let samples = (0..active_len)
            .map(|radial| large_column[radial].powi(2) + small_column[radial].powi(2))
            .collect::<Vec<_>>();
        let normalized = somm2(
            &radius_values,
            &samples,
            step,
            2.0 * final_l + 2.0,
            norman_radius,
            0,
        )?;
        assert!(
            (normalized - 1.0).abs() < 1.0e-10,
            "generated projector row {row_index} normalized to {normalized}"
        );
    }

    let mut duplicate_generated_checked = false;
    for left in 0..plan.matrix_size {
        for right in (left + 1)..plan.matrix_size {
            if plan.basis.rows[left].projector_orbital_selector
                != plan.basis.rows[right].projector_orbital_selector
            {
                continue;
            }
            duplicate_generated_checked = true;
            for radial in 0..active_len {
                assert!(
                    (projectors.localized_large[(radial, left)]
                        - projectors.localized_large[(radial, right)])
                        .abs()
                        < 1.0e-14
                );
                assert!(
                    (projectors.localized_small[(radial, left)]
                        - projectors.localized_small[(radial, right)])
                        .abs()
                        < 1.0e-14
                );
            }
        }
    }
    assert!(duplicate_generated_checked);

    for positive_final_kappa in [false, true] {
        let mut row0 = None;
        let mut row1 = None;
        let mut final_l = None;
        for (row_index, basis_row) in plan.basis.rows.iter().enumerate() {
            if let XsphTdldaProjectorSelector::GeneratedBasis {
                basis_index,
                positive_final_kappa: row_positive,
            } = xsph_tdlda_decode_projector_selector(basis_row.projector_orbital_selector)?
                && row_positive == positive_final_kappa
            {
                if basis_index == 0 {
                    row0.get_or_insert(row_index);
                    final_l.get_or_insert(if basis_row.final_kappa > 0 {
                        basis_row.final_kappa
                    } else {
                        basis_row.final_kappa.abs() - 1
                    } as f64);
                } else if basis_index == 1 {
                    row1.get_or_insert(row_index);
                }
            }
        }
        let (Some(row0), Some(row1), Some(final_l)) = (row0, row1, final_l) else {
            continue;
        };
        let overlap_samples = (0..active_len)
            .map(|radial| {
                projectors.localized_large[(radial, row0)]
                    * projectors.localized_large[(radial, row1)]
                    + projectors.localized_small[(radial, row0)]
                        * projectors.localized_small[(radial, row1)]
            })
            .collect::<Vec<_>>();
        let overlap = somm2(
            &radius_values,
            &overlap_samples,
            step,
            2.0 * final_l + 2.0,
            norman_radius,
            0,
        )?;
        assert!(
            overlap.abs() < 1.0e-10,
            "generated projector partner {positive_final_kappa} overlap was {overlap}"
        );
    }

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_raw_response_inputs_from_source_projectors() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a generated-basis TDLDA plan")?;
    let generated_basis_count = plan
        .basis
        .rows
        .iter()
        .filter_map(|row| {
            match xsph_tdlda_decode_projector_selector(row.projector_orbital_selector).ok()? {
                XsphTdldaProjectorSelector::GeneratedBasis { basis_index, .. } => {
                    Some(basis_index + 1)
                }
                XsphTdldaProjectorSelector::OccupiedOrbital { .. } => None,
            }
        })
        .max()
        .context("generated-basis TDLDA plan should have generated selectors")?;

    let active_len = 18;
    let step = 0.08;
    let radii = Array1::from_shape_fn(active_len, |row| 0.35 * (step * row as f64).exp());
    let generated_large = Array3::from_shape_fn(
        (active_len, generated_basis_count, 2),
        |(radial, basis, partner)| {
            let r = (radial + 1) as f64;
            let b = (basis + 1) as f64;
            let p = (partner + 1) as f64;
            0.18 + 0.014 * b + 0.005 * p + 0.028 * (0.15 * r * (b + p)).sin()
        },
    );
    let generated_small = Array3::from_shape_fn(
        (active_len, generated_basis_count, 2),
        |(radial, basis, partner)| {
            let r = (radial + 1) as f64;
            let b = (basis + 1) as f64;
            let p = (partner + 1) as f64;
            0.04 + 0.008 * b - 0.003 * p + 0.015 * (0.09 * r * (b + 2.0 * p)).cos()
        },
    );
    let bound_large_components = pot.large_components.index_axis(Axis(2), 0);
    let bound_small_components = pot.small_components.index_axis(Axis(2), 0);
    let projectors = tdlda_projector_rows_from_source_plan(
        &plan,
        active_len,
        step,
        radii[active_len - 1],
        radii.view(),
        bound_large_components,
        bound_small_components,
        Some(generated_large.view()),
        Some(generated_small.view()),
    )?;
    let full_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        projectors.localized_large[(radial, row)] + 0.012 + 0.0005 * row as f64
    });
    let full_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        projectors.localized_small[(radial, row)] - 0.006 + 0.0003 * radial as f64
    });
    let initial_large = Array1::from_shape_fn(active_len, |radial| {
        0.22 + 0.009 * radial as f64 + 0.02 * (0.12 * radial as f64).sin()
    });
    let initial_small = Array1::from_shape_fn(active_len, |radial| {
        0.05 + 0.004 * radial as f64 + 0.01 * (0.10 * radial as f64).cos()
    });
    let xray_bessel = Array2::from_shape_fn((3, active_len), |(row, radial)| {
        0.18 + 0.025 * row as f64 + 0.002 * radial as f64
    });

    let inputs = tdlda_raw_response_inputs_from_source_plan(
        &plan,
        active_len,
        step,
        radii.view(),
        initial_large.view(),
        initial_small.view(),
        xray_bessel.view(),
        projectors.localized_large.view(),
        projectors.localized_small.view(),
        full_large.view(),
        full_small.view(),
    )?;

    assert_eq!(inputs.overlaps.len(), plan.matrix_size);
    assert_eq!(inputs.localized_dipoles.len(), plan.matrix_size);
    assert_eq!(inputs.full_dipoles.len(), plan.matrix_size);
    assert!(inputs.overlaps.iter().all(|value| value.is_finite()));
    assert!(
        inputs
            .localized_dipoles
            .iter()
            .chain(inputs.full_dipoles.iter())
            .all(|value| value.is_finite())
    );

    let row_index = 0;
    let row = &plan.basis.rows[row_index];
    let final_l = if row.final_kappa > 0 {
        row.final_kappa
    } else {
        row.final_kappa.abs() - 1
    } as f64;
    let radius_values = radii.iter().copied().collect::<Vec<_>>();
    let overlap_samples = (0..active_len)
        .map(|radial| {
            projectors.localized_large[(radial, row_index)] * full_large[(radial, row_index)]
                + projectors.localized_small[(radial, row_index)] * full_small[(radial, row_index)]
        })
        .collect::<Vec<_>>();
    let expected_overlap = somm2(
        &radius_values,
        &overlap_samples,
        step,
        2.0 * final_l + 2.0,
        radii[active_len - 1],
        0,
    )?;
    assert!((inputs.overlaps[row_index] - expected_overlap).abs() < 1.0e-12);

    let localized_large = Array1::from_shape_fn(active_len, |radial| {
        Complex64::new(projectors.localized_large[(radial, row_index)], 0.0)
    });
    let localized_small = Array1::from_shape_fn(active_len, |radial| {
        Complex64::new(projectors.localized_small[(radial, row_index)], 0.0)
    });
    let expected_localized = xsph_radial_integral(XsphRadialIntegralInput {
        mode: XsphRadialIntegralMode::RelativisticMatrixElement,
        multipole: XsphTransitionMultipole::ElectricDipole,
        initial_kappa: row.initial_kappa,
        final_kappa: row.final_kappa,
        initial_large: initial_large.view(),
        initial_small: initial_small.view(),
        final_large_regular: localized_large.view(),
        final_small_regular: localized_small.view(),
        xray_bessel: xray_bessel.view(),
        radii: radii.view(),
        log_step: step,
        active_len,
    })?;
    let polarization_m2 = row.final_m2 - row.initial_m2;
    let angular = refeff_core::wigner_3j(
        row.final_j2,
        2,
        row.initial_j2,
        -row.final_m2,
        polarization_m2,
        2,
    )?;
    let phase = if ((row.final_j2 - row.final_m2) / 2) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    assert!(
        (inputs.localized_dipoles[row_index] - expected_localized.value.im * angular * phase).abs()
            < 1.0e-12
    );

    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.25, Complex64::new(0.0, 0.0))?;
    let raw = tdlda_raw_response_from_source_plan(
        &plan,
        0.25,
        Complex64::new(0.0, 0.0),
        0.0,
        inputs.overlaps.view(),
        inputs.localized_dipoles.view(),
        inputs.full_dipoles.view(),
    )?;
    assert_eq!(
        raw.raw_imaginary_response.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert!(raw.occupied_rows.iter().any(|value| *value));
    assert!(
        raw.raw_imaginary_response
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert_eq!(
        raw.raw_imaginary_response[(row_index, row_index)],
        -2.0 * row_wave_numbers.row_wave_numbers[row_index]
            * inputs.overlaps[row_index]
            * inputs.overlaps[row_index]
    );

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_raw_response_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;

    assert_eq!(plan.initial_l, 1);
    assert_eq!(plan.plus_basis_count, 3);
    let overlaps = Array1::from_shape_fn(plan.matrix_size, |row| 0.2 + 0.005 * row as f64);
    let localized_dipoles = Array1::from_shape_fn(plan.matrix_size, |row| 1.0 + 0.1 * row as f64);
    let full_dipoles = Array1::from_shape_fn(plan.matrix_size, |row| 2.0 + 0.1 * row as f64);
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.05, Complex64::new(0.0, 0.25))?;

    let raw = tdlda_raw_response_from_source_plan(
        &plan,
        0.05,
        Complex64::new(0.0, 0.25),
        0.0,
        overlaps.view(),
        localized_dipoles.view(),
        full_dipoles.view(),
    )?;

    assert_eq!(
        raw.raw_imaginary_response.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    let split_row = plan
        .reference_shifts
        .iter()
        .position(|value| *value < -1.0e-12)
        .context("split-edge rows should carry a negative reference shift")?;
    assert!(!raw.occupied_rows[split_row]);
    assert_eq!(raw.localized_dipoles[split_row], 0.0);
    assert_eq!(raw.full_dipoles[split_row], 0.0);

    let plus_stride = 3 * (2 * plan.initial_l as usize + 1);
    let plus_row = (plus_stride..plan.matrix_size)
        .find(|&row| plan.reference_shifts[row].abs() < 1.0e-14)
        .context("source plan should contain an occupied plus-basis row with a predecessor")?;
    let plus_column = plus_row - plus_stride;
    assert!((row_wave_numbers.momentum_squared[plus_row] - 0.05).abs() < 1.0e-14);
    let expected = -2.0
        * row_wave_numbers.row_wave_numbers[plus_row]
        * overlaps[plus_row]
        * overlaps[plus_column];
    assert!((raw.raw_imaginary_response[(plus_row, plus_column)] - expected).abs() < 1.0e-14);
    assert!((raw.raw_imaginary_response[(plus_column, plus_row)] - expected).abs() < 1.0e-14);
    assert_eq!(raw.localized_dipoles[plus_row], localized_dipoles[plus_row]);
    assert_eq!(raw.full_dipoles[plus_row], full_dipoles[plus_row]);

    Ok(())
}

#[test]
fn xsph_folds_tdlda_projected_kernel_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;

    assert_eq!(plan.initial_l, 1);
    assert_eq!(plan.plus_basis_count, 3);
    assert_eq!(plan.minus_basis_count, 0);
    let projected_kernel =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(row, column)| {
            Complex64::new(
                100.0 * row as f64 + column as f64 + 1.0,
                row as f64 - column as f64,
            )
        });

    let folded = tdlda_projected_kernel_from_source_plan(&plan, projected_kernel.view())?;

    let plus_stride = 3 * (2 * plan.initial_l as usize + 1);
    for column in 0..plan.matrix_size {
        assert_eq!(
            folded.projected_kernel[(0, column)],
            projected_kernel[(0, column)]
        );
        assert_eq!(
            folded.projected_kernel[(plus_stride - 1, column)],
            projected_kernel[(plus_stride - 1, column)]
        );
        for row in plus_stride..plan.matrix_size {
            assert_eq!(
                folded.projected_kernel[(row, column)],
                Complex64::new(0.0, 0.0)
            );
        }
    }

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_direct_kernel_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.2, Complex64::new(0.0, 0.0))?;
    let active_len = 3;
    let radii = Array1::from_vec(vec![1.0, 1.4, 2.0]);
    let core_hole_potential = Array1::from_vec(vec![0.5, 0.6, 0.8]);
    let localized_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        0.05 * (radial as f64 + 1.0) + 0.002 * row as f64
    });
    let localized_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        0.02 * (radial as f64 + 1.0) + 0.001 * row as f64
    });
    let full_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        localized_large[(radial, row)] + 0.1
    });
    let full_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        localized_small[(radial, row)] + 0.05
    });

    let direct = tdlda_direct_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        0.2,
        0.0,
        0.25,
        active_len,
        radii.view(),
        core_hole_potential.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
    )?;

    assert_eq!(direct.kernel.dim(), (plan.matrix_size, plan.matrix_size));
    assert_eq!(
        direct.projected_kernel.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    let plus_stride = 3 * (2 * plan.initial_l as usize + 1);
    assert!(direct.kernel[(0, 0)].re > 0.0);
    assert!(direct.kernel[(plus_stride, 0)].re > 0.0);
    assert_eq!(
        direct.kernel[(0, plus_stride)],
        direct.kernel[(plus_stride, 0)]
    );
    assert!(direct.projected_kernel[(0, plus_stride)].re > 0.0);

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_coulomb_fields_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;

    let active_len = 6;
    let source_len = active_len - 1;
    let coefficient_count = 4;
    let step = 0.08;
    let multipole = 1;
    let radii = Array1::from_shape_fn(active_len, |radial| 0.55 * (step * radial as f64).exp());
    let orbital_count = plan
        .basis
        .rows
        .iter()
        .map(|row| row.core_orbital_index_1based as usize)
        .max()
        .context("TDLDA source plan should contain core orbital indices")?;
    let orbital_large = Array2::from_shape_fn((active_len, orbital_count), |(radial, orbital)| {
        0.18 + 0.025 * radial as f64 + 0.007 * orbital as f64
    });
    let orbital_small = Array2::from_shape_fn((active_len, orbital_count), |(radial, orbital)| {
        0.04 + 0.012 * radial as f64 + 0.003 * orbital as f64
    });
    let orbital_large_coefficients = Array2::from_shape_fn(
        (coefficient_count, orbital_count),
        |(coefficient, orbital)| 0.06 + 0.01 * coefficient as f64 + 0.004 * orbital as f64,
    );
    let orbital_small_coefficients = Array2::from_shape_fn(
        (coefficient_count, orbital_count),
        |(coefficient, orbital)| 0.02 + 0.006 * coefficient as f64 + 0.002 * orbital as f64,
    );
    let orbital_powers =
        Array1::from_shape_fn(orbital_count, |orbital| 0.45 + 0.05 * orbital as f64);
    let orbital_lengths = Array1::from_elem(orbital_count, source_len);
    let target_powers = Array1::from_shape_fn(plan.matrix_size, |row| 1.0 + 0.015 * row as f64);
    let target_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.22 + 0.018 * radial as f64 + 0.001 * row as f64,
            0.010 * radial as f64 - 0.0002 * row as f64,
        )
    });
    let target_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.05 + 0.009 * radial as f64 + 0.0005 * row as f64,
            -0.003 * radial as f64 + 0.0001 * row as f64,
        )
    });
    let target_large_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| {
            Complex64::new(
                0.04 + 0.008 * coefficient as f64 + 0.0003 * row as f64,
                0.002 * coefficient as f64,
            )
        },
    );
    let target_small_coefficients = Array2::from_shape_fn(
        (coefficient_count, plan.matrix_size),
        |(coefficient, row)| {
            Complex64::new(
                0.015 + 0.004 * coefficient as f64 + 0.0002 * row as f64,
                -0.001 * coefficient as f64,
            )
        },
    );

    let fields = tdlda_coulomb_fields_from_source_plan(
        &plan,
        active_len,
        source_len,
        coefficient_count,
        step,
        multipole,
        radii.view(),
        orbital_large.view(),
        orbital_small.view(),
        orbital_large_coefficients.view(),
        orbital_small_coefficients.view(),
        orbital_powers.view(),
        orbital_lengths.view(),
        target_large.view(),
        target_small.view(),
        target_large_coefficients.view(),
        target_small_coefficients.view(),
        target_powers.view(),
    )?;

    assert_eq!(fields.fields.dim(), (active_len, plan.matrix_size));
    assert_eq!(fields.computed_lengths.len(), plan.matrix_size);
    assert!(
        fields
            .computed_lengths
            .iter()
            .all(|length| *length == source_len + 1)
    );
    assert!(fields.fields.iter().any(|value| value.norm() > 0.0));
    assert!(
        fields
            .origin_constants
            .iter()
            .all(|value| value.re.is_finite())
    );
    assert!(
        fields
            .origin_constants
            .iter()
            .all(|value| value.im.is_finite())
    );

    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.2, Complex64::new(0.0, 0.0))?;
    let fxc0 = Array1::from_shape_fn(active_len, |radial| 0.05 + 0.01 * radial as f64);
    let fxc = Array1::from_shape_fn(active_len, |radial| 0.08 + 0.015 * radial as f64);
    let fxcim = Array1::from_shape_fn(active_len, |radial| 0.004 + 0.002 * radial as f64);
    let response_large = target_large.mapv(|value| value + Complex64::new(0.03, 0.002));
    let response_small = target_small.mapv(|value| value + Complex64::new(0.01, -0.001));
    let localized_large = target_large.mapv(|value| value + Complex64::new(0.09, -0.004));
    let localized_small = target_small.mapv(|value| value + Complex64::new(0.02, 0.001));
    let full_large = localized_large.mapv(|value| value + Complex64::new(0.06, -0.002));
    let full_small = localized_small.mapv(|value| value + Complex64::new(0.015, 0.001));

    let radial = tdlda_radial_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        1,
        0.75,
        active_len,
        radii.view(),
        fxc0.view(),
        fxc.view(),
        fxcim.view(),
        response_large.view(),
        response_small.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
        fields.fields.view(),
    )?;

    assert_eq!(
        radial.radial_integrals.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert!(
        radial
            .radial_integrals
            .iter()
            .any(|value| value.norm() > 0.0)
    );

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_nonlocal_exchange_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.25, Complex64::new(0.0, 0.0))?;

    let active_len = 6;
    let source_len = active_len - 1;
    let coefficient_count = 4;
    let step = 0.07;
    let radii = Array1::from_shape_fn(active_len, |radial| 0.50 * (step * radial as f64).exp());
    let orbital_count = plan
        .basis
        .rows
        .iter()
        .map(|row| row.core_orbital_index_1based as usize)
        .max()
        .context("TDLDA source plan should contain core orbital indices")?;
    let orbital_large = Array2::from_shape_fn((active_len, orbital_count), |(radial, orbital)| {
        0.20 + 0.020 * radial as f64 + 0.006 * orbital as f64
    });
    let orbital_small = Array2::from_shape_fn((active_len, orbital_count), |(radial, orbital)| {
        0.05 + 0.010 * radial as f64 + 0.003 * orbital as f64
    });
    let orbital_large_coefficients = Array2::from_shape_fn(
        (coefficient_count, orbital_count),
        |(coefficient, orbital)| 0.05 + 0.008 * coefficient as f64 + 0.003 * orbital as f64,
    );
    let orbital_small_coefficients = Array2::from_shape_fn(
        (coefficient_count, orbital_count),
        |(coefficient, orbital)| 0.018 + 0.004 * coefficient as f64 + 0.0015 * orbital as f64,
    );
    let orbital_powers =
        Array1::from_shape_fn(orbital_count, |orbital| 0.60 + 0.08 * orbital as f64);
    let orbital_lengths = Array1::from_elem(orbital_count, source_len);
    let localized_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.26 + 0.016 * radial as f64 + 0.0015 * row as f64,
            0.002 * radial as f64 - 0.0002 * row as f64,
        )
    });
    let localized_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.055 + 0.008 * radial as f64 + 0.0007 * row as f64,
            -0.001 * radial as f64 + 0.0001 * row as f64,
        )
    });
    let full_large = localized_large.mapv(|value| value + Complex64::new(0.07, -0.003));
    let full_small = localized_small.mapv(|value| value + Complex64::new(0.018, 0.001));

    let nonlocal = tdlda_nonlocal_exchange_from_source_plan(
        &plan,
        &row_wave_numbers,
        active_len,
        source_len,
        coefficient_count,
        step,
        2,
        0.80,
        radii.view(),
        orbital_large.view(),
        orbital_small.view(),
        orbital_large_coefficients.view(),
        orbital_small_coefficients.view(),
        orbital_powers.view(),
        orbital_lengths.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
    )?;

    assert_eq!(
        nonlocal.radial_integrals.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert_eq!(
        nonlocal.projected_radial_integrals.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    let same_kappa_pair = (0..plan.matrix_size)
        .flat_map(|row| (0..plan.matrix_size).map(move |column| (row, column)))
        .find(|&(row, column)| {
            row != column
                && plan.basis.rows[row].initial_kappa == plan.basis.rows[column].initial_kappa
        })
        .context("fixture should contain same-kappa row pairs")?;
    assert_eq!(
        nonlocal.radial_integrals[same_kappa_pair],
        Complex64::new(0.0, 0.0)
    );
    assert!(
        nonlocal
            .radial_integrals
            .iter()
            .any(|value| value.norm() > 0.0)
    );

    let zero_radial = Array2::<Complex64>::zeros((plan.matrix_size, plan.matrix_size));
    let angular = tdlda_angular_kernel_from_source_plan(
        &plan,
        row_wave_numbers.positive_momentum_rows.view(),
        zero_radial.view(),
        zero_radial.view(),
        Some(nonlocal.radial_integrals.view()),
        Some(nonlocal.projected_radial_integrals.view()),
    )?;

    assert!(
        angular
            .nonlocal_prefactors
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(angular.kernel.iter().any(|value| value.norm() > 0.0));
    assert!(
        angular
            .projected_kernel
            .iter()
            .any(|value| value.norm() > 0.0)
    );

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_getchi0_kernel_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.22, Complex64::new(0.0, 0.0))?;
    let active_len = 4;
    let radii = Array1::from_vec(vec![0.6, 0.9, 1.4, 2.1]);
    let core_hole_potential =
        Array1::from_shape_fn(active_len, |radial| 0.30 + 0.04 * radial as f64);
    let fxc0 = Array1::from_shape_fn(active_len, |radial| 0.05 + 0.01 * radial as f64);
    let fxc = Array1::from_shape_fn(active_len, |radial| 0.08 + 0.015 * radial as f64);
    let fxcim = Array1::from_shape_fn(active_len, |radial| 0.004 + 0.002 * radial as f64);
    let response_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.21 + 0.018 * radial as f64 + 0.002 * row as f64,
            0.001 * row as f64,
        )
    });
    let response_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.045 + 0.008 * radial as f64 + 0.001 * row as f64,
            -0.0004 * radial as f64,
        )
    });
    let localized_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.31 + 0.014 * radial as f64 + 0.002 * row as f64,
            0.0006 * row as f64,
        )
    });
    let localized_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.034 + 0.006 * radial as f64 + 0.001 * row as f64,
            0.0003 * radial as f64,
        )
    });
    let full_large = localized_large.mapv(|value| value + Complex64::new(0.07, -0.002));
    let full_small = localized_small.mapv(|value| value + Complex64::new(0.018, 0.001));
    let direct_localized_large = localized_large.mapv(|value| value.re);
    let direct_localized_small = localized_small.mapv(|value| value.re);
    let direct_full_large = full_large.mapv(|value| value.re);
    let direct_full_small = full_small.mapv(|value| value.re);
    let coulomb_fields = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.42 + 0.025 * radial as f64 + 0.001 * row as f64,
            0.02 * radial as f64,
        )
    });
    let nonlocal_radial =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(row, column)| {
            Complex64::new(
                0.020 + 0.001 * row as f64 + 0.0005 * column as f64,
                0.0007 * row as f64,
            )
        });
    let nonlocal_projected =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(row, column)| {
            Complex64::new(0.015 + 0.0008 * column as f64, -0.0004 * row as f64)
        });

    let assembled = tdlda_getchi0_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        1,
        0.70,
        0.22,
        0.0,
        0.25,
        active_len,
        radii.view(),
        core_hole_potential.view(),
        fxc0.view(),
        fxc.view(),
        fxcim.view(),
        direct_localized_large.view(),
        direct_localized_small.view(),
        direct_full_large.view(),
        direct_full_small.view(),
        response_large.view(),
        response_small.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
        coulomb_fields.view(),
        Some(nonlocal_radial.view()),
        Some(nonlocal_projected.view()),
    )?;
    let direct = tdlda_direct_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        0.22,
        0.0,
        0.25,
        active_len,
        radii.view(),
        core_hole_potential.view(),
        direct_localized_large.view(),
        direct_localized_small.view(),
        direct_full_large.view(),
        direct_full_small.view(),
    )?;
    let radial = tdlda_radial_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        1,
        0.70,
        active_len,
        radii.view(),
        fxc0.view(),
        fxc.view(),
        fxcim.view(),
        response_large.view(),
        response_small.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
        coulomb_fields.view(),
    )?;
    let angular = tdlda_angular_kernel_from_source_plan(
        &plan,
        row_wave_numbers.positive_momentum_rows.view(),
        radial.radial_integrals.view(),
        radial.projected_radial_integrals.view(),
        Some(nonlocal_radial.view()),
        Some(nonlocal_projected.view()),
    )?;
    let expected_kernel = &direct.kernel + &angular.kernel;
    let expected_projected = &direct.projected_kernel + &angular.projected_kernel;

    assert_eq!(assembled.direct, direct);
    assert_eq!(assembled.radial, radial);
    assert_eq!(assembled.angular, angular);
    assert_eq!(assembled.kernel, expected_kernel);
    assert_eq!(assembled.projected_kernel, expected_projected);
    assert!(assembled.kernel.iter().any(|value| value.norm() > 0.0));
    assert!(
        assembled
            .projected_kernel
            .iter()
            .any(|value| value.norm() > 0.0)
    );

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_radial_kernel_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.2, Complex64::new(0.0, 0.0))?;
    let active_len = 4;
    let radii = Array1::from_vec(vec![0.6, 0.9, 1.4, 2.2]);
    let fxc0 = Array1::from_shape_fn(active_len, |radial| 0.05 + 0.01 * radial as f64);
    let fxc = Array1::from_shape_fn(active_len, |radial| 0.08 + 0.015 * radial as f64);
    let fxcim = Array1::from_shape_fn(active_len, |radial| 0.004 + 0.002 * radial as f64);
    let response_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.20 + 0.02 * radial as f64 + 0.002 * row as f64,
            0.001 * row as f64,
        )
    });
    let response_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.04 + 0.01 * radial as f64 + 0.001 * row as f64,
            -0.0005 * radial as f64,
        )
    });
    let localized_large = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.30 + 0.015 * radial as f64 + 0.003 * row as f64,
            0.0007 * row as f64,
        )
    });
    let localized_small = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.03 + 0.008 * radial as f64 + 0.0015 * row as f64,
            0.0004 * radial as f64,
        )
    });
    let full_large = localized_large.mapv(|value| value + Complex64::new(0.08, -0.002));
    let full_small = localized_small.mapv(|value| value + Complex64::new(0.02, 0.001));
    let coulomb_fields = Array2::from_shape_fn((active_len, plan.matrix_size), |(radial, row)| {
        Complex64::new(
            0.40 + 0.02 * radial as f64 + 0.001 * row as f64,
            0.5 * row as f64,
        )
    });

    let radial = tdlda_radial_kernel_from_source_plan(
        &plan,
        &row_wave_numbers,
        1,
        0.75,
        active_len,
        radii.view(),
        fxc0.view(),
        fxc.view(),
        fxcim.view(),
        response_large.view(),
        response_small.view(),
        localized_large.view(),
        localized_small.view(),
        full_large.view(),
        full_small.view(),
        coulomb_fields.view(),
    )?;
    let angular = tdlda_angular_kernel_from_source_plan(
        &plan,
        row_wave_numbers.positive_momentum_rows.view(),
        radial.radial_integrals.view(),
        radial.projected_radial_integrals.view(),
        None,
        None,
    )?;

    assert_eq!(
        radial.radial_integrals.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert_eq!(
        radial.projected_radial_integrals.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert!(
        radial
            .radial_integrals
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(angular.kernel.iter().any(|value| value.norm() > 0.0));

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_angular_kernel_from_source_plan() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 1;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_wave_numbers =
        tdlda_row_wave_numbers_from_source_plan(&plan, 0.2, Complex64::new(0.0, 0.0))?;
    let radial_integrals = Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(r, c)| {
        Complex64::new(0.20 + 0.01 * r as f64, 0.03 * c as f64)
    });
    let projected_radial_integrals =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(r, c)| {
            Complex64::new(0.10 + 0.02 * c as f64, -0.01 * r as f64)
        });
    let nonlocal_radial_integrals =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(r, c)| {
            Complex64::new(0.05 + 0.002 * (r + c) as f64, 0.01 * r as f64)
        });
    let nonlocal_projected_radial_integrals =
        Array2::from_shape_fn((plan.matrix_size, plan.matrix_size), |(r, c)| {
            Complex64::new(0.04 + 0.003 * c as f64, -0.004 * r as f64)
        });

    let angular = tdlda_angular_kernel_from_source_plan(
        &plan,
        row_wave_numbers.positive_momentum_rows.view(),
        radial_integrals.view(),
        projected_radial_integrals.view(),
        Some(nonlocal_radial_integrals.view()),
        Some(nonlocal_projected_radial_integrals.view()),
    )?;

    assert_eq!(angular.kernel.dim(), (plan.matrix_size, plan.matrix_size));
    assert_eq!(
        angular.projected_kernel.dim(),
        (plan.matrix_size, plan.matrix_size)
    );
    assert!(
        angular
            .prefactors
            .iter()
            .zip(angular.kernel.iter())
            .any(|(prefactor, value)| prefactor.abs() > 0.0 && value.norm() > 0.0)
    );
    assert!(
        angular
            .nonlocal_prefactors
            .iter()
            .any(|prefactor| prefactor.abs() > 0.0)
    );

    Ok(())
}

#[test]
fn xsph_assembles_tdlda_xsedge_from_raw_source_components() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.advanced.ipmbse = 2;
    input.advanced.itdlda = 2;
    input.advanced.ibasis = 0;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let plan = tdlda_xsectd_source_plan_from_caches(&caches, &input, &pot, &orbital_tables)?
        .context("L3 PMBSE source handoff should produce a TDLDA plan")?;
    let row_count = plan.multipliers.energy_hartree.len();
    let reference_energy = Array1::from_elem(row_count, Complex64::new(0.0, 0.0));
    let energy_rows =
        tdlda_energy_rows_from_source_plan(&plan, &input, reference_energy.view(), 0.0, 0.0)?;
    let raw_response = Array3::<f64>::zeros((row_count, plan.matrix_size, plan.matrix_size));
    let localized_dipole = Array2::from_elem((row_count, plan.matrix_size), 0.5);
    let full_dipole = Array2::from_shape_fn((row_count, plan.matrix_size), |(_, row)| {
        0.25 + 0.01 * row as f64
    });
    let kernel = Array3::<Complex64>::zeros((row_count, plan.matrix_size, plan.matrix_size));
    let projected_kernel =
        Array3::<Complex64>::zeros((row_count, plan.matrix_size, plan.matrix_size));

    let xsedge = tdlda_xsedge_dat_from_raw_source_components(
        &plan,
        &energy_rows,
        raw_response.view(),
        localized_dipole.view(),
        full_dipole.view(),
        kernel.view(),
        projected_kernel.view(),
        0.0,
        0.0,
    )?;

    assert_eq!(xsedge.row_count(), row_count);
    assert!(xsedge.has_branch_columns());
    for (actual, expected) in xsedge
        .energy_ev
        .iter()
        .zip(plan.multipliers.energy_hartree.iter())
    {
        assert!((actual - expected * FEFF_HARTREE_EV).abs() < 1.0e-8);
    }
    for row in 0..row_count {
        assert!(xsedge.total_single_particle[row].is_finite());
        assert!(xsedge.total_screened[row].is_finite());
        assert!(
            (xsedge.total_single_particle[row] - xsedge.total_screened[row]).abs()
                <= 1.0e-8 + xsedge.total_single_particle[row].abs() * 1.0e-12
        );
    }
    Ok(())
}

#[test]
fn xsph_writes_tdlda_xsedge_dat_from_source_components() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_split_pmbse_xmu_sources(temp.path())?;
    let multipliers = tdlda_pmbse_channel_multipliers_from_source(temp.path(), -2, 10)?
        .context("PMBSE source should produce channel multipliers for xsedge.dat")?;
    let row_count = multipliers.energy_hartree.len();
    let spectra = XsphTdldaBroadenedChannelSpectra {
        single_particle_channels: Array2::from_shape_fn((row_count, 4), |(_, channel)| {
            10.0 * (channel as f64 + 1.0)
        }),
        screened_channels: Array2::from_shape_fn((row_count, 4), |(_, channel)| {
            channel as f64 + 1.0
        }),
    };

    let xsedge =
        write_tdlda_xsedge_dat_from_source_components(temp.path(), 4, &spectra, &multipliers)?;

    assert_eq!(xsedge.row_count(), row_count);
    assert!(xsedge.has_branch_columns());
    assert_eq!(
        xsedge.energy_ev.to_vec(),
        vec![0.0, 1.0, 2.0, 2.5, 3.0, 3.5]
    );
    assert_eq!(xsedge.total_single_particle[0], 2400.0);
    assert_eq!(
        xsedge
            .plus_branch_single_particle
            .as_ref()
            .map(|values| values[0]),
        Some(200.0)
    );
    assert_eq!(
        xsedge
            .minus_branch_screened
            .as_ref()
            .map(|values| values[0]),
        Some(220.0)
    );
    assert_eq!(read_xsedge_dat(temp.path().join("xsedge.dat"))?, xsedge);
    Ok(())
}

#[test]
fn xsph_module_skips_source_handoff_when_config_orbitals_exceed_pot_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    let mut config = sample_normal_phase_config_dat();
    config.potentials[0].occupations[4] = 0.25;
    write_config_dat(temp.path().join("config.dat"), &config)?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_phase_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("incomplete XSPH source handoff should require complete source state")?;

    assert!(
        error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn xsph_module_requires_wscrn_for_screened_core_hole_source_generation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.ispec = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.nohole = super::XSPH_SCREENED_CORE_HOLE_SELECTOR;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_supported_xsph_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("screened core-hole XSPH without wscrn.dat should require source state")?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    assert!(!temp.path().join("phase.bin").is_file());
    assert!(!temp.path().join("xsect.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_recovers_screened_core_hole_wscrn_from_vtot_and_apot_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.ispec = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.nohole = super::XSPH_SCREENED_CORE_HOLE_SELECTOR;
    write_pot_bin(temp.path().join("pot.bin"), &pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    let vtot = sample_screened_core_hole_vtot_dat();
    write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
    let (large_component, small_component) = sample_screened_core_hole_components();
    write_apot_bin(
        temp.path().join("apot.bin"),
        &sample_screened_core_hole_apot_bin(&large_component, &small_component),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
    assert_eq!(wscrn.radius_bohr, vtot.radius_bohr);
    assert_eq!(wscrn.screened_potential, vtot.screened_core_hole_potential);
    assert!(
        wscrn
            .core_hole_potential
            .iter()
            .any(|value| value.abs() > 1.0e-12)
    );
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(!temp.path().join("logscreen.dat").exists());
    Ok(())
}

#[test]
fn xsph_module_generates_missing_rl_from_source_handoff_without_replacing_cached_phase_xsect()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
        input.print_rl = true;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let rl_path = temp.path().join("rl.dat");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    std::fs::remove_file(&rl_path)?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    let radial = read_xsph_rl_dat(&rl_path)?;
    assert!(radial.record_count() > 0);
    assert!(radial.records.iter().all(|record| {
        record.regular_large.len() == radial.radial_count()
            && record.regular_small.len() == radial.radial_count()
            && record.angular_momentum <= radial.angular_limit
    }));
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_rl_from_source_handoff_without_replacing_cached_phase_xsect()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
        input.print_rl = true;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let rl_path = temp.path().join("rl.dat");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    std::fs::write(&rl_path, "not rl.dat\n")?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    let radial = read_xsph_rl_dat(&rl_path)?;
    assert!(radial.record_count() > 0);
    assert!(radial.records.iter().all(|record| {
        record.regular_large.len() == radial.radial_count()
            && record.regular_small.len() == radial.radial_count()
            && record.angular_momentum <= radial.angular_limit
    }));
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_rl_from_source_handoff_without_replacing_cached_phase_xsect()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
        input.print_rl = true;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let rl_path = temp.path().join("rl.dat");
    let expected_rl = read_xsph_rl_dat(&rl_path)?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let mut stale_rl = expected_rl.clone();
    stale_rl.records[0].regular_large[0] += Complex64::new(0.25, -0.125);
    write_xsph_rl_dat(&rl_path, &stale_rl)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 6);
    let actual_phase = read_phase_bin(&phase_path)?;
    let actual_xsect = read_xsect_dat(&xsect_path)?;
    assert_phase_cache_sentinel_preserved(&actual_phase, &expected_phase);
    assert_xsect_table_preserved(&actual_xsect, &expected_xsect);
    assert_eq!(read_xsph_rl_dat(&rl_path)?, expected_rl);
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_module_log_for_missing_rl_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
        input.print_rl = true;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    run_in_dir(temp.path())?;

    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let rl_path = temp.path().join("rl.dat");
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    std::fs::remove_file(&rl_path)?;
    std::fs::write(temp.path().join("log2.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 6);
    assert_phase_cache_sentinel_preserved(&read_phase_bin(&phase_path)?, &expected_phase);
    assert_xsect_table_preserved(&read_xsect_dat(&xsect_path)?, &expected_xsect);
    assert!(read_xsph_rl_dat(&rl_path)?.record_count() > 0);
    let log = read_module_log_dat(temp.path().join("log2.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Done with module: cross-section and phases (XSPH)."))
    );
    Ok(())
}

#[test]
fn xsph_module_generates_l2lp_filtered_xsect_from_pot_and_config_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_uses_global_l2lp_for_source_xsect_from_pot_and_config() -> Result<()> {
    let global_l2lp = tempfile::tempdir()?;
    write_xsph_input_custom(global_l2lp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(global_l2lp.path(), 0, 1)?;
    write_grid_inp(
        global_l2lp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(
        global_l2lp.path().join("pot.bin"),
        &sample_normal_phase_pot_bin(),
    )?;
    write_config_dat(
        global_l2lp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    let explicit_l2lp = tempfile::tempdir()?;
    write_xsph_input_custom(explicit_l2lp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        explicit_l2lp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(
        explicit_l2lp.path().join("pot.bin"),
        &sample_normal_phase_pot_bin(),
    )?;
    write_config_dat(
        explicit_l2lp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(global_l2lp.path())?);
    assert!(has_supported_xsph_output(explicit_l2lp.path())?);
    run_in_dir(global_l2lp.path())?;
    run_in_dir(explicit_l2lp.path())?;

    assert_eq!(
        read_xsect_dat(global_l2lp.path().join("xsect.dat"))?,
        read_xsect_dat(explicit_l2lp.path().join("xsect.dat"))?
    );
    assert_eq!(
        read_phase_bin(global_l2lp.path().join("phase.bin"))?.transition_moments,
        read_phase_bin(explicit_l2lp.path().join("phase.bin"))?.transition_moments
    );
    Ok(())
}

#[test]
fn xsph_module_generates_global_e2_xsect_from_pot_and_config_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 2, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0),
        "expected source-backed XSPH transition moments"
    );
    assert!(
        (3..phase.transition_count).any(|transition| {
            (0..phase.energy_count)
                .any(|energy| phase.transition_moments[(energy, 0, transition, 0)].norm() > 0.0)
        }),
        "expected E2 higher-multipole transition slots to be populated"
    );
    Ok(())
}

#[test]
fn xsph_module_uses_global_angular_controls_for_source_xsect_from_pot_and_config() -> Result<()> {
    let averaged = tempfile::tempdir()?;
    write_xsph_input_custom(averaged.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(averaged.path(), 2, 0)?;
    write_grid_inp(
        averaged.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(
        averaged.path().join("pot.bin"),
        &sample_normal_phase_pot_bin(),
    )?;
    write_config_dat(
        averaged.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    let polarized = tempfile::tempdir()?;
    write_xsph_input_custom(polarized.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input_custom(polarized.path(), |global| {
        global.control.ipol = 1;
        global.control.ispin = 1;
        global.control.le2 = 2;
        global.control.angks = 0.3;
        global.control.l2lp = 0;
        global.evec = [1.0, 0.0, 0.0];
        global.spvec = [0.0, 0.0, 1.0];
        global.polarization_tensor = [
            [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            [0.0; 6],
            [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        ];
    })?;
    write_grid_inp(
        polarized.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(
        polarized.path().join("pot.bin"),
        &sample_normal_phase_pot_bin(),
    )?;
    write_config_dat(
        polarized.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(averaged.path())?);
    assert!(has_supported_xsph_output(polarized.path())?);
    run_in_dir(averaged.path())?;
    run_in_dir(polarized.path())?;

    let averaged_xsect = read_xsect_dat(averaged.path().join("xsect.dat"))?;
    let polarized_xsect = read_xsect_dat(polarized.path().join("xsect.dat"))?;
    assert_ne!(
        averaged_xsect, polarized_xsect,
        "expected global angular controls to change generated xsect.dat"
    );
    Ok(())
}

#[test]
fn xsph_module_generates_polarized_multipoles_three_with_additive_source_parity() -> Result<()> {
    let mut generated = Vec::new();
    for le2 in 0..=3 {
        let temp = tempfile::tempdir()?;
        write_xsph_source_setup_with_e2_controls(temp.path())?;
        write_global_input_custom(temp.path(), |global| {
            global.control.ipol = 1;
            global.control.ispin = 0;
            global.control.le2 = le2;
            global.control.angks = 0.3;
            global.control.l2lp = 0;
            global.evec = [1.0, 0.0, 0.0];
            global.xivec = [0.0, 0.0, 1.0];
            global.spvec = [0.0, 0.0, 1.0];
            global.polarization_tensor = [
                [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
                [0.0; 6],
                [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            ];
        })?;

        assert!(
            has_supported_xsph_output(temp.path())?,
            "polarized MULTIPOLES={le2} should have a source-backed phase/xsect path"
        );
        run_in_dir(temp.path())?;
        generated.push((
            read_phase_bin(temp.path().join("phase.bin"))?,
            read_xsect_dat(temp.path().join("xsect.dat"))?,
        ));
    }

    let (dipole_phase, dipole_xsect) = &generated[0];
    let (magnetic_phase, magnetic_xsect) = &generated[1];
    let (quadrupole_phase, quadrupole_xsect) = &generated[2];
    let (combined_phase, combined_xsect) = &generated[3];

    assert_serialized_combined_column_close(
        &combined_xsect.normalized_background,
        &dipole_xsect.normalized_background,
        &magnetic_xsect.normalized_background,
        &quadrupole_xsect.normalized_background,
    );
    assert_serialized_combined_complex_column_close(
        &combined_xsect.cross_section,
        &dipole_xsect.cross_section,
        &magnetic_xsect.cross_section,
        &quadrupole_xsect.cross_section,
    );

    let mut expected_phase = quadrupole_phase.clone();
    expected_phase.transition_moments = &quadrupole_phase.transition_moments
        + &magnetic_phase.transition_moments
        - &dipole_phase.transition_moments;
    assert_phase_transition_moments_close(combined_phase, &expected_phase, 1.0e-8);
    assert!(
        (3..combined_phase.transition_count).any(|transition| {
            combined_phase
                .transition_moments
                .index_axis(Axis(2), transition)
                .iter()
                .any(|value| value.norm() > 0.0)
        }),
        "expected combined E2/M1 transition slots in source-generated phase.bin"
    );
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_xsect_cache_when_source_angular_controls_change() -> Result<()> {
    let averaged = tempfile::tempdir()?;
    write_xsph_source_setup_with_e2_controls(averaged.path())?;
    write_global_input(averaged.path(), 2, 0)?;
    run_in_dir(averaged.path())?;
    let stale_xsect = read_xsect_dat(averaged.path().join("xsect.dat"))?;

    let expected = tempfile::tempdir()?;
    write_xsph_source_setup_with_e2_controls(expected.path())?;
    write_polarized_e2_global_input(expected.path())?;
    run_in_dir(expected.path())?;
    let expected_xsect = read_xsect_dat(expected.path().join("xsect.dat"))?;
    assert_ne!(
        stale_xsect, expected_xsect,
        "test setup should produce a stale same-shape xsect.dat cache"
    );

    let stale = tempfile::tempdir()?;
    write_xsph_source_setup_with_e2_controls(stale.path())?;
    write_polarized_e2_global_input(stale.path())?;
    std::fs::copy(
        averaged.path().join("phase.bin"),
        stale.path().join("phase.bin"),
    )?;
    std::fs::copy(
        averaged.path().join("xsect.dat"),
        stale.path().join("xsect.dat"),
    )?;

    assert!(!has_cached_xsph_output(stale.path())?);
    assert!(has_supported_xsph_output(stale.path())?);

    let written = run_in_dir(stale.path())?;

    assert!(written >= 2);
    assert_eq!(
        read_xsect_dat(stale.path().join("xsect.dat"))?,
        expected_xsect
    );
    Ok(())
}

#[test]
fn xsph_module_generates_phase_handoff_when_xsect_branch_is_unsupported() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 1, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert_eq!(written, 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].atomic_number, 29);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(temp.path().join("log2.dat").is_file());
    assert!(!has_supported_phase_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_phase_cache_as_phase_handoff_when_xsect_branch_is_unsupported()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 1;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input(temp.path(), 1, 0)?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert_eq!(written, 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 1);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(temp.path().join("log2.dat").is_file());
    assert!(!has_supported_phase_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_fprime_xsect_with_feff_imaginary_convention() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.ispec = 4;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    for (index, (cross_section, normalized_background)) in xsect
        .cross_section
        .iter()
        .zip(xsect.normalized_background.iter())
        .enumerate()
    {
        assert!(
            cross_section.re.abs() <= 1.0e-12,
            "FPRIME xsect row {} has nonzero real part {}",
            index + 1,
            cross_section.re
        );
        assert!(
            (cross_section.im - normalized_background).abs() <= 1.0e-12,
            "FPRIME xsect row {} imaginary part {} does not match xsnorm {}",
            index + 1,
            cross_section.im,
            normalized_background
        );
    }
    Ok(())
}

#[test]
fn normal_xsect_spin_ground_state_validation_covers_ordinary_selected_and_two_spin_modes()
-> Result<()> {
    let tensor = [[Complex64::new(0.0, 0.0); 3]; 3];
    let ordinary = xsect_angular_controls_from_values(0, tensor, 0, 0, 0.0, 0)?
        .context("ordinary XSPH controls should be supported")?;
    let selected = xsect_angular_controls_from_values(0, tensor, 0, 2, 0.0, 0)?
        .context("selected-spin XSPH controls should be supported")?;
    let two_spin = xsect_angular_controls_from_values(0, tensor, 0, 1, 0.0, 0)?
        .context("two-spin XSPH controls should be supported")?;

    validate_xsect_spin_ground_states(1, 1, &[0], ordinary)?;
    validate_xsect_spin_ground_states(1, 1, &[2], selected)?;
    validate_xsect_spin_ground_states(2, 2, &[-1, 1], two_spin)?;

    let incomplete = validate_xsect_spin_ground_states(2, 1, &[-1, 1], two_spin)
        .expect_err("two-spin XSECT must reject a missing prepared ground state");
    assert!(incomplete.to_string().contains("spin state is incomplete"));

    let mismatched = validate_xsect_spin_ground_states(2, 2, &[1, -1], two_spin)
        .expect_err("two-spin XSECT must reject selectors in the wrong FEFF order");
    assert!(
        mismatched
            .to_string()
            .contains("disagrees with angular-control selector")
    );
    Ok(())
}

#[test]
fn xsph_eels_tensor_override_requires_normal_eels_and_first_phase_pass() -> Result<()> {
    let source_tensor_rows = [
        [-0.75, 0.25, 0.5, -1.0, -0.25, 0.75],
        [1.5, -0.5, -1.25, 0.125, 0.625, -0.375],
        [-2.0, 0.875, 1.75, -0.625, -1.5, 0.5],
    ];
    let source_tensor = [
        [
            Complex64::new(-0.75, 0.25),
            Complex64::new(0.5, -1.0),
            Complex64::new(-0.25, 0.75),
        ],
        [
            Complex64::new(1.5, -0.5),
            Complex64::new(-1.25, 0.125),
            Complex64::new(0.625, -0.375),
        ],
        [
            Complex64::new(-2.0, 0.875),
            Complex64::new(1.75, -0.625),
            Complex64::new(-1.5, 0.5),
        ],
    ];
    let averaged_tensor = [
        [
            Complex64::new(1.0 / 3.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 3.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 3.0, 0.0),
        ],
    ];

    for (calculation_mode, mphase, expected_tensor) in [
        (1, 1, averaged_tensor),
        (9, 1, source_tensor),
        (1, 2, source_tensor),
    ] {
        let temp = tempfile::tempdir()?;
        write_global_input_custom(temp.path(), |global| {
            global.control.ipol = 1;
            global.polarization_tensor = source_tensor_rows;
        })?;
        write_eels_input_with_calculation_mode(temp.path(), calculation_mode)?;

        let input = sample_xsph_input(mphase, 0);
        let controls = xsect_angular_controls(&XsphCachePaths::new(temp.path()), &input)?
            .context("test angular controls should be supported")?;

        assert_eq!(
            controls.polarization, 1,
            "EELS mode {calculation_mode}, mphase {mphase} must preserve ipol"
        );
        assert_eq!(
            controls.polarization_tensor, expected_tensor,
            "unexpected tensor for EELS mode {calculation_mode}, mphase {mphase}"
        );
    }
    Ok(())
}

#[test]
fn xsph_module_generates_two_spin_phase_handoff_without_marking_xsect_complete() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![1.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_handoff(temp.path())?);
    let written = run_supported_phase_handoff_in_dir(temp.path())?;

    assert_eq!(written, 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(phase.reference_energy.dim(), (phase.energy_count, 2));
    assert_eq!(phase.potential_count(), 1);
    assert_eq!(phase.potentials[0].phase_shifts.dim().2, 2);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(temp.path().join("log2.dat").is_file());
    assert!(!has_supported_xsph_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_two_spin_filtered_xsect_handoff_from_global_spin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![1.0];
    })?;
    write_global_input_custom(temp.path(), |global| {
        global.control.ispin = 1;
        global.control.l2lp = 1;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(phase.transition_moments.dim().3, 2);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    for spin in 0..2 {
        assert!(
            phase
                .transition_moments
                .index_axis(Axis(3), spin)
                .iter()
                .any(|value| value.norm() > 0.0),
            "expected source-backed transition moments for spin {spin}"
        );
    }
    assert!(has_supported_xsph_output(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_two_spin_unfiltered_xsect_handoff_from_global_spin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![1.0];
    })?;
    write_global_input_custom(temp.path(), |global| {
        global.control.ispin = 1;
        global.control.l2lp = 0;
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 2);
    assert_eq!(phase.transition_moments.dim().3, 2);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    for spin in 0..2 {
        for transition in 0..2 {
            assert!(
                phase
                    .transition_moments
                    .index_axis(Axis(3), spin)
                    .index_axis(Axis(2), transition)
                    .iter()
                    .any(|value| value.norm() > 0.0),
                "expected unfiltered source-backed transition {transition} for spin {spin}"
            );
        }
    }
    Ok(())
}

#[test]
fn xsph_module_generates_two_spin_higher_multipole_xsect_handoff_from_global_spin() -> Result<()> {
    for (label, le2) in [("M1", 1), ("E2", 2)] {
        let temp = tempfile::tempdir()?;
        write_xsph_input_custom(temp.path(), |input| {
            input.control.nph = 0;
            input.control.i_core_state = 1;
            input.control.ixc = 2;
            input.control.lreal = 1;
            input.control.i_grid = 1;
            input.control.l2lp = 0;
            input.lmaxph = vec![3];
            input.pot_labels = vec!["Cu".to_string()];
            input.spinph = vec![1.0];
        })?;
        write_global_input_custom(temp.path(), |global| {
            global.control.ispin = 1;
            global.control.le2 = le2;
            global.control.l2lp = 0;
        })?;
        write_grid_inp(
            temp.path().join("grid.inp"),
            &sample_single_point_grid_input(),
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_normal_phase_pot_bin())?;
        write_config_dat(
            temp.path().join("config.dat"),
            &sample_normal_phase_config_dat(),
        )?;

        assert!(has_supported_xsph_output(temp.path())?);
        let written = run_in_dir(temp.path())?;

        assert_eq!(written, 5);
        let phase = read_phase_bin(temp.path().join("phase.bin"))?;
        let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
        assert_eq!(phase.spin_count, 2);
        assert_eq!(xsect.energy_count(), phase.energy_count);
        assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
        for spin in 0..2 {
            assert!(
                (3..phase.transition_count).any(|transition| {
                    phase
                        .transition_moments
                        .index_axis(Axis(3), spin)
                        .index_axis(Axis(2), transition)
                        .iter()
                        .any(|value| value.norm() > 0.0)
                }),
                "expected source-backed {label} transition moments for spin {spin}"
            );
        }
    }
    Ok(())
}

#[test]
fn xsph_module_generates_mpse_phase_and_xsect_from_loss_without_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 0;
        input.control.lreal = 0;
        input.control.i_grid = 1;
        input.control.i_plsmn = 1;
        input.control.n_poles = 3;
        input.grid.eps0 = -1.0;
        input.grid.egap = 0.03 * FEFF_HARTREE_EV;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let mut source_pot = sample_normal_phase_pot_bin();
    source_pot.scalars.density_radius = 1.823_121_447;
    write_pot_bin(temp.path().join("pot.bin"), &source_pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_loss_dat(temp.path().join("loss.dat"), &sample_loss_dat())?;
    std::fs::write(
        temp.path().join("exc.dat"),
        "  28.45243  0.02845  1.00000\n",
    )?;
    std::fs::write(
        temp.path().join("specfunct.dat"),
        b"stale one-pole spectral cache",
    )?;

    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 7);
    let excitation_poles = read_exc_dat(temp.path().join("exc.dat"))?;
    assert_eq!(excitation_poles.pole_count(), 3);
    assert!(
        !temp.path().join("specfunct.dat").exists(),
        "changing XSPH excitation poles must invalidate the dependent spectral cache"
    );
    std::fs::write(
        temp.path().join("specfunct.dat"),
        b"compatible-pole cache marker",
    )?;
    run_in_dir(temp.path())?;
    assert_eq!(
        std::fs::read(temp.path().join("specfunct.dat"))?,
        b"compatible-pole cache marker",
        "unchanged XSPH excitation poles must preserve the dependent spectral cache"
    );
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsect.main_energy_count, phase.main_energy_count);
    assert_eq!(xsect.fermi_index, phase.fermi_index as usize);
    assert!(
        phase
            .reference_energy
            .iter()
            .any(|value| value.im.abs() > 0.0)
    );
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    let rendered_xsect = xsect_dat_string(&xsect)?;
    let material = sfconv_so2conv_header_from_text("xsect.dat", &rendered_xsect)?.material;
    assert!(
        (material.core_hole_width_ev - 1.0).abs() <= 1.0e-3,
        "{material:?}"
    );
    assert!(
        (material.wigner_seitz_radius - source_pot.scalars.density_radius).abs() <= 5.0e-4,
        "{material:?}"
    );
    assert_eq!(material.wigner_seitz_radius, 1.823);
    assert!(rendered_xsect.contains("Rs_int= 1.823"));
    assert!(
        (material.interstitial_potential_ev
            - source_pot.scalars.interstitial_potential * FEFF_HARTREE_EV)
            .abs()
            <= 1.0e-2,
        "{material:?}"
    );
    assert!(
        (material.chemical_potential_ev - phase.scalars.edge_energy * FEFF_HARTREE_EV).abs()
            <= 1.0e-2,
        "{material:?}"
    );
    assert!(
        (material.fermi_wave_number_inv_angstrom
            - source_pot.scalars.fermi_momentum / FEFF_BOHR_ANGSTROM)
            .abs()
            <= 1.0e-3,
        "{material:?}"
    );
    let mpse = read_mpse_dat(temp.path().join("mpse.dat"))?;
    assert!(mpse.self_energy.iter().any(|value| value.im.abs() > 0.0));
    Ok(())
}

#[test]
fn xsph_default_mesh_keeps_ordinary_l2lp_filter_on_modern_compiled_capacity() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let caches = super::XsphCachePaths::new(temp.path());
    let pot = sample_normal_phase_pot_bin();
    let mut baseline = sample_xsph_input(1, 0);
    baseline.control.nph = 0;
    baseline.control.ispec = 1;
    baseline.grid.xkmax = 1000.0;
    baseline.lmaxph = vec![1];
    baseline.pot_labels = vec!["Cu".to_string()];
    baseline.spinph = vec![0.0];

    let mut filtered = baseline.clone();
    filtered.control.l2lp = 1;
    let mut nrixs = baseline.clone();
    nrixs.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    let mut nrixs_jas = nrixs.clone();
    nrixs_jas.grid.xkmax = -10.0;
    let global_nrixs = tempfile::tempdir()?;
    let global_nrixs_caches = super::XsphCachePaths::new(global_nrixs.path());
    let global_nrixs_input = baseline.clone();
    write_global_input_custom(global_nrixs.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;

    let baseline_mesh = super::generate_initial_phase_mesh_from_pot(&caches, &baseline, &pot)?;
    let filtered_mesh = super::generate_initial_phase_mesh_from_pot(&caches, &filtered, &pot)?;
    let nrixs_mesh = super::generate_initial_phase_mesh_from_pot(&caches, &nrixs, &pot)?;
    let nrixs_jas_mesh = super::generate_initial_phase_mesh_from_pot(&caches, &nrixs_jas, &pot)?;
    let global_nrixs_mesh = super::generate_initial_phase_mesh_from_pot(
        &global_nrixs_caches,
        &global_nrixs_input,
        &pot,
    )?;

    assert_eq!(
        filtered_mesh.horizontal_count,
        super::XSPH_COMPILED_PHASE_MESH_CAPACITY
    );
    assert_eq!(
        filtered_mesh.horizontal_count,
        baseline_mesh.horizontal_count
    );
    assert_eq!(filtered_mesh.energies, baseline_mesh.energies);
    assert_eq!(
        nrixs_mesh.horizontal_count,
        super::XSPH_NRIXS_PHASE_MESH_CAPACITY
    );
    assert_eq!(
        global_nrixs_mesh.horizontal_count,
        super::XSPH_NRIXS_PHASE_MESH_CAPACITY
    );
    assert_eq!(
        nrixs_jas_mesh.horizontal_count,
        super::XSPH_NRIXS_PHASE_MESH_CAPACITY
    );
    assert!(nrixs_jas_mesh.energies.len() > nrixs_jas_mesh.horizontal_count);
    assert_eq!(nrixs_jas_mesh.fermi_index_1based, 11);
    assert_ne!(nrixs_jas_mesh.energies, nrixs_mesh.energies);
    Ok(())
}

#[test]
fn normal_xsect_phiscf_occupied_table_maps_config_slots_to_pot_energies() -> Result<()> {
    let mut pot = sample_normal_phase_pot_bin();
    pot.orbital_energies =
        Array1::from_shape_fn(POT_BIN_ORBITALS, |slot| -0.75 + 0.031 * slot as f64);
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let active_orbital_count = orbital_tables.bound_orbital_counts[0];

    let table = normal_xsect_phiscf_occupied_table(&pot, &orbital_tables, 0, active_orbital_count)?;

    assert_eq!(
        table.orbital_energy_counts.to_vec(),
        vec![1; active_orbital_count]
    );
    assert_eq!(table.occupied_energies.dim(), (1, active_orbital_count));
    assert_eq!(table.occupation_fractions.dim(), (1, active_orbital_count));
    for orbital_index in 0..active_orbital_count {
        let slot = orbital_tables.orbital_slots_by_potential[(orbital_index, 0)];
        assert_eq!(
            table.occupied_energies[(0, orbital_index)],
            pot.orbital_energies[slot]
        );

        let electron_count = orbital_tables.electron_counts_by_potential[(orbital_index, 0)];
        let kappa = orbital_tables.kappa_by_potential[(orbital_index, 0)];
        let shell_capacity = 2.0 * f64::from(kappa.unsigned_abs());
        assert_eq!(
            table.occupation_fractions[(0, orbital_index)],
            electron_count / shell_capacity
        );
    }
    Ok(())
}

#[test]
fn normal_xsect_phiscf_occupied_table_prefers_explicit_valence_response_rows() -> Result<()> {
    let mut pot = sample_normal_phase_pot_bin();
    pot.orbital_energies =
        Array1::from_shape_fn(POT_BIN_ORBITALS, |slot| -0.75 + 0.031 * slot as f64);
    let mut config = sample_normal_phase_config_dat();
    config.potentials[0].valence_occupations[2] = 0.35;
    config.potentials[0].valence_occupations[3] = 0.25;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)?;
    let active_orbital_count = orbital_tables.bound_orbital_counts[0];

    let table = normal_xsect_phiscf_occupied_table(&pot, &orbital_tables, 0, active_orbital_count)?;

    assert_eq!(table.orbital_energy_counts.to_vec(), vec![0, 0, 1, 1]);
    assert_eq!(table.occupied_energies[(0, 0)], 0.0);
    assert_eq!(table.occupied_energies[(0, 1)], 0.0);
    for orbital_index in [2, 3] {
        let slot = orbital_tables.orbital_slots_by_potential[(orbital_index, 0)];
        assert_eq!(
            table.occupied_energies[(0, orbital_index)],
            pot.orbital_energies[slot]
        );

        let valence_count = orbital_tables.valence_counts_by_potential[(orbital_index, 0)];
        let kappa = orbital_tables.kappa_by_potential[(orbital_index, 0)];
        let shell_capacity = 2.0 * f64::from(kappa.unsigned_abs());
        assert_eq!(
            table.occupation_fractions[(0, orbital_index)],
            valence_count / shell_capacity
        );
    }
    Ok(())
}

#[test]
fn normal_xsect_phiscf_occupied_table_rejects_out_of_range_orbital_slot() -> Result<()> {
    let pot = sample_normal_phase_pot_bin();
    let mut orbital_tables =
        rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let active_orbital_count = orbital_tables.bound_orbital_counts[0];
    orbital_tables.orbital_slots_by_potential[(0, 0)] = pot.orbital_energies.len();

    let error = normal_xsect_phiscf_occupied_table(&pot, &orbital_tables, 0, active_orbital_count)
        .err()
        .context("invalid orbital slot should be rejected")?;

    assert!(
        error.to_string().contains("exceeds pot.bin eorb length"),
        "{error}"
    );
    Ok(())
}

#[test]
fn normal_xsect_phiscf_coarse_count_covers_active_prefix() -> Result<()> {
    assert_eq!(normal_xsect_phiscf_coarse_count_for_active_len(1, 1)?, 1);
    assert_eq!(normal_xsect_phiscf_fine_len_for_coarse_count(1)?, 1);
    assert_eq!(normal_xsect_phiscf_coarse_count_for_active_len(11, 11)?, 3);
    assert_eq!(normal_xsect_phiscf_fine_len_for_coarse_count(3)?, 11);
    assert_eq!(normal_xsect_phiscf_coarse_count_for_active_len(12, 16)?, 4);
    assert_eq!(normal_xsect_phiscf_fine_len_for_coarse_count(4)?, 16);

    let error = normal_xsect_phiscf_coarse_count_for_active_len(12, 15)
        .err()
        .context("coarse grid should reject insufficient radial capacity")?;
    assert!(
        error.to_string().contains("exceeds radial capacity"),
        "{error}"
    );
    Ok(())
}

#[test]
fn normal_xsect_hole_orbital_energy_maps_ihole_to_eorb_slot() -> Result<()> {
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 2;
    pot.orbital_energies[1] = -0.37;

    assert_eq!(normal_xsect_hole_orbital_energy(&pot)?, -0.37);

    pot.ihole = 0;
    let error = normal_xsect_hole_orbital_energy(&pot)
        .err()
        .context("nonpositive ihole should be rejected")?;
    assert!(error.to_string().contains("must be positive"), "{error}");
    Ok(())
}

#[test]
fn normal_xsect_phiscf_wfirdc_assembly_prepares_source_rows() -> Result<()> {
    let mut pot = sample_normal_phase_pot_bin();
    pot.orbital_energies =
        Array1::from_shape_fn(POT_BIN_ORBITALS, |slot| -0.75 + 0.031 * slot as f64);
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_normal_phase_config_dat())?;
    let active_orbital_count = orbital_tables.bound_orbital_counts[0];
    let occupied_table =
        normal_xsect_phiscf_occupied_table(&pot, &orbital_tables, 0, active_orbital_count)?;

    let active_len = 220;
    let step = 0.05;
    let radii = Array1::from_shape_fn(active_len, |row| (-8.8 + step * row as f64).exp());
    let exchange_correlation_potential = Array1::from_shape_fn(active_len, |row| {
        Complex64::new(-0.45 + 0.0007 * row as f64, -0.025)
    });
    let local_field = Array1::from_shape_fn(active_len, |row| 0.03 + 0.0001 * row as f64);
    let bound_large_components = pot.large_components.index_axis(Axis(2), 0);
    let bound_small_components = pot.small_components.index_axis(Axis(2), 0);
    let bound_large_coefficients = pot.large_coefficients.index_axis(Axis(2), 0);
    let bound_small_coefficients = pot.small_coefficients.index_axis(Axis(2), 0);
    let electron_counts = orbital_tables
        .electron_counts_by_potential
        .index_axis(Axis(1), 0);
    let valence_counts = pot.orbital_occupancy.index_axis(Axis(1), 0);
    let kappa = orbital_tables.kappa_by_potential.index_axis(Axis(1), 0);

    let assembly = normal_xsect_phiscf_wfirdc_assembly(NormalXsectPhiscfWfirdcAssemblyInput {
        momentum_squared: Complex64::new(1.25, 0.25),
        edge_energy: 0.0,
        chemical_potential: 1.0,
        hole_orbital_energy: pot.orbital_energies[0],
        scale_function: 0.75,
        occupied_table: &occupied_table,
        orbital_kappas: kappa,
        radii: radii.view(),
        exchange_correlation_potential: exchange_correlation_potential.view(),
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        electron_counts,
        valence_counts,
        local_field: local_field.view(),
        nuclear_charge: pot.atomic_numbers[0] as f64,
        muffin_tin_radius: pot.muffin_tin_radii[0],
        step,
        target_last_index_1based: active_len,
        active_len,
        coarse_count: 3,
        c3_scale: 1,
    })?;

    assert!(!assembly.rows.is_empty());
    assert_eq!(assembly.bound_orbital_count, active_orbital_count);
    assert_eq!(
        assembly.bound_large_coefficients.dim(),
        (POT_BIN_COEFFICIENTS, active_orbital_count)
    );

    let first = &assembly.rows[0];
    assert_eq!(first.kappa.len(), active_orbital_count + 1);
    assert_eq!(
        first.kappa[active_orbital_count],
        first.plan_row.final_kappa
    );
    assert_eq!(first.orbital_lengths[active_orbital_count], active_len);
    assert!(
        first.c3_potential.iter().any(|value| value.norm() > 0.0),
        "C3 potential should be source-backed and nonzero before the match row"
    );

    let contribution = assembly.contribution_input(0)?;
    let orbital_index = first.plan_row.orbital_index_1based - 1;
    assert_eq!(contribution.coarse_count, 3);
    assert_eq!(contribution.wave_number, first.radial_setup.wave_number);
    assert_eq!(contribution.response_scale, first.plan_row.rule.scale);
    assert_eq!(
        contribution.include_response_imaginary,
        first.plan_row.rule.include_imaginary
    );
    assert_eq!(
        contribution.orbital_large[0],
        assembly.bound_large_components[(0, orbital_index)]
    );
    assert_eq!(
        contribution.orbital_small[0],
        assembly.bound_small_components[(0, orbital_index)]
    );
    assert_eq!(
        contribution.wfirdc_input.energy,
        first.plan_row.pole.pole_energy
    );
    assert_eq!(contribution.wfirdc_input.c3_scale, 1);
    assert_eq!(
        contribution.wfirdc_input.muffin_tin_radius,
        first.radial_setup.match_radius
    );
    assert_eq!(
        contribution.wfirdc_input.radial_match_index,
        first.radial_setup.match_index
    );
    assert_eq!(
        contribution.wfirdc_input.wkb_index,
        first.radial_setup.wkb_index
    );
    assert_eq!(
        contribution.wfirdc_input.kappa[active_orbital_count],
        first.plan_row.final_kappa
    );
    assert_eq!(
        contribution.wfirdc_input.orbital_lengths[active_orbital_count],
        active_len
    );
    assert_eq!(
        contribution.wfirdc_input.exchange_correlation_potential[assembly.active_len - 1],
        contribution.wfirdc_input.exchange_correlation_potential
            [first.radial_setup.match_index + 1]
    );
    let mut collector_assembly = assembly.clone();
    collector_assembly.rows.truncate(1);
    let basis_fields = Array2::<Complex64>::zeros((active_len, 0));
    let collected =
        collector_assembly.collect_wfirdc_contributions(radii.view(), basis_fields.view(), 0)?;

    assert_eq!(collected.contributions.len(), 1);
    assert_eq!(collected.screened_solution.screened_field.len(), 11);
    assert_eq!(
        collected.screened_solution.screened_basis_fields.dim(),
        (11, 0)
    );
    assert!(
        collected
            .screened_solution
            .screened_field
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    Ok(())
}

#[test]
fn xsph_module_generates_reference_normal_phase_from_pot_and_config_without_cache() -> Result<()> {
    let Some(reference_dir) = reference_xsph_with_pot_config_dir()? else {
        crate::require_fixture!(
            "XSPH normal phase reference test; generated EXAFS/Cu reference not found"
        );
    };

    assert_reference_normal_phase_and_xsect_from_pot_config(
        &reference_dir,
        ReferenceNormalPhaseXsectTolerance::default(),
    )
}

#[test]
fn xsph_module_generates_reference_xanes_mesh_and_xsect_from_pot_and_config_without_cache()
-> Result<()> {
    let Some(reference_dir) = reference_xanes_xsph_with_pot_config_dir()? else {
        crate::require_fixture!(
            "XSPH XANES phase/xsect reference test; generated XANES/Cu reference not found"
        );
    };

    assert_reference_normal_phase_and_xsect_from_pot_config(
        &reference_dir,
        ReferenceNormalPhaseXsectTolerance::default(),
    )
}

#[test]
fn xsph_module_generates_reference_danes_mesh_and_xsect_from_pot_and_config_without_cache()
-> Result<()> {
    let Some(reference_dir) = reference_danes_xsph_with_pot_config_dir()? else {
        crate::require_fixture!(
            "XSPH DANES phase/xsect reference test; generated DANES/Cu reference not found"
        );
    };

    assert_reference_normal_phase_and_xsect_from_pot_config(
        &reference_dir,
        ReferenceNormalPhaseXsectTolerance {
            background_absolute: 5.0e-5,
            cross_section_absolute: 1.0e-4,
            ..ReferenceNormalPhaseXsectTolerance::default()
        },
    )
}

#[test]
fn xsph_module_generates_reference_fprime_phase_and_xsect_from_pot_and_config_without_cache()
-> Result<()> {
    let Some(reference_dir) = reference_fprime_xsph_with_pot_config_dir()? else {
        crate::require_fixture!(
            "XSPH FPRIME phase/xsect reference test; generated FPRIME/GeCl4 reference not found"
        );
    };

    assert_reference_normal_phase_and_xsect_from_pot_config(
        &reference_dir,
        ReferenceNormalPhaseXsectTolerance {
            background_absolute: 5.0e-5,
            cross_section_absolute: 5.0e-5,
            mpse_required: false,
            ..ReferenceNormalPhaseXsectTolerance::default()
        },
    )
}

#[test]
fn xsph_module_generates_reference_xes_cu_phase_and_xsect_from_zip_pot_config_without_cache()
-> Result<()> {
    let Some(zip_path) = reference_xes_cu_xsph_zip()? else {
        crate::require_fixture!("XSPH XES/Cu reference test; reference zip not found");
    };
    let reference = tempfile::tempdir()?;
    write_reference_zip_entries(
        &zip_path,
        reference.path(),
        [
            "xsph.inp",
            "global.inp",
            "pot.bin",
            "config.dat",
            "wscrn.dat",
            "phase.bin",
            "xsect.dat",
        ],
    )?;

    assert_reference_normal_phase_and_xsect_from_pot_config(
        reference.path(),
        ReferenceNormalPhaseXsectTolerance {
            phase_shift: 1.0e-4,
            ..ReferenceNormalPhaseXsectTolerance::default()
        },
    )
}

#[test]
fn xsph_module_matches_broader_source_generated_reference_when_present() -> Result<()> {
    let fixtures = reference_xsph_source_release_fixtures()?;
    if fixtures.is_empty() {
        crate::require_fixture!("XSPH source release gate; reference source bundles not found");
    }

    for fixture in fixtures {
        eprintln!("checking XSPH source fixture {}", fixture.label);
        assert_reference_normal_phase_and_xsect_from_source_fixture(&fixture)
            .with_context(|| format!("XSPH source fixture {} failed", fixture.label))?;
    }
    Ok(())
}

#[test]
fn xsph_module_generates_elnes_cu_positive_atomic_background_from_reference() -> Result<()> {
    let fixture = reference_xsph_source_release_fixtures()?
        .into_iter()
        .find(|fixture| fixture.label == "ELNES/Cu");
    let Some(fixture) = fixture else {
        crate::require_fixture!("XSPH ELNES/Cu source fixture not found");
    };
    assert_reference_normal_phase_and_xsect_from_source_fixture(&fixture)
        .context("XSPH ELNES/Cu positive atomic-background regression")
}

#[test]
fn xsph_exchange_vr0_shifts_edge_and_chemical_potential_but_preserves_relative_mesh() -> Result<()>
{
    const VR0_EV: f64 = 1.0;
    let baseline = tempfile::tempdir()?;
    let shifted = tempfile::tempdir()?;
    write_normal_xsph_source_with_vr0(baseline.path(), 0.0)?;
    write_normal_xsph_source_with_vr0(shifted.path(), VR0_EV)?;

    run_in_dir(baseline.path())?;
    run_in_dir(shifted.path())?;

    let baseline_phase = read_phase_bin(baseline.path().join("phase.bin"))?;
    let shifted_phase = read_phase_bin(shifted.path().join("phase.bin"))?;
    let baseline_xsect = read_xsect_dat(baseline.path().join("xsect.dat"))?;
    let shifted_xsect = read_xsect_dat(shifted.path().join("xsect.dat"))?;
    let shift_hartree = VR0_EV / FEFF_HARTREE_EV;

    assert!(
        (shifted_phase.scalars.edge_energy - baseline_phase.scalars.edge_energy + shift_hartree)
            .abs()
            <= 1.0e-10
    );
    assert_eq!(
        baseline_phase.energy_grid.len(),
        shifted_phase.energy_grid.len()
    );
    for (baseline_energy, shifted_energy) in baseline_phase
        .energy_grid
        .iter()
        .zip(shifted_phase.energy_grid.iter())
    {
        let baseline_relative = baseline_energy.re - baseline_phase.scalars.edge_energy;
        let shifted_relative = shifted_energy.re - shifted_phase.scalars.edge_energy;
        assert!((baseline_relative - shifted_relative).abs() <= 1.0e-10);
    }
    assert!(
        (shifted_xsect.scalars.chemical_potential - baseline_xsect.scalars.chemical_potential
            + shift_hartree)
            .abs()
            <= 5.0e-8,
        "xsect emu shift was {}, expected {}",
        shifted_xsect.scalars.chemical_potential - baseline_xsect.scalars.chemical_potential,
        -shift_hartree
    );

    let baseline_handoff = xsect_dat_ff2x_handoff(&baseline_xsect, 0.0, 0)?;
    let shifted_handoff = xsect_dat_ff2x_handoff(&shifted_xsect, 0.0, 0)?;
    for (baseline_omega, shifted_omega) in baseline_handoff
        .omega_hartree
        .iter()
        .zip(shifted_handoff.omega_hartree.iter())
    {
        assert!((shifted_omega - baseline_omega + shift_hartree).abs() <= 1.0e-8);
    }
    Ok(())
}

#[test]
fn xsph_module_bn_xsect_keeps_feff_photon_prefactor_and_ixc0_transition_moments() -> Result<()> {
    let workspace = reference_workspace()?;
    let archive = workspace.join("reference-work/golden/XANES/BN/REFERENCE.zip");
    if !archive.is_file() {
        crate::require_fixture!(
            "XSPH BN photon-prefactor regression; XANES/BN reference zip not found"
        );
    }

    let generated = tempfile::tempdir()?;
    write_reference_zip_required_entries(
        &archive,
        generated.path(),
        ["xsph.inp", "global.inp", "pot.bin", "config.dat"],
    )?;
    let expected_dir = tempfile::tempdir()?;
    write_reference_zip_required_entries(
        &archive,
        expected_dir.path(),
        ["phase.bin", "xsect.dat"],
    )?;
    let expected = read_xsect_dat(expected_dir.path().join("xsect.dat"))?;
    let expected_phase = read_phase_bin(expected_dir.path().join("phase.bin"))?;

    run_in_dir(generated.path())?;
    let actual = read_xsect_dat(generated.path().join("xsect.dat"))?;
    let phase = read_phase_bin(generated.path().join("phase.bin"))?;
    let pot = read_pot_bin(generated.path().join("pot.bin"))?;
    assert_eq!(actual.energy_count(), expected.energy_count());
    assert_eq!(actual.main_energy_count, expected.main_energy_count);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_eq!(
        phase.transition_moments.dim(),
        expected_phase.transition_moments.dim()
    );

    let first_photon_energy =
        phase.energy_grid[0].re - phase.scalars.edge_energy + pot.scalars.edge_position;
    assert!(
        (6.0..7.0).contains(&first_photon_energy),
        "BN first-row omega must use raw Hartree emu, not a second eV-to-Hartree conversion: \
         em={:.12e}, edge={:.12e}, emu={:.12e}, omega={first_photon_energy:.12e}",
        phase.energy_grid[0].re,
        phase.scalars.edge_energy,
        pot.scalars.edge_position
    );

    // The FEFF xsect prefactor already contains 1/omega.  A second photon-energy
    // ratio in the file handoff tilts this otherwise flat comparison by roughly
    // 30% from the first row to the top of the main mesh.
    let mesh_indices = [
        0,
        expected.fermi_index.saturating_sub(1),
        expected.main_energy_count.saturating_sub(1),
    ];
    for energy_index in mesh_indices {
        let expected_norm = expected.normalized_background[energy_index];
        assert!(
            expected_norm > 0.0,
            "BN reference xsnorm row {} must be nonzero",
            energy_index + 1
        );
        let ratio = actual.normalized_background[energy_index] / expected_norm;
        assert!(
            (ratio - 1.0).abs() <= 0.03,
            "BN xsnorm row {} lost FEFF's photon-energy prefactor: \
             actual={:.12e}, expected={:.12e}, ratio={ratio:.9}",
            energy_index + 1,
            actual.normalized_background[energy_index],
            expected_norm
        );
    }

    // FEFF deliberately evaluates the cross-section transition moments with
    // ixc0, while the scattering phase shifts use ixc.  BN has ixc=0 and
    // ixc0=2, so using the phase selector here introduces a large, complex
    // self-energy into the RKK values above the Fermi level.
    let representative_energy_index = 51;
    for transition_index in 0..2 {
        let actual_moment =
            phase.transition_moments[(representative_energy_index, 0, transition_index, 0)];
        let expected_moment = expected_phase.transition_moments
            [(representative_energy_index, 0, transition_index, 0)];
        let difference = (actual_moment - expected_moment).norm();
        let tolerance = 5.0e-4 * expected_moment.norm().max(1.0);
        assert!(
            difference <= tolerance,
            "BN transition moment row {}, channel {} must use ixc0: \
             actual={actual_moment:?}, expected={expected_moment:?}, \
             difference={difference:.6e}, tolerance={tolerance:.6e}",
            representative_energy_index + 1,
            transition_index + 1
        );
    }
    Ok(())
}

#[test]
fn xsph_module_generates_legacy_archive_without_config_dat_when_present() -> Result<()> {
    let workspace = reference_workspace()?;
    let archive = workspace.join("reference-work/golden/XANES/GeCl_4/REFERENCE.zip");
    if !archive.is_file() {
        crate::require_fixture!(
            "legacy XSPH no-config source test; XANES/GeCl_4 archive not found"
        );
    }

    let temp = tempfile::tempdir()?;
    write_reference_zip_required_entries(
        &archive,
        temp.path(),
        ["xsph.inp", "global.inp", "pot.bin"],
    )?;
    let expected = tempfile::tempdir()?;
    std::fs::write(
        expected.path().join("phase.bin"),
        unzip_reference_entry(&archive, "REFERENCE/phase.bin")?,
    )?;
    let expected_phase = read_phase_bin(expected.path().join("phase.bin"))?;

    assert!(!temp.path().join("config.dat").exists());
    assert!(has_supported_xsph_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.spin_count, 1);
    assert_eq!(phase.potential_count(), 2);
    assert_eq!(phase.energy_count, expected_phase.energy_count);
    assert_eq!(phase.main_energy_count, expected_phase.main_energy_count);
    assert_eq!(
        phase.auxiliary_energy_count,
        expected_phase.auxiliary_energy_count
    );
    assert_eq!(phase.fermi_index, expected_phase.fermi_index);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(phase.potentials.iter().any(|potential| {
        potential
            .phase_shifts
            .iter()
            .any(|phase_shift| phase_shift.norm() > 0.0)
    }));
    assert!(xsect.cross_section.iter().any(|value| value.norm() > 0.0));
    Ok(())
}

#[test]
fn xsph_module_generates_mnf2_xmcd_phase_ltot_capacity_from_zip_pot_without_cache() -> Result<()> {
    let workspace = reference_workspace()?;
    let archive = workspace.join("reference-work/golden/XMCD/MnF2_SPXAS/REFERENCE.zip");
    if !archive.is_file() {
        crate::require_fixture!("XMCD MnF2 phase source test; XMCD/MnF2_SPXAS archive not found");
    }

    let temp = tempfile::tempdir()?;
    write_reference_zip_required_entries(
        &archive,
        temp.path(),
        ["xsph.inp", "global.inp", "pot.bin"],
    )?;
    let expected = tempfile::tempdir()?;
    write_reference_zip_required_entries(&archive, expected.path(), ["phase.bin", "xsect.dat"])?;
    let expected_phase = read_phase_bin(expected.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(expected.path().join("xsect.dat"))?;

    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.spin_count, expected_phase.spin_count);
    assert_eq!(phase.energy_count, expected_phase.energy_count);
    assert_eq!(phase.main_energy_count, expected_phase.main_energy_count);
    assert_eq!(
        phase.auxiliary_energy_count,
        expected_phase.auxiliary_energy_count
    );
    assert_eq!(phase.ihole, expected_phase.ihole);
    assert_eq!(phase.fermi_index, expected_phase.fermi_index);
    assert_eq!(phase.potential_count(), 4);
    assert_eq!(phase.potential_count(), expected_phase.potential_count());
    for (actual, expected) in phase
        .potentials
        .iter()
        .zip(expected_phase.potentials.iter())
    {
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_eq!(actual.lmax, expected.lmax);
        assert!(actual.lmax <= 24);
    }
    assert!(
        phase.potentials.iter().any(|potential| potential.lmax > 20),
        "XMCD MnF2 should exercise FEFF ltot capacity above the old Rust cap"
    );
    assert_complex_column_close(&phase.energy_grid, &expected_phase.energy_grid, 1.0e-10);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert_xmcd_xsect_reference_close(&xsect, &expected_xsect);
    Ok(())
}

#[test]
fn xsph_module_generates_gd_l1_xmcd_phase_radial_capacity_from_zip_pot_without_cache() -> Result<()>
{
    let workspace = reference_workspace()?;
    let archive = workspace.join("reference-work/golden/XMCD/Gd_L1/REFERENCE.zip");
    if !archive.is_file() {
        crate::require_fixture!("XMCD Gd L1 phase source test; XMCD/Gd_L1 archive not found");
    }

    let temp = tempfile::tempdir()?;
    write_reference_zip_required_entries(
        &archive,
        temp.path(),
        ["xsph.inp", "global.inp", "pot.bin"],
    )?;
    let expected = tempfile::tempdir()?;
    write_reference_zip_required_entries(&archive, expected.path(), ["phase.bin", "xsect.dat"])?;
    let input = super::read_input(temp.path())?;
    let expected_phase = read_phase_bin(expected.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(expected.path().join("xsect.dat"))?;

    assert!((input.grid.rgrd - 0.01).abs() <= 1.0e-12);
    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(phase.spin_count, expected_phase.spin_count);
    assert_eq!(phase.energy_count, expected_phase.energy_count);
    assert_eq!(phase.main_energy_count, expected_phase.main_energy_count);
    assert_eq!(
        phase.auxiliary_energy_count,
        expected_phase.auxiliary_energy_count
    );
    assert_eq!(phase.ihole, expected_phase.ihole);
    assert_eq!(phase.fermi_index, expected_phase.fermi_index);
    assert_eq!(phase.potential_count(), 2);
    assert_eq!(phase.potential_count(), expected_phase.potential_count());
    for (actual, expected) in phase
        .potentials
        .iter()
        .zip(expected_phase.potentials.iter())
    {
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_eq!(actual.lmax, expected.lmax);
        assert!(actual.lmax <= 24);
    }
    assert_complex_column_close(&phase.energy_grid, &expected_phase.energy_grid, 1.0e-10);
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
    assert_xmcd_xsect_reference_close(&xsect, &expected_xsect);
    Ok(())
}

#[test]
fn xsph_module_matches_current_mnf2_xmcd_phase_and_xsect() -> Result<()> {
    let Some(reference) = current_generated_xsph_reference("XMCD/MnF2_SPXAS")? else {
        crate::require_fixture!("current XMCD/MnF2_SPXAS generated reference not found");
    };
    assert_current_xmcd_phase_and_xsect(&reference, 1.0e-4)
}

#[test]
fn xsph_module_matches_current_gd_l1_xmcd_phase_and_xsect() -> Result<()> {
    let Some(reference) = current_generated_xsph_reference("XMCD/Gd_L1")? else {
        crate::require_fixture!("current XMCD/Gd_L1 generated reference not found");
    };
    assert_current_xmcd_phase_and_xsect(&reference, 1.0e-4)
}

#[test]
fn xsph_module_keeps_current_bn_ordinary_xsect_on_unpolarized_spin_path() -> Result<()> {
    let Some(reference) = current_generated_xsph_reference("XANES/BN")? else {
        crate::require_fixture!("current XANES/BN generated reference not found");
    };
    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "eels.inp",
        "pot.bin",
        "pot.inp",
        "geom.dat",
        "config.dat",
        "wscrn.dat",
    ] {
        let source = reference.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    let expected_phase = read_phase_bin(reference.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference.join("xsect.dat"))?;

    run_in_dir(temp.path())?;

    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_eq!(phase.spin_count, 1);
    assert_reference_phase_close(&phase, &expected_phase, 1.0e-4);
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), expected_xsect.energy_count());
    assert_complex_column_close(
        &xsect.energy_grid_ev,
        &expected_xsect.energy_grid_ev,
        1.0e-8,
    );
    assert_column_close_mixed(
        &xsect.normalized_background,
        &expected_xsect.normalized_background,
        1.1e-6,
        2.5e-5,
    );
    assert_complex_column_close_mixed(
        &xsect.cross_section,
        &expected_xsect.cross_section,
        1.1e-6,
        2.5e-5,
    );
    Ok(())
}

fn assert_current_xmcd_phase_and_xsect(reference: &Path, phase_shift_tolerance: f64) -> Result<()> {
    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "eels.inp",
        "pot.bin",
        "pot.inp",
        "geom.dat",
        "config.dat",
        "wscrn.dat",
    ] {
        let source = reference.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    let expected_phase = read_phase_bin(reference.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference.join("xsect.dat"))?;

    let written = run_in_dir(temp.path())?;

    assert!(written >= 4);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_reference_phase_close(&phase, &expected_phase, phase_shift_tolerance);
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_xmcd_xsect_reference_close(&xsect, &expected_xsect);
    assert_current_xmcd_spin_columns_close(&xsect, &expected_xsect);
    Ok(())
}

fn assert_current_xmcd_spin_columns_close(actual: &XsectDatData, expected: &XsectDatData) {
    let expected_real_peak = expected
        .cross_section
        .iter()
        .map(|value| value.re.abs())
        .fold(0.0_f64, f64::max);
    let expected_imaginary_peak = expected
        .cross_section
        .iter()
        .map(|value| value.im.abs())
        .fold(0.0_f64, f64::max);
    let actual_real_peak = actual
        .cross_section
        .iter()
        .map(|value| value.re.abs())
        .fold(0.0_f64, f64::max);
    let actual_imaginary_peak = actual
        .cross_section
        .iter()
        .map(|value| value.im.abs())
        .fold(0.0_f64, f64::max);

    assert!(
        expected_real_peak > 0.0 && expected_imaginary_peak > 0.0,
        "pinned XMCD fixture must exercise both magnetic xsect columns"
    );
    assert!(
        actual_real_peak > 0.0 && actual_imaginary_peak > 0.0,
        "generated XMCD xsect must retain both magnetic columns: real peak={actual_real_peak}, imaginary peak={actual_imaginary_peak}"
    );
    assert_column_close_mixed(
        &actual.normalized_background,
        &expected.normalized_background,
        1.1e-8,
        2.5e-5,
    );
    assert_complex_column_close_mixed(
        &actual.cross_section,
        &expected.cross_section,
        1.1e-8,
        2.5e-5,
    );
}

#[derive(Debug, Clone, Copy)]
struct ReferenceNormalPhaseXsectTolerance {
    phase_shift: f64,
    background_absolute: f64,
    background_relative: f64,
    cross_section_absolute: f64,
    cross_section_relative: f64,
    sidecar_absolute: f64,
    sidecar_relative: f64,
    mpse_required: bool,
}

impl Default for ReferenceNormalPhaseXsectTolerance {
    fn default() -> Self {
        Self {
            phase_shift: 5.0e-5,
            background_absolute: 1.0e-5,
            background_relative: 0.20,
            cross_section_absolute: 2.0e-5,
            cross_section_relative: 0.25,
            sidecar_absolute: 2.0e-5,
            sidecar_relative: 0.25,
            mpse_required: true,
        }
    }
}

#[derive(Debug, Clone)]
struct ReferenceXsphSourceFixture {
    label: &'static str,
    source: ReferenceXsphSource,
    tolerance: ReferenceNormalPhaseXsectTolerance,
}

#[derive(Debug, Clone)]
enum ReferenceXsphSource {
    Directory(PathBuf),
    Zip(PathBuf),
}

fn reference_xsph_source_release_fixtures() -> Result<Vec<ReferenceXsphSourceFixture>> {
    let workspace = reference_workspace()?;
    let directory_fixtures = [
        (
            "DEBYE/DM/EXAFS/Cu",
            "reference-work/golden/DEBYE/DM/EXAFS/Cu",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
        (
            "DEBYE/DM/XANES/Cu",
            "reference-work/golden/DEBYE/DM/XANES/Cu",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
        (
            "ELNES/Cu",
            "reference-work/golden/ELNES/Cu",
            ReferenceNormalPhaseXsectTolerance {
                background_absolute: 3.0e-4,
                cross_section_absolute: 3.0e-4,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
        (
            "EXAFS/Cu_SCF",
            "reference-work/golden/EXAFS/Cu_SCF",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
        (
            "LDOS/XANES_Cu_fms",
            "reference-work/golden/LDOS/XANES_Cu_fms",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
        (
            "LDOS/XANES_Cu_spin_fms_short",
            "reference-work/golden/LDOS/XANES_Cu_spin_fms_short",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
        (
            "LDOS/XANES_Cu_spin_no_fms",
            "reference-work/golden/LDOS/XANES_Cu_spin_no_fms",
            ReferenceNormalPhaseXsectTolerance::default(),
        ),
    ]
    .into_iter()
    .filter_map(|(label, relative, tolerance)| {
        let path = workspace.join(relative);
        has_xsph_source_reference(&path).then_some(ReferenceXsphSourceFixture {
            label,
            source: ReferenceXsphSource::Directory(path),
            tolerance,
        })
    });
    let zip_fixtures = [
        (
            "XANES/BN",
            "reference-work/golden/XANES/BN/REFERENCE.zip",
            ReferenceNormalPhaseXsectTolerance {
                background_absolute: 5.0e-5,
                background_relative: 0.30,
                cross_section_absolute: 2.0e-4,
                cross_section_relative: 0.30,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
        (
            "XANES/GeCl_4",
            "reference-work/golden/XANES/GeCl_4/REFERENCE.zip",
            ReferenceNormalPhaseXsectTolerance {
                mpse_required: false,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
        (
            "XES/BN",
            "reference-work/golden/XES/BN/REFERENCE.zip",
            ReferenceNormalPhaseXsectTolerance {
                phase_shift: 1.0e-4,
                background_absolute: 5.0e-5,
                background_relative: 0.30,
                cross_section_absolute: 2.0e-4,
                cross_section_relative: 0.30,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
        (
            "XES/GeCl_4",
            "reference-work/golden/XES/GeCl_4/REFERENCE.zip",
            ReferenceNormalPhaseXsectTolerance {
                phase_shift: 1.0e-4,
                mpse_required: false,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
        (
            "NRIXS/GeCl_4",
            "reference-work/golden/NRIXS/GeCl_4/REFERENCE.zip",
            ReferenceNormalPhaseXsectTolerance {
                background_absolute: 5.0e-4,
                background_relative: 0.30,
                cross_section_absolute: 5.0e-4,
                cross_section_relative: 0.30,
                sidecar_absolute: 1.5e-3,
                sidecar_relative: 0.30,
                mpse_required: false,
                ..ReferenceNormalPhaseXsectTolerance::default()
            },
        ),
    ]
    .into_iter()
    .filter_map(|(label, relative, tolerance)| {
        let path = workspace.join(relative);
        path.is_file().then_some(ReferenceXsphSourceFixture {
            label,
            source: ReferenceXsphSource::Zip(path),
            tolerance,
        })
    });

    Ok(directory_fixtures.chain(zip_fixtures).collect())
}

#[test]
fn xsph_module_generates_nrixs_reference_outputs_from_pot_and_config_without_cache() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_xsph_with_pot_and_emesh_dir()? else {
        crate::require_fixture!(
            "XSPH NRIXS phase reference test; generated NRIXS/GeCl_4 reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin", "config.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let written = run_required_in_dir(temp.path())?;

    assert!(written >= 6);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_reference_phase_close(
        &phase,
        &expected_phase,
        ReferenceNormalPhaseXsectTolerance::default().phase_shift,
    );
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
            || xsect.cross_section.iter().any(|value| value.norm() > 0.0),
        "expected source-generated NRIXS/JAS xsect.dat rows"
    );
    assert!(temp.path().join("xsecl.dat").is_file());
    assert!(temp.path().join("xsecl2.dat").is_file());
    assert!(temp.path().join("xsecl.bin").is_file());
    assert_emesh_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        1.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_matches_nrixs_mgb2_phase_xsect_and_sidecars() -> Result<()> {
    let Some(reference) = current_generated_xsph_reference("NRIXS/MgB2")? else {
        crate::require_fixture!("XSPH NRIXS/MgB2 current generated reference not found");
    };
    assert_reference_normal_phase_and_xsect_from_source_fixture(&ReferenceXsphSourceFixture {
        label: "NRIXS/MgB2",
        source: ReferenceXsphSource::Directory(reference),
        tolerance: ReferenceNormalPhaseXsectTolerance {
            phase_shift: 1.0e-4,
            background_absolute: 5.0e-4,
            background_relative: 0.30,
            cross_section_absolute: 5.0e-4,
            cross_section_relative: 0.30,
            sidecar_absolute: 1.5e-3,
            sidecar_relative: 0.30,
            mpse_required: false,
        },
    })
}

fn reference_workspace() -> Result<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("failed to find workspace root")
}

fn current_generated_xsph_reference(relative: &str) -> Result<Option<PathBuf>> {
    let workspace = reference_workspace()?;
    for root in ["reference-work/golden", "reference-work/tmp/pinned-golden"] {
        let path = workspace.join(root).join(relative);
        if [
            "xsph.inp",
            "global.inp",
            "pot.bin",
            "pot.inp",
            "geom.dat",
            "config.dat",
            "phase.bin",
            "xsect.dat",
        ]
        .iter()
        .all(|name| path.join(name).is_file())
        {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn has_xsph_source_reference(path: &Path) -> bool {
    [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "pot.inp",
        "geom.dat",
        "config.dat",
        "phase.bin",
        "xsect.dat",
    ]
    .iter()
    .all(|name| path.join(name).is_file())
}

fn assert_reference_phase_close(
    phase: &PhaseBinData,
    expected_phase: &PhaseBinData,
    phase_shift_tolerance: f64,
) {
    assert_eq!(phase.spin_count, expected_phase.spin_count);
    assert_eq!(phase.energy_count, expected_phase.energy_count);
    assert_eq!(phase.main_energy_count, expected_phase.main_energy_count);
    assert_eq!(
        phase.auxiliary_energy_count,
        expected_phase.auxiliary_energy_count
    );
    assert_eq!(phase.ihole, expected_phase.ihole);
    assert_eq!(phase.fermi_index, expected_phase.fermi_index);
    assert_eq!(phase.potential_count(), expected_phase.potential_count());
    for (actual, expected) in phase
        .potentials
        .iter()
        .zip(expected_phase.potentials.iter())
    {
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_eq!(actual.lmax, expected.lmax);
    }
    let (reference_energy_delta, reference_energy_location) =
        max_complex_array2_delta_with_location(
            &phase.reference_energy,
            &expected_phase.reference_energy,
        );
    assert!(
        reference_energy_delta <= 2.0e-7,
        "reference-energy delta {reference_energy_delta} at {reference_energy_location:?} exceeds tolerance 2e-7: actual={:?}, expected={:?}",
        phase.reference_energy[reference_energy_location],
        expected_phase.reference_energy[reference_energy_location]
    );
    assert_complex_column_close(&phase.energy_grid, &expected_phase.energy_grid, 1.0e-10);
    let (phase_shift_delta, phase_shift_location) =
        max_phase_shift_delta_with_location(phase, expected_phase, 0, phase.energy_count);
    assert!(
        phase_shift_delta <= phase_shift_tolerance,
        "phase-shift delta {phase_shift_delta} at {phase_shift_location:?} exceeds tolerance {phase_shift_tolerance}: actual={:?}, expected={:?}",
        phase.potentials[phase_shift_location.0].phase_shifts[(
            phase_shift_location.1,
            phase_shift_location.2,
            phase_shift_location.3,
        )],
        expected_phase.potentials[phase_shift_location.0].phase_shifts[(
            phase_shift_location.1,
            phase_shift_location.2,
            phase_shift_location.3,
        )]
    );
    assert!(
        phase
            .potentials
            .iter()
            .flat_map(|potential| potential.phase_shifts.iter())
            .any(|phase_shift| phase_shift.norm() > 0.0)
    );
}

fn assert_reference_normal_phase_and_xsect_from_pot_config(
    reference_dir: &Path,
    tolerance: ReferenceNormalPhaseXsectTolerance,
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    for name in [
        "xsph.inp",
        "global.inp",
        "eels.inp",
        "pot.bin",
        "pot.inp",
        "geom.dat",
        "config.dat",
        "wscrn.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    let expected_phase = read_phase_bin(reference_dir.join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let expected_xsecl = optional_xsecl_dat(reference_dir.join("xsecl.dat"))?;
    let expected_xsecl2 = optional_xsecl2_dat(reference_dir.join("xsecl2.dat"))?;
    let expected_xsecl_bin = optional_xsecl_bin(
        reference_dir.join("xsecl.bin"),
        expected_phase.pad_width,
        expected_phase.energy_count,
    )?;
    let input = super::read_input(temp.path())?;

    let written = run_in_dir(temp.path())?;

    let generated_nrixs_sidecar_count = ["xsecl.dat", "xsecl2.dat", "xsecl.bin"]
        .iter()
        .filter(|name| temp.path().join(name).is_file())
        .count();
    assert_eq!(
        written,
        5 + usize::from(input.control.ipr2 >= 1)
            + usize::from(tolerance.mpse_required)
            + generated_nrixs_sidecar_count
    );
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    assert_reference_phase_close(&phase, &expected_phase, tolerance.phase_shift);
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_eq!(xsect.energy_count(), expected_xsect.energy_count());
    assert_eq!(xsect.main_energy_count, expected_xsect.main_energy_count);
    assert_eq!(xsect.fermi_index, expected_xsect.fermi_index);
    assert_complex_column_close(
        &xsect.energy_grid_ev,
        &expected_xsect.energy_grid_ev,
        1.0e-8,
    );
    assert_column_close_mixed(
        &xsect.normalized_background,
        &expected_xsect.normalized_background,
        tolerance.background_absolute,
        tolerance.background_relative,
    );
    assert_complex_column_close_mixed(
        &xsect.cross_section,
        &expected_xsect.cross_section,
        tolerance.cross_section_absolute,
        tolerance.cross_section_relative,
    );
    let eels_path = reference_dir.join("eels.inp");
    if eels_path.is_file() {
        let text = std::fs::read_to_string(&eels_path)?;
        let eels = EelsInput::parse_str(&eels_path, &text)?;
        if eels.calculation_mode == 1 {
            let maximum_atomic_background = xsect
                .cross_section
                .iter()
                .take(xsect.main_energy_count)
                .map(|value| value.im)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                maximum_atomic_background.is_finite() && maximum_atomic_background > 0.0,
                "ELNES XSPH atomic cross section must contain a positive background, got maximum imaginary xsec {maximum_atomic_background}"
            );
        }
    }
    if input.control.ipr2 >= 1 {
        assert!(temp.path().join("axafs.dat").is_file());
    }
    if tolerance.mpse_required {
        assert!(temp.path().join("mpse.dat").is_file());
    }
    if let Some(expected) = &expected_xsecl {
        assert_xsecl_dat_close(
            &read_xsecl_dat(temp.path().join("xsecl.dat"))?,
            expected,
            tolerance.sidecar_absolute,
            tolerance.sidecar_relative,
        );
    }
    if let Some(expected) = &expected_xsecl2 {
        assert_xsecl_dat_close(
            &read_xsecl2_dat(temp.path().join("xsecl2.dat"))?,
            expected,
            tolerance.sidecar_absolute,
            tolerance.sidecar_relative,
        );
    }
    if let Some(expected) = &expected_xsecl_bin {
        assert_xsecl_bin_close(
            &read_xsecl_bin(
                temp.path().join("xsecl.bin"),
                expected_phase.pad_width,
                expected_phase.energy_count,
            )?,
            expected,
            tolerance.sidecar_absolute,
            tolerance.sidecar_relative,
        );
    }
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

fn assert_xmcd_xsect_reference_close(actual: &XsectDatData, expected: &XsectDatData) {
    assert_eq!(actual.energy_count(), expected.energy_count());
    assert_eq!(actual.main_energy_count, expected.main_energy_count);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_complex_column_close(&actual.energy_grid_ev, &expected.energy_grid_ev, 1.0e-8);
    assert_column_close_mixed(
        &actual.normalized_background,
        &expected.normalized_background,
        1.0e-4,
        1.0e-8,
    );
    assert_complex_column_close_mixed(
        &actual.cross_section,
        &expected.cross_section,
        1.0e-4,
        1.0e-8,
    );
}

fn assert_reference_normal_phase_and_xsect_from_source_fixture(
    fixture: &ReferenceXsphSourceFixture,
) -> Result<()> {
    match &fixture.source {
        ReferenceXsphSource::Directory(path) => {
            assert_reference_normal_phase_and_xsect_from_pot_config(path, fixture.tolerance)
        }
        ReferenceXsphSource::Zip(path) => {
            let reference = tempfile::tempdir()?;
            write_reference_zip_required_entries(
                path,
                reference.path(),
                [
                    "xsph.inp",
                    "global.inp",
                    "pot.bin",
                    "phase.bin",
                    "xsect.dat",
                ],
            )?;
            write_reference_zip_optional_entries(
                path,
                reference.path(),
                [
                    "config.dat",
                    "eels.inp",
                    "wscrn.dat",
                    "xsecl.dat",
                    "xsecl2.dat",
                    "xsecl.bin",
                ],
            )?;
            assert_reference_normal_phase_and_xsect_from_pot_config(
                reference.path(),
                fixture.tolerance,
            )
        }
    }
}

#[test]
fn xsph_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let axafs_path = temp.path().join("axafs.dat");
    let xsecl_path = temp.path().join("xsecl.dat");
    let xsecl2_path = temp.path().join("xsecl2.dat");
    let xsecl_bin_path = temp.path().join("xsecl.bin");
    let mpse_path = temp.path().join("mpse.dat");
    let emesh_path = temp.path().join("emesh.dat");
    let emesh_bin_path = temp.path().join("emesh.bin");
    let log2_path = temp.path().join("log2.dat");

    let phase = sample_phase_bin();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(&phase_path, &phase)?;
    write_xsect_dat(&xsect_path, &xsect)?;
    write_axafs_dat(&axafs_path, &sample_axafs_dat())?;
    let xsecl = sample_xsecl_dat_for_phase(&phase)?;
    write_xsecl_dat(&xsecl_path, &xsecl)?;
    write_xsecl2_dat(&xsecl2_path, &xsecl)?;
    write_xsecl_bin(&xsecl_bin_path, &sample_xsecl_bin())?;
    write_mpse_dat(&mpse_path, &sample_mpse_dat())?;
    write_emesh_dat(&emesh_path, &sample_emesh_dat())?;
    write_emesh_bin(&emesh_bin_path, &sample_emesh_bin())?;
    write_module_log_dat(&log2_path, &sample_module_log())?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_axafs = read_axafs_dat(&axafs_path)?;
    let expected_xsecl = read_xsecl_dat(&xsecl_path)?;
    let expected_xsecl2 = read_xsecl2_dat(&xsecl2_path)?;
    let expected_xsecl_bin = read_xsecl_bin(
        &xsecl_bin_path,
        expected_phase.pad_width,
        expected_phase.energy_count,
    )?;
    let expected_mpse = read_mpse_dat(&mpse_path)?;
    let expected_emesh = read_emesh_dat(&emesh_path)?;
    let expected_emesh_bin = read_emesh_bin(&emesh_bin_path)?;
    let expected_log = read_module_log_dat(&log2_path)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 10);
    assert!(has_cached_xsph_output(temp.path())?);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, expected_xsect);
    assert_eq!(read_axafs_dat(&axafs_path)?, expected_axafs);
    assert_eq!(read_xsecl_dat(&xsecl_path)?, expected_xsecl);
    assert_eq!(read_xsecl2_dat(&xsecl2_path)?, expected_xsecl2);
    assert_eq!(
        read_xsecl_bin(
            &xsecl_bin_path,
            expected_phase.pad_width,
            expected_phase.energy_count
        )?,
        expected_xsecl_bin
    );
    assert_eq!(read_mpse_dat(&mpse_path)?, expected_mpse);
    assert_eq!(read_emesh_dat(&emesh_path)?, expected_emesh);
    assert_eq!(read_emesh_bin(&emesh_bin_path)?, expected_emesh_bin);
    assert_eq!(read_module_log_dat(&log2_path)?, expected_log);
    Ok(())
}

#[test]
fn xsph_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;
    std::fs::write(temp.path().join("log2.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(!has_cached_xsph_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed log2.dat should fail through cached XSPH runner")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("log2.dat"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_does_not_advertise_malformed_phase_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(!has_supported_phase_mesh_handoff(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed phase.bin should fail through the explicit XSPH runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_does_not_advertise_malformed_xsect_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    std::fs::write(temp.path().join("xsect.dat"), "not xsect.dat\n")?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed xsect.dat should fail through the explicit XSPH runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsect.dat"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_does_not_preserve_xsecl_cache_with_mismatched_phase_mesh() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let mut xsecl = sample_xsecl_dat();
    xsecl.header.real_energy_count += 1;
    write_xsecl_dat(temp.path().join("xsecl.dat"), &xsecl)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("stale xsecl.dat should fail against the active phase mesh")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("xsecl.dat"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    assert!(chain.contains("real energy count"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_requires_complete_nrixs_xsecl_sidecars_for_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("NRIXS cached output should require complete xsectjas sidecars")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("NRIXS"), "{chain}");
    assert!(chain.contains("xsecl.dat"), "{chain}");
    assert!(chain.contains("xsecl2.dat"), "{chain}");
    assert!(chain.contains("xsecl.bin"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_accepts_complete_nrixs_xsecl_sidecars_for_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    let xsecl = sample_xsecl_dat_for_phase(&phase)?;
    write_xsecl_dat(temp.path().join("xsecl.dat"), &xsecl)?;
    write_xsecl2_dat(temp.path().join("xsecl2.dat"), &xsecl)?;
    write_xsecl_bin(temp.path().join("xsecl.bin"), &sample_xsecl_bin())?;

    assert!(has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);

    let count = run_required_in_dir(temp.path())?;

    assert!(count >= 5);
    assert!(has_supported_xsph_output(temp.path())?);
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_xsectjas_sidecars_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ispec = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.le2 = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
        global.control.ldecmx = 1;
        global.control.lj = 1;
        global.q_control.nq = 1;
        global.q_control.qaverage = false;
        global.q_vectors = vec![refeff_io::GlobalQVector {
            q: [0.0, 0.0, 1.25],
            norm: 1.25,
            weight: [1.0, 0.0],
            trig: [-1.0, 0.0, 1.0, 0.0],
        }];
    })?;
    write_grid_inp(
        temp.path().join("grid.inp"),
        &sample_single_point_grid_input(),
    )?;
    let source_pot = sample_normal_phase_pot_bin();
    write_pot_bin(temp.path().join("pot.bin"), &source_pot)?;
    write_config_dat(
        temp.path().join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;

    assert!(has_supported_xsph_output(temp.path())?);

    let written = run_required_in_dir(temp.path())?;

    assert!(written >= 6);
    assert!(has_supported_xsph_output(temp.path())?);
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    let xsecl = read_xsecl_dat(temp.path().join("xsecl.dat"))?;
    let xsecl2 = read_xsecl2_dat(temp.path().join("xsecl2.dat"))?;
    let xsecl_bin = read_xsecl_bin(
        temp.path().join("xsecl.bin"),
        phase.pad_width,
        phase.energy_count,
    )?;
    assert_eq!(phase.q_count, 1);
    assert_eq!(phase.transition_count, 3);
    assert_eq!(phase.final_state_count, 24);
    assert_eq!(xsect.energy_count(), phase.energy_count);
    assert_eq!(xsecl.row_count(), phase.energy_count);
    assert_eq!(xsecl2.row_count(), phase.energy_count);
    assert_eq!(xsecl.channel_count(), 2);
    assert_eq!(xsecl2.channel_count(), 2);
    assert_eq!(xsecl_bin.energy_count(), phase.energy_count);
    assert_eq!(xsecl_bin.transition_index_count(), phase.transition_count);
    assert!(
        phase
            .transition_moments
            .iter()
            .any(|value| value.norm() > 0.0),
        "expected source-generated NRIXS/JAS phase transition moments"
    );
    assert!(
        xsect
            .normalized_background
            .iter()
            .any(|value| value.abs() > 0.0)
            || xsect.cross_section.iter().any(|value| value.norm() > 0.0),
        "expected source-generated NRIXS/JAS xsect.dat rows"
    );
    assert!(
        xsecl
            .channel_cross_sections
            .iter()
            .chain(xsecl2.channel_cross_sections.iter())
            .chain(xsecl_bin.atom_cross_sections.iter())
            .any(|value| value.norm() > 0.0),
        "expected source-generated NRIXS/JAS sidecar spectra"
    );
    Ok(())
}

#[test]
fn xsph_nrixs_phase_transition_dimensions_use_global_handoff_controls() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let caches = super::XsphCachePaths::new(temp.path());
    let mut input = sample_xsph_input(1, 0);
    input.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    let pot = sample_normal_phase_pot_bin();
    let mut global = sample_global_input(3, super::XSPH_NRIXS_L2LP_SENTINEL);
    global.control.do_nrixs = 1;
    global.control.ldecmx = 3;
    global.control.lj = 3;
    global.q_control.nq = 2;
    global.q_vectors = vec![
        refeff_io::GlobalQVector {
            q: [0.0, 0.0, 2.0],
            norm: 2.0,
            weight: [0.25, 0.10],
            trig: [-1.0, 0.0, 1.0, 0.0],
        },
        refeff_io::GlobalQVector {
            q: [0.0, 1.0, 0.0],
            norm: 1.0,
            weight: [0.75, -0.20],
            trig: [0.0, 1.0, 1.0, 0.0],
        },
    ];
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;

    let dimensions = phase_transition_dimensions(&caches, &input, &pot)?;

    assert_eq!(dimensions.final_state_count, 48);
    assert_eq!(dimensions.transition_count, 7);
    assert_eq!(dimensions.q_count, 2);
    Ok(())
}

#[test]
fn xsph_nrixs_spectrum_source_plan_reconstructs_handoff_work_arrays() -> Result<()> {
    let input = sample_xsph_input(1, 0);
    let mut phase = sample_phase_bin();
    phase.transition_count = 3;
    phase.final_state_count = 24;
    phase.transition_moments = Array4::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    let mut global = sample_global_input(1, 30);
    global.control.do_nrixs = 1;
    global.control.ldecmx = 1;
    global.control.lj = 1;
    global.q_control.nq = 1;
    global.q_control.qaverage = false;
    global.q_vectors = vec![refeff_io::GlobalQVector {
        q: [0.0, 0.0, 2.0],
        norm: 2.0,
        weight: [0.25, 0.10],
        trig: [-1.0, 0.0, 1.0, 0.0],
    }];
    let radii = Array1::from_vec(vec![0.10, 0.20, 0.40]);

    let plan =
        nrixs_spectrum_source_plan_from_handoffs(&input, &global, &phase, Some(radii.view()))?;

    assert_eq!(plan.initial_kappa, -1);
    assert_eq!(plan.initial_state_j, 1);
    assert_eq!(plan.max_angular_momentum, 24);
    assert_eq!(plan.final_lj_max, 1);
    assert_eq!(plan.final_state_count, 24);
    assert_eq!(plan.active_len, 3);
    assert_eq!(plan.kind.to_vec(), vec![-1, 1, -2]);
    assert_eq!(plan.decomposition_l.to_vec(), vec![0, 1, 1]);
    assert_eq!(plan.final_lj.to_vec(), vec![0, 1, 1]);
    assert_eq!(plan.orbital_l.to_vec(), vec![0, 1, 1]);
    assert_eq!(plan.calculation_plan.index_map.to_vec(), vec![1, 2, 3]);
    assert_eq!(plan.lj_needed_by_calculation[0].to_vec(), vec![1, 0]);
    assert_eq!(plan.lj_needed_by_calculation[1].to_vec(), vec![0, 1]);
    assert_eq!(plan.transitions.len(), 3);
    assert_eq!(plan.transitions[0].final_state_kappa, -1);
    assert_eq!(plan.transitions[2].total_angular_momentum_channel, 1);
    assert_eq!(plan.q_weights[0], Complex64::new(0.25, 0.10));
    assert_eq!(plan.q_cosines.dim(), (1, 1));
    assert_eq!(plan.q_cosines[(0, 0)], 1.0);
    assert_eq!(plan.transition_weights.dim(), (2, 3, 2));
    let q_bessel = plan.q_bessel.context("expected q-Bessel tables")?;
    assert_eq!(q_bessel.dim(), (3, 2, 1));
    assert!(q_bessel[(0, 0, 0)].is_finite());
    Ok(())
}

#[test]
fn xsph_nrixs_radial_source_context_preserves_jas_core_spinor() -> Result<()> {
    let input = sample_xsph_input(1, 0);
    let mut phase = sample_phase_bin();
    phase.transition_count = 3;
    phase.final_state_count = 24;
    phase.transition_moments = Array4::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    let mut global = sample_global_input(1, 30);
    global.control.do_nrixs = 1;
    global.control.ldecmx = 1;
    global.control.lj = 1;
    global.q_control.nq = 1;
    global.q_vectors = vec![refeff_io::GlobalQVector {
        q: [0.0, 0.0, 1.5],
        norm: 1.5,
        weight: [1.0, 0.0],
        trig: [-1.0, 0.0, 1.0, 0.0],
    }];
    let radii = Array1::from_vec(vec![0.10, 0.15, 0.22, 0.33, 0.49, 0.72, 1.06]);
    let initial_large = Array1::from_shape_fn(radii.len(), |index| 0.21 + 0.015 * index as f64);
    let initial_small = Array1::from_shape_fn(radii.len(), |index| 0.08 + 0.006 * index as f64);
    let plan =
        nrixs_spectrum_source_plan_from_handoffs(&input, &global, &phase, Some(radii.view()))?;

    let context = nrixs_spectrum_radial_source_context_from_handoffs(
        &plan,
        0,
        initial_large.view(),
        initial_small.view(),
        radii.view(),
        0.05,
        radii.len(),
        5,
    )?;

    assert_eq!(context.active_radial_len, 5);
    assert_eq!(context.initial_l, 0);
    assert_eq!(context.initial_large.len(), 5);
    assert_eq!(context.initial_small.len(), 5);
    assert_eq!(
        context.radii.to_vec(),
        radii
            .slice_axis(Axis(0), ndarray::Slice::from(..5))
            .to_vec()
    );
    assert_eq!(context.q_bessel.dim(), (5, 2, 1));
    assert_eq!(context.orthogonality_correction.dim(), (2, 1));
    assert!(context.hole_normalization > 0.0);
    assert!((context.initial_large[0] - initial_large[0]).abs() < 1.0e-12);
    assert!(context.orthogonality_normalization.re.is_finite());
    assert!(context.orthogonality_normalization.im.is_finite());
    assert!(
        context
            .orthogonality_correction
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    Ok(())
}

#[test]
fn xsph_nrixs_spectrum_row_assembles_jas_radial_channels() -> Result<()> {
    let input = sample_xsph_input(1, 0);
    let mut phase = sample_phase_bin();
    phase.transition_count = 3;
    phase.final_state_count = 24;
    phase.transition_moments = Array4::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    let mut global = sample_global_input(1, 30);
    global.control.do_nrixs = 1;
    global.control.ldecmx = 1;
    global.control.lj = 1;
    global.q_control.nq = 1;
    global.q_vectors = vec![refeff_io::GlobalQVector {
        q: [0.0, 0.0, 1.25],
        norm: 1.25,
        weight: [0.8, 0.1],
        trig: [-1.0, 0.0, 1.0, 0.0],
    }];
    let radii = Array1::from_vec(vec![0.10, 0.15, 0.22, 0.33, 0.49, 0.72, 1.06]);
    let initial_large = Array1::from_shape_fn(radii.len(), |index| 0.19 + 0.013 * index as f64);
    let initial_small = Array1::from_shape_fn(radii.len(), |index| 0.07 + 0.004 * index as f64);
    let plan =
        nrixs_spectrum_source_plan_from_handoffs(&input, &global, &phase, Some(radii.view()))?;
    let context = nrixs_spectrum_radial_source_context_from_handoffs(
        &plan,
        0,
        initial_large.view(),
        initial_small.view(),
        radii.view(),
        0.05,
        radii.len(),
        5,
    )?;
    let calculation_count = plan.calculation_plan.calculations.nrows();
    let regular_large = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.020 * calc * row, -0.002 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let regular_small = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.007 * calc * row, 0.0015 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let irregular_large = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.014 * calc * row, 0.003 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let irregular_small = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.005 * calc * row, -0.001 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let channels = (0..calculation_count)
        .map(|calculation| NrixsSpectrumRadialChannel {
            final_kappa: plan.calculation_plan.calculations[(calculation, 0)],
            phase_shift: Complex64::new(0.03 * (calculation + 1) as f64, 0.0),
            regular_large: regular_large[calculation].view(),
            regular_small: regular_small[calculation].view(),
            irregular_large: irregular_large[calculation].view(),
            irregular_small: irregular_small[calculation].view(),
        })
        .collect::<Vec<_>>();

    let row = nrixs_spectrum_row_from_radial_channels(&plan, &context, &channels, 0, false, 0)?;

    assert_eq!(row.decomposition_cross_sections.len(), 2);
    assert_eq!(row.total_angular_cross_sections.len(), 2);
    assert_eq!(row.atom_cross_sections.len(), 24);
    assert!(row.total_spectrum_norm.is_finite());
    assert!(
        row.decomposition_cross_sections
            .iter()
            .chain(row.total_angular_cross_sections.iter())
            .chain(row.atom_cross_sections.iter())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        row.decomposition_cross_sections
            .iter()
            .chain(row.total_angular_cross_sections.iter())
            .chain(row.atom_cross_sections.iter())
            .any(|value| value.norm() > 0.0)
    );
    let rows = vec![row.clone(), row];
    let chemical_potential_ev = 12.5;
    let handoffs =
        nrixs_spectrum_handoffs_from_rows(&phase, &plan, &rows, chemical_potential_ev, 0.05)?;
    assert_eq!(handoffs.xsecl.row_count(), phase.energy_count);
    assert_eq!(handoffs.xsecl.channel_count(), 2);
    assert_eq!(handoffs.xsecl2.channel_count(), 2);
    assert_eq!(handoffs.xsecl.header.emu, chemical_potential_ev);
    assert_eq!(handoffs.xsecl2.header.emu, chemical_potential_ev);
    assert_eq!(handoffs.xsecl_bin.energy_count(), phase.energy_count);
    assert_eq!(
        handoffs.xsecl_bin.final_state_count(),
        phase.final_state_count
    );
    assert_eq!(
        handoffs.xsecl_bin.transition_index_count(),
        phase.transition_count
    );
    Ok(())
}

#[test]
fn xsph_nrixs_spectrum_row_assembles_multiple_q_vectors() -> Result<()> {
    let input = sample_xsph_input(1, 0);
    let mut phase = sample_phase_bin();
    phase.q_count = 2;
    phase.transition_count = 3;
    phase.final_state_count = 24;
    phase.transition_moments = Array4::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    let mut global = sample_global_input(1, 30);
    global.control.do_nrixs = 1;
    global.control.ldecmx = 1;
    global.control.lj = 1;
    global.q_control.nq = 2;
    global.q_control.qaverage = false;
    global.q_vectors = vec![
        refeff_io::GlobalQVector {
            q: [0.0, 0.0, 1.25],
            norm: 1.25,
            weight: [0.6, 0.0],
            trig: [-1.0, 0.0, 1.0, 0.0],
        },
        refeff_io::GlobalQVector {
            q: [0.0, 1.75, 0.0],
            norm: 1.75,
            weight: [0.4, 0.0],
            trig: [0.0, 1.0, 1.0, 0.0],
        },
    ];
    let radii = Array1::from_vec(vec![0.10, 0.15, 0.22, 0.33, 0.49, 0.72, 1.06]);
    let initial_large = Array1::from_shape_fn(radii.len(), |index| 0.19 + 0.013 * index as f64);
    let initial_small = Array1::from_shape_fn(radii.len(), |index| 0.07 + 0.004 * index as f64);
    let plan =
        nrixs_spectrum_source_plan_from_handoffs(&input, &global, &phase, Some(radii.view()))?;
    let context = nrixs_spectrum_radial_source_context_from_handoffs(
        &plan,
        0,
        initial_large.view(),
        initial_small.view(),
        radii.view(),
        0.05,
        radii.len(),
        5,
    )?;
    assert_eq!(context.q_bessel.dim(), (5, 2, 2));
    assert_eq!(context.orthogonality_correction.dim(), (2, 2));

    let calculation_count = plan.calculation_plan.calculations.nrows();
    let regular_large = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.020 * calc * row, -0.002 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let regular_small = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.007 * calc * row, 0.0015 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let irregular_large = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.014 * calc * row, 0.003 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let irregular_small = (0..calculation_count)
        .map(|calculation| {
            Array1::from_shape_fn(context.active_radial_len, |index| {
                let row = (index + 1) as f64;
                let calc = (calculation + 1) as f64;
                Complex64::new(0.005 * calc * row, -0.001 * calc * row)
            })
        })
        .collect::<Vec<_>>();
    let channels = (0..calculation_count)
        .map(|calculation| NrixsSpectrumRadialChannel {
            final_kappa: plan.calculation_plan.calculations[(calculation, 0)],
            phase_shift: Complex64::new(0.03 * (calculation + 1) as f64, 0.0),
            regular_large: regular_large[calculation].view(),
            regular_small: regular_small[calculation].view(),
            irregular_large: irregular_large[calculation].view(),
            irregular_small: irregular_small[calculation].view(),
        })
        .collect::<Vec<_>>();

    let row = nrixs_spectrum_row_from_radial_channels(&plan, &context, &channels, 0, false, 0)?;

    assert_eq!(row.decomposition_cross_sections.len(), 2);
    assert_eq!(row.total_angular_cross_sections.len(), 2);
    assert_eq!(row.atom_cross_sections.len(), 24);
    assert!(row.total_spectrum_norm.is_finite());
    assert!(
        row.decomposition_cross_sections
            .iter()
            .chain(row.total_angular_cross_sections.iter())
            .chain(row.atom_cross_sections.iter())
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        row.decomposition_cross_sections
            .iter()
            .chain(row.total_angular_cross_sections.iter())
            .chain(row.atom_cross_sections.iter())
            .any(|value| value.norm() > 0.0)
    );
    let rows = vec![row.clone(), row];
    let handoffs = nrixs_spectrum_handoffs_from_rows(&phase, &plan, &rows, 12.5, 0.05)?;
    assert_eq!(handoffs.xsecl.row_count(), phase.energy_count);
    assert_eq!(
        handoffs.xsecl_bin.final_state_count(),
        phase.final_state_count
    );
    Ok(())
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_text_energy_grid() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |xsecl, _, _| {
            xsecl.energy[0] += 1.0;
        },
        &["energy grid", "phase.bin"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_malformed_text() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let malformed_xsecl = b"not an xsecl.dat cache\n";
    std::fs::write(temp.path().join("xsecl.dat"), malformed_xsecl)?;
    let xsecl2 = sample_xsecl_dat_for_phase(&phase)?;
    write_xsecl2_dat(temp.path().join("xsecl2.dat"), &xsecl2)?;
    write_xsecl_bin(temp.path().join("xsecl.bin"), &sample_xsecl_bin())?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("malformed NRIXS xsectjas text sidecar should fail the direct XSPH runner")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsecl.dat"), "{chain}");
    assert_eq!(
        std::fs::read(temp.path().join("xsecl.dat"))?,
        malformed_xsecl
    );
    Ok(())
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_secondary_text_energy_grid() -> Result<()>
{
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, xsecl2, _| {
            xsecl2.energy[0] += 1.0;
        },
        &["energy grid", "phase.bin"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_malformed_secondary_text() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let xsecl = sample_xsecl_dat_for_phase(&phase)?;
    write_xsecl_dat(temp.path().join("xsecl.dat"), &xsecl)?;
    let malformed_xsecl2 = b"not an xsecl2.dat cache\n";
    std::fs::write(temp.path().join("xsecl2.dat"), malformed_xsecl2)?;
    write_xsecl_bin(temp.path().join("xsecl.bin"), &sample_xsecl_bin())?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_required_in_dir(temp.path()).err().context(
        "malformed NRIXS xsectjas secondary text sidecar should fail the direct XSPH runner",
    )?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsecl2.dat"), "{chain}");
    assert_eq!(
        std::fs::read(temp.path().join("xsecl2.dat"))?,
        malformed_xsecl2
    );
    Ok(())
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_text_sum() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |xsecl, _, _| {
            xsecl.channel_sum[0] += Complex64::new(0.01, -0.02);
        },
        &["channel sum", "channel columns"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_secondary_text_sum() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, xsecl2, _| {
            xsecl2.channel_sum[0] += Complex64::new(0.01, -0.02);
        },
        &["channel sum", "channel columns"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_mismatched_text_channels() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, xsecl2, _| {
            let rows = xsecl2.row_count();
            xsecl2.channel_cross_sections = Array2::from_shape_fn((rows, 1), |(energy, _)| {
                xsecl2.channel_cross_sections[(energy, 0)]
            });
            xsecl2.channel_sum = xsecl2.channel_cross_sections.sum_axis(Axis(1));
        },
        &["channel count", "xsecl2.dat"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_mismatched_text_header() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, xsecl2, _| {
            xsecl2.header.edge += 0.125;
        },
        &["edge", "xsecl2.dat"],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_bin_transition_count() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, _, xsecl_bin| {
            xsecl_bin.transitions.pop();
        },
        &[
            "xsecl.bin",
            "transition index count",
            "phase.bin transition count",
        ],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_stale_bin_final_state_count() -> Result<()> {
    assert_rejects_invalid_nrixs_xsectjas_sidecars(
        |_, _, xsecl_bin| {
            let atom_cross_sections = xsecl_bin.atom_cross_sections.clone();
            xsecl_bin.atom_cross_sections = Array2::from_shape_fn(
                (
                    atom_cross_sections.len_of(Axis(0)),
                    atom_cross_sections.len_of(Axis(1)) - 1,
                ),
                |(energy, final_state)| atom_cross_sections[(energy, final_state)],
            );
        },
        &[
            "xsecl.bin",
            "final-state count",
            "phase.bin final-state count",
        ],
    )
}

#[test]
fn xsph_module_rejects_nrixs_xsectjas_sidecars_with_malformed_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let xsecl = sample_xsecl_dat_for_phase(&phase)?;
    write_xsecl_dat(temp.path().join("xsecl.dat"), &xsecl)?;
    write_xsecl2_dat(temp.path().join("xsecl2.dat"), &xsecl)?;
    let malformed_bin = b"not an xsecl.bin cache\n";
    std::fs::write(temp.path().join("xsecl.bin"), malformed_bin)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("malformed NRIXS xsectjas binary sidecar should fail the direct XSPH runner")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsecl.bin"), "{chain}");
    assert_eq!(std::fs::read(temp.path().join("xsecl.bin"))?, malformed_bin);
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_emesh_sidecar_from_phase_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let expected_emesh = expected_emesh_dat_from_phase(&phase, 0)?;
    let expected_emesh_bin = emesh_bin_from_phase_bin(&phase)?;
    std::fs::write(temp.path().join("emesh.dat"), "not emesh.dat\n")?;

    assert!(has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_eq!(
        read_emesh_dat(temp.path().join("emesh.dat"))?,
        expected_emesh
    );
    assert_eq!(
        read_emesh_bin(temp.path().join("emesh.bin"))?,
        expected_emesh_bin
    );
    Ok(())
}

#[test]
fn xsph_module_roundtrips_cached_hubbard_aphase_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let aphase_path = temp.path().join("aphase_hubbard.bin");

    let phase = sample_phase_bin();
    let aphase = sample_aphase_hubbard_bin(phase.energy_count, phase.potential_count());
    write_phase_bin(&phase_path, &phase)?;
    write_xsect_dat(&xsect_path, &sample_xsect_dat())?;
    write_aphase_hubbard_bin(&aphase_path, &aphase)?;

    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_aphase = read_aphase_hubbard_bin_inferred(
        &aphase_path,
        phase.energy_count,
        phase.potential_count(),
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(
        read_aphase_hubbard_bin_inferred(
            &aphase_path,
            phase.energy_count,
            phase.potential_count()
        )?,
        expected_aphase
    );
    Ok(())
}

#[test]
fn xsph_module_generates_emesh_sidecars_from_phase_only_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let emesh_path = temp.path().join("emesh.dat");
    let emesh_bin_path = temp.path().join("emesh.bin");

    write_phase_bin(&phase_path, &sample_phase_bin())?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_emesh = expected_emesh_dat_from_phase(&expected_phase, 0)?;
    let expected_emesh_bin = emesh_bin_from_phase_bin(&expected_phase)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!temp.path().join("xsect.dat").is_file());
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(read_emesh_dat(&emesh_path)?, expected_emesh);
    assert_eq!(read_emesh_bin(&emesh_bin_path)?, expected_emesh_bin);
    let log = read_module_log_dat(temp.path().join("log2.dat"))?;
    assert_log_contains(&log, "Calculating cross-section and phases ...");
    assert_log_contains(&log, "    absorption cross section");
    assert_log_contains(&log, "    phase shifts for unique potential    0");
    assert_log_contains(&log, "    phase shifts for unique potential    1");
    assert_log_contains(&log, "Done with module: cross-section and phases (XSPH).");
    Ok(())
}

#[test]
fn xsph_required_stage_rejects_phase_only_cache_after_refreshing_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;

    let error = run_required_in_dir(temp.path())
        .err()
        .context("required XSPH should reject phase-only caches")?;

    assert!(
        error
            .to_string()
            .contains("XSPH required stage needs complete phase.bin/xsect.dat caches"),
        "{error:?}"
    );
    assert!(read_emesh_dat(temp.path().join("emesh.dat"))?.point_count() > 0);
    assert!(read_emesh_bin(temp.path().join("emesh.bin"))?.point_count() > 0);
    assert!(temp.path().join("log2.dat").is_file());
    assert!(!temp.path().join("xsect.dat").exists());
    Ok(())
}

#[test]
fn xsph_module_skips_invalid_optional_mpse_generation_from_cached_phase_xsect() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");

    write_phase_bin(&phase_path, &sample_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_xsect_dat())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert!(has_cached_xsph_output(temp.path())?);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, expected_xsect);
    assert!(!temp.path().join("mpse.dat").is_file());
    assert!(temp.path().join("emesh.dat").is_file());
    assert!(temp.path().join("emesh.bin").is_file());
    assert!(temp.path().join("log2.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_phase_text_sidecars_when_print_requested() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 2)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_axafs_source_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 10);
    assert!(has_cached_xsph_output(temp.path())?);
    for name in ["phase00.dat", "phmin00.dat", "phase01.dat", "phmin01.dat"] {
        assert!(temp.path().join(name).is_file(), "missing {name}");
    }

    let phase00 = std::fs::read_to_string(temp.path().join("phase00.dat"))?;
    assert!(phase00.contains("unique pot,  lmax, ne"));
    assert!(phase00.contains("energy(eV)     re(eref)(eV)"));
    assert!(phase00.contains("  0.000000E+00"));

    let phmin00 = std::fs::read_to_string(temp.path().join("phmin00.dat"))?;
    assert!(phmin00.contains("phase(0) phase(1) phase(2)"));
    assert!(phmin00.contains("  0.00000E+00"));
    Ok(())
}

#[test]
fn xsph_module_generates_phase_text_handoff_from_phase_only_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 2)?;
    let phase_path = temp.path().join("phase.bin");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_text_handoff(temp.path())?);
    let count = run_supported_phase_text_handoff_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(!temp.path().join("emesh.dat").exists());
    assert!(!temp.path().join("emesh.bin").exists());
    for name in ["phase00.dat", "phmin00.dat", "phase01.dat", "phmin01.dat"] {
        assert!(temp.path().join(name).is_file(), "missing {name}");
    }
    assert!(!has_supported_phase_text_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_rewrites_stale_phase_text_handoff_from_phase_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 2)?;
    let phase_path = temp.path().join("phase.bin");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;
    std::fs::write(temp.path().join("phase00.dat"), "stale phase text\n")?;

    assert!(!has_supported_xsph_output(temp.path())?);
    assert!(has_supported_phase_text_handoff(temp.path())?);
    let count = run_supported_phase_text_handoff_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let phase00 = std::fs::read_to_string(temp.path().join("phase00.dat"))?;
    assert_ne!(phase00, "stale phase text\n");
    assert!(phase00.contains("unique pot,  lmax, ne"));
    assert!(!temp.path().join("xsect.dat").exists());
    assert!(!has_supported_phase_text_handoff(temp.path())?);
    Ok(())
}

#[test]
fn xsph_module_generates_reference_emesh_from_phase_cache() -> Result<()> {
    let Some(reference_dir) = reference_xsph_dir()? else {
        crate::require_fixture!(
            "XSPH emesh reference test; generated EXAFS/Cu reference not found"
        );
    };
    if !reference_dir.join("emesh.dat").is_file() || !reference_dir.join("emesh.bin").is_file() {
        crate::require_fixture!("XSPH emesh reference test; reference emesh sidecars not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "phase.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_eq!(
        read_emesh_dat(temp.path().join("emesh.dat"))?,
        expected_emesh
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_generates_reference_emesh_from_pot_before_source_requirement() -> Result<()> {
    let Some(reference_dir) = reference_xsph_with_pot_and_emesh_dir()? else {
        crate::require_fixture!(
            "XSPH pre-phase emesh reference test; generated EXAFS/Cu reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    assert!(!temp.path().join("config.dat").exists());
    read_phase_bin(temp.path().join("phase.bin"))?;
    read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_emesh_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        1.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_reference_emesh_from_pot_before_source_requirement() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_xsph_with_pot_and_emesh_dir()? else {
        crate::require_fixture!(
            "XSPH NRIXS pre-phase emesh reference test; generated NRIXS/GeCl_4 reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 4);
    assert!(!temp.path().join("config.dat").exists());
    read_phase_bin(temp.path().join("phase.bin"))?;
    assert!(!temp.path().join("xsect.dat").exists());
    assert_emesh_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        1.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_generates_fprime_reference_emesh_from_pot_before_source_requirement() -> Result<()> {
    let Some(reference_dir) = reference_fprime_xsph_with_pot_and_emesh_dir()? else {
        crate::require_fixture!(
            "XSPH FPRIME pre-phase emesh reference test; generated FPRIME/GeCl4 reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "global.inp", "pot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_emesh = read_emesh_dat(reference_dir.join("emesh.dat"))?;
    let expected_emesh_bin = read_emesh_bin(reference_dir.join("emesh.bin"))?;

    let written = run_in_dir(temp.path())?;

    assert!(written >= 5);
    assert!(!temp.path().join("config.dat").exists());
    read_phase_bin(temp.path().join("phase.bin"))?;
    read_xsect_dat(temp.path().join("xsect.dat"))?;
    assert_emesh_close(
        &read_emesh_dat(temp.path().join("emesh.dat"))?,
        &expected_emesh,
        1.0e-5,
    );
    assert_emesh_bin_close(
        &read_emesh_bin(temp.path().join("emesh.bin"))?,
        &expected_emesh_bin,
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_generates_user_grid_emesh_from_pot_before_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.i_grid = 1;
        input.control.ispec = 1;
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    write_grid_inp(temp.path().join("grid.inp"), &sample_xsph_grid_input())?;

    let error = run_in_dir(temp.path()).err().context(
        "XSPH should still require complete source handoff after user-grid pre-phase mesh generation",
    )?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.spectrum, 1);
    assert_eq!(emesh.fermi_index, 4);
    assert_eq!(emesh.point_count(), emesh_bin.point_count());
    assert_eq!(emesh_bin.point_count_declared, emesh_bin.point_count());
    assert_eq!(emesh_bin.horizontal_count, 9);
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert!(emesh_bin.point_count() > emesh_bin.horizontal_count);
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_jas_emesh_from_pot_before_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.ispec = 1;
        input.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
        input.grid.xkmax = -10.0;
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;

    let error = run_in_dir(temp.path()).err().context(
        "XSPH should still require complete source handoff after JAS/NRIXS pre-phase mesh generation",
    )?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.spectrum, 1);
    assert_eq!(emesh.fermi_index, 11);
    assert_eq!(emesh.point_count(), emesh_bin.point_count());
    assert_eq!(emesh_bin.point_count_declared, emesh_bin.point_count());
    assert_eq!(
        emesh_bin.horizontal_count,
        super::XSPH_NRIXS_PHASE_MESH_CAPACITY
    );
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert!(emesh_bin.point_count() > emesh_bin.horizontal_count);
    Ok(())
}

#[test]
fn xsph_module_generates_nrixs_user_grid_emesh_from_pot_before_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.i_grid = 1;
        input.control.ispec = 5;
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    write_grid_inp(temp.path().join("grid.inp"), &sample_xsph_grid_input())?;

    let error = run_in_dir(temp.path()).err().context(
        "XSPH should still require complete source handoff after NRIXS user-grid pre-phase mesh generation",
    )?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.spectrum, 5);
    assert_eq!(emesh.fermi_index, 4);
    assert_eq!(emesh.point_count(), 9);
    assert_eq!(emesh.point_count(), emesh_bin.point_count());
    assert_eq!(emesh_bin.point_count_declared, 9);
    assert_eq!(emesh_bin.horizontal_count, 9);
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert!(
        emesh_bin
            .energy_hartree
            .iter()
            .all(|energy| energy.im == 0.0)
    );
    Ok(())
}

#[test]
fn xsph_module_generates_thermal_user_grid_emesh_from_pot_before_source_requirement() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.i_grid = 1;
        input.control.ispec = 1;
        input.electronic_temperature = 5.0;
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    write_grid_inp(temp.path().join("grid.inp"), &sample_xsph_grid_input())?;

    let error = run_in_dir(temp.path()).err().context(
        "XSPH should still require complete source handoff after thermal pre-phase mesh generation",
    )?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.spectrum, 1);
    assert_eq!(emesh.fermi_index, 4);
    assert_eq!(emesh.point_count(), emesh_bin.point_count());
    assert_eq!(emesh_bin.point_count_declared, emesh_bin.point_count());
    assert_eq!(emesh_bin.horizontal_count, 9);
    assert_eq!(emesh_bin.danes_extension_count, 0);
    assert_eq!(emesh_bin.point_count(), 30);
    assert!(emesh_bin.energy_hartree[0].im > emesh_bin.energy_hartree[9].im);
    Ok(())
}

#[test]
fn xsph_module_generates_xes_emesh_from_pot_before_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_custom(temp.path(), |input| {
        input.control.ispec = 2;
        input.grid.xkmax = -5.0 / FEFF_BOHR_ANGSTROM;
        input.grid.xkstep = 10.0 / FEFF_BOHR_ANGSTROM;
        input.grid.vixan = 0.25 * FEFF_HARTREE_EV;
    })?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;

    let error = run_in_dir(temp.path()).err().context(
        "XSPH should still require complete source handoff after XES pre-phase mesh generation",
    )?;

    assert!(error.to_string().contains(XSPH_SOURCE_REQUIREMENT_ERROR));
    let emesh = read_emesh_dat(temp.path().join("emesh.dat"))?;
    let emesh_bin = read_emesh_bin(temp.path().join("emesh.bin"))?;
    assert_eq!(emesh.spectrum, 2);
    assert_eq!(emesh.point_count(), emesh_bin.point_count());
    assert_eq!(emesh_bin.point_count_declared, emesh_bin.point_count());
    assert!(emesh_bin.point_count() > 0);
    Ok(())
}

#[test]
fn xsph_module_generates_missing_mpse_from_phase_and_pot_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let mpse_path = temp.path().join("mpse.dat");

    write_phase_bin(&phase_path, &sample_mpse_phase_bin())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert!(!temp.path().join("xsect.dat").is_file());
    let actual = read_mpse_dat(mpse_path)?;
    assert_eq!(actual.point_count(), 2);
    assert!(actual.header_lines[0].starts_with("#HD#"));
    assert_column_close(
        &actual.energy_ev,
        &Array1::from_vec(vec![2.0, 6.0]),
        1.0e-10,
    );
    assert_complex_column_close(
        &actual.self_energy,
        &Array1::from_vec(vec![
            Complex64::new(0.25, -0.05),
            Complex64::new(0.40, -0.10),
        ]),
        1.0e-10,
    );
    assert_column_close(
        actual
            .inelastic_mean_free_path
            .as_ref()
            .context("generated mpse.dat missing IMFP column")?,
        &Array1::from_vec(vec![
            (1.0 / FEFF_HARTREE_EV).sqrt() / (0.05 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
            (3.0 / FEFF_HARTREE_EV).sqrt() / (0.10 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
        ]),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_mpse_from_phase_and_pot_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let mpse_path = temp.path().join("mpse.dat");

    write_phase_bin(&phase_path, &sample_mpse_phase_bin())?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    std::fs::write(&mpse_path, "not mpse.dat\n")?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert!(!temp.path().join("xsect.dat").is_file());
    let actual = read_mpse_dat(mpse_path)?;
    assert_eq!(actual.point_count(), 2);
    assert!(actual.header_lines[0].starts_with("#HD#"));
    assert_column_close(
        &actual.energy_ev,
        &Array1::from_vec(vec![2.0, 6.0]),
        1.0e-10,
    );
    assert_complex_column_close(
        &actual.self_energy,
        &Array1::from_vec(vec![
            Complex64::new(0.25, -0.05),
            Complex64::new(0.40, -0.10),
        ]),
        1.0e-10,
    );
    assert_column_close(
        actual
            .inelastic_mean_free_path
            .as_ref()
            .context("generated mpse.dat missing IMFP column")?,
        &Array1::from_vec(vec![
            (1.0 / FEFF_HARTREE_EV).sqrt() / (0.05 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
            (3.0 / FEFF_HARTREE_EV).sqrt() / (0.10 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
        ]),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_mpse_from_phase_and_pot_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let mpse_path = temp.path().join("mpse.dat");

    let phase = sample_mpse_phase_bin();
    let xsect = sample_xsect_dat_for_phase(&phase);
    write_phase_bin(&phase_path, &phase)?;
    write_xsect_dat(&xsect_path, &xsect)?;
    write_pot_bin(temp.path().join("pot.bin"), &sample_mpse_pot_bin())?;
    write_mpse_dat(&mpse_path, &sample_mpse_dat())?;
    let stale_mpse = read_mpse_dat(&mpse_path)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(has_cached_xsph_output(temp.path())?);
    let actual = read_mpse_dat(&mpse_path)?;
    assert_ne!(
        actual, stale_mpse,
        "run should replace the stale readable mpse.dat cache"
    );
    assert_eq!(actual.point_count(), 2);
    assert!(actual.header_lines[0].starts_with("#HD#"));
    assert_column_close(
        &actual.energy_ev,
        &Array1::from_vec(vec![2.0, 6.0]),
        1.0e-10,
    );
    assert_complex_column_close(
        &actual.self_energy,
        &Array1::from_vec(vec![
            Complex64::new(0.25, -0.05),
            Complex64::new(0.40, -0.10),
        ]),
        1.0e-10,
    );
    assert_column_close(
        actual
            .inelastic_mean_free_path
            .as_ref()
            .context("generated mpse.dat missing IMFP column")?,
        &Array1::from_vec(vec![
            (1.0 / FEFF_HARTREE_EV).sqrt() / (0.05 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
            (3.0 / FEFF_HARTREE_EV).sqrt() / (0.10 / FEFF_HARTREE_EV) * FEFF_BOHR_ANGSTROM,
        ]),
        1.0e-8,
    );
    let written_phase = read_phase_bin(&phase_path)?;
    assert_eq!(written_phase.energy_count, phase.energy_count);
    assert_eq!(written_phase.main_energy_count, phase.main_energy_count);
    assert_eq!(written_phase.potential_count(), phase.potential_count());
    let written_xsect = read_xsect_dat(&xsect_path)?;
    assert_eq!(written_xsect.main_energy_count, xsect.main_energy_count);
    assert_eq!(written_xsect.fermi_index, xsect.fermi_index);
    assert_complex_column_close(&written_xsect.energy_grid_ev, &xsect.energy_grid_ev, 1.0e-8);
    assert_column_close(
        &written_xsect.normalized_background,
        &xsect.normalized_background,
        1.0e-8,
    );
    assert_complex_column_close(&written_xsect.cross_section, &xsect.cross_section, 1.0e-8);
    Ok(())
}

#[test]
fn xsph_module_generates_reference_mpse_from_phase_and_pot_cache() -> Result<()> {
    let Some(reference_dir) = reference_xsph_with_pot_and_mpse_dir()? else {
        crate::require_fixture!("XSPH mpse reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "phase.bin", "pot.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let expected = read_mpse_dat(reference_dir.join("mpse.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_mpse_close(
        &read_mpse_dat(temp.path().join("mpse.dat"))?,
        &expected,
        1.0e-4,
    );
    Ok(())
}

#[test]
fn xsph_mpse_source_handoff_preserves_csigz_renormalization() -> Result<()> {
    let Some(reference_dir) = reference_mpse_cu_opcons_dir()? else {
        crate::require_fixture!(
            "XSPH CSigZ mpse.dat reference test; generated MPSE/Cu_OPCONS reference not found"
        );
    };

    let expected = read_mpse_dat(reference_dir.join("mpse.dat"))?;
    let actual = super::generate_mpse_dat_from_source_handoff(&reference_dir)?
        .context("MPSE/Cu_OPCONS source handoffs did not generate mpse.dat")?;

    assert!(
        actual.renormalization.as_ref().is_some_and(|values| values
            .iter()
            .any(|value| (*value - Complex64::new(1.0, 0.0)).norm() > 1.0e-3)),
        "active MPSE must persist the CSigZ renormalization instead of an identity placeholder"
    );
    assert_mpse_close(&actual, &expected, 1.0e-4);
    Ok(())
}

#[test]
fn xsph_active_mpse_source_handoff_requires_loss_poles() -> Result<()> {
    let Some(reference_dir) = reference_mpse_cu_opcons_dir()? else {
        crate::require_fixture!(
            "XSPH missing-loss MPSE test; generated MPSE/Cu_OPCONS reference not found"
        );
    };
    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "phase.bin", "pot.bin", "mpse.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    assert!(
        super::generate_mpse_dat_from_source_handoff(temp.path())?.is_none(),
        "active MPSE without loss.dat pole data must not manufacture identity renormalization"
    );
    let caches = super::XsphCachePaths::new(temp.path());
    let input = super::read_input(temp.path())?;
    let phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let expected = read_mpse_dat(temp.path().join("mpse.dat"))?;
    assert_eq!(
        super::write_or_generate_mpse_cache(&caches, &input, &phase)?,
        (1, false),
        "missing loss poles must preserve an existing valid mpse.dat cache"
    );
    let preserved = read_mpse_dat(temp.path().join("mpse.dat"))?;
    assert_mpse_close(&preserved, &expected, 1.0e-12);
    Ok(())
}

#[test]
fn xsph_module_requires_xsect_when_generating_axafs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_axafs_phase_bin())?;

    let error = run_in_dir(temp.path())
        .err()
        .context("AXAFS generation should require xsect.dat")?;

    assert!(
        error
            .to_string()
            .contains("XSPH AXAFS generation requires xsect.dat cross-section handoff")
    );
    assert!(!temp.path().join("axafs.dat").is_file());
    Ok(())
}

#[test]
fn xsph_module_generates_axafs_when_print_requested() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let axafs_path = temp.path().join("axafs.dat");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_axafs_source_xsect_dat())?;
    let expected_phase = read_phase_bin(&phase_path)?;
    let expected_xsect = read_xsect_dat(&xsect_path)?;
    let expected_axafs = expected_axafs_dat_from_phase_and_xsect(&expected_phase, &expected_xsect)?;
    let expected_emesh = expected_emesh_dat_from_phase(&expected_phase, 0)?;
    let expected_emesh_bin = emesh_bin_from_phase_bin(&expected_phase)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(has_cached_xsph_output(temp.path())?);
    assert_eq!(read_phase_bin(&phase_path)?, expected_phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, expected_xsect);
    assert_eq!(read_axafs_dat(&axafs_path)?, expected_axafs);
    assert_eq!(
        read_emesh_dat(temp.path().join("emesh.dat"))?,
        expected_emesh
    );
    assert_eq!(
        read_emesh_bin(temp.path().join("emesh.bin"))?,
        expected_emesh_bin
    );
    let log = read_module_log_dat(temp.path().join("log2.dat"))?;
    assert_log_contains(&log, "Calculating cross-section and phases ...");
    assert_log_contains(&log, "    absorption cross section");
    Ok(())
}

#[test]
fn xsph_module_recovers_malformed_axafs_from_phase_and_xsect_when_print_requested() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let axafs_path = temp.path().join("axafs.dat");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_axafs_source_xsect_dat())?;
    std::fs::write(&axafs_path, "not an axafs.dat table\n")?;
    let phase = read_phase_bin(&phase_path)?;
    let xsect = read_xsect_dat(&xsect_path)?;
    let expected_axafs = expected_axafs_dat_from_phase_and_xsect(&phase, &xsect)?;

    assert!(has_cached_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_eq!(read_axafs_dat(&axafs_path)?, expected_axafs);
    assert_eq!(read_phase_bin(&phase_path)?, phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, xsect);
    Ok(())
}

#[test]
fn xsph_module_regenerates_stale_axafs_from_phase_and_xsect_when_print_requested() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 1)?;
    let phase_path = temp.path().join("phase.bin");
    let xsect_path = temp.path().join("xsect.dat");
    let axafs_path = temp.path().join("axafs.dat");

    write_phase_bin(&phase_path, &sample_axafs_phase_bin())?;
    write_xsect_dat(&xsect_path, &sample_axafs_source_xsect_dat())?;
    write_axafs_dat(&axafs_path, &sample_axafs_dat())?;
    let phase = read_phase_bin(&phase_path)?;
    let xsect = read_xsect_dat(&xsect_path)?;
    let stale_axafs = read_axafs_dat(&axafs_path)?;
    let expected_axafs = expected_axafs_dat_from_phase_and_xsect(&phase, &xsect)?;
    assert_ne!(
        stale_axafs, expected_axafs,
        "test setup should start with a stale readable axafs.dat cache"
    );

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(has_supported_xsph_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(has_cached_xsph_output(temp.path())?);
    assert_eq!(read_axafs_dat(&axafs_path)?, expected_axafs);
    assert_eq!(read_phase_bin(&phase_path)?, phase);
    assert_eq!(read_xsect_dat(&xsect_path)?, xsect);
    Ok(())
}

#[test]
fn xsph_module_does_not_recover_malformed_axafs_when_print_not_requested() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input_with_print_level(temp.path(), 1, 0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_axafs_phase_bin())?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_axafs_source_xsect_dat(),
    )?;
    std::fs::write(temp.path().join("axafs.dat"), "not an axafs.dat table\n")?;

    assert!(!has_cached_xsph_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed axafs.dat should stay strict when AXAFS is not requested")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("axafs.dat"), "{chain}");
    Ok(())
}

#[test]
fn xsph_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_xsph_dir()? else {
        crate::require_fixture!("XSPH reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["xsph.inp", "phase.bin", "xsect.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    for name in [
        "xsecl.dat",
        "xsecl2.dat",
        "xsecl.bin",
        "axafs.dat",
        "mpse.dat",
        "emesh.dat",
        "emesh.bin",
        "log2.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }

    let expected_phase = read_phase_bin(temp.path().join("phase.bin"))?;
    let expected_xsect = read_xsect_dat(temp.path().join("xsect.dat"))?;
    let expected_axafs = optional_axafs_dat(temp.path().join("axafs.dat"))?;
    let expected_xsecl = optional_xsecl_dat(temp.path().join("xsecl.dat"))?;
    let expected_xsecl2 = optional_xsecl2_dat(temp.path().join("xsecl2.dat"))?;
    let expected_xsecl_bin = optional_xsecl_bin(
        temp.path().join("xsecl.bin"),
        expected_phase.pad_width,
        expected_phase.energy_count,
    )?;
    let expected_mpse = optional_mpse_dat(temp.path().join("mpse.dat"))?;
    let expected_emesh = optional_emesh_dat(temp.path().join("emesh.dat"))?;
    let expected_emesh_bin = optional_emesh_bin(temp.path().join("emesh.bin"))?;
    let expected_log = optional_module_log(temp.path().join("log2.dat"))?;

    let count = run_in_dir(temp.path())?;

    let optional_count = [
        expected_xsecl.as_ref().map(|_| 1_usize),
        expected_xsecl2.as_ref().map(|_| 1_usize),
        expected_xsecl_bin.as_ref().map(|_| 1_usize),
        expected_axafs.as_ref().map(|_| 1_usize),
        expected_mpse.as_ref().map(|_| 1_usize),
        expected_emesh.as_ref().map(|_| 1_usize),
        expected_emesh_bin.as_ref().map(|_| 1_usize),
        expected_log.as_ref().map(|_| 1_usize),
    ]
    .into_iter()
    .flatten()
    .sum::<usize>();
    assert_eq!(count, 2 + optional_count);
    assert_eq!(
        read_phase_bin(temp.path().join("phase.bin"))?,
        expected_phase
    );
    assert_eq!(
        read_xsect_dat(temp.path().join("xsect.dat"))?,
        expected_xsect
    );
    if let Some(expected) = expected_axafs {
        assert_eq!(read_axafs_dat(temp.path().join("axafs.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl {
        assert_eq!(read_xsecl_dat(temp.path().join("xsecl.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl2 {
        assert_eq!(read_xsecl2_dat(temp.path().join("xsecl2.dat"))?, expected);
    }
    if let Some(expected) = expected_xsecl_bin {
        assert_eq!(
            read_xsecl_bin(
                temp.path().join("xsecl.bin"),
                expected_phase.pad_width,
                expected_phase.energy_count
            )?,
            expected
        );
    }
    if let Some(expected) = expected_mpse {
        assert_eq!(read_mpse_dat(temp.path().join("mpse.dat"))?, expected);
    }
    if let Some(expected) = expected_emesh {
        assert_eq!(read_emesh_dat(temp.path().join("emesh.dat"))?, expected);
    }
    if let Some(expected) = expected_emesh_bin {
        assert_eq!(read_emesh_bin(temp.path().join("emesh.bin"))?, expected);
    }
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log2.dat"))?, expected);
    }
    Ok(())
}

fn write_xsph_input(work_dir: &Path, mphase: i32) -> Result<()> {
    write_xsph_input_with_print_level(work_dir, mphase, 0)
}

fn assert_rejects_invalid_nrixs_xsectjas_sidecars(
    mutate: impl FnOnce(&mut XseclDatData, &mut XseclDatData, &mut XseclBinData),
    expected_fragments: &[&str],
) -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xsph_input(temp.path(), 1)?;
    write_global_input_custom(temp.path(), |global| {
        global.control.do_nrixs = 1;
        global.control.l2lp = super::XSPH_NRIXS_L2LP_SENTINEL;
    })?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xsect_dat_for_phase(&phase),
    )?;
    let mut xsecl = sample_xsecl_dat_for_phase(&phase)?;
    let mut xsecl2 = xsecl.clone();
    let mut xsecl_bin = sample_xsecl_bin();
    mutate(&mut xsecl, &mut xsecl2, &mut xsecl_bin);
    write_xsecl_dat(temp.path().join("xsecl.dat"), &xsecl)?;
    write_xsecl2_dat(temp.path().join("xsecl2.dat"), &xsecl2)?;
    write_xsecl_bin(temp.path().join("xsecl.bin"), &xsecl_bin)?;

    assert!(!has_cached_xsph_output(temp.path())?);
    assert!(!has_supported_xsph_output(temp.path())?);

    let error = run_required_in_dir(temp.path())
        .err()
        .context("invalid NRIXS xsectjas sidecars should fail the direct XSPH runner")?;
    let chain = format!("{error:?}");
    for fragment in expected_fragments {
        assert!(chain.contains(fragment), "{chain}");
    }
    Ok(())
}

fn write_xsph_input_with_print_level(work_dir: &Path, mphase: i32, ipr2: i32) -> Result<()> {
    let input = sample_xsph_input(mphase, ipr2);
    std::fs::write(work_dir.join("xsph.inp"), xsph_input_string(&input)?)?;
    Ok(())
}

fn write_xsph_input_custom(work_dir: &Path, update: impl FnOnce(&mut XsphInput)) -> Result<()> {
    let mut input = sample_xsph_input(1, 0);
    update(&mut input);
    std::fs::write(work_dir.join("xsph.inp"), xsph_input_string(&input)?)?;
    Ok(())
}

fn write_xsph_source_setup_with_e2_controls(work_dir: &Path) -> Result<()> {
    write_xsph_input_custom(work_dir, |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.control.l2lp = 0;
        input.lmaxph = vec![3];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(work_dir.join("grid.inp"), &sample_single_point_grid_input())?;
    write_pot_bin(work_dir.join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        work_dir.join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    Ok(())
}

fn write_normal_xsph_source_with_vr0(work_dir: &Path, vr0: f64) -> Result<()> {
    write_xsph_input_custom(work_dir, |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.ixc0 = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.vr0 = vr0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(work_dir.join("grid.inp"), &sample_single_point_grid_input())?;
    write_pot_bin(work_dir.join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        work_dir.join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    Ok(())
}

fn write_empty_cell_xsph_input(work_dir: &Path) -> Result<()> {
    write_xsph_input_custom(work_dir, |input| {
        input.control.i_core_state = 1;
        input.control.i_plsmn = 1;
        input.lmaxph = vec![1, 2];
        input.pot_labels = vec!["E0".to_string(), "E1".to_string()];
    })
}

fn write_global_input(work_dir: &Path, le2: i32, l2lp: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_global_input(le2, l2lp))?,
    )?;
    Ok(())
}

fn write_global_input_custom(work_dir: &Path, update: impl FnOnce(&mut GlobalInput)) -> Result<()> {
    let mut input = sample_global_input(0, 0);
    update(&mut input);
    std::fs::write(work_dir.join("global.inp"), global_input_string(&input)?)?;
    Ok(())
}

fn write_eels_input_with_calculation_mode(work_dir: &Path, calculation_mode: i32) -> Result<()> {
    let input = EelsInput {
        calculate_elnes: calculation_mode != 0,
        calculation_mode,
        control: EelsControl {
            average: 0,
            relativistic: 1,
            cross_terms: 1,
            input: 1,
            spectrum_column: 4,
        },
        polarization: EelsPolarization {
            min: 1,
            step: 1,
            max: 1,
        },
        beam_energy: 100_000.0,
        beam_direction: [0.0, 0.0, 1.0],
        angles: EelsAngles {
            collection: 0.01,
            convergence: 0.0,
        },
        qmesh: EelsQMesh {
            radial: 3,
            angular: 4,
        },
        detector: [0.0, 0.0],
        magic: 0,
        magic_energy: 0.0,
    };
    std::fs::write(work_dir.join("eels.inp"), eels_input_string(&input)?)?;
    Ok(())
}

fn write_polarized_e2_global_input(work_dir: &Path) -> Result<()> {
    write_global_input_custom(work_dir, |global| {
        global.control.ipol = 1;
        global.control.ispin = 1;
        global.control.le2 = 2;
        global.control.angks = 0.3;
        global.control.l2lp = 0;
        global.evec = [1.0, 0.0, 0.0];
        global.spvec = [0.0, 0.0, 1.0];
        global.polarization_tensor = [
            [0.5, 0.0, 0.0, 0.0, -0.5, 0.0],
            [0.0; 6],
            [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        ];
    })
}

fn write_hubbard_input(work_dir: &Path, mldos_hubb: i32) -> Result<()> {
    let input = HubbardInput {
        i_hubbard: if mldos_hubb == 2 { 2 } else { 1 },
        mldos_hubb,
        u: 4.0,
        j: 0.5,
        fermi_shift: 0.0,
        l: 2,
    };
    std::fs::write(work_dir.join("hubbard.inp"), hubbard_input_string(&input)?)?;
    Ok(())
}

fn write_active_hubbard_source_handoff_inputs(work_dir: &Path) -> Result<()> {
    write_xsph_input_custom(work_dir, |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![0.0];
    })?;
    write_grid_inp(work_dir.join("grid.inp"), &sample_single_point_grid_input())?;
    write_pot_bin(work_dir.join("pot.bin"), &sample_normal_phase_pot_bin())?;
    write_config_dat(
        work_dir.join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_hubbard_input(work_dir, 2)?;
    write_v_hubbard_bin(work_dir.join("v_hubbard.bin"), &sample_v_hubbard_bin(1))?;
    Ok(())
}

fn write_split_pmbse_xmu_sources(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("listedges.pmbse"),
        "Oddp1\nEvenp1\nOddm1\nEvenm1\n",
    )?;
    write_pmbse_xmu_channel(
        work_dir,
        "Oddp1",
        &[100.0, 101.0, 102.0, 103.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
    )?;
    write_pmbse_xmu_channel(
        work_dir,
        "Evenp1",
        &[102.0, 102.5, 103.0, 103.5],
        &[0.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 3.0],
        &[9.0, 19.0, 29.0, 39.0],
    )?;
    write_pmbse_xmu_channel(
        work_dir,
        "Oddm1",
        &[100.0, 101.0, 102.0, 103.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0, 7.0],
    )?;
    write_pmbse_xmu_channel(
        work_dir,
        "Evenm1",
        &[102.0, 102.5, 103.0, 103.5],
        &[0.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 3.0],
        &[49.0, 59.0, 69.0, 79.0],
    )?;
    Ok(())
}

fn write_tdlda_no_cache_source_case(work_dir: &Path, nonlocal: i32, two_spin: bool) -> Result<()> {
    write_xsph_input_custom(work_dir, |input| {
        input.control.nph = 0;
        input.control.i_core_state = 1;
        input.control.ixc = 2;
        input.control.lreal = 1;
        input.control.i_grid = 1;
        input.advanced.izstd = 0;
        input.advanced.ifxc = 5;
        input.advanced.ipmbse = 2;
        input.advanced.itdlda = 2;
        input.advanced.nonlocal = nonlocal;
        input.advanced.ibasis = 0;
        input.lmaxph = vec![1];
        input.pot_labels = vec!["Cu".to_string()];
        input.spinph = vec![if two_spin { 1.0 } else { 0.0 }];
    })?;
    if two_spin {
        write_global_input_custom(work_dir, |global| global.control.ispin = 1)?;
    }
    write_grid_inp(work_dir.join("grid.inp"), &sample_single_point_grid_input())?;
    let mut pot = sample_normal_phase_pot_bin();
    pot.ihole = 4;
    write_pot_bin(work_dir.join("pot.bin"), &pot)?;
    write_config_dat(
        work_dir.join("config.dat"),
        &sample_normal_phase_config_dat(),
    )?;
    write_split_pmbse_xmu_sources(work_dir)
}

fn write_tdlda_screened_potential_source(path: PathBuf) -> Result<()> {
    let mut text = String::from("# radius screened core-hole\n");
    for row in 0..POT_BIN_RADIAL_POINTS {
        let radius = (-8.8 + 0.05 * row as f64).exp();
        let screened = 0.05 * (-0.01 * row as f64).exp();
        text.push_str(&format!("{radius:.12e} {screened:.12e} 0.0\n"));
    }
    std::fs::write(path, text)?;
    Ok(())
}

fn write_tdlda_file_basis_orbitals(work_dir: &Path) -> Result<()> {
    let orbital_dir = work_dir.join("Vila").join("Orbs");
    std::fs::create_dir_all(&orbital_dir)?;
    std::fs::write(
        orbital_dir.join("mg.3p.dat"),
        tdlda_file_basis_orbital_text(2.0),
    )?;
    std::fs::write(
        orbital_dir.join("mg.4p.dat"),
        tdlda_file_basis_orbital_text(3.0),
    )?;
    Ok(())
}

fn tdlda_file_basis_orbital_text(scale: f64) -> String {
    (1..=10)
        .map(|index| {
            let radius = 0.05 * index as f64;
            format!("{radius:.8} {scale:.8}\n")
        })
        .collect()
}

fn write_pmbse_xmu_channel(
    work_dir: &Path,
    channel_dir: &str,
    photon_energy_ev: &[f64],
    relative_energy_ev: &[f64],
    wave_number: &[f64],
    chi: &[f64],
) -> Result<()> {
    let channel_path = work_dir.join(channel_dir);
    std::fs::create_dir_all(&channel_path)?;
    let mu0 = Array1::from_elem(photon_energy_ev.len(), 1.0);
    let chi = Array1::from_vec(chi.to_vec());
    let mu = &mu0 + &chi;
    let data = XmuDatData {
        header_lines: vec!["# PMBSE xmu.dat test channel".to_string()],
        normalization: None,
        photon_energy_ev: Array1::from_vec(photon_energy_ev.to_vec()),
        relative_energy_ev: Array1::from_vec(relative_energy_ev.to_vec()),
        wave_number: Array1::from_vec(wave_number.to_vec()),
        mu,
        mu0,
        chi,
    };
    write_xmu_dat(channel_path.join("xmu.dat"), &data)?;
    Ok(())
}

fn sample_xsph_input(mphase: i32, ipr2: i32) -> XsphInput {
    XsphInput {
        control: XsphControl {
            mphase,
            ipr2,
            ixc: 0,
            ixc0: 0,
            ispec: 0,
            lreal: 0,
            lfms2: 0,
            nph: 1,
            l2lp: 0,
            i_plsmn: 0,
            n_poles: 100,
            i_gamma_ch: 0,
            i_grid: 0,
            i_core_state: -1,
            iscfxc: 11,
        },
        vr0: 0.0,
        vi0: 0.0,
        lmaxph: vec![1, 1],
        pot_labels: vec!["Cu".to_string(), "O".to_string()],
        grid: XsphGrid {
            rgrd: 0.05,
            rfms2: 0.0,
            gamach: 1.0,
            xkstep: 0.05,
            xkmax: 10.0,
            vixan: 0.0,
            eps0: 0.0,
            egap: 0.0,
        },
        spinph: vec![0.0, 0.0],
        advanced: XsphAdvanced {
            izstd: 0,
            ifxc: 0,
            ipmbse: 0,
            itdlda: 0,
            nonlocal: 0,
            ibasis: 0,
        },
        electronic_temperature: 0.0,
        chsh_type: 0,
        decomposition_channels: -1,
        lopt: false,
        print_rl: false,
        source_format: XsphInputSourceFormat::modern(),
    }
}

fn sample_global_input(le2: i32, l2lp: i32) -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 0,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: 0,
            ispin: 0,
            le2,
            elpty: 0.0,
            angks: 0.0,
            l2lp,
            do_nrixs: 0,
            ldecmx: -1,
            lj: 0,
        },
        evec: [0.0; 3],
        xivec: [0.0; 3],
        spvec: [0.0; 3],
        polarization_tensor: [[0.0; 6]; 3],
        norms: GlobalNorms {
            evnorm: 0.0,
            xivnorm: 0.0,
            spvnorm: 0.0,
        },
        q_control: GlobalQControl {
            nq: 0,
            imdff: 0,
            qaverage: false,
            mixdff: false,
        },
        q_vectors: Vec::new(),
        mdff: None,
    }
}

fn sample_xsph_grid_input() -> GridInput {
    GridInput {
        records: vec![
            GridRecord::Regular(GridRegularRecord {
                kind: GridKind::Energy,
                minimum: GridMinimum::Value(-2.0),
                maximum: 2.0,
                step: 1.0,
            }),
            GridRecord::Regular(GridRegularRecord {
                kind: GridKind::WaveNumber,
                minimum: GridMinimum::Last,
                maximum: 3.0,
                step: 1.0,
            }),
            GridRecord::User(GridUserRecord {
                points: vec![
                    GridPoint {
                        real: -5.0,
                        imaginary: 0.2,
                    },
                    GridPoint {
                        real: 0.0004,
                        imaginary: 0.0,
                    },
                    GridPoint {
                        real: 12.0,
                        imaginary: -0.1,
                    },
                ],
            }),
        ],
    }
}

fn sample_single_point_grid_input() -> GridInput {
    GridInput {
        records: vec![GridRecord::User(GridUserRecord {
            points: vec![GridPoint {
                real: 4.0,
                imaginary: 0.01,
            }],
        })],
    }
}

fn sample_loss_dat() -> LossDatData {
    LossDatData {
        header_lines: vec!["# XSPH MPSE loss smoke test".to_string()],
        energy_ev: Array1::from_vec(vec![5.0, 12.0, 25.0, 60.0, 120.0, 250.0, 500.0]),
        loss: Array1::from_vec(vec![0.18, 0.45, 0.32, 0.20, 0.11, 0.05, 0.02]),
    }
}

fn sample_phase_bin() -> PhaseBinData {
    let spin_count = 1;
    let energy_count = 2;
    let transition_count = 2;
    let q_count = 1;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 2,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 4,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: -0.35,
            edge_energy: 9.8,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + energy as f64, 0.01 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
            Complex64::new(-1.0 + 0.2 * energy as f64, 0.0)
        }),
        potentials: vec![
            sample_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_potential(1, 8, "O", energy_count, spin_count, 0.2),
        ],
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

fn sample_aphase_hubbard_bin(energy_count: usize, potential_count: usize) -> HubbardAphaseBinData {
    let angular_limit = 1;
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    let mut next = 1.0;
    let values = Array5::from_shape_fn(
        (
            potential_count,
            2,
            energy_count,
            angular_count,
            magnetic_count,
        ),
        |_| {
            let value = Complex64::new(next, -next);
            next += 1.0;
            value
        },
    );
    HubbardAphaseBinData {
        angular_limit,
        values,
    }
}

fn sample_v_hubbard_bin(potential_count: usize) -> HubbardVnlmBinData {
    let angular_limit = 1;
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    let mut next = 0.05;
    let values = Array4::from_shape_fn((potential_count, 2, angular_count, magnetic_count), |_| {
        let value = next;
        next += 0.05;
        value
    });
    HubbardVnlmBinData {
        angular_limit,
        values,
    }
}

fn sample_axafs_phase_bin() -> PhaseBinData {
    let spin_count = 1;
    let energy_count = 5;
    let transition_count = 2;
    let q_count = 1;
    let energy_grid_ev = [
        Complex64::new(0.0, 0.0),
        Complex64::new(10.0, 0.01),
        Complex64::new(30.0, 0.02),
        Complex64::new(60.0, 0.03),
        Complex64::new(100.0, 0.04),
    ];
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 5,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 4,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: -0.35,
            edge_energy: 9.8,
        },
        energy_grid: Array1::from_iter(
            energy_grid_ev
                .iter()
                .copied()
                .map(|energy| energy / FEFF_HARTREE_EV),
        ),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, _)| {
            Complex64::new(-1.0 + 0.2 * energy as f64, 0.0)
        }),
        potentials: vec![
            sample_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_potential(1, 8, "O", energy_count, spin_count, 0.2),
        ],
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

fn sample_mpse_phase_bin() -> PhaseBinData {
    let mut phase = sample_phase_bin();
    phase.energy_count = 3;
    phase.main_energy_count = 3;
    phase.fermi_index = 1;
    phase.scalars.fermi_level = 0.0;
    phase.energy_grid = Array1::from_vec(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0 / FEFF_HARTREE_EV, 0.0),
        Complex64::new(6.0 / FEFF_HARTREE_EV, 0.0),
    ]);
    phase.reference_energy = Array2::from_shape_vec(
        (3, 1),
        vec![
            Complex64::new(-0.75, 0.0),
            Complex64::new(-0.75 + 0.25 / FEFF_HARTREE_EV, -0.05 / FEFF_HARTREE_EV),
            Complex64::new(-0.75 + 0.40 / FEFF_HARTREE_EV, -0.10 / FEFF_HARTREE_EV),
        ],
    )
    .expect("valid sample mpse reference-energy shape");
    phase.potentials = vec![sample_potential(1, 29, "Cu", 3, 1, 0.1)];
    phase.transition_moments = Array4::zeros((3, phase.q_count, phase.transition_count, 1));
    phase.raw_pads = None;
    phase
}

fn sample_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    scale: f64,
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
                    scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
    }
}

fn sample_xsect_dat() -> XsectDatData {
    let phase = sample_phase_bin();
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
        main_energy_count: phase.main_energy_count,
        fermi_index: phase.fermi_index as usize,
        energy_grid_ev: phase.energy_grid.mapv(|energy| energy * FEFF_HARTREE_EV),
        normalized_background: Array1::from_vec(vec![2.0, 2.5]),
        cross_section: Array1::from_vec(vec![Complex64::new(3.0, -0.4), Complex64::new(3.5, -0.5)]),
    }
}

fn sample_xsect_dat_for_phase(phase: &PhaseBinData) -> XsectDatData {
    XsectDatData {
        titles: vec!["cached empty-cell xsect".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.85,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: phase.scalars.edge_energy * FEFF_HARTREE_EV,
            chemical_potential: phase.scalars.fermi_level * FEFF_HARTREE_EV,
        },
        core_hole_width_ev: 1.23,
        main_energy_count: phase.main_energy_count,
        fermi_index: phase.fermi_index as usize,
        energy_grid_ev: phase.energy_grid.mapv(|energy| energy * FEFF_HARTREE_EV),
        normalized_background: Array1::from_shape_fn(phase.energy_count, |index| {
            1.0 + 0.25 * index as f64
        }),
        cross_section: Array1::from_shape_fn(phase.energy_count, |index| {
            let value = 2.0 + 0.5 * index as f64;
            Complex64::new(value, -0.1 * value)
        }),
    }
}

fn sample_axafs_source_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu crystal".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            edge_energy: 0.0,
            chemical_potential: 0.0,
        },
        core_hole_width_ev: 1.23,
        main_energy_count: 5,
        fermi_index: 1,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(10.0, 0.01),
            Complex64::new(30.0, 0.02),
            Complex64::new(60.0, 0.03),
            Complex64::new(100.0, 0.04),
        ]),
        normalized_background: Array1::from_vec(vec![1.0, 1.2, 1.6, 2.1, 2.7]),
        cross_section: Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 1.1),
            Complex64::new(2.3, 1.7),
            Complex64::new(2.8, 2.8),
            Complex64::new(3.5, 4.4),
        ]),
    }
}

fn sample_mpse_pot_bin() -> PotBinData {
    let potentials = 1;
    PotBinData {
        titles: vec!["XSPH MPSE smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.0,
            fermi_level: 0.0,
            interstitial_potential: -0.75,
            interstitial_density: 0.08,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![20]),
        muffin_tin_radii: Array1::from_vec(vec![0.0123]),
        norman_indices: Array1::from_vec(vec![40]),
        atomic_numbers: Array1::from_vec(vec![29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.1]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), 0.08),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), -0.5),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_normal_phase_pot_bin() -> PotBinData {
    let potentials = 1;
    let occupied_orbitals = 4;
    let mut large_components = Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    let mut small_components = Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
    let mut large_coefficients =
        Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials));
    let mut small_coefficients =
        Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials));
    let mut orbital_occupancy = Array2::zeros((POT_BIN_ORBITALS, potentials));

    for orbital in 0..occupied_orbitals {
        orbital_occupancy[(orbital, 0)] = if orbital == 3 { 0.2 } else { 0.0 };
        for row in 0..POT_BIN_RADIAL_POINTS {
            let r = (row + 1) as f64;
            let o = (orbital + 1) as f64;
            large_components[(row, orbital, 0)] =
                0.012 * o * (0.035 * r * o).sin() * (-0.002 * r).exp();
            small_components[(row, orbital, 0)] =
                0.007 * o * (0.029 * r * o).cos() * (-0.0025 * r).exp();
        }
        for coefficient in 0..POT_BIN_COEFFICIENTS {
            let c = (coefficient + 1) as f64;
            let o = (orbital + 1) as f64;
            large_coefficients[(coefficient, orbital, 0)] =
                0.006 * c + 0.0009 * o * (0.17 * c * o).cos();
            small_coefficients[(coefficient, orbital, 0)] =
                -0.004 * c + 0.0006 * o * (0.13 * c * o).sin();
        }
    }

    PotBinData {
        titles: vec!["XSPH normal phase smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 1.5,
            fermi_level: 0.0,
            interstitial_potential: -0.45,
            interstitial_density: 0.06,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![184]),
        muffin_tin_radii: Array1::from_vec(vec![1.42]),
        norman_indices: Array1::from_vec(vec![220]),
        atomic_numbers: Array1::from_vec(vec![29]),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.0]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            let r = (row + 1) as f64;
            0.018 * (0.041 * r).sin() * (-0.002 * r).exp()
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            let r = (row + 1) as f64;
            0.006 * (0.033 * r).cos() * (-0.0025 * r).exp()
        }),
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        electron_density: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            0.055 + 0.000_01 * row as f64
        }),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            -0.52 + 0.000_4 * row as f64
        }),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy,
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_normal_phase_config_dat() -> ConfigDatData {
    let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    occupations[0] = 1.8;
    occupations[1] = 1.0;
    occupations[2] = 0.7;
    occupations[3] = 0.5;

    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![ConfigDatPotential {
            potential_index: 0,
            atomic_number: 29,
            element: "Cu".to_string(),
            occupations,
            valence_occupations: Array1::zeros(CONFIG_DAT_ORBITAL_COUNT),
            spin_occupations: None,
        }],
    }
}

fn sample_screened_core_hole_vtot_dat() -> VtotDatData {
    VtotDatData {
        header_lines: Vec::new(),
        radius_bohr: Array1::from_vec(vec![
            0.150_733_046_3E-03,
            0.158_461_294_9E-03,
            0.166_585_779_2E-03,
        ]),
        total_potential: Array1::from_vec(vec![
            -0.182_900_150_0E+06,
            -0.182_900_133_6E+06,
            -0.182_900_100_2E+06,
        ]),
        screened_core_hole_potential: Array1::from_vec(vec![
            0.267_288_234_6E+02,
            0.267_288_167_8E+02,
            0.267_288_030_6E+02,
        ]),
    }
}

fn sample_screened_core_hole_components() -> (Array1<f64>, Array1<f64>) {
    let large_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
        if row < 3 {
            0.45 + 0.05 * row as f64
        } else {
            0.0
        }
    });
    let small_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
        if row < 3 {
            -0.06 - 0.01 * row as f64
        } else {
            0.0
        }
    });
    (large_component, small_component)
}

fn sample_screened_core_hole_apot_bin(
    large_component: &Array1<f64>,
    small_component: &Array1<f64>,
) -> ApotBinData {
    ApotBinData {
        sections: vec![ApotBinSection {
            section_number: 5,
            headers: vec![
                "dgc0   - upper component of core hole orbital".to_string(),
                "dpc0   - lower component of core hole orbital".to_string(),
                "drho   - core hole density.".to_string(),
                "dvcoul - core hole coulomb potential.".to_string(),
            ],
            header_texts: vec![
                " dgc0   - upper component of core hole orbital".to_string(),
                " dpc0   - lower component of core hole orbital".to_string(),
                " drho   - core hole density.".to_string(),
                " dvcoul - core hole coulomb potential.".to_string(),
            ],
            column_labels: vec![
                "dgc0".to_string(),
                "dpc0".to_string(),
                "drho".to_string(),
                "dvcoul".to_string(),
            ],
            column_label_text: Some(
                "            dgc0                 dpc0                 drho               dvcoul "
                    .to_string(),
            ),
            payload: ApotBinPayload::Records(refeff_io::ApotBinRecords {
                column_types: vec![ApotBinType::Double; 4],
                rows: (0..POT_BIN_RADIAL_POINTS)
                    .map(|row| {
                        vec![
                            ApotBinValue::Real(large_component[row]),
                            ApotBinValue::Real(small_component[row]),
                            ApotBinValue::Real(0.0),
                            ApotBinValue::Real(0.0),
                        ]
                    })
                    .collect(),
            }),
            trailing_headers: Vec::new(),
            trailing_header_texts: Vec::new(),
        }],
    }
}

fn sample_empty_cell_pot_bin() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["XSPH empty-cell smoke test".to_string()],
        pad_width: 8,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
        automatic_folp: 0,
        jump_mode: 0,
        unfreeze_f: 0,
        scalars: PotBinScalars {
            average_norman_radius: 2.0,
            fermi_level: 0.0,
            interstitial_potential: 0.0,
            interstitial_density: 0.02,
            edge_position: 0.0,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            core_valence_energy: 0.0,
            density_radius: 1.0,
            fermi_momentum: 0.0,
            total_charge: 0.0,
            total_volume: 1.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![20, 20]),
        muffin_tin_radii: Array1::from_vec(vec![2.3, 1.7]),
        norman_indices: Array1::from_vec(vec![40, 40]),
        atomic_numbers: Array1::zeros(potentials),
        kappa: Array1::zeros(POT_BIN_ORBITALS),
        norman_radii: Array1::from_vec(vec![2.4, 1.9]),
        overlap_factors: Array1::ones(potentials),
        max_overlap_factors: Array1::ones(potentials),
        potential_multiplicities: Array1::ones(potentials),
        ionization: Array1::zeros(potentials),
        initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
        large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
        large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
        electron_density: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), 0.02),
        coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        total_potential: Array2::from_elem((POT_BIN_RADIAL_POINTS, potentials), -0.2),
        valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
        orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
        orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
        occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
        norman_charges: Array1::zeros(potentials),
        valence_occupancy: Array2::zeros((4, potentials)),
        raw_text: None,
    }
}

fn sample_axafs_dat() -> AxafsDatData {
    AxafsDatData {
        header_lines: vec![
            " # File contains AXAFS. See manual for details.".to_string(),
            " #--------------------------------------------------------------".to_string(),
            " #  e, e(wrt edge), k, mu_at=(1+chi_at)*mu0_at, mu0_at, chi_at @#".to_string(),
        ],
        energy_ev: Array1::from_vec(vec![8979.0, 8980.5]),
        edge_relative_energy_ev: Array1::from_vec(vec![0.0, 1.5]),
        wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627]),
        atomic_absorption: Array1::from_vec(vec![1.234_56, 1.061_11]),
        atomic_background: Array1::from_vec(vec![1.0, 1.111_11]),
        chi_atomic: Array1::from_vec(vec![0.234_56, -0.045]),
    }
}

fn expected_axafs_dat_from_phase_and_xsect(
    phase: &PhaseBinData,
    xsect: &XsectDatData,
) -> Result<AxafsDatData> {
    let handoff = xsect_dat_ff2x_handoff(xsect, xsect.scalars.amplitude_reduction, 0)?;
    let axafs = xsph_axafs(XsphAxafsInput {
        energies: phase.energy_grid.view(),
        cross_section: handoff.cross_section.view(),
        fermi_energy: handoff.chemical_potential_hartree,
        horizontal_count: handoff.main_energy_count,
        zero_wave_index: handoff.fermi_index,
    })?;
    let data = axafs_dat_from_xsph_axafs(&axafs)?;
    Ok(parse_axafs_dat(&axafs_dat_string(&data)?)?)
}

fn expected_emesh_dat_from_phase(phase: &PhaseBinData, spectrum: i32) -> Result<EmeshDatData> {
    let data = emesh_dat_from_phase_bin(phase, spectrum)?;
    Ok(parse_emesh_dat(&emesh_dat_string(&data)?)?)
}

fn sample_xsecl_dat() -> XseclDatData {
    XseclDatData {
        header: XseclDatHeader {
            real_energy_count: 2,
            fermi_index: 1,
            edge: -0.25,
            emu: 408.0,
            core_hole_width: 0.083_949_386_5,
        },
        energy: Array1::from_vec(vec![408.083_58, 408.118_59]),
        channel_cross_sections: Array2::from_shape_fn((2, 2), |(energy, channel)| {
            let real = match (energy, channel) {
                (0, 0) => -0.000_094_722_801,
                (0, 1) => 0.000_058_529_371,
                (1, 0) => -0.000_042_446_685,
                (1, 1) => -0.000_117_763_55,
                _ => 0.0,
            };
            let imag = match (energy, channel) {
                (0, 0) => 0.000_115_562_54,
                (0, 1) => -0.000_120_865_91,
                (1, 0) => 0.000_105_705_03,
                (1, 1) => -0.000_144_091_45,
                _ => 0.0,
            };
            Complex64::new(real, imag)
        }),
        channel_sum: Array1::from_vec(vec![
            Complex64::new(-0.000_036_126_732, -0.000_005_278_514_8),
            Complex64::new(-0.000_160_211_14, -0.000_038_440_289),
        ]),
    }
}

fn sample_xsecl_dat_for_phase(phase: &PhaseBinData) -> Result<XseclDatData> {
    let fermi_index =
        usize::try_from(phase.fermi_index).context("sample fermi index is negative")?;
    let energy = super::xsecl_energy_grid_from_phase(phase, phase.scalars.fermi_level);
    let channel_cross_sections = Array2::from_shape_fn(
        (phase.energy_count, phase.transition_count),
        |(energy, channel)| {
            Complex64::new(
                0.01 * (energy + 1) as f64 + 0.002 * channel as f64,
                -0.005 * (energy + 1) as f64 - 0.001 * channel as f64,
            )
        },
    );
    let channel_sum = channel_cross_sections.sum_axis(Axis(1));
    Ok(XseclDatData {
        header: XseclDatHeader {
            real_energy_count: phase.main_energy_count,
            fermi_index,
            edge: phase.scalars.edge_energy,
            emu: phase.scalars.fermi_level,
            core_hole_width: 0.083_949_386_5,
        },
        energy,
        channel_cross_sections,
        channel_sum,
    })
}

fn sample_xsecl_bin() -> XseclBinData {
    XseclBinData {
        pad_width: 8,
        initial_state_j: 1,
        transitions: vec![
            XseclBinTransition {
                final_state_kappa: -1,
                decomposition_channel: 0,
                total_angular_momentum_channel: 0,
                orbital_angular_momentum: 0,
            },
            XseclBinTransition {
                final_state_kappa: 2,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
        ],
        atom_cross_sections: Array2::from_shape_fn((2, 4), |(energy, final_state)| {
            Complex64::new(
                0.1 * (energy + 1) as f64 + 0.01 * final_state as f64,
                -0.05 * (energy + 1) as f64 - 0.005 * final_state as f64,
            )
        }),
        raw_atom_cross_section_pad: None,
    }
}

fn sample_mpse_dat() -> MpseDatData {
    MpseDatData {
        header_lines: vec!["# XSPH MPSE self-energy sidecar".to_string()],
        energy_ev: Array1::from_vec(vec![0.038_099_840_30, 0.152_399_361_2]),
        self_energy: Array1::from_vec(vec![
            Complex64::new(0.001_436_696_198, -0.000_007_842_984_015),
            Complex64::new(0.005_774_807_411, -0.000_124_742_315_9),
        ]),
        renormalization: Some(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])),
        renormalization_magnitude: Some(Array1::from_vec(vec![1.0, 1.0])),
        renormalization_phase: Some(Array1::from_vec(vec![0.0, 0.0])),
        inelastic_mean_free_path: Some(Array1::from_vec(vec![48_578.245_52, 6_108.567_091])),
    }
}

fn sample_emesh_dat() -> EmeshDatData {
    EmeshDatData {
        edge_hartree: 333.333,
        bohr_angstrom: 0.529_177_249,
        edge_ev: 9_071.2,
        spectrum: 0,
        fermi_index: 1,
        indices: Array1::from_vec(vec![1, 2, 3]),
        energy_ev: Array1::from_vec(vec![0.0, 1.5, 3.0]),
        wave_number_inverse_angstrom: Array1::from_vec(vec![0.0, 0.627, 0.887]),
    }
}

fn sample_emesh_bin() -> EmeshBinData {
    EmeshBinData {
        point_count_declared: 3,
        horizontal_count: 2,
        danes_extension_count: 1,
        energy_hartree: Array1::from_vec(vec![
            Complex64::new(-0.25, 0.01),
            Complex64::new(0.0, 0.02),
            Complex64::new(0.5, 0.03),
        ]),
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating potentials and phases ...".to_string(),
            "Done with module: potentials and phases.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn optional_xsecl_dat(path: impl AsRef<Path>) -> Result<Option<XseclDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_axafs_dat(path: impl AsRef<Path>) -> Result<Option<AxafsDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_axafs_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_xsecl2_dat(path: impl AsRef<Path>) -> Result<Option<XseclDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl2_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_xsecl_bin(
    path: impl AsRef<Path>,
    pad_width: usize,
    energy_count: usize,
) -> Result<Option<XseclBinData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_xsecl_bin(path, pad_width, energy_count)?))
    } else {
        Ok(None)
    }
}

fn optional_mpse_dat(path: impl AsRef<Path>) -> Result<Option<MpseDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_mpse_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_emesh_dat(path: impl AsRef<Path>) -> Result<Option<EmeshDatData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_emesh_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_emesh_bin(path: impl AsRef<Path>) -> Result<Option<EmeshBinData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_emesh_bin(path)?))
    } else {
        Ok(None)
    }
}

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_module_log_dat(path)?))
    } else {
        Ok(None)
    }
}

fn assert_log_contains(log: &ModuleLogData, expected: &str) {
    assert!(
        log.lines.iter().any(|line| line == expected),
        "expected log to contain {expected:?}, got {:?}",
        log.lines
    );
}

fn assert_phase_cache_sentinel_preserved(actual: &PhaseBinData, expected: &PhaseBinData) {
    assert_eq!(actual.scalars.edge_energy, expected.scalars.edge_energy);
    assert_eq!(actual.energy_count, expected.energy_count);
    assert_eq!(actual.main_energy_count, expected.main_energy_count);
    assert_eq!(actual.potential_count(), expected.potential_count());
}

fn assert_phase_transition_moments_close(
    actual: &PhaseBinData,
    expected: &PhaseBinData,
    tolerance: f64,
) {
    assert_eq!(
        actual.transition_moments.dim(),
        expected.transition_moments.dim()
    );
    for ((energy, q, transition, spin), actual) in actual.transition_moments.indexed_iter() {
        let expected = expected.transition_moments[(energy, q, transition, spin)];
        let difference = (*actual - expected).norm();
        let tolerance = tolerance * expected.norm().max(1.0);
        assert!(
            difference <= tolerance,
            "transition moment differs at ({energy}, {q}, {transition}, {spin}) by {difference:e}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:e}"
        );
    }
}

fn assert_xsect_table_preserved(actual: &XsectDatData, expected: &XsectDatData) {
    assert_eq!(actual.energy_count(), expected.energy_count());
    assert_eq!(actual.main_energy_count, expected.main_energy_count);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_complex_column_close(&actual.energy_grid_ev, &expected.energy_grid_ev, 1.0e-5);
    assert_column_close(
        &actual.normalized_background,
        &expected.normalized_background,
        1.0e-8,
    );
    assert_complex_column_close(&actual.cross_section, &expected.cross_section, 1.0e-8);
}

fn assert_emesh_bin_close(actual: &EmeshBinData, expected: &EmeshBinData, tolerance: f64) {
    assert_eq!(actual.point_count_declared, expected.point_count_declared);
    assert_eq!(actual.horizontal_count, expected.horizontal_count);
    assert_eq!(actual.danes_extension_count, expected.danes_extension_count);
    assert_eq!(actual.energy_hartree.len(), expected.energy_hartree.len());
    for (index, (actual, expected)) in actual
        .energy_hartree
        .iter()
        .zip(expected.energy_hartree.iter())
        .enumerate()
    {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "emesh.bin real value differs at row {}: actual={} expected={}",
            index + 1,
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "emesh.bin imaginary value differs at row {}: actual={} expected={}",
            index + 1,
            actual.im,
            expected.im
        );
    }
}

fn assert_emesh_close(actual: &EmeshDatData, expected: &EmeshDatData, tolerance: f64) {
    assert!((actual.edge_hartree - expected.edge_hartree).abs() <= tolerance);
    assert!((actual.bohr_angstrom - expected.bohr_angstrom).abs() <= tolerance);
    assert!((actual.edge_ev - expected.edge_ev).abs() <= tolerance);
    assert_eq!(actual.spectrum, expected.spectrum);
    assert_eq!(actual.fermi_index, expected.fermi_index);
    assert_eq!(actual.indices, expected.indices);
    assert_column_close(&actual.energy_ev, &expected.energy_ev, tolerance);
    assert_column_close(
        &actual.wave_number_inverse_angstrom,
        &expected.wave_number_inverse_angstrom,
        tolerance,
    );
}

fn assert_mpse_close(actual: &MpseDatData, expected: &MpseDatData, tolerance: f64) {
    assert_eq!(actual.header_lines, expected.header_lines);
    assert_eq!(actual.point_count(), expected.point_count());
    assert_column_close(&actual.energy_ev, &expected.energy_ev, tolerance);
    assert_complex_column_close(&actual.self_energy, &expected.self_energy, tolerance);
    match (&actual.renormalization, &expected.renormalization) {
        (Some(actual), Some(expected)) => assert_complex_column_close(actual, expected, tolerance),
        (None, None) => {}
        _ => panic!("mpse.dat renormalization column presence differs"),
    }
    assert_optional_column_close(
        &actual.renormalization_magnitude,
        &expected.renormalization_magnitude,
        tolerance,
        "renormalization magnitude",
    );
    assert_optional_column_close(
        &actual.renormalization_phase,
        &expected.renormalization_phase,
        tolerance,
        "renormalization phase",
    );
    assert_optional_column_close(
        &actual.inelastic_mean_free_path,
        &expected.inelastic_mean_free_path,
        tolerance,
        "IMFP",
    );
}

fn assert_optional_column_close(
    actual: &Option<Array1<f64>>,
    expected: &Option<Array1<f64>>,
    tolerance: f64,
    name: &str,
) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_column_close(actual, expected, tolerance),
        (None, None) => {}
        _ => panic!("mpse.dat {name} column presence differs"),
    }
}

fn assert_column_close(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual,
            expected,
            tolerance
        );
    }
}

fn assert_column_close_mixed(
    actual: &Array1<f64>,
    expected: &Array1<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "value differs at row {}: actual={} expected={} tolerance={} (absolute={}, relative={})",
            index + 1,
            actual,
            expected,
            tolerance,
            absolute_tolerance,
            relative_tolerance
        );
    }
}

fn assert_complex_column_close(
    actual: &Array1<Complex64>,
    expected: &Array1<Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual.re - expected.re).abs() <= tolerance,
            "real value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual.re,
            expected.re,
            tolerance
        );
        assert!(
            (actual.im - expected.im).abs() <= tolerance,
            "imaginary value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual.im,
            expected.im,
            tolerance
        );
    }
}

fn assert_complex_column_close_mixed(
    actual: &Array1<Complex64>,
    expected: &Array1<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let real_tolerance = absolute_tolerance + relative_tolerance * expected.re.abs();
        assert!(
            (actual.re - expected.re).abs() <= real_tolerance,
            "real value differs at row {}: actual={} expected={} tolerance={} (absolute={}, relative={})",
            index + 1,
            actual.re,
            expected.re,
            real_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
        let imaginary_tolerance = absolute_tolerance + relative_tolerance * expected.im.abs();
        assert!(
            (actual.im - expected.im).abs() <= imaginary_tolerance,
            "imaginary value differs at row {}: actual={} expected={} tolerance={} (absolute={}, relative={})",
            index + 1,
            actual.im,
            expected.im,
            imaginary_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
    }
}

fn assert_serialized_combined_column_close(
    actual: &Array1<f64>,
    dipole: &Array1<f64>,
    magnetic_dipole: &Array1<f64>,
    electric_quadrupole: &Array1<f64>,
) {
    assert_eq!(actual.len(), dipole.len());
    assert_eq!(actual.len(), magnetic_dipole.len());
    assert_eq!(actual.len(), electric_quadrupole.len());
    for (index, (((actual, dipole), magnetic_dipole), electric_quadrupole)) in actual
        .iter()
        .zip(dipole.iter())
        .zip(magnetic_dipole.iter())
        .zip(electric_quadrupole.iter())
        .enumerate()
    {
        let expected = electric_quadrupole + magnetic_dipole - dipole;
        // Every operand was independently serialized with FEFF's zero-scaled
        // E13.5 format. Its five significant digits contribute at most about
        // 5e-5 of each value to this four-term comparison.
        let tolerance = 2.0e-7
            + 5.1e-5
                * (actual.abs() + dipole.abs() + magnetic_dipole.abs() + electric_quadrupole.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "serialized combined value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual,
            expected,
            tolerance
        );
    }
}

fn assert_serialized_combined_complex_column_close(
    actual: &Array1<Complex64>,
    dipole: &Array1<Complex64>,
    magnetic_dipole: &Array1<Complex64>,
    electric_quadrupole: &Array1<Complex64>,
) {
    assert_eq!(actual.len(), dipole.len());
    assert_eq!(actual.len(), magnetic_dipole.len());
    assert_eq!(actual.len(), electric_quadrupole.len());
    for (index, (((actual, dipole), magnetic_dipole), electric_quadrupole)) in actual
        .iter()
        .zip(dipole.iter())
        .zip(magnetic_dipole.iter())
        .zip(electric_quadrupole.iter())
        .enumerate()
    {
        let expected = electric_quadrupole + magnetic_dipole - dipole;
        let real_tolerance = 2.0e-7
            + 5.1e-5
                * (actual.re.abs()
                    + dipole.re.abs()
                    + magnetic_dipole.re.abs()
                    + electric_quadrupole.re.abs());
        assert!(
            (actual.re - expected.re).abs() <= real_tolerance,
            "serialized combined real value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual.re,
            expected.re,
            real_tolerance
        );
        let imaginary_tolerance = 2.0e-7
            + 5.1e-5
                * (actual.im.abs()
                    + dipole.im.abs()
                    + magnetic_dipole.im.abs()
                    + electric_quadrupole.im.abs());
        assert!(
            (actual.im - expected.im).abs() <= imaginary_tolerance,
            "serialized combined imaginary value differs at row {}: actual={} expected={} tolerance={}",
            index + 1,
            actual.im,
            expected.im,
            imaginary_tolerance
        );
    }
}

fn assert_complex_array2_close_mixed(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for ((row, column), actual) in actual.indexed_iter() {
        let expected = expected[(row, column)];
        let real_tolerance = absolute_tolerance + relative_tolerance * expected.re.abs();
        assert!(
            (actual.re - expected.re).abs() <= real_tolerance,
            "real value differs at ({}, {}): actual={} expected={} tolerance={} (absolute={}, relative={})",
            row + 1,
            column + 1,
            actual.re,
            expected.re,
            real_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
        let imaginary_tolerance = absolute_tolerance + relative_tolerance * expected.im.abs();
        assert!(
            (actual.im - expected.im).abs() <= imaginary_tolerance,
            "imaginary value differs at ({}, {}): actual={} expected={} tolerance={} (absolute={}, relative={})",
            row + 1,
            column + 1,
            actual.im,
            expected.im,
            imaginary_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
    }
}

fn assert_xsecl_dat_close(
    actual: &XseclDatData,
    expected: &XseclDatData,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(
        actual.header.real_energy_count,
        expected.header.real_energy_count
    );
    assert_eq!(actual.header.fermi_index, expected.header.fermi_index);
    assert!(
        (actual.header.edge - expected.header.edge).abs() <= 1.0e-5,
        "xsecl edge differs: actual={} expected={} delta={}",
        actual.header.edge,
        expected.header.edge,
        (actual.header.edge - expected.header.edge).abs()
    );
    assert!(
        (actual.header.emu - expected.header.emu).abs() <= 1.0e-5,
        "xsecl emu differs: actual={} expected={} delta={}",
        actual.header.emu,
        expected.header.emu,
        (actual.header.emu - expected.header.emu).abs()
    );
    assert!(
        (actual.header.core_hole_width - expected.header.core_hole_width).abs() <= 1.0e-5,
        "xsecl core-hole width differs: actual={} expected={} delta={}",
        actual.header.core_hole_width,
        expected.header.core_hole_width,
        (actual.header.core_hole_width - expected.header.core_hole_width).abs()
    );
    assert_column_close_mixed(&actual.energy, &expected.energy, 1.0e-5, 0.0);
    assert_complex_column_close_mixed(
        &actual.channel_sum,
        &expected.channel_sum,
        absolute_tolerance,
        relative_tolerance,
    );
    assert_complex_array2_close_mixed(
        &actual.channel_cross_sections,
        &expected.channel_cross_sections,
        absolute_tolerance,
        relative_tolerance,
    );
}

fn assert_xsecl_bin_close(
    actual: &XseclBinData,
    expected: &XseclBinData,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.pad_width, expected.pad_width);
    assert_eq!(actual.initial_state_j, expected.initial_state_j);
    assert_eq!(actual.transitions, expected.transitions);
    assert_eq!(
        actual.final_state_count(),
        expected.final_state_count(),
        "xsecl.bin final-state count differs"
    );
    assert!(
        actual.energy_count() <= expected.energy_count(),
        "generated xsecl.bin has more energy rows than reference: actual={} expected={}",
        actual.energy_count(),
        expected.energy_count()
    );
    for ((row, column), actual) in actual.atom_cross_sections.indexed_iter() {
        let expected = expected.atom_cross_sections[(row, column)];
        let real_tolerance = absolute_tolerance + relative_tolerance * expected.re.abs();
        assert!(
            (actual.re - expected.re).abs() <= real_tolerance,
            "xsecl.bin real value differs at ({}, {}): actual={} expected={} tolerance={} (absolute={}, relative={})",
            row + 1,
            column + 1,
            actual.re,
            expected.re,
            real_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
        let imaginary_tolerance = absolute_tolerance + relative_tolerance * expected.im.abs();
        assert!(
            (actual.im - expected.im).abs() <= imaginary_tolerance,
            "xsecl.bin imaginary value differs at ({}, {}): actual={} expected={} tolerance={} (absolute={}, relative={})",
            row + 1,
            column + 1,
            actual.im,
            expected.im,
            imaginary_tolerance,
            absolute_tolerance,
            relative_tolerance
        );
    }
}

fn assert_phase_lmax_within_compiled_capacity(phase: &PhaseBinData) {
    assert!(
        phase
            .potentials
            .iter()
            .all(|potential| { potential.lmax >= 5 && potential.lmax <= 24 })
    );
}

fn max_complex_array2_delta_with_location(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
) -> (f64, (usize, usize)) {
    assert_eq!(actual.dim(), expected.dim());
    actual
        .indexed_iter()
        .map(|(location, actual)| ((*actual - expected[location]).norm(), location))
        .max_by(|(left, _), (right, _)| left.total_cmp(right))
        .unwrap_or((0.0, (0, 0)))
}

fn max_phase_shift_delta_with_location(
    actual: &PhaseBinData,
    expected: &PhaseBinData,
    start_energy: usize,
    end_energy: usize,
) -> (f64, (usize, usize, usize, usize)) {
    assert_eq!(actual.potential_count(), expected.potential_count());
    let mut max_delta = 0.0_f64;
    let mut location = (0_usize, 0_usize, 0_usize, 0_usize);
    for (potential_index, (actual, expected)) in actual
        .potentials
        .iter()
        .zip(expected.potentials.iter())
        .enumerate()
    {
        assert_eq!(actual.phase_shifts.dim(), expected.phase_shifts.dim());
        let (energy_count, l_count, spin_count) = actual.phase_shifts.dim();
        let end_energy = end_energy.min(energy_count);
        for energy_index in start_energy.min(end_energy)..end_energy {
            for l_index in 0..l_count {
                for spin_index in 0..spin_count {
                    let delta = (actual.phase_shifts[(energy_index, l_index, spin_index)]
                        - expected.phase_shifts[(energy_index, l_index, spin_index)])
                        .norm();
                    if delta > max_delta {
                        max_delta = delta;
                        location = (potential_index, energy_index, l_index, spin_index);
                    }
                }
            }
        }
    }
    (max_delta, location)
}

fn reference_xsph_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["xsph.inp", "global.inp", "phase.bin", "xsect.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_xsph_with_pot_and_mpse_dir() -> Result<Option<PathBuf>> {
    let Some(path) = reference_xsph_dir()? else {
        return Ok(None);
    };
    Ok((path.join("pot.bin").is_file() && path.join("mpse.dat").is_file()).then_some(path))
}

fn reference_mpse_cu_opcons_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/MPSE/Cu_OPCONS");
    let required = ["xsph.inp", "phase.bin", "pot.bin", "loss.dat", "mpse.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_xsph_with_pot_config_dir() -> Result<Option<PathBuf>> {
    let Some(path) = reference_xsph_dir()? else {
        return Ok(None);
    };
    Ok((path.join("pot.bin").is_file()
        && path.join("config.dat").is_file()
        && path.join("global.inp").is_file())
    .then_some(path))
}

fn reference_xanes_xsph_with_pot_config_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/XANES/Cu");
    let required = [
        "xsph.inp",
        "phase.bin",
        "xsect.dat",
        "pot.bin",
        "config.dat",
        "global.inp",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_danes_xsph_with_pot_config_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/DANES/Cu");
    let required = [
        "xsph.inp",
        "phase.bin",
        "xsect.dat",
        "pot.bin",
        "config.dat",
        "global.inp",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_xes_cu_xsph_zip() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/XES/Cu/REFERENCE.zip");
    Ok(path.is_file().then_some(path))
}

fn write_reference_zip_entries<'a>(
    zip_path: &Path,
    target_dir: &Path,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    write_reference_zip_required_entries(zip_path, target_dir, names)
}

fn write_reference_zip_required_entries<'a>(
    zip_path: &Path,
    target_dir: &Path,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    for name in names {
        let entry = format!("REFERENCE/{name}");
        let bytes = unzip_reference_entry(zip_path, &entry)?;
        std::fs::write(target_dir.join(name), bytes)
            .with_context(|| format!("failed to write extracted {entry}"))?;
    }
    Ok(())
}

fn write_reference_zip_optional_entries<'a>(
    zip_path: &Path,
    target_dir: &Path,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    for name in names {
        let entry = format!("REFERENCE/{name}");
        if let Some(bytes) = unzip_reference_entry_if_present(zip_path, &entry)? {
            std::fs::write(target_dir.join(name), bytes)
                .with_context(|| format!("failed to write extracted {entry}"))?;
        }
    }
    Ok(())
}

fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
    let output = Command::new("unzip")
        .arg("-p")
        .arg(zip_path)
        .arg(entry)
        .output()
        .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "failed to extract {entry} from {}: {stderr}",
            zip_path.display()
        );
    }
    Ok(output.stdout)
}

fn unzip_reference_entry_if_present(zip_path: &Path, entry: &str) -> Result<Option<Vec<u8>>> {
    let output = Command::new("unzip")
        .arg("-p")
        .arg(zip_path)
        .arg(entry)
        .output()
        .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
    Ok(output.status.success().then_some(output.stdout))
}

fn reference_xsph_with_pot_and_emesh_dir() -> Result<Option<PathBuf>> {
    let Some(path) = reference_xsph_dir()? else {
        return Ok(None);
    };
    Ok((path.join("pot.bin").is_file()
        && path.join("global.inp").is_file()
        && path.join("emesh.dat").is_file()
        && path.join("emesh.bin").is_file())
    .then_some(path))
}

fn reference_nrixs_xsph_with_pot_and_emesh_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/NRIXS/GeCl_4");
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_fprime_xsph_with_pot_and_emesh_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/FPRIME/GeCl4");
    let required = [
        "xsph.inp",
        "global.inp",
        "pot.bin",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_fprime_xsph_with_pot_config_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/FPRIME/GeCl4");
    let required = [
        "xsph.inp",
        "pot.bin",
        "config.dat",
        "global.inp",
        "xsect.dat",
        "phase.bin",
        "emesh.dat",
        "emesh.bin",
    ];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn synthetic_bphl_dat() -> String {
    let mut text = String::new();
    for radius in 1..=BPHL_RADIUS_COUNT {
        for reduced_energy in 1..BPHL_REDUCED_ENERGY_COUNT {
            let radius = radius as f64;
            let reduced_energy = reduced_energy as f64;
            text.push_str(&format!(
                "{radius:.6E} {reduced_energy:.6E} 0.000000E0 0.000000E0\n"
            ));
        }
    }
    text
}
