use super::{
    GenfmtGenerationDriver, genfmt_output_filenames, has_cached_genfmt_output,
    prepare_generation_context, read_input, run_in_dir,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
use refeff_io::{
    CfAverage, FeffBinData, FeffBinPath, FeffBinPotential, GenfmtControl, GenfmtInput,
    GlobalControl, GlobalInput, GlobalNorms, GlobalQControl, HubbardInput, ListDatData,
    ListDatEntry, ModuleLogData, NStarDatData, NStarDatEntry, PathsDatAtom, PathsDatData,
    PathsDatPath, PhaseBinData, PhaseBinPotential, PhaseBinScalars, genfmt_input_string,
    global_input_string, hubbard_input_string, read_feff_bin, read_feffl_bin, read_list_dat,
    read_module_log_dat, read_nstar_dat, write_feff_bin, write_feffl_bin, write_list_dat,
    write_module_log_dat, write_nstar_dat, write_paths_dat, write_phase_bin,
};
use std::path::{Path, PathBuf};

#[test]
fn genfmt_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 0)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_genfmt_output(temp.path())?);
    Ok(())
}

#[test]
fn genfmt_module_rejects_generation_without_cache_or_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled GENFMT should require cache files or handoffs")?;

    assert!(
        error.to_string().contains(
            "GENFMT generation requires cached feff.bin/list.dat outputs or global.inp, phase.bin, and paths.dat handoffs"
        )
    );
    Ok(())
}

#[test]
fn genfmt_module_prepares_generation_handoffs_before_solver_loop() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    assert!(has_cached_genfmt_output(temp.path())?);
    let input = read_input(temp.path())?;

    let context = prepare_generation_context(temp.path(), &input)?;

    assert_eq!(context.prepared_counts(), (3, 1, 1));
    assert_eq!(context.validate_prepared_path_inputs()?, 1);
    match &context.driver {
        GenfmtGenerationDriver::Ordinary(data) => {
            let setup = &data.setup;
            assert_eq!(setup.header.version, "refeff-rust");
            assert_eq!(setup.header.order, 2);
            assert_eq!(setup.header.potentials[0].label, "Cu");
            assert_eq!(setup.header.potentials[1].atomic_number, 8);
            assert_eq!(data.transition_b_matrix.matrix.shape(), &[3, 2, 8, 3, 2, 8]);
            assert_eq!(data.spin_radial_factors.dim(), (3, 8, 1));
            assert_eq!(data.transition_matrices.len(), 1);
            assert_eq!(
                data.transition_matrices[0].matrices.shape(),
                &[1, 9, 8, 9, 8]
            );
        }
        GenfmtGenerationDriver::Jas(_) => panic!("ordinary GENFMT handoff expected"),
    }
    assert_eq!(context.path_setups.len(), 1);
    let path_setup = &context.path_setups[0];
    assert_eq!(
        path_setup.rotations.real_leg_count,
        context.paths[0].leg_count()
    );
    assert_eq!(path_setup.lambda.order, 3);
    assert_eq!(path_setup.lambda.max_m_plus_one, 2);
    assert_eq!(path_setup.rotations.rotations.shape(), &[2, 2, 3, 3]);
    assert_eq!(context.legendre_normalization.shape(), &[25, 25]);
    assert!(path_setup.rotations.polarized_extra_rotation()?.is_none());
    assert!(context.nstar_rows.is_none());

    let written = run_in_dir(temp.path())?;
    assert_eq!(written, 3);
    let feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let list = read_list_dat(temp.path().join("list.dat"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.order, 2);
    assert_eq!(feff.potentials[0].label, "Cu");
    assert_eq!(feff.energy_count(), 3);
    assert_eq!(list.titles, vec!["PATH  Rmax= 6.000".to_string()]);
    assert!(
        read_module_log_dat(temp.path().join("log5.dat"))?
            .lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters"))
    );
    assert!(has_cached_genfmt_output(temp.path())?);
    Ok(())
}

#[test]
fn genfmt_module_does_not_advertise_malformed_base_cache_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("feff.bin"), b"not a feff.bin cache\n")?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;

    assert!(!has_cached_genfmt_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed GENFMT base cache should fail without source handoffs")?;

    assert!(error.to_string().contains("failed to read"));
    Ok(())
}

#[test]
fn genfmt_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("genfmt.inp"),
        b"not a genfmt.inp handoff\n",
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;

    assert!(!has_cached_genfmt_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed GENFMT input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("genfmt.inp"), "{chain}");
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn genfmt_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;

    assert!(!has_cached_genfmt_output(temp.path())?);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn genfmt_module_does_not_claim_cached_output_with_malformed_phase_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin source\n")?;

    assert!(!has_cached_genfmt_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed GENFMT phase source should block cached GENFMT completion")?;
    let chain = format!("{error:#}");

    assert!(chain.contains("phase.bin"), "{chain}");
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert!(!temp.path().join("log5.dat").exists());
    Ok(())
}

#[test]
fn genfmt_module_recovers_malformed_feff_bin_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    std::fs::write(temp.path().join("feff.bin"), b"not a feff.bin cache\n")?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;

    assert!(has_cached_genfmt_output(temp.path())?);

    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 3);
    let feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let list = read_list_dat(temp.path().join("list.dat"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.energy_count(), 3);
    assert_eq!(list.titles, vec!["PATH  Rmax= 6.000".to_string()]);
    assert!(
        read_module_log_dat(temp.path().join("log5.dat"))?
            .lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters"))
    );
    Ok(())
}

#[test]
fn genfmt_module_regenerates_stale_readable_base_cache_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    run_in_dir(temp.path())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let mut stale_feff = expected_feff.clone();
    stale_feff.version = "stale-cache".to_string();
    write_feff_bin(temp.path().join("feff.bin"), &stale_feff)?;

    assert!(has_cached_genfmt_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 3);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    Ok(())
}

#[test]
fn genfmt_module_regenerates_missing_nstar_from_source_handoffs_with_readable_base_cache()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input_with_nstar(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.4)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    run_in_dir(temp.path())?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    std::fs::remove_file(temp.path().join("nstar.dat"))?;

    assert!(has_cached_genfmt_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 4);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    Ok(())
}

#[test]
fn genfmt_module_regenerates_stale_nstar_from_source_handoffs_with_readable_base_cache()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input_with_nstar(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.4)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    run_in_dir(temp.path())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    let mut stale_nstar = expected_nstar.clone();
    stale_nstar.entries[0].nstar += 1.0;
    write_nstar_dat(temp.path().join("nstar.dat"), &stale_nstar)?;

    assert!(has_cached_genfmt_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 4);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    Ok(())
}

#[test]
fn genfmt_module_generates_active_hubbard_source_handoff_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    write_active_hubbard_input(temp.path())?;

    assert!(has_cached_genfmt_output(temp.path())?);

    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 3);
    let feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let list = read_list_dat(temp.path().join("list.dat"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.energy_count(), 3);
    assert_eq!(list.titles, vec!["PATH  Rmax= 6.000".to_string()]);
    assert!(
        read_module_log_dat(temp.path().join("log5.dat"))?
            .lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters"))
    );
    Ok(())
}

#[test]
fn genfmt_module_does_not_advertise_unsupported_phase_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.0)?;
    let mut phase = sample_phase_bin_data();
    phase.final_state_count = 4;
    phase.transition_count = 2;
    phase.transition_moments = Array4::zeros((
        phase.energy_count,
        phase.q_count,
        phase.transition_count,
        phase.spin_count,
    ));
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;

    assert!(!has_cached_genfmt_output(temp.path())?);
    Ok(())
}

#[test]
fn genfmt_module_prepares_nstar_handoff_rows_before_solver_loop() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input_with_nstar(temp.path(), 1)?;
    write_global_input(temp.path(), 0, 0.4)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    let input = read_input(temp.path())?;

    let context = prepare_generation_context(temp.path(), &input)?;
    let nstar = context.nstar_rows.as_ref().context("nstar rows")?;

    assert_eq!(nstar.primary_polarization, [0.0, 0.0, 1.0]);
    assert_eq!(nstar.rows.len(), 1);
    assert_eq!(nstar.rows[0].path_number, 1);

    let written = run_in_dir(temp.path())?;
    assert_eq!(written, 4);
    let nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    assert_eq!(nstar.entries.len(), 1);
    assert_eq!(nstar.entries[0].path_number, 1);
    Ok(())
}

#[test]
fn genfmt_module_generates_jas_handoff_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input_with_nstar(temp.path(), 1)?;
    write_global_input(temp.path(), 1, 0.0)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_jas_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    let input = read_input(temp.path())?;

    let context = prepare_generation_context(temp.path(), &input)?;

    assert_eq!(context.prepared_counts(), (3, 1, 1));
    assert_eq!(context.validate_prepared_path_inputs()?, 1);
    match &context.driver {
        GenfmtGenerationDriver::Jas(data) => {
            assert_eq!(data.setup.header.version, "refeff-rust");
            assert_eq!(data.transition_setups.len(), 1);
            assert_eq!(
                data.transition_setups[0].transition_count.transition_count,
                1
            );
        }
        GenfmtGenerationDriver::Ordinary(_) => panic!("GENFMTJAS handoff expected"),
    }

    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 4);
    let feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let list = read_list_dat(temp.path().join("list.dat"))?;
    let nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    assert_eq!(feff.version, "refeff-rust");
    assert_eq!(feff.ihole, 1);
    assert_eq!(feff.energy_count(), 3);
    assert_eq!(list.titles, vec!["PATH  Rmax= 6.000".to_string()]);
    assert_eq!(nstar.entries.len(), 1);
    assert_eq!(nstar.entries[0].path_number, 1);
    assert!(
        read_module_log_dat(temp.path().join("log5.dat"))?
            .lines
            .iter()
            .any(|line| line.contains("Done with module: EXAFS parameters"))
    );
    assert!(has_cached_genfmt_output(temp.path())?);
    Ok(())
}

#[test]
fn genfmt_module_regenerates_stale_feffl_from_decomposed_jas_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input_with_decomposition(temp.path(), 1, 1)?;
    write_global_input_with_decomposition(temp.path(), 1)?;
    write_phase_bin(temp.path().join("phase.bin"), &sample_jas_phase_bin_data())?;
    write_paths_dat(temp.path().join("paths.dat"), &sample_paths_dat())?;
    run_in_dir(temp.path())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;
    let expected_feffl = read_feffl_bin(
        temp.path().join("feffl.bin"),
        expected_feff.pad_width,
        expected_feff.paths.len(),
        expected_feff.energy_count(),
        1,
    )?;
    let mut stale_feffl = expected_feffl.clone();
    stale_feffl.amplitudes[(0, 0, 0, 0)] += 0.5;
    write_feffl_bin(temp.path().join("feffl.bin"), &stale_feffl)?;

    assert!(has_cached_genfmt_output(temp.path())?);
    let written = run_in_dir(temp.path())?;

    assert_eq!(written, 5);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    assert_eq!(
        read_feffl_bin(
            temp.path().join("feffl.bin"),
            expected_feff.pad_width,
            expected_feff.paths.len(),
            expected_feff.energy_count(),
            1,
        )?,
        expected_feffl
    );
    Ok(())
}

#[test]
fn genfmt_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_module_log_dat(temp.path().join("log5.dat"), &sample_module_log())?;
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_log = read_module_log_dat(temp.path().join("log5.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(has_cached_genfmt_output(temp.path())?);
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    assert_eq!(
        read_module_log_dat(temp.path().join("log5.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn genfmt_module_roundtrips_optional_nstar_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_nstar_dat(temp.path().join("nstar.dat"), &sample_nstar_dat())?;
    write_module_log_dat(temp.path().join("log5.dat"), &sample_module_log())?;
    let expected_nstar = read_nstar_dat(temp.path().join("nstar.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_nstar_dat(temp.path().join("nstar.dat"))?,
        expected_nstar
    );
    Ok(())
}

#[test]
fn genfmt_output_filenames_match_feff_elnes_reference() -> Result<()> {
    let base = genfmt_output_filenames(1)?;
    assert_eq!(base.feff_bin, "feff.bin");
    assert_eq!(base.list_dat, "list.dat");

    let second = genfmt_output_filenames(2)?;
    assert_eq!(second.feff_bin, "feff02.bin");
    assert_eq!(second.list_dat, "list02.dat");

    let ninth = genfmt_output_filenames(9)?;
    assert_eq!(ninth.feff_bin, "feff09.bin");
    assert_eq!(ninth.list_dat, "list09.dat");

    let tenth = genfmt_output_filenames(10)?;
    assert_eq!(tenth.feff_bin, "feff10.bin");
    assert_eq!(tenth.list_dat, "list10.dat");

    assert!(genfmt_output_filenames(0).is_err());
    assert!(genfmt_output_filenames(11).is_err());
    Ok(())
}

#[test]
fn genfmt_module_roundtrips_elnes_suffixed_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    for polarization_index in [1, 2, 10] {
        let filenames = genfmt_output_filenames(polarization_index)?;
        write_feff_bin(temp.path().join(&filenames.feff_bin), &feff)?;
        write_list_dat(temp.path().join(&filenames.list_dat), &list)?;
    }
    write_module_log_dat(temp.path().join("log5.dat"), &sample_module_log())?;
    std::fs::write(temp.path().join("feff2.bin"), "not a FEFF cache")?;
    std::fs::write(temp.path().join("list2.dat"), "not a list cache")?;
    std::fs::write(temp.path().join("feff11.bin"), "not a FEFF cache")?;
    std::fs::write(temp.path().join("list11.dat"), "not a list cache")?;

    let expected_feff02 = read_feff_bin(temp.path().join("feff02.bin"))?;
    let expected_feff10 = read_feff_bin(temp.path().join("feff10.bin"))?;
    let expected_list02 = read_list_dat(temp.path().join("list02.dat"))?;
    let expected_list10 = read_list_dat(temp.path().join("list10.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 7);
    assert_eq!(
        read_feff_bin(temp.path().join("feff02.bin"))?,
        expected_feff02
    );
    assert_eq!(
        read_feff_bin(temp.path().join("feff10.bin"))?,
        expected_feff10
    );
    assert_eq!(
        read_list_dat(temp.path().join("list02.dat"))?,
        expected_list02
    );
    assert_eq!(
        read_list_dat(temp.path().join("list10.dat"))?,
        expected_list10
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("feff11.bin"))?,
        "not a FEFF cache"
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("list11.dat"))?,
        "not a list cache"
    );
    Ok(())
}

#[test]
fn genfmt_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let log = read_module_log_dat(temp.path().join("log5.dat"))?;
    assert_log_contains(&log, "Calculating EXAFS parameters ...");
    assert_log_contains(&log, "Done with module: EXAFS parameters (GENFMT).");
    Ok(())
}

#[test]
fn genfmt_module_generates_nrixs_spherical_average_log_line() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_genfmt_input(temp.path(), 1)?;
    write_global_input(temp.path(), 1, -1.0)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let log = read_module_log_dat(temp.path().join("log5.dat"))?;
    assert_log_contains(
        &log,
        "Spherically averaged NRIXS in module genft - setting jinit=jmax.",
    );
    assert_log_contains(&log, "Calculating EXAFS parameters ...");
    Ok(())
}

#[test]
fn genfmt_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_genfmt_dir()? else {
        eprintln!("skipping GENFMT reference test; generated EXAFS/Cu reference not found");
        return Ok(());
    };

    let temp = tempfile::tempdir()?;
    for name in ["genfmt.inp", "feff.bin", "list.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let log_source = reference_dir.join("log5.dat");
    if log_source.is_file() {
        std::fs::copy(log_source, temp.path().join("log5.dat"))?;
    }
    let expected_feff = read_feff_bin(temp.path().join("feff.bin"))?;
    let expected_list = read_list_dat(temp.path().join("list.dat"))?;
    let expected_log = optional_module_log(temp.path().join("log5.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2 + usize::from(expected_log.is_some()));
    assert_eq!(read_feff_bin(temp.path().join("feff.bin"))?, expected_feff);
    assert_eq!(read_list_dat(temp.path().join("list.dat"))?, expected_list);
    if let Some(expected) = expected_log {
        assert_eq!(read_module_log_dat(temp.path().join("log5.dat"))?, expected);
    }
    Ok(())
}

fn write_genfmt_input(work_dir: &Path, mfeff: i32) -> Result<()> {
    write_genfmt_input_options(work_dir, mfeff, false)
}

fn write_genfmt_input_with_nstar(work_dir: &Path, mfeff: i32) -> Result<()> {
    write_genfmt_input_options(work_dir, mfeff, true)
}

fn write_genfmt_input_with_decomposition(
    work_dir: &Path,
    mfeff: i32,
    decomposition_channels: i32,
) -> Result<()> {
    let input = GenfmtInput {
        control: GenfmtControl {
            mfeff,
            ipr5: 0,
            iorder: 2,
            critcw: 4.0,
            wnstar: true,
        },
        decomposition_channels,
    };
    std::fs::write(work_dir.join("genfmt.inp"), genfmt_input_string(&input)?)?;
    Ok(())
}

fn write_active_hubbard_input(work_dir: &Path) -> Result<()> {
    let input = HubbardInput {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: 4.0,
        j: 0.5,
        fermi_shift: 0.0,
        l: 2,
    };
    std::fs::write(work_dir.join("hubbard.inp"), hubbard_input_string(&input)?)?;
    Ok(())
}

fn write_genfmt_input_options(work_dir: &Path, mfeff: i32, wnstar: bool) -> Result<()> {
    let input = GenfmtInput {
        control: GenfmtControl {
            mfeff,
            ipr5: 0,
            iorder: 2,
            critcw: 4.0,
            wnstar,
        },
        decomposition_channels: -1,
    };
    std::fs::write(work_dir.join("genfmt.inp"), genfmt_input_string(&input)?)?;
    Ok(())
}

fn write_global_input(work_dir: &Path, do_nrixs: i32, elpty: f64) -> Result<()> {
    write_global_input_options(work_dir, do_nrixs, elpty, -1)
}

fn write_global_input_with_decomposition(work_dir: &Path, ldecmx: i32) -> Result<()> {
    write_global_input_options(work_dir, 1, 0.0, ldecmx)
}

fn write_global_input_options(
    work_dir: &Path,
    do_nrixs: i32,
    elpty: f64,
    ldecmx: i32,
) -> Result<()> {
    let jas = do_nrixs == 1;
    let input = GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: if jas { 1 } else { 0 },
            ispin: 0,
            le2: 0,
            elpty,
            angks: 0.0,
            l2lp: if jas { 1 } else { 0 },
            do_nrixs,
            ldecmx,
            lj: -1,
        },
        evec: [0.0, 0.0, 1.0],
        xivec: [1.0, 0.0, 0.0],
        spvec: [0.0, 0.0, 1.0],
        polarization_tensor: [[0.0; 6]; 3],
        norms: GlobalNorms {
            evnorm: 1.0,
            xivnorm: 1.0,
            spvnorm: 1.0,
        },
        q_control: GlobalQControl {
            nq: 0,
            imdff: 0,
            qaverage: jas,
            mixdff: false,
        },
        q_vectors: Vec::new(),
        mdff: None,
    };
    std::fs::write(work_dir.join("global.inp"), global_input_string(&input)?)?;
    Ok(())
}

fn sample_feff_bin_data() -> FeffBinData {
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
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
        central_phase_shift: Array1::from_vec(vec![
            Complex64::new(0.1, -0.01),
            Complex64::new(0.2, -0.02),
            Complex64::new(0.3, -0.03),
        ]),
        complex_momentum: Array1::from_vec(vec![
            Complex64::new(1.0, 0.1),
            Complex64::new(1.1, 0.2),
            Complex64::new(1.2, 0.3),
        ]),
        real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
        paths: vec![FeffBinPath {
            index: 17,
            degeneracy: 4.0,
            effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
            criterion: 12.5,
            potential_indices: Array1::from_vec(vec![0, 1, 0]),
            positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                (0, 0..=2) => 0.0,
                (1, 0) => 1.0,
                (1, 1) => 0.5,
                (1, 2) => 0.0,
                (2, 0) => -1.0,
                (2, 1) => 0.25,
                (2, 2) => 0.0,
                _ => 0.0,
            }),
            beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
            eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
            leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
            phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
        }],
        raw_text: None,
    }
}

fn sample_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.0,
            amplitude_ratio: 12.5,
            degeneracy: 4.0,
            leg_count: 3,
            effective_half_path_length_angstrom: 2.5,
        }],
    }
}

fn sample_nstar_dat() -> NStarDatData {
    NStarDatData {
        polarization: [1.0, 0.0, -0.5],
        entries: vec![NStarDatEntry {
            path_number: 17,
            nstar: 2.345,
        }],
    }
}

fn sample_phase_bin_data() -> PhaseBinData {
    let spin_count = 1;
    let energy_count = 3;
    let transition_count = 8;
    let q_count = 1;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 2,
        auxiliary_energy_count: 1,
        ihole: 1,
        fermi_index: 1,
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: transition_count,
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
            sample_phase_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_phase_potential(1, 8, "O", energy_count, spin_count, 0.2),
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

fn sample_jas_phase_bin_data() -> PhaseBinData {
    let mut data = sample_phase_bin_data();
    data.final_state_count = 12;
    data.transition_count = 1;
    data.q_count = 1;
    data.transition_moments = Array4::from_shape_fn(
        (
            data.energy_count,
            data.q_count,
            data.transition_count,
            data.spin_count,
        ),
        |(energy, q_index, transition, spin)| {
            Complex64::new(
                0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                -0.02 * spin as f64,
            )
        },
    );
    data
}

fn sample_phase_potential(
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

fn sample_paths_dat() -> PathsDatData {
    PathsDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        paths: vec![PathsDatPath {
            index: 17,
            degeneracy: 4.0,
            effective_half_path_length_angstrom: 2.5,
            row_header:
                "      x           y           z     ipot  label      rleg      beta        eta"
                    .to_string(),
            atoms: vec![
                PathsDatAtom {
                    position_angstrom: [1.0, 0.0, 0.0],
                    potential_index: 1,
                    label: "O".to_string(),
                    leg_distance_angstrom: Some(1.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(0.0),
                },
                PathsDatAtom {
                    position_angstrom: [0.0, 0.0, 0.0],
                    potential_index: 0,
                    label: "Cu".to_string(),
                    leg_distance_angstrom: Some(1.0),
                    beta_degrees: Some(90.0),
                    eta_degrees: Some(0.0),
                },
            ],
        }],
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS parameters ...".to_string(),
            "Done with module: EXAFS parameters (GENFMT).".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
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

fn reference_genfmt_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["genfmt.inp", "feff.bin", "list.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}
