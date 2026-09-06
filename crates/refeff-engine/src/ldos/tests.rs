use super::{
    LDOS_SOURCE_REQUIREMENT_ERROR, has_cached_ldos_output, has_supported_kmesh_handoff,
    has_supported_source_output_handoff, run_in_dir, run_supported_kmesh_handoff_in_dir,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, Axis, ShapeBuilder, array};
use num_complex::{Complex32, Complex64};
use refeff_core::FEFF_HARTREE_EV;
use refeff_io::pot_bin::{
    POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
    POT_BIN_RADIAL_POINTS,
};
use refeff_io::{
    CONFIG_DAT_ORBITAL_COUNT, CfAverage, ConfigDatData, ConfigDatPotential, FmsCluster, FmsControl,
    FmsDebye, FmsInput, GeomDat, GeomDatRow, GlobalControl, GlobalInput, GlobalNorms,
    GlobalQControl, GtrBinData, HubbardInput, HubbardLdosGtrBinData, HubbardLdosGtrMBinData,
    HubbardLdosGtrOffBinData, HubbardVnlmBinData, KmeshMetadata, KmeshRow, LdosDatData,
    LdosMagneticDatData, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
    PotBinData, PotBinScalars, PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp,
    PotRun, PotScattering, PotThermal, PotTolerances, ReciprocalCell, ReciprocalInput,
    ReciprocalKMesh, config_dat_string, fms_input_string, geom_dat_string, global_input_string,
    hubbard_input_string, ldos_input_string, pot_input_string, read_gtr_bin, read_kmesh_dat,
    read_ldos_dat, read_lmdos_dat, read_module_log_dat, read_pot_bin, read_rhoc_dat,
    read_rhocm_dat, read_v_hubbard_bin_inferred, reciprocal_input_string, write_gtr_bin,
    write_hubbard_ldos_gtr_bin, write_hubbard_ldos_gtr_m_bin, write_hubbard_ldos_gtr_off_bin,
    write_ldos_dat, write_lmdos_dat, write_module_log_dat, write_phase_bin, write_pot_bin,
    write_rhoc_dat, write_rhocm_dat, write_v_hubbard_bin,
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[test]
fn ldos_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), false)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled LDOS should require complete source state")?;

    assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
    Ok(())
}

#[test]
fn ldos_module_generates_kmesh_before_source_requirement() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled LDOS should still require complete source state")?;

    assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
    let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(data.rows.len(), 8);
    assert_eq!(
        data.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    Ok(())
}

#[test]
fn ldos_module_generates_gtr_bin_from_source_fms_handoffs_before_source_requirement() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_ldos_source_input(temp.path())?;
    write_ldos_fms_source_handoffs(temp.path())?;

    let error = run_in_dir(temp.path())
        .err()
        .context("LDOS should still require complete source state")?;

    assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
    let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    assert_eq!(gtr00.energy_count(), 2);
    assert_eq!(gtr00.potential_count(), 2);
    assert_eq!(gtr00.angular_channel_count(), 2);
    assert_eq!(gtr00.highest_potential_index, 1);
    assert_eq!(gtr00.fms_mode, 2);
    assert!(
        gtr00
            .values
            .index_axis(Axis(1), 0)
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(
        gtr00
            .values
            .index_axis(Axis(1), 1)
            .iter()
            .all(|value| value.norm() == 0.0)
    );
    let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
    assert_eq!(gtr01.energy_count(), 2);
    assert_eq!(gtr01.potential_count(), 2);
    assert_eq!(gtr01.angular_channel_count(), 2);
    assert!(
        gtr01
            .values
            .index_axis(Axis(1), 0)
            .iter()
            .all(|value| value.norm() == 0.0)
    );
    assert!(
        gtr01
            .values
            .index_axis(Axis(1), 1)
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_skips_fms_source_when_ldos_mesh_differs_from_phase_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_source_input_with_neldos(temp.path(), 3)?;
    write_ldos_fms_source_handoffs(temp.path())?;

    let error = run_in_dir(temp.path())
        .err()
        .context("mismatched LDOS/FMS mesh should stop at the source requirement")?;

    assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
    assert!(!temp.path().join("gtr00.bin").exists());
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_regenerates_stale_fms_gtr_on_ldos_mesh() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_neldos(temp.path(), 1, 3)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;
    write_gtr_bin(temp.path().join("gtr00.bin"), &sample_ldos_gtr_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    assert_eq!(gtr00.energy_count(), 3);
    assert_eq!(gtr00.angular_channel_count(), 4);
    assert!(
        gtr00
            .values
            .index_axis(Axis(1), 0)
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    assert!(
        gtr00
            .values
            .index_axis(Axis(1), 1)
            .iter()
            .all(|value| value.norm() == 0.0)
    );
    let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
    assert_eq!(gtr01.energy_count(), 3);
    assert_eq!(gtr01.angular_channel_count(), 4);
    assert!(
        gtr01
            .values
            .index_axis(Axis(1), 0)
            .iter()
            .all(|value| value.norm() == 0.0)
    );
    assert!(
        gtr01
            .values
            .index_axis(Axis(1), 1)
            .iter()
            .any(|value| value.norm() > 0.0)
    );
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density.dim(), (3, 4));
        assert_eq!(rhoc.density.dim(), (3, 4));
        assert!(ldos.density.iter().all(|value| value.is_finite()));
        assert!(rhoc.density.iter().all(|value| value.is_finite()));
    }
    Ok(())
}

#[test]
fn ldos_module_generates_zero_fms_trace_for_nonpositive_cluster_radius() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input(temp.path())?;
    let input_text = std::fs::read_to_string(temp.path().join("ldos.inp"))?;
    let mut input = refeff_io::LdosInput::parse_str(temp.path().join("ldos.inp"), &input_text)?;
    input.mesh.rfms2 = -1.0;
    std::fs::write(temp.path().join("ldos.inp"), ldos_input_string(&input)?)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;

    assert!(has_supported_source_output_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!has_supported_source_output_handoff(temp.path())?);
    let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    assert_eq!(gtr00.energy_count(), 2);
    assert_eq!(gtr00.potential_count(), 2);
    assert_eq!(gtr00.fms_mode, 2);
    assert!(gtr00.values.iter().all(|value| value.norm() == 0.0));
    let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
    assert_eq!(gtr01.energy_count(), 2);
    assert_eq!(gtr01.potential_count(), 2);
    assert_eq!(gtr01.fms_mode, 2);
    assert!(gtr01.values.iter().all(|value| value.norm() == 0.0));
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density, rhoc.density);
    }
    Ok(())
}

#[test]
fn ldos_module_generates_tables_from_wavefunction_and_gtr_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input(temp.path())?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    write_gtr_bin(temp.path().join("gtr00.bin"), &sample_ldos_gtr_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density.dim(), (2, 4));
        assert_eq!(rhoc.density.dim(), (2, 4));
        assert!(ldos.density.iter().all(|value| value.is_finite()));
        assert!(rhoc.density.iter().all(|value| value.is_finite()));
        assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
        assert!(rhoc.density.iter().any(|value| value.abs() > 0.0));
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_generates_no_fms_tables_from_wavefunction_source_without_gtr() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;

    assert!(has_supported_source_output_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!has_supported_source_output_handoff(temp.path())?);
    assert!(!temp.path().join("gtr00.bin").exists());
    assert!(!temp.path().join("gtr01.bin").exists());
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density.dim(), (2, 4));
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_runs_independent_fms_for_each_potential_when_lfms2_is_zero() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    expand_ldos_wavefunction_source_valence_occupancy(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;

    assert!(has_supported_source_output_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
        assert_eq!(gtr.fms_mode, 2);
        assert!(
            gtr.values
                .index_axis(Axis(1), potential)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density.ncols(), 4);
        assert_eq!(ldos.electron_counts.len(), 4);
        assert_ne!(ldos.density, rhoc.density);
    }
    Ok(())
}

#[test]
fn ldos_module_falls_back_to_radial_no_fms_tables_when_spin_source_is_incomplete() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    let mut input = super::read_input(temp.path())?;
    input.control.ispin = 1;
    std::fs::write(temp.path().join("ldos.inp"), ldos_input_string(&input)?)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;

    assert!(!temp.path().join("xsph.inp").exists());
    assert!(has_supported_source_output_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("gtr00.bin").exists());
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density.ncols(), super::LDOS_ORBITAL_COUNT);
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
    }
    Ok(())
}

#[test]
fn ldos_module_uses_dimensions_lx_for_numeric_columns_and_electron_counts() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    expand_ldos_wavefunction_source_valence_occupancy(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;
    std::fs::write(
        temp.path().join(".dimensions.dat"),
        "       2       2       1       1\n",
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
        assert_eq!(gtr.angular_channel_count(), 3);
        assert!(matches!(
            super::cached_ldos_ordinary_source_contract(
                temp.path(),
                &format!("{potential:02}"),
            ),
            super::LdosSourceContract::Present(contract)
                if contract.density_column_count == 3
        ));
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.density.ncols(), 3);
        assert_eq!(rhoc.density.ncols(), 3);
        assert_eq!(ldos.electron_counts.len(), 3);
        assert!(
            ldos.header_lines
                .last()
                .is_some_and(|line| line.contains("fDOS"))
        );
    }
    Ok(())
}

#[test]
fn ldos_module_does_not_claim_malformed_wavefunction_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;

    assert!(!has_supported_source_output_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed LDOS wavefunction source should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("rhoc00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_does_not_claim_cached_tables_with_malformed_wavefunction_source_handoff()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;

    assert!(!has_cached_ldos_output(temp.path())?);
    assert!(!has_supported_source_output_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed LDOS wavefunction source should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_regenerates_stale_no_fms_tables_from_wavefunction_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    run_in_dir(temp.path())?;
    let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    let mut stale_ldos = expected_ldos.clone();
    let mut stale_rhoc = expected_rhoc.clone();
    stale_ldos.density[(0, 0)] += 0.25;
    stale_rhoc.density[(0, 0)] += 0.25;
    write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &stale_rhoc)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        expected_ldos
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        expected_rhoc
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_regenerates_stale_pre_fix_charge_transfer_header() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    run_in_dir(temp.path())?;
    let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    let mut stale_ldos = expected_ldos.clone();
    stale_ldos.charge_transfer = stale_ldos.charge_transfer.map(|charge| -charge);
    stale_ldos.header_lines.clear();
    assert_ne!(
        stale_ldos.charge_transfer, expected_ldos.charge_transfer,
        "fixture charge transfer must be nonzero to model the pre-fix qnrm sign"
    );
    write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;

    assert!(
        !has_cached_ldos_output(temp.path())?,
        "a sign-reversed qnrm header must invalidate an otherwise identical LDOS cache"
    );
    assert!(run_in_dir(temp.path())? > 0);

    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        expected_ldos
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        expected_rhoc
    );
    Ok(())
}

#[test]
fn ldos_module_regenerates_stale_fms_tables_from_wavefunction_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input(temp.path())?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;
    run_in_dir(temp.path())?;
    let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    let mut stale_ldos = expected_ldos.clone();
    let mut stale_rhoc = expected_rhoc.clone();
    stale_ldos.density[(0, 0)] += 0.25;
    stale_rhoc.density[(0, 0)] += 0.25;
    write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &stale_rhoc)?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        expected_ldos
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        expected_rhoc
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_generates_no_fms_tables_from_radial_source_without_fms_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    std::fs::remove_file(temp.path().join("fms.inp"))?;

    assert!(has_supported_source_output_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(!temp.path().join("gtr00.bin").exists());
    assert!(!temp.path().join("gtr01.bin").exists());
    for potential in 0..=1 {
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_hubbard_second_pass_solves_magnetic_radial_source_tables() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 1)?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;
    let v_hubbard = HubbardVnlmBinData {
        angular_limit: 3,
        values: Array4::from_shape_fn((2, 2, 4, 16), |(_, spin, angular, magnetic)| {
            if angular == 2 && (4..9).contains(&magnetic) {
                (spin as f64 - 0.5) * 1.0e-3
            } else {
                0.0
            }
        }),
    };
    write_v_hubbard_bin(temp.path().join("v_hubbard.bin"), &v_hubbard)?;
    let input = super::read_input(temp.path())?;
    let energy_grid = super::ldos_input_energy_grid_hartree(&input)?;

    let tables = crate::rhorrp::read_hubbard_ldos_rhol_source_tables(
        temp.path(),
        energy_grid.clone(),
        &[0, 1],
        4,
        &v_hubbard,
    )?;

    assert_eq!(tables.len(), 2);
    for (potential, table) in tables.iter().enumerate() {
        assert_eq!(table.potential_index, potential);
        assert_eq!(
            table.embedded_magnetic_ldos.dim(),
            (4, 16, 2, energy_grid.len())
        );
        assert_eq!(
            table.scattering_magnetic_ldos.dim(),
            (4, 16, 2, energy_grid.len())
        );
        assert!(
            table
                .embedded_magnetic_ldos
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            table
                .scattering_magnetic_ldos
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(
            table
                .embedded_magnetic_ldos
                .iter()
                .any(|value| value.abs() > 0.0)
        );
    }
    Ok(())
}

#[test]
fn ldos_module_generates_spin_hubbard_independent_center_tables_from_source_handoffs() -> Result<()>
{
    let Some(reference) = reference_hubbard_full_potential_source_dir()? else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    for name in [
        ".dimensions.dat",
        "config.dat",
        "fms.inp",
        "geom.dat",
        "global.inp",
        "hubbard.inp",
        "ldos.inp",
        "phase.bin",
        "pot.bin",
        "pot.inp",
        "xsph.inp",
    ] {
        std::fs::copy(reference.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy spin-Hubbard LDOS fixture {name}"))?;
    }
    reduce_hubbard_full_potential_energy_grid(temp.path())?;

    let count = run_in_dir(temp.path())?;
    assert!(count >= 18, "generated only {count} output files");
    for potential in 0..3 {
        let index = format!("{potential:02}");
        let ldos = read_ldos_dat(temp.path().join(format!("ldos{index}.dat")))?;
        let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{index}.dat")))?;
        let lmdos = read_lmdos_dat(temp.path().join(format!("lmdos{index}.dat")))?;
        let rhocm = read_rhocm_dat(temp.path().join(format!("rhocm{index}.dat")))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(lmdos.energy_ev, rhocm.energy_ev);
        assert_eq!(ldos.energy_ev, lmdos.energy_ev);
        assert!(
            lmdos
                .density
                .iter()
                .chain(rhocm.density.iter())
                .all(|value| value.is_finite())
        );
    }
    for name in [
        "gtr_m00.bin",
        "gtr_off00.bin",
        "transformation_hubbard.bin",
        "v_hubbard.bin",
    ] {
        assert!(temp.path().join(name).is_file(), "missing generated {name}");
    }
    let v_hubbard = read_v_hubbard_bin_inferred(temp.path().join("v_hubbard.bin"), 3)?;
    assert_eq!(v_hubbard.angular_limit, 2);
    assert!(
        v_hubbard.values.iter().all(|value| value.abs() <= 1.0e-12),
        "zero-lmax Hubbard compatibility pass produced a nonzero potential"
    );
    let transformation = refeff_io::read_transformation_hubbard_bin_inferred(
        temp.path().join("transformation_hubbard.bin"),
        2,
        3,
    )?;
    assert_eq!(transformation.angular_limit, 2);
    for potential in 0..3 {
        for spin in 0..2 {
            for angular in 0..=transformation.angular_limit {
                for row in 0..transformation.row_count() {
                    for column in 0..transformation.column_count() {
                        let expected = if row == column {
                            Complex32::new(1.0, 0.0)
                        } else {
                            Complex32::new(0.0, 0.0)
                        };
                        assert_eq!(
                            transformation.transform[(potential, spin, angular, row, column)],
                            expected
                        );
                        assert_eq!(
                            transformation.inverse[(potential, spin, angular, row, column)],
                            expected
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn ldos_module_limits_no_fms_source_tables_to_available_potentials() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_wavefunction_source_input_with_lfms2_and_lmax(temp.path(), 0, &[3, 3, 3])?;
    write_ldos_wavefunction_source_handoffs(temp.path())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(temp.path().join("ldos00.dat").is_file());
    assert!(temp.path().join("rhoc00.dat").is_file());
    assert!(temp.path().join("ldos01.dat").is_file());
    assert!(temp.path().join("rhoc01.dat").is_file());
    assert!(!temp.path().join("ldos02.dat").exists());
    assert!(!temp.path().join("rhoc02.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_generates_supported_kmesh_handoff_without_solver() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;

    assert!(has_supported_kmesh_handoff(temp.path())?);
    let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(data.rows.len(), 8);
    assert_eq!(
        data.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_does_not_claim_malformed_reciprocal_kmesh_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        "not a reciprocal.inp handoff\n",
    )?;

    assert!(!has_supported_kmesh_handoff(temp.path())?);
    assert_eq!(run_supported_kmesh_handoff_in_dir(temp.path())?, 0);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed reciprocal.inp should fail through the explicit LDOS runner")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("reciprocal.inp"), "{chain}");
    assert!(!temp.path().join("kmesh.dat").exists());
    assert!(!temp.path().join("ldos00.dat").exists());
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_recovers_existing_malformed_log_for_supported_kmesh_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;
    std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_supported_kmesh_handoff(temp.path())?);
    let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(data.rows.len(), 8);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    let log = sample_module_log();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_module_log_dat(temp.path().join("logdos.dat"), &log)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(read_module_log_dat(temp.path().join("logdos.dat"))?, log);
    Ok(())
}

#[test]
fn ldos_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    Ok(())
}

#[test]
fn ldos_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    std::fs::write(temp.path().join("ldos.inp"), "not an ldos.inp handoff\n")?;

    assert!(!has_cached_ldos_output(temp.path())?);
    assert!(!has_supported_source_output_handoff(temp.path())?);
    assert!(!has_supported_kmesh_handoff(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed ldos.inp should fail through the explicit LDOS runner")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("ldos.inp"), "{chain}");
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert!(!temp.path().join("logdos.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_roundtrips_active_hubbard_magnetic_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    let lmdos = sample_lmdos_dat();
    let rhocm = sample_rhocm_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
    assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_rejects_truncated_active_hubbard_gtr_as_ordinary_trace() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let mut truncated = sample_ldos_gtr_bin();
    truncated.values = Array3::from_shape_fn((2, 2, 3), |(energy, potential, angular)| {
        Complex64::new(
            0.08 + 0.02 * energy as f64 + 0.03 * potential as f64 + 0.01 * angular as f64,
            -0.04 + 0.005 * angular as f64,
        )
    });
    write_gtr_bin(temp.path().join("gtr00.bin"), &truncated)?;

    assert!(matches!(
        super::cached_ldos_ordinary_source_contract(temp.path(), "00"),
        super::LdosSourceContract::Incompatible
    ));
    Ok(())
}

#[test]
fn ldos_transient_hubbard_gtr_restores_preexisting_bytes_on_success_and_error() -> Result<()> {
    const ORIGINAL: &[u8] = b"preexisting truncated Hubbard GTR";
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let path = temp.path().join("gtr00.bin");
    std::fs::write(&path, ORIGINAL)?;

    let value =
        super::with_transient_active_hubbard_gtr_sources(temp.path(), 1, || -> Result<_> {
            std::fs::write(&path, b"successful transient replacement")?;
            Ok(7)
        })?;
    assert_eq!(value, 7);
    assert_eq!(std::fs::read(&path)?, ORIGINAL);

    let error =
        super::with_transient_active_hubbard_gtr_sources(temp.path(), 1, || -> Result<()> {
            std::fs::write(&path, b"failing transient replacement")?;
            Err(anyhow::anyhow!("induced generation failure"))
        })
        .expect_err("induced transient generation error should propagate");
    assert!(format!("{error:#}").contains("induced generation failure"));
    assert_eq!(std::fs::read(&path)?, ORIGINAL);
    Ok(())
}

#[test]
fn ldos_preserves_internal_hubbard_magnetic_traces_across_spectrum_operation() -> Result<()> {
    const ORIGINAL_MAGNETIC: &[u8] = b"101-point LDOS magnetic trace";
    const ORIGINAL_OFFDIAG: &[u8] = b"101-point LDOS off-diagonal trace";
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let magnetic_path = temp.path().join("gtr_m00.bin");
    let offdiag_path = temp.path().join("gtr_off00.bin");
    let new_specific_path = temp.path().join("gtr_m01.bin");
    std::fs::write(&magnetic_path, ORIGINAL_MAGNETIC)?;
    std::fs::write(&offdiag_path, ORIGINAL_OFFDIAG)?;

    let value = super::with_preserved_active_hubbard_ldos_magnetic_sources(
        temp.path(),
        || -> Result<_> {
            std::fs::write(&magnetic_path, b"83-point spectrum trace")?;
            std::fs::write(&offdiag_path, b"83-point spectrum off-diagonal trace")?;
            std::fs::write(&new_specific_path, b"transient spectrum trace")?;
            Ok(11)
        },
    )?;

    assert_eq!(value, 11);
    assert_eq!(std::fs::read(&magnetic_path)?, ORIGINAL_MAGNETIC);
    assert_eq!(std::fs::read(&offdiag_path)?, ORIGINAL_OFFDIAG);
    assert!(!new_specific_path.exists());
    Ok(())
}

#[test]
fn ldos_reports_active_hubbard_spectrum_bootstrap_only_before_v_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    std::fs::write(temp.path().join("xsph.inp"), "spectrum stage marker\n")?;

    assert!(super::active_hubbard_spectrum_bootstrap_pending(
        temp.path()
    )?);

    std::fs::write(temp.path().join("v_hubbard.bin"), b"source boundary marker")?;
    assert!(!super::active_hubbard_spectrum_bootstrap_pending(
        temp.path()
    )?);
    Ok(())
}

#[test]
fn ldos_module_accepts_active_hubbard_cache_with_matching_source_contracts() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    let lmdos = sample_lmdos_dat();
    let rhocm = sample_rhocm_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_hubbard_gtr_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_hubbard_gtr_m_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off00.bin"),
        &sample_hubbard_gtr_off_source_contract(2, 1, 3, 1),
    )?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
    assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_accepts_active_hubbard_cache_with_fallback_source_contracts() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_hubbard_gtr_source_contract(1, 3, 2),
    )?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_hubbard_gtr_m_source_contract(1, 3, 2),
    )?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off00.bin"),
        &sample_hubbard_gtr_off_source_contract(2, 1, 3, 2),
    )?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos01.dat"))?,
        sample_ldos_dat()
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc01.dat"))?,
        sample_rhoc_dat()
    );
    assert_eq!(
        read_lmdos_dat(temp.path().join("lmdos01.dat"))?,
        sample_lmdos_dat()
    );
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm01.dat"))?,
        sample_rhocm_dat()
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_requires_active_hubbard_magnetic_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard LDOS should require magnetic sidecars")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    assert!(!temp.path().join("lmdos00.dat").exists());
    assert!(!temp.path().join("rhocm00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_recovers_missing_active_hubbard_rhocm_without_fms() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_ldos_magnetic_table_close(
        &read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        &super::rhocm_from_lmdos_without_scattering(&sample_lmdos_dat())?,
        "active-Hubbard no-FMS rhocm repair",
    );
    Ok(())
}

#[test]
fn ldos_module_recovers_missing_active_hubbard_lmdos_without_fms() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_ldos_magnetic_table_close(
        &read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
        &super::lmdos_from_rhocm_without_scattering(&sample_rhocm_dat())?,
        "active-Hubbard no-FMS lmdos repair",
    );
    Ok(())
}

#[test]
fn ldos_module_does_not_advertise_active_hubbard_cache_without_rhoc_pair() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        super::rhoc_from_ldos_without_scattering(&sample_ldos_dat(), 0)?
    );
    Ok(())
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_ordinary_ldos_energy_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let mut ldos = sample_ldos_dat();
    ldos.energy_ev[1] += 0.25;
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard ordinary LDOS/RHOC pair should share an energy mesh")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_ordinary_rhoc_energy_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let mut rhoc = sample_rhoc_dat();
    rhoc.energy_ev[1] += 0.25;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard ordinary LDOS/RHOC pair should share an energy mesh")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_ordinary_density_layout() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let mut rhoc = sample_rhoc_dat();
    rhoc.density = array![
        [5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4, 9.0E-4, 10.0E-4],
        [5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4, 9.1E-4, 10.1E-4],
        [5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4, 9.2E-4, 10.2E-4]
    ];
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard ordinary LDOS/RHOC pair should share a density layout")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_active_hubbard_ordinary_ldos_pair() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
    let rhoc = sample_rhoc_dat();
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_active_hubbard_ordinary_rhoc_pair() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    let ldos = sample_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    Ok(())
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_magnetic_energy_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    let mut lmdos = sample_lmdos_dat();
    lmdos.energy_ev[2] += 0.25;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard magnetic sidecars should match LDOS energy mesh")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_rhocm_energy_grid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    let mut rhocm = sample_rhocm_dat();
    rhocm.energy_ev[0] -= 0.25;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard rhocm sidecar should match LDOS energy mesh")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_magnetic_layout_rejects_divergent_energy_grids() {
    let mut lmdos = sample_lmdos_dat();
    let mut rhocm = sample_rhocm_dat();
    lmdos.energy_ev[1] += 4.0e-5;
    rhocm.energy_ev[1] -= 4.0e-5;

    assert!(
        !super::ldos_magnetic_layouts_match(&lmdos, &rhocm),
        "magnetic sidecars that diverge from each other must not share one LDOS layout"
    );
}

#[test]
fn ldos_module_rejects_stale_active_hubbard_magnetic_layout() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    let mut rhocm = sample_rhocm_dat();
    rhocm.angular_limit = 0;
    rhocm.density = array![[9.0E-4, 8.0E-4], [9.1E-4, 8.1E-4], [9.2E-4, 8.2E-4]];
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
    assert_eq!(
        read_rhocm_dat(temp.path().join("rhocm00.dat"))?.angular_limit,
        0
    );

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard magnetic sidecars should share an lx layout")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_accepts_active_hubbard_cache_with_gtr_m_source_superset() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_hubbard_gtr_m_source_contract(2, 3, 1),
    )?;

    assert!(has_cached_ldos_output(temp.path())?);
    assert_eq!(run_in_dir(temp.path())?, 4);
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_cache_that_conflicts_with_gtr_off_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off00.bin"),
        &sample_hubbard_gtr_off_source_contract(2, 1, 2, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard magnetic cache should match valid gtr_off source layout")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_gtr_source_that_omits_cached_potential() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr01.bin"),
        &sample_hubbard_gtr_source_contract(1, 3, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard gtr source should cover cached potential index")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_gtr_m_source_that_omits_cached_potential() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m01.bin"),
        &sample_hubbard_gtr_m_source_contract(1, 3, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard gtr_m source should cover cached potential index")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_gtr_off_source_that_omits_cached_potential() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_active_hubbard_cached_tables(temp.path(), "01")?;
    write_hubbard_ldos_gtr_off_bin(
        temp.path().join("gtr_off01.bin"),
        &sample_hubbard_gtr_off_source_contract(2, 1, 3, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard gtr_off source should cover cached potential index")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_ordinary_cache_that_conflicts_with_gtr_source() -> Result<()>
{
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_hubbard_gtr_source_contract(2, 3, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard ordinary cache should match valid gtr source layout")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_rejects_active_hubbard_source_traces_with_conflicting_layouts() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    let mut lmdos = sample_lmdos_dat();
    lmdos.angular_limit = 0;
    lmdos.density = array![[1.0E-4, 2.0E-4], [1.1E-4, 2.1E-4], [1.2E-4, 2.2E-4]];
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
    let mut rhocm = sample_rhocm_dat();
    rhocm.angular_limit = 0;
    rhocm.density = array![[9.0E-4, 8.0E-4], [9.1E-4, 8.1E-4], [9.2E-4, 8.2E-4]];
    write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
    write_hubbard_ldos_gtr_bin(
        temp.path().join("gtr00.bin"),
        &sample_hubbard_gtr_source_contract(1, 3, 1),
    )?;
    write_hubbard_ldos_gtr_m_bin(
        temp.path().join("gtr_m00.bin"),
        &sample_hubbard_gtr_m_source_contract(0, 3, 1),
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active-Hubbard gtr and gtr_m source layouts should agree")?;
    let chain = format!("{error:?}");

    assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_roundtrips_hubbard_nio_reference_zip_magnetic_sidecars() -> Result<()> {
    let Some(zip_path) = reference_hubbard_nio_ldos_zip()? else {
        crate::require_fixture!(
            "LDOS NiO Hubbard magnetic reference test; REFERENCE.zip not found"
        );
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        crate::require_fixture!(
            "LDOS NiO Hubbard magnetic reference test; unzip command not found"
        );
    }

    let temp = tempfile::tempdir()?;
    for name in ["ldos.inp", "hubbard.inp"] {
        std::fs::write(
            temp.path().join(name),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
        )?;
    }

    let hubbard_text = std::fs::read_to_string(temp.path().join("hubbard.inp"))?;
    let hubbard = HubbardInput::parse_str(temp.path().join("hubbard.inp"), &hubbard_text)?;
    assert_eq!(
        hubbard.mldos_hubb, 2,
        "NiO reference should exercise active Hubbard magnetic LDOS"
    );

    let mut expected_ldos = Vec::new();
    let mut expected_rhoc = Vec::new();
    let mut expected_lmdos = Vec::new();
    let mut expected_rhocm = Vec::new();
    for potential in 0..=2 {
        for prefix in ["ldos", "rhoc", "lmdos", "rhocm"] {
            let name = format!("{prefix}{potential:02}.dat");
            std::fs::write(
                temp.path().join(&name),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
            )?;
        }
        expected_ldos.push(read_ldos_dat(
            temp.path().join(format!("ldos{potential:02}.dat")),
        )?);
        expected_rhoc.push(read_rhoc_dat(
            temp.path().join(format!("rhoc{potential:02}.dat")),
        )?);
        expected_lmdos.push(read_lmdos_dat(
            temp.path().join(format!("lmdos{potential:02}.dat")),
        )?);
        expected_rhocm.push(read_rhocm_dat(
            temp.path().join(format!("rhocm{potential:02}.dat")),
        )?);
    }

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 12);
    for potential in 0..=2 {
        assert_eq!(
            read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?,
            expected_ldos[potential]
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?,
            expected_rhoc[potential]
        );
        assert_eq!(
            read_lmdos_dat(temp.path().join(format!("lmdos{potential:02}.dat")))?,
            expected_lmdos[potential]
        );
        assert_eq!(
            read_rhocm_dat(temp.path().join(format!("rhocm{potential:02}.dat")))?,
            expected_rhocm[potential]
        );
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_rejects_malformed_active_hubbard_magnetic_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    std::fs::write(
        temp.path().join("lmdos00.dat"),
        "not a magnetic ldos table\n",
    )?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("malformed active-Hubbard magnetic sidecar should fail validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("lmdos00.dat"), "{chain}");

    let error = run_in_dir(temp.path())
        .err()
        .context("explicit LDOS run should reject malformed active-Hubbard magnetic sidecar")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("lmdos00.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_active_hubbard_rhocm_sidecar_without_fms() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_hubbard_input(temp.path(), 2)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
    std::fs::write(
        temp.path().join("rhocm00.dat"),
        "not a magnetic rhoc table\n",
    )?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_ldos_magnetic_table_close(
        &read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
        &super::rhocm_from_lmdos_without_scattering(&sample_lmdos_dat())?,
        "active-Hubbard no-FMS malformed rhocm repair",
    );
    Ok(())
}

#[test]
fn ldos_module_ignores_non_hubbard_magnetic_sidecar_files() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let ldos = sample_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(
        temp.path().join("lmdos00.dat"),
        "not a magnetic ldos table\n",
    )?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    assert!(read_rhoc_dat(temp.path().join("rhoc00.dat")).is_ok());
    Ok(())
}

#[test]
fn ldos_module_does_not_advertise_malformed_ldos_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("malformed LDOS cache should fail predicate validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("ldos00.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_generates_kmesh_handoff_when_malformed_ldos_cache_exists() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;

    assert!(!has_cached_ldos_output(temp.path())?);
    assert!(has_supported_kmesh_handoff(temp.path())?);
    let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

    assert_eq!(count, 1);
    let kmesh = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(kmesh.rows.len(), 8);
    assert!(!temp.path().join("logdos.dat").exists());

    let error = run_in_dir(temp.path())
        .err()
        .context("stale final LDOS cache should fall through to the source requirement")?;
    assert!(
        error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn ldos_module_does_not_advertise_malformed_rhoc_sidecar_with_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 1)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("malformed RHOC sidecar should fail predicate validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("rhoc00.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_does_not_advertise_malformed_kmesh_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("malformed kmesh.dat should fail predicate validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("kmesh.dat"), "{chain}");

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed kmesh.dat should fail through the explicit LDOS runner")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("kmesh.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_kmesh_from_reciprocal_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;
    std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(data.rows.len(), 8);
    assert_eq!(
        data.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert_kmesh_row_close(
        data.rows[0],
        KmeshRow {
            index: 1,
            k_point: [0.831_446_454_055_273_6; 3],
            weight: 0.5,
            metadata: data.rows[0].metadata,
        },
        5.0e-4,
    );
    Ok(())
}

#[test]
fn ldos_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
    std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("malformed logdos.dat should fail predicate validation")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("logdos.dat"), "{chain}");

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed logdos.dat should fail through the explicit LDOS runner")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("logdos.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_keeps_malformed_log_strict_when_kmesh_is_already_valid() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
    std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    let error = has_cached_ldos_output(temp.path())
        .err()
        .context("valid existing kmesh.dat should not mask a malformed LDOS log")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("logdos.dat"), "{chain}");

    let error = run_in_dir(temp.path())
        .err()
        .context("explicit LDOS run should keep the malformed log strict")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("logdos.dat"), "{chain}");
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_module_log_for_recoverable_ldos_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let rhoc = sample_rhoc_dat();
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
    std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_module_log_for_recoverable_kmesh_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;
    std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;
    std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
    Ok(())
}

#[test]
fn ldos_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    let ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_generates_missing_kmesh_from_reciprocal_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input(temp.path(), true)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
    let reciprocal = sample_reciprocal_input(8);
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        reciprocal_input_string(&reciprocal)?,
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
    assert_eq!(data.rows.len(), 8);
    assert_eq!(
        data.rows[0].metadata,
        Some(KmeshMetadata {
            requested_points: 8,
            irreducible_points: 8,
            divisions: [2, 2, 2],
        })
    );
    assert_kmesh_row_close(
        data.rows[0],
        KmeshRow {
            index: 1,
            k_point: [0.831_446_454_055_273_6; 3],
            weight: 0.5,
            metadata: data.rows[0].metadata,
        },
        5.0e-4,
    );
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert_eq!(rhoc.energy_ev, sample_ldos_dat().energy_ev);
    assert_eq!(rhoc.density, sample_ldos_dat().density);
    Ok(())
}

#[test]
fn ldos_module_generates_missing_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let ldos = sample_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_generates_missing_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let rhoc = sample_rhoc_dat();
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert!(ldos.header_lines.iter().any(|line| line.contains("sDOS")));
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    assert_eq!(
        read_module_log_dat(temp.path().join("logdos.dat"))?,
        sample_module_log()
    );
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let rhoc = sample_rhoc_dat();
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
    std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert!(ldos.header_lines.iter().any(|line| line.contains("sDOS")));
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let ldos = sample_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    Ok(())
}

#[test]
fn ldos_module_generates_missing_ldos_for_partial_rhoc_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 0)?;
    let cached_ldos = sample_ldos_dat();
    let rhoc = sample_rhoc_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &cached_ldos)?;
    write_rhoc_dat(temp.path().join("rhoc01.dat"), &rhoc)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, cached_ldos);
    let generated_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert_eq!(generated_rhoc.energy_ev, cached_ldos.energy_ev);
    assert_eq!(generated_rhoc.density, cached_ldos.density);
    let generated = read_ldos_dat(temp.path().join("ldos01.dat"))?;
    assert_eq!(generated.energy_ev, rhoc.energy_ev);
    assert_eq!(generated.density, rhoc.density);
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc01.dat"))?, rhoc);
    Ok(())
}

#[test]
fn ldos_module_generates_missing_spin_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
    let rhoc = sample_spin_rhoc_dat();
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    assert!(ldos.is_spin_resolved());
    assert_eq!(ldos.energy_ev, rhoc.energy_ev);
    assert_eq!(ldos.density, rhoc.density);
    assert!(
        ldos.header_lines
            .iter()
            .any(|line| line.contains("sDOS(up)") && line.contains("sDOS(down)"))
    );
    assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
    Ok(())
}

#[test]
fn ldos_module_generates_missing_spin_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
    let ldos = sample_spin_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert!(rhoc.is_spin_resolved());
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    Ok(())
}

#[test]
fn ldos_module_recovers_malformed_spin_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
    let ldos = sample_spin_ldos_dat();
    write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
    std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

    assert!(has_cached_ldos_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
    let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
    assert!(rhoc.header_lines.is_empty());
    assert_eq!(rhoc.fermi_level_ev, None);
    assert!(rhoc.is_spin_resolved());
    assert_eq!(rhoc.energy_ev, ldos.energy_ev);
    assert_eq!(rhoc.density, ldos.density);
    Ok(())
}

#[test]
fn ldos_module_rejects_rhoc_spin_shape_that_disagrees_with_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 0)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_spin_rhoc_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("spin-shaped rhoc should not satisfy non-spin LDOS input")?;

    assert!(error.chain().any(|cause| {
        cause
            .to_string()
            .contains("LDOS rhoc handoff spin shape does not match ldos.inp ispin=0")
    }));
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_rejects_ldos_spin_shape_for_non_spin_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 0)?;
    write_ldos_dat(temp.path().join("ldos00.dat"), &sample_spin_ldos_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("spin-shaped ldos should not satisfy non-spin LDOS input")?;

    assert!(error.chain().any(|cause| {
        cause
            .to_string()
            .contains("LDOS ldos handoff spin shape does not match ldos.inp ispin=0")
    }));
    assert!(!temp.path().join("rhoc00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_rejects_rhoc_only_when_fms_scattering_is_enabled() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ldos_input_with_lfms2(temp.path(), true, 1)?;
    write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;

    assert!(!has_cached_ldos_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("FMS LDOS should still require complete scattering source state")?;

    assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
    assert!(!temp.path().join("ldos00.dat").exists());
    Ok(())
}

#[test]
fn ldos_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_ldos_dir()? else {
        crate::require_fixture!("LDOS reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "ldos.inp",
        "ldos00.dat",
        "ldos01.dat",
        "rhoc00.dat",
        "rhoc01.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
    let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_eq!(
        read_ldos_dat(temp.path().join("ldos00.dat"))?,
        expected_ldos
    );
    assert_eq!(
        read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
        expected_rhoc
    );
    Ok(())
}

#[test]
fn ldos_module_generates_no_fms_reference_tables_from_source_handoffs() -> Result<()> {
    let cases = reference_ldos_source_cases()?;
    if cases.is_empty() {
        crate::require_fixture!("LDOS source reference test; no generated reference cases found");
    }

    for case in cases {
        let temp = tempfile::tempdir()?;
        std::fs::copy(
            case.expected_dir.join("ldos.inp"),
            temp.path().join("ldos.inp"),
        )
        .with_context(|| format!("failed to copy LDOS input for {}", case.label))?;
        for name in [
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
        ] {
            std::fs::copy(case.source_dir.join(name), temp.path().join(name))
                .with_context(|| format!("failed to copy {name} for {}", case.label))?;
        }
        if case.source_dir.join("geom.dat").is_file() {
            std::fs::copy(
                case.source_dir.join("geom.dat"),
                temp.path().join("geom.dat"),
            )
            .with_context(|| format!("failed to copy geom.dat for {}", case.label))?;
        }

        let count = run_in_dir(temp.path())
            .with_context(|| format!("failed to generate LDOS source tables for {}", case.label))?;

        assert_eq!(count, case.potential_count * 2, "{}", case.label);
        if temp.path().join("gtr00.bin").is_file() {
            for potential in 0..case.potential_count {
                let gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
                assert_eq!(gtr.fms_mode, 2, "{}", case.label);
            }
        }
        for potential in 0..case.potential_count {
            let generated_ldos =
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let generated_rhoc =
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            let reference_ldos =
                read_ldos_dat(case.expected_dir.join(format!("ldos{potential:02}.dat")))?;
            let reference_rhoc =
                read_rhoc_dat(case.expected_dir.join(format!("rhoc{potential:02}.dat")))?;

            assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
            assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
            assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, case.label);
            assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, case.label);
            assert!(generated_ldos.density.iter().all(|value| value.is_finite()));
            assert!(generated_rhoc.density.iter().all(|value| value.is_finite()));
            assert!(generated_ldos.density.iter().any(|value| value.abs() > 0.0));
            assert_eq!(generated_ldos.energy_ev, generated_rhoc.energy_ev);
            assert_eq!(generated_ldos.density, generated_rhoc.density);
        }
    }
    Ok(())
}

#[test]
fn ldos_module_generates_zero_cluster_fms_reference_trace_from_source_handoffs() -> Result<()> {
    let Some(reference_dir) = reference_ldos_dir()? else {
        crate::require_fixture!(
            "LDOS FMS source reference test; generated EXAFS/Cu reference not found"
        );
    };
    if !reference_dir.join("gtr00.bin").is_file()
        || !reference_ldos_source_present(&reference_dir)
        || !reference_dir.join("geom.dat").is_file()
    {
        crate::require_fixture!("LDOS FMS source reference test; EXAFS/Cu FMS handoffs not found");
    }

    let temp = tempfile::tempdir()?;
    let input_text = std::fs::read_to_string(reference_dir.join("ldos.inp"))?;
    let mut input = refeff_io::LdosInput::parse_str(reference_dir.join("ldos.inp"), &input_text)?;
    input.control.lfms2 = 1;
    std::fs::write(temp.path().join("ldos.inp"), ldos_input_string(&input)?)?;
    for name in [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
        "geom.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy {name} for LDOS FMS source reference"))?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_zero_gtr_reference_compatible(
        &read_gtr_bin(temp.path().join("gtr00.bin"))?,
        &read_gtr_bin(reference_dir.join("gtr00.bin"))?,
    );
    if reference_dir.join("gtr01.bin").is_file() {
        assert_zero_gtr_reference_compatible(
            &read_gtr_bin(temp.path().join("gtr01.bin"))?,
            &read_gtr_bin(reference_dir.join("gtr01.bin"))?,
        );
    }
    for potential in 0..=1 {
        let generated_ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let generated_rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        let reference_ldos = read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
        let reference_rhoc = read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
        assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
        assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
        assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, "EXAFS/Cu zero-FMS");
        assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, "EXAFS/Cu zero-FMS");
    }
    Ok(())
}

#[test]
fn ldos_module_matches_nonzero_fms_reference_from_source_handoffs() -> Result<()> {
    let Some(reference_dir) = reference_ldos_nonzero_fms_dir()? else {
        crate::require_fixture!(
            "LDOS nonzero FMS source reference test; generated XANES/Cu FMS reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "ldos.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
        "geom.dat",
        ".dimensions.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
            format!("failed to copy {name} for LDOS nonzero FMS source reference")
        })?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
        let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
        assert_gtr_reference_close(
            &generated_gtr,
            &reference_gtr,
            &format!("XANES/Cu FMS gtr{potential:02}.bin"),
        );

        let generated_ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let generated_rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        let reference_ldos = read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
        let reference_rhoc = read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
        assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
        assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
        assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, "XANES/Cu FMS");
        assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, "XANES/Cu FMS");
    }
    Ok(())
}

#[test]
fn ldos_module_matches_ordinary_spin_fms_reference_from_source_handoffs() -> Result<()> {
    let Some(reference_dir) = reference_ldos_ordinary_spin_fms_dir()? else {
        crate::require_fixture!(
            "LDOS ordinary-spin FMS source reference test; generated XANES/Cu ordinary-spin FMS reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "ldos.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
        "geom.dat",
        "xsph.inp",
        ".dimensions.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
            format!("failed to copy {name} for LDOS ordinary-spin FMS source reference")
        })?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
        let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
        assert_gtr_reference_close(
            &generated_gtr,
            &reference_gtr,
            &format!("XANES/Cu ordinary-spin FMS gtr{potential:02}.bin"),
        );

        let generated_ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let generated_rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        let reference_ldos = read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
        let reference_rhoc = read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
        assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
        assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
        assert_ldos_density_grid_close(
            &generated_ldos,
            &reference_ldos,
            "XANES/Cu ordinary-spin FMS",
        );
        assert_ldos_density_grid_close(
            &generated_rhoc,
            &reference_rhoc,
            "XANES/Cu ordinary-spin FMS",
        );
    }
    Ok(())
}

#[test]
fn ldos_module_matches_production_fms_reference_from_source_handoffs() -> Result<()> {
    let Some(reference_dir) = reference_ldos_production_fms_dir()? else {
        crate::require_fixture!(
            "LDOS production FMS source reference test; generated XANES/Cu production FMS reference not found"
        );
    };

    let temp = tempfile::tempdir()?;
    for name in [
        "ldos.inp",
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
        "geom.dat",
        ".dimensions.dat",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
            format!("failed to copy {name} for LDOS production FMS source reference")
        })?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    for potential in 0..=1 {
        let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
        let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
        assert_gtr_reference_close(
            &generated_gtr,
            &reference_gtr,
            &format!("XANES/Cu production FMS gtr{potential:02}.bin"),
        );

        let generated_ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
        let generated_rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
        let reference_ldos = read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
        let reference_rhoc = read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
        assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
        assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
        assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, "XANES/Cu production FMS");
        assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, "XANES/Cu production FMS");
    }
    Ok(())
}

fn sample_reciprocal_input(total_kpoints: i32) -> ReciprocalInput {
    ReciprocalInput {
        ispace: 0,
        cell: Some(ReciprocalCell {
            lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            volume_scale: -1.0,
            imaginary_energy: 0.0,
            core_hole_strength: 1.0,
            lattice_name: "P".to_string(),
            space_group_hm: "Pm-3m".to_string(),
            space_group: 221,
            atom_count: 1,
            absorber: 1,
            core_hole: 1,
            k_mesh: ReciprocalKMesh {
                total: total_kpoints,
                x: total_kpoints,
                y: 0,
                z: 0,
                kind: 3,
                use_symmetry: false,
            },
            positions: vec![[0.0, 0.0, 0.0]],
            potentials: vec![0],
            labels: vec!["Cu".to_string()],
            stretch: [0.0, 0.0, 0.0],
        }),
    }
}

fn assert_kmesh_row_close(actual: KmeshRow, expected: KmeshRow, tolerance: f64) {
    assert_eq!(actual.index, expected.index);
    assert_eq!(actual.metadata, expected.metadata);
    assert!((actual.weight - expected.weight).abs() <= tolerance);
    for (actual, expected) in actual.k_point.iter().zip(expected.k_point.iter()) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }
}

fn write_ldos_input(work_dir: &Path, enabled: bool) -> Result<()> {
    write_ldos_input_with_lfms2(work_dir, enabled, 0)
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

fn write_active_hubbard_cached_tables(work_dir: &Path, index: &str) -> Result<()> {
    write_ldos_dat(
        work_dir.join(format!("ldos{index}.dat")),
        &sample_ldos_dat(),
    )?;
    write_rhoc_dat(
        work_dir.join(format!("rhoc{index}.dat")),
        &sample_rhoc_dat(),
    )?;
    write_lmdos_dat(
        work_dir.join(format!("lmdos{index}.dat")),
        &sample_lmdos_dat(),
    )?;
    write_rhocm_dat(
        work_dir.join(format!("rhocm{index}.dat")),
        &sample_rhocm_dat(),
    )?;
    Ok(())
}

fn write_ldos_input_with_lfms2(work_dir: &Path, enabled: bool, lfms2: i32) -> Result<()> {
    write_ldos_input_with_lfms2_and_ispin(work_dir, enabled, lfms2, 0)
}

fn write_ldos_input_with_lfms2_and_ispin(
    work_dir: &Path,
    enabled: bool,
    lfms2: i32,
    ispin: i32,
) -> Result<()> {
    std::fs::write(
        work_dir.join("ldos.inp"),
        format!(
            concat!(
                "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                "{:4}{:4}{:4}{:4}{:4} {:7} {:4}\n",
                "rfms2, emin, emax, eimag, rgrd\n",
                "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}\n",
                "rdirec, toler1, toler2\n",
                "{:13.5}{:13.5}{:13.5}\n",
                " lmaxph(0:nph)\n",
                "{:4}{:4}\n",
                "ldostype\n",
                "{:4}\n"
            ),
            i32::from(enabled),
            lfms2,
            0,
            ispin,
            0,
            3,
            11,
            -1.0,
            -1.0,
            1.0,
            0.1,
            0.05,
            -1.0,
            0.001,
            0.001,
            3,
            3,
            0
        ),
    )?;
    Ok(())
}

fn write_ldos_source_input(work_dir: &Path) -> Result<()> {
    write_ldos_source_input_with_neldos(work_dir, 2)
}

fn write_ldos_source_input_with_neldos(work_dir: &Path, neldos: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("ldos.inp"),
        format!(
            concat!(
                "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                "   1   1   0   0   0{:8}   11\n",
                "rfms2, emin, emax, eimag, rgrd\n",
                "      3.00000     -1.00000      1.00000      0.10000      0.05000\n",
                "rdirec, toler1, toler2\n",
                "      5.00000      0.00100      0.00100\n",
                " lmaxph(0:nph)\n",
                "   1   1\n",
                "ldostype\n",
                "   0\n",
            ),
            neldos
        ),
    )?;
    Ok(())
}

fn write_ldos_fms_source_handoffs(work_dir: &Path) -> Result<()> {
    let fms = FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: -1,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: -1.0,
            rdirec: -1.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 0.0,
            thetad: 0.0,
            sig2g: 0.0,
        },
        lmaxph: vec![1, 1],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    };
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&fms)?)?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_global_input())?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_ldos_source_geom())?,
    )?;
    write_phase_bin(work_dir.join("phase.bin"), &sample_ldos_source_phase_bin())?;
    Ok(())
}

fn write_ldos_wavefunction_source_input(work_dir: &Path) -> Result<()> {
    write_ldos_wavefunction_source_input_with_lfms2(work_dir, 1)
}

fn write_ldos_wavefunction_source_input_with_lfms2(work_dir: &Path, lfms2: i32) -> Result<()> {
    write_ldos_wavefunction_source_input_with_neldos(work_dir, lfms2, 2)
}

fn write_ldos_wavefunction_source_input_with_neldos(
    work_dir: &Path,
    lfms2: i32,
    neldos: i32,
) -> Result<()> {
    write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(work_dir, lfms2, &[3, 3], neldos)
}

fn write_ldos_wavefunction_source_input_with_lfms2_and_lmax(
    work_dir: &Path,
    lfms2: i32,
    lmaxph: &[i32],
) -> Result<()> {
    write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(work_dir, lfms2, lmaxph, 2)
}

fn write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(
    work_dir: &Path,
    lfms2: i32,
    lmaxph: &[i32],
    neldos: i32,
) -> Result<()> {
    let lmaxph = lmaxph
        .iter()
        .map(|value| format!("{value:4}"))
        .collect::<String>();
    std::fs::write(
        work_dir.join("ldos.inp"),
        format!(
            concat!(
                "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                "   1{:4}   0   0   0{:8}   11\n",
                "rfms2, emin, emax, eimag, rgrd\n",
                "      3.00000     -1.00000      1.00000      0.10000      0.05000\n",
                "rdirec, toler1, toler2\n",
                "      5.00000      0.00100      0.00100\n",
                " lmaxph(0:nph)\n",
                "{}\n",
                "ldostype\n",
                "   0\n",
            ),
            lfms2, neldos, lmaxph
        ),
    )?;
    Ok(())
}

fn write_ldos_wavefunction_source_handoffs(work_dir: &Path) -> Result<()> {
    write_pot_bin(
        work_dir.join("pot.bin"),
        &sample_ldos_wavefunction_pot_bin(),
    )?;
    std::fs::write(
        work_dir.join("config.dat"),
        config_dat_string(&sample_ldos_wavefunction_config_dat())?,
    )?;
    write_phase_bin(
        work_dir.join("phase.bin"),
        &sample_ldos_wavefunction_phase_bin(),
    )?;
    std::fs::write(
        work_dir.join("pot.inp"),
        pot_input_string(&sample_ldos_wavefunction_pot_input())?,
    )?;
    std::fs::write(
        work_dir.join("fms.inp"),
        fms_input_string(&sample_ldos_wavefunction_fms_input())?,
    )?;
    Ok(())
}

fn expand_ldos_wavefunction_source_valence_occupancy(work_dir: &Path) -> Result<()> {
    let path = work_dir.join("pot.bin");
    let mut pot = read_pot_bin(&path)?;
    pot.valence_occupancy =
        Array2::from_shape_fn((4, pot.potential_count()), |(angular, potential)| {
            angular as f64 + 0.1 * potential as f64
        });
    write_pot_bin(path, &pot)?;
    Ok(())
}

fn write_ldos_wavefunction_fms_geometry_handoffs(work_dir: &Path) -> Result<()> {
    let mut fms = sample_ldos_wavefunction_fms_input();
    fms.control.idwopt = -1;
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&fms)?)?;
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_global_input())?,
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_ldos_source_geom())?,
    )?;
    Ok(())
}

fn sample_ldos_wavefunction_pot_bin() -> PotBinData {
    let potentials = 2;
    PotBinData {
        titles: vec!["LDOS wavefunction source test".to_string()],
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: 0,
        ihole: 1,
        interstitial_selector: 0,
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
        muffin_tin_indices: Array1::from_vec(vec![12, 13]),
        muffin_tin_radii: Array1::from_vec(vec![1.1, 1.2]),
        norman_indices: Array1::from_vec(vec![20, 21]),
        atomic_numbers: Array1::from_vec(vec![29, 8]),
        kappa: Array1::from_iter(-20..=20),
        norman_radii: Array1::from_vec(vec![2.1, 2.2]),
        overlap_factors: Array1::from_vec(vec![0.9, 0.8]),
        max_overlap_factors: Array1::from_vec(vec![1.3, 1.4]),
        potential_multiplicities: Array1::from_vec(vec![1.0, 1.0]),
        ionization: Array1::from_vec(vec![0.0, 1.0]),
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
            |(coefficient, orbital, potential)| {
                0.01 * (coefficient + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coefficient, orbital, potential)| {
                -0.01 * (coefficient + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        electron_density: ldos_wavefunction_radial_matrix(potentials, 0.01),
        coulomb_potential: ldos_wavefunction_radial_matrix(potentials, -0.02),
        total_potential: ldos_wavefunction_radial_matrix(potentials, -0.03),
        valence_density: ldos_wavefunction_radial_matrix(potentials, 0.004),
        valence_potential: ldos_wavefunction_radial_matrix(potentials, -0.005),
        magnetization_density: ldos_wavefunction_radial_matrix(potentials, 0.0002),
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
        norman_charges: Array1::from_vec(vec![28.5, 7.5]),
        valence_occupancy: Array2::from_shape_fn((1, potentials), |(_, potential)| {
            potential as f64
        }),
        raw_text: None,
    }
}

fn ldos_wavefunction_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

fn sample_ldos_wavefunction_config_dat() -> ConfigDatData {
    let mut first_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut first_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    first_occupations[0] = 1.0;
    first_occupations[1] = 2.0;
    first_valence[1] = 0.5;

    let mut second_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    let mut second_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
    second_occupations[0] = 2.0;
    second_occupations[1] = 2.0;
    second_occupations[2] = 1.0;
    second_valence[2] = 1.0;

    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![
            ConfigDatPotential {
                potential_index: 0,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: first_occupations,
                valence_occupations: first_valence,
                spin_occupations: None,
            },
            ConfigDatPotential {
                potential_index: 1,
                atomic_number: 8,
                element: "O".to_string(),
                occupations: second_occupations,
                valence_occupations: second_valence,
                spin_occupations: None,
            },
        ],
    }
}

fn sample_ldos_wavefunction_phase_bin() -> PhaseBinData {
    let spin_count = 1;
    let energy_grid = Array1::from_vec(vec![
        Complex64::new(-1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
        Complex64::new(1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
    ]);
    let energy_count = energy_grid.len();
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: energy_count,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: 1,
        transition_count: 1,
        q_count: 1,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: 0.045,
            edge_energy: 9.8,
        },
        energy_grid,
        reference_energy: Array2::zeros((energy_count, spin_count)),
        potentials: vec![
            sample_ldos_wavefunction_phase_potential(29, "Cu", energy_count, spin_count),
            sample_ldos_wavefunction_phase_potential(8, "O", energy_count, spin_count),
        ],
        transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
        raw_pads: None,
    }
}

fn sample_ldos_wavefunction_phase_potential(
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
) -> PhaseBinPotential {
    PhaseBinPotential {
        lmax: 3,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::zeros((energy_count, 7, spin_count)),
    }
}

fn sample_ldos_wavefunction_pot_input() -> PotInput {
    PotInput {
        control: PotControl {
            mpot: 1,
            nph: 1,
            ntitle: 1,
            ihole: 1,
            ipr1: 0,
            iafolp: 0,
            ixc: 0,
            ispec: 0,
            iscfxc: 0,
        },
        run: PotRun {
            nmix: 0,
            nohole: 0,
            jumprm: 0,
            inters: 0,
            nscmt: 0,
            icoul: 0,
            lfms1: 0,
            iunf: 0,
        },
        titles: vec!["LDOS wavefunction source test".to_string()],
        scattering: PotScattering {
            gamach: 0.0,
            rgrd: 0.05,
            ca1: 0.0,
            ecv: 0.0,
            totvol: 1.0,
            rfms1: 0.0,
            corval_emin: 0.0,
        },
        potentials: vec![
            PotPotential {
                z: 29,
                lmaxsc: 3,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
            PotPotential {
                z: 8,
                lmaxsc: 3,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
        ],
        external_pot: false,
        start_from_file: false,
        overlap_shells: vec![Vec::<PotOverlapShell>::new(), Vec::<PotOverlapShell>::new()],
        chsh_type: 0,
        config_type: 1,
        thermal: PotThermal {
            scf_temperature: 0.0,
            scf_thermal_vxc: 0,
            iscfth: 0,
            xntol: 0.0,
            nmu: 0,
            negrid: 0,
            emaxscf: 0.0,
        },
        finite_nucleus: false,
        warn_ion: false,
        ramp: PotRamp {
            ramp_scf: false,
            rfms_start: 0.0,
            nramp: 0,
        },
        tolerances: PotTolerances {
            tolmu: 0.0,
            tolq: 0.0,
            tolqp: 0.0,
        },
    }
}

fn sample_ldos_wavefunction_fms_input() -> FmsInput {
    FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: 0,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 3.0,
            rdirec: 0.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 0.0,
            thetad: 0.0,
            sig2g: 0.0,
        },
        lmaxph: vec![3, 3],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    }
}

fn sample_ldos_gtr_bin() -> GtrBinData {
    GtrBinData {
        point_count_declared: 2,
        horizontal_count: 2,
        danes_extension_count: 0,
        highest_potential_index: 1,
        fms_mode: 2,
        values: Array3::from_shape_fn((2, 2, 4), |(energy, potential, angular)| {
            Complex64::new(
                0.08 + 0.02 * energy as f64 + 0.03 * potential as f64 + 0.01 * angular as f64,
                -0.04 + 0.005 * angular as f64,
            )
        }),
    }
}

fn sample_global_input() -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 0.0,
        },
        control: GlobalControl {
            ipol: 0,
            ispin: 0,
            le2: 0,
            elpty: 0.0,
            angks: 0.0,
            l2lp: 0,
            do_nrixs: 0,
            ldecmx: 0,
            lj: 0,
        },
        evec: [0.0, 0.0, 1.0],
        xivec: [1.0, 0.0, 0.0],
        spvec: [0.0, 0.0, 1.0],
        polarization_tensor: [
            [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0],
        ],
        norms: GlobalNorms {
            evnorm: 1.0,
            xivnorm: 1.0,
            spvnorm: 1.0,
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

fn sample_ldos_source_geom() -> GeomDat {
    GeomDat {
        nat: 2,
        nph: 1,
        model_atoms: vec![1, 2],
        atoms: vec![
            GeomDatRow {
                index: 1,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                iph: 0,
                boundary: 0,
            },
            GeomDatRow {
                index: 2,
                x: 1.4,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 0,
            },
        ],
    }
}

fn sample_ldos_source_phase_bin() -> PhaseBinData {
    let energy_count = 2;
    let spin_count = 1;
    let transition_count = 8;
    let energy_grid = Array1::from_vec(vec![
        Complex64::new(-1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
        Complex64::new(1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
    ]);
    let reference_energy = Array2::zeros((energy_count, spin_count));
    let potentials = (0..2)
        .map(|potential| PhaseBinPotential {
            lmax: 1,
            atomic_number: 29,
            label: format!("Cu{potential}"),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, 3, spin_count),
                |(energy, signed_l, _)| {
                    let scale = 0.03 * (potential + 1) as f64
                        + 0.01 * energy as f64
                        + 0.02 * signed_l as f64;
                    Complex64::new(scale, 0.004 * (signed_l + 1) as f64)
                },
            ),
        })
        .collect();
    let transition_moments = Array4::<Complex64>::from_shape_fn(
        (energy_count, 1, transition_count, spin_count).f(),
        |(energy, _, transition, _)| {
            Complex64::new(
                0.2 + 0.03 * energy as f64 + 0.015 * transition as f64,
                -0.01 * transition as f64,
            )
        },
    );

    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: energy_count,
        auxiliary_energy_count: 0,
        ihole: 1,
        fermi_index: 1,
        pad_width: 8,
        final_state_count: transition_count,
        transition_count,
        q_count: 1,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: 0.0,
            edge_energy: 8_979.0,
        },
        energy_grid,
        reference_energy,
        potentials,
        transition_moments,
        raw_pads: None,
    }
}

fn sample_ldos_dat() -> LdosDatData {
    LdosDatData {
        header_lines: vec![
            "#  Fermi level (eV):  -3.777".to_string(),
            "#      e        sDOS           pDOS          dDOS          fDOS".to_string(),
        ],
        fermi_level_ev: Some(-3.777),
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: array![-1.0, 0.0, 1.0],
        density: array![
            [1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4],
            [1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4],
            [1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4]
        ],
    }
}

fn sample_rhoc_dat() -> LdosDatData {
    LdosDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: array![-1.0, 0.0, 1.0],
        density: array![
            [5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4],
            [5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4],
            [5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4]
        ],
    }
}

fn sample_lmdos_dat() -> LdosMagneticDatData {
    LdosMagneticDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e   s(+0)DOS-up   p(-1)DOS-up   p(+0)DOS-up   p(+1)DOS-up   s(+0)DOS-dn   p(-1)DOS-dn   p(+0)DOS-dn   p(+1)DOS-dn".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            angular_limit: 1,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
                ],
                [
                    1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
                ],
                [
                    1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
                ]
            ],
        }
}

fn sample_rhocm_dat() -> LdosMagneticDatData {
    LdosMagneticDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        angular_limit: 1,
        energy_ev: array![-1.0, 0.0, 1.0],
        density: array![
            [
                9.0E-4, 8.0E-4, 7.0E-4, 6.0E-4, 5.0E-4, 4.0E-4, 3.0E-4, 2.0E-4
            ],
            [
                9.1E-4, 8.1E-4, 7.1E-4, 6.1E-4, 5.1E-4, 4.1E-4, 3.1E-4, 2.1E-4
            ],
            [
                9.2E-4, 8.2E-4, 7.2E-4, 6.2E-4, 5.2E-4, 4.2E-4, 3.2E-4, 2.2E-4
            ]
        ],
    }
}

fn sample_hubbard_gtr_m_source_contract(
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrMBinData {
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    HubbardLdosGtrMBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        angular_limit,
        values: Array5::from_elem(
            (
                2,
                energy_count,
                potential_count,
                angular_count,
                magnetic_count,
            ),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_hubbard_gtr_source_contract(
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrBinData {
    let angular_count = angular_limit + 1;
    HubbardLdosGtrBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        angular_limit,
        values: Array4::from_elem(
            (2, energy_count, potential_count, angular_count),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_hubbard_gtr_off_source_contract(
    hubbard_l: usize,
    angular_limit: usize,
    energy_count: usize,
    potential_count: usize,
) -> HubbardLdosGtrOffBinData {
    let angular_count = angular_limit + 1;
    let order = (hubbard_l + 1) * (hubbard_l + 1);
    HubbardLdosGtrOffBinData {
        point_count_declared: energy_count,
        horizontal_count: energy_count,
        danes_extension_count: 0,
        highest_potential_index: potential_count.saturating_sub(1),
        fms_mode: 2,
        hubbard_l,
        angular_limit,
        values: Array6::from_elem(
            (
                angular_count,
                2,
                energy_count,
                potential_count,
                order,
                order,
            ),
            Complex32::new(0.0, 0.0),
        ),
    }
}

fn sample_spin_rhoc_dat() -> LdosDatData {
    LdosDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: array![-1.0, 0.0, 1.0],
        density: array![
            [
                1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
            ],
            [
                1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
            ],
            [
                1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
            ]
        ],
    }
}

fn sample_spin_ldos_dat() -> LdosDatData {
    LdosDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e        sDOS(up)   pDOS(up)      dDOS(up)    fDOS(up)   sDOS(down)    pDOS(down)   dDOS(down)   fDOS(down)    @#".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
                ],
                [
                    1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
                ],
                [
                    1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
                ]
            ],
        }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            " Calculating LDOS ...".to_string(),
            "FEFF-serial using 1 thread.".to_string(),
            " Done with LDOS.".to_string(),
            "Done with module: LDOS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 4],
    }
}

struct LdosSourceReferenceCase {
    label: &'static str,
    source_dir: PathBuf,
    expected_dir: PathBuf,
    potential_count: usize,
}

fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("failed to find workspace root")
}

fn reference_hubbard_full_potential_source_dir() -> Result<Option<PathBuf>> {
    let parent = workspace_root()?.join("reference-work/tmp");
    if !parent.is_dir() {
        return Ok(None);
    }
    let required = [
        ".dimensions.dat",
        "config.dat",
        "fms.inp",
        "geom.dat",
        "global.inp",
        "hubbard.inp",
        "ldos.inp",
        "phase.bin",
        "pot.bin",
        "pot.inp",
        "xsph.inp",
    ];
    for entry in std::fs::read_dir(&parent)
        .with_context(|| format!("failed to read {}", parent.display()))?
    {
        let entry = entry?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("feff-ldos-spin-hubbard-full-potential."))
            && entry.file_type()?.is_dir()
            && required
                .iter()
                .all(|required_name| entry.path().join(required_name).is_file())
        {
            return Ok(Some(entry.path()));
        }
    }
    Ok(None)
}

fn reduce_hubbard_full_potential_energy_grid(work_dir: &Path) -> Result<()> {
    const TEST_ENERGY_COUNT: i32 = 4;

    let ldos_path = work_dir.join("ldos.inp");
    let ldos_text = std::fs::read_to_string(&ldos_path)
        .with_context(|| format!("failed to read {}", ldos_path.display()))?;
    let mut ldos = refeff_io::LdosInput::parse_str(&ldos_path, &ldos_text)
        .with_context(|| format!("failed to parse {}", ldos_path.display()))?;
    // Preserve the complete production geometry and FMS source contract,
    // but sample enough energies to exercise both Hubbard passes without
    // making a debug unit test repeat the 51-atom solve 101 times.
    // The stock NiO and CeO2 cards use FEFF's independent-center LDOS
    // mode, which must still execute both Hubbard magnetic passes.
    ldos.control.lfms2 = 0;
    ldos.control.neldos = TEST_ENERGY_COUNT;
    std::fs::write(&ldos_path, ldos_input_string(&ldos)?)
        .with_context(|| format!("failed to write {}", ldos_path.display()))?;

    // Preserve one representative of every stock NiO potential while
    // keeping the debug regression small enough to run routinely.
    std::fs::write(
        work_dir.join("geom.dat"),
        concat!(
            "nat, nph =     3    2\n",
            "       1       2       3\n",
            " iat     x       y        z       iph  \n",
            " -----------------------------------------------------------------------\n",
            "   1      0.00000      0.00000      0.00000   0   1\n",
            "   2     -2.00083      0.00000      0.00000   1   1\n",
            "   3     -2.00083      2.00083      0.00000   2   1\n",
        ),
    )
    .context("failed to write reduced Hubbard representative geometry")
}

fn reference_ldos_dir() -> Result<Option<PathBuf>> {
    let workspace = workspace_root()?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    Ok(reference_ldos_expected_present(&path, 2).then_some(path))
}

fn reference_ldos_nonzero_fms_dir() -> Result<Option<PathBuf>> {
    let workspace = workspace_root()?;
    let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_fms_short");
    Ok((reference_ldos_expected_present(&path, 2)
        && reference_ldos_source_grid_present(&path)
        && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
    .then_some(path))
}

fn reference_ldos_production_fms_dir() -> Result<Option<PathBuf>> {
    let workspace = workspace_root()?;
    let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_fms");
    Ok((reference_ldos_expected_present(&path, 2)
        && reference_ldos_source_grid_present(&path)
        && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
    .then_some(path))
}

fn reference_ldos_ordinary_spin_fms_dir() -> Result<Option<PathBuf>> {
    let workspace = workspace_root()?;
    let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_spin_fms_short");
    Ok((reference_ldos_expected_present(&path, 2)
        && reference_ldos_source_grid_present(&path)
        && path.join("xsph.inp").is_file()
        && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
    .then_some(path))
}

fn reference_hubbard_nio_ldos_zip() -> Result<Option<PathBuf>> {
    let workspace = workspace_root()?;
    let path = workspace.join("reference-work/golden/HUBBARD/NiO/REFERENCE.zip");
    Ok(path.is_file().then_some(path))
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
        anyhow::bail!(
            "failed to extract {entry} from {}: {stderr}",
            zip_path.display()
        );
    }
    Ok(output.stdout)
}

fn reference_ldos_source_cases() -> Result<Vec<LdosSourceReferenceCase>> {
    let workspace = workspace_root()?;
    let mut cases = Vec::new();
    let exafs_cu = workspace.join("reference-work/golden/EXAFS/Cu");
    if reference_ldos_expected_present(&exafs_cu, 2) && reference_ldos_source_present(&exafs_cu) {
        cases.push(LdosSourceReferenceCase {
            label: "EXAFS/Cu no-FMS",
            source_dir: exafs_cu.clone(),
            expected_dir: exafs_cu,
            potential_count: 2,
        });
    }

    let xanes_cu_source = workspace.join("reference-work/golden/XANES/Cu");
    let xanes_cu_expected = workspace.join("reference-work/golden/LDOS/XANES_Cu_no_fms");
    if reference_ldos_expected_present(&xanes_cu_expected, 2)
        && reference_ldos_source_present(&xanes_cu_source)
    {
        cases.push(LdosSourceReferenceCase {
            label: "XANES/Cu source with no-FMS LDOS card",
            source_dir: xanes_cu_source.clone(),
            expected_dir: xanes_cu_expected,
            potential_count: 2,
        });
    }

    let xanes_cu_spin_expected = workspace.join("reference-work/golden/LDOS/XANES_Cu_spin_no_fms");
    if reference_ldos_expected_present(&xanes_cu_spin_expected, 2)
        && reference_ldos_source_present(&xanes_cu_source)
    {
        cases.push(LdosSourceReferenceCase {
            label: "XANES/Cu source with ordinary spin no-FMS LDOS card",
            source_dir: xanes_cu_source,
            expected_dir: xanes_cu_spin_expected,
            potential_count: 2,
        });
    }

    let gecl4_source = workspace.join("reference-work/golden/NRIXS/GeCl_4");
    let gecl4_production_expected = workspace.join("reference-work/golden/LDOS/GeCl4_no_fms");
    if reference_ldos_expected_present(&gecl4_production_expected, 2)
        && reference_ldos_source_present(&gecl4_source)
    {
        cases.push(LdosSourceReferenceCase {
            label: "NRIXS/GeCl4 source with production no-FMS LDOS card",
            source_dir: gecl4_source.clone(),
            expected_dir: gecl4_production_expected,
            potential_count: 2,
        });
    }

    let gecl4_expected = workspace.join("reference-work/golden/LDOS/GeCl4_no_fms_short");
    if reference_ldos_expected_present(&gecl4_expected, 2)
        && reference_ldos_source_present(&gecl4_source)
    {
        cases.push(LdosSourceReferenceCase {
            label: "NRIXS/GeCl4 source with short no-FMS LDOS card",
            source_dir: gecl4_source,
            expected_dir: gecl4_expected,
            potential_count: 2,
        });
    }

    Ok(cases)
}

fn reference_ldos_source_present(path: &Path) -> bool {
    [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "global.inp",
    ]
    .iter()
    .all(|name| path.join(name).is_file())
}

fn reference_ldos_source_grid_present(path: &Path) -> bool {
    reference_ldos_source_present(path)
        && path.join("geom.dat").is_file()
        && path.join(".dimensions.dat").is_file()
}

fn reference_ldos_expected_present(path: &Path, potential_count: usize) -> bool {
    path.join("ldos.inp").is_file()
        && (0..potential_count).all(|potential| {
            path.join(format!("ldos{potential:02}.dat")).is_file()
                && path.join(format!("rhoc{potential:02}.dat")).is_file()
        })
}

fn assert_ldos_table_mesh_matches(actual: &LdosDatData, expected: &LdosDatData) {
    assert_eq!(actual.fermi_level_ev, expected.fermi_level_ev);
    assert_eq!(actual.charge_transfer, expected.charge_transfer);
    assert_eq!(actual.electron_counts, expected.electron_counts);
    assert_eq!(actual.atom_count, expected.atom_count);
    assert_eq!(actual.lorentzian_hwhh_ev, expected.lorentzian_hwhh_ev);
    assert_eq!(actual.energy_ev.len(), expected.energy_ev.len());
    assert_eq!(actual.density.dim(), expected.density.dim());
    for (actual, expected) in actual.energy_ev.iter().zip(expected.energy_ev.iter()) {
        assert!(
            (actual - expected).abs() <= 5.0e-4,
            "energy actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }
}

fn assert_ldos_density_grid_close(actual: &LdosDatData, expected: &LdosDatData, label: &str) {
    let abs_tolerance = 5.0e-5;
    let rel_tolerance = 1.5e-3;
    for ((row, column), actual) in actual.density.indexed_iter() {
        let expected = expected.density[(row, column)];
        let diff = (actual - expected).abs();
        let rel = diff / expected.abs().max(1.0e-30);
        assert!(
            diff <= abs_tolerance || rel <= rel_tolerance,
            "{label}: density[{row},{column}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
        );
    }
}

fn assert_ldos_magnetic_table_close(
    actual: &LdosMagneticDatData,
    expected: &LdosMagneticDatData,
    label: &str,
) {
    assert_eq!(actual.fermi_level_ev, expected.fermi_level_ev);
    assert_eq!(actual.charge_transfer, expected.charge_transfer);
    assert_eq!(actual.electron_counts, expected.electron_counts);
    assert_eq!(actual.atom_count, expected.atom_count);
    assert_eq!(actual.lorentzian_hwhh_ev, expected.lorentzian_hwhh_ev);
    assert_eq!(actual.angular_limit, expected.angular_limit);
    assert_eq!(actual.energy_ev.len(), expected.energy_ev.len());
    assert_eq!(actual.density.dim(), expected.density.dim());
    for (actual, expected) in actual.energy_ev.iter().zip(expected.energy_ev.iter()) {
        assert!(
            (actual - expected).abs() <= 5.0e-4,
            "{label}: energy actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }
    let abs_tolerance = 5.0e-9;
    let rel_tolerance = 1.0e-5;
    for ((row, column), actual) in actual.density.indexed_iter() {
        let expected = expected.density[(row, column)];
        let diff = (actual - expected).abs();
        let rel = diff / expected.abs().max(1.0e-30);
        assert!(
            diff <= abs_tolerance || rel <= rel_tolerance,
            "{label}: magnetic density[{row},{column}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
        );
    }
}

fn assert_zero_gtr_reference_compatible(actual: &GtrBinData, expected: &GtrBinData) {
    assert_eq!(actual.energy_count(), expected.energy_count());
    assert_eq!(actual.horizontal_count, expected.horizontal_count);
    assert_eq!(actual.danes_extension_count, expected.danes_extension_count);
    assert_eq!(
        actual.highest_potential_index,
        expected.highest_potential_index
    );
    assert_eq!(actual.fms_mode, expected.fms_mode);
    assert!(actual.angular_channel_count() >= expected.angular_channel_count());
    assert!(expected.values.iter().all(|value| value.norm() == 0.0));
    assert!(actual.values.iter().all(|value| value.norm() == 0.0));
}

fn assert_gtr_reference_close(actual: &GtrBinData, expected: &GtrBinData, label: &str) {
    assert_eq!(actual.energy_count(), expected.energy_count(), "{label}");
    assert_eq!(
        actual.horizontal_count, expected.horizontal_count,
        "{label}"
    );
    assert_eq!(
        actual.danes_extension_count, expected.danes_extension_count,
        "{label}"
    );
    assert_eq!(
        actual.highest_potential_index, expected.highest_potential_index,
        "{label}"
    );
    assert_eq!(actual.fms_mode, expected.fms_mode, "{label}");
    assert_eq!(actual.values.dim(), expected.values.dim(), "{label}");
    assert!(
        expected.values.iter().any(|value| value.norm() > 0.0),
        "{label}: reference trace is zero"
    );
    assert!(
        actual.values.iter().any(|value| value.norm() > 0.0),
        "{label}: generated trace is zero"
    );

    let abs_tolerance = 1.0e-4;
    let rel_tolerance = 2.0e-3;
    for ((energy, potential, angular), actual) in actual.values.indexed_iter() {
        let expected = expected.values[(energy, potential, angular)];
        let diff = (*actual - expected).norm();
        let rel = diff / expected.norm().max(1.0e-30);
        assert!(
            diff <= abs_tolerance || rel <= rel_tolerance,
            "{label}: gtr[{energy},{potential},{angular}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
        );
    }
}
