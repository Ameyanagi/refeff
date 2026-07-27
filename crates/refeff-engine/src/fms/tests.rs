use super::{
    PotScfFmsPipelineCache, PotScfFmsSourceGridInput, blocks_downstream_source_generation,
    build_pot_scf_fms_source_grid_handoff, build_reciprocal_fms_source_outputs,
    generated_cached_fms_module_log, has_cached_fms_output, has_cached_fms_solver_output,
    has_cached_mkgtr_output, has_runnable_fms_solver, normalize_hubbard_fms_trace,
    read_optional_fms_reciprocal_input, reciprocal_sig2_prefactor, run_fms_in_dir, run_in_dir,
    run_mkgtr_in_dir, write_hubbard_ldos_first_pass_traces,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4, Array5, Axis, ShapeBuilder};
use num_complex::{Complex32, Complex64};
use refeff_core::{
    FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, MkgtrGreenTraceInput, TransitionBMatrixInput,
    XsphNrixsTransitionIndicesInput, core_hole_quantum_numbers, mkgtr_green_trace,
    transition_b_matrix, xsph_nrixs_transition_indices,
};
use refeff_io::{
    CfAverage, FmsBinData, FmsCluster, FmsControl, FmsDebye, FmsInput, FmslBinData, GeomDat,
    GeomDatRow, GgDatData, GgDatSection, GlobalControl, GlobalInput, GlobalNorms, GlobalQControl,
    GlobalQVector, GtrBinData, GtrDatData, GtrlDatData, HubbardAphaseBinData, HubbardInput,
    HubbardTransformationBinData, HubbardVnlmBinData, LdosControl, LdosFms, LdosInput, LdosMesh,
    ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotControl, PotInput,
    PotPotential, PotRamp, PotRun, PotScattering, PotThermal, PotTolerances, ReciprocalCell,
    ReciprocalKMesh, RhorrpGgDiagBinData, RhorrpGgSliceBinData, fms_input_string, geom_dat_string,
    global_input_string, hubbard_input_string, parse_gtrl_dat, read_fms_bin, read_fmsl_bin,
    read_gg_bin, read_gg_dat, read_gtr_bin, read_gtr_dat, read_gtrl_dat,
    read_hubbard_ldos_gtr_m_bin_inferred, read_hubbard_ldos_gtr_off_bin, read_module_log_dat,
    read_rhorrp_gg_diag_bin, read_rhorrp_gg_slice_bin, read_transformation_hubbard_bin_inferred,
    write_aphase_hubbard_bin, write_fms_bin, write_fmsl_bin, write_gg_bin, write_gg_dat,
    write_gtr_bin, write_gtr_dat, write_gtrl_dat, write_module_log_dat, write_phase_bin,
    write_transformation_hubbard_bin, write_v_hubbard_bin,
};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[test]
fn fms_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 0, -1)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_fms_output(temp.path())?);
    Ok(())
}

#[test]
fn fms_module_skips_when_input_is_missing_and_no_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;

    assert!(!has_cached_fms_output(temp.path())?);
    Ok(())
}

#[test]
fn fms_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_gtr_bin(temp.path().join("gtr00.bin"), &sample_gtr_bin())?;

    assert!(!has_cached_fms_output(temp.path())?);
    Ok(())
}

#[test]
fn fms_solver_does_not_treat_mkgtr_only_cache_as_runnable() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

    assert!(has_cached_fms_output(temp.path())?);
    assert!(!has_cached_fms_solver_output(temp.path())?);
    assert!(!has_runnable_fms_solver(temp.path())?);
    assert!(blocks_downstream_source_generation(temp.path())?);
    Ok(())
}

#[test]
fn fms_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("fms.inp"), "not an fms.inp handoff\n")?;

    assert!(!has_cached_fms_output(temp.path())?);
    assert!(blocks_downstream_source_generation(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed FMS input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("fms.inp"), "{chain}");
    Ok(())
}

#[test]
fn reciprocal_fms_rejects_malformed_declared_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(temp.path().join("reciprocal.inp"), "ispace\n0\ntruncated\n")?;

    let error = read_optional_fms_reciprocal_input(temp.path())
        .expect_err("a present malformed reciprocal.inp must fail closed");
    assert!(error.to_string().contains("failed to parse"));
    Ok(())
}

#[test]
fn reciprocal_fms_core_hole_justone_fails_closed() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_fms_source_input_with_do_fms_options(temp.path(), -1, 0.0, 3.0, 5.0, false, 1)?;
    let input = super::read_input(temp.path())?;
    let cell = ReciprocalCell {
        lattice_vectors: [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        volume_scale: -1.0,
        imaginary_energy: 0.0,
        core_hole_strength: 1.0,
        lattice_name: "P".to_string(),
        space_group_hm: "P1".to_string(),
        space_group: 1,
        atom_count: 1,
        absorber: 1,
        core_hole: 1,
        k_mesh: ReciprocalKMesh {
            total: 1,
            x: 1,
            y: 1,
            z: 1,
            kind: 1,
            use_symmetry: false,
        },
        positions: vec![[0.0, 0.0, 0.0]],
        potentials: vec![1],
        labels: vec!["Cu".to_string()],
        stretch: [0.0, 0.0, 0.0],
    };

    let mut adaptive = cell.clone();
    adaptive.k_mesh.kind = 3;
    let adaptive_error = match build_reciprocal_fms_source_outputs(
        temp.path(),
        &input,
        &global,
        &phase,
        &adaptive,
    ) {
        Ok(_) => anyhow::bail!("adaptive reciprocal FMS did not fail closed"),
        Err(error) => error,
    };
    assert!(adaptive_error.to_string().contains("ktype=3"));

    std::fs::write(temp.path().join("klist.in"), "unsupported override\n")?;
    let klist_error =
        match build_reciprocal_fms_source_outputs(temp.path(), &input, &global, &phase, &cell) {
            Ok(_) => anyhow::bail!("klist.in reciprocal FMS did not fail closed"),
            Err(error) => error,
        };
    assert!(klist_error.to_string().contains("klist.in"));
    std::fs::rename(
        temp.path().join("klist.in"),
        temp.path().join("klist.in.tested"),
    )?;

    let mut zero_potential = cell.clone();
    zero_potential.potentials[0] = 0;
    let ppot_error = match build_reciprocal_fms_source_outputs(
        temp.path(),
        &input,
        &global,
        &phase,
        &zero_potential,
    ) {
        Ok(_) => anyhow::bail!("ppot=0 reciprocal FMS did not fail closed"),
        Err(error) => error,
    };
    assert!(ppot_error.to_string().contains("ppot site 1"));

    let mut phase_with_missing_type = phase.clone();
    phase_with_missing_type
        .potentials
        .push(phase_with_missing_type.potentials[1].clone());
    let mut no_hole = cell.clone();
    no_hole.core_hole = 0;
    let missing_error = match build_reciprocal_fms_source_outputs(
        temp.path(),
        &input,
        &global,
        &phase_with_missing_type,
        &no_hole,
    ) {
        Ok(_) => anyhow::bail!("missing reciprocal potential type did not fail closed"),
        Err(error) => error,
    };
    assert!(
        missing_error
            .to_string()
            .contains("no unit-cell site for potential 2")
    );

    let error =
        match build_reciprocal_fms_source_outputs(temp.path(), &input, &global, &phase, &cell) {
            Ok(_) => anyhow::bail!("unique-potential core-hole reciprocal FMS did not fail closed"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("justone"));

    let mut disabled = input.clone();
    disabled.do_fms = 0;
    let zero = build_reciprocal_fms_source_outputs(temp.path(), &disabled, &global, &phase, &cell)?;
    assert!(
        zero.gg
            .sections
            .iter()
            .flat_map(|section| section.values.iter())
            .all(|value| *value == Complex64::new(0.0, 0.0))
    );
    Ok(())
}

#[test]
fn reciprocal_fms_sig2_prefactor_uses_inverse_angstrom_wave_number_once() -> Result<()> {
    let sig2 = 0.02_f64;
    let wave_number = Complex32::new((1.0 / FEFF_BOHR_ANGSTROM) as f32, 0.0);
    let prefactor = reciprocal_sig2_prefactor(sig2, wave_number)?;
    let expected = (-sig2 / FEFF_BOHR_ANGSTROM.powi(2)).exp();
    assert!((f64::from(prefactor.re) - expected).abs() < 2.0e-7);
    assert_eq!(prefactor.im, 0.0);
    assert!(
        reciprocal_sig2_prefactor(f64::MAX, wave_number)
            .expect_err("huge finite reciprocal SIG2 must fail closed")
            .to_string()
            .contains("single-precision range")
    );
    assert!(
        reciprocal_sig2_prefactor(1.0, Complex32::new(f32::MAX, f32::MAX))
            .expect_err("overflowing reciprocal SIG2 exponent must fail closed")
            .to_string()
            .contains("exponent is not finite")
    );
    Ok(())
}

#[test]
fn fms_module_requires_cache_or_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled FMS should require cache or source handoffs")?;

    assert!(error.to_string().contains(
        "FMS Green's-function generation requires cached FMS output or supported phase.bin/geom.dat/global.inp source handoffs"
    ));
    Ok(())
}

#[test]
fn fms_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, 2)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;
    write_fmsl_bin(temp.path().join("fmsl.bin"), &sample_fmsl_bin())?;
    write_gg_bin(temp.path().join("gg.bin"), &sample_gg_dat())?;
    write_gg_dat(temp.path().join("gg.dat"), &sample_gg_dat())?;
    write_gtr_dat(temp.path().join("gtr.dat"), &sample_gtr_dat())?;
    write_gtr_bin(temp.path().join("gtr00.bin"), &sample_gtr_bin())?;
    write_gtrl_dat(temp.path().join("gtrl.dat"), &sample_gtrl_dat()?)?;
    write_module_log_dat(temp.path().join("log3.dat"), &sample_module_log())?;

    let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let expected_fmsl = read_fmsl_bin(
        temp.path().join("fmsl.bin"),
        expected_fms.pad_width,
        expected_fms.energy_count,
        2,
    )?;
    let expected_gg_bin = read_gg_bin(temp.path().join("gg.bin"))?;
    let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
    let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let expected_gtr_bin = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    let expected_gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;
    let expected_log = read_module_log_dat(temp.path().join("log3.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 8);
    assert!(has_cached_fms_output(temp.path())?);
    assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
    assert_eq!(
        read_fmsl_bin(
            temp.path().join("fmsl.bin"),
            expected_fms.pad_width,
            expected_fms.energy_count,
            2,
        )?,
        expected_fmsl
    );
    assert_eq!(read_gg_bin(temp.path().join("gg.bin"))?, expected_gg_bin);
    assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
    assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
    assert_eq!(
        read_gtr_bin(temp.path().join("gtr00.bin"))?,
        expected_gtr_bin
    );
    assert_eq!(read_gtrl_dat(temp.path().join("gtrl.dat"))?, expected_gtrl);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn fms_module_roundtrips_cached_hubbard_transformation_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_hubbard_input(temp.path(), 1)?;
    let transformation = sample_hubbard_transformation_bin(phase.potential_count());
    let transformation_path = temp.path().join("transformation_hubbard.bin");
    write_transformation_hubbard_bin(&transformation_path, &transformation)?;

    let expected_transformation =
        read_transformation_hubbard_bin_inferred(&transformation_path, 1, phase.potential_count())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(has_cached_fms_output(temp.path())?);
    assert_eq!(
        read_transformation_hubbard_bin_inferred(&transformation_path, 1, phase.potential_count())?,
        expected_transformation
    );
    Ok(())
}

#[test]
fn fms_module_generates_missing_module_log_from_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_fms_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_generates_mkgtr_outputs_from_cached_gg() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input_with_lmax(temp.path(), 1, -1, &[1])?;
    let global = sample_global_input();
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    let gg = sample_mkgtr_gg();
    write_gg_bin(temp.path().join("gg.bin"), &gg)?;

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 1.0e-12);
    assert_eq!(fms.declared_spectrum_count, Some(0));
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_eq!(fms.main_energy_count, phase.main_energy_count);
    assert_eq!(fms.highest_potential_index, phase.potential_count() - 1);
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 1.0e-8);
    assert_eq!(gtr.energy, phase.energy_grid);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(2)
    );
    Ok(())
}

#[test]
fn mkgtr_generates_sparse_cartesian_eels_spectra_and_repairs_stale_range() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input_with_lmax(temp.path(), 1, -1, &[1])?;
    let mut global = sample_global_input();
    global.control.ipol = 1;
    global.polarization_tensor = [[0.0; 6]; 3];
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    let phase = sample_phase_bin();
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    write_gg_bin(temp.path().join("gg.bin"), &sample_mkgtr_gg())?;
    write_eels_input(temp.path(), 1, 4, 9)?;

    assert!(run_mkgtr_in_dir(temp.path())? > 0);
    let sparse = read_fms_bin(temp.path().join("fms.bin"))?;
    let sparse_gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_eq!(sparse.declared_spectrum_count, Some(0));
    assert_eq!(sparse.spectrum_count(), 3);
    assert_complex_vec_close(sparse.spectra.row(0), sparse_gtr.trace.view(), 2.0e-6);
    assert!(
        sparse
            .spectra
            .rows()
            .into_iter()
            .all(|row| row.iter().any(|value| value.norm() > 1.0e-10))
    );
    assert!(sparse.spectra.rows().into_iter().skip(1).all(|row| {
        row.iter()
            .zip(sparse.spectra.row(0))
            .any(|(actual, first)| (*actual - *first).norm() > 1.0e-8)
    }));

    let first_fms_bytes = std::fs::read(temp.path().join("fms.bin"))?;
    let first_gtr_bytes = std::fs::read(temp.path().join("gtr.dat"))?;
    run_mkgtr_in_dir(temp.path())?;
    assert_eq!(std::fs::read(temp.path().join("fms.bin"))?, first_fms_bytes);
    assert_eq!(std::fs::read(temp.path().join("gtr.dat"))?, first_gtr_bytes);

    // A narrower selector request must not reuse the extra spectra left by
    // the previous range: ordinal spectrum indices would otherwise map to the
    // wrong Cartesian tensors downstream in FF2X.
    write_eels_input(temp.path(), 5, 1, 5)?;
    assert!(!has_cached_mkgtr_output(temp.path())?);
    run_mkgtr_in_dir(temp.path())?;
    let single = read_fms_bin(temp.path().join("fms.bin"))?;
    assert_eq!(single.spectrum_count(), 1);
    assert_ne!(single.spectra.row(0), sparse.spectra.row(0));
    assert!(has_cached_mkgtr_output(temp.path())?);

    // Without the authoritative phase/global handoffs, a later range change
    // cannot be reconstructed safely and must not claim the stale cache.
    write_eels_input(temp.path(), 1, 4, 9)?;
    std::fs::remove_file(temp.path().join("phase.bin"))?;
    std::fs::remove_file(temp.path().join("global.inp"))?;
    let error = run_mkgtr_in_dir(temp.path())
        .err()
        .context("stale EELS range without source handoffs should fail closed")?;
    assert!(
        error
            .to_string()
            .contains("MKGTR EELS fms.bin has 1 spectrum payload(s), expected 3"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn mkgtr_rejects_invalid_eels_polarization_ranges() -> Result<()> {
    for (minimum, step, maximum) in [(0, 1, 9), (1, 0, 9), (9, 1, 1), (1, 1, 11)] {
        let temp = tempfile::tempdir()?;
        write_fms_input_with_lmax(temp.path(), 1, -1, &[1])?;
        std::fs::write(
            temp.path().join("global.inp"),
            global_input_string(&sample_global_input())?,
        )?;
        write_phase_bin(temp.path().join("phase.bin"), &sample_phase_bin())?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_mkgtr_gg())?;
        write_eels_input(temp.path(), minimum, step, maximum)?;

        let error = run_mkgtr_in_dir(temp.path())
            .err()
            .context("invalid EELS polarization range should fail MKGTR")?;
        assert!(
            error
                .to_string()
                .contains("MKGTR EELS polarization range must satisfy"),
            "{error:?}"
        );
    }
    Ok(())
}

#[test]
fn mkgtr_module_matches_pinned_elnes_cartesian_spectra() -> Result<()> {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference-work/golden/ELNES/Cu");
    let required = [
        "fms.inp",
        "global.inp",
        "phase.bin",
        "gg.bin",
        "eels.inp",
        "fms.bin",
        "gtr.dat",
    ];
    if !required.iter().all(|name| reference.join(name).is_file()) {
        crate::require_fixture!("pinned ELNES/Cu MKGTR reference is incomplete");
    }

    let expected_fms = read_fms_bin(reference.join("fms.bin"))?;
    let expected_gtr = read_gtr_dat(reference.join("gtr.dat"))?;
    let temp = tempfile::tempdir()?;
    for name in ["fms.inp", "global.inp", "phase.bin", "gg.bin", "eels.inp"] {
        std::fs::copy(reference.join(name), temp.path().join(name))?;
    }

    assert!(run_mkgtr_in_dir(temp.path())? > 0);
    let actual_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let actual_gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_eq!(actual_fms.spectrum_count(), 9);
    assert_eq!(actual_fms.spectra.dim(), expected_fms.spectra.dim());
    assert_complex_table_close(
        actual_fms.spectra.view(),
        expected_fms.spectra.view(),
        2.0e-7,
    );
    assert_complex_vec_close(actual_gtr.trace.view(), expected_gtr.trace.view(), 2.0e-6);
    Ok(())
}

#[test]
fn fms_module_generates_gg_and_mkgtr_outputs_from_phase_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));
    assert!(gg.sections.iter().any(|section| {
        section
            .values
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    }));
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
    assert_eq!(gtr.energy, phase.energy_grid);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);

    let log = read_module_log_dat(temp.path().join("log3.dat"))?;
    assert_log_contains(&log, "FMS calculation of full Green's function ...");
    assert_log_contains(&log, "Using     2 energy points.");
    assert_log_contains(&log, "Done with module: FMS.");
    assert_log_contains(&log, "MKGTR: Tracing over Green's function ...");
    assert_log_contains(&log, "Done with module: MKGTR.");
    Ok(())
}

#[test]
fn standalone_fms_and_mkgtr_preserve_the_upstream_stage_boundary() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), -1, 0.0)?;

    assert!(!has_cached_fms_solver_output(temp.path())?);
    assert!(!has_cached_mkgtr_output(temp.path())?);
    assert!(run_fms_in_dir(temp.path())? > 0);
    assert!(has_cached_fms_solver_output(temp.path())?);
    assert!(!has_cached_mkgtr_output(temp.path())?);
    assert!(temp.path().join("gg.bin").is_file());
    assert!(temp.path().join("gg.dat").is_file());
    assert!(!temp.path().join("fms.bin").exists());
    assert!(!temp.path().join("gtr.dat").exists());

    assert!(run_mkgtr_in_dir(temp.path())? > 0);
    assert!(has_cached_mkgtr_output(temp.path())?);
    assert!(temp.path().join("fms.bin").is_file());
    assert!(temp.path().join("gtr.dat").is_file());
    let log = read_module_log_dat(temp.path().join("log3.dat"))?;
    assert_log_contains(&log, "Done with module: FMS.");
    assert_log_contains(&log, "Done with module: MKGTR.");
    Ok(())
}

#[test]
fn full_run_scheduler_reports_fms_and_mkgtr_as_separate_stages() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), -1, 0.0)?;

    let first = crate::run_supported_cached_modules(temp.path())?;
    let first_fms = first
        .iter()
        .find(|report| report.name == "fms")
        .context("source run did not report FMS")?;
    let first_mkgtr = first
        .iter()
        .find(|report| report.name == "mkgtr")
        .context("source run did not report MKGTR")?;
    assert_eq!(first_fms.status, crate::StageStatus::Generated);
    assert_eq!(first_mkgtr.status, crate::StageStatus::Generated);

    let second = crate::run_supported_cached_modules(temp.path())?;
    let second_fms = second
        .iter()
        .find(|report| report.name == "fms")
        .context("cached run did not report FMS")?;
    let second_mkgtr = second
        .iter()
        .find(|report| report.name == "mkgtr")
        .context("cached run did not report MKGTR")?;
    assert_eq!(second_fms.status, crate::StageStatus::Cached);
    assert_eq!(second_mkgtr.status, crate::StageStatus::Cached);
    Ok(())
}

#[test]
fn standalone_fms_and_mkgtr_generate_cache_free_nrixs_jas_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let phase = write_fms_nrixs_source_handoffs(temp.path())?;

    assert!(!has_cached_fms_solver_output(temp.path())?);
    assert!(!has_cached_mkgtr_output(temp.path())?);
    assert!(run_fms_in_dir(temp.path())? > 0);
    assert!(has_cached_fms_solver_output(temp.path())?);
    assert!(!has_cached_mkgtr_output(temp.path())?);
    assert!(!temp.path().join("fms.bin").exists());
    assert!(!temp.path().join("fmsl.bin").exists());

    assert!(run_mkgtr_in_dir(temp.path())? >= 4);
    assert!(has_cached_mkgtr_output(temp.path())?);
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let fmsl = read_fmsl_bin(
        temp.path().join("fmsl.bin"),
        fms.pad_width,
        fms.energy_count,
        2,
    )?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;
    assert_eq!(fms.declared_spectrum_count, None);
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_eq!(fms.spectrum_count(), 1);
    assert_eq!(fmsl.traces.dim(), (phase.energy_count, 3, 3));
    assert_eq!(gtrl.row_count(), phase.energy_count);
    assert_eq!(gtrl.component_count(), 9);
    assert_complex_vec_close(fms.spectra.row(0), gtr.trace.view(), 2.0e-6);
    assert!(
        gtr.trace
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-10)
    );
    Ok(())
}

#[test]
fn full_run_scheduler_generates_cache_free_nrixs_jas_fms_and_mkgtr() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_nrixs_source_handoffs(temp.path())?;

    let first = crate::run_supported_cached_modules(temp.path())?;
    let first_fms = first
        .iter()
        .find(|report| report.name == "fms")
        .context("NRIXS source run did not report FMS")?;
    let first_mkgtr = first
        .iter()
        .find(|report| report.name == "mkgtr")
        .context("NRIXS source run did not report MKGTR")?;
    assert_eq!(first_fms.status, crate::StageStatus::Generated);
    assert_eq!(first_mkgtr.status, crate::StageStatus::Generated);
    for name in ["gg.bin", "fms.bin", "gtr.dat", "fmsl.bin", "gtrl.dat"] {
        assert!(temp.path().join(name).is_file(), "missing {name}");
    }

    let second = crate::run_supported_cached_modules(temp.path())?;
    let second_fms = second
        .iter()
        .find(|report| report.name == "fms")
        .context("cached NRIXS run did not report FMS")?;
    let second_mkgtr = second
        .iter()
        .find(|report| report.name == "mkgtr")
        .context("cached NRIXS run did not report MKGTR")?;
    assert_eq!(second_fms.status, crate::StageStatus::Cached);
    assert_eq!(second_mkgtr.status, crate::StageStatus::Cached);
    Ok(())
}

#[test]
fn mkgtr_module_matches_pinned_nrixs_jas_reference_traces() -> Result<()> {
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference-work/golden/NRIXS/GeCl_4");
    for name in [
        "fms.inp",
        "global.inp",
        "phase.bin",
        "gg.bin",
        "fms.bin",
        "gtr.dat",
        "fmsl.bin",
        "gtrl.dat",
    ] {
        if !reference.join(name).is_file() {
            crate::require_fixture!("pinned NRIXS/JAS GeCl4 MKGTR reference is incomplete");
        }
    }
    let expected_fms = read_fms_bin(reference.join("fms.bin"))?;
    let expected_gtr = read_gtr_dat(reference.join("gtr.dat"))?;
    let expected_fmsl = read_fmsl_bin(
        reference.join("fmsl.bin"),
        expected_fms.pad_width,
        expected_fms.energy_count,
        2,
    )?;
    let expected_gtrl = read_gtrl_dat(reference.join("gtrl.dat"))?;
    let temp = tempfile::tempdir()?;
    for name in ["fms.inp", "global.inp", "phase.bin", "gg.bin"] {
        std::fs::copy(reference.join(name), temp.path().join(name))?;
    }

    assert!(run_mkgtr_in_dir(temp.path())? >= 4);

    let actual_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let actual_gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let actual_fmsl = read_fmsl_bin(
        temp.path().join("fmsl.bin"),
        actual_fms.pad_width,
        actual_fms.energy_count,
        2,
    )?;
    let actual_gtrl = read_gtrl_dat(temp.path().join("gtrl.dat"))?;
    assert_complex_table_close(
        actual_fms.spectra.view(),
        expected_fms.spectra.view(),
        2.0e-5,
    );
    assert_complex_vec_close(actual_gtr.trace.view(), expected_gtr.trace.view(), 2.0e-5);
    assert_complex_array3_close(&actual_fmsl.traces, &expected_fmsl.traces, 2.0e-5);
    assert_complex_table_close(
        actual_gtrl.decomposed_trace.view(),
        expected_gtrl.decomposed_trace.view(),
        2.0e-5,
    );
    Ok(())
}

#[test]
fn fms_module_does_not_advertise_malformed_phase_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

    assert!(!has_cached_fms_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed phase.bin should fail through the explicit FMS runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    Ok(())
}

#[test]
fn fms_module_does_not_claim_cached_gg_with_malformed_phase_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    let gg = sample_gg_dat();
    write_gg_dat(temp.path().join("gg.dat"), &gg)?;
    std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

    assert!(!has_cached_fms_output(temp.path())?);
    assert!(blocks_downstream_source_generation(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed phase.bin should fail through the explicit FMS runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("phase.bin"), "{chain}");
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 0.0);
    assert!(!temp.path().join("log3.dat").exists());
    Ok(())
}

#[test]
fn fms_module_does_not_advertise_malformed_dmdw_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_dmdw_source_handoffs(temp.path(), 5)?;
    write_fms_dmdw_handoffs(temp.path())?;
    std::fs::write(temp.path().join("feff.dym"), "not a dym source\n")?;

    assert!(!has_cached_fms_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed feff.dym should fail through the explicit FMS runner")?;
    let chain = format!("{error:?}");
    assert!(chain.contains("idwopt=5"), "{chain}");
    assert!(chain.contains("dym file"), "{chain}");
    Ok(())
}

#[test]
fn fms_module_recovers_malformed_gg_dat_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    std::fs::write(temp.path().join("gg.dat"), b"not a gg.dat cache\n")?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);
    Ok(())
}

#[test]
fn fms_module_regenerates_stale_readable_gg_dat_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    run_in_dir(temp.path())?;
    let expected_gg = read_gg_dat(temp.path().join("gg.dat"))?;
    let mut stale_gg = expected_gg.clone();
    stale_gg.sections[0].values[(0, 0)].re += 0.25;
    write_gg_dat(temp.path().join("gg.dat"), &stale_gg)?;
    let mut stale_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    stale_fms.spectra[(0, 0)] += Complex64::new(0.5, -0.25);
    write_fms_bin(temp.path().join("fms.bin"), &stale_fms)?;
    let mut stale_gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    stale_gtr.trace[0] += Complex64::new(-0.125, 0.375);
    write_gtr_dat(temp.path().join("gtr.dat"), &stale_gtr)?;

    assert!(has_cached_fms_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_gg_values_close(&gg, &expected_gg, 1.0e-12);
    assert_gg_values_close(
        &read_gg_bin(temp.path().join("gg.bin"))?,
        &expected_gg,
        1.0e-12,
    );
    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);
    Ok(())
}

#[test]
fn fms_ldos_gtr_phase_grid_handoff_regenerates_stale_readable_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, mut phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    for energy in &mut phase.energy_grid {
        energy.im = 0.02;
    }
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    let ldos = sample_ldos_fmsdos_input_for_phase(&phase);

    let count = super::write_ldos_gtr_bin_source_handoffs(temp.path(), &ldos)?;

    assert_eq!(count, 2);
    let expected_gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
    let expected_gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
    assert!(
        expected_gtr01
            .values
            .iter()
            .any(|value| value.norm() > f64::EPSILON),
        "non-absorber LDOS center trace is empty"
    );
    let mut stale_gtr00 = expected_gtr00.clone();
    stale_gtr00.values[(0, 0, 0)] += Complex64::new(0.25, -0.125);
    write_gtr_bin(temp.path().join("gtr00.bin"), &stale_gtr00)?;

    let count = super::write_ldos_gtr_bin_source_handoffs(temp.path(), &ldos)?;

    assert_eq!(count, 1);
    assert_eq!(read_gtr_bin(temp.path().join("gtr00.bin"))?, expected_gtr00);
    assert_eq!(read_gtr_bin(temp.path().join("gtr01.bin"))?, expected_gtr01);
    Ok(())
}

#[test]
fn fms_module_recovers_paired_malformed_gg_caches_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    std::fs::write(temp.path().join("gg.bin"), b"not a gg.bin cache\n")?;
    std::fs::write(temp.path().join("gg.dat"), b"not a gg.dat cache\n")?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    Ok(())
}

#[test]
fn fms_module_bootstraps_active_hubbard_control_without_v_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_hubbard_input(temp.path(), 1)?;

    assert!(has_cached_fms_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    assert_eq!(
        read_gg_dat(temp.path().join("gg.dat"))?.section_count(),
        phase.energy_count
    );
    assert!(temp.path().join("fms.bin").is_file());
    assert!(temp.path().join("gtr.dat").is_file());
    assert!(!temp.path().join("gtr_m00.bin").exists());
    Ok(())
}

#[test]
fn fms_module_requires_complete_active_hubbard_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_hubbard_input(temp.path(), 1)?;
    write_active_hubbard_v_source(temp.path(), 1, phase.potential_count())?;

    assert!(!has_cached_fms_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("active Hubbard FMS source handoff should require Hubbard side files")?;

    assert!(
        error.to_string().contains(
            "active Hubbard FMS source generation requires aphase_hubbard.bin and transformation_hubbard.bin"
        ),
        "{error:?}"
    );
    assert!(!temp.path().join("gg.bin").is_file());
    assert!(!temp.path().join("gg.dat").is_file());
    Ok(())
}

#[test]
fn fms_module_rejects_malformed_active_hubbard_v_source_without_ordinary_fallback() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_hubbard_input(temp.path(), 1)?;
    std::fs::write(
        temp.path().join("v_hubbard.bin"),
        b"not a Hubbard V source\n",
    )?;

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed active Hubbard V source must fail closed")?;

    assert!(format!("{error:#}").contains("v_hubbard.bin"), "{error:?}");
    assert!(!temp.path().join("gg.bin").exists());
    assert!(!temp.path().join("gg.dat").exists());
    Ok(())
}

#[test]
fn fms_module_generates_active_hubbard_gg_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_hubbard_input(temp.path(), 1)?;
    write_active_hubbard_v_source(temp.path(), 1, phase.potential_count())?;
    write_aphase_hubbard_bin(
        temp.path().join("aphase_hubbard.bin"),
        &sample_active_aphase_hubbard_bin(&phase),
    )?;
    write_transformation_hubbard_bin(
        temp.path().join("transformation_hubbard.bin"),
        &sample_active_hubbard_transformation_bin(phase.potential_count()),
    )?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));
    assert!(gg.sections.iter().any(|section| {
        section
            .values
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    }));
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
    assert_eq!(fms.energy_count, phase.energy_count);
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
    assert_eq!(gtr.energy, phase.energy_grid);
    assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);

    let log = read_module_log_dat(temp.path().join("log3.dat"))?;
    assert_log_contains(&log, "FMS calculation of full Green's function ...");
    assert_log_contains(&log, "Using     2 energy points.");
    assert_log_contains(&log, "Done with module: FMS.");
    assert_log_contains(&log, "MKGTR: Tracing over Green's function ...");
    assert_log_contains(&log, "Done with module: MKGTR.");
    Ok(())
}

#[test]
fn fms_active_hubbard_gg_uses_dimensions_spin_capacity_and_absorber_lmax() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    let mut input = super::read_input(temp.path())?;
    input.lmaxph = vec![1, 2];
    std::fs::write(temp.path().join("fms.inp"), fms_input_string(&input)?)?;
    std::fs::write(
        temp.path().join(".dimensions.dat"),
        format!("{:12}{:12}{:12}{:12}\n", 35, 2, 1, 2),
    )?;
    write_hubbard_input(temp.path(), 1)?;
    write_active_hubbard_v_source(temp.path(), 1, phase.potential_count())?;
    write_aphase_hubbard_bin(
        temp.path().join("aphase_hubbard.bin"),
        &sample_active_aphase_hubbard_bin_with_limit(&phase, 2),
    )?;
    write_transformation_hubbard_bin(
        temp.path().join("transformation_hubbard.bin"),
        &sample_active_hubbard_transformation_bin_with_limit(phase.potential_count(), 2),
    )?;

    run_in_dir(temp.path())?;

    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (8, 8)));
    for section in &gg.sections {
        assert!(
            section
                .values
                .iter()
                .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
        );
        for column in 4..8 {
            for row in 0..8 {
                assert_eq!(section.values[(row, column)], Complex64::new(0.0, 0.0));
                assert_eq!(section.values[(column, row)], Complex64::new(0.0, 0.0));
            }
        }
    }
    Ok(())
}

#[test]
fn fms_module_generates_two_spin_active_hubbard_gg_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_input(temp.path(), -1, 0.0)?;
    let global = sample_global_input();
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    let mut phase = sample_two_spin_fms_source_phase_bin();
    let imaginary_energy = phase.energy_grid[0].im;
    phase
        .energy_grid
        .iter_mut()
        .for_each(|energy| energy.im = imaginary_energy);
    write_phase_bin(temp.path().join("phase.bin"), &phase)?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&sample_fms_source_geom())?,
    )?;
    write_hubbard_input(temp.path(), 1)?;
    write_active_hubbard_v_source(temp.path(), 1, phase.potential_count())?;
    write_aphase_hubbard_bin(
        temp.path().join("aphase_hubbard.bin"),
        &sample_active_aphase_hubbard_bin(&phase),
    )?;
    write_transformation_hubbard_bin(
        temp.path().join("transformation_hubbard.bin"),
        &sample_active_hubbard_transformation_bin(phase.potential_count()),
    )?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert!(count >= 2);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (8, 8)));
    assert!(gg.sections.iter().any(|section| {
        section
            .values
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    }));
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    let gtr_m = read_hubbard_ldos_gtr_m_bin_inferred(temp.path().join("gtr_m00.bin"))?;
    assert_eq!(gtr_m.energy_count(), phase.energy_count);
    assert_eq!(gtr_m.horizontal_count, phase.main_energy_count);
    assert_eq!(gtr_m.danes_extension_count, phase.auxiliary_energy_count);
    assert_eq!(gtr_m.potential_count(), phase.potential_count());
    assert_eq!(gtr_m.values.len_of(Axis(0)), 2);
    assert!(
        gtr_m
            .values
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    );
    assert_ne!(gtr_m.values[(0, 0, 0, 1, 1)], gtr_m.values[(1, 0, 0, 1, 1)]);
    Ok(())
}

#[test]
fn fms_generates_hubbard_first_pass_magnetic_and_offdiagonal_traces() -> Result<()> {
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
    ] {
        std::fs::copy(reference.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy Hubbard LDOS fixture {name}"))?;
    }
    let phase = refeff_io::read_phase_bin(temp.path().join("phase.bin"))?;
    let mut ldos = crate::ldos::read_input(temp.path())?;
    ldos.control.lfms2 = 0;
    ldos.control.neldos = 2;
    let hubbard = super::read_hubbard_input(temp.path())?;
    let hubbard_l = usize::try_from(hubbard.l)?;

    assert_eq!(write_hubbard_ldos_first_pass_traces(temp.path(), &ldos)?, 2);

    let gtr_m = read_hubbard_ldos_gtr_m_bin_inferred(temp.path().join("gtr_m00.bin"))?;
    let angular_limit = gtr_m.angular_limit;
    let gtr_off =
        read_hubbard_ldos_gtr_off_bin(temp.path().join("gtr_off00.bin"), hubbard_l, angular_limit)?;
    assert_eq!(gtr_m.energy_count(), ldos.control.neldos as usize);
    assert_eq!(gtr_m.potential_count(), phase.potential_count());
    assert_eq!(gtr_m.angular_limit, 2);
    assert!(gtr_m.angular_limit >= hubbard_l);
    assert_eq!(gtr_off.energy_count(), ldos.control.neldos as usize);
    assert_eq!(gtr_off.potential_count(), phase.potential_count());
    assert_eq!(gtr_off.order(), (hubbard_l + 1).pow(2));
    assert!(
        gtr_m
            .values
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    );
    assert!(
        gtr_off
            .values
            .iter()
            .all(|value| *value == Complex32::new(0.0, 0.0))
    );
    for potential in 0..phase.potential_count() {
        assert!(
            gtr_m
                .values
                .index_axis(Axis(2), potential)
                .index_axis(Axis(2), 0)
                .iter()
                .any(|value| value.re.abs() + value.im.abs() > 1.0e-8),
            "Hubbard first pass produced no s-channel trace for potential {potential}"
        );
    }
    for angular in 1..=gtr_m.angular_limit {
        assert!(
            gtr_m
                .values
                .index_axis(Axis(3), angular)
                .iter()
                .all(|value| *value == Complex32::new(0.0, 0.0)),
            "Hubbard first-pass compatibility solve populated l={angular}"
        );
    }
    Ok(())
}

#[test]
fn fms_hubbard_zero_lmax_compatibility_retains_hubbard_s_offdiagonal_trace() -> Result<()> {
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
    ] {
        std::fs::copy(reference.join(name), temp.path().join(name))
            .with_context(|| format!("failed to copy Hubbard LDOS fixture {name}"))?;
    }
    let mut hubbard = super::read_hubbard_input(temp.path())?;
    hubbard.l = 0;
    std::fs::write(
        temp.path().join("hubbard.inp"),
        hubbard_input_string(&hubbard)?,
    )?;
    let mut ldos = crate::ldos::read_input(temp.path())?;
    ldos.control.lfms2 = 0;
    ldos.control.neldos = 2;

    assert_eq!(write_hubbard_ldos_first_pass_traces(temp.path(), &ldos)?, 2);

    let gtr_off = read_hubbard_ldos_gtr_off_bin(temp.path().join("gtr_off00.bin"), 0, 2)?;
    assert!(
        gtr_off
            .values
            .index_axis(Axis(0), 0)
            .iter()
            .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
    );
    for angular in 1..=gtr_off.angular_limit {
        assert!(
            gtr_off
                .values
                .index_axis(Axis(0), angular)
                .iter()
                .all(|value| *value == Complex32::new(0.0, 0.0))
        );
    }
    Ok(())
}

#[test]
fn hubbard_fms_trace_normalization_matches_feff_formula() {
    let scattering = Complex32::new(0.75, -0.25);
    let phase = Complex32::new(0.2, 0.03);
    let angular = 2;
    let expected = scattering * (Complex32::new(0.0, 2.0) * phase).exp() / (2 * angular + 1) as f32;
    assert_eq!(
        normalize_hubbard_fms_trace(scattering, phase, angular),
        expected
    );
}

#[test]
fn fms_module_generates_active_hubbard_saved_scattering_slices() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_fms_source_input_with_options(temp.path(), -1, 0.0, 3.0, 5.0, true)?;
    write_hubbard_input(temp.path(), 1)?;
    write_active_hubbard_v_source(temp.path(), 1, phase.potential_count())?;
    write_aphase_hubbard_bin(
        temp.path().join("aphase_hubbard.bin"),
        &sample_active_aphase_hubbard_bin(&phase),
    )?;
    write_transformation_hubbard_bin(
        temp.path().join("transformation_hubbard.bin"),
        &sample_active_hubbard_transformation_bin(phase.potential_count()),
    )?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 8);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));
    let slice = read_rhorrp_gg_slice_bin(temp.path().join("gg_slice.bin"))?;
    let diag = read_rhorrp_gg_diag_bin(temp.path().join("gg_diag.bin"))?;
    assert_eq!(slice.energy_count(), phase.energy_count);
    assert_eq!(slice.row_count(), 4);
    assert_eq!(slice.column_count(), 8);
    assert_eq!(diag.energy_count(), phase.energy_count);
    assert_eq!(diag.atom_count(), 2);
    assert_eq!(diag.row_count(), 4);
    assert_eq!(diag.column_count(), 4);
    assert_saved_scattering_absorber_block_matches_gg(&gg, &slice, &diag, 5.0e-6);
    assert!(slice.values.iter().any(|value| value.norm() > 1.0e-8));
    assert!(diag.values.iter().any(|value| value.norm() > 1.0e-8));
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_iterative_minv_solvers() -> Result<()> {
    for minv in [1, 2, 3, 4] {
        let temp = tempfile::tempdir()?;
        let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
        write_fms_source_input_with_solver_options(temp.path(), -1, 0.0, 3.0, 5.0, false, 0, minv)?;

        assert!(has_cached_fms_output(temp.path())?);

        let count =
            run_in_dir(temp.path()).with_context(|| format!("running FMS source minv={minv}"))?;

        assert_eq!(count, 5, "minv={minv}");
        let gg = read_gg_dat(temp.path().join("gg.dat"))?;
        assert_eq!(gg.section_count(), phase.energy_count, "minv={minv}");
        assert!(
            gg.sections.iter().all(|section| section.shape() == (4, 4)),
            "minv={minv}"
        );
        assert!(
            gg.sections.iter().any(|section| {
                section
                    .values
                    .iter()
                    .any(|value| value.re.abs() + value.im.abs() > 1.0e-8)
            }),
            "minv={minv}"
        );
        assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);

        let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
        let fms = read_fms_bin(temp.path().join("fms.bin"))?;
        let gtr = read_gtr_dat(temp.path().join("gtr.dat"))?;
        assert_eq!(fms.energy_count, phase.energy_count, "minv={minv}");
        assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
        assert_eq!(gtr.energy, phase.energy_grid, "minv={minv}");
        assert_complex_vec_close(gtr.trace.view(), expected_trace.row(0), 2.0e-6);

        let log = read_module_log_dat(temp.path().join("log3.dat"))?;
        assert_log_contains(&log, "FMS calculation of full Green's function ...");
        assert_log_contains(&log, "Done with module: FMS.");
        assert_log_contains(&log, "Done with module: MKGTR.");
    }
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_global_sig2_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(undamped_temp.path(), -1, 0.0)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(damped_temp.path(), -1, 0.025)?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_correlated_debye_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(undamped_temp.path(), -1, 0.0)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(damped_temp.path(), 0, 0.0)?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using a Correlated Debye model.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_classical_debye_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(undamped_temp.path(), -1, 0.0)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(damped_temp.path(), 3, 0.0)?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using the Classical Debye model.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_sig2_dat_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(undamped_temp.path(), -1, 0.0)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_source_handoffs(damped_temp.path(), 4, 0.0)?;
    std::fs::write(damped_temp.path().join("sig2.dat"), "1 2 0.025 1.4\n")?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using the sig.dat file.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_dmdw_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_dmdw_source_handoffs(undamped_temp.path(), -1)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_dmdw_source_handoffs(damped_temp.path(), 5)?;
    write_fms_dmdw_handoffs(damped_temp.path())?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using the ab-initio Dynamical Matrix model.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_recursion_debye_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_spring_source_handoffs(undamped_temp.path(), -1)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_spring_source_handoffs(damped_temp.path(), 2)?;
    write_fms_spring_handoffs(damped_temp.path())?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using the Recursion method.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_equation_of_motion_debye_damping() -> Result<()> {
    let undamped_temp = tempfile::tempdir()?;
    write_fms_spring_source_handoffs(undamped_temp.path(), -1)?;
    let damped_temp = tempfile::tempdir()?;
    write_fms_spring_source_handoffs(damped_temp.path(), 1)?;
    write_fms_spring_handoffs(damped_temp.path())?;

    assert!(has_cached_fms_output(damped_temp.path())?);

    assert_eq!(run_in_dir(undamped_temp.path())?, 5);
    assert_eq!(run_in_dir(damped_temp.path())?, 5);
    let undamped = read_gg_dat(undamped_temp.path().join("gg.dat"))?;
    let damped = read_gg_dat(damped_temp.path().join("gg.dat"))?;

    assert_gg_values_differ(&damped, &undamped, 1.0e-7);
    let log = read_module_log_dat(damped_temp.path().join("log3.dat"))?;
    assert_log_contains(
        &log,
        "Applying Debye-Waller factors using the Equation-of-Motion method.",
    );
    Ok(())
}

#[test]
fn fms_module_generates_saved_scattering_slices_from_phase_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (_global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_fms_source_input_with_options(temp.path(), -1, 0.0, 3.0, 5.0, true)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 7);
    let slice = read_rhorrp_gg_slice_bin(temp.path().join("gg_slice.bin"))?;
    let diag = read_rhorrp_gg_diag_bin(temp.path().join("gg_diag.bin"))?;
    assert_eq!(slice.energy_count(), phase.energy_count);
    assert_eq!(slice.row_count(), 4);
    assert_eq!(slice.column_count(), 8);
    assert_eq!(diag.energy_count(), phase.energy_count);
    assert_eq!(diag.atom_count(), 2);
    assert_eq!(diag.row_count(), 4);
    assert_eq!(diag.column_count(), 4);
    assert_saved_scattering_absorber_block_matches_gg(
        &read_gg_dat(temp.path().join("gg.dat"))?,
        &slice,
        &diag,
        5.0e-6,
    );
    assert!(slice.values.iter().any(|value| value.norm() > 1.0e-8));
    assert!(diag.values.iter().any(|value| value.norm() > 1.0e-8));
    Ok(())
}

#[test]
fn fms_module_generates_full_cluster_source_gg_and_progress_log() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), -1, 0.0)?;
    write_fms_source_input_with_do_fms_options(temp.path(), -1, 0.0, 3.0, 5.0, false, 1)?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);

    let log = read_module_log_dat(temp.path().join("log3.dat"))?;
    assert_log_contains(&log, "FMS for a cluster of    2 atoms");
    assert_log_contains(&log, "Energy point    1/   2");
    Ok(())
}

#[test]
fn fms_module_generates_source_gg_with_feff_default_absorber_cluster() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let (global, phase) = write_fms_source_handoffs(temp.path(), 0, 0.0)?;
    write_fms_source_input_with_cluster(temp.path(), 0, 0.0, -1.0, -1.0)?;

    assert!(has_cached_fms_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 5);
    let gg = read_gg_dat(temp.path().join("gg.dat"))?;
    assert_eq!(gg.section_count(), phase.energy_count);
    assert!(gg.sections.iter().all(|section| section.shape() == (4, 4)));

    let expected_trace = expected_mkgtr_trace(&global, &phase, &gg, 1)?;
    let fms = read_fms_bin(temp.path().join("fms.bin"))?;
    assert_eq!(fms.cluster_radius_angstrom, -1.0);
    assert_complex_table_close(fms.spectra.view(), expected_trace.view(), 2.0e-6);
    Ok(())
}

#[test]
fn pot_scf_fms_source_grid_builds_all_potential_trace() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&sample_fms_source_geom())?,
    )?;
    let pot = sample_pot_scf_fms_input(1, 3.0);
    let energies = Array1::from_vec(vec![Complex64::new(1.0, 0.02), Complex64::new(1.3, 0.03)]);
    let references = Array2::from_shape_fn((energies.len(), 2), |(energy, potential)| {
        Complex64::new(0.05 * potential as f64, -energies[energy].im)
    });
    let phase_shifts =
        Array3::from_shape_fn((energies.len(), 2, 2), |(energy, angular, potential)| {
            Complex64::new(
                0.02 * (energy + 1) as f64 + 0.03 * angular as f64,
                -0.004 * potential as f64,
            )
        });

    let handoff = build_pot_scf_fms_source_grid_handoff(
        temp.path(),
        PotScfFmsSourceGridInput {
            pot: &pot,
            energy_grid_hartree: energies.view(),
            reference_energies_hartree: references.view(),
            phase_shifts: phase_shifts.view(),
            angular_count: 2,
        },
    )?;

    assert_eq!(handoff.energies_hartree, energies);
    assert_eq!(handoff.scattering_trace.dim(), (2, 2, 2));
    let nonzero_norm = handoff
        .scattering_trace
        .iter()
        .map(|value| value.norm())
        .sum::<f64>();
    assert!(nonzero_norm > 0.0);
    Ok(())
}

#[test]
fn pot_scf_fms_source_grid_keeps_each_independent_center_in_slot_zero() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&sample_fms_source_geom())?,
    )?;
    let pot = sample_pot_scf_fms_input(0, 3.0);
    let energies = Array1::from_vec(vec![Complex64::new(1.0, 0.02)]);
    let references = Array2::from_shape_fn((1, 2), |(_, potential)| {
        Complex64::new(0.05 * potential as f64, -energies[0].im)
    });
    let phase_shifts = Array3::from_shape_fn((1, 2, 2), |(_, angular, potential)| {
        Complex64::new(0.02 + 0.03 * angular as f64, -0.004 * potential as f64)
    });

    let handoff = build_pot_scf_fms_source_grid_handoff(
        temp.path(),
        PotScfFmsSourceGridInput {
            pot: &pot,
            energy_grid_hartree: energies.view(),
            reference_energies_hartree: references.view(),
            phase_shifts: phase_shifts.view(),
            angular_count: 2,
        },
    )?;

    for potential in 0..2 {
        let norm = handoff
            .scattering_trace
            .index_axis(Axis(2), potential)
            .iter()
            .map(|value| value.norm())
            .sum::<f64>();
        assert!(
            norm > f64::EPSILON,
            "independent central potential {potential} trace is empty"
        );
    }
    Ok(())
}

#[test]
fn pot_scf_fms_source_grid_zeros_single_atom_cluster_trace() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("geom.dat"),
        geom_dat_string(&sample_single_atom_fms_geom())?,
    )?;
    let mut pot = sample_pot_scf_fms_input(1, 3.0);
    pot.control.mpot = 0;
    pot.control.nph = 0;
    pot.potentials.truncate(1);
    let energies = Array1::from_vec(vec![Complex64::new(1.0, 0.02), Complex64::new(1.3, 0.03)]);
    let references = Array2::from_shape_fn((energies.len(), 1), |(energy, _)| {
        Complex64::new(0.0, -energies[energy].im)
    });
    let phase_shifts = Array3::from_shape_fn((energies.len(), 2, 1), |(energy, angular, _)| {
        Complex64::new(0.02 * (energy + 1) as f64 + 0.03 * angular as f64, 0.0)
    });

    let handoff = build_pot_scf_fms_source_grid_handoff(
        temp.path(),
        PotScfFmsSourceGridInput {
            pot: &pot,
            energy_grid_hartree: energies.view(),
            reference_energies_hartree: references.view(),
            phase_shifts: phase_shifts.view(),
            angular_count: 2,
        },
    )?;

    assert_eq!(handoff.energies_hartree, energies);
    assert_eq!(handoff.scattering_trace.dim(), (2, 2, 1));
    assert!(
        handoff
            .scattering_trace
            .iter()
            .all(|value| value.norm() <= f64::EPSILON)
    );
    Ok(())
}

#[test]
fn pot_scf_reciprocal_fms_honors_zero_radius_without_kspace_solve() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("reciprocal.inp"),
        "ispace\n\
   0\n\
lattice vectors\n\
      3.00000      0.00000      0.00000\n\
      0.00000      3.00000      0.00000\n\
      0.00000      0.00000      3.00000\n\
Volume scaling factor; eimag; core hole\n\
     -1.00000      0.00000      1.00000\n\
lattice type\n\
P      P1              1\n\
#atoms; absorber; corehole\n\
   1   1   0\n\
# k-points; ktype; symmetry\n\
           1           1           1           1           1           0\n\
ppos\n\
      0.00000      0.00000      0.00000\n\
ppot\n\
           1\n\
label\n\
Cu\n\
streta,strgmax,strrmax\n\
      0.00000      0.00000      0.00000\n",
    )?;
    let pot = sample_pot_scf_fms_input(1, 0.0);
    let energies = Array1::from_vec(vec![Complex64::new(1.0, 0.02)]);
    let references = Array2::zeros((1, 2));
    let phase_shifts = Array3::zeros((1, 2, 2));
    let handoff = build_pot_scf_fms_source_grid_handoff(
        temp.path(),
        PotScfFmsSourceGridInput {
            pot: &pot,
            energy_grid_hartree: energies.view(),
            reference_energies_hartree: references.view(),
            phase_shifts: phase_shifts.view(),
            angular_count: 2,
        },
    )?;
    assert!(
        handoff
            .scattering_trace
            .iter()
            .all(|value| *value == Complex64::new(0.0, 0.0))
    );
    Ok(())
}

#[test]
fn pot_scf_reciprocal_geometry_cache_reuses_and_fails_closed_on_mutation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("reciprocal.inp");
    std::fs::write(&path, b"snapshot one\n")?;
    let source_one = std::fs::read(&path)?;
    let cell = ReciprocalCell {
        lattice_vectors: [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        volume_scale: -1.0,
        imaginary_energy: 0.0,
        core_hole_strength: 1.0,
        lattice_name: "P".to_string(),
        space_group_hm: "P1".to_string(),
        space_group: 1,
        atom_count: 1,
        absorber: 1,
        core_hole: 0,
        k_mesh: ReciprocalKMesh {
            total: 1,
            x: 1,
            y: 1,
            z: 1,
            kind: 1,
            use_symmetry: false,
        },
        positions: vec![[0.0, 0.0, 0.0]],
        potentials: vec![1],
        labels: vec!["Cu".to_string()],
        stretch: [1.0, 0.0, 0.0],
    };

    let mut cache = PotScfFmsPipelineCache::default();
    let first = cache.reciprocal_static_setup(&path, &source_one, &cell, 0, 1, 2)?;
    let second = cache.reciprocal_static_setup(&path, &source_one, &cell, 0, 1, 2)?;
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.initial_ewald_tables.eta, first.kspace_lattice.eta);

    let key_error = cache
        .reciprocal_static_setup(&path, &source_one, &cell, 1, 1, 2)
        .expect_err("an active reciprocal geometry cache must reject key changes");
    assert!(key_error.to_string().contains("geometry key changed"));

    std::fs::write(&path, b"snapshot two\n")?;
    let source_two = std::fs::read(&path)?;
    let stale_error = cache
        .validate_reciprocal_snapshot(&path, Some(&source_two))
        .expect_err("an active reciprocal geometry cache must reject source mutation");
    assert!(
        stale_error
            .to_string()
            .contains("changed during active POT SCF pipeline")
    );

    let mut rebuilt_cache = PotScfFmsPipelineCache::default();
    let rebuilt = rebuilt_cache.reciprocal_static_setup(&path, &source_two, &cell, 0, 1, 2)?;
    assert!(!Arc::ptr_eq(&first, &rebuilt));
    assert_eq!(*first, *rebuilt);
    Ok(())
}

#[test]
fn fms_module_requires_spring_input_for_equation_of_motion_debye_generation() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_source_handoffs(temp.path(), 1, 0.0)?;

    assert!(!has_cached_fms_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("FMS idwopt=1 source generation should require spring.inp")?;

    assert!(
        error
            .to_string()
            .contains("FMS idwopt=1 source generation requires spring.inp")
    );
    assert!(!temp.path().join("gg.bin").exists());
    assert!(!temp.path().join("gg.dat").exists());
    Ok(())
}

#[test]
fn fms_module_generates_missing_gg_dat_from_gg_bin_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    let gg = sample_gg_dat();
    write_gg_bin(temp.path().join("gg.bin"), &gg)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 0.0);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_generates_missing_gg_bin_from_gg_dat_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    let gg = sample_gg_dat();
    write_gg_dat(temp.path().join("gg.dat"), &gg)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 0.0);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_recovers_malformed_gg_dat_from_gg_bin_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    let gg = sample_gg_dat();
    write_gg_bin(temp.path().join("gg.bin"), &gg)?;
    std::fs::write(temp.path().join("gg.dat"), b"not a gg.dat cache\n")?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 0.0);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_recovers_malformed_gg_bin_from_gg_dat_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fms_input(temp.path(), 1, -1)?;
    let gg = sample_gg_dat();
    write_gg_dat(temp.path().join("gg.dat"), &gg)?;
    std::fs::write(temp.path().join("gg.bin"), b"not a gg.bin cache\n")?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &gg, 0.0);
    assert_gg_values_close(&read_gg_bin(temp.path().join("gg.bin"))?, &gg, 0.0);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_generates_reference_gg_dat_from_gg_bin_cache() -> Result<()> {
    let Some(reference_dir) = reference_fms_dir()? else {
        crate::require_fixture!(
            "FMS gg companion reference test; generated EXAFS/Cu reference not found"
        );
    };
    if !reference_dir.join("gg.bin").is_file() || !reference_dir.join("gg.dat").is_file() {
        crate::require_fixture!("FMS gg companion reference test; reference gg caches not found");
    }

    let temp = tempfile::tempdir()?;
    for name in ["fms.inp", "gg.bin"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let expected = read_gg_dat(reference_dir.join("gg.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_gg_values_close(&read_gg_dat(temp.path().join("gg.dat"))?, &expected, 1.0e-8);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        generated_cached_fms_module_log(0)
    );
    Ok(())
}

#[test]
fn fms_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_fms_dir()? else {
        crate::require_fixture!("FMS reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
    for name in required {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    for name in ["gg.bin", "gtrl.dat", "fmsl.bin", "log3.dat"] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
        }
    }
    copy_gtr_bin_references(&reference_dir, temp.path())?;

    let expected_fms = read_fms_bin(temp.path().join("fms.bin"))?;
    let expected_gg_dat = read_gg_dat(temp.path().join("gg.dat"))?;
    let expected_gtr_dat = read_gtr_dat(temp.path().join("gtr.dat"))?;
    let expected_log = optional_module_log(temp.path().join("log3.dat"))?
        .unwrap_or_else(|| generated_cached_fms_module_log(0));

    let count = run_in_dir(temp.path())?;

    assert!(count >= required.len() - 1);
    assert_eq!(read_fms_bin(temp.path().join("fms.bin"))?, expected_fms);
    assert_eq!(read_gg_dat(temp.path().join("gg.dat"))?, expected_gg_dat);
    assert_eq!(read_gtr_dat(temp.path().join("gtr.dat"))?, expected_gtr_dat);
    assert_eq!(
        read_module_log_dat(temp.path().join("log3.dat"))?,
        expected_log
    );
    Ok(())
}

fn write_fms_input(work_dir: &Path, mfms: i32, decomposition_channels: i32) -> Result<()> {
    write_fms_input_with_lmax(work_dir, mfms, decomposition_channels, &[2, 2])
}

fn write_hubbard_input(work_dir: &Path, hubbard_l: i32) -> Result<()> {
    let input = HubbardInput {
        i_hubbard: 2,
        mldos_hubb: 2,
        u: 4.0,
        j: 0.5,
        fermi_shift: 0.0,
        l: hubbard_l,
    };
    std::fs::write(work_dir.join("hubbard.inp"), hubbard_input_string(&input)?)?;
    Ok(())
}

fn write_active_hubbard_v_source(
    work_dir: &Path,
    hubbard_l: usize,
    potential_count: usize,
) -> Result<()> {
    let angular_count = hubbard_l + 1;
    write_v_hubbard_bin(
        work_dir.join("v_hubbard.bin"),
        &HubbardVnlmBinData {
            angular_limit: hubbard_l,
            values: Array4::zeros((
                potential_count,
                2,
                angular_count,
                angular_count * angular_count,
            )),
        },
    )?;
    Ok(())
}

fn write_fms_input_with_lmax(
    work_dir: &Path,
    mfms: i32,
    decomposition_channels: i32,
    lmaxph: &[i32],
) -> Result<()> {
    let input = FmsInput {
        control: FmsControl {
            mfms,
            idwopt: 0,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: -1.0,
            rdirec: -1.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 190.0,
            thetad: 315.0,
            sig2g: 0.0,
        },
        lmaxph: lmaxph.to_vec(),
        decomposition_channels,
        save_gg_slice: false,
        do_fms: 0,
    };
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
    Ok(())
}

fn write_eels_input(work_dir: &Path, minimum: i32, step: i32, maximum: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("eels.inp"),
        format!(
            "calculate ELNES?\n   1\n\
average? relativistic? cross-terms? Which input?\n   0   1   1   1   4\n\
polarizations to be used ; min step max\n{minimum:4}{step:4}{maximum:4}\n\
beam energy in eV\n 300000.00000\n\
beam direction in arbitrary units\n      0.00000      1.00000      0.00000\n\
collection and convergence semiangle in rad\n      0.00240      0.00000\n\
qmesh - radial and angular grid size\n   5   3\n\
detector positions - two angles in rad\n      0.00000      0.00000\n\
calculate magic angle if magic=1\n   0\n\
energy for magic angle - eV above threshold\n      0.00000\n"
        ),
    )?;
    Ok(())
}

fn write_fms_source_handoffs(
    work_dir: &Path,
    idwopt: i32,
    sig2g: f64,
) -> Result<(GlobalInput, PhaseBinData)> {
    write_fms_source_input(work_dir, idwopt, sig2g)?;
    let global = sample_global_input();
    std::fs::write(work_dir.join("global.inp"), global_input_string(&global)?)?;
    let phase = sample_fms_source_phase_bin();
    write_phase_bin(work_dir.join("phase.bin"), &phase)?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_fms_source_geom())?,
    )?;
    Ok((global, phase))
}

fn write_fms_nrixs_source_handoffs(work_dir: &Path) -> Result<PhaseBinData> {
    let mut input = FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt: -1,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 3.0,
            rdirec: 5.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: 0.0,
            thetad: 0.0,
            sig2g: 0.0,
        },
        lmaxph: vec![1, 1],
        decomposition_channels: 2,
        save_gg_slice: false,
        do_fms: 0,
    };
    // Keep the explicit binding mutable so future NRIXS solver-branch
    // regressions can vary `minv` without rebuilding the fixture.
    input.control.minv = 0;
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;

    let mut global = sample_global_input();
    global.control.ipol = 1;
    global.control.le2 = 1;
    global.control.elpty = 0.0;
    global.control.l2lp = 4;
    global.control.do_nrixs = 1;
    global.control.ldecmx = 2;
    global.q_control = GlobalQControl {
        nq: 1,
        imdff: 0,
        qaverage: false,
        mixdff: false,
    };
    global.q_vectors = vec![GlobalQVector {
        q: [0.0, 0.0, 2.0],
        norm: 2.0,
        weight: [1.0, 0.0],
        trig: [-1.0, 0.0, 1.0, 0.0],
    }];
    std::fs::write(work_dir.join("global.inp"), global_input_string(&global)?)?;

    let mut phase = sample_fms_source_phase_bin();
    let core_hole = core_hole_quantum_numbers(phase.ihole)?;
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: core_hole.kappa,
        multipole: global.control.le2,
        max_angular_momentum: 24,
    })?;
    phase.final_state_count = indices.final_state_capacity;
    phase.transition_count = indices.transitions.len();
    phase.q_count = 1;
    phase.transition_moments = Array4::from_shape_fn(
        (
            phase.energy_count,
            phase.q_count,
            phase.transition_count,
            phase.spin_count,
        )
            .f(),
        |(energy, q, transition, spin)| {
            Complex64::new(
                0.2 + 0.03 * energy as f64 + 0.02 * transition as f64,
                -0.01 * (q + transition + spin) as f64,
            )
        },
    );
    write_phase_bin(work_dir.join("phase.bin"), &phase)?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_fms_source_geom())?,
    )?;
    Ok(phase)
}

fn write_fms_dmdw_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        "   1\n   1\n   1    190.000\n   0\nfeff.dym\n   0\n",
    )?;
    std::fs::write(work_dir.join("feff.dym"), sample_fms_dmdw_dym())?;
    Ok(())
}

fn write_fms_dmdw_source_handoffs(
    work_dir: &Path,
    idwopt: i32,
) -> Result<(GlobalInput, PhaseBinData)> {
    let handoffs = write_fms_source_handoffs(work_dir, idwopt, 0.0)?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_fms_dmdw_geom())?,
    )?;
    Ok(handoffs)
}

fn sample_ldos_fmsdos_input_for_phase(phase: &PhaseBinData) -> LdosInput {
    let first_energy = phase.energy_grid[0];
    let last_energy = phase.energy_grid[phase.energy_grid.len() - 1];
    LdosInput {
        control: LdosControl {
            mldos: 1,
            lfms2: 1,
            ixc: 0,
            ispin: 0,
            minv: 0,
            neldos: phase.energy_grid.len() as i32,
            iscfxc: 11,
        },
        mesh: LdosMesh {
            rfms2: 3.0,
            emin: first_energy.re * FEFF_HARTREE_EV,
            emax: last_energy.re * FEFF_HARTREE_EV,
            eimag: first_energy.im * FEFF_HARTREE_EV,
            rgrd: 0.05,
        },
        fms: LdosFms {
            rdirec: 5.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        lmaxph: vec![1, 1],
        ldostype: 0,
    }
}

fn write_fms_spring_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("spring.inp"),
        "VDOS 0.03 0.5 1.0 2.5\nSTRETCHES\n0 1 27.9 2.0\n1 2 12.0 2.0\nEND\n",
    )?;
    Ok(())
}

fn write_fms_spring_source_handoffs(
    work_dir: &Path,
    idwopt: i32,
) -> Result<(GlobalInput, PhaseBinData)> {
    write_fms_spring_source_input(work_dir, idwopt)?;
    let global = sample_global_input();
    std::fs::write(work_dir.join("global.inp"), global_input_string(&global)?)?;
    let phase = sample_fms_spring_phase_bin();
    write_phase_bin(work_dir.join("phase.bin"), &phase)?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_fms_spring_geom())?,
    )?;
    Ok((global, phase))
}

fn write_fms_spring_source_input(work_dir: &Path, idwopt: i32) -> Result<()> {
    let input = FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt,
            minv: 0,
        },
        cluster: FmsCluster {
            rfms2: 5.0,
            rdirec: 5.0,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: if idwopt < 0 { 0.0 } else { 190.0 },
            thetad: if idwopt < 0 { 0.0 } else { 315.0 },
            sig2g: 0.0,
        },
        lmaxph: vec![1, 1, 1],
        decomposition_channels: -1,
        save_gg_slice: false,
        do_fms: 0,
    };
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
    Ok(())
}

fn write_fms_source_input(work_dir: &Path, idwopt: i32, sig2g: f64) -> Result<()> {
    write_fms_source_input_with_cluster(work_dir, idwopt, sig2g, 3.0, 5.0)
}

fn write_fms_source_input_with_cluster(
    work_dir: &Path,
    idwopt: i32,
    sig2g: f64,
    rfms2: f64,
    rdirec: f64,
) -> Result<()> {
    write_fms_source_input_with_options(work_dir, idwopt, sig2g, rfms2, rdirec, false)
}

fn write_fms_source_input_with_options(
    work_dir: &Path,
    idwopt: i32,
    sig2g: f64,
    rfms2: f64,
    rdirec: f64,
    save_gg_slice: bool,
) -> Result<()> {
    write_fms_source_input_with_do_fms_options(
        work_dir,
        idwopt,
        sig2g,
        rfms2,
        rdirec,
        save_gg_slice,
        0,
    )
}

fn write_fms_source_input_with_do_fms_options(
    work_dir: &Path,
    idwopt: i32,
    sig2g: f64,
    rfms2: f64,
    rdirec: f64,
    save_gg_slice: bool,
    do_fms: i32,
) -> Result<()> {
    write_fms_source_input_with_solver_options(
        work_dir,
        idwopt,
        sig2g,
        rfms2,
        rdirec,
        save_gg_slice,
        do_fms,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_fms_source_input_with_solver_options(
    work_dir: &Path,
    idwopt: i32,
    sig2g: f64,
    rfms2: f64,
    rdirec: f64,
    save_gg_slice: bool,
    do_fms: i32,
    minv: i32,
) -> Result<()> {
    let input = FmsInput {
        control: FmsControl {
            mfms: 1,
            idwopt,
            minv,
        },
        cluster: FmsCluster {
            rfms2,
            rdirec,
            toler1: 0.001,
            toler2: 0.001,
        },
        debye: FmsDebye {
            tk: if idwopt < 0 { 0.0 } else { 190.0 },
            thetad: if idwopt < 0 { 0.0 } else { 315.0 },
            sig2g,
        },
        lmaxph: vec![1, 1],
        decomposition_channels: -1,
        save_gg_slice,
        do_fms,
    };
    std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
    Ok(())
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

fn sample_pot_scf_fms_input(lfms1: i32, rfms1: f64) -> PotInput {
    PotInput {
        control: PotControl {
            mpot: 1,
            nph: 1,
            ntitle: 0,
            ihole: 1,
            ipr1: 0,
            iafolp: 0,
            ixc: 0,
            ispec: 0,
            iscfxc: 0,
        },
        run: PotRun {
            nmix: 1,
            nohole: 0,
            jumprm: 0,
            inters: 0,
            nscmt: 1,
            icoul: 0,
            lfms1,
            iunf: 0,
        },
        titles: Vec::new(),
        scattering: PotScattering {
            gamach: 0.0,
            rgrd: 0.05,
            ca1: 0.0,
            ecv: 0.0,
            totvol: 0.0,
            rfms1,
            corval_emin: 0.0,
        },
        potentials: vec![
            PotPotential {
                z: 29,
                lmaxsc: 1,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
            PotPotential {
                z: 29,
                lmaxsc: 1,
                xnatph: 1.0,
                xion: 0.0,
                folp: 1.0,
            },
        ],
        external_pot: false,
        start_from_file: false,
        overlap_shells: vec![Vec::new(), Vec::new()],
        chsh_type: 0,
        config_type: 0,
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

fn sample_fms_source_geom() -> GeomDat {
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

fn sample_single_atom_fms_geom() -> GeomDat {
    GeomDat {
        nat: 1,
        nph: 0,
        model_atoms: vec![1],
        atoms: vec![GeomDatRow {
            index: 1,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            iph: 0,
            boundary: 0,
        }],
    }
}

fn sample_phase_bin() -> PhaseBinData {
    let energy_count = 2;
    let spin_count = 1;
    let transition_count = 8;
    let energy_grid = Array1::from_vec(vec![Complex64::new(1.0, 0.1), Complex64::new(2.0, 0.2)]);
    let reference_energy = Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
        Complex64::new(0.01 * (energy + 1) as f64, -0.02 * spin as f64)
    });
    let phase_shifts =
        Array3::from_shape_fn((energy_count, 3, spin_count), |(energy, angular, spin)| {
            Complex64::new(
                0.1 * (energy + 1) as f64 + 0.01 * angular as f64,
                -0.005 * spin as f64,
            )
        });
    let mut transition_moments =
        Array4::<Complex64>::zeros((energy_count, 1, transition_count, spin_count).f());
    for energy in 0..energy_count {
        for transition in 0..transition_count {
            transition_moments[(energy, 0, transition, 0)] = Complex64::new(
                0.25 + 0.1 * energy as f64 + 0.03 * transition as f64,
                -0.02 * transition as f64,
            );
        }
    }

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
        potentials: vec![PhaseBinPotential {
            lmax: 1,
            atomic_number: 29,
            label: "Cu".to_string(),
            phase_shifts,
        }],
        transition_moments,
        raw_pads: None,
    }
}

fn sample_hubbard_transformation_bin(potential_count: usize) -> HubbardTransformationBinData {
    let mut next = 1.0_f32;
    let mut transform = Array5::from_elem((potential_count, 2, 2, 3, 3), Complex32::new(0.0, 0.0));
    let mut inverse = Array5::from_elem((potential_count, 2, 2, 3, 3), Complex32::new(0.0, 0.0));
    for potential in 0..potential_count {
        for angular in 0..2 {
            for spin in 0..2 {
                for column in 0..3 {
                    for row in 0..3 {
                        transform[(potential, spin, angular, row, column)] =
                            Complex32::new(next, -next);
                        let inverse_value = 1000.0 + next;
                        inverse[(potential, spin, angular, row, column)] =
                            Complex32::new(inverse_value, -inverse_value);
                        next += 1.0;
                    }
                }
            }
        }
    }
    HubbardTransformationBinData {
        hubbard_l: 1,
        angular_limit: 1,
        transform,
        inverse,
    }
}

fn sample_active_aphase_hubbard_bin(phase: &PhaseBinData) -> HubbardAphaseBinData {
    sample_active_aphase_hubbard_bin_with_limit(phase, 1)
}

fn sample_active_aphase_hubbard_bin_with_limit(
    phase: &PhaseBinData,
    angular_limit: usize,
) -> HubbardAphaseBinData {
    let angular_count = angular_limit + 1;
    let magnetic_count = angular_count * angular_count;
    let values = Array5::from_shape_fn(
        (
            phase.potential_count(),
            2,
            phase.energy_count,
            angular_count,
            magnetic_count,
        ),
        |(potential, spin, energy, angular, magnetic)| {
            let scale = 0.02 * (potential + 1) as f64
                + 0.01 * energy as f64
                + 0.005 * angular as f64
                + 0.001 * magnetic as f64
                + 0.0005 * spin as f64;
            Complex64::new(scale, 0.25 * scale)
        },
    );
    HubbardAphaseBinData {
        angular_limit,
        values,
    }
}

fn sample_active_hubbard_transformation_bin(
    potential_count: usize,
) -> HubbardTransformationBinData {
    sample_active_hubbard_transformation_bin_with_limit(potential_count, 1)
}

fn sample_active_hubbard_transformation_bin_with_limit(
    potential_count: usize,
    angular_limit: usize,
) -> HubbardTransformationBinData {
    let hubbard_l = 1;
    let angular_count = angular_limit + 1;
    let block = 2 * hubbard_l + 1;
    let mut transform = Array5::from_elem(
        (potential_count, 2, angular_count, block, block),
        Complex32::new(0.0, 0.0),
    );
    let mut inverse = transform.clone();
    for potential in 0..potential_count {
        for spin in 0..2 {
            for angular in 0..angular_count {
                for row in 0..block {
                    let column = block - 1 - row;
                    transform[(potential, spin, angular, row, column)] = Complex32::new(1.0, 0.0);
                    inverse[(potential, spin, angular, row, column)] = Complex32::new(1.0, 0.0);
                }
            }
        }
    }
    HubbardTransformationBinData {
        hubbard_l,
        angular_limit,
        transform,
        inverse,
    }
}

fn sample_fms_dmdw_geom() -> GeomDat {
    GeomDat {
        nat: 3,
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
            GeomDatRow {
                index: 3,
                x: 0.2,
                y: 1.1,
                z: 0.8,
                iph: 1,
                boundary: 0,
            },
        ],
    }
}

fn sample_fms_spring_geom() -> GeomDat {
    GeomDat {
        nat: 3,
        nph: 2,
        model_atoms: vec![1, 2, 3],
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
                x: 2.0,
                y: 0.0,
                z: 0.0,
                iph: 1,
                boundary: 0,
            },
            GeomDatRow {
                index: 3,
                x: 3.8,
                y: 0.0,
                z: 0.0,
                iph: 2,
                boundary: 0,
            },
        ],
    }
}

fn sample_fms_spring_phase_bin() -> PhaseBinData {
    let mut phase = sample_fms_source_phase_bin();
    phase.potentials[0].atomic_number = 29;
    phase.potentials[0].label = "Cu".to_string();
    phase.potentials[1].atomic_number = 30;
    phase.potentials[1].label = "Zn".to_string();
    let mut third = phase.potentials[1].clone();
    third.atomic_number = 31;
    third.label = "Ga".to_string();
    phase.potentials.push(third);
    phase
}

fn sample_fms_dmdw_dym() -> String {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.4 / FEFF_BOHR_ANGSTROM, 0.0, 0.0],
        [
            0.2 / FEFF_BOHR_ANGSTROM,
            1.1 / FEFF_BOHR_ANGSTROM,
            0.8 / FEFF_BOHR_ANGSTROM,
        ],
    ];
    let mut out = String::new();
    let _ = writeln!(out, "    1");
    let _ = writeln!(out, "    3");
    for _ in positions {
        let _ = writeln!(out, "   29");
    }
    for _ in positions {
        let _ = writeln!(out, "   63.546000");
    }
    for position in positions {
        let _ = writeln!(
            out,
            "{:14.8}{:14.8}{:14.8}",
            position[0], position[1], position[2]
        );
    }
    for first in 0..positions.len() {
        for second in 0..positions.len() {
            let diagonal = if first == second { 4.0 } else { 0.0 };
            let _ = writeln!(out, "{:5}{:5}", first + 1, second + 1);
            for row in 0..3 {
                let _ = writeln!(
                    out,
                    " {:13.6E} {:13.6E} {:13.6E}",
                    if row == 0 { diagonal } else { 0.0 },
                    if row == 1 { diagonal } else { 0.0 },
                    if row == 2 { diagonal } else { 0.0 }
                );
            }
        }
    }
    out
}

fn sample_fms_source_phase_bin() -> PhaseBinData {
    let energy_count = 2;
    let spin_count = 1;
    let transition_count = 8;
    let energy_grid = Array1::from_vec(vec![Complex64::new(1.0, 0.02), Complex64::new(1.4, 0.03)]);
    let reference_energy = Array2::zeros((energy_count, spin_count));
    let potentials = (0..2)
        .map(|potential| PhaseBinPotential {
            lmax: 2,
            atomic_number: 29,
            label: "Cu".to_string(),
            phase_shifts: Array3::from_shape_fn(
                (energy_count, 5, spin_count),
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

fn sample_two_spin_fms_source_phase_bin() -> PhaseBinData {
    let mut phase = sample_fms_source_phase_bin();
    let spin_count = 2;
    phase.spin_count = spin_count;
    phase.reference_energy =
        Array2::from_shape_fn((phase.energy_count, spin_count), |(energy, spin)| {
            Complex64::new(0.01 * energy as f64, -0.002 * spin as f64)
        });
    for potential in &mut phase.potentials {
        let base = potential.phase_shifts.clone();
        let signed_l_count = base.shape()[1];
        potential.phase_shifts = Array3::from_shape_fn(
            (phase.energy_count, signed_l_count, spin_count),
            |(energy, signed_l, spin)| {
                base[(energy, signed_l, 0)] + Complex64::new(0.001 * spin as f64, 0.0)
            },
        );
    }
    let base_moments = phase.transition_moments.clone();
    phase.transition_moments = Array4::from_shape_fn(
        (
            phase.energy_count,
            phase.q_count,
            phase.transition_count,
            spin_count,
        )
            .f(),
        |(energy, q, transition, spin)| {
            base_moments[(energy, q, transition, 0)]
                + Complex64::new(0.002 * spin as f64, -0.001 * spin as f64)
        },
    );
    phase
}

fn sample_mkgtr_gg() -> GgDatData {
    GgDatData {
        sections: (0..2)
            .map(|energy| GgDatSection {
                section_number: energy + 1,
                values: Array2::from_shape_fn((4, 4), |(row, column)| {
                    let base =
                        0.15 + 0.2 * energy as f64 + 0.03 * row as f64 + 0.01 * column as f64;
                    Complex64::new(base, -0.5 * base)
                }),
                raw_prefix_lines: None,
            })
            .collect(),
    }
}

fn expected_mkgtr_trace(
    global: &GlobalInput,
    phase: &PhaseBinData,
    gg: &GgDatData,
    lmax: usize,
) -> Result<Array2<Complex64>> {
    let core_hole = core_hole_quantum_numbers(phase.ihole)?;
    let transition_matrix = transition_b_matrix(TransitionBMatrixInput {
        lmax,
        initial_kappa: core_hole.kappa,
        polarization: global.control.ipol,
        polarization_tensor: super::polarization_tensor(global),
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channels: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })?;
    let green_functions = super::green_functions_from_gg(gg, phase.energy_count)?;
    let transition_moments = phase.transition_moments.index_axis(Axis(1), 0);
    Ok(mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 1,
        green_functions: green_functions.view(),
        transition_matrices: &[transition_matrix],
        transition_moments,
    })?
    .traces)
}

fn assert_complex_table_close(
    actual: ndarray::ArrayView2<'_, Complex64>,
    expected: ndarray::ArrayView2<'_, Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "complex table mismatch at {index}: actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_complex_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "complex array mismatch at {index}: actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_complex_vec_close(
    actual: ndarray::ArrayView1<'_, Complex64>,
    expected: ndarray::ArrayView1<'_, Complex64>,
    tolerance: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "complex vector mismatch at {index}: actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_saved_scattering_absorber_block_matches_gg(
    gg: &GgDatData,
    slice: &RhorrpGgSliceBinData,
    diag: &RhorrpGgDiagBinData,
    tolerance: f64,
) {
    assert_eq!(slice.energy_count(), gg.section_count());
    assert_eq!(diag.energy_count(), gg.section_count());
    assert!(diag.atom_count() >= 1);
    for (energy, section) in gg.sections.iter().enumerate() {
        assert_eq!(section.shape(), (diag.row_count(), diag.column_count()));
        assert!(slice.column_count() >= section.column_count());
        for ((row, column), expected) in section.values.indexed_iter() {
            let slice_value = slice.values[(energy, row, column)];
            let diag_value = diag.values[(energy, 0, row, column)];
            assert_complex64_32_close(
                *expected,
                slice_value,
                tolerance,
                "gg_slice absorber block",
                energy,
                row,
                column,
            );
            assert_complex64_32_close(
                *expected,
                diag_value,
                tolerance,
                "gg_diag absorber block",
                energy,
                row,
                column,
            );
        }
    }
}

fn assert_complex64_32_close(
    expected: Complex64,
    actual: Complex32,
    tolerance: f64,
    context: &str,
    energy: usize,
    row: usize,
    column: usize,
) {
    let actual = Complex64::new(actual.re as f64, actual.im as f64);
    assert!(
        (actual - expected).norm() <= tolerance,
        "{context} mismatch at energy {} row {} column {}: actual={actual:?} expected={expected:?}",
        energy + 1,
        row,
        column
    );
}

fn assert_gg_values_close(actual: &GgDatData, expected: &GgDatData, tolerance: f64) {
    assert_eq!(actual.sections.len(), expected.sections.len());
    for (section_index, (actual, expected)) in actual
        .sections
        .iter()
        .zip(expected.sections.iter())
        .enumerate()
    {
        assert_eq!(actual.section_number, expected.section_number);
        assert_eq!(actual.values.dim(), expected.values.dim());
        for ((row, column), actual_value) in actual.values.indexed_iter() {
            let expected_value = expected.values[(row, column)];
            assert!(
                (actual_value.re - expected_value.re).abs() <= tolerance,
                "gg section {} row {} column {} real differs: actual={} expected={}",
                section_index + 1,
                row,
                column,
                actual_value.re,
                expected_value.re
            );
            assert!(
                (actual_value.im - expected_value.im).abs() <= tolerance,
                "gg section {} row {} column {} imaginary differs: actual={} expected={}",
                section_index + 1,
                row,
                column,
                actual_value.im,
                expected_value.im
            );
        }
    }
}

fn assert_gg_values_differ(actual: &GgDatData, expected: &GgDatData, tolerance: f64) {
    assert_eq!(actual.sections.len(), expected.sections.len());
    let mut total_delta = 0.0;
    for (actual, expected) in actual.sections.iter().zip(expected.sections.iter()) {
        assert_eq!(actual.section_number, expected.section_number);
        assert_eq!(actual.values.dim(), expected.values.dim());
        total_delta += actual
            .values
            .iter()
            .zip(expected.values.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .sum::<f64>();
    }
    assert!(
        total_delta > tolerance,
        "gg tables did not differ enough: total delta {total_delta} <= {tolerance}"
    );
}

fn sample_fms_bin() -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 2,
        main_energy_count: 1,
        auxiliary_energy_count: 0,
        highest_potential_index: 1,
        pad_width: 8,
        declared_spectrum_count: Some(2),
        spectra: Array2::from_shape_fn((2, 2), |(spectrum, energy)| {
            Complex64::new(
                0.25 * (energy + 1) as f64 + spectrum as f64,
                -0.05 * (energy + 1) as f64 - spectrum as f64,
            )
        }),
    }
}

fn sample_fmsl_bin() -> FmslBinData {
    FmslBinData {
        pad_width: 8,
        max_decomposition_channel: 2,
        traces: Array3::from_shape_fn((2, 3, 3), |(energy, lg2, lg1)| {
            Complex64::new(
                energy as f64 + 0.1 * lg2 as f64 + 0.01 * lg1 as f64,
                -(energy as f64) - 0.2 * lg2 as f64 - 0.02 * lg1 as f64,
            )
        }),
    }
}

fn sample_gg_dat() -> GgDatData {
    GgDatData {
        sections: vec![
            GgDatSection {
                section_number: 1,
                values: Array2::from_shape_fn((2, 2), |(row, column)| {
                    let value = 1.0 + row as f64 + 2.0 * column as f64;
                    Complex64::new(value, -0.5 * value)
                }),
                raw_prefix_lines: None,
            },
            GgDatSection {
                section_number: 2,
                values: Array2::from_shape_fn((1, 2), |(_, column)| {
                    let value = 5.0 + column as f64;
                    Complex64::new(value, -value - 0.5)
                }),
                raw_prefix_lines: None,
            },
        ],
    }
}

fn sample_gtr_dat() -> GtrDatData {
    GtrDatData {
        energy: Array1::from_vec(vec![
            Complex64::new(-0.138_801, 0.031_773),
            Complex64::new(-0.137_401, 0.031_773),
            Complex64::new(55.866_911, 0.031_773),
        ]),
        trace: Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.624_106, 1.081_113),
        ]),
    }
}

fn sample_gtr_bin() -> GtrBinData {
    GtrBinData {
        point_count_declared: 2,
        horizontal_count: 1,
        danes_extension_count: 0,
        highest_potential_index: 1,
        fms_mode: 2,
        values: Array3::from_shape_fn((2, 2, 2), |(energy, potential, angular)| {
            let value = energy as f64 + 0.1 * potential as f64 + 0.01 * angular as f64;
            Complex64::new(value, -value)
        }),
    }
}

fn sample_gtrl_dat() -> Result<GtrlDatData> {
    Ok(parse_gtrl_dat(
        r#"    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01
    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01
"#,
    )?)
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "FMS calculation of full Green's function ...".to_string(),
            "Done with module: FMS.".to_string(),
            "MKGTR: Tracing over Green's function ...".to_string(),
            "Done with module: MKGTR.".to_string(),
        ],
        line_terminators: vec![
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
            "\n".to_string(),
        ],
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

fn copy_gtr_bin_references(source_dir: &Path, target_dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(source_dir)
        .with_context(|| format!("failed to read {}", source_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", source_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if super::is_gtr_bin_name(name) {
            std::fs::copy(entry.path(), target_dir.join(name))?;
        }
    }
    Ok(())
}

fn reference_fms_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["fms.inp", "fms.bin", "gg.dat", "gtr.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_hubbard_full_potential_source_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let parent = workspace.join("reference-work/tmp");
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
    ];
    for entry in std::fs::read_dir(&parent)
        .with_context(|| format!("failed to read {}", parent.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        if name
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
