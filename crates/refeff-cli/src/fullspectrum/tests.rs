use super::{has_cached_optical_inputs, run_in_dir};
use anyhow::Result;
use ndarray::array;
use num_complex::Complex64;
use refeff_io::{
    DrudeDatData, EpsDatData, FullSpectrumInput, HamakerDatData, ModuleLogData, OscStrDatData,
    OscStrRow, fullspectrum_input_string, read_drude_dat, read_hamaker_dat, read_module_log_dat,
    read_opcons_dat, read_osc_str_dat, write_drude_dat, write_eps_dat, write_hamaker_dat,
    write_module_log_dat, write_osc_str_dat,
};

fn sample_eps_dat() -> EpsDatData {
    EpsDatData {
        header_lines: vec!["# sample eps.dat".to_string()],
        omega: array![1.0, 2.0, 4.0, 7.0],
        epsilon: array![
            Complex64::new(0.2, 0.05),
            Complex64::new(0.4, 0.12),
            Complex64::new(0.1, 0.07),
            Complex64::new(0.3, 0.03),
        ],
        background_epsilon: array![
            Complex64::new(0.1, 0.02),
            Complex64::new(0.2, 0.04),
            Complex64::new(0.05, 0.025),
            Complex64::new(0.15, 0.01),
        ],
        sigma: array![0.01, 0.02, 0.03, 0.04],
    }
}

fn write_fullspectrum_input(path: &std::path::Path, flag: i32) -> Result<()> {
    std::fs::write(
        path.join("fullspectrum.inp"),
        fullspectrum_input_string(&FullSpectrumInput {
            m_full_spectrum: flag,
        })?,
    )?;
    Ok(())
}

fn sample_osc_str_dat() -> OscStrDatData {
    OscStrDatData {
        header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
        rows: vec![OscStrRow {
            component: "Cu".to_string(),
            edge: "K".to_string(),
            core_hole_index: 1,
            effective_electron_count: 5.123,
        }],
    }
}

fn sample_hamaker_dat() -> HamakerDatData {
    HamakerDatData {
        header_lines: vec!["# cached hamaker transform".to_string()],
        omega: array![1.0, 2.0, 4.0],
        imaginary_axis_epsilon: array![
            Complex64::new(0.35, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(0.10, 0.0),
        ],
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating full spectrum optical constants ...".to_string(),
            "Done with module: FULLSPECTRUM.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn assert_no_fullspectrum_outputs(path: &std::path::Path) {
    assert!(!path.join("opcons.dat").exists());
    assert!(!path.join("opconsKK.dat").exists());
    assert!(!path.join("opcons0.dat").exists());
    assert!(!path.join("sumrules.dat").exists());
}

#[test]
fn detects_cached_optical_inputs_only_when_enabled_and_complete() -> Result<()> {
    let temp = tempfile::tempdir()?;
    assert!(!has_cached_optical_inputs(temp.path())?);

    write_fullspectrum_input(temp.path(), 0)?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;
    assert!(!has_cached_optical_inputs(temp.path())?);

    write_fullspectrum_input(temp.path(), 1)?;
    assert!(has_cached_optical_inputs(temp.path())?);
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_malformed_eps_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("eps.dat"), b"not an eps.dat cache\n")?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_cached_output_with_malformed_drude_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;
    std::fs::write(temp.path().join("drude.dat"), b"not a drude.dat sidecar\n")?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .expect_err("malformed FULLSPECTRUM drude.dat should fail through explicit run");
    let chain = format!("{error:?}");

    assert!(chain.contains("drude.dat"), "{chain}");
    assert_no_fullspectrum_outputs(temp.path());
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_cached_output_with_malformed_osc_str_sidecar() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;
    std::fs::write(
        temp.path().join("osc_str.dat"),
        b"not an osc_str.dat sidecar\n",
    )?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .expect_err("malformed FULLSPECTRUM osc_str.dat should fail through explicit run");
    let chain = format!("{error:?}");

    assert!(chain.contains("osc_str.dat"), "{chain}");
    assert_no_fullspectrum_outputs(temp.path());
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_cached_output_with_malformed_pot_sumrules_source()
-> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;
    std::fs::write(temp.path().join("pot.bin"), b"not a pot.bin source\n")?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .expect_err("malformed FULLSPECTRUM pot.bin source should fail through explicit run");
    let chain = format!("{error:?}");

    assert!(chain.contains("pot.bin"), "{chain}");
    assert_no_fullspectrum_outputs(temp.path());
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("fullspectrum.inp"),
        b"not a fullspectrum.inp handoff\n",
    )?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .expect_err("malformed FULLSPECTRUM input should fail through explicit run");
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("fullspectrum.inp"), "{chain}");
    assert_no_fullspectrum_outputs(temp.path());
    Ok(())
}

#[test]
fn fullspectrum_module_does_not_claim_orphan_eps_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    assert_no_fullspectrum_outputs(temp.path());
    assert!(!temp.path().join("logfullspectrum.dat").exists());
    Ok(())
}

#[test]
fn skips_disabled_fullspectrum_without_reading_eps_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 0)?;

    assert_eq!(run_in_dir(temp.path())?, 0);
    assert!(!temp.path().join("opcons.dat").exists());
    Ok(())
}

#[test]
fn adds_cached_drude_term_to_fullspectrum_optical_tables() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let eps = sample_eps_dat();
    let drude = DrudeDatData {
        gamma_ev: 0.5,
        plasma_frequency_ev: 2.0,
        omega: eps.omega.clone(),
        epsilon: array![
            Complex64::new(-0.01, 0.005),
            Complex64::new(-0.02, 0.006),
            Complex64::new(-0.03, 0.007),
            Complex64::new(-0.04, 0.008),
        ],
    };
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &eps)?;
    write_drude_dat(temp.path().join("drude.dat"), &drude)?;
    let expected_drude = read_drude_dat(temp.path().join("drude.dat"))?;

    assert_eq!(run_in_dir(temp.path())?, eps.point_count());

    assert_eq!(
        read_drude_dat(temp.path().join("drude.dat"))?,
        expected_drude
    );
    let opcons = read_opcons_dat(temp.path().join("opcons.dat"))?;
    let opcons_kk = read_opcons_dat(temp.path().join("opconsKK.dat"))?;
    let opcons0 = read_opcons_dat(temp.path().join("opcons0.dat"))?;
    for ((actual, bound), free) in opcons
        .epsilon_minus_one
        .iter()
        .zip(eps.epsilon.iter())
        .zip(drude.epsilon.iter())
    {
        assert!((actual.re - (bound.re + free.re)).abs() < 1.0e-10);
        assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
    }
    for ((actual, bound), free) in opcons_kk
        .epsilon_minus_one
        .iter()
        .zip(eps.epsilon.iter())
        .zip(drude.epsilon.iter())
    {
        assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
    }
    for ((actual, bound), free) in opcons0
        .epsilon_minus_one
        .iter()
        .zip(eps.background_epsilon.iter())
        .zip(drude.epsilon.iter())
    {
        assert!((actual.im - (bound.im + free.im)).abs() < 1.0e-10);
    }
    Ok(())
}

#[test]
fn preserves_cached_fullspectrum_sidecars() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let eps = sample_eps_dat();
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &eps)?;
    write_osc_str_dat(temp.path().join("osc_str.dat"), &sample_osc_str_dat())?;
    write_hamaker_dat(temp.path().join("hamaker.dat"), &sample_hamaker_dat())?;
    write_module_log_dat(
        temp.path().join("logfullspectrum.dat"),
        &sample_module_log(),
    )?;
    let expected_osc_str = read_osc_str_dat(temp.path().join("osc_str.dat"))?;
    let expected_hamaker = read_hamaker_dat(temp.path().join("hamaker.dat"))?;
    let expected_log = read_module_log_dat(temp.path().join("logfullspectrum.dat"))?;

    assert_eq!(run_in_dir(temp.path())?, eps.point_count());

    assert_eq!(
        read_osc_str_dat(temp.path().join("osc_str.dat"))?,
        expected_osc_str
    );
    assert_eq!(
        read_hamaker_dat(temp.path().join("hamaker.dat"))?,
        expected_hamaker
    );
    assert_eq!(
        read_module_log_dat(temp.path().join("logfullspectrum.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn writes_cached_fullspectrum_optical_tables_from_eps_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let eps = sample_eps_dat();
    write_fullspectrum_input(temp.path(), 1)?;
    write_eps_dat(temp.path().join("eps.dat"), &eps)?;

    assert_eq!(run_in_dir(temp.path())?, eps.point_count());

    let opcons = read_opcons_dat(temp.path().join("opcons.dat"))?;
    let opcons_kk = read_opcons_dat(temp.path().join("opconsKK.dat"))?;
    let opcons0 = read_opcons_dat(temp.path().join("opcons0.dat"))?;

    assert_eq!(opcons.point_count(), eps.point_count());
    assert_eq!(opcons_kk.point_count(), eps.point_count());
    assert_eq!(opcons0.point_count(), eps.point_count());
    assert!(!temp.path().join("sumrules.dat").exists());
    assert!(!temp.path().join("hamaker.dat").exists());
    for (actual, expected) in opcons.epsilon_minus_one.iter().zip(eps.epsilon.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-10);
        assert!((actual.im - expected.im).abs() < 1.0e-10);
    }
    for (actual, expected) in opcons_kk.epsilon_minus_one.iter().zip(eps.epsilon.iter()) {
        assert!((actual.im - expected.im).abs() < 1.0e-10);
    }
    for (actual, expected) in opcons0
        .epsilon_minus_one
        .iter()
        .zip(eps.background_epsilon.iter())
    {
        assert!((actual.im - expected.im).abs() < 1.0e-10);
    }
    Ok(())
}
