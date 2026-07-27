use super::{has_cached_optical_inputs, run_in_dir};
use anyhow::Result;
use ndarray::array;
use num_complex::Complex64;
use refeff_core::{FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV};
use refeff_io::{
    DrudeDatData, EpsDatData, FullSpectrumInput, HamakerDatData, ModuleLogData, OscStrDatData,
    OscStrRow, fullspectrum_input_string, read_drude_dat, read_eps_dat, read_hamaker_dat,
    read_module_log_dat, read_opcons_dat, read_osc_str_dat, read_xmu_dat, write_drude_dat,
    write_eps_dat, write_hamaker_dat, write_module_log_dat, write_osc_str_dat,
};

use crate::run_supported_cached_modules;

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

fn write_source_fullspectrum_input(
    path: &std::path::Path,
    edge_mode: &str,
    drude: bool,
) -> Result<()> {
    let mut text = fullspectrum_input_string(&FullSpectrumInput { m_full_spectrum: 1 })?;
    text.push_str("EGRID 1.0 20.0 80\n");
    if drude {
        text.push_str("DRUDE 1.0e-15 1.5\n");
    }
    text.push_str("COMPONENT Cu 29 0.050 EDGES\n");
    text.push_str(&format!("K {edge_mode}\n"));
    std::fs::write(path.join("fullspectrum.inp"), text)?;
    Ok(())
}

fn disable_fullspectrum_optical_outputs(path: &std::path::Path) -> Result<()> {
    let input_path = path.join("fullspectrum.inp");
    let mut text = std::fs::read_to_string(&input_path)?;
    text.push_str("CONTROL 1 1 1 1 1 0\n");
    std::fs::write(input_path, text)?;
    Ok(())
}

fn write_fprime_source(edge_path: &std::path::Path) -> Result<()> {
    let path = edge_path.join("fprime1");
    std::fs::create_dir_all(&path)?;
    std::fs::write(
        path.join("xmu.dat"),
        r#"# FEFF FPRIME xmu.dat
# omega e f' f' f'' f''
  0.010  0.010 -2.00000E+00 -2.00000E+00 0.00000E+00 0.00000E+00
 10.000 10.000 -1.50000E+00 -1.50000E+00 3.00000E-01 3.00000E-01
 30.000 30.000 -1.00000E+00 -1.00000E+00 5.00000E-01 5.00000E-01
"#,
    )?;
    Ok(())
}

fn write_fine_structure_source(edge_path: &std::path::Path, branch: &str) -> Result<()> {
    let path = edge_path.join(branch);
    std::fs::create_dir_all(&path)?;
    let imaginary = branch.ends_with("_im");
    let header = if imaginary {
        "# xsedge+ 50, used to normalize mu 1.0000E+00\n"
    } else {
        ""
    };
    std::fs::write(
        path.join("xmu.dat"),
        format!(
            "# FEFF {branch} xmu.dat\n{header}# omega e k mu mu0 chi\n\
               1.000  0.000  1.000  2.00000E-01 1.50000E-01 5.00000E-02\n\
              10.000  9.000  3.000  3.00000E-01 2.00000E-01 1.00000E-01\n\
              15.000 14.000  4.000  4.00000E-01 2.50000E-01 1.50000E-01\n\
              30.000 29.000  6.000  5.00000E-01 3.00000E-01 2.00000E-01\n"
        ),
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
fn generates_eps_and_optical_tables_from_background_edge_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_source_fullspectrum_input(temp.path(), "BACKGROUND", false)?;
    let edge_path = temp.path().join("edges").join("Cu").join("K");
    write_fprime_source(&edge_path)?;

    assert!(has_cached_optical_inputs(temp.path())?);
    let point_count = run_in_dir(temp.path())?;

    let eps = read_eps_dat(temp.path().join("eps.dat"))?;
    let oscillator_strength = read_osc_str_dat(temp.path().join("osc_str.dat"))?;
    assert_eq!(point_count, eps.point_count());
    assert!(point_count > 10);
    assert_eq!(oscillator_strength.rows.len(), 1);
    assert_eq!(oscillator_strength.rows[0].edge, "K");
    assert!(eps.epsilon.iter().any(|value| value.im > 0.0));
    assert_eq!(
        read_opcons_dat(temp.path().join("opcons.dat"))?.point_count(),
        point_count
    );

    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), point_count);
    assert_eq!(xmu.normalization, Some(1.0));
    assert_eq!(&xmu.header_lines[..2], ["# component  edge  n_eff", " "]);
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line.starts_with("#         Cu     K")),
        "{:?}",
        xmu.header_lines
    );
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "#     0/   0 paths used")
    );
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "#  xsedge+ 50, used to normalize mu           1.0000E+00")
    );
    for row in 0..point_count {
        let expected_energy = eps.omega[row] * FEFF_HARTREE_EV;
        let expected_wave_number = (2.0 * eps.omega[row]).sqrt() / FEFF_BOHR_ANGSTROM;
        assert!((xmu.photon_energy_ev[row] - expected_energy).abs() < 1.0e-8);
        assert!((xmu.relative_energy_ev[row] - expected_energy).abs() < 1.0e-8);
        assert!((xmu.wave_number[row] - expected_wave_number).abs() < 1.0e-8);
        assert!((xmu.mu[row] - eps.sigma[row]).abs() < 1.0e-10);
        assert!((xmu.mu0[row] - eps.sigma[row]).abs() < 1.0e-10);
        assert_eq!(xmu.chi[row], 0.0);
    }
    Ok(())
}

#[test]
fn control_six_skips_optical_outputs_but_keeps_source_spectrum_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_source_fullspectrum_input(temp.path(), "BACKGROUND", true)?;
    disable_fullspectrum_optical_outputs(temp.path())?;
    let edge_path = temp.path().join("edges").join("Cu").join("K");
    write_fprime_source(&edge_path)?;

    assert!(has_cached_optical_inputs(temp.path())?);
    assert_eq!(run_in_dir(temp.path())?, 0);

    assert!(temp.path().join("eps.dat").is_file());
    assert!(temp.path().join("osc_str.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    for name in [
        "drude.dat",
        "opcons.dat",
        "opconsKK.dat",
        "opcons0.dat",
        "sumrules.dat",
        "hamaker.dat",
    ] {
        assert!(
            !temp.path().join(name).exists(),
            "CONTROL(6)=0 unexpectedly wrote {name}"
        );
    }
    Ok(())
}

#[test]
fn control_six_leaves_stale_optical_outputs_untouched_and_unreported() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_source_fullspectrum_input(temp.path(), "BACKGROUND", true)?;
    disable_fullspectrum_optical_outputs(temp.path())?;
    let edge_path = temp.path().join("edges").join("Cu").join("K");
    write_fprime_source(&edge_path)?;

    let stale_names = [
        "drude.dat",
        "opcons.dat",
        "opconsKK.dat",
        "opcons0.dat",
        "sumrules.dat",
        "hamaker.dat",
    ];
    for name in stale_names {
        std::fs::write(temp.path().join(name), format!("stale {name}\n"))?;
    }

    let reports = run_supported_cached_modules(temp.path())?;

    assert!(
        reports.iter().all(|report| report.name != "fullspectrum"),
        "CONTROL(6)=0 must not advertise stale optical outputs: {reports:?}"
    );
    assert!(temp.path().join("eps.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    for name in stale_names {
        assert_eq!(
            std::fs::read_to_string(temp.path().join(name))?,
            format!("stale {name}\n"),
            "CONTROL(6)=0 changed stale {name}"
        );
    }
    Ok(())
}

#[test]
fn control_six_does_not_advertise_eps_only_restart_as_optical_work() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_fullspectrum_input(temp.path(), 1)?;
    disable_fullspectrum_optical_outputs(temp.path())?;
    write_eps_dat(temp.path().join("eps.dat"), &sample_eps_dat())?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    assert_eq!(run_in_dir(temp.path())?, 0);
    assert!(!temp.path().join("xmu.dat").exists());
    assert_no_fullspectrum_outputs(temp.path());
    Ok(())
}

#[test]
fn generates_detailed_eps_and_drude_from_all_edge_source_branches() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_source_fullspectrum_input(temp.path(), "DETAIL", true)?;
    let edge_path = temp.path().join("edges").join("Cu").join("K");
    write_fprime_source(&edge_path)?;
    for branch in ["fms_re", "path_re", "fms_im", "path_im"] {
        write_fine_structure_source(&edge_path, branch)?;
    }

    let point_count = run_in_dir(temp.path())?;
    let eps = read_eps_dat(temp.path().join("eps.dat"))?;
    let drude = read_drude_dat(temp.path().join("drude.dat"))?;

    assert_eq!(eps.point_count(), point_count);
    assert_eq!(drude.point_count(), point_count);
    assert!(drude.plasma_frequency_ev > 0.0);
    assert!(
        eps.epsilon
            .iter()
            .zip(eps.background_epsilon.iter())
            .any(|(total, background)| (*total - *background).norm() > 1.0e-12)
    );
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
fn fullspectrum_module_does_not_claim_malformed_edge_sources() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_source_fullspectrum_input(temp.path(), "BACKGROUND", false)?;
    let source = temp
        .path()
        .join("edges")
        .join("Cu")
        .join("K")
        .join("fprime1");
    std::fs::create_dir_all(&source)?;
    std::fs::write(source.join("xmu.dat"), b"not an xmu.dat source\n")?;

    assert!(!has_cached_optical_inputs(temp.path())?);
    let error = run_in_dir(temp.path())
        .expect_err("malformed FULLSPECTRUM edge source should fail through explicit run");
    let chain = format!("{error:?}");
    assert!(chain.contains("fprime1"), "{chain}");
    assert!(chain.contains("xmu.dat"), "{chain}");
    assert_no_fullspectrum_outputs(temp.path());
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
    assert!(!temp.path().join("xmu.dat").exists());
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
    assert!(
        !temp.path().join("xmu.dat").exists(),
        "an eps.dat restart has no successful edge assembly to report in FULLSPECTRUM xmu.dat"
    );
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
