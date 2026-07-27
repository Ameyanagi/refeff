use super::{
    has_cached_self_output, has_supported_self_source_handoff, has_supported_sfconv_source_handoff,
    read_input, run_for_input, run_in_dir, run_self_in_dir, so2conv_existing_target_data_for_dir,
    so2conv_specfunct_cache_preflight_for_dir, so2conv_targets_for_dir,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, array};
use refeff_core::{
    SfconvSo2convMaterialInput, make_excitation_poles, sfconv_plasmon_threshold_momentum,
    sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
};
use refeff_io::{
    ExcDatData, ListDatData, ListDatEntry, SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvSpecfunctData,
    exc_dat_from_excitation_poles, parse_exc_dat, parse_xmu_dat, read_exc_dat, read_specfunct_dat,
    read_xmu_dat, sfconv_apl_dat_string, sfconv_rdeps_fallback_poles, write_exc_dat,
    write_list_dat, write_specfunct_dat,
};
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn sfconv_module_writes_empty_log_when_disabled() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 0)?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    assert!(!has_supported_self_source_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        ""
    );
    Ok(())
}

#[test]
fn sfconv_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_sfconv_dir()? else {
        crate::require_fixture!("SFCONV reference test; generated XANES/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    std::fs::copy(
        reference_dir.join("sfconv.inp"),
        temp.path().join("sfconv.inp"),
    )?;
    std::fs::copy(
        reference_dir.join("logsfconv.dat"),
        temp.path().join("logsfconv.dat"),
    )?;
    let expected_log = std::fs::read(temp.path().join("logsfconv.dat"))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert_eq!(
        std::fs::read(temp.path().join("logsfconv.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn sfconv_module_uses_input_parent_directory() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 0)?;

    let count = run_for_input(&temp.path().join("feff.inp"))?;

    assert_eq!(count, 0);
    assert!(temp.path().join("logsfconv.dat").is_file());
    Ok(())
}

#[test]
fn sfconv_module_skips_missing_targets_like_feff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!temp.path().join("exc.dat").exists());
    assert!(!temp.path().join("apl.dat").exists());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_so2conv_log()
    );
    Ok(())
}

#[test]
fn sfconv_module_does_not_claim_malformed_target_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("xmu.dat"), "not an xmu.dat target\n")?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed SFCONV target should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(
        chain.contains("failed to parse SO2CONV target data"),
        "{chain}"
    );
    assert!(chain.contains("xmu.dat"), "{chain}");
    assert!(!temp.path().join("specfunct.dat").exists());
    assert!(!temp.path().join("apl.dat").exists());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        "Calculating S0^2 ...\n"
    );
    Ok(())
}

#[test]
fn sfconv_module_rejects_numeric_xmu_without_material_header() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;
    let path = temp.path().join("xmu.dat");
    let without_material_header = std::fs::read_to_string(&path)?
        .lines()
        .filter(|line| !line.contains("Gam_ch="))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&path, format!("{without_material_header}\n"))?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("SFCONV must reject numeric xmu.dat without its material header")?;
    let chain = format!("{error:?}");
    assert!(
        chain.contains("missing SO2CONV header field Gam_ch"),
        "{chain}"
    );
    assert!(!temp.path().join("specfunct.dat").exists());
    assert!(!temp.path().join("apl.dat").exists());
    Ok(())
}

#[test]
fn sfconv_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    std::fs::write(
        temp.path().join("sfconv.inp"),
        b"not an sfconv.inp handoff\n",
    )?;
    write_xmu_header(temp.path(), false)?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);

    let sfconv_error = run_in_dir(temp.path())
        .err()
        .context("malformed SFCONV input should fail through explicit run")?;
    let sfconv_chain = format!("{sfconv_error:?}");
    assert!(sfconv_chain.contains("failed to parse"), "{sfconv_chain}");
    assert!(sfconv_chain.contains("sfconv.inp"), "{sfconv_chain}");

    let self_error = run_self_in_dir(temp.path())
        .err()
        .context("malformed SELF input should fail through explicit run")?;
    let self_chain = format!("{self_error:?}");
    assert!(self_chain.contains("failed to parse"), "{self_chain}");
    assert!(self_chain.contains("sfconv.inp"), "{self_chain}");

    assert!(!temp.path().join("logsfconv.dat").exists());
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, sample_exc_dat());
    Ok(())
}

#[test]
fn sfconv_module_does_not_claim_orphan_outputs_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_xmu_header(temp.path(), false)?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;

    assert!(!has_supported_sfconv_source_handoff(temp.path())?);
    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, sample_exc_dat());
    assert!(!temp.path().join("logsfconv.dat").exists());
    Ok(())
}

#[test]
fn sfconv_module_generates_specfunct_cache_without_reusable_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_so2conv_log()
    );
    let rendered = std::fs::read_to_string(temp.path().join("xmu.dat"))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    let cache = read_specfunct_dat(temp.path().join("specfunct.dat"))?;
    assert_eq!(cache.asymmetric_phase, 1);
    assert_eq!(cache.satellite_type, 0);
    assert_eq!(cache.low_q_mode, 0);
    assert_eq!(cache.pole_count, 1);
    assert_eq!(cache.pole_capacity(), super::SO2CONV_POLE_CAPACITY);
    assert_eq!(cache.momentum_count(), 66);
    assert_eq!(cache.spectral_point_count(), 112);
    assert!(temp.path().join("exc.dat").is_file());
    assert!(temp.path().join("apl.dat").is_file());
    Ok(())
}

#[test]
fn sfconv_module_applies_compatible_specfunct_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 1)?;

    assert!(has_supported_sfconv_source_handoff(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_so2conv_log()
    );
    let rendered = std::fs::read_to_string(temp.path().join("xmu.dat"))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    assert!(temp.path().join("exc.dat").is_file());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("apl.dat"))?,
        expected_apl_dat()?
    );
    Ok(())
}

#[test]
fn sfconv_module_applies_compatible_exafs_specfunct_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_chi_header(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 0)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_so2conv_log()
    );
    let rendered = std::fs::read_to_string(temp.path().join("chi.dat"))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    assert!(temp.path().join("exc.dat").is_file());
    assert!(temp.path().join("apl.dat").is_file());
    Ok(())
}

#[test]
fn sfconv_module_applies_compatible_feff_path_specfunct_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input_with_spectrum(temp.path(), 1, 0, 3)?;
    write_single_path_list(temp.path())?;
    write_feff_path_target(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 0)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_so2conv_log()
    );
    let rendered = std::fs::read_to_string(temp.path().join("feff0001.dat"))?;
    assert_eq!(
        rendered.lines().next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    assert!(temp.path().join("exc.dat").is_file());
    assert!(temp.path().join("apl.dat").is_file());
    Ok(())
}

#[test]
fn sfconv_module_detects_incompatible_specfunct_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 0)?;

    let input = read_input(temp.path())?;
    let targets = so2conv_targets_for_dir(temp.path(), &input)?;
    let target_data = so2conv_existing_target_data_for_dir(temp.path(), &targets)?;
    let cache = so2conv_specfunct_cache_preflight_for_dir(temp.path(), &target_data)?
        .context("expected specfunct.dat cache")?;

    assert!(cache.compatible_targets.is_empty());
    assert_eq!(cache.incompatible_targets, vec!["xmu.dat"]);
    assert!(temp.path().join("exc.dat").is_file());
    assert!(temp.path().join("apl.dat").is_file());
    Ok(())
}

#[test]
fn sfconv_module_accepts_feff_reference_specfunct_cache_when_present() -> Result<()> {
    let Some(zip_path) = reference_so2conv_zip()? else {
        crate::require_fixture!("SO2CONV reference cache test; Cu_OPCONS REFERENCE.zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        crate::require_fixture!("SO2CONV reference cache test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    let golden_dir = zip_path
        .parent()
        .context("Cu_OPCONS REFERENCE.zip has no parent directory")?;
    std::fs::copy(golden_dir.join("ff2x.inp"), temp.path().join("ff2x.inp"))?;
    for name in ["xsect.dat", "feff.bin", "fms.bin", "list.dat", "global.inp"] {
        std::fs::write(
            temp.path().join(name),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
        )?;
    }
    crate::ff2x::run_in_dir(temp.path())
        .context("failed to regenerate the exact unconvoluted Cu_OPCONS xmu.dat handoff")?;
    assert_ne!(
        std::fs::read_to_string(temp.path().join("xmu.dat"))?
            .lines()
            .next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );

    for name in ["sfconv.inp", "exc.dat", "specfunct.dat"] {
        std::fs::write(
            temp.path().join(name),
            unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
        )?;
    }
    let expected_specfunct = std::fs::read(temp.path().join("specfunct.dat"))?;
    let expected_log =
        String::from_utf8(unzip_reference_entry(&zip_path, "REFERENCE/logsfconv.dat")?)?;
    let expected_xmu = parse_xmu_dat(&String::from_utf8(unzip_reference_entry(
        &zip_path,
        "REFERENCE/xmu.dat",
    )?)?)?;

    let input = read_input(temp.path())?;
    let targets = so2conv_targets_for_dir(temp.path(), &input)?;
    let target_data = so2conv_existing_target_data_for_dir(temp.path(), &targets)?;
    let cache = so2conv_specfunct_cache_preflight_for_dir(temp.path(), &target_data)?
        .context("expected FEFF reference specfunct.dat cache")?;

    assert_eq!(cache.compatible_targets, vec!["xmu.dat"]);
    assert!(cache.incompatible_targets.is_empty());
    assert_eq!(cache.data.pole_count, 86);
    assert_eq!(cache.data.pole_capacity(), 5000);
    assert_eq!(cache.data.momentum_count(), 66);
    assert_eq!(cache.data.spectral_point_count(), 112);
    let (photoelectron_momentum, _) =
        super::so2conv_photoelectron_momentum_for_target(&cache.data, &target_data[0].data)?;
    assert!(
        (photoelectron_momentum[0] - 1.049_080_168_7).abs() <= 5.0e-7,
        "Cu_OPCONS first SO2CONV photoelectron momentum {}",
        photoelectron_momentum[0]
    );
    assert!(
        (photoelectron_momentum[1] - 1.082_029_393_4).abs() <= 5.0e-7,
        "Cu_OPCONS second SO2CONV photoelectron momentum {}",
        photoelectron_momentum[1]
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("apl.dat"))?,
        String::from_utf8(unzip_reference_entry(&zip_path, "REFERENCE/apl.dat")?)?,
    );

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
    assert_eq!(
        std::fs::read(temp.path().join("specfunct.dat"))?,
        expected_specfunct
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        expected_log
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("xmu.dat"))?
            .lines()
            .next(),
        Some(SFCONV_SO2CONV_CONVOLUTED_MARKER)
    );
    let actual_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    for (name, actual, expected, tolerance) in [
        ("mu", actual_xmu.mu.view(), expected_xmu.mu.view(), 5.0e-5),
        (
            "mu0",
            actual_xmu.mu0.view(),
            expected_xmu.mu0.view(),
            5.0e-5,
        ),
        (
            "chi",
            actual_xmu.chi.view(),
            expected_xmu.chi.view(),
            1.0e-4,
        ),
    ] {
        let relative_l2 = relative_l2(actual, expected);
        assert!(
            relative_l2 <= tolerance,
            "Cu_OPCONS FEFF handoff roundtrip {name} relative L2 {relative_l2:.6e} exceeds {tolerance:.1e}"
        );
    }
    Ok(())
}

#[test]
fn sfconv_module_rejects_already_convoluted_target_like_feff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), true)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("previously convoluted target should stop SO2CONV")?;

    assert!(error.to_string().contains("has already been convoluted"));
    assert!(error.to_string().contains("xmu.dat"));
    Ok(())
}

#[test]
fn self_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 0)?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    let count = run_self_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_self_output(temp.path())?);
    Ok(())
}

#[test]
fn self_module_generates_exc_dat_from_loss_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 0, 4, 12.0)?;
    write_loss_dat(temp.path())?;

    assert!(has_supported_self_source_handoff(temp.path())?);
    let loss_energy = sample_loss_energy();
    let loss = sample_loss();
    let expected = exc_dat_from_excitation_poles(&make_excitation_poles(
        loss_energy.view(),
        loss.view(),
        12.0,
        4,
    )?)?;

    let count = run_self_in_dir(temp.path())?;

    assert_eq!(count, expected.pole_count());
    assert!(has_cached_self_output(temp.path())?);
    assert_exc_dat_close(&read_exc_dat(temp.path().join("exc.dat"))?, &expected);
    Ok(())
}

#[test]
fn self_module_regenerates_stale_exc_dat_from_loss_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 0, 4, 12.0)?;
    write_loss_dat(temp.path())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let stale = read_exc_dat(temp.path().join("exc.dat"))?;
    let expected = exc_dat_from_excitation_poles(&make_excitation_poles(
        sample_loss_energy().view(),
        sample_loss().view(),
        12.0,
        4,
    )?)?;

    let count = run_self_in_dir(temp.path())?;

    assert_eq!(count, expected.pole_count());
    let actual = read_exc_dat(temp.path().join("exc.dat"))?;
    assert_ne!(actual, stale);
    assert_exc_dat_close(&actual, &expected);
    Ok(())
}

#[test]
fn self_module_recovers_malformed_exc_dat_from_loss_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 0, 4, 12.0)?;
    write_loss_dat(temp.path())?;
    std::fs::write(temp.path().join("exc.dat"), "not exc.dat\n")?;

    assert!(has_cached_self_output(temp.path())?);

    let count = run_self_in_dir(temp.path())?;

    let actual = read_exc_dat(temp.path().join("exc.dat"))?;
    assert_eq!(count, actual.pole_count());
    assert_eq!(actual.pole_count(), 4);
    assert!(actual.has_auxiliary_weight());
    Ok(())
}

#[test]
fn self_module_does_not_advertise_malformed_exc_dat_without_source() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    std::fs::write(temp.path().join("exc.dat"), "not exc.dat\n")?;

    assert!(!has_cached_self_output(temp.path())?);

    let error = run_self_in_dir(temp.path())
        .err()
        .context("malformed standalone SELF cache should fail")?;
    assert!(format!("{error:#}").contains("exc.dat"));
    Ok(())
}

#[test]
fn self_module_does_not_claim_malformed_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    std::fs::write(temp.path().join("xsph.inp"), "not an xsph.inp handoff\n")?;
    write_loss_dat(temp.path())?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);

    let error = run_self_in_dir(temp.path())
        .err()
        .context("malformed SELF source should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("xsph.inp"), "{chain}");
    assert!(!temp.path().join("exc.dat").exists());
    Ok(())
}

#[test]
fn self_module_does_not_claim_cached_output_with_malformed_xsph_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 0, 4, 12.0)?;
    write_loss_dat(temp.path())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(temp.path().join("exc.dat"))?;
    std::fs::write(temp.path().join("xsph.inp"), "not an xsph.inp handoff\n")?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);
    let error = run_self_in_dir(temp.path())
        .err()
        .context("malformed SELF xsph source should block cached SELF completion")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("xsph.inp"), "{chain}");
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, expected);
    Ok(())
}

#[test]
fn self_module_does_not_claim_cached_output_with_malformed_loss_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 0, 4, 12.0)?;
    write_loss_dat(temp.path())?;
    write_exc_dat(temp.path().join("exc.dat"), &sample_exc_dat())?;
    let expected = read_exc_dat(temp.path().join("exc.dat"))?;
    std::fs::write(temp.path().join("loss.dat"), "not a loss.dat source\n")?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);
    let error = run_self_in_dir(temp.path())
        .err()
        .context("malformed SELF loss source should block cached SELF completion")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("loss.dat"), "{chain}");
    assert_eq!(read_exc_dat(temp.path().join("exc.dat"))?, expected);
    Ok(())
}

#[test]
fn self_module_does_not_treat_unsupported_loss_source_as_cached_output() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    write_xsph_input(temp.path(), 1, 1, 4, 12.0)?;
    write_loss_dat(temp.path())?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    assert!(!has_cached_self_output(temp.path())?);

    let error = run_self_in_dir(temp.path())
        .err()
        .context("unsupported SELF source should fail during generation")?;

    assert!(
        error
            .to_string()
            .contains("requires Hedin-Lundqvist exchange"),
        "{error:#}"
    );
    assert!(!temp.path().join("exc.dat").exists());
    Ok(())
}

#[test]
fn self_module_generates_feff_reference_exc_dat_from_loss_dat() -> Result<()> {
    let Some(reference_dir) = reference_self_dir()? else {
        crate::require_fixture!("SELF reference test; MPSE/Cu reference directory not found");
    };
    let Some(zip_path) = reference_self_zip()? else {
        crate::require_fixture!("SELF reference test; MPSE/Cu REFERENCE.zip not found");
    };
    if Command::new("unzip").arg("-v").output().is_err() {
        crate::require_fixture!("SELF reference test; unzip command not found");
    }

    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    std::fs::copy(reference_dir.join("xsph.inp"), temp.path().join("xsph.inp"))?;
    std::fs::copy(reference_dir.join("loss.dat"), temp.path().join("loss.dat"))?;
    let expected = parse_exc_dat(&String::from_utf8(unzip_reference_entry(
        &zip_path,
        "REFERENCE/exc.dat",
    )?)?)?;

    let count = run_self_in_dir(temp.path())?;

    let actual = read_exc_dat(temp.path().join("exc.dat"))?;
    assert_eq!(count, expected.pole_count());
    assert_exc_dat_close(&actual, &expected);
    Ok(())
}

#[test]
fn self_module_roundtrips_cached_exc_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    let path = temp.path().join("exc.dat");
    write_exc_dat(&path, &sample_exc_dat())?;
    let expected = read_exc_dat(&path)?;

    assert!(!has_supported_self_source_handoff(temp.path())?);
    let count = run_self_in_dir(temp.path())?;

    assert_eq!(count, expected.pole_count());
    assert!(has_cached_self_output(temp.path())?);
    assert_eq!(read_exc_dat(&path)?, expected);
    Ok(())
}

fn write_sfconv_input(work_dir: &Path, msfconv: i32) -> Result<()> {
    write_sfconv_input_with_control_and_spectrum(work_dir, msfconv, 0, 0, 0)
}

fn write_self_input(work_dir: &Path) -> Result<()> {
    write_sfconv_input_with_control_and_spectrum(work_dir, 0, 1, 0, 0)
}

fn write_sfconv_input_with_spectrum(
    work_dir: &Path,
    msfconv: i32,
    ispec: i32,
    ipr6: i32,
) -> Result<()> {
    write_sfconv_input_with_control_and_spectrum(work_dir, msfconv, 0, ispec, ipr6)
}

fn write_sfconv_input_with_control_and_spectrum(
    work_dir: &Path,
    msfconv: i32,
    ipse: i32,
    ispec: i32,
    ipr6: i32,
) -> Result<()> {
    std::fs::write(
        work_dir.join("sfconv.inp"),
        format!(
            concat!(
                "msfconv, ipse, ipsk\n",
                "{:4}{:4}{:4}\n",
                "wsigk, cen\n",
                "{:13.5}{:13.5}\n",
                "ispec, ipr6\n",
                "{:4}{:4}\n",
                "cfname\n",
                "NULL        \n",
            ),
            msfconv, ipse, 0, 0.0, 0.0, ispec, ipr6
        ),
    )?;
    Ok(())
}

fn write_xsph_input(
    work_dir: &Path,
    i_plsmn: i32,
    ixc: i32,
    n_poles: i32,
    eps0: f64,
) -> Result<()> {
    std::fs::write(
        work_dir.join("xsph.inp"),
        format!(
            concat!(
                "mphase,ipr2,ixc,ixc0,ispec,lreal,lfms2,nph,l2lp,iPlsmn,NPoles,iGammaCH,iGrid,iCoreState,iscfxc\n",
                "{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}{:4}\n",
                "vr0, vi0\n",
                "{:13.5}{:13.5}\n",
                " lmaxph(0:nph)\n",
                "{:4}\n",
                " potlbl(iph)\n",
                "Cu    \n",
                "rgrd, rfms2, gamach, xkstep, xkmax, vixan, Eps0, EGap\n",
                "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}\n",
                "spinph(0:nph)\n",
                "{:13.5}\n",
                "izstd, ifxc, ipmbse, itdlda, nonlocal, ibasis\n",
                "{:4}{:4}{:4}{:4}{:4}{:4}\n",
                "electronic temperature\n",
                "{:13.5}\n",
                "ChSh_Type:\n",
                "{:4}\n",
                " the number of decomposition channels ; only used for nrixs\n",
                "{:5}\n",
                "lopt\n",
                " F\n",
                "PrintRL\n",
                " F\n",
            ),
            1,
            0,
            ixc,
            ixc,
            1,
            0,
            0,
            0,
            0,
            i_plsmn,
            n_poles,
            0,
            0,
            -1,
            0,
            0.0,
            0.0,
            3,
            0.05,
            6.0,
            1.729,
            0.07,
            8.0,
            0.0,
            eps0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            -1
        ),
    )?;
    Ok(())
}

fn write_loss_dat(work_dir: &Path) -> Result<()> {
    let mut text = String::new();
    for (&energy, &loss) in sample_loss_energy().iter().zip(sample_loss().iter()) {
        text.push_str(&format!("{energy:12.6E} {loss:12.6E}\n"));
    }
    std::fs::write(work_dir.join("loss.dat"), text)?;
    Ok(())
}

fn sample_exc_dat() -> ExcDatData {
    ExcDatData {
        header_lines: vec!["# SELF excitation poles".to_string()],
        energy_ev: Array1::from_vec(vec![15.0, 27.5]),
        broadening_ev: Array1::from_vec(vec![0.15, 0.275]),
        oscillator_strength: Array1::from_vec(vec![0.75, 0.25]),
        auxiliary_weight: Some(Array1::from_vec(vec![1.0, 0.5])),
    }
}

fn sample_loss_energy() -> Array1<f64> {
    array![5.0, 12.0, 25.0, 60.0, 120.0, 250.0, 500.0]
}

fn sample_loss() -> Array1<f64> {
    array![0.18, 0.45, 0.32, 0.20, 0.11, 0.05, 0.02]
}

fn write_xmu_header(work_dir: &Path, already_convoluted: bool) -> Result<()> {
    let marker = if already_convoluted {
        "# Convoluted with A(omega).\n"
    } else {
        ""
    };
    let mut text = format!(
        concat!(
            "{}",
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
        ),
        marker
    );
    for row in 0..24 {
        let row = row as f64;
        text.push_str(&format!(
            "  {:10.4} {:10.4} {:10.4} {:13.6E} {:13.6E} {:13.6E}\n",
            100.0 + 5.0 * row,
            1.0 + 5.0 * row,
            0.20 + 0.02 * row,
            1.0 + 0.01 * row,
            0.80 + 0.005 * row,
            0.20 + 0.005 * row
        ));
    }
    std::fs::write(work_dir.join("xmu.dat"), text)?;
    Ok(())
}

fn write_chi_header(work_dir: &Path, already_convoluted: bool) -> Result<()> {
    let marker = if already_convoluted {
        "# Convoluted with A(omega).\n"
    } else {
        ""
    };
    let mut text = format!(
        concat!(
            "{}",
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
        ),
        marker
    );
    for row in 0..24 {
        let row = row as f64;
        text.push_str(&format!(
            "  {:10.4} {:13.6E} {:13.6E} {:13.6E}\n",
            0.20 + 0.02 * row,
            0.01 * row,
            1.0 + 0.02 * row,
            0.10 + 0.03 * row
        ));
    }
    std::fs::write(work_dir.join("chi.dat"), text)?;
    Ok(())
}

fn write_single_path_list(work_dir: &Path) -> Result<()> {
    write_list_dat(
        work_dir.join("list.dat"),
        &ListDatData {
            titles: Vec::new(),
            entries: vec![ListDatEntry {
                path_index: 1,
                sigma2: 0.0,
                amplitude_ratio: 1.0,
                degeneracy: 2.0,
                leg_count: 4,
                effective_half_path_length_angstrom: 2.5,
            }],
        },
    )?;
    Ok(())
}

fn write_feff_path_target(work_dir: &Path, already_convoluted: bool) -> Result<()> {
    let marker = if already_convoluted {
        "# Convoluted with A(omega).\n"
    } else {
        ""
    };
    let mut text = format!(
        concat!(
            "{}",
            "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
            "Mu= 18.76000 kf= 1.230000\n",
            " ------------------------------------------------------------------------------\n",
            "#    4   2.000   2.5000 reff path metadata\n",
            "#       k          phase @#\n",
        ),
        marker
    );
    for row in 0..=400 {
        let row = row as f64;
        let wave_number = 0.05 * row;
        text.push_str(&format!(
            "  {:6.3} {:11.4E} {:11.4E} {:11.4E} {:10.3E} {:11.4E} {:11.4E}\n",
            wave_number,
            0.10 + 0.001 * row,
            1.00 + 0.001 * row,
            0.20 + 0.001 * row,
            0.90 - 0.0001 * row,
            8.00 + 0.010 * row,
            wave_number
        ));
    }
    std::fs::write(work_dir.join("feff0001.dat"), text)?;
    Ok(())
}

fn write_specfunct_cache(work_dir: &Path, asymmetric_phase: i32) -> Result<()> {
    let material = xmu_header_material();
    let parameters = sfconv_so2conv_material_parameters(material)?;
    let threshold_momentum = sfconv_plasmon_threshold_momentum(
        parameters.plasma_frequency,
        parameters.dispersion_parameter,
        parameters.fermi_energy,
        parameters.fermi_momentum,
    )?;
    let momentum_grid =
        sfconv_so2conv_momentum_grid(parameters.fermi_momentum, threshold_momentum)?;
    let momentum_count = momentum_grid.len();
    let spectral_point_count = 2;
    let mut spectral_info = Array2::zeros((momentum_count, 8));
    for (row, &momentum) in momentum_grid.iter().enumerate() {
        spectral_info[[row, 0]] = momentum;
    }
    let spectral_table =
        Array2::from_shape_fn((momentum_count, spectral_point_count), |(_, column)| {
            column as f64
        });
    let mut weights = Array2::zeros((momentum_count, 8));
    for row in 0..momentum_count {
        weights[[row, 0]] = 1.0;
    }

    write_specfunct_dat(
        work_dir.join("specfunct.dat"),
        &SfconvSpecfunctData {
            wigner_seitz_radius: material.wigner_seitz_radius,
            core_hole_lifetime: parameters.core_hole_lifetime,
            asymmetric_phase,
            satellite_type: 0,
            low_q_mode: 0,
            pole_count: 1,
            pole_energy: array![parameters.plasma_frequency],
            pole_broadening: array![0.001 * parameters.plasma_frequency],
            pole_weight: array![1.0],
            spectral_info,
            weights,
            extrinsic_quasiparticle: spectral_table.clone(),
            extrinsic_satellite: spectral_table.clone(),
            interference_quasiparticle: spectral_table.clone(),
            interference_satellite: spectral_table.clone(),
            intrinsic_satellite: spectral_table.clone(),
            clipped_extrinsic_satellite: spectral_table.clone(),
            energy_grid: spectral_table,
        },
    )?;
    Ok(())
}

fn xmu_header_material() -> SfconvSo2convMaterialInput {
    SfconvSo2convMaterialInput {
        core_hole_width_ev: 1.729,
        wigner_seitz_radius: 2.05,
        interstitial_potential_ev: 12.34,
        chemical_potential_ev: 18.76,
        fermi_wave_number_inv_angstrom: 1.23,
    }
}

fn expected_apl_dat() -> Result<String> {
    let parameters = sfconv_so2conv_material_parameters(xmu_header_material())?;
    let poles = sfconv_rdeps_fallback_poles(parameters.plasma_frequency, 1)?;
    Ok(sfconv_apl_dat_string(&poles)?)
}

fn expected_so2conv_log() -> &'static str {
    "Calculating S0^2 ...\nDone with module: S0^2.\r\n\n"
}

fn assert_exc_dat_close(actual: &ExcDatData, expected: &ExcDatData) {
    assert_eq!(actual.header_lines, expected.header_lines);
    assert_close_array(
        "energy_ev",
        actual.energy_ev.view(),
        expected.energy_ev.view(),
    );
    assert_close_array(
        "broadening_ev",
        actual.broadening_ev.view(),
        expected.broadening_ev.view(),
    );
    assert_close_array(
        "oscillator_strength",
        actual.oscillator_strength.view(),
        expected.oscillator_strength.view(),
    );
    match (&actual.auxiliary_weight, &expected.auxiliary_weight) {
        (Some(actual), Some(expected)) => {
            assert_close_array("auxiliary_weight", actual.view(), expected.view());
        }
        (None, None) => {}
        _ => panic!("auxiliary_weight presence mismatch"),
    }
}

fn assert_close_array(
    name: &str,
    actual: ndarray::ArrayView1<'_, f64>,
    expected: ndarray::ArrayView1<'_, f64>,
) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= 5.0e-10 * scale,
            "{name}[{index}] actual={actual} expected={expected}"
        );
    }
}

fn relative_l2(
    actual: ndarray::ArrayView1<'_, f64>,
    expected: ndarray::ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(actual.len(), expected.len(), "relative L2 length mismatch");
    let squared_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(&actual, &expected)| (actual - expected).powi(2))
        .sum::<f64>();
    let squared_reference = expected.iter().map(|value| value.powi(2)).sum::<f64>();
    (squared_error / squared_reference).sqrt()
}

fn reference_sfconv_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/XANES/Cu");
    Ok((path.join("sfconv.inp").is_file() && path.join("logsfconv.dat").is_file()).then_some(path))
}

fn reference_so2conv_zip() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/MPSE/Cu_OPCONS/REFERENCE.zip");
    Ok(path.is_file().then_some(path))
}

fn reference_self_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/MPSE/Cu");
    Ok((path.join("xsph.inp").is_file() && path.join("loss.dat").is_file()).then_some(path))
}

fn reference_self_zip() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/MPSE/Cu/REFERENCE.zip");
    Ok(path.is_file().then_some(path))
}

fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
    let output = Command::new("unzip")
        .arg("-p")
        .arg(zip_path)
        .arg(entry)
        .output()
        .with_context(|| format!("failed to read {entry} from {}", zip_path.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "failed to extract {entry} from {}: {stderr}",
            zip_path.display()
        );
    }
    Ok(output.stdout)
}
