use super::{
    Ff2xDecomposedPathSum, Ff2xMomentumGrid, Ff2xNrixsXmulComponents, Ff2xPathSum,
    Ff2xXanesCorrectedComponents, Ff2xXmuComponents, Ff2xXscorrInput,
    ff2x_atomic_xscorr_with_background, ff2x_chi_ckp_columns, ff2x_chi_dat_from_path_sum,
    ff2x_chip_dat_from_path_signal, ff2x_exafs_fine_momentum_grid, ff2x_momentum_grid,
    ff2x_nrixs_channel_background_from_xsecl, ff2x_nrixs_combined_decomposed_trace,
    ff2x_nrixs_optional_fmsl_trace, ff2x_nrixs_total_single_electron_response,
    ff2x_nrixs_xmul_dat_from_components, ff2x_nrixs_xmul_output_grid, ff2x_output_energy_grid,
    ff2x_output_energy_grid_for_input, ff2x_path_damping, ff2x_path_damping_in_dir,
    ff2x_path_summary_header_lines, ff2x_prepared_paths, ff2x_source_aligned_output_momentum,
    ff2x_sum_decomposed_paths, ff2x_sum_decomposed_paths_with_source_len, ff2x_sum_prepared_paths,
    ff2x_wrhead_lines, ff2x_xanes_apply_vicorr_convolution, ff2x_xanes_combined_trace,
    ff2x_xanes_corrected_background, ff2x_xanes_fms_trace, ff2x_xmu_dat_from_components,
    ff2x_xscorr, generated_ff2x_generation_module_log, generated_ff2x_module_log,
    has_cached_ff2x_output, run_in_dir,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use refeff_core::{
    FEFF_BOHR_ANGSTROM, FEFF_HARTREE_EV, Ff2xExcitationConvolutionInput, conv as lorentz_convolve,
    ff2x_excitation_convolve, wave_number_from_hartree,
};
use refeff_io::feff_bin::{FEFF_BIN_BOHR, FEFF_BIN_DEFAULT_PAD_WIDTH};
use refeff_io::{
    CfAverage, ChiDatData, ChiaBinData, CumDatData, CumDatEntry, DanesDatData, EelsAngles,
    EelsControl, EelsInput, EelsPolarization, EelsQMesh, FMS_BIN_DEFAULT_PAD_WIDTH, FeffBinData,
    FeffBinPath, FeffBinPotential, FefflBinData, Ff2xControl, Ff2xCorrections, Ff2xDebye,
    Ff2xInput, FmsBinData, FmslBinData, GeomDat, GeomDatRow, GlobalControl, GlobalInput,
    GlobalNorms, GlobalQControl, HubbardInput, IoError, ListDatData, ListDatEntry, ModuleLogData,
    SfconvSo2convTarget, SfconvSo2convTargetData, SfconvSo2convTargetKind, XmuDatData,
    XscorrComplexTable, XscorrCurveDatData, XscorrRawDatData, XseclBinData, XseclBinTransition,
    XsectDatData, XsectDatScalars, XsectFf2xHandoff, eels_input_string, ff2x_input_string,
    geom_dat_string, global_input_string, hubbard_input_string, read_chi_dat, read_chia_bin,
    read_contour_dat, read_cum_dat, read_curve_dat, read_danes_dat, read_feff_bin, read_gtrl_dat,
    read_list_dat, read_module_log_dat, read_prexmu_dat, read_residue_dat, read_xmu_dat,
    read_xmul_dat, read_xscorr_raw_dat, read_xsect_dat, sfconv_so2conv_target_data_from_text,
    write_chi_dat, write_chia_bin, write_contour_dat, write_cum_dat, write_curve_dat,
    write_danes_dat, write_feff_bin, write_feffl_bin, write_fms_bin, write_list_dat,
    write_module_log_dat, write_prexmu_dat, write_residue_dat, write_xmu_dat, write_xmul_dat,
    write_xscorr_raw_dat, write_xsecl_bin, write_xsect_dat, xmul_dat::XmulDatData,
    xsect_dat_ff2x_handoff,
};
use std::path::{Path, PathBuf};

#[test]
fn ff2x_module_skips_disabled_input() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 0)?;
    write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_ff2x_output(temp.path())?);
    Ok(())
}

#[test]
fn ff2x_module_skips_mchi_values_other_than_one() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 2)?;
    let xmu = sample_xmu_dat();
    write_xmu_dat(temp.path().join("xmu.dat"), &xmu)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_ff2x_output(temp.path())?);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, xmu);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_requires_cache_or_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled FF2X should require cache or source handoffs")?;

    assert!(error.to_string().contains(
        "FF2X spectrum generation requires cached final-spectrum output or xsect.dat/feff.bin/list.dat source handoffs"
    ));
    Ok(())
}

#[test]
fn ff2x_module_does_not_claim_malformed_cache_without_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("xmu.dat"), b"not an xmu.dat cache\n")?;

    assert!(!has_cached_ff2x_output(temp.path())?);
    Ok(())
}

#[test]
fn ff2x_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_xmu_dat();
    std::fs::write(temp.path().join("ff2x.inp"), b"not an ff2x.inp handoff\n")?;
    write_xmu_dat(temp.path().join("xmu.dat"), &expected)?;

    assert!(!has_cached_ff2x_output(temp.path())?);
    let error = run_in_dir(temp.path())
        .err()
        .context("malformed FF2X input should fail through explicit run")?;
    let chain = format!("{error:?}");

    assert!(chain.contains("failed to parse"), "{chain}");
    assert!(chain.contains("ff2x.inp"), "{chain}");
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let expected = sample_xmu_dat();
    write_xmu_dat(temp.path().join("xmu.dat"), &expected)?;

    assert!(!has_cached_ff2x_output(temp.path())?);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_does_not_claim_malformed_xsect_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    std::fs::write(temp.path().join("xsect.dat"), b"not an xsect.dat source\n")?;

    assert!(!has_cached_ff2x_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed FF2X xsect.dat should fail through the explicit FF2X runner")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsect.dat"), "{chain}");
    assert!(!temp.path().join("xmu.dat").exists());
    assert!(!temp.path().join("chi.dat").exists());
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_does_not_claim_cached_output_with_malformed_xsect_source_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    let xmu = sample_xmu_dat();
    let chi = sample_chi_dat();
    write_xmu_dat(temp.path().join("xmu.dat"), &xmu)?;
    write_chi_dat(temp.path().join("chi.dat"), &chi)?;
    std::fs::write(temp.path().join("xsect.dat"), b"not an xsect.dat source\n")?;

    assert!(!has_cached_ff2x_output(temp.path())?);

    let error = run_in_dir(temp.path())
        .err()
        .context("malformed FF2X xsect.dat should block cached FF2X completion")?;
    let chain = format!("{error:#}");
    assert!(chain.contains("failed to read"), "{chain}");
    assert!(chain.contains("xsect.dat"), "{chain}");
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, chi);
    assert!(!temp.path().join("log6.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_roundtrips_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    let xmu = sample_xmu_dat();
    let chi = sample_chi_dat();
    let danes = sample_danes_dat();
    let xscorr = sample_xscorr_complex_table();
    let curve = sample_xscorr_curve_dat();
    let raw = sample_xscorr_raw_dat();
    let cum = sample_cum_dat();
    let log = sample_module_log();
    write_xmu_dat(temp.path().join("xmu.dat"), &xmu)?;
    write_chi_dat(temp.path().join("chi.dat"), &chi)?;
    write_danes_dat(temp.path().join("danes.dat"), &danes)?;
    write_prexmu_dat(temp.path().join("prexmu.dat"), &xscorr)?;
    write_residue_dat(temp.path().join("residue.dat"), &xscorr)?;
    write_contour_dat(temp.path().join("contour.dat"), &xscorr)?;
    write_curve_dat(temp.path().join("curve.dat"), &curve)?;
    write_xscorr_raw_dat(temp.path().join("raw.dat"), &raw)?;
    write_cum_dat(temp.path().join("cum.dat"), &cum)?;
    write_module_log_dat(temp.path().join("log6.dat"), &log)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 10);
    assert!(has_cached_ff2x_output(temp.path())?);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, chi);
    assert_eq!(read_danes_dat(temp.path().join("danes.dat"))?, danes);
    assert_eq!(read_prexmu_dat(temp.path().join("prexmu.dat"))?, xscorr);
    assert_eq!(read_residue_dat(temp.path().join("residue.dat"))?, xscorr);
    assert_eq!(read_contour_dat(temp.path().join("contour.dat"))?, xscorr);
    assert_eq!(read_curve_dat(temp.path().join("curve.dat"))?, curve);
    assert_eq!(read_xscorr_raw_dat(temp.path().join("raw.dat"))?, raw);
    assert_eq!(read_cum_dat(temp.path().join("cum.dat"))?, cum);
    assert_eq!(read_module_log_dat(temp.path().join("log6.dat"))?, log);
    Ok(())
}

#[test]
fn ff2x_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    write_xmu_dat(temp.path().join("xmu.dat"), &sample_xmu_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, sample_xmu_dat());
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_module_log()
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_two_chip_outputs_from_cached_spectrum() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 2;
    let cached_chi = sample_chi_dat();
    let feff = sample_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_chi_dat(temp.path().join("chi.dat"), &cached_chi)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let generated_chi = read_chi_dat(temp.path().join("chi.dat"))?;
    assert_ne!(generated_chi, cached_chi);
    assert!(temp.path().join("xmu.dat").is_file());
    let chip = read_chi_dat(temp.path().join("chip0017.dat"))?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    assert_eq!(chip.point_count(), grid.output_momentum.len());
    assert!(chip.has_path_phase());
    assert!(chip.header_lines[0].starts_with("# PATH"));
    assert!(!temp.path().join("files.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_generates_cached_polarized_chip_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 2;
    let cached_chi = sample_chi_dat();
    let feff = sample_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_eels_polarization_input(temp.path(), 5, 1, 5)?;
    write_chi_dat(temp.path().join("chi05.dat"), &cached_chi)?;
    write_feff_bin(temp.path().join("feff05.bin"), &feff)?;
    write_list_dat(temp.path().join("list05.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let generated_chi = read_chi_dat(temp.path().join("chi05.dat"))?;
    assert_ne!(generated_chi, cached_chi);
    assert!(temp.path().join("xmu05.dat").is_file());
    let chip = read_chi_dat(temp.path().join("chip0017.dat"))?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    assert_eq!(chip.point_count(), grid.output_momentum.len());
    assert!(chip.has_path_phase());
    assert!(!temp.path().join("chip0005.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_writes_cached_sig3_cum_dat_without_ipr6_diagnostics() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.thetae = 400.0;
    let cached_xmu = sample_xmu_dat();
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();
    let expected_damping = ff2x_path_damping(&input, &feff, &list)?;
    let expected_cumulants = expected_damping[0]
        .cumulants
        .context("single-scattering path should have cumulants")?;
    write_ff2x_input_data(temp.path(), &input)?;
    write_xmu_dat(temp.path().join("xmu.dat"), &cached_xmu)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_ne!(read_xmu_dat(temp.path().join("xmu.dat"))?, cached_xmu);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(!temp.path().join("chip0017.dat").exists());
    assert!(!temp.path().join("files.dat").exists());
    let cum = read_cum_dat(temp.path().join("cum.dat"))?;
    assert_eq!(cum.entries.len(), 1);
    assert_eq!(cum.entries[0].path_index, 17);
    assert_close(
        cum.entries[0].first_cumulant_angstrom,
        expected_cumulants.first_cumulant_bohr * FEFF_BIN_BOHR,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].sigma2_angstrom2,
        expected_damping[0].total_sigma2_angstrom2,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].third_cumulant_angstrom3,
        expected_cumulants.third_cumulant_bohr3 * FEFF_BIN_BOHR.powi(3),
        5.0e-8,
    );
    Ok(())
}

#[test]
fn ff2x_module_regenerates_stale_sig3_cum_dat_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.thetae = 400.0;
    let cached_xmu = sample_xmu_dat();
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();
    let expected_damping = ff2x_path_damping(&input, &feff, &list)?;
    let expected_cumulants = expected_damping[0]
        .cumulants
        .context("single-scattering path should have cumulants")?;
    let stale_cum = CumDatData {
        einstein_temperature: 25.0,
        thermal_expansion: 0.5,
        entries: vec![CumDatEntry {
            path_index: 99,
            first_cumulant_angstrom: 1.0,
            sigma2_angstrom2: 2.0,
            third_cumulant_angstrom3: 3.0,
        }],
    };
    write_ff2x_input_data(temp.path(), &input)?;
    write_xmu_dat(temp.path().join("xmu.dat"), &cached_xmu)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;
    write_cum_dat(temp.path().join("cum.dat"), &stale_cum)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert_ne!(read_xmu_dat(temp.path().join("xmu.dat"))?, cached_xmu);
    let cum = read_cum_dat(temp.path().join("cum.dat"))?;
    assert_ne!(cum, stale_cum);
    assert_eq!(cum.einstein_temperature, 400.0);
    assert_eq!(cum.thermal_expansion, 0.034);
    assert_eq!(cum.entries.len(), 1);
    assert_eq!(cum.entries[0].path_index, 17);
    assert_close(
        cum.entries[0].first_cumulant_angstrom,
        expected_cumulants.first_cumulant_bohr * FEFF_BIN_BOHR,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].sigma2_angstrom2,
        expected_damping[0].total_sigma2_angstrom2,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].third_cumulant_angstrom3,
        expected_cumulants.third_cumulant_bohr3 * FEFF_BIN_BOHR.powi(3),
        5.0e-8,
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_three_feff_path_outputs_from_cached_spectrum() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 3;
    let cached_xmu = sample_xmu_dat();
    let feff = sample_feff_bin_data();
    let mut xsect = sample_xsect_dat();
    xsect.titles = sample_so2conv_header_titles();
    write_ff2x_input_data(temp.path(), &input)?;
    write_xmu_dat(temp.path().join("xmu.dat"), &cached_xmu)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert_ne!(read_xmu_dat(temp.path().join("xmu.dat"))?, cached_xmu);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("chip0017.dat").is_file());
    let files = std::fs::read_to_string(temp.path().join("files.dat"))?;
    assert!(files.contains("feff0017.dat"));
    assert!(files.contains("    12.500"));
    let target = SfconvSo2convTarget {
        file_name: "feff0017.dat".to_string(),
        kind: SfconvSo2convTargetKind::FeffPath,
    };
    let text = std::fs::read_to_string(temp.path().join("feff0017.dat"))?;
    let data =
        sfconv_so2conv_target_data_from_text(temp.path().join("feff0017.dat"), &target, &text)?;
    let SfconvSo2convTargetData::FeffPath { header, data } = data else {
        unreachable!("target kind selects feff path data");
    };
    assert_eq!(header.material.core_hole_width_ev, 1.729);
    assert_eq!(data.point_count(), xsect.main_energy_count);
    assert_eq!(data.leg_count, feff.paths[0].leg_count());
    assert_close(data.degeneracy, feff.paths[0].degeneracy, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_module_roundtrips_cached_polarized_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_ff2x_input(temp.path(), 1)?;
    let xmu = sample_xmu_dat();
    let chi = sample_chi_dat();
    write_xmu_dat(temp.path().join("xmu05.dat"), &xmu)?;
    write_chi_dat(temp.path().join("chi05.dat"), &chi)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(has_cached_ff2x_output(temp.path())?);
    assert_eq!(read_xmu_dat(temp.path().join("xmu05.dat"))?, xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi05.dat"))?, chi);
    assert!(temp.path().join("log6.dat").is_file());
    Ok(())
}

#[test]
fn ff2x_module_roundtrips_generated_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_ff2x_dir()? else {
        crate::require_fixture!("FF2X reference test; generated EXAFS/Cu reference not found");
    };

    let temp = tempfile::tempdir()?;
    for name in ["ff2x.inp", "xmu.dat", "chi.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }
    let mut sidecar_count = 0_usize;
    for name in [
        "prexmu.dat",
        "residue.dat",
        "contour.dat",
        "curve.dat",
        "raw.dat",
        "cum.dat",
    ] {
        let source = reference_dir.join(name);
        if source.is_file() {
            std::fs::copy(source, temp.path().join(name))?;
            sidecar_count += 1;
        }
    }
    let has_log = reference_dir.join("log6.dat").is_file();
    if has_log {
        std::fs::copy(reference_dir.join("log6.dat"), temp.path().join("log6.dat"))?;
    }
    let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let expected_chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let expected_prexmu =
        optional_read(temp.path().join("prexmu.dat"), |path| read_prexmu_dat(path))?;
    let expected_residue = optional_read(temp.path().join("residue.dat"), |path| {
        read_residue_dat(path)
    })?;
    let expected_contour = optional_read(temp.path().join("contour.dat"), |path| {
        read_contour_dat(path)
    })?;
    let expected_curve = optional_read(temp.path().join("curve.dat"), |path| read_curve_dat(path))?;
    let expected_raw = optional_read(temp.path().join("raw.dat"), |path| {
        read_xscorr_raw_dat(path)
    })?;
    let expected_cum = optional_read(temp.path().join("cum.dat"), |path| read_cum_dat(path))?;
    let expected_log = optional_module_log(temp.path().join("log6.dat"))?
        .unwrap_or_else(generated_ff2x_module_log);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2 + sidecar_count + 1);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, expected_chi);
    if let Some(expected) = expected_prexmu {
        assert_eq!(read_prexmu_dat(temp.path().join("prexmu.dat"))?, expected);
    }
    if let Some(expected) = expected_residue {
        assert_eq!(read_residue_dat(temp.path().join("residue.dat"))?, expected);
    }
    if let Some(expected) = expected_contour {
        assert_eq!(read_contour_dat(temp.path().join("contour.dat"))?, expected);
    }
    if let Some(expected) = expected_curve {
        assert_eq!(read_curve_dat(temp.path().join("curve.dat"))?, expected);
    }
    if let Some(expected) = expected_raw {
        assert_eq!(read_xscorr_raw_dat(temp.path().join("raw.dat"))?, expected);
    }
    if let Some(expected) = expected_cum {
        assert_eq!(read_cum_dat(temp.path().join("cum.dat"))?, expected);
    }
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        expected_log
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_prepares_correlated_debye_values() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.sig2g = 0.002;
    let feff = sample_feff_bin_data();
    let mut list = sample_list_dat();
    list.entries[0].sigma2 = 0.0015;

    let damping = ff2x_path_damping(&input, &feff, &list)?;

    assert_eq!(damping.len(), 1);
    let path = damping[0];
    assert_eq!(path.path_index, 17);
    assert_eq!(path.global_sigma2_angstrom2, 0.002);
    assert_eq!(path.user_sigma2_angstrom2, 0.0015);
    assert!(path.debye_sigma2_angstrom2 > 0.0);
    assert_close(
        path.total_sigma2_angstrom2,
        path.global_sigma2_angstrom2 + path.user_sigma2_angstrom2 + path.debye_sigma2_angstrom2,
        1.0e-12,
    );
    assert_eq!(path.criterion, 12.5);
    assert_eq!(path.degeneracy, 4.0);
    assert_eq!(path.leg_count, 3);
    assert_close(path.effective_half_path_length_angstrom, 2.5, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_path_damping_supports_classical_debye_values() -> Result<()> {
    let mut quantum = sample_ff2x_input(1);
    quantum.control.idwopt = 0;
    let mut classical = quantum;
    classical.control.idwopt = 3;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();

    let quantum_damping = ff2x_path_damping(&quantum, &feff, &list)?;
    let classical_damping = ff2x_path_damping(&classical, &feff, &list)?;

    assert_eq!(classical_damping.len(), 1);
    assert!(classical_damping[0].debye_sigma2_angstrom2 > 0.0);
    assert_ne!(
        classical_damping[0].debye_sigma2_angstrom2,
        quantum_damping[0].debye_sigma2_angstrom2
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_supports_dmdw_values() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.idwopt = 5;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    write_ff2x_dmdw_handoffs(temp.path(), &feff.paths[0])?;

    let damping = ff2x_path_damping_in_dir(Some(temp.path()), &input, &feff, &list)?;

    assert_eq!(damping.len(), 1);
    assert!(damping[0].debye_sigma2_angstrom2 > 0.0);
    assert_close(
        damping[0].total_sigma2_angstrom2,
        damping[0].debye_sigma2_angstrom2,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_supports_recursion_debye_values() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.idwopt = 2;
    let feff = sample_ff2x_spring_feff_bin_data();
    let list = sample_ff2x_spring_list_dat();
    write_ff2x_spring_handoffs(temp.path())?;

    let damping = ff2x_path_damping_in_dir(Some(temp.path()), &input, &feff, &list)?;

    assert_eq!(damping.len(), 1);
    assert_close(
        damping[0].debye_sigma2_angstrom2,
        0.031_891_103_846_101_55,
        1.0e-14,
    );
    assert_eq!(
        damping[0].total_sigma2_angstrom2,
        damping[0].debye_sigma2_angstrom2
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_supports_equation_of_motion_debye_values() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.idwopt = 1;
    let feff = sample_ff2x_spring_feff_bin_data();
    let list = sample_ff2x_spring_list_dat();
    write_ff2x_spring_handoffs(temp.path())?;

    let damping = ff2x_path_damping_in_dir(Some(temp.path()), &input, &feff, &list)?;

    assert_eq!(damping.len(), 1);
    assert_close(
        damping[0].debye_sigma2_angstrom2,
        0.130_988_883_045_134_37,
        1.0e-14,
    );
    assert_eq!(
        damping[0].total_sigma2_angstrom2,
        damping[0].debye_sigma2_angstrom2
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_prepares_morse_cumulants_for_single_scattering() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 1.0e-5;
    input.debye.thetae = 400.0;
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();

    let damping = ff2x_path_damping(&input, &feff, &list)?;

    assert_eq!(damping.len(), 1);
    let cumulants = damping[0]
        .cumulants
        .context("single-scattering path should have cumulants")?;
    assert!(cumulants.first_cumulant_bohr > 0.0);
    assert!(cumulants.third_cumulant_bohr3 > 0.0);
    Ok(())
}

#[test]
fn ff2x_prepared_paths_apply_s02_and_degeneracy_without_dw() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    input.corrections.s02 = 0.5;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();

    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;

    assert_eq!(prepared.len(), 1);
    assert_close(prepared[0].amplitude[0], 2.0 * 0.5 * 4.0, 1.0e-12);
    assert_close(prepared[0].amplitude[1], 2.1 * 0.5 * 4.0, 1.0e-12);
    assert_close(prepared[0].phase[0], -0.1, 1.0e-12);
    assert_close(prepared[0].phase[1], -0.2, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_prepared_paths_ignore_s02_for_many_body_convolution() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.mbconv = 1;
    input.debye.tk = 0.0;
    input.corrections.s02 = 0.25;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();

    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;

    assert_eq!(prepared.len(), 1);
    assert_close(prepared[0].amplitude[0], 2.0 * 4.0, 1.0e-12);
    assert_close(prepared[0].amplitude[1], 2.1 * 4.0, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_source_aligned_output_momentum_applies_real_correction_in_hartrees() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.corrections.vrcorr = 0.02 * FEFF_HARTREE_EV;
    let feff = sample_feff_bin_data();

    let output = ff2x_source_aligned_output_momentum(&input, &feff)?;

    assert_eq!(output.len(), feff.real_momentum.len());
    assert_close(output[0], (0.5_f64 * 0.5 + 0.04).sqrt(), 1.0e-12);
    assert_close(output[1], (0.6_f64 * 0.6 + 0.04).sqrt(), 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_exafs_fine_momentum_grid_matches_feff_spacing_and_stop() -> Result<()> {
    let input = sample_ff2x_input(1);
    let feff = sample_feff_bin_data();

    let grid = ff2x_exafs_fine_momentum_grid(&input, &feff, 100)?;

    let delta = 0.05 * FEFF_BIN_BOHR;
    assert_eq!(grid.output_momentum.len(), 8);
    assert_eq!(
        grid.output_momentum.len(),
        grid.interpolation_momentum.len()
    );
    assert_close(grid.output_momentum[0], 19.0 * delta, 1.0e-12);
    for row in 1..grid.output_momentum.len() {
        assert_close(
            grid.output_momentum[row] - grid.output_momentum[row - 1],
            delta,
            1.0e-12,
        );
        assert!(grid.interpolation_momentum[row] > grid.interpolation_momentum[row - 1]);
    }
    for row in 0..grid.output_momentum.len() {
        assert_close(
            grid.interpolation_momentum[row],
            grid.output_momentum[row],
            1.0e-12,
        );
    }
    assert!(grid.interpolation_momentum[grid.interpolation_momentum.len() - 1] <= 0.7 + 1.0e-4);
    assert!(grid.output_momentum[grid.output_momentum.len() - 1] + delta > 0.7 + 1.0e-4);
    Ok(())
}

#[test]
fn ff2x_path_sum_uses_monotonic_feff_grid_prefix_before_padding() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let mut feff = sample_feff_bin_data();
    feff.central_phase_shift = Array1::from_vec(vec![
        Complex64::new(0.1, -0.01),
        Complex64::new(0.2, -0.02),
        Complex64::new(0.3, -0.03),
        Complex64::new(9.9, 0.0),
    ]);
    feff.complex_momentum = Array1::from_vec(vec![
        Complex64::new(1.0, 0.1),
        Complex64::new(1.1, 0.2),
        Complex64::new(1.2, 0.3),
        Complex64::new(0.0, 0.0),
    ]);
    feff.real_momentum = Array1::from_vec(vec![0.5, 0.6, 0.7, 0.0]);
    feff.paths[0].amplitude = Array1::from_vec(vec![2.0, 2.1, 2.2, 99.0]);
    feff.paths[0].phase = Array1::from_vec(vec![-0.1, -0.2, -0.3, 99.0]);
    let prepared = ff2x_prepared_paths(&input, &feff, &sample_list_dat())?;
    let output_momentum = Array1::from_vec(vec![0.65]);

    let summed = ff2x_sum_prepared_paths(&feff, &prepared, output_momentum.view())?;

    let expected = expected_ff2x_path_signal(
        2.15 * feff.paths[0].degeneracy,
        -0.25,
        0.65,
        feff.paths[0].effective_half_path_length_bohr,
    );
    assert_complex_close(summed.total[0], expected, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_decomposed_path_sum_uses_feffl_channel_amplitude_phase() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let prepared = ff2x_prepared_paths(&input, &feff, &sample_list_dat())?;
    let feffl = sample_feffl_bin_data(prepared.len(), feff.energy_count(), 1);
    let output_momentum = Array1::from_vec(vec![feff.real_momentum[0], feff.real_momentum[1]]);

    let summed = ff2x_sum_decomposed_paths(&feff, &feffl, &prepared, output_momentum.view())?;

    assert_eq!(summed.total.shape(), &[2, 2, 2]);
    assert_eq!(summed.paths.len(), prepared.len());
    assert_eq!(summed.paths[0].path_index, feff.paths[0].index);
    assert_eq!(summed.paths[0].signal.shape(), &[2, 2, 2]);
    for row in 0..2 {
        for lg2 in 0..2 {
            for lg1 in 0..2 {
                let expected = expected_ff2x_path_signal(
                    sample_feffl_amplitude(0, lg2, lg1, row),
                    sample_feffl_phase(0, lg2, lg1, row),
                    output_momentum[row],
                    prepared[0].damping.effective_half_path_length_bohr,
                );
                assert_complex_close(summed.paths[0].signal[(row, lg2, lg1)], expected, 1.0e-12);
                assert_complex_close(summed.total[(row, lg2, lg1)], expected, 1.0e-12);
            }
        }
    }
    Ok(())
}

#[test]
fn ff2x_decomposed_path_sum_rejects_feffl_shape_mismatch() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let prepared = ff2x_prepared_paths(&input, &feff, &sample_list_dat())?;
    let mut feffl = sample_feffl_bin_data(prepared.len(), feff.energy_count(), 1);
    feffl.amplitudes = Array4::zeros((prepared.len() + 1, 2, 2, feff.energy_count()));
    let output_momentum = Array1::from_vec(vec![feff.real_momentum[0], feff.real_momentum[1]]);

    let error = ff2x_sum_decomposed_paths_with_source_len(
        &feff,
        &feffl,
        &prepared,
        2,
        output_momentum.view(),
    )
    .err()
    .context("mismatched feffl.bin shape should be rejected")?;

    assert!(error.to_string().contains("FF2X feffl.bin amplitude shape"));
    Ok(())
}

#[test]
fn ff2x_momentum_grid_selects_fine_grid_for_exafs_and_source_grid_for_xanes() -> Result<()> {
    let feff = sample_feff_bin_data();
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 0;

    let exafs = ff2x_momentum_grid(&input, &feff)?;

    assert!(exafs.output_momentum.len() > feff.real_momentum.len());
    input.control.ispec = 1;
    let xanes = ff2x_momentum_grid(&input, &feff)?;
    assert_eq!(xanes.output_momentum.len(), feff.real_momentum.len());
    assert_eq!(xanes.output_momentum, xanes.interpolation_momentum);
    Ok(())
}

#[test]
fn ff2x_output_energy_grid_matches_feff_omegax_rule() -> Result<()> {
    let xsect = sample_xsect_handoff();
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![-0.4, 0.0, 0.5]),
        interpolation_momentum: Array1::from_vec(vec![-0.4, 0.0, 0.5]),
    };

    let output = ff2x_output_energy_grid(1.25, &xsect, &momentum_grid)?;

    assert_close(output.fermi_energy_hartree, 0.45, 1.0e-12);
    assert_close(
        output.photon_energy_hartree[0],
        0.45 - 0.4_f64.powi(2) / 2.0,
        1.0e-12,
    );
    assert_close(output.photon_energy_hartree[1], 0.45, 1.0e-12);
    assert_close(
        output.photon_energy_hartree[2],
        0.45 + 0.5_f64.powi(2) / 2.0,
        1.0e-12,
    );
    assert_close(
        output.relative_energy_hartree[0],
        1.25 - 0.4_f64.powi(2) / 2.0,
        1.0e-12,
    );
    assert_close(output.relative_energy_hartree[1], 1.25, 1.0e-12);
    assert_close(
        output.relative_energy_hartree[2],
        1.25 + 0.5_f64.powi(2) / 2.0,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn ff2x_output_energy_grid_for_input_applies_feff_real_edge_shift() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.corrections.vrcorr = 0.25 * FEFF_HARTREE_EV;
    let xsect = sample_xsect_handoff();
    let mut feff = sample_feff_bin_data();
    feff.edge_energy = 1.25;
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![0.0, 0.5]),
        interpolation_momentum: Array1::from_vec(vec![0.0, 0.5]),
    };

    let shifted = ff2x_output_energy_grid_for_input(&input, &feff, &xsect, &momentum_grid)?;
    let unshifted = ff2x_output_energy_grid(feff.edge_energy, &xsect, &momentum_grid)?;

    assert_close(
        shifted.fermi_energy_hartree,
        unshifted.fermi_energy_hartree - 0.25,
        1.0e-12,
    );
    assert_close(
        shifted.photon_energy_hartree[1],
        unshifted.photon_energy_hartree[1] - 0.25,
        1.0e-12,
    );
    assert_close(
        shifted.relative_energy_hartree[1],
        unshifted.relative_energy_hartree[1] - 0.25,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn ff2x_xmu_dat_rows_interpolate_components_without_normalization() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.absolu = 1;
    let xsect = sample_xmu_row_xsect_handoff();
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![0.5, 1.5]),
        interpolation_momentum: Array1::from_vec(vec![0.5, 1.5]),
    };
    let output_energy = ff2x_output_energy_grid(1.0, &xsect, &momentum_grid)?;
    let path_sum = Ff2xPathSum {
        total: Array1::from_vec(vec![Complex64::new(0.0, 0.1), Complex64::new(0.0, 0.2)]),
        paths: Vec::new(),
    };
    let path_chi = Array1::from_vec(vec![0.1, 0.2]);
    let corrected_atomic_cross_section = Array1::from_vec(vec![10.0, 20.0, 30.0]);

    let xmu = ff2x_xmu_dat_from_components(Ff2xXmuComponents {
        input: &input,
        xsect: &xsect,
        momentum_grid: &momentum_grid,
        output_energy: &output_energy,
        path_sum: &path_sum,
        path_chi: path_chi.view(),
        corrected_background: xsect.normalized_background.view(),
        corrected_atomic_cross_section: corrected_atomic_cross_section.view(),
        pre_table_header_lines: &[],
        used_path_count: 2,
        total_path_count: 5,
    })?;

    assert_eq!(xmu.header_lines[0], "#     2/   5 paths used");
    assert_eq!(xmu.normalization, Some(1.0));
    assert_close(xmu.wave_number[0], 0.5 / FEFF_BIN_BOHR, 1.0e-12);
    assert_close(xmu.mu0[0], 15.0, 1.0e-12);
    assert_close(xmu.mu[0], 15.0 + 3.0 * 0.1, 1.0e-12);
    assert_close(xmu.chi[0], 0.1, 1.0e-12);
    assert_close(xmu.mu0[1], 25.0, 1.0e-12);
    assert_close(xmu.mu[1], 25.0 + 5.0 * 0.2, 1.0e-12);
    assert_close(xmu.chi[1], 0.2, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_xmu_dat_rows_apply_xsedge_and_pre_edge_chi_clamp() -> Result<()> {
    let input = sample_ff2x_input(1);
    let mut xsect = sample_xmu_row_xsect_handoff();
    xsect.energy_grid_hartree = Array1::from_vec(vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    xsect.omega_hartree = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    xsect.wave_number = Array1::from_vec(vec![-0.2, 0.0, 0.5]);
    xsect.normalized_background = Array1::from_vec(vec![3.0, 4.0, 5.0]);
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![-0.4, -0.2, 0.5]),
        interpolation_momentum: Array1::from_vec(vec![-0.2, -0.1, 0.5]),
    };
    let output_energy = ff2x_output_energy_grid(1.0, &xsect, &momentum_grid)?;
    let path_sum = Ff2xPathSum {
        total: Array1::from_vec(vec![
            Complex64::new(0.0, 0.1),
            Complex64::new(0.0, 0.7),
            Complex64::new(0.0, 0.2),
        ]),
        paths: Vec::new(),
    };
    let path_chi = Array1::from_vec(vec![0.1, 0.7, 0.2]);
    let corrected_atomic_cross_section = Array1::from_vec(vec![10.0, 20.0, 30.0]);

    let xmu = ff2x_xmu_dat_from_components(Ff2xXmuComponents {
        input: &input,
        xsect: &xsect,
        momentum_grid: &momentum_grid,
        output_energy: &output_energy,
        path_sum: &path_sum,
        path_chi: path_chi.view(),
        corrected_background: xsect.normalized_background.view(),
        corrected_atomic_cross_section: corrected_atomic_cross_section.view(),
        pre_table_header_lines: &[],
        used_path_count: 1,
        total_path_count: 3,
    })?;

    let xsedge = 3.0 + (50.0 / FEFF_HARTREE_EV - 1.0);
    assert_close(xmu.normalization.unwrap(), xsedge, 1.0e-12);
    assert_close(xmu.mu0[0], 10.0 / xsedge, 1.0e-12);
    assert_close(xmu.mu[0], (10.0 + 3.0 * 0.7) / xsedge, 1.0e-12);
    assert_close(xmu.chi[0], 0.1, 1.0e-12);
    assert_close(xmu.mu0[2], 30.0 / xsedge, 1.0e-12);
    assert_close(xmu.mu[2], (30.0 + 5.0 * 0.2) / xsedge, 1.0e-12);
    assert_close(xmu.chi[2], 0.2, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_xscorr_constant_atomic_cross_section_matches_astep() -> Result<()> {
    let input = sample_ff2x_input(1);
    let xsect = sample_constant_xscorr_handoff();

    let xscorr =
        ff2x_atomic_xscorr_with_background(&input, &xsect, xsect.normalized_background.view())?;

    let xloss = 0.1;
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / xloss).atan() / std::f64::consts::PI;
        assert_close(
            xscorr.corrected_atomic_cross_section[row],
            10.0 * step,
            1.0e-12,
        );
        assert_complex_close(
            xscorr.cchi[row],
            Complex64::new(0.0, 10.0 * (step - 1.0)),
            1.0e-12,
        );
    }
    Ok(())
}

#[test]
fn ff2x_thermal_xscorr_constant_atomic_cross_section_matches_feff_grid() -> Result<()> {
    let xsect = sample_constant_thermal_xscorr_handoff();
    let zero_path_chi = Array1::<Complex64>::zeros(xsect.energy_count());

    let cchi = ff2x_xscorr(Ff2xXscorrInput {
        ispec: 1,
        energy_grid_hartree: xsect.energy_grid_hartree.view(),
        main_energy_count: xsect.main_energy_count,
        fermi_index: xsect.fermi_index,
        cross_section: xsect.cross_section.view(),
        background: xsect.normalized_background.view(),
        path_chi: zero_path_chi.view(),
        real_correction_hartree: 0.0,
        electronic_temperature_ev: 0.5,
    })?;

    let expected_corrected = [
        2.585_850_230_196_276,
        5.000_003_995_527_989,
        7.414_157_694_791_554,
    ];
    for (row, &expected) in expected_corrected.iter().enumerate() {
        assert_complex_close(cchi[row], Complex64::new(0.0, expected - 10.0), 1.0e-11);
        assert_close((xsect.cross_section[row] + cchi[row]).im, expected, 1.0e-11);
    }
    Ok(())
}

#[test]
fn ff2x_chi_dat_rows_use_output_grid_and_unwrapped_phase() -> Result<()> {
    let input = sample_ff2x_input(1);
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![FEFF_BIN_BOHR, 2.0 * FEFF_BIN_BOHR]),
        interpolation_momentum: Array1::from_vec(vec![0.5, 0.6]),
    };
    let path_sum = Ff2xPathSum {
        total: Array1::from_vec(vec![
            complex_from_polar(2.0, 3.0),
            complex_from_polar(2.5, -3.0),
        ]),
        paths: Vec::new(),
    };

    let chi = ff2x_chi_dat_from_path_sum(&input, &momentum_grid, &path_sum, &[], 1, 3)?;

    assert_eq!(chi.header_lines[0], "#     1/   3 paths used");
    assert_close(chi.wave_number[0], 1.0, 1.0e-12);
    assert_close(chi.wave_number[1], 2.0, 1.0e-12);
    assert_close(chi.chi[0], path_sum.total[0].im, 1.0e-12);
    assert_close(chi.magnitude[1], 2.5, 1.0e-12);
    assert_close(chi.phase[0], 3.0, 1.0e-12);
    assert_close(chi.phase[1], -3.0 + std::f64::consts::TAU, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_chi_dat_rows_use_real_part_for_ispec_three() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 3;
    let momentum_grid = Ff2xMomentumGrid {
        output_momentum: Array1::from_vec(vec![FEFF_BIN_BOHR]),
        interpolation_momentum: Array1::from_vec(vec![0.5]),
    };
    let path_sum = Ff2xPathSum {
        total: Array1::from_vec(vec![Complex64::new(1.25, -0.5)]),
        paths: Vec::new(),
    };

    let chi = ff2x_chi_dat_from_path_sum(&input, &momentum_grid, &path_sum, &[], 1, 1)?;

    assert_close(chi.chi[0], 1.25, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_path_summary_header_lines_include_user_sigma_when_present() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;

    let headers = ff2x_path_summary_header_lines(&prepared);

    assert_eq!(headers.len(), 1);
    let tokens = headers[0]
        .trim_start_matches('#')
        .split_whitespace()
        .collect::<Vec<_>>();
    assert_eq!(
        tokens,
        ["17", "0.00100", "100.00", "12.00", "2", "2.5000", "0.00100"]
    );
    Ok(())
}

#[test]
fn ff2x_wrhead_lines_match_regular_exafs_header_shape() {
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.sig2g = 0.00123;
    input.corrections.vrcorr = 1.25;
    input.corrections.vicorr = 0.5;
    input.corrections.critcw = 4.5;
    let mut xsect = sample_xsect_handoff();
    xsect.amplitude_reduction = 0.85;
    let mut list = sample_list_dat();
    list.titles.push("PATH second title".to_string());

    let headers = ff2x_wrhead_lines(&input, &xsect, &list);

    assert!(headers[0].starts_with("# # Cu test"));
    assert!(headers[0].contains("FEFF 10.0.0"));
    assert_eq!(headers[1], "# # PATH  Rmax= 6.000");
    assert_eq!(headers[2], "# # PATH second title");
    assert_eq!(
        headers[3],
        "#  S02=0.850  Temp= 190.00  Debye_temp= 315.00  Global_sig2= 0.00123"
    );
    assert!(headers[4].contains("1st and 3rd cumulants"));
    assert!(headers[5].contains("1.25000E"));
    assert!(headers[5].contains("5.00000E"));
    assert_eq!(headers[6], "#  Curved wave amplitude ratio filter   4.500%");
    assert_eq!(
        headers[7],
        "#     file         sig2 tot  cw amp ratio   deg  nlegs   reff  inp sig2"
    );
}

#[test]
fn ff2x_generation_module_log_matches_ff2chi_screen_summary() {
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.sig2g = 0.00123;
    input.corrections.vrcorr = 0.02 * FEFF_HARTREE_EV;
    input.corrections.vicorr = 0.01 * FEFF_HARTREE_EV;
    input.corrections.critcw = 12.5;
    let mut xsect = sample_xsect_handoff();
    xsect.amplitude_reduction = 0.85;

    let log = generated_ff2x_generation_module_log(&input, &xsect, 1);

    assert_eq!(log.lines[0], "Calculating XAS spectra ...");
    assert!(log.lines[1].contains("1st and 3rd cumulants"));
    let shift_tokens = log.lines[2].split_whitespace().collect::<Vec<_>>();
    assert_eq!(
        &shift_tokens[..5],
        ["Energy", "zero", "shift,", "vr,", "vi"]
    );
    assert_close(
        shift_tokens[5].parse::<f64>().unwrap(),
        0.02 * FEFF_HARTREE_EV,
        5.0e-7,
    );
    assert_close(
        shift_tokens[6].parse::<f64>().unwrap(),
        0.01 * FEFF_HARTREE_EV,
        5.0e-7,
    );
    assert_eq!(
        log.lines[3],
        "    Use all paths with cw amplitude ratio  12.50%"
    );
    assert_eq!(
        log.lines[4],
        "    S02  0.850  Temp  190.00  Debye temp  315.00  Global sig2  0.00123"
    );
    assert_eq!(
        log.lines[5],
        "Applying Debye-Waller factors using a Correlated Debye model."
    );
    assert_eq!(
        log.lines[6],
        "Done with module: XAS spectra (FF2X: DW + final sum over paths)."
    );
}

#[test]
fn ff2x_generation_module_log_reports_classical_debye_method_for_used_paths() {
    let mut input = sample_ff2x_input(1);
    input.control.idwopt = 3;
    let xsect = sample_xsect_handoff();

    let log = generated_ff2x_generation_module_log(&input, &xsect, 1);

    assert!(
        log.lines
            .iter()
            .any(|line| line == "Applying Debye-Waller factors using the Classical Debye model.")
    );
}

#[test]
fn ff2x_generation_module_log_omits_debye_method_without_used_paths() {
    let input = sample_ff2x_input(1);
    let xsect = sample_xsect_handoff();

    let log = generated_ff2x_generation_module_log(&input, &xsect, 0);

    assert!(
        !log.lines
            .iter()
            .any(|line| line.starts_with("Applying Debye-Waller factors"))
    );
}

#[test]
fn ff2x_prepared_paths_apply_imaginary_correction_to_amplitude() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    input.corrections.vicorr = 0.02 * FEFF_HARTREE_EV;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();

    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;

    assert_eq!(prepared.len(), 1);
    let expected_correction = ff2x_test_imaginary_correction(
        feff.complex_momentum[0],
        input.corrections.vicorr,
        feff.paths[0].effective_half_path_length_bohr,
    );
    assert_close(
        prepared[0].amplitude[0],
        2.0 * expected_correction * 4.0,
        1.0e-12,
    );
    assert_close(prepared[0].phase[0], -0.1, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_prepared_paths_apply_debye_and_cumulant_phase() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.thetae = 400.0;
    input.corrections.s02 = 0.8;
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();

    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;

    assert_eq!(prepared.len(), 1);
    let path = &prepared[0];
    let cumulants = path
        .damping
        .cumulants
        .context("single-scattering path should have cumulants")?;
    let sigma2_bohr2 = path.damping.total_sigma2_angstrom2 / FEFF_BIN_BOHR.powi(2);
    let dw = ff2x_test_dw_factor(feff.complex_momentum[0], sigma2_bohr2, cumulants);
    assert_close(
        path.amplitude[0],
        feff.paths[0].amplitude[0] * dw.norm() * input.corrections.s02 * feff.paths[0].degeneracy,
        1.0e-12,
    );
    assert_close(
        path.phase[0],
        feff.paths[0].phase[0] + dw.im.atan2(dw.re),
        1.0e-12,
    );
    Ok(())
}

#[test]
fn ff2x_path_sum_interpolates_and_accumulates_prepared_paths() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let mut prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    let mut second = prepared[0].clone();
    second.damping.path_index = 23;
    second.amplitude = second.amplitude.mapv(|value| value * 0.5);
    second.phase = second.phase.mapv(|value| value + 0.25);
    prepared.push(second);
    let output_momentum = Array1::from_vec(vec![0.55]);

    let summed = ff2x_sum_prepared_paths(&feff, &prepared, output_momentum.view())?;

    assert_eq!(summed.paths.len(), 2);
    assert_eq!(summed.paths[0].path_index, 17);
    assert_eq!(summed.paths[1].path_index, 23);
    let reff = feff.paths[0].effective_half_path_length_bohr;
    let first = expected_ff2x_path_signal(8.2, -0.15, 0.55, reff);
    let second = expected_ff2x_path_signal(4.1, 0.10, 0.55, reff);
    assert_complex_close(summed.paths[0].signal[0], first, 1.0e-12);
    assert_complex_close(summed.paths[1].signal[0], second, 1.0e-12);
    assert_complex_close(summed.total[0], first + second, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_module_writes_sig3_cum_dat_before_missing_source_error() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.thetae = 400.0;
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();
    let expected_damping = ff2x_path_damping(&input, &feff, &list)?;
    let expected_cumulants = expected_damping[0]
        .cumulants
        .context("single-scattering path should have cumulants")?;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("FF2X should require xsect.dat before final-spectrum generation")?;

    assert!(error.to_string().contains(
        "FF2X spectrum generation requires cached final-spectrum output or xsect.dat/feff.bin/list.dat source handoffs"
    ));
    let cum = read_cum_dat(temp.path().join("cum.dat"))?;
    assert_eq!(cum.einstein_temperature, 400.0);
    assert_eq!(cum.thermal_expansion, 0.034);
    assert_eq!(cum.entries.len(), 1);
    assert_eq!(cum.entries[0].path_index, 17);
    assert_close(
        cum.entries[0].first_cumulant_angstrom,
        expected_cumulants.first_cumulant_bohr * FEFF_BIN_BOHR,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].sigma2_angstrom2,
        expected_damping[0].total_sigma2_angstrom2,
        5.0e-6,
    );
    assert_close(
        cum.entries[0].third_cumulant_angstrom3,
        expected_cumulants.third_cumulant_bohr3 * FEFF_BIN_BOHR.powi(3),
        5.0e-8,
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_uncached_exafs_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    let xsect_dat = sample_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect_dat)?;

    assert!(has_cached_ff2x_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(has_cached_ff2x_output(temp.path())?);
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let path_headers = read_chi_path_headers(temp.path().join("chi.dat"))?;
    let path_header_line = chi
        .header_lines
        .iter()
        .find(|line| {
            line.trim_start_matches('#')
                .split_whitespace()
                .next()
                .is_some_and(|token| token == "17")
        })
        .context("chi.dat should include a FEFF path-summary header")?;
    assert_eq!(chi.point_count(), 8);
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(chi.header_lines[0].starts_with("# # Cu test"));
    assert!(chi.header_lines[0].contains("FEFF 10.0.0"));
    assert_eq!(chi.header_lines[0], xmu.header_lines[0]);
    assert!(
        chi.header_lines
            .iter()
            .any(|line| line.contains("S02=") && line.contains("Debye_temp="))
    );
    assert_eq!(path_headers.len(), 1);
    assert_eq!(path_headers[0].path_index, 17);
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line.as_str() == path_header_line.as_str())
    );
    assert!(xmu.normalization.is_some());
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, 1)
    );
    Ok(())
}

#[test]
fn ff2x_module_recovers_malformed_chi_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    let xsect_dat = sample_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect_dat)?;
    std::fs::write(temp.path().join("chi.dat"), b"not a chi.dat cache\n")?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(chi.point_count(), 8);
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(
        chi.header_lines
            .iter()
            .any(|line| line.contains("S02=") && line.contains("Debye_temp="))
    );
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, 1)
    );
    Ok(())
}

#[test]
fn ff2x_module_regenerates_stale_readable_exafs_outputs_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    let xsect_dat = sample_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect_dat)?;
    run_in_dir(temp.path())?;
    let expected_chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let mut stale_chi = expected_chi.clone();
    stale_chi.chi[0] += 0.25;
    write_chi_dat(temp.path().join("chi.dat"), &stale_chi)?;

    assert!(has_cached_ff2x_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_eq!(read_chi_dat(temp.path().join("chi.dat"))?, expected_chi);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, 1)
    );
    Ok(())
}

#[test]
fn ff2x_module_regenerates_stale_readable_xanes_output_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    run_in_dir(temp.path())?;
    let expected_xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let mut stale_xmu = expected_xmu.clone();
    stale_xmu.mu[0] += 0.25;
    write_xmu_dat(temp.path().join("xmu.dat"), &stale_xmu)?;

    assert!(has_cached_ff2x_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_xmu_dat(temp.path().join("xmu.dat"))?, expected_xmu);
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), input.corrections.s02, 0)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, 0)
    );
    Ok(())
}

#[test]
fn ff2x_module_regenerates_stale_readable_nrixs_xmul_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    input.decomposition_channels = 1;
    write_ff2x_input_data(temp.path(), &input)?;
    write_global_input(temp.path(), 1, 1)?;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_feffl_bin(
        temp.path().join("feffl.bin"),
        &sample_feffl_bin_data(prepared.len(), feff.energy_count(), 1),
    )?;
    write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_xsecl_bin_data(sample_xanes_xsect_dat().energy_count()),
    )?;
    run_in_dir(temp.path())?;
    let expected_xmul = read_xmul_dat(temp.path().join("xmul.dat"))?;
    let mut stale_xmul = expected_xmul.clone();
    stale_xmul.channel_background[(0, 0)] += 0.25;
    write_xmul_dat(temp.path().join("xmul.dat"), &stale_xmul)?;

    assert!(has_cached_ff2x_output(temp.path())?);
    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert_eq!(read_xmul_dat(temp.path().join("xmul.dat"))?, expected_xmul);
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), input.corrections.s02, 0)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, prepared.len())
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_active_hubbard_source_handoff_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    let xsect_dat = sample_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect_dat)?;
    write_active_hubbard_input(temp.path())?;

    assert!(has_cached_ff2x_output(temp.path())?);

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(chi.point_count(), 8);
    assert_eq!(xmu.point_count(), chi.point_count());
    assert!(chi.header_lines[0].starts_with("# # Cu test"));
    assert!(xmu.normalization.is_some());
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &xsect, 1)
    );
    Ok(())
}

#[test]
fn ff2x_module_ignores_gamma_channel_for_regular_exafs() -> Result<()> {
    let baseline = tempfile::tempdir()?;
    let gamma = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    let mut gamma_input = input;
    gamma_input.control.i_gamma_ch = 1;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let xsect = sample_xsect_dat();

    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(baseline.path().join("feff.bin"), &feff)?;
    write_list_dat(baseline.path().join("list.dat"), &list)?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect)?;
    write_ff2x_input_data(gamma.path(), &gamma_input)?;
    write_feff_bin(gamma.path().join("feff.bin"), &feff)?;
    write_list_dat(gamma.path().join("list.dat"), &list)?;
    write_xsect_dat(gamma.path().join("xsect.dat"), &xsect)?;

    let baseline_count = run_in_dir(baseline.path())?;
    let gamma_count = run_in_dir(gamma.path())?;

    assert_eq!(gamma_count, baseline_count);
    assert_eq!(
        read_chi_dat(gamma.path().join("chi.dat"))?,
        read_chi_dat(baseline.path().join("chi.dat"))?
    );
    assert_eq!(
        read_xmu_dat(gamma.path().join("xmu.dat"))?,
        read_xmu_dat(baseline.path().join("xmu.dat"))?
    );
    assert_eq!(
        read_module_log_dat(gamma.path().join("log6.dat"))?,
        read_module_log_dat(baseline.path().join("log6.dat"))?
    );
    Ok(())
}

#[test]
fn ff2x_module_uses_regular_exafs_when_global_disables_nrixs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.decomposition_channels = 2;
    write_ff2x_input_data(temp.path(), &input)?;
    write_global_input(temp.path(), 0, 2)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn ff2x_module_routes_nondecomposed_nrixs_non_xanes_through_ms_expansion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = sample_ff2x_input(1);
    write_ff2x_input_data(temp.path(), &input)?;
    write_global_input(temp.path(), 1, -1)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("xmul.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_generates_decomposed_nrixs_xmul_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    input.decomposition_channels = 1;
    write_ff2x_input_data(temp.path(), &input)?;
    write_global_input(temp.path(), 1, 1)?;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_feffl_bin(
        temp.path().join("feffl.bin"),
        &sample_feffl_bin_data(prepared.len(), feff.energy_count(), 1),
    )?;
    write_xsecl_bin(
        temp.path().join("xsecl.bin"),
        &sample_xsecl_bin_data(sample_xanes_xsect_dat().energy_count()),
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let xmul = read_xmul_dat(temp.path().join("xmul.dat"))?;
    assert_eq!(
        xmul.point_count(),
        sample_xanes_xsect_dat().main_energy_count
    );
    assert_eq!(xmul.channel_count(), 2);
    let raw_channel_0 = 0.2 / FEFF_HARTREE_EV;
    let raw_channel_1 = 0.4 / FEFF_HARTREE_EV;
    assert!(xmul.channel_background[(0, 0)].is_finite());
    assert!(xmul.channel_background[(0, 1)].is_finite());
    assert!(xmul.channel_background[(0, 0)] > 0.0);
    assert!(xmul.channel_background[(0, 1)] > 0.0);
    assert!((xmul.channel_background[(0, 0)] - raw_channel_0).abs() > 1.0e-4);
    assert!((xmul.channel_background[(0, 1)] - raw_channel_1).abs() > 1.0e-4);
    assert_close(
        xmul.total_single_electron[0],
        xmul.channel_background[(0, 0)] + xmul.channel_background[(0, 1)],
        1.0e-4,
    );
    assert!(
        xmul.normalized_fine_structure
            .iter()
            .any(|value| value.abs() > 1.0e-8)
    );
    assert!(temp.path().join("log6.dat").is_file());
    assert!(!temp.path().join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn ff2x_module_generates_gecl4_nrixs_xmul_matching_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_gecl4_dir()? else {
        crate::require_fixture!(
            "NRIXS GeCl4 generation reference test; reference handoffs not found"
        );
    };
    let required = [
        "ff2x.inp",
        "global.inp",
        "feff.bin",
        "feffl.bin",
        "list.dat",
        "xsect.dat",
        "xsecl.bin",
        "fms.bin",
        "fmsl.bin",
        "xmul.dat",
    ];
    if !required
        .iter()
        .all(|name| reference_dir.join(name).is_file())
    {
        crate::require_fixture!("NRIXS GeCl4 generation reference test; handoffs not found");
    }
    let temp = tempfile::tempdir()?;
    for name in [
        "ff2x.inp",
        "global.inp",
        "feff.bin",
        "feffl.bin",
        "list.dat",
        "xsect.dat",
        "xsecl.bin",
        "fms.bin",
        "fmsl.bin",
    ] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chi.dat").is_file());
    assert!(!temp.path().join("xmu.dat").is_file());
    assert_xmul_dat_close(
        &read_xmul_dat(temp.path().join("xmul.dat"))?,
        &read_xmul_dat(reference_dir.join("xmul.dat"))?,
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_nondecomposed_nrixs_xmu_from_source_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_global_input(temp.path(), 1, -1)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("xmul.dat").exists());
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(
        xmu.point_count(),
        sample_xanes_xsect_dat().main_energy_count
    );
    assert_eq!(xmu.normalization, None);
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "# Contribution to S(q,w) from a single electron")
    );
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
        assert_close(xmu.mu0[row], 10.0 * step / FEFF_HARTREE_EV, 2.0e-6);
        assert_close(xmu.chi[row], 2.0 * step / FEFF_HARTREE_EV, 2.0e-6);
        assert_close(xmu.mu[row], 12.0 * step / FEFF_HARTREE_EV, 2.0e-6);
    }
    Ok(())
}

#[test]
fn ff2x_module_routes_positive_other_ispec_through_ms_expansion() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 9;

    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    assert_eq!(
        read_chi_dat(temp.path().join("chi.dat"))?.point_count(),
        read_xmu_dat(temp.path().join("xmu.dat"))?.point_count()
    );
    Ok(())
}

#[test]
fn ff2x_module_ignores_standalone_decomposition_channels_without_nrixs_global() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.decomposition_channels = 2;

    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    Ok(())
}

#[test]
fn ff2x_module_writes_chia_bin_for_configuration_average_first_absorber() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let xsect = sample_xsect_dat();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 2;

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("chia.bin").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    assert!(!temp.path().join("xmu.dat").exists());

    let momentum_grid = ff2x_momentum_grid(&input, &feff)?;
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    let expected = ff2x_sum_prepared_paths(
        &feff,
        &prepared,
        momentum_grid.interpolation_momentum.view(),
    )?;

    let chia = read_chia_bin(temp.path().join("chia.bin"))?;
    assert_eq!(chia.values.len(), expected.total.len());
    for (actual, expected) in chia.values.iter().zip(expected.total.iter()) {
        assert_complex_close(*actual, *expected / 2.0, 1.0e-12);
    }
    let log = read_module_log_dat(temp.path().join("log6.dat"))?;
    assert!(log.lines.iter().any(|line| {
        line == "Done with module: XAS spectra (FF2X: DW + final sum over paths)."
    }));
    Ok(())
}

#[test]
fn ff2x_module_finalizes_two_absorber_configuration_average_from_chia_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let xsect = sample_xsect_dat();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 2;

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let momentum_grid = ff2x_momentum_grid(&input, &feff)?;
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    let current = ff2x_sum_prepared_paths(
        &feff,
        &prepared,
        momentum_grid.interpolation_momentum.view(),
    )?;
    let prior = current
        .total
        .iter()
        .enumerate()
        .map(|(index, _)| Complex64::new(0.05 * index as f64, -0.025 * index as f64))
        .collect::<Vec<_>>();
    write_chia_bin(
        temp.path().join("chia.bin"),
        &ChiaBinData {
            values: prior.clone(),
        },
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(!temp.path().join("chia.bin").exists());
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());

    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    assert_eq!(chi.point_count(), current.total.len());
    for (row, (&prior, &current)) in prior
        .iter()
        .zip(current.total.iter())
        .enumerate()
        .take(chi.point_count())
    {
        let expected = prior + current / 2.0;
        assert_close(chi.chi[row], expected.im, 1.0e-6);
        assert_close(chi.magnitude[row], expected.norm(), 1.0e-6);
    }
    Ok(())
}

#[test]
fn ff2x_module_accumulates_three_absorber_configuration_average_with_state() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.debye.tk = 0.0;
    let feff = sample_feff_bin_data();
    let list = sample_list_dat();
    let xsect = sample_xsect_dat();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 3;

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &list)?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let momentum_grid = ff2x_momentum_grid(&input, &feff)?;
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    let current = ff2x_sum_prepared_paths(
        &feff,
        &prepared,
        momentum_grid.interpolation_momentum.view(),
    )?;
    let state_path = temp.path().join(".refeff-ff2x-cfaverage-state");

    let first_count = run_in_dir(temp.path())?;

    assert_eq!(first_count, 2);
    assert!(temp.path().join("chia.bin").is_file());
    assert!(state_path.is_file());
    assert!(!temp.path().join("chi.dat").exists());
    let first_chia = read_chia_bin(temp.path().join("chia.bin"))?;
    for (actual, expected) in first_chia.values.iter().zip(current.total.iter()) {
        assert_complex_close(*actual, *expected / 3.0, 1.0e-12);
    }

    let second_count = run_in_dir(temp.path())?;

    assert_eq!(second_count, 2);
    assert!(temp.path().join("chia.bin").is_file());
    assert!(state_path.is_file());
    assert!(!temp.path().join("chi.dat").exists());
    let second_chia = read_chia_bin(temp.path().join("chia.bin"))?;
    for (actual, expected) in second_chia.values.iter().zip(current.total.iter()) {
        assert_complex_close(*actual, *expected * (2.0 / 3.0), 1.0e-12);
    }

    let final_count = run_in_dir(temp.path())?;

    assert_eq!(final_count, 3);
    assert!(!temp.path().join("chia.bin").exists());
    assert!(!state_path.exists());
    assert!(temp.path().join("chi.dat").is_file());
    assert!(temp.path().join("xmu.dat").is_file());
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    assert_eq!(chi.point_count(), current.total.len());
    for (row, expected) in current.total.iter().enumerate().take(chi.point_count()) {
        assert_close(chi.chi[row], expected.im, 1.0e-6);
        assert_close(chi.magnitude[row], expected.norm(), 1.0e-6);
    }
    Ok(())
}

#[test]
fn ff2x_module_writes_xanes_chia_bin_from_fms_trace_for_configuration_average() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 3;

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("chia.bin").is_file());
    assert!(!temp.path().join("xmu.dat").exists());

    let chia = read_chia_bin(temp.path().join("chia.bin"))?;
    assert_eq!(
        chia.values.len(),
        sample_xanes_xsect_dat().energy_grid_ev.len()
    );
    for value in chia.values {
        assert_complex_close(value, Complex64::new(0.0, 2.0 / 3.0), 1.0e-12);
    }
    Ok(())
}

#[test]
fn ff2x_module_finalizes_xanes_configuration_average_from_chia_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let xsect = sample_xanes_xsect_dat();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 2;
    let prior_trace = Complex64::new(0.0, 4.0);

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    write_chia_bin(
        temp.path().join("chia.bin"),
        &ChiaBinData {
            values: vec![prior_trace; xsect.energy_grid_ev.len()],
        },
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chia.bin").exists());
    assert!(!temp.path().join("chi.dat").exists());

    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), xsect.main_energy_count);
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
        assert_close(xmu.mu0[row], 10.0 * step, 5.0e-5);
        assert_close(xmu.chi[row], 5.0 * step, 5.0e-5);
        assert_close(xmu.mu[row], 15.0 * step, 5.0e-5);
    }
    Ok(())
}

#[test]
fn ff2x_module_generates_xanes_xmu_from_fms_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), 3);
    assert!(xmu.header_lines[0].starts_with("# # Cu XANES test"));
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "#     0/   0 paths used")
    );
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
        assert_close(xmu.mu0[row], 10.0 * step, 5.0e-5);
        assert_close(xmu.chi[row], 2.0 * step, 5.0e-5);
        assert_close(xmu.mu[row], 12.0 * step, 5.0e-5);
    }
    Ok(())
}

#[test]
fn ff2x_module_generates_xanes_xmu_with_electronic_temperature() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    input.electronic_temperature = 0.5;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(
        temp.path().join("xsect.dat"),
        &sample_xanes_thermal_xsect_dat(),
    )?;
    write_fms_bin(
        temp.path().join("fms.bin"),
        &sample_xanes_thermal_fms_bin(2.0),
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), 3);
    let expected_mu0 = [
        2.585_850_230_196_276,
        5.000_003_995_527_989,
        7.414_157_694_791_554,
    ];
    for (row, &mu0) in expected_mu0.iter().enumerate() {
        assert_close(xmu.mu0[row], mu0, 5.0e-5);
        assert_close(xmu.chi[row], 0.2 * mu0, 5.0e-5);
        assert_close(xmu.mu[row], 1.2 * mu0, 5.0e-5);
    }
    Ok(())
}

#[test]
fn ff2x_module_rejects_xanes_mismatched_feff_and_xsect_grids() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.debye.tk = 0.0;
    let mut feff = sample_xanes_feff_bin_data();
    feff.real_momentum[1] += 0.1;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let error = run_in_dir(temp.path())
        .err()
        .context("mismatched XANES grids should be rejected")?;

    assert!(
        error
            .to_string()
            .contains("FF2X XANES Emesh in feff.bin and xsect.dat different")
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_xanes_path_only_output_without_fms_handoff() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(
        temp.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(temp.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(
        xmu.point_count(),
        sample_xanes_xsect_dat().main_energy_count
    );
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "#     1/   1 paths used")
    );
    assert!(
        xmu.chi.iter().any(|value| value.abs() > 1.0e-8),
        "path-only XANES generation should include path fine structure"
    );
    for row in 0..xmu.point_count() {
        assert_close(xmu.mu[row], xmu.mu0[row] + xmu.chi[row], 1.0e-9);
    }
    Ok(())
}

#[test]
fn ff2x_module_generates_xanes_outputs_for_eels_polarizations() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_eels_polarization_input(temp.path(), 5, 4, 9)?;
    write_feff_bin(
        temp.path().join("feff05.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(temp.path().join("list05.dat"), &sample_empty_list_dat())?;
    write_feff_bin(
        temp.path().join("feff09.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(temp.path().join("list09.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(
        temp.path().join("fms.bin"),
        &sample_xanes_fms_bin_for_polarization_offsets(5),
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert!(!temp.path().join("xmu.dat").exists());
    assert!(temp.path().join("xmu05.dat").is_file());
    assert!(temp.path().join("xmu09.dat").is_file());
    let xmu05 = read_xmu_dat(temp.path().join("xmu05.dat"))?;
    let xmu09 = read_xmu_dat(temp.path().join("xmu09.dat"))?;
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
        assert_close(xmu05.chi[row], step, 5.0e-5);
        assert_close(xmu09.chi[row], 5.0 * step, 5.0e-5);
        assert_close(xmu05.mu[row], xmu05.mu0[row] + xmu05.chi[row], 1.0e-9);
        assert_close(xmu09.mu[row], xmu09.mu0[row] + xmu09.chi[row], 1.0e-9);
    }
    Ok(())
}

#[test]
fn ff2x_module_routes_negative_ispec_through_regular_xanes() -> Result<()> {
    let baseline = tempfile::tempdir()?;
    let negative = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.control.i_gamma_ch = 1;
    input.debye.tk = 0.0;
    let mut xsect = sample_xanes_xsect_dat();
    xsect.core_hole_width_ev = 2.0;

    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(
        baseline.path().join("feff.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(baseline.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let mut negative_input = input;
    negative_input.control.ispec = -1;
    write_ff2x_input_data(negative.path(), &negative_input)?;
    write_feff_bin(
        negative.path().join("feff.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(negative.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(negative.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(negative.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let baseline_count = run_in_dir(baseline.path())?;
    let negative_count = run_in_dir(negative.path())?;

    assert_eq!(negative_count, baseline_count);
    assert!(!negative.path().join("chi.dat").exists());
    assert_eq!(
        read_xmu_dat(negative.path().join("xmu.dat"))?,
        read_xmu_dat(baseline.path().join("xmu.dat"))?
    );
    assert_eq!(
        read_module_log_dat(negative.path().join("log6.dat"))?,
        read_module_log_dat(baseline.path().join("log6.dat"))?
    );
    Ok(())
}

#[test]
fn ff2x_module_routes_negative_danes_ispec_through_ff2xmu() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = -3;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(
        xmu.point_count(),
        sample_xanes_xsect_dat().main_energy_count
    );
    assert!(xmu.mu.iter().all(|value| value.is_finite()));
    assert!(xmu.mu0.iter().all(|value| value.is_finite()));
    assert!(xmu.chi.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn ff2x_module_routes_ispec_two_through_regular_xanes() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 2;
    input.debye.tk = 0.0;
    let xsect = sample_xanes_mbconv_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), 3);
    assert_close(
        xmu.normalization
            .context("ispec=2 XANES should write normalization")?,
        xsect.normalized_background[0],
        5.0e-5,
    );
    Ok(())
}

#[test]
fn ff2x_module_applies_siggk_to_generated_xanes_chi() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    input.debye.sig_gk = 0.25;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    run_in_dir(temp.path())?;

    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let feff = sample_xanes_feff_bin_data();
    for (row, &energy) in [0.9_f64, 1.0, 1.1].iter().enumerate() {
        let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
        let k_inverse_angstrom = feff.real_momentum[row] / FEFF_BIN_BOHR;
        let damping = (-(input.debye.sig_gk * k_inverse_angstrom).powi(2)).exp();
        assert_close(xmu.mu0[row], 10.0 * step, 5.0e-5);
        assert_close(xmu.chi[row], 2.0 * step * damping, 5.0e-5);
        assert_close(xmu.mu[row], xmu.mu0[row] + xmu.chi[row], 1.0e-9);
    }
    Ok(())
}

#[test]
fn ff2x_xanes_combined_trace_adds_paths_on_main_grid_only() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;
    let fms_trace = Array1::from_vec(vec![
        Complex64::new(0.0, 10.0),
        Complex64::new(1.0, 11.0),
        Complex64::new(2.0, 12.0),
        Complex64::new(3.0, 13.0),
        Complex64::new(4.0, 14.0),
        Complex64::new(5.0, 15.0),
    ]);
    let path_sum = Ff2xPathSum {
        total: Array1::from_vec(vec![
            Complex64::new(0.5, 1.5),
            Complex64::new(0.6, 1.6),
            Complex64::new(0.7, 1.7),
        ]),
        paths: Vec::new(),
    };

    let combined = ff2x_xanes_combined_trace(&xsect, fms_trace.view(), &path_sum)?;

    assert_eq!(combined.len(), fms_trace.len());
    assert_eq!(combined[0], Complex64::new(0.5, 11.5));
    assert_eq!(combined[1], Complex64::new(1.6, 12.6));
    assert_eq!(combined[2], Complex64::new(2.7, 13.7));
    assert_eq!(combined[3], fms_trace[3]);
    assert_eq!(combined[4], fms_trace[4]);
    assert_eq!(combined[5], fms_trace[5]);
    Ok(())
}

#[test]
fn ff2x_nrixs_combined_decomposed_trace_adds_paths_on_main_grid_only() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 0.5, 0)?;
    let fmsl = sample_fmsl_bin_data(6, 1);
    let path_sum = Ff2xDecomposedPathSum {
        total: Array3::from_shape_fn((3, 2, 2), |(row, lg2, lg1)| {
            Complex64::new(
                0.25 + row as f64 + lg2 as f64 * 0.1 + lg1 as f64 * 0.01,
                -0.5 - row as f64 - lg2 as f64 * 0.2 - lg1 as f64 * 0.02,
            )
        }),
        paths: Vec::new(),
    };

    let combined = ff2x_nrixs_combined_decomposed_trace(&xsect, &fmsl, &path_sum)?;

    assert_eq!(combined.shape(), &[6, 2, 2]);
    for row in 0..6 {
        for lg2 in 0..2 {
            for lg1 in 0..2 {
                let mut expected = fmsl.traces[(row, lg2, lg1)] * xsect.amplitude_reduction;
                if row < xsect.main_energy_count {
                    expected += path_sum.total[(row, lg2, lg1)];
                }
                assert_complex_close(combined[(row, lg2, lg1)], expected, 1.0e-12);
            }
        }
    }
    Ok(())
}

#[test]
fn ff2x_nrixs_combined_decomposed_trace_rejects_shape_mismatch() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;
    let fmsl = sample_fmsl_bin_data(6, 1);
    let path_sum = Ff2xDecomposedPathSum {
        total: Array3::zeros((2, 2, 2)),
        paths: Vec::new(),
    };

    let error = ff2x_nrixs_combined_decomposed_trace(&xsect, &fmsl, &path_sum)
        .err()
        .context("mismatched decomposed path-sum shape should be rejected")?;

    assert!(
        error
            .to_string()
            .contains("FF2X NRIXS decomposed path-sum shape")
    );
    Ok(())
}

#[test]
fn ff2x_nrixs_optional_fmsl_trace_returns_zero_matrix_when_absent() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;

    let fmsl = ff2x_nrixs_optional_fmsl_trace(temp.path(), &xsect, 1)?;

    assert_eq!(fmsl.pad_width, FMS_BIN_DEFAULT_PAD_WIDTH);
    assert_eq!(fmsl.max_decomposition_channel, 1);
    assert_eq!(fmsl.traces.shape(), &[xsect.energy_count(), 2, 2]);
    assert!(
        fmsl.traces
            .iter()
            .all(|value| *value == Complex64::new(0.0, 0.0))
    );
    Ok(())
}

#[test]
fn ff2x_nrixs_optional_fmsl_trace_reads_gecl4_reference_handoff() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_gecl4_dir()? else {
        crate::require_fixture!("NRIXS GeCl4 reference test; reference handoffs not found");
    };
    let xsect_dat = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, 1.0, 0)?;
    let gtrl = read_gtrl_dat(reference_dir.join("gtrl.dat"))?;

    let fmsl = ff2x_nrixs_optional_fmsl_trace(&reference_dir, &xsect, 2)?;

    assert_eq!(fmsl.traces.shape(), &[xsect.energy_count(), 3, 3]);
    assert_eq!(gtrl.row_count(), xsect.energy_count());
    assert_eq!(gtrl.component_count(), 9);
    for row in [0, xsect.fermi_index, xsect.main_energy_count - 1] {
        for lg1 in 0..3 {
            for lg2 in 0..3 {
                let component = lg1 * 3 + lg2;
                assert_complex_close(
                    fmsl.traces[(row, lg2, lg1)],
                    gtrl.decomposed_trace[(row, component)],
                    5.0e-8,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn ff2x_nrixs_channel_background_ignores_channels_above_ldecmx() -> Result<()> {
    let xsecl = XseclBinData {
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        initial_state_j: 1,
        transitions: vec![
            XseclBinTransition {
                final_state_kappa: -1,
                decomposition_channel: 0,
                total_angular_momentum_channel: 0,
                orbital_angular_momentum: 0,
            },
            XseclBinTransition {
                final_state_kappa: -2,
                decomposition_channel: 3,
                total_angular_momentum_channel: 3,
                orbital_angular_momentum: 3,
            },
            XseclBinTransition {
                final_state_kappa: 2,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
        ],
        atom_cross_sections: Array2::from_shape_vec(
            (2, 3),
            vec![
                Complex64::new(0.0, 2.0 * FEFF_HARTREE_EV),
                Complex64::new(0.0, 100.0 * FEFF_HARTREE_EV),
                Complex64::new(0.0, 4.0 * FEFF_HARTREE_EV),
                Complex64::new(0.0, 3.0 * FEFF_HARTREE_EV),
                Complex64::new(0.0, 200.0 * FEFF_HARTREE_EV),
                Complex64::new(0.0, 5.0 * FEFF_HARTREE_EV),
            ],
        )?,
        raw_atom_cross_section_pad: None,
    };

    let background = ff2x_nrixs_channel_background_from_xsecl(&xsecl, 1, 2)?;

    assert_eq!(background.shape(), &[2, 2]);
    assert_close(background[(0, 0)], 2.0, 1.0e-12);
    assert_close(background[(0, 1)], 4.0, 1.0e-12);
    assert_close(background[(1, 0)], 3.0, 1.0e-12);
    assert_close(background[(1, 1)], 5.0, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_nrixs_total_single_electron_response_sums_channel_backgrounds() -> Result<()> {
    let channel_background = Array2::from_shape_vec(
        (3, 3),
        vec![0.25, 1.5, 0.75, 2.0, 0.125, 0.875, 4.0, 8.0, 16.0],
    )?;

    let total = ff2x_nrixs_total_single_electron_response(channel_background.view())?;

    assert_eq!(total.len(), channel_background.nrows());
    assert_close(total[0], 2.5, 1.0e-12);
    assert_close(total[1], 3.0, 1.0e-12);
    assert_close(total[2], 28.0, 1.0e-12);
    Ok(())
}

#[test]
fn ff2x_nrixs_total_single_electron_response_rejects_invalid_channel_background() -> Result<()> {
    let empty_rows = Array2::<f64>::zeros((0, 2));
    let error = ff2x_nrixs_total_single_electron_response(empty_rows.view())
        .err()
        .context("empty NRIXS background rows should be rejected")?;
    assert!(
        error
            .to_string()
            .contains("total response requires at least one energy row")
    );

    let empty_channels = Array2::<f64>::zeros((2, 0));
    let error = ff2x_nrixs_total_single_electron_response(empty_channels.view())
        .err()
        .context("empty NRIXS background channels should be rejected")?;
    assert!(
        error
            .to_string()
            .contains("total response requires at least one channel column")
    );

    let channel_background = Array2::from_shape_vec((1, 3), vec![0.25, f64::NAN, 0.75])?;
    let error = ff2x_nrixs_total_single_electron_response(channel_background.view())
        .err()
        .context("non-finite NRIXS background should be rejected")?;
    assert!(
        error
            .to_string()
            .contains("channel background row 0 channel 1 is not finite")
    );
    Ok(())
}

#[test]
fn ff2x_nrixs_total_single_electron_response_matches_gecl4_xmul_reference() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_gecl4_dir()? else {
        crate::require_fixture!("NRIXS GeCl4 reference test; reference handoffs not found");
    };
    let xmul = read_xmul_dat(reference_dir.join("xmul.dat"))?;

    let total = ff2x_nrixs_total_single_electron_response(xmul.channel_background.view())?;

    assert_eq!(total.len(), xmul.point_count());
    for row in 0..xmul.point_count() {
        assert_close(total[row], xmul.total_single_electron[row], 1.1e-7);
    }
    Ok(())
}

#[test]
fn ff2x_nrixs_xmul_output_grid_uses_xsect_main_grid() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;

    let (photon_energy_ev, wave_number) = ff2x_nrixs_xmul_output_grid(&xsect)?;

    assert_eq!(photon_energy_ev.len(), xsect.main_energy_count);
    assert_eq!(wave_number.len(), xsect.main_energy_count);
    for row in 0..xsect.main_energy_count {
        assert_close(
            photon_energy_ev[row],
            xsect.omega_hartree[row] * FEFF_HARTREE_EV,
            1.0e-12,
        );
        assert_close(
            wave_number[row],
            xsect.wave_number[row] / FEFF_BOHR_ANGSTROM,
            1.0e-12,
        );
    }
    Ok(())
}

#[test]
fn ff2x_nrixs_xmul_output_grid_rejects_invalid_handoff() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;

    let mut zero_main = xsect.clone();
    zero_main.main_energy_count = 0;
    let error = ff2x_nrixs_xmul_output_grid(&zero_main)
        .err()
        .context("zero main-grid rows should be rejected")?;
    assert!(
        error
            .to_string()
            .contains("xmul grid requires at least one main energy row")
    );

    let mut too_many_main = xsect.clone();
    too_many_main.main_energy_count = too_many_main.energy_count() + 1;
    let error = ff2x_nrixs_xmul_output_grid(&too_many_main)
        .err()
        .context("oversized main-grid rows should be rejected")?;
    assert!(error.to_string().contains("exceeds xsect.dat energy count"));

    let mut short_omega = xsect.clone();
    short_omega.omega_hartree = Array1::zeros(1);
    let error = ff2x_nrixs_xmul_output_grid(&short_omega)
        .err()
        .context("omega length mismatch should be rejected")?;
    assert!(error.to_string().contains("omega length"));

    let mut nonfinite_wave_number = xsect.clone();
    nonfinite_wave_number.wave_number[1] = f64::INFINITY;
    let error = ff2x_nrixs_xmul_output_grid(&nonfinite_wave_number)
        .err()
        .context("non-finite wave number should be rejected")?;
    assert!(
        error
            .to_string()
            .contains("source wave number row 1 is not finite")
    );
    Ok(())
}

#[test]
fn ff2x_nrixs_xmul_dat_from_components_builds_grid_and_totals() -> Result<()> {
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;
    let channel_background = Array2::from_shape_vec(
        (xsect.main_energy_count, 2),
        vec![0.25, 1.75, 0.5, 2.25, 4.0, 8.0],
    )?;
    let normalized_fine_structure =
        Array3::from_shape_fn((xsect.main_energy_count, 2, 2), |(row, lstar, l)| {
            0.1 * row as f64 + 0.01 * lstar as f64 + 0.001 * l as f64
        });

    let data = ff2x_nrixs_xmul_dat_from_components(Ff2xNrixsXmulComponents {
        header_lines: &[],
        max_decomposition_channel: 1,
        xsect: &xsect,
        channel_background: channel_background.view(),
        normalized_fine_structure: normalized_fine_structure.view(),
    })?;

    assert_eq!(data.point_count(), xsect.main_energy_count);
    assert_eq!(data.channel_count(), 2);
    assert_close(data.total_single_electron[0], 2.0, 1.0e-12);
    assert_close(data.total_single_electron[1], 2.75, 1.0e-12);
    assert_close(data.total_single_electron[2], 12.0, 1.0e-12);
    assert_close(
        data.photon_energy_ev[0],
        xsect.omega_hartree[0] * FEFF_HARTREE_EV,
        1.0e-12,
    );
    assert_close(
        data.wave_number[0],
        xsect.wave_number[0] / FEFF_BOHR_ANGSTROM,
        1.0e-12,
    );
    assert_eq!(data.channel_background, channel_background);
    assert_eq!(data.normalized_fine_structure, normalized_fine_structure);
    Ok(())
}

#[test]
fn ff2x_nrixs_xmul_output_grid_matches_gecl4_reference() -> Result<()> {
    let Some(reference_dir) = reference_nrixs_gecl4_dir()? else {
        crate::require_fixture!("NRIXS GeCl4 reference test; reference handoffs not found");
    };
    let xsect_dat = read_xsect_dat(reference_dir.join("xsect.dat"))?;
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, 1.0, 0)?;
    let xmul = read_xmul_dat(reference_dir.join("xmul.dat"))?;

    let (photon_energy_ev, wave_number) = ff2x_nrixs_xmul_output_grid(&xsect)?;

    assert_eq!(photon_energy_ev.len(), xmul.point_count());
    assert_eq!(wave_number.len(), xmul.point_count());
    for row in 0..xmul.point_count() {
        assert_close(photon_energy_ev[row], xmul.photon_energy_ev[row], 1.0e-3);
        assert_close(wave_number[row], xmul.wave_number[row], 1.0e-3);
    }
    Ok(())
}

#[test]
fn ff2x_xanes_corrected_background_applies_mbconv_on_main_grid_only() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.mbconv = 1;
    let xsect_dat = sample_xanes_mbconv_xsect_dat();
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, input.control.mbconv)?;
    let feff = sample_xanes_feff_bin_data();
    let grid = ff2x_momentum_grid(&input, &feff)?;
    let output_energy = ff2x_output_energy_grid(feff.edge_energy, &xsect, &grid)?;

    let actual = ff2x_xanes_corrected_background(&input, &xsect, &output_energy)?;

    let energy = Array1::from_iter(
        xsect
            .omega_hartree
            .iter()
            .take(xsect.main_energy_count)
            .copied(),
    );
    let background = Array1::from_iter(
        xsect
            .normalized_background
            .iter()
            .take(xsect.main_energy_count)
            .copied(),
    );
    let expected = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: energy.view(),
        xmu: background.view(),
        fermi_energy: xsect.omega_hartree[0],
        amplitude_reduction: xsect.file_amplitude_reduction,
        relaxation_energy: xsect.relaxation_energy,
        plasmon_frequency: xsect.plasmon_frequency * 0.5,
    })?;
    for row in 0..xsect.main_energy_count {
        assert_close(actual[row], expected[row], 1.0e-12);
    }
    for row in xsect.main_energy_count..xsect.energy_count() {
        assert_close(actual[row], xsect.normalized_background[row], 1.0e-12);
    }

    let fms_trace = ff2x_xanes_fms_trace(&xsect, &sample_xanes_fms_bin(2.0))?;
    assert_close(
        fms_trace[0].im,
        2.0 * xsect.file_amplitude_reduction,
        1.0e-12,
    );
    Ok(())
}

#[test]
fn ff2x_module_adds_xanes_path_list_contributions_to_fms_trace() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;

    let baseline = tempfile::tempdir()?;
    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(
        baseline.path().join("feff.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(baseline.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    run_in_dir(baseline.path())?;
    let baseline_xmu = read_xmu_dat(baseline.path().join("xmu.dat"))?;

    let with_paths = tempfile::tempdir()?;
    write_ff2x_input_data(with_paths.path(), &input)?;
    write_feff_bin(
        with_paths.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(with_paths.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(
        with_paths.path().join("xsect.dat"),
        &sample_xanes_xsect_dat(),
    )?;
    write_fms_bin(
        with_paths.path().join("fms.bin"),
        &sample_xanes_fms_bin(2.0),
    )?;

    let count = run_in_dir(with_paths.path())?;

    assert_eq!(count, 2);
    assert!(!with_paths.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(with_paths.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), baseline_xmu.point_count());
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line == "#     1/   1 paths used")
    );
    assert!(
        xmu.chi
            .iter()
            .zip(baseline_xmu.chi.iter())
            .any(|(&actual, &baseline)| (actual - baseline).abs() > 1.0e-6),
        "XANES path-list contribution should change at least one chi point"
    );
    for row in 0..xmu.point_count() {
        assert_close(xmu.mu[row], xmu.mu0[row] + xmu.chi[row], 1.0e-9);
    }
    Ok(())
}

#[test]
fn ff2x_module_generates_mbconv_xanes_outputs_from_handoffs() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let xsect = sample_xanes_mbconv_xsect_dat();

    let baseline = tempfile::tempdir()?;
    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(
        baseline.path().join("feff.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(baseline.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    run_in_dir(baseline.path())?;
    let baseline_xmu = read_xmu_dat(baseline.path().join("xmu.dat"))?;

    let mbconv = tempfile::tempdir()?;
    input.control.mbconv = 1;
    write_ff2x_input_data(mbconv.path(), &input)?;
    write_feff_bin(
        mbconv.path().join("feff.bin"),
        &sample_xanes_feff_bin_data(),
    )?;
    write_list_dat(mbconv.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(mbconv.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(mbconv.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(mbconv.path())?;

    assert_eq!(count, 2);
    assert!(!mbconv.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(mbconv.path().join("xmu.dat"))?;
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line.contains("S02=0.700"))
    );
    assert!(
        xmu.chi
            .iter()
            .zip(baseline_xmu.chi.iter())
            .any(|(&actual, &baseline)| (actual - baseline).abs() > 1.0e-6),
        "XANES mbconv should change at least one chi point"
    );
    Ok(())
}

#[test]
fn ff2x_xanes_vicorr_convolution_matches_feff_conv_without_listed_paths() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.corrections.vicorr = 0.02 * FEFF_HARTREE_EV;
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), 1.0, 0)?;
    let corrected = Ff2xXanesCorrectedComponents {
        total: Array1::from_vec(vec![11.0, 12.0, 14.0]),
        atomic: Array1::from_vec(vec![9.0, 10.0, 13.0]),
        fine_structure: Array1::from_vec(vec![2.0, 2.0, 1.0]),
    };

    let actual = ff2x_xanes_apply_vicorr_convolution(&input, &xsect, 0, corrected.clone())?;

    let spectrum = (0..xsect.main_energy_count)
        .map(|row| Complex64::new(corrected.total[row], corrected.atomic[row]))
        .collect::<Vec<_>>();
    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("test xsect omega should be contiguous")?;
    let expected = lorentz_convolve(&omega[..xsect.main_energy_count], &spectrum, 0.02)?;
    for row in 0..xsect.main_energy_count {
        assert_close(actual.total[row], expected[row].re, 1.0e-12);
        assert_close(actual.atomic[row], expected[row].im, 1.0e-12);
        assert_close(
            actual.fine_structure[row],
            expected[row].re - expected[row].im,
            1.0e-12,
        );
    }

    let skipped = ff2x_xanes_apply_vicorr_convolution(&input, &xsect, 1, corrected.clone())?;
    assert_eq!(skipped, corrected);
    Ok(())
}

#[test]
fn ff2x_module_applies_xanes_vicorr_convolution_without_listed_paths() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    input.corrections.vicorr = 0.02 * FEFF_HARTREE_EV;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chi.dat").exists());
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    let xsect = xsect_dat_ff2x_handoff(&sample_xanes_xsect_dat(), input.corrections.s02, 0)?;
    let spectrum = [0.9_f64, 1.0, 1.1]
        .iter()
        .map(|&energy| {
            let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
            Complex64::new(12.0 * step, 10.0 * step)
        })
        .collect::<Vec<_>>();
    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("test xsect omega should be contiguous")?;
    let expected = lorentz_convolve(&omega[..xsect.main_energy_count], &spectrum, 0.02)?;
    for row in 0..xmu.point_count() {
        assert_close(xmu.mu[row], expected[row].re, 5.0e-5);
        assert_close(xmu.mu0[row], expected[row].im, 5.0e-5);
        assert_close(xmu.chi[row], expected[row].re - expected[row].im, 5.0e-5);
    }
    Ok(())
}

#[test]
fn ff2x_module_applies_xanes_gamma_channel_broadening_without_listed_paths() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.control.i_gamma_ch = 1;
    input.debye.tk = 0.0;
    let mut xsect_dat = sample_xanes_xsect_dat();
    xsect_dat.core_hole_width_ev = 2.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_xanes_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect_dat)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert!(
        xmu.header_lines
            .iter()
            .any(|line| line.contains("Energy zero shift") && line.contains("1.00000E0"))
    );
    let log = read_module_log_dat(temp.path().join("log6.dat"))?;
    assert!(
        log.lines
            .iter()
            .any(|line| line.contains("Energy zero shift") && line.contains("1.00000E0"))
    );
    let xsect = xsect_dat_ff2x_handoff(&xsect_dat, input.corrections.s02, 0)?;
    let spectrum = [0.9_f64, 1.0, 1.1]
        .iter()
        .map(|&energy| {
            let step = 0.5 + ((energy - 1.0) / 0.1).atan() / std::f64::consts::PI;
            Complex64::new(12.0 * step, 10.0 * step)
        })
        .collect::<Vec<_>>();
    let omega = xsect
        .omega_hartree
        .as_slice()
        .context("test xsect omega should be contiguous")?;
    let expected = lorentz_convolve(
        &omega[..xsect.main_energy_count],
        &spectrum,
        xsect.core_hole_width_hartree * 0.5,
    )?;
    for row in 0..xmu.point_count() {
        assert_close(xmu.mu[row], expected[row].re, 5.0e-5);
        assert_close(xmu.mu0[row], expected[row].im, 5.0e-5);
        assert_close(xmu.chi[row], expected[row].re - expected[row].im, 5.0e-5);
    }
    Ok(())
}

#[test]
fn ff2x_module_applies_xanes_gamma_channel_to_path_damping() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let mut xsect_dat = sample_xanes_xsect_dat();
    xsect_dat.core_hole_width_ev = 4.0;

    let baseline = tempfile::tempdir()?;
    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(
        baseline.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(baseline.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect_dat)?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    run_in_dir(baseline.path())?;
    let baseline_xmu = read_xmu_dat(baseline.path().join("xmu.dat"))?;

    let gamma = tempfile::tempdir()?;
    input.control.i_gamma_ch = 1;
    write_ff2x_input_data(gamma.path(), &input)?;
    write_feff_bin(
        gamma.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(gamma.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(gamma.path().join("xsect.dat"), &xsect_dat)?;
    write_fms_bin(gamma.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(gamma.path())?;

    assert_eq!(count, 2);
    let xmu = read_xmu_dat(gamma.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), baseline_xmu.point_count());
    assert!(
        xmu.chi
            .iter()
            .zip(baseline_xmu.chi.iter())
            .any(|(&actual, &baseline)| (actual - baseline).abs() > 1.0e-6),
        "iGammaCH path damping should change at least one XANES chi point"
    );
    Ok(())
}

#[test]
fn ff2x_module_treats_non_one_xanes_gamma_channel_like_zero() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let mut xsect_dat = sample_xanes_xsect_dat();
    xsect_dat.core_hole_width_ev = 4.0;

    let baseline = tempfile::tempdir()?;
    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(
        baseline.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(baseline.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect_dat)?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;
    run_in_dir(baseline.path())?;

    let non_one = tempfile::tempdir()?;
    input.control.i_gamma_ch = 2;
    write_ff2x_input_data(non_one.path(), &input)?;
    write_feff_bin(
        non_one.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(non_one.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(non_one.path().join("xsect.dat"), &xsect_dat)?;
    write_fms_bin(non_one.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(non_one.path())?;

    assert_eq!(count, 2);
    assert_eq!(
        read_xmu_dat(non_one.path().join("xmu.dat"))?,
        read_xmu_dat(baseline.path().join("xmu.dat"))?
    );
    Ok(())
}

#[test]
fn ff2x_module_accepts_xanes_ipr6_two_without_chip_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.control.ipr6 = 2;
    input.debye.tk = 0.0;
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(
        temp.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(temp.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xanes_xsect_dat())?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    assert!(!temp.path().join("chip0017.dat").exists());
    assert!(!temp.path().join("files.dat").exists());
    Ok(())
}

#[test]
fn ff2x_module_generates_xanes_ipr6_three_feff_path_outputs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 1;
    input.control.absolu = 1;
    input.control.ipr6 = 3;
    input.debye.tk = 0.0;
    let mut xsect = sample_xanes_xsect_dat();
    xsect.titles = sample_so2conv_header_titles();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(
        temp.path().join("feff.bin"),
        &sample_xanes_feff_bin_with_path(),
    )?;
    write_list_dat(temp.path().join("list.dat"), &sample_xanes_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_xanes_fms_bin(2.0))?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    assert!(!temp.path().join("chip0017.dat").exists());
    let files = std::fs::read_to_string(temp.path().join("files.dat"))?;
    assert!(files.contains("feff0017.dat"));

    let target = SfconvSo2convTarget {
        file_name: "feff0017.dat".to_string(),
        kind: SfconvSo2convTargetKind::FeffPath,
    };
    let text = std::fs::read_to_string(temp.path().join("feff0017.dat"))?;
    let data =
        sfconv_so2conv_target_data_from_text(temp.path().join("feff0017.dat"), &target, &text)?;
    let SfconvSo2convTargetData::FeffPath { header, data } = data else {
        unreachable!("target kind selects feff path data");
    };

    assert_eq!(header.material.core_hole_width_ev, 1.729);
    assert_eq!(data.point_count(), xsect.main_energy_count);
    assert_eq!(data.leg_count, 2);
    assert_close(data.degeneracy, 1.5, 1.0e-12);
    assert_close(data.effective_half_path_length_angstrom, 1.0, 5.0e-5);
    Ok(())
}

#[test]
fn ff2x_module_generates_fprime_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 4;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let xsect = sample_fprime_xsect_dat();
    let feff = sample_fprime_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    let text = std::fs::read_to_string(temp.path().join("xmu.dat"))?;
    assert!(text.contains("f'"));
    assert!(text.contains("E+"));
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), xsect.main_energy_count);
    assert_eq!(xmu.normalization, Some(1.0));
    for row in 0..xmu.point_count() {
        assert_close(xmu.wave_number[row], xmu.mu[row], 5.0e-7);
        assert_close(xmu.mu0[row], xmu.chi[row], 5.0e-7);
    }
    let handoff = xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &handoff, 0)
    );
    Ok(())
}

#[test]
fn ff2x_module_finalizes_fprime_configuration_average_from_chia_bin() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 4;
    input.control.absolu = 1;
    input.debye.tk = 0.0;
    let xsect = sample_fprime_xsect_dat();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 2;

    write_ff2x_input_data(temp.path(), &input)?;
    std::fs::write(
        temp.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(temp.path().join("feff.bin"), &sample_fprime_feff_bin_data())?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    write_chia_bin(
        temp.path().join("chia.bin"),
        &ChiaBinData {
            values: vec![Complex64::new(0.0, 2.0); xsect.energy_grid_ev.len()],
        },
    )?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(!temp.path().join("chia.bin").exists());
    assert!(!temp.path().join("chi.dat").exists());

    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), xsect.main_energy_count);
    for row in 0..xmu.point_count() {
        assert_close(xmu.mu0[row], 3.0 * xmu.chi[row], 5.0e-3);
    }
    Ok(())
}

#[test]
fn ff2x_module_generates_danes_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 3;
    input.control.absolu = 1;
    input.corrections.vicorr = 0.5;
    input.debye.tk = 0.0;
    let xsect = sample_danes_xsect_dat();
    let feff = sample_danes_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(temp.path().join("fms.bin"), &sample_danes_fms_bin())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 2);
    assert!(temp.path().join("xmu.dat").is_file());
    assert!(!temp.path().join("chi.dat").exists());
    let text = std::fs::read_to_string(temp.path().join("xmu.dat"))?;
    assert!(!text.contains("f'"));
    let xmu = read_xmu_dat(temp.path().join("xmu.dat"))?;
    assert_eq!(xmu.point_count(), xsect.main_energy_count);
    assert_eq!(xmu.normalization, Some(1.0));
    for row in 0..xmu.point_count() {
        assert!(xmu.wave_number[row].is_finite());
        assert!(xmu.mu[row].is_finite());
        assert!(xmu.mu0[row].is_finite());
        assert!(xmu.chi[row].is_finite());
        assert_close(xmu.chi[row], xmu.mu[row] - xmu.mu0[row], 2.0e-5);
    }
    assert!(xmu.chi.iter().any(|value| value.abs() > 1.0e-8));
    let handoff = xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv)?;
    assert_eq!(
        read_module_log_dat(temp.path().join("log6.dat"))?,
        generated_ff2x_generation_module_log(&input, &handoff, 0)
    );
    Ok(())
}

#[test]
fn ff2x_module_finalizes_danes_configuration_average_from_chia_bin() -> Result<()> {
    let baseline = tempfile::tempdir()?;
    let averaged = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ispec = 3;
    input.control.absolu = 1;
    input.corrections.vicorr = 0.5;
    input.debye.tk = 0.0;
    let xsect = sample_danes_xsect_dat();
    let feff = sample_danes_feff_bin_data();
    let mut global = sample_global_input(0, -1);
    global.cfaverage.nabs = 2;

    write_ff2x_input_data(baseline.path(), &input)?;
    write_feff_bin(baseline.path().join("feff.bin"), &feff)?;
    write_list_dat(baseline.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(baseline.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(baseline.path().join("fms.bin"), &sample_danes_fms_bin())?;

    write_ff2x_input_data(averaged.path(), &input)?;
    std::fs::write(
        averaged.path().join("global.inp"),
        global_input_string(&global)?,
    )?;
    write_feff_bin(averaged.path().join("feff.bin"), &feff)?;
    write_list_dat(averaged.path().join("list.dat"), &sample_empty_list_dat())?;
    write_xsect_dat(averaged.path().join("xsect.dat"), &xsect)?;
    write_fms_bin(averaged.path().join("fms.bin"), &sample_danes_fms_bin())?;
    write_chia_bin(
        averaged.path().join("chia.bin"),
        &ChiaBinData {
            values: vec![Complex64::new(0.0, 2.0); xsect.energy_grid_ev.len()],
        },
    )?;

    let baseline_count = run_in_dir(baseline.path())?;
    let averaged_count = run_in_dir(averaged.path())?;

    assert_eq!(baseline_count, averaged_count);
    assert_eq!(averaged_count, 2);
    assert!(!averaged.path().join("chia.bin").exists());

    let baseline_xmu = read_xmu_dat(baseline.path().join("xmu.dat"))?;
    let averaged_xmu = read_xmu_dat(averaged.path().join("xmu.dat"))?;
    assert_eq!(averaged_xmu.point_count(), xsect.main_energy_count);
    assert!(
        averaged_xmu
            .chi
            .iter()
            .zip(baseline_xmu.chi.iter())
            .any(|(&averaged, &baseline)| (averaged - baseline).abs() > 1.0e-4),
        "DANES configuration average should use the accumulated chia.bin trace"
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_mbconv_exafs_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.mbconv = 1;
    input.debye.tk = 0.0;
    let feff = sample_mbconv_feff_bin_data();
    let xsect = sample_mbconv_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let handoff = xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv)?;
    let prepared = ff2x_prepared_paths(&input, &feff, &sample_list_dat())?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    let path_sum = ff2x_sum_prepared_paths(&feff, &prepared, grid.interpolation_momentum.view())?;
    let output_energy = ff2x_output_energy_grid(feff.edge_energy, &handoff, &grid)?;
    let unconvolved_chi = Array1::from_iter(path_sum.total.iter().map(|value| value.im));
    let expected_chi = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
        energy: output_energy.photon_energy_hartree.view(),
        xmu: unconvolved_chi.view(),
        fermi_energy: output_energy.fermi_energy_hartree,
        amplitude_reduction: handoff.file_amplitude_reduction,
        relaxation_energy: handoff.relaxation_energy,
        plasmon_frequency: handoff.plasmon_frequency * 0.5,
    })?;

    assert_eq!(chi.point_count(), expected_chi.len());
    assert_close(chi.chi[0], expected_chi[0], 5.0e-7);
    assert_close(chi.chi[1], expected_chi[1], 5.0e-7);
    assert_close(chi.magnitude[1], path_sum.total[1].norm(), 5.0e-7);
    assert!(
        read_xmu_dat(temp.path().join("xmu.dat"))?
            .normalization
            .is_some()
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_two_chip_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 2;
    let feff = sample_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 4);
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    let chip = read_chi_dat(temp.path().join("chip0017.dat"))?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    assert_eq!(chip.point_count(), chi.point_count());
    assert!(chip.has_path_phase());
    assert_eq!(chip.header_lines[0], "# PATH  Rmax= 6.000");
    assert!(
        chip.header_lines
            .iter()
            .any(|line| line.contains("S02") && line.contains("Debye temp"))
    );
    assert!(
        chip.header_lines
            .iter()
            .any(|line| line.contains("Debye-waller factor"))
    );
    assert_eq!(
        chip.header_lines.iter().rev().nth(1).map(String::as_str),
        Some(" -----------------------------------------------------------------------")
    );
    let phase_minus_2kr = chip
        .phase_minus_2kr
        .as_ref()
        .context("chip output should include phase-2kr")?;
    assert_close(chip.wave_number[0], chi.wave_number[0], 1.0e-12);
    assert_close(
        phase_minus_2kr[0],
        chip.phase[0]
            - 2.0 * grid.interpolation_momentum[0] * feff.paths[0].effective_half_path_length_bohr,
        5.0e-6,
    );
    Ok(())
}

#[test]
fn ff2x_chip_header_includes_damping_cumulants_and_hartree_shift() -> Result<()> {
    let mut input = sample_ff2x_input(1);
    input.debye.alphat = 0.034;
    input.debye.thetae = 400.0;
    input.debye.sig2g = 0.00123;
    input.corrections.vrcorr = 0.02 * FEFF_HARTREE_EV;
    input.corrections.vicorr = 0.01 * FEFF_HARTREE_EV;
    let mut xsect = sample_xsect_handoff();
    xsect.amplitude_reduction = 0.85;
    let feff = sample_single_scattering_feff_bin_data();
    let list = sample_single_scattering_list_dat();
    let prepared = ff2x_prepared_paths(&input, &feff, &list)?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    let path_sum = ff2x_sum_prepared_paths(&feff, &prepared, grid.interpolation_momentum.view())?;

    let chip = ff2x_chip_dat_from_path_signal(
        &input,
        &xsect,
        &list,
        &feff,
        &grid,
        &prepared[0],
        &path_sum.paths[0],
    )?;

    assert_eq!(chip.header_lines[0], "# PATH  Rmax= 6.000");
    assert_eq!(
        chip.header_lines[1],
        " S02  0.850  Temp  190.00  Debye temp  315.00  Global sig2  0.00123"
    );
    assert!(
        chip.header_lines
            .iter()
            .any(|line| line.contains("1st and 3rd cumulants"))
    );
    let shift_line = chip
        .header_lines
        .iter()
        .find(|line| line.contains("Energy zero shift"))
        .context("chip header should include nonzero correction shifts")?;
    let shift_tokens = shift_line.split_whitespace().collect::<Vec<_>>();
    assert_close(shift_tokens[5].parse::<f64>()?, 0.02, 1.0e-12);
    assert_close(shift_tokens[6].parse::<f64>()?, 0.01, 1.0e-12);
    let dw_line = chip
        .header_lines
        .iter()
        .find(|line| line.contains("Debye-waller factor"))
        .context("chip header should include Debye-Waller factor")?;
    let dw_tokens = dw_line.split_whitespace().collect::<Vec<_>>();
    assert!(dw_tokens[2].parse::<f64>()? > 0.0);
    assert!(dw_tokens[3].parse::<f64>()? > 0.0);
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_three_feff_path_outputs_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 3;
    let feff = sample_feff_bin_data();
    let mut xsect = sample_xsect_dat();
    xsect.titles = sample_so2conv_header_titles();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(temp.path().join("chip0017.dat").exists());
    let files = std::fs::read_to_string(temp.path().join("files.dat"))?;
    assert!(files.contains("feff0017.dat"));
    assert!(files.contains("0.00000"));

    let target = SfconvSo2convTarget {
        file_name: "feff0017.dat".to_string(),
        kind: SfconvSo2convTargetKind::FeffPath,
    };
    let text = std::fs::read_to_string(temp.path().join("feff0017.dat"))?;
    let data =
        sfconv_so2conv_target_data_from_text(temp.path().join("feff0017.dat"), &target, &text)?;
    let SfconvSo2convTargetData::FeffPath { header, data } = data else {
        unreachable!("target kind selects feff path data");
    };

    assert_eq!(header.material.core_hole_width_ev, 1.729);
    assert_eq!(data.point_count(), xsect.main_energy_count);
    assert_eq!(data.leg_count, feff.paths[0].leg_count());
    assert_close(data.degeneracy, feff.paths[0].degeneracy, 1.0e-12);
    assert_close(
        data.effective_half_path_length_angstrom,
        feff.paths[0].effective_half_path_length_bohr * FEFF_BIN_BOHR,
        5.0e-5,
    );
    assert_close(
        data.wave_number_inverse_angstrom[0],
        feff.real_momentum[0] / FEFF_BIN_BOHR,
        5.0e-4,
    );
    assert_close(data.central_phase[0], 0.2, 5.0e-5);
    assert_close(data.effective_phase[0], -0.3, 5.0e-5);
    assert_close(
        data.mean_free_path_angstrom[0],
        FEFF_BIN_BOHR / feff.complex_momentum[0].im,
        5.0e-5,
    );
    assert_close(
        data.real_momentum_inverse_angstrom[0],
        feff.complex_momentum[0].re / FEFF_BIN_BOHR,
        5.0e-4,
    );
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_four_chi_ckp_columns_from_handoffs() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 4;
    let feff = sample_four_point_feff_bin_data();
    let xsect = sample_four_point_xsect_dat();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &xsect)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(temp.path().join("chip0017.dat").exists());
    assert!(temp.path().join("feff0017.dat").exists());
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    assert!(chi.has_complex_wave_number());
    let ckp_real = chi
        .ckp_real
        .as_ref()
        .context("ipr6=4 chi.dat should include real ckp")?;
    let ckp_imag = chi
        .ckp_imag
        .as_ref()
        .context("ipr6=4 chi.dat should include imaginary ckp")?;
    let grid = ff2x_momentum_grid(&input, &feff)?;
    let handoff = xsect_dat_ff2x_handoff(&xsect, input.corrections.s02, input.control.mbconv)?;
    let (expected_real, expected_imag) = ff2x_chi_ckp_columns(&feff, &handoff, &grid)?;
    assert_eq!(ckp_real.len(), chi.point_count());
    assert_eq!(ckp_imag.len(), chi.point_count());
    assert_close(ckp_real[0], expected_real[0], 5.0e-7);
    assert_close(ckp_imag[0], expected_imag[0], 5.0e-7);
    Ok(())
}

#[test]
fn ff2x_module_generates_ipr6_above_four_like_fortran_ff2chi() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let mut input = sample_ff2x_input(1);
    input.control.ipr6 = 5;
    let feff = sample_feff_bin_data();
    write_ff2x_input_data(temp.path(), &input)?;
    write_feff_bin(temp.path().join("feff.bin"), &feff)?;
    write_list_dat(temp.path().join("list.dat"), &sample_list_dat())?;
    write_xsect_dat(temp.path().join("xsect.dat"), &sample_xsect_dat())?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 6);
    assert!(temp.path().join("chip0017.dat").exists());
    assert!(temp.path().join("feff0017.dat").exists());
    let chi = read_chi_dat(temp.path().join("chi.dat"))?;
    assert!(!chi.has_complex_wave_number());
    let files = std::fs::read_to_string(temp.path().join("files.dat"))?;
    assert!(files.contains("feff0017.dat"));
    Ok(())
}

#[test]
fn ff2x_module_generates_uncached_exafs_outputs_matching_reference_when_present() -> Result<()> {
    let Some(reference_dir) = reference_ff2x_dir()? else {
        crate::require_fixture!(
            "FF2X generation reference test; generated EXAFS/Cu reference not found"
        );
    };
    let required = [
        "ff2x.inp",
        "feff.bin",
        "list.dat",
        "xsect.dat",
        "chi.dat",
        "xmu.dat",
    ];
    if !required
        .iter()
        .all(|name| reference_dir.join(name).is_file())
    {
        crate::require_fixture!(
            "FF2X generation reference test; generated EXAFS/Cu handoffs not found"
        );
    }
    let temp = tempfile::tempdir()?;
    for name in ["ff2x.inp", "feff.bin", "list.dat", "xsect.dat"] {
        std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
    }

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 3);
    assert_chi_dat_close(
        &read_chi_dat(temp.path().join("chi.dat"))?,
        &read_chi_dat(reference_dir.join("chi.dat"))?,
    );
    assert_xmu_dat_close(
        &read_xmu_dat(temp.path().join("xmu.dat"))?,
        &read_xmu_dat(reference_dir.join("xmu.dat"))?,
    );
    Ok(())
}

#[test]
fn ff2x_path_damping_matches_generated_feff_path_headers_when_present() -> Result<()> {
    let Some(reference_dir) = reference_ff2x_dir()? else {
        crate::require_fixture!(
            "FF2X damping reference test; generated EXAFS/Cu reference not found"
        );
    };
    let required = ["ff2x.inp", "feff.bin", "list.dat", "chi.dat"];
    if !required
        .iter()
        .all(|name| reference_dir.join(name).is_file())
    {
        crate::require_fixture!(
            "FF2X damping reference test; generated EXAFS/Cu handoffs not found"
        );
    }

    let input_path = reference_dir.join("ff2x.inp");
    let input = Ff2xInput::parse_str(&input_path, &std::fs::read_to_string(&input_path)?)?;
    let feff = read_feff_bin(reference_dir.join("feff.bin"))?;
    let list = read_list_dat(reference_dir.join("list.dat"))?;
    let damping = ff2x_path_damping(&input, &feff, &list)?;
    let headers = read_chi_path_headers(reference_dir.join("chi.dat"))?;

    assert_eq!(damping.len(), headers.len());
    for (actual, expected) in damping.iter().zip(headers) {
        assert_eq!(actual.path_index, expected.path_index);
        assert_close(actual.total_sigma2_angstrom2, expected.sigma2, 1.0e-5);
        assert_close(actual.criterion, expected.criterion, 5.0e-3);
        assert_close(actual.degeneracy, expected.degeneracy, 5.0e-3);
        assert_eq!(actual.leg_count, expected.leg_count);
        assert_close(
            actual.effective_half_path_length_angstrom,
            expected.effective_half_path_length_angstrom,
            5.0e-5,
        );
    }
    Ok(())
}

fn write_ff2x_input(work_dir: &Path, mchi: i32) -> Result<()> {
    write_ff2x_input_data(work_dir, &sample_ff2x_input(mchi))
}

fn write_ff2x_input_data(work_dir: &Path, input: &Ff2xInput) -> Result<()> {
    std::fs::write(work_dir.join("ff2x.inp"), ff2x_input_string(input)?)?;
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

fn write_global_input(work_dir: &Path, do_nrixs: i32, ldecmx: i32) -> Result<()> {
    std::fs::write(
        work_dir.join("global.inp"),
        global_input_string(&sample_global_input(do_nrixs, ldecmx))?,
    )?;
    Ok(())
}

fn write_eels_polarization_input(work_dir: &Path, min: i32, step: i32, max: i32) -> Result<()> {
    let input = EelsInput {
        calculate_elnes: true,
        calculation_mode: 1,
        control: EelsControl {
            average: 0,
            relativistic: 1,
            cross_terms: 1,
            input: 1,
            spectrum_column: 4,
        },
        polarization: EelsPolarization { min, step, max },
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

fn write_ff2x_dmdw_handoffs(work_dir: &Path, path: &FeffBinPath) -> Result<()> {
    std::fs::write(
        work_dir.join("dmdw.inp"),
        "   1\n   1\n   1    190.000\n   0\nfeff.dym\n   0\n",
    )?;
    std::fs::write(work_dir.join("feff.dym"), sample_ff2x_dmdw_dym(path)?)?;
    Ok(())
}

fn sample_ff2x_input(mchi: i32) -> Ff2xInput {
    Ff2xInput {
        control: Ff2xControl {
            mchi,
            ispec: 0,
            idwopt: 0,
            ipr6: 0,
            mbconv: 0,
            absolu: 0,
            i_gamma_ch: 0,
        },
        corrections: Ff2xCorrections {
            vrcorr: 0.0,
            vicorr: 0.0,
            s02: 1.0,
            critcw: 4.0,
        },
        debye: Ff2xDebye {
            tk: 190.0,
            thetad: 315.0,
            alphat: 0.0,
            thetae: 0.0,
            sig2g: 0.0,
            sig_gk: 0.0,
        },
        momentum_transfer: [0.0, 0.0, 0.0],
        decomposition_channels: -1,
        electronic_temperature: 0.0,
    }
}

fn sample_global_input(do_nrixs: i32, ldecmx: i32) -> GlobalInput {
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
            do_nrixs,
            ldecmx,
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

fn write_ff2x_spring_handoffs(work_dir: &Path) -> Result<()> {
    std::fs::write(
        work_dir.join("spring.inp"),
        "VDOS 0.03 0.5 1.0 2.5\nSTRETCHES\n0 1 27.9 2.0\n1 2 12.0 2.0\nEND\n",
    )?;
    std::fs::write(
        work_dir.join("geom.dat"),
        geom_dat_string(&sample_ff2x_spring_geom())?,
    )?;
    Ok(())
}

fn sample_ff2x_spring_geom() -> GeomDat {
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

fn sample_ff2x_spring_feff_bin_data() -> FeffBinData {
    let mut feff = sample_feff_bin_data();
    feff.potentials = vec![
        FeffBinPotential {
            label: "Cu".to_string(),
            atomic_number: 29,
        },
        FeffBinPotential {
            label: "Zn".to_string(),
            atomic_number: 30,
        },
        FeffBinPotential {
            label: "Ga".to_string(),
            atomic_number: 31,
        },
    ];
    feff.paths = vec![FeffBinPath {
        index: 17,
        degeneracy: 1.0,
        effective_half_path_length_bohr: 3.8 / FEFF_BIN_BOHR,
        criterion: 12.5,
        potential_indices: Array1::from_vec(vec![2, 0]),
        positions: Array2::from_shape_fn((2, 3), |(leg, axis)| match (leg, axis) {
            (0, 0) => 3.8 / FEFF_BIN_BOHR,
            (0, 1..=2) => 0.0,
            (1, 0..=2) => 0.0,
            _ => 0.0,
        }),
        beta: Array1::from_vec(vec![0.1, 0.2]),
        eta: Array1::from_vec(vec![0.4, 0.5]),
        leg_distances: Array1::from_vec(vec![1.9 / FEFF_BIN_BOHR, 1.9 / FEFF_BIN_BOHR]),
        amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
        phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
    }];
    feff
}

fn sample_ff2x_spring_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.0,
            amplitude_ratio: 12.5,
            degeneracy: 1.0,
            leg_count: 2,
            effective_half_path_length_angstrom: 3.8,
        }],
    }
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

fn sample_feffl_bin_data(
    path_count: usize,
    energy_count: usize,
    max_decomposition_channel: usize,
) -> FefflBinData {
    let channel_count = max_decomposition_channel + 1;
    FefflBinData {
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        amplitudes: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| sample_feffl_amplitude(path, lg2, lg1, energy),
        ),
        phases: Array4::from_shape_fn(
            (path_count, channel_count, channel_count, energy_count),
            |(path, lg2, lg1, energy)| sample_feffl_phase(path, lg2, lg1, energy),
        ),
    }
}

fn sample_feffl_amplitude(path: usize, lg2: usize, lg1: usize, energy: usize) -> f64 {
    1.0 + path as f64 * 10.0 + lg2 as f64 * 2.0 + lg1 as f64 * 0.5 + energy as f64 * 0.25
}

fn sample_feffl_phase(path: usize, lg2: usize, lg1: usize, energy: usize) -> f64 {
    -0.2 + path as f64 * 0.1 + lg2 as f64 * 0.05 - lg1 as f64 * 0.02 - energy as f64 * 0.03
}

fn sample_fmsl_bin_data(energy_count: usize, max_decomposition_channel: usize) -> FmslBinData {
    let channel_count = max_decomposition_channel + 1;
    FmslBinData {
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        max_decomposition_channel,
        traces: Array3::from_shape_fn(
            (energy_count, channel_count, channel_count),
            |(energy, lg2, lg1)| {
                Complex64::new(
                    energy as f64 + lg2 as f64 * 0.1 + lg1 as f64 * 0.01,
                    -(energy as f64) - lg2 as f64 * 0.2 - lg1 as f64 * 0.02,
                )
            },
        ),
    }
}

fn sample_xsecl_bin_data(energy_count: usize) -> XseclBinData {
    XseclBinData {
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
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
        atom_cross_sections: Array2::from_shape_fn((energy_count, 2), |(energy, channel)| {
            let scale = (energy + 1) as f64 * (channel + 1) as f64;
            Complex64::new(-0.01 * scale, 0.2 * scale)
        }),
        raw_atom_cross_section_pad: None,
    }
}

fn sample_ff2x_dmdw_dym(path: &FeffBinPath) -> Result<String> {
    let leg_count = path.leg_count();
    if leg_count == 0 {
        return Err(anyhow::anyhow!("test path requires at least one leg"));
    }
    let (position_rows, position_columns) = path.positions.dim();
    if position_rows != leg_count || position_columns != 3 {
        return Err(anyhow::anyhow!(
            "test path positions shape ({position_rows}, {position_columns}) does not match leg count {leg_count}"
        ));
    }

    let mut positions = Vec::with_capacity(leg_count);
    positions.push([
        path.positions[(leg_count - 1, 0)],
        path.positions[(leg_count - 1, 1)],
        path.positions[(leg_count - 1, 2)],
    ]);
    for leg in 0..leg_count.saturating_sub(1) {
        positions.push([
            path.positions[(leg, 0)],
            path.positions[(leg, 1)],
            path.positions[(leg, 2)],
        ]);
    }

    let mut out = String::new();
    out.push_str("    1\n");
    out.push_str(&format!("{:5}\n", positions.len()));
    for _ in &positions {
        out.push_str("   29\n");
    }
    for _ in &positions {
        out.push_str("   63.546000\n");
    }
    for position in &positions {
        out.push_str(&format!(
            "{:14.8}{:14.8}{:14.8}\n",
            position[0], position[1], position[2]
        ));
    }
    for first in 0..positions.len() {
        for second in 0..positions.len() {
            let diagonal = if first == second { 4.0 } else { 0.0 };
            out.push_str(&format!("{:5}{:5}\n", first + 1, second + 1));
            for row in 0..3 {
                out.push_str(&format!(
                    " {:13.6E} {:13.6E} {:13.6E}\n",
                    if row == 0 { diagonal } else { 0.0 },
                    if row == 1 { diagonal } else { 0.0 },
                    if row == 2 { diagonal } else { 0.0 }
                ));
            }
        }
    }
    Ok(out)
}

fn sample_four_point_feff_bin_data() -> FeffBinData {
    let mut feff = sample_feff_bin_data();
    feff.central_phase_shift = Array1::from_vec(vec![
        Complex64::new(0.1, -0.01),
        Complex64::new(0.2, -0.02),
        Complex64::new(0.3, -0.03),
        Complex64::new(0.4, -0.04),
    ]);
    feff.complex_momentum = Array1::from_vec(vec![
        Complex64::new(1.0, 0.1),
        Complex64::new(1.1, 0.2),
        Complex64::new(1.2, 0.3),
        Complex64::new(1.3, 0.4),
    ]);
    feff.real_momentum = Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]);
    feff.paths[0].amplitude = Array1::from_vec(vec![2.0, 2.1, 2.2, 2.3]);
    feff.paths[0].phase = Array1::from_vec(vec![-0.1, -0.2, -0.3, -0.4]);
    feff
}

fn sample_mbconv_feff_bin_data() -> FeffBinData {
    let mut feff = sample_four_point_feff_bin_data();
    feff.edge_energy = 0.0;
    feff.central_phase_shift = Array1::from_vec(vec![
        Complex64::new(0.1, -0.001),
        Complex64::new(0.2, -0.002),
        Complex64::new(0.3, -0.003),
        Complex64::new(0.4, -0.004),
    ]);
    feff.complex_momentum = Array1::from_vec(vec![
        Complex64::new(0.0, 0.01),
        Complex64::new(0.1, 0.01),
        Complex64::new(0.2, 0.01),
        Complex64::new(0.3, 0.01),
    ]);
    feff.real_momentum = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
    feff.paths[0].amplitude = Array1::from_vec(vec![1.0, 1.1, 1.2, 1.3]);
    feff.paths[0].phase = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    feff
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

fn sample_empty_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: Vec::new(),
    }
}

fn sample_xanes_feff_bin_data() -> FeffBinData {
    let momenta = [0.0, 0.1, 0.2].map(wave_number_from_hartree);
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: 0.1,
        edge_energy: 0.9,
        potentials: vec![FeffBinPotential {
            label: "Cu".to_string(),
            atomic_number: 29,
        }],
        central_phase_shift: Array1::from_vec(vec![
            Complex64::new(0.1, -0.01),
            Complex64::new(0.2, -0.02),
            Complex64::new(0.3, -0.03),
        ]),
        complex_momentum: Array1::from_vec(
            momenta
                .iter()
                .map(|&momentum| Complex64::new(momentum, 0.01))
                .collect(),
        ),
        real_momentum: Array1::from_vec(momenta.to_vec()),
        paths: Vec::new(),
        raw_text: None,
    }
}

fn sample_xanes_feff_bin_with_path() -> FeffBinData {
    let mut feff = sample_xanes_feff_bin_data();
    feff.paths = vec![FeffBinPath {
        index: 17,
        degeneracy: 1.5,
        effective_half_path_length_bohr: 1.0 / FEFF_BIN_BOHR,
        criterion: 20.0,
        potential_indices: Array1::from_vec(vec![0, 0]),
        positions: Array2::from_shape_fn((2, 3), |(leg, axis)| match (leg, axis) {
            (0, 0..=2) => 0.0,
            (1, 0) => 1.0 / FEFF_BIN_BOHR,
            (1, 1..=2) => 0.0,
            _ => 0.0,
        }),
        beta: Array1::from_vec(vec![0.0, 0.0]),
        eta: Array1::from_vec(vec![0.0, 0.0]),
        leg_distances: Array1::from_vec(vec![1.0 / FEFF_BIN_BOHR, 1.0 / FEFF_BIN_BOHR]),
        amplitude: Array1::from_vec(vec![0.8, 0.9, 1.0]),
        phase: Array1::from_vec(vec![0.4, 0.5, 0.6]),
    }];
    feff
}

fn sample_xanes_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.0,
            amplitude_ratio: 20.0,
            degeneracy: 1.5,
            leg_count: 2,
            effective_half_path_length_angstrom: 1.0,
        }],
    }
}

fn sample_single_scattering_feff_bin_data() -> FeffBinData {
    let mut feff = sample_feff_bin_data();
    feff.paths = vec![FeffBinPath {
        index: 17,
        degeneracy: 12.0,
        effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
        criterion: 100.0,
        potential_indices: Array1::from_vec(vec![1, 0]),
        positions: Array2::from_shape_fn((2, 3), |(leg, axis)| match (leg, axis) {
            (0, 0) => 2.5 / FEFF_BIN_BOHR,
            (0, 1..=2) => 0.0,
            (1, 0..=2) => 0.0,
            _ => 0.0,
        }),
        beta: Array1::from_vec(vec![0.1, 0.2]),
        eta: Array1::from_vec(vec![0.3, 0.4]),
        leg_distances: Array1::from_vec(vec![2.5 / FEFF_BIN_BOHR, 2.5 / FEFF_BIN_BOHR]),
        amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
        phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
    }];
    feff
}

fn sample_single_scattering_list_dat() -> ListDatData {
    ListDatData {
        titles: vec!["PATH  Rmax= 6.000".to_string()],
        entries: vec![ListDatEntry {
            path_index: 17,
            sigma2: 0.001,
            amplitude_ratio: 100.0,
            degeneracy: 12.0,
            leg_count: 2,
            effective_half_path_length_angstrom: 2.5,
        }],
    }
}

fn sample_xmu_dat() -> XmuDatData {
    XmuDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0"
                .to_string(),
            "# xsedge+ 50, used to normalize mu           1.234500E+00".to_string(),
        ],
        normalization: Some(1.2345),
        photon_energy_ev: Array1::from_vec(vec![8979.0, 8980.0, 8981.0]),
        relative_energy_ev: Array1::from_vec(vec![0.0, 1.0, 2.0]),
        wave_number: Array1::from_vec(vec![0.0, 0.512, 0.724]),
        mu: Array1::from_vec(vec![1.0, 1.1, 1.2]),
        mu0: Array1::from_vec(vec![0.9, 0.95, 1.0]),
        chi: Array1::from_vec(vec![0.1, 0.15, 0.2]),
    }
}

fn sample_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.9,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.95,
            chemical_potential: 0.15,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 3,
        fermi_index: 2,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.95 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.15 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.02 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.05 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.00),
        ]),
        normalized_background: Array1::from_vec(vec![1.0, 1.1, 1.2, 1.1, 1.1, 1.1]),
        cross_section: Array1::from_vec(vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
        ]),
    }
}

fn sample_fprime_xsect_dat() -> XsectDatData {
    let energy_grid_hartree = [
        Complex64::new(-0.08, 0.06),
        Complex64::new(0.02, 0.06),
        Complex64::new(0.15, 0.06),
        Complex64::new(0.31, 0.06),
        Complex64::new(0.38, 0.0),
        Complex64::new(0.55, 0.0),
        Complex64::new(0.80, 0.0),
        Complex64::new(1.10, 0.0),
    ];
    XsectDatData {
        titles: vec!["Cu FPRIME test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.0,
            chemical_potential: 0.0,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 4,
        fermi_index: 1,
        energy_grid_ev: Array1::from_iter(
            energy_grid_hartree
                .into_iter()
                .map(|energy| energy * FEFF_HARTREE_EV),
        ),
        normalized_background: Array1::from_iter((1..=8).map(|index| {
            let index = index as f64;
            0.7 + 0.05 * index + 0.001 * index * index
        })),
        cross_section: Array1::from_iter((1..=8).map(|index| {
            let index = index as f64;
            Complex64::new(
                0.4 + 0.03 * index + 0.002 * index * index,
                -0.08 + 0.015 * index,
            )
        })),
    }
}

fn sample_fprime_feff_bin_data() -> FeffBinData {
    let xsect = sample_fprime_xsect_dat();
    let momenta = xsect
        .energy_grid_ev
        .iter()
        .map(|energy| {
            wave_number_from_hartree(energy.re / FEFF_HARTREE_EV - xsect.scalars.edge_energy)
        })
        .collect::<Vec<_>>();
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: 0.38,
        edge_energy: 0.38,
        potentials: vec![FeffBinPotential {
            label: "Cu".to_string(),
            atomic_number: 29,
        }],
        central_phase_shift: Array1::from_elem(momenta.len(), Complex64::new(0.0, 0.0)),
        complex_momentum: Array1::from_iter(
            momenta
                .iter()
                .map(|&momentum| Complex64::new(momentum, 0.01)),
        ),
        real_momentum: Array1::from_vec(momenta),
        paths: Vec::new(),
        raw_text: None,
    }
}

fn sample_danes_xsect_dat() -> XsectDatData {
    let energy_grid_hartree = [
        Complex64::new(0.12, 0.07),
        Complex64::new(0.24, 0.07),
        Complex64::new(0.34, 0.07),
        Complex64::new(0.42, 0.07),
        Complex64::new(0.58, 0.07),
        Complex64::new(0.42, 0.035),
        Complex64::new(0.42, 0.070),
        Complex64::new(0.42, 0.120),
        Complex64::new(0.42, 0.200),
        Complex64::new(0.66, 1.0e-8),
        Complex64::new(0.95, 1.0e-8),
        Complex64::new(1.35, 1.0e-8),
        Complex64::new(2.10, 1.0e-8),
    ];
    XsectDatData {
        titles: vec!["Cu DANES test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.0,
            chemical_potential: 0.0,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 5,
        fermi_index: 3,
        energy_grid_ev: Array1::from_iter(
            energy_grid_hartree
                .into_iter()
                .map(|energy| energy * FEFF_HARTREE_EV),
        ),
        normalized_background: Array1::from_iter((1..=13).map(|index| {
            let index = index as f64;
            0.9 + 0.04 * index + 0.002 * index * index
        })),
        cross_section: Array1::from_iter((1..=13).map(|index| {
            let index = index as f64;
            Complex64::new(0.3 + 0.02 * index, -0.04 + 0.01 * index)
        })),
    }
}

fn sample_danes_feff_bin_data() -> FeffBinData {
    let xsect = sample_danes_xsect_dat();
    let momenta = xsect
        .energy_grid_ev
        .iter()
        .map(|energy| {
            wave_number_from_hartree(energy.re / FEFF_HARTREE_EV - xsect.scalars.edge_energy)
        })
        .collect::<Vec<_>>();
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: 0.42,
        edge_energy: 0.42,
        potentials: vec![FeffBinPotential {
            label: "Cu".to_string(),
            atomic_number: 29,
        }],
        central_phase_shift: Array1::from_elem(momenta.len(), Complex64::new(0.0, 0.0)),
        complex_momentum: Array1::from_iter(
            momenta
                .iter()
                .map(|&momentum| Complex64::new(momentum, 0.01)),
        ),
        real_momentum: Array1::from_vec(momenta),
        paths: Vec::new(),
        raw_text: None,
    }
}

fn sample_danes_fms_bin() -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 13,
        main_energy_count: 5,
        auxiliary_energy_count: 4,
        highest_potential_index: 1,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(1),
        spectra: Array2::from_shape_fn((1, 13), |(_, row)| {
            let row = row as f64 + 1.0;
            Complex64::new(0.005 * row, 0.01 + 0.001 * row)
        }),
    }
}

fn sample_xanes_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu XANES test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.9,
            chemical_potential: 0.1,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 3,
        fermi_index: 2,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.9 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.1 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.02 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.05 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.0),
        ]),
        normalized_background: Array1::from_vec(vec![1.0; 6]),
        cross_section: Array1::from_vec(vec![Complex64::new(0.0, 10.0); 6]),
    }
}

fn sample_xanes_thermal_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu thermal XANES test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 1.0,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.9,
            chemical_potential: 0.1,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 3,
        fermi_index: 2,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.9 * FEFF_HARTREE_EV, 0.2 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.2 * FEFF_HARTREE_EV),
            Complex64::new(1.1 * FEFF_HARTREE_EV, 0.2 * FEFF_HARTREE_EV),
            Complex64::new(0.9 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.1 * FEFF_HARTREE_EV, 0.1 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.01 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.02 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.03 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.04 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.05 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.06 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.07 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.08 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.09 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.0 * FEFF_HARTREE_EV, 0.0),
        ]),
        normalized_background: Array1::from_vec(vec![1.0; 17]),
        cross_section: Array1::from_vec(vec![Complex64::new(0.0, 10.0); 17]),
    }
}

fn sample_xanes_mbconv_xsect_dat() -> XsectDatData {
    let mut xsect = sample_xanes_xsect_dat();
    xsect.scalars.amplitude_reduction = 0.7;
    xsect.scalars.relaxation_energy = 0.09;
    xsect.scalars.plasmon_frequency = 0.10;
    xsect.normalized_background = Array1::from_vec(vec![1.0, 1.2, 1.6, 1.2, 1.2, 1.2]);
    xsect
}

fn sample_xanes_fms_bin(fine_structure: f64) -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 6,
        main_energy_count: 3,
        auxiliary_energy_count: 3,
        highest_potential_index: 1,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(1),
        spectra: Array2::from_shape_fn((1, 6), |(_, _)| Complex64::new(0.0, fine_structure)),
    }
}

fn sample_xanes_thermal_fms_bin(fine_structure: f64) -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 17,
        main_energy_count: 3,
        auxiliary_energy_count: 14,
        highest_potential_index: 1,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(1),
        spectra: Array2::from_shape_fn((1, 17), |(_, _)| Complex64::new(0.0, fine_structure)),
    }
}

fn sample_xanes_fms_bin_for_polarization_offsets(spectrum_count: usize) -> FmsBinData {
    FmsBinData {
        cluster_radius_angstrom: 5.5,
        energy_count: 6,
        main_energy_count: 3,
        auxiliary_energy_count: 3,
        highest_potential_index: 1,
        pad_width: FMS_BIN_DEFAULT_PAD_WIDTH,
        declared_spectrum_count: Some(spectrum_count),
        spectra: Array2::from_shape_fn((spectrum_count, 6), |(spectrum, _)| {
            Complex64::new(0.0, spectrum as f64 + 1.0)
        }),
    }
}

fn sample_four_point_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.9,
            relaxation_energy: 0.0,
            plasmon_frequency: 0.0,
            edge_energy: 0.95,
            chemical_potential: 0.15,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 4,
        fermi_index: 2,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.95 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.15 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.25 * FEFF_HARTREE_EV, 0.10 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.02 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.05 * FEFF_HARTREE_EV),
            Complex64::new(1.05 * FEFF_HARTREE_EV, 0.00),
        ]),
        normalized_background: Array1::from_vec(vec![1.0, 1.1, 1.2, 1.3, 1.1, 1.1, 1.1]),
        cross_section: Array1::from_vec(vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
        ]),
    }
}

fn sample_mbconv_xsect_dat() -> XsectDatData {
    XsectDatData {
        titles: vec!["Cu mbconv test".to_string()],
        scalars: XsectDatScalars {
            amplitude_reduction: 0.72,
            relaxation_energy: 0.08,
            plasmon_frequency: 0.10,
            edge_energy: 0.0,
            chemical_potential: 0.0,
        },
        core_hole_width_ev: 1.0,
        main_energy_count: 4,
        fermi_index: 1,
        energy_grid_ev: Array1::from_vec(vec![
            Complex64::new(0.00 * FEFF_HARTREE_EV, 0.03 * FEFF_HARTREE_EV),
            Complex64::new(0.05 * FEFF_HARTREE_EV, 0.03 * FEFF_HARTREE_EV),
            Complex64::new(0.10 * FEFF_HARTREE_EV, 0.03 * FEFF_HARTREE_EV),
            Complex64::new(0.20 * FEFF_HARTREE_EV, 0.03 * FEFF_HARTREE_EV),
            Complex64::new(0.00 * FEFF_HARTREE_EV, 0.01 * FEFF_HARTREE_EV),
            Complex64::new(0.00 * FEFF_HARTREE_EV, 0.02 * FEFF_HARTREE_EV),
            Complex64::new(0.00, 0.00),
        ]),
        normalized_background: Array1::from_vec(vec![1.0, 1.1, 1.3, 1.6, 1.0, 1.0, 1.0]),
        cross_section: Array1::from_vec(vec![
            Complex64::new(0.0, 10.0),
            Complex64::new(0.0, 11.0),
            Complex64::new(0.0, 13.0),
            Complex64::new(0.0, 16.0),
            Complex64::new(0.0, 10.0),
            Complex64::new(0.0, 10.0),
            Complex64::new(0.0, 10.0),
        ]),
    }
}

fn sample_so2conv_header_titles() -> Vec<String> {
    vec![
        "Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000".to_string(),
    ]
}

fn sample_xsect_handoff() -> XsectFf2xHandoff {
    XsectFf2xHandoff {
        titles: vec!["Cu test".to_string()],
        title_count: 1,
        amplitude_reduction: 1.0,
        file_amplitude_reduction: 1.0,
        relaxation_energy: 0.0,
        plasmon_frequency: 0.0,
        edge_energy_hartree: 1.0,
        chemical_potential_hartree: 0.2,
        core_hole_width_hartree: 0.01,
        main_energy_count: 2,
        fermi_index_1based: 1,
        fermi_index: 0,
        cross_section_count: 2,
        energy_grid_hartree: Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.1, 0.0),
        ]),
        omega_hartree: Array1::from_vec(vec![0.2, 0.3]),
        wave_number: Array1::from_vec(vec![0.0, 0.1]),
        normalized_background: Array1::from_vec(vec![1.0, 1.1]),
        cross_section: Array1::from_vec(vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, 1.1)]),
    }
}

fn sample_constant_xscorr_handoff() -> XsectFf2xHandoff {
    XsectFf2xHandoff {
        titles: vec!["Cu test".to_string()],
        title_count: 1,
        amplitude_reduction: 1.0,
        file_amplitude_reduction: 1.0,
        relaxation_energy: 0.0,
        plasmon_frequency: 0.0,
        edge_energy_hartree: 0.9,
        chemical_potential_hartree: 0.1,
        core_hole_width_hartree: 0.1,
        main_energy_count: 3,
        fermi_index_1based: 2,
        fermi_index: 1,
        cross_section_count: 6,
        energy_grid_hartree: Array1::from_vec(vec![
            Complex64::new(0.9, 0.1),
            Complex64::new(1.0, 0.1),
            Complex64::new(1.1, 0.1),
            Complex64::new(1.0, 0.02),
            Complex64::new(1.0, 0.05),
            Complex64::new(1.0, 0.0),
        ]),
        omega_hartree: Array1::from_vec(vec![0.1, 0.2, 0.3, 0.2, 0.2, 0.2]),
        wave_number: Array1::from_vec(vec![0.0, 0.447_213_595_5, 0.632_455_532, 0.0, 0.0, 0.0]),
        normalized_background: Array1::from_vec(vec![1.0; 6]),
        cross_section: Array1::from_vec(vec![Complex64::new(0.0, 10.0); 6]),
    }
}

fn sample_constant_thermal_xscorr_handoff() -> XsectFf2xHandoff {
    let energy_grid_hartree = Array1::from_vec(vec![
        Complex64::new(0.9, 0.2),
        Complex64::new(1.0, 0.2),
        Complex64::new(1.1, 0.2),
        Complex64::new(0.9, 0.1),
        Complex64::new(1.0, 0.1),
        Complex64::new(1.1, 0.1),
        Complex64::new(1.0, 0.01),
        Complex64::new(1.0, 0.02),
        Complex64::new(1.0, 0.03),
        Complex64::new(1.0, 0.04),
        Complex64::new(1.0, 0.05),
        Complex64::new(1.0, 0.06),
        Complex64::new(1.0, 0.07),
        Complex64::new(1.0, 0.08),
        Complex64::new(1.0, 0.09),
        Complex64::new(1.0, 0.10),
        Complex64::new(1.0, 0.0),
    ]);
    let energy_count = energy_grid_hartree.len();
    XsectFf2xHandoff {
        titles: vec!["Cu thermal test".to_string()],
        title_count: 1,
        amplitude_reduction: 1.0,
        file_amplitude_reduction: 1.0,
        relaxation_energy: 0.0,
        plasmon_frequency: 0.0,
        edge_energy_hartree: 0.9,
        chemical_potential_hartree: 0.1,
        core_hole_width_hartree: 0.1,
        main_energy_count: 3,
        fermi_index_1based: 2,
        fermi_index: 1,
        cross_section_count: energy_count,
        omega_hartree: Array1::from_vec(vec![0.0; energy_count]),
        wave_number: Array1::from_vec(vec![0.0; energy_count]),
        normalized_background: Array1::from_vec(vec![1.0; energy_count]),
        cross_section: Array1::from_vec(vec![Complex64::new(0.0, 10.0); energy_count]),
        energy_grid_hartree,
    }
}

fn sample_xmu_row_xsect_handoff() -> XsectFf2xHandoff {
    XsectFf2xHandoff {
        titles: vec!["Cu test".to_string()],
        title_count: 1,
        amplitude_reduction: 1.0,
        file_amplitude_reduction: 1.0,
        relaxation_energy: 0.0,
        plasmon_frequency: 0.0,
        edge_energy_hartree: 1.0,
        chemical_potential_hartree: 0.0,
        core_hole_width_hartree: 0.01,
        main_energy_count: 3,
        fermi_index_1based: 1,
        fermi_index: 0,
        cross_section_count: 3,
        energy_grid_hartree: Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]),
        omega_hartree: Array1::from_vec(vec![0.0, 1.0, 2.0]),
        wave_number: Array1::from_vec(vec![0.0, 1.0, 2.0]),
        normalized_background: Array1::from_vec(vec![2.0, 4.0, 6.0]),
        cross_section: Array1::from_vec(vec![
            Complex64::new(0.0, 10.0),
            Complex64::new(0.0, 20.0),
            Complex64::new(0.0, 30.0),
        ]),
    }
}

fn sample_chi_dat() -> ChiDatData {
    ChiDatData {
        header_lines: vec![
            "# # Cu                                                           FEFF 10.0"
                .to_string(),
            "#       k          chi          mag           phase @#".to_string(),
        ],
        wave_number: Array1::from_vec(vec![0.0, 0.05, 0.1]),
        chi: Array1::from_vec(vec![-0.115_938_3, -0.119_413_8, -0.122_912_6]),
        magnitude: Array1::from_vec(vec![0.270_227_8, 0.272_670_8, 0.275_083_6]),
        phase: Array1::from_vec(vec![-2.698_164, -2.688_285, -2.678_386]),
        phase_minus_2kr: None,
        ckp_real: None,
        ckp_imag: None,
    }
}

fn sample_danes_dat() -> DanesDatData {
    DanesDatData {
        header_lines: vec!["# E  matsub. sommerf. anomal. tale, total, differ.".to_string()],
        energy_ev: Array1::from_vec(vec![-18.690, -17.122, -15.703]),
        matsubara: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        sommerfeld: Array1::from_vec(vec![0.0, 0.0, 0.0]),
        anomalous: Array1::from_vec(vec![10.097, 10.603, 11.159]),
        tail: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
        total: Array1::from_vec(vec![4.6396, 4.9442, 5.2935]),
        difference: Array1::from_vec(vec![-5.4576, -5.6591, -5.8651]),
    }
}

fn sample_xscorr_complex_table() -> XscorrComplexTable {
    XscorrComplexTable {
        energy_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_020_637_731_56, 0.000_120_322_770_8),
            Complex64::new(-0.000_021_177_763_91, 0.000_123_685_052_9),
        ]),
    }
}

fn sample_xscorr_curve_dat() -> XscorrCurveDatData {
    XscorrCurveDatData {
        energy: Array1::from_vec(vec![
            Complex64::new(-0.138_801_301_5, 0.000_183_746_545),
            Complex64::new(-0.138_801_301_5, 0.000_367_493_09),
        ]),
        values: Array1::from_vec(vec![
            Complex64::new(-0.000_028_662, 0.000_237_48),
            Complex64::new(-0.000_028_683, 0.000_237_44),
        ]),
    }
}

fn sample_xscorr_raw_dat() -> XscorrRawDatData {
    XscorrRawDatData {
        temperature_hartree: 0.0,
        electronic_temperature_ev: 0.0,
        loss_ev: 0.864_59,
        fermi_energy_ev: -3.776_977_18,
        pole_count: 0,
        omega_hartree: Array1::from_vec(vec![-0.138_801_301_5, -0.137_401_158_7]),
        cchi: Array1::from_vec(vec![
            Complex64::new(-0.000_016_299_5, 0.000_115_24),
            Complex64::new(-0.000_016_898_337_65, 0.000_118_558_222_9),
        ]),
        one_minus_fermi: Array1::from_vec(vec![0.5, 0.514_017_875_2]),
        xmu0: Array1::from_vec(vec![
            Complex64::new(-0.000_032_599, 0.000_230_48),
            Complex64::new(-0.000_032_875, 0.000_230_65),
        ]),
    }
}

fn sample_cum_dat() -> CumDatData {
    CumDatData {
        einstein_temperature: 400.0,
        thermal_expansion: 0.034,
        entries: vec![CumDatEntry {
            path_index: 17,
            first_cumulant_angstrom: 0.00013,
            sigma2_angstrom2: 0.00610,
            third_cumulant_angstrom3: 0.0000009,
        }],
    }
}

fn sample_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating EXAFS ...".to_string(),
            "Done with module: EXAFS spectra.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(), "\n".to_string()],
    }
}

fn reference_ff2x_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu");
    let required = ["ff2x.inp", "xmu.dat", "chi.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn reference_nrixs_gecl4_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/NRIXS/GeCl_4");
    let required = ["fms.bin", "fmsl.bin", "gtrl.dat", "xmul.dat", "xsect.dat"];
    Ok(required
        .iter()
        .all(|name| path.join(name).is_file())
        .then_some(path))
}

fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read_module_log_dat(path)?))
    } else {
        Ok(None)
    }
}

fn optional_read<T>(
    path: impl AsRef<Path>,
    read: impl FnOnce(&Path) -> std::result::Result<T, IoError>,
) -> Result<Option<T>> {
    let path = path.as_ref();
    if path.is_file() {
        Ok(Some(read(path)?))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Ff2xPathHeader {
    path_index: usize,
    sigma2: f64,
    criterion: f64,
    degeneracy: f64,
    leg_count: usize,
    effective_half_path_length_angstrom: f64,
}

fn read_chi_path_headers(path: impl AsRef<Path>) -> Result<Vec<Ff2xPathHeader>> {
    let text = std::fs::read_to_string(path)?;
    let mut headers = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start_matches('#').trim();
        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 6 {
            continue;
        }
        let Ok(path_index) = tokens[0].parse::<usize>() else {
            continue;
        };
        let Ok(sigma2) = tokens[1].parse::<f64>() else {
            continue;
        };
        let Ok(criterion) = tokens[2].parse::<f64>() else {
            continue;
        };
        let Ok(degeneracy) = tokens[3].parse::<f64>() else {
            continue;
        };
        let Ok(leg_count) = tokens[4].parse::<usize>() else {
            continue;
        };
        let Ok(effective_half_path_length_angstrom) = tokens[5].parse::<f64>() else {
            continue;
        };
        headers.push(Ff2xPathHeader {
            path_index,
            sigma2,
            criterion,
            degeneracy,
            leg_count,
            effective_half_path_length_angstrom,
        });
    }
    Ok(headers)
}

fn ff2x_test_dw_factor(
    ck: Complex64,
    sigma2_bohr2: f64,
    cumulants: super::Ff2xPathCumulants,
) -> Complex64 {
    let ck2 = ck * ck;
    let ck3 = ck2 * ck;
    (ck2 * Complex64::new(-2.0 * sigma2_bohr2, 0.0)).exp()
        * (ck * Complex64::new(0.0, 2.0 * cumulants.first_cumulant_bohr)).exp()
        * (ck3 * Complex64::new(0.0, -4.0 * cumulants.third_cumulant_bohr3 / 3.0)).exp()
}

fn ff2x_test_imaginary_correction(ck: Complex64, vicorr: f64, reff_bohr: f64) -> f64 {
    let shifted_momentum = (ck * ck + Complex64::new(0.0, 2.0 * vicorr / FEFF_HARTREE_EV)).sqrt();
    (2.0 * reff_bohr * (ck.im - shifted_momentum.im)).exp()
}

fn complex_from_polar(magnitude: f64, phase: f64) -> Complex64 {
    Complex64::new(magnitude * phase.cos(), magnitude * phase.sin())
}

fn expected_ff2x_path_signal(
    amplitude: f64,
    phase: f64,
    momentum: f64,
    reff_bohr: f64,
) -> Complex64 {
    let angle = 2.0 * momentum * reff_bohr + phase;
    Complex64::new(amplitude * angle.cos(), amplitude * angle.sin())
}

fn assert_chi_dat_close(actual: &ChiDatData, expected: &ChiDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    for row in 0..actual.point_count() {
        assert_close(actual.wave_number[row], expected.wave_number[row], 1.0e-12);
        assert_close(actual.chi[row], expected.chi[row], 5.0e-7);
        assert_close(actual.magnitude[row], expected.magnitude[row], 5.0e-7);
        assert_close(actual.phase[row], expected.phase[row], 2.0e-5);
    }
}

fn assert_xmu_dat_close(actual: &XmuDatData, expected: &XmuDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_eq!(
        actual.normalization.is_some(),
        expected.normalization.is_some()
    );
    if let (Some(actual), Some(expected)) = (actual.normalization, expected.normalization) {
        assert_close(actual, expected, 5.0e-9);
    }
    for row in 0..actual.point_count() {
        assert_close(
            actual.photon_energy_ev[row],
            expected.photon_energy_ev[row],
            1.0e-3,
        );
        assert_close(
            actual.relative_energy_ev[row],
            expected.relative_energy_ev[row],
            1.0e-3,
        );
        assert_close(actual.wave_number[row], expected.wave_number[row], 1.0e-12);
        assert_close(actual.mu[row], expected.mu[row], 1.0e-5);
        assert_close(actual.mu0[row], expected.mu0[row], 1.0e-5);
        assert_close(actual.chi[row], expected.chi[row], 1.0e-6);
    }
}

fn assert_xmul_dat_close(actual: &XmulDatData, expected: &XmulDatData) {
    assert_eq!(actual.point_count(), expected.point_count());
    assert_eq!(actual.channel_count(), expected.channel_count());
    for row in 0..actual.point_count() {
        assert_close(
            actual.photon_energy_ev[row],
            expected.photon_energy_ev[row],
            1.0e-3,
        );
        assert_close(actual.wave_number[row], expected.wave_number[row], 1.0e-3);
        assert_close(
            actual.total_single_electron[row],
            expected.total_single_electron[row],
            1.1e-7,
        );
        for channel in 0..actual.channel_count() {
            assert_close(
                actual.channel_background[(row, channel)],
                expected.channel_background[(row, channel)],
                1.1e-7,
            );
            for l_star in 0..actual.channel_count() {
                assert_close(
                    actual.normalized_fine_structure[(row, l_star, channel)],
                    expected.normalized_fine_structure[(row, l_star, channel)],
                    2.0e-4,
                );
            }
        }
    }
}

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert_close(actual.re, expected.re, tolerance);
    assert_close(actual.im, expected.im, tolerance);
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual} differs from expected {expected} by more than {tolerance}"
    );
}
