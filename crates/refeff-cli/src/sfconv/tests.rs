use super::{has_cached_self_output, run_for_input, run_in_dir, run_self_in_dir};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, array};
use refeff_core::{
    SfconvSo2convMaterialInput, sfconv_plasmon_threshold_momentum,
    sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
};
use refeff_io::{
    ExcDatData, ListDatData, ListDatEntry, SFCONV_SO2CONV_CONVOLUTED_MARKER, SfconvSpecfunctData,
    read_exc_dat, sfconv_apl_dat_string, sfconv_rdeps_fallback_poles, write_exc_dat,
    write_list_dat, write_specfunct_dat,
};
use std::path::{Path, PathBuf};

#[test]
fn sfconv_module_writes_empty_log_when_disabled() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 0)?;

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
        eprintln!("skipping SFCONV reference test; generated XANES/Cu reference not found");
        return Ok(());
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

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!temp.path().join("exc.dat").exists());
    assert!(!temp.path().join("apl.dat").exists());
    assert_eq!(
        std::fs::read_to_string(temp.path().join("logsfconv.dat"))?,
        "Calculating S0^2 ...\n"
    );
    Ok(())
}

#[test]
fn sfconv_module_reads_existing_so2conv_material_header_before_stop() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled SFCONV should still stop before numerical SO2CONV")?;

    let message = error.to_string();
    assert!(message.contains("read 1 existing target data file(s)"));
    assert!(message.contains("first target xmu.dat"));
    assert!(message.contains("24 row(s)"));
    assert!(message.contains("Gam_ch=1.729000"));
    assert!(message.contains("Rs_int=2.050"));
    Ok(())
}

#[test]
fn sfconv_module_applies_compatible_specfunct_cache() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 1)?;

    let count = run_in_dir(temp.path())?;

    assert_eq!(count, 1);
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
fn sfconv_module_reports_incompatible_specfunct_cache_before_stop() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_sfconv_input(temp.path(), 1)?;
    write_xmu_header(temp.path(), false)?;
    write_specfunct_cache(temp.path(), 0)?;

    let error = run_in_dir(temp.path())
        .err()
        .context("enabled SFCONV should still stop before numerical SO2CONV")?;

    let message = error.to_string();
    assert!(message.contains("specfunct.dat cache npl=1, nqpts=66"));
    assert!(message.contains("compatible target(s): none"));
    assert!(message.contains("incompatible target(s): xmu.dat"));
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

    let count = run_self_in_dir(temp.path())?;

    assert_eq!(count, 0);
    assert!(!has_cached_self_output(temp.path())?);
    Ok(())
}

#[test]
fn self_module_rejects_generation_until_solver_is_ported() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;

    let error = run_self_in_dir(temp.path())
        .err()
        .context("enabled SELF should require the numerical solver")?;

    assert!(
        error.to_string().contains(
            "SELF excitation-pole generation requires the unported SELF numerical solver"
        )
    );
    Ok(())
}

#[test]
fn self_module_roundtrips_cached_exc_dat() -> Result<()> {
    let temp = tempfile::tempdir()?;
    write_self_input(temp.path())?;
    let path = temp.path().join("exc.dat");
    write_exc_dat(&path, &sample_exc_dat())?;
    let expected = read_exc_dat(&path)?;

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

fn sample_exc_dat() -> ExcDatData {
    ExcDatData {
        header_lines: vec!["# SELF excitation poles".to_string()],
        energy_ev: Array1::from_vec(vec![15.0, 27.5]),
        broadening_ev: Array1::from_vec(vec![0.15, 0.275]),
        oscillator_strength: Array1::from_vec(vec![0.75, 0.25]),
        auxiliary_weight: Some(Array1::from_vec(vec![1.0, 0.5])),
    }
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

fn reference_sfconv_dir() -> Result<Option<PathBuf>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .context("failed to find workspace root")?;
    let path = workspace.join("reference-work/golden/XANES/Cu");
    Ok((path.join("sfconv.inp").is_file() && path.join("logsfconv.dat").is_file()).then_some(path))
}
