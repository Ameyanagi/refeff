use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    SfconvSo2convMaterialInput, sfconv_plasmon_threshold_momentum,
    sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
};
use refeff_io::{
    EelsInput, SfconvInput, SfconvRdepsPoleTable, SfconvSo2convTarget, SfconvSo2convTargetData,
    SfconvSo2convTargetKind, SfconvSpecfunctCompatibilityInput, SfconvSpecfunctData,
    parse_specfunct_dat, read_list_dat, read_or_create_sfconv_rdeps,
    sfconv_so2conv_target_data_from_text, sfconv_so2conv_targets,
    sfconv_specfunct_matches_so2conv_inputs, write_sfconv_apl_dat,
};

use crate::work_dir_for_input;

/// Run FEFF `SFCONV` startup behavior beside the requested input.
///
/// The full `SO2CONV` spectral-function generator is still unported. This
/// function preserves the FEFF module boundary for disabled SFCONV inputs and
/// preflights reusable `specfunct.dat` caches for enabled inputs.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Run the supported disabled `SFCONV` path from an existing `sfconv.inp`.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let log_path = work_dir.join("logsfconv.dat");
    let log_text = if input.control.msfconv == 1 {
        "Calculating S0^2 ...\n"
    } else {
        ""
    };
    std::fs::write(&log_path, log_text)
        .with_context(|| format!("failed to write {}", log_path.display()))?;

    if input.control.msfconv == 1 {
        let targets = so2conv_targets_for_dir(work_dir, &input)?;
        let target_data = so2conv_existing_target_data_for_dir(work_dir, &targets)?;
        if target_data.is_empty() {
            return Ok(0);
        }
        let cache = so2conv_specfunct_cache_preflight_for_dir(work_dir, &target_data)?;
        bail!(
            "SFCONV S0^2 convolution requires the unported SO2CONV numerical driver; discovered {} target file(s): {}; read {} existing target data file(s){}{}",
            targets.len(),
            so2conv_target_summary(&targets),
            target_data.len(),
            so2conv_material_summary(&target_data),
            so2conv_cache_summary(cache.as_ref())
        );
    }

    Ok(0)
}

fn read_input(work_dir: &Path) -> Result<SfconvInput> {
    let input_path = work_dir.join("sfconv.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    SfconvInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn so2conv_targets_for_dir(
    work_dir: &Path,
    input: &SfconvInput,
) -> Result<Vec<SfconvSo2convTarget>> {
    let list = if input.spectrum.ispec.abs() == 0 && input.spectrum.ipr6 >= 2 {
        let list_path = work_dir.join("list.dat");
        Some(read_list_dat(&list_path).with_context(|| {
            format!(
                "failed to read SO2CONV path selection from {}",
                list_path.display()
            )
        })?)
    } else {
        None
    };
    let eels = read_optional_eels_input(work_dir)?;
    sfconv_so2conv_targets(input, list.as_ref(), eels.as_ref())
        .context("failed to select SO2CONV target files")
}

fn read_optional_eels_input(work_dir: &Path) -> Result<Option<EelsInput>> {
    let path = work_dir.join("eels.inp");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    EelsInput::parse_str(&path, &text)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

#[derive(Debug, Clone)]
struct ExistingSo2convTargetData {
    target: SfconvSo2convTarget,
    data: SfconvSo2convTargetData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct So2convSpecfunctCachePreflight {
    pole_count: usize,
    momentum_count: usize,
    compatible_targets: Vec<String>,
    incompatible_targets: Vec<String>,
}

fn so2conv_existing_target_data_for_dir(
    work_dir: &Path,
    targets: &[SfconvSo2convTarget],
) -> Result<Vec<ExistingSo2convTargetData>> {
    let mut target_data = Vec::new();
    for target in targets {
        let path = work_dir.join(&target.file_name);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        let data =
            sfconv_so2conv_target_data_from_text(&path, target, &text).with_context(|| {
                format!("failed to parse SO2CONV target data in {}", path.display())
            })?;
        if data.header().already_convoluted {
            bail!(
                "SFCONV target {} has already been convoluted with A(omega); rerun the upstream spectrum module before SO2CONV",
                path.display()
            );
        }
        target_data.push(ExistingSo2convTargetData {
            target: target.clone(),
            data,
        });
    }
    Ok(target_data)
}

fn so2conv_specfunct_cache_preflight_for_dir(
    work_dir: &Path,
    target_data: &[ExistingSo2convTargetData],
) -> Result<Option<So2convSpecfunctCachePreflight>> {
    if target_data.is_empty() {
        return Ok(None);
    }

    let path = work_dir.join("specfunct.dat");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let cache = parse_specfunct_dat(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let mut compatible_targets = Vec::new();
    let mut incompatible_targets = Vec::new();
    for existing in target_data {
        let matches = so2conv_specfunct_cache_matches_target(work_dir, &cache, existing)?;
        if matches {
            compatible_targets.push(existing.target.file_name.clone());
        } else {
            incompatible_targets.push(existing.target.file_name.clone());
        }
    }

    Ok(Some(So2convSpecfunctCachePreflight {
        pole_count: cache.pole_count,
        momentum_count: cache.momentum_count(),
        compatible_targets,
        incompatible_targets,
    }))
}

fn so2conv_specfunct_cache_matches_target(
    work_dir: &Path,
    cache: &SfconvSpecfunctData,
    existing: &ExistingSo2convTargetData,
) -> Result<bool> {
    let material = existing.data.header().material;
    let parameters = sfconv_so2conv_material_parameters(material)
        .context("failed to derive SO2CONV material parameters")?;
    let poles = read_or_create_sfconv_rdeps(
        work_dir.join("exc.dat"),
        parameters.plasma_frequency,
        cache.pole_capacity(),
    )
    .context("failed to read SO2CONV excitation-pole input")?;
    write_sfconv_apl_dat(work_dir.join("apl.dat"), &poles)
        .context("failed to write SO2CONV apl.dat pole diagnostics")?;
    let pole_weight = so2conv_pole_weights(&poles, parameters.plasma_frequency)?;
    let threshold_momentum = sfconv_plasmon_threshold_momentum(
        parameters.plasma_frequency,
        parameters.dispersion_parameter,
        parameters.fermi_energy,
        parameters.fermi_momentum,
    )
    .context("failed to derive SO2CONV plasmon threshold")?;
    let momentum_grid = sfconv_so2conv_momentum_grid(parameters.fermi_momentum, threshold_momentum)
        .context("failed to derive SO2CONV momentum grid")?;

    sfconv_specfunct_matches_so2conv_inputs(
        cache,
        SfconvSpecfunctCompatibilityInput {
            wigner_seitz_radius: material.wigner_seitz_radius,
            core_hole_lifetime: parameters.core_hole_lifetime,
            asymmetric_phase: so2conv_target_asymmetric_phase(&existing.target),
            satellite_type: 0,
            low_q_mode: 0,
            pole_count: poles.pole_count(),
            pole_energy: poles.energy_hartree.view(),
            pole_broadening: poles.broadening_hartree.view(),
            pole_weight: pole_weight.view(),
            momentum_grid: momentum_grid.view(),
        },
    )
    .context("failed to compare SO2CONV specfunct.dat cache")
}

fn so2conv_pole_weights(
    poles: &SfconvRdepsPoleTable,
    plasma_frequency: f64,
) -> Result<Array1<f64>> {
    if !plasma_frequency.is_finite() || plasma_frequency <= 0.0 {
        bail!("SO2CONV plasma frequency must be positive, got {plasma_frequency}");
    }
    Ok(Array1::from_iter(
        poles
            .oscillator_strength
            .iter()
            .zip(poles.energy_hartree.iter())
            .map(|(&strength, &energy)| {
                (strength * energy.powi(2) / plasma_frequency.powi(2)).abs()
            }),
    ))
}

fn so2conv_target_asymmetric_phase(target: &SfconvSo2convTarget) -> i32 {
    match target.kind {
        SfconvSo2convTargetKind::Xmu => 1,
        SfconvSo2convTargetKind::Chi | SfconvSo2convTargetKind::FeffPath => 0,
    }
}

fn so2conv_target_summary(targets: &[SfconvSo2convTarget]) -> String {
    let names = targets
        .iter()
        .take(6)
        .map(|target| target.file_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if targets.len() > 6 {
        format!("{names}, ...")
    } else {
        names
    }
}

fn so2conv_material_summary(target_data: &[ExistingSo2convTargetData]) -> String {
    match target_data.first() {
        Some(existing) => format!(
            "; first target {}: {} row(s), {}",
            existing.target.file_name,
            existing.data.point_count(),
            material_input_summary(existing.data.header().material)
        ),
        None => String::new(),
    }
}

fn so2conv_cache_summary(cache: Option<&So2convSpecfunctCachePreflight>) -> String {
    match cache {
        Some(cache) => format!(
            "; specfunct.dat cache npl={}, nqpts={}, compatible target(s): {}; incompatible target(s): {}",
            cache.pole_count,
            cache.momentum_count,
            target_name_list(&cache.compatible_targets),
            target_name_list(&cache.incompatible_targets)
        ),
        None => "; no specfunct.dat cache available for reuse".to_string(),
    }
}

fn target_name_list(targets: &[String]) -> String {
    if targets.is_empty() {
        return "none".to_string();
    }
    targets
        .iter()
        .take(6)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

fn material_input_summary(material: SfconvSo2convMaterialInput) -> String {
    format!(
        "Gam_ch={:.6}, Rs_int={:.3}, Vint={:.5}, Mu={:.5}, kf={:.6}",
        material.core_hole_width_ev,
        material.wigner_seitz_radius,
        material.interstitial_potential_ev,
        material.chemical_potential_ev,
        material.fermi_wave_number_inv_angstrom
    )
}

#[cfg(test)]
mod tests {
    use super::{run_for_input, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array2, array};
    use refeff_core::{
        SfconvSo2convMaterialInput, sfconv_plasmon_threshold_momentum,
        sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
    };
    use refeff_io::{
        SfconvSpecfunctData, sfconv_apl_dat_string, sfconv_rdeps_fallback_poles,
        write_specfunct_dat,
    };
    use std::path::Path;

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
        assert!(message.contains("1 row(s)"));
        assert!(message.contains("Gam_ch=1.729000"));
        assert!(message.contains("Rs_int=2.050"));
        Ok(())
    }

    #[test]
    fn sfconv_module_reports_compatible_specfunct_cache_before_stop() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_sfconv_input(temp.path(), 1)?;
        write_xmu_header(temp.path(), false)?;
        write_specfunct_cache(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled SFCONV should still stop before numerical SO2CONV")?;

        let message = error.to_string();
        assert!(message.contains("specfunct.dat cache npl=1, nqpts=66"));
        assert!(message.contains("compatible target(s): xmu.dat"));
        assert!(message.contains("incompatible target(s): none"));
        assert!(temp.path().join("exc.dat").is_file());
        assert_eq!(
            std::fs::read_to_string(temp.path().join("apl.dat"))?,
            expected_apl_dat()?
        );
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

    fn write_sfconv_input(work_dir: &Path, msfconv: i32) -> Result<()> {
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
                msfconv, 0, 0, 0.0, 0.0, 0, 0
            ),
        )?;
        Ok(())
    }

    fn write_xmu_header(work_dir: &Path, already_convoluted: bool) -> Result<()> {
        let marker = if already_convoluted {
            "# Convoluted with A(omega).\n"
        } else {
            ""
        };
        std::fs::write(
            work_dir.join("xmu.dat"),
            format!(
                concat!(
                    "{}",
                    "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 ",
                    "Mu= 18.76000 kf= 1.230000\n",
                    " ------------------------------------------------------------------------------\n",
                    "  0.0 0.0 0.0 0.0 0.0 0.0\n",
                ),
                marker
            ),
        )?;
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
                weights: Array2::zeros((momentum_count, 8)),
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
}
