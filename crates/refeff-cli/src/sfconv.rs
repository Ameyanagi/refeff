use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::Array1;
use refeff_core::{
    SFCONV_SO2CONV_BOHR_ANGSTROM, SfconvPhotoelectronMomentumInput, SfconvSo2convMaterialInput,
    SfconvSo2convSelfEnergyGridInput, SfconvSo2convSelfEnergySampleInput,
    sfconv_plasmon_threshold_momentum, sfconv_so2conv_broadened_self_energy_grid,
    sfconv_so2conv_broadened_self_energy_sample, sfconv_so2conv_material_parameters,
    sfconv_so2conv_momentum_grid, sfconv_so2conv_photoelectron_momentum,
};
use refeff_io::{
    EelsInput, ExcDatData, SfconvInput, SfconvRdepsPoleTable, SfconvSo2convTarget,
    SfconvSo2convTargetData, SfconvSo2convTargetKind, SfconvSpecfunctCompatibilityInput,
    SfconvSpecfunctData, SfconvSpecfunctTargetDataInput, parse_specfunct_dat, read_exc_dat,
    read_list_dat, read_or_create_sfconv_rdeps, sfconv_so2conv_target_data_from_text,
    sfconv_so2conv_targets, sfconv_specfunct_matches_so2conv_inputs,
    sfconv_specfunct_target_data_from_cache, write_exc_dat, write_sfconv_apl_dat,
    write_sfconv_so2conv_convoluted_target_data,
};

use crate::work_dir_for_input;

const SO2CONV_WORK_LEN: usize = 401;

/// Run FEFF `SFCONV` startup behavior beside the requested input.
///
/// The full `SO2CONV` spectral-function generator is still unported. This
/// function preserves the FEFF module boundary for disabled SFCONV inputs and
/// applies compatible reusable `specfunct.dat` caches for enabled inputs.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Run the supported FEFF SELF cached-output path beside the requested input.
pub(crate) fn run_self_for_input(input: &Path) -> Result<usize> {
    run_self_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF SELF run can be satisfied from an existing `exc.dat` cache.
pub(crate) fn has_cached_self_output(work_dir: &Path) -> Result<bool> {
    let path = work_dir.join("exc.dat");
    if !path.is_file() {
        return Ok(false);
    }
    Ok(self_enabled(&read_input(work_dir)?))
}

/// Run the supported `SFCONV` path from an existing `sfconv.inp`.
///
/// Enabled S0^2 runs are satisfied only when every discovered target can reuse
/// the current `specfunct.dat` cache. Otherwise the function stops before the
/// still-unported spectral-function generator.
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
        if let Some(cache) = cache.as_ref()
            && cache.incompatible_targets.is_empty()
            && cache.compatible_targets.len() == target_data.len()
        {
            return so2conv_apply_specfunct_cache_for_dir(work_dir, &cache.data, &target_data);
        }
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

/// Run the FEFF SELF cached-output path from an existing excitation-pole table.
///
/// The SELF many-pole generator is still unported. This keeps cached FEFF
/// `exc.dat` files usable by validating and re-rendering them through the typed
/// codec when `sfconv.inp` enables the SELF branch.
pub(crate) fn run_self_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !self_enabled(&input) {
        return Ok(0);
    }

    let path = work_dir.join("exc.dat");
    if !path.is_file() {
        bail!("SELF excitation-pole generation requires the unported SELF numerical solver");
    }
    let data = read_exc_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    write_self_cache(&path, &data)?;
    Ok(data.pole_count())
}

fn self_enabled(input: &SfconvInput) -> bool {
    input.control.ipse != 0
}

fn read_input(work_dir: &Path) -> Result<SfconvInput> {
    let input_path = work_dir.join("sfconv.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    SfconvInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_self_cache(path: &Path, data: &ExcDatData) -> Result<()> {
    write_exc_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
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

#[derive(Debug, Clone)]
struct So2convSpecfunctCachePreflight {
    data: SfconvSpecfunctData,
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

    let pole_count = cache.pole_count;
    let momentum_count = cache.momentum_count();
    Ok(Some(So2convSpecfunctCachePreflight {
        data: cache,
        pole_count,
        momentum_count,
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

fn so2conv_apply_specfunct_cache_for_dir(
    work_dir: &Path,
    cache: &SfconvSpecfunctData,
    target_data: &[ExistingSo2convTargetData],
) -> Result<usize> {
    for existing in target_data {
        let (photoelectron_momentum, work_len) =
            so2conv_photoelectron_momentum_for_target(cache, &existing.data).with_context(
                || {
                    format!(
                        "failed to compute SO2CONV photoelectron momentum for {}",
                        existing.target.file_name
                    )
                },
            )?;
        let convolved = sfconv_specfunct_target_data_from_cache(SfconvSpecfunctTargetDataInput {
            cache,
            source: &existing.data,
            photoelectron_momentum: photoelectron_momentum.view(),
            work_len,
        })
        .with_context(|| {
            format!(
                "failed to apply SO2CONV specfunct.dat cache to {}",
                existing.target.file_name
            )
        })?;
        let path = work_dir.join(&existing.target.file_name);
        write_sfconv_so2conv_convoluted_target_data(&path, &convolved)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(target_data.len())
}

fn so2conv_photoelectron_momentum_for_target(
    cache: &SfconvSpecfunctData,
    target: &SfconvSo2convTargetData,
) -> Result<(Array1<f64>, usize)> {
    let material = sfconv_so2conv_material_parameters(target.header().material)
        .context("failed to derive SO2CONV material parameters")?;
    let work_len = SO2CONV_WORK_LEN.max(target.point_count());
    let momentum = so2conv_target_momentum(target, work_len);
    let fermi_self_energy =
        sfconv_so2conv_broadened_self_energy_sample(SfconvSo2convSelfEnergySampleInput {
            material,
            energy: 0.0,
            quasiparticle_energy: material.fermi_energy,
            photoelectron_momentum: material.fermi_momentum,
            pole_count: cache.pole_count,
            pole_energy: cache.pole_energy.view(),
            pole_weight: cache.pole_weight.view(),
            pole_broadening: cache.pole_broadening.view(),
            include_below_fermi: false,
        })
        .context("failed to derive SO2CONV Fermi self energy")?;
    let fermi_level = material.fermi_energy + fermi_self_energy;
    let self_energy_grid =
        sfconv_so2conv_broadened_self_energy_grid(SfconvSo2convSelfEnergyGridInput {
            momentum: momentum.view(),
            chemical_potential: material.chemical_potential_offset,
            fermi_level,
            material,
            pole_count: cache.pole_count,
            pole_energy: cache.pole_energy.view(),
            pole_weight: cache.pole_weight.view(),
            pole_broadening: cache.pole_broadening.view(),
            include_below_fermi: false,
        })
        .context("failed to derive SO2CONV self-energy grid")?;
    let momentum = sfconv_so2conv_photoelectron_momentum(SfconvPhotoelectronMomentumInput {
        momentum: momentum.view(),
        chemical_potential: material.chemical_potential_offset,
        fermi_momentum: material.fermi_momentum,
        fermi_level,
        fermi_self_energy: self_energy_grid.fermi_self_energy,
        self_energy: self_energy_grid.self_energy.view(),
    })
    .context("failed to refine SO2CONV photoelectron momentum")?;
    Ok((momentum.photoelectron_momentum, work_len))
}

fn so2conv_target_momentum(target: &SfconvSo2convTargetData, work_len: usize) -> Array1<f64> {
    match target {
        SfconvSo2convTargetData::Xmu { data, .. } => data
            .wave_number
            .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM),
        SfconvSo2convTargetData::Chi { data, .. } => data
            .wave_number
            .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM),
        SfconvSo2convTargetData::FeffPath { .. } => Array1::from_shape_fn(work_len, |row| {
            0.05 * row as f64 / SFCONV_SO2CONV_BOHR_ANGSTROM
        }),
    }
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
    use super::{has_cached_self_output, run_for_input, run_in_dir, run_self_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, array};
    use refeff_core::{
        SfconvSo2convMaterialInput, sfconv_plasmon_threshold_momentum,
        sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
    };
    use refeff_io::{
        ExcDatData, ListDatData, ListDatEntry, SFCONV_SO2CONV_CONVOLUTED_MARKER,
        SfconvSpecfunctData, read_exc_dat, sfconv_apl_dat_string, sfconv_rdeps_fallback_poles,
        write_exc_dat, write_list_dat, write_specfunct_dat,
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

        assert!(error.to_string().contains(
            "SELF excitation-pole generation requires the unported SELF numerical solver"
        ));
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
        Ok(
            (path.join("sfconv.inp").is_file() && path.join("logsfconv.dat").is_file())
                .then_some(path),
        )
    }
}
