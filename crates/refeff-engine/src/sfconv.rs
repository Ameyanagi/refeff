use std::io::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, ArrayView1};
use refeff_core::{
    SFCONV_SO2CONV_BOHR_ANGSTROM, SfconvPhotoelectronMomentumInput,
    SfconvSo2convSelfEnergyGridInput, SfconvSo2convSpecfunctInput, make_excitation_poles,
    sfconv_plasmon_threshold_momentum, sfconv_so2conv_broadened_self_energy_grid,
    sfconv_so2conv_material_parameters, sfconv_so2conv_momentum_grid,
    sfconv_so2conv_photoelectron_momentum, sfconv_so2conv_specfunct_grid,
};
use refeff_io::{
    EelsInput, ExcDatData, SfconvInput, SfconvRdepsPoleTable, SfconvSo2convTarget,
    SfconvSo2convTargetData, SfconvSo2convTargetKind, SfconvSpecfunctCompatibilityInput,
    SfconvSpecfunctData, SfconvSpecfunctSpectralRowsInput, SfconvSpecfunctTargetDataInput,
    XsphInput, exc_dat_from_excitation_poles, exc_dat_string, parse_specfunct_dat, read_exc_dat,
    read_list_dat, read_loss_dat, read_or_create_sfconv_rdeps,
    sfconv_so2conv_target_data_from_text, sfconv_so2conv_targets,
    sfconv_specfunct_data_from_spectral_rows, sfconv_specfunct_matches_so2conv_inputs,
    sfconv_specfunct_target_data_from_cache, write_exc_dat, write_sfconv_apl_dat,
    write_sfconv_so2conv_convoluted_target_data, write_specfunct_dat,
};

use crate::work_dir_for_input;

const SO2CONV_WORK_LEN: usize = 401;
const SO2CONV_POLE_CAPACITY: usize = 5000;
const SO2CONV_LOG_START: &str = "Calculating S0^2 ...\n";
const SO2CONV_LOG_DONE: &str = "Done with module: S0^2.\r\n\n";

/// Run FEFF `SFCONV` startup behavior beside the requested input.
///
/// Disabled SFCONV inputs only refresh the module log. Enabled S0^2 inputs
/// reuse compatible `specfunct.dat` caches when possible and otherwise run the
/// ported SO2CONV spectral-function generator before convolution.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Run the supported FEFF SELF cached-output path beside the requested input.
pub(crate) fn run_self_for_input(input: &Path) -> Result<usize> {
    run_self_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF SELF run can be satisfied from an existing `exc.dat` cache.
pub(crate) fn has_cached_self_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("sfconv.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !self_enabled(&input) {
        return Ok(false);
    }
    let path = work_dir.join("exc.dat");
    if path.is_file() && read_exc_dat(&path).is_ok() {
        return Ok(validate_declared_self_source_handoff(work_dir).is_ok());
    }
    has_supported_self_source_handoff_for_input(work_dir, &input)
}

/// Whether FEFF SFCONV can generate S0^2-convoluted targets from supported
/// source handoffs.
pub(crate) fn has_supported_sfconv_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("sfconv.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.msfconv != 1 {
        return Ok(false);
    }
    let Ok(targets) = so2conv_targets_for_dir(work_dir, &input) else {
        return Ok(false);
    };
    let Ok(target_data) = so2conv_existing_target_data_for_dir(work_dir, &targets) else {
        return Ok(false);
    };
    Ok(!target_data.is_empty())
}

/// Whether FEFF SELF can generate `exc.dat` from `xsph.inp` plus `loss.dat`.
pub(crate) fn has_supported_self_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("sfconv.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    has_supported_self_source_handoff_for_input(work_dir, &input)
}

fn has_supported_self_source_handoff_for_input(
    work_dir: &Path,
    input: &SfconvInput,
) -> Result<bool> {
    if !self_enabled(input)
        || !work_dir.join("xsph.inp").is_file()
        || !work_dir.join("loss.dat").is_file()
    {
        return Ok(false);
    }
    let Ok(xsph) = read_xsph_input(work_dir) else {
        return Ok(false);
    };
    Ok(xsph.control.i_plsmn > 0
        && xsph.control.ixc == 0
        && usize::try_from(xsph.control.n_poles)
            .ok()
            .is_some_and(|count| count > 0)
        && generate_self_exc_dat(work_dir).is_ok())
}

/// Run the supported `SFCONV` path from an existing `sfconv.inp`.
///
/// Enabled S0^2 runs reuse compatible `specfunct.dat` caches when possible and
/// otherwise generate the FEFF `SO2CONV` spectral-function cache before applying
/// it to each selected spectrum target.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    let log_path = work_dir.join("logsfconv.dat");
    let log_text = if input.control.msfconv == 1 {
        SO2CONV_LOG_START
    } else {
        ""
    };
    std::fs::write(&log_path, log_text)
        .with_context(|| format!("failed to write {}", log_path.display()))?;

    if input.control.msfconv == 1 {
        let targets = so2conv_targets_for_dir(work_dir, &input)?;
        let target_data = so2conv_existing_target_data_for_dir(work_dir, &targets)?;
        if target_data.is_empty() {
            finish_so2conv_log(&log_path)?;
            return Ok(0);
        }
        let mut cache = so2conv_specfunct_cache_preflight_for_dir(work_dir, &target_data)?;
        if let Some(cache) = cache.as_ref()
            && cache.incompatible_targets.is_empty()
            && cache.compatible_targets.len() == target_data.len()
        {
            let processed =
                so2conv_apply_specfunct_cache_for_dir(work_dir, &cache.data, &target_data)?;
            finish_so2conv_log(&log_path)?;
            return Ok(processed);
        }

        let mut processed = 0;
        for existing in &target_data {
            let target_cache = if let Some(cache) = cache.as_ref()
                && so2conv_specfunct_cache_matches_target(work_dir, &cache.data, existing)?
            {
                cache.data.clone()
            } else {
                so2conv_generate_specfunct_cache_for_target(work_dir, existing).with_context(
                    || {
                        format!(
                            "failed to generate SO2CONV specfunct.dat for {}",
                            existing.target.file_name
                        )
                    },
                )?
            };
            processed += so2conv_apply_specfunct_cache_for_dir(
                work_dir,
                &target_cache,
                std::slice::from_ref(existing),
            )?;
            cache = Some(So2convSpecfunctCachePreflight {
                compatible_targets: vec![existing.target.file_name.clone()],
                incompatible_targets: Vec::new(),
                data: target_cache,
            });
        }
        finish_so2conv_log(&log_path)?;
        return Ok(processed);
    }

    Ok(0)
}

/// Run the FEFF SELF excitation-pole path.
///
/// Existing FEFF `exc.dat` files are validated and re-rendered through the typed
/// codec. When `exc.dat` is absent, the ported `SELF/MkExc` many-pole generator
/// builds it from `loss.dat` using `xsph.inp` MPSE controls.
pub(crate) fn run_self_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !self_enabled(&input) {
        return Ok(0);
    }
    validate_declared_self_source_handoff(work_dir)?;

    let path = work_dir.join("exc.dat");
    let data = if let Some(generated) = recover_self_exc_dat_from_source_if_available(work_dir)? {
        if path.is_file() {
            match read_exc_dat(&path) {
                Ok(cached) if self_exc_dat_matches_generated(&cached, &generated)? => cached,
                Ok(_) | Err(_) => generated,
            }
        } else {
            generated
        }
    } else if path.is_file() {
        read_exc_dat(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        generate_self_exc_dat(work_dir)?
    };
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

fn recover_self_exc_dat_from_source_if_available(work_dir: &Path) -> Result<Option<ExcDatData>> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(None);
    }
    validate_declared_self_source_handoff(work_dir)?;
    if !work_dir.join("loss.dat").is_file() {
        return Ok(None);
    }
    if !has_supported_self_source_handoff_for_input(work_dir, &read_input(work_dir)?)? {
        return Ok(None);
    }
    generate_self_exc_dat(work_dir).map(Some)
}

fn validate_declared_self_source_handoff(work_dir: &Path) -> Result<()> {
    if !work_dir.join("xsph.inp").is_file() {
        return Ok(());
    }

    read_xsph_input(work_dir)?;
    let loss_path = work_dir.join("loss.dat");
    if loss_path.is_file() {
        read_loss_dat(&loss_path)
            .with_context(|| format!("failed to read {}", loss_path.display()))?;
    }
    Ok(())
}

fn self_exc_dat_matches_generated(cached: &ExcDatData, generated: &ExcDatData) -> Result<bool> {
    Ok(exc_dat_string(cached)? == exc_dat_string(generated)?)
}

fn generate_self_exc_dat(work_dir: &Path) -> Result<ExcDatData> {
    let xsph = read_xsph_input(work_dir)?;
    if xsph.control.i_plsmn <= 0 {
        bail!(
            "SELF excitation-pole generation requires MPSE/plasmon controls in xsph.inp when exc.dat is absent"
        );
    }
    if xsph.control.ixc != 0 {
        bail!(
            "SELF excitation-pole generation from loss.dat requires Hedin-Lundqvist exchange (ixc = 0), got {}",
            xsph.control.ixc
        );
    }
    let requested_poles = usize::try_from(xsph.control.n_poles)
        .ok()
        .filter(|&count| count > 0)
        .context("SELF excitation-pole generation requires a positive NPoles value in xsph.inp")?;
    let loss_path = work_dir.join("loss.dat");
    let loss = read_loss_dat(&loss_path)
        .with_context(|| format!("failed to read {}", loss_path.display()))?;
    let poles = make_excitation_poles(
        loss.energy_ev.view(),
        loss.loss.view(),
        xsph.grid.eps0,
        requested_poles,
    )
    .context("failed to calculate SELF excitation-pole table")?;
    exc_dat_from_excitation_poles(&poles).context("failed to assemble SELF exc.dat")
}

fn read_xsph_input(work_dir: &Path) -> Result<XsphInput> {
    let path = work_dir.join("xsph.inp");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    XsphInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn finish_so2conv_log(log_path: &Path) -> Result<()> {
    std::fs::OpenOptions::new()
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?
        .write_all(SO2CONV_LOG_DONE.as_bytes())
        .with_context(|| format!("failed to write {}", log_path.display()))
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
        data: cache,
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

fn so2conv_generate_specfunct_cache_for_target(
    work_dir: &Path,
    existing: &ExistingSo2convTargetData,
) -> Result<SfconvSpecfunctData> {
    let material = existing.data.header().material;
    let parameters = sfconv_so2conv_material_parameters(material)
        .context("failed to derive SO2CONV material parameters")?;
    let poles = read_or_create_sfconv_rdeps(
        work_dir.join("exc.dat"),
        parameters.plasma_frequency,
        SO2CONV_POLE_CAPACITY,
    )
    .context("failed to read SO2CONV excitation-pole input")?;
    write_sfconv_apl_dat(work_dir.join("apl.dat"), &poles)
        .context("failed to write SO2CONV apl.dat pole diagnostics")?;

    let pole_energy = so2conv_padded_pole_array(poles.energy_hartree.view(), SO2CONV_POLE_CAPACITY);
    let pole_broadening =
        so2conv_padded_pole_array(poles.broadening_hartree.view(), SO2CONV_POLE_CAPACITY);
    let pole_weight = so2conv_padded_pole_array(
        so2conv_pole_weights(&poles, parameters.plasma_frequency)?.view(),
        SO2CONV_POLE_CAPACITY,
    );
    let threshold_momentum = sfconv_plasmon_threshold_momentum(
        parameters.plasma_frequency,
        parameters.dispersion_parameter,
        parameters.fermi_energy,
        parameters.fermi_momentum,
    )
    .context("failed to derive SO2CONV plasmon threshold")?;
    let momentum_grid = sfconv_so2conv_momentum_grid(parameters.fermi_momentum, threshold_momentum)
        .context("failed to derive SO2CONV momentum grid")?;
    let generated = sfconv_so2conv_specfunct_grid(SfconvSo2convSpecfunctInput {
        material: parameters,
        momentum_grid: momentum_grid.view(),
        pole_count: poles.pole_count(),
        pole_energy: pole_energy.view(),
        pole_weight: pole_weight.view(),
        pole_broadening: pole_broadening.view(),
    })
    .context("failed to calculate SO2CONV spectral-function grid")?;

    let cache = sfconv_specfunct_data_from_spectral_rows(SfconvSpecfunctSpectralRowsInput {
        wigner_seitz_radius: material.wigner_seitz_radius,
        core_hole_lifetime: parameters.core_hole_lifetime,
        asymmetric_phase: so2conv_target_asymmetric_phase(&existing.target),
        satellite_type: 0,
        low_q_mode: 0,
        pole_count: poles.pole_count(),
        pole_energy: pole_energy.view(),
        pole_broadening: pole_broadening.view(),
        pole_weight: pole_weight.view(),
        spectral_info: generated.spectral_info.view(),
        weights: generated.weights.view(),
        spectral_function: generated.spectral_function.view(),
        energy_grid: generated.energy_grid.view(),
    })
    .context("failed to assemble SO2CONV specfunct.dat cache")?;
    write_specfunct_dat(work_dir.join("specfunct.dat"), &cache)
        .context("failed to write SO2CONV specfunct.dat")?;
    Ok(cache)
}

fn so2conv_padded_pole_array(values: ArrayView1<'_, f64>, capacity: usize) -> Array1<f64> {
    let mut padded = Array1::zeros(capacity);
    for (index, &value) in values.iter().enumerate() {
        padded[index] = value;
    }
    padded
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
    // `so2conv.f90` refines the target momentum before `mkspectf` assigns the
    // COMMON-block `fmu`. A fresh SFCONV process therefore uses its
    // zero-initialized value here, including when a compatible cache is read.
    let fermi_level = 0.0;
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
            .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM),
        SfconvSo2convTargetData::Chi { data, .. } => data
            .wave_number
            .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM),
        SfconvSo2convTargetData::FeffPath { .. } => Array1::from_shape_fn(work_len, |row| {
            0.05 * row as f64 * SFCONV_SO2CONV_BOHR_ANGSTROM
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

#[cfg(test)]
mod tests;
