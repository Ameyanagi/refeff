use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use refeff_core::{
    GenfmtDriverSetup, GenfmtJasDriverOutput, GenfmtJasDriverOutputInput, GenfmtJasDriverSetup,
    GenfmtJasEnergyGridBranchFromTransitionSetupInput, GenfmtJasPathEnergyGridFromSetupInput,
    GenfmtJasPathEvaluationFromDriverSetupInput, GenfmtJasTransitionSetup, GenfmtNStarDriverInput,
    GenfmtNStarRows, GenfmtOrdinaryDriverOutput, GenfmtOrdinaryDriverOutputInput,
    GenfmtOrdinaryPathEnergyGridFromDriverSetupInput,
    GenfmtOrdinaryPathEvaluationFromDriverSetupInput, GenfmtOrdinaryTransitionMatrices,
    GenfmtPathSetup, TransitionBMatrix, genfmt_jas_driver_output,
    genfmt_jas_energy_grid_branch_from_transition_setup, genfmt_ordinary_driver_output,
};
use refeff_io::{
    FeffBinData, GenfmtInput, GenfmtOutputData, GenfmtOutputPaths, GlobalInput, ListDatData,
    ModuleLogData, PathsDatGenfmtPath, PhaseBinGenfmtData, feff_bin_string, feffl_bin_string,
    genfmt_core_legendre_normalization_from_feff_dims, genfmt_driver_setup_from_handoffs,
    genfmt_edge_start_index_from_phase, genfmt_jas_driver_setup_from_handoffs,
    genfmt_jas_path_setups_from_handoffs, genfmt_jas_q_angles_from_handoffs,
    genfmt_jas_transition_indices_from_handoffs, genfmt_jas_transition_setups_from_handoff_setups,
    genfmt_nstar_driver_input_from_handoffs, genfmt_nstar_rows_from_handoffs,
    genfmt_ordinary_path_setups_from_handoffs, genfmt_ordinary_spin_radial_factors_from_phase,
    genfmt_ordinary_transition_b_matrix_from_handoffs,
    genfmt_ordinary_transition_matrices_from_handoff_setups, list_dat_string, nstar_dat_string,
    read_feff_bin, read_feffl_bin, read_list_dat, read_module_log_dat, read_nstar_dat,
    read_paths_dat, read_phase_bin, write_feff_bin, write_feffl_bin, write_genfmt_output_files,
    write_list_dat, write_module_log_dat, write_nstar_dat,
};

use crate::{fms, work_dir_for_input};

/// Run the supported FEFF GENFMT cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF GENFMT run can be satisfied from existing caches or source handoffs.
pub(crate) fn has_cached_genfmt_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("genfmt.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !genfmt_enabled(&input) {
        return Ok(false);
    }
    let caches = cached_output_paths(work_dir)?;
    if caches.has_required_base_outputs() {
        if validate_cached_outputs_readable(&caches, &input).is_ok() {
            if validate_declared_generation_handoffs(work_dir, &input).is_err() {
                return Ok(false);
            }
            return Ok(true);
        }
        return has_supported_generation_handoffs(work_dir, &input);
    }
    has_supported_generation_handoffs(work_dir, &input)
}

/// Run FEFF GENFMT from generated handoffs or existing cached output files.
///
/// When `global.inp`, `phase.bin`, and `paths.dat` are present, this generates
/// ordinary GENFMT or GENFMTJAS outputs. Otherwise it keeps cached FEFF output
/// usable by validating and re-rendering `feff.bin`, suffixed `feffNN.bin`
/// files, `list.dat`, suffixed `listNN.dat` files, and optional `nstar.dat`,
/// `feffl.bin` NRIXS decomposition caches plus `log5.dat` diagnostics.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !genfmt_enabled(&input) {
        return Ok(0);
    }

    let caches = cached_output_paths(work_dir)?;
    if !caches.has_required_base_outputs() {
        if has_supported_generation_handoffs(work_dir, &input)? {
            return generate_outputs_from_handoffs(work_dir, &input);
        }
        bail!(
            "GENFMT generation requires cached feff.bin/list.dat outputs or global.inp, phase.bin, and paths.dat handoffs"
        );
    }
    if let Err(cache_error) = validate_cached_outputs_readable(&caches, &input) {
        if has_supported_generation_handoffs(work_dir, &input)? {
            return generate_outputs_from_handoffs(work_dir, &input);
        }
        return Err(cache_error);
    }
    if cached_outputs_are_stale_against_source(work_dir, &input)? {
        return generate_outputs_from_handoffs(work_dir, &input);
    }

    let mut written = 0_usize;
    let mut base_feff = None;
    for path in caches.feff_bins {
        let data =
            read_feff_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
        if path.file_name().and_then(|name| name.to_str()) == Some("feff.bin") {
            base_feff = Some(data.clone());
        }
        write_feff_cache(&path, &data)?;
        written += 1;
    }

    for path in caches.list_dats {
        let data =
            read_list_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        write_list_cache(&path, &data)?;
        written += 1;
    }

    if let Some(path) = caches.nstar_dat {
        let data =
            read_nstar_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
        write_nstar_dat(&path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
    }

    if let Some(path) = caches.feffl_bin {
        let base = base_feff.context("feffl.bin cache requires feff.bin metadata")?;
        let max_channel = decomposition_channel(&input)?;
        let data = read_feffl_bin(
            &path,
            base.pad_width,
            base.paths.len(),
            base.energy_count(),
            max_channel,
        )
        .with_context(|| format!("failed to read {}", path.display()))?;
        write_feffl_bin(&path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
        written += 1;
    }

    written += write_or_generate_module_log(work_dir)?;
    Ok(written)
}

fn generate_outputs_from_handoffs(work_dir: &Path, input: &GenfmtInput) -> Result<usize> {
    let context = prepare_generation_context(work_dir, input)?;
    let _ = context.prepared_counts();
    let written = context.write_generated_outputs(work_dir)?;
    Ok(written + write_or_generate_module_log(work_dir)?)
}

fn validate_cached_outputs_readable(caches: &GenfmtCachePaths, input: &GenfmtInput) -> Result<()> {
    let mut base_feff = None;
    for path in &caches.feff_bins {
        let data =
            read_feff_bin(path).with_context(|| format!("failed to read {}", path.display()))?;
        if file_name_is(path, "feff.bin") {
            base_feff = Some(data);
        }
    }

    for path in &caches.list_dats {
        read_list_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    }

    if let Some(path) = &caches.nstar_dat {
        read_nstar_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    }

    if let Some(path) = &caches.feffl_bin {
        let base = base_feff.context("feffl.bin cache requires feff.bin metadata")?;
        let max_channel = decomposition_channel(input)?;
        read_feffl_bin(
            path,
            base.pad_width,
            base.paths.len(),
            base.energy_count(),
            max_channel,
        )
        .with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn cached_outputs_are_stale_against_source(work_dir: &Path, input: &GenfmtInput) -> Result<bool> {
    if !generation_handoffs_present(work_dir) {
        validate_present_generation_handoffs(work_dir)?;
        return Ok(false);
    }

    if fms::blocks_downstream_source_generation(work_dir)? {
        validate_present_generation_handoffs(work_dir)?;
        return Ok(false);
    }

    let context = prepare_generation_context(work_dir, input)?;
    let (paths, generated) = context.generated_outputs(work_dir)?;

    let cached_feff = read_feff_bin(&paths.feff_bin)
        .with_context(|| format!("failed to read {}", paths.feff_bin.display()))?;
    if feff_bin_string(&cached_feff)? != feff_bin_string(&generated.feff_bin)? {
        return Ok(true);
    }

    let cached_list = read_list_dat(&paths.list_dat)
        .with_context(|| format!("failed to read {}", paths.list_dat.display()))?;
    if list_dat_string(&cached_list)? != list_dat_string(&generated.list_dat)? {
        return Ok(true);
    }

    if let (Some(path), Some(expected)) = (&paths.nstar_dat, &generated.nstar_dat) {
        let Ok(cached) = read_nstar_dat(path) else {
            return Ok(true);
        };
        if nstar_dat_string(&cached)? != nstar_dat_string(expected)? {
            return Ok(true);
        }
    }

    if let (Some(path), Some(expected)) = (&paths.feffl_bin, &generated.feffl_bin) {
        let Ok(cached) = read_feffl_bin(
            path,
            generated.feff_bin.pad_width,
            generated.feff_bin.paths.len(),
            generated.feff_bin.energy_count(),
            decomposition_channel(input)?,
        ) else {
            return Ok(true);
        };
        if feffl_bin_string(&cached)? != feffl_bin_string(expected)? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn genfmt_enabled(input: &GenfmtInput) -> bool {
    input.control.mfeff != 0
}

fn decomposition_channel(input: &GenfmtInput) -> Result<usize> {
    if input.decomposition_channels < 0 {
        bail!("feffl.bin cache requires a nonnegative GENFMT decomposition channel count");
    }
    Ok(input.decomposition_channels as usize)
}

fn nonnegative_usize(value: i32, field: &'static str) -> Result<usize> {
    usize::try_from(value).with_context(|| format!("{field} must be nonnegative"))
}

fn read_input(work_dir: &Path) -> Result<GenfmtInput> {
    let input_path = work_dir.join("genfmt.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    GenfmtInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn generation_handoffs_present(work_dir: &Path) -> bool {
    ["global.inp", "phase.bin", "paths.dat"]
        .iter()
        .all(|name| work_dir.join(name).is_file())
}

fn validate_declared_generation_handoffs(work_dir: &Path, input: &GenfmtInput) -> Result<()> {
    if generation_handoffs_present(work_dir) && !fms::blocks_downstream_source_generation(work_dir)?
    {
        prepare_generation_context(work_dir, input)?;
    } else {
        validate_present_generation_handoffs(work_dir)?;
    }
    Ok(())
}

fn validate_present_generation_handoffs(work_dir: &Path) -> Result<()> {
    let global_path = work_dir.join("global.inp");
    if global_path.is_file() {
        read_global_input_path(&global_path)?;
    }

    let phase_path = work_dir.join("phase.bin");
    let phase = if phase_path.is_file() {
        let phase = read_phase_bin(&phase_path)
            .with_context(|| format!("failed to read {}", phase_path.display()))?;
        phase
            .to_genfmt_data()
            .context("failed to validate GENFMT phase.bin source handoff")?;
        Some(phase)
    } else {
        None
    };

    let paths_path = work_dir.join("paths.dat");
    if paths_path.is_file() {
        let paths = read_paths_dat(&paths_path)
            .with_context(|| format!("failed to read {}", paths_path.display()))?;
        if let Some(phase) = &phase {
            let max_potential_index = phase
                .potential_count()
                .checked_sub(1)
                .context("phase.bin contains no potentials for GENFMT")?;
            paths
                .to_genfmt_paths(max_potential_index)
                .with_context(|| {
                    format!("failed to convert {} for GENFMT", paths_path.display())
                })?;
        }
    }

    Ok(())
}

fn has_supported_generation_handoffs(work_dir: &Path, input: &GenfmtInput) -> Result<bool> {
    if !generation_handoffs_present(work_dir) {
        return Ok(false);
    }
    if fms::blocks_downstream_source_generation(work_dir)? {
        return Ok(false);
    }
    Ok(prepare_generation_context(work_dir, input).is_ok())
}

#[derive(Debug)]
struct GenfmtGenerationContext {
    driver: GenfmtGenerationDriver,
    titles: Vec<String>,
    paths: Vec<PathsDatGenfmtPath>,
    path_setups: Vec<GenfmtPathSetup>,
    phase_tables: PhaseBinGenfmtData,
    legendre_normalization: Array2<f64>,
    nstar_rows: Option<GenfmtNStarRows>,
    nstar_driver: Option<GenfmtNStarDriverInput>,
    edge_start_index: usize,
    active_energy_count: usize,
}

impl GenfmtGenerationContext {
    fn prepared_counts(&self) -> (usize, usize, usize) {
        debug_assert_eq!(self.paths.len(), self.path_setups.len());
        debug_assert!(self.legendre_normalization.shape()[0] >= 5);
        debug_assert!(self.edge_start_index < self.active_energy_count);
        if let Some(nstar_rows) = &self.nstar_rows {
            debug_assert_eq!(nstar_rows.rows.len(), self.path_setups.len());
        }
        let energy_count = match &self.driver {
            GenfmtGenerationDriver::Ordinary(data) => {
                debug_assert_eq!(data.transition_matrices.len(), self.path_setups.len());
                debug_assert_eq!(data.transition_b_matrix.orbital_momenta.len(), 8);
                debug_assert_eq!(data.transition_angular_momenta.len(), 8);
                data.setup.header.central_phase_shifts.len()
            }
            GenfmtGenerationDriver::Jas(data) => {
                debug_assert_eq!(data.transition_setups.len(), self.path_setups.len());
                debug_assert!(
                    data.transition_setups
                        .iter()
                        .all(|setup| setup.transition_count.transition_count > 0)
                );
                data.setup.header.central_phase_shifts.len()
            }
        };
        (energy_count, self.titles.len(), self.path_setups.len())
    }

    #[cfg(test)]
    fn validate_prepared_path_inputs(&self) -> Result<usize> {
        match &self.driver {
            GenfmtGenerationDriver::Ordinary(data) => data.validate_path_energy_inputs(
                &self.paths,
                &self.path_setups,
                &self.phase_tables,
                self.legendre_normalization.view(),
            ),
            GenfmtGenerationDriver::Jas(data) => data.validate_path_energy_inputs(
                &self.paths,
                &self.path_setups,
                &self.phase_tables,
                self.legendre_normalization.view(),
            ),
        }
    }

    fn generated_outputs(&self, work_dir: &Path) -> Result<(GenfmtOutputPaths, GenfmtOutputData)> {
        match &self.driver {
            GenfmtGenerationDriver::Ordinary(data) => {
                let output = data.driver_output(
                    &self.paths,
                    &self.path_setups,
                    &self.phase_tables,
                    self.legendre_normalization.view(),
                    self.nstar_driver,
                )?;
                let mut paths = GenfmtOutputPaths::standard(work_dir);
                paths.feffl_bin = None;
                Ok((
                    paths,
                    GenfmtOutputData::from_genfmt_ordinary_driver_output(&self.titles, &output),
                ))
            }
            GenfmtGenerationDriver::Jas(data) => {
                let output = data.driver_output(
                    &self.paths,
                    &self.path_setups,
                    &self.phase_tables,
                    self.legendre_normalization.view(),
                    self.nstar_driver,
                )?;
                Ok((
                    GenfmtOutputPaths::standard(work_dir),
                    GenfmtOutputData::from_genfmt_jas_driver_output(
                        &self.titles,
                        data.decomposition_channels,
                        &output,
                    )?,
                ))
            }
        }
    }

    fn write_generated_outputs(&self, work_dir: &Path) -> Result<usize> {
        let data = self.generated_outputs(work_dir)?;
        let written = write_genfmt_output_files(&data.0, &data.1)
            .with_context(|| format!("failed to write GENFMT outputs in {}", work_dir.display()))?;
        Ok(written)
    }
}

#[derive(Debug)]
enum GenfmtGenerationDriver {
    Ordinary(Box<GenfmtOrdinaryGenerationData>),
    Jas(Box<GenfmtJasGenerationData>),
}

#[derive(Debug)]
struct GenfmtJasGenerationData {
    setup: GenfmtJasDriverSetup,
    transition_angular_momenta: Array1<i32>,
    transition_setups: Vec<GenfmtJasTransitionSetup>,
    q_weights: Array1<Complex64>,
    max_angular_momentum: usize,
    decomposition_l_max: Option<usize>,
    decomposition_channels: usize,
    edge_start_index: usize,
    active_energy_count: usize,
    print_level: i32,
    curved_wave_criterion_percent: f64,
}

impl GenfmtJasGenerationData {
    #[cfg(test)]
    fn validate_path_energy_inputs(
        &self,
        paths: &[PathsDatGenfmtPath],
        path_setups: &[GenfmtPathSetup],
        phase_tables: &PhaseBinGenfmtData,
        legendre_normalization: ndarray::ArrayView2<'_, f64>,
    ) -> Result<usize> {
        let _ = self.driver_output(
            paths,
            path_setups,
            phase_tables,
            legendre_normalization,
            None,
        )?;
        Ok(paths.len())
    }

    fn driver_output(
        &self,
        paths: &[PathsDatGenfmtPath],
        path_setups: &[GenfmtPathSetup],
        phase_tables: &PhaseBinGenfmtData,
        legendre_normalization: ndarray::ArrayView2<'_, f64>,
        nstar: Option<GenfmtNStarDriverInput>,
    ) -> Result<GenfmtJasDriverOutput> {
        if paths.len() != path_setups.len() || paths.len() != self.transition_setups.len() {
            bail!(
                "GENFMTJAS path input count mismatch: paths={}, setups={}, transition_setups={}",
                paths.len(),
                path_setups.len(),
                self.transition_setups.len()
            );
        }

        let mut path_inputs = Vec::with_capacity(paths.len());
        for ((path, path_setup), transition_setup) in paths
            .iter()
            .zip(path_setups.iter())
            .zip(self.transition_setups.iter())
        {
            let branch = genfmt_jas_energy_grid_branch_from_transition_setup(
                GenfmtJasEnergyGridBranchFromTransitionSetupInput {
                    transition_setup,
                    transition_angular_momenta: self.transition_angular_momenta.view(),
                    radial_factors: self.setup.radial_factors.radial_factors.view(),
                    q_weights: self.q_weights.view(),
                    transition_magnetic_offset: path_setup.rotations.rotation_magnetic_offset,
                    max_angular_momentum: self.max_angular_momentum,
                    decomposition_l_max: self.decomposition_l_max,
                },
            )
            .context("failed to prepare GENFMTJAS path energy branch")?;

            path_inputs.push(GenfmtJasPathEvaluationFromDriverSetupInput {
                energy_grid: GenfmtJasPathEnergyGridFromSetupInput {
                    driver_setup: &self.setup,
                    path_setup,
                    path_potential_indices: path.potential_indices.view(),
                    angular_limits: phase_tables.angular_limits.view(),
                    signed_angular_offset: phase_tables.signed_angular_offset,
                    xnlm: legendre_normalization,
                    branch,
                    momentum_zero_epsilon: GENFMT_EPSILON,
                },
                path_index: path.index,
                print_level: self.print_level,
                curved_wave_criterion_percent: self.curved_wave_criterion_percent,
                edge_start_index: self.edge_start_index,
                active_energy_count: self.active_energy_count,
                degeneracy: path.degeneracy,
                current_normalization: -1.0,
                positions: path.positions_bohr.view(),
                phase_epsilon: GENFMT_EPSILON,
            });
        }

        genfmt_jas_driver_output(GenfmtJasDriverOutputInput {
            driver_setup: &self.setup,
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
            nstar,
        })
        .context("failed to generate GENFMTJAS driver output")
    }
}

#[derive(Debug)]
struct GenfmtOrdinaryGenerationData {
    setup: GenfmtDriverSetup,
    transition_b_matrix: TransitionBMatrix,
    transition_angular_momenta: Array1<i32>,
    spin_radial_factors: Array3<Complex64>,
    transition_matrices: Vec<GenfmtOrdinaryTransitionMatrices>,
    edge_start_index: usize,
    active_energy_count: usize,
    print_level: i32,
    curved_wave_criterion_percent: f64,
}

impl GenfmtOrdinaryGenerationData {
    #[cfg(test)]
    fn validate_path_energy_inputs(
        &self,
        paths: &[PathsDatGenfmtPath],
        path_setups: &[GenfmtPathSetup],
        phase_tables: &PhaseBinGenfmtData,
        legendre_normalization: ndarray::ArrayView2<'_, f64>,
    ) -> Result<usize> {
        if self.edge_start_index >= self.active_energy_count {
            bail!(
                "ordinary GENFMT edge index {} is outside active energy count {}",
                self.edge_start_index,
                self.active_energy_count
            );
        }
        if paths.len() != path_setups.len() || paths.len() != self.transition_matrices.len() {
            bail!(
                "ordinary GENFMT path input count mismatch: paths={}, setups={}, transition_matrices={}",
                paths.len(),
                path_setups.len(),
                self.transition_matrices.len()
            );
        }

        for (path_index, ((path, path_setup), transition_matrices)) in paths
            .iter()
            .zip(path_setups.iter())
            .zip(self.transition_matrices.iter())
            .enumerate()
        {
            refeff_core::genfmt_ordinary_path_energy_grid_from_driver_setup(
                GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
                    driver_setup: &self.setup,
                    path_setup,
                    path_potential_indices: path.potential_indices.view(),
                    angular_limits: phase_tables.angular_limits.view(),
                    spin_phase_shifts: phase_tables.spin_phase_shifts.view(),
                    signed_angular_offset: phase_tables.signed_angular_offset,
                    momentum_zero_epsilon: GENFMT_EPSILON,
                    xnlm: legendre_normalization,
                    transition_angular_momenta: self.transition_angular_momenta.view(),
                    spin_radial_factors: self.spin_radial_factors.view(),
                    transition_matrices: transition_matrices.matrices.view(),
                    transition_magnetic_offset: transition_magnetic_offset(transition_matrices)?,
                },
            )
            .with_context(|| {
                format!(
                    "failed to prepare ordinary GENFMT path energy input {}",
                    path_index + 1
                )
            })?;
        }

        Ok(paths.len())
    }

    fn driver_output(
        &self,
        paths: &[PathsDatGenfmtPath],
        path_setups: &[GenfmtPathSetup],
        phase_tables: &PhaseBinGenfmtData,
        legendre_normalization: ndarray::ArrayView2<'_, f64>,
        nstar: Option<GenfmtNStarDriverInput>,
    ) -> Result<GenfmtOrdinaryDriverOutput> {
        if paths.len() != path_setups.len() || paths.len() != self.transition_matrices.len() {
            bail!(
                "ordinary GENFMT path input count mismatch: paths={}, setups={}, transition_matrices={}",
                paths.len(),
                path_setups.len(),
                self.transition_matrices.len()
            );
        }

        let mut path_inputs = Vec::with_capacity(paths.len());
        for ((path, path_setup), transition_matrices) in paths
            .iter()
            .zip(path_setups.iter())
            .zip(self.transition_matrices.iter())
        {
            path_inputs.push(GenfmtOrdinaryPathEvaluationFromDriverSetupInput {
                energy_grid: GenfmtOrdinaryPathEnergyGridFromDriverSetupInput {
                    driver_setup: &self.setup,
                    path_setup,
                    path_potential_indices: path.potential_indices.view(),
                    angular_limits: phase_tables.angular_limits.view(),
                    spin_phase_shifts: phase_tables.spin_phase_shifts.view(),
                    signed_angular_offset: phase_tables.signed_angular_offset,
                    momentum_zero_epsilon: GENFMT_EPSILON,
                    xnlm: legendre_normalization,
                    transition_angular_momenta: self.transition_angular_momenta.view(),
                    spin_radial_factors: self.spin_radial_factors.view(),
                    transition_matrices: transition_matrices.matrices.view(),
                    transition_magnetic_offset: transition_magnetic_offset(transition_matrices)?,
                },
                path_index: path.index,
                print_level: self.print_level,
                curved_wave_criterion_percent: self.curved_wave_criterion_percent,
                edge_start_index: self.edge_start_index,
                active_energy_count: self.active_energy_count,
                degeneracy: path.degeneracy,
                current_normalization: -1.0,
                positions: path.positions_bohr.view(),
                phase_epsilon: GENFMT_EPSILON,
            });
        }

        genfmt_ordinary_driver_output(GenfmtOrdinaryDriverOutputInput {
            driver_setup: &self.setup,
            path_inputs: &path_inputs,
            initial_normalization: -1.0,
            nstar,
        })
        .context("failed to generate ordinary GENFMT driver output")
    }
}

fn prepare_generation_context(
    work_dir: &Path,
    input: &GenfmtInput,
) -> Result<GenfmtGenerationContext> {
    let global = read_required_global_input(work_dir)?;
    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let paths_path = work_dir.join("paths.dat");
    let paths_data = read_paths_dat(&paths_path)
        .with_context(|| format!("failed to read {}", paths_path.display()))?;
    let max_potential_index = phase
        .potential_count()
        .checked_sub(1)
        .context("phase.bin contains no potentials for GENFMT")?;
    let paths = paths_data
        .to_genfmt_paths(max_potential_index)
        .with_context(|| format!("failed to convert {} for GENFMT", paths_path.display()))?;
    let phase_tables = phase
        .to_genfmt_data()
        .context("failed to prepare GENFMT phase tables")?;
    let legendre_normalization = genfmt_core_legendre_normalization_from_feff_dims()
        .context("failed to prepare GENFMT Legendre normalization table")?;
    let nstar_rows = genfmt_nstar_rows_from_handoffs(input, &global, &phase, &paths)
        .context("failed to prepare GENFMT nstar rows")?;
    let nstar_driver = genfmt_nstar_driver_input_from_handoffs(input, &global, &phase)
        .context("failed to prepare GENFMT nstar driver input")?;
    let edge_start_index = genfmt_edge_start_index_from_phase(&phase)
        .context("failed to prepare GENFMT edge start index")?;
    let active_energy_count = phase.main_energy_count;

    let (driver, path_setups) = if global.control.do_nrixs == 1 {
        let setup = genfmt_jas_driver_setup_from_handoffs(GENFMT_VERSION, input, &global, &phase)
            .context("failed to prepare GENFMTJAS driver setup")?;
        let path_setups = genfmt_jas_path_setups_from_handoffs(input, &global, &phase, &paths)
            .context("failed to prepare GENFMTJAS path setup")?;
        let transition_setups =
            genfmt_jas_transition_setups_from_handoff_setups(&global, &phase, &path_setups)
                .context("failed to prepare GENFMTJAS transition setup")?;
        let transition_indices = genfmt_jas_transition_indices_from_handoffs(&global, &phase)
            .context("failed to prepare GENFMTJAS transition indices")?;
        let transition_angular_momenta = Array1::from_iter(
            transition_indices
                .transitions
                .iter()
                .map(|transition| transition.orbital_angular_momentum),
        );
        let q_weights = genfmt_jas_q_angles_from_handoffs(&global, &phase)
            .context("failed to prepare GENFMTJAS q weights")?
            .weights;
        let max_angular_momentum =
            nonnegative_usize(global.control.l2lp, "l2lp").context("failed to prepare l2lp")?;
        let decomposition_l_max = if global.control.ldecmx >= 0 {
            Some(
                nonnegative_usize(global.control.ldecmx, "ldecmx")
                    .context("failed to prepare ldecmx")?,
            )
        } else {
            None
        };
        let decomposition_channels = if decomposition_l_max.is_some() {
            decomposition_channel(input)?
        } else {
            0
        };
        (
            GenfmtGenerationDriver::Jas(Box::new(GenfmtJasGenerationData {
                setup,
                transition_angular_momenta,
                transition_setups,
                q_weights,
                max_angular_momentum,
                decomposition_l_max,
                decomposition_channels,
                edge_start_index,
                active_energy_count,
                print_level: input.control.ipr5,
                curved_wave_criterion_percent: input.control.critcw,
            })),
            path_setups,
        )
    } else {
        let setup = genfmt_driver_setup_from_handoffs(GENFMT_VERSION, input, &global, &phase)
            .context("failed to prepare ordinary GENFMT driver setup")?;
        let path_setups = genfmt_ordinary_path_setups_from_handoffs(input, &global, &phase, &paths)
            .context("failed to prepare ordinary GENFMT path setup")?;
        let transition_b_matrix =
            genfmt_ordinary_transition_b_matrix_from_handoffs(&global, &phase)
                .context("failed to prepare ordinary GENFMT transition B matrix")?;
        let transition_angular_momenta =
            Array1::from_vec(transition_b_matrix.orbital_momenta.to_vec());
        let spin_radial_factors = genfmt_ordinary_spin_radial_factors_from_phase(&phase)
            .context("failed to prepare ordinary GENFMT radial transition factors")?;
        let transition_matrices = genfmt_ordinary_transition_matrices_from_handoff_setups(
            &global,
            &phase,
            &path_setups,
            &transition_b_matrix,
        )
        .context("failed to prepare ordinary GENFMT transition matrices")?;
        (
            GenfmtGenerationDriver::Ordinary(Box::new(GenfmtOrdinaryGenerationData {
                setup,
                transition_b_matrix,
                transition_angular_momenta,
                spin_radial_factors,
                transition_matrices,
                edge_start_index,
                active_energy_count,
                print_level: input.control.ipr5,
                curved_wave_criterion_percent: input.control.critcw,
            })),
            path_setups,
        )
    };

    Ok(GenfmtGenerationContext {
        driver,
        titles: paths_data.titles,
        paths,
        path_setups,
        phase_tables,
        legendre_normalization,
        nstar_rows,
        nstar_driver,
        edge_start_index,
        active_energy_count,
    })
}

const GENFMT_VERSION: &str = "refeff-rust";
const GENFMT_EPSILON: f64 = 1.0e-16;

fn transition_magnetic_offset(matrices: &GenfmtOrdinaryTransitionMatrices) -> Result<usize> {
    let magnetic_axis_len = matrices
        .matrices
        .shape()
        .get(1)
        .copied()
        .context("ordinary GENFMT transition matrix is missing magnetic axis")?;
    if magnetic_axis_len == 0 || magnetic_axis_len % 2 == 0 {
        bail!("ordinary GENFMT transition matrix magnetic axis must be odd and nonzero");
    }
    Ok((magnetic_axis_len - 1) / 2)
}

fn write_feff_cache(path: &Path, data: &FeffBinData) -> Result<()> {
    write_feff_bin(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_list_cache(path: &Path, data: &ListDatData) -> Result<()> {
    write_list_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_optional_module_log(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_or_generate_module_log(work_dir: &Path) -> Result<usize> {
    let path = work_dir.join("log5.dat");
    if path.is_file() {
        return write_optional_module_log(&path);
    }

    let global = read_optional_global_input(work_dir)?;
    let data = generated_genfmt_module_log(global.as_ref());
    write_module_log(&path, &data)?;
    Ok(1)
}

fn read_optional_global_input(work_dir: &Path) -> Result<Option<GlobalInput>> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(None);
    }
    read_global_input_path(&path).map(Some)
}

fn read_required_global_input(work_dir: &Path) -> Result<GlobalInput> {
    read_global_input_path(&work_dir.join("global.inp"))
}

fn read_global_input_path(path: &Path) -> Result<GlobalInput> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    GlobalInput::parse_str(path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn generated_genfmt_module_log(global: Option<&GlobalInput>) -> ModuleLogData {
    let mut lines = Vec::new();
    if global.is_some_and(|global| global.control.do_nrixs == 1 && global.control.elpty < 0.0) {
        lines.push("Spherically averaged NRIXS in module genft - setting jinit=jmax.".to_string());
    }
    lines.push("Calculating EXAFS parameters ...".to_string());
    lines.push("Done with module: EXAFS parameters (GENFMT).".to_string());

    let line_terminators = vec!["\n".to_string(); lines.len()];
    ModuleLogData {
        lines,
        line_terminators,
    }
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

#[derive(Debug, Default)]
struct GenfmtCachePaths {
    feff_bins: Vec<PathBuf>,
    list_dats: Vec<PathBuf>,
    nstar_dat: Option<PathBuf>,
    feffl_bin: Option<PathBuf>,
}

impl GenfmtCachePaths {
    fn has_required_base_outputs(&self) -> bool {
        self.feff_bins
            .iter()
            .any(|path| file_name_is(path, "feff.bin"))
            && self
                .list_dats
                .iter()
                .any(|path| file_name_is(path, "list.dat"))
    }
}

fn cached_output_paths(work_dir: &Path) -> Result<GenfmtCachePaths> {
    let mut paths = GenfmtCachePaths::default();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", work_dir.display()))?;
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
        if is_feff_bin_cache_name(name) {
            paths.feff_bins.push(entry.path());
        } else if is_list_dat_cache_name(name) {
            paths.list_dats.push(entry.path());
        } else if name == "nstar.dat" {
            paths.nstar_dat = Some(entry.path());
        } else if name == "feffl.bin" {
            paths.feffl_bin = Some(entry.path());
        }
    }

    paths.feff_bins.sort();
    paths.list_dats.sort();
    Ok(paths)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GenfmtOutputFilenames {
    feff_bin: String,
    list_dat: String,
}

fn genfmt_output_filenames(polarization_index: usize) -> Result<GenfmtOutputFilenames> {
    match polarization_index {
        1 => Ok(GenfmtOutputFilenames {
            feff_bin: "feff.bin".to_string(),
            list_dat: "list.dat".to_string(),
        }),
        2..=9 => Ok(GenfmtOutputFilenames {
            feff_bin: format!("feff0{polarization_index}.bin"),
            list_dat: format!("list0{polarization_index}.dat"),
        }),
        10 => Ok(GenfmtOutputFilenames {
            feff_bin: "feff10.bin".to_string(),
            list_dat: "list10.dat".to_string(),
        }),
        _ => bail!(
            "invalid GENFMT polarization index {polarization_index}; FEFF supports 1 through 10"
        ),
    }
}

fn is_feff_bin_cache_name(name: &str) -> bool {
    (1..=10).any(|polarization_index| {
        genfmt_output_filenames(polarization_index)
            .is_ok_and(|filenames| filenames.feff_bin == name)
    })
}

fn is_list_dat_cache_name(name: &str) -> bool {
    (1..=10).any(|polarization_index| {
        genfmt_output_filenames(polarization_index)
            .is_ok_and(|filenames| filenames.list_dat == name)
    })
}

fn file_name_is(path: &Path, expected: &str) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some(expected)
}

#[cfg(test)]
mod tests;
