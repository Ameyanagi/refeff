use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array4, Axis, Slice};
use num_complex::Complex64;
use refeff_core::{
    ComptonRhorrpDensityInput, FovrgDiracSolverInput, LdosRholChannelInput, LdosRholDensityInput,
    LdosRholTableDriver, LdosRholTableDriverInput, RhorrpAtomicDensityInput,
    RhorrpWavefunctionSetupInput, ldos_rhol_channel, ldos_rhol_density, ldos_rhol_table_driver,
    rhorrp_atomic_density, rhorrp_photoelectron_kappa, rhorrp_prepare_wavefunction_grids,
    rhorrp_wavefunction_setup, screen_radial_grid,
};
use refeff_io::{
    ConfigDatData, DensityGrid, DensityInput, FmsInput, GeomDat, GlobalInput, HubbardVnlmBinData,
    ModuleLogData, PotBinData, PotInput, RHORRP_WAVEFUNCTION_RADIAL_COUNT,
    RhorrpDensityGridTablesHandoffInput, RhorrpDensityOutputData, RhorrpFmsInputHandoff,
    RhorrpGeomHandoff, RhorrpPhaseBinHandoff, RhorrpWavefunctionTablesHandoff,
    RhorrpWavefunctionTablesHandoffInput, read_config_dat, read_module_log_dat, read_phase_bin,
    read_pot_bin, read_rhorrp_density_bin, read_rhorrp_density_text, read_rhorrp_gg_diag_bin,
    read_rhorrp_gg_slice_bin, rhorrp_controls_from_pot_input, rhorrp_density_bin_bytes,
    rhorrp_density_filename_is_binary, rhorrp_density_grid_tables_input_from_handoffs,
    rhorrp_density_output_from_grid_tables, rhorrp_density_output_from_grid_with_nearest,
    rhorrp_density_text_string, rhorrp_geom_handoff_from_geom_dat, rhorrp_gg_diag_matrices,
    rhorrp_gg_slice_central_matrices, rhorrp_handoff_from_fms_input,
    rhorrp_orbital_tables_from_config_dat, rhorrp_phase_handoff_from_phase_bin,
    rhorrp_wavefunction_handoff_from_pot_bin, rhorrp_wavefunction_tables_from_handoffs,
    write_module_log_dat, write_rhorrp_density_bin, write_rhorrp_density_text,
};

use crate::work_dir_for_input;

const RHORRP_ATOMIC_RADIAL_DX: f64 = 0.05;
const RHORRP_ATOMIC_RADIAL_X0: f64 = 8.8;

/// Run the supported FEFF RHORRP cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    run_in_dir(work_dir_for_input(input))
}

/// Whether a FEFF RHORRP run can be satisfied from caches or handoff files.
pub(crate) fn has_supported_rhorrp_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("density.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.grids.is_empty() {
        return Ok(false);
    }

    let outputs = cached_output_paths(work_dir, &input)?;
    if validate_declared_density_sources(work_dir, &input).is_err() {
        return Ok(false);
    }
    Ok(density_outputs_are_available(work_dir, &input, &outputs))
}

/// Run the FEFF RHORRP density-output path from existing caches or handoff files.
///
/// This preserves cached FEFF `DENSITY` runs by validating and re-rendering
/// typed ASCII and binary RHORRP density-grid output files named by
/// `density.inp`. Missing `core` density grids are generated from the
/// Fortran-compatible atomic-density branch when `pot.bin` and `geom.dat` are
/// available. Missing non-core grids are generated from the ported RHORRP
/// same-point density path when FEFF's RHORRP handoff files are available.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    if !work_dir.join("density.inp").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if input.grids.is_empty() {
        return Ok(0);
    }

    let outputs = cached_output_paths(work_dir, &input)?;
    let states = cached_output_states(&outputs);
    validate_declared_density_sources(work_dir, &input)?;
    let needs_table_generation = input
        .grids
        .iter()
        .zip(outputs.iter())
        .zip(states.iter())
        .any(|((grid, _), state)| !grid.core && *state != CachedOutputState::Valid);
    if let Some((state, output)) = input
        .grids
        .iter()
        .zip(outputs.iter())
        .zip(states.iter())
        .find_map(|((grid, output), state)| {
            (!grid.core && *state != CachedOutputState::Valid).then_some((state, output))
        })
        .filter(|_| !has_table_density_source(work_dir))
    {
        if *state == CachedOutputState::Invalid {
            write_cached_output(output)?;
        } else {
            bail!(
                "RHORRP density generation requires cached output or RHORRP table density source files; missing {}",
                output.path.display()
            );
        }
    }

    let needs_core_generation = input
        .grids
        .iter()
        .zip(outputs.iter())
        .zip(states.iter())
        .any(|((grid, _), state)| grid.core && *state != CachedOutputState::Valid);
    if needs_core_generation && !has_core_density_source(work_dir) {
        let (state, output) = input
            .grids
            .iter()
            .zip(outputs.iter())
            .zip(states.iter())
            .find_map(|((grid, output), state)| {
                (grid.core && *state != CachedOutputState::Valid).then_some((state, output))
            })
            .context("RHORRP core density generation requested without a missing core output")?;
        if *state == CachedOutputState::Invalid {
            write_cached_output(output)?;
        } else {
            bail!(
                "RHORRP core density generation requires cached output or pot.bin/geom.dat source files; missing {}",
                output.path.display()
            );
        }
    }
    let has_valid_core_outputs = input
        .grids
        .iter()
        .zip(states.iter())
        .any(|(grid, state)| grid.core && *state == CachedOutputState::Valid);
    let has_valid_table_outputs = input
        .grids
        .iter()
        .zip(states.iter())
        .any(|(grid, state)| !grid.core && *state == CachedOutputState::Valid);
    let core_source = if needs_core_generation {
        Some(read_core_density_source(work_dir)?)
    } else if has_valid_core_outputs {
        optional_core_density_source_for_cache_comparison(work_dir)?
    } else {
        None
    };
    let table_source = if needs_table_generation {
        Some(read_table_density_source(work_dir)?)
    } else if has_valid_table_outputs {
        optional_table_density_source_for_cache_comparison(work_dir)?
    } else {
        None
    };

    for ((grid, output), state) in input.grids.iter().zip(outputs.iter()).zip(states.iter()) {
        if *state == CachedOutputState::Valid {
            if grid.core {
                if let Some(source) = core_source.as_ref()
                    && write_generated_core_density_output_if_stale(grid, output, source)?
                {
                    continue;
                }
            } else if let Some(source) = table_source.as_ref()
                && write_generated_table_density_output_if_stale(grid, output, source)?
            {
                continue;
            }
            write_cached_output(output)?;
        } else if grid.core {
            let source = core_source
                .as_ref()
                .context("missing RHORRP core density source after generation check")?;
            write_generated_core_density_output(grid, output, source)?;
        } else {
            let source = table_source
                .as_ref()
                .context("missing RHORRP table density source after generation check")?;
            write_generated_table_density_output(grid, output, source)?;
        }
    }

    write_or_generate_module_log(&work_dir.join("logrhorrp.dat"), &input)?;
    Ok(outputs.len())
}

fn read_input(work_dir: &Path) -> Result<DensityInput> {
    let input_path = work_dir.join("density.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    DensityInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputKind {
    Text,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedOutputState {
    Missing,
    Invalid,
    Valid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedOutputPath {
    path: PathBuf,
    kind: CachedOutputKind,
}

fn cached_output_paths(work_dir: &Path, input: &DensityInput) -> Result<Vec<CachedOutputPath>> {
    input
        .grids
        .iter()
        .map(|grid| {
            let path = output_path(work_dir, &grid.filename)?;
            let kind = if rhorrp_density_filename_is_binary(&grid.filename) {
                CachedOutputKind::Binary
            } else {
                CachedOutputKind::Text
            };
            Ok(CachedOutputPath { path, kind })
        })
        .collect()
}

fn write_cached_output(output: &CachedOutputPath) -> Result<()> {
    match output.kind {
        CachedOutputKind::Text => {
            let data = read_rhorrp_density_text(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            write_rhorrp_density_text(&output.path, &data)
                .with_context(|| format!("failed to write {}", output.path.display()))
        }
        CachedOutputKind::Binary => {
            let data = read_rhorrp_density_bin(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            write_rhorrp_density_bin(&output.path, &data)
                .with_context(|| format!("failed to write {}", output.path.display()))
        }
    }
}

fn cached_output_states(outputs: &[CachedOutputPath]) -> Vec<CachedOutputState> {
    outputs.iter().map(cached_output_state).collect()
}

fn cached_output_state(output: &CachedOutputPath) -> CachedOutputState {
    if !output.path.is_file() {
        return CachedOutputState::Missing;
    }
    if cached_output_is_valid(output) {
        CachedOutputState::Valid
    } else {
        CachedOutputState::Invalid
    }
}

fn output_path(work_dir: &Path, filename: &str) -> Result<PathBuf> {
    let path = Path::new(filename);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("RHORRP density output filename must stay within the work directory: {filename}");
    }
    Ok(work_dir.join(path))
}

fn density_outputs_are_available(
    work_dir: &Path,
    input: &DensityInput,
    outputs: &[CachedOutputPath],
) -> bool {
    let core_source_available = input
        .grids
        .iter()
        .zip(outputs.iter())
        .any(|(grid, output)| grid.core && cached_output_state(output) != CachedOutputState::Valid)
        && can_use_core_density_source(work_dir);
    let table_source_available = input
        .grids
        .iter()
        .zip(outputs.iter())
        .any(|(grid, output)| {
            !grid.core && cached_output_state(output) != CachedOutputState::Valid
        })
        && can_use_table_density_source(work_dir);

    input
        .grids
        .iter()
        .zip(outputs.iter())
        .all(|(grid, output)| match cached_output_state(output) {
            CachedOutputState::Valid => true,
            CachedOutputState::Missing | CachedOutputState::Invalid if grid.core => {
                core_source_available
            }
            CachedOutputState::Missing | CachedOutputState::Invalid => table_source_available,
        })
}

fn has_core_density_source(work_dir: &Path) -> bool {
    work_dir.join("pot.bin").is_file() && work_dir.join("geom.dat").is_file()
}

fn can_use_core_density_source(work_dir: &Path) -> bool {
    has_core_density_source(work_dir) && read_core_density_source(work_dir).is_ok()
}

fn has_table_density_source(work_dir: &Path) -> bool {
    [
        "pot.bin",
        "config.dat",
        "phase.bin",
        "pot.inp",
        "fms.inp",
        "geom.dat",
        "gg_diag.bin",
    ]
    .iter()
    .all(|name| work_dir.join(name).is_file())
}

fn can_use_table_density_source(work_dir: &Path) -> bool {
    has_table_density_source(work_dir) && read_table_density_source(work_dir).is_ok()
}

fn validate_declared_density_sources(work_dir: &Path, input: &DensityInput) -> Result<()> {
    if input.grids.iter().any(|grid| grid.core) && has_core_density_source(work_dir) {
        read_core_density_source(work_dir)?;
    }
    if input.grids.iter().any(|grid| !grid.core) && has_table_density_source(work_dir) {
        read_table_density_source(work_dir)?;
    }
    Ok(())
}

fn optional_core_density_source_for_cache_comparison(
    work_dir: &Path,
) -> Result<Option<CoreDensitySource>> {
    if has_core_density_source(work_dir) {
        Ok(Some(read_core_density_source(work_dir)?))
    } else {
        Ok(None)
    }
}

fn optional_table_density_source_for_cache_comparison(
    work_dir: &Path,
) -> Result<Option<TableDensitySource>> {
    if has_table_density_source(work_dir) {
        Ok(Some(read_table_density_source(work_dir)?))
    } else {
        Ok(None)
    }
}

fn cached_output_is_valid(output: &CachedOutputPath) -> bool {
    if !output.path.is_file() {
        return false;
    }
    match output.kind {
        CachedOutputKind::Text => read_rhorrp_density_text(&output.path).is_ok(),
        CachedOutputKind::Binary => read_rhorrp_density_bin(&output.path).is_ok(),
    }
}

pub(crate) fn has_rhorrp_density_callback_source(work_dir: &Path) -> bool {
    has_declared_rhorrp_density_callback_source(work_dir)
        && read_rhorrp_density_callback_source(work_dir).is_ok()
}

pub(crate) fn validate_declared_rhorrp_density_callback_source(work_dir: &Path) -> Result<()> {
    if has_declared_rhorrp_density_callback_source(work_dir) {
        read_rhorrp_density_callback_source(work_dir)?;
    }
    Ok(())
}

fn has_declared_rhorrp_density_callback_source(work_dir: &Path) -> bool {
    has_table_density_source(work_dir) && work_dir.join("gg_slice.bin").is_file()
}

#[derive(Debug)]
struct CoreDensitySource {
    atom_positions_bohr: Array2<f64>,
    atom_potentials: Vec<usize>,
    radial_grid: Vec<f64>,
    pot: PotBinData,
}

#[derive(Debug)]
pub(crate) struct TableDensitySource {
    geometry: RhorrpGeomHandoff,
    fms: RhorrpFmsInputHandoff,
    phase: RhorrpPhaseBinHandoff,
    wavefunctions: RhorrpWavefunctionTablesHandoff,
    diagonal_scattering_matrices: Array4<Complex64>,
    central_scattering_matrices: Option<Array4<Complex64>>,
    temperature_hartree: f64,
    central_norman_radius_bohr: f64,
}

#[derive(Debug)]
pub(crate) struct LdosWavefunctionSource {
    pub(crate) phase: RhorrpPhaseBinHandoff,
    pub(crate) wavefunctions: RhorrpWavefunctionTablesHandoff,
    pub(crate) norman_radii_bohr: Vec<f64>,
}

pub(crate) struct LdosRholSourceTable {
    pub(crate) potential_index: usize,
    pub(crate) chemical_potential_hartree: f64,
    pub(crate) table: LdosRholTableDriver,
}

pub(crate) struct HubbardLdosRholSourceTable {
    pub(crate) potential_index: usize,
    pub(crate) embedded_magnetic_ldos: Array4<f64>,
    pub(crate) scattering_magnetic_ldos: Array4<Complex64>,
}

impl TableDensitySource {
    pub(crate) fn central_norman_radius_bohr(&self) -> f64 {
        self.central_norman_radius_bohr
    }

    pub(crate) fn compton_density_input(
        &self,
        chemical_potential_override_hartree: Option<f64>,
    ) -> Result<ComptonRhorrpDensityInput<'_>> {
        let Some(central_scattering_matrices) = self.central_scattering_matrices.as_ref() else {
            bail!("COMPTON RHORRP density callback requires gg_slice.bin");
        };
        Ok(ComptonRhorrpDensityInput {
            atom_positions: self.geometry.atom_positions_bohr.view(),
            atom_potentials: &self.geometry.atom_potentials,
            fms_atom_count: Some(self.fms.central_fms_atom_count(&self.geometry)?),
            energies_hartree: self.phase.energies_hartree.view(),
            reference_energy_hartree: self.wavefunctions.reference_energy_hartree,
            regular_large: self.wavefunctions.wavefunctions.regular_large.view(),
            irregular_large: self.wavefunctions.wavefunctions.irregular_large.view(),
            regular_small: self.wavefunctions.wavefunctions.regular_small.view(),
            irregular_small: self.wavefunctions.wavefunctions.irregular_small.view(),
            phase: self.wavefunctions.wavefunctions.phase_shifts.view(),
            diagonal_scattering_matrices: Some(self.diagonal_scattering_matrices.view()),
            central_scattering_matrices: Some(central_scattering_matrices.view()),
            radial_x0: self.wavefunctions.radial_x0,
            radial_dx: self.wavefunctions.radial_dx,
            radial_count: self.wavefunctions.wavefunctions.radial_count(),
            real_axis_count: self.phase.real_axis_count,
            chemical_potential_hartree: self.phase.chemical_potential_hartree,
            temperature_hartree: self.temperature_hartree,
            chemical_potential_override_hartree,
        })
    }
}

pub(crate) fn read_ldos_wavefunction_source_on_energy_grid(
    work_dir: &Path,
    energies_hartree: Array1<Complex64>,
    angular_count: usize,
) -> Result<LdosWavefunctionSource> {
    read_ldos_wavefunction_source_with_energy_grid(
        work_dir,
        Some(energies_hartree),
        None,
        Some(angular_count),
    )
}

pub(crate) fn read_ldos_wavefunction_source_on_energy_grid_for_spin(
    work_dir: &Path,
    energies_hartree: Array1<Complex64>,
    spin_selector: i32,
) -> Result<LdosWavefunctionSource> {
    read_ldos_wavefunction_source_with_energy_grid(
        work_dir,
        Some(energies_hartree),
        Some(spin_selector),
        None,
    )
}

fn read_ldos_wavefunction_source_with_energy_grid(
    work_dir: &Path,
    energies_hartree: Option<Array1<Complex64>>,
    spin_selector_override: Option<i32>,
    angular_count_override: Option<usize>,
) -> Result<LdosWavefunctionSource> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let norman_radii_bohr = pot.norman_radii.to_vec();
    let config = read_config(work_dir)?;
    let controls = read_pot_controls(work_dir)?;
    let mut phase = match spin_selector_override {
        Some(spin_selector) => {
            let phase_path = work_dir.join("phase.bin");
            let phase = read_phase_bin(&phase_path)
                .with_context(|| format!("failed to read {}", phase_path.display()))?;
            rhorrp_phase_handoff_from_phase_bin(&phase, spin_selector).with_context(|| {
                format!(
                    "failed to build spin {spin_selector} RHORRP phase handoff from {}",
                    phase_path.display()
                )
            })?
        }
        None => read_phase_handoff(work_dir)?,
    };
    if let Some(energies_hartree) = energies_hartree {
        phase.energies_hartree = energies_hartree;
        phase.real_axis_count = phase.energies_hartree.len();
    }
    let mut fms = read_fms_handoff(work_dir, pot.potential_count())?;
    if let Some(angular_count) = angular_count_override {
        fms.max_angular_momentum = angular_count
            .checked_sub(1)
            .context("LDOS angular channel count must be positive")?;
        fms.angular_momentum_count = angular_count;
    } else {
        // Spin/Hubbard LDOS still uses the full l=0..3 work table.
        fms.max_angular_momentum = fms.max_angular_momentum.max(3);
        fms.angular_momentum_count = fms.angular_momentum_count.max(4);
    }
    let wavefunctions =
        rhorrp_wavefunction_tables_from_handoffs(RhorrpWavefunctionTablesHandoffInput {
            pot: &pot,
            config: &config,
            phase: &phase,
            fms,
            controls,
            radial_count: RHORRP_WAVEFUNCTION_RADIAL_COUNT,
        })
        .context("failed to build LDOS wavefunction tables from handoff files")?;

    Ok(LdosWavefunctionSource {
        phase,
        wavefunctions,
        norman_radii_bohr,
    })
}

pub(crate) fn read_no_fms_ldos_rhol_source_tables(
    work_dir: &Path,
    relative_energy_grid_hartree: Array1<Complex64>,
    potential_indices: &[usize],
    angular_count: usize,
) -> Result<Vec<LdosRholSourceTable>> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let pot_handoff = rhorrp_wavefunction_handoff_from_pot_bin(&pot)
        .context("failed to build RHORRP wavefunction handoff from pot.bin")?;
    let config = read_config(work_dir)?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)
        .context("failed to compact config.dat orbitals for LDOS rhol source")?;
    let controls = read_pot_controls(work_dir)?;
    let phase = read_phase_handoff(work_dir)?;
    let prepared = rhorrp_prepare_wavefunction_grids(pot_handoff.grid_preparation_input(
        controls.target_radial_dx,
        controls.exchange_index,
        RHORRP_WAVEFUNCTION_RADIAL_COUNT,
    ))
    .context("failed to prepare LDOS rhol source wavefunction grids")?;

    let solver_energy_grid = relative_energy_grid_hartree;
    let source_potential_count = pot_handoff
        .potential_count()
        .min(config.potential_count())
        .min(phase.potential_count())
        .min(prepared.potential_count())
        .min(orbital_tables.bound_orbital_counts.len());
    let zero = Complex64::new(0.0, 0.0);
    let mut tables = Vec::new();

    for &potential_index in potential_indices {
        if potential_index >= source_potential_count {
            continue;
        }

        let radial_match_index = prepared.reference_indices_1based[potential_index]
            .checked_sub(2)
            .context("LDOS rhol source reference index precedes FOVRG match row")?;
        let muffin_tin_radius = pot_handoff.muffin_tin_radii()[potential_index];
        let norman_radius = pot_handoff.norman_radii()[potential_index];
        let atomic_number = pot_handoff.atomic_numbers()[potential_index];
        let bound_large_coefficients = pot.large_coefficients.index_axis(Axis(2), potential_index);
        let bound_small_coefficients = pot.small_coefficients.index_axis(Axis(2), potential_index);
        let electron_counts = orbital_tables
            .electron_counts_by_potential
            .index_axis(Axis(1), potential_index);
        let valence_counts = pot.orbital_occupancy.index_axis(Axis(1), potential_index);
        let kappa = orbital_tables
            .kappa_by_potential
            .index_axis(Axis(1), potential_index);
        let bound_orbital_count = orbital_tables.bound_orbital_counts[potential_index];
        let total_potential = prepared
            .total_potential
            .index_axis(Axis(1), potential_index);
        let valence_potential = prepared
            .valence_potential
            .index_axis(Axis(1), potential_index);
        let bound_large_components = prepared
            .bound_large_components
            .index_axis(Axis(2), potential_index);
        let bound_small_components = prepared
            .bound_small_components
            .index_axis(Axis(2), potential_index);

        let mut solvers = Vec::with_capacity(solver_energy_grid.len() * angular_count);
        let mut wave_numbers = Vec::with_capacity(solver_energy_grid.len());
        for energy_index in 0..solver_energy_grid.len() {
            let setup = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
                energy_hartree: solver_energy_grid[energy_index],
                reference_energy_hartree: prepared.reference_energies_hartree[potential_index],
                muffin_tin_radius,
                norman_radius,
                radial_x0: RHORRP_ATOMIC_RADIAL_X0,
                radial_dx: prepared.radial_dx,
                radial_capacity: prepared.radii.len(),
                exchange_index: controls.exchange_index,
            })
            .with_context(|| {
                format!(
                    "failed to build LDOS rhol source setup for potential {potential_index} energy {energy_index}"
                )
            })?;
            wave_numbers.push(setup.wave_number);

            for angular in 0..angular_count {
                solvers.push(FovrgDiracSolverInput {
                    exchange_cycle_count: setup.dirac_cycle_count,
                    target_kappa: rhorrp_photoelectron_kappa(angular)?,
                    muffin_tin_radius,
                    target_last_index: setup.last_integration_index_1based - 1,
                    energy: setup.kinetic_energy_hartree,
                    step: prepared.radial_dx,
                    radii: prepared.radii.view(),
                    exchange_correlation_potential: total_potential,
                    valence_exchange_correlation_potential: valence_potential,
                    bound_large_components,
                    bound_small_components,
                    bound_large_coefficients,
                    bound_small_coefficients,
                    electron_counts,
                    valence_counts,
                    kappa,
                    muffin_tin_large_component: zero,
                    muffin_tin_small_component: zero,
                    atomic_number,
                    irregular: false,
                    c3_scale: ldos_c3_scale_for_angular_momentum(angular),
                    radial_match_index,
                    bound_orbital_count,
                });
            }
        }
        let wave_numbers = Array1::from_vec(wave_numbers);
        let scattering_trace = Array2::zeros((angular_count, solver_energy_grid.len()));
        let table = ldos_rhol_table_driver(LdosRholTableDriverInput {
            solvers: &solvers,
            energy_grid_hartree: solver_energy_grid.view(),
            wave_numbers: wave_numbers.view(),
            scattering_trace: scattering_trace.view(),
            radial_step: prepared.radial_dx,
            norman_radius,
            angular_count,
            apply_scattering: false,
        })
        .with_context(|| {
            format!("failed to solve LDOS rhol source table for potential {potential_index}")
        })?;

        tables.push(LdosRholSourceTable {
            potential_index,
            chemical_potential_hartree: phase.chemical_potential_hartree,
            table,
        });
    }

    Ok(tables)
}

pub(crate) fn read_hubbard_ldos_rhol_source_tables(
    work_dir: &Path,
    relative_energy_grid_hartree: Array1<Complex64>,
    potential_indices: &[usize],
    angular_count: usize,
    hubbard: &HubbardVnlmBinData,
) -> Result<Vec<HubbardLdosRholSourceTable>> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let pot_handoff = rhorrp_wavefunction_handoff_from_pot_bin(&pot)
        .context("failed to build Hubbard LDOS wavefunction handoff from pot.bin")?;
    let config = read_config(work_dir)?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&config)
        .context("failed to compact config.dat orbitals for Hubbard LDOS")?;
    let controls = read_pot_controls(work_dir)?;
    let prepared = rhorrp_prepare_wavefunction_grids(pot_handoff.grid_preparation_input(
        controls.target_radial_dx,
        controls.exchange_index,
        RHORRP_WAVEFUNCTION_RADIAL_COUNT,
    ))
    .context("failed to prepare Hubbard LDOS radial grids")?;

    if hubbard.potential_count() < pot_handoff.potential_count() {
        bail!(
            "v_hubbard.bin has {} potential blocks, expected at least {}",
            hubbard.potential_count(),
            pot_handoff.potential_count()
        );
    }
    if hubbard.angular_limit + 1 < angular_count {
        bail!(
            "v_hubbard.bin has {} angular channels, expected at least {angular_count}",
            hubbard.angular_limit + 1
        );
    }

    let energy_count = relative_energy_grid_hartree.len();
    let magnetic_count = angular_count
        .checked_mul(angular_count)
        .context("Hubbard LDOS magnetic channel count overflow")?;
    let source_potential_count = pot_handoff
        .potential_count()
        .min(config.potential_count())
        .min(prepared.potential_count())
        .min(orbital_tables.bound_orbital_counts.len());
    let zero = Complex64::new(0.0, 0.0);
    let mut tables = Vec::new();

    for &potential_index in potential_indices {
        if potential_index >= source_potential_count {
            continue;
        }
        let radial_match_index = prepared.reference_indices_1based[potential_index]
            .checked_sub(2)
            .context("Hubbard LDOS reference index precedes FOVRG match row")?;
        let muffin_tin_radius = pot_handoff.muffin_tin_radii()[potential_index];
        let norman_radius = pot_handoff.norman_radii()[potential_index];
        let atomic_number = pot_handoff.atomic_numbers()[potential_index];
        let bound_large_coefficients = pot.large_coefficients.index_axis(Axis(2), potential_index);
        let bound_small_coefficients = pot.small_coefficients.index_axis(Axis(2), potential_index);
        let electron_counts = orbital_tables
            .electron_counts_by_potential
            .index_axis(Axis(1), potential_index);
        let valence_counts = pot.orbital_occupancy.index_axis(Axis(1), potential_index);
        let kappa = orbital_tables
            .kappa_by_potential
            .index_axis(Axis(1), potential_index);
        let bound_orbital_count = orbital_tables.bound_orbital_counts[potential_index];
        let base_total_potential = prepared
            .total_potential
            .index_axis(Axis(1), potential_index);
        let base_valence_potential = prepared
            .valence_potential
            .index_axis(Axis(1), potential_index);
        let bound_large_components = prepared
            .bound_large_components
            .index_axis(Axis(2), potential_index);
        let bound_small_components = prepared
            .bound_small_components
            .index_axis(Axis(2), potential_index);

        let mut embedded_magnetic_ldos =
            Array4::<f64>::zeros((angular_count, magnetic_count, 2, energy_count));
        let mut scattering_magnetic_ldos =
            Array4::<Complex64>::zeros((angular_count, magnetic_count, 2, energy_count));
        for spin in 0..2 {
            for energy_index in 0..energy_count {
                let setup = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
                    energy_hartree: relative_energy_grid_hartree[energy_index],
                    reference_energy_hartree: prepared.reference_energies_hartree[potential_index],
                    muffin_tin_radius,
                    norman_radius,
                    radial_x0: RHORRP_ATOMIC_RADIAL_X0,
                    radial_dx: prepared.radial_dx,
                    radial_capacity: prepared.radii.len(),
                    exchange_index: controls.exchange_index,
                })
                .with_context(|| {
                    format!(
                        "failed to build Hubbard LDOS setup for potential {potential_index}, spin {}, energy {}",
                        spin + 1,
                        energy_index + 1
                    )
                })?;
                for angular in 0..angular_count {
                    let magnetic_start = angular * angular;
                    let magnetic_end = (angular + 1) * (angular + 1);
                    for magnetic in magnetic_start..magnetic_end {
                        let shift = hubbard.values[(potential_index, spin, angular, magnetic)];
                        let shifted_total =
                            base_total_potential.mapv(|potential| potential + shift);
                        let shifted_valence =
                            base_valence_potential.mapv(|potential| potential + shift);
                        let solver = FovrgDiracSolverInput {
                            exchange_cycle_count: setup.dirac_cycle_count,
                            target_kappa: rhorrp_photoelectron_kappa(angular)?,
                            muffin_tin_radius,
                            target_last_index: setup.last_integration_index_1based - 1,
                            energy: setup.kinetic_energy_hartree,
                            step: prepared.radial_dx,
                            radii: prepared.radii.view(),
                            exchange_correlation_potential: shifted_total.view(),
                            valence_exchange_correlation_potential: shifted_valence.view(),
                            bound_large_components,
                            bound_small_components,
                            bound_large_coefficients,
                            bound_small_coefficients,
                            electron_counts,
                            valence_counts,
                            kappa,
                            muffin_tin_large_component: zero,
                            muffin_tin_small_component: zero,
                            atomic_number,
                            irregular: false,
                            c3_scale: ldos_c3_scale_for_angular_momentum(angular),
                            radial_match_index,
                            bound_orbital_count,
                        };
                        let channel = ldos_rhol_channel(LdosRholChannelInput {
                            solver,
                            angular_momentum: angular,
                            wave_number: setup.wave_number,
                        })
                        .with_context(|| {
                            format!(
                                "failed to solve Hubbard LDOS channel for potential {potential_index}, spin {}, energy {}, l={angular}, m-slot={magnetic}",
                                spin + 1,
                                energy_index + 1
                            )
                        })?;
                        let radial_count = channel.radial_components.row_count();
                        let density = ldos_rhol_density(LdosRholDensityInput {
                            radii: prepared
                                .radii
                                .slice_axis(Axis(0), Slice::from(..radial_count)),
                            regular_large: channel.radial_components.regular_large.view(),
                            regular_small: channel.radial_components.regular_small.view(),
                            irregular_large: channel.radial_components.irregular_large.view(),
                            irregular_small: channel.radial_components.irregular_small.view(),
                            radial_step: prepared.radial_dx,
                            norman_radius,
                            wave_number: setup.wave_number,
                            angular_momentum: angular,
                        })
                        .with_context(|| {
                            format!(
                                "failed to integrate Hubbard LDOS channel for potential {potential_index}, spin {}, energy {}, l={angular}, m-slot={magnetic}",
                                spin + 1,
                                energy_index + 1
                            )
                        })?;
                        embedded_magnetic_ldos[(angular, magnetic, spin, energy_index)] =
                            density.embedded_ldos;
                        scattering_magnetic_ldos[(angular, magnetic, spin, energy_index)] =
                            density.scattering_ldos;
                    }
                }
            }
        }
        tables.push(HubbardLdosRholSourceTable {
            potential_index,
            embedded_magnetic_ldos,
            scattering_magnetic_ldos,
        });
    }

    Ok(tables)
}

fn ldos_c3_scale_for_angular_momentum(angular_momentum: usize) -> i32 {
    if angular_momentum == 0 { 0 } else { 1 }
}

fn read_core_density_source(work_dir: &Path) -> Result<CoreDensitySource> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let geom = read_geom_dat(work_dir)?;
    let geom = rhorrp_geom_handoff_from_geom_dat(&geom)
        .context("failed to build RHORRP geometry handoff from geom.dat")?;

    for (row, &potential) in geom.atom_potentials.iter().enumerate() {
        if potential >= pot.potential_count() {
            bail!(
                "geom.dat atom {} potential {} exceeds pot.bin potential count {} for RHORRP core density generation",
                row + 1,
                potential,
                pot.potential_count()
            );
        }
    }

    let radial_grid = screen_radial_grid(
        RHORRP_ATOMIC_RADIAL_DX,
        RHORRP_ATOMIC_RADIAL_X0,
        pot.large_components.shape()[0],
    )
    .context("failed to build RHORRP atomic-density radial grid")?
    .to_vec();

    Ok(CoreDensitySource {
        atom_positions_bohr: geom.atom_positions_bohr,
        atom_potentials: geom.atom_potentials,
        radial_grid,
        pot,
    })
}

fn read_geom_dat(work_dir: &Path) -> Result<GeomDat> {
    let geom_path = work_dir.join("geom.dat");
    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))
}

fn read_table_density_source(work_dir: &Path) -> Result<TableDensitySource> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let central_norman_radius_bohr = *pot
        .norman_radii
        .get(0)
        .context("pot.bin has no central Norman radius for RHORRP density source")?;
    let config = read_config(work_dir)?;
    let controls = read_pot_controls(work_dir)?;
    let phase = read_phase_handoff(work_dir)?;
    let fms = read_fms_handoff(work_dir, pot.potential_count())?;
    let geometry = rhorrp_geom_handoff_from_geom_dat(&read_geom_dat(work_dir)?)
        .context("failed to build RHORRP geometry handoff from geom.dat")?;
    let diagonal_scattering_matrices = read_diagonal_scattering_matrices(work_dir)?;
    let central_scattering_matrices =
        read_central_scattering_matrices(work_dir, fms.angular_momentum_count)?;
    let wavefunctions =
        rhorrp_wavefunction_tables_from_handoffs(RhorrpWavefunctionTablesHandoffInput {
            pot: &pot,
            config: &config,
            phase: &phase,
            fms,
            controls,
            radial_count: RHORRP_WAVEFUNCTION_RADIAL_COUNT,
        })
        .context("failed to build RHORRP wavefunction tables from handoff files")?;

    Ok(TableDensitySource {
        geometry,
        fms,
        phase,
        wavefunctions,
        diagonal_scattering_matrices,
        central_scattering_matrices,
        temperature_hartree: controls.temperature_hartree,
        central_norman_radius_bohr,
    })
}

pub(crate) fn read_rhorrp_density_callback_source(work_dir: &Path) -> Result<TableDensitySource> {
    let source = read_table_density_source(work_dir)?;
    if source.central_scattering_matrices.is_none() {
        bail!("COMPTON RHORRP density callback requires gg_slice.bin");
    }
    Ok(source)
}

fn read_config(work_dir: &Path) -> Result<ConfigDatData> {
    let path = work_dir.join("config.dat");
    read_config_dat(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn read_pot_controls(work_dir: &Path) -> Result<refeff_io::RhorrpPotInputControls> {
    let path = work_dir.join("pot.inp");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = PotInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    rhorrp_controls_from_pot_input(&input)
        .with_context(|| format!("failed to build RHORRP controls from {}", path.display()))
}

fn read_phase_handoff(work_dir: &Path) -> Result<RhorrpPhaseBinHandoff> {
    let path = work_dir.join("phase.bin");
    let phase =
        read_phase_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let spin_selector = read_spin_selector(work_dir)?;
    rhorrp_phase_handoff_from_phase_bin(&phase, spin_selector).with_context(|| {
        format!(
            "failed to build RHORRP phase handoff from {}",
            path.display()
        )
    })
}

fn read_fms_handoff(work_dir: &Path, potential_count: usize) -> Result<RhorrpFmsInputHandoff> {
    let path = work_dir.join("fms.inp");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = FmsInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    rhorrp_handoff_from_fms_input(&input, potential_count)
        .with_context(|| format!("failed to build RHORRP FMS handoff from {}", path.display()))
}

fn read_spin_selector(work_dir: &Path) -> Result<i32> {
    let path = work_dir.join("global.inp");
    if !path.is_file() {
        return Ok(0);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let input = GlobalInput::parse_str(&path, &text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(input.control.ispin)
}

fn read_diagonal_scattering_matrices(work_dir: &Path) -> Result<Array4<Complex64>> {
    let path = work_dir.join("gg_diag.bin");
    let data = read_rhorrp_gg_diag_bin(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    rhorrp_gg_diag_matrices(&data).with_context(|| {
        format!(
            "failed to promote RHORRP gg_diag matrices from {}",
            path.display()
        )
    })
}

fn read_central_scattering_matrices(
    work_dir: &Path,
    angular_momentum_count: usize,
) -> Result<Option<Array4<Complex64>>> {
    let path = work_dir.join("gg_slice.bin");
    if !path.is_file() {
        return Ok(None);
    }
    let data = read_rhorrp_gg_slice_bin(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let block_dimension = angular_momentum_count
        .checked_mul(angular_momentum_count)
        .context("RHORRP gg_slice block dimension overflowed")?;
    rhorrp_gg_slice_central_matrices(&data, block_dimension)
        .map(Some)
        .with_context(|| {
            format!(
                "failed to promote RHORRP gg_slice matrices from {}",
                path.display()
            )
        })
}

fn write_generated_core_density_output(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &CoreDensitySource,
) -> Result<()> {
    let data = generated_core_density_output_data(grid, output, source)?;
    write_density_output(output, data)
}

fn write_generated_core_density_output_if_stale(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &CoreDensitySource,
) -> Result<bool> {
    let generated = generated_core_density_output_data(grid, output, source)?;
    if cached_output_matches_generated(output, &generated)? {
        return Ok(false);
    }
    write_density_output(output, generated)?;
    Ok(true)
}

fn generated_core_density_output_data(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &CoreDensitySource,
) -> Result<RhorrpDensityOutputData> {
    let grid = grid.to_bohr_grid()?;
    rhorrp_density_output_from_grid_with_nearest(
        refeff_io::RhorrpDensityGridNearestOutputInput {
            grid: &grid,
            atom_positions_bohr: source.atom_positions_bohr.view(),
            atom_potentials: &source.atom_potentials,
            fms_atom_count: None,
        },
        |point| {
            rhorrp_atomic_density(RhorrpAtomicDensityInput {
                point,
                orbital_index_1based: 1,
                atom_positions: source.atom_positions_bohr.view(),
                atom_potentials: &source.atom_potentials,
                radii: &source.radial_grid,
                large_components: source.pot.large_components.view(),
                small_components: source.pot.small_components.view(),
            })
        },
    )
    .with_context(|| format!("failed to generate {}", output.path.display()))
}

fn write_generated_table_density_output(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &TableDensitySource,
) -> Result<()> {
    let data = generated_table_density_output_data(grid, output, source)?;
    write_density_output(output, data)
}

fn write_generated_table_density_output_if_stale(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &TableDensitySource,
) -> Result<bool> {
    let generated = generated_table_density_output_data(grid, output, source)?;
    if cached_output_matches_generated(output, &generated)? {
        return Ok(false);
    }
    write_density_output(output, generated)?;
    Ok(true)
}

fn generated_table_density_output_data(
    grid: &DensityGrid,
    output: &CachedOutputPath,
    source: &TableDensitySource,
) -> Result<RhorrpDensityOutputData> {
    let grid = grid.to_bohr_grid()?;
    let input =
        rhorrp_density_grid_tables_input_from_handoffs(RhorrpDensityGridTablesHandoffInput {
            grid: &grid,
            geometry: &source.geometry,
            fms: &source.fms,
            phase: &source.phase,
            reference_energy_hartree: source.wavefunctions.reference_energy_hartree,
            wavefunctions: &source.wavefunctions.wavefunctions,
            diagonal_scattering_matrices: Some(source.diagonal_scattering_matrices.view()),
            radial_x0: source.wavefunctions.radial_x0,
            radial_dx: source.wavefunctions.radial_dx,
            temperature_hartree: source.temperature_hartree,
            chemical_potential_override_hartree: None,
        })
        .with_context(|| {
            format!(
                "failed to build RHORRP density input for {}",
                output.path.display()
            )
        })?;
    rhorrp_density_output_from_grid_tables(input)
        .with_context(|| format!("failed to generate {}", output.path.display()))
}

fn write_density_output(output: &CachedOutputPath, data: RhorrpDensityOutputData) -> Result<()> {
    match data {
        RhorrpDensityOutputData::Text(data) => write_rhorrp_density_text(&output.path, &data),
        RhorrpDensityOutputData::Binary(data) => write_rhorrp_density_bin(&output.path, &data),
    }
    .with_context(|| format!("failed to write {}", output.path.display()))
}

fn cached_output_matches_generated(
    output: &CachedOutputPath,
    generated: &RhorrpDensityOutputData,
) -> Result<bool> {
    match (output.kind, generated) {
        (CachedOutputKind::Text, RhorrpDensityOutputData::Text(generated)) => {
            let cached = read_rhorrp_density_text(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            let cached = rhorrp_density_text_string(&cached)
                .with_context(|| format!("failed to render {}", output.path.display()))?;
            let generated = rhorrp_density_text_string(generated)
                .with_context(|| format!("failed to render generated {}", output.path.display()))?;
            Ok(cached == generated)
        }
        (CachedOutputKind::Binary, RhorrpDensityOutputData::Binary(generated)) => {
            let cached = read_rhorrp_density_bin(&output.path)
                .with_context(|| format!("failed to read {}", output.path.display()))?;
            let cached = rhorrp_density_bin_bytes(&cached)
                .with_context(|| format!("failed to render {}", output.path.display()))?;
            let generated = rhorrp_density_bin_bytes(generated)
                .with_context(|| format!("failed to render generated {}", output.path.display()))?;
            Ok(cached == generated)
        }
        _ => Ok(false),
    }
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(path: &Path, input: &DensityInput) -> Result<()> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    let data = generated_rhorrp_module_log(input)?;
    write_module_log_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))
}

fn generated_rhorrp_module_log(input: &DensityInput) -> Result<ModuleLogData> {
    let mut lines = vec!["Calculating charge density output ...".to_string()];
    for grid in &input.grids {
        lines.push(format!(
            "Calculate density: {} ({} total points)",
            grid.filename,
            fortran_i8(density_grid_point_count(grid)?)
        ));
        lines.push("Calculation complete, transfer back to master.".to_string());
        lines.push("Saving density calculation.".to_string());
    }
    lines.push("Done with module: charge density output (RHORRP).".to_string());

    let line_terminators = vec!["\n".to_string(); lines.len()];
    Ok(ModuleLogData {
        lines,
        line_terminators,
    })
}

fn fortran_i8(value: usize) -> String {
    if value > 99_999_999 {
        return "********".to_string();
    }
    format!("{value:>8}")
}

fn density_grid_point_count(grid: &DensityGrid) -> Result<usize> {
    grid.axes.iter().try_fold(1usize, |points, axis| {
        points
            .checked_mul(axis.points)
            .context("RHORRP density grid point count is too large")
    })
}

#[cfg(test)]
mod tests {
    use super::{has_rhorrp_density_callback_source, has_supported_rhorrp_output, run_in_dir};
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4, arr1, arr2};
    use num_complex::{Complex32, Complex64};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
        POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        ConfigDatData, ConfigDatPotential, FmsCluster, FmsControl, FmsDebye, FmsInput, GeomDat,
        GeomDatRow, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData,
        PotBinScalars, PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp, PotRun,
        PotScattering, PotThermal, PotTolerances, RHORRP_POT_BIN_RADIAL_DX, RhorrpDensityBinData,
        RhorrpDensityTextData, RhorrpGgDiagBinData, RhorrpGgSliceBinData, RhorrpNearestAtomColumns,
        config_dat_string, fms_input_string, geom_dat_string, phase_bin_string, pot_input_string,
        read_module_log_dat, read_rhorrp_density_bin, read_rhorrp_density_text,
        rhorrp_gg_slice_bin_bytes, write_module_log_dat, write_pot_bin, write_rhorrp_density_bin,
        write_rhorrp_density_text,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn rhorrp_module_skips_empty_density_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("density.inp"), "")?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_supported_rhorrp_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;
        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        std::fs::write(temp.path().join("density.inp"), "line density.dat 0.0\n")?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed density.inp should fail through the explicit RHORRP runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("density.inp"), "{chain}");
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        assert!(!temp.path().join("logrhorrp.dat").exists());
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_claim_orphan_outputs_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;
        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;

        assert!(!has_supported_rhorrp_output(temp.path())?);
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        assert!(!temp.path().join("logrhorrp.dat").exists());
        Ok(())
    }

    #[test]
    fn rhorrp_module_rejects_generation_without_sources_or_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled RHORRP should require cached output or density sources")?;

        assert!(
            error.to_string().contains(
                "RHORRP core density generation requires cached output or pot.bin/geom.dat"
            )
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_generates_core_density_outputs_from_pot_and_geom() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        write_geom_dat(temp.path())?;

        assert!(has_supported_rhorrp_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        assert_eq!(text.point_count(), 2);
        assert!(text.density_per_angstrom3.iter().all(|value| *value > 0.0));
        let nearest = text
            .nearest
            .as_ref()
            .context("generated text density should include nearest-atom diagnostics")?;
        assert_eq!(nearest.atom_indices.to_vec(), vec![0, 0]);
        assert_eq!(nearest.potential_indices.to_vec(), vec![0, 0]);

        let binary = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        assert_eq!(binary.point_count(), 2);
        assert!(
            binary
                .density_per_angstrom3
                .iter()
                .all(|value| *value > 0.0)
        );

        let log = read_module_log_dat(temp.path().join("logrhorrp.dat"))?;
        assert_log_contains(
            &log,
            "Calculate density: density.dat (       2 total points)",
        );
        assert_log_contains(
            &log,
            "Calculate density: density.bin (       2 total points)",
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_advertise_malformed_core_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        std::fs::write(temp.path().join("geom.dat"), "not geom.dat\n")?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed core source should be rejected during explicit RHORRP run")?;

        assert!(error.to_string().contains("failed to parse"));
        assert!(format!("{error:#}").contains("geom.dat"));
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_claim_cached_core_outputs_with_malformed_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;
        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        std::fs::write(temp.path().join("geom.dat"), "not geom.dat\n")?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("cached core density should not mask malformed core source")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("geom.dat"), "{chain}");
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        assert!(!temp.path().join("logrhorrp.dat").exists());
        Ok(())
    }

    #[test]
    fn rhorrp_module_generates_missing_core_outputs_while_preserving_cached_non_core_outputs()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_mixed_density_input(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        write_geom_dat(temp.path())?;
        let expected_non_core = sample_density_text();
        write_rhorrp_density_text(temp.path().join("valence.dat"), &expected_non_core)?;
        let expected_non_core = read_rhorrp_density_text(temp.path().join("valence.dat"))?;

        assert!(has_supported_rhorrp_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let generated_core = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        assert_eq!(generated_core.point_count(), 2);
        assert!(
            generated_core
                .density_per_angstrom3
                .iter()
                .all(|value| *value > 0.0)
        );
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("valence.dat"))?,
            expected_non_core
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_rejects_missing_non_core_density_when_core_source_is_available() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_mixed_density_input(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        write_geom_dat(temp.path())?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("missing non-core density should require table density sources")?;

        assert!(error.to_string().contains(
            "RHORRP density generation requires cached output or RHORRP table density source files"
        ));
        assert!(!temp.path().join("density.dat").exists());
        Ok(())
    }

    #[test]
    fn rhorrp_module_generates_non_core_density_outputs_from_handoff_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_non_core_density_input(temp.path())?;
        write_table_density_source_handoffs(temp.path())?;

        assert!(has_supported_rhorrp_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let text = read_rhorrp_density_text(temp.path().join("valence.dat"))?;
        assert_eq!(text.point_count(), 2);
        assert!(
            text.density_per_angstrom3
                .iter()
                .all(|value| value.is_finite())
        );
        let nearest = text
            .nearest
            .as_ref()
            .context("generated non-core density should include nearest-atom diagnostics")?;
        assert_eq!(nearest.atom_indices.len(), 2);
        assert_eq!(nearest.potential_indices.len(), 2);

        let binary = read_rhorrp_density_bin(temp.path().join("valence.bin"))?;
        assert_eq!(binary.point_count(), 2);
        assert!(
            binary
                .density_per_angstrom3
                .iter()
                .all(|value| value.is_finite())
        );

        let log = read_module_log_dat(temp.path().join("logrhorrp.dat"))?;
        assert_log_contains(
            &log,
            "Calculate density: valence.dat (       2 total points)",
        );
        assert_log_contains(
            &log,
            "Calculate density: valence.bin (       2 total points)",
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_advertise_malformed_table_phase_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_non_core_density_input(temp.path())?;
        write_table_density_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed table phase handoff should be rejected during explicit run")?;

        assert!(format!("{error:#}").contains("phase.bin"));
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_claim_cached_non_core_outputs_with_malformed_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_non_core_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("valence.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("valence.bin"), &sample_density_bin())?;
        let expected_text = read_rhorrp_density_text(temp.path().join("valence.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("valence.bin"))?;
        write_table_density_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("cached non-core density should not mask malformed table source")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("valence.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("valence.bin"))?,
            expected_bin
        );
        assert!(!temp.path().join("logrhorrp.dat").exists());
        Ok(())
    }

    #[test]
    fn rhorrp_density_callback_source_rejects_malformed_phase_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_table_density_callback_handoffs(temp.path())?;
        std::fs::write(temp.path().join("phase.bin"), "not phase.bin\n")?;

        assert!(!has_rhorrp_density_callback_source(temp.path()));
        Ok(())
    }

    #[test]
    fn rhorrp_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        let expected_text = sample_density_text();
        let expected_bin = sample_density_bin();
        write_rhorrp_density_text(temp.path().join("density.dat"), &expected_text)?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &expected_bin)?;
        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert!(has_supported_rhorrp_output(temp.path())?);
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_does_not_advertise_malformed_cached_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        std::fs::write(temp.path().join("density.dat"), "not RHORRP density\n")?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;

        assert!(!has_supported_rhorrp_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached density should be rejected during explicit RHORRP run")?;

        assert!(format!("{error:#}").contains("density.dat"));
        Ok(())
    }

    #[test]
    fn rhorrp_module_recovers_malformed_cached_core_output_from_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        std::fs::write(temp.path().join("density.dat"), "not RHORRP density\n")?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        write_geom_dat(temp.path())?;

        assert!(has_supported_rhorrp_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        assert_eq!(text.point_count(), 2);
        assert!(text.density_per_angstrom3.iter().all(|value| *value > 0.0));
        let binary = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        assert_eq!(binary.point_count(), 2);
        assert!(
            binary
                .density_per_angstrom3
                .iter()
                .all(|value| *value > 0.0)
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_regenerates_stale_cached_core_output_from_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;
        let stale_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let stale_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_core_pot_bin())?;
        write_geom_dat(temp.path())?;

        assert!(has_supported_rhorrp_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        assert_ne!(text, stale_text);
        assert_eq!(text.point_count(), 2);
        assert!(text.density_per_angstrom3.iter().all(|value| *value > 0.0));
        assert!(text.nearest.is_some());
        let binary = read_rhorrp_density_bin(temp.path().join("density.bin"))?;
        assert_ne!(binary, stale_bin);
        assert_eq!(binary.point_count(), 2);
        assert!(
            binary
                .density_per_angstrom3
                .iter()
                .all(|value| *value > 0.0)
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_roundtrips_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;
        write_module_log_dat(temp.path().join("logrhorrp.dat"), &sample_module_log())?;
        let expected_log = read_module_log_dat(temp.path().join("logrhorrp.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_module_log_dat(temp.path().join("logrhorrp.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn rhorrp_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_density_input(temp.path())?;
        write_rhorrp_density_text(temp.path().join("density.dat"), &sample_density_text())?;
        write_rhorrp_density_bin(temp.path().join("density.bin"), &sample_density_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let log = read_module_log_dat(temp.path().join("logrhorrp.dat"))?;
        assert_log_contains(&log, "Calculating charge density output ...");
        assert_log_contains(
            &log,
            "Calculate density: density.dat (       2 total points)",
        );
        assert_log_contains(
            &log,
            "Calculate density: density.bin (       2 total points)",
        );
        assert_log_contains(&log, "Calculation complete, transfer back to master.");
        assert_log_contains(&log, "Saving density calculation.");
        assert_log_contains(&log, "Done with module: charge density output (RHORRP).");
        Ok(())
    }

    #[test]
    fn rhorrp_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_rhorrp_dir()? else {
            crate::require_fixture!("RHORRP reference test; generated RHORRP reference not found");
        };

        let temp = tempfile::tempdir()?;
        for name in ["density.inp", "density.dat", "density.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }

        let expected_text = read_rhorrp_density_text(temp.path().join("density.dat"))?;
        let expected_bin = read_rhorrp_density_bin(temp.path().join("density.bin"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_rhorrp_density_text(temp.path().join("density.dat"))?,
            expected_text
        );
        assert_eq!(
            read_rhorrp_density_bin(temp.path().join("density.bin"))?,
            expected_bin
        );
        Ok(())
    }

    fn write_geom_dat(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&sample_geom_dat())?,
        )?;
        Ok(())
    }

    fn write_mixed_density_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("density.inp"),
            concat!(
                "line density.dat 0.0 0.0 0.0 core\n",
                "1.0 0.0 0.0 2\n",
                "line valence.dat 0.0 0.0 0.0\n",
                "1.0 0.0 0.0 2\n",
            ),
        )?;
        Ok(())
    }

    fn write_density_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("density.inp"),
            concat!(
                "line density.dat 0.0 0.0 0.0 core\n",
                "1.0 0.0 0.0 2\n",
                "line density.bin 0.0 0.0 0.0 core\n",
                "1.0 0.0 0.0 2\n",
            ),
        )?;
        Ok(())
    }

    fn write_non_core_density_input(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("density.inp"),
            concat!(
                "line valence.dat 0.0 0.0 0.0\n",
                "0.5 0.0 0.0 2\n",
                "line valence.bin 0.0 0.0 0.0\n",
                "0.5 0.0 0.0 2\n",
            ),
        )?;
        Ok(())
    }

    fn write_table_density_source_handoffs(work_dir: &Path) -> Result<()> {
        write_pot_bin(work_dir.join("pot.bin"), &sample_table_pot_bin())?;
        std::fs::write(
            work_dir.join("config.dat"),
            config_dat_string(&sample_table_config_dat())?,
        )?;
        std::fs::write(
            work_dir.join("phase.bin"),
            phase_bin_string(&sample_table_phase_bin())?,
        )?;
        std::fs::write(
            work_dir.join("pot.inp"),
            pot_input_string(&sample_table_pot_input())?,
        )?;
        std::fs::write(
            work_dir.join("fms.inp"),
            fms_input_string(&sample_table_fms_input())?,
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&sample_table_geom_dat())?,
        )?;
        std::fs::write(
            work_dir.join("gg_diag.bin"),
            refeff_io::rhorrp_gg_diag_bin_bytes(&sample_table_gg_diag())?,
        )?;
        Ok(())
    }

    fn write_table_density_callback_handoffs(work_dir: &Path) -> Result<()> {
        write_table_density_source_handoffs(work_dir)?;
        std::fs::write(
            work_dir.join("gg_slice.bin"),
            rhorrp_gg_slice_bin_bytes(&sample_table_gg_slice())?,
        )?;
        Ok(())
    }

    fn sample_geom_dat() -> GeomDat {
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

    fn sample_table_geom_dat() -> GeomDat {
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
                    x: 0.8,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 0,
                },
            ],
        }
    }

    fn sample_core_pot_bin() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["RHORRP core density test".to_string()],
            pad_width: 8,
            nohole: 0,
            ihole: 1,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                interstitial_potential: 0.0,
                interstitial_density: 0.0,
                edge_position: 0.0,
                amplitude_reduction: 1.0,
                relaxation_energy: 0.0,
                plasmon_frequency: 0.0,
                core_valence_energy: 0.0,
                density_radius: 1.0,
                fermi_momentum: 0.0,
                total_charge: 0.0,
                total_volume: 1.0,
            },
            muffin_tin_indices: Array1::from_vec(vec![12]),
            muffin_tin_radii: Array1::from_vec(vec![1.1]),
            norman_indices: Array1::from_vec(vec![40]),
            atomic_numbers: Array1::from_vec(vec![29]),
            kappa: Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: Array1::from_vec(vec![2.1]),
            overlap_factors: Array1::ones(potentials),
            max_overlap_factors: Array1::ones(potentials),
            potential_multiplicities: Array1::ones(potentials),
            ionization: Array1::zeros(potentials),
            initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            large_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
                |(_, orbital, _)| if orbital == 0 { 1.0 } else { 0.0 },
            ),
            small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            electron_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            total_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
            orbital_energies: Array1::zeros(POT_BIN_ORBITALS),
            occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
            norman_charges: Array1::zeros(potentials),
            valence_occupancy: Array2::zeros((4, potentials)),
            raw_text: None,
        }
    }

    fn sample_table_pot_bin() -> PotBinData {
        let potentials = 2;
        PotBinData {
            titles: vec!["RHORRP table density test".to_string()],
            pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
            nohole: 0,
            ihole: 1,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.25,
                fermi_level: -0.4,
                interstitial_potential: -1.2,
                interstitial_density: 0.03,
                edge_position: 9.1,
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                core_valence_energy: -3.0,
                density_radius: 1.7,
                fermi_momentum: 0.9,
                total_charge: 42.0,
                total_volume: 11.0,
            },
            muffin_tin_indices: Array1::from_vec(vec![12, 13]),
            muffin_tin_radii: Array1::from_vec(vec![1.1, 1.2]),
            norman_indices: Array1::from_vec(vec![20, 21]),
            atomic_numbers: Array1::from_vec(vec![29, 8]),
            kappa: Array1::from_iter(-20..=20),
            norman_radii: Array1::from_vec(vec![2.1, 2.2]),
            overlap_factors: Array1::from_vec(vec![0.9, 0.8]),
            max_overlap_factors: Array1::from_vec(vec![1.3, 1.4]),
            potential_multiplicities: Array1::from_vec(vec![1.0, 1.0]),
            ionization: Array1::from_vec(vec![0.0, 1.0]),
            initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                0.001 * (row + 1) as f64
            }),
            initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
                -0.001 * (row + 1) as f64
            }),
            large_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
                |(row, orbital, potential)| {
                    0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
                },
            ),
            small_components: Array3::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
                |(row, orbital, potential)| {
                    -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
                },
            ),
            large_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
                |(coef, orbital, potential)| {
                    0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
                },
            ),
            small_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
                |(coef, orbital, potential)| {
                    -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
                },
            ),
            electron_density: table_radial_matrix(potentials, 0.01),
            coulomb_potential: table_radial_matrix(potentials, -0.02),
            total_potential: table_radial_matrix(potentials, -0.03),
            valence_density: table_radial_matrix(potentials, 0.004),
            valence_potential: table_radial_matrix(potentials, -0.005),
            magnetization_density: table_radial_matrix(potentials, 0.0002),
            orbital_occupancy: Array2::from_shape_fn(
                (POT_BIN_ORBITALS, potentials),
                |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
            ),
            orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
                -10.0 + orbital as f64 * 0.25
            }),
            occupied_orbital_indices: Array2::from_shape_fn(
                (POT_BIN_IORB_SLOTS, potentials),
                |(slot, _)| slot as i32 - 5,
            ),
            norman_charges: Array1::from_vec(vec![28.5, 7.5]),
            valence_occupancy: Array2::from_shape_fn((1, potentials), |(_, potential)| {
                potential as f64
            }),
            raw_text: None,
        }
    }

    fn table_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
        Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
            scale * (row + 1) as f64 + potential as f64 * 0.125
        })
    }

    fn sample_table_config_dat() -> ConfigDatData {
        let mut first_occupations = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
        let mut first_valence = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
        first_occupations[0] = 1.0;
        first_occupations[1] = 2.0;
        first_valence[1] = 0.5;

        let mut second_occupations = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
        let mut second_valence = Array1::zeros(refeff_io::CONFIG_DAT_ORBITAL_COUNT);
        second_occupations[0] = 2.0;
        second_occupations[1] = 2.0;
        second_occupations[2] = 1.0;
        second_valence[2] = 1.0;

        ConfigDatData {
            header_lines: Vec::new(),
            potentials: vec![
                ConfigDatPotential {
                    potential_index: 0,
                    atomic_number: 29,
                    element: "Cu".to_string(),
                    occupations: first_occupations,
                    valence_occupations: first_valence,
                    spin_occupations: None,
                },
                ConfigDatPotential {
                    potential_index: 1,
                    atomic_number: 8,
                    element: "O".to_string(),
                    occupations: second_occupations,
                    valence_occupations: second_valence,
                    spin_occupations: None,
                },
            ],
        }
    }

    fn sample_table_phase_bin() -> PhaseBinData {
        let spin_count = 1;
        let energy_grid = Array1::from_vec(vec![
            Complex64::new(0.15, 0.02),
            Complex64::new(0.15, 0.04),
            Complex64::new(0.20, 0.04),
            Complex64::new(0.25, 0.04),
        ]);
        let energy_count = energy_grid.len();
        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 1,
            pad_width: 8,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: 0.045,
                edge_energy: 9.8,
            },
            energy_grid,
            reference_energy: Array2::zeros((energy_count, spin_count)),
            potentials: vec![
                sample_table_phase_potential(29, "Cu", energy_count, spin_count),
                sample_table_phase_potential(8, "O", energy_count, spin_count),
            ],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_table_phase_potential(
        atomic_number: usize,
        label: &str,
        energy_count: usize,
        spin_count: usize,
    ) -> PhaseBinPotential {
        PhaseBinPotential {
            lmax: 0,
            atomic_number,
            label: label.to_string(),
            phase_shifts: Array3::zeros((energy_count, 1, spin_count)),
        }
    }

    fn sample_table_pot_input() -> PotInput {
        PotInput {
            control: PotControl {
                mpot: 1,
                nph: 1,
                ntitle: 1,
                ihole: 1,
                ipr1: 0,
                iafolp: 0,
                ixc: 0,
                ispec: 0,
                iscfxc: 0,
            },
            run: PotRun {
                nmix: 0,
                nohole: 0,
                jumprm: 0,
                inters: 0,
                nscmt: 0,
                icoul: 0,
                lfms1: 0,
                iunf: 0,
            },
            titles: vec!["RHORRP table density test".to_string()],
            scattering: PotScattering {
                gamach: 0.0,
                rgrd: RHORRP_POT_BIN_RADIAL_DX,
                ca1: 0.0,
                ecv: 0.0,
                totvol: 1.0,
                rfms1: 0.0,
                corval_emin: 0.0,
            },
            potentials: vec![
                PotPotential {
                    z: 29,
                    lmaxsc: 0,
                    xnatph: 1.0,
                    xion: 0.0,
                    folp: 1.0,
                },
                PotPotential {
                    z: 8,
                    lmaxsc: 0,
                    xnatph: 1.0,
                    xion: 0.0,
                    folp: 1.0,
                },
            ],
            external_pot: false,
            start_from_file: false,
            overlap_shells: vec![Vec::<PotOverlapShell>::new(), Vec::<PotOverlapShell>::new()],
            chsh_type: 0,
            config_type: 1,
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

    fn sample_table_fms_input() -> FmsInput {
        FmsInput {
            control: FmsControl {
                mfms: 1,
                idwopt: 0,
                minv: 0,
            },
            cluster: FmsCluster {
                rfms2: 3.0,
                rdirec: 0.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: FmsDebye {
                tk: 0.0,
                thetad: 0.0,
                sig2g: 0.0,
            },
            lmaxph: vec![0, 0],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 0,
        }
    }

    fn sample_table_gg_diag() -> RhorrpGgDiagBinData {
        RhorrpGgDiagBinData {
            values: Array4::from_elem((4, 2, 1, 1), Complex32::new(0.0, 0.0)),
        }
    }

    fn sample_table_gg_slice() -> RhorrpGgSliceBinData {
        RhorrpGgSliceBinData {
            values: Array3::from_elem((4, 1, 2), Complex32::new(0.0, 0.0)),
        }
    }

    fn sample_density_text() -> RhorrpDensityTextData {
        RhorrpDensityTextData {
            points_angstrom: arr2(&[[0.0, 0.0, 0.0], [0.529_177_249, 0.0, 0.0]]),
            density_per_angstrom3: arr1(&[1.0, 2.0]),
            nearest: Some(RhorrpNearestAtomColumns {
                displacement_bohr: arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atom_indices: arr1(&[0, 0]),
                potential_indices: arr1(&[0, 0]),
            }),
        }
    }

    fn sample_density_bin() -> RhorrpDensityBinData {
        RhorrpDensityBinData {
            origin_angstrom: [0.0, 0.0, 0.0],
            axes_angstrom: arr2(&[[1.0], [0.0], [0.0]]),
            points_per_axis: vec![2],
            density_per_angstrom3: arr1(&[1.0, 2.0]),
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating charge density output ...".to_string(),
                "Calculate density: density.dat (       2 total points)".to_string(),
                "Calculation complete, transfer back to master.".to_string(),
                "Saving density calculation.".to_string(),
                "Done with module: charge density output (RHORRP).".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 5],
        }
    }

    fn assert_log_contains(log: &ModuleLogData, expected: &str) {
        assert!(
            log.lines.iter().any(|line| line.contains(expected)),
            "expected log to contain {expected:?}, got {:?}",
            log.lines
        );
    }

    fn reference_rhorrp_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(Path::parent)
            .context("failed to find workspace root")?;
        let path = workspace.join("reference-work/golden/XANES/Cu/rhorrp-density");
        let required = ["density.inp", "density.dat", "density.bin"];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
