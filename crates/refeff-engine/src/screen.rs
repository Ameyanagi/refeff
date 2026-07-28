use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array3};
use refeff_core::{
    Complex, SCREEN_HARTREE_EV, ScreenContourEnergyGridInput,
    screen::screen_bare_core_hole_potential, screen::screen_radial_grid,
    screen_contour_energy_grid,
};
use refeff_io::{
    ConfigDatData, CrpaInput, ModuleLogData, PhaseBinData, PotBinData, PotInput,
    ScreenFmsClusterGreenHandoff, ScreenFmsClusterGreenHandoffInput,
    ScreenFovrgPhaseGridHandoffInput, ScreenFovrgRadialHandoff, ScreenFovrgRadialHandoffInput,
    ScreenInput, ScreenPotentialKernelHandoff, ScreenPotentialKernelHandoffInput,
    ScreenResponseAssemblyHandoffInput, VtotDatData, WscrnDatData, apot_core_hole_columns,
    read_apot_bin, read_config_dat, read_gg_bin, read_module_log_dat, read_phase_bin, read_pot_bin,
    read_vtot_dat, read_wscrn_dat, screen_fms_cluster_green_handoff,
    screen_fovrg_phase_grid_handoff, screen_fovrg_radial_handoff, screen_potential_kernel_handoff,
    screen_response_assembly_handoff, vtot_dat_from_wscrn_and_pot_bin, write_module_log_dat,
    write_vtot_dat, write_wscrn_dat,
};

use crate::fms::{
    ScreenFmsSourceGridInput, build_screen_fms_source_grid_handoff,
    build_screen_fms_source_grid_handoff_from_generated_phases, screen_fms_source_angular_count,
    screen_fms_source_angular_count_for_potential_count,
};
use crate::work_dir_for_input;

const SCREEN_RADIAL_POINTS: usize = 251;
const SCREEN_RADIAL_GRID_STEP: f64 = 0.05;
const SCREEN_RADIAL_GRID_ORIGIN: f64 = 8.8;
const SCREEN_CORE_HOLE_TOLERANCE: f64 = 1.0e-4;
const SCREEN_VTOT_HANDOFF_TOLERANCE: f64 = 1.0e-8;
const SCREEN_SOURCE_REQUIREMENT_ERROR: &str = "SCREEN generation requires cached wscrn.dat, recoverable vtot.dat/apot.bin, or complete pot/config/FMS source handoffs";

/// Source response state shared by SCREEN and CRPA production handoffs.
pub(crate) struct ScreenSourceResponseComponents {
    pub(crate) potential: ScreenPotentialKernelHandoff,
    pub(crate) fms: ScreenFmsClusterGreenHandoff,
    pub(crate) radial: ScreenFovrgRadialHandoff,
    pub(crate) fermi_level_hartree: f64,
}

/// Run the supported FEFF SCREEN cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_screen_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_wscrn_handoff(work_dir)? {
        return run_supported_wscrn_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF SCREEN run can be satisfied from an existing `wscrn.dat`.
pub(crate) fn has_cached_screen_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("screen.inp").is_file() || !work_dir.join("wscrn.dat").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !screen_enabled(work_dir)? {
        return Ok(false);
    }
    Ok(can_use_screen_output(work_dir, &input))
}

/// Whether SCREEN has already completed its cached-output compatibility stage.
///
/// Validation-only `screen-wscrn` handoffs deliberately do not create
/// `logscreen.dat`, so required full-run orchestration should not skip the
/// SCREEN stage solely because that sidecar recovered `wscrn.dat`.
pub(crate) fn has_completed_screen_output(work_dir: &Path) -> Result<bool> {
    if !has_cached_screen_output(work_dir)? {
        return Ok(false);
    }
    let log_path = work_dir.join("logscreen.dat");
    Ok(log_path.is_file() && prepare_module_log_cache(&log_path).is_ok())
}

/// Whether a usable cached `wscrn.dat` can complete the SCREEN stage by
/// regenerating the optional `vtot.dat` handoff.
pub(crate) fn has_recoverable_cached_screen_stage(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("screen.inp").is_file() {
        return Ok(false);
    }
    let input = match read_input(work_dir) {
        Ok(input) => input,
        Err(_) => return Ok(false),
    };
    if !screen_enabled(work_dir)? || has_malformed_screen_source_handoff(work_dir, &input) {
        return Ok(false);
    }
    let Ok(data) = prepare_cached_wscrn_output(work_dir) else {
        return Ok(false);
    };
    Ok(optional_vtot_needs_generation_from_pot_bin(work_dir, &data)
        || !work_dir.join("logscreen.dat").is_file())
}

/// Whether SCREEN can recover `wscrn.dat` from `vtot.dat` and `apot.bin`
/// before the remaining screened-core-hole solver boundary.
pub(crate) fn has_supported_wscrn_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("screen.inp").is_file() || has_cached_screen_output(work_dir)? {
        return Ok(false);
    }
    if read_input(work_dir).is_err() {
        return Ok(false);
    }
    Ok(screen_enabled(work_dir)? && has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir))
}

/// Whether SCREEN can generate full `wscrn.dat` output from complete Rust
/// source handoffs instead of a cached table or recovery-only sidecar.
pub(crate) fn has_supported_screen_source_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("screen.inp").is_file()
        || has_cached_screen_output(work_dir)?
        || has_usable_wscrn_handoff_in_dir(work_dir)
    {
        return Ok(false);
    }

    let input = match read_input(work_dir) {
        Ok(input) => input,
        Err(_) => return Ok(false),
    };
    if !screen_enabled(work_dir)? {
        return Ok(false);
    }

    match build_source_screen_response_components(work_dir, &input) {
        Ok(components) => Ok(components.is_some()),
        Err(_) => Ok(false),
    }
}

/// Recover only the Rust-backed SCREEN `wscrn.dat` sidecar.
///
/// This intentionally does not write `logscreen.dat` or report the full SCREEN
/// module as complete.
pub(crate) fn run_supported_wscrn_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    if !work_dir.join("screen.inp").is_file() || has_cached_screen_output(work_dir)? {
        return Ok(0);
    }
    read_input(work_dir)?;
    if !screen_enabled(work_dir)? || !has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir) {
        return Ok(0);
    }
    recover_wscrn_from_vtot_and_apot_in_dir(work_dir)
}

/// Complete a SCREEN cached-output stage from an existing `wscrn.dat` plus a
/// recoverable `vtot.dat` handoff, without invoking source SCREEN generation.
pub(crate) fn run_recoverable_cached_screen_stage_in_dir(work_dir: &Path) -> Result<usize> {
    read_input(work_dir)?;
    if !screen_enabled(work_dir)? {
        return Ok(0);
    }

    let output_path = work_dir.join("wscrn.dat");
    let data = prepare_cached_wscrn_output(work_dir)?;
    let row_count = data.row_count();
    write_cached_output(&output_path, &data)?;
    let vtot_source_handoff = optional_vtot_needs_generation_from_pot_bin(work_dir, &data);
    let vtot_rows = write_optional_vtot_cache(work_dir, &data)?;
    write_or_recover_module_log(&work_dir.join("logscreen.dat"), vtot_source_handoff)?;
    Ok(row_count + vtot_rows)
}

/// Run the FEFF SCREEN path from cached output or complete source handoffs.
///
/// This path preserves the module boundary for existing FEFF caches by
/// validating and re-rendering the radial screened-potential table plus the
/// optional `vtot.dat` table that XSPH can produce after applying the screened
/// core-hole potential. When the XSPH sidecar is missing but `pot.bin` is
/// present, it regenerates `vtot.dat` from the FEFF handoff state. Malformed
/// `wscrn.dat` caches are also recovered from valid `vtot.dat` plus `apot.bin`
/// handoffs, and complete `pot.bin`/`config.dat`/FMS source bundles generate
/// `wscrn.dat` directly.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !screen_enabled(work_dir)? {
        return Ok(0);
    }

    let output_path = work_dir.join("wscrn.dat");
    let generated_source_rows = if !can_use_cached_wscrn_output_for_input(work_dir, &input) {
        write_source_screen_outputs(work_dir, &input)
            .context("failed to generate SCREEN wscrn.dat from source handoffs")?
    } else {
        0
    };
    let generated_wscrn_rows = if !can_use_cached_wscrn_output(work_dir) {
        write_generated_wscrn_from_vtot_and_apot(work_dir)
            .context("failed to generate SCREEN wscrn.dat from vtot.dat and apot.bin")?
    } else {
        0
    };
    if !output_path.is_file() {
        validate_optional_source_potential_kernel_handoff(work_dir, &input)?;
        validate_optional_source_fms_trace_handoff(work_dir, &input)?;
        bail!(SCREEN_SOURCE_REQUIREMENT_ERROR);
    }

    let mut data = read_wscrn_dat(&output_path)
        .with_context(|| format!("failed to read {}", output_path.display()))?;
    refresh_wscrn_core_hole_from_apot(work_dir, &mut data)
        .with_context(|| format!("failed to refresh {}", output_path.display()))?;
    let row_count = data.row_count();
    write_cached_output(&output_path, &data)?;
    let vtot_source_handoff = optional_vtot_needs_generation_from_pot_bin(work_dir, &data);
    let vtot_rows = write_optional_vtot_cache(work_dir, &data)?;
    if generated_source_rows > 0 || generated_wscrn_rows > 0 {
        write_generated_module_log(&work_dir.join("logscreen.dat"))?;
    } else {
        write_or_recover_module_log(&work_dir.join("logscreen.dat"), vtot_source_handoff)?;
    }
    Ok(row_count + vtot_rows)
}

/// Recover a shared SCREEN `wscrn.dat` sidecar from `vtot.dat` and `apot.bin`.
///
/// RIXS can consume the same screened-core-hole handoff even when a standalone
/// SCREEN stage is not being reported, so keep this recovery step available
/// without requiring `screen.inp` parsing.
pub(crate) fn recover_wscrn_from_vtot_and_apot_in_dir(work_dir: &Path) -> Result<usize> {
    write_generated_wscrn_from_vtot_and_apot(work_dir)
}

/// Build a shared SCREEN `wscrn.dat` sidecar from `vtot.dat` and `apot.bin`
/// without mutating the work directory.
pub(crate) fn generated_wscrn_from_vtot_and_apot_in_dir(
    work_dir: &Path,
) -> Result<Option<WscrnDatData>> {
    if !has_recoverable_screen_output(work_dir) {
        return Ok(None);
    }
    let data = generated_wscrn_from_vtot_and_apot(work_dir)?;
    Ok(Some(data))
}

/// Whether a missing or unreadable shared SCREEN `wscrn.dat` sidecar can be
/// regenerated from typed `vtot.dat` and `apot.bin` handoffs.
pub(crate) fn has_recoverable_wscrn_from_vtot_and_apot_in_dir(work_dir: &Path) -> bool {
    !can_use_cached_wscrn_output(work_dir) && can_recover_wscrn_from_vtot_and_apot(work_dir)
}

pub(crate) fn has_usable_wscrn_handoff_in_dir(work_dir: &Path) -> bool {
    can_use_cached_wscrn_output(work_dir)
}

fn validate_optional_source_potential_kernel_handoff(
    work_dir: &Path,
    input: &ScreenInput,
) -> Result<()> {
    let pot_path = work_dir.join("pot.bin");
    if !pot_path.is_file() {
        return Ok(());
    }

    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
        screen: input,
        pot: &pot,
        potential_index: 0,
    })
    .with_context(|| {
        format!(
            "failed to build SCREEN potential-kernel handoff from {}",
            pot_path.display()
        )
    })?;
    Ok(())
}

fn validate_optional_source_fms_trace_handoff(work_dir: &Path, input: &ScreenInput) -> Result<()> {
    let phase_path = work_dir.join("phase.bin");
    let green_path = work_dir.join("gg.bin");
    if !(phase_path.is_file() && green_path.is_file()) {
        return Ok(());
    }

    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    let green = read_gg_bin(&green_path)
        .with_context(|| format!("failed to read {}", green_path.display()))?;
    let angular_count = screen_angular_count(input)?;
    screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
        phase: &phase,
        green: &green,
        potential_index: 0,
        spin_index: 0,
        angular_count,
    })
    .with_context(|| {
        format!(
            "failed to build SCREEN FMS trace handoff from {} and {}",
            phase_path.display(),
            green_path.display()
        )
    })?;
    Ok(())
}

fn write_source_screen_outputs(work_dir: &Path, input: &ScreenInput) -> Result<usize> {
    let Some(data) = generated_source_wscrn_data(work_dir, input)? else {
        return Ok(0);
    };

    let row_count = data.row_count();
    write_cached_output(&work_dir.join("wscrn.dat"), &data)?;
    Ok(row_count)
}

fn generated_source_wscrn_data(
    work_dir: &Path,
    input: &ScreenInput,
) -> Result<Option<WscrnDatData>> {
    let components = match build_source_screen_response_components(work_dir, input) {
        Ok(Some(components)) => components,
        Ok(None) => return Ok(None),
        Err(error) if screen_source_response_unsupported(&error) => return Ok(None),
        Err(error) => return Err(error),
    };

    let response = screen_response_assembly_handoff(ScreenResponseAssemblyHandoffInput {
        potential: &components.potential,
        fms: &components.fms,
        reference_energies_hartree: components.radial.reference_energies_hartree.view(),
        regular_solutions: components
            .radial
            .matched
            .solved
            .radial_cubes
            .regular_large
            .view(),
        irregular_solutions: components
            .radial
            .matched
            .solved
            .radial_cubes
            .irregular_large
            .view(),
        header_lines: &[],
    })
    .context("failed to assemble SCREEN response from source handoffs")?;
    let mut wscrn = response.wscrn;
    refresh_wscrn_core_hole_from_apot(work_dir, &mut wscrn)
        .with_context(|| "failed to refresh source-generated SCREEN wscrn.dat")?;
    Ok(Some(wscrn))
}

fn screen_source_response_unsupported(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("FOVRG orbital_length count 0 is below minimum 1")
            || message.contains("POT SCF FOVRG source grid has no active bound orbital")
    })
}

fn has_malformed_screen_source_handoff(work_dir: &Path, input: &ScreenInput) -> bool {
    match build_source_screen_response_components(work_dir, input) {
        Ok(_) => false,
        Err(error) => !screen_source_response_unsupported(&error),
    }
}

pub(crate) fn build_source_screen_response_components(
    work_dir: &Path,
    input: &ScreenInput,
) -> Result<Option<ScreenSourceResponseComponents>> {
    let pot_path = work_dir.join("pot.bin");
    let config_path = work_dir.join("config.dat");
    let phase_path = work_dir.join("phase.bin");
    let green_path = work_dir.join("gg.bin");
    let has_source_fms = work_dir.join("fms.inp").is_file() && work_dir.join("geom.dat").is_file();
    let has_phase = phase_path.is_file();
    let has_cached_green = green_path.is_file();
    if !(pot_path.is_file()
        && config_path.is_file()
        && (has_source_fms || (has_phase && has_cached_green)))
    {
        return Ok(None);
    }

    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let config = read_config_dat(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let phase = if has_phase {
        Some(
            read_phase_bin(&phase_path)
                .with_context(|| format!("failed to read {}", phase_path.display()))?,
        )
    } else {
        None
    };
    let potential_count = phase
        .as_ref()
        .map_or_else(|| pot.potential_count(), PhaseBinData::potential_count);
    ensure!(
        potential_count > 0,
        "SCREEN source generation requires at least one potential"
    );
    let angular_count = screen_angular_count(input)?;
    let (source_phase_angular_count, source_fms_angular_count) = if has_source_fms {
        let fms_angular_count = if let Some(phase) = phase.as_ref() {
            screen_fms_source_angular_count(work_dir, phase)?
        } else {
            screen_fms_source_angular_count_for_potential_count(work_dir, potential_count)?
        };
        let raw_fms_angular_count = fms_angular_count.unwrap_or(angular_count);
        let source_fms_angular_count = raw_fms_angular_count.min(angular_count);
        (
            raw_fms_angular_count.max(angular_count),
            source_fms_angular_count,
        )
    } else {
        (angular_count, angular_count)
    };
    let potential = screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
        screen: input,
        pot: &pot,
        potential_index: 0,
    })
    .with_context(|| {
        format!(
            "failed to build SCREEN potential-kernel handoff from {}",
            pot_path.display()
        )
    })?;

    let screen_energies = screen_contour_energies_hartree(input, &pot)
        .context("failed to build SCREEN contour energy grid from screen.inp")?;

    let source_fms = if has_source_fms {
        let (radial, phase_shifts) = build_screen_fovrg_phase_grid(
            input,
            &pot,
            &config,
            potential_count,
            &potential,
            screen_energies.view(),
            source_phase_angular_count,
        )
        .with_context(|| {
            format!(
                "failed to build SCREEN FOVRG source phase grid from {} and {}",
                pot_path.display(),
                config_path.display()
            )
        })?;
        let grid = ScreenFmsSourceGridInput {
            energy_grid_hartree: screen_energies.view(),
            wave_numbers_bohr: radial.wave_numbers.view(),
            phase_shifts: phase_shifts.view(),
            angular_count: source_fms_angular_count,
            cluster_radius_angstrom: input.rfms,
            direct_cutoff_angstrom: input.rfms * 2.0,
        };
        if let Some(phase) = phase.as_ref() {
            build_screen_fms_source_grid_handoff(work_dir, phase, grid)?
        } else {
            build_screen_fms_source_grid_handoff_from_generated_phases(work_dir, &pot, grid)?
        }
        .map(|fms| (fms, radial))
    } else {
        None
    };

    let (fms, radial) = if let Some(source_fms) = source_fms {
        source_fms
    } else {
        let Some(phase) = phase.as_ref() else {
            return Ok(None);
        };
        let green = read_gg_bin(&green_path)
            .with_context(|| format!("failed to read {}", green_path.display()))?;
        let fms = screen_fms_cluster_green_handoff(ScreenFmsClusterGreenHandoffInput {
            phase,
            green: &green,
            potential_index: 0,
            spin_index: 0,
            angular_count,
        })
        .with_context(|| {
            format!(
                "failed to build SCREEN FMS trace handoff from {} and {}",
                phase_path.display(),
                green_path.display()
            )
        })?;
        let radial =
            build_screen_absorber_fovrg_radial(input, &pot, &config, phase, &potential, &fms)?;
        (fms, radial)
    };

    Ok(Some(ScreenSourceResponseComponents {
        potential,
        fms,
        radial,
        fermi_level_hartree: pot.scalars.fermi_level,
    }))
}

fn build_screen_absorber_fovrg_radial(
    input: &ScreenInput,
    pot: &PotBinData,
    config: &ConfigDatData,
    phase: &PhaseBinData,
    potential: &ScreenPotentialKernelHandoff,
    fms: &refeff_io::ScreenFmsClusterGreenHandoff,
) -> Result<ScreenFovrgRadialHandoff> {
    ensure!(
        phase.reference_energy.ncols() > 0,
        "SCREEN phase.bin has no reference-energy spin columns"
    );
    ensure!(
        phase.reference_energy.nrows() >= fms.energies_hartree.len(),
        "SCREEN phase.bin reference-energy row count {} is smaller than FMS energy count {}",
        phase.reference_energy.nrows(),
        fms.energies_hartree.len()
    );
    let reference_energies = Array1::from_iter(
        phase
            .reference_energy
            .column(0)
            .iter()
            .take(fms.energies_hartree.len())
            .copied(),
    );
    Ok(screen_fovrg_radial_handoff(
        ScreenFovrgRadialHandoffInput {
            potential,
            pot,
            config,
            energies_hartree: fms.energies_hartree.view(),
            reference_energies_hartree: reference_energies.view(),
            angular_count: fms.cluster_greens.ncols(),
            use_hankel_boundary: input.irrh != 0,
        },
    )?)
}

fn build_screen_fovrg_phase_grid(
    input: &ScreenInput,
    pot: &PotBinData,
    config: &ConfigDatData,
    potential_count: usize,
    absorber_potential: &ScreenPotentialKernelHandoff,
    energies_hartree: ndarray::ArrayView1<'_, Complex>,
    angular_count: usize,
) -> Result<(ScreenFovrgRadialHandoff, Array3<Complex>)> {
    ensure!(
        potential_count > 0,
        "SCREEN FOVRG source phase grid requires at least one potential"
    );
    let reference_energies = Array1::zeros(energies_hartree.len());
    let mut potentials = Vec::with_capacity(potential_count);
    for potential_index in 0..potential_count {
        if potential_index == absorber_potential.potential_index {
            potentials.push(absorber_potential.clone());
            continue;
        }
        potentials.push(
            screen_potential_kernel_handoff(ScreenPotentialKernelHandoffInput {
                screen: input,
                pot,
                potential_index,
            })
            .with_context(|| {
                format!(
                    "failed to build SCREEN potential-kernel handoff for potential {}",
                    potential_index
                )
            })?,
        );
    }

    let handoff = screen_fovrg_phase_grid_handoff(ScreenFovrgPhaseGridHandoffInput {
        potentials: &potentials,
        absorber_potential_index: absorber_potential.potential_index,
        pot,
        config,
        energies_hartree,
        reference_energies_hartree: reference_energies.view(),
        angular_count,
        use_hankel_boundary: input.irrh != 0,
    })?;

    Ok((handoff.absorber_radial, handoff.phase_shifts))
}

fn screen_contour_energies_hartree(
    input: &ScreenInput,
    pot: &PotBinData,
) -> Result<Array1<Complex>> {
    let real_points = usize::try_from(input.ner).context("screen.inp ner does not fit usize")?;
    let imaginary_points =
        usize::try_from(input.nei).context("screen.inp nei does not fit usize")?;
    let capacity = real_points
        .checked_add(
            imaginary_points
                .checked_mul(2)
                .context("SCREEN contour point capacity overflowed")?,
        )
        .and_then(|value| value.checked_add(8))
        .context("SCREEN contour point capacity overflowed")?;
    let grid = screen_contour_energy_grid(ScreenContourEnergyGridInput {
        min_real_energy: pot.scalars.fermi_level + input.emin / SCREEN_HARTREE_EV,
        max_real_energy: pot.scalars.fermi_level + input.emax / SCREEN_HARTREE_EV,
        max_imaginary_energy: input.eimax / SCREEN_HARTREE_EV,
        min_imaginary_energy: input.ermin / SCREEN_HARTREE_EV,
        real_points,
        imaginary_points,
        max_points: capacity,
    })?;
    Ok(Array1::from_iter(
        grid.energies.iter().take(grid.active_len).copied(),
    ))
}

fn screen_angular_count(input: &ScreenInput) -> Result<usize> {
    ensure!(
        input.maxl > 0,
        "screen.inp maxl must be positive for SCREEN FMS handoff validation, got {}",
        input.maxl
    );
    usize::try_from(input.maxl).context("screen.inp maxl does not fit SCREEN angular_count")
}

fn read_input(work_dir: &Path) -> Result<ScreenInput> {
    let input_path = work_dir.join("screen.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    ScreenInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn screen_enabled(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file() {
        return Ok(!crpa_enabled(work_dir)?);
    }

    screen_requested(work_dir)
}

fn crpa_enabled(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("crpa.inp");
    if !input_path.is_file() {
        return Ok(false);
    }

    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let input = CrpaInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))?;
    Ok(input.enabled)
}

fn screen_requested(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file() {
        return Ok(false);
    }

    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    let input = PotInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))?;
    Ok(input.run.nohole == 2)
}

fn can_use_screen_output(work_dir: &Path, input: &ScreenInput) -> bool {
    if let Ok(data) = prepare_cached_wscrn_output_for_input(work_dir, input) {
        if optional_vtot_needs_generation_from_pot_bin(work_dir, &data) {
            return true;
        }
        return prepare_module_log_cache(&work_dir.join("logscreen.dat")).is_ok();
    }
    false
}

fn can_use_cached_wscrn_output(work_dir: &Path) -> bool {
    prepare_cached_wscrn_output(work_dir).is_ok()
}

fn can_use_cached_wscrn_output_for_input(work_dir: &Path, input: &ScreenInput) -> bool {
    prepare_cached_wscrn_output_for_input(work_dir, input).is_ok()
}

fn prepare_cached_wscrn_output(work_dir: &Path) -> Result<WscrnDatData> {
    prepare_cached_wscrn_output_impl(work_dir, None)
}

fn prepare_cached_wscrn_output_for_input(
    work_dir: &Path,
    input: &ScreenInput,
) -> Result<WscrnDatData> {
    prepare_cached_wscrn_output_impl(work_dir, Some(input))
}

fn prepare_cached_wscrn_output_impl(
    work_dir: &Path,
    input: Option<&ScreenInput>,
) -> Result<WscrnDatData> {
    let path = work_dir.join("wscrn.dat");
    let mut data =
        read_wscrn_dat(&path).with_context(|| format!("failed to read {}", path.display()))?;
    refresh_wscrn_core_hole_from_apot(work_dir, &mut data)
        .with_context(|| format!("failed to refresh {}", path.display()))?;
    if recoverable_wscrn_handoff_supersedes_cache(work_dir, &data) {
        bail!("SCREEN wscrn.dat cache is stale against vtot.dat/apot.bin handoff");
    }
    if let Some(input) = input
        && cached_wscrn_output_is_stale_against_source(work_dir, input, &data)?
    {
        bail!("SCREEN wscrn.dat cache is stale against source handoffs");
    }
    if !can_prepare_optional_vtot_cache(work_dir, &data) {
        bail!("SCREEN vtot.dat sidecar is not usable or recoverable");
    }
    Ok(data)
}

fn cached_wscrn_output_is_stale_against_source(
    work_dir: &Path,
    input: &ScreenInput,
    cached: &WscrnDatData,
) -> Result<bool> {
    let Some(generated) = generated_source_wscrn_data(work_dir, input)? else {
        return Ok(false);
    };
    Ok(!wscrn_matches_generated_handoff(cached, &generated))
}

fn recoverable_wscrn_handoff_supersedes_cache(work_dir: &Path, data: &WscrnDatData) -> bool {
    if !has_recoverable_screen_output(work_dir) {
        return false;
    }
    let Ok(generated) = generated_wscrn_from_vtot_and_apot(work_dir) else {
        return false;
    };
    !wscrn_matches_generated_handoff(data, &generated)
}

fn wscrn_matches_generated_handoff(actual: &WscrnDatData, expected: &WscrnDatData) -> bool {
    actual.row_count() == expected.row_count()
        && columns_match(
            &actual.radius_bohr,
            &expected.radius_bohr,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
        && columns_match(
            &actual.screened_potential,
            &expected.screened_potential,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
        && columns_match(
            &actual.core_hole_potential,
            &expected.core_hole_potential,
            SCREEN_CORE_HOLE_TOLERANCE,
        )
}

fn can_recover_wscrn_from_vtot_and_apot(work_dir: &Path) -> bool {
    if !has_recoverable_screen_output(work_dir) {
        return false;
    }

    let vtot_path = work_dir.join("vtot.dat");
    let vtot = match read_vtot_dat(&vtot_path) {
        Ok(vtot) => vtot,
        Err(_) => return false,
    };
    let row_count = vtot.row_count();
    let mut data = WscrnDatData {
        header_lines: vec!["# r       w_scrn(r)      v_ch(r)".to_string()],
        radius_bohr: vtot.radius_bohr,
        screened_potential: vtot.screened_core_hole_potential,
        core_hole_potential: Array1::zeros(row_count),
    };
    refresh_wscrn_core_hole_from_apot(work_dir, &mut data).is_ok()
        && can_prepare_optional_vtot_cache(work_dir, &data)
}

fn can_prepare_optional_vtot_cache(work_dir: &Path, wscrn: &WscrnDatData) -> bool {
    let vtot_path = work_dir.join("vtot.dat");
    if vtot_path.is_file() {
        return match read_vtot_dat(&vtot_path) {
            Ok(vtot) if vtot_matches_current_handoff(work_dir, wscrn, &vtot) => true,
            Ok(_) | Err(_) => can_generate_optional_vtot_from_pot_bin(work_dir, wscrn),
        };
    }

    if can_generate_optional_vtot_from_pot_bin(work_dir, wscrn) {
        return true;
    }
    !work_dir.join("pot.bin").is_file()
}

fn can_generate_optional_vtot_from_pot_bin(work_dir: &Path, wscrn: &WscrnDatData) -> bool {
    generated_optional_vtot_from_pot_bin(work_dir, wscrn).is_some()
}

fn generated_optional_vtot_from_pot_bin(
    work_dir: &Path,
    wscrn: &WscrnDatData,
) -> Option<VtotDatData> {
    let pot_path = work_dir.join("pot.bin");
    if !pot_path.is_file() {
        return None;
    }
    let pot = match read_pot_bin(&pot_path) {
        Ok(pot) => pot,
        Err(_) => return None,
    };
    vtot_dat_from_wscrn_and_pot_bin(wscrn, &pot).ok()
}

fn vtot_matches_current_handoff(work_dir: &Path, wscrn: &WscrnDatData, vtot: &VtotDatData) -> bool {
    if let Some(generated) = generated_optional_vtot_from_pot_bin(work_dir, wscrn) {
        return vtot_matches_generated_handoff(vtot, &generated);
    }
    vtot_matches_wscrn_handoff(vtot, wscrn)
}

fn vtot_matches_generated_handoff(actual: &VtotDatData, expected: &VtotDatData) -> bool {
    actual.row_count() == expected.row_count()
        && columns_match(
            &actual.radius_bohr,
            &expected.radius_bohr,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
        && columns_match(
            &actual.screened_core_hole_potential,
            &expected.screened_core_hole_potential,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
}

fn vtot_matches_wscrn_handoff(vtot: &VtotDatData, wscrn: &WscrnDatData) -> bool {
    vtot.row_count() == wscrn.row_count()
        && columns_match(
            &vtot.radius_bohr,
            &wscrn.radius_bohr,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
        && columns_match(
            &vtot.screened_core_hole_potential,
            &wscrn.screened_potential,
            SCREEN_VTOT_HANDOFF_TOLERANCE,
        )
}

fn columns_match(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| {
                let allowed = tolerance * expected.abs().max(1.0);
                actual.is_finite() && (*actual - *expected).abs() <= allowed
            })
}

fn optional_vtot_needs_generation_from_pot_bin(work_dir: &Path, wscrn: &WscrnDatData) -> bool {
    let vtot_path = work_dir.join("vtot.dat");
    if vtot_path.is_file() {
        match read_vtot_dat(&vtot_path) {
            Ok(vtot) if vtot_matches_current_handoff(work_dir, wscrn, &vtot) => return false,
            Ok(_) | Err(_) => {}
        }
    }
    can_generate_optional_vtot_from_pot_bin(work_dir, wscrn)
}

fn write_cached_output(path: &Path, data: &WscrnDatData) -> Result<()> {
    write_wscrn_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn has_recoverable_screen_output(work_dir: &Path) -> bool {
    work_dir.join("vtot.dat").is_file() && work_dir.join("apot.bin").is_file()
}

fn write_generated_wscrn_from_vtot_and_apot(work_dir: &Path) -> Result<usize> {
    if can_use_cached_wscrn_output(work_dir) || !has_recoverable_screen_output(work_dir) {
        return Ok(0);
    }

    let path = work_dir.join("wscrn.dat");
    let data = generated_wscrn_from_vtot_and_apot(work_dir)?;
    let row_count = data.row_count();
    write_cached_output(&path, &data)?;
    Ok(row_count)
}

fn generated_wscrn_from_vtot_and_apot(work_dir: &Path) -> Result<WscrnDatData> {
    let vtot_path = work_dir.join("vtot.dat");
    let vtot = read_vtot_dat(&vtot_path)
        .with_context(|| format!("failed to read {}", vtot_path.display()))?;
    let row_count = vtot.row_count();
    let mut data = WscrnDatData {
        header_lines: vec!["# r       w_scrn(r)      v_ch(r)".to_string()],
        radius_bohr: vtot.radius_bohr,
        screened_potential: vtot.screened_core_hole_potential,
        core_hole_potential: Array1::zeros(row_count),
    };
    refresh_wscrn_core_hole_from_apot(work_dir, &mut data)?;
    Ok(data)
}

fn write_optional_vtot_cache(work_dir: &Path, wscrn: &WscrnDatData) -> Result<usize> {
    let path = work_dir.join("vtot.dat");
    if !path.is_file() {
        return write_generated_vtot_from_pot_bin(work_dir, wscrn);
    }
    let data = match read_vtot_dat(&path) {
        Ok(data) if vtot_matches_current_handoff(work_dir, wscrn, &data) => data,
        Ok(_) => {
            if can_generate_optional_vtot_from_pot_bin(work_dir, wscrn) {
                return write_generated_vtot_from_pot_bin(work_dir, wscrn);
            }
            bail!("SCREEN vtot.dat sidecar does not match wscrn.dat");
        }
        Err(error) => {
            if can_generate_optional_vtot_from_pot_bin(work_dir, wscrn) {
                return write_generated_vtot_from_pot_bin(work_dir, wscrn);
            }
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    };
    let row_count = data.row_count();
    write_vtot_cache(&path, &data)?;
    Ok(row_count)
}

fn write_vtot_cache(path: &Path, data: &VtotDatData) -> Result<()> {
    write_vtot_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_generated_vtot_from_pot_bin(work_dir: &Path, wscrn: &WscrnDatData) -> Result<usize> {
    let pot_path = work_dir.join("pot.bin");
    if !pot_path.is_file() {
        return Ok(0);
    }

    let path = work_dir.join("vtot.dat");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let data = vtot_dat_from_wscrn_and_pot_bin(wscrn, &pot)
        .with_context(|| format!("failed to generate {}", path.display()))?;
    let row_count = data.row_count();
    write_vtot_cache(&path, &data)?;
    Ok(row_count)
}

fn refresh_wscrn_core_hole_from_apot(work_dir: &Path, wscrn: &mut WscrnDatData) -> Result<()> {
    let path = work_dir.join("apot.bin");
    if !path.is_file() {
        return Ok(());
    }

    let apot =
        read_apot_bin(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let core_hole = apot_core_hole_columns(&apot).context("SCREEN apot.bin core-hole payload")?;
    let row_count = wscrn.row_count();
    ensure!(
        row_count <= SCREEN_RADIAL_POINTS,
        "SCREEN wscrn.dat has {row_count} rows, expected at most {SCREEN_RADIAL_POINTS}"
    );
    ensure!(
        wscrn.core_hole_potential.len() == row_count,
        "SCREEN wscrn.dat core-hole potential has {} rows, expected {row_count}",
        wscrn.core_hole_potential.len()
    );

    let radii = screen_radial_grid(
        SCREEN_RADIAL_GRID_STEP,
        SCREEN_RADIAL_GRID_ORIGIN,
        row_count,
    )?;
    let expected = screen_bare_core_hole_potential(
        radii
            .as_slice()
            .context("SCREEN radial grid storage is not contiguous")?,
        core_hole
            .large_component
            .as_slice()
            .context("dgc0 storage is not contiguous")?,
        core_hole
            .small_component
            .as_slice()
            .context("dpc0 storage is not contiguous")?,
        SCREEN_RADIAL_GRID_STEP,
        row_count,
    )?;

    for (row, (actual, expected)) in wscrn
        .core_hole_potential
        .iter_mut()
        .zip(expected.iter())
        .enumerate()
    {
        let allowed = SCREEN_CORE_HOLE_TOLERANCE * expected.abs().max(1.0);
        let difference = (*actual - expected).abs();
        if !actual.is_finite() || difference > allowed {
            *actual = *expected;
        }
        ensure!(
            actual.is_finite(),
            "SCREEN wscrn.dat core-hole potential row {row} is non-finite"
        );
    }
    Ok(())
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
}

fn prepare_module_log_cache(path: &Path) -> Result<()> {
    if path.is_file() {
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn write_module_log(path: &Path, data: &ModuleLogData) -> Result<()> {
    write_module_log_dat(path, data).with_context(|| format!("failed to write {}", path.display()))
}

fn write_or_generate_module_log(path: &Path) -> Result<()> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_generated_module_log(path)
}

fn write_or_recover_module_log(path: &Path, source_handoff_written: bool) -> Result<()> {
    if source_handoff_written && path.is_file() && read_module_log_dat(path).is_err() {
        return write_generated_module_log(path);
    }
    write_or_generate_module_log(path)
}

fn write_generated_module_log(path: &Path) -> Result<()> {
    write_module_log(path, &generated_screen_module_log())
}

fn generated_screen_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            "Calculating screened core-hole potential ...".to_string(),
            "Done with module: screened core-hole potential.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 2],
    }
}

#[cfg(all(test, feature = "full"))]
mod tests {
    use super::{
        SCREEN_SOURCE_REQUIREMENT_ERROR, has_cached_screen_output, has_completed_screen_output,
        has_supported_screen_source_handoff, has_supported_wscrn_handoff, run_for_input,
        run_in_dir, run_supported_wscrn_handoff_in_dir,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4, array};
    use num_complex::Complex64;
    use refeff_core::screen::{screen_bare_core_hole_potential, screen_radial_grid};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        ApotBinData, ApotBinPayload, ApotBinSection, ApotBinType, ApotBinValue,
        CONFIG_DAT_ORBITAL_COUNT, ConfigDatData, ConfigDatPotential, FeffDocument, FeffInput,
        FmsCluster, FmsControl, FmsDebye, FmsInput, GgDatData, GgDatSection, ModuleLogData,
        PhaseBinData, PhaseBinPotential, PhaseBinScalars, PotBinData, PotBinScalars, PotInput,
        ScreenInput, VtotDatData, WscrnDatData, fms_input_string, pot_input_string, rdinp,
        read_module_log_dat, read_vtot_dat, read_wscrn_dat, screen_input_string, write_apot_bin,
        write_config_dat, write_gg_bin, write_module_log_dat, write_phase_bin, write_pot_bin,
        write_vtot_dat, write_wscrn_dat,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn screen_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN should require cached output or source handoffs")?;

        assert!(error.to_string().contains(SCREEN_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn screen_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;

        assert!(!has_cached_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_does_not_claim_malformed_screen_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("screen.inp"), "not a screen.inp handoff\n")?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(!has_supported_screen_source_handoff(temp.path())?);
        assert!(!has_supported_wscrn_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed SCREEN input should fail through the explicit SCREEN runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("screen.inp"), "{chain}");
        assert!(!temp.path().join("wscrn.dat").exists());
        assert!(!temp.path().join("vtot.dat").exists());
        assert!(!temp.path().join("logscreen.dat").exists());
        Ok(())
    }

    #[test]
    fn screen_module_validates_phase_gg_source_bundle_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 2)?;
        write_phase_bin(
            temp.path().join("phase.bin"),
            &sample_screen_phase_bin(2, 1),
        )?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_screen_gg_bin(2, 4))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN source bundle should still require complete source handoffs")?;
        let chain = format!("{error:#}");

        assert!(chain.contains(SCREEN_SOURCE_REQUIREMENT_ERROR), "{chain}");
        assert!(
            !chain.contains("failed to build SCREEN FMS trace handoff"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn screen_module_rejects_bad_phase_gg_source_bundle_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 2)?;
        write_phase_bin(
            temp.path().join("phase.bin"),
            &sample_screen_phase_bin(2, 1),
        )?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_screen_gg_bin(2, 3))?;

        let error = run_in_dir(temp.path())
            .err()
            .context("bad SCREEN FMS source bundle should fail before source requirement")?;
        let chain = format!("{error:#}");

        assert!(
            chain.contains("failed to build SCREEN FMS trace handoff"),
            "{chain}"
        );
        assert!(chain.contains("required order 4"), "{chain}");
        assert!(!chain.contains(SCREEN_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn screen_module_validates_pot_bin_kernel_source_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;

        let error = run_in_dir(temp.path()).err().context(
            "SCREEN potential-kernel source should still require complete source handoffs",
        )?;
        let chain = format!("{error:#}");

        assert!(chain.contains(SCREEN_SOURCE_REQUIREMENT_ERROR), "{chain}");
        assert!(
            !chain.contains("failed to build SCREEN potential-kernel handoff"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn screen_module_generates_outputs_from_source_handoffs_without_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;
        write_phase_bin(
            temp.path().join("phase.bin"),
            &sample_screen_phase_bin(2, 0),
        )?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_screen_gg_bin(2, 1))?;

        assert!(has_supported_screen_source_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

        assert_eq!(count, wscrn.row_count() + vtot.row_count());
        assert_eq!(vtot.row_count(), wscrn.row_count());
        assert_eq!(vtot.screened_core_hole_potential, wscrn.screened_potential);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        assert!(has_cached_screen_output(temp.path())?);
        assert!(has_completed_screen_output(temp.path())?);
        assert!(!has_supported_screen_source_handoff(temp.path())?);
        assert!(
            wscrn
                .core_hole_potential
                .iter()
                .any(|value| value.abs() > 1.0e-12)
        );
        Ok(())
    }

    #[test]
    fn screen_module_regenerates_stale_wscrn_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;
        write_phase_bin(
            temp.path().join("phase.bin"),
            &sample_screen_phase_bin(2, 0),
        )?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_screen_gg_bin(2, 1))?;

        let source_count = run_in_dir(temp.path())?;
        let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let expected_vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;
        let mut stale = expected_wscrn.clone();
        stale.screened_potential[0] += 1.0;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &stale)?;

        assert!(!has_cached_screen_output(temp.path())?);
        let repaired_count = run_in_dir(temp.path())?;

        assert_eq!(repaired_count, source_count);
        assert_wscrn_close(
            &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            &expected_wscrn,
            1.0e-10,
        );
        assert_eq!(read_vtot_dat(temp.path().join("vtot.dat"))?, expected_vtot);
        assert!(has_cached_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_does_not_claim_cached_output_with_malformed_phase_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 1)?;
        let expected_wscrn = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &expected_wscrn)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;
        std::fs::write(temp.path().join("phase.bin"), "not a phase.bin source\n")?;
        write_gg_bin(temp.path().join("gg.bin"), &sample_screen_gg_bin(2, 1))?;

        assert!(!has_cached_screen_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed SCREEN phase source should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("phase.bin"), "{chain}");
        assert_eq!(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn
        );
        assert!(!temp.path().join("logscreen.dat").exists());
        Ok(())
    }

    #[test]
    fn screen_module_generates_outputs_from_inline_fms_source_without_gg_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 1)?;
        write_inline_screen_fms_input_with_lmax(temp.path(), 1)?;
        write_single_atom_geom_dat(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;
        write_phase_bin(
            temp.path().join("phase.bin"),
            &sample_screen_phase_bin(2, 1),
        )?;

        let count = run_in_dir(temp.path())?;
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

        assert_eq!(count, wscrn.row_count() + vtot.row_count());
        assert_eq!(vtot.row_count(), wscrn.row_count());
        assert!(!temp.path().join("gg.bin").is_file());
        assert!(has_completed_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_generates_inline_fms_source_without_phase_or_gg_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 1)?;
        write_inline_screen_fms_input_with_lmax_and_idwopt(temp.path(), 1, -1)?;
        write_single_atom_geom_dat(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;

        let count = run_in_dir(temp.path())?;
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

        assert_eq!(count, wscrn.row_count() + vtot.row_count());
        assert_eq!(vtot.row_count(), wscrn.row_count());
        assert!(!temp.path().join("phase.bin").is_file());
        assert!(!temp.path().join("gg.bin").is_file());
        assert!(has_completed_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_uses_pot_debye_metadata_without_phase_cache() -> Result<()> {
        for idwopt in [0, 3] {
            let temp = tempfile::tempdir()?;
            write_screen_input_with_maxl(temp.path(), 1)?;
            write_inline_screen_fms_input_with_lmax_idwopt_and_potential_count(
                temp.path(),
                1,
                idwopt,
                2,
            )?;
            write_two_atom_geom_dat(temp.path())?;
            write_pot_bin(
                temp.path().join("pot.bin"),
                &sample_source_pot_bin_with_potential_count(2),
            )?;
            write_config_dat(
                temp.path().join("config.dat"),
                &sample_source_config_dat_with_potential_count(2),
            )?;

            let count = run_in_dir(temp.path())?;
            let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
            let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

            assert_eq!(count, wscrn.row_count() + vtot.row_count());
            assert!(!temp.path().join("phase.bin").is_file());
            assert!(!temp.path().join("gg.bin").is_file());
            assert!(has_completed_screen_output(temp.path())?);
        }
        Ok(())
    }

    #[test]
    fn screen_module_retains_phase_requirement_for_spring_damping() -> Result<()> {
        for idwopt in [1, 2] {
            let temp = tempfile::tempdir()?;
            write_screen_input_with_maxl(temp.path(), 1)?;
            write_inline_screen_fms_input_with_lmax_and_idwopt(temp.path(), 1, idwopt)?;
            write_single_atom_geom_dat(temp.path())?;
            write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
            write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;

            let error = run_in_dir(temp.path())
                .err()
                .context("spring damping without phase.bin must fail closed")?;
            let chain = format!("{error:#}");

            assert!(chain.contains("phase.bin metadata"), "{chain}");
            assert!(!temp.path().join("wscrn.dat").exists());
        }
        Ok(())
    }

    #[test]
    fn screen_module_caps_inline_fms_trace_to_fms_lmax_without_phase_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input_with_maxl(temp.path(), 2)?;
        write_inline_screen_fms_input_with_lmax_and_idwopt(temp.path(), 0, -1)?;
        write_single_atom_geom_dat(temp.path())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_source_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_source_config_dat())?;

        let count = run_in_dir(temp.path())?;
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

        assert_eq!(count, wscrn.row_count() + vtot.row_count());
        assert_eq!(vtot.row_count(), wscrn.row_count());
        assert!(!temp.path().join("phase.bin").is_file());
        assert!(!temp.path().join("gg.bin").is_file());
        assert!(has_completed_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_rejects_bad_pot_bin_kernel_source_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let mut pot = sample_pot_bin();
        pot.muffin_tin_radii[0] = 1.0e-300;
        write_pot_bin(temp.path().join("pot.bin"), &pot)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("bad SCREEN potential-kernel source should fail before source requirement")?;
        let chain = format!("{error:#}");

        assert!(
            chain.contains("failed to build SCREEN potential-kernel handoff"),
            "{chain}"
        );
        assert!(chain.contains("radial bounds setup failed"), "{chain}");
        assert!(!chain.contains(SCREEN_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn screen_module_skips_non_rpa_corehole_pot_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 0)?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_screen_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn screen_module_skips_crpa_wscrn_sidecar_without_pot_screen_request() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_crpa_input(temp.path(), true)?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_screen_output(temp.path())?);
        assert!(!temp.path().join("logscreen.dat").exists());
        Ok(())
    }

    #[test]
    fn screen_module_roundtrips_cached_wscrn_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let expected = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_wscrn_dat(temp.path().join("wscrn.dat"))?, expected);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn screen_module_generates_missing_module_log_from_cached_wscrn_output() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn screen_module_preserves_cached_vtot_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let wscrn = sample_wscrn_dat();
        let vtot = sample_vtot_dat();
        let log = sample_module_log();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &vtot)?;
        write_module_log_dat(temp.path().join("logscreen.dat"), &log)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(read_wscrn_dat(temp.path().join("wscrn.dat"))?, wscrn);
        assert_eq!(read_vtot_dat(temp.path().join("vtot.dat"))?, vtot);
        assert_eq!(read_module_log_dat(temp.path().join("logscreen.dat"))?, log);
        Ok(())
    }

    #[test]
    fn screen_module_generates_missing_vtot_sidecar_from_pot_bin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let wscrn = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_vtot_dat(temp.path().join("vtot.dat"))?,
            sample_generated_vtot_dat()
        );
        Ok(())
    }

    #[test]
    fn screen_module_recovers_malformed_vtot_sidecar_from_pot_bin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let wscrn = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("vtot.dat"), "not a vtot.dat table\n")?;

        assert!(has_cached_screen_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_vtot_dat(temp.path().join("vtot.dat"))?,
            sample_generated_vtot_dat()
        );
        Ok(())
    }

    #[test]
    fn screen_module_recovers_stale_vtot_sidecar_from_pot_bin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let wscrn = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        let mut stale = sample_generated_vtot_dat();
        stale.screened_core_hole_potential[0] += 1.0;
        write_vtot_dat(temp.path().join("vtot.dat"), &stale)?;

        assert!(has_cached_screen_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_vtot_dat(temp.path().join("vtot.dat"))?,
            sample_generated_vtot_dat()
        );
        Ok(())
    }

    #[test]
    fn screen_module_recovers_malformed_module_log_for_recoverable_vtot_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let wscrn = sample_wscrn_dat();
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("vtot.dat"), "not a vtot.dat table\n")?;
        std::fs::write(temp.path().join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_screen_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        assert_eq!(
            read_vtot_dat(temp.path().join("vtot.dat"))?,
            sample_generated_vtot_dat()
        );
        Ok(())
    }

    #[test]
    fn screen_module_generates_missing_wscrn_from_vtot_and_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 2)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, 3)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, 6);
        assert_column_close(&actual.radius_bohr, &sample_vtot_dat().radius_bohr, 1.0e-12);
        assert_column_close(
            &actual.screened_potential,
            &sample_vtot_dat().screened_core_hole_potential,
            1.0e-12,
        );
        assert_column_close(&actual.core_hole_potential, &expected_core_hole, 1.0e-10);
        Ok(())
    }

    #[test]
    fn screen_module_recovers_malformed_wscrn_from_vtot_and_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 2)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, 3)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;
        std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat table\n")?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, 6);
        assert_column_close(&actual.radius_bohr, &sample_vtot_dat().radius_bohr, 1.0e-12);
        assert_column_close(
            &actual.screened_potential,
            &sample_vtot_dat().screened_core_hole_potential,
            1.0e-12,
        );
        assert_column_close(&actual.core_hole_potential, &expected_core_hole, 1.0e-10);
        Ok(())
    }

    #[test]
    fn screen_module_regenerates_stale_wscrn_from_vtot_and_apot_before_pot_vtot_recovery()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 2)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, 3)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;
        let mut stale = sample_wscrn_dat();
        stale.screened_potential[0] += 1.0;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &stale)?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, 6);
        assert_column_close(&actual.radius_bohr, &sample_vtot_dat().radius_bohr, 1.0e-12);
        assert_column_close(
            &actual.screened_potential,
            &sample_vtot_dat().screened_core_hole_potential,
            1.0e-12,
        );
        assert_column_close(&actual.core_hole_potential, &expected_core_hole, 1.0e-10);
        assert_eq!(
            read_vtot_dat(temp.path().join("vtot.dat"))?,
            sample_vtot_dat()
        );
        Ok(())
    }

    #[test]
    fn screen_module_advertises_standalone_recoverable_wscrn_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, 3)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_supported_wscrn_handoff_in_dir(temp.path())?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, 3);
        assert_column_close(&actual.core_hole_potential, &expected_core_hole, 1.0e-10);
        assert!(!temp.path().join("logscreen.dat").exists());
        Ok(())
    }

    #[test]
    fn screen_module_input_runner_recovers_wscrn_handoff_without_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        let expected_core_hole =
            expected_core_hole_potential(&large_component, &small_component, 3)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        let count = run_for_input(&temp.path().join("feff.inp"))?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, 3);
        assert_column_close(&actual.core_hole_potential, &expected_core_hole, 1.0e-10);
        assert!(!temp.path().join("logscreen.dat").exists());
        Ok(())
    }

    #[test]
    fn screen_module_wscrn_handoff_does_not_mark_screen_complete() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(!has_completed_screen_output(temp.path())?);
        let handoff_count = run_supported_wscrn_handoff_in_dir(temp.path())?;

        assert_eq!(handoff_count, 3);
        assert!(has_cached_screen_output(temp.path())?);
        assert!(!has_completed_screen_output(temp.path())?);

        let screen_count = run_in_dir(temp.path())?;

        assert_eq!(screen_count, 6);
        assert!(has_completed_screen_output(temp.path())?);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn screen_module_does_not_advertise_malformed_wscrn_without_recoverable_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        std::fs::write(temp.path().join("wscrn.dat"), "not a wscrn.dat table\n")?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(!has_supported_wscrn_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN should reject malformed wscrn.dat without recovery handoffs")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("wscrn.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn screen_module_does_not_advertise_malformed_recoverable_wscrn_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 2)?;
        std::fs::write(temp.path().join("vtot.dat"), "not a vtot.dat table\n")?;
        let (large_component, small_component) = sample_core_hole_components();
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        assert!(!has_cached_screen_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN should reject malformed recoverable vtot.dat handoff")?;
        let chain = format!("{error:?}");

        assert!(
            chain.contains("failed to generate SCREEN wscrn.dat from vtot.dat and apot.bin"),
            "{chain}"
        );
        assert!(chain.contains("vtot.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn screen_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &sample_wscrn_dat())?;
        std::fs::write(temp.path().join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_screen_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("SCREEN should reject malformed cached logscreen.dat")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logscreen.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn screen_module_recovers_malformed_module_log_for_recoverable_wscrn_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        write_pot_input(temp.path(), 2)?;
        write_vtot_dat(temp.path().join("vtot.dat"), &sample_vtot_dat())?;
        let (large_component, small_component) = sample_core_hole_components();
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;
        std::fs::write(temp.path().join("logscreen.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_screen_output(temp.path())?);
        assert!(has_supported_wscrn_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 6);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            sample_module_log()
        );
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        assert_column_close(&actual.radius_bohr, &sample_vtot_dat().radius_bohr, 1.0e-12);
        assert_column_close(
            &actual.screened_potential,
            &sample_vtot_dat().screened_core_hole_potential,
            1.0e-12,
        );
        assert!(
            actual
                .core_hole_potential
                .iter()
                .any(|value| value.abs() > 1.0e-12)
        );
        Ok(())
    }

    #[test]
    fn screen_module_regenerates_cached_bare_core_hole_from_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_screen_input(temp.path())?;
        let mut wscrn = sample_wscrn_dat();
        wscrn.core_hole_potential.fill(0.0);
        let (large_component, small_component) = sample_core_hole_components();
        let expected =
            expected_core_hole_potential(&large_component, &small_component, wscrn.row_count())?;
        write_wscrn_dat(temp.path().join("wscrn.dat"), &wscrn)?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &sample_apot_bin(&large_component, &small_component),
        )?;

        let count = run_in_dir(temp.path())?;
        let actual = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        assert_eq!(count, wscrn.row_count());
        assert_column_close(
            &actual.core_hole_potential,
            &expected,
            super::SCREEN_CORE_HOLE_TOLERANCE,
        );
        Ok(())
    }

    #[test]
    fn screen_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_screen_dir()? else {
            crate::require_fixture!(
                "SCREEN reference test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "screen.inp",
            "pot.inp",
            "wscrn.dat",
            "vtot.dat",
            "logscreen.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let expected_vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;
        let expected_log = read_module_log_dat(temp.path().join("logscreen.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(
            count,
            expected_wscrn.row_count() + expected_vtot.row_count()
        );
        assert!(has_cached_screen_output(temp.path())?);
        assert_eq!(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn
        );
        assert_eq!(read_vtot_dat(temp.path().join("vtot.dat"))?, expected_vtot);
        assert_eq!(
            read_module_log_dat(temp.path().join("logscreen.dat"))?,
            expected_log
        );
        Ok(())
    }

    #[test]
    fn screen_module_matches_no_cache_inline_fms_generated_reference_when_present() -> Result<()> {
        let reference_dirs = reference_screen_source_dirs()?;
        if reference_dirs.is_empty() {
            crate::require_fixture!(
                "SCREEN inline-FMS reference test; generated source bundles not found"
            );
        }

        for (label, reference_dir) in reference_dirs {
            assert_no_cache_inline_fms_screen_reference(label, &reference_dir)
                .with_context(|| format!("SCREEN source fixture {label} failed"))?;
        }
        Ok(())
    }

    #[test]
    fn screen_module_matches_graphite_reference_zip_without_phase_or_gg_cache() -> Result<()> {
        let Some(zip_path) = reference_graphite_zip()? else {
            crate::require_fixture!(
                "SCREEN Graphite source reference test; KSPACE/Graphite REFERENCE.zip not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!(
                "SCREEN Graphite source reference test; unzip command not found"
            );
        }

        let temp = tempfile::tempdir()?;
        for entry in ["pot.bin", "config.dat", "geom.dat", "fms.inp", "apot.bin"] {
            std::fs::write(
                temp.path().join(entry),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{entry}"))?,
            )?;
        }
        let mut screen_input = unzip_reference_entry(&zip_path, "REFERENCE/screen.inp")?;
        if !String::from_utf8_lossy(&screen_input).contains("icore") {
            screen_input.extend_from_slice(b" icore          -1\n");
        }
        std::fs::write(temp.path().join("screen.inp"), screen_input)?;

        let expected = tempfile::tempdir()?;
        std::fs::write(
            expected.path().join("wscrn.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/wscrn.dat")?,
        )?;
        std::fs::write(
            expected.path().join("vtot.dat"),
            unzip_reference_entry(&zip_path, "REFERENCE/vtot.dat")?,
        )?;
        let expected_wscrn = read_wscrn_dat(expected.path().join("wscrn.dat"))?;
        let expected_vtot = read_vtot_dat(expected.path().join("vtot.dat"))?;

        let count = run_in_dir(temp.path())?;

        let actual_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let actual_vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;
        assert_eq!(count, actual_wscrn.row_count() + actual_vtot.row_count());
        assert!(!temp.path().join("phase.bin").is_file());
        assert!(!temp.path().join("gg.bin").is_file());
        assert!(
            has_completed_screen_output(temp.path())?,
            "SCREEN Graphite source fixture did not write a completed module log"
        );
        assert_wscrn_close(&actual_wscrn, &expected_wscrn, 1.0e-4);
        assert_vtot_close(&actual_vtot, &expected_vtot, 1.0e-4);
        Ok(())
    }

    fn assert_no_cache_inline_fms_screen_reference(
        label: &str,
        reference_dir: &Path,
    ) -> Result<()> {
        let expected_wscrn = read_wscrn_dat(reference_dir.join("wscrn.dat"))
            .with_context(|| format!("failed to read {label} wscrn.dat reference"))?;
        let expected_vtot = read_vtot_dat(reference_dir.join("vtot.dat"))
            .with_context(|| format!("failed to read {label} vtot.dat reference"))?;

        let temp = tempfile::tempdir()?;
        copy_screen_source_fixture(reference_dir, temp.path())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(
            count,
            expected_wscrn.row_count() + expected_vtot.row_count(),
            "row count mismatch for {label}"
        );
        assert!(!temp.path().join("gg.bin").is_file());
        assert!(!temp.path().join("gg.dat").is_file());
        assert!(
            has_completed_screen_output(temp.path())?,
            "SCREEN source fixture {label} did not write a completed module log"
        );
        assert_wscrn_close(
            &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            &expected_wscrn,
            1.0e-4,
        );
        assert_vtot_close(
            &read_vtot_dat(temp.path().join("vtot.dat"))?,
            &expected_vtot,
            1.0e-4,
        );
        Ok(())
    }

    fn copy_screen_source_fixture(reference_dir: &Path, work_dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(reference_dir)? {
            let entry = entry?;
            let source = entry.path();
            if !source.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                "wscrn.dat" | "vtot.dat" | "logscreen.dat" | "logscrn.dat" | "gg.bin" | "gg.dat"
            ) {
                continue;
            }
            std::fs::copy(&source, work_dir.join(name.as_ref()))?;
        }
        Ok(())
    }

    #[test]
    fn screen_module_validates_reference_bare_core_hole_from_apot_when_present() -> Result<()> {
        let Some(reference_dir) = reference_screen_with_apot_dir()? else {
            crate::require_fixture!(
                "SCREEN apot reference test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["screen.inp", "pot.inp", "wscrn.dat", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected_wscrn.row_count());
        assert_eq!(
            read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            expected_wscrn
        );
        Ok(())
    }

    #[test]
    fn screen_module_generates_reference_wscrn_from_vtot_and_apot_when_present() -> Result<()> {
        let Some(reference_dir) = reference_screen_with_apot_dir()? else {
            crate::require_fixture!(
                "SCREEN wscrn generation test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["screen.inp", "pot.inp", "vtot.dat", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_wscrn = read_wscrn_dat(reference_dir.join("wscrn.dat"))?;
        let vtot = read_vtot_dat(temp.path().join("vtot.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, expected_wscrn.row_count() + vtot.row_count());
        assert_wscrn_close(
            &read_wscrn_dat(temp.path().join("wscrn.dat"))?,
            &expected_wscrn,
            1.0e-4,
        );
        Ok(())
    }

    #[test]
    fn screen_module_generates_reference_vtot_from_wscrn_and_pot_bin_when_present() -> Result<()> {
        let Some(reference_dir) = reference_screen_with_pot_dir()? else {
            crate::require_fixture!(
                "SCREEN vtot generation test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["screen.inp", "pot.inp", "wscrn.dat", "pot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let wscrn = read_wscrn_dat(temp.path().join("wscrn.dat"))?;
        let expected_vtot = read_vtot_dat(reference_dir.join("vtot.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, wscrn.row_count() + expected_vtot.row_count());
        assert_vtot_close(
            &read_vtot_dat(temp.path().join("vtot.dat"))?,
            &expected_vtot,
            1.0e-4,
        );
        Ok(())
    }

    fn write_screen_input(work_dir: &Path) -> Result<()> {
        std::fs::write(work_dir.join("screen.inp"), rdinp::screen_inp_string())?;
        Ok(())
    }

    fn write_screen_input_with_maxl(work_dir: &Path, maxl: i32) -> Result<()> {
        let input = ScreenInput {
            maxl,
            ..ScreenInput::default()
        };
        std::fs::write(work_dir.join("screen.inp"), screen_input_string(&input)?)?;
        Ok(())
    }

    fn write_inline_screen_fms_input_with_lmax(work_dir: &Path, lmax: i32) -> Result<()> {
        write_inline_screen_fms_input_with_lmax_and_idwopt(work_dir, lmax, 0)
    }

    fn write_inline_screen_fms_input_with_lmax_and_idwopt(
        work_dir: &Path,
        lmax: i32,
        idwopt: i32,
    ) -> Result<()> {
        write_inline_screen_fms_input_with_lmax_idwopt_and_potential_count(
            work_dir, lmax, idwopt, 1,
        )
    }

    fn write_inline_screen_fms_input_with_lmax_idwopt_and_potential_count(
        work_dir: &Path,
        lmax: i32,
        idwopt: i32,
        potential_count: usize,
    ) -> Result<()> {
        let input = FmsInput {
            control: FmsControl {
                mfms: 1,
                idwopt,
                minv: 0,
            },
            cluster: FmsCluster {
                rfms2: 1.0,
                rdirec: 2.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: FmsDebye {
                tk: 0.0,
                thetad: 400.0,
                sig2g: 0.0,
            },
            lmaxph: vec![lmax; potential_count],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 1,
        };
        std::fs::write(work_dir.join("fms.inp"), fms_input_string(&input)?)?;
        Ok(())
    }

    fn write_single_atom_geom_dat(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("geom.dat"),
            concat!(
                "nat, nph =     1    0\n",
                "       1\n",
                " iat     x       y        z       iph  \n",
                " -----------------------------------------------------------------------\n",
                "   1      0.00000      0.00000      0.00000   0   0\n",
            ),
        )?;
        Ok(())
    }

    fn write_two_atom_geom_dat(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("geom.dat"),
            concat!(
                "nat, nph =     2    1\n",
                "       1       2\n",
                " iat     x       y        z       iph  \n",
                " -----------------------------------------------------------------------\n",
                "   1      0.00000      0.00000      0.00000   0   0\n",
                "   2      1.80000      0.00000      0.00000   1   0\n",
            ),
        )?;
        Ok(())
    }

    fn write_pot_input(work_dir: &Path, nohole: i32) -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu screen smoke test
EDGE K
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut pot_input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        pot_input.run.nohole = nohole;
        std::fs::write(work_dir.join("pot.inp"), pot_input_string(&pot_input)?)?;
        Ok(())
    }

    fn write_crpa_input(work_dir: &Path, enabled: bool) -> Result<()> {
        std::fs::write(
            work_dir.join("crpa.inp"),
            format!(
                concat!(" do_CRPA{:12}\n", " rcut{:21.16}     \n", " l_crpa{:12}\n",),
                i32::from(enabled),
                3.5,
                2,
            ),
        )?;
        Ok(())
    }

    fn sample_screen_phase_bin(energy_count: usize, lmax: usize) -> PhaseBinData {
        let l_count = 2 * lmax + 1;
        PhaseBinData {
            spin_count: 1,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 0,
            pad_width: 8,
            final_state_count: 8,
            transition_count: 8,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 2.0,
                fermi_level: 0.5,
                edge_energy: 1.0,
            },
            energy_grid: Array1::from_iter(
                (0..energy_count).map(|energy| Complex64::new(0.1 * energy as f64, 0.02)),
            ),
            reference_energy: Array2::zeros((energy_count, 1)),
            potentials: vec![PhaseBinPotential {
                lmax,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, l_count, 1),
                    |(energy, l_slot, _)| {
                        Complex64::new(
                            0.1 + energy as f64 * 0.01 + l_slot as f64 * 0.05,
                            -0.03 * l_slot as f64,
                        )
                    },
                ),
            }],
            transition_moments: Array4::zeros((energy_count, 1, 8, 1)),
            raw_pads: None,
        }
    }

    fn sample_screen_gg_bin(energy_count: usize, order: usize) -> GgDatData {
        GgDatData {
            sections: (0..energy_count)
                .map(|energy| GgDatSection {
                    section_number: energy + 1,
                    values: Array2::from_shape_fn((order, order), |(row, column)| {
                        let base = (energy + 1) as f64 * 10.0 + row as f64 + column as f64 * 0.25;
                        Complex64::new(base, -base * 0.1)
                    }),
                    raw_prefix_lines: None,
                })
                .collect(),
        }
    }

    fn sample_wscrn_dat() -> WscrnDatData {
        WscrnDatData {
            header_lines: vec![" # r       w_scrn(r)      v_ch(r)".to_string()],
            radius_bohr: array![
                0.150_733_046_3E-03,
                0.158_461_294_9E-03,
                0.166_585_779_2E-03
            ],
            screened_potential: array![
                0.267_288_234_6E+02,
                0.267_288_167_8E+02,
                0.267_288_030_6E+02
            ],
            core_hole_potential: array![
                0.291_616_524_4E+02,
                0.291_616_457_6E+02,
                0.291_616_320_4E+02
            ],
        }
    }

    fn sample_core_hole_components() -> (Array1<f64>, Array1<f64>) {
        let large_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            if row < 3 {
                0.45 + 0.05 * row as f64
            } else {
                0.0
            }
        });
        let small_component = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            if row < 3 {
                -0.06 - 0.01 * row as f64
            } else {
                0.0
            }
        });
        (large_component, small_component)
    }

    fn expected_core_hole_potential(
        large_component: &Array1<f64>,
        small_component: &Array1<f64>,
        row_count: usize,
    ) -> Result<Array1<f64>> {
        let radii = screen_radial_grid(
            super::SCREEN_RADIAL_GRID_STEP,
            super::SCREEN_RADIAL_GRID_ORIGIN,
            row_count,
        )?;
        Ok(screen_bare_core_hole_potential(
            radii
                .as_slice()
                .context("radial grid storage is not contiguous")?,
            large_component
                .as_slice()
                .context("large component storage is not contiguous")?,
            small_component
                .as_slice()
                .context("small component storage is not contiguous")?,
            super::SCREEN_RADIAL_GRID_STEP,
            row_count,
        )?)
    }

    fn sample_apot_bin(
        large_component: &Array1<f64>,
        small_component: &Array1<f64>,
    ) -> ApotBinData {
        ApotBinData {
            sections: vec![ApotBinSection {
                section_number: 5,
                headers: vec![
                    "dgc0   - upper component of core hole orbital".to_string(),
                    "dpc0   - lower component of core hole orbital".to_string(),
                    "drho   - core hole density.".to_string(),
                    "dvcoul - core hole coulomb potential.".to_string(),
                ],
                header_texts: vec![
                    " dgc0   - upper component of core hole orbital".to_string(),
                    " dpc0   - lower component of core hole orbital".to_string(),
                    " drho   - core hole density.".to_string(),
                    " dvcoul - core hole coulomb potential.".to_string(),
                ],
                column_labels: vec![
                    "dgc0".to_string(),
                    "dpc0".to_string(),
                    "drho".to_string(),
                    "dvcoul".to_string(),
                ],
                column_label_text: Some(
                    "            dgc0                 dpc0                 drho               dvcoul "
                        .to_string(),
                ),
                payload: ApotBinPayload::Records(refeff_io::ApotBinRecords {
                    column_types: vec![ApotBinType::Double; 4],
                    rows: (0..POT_BIN_RADIAL_POINTS)
                        .map(|row| {
                            vec![
                                ApotBinValue::Real(large_component[row]),
                                ApotBinValue::Real(small_component[row]),
                                ApotBinValue::Real(0.0),
                                ApotBinValue::Real(0.0),
                            ]
                        })
                        .collect(),
                }),
                trailing_headers: vec![],
                trailing_header_texts: vec![],
            }],
        }
    }

    fn sample_vtot_dat() -> VtotDatData {
        VtotDatData {
            header_lines: Vec::new(),
            radius_bohr: array![
                0.150_733_046_3E-03,
                0.158_461_294_9E-03,
                0.166_585_779_2E-03
            ],
            total_potential: array![
                -0.182_900_150_0E+06,
                -0.182_900_133_6E+06,
                -0.182_900_100_2E+06
            ],
            screened_core_hole_potential: array![
                0.267_288_234_6E+02,
                0.267_288_167_8E+02,
                0.267_288_030_6E+02
            ],
        }
    }

    fn sample_generated_vtot_dat() -> VtotDatData {
        let wscrn = sample_wscrn_dat();
        VtotDatData {
            header_lines: Vec::new(),
            radius_bohr: wscrn.radius_bohr,
            total_potential: array![-1000.0, -1000.25, -1000.5],
            screened_core_hole_potential: wscrn.screened_potential,
        }
    }

    fn sample_pot_bin() -> PotBinData {
        sample_pot_bin_with_potential_count(1)
    }

    fn sample_pot_bin_with_potential_count(potentials: usize) -> PotBinData {
        PotBinData {
            titles: vec!["SCREEN vtot smoke test".to_string()],
            pad_width: 8,
            nohole: 2,
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
            muffin_tin_indices: Array1::from_elem(potentials, 12),
            muffin_tin_radii: Array1::from_elem(potentials, 1.1),
            norman_indices: Array1::from_elem(potentials, 40),
            atomic_numbers: Array1::from_elem(potentials, 29),
            kappa: Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: Array1::from_elem(potentials, 2.1),
            overlap_factors: Array1::ones(potentials),
            max_overlap_factors: Array1::ones(potentials),
            potential_multiplicities: Array1::ones(potentials),
            ionization: Array1::zeros(potentials),
            initial_large_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            initial_small_component: Array1::zeros(POT_BIN_RADIAL_POINTS),
            large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            electron_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            coulomb_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            total_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -1000.0 - 0.25 * row as f64,
            ),
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

    fn sample_source_pot_bin() -> PotBinData {
        sample_source_pot_bin_with_potential_count(1)
    }

    fn sample_source_pot_bin_with_potential_count(potentials: usize) -> PotBinData {
        let mut pot = sample_pot_bin_with_potential_count(potentials);
        let radii = screen_radial_grid(
            super::SCREEN_RADIAL_GRID_STEP,
            super::SCREEN_RADIAL_GRID_ORIGIN,
            POT_BIN_RADIAL_POINTS,
        )
        .expect("sample SCREEN radial grid");
        pot.muffin_tin_radii.fill(radii[12]);
        pot.norman_radii.fill(radii[20]);
        pot.scalars.interstitial_potential = -1.2;
        pot.scalars.interstitial_density = 0.03;
        pot.kappa = Array1::from_iter(-20..=20);
        pot.initial_large_component =
            Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| 0.001 * (row + 1) as f64);
        pot.initial_small_component =
            Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| -0.0005 * (row + 1) as f64);
        pot.large_components = Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, _)| 0.0001 * (row + 1) as f64 + 0.01 * orbital as f64,
        );
        pot.small_components = Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, _)| -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64,
        );
        pot.large_coefficients = Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coefficient, orbital, _)| 0.01 * (coefficient + 1) as f64 + 0.001 * orbital as f64,
        );
        pot.small_coefficients = Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coefficient, orbital, _)| -0.01 * (coefficient + 1) as f64 - 0.001 * orbital as f64,
        );
        pot.electron_density =
            Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
                0.01 * (row + 1) as f64
            });
        pot.valence_density =
            Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
                0.005 * (row + 1) as f64
            });
        pot.orbital_energies =
            Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| -10.0 + orbital as f64 * 0.25);
        pot
    }

    fn sample_source_config_dat() -> ConfigDatData {
        sample_source_config_dat_with_potential_count(1)
    }

    fn sample_source_config_dat_with_potential_count(potential_count: usize) -> ConfigDatData {
        ConfigDatData {
            header_lines: Vec::new(),
            potentials: (0..potential_count)
                .map(|potential_index| {
                    let mut occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
                    occupations[0] = 2.0;
                    ConfigDatPotential {
                        potential_index: i32::try_from(potential_index)
                            .expect("sample potential index fits in i32"),
                        atomic_number: 29,
                        element: "Cu".to_string(),
                        occupations,
                        valence_occupations: Array1::zeros(CONFIG_DAT_ORBITAL_COUNT),
                        spin_occupations: None,
                    }
                })
                .collect(),
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating screened core-hole potential ...".to_string(),
                "Done with module: screened core-hole potential.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn reference_screen_dir() -> Result<Option<PathBuf>> {
        let workspace = reference_workspace()?;
        let path = workspace.join("reference-work/golden/XANES/Cu");
        Ok((path.join("screen.inp").is_file()
            && path.join("pot.inp").is_file()
            && path.join("wscrn.dat").is_file()
            && path.join("vtot.dat").is_file()
            && path.join("logscreen.dat").is_file())
        .then_some(path))
    }

    fn reference_screen_with_pot_dir() -> Result<Option<PathBuf>> {
        let Some(path) = reference_screen_dir()? else {
            return Ok(None);
        };
        Ok(path.join("pot.bin").is_file().then_some(path))
    }

    fn reference_screen_with_apot_dir() -> Result<Option<PathBuf>> {
        let Some(path) = reference_screen_dir()? else {
            return Ok(None);
        };
        Ok(path.join("apot.bin").is_file().then_some(path))
    }

    fn reference_screen_source_dirs() -> Result<Vec<(&'static str, PathBuf)>> {
        let workspace = reference_workspace()?;
        Ok([
            ("DANES/Cu", "reference-work/golden/DANES/Cu"),
            ("ELNES/Cu", "reference-work/golden/ELNES/Cu"),
            (
                "LDOS/XANES_Cu_fms",
                "reference-work/golden/LDOS/XANES_Cu_fms",
            ),
            (
                "LDOS/XANES_Cu_spin_fms_short",
                "reference-work/golden/LDOS/XANES_Cu_spin_fms_short",
            ),
            (
                "LDOS/XANES_Cu_spin_no_fms",
                "reference-work/golden/LDOS/XANES_Cu_spin_no_fms",
            ),
            ("XANES/Cu", "reference-work/golden/XANES/Cu"),
        ]
        .into_iter()
        .filter_map(|(label, relative)| {
            let path = workspace.join(relative);
            has_screen_source_reference(&path).then_some((label, path))
        })
        .collect())
    }

    fn reference_graphite_zip() -> Result<Option<PathBuf>> {
        let workspace = reference_workspace()?;
        let path = workspace.join("reference-work/golden/KSPACE/Graphite/REFERENCE.zip");
        Ok(path.is_file().then_some(path))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }

    fn reference_workspace() -> Result<PathBuf> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .context("failed to find workspace root")
    }

    fn has_screen_source_reference(path: &Path) -> bool {
        [
            "screen.inp",
            "pot.inp",
            "pot.bin",
            "apot.bin",
            "config.dat",
            "phase.bin",
            "fms.inp",
            "geom.dat",
            "global.inp",
            "feff.inp",
            "wscrn.dat",
            "vtot.dat",
        ]
        .iter()
        .all(|name| path.join(name).is_file())
    }

    fn assert_wscrn_close(actual: &WscrnDatData, expected: &WscrnDatData, tolerance: f64) {
        assert_eq!(actual.row_count(), expected.row_count());
        assert_column_close(&actual.radius_bohr, &expected.radius_bohr, tolerance);
        assert_column_close(
            &actual.screened_potential,
            &expected.screened_potential,
            tolerance,
        );
        assert_column_close(
            &actual.core_hole_potential,
            &expected.core_hole_potential,
            tolerance,
        );
    }

    fn assert_vtot_close(actual: &VtotDatData, expected: &VtotDatData, tolerance: f64) {
        assert_eq!(actual.header_lines, expected.header_lines);
        assert_eq!(actual.row_count(), expected.row_count());
        assert_column_close(&actual.radius_bohr, &expected.radius_bohr, tolerance);
        assert_column_close(
            &actual.total_potential,
            &expected.total_potential,
            tolerance,
        );
        assert_column_close(
            &actual.screened_core_hole_potential,
            &expected.screened_core_hole_potential,
            tolerance,
        );
    }

    fn assert_column_close(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "column mismatch at row {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}
