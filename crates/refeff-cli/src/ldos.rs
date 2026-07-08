use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1};
use num_complex::Complex64;
use refeff_core::{
    FEFF_HARTREE_EV, LdosRholWavefunctionTablesInput, ldos_rhol_wavefunction_tables,
};
use refeff_io::{
    FmsInput, GeomDat, HubbardInput, LdosDatData, LdosDatFromFf2rhoInput, LdosElectronCount,
    LdosInput, LdosMagneticDatData, LdosMagneticDatFromFf2rhoInput, LdosSpinDatFromFf2rhoInput,
    ModuleLogData, PotBinData, PotInput, RhocDatData, XsphInput, gtr_bin_ldos_trace_handoff,
    hubbard_ldos_gtr_m_trace_handoff, ldos_dat_from_ff2rho, ldos_magnetic_dat_from_ff2rho,
    ldos_spin_dat_from_ff2rho, read_config_dat, read_gtr_bin, read_hubbard_ldos_gtr_bin_inferred,
    read_hubbard_ldos_gtr_m_bin_inferred, read_hubbard_ldos_gtr_off_bin, read_ldos_dat,
    read_lmdos_dat, read_module_log_dat, read_phase_bin, read_pot_bin, read_rhoc_dat,
    read_rhocm_dat, rhorrp_geom_handoff_from_geom_dat, write_ldos_dat, write_lmdos_dat,
    write_module_log_dat, write_rhoc_dat, write_rhocm_dat,
};

use crate::band::kmesh::{
    has_reciprocal_kmesh_source_handoff,
    has_supported_kmesh_handoff as has_supported_reciprocal_kmesh_handoff,
    prepare_optional_or_generated_kmesh, write_optional_or_generated_kmesh,
};
use crate::work_dir_for_input;

const LDOS_ORBITAL_COUNT: usize = 4;
const LDOS_SPIN_COUNT: usize = 2;
const LDOS_SPINPH_ZERO_TOLERANCE: f64 = 1.0e-12;
const LDOS_SOURCE_REQUIREMENT_ERROR: &str =
    "LDOS generation requires cached tables or complete radial/FMS source handoffs";

/// Run the supported FEFF LDOS cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_ldos_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_source_output_handoff(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_kmesh_handoff(work_dir)? {
        return run_supported_kmesh_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF LDOS run can be satisfied from existing `ldosNN.dat` caches.
pub(crate) fn has_cached_ldos_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("ldos.inp").is_file() {
        return Ok(false);
    }
    let tables = cached_output_paths(work_dir)?;
    if tables.is_empty() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.mldos != 1 {
        return Ok(false);
    }
    has_usable_ldos_cache_with_tables(work_dir, &tables, &input)
}

/// Whether a FEFF LDOS run can repair paired cached outputs before reporting
/// the LDOS stage complete.
pub(crate) fn has_recoverable_ldos_output(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("ldos.inp").is_file() {
        return Ok(false);
    }
    let tables = cached_output_paths(work_dir)?;
    if tables.is_empty() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.mldos != 1 {
        return Ok(false);
    }
    has_recoverable_ldos_cache_with_tables(work_dir, &tables, &input)
}

/// Whether a FEFF LDOS run can validate or generate the reciprocal `kmesh.dat`
/// source handoff before the remaining LDOS density solver is available.
pub(crate) fn has_supported_kmesh_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("ldos.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.mldos != 1 || has_usable_ldos_cache(work_dir, &input)? {
        return Ok(false);
    }
    has_supported_reciprocal_kmesh_handoff(work_dir)
}

/// Whether LDOS can produce final table outputs from complete source handoffs.
///
pub(crate) fn has_supported_source_output_handoff(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("ldos.inp").is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.mldos != 1 {
        return Ok(false);
    }

    let Ok(can_generate) = can_generate_ldos_from_wavefunction_source_handoffs(&input, work_dir)
    else {
        return Ok(false);
    };
    if !can_generate {
        return Ok(false);
    }

    match has_usable_ldos_cache(work_dir, &input) {
        Ok(cache_available) => Ok(!cache_available),
        Err(_) => Ok(true),
    }
}

/// Generate only the Rust-backed reciprocal `kmesh.dat` handoff for LDOS.
pub(crate) fn run_supported_kmesh_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    if !work_dir.join("ldos.inp").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if input.control.mldos != 1 || has_usable_ldos_cache(work_dir, &input)? {
        return Ok(0);
    }
    if !has_supported_reciprocal_kmesh_handoff(work_dir)? {
        return Ok(0);
    }
    let written = write_optional_or_generated_kmesh(work_dir, &work_dir.join("kmesh.dat"))?;
    let log_written =
        recover_existing_module_log_if_malformed(&work_dir.join("logdos.dat"), written > 0)?;
    Ok(written + log_written)
}

/// Run the FEFF LDOS path from existing or source-backed `ldosNN.dat` /
/// `rhocNN.dat` handoffs.
///
/// Supported non-spin handoffs can combine source `gtrNN.bin` traces with
/// source RHORRP wavefunction tables to generate final LDOS and
/// embedded-density tables; the no-FMS branch uses the same radial source with
/// a zero scattering trace. Missing or incomplete source state reports a normal
/// source requirement.
/// Cached FEFF directories are preserved by validating and re-rendering the
/// per-potential tables that downstream modules read, preserving or
/// regenerating the deterministic top-level `logdos.dat` wrapper.
/// Active Hubbard LDOS caches require matching magnetic-orbital sidecars
/// (`lmdosNN.dat` and `rhocmNN.dat`) for every cached potential until the
/// spin-Hubbard source generator can write them directly.
/// For no-FMS LDOS (`lfms2 == 0`), a missing `ldosNN.dat` can also be generated
/// from the matching embedded-density `rhocNN.dat` via the Rust `ff2rho` table
/// adapter with the scattering correction disabled.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if input.control.mldos != 1 {
        return Ok(0);
    }

    let kmesh_source_handoff = has_supported_reciprocal_kmesh_handoff(work_dir)?;
    let kmesh_source_available = has_reciprocal_kmesh_source_handoff(work_dir)?;
    let kmesh_count = write_optional_or_generated_kmesh(work_dir, &work_dir.join("kmesh.dat"))?;
    let mut tables = cached_output_paths(work_dir)?;
    let mut source_handoff_written = kmesh_source_handoff;
    let wavefunction_source_present = ldos_wavefunction_source_files_present(&input, work_dir);
    let gtr_count = if !has_ldos_table(&tables) && !wavefunction_source_present {
        crate::fms::write_ldos_gtr_bin_source_handoffs(work_dir, &input)?
    } else {
        0
    };
    if gtr_count > 0 {
        source_handoff_written = true;
    }
    if write_missing_or_unusable_ldos_from_rhoc_handoffs(&input, &tables)? > 0 {
        source_handoff_written = true;
        tables = cached_output_paths(work_dir)?;
    }
    if write_ldos_from_wavefunction_source_handoffs(&input, work_dir)? > 0 {
        source_handoff_written = true;
        tables = cached_output_paths(work_dir)?;
    }
    if kmesh_source_available && has_unrecoverable_ldos_table(&tables, &input) {
        bail!(LDOS_SOURCE_REQUIREMENT_ERROR);
    }
    if write_missing_or_unusable_rhoc_from_ldos_handoffs(&input, &tables)? > 0 {
        source_handoff_written = true;
        tables = cached_output_paths(work_dir)?;
    }
    let magnetic_repair_count =
        write_missing_or_unusable_magnetic_from_hubbard_handoffs(&input, work_dir, &tables)?;
    if magnetic_repair_count > 0 {
        source_handoff_written = true;
    }
    if !has_ldos_table(&tables) {
        bail!(LDOS_SOURCE_REQUIREMENT_ERROR);
    }
    if !active_hubbard_magnetic_outputs_complete(work_dir, &tables, &input, false)? {
        bail!(LDOS_SOURCE_REQUIREMENT_ERROR);
    }

    for table in &tables {
        write_cached_output(table)?;
    }
    let magnetic_table_count = write_cached_magnetic_outputs_if_active(work_dir)?;
    write_or_recover_module_log(&work_dir.join("logdos.dat"), source_handoff_written)?;
    Ok(tables.len() + magnetic_table_count + magnetic_repair_count + kmesh_count + gtr_count)
}

fn read_input(work_dir: &Path) -> Result<LdosInput> {
    let input_path = work_dir.join("ldos.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    LdosInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn write_cached_output(table: &CachedTable) -> Result<()> {
    match table.kind {
        CachedTableKind::Ldos => {
            let data = read_ldos_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_ldos_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
        CachedTableKind::Rhoc => {
            let data = read_rhoc_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_rhoc_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
    }
}

fn write_cached_magnetic_outputs_if_active(work_dir: &Path) -> Result<usize> {
    if !active_hubbard_ldos_enabled(work_dir)? {
        return Ok(0);
    }

    let tables = cached_magnetic_output_paths(work_dir)?;
    for table in &tables {
        write_cached_magnetic_output(table)?;
    }
    Ok(tables.len())
}

fn write_cached_magnetic_output(table: &CachedMagneticTable) -> Result<()> {
    match table.kind {
        CachedMagneticTableKind::Lmdos => {
            let data = read_lmdos_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_lmdos_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
        CachedMagneticTableKind::Rhocm => {
            let data = read_rhocm_dat(&table.path)
                .with_context(|| format!("failed to read {}", table.path.display()))?;
            write_rhocm_dat(&table.path, &data)
                .with_context(|| format!("failed to write {}", table.path.display()))
        }
    }
}

fn write_missing_or_unusable_ldos_from_rhoc_handoffs(
    input: &LdosInput,
    tables: &[CachedTable],
) -> Result<usize> {
    if input.control.lfms2 != 0 {
        return Ok(0);
    }

    let mut written = 0;
    for table in tables
        .iter()
        .filter(|table| table.kind == CachedTableKind::Rhoc)
    {
        let Some(output_path) = ldos_path_for_rhoc_table(&table.path) else {
            continue;
        };
        if ldos_table_is_usable(&output_path) {
            continue;
        }
        let rhoc = read_rhoc_dat(&table.path)
            .with_context(|| format!("failed to read {}", table.path.display()))?;
        let ldos = ldos_from_rhoc_without_scattering(&rhoc, input.control.ispin)
            .with_context(|| format!("failed to generate {}", output_path.display()))?;
        write_ldos_dat(&output_path, &ldos)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        written += 1;
    }
    Ok(written)
}

fn write_missing_or_unusable_rhoc_from_ldos_handoffs(
    input: &LdosInput,
    tables: &[CachedTable],
) -> Result<usize> {
    if input.control.lfms2 != 0 {
        return Ok(0);
    }

    let mut written = 0;
    for table in tables
        .iter()
        .filter(|table| table.kind == CachedTableKind::Ldos)
    {
        let Some(output_path) = rhoc_path_for_ldos_table(&table.path) else {
            continue;
        };
        if rhoc_table_is_usable(&output_path) {
            continue;
        }
        let ldos = read_ldos_dat(&table.path)
            .with_context(|| format!("failed to read {}", table.path.display()))?;
        let rhoc = rhoc_from_ldos_without_scattering(&ldos, input.control.ispin)
            .with_context(|| format!("failed to generate {}", output_path.display()))?;
        write_rhoc_dat(&output_path, &rhoc)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        written += 1;
    }
    Ok(written)
}

fn write_missing_or_unusable_magnetic_from_hubbard_handoffs(
    input: &LdosInput,
    work_dir: &Path,
    tables: &[CachedTable],
) -> Result<usize> {
    if input.control.lfms2 != 0 || !active_hubbard_ldos_enabled(work_dir)? {
        return Ok(0);
    }

    let mut written = 0;
    for index in cached_ldos_output_indices(tables) {
        let lmdos_path = work_dir.join(format!("lmdos{index}.dat"));
        let rhocm_path = work_dir.join(format!("rhocm{index}.dat"));
        let lmdos = read_lmdos_dat(&lmdos_path).ok();
        let rhocm = read_rhocm_dat(&rhocm_path).ok();

        match (lmdos, rhocm) {
            (None, Some(rhocm)) => {
                let generated = lmdos_from_rhocm_without_scattering(&rhocm)
                    .with_context(|| format!("failed to generate {}", lmdos_path.display()))?;
                write_lmdos_dat(&lmdos_path, &generated)
                    .with_context(|| format!("failed to write {}", lmdos_path.display()))?;
                written += 1;
            }
            (Some(lmdos), None) => {
                let generated = rhocm_from_lmdos_without_scattering(&lmdos)
                    .with_context(|| format!("failed to generate {}", rhocm_path.display()))?;
                write_rhocm_dat(&rhocm_path, &generated)
                    .with_context(|| format!("failed to write {}", rhocm_path.display()))?;
                written += 1;
            }
            (Some(_), Some(_)) | (None, None) => {}
        }
    }

    Ok(written)
}

fn write_ldos_from_wavefunction_source_handoffs(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<usize> {
    if !supported_ldos_wavefunction_source_controls(input)
        || !ldos_wavefunction_source_files_present(input, work_dir)
    {
        return Ok(0);
    }

    if input.control.lfms2 == 0 {
        return write_no_fms_ldos_from_rhol_source_handoffs(input, work_dir);
    }

    let Some(fms_input) = ldos_effective_fms_source_input(input, work_dir)? else {
        return Ok(0);
    };

    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let source =
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(work_dir, energy_grid.clone())?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < LDOS_ORBITAL_COUNT {
        return Ok(0);
    }
    let potential_count = source
        .phase
        .potential_count()
        .min(source.wavefunctions.wavefunctions.potential_count())
        .min(source.norman_radii_bohr.len())
        .min(input.lmaxph.len());
    let metadata_by_potential = read_ldos_source_metadata(work_dir, potential_count)?;
    let mut written = crate::fms::write_ldos_gtr_bin_source_grid_handoff(
        work_dir,
        &fms_input,
        source.phase.energies_hartree.view(),
        source.wavefunctions.wavefunctions.wave_numbers.view(),
        source.wavefunctions.wavefunctions.phase_shifts.view(),
    )?;

    for potential in 0..potential_count {
        let ldos_path = work_dir.join(format!("ldos{potential:02}.dat"));
        let rhoc_path = work_dir.join(format!("rhoc{potential:02}.dat"));
        let ldos_valid = ldos_table_is_usable(&ldos_path);
        let rhoc_valid = rhoc_table_is_usable(&rhoc_path);

        let metadata = metadata_by_potential
            .get(potential)
            .cloned()
            .unwrap_or_default();
        let Some(scattering_trace) =
            ldos_source_scattering_trace(input, work_dir, potential, source.phase.energy_count())?
        else {
            continue;
        };

        let solved = ldos_rhol_wavefunction_tables(LdosRholWavefunctionTablesInput {
            wavefunctions: &source.wavefunctions.wavefunctions,
            radii: source.wavefunctions.prepared.radii.view(),
            potential_index: potential,
            energy_grid_hartree: source.phase.energies_hartree.view(),
            scattering_trace: scattering_trace.view(),
            radial_step: source.wavefunctions.radial_dx,
            norman_radius: source.norman_radii_bohr[potential],
            angular_count: LDOS_ORBITAL_COUNT,
            apply_scattering: input.control.lfms2 != 0,
        })
        .with_context(|| {
            format!("failed to solve LDOS rhol source table for potential {potential}")
        })?;
        let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(source.phase.chemical_potential_hartree),
            charge_transfer: metadata.charge_transfer,
            electron_counts: &metadata.electron_counts,
            atom_count: metadata.atom_count,
            lorentzian_hwhh_hartree: source
                .phase
                .energies_hartree
                .first()
                .map(|energy| energy.im),
            energy_grid_hartree: source.phase.energies_hartree.view(),
            embedded_ldos: solved.density_grid.embedded_ldos.view(),
            scattering_ldos: solved.density_grid.scattering_ldos.view(),
            scattering_trace: scattering_trace.view(),
            apply_scattering: input.control.lfms2 != 0,
        })
        .with_context(|| format!("failed to build LDOS dat payload for potential {potential}"))?;

        if !ldos_valid || !ldos_table_matches_source_output(&ldos_path, &handoff.ldos) {
            write_ldos_dat(&ldos_path, &handoff.ldos)
                .with_context(|| format!("failed to write {}", ldos_path.display()))?;
            written += 1;
        }
        if !rhoc_valid || !rhoc_table_matches_source_output(&rhoc_path, &handoff.rhoc) {
            write_rhoc_dat(&rhoc_path, &handoff.rhoc)
                .with_context(|| format!("failed to write {}", rhoc_path.display()))?;
            written += 1;
        }
    }

    Ok(written)
}

fn write_no_fms_ldos_from_rhol_source_handoffs(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<usize> {
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let requested_potentials = (0..input.lmaxph.len()).collect::<Vec<_>>();

    let sources = crate::rhorrp::read_no_fms_ldos_rhol_source_tables(
        work_dir,
        energy_grid.clone(),
        &requested_potentials,
        LDOS_ORBITAL_COUNT,
    )?;
    let metadata_by_potential = read_ldos_source_metadata(work_dir, input.lmaxph.len())?;
    let mut written = 0;

    for source in sources {
        let potential = source.potential_index;
        let ldos_path = work_dir.join(format!("ldos{potential:02}.dat"));
        let rhoc_path = work_dir.join(format!("rhoc{potential:02}.dat"));
        let ldos_valid = ldos_table_is_usable(&ldos_path);
        let rhoc_valid = rhoc_table_is_usable(&rhoc_path);

        let metadata = metadata_by_potential
            .get(potential)
            .cloned()
            .unwrap_or_default();
        let scattering_trace = Array2::zeros((LDOS_ORBITAL_COUNT, energy_grid.len()));
        let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(source.chemical_potential_hartree),
            charge_transfer: metadata.charge_transfer,
            electron_counts: &metadata.electron_counts,
            atom_count: metadata.atom_count,
            lorentzian_hwhh_hartree: energy_grid.first().map(|energy| energy.im),
            energy_grid_hartree: energy_grid.view(),
            embedded_ldos: source.table.density_grid.embedded_ldos.view(),
            scattering_ldos: source.table.density_grid.scattering_ldos.view(),
            scattering_trace: scattering_trace.view(),
            apply_scattering: false,
        })
        .with_context(|| format!("failed to build LDOS dat payload for potential {potential}"))?;

        if !ldos_valid || !ldos_table_matches_source_output(&ldos_path, &handoff.ldos) {
            write_ldos_dat(&ldos_path, &handoff.ldos)
                .with_context(|| format!("failed to write {}", ldos_path.display()))?;
            written += 1;
        }
        if !rhoc_valid || !rhoc_table_matches_source_output(&rhoc_path, &handoff.rhoc) {
            write_rhoc_dat(&rhoc_path, &handoff.rhoc)
                .with_context(|| format!("failed to write {}", rhoc_path.display()))?;
            written += 1;
        }
    }

    Ok(written)
}

fn ldos_table_matches_source_output(path: &Path, source: &LdosDatData) -> bool {
    let Ok(cached) = read_ldos_dat(path) else {
        return false;
    };
    ldos_dat_matches_source_output(&cached, source)
}

fn rhoc_table_matches_source_output(path: &Path, source: &LdosDatData) -> bool {
    let Ok(cached) = read_rhoc_dat(path) else {
        return false;
    };
    ldos_dat_matches_source_output(&cached, source)
}

fn ldos_dat_matches_source_output(cached: &LdosDatData, source: &LdosDatData) -> bool {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-4;
    const DENSITY_ABS_TOLERANCE: f64 = 5.0e-5;
    const DENSITY_REL_TOLERANCE: f64 = 1.5e-3;

    if cached.energy_ev.len() != source.energy_ev.len()
        || cached.density.dim() != source.density.dim()
    {
        return false;
    }
    if !cached
        .energy_ev
        .iter()
        .zip(source.energy_ev.iter())
        .all(|(cached, source)| (cached - source).abs() <= ENERGY_TOLERANCE_EV)
    {
        return false;
    }
    cached
        .density
        .iter()
        .zip(source.density.iter())
        .all(|(cached, source)| {
            let diff = (cached - source).abs();
            let rel = diff / source.abs().max(1.0e-30);
            diff <= DENSITY_ABS_TOLERANCE || rel <= DENSITY_REL_TOLERANCE
        })
}

fn ldos_cache_matches_wavefunction_source_output(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<Option<bool>> {
    if !supported_ldos_wavefunction_source_controls(input)
        || !ldos_wavefunction_source_files_present(input, work_dir)
    {
        return Ok(None);
    }

    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    if input.control.lfms2 == 0 {
        let requested_potentials = (0..input.lmaxph.len()).collect::<Vec<_>>();
        let sources = crate::rhorrp::read_no_fms_ldos_rhol_source_tables(
            work_dir,
            energy_grid.clone(),
            &requested_potentials,
            LDOS_ORBITAL_COUNT,
        )?;
        let metadata_by_potential = read_ldos_source_metadata(work_dir, input.lmaxph.len())?;

        for source in sources {
            let potential = source.potential_index;
            let metadata = metadata_by_potential
                .get(potential)
                .cloned()
                .unwrap_or_default();
            let scattering_trace = Array2::zeros((LDOS_ORBITAL_COUNT, energy_grid.len()));
            let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
                header_lines: &[],
                fermi_level_hartree: Some(source.chemical_potential_hartree),
                charge_transfer: metadata.charge_transfer,
                electron_counts: &metadata.electron_counts,
                atom_count: metadata.atom_count,
                lorentzian_hwhh_hartree: energy_grid.first().map(|energy| energy.im),
                energy_grid_hartree: energy_grid.view(),
                embedded_ldos: source.table.density_grid.embedded_ldos.view(),
                scattering_ldos: source.table.density_grid.scattering_ldos.view(),
                scattering_trace: scattering_trace.view(),
                apply_scattering: false,
            })
            .with_context(|| {
                format!("failed to build LDOS dat payload for potential {potential}")
            })?;

            if !ldos_table_matches_source_output(
                &work_dir.join(format!("ldos{potential:02}.dat")),
                &handoff.ldos,
            ) || !rhoc_table_matches_source_output(
                &work_dir.join(format!("rhoc{potential:02}.dat")),
                &handoff.rhoc,
            ) {
                return Ok(Some(false));
            }
        }

        return Ok(Some(true));
    }

    let Some(_) = ldos_effective_fms_source_input(input, work_dir)? else {
        return Ok(None);
    };
    let source =
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(work_dir, energy_grid)?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < LDOS_ORBITAL_COUNT {
        return Ok(None);
    }
    let potential_count = source
        .phase
        .potential_count()
        .min(source.wavefunctions.wavefunctions.potential_count())
        .min(source.norman_radii_bohr.len())
        .min(input.lmaxph.len());
    let metadata_by_potential = read_ldos_source_metadata(work_dir, potential_count)?;

    for potential in 0..potential_count {
        let metadata = metadata_by_potential
            .get(potential)
            .cloned()
            .unwrap_or_default();
        let Some(scattering_trace) =
            ldos_source_scattering_trace(input, work_dir, potential, source.phase.energy_count())?
        else {
            return Ok(Some(false));
        };

        let solved = ldos_rhol_wavefunction_tables(LdosRholWavefunctionTablesInput {
            wavefunctions: &source.wavefunctions.wavefunctions,
            radii: source.wavefunctions.prepared.radii.view(),
            potential_index: potential,
            energy_grid_hartree: source.phase.energies_hartree.view(),
            scattering_trace: scattering_trace.view(),
            radial_step: source.wavefunctions.radial_dx,
            norman_radius: source.norman_radii_bohr[potential],
            angular_count: LDOS_ORBITAL_COUNT,
            apply_scattering: true,
        })
        .with_context(|| {
            format!("failed to solve LDOS rhol source table for potential {potential}")
        })?;
        let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: Some(source.phase.chemical_potential_hartree),
            charge_transfer: metadata.charge_transfer,
            electron_counts: &metadata.electron_counts,
            atom_count: metadata.atom_count,
            lorentzian_hwhh_hartree: source
                .phase
                .energies_hartree
                .first()
                .map(|energy| energy.im),
            energy_grid_hartree: source.phase.energies_hartree.view(),
            embedded_ldos: solved.density_grid.embedded_ldos.view(),
            scattering_ldos: solved.density_grid.scattering_ldos.view(),
            scattering_trace: scattering_trace.view(),
            apply_scattering: true,
        })
        .with_context(|| format!("failed to build LDOS dat payload for potential {potential}"))?;

        if !ldos_table_matches_source_output(
            &work_dir.join(format!("ldos{potential:02}.dat")),
            &handoff.ldos,
        ) || !rhoc_table_matches_source_output(
            &work_dir.join(format!("rhoc{potential:02}.dat")),
            &handoff.rhoc,
        ) {
            return Ok(Some(false));
        }
    }

    Ok(Some(true))
}

fn ldos_source_scattering_trace(
    input: &LdosInput,
    work_dir: &Path,
    potential: usize,
    energy_count: usize,
) -> Result<Option<Array2<Complex64>>> {
    if input.control.lfms2 == 0 {
        return Ok(Some(Array2::zeros((LDOS_ORBITAL_COUNT, energy_count))));
    }

    let potential_gtr_path = work_dir.join(format!("gtr{potential:02}.bin"));
    let fallback_gtr_path = work_dir.join("gtr00.bin");
    let gtr_path = if potential_gtr_path.is_file() {
        potential_gtr_path
    } else if fallback_gtr_path.is_file() {
        fallback_gtr_path
    } else {
        return Ok(None);
    };
    let gtr = read_gtr_bin(&gtr_path)
        .with_context(|| format!("failed to read {}", gtr_path.display()))?;
    let trace = gtr_bin_ldos_trace_handoff(&gtr, potential, LDOS_ORBITAL_COUNT)
        .with_context(|| format!("failed to select LDOS trace from {}", gtr_path.display()))?;
    if trace.energy_count != energy_count {
        bail!(
            "{} energy count {} does not match phase.bin energy count {}",
            gtr_path.display(),
            trace.energy_count,
            energy_count
        );
    }
    Ok(Some(trace.trace))
}

#[derive(Debug, Clone, Default)]
struct LdosSourceMetadata {
    charge_transfer: Option<f64>,
    electron_counts: Vec<LdosElectronCount>,
    atom_count: Option<usize>,
}

fn read_ldos_source_metadata(
    work_dir: &Path,
    potential_count: usize,
) -> Result<Vec<LdosSourceMetadata>> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let atom_count = read_optional_ldos_atom_count(work_dir, pot.potential_count())?;
    Ok((0..potential_count)
        .map(|potential| ldos_source_metadata_from_pot_bin(&pot, potential, atom_count))
        .collect())
}

fn read_optional_ldos_atom_count(work_dir: &Path, potential_count: usize) -> Result<Option<usize>> {
    let fms_path = work_dir.join("fms.inp");
    if !fms_path.is_file() {
        return Ok(None);
    }
    let fms_text = std::fs::read_to_string(&fms_path)
        .with_context(|| format!("failed to read {}", fms_path.display()))?;
    let fms = FmsInput::parse_str(&fms_path, &fms_text)
        .with_context(|| format!("failed to parse {}", fms_path.display()))?;
    if fms.cluster.rfms2 < 0.0 {
        return Ok(Some(0));
    }

    let geom_path = work_dir.join("geom.dat");
    if !geom_path.is_file() {
        return Ok(None);
    }

    let geom_text = std::fs::read_to_string(&geom_path)
        .with_context(|| format!("failed to read {}", geom_path.display()))?;
    let geom = GeomDat::parse_str(&geom_path, &geom_text)
        .with_context(|| format!("failed to parse {}", geom_path.display()))?;
    let geometry = rhorrp_geom_handoff_from_geom_dat(&geom).with_context(|| {
        format!(
            "failed to build LDOS geometry handoff from {}",
            geom_path.display()
        )
    })?;
    let fms = fms.to_rhorrp_handoff(potential_count).with_context(|| {
        format!(
            "failed to build LDOS FMS handoff from {}",
            fms_path.display()
        )
    })?;
    fms.central_fms_atom_count(&geometry)
        .map(Some)
        .with_context(|| {
            format!(
                "failed to compute LDOS FMS inclusion count from {}",
                geom_path.display()
            )
        })
}

fn ldos_source_metadata_from_pot_bin(
    pot: &PotBinData,
    potential: usize,
    atom_count: Option<usize>,
) -> LdosSourceMetadata {
    let charge_transfer = pot.norman_charges.get(potential).copied();
    let electron_counts = (0..LDOS_ORBITAL_COUNT)
        .filter_map(|angular_momentum| {
            pot.valence_occupancy
                .get((angular_momentum, potential))
                .copied()
                .map(|count| LdosElectronCount {
                    angular_momentum,
                    count,
                })
        })
        .collect();

    LdosSourceMetadata {
        charge_transfer,
        electron_counts,
        atom_count,
    }
}

fn ldos_input_energy_grid_hartree(input: &LdosInput) -> Result<Array1<Complex64>> {
    let energy_count = usize::try_from(input.control.neldos)
        .context("ldos.inp neldos must be non-negative and fit in usize")?;
    if energy_count == 0 {
        bail!("ldos.inp neldos must be positive");
    }

    let step_ev = if energy_count > 1 {
        (input.mesh.emax - input.mesh.emin) / (energy_count - 1) as f64
    } else {
        0.0
    };
    Ok(Array1::from_shape_fn(energy_count, |index| {
        Complex64::new(
            (input.mesh.emin + step_ev * index as f64) / FEFF_HARTREE_EV,
            input.mesh.eimag / FEFF_HARTREE_EV,
        )
    }))
}

fn supported_ldos_wavefunction_source_controls(input: &LdosInput) -> bool {
    input.control.mldos == 1 && input.ldostype <= 0 && input.lmaxph.iter().any(|&lmax| lmax >= 3)
}

fn can_generate_ldos_from_wavefunction_source_handoffs(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<bool> {
    if !supported_ldos_wavefunction_source_controls(input)
        || !ldos_wavefunction_source_files_present(input, work_dir)
    {
        return Ok(false);
    }

    validate_ldos_wavefunction_source_handoff_files(input, work_dir)?;
    if input.control.lfms2 == 0 {
        return Ok(true);
    }

    let Some(fms_input) = ldos_effective_fms_source_input(input, work_dir)? else {
        return Ok(false);
    };
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let source =
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(work_dir, energy_grid)?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < LDOS_ORBITAL_COUNT {
        return Ok(false);
    }

    crate::fms::has_supported_ldos_gtr_bin_source_grid_handoff(
        work_dir,
        &fms_input,
        source.phase.energies_hartree.view(),
        source.wavefunctions.wavefunctions.wave_numbers.view(),
        source.wavefunctions.wavefunctions.phase_shifts.view(),
    )
}

fn validate_ldos_wavefunction_source_handoff_files(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<()> {
    let pot_bin_path = work_dir.join("pot.bin");
    read_pot_bin(&pot_bin_path)
        .with_context(|| format!("failed to read {}", pot_bin_path.display()))?;

    let config_path = work_dir.join("config.dat");
    read_config_dat(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;

    let phase_path = work_dir.join("phase.bin");
    read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;

    let pot_input_path = work_dir.join("pot.inp");
    let pot_input_text = std::fs::read_to_string(&pot_input_path)
        .with_context(|| format!("failed to read {}", pot_input_path.display()))?;
    PotInput::parse_str(&pot_input_path, &pot_input_text)
        .with_context(|| format!("failed to parse {}", pot_input_path.display()))?;

    if input.control.lfms2 != 0 {
        let fms_input_path = work_dir.join("fms.inp");
        let fms_input_text = std::fs::read_to_string(&fms_input_path)
            .with_context(|| format!("failed to read {}", fms_input_path.display()))?;
        FmsInput::parse_str(&fms_input_path, &fms_input_text)
            .with_context(|| format!("failed to parse {}", fms_input_path.display()))?;
    }

    Ok(())
}

fn ldos_effective_fms_source_input(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<Option<LdosInput>> {
    if input.control.ispin == 0 {
        return Ok(Some(input.clone()));
    }

    if !ldos_nonmagnetic_ordinary_spin_fms_source_supported(work_dir)? {
        return Ok(None);
    }

    let mut effective = input.clone();
    effective.control.ispin = 0;
    Ok(Some(effective))
}

fn ldos_nonmagnetic_ordinary_spin_fms_source_supported(work_dir: &Path) -> Result<bool> {
    let xsph_path = work_dir.join("xsph.inp");
    if !xsph_path.is_file() {
        return Ok(false);
    }

    let xsph_text = std::fs::read_to_string(&xsph_path)
        .with_context(|| format!("failed to read {}", xsph_path.display()))?;
    let xsph = XsphInput::parse_str(&xsph_path, &xsph_text)
        .with_context(|| format!("failed to parse {}", xsph_path.display()))?;
    Ok(xsph
        .spinph
        .iter()
        .all(|spin| spin.abs() <= LDOS_SPINPH_ZERO_TOLERANCE))
}

fn ldos_wavefunction_source_files_present(input: &LdosInput, work_dir: &Path) -> bool {
    let radial_sources_present = ["pot.bin", "config.dat", "phase.bin", "pot.inp"]
        .iter()
        .all(|name| work_dir.join(name).is_file());
    radial_sources_present && (input.control.lfms2 == 0 || work_dir.join("fms.inp").is_file())
}

fn ldos_table_is_usable(path: &Path) -> bool {
    path.is_file() && read_ldos_dat(path).is_ok()
}

fn rhoc_table_is_usable(path: &Path) -> bool {
    path.is_file() && read_rhoc_dat(path).is_ok()
}

fn active_hubbard_ldos_enabled(work_dir: &Path) -> Result<bool> {
    Ok(read_hubbard_input_optional(work_dir)?.is_some_and(|input| input.mldos_hubb == 2))
}

fn read_hubbard_input_optional(work_dir: &Path) -> Result<Option<HubbardInput>> {
    let input_path = work_dir.join("hubbard.inp");
    if !input_path.is_file() {
        return Ok(None);
    }
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    HubbardInput::parse_str(input_path.clone(), &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
        .map(Some)
}

fn recoverable_ldos_from_rhoc_path(path: &Path, ispin: i32) -> bool {
    let Ok(rhoc) = read_rhoc_dat(path) else {
        return false;
    };
    ldos_from_rhoc_without_scattering(&rhoc, ispin).is_ok()
}

fn recoverable_rhoc_from_ldos_path(path: &Path, ispin: i32) -> bool {
    let Ok(ldos) = read_ldos_dat(path) else {
        return false;
    };
    rhoc_from_ldos_without_scattering(&ldos, ispin).is_ok()
}

fn ldos_from_rhoc_without_scattering(rhoc: &RhocDatData, ispin: i32) -> Result<LdosDatData> {
    if ispin == 0 && rhoc.is_spin_resolved() {
        bail!(
            "LDOS rhoc handoff spin shape does not match ldos.inp ispin={ispin}: rhoc spin_resolved={}",
            rhoc.is_spin_resolved()
        );
    }

    let energy_grid_hartree = Array1::from_iter(
        rhoc.energy_ev
            .iter()
            .map(|energy| Complex64::new(*energy / FEFF_HARTREE_EV, 0.0)),
    );
    if rhoc.is_spin_resolved() {
        if rhoc.density.ncols() != LDOS_ORBITAL_COUNT * LDOS_SPIN_COUNT {
            bail!(
                "LDOS rhoc handoff truncated spin shape requires cached ldos table: columns={}",
                rhoc.density.ncols()
            );
        }
        let embedded_ldos = spin_embedded_ldos_from_rhoc(rhoc);
        let zeros = Array3::<Complex64>::zeros(embedded_ldos.dim());
        let handoff = ldos_spin_dat_from_ff2rho(LdosSpinDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: rhoc
                .fermi_level_ev
                .map(|fermi_level| fermi_level / FEFF_HARTREE_EV),
            charge_transfer: rhoc.charge_transfer,
            electron_counts: &rhoc.electron_counts,
            atom_count: rhoc.atom_count,
            lorentzian_hwhh_hartree: rhoc.lorentzian_hwhh_ev.map(|width| width / FEFF_HARTREE_EV),
            energy_grid_hartree: energy_grid_hartree.view(),
            embedded_ldos: embedded_ldos.view(),
            scattering_ldos: zeros.view(),
            scattering_trace: zeros.view(),
            apply_scattering: false,
        })?;
        return Ok(handoff.ldos);
    }

    let embedded_ldos = rhoc.density.t().to_owned();
    let zeros = Array2::<Complex64>::zeros(embedded_ldos.dim());
    let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
        header_lines: &[],
        fermi_level_hartree: rhoc
            .fermi_level_ev
            .map(|fermi_level| fermi_level / FEFF_HARTREE_EV),
        charge_transfer: rhoc.charge_transfer,
        electron_counts: &rhoc.electron_counts,
        atom_count: rhoc.atom_count,
        lorentzian_hwhh_hartree: rhoc.lorentzian_hwhh_ev.map(|width| width / FEFF_HARTREE_EV),
        energy_grid_hartree: energy_grid_hartree.view(),
        embedded_ldos: embedded_ldos.view(),
        scattering_ldos: zeros.view(),
        scattering_trace: zeros.view(),
        apply_scattering: false,
    })?;
    Ok(handoff.ldos)
}

fn rhoc_from_ldos_without_scattering(ldos: &LdosDatData, ispin: i32) -> Result<RhocDatData> {
    if ispin == 0 && ldos.is_spin_resolved() {
        bail!(
            "LDOS ldos handoff spin shape does not match ldos.inp ispin={ispin}: ldos spin_resolved={}",
            ldos.is_spin_resolved()
        );
    }

    Ok(RhocDatData {
        header_lines: Vec::new(),
        fermi_level_ev: None,
        charge_transfer: None,
        electron_counts: Vec::new(),
        atom_count: None,
        lorentzian_hwhh_ev: None,
        energy_ev: ldos.energy_ev.clone(),
        density: ldos.density.clone(),
    })
}

fn lmdos_from_rhocm_without_scattering(rhocm: &LdosMagneticDatData) -> Result<LdosMagneticDatData> {
    let angular_count = validate_magnetic_no_fms_pair_shape(rhocm)?;
    let energy_grid_hartree = magnetic_energy_grid_hartree(rhocm);
    let embedded_magnetic_ldos = magnetic_embedded_ldos_from_rhocm(rhocm, angular_count);
    let zeros = Array4::<Complex64>::zeros(embedded_magnetic_ldos.dim());
    let handoff = ldos_magnetic_dat_from_ff2rho(LdosMagneticDatFromFf2rhoInput {
        header_lines: &[],
        fermi_level_hartree: rhocm
            .fermi_level_ev
            .map(|fermi_level| fermi_level / FEFF_HARTREE_EV),
        charge_transfer: rhocm.charge_transfer,
        electron_counts: &rhocm.electron_counts,
        atom_count: rhocm.atom_count,
        lorentzian_hwhh_hartree: rhocm
            .lorentzian_hwhh_ev
            .map(|width| width / FEFF_HARTREE_EV),
        energy_grid_hartree: energy_grid_hartree.view(),
        embedded_magnetic_ldos: embedded_magnetic_ldos.view(),
        scattering_magnetic_ldos: zeros.view(),
        magnetic_scattering_trace: zeros.view(),
        angular_count,
    })?;
    Ok(handoff.lmdos)
}

fn rhocm_from_lmdos_without_scattering(lmdos: &LdosMagneticDatData) -> Result<LdosMagneticDatData> {
    let angular_count = validate_magnetic_no_fms_pair_shape(lmdos)?;
    let energy_grid_hartree = magnetic_energy_grid_hartree(lmdos);
    let embedded_magnetic_ldos = magnetic_embedded_ldos_from_lmdos(lmdos, angular_count);
    let zeros = Array4::<Complex64>::zeros(embedded_magnetic_ldos.dim());
    let handoff = ldos_magnetic_dat_from_ff2rho(LdosMagneticDatFromFf2rhoInput {
        header_lines: &[],
        fermi_level_hartree: lmdos
            .fermi_level_ev
            .map(|fermi_level| fermi_level / FEFF_HARTREE_EV),
        charge_transfer: lmdos.charge_transfer,
        electron_counts: &lmdos.electron_counts,
        atom_count: lmdos.atom_count,
        lorentzian_hwhh_hartree: lmdos
            .lorentzian_hwhh_ev
            .map(|width| width / FEFF_HARTREE_EV),
        energy_grid_hartree: energy_grid_hartree.view(),
        embedded_magnetic_ldos: embedded_magnetic_ldos.view(),
        scattering_magnetic_ldos: zeros.view(),
        magnetic_scattering_trace: zeros.view(),
        angular_count,
    })?;
    Ok(handoff.rhocm)
}

fn validate_magnetic_no_fms_pair_shape(data: &LdosMagneticDatData) -> Result<usize> {
    let angular_count = data
        .angular_limit
        .checked_add(1)
        .context("LDOS magnetic angular count is too large")?;
    let magnetic_count = angular_count
        .checked_mul(angular_count)
        .context("LDOS magnetic column count is too large")?;
    let expected_columns = magnetic_count
        .checked_mul(LDOS_SPIN_COUNT)
        .context("LDOS magnetic spin column count is too large")?;
    if data.density.nrows() != data.energy_ev.len() || data.density.ncols() != expected_columns {
        bail!(
            "LDOS magnetic handoff shape {:?} does not match energy count {} and angular limit {}",
            data.density.dim(),
            data.energy_ev.len(),
            data.angular_limit
        );
    }
    Ok(angular_count)
}

fn magnetic_energy_grid_hartree(data: &LdosMagneticDatData) -> Array1<Complex64> {
    Array1::from_iter(
        data.energy_ev
            .iter()
            .map(|energy| Complex64::new(*energy / FEFF_HARTREE_EV, 0.0)),
    )
}

fn magnetic_embedded_ldos_from_rhocm(
    rhocm: &LdosMagneticDatData,
    angular_count: usize,
) -> Array4<f64> {
    magnetic_embedded_ldos_from_density(rhocm, angular_count, |_angular, density| density)
}

fn magnetic_embedded_ldos_from_lmdos(
    lmdos: &LdosMagneticDatData,
    angular_count: usize,
) -> Array4<f64> {
    magnetic_embedded_ldos_from_density(lmdos, angular_count, |angular, density| {
        density * (2 * angular + 1) as f64
    })
}

fn magnetic_embedded_ldos_from_density(
    data: &LdosMagneticDatData,
    angular_count: usize,
    scale_density: impl Fn(usize, f64) -> f64,
) -> Array4<f64> {
    let magnetic_count = angular_count * angular_count;
    Array4::from_shape_fn(
        (
            angular_count,
            magnetic_count,
            LDOS_SPIN_COUNT,
            data.energy_ev.len(),
        ),
        |(angular, magnetic, spin, energy)| {
            let angular_start = angular * angular;
            let angular_end = (angular + 1) * (angular + 1);
            if !(angular_start..angular_end).contains(&magnetic) {
                return 0.0;
            }
            let column = spin * magnetic_count + magnetic;
            scale_density(angular, data.density[(energy, column)])
        },
    )
}

fn spin_embedded_ldos_from_rhoc(rhoc: &RhocDatData) -> Array3<f64> {
    Array3::from_shape_fn(
        (LDOS_ORBITAL_COUNT, LDOS_SPIN_COUNT, rhoc.energy_ev.len()),
        |(angular, spin, energy_index)| {
            rhoc.density[(energy_index, spin * LDOS_ORBITAL_COUNT + angular)]
        },
    )
}

fn ldos_path_for_rhoc_table(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let index = name
        .strip_prefix("rhoc")
        .and_then(|suffix| suffix.strip_suffix(".dat"))?;
    Some(path.with_file_name(format!("ldos{index}.dat")))
}

fn rhoc_path_for_ldos_table(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let index = name
        .strip_prefix("ldos")
        .and_then(|suffix| suffix.strip_suffix(".dat"))?;
    Some(path.with_file_name(format!("rhoc{index}.dat")))
}

fn write_optional_module_log(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let data =
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_module_log(path, &data)
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

fn recover_existing_module_log_if_malformed(
    path: &Path,
    source_handoff_written: bool,
) -> Result<usize> {
    if !source_handoff_written || !path.is_file() || read_module_log_dat(path).is_ok() {
        return Ok(0);
    }
    write_generated_module_log(path)?;
    Ok(1)
}

fn write_generated_module_log(path: &Path) -> Result<()> {
    write_module_log(path, &generated_ldos_module_log())
}

fn generated_ldos_module_log() -> ModuleLogData {
    ModuleLogData {
        lines: vec![
            " Calculating LDOS ...".to_string(),
            "FEFF-serial using 1 thread.".to_string(),
            " Done with LDOS.".to_string(),
            "Done with module: LDOS.".to_string(),
        ],
        line_terminators: vec!["\n".to_string(); 4],
    }
}

fn cached_output_paths(work_dir: &Path) -> Result<Vec<CachedTable>> {
    let mut tables = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", work_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(kind) = cached_table_kind(name) {
            tables.push(CachedTable { path, kind });
        }
    }
    tables.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(tables)
}

fn cached_magnetic_output_paths(work_dir: &Path) -> Result<Vec<CachedMagneticTable>> {
    let mut tables = Vec::new();
    for entry in std::fs::read_dir(work_dir)
        .with_context(|| format!("failed to read {}", work_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", work_dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(kind) = cached_magnetic_table_kind(name) {
            tables.push(CachedMagneticTable { path, kind });
        }
    }
    tables.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(tables)
}

fn has_ldos_table(tables: &[CachedTable]) -> bool {
    tables
        .iter()
        .any(|table| table.kind == CachedTableKind::Ldos)
}

fn has_usable_ldos_cache(work_dir: &Path, input: &LdosInput) -> Result<bool> {
    let tables = cached_output_paths(work_dir)?;
    if tables.is_empty() {
        return Ok(false);
    }
    has_usable_ldos_cache_with_tables(work_dir, &tables, input)
}

fn has_usable_ldos_cache_with_tables(
    work_dir: &Path,
    tables: &[CachedTable],
    input: &LdosInput,
) -> Result<bool> {
    match can_use_cached_ldos_outputs(work_dir, tables, input, false) {
        Ok(can_use) => Ok(can_use),
        Err(_)
            if has_reciprocal_kmesh_source_handoff(work_dir)?
                && has_unrecoverable_ldos_table(tables, input) =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

fn has_recoverable_ldos_cache_with_tables(
    work_dir: &Path,
    tables: &[CachedTable],
    input: &LdosInput,
) -> Result<bool> {
    match can_use_cached_ldos_outputs(work_dir, tables, input, true) {
        Ok(can_use) => Ok(can_use),
        Err(_)
            if has_reciprocal_kmesh_source_handoff(work_dir)?
                && has_unrecoverable_ldos_table(tables, input) =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

fn has_unrecoverable_ldos_table(tables: &[CachedTable], input: &LdosInput) -> bool {
    tables
        .iter()
        .filter(|table| table.kind == CachedTableKind::Ldos)
        .any(|table| {
            if read_ldos_dat(&table.path).is_ok() {
                return false;
            }
            if input.control.lfms2 == 0
                && let Some(rhoc_path) = rhoc_path_for_ldos_table(&table.path)
                && recoverable_ldos_from_rhoc_path(&rhoc_path, input.control.ispin)
            {
                return false;
            }
            true
        })
}

fn prepare_module_log_cache(path: &Path) -> Result<()> {
    if path.is_file() {
        read_module_log_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(())
}

fn can_use_cached_ldos_outputs(
    work_dir: &Path,
    tables: &[CachedTable],
    input: &LdosInput,
    allow_recoverable_active_hubbard_pair: bool,
) -> Result<bool> {
    let can_generate_kmesh = has_supported_reciprocal_kmesh_handoff(work_dir)?;
    prepare_optional_or_generated_kmesh(work_dir, &work_dir.join("kmesh.dat"))?;

    let mut has_ldos = false;
    let mut can_generate_ldos = false;
    let mut can_generate_rhoc = false;
    if !active_hubbard_magnetic_outputs_complete(
        work_dir,
        tables,
        input,
        allow_recoverable_active_hubbard_pair,
    )? {
        return Ok(false);
    }

    for table in tables {
        match table.kind {
            CachedTableKind::Ldos => {
                let ldos = match read_ldos_dat(&table.path) {
                    Ok(ldos) => ldos,
                    Err(error) => {
                        if input.control.lfms2 == 0
                            && let Some(rhoc_path) = rhoc_path_for_ldos_table(&table.path)
                            && recoverable_ldos_from_rhoc_path(&rhoc_path, input.control.ispin)
                        {
                            can_generate_ldos = true;
                            continue;
                        }
                        return Err(error)
                            .with_context(|| format!("failed to read {}", table.path.display()));
                    }
                };
                has_ldos = true;
                if input.control.lfms2 == 0
                    && let Some(rhoc_path) = rhoc_path_for_ldos_table(&table.path)
                    && !rhoc_table_is_usable(&rhoc_path)
                {
                    if rhoc_from_ldos_without_scattering(&ldos, input.control.ispin).is_err() {
                        return Ok(false);
                    }
                    can_generate_rhoc = true;
                }
            }
            CachedTableKind::Rhoc => {
                let rhoc = match read_rhoc_dat(&table.path) {
                    Ok(rhoc) => rhoc,
                    Err(error) => {
                        if input.control.lfms2 == 0
                            && let Some(ldos_path) = ldos_path_for_rhoc_table(&table.path)
                            && recoverable_rhoc_from_ldos_path(&ldos_path, input.control.ispin)
                        {
                            has_ldos = true;
                            can_generate_rhoc = true;
                            continue;
                        }
                        return Err(error)
                            .with_context(|| format!("failed to read {}", table.path.display()));
                    }
                };
                if input.control.lfms2 == 0
                    && let Some(ldos_path) = ldos_path_for_rhoc_table(&table.path)
                    && !ldos_table_is_usable(&ldos_path)
                    && ldos_from_rhoc_without_scattering(&rhoc, input.control.ispin).is_ok()
                {
                    can_generate_ldos = true;
                }
            }
        }
    }

    if !can_generate_kmesh && !can_generate_ldos && !can_generate_rhoc {
        prepare_module_log_cache(&work_dir.join("logdos.dat"))?;
    }
    match ldos_cache_matches_wavefunction_source_output(input, work_dir) {
        Ok(Some(false)) | Err(_) => return Ok(false),
        Ok(None | Some(true)) => {}
    }

    Ok(has_ldos || can_generate_ldos)
}

fn active_hubbard_magnetic_outputs_complete(
    work_dir: &Path,
    tables: &[CachedTable],
    input: &LdosInput,
    allow_recoverable_ordinary_pair: bool,
) -> Result<bool> {
    if !active_hubbard_ldos_enabled(work_dir)? {
        return Ok(true);
    }

    let expected_indices = cached_ldos_output_indices(tables);
    if expected_indices.is_empty() {
        return Ok(false);
    }

    for index in expected_indices {
        let ordinary_pair = match cached_ldos_ordinary_pair(work_dir, &index)? {
            Some(pair) => Some(pair),
            None if allow_recoverable_ordinary_pair => {
                cached_ldos_recoverable_ordinary_pair(work_dir, &index, input.control.ispin)?
            }
            None => None,
        };
        let Some((ldos, rhoc)) = ordinary_pair else {
            return Ok(false);
        };
        if !ldos_ordinary_layouts_match(&ldos, &rhoc) {
            return Ok(false);
        }
        let ordinary_source_contract = match cached_ldos_ordinary_source_contract(work_dir, &index)
        {
            LdosSourceContract::Absent => None,
            LdosSourceContract::Present(contract) => Some(contract),
            LdosSourceContract::Incompatible => return Ok(false),
        };
        if let Some(source_contract) = ordinary_source_contract
            && (!ldos_ordinary_matches_source_contract(&ldos, source_contract)
                || !ldos_ordinary_matches_source_contract(&rhoc, source_contract))
        {
            return Ok(false);
        }
        let Some((lmdos, rhocm)) =
            cached_ldos_magnetic_pair_or_recoverable(work_dir, &index, input.control.lfms2)?
        else {
            return Ok(false);
        };
        if !ldos_energy_grids_match(lmdos.energy_ev.view(), ldos.energy_ev.view())
            || !ldos_energy_grids_match(rhocm.energy_ev.view(), ldos.energy_ev.view())
        {
            return Ok(false);
        }
        if !ldos_magnetic_layouts_match(&lmdos, &rhocm) {
            return Ok(false);
        }
        let magnetic_source_contract = match cached_ldos_magnetic_source_contract(work_dir, &index)
        {
            LdosSourceContract::Absent => None,
            LdosSourceContract::Present(contract) => Some(contract),
            LdosSourceContract::Incompatible => return Ok(false),
        };
        if let Some(source_contract) = magnetic_source_contract
            && (!ldos_magnetic_matches_source_contract(&lmdos, source_contract)
                || !ldos_magnetic_matches_source_contract(&rhocm, source_contract))
        {
            return Ok(false);
        }
        let offdiag_source_contract =
            match cached_ldos_offdiag_source_contract(work_dir, &index, lmdos.angular_limit) {
                LdosSourceContract::Absent => None,
                LdosSourceContract::Present(contract) => Some(contract),
                LdosSourceContract::Incompatible => return Ok(false),
            };
        if let Some(source_contract) = offdiag_source_contract {
            if !ldos_offdiag_matches_magnetic_cache(&lmdos, source_contract)
                || !ldos_offdiag_matches_magnetic_cache(&rhocm, source_contract)
            {
                return Ok(false);
            }
            if let Some(ordinary) = ordinary_source_contract
                && !ldos_hubbard_offdiag_source_contract_matches_ordinary(ordinary, source_contract)
            {
                return Ok(false);
            }
            if let Some(magnetic) = magnetic_source_contract
                && !ldos_hubbard_offdiag_source_contract_matches_magnetic(magnetic, source_contract)
            {
                return Ok(false);
            }
        }
        if let (Some(ordinary), Some(magnetic)) =
            (ordinary_source_contract, magnetic_source_contract)
            && !ldos_hubbard_source_contracts_match(ordinary, magnetic)
        {
            return Ok(false);
        }
    }

    Ok(true)
}

fn cached_ldos_magnetic_pair_or_recoverable(
    work_dir: &Path,
    index: &str,
    lfms2: i32,
) -> Result<Option<(LdosMagneticDatData, LdosMagneticDatData)>> {
    let lmdos_path = work_dir.join(format!("lmdos{index}.dat"));
    let rhocm_path = work_dir.join(format!("rhocm{index}.dat"));

    let lmdos = lmdos_path.is_file().then(|| {
        read_lmdos_dat(&lmdos_path)
            .with_context(|| format!("failed to read {}", lmdos_path.display()))
    });
    let rhocm = rhocm_path.is_file().then(|| {
        read_rhocm_dat(&rhocm_path)
            .with_context(|| format!("failed to read {}", rhocm_path.display()))
    });

    match (lmdos, rhocm) {
        (Some(Ok(lmdos)), Some(Ok(rhocm))) => Ok(Some((lmdos, rhocm))),
        (Some(Ok(lmdos)), _) if lfms2 == 0 => {
            let rhocm = rhocm_from_lmdos_without_scattering(&lmdos)
                .with_context(|| format!("failed to recover {}", rhocm_path.display()))?;
            Ok(Some((lmdos, rhocm)))
        }
        (_, Some(Ok(rhocm))) if lfms2 == 0 => {
            let lmdos = lmdos_from_rhocm_without_scattering(&rhocm)
                .with_context(|| format!("failed to recover {}", lmdos_path.display()))?;
            Ok(Some((lmdos, rhocm)))
        }
        (Some(Err(error)), _) => Err(error),
        (_, Some(Err(error))) => Err(error),
        (None, None) | (Some(Ok(_)), None) | (None, Some(Ok(_))) => Ok(None),
    }
}

fn cached_ldos_ordinary_pair(
    work_dir: &Path,
    index: &str,
) -> Result<Option<(LdosDatData, RhocDatData)>> {
    let ldos_path = work_dir.join(format!("ldos{index}.dat"));
    let rhoc_path = work_dir.join(format!("rhoc{index}.dat"));
    if !ldos_path.is_file() || !rhoc_path.is_file() {
        return Ok(None);
    }

    let Ok(ldos) = read_ldos_dat(&ldos_path) else {
        return Ok(None);
    };
    let Ok(rhoc) = read_rhoc_dat(&rhoc_path) else {
        return Ok(None);
    };
    Ok(Some((ldos, rhoc)))
}

fn cached_ldos_recoverable_ordinary_pair(
    work_dir: &Path,
    index: &str,
    ispin: i32,
) -> Result<Option<(LdosDatData, RhocDatData)>> {
    let ldos_path = work_dir.join(format!("ldos{index}.dat"));
    let rhoc_path = work_dir.join(format!("rhoc{index}.dat"));
    if rhoc_path.is_file()
        && let Ok(rhoc) = read_rhoc_dat(&rhoc_path)
        && let Ok(ldos) = ldos_from_rhoc_without_scattering(&rhoc, ispin)
    {
        return Ok(Some((ldos, rhoc)));
    }
    if ldos_path.is_file()
        && let Ok(ldos) = read_ldos_dat(&ldos_path)
        && let Ok(rhoc) = rhoc_from_ldos_without_scattering(&ldos, ispin)
    {
        return Ok(Some((ldos, rhoc)));
    }
    Ok(None)
}

fn ldos_energy_grids_match(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> bool {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-5;

    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| (left - right).abs() <= ENERGY_TOLERANCE_EV)
}

fn ldos_ordinary_layouts_match(left: &LdosDatData, right: &RhocDatData) -> bool {
    ldos_energy_grids_match(left.energy_ev.view(), right.energy_ev.view())
        && left.density.ncols() == right.density.ncols()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LdosSourceContract<T> {
    Absent,
    Present(T),
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LdosOrdinarySourceContract {
    energy_count: usize,
    angular_count: usize,
    density_column_count: usize,
}

fn cached_ldos_ordinary_source_contract(
    work_dir: &Path,
    index: &str,
) -> LdosSourceContract<LdosOrdinarySourceContract> {
    let Ok(potential) = index.parse::<usize>() else {
        return LdosSourceContract::Absent;
    };
    let specific_path = work_dir.join(format!("gtr{index}.bin"));
    let fallback_path = work_dir.join("gtr00.bin");
    let path = if specific_path.is_file() {
        specific_path
    } else if fallback_path.is_file() {
        fallback_path
    } else {
        return LdosSourceContract::Absent;
    };
    let Ok(source) = read_hubbard_ldos_gtr_bin_inferred(&path) else {
        return LdosSourceContract::Absent;
    };
    if potential >= source.potential_count() {
        return LdosSourceContract::Incompatible;
    }
    LdosSourceContract::Present(LdosOrdinarySourceContract {
        energy_count: source.energy_count(),
        angular_count: source.angular_count(),
        density_column_count: 2 * source.angular_count().min(LDOS_ORBITAL_COUNT),
    })
}

fn ldos_ordinary_matches_source_contract(
    data: &LdosDatData,
    source: LdosOrdinarySourceContract,
) -> bool {
    data.point_count() == source.energy_count && data.density.ncols() == source.density_column_count
}

fn ldos_magnetic_layouts_match(left: &LdosMagneticDatData, right: &LdosMagneticDatData) -> bool {
    ldos_energy_grids_match(left.energy_ev.view(), right.energy_ev.view())
        && left.angular_limit == right.angular_limit
        && left.magnetic_columns_per_spin() == right.magnetic_columns_per_spin()
        && left.density_column_count() == right.density_column_count()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LdosMagneticSourceContract {
    energy_count: usize,
    angular_limit: usize,
    magnetic_count: usize,
}

fn cached_ldos_magnetic_source_contract(
    work_dir: &Path,
    index: &str,
) -> LdosSourceContract<LdosMagneticSourceContract> {
    let Ok(potential) = index.parse::<usize>() else {
        return LdosSourceContract::Absent;
    };
    let specific_path = work_dir.join(format!("gtr_m{index}.bin"));
    let fallback_path = work_dir.join("gtr_m00.bin");
    let path = if specific_path.is_file() {
        specific_path
    } else if fallback_path.is_file() {
        fallback_path
    } else {
        return LdosSourceContract::Absent;
    };
    let Ok(source) = read_hubbard_ldos_gtr_m_bin_inferred(&path) else {
        return LdosSourceContract::Absent;
    };
    let Ok(handoff) = hubbard_ldos_gtr_m_trace_handoff(&source, potential) else {
        return LdosSourceContract::Incompatible;
    };
    let Some(angular_limit) = handoff.angular_count.checked_sub(1) else {
        return LdosSourceContract::Incompatible;
    };
    LdosSourceContract::Present(LdosMagneticSourceContract {
        energy_count: handoff.energy_count,
        angular_limit,
        magnetic_count: handoff.magnetic_count,
    })
}

fn ldos_magnetic_matches_source_contract(
    data: &LdosMagneticDatData,
    source: LdosMagneticSourceContract,
) -> bool {
    data.point_count() == source.energy_count
        && data.angular_limit == source.angular_limit
        && data.magnetic_columns_per_spin() == source.magnetic_count
}

fn ldos_hubbard_source_contracts_match(
    ordinary: LdosOrdinarySourceContract,
    magnetic: LdosMagneticSourceContract,
) -> bool {
    ordinary.energy_count == magnetic.energy_count
        && Some(ordinary.angular_count) == magnetic.angular_limit.checked_add(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LdosOffdiagSourceContract {
    energy_count: usize,
    angular_count: usize,
    order: usize,
}

fn cached_ldos_offdiag_source_contract(
    work_dir: &Path,
    index: &str,
    angular_limit: usize,
) -> LdosSourceContract<LdosOffdiagSourceContract> {
    let Ok(potential) = index.parse::<usize>() else {
        return LdosSourceContract::Absent;
    };
    let Ok(Some(hubbard)) = read_hubbard_input_optional(work_dir) else {
        return LdosSourceContract::Absent;
    };
    let Ok(hubbard_l) = usize::try_from(hubbard.l) else {
        return LdosSourceContract::Absent;
    };
    let specific_path = work_dir.join(format!("gtr_off{index}.bin"));
    let fallback_path = work_dir.join("gtr_off00.bin");
    let path = if specific_path.is_file() {
        specific_path
    } else if fallback_path.is_file() {
        fallback_path
    } else {
        return LdosSourceContract::Absent;
    };
    let Ok(source) = read_hubbard_ldos_gtr_off_bin(&path, hubbard_l, angular_limit) else {
        return LdosSourceContract::Absent;
    };
    if potential >= source.potential_count() {
        return LdosSourceContract::Incompatible;
    }
    LdosSourceContract::Present(LdosOffdiagSourceContract {
        energy_count: source.energy_count(),
        angular_count: source.angular_count(),
        order: source.order(),
    })
}

fn ldos_offdiag_matches_magnetic_cache(
    data: &LdosMagneticDatData,
    source: LdosOffdiagSourceContract,
) -> bool {
    source.order > 0
        && data.point_count() == source.energy_count
        && data.angular_limit.checked_add(1) == Some(source.angular_count)
}

fn ldos_hubbard_offdiag_source_contract_matches_ordinary(
    ordinary: LdosOrdinarySourceContract,
    offdiag: LdosOffdiagSourceContract,
) -> bool {
    ordinary.energy_count == offdiag.energy_count && ordinary.angular_count == offdiag.angular_count
}

fn ldos_hubbard_offdiag_source_contract_matches_magnetic(
    magnetic: LdosMagneticSourceContract,
    offdiag: LdosOffdiagSourceContract,
) -> bool {
    magnetic.energy_count == offdiag.energy_count
        && magnetic.angular_limit.checked_add(1) == Some(offdiag.angular_count)
}

fn cached_ldos_output_indices(tables: &[CachedTable]) -> BTreeSet<String> {
    tables
        .iter()
        .filter_map(|table| cached_ldos_table_index(&table.path))
        .collect()
}

fn cached_ldos_table_index(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let index = name
        .strip_prefix("ldos")
        .or_else(|| name.strip_prefix("rhoc"))?
        .strip_suffix(".dat")?;
    is_feff_potential_index(index).then(|| index.to_string())
}

fn cached_table_kind(name: &str) -> Option<CachedTableKind> {
    let index = name
        .strip_prefix("ldos")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedTableKind::Ldos);
    }

    let index = name
        .strip_prefix("rhoc")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedTableKind::Rhoc);
    }

    None
}

fn cached_magnetic_table_kind(name: &str) -> Option<CachedMagneticTableKind> {
    let index = name
        .strip_prefix("lmdos")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedMagneticTableKind::Lmdos);
    }

    let index = name
        .strip_prefix("rhocm")
        .and_then(|suffix| suffix.strip_suffix(".dat"));
    if index.is_some_and(is_feff_potential_index) {
        return Some(CachedMagneticTableKind::Rhocm);
    }

    None
}

fn is_feff_potential_index(index: &str) -> bool {
    index.len() == 2 && index.chars().all(|digit| digit.is_ascii_digit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedTable {
    path: PathBuf,
    kind: CachedTableKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedTableKind {
    Ldos,
    Rhoc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedMagneticTable {
    path: PathBuf,
    kind: CachedMagneticTableKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedMagneticTableKind {
    Lmdos,
    Rhocm,
}

#[cfg(test)]
mod tests {
    use super::{
        LDOS_SOURCE_REQUIREMENT_ERROR, has_cached_ldos_output, has_supported_kmesh_handoff,
        has_supported_source_output_handoff, run_in_dir, run_supported_kmesh_handoff_in_dir,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, Axis, ShapeBuilder, array};
    use num_complex::{Complex32, Complex64};
    use refeff_core::FEFF_HARTREE_EV;
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
        POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        CONFIG_DAT_ORBITAL_COUNT, CfAverage, ConfigDatData, ConfigDatPotential, FmsCluster,
        FmsControl, FmsDebye, FmsInput, GeomDat, GeomDatRow, GlobalControl, GlobalInput,
        GlobalNorms, GlobalQControl, GtrBinData, HubbardInput, HubbardLdosGtrBinData,
        HubbardLdosGtrMBinData, HubbardLdosGtrOffBinData, KmeshMetadata, KmeshRow, LdosDatData,
        LdosMagneticDatData, ModuleLogData, PhaseBinData, PhaseBinPotential, PhaseBinScalars,
        PotBinData, PotBinScalars, PotControl, PotInput, PotOverlapShell, PotPotential, PotRamp,
        PotRun, PotScattering, PotThermal, PotTolerances, ReciprocalCell, ReciprocalInput,
        ReciprocalKMesh, config_dat_string, fms_input_string, geom_dat_string, global_input_string,
        hubbard_input_string, ldos_input_string, pot_input_string, read_gtr_bin, read_kmesh_dat,
        read_ldos_dat, read_lmdos_dat, read_module_log_dat, read_rhoc_dat, read_rhocm_dat,
        reciprocal_input_string, write_gtr_bin, write_hubbard_ldos_gtr_bin,
        write_hubbard_ldos_gtr_m_bin, write_hubbard_ldos_gtr_off_bin, write_ldos_dat,
        write_lmdos_dat, write_module_log_dat, write_phase_bin, write_pot_bin, write_rhoc_dat,
        write_rhocm_dat,
    };
    use std::{
        path::{Path, PathBuf},
        process::Command,
    };

    #[test]
    fn ldos_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), false)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_generation_without_cache_or_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled LDOS should require complete source state")?;

        assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
        Ok(())
    }

    #[test]
    fn ldos_module_generates_kmesh_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled LDOS should still require complete source state")?;

        assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_gtr_bin_from_source_fms_handoffs_before_source_requirement()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_source_input(temp.path())?;
        write_ldos_fms_source_handoffs(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("LDOS should still require complete source state")?;

        assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
        let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
        assert_eq!(gtr00.energy_count(), 2);
        assert_eq!(gtr00.potential_count(), 2);
        assert_eq!(gtr00.angular_channel_count(), 2);
        assert_eq!(gtr00.highest_potential_index, 1);
        assert_eq!(gtr00.fms_mode, 2);
        assert!(
            gtr00
                .values
                .index_axis(Axis(1), 0)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(
            gtr00
                .values
                .index_axis(Axis(1), 1)
                .iter()
                .all(|value| value.norm() == 0.0)
        );
        let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
        assert_eq!(gtr01.energy_count(), 2);
        assert_eq!(gtr01.potential_count(), 2);
        assert_eq!(gtr01.angular_channel_count(), 2);
        assert!(
            gtr01
                .values
                .index_axis(Axis(1), 0)
                .iter()
                .all(|value| value.norm() == 0.0)
        );
        assert!(
            gtr01
                .values
                .index_axis(Axis(1), 1)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_skips_fms_source_when_ldos_mesh_differs_from_phase_grid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_source_input_with_neldos(temp.path(), 3)?;
        write_ldos_fms_source_handoffs(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("mismatched LDOS/FMS mesh should stop at the source requirement")?;

        assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
        assert!(!temp.path().join("gtr00.bin").exists());
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_regenerates_stale_fms_gtr_on_ldos_mesh() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_neldos(temp.path(), 1, 3)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;
        write_gtr_bin(temp.path().join("gtr00.bin"), &sample_ldos_gtr_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
        assert_eq!(gtr00.energy_count(), 3);
        assert_eq!(gtr00.angular_channel_count(), 4);
        assert!(
            gtr00
                .values
                .index_axis(Axis(1), 0)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        assert!(
            gtr00
                .values
                .index_axis(Axis(1), 1)
                .iter()
                .all(|value| value.norm() == 0.0)
        );
        let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
        assert_eq!(gtr01.energy_count(), 3);
        assert_eq!(gtr01.angular_channel_count(), 4);
        assert!(
            gtr01
                .values
                .index_axis(Axis(1), 0)
                .iter()
                .all(|value| value.norm() == 0.0)
        );
        assert!(
            gtr01
                .values
                .index_axis(Axis(1), 1)
                .iter()
                .any(|value| value.norm() > 0.0)
        );
        for potential in 0..=1 {
            let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            assert_eq!(ldos.density.dim(), (3, 4));
            assert_eq!(rhoc.density.dim(), (3, 4));
            assert!(ldos.density.iter().all(|value| value.is_finite()));
            assert!(rhoc.density.iter().all(|value| value.is_finite()));
        }
        Ok(())
    }

    #[test]
    fn ldos_module_generates_zero_fms_trace_for_nonpositive_cluster_radius() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input(temp.path())?;
        let input_text = std::fs::read_to_string(temp.path().join("ldos.inp"))?;
        let mut input = refeff_io::LdosInput::parse_str(temp.path().join("ldos.inp"), &input_text)?;
        input.mesh.rfms2 = -1.0;
        std::fs::write(temp.path().join("ldos.inp"), ldos_input_string(&input)?)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;

        assert!(has_supported_source_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!has_supported_source_output_handoff(temp.path())?);
        let gtr00 = read_gtr_bin(temp.path().join("gtr00.bin"))?;
        assert_eq!(gtr00.energy_count(), 2);
        assert_eq!(gtr00.potential_count(), 2);
        assert_eq!(gtr00.fms_mode, 2);
        assert!(gtr00.values.iter().all(|value| value.norm() == 0.0));
        let gtr01 = read_gtr_bin(temp.path().join("gtr01.bin"))?;
        assert_eq!(gtr01.energy_count(), 2);
        assert_eq!(gtr01.potential_count(), 2);
        assert_eq!(gtr01.fms_mode, 2);
        assert!(gtr01.values.iter().all(|value| value.norm() == 0.0));
        for potential in 0..=1 {
            let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            assert_eq!(ldos.density, rhoc.density);
        }
        Ok(())
    }

    #[test]
    fn ldos_module_generates_tables_from_wavefunction_and_gtr_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input(temp.path())?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        write_gtr_bin(temp.path().join("gtr00.bin"), &sample_ldos_gtr_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        for potential in 0..=1 {
            let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            assert_eq!(ldos.density.dim(), (2, 4));
            assert_eq!(rhoc.density.dim(), (2, 4));
            assert!(ldos.density.iter().all(|value| value.is_finite()));
            assert!(rhoc.density.iter().all(|value| value.is_finite()));
            assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
            assert!(rhoc.density.iter().any(|value| value.abs() > 0.0));
        }
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_no_fms_tables_from_wavefunction_source_without_gtr() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;

        assert!(has_supported_source_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!has_supported_source_output_handoff(temp.path())?);
        assert!(!temp.path().join("gtr00.bin").exists());
        assert!(!temp.path().join("gtr01.bin").exists());
        for potential in 0..=1 {
            let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            assert_eq!(ldos.density.dim(), (2, 4));
            assert_eq!(ldos.energy_ev, rhoc.energy_ev);
            assert_eq!(ldos.density, rhoc.density);
            assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
        }
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_claim_malformed_wavefunction_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;

        assert!(!has_supported_source_output_handoff(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed LDOS wavefunction source should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert!(!temp.path().join("ldos00.dat").exists());
        assert!(!temp.path().join("rhoc00.dat").exists());
        assert!(!temp.path().join("logdos.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_claim_cached_tables_with_malformed_wavefunction_source_handoff()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        std::fs::write(temp.path().join("phase.bin"), b"not a phase.bin handoff\n")?;

        assert!(!has_cached_ldos_output(temp.path())?);
        assert!(!has_supported_source_output_handoff(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("malformed LDOS wavefunction source should fail through explicit run")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("phase.bin"), "{chain}");
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert!(!temp.path().join("logdos.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_regenerates_stale_no_fms_tables_from_wavefunction_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        run_in_dir(temp.path())?;
        let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        let mut stale_ldos = expected_ldos.clone();
        let mut stale_rhoc = expected_rhoc.clone();
        stale_ldos.density[(0, 0)] += 0.25;
        stale_rhoc.density[(0, 0)] += 0.25;
        write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &stale_rhoc)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_ldos_dat(temp.path().join("ldos00.dat"))?,
            expected_ldos
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
            expected_rhoc
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_regenerates_stale_fms_tables_from_wavefunction_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input(temp.path())?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        write_ldos_wavefunction_fms_geometry_handoffs(temp.path())?;
        run_in_dir(temp.path())?;
        let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        let mut stale_ldos = expected_ldos.clone();
        let mut stale_rhoc = expected_rhoc.clone();
        stale_ldos.density[(0, 0)] += 0.25;
        stale_rhoc.density[(0, 0)] += 0.25;
        write_ldos_dat(temp.path().join("ldos00.dat"), &stale_ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &stale_rhoc)?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_ldos_dat(temp.path().join("ldos00.dat"))?,
            expected_ldos
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
            expected_rhoc
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_no_fms_tables_from_radial_source_without_fms_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2(temp.path(), 0)?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;
        std::fs::remove_file(temp.path().join("fms.inp"))?;

        assert!(has_supported_source_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!temp.path().join("gtr00.bin").exists());
        assert!(!temp.path().join("gtr01.bin").exists());
        for potential in 0..=1 {
            let ldos = read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let rhoc = read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            assert_eq!(ldos.energy_ev, rhoc.energy_ev);
            assert_eq!(ldos.density, rhoc.density);
            assert!(ldos.density.iter().any(|value| value.abs() > 0.0));
        }
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_limits_no_fms_source_tables_to_available_potentials() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_wavefunction_source_input_with_lfms2_and_lmax(temp.path(), 0, &[3, 3, 3])?;
        write_ldos_wavefunction_source_handoffs(temp.path())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(temp.path().join("ldos00.dat").is_file());
        assert!(temp.path().join("rhoc00.dat").is_file());
        assert!(temp.path().join("ldos01.dat").is_file());
        assert!(temp.path().join("rhoc01.dat").is_file());
        assert!(!temp.path().join("ldos02.dat").exists());
        assert!(!temp.path().join("rhoc02.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_generates_supported_kmesh_handoff_without_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(has_supported_kmesh_handoff(temp.path())?);
        let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert!(!temp.path().join("ldos00.dat").exists());
        assert!(!temp.path().join("logdos.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_claim_malformed_reciprocal_kmesh_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            "not a reciprocal.inp handoff\n",
        )?;

        assert!(!has_supported_kmesh_handoff(temp.path())?);
        assert_eq!(run_supported_kmesh_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed reciprocal.inp should fail through the explicit LDOS runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("reciprocal.inp"), "{chain}");
        assert!(!temp.path().join("kmesh.dat").exists());
        assert!(!temp.path().join("ldos00.dat").exists());
        assert!(!temp.path().join("logdos.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_existing_malformed_log_for_supported_kmesh_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_supported_kmesh_handoff(temp.path())?);
        let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        let log = sample_module_log();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_module_log_dat(temp.path().join("logdos.dat"), &log)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert_eq!(read_module_log_dat(temp.path().join("logdos.dat"))?, log);
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        std::fs::write(temp.path().join("ldos.inp"), "not an ldos.inp handoff\n")?;

        assert!(!has_cached_ldos_output(temp.path())?);
        assert!(!has_supported_source_output_handoff(temp.path())?);
        assert!(!has_supported_kmesh_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed ldos.inp should fail through the explicit LDOS runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("ldos.inp"), "{chain}");
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert!(!temp.path().join("logdos.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_active_hubbard_magnetic_sidecars() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        let lmdos = sample_lmdos_dat();
        let rhocm = sample_rhocm_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
        assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_accepts_active_hubbard_cache_with_matching_source_contracts() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        let lmdos = sample_lmdos_dat();
        let rhocm = sample_rhocm_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
        write_hubbard_ldos_gtr_bin(
            temp.path().join("gtr00.bin"),
            &sample_hubbard_gtr_source_contract(1, 3, 1),
        )?;
        write_hubbard_ldos_gtr_m_bin(
            temp.path().join("gtr_m00.bin"),
            &sample_hubbard_gtr_m_source_contract(1, 3, 1),
        )?;
        write_hubbard_ldos_gtr_off_bin(
            temp.path().join("gtr_off00.bin"),
            &sample_hubbard_gtr_off_source_contract(1, 1, 3, 1),
        )?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert_eq!(read_lmdos_dat(temp.path().join("lmdos00.dat"))?, lmdos);
        assert_eq!(read_rhocm_dat(temp.path().join("rhocm00.dat"))?, rhocm);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_accepts_active_hubbard_cache_with_fallback_source_contracts() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_active_hubbard_cached_tables(temp.path(), "01")?;
        write_hubbard_ldos_gtr_bin(
            temp.path().join("gtr00.bin"),
            &sample_hubbard_gtr_source_contract(1, 3, 2),
        )?;
        write_hubbard_ldos_gtr_m_bin(
            temp.path().join("gtr_m00.bin"),
            &sample_hubbard_gtr_m_source_contract(1, 3, 2),
        )?;
        write_hubbard_ldos_gtr_off_bin(
            temp.path().join("gtr_off00.bin"),
            &sample_hubbard_gtr_off_source_contract(1, 1, 3, 2),
        )?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_ldos_dat(temp.path().join("ldos01.dat"))?,
            sample_ldos_dat()
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc01.dat"))?,
            sample_rhoc_dat()
        );
        assert_eq!(
            read_lmdos_dat(temp.path().join("lmdos01.dat"))?,
            sample_lmdos_dat()
        );
        assert_eq!(
            read_rhocm_dat(temp.path().join("rhocm01.dat"))?,
            sample_rhocm_dat()
        );
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_requires_active_hubbard_magnetic_sidecars() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard LDOS should require magnetic sidecars")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        assert!(!temp.path().join("lmdos00.dat").exists());
        assert!(!temp.path().join("rhocm00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_missing_active_hubbard_rhocm_without_fms() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert_ldos_magnetic_table_close(
            &read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
            &super::rhocm_from_lmdos_without_scattering(&sample_lmdos_dat())?,
            "active-Hubbard no-FMS rhocm repair",
        );
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_missing_active_hubbard_lmdos_without_fms() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert_ldos_magnetic_table_close(
            &read_lmdos_dat(temp.path().join("lmdos00.dat"))?,
            &super::lmdos_from_rhocm_without_scattering(&sample_rhocm_dat())?,
            "active-Hubbard no-FMS lmdos repair",
        );
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_advertise_active_hubbard_cache_without_rhoc_pair() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
            super::rhoc_from_ldos_without_scattering(&sample_ldos_dat(), 0)?
        );
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_ordinary_ldos_energy_grid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        let mut ldos = sample_ldos_dat();
        ldos.energy_ev[1] += 0.25;
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard ordinary LDOS/RHOC pair should share an energy mesh")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_ordinary_rhoc_energy_grid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let mut rhoc = sample_rhoc_dat();
        rhoc.energy_ev[1] += 0.25;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard ordinary LDOS/RHOC pair should share an energy mesh")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_ordinary_density_layout() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let mut rhoc = sample_rhoc_dat();
        rhoc.density = array![
            [5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4, 9.0E-4, 10.0E-4],
            [5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4, 9.1E-4, 10.1E-4],
            [5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4, 9.2E-4, 10.2E-4]
        ];
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard ordinary LDOS/RHOC pair should share a density layout")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_active_hubbard_ordinary_ldos_pair() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
        let rhoc = sample_rhoc_dat();
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_active_hubbard_ordinary_rhoc_pair() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        let ldos = sample_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert_eq!(rhoc.energy_ev, ldos.energy_ev);
        assert_eq!(rhoc.density, ldos.density);
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_magnetic_energy_grid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        let mut lmdos = sample_lmdos_dat();
        lmdos.energy_ev[2] += 0.25;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard magnetic sidecars should match LDOS energy mesh")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_rhocm_energy_grid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        let mut rhocm = sample_rhocm_dat();
        rhocm.energy_ev[0] -= 0.25;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard rhocm sidecar should match LDOS energy mesh")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_magnetic_layout_rejects_divergent_energy_grids() {
        let mut lmdos = sample_lmdos_dat();
        let mut rhocm = sample_rhocm_dat();
        lmdos.energy_ev[1] += 4.0e-5;
        rhocm.energy_ev[1] -= 4.0e-5;

        assert!(
            !super::ldos_magnetic_layouts_match(&lmdos, &rhocm),
            "magnetic sidecars that diverge from each other must not share one LDOS layout"
        );
    }

    #[test]
    fn ldos_module_rejects_stale_active_hubbard_magnetic_layout() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        let mut rhocm = sample_rhocm_dat();
        rhocm.angular_limit = 0;
        rhocm.density = array![[9.0E-4, 8.0E-4], [9.1E-4, 8.1E-4], [9.2E-4, 8.2E-4]];
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
        assert_eq!(
            read_rhocm_dat(temp.path().join("rhocm00.dat"))?.angular_limit,
            0
        );

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard magnetic sidecars should share an lx layout")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_cache_that_conflicts_with_gtr_m_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
        write_hubbard_ldos_gtr_m_bin(
            temp.path().join("gtr_m00.bin"),
            &sample_hubbard_gtr_m_source_contract(2, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard magnetic cache should match valid gtr_m source layout")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_cache_that_conflicts_with_gtr_off_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
        write_hubbard_ldos_gtr_off_bin(
            temp.path().join("gtr_off00.bin"),
            &sample_hubbard_gtr_off_source_contract(2, 1, 2, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard magnetic cache should match valid gtr_off source layout")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_gtr_source_that_omits_cached_potential() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_active_hubbard_cached_tables(temp.path(), "01")?;
        write_hubbard_ldos_gtr_bin(
            temp.path().join("gtr01.bin"),
            &sample_hubbard_gtr_source_contract(1, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard gtr source should cover cached potential index")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_gtr_m_source_that_omits_cached_potential() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_active_hubbard_cached_tables(temp.path(), "01")?;
        write_hubbard_ldos_gtr_m_bin(
            temp.path().join("gtr_m01.bin"),
            &sample_hubbard_gtr_m_source_contract(1, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard gtr_m source should cover cached potential index")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_gtr_off_source_that_omits_cached_potential() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_active_hubbard_cached_tables(temp.path(), "01")?;
        write_hubbard_ldos_gtr_off_bin(
            temp.path().join("gtr_off01.bin"),
            &sample_hubbard_gtr_off_source_contract(2, 1, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard gtr_off source should cover cached potential index")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_ordinary_cache_that_conflicts_with_gtr_source()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &sample_rhocm_dat())?;
        write_hubbard_ldos_gtr_bin(
            temp.path().join("gtr00.bin"),
            &sample_hubbard_gtr_source_contract(2, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard ordinary cache should match valid gtr source layout")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_active_hubbard_source_traces_with_conflicting_layouts() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        let mut lmdos = sample_lmdos_dat();
        lmdos.angular_limit = 0;
        lmdos.density = array![[1.0E-4, 2.0E-4], [1.1E-4, 2.1E-4], [1.2E-4, 2.2E-4]];
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &lmdos)?;
        let mut rhocm = sample_rhocm_dat();
        rhocm.angular_limit = 0;
        rhocm.density = array![[9.0E-4, 8.0E-4], [9.1E-4, 8.1E-4], [9.2E-4, 8.2E-4]];
        write_rhocm_dat(temp.path().join("rhocm00.dat"), &rhocm)?;
        write_hubbard_ldos_gtr_bin(
            temp.path().join("gtr00.bin"),
            &sample_hubbard_gtr_source_contract(1, 3, 1),
        )?;
        write_hubbard_ldos_gtr_m_bin(
            temp.path().join("gtr_m00.bin"),
            &sample_hubbard_gtr_m_source_contract(0, 3, 1),
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("active-Hubbard gtr and gtr_m source layouts should agree")?;
        let chain = format!("{error:?}");

        assert!(chain.contains(LDOS_SOURCE_REQUIREMENT_ERROR), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_hubbard_nio_reference_zip_magnetic_sidecars() -> Result<()> {
        let Some(zip_path) = reference_hubbard_nio_ldos_zip()? else {
            eprintln!("skipping LDOS NiO Hubbard magnetic reference test; REFERENCE.zip not found");
            return Ok(());
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            eprintln!("skipping LDOS NiO Hubbard magnetic reference test; unzip command not found");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        for name in ["ldos.inp", "hubbard.inp"] {
            std::fs::write(
                temp.path().join(name),
                unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
            )?;
        }

        let hubbard_text = std::fs::read_to_string(temp.path().join("hubbard.inp"))?;
        let hubbard = HubbardInput::parse_str(temp.path().join("hubbard.inp"), &hubbard_text)?;
        assert_eq!(
            hubbard.mldos_hubb, 2,
            "NiO reference should exercise active Hubbard magnetic LDOS"
        );

        let mut expected_ldos = Vec::new();
        let mut expected_rhoc = Vec::new();
        let mut expected_lmdos = Vec::new();
        let mut expected_rhocm = Vec::new();
        for potential in 0..=2 {
            for prefix in ["ldos", "rhoc", "lmdos", "rhocm"] {
                let name = format!("{prefix}{potential:02}.dat");
                std::fs::write(
                    temp.path().join(&name),
                    unzip_reference_entry(&zip_path, &format!("REFERENCE/{name}"))?,
                )?;
            }
            expected_ldos.push(read_ldos_dat(
                temp.path().join(format!("ldos{potential:02}.dat")),
            )?);
            expected_rhoc.push(read_rhoc_dat(
                temp.path().join(format!("rhoc{potential:02}.dat")),
            )?);
            expected_lmdos.push(read_lmdos_dat(
                temp.path().join(format!("lmdos{potential:02}.dat")),
            )?);
            expected_rhocm.push(read_rhocm_dat(
                temp.path().join(format!("rhocm{potential:02}.dat")),
            )?);
        }

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 12);
        for potential in 0..=2 {
            assert_eq!(
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?,
                expected_ldos[potential]
            );
            assert_eq!(
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?,
                expected_rhoc[potential]
            );
            assert_eq!(
                read_lmdos_dat(temp.path().join(format!("lmdos{potential:02}.dat")))?,
                expected_lmdos[potential]
            );
            assert_eq!(
                read_rhocm_dat(temp.path().join(format!("rhocm{potential:02}.dat")))?,
                expected_rhocm[potential]
            );
        }
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_malformed_active_hubbard_magnetic_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        std::fs::write(
            temp.path().join("lmdos00.dat"),
            "not a magnetic ldos table\n",
        )?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("malformed active-Hubbard magnetic sidecar should fail validation")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("lmdos00.dat"), "{chain}");

        let error = run_in_dir(temp.path())
            .err()
            .context("explicit LDOS run should reject malformed active-Hubbard magnetic sidecar")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("lmdos00.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_active_hubbard_rhocm_sidecar_without_fms() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_hubbard_input(temp.path(), 2)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        write_lmdos_dat(temp.path().join("lmdos00.dat"), &sample_lmdos_dat())?;
        std::fs::write(
            temp.path().join("rhocm00.dat"),
            "not a magnetic rhoc table\n",
        )?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 5);
        assert_ldos_magnetic_table_close(
            &read_rhocm_dat(temp.path().join("rhocm00.dat"))?,
            &super::rhocm_from_lmdos_without_scattering(&sample_lmdos_dat())?,
            "active-Hubbard no-FMS malformed rhocm repair",
        );
        Ok(())
    }

    #[test]
    fn ldos_module_ignores_non_hubbard_magnetic_sidecar_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let ldos = sample_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        std::fs::write(
            temp.path().join("lmdos00.dat"),
            "not a magnetic ldos table\n",
        )?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        assert!(read_rhoc_dat(temp.path().join("rhoc00.dat")).is_ok());
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_advertise_malformed_ldos_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("malformed LDOS cache should fail predicate validation")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("ldos00.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_generates_kmesh_handoff_when_malformed_ldos_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        assert!(!has_cached_ldos_output(temp.path())?);
        assert!(has_supported_kmesh_handoff(temp.path())?);
        let count = run_supported_kmesh_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        let kmesh = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(kmesh.rows.len(), 8);
        assert!(!temp.path().join("logdos.dat").exists());

        let error = run_in_dir(temp.path())
            .err()
            .context("stale final LDOS cache should fall through to the source requirement")?;
        assert!(
            error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_advertise_malformed_rhoc_sidecar_with_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 1)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("malformed RHOC sidecar should fail predicate validation")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("rhoc00.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_advertise_malformed_kmesh_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("malformed kmesh.dat should fail predicate validation")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("kmesh.dat"), "{chain}");

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed kmesh.dat should fail through the explicit LDOS runner")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("kmesh.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_kmesh_from_reciprocal_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert_kmesh_row_close(
            data.rows[0],
            KmeshRow {
                index: 1,
                k_point: [0.831_446_454_055_273_6; 3],
                weight: 0.5,
                metadata: data.rows[0].metadata,
            },
            5.0e-4,
        );
        Ok(())
    }

    #[test]
    fn ldos_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;
        std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("malformed logdos.dat should fail predicate validation")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logdos.dat"), "{chain}");

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed logdos.dat should fail through the explicit LDOS runner")?;
        let chain = format!("{error:?}");

        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logdos.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_keeps_malformed_log_strict_when_kmesh_is_already_valid() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

        let error = has_cached_ldos_output(temp.path())
            .err()
            .context("valid existing kmesh.dat should not mask a malformed LDOS log")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logdos.dat"), "{chain}");

        let error = run_in_dir(temp.path())
            .err()
            .context("explicit LDOS run should keep the malformed log strict")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("logdos.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_module_log_for_recoverable_ldos_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let rhoc = sample_rhoc_dat();
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;
        std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_module_log_for_recoverable_kmesh_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;
        std::fs::write(temp.path().join("kmesh.dat"), "not kmesh.dat\n")?;
        std::fs::write(temp.path().join("logdos.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        assert_eq!(read_kmesh_dat(temp.path().join("kmesh.dat"))?.rows.len(), 8);
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_module_log_from_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        let ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_kmesh_from_reciprocal_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input(temp.path(), true)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_ldos_dat())?;
        let reciprocal = sample_reciprocal_input(8);
        std::fs::write(
            temp.path().join("reciprocal.inp"),
            reciprocal_input_string(&reciprocal)?,
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let data = read_kmesh_dat(temp.path().join("kmesh.dat"))?;
        assert_eq!(data.rows.len(), 8);
        assert_eq!(
            data.rows[0].metadata,
            Some(KmeshMetadata {
                requested_points: 8,
                irreducible_points: 8,
                divisions: [2, 2, 2],
            })
        );
        assert_kmesh_row_close(
            data.rows[0],
            KmeshRow {
                index: 1,
                k_point: [0.831_446_454_055_273_6; 3],
                weight: 0.5,
                metadata: data.rows[0].metadata,
            },
            5.0e-4,
        );
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert_eq!(rhoc.energy_ev, sample_ldos_dat().energy_ev);
        assert_eq!(rhoc.density, sample_ldos_dat().density);
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let ldos = sample_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert!(rhoc.header_lines.is_empty());
        assert_eq!(rhoc.fermi_level_ev, None);
        assert_eq!(rhoc.energy_ev, ldos.energy_ev);
        assert_eq!(rhoc.density, ldos.density);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let rhoc = sample_rhoc_dat();
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert!(ldos.header_lines.iter().any(|line| line.contains("sDOS")));
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        assert_eq!(
            read_module_log_dat(temp.path().join("logdos.dat"))?,
            sample_module_log()
        );
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let rhoc = sample_rhoc_dat();
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;
        std::fs::write(temp.path().join("ldos00.dat"), "not an ldos table\n")?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert!(ldos.header_lines.iter().any(|line| line.contains("sDOS")));
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let ldos = sample_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert!(rhoc.header_lines.is_empty());
        assert_eq!(rhoc.fermi_level_ev, None);
        assert_eq!(rhoc.energy_ev, ldos.energy_ev);
        assert_eq!(rhoc.density, ldos.density);
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_ldos_for_partial_rhoc_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 0)?;
        let cached_ldos = sample_ldos_dat();
        let rhoc = sample_rhoc_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &cached_ldos)?;
        write_rhoc_dat(temp.path().join("rhoc01.dat"), &rhoc)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, cached_ldos);
        let generated_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert_eq!(generated_rhoc.energy_ev, cached_ldos.energy_ev);
        assert_eq!(generated_rhoc.density, cached_ldos.density);
        let generated = read_ldos_dat(temp.path().join("ldos01.dat"))?;
        assert_eq!(generated.energy_ev, rhoc.energy_ev);
        assert_eq!(generated.density, rhoc.density);
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc01.dat"))?, rhoc);
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_spin_ldos_from_rhoc_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
        let rhoc = sample_spin_rhoc_dat();
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &rhoc)?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        let ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        assert!(ldos.is_spin_resolved());
        assert_eq!(ldos.energy_ev, rhoc.energy_ev);
        assert_eq!(ldos.density, rhoc.density);
        assert!(
            ldos.header_lines
                .iter()
                .any(|line| line.contains("sDOS(up)") && line.contains("sDOS(down)"))
        );
        assert_eq!(read_rhoc_dat(temp.path().join("rhoc00.dat"))?, rhoc);
        Ok(())
    }

    #[test]
    fn ldos_module_generates_missing_spin_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
        let ldos = sample_spin_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert!(rhoc.header_lines.is_empty());
        assert_eq!(rhoc.fermi_level_ev, None);
        assert!(rhoc.is_spin_resolved());
        assert_eq!(rhoc.energy_ev, ldos.energy_ev);
        assert_eq!(rhoc.density, ldos.density);
        Ok(())
    }

    #[test]
    fn ldos_module_recovers_malformed_spin_rhoc_from_ldos_without_fms_scattering() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 1)?;
        let ldos = sample_spin_ldos_dat();
        write_ldos_dat(temp.path().join("ldos00.dat"), &ldos)?;
        std::fs::write(temp.path().join("rhoc00.dat"), "not an rhoc table\n")?;

        assert!(has_cached_ldos_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(read_ldos_dat(temp.path().join("ldos00.dat"))?, ldos);
        let rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;
        assert!(rhoc.header_lines.is_empty());
        assert_eq!(rhoc.fermi_level_ev, None);
        assert!(rhoc.is_spin_resolved());
        assert_eq!(rhoc.energy_ev, ldos.energy_ev);
        assert_eq!(rhoc.density, ldos.density);
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_rhoc_spin_shape_that_disagrees_with_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 0)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_spin_rhoc_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("spin-shaped rhoc should not satisfy non-spin LDOS input")?;

        assert!(error.chain().any(|cause| {
            cause
                .to_string()
                .contains("LDOS rhoc handoff spin shape does not match ldos.inp ispin=0")
        }));
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_ldos_spin_shape_for_non_spin_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2_and_ispin(temp.path(), true, 0, 0)?;
        write_ldos_dat(temp.path().join("ldos00.dat"), &sample_spin_ldos_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("spin-shaped ldos should not satisfy non-spin LDOS input")?;

        assert!(error.chain().any(|cause| {
            cause
                .to_string()
                .contains("LDOS ldos handoff spin shape does not match ldos.inp ispin=0")
        }));
        assert!(!temp.path().join("rhoc00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_rejects_rhoc_only_when_fms_scattering_is_enabled() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_ldos_input_with_lfms2(temp.path(), true, 1)?;
        write_rhoc_dat(temp.path().join("rhoc00.dat"), &sample_rhoc_dat())?;

        assert!(!has_cached_ldos_output(temp.path())?);
        let error = run_in_dir(temp.path())
            .err()
            .context("FMS LDOS should still require complete scattering source state")?;

        assert!(error.to_string().contains(LDOS_SOURCE_REQUIREMENT_ERROR));
        assert!(!temp.path().join("ldos00.dat").exists());
        Ok(())
    }

    #[test]
    fn ldos_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_ldos_dir()? else {
            eprintln!("skipping LDOS reference test; generated EXAFS/Cu reference not found");
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "ldos.inp",
            "ldos00.dat",
            "ldos01.dat",
            "rhoc00.dat",
            "rhoc01.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_ldos = read_ldos_dat(temp.path().join("ldos00.dat"))?;
        let expected_rhoc = read_rhoc_dat(temp.path().join("rhoc00.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_ldos_dat(temp.path().join("ldos00.dat"))?,
            expected_ldos
        );
        assert_eq!(
            read_rhoc_dat(temp.path().join("rhoc00.dat"))?,
            expected_rhoc
        );
        Ok(())
    }

    #[test]
    fn ldos_module_generates_no_fms_reference_tables_from_source_handoffs() -> Result<()> {
        let cases = reference_ldos_source_cases()?;
        if cases.is_empty() {
            eprintln!("skipping LDOS source reference test; no generated reference cases found");
            return Ok(());
        }

        for case in cases {
            let temp = tempfile::tempdir()?;
            std::fs::copy(
                case.expected_dir.join("ldos.inp"),
                temp.path().join("ldos.inp"),
            )
            .with_context(|| format!("failed to copy LDOS input for {}", case.label))?;
            for name in [
                "pot.bin",
                "config.dat",
                "phase.bin",
                "pot.inp",
                "fms.inp",
                "global.inp",
            ] {
                std::fs::copy(case.source_dir.join(name), temp.path().join(name))
                    .with_context(|| format!("failed to copy {name} for {}", case.label))?;
            }
            if case.source_dir.join("geom.dat").is_file() {
                std::fs::copy(
                    case.source_dir.join("geom.dat"),
                    temp.path().join("geom.dat"),
                )
                .with_context(|| format!("failed to copy geom.dat for {}", case.label))?;
            }

            let count = run_in_dir(temp.path()).with_context(|| {
                format!("failed to generate LDOS source tables for {}", case.label)
            })?;

            assert_eq!(count, case.potential_count * 2, "{}", case.label);
            assert!(!temp.path().join("gtr00.bin").exists(), "{}", case.label);
            for potential in 0..case.potential_count {
                let generated_ldos =
                    read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
                let generated_rhoc =
                    read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
                let reference_ldos =
                    read_ldos_dat(case.expected_dir.join(format!("ldos{potential:02}.dat")))?;
                let reference_rhoc =
                    read_rhoc_dat(case.expected_dir.join(format!("rhoc{potential:02}.dat")))?;

                assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
                assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
                assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, case.label);
                assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, case.label);
                assert!(generated_ldos.density.iter().all(|value| value.is_finite()));
                assert!(generated_rhoc.density.iter().all(|value| value.is_finite()));
                assert!(generated_ldos.density.iter().any(|value| value.abs() > 0.0));
                assert_eq!(generated_ldos.energy_ev, generated_rhoc.energy_ev);
                assert_eq!(generated_ldos.density, generated_rhoc.density);
            }
        }
        Ok(())
    }

    #[test]
    fn ldos_module_generates_zero_cluster_fms_reference_trace_from_source_handoffs() -> Result<()> {
        let Some(reference_dir) = reference_ldos_dir()? else {
            eprintln!(
                "skipping LDOS FMS source reference test; generated EXAFS/Cu reference not found"
            );
            return Ok(());
        };
        if !reference_dir.join("gtr00.bin").is_file()
            || !reference_ldos_source_present(&reference_dir)
            || !reference_dir.join("geom.dat").is_file()
        {
            eprintln!("skipping LDOS FMS source reference test; EXAFS/Cu FMS handoffs not found");
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let input_text = std::fs::read_to_string(reference_dir.join("ldos.inp"))?;
        let mut input =
            refeff_io::LdosInput::parse_str(reference_dir.join("ldos.inp"), &input_text)?;
        input.control.lfms2 = 1;
        std::fs::write(temp.path().join("ldos.inp"), ldos_input_string(&input)?)?;
        for name in [
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
            "geom.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))
                .with_context(|| format!("failed to copy {name} for LDOS FMS source reference"))?;
        }

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_zero_gtr_reference_compatible(
            &read_gtr_bin(temp.path().join("gtr00.bin"))?,
            &read_gtr_bin(reference_dir.join("gtr00.bin"))?,
        );
        if reference_dir.join("gtr01.bin").is_file() {
            assert_zero_gtr_reference_compatible(
                &read_gtr_bin(temp.path().join("gtr01.bin"))?,
                &read_gtr_bin(reference_dir.join("gtr01.bin"))?,
            );
        }
        for potential in 0..=1 {
            let generated_ldos =
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let generated_rhoc =
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            let reference_ldos =
                read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
            let reference_rhoc =
                read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
            assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
            assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
            assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, "EXAFS/Cu zero-FMS");
            assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, "EXAFS/Cu zero-FMS");
        }
        Ok(())
    }

    #[test]
    fn ldos_module_matches_nonzero_fms_reference_from_source_handoffs() -> Result<()> {
        let Some(reference_dir) = reference_ldos_nonzero_fms_dir()? else {
            eprintln!(
                "skipping LDOS nonzero FMS source reference test; generated XANES/Cu FMS reference not found"
            );
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "ldos.inp",
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
            "geom.dat",
            ".dimensions.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
                format!("failed to copy {name} for LDOS nonzero FMS source reference")
            })?;
        }

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        for potential in 0..=1 {
            let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
            let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
            assert_gtr_reference_close(
                &generated_gtr,
                &reference_gtr,
                &format!("XANES/Cu FMS gtr{potential:02}.bin"),
            );

            let generated_ldos =
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let generated_rhoc =
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            let reference_ldos =
                read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
            let reference_rhoc =
                read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
            assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
            assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
            assert_ldos_density_grid_close(&generated_ldos, &reference_ldos, "XANES/Cu FMS");
            assert_ldos_density_grid_close(&generated_rhoc, &reference_rhoc, "XANES/Cu FMS");
        }
        Ok(())
    }

    #[test]
    fn ldos_module_matches_ordinary_spin_fms_reference_from_source_handoffs() -> Result<()> {
        let Some(reference_dir) = reference_ldos_ordinary_spin_fms_dir()? else {
            eprintln!(
                "skipping LDOS ordinary-spin FMS source reference test; generated XANES/Cu ordinary-spin FMS reference not found"
            );
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "ldos.inp",
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
            "geom.dat",
            "xsph.inp",
            ".dimensions.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
                format!("failed to copy {name} for LDOS ordinary-spin FMS source reference")
            })?;
        }

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        for potential in 0..=1 {
            let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
            let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
            assert_gtr_reference_close(
                &generated_gtr,
                &reference_gtr,
                &format!("XANES/Cu ordinary-spin FMS gtr{potential:02}.bin"),
            );

            let generated_ldos =
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let generated_rhoc =
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            let reference_ldos =
                read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
            let reference_rhoc =
                read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
            assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
            assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
            assert_ldos_density_grid_close(
                &generated_ldos,
                &reference_ldos,
                "XANES/Cu ordinary-spin FMS",
            );
            assert_ldos_density_grid_close(
                &generated_rhoc,
                &reference_rhoc,
                "XANES/Cu ordinary-spin FMS",
            );
        }
        Ok(())
    }

    #[test]
    fn ldos_module_matches_production_fms_reference_from_source_handoffs() -> Result<()> {
        let Some(reference_dir) = reference_ldos_production_fms_dir()? else {
            eprintln!(
                "skipping LDOS production FMS source reference test; generated XANES/Cu production FMS reference not found"
            );
            return Ok(());
        };

        let temp = tempfile::tempdir()?;
        for name in [
            "ldos.inp",
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
            "geom.dat",
            ".dimensions.dat",
        ] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name)).with_context(|| {
                format!("failed to copy {name} for LDOS production FMS source reference")
            })?;
        }

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        for potential in 0..=1 {
            let generated_gtr = read_gtr_bin(temp.path().join(format!("gtr{potential:02}.bin")))?;
            let reference_gtr = read_gtr_bin(reference_dir.join(format!("gtr{potential:02}.bin")))?;
            assert_gtr_reference_close(
                &generated_gtr,
                &reference_gtr,
                &format!("XANES/Cu production FMS gtr{potential:02}.bin"),
            );

            let generated_ldos =
                read_ldos_dat(temp.path().join(format!("ldos{potential:02}.dat")))?;
            let generated_rhoc =
                read_rhoc_dat(temp.path().join(format!("rhoc{potential:02}.dat")))?;
            let reference_ldos =
                read_ldos_dat(reference_dir.join(format!("ldos{potential:02}.dat")))?;
            let reference_rhoc =
                read_rhoc_dat(reference_dir.join(format!("rhoc{potential:02}.dat")))?;
            assert_ldos_table_mesh_matches(&generated_ldos, &reference_ldos);
            assert_ldos_table_mesh_matches(&generated_rhoc, &reference_rhoc);
            assert_ldos_density_grid_close(
                &generated_ldos,
                &reference_ldos,
                "XANES/Cu production FMS",
            );
            assert_ldos_density_grid_close(
                &generated_rhoc,
                &reference_rhoc,
                "XANES/Cu production FMS",
            );
        }
        Ok(())
    }

    fn sample_reciprocal_input(total_kpoints: i32) -> ReciprocalInput {
        ReciprocalInput {
            ispace: 0,
            cell: Some(ReciprocalCell {
                lattice_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                volume_scale: -1.0,
                imaginary_energy: 0.0,
                core_hole_strength: 1.0,
                lattice_name: "P".to_string(),
                space_group_hm: "Pm-3m".to_string(),
                space_group: 221,
                atom_count: 1,
                absorber: 1,
                core_hole: 1,
                k_mesh: ReciprocalKMesh {
                    total: total_kpoints,
                    x: total_kpoints,
                    y: 0,
                    z: 0,
                    kind: 3,
                    use_symmetry: false,
                },
                positions: vec![[0.0, 0.0, 0.0]],
                potentials: vec![0],
                labels: vec!["Cu".to_string()],
                stretch: [0.0, 0.0, 0.0],
            }),
        }
    }

    fn assert_kmesh_row_close(actual: KmeshRow, expected: KmeshRow, tolerance: f64) {
        assert_eq!(actual.index, expected.index);
        assert_eq!(actual.metadata, expected.metadata);
        assert!((actual.weight - expected.weight).abs() <= tolerance);
        for (actual, expected) in actual.k_point.iter().zip(expected.k_point.iter()) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }

    fn write_ldos_input(work_dir: &Path, enabled: bool) -> Result<()> {
        write_ldos_input_with_lfms2(work_dir, enabled, 0)
    }

    fn write_hubbard_input(work_dir: &Path, mldos_hubb: i32) -> Result<()> {
        let input = HubbardInput {
            i_hubbard: if mldos_hubb == 2 { 2 } else { 1 },
            mldos_hubb,
            u: 4.0,
            j: 0.5,
            fermi_shift: 0.0,
            l: 2,
        };
        std::fs::write(work_dir.join("hubbard.inp"), hubbard_input_string(&input)?)?;
        Ok(())
    }

    fn write_active_hubbard_cached_tables(work_dir: &Path, index: &str) -> Result<()> {
        write_ldos_dat(
            work_dir.join(format!("ldos{index}.dat")),
            &sample_ldos_dat(),
        )?;
        write_rhoc_dat(
            work_dir.join(format!("rhoc{index}.dat")),
            &sample_rhoc_dat(),
        )?;
        write_lmdos_dat(
            work_dir.join(format!("lmdos{index}.dat")),
            &sample_lmdos_dat(),
        )?;
        write_rhocm_dat(
            work_dir.join(format!("rhocm{index}.dat")),
            &sample_rhocm_dat(),
        )?;
        Ok(())
    }

    fn write_ldos_input_with_lfms2(work_dir: &Path, enabled: bool, lfms2: i32) -> Result<()> {
        write_ldos_input_with_lfms2_and_ispin(work_dir, enabled, lfms2, 0)
    }

    fn write_ldos_input_with_lfms2_and_ispin(
        work_dir: &Path,
        enabled: bool,
        lfms2: i32,
        ispin: i32,
    ) -> Result<()> {
        std::fs::write(
            work_dir.join("ldos.inp"),
            format!(
                concat!(
                    "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                    "{:4}{:4}{:4}{:4}{:4} {:7} {:4}\n",
                    "rfms2, emin, emax, eimag, rgrd\n",
                    "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}\n",
                    "rdirec, toler1, toler2\n",
                    "{:13.5}{:13.5}{:13.5}\n",
                    " lmaxph(0:nph)\n",
                    "{:4}{:4}\n",
                    "ldostype\n",
                    "{:4}\n"
                ),
                i32::from(enabled),
                lfms2,
                0,
                ispin,
                0,
                3,
                11,
                -1.0,
                -1.0,
                1.0,
                0.1,
                0.05,
                -1.0,
                0.001,
                0.001,
                3,
                3,
                0
            ),
        )?;
        Ok(())
    }

    fn write_ldos_source_input(work_dir: &Path) -> Result<()> {
        write_ldos_source_input_with_neldos(work_dir, 2)
    }

    fn write_ldos_source_input_with_neldos(work_dir: &Path, neldos: i32) -> Result<()> {
        std::fs::write(
            work_dir.join("ldos.inp"),
            format!(
                concat!(
                    "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                    "   1   1   0   0   0{:8}   11\n",
                    "rfms2, emin, emax, eimag, rgrd\n",
                    "      3.00000     -1.00000      1.00000      0.10000      0.05000\n",
                    "rdirec, toler1, toler2\n",
                    "      5.00000      0.00100      0.00100\n",
                    " lmaxph(0:nph)\n",
                    "   1   1\n",
                    "ldostype\n",
                    "   0\n",
                ),
                neldos
            ),
        )?;
        Ok(())
    }

    fn write_ldos_fms_source_handoffs(work_dir: &Path) -> Result<()> {
        let fms = FmsInput {
            control: FmsControl {
                mfms: 1,
                idwopt: -1,
                minv: 0,
            },
            cluster: FmsCluster {
                rfms2: -1.0,
                rdirec: -1.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: FmsDebye {
                tk: 0.0,
                thetad: 0.0,
                sig2g: 0.0,
            },
            lmaxph: vec![1, 1],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 0,
        };
        std::fs::write(work_dir.join("fms.inp"), fms_input_string(&fms)?)?;
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&sample_global_input())?,
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&sample_ldos_source_geom())?,
        )?;
        write_phase_bin(work_dir.join("phase.bin"), &sample_ldos_source_phase_bin())?;
        Ok(())
    }

    fn write_ldos_wavefunction_source_input(work_dir: &Path) -> Result<()> {
        write_ldos_wavefunction_source_input_with_lfms2(work_dir, 1)
    }

    fn write_ldos_wavefunction_source_input_with_lfms2(work_dir: &Path, lfms2: i32) -> Result<()> {
        write_ldos_wavefunction_source_input_with_neldos(work_dir, lfms2, 2)
    }

    fn write_ldos_wavefunction_source_input_with_neldos(
        work_dir: &Path,
        lfms2: i32,
        neldos: i32,
    ) -> Result<()> {
        write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(
            work_dir,
            lfms2,
            &[3, 3],
            neldos,
        )
    }

    fn write_ldos_wavefunction_source_input_with_lfms2_and_lmax(
        work_dir: &Path,
        lfms2: i32,
        lmaxph: &[i32],
    ) -> Result<()> {
        write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(work_dir, lfms2, lmaxph, 2)
    }

    fn write_ldos_wavefunction_source_input_with_lfms2_lmax_neldos(
        work_dir: &Path,
        lfms2: i32,
        lmaxph: &[i32],
        neldos: i32,
    ) -> Result<()> {
        let lmaxph = lmaxph
            .iter()
            .map(|value| format!("{value:4}"))
            .collect::<String>();
        std::fs::write(
            work_dir.join("ldos.inp"),
            format!(
                concat!(
                    "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc\n",
                    "   1{:4}   0   0   0{:8}   11\n",
                    "rfms2, emin, emax, eimag, rgrd\n",
                    "      3.00000     -1.00000      1.00000      0.10000      0.05000\n",
                    "rdirec, toler1, toler2\n",
                    "      5.00000      0.00100      0.00100\n",
                    " lmaxph(0:nph)\n",
                    "{}\n",
                    "ldostype\n",
                    "   0\n",
                ),
                lfms2, neldos, lmaxph
            ),
        )?;
        Ok(())
    }

    fn write_ldos_wavefunction_source_handoffs(work_dir: &Path) -> Result<()> {
        write_pot_bin(
            work_dir.join("pot.bin"),
            &sample_ldos_wavefunction_pot_bin(),
        )?;
        std::fs::write(
            work_dir.join("config.dat"),
            config_dat_string(&sample_ldos_wavefunction_config_dat())?,
        )?;
        write_phase_bin(
            work_dir.join("phase.bin"),
            &sample_ldos_wavefunction_phase_bin(),
        )?;
        std::fs::write(
            work_dir.join("pot.inp"),
            pot_input_string(&sample_ldos_wavefunction_pot_input())?,
        )?;
        std::fs::write(
            work_dir.join("fms.inp"),
            fms_input_string(&sample_ldos_wavefunction_fms_input())?,
        )?;
        Ok(())
    }

    fn write_ldos_wavefunction_fms_geometry_handoffs(work_dir: &Path) -> Result<()> {
        let mut fms = sample_ldos_wavefunction_fms_input();
        fms.control.idwopt = -1;
        std::fs::write(work_dir.join("fms.inp"), fms_input_string(&fms)?)?;
        std::fs::write(
            work_dir.join("global.inp"),
            global_input_string(&sample_global_input())?,
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&sample_ldos_source_geom())?,
        )?;
        Ok(())
    }

    fn sample_ldos_wavefunction_pot_bin() -> PotBinData {
        let potentials = 2;
        PotBinData {
            titles: vec!["LDOS wavefunction source test".to_string()],
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
                |(coefficient, orbital, potential)| {
                    0.01 * (coefficient + 1) as f64
                        + 0.001 * orbital as f64
                        + 0.1 * potential as f64
                },
            ),
            small_coefficients: Array3::from_shape_fn(
                (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
                |(coefficient, orbital, potential)| {
                    -0.01 * (coefficient + 1) as f64
                        - 0.001 * orbital as f64
                        - 0.1 * potential as f64
                },
            ),
            electron_density: ldos_wavefunction_radial_matrix(potentials, 0.01),
            coulomb_potential: ldos_wavefunction_radial_matrix(potentials, -0.02),
            total_potential: ldos_wavefunction_radial_matrix(potentials, -0.03),
            valence_density: ldos_wavefunction_radial_matrix(potentials, 0.004),
            valence_potential: ldos_wavefunction_radial_matrix(potentials, -0.005),
            magnetization_density: ldos_wavefunction_radial_matrix(potentials, 0.0002),
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

    fn ldos_wavefunction_radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
        Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
            scale * (row + 1) as f64 + potential as f64 * 0.125
        })
    }

    fn sample_ldos_wavefunction_config_dat() -> ConfigDatData {
        let mut first_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        let mut first_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        first_occupations[0] = 1.0;
        first_occupations[1] = 2.0;
        first_valence[1] = 0.5;

        let mut second_occupations = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
        let mut second_valence = Array1::zeros(CONFIG_DAT_ORBITAL_COUNT);
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

    fn sample_ldos_wavefunction_phase_bin() -> PhaseBinData {
        let spin_count = 1;
        let energy_grid = Array1::from_vec(vec![
            Complex64::new(-1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
            Complex64::new(1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
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
                sample_ldos_wavefunction_phase_potential(29, "Cu", energy_count, spin_count),
                sample_ldos_wavefunction_phase_potential(8, "O", energy_count, spin_count),
            ],
            transition_moments: Array4::zeros((energy_count, 1, 1, spin_count)),
            raw_pads: None,
        }
    }

    fn sample_ldos_wavefunction_phase_potential(
        atomic_number: usize,
        label: &str,
        energy_count: usize,
        spin_count: usize,
    ) -> PhaseBinPotential {
        PhaseBinPotential {
            lmax: 3,
            atomic_number,
            label: label.to_string(),
            phase_shifts: Array3::zeros((energy_count, 7, spin_count)),
        }
    }

    fn sample_ldos_wavefunction_pot_input() -> PotInput {
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
            titles: vec!["LDOS wavefunction source test".to_string()],
            scattering: PotScattering {
                gamach: 0.0,
                rgrd: 0.05,
                ca1: 0.0,
                ecv: 0.0,
                totvol: 1.0,
                rfms1: 0.0,
                corval_emin: 0.0,
            },
            potentials: vec![
                PotPotential {
                    z: 29,
                    lmaxsc: 3,
                    xnatph: 1.0,
                    xion: 0.0,
                    folp: 1.0,
                },
                PotPotential {
                    z: 8,
                    lmaxsc: 3,
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

    fn sample_ldos_wavefunction_fms_input() -> FmsInput {
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
            lmaxph: vec![3, 3],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 0,
        }
    }

    fn sample_ldos_gtr_bin() -> GtrBinData {
        GtrBinData {
            point_count_declared: 2,
            horizontal_count: 2,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            values: Array3::from_shape_fn((2, 2, 4), |(energy, potential, angular)| {
                Complex64::new(
                    0.08 + 0.02 * energy as f64 + 0.03 * potential as f64 + 0.01 * angular as f64,
                    -0.04 + 0.005 * angular as f64,
                )
            }),
        }
    }

    fn sample_global_input() -> GlobalInput {
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
                do_nrixs: 0,
                ldecmx: 0,
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

    fn sample_ldos_source_geom() -> GeomDat {
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
                    x: 1.4,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 0,
                },
            ],
        }
    }

    fn sample_ldos_source_phase_bin() -> PhaseBinData {
        let energy_count = 2;
        let spin_count = 1;
        let transition_count = 8;
        let energy_grid = Array1::from_vec(vec![
            Complex64::new(-1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
            Complex64::new(1.0 / FEFF_HARTREE_EV, 0.1 / FEFF_HARTREE_EV),
        ]);
        let reference_energy = Array2::zeros((energy_count, spin_count));
        let potentials = (0..2)
            .map(|potential| PhaseBinPotential {
                lmax: 1,
                atomic_number: 29,
                label: format!("Cu{potential}"),
                phase_shifts: Array3::from_shape_fn(
                    (energy_count, 3, spin_count),
                    |(energy, signed_l, _)| {
                        let scale = 0.03 * (potential + 1) as f64
                            + 0.01 * energy as f64
                            + 0.02 * signed_l as f64;
                        Complex64::new(scale, 0.004 * (signed_l + 1) as f64)
                    },
                ),
            })
            .collect();
        let transition_moments = Array4::<Complex64>::from_shape_fn(
            (energy_count, 1, transition_count, spin_count).f(),
            |(energy, _, transition, _)| {
                Complex64::new(
                    0.2 + 0.03 * energy as f64 + 0.015 * transition as f64,
                    -0.01 * transition as f64,
                )
            },
        );

        PhaseBinData {
            spin_count,
            energy_count,
            main_energy_count: energy_count,
            auxiliary_energy_count: 0,
            ihole: 1,
            fermi_index: 1,
            pad_width: 8,
            final_state_count: transition_count,
            transition_count,
            q_count: 1,
            scalars: PhaseBinScalars {
                average_norman_radius: 1.2,
                fermi_level: 0.0,
                edge_energy: 8_979.0,
            },
            energy_grid,
            reference_energy,
            potentials,
            transition_moments,
            raw_pads: None,
        }
    }

    fn sample_ldos_dat() -> LdosDatData {
        LdosDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e        sDOS           pDOS          dDOS          fDOS".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4],
                [1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4],
                [1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4]
            ],
        }
    }

    fn sample_rhoc_dat() -> LdosDatData {
        LdosDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4],
                [5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4],
                [5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4]
            ],
        }
    }

    fn sample_lmdos_dat() -> LdosMagneticDatData {
        LdosMagneticDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e   s(+0)DOS-up   p(-1)DOS-up   p(+0)DOS-up   p(+1)DOS-up   s(+0)DOS-dn   p(-1)DOS-dn   p(+0)DOS-dn   p(+1)DOS-dn".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            angular_limit: 1,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
                ],
                [
                    1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
                ],
                [
                    1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
                ]
            ],
        }
    }

    fn sample_rhocm_dat() -> LdosMagneticDatData {
        LdosMagneticDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            angular_limit: 1,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    9.0E-4, 8.0E-4, 7.0E-4, 6.0E-4, 5.0E-4, 4.0E-4, 3.0E-4, 2.0E-4
                ],
                [
                    9.1E-4, 8.1E-4, 7.1E-4, 6.1E-4, 5.1E-4, 4.1E-4, 3.1E-4, 2.1E-4
                ],
                [
                    9.2E-4, 8.2E-4, 7.2E-4, 6.2E-4, 5.2E-4, 4.2E-4, 3.2E-4, 2.2E-4
                ]
            ],
        }
    }

    fn sample_hubbard_gtr_m_source_contract(
        angular_limit: usize,
        energy_count: usize,
        potential_count: usize,
    ) -> HubbardLdosGtrMBinData {
        let angular_count = angular_limit + 1;
        let magnetic_count = angular_count * angular_count;
        HubbardLdosGtrMBinData {
            point_count_declared: energy_count,
            horizontal_count: energy_count,
            danes_extension_count: 0,
            highest_potential_index: potential_count.saturating_sub(1),
            fms_mode: 2,
            angular_limit,
            values: Array5::from_elem(
                (
                    2,
                    energy_count,
                    potential_count,
                    angular_count,
                    magnetic_count,
                ),
                Complex32::new(0.0, 0.0),
            ),
        }
    }

    fn sample_hubbard_gtr_source_contract(
        angular_limit: usize,
        energy_count: usize,
        potential_count: usize,
    ) -> HubbardLdosGtrBinData {
        let angular_count = angular_limit + 1;
        HubbardLdosGtrBinData {
            point_count_declared: energy_count,
            horizontal_count: energy_count,
            danes_extension_count: 0,
            highest_potential_index: potential_count.saturating_sub(1),
            fms_mode: 2,
            angular_limit,
            values: Array4::from_elem(
                (2, energy_count, potential_count, angular_count),
                Complex32::new(0.0, 0.0),
            ),
        }
    }

    fn sample_hubbard_gtr_off_source_contract(
        hubbard_l: usize,
        angular_limit: usize,
        energy_count: usize,
        potential_count: usize,
    ) -> HubbardLdosGtrOffBinData {
        let angular_count = angular_limit + 1;
        let order = (hubbard_l + 1) * (hubbard_l + 1);
        HubbardLdosGtrOffBinData {
            point_count_declared: energy_count,
            horizontal_count: energy_count,
            danes_extension_count: 0,
            highest_potential_index: potential_count.saturating_sub(1),
            fms_mode: 2,
            hubbard_l,
            angular_limit,
            values: Array6::from_elem(
                (
                    angular_count,
                    2,
                    energy_count,
                    potential_count,
                    order,
                    order,
                ),
                Complex32::new(0.0, 0.0),
            ),
        }
    }

    fn sample_spin_rhoc_dat() -> LdosDatData {
        LdosDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
                ],
                [
                    1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
                ],
                [
                    1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
                ]
            ],
        }
    }

    fn sample_spin_ldos_dat() -> LdosDatData {
        LdosDatData {
            header_lines: vec![
                "#  Fermi level (eV):  -3.777".to_string(),
                "#      e        sDOS(up)   pDOS(up)      dDOS(up)    fDOS(up)   sDOS(down)    pDOS(down)   dDOS(down)   fDOS(down)    @#".to_string(),
            ],
            fermi_level_ev: Some(-3.777),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: array![-1.0, 0.0, 1.0],
            density: array![
                [
                    1.0E-4, 2.0E-4, 3.0E-4, 4.0E-4, 5.0E-4, 6.0E-4, 7.0E-4, 8.0E-4
                ],
                [
                    1.1E-4, 2.1E-4, 3.1E-4, 4.1E-4, 5.1E-4, 6.1E-4, 7.1E-4, 8.1E-4
                ],
                [
                    1.2E-4, 2.2E-4, 3.2E-4, 4.2E-4, 5.2E-4, 6.2E-4, 7.2E-4, 8.2E-4
                ]
            ],
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                " Calculating LDOS ...".to_string(),
                "FEFF-serial using 1 thread.".to_string(),
                " Done with LDOS.".to_string(),
                "Done with module: LDOS.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 4],
        }
    }

    struct LdosSourceReferenceCase {
        label: &'static str,
        source_dir: PathBuf,
        expected_dir: PathBuf,
        potential_count: usize,
    }

    fn workspace_root() -> Result<PathBuf> {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .context("failed to find workspace root")
    }

    fn reference_ldos_dir() -> Result<Option<PathBuf>> {
        let workspace = workspace_root()?;
        let path = workspace.join("reference-work/golden/EXAFS/Cu");
        Ok(reference_ldos_expected_present(&path, 2).then_some(path))
    }

    fn reference_ldos_nonzero_fms_dir() -> Result<Option<PathBuf>> {
        let workspace = workspace_root()?;
        let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_fms_short");
        Ok((reference_ldos_expected_present(&path, 2)
            && reference_ldos_source_grid_present(&path)
            && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
        .then_some(path))
    }

    fn reference_ldos_production_fms_dir() -> Result<Option<PathBuf>> {
        let workspace = workspace_root()?;
        let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_fms");
        Ok((reference_ldos_expected_present(&path, 2)
            && reference_ldos_source_grid_present(&path)
            && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
        .then_some(path))
    }

    fn reference_ldos_ordinary_spin_fms_dir() -> Result<Option<PathBuf>> {
        let workspace = workspace_root()?;
        let path = workspace.join("reference-work/golden/LDOS/XANES_Cu_spin_fms_short");
        Ok((reference_ldos_expected_present(&path, 2)
            && reference_ldos_source_grid_present(&path)
            && path.join("xsph.inp").is_file()
            && (0..2).all(|potential| path.join(format!("gtr{potential:02}.bin")).is_file()))
        .then_some(path))
    }

    fn reference_hubbard_nio_ldos_zip() -> Result<Option<PathBuf>> {
        let workspace = workspace_root()?;
        let path = workspace.join("reference-work/golden/HUBBARD/NiO/REFERENCE.zip");
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

    fn reference_ldos_source_cases() -> Result<Vec<LdosSourceReferenceCase>> {
        let workspace = workspace_root()?;
        let mut cases = Vec::new();
        let exafs_cu = workspace.join("reference-work/golden/EXAFS/Cu");
        if reference_ldos_expected_present(&exafs_cu, 2) && reference_ldos_source_present(&exafs_cu)
        {
            cases.push(LdosSourceReferenceCase {
                label: "EXAFS/Cu no-FMS",
                source_dir: exafs_cu.clone(),
                expected_dir: exafs_cu,
                potential_count: 2,
            });
        }

        let xanes_cu_source = workspace.join("reference-work/golden/XANES/Cu");
        let xanes_cu_expected = workspace.join("reference-work/golden/LDOS/XANES_Cu_no_fms");
        if reference_ldos_expected_present(&xanes_cu_expected, 2)
            && reference_ldos_source_present(&xanes_cu_source)
        {
            cases.push(LdosSourceReferenceCase {
                label: "XANES/Cu source with no-FMS LDOS card",
                source_dir: xanes_cu_source.clone(),
                expected_dir: xanes_cu_expected,
                potential_count: 2,
            });
        }

        let xanes_cu_spin_expected =
            workspace.join("reference-work/golden/LDOS/XANES_Cu_spin_no_fms");
        if reference_ldos_expected_present(&xanes_cu_spin_expected, 2)
            && reference_ldos_source_present(&xanes_cu_source)
        {
            cases.push(LdosSourceReferenceCase {
                label: "XANES/Cu source with ordinary spin no-FMS LDOS card",
                source_dir: xanes_cu_source,
                expected_dir: xanes_cu_spin_expected,
                potential_count: 2,
            });
        }

        let gecl4_source = workspace.join("reference-work/golden/NRIXS/GeCl_4");
        let gecl4_production_expected = workspace.join("reference-work/golden/LDOS/GeCl4_no_fms");
        if reference_ldos_expected_present(&gecl4_production_expected, 2)
            && reference_ldos_source_present(&gecl4_source)
        {
            cases.push(LdosSourceReferenceCase {
                label: "NRIXS/GeCl4 source with production no-FMS LDOS card",
                source_dir: gecl4_source.clone(),
                expected_dir: gecl4_production_expected,
                potential_count: 2,
            });
        }

        let gecl4_expected = workspace.join("reference-work/golden/LDOS/GeCl4_no_fms_short");
        if reference_ldos_expected_present(&gecl4_expected, 2)
            && reference_ldos_source_present(&gecl4_source)
        {
            cases.push(LdosSourceReferenceCase {
                label: "NRIXS/GeCl4 source with short no-FMS LDOS card",
                source_dir: gecl4_source,
                expected_dir: gecl4_expected,
                potential_count: 2,
            });
        }

        Ok(cases)
    }

    fn reference_ldos_source_present(path: &Path) -> bool {
        [
            "pot.bin",
            "config.dat",
            "phase.bin",
            "pot.inp",
            "fms.inp",
            "global.inp",
        ]
        .iter()
        .all(|name| path.join(name).is_file())
    }

    fn reference_ldos_source_grid_present(path: &Path) -> bool {
        reference_ldos_source_present(path)
            && path.join("geom.dat").is_file()
            && path.join(".dimensions.dat").is_file()
    }

    fn reference_ldos_expected_present(path: &Path, potential_count: usize) -> bool {
        path.join("ldos.inp").is_file()
            && (0..potential_count).all(|potential| {
                path.join(format!("ldos{potential:02}.dat")).is_file()
                    && path.join(format!("rhoc{potential:02}.dat")).is_file()
            })
    }

    fn assert_ldos_table_mesh_matches(actual: &LdosDatData, expected: &LdosDatData) {
        assert_eq!(actual.fermi_level_ev, expected.fermi_level_ev);
        assert_eq!(actual.charge_transfer, expected.charge_transfer);
        assert_eq!(actual.electron_counts, expected.electron_counts);
        assert_eq!(actual.atom_count, expected.atom_count);
        assert_eq!(actual.lorentzian_hwhh_ev, expected.lorentzian_hwhh_ev);
        assert_eq!(actual.energy_ev.len(), expected.energy_ev.len());
        assert_eq!(actual.density.dim(), expected.density.dim());
        for (actual, expected) in actual.energy_ev.iter().zip(expected.energy_ev.iter()) {
            assert!(
                (actual - expected).abs() <= 5.0e-4,
                "energy actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }

    fn assert_ldos_density_grid_close(actual: &LdosDatData, expected: &LdosDatData, label: &str) {
        let abs_tolerance = 5.0e-5;
        let rel_tolerance = 1.5e-3;
        for ((row, column), actual) in actual.density.indexed_iter() {
            let expected = expected.density[(row, column)];
            let diff = (actual - expected).abs();
            let rel = diff / expected.abs().max(1.0e-30);
            assert!(
                diff <= abs_tolerance || rel <= rel_tolerance,
                "{label}: density[{row},{column}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
            );
        }
    }

    fn assert_ldos_magnetic_table_close(
        actual: &LdosMagneticDatData,
        expected: &LdosMagneticDatData,
        label: &str,
    ) {
        assert_eq!(actual.fermi_level_ev, expected.fermi_level_ev);
        assert_eq!(actual.charge_transfer, expected.charge_transfer);
        assert_eq!(actual.electron_counts, expected.electron_counts);
        assert_eq!(actual.atom_count, expected.atom_count);
        assert_eq!(actual.lorentzian_hwhh_ev, expected.lorentzian_hwhh_ev);
        assert_eq!(actual.angular_limit, expected.angular_limit);
        assert_eq!(actual.energy_ev.len(), expected.energy_ev.len());
        assert_eq!(actual.density.dim(), expected.density.dim());
        for (actual, expected) in actual.energy_ev.iter().zip(expected.energy_ev.iter()) {
            assert!(
                (actual - expected).abs() <= 5.0e-4,
                "{label}: energy actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
        let abs_tolerance = 5.0e-9;
        let rel_tolerance = 1.0e-5;
        for ((row, column), actual) in actual.density.indexed_iter() {
            let expected = expected.density[(row, column)];
            let diff = (actual - expected).abs();
            let rel = diff / expected.abs().max(1.0e-30);
            assert!(
                diff <= abs_tolerance || rel <= rel_tolerance,
                "{label}: magnetic density[{row},{column}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
            );
        }
    }

    fn assert_zero_gtr_reference_compatible(actual: &GtrBinData, expected: &GtrBinData) {
        assert_eq!(actual.energy_count(), expected.energy_count());
        assert_eq!(actual.horizontal_count, expected.horizontal_count);
        assert_eq!(actual.danes_extension_count, expected.danes_extension_count);
        assert_eq!(
            actual.highest_potential_index,
            expected.highest_potential_index
        );
        assert_eq!(actual.fms_mode, expected.fms_mode);
        assert!(actual.angular_channel_count() >= expected.angular_channel_count());
        assert!(expected.values.iter().all(|value| value.norm() == 0.0));
        assert!(actual.values.iter().all(|value| value.norm() == 0.0));
    }

    fn assert_gtr_reference_close(actual: &GtrBinData, expected: &GtrBinData, label: &str) {
        assert_eq!(actual.energy_count(), expected.energy_count(), "{label}");
        assert_eq!(
            actual.horizontal_count, expected.horizontal_count,
            "{label}"
        );
        assert_eq!(
            actual.danes_extension_count, expected.danes_extension_count,
            "{label}"
        );
        assert_eq!(
            actual.highest_potential_index, expected.highest_potential_index,
            "{label}"
        );
        assert_eq!(actual.fms_mode, expected.fms_mode, "{label}");
        assert_eq!(actual.values.dim(), expected.values.dim(), "{label}");
        assert!(
            expected.values.iter().any(|value| value.norm() > 0.0),
            "{label}: reference trace is zero"
        );
        assert!(
            actual.values.iter().any(|value| value.norm() > 0.0),
            "{label}: generated trace is zero"
        );

        let abs_tolerance = 1.0e-4;
        let rel_tolerance = 2.0e-3;
        for ((energy, potential, angular), actual) in actual.values.indexed_iter() {
            let expected = expected.values[(energy, potential, angular)];
            let diff = (*actual - expected).norm();
            let rel = diff / expected.norm().max(1.0e-30);
            assert!(
                diff <= abs_tolerance || rel <= rel_tolerance,
                "{label}: gtr[{energy},{potential},{angular}] actual={actual}, expected={expected}, diff={diff}, rel={rel}"
            );
        }
    }
}
