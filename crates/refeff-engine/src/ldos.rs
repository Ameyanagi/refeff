use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayView1, Axis};
use num_complex::{Complex32, Complex64};
use refeff_core::{
    FEFF_HARTREE_EV, LdosHubbardStep1Input, LdosRholWavefunctionTablesInput, ldos_hubbard_step1,
    ldos_rhol_wavefunction_tables,
};
use refeff_io::{
    DimensionsDat, FmsInput, GeomDat, HubbardInput, HubbardTransformationBinData,
    HubbardVnlmBinData, LdosDatData, LdosDatFromFf2rhoInput, LdosElectronCount, LdosInput,
    LdosMagneticDatData, LdosMagneticDatFromFf2rhoInput, LdosSpinDatFromFf2rhoInput, ModuleLogData,
    PotBinData, PotInput, RhocDatData, XsphInput, gtr_bin_ldos_trace_handoff,
    hubbard_ldos_gtr_m_trace_handoff, ldos_dat_from_ff2rho, ldos_magnetic_dat_from_ff2rho,
    ldos_spin_dat_from_ff2rho, read_config_dat, read_gtr_bin, read_hubbard_ldos_gtr_bin_inferred,
    read_hubbard_ldos_gtr_m_bin_inferred, read_hubbard_ldos_gtr_off_bin, read_ldos_dat,
    read_lmdos_dat, read_module_log_dat, read_phase_bin, read_pot_bin, read_rhoc_dat,
    read_rhocm_dat, read_v_hubbard_bin_inferred, rhorrp_geom_handoff_from_geom_dat, write_ldos_dat,
    write_lmdos_dat, write_module_log_dat, write_rhoc_dat, write_rhocm_dat,
    write_transformation_hubbard_bin, write_v_hubbard_bin,
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
/// embedded-density tables. As in FEFF, `lfms2 == 0` runs an independent FMS
/// solve centered on each potential when the geometry handoffs are present;
/// incomplete legacy handoffs retain the embedded-density-only fallback.
/// Missing or incomplete source state reports a normal source requirement.
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
        write_missing_or_unusable_magnetic_from_hubbard_handoffs(&input, work_dir, &tables)
            .context(LDOS_SOURCE_REQUIREMENT_ERROR)?;
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

pub(crate) fn read_input(work_dir: &Path) -> Result<LdosInput> {
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
    if !active_hubbard_ldos_enabled(work_dir)? {
        return Ok(0);
    }
    let magnetic_cache_compatible =
        active_hubbard_magnetic_outputs_complete(work_dir, tables, input, false)?;
    // Discovery can validate an algebraically recoverable pair even if one
    // file is missing or damaged. Only skip writes when both real files parse.
    let magnetic_files_present = cached_ldos_output_indices(tables).into_iter().all(|index| {
        read_lmdos_dat(work_dir.join(format!("lmdos{index}.dat"))).is_ok()
            && read_rhocm_dat(work_dir.join(format!("rhocm{index}.dat"))).is_ok()
    });
    if magnetic_files_present && magnetic_cache_compatible {
        return Ok(0);
    }
    if input.control.lfms2 != 0 {
        return write_full_fms_magnetic_from_hubbard_handoffs(input, work_dir, tables);
    }

    // In FEFF's independent-center (`lfms2=0`) mode a one-sided cached pair
    // can still be reconstructed algebraically, but a fresh calculation has
    // neither magnetic table. It must execute the same two-pass Hubbard
    // workflow, with the first FMS pass solved once per central potential.
    let needs_fresh_pair = cached_ldos_output_indices(tables).into_iter().any(|index| {
        read_lmdos_dat(work_dir.join(format!("lmdos{index}.dat"))).is_err()
            && read_rhocm_dat(work_dir.join(format!("rhocm{index}.dat"))).is_err()
    });
    if needs_fresh_pair {
        return write_full_fms_magnetic_from_hubbard_handoffs(input, work_dir, tables);
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

fn write_full_fms_magnetic_from_hubbard_handoffs(
    input: &LdosInput,
    work_dir: &Path,
    tables: &[CachedTable],
) -> Result<usize> {
    let v_hubbard_path = work_dir.join("v_hubbard.bin");
    if !v_hubbard_path.is_file()
        && (!work_dir.join("gtr_m00.bin").is_file() || !work_dir.join("gtr_off00.bin").is_file())
    {
        crate::fms::write_hubbard_ldos_first_pass_traces(work_dir, input)
            .context("failed to generate active Hubbard first-pass FMS traces")?;
    }
    let generated_first_pass =
        !v_hubbard_path.is_file() && generate_hubbard_step1_handoffs(input, work_dir)?;
    if generated_first_pass {
        // The input gtr_m/gtr_off files belong to FEFF's first FMS pass.
        // Consume the generated Vnlm/TFrm through XSPH and FMS before using
        // gtr_m again, because only that second trace is valid for final tables.
        crate::xsph::write_hubbard_phase_on_ldos_grid(
            work_dir,
            ldos_input_energy_grid_hartree(input)?,
        )
        .context("failed to generate active Hubbard second-pass phase shifts")?;
        with_transient_active_hubbard_gtr_sources(work_dir, input.lmaxph.len(), || {
            let fms_count = crate::fms::run_fms_in_dir(work_dir)
                .context("failed to generate active Hubbard second-pass FMS magnetic trace")?;
            let mkgtr_count = crate::fms::run_mkgtr_in_dir(work_dir)
                .context("failed to rebuild MKGTR outputs after active Hubbard FMS refresh")?;
            let independent_trace_count =
                crate::fms::write_hubbard_ldos_independent_second_pass_trace(work_dir, input)
                    .context(
                        "failed to generate active Hubbard independent second-pass magnetic trace",
                    )?;
            Ok(fms_count + mkgtr_count + independent_trace_count)
        })?;
    }
    if !v_hubbard_path.is_file() {
        return Ok(0);
    }
    let potential_count = input.lmaxph.len();
    let v_hubbard = read_v_hubbard_bin_inferred(&v_hubbard_path, potential_count)
        .with_context(|| format!("failed to read {}", v_hubbard_path.display()))?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let requested = cached_ldos_output_indices(tables)
        .into_iter()
        .filter_map(|index| index.parse::<usize>().ok())
        .collect::<Vec<_>>();
    if requested.is_empty() {
        return Ok(0);
    }
    let hubbard = read_hubbard_input_optional(work_dir)?
        .context("active Hubbard magnetic LDOS generation requires hubbard.inp")?;
    let hubbard_angular_count = usize::try_from(hubbard.l)
        .context("hubbard.inp l_hubbard must be nonnegative")?
        .checked_add(1)
        .context("hubbard.inp l_hubbard is too large")?;
    let angular_count = LDOS_ORBITAL_COUNT
        .min(v_hubbard.angular_count())
        .min(hubbard_angular_count);
    let radial_sources = crate::rhorrp::read_hubbard_ldos_rhol_source_tables(
        work_dir,
        energy_grid.clone(),
        &requested,
        angular_count,
        &v_hubbard,
    )?;
    let metadata = read_ldos_source_metadata(work_dir, potential_count, LDOS_ORBITAL_COUNT)?;
    let mut written = 0;

    for source in radial_sources {
        let potential = source.potential_index;
        let index = format!("{potential:02}");
        let specific_trace_path = work_dir.join(format!("gtr_m{index}.bin"));
        let fallback_trace_path = work_dir.join("gtr_m00.bin");
        let trace_path = if specific_trace_path.is_file() {
            specific_trace_path
        } else {
            fallback_trace_path
        };
        let trace_source = read_hubbard_ldos_gtr_m_bin_inferred(&trace_path)
            .with_context(|| format!("failed to read {}", trace_path.display()))?;
        let trace =
            hubbard_ldos_gtr_m_trace_handoff(&trace_source, potential).with_context(|| {
                format!(
                    "failed to select potential {potential} from {}",
                    trace_path.display()
                )
            })?;
        if trace.energy_count != energy_grid.len()
            || trace.angular_count < angular_count
            || trace.magnetic_count < angular_count * angular_count
        {
            bail!(
                "{} dimensions energy={} angular={} magnetic={} do not cover Hubbard LDOS energy={} angular={} magnetic={}",
                trace_path.display(),
                trace.energy_count,
                trace.angular_count,
                trace.magnetic_count,
                energy_grid.len(),
                angular_count,
                angular_count * angular_count
            );
        }
        let trace = trace
            .trace
            .slice_axis(ndarray::Axis(0), ndarray::Slice::from(..angular_count))
            .slice_axis(
                ndarray::Axis(1),
                ndarray::Slice::from(..angular_count * angular_count),
            )
            .to_owned();
        let source_metadata = metadata.get(potential).cloned().unwrap_or_default();
        let handoff = ldos_magnetic_dat_from_ff2rho(LdosMagneticDatFromFf2rhoInput {
            header_lines: &[],
            fermi_level_hartree: read_phase_bin(work_dir.join("phase.bin"))
                .ok()
                .map(|phase| phase.scalars.fermi_level),
            charge_transfer: source_metadata.charge_transfer,
            electron_counts: &source_metadata.electron_counts,
            atom_count: source_metadata.atom_count,
            lorentzian_hwhh_hartree: energy_grid.first().map(|energy| energy.im),
            energy_grid_hartree: energy_grid.view(),
            embedded_magnetic_ldos: source.embedded_magnetic_ldos.view(),
            scattering_magnetic_ldos: source.scattering_magnetic_ldos.view(),
            magnetic_scattering_trace: trace.view(),
            angular_count,
        })
        .with_context(|| {
            format!("failed to build Hubbard magnetic LDOS tables for potential {potential}")
        })?;
        let lmdos_path = work_dir.join(format!("lmdos{index}.dat"));
        if !ldos_magnetic_table_matches_source_output(&lmdos_path, &handoff.lmdos, true) {
            write_lmdos_dat(&lmdos_path, &handoff.lmdos)
                .with_context(|| format!("failed to write {}", lmdos_path.display()))?;
            written += 1;
        }
        let rhocm_path = work_dir.join(format!("rhocm{index}.dat"));
        if !ldos_magnetic_table_matches_source_output(&rhocm_path, &handoff.rhocm, false) {
            write_rhocm_dat(&rhocm_path, &handoff.rhocm)
                .with_context(|| format!("failed to write {}", rhocm_path.display()))?;
            written += 1;
        }
    }
    Ok(written)
}

fn generate_hubbard_step1_handoffs(input: &LdosInput, work_dir: &Path) -> Result<bool> {
    let Some(hubbard) = read_hubbard_input_optional(work_dir)? else {
        return Ok(false);
    };
    let hubbard_l =
        usize::try_from(hubbard.l).context("hubbard.inp l_hubbard must be nonnegative")?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let phase_path = work_dir.join("phase.bin");
    let phase = read_phase_bin(&phase_path)
        .with_context(|| format!("failed to read {}", phase_path.display()))?;
    // FEFF sizes the Hubbard handoff arrays with the compiled `DimsMod::lx`
    // capacity recorded by `rdinp` in `.dimensions.dat`.  The radial source
    // may expose an extra channel, but that channel is outside the active
    // Hubbard/LDOS contract for smaller-lx builds (for example NiO uses
    // `lx=2` while the radial source carries l=0..3).
    let angular_count = ldos_angular_count(input, work_dir)?;
    let magnetic_count = angular_count * angular_count;
    let potential_count = input.lmaxph.len().min(phase.potential_count());
    let spin_sources = [
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid_for_spin(
            work_dir,
            energy_grid.clone(),
            1,
        )?,
        crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid_for_spin(
            work_dir,
            energy_grid.clone(),
            -1,
        )?,
    ];

    let mut vnlm = HubbardVnlmBinData {
        angular_limit: angular_count - 1,
        values: Array4::zeros((potential_count, 2, angular_count, magnetic_count)),
    };
    let order = 2 * hubbard_l + 1;
    let mut transformation = HubbardTransformationBinData {
        hubbard_l,
        angular_limit: angular_count - 1,
        transform: ndarray::Array5::zeros((potential_count, 2, angular_count, order, order)),
        inverse: ndarray::Array5::zeros((potential_count, 2, angular_count, order, order)),
    };

    for potential in 0..potential_count {
        let mut embedded = Array3::<f64>::zeros((angular_count, 2, energy_grid.len()));
        let mut scattering = Array3::<Complex64>::zeros((angular_count, 2, energy_grid.len()));
        for (spin, source) in spin_sources.iter().enumerate() {
            let zero_trace = Array2::zeros((angular_count, energy_grid.len()));
            let solved = ldos_rhol_wavefunction_tables(LdosRholWavefunctionTablesInput {
                wavefunctions: &source.wavefunctions.wavefunctions,
                radii: source.wavefunctions.prepared.radii.view(),
                potential_index: potential,
                energy_grid_hartree: energy_grid.view(),
                scattering_trace: zero_trace.view(),
                radial_step: source.wavefunctions.radial_dx,
                norman_radius: source.norman_radii_bohr[potential],
                angular_count,
                apply_scattering: false,
            })
            .with_context(|| {
                format!(
                    "failed to solve Hubbard first-pass ordinary radial tables for potential {potential}, spin {}",
                    spin + 1
                )
            })?;
            embedded
                .index_axis_mut(Axis(1), spin)
                .assign(&solved.density_grid.embedded_ldos);
            scattering
                .index_axis_mut(Axis(1), spin)
                .assign(&solved.density_grid.scattering_ldos);
        }

        let index = format!("{potential:02}");
        let gtr_m_path = {
            let specific = work_dir.join(format!("gtr_m{index}.bin"));
            if specific.is_file() {
                specific
            } else {
                work_dir.join("gtr_m00.bin")
            }
        };
        let gtr_off_path = {
            let specific = work_dir.join(format!("gtr_off{index}.bin"));
            if specific.is_file() {
                specific
            } else {
                work_dir.join("gtr_off00.bin")
            }
        };
        if !gtr_m_path.is_file() || !gtr_off_path.is_file() {
            return Ok(false);
        }
        let gtr_m_source = read_hubbard_ldos_gtr_m_bin_inferred(&gtr_m_path)
            .with_context(|| format!("failed to read {}", gtr_m_path.display()))?;
        let gtr_m = hubbard_ldos_gtr_m_trace_handoff(&gtr_m_source, potential)
            .with_context(|| format!("failed to select potential from {}", gtr_m_path.display()))?;
        let gtr_off_source =
            read_hubbard_ldos_gtr_off_bin(&gtr_off_path, hubbard_l, angular_count - 1)
                .with_context(|| format!("failed to read {}", gtr_off_path.display()))?;
        if gtr_m.energy_count != energy_grid.len()
            || gtr_off_source.energy_count() != energy_grid.len()
            || potential >= gtr_off_source.potential_count()
        {
            bail!(
                "Hubbard first-pass trace dimensions do not match potential {potential} LDOS mesh"
            );
        }
        let off_count = gtr_off_source.order();
        let mut off_diagonal =
            Array5::<Complex64>::zeros((angular_count, off_count, off_count, 2, energy_grid.len()));
        for angular in 0..angular_count {
            for spin in 0..2 {
                for energy in 0..energy_grid.len() {
                    for row in 0..off_count {
                        for column in 0..off_count {
                            let value = gtr_off_source.values
                                [(angular, spin, energy, potential, row, column)];
                            off_diagonal[(angular, row, column, spin, energy)] =
                                Complex64::new(value.re as f64, value.im as f64);
                        }
                    }
                }
            }
        }
        let step1 = ldos_hubbard_step1(LdosHubbardStep1Input {
            energy_grid_hartree: energy_grid.view(),
            embedded_ldos: embedded.view(),
            scattering_ldos: scattering.view(),
            magnetic_scattering_trace: gtr_m.trace.view(),
            off_diagonal_scattering_trace: off_diagonal.view(),
            chemical_potential_hartree: phase.scalars.fermi_level,
            fermi_shift_ev: hubbard.fermi_shift,
            hubbard_u_ev: hubbard.u,
            hubbard_j_ev: hubbard.j,
            hubbard_l,
            potential_index: potential,
            angular_count,
        })
        .with_context(|| format!("failed Hubbard first-pass assembly for potential {potential}"))?;
        for spin in 0..2 {
            for angular in 0..angular_count {
                for magnetic in 0..magnetic_count {
                    vnlm.values[(potential, spin, angular, magnetic)] =
                        step1.hubbard_potential[(spin, angular, magnetic)];
                }
                for row in 0..order {
                    for column in 0..order {
                        let value = step1.transform[(spin, angular, row, column)];
                        transformation.transform[(potential, spin, angular, row, column)] =
                            Complex32::new(value.re as f32, value.im as f32);
                        let value = step1.inverse_transform[(spin, angular, row, column)];
                        transformation.inverse[(potential, spin, angular, row, column)] =
                            Complex32::new(value.re as f32, value.im as f32);
                    }
                }
            }
        }
    }

    write_v_hubbard_bin(work_dir.join("v_hubbard.bin"), &vnlm)
        .context("failed to write generated v_hubbard.bin")?;
    write_transformation_hubbard_bin(work_dir.join("transformation_hubbard.bin"), &transformation)
        .context("failed to write generated transformation_hubbard.bin")?;
    Ok(true)
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

    if input.control.lfms2 == 0 && !work_dir.join("fms.inp").is_file() {
        return write_no_fms_ldos_from_rhol_source_handoffs(input, work_dir);
    }

    let Some(fms_input) = ldos_effective_fms_source_input(input, work_dir)? else {
        return if input.control.lfms2 == 0 {
            write_no_fms_ldos_from_rhol_source_handoffs(input, work_dir)
        } else {
            Ok(0)
        };
    };

    let angular_count = ldos_angular_count(input, work_dir)?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let source = crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(
        work_dir,
        energy_grid.clone(),
        angular_count,
    )?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < angular_count {
        return Ok(0);
    }
    let has_fms_source = crate::fms::has_supported_ldos_gtr_bin_source_grid_handoff(
        work_dir,
        &fms_input,
        source.phase.energies_hartree.view(),
        source.wavefunctions.wavefunctions.wave_numbers.view(),
        source.wavefunctions.wavefunctions.phase_shifts.view(),
    )?;
    if input.control.lfms2 == 0 && !has_fms_source {
        return write_no_fms_ldos_from_rhol_source_handoffs(input, work_dir);
    }
    let potential_count = source
        .phase
        .potential_count()
        .min(source.wavefunctions.wavefunctions.potential_count())
        .min(source.norman_radii_bohr.len())
        .min(input.lmaxph.len());
    let metadata_by_potential =
        read_ldos_source_metadata(work_dir, potential_count, angular_count)?;
    with_transient_active_hubbard_gtr_sources(work_dir, potential_count, || {
        let transient_gtr = active_hubbard_ldos_enabled(work_dir)?;
        let gtr_written = crate::fms::write_ldos_gtr_bin_source_grid_handoff(
            work_dir,
            &fms_input,
            source.phase.energies_hartree.view(),
            source.wavefunctions.wavefunctions.wave_numbers.view(),
            source.wavefunctions.wavefunctions.phase_shifts.view(),
        )?;
        let mut written = if transient_gtr { 0 } else { gtr_written };

        for potential in 0..potential_count {
            let ldos_path = work_dir.join(format!("ldos{potential:02}.dat"));
            let rhoc_path = work_dir.join(format!("rhoc{potential:02}.dat"));
            let ldos_valid = ldos_table_is_usable(&ldos_path);
            let rhoc_valid = rhoc_table_is_usable(&rhoc_path);

            let metadata = metadata_by_potential
                .get(potential)
                .cloned()
                .unwrap_or_default();
            let Some(scattering_trace) = ldos_source_scattering_trace(
                work_dir,
                potential,
                source.phase.energy_count(),
                angular_count,
            )?
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
                angular_count,
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
                angular_count,
                apply_scattering: true,
            })
            .with_context(|| {
                format!("failed to build LDOS dat payload for potential {potential}")
            })?;

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
    })
}

type PreservedHubbardGtrSources = Vec<(PathBuf, Option<Vec<u8>>)>;

fn with_transient_active_hubbard_gtr_sources<T>(
    work_dir: &Path,
    potential_count: usize,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let preserved = preserve_active_hubbard_gtr_sources(work_dir, potential_count)?;
    let result = operation();
    restore_active_hubbard_gtr_sources(preserved)?;
    result
}

/// Keep LDOS's internal Hubbard magnetic traces separate from the later
/// normal-spectrum FMS refresh.
///
/// FEFF's LDOS driver holds these 101-point traces in memory; the standalone
/// spectrum FMS that follows operates on its own (typically 83-point) mesh.
/// The Rust bridge serializes the LDOS traces, so preserve them while FMS
/// publishes its spectrum-side magnetic trace.  Otherwise a final cache audit
/// sees that unrelated spectrum trace and incorrectly marks the completed
/// LDOS tables stale.
pub(crate) fn with_preserved_active_hubbard_ldos_magnetic_sources<T>(
    work_dir: &Path,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !work_dir.join("ldos.inp").is_file() || !active_hubbard_ldos_enabled(work_dir)? {
        return operation();
    }

    let input = read_input(work_dir)?;
    if input.control.mldos != 1 {
        return operation();
    }
    let potential_count = input.lmaxph.len();
    let mut paths = Vec::with_capacity(2 * potential_count.max(1));
    for potential in 0..potential_count {
        for stem in ["gtr_m", "gtr_off"] {
            let path = work_dir.join(format!("{stem}{potential:02}.bin"));
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    for name in ["gtr_m00.bin", "gtr_off00.bin"] {
        let path = work_dir.join(name);
        if !paths.contains(&path) {
            paths.push(path);
        }
    }

    let mut preserved = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = if path.is_file() {
            Some(
                std::fs::read(&path)
                    .with_context(|| format!("failed to preserve {}", path.display()))?,
            )
        } else {
            None
        };
        preserved.push((path, bytes));
    }

    let result = operation();
    restore_active_hubbard_gtr_sources(Some(preserved))?;
    result
}

fn preserve_active_hubbard_gtr_sources(
    work_dir: &Path,
    potential_count: usize,
) -> Result<Option<PreservedHubbardGtrSources>> {
    if !active_hubbard_ldos_enabled(work_dir)? {
        return Ok(None);
    }

    let mut preserved = Vec::with_capacity(potential_count);
    for potential in 0..potential_count {
        let path = work_dir.join(format!("gtr{potential:02}.bin"));
        let bytes = if path.is_file() {
            Some(
                std::fs::read(&path)
                    .with_context(|| format!("failed to preserve {}", path.display()))?,
            )
        } else {
            None
        };
        preserved.push((path, bytes));
    }
    Ok(Some(preserved))
}

fn restore_active_hubbard_gtr_sources(preserved: Option<PreservedHubbardGtrSources>) -> Result<()> {
    let Some(preserved) = preserved else {
        return Ok(());
    };

    for (path, bytes) in preserved {
        if let Some(bytes) = bytes {
            std::fs::write(&path, bytes)
                .with_context(|| format!("failed to restore {}", path.display()))?;
        } else if path.is_file() {
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove transient {}", path.display()))?;
        }
    }
    Ok(())
}

fn write_no_fms_ldos_from_rhol_source_handoffs(
    input: &LdosInput,
    work_dir: &Path,
) -> Result<usize> {
    let angular_count = ldos_angular_count(input, work_dir)?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let requested_potentials = (0..input.lmaxph.len()).collect::<Vec<_>>();

    let sources = crate::rhorrp::read_no_fms_ldos_rhol_source_tables(
        work_dir,
        energy_grid.clone(),
        &requested_potentials,
        angular_count,
    )?;
    let metadata_by_potential =
        read_ldos_source_metadata(work_dir, input.lmaxph.len(), angular_count)?;
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
        let scattering_trace = Array2::zeros((angular_count, energy_grid.len()));
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
            angular_count,
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

fn ldos_magnetic_table_matches_source_output(
    path: &Path,
    source: &LdosMagneticDatData,
    is_lmdos: bool,
) -> bool {
    let cached = if is_lmdos {
        read_lmdos_dat(path)
    } else {
        read_rhocm_dat(path)
    };
    let Ok(cached) = cached else {
        return false;
    };
    if cached.angular_limit != source.angular_limit {
        return false;
    }
    let cached_as_ldos = LdosDatData {
        header_lines: Vec::new(),
        fermi_level_ev: cached.fermi_level_ev,
        charge_transfer: cached.charge_transfer,
        electron_counts: cached.electron_counts,
        atom_count: cached.atom_count,
        lorentzian_hwhh_ev: cached.lorentzian_hwhh_ev,
        energy_ev: cached.energy_ev,
        density: cached.density,
    };
    let source_as_ldos = LdosDatData {
        header_lines: Vec::new(),
        fermi_level_ev: source.fermi_level_ev,
        charge_transfer: source.charge_transfer,
        electron_counts: source.electron_counts.clone(),
        atom_count: source.atom_count,
        lorentzian_hwhh_ev: source.lorentzian_hwhh_ev,
        energy_ev: source.energy_ev.clone(),
        density: source.density.clone(),
    };
    ldos_dat_matches_source_output(&cached_as_ldos, &source_as_ldos)
}

fn ldos_dat_matches_source_output(cached: &LdosDatData, source: &LdosDatData) -> bool {
    const ENERGY_TOLERANCE_EV: f64 = 5.0e-4;
    const HEADER_THREE_DECIMAL_TOLERANCE: f64 = 5.0e-4;
    const HEADER_FOUR_DECIMAL_TOLERANCE: f64 = 5.0e-5;
    const DENSITY_ABS_TOLERANCE: f64 = 5.0e-5;
    const DENSITY_REL_TOLERANCE: f64 = 1.5e-3;

    if cached.energy_ev.len() != source.energy_ev.len()
        || cached.density.dim() != source.density.dim()
        || !optional_ldos_header_scalar_matches(
            cached.fermi_level_ev,
            source.fermi_level_ev,
            HEADER_THREE_DECIMAL_TOLERANCE,
        )
        || !optional_ldos_header_scalar_matches(
            cached.charge_transfer,
            source.charge_transfer,
            HEADER_THREE_DECIMAL_TOLERANCE,
        )
        || cached.atom_count != source.atom_count
        || !optional_ldos_header_scalar_matches(
            cached.lorentzian_hwhh_ev,
            source.lorentzian_hwhh_ev,
            HEADER_FOUR_DECIMAL_TOLERANCE,
        )
        || cached.electron_counts.len() != source.electron_counts.len()
        || !cached
            .electron_counts
            .iter()
            .zip(source.electron_counts.iter())
            .all(|(cached, source)| {
                cached.angular_momentum == source.angular_momentum
                    && (cached.count - source.count).abs() <= HEADER_THREE_DECIMAL_TOLERANCE
            })
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

fn optional_ldos_header_scalar_matches(
    cached: Option<f64>,
    source: Option<f64>,
    tolerance: f64,
) -> bool {
    match (cached, source) {
        (Some(cached), Some(source)) => (cached - source).abs() <= tolerance,
        (None, None) => true,
        _ => false,
    }
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

    let angular_count = ldos_angular_count(input, work_dir)?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    if input.control.lfms2 == 0 && !work_dir.join("fms.inp").is_file() {
        let requested_potentials = (0..input.lmaxph.len()).collect::<Vec<_>>();
        let sources = crate::rhorrp::read_no_fms_ldos_rhol_source_tables(
            work_dir,
            energy_grid.clone(),
            &requested_potentials,
            angular_count,
        )?;
        let metadata_by_potential =
            read_ldos_source_metadata(work_dir, input.lmaxph.len(), angular_count)?;

        for source in sources {
            let potential = source.potential_index;
            let metadata = metadata_by_potential
                .get(potential)
                .cloned()
                .unwrap_or_default();
            let scattering_trace = Array2::zeros((angular_count, energy_grid.len()));
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
                angular_count,
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

    let Some(fms_input) = ldos_effective_fms_source_input(input, work_dir)? else {
        return Ok(None);
    };
    let source = crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(
        work_dir,
        energy_grid,
        angular_count,
    )?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < angular_count {
        return Ok(None);
    }
    let has_fms_source = crate::fms::has_supported_ldos_gtr_bin_source_grid_handoff(
        work_dir,
        &fms_input,
        source.phase.energies_hartree.view(),
        source.wavefunctions.wavefunctions.wave_numbers.view(),
        source.wavefunctions.wavefunctions.phase_shifts.view(),
    )?;
    if input.control.lfms2 == 0 && !has_fms_source {
        let requested_potentials = (0..input.lmaxph.len()).collect::<Vec<_>>();
        let no_fms_energy_grid = source.phase.energies_hartree.clone();
        let sources = crate::rhorrp::read_no_fms_ldos_rhol_source_tables(
            work_dir,
            no_fms_energy_grid.clone(),
            &requested_potentials,
            angular_count,
        )?;
        let metadata_by_potential =
            read_ldos_source_metadata(work_dir, input.lmaxph.len(), angular_count)?;

        for source in sources {
            let potential = source.potential_index;
            let metadata = metadata_by_potential
                .get(potential)
                .cloned()
                .unwrap_or_default();
            let scattering_trace = Array2::zeros((angular_count, no_fms_energy_grid.len()));
            let handoff = ldos_dat_from_ff2rho(LdosDatFromFf2rhoInput {
                header_lines: &[],
                fermi_level_hartree: Some(source.chemical_potential_hartree),
                charge_transfer: metadata.charge_transfer,
                electron_counts: &metadata.electron_counts,
                atom_count: metadata.atom_count,
                lorentzian_hwhh_hartree: no_fms_energy_grid.first().map(|energy| energy.im),
                energy_grid_hartree: no_fms_energy_grid.view(),
                embedded_ldos: source.table.density_grid.embedded_ldos.view(),
                scattering_ldos: source.table.density_grid.scattering_ldos.view(),
                scattering_trace: scattering_trace.view(),
                angular_count,
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
    let potential_count = source
        .phase
        .potential_count()
        .min(source.wavefunctions.wavefunctions.potential_count())
        .min(source.norman_radii_bohr.len())
        .min(input.lmaxph.len());
    let metadata_by_potential =
        read_ldos_source_metadata(work_dir, potential_count, angular_count)?;

    for potential in 0..potential_count {
        let metadata = metadata_by_potential
            .get(potential)
            .cloned()
            .unwrap_or_default();
        let Some(scattering_trace) = ldos_source_scattering_trace(
            work_dir,
            potential,
            source.phase.energy_count(),
            angular_count,
        )?
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
            angular_count,
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
            angular_count,
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
    work_dir: &Path,
    potential: usize,
    energy_count: usize,
    angular_count: usize,
) -> Result<Option<Array2<Complex64>>> {
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
    let available_angular_count = gtr.angular_channel_count().min(angular_count);
    let trace = gtr_bin_ldos_trace_handoff(&gtr, potential, available_angular_count)
        .with_context(|| format!("failed to select LDOS trace from {}", gtr_path.display()))?;
    if trace.energy_count != energy_count {
        bail!(
            "{} energy count {} does not match phase.bin energy count {}",
            gtr_path.display(),
            trace.energy_count,
            energy_count
        );
    }
    if available_angular_count == angular_count {
        return Ok(Some(trace.trace));
    }

    // Missing higher-l FMS channels carry a zero multiple-scattering
    // correction through FEFF's active runtime `lx` table width.
    let mut padded = Array2::zeros((angular_count, energy_count));
    for angular in 0..available_angular_count {
        padded
            .index_axis_mut(Axis(0), angular)
            .assign(&trace.trace.index_axis(Axis(0), angular));
    }
    Ok(Some(padded))
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
    angular_count: usize,
) -> Result<Vec<LdosSourceMetadata>> {
    let pot_path = work_dir.join("pot.bin");
    let pot = read_pot_bin(&pot_path)
        .with_context(|| format!("failed to read {}", pot_path.display()))?;
    let atom_count = read_optional_ldos_atom_count(work_dir, pot.potential_count())?;
    Ok((0..potential_count)
        .map(|potential| {
            ldos_source_metadata_from_pot_bin(&pot, potential, atom_count, angular_count)
        })
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
    angular_count: usize,
) -> LdosSourceMetadata {
    let charge_transfer = pot.norman_charges.get(potential).copied();
    let electron_counts = (0..angular_count)
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

fn ldos_angular_count(input: &LdosInput, work_dir: &Path) -> Result<usize> {
    let dimensions_path = work_dir.join(".dimensions.dat");
    let angular_count = if dimensions_path.is_file() {
        let text = std::fs::read_to_string(&dimensions_path)
            .with_context(|| format!("failed to read {}", dimensions_path.display()))?;
        let dimensions = DimensionsDat::parse_str(&dimensions_path, &text)
            .with_context(|| format!("failed to parse {}", dimensions_path.display()))?;
        usize::try_from(dimensions.lx)
            .context(".dimensions.dat lx must be nonnegative")?
            .checked_add(1)
            .context(".dimensions.dat lx is too large")?
    } else {
        input
            .lmaxph
            .iter()
            .copied()
            .max()
            .context("ldos.inp requires at least one lmaxph value")
            .and_then(|lmax| usize::try_from(lmax).context("ldos.inp lmaxph must be nonnegative"))?
            .checked_add(1)
            .context("ldos.inp lmaxph is too large")?
    };
    if !matches!(angular_count, 3 | 4) {
        bail!("ordinary LDOS output supports FEFF lx=2 or lx=3, got angular count {angular_count}");
    }
    Ok(angular_count)
}

fn supported_ldos_wavefunction_source_controls(input: &LdosInput) -> bool {
    input.control.mldos == 1
        && input.ldostype <= 0
        && !input.lmaxph.is_empty()
        && input.lmaxph.iter().all(|&lmax| lmax >= 0)
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
    let angular_count = ldos_angular_count(input, work_dir)?;
    let energy_grid = ldos_input_energy_grid_hartree(input)?;
    let source = crate::rhorrp::read_ldos_wavefunction_source_on_energy_grid(
        work_dir,
        energy_grid,
        angular_count,
    )?;
    if source.wavefunctions.wavefunctions.angular_momentum_count() < angular_count {
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

/// Whether this run still needs FEFF's active-Hubbard spectrum refresh.
///
/// The Rust LDOS bridge currently consumes an ordinary XSPH/FMS bootstrap
/// phase, then creates `v_hubbard.bin` while executing the two LDOS passes.
/// FEFF subsequently reruns the normal XSPH/FMS spectrum stages with that
/// Hubbard source.  The scheduler snapshots this boundary before its first
/// XSPH pass so it can reproduce the final active-spectrum refresh after
/// LDOS without rerunning it for an already active cache.
pub(crate) fn active_hubbard_spectrum_bootstrap_pending(work_dir: &Path) -> Result<bool> {
    if work_dir.join("v_hubbard.bin").is_file()
        || !work_dir.join("ldos.inp").is_file()
        || !work_dir.join("xsph.inp").is_file()
        || !active_hubbard_ldos_enabled(work_dir)?
    {
        return Ok(false);
    }

    Ok(read_input(work_dir)?.control.mldos == 1)
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
        angular_count: embedded_ldos.nrows(),
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
    let active_hubbard = match active_hubbard_ldos_enabled(work_dir) {
        Ok(active) => active,
        Err(_) => return LdosSourceContract::Incompatible,
    };
    if active_hubbard {
        let Ok(source) = read_hubbard_ldos_gtr_bin_inferred(&path) else {
            // Active-Hubbard gtr traces are required to contain both spin
            // blocks. Do not reinterpret a truncated one-spin payload as an
            // ordinary GTR source with twice as many angular channels.
            return LdosSourceContract::Incompatible;
        };
        return hubbard_ldos_ordinary_source_contract(&source, potential);
    }
    if let Ok(source) = read_gtr_bin(&path)
        && source.angular_channel_count() <= LDOS_ORBITAL_COUNT
    {
        let Ok(input) = read_input(work_dir) else {
            return LdosSourceContract::Incompatible;
        };
        let Ok(density_column_count) = ldos_angular_count(&input, work_dir) else {
            return LdosSourceContract::Incompatible;
        };
        if potential >= source.potential_count() {
            return LdosSourceContract::Incompatible;
        }
        return LdosSourceContract::Present(LdosOrdinarySourceContract {
            energy_count: source.energy_count(),
            angular_count: source.angular_channel_count(),
            density_column_count,
        });
    }

    let Ok(source) = read_hubbard_ldos_gtr_bin_inferred(&path) else {
        return LdosSourceContract::Absent;
    };
    hubbard_ldos_ordinary_source_contract(&source, potential)
}

fn hubbard_ldos_ordinary_source_contract(
    source: &refeff_io::HubbardLdosGtrBinData,
    potential: usize,
) -> LdosSourceContract<LdosOrdinarySourceContract> {
    if potential >= source.potential_count() {
        return LdosSourceContract::Incompatible;
    }
    let density_column_count = 2 * source.angular_count().min(LDOS_ORBITAL_COUNT);
    LdosSourceContract::Present(LdosOrdinarySourceContract {
        energy_count: source.energy_count(),
        // FEFF's Hubbard gtr stores two spin blocks. The ordinary LDOS table
        // exposes those as a spin-resolved angular layout, so keep the
        // effective column count here for comparison with gtr_m.
        angular_count: density_column_count,
        density_column_count,
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
        && data.angular_limit <= source.angular_limit
        && data.magnetic_columns_per_spin() <= source.magnetic_count
}

fn ldos_hubbard_source_contracts_match(
    ordinary: LdosOrdinarySourceContract,
    magnetic: LdosMagneticSourceContract,
) -> bool {
    let Some(magnetic_angular_count) = magnetic.angular_limit.checked_add(1) else {
        return false;
    };
    ordinary.energy_count == magnetic.energy_count
        && (ordinary.angular_count <= magnetic_angular_count
            || ordinary.angular_count == 2 * magnetic_angular_count)
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
    ordinary.energy_count == offdiag.energy_count
        && (ordinary.angular_count == offdiag.angular_count
            || ordinary.angular_count == 2 * offdiag.angular_count)
}

fn ldos_hubbard_offdiag_source_contract_matches_magnetic(
    magnetic: LdosMagneticSourceContract,
    offdiag: LdosOffdiagSourceContract,
) -> bool {
    magnetic.energy_count == offdiag.energy_count
        && magnetic
            .angular_limit
            .checked_add(1)
            .is_some_and(|magnetic_angular_count| offdiag.angular_count <= magnetic_angular_count)
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
mod tests;
