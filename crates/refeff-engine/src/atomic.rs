use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, concatenate};
use num_complex::{Complex32, Complex64};
use refeff_core::{
    AtomicCoulombCoefficientInput, AtomicDifferentialIntegralInput, AtomicDifferentialIntegralKind,
    AtomicFormFactor, AtomicFormFactorInput, AtomicOverlapAmplitudeReductionInput,
    AtomicTabulationInput, AtomicTotalEnergyRadialInput, BroydenWorkspace, CoulombUpdateMode,
    DensityError, FEFF_FERMI_MOMENTUM_FACTOR, FEFF_HARTREE_EV, FEFF_KAPPA_PROJECTION_COUNT,
    FEFF_ORBITAL_KAPPAS, FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS, FEFF_ORBITAL_SLOT_COUNT,
    FeffConfigurationRecipe, FeffDefaultConfigurationRows, FermiLevelInput, GridError,
    InterstitialShellValuesInput, MuffinTinInterstitialParameters,
    MuffinTinInterstitialParametersInput, MuffinTinOverlapNeighbor, MuffinTinRadiusParametersInput,
    NormanRadiusInput, OrbitalConfiguration, OrbitalConfigurationInput, OverlapDensityIndicesInput,
    PotScfContourRun, PotScfContourRunInput, PotScfContourRunStatus, PotScfContourSourceRows,
    PotScfContourSourceRowsInput, PotScfIteration, PotScfIterationStatus,
    PotScfOuterIterationInput, PotScfOuterIterationStatus, PotScfState, PotScfStateAdvance,
    PotScfStateAdvanceInput, PotentialOverlapInput, PotentialOverlapNeighbor, ScfDensityStepInput,
    ScmtEnergyGrid, ScmtEnergyGridInput, advance_pot_scf_state,
    atomic::{
        AtomicLocalDensityExchangeMode, AtomicScfState, AtomicScfStateInput,
        atomic_coulomb_coefficients, atomic_differential_integral, atomic_form_factor,
        atomic_overlap_amplitude_reduction, atomic_scf_state_from_configuration, atomic_symbol,
        atomic_tabulation, atomic_total_energy_from_radials,
    },
    dirac_hara_exchange_potential, feff_default_configuration_rows, finish_pot_scf_outer_iteration,
    interstitial_fermi_level, interstitial_shell_values, karasiev_sjostrom_dufty_trickey_vxc,
    muffin_tin_interstitial_parameters, muffin_tin_radius_parameters, norman_radius_from_density,
    orbital_configuration, overlap_density_indices, overlap_potential_density, perdew_zunger_vxc,
    perrot_dharma_wardana_vxc, pot_scf_contour_source_rows, scmt_energy_grid, terp,
    update_scf_density_potential, von_barth_hedin_potential,
};
use refeff_io::{
    APOT_CORE_HOLE_RADIAL_POINTS, APOT_CORE_HOLE_SECTION_NUMBER, ApotAtomicPotsSectionsInput,
    ApotAtomicScfStateRef, ApotAtomicScfStateSectionsInput, ApotBinData, ApotBinMatrixValues,
    ApotBinPayload, ApotBinSection, ApotBinValue, ApotCoreHoleColumns, AtomDatData, ConfigDatData,
    ConfigDatPotential, ConfigRecord, ConfigSlotRows, FEFF_BOHR_ANGSTROM, Fpf0DatData,
    Fpf0Oscillator, GeomDat, ModuleLogData, MtdpData, PotBinData, PotBinScalars, PotInput,
    PotScfCorvalLdosHandoffInput, PotScfFmsSourceGridHandoff, PotScfFovrgSourceGridFromPlanInput,
    PotScfFovrgSourceGridHandoff, PotScfFovrgSourceGridPlan, PotScfFovrgSourceGridPlanInput,
    apot_atomic_pots_sections, apot_atomic_scf_sections_from_states, apot_bin_string,
    apot_core_hole_columns, apot_core_hole_coulomb_from_density, apot_core_hole_radii,
    config_record_slot_rows,
    pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_DEFAULT_PAD_WIDTH, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS,
        POT_BIN_RADIAL_POINTS,
    },
    pot_bin_string, pot_input_string, pot_scf_corval_ldos_handoff,
    pot_scf_fovrg_source_grid_handoff_from_plan, pot_scf_fovrg_source_grid_plan,
    potential_dat_outputs_from_bins, read_apot_bin, read_config_dat, read_config_inp,
    read_fpf0_dat, read_module_log_dat, read_mtdp, read_pot_bin,
    refresh_apot_core_hole_coulomb_payload, rhorrp_orbital_tables_from_config_dat, write_apot_bin,
    write_atom_dat, write_config_dat, write_fpf0_dat, write_module_log_dat, write_pot_bin,
};

use crate::fms::{
    PotScfFmsPipelineCache, PotScfFmsSourceGridInput,
    build_pot_scf_fms_source_grid_handoff_with_cache,
};
use crate::work_dir_for_input;

const ATOM_RADIAL_POINTS: usize = APOT_CORE_HOLE_RADIAL_POINTS;
const ATOM_FPF0_NORB_SECTION_NUMBER: usize = 3;
const ATOM_FPF0_DENSITY_SECTION_NUMBER: usize = 8;
const ATOM_FPF0_EORB_SECTION_NUMBER: usize = 14;
const ATOM_FPF0_KAPPA_SECTION_NUMBER: usize = 20;
const ATOM_FPF0_DGC_SECTION_START: usize = 22;
const ATOM_TOTAL_ENERGY_SPEED_OF_LIGHT: f64 = 137.0373;
const ATOM_NORMAN_VALENCE_CHANNEL_COUNT: usize = 4;
const ATOM_APOT_NORMAN_CHARGE_REL_TOLERANCE: f64 = 1.0e-5;
const ATOM_APOT_NORMAN_CHARGE_SCALE_PAD: f64 = 1.0e-6;
const ATOM_APOT_GEOMETRY_OVERLAP_CUTOFF: f64 = 12.0;
const POT_SCMT_MAX_ENERGY_POINTS: usize = 80;
const POT_SCMT_FLOOR_COUNT: usize = 17;
// Finite-nucleus spectra can need substantially more upward contour steps
// than the point-nucleus seed grid before the electron-count root is bracketed.
const POT_SCMT_MAX_ADAPTIVE_SOURCE_POINTS: usize = POT_SCMT_MAX_ENERGY_POINTS * 8;
const POT_SCMT_CHARGE_SUM_TOLERANCE: f64 = 0.05;
const POT_THERMAL_VERTICAL_POINTS: usize = 10;
const POT_THERMAL_INTERPOLATION_POINTS: usize = 1000;
const POT_THERMAL_INTERPOLATION_WINDOW: f64 = 12.0;
const POT_THERMAL_GRID_WINDOW_PAD: f64 = 2.0;
const POT_THERMAL_MAX_IMAGINARY_HARTREE: f64 = 0.15;
const POT_THERMAL_SECANT_STEPS: usize = 30;
const POT_THERMAL_MAX_GRID_POINTS: usize = 4096;
const POT_THERMAL_MAX_CHEMICAL_ITERATIONS: usize = 4096;
const POT_THERMAL_MAX_MATSUBARA_POLES: usize = 4096;
const POT_THERMAL_FMS_MAX_LMAX: usize = 3;
const POT_THERMAL_CHEMICAL_STALL_HARTREE: f64 = 1.0e-20;
const POT_CORVAL_TOLERANCE_EV: f64 = 5.0;
const POT_CORVAL_HIGH_EV: f64 = -20.0;
const POT_CORVAL_LDOS_IMAGINARY_EV: f64 = 1.5;
const POT_CORVAL_SCAN_BATCH_POINTS: usize = 12;
const POT_SCF_MAX_START_ATTEMPTS: usize = 3;
const POT_SCF_MIN_RETRY_MIXING: f64 = 0.01;
const POT_SCF_ECV_RETRY_TOLERANCE_HARTREE: f64 = 0.05;
const POT_SCF_TRAILING_ORBITAL_COMPONENT_THRESHOLD: f64 = 1.0e-11;
const POT_SCF_ORBITAL_OCCUPANCY_TOLERANCE: f64 = 1.0e-8;
const POT_SCF_CACHE_PROVENANCE_FILE: &str = ".refeff-pot-scf-cache";
const POT_SCF_CACHE_PROVENANCE_VERSION: &str = "refeff-pot-scf-cache-v1";
const POT_SCF_CACHE_FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const POT_SCF_CACHE_FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const POT_EXTERNAL_MTDP_FILE: &str = "GeCl4.04.dft.mtdp";
const POT_EXTERNAL_SORT_FILE: &str = "sort.aip";
const ATOM_POINT_NUCLEUS_REQUEST_INDEX: isize = 11;
// FEFF10 `wfirdf` uses five in-nucleus points; its source notes that the old
// eleven-point request fails for He and Cu.
const ATOM_FINITE_NUCLEUS_REQUEST_INDEX: isize = -5;
#[allow(dead_code)]
const ATOM_SCF_MAX_ORBITAL_ITERATIONS: usize = 40;
#[allow(dead_code)]
const ATOM_RADIAL_STEP: f64 = 0.05;
#[allow(dead_code)]
const ATOM_FIRST_RADIUS_LOG: f64 = -8.8;

/// Run the supported FEFF `ATOM` cached-output path beside the requested input.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    if has_cached_atomic_output(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_atomic_source_handoff(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_config_handoff(work_dir)? {
        return run_supported_config_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

/// Whether a FEFF `ATOM` run can be satisfied from an existing `apot.bin`.
pub(crate) fn has_cached_atomic_output(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.apot_bin.is_file() || !caches.pot_inp.is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if !atomic_enabled(&input) {
        return Ok(false);
    }
    Ok(can_use_cached_atomic_output(&caches, &input))
}

/// Whether FEFF `ATOM` can build the full source-backed `apot.bin` stream from
/// typed `pot.inp`, `geom.dat`, and `pot.bin` handoffs.
pub(crate) fn has_supported_atomic_source_handoff(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if caches.apot_bin.is_file() && can_use_cached_atomic_output(&caches, &input) {
        return Ok(false);
    }
    Ok(atomic_enabled(&input) && can_generate_atomic_apot_from_sources(&caches, &input))
}

/// Whether FEFF `ATOM` can validate or generate a source-backed `config.dat`
/// handoff from typed RDINP inputs before the remaining `apot.bin` solver
/// boundary.
pub(crate) fn has_supported_config_handoff(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if caches.apot_bin.is_file() && can_use_cached_atomic_output(&caches, &input) {
        return Ok(false);
    }
    if !atomic_enabled(&input) || !config_handoff_source_is_available(&caches) {
        return Ok(false);
    }

    let Ok(needs_generation) =
        config_handoff_needs_generation(&caches.config_dat, &input, &caches.config_inp)
    else {
        return Ok(false);
    };
    if needs_generation {
        return Ok(true);
    }
    Ok(config_handoff_matches_input(&caches.config_dat, &input).unwrap_or(false))
}

/// Validate or generate only the Rust-backed FEFF `config.dat` handoff needed
/// by downstream source paths. This intentionally stops before full APOT
/// generation when the complete `pot.bin`/`geom.dat` source bundle is absent.
pub(crate) fn run_supported_config_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    let written = write_or_generate_config(&caches.config_dat, &caches.config_inp, &input)?;
    let log_written = recover_existing_module_log_if_malformed(
        &caches.log1_dat,
        &input,
        can_recover_atomic_module_log(&caches, written > 0),
    )?;
    Ok(written + log_written)
}

fn pot_bin_potential_count(caches: &AtomicCachePaths) -> Result<usize> {
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    Ok(pot.potential_count())
}

fn config_handoff_source_is_available(caches: &AtomicCachePaths) -> bool {
    caches.pot_inp.is_file()
}

fn config_handoff_source_is_compatible(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<bool> {
    Ok(caches.pot_bin.is_file()
        && caches.pot_inp.is_file()
        && pot_bin_potential_count(caches)? == input.potentials.len())
}

fn can_use_cached_atomic_output(caches: &AtomicCachePaths, input: &PotInput) -> bool {
    prepare_cached_atomic_output(caches, input).is_ok()
}

fn prepare_cached_atomic_output(caches: &AtomicCachePaths, input: &PotInput) -> Result<()> {
    let config_source_handoff = prepare_config_cache(caches, input)?;

    let apot = read_apot_bin(&caches.apot_bin)
        .with_context(|| format!("failed to read {}", caches.apot_bin.display()))?;
    let mut apot = apot;
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;

    let fpf0_source_handoff = can_generate_fpf0_from_sources(caches, &apot, input)?;
    prepare_fpf0_cache(caches, &apot, input)?;
    if !can_recover_atomic_module_log(caches, config_source_handoff || fpf0_source_handoff) {
        prepare_module_log_cache(caches)?;
    }
    Ok(())
}

fn can_generate_atomic_apot_from_sources(caches: &AtomicCachePaths, input: &PotInput) -> bool {
    generated_atomic_apot_bin_from_sources(caches, input).is_ok()
}

pub(crate) fn can_write_atomic_apot_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    Ok(atomic_enabled(&input) && can_generate_atomic_apot_from_sources(&caches, &input))
}

pub(crate) fn write_atomic_apot_from_sources_in_dir(work_dir: &Path) -> Result<usize> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(0);
    }

    let mut apot = generated_atomic_apot_bin_from_sources(&caches, &input)
        .context("failed to generate ATOM apot.bin from source handoffs")?;
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    Ok(1)
}

pub(crate) fn can_write_no_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(false);
    }
    if input.run.nscmt != 0 {
        return Ok(false);
    }
    if !input.start_from_file && caches.pot_bin.is_file() && read_pot_bin(&caches.pot_bin).is_ok() {
        return Ok(false);
    }
    can_generate_no_scf_pot_bin_from_sources(&caches, &input)
}

/// Source fingerprint for a no-SCF POT preparation.
///
/// This deliberately excludes `pot.bin` and `apot.bin`: those are outputs
/// compared against the prepared state, not inputs to an ordinary no-SCF
/// run. `START_FROM_FILE` is excluded from this cache altogether because its
/// `pot.bin` is both a restart input and a subsequently overwritten output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NoScfPotSourceFingerprint {
    pot_inp: Vec<u8>,
    geom_dat: Vec<u8>,
    config_inp: Option<Vec<u8>>,
    external_mtdp: Option<Vec<u8>>,
    external_sort: Option<Vec<u8>>,
}

/// Fully prepared ordinary no-SCF POT outputs.
///
/// Keeping this value run-scoped lets discovery, staleness checks, and POT
/// execution share the expensive atomic SCF result without introducing a
/// process-global cache.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedNoScfPotOutputs {
    pot: PotBinData,
    apot: ApotBinData,
    pot_text: std::sync::OnceLock<String>,
    apot_text: std::sync::OnceLock<String>,
}

pub(crate) fn no_scf_pot_source_fingerprint_in_dir(
    work_dir: &Path,
) -> Result<Option<NoScfPotSourceFingerprint>> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(None);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt != 0 || input.start_from_file {
        return Ok(None);
    }
    if input.config_type == 2 && !caches.config_inp.is_file() {
        return Ok(None);
    }

    let (external_mtdp, external_sort) = if input.external_pot {
        (
            Some(read_no_scf_fingerprint_source(
                &pot_scf_cache_external_path(&caches, POT_EXTERNAL_MTDP_FILE),
            )?),
            Some(read_no_scf_fingerprint_source(
                &pot_scf_cache_external_path(&caches, POT_EXTERNAL_SORT_FILE),
            )?),
        )
    } else {
        (None, None)
    };

    Ok(Some(NoScfPotSourceFingerprint {
        pot_inp: read_no_scf_fingerprint_source(&caches.pot_inp)?,
        geom_dat: read_no_scf_fingerprint_source(&caches.geom_dat)?,
        config_inp: if input.config_type == 2 {
            Some(read_no_scf_fingerprint_source(&caches.config_inp)?)
        } else {
            None
        },
        external_mtdp,
        external_sort,
    }))
}

fn read_no_scf_fingerprint_source(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))
}

pub(crate) fn prepare_no_scf_pot_outputs_in_dir(
    work_dir: &Path,
) -> Result<Option<(NoScfPotSourceFingerprint, PreparedNoScfPotOutputs)>> {
    let Some(fingerprint) = no_scf_pot_source_fingerprint_in_dir(work_dir)? else {
        return Ok(None);
    };
    let caches = AtomicCachePaths::new(work_dir);
    let input = read_input(work_dir)?;
    let (pot, mut apot) = match generated_no_scf_pot_bin_and_apot_from_sources(&caches, &input) {
        Ok(generated) => generated,
        Err(error) if no_scf_pot_generation_unsupported(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;
    Ok(Some((
        fingerprint,
        PreparedNoScfPotOutputs {
            pot,
            apot,
            pot_text: Default::default(),
            apot_text: Default::default(),
        },
    )))
}

pub(crate) fn prepared_no_scf_pot_outputs_match_cached(
    work_dir: &Path,
    prepared: &PreparedNoScfPotOutputs,
) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    Ok(prepared_no_scf_pot_bin_matches_cached(work_dir, prepared)?
        && canonical_output_matches(
            &caches.apot_bin,
            prepared_text(&prepared.apot_text, || apot_bin_string(&prepared.apot))?,
            |text| apot_bin_string(&refeff_io::parse_apot_bin(text)?),
        )?)
}

pub(crate) fn prepared_no_scf_pot_bin_matches_cached(
    work_dir: &Path,
    prepared: &PreparedNoScfPotOutputs,
) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    canonical_output_matches(
        &caches.pot_bin,
        prepared_text(&prepared.pot_text, || pot_bin_string(&prepared.pot))?,
        |text| pot_bin_string(&refeff_io::parse_pot_bin(text)?),
    )
}

fn prepared_text(
    cached: &std::sync::OnceLock<String>,
    render: impl FnOnce() -> refeff_io::Result<String>,
) -> Result<&str> {
    if cached.get().is_none() {
        let _ = cached.set(render()?);
    }
    cached
        .get()
        .map(String::as_str)
        .context("missing prepared POT canonical output")
}

/// Exact canonical bytes need no parse/format cycle. Legacy formatting still
/// uses the existing semantic comparison; payloads are read on every check.
fn canonical_output_matches(
    path: &Path,
    expected: &str,
    canonicalize: impl FnOnce(&str) -> refeff_io::Result<String>,
) -> Result<bool> {
    let Ok(actual) = std::fs::read_to_string(path) else {
        return Ok(false);
    };
    if actual == expected {
        return Ok(true);
    }
    Ok(canonicalize(&actual).is_ok_and(|canonical| canonical == expected))
}

pub(crate) fn write_prepared_no_scf_pot_outputs_in_dir(
    work_dir: &Path,
    prepared: &PreparedNoScfPotOutputs,
) -> Result<usize> {
    let caches = AtomicCachePaths::new(work_dir);
    write_pot_bin(&caches.pot_bin, &prepared.pot)
        .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &prepared.apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    Ok(2)
}

pub(crate) fn write_no_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(0);
    }
    if input.run.nscmt != 0 {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    if !input.start_from_file && caches.pot_bin.is_file() && read_pot_bin(&caches.pot_bin).is_ok() {
        return Ok(0);
    }

    if input.start_from_file {
        let pot = generated_no_scf_pot_bin_from_sources(&caches, &input)?;
        write_pot_bin(&caches.pot_bin, &pot)
            .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
        return Ok(1);
    }

    let (pot, mut apot) = generated_no_scf_pot_bin_and_apot_from_sources(&caches, &input)?;
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;
    write_pot_bin(&caches.pot_bin, &pot)
        .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    Ok(2)
}

pub(crate) fn refresh_no_scf_pot_bin_from_sources_if_stale_in_dir(
    work_dir: &Path,
) -> Result<usize> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt != 0 {
        return Ok(0);
    }

    if input.start_from_file {
        return write_no_scf_pot_bin_from_sources_in_dir(work_dir);
    }

    let (pot, mut apot) = match generated_no_scf_pot_bin_and_apot_from_sources(&caches, &input) {
        Ok(generated) => generated,
        Err(error) if no_scf_pot_generation_unsupported(&error) => return Ok(0),
        Err(error) => return Err(error),
    };
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;
    if cached_pot_bin_matches_generated(&caches.pot_bin, &pot)?
        && cached_apot_bin_matches_generated(&caches.apot_bin, &apot)?
    {
        return Ok(0);
    }

    write_pot_bin(&caches.pot_bin, &pot)
        .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    Ok(2)
}

fn cached_pot_bin_matches_generated(path: &Path, generated: &PotBinData) -> Result<bool> {
    let Ok(cached) = read_pot_bin(path) else {
        return Ok(false);
    };
    Ok(pot_bin_string(&cached)? == pot_bin_string(generated)?)
}

fn cached_apot_bin_matches_generated(path: &Path, generated: &ApotBinData) -> Result<bool> {
    let Ok(cached) = read_apot_bin(path) else {
        return Ok(false);
    };
    Ok(apot_bin_string(&cached)? == apot_bin_string(generated)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PotScfCacheDigest {
    byte_len: usize,
    fnv64: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PotScfCacheProvenance {
    pot_inp: PotScfCacheDigest,
    geom_dat: PotScfCacheDigest,
    config_inp: Option<PotScfCacheDigest>,
    external_mtdp: Option<PotScfCacheDigest>,
    external_sort: Option<PotScfCacheDigest>,
    pot_bin: PotScfCacheDigest,
    apot_bin: PotScfCacheDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PotScfCacheProvenanceStatus {
    Fresh,
    Stale,
    Unknown,
}

fn pot_scf_cache_digest_bytes(bytes: &[u8]) -> PotScfCacheDigest {
    let mut hash = POT_SCF_CACHE_FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(POT_SCF_CACHE_FNV_PRIME);
    }
    PotScfCacheDigest {
        byte_len: bytes.len(),
        fnv64: hash,
    }
}

fn pot_scf_cache_digest_text(text: &str) -> PotScfCacheDigest {
    pot_scf_cache_digest_bytes(text.as_bytes())
}

fn pot_scf_cache_digest_path(path: &Path) -> Result<PotScfCacheDigest> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(pot_scf_cache_digest_bytes(&bytes))
}

fn pot_scf_cache_optional_digest_path(path: &Path) -> Result<Option<PotScfCacheDigest>> {
    if path.is_file() {
        return pot_scf_cache_digest_path(path).map(Some);
    }
    Ok(None)
}

fn pot_scf_cache_external_path(caches: &AtomicCachePaths, file_name: &str) -> PathBuf {
    caches
        .pot_inp
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(file_name)
}

fn pot_scf_cache_provenance_for_outputs(
    caches: &AtomicCachePaths,
    input: &PotInput,
    pot: &PotBinData,
    apot: &ApotBinData,
) -> Result<PotScfCacheProvenance> {
    let pot_input = pot_input_string(input).context("failed to render POT SCF pot.inp digest")?;
    let (external_mtdp, external_sort) = if input.external_pot {
        (
            Some(pot_scf_cache_digest_path(&pot_scf_cache_external_path(
                caches,
                POT_EXTERNAL_MTDP_FILE,
            ))?),
            Some(pot_scf_cache_digest_path(&pot_scf_cache_external_path(
                caches,
                POT_EXTERNAL_SORT_FILE,
            ))?),
        )
    } else {
        (None, None)
    };

    Ok(PotScfCacheProvenance {
        pot_inp: pot_scf_cache_digest_text(&pot_input),
        geom_dat: pot_scf_cache_digest_path(&caches.geom_dat)?,
        config_inp: pot_scf_cache_optional_digest_path(&caches.config_inp)?,
        external_mtdp,
        external_sort,
        pot_bin: pot_scf_cache_digest_text(
            &pot_bin_string(pot).context("failed to render POT SCF pot.bin digest")?,
        ),
        apot_bin: pot_scf_cache_digest_text(
            &apot_bin_string(apot).context("failed to render POT SCF apot.bin digest")?,
        ),
    })
}

fn current_pot_scf_cache_provenance(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotScfCacheProvenance> {
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    let apot = read_apot_bin(&caches.apot_bin)
        .with_context(|| format!("failed to read {}", caches.apot_bin.display()))?;
    pot_scf_cache_provenance_for_outputs(caches, input, &pot, &apot)
}

fn write_pot_scf_cache_provenance(
    caches: &AtomicCachePaths,
    input: &PotInput,
    pot: &PotBinData,
    apot: &ApotBinData,
) -> Result<()> {
    let provenance = pot_scf_cache_provenance_for_outputs(caches, input, pot, apot)?;
    std::fs::write(
        &caches.pot_scf_cache,
        pot_scf_cache_provenance_string(&provenance),
    )
    .with_context(|| format!("failed to write {}", caches.pot_scf_cache.display()))
}

fn pot_scf_cache_provenance_status(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotScfCacheProvenanceStatus> {
    if !caches.pot_scf_cache.is_file() {
        return Ok(PotScfCacheProvenanceStatus::Unknown);
    }
    let stored = match read_pot_scf_cache_provenance(&caches.pot_scf_cache) {
        Ok(stored) => stored,
        Err(_) => return Ok(PotScfCacheProvenanceStatus::Unknown),
    };
    let current = match current_pot_scf_cache_provenance(caches, input) {
        Ok(current) => current,
        Err(_) => return Ok(PotScfCacheProvenanceStatus::Stale),
    };
    Ok(if stored == current {
        PotScfCacheProvenanceStatus::Fresh
    } else {
        PotScfCacheProvenanceStatus::Stale
    })
}

fn pot_scf_cache_provenance_string(provenance: &PotScfCacheProvenance) -> String {
    [
        POT_SCF_CACHE_PROVENANCE_VERSION.to_string(),
        format!(
            "pot_inp={}",
            pot_scf_cache_digest_string(Some(provenance.pot_inp))
        ),
        format!(
            "geom_dat={}",
            pot_scf_cache_digest_string(Some(provenance.geom_dat))
        ),
        format!(
            "config_inp={}",
            pot_scf_cache_digest_string(provenance.config_inp)
        ),
        format!(
            "external_mtdp={}",
            pot_scf_cache_digest_string(provenance.external_mtdp)
        ),
        format!(
            "external_sort={}",
            pot_scf_cache_digest_string(provenance.external_sort)
        ),
        format!(
            "pot_bin={}",
            pot_scf_cache_digest_string(Some(provenance.pot_bin))
        ),
        format!(
            "apot_bin={}",
            pot_scf_cache_digest_string(Some(provenance.apot_bin))
        ),
    ]
    .join("\n")
        + "\n"
}

fn pot_scf_cache_digest_string(digest: Option<PotScfCacheDigest>) -> String {
    match digest {
        Some(digest) => format!("{}:{:016x}", digest.byte_len, digest.fnv64),
        None => "none".to_string(),
    }
}

fn read_pot_scf_cache_provenance(path: &Path) -> Result<PotScfCacheProvenance> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = text.lines();
    ensure!(
        lines.next() == Some(POT_SCF_CACHE_PROVENANCE_VERSION),
        "POT SCF cache provenance has unsupported version"
    );
    let provenance = PotScfCacheProvenance {
        pot_inp: parse_pot_scf_cache_required_digest_line(&mut lines, "pot_inp")?,
        geom_dat: parse_pot_scf_cache_required_digest_line(&mut lines, "geom_dat")?,
        config_inp: parse_pot_scf_cache_optional_digest_line(&mut lines, "config_inp")?,
        external_mtdp: parse_pot_scf_cache_optional_digest_line(&mut lines, "external_mtdp")?,
        external_sort: parse_pot_scf_cache_optional_digest_line(&mut lines, "external_sort")?,
        pot_bin: parse_pot_scf_cache_required_digest_line(&mut lines, "pot_bin")?,
        apot_bin: parse_pot_scf_cache_required_digest_line(&mut lines, "apot_bin")?,
    };
    ensure!(
        lines.next().is_none(),
        "POT SCF cache provenance has trailing records"
    );
    Ok(provenance)
}

fn parse_pot_scf_cache_required_digest_line<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    key: &str,
) -> Result<PotScfCacheDigest> {
    parse_pot_scf_cache_optional_digest_line(lines, key)?
        .with_context(|| format!("POT SCF cache provenance missing required {key} digest"))
}

fn parse_pot_scf_cache_optional_digest_line<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    key: &str,
) -> Result<Option<PotScfCacheDigest>> {
    let line = lines
        .next()
        .with_context(|| format!("POT SCF cache provenance missing {key} record"))?;
    let Some(value) = line.strip_prefix(&format!("{key}=")) else {
        bail!("POT SCF cache provenance expected {key} record");
    };
    if value == "none" {
        return Ok(None);
    }
    parse_pot_scf_cache_digest(value).map(Some)
}

fn parse_pot_scf_cache_digest(value: &str) -> Result<PotScfCacheDigest> {
    let (byte_len, hash) = value
        .split_once(':')
        .with_context(|| format!("POT SCF cache digest {value:?} is missing ':'"))?;
    let byte_len = byte_len
        .parse::<usize>()
        .with_context(|| format!("POT SCF cache digest length {byte_len:?} is invalid"))?;
    let fnv64 = u64::from_str_radix(hash, 16)
        .with_context(|| format!("POT SCF cache digest hash {hash:?} is invalid"))?;
    Ok(PotScfCacheDigest { byte_len, fnv64 })
}

pub(crate) fn can_prepare_scf_pot_initial_state_from_sources_in_dir(
    work_dir: &Path,
) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(false);
    }
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(false);
    }
    Ok(generated_scf_pot_initial_state_from_sources(&caches, &input).is_ok())
}

pub(crate) fn can_prepare_scf_pot_loop_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(false);
    }
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(false);
    }
    Ok(generated_scf_pot_run_from_sources(&caches, &input).is_ok())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PotScfSourceHandoffOutcome {
    NotApplicable,
    LoopValidated { count: usize },
    FinalOutput { count: usize },
}

pub(crate) fn can_write_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(false);
    }
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(false);
    }
    Ok(generated_scf_pot_run_from_sources(&caches, &input)
        .map(|run| run.final_pot.is_some())
        .unwrap_or(false))
}

pub(crate) fn prepare_scf_pot_initial_state_from_sources_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(0);
    }

    generated_scf_pot_initial_state_from_sources(&caches, &input)?;
    Ok(1)
}

pub(crate) fn prepare_scf_pot_loop_from_sources_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(0);
    }

    generated_scf_pot_run_from_sources(&caches, &input)?;
    Ok(1)
}

pub(crate) fn run_scf_pot_source_handoff_once_in_dir(
    work_dir: &Path,
) -> Result<PotScfSourceHandoffOutcome> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(PotScfSourceHandoffOutcome::NotApplicable);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(PotScfSourceHandoffOutcome::NotApplicable);
    }
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(PotScfSourceHandoffOutcome::NotApplicable);
    }

    let run = generated_scf_pot_run_from_sources(&caches, &input)?;
    let Some(final_pot) = run.final_pot.as_ref() else {
        return Ok(PotScfSourceHandoffOutcome::LoopValidated { count: 1 });
    };
    let final_apot = run
        .final_apot
        .as_ref()
        .context("POT SCF final apot.bin sidecar is unavailable")?;
    write_pot_bin(&caches.pot_bin, final_pot)
        .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, final_apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    write_pot_scf_cache_provenance(&caches, &input, final_pot, final_apot)?;
    Ok(PotScfSourceHandoffOutcome::FinalOutput { count: 2 })
}

pub(crate) fn try_write_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    if existing_pot_bin_is_fresh_final_scf_cache(&caches, &input)? {
        return Ok(0);
    }

    let run = generated_scf_pot_run_from_sources(&caches, &input)?;
    let Some(final_pot) = run.final_pot.as_ref() else {
        return Ok(0);
    };
    let final_apot = run
        .final_apot
        .as_ref()
        .context("POT SCF final apot.bin sidecar is unavailable")?;
    write_pot_bin(&caches.pot_bin, final_pot)
        .with_context(|| format!("failed to write {}", caches.pot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, final_apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;
    write_pot_scf_cache_provenance(&caches, &input, final_pot, final_apot)?;
    Ok(2)
}

fn existing_pot_bin_is_final_scf_cache(caches: &AtomicCachePaths, input: &PotInput) -> bool {
    !input.start_from_file && caches.pot_bin.is_file() && read_pot_bin(&caches.pot_bin).is_ok()
}

fn existing_pot_bin_is_fresh_final_scf_cache(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<bool> {
    if !existing_pot_bin_is_final_scf_cache(caches, input) {
        return Ok(false);
    }
    Ok(!scf_pot_final_cache_is_stale_against_sources(
        caches, input,
    )?)
}

pub(crate) fn has_stale_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt <= 0 || input.start_from_file {
        return Ok(false);
    }

    scf_pot_final_cache_is_stale_against_sources(&caches, &input)
}

fn scf_pot_final_cache_is_stale_against_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<bool> {
    if !existing_pot_bin_is_final_scf_cache(caches, input) {
        return Ok(false);
    }

    match pot_scf_cache_provenance_status(caches, input)? {
        PotScfCacheProvenanceStatus::Fresh => return Ok(false),
        PotScfCacheProvenanceStatus::Stale => return Ok(true),
        PotScfCacheProvenanceStatus::Unknown => {}
    }

    let Ok(run) = generated_scf_pot_run_from_sources(caches, input) else {
        return Ok(false);
    };
    let (Some(final_pot), Some(final_apot)) = (run.final_pot.as_ref(), run.final_apot.as_ref())
    else {
        return Ok(false);
    };

    let stale = !cached_pot_bin_matches_generated(&caches.pot_bin, final_pot)?
        || !cached_apot_bin_matches_generated(&caches.apot_bin, final_apot)?;
    if !stale {
        write_pot_scf_cache_provenance(caches, input, final_pot, final_apot)?;
    }
    Ok(stale)
}

fn atomic_apot_source_files_present(caches: &AtomicCachePaths) -> bool {
    caches.geom_dat.is_file()
}

fn atomic_apot_pot_source_files_present(caches: &AtomicCachePaths) -> bool {
    caches.pot_bin.is_file() && caches.geom_dat.is_file()
}

fn prepare_atomic_apot_pot_source_handoff(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<(GeomDat, PotBinData)> {
    ensure!(
        atomic_apot_pot_source_files_present(caches),
        "ATOM source apot.bin generation requires pot.bin and geom.dat handoffs"
    );
    let state_count = apot_state_count(input)?;
    ensure!(
        input.potentials.len() + 1 == state_count,
        "ATOM pot.inp has {} potential row(s), expected {} from nph={}",
        input.potentials.len(),
        state_count - 1,
        input.control.nph
    );

    let geom = read_geom_dat(&caches.geom_dat)?;
    let pot = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read {}", caches.pot_bin.display()))?;
    atomic_apot_static_arrays_from_handoffs(input, &geom, &pot)
        .context("failed to validate ATOM apot.bin source handoffs")?;
    Ok((geom, pot))
}

fn generated_atomic_apot_bin_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<ApotBinData> {
    generated_atomic_apot_and_states_from_sources(caches, input).map(|(apot, _states)| apot)
}

fn generated_atomic_apot_and_states_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<(ApotBinData, Vec<AtomicScfState>)> {
    let states = generated_atomic_scf_states(input, &caches.config_inp)?;
    if atomic_apot_pot_source_files_present(caches) {
        let (geom, pot) = prepare_atomic_apot_pot_source_handoff(caches, input)?;
        let apot =
            generated_atomic_apot_bin_from_states(input, &caches.config_inp, &geom, &pot, &states)
                .context(
                    "failed to generate ATOM apot.bin from pot.bin/geom.dat source handoffs",
                )?;
        return Ok((apot, states));
    }

    let geom = prepare_atomic_apot_geometry_source_handoff(caches, input)?;
    let unique_count = apot_unique_potential_count(input)?;
    let static_arrays = atomic_apot_static_arrays_from_source_geometry(
        input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )?;
    let sections = generated_atomic_apot_sections_from_static_arrays_and_states(
        input,
        &caches.config_inp,
        &static_arrays,
        &states,
    )
    .context("failed to generate ATOM apot.bin from pot.inp/geom.dat source handoffs")?;
    Ok((ApotBinData { sections }, states))
}

fn prepare_atomic_apot_geometry_source_handoff(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<GeomDat> {
    ensure!(
        caches.geom_dat.is_file(),
        "ATOM source apot.bin generation requires geom.dat handoff"
    );
    let state_count = apot_state_count(input)?;
    ensure!(
        input.potentials.len() + 1 == state_count,
        "ATOM pot.inp has {} potential row(s), expected {} from nph={}",
        input.potentials.len(),
        state_count - 1,
        input.control.nph
    );

    let geom = read_geom_dat(&caches.geom_dat)?;
    let unique_count = apot_unique_potential_count(input)?;
    atomic_apot_static_arrays_from_source_geometry(
        input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )
    .context("failed to validate ATOM source geometry")?;
    Ok(geom)
}

fn can_generate_no_scf_pot_bin_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<bool> {
    if input.config_type == 2 && !caches.config_inp.is_file() {
        return Ok(false);
    }
    match generated_no_scf_pot_source_state(caches, input) {
        Ok(_) => Ok(true),
        Err(error) if no_scf_pot_generation_unsupported(&error) => Ok(false),
        Err(error) => Err(error).context("failed to validate POT no-SCF source generation"),
    }
}

fn no_scf_pot_generation_unsupported(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| unsupported_no_scf_grid_error(cause.downcast_ref::<GridError>()))
        || error
            .chain()
            .any(|cause| match cause.downcast_ref::<DensityError>() {
                Some(DensityError::Grid(error)) => unsupported_no_scf_grid_error(Some(error)),
                _ => false,
            })
        || error.chain().any(unsupported_no_scf_source_selector_error)
}

fn unsupported_no_scf_grid_error(error: Option<&GridError>) -> bool {
    matches!(
        error,
        Some(GridError::InvalidRadius { .. } | GridError::MuffinTinOverlapTooLarge { .. })
    )
}

fn unsupported_no_scf_source_selector_error(error: &(dyn std::error::Error + 'static)) -> bool {
    let message = error.to_string();
    message.starts_with("POT no-SCF ground-state XC selector iscfxc=")
        && message.ends_with(" is unsupported")
}

fn generated_no_scf_pot_bin_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotBinData> {
    generated_no_scf_pot_source_state(caches, input).map(|(_, _, pot)| pot)
}

fn generated_no_scf_pot_bin_and_apot_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<(PotBinData, ApotBinData)> {
    let (geom, states, pot) = generated_no_scf_pot_source_state(caches, input)?;
    let apot =
        generated_atomic_apot_bin_from_states(input, &caches.config_inp, &geom, &pot, &states)
            .context("failed to generate ATOM apot.bin sidecar from no-SCF source states")?;
    Ok((pot, apot))
}

fn generated_no_scf_pot_source_state(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<(GeomDat, Vec<AtomicScfState>, PotBinData)> {
    let geom = prepare_no_scf_pot_bin_source_handoff(caches, input)?;
    let states = generated_atomic_scf_states(input, &caches.config_inp)
        .context("failed to prepare POT no-SCF atomic state columns")?;
    let mut pot = generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        input,
        &caches.config_inp,
        &geom,
        &states,
        None,
    )
    .context("failed to generate POT pot.bin from no-SCF source handoffs")?;
    if input.external_pot {
        apply_pot_external_potential_state(caches, &mut pot)?;
    }
    if let Some(restart) = pot_start_from_file_source_pot(caches, input, pot.potential_count())? {
        apply_pot_start_from_file_import_state(&mut pot, &restart)?;
    }
    Ok((geom, states, pot))
}

#[derive(Debug, Clone, PartialEq)]
struct PotScfSourceContext {
    geom: GeomDat,
    static_arrays: AtomicApotStaticArrays,
    config: ConfigDatData,
    pot: PotBinData,
    atomic_states: Vec<AtomicScfState>,
    external_source: Option<PotExternalPotentialSource>,
    restart_pot: Option<PotBinData>,
}

#[derive(Debug, Clone, PartialEq)]
struct PotScfInitialState {
    pot: PotBinData,
    state: PotScfState,
    istprm: MuffinTinInterstitialParameters,
    external_pot_imported: bool,
    restart_pot_imported: bool,
    energy_grid: ScmtEnergyGrid,
    fovrg_grid: Option<PotScfFovrgSourceGridHandoff>,
    fovrg_grid_unavailable: Option<String>,
    fms_grid: Option<PotScfFmsSourceGridHandoff>,
    fms_grid_unavailable: Option<String>,
    contour_rows: Option<PotScfContourSourceRows>,
    contour_rows_unavailable: Option<String>,
    state_advance: Option<PotScfStateAdvance>,
    state_advance_unavailable: Option<String>,
    next_iteration: Option<PotScfPreparedNextIteration>,
    next_iteration_unavailable: Option<String>,
    last_indices: Array1<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct PotScfPreparedNextIteration {
    iteration: usize,
    pot: PotBinData,
    state: PotScfState,
    istprm: MuffinTinInterstitialParameters,
    energy_grid: ScmtEnergyGrid,
    fovrg_grid: Option<PotScfFovrgSourceGridHandoff>,
    fovrg_grid_unavailable: Option<String>,
    fms_grid: Option<PotScfFmsSourceGridHandoff>,
    fms_grid_unavailable: Option<String>,
    contour_rows: Option<PotScfContourSourceRows>,
    contour_rows_unavailable: Option<String>,
    state_advance: Option<PotScfStateAdvance>,
    state_advance_unavailable: Option<String>,
    last_indices: Array1<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct PotScfAdaptiveSourceAdvance {
    fovrg_grid: PotScfFovrgSourceGridHandoff,
    fms_grid: PotScfFmsSourceGridHandoff,
    contour_rows: PotScfContourSourceRows,
    state_advance: PotScfStateAdvance,
}

#[derive(Debug, Clone, PartialEq)]
struct PotThermalScfGrid {
    energies: Array1<Complex64>,
    pole_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct PotThermalScfDensities {
    angular: Array3<Complex64>,
    radial: Array3<Complex64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PotThermalScfIntegral {
    electron_count: f64,
    occupancy_by_l: Array2<f64>,
    valence_density: Array2<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PotThermalScfAdvance {
    fovrg_grid: PotScfFovrgSourceGridHandoff,
    fms_grid: PotScfFmsSourceGridHandoff,
    contour_rows: PotScfContourSourceRows,
    state_advance: PotScfStateAdvance,
}

#[derive(Debug, Clone, PartialEq)]
struct PotScfSourceRun {
    initial: PotScfInitialState,
    prepared_iterations: Vec<PotScfPreparedNextIteration>,
    final_status: Option<PotScfOuterIterationStatus>,
    final_iteration: Option<usize>,
    final_pot: Option<PotBinData>,
    final_apot: Option<ApotBinData>,
    final_pot_unavailable: Option<String>,
    terminal_unavailable: Option<String>,
}

fn generated_scf_pot_initial_state_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotScfInitialState> {
    let context = scf_pot_source_context_from_sources(caches, input, true)?;
    let work_dir = caches.pot_inp.parent().unwrap_or_else(|| Path::new("."));
    let mut fms_cache = PotScfFmsPipelineCache::default();
    scf_pot_initial_state_from_generated_pot(
        work_dir,
        input,
        &context.static_arrays,
        &context.config,
        context.pot,
        context.external_source.as_ref(),
        context.restart_pot.as_ref(),
        true,
        &mut fms_cache,
    )
}

fn generated_scf_pot_run_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotScfSourceRun> {
    let work_dir = caches.pot_inp.parent().unwrap_or_else(|| Path::new("."));
    scf_pot_run_with_retries(input, |retry_input, first_scmt_call| {
        let context = scf_pot_source_context_from_sources(caches, retry_input, first_scmt_call)?;
        let mut fms_cache = PotScfFmsPipelineCache::default();
        let initial = scf_pot_initial_state_from_generated_pot(
            work_dir,
            retry_input,
            &context.static_arrays,
            &context.config,
            context.pot.clone(),
            context.external_source.as_ref(),
            context.restart_pot.as_ref(),
            first_scmt_call,
            &mut fms_cache,
        )?;
        let run = scf_pot_run_from_initial_state(
            work_dir,
            retry_input,
            &context.static_arrays,
            &context.config,
            initial,
            &mut fms_cache,
        )?;
        let mut run = run;
        attach_scf_pot_final_apot(&mut run, retry_input, &caches.config_inp, &context)?;
        Ok(run)
    })
}

fn scf_pot_run_with_retries(
    input: &PotInput,
    mut run_attempt: impl FnMut(&PotInput, bool) -> Result<PotScfSourceRun>,
) -> Result<PotScfSourceRun> {
    let mut retry_input = input.clone();
    let mut attempt = 1usize;
    let mut run = loop {
        let run = run_attempt(&retry_input, attempt == 1)?;
        validate_scf_pot_source_run(&run)?;
        if run.final_status != Some(PotScfOuterIterationStatus::RepeatRequired) {
            return Ok(run);
        }
        let retry_core_valence_energy = run.initial.pot.scalars.core_valence_energy;
        if attempt >= POT_SCF_MAX_START_ATTEMPTS {
            break run;
        }
        attempt =
            update_scf_pot_retry_controls(&mut retry_input, retry_core_valence_energy, attempt)?;
    };
    if run.final_status == Some(PotScfOuterIterationStatus::RepeatRequired) {
        let base = run.final_pot_unavailable.take().unwrap_or_else(|| {
            "POT SCF loop ended with RepeatRequired; final pot.bin requires convergence or iteration-limit state"
                .to_string()
        });
        run.final_pot_unavailable = Some(format!(
            "{base} after {POT_SCF_MAX_START_ATTEMPTS} FEFF-style start attempt(s)"
        ));
    }
    validate_scf_pot_source_run(&run)?;
    Ok(run)
}

fn update_scf_pot_retry_controls(
    input: &mut PotInput,
    adjusted_core_valence_energy: f64,
    attempt: usize,
) -> Result<usize> {
    ensure!(
        attempt > 0 && attempt < POT_SCF_MAX_START_ATTEMPTS,
        "POT SCF retry attempt {attempt} cannot advance toward {POT_SCF_MAX_START_ATTEMPTS}"
    );
    ensure!(
        adjusted_core_valence_energy.is_finite(),
        "POT SCF retry core-valence energy is non-finite: {adjusted_core_valence_energy}"
    );
    let previous_core_valence_energy = pot_input_core_valence_energy_hartree(input)?;
    ensure!(
        previous_core_valence_energy.is_finite(),
        "POT SCF retry previous core-valence energy is non-finite: {previous_core_valence_energy}"
    );
    let ecv_unchanged = (adjusted_core_valence_energy - previous_core_valence_energy).abs()
        < POT_SCF_ECV_RETRY_TOLERANCE_HARTREE;
    input.scattering.ecv = adjusted_core_valence_energy * FEFF_HARTREE_EV;

    let mut next_attempt = attempt + 1;
    if ecv_unchanged && next_attempt == 2 {
        next_attempt = POT_SCF_MAX_START_ATTEMPTS;
    }
    if ecv_unchanged || next_attempt == POT_SCF_MAX_START_ATTEMPTS {
        input.scattering.ca1 = (input.scattering.ca1 / 5.0).max(POT_SCF_MIN_RETRY_MIXING);
    }
    Ok(next_attempt)
}

fn pot_input_core_valence_energy_hartree(input: &PotInput) -> Result<f64> {
    ensure!(
        input.scattering.ecv.is_finite(),
        "POT input core-valence energy is non-finite: {}",
        input.scattering.ecv
    );
    Ok(input.scattering.ecv / FEFF_HARTREE_EV)
}

fn attach_scf_pot_final_apot(
    run: &mut PotScfSourceRun,
    input: &PotInput,
    config_inp: &Path,
    context: &PotScfSourceContext,
) -> Result<()> {
    let Some(final_pot) = run.final_pot.as_ref() else {
        return Ok(());
    };
    let mut apot = generated_atomic_apot_bin_from_states(
        input,
        config_inp,
        &context.geom,
        final_pot,
        &context.atomic_states,
    )
    .context("failed to prepare POT final SCF apot.bin sidecar from source states")?;
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .context("failed to refresh POT final SCF apot.bin sidecar core-hole payload")?;
    run.final_apot = Some(apot);
    Ok(())
}

fn scf_pot_source_context_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
    import_restart: bool,
) -> Result<PotScfSourceContext> {
    let geom = prepare_scf_pot_initial_state_source_handoff(caches, input)?;
    let unique_count = apot_unique_potential_count(input)?;
    let static_arrays = atomic_apot_static_arrays_from_source_geometry(
        input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )?;
    let states = generated_atomic_scf_states(input, &caches.config_inp)
        .context("failed to prepare POT initial SCF atomic state columns")?;
    let provisional_pot = generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        input,
        &caches.config_inp,
        &geom,
        &states,
        None,
    )
    .context("failed to prepare POT initial SCF pot.bin state from source handoffs")?;
    let config = generated_config_dat(input, &caches.config_inp)
        .context("failed to prepare POT initial SCF config.dat state from source handoffs")?;
    let preliminary_core_valence = no_scf_pot_core_valence_selection(
        input,
        &states,
        &provisional_pot.atomic_numbers,
        provisional_pot.scalars.interstitial_potential,
        None,
    )
    .context("failed to prepare POT preliminary core-valence selection")?;
    let core_valence_peaks = scf_pot_corval_peak_energies_for_selection(
        input,
        &config,
        &provisional_pot,
        &preliminary_core_valence,
    )
    .context("failed to scan POT corval LDOS peaks from source handoffs")?;
    let pot = generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        input,
        &caches.config_inp,
        &geom,
        &states,
        Some(core_valence_peaks.view()),
    )
    .context("failed to prepare POT initial SCF pot.bin state with corval LDOS peaks")?;
    let external_source = if input.external_pot {
        Some(read_pot_external_potential_source(
            caches,
            pot.potential_count(),
        )?)
    } else {
        None
    };
    let restart_pot = if import_restart {
        pot_start_from_file_source_pot(caches, input, pot.potential_count())?
    } else {
        None
    };
    Ok(PotScfSourceContext {
        geom,
        static_arrays,
        config,
        pot,
        atomic_states: states,
        external_source,
        restart_pot,
    })
}

fn prepare_no_scf_pot_bin_source_handoff(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<GeomDat> {
    ensure!(
        caches.geom_dat.is_file(),
        "POT no-SCF pot.bin generation requires geom.dat handoff"
    );
    ensure!(
        input.run.nscmt == 0,
        "POT no-SCF pot.bin generation requires nscmt=0, got {}",
        input.run.nscmt
    );
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        unique_count >= 1,
        "POT no-SCF pot.bin generation requires at least one potential"
    );
    if input.external_pot {
        read_pot_external_potential_source(caches, unique_count)?;
    }

    let geom = read_geom_dat(&caches.geom_dat)?;
    atomic_apot_static_arrays_from_source_geometry(
        input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )
    .context("failed to validate POT no-SCF source geometry")?;
    Ok(geom)
}

fn prepare_scf_pot_initial_state_source_handoff(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<GeomDat> {
    ensure!(
        caches.geom_dat.is_file(),
        "POT initial SCF state preparation requires geom.dat handoff"
    );
    ensure!(
        input.run.nscmt > 0,
        "POT initial SCF state preparation requires nscmt>0, got {}",
        input.run.nscmt
    );
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        unique_count >= 1,
        "POT initial SCF state preparation requires at least one potential"
    );
    if input.external_pot {
        read_pot_external_potential_source(caches, unique_count)?;
    }
    pot_start_from_file_source_pot(caches, input, unique_count)?;

    let geom = read_geom_dat(&caches.geom_dat)?;
    atomic_apot_static_arrays_from_source_geometry(
        input,
        &geom,
        Array1::from_elem(unique_count, 1.0),
    )
    .context("failed to validate POT initial SCF source geometry")?;
    Ok(geom)
}

#[derive(Debug, Clone, PartialEq)]
struct PotExternalPotentialSource {
    mtdp: MtdpData,
    sort_indices: Vec<Option<usize>>,
}

fn read_pot_external_potential_source(
    caches: &AtomicCachePaths,
    expected_count: usize,
) -> Result<PotExternalPotentialSource> {
    let work_dir = caches.pot_inp.parent().unwrap_or_else(|| Path::new("."));
    let mtdp_path = work_dir.join(POT_EXTERNAL_MTDP_FILE);
    let sort_path = work_dir.join(POT_EXTERNAL_SORT_FILE);
    let mtdp = read_mtdp(&mtdp_path)
        .with_context(|| format!("failed to read external POT MTDP {}", mtdp_path.display()))?;
    ensure!(
        mtdp.radial_count > 0 && mtdp.radial_count <= POT_BIN_RADIAL_POINTS,
        "POT EXTERNAL_POT MTDP radial count {} must be in 1..={POT_BIN_RADIAL_POINTS}",
        mtdp.radial_count
    );
    let sort_indices = read_pot_external_sort_indices(&sort_path)
        .with_context(|| format!("failed to read external POT sort {}", sort_path.display()))?;
    ensure!(
        !sort_indices.is_empty(),
        "POT EXTERNAL_POT sort.aip must contain at least one mapping"
    );
    let source_count = mtdp.atomic_numbers.len() + mtdp.empty_sphere_radii.len();
    for (source, target) in sort_indices.iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        ensure!(
            source < source_count,
            "POT EXTERNAL_POT sort source {} exceeds MTDP source count {source_count}",
            source + 1
        );
        ensure!(
            *target < expected_count,
            "POT EXTERNAL_POT sort target {target} exceeds {expected_count} potential(s)"
        );
    }
    Ok(PotExternalPotentialSource { mtdp, sort_indices })
}

pub(crate) fn validate_pot_external_source_handoff_in_dir(
    work_dir: &Path,
    expected_count: usize,
) -> Result<()> {
    let caches = AtomicCachePaths::new(work_dir);
    read_pot_external_potential_source(&caches, expected_count).map(|_| ())
}

fn read_pot_external_sort_indices(path: &Path) -> Result<Vec<Option<usize>>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut values = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for token in line.split_whitespace() {
            let value = token
                .parse::<isize>()
                .with_context(|| format!("invalid POT EXTERNAL_POT sort token {token:?}"))?;
            values.push(if value < 0 {
                None
            } else {
                Some(usize::try_from(value).context("POT EXTERNAL_POT sort index overflowed")?)
            });
        }
    }
    Ok(values)
}

fn apply_pot_external_potential_state(
    caches: &AtomicCachePaths,
    pot: &mut PotBinData,
) -> Result<()> {
    let source = read_pot_external_potential_source(caches, pot.potential_count())?;
    apply_pot_external_potential_source(pot, &source)
}

fn apply_pot_external_potential_source(
    pot: &mut PotBinData,
    source: &PotExternalPotentialSource,
) -> Result<()> {
    let potential_count = pot.potential_count();
    ensure!(
        pot.total_potential.dim() == (POT_BIN_RADIAL_POINTS, potential_count)
            && pot.electron_density.dim() == (POT_BIN_RADIAL_POINTS, potential_count),
        "POT EXTERNAL_POT target pot.bin shapes total={:?}, density={:?}, expected {POT_BIN_RADIAL_POINTS}x{potential_count}",
        pot.total_potential.dim(),
        pot.electron_density.dim()
    );
    pot.scalars.interstitial_potential = source.mtdp.interstitial_potential;
    pot.scalars.fermi_level = 0.5 * (source.mtdp.homo_energy + source.mtdp.lumo_energy);

    for (source_index, target) in source.sort_indices.iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        if source_index < source.mtdp.atomic_numbers.len() {
            apply_pot_external_potential_column(
                pot,
                *target,
                source.mtdp.atom_radius_indices[source_index],
                source.mtdp.atom_radii[source_index],
                source.mtdp.atom_potential.column(source_index),
                source.mtdp.atom_density.column(source_index),
                source.mtdp.radial_count,
            )?;
        } else {
            let empty_index = source_index - source.mtdp.atomic_numbers.len();
            apply_pot_external_potential_column(
                pot,
                *target,
                source.mtdp.empty_sphere_radius_indices[empty_index],
                source.mtdp.empty_sphere_radii[empty_index],
                source.mtdp.empty_sphere_potential.column(empty_index),
                source.mtdp.empty_sphere_density.column(empty_index),
                source.mtdp.radial_count,
            )?;
        }
    }
    Ok(())
}

fn apply_pot_external_potential_column(
    pot: &mut PotBinData,
    potential: usize,
    muffin_tin_index: usize,
    muffin_tin_radius: f64,
    total_potential: ArrayView1<'_, f64>,
    electron_density: ArrayView1<'_, f64>,
    radial_count: usize,
) -> Result<()> {
    ensure!(
        potential < pot.potential_count(),
        "POT EXTERNAL_POT target potential {potential} exceeds {} potential(s)",
        pot.potential_count()
    );
    ensure!(
        total_potential.len() >= radial_count && electron_density.len() >= radial_count,
        "POT EXTERNAL_POT source column lengths total={}, density={}, expected at least {radial_count}",
        total_potential.len(),
        electron_density.len()
    );
    ensure!(
        muffin_tin_radius.is_finite() && muffin_tin_radius > 0.0,
        "POT EXTERNAL_POT muffin-tin radius for potential {potential} must be positive and finite"
    );
    pot.muffin_tin_indices[potential] = muffin_tin_index;
    pot.muffin_tin_radii[potential] = muffin_tin_radius;
    for row in 0..radial_count {
        pot.total_potential[(row, potential)] = total_potential[row];
        pot.electron_density[(row, potential)] = electron_density[row];
    }
    for row in radial_count..POT_BIN_RADIAL_POINTS {
        pot.total_potential[(row, potential)] = pot.scalars.interstitial_potential;
        pot.electron_density[(row, potential)] = pot.scalars.interstitial_density;
    }
    Ok(())
}

fn pot_start_from_file_source_pot(
    caches: &AtomicCachePaths,
    input: &PotInput,
    expected_count: usize,
) -> Result<Option<PotBinData>> {
    if !input.start_from_file {
        return Ok(None);
    }
    ensure!(
        caches.pot_bin.is_file(),
        "POT START_FROM_FILE initial SCF state preparation requires a pot.bin restart handoff"
    );
    let restart = read_pot_bin(&caches.pot_bin)
        .with_context(|| format!("failed to read restart {}", caches.pot_bin.display()))?;
    ensure!(
        restart.potential_count() == expected_count,
        "POT START_FROM_FILE restart pot.bin has {} potential(s), expected {expected_count}",
        restart.potential_count()
    );
    ensure!(
        restart.total_potential.dim() == (POT_BIN_RADIAL_POINTS, expected_count)
            && restart.electron_density.dim() == (POT_BIN_RADIAL_POINTS, expected_count),
        "POT START_FROM_FILE restart pot.bin array shapes total={:?}, density={:?}, expected {POT_BIN_RADIAL_POINTS}x{expected_count}",
        restart.total_potential.dim(),
        restart.electron_density.dim()
    );
    ensure!(
        restart
            .total_potential
            .iter()
            .all(|value| value.is_finite())
            && restart
                .electron_density
                .iter()
                .all(|value| value.is_finite())
            && restart.scalars.fermi_level.is_finite()
            && restart.scalars.interstitial_potential.is_finite()
            && restart.scalars.interstitial_density.is_finite(),
        "POT START_FROM_FILE restart pot.bin contains non-finite imported state"
    );
    Ok(Some(restart))
}

fn scf_pot_initial_state_from_generated_pot(
    work_dir: &Path,
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    config: &ConfigDatData,
    mut pot: PotBinData,
    external_source: Option<&PotExternalPotentialSource>,
    restart_pot: Option<&PotBinData>,
    first_scmt_call: bool,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfInitialState> {
    let istprm = scf_pot_istprm_from_initial_state(input, static_arrays, &pot)
        .context("failed to validate POT initial SCF istprm state")?;
    apply_scf_pot_istprm_state(&mut pot, input, &istprm)?;
    let external_pot_imported = if let Some(source) = external_source {
        apply_pot_external_potential_source(&mut pot, source)?;
        true
    } else {
        false
    };
    let restart_pot_imported = if let Some(restart) = restart_pot {
        apply_pot_start_from_file_import_state(&mut pot, restart)?;
        true
    } else {
        false
    };
    let energy_grid = scf_pot_energy_grid_from_initial_state(&pot)
        .context("failed to validate POT initial SCF energy grid")?;
    let last_indices = scf_pot_rholie_last_indices(&pot)
        .context("failed to derive POT initial SCF rholie radial bounds")?;
    let workspace = validate_scf_pot_density_step_from_initial_state(
        input,
        static_arrays,
        &pot,
        last_indices.view(),
    )?;
    let initial_norman_charges = Array1::zeros(pot.potential_count());
    let state = PotScfState {
        fermi_energy: pot.scalars.fermi_level,
        norman_charges: initial_norman_charges.clone(),
        norman_charge_reference: initial_norman_charges,
        occupancy_by_l: Array2::zeros(pot.valence_occupancy.dim()),
        overlapped_density: pot.electron_density.clone(),
        overlapped_valence_density: pot.valence_density.clone(),
        coulomb_potential: pot.coulomb_potential.clone(),
        workspace,
    };
    let (
        fovrg_grid,
        fovrg_grid_unavailable,
        fms_grid,
        fms_grid_unavailable,
        contour_rows,
        contour_rows_unavailable,
        state_advance,
        state_advance_unavailable,
    ) = match if scf_pot_uses_thermal_occupations(input)? {
        scf_pot_thermal_source_advance(
            work_dir,
            input,
            static_arrays,
            config,
            &pot,
            last_indices.view(),
            &state,
            1,
            fms_cache,
        )
        .map(|output| PotScfAdaptiveSourceAdvance {
            fovrg_grid: output.fovrg_grid,
            fms_grid: output.fms_grid,
            contour_rows: output.contour_rows,
            state_advance: output.state_advance,
        })
    } else {
        scf_pot_adaptive_source_advance(
            work_dir,
            input,
            static_arrays,
            config,
            &pot,
            &energy_grid,
            last_indices.view(),
            &state,
            1,
            first_scmt_call,
            fms_cache,
        )
    } {
        Ok(output) => (
            Some(output.fovrg_grid),
            None,
            Some(output.fms_grid),
            None,
            Some(output.contour_rows),
            None,
            Some(output.state_advance),
            None,
        ),
        Err(error) => {
            let reason = format!("{error:#}");
            (
                None,
                Some(reason.clone()),
                None,
                Some("POT initial SCF FOVRG source grid is unavailable".to_string()),
                None,
                Some("POT initial SCF FMS source grid is unavailable".to_string()),
                None,
                Some(reason),
            )
        }
    };
    let (next_iteration, next_iteration_unavailable) = match &state_advance {
        Some(advance) if advance.outer.status == PotScfOuterIterationStatus::NeedsNextIteration => {
            match scf_pot_next_iteration_preparation_from_state(
                work_dir,
                input,
                static_arrays,
                config,
                &pot,
                &advance.state,
                2,
                fms_cache,
            ) {
                Ok(next) => (Some(next), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            }
        }
        Some(advance) => (
            None,
            Some(format!(
                "POT initial SCF state advance ended with {:?}; next istprm pass is not required",
                advance.outer.status
            )),
        ),
        None => (
            None,
            Some("POT initial SCF state advance is unavailable".to_string()),
        ),
    };
    let initial = PotScfInitialState {
        pot,
        state,
        istprm,
        external_pot_imported,
        restart_pot_imported,
        energy_grid,
        fovrg_grid,
        fovrg_grid_unavailable,
        fms_grid,
        fms_grid_unavailable,
        contour_rows,
        contour_rows_unavailable,
        state_advance,
        state_advance_unavailable,
        next_iteration,
        next_iteration_unavailable,
        last_indices,
    };
    validate_scf_pot_initial_state(&initial)?;
    Ok(initial)
}

fn scf_pot_run_from_initial_state(
    work_dir: &Path,
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    config: &ConfigDatData,
    initial: PotScfInitialState,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfSourceRun> {
    let max_iterations = scf_pot_max_iterations(input)?;
    let mut prepared_iterations = Vec::new();
    let mut final_status = None;
    let mut final_iteration = None;
    let mut final_pot = None;
    let mut final_pot_unavailable = None;
    let mut terminal_unavailable = None;

    let Some(initial_advance) = initial.state_advance.as_ref() else {
        terminal_unavailable = Some(
            initial
                .state_advance_unavailable
                .clone()
                .unwrap_or_else(|| "POT initial SCF state advance is unavailable".to_string()),
        );
        return Ok(PotScfSourceRun {
            initial,
            prepared_iterations,
            final_status,
            final_iteration,
            final_pot,
            final_apot: None,
            final_pot_unavailable: Some(
                terminal_unavailable
                    .clone()
                    .unwrap_or_else(|| "POT initial SCF state advance is unavailable".to_string()),
            ),
            terminal_unavailable,
        });
    };

    final_status = Some(initial_advance.outer.status);
    final_iteration = Some(1);
    let mut previous_pot = initial.pot.clone();
    let mut current_state = initial_advance.state.clone();
    let mut status = initial_advance.outer.status;
    if scf_pot_status_has_final_pot(status) {
        final_pot = Some(
            scf_pot_final_pot_from_state(
                &previous_pot,
                &current_state,
                initial_advance.outer.reported_charge_transfer.view(),
                status,
                config,
            )
            .context("failed to assemble POT final pot.bin candidate from initial state")?,
        );
    }
    let mut iteration = 2usize;

    while status == PotScfOuterIterationStatus::NeedsNextIteration && iteration <= max_iterations {
        let prepared = if iteration == 2 {
            match initial.next_iteration.clone() {
                Some(next) => next,
                None => {
                    terminal_unavailable = Some(
                        initial
                            .next_iteration_unavailable
                            .clone()
                            .unwrap_or_else(|| {
                                "POT next SCF iteration preparation is unavailable".to_string()
                            }),
                    );
                    break;
                }
            }
        } else {
            scf_pot_next_iteration_preparation_from_state(
                work_dir,
                input,
                static_arrays,
                config,
                &previous_pot,
                &current_state,
                iteration,
                fms_cache,
            )?
        };
        ensure!(
            prepared.iteration == iteration,
            "POT SCF loop prepared iteration {} while expecting {iteration}",
            prepared.iteration
        );

        let advance = prepared.state_advance.clone();
        let unavailable = prepared.state_advance_unavailable.clone();
        previous_pot = prepared.pot.clone();
        prepared_iterations.push(prepared);
        match advance {
            Some(advance) => {
                status = advance.outer.status;
                final_status = Some(status);
                final_iteration = Some(iteration);
                let reported_charge_transfer = advance.outer.reported_charge_transfer.clone();
                current_state = advance.state;
                if scf_pot_status_has_final_pot(status) {
                    final_pot = Some(
                        scf_pot_final_pot_from_state(
                            &previous_pot,
                            &current_state,
                            reported_charge_transfer.view(),
                            status,
                            config,
                        )
                            .with_context(|| {
                                format!(
                                    "failed to assemble POT final pot.bin candidate from iteration {iteration}"
                                )
                            })?,
                    );
                }
                iteration += 1;
            }
            None => {
                terminal_unavailable = Some(unavailable.unwrap_or_else(|| {
                    format!("POT SCF iteration {iteration} state advance is unavailable")
                }));
                break;
            }
        }
    }

    if status == PotScfOuterIterationStatus::NeedsNextIteration && iteration > max_iterations {
        terminal_unavailable = Some(format!(
            "POT SCF loop still requested iteration {iteration} after nscmt={max_iterations}"
        ));
    }
    if final_pot.is_none() {
        final_pot_unavailable = Some(terminal_unavailable.clone().unwrap_or_else(|| {
            format!(
                "POT SCF loop ended with {status:?}; final pot.bin requires convergence or iteration-limit state"
            )
        }));
    }

    Ok(PotScfSourceRun {
        initial,
        prepared_iterations,
        final_status,
        final_iteration,
        final_pot,
        final_apot: None,
        final_pot_unavailable,
        terminal_unavailable,
    })
}

fn validate_scf_pot_source_run(run: &PotScfSourceRun) -> Result<()> {
    let potential_count = run.initial.pot.potential_count();
    for (offset, prepared) in run.prepared_iterations.iter().enumerate() {
        let expected_iteration = offset + 2;
        ensure!(
            prepared.iteration == expected_iteration
                && prepared.pot.potential_count() == potential_count,
            "POT SCF source run prepared iteration {} is inconsistent with expected iteration {expected_iteration} or potential count {potential_count}",
            prepared.iteration
        );
    }
    if let Some(iteration) = run.final_iteration {
        let max_iteration = run.prepared_iterations.len() + 1;
        ensure!(
            iteration > 0 && iteration <= max_iteration,
            "POT SCF source run final iteration {iteration} is outside 1..={max_iteration}"
        );
    }
    if let Some(reason) = &run.terminal_unavailable {
        ensure!(
            !reason.is_empty(),
            "POT SCF source run terminal unavailable reason is empty"
        );
    }

    match run.final_status {
        Some(status) if scf_pot_status_has_final_pot(status) => {
            let final_pot = run
                .final_pot
                .as_ref()
                .context("POT SCF source run reached a final status without final pot.bin")?;
            ensure!(
                run.final_pot_unavailable.is_none()
                    && final_pot.potential_count() == potential_count
                    && final_pot.electron_density.dim() == run.initial.pot.electron_density.dim()
                    && final_pot.valence_density.dim() == run.initial.pot.valence_density.dim()
                    && final_pot.coulomb_potential.dim() == run.initial.pot.coulomb_potential.dim()
                    && final_pot.valence_occupancy.ncols() == potential_count,
                "POT SCF source run final pot.bin candidate is inconsistent"
            );
            pot_bin_string(final_pot)
                .context("POT SCF source run final pot.bin candidate is not renderable")?;
            let final_apot = run
                .final_apot
                .as_ref()
                .context("POT SCF source run reached a final status without final apot.bin")?;
            potential_dat_outputs_from_bins(final_pot, final_apot)
                .context("POT SCF source run final pot.bin/apot.bin pair is not renderable")?;
        }
        _ => {
            ensure!(
                run.final_pot.is_none(),
                "POT SCF source run has a final pot.bin candidate without a terminal final status"
            );
            ensure!(
                run.final_apot.is_none(),
                "POT SCF source run has a final apot.bin candidate without a terminal final status"
            );
            let has_reason = match run.final_pot_unavailable.as_ref() {
                Some(reason) => !reason.is_empty(),
                None => false,
            };
            ensure!(
                has_reason,
                "POT SCF source run is missing a final pot.bin unavailable reason"
            );
        }
    }
    Ok(())
}

fn scf_pot_status_has_final_pot(status: PotScfOuterIterationStatus) -> bool {
    matches!(
        status,
        PotScfOuterIterationStatus::Converged | PotScfOuterIterationStatus::ReachedIterationLimit
    )
}

fn scf_pot_final_pot_from_state(
    pot: &PotBinData,
    state: &PotScfState,
    reported_charge_transfer: ArrayView1<'_, f64>,
    status: PotScfOuterIterationStatus,
    config: &ConfigDatData,
) -> Result<PotBinData> {
    ensure!(
        scf_pot_status_has_final_pot(status),
        "POT final pot.bin candidate requires converged or iteration-limit SCF status, got {status:?}"
    );
    let potential_count = pot.potential_count();
    ensure!(
        state.norman_charges.len() == potential_count
            && reported_charge_transfer.len() == potential_count
            && state.occupancy_by_l.ncols() == potential_count
            && state.occupancy_by_l.nrows() > 0
            && state.overlapped_density.dim() == pot.electron_density.dim()
            && state.overlapped_valence_density.dim() == pot.valence_density.dim()
            && state.coulomb_potential.dim() == pot.coulomb_potential.dim(),
        "POT final SCF state shapes do not match pot.bin candidate"
    );
    ensure!(
        state.fermi_energy.is_finite(),
        "POT final SCF Fermi level is non-finite"
    );
    ensure!(
        reported_charge_transfer
            .iter()
            .all(|charge| charge.is_finite()),
        "POT final SCF reported charge transfer is non-finite"
    );

    let mut final_pot = pot.clone();
    final_pot.raw_text = None;
    final_pot.scalars.fermi_level = state.fermi_energy;
    // FEFF keeps raw Norman charges in `qnrm` while iterating, then converts
    // them exactly once for terminal pot.bin output (`-qnrm + xion`).
    final_pot.norman_charges = reported_charge_transfer.to_owned();
    final_pot.valence_occupancy = state.occupancy_by_l.clone();
    final_pot.electron_density = state.overlapped_density.clone();
    final_pot.valence_density = state.overlapped_valence_density.clone();
    final_pot.coulomb_potential = state.coulomb_potential.clone();
    scf_pot_repair_trailing_valence_orbital_occupancy(&mut final_pot, config)
        .context("failed to repair POT final SCF orbital occupancy from config.dat")?;
    pot_bin_string(&final_pot).context("POT final pot.bin candidate is not renderable")?;
    Ok(final_pot)
}

fn scf_pot_repair_trailing_valence_orbital_occupancy(
    pot: &mut PotBinData,
    config: &ConfigDatData,
) -> Result<()> {
    let tables = rhorrp_orbital_tables_from_config_dat(config)?;
    let potential_count = pot.potential_count();
    ensure!(
        tables.bound_orbital_counts.len() == potential_count,
        "POT final SCF config potential count {} does not match pot.bin potential count {potential_count}",
        tables.bound_orbital_counts.len()
    );
    ensure!(
        pot.orbital_occupancy.ncols() == potential_count,
        "POT final SCF orbital occupancy potential count {} does not match pot.bin potential count {potential_count}",
        pot.orbital_occupancy.ncols()
    );

    let (_, large_orbitals, large_potentials) = pot.large_components.dim();
    let (_, small_orbitals, small_potentials) = pot.small_components.dim();
    ensure!(
        large_potentials == potential_count && small_potentials == potential_count,
        "POT final SCF component potential counts large={large_potentials}, small={small_potentials} do not match pot.bin potential count {potential_count}"
    );

    for potential in 0..potential_count {
        let bound_orbitals = tables.bound_orbital_counts[potential];
        ensure!(
            bound_orbitals <= pot.orbital_occupancy.nrows()
                && bound_orbitals <= large_orbitals
                && bound_orbitals <= small_orbitals,
            "POT final SCF config bound orbital count {bound_orbitals} exceeds pot.bin shapes occupancy={}, large={large_orbitals}, small={small_orbitals}",
            pot.orbital_occupancy.nrows()
        );

        let active_bound_orbitals =
            scf_pot_active_bound_orbital_count(pot, potential, bound_orbitals)?;
        for orbital in active_bound_orbitals..bound_orbitals {
            let electron_count = tables.electron_counts_by_potential[(orbital, potential)];
            let valence_count = tables.valence_counts_by_potential[(orbital, potential)];
            let current_count = pot.orbital_occupancy[(orbital, potential)];
            ensure!(
                electron_count.is_finite()
                    && valence_count.is_finite()
                    && current_count.is_finite(),
                "POT final SCF orbital occupancy contains non-finite count for potential {potential} orbital {orbital}"
            );
            if valence_count.abs() > POT_SCF_ORBITAL_OCCUPANCY_TOLERANCE
                && (electron_count - valence_count).abs() <= POT_SCF_ORBITAL_OCCUPANCY_TOLERANCE
                && (current_count - valence_count).abs() > POT_SCF_ORBITAL_OCCUPANCY_TOLERANCE
            {
                pot.orbital_occupancy[(orbital, potential)] = valence_count;
            }
        }
    }

    Ok(())
}

fn scf_pot_active_bound_orbital_count(
    pot: &PotBinData,
    potential: usize,
    bound_orbitals: usize,
) -> Result<usize> {
    let radial_points = pot.large_components.len_of(Axis(0));
    ensure!(
        pot.small_components.len_of(Axis(0)) == radial_points,
        "POT final SCF large/small component radial grids do not match"
    );

    let mut active_bound_orbitals = 0usize;
    for orbital in 0..bound_orbitals {
        let mut has_component = false;
        for radial in 0..radial_points {
            if pot.large_components[(radial, orbital, potential)].abs()
                >= POT_SCF_TRAILING_ORBITAL_COMPONENT_THRESHOLD
                || pot.small_components[(radial, orbital, potential)].abs()
                    >= POT_SCF_TRAILING_ORBITAL_COMPONENT_THRESHOLD
            {
                has_component = true;
                break;
            }
        }
        if has_component {
            active_bound_orbitals = orbital + 1;
        }
    }

    Ok(active_bound_orbitals)
}

fn scf_pot_energy_grid_from_initial_state(pot: &PotBinData) -> Result<ScmtEnergyGrid> {
    let grid = scmt_energy_grid(ScmtEnergyGridInput {
        core_valence_energy: pot.scalars.core_valence_energy,
        fermi_energy: pot.scalars.fermi_level,
        max_points: POT_SCMT_MAX_ENERGY_POINTS,
        step_count: POT_SCMT_FLOOR_COUNT,
    })?;
    ensure!(
        grid.active_len > 0 && grid.active_len <= grid.energies.len(),
        "POT initial SCF energy grid active length {} is invalid for {} row(s)",
        grid.active_len,
        grid.energies.len()
    );
    ensure!(
        grid.steps.len() == POT_SCMT_FLOOR_COUNT,
        "POT initial SCF energy grid has {} floor step(s), expected {POT_SCMT_FLOOR_COUNT}",
        grid.steps.len()
    );
    Ok(grid)
}

fn scf_pot_fovrg_source_grid_plan(
    input: &PotInput,
    pot: &PotBinData,
    config: &ConfigDatData,
) -> Result<PotScfFovrgSourceGridPlan> {
    let angular_count = scf_pot_fovrg_angular_count(input)?;
    pot_scf_fovrg_source_grid_plan(PotScfFovrgSourceGridPlanInput {
        pot,
        config,
        exchange_selector: input.control.ixc,
        angular_count,
        use_hankel_boundary: false,
    })
    .context("failed to prepare reusable POT SCF FOVRG source-grid plan")
}

fn scf_pot_fovrg_source_grid_for_energies_from_plan(
    input: &PotInput,
    plan: &PotScfFovrgSourceGridPlan,
    energies: ArrayView1<'_, Complex64>,
) -> Result<PotScfFovrgSourceGridHandoff> {
    ensure!(
        !energies.is_empty(),
        "POT SCF FOVRG source grid requires at least one energy row"
    );
    let mut grid =
        pot_scf_fovrg_source_grid_handoff_from_plan(PotScfFovrgSourceGridFromPlanInput {
            plan,
            energies_hartree: energies,
        })
        .context("failed to build POT SCF FOVRG source grid from reusable plan")?;
    scf_pot_mask_inactive_fovrg_angular_channels(input, &mut grid)?;
    Ok(grid)
}

fn scf_pot_fms_source_grid_from_initial_state(
    work_dir: &Path,
    input: &PotInput,
    fovrg_grid: &PotScfFovrgSourceGridHandoff,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfFmsSourceGridHandoff> {
    let angular_count = fovrg_grid.phase_shifts.dim().1;
    ensure!(
        angular_count > 0,
        "POT initial SCF FMS source grid requires at least one phase angular channel"
    );
    let fms_angular_count = scf_pot_fms_solve_angular_count(input, angular_count)?;
    let thermal = fms_angular_count < angular_count;
    let capped_input = thermal.then(|| {
        let mut capped = input.clone();
        for potential in &mut capped.potentials {
            potential.lmaxsc = potential.lmaxsc.min(POT_THERMAL_FMS_MAX_LMAX as i32);
        }
        capped
    });
    let fms_input = capped_input.as_ref().unwrap_or(input);
    let mut grid = build_pot_scf_fms_source_grid_handoff_with_cache(
        work_dir,
        PotScfFmsSourceGridInput {
            pot: fms_input,
            energy_grid_hartree: fovrg_grid.energies_hartree.view(),
            reference_energies_hartree: fovrg_grid.reference_energies_hartree.view(),
            phase_shifts: fovrg_grid.phase_shifts.view(),
            angular_count: fms_angular_count,
        },
        fms_cache,
    )
    .context("failed to build POT initial SCF FMS source grid from FOVRG phases")?;
    if fms_angular_count < angular_count {
        let (energy_count, _, potential_count) = grid.scattering_trace.dim();
        let mut expanded =
            Array3::<Complex64>::zeros((energy_count, angular_count, potential_count));
        for energy in 0..energy_count {
            for angular in 0..fms_angular_count {
                for potential in 0..potential_count {
                    expanded[(energy, angular, potential)] =
                        grid.scattering_trace[(energy, angular, potential)];
                }
            }
        }
        grid.scattering_trace = expanded;
    }
    scf_pot_mask_inactive_fms_angular_channels(input, &mut grid)?;
    Ok(grid)
}

fn scf_pot_fms_solve_angular_count(input: &PotInput, fovrg_angular_count: usize) -> Result<usize> {
    ensure!(
        fovrg_angular_count > 0,
        "POT initial SCF FMS solve requires at least one FOVRG angular channel"
    );
    if scf_pot_uses_thermal_occupations(input)? {
        Ok(fovrg_angular_count.min(POT_THERMAL_FMS_MAX_LMAX + 1))
    } else {
        Ok(fovrg_angular_count)
    }
}

fn scf_pot_contour_source_rows_from_initial_state(
    pot: &PotBinData,
    fovrg_grid: &PotScfFovrgSourceGridHandoff,
    fms_grid: &PotScfFmsSourceGridHandoff,
) -> Result<PotScfContourSourceRows> {
    let potential_count = pot.potential_count();
    ensure!(
        potential_count > 0,
        "POT initial SCF contour source rows require at least one potential"
    );
    ensure!(
        fms_grid.energies_hartree == fovrg_grid.energies_hartree,
        "POT initial SCF FMS and FOVRG source grids use different energy rows"
    );
    let angular_count = fovrg_grid.phase_shifts.dim().1;
    let scattering_trace = pot_scf_scattering_trace_as_complex32(fms_grid.scattering_trace.view())?;
    let output_radii = scf_pot_density_output_grid();
    let rows = pot_scf_contour_source_rows(PotScfContourSourceRowsInput {
        source_energies: fovrg_grid.energies_hartree.view(),
        source_radii: fovrg_grid.source_radii.view(),
        output_radii: output_radii.view(),
        radial_step: ATOM_RADIAL_STEP,
        highest_potential_index: potential_count - 1,
        norman_radii: pot.norman_radii.view(),
        wave_numbers: fovrg_grid.wave_numbers.view(),
        angular_count,
        scattering_trace: scattering_trace.view(),
        regular_large: fovrg_grid.regular_large.view(),
        regular_small: fovrg_grid.regular_small.view(),
        irregular_large: fovrg_grid.irregular_large.view(),
        irregular_small: fovrg_grid.irregular_small.view(),
    })
    .context("failed to assemble POT initial SCF contour source rows")?;
    Ok(rows)
}

fn scf_pot_density_output_grid() -> Array1<f64> {
    // SCMT's saved ri05 table evaluates the entire expression in REAL,
    // including exp, before promoting to double for RHOLIE interpolation.
    Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
        f64::from(0.05_f32.mul_add(row as f32, -8.8).exp())
    })
}

fn scf_pot_corval_peak_energies_for_selection(
    input: &PotInput,
    config: &ConfigDatData,
    pot: &PotBinData,
    selection: &PotCoreValenceSelection,
) -> Result<Array2<f64>> {
    let potential_count = pot.potential_count();
    ensure!(
        potential_count > 0,
        "POT corval LDOS peak scan requires at least one potential"
    );
    let mut requested =
        Array2::<bool>::from_elem((ATOM_NORMAN_VALENCE_CHANNEL_COUNT, potential_count), false);
    for marker in &selection.markers {
        ensure!(
            marker.angular < ATOM_NORMAN_VALENCE_CHANNEL_COUNT
                && marker.potential < potential_count,
            "POT corval marker l={}, potential={} is outside {}x{} peak table",
            marker.angular,
            marker.potential,
            ATOM_NORMAN_VALENCE_CHANNEL_COUNT,
            potential_count
        );
        requested[(marker.angular, marker.potential)] = true;
    }
    if !requested.iter().any(|requested| *requested) {
        return Ok(Array2::from_elem(
            (ATOM_NORMAN_VALENCE_CHANNEL_COUNT, potential_count),
            f64::NAN,
        ));
    }

    scf_pot_corval_peak_energies_for_request_mask(input, config, pot, &requested)
}

fn scf_pot_corval_peak_energies_for_request_mask(
    input: &PotInput,
    config: &ConfigDatData,
    pot: &PotBinData,
    requested: &Array2<bool>,
) -> Result<Array2<f64>> {
    let potential_count = pot.potential_count();
    ensure!(
        requested.nrows() >= ATOM_NORMAN_VALENCE_CHANNEL_COUNT
            && requested.ncols() >= potential_count,
        "POT corval LDOS peak request mask shape {:?} cannot provide {}x{} channels",
        requested.dim(),
        ATOM_NORMAN_VALENCE_CHANNEL_COUNT,
        potential_count
    );
    let energies = pot_corval_scan_energy_grid(
        pot.scalars.core_valence_energy,
        input.scattering.corval_emin,
    )?;
    let mut peaks = Array2::from_elem(
        (ATOM_NORMAN_VALENCE_CHANNEL_COUNT, potential_count),
        f64::NAN,
    );
    let mut accumulated_energies: Option<Array1<Complex64>> = None;
    let mut accumulated_embedded_ldos: Option<Array3<Complex64>> = None;

    for start in (0..energies.len()).step_by(POT_CORVAL_SCAN_BATCH_POINTS) {
        let end = (start + POT_CORVAL_SCAN_BATCH_POINTS).min(energies.len());
        let batch_energies = Array1::from_vec(
            energies
                .iter()
                .skip(start)
                .take(end - start)
                .copied()
                .collect(),
        );
        let ldos = pot_scf_corval_ldos_handoff(PotScfCorvalLdosHandoffInput {
            pot,
            config,
            energies_hartree: batch_energies.view(),
            exchange_selector: input.control.ixc,
            requested_channels: requested.view(),
            use_hankel_boundary: false,
        })
        .context("failed to assemble POT corval rholie LDOS rows")?;

        accumulated_energies = Some(match accumulated_energies {
            Some(previous) => {
                concatenate(Axis(0), &[previous.view(), ldos.energies_hartree.view()])
                    .context("failed to append POT corval source energies")?
            }
            None => ldos.energies_hartree.clone(),
        });
        accumulated_embedded_ldos = Some(match accumulated_embedded_ldos {
            Some(previous) => concatenate(
                Axis(0),
                &[previous.view(), ldos.embedded_ldos_source.view()],
            )
            .context("failed to append POT corval embedded LDOS rows")?,
            None => ldos.embedded_ldos_source.clone(),
        });

        let found = pot_corval_ldos_peak_energies(
            accumulated_energies
                .as_ref()
                .context("POT corval source energies are unavailable")?
                .view(),
            accumulated_embedded_ldos
                .as_ref()
                .context("POT corval embedded LDOS rows are unavailable")?
                .view(),
            ATOM_NORMAN_VALENCE_CHANNEL_COUNT,
            end == energies.len(),
        )?;
        for angular in 0..ATOM_NORMAN_VALENCE_CHANNEL_COUNT {
            for potential in 0..potential_count {
                if requested[(angular, potential)] && peaks[(angular, potential)].is_nan() {
                    let peak = found[(angular, potential)];
                    if peak.is_finite() {
                        peaks[(angular, potential)] = peak;
                    }
                }
            }
        }
        if corval_peak_requests_are_satisfied(requested, &peaks, potential_count) {
            break;
        }
    }

    Ok(peaks)
}

fn corval_peak_requests_are_satisfied(
    requested: &Array2<bool>,
    peaks: &Array2<f64>,
    potential_count: usize,
) -> bool {
    (0..ATOM_NORMAN_VALENCE_CHANNEL_COUNT).all(|angular| {
        (0..potential_count).all(|potential| {
            !requested[(angular, potential)] || peaks[(angular, potential)].is_finite()
        })
    })
}

fn pot_corval_scan_energy_grid(
    core_valence_energy: f64,
    corval_emin_ev: f64,
) -> Result<Array1<Complex64>> {
    ensure!(
        core_valence_energy.is_finite(),
        "POT corval scan core-valence energy is non-finite: {core_valence_energy}"
    );
    ensure!(
        corval_emin_ev.is_finite(),
        "POT corval scan lower bound is non-finite: {corval_emin_ev}"
    );
    let lower = (corval_emin_ev / FEFF_HARTREE_EV).min(core_valence_energy);
    let upper = POT_CORVAL_HIGH_EV / FEFF_HARTREE_EV;
    let imaginary = POT_CORVAL_LDOS_IMAGINARY_EV / FEFF_HARTREE_EV;
    ensure!(
        imaginary.is_finite() && imaginary > 0.0,
        "POT corval scan imaginary broadening is invalid: {imaginary}"
    );
    if upper <= lower {
        return Ok(Array1::from_vec(vec![Complex64::new(lower, imaginary)]));
    }
    let interval_count = ((upper - lower) * 2.0 * FEFF_HARTREE_EV).round().max(1.0) as usize;
    let point_count = interval_count + 1;
    let step = (upper - lower) / interval_count as f64;
    Ok(Array1::from_shape_fn(point_count, |index| {
        Complex64::new(lower + step * index as f64, imaginary)
    }))
}

fn pot_corval_ldos_peak_energies(
    energies: ArrayView1<'_, Complex64>,
    embedded_ldos: ndarray::ArrayView3<'_, Complex64>,
    output_angular_count: usize,
    allow_terminal_peak: bool,
) -> Result<Array2<f64>> {
    ensure!(
        !energies.is_empty(),
        "POT corval LDOS peak scan requires at least one energy"
    );
    ensure!(
        output_angular_count > 0 && embedded_ldos.dim().1 >= output_angular_count,
        "POT corval LDOS peak scan has embedded_ldos shape {:?}, cannot provide {} angular channel(s)",
        embedded_ldos.dim(),
        output_angular_count
    );
    ensure!(
        embedded_ldos.dim().0 == energies.len(),
        "POT corval LDOS peak scan energy count {} does not match LDOS rows {}",
        energies.len(),
        embedded_ldos.dim().0
    );
    let potential_count = embedded_ldos.dim().2;
    let imaginary = POT_CORVAL_LDOS_IMAGINARY_EV / FEFF_HARTREE_EV;
    let mut peaks = Array2::<f64>::from_elem((output_angular_count, potential_count), f64::NAN);

    for angular in 0..output_angular_count {
        let threshold = (2 * angular + 1) as f64 / (6.0 * imaginary * std::f64::consts::PI);
        for potential in 0..potential_count {
            let mut previous = 0.0;
            for energy_index in 0..energies.len() {
                let current = embedded_ldos[(energy_index, angular, potential)].im;
                ensure!(
                    current.is_finite(),
                    "POT corval LDOS imaginary value at energy {}, l {}, potential {} is non-finite",
                    energy_index,
                    angular,
                    potential
                );
                if ((allow_terminal_peak && energy_index + 1 == energies.len())
                    || current < previous)
                    && previous > threshold
                {
                    let peak_index = energy_index.saturating_sub(1);
                    peaks[(angular, potential)] = energies[peak_index].re;
                    break;
                }
                previous = current;
            }
        }
    }

    Ok(peaks)
}

fn scf_pot_source_grids_for_energies_from_fovrg_plan(
    work_dir: &Path,
    input: &PotInput,
    pot: &PotBinData,
    fovrg_plan: &PotScfFovrgSourceGridPlan,
    energies: ArrayView1<'_, Complex64>,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<(
    PotScfFovrgSourceGridHandoff,
    PotScfFmsSourceGridHandoff,
    PotScfContourSourceRows,
)> {
    let fovrg_grid = scf_pot_fovrg_source_grid_for_energies_from_plan(input, fovrg_plan, energies)
        .context("failed to build POT SCF FOVRG source grid")?;
    let fms_grid =
        scf_pot_fms_source_grid_from_initial_state(work_dir, input, &fovrg_grid, fms_cache)
            .context("failed to build POT SCF FMS source grid")?;
    let contour_rows = scf_pot_contour_source_rows_from_initial_state(pot, &fovrg_grid, &fms_grid)
        .context("failed to assemble POT SCF contour source rows")?;
    Ok((fovrg_grid, fms_grid, contour_rows))
}

fn append_scf_pot_contour_source_rows(
    accumulated: Option<PotScfContourSourceRows>,
    next: PotScfContourSourceRows,
) -> Result<PotScfContourSourceRows> {
    let Some(accumulated) = accumulated else {
        return Ok(next);
    };

    Ok(PotScfContourSourceRows {
        source_energies: concatenate(
            Axis(0),
            &[
                accumulated.source_energies.view(),
                next.source_energies.view(),
            ],
        )
        .context("failed to append POT SCF source energies")?,
        scattering_trace: concatenate(
            Axis(0),
            &[
                accumulated.scattering_trace.view(),
                next.scattering_trace.view(),
            ],
        )
        .context("failed to append POT SCF scattering trace rows")?,
        scattering_ldos: concatenate(
            Axis(0),
            &[
                accumulated.scattering_ldos.view(),
                next.scattering_ldos.view(),
            ],
        )
        .context("failed to append POT SCF scattering LDOS rows")?,
        embedded_ldos_source: concatenate(
            Axis(0),
            &[
                accumulated.embedded_ldos_source.view(),
                next.embedded_ldos_source.view(),
            ],
        )
        .context("failed to append POT SCF embedded LDOS rows")?,
        scattering_density: concatenate(
            Axis(0),
            &[
                accumulated.scattering_density.view(),
                next.scattering_density.view(),
            ],
        )
        .context("failed to append POT SCF scattering-density rows")?,
        embedded_density_source: concatenate(
            Axis(0),
            &[
                accumulated.embedded_density_source.view(),
                next.embedded_density_source.view(),
            ],
        )
        .context("failed to append POT SCF embedded-density rows")?,
        density_scale: concatenate(
            Axis(0),
            &[accumulated.density_scale.view(), next.density_scale.view()],
        )
        .context("failed to append POT SCF density-scale rows")?,
    })
}

fn append_scf_pot_fovrg_source_grid(
    accumulated: Option<PotScfFovrgSourceGridHandoff>,
    next: PotScfFovrgSourceGridHandoff,
) -> Result<PotScfFovrgSourceGridHandoff> {
    let Some(accumulated) = accumulated else {
        return Ok(next);
    };

    ensure!(
        accumulated.source_radii == next.source_radii
            && accumulated.radial_active_counts == next.radial_active_counts
            && accumulated.rholie_active_counts == next.rholie_active_counts
            && accumulated.muffin_tin_indices_1based == next.muffin_tin_indices_1based
            && accumulated.norman_indices_1based == next.norman_indices_1based
            && accumulated.radial_handoffs.len() == next.radial_handoffs.len(),
        "POT SCF FOVRG source-grid rows use incompatible radial metadata"
    );

    Ok(PotScfFovrgSourceGridHandoff {
        source_radii: accumulated.source_radii,
        energies_hartree: concatenate(
            Axis(0),
            &[
                accumulated.energies_hartree.view(),
                next.energies_hartree.view(),
            ],
        )
        .context("failed to append POT SCF FOVRG source energies")?,
        reference_energies_hartree: concatenate(
            Axis(0),
            &[
                accumulated.reference_energies_hartree.view(),
                next.reference_energies_hartree.view(),
            ],
        )
        .context("failed to append POT SCF FOVRG reference energies")?,
        wave_numbers: concatenate(
            Axis(0),
            &[accumulated.wave_numbers.view(), next.wave_numbers.view()],
        )
        .context("failed to append POT SCF FOVRG wave numbers")?,
        regular_large: concatenate(
            Axis(0),
            &[accumulated.regular_large.view(), next.regular_large.view()],
        )
        .context("failed to append POT SCF FOVRG regular-large rows")?,
        regular_small: concatenate(
            Axis(0),
            &[accumulated.regular_small.view(), next.regular_small.view()],
        )
        .context("failed to append POT SCF FOVRG regular-small rows")?,
        irregular_large: concatenate(
            Axis(0),
            &[
                accumulated.irregular_large.view(),
                next.irregular_large.view(),
            ],
        )
        .context("failed to append POT SCF FOVRG irregular-large rows")?,
        irregular_small: concatenate(
            Axis(0),
            &[
                accumulated.irregular_small.view(),
                next.irregular_small.view(),
            ],
        )
        .context("failed to append POT SCF FOVRG irregular-small rows")?,
        phase_shifts: concatenate(
            Axis(0),
            &[accumulated.phase_shifts.view(), next.phase_shifts.view()],
        )
        .context("failed to append POT SCF FOVRG phase-shift rows")?,
        phase_amplitudes: concatenate(
            Axis(0),
            &[
                accumulated.phase_amplitudes.view(),
                next.phase_amplitudes.view(),
            ],
        )
        .context("failed to append POT SCF FOVRG phase-amplitude rows")?,
        radial_active_counts: accumulated.radial_active_counts,
        rholie_active_counts: accumulated.rholie_active_counts,
        muffin_tin_indices_1based: accumulated.muffin_tin_indices_1based,
        norman_indices_1based: accumulated.norman_indices_1based,
        radial_handoffs: accumulated.radial_handoffs,
    })
}

fn append_scf_pot_fms_source_grid(
    accumulated: Option<PotScfFmsSourceGridHandoff>,
    next: PotScfFmsSourceGridHandoff,
) -> Result<PotScfFmsSourceGridHandoff> {
    let Some(accumulated) = accumulated else {
        return Ok(next);
    };

    Ok(PotScfFmsSourceGridHandoff {
        energies_hartree: concatenate(
            Axis(0),
            &[
                accumulated.energies_hartree.view(),
                next.energies_hartree.view(),
            ],
        )
        .context("failed to append POT SCF FMS source energies")?,
        scattering_trace: concatenate(
            Axis(0),
            &[
                accumulated.scattering_trace.view(),
                next.scattering_trace.view(),
            ],
        )
        .context("failed to append POT SCF FMS scattering-trace rows")?,
    })
}

fn scf_pot_uses_thermal_occupations(input: &PotInput) -> Result<bool> {
    let temperature = input.thermal.scf_temperature;
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "POT electronic temperature must be non-negative and finite, got {temperature}"
    );
    if temperature == 0.0 {
        return Ok(false);
    }
    match input.thermal.iscfth {
        2 => Ok(true),
        1 => bail!(
            "POT thermal SCF Sommerfeld method is not implemented; use SCFTH method 2 (contour)"
        ),
        method => bail!(
            "POT thermal SCF method {method} is unsupported at positive electronic temperature; use SCFTH method 2 (contour)"
        ),
    }
}

fn scf_pot_thermal_chemical_iteration_count(input: &PotInput) -> Result<usize> {
    let count = usize::try_from(input.thermal.nmu)
        .context("POT thermal SCF nmu cannot be represented as usize")?;
    ensure!(
        count > 0 && count <= POT_THERMAL_MAX_CHEMICAL_ITERATIONS,
        "POT thermal SCF nmu {count} must be in 1..={POT_THERMAL_MAX_CHEMICAL_ITERATIONS}"
    );
    Ok(count)
}

fn scf_pot_thermal_chemical_update_stalled(current: f64, next: f64) -> Result<bool> {
    ensure!(
        current.is_finite() && next.is_finite(),
        "POT thermal SCF chemical-potential update is non-finite"
    );
    Ok((next - current).abs() < POT_THERMAL_CHEMICAL_STALL_HARTREE)
}

fn scf_pot_thermal_grid(
    input: &PotInput,
    pot: &PotBinData,
    chemical_potential: f64,
) -> Result<PotThermalScfGrid> {
    let temperature_ev = input.thermal.scf_temperature;
    ensure!(
        temperature_ev.is_finite() && temperature_ev > 0.0,
        "POT thermal SCF temperature must be positive and finite, got {temperature_ev}"
    );
    ensure!(
        chemical_potential.is_finite(),
        "POT thermal SCF chemical-potential seed is non-finite"
    );
    ensure!(
        pot.scalars.core_valence_energy.is_finite(),
        "POT thermal SCF core-valence energy is non-finite"
    );
    let energy_count = usize::try_from(input.thermal.negrid)
        .context("POT thermal SCF negrid cannot be represented as usize")?;
    ensure!(
        energy_count > POT_THERMAL_VERTICAL_POINTS && energy_count <= POT_THERMAL_MAX_GRID_POINTS,
        "POT thermal SCF negrid {energy_count} must be in {}..={POT_THERMAL_MAX_GRID_POINTS}",
        POT_THERMAL_VERTICAL_POINTS + 1
    );
    ensure!(
        input.thermal.emaxscf.is_finite() && input.thermal.emaxscf > 0.0,
        "POT thermal SCF emaxscf must be positive and finite, got {}",
        input.thermal.emaxscf
    );

    let temperature = temperature_ev / FEFF_HARTREE_EV;
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "POT thermal SCF Hartree temperature is invalid"
    );
    let mut imaginary = 2.0 * std::f64::consts::PI * temperature;
    let mut pole_count = 1usize;
    if imaginary < POT_THERMAL_MAX_IMAGINARY_HARTREE {
        pole_count = (POT_THERMAL_MAX_IMAGINARY_HARTREE / imaginary).ceil() as usize;
        ensure!(
            pole_count > 0 && pole_count <= POT_THERMAL_MAX_MATSUBARA_POLES,
            "POT thermal SCF Matsubara pole count {pole_count} exceeds the explicit resource limit {POT_THERMAL_MAX_MATSUBARA_POLES}; increase the electronic temperature above the supported minimum"
        );
        imaginary = pole_count as f64 * 2.0 * std::f64::consts::PI * temperature;
    }
    if std::f64::consts::PI * temperature / 2.0 > POT_THERMAL_MAX_IMAGINARY_HARTREE {
        imaginary = POT_THERMAL_MAX_IMAGINARY_HARTREE;
        pole_count = 0;
    }

    let upper_energy = (chemical_potential
        + temperature * (POT_THERMAL_INTERPOLATION_WINDOW + POT_THERMAL_GRID_WINDOW_PAD))
        .max(chemical_potential + input.thermal.emaxscf / FEFF_HARTREE_EV);
    ensure!(
        upper_energy.is_finite() && upper_energy > pot.scalars.core_valence_energy,
        "POT thermal SCF upper grid energy {upper_energy} does not exceed ecv {}",
        pot.scalars.core_valence_energy
    );
    let horizontal_count = energy_count - POT_THERMAL_VERTICAL_POINTS;
    let vertical_step =
        imaginary / (POT_THERMAL_VERTICAL_POINTS * POT_THERMAL_VERTICAL_POINTS) as f64;
    let horizontal_step =
        (upper_energy - pot.scalars.core_valence_energy) / horizontal_count as f64;
    let mut energies = Array1::<Complex64>::zeros(energy_count);
    for index in 0..POT_THERMAL_VERTICAL_POINTS {
        let one_based = index + 1;
        energies[index] = Complex64::new(
            pot.scalars.core_valence_energy,
            vertical_step * (one_based * one_based) as f64,
        );
    }
    for index in 0..horizontal_count {
        let one_based = index + 1;
        energies[POT_THERMAL_VERTICAL_POINTS + index] = Complex64::new(
            pot.scalars.core_valence_energy + horizontal_step * one_based as f64,
            imaginary,
        );
    }
    ensure!(
        energies
            .iter()
            .all(|energy| energy.re.is_finite() && energy.im.is_finite()),
        "POT thermal SCF grid contains a non-finite energy"
    );
    Ok(PotThermalScfGrid {
        energies,
        pole_count,
    })
}

fn scf_pot_thermal_densities_from_rows(
    rows: &PotScfContourSourceRows,
    include_high_l: bool,
) -> Result<PotThermalScfDensities> {
    let (energy_count, angular_count, potential_count) = rows.scattering_trace.dim();
    let radial_count = rows.embedded_density_source.dim().1;
    ensure!(
        rows.source_energies.len() == energy_count
            && rows.scattering_ldos.dim() == (energy_count, angular_count, potential_count)
            && rows.embedded_ldos_source.dim() == (energy_count, angular_count, potential_count)
            && rows.scattering_density.dim()
                == (energy_count, radial_count, angular_count, potential_count)
            && rows.embedded_density_source.dim() == (energy_count, radial_count, potential_count),
        "POT thermal SCF source-row shapes are inconsistent"
    );
    let mut angular = rows.embedded_ldos_source.clone();
    let mut radial = rows.embedded_density_source.clone();
    for energy in 0..energy_count {
        for potential in 0..potential_count {
            for momentum in 0..angular_count {
                let trace = rows.scattering_trace[(energy, momentum, potential)];
                let trace = Complex64::new(f64::from(trace.re), f64::from(trace.im));
                angular[(energy, momentum, potential)] +=
                    trace * rows.scattering_ldos[(energy, momentum, potential)];
                if include_high_l || momentum <= 2 {
                    for radius in 0..radial_count {
                        radial[(energy, radius, potential)] +=
                            trace * rows.scattering_density[(energy, radius, momentum, potential)];
                    }
                }
            }
        }
    }
    ensure!(
        angular
            .iter()
            .chain(radial.iter())
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "POT thermal SCF source densities contain a non-finite value"
    );
    Ok(PotThermalScfDensities { angular, radial })
}

fn scf_pot_thermal_fermi(
    energy: Complex64,
    temperature: f64,
    chemical_potential: f64,
) -> Result<Complex64> {
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "POT thermal SCF Fermi function requires positive finite temperature"
    );
    let reduced = (energy - Complex64::new(chemical_potential, 0.0)) / temperature;
    if reduced.re > 500.0 {
        return Ok(Complex64::new(0.0, 0.0));
    }
    let denominator = Complex64::new(1.0, 0.0) + reduced.exp();
    ensure!(
        denominator.re.is_finite() && denominator.im.is_finite() && denominator.norm_sqr() > 0.0,
        "POT thermal SCF Fermi function reached a non-finite pole"
    );
    Ok(Complex64::new(1.0, 0.0) / denominator)
}

fn scf_pot_thermal_interpolation_index(
    energies: ArrayView1<'_, Complex64>,
    energy: f64,
) -> Result<usize> {
    ensure!(
        energies.len() >= 2 && energy.is_finite(),
        "POT thermal SCF interpolation requires a finite energy and at least two rows"
    );
    let index = energies
        .iter()
        .position(|candidate| candidate.re >= energy)
        .unwrap_or(energies.len());
    ensure!(
        index > 0 && index < energies.len(),
        "POT thermal SCF interpolation energy {energy} is outside [{}, {}]",
        energies[0].re,
        energies[energies.len() - 1].re
    );
    ensure!(
        energies[index].re > energies[index - 1].re,
        "POT thermal SCF interpolation interval collapsed at energy {energy}"
    );
    Ok(index)
}

fn scf_pot_thermal_integral(
    input: &PotInput,
    pot: &PotBinData,
    grid: &PotThermalScfGrid,
    contour: &PotThermalScfDensities,
    poles: Option<&PotThermalScfDensities>,
    chemical_potential: f64,
    last_indices: ArrayView1<'_, usize>,
) -> Result<PotThermalScfIntegral> {
    let temperature = input.thermal.scf_temperature / FEFF_HARTREE_EV;
    let energy_count = grid.energies.len();
    let (contour_count, angular_count, potential_count) = contour.angular.dim();
    let radial_count = contour.radial.dim().1;
    ensure!(
        contour_count == energy_count
            && contour.radial.dim() == (energy_count, radial_count, potential_count)
            && last_indices.len() == potential_count
            && pot.potential_multiplicities.len() == potential_count,
        "POT thermal SCF integration inputs have inconsistent shapes"
    );
    match (grid.pole_count, poles) {
        (0, None) => {}
        (count, Some(poles)) if count > 0 => {
            ensure!(
                poles.angular.dim() == (count, angular_count, potential_count)
                    && poles.radial.dim() == (count, radial_count, potential_count),
                "POT thermal SCF Matsubara density shapes are inconsistent"
            );
        }
        _ => bail!("POT thermal SCF Matsubara density handoff is incomplete"),
    }
    ensure!(
        last_indices
            .iter()
            .all(|count| *count > 0 && *count <= radial_count),
        "POT thermal SCF radial integration bounds are invalid"
    );

    let window = POT_THERMAL_INTERPOLATION_WINDOW * temperature;
    let mut lower = chemical_potential - window;
    let upper = chemical_potential + window;
    let upper_index = scf_pot_thermal_interpolation_index(grid.energies.view(), upper)
        .context("POT thermal SCF upper interpolation window left the source grid")?;
    let lower_index = grid
        .energies
        .iter()
        .position(|candidate| candidate.re >= lower)
        .unwrap_or(energy_count);
    let prefix_count = if lower_index == 0 {
        lower = grid.energies[POT_THERMAL_VERTICAL_POINTS].re;
        POT_THERMAL_VERTICAL_POINTS + 1
    } else {
        lower_index
    };
    ensure!(
        prefix_count > 0 && prefix_count <= energy_count && upper_index < energy_count,
        "POT thermal SCF interpolation window produced invalid source bounds"
    );
    let interpolation_step = (upper - lower) / POT_THERMAL_INTERPOLATION_POINTS as f64;
    ensure!(
        interpolation_step.is_finite() && interpolation_step > 0.0,
        "POT thermal SCF interpolation step is invalid"
    );

    let mut occupancy_by_l = Array2::<f64>::zeros((angular_count, potential_count));
    let mut valence_density = Array2::<f64>::zeros((radial_count, potential_count));
    let mut previous_energy = Complex64::new(grid.energies[0].re, 0.0);
    let mut previous_angular = contour.angular.index_axis(Axis(0), 0).to_owned();
    let mut previous_radial = contour.radial.index_axis(Axis(0), 0).to_owned();
    let mut previous_fermi =
        scf_pot_thermal_fermi(previous_energy, temperature, chemical_potential)?;

    let integrate_row = |energy: Complex64,
                         current_angular: ndarray::ArrayView2<'_, Complex64>,
                         current_radial: ndarray::ArrayView2<'_, Complex64>,
                         occupancy: &mut Array2<f64>,
                         radial_density: &mut Array2<f64>,
                         previous_energy_ref: &mut Complex64,
                         previous_angular_ref: &mut Array2<Complex64>,
                         previous_radial_ref: &mut Array2<Complex64>,
                         previous_fermi_ref: &mut Complex64|
     -> Result<()> {
        let current_fermi = scf_pot_thermal_fermi(energy, temperature, chemical_potential)?;
        let energy_step = energy - *previous_energy_ref;
        for potential in 0..potential_count {
            for angular in 0..angular_count {
                if input.run.iunf != 0 || angular <= 2 {
                    occupancy[(angular, potential)] += ((current_angular[(angular, potential)]
                        * current_fermi
                        + previous_angular_ref[(angular, potential)] * *previous_fermi_ref)
                        * energy_step)
                        .im;
                }
            }
            for radius in 0..last_indices[potential] {
                radial_density[(radius, potential)] += ((current_radial[(radius, potential)]
                    * current_fermi
                    + previous_radial_ref[(radius, potential)] * *previous_fermi_ref)
                    * energy_step)
                    .im;
            }
        }
        *previous_energy_ref = energy;
        previous_angular_ref.assign(&current_angular);
        previous_radial_ref.assign(&current_radial);
        *previous_fermi_ref = current_fermi;
        Ok(())
    };

    for index in 0..prefix_count {
        integrate_row(
            grid.energies[index],
            contour.angular.index_axis(Axis(0), index),
            contour.radial.index_axis(Axis(0), index),
            &mut occupancy_by_l,
            &mut valence_density,
            &mut previous_energy,
            &mut previous_angular,
            &mut previous_radial,
            &mut previous_fermi,
        )?;
    }
    for index in 0..POT_THERMAL_INTERPOLATION_POINTS {
        let energy_real = lower + interpolation_step * index as f64;
        let right = scf_pot_thermal_interpolation_index(grid.energies.view(), energy_real)?;
        let left = right - 1;
        let fraction = (energy_real - grid.energies[left].re)
            / (grid.energies[right].re - grid.energies[left].re);
        let current_angular = contour.angular.index_axis(Axis(0), left).to_owned()
            + (contour.angular.index_axis(Axis(0), right).to_owned()
                - contour.angular.index_axis(Axis(0), left))
                * fraction;
        let current_radial = contour.radial.index_axis(Axis(0), left).to_owned()
            + (contour.radial.index_axis(Axis(0), right).to_owned()
                - contour.radial.index_axis(Axis(0), left))
                * fraction;
        integrate_row(
            Complex64::new(energy_real, grid.energies[POT_THERMAL_VERTICAL_POINTS].im),
            current_angular.view(),
            current_radial.view(),
            &mut occupancy_by_l,
            &mut valence_density,
            &mut previous_energy,
            &mut previous_angular,
            &mut previous_radial,
            &mut previous_fermi,
        )?;
    }
    for index in upper_index..energy_count {
        integrate_row(
            grid.energies[index],
            contour.angular.index_axis(Axis(0), index),
            contour.radial.index_axis(Axis(0), index),
            &mut occupancy_by_l,
            &mut valence_density,
            &mut previous_energy,
            &mut previous_angular,
            &mut previous_radial,
            &mut previous_fermi,
        )?;
    }

    if let Some(poles) = poles {
        let residue = Complex64::new(0.0, -4.0 * std::f64::consts::PI * temperature);
        for pole in 0..grid.pole_count {
            for potential in 0..potential_count {
                for angular in 0..angular_count {
                    if input.run.iunf != 0 || angular <= 2 {
                        occupancy_by_l[(angular, potential)] +=
                            (residue * poles.angular[(pole, angular, potential)]).im;
                    }
                }
                for radius in 0..last_indices[potential] {
                    valence_density[(radius, potential)] +=
                        (residue * poles.radial[(pole, radius, potential)]).im;
                }
            }
        }
    }

    let mut electron_count = 0.0;
    for potential in 0..potential_count {
        for angular in 0..angular_count {
            electron_count +=
                occupancy_by_l[(angular, potential)] * pot.potential_multiplicities[potential];
        }
    }
    ensure!(
        electron_count.is_finite()
            && occupancy_by_l.iter().all(|value| value.is_finite())
            && valence_density.iter().all(|value| value.is_finite()),
        "POT thermal SCF integration produced a non-finite result"
    );
    Ok(PotThermalScfIntegral {
        electron_count,
        occupancy_by_l,
        valence_density,
    })
}

fn scf_pot_thermal_bad_occupation_count(
    actual: ArrayView2<'_, f64>,
    expected: ArrayView2<'_, f64>,
) -> Result<usize> {
    ensure!(
        actual.dim() == expected.dim(),
        "POT thermal SCF occupation shapes actual={:?}, expected={:?} differ",
        actual.dim(),
        expected.dim()
    );
    let mut bad = 0usize;
    for ((angular, potential), value) in actual.indexed_iter() {
        let difference = (*value - expected[(angular, potential)]).abs();
        let limit = match angular {
            0 => 1.95,
            1 => 5.1,
            2 => 9.1,
            _ => 13.1,
        };
        if difference > limit {
            bad += 1;
        }
    }
    Ok(bad)
}

#[allow(clippy::too_many_arguments)]
fn scf_pot_thermal_state_advance(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    pot: &PotBinData,
    grid: &PotThermalScfGrid,
    contour_densities: &PotThermalScfDensities,
    integral: PotThermalScfIntegral,
    chemical_potential: f64,
    last_indices: ArrayView1<'_, usize>,
    state: &PotScfState,
    iteration: usize,
) -> Result<PotScfStateAdvance> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        integral.occupancy_by_l.ncols() == unique_count
            && integral.occupancy_by_l.nrows() <= pot.valence_occupancy.nrows(),
        "POT thermal SCF occupation output shape {:?} is incompatible with {:?}",
        integral.occupancy_by_l.dim(),
        pot.valence_occupancy.dim()
    );
    let mut occupancy_by_l = Array2::<f64>::zeros(pot.valence_occupancy.dim());
    for potential in 0..unique_count {
        for angular in 0..integral.occupancy_by_l.nrows() {
            occupancy_by_l[(angular, potential)] = integral.occupancy_by_l[(angular, potential)];
        }
    }
    let bad_occupation_count =
        scf_pot_thermal_bad_occupation_count(occupancy_by_l.view(), pot.valence_occupancy.view())?;
    let repeat_required = iteration > 1 && bad_occupation_count > 0;

    let last_energy_index = grid.energies.len() - 1;
    let previous_energy_index = last_energy_index.saturating_sub(1);
    let embedded_ldos = contour_densities
        .angular
        .index_axis(Axis(0), last_energy_index)
        .to_owned();
    let previous_ldos = contour_densities
        .angular
        .index_axis(Axis(0), previous_energy_index)
        .to_owned();
    let embedded_density = contour_densities
        .radial
        .index_axis(Axis(0), last_energy_index)
        .to_owned();
    let previous_density = contour_densities
        .radial
        .index_axis(Axis(0), previous_energy_index)
        .to_owned();
    let contour = PotScfContourRun {
        status: PotScfContourRunStatus::Bracketed,
        energy_points_used: grid.energies.len(),
        current_energy: grid.energies[last_energy_index],
        previous_energy: grid.energies[previous_energy_index],
        current_floor: 1,
        previous_floor: 1,
        direction: 1,
        can_step_up: false,
        current_electron_delta: integral.electron_count - scf_pot_electron_count_target(pot)?,
        previous_electron_delta: integral.electron_count - scf_pot_electron_count_target(pot)?,
        total_electron_count: integral.electron_count,
        left_sum: Complex64::new(0.0, 0.0),
        right_sum: Complex64::new(0.0, 0.0),
        fermi_energy: Some(chemical_potential),
        interpolation_fraction: None,
        embedded_ldos,
        previous_ldos,
        embedded_density,
        previous_density,
        valence_density: integral.valence_density,
        occupancy_by_l,
    };

    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let atomic_numbers = atomic_apot_usize_atomic_numbers(static_arrays, unique_count)?;
    let density_step = if repeat_required {
        None
    } else {
        Some(update_scf_density_potential(ScfDensityStepInput {
            iteration,
            accelerator: input.scattering.ca1,
            coulomb_mode: scf_pot_coulomb_mode(input),
            highest_potential_index: unique_count - 1,
            valence_occupancy: pot.valence_occupancy.view(),
            last_indices,
            potential_multiplicities: pot.potential_multiplicities.view(),
            norman_radii: pot.norman_radii.view(),
            norman_charges: state.norman_charges.view(),
            overlapped_valence_density: state.overlapped_valence_density.view(),
            integrated_valence_density: contour.valence_density.view(),
            workspace: &state.workspace,
            overlapped_density: state.overlapped_density.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            atom_potentials: atom_potentials.view(),
            atomic_numbers: atomic_numbers.view(),
            coulomb_potential: state.coulomb_potential.view(),
        })?)
    };

    let mut overlapped_density = state.overlapped_density.clone();
    let mut overlapped_valence_density = state.overlapped_valence_density.clone();
    if let Some(step) = density_step.as_ref() {
        for potential in 0..unique_count {
            for radius in 0..last_indices[potential] {
                overlapped_density[(radius, potential)] = state.overlapped_density
                    [(radius, potential)]
                    - state.overlapped_valence_density[(radius, potential)]
                    + step.valence_density[(radius, potential)];
            }
            for radius in last_indices[potential]..overlapped_density.nrows() {
                overlapped_density[(radius, potential)] = 0.0;
                overlapped_valence_density[(radius, potential)] = 0.0;
            }
        }
    }
    let iteration_result = PotScfIteration {
        status: if repeat_required {
            PotScfIterationStatus::RepeatRequired
        } else {
            PotScfIterationStatus::Updated
        },
        contour,
        density_step,
        bad_occupation_count,
        overlapped_density,
        overlapped_valence_density,
    };
    let ion_charges = scf_pot_ion_charges(input, unique_count)?;
    let outer = finish_pot_scf_outer_iteration(PotScfOuterIterationInput {
        iteration_result: &iteration_result,
        iteration,
        max_iterations: scf_pot_max_iterations(input)?,
        minimum_iterations: scf_pot_minimum_iterations(input),
        previous_fermi_energy: state.fermi_energy,
        previous_norman_charges: state.norman_charge_reference.view(),
        previous_occupancy_by_l: state.occupancy_by_l.view(),
        expected_valence_occupancy: pot.valence_occupancy.view(),
        ion_charges: ion_charges.view(),
        previous_coulomb_potential: state.coulomb_potential.view(),
        fermi_tolerance: input.tolerances.tolmu,
        charge_tolerance: input.tolerances.tolq,
        charge_sum_tolerance: POT_SCMT_CHARGE_SUM_TOLERANCE,
        partial_charge_tolerance: input.tolerances.tolqp,
    })?;
    let (norman_charges, final_occupancy, workspace) =
        if let Some(step) = iteration_result.density_step.as_ref() {
            (
                step.norman_charges.clone(),
                iteration_result.contour.occupancy_by_l.clone(),
                step.workspace.clone(),
            )
        } else {
            (
                state.norman_charges.clone(),
                state.occupancy_by_l.clone(),
                state.workspace.clone(),
            )
        };
    let next_state = PotScfState {
        fermi_energy: outer.fermi_energy,
        norman_charges,
        norman_charge_reference: outer.norman_charge_reference.clone(),
        occupancy_by_l: final_occupancy,
        overlapped_density: outer.overlapped_density.clone(),
        overlapped_valence_density: outer.overlapped_valence_density.clone(),
        coulomb_potential: outer.coulomb_potential.clone(),
        workspace,
    };
    Ok(PotScfStateAdvance {
        iteration: iteration_result,
        outer,
        state: next_state,
    })
}

#[allow(clippy::too_many_arguments)]
fn scf_pot_thermal_source_advance(
    work_dir: &Path,
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    config: &ConfigDatData,
    pot: &PotBinData,
    last_indices: ArrayView1<'_, usize>,
    state: &PotScfState,
    iteration: usize,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotThermalScfAdvance> {
    ensure!(
        scf_pot_uses_thermal_occupations(input)?,
        "POT thermal SCF advance requires positive electronic temperature"
    );
    let chemical_iteration_count = scf_pot_thermal_chemical_iteration_count(input)?;
    ensure!(
        input.thermal.xntol.is_finite() && input.thermal.xntol > 0.0,
        "POT thermal SCF xntol must be positive and finite, got {}",
        input.thermal.xntol
    );
    let target = scf_pot_electron_count_target(pot)?;
    let temperature = input.thermal.scf_temperature / FEFF_HARTREE_EV;
    let fovrg_plan = scf_pot_fovrg_source_grid_plan(input, pot, config)
        .context("failed to prepare POT thermal SCF FOVRG plan")?;
    let mut chemical_potential = state.fermi_energy;
    let mut previous_chemical_potential = chemical_potential;
    let mut previous_delta = -target;
    let mut history = Vec::<(f64, f64)>::with_capacity(chemical_iteration_count);

    let mut grid = scf_pot_thermal_grid(input, pot, chemical_potential)?;
    let (mut fovrg_grid, mut fms_grid, mut contour_rows) =
        scf_pot_source_grids_for_energies_from_fovrg_plan(
            work_dir,
            input,
            pot,
            &fovrg_plan,
            grid.energies.view(),
            fms_cache,
        )
        .context("failed to build POT thermal SCF contour source rows")?;
    let mut contour_densities =
        scf_pot_thermal_densities_from_rows(&contour_rows, input.run.iunf != 0)?;

    let mut converged = None;
    for chemical_iteration in 1..=chemical_iteration_count {
        crate::execution::advance("pot-scf", chemical_iteration - 1, chemical_iteration_count)?;
        let upper_window = chemical_potential + POT_THERMAL_INTERPOLATION_WINDOW * temperature;
        let grid_lower = grid.energies[0].re;
        let grid_upper = grid.energies[grid.energies.len() - 1].re;
        if !(upper_window > grid_lower && upper_window < grid_upper) {
            grid = scf_pot_thermal_grid(input, pot, chemical_potential)
                .context("failed to reconstruct POT thermal SCF source grid")?;
            (fovrg_grid, fms_grid, contour_rows) =
                scf_pot_source_grids_for_energies_from_fovrg_plan(
                    work_dir,
                    input,
                    pot,
                    &fovrg_plan,
                    grid.energies.view(),
                    fms_cache,
                )
                .context("failed to rebuild POT thermal SCF contour source rows")?;
            contour_densities =
                scf_pot_thermal_densities_from_rows(&contour_rows, input.run.iunf != 0)?;
            let rebuilt_upper = grid.energies[grid.energies.len() - 1].re;
            ensure!(
                upper_window > grid.energies[0].re && upper_window < rebuilt_upper,
                "POT thermal SCF grid reconstruction exhausted at chemical potential {chemical_potential}"
            );
        }

        let pole_densities = if grid.pole_count == 0 {
            None
        } else {
            let pole_energies = Array1::from_shape_fn(grid.pole_count, |pole| {
                Complex64::new(
                    chemical_potential,
                    std::f64::consts::PI * temperature * (2 * (pole + 1) - 1) as f64,
                )
            });
            let (_, _, pole_rows) = scf_pot_source_grids_for_energies_from_fovrg_plan(
                work_dir,
                input,
                pot,
                &fovrg_plan,
                pole_energies.view(),
                fms_cache,
            )
            .context("failed to build POT thermal SCF Matsubara source rows")?;
            Some(scf_pot_thermal_densities_from_rows(
                &pole_rows,
                input.run.iunf != 0,
            )?)
        };
        let integral = scf_pot_thermal_integral(
            input,
            pot,
            &grid,
            &contour_densities,
            pole_densities.as_ref(),
            chemical_potential,
            last_indices,
        )?;
        let delta = integral.electron_count - target;
        ensure!(
            delta.is_finite(),
            "POT thermal SCF chemical-potential residual is non-finite"
        );
        history.push((chemical_potential, integral.electron_count));
        if delta.abs() < input.thermal.xntol {
            converged = Some((chemical_potential, integral));
            break;
        }

        let next_chemical_potential = if chemical_iteration == 1 {
            if delta < 0.0 {
                chemical_potential + 0.1
            } else {
                chemical_potential - 0.1
            }
        } else if chemical_iteration <= POT_THERMAL_SECANT_STEPS {
            let denominator = delta - previous_delta;
            ensure!(
                denominator.is_finite() && denominator.abs() > f64::EPSILON,
                "POT thermal SCF secant denominator collapsed before convergence"
            );
            let mut step_scale = 1.0;
            let mut candidate = chemical_potential;
            let mut in_grid = false;
            for _ in 0..=9 {
                candidate = chemical_potential
                    - step_scale * delta * (chemical_potential - previous_chemical_potential)
                        / denominator;
                let lower = grid.energies[0].re;
                let upper = grid.energies[grid.energies.len() - 1].re;
                if candidate >= lower && candidate <= upper {
                    in_grid = true;
                    break;
                }
                step_scale *= 0.5;
            }
            ensure!(
                in_grid && candidate.is_finite(),
                "POT thermal SCF secant step left the finite source grid"
            );
            candidate
        } else {
            let mut sorted = history.clone();
            sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
            let right = sorted
                .iter()
                .position(|(_, count)| *count >= target)
                .context("POT thermal SCF bracketing exhausted without an upper electron count")?;
            ensure!(
                right > 0,
                "POT thermal SCF bracketing exhausted without a lower electron count"
            );
            let (lower_mu, lower_count) = sorted[right - 1];
            let (upper_mu, upper_count) = sorted[right];
            let denominator = upper_count - lower_count;
            ensure!(
                denominator.is_finite() && denominator.abs() > f64::EPSILON,
                "POT thermal SCF regula-falsi denominator collapsed"
            );
            ((upper_count - target) * lower_mu - (lower_count - target) * upper_mu) / denominator
        };
        if scf_pot_thermal_chemical_update_stalled(chemical_potential, next_chemical_potential)? {
            eprintln!(
                "warning: POT thermal SCF chemical potential stalled below {:.1e} Hartree; accepting the current FEFF plateau state",
                POT_THERMAL_CHEMICAL_STALL_HARTREE
            );
            converged = Some((chemical_potential, integral));
            break;
        }
        previous_chemical_potential = chemical_potential;
        previous_delta = delta;
        chemical_potential = next_chemical_potential;
    }

    let (chemical_potential, integral) = converged.with_context(|| {
        format!(
            "POT thermal SCF chemical-potential search did not converge in {chemical_iteration_count} iteration(s)"
        )
    })?;
    let state_advance = scf_pot_thermal_state_advance(
        input,
        static_arrays,
        pot,
        &grid,
        &contour_densities,
        integral,
        chemical_potential,
        last_indices,
        state,
        iteration,
    )?;
    Ok(PotThermalScfAdvance {
        fovrg_grid,
        fms_grid,
        contour_rows,
        state_advance,
    })
}

#[allow(clippy::too_many_arguments)]
fn scf_pot_adaptive_source_advance(
    work_dir: &Path,
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    config: &ConfigDatData,
    pot: &PotBinData,
    energy_grid: &ScmtEnergyGrid,
    last_indices: ArrayView1<'_, usize>,
    state: &PotScfState,
    iteration: usize,
    first_scmt_call: bool,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfAdaptiveSourceAdvance> {
    ensure!(
        energy_grid.active_len > 0 && energy_grid.active_len <= energy_grid.energies.len(),
        "POT SCF adaptive source advance requires a valid active energy prefix"
    );

    let prefix_len = if first_scmt_call {
        energy_grid.steps.len()
    } else {
        energy_grid.active_len
    };
    ensure!(
        prefix_len > 0 && prefix_len <= energy_grid.energies.len(),
        "POT SCF adaptive source advance prefix length {prefix_len} is invalid for {} energy row(s)",
        energy_grid.energies.len()
    );
    let max_source_points = POT_SCMT_MAX_ADAPTIVE_SOURCE_POINTS
        .max(energy_grid.active_len)
        .max(prefix_len);
    let fovrg_plan = scf_pot_fovrg_source_grid_plan(input, pot, config)
        .context("failed to prepare adaptive POT SCF FOVRG source-grid plan")?;
    let prefix_energies = Array1::from_vec(
        energy_grid
            .energies
            .iter()
            .take(prefix_len)
            .copied()
            .collect(),
    );
    let (mut accumulated_fovrg_grid, mut accumulated_fms_grid, mut accumulated_rows) =
        scf_pot_source_grids_for_energies_from_fovrg_plan(
            work_dir,
            input,
            pot,
            &fovrg_plan,
            prefix_energies.view(),
            fms_cache,
        )
        .context("failed to build adaptive POT SCF source prefix")?;
    let mut state_advance = scf_pot_state_advance_from_rows(
        input,
        static_arrays,
        pot,
        energy_grid,
        &accumulated_rows,
        last_indices,
        state,
        iteration,
        first_scmt_call,
    )
    .context("failed to advance adaptive POT SCF state from source prefix")?;
    if state_advance.outer.status != PotScfOuterIterationStatus::NeedsMoreSourcePoints {
        return Ok(PotScfAdaptiveSourceAdvance {
            fovrg_grid: accumulated_fovrg_grid,
            fms_grid: accumulated_fms_grid,
            contour_rows: accumulated_rows,
            state_advance,
        });
    }

    let mut next_energy = state_advance.iteration.contour.current_energy;
    for _ in prefix_len..max_source_points {
        let energy_row = Array1::from_vec(vec![next_energy]);
        let (fovrg_grid, fms_grid, contour_row) =
            scf_pot_source_grids_for_energies_from_fovrg_plan(
                work_dir,
                input,
                pot,
                &fovrg_plan,
                energy_row.view(),
                fms_cache,
            )
            .context("failed to build adaptive POT SCF source row")?;
        accumulated_fovrg_grid =
            append_scf_pot_fovrg_source_grid(Some(accumulated_fovrg_grid), fovrg_grid)?;
        accumulated_fms_grid =
            append_scf_pot_fms_source_grid(Some(accumulated_fms_grid), fms_grid)?;
        accumulated_rows = append_scf_pot_contour_source_rows(Some(accumulated_rows), contour_row)?;
        state_advance = scf_pot_state_advance_from_rows(
            input,
            static_arrays,
            pot,
            energy_grid,
            &accumulated_rows,
            last_indices,
            state,
            iteration,
            first_scmt_call,
        )
        .context("failed to advance adaptive POT SCF state from source rows")?;
        next_energy = state_advance.iteration.contour.current_energy;
        let status = state_advance.outer.status;
        if status != PotScfOuterIterationStatus::NeedsMoreSourcePoints {
            return Ok(PotScfAdaptiveSourceAdvance {
                fovrg_grid: accumulated_fovrg_grid,
                fms_grid: accumulated_fms_grid,
                contour_rows: accumulated_rows,
                state_advance,
            });
        }
    }

    Ok(PotScfAdaptiveSourceAdvance {
        fovrg_grid: accumulated_fovrg_grid,
        fms_grid: accumulated_fms_grid,
        contour_rows: accumulated_rows,
        state_advance,
    })
}

#[allow(clippy::too_many_arguments)]
fn scf_pot_state_advance_from_rows(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    pot: &PotBinData,
    energy_grid: &ScmtEnergyGrid,
    contour_rows: &PotScfContourSourceRows,
    last_indices: ArrayView1<'_, usize>,
    state: &PotScfState,
    iteration: usize,
    first_scmt_call: bool,
) -> Result<PotScfStateAdvance> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        iteration > 0,
        "POT SCF state advance iteration must be positive"
    );
    ensure!(
        pot.potential_count() == unique_count,
        "POT SCF state advance has {} potential(s), expected {unique_count}",
        pot.potential_count()
    );
    ensure!(
        energy_grid.active_len > 0 && energy_grid.active_len <= energy_grid.energies.len(),
        "POT SCF state advance requires a valid active energy prefix"
    );
    ensure!(
        !contour_rows.source_energies.is_empty(),
        "POT SCF state advance requires at least one contour source row"
    );
    ensure!(
        last_indices.len() == unique_count,
        "POT SCF state advance has {} last radial index value(s), expected {unique_count}",
        last_indices.len()
    );
    let max_iterations = scf_pot_max_iterations(input)?;
    let minimum_iterations = scf_pot_minimum_iterations(input);
    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let atomic_numbers = atomic_apot_usize_atomic_numbers(static_arrays, unique_count)?;
    let ion_charges = scf_pot_ion_charges(input, unique_count)?;
    let coulomb_mode = scf_pot_coulomb_mode(input);
    let electron_count_target = scf_pot_electron_count_target(pot)?;

    advance_pot_scf_state(PotScfStateAdvanceInput {
        contour: PotScfContourRunInput {
            first_scmt_call,
            electron_count_target,
            active_energy_count: energy_grid.active_len,
            floor_count: energy_grid.steps.len(),
            energy_grid: energy_grid.energies.view(),
            steps: energy_grid.steps.view(),
            source_energies: contour_rows.source_energies.view(),
            highest_potential_index: unique_count - 1,
            last_indices,
            potential_multiplicities: pot.potential_multiplicities.view(),
            scattering_trace: contour_rows.scattering_trace.view(),
            scattering_ldos: contour_rows.scattering_ldos.view(),
            embedded_ldos_source: contour_rows.embedded_ldos_source.view(),
            scattering_density: contour_rows.scattering_density.view(),
            embedded_density_source: contour_rows.embedded_density_source.view(),
            include_high_l: input.run.iunf != 0,
        },
        state,
        iteration,
        max_iterations,
        minimum_iterations,
        accelerator: input.scattering.ca1,
        coulomb_mode,
        repeat_on_bad_counts: !first_scmt_call,
        expected_valence_occupancy: pot.valence_occupancy.view(),
        norman_radii: pot.norman_radii.view(),
        ion_charges: ion_charges.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        atom_potentials: atom_potentials.view(),
        atomic_numbers: atomic_numbers.view(),
        fermi_tolerance: input.tolerances.tolmu,
        charge_tolerance: input.tolerances.tolq,
        charge_sum_tolerance: POT_SCMT_CHARGE_SUM_TOLERANCE,
        partial_charge_tolerance: input.tolerances.tolqp,
    })
    .context("failed to advance POT SCF state from source rows")
}

fn scf_pot_next_iteration_preparation_from_state(
    work_dir: &Path,
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    config: &ConfigDatData,
    pot: &PotBinData,
    state: &PotScfState,
    iteration: usize,
    fms_cache: &mut PotScfFmsPipelineCache,
) -> Result<PotScfPreparedNextIteration> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        iteration > 1,
        "POT next SCF iteration preparation requires iteration > 1, got {iteration}"
    );
    ensure!(
        pot.potential_count() == unique_count,
        "POT next SCF iteration preparation has {} potential(s), expected {unique_count}",
        pot.potential_count()
    );
    ensure!(
        state.overlapped_density.dim() == pot.electron_density.dim()
            && state.overlapped_valence_density.dim() == pot.valence_density.dim()
            && state.coulomb_potential.dim() == pot.coulomb_potential.dim()
            && state.norman_charges.len() == unique_count
            && state.norman_charge_reference.len() == unique_count
            && state.occupancy_by_l.dim() == pot.valence_occupancy.dim(),
        "POT next SCF iteration state shapes are inconsistent"
    );

    let mut next_pot = pot.clone();
    next_pot.scalars.fermi_level = state.fermi_energy;
    next_pot.norman_charges = state.norman_charges.clone();
    next_pot.electron_density = state.overlapped_density.clone();
    next_pot.valence_density = state.overlapped_valence_density.clone();
    next_pot.coulomb_potential = state.coulomb_potential.clone();

    let istprm = scf_pot_istprm_from_initial_state(input, static_arrays, &next_pot)
        .context("failed to prepare POT next SCF istprm state")?;
    apply_scf_pot_istprm_state_preserving_fermi(&mut next_pot, input, &istprm, state.fermi_energy)
        .context("failed to apply POT next SCF istprm state")?;
    let energy_grid = scf_pot_energy_grid_from_initial_state(&next_pot)
        .context("failed to prepare POT next SCF energy grid")?;
    let last_indices = scf_pot_rholie_last_indices(&next_pot)
        .context("failed to derive POT next SCF rholie radial bounds")?;
    let next_state = PotScfState {
        fermi_energy: state.fermi_energy,
        norman_charges: state.norman_charges.clone(),
        norman_charge_reference: state.norman_charge_reference.clone(),
        occupancy_by_l: state.occupancy_by_l.clone(),
        overlapped_density: state.overlapped_density.clone(),
        overlapped_valence_density: state.overlapped_valence_density.clone(),
        coulomb_potential: state.coulomb_potential.clone(),
        workspace: state.workspace.clone(),
    };
    let (
        fovrg_grid,
        fovrg_grid_unavailable,
        fms_grid,
        fms_grid_unavailable,
        contour_rows,
        contour_rows_unavailable,
        state_advance,
        state_advance_unavailable,
    ) = match if scf_pot_uses_thermal_occupations(input)? {
        scf_pot_thermal_source_advance(
            work_dir,
            input,
            static_arrays,
            config,
            &next_pot,
            last_indices.view(),
            &next_state,
            iteration,
            fms_cache,
        )
        .map(|output| PotScfAdaptiveSourceAdvance {
            fovrg_grid: output.fovrg_grid,
            fms_grid: output.fms_grid,
            contour_rows: output.contour_rows,
            state_advance: output.state_advance,
        })
    } else {
        scf_pot_adaptive_source_advance(
            work_dir,
            input,
            static_arrays,
            config,
            &next_pot,
            &energy_grid,
            last_indices.view(),
            &next_state,
            iteration,
            false,
            fms_cache,
        )
    } {
        Ok(output) => (
            Some(output.fovrg_grid),
            None,
            Some(output.fms_grid),
            None,
            Some(output.contour_rows),
            None,
            Some(output.state_advance),
            None,
        ),
        Err(error) => {
            let reason = format!("{error:#}");
            (
                None,
                Some(reason.clone()),
                None,
                Some("POT next SCF FOVRG source grid is unavailable".to_string()),
                None,
                Some("POT next SCF FMS source grid is unavailable".to_string()),
                None,
                Some(reason),
            )
        }
    };

    Ok(PotScfPreparedNextIteration {
        iteration,
        pot: next_pot,
        state: next_state,
        istprm,
        energy_grid,
        fovrg_grid,
        fovrg_grid_unavailable,
        fms_grid,
        fms_grid_unavailable,
        contour_rows,
        contour_rows_unavailable,
        state_advance,
        state_advance_unavailable,
        last_indices,
    })
}

fn scf_pot_max_iterations(input: &PotInput) -> Result<usize> {
    let max_iterations = usize::try_from(input.run.nscmt)
        .context("POT initial SCF nscmt cannot be represented as usize")?;
    ensure!(
        max_iterations > 0,
        "POT initial SCF state advance requires positive nscmt"
    );
    Ok(max_iterations)
}

fn scf_pot_minimum_iterations(input: &PotInput) -> usize {
    if input.start_from_file { 2 } else { 3 }
}

fn scf_pot_coulomb_mode(input: &PotInput) -> CoulombUpdateMode {
    if input.run.icoul == 1 {
        CoulombUpdateMode::LongRange
    } else {
        CoulombUpdateMode::Norman
    }
}

fn scf_pot_ion_charges(input: &PotInput, unique_count: usize) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT initial SCF input has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut ion_charges = Array1::<f64>::zeros(unique_count);
    for (potential_index, potential) in input.potentials.iter().enumerate() {
        ensure!(
            potential.xion.is_finite(),
            "POT initial SCF ion charge for potential {potential_index} is non-finite"
        );
        ion_charges[potential_index] = potential.xion;
    }
    Ok(ion_charges)
}

fn scf_pot_electron_count_target(pot: &PotBinData) -> Result<f64> {
    let potential_count = pot.potential_count();
    ensure!(
        pot.valence_occupancy.ncols() == potential_count
            && pot.potential_multiplicities.len() == potential_count,
        "POT initial SCF valence occupation shape {:?} or multiplicity length {} does not match {potential_count} potential(s)",
        pot.valence_occupancy.dim(),
        pot.potential_multiplicities.len()
    );
    let mut target = 0.0;
    for potential_index in 0..potential_count {
        let multiplicity = pot.potential_multiplicities[potential_index];
        ensure!(
            multiplicity.is_finite(),
            "POT initial SCF multiplicity for potential {potential_index} is non-finite"
        );
        for angular in 0..pot.valence_occupancy.nrows() {
            let occupation = pot.valence_occupancy[(angular, potential_index)];
            ensure!(
                occupation.is_finite(),
                "POT initial SCF valence occupation for l {angular}, potential {potential_index} is non-finite"
            );
            target += multiplicity * occupation;
        }
    }
    ensure!(
        target.is_finite(),
        "POT initial SCF electron count target is non-finite"
    );
    Ok(target)
}

fn pot_scf_scattering_trace_as_complex32(
    trace: ndarray::ArrayView3<'_, num_complex::Complex64>,
) -> Result<Array3<Complex32>> {
    let mut narrowed = Array3::<Complex32>::zeros(trace.dim());
    for ((energy, angular, potential), value) in trace.indexed_iter() {
        let real =
            narrow_pot_scf_fms_trace_component(value.re, "real", energy, angular, potential)?;
        let imaginary =
            narrow_pot_scf_fms_trace_component(value.im, "imaginary", energy, angular, potential)?;
        narrowed[(energy, angular, potential)] = Complex32::new(real, imaginary);
    }
    Ok(narrowed)
}

fn narrow_pot_scf_fms_trace_component(
    value: f64,
    component: &'static str,
    energy: usize,
    angular: usize,
    potential: usize,
) -> Result<f32> {
    ensure!(
        value.is_finite() && value.abs() <= f32::MAX as f64,
        "POT initial SCF FMS trace {component} component at energy {energy}, l {angular}, potential {potential} is not finite single precision"
    );
    Ok(value as f32)
}

fn scf_pot_fovrg_angular_count(input: &PotInput) -> Result<usize> {
    let expected_potential_count = apot_unique_potential_count(input)?;
    ensure!(
        input.potentials.len() == expected_potential_count && expected_potential_count > 0,
        "POT initial SCF has {} potential row(s), expected {expected_potential_count} from nph={}",
        input.potentials.len(),
        input.control.nph
    );
    let mut max_lmax = 0usize;
    for (potential_index, potential) in input.potentials.iter().enumerate() {
        ensure!(
            potential.lmaxsc >= 0,
            "POT initial SCF lmaxsc for potential {potential_index} must be non-negative, got {}",
            potential.lmaxsc
        );
        let lmax = usize::try_from(potential.lmaxsc)
            .context("POT initial SCF lmaxsc cannot be represented as usize")?;
        max_lmax = max_lmax.max(lmax);
    }
    max_lmax
        .checked_add(1)
        .context("POT initial SCF angular channel count overflowed")
}

fn scf_pot_local_angular_counts(
    input: &PotInput,
    potential_count: usize,
    angular_count: usize,
) -> Result<Array1<usize>> {
    ensure!(
        angular_count > 0,
        "POT initial SCF local angular limits require at least one global channel"
    );
    ensure!(
        input.potentials.len() == potential_count,
        "POT initial SCF local angular limits have {} potential row(s), expected {potential_count}",
        input.potentials.len()
    );
    let mut counts = Array1::<usize>::zeros(potential_count);
    for (potential_index, potential) in input.potentials.iter().enumerate() {
        ensure!(
            potential.lmaxsc >= 0,
            "POT initial SCF lmaxsc for potential {potential_index} must be non-negative, got {}",
            potential.lmaxsc
        );
        let count = usize::try_from(potential.lmaxsc)
            .context("POT initial SCF lmaxsc cannot be represented as usize")?
            .checked_add(1)
            .context("POT initial SCF local angular channel count overflowed")?;
        ensure!(
            count <= angular_count,
            "POT initial SCF local angular channel count {count} for potential {potential_index} exceeds global count {angular_count}"
        );
        counts[potential_index] = count;
    }
    Ok(counts)
}

/// FEFF `POT/rholie.f90` solves only `l=0..lmaxsc(iph)` for each potential.
///
/// The reusable IO plan is intentionally rectangular and therefore uses the
/// largest requested angular count. Mask its packed result back to FEFF's
/// per-potential limits before FMS and contour integration.
fn scf_pot_mask_inactive_fovrg_angular_channels(
    input: &PotInput,
    grid: &mut PotScfFovrgSourceGridHandoff,
) -> Result<()> {
    let (energy_count, angular_count, potential_count) = grid.phase_shifts.dim();
    ensure!(
        grid.phase_amplitudes.dim() == (energy_count, angular_count, potential_count),
        "POT initial SCF FOVRG phase shapes disagree: shifts={:?}, amplitudes={:?}",
        grid.phase_shifts.dim(),
        grid.phase_amplitudes.dim()
    );
    let radial_shape = grid.regular_large.dim();
    ensure!(
        radial_shape.0 == energy_count
            && radial_shape.1 == potential_count
            && radial_shape.2 == angular_count
            && grid.regular_small.dim() == radial_shape
            && grid.irregular_large.dim() == radial_shape
            && grid.irregular_small.dim() == radial_shape,
        "POT initial SCF FOVRG radial shapes disagree with phase shape {:?}: regular_large={:?}, regular_small={:?}, irregular_large={:?}, irregular_small={:?}",
        grid.phase_shifts.dim(),
        grid.regular_large.dim(),
        grid.regular_small.dim(),
        grid.irregular_large.dim(),
        grid.irregular_small.dim()
    );
    let local_counts = scf_pot_local_angular_counts(input, potential_count, angular_count)?;
    for potential in 0..potential_count {
        for angular in local_counts[potential]..angular_count {
            for energy in 0..energy_count {
                grid.phase_shifts[(energy, angular, potential)] = Complex64::new(0.0, 0.0);
                grid.phase_amplitudes[(energy, angular, potential)] = Complex64::new(0.0, 0.0);
                for radial in 0..radial_shape.3 {
                    grid.regular_large[(energy, potential, angular, radial)] =
                        Complex64::new(0.0, 0.0);
                    grid.regular_small[(energy, potential, angular, radial)] =
                        Complex64::new(0.0, 0.0);
                    grid.irregular_large[(energy, potential, angular, radial)] =
                        Complex64::new(0.0, 0.0);
                    grid.irregular_small[(energy, potential, angular, radial)] =
                        Complex64::new(0.0, 0.0);
                }
            }
        }
    }
    Ok(())
}

fn scf_pot_mask_inactive_fms_angular_channels(
    input: &PotInput,
    grid: &mut PotScfFmsSourceGridHandoff,
) -> Result<()> {
    let (energy_count, angular_count, potential_count) = grid.scattering_trace.dim();
    ensure!(
        grid.energies_hartree.len() == energy_count,
        "POT initial SCF FMS trace energy count {energy_count} does not match grid length {}",
        grid.energies_hartree.len()
    );
    let local_counts = scf_pot_local_angular_counts(input, potential_count, angular_count)?;
    for potential in 0..potential_count {
        for angular in local_counts[potential]..angular_count {
            for energy in 0..energy_count {
                grid.scattering_trace[(energy, angular, potential)] = Complex64::new(0.0, 0.0);
            }
        }
    }
    Ok(())
}

fn scf_pot_rholie_last_indices(pot: &PotBinData) -> Result<Array1<usize>> {
    let potential_count = pot.potential_count();
    ensure!(
        pot.norman_radii.len() == potential_count,
        "POT SCF rholie last-index setup has {} Norman radii, expected {potential_count}",
        pot.norman_radii.len()
    );
    let mut last_indices = Array1::<usize>::zeros(potential_count);
    for potential in 0..potential_count {
        let norman_radius = pot.norman_radii[potential];
        ensure!(
            norman_radius.is_finite() && norman_radius > 0.0,
            "POT SCF rholie Norman radius for potential {potential} must be positive and finite, got {norman_radius}"
        );
        let raw_index = (norman_radius.ln() - ATOM_FIRST_RADIUS_LOG) / ATOM_RADIAL_STEP + 5.0;
        ensure!(
            raw_index.is_finite() && raw_index >= 1.0 && raw_index <= usize::MAX as f64,
            "POT SCF rholie last index for potential {potential} is out of range: {raw_index}"
        );
        last_indices[potential] = (raw_index.trunc() as usize).min(POT_BIN_RADIAL_POINTS);
    }
    Ok(last_indices)
}

fn validate_scf_pot_density_step_from_initial_state(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    pot: &PotBinData,
    last_indices: ArrayView1<'_, usize>,
) -> Result<BroydenWorkspace> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        pot.potential_count() == unique_count,
        "POT initial SCF state has {} potential(s), expected {unique_count}",
        pot.potential_count()
    );
    let max_iterations = scf_pot_max_iterations(input)?;
    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let workspace = BroydenWorkspace::zeros(max_iterations, unique_count);
    let coulomb_mode = scf_pot_coulomb_mode(input);

    update_scf_density_potential(ScfDensityStepInput {
        iteration: 1,
        accelerator: input.scattering.ca1,
        coulomb_mode,
        highest_potential_index: unique_count - 1,
        valence_occupancy: pot.valence_occupancy.view(),
        last_indices,
        potential_multiplicities: pot.potential_multiplicities.view(),
        norman_radii: pot.norman_radii.view(),
        norman_charges: pot.norman_charges.view(),
        overlapped_valence_density: pot.valence_density.view(),
        integrated_valence_density: pot.valence_density.view(),
        workspace: &workspace,
        overlapped_density: pot.electron_density.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        atom_potentials: atom_potentials.view(),
        atomic_numbers: pot.atomic_numbers.view(),
        coulomb_potential: pot.coulomb_potential.view(),
    })
    .context("failed to validate POT initial SCF density/coulomb update")?;
    Ok(workspace)
}

fn scf_pot_istprm_from_initial_state(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    pot: &PotBinData,
) -> Result<MuffinTinInterstitialParameters> {
    let unique_count = apot_unique_potential_count(input)?;
    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let explicit_overlaps = no_scf_pot_muffin_tin_overlaps(static_arrays, unique_count)?;
    let explicit_overlap_refs = explicit_overlaps
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let interstitial_selector = usize::try_from(input.run.inters)
        .context("POT initial SCF interstitial selector cannot be represented as usize")?;
    let radius_input = MuffinTinRadiusParametersInput {
        highest_potential_index: unique_count - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        explicit_overlaps: &explicit_overlap_refs,
        norman_radii: pot.norman_radii.view(),
        overlap_factors: pot.overlap_factors.view(),
        max_overlap_factors: pot.max_overlap_factors.view(),
        coulomb_potential: pot.coulomb_potential.view(),
        afolp_enabled: input.control.iafolp > 0,
        interstitial_selector,
    };
    let radius_state = match muffin_tin_radius_parameters(radius_input) {
        Ok(state) => Some(state),
        Err(GridError::NoMuffinTinNeighbor { .. }) if unique_count == 1 => None,
        Err(error) => {
            return Err(error)
                .context("failed to calculate POT initial SCF muffin-tin radius state");
        }
    };
    let muffin_tin_radii = radius_state
        .as_ref()
        .map(|state| state.muffin_tin_radii.view())
        .unwrap_or_else(|| pot.muffin_tin_radii.view());
    let norman_radii = radius_state
        .as_ref()
        .map(|state| state.norman_radii.view())
        .unwrap_or_else(|| pot.norman_radii.view());
    let fallback_near_neighbor_flags = Array1::<bool>::from_elem(unique_count, false);
    let near_neighbor_flags = radius_state
        .as_ref()
        .map(|state| state.near_neighbor_flags.view())
        .unwrap_or_else(|| fallback_near_neighbor_flags.view());
    let interstitial_selector = radius_state
        .as_ref()
        .map(|state| state.interstitial_selector)
        .unwrap_or(interstitial_selector);
    let result = muffin_tin_interstitial_parameters(MuffinTinInterstitialParametersInput {
        highest_potential_index: unique_count - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        potential_multiplicities: pot.potential_multiplicities.view(),
        explicit_overlaps: &explicit_overlap_refs,
        electron_density: pot.electron_density.view(),
        valence_density: pot.valence_density.view(),
        magnetization: pot.magnetization_density.view(),
        coulomb_potential: pot.coulomb_potential.view(),
        muffin_tin_radii,
        norman_radii,
        near_neighbor_flags,
        exchange_selector: input.control.ixc,
        scf_exchange_selector: input.control.iscfxc,
        spin_polarization: 0,
        scf_temperature_hartree: input.thermal.scf_temperature / FEFF_HARTREE_EV,
        total_charge: pot.scalars.total_charge,
        fermi_level: pot.scalars.fermi_level,
        total_volume: pot_input_total_volume_bohr3(input)?,
        interstitial_selector,
    })?;

    ensure!(
        result.fermi.chemical_potential.is_finite()
            && result.interstitial_density > 0.0
            && result.interstitial_potential.is_finite(),
        "POT initial SCF istprm returned invalid interstitial state"
    );
    Ok(result)
}

fn apply_scf_pot_istprm_state(
    pot: &mut PotBinData,
    input: &PotInput,
    istprm: &MuffinTinInterstitialParameters,
) -> Result<()> {
    apply_scf_pot_istprm_state_preserving_fermi(
        pot,
        input,
        istprm,
        istprm.fermi.chemical_potential,
    )?;
    // FEFF `potsub.f90` evaluates this once after the initial `istprm`
    // state: `wp=sqrt(rhoint)`. Later SCMT iterations preserve it.
    pot.scalars.plasmon_frequency = istprm.interstitial_density.sqrt();
    Ok(())
}

fn apply_scf_pot_istprm_state_preserving_fermi(
    pot: &mut PotBinData,
    input: &PotInput,
    istprm: &MuffinTinInterstitialParameters,
    fermi_level: f64,
) -> Result<()> {
    let unique_count = pot.potential_count();
    ensure!(
        istprm.total_potential.dim() == (POT_BIN_RADIAL_POINTS, unique_count)
            && istprm.valence_potential.dim() == (POT_BIN_RADIAL_POINTS, unique_count),
        "POT initial SCF istprm potential shapes total={:?}, valence={:?}, expected {POT_BIN_RADIAL_POINTS}x{unique_count}",
        istprm.total_potential.dim(),
        istprm.valence_potential.dim()
    );
    ensure!(
        istprm.max_density_indices.len() == unique_count
            && istprm.muffin_tin_indices.len() == unique_count
            && istprm.muffin_tin_radii.len() == unique_count
            && istprm.norman_indices.len() == unique_count
            && istprm.norman_radii.len() == unique_count,
        "POT initial SCF istprm radial state length mismatch for {unique_count} potential(s)"
    );
    ensure!(
        istprm.fermi.chemical_potential.is_finite()
            && istprm.interstitial_potential.is_finite()
            && istprm.interstitial_density.is_finite()
            && istprm.interstitial_density > 0.0
            && istprm.interstitial_volume.is_finite()
            && istprm.interstitial_volume > 0.0,
        "POT initial SCF istprm returned invalid scalar state"
    );
    ensure!(
        fermi_level.is_finite(),
        "POT SCF istprm Fermi level is non-finite"
    );

    pot.scalars.average_norman_radius = istprm.average_norman_radius;
    pot.scalars.fermi_level = fermi_level;
    pot.scalars.interstitial_potential = istprm.interstitial_potential;
    pot.scalars.interstitial_density = istprm.interstitial_density;
    pot.scalars.density_radius = istprm.fermi.density_parameter;
    pot.scalars.fermi_momentum = istprm.fermi.fermi_momentum;
    // FEFF computes `wp=sqrt(rhoint)` once from the pre-SCF interstitial
    // density in `potsub.f90`, then carries that value through every SCMT
    // update. Do not recompute it from the iteration's new `rhoint`.
    // FEFF passes the original `totvol` through ISTPRM and writes that input
    // field to pot.bin. The positive ISTPRM interstitial volume is a separate
    // derived quantity and must not replace a non-positive `totvol`.
    pot.scalars.total_volume = pot_input_total_volume_bohr3(input)?;
    ensure!(
        pot.scalars.plasmon_frequency.is_finite(),
        "POT initial SCF plasmon frequency is non-finite"
    );

    pot.muffin_tin_indices = istprm.muffin_tin_indices.clone();
    pot.muffin_tin_radii = istprm.muffin_tin_radii.clone();
    pot.norman_indices = istprm.norman_indices.clone();
    pot.norman_radii = istprm.norman_radii.clone();
    pot.total_potential = istprm.total_potential.clone();
    pot.valence_potential = istprm.valence_potential.clone();
    Ok(())
}

fn apply_pot_start_from_file_import_state(
    pot: &mut PotBinData,
    restart: &PotBinData,
) -> Result<()> {
    let expected = (POT_BIN_RADIAL_POINTS, pot.potential_count());
    ensure!(
        restart.total_potential.dim() == expected && restart.electron_density.dim() == expected,
        "POT START_FROM_FILE restart shapes total={:?}, density={:?}, expected {:?}",
        restart.total_potential.dim(),
        restart.electron_density.dim(),
        expected
    );
    ensure!(
        restart.scalars.fermi_level.is_finite()
            && restart.scalars.interstitial_potential.is_finite()
            && restart.scalars.interstitial_density.is_finite(),
        "POT START_FROM_FILE restart scalar state is non-finite"
    );

    pot.total_potential = restart.total_potential.clone();
    pot.electron_density = restart.electron_density.clone();
    pot.scalars.fermi_level = restart.scalars.fermi_level;
    pot.scalars.interstitial_potential = restart.scalars.interstitial_potential;
    pot.scalars.interstitial_density = restart.scalars.interstitial_density;
    Ok(())
}

fn validate_scf_pot_initial_state(initial: &PotScfInitialState) -> Result<()> {
    let unique_count = initial.pot.potential_count();
    ensure!(
        initial.last_indices.len() == unique_count
            && initial.state.norman_charges.len() == unique_count
            && initial.state.norman_charge_reference.len() == unique_count,
        "POT initial SCF state length mismatch for {unique_count} potential(s)"
    );
    ensure!(
        initial.state.fermi_energy == initial.pot.scalars.fermi_level,
        "POT initial SCF Fermi state is inconsistent"
    );
    if !initial.restart_pot_imported && !initial.external_pot_imported {
        ensure!(
            initial.istprm.fermi.chemical_potential == initial.state.fermi_energy,
            "POT initial SCF Fermi state is inconsistent with istprm"
        );
    }
    ensure!(
        initial
            .state
            .norman_charges
            .iter()
            .all(|charge| *charge == 0.0)
            && initial
                .state
                .norman_charge_reference
                .iter()
                .all(|charge| *charge == 0.0),
        "POT initial SCF Norman-charge history must start from zero"
    );
    ensure!(
        initial.energy_grid.active_len > 0
            && initial.energy_grid.active_len <= initial.energy_grid.energies.len()
            && initial.energy_grid.steps.len() == POT_SCMT_FLOOR_COUNT,
        "POT initial SCF energy grid is inconsistent"
    );
    if let Some(fovrg_grid) = &initial.fovrg_grid {
        let fovrg_shape = fovrg_grid.regular_large.dim();
        let source_energy_count = fovrg_grid.energies_hartree.len();
        ensure!(
            initial.fovrg_grid_unavailable.is_none()
                && source_energy_count > 0
                && fovrg_grid.wave_numbers.dim() == (source_energy_count, unique_count)
                && fovrg_grid.reference_energies_hartree.dim()
                    == (source_energy_count, unique_count)
                && fovrg_grid.phase_shifts.dim()
                    == (source_energy_count, fovrg_shape.2, unique_count)
                && fovrg_grid.phase_amplitudes.dim()
                    == (source_energy_count, fovrg_shape.2, unique_count)
                && fovrg_grid.regular_small.dim() == fovrg_shape
                && fovrg_grid.irregular_large.dim() == fovrg_shape
                && fovrg_grid.irregular_small.dim() == fovrg_shape
                && fovrg_shape.0 == source_energy_count
                && fovrg_shape.1 == unique_count
                && fovrg_shape.2 > 0
                && fovrg_shape.3 == fovrg_grid.source_radii.len()
                && fovrg_grid.radial_active_counts.len() == unique_count
                && fovrg_grid.rholie_active_counts.len() == unique_count
                && fovrg_grid.muffin_tin_indices_1based.len() == unique_count
                && fovrg_grid.norman_indices_1based.len() == unique_count
                && fovrg_grid.radial_handoffs.len() == unique_count,
            "POT initial SCF FOVRG source grid is inconsistent"
        );
        ensure!(
            fovrg_grid
                .radial_active_counts
                .iter()
                .all(|count| *count > 0 && *count <= fovrg_grid.source_radii.len()),
            "POT initial SCF FOVRG radial active counts are outside the source grid"
        );
        ensure!(
            fovrg_grid
                .rholie_active_counts
                .iter()
                .all(|count| *count > 0 && *count <= fovrg_grid.source_radii.len()),
            "POT initial SCF rholie active counts are outside the source grid"
        );
        ensure!(
            fovrg_grid
                .radial_active_counts
                .iter()
                .zip(fovrg_grid.rholie_active_counts.iter())
                .all(|(radial_count, rholie_count)| radial_count >= rholie_count),
            "POT initial SCF FOVRG active counts are shorter than rholie counts"
        );
        ensure!(
            fovrg_grid.rholie_active_counts == initial.last_indices,
            "POT initial SCF rholie active counts do not match last indices"
        );
    } else {
        ensure!(
            initial
                .fovrg_grid_unavailable
                .as_ref()
                .is_some_and(|reason| !reason.is_empty()),
            "POT initial SCF FOVRG source grid is absent without a reason"
        );
    }
    if let Some(fms_grid) = &initial.fms_grid {
        let source_energy_count = fms_grid.energies_hartree.len();
        let angular_count = initial
            .fovrg_grid
            .as_ref()
            .map(|grid| grid.phase_shifts.dim().1)
            .unwrap_or(fms_grid.scattering_trace.dim().1);
        ensure!(
            initial.fms_grid_unavailable.is_none()
                && initial.fovrg_grid.is_some()
                && source_energy_count > 0
                && fms_grid.scattering_trace.dim()
                    == (source_energy_count, angular_count, unique_count),
            "POT initial SCF FMS source grid is inconsistent"
        );
    } else {
        ensure!(
            initial
                .fms_grid_unavailable
                .as_ref()
                .is_some_and(|reason| !reason.is_empty()),
            "POT initial SCF FMS source grid is absent without a reason"
        );
    }
    if let Some(contour_rows) = &initial.contour_rows {
        let source_energy_count = contour_rows.source_energies.len();
        let angular_count = initial
            .fovrg_grid
            .as_ref()
            .map(|grid| grid.phase_shifts.dim().1)
            .unwrap_or(contour_rows.scattering_trace.dim().1);
        let radial_count = contour_rows.embedded_density_source.dim().1;
        ensure!(
            initial.contour_rows_unavailable.is_none()
                && initial.fovrg_grid.is_some()
                && initial.fms_grid.is_some()
                && source_energy_count > 0
                && contour_rows.scattering_trace.dim()
                    == (source_energy_count, angular_count, unique_count)
                && contour_rows.scattering_ldos.dim()
                    == (source_energy_count, angular_count, unique_count)
                && contour_rows.embedded_ldos_source.dim()
                    == (source_energy_count, angular_count, unique_count)
                && contour_rows.scattering_density.dim()
                    == (
                        source_energy_count,
                        radial_count,
                        angular_count,
                        unique_count
                    )
                && contour_rows.embedded_density_source.dim()
                    == (source_energy_count, radial_count, unique_count)
                && contour_rows.density_scale.dim()
                    == (source_energy_count, angular_count, unique_count),
            "POT initial SCF contour source rows are inconsistent"
        );
    } else {
        ensure!(
            initial
                .contour_rows_unavailable
                .as_ref()
                .is_some_and(|reason| !reason.is_empty()),
            "POT initial SCF contour source rows are absent without a reason"
        );
    }
    if let Some(advance) = &initial.state_advance {
        ensure!(
            initial.state_advance_unavailable.is_none()
                && initial.contour_rows.as_ref().is_some_and(|rows| {
                    advance.iteration.contour.energy_points_used <= rows.source_energies.len()
                })
                && advance.iteration.contour.embedded_ldos.ncols() == unique_count
                && advance.iteration.contour.valence_density.dim()
                    == initial.state.overlapped_valence_density.dim()
                && advance.iteration.overlapped_density.dim()
                    == initial.state.overlapped_density.dim()
                && advance.iteration.overlapped_valence_density.dim()
                    == initial.state.overlapped_valence_density.dim()
                && advance.outer.norman_charge_reference.len() == unique_count
                && advance.outer.reported_charge_transfer.len() == unique_count
                && advance.outer.overlapped_density.dim() == initial.state.overlapped_density.dim()
                && advance.outer.overlapped_valence_density.dim()
                    == initial.state.overlapped_valence_density.dim()
                && advance.outer.coulomb_potential.dim() == initial.state.coulomb_potential.dim()
                && advance.state.norman_charges.len() == unique_count
                && advance.state.norman_charge_reference.len() == unique_count
                && advance.state.overlapped_density.dim() == initial.state.overlapped_density.dim()
                && advance.state.overlapped_valence_density.dim()
                    == initial.state.overlapped_valence_density.dim()
                && advance.state.coulomb_potential.dim() == initial.state.coulomb_potential.dim(),
            "POT initial SCF state advance is inconsistent"
        );
    } else {
        ensure!(
            initial
                .state_advance_unavailable
                .as_ref()
                .is_some_and(|reason| !reason.is_empty()),
            "POT initial SCF state advance is absent without a reason"
        );
    }
    if let Some(next) = &initial.next_iteration {
        ensure!(
            initial.next_iteration_unavailable.is_none()
                && initial
                    .state_advance
                    .as_ref()
                    .is_some_and(|advance| advance.outer.status
                        == PotScfOuterIterationStatus::NeedsNextIteration)
                && next.iteration == 2
                && next.pot.potential_count() == unique_count
                && next.state.norman_charges.len() == unique_count
                && next.state.norman_charge_reference.len() == unique_count
                && next.state.occupancy_by_l.dim() == initial.pot.valence_occupancy.dim()
                && next.state.overlapped_density.dim() == initial.state.overlapped_density.dim()
                && next.state.overlapped_valence_density.dim()
                    == initial.state.overlapped_valence_density.dim()
                && next.state.coulomb_potential.dim() == initial.state.coulomb_potential.dim()
                && next.pot.scalars.fermi_level == next.state.fermi_energy
                && next.pot.electron_density == next.state.overlapped_density
                && next.pot.valence_density == next.state.overlapped_valence_density
                && next.pot.coulomb_potential == next.state.coulomb_potential
                && next.pot.total_potential == next.istprm.total_potential
                && next.pot.valence_potential == next.istprm.valence_potential
                && next.energy_grid.active_len > 0
                && next.energy_grid.active_len <= next.energy_grid.energies.len()
                && next.energy_grid.steps.len() == POT_SCMT_FLOOR_COUNT,
            "POT next SCF iteration preparation is inconsistent"
        );
        if let Some(fovrg_grid) = &next.fovrg_grid {
            let fovrg_shape = fovrg_grid.regular_large.dim();
            let source_energy_count = fovrg_grid.energies_hartree.len();
            ensure!(
                next.fovrg_grid_unavailable.is_none()
                    && source_energy_count > 0
                    && fovrg_grid.wave_numbers.dim() == (source_energy_count, unique_count)
                    && fovrg_grid.reference_energies_hartree.dim()
                        == (source_energy_count, unique_count)
                    && fovrg_grid.phase_shifts.dim()
                        == (source_energy_count, fovrg_shape.2, unique_count)
                    && fovrg_grid.phase_amplitudes.dim()
                        == (source_energy_count, fovrg_shape.2, unique_count)
                    && fovrg_grid.regular_small.dim() == fovrg_shape
                    && fovrg_grid.irregular_large.dim() == fovrg_shape
                    && fovrg_grid.irregular_small.dim() == fovrg_shape
                    && fovrg_shape.0 == source_energy_count
                    && fovrg_shape.1 == unique_count
                    && fovrg_shape.2 > 0
                    && fovrg_shape.3 == fovrg_grid.source_radii.len()
                    && fovrg_grid.radial_active_counts.len() == unique_count
                    && fovrg_grid.rholie_active_counts.len() == unique_count
                    && fovrg_grid.muffin_tin_indices_1based.len() == unique_count
                    && fovrg_grid.norman_indices_1based.len() == unique_count
                    && fovrg_grid.radial_handoffs.len() == unique_count,
                "POT next SCF FOVRG source grid is inconsistent"
            );
            ensure!(
                fovrg_grid
                    .radial_active_counts
                    .iter()
                    .all(|count| *count > 0 && *count <= fovrg_grid.source_radii.len()),
                "POT next SCF FOVRG radial active counts are outside the source grid"
            );
            ensure!(
                fovrg_grid
                    .rholie_active_counts
                    .iter()
                    .all(|count| *count > 0 && *count <= fovrg_grid.source_radii.len()),
                "POT next SCF rholie active counts are outside the source grid"
            );
            ensure!(
                fovrg_grid
                    .radial_active_counts
                    .iter()
                    .zip(fovrg_grid.rholie_active_counts.iter())
                    .all(|(radial_count, rholie_count)| radial_count >= rholie_count),
                "POT next SCF FOVRG active counts are shorter than rholie counts"
            );
            ensure!(
                fovrg_grid.rholie_active_counts == next.last_indices,
                "POT next SCF rholie active counts do not match last indices"
            );
        } else {
            ensure!(
                next.fovrg_grid_unavailable
                    .as_ref()
                    .is_some_and(|reason| !reason.is_empty()),
                "POT next SCF FOVRG source grid is absent without a reason"
            );
        }
        if let Some(fms_grid) = &next.fms_grid {
            let source_energy_count = fms_grid.energies_hartree.len();
            let angular_count = next
                .fovrg_grid
                .as_ref()
                .map(|grid| grid.phase_shifts.dim().1)
                .unwrap_or(fms_grid.scattering_trace.dim().1);
            ensure!(
                next.fms_grid_unavailable.is_none()
                    && next.fovrg_grid.is_some()
                    && source_energy_count > 0
                    && fms_grid.scattering_trace.dim()
                        == (source_energy_count, angular_count, unique_count),
                "POT next SCF FMS source grid is inconsistent"
            );
        } else {
            ensure!(
                next.fms_grid_unavailable
                    .as_ref()
                    .is_some_and(|reason| !reason.is_empty()),
                "POT next SCF FMS source grid is absent without a reason"
            );
        }
        if let Some(contour_rows) = &next.contour_rows {
            let source_energy_count = contour_rows.source_energies.len();
            let angular_count = next
                .fovrg_grid
                .as_ref()
                .map(|grid| grid.phase_shifts.dim().1)
                .unwrap_or(contour_rows.scattering_trace.dim().1);
            let radial_count = contour_rows.embedded_density_source.dim().1;
            ensure!(
                next.contour_rows_unavailable.is_none()
                    && next.fovrg_grid.is_some()
                    && next.fms_grid.is_some()
                    && source_energy_count > 0
                    && contour_rows.scattering_trace.dim()
                        == (source_energy_count, angular_count, unique_count)
                    && contour_rows.scattering_ldos.dim()
                        == (source_energy_count, angular_count, unique_count)
                    && contour_rows.embedded_ldos_source.dim()
                        == (source_energy_count, angular_count, unique_count)
                    && contour_rows.scattering_density.dim()
                        == (
                            source_energy_count,
                            radial_count,
                            angular_count,
                            unique_count
                        )
                    && contour_rows.embedded_density_source.dim()
                        == (source_energy_count, radial_count, unique_count)
                    && contour_rows.density_scale.dim()
                        == (source_energy_count, angular_count, unique_count),
                "POT next SCF contour source rows are inconsistent"
            );
        } else {
            ensure!(
                next.contour_rows_unavailable
                    .as_ref()
                    .is_some_and(|reason| !reason.is_empty()),
                "POT next SCF contour source rows are absent without a reason"
            );
        }
        if let Some(advance) = &next.state_advance {
            ensure!(
                next.state_advance_unavailable.is_none()
                    && next.contour_rows.as_ref().is_some_and(|rows| {
                        advance.iteration.contour.energy_points_used <= rows.source_energies.len()
                    })
                    && advance.iteration.contour.embedded_ldos.ncols() == unique_count
                    && advance.iteration.contour.valence_density.dim()
                        == next.state.overlapped_valence_density.dim()
                    && advance.iteration.overlapped_density.dim()
                        == next.state.overlapped_density.dim()
                    && advance.iteration.overlapped_valence_density.dim()
                        == next.state.overlapped_valence_density.dim()
                    && advance.outer.norman_charge_reference.len() == unique_count
                    && advance.outer.reported_charge_transfer.len() == unique_count
                    && advance.outer.overlapped_density.dim()
                        == next.state.overlapped_density.dim()
                    && advance.outer.overlapped_valence_density.dim()
                        == next.state.overlapped_valence_density.dim()
                    && advance.outer.coulomb_potential.dim() == next.state.coulomb_potential.dim()
                    && advance.state.norman_charges.len() == unique_count
                    && advance.state.norman_charge_reference.len() == unique_count
                    && advance.state.overlapped_density.dim()
                        == next.state.overlapped_density.dim()
                    && advance.state.overlapped_valence_density.dim()
                        == next.state.overlapped_valence_density.dim()
                    && advance.state.coulomb_potential.dim() == next.state.coulomb_potential.dim(),
                "POT next SCF state advance is inconsistent"
            );
        } else {
            ensure!(
                next.state_advance_unavailable
                    .as_ref()
                    .is_some_and(|reason| !reason.is_empty()),
                "POT next SCF state advance is absent without a reason"
            );
        }
    } else {
        ensure!(
            initial
                .next_iteration_unavailable
                .as_ref()
                .is_some_and(|reason| !reason.is_empty()),
            "POT next SCF iteration preparation is absent without a reason"
        );
    }
    ensure!(
        initial.state.overlapped_density.dim() == initial.pot.electron_density.dim()
            && initial.state.overlapped_valence_density.dim() == initial.pot.valence_density.dim()
            && initial.state.coulomb_potential.dim() == initial.pot.coulomb_potential.dim(),
        "POT initial SCF array shapes are inconsistent"
    );
    ensure!(
        initial
            .last_indices
            .iter()
            .all(|index| *index > 0 && *index <= POT_BIN_RADIAL_POINTS),
        "POT initial SCF last radial indices must be inside the pot.bin radial grid"
    );
    Ok(())
}

fn prepare_config_cache(caches: &AtomicCachePaths, input: &PotInput) -> Result<bool> {
    let source_handoff =
        config_handoff_needs_generation(&caches.config_dat, input, &caches.config_inp)?;
    if source_handoff {
        generated_config_dat(input, &caches.config_inp)?;
    }
    Ok(source_handoff)
}

fn prepare_fpf0_cache(
    caches: &AtomicCachePaths,
    apot: &ApotBinData,
    input: &PotInput,
) -> Result<()> {
    if fpf0_needs_generation(&caches.fpf0_dat, apot, input, &caches.config_inp)? {
        generated_fpf0_dat(apot, input, &caches.config_inp)?;
        return Ok(());
    }
    if caches.fpf0_dat.is_file() {
        read_fpf0_dat(&caches.fpf0_dat)
            .with_context(|| format!("failed to read {}", caches.fpf0_dat.display()))?;
    }
    Ok(())
}

fn prepare_module_log_cache(caches: &AtomicCachePaths) -> Result<()> {
    if caches.log1_dat.is_file() {
        read_module_log_dat(&caches.log1_dat)
            .with_context(|| format!("failed to read {}", caches.log1_dat.display()))?;
    }
    Ok(())
}

/// Run FEFF `ATOM` from cached output or supported typed source handoffs.
///
/// Existing FEFF `apot.bin` caches remain supported. Ordinary source runs can
/// generate the full `apot.bin` stream from `pot.inp` and `geom.dat`, then
/// validate or regenerate the neighboring `config.dat`, `fpf0.dat`, and
/// deterministic `log1.dat` handoffs from typed metadata.
pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    run_in_dir_with_prepared_no_scf(work_dir, None)
}

pub(crate) fn run_in_dir_with_prepared_no_scf(
    work_dir: &Path,
    prepared_no_scf: Option<&PreparedNoScfPotOutputs>,
) -> Result<usize> {
    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) {
        return Ok(0);
    }

    let caches = AtomicCachePaths::new(work_dir);
    let config_source_handoff =
        config_handoff_needs_generation(&caches.config_dat, &input, &caches.config_inp)?;
    let mut written = write_or_generate_config(&caches.config_dat, &caches.config_inp, &input)?;
    let mut apot_source_handoff = false;
    let mut generated_states = None;

    let apot = if !caches.apot_bin.is_file() {
        if atomic_apot_source_files_present(&caches) {
            apot_source_handoff = true;
            if let Some(prepared) = prepared_no_scf {
                prepared.apot.clone()
            } else {
                let (apot, states) =
                    generated_atomic_apot_and_states_from_sources(&caches, &input)?;
                generated_states = Some(states);
                apot
            }
        } else {
            bail!("ATOM source apot.bin generation requires geom.dat handoff");
        }
    } else {
        match read_apot_bin(&caches.apot_bin)
            .with_context(|| format!("failed to read {}", caches.apot_bin.display()))
        {
            Ok(apot) => apot,
            Err(error) => {
                if atomic_apot_source_files_present(&caches) {
                    apot_source_handoff = true;
                    if let Some(prepared) = prepared_no_scf {
                        prepared.apot.clone()
                    } else {
                        let (apot, states) =
                            generated_atomic_apot_and_states_from_sources(&caches, &input)?;
                        generated_states = Some(states);
                        apot
                    }
                } else if config_handoff_source_is_compatible(&caches, &input)? {
                    bail!("ATOM source apot.bin generation requires geom.dat handoff");
                } else {
                    return Err(error);
                }
            }
        }
    };
    let mut apot = apot;
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;
    write_apot_bin(&caches.apot_bin, &apot)
        .with_context(|| format!("failed to write {}", caches.apot_bin.display()))?;

    written += 1_usize;
    let fpf0_source_handoff = can_generate_fpf0_from_sources(&caches, &apot, &input)?;
    written += write_or_generate_fpf0(&caches.fpf0_dat, &apot, &input, &caches.config_inp)?;
    let log_source_handoff = if apot_source_handoff {
        true
    } else {
        can_recover_atomic_module_log(&caches, config_source_handoff || fpf0_source_handoff)
    };
    written += write_or_recover_module_log(&caches.log1_dat, &input, log_source_handoff)?;
    if input.control.ipr1 >= 3 {
        if generated_states.is_none() && atomic_apot_source_files_present(&caches) {
            generated_states = Some(generated_atomic_scf_states(&input, &caches.config_inp)?);
        }
        if let Some(states) = generated_states.as_deref() {
            written += write_atomic_diagnostic_outputs(work_dir, &input, states)?;
        }
    }
    Ok(written)
}

fn write_atomic_diagnostic_outputs(
    work_dir: &Path,
    input: &PotInput,
    states: &[AtomicScfState],
) -> Result<usize> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        states.len() >= unique_count,
        "ATOM diagnostic output has {} state(s), expected at least {unique_count}",
        states.len()
    );

    for (potential, state) in states.iter().take(unique_count).enumerate() {
        let total_energy = atomic_total_energy_from_state(input, potential, state)
            .with_context(|| format!("failed to compute atom{potential:02}.dat total energy"))?;
        let tabulation = (input.control.ipr1 >= 5)
            .then(|| atomic_tabulation_from_state(state))
            .transpose()
            .with_context(|| format!("failed to compute atom{potential:02}.dat tabulation"))?;
        let first_radius = *state
            .initial_orbitals
            .radii
            .first()
            .context("ATOM diagnostic radial grid is empty")?;
        let data = AtomDatData {
            potential_index: potential,
            print_level: input.control.ipr1,
            max_orbital_iterations: ATOM_SCF_MAX_ORBITAL_ITERATIONS,
            energy_precision: state.orbital_initialization.energy_precision,
            wavefunction_precision: state.orbital_initialization.wavefunction_precision,
            radial_count: state.orbital_initialization.radial_count,
            first_radius,
            radial_step: ATOM_RADIAL_STEP,
            matching_precision: state.orbital_initialization.primary_matching_precision,
            matching_attempts: state.orbital_initialization.attempt_count,
            finite_nucleus: state.initial_orbitals.nucleus_index > 1,
            total_energy,
            tabulation,
        };
        let path = work_dir.join(format!("atom{potential:02}.dat"));
        write_atom_dat(&path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(unique_count)
}

fn atomic_tabulation_from_state(state: &AtomicScfState) -> Result<refeff_core::AtomicTabulation> {
    let principal_quantum_numbers = state
        .principal_quantum_numbers
        .as_slice()
        .context("ATOM principal quantum-number storage is not contiguous")?;
    let kappas = state
        .kappas
        .as_slice()
        .context("ATOM kappa storage is not contiguous")?;
    let occupations = state
        .occupations
        .as_slice()
        .context("ATOM occupation storage is not contiguous")?;
    let orbital_energies = state
        .scf
        .orbital_energies
        .as_slice()
        .context("ATOM orbital-energy storage is not contiguous")?;
    let active_lengths = state
        .scf
        .active_lengths
        .as_slice()
        .context("ATOM active-length storage is not contiguous")?;
    let orbital_powers = state
        .initial_orbitals
        .orbital_powers
        .as_slice()
        .context("ATOM origin-power storage is not contiguous")?;
    let radial_count = state.initial_orbitals.radii.len();
    let coefficient_count = state.scf.large_coefficients.nrows();
    let derivative_large = Array1::zeros(radial_count);
    let derivative_small = Array1::zeros(radial_count);
    let derivative_large_coefficients = Array1::zeros(coefficient_count);
    let derivative_small_coefficients = Array1::zeros(coefficient_count);

    atomic_tabulation(
        AtomicTabulationInput {
            principal_quantum_numbers,
            kappas,
            occupations,
            orbital_energies,
        },
        |request| {
            atomic_differential_integral(AtomicDifferentialIntegralInput {
                kind: AtomicDifferentialIntegralKind::ComponentOverlap {
                    left_orbital_1based: request.left + 1,
                    right_orbital_1based: request.right + 1,
                    multiply_by_derivative: false,
                },
                power: request.power,
                origin_power: 0.0,
                step: ATOM_RADIAL_STEP,
                radii: state.initial_orbitals.radii.view(),
                active_lengths,
                orbital_powers,
                large_components: state.scf.large_components.view(),
                small_components: state.scf.small_components.view(),
                large_coefficients: state.scf.large_coefficients.view(),
                small_coefficients: state.scf.small_coefficients.view(),
                derivative_large: derivative_large.view(),
                derivative_small: derivative_small.view(),
                derivative_large_coefficients: derivative_large_coefficients.view(),
                derivative_small_coefficients: derivative_small_coefficients.view(),
            })
        },
    )
    .context("failed to assemble FEFF ATOM tabrat data")
}

fn atomic_enabled(input: &PotInput) -> bool {
    input.control.mpot == 1
}

fn read_input(work_dir: &Path) -> Result<PotInput> {
    let input_path = work_dir.join("pot.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    PotInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn read_geom_dat(path: &Path) -> Result<GeomDat> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    GeomDat::parse_str(path, &text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_optional_config(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data =
        read_config_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_config_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn config_handoff_needs_generation(
    path: &Path,
    input: &PotInput,
    config_inp: &Path,
) -> Result<bool> {
    if !path.is_file() {
        generated_config_dat(input, config_inp)?;
        return Ok(true);
    }
    match read_config_dat(path) {
        Ok(data) if config_dat_matches_input(&data, input).is_ok() => Ok(false),
        Ok(_) => {
            generated_config_dat(input, config_inp)?;
            Ok(true)
        }
        Err(_) => {
            generated_config_dat(input, config_inp)?;
            Ok(true)
        }
    }
}

fn config_handoff_matches_input(path: &Path, input: &PotInput) -> Result<bool> {
    let Ok(data) = read_config_dat(path) else {
        return Ok(false);
    };
    Ok(config_dat_matches_input(&data, input).is_ok())
}

fn config_dat_matches_input(data: &ConfigDatData, input: &PotInput) -> Result<()> {
    ensure!(
        data.potentials.len() == input.potentials.len(),
        "ATOM config.dat has {} potential row(s), expected {} from pot.inp",
        data.potentials.len(),
        input.potentials.len()
    );
    for (index, (actual, expected)) in data
        .potentials
        .iter()
        .zip(input.potentials.iter())
        .enumerate()
    {
        ensure!(
            actual.potential_index == index as i32,
            "ATOM config.dat row {} has potential index {}, expected {}",
            index,
            actual.potential_index,
            index
        );
        ensure!(
            actual.atomic_number == expected.z,
            "ATOM config.dat row {} has atomic number {}, expected {}",
            index,
            actual.atomic_number,
            expected.z
        );
        let expected_element = atomic_symbol(checked_atomic_number(expected.z)?)?;
        ensure!(
            actual.element == expected_element,
            "ATOM config.dat row {} has element {}, expected {}",
            index,
            actual.element,
            expected_element
        );
        ensure!(
            actual
                .occupations
                .iter()
                .any(|occupation| occupation.is_finite() && *occupation > 0.0),
            "ATOM config.dat row {index} has no occupied orbitals"
        );
    }
    Ok(())
}

fn write_or_generate_config(path: &Path, config_inp: &Path, input: &PotInput) -> Result<usize> {
    if !config_handoff_needs_generation(path, input, config_inp)? {
        return write_optional_config(path);
    }
    let data = generated_config_dat(input, config_inp)?;
    write_config_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_optional_fpf0(path: &Path) -> Result<usize> {
    if !path.is_file() {
        return Ok(0);
    }
    let data = read_fpf0_dat(path).with_context(|| format!("failed to read {}", path.display()))?;
    write_fpf0_dat(path, &data).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn write_or_generate_fpf0(
    path: &Path,
    apot: &ApotBinData,
    input: &PotInput,
    config_inp: &Path,
) -> Result<usize> {
    if fpf0_needs_generation(path, apot, input, config_inp)? {
        let data = generated_fpf0_dat(apot, input, config_inp)?;
        write_fpf0_dat(path, &data)
            .with_context(|| format!("failed to write {}", path.display()))?;
        return Ok(1);
    }
    if path.is_file() {
        return write_optional_fpf0(path);
    }
    Ok(0)
}

fn can_generate_fpf0_from_sources(
    caches: &AtomicCachePaths,
    apot: &ApotBinData,
    input: &PotInput,
) -> Result<bool> {
    fpf0_needs_generation(&caches.fpf0_dat, apot, input, &caches.config_inp)
}

fn can_recover_atomic_module_log(caches: &AtomicCachePaths, source_handoff_written: bool) -> bool {
    source_handoff_written && !cached_pot_stage_can_share_module_log(caches)
}

fn cached_pot_stage_can_share_module_log(caches: &AtomicCachePaths) -> bool {
    if !caches.pot_bin.is_file() || !caches.apot_bin.is_file() {
        return false;
    }
    let Ok(pot) = read_pot_bin(&caches.pot_bin) else {
        return false;
    };
    let Ok(apot) = read_apot_bin(&caches.apot_bin) else {
        return false;
    };
    potential_dat_outputs_from_bins(&pot, &apot).is_ok()
}

fn has_fpf0_source_sections(apot: &ApotBinData, input: &PotInput) -> Result<bool> {
    let state_count = apot_state_count(input)?;
    let component_column = 0_usize;
    let required_sections = [
        APOT_CORE_HOLE_SECTION_NUMBER,
        ATOM_FPF0_NORB_SECTION_NUMBER,
        ATOM_FPF0_DENSITY_SECTION_NUMBER,
        ATOM_FPF0_EORB_SECTION_NUMBER,
        ATOM_FPF0_KAPPA_SECTION_NUMBER,
        ATOM_FPF0_DGC_SECTION_START + component_column,
        ATOM_FPF0_DGC_SECTION_START + state_count + component_column,
    ];
    Ok(required_sections
        .iter()
        .all(|section_number| apot_has_section(apot, *section_number)))
}

fn has_fpf0_total_energy_source_sections(apot: &ApotBinData, input: &PotInput) -> Result<bool> {
    let state_count = apot_state_count(input)?;
    let column = fpf0_total_energy_column(input, state_count)?;
    let required_sections = [
        ATOM_FPF0_NORB_SECTION_NUMBER,
        ATOM_FPF0_EORB_SECTION_NUMBER,
        ATOM_FPF0_KAPPA_SECTION_NUMBER,
        ATOM_FPF0_DGC_SECTION_START + column,
        ATOM_FPF0_DGC_SECTION_START + state_count + column,
        ATOM_FPF0_DGC_SECTION_START + 2 * state_count + column,
        ATOM_FPF0_DGC_SECTION_START + 3 * state_count + column,
    ];
    Ok(required_sections
        .iter()
        .all(|section_number| apot_has_section(apot, *section_number)))
}

fn fpf0_source_is_available(apot: &ApotBinData, input: &PotInput) -> Result<bool> {
    Ok(input.control.ihole > 0
        && has_fpf0_source_sections(apot, input)?
        && has_fpf0_total_energy_source_sections(apot, input)?)
}

fn fpf0_needs_generation(
    path: &Path,
    apot: &ApotBinData,
    input: &PotInput,
    config_inp: &Path,
) -> Result<bool> {
    if !fpf0_source_is_available(apot, input)? {
        return Ok(false);
    }
    if !path.is_file() {
        return Ok(true);
    }

    let existing = match read_fpf0_dat(path) {
        Ok(existing) => existing,
        Err(_) => return Ok(true),
    };
    let Ok(generated) = generated_fpf0_dat(apot, input, config_inp) else {
        return Ok(false);
    };
    Ok(!fpf0_matches_source_structure(&existing, &generated))
}

fn fpf0_matches_source_structure(actual: &Fpf0DatData, expected: &Fpf0DatData) -> bool {
    actual.atomic_number == expected.atomic_number
        && actual.oscillator_count() == expected.oscillator_count()
        && actual
            .oscillators
            .iter()
            .zip(expected.oscillators.iter())
            .all(|(actual, expected)| actual.orbital_index == expected.orbital_index)
        && actual.form_factor_count() == expected.form_factor_count()
}

/// Generate the source-backed ATOM SCF subset of `apot.bin`.
///
/// This covers FEFF sections derived directly from converged single-atom
/// `scfdat` states. Use [`generated_atomic_apot_bin`] when the static
/// geometry/POT handoffs are available and the full `WriteAtomicPots` stream is
/// required.
#[allow(dead_code)]
pub(crate) fn generated_atomic_scf_apot_bin(
    input: &PotInput,
    config_inp: &Path,
) -> Result<ApotBinData> {
    Ok(ApotBinData {
        sections: generated_atomic_scf_apot_sections(input, config_inp)?,
    })
}

#[allow(dead_code)]
pub(crate) fn generated_atomic_scf_apot_sections(
    input: &PotInput,
    config_inp: &Path,
) -> Result<Vec<ApotBinSection>> {
    let state_count = apot_state_count(input)?;
    let states = generated_atomic_scf_states(input, config_inp)?;
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );

    let states = states
        .iter()
        .map(atomic_output_state)
        .collect::<Result<Vec<_>>>()?;
    let refs = states
        .iter()
        .enumerate()
        .map(|(state_index, state)| ApotAtomicScfStateRef { state_index, state })
        .collect::<Vec<_>>();
    apot_atomic_scf_sections_from_states(state_count, &refs)
        .context("failed to assemble ATOM SCF apot.bin sections")
}

/// Generate the source-backed full FEFF `WriteAtomicPots` `apot.bin` stream
/// from typed ATOM/POT handoffs.
#[allow(dead_code)]
pub(crate) fn generated_atomic_apot_bin(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
    pot: &PotBinData,
) -> Result<ApotBinData> {
    Ok(ApotBinData {
        sections: generated_atomic_apot_sections(input, config_inp, geom, pot)?,
    })
}

#[allow(dead_code)]
pub(crate) fn generated_atomic_apot_sections(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
    pot: &PotBinData,
) -> Result<Vec<ApotBinSection>> {
    let static_arrays = atomic_apot_static_arrays_from_handoffs(input, geom, pot)?;
    generated_atomic_apot_sections_from_static_arrays(input, config_inp, &static_arrays)
}

fn generated_atomic_apot_bin_from_states(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
    pot: &PotBinData,
    states: &[AtomicScfState],
) -> Result<ApotBinData> {
    let static_arrays = atomic_apot_static_arrays_from_handoffs(input, geom, pot)?;
    Ok(ApotBinData {
        sections: generated_atomic_apot_sections_from_static_arrays_and_states(
            input,
            config_inp,
            &static_arrays,
            states,
        )?,
    })
}

#[allow(dead_code)]
fn generated_atomic_apot_sections_from_static_arrays(
    input: &PotInput,
    config_inp: &Path,
    static_arrays: &AtomicApotStaticArrays,
) -> Result<Vec<ApotBinSection>> {
    let unique_count = apot_unique_potential_count(input)?;
    let state_count = apot_state_count(input)?;
    ensure!(
        static_arrays.unique_potential_count == unique_count,
        "ATOM static APOT arrays have {} unique potential(s), expected {unique_count}",
        static_arrays.unique_potential_count
    );

    let states = generated_atomic_scf_states(input, config_inp)?;
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    generated_atomic_apot_sections_from_static_arrays_and_states(
        input,
        config_inp,
        static_arrays,
        &states,
    )
}

fn generated_atomic_apot_sections_from_static_arrays_and_states(
    input: &PotInput,
    config_inp: &Path,
    static_arrays: &AtomicApotStaticArrays,
    states: &[AtomicScfState],
) -> Result<Vec<ApotBinSection>> {
    let unique_count = apot_unique_potential_count(input)?;
    let state_count = apot_state_count(input)?;
    ensure!(
        static_arrays.unique_potential_count == unique_count,
        "ATOM static APOT arrays have {} unique potential(s), expected {unique_count}",
        static_arrays.unique_potential_count
    );
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    let output_states = states
        .iter()
        .map(atomic_output_state)
        .collect::<Result<Vec<_>>>()?;
    let state_inputs = output_states
        .iter()
        .enumerate()
        .map(|(state_index, state)| {
            ApotAtomicScfStateSectionsInput::from_atomic_scf_state(state_count, state_index, state)
        })
        .collect::<Vec<_>>();
    let core_hole = atomic_core_hole_columns_from_states(input, config_inp, states)?;
    let overlap_arrays = atomic_apot_overlap_arrays_from_states(input, static_arrays, states)?;
    let energy_scalars =
        atomic_apot_energy_scalars_from_states(input, config_inp, states, &overlap_arrays)?;
    let amplitude_reduction =
        atomic_apot_amplitude_reduction_from_states(input, config_inp, states)?;
    let norman_valence_counts = atomic_norman_valence_counts_by_l(input, config_inp)?;
    let orbital_indices_by_kappa = atomic_orbital_indices_by_kappa(input, config_inp)?;
    let nph =
        usize::try_from(input.control.nph).context("ATOM nph cannot be represented as usize")?;

    apot_atomic_pots_sections(ApotAtomicPotsSectionsInput {
        unique_potential_count: nph,
        atom_count: static_arrays.atom_count,
        hole_index: i64::from(input.control.ihole),
        relaxation_energy: energy_scalars.relaxation_energy,
        edge_energy: energy_scalars.edge_energy,
        amplitude_reduction,
        atomic_numbers: static_arrays.atomic_numbers.view(),
        model_atom_indices: static_arrays.model_atom_indices.view(),
        overlap_shell_counts: static_arrays.overlap_shell_counts.view(),
        norman_radii: overlap_arrays.norman_radii.view(),
        atom_potential_indices: static_arrays.atom_potential_indices.view(),
        core_hole_large_component: core_hole.large_component.view(),
        core_hole_small_component: core_hole.small_component.view(),
        core_hole_density: core_hole.density.view(),
        core_hole_coulomb_potential: core_hole.coulomb_potential.view(),
        overlap_potential_indices: static_arrays.overlap_potential_indices.view(),
        overlap_shell_atom_counts: static_arrays.overlap_shell_atom_counts.view(),
        magnetization_density: overlap_arrays.magnetization_density.view(),
        norman_valence_counts: norman_valence_counts.view(),
        atom_positions: static_arrays.atom_positions.view(),
        overlap_radii: static_arrays.overlap_radii.view(),
        overlapped_density: overlap_arrays.overlapped_density.view(),
        overlapped_valence_density: overlap_arrays.overlapped_valence_density.view(),
        overlapped_coulomb_potential: overlap_arrays.overlapped_coulomb_potential.view(),
        orbital_indices_by_kappa: orbital_indices_by_kappa.view(),
        states: &state_inputs,
    })
    .context("failed to assemble full ATOM apot.bin sections")
}

#[allow(dead_code)]
pub(crate) fn generated_no_scf_pot_bin(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
) -> Result<PotBinData> {
    generated_no_scf_pot_bin_with_core_valence_peaks(input, config_inp, geom, None)
}

fn generated_no_scf_pot_bin_with_core_valence_peaks(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
    core_valence_peak_energies: Option<ArrayView2<'_, f64>>,
) -> Result<PotBinData> {
    let states = generated_atomic_scf_states(input, config_inp)?;
    generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
        input,
        config_inp,
        geom,
        &states,
        core_valence_peak_energies,
    )
}

fn apot_handoff_value(value: f64) -> Result<f64> {
    if value == 0.0 {
        return Ok(value);
    }
    let mut field = String::with_capacity(24);
    refeff_io::format::write_fortran_zero_scaled_exp(&mut field, value, 20, 10)?;
    field
        .trim()
        .parse()
        .context("invalid formatted APOT numeric handoff")
}

fn round_apot_handoff_values<'a>(values: impl Iterator<Item = &'a mut f64>) -> Result<()> {
    for value in values {
        *value = apot_handoff_value(*value)?;
    }
    Ok(())
}

fn generated_no_scf_pot_bin_with_core_valence_peaks_from_states(
    input: &PotInput,
    config_inp: &Path,
    geom: &GeomDat,
    states: &[AtomicScfState],
    core_valence_peak_energies: Option<ArrayView2<'_, f64>>,
) -> Result<PotBinData> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        unique_count >= 1,
        "POT no-SCF pot.bin generation requires at least one potential"
    );

    ensure!(
        states.len() == unique_count + 1,
        "POT no-SCF generated {} SCF state(s), expected {}",
        states.len(),
        unique_count + 1
    );

    let mut static_arrays = atomic_apot_static_arrays_from_source_geometry(
        input,
        geom,
        Array1::from_elem(unique_count, 1.0),
    )?;
    let mut overlap_arrays = atomic_apot_overlap_arrays_from_states(input, &static_arrays, states)?;
    let mut core_hole = atomic_core_hole_columns_from_states(input, config_inp, states)?;
    round_apot_handoff_values(
        core_hole
            .large_component
            .iter_mut()
            .chain(core_hole.small_component.iter_mut()),
    )?;
    let mut energy_scalars =
        atomic_apot_energy_scalars_from_states(input, config_inp, states, &overlap_arrays)?;
    energy_scalars.edge_energy = apot_handoff_value(energy_scalars.edge_energy)?;
    energy_scalars.relaxation_energy = apot_handoff_value(energy_scalars.relaxation_energy)?;
    let amplitude_reduction = apot_handoff_value(atomic_apot_amplitude_reduction_from_states(
        input, config_inp, states,
    )?)?;
    // Native POT consumes ATOM through the E20.10 APOT text handoff.
    // Match that precision in memory before the single-precision projection.
    round_apot_handoff_values(
        static_arrays
            .overlap_radii
            .iter_mut()
            .chain(overlap_arrays.norman_radii.iter_mut())
            .chain(overlap_arrays.magnetization_density.iter_mut())
            .chain(overlap_arrays.overlapped_density.iter_mut())
            .chain(overlap_arrays.overlapped_valence_density.iter_mut())
            .chain(overlap_arrays.overlapped_coulomb_potential.iter_mut()),
    )?;
    let atomic_numbers = no_scf_pot_atomic_numbers(input, unique_count)?;
    let potential_multiplicities = generated_pot_potential_multiplicities(input, unique_count)?;
    let ionization = no_scf_pot_ionization(input, unique_count)?;
    let total_charge =
        no_scf_pot_total_charge(&atomic_numbers, &potential_multiplicities, &ionization)?;
    // With `nscmt=0`, FEFF initializes qnrm to zero and never enters the
    // outer SCMT loop that converts it to reported charge transfer.
    let norman_charges = Array1::zeros(unique_count);
    let istprm = no_scf_pot_istprm_state(
        input,
        &static_arrays,
        &overlap_arrays,
        potential_multiplicities.view(),
        total_charge,
    )?;
    let core_valence = no_scf_pot_core_valence_selection(
        input,
        states,
        &atomic_numbers,
        istprm.interstitial.interstitial_potential,
        core_valence_peak_energies,
    )
    .context("failed to calculate POT no-SCF core-valence separation")?;

    let mut kappa = Array1::<i32>::zeros(POT_BIN_ORBITALS);
    let mut orbital_energies = Array1::<f64>::zeros(POT_BIN_ORBITALS);
    let mut orbital_occupancy = Array2::<f64>::zeros((POT_BIN_ORBITALS, unique_count));
    let mut large_components =
        Array3::<f64>::zeros((ATOM_RADIAL_POINTS, POT_BIN_ORBITALS, unique_count));
    let mut small_components =
        Array3::<f64>::zeros((ATOM_RADIAL_POINTS, POT_BIN_ORBITALS, unique_count));
    let mut large_coefficients =
        Array3::<f64>::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, unique_count));
    let mut small_coefficients =
        Array3::<f64>::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, unique_count));
    for (potential, state) in states.iter().enumerate().take(unique_count) {
        no_scf_pot_copy_state_orbitals(
            state,
            potential,
            &mut kappa,
            &mut orbital_energies,
            &mut orbital_occupancy,
            &mut large_components,
            &mut small_components,
            &mut large_coefficients,
            &mut small_coefficients,
        )?;
    }

    round_apot_handoff_values(
        orbital_energies
            .iter_mut()
            .chain(orbital_occupancy.iter_mut())
            .chain(large_components.iter_mut())
            .chain(small_components.iter_mut())
            .chain(large_coefficients.iter_mut())
            .chain(small_coefficients.iter_mut()),
    )?;

    let occupied_orbital_indices =
        no_scf_pot_occupied_orbital_indices(input, config_inp, unique_count)?;
    let mut valence_occupancy = no_scf_pot_valence_occupancy(input, config_inp, unique_count)?;
    let mut valence_density = overlap_arrays.overlapped_valence_density.clone();
    no_scf_pot_apply_core_valence_selection(
        &core_valence,
        &mut orbital_occupancy,
        &mut valence_occupancy,
        &mut valence_density,
        large_components.view(),
        small_components.view(),
    )?;

    Ok(PotBinData {
        titles: input.titles.clone(),
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: input.run.nohole,
        ihole: input.control.ihole,
        interstitial_selector: input.run.inters,
        automatic_folp: input.control.iafolp,
        jump_mode: input.run.jumprm,
        unfreeze_f: input.run.iunf,
        scalars: PotBinScalars {
            average_norman_radius: istprm.interstitial.average_norman_radius,
            fermi_level: istprm.interstitial.fermi.chemical_potential,
            interstitial_potential: istprm.interstitial.interstitial_potential,
            interstitial_density: istprm.interstitial.interstitial_density,
            edge_position: energy_scalars.edge_energy,
            amplitude_reduction,
            relaxation_energy: energy_scalars.relaxation_energy,
            plasmon_frequency: istprm.interstitial.interstitial_density.sqrt(),
            core_valence_energy: core_valence.core_valence_energy,
            density_radius: istprm.interstitial.fermi.density_parameter,
            fermi_momentum: istprm.interstitial.fermi.fermi_momentum,
            total_charge,
            // `totvol` is an input/output field in FEFF. ISTPRM derives a
            // separate interstitial volume from non-positive inputs without
            // replacing the value serialized into pot.bin.
            total_volume: pot_input_total_volume_bohr3(input)?,
        },
        muffin_tin_indices: istprm.interstitial.muffin_tin_indices,
        muffin_tin_radii: istprm.interstitial.muffin_tin_radii,
        norman_indices: istprm.interstitial.norman_indices,
        atomic_numbers,
        kappa,
        norman_radii: istprm.interstitial.norman_radii,
        overlap_factors: istprm.overlap_factors,
        max_overlap_factors: istprm.max_overlap_factors,
        potential_multiplicities,
        ionization,
        initial_large_component: core_hole.large_component,
        initial_small_component: core_hole.small_component,
        large_components,
        small_components,
        large_coefficients,
        small_coefficients,
        electron_density: overlap_arrays.overlapped_density,
        coulomb_potential: overlap_arrays.overlapped_coulomb_potential,
        total_potential: istprm.interstitial.total_potential,
        valence_density,
        valence_potential: istprm.interstitial.valence_potential,
        magnetization_density: Array2::from_shape_fn(
            (overlap_arrays.magnetization_density.nrows(), unique_count),
            |(row, potential)| overlap_arrays.magnetization_density[(row, potential)],
        ),
        orbital_occupancy,
        orbital_energies,
        occupied_orbital_indices,
        norman_charges,
        valence_occupancy,
        raw_text: None,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct NoScfPotIstprmState {
    interstitial: MuffinTinInterstitialParameters,
    overlap_factors: Array1<f64>,
    max_overlap_factors: Array1<f64>,
}

fn no_scf_pot_istprm_state(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    overlap: &AtomicApotOverlapArrays,
    potential_multiplicities: ArrayView1<'_, f64>,
    total_charge: f64,
) -> Result<NoScfPotIstprmState> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        unique_count > 0 && potential_multiplicities.len() == unique_count,
        "POT no-SCF ISTPRM requires {unique_count} positive potential multiplicities, got {}",
        potential_multiplicities.len()
    );
    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let explicit_overlaps = no_scf_pot_muffin_tin_overlaps(static_arrays, unique_count)?;
    let explicit_overlap_refs = explicit_overlaps
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let requested_overlap_factors = no_scf_pot_overlap_factors(input, unique_count)?;
    if unique_count == 1 {
        return no_scf_single_potential_istprm_state(input, overlap, requested_overlap_factors);
    }
    let initial_overlap_factors = if input.control.iafolp >= 0 {
        Array1::ones(unique_count)
    } else {
        requested_overlap_factors.clone()
    };
    let interstitial_selector = usize::try_from(input.run.inters)
        .context("POT no-SCF interstitial selector cannot be represented as usize")?;
    let radius_state = muffin_tin_radius_parameters(MuffinTinRadiusParametersInput {
        highest_potential_index: unique_count - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        explicit_overlaps: &explicit_overlap_refs,
        norman_radii: overlap.norman_radii.view(),
        overlap_factors: initial_overlap_factors.view(),
        max_overlap_factors: requested_overlap_factors.view(),
        coulomb_potential: overlap.overlapped_coulomb_potential.view(),
        afolp_enabled: input.control.iafolp > 0,
        interstitial_selector,
    })
    .map(|state| {
        (
            state.muffin_tin_radii,
            state.norman_radii,
            state.max_overlap_factors,
            state.near_neighbor_flags,
            state.interstitial_selector,
        )
    });
    let (
        initial_muffin_tin_radii,
        norman_radii,
        max_overlap_factors,
        near_neighbor_flags,
        interstitial_selector,
    ) = match radius_state {
        Ok(state) => state,
        Err(error) => {
            return Err(error).context("failed to calculate POT no-SCF FEFF ISTPRM radius state");
        }
    };

    let overlap_factors = if input.control.iafolp >= 0 {
        max_overlap_factors.clone()
    } else {
        initial_overlap_factors.clone()
    };
    let muffin_tin_radii = Array1::from_shape_fn(unique_count, |potential| {
        initial_muffin_tin_radii[potential] * overlap_factors[potential]
            / initial_overlap_factors[potential]
    });
    let interstitial = muffin_tin_interstitial_parameters(MuffinTinInterstitialParametersInput {
        highest_potential_index: unique_count - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        potential_multiplicities,
        explicit_overlaps: &explicit_overlap_refs,
        electron_density: overlap.overlapped_density.view(),
        valence_density: overlap.overlapped_valence_density.view(),
        magnetization: overlap.magnetization_density.view(),
        coulomb_potential: overlap.overlapped_coulomb_potential.view(),
        muffin_tin_radii: muffin_tin_radii.view(),
        norman_radii: norman_radii.view(),
        near_neighbor_flags: near_neighbor_flags.view(),
        exchange_selector: input.control.ixc,
        scf_exchange_selector: input.control.iscfxc,
        spin_polarization: 0,
        scf_temperature_hartree: input.thermal.scf_temperature / FEFF_HARTREE_EV,
        total_charge,
        // FEFF initializes xmu to 100 Ha before the first ISTPRM pass.
        fermi_level: 100.0,
        total_volume: pot_input_total_volume_bohr3(input)?,
        interstitial_selector,
    })
    .context("failed to calculate POT no-SCF FEFF ISTPRM interstitial state")?;

    Ok(NoScfPotIstprmState {
        interstitial,
        overlap_factors,
        max_overlap_factors,
    })
}

fn no_scf_single_potential_istprm_state(
    input: &PotInput,
    overlap: &AtomicApotOverlapArrays,
    overlap_factors: Array1<f64>,
) -> Result<NoScfPotIstprmState> {
    let total_potential = no_scf_pot_total_potential(input, overlap)?;
    let valence_potential = no_scf_pot_valence_potential(input, overlap, &total_potential)?;
    let norman_radius = overlap.norman_radii[0];
    let mut selected_indices = None;
    let mut last_error = None;
    for divisor in [overlap_factors[0].max(1.05), 1.25, 1.5, 2.0, 3.0, 5.0, 8.0] {
        let muffin_tin_radius = norman_radius / divisor;
        match overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: overlap.overlapped_density.column(0),
            muffin_tin_radius,
            norman_radius,
        }) {
            Ok(indices) => {
                selected_indices = Some(indices);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let indices = match selected_indices {
        Some(indices) => indices,
        None => {
            let error = last_error.context("POT no-SCF radius candidate list was empty")?;
            return Err(error)
                .context("failed to locate POT no-SCF single-potential radial indices");
        }
    };
    let shell = interstitial_shell_values(InterstitialShellValuesInput {
        total_potential: total_potential.column(0),
        overlapped_density: overlap.overlapped_density.column(0),
        muffin_tin_radius: indices.muffin_tin_radius,
        muffin_tin_index: indices.muffin_tin_index,
        wigner_seitz_radius: indices.norman_radius,
        wigner_seitz_index: indices.norman_index,
    })
    .context("failed to calculate POT no-SCF single-potential interstitial shell")?;
    let fermi = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: shell.interstitial_density,
        interstitial_potential: shell.interstitial_potential,
    })
    .context("failed to calculate POT no-SCF single-potential Fermi level")?;

    Ok(NoScfPotIstprmState {
        interstitial: MuffinTinInterstitialParameters {
            total_potential,
            valence_potential,
            max_density_indices: Array1::from_vec(vec![indices.max_density_index]),
            muffin_tin_indices: Array1::from_vec(vec![indices.muffin_tin_index]),
            muffin_tin_radii: Array1::from_vec(vec![indices.muffin_tin_radius]),
            norman_indices: Array1::from_vec(vec![indices.norman_index]),
            norman_radii: Array1::from_vec(vec![indices.norman_radius]),
            average_norman_radius: indices.norman_radius,
            interstitial_volume: 4.0 * std::f64::consts::PI * shell.shell_volume,
            interstitial_potential: shell.interstitial_potential,
            interstitial_density: shell.interstitial_density,
            fermi,
            interstitial_potential_limited: false,
        },
        max_overlap_factors: overlap_factors.clone(),
        overlap_factors,
    })
}

fn no_scf_pot_total_potential(
    input: &PotInput,
    overlap: &AtomicApotOverlapArrays,
) -> Result<Array2<f64>> {
    let (rows, potentials) = overlap.overlapped_density.dim();
    ensure!(
        overlap.overlapped_coulomb_potential.dim() == (rows, potentials),
        "POT no-SCF Coulomb potential shape {:?} does not match density shape {:?}",
        overlap.overlapped_coulomb_potential.dim(),
        overlap.overlapped_density.dim()
    );
    let magnetization_shape = overlap.magnetization_density.dim();
    ensure!(
        magnetization_shape.0 == rows && magnetization_shape.1 >= potentials,
        "POT no-SCF magnetization shape {:?} does not cover density shape {:?}",
        magnetization_shape,
        overlap.overlapped_density.dim()
    );

    let mut total = Array2::<f64>::zeros((rows, potentials));
    for potential in 0..potentials {
        for row in 0..rows {
            let density = overlap.overlapped_density[(row, potential)];
            let vxc = if density > 0.0 {
                // FEFF POT hard-codes `idmag = 0` before every ISTPRM call,
                // so the ground-state XC potential is unpolarized even when
                // ATOM supplied a nonzero `dmag / edens` diagnostic.
                no_scf_pot_ground_state_vxc(input, density, 0.0, row, potential)?
            } else {
                0.0
            };
            let value = overlap.overlapped_coulomb_potential[(row, potential)] + vxc;
            ensure!(
                value.is_finite(),
                "POT no-SCF total potential row {row} potential {potential} is non-finite"
            );
            total[(row, potential)] = value;
        }
    }
    Ok(total)
}

fn no_scf_pot_valence_potential(
    input: &PotInput,
    overlap: &AtomicApotOverlapArrays,
    total_potential: &Array2<f64>,
) -> Result<Array2<f64>> {
    let branch = no_scf_pot_exchange_branch(input.control.ixc);
    if branch < 5 {
        return Ok(total_potential.clone());
    }

    let (rows, potentials) = overlap.overlapped_density.dim();
    ensure!(
        overlap.overlapped_valence_density.dim() == (rows, potentials),
        "POT no-SCF valence density shape {:?} does not match density shape {:?}",
        overlap.overlapped_valence_density.dim(),
        overlap.overlapped_density.dim()
    );
    ensure!(
        overlap.overlapped_coulomb_potential.dim() == (rows, potentials),
        "POT no-SCF Coulomb potential shape {:?} does not match density shape {:?}",
        overlap.overlapped_coulomb_potential.dim(),
        overlap.overlapped_density.dim()
    );
    let magnetization_shape = overlap.magnetization_density.dim();
    ensure!(
        magnetization_shape.0 == rows && magnetization_shape.1 >= potentials,
        "POT no-SCF magnetization shape {:?} does not cover density shape {:?}",
        magnetization_shape,
        overlap.overlapped_density.dim()
    );
    ensure!(
        total_potential.dim() == (rows, potentials),
        "POT no-SCF total potential shape {:?} does not match density shape {:?}",
        total_potential.dim(),
        overlap.overlapped_density.dim()
    );

    let mut valence = Array2::<f64>::zeros((rows, potentials));
    for potential in 0..potentials {
        for row in 0..rows {
            let density = overlap.overlapped_density[(row, potential)];
            let valence_density = overlap.overlapped_valence_density[(row, potential)];
            let coulomb = overlap.overlapped_coulomb_potential[(row, potential)];
            let total = total_potential[(row, potential)];
            let value = if branch == 5 {
                let mut valence_radius = 10.0;
                if valence_density > 1.0e-5 {
                    valence_radius = (valence_density / 3.0).powf(-1.0 / 3.0);
                }
                if valence_radius > 10.0 {
                    valence_radius = 10.0;
                }
                // FEFF POT keeps `idmag = 0`, hence xmagvl is exactly 1.
                let valence_spin_fraction_twice = 1.0;
                coulomb
                    + von_barth_hedin_potential(valence_radius, valence_spin_fraction_twice)
                        .with_context(|| {
                            format!(
                                "failed to calculate POT no-SCF valence XC row {row} potential {potential} for ixc={}",
                                input.control.ixc
                            )
                        })?
            } else {
                let core_radius = if density <= valence_density {
                    101.0
                } else {
                    ((density - valence_density) / 3.0).powf(-1.0 / 3.0)
                };
                // FEFF POT keeps `idmag = 0`, so rsmag is based on the
                // unpolarized total density.
                let magnetized_density = density;
                let magnetized_radius = if magnetized_density > 0.0 {
                    (magnetized_density / 3.0).powf(-1.0 / 3.0)
                } else {
                    100.0
                };
                let magnetized_fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / magnetized_radius;
                total - dirac_hara_exchange_potential(core_radius, magnetized_fermi_momentum)
                    .with_context(|| {
                        format!(
                            "failed to calculate POT no-SCF valence Dirac-Hara correction row {row} potential {potential} for ixc={}",
                            input.control.ixc
                        )
                    })?
            };
            ensure!(
                value.is_finite(),
                "POT no-SCF valence potential row {row} potential {potential} is non-finite"
            );
            valence[(row, potential)] = value;
        }
    }
    Ok(valence)
}

fn no_scf_pot_ground_state_vxc(
    input: &PotInput,
    density: f64,
    magnetization_ratio: f64,
    row: usize,
    potential: usize,
) -> Result<f64> {
    ensure!(
        density.is_finite() && density > 0.0,
        "POT no-SCF ground-state XC density row {row} potential {potential} must be positive and finite, got {density}"
    );
    ensure!(
        magnetization_ratio.is_finite(),
        "POT no-SCF ground-state XC magnetization row {row} potential {potential} is non-finite"
    );
    let density_radius = (density / 3.0).powf(-1.0 / 3.0);
    let spin_fraction_twice = 1.0 + magnetization_ratio;
    ensure!(
        spin_fraction_twice.is_finite() && spin_fraction_twice >= 0.0,
        "POT no-SCF ground-state XC spin fraction row {row} potential {potential} must be non-negative and finite, got {spin_fraction_twice}"
    );
    let scf_temperature_hartree = input.thermal.scf_temperature / FEFF_HARTREE_EV;
    let vxc = match input.control.iscfxc {
        11 => von_barth_hedin_potential(density_radius, spin_fraction_twice),
        12 => perdew_zunger_vxc(density_radius),
        21 => perrot_dharma_wardana_vxc(density_radius, scf_temperature_hartree),
        22 => karasiev_sjostrom_dufty_trickey_vxc(density_radius, scf_temperature_hartree),
        selector => bail!("POT no-SCF ground-state XC selector iscfxc={selector} is unsupported"),
    }
    .with_context(|| {
        format!(
            "failed to calculate POT no-SCF ground-state XC row {row} potential {potential} with iscfxc={}",
            input.control.iscfxc
        )
    })?;
    ensure!(
        vxc.is_finite(),
        "POT no-SCF ground-state XC row {row} potential {potential} is non-finite"
    );
    Ok(vxc)
}

fn no_scf_pot_exchange_branch(exchange_selector: i32) -> i32 {
    exchange_selector.rem_euclid(10)
}

fn no_scf_pot_muffin_tin_overlaps(
    static_arrays: &AtomicApotStaticArrays,
    unique_count: usize,
) -> Result<Vec<Vec<MuffinTinOverlapNeighbor>>> {
    (0..unique_count)
        .map(|potential| {
            atomic_apot_explicit_overlap_neighbors(static_arrays, potential)?
                .into_iter()
                .map(|neighbor| {
                    ensure!(
                        neighbor.multiplicity.is_finite()
                            && neighbor.multiplicity > 0.0
                            && neighbor.multiplicity.fract() == 0.0,
                        "POT no-SCF explicit overlap multiplicity for potential {potential} must be a positive integer, got {}",
                        neighbor.multiplicity
                    );
                    Ok(MuffinTinOverlapNeighbor {
                        source_potential: neighbor.source_potential,
                        multiplicity: neighbor.multiplicity as usize,
                        distance: neighbor.distance,
                    })
                })
                .collect()
        })
        .collect()
}

fn pot_input_total_volume_bohr3(input: &PotInput) -> Result<f64> {
    let volume = input.scattering.totvol;
    ensure!(
        volume.is_finite(),
        "POT total volume must be finite, got {volume}"
    );
    if volume <= 0.0 {
        return Ok(volume);
    }
    let converted = volume / FEFF_BOHR_ANGSTROM.powi(3);
    ensure!(
        converted.is_finite() && converted > 0.0,
        "POT total volume conversion to Bohr^3 produced invalid value {converted}"
    );
    Ok(converted)
}

#[allow(clippy::too_many_arguments)]
fn no_scf_pot_copy_state_orbitals(
    state: &AtomicScfState,
    potential_index: usize,
    kappa: &mut Array1<i32>,
    orbital_energies: &mut Array1<f64>,
    orbital_occupancy: &mut Array2<f64>,
    large_components: &mut Array3<f64>,
    small_components: &mut Array3<f64>,
    large_coefficients: &mut Array3<f64>,
    small_coefficients: &mut Array3<f64>,
) -> Result<()> {
    let orbital_count = state.kappas.len();
    ensure!(
        orbital_count <= POT_BIN_ORBITALS,
        "POT no-SCF state has {orbital_count} orbital(s), but pot.bin stores {POT_BIN_ORBITALS}"
    );
    ensure!(
        state.scf.large_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.small_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.large_components.ncols() >= orbital_count
            && state.scf.small_components.ncols() >= orbital_count,
        "POT no-SCF state component shapes large={:?}, small={:?}, expected at least {ATOM_RADIAL_POINTS}x{orbital_count}",
        state.scf.large_components.dim(),
        state.scf.small_components.dim()
    );
    ensure!(
        state.scf.large_coefficients.ncols() >= orbital_count
            && state.scf.small_coefficients.ncols() >= orbital_count,
        "POT no-SCF state coefficient shapes large={:?}, small={:?}, expected one column per orbital",
        state.scf.large_coefficients.dim(),
        state.scf.small_coefficients.dim()
    );
    ensure!(
        state.valence_occupations.len() >= orbital_count
            && state.scf.orbital_energies.len() >= orbital_count,
        "POT no-SCF state has orbital_count={orbital_count} but valence occupation len={} and energy len={}",
        state.valence_occupations.len(),
        state.scf.orbital_energies.len()
    );

    let output = atomic_output_state(state)?;
    let state = &output;
    for orbital in 0..orbital_count {
        if potential_index == 0 {
            kappa[orbital] = state.kappas[orbital];
            orbital_energies[orbital] = state.scf.orbital_energies[orbital];
        }
        orbital_occupancy[(orbital, potential_index)] = state.valence_occupations[orbital];
        for row in 0..ATOM_RADIAL_POINTS {
            large_components[(row, orbital, potential_index)] =
                state.scf.large_components[(row, orbital)];
            small_components[(row, orbital, potential_index)] =
                state.scf.small_components[(row, orbital)];
        }
        let coefficient_count = POT_BIN_COEFFICIENTS
            .min(state.scf.large_coefficients.nrows())
            .min(state.scf.small_coefficients.nrows());
        for coefficient in 0..coefficient_count {
            large_coefficients[(coefficient, orbital, potential_index)] =
                state.scf.large_coefficients[(coefficient, orbital)];
            small_coefficients[(coefficient, orbital, potential_index)] =
                state.scf.small_coefficients[(coefficient, orbital)];
        }
    }
    Ok(())
}

fn no_scf_pot_occupied_orbital_indices(
    input: &PotInput,
    config_inp: &Path,
    unique_count: usize,
) -> Result<Array2<i32>> {
    let source = atomic_orbital_indices_by_kappa(input, config_inp)?;
    ensure!(
        source.nrows() == POT_BIN_IORB_SLOTS && source.ncols() >= unique_count,
        "POT no-SCF iorb shape {:?} cannot provide {POT_BIN_IORB_SLOTS}x{unique_count}",
        source.dim()
    );
    let mut values = Array2::<i32>::zeros((POT_BIN_IORB_SLOTS, unique_count));
    for potential in 0..unique_count {
        for slot in 0..POT_BIN_IORB_SLOTS {
            values[(slot, potential)] = i32::try_from(source[(slot, potential)])
                .context("POT no-SCF iorb value cannot be represented as i32")?;
        }
    }
    Ok(values)
}

fn no_scf_pot_valence_occupancy(
    input: &PotInput,
    config_inp: &Path,
    unique_count: usize,
) -> Result<Array2<f64>> {
    let source = atomic_norman_valence_counts_by_l(input, config_inp)?;
    ensure!(
        source.ncols() >= unique_count,
        "POT no-SCF valence occupancy shape {:?} cannot provide {unique_count} potential(s)",
        source.dim()
    );
    Ok(Array2::from_shape_fn(
        (source.nrows(), unique_count),
        |(row, potential)| source[(row, potential)],
    ))
}

#[derive(Debug, Clone, Copy)]
struct PotCoreValenceMarker {
    potential: usize,
    angular: usize,
    orbital: usize,
    energy: f64,
    initial_is_valence: bool,
    is_valence: bool,
}

#[derive(Debug, Clone)]
struct PotCoreValenceSelection {
    core_valence_energy: f64,
    markers: Vec<PotCoreValenceMarker>,
}

fn no_scf_pot_core_valence_selection(
    input: &PotInput,
    states: &[AtomicScfState],
    atomic_numbers: &Array1<usize>,
    interstitial_potential: f64,
    peak_energies: Option<ArrayView2<'_, f64>>,
) -> Result<PotCoreValenceSelection> {
    ensure!(
        interstitial_potential.is_finite(),
        "POT core-valence interstitial potential is non-finite: {interstitial_potential}"
    );
    let input_core_valence_energy = pot_input_core_valence_energy_hartree(input)?;
    ensure!(
        input.scattering.corval_emin.is_finite(),
        "POT CORVAL lower bound is non-finite: {}",
        input.scattering.corval_emin
    );
    let unique_count = atomic_numbers.len();
    if let Some(peaks) = peak_energies {
        ensure!(
            peaks.nrows() >= ATOM_NORMAN_VALENCE_CHANNEL_COUNT && peaks.ncols() >= unique_count,
            "POT core-valence LDOS peak table shape {:?} cannot provide {}x{} channels",
            peaks.dim(),
            ATOM_NORMAN_VALENCE_CHANNEL_COUNT,
            unique_count
        );
    }
    ensure!(
        states.len() >= unique_count,
        "POT core-valence selection has {} ATOM state(s), expected at least {unique_count}",
        states.len()
    );

    let tolerance = POT_CORVAL_TOLERANCE_EV / FEFF_HARTREE_EV;
    let mut core_valence_energy = input_core_valence_energy;
    if interstitial_potential - core_valence_energy < tolerance {
        core_valence_energy = interstitial_potential - tolerance;
    }
    let lower_bound = (input.scattering.corval_emin / FEFF_HARTREE_EV).min(core_valence_energy);
    let upper_bound = POT_CORVAL_HIGH_EV / FEFF_HARTREE_EV;
    let mut markers = vec![None; unique_count.saturating_mul(ATOM_NORMAN_VALENCE_CHANNEL_COUNT)];

    for potential in 0..unique_count {
        let state = &states[potential];
        let orbital_count = state.kappas.len();
        ensure!(
            state.valence_occupations.len() >= orbital_count
                && state.scf.orbital_energies.len() >= orbital_count,
            "POT core-valence state {potential} has orbital_count={orbital_count} but valence occupation len={} and energy len={}",
            state.valence_occupations.len(),
            state.scf.orbital_energies.len()
        );
        let atomic_number = atomic_numbers[potential];
        for orbital in 0..orbital_count {
            let mut energy = apot_handoff_value(state.scf.orbital_energies[orbital])?;
            ensure!(
                energy.is_finite(),
                "POT core-valence state {potential} orbital {} has non-finite energy {energy}",
                orbital + 1
            );
            if !(energy < upper_bound - tolerance && energy > lower_bound) {
                continue;
            }
            let angular =
                atomic_angular_momentum_from_kappa(state.kappas[orbital]).with_context(|| {
                    format!(
                        "POT core-valence state {potential} orbital {} has invalid kappa {}",
                        orbital + 1,
                        state.kappas[orbital]
                    )
                })?;
            if ((input.run.iunf == 0 || (71..=73).contains(&atomic_number)) && angular >= 3)
                || angular >= ATOM_NORMAN_VALENCE_CHANNEL_COUNT
            {
                continue;
            }
            if let Some(peaks) = peak_energies {
                let peak = peaks[(angular, potential)];
                if peak.is_finite() {
                    energy = peak;
                }
            }
            let valence_occupation = state.valence_occupations[orbital];
            ensure!(
                valence_occupation.is_finite(),
                "POT core-valence state {potential} orbital {} has non-finite valence occupation {valence_occupation}",
                orbital + 1
            );
            markers[potential * ATOM_NORMAN_VALENCE_CHANNEL_COUNT + angular] =
                Some(PotCoreValenceMarker {
                    potential,
                    angular,
                    orbital,
                    energy,
                    initial_is_valence: valence_occupation >= 0.1,
                    is_valence: valence_occupation >= 0.1,
                });
        }
    }

    let mut markers = markers.into_iter().flatten().collect::<Vec<_>>();
    if markers.is_empty() {
        return Ok(PotCoreValenceSelection {
            core_valence_energy,
            markers,
        });
    }
    markers.sort_by(|left, right| left.energy.total_cmp(&right.energy));
    if let (Some(core), Some(valence)) = (
        markers.iter().rposition(|marker| !marker.is_valence),
        markers.iter().position(|marker| marker.is_valence),
    ) && valence < core
    {
        for marker in markers.iter_mut().take(core + 1).skip(valence + 1) {
            marker.is_valence = true;
        }
    }

    for _ in 0..=markers.len() {
        let highest_core = markers.iter().rposition(|marker| !marker.is_valence);
        let lowest_valence = markers.iter().position(|marker| marker.is_valence);
        let ok = match (highest_core, lowest_valence) {
            (Some(core), Some(valence)) => {
                core_valence_energy - markers[core].energy > tolerance
                    && markers[valence].energy - core_valence_energy > tolerance
            }
            (Some(core), None) => core_valence_energy - markers[core].energy > tolerance,
            (None, Some(valence)) => markers[valence].energy - core_valence_energy > tolerance,
            (None, None) => true,
        };
        if ok {
            return Ok(PotCoreValenceSelection {
                core_valence_energy,
                markers,
            });
        }

        core_valence_energy = interstitial_potential - tolerance;
        if let Some(valence) = lowest_valence {
            core_valence_energy = core_valence_energy.min(markers[valence].energy - tolerance);
        }
        let Some(core) = highest_core else {
            return Ok(PotCoreValenceSelection {
                core_valence_energy,
                markers,
            });
        };
        if core_valence_energy - markers[core].energy > tolerance {
            return Ok(PotCoreValenceSelection {
                core_valence_energy,
                markers,
            });
        }
        markers[core].is_valence = true;
    }

    bail!("POT core-valence selection did not converge after marker reassignment")
}

fn no_scf_pot_apply_core_valence_selection(
    selection: &PotCoreValenceSelection,
    orbital_occupancy: &mut Array2<f64>,
    valence_occupancy: &mut Array2<f64>,
    valence_density: &mut Array2<f64>,
    large_components: ndarray::ArrayView3<'_, f64>,
    small_components: ndarray::ArrayView3<'_, f64>,
) -> Result<()> {
    // CORVAL initializes ri05 with the same entirely REAL expression as
    // SCMT. Reassigned bound-state density must use those saved radii.
    let radii = scf_pot_density_output_grid();
    ensure!(
        valence_density.nrows() >= POT_BIN_RADIAL_POINTS
            && large_components.dim().0 >= POT_BIN_RADIAL_POINTS
            && small_components.dim().0 >= POT_BIN_RADIAL_POINTS,
        "POT core-valence density/component tables must provide {POT_BIN_RADIAL_POINTS} radial rows"
    );
    ensure!(
        large_components.dim() == small_components.dim(),
        "POT core-valence large/small component shapes differ: {:?} vs {:?}",
        large_components.dim(),
        small_components.dim()
    );

    for marker in selection
        .markers
        .iter()
        .filter(|marker| marker.is_valence && !marker.initial_is_valence)
    {
        ensure!(
            marker.potential < orbital_occupancy.ncols()
                && marker.potential < valence_occupancy.ncols()
                && marker.potential < valence_density.ncols()
                && marker.potential < large_components.dim().2,
            "POT core-valence reassignment potential {} is outside available tables",
            marker.potential
        );
        ensure!(
            marker.angular < valence_occupancy.nrows(),
            "POT core-valence reassignment angular channel {} is outside xnvmu rows {}",
            marker.angular,
            valence_occupancy.nrows()
        );
        ensure!(
            marker.orbital < orbital_occupancy.nrows() && marker.orbital < large_components.dim().1,
            "POT core-valence reassignment orbital {} is outside available orbital tables",
            marker.orbital
        );

        valence_occupancy[(marker.angular, marker.potential)] += (4 * marker.angular + 2) as f64;
        no_scf_pot_set_valence_orbital_occupancy(
            orbital_occupancy,
            marker.potential,
            marker.orbital,
            marker.angular,
        )?;
        no_scf_pot_add_core_valence_density(
            valence_density,
            large_components,
            small_components,
            radii.view(),
            marker.potential,
            marker.orbital,
            (2 * (marker.angular + 1)) as f64,
        )?;
        if marker.angular != 0 {
            let partner = marker.orbital.checked_sub(1).with_context(|| {
                format!(
                    "POT core-valence l={} reassignment for orbital {} has no spin-orbit partner",
                    marker.angular, marker.orbital
                )
            })?;
            no_scf_pot_set_valence_orbital_occupancy(
                orbital_occupancy,
                marker.potential,
                partner,
                marker.angular - 1,
            )?;
            no_scf_pot_add_core_valence_density(
                valence_density,
                large_components,
                small_components,
                radii.view(),
                marker.potential,
                partner,
                (2 * marker.angular) as f64,
            )?;
        }
    }
    Ok(())
}

fn no_scf_pot_set_valence_orbital_occupancy(
    orbital_occupancy: &mut Array2<f64>,
    potential: usize,
    orbital: usize,
    angular: usize,
) -> Result<()> {
    ensure!(
        orbital < orbital_occupancy.nrows() && potential < orbital_occupancy.ncols(),
        "POT core-valence xnval update orbital {orbital}, potential {potential} is outside {:?}",
        orbital_occupancy.dim()
    );
    if orbital_occupancy[(orbital, potential)] < 0.1 {
        orbital_occupancy[(orbital, potential)] = (2 * angular + 2) as f64;
    }
    Ok(())
}

fn no_scf_pot_add_core_valence_density(
    valence_density: &mut Array2<f64>,
    large_components: ndarray::ArrayView3<'_, f64>,
    small_components: ndarray::ArrayView3<'_, f64>,
    radii: ArrayView1<'_, f64>,
    potential: usize,
    orbital: usize,
    weight: f64,
) -> Result<()> {
    ensure!(
        orbital < large_components.dim().1
            && potential < large_components.dim().2
            && potential < valence_density.ncols(),
        "POT core-valence density update orbital {orbital}, potential {potential} is outside component/density tables"
    );
    for radial in 0..POT_BIN_RADIAL_POINTS {
        let radius = radii[radial];
        ensure!(
            radius.is_finite() && radius > 0.0,
            "POT core-valence radial grid row {} is invalid: {radius}",
            radial + 1
        );
        let large = large_components[(radial, orbital, potential)];
        let small = small_components[(radial, orbital, potential)];
        valence_density[(radial, potential)] +=
            weight * (large * large + small * small) / radius.powi(2);
    }
    Ok(())
}

fn no_scf_pot_atomic_numbers(input: &PotInput, unique_count: usize) -> Result<Array1<usize>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT no-SCF pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut values = Array1::<usize>::zeros(unique_count);
    for potential in 0..unique_count {
        values[potential] = checked_atomic_number(input.potentials[potential].z)?;
    }
    Ok(values)
}

fn generated_pot_potential_multiplicities(
    input: &PotInput,
    unique_count: usize,
) -> Result<Array1<f64>> {
    // FEFF passes POT's xnatph values to ISTPRM for both SCF and no-SCF
    // runs. geom.dat contains only the finite calculation cluster and its
    // row counts are not crystallographic potential multiplicities.
    pot_input_potential_multiplicities(input, unique_count)
}

fn pot_input_potential_multiplicities(
    input: &PotInput,
    unique_count: usize,
) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut values = Array1::<f64>::zeros(unique_count);
    for potential in 0..unique_count {
        let multiplicity = input.potentials[potential].xnatph;
        ensure!(
            multiplicity.is_finite() && multiplicity > 0.0,
            "POT xnatph for potential {potential} must be positive and finite, got {multiplicity}"
        );
        values[potential] = multiplicity;
    }
    Ok(values)
}

fn no_scf_pot_ionization(input: &PotInput, unique_count: usize) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT no-SCF pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    Ok(Array1::from_shape_fn(unique_count, |potential| {
        input.potentials[potential].xion
    }))
}

fn no_scf_pot_overlap_factors(input: &PotInput, unique_count: usize) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT no-SCF pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut values = Array1::<f64>::zeros(unique_count);
    for potential in 0..unique_count {
        let value = input.potentials[potential].folp;
        ensure!(
            value.is_finite() && value > 0.0,
            "POT no-SCF folp for potential {potential} must be positive and finite, got {value}"
        );
        values[potential] = value;
    }
    Ok(values)
}

fn no_scf_pot_total_charge(
    atomic_numbers: &Array1<usize>,
    potential_multiplicities: &Array1<f64>,
    ionization: &Array1<f64>,
) -> Result<f64> {
    ensure!(
        atomic_numbers.len() == potential_multiplicities.len()
            && atomic_numbers.len() == ionization.len(),
        "POT no-SCF total charge arrays have mismatched lengths iz={}, xnatph={}, xion={}",
        atomic_numbers.len(),
        potential_multiplicities.len(),
        ionization.len()
    );
    let mut total = 0.0;
    for potential in 0..atomic_numbers.len() {
        total += potential_multiplicities[potential]
            * (atomic_numbers[potential] as f64 - ionization[potential]);
    }
    ensure!(
        total.is_finite(),
        "POT no-SCF total charge is non-finite: {total}"
    );
    Ok(total)
}

#[allow(dead_code)]
fn generated_atomic_scf_states(input: &PotInput, config_inp: &Path) -> Result<Vec<AtomicScfState>> {
    let state_count = apot_state_count(input)?;
    ensure!(
        input.potentials.len() + 1 == state_count,
        "ATOM pot.inp has {} potential row(s), expected {} from nph={}",
        input.potentials.len(),
        state_count - 1,
        input.control.nph
    );

    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );

    configurations
        .iter()
        .enumerate()
        .map(|(state_index, configuration)| {
            let atomic_number = atomic_number_for_apot_state(input, state_index)?;
            let ionicity = effective_ionicity_for_apot_state(input, state_index, configuration)?;
            atomic_scf_state_from_configuration(AtomicScfStateInput {
                atomic_number,
                ionicity,
                thomas_fermi_ionicity: -ionicity - 1.0,
                configuration,
                exchange_mode: AtomicLocalDensityExchangeMode::DiracFockOnly,
                max_orbital_iterations: ATOM_SCF_MAX_ORBITAL_ITERATIONS,
                speed_of_light: ATOM_TOTAL_ENERGY_SPEED_OF_LIGHT,
                step: ATOM_RADIAL_STEP,
                requested_nucleus_index: atomic_nucleus_request_index(input),
                first_radius_times_charge: atomic_first_radius_times_charge(atomic_number),
            })
            .with_context(|| format!("failed to generate ATOM SCF state column {state_index}"))
        })
        .collect()
}

#[allow(dead_code)]
fn atomic_scf_configurations_from_pot_input(
    input: &PotInput,
    config_inp: &Path,
) -> Result<Vec<OrbitalConfiguration>> {
    let recipe = configuration_recipe(input.config_type);
    let source = ConfigurationRowsSource::new(input, recipe, config_inp)?;
    atomic_scf_orbital_configurations(input, &source)
}

#[allow(dead_code)]
fn generated_atomic_core_hole_columns(
    input: &PotInput,
    config_inp: &Path,
) -> Result<ApotCoreHoleColumns> {
    let states = generated_atomic_scf_states(input, config_inp)?;
    atomic_core_hole_columns_from_states(input, config_inp, &states)
}

#[allow(dead_code)]
fn atomic_core_hole_columns_from_states(
    input: &PotInput,
    config_inp: &Path,
    states: &[AtomicScfState],
) -> Result<ApotCoreHoleColumns> {
    let state_count = apot_state_count(input)?;
    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );

    let initial_column = fpf0_initial_absorber_column(input)?;
    ensure!(
        initial_column < state_count,
        "ATOM core-hole initial absorber column {initial_column} exceeds apot state count {state_count}"
    );
    let (large_component, small_component) = atomic_core_hole_components(
        input,
        &configurations[initial_column],
        &states[initial_column],
    )?;
    let density = atomic_core_hole_density(
        input,
        large_component.view(),
        small_component.view(),
        states,
    )?;
    let coulomb_potential = apot_core_hole_coulomb_from_density(density.view(), input.run.nohole)
        .context("failed to generate ATOM core-hole Coulomb potential")?;

    Ok(ApotCoreHoleColumns {
        large_component,
        small_component,
        density,
        coulomb_potential,
    })
}

#[allow(dead_code)]
fn atomic_core_hole_components(
    input: &PotInput,
    configuration: &OrbitalConfiguration,
    state: &AtomicScfState,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let mut large_component = Array1::<f64>::zeros(ATOM_RADIAL_POINTS);
    let mut small_component = Array1::<f64>::zeros(ATOM_RADIAL_POINTS);
    let hole_index = checked_hole_index(input.control.ihole)?;
    let Some(orbital) = compact_orbital_for_feff_slot(configuration, hole_index)? else {
        return Ok((large_component, small_component));
    };

    ensure!(
        state.scf.large_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.small_components.nrows() >= ATOM_RADIAL_POINTS,
        "ATOM core-hole radial component tables are too short: large={:?}, small={:?}",
        state.scf.large_components.dim(),
        state.scf.small_components.dim()
    );
    ensure!(
        state.scf.large_components.ncols() > orbital
            && state.scf.small_components.ncols() > orbital
            && state.scf.active_lengths.len() > orbital,
        "ATOM core-hole orbital {} is outside generated SCF table shapes large={:?}, small={:?}, active_lengths={}",
        orbital + 1,
        state.scf.large_components.dim(),
        state.scf.small_components.dim(),
        state.scf.active_lengths.len()
    );

    let active_len = state.scf.active_lengths[orbital].min(ATOM_RADIAL_POINTS);
    for row in 0..active_len {
        large_component[row] = state.scf.large_components[(row, orbital)];
        small_component[row] = state.scf.small_components[(row, orbital)];
    }
    Ok((
        atomic_output_bound_quantity(state, large_component.view())?,
        atomic_output_bound_quantity(state, small_component.view())?,
    ))
}

#[allow(dead_code)]
fn atomic_core_hole_density(
    input: &PotInput,
    large_component: ndarray::ArrayView1<'_, f64>,
    small_component: ndarray::ArrayView1<'_, f64>,
    states: &[AtomicScfState],
) -> Result<Array1<f64>> {
    if input.run.nohole <= 0 {
        return Ok(Array1::zeros(ATOM_RADIAL_POINTS));
    }
    ensure!(
        large_component.len() == ATOM_RADIAL_POINTS && small_component.len() == ATOM_RADIAL_POINTS,
        "ATOM core-hole component lengths are {} and {}, expected {ATOM_RADIAL_POINTS}",
        large_component.len(),
        small_component.len()
    );

    if input.run.nohole == 1 {
        let radii = apot_core_hole_radii(ATOM_RADIAL_POINTS);
        return Ok(Array1::from_shape_fn(ATOM_RADIAL_POINTS, |row| {
            (large_component[row] * large_component[row]
                + small_component[row] * small_component[row])
                / (2.0 * radii[row] * radii[row])
        }));
    }

    let state_count = apot_state_count(input)?;
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    let final_column = state_count
        .checked_sub(1)
        .context("ATOM apot state count must be positive")?;
    let initial = states
        .first()
        .context("ATOM core-hole density requires an initial absorber state")?;
    let final_state = states
        .get(final_column)
        .context("ATOM core-hole density requires a final absorber state")?;
    ensure_atomic_radial_state("initial absorber", initial)?;
    ensure_atomic_radial_state("final absorber", final_state)?;

    let initial = atomic_output_state(initial)?;
    let final_state = atomic_output_state(final_state)?;
    Ok(Array1::from_shape_fn(ATOM_RADIAL_POINTS, |row| {
        0.5 * (initial.scf.density_4pi[row]
            - initial.scf.valence_density_4pi[row]
            - final_state.scf.density_4pi[row]
            + final_state.scf.valence_density_4pi[row])
    }))
}

#[allow(dead_code)]
fn ensure_atomic_radial_state(label: &'static str, state: &AtomicScfState) -> Result<()> {
    ensure!(
        state.scf.density_4pi.len() >= ATOM_RADIAL_POINTS
            && state.scf.valence_density_4pi.len() >= ATOM_RADIAL_POINTS,
        "ATOM {label} density lengths are {} and {}, expected {ATOM_RADIAL_POINTS}",
        state.scf.density_4pi.len(),
        state.scf.valence_density_4pi.len()
    );
    Ok(())
}

#[allow(dead_code)]
fn compact_orbital_for_feff_slot(
    configuration: &OrbitalConfiguration,
    slot: usize,
) -> Result<Option<usize>> {
    if slot == 0 {
        return Ok(None);
    }
    ensure!(
        slot <= FEFF_ORBITAL_SLOT_COUNT,
        "ATOM core-hole slot {slot} is outside FEFF's 1..={FEFF_ORBITAL_SLOT_COUNT} configuration slots"
    );
    let principal_quantum_number = FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS[slot - 1];
    let kappa = FEFF_ORBITAL_KAPPAS[slot - 1];
    configuration
        .principal_quantum_numbers
        .iter()
        .zip(configuration.kappa.iter())
        .position(|(&principal, &candidate_kappa)| {
            principal == principal_quantum_number && candidate_kappa == kappa
        })
        .map(Some)
        .with_context(|| {
            format!(
                "ATOM compacted configuration has no orbital for FEFF slot {slot} (n={principal_quantum_number}, kappa={kappa})"
            )
        })
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct AtomicApotStaticArrays {
    unique_potential_count: usize,
    atom_count: usize,
    atomic_numbers: Array1<i64>,
    model_atom_indices: Array1<i64>,
    overlap_shell_counts: Array1<i64>,
    norman_radii: Array1<f64>,
    atom_potential_indices: Array1<i64>,
    atom_positions: Array2<f64>,
    overlap_potential_indices: Array2<i64>,
    overlap_shell_atom_counts: Array2<i64>,
    overlap_radii: Array2<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
struct AtomicApotOverlapArrays {
    norman_radii: Array1<f64>,
    magnetization_density: Array2<f64>,
    overlapped_density: Array2<f64>,
    overlapped_valence_density: Array2<f64>,
    overlapped_coulomb_potential: Array2<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct AtomicApotEnergyScalars {
    initial_total_energy: f64,
    final_total_energy: f64,
    frozen_orbital_energy: f64,
    relaxation_energy: f64,
    edge_energy: f64,
}

#[allow(dead_code)]
fn atomic_apot_overlap_arrays_from_states(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    states: &[AtomicScfState],
) -> Result<AtomicApotOverlapArrays> {
    let unique_count = apot_unique_potential_count(input)?;
    let state_count = apot_state_count(input)?;
    ensure!(
        static_arrays.unique_potential_count == unique_count,
        "ATOM static APOT arrays have {} unique potential(s), expected {unique_count}",
        static_arrays.unique_potential_count
    );
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    ensure!(
        state_count == unique_count + 1,
        "ATOM state count {state_count} does not match unique potential count {unique_count}"
    );

    let atom_potentials = atomic_apot_usize_potential_indices(
        "iphat",
        &static_arrays.atom_potential_indices,
        unique_count,
    )?;
    let representative_atoms = atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
    let atomic_numbers = atomic_apot_usize_atomic_numbers(static_arrays, unique_count)?;
    let atom_positions = atomic_apot_core_atom_positions(static_arrays)?;

    let mut electron_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    let mut valence_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    let mut coulomb_potential = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    let mut spin_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    for potential_index in 0..unique_count {
        let state = &states[potential_index];
        atomic_apot_ensure_overlap_state(potential_index, state)?;
        let free_spin_density = atomic_apot_free_spin_density_from_state(potential_index, state)?;
        let output = atomic_output_state(state)?;
        let state = &output;
        for row in 0..ATOM_RADIAL_POINTS {
            electron_density[(row, potential_index)] = state.scf.density_4pi[row];
            valence_density[(row, potential_index)] = state.scf.valence_density_4pi[row];
            coulomb_potential[(row, potential_index)] = state.scf.coulomb_potential[row];
            spin_density[(row, potential_index)] = free_spin_density[row];
        }
    }

    let mut norman_radii = Array1::<f64>::zeros(unique_count);
    let mut magnetization_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, state_count));
    let mut overlapped_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    let mut overlapped_valence_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    let mut overlapped_coulomb_potential = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));

    for potential_index in 0..unique_count {
        let explicit_overlaps =
            atomic_apot_explicit_overlap_neighbors(static_arrays, potential_index)?;
        if explicit_overlaps.is_empty()
            && !atomic_apot_has_geometry_overlap_neighbor(
                &atom_potentials,
                &atom_positions,
                &representative_atoms,
                potential_index,
            )?
        {
            let mut normalized_density = electron_density.column(potential_index).to_owned();
            atomic_apot_normalize_density_for_norman_radius(
                atomic_numbers[potential_index],
                &mut normalized_density,
            )?;
            for row in 0..ATOM_RADIAL_POINTS {
                electron_density[(row, potential_index)] = normalized_density[row];
            }
        }
        let overlap = overlap_potential_density(PotentialOverlapInput {
            potential_index,
            atom_potentials: atom_potentials.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            atomic_numbers: atomic_numbers.view(),
            explicit_overlaps: &explicit_overlaps,
            electron_density: electron_density.view(),
            spin_density: spin_density.view(),
            valence_density: valence_density.view(),
            coulomb_potential: coulomb_potential.view(),
        })
        .with_context(|| format!("failed to overlap ATOM potential {potential_index}"))?;

        norman_radii[potential_index] = overlap.norman_radius.radius;
        for row in 0..ATOM_RADIAL_POINTS {
            magnetization_density[(row, potential_index)] = overlap.spin_density_ratio[row];
            overlapped_density[(row, potential_index)] = overlap.electron_density[row];
            overlapped_valence_density[(row, potential_index)] = overlap.valence_density[row];
            overlapped_coulomb_potential[(row, potential_index)] = overlap.coulomb_potential[row];
        }
    }
    let alternate_absorber_spin =
        atomic_apot_free_spin_density_from_state(unique_count, &states[unique_count])?;
    for row in 0..ATOM_RADIAL_POINTS {
        // FEFF overlaps only the unique-potential columns `0:nph`.
        // Column `nph+1` remains the alternate absorber's free-atom dmag.
        magnetization_density[(row, unique_count)] = alternate_absorber_spin[row];
    }

    Ok(AtomicApotOverlapArrays {
        norman_radii,
        magnetization_density,
        overlapped_density,
        overlapped_valence_density,
        overlapped_coulomb_potential,
    })
}

/// Construct FEFF `ATOM/scfdat.f90`'s free-atom `dmag` column.
///
/// `xmag` selects the spin-polarizable orbitals. FEFF sums their converged
/// radial probability densities, normalizes only a strictly positive total
/// atomic moment, and divides by `r**2` before `ovrlp` converts it to the
/// stored `dmag / edens` ratio. Finite-nucleus states use a displaced native
/// radial grid, so FEFF cubic-interpolates the finished density in `log(r)`
/// onto the fixed APOT grid before overlap or alternate-absorber storage.
fn atomic_apot_free_spin_density_from_state(
    potential_index: usize,
    state: &AtomicScfState,
) -> Result<Array1<f64>> {
    let orbital_count = state.spin_magnetization.len();
    let radial_count = state.initial_orbitals.radii.len();
    ensure!(
        orbital_count == state.occupations.len()
            && orbital_count == state.scf.large_components.ncols()
            && orbital_count == state.scf.small_components.ncols(),
        "ATOM spin-density potential {potential_index} orbital shapes disagree: xmag={}, occupations={}, large={:?}, small={:?}",
        orbital_count,
        state.occupations.len(),
        state.scf.large_components.dim(),
        state.scf.small_components.dim()
    );
    ensure!(
        radial_count >= ATOM_RADIAL_POINTS
            && state.scf.large_components.nrows() == radial_count
            && state.scf.small_components.nrows() == radial_count,
        "ATOM spin-density potential {potential_index} radial shapes disagree: large={:?}, small={:?}, radii={}",
        state.scf.large_components.dim(),
        state.scf.small_components.dim(),
        radial_count
    );

    let mut moment = 0.0;
    for (orbital, &weight) in state.spin_magnetization.iter().enumerate() {
        ensure!(
            weight.is_finite(),
            "ATOM spin-density potential {potential_index} xmag for orbital {} is non-finite",
            orbital + 1
        );
        moment += weight;
    }
    ensure!(
        moment.is_finite(),
        "ATOM spin-density potential {potential_index} moment is non-finite"
    );
    let normalization = if moment > 0.0 { moment } else { 1.0 };

    let mut native_spin_density = Array1::<f64>::zeros(radial_count);
    let mut previous_radius = 0.0;
    for row in 0..radial_count {
        let radius = state.initial_orbitals.radii[row];
        ensure!(
            radius.is_finite() && radius > previous_radius,
            "ATOM spin-density potential {potential_index} radius row {row} must be positive, finite, and strictly increasing, got {radius} after {previous_radius}"
        );
        previous_radius = radius;
        let mut weighted_probability = 0.0;
        for orbital in 0..orbital_count {
            let large = state.scf.large_components[(row, orbital)];
            let small = state.scf.small_components[(row, orbital)];
            ensure!(
                large.is_finite() && small.is_finite(),
                "ATOM spin-density potential {potential_index} orbital {} row {row} has non-finite Dirac components",
                orbital + 1
            );
            weighted_probability +=
                state.spin_magnetization[orbital] * (large * large + small * small);
        }
        let density = weighted_probability / normalization / (radius * radius);
        ensure!(
            density.is_finite(),
            "ATOM spin-density potential {potential_index} row {row} is non-finite"
        );
        native_spin_density[row] = density;
    }

    atomic_output_bound_quantity(state, native_spin_density.view())
}

#[allow(dead_code)]
fn atomic_apot_normalize_density_for_norman_radius(
    atomic_number: usize,
    density: &mut Array1<f64>,
) -> Result<()> {
    match norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number,
    }) {
        Ok(_) => Ok(()),
        Err(GridError::InsufficientNormanCharge { charge_found, .. }) => {
            let target_charge = atomic_number as f64;
            let deficit = target_charge - charge_found;
            ensure!(
                charge_found > 0.0
                    && deficit > 0.0
                    && deficit <= target_charge * ATOM_APOT_NORMAN_CHARGE_REL_TOLERANCE,
                "ATOM generated density integrates to {charge_found:.12e} electron(s), too far below Z={atomic_number} for Norman-radius normalization"
            );
            let scale = target_charge * (1.0 + ATOM_APOT_NORMAN_CHARGE_SCALE_PAD) / charge_found;
            for value in density.iter_mut() {
                *value *= scale;
            }
            norman_radius_from_density(NormanRadiusInput {
                overlapped_density: density.view(),
                atomic_number,
            })
            .context("ATOM normalized generated density still cannot determine Norman radius")?;
            Ok(())
        }
        Err(error) => Err(error).context("failed to validate ATOM generated density"),
    }
}

#[allow(dead_code)]
fn atomic_apot_has_geometry_overlap_neighbor(
    atom_potentials: &Array1<usize>,
    atom_positions: &Array2<f64>,
    representative_atoms: &Array1<usize>,
    potential_index: usize,
) -> Result<bool> {
    ensure!(
        potential_index < representative_atoms.len(),
        "ATOM representative atom list has {} value(s), missing potential {potential_index}",
        representative_atoms.len()
    );
    ensure!(
        atom_potentials.len() == atom_positions.nrows(),
        "ATOM atom potential count {} does not match atom position row count {}",
        atom_potentials.len(),
        atom_positions.nrows()
    );
    ensure!(
        atom_positions.ncols() == 3,
        "ATOM atom position table has {} column(s), expected 3",
        atom_positions.ncols()
    );
    let representative = representative_atoms[potential_index];
    ensure!(
        representative < atom_positions.nrows(),
        "ATOM representative atom {representative} for potential {potential_index} exceeds atom count {}",
        atom_positions.nrows()
    );
    for atom in 0..atom_positions.nrows() {
        if atom == representative {
            continue;
        }
        let dx = atom_positions[(atom, 0)] - atom_positions[(representative, 0)];
        let dy = atom_positions[(atom, 1)] - atom_positions[(representative, 1)];
        let dz = atom_positions[(atom, 2)] - atom_positions[(representative, 2)];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        if distance <= ATOM_APOT_GEOMETRY_OVERLAP_CUTOFF {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(dead_code)]
fn atomic_apot_energy_scalars_from_states(
    input: &PotInput,
    config_inp: &Path,
    states: &[AtomicScfState],
    overlap_arrays: &AtomicApotOverlapArrays,
) -> Result<AtomicApotEnergyScalars> {
    let state_count = apot_state_count(input)?;
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    let hole_index = checked_hole_index(input.control.ihole)?;
    let (initial_column, final_column) = atomic_apot_absorber_state_columns(input, state_count)?;
    let initial_total =
        atomic_total_energy_from_state(input, initial_column, &states[initial_column])
            .context("failed to generate ATOM initial-state total energy")?;

    if hole_index == 0 {
        return Ok(AtomicApotEnergyScalars {
            initial_total_energy: initial_total.total,
            final_total_energy: initial_total.total,
            frozen_orbital_energy: 0.0,
            relaxation_energy: 0.0,
            edge_energy: 0.0,
        });
    }

    let final_total = atomic_total_energy_from_state(input, final_column, &states[final_column])
        .context("failed to generate ATOM final-state total energy")?;
    let frozen_orbital_energy =
        atomic_apot_frozen_orbital_energy(input, config_inp, &states[initial_column])?;
    let adiabatic_edge = final_total.total - initial_total.total;
    let relaxation_energy = -frozen_orbital_energy - adiabatic_edge;
    let mut edge_energy = if adiabatic_edge <= 0.0 {
        -frozen_orbital_energy
    } else {
        adiabatic_edge
    };

    ensure!(
        states[0].scf.coulomb_potential.len() >= ATOM_RADIAL_POINTS,
        "ATOM state 0 Coulomb potential has {} radial point(s), expected {ATOM_RADIAL_POINTS}",
        states[0].scf.coulomb_potential.len()
    );
    ensure!(
        overlap_arrays.overlapped_coulomb_potential.nrows() >= ATOM_RADIAL_POINTS
            && overlap_arrays.overlapped_coulomb_potential.ncols() >= 1,
        "ATOM overlapped Coulomb potential shape {:?} cannot provide vclap(1,0)",
        overlap_arrays.overlapped_coulomb_potential.dim()
    );
    let free_coulomb = atomic_output_coulomb(&states[0])?;
    edge_energy += free_coulomb[0] - overlap_arrays.overlapped_coulomb_potential[(0, 0)];

    Ok(AtomicApotEnergyScalars {
        initial_total_energy: initial_total.total,
        final_total_energy: final_total.total,
        frozen_orbital_energy,
        relaxation_energy,
        edge_energy,
    })
}

#[allow(dead_code)]
fn atomic_total_energy_from_state(
    input: &PotInput,
    column: usize,
    state: &AtomicScfState,
) -> Result<refeff_core::AtomicTotalEnergy> {
    let state_count = apot_state_count(input)?;
    ensure!(
        column < state_count,
        "ATOM total-energy column {column} exceeds apot state count {state_count}"
    );
    let orbital_count = state.kappas.len();
    ensure!(
        orbital_count > 0,
        "ATOM source total-energy state column {column} has no orbitals"
    );
    ensure!(
        state.occupations.len() >= orbital_count
            && state.scf.orbital_energies.len() >= orbital_count
            && state.scf.active_lengths.len() >= orbital_count,
        "ATOM source total-energy state column {column} has orbital_count={orbital_count} but occupation len={}, energy len={}, active length len={}",
        state.occupations.len(),
        state.scf.orbital_energies.len(),
        state.scf.active_lengths.len()
    );
    ensure!(
        state.scf.large_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.small_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.large_components.ncols() >= orbital_count
            && state.scf.small_components.ncols() >= orbital_count,
        "ATOM source total-energy state column {column} component shapes large={:?}, small={:?}, expected at least {ATOM_RADIAL_POINTS}x{orbital_count}",
        state.scf.large_components.dim(),
        state.scf.small_components.dim()
    );
    ensure!(
        state.scf.large_coefficients.ncols() >= orbital_count
            && state.scf.small_coefficients.ncols() >= orbital_count,
        "ATOM source total-energy state column {column} coefficient shapes large={:?}, small={:?}, expected one column per orbital",
        state.scf.large_coefficients.dim(),
        state.scf.small_coefficients.dim()
    );

    let coefficient_count = state.scf.large_coefficients.nrows();
    ensure!(
        state.scf.small_coefficients.nrows() == coefficient_count,
        "ATOM source total-energy state column {column} coefficient row mismatch: large={}, small={}",
        coefficient_count,
        state.scf.small_coefficients.nrows()
    );

    let kappas = state
        .kappas
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let occupations = state
        .occupations
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let valence_occupations = atomic_total_energy_valence_occupations(orbital_count);
    let orbital_energies = state
        .scf
        .orbital_energies
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let active_lengths = state
        .scf
        .active_lengths
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let orbital_powers = state.initial_orbitals.orbital_powers.to_vec();
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })
    .context("failed to generate ATOM total-energy Coulomb coefficients from state")?;
    let radii = &state.initial_orbitals.radii;
    let large_components =
        Array2::from_shape_fn((ATOM_RADIAL_POINTS, orbital_count), |(row, col)| {
            state.scf.large_components[(row, col)]
        });
    let small_components =
        Array2::from_shape_fn((ATOM_RADIAL_POINTS, orbital_count), |(row, col)| {
            state.scf.small_components[(row, col)]
        });
    let large_coefficients =
        Array2::from_shape_fn((coefficient_count, orbital_count), |(row, col)| {
            state.scf.large_coefficients[(row, col)]
        });
    let small_coefficients =
        Array2::from_shape_fn((coefficient_count, orbital_count), |(row, col)| {
            state.scf.small_coefficients[(row, col)]
        });

    atomic_total_energy_from_radials(AtomicTotalEnergyRadialInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
        orbital_energies: &orbital_energies,
        coulomb_coefficients: coulomb_coefficients.view(),
        large_small: false,
        step: ATOM_RADIAL_STEP,
        radii: radii.view(),
        active_lengths: &active_lengths,
        orbital_powers: &orbital_powers,
        large_components: large_components.view(),
        small_components: small_components.view(),
        large_coefficients: large_coefficients.view(),
        small_coefficients: small_coefficients.view(),
    })
    .context("failed to generate ATOM total energy from state")
}

#[allow(dead_code)]
fn atomic_apot_frozen_orbital_energy(
    input: &PotInput,
    config_inp: &Path,
    initial_state: &AtomicScfState,
) -> Result<f64> {
    let hole_index = checked_hole_index(input.control.ihole)?;
    ensure!(
        hole_index > 0,
        "ATOM frozen orbital energy requires a positive ihole"
    );
    let state_count = apot_state_count(input)?;
    let (initial_column, _) = atomic_apot_absorber_state_columns(input, state_count)?;
    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );
    let initial_configuration = &configurations[initial_column];
    let frozen_orbital = compact_orbital_for_feff_slot(initial_configuration, hole_index)?
        .with_context(|| format!("ATOM ihole {hole_index} did not resolve to a compact orbital"))?;
    ensure!(
        frozen_orbital < initial_state.scf.orbital_energies.len(),
        "ATOM frozen orbital {} exceeds initial-state energy count {}",
        frozen_orbital + 1,
        initial_state.scf.orbital_energies.len()
    );
    Ok(initial_state.scf.orbital_energies[frozen_orbital])
}

#[allow(dead_code)]
fn atomic_apot_ensure_overlap_state(potential_index: usize, state: &AtomicScfState) -> Result<()> {
    ensure_atomic_radial_state("overlap", state)?;
    ensure!(
        state.scf.coulomb_potential.len() >= ATOM_RADIAL_POINTS,
        "ATOM overlap potential {potential_index} Coulomb potential has {} radial point(s), expected {ATOM_RADIAL_POINTS}",
        state.scf.coulomb_potential.len()
    );
    Ok(())
}

#[allow(dead_code)]
fn atomic_apot_usize_atomic_numbers(
    static_arrays: &AtomicApotStaticArrays,
    unique_count: usize,
) -> Result<Array1<usize>> {
    ensure!(
        static_arrays.atomic_numbers.len() == unique_count,
        "ATOM static APOT iz has {} value(s), expected {unique_count}",
        static_arrays.atomic_numbers.len()
    );
    let mut values = Array1::<usize>::zeros(unique_count);
    for (potential_index, &atomic_number) in static_arrays.atomic_numbers.iter().enumerate() {
        ensure!(
            atomic_number > 0,
            "ATOM static APOT iz for potential {potential_index} must be positive, got {atomic_number}"
        );
        values[potential_index] = usize::try_from(atomic_number)
            .context("ATOM static APOT iz cannot be represented as usize")?;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_usize_potential_indices(
    name: &'static str,
    values: &Array1<i64>,
    unique_count: usize,
) -> Result<Array1<usize>> {
    let mut converted = Array1::<usize>::zeros(values.len());
    for (index, &value) in values.iter().enumerate() {
        ensure!(
            value >= 0,
            "ATOM static APOT {name}[{index}] has negative potential index {value}"
        );
        let potential = usize::try_from(value)
            .with_context(|| format!("ATOM static APOT {name}[{index}] exceeds usize range"))?;
        ensure!(
            potential < unique_count,
            "ATOM static APOT {name}[{index}] has potential {potential}, expected <= {}",
            unique_count - 1
        );
        converted[index] = potential;
    }
    Ok(converted)
}

#[allow(dead_code)]
fn atomic_apot_zero_based_model_atoms(
    static_arrays: &AtomicApotStaticArrays,
    unique_count: usize,
) -> Result<Array1<usize>> {
    ensure!(
        static_arrays.model_atom_indices.len() == unique_count,
        "ATOM static APOT iatph has {} value(s), expected {unique_count}",
        static_arrays.model_atom_indices.len()
    );
    let mut values = Array1::<usize>::zeros(unique_count);
    for (potential_index, &model_atom) in static_arrays.model_atom_indices.iter().enumerate() {
        ensure!(
            model_atom > 0,
            "ATOM static APOT iatph for potential {potential_index} must be one-based, got {model_atom}"
        );
        let zero_based = usize::try_from(model_atom - 1)
            .context("ATOM static APOT iatph cannot be represented as usize")?;
        ensure!(
            zero_based < static_arrays.atom_count,
            "ATOM static APOT iatph for potential {potential_index} points at atom {model_atom}, but nat={}",
            static_arrays.atom_count
        );
        values[potential_index] = zero_based;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_core_atom_positions(static_arrays: &AtomicApotStaticArrays) -> Result<Array2<f64>> {
    ensure!(
        static_arrays.atom_positions.dim() == (3, static_arrays.atom_count),
        "ATOM static APOT rat has shape {:?}, expected (3,{})",
        static_arrays.atom_positions.dim(),
        static_arrays.atom_count
    );
    let mut positions = Array2::<f64>::zeros((static_arrays.atom_count, 3));
    for atom_index in 0..static_arrays.atom_count {
        for axis in 0..3 {
            let value = static_arrays.atom_positions[(axis, atom_index)];
            ensure!(
                value.is_finite(),
                "ATOM static APOT rat({axis},{}) is non-finite: {value}",
                atom_index + 1
            );
            positions[(atom_index, axis)] = value;
        }
    }
    Ok(positions)
}

#[allow(dead_code)]
fn atomic_apot_explicit_overlap_neighbors(
    static_arrays: &AtomicApotStaticArrays,
    potential_index: usize,
) -> Result<Vec<PotentialOverlapNeighbor>> {
    ensure!(
        potential_index < static_arrays.unique_potential_count,
        "ATOM overlap potential {potential_index} exceeds nph={}",
        static_arrays.unique_potential_count - 1
    );
    ensure!(
        static_arrays.overlap_shell_counts.len() == static_arrays.unique_potential_count,
        "ATOM static APOT novr has {} value(s), expected {}",
        static_arrays.overlap_shell_counts.len(),
        static_arrays.unique_potential_count
    );
    ensure!(
        static_arrays.overlap_potential_indices.ncols() == static_arrays.unique_potential_count
            && static_arrays.overlap_shell_atom_counts.ncols()
                == static_arrays.unique_potential_count
            && static_arrays.overlap_radii.ncols() == static_arrays.unique_potential_count,
        "ATOM static APOT overlap tables must have one column per unique potential"
    );

    let count = static_arrays.overlap_shell_counts[potential_index];
    ensure!(
        count >= 0,
        "ATOM static APOT novr for potential {potential_index} is negative: {count}"
    );
    let count = usize::try_from(count).context("ATOM static APOT novr exceeds usize range")?;
    ensure!(
        count <= static_arrays.overlap_potential_indices.nrows()
            && count <= static_arrays.overlap_shell_atom_counts.nrows()
            && count <= static_arrays.overlap_radii.nrows(),
        "ATOM static APOT novr for potential {potential_index} has {count} shell(s), but overlap table row counts are iphovr={}, nnovr={}, rovr={}",
        static_arrays.overlap_potential_indices.nrows(),
        static_arrays.overlap_shell_atom_counts.nrows(),
        static_arrays.overlap_radii.nrows()
    );

    let mut neighbors = Vec::with_capacity(count);
    for shell_index in 0..count {
        let source = static_arrays.overlap_potential_indices[(shell_index, potential_index)];
        ensure!(
            source >= 0,
            "ATOM static APOT iphovr({shell_index},{potential_index}) is negative: {source}"
        );
        let source = usize::try_from(source)
            .context("ATOM static APOT iphovr cannot be represented as usize")?;
        ensure!(
            source < static_arrays.unique_potential_count,
            "ATOM static APOT iphovr({shell_index},{potential_index}) has potential {source}, expected <= {}",
            static_arrays.unique_potential_count - 1
        );
        let multiplicity = static_arrays.overlap_shell_atom_counts[(shell_index, potential_index)];
        ensure!(
            multiplicity >= 0,
            "ATOM static APOT nnovr({shell_index},{potential_index}) is negative: {multiplicity}"
        );
        let distance = static_arrays.overlap_radii[(shell_index, potential_index)];
        ensure!(
            distance.is_finite() && distance > 0.0,
            "ATOM static APOT rovr({shell_index},{potential_index}) must be positive and finite, got {distance}"
        );
        neighbors.push(PotentialOverlapNeighbor {
            source_potential: source,
            multiplicity: multiplicity as f64,
            distance,
        });
    }
    Ok(neighbors)
}

#[allow(dead_code)]
fn atomic_apot_static_arrays_from_handoffs(
    input: &PotInput,
    geom: &GeomDat,
    pot: &PotBinData,
) -> Result<AtomicApotStaticArrays> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        input.potentials.len() == unique_count,
        "ATOM pot.inp has {} potential row(s), expected {unique_count} from nph={}",
        input.potentials.len(),
        input.control.nph
    );
    ensure!(
        input.overlap_shells.len() == unique_count,
        "ATOM pot.inp has {} overlap-shell group(s), expected {unique_count}",
        input.overlap_shells.len()
    );
    ensure!(
        geom.nph + 1 == unique_count,
        "ATOM geom.dat nph={} does not match pot.inp nph={}",
        geom.nph,
        input.control.nph
    );
    ensure!(
        geom.nat == geom.atoms.len(),
        "ATOM geom.dat nat {} does not match row count {}",
        geom.nat,
        geom.atoms.len()
    );
    ensure!(
        geom.model_atoms.len() == unique_count,
        "ATOM geom.dat has {} model atom(s), expected {unique_count}",
        geom.model_atoms.len()
    );
    ensure!(
        pot.potential_count() == unique_count,
        "ATOM pot.bin has {} potential row(s), expected {unique_count}",
        pot.potential_count()
    );

    let atomic_numbers = atomic_apot_atomic_numbers(input, pot, unique_count)?;
    let model_atom_indices = atomic_apot_model_atom_indices(geom, unique_count)?;
    let norman_radii = atomic_apot_norman_radii(pot, unique_count)?;
    let atom_potential_indices = atomic_apot_atom_potential_indices(geom, unique_count)?;
    let atom_positions = atomic_apot_atom_positions(geom)?;
    let (overlap_shell_counts, overlap_potential_indices, overlap_shell_atom_counts, overlap_radii) =
        atomic_apot_overlap_shell_arrays(input, unique_count)?;

    Ok(AtomicApotStaticArrays {
        unique_potential_count: unique_count,
        atom_count: geom.nat,
        atomic_numbers,
        model_atom_indices,
        overlap_shell_counts,
        norman_radii,
        atom_potential_indices,
        atom_positions,
        overlap_potential_indices,
        overlap_shell_atom_counts,
        overlap_radii,
    })
}

#[allow(dead_code)]
fn atomic_apot_static_arrays_from_source_geometry(
    input: &PotInput,
    geom: &GeomDat,
    norman_radii: Array1<f64>,
) -> Result<AtomicApotStaticArrays> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        input.potentials.len() == unique_count,
        "ATOM pot.inp has {} potential row(s), expected {unique_count} from nph={}",
        input.potentials.len(),
        input.control.nph
    );
    ensure!(
        input.overlap_shells.len() == unique_count,
        "ATOM pot.inp has {} overlap-shell group(s), expected {unique_count}",
        input.overlap_shells.len()
    );
    ensure!(
        geom.nph + 1 == unique_count,
        "ATOM geom.dat nph={} does not match pot.inp nph={}",
        geom.nph,
        input.control.nph
    );
    ensure!(
        geom.nat == geom.atoms.len(),
        "ATOM geom.dat nat {} does not match row count {}",
        geom.nat,
        geom.atoms.len()
    );
    ensure!(
        geom.model_atoms.len() == unique_count,
        "ATOM geom.dat has {} model atom(s), expected {unique_count}",
        geom.model_atoms.len()
    );
    ensure!(
        norman_radii.len() == unique_count,
        "ATOM source geometry has {} Norman radius value(s), expected {unique_count}",
        norman_radii.len()
    );
    for (potential_index, &radius) in norman_radii.iter().enumerate() {
        ensure!(
            radius.is_finite() && radius > 0.0,
            "ATOM source geometry Norman radius for potential {potential_index} must be positive and finite, got {radius}"
        );
    }

    let atomic_numbers = atomic_apot_input_atomic_numbers(input, unique_count)?;
    let model_atom_indices = atomic_apot_model_atom_indices(geom, unique_count)?;
    let atom_potential_indices = atomic_apot_atom_potential_indices(geom, unique_count)?;
    let atom_positions = atomic_apot_atom_positions(geom)?;
    let (overlap_shell_counts, overlap_potential_indices, overlap_shell_atom_counts, overlap_radii) =
        atomic_apot_overlap_shell_arrays(input, unique_count)?;

    Ok(AtomicApotStaticArrays {
        unique_potential_count: unique_count,
        atom_count: geom.nat,
        atomic_numbers,
        model_atom_indices,
        overlap_shell_counts,
        norman_radii,
        atom_potential_indices,
        atom_positions,
        overlap_potential_indices,
        overlap_shell_atom_counts,
        overlap_radii,
    })
}

#[allow(dead_code)]
fn atomic_apot_input_atomic_numbers(input: &PotInput, unique_count: usize) -> Result<Array1<i64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "ATOM pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut values = Array1::<i64>::zeros(unique_count);
    for potential_index in 0..unique_count {
        values[potential_index] =
            i64::try_from(checked_atomic_number(input.potentials[potential_index].z)?)
                .context("ATOM atomic number cannot be represented as i64")?;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_atomic_numbers(
    input: &PotInput,
    pot: &PotBinData,
    unique_count: usize,
) -> Result<Array1<i64>> {
    ensure!(
        pot.atomic_numbers.len() == unique_count,
        "ATOM pot.bin has {} atomic number(s), expected {unique_count}",
        pot.atomic_numbers.len()
    );
    let mut values = Array1::<i64>::zeros(unique_count);
    for potential_index in 0..unique_count {
        let input_atomic_number = checked_atomic_number(input.potentials[potential_index].z)?;
        let pot_atomic_number = pot.atomic_numbers[potential_index];
        ensure!(
            pot_atomic_number == input_atomic_number,
            "ATOM pot.bin atomic number {pot_atomic_number} for potential {potential_index} does not match pot.inp {input_atomic_number}"
        );
        values[potential_index] = i64::try_from(input_atomic_number)
            .context("ATOM atomic number cannot be represented as i64")?;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_model_atom_indices(geom: &GeomDat, unique_count: usize) -> Result<Array1<i64>> {
    let mut values = Array1::<i64>::zeros(unique_count);
    for (potential_index, &model_atom) in geom.model_atoms.iter().enumerate() {
        ensure!(
            model_atom > 0 && model_atom <= geom.nat,
            "ATOM geom.dat model atom {model_atom} for potential {potential_index} is outside 1..={}",
            geom.nat
        );
        values[potential_index] = i64::try_from(model_atom)
            .context("ATOM model atom index cannot be represented as i64")?;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_norman_radii(pot: &PotBinData, unique_count: usize) -> Result<Array1<f64>> {
    ensure!(
        pot.norman_radii.len() == unique_count,
        "ATOM pot.bin has {} Norman radius value(s), expected {unique_count}",
        pot.norman_radii.len()
    );
    for (potential_index, &radius) in pot.norman_radii.iter().enumerate() {
        ensure!(
            radius.is_finite() && radius > 0.0,
            "ATOM pot.bin Norman radius for potential {potential_index} must be positive and finite, got {radius}"
        );
    }
    Ok(pot.norman_radii.clone())
}

#[allow(dead_code)]
fn atomic_apot_atom_potential_indices(geom: &GeomDat, unique_count: usize) -> Result<Array1<i64>> {
    let mut values = Array1::<i64>::zeros(geom.nat);
    for (atom_index, atom) in geom.atoms.iter().enumerate() {
        ensure!(
            atom.iph >= 0,
            "ATOM geom.dat atom {} has negative potential index {}",
            atom_index + 1,
            atom.iph
        );
        let potential = usize::try_from(atom.iph)
            .context("ATOM atom potential index cannot be represented as usize")?;
        ensure!(
            potential < unique_count,
            "ATOM geom.dat atom {} potential {potential} exceeds nph={}",
            atom_index + 1,
            unique_count - 1
        );
        values[atom_index] =
            i64::try_from(potential).context("ATOM atom potential index exceeds i64 range")?;
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_apot_atom_positions(geom: &GeomDat) -> Result<Array2<f64>> {
    let mut positions = Array2::<f64>::zeros((3, geom.nat));
    for (atom_index, atom) in geom.atoms.iter().enumerate() {
        let coordinates = [atom.x, atom.y, atom.z];
        for (axis, coordinate) in coordinates.into_iter().enumerate() {
            ensure!(
                coordinate.is_finite(),
                "ATOM geom.dat atom {} coordinate {axis} is non-finite: {coordinate}",
                atom_index + 1
            );
            let converted = coordinate / FEFF_BOHR_ANGSTROM;
            ensure!(
                converted.is_finite(),
                "ATOM geom.dat atom {} coordinate {axis} conversion produced a non-finite value",
                atom_index + 1
            );
            positions[(axis, atom_index)] = converted;
        }
    }
    Ok(positions)
}

type AtomicApotOverlapShellArrays = (Array1<i64>, Array2<i64>, Array2<i64>, Array2<f64>);

#[allow(dead_code)]
fn atomic_apot_overlap_shell_arrays(
    input: &PotInput,
    unique_count: usize,
) -> Result<AtomicApotOverlapShellArrays> {
    let row_count = input
        .overlap_shells
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut shell_counts = Array1::<i64>::zeros(unique_count);
    let mut potential_indices = Array2::<i64>::zeros((row_count, unique_count));
    let mut atom_counts = Array2::<i64>::zeros((row_count, unique_count));
    let mut radii = Array2::<f64>::zeros((row_count, unique_count));

    for (potential_index, shells) in input.overlap_shells.iter().enumerate() {
        shell_counts[potential_index] = i64::try_from(shells.len())
            .context("ATOM overlap shell count cannot be represented as i64")?;
        for (shell_index, shell) in shells.iter().enumerate() {
            ensure!(
                shell.iphovr >= 0,
                "ATOM overlap shell {shell_index} for potential {potential_index} has negative iphovr {}",
                shell.iphovr
            );
            let neighbor_potential = usize::try_from(shell.iphovr)
                .context("ATOM overlap iphovr cannot be represented as usize")?;
            ensure!(
                neighbor_potential < unique_count,
                "ATOM overlap shell {shell_index} for potential {potential_index} has iphovr {neighbor_potential}, expected <= {}",
                unique_count - 1
            );
            ensure!(
                shell.nnovr >= 0,
                "ATOM overlap shell {shell_index} for potential {potential_index} has negative nnovr {}",
                shell.nnovr
            );
            ensure!(
                shell.rovr.is_finite() && shell.rovr >= 0.0,
                "ATOM overlap shell {shell_index} for potential {potential_index} has invalid rovr {}",
                shell.rovr
            );

            potential_indices[(shell_index, potential_index)] = i64::from(shell.iphovr);
            atom_counts[(shell_index, potential_index)] = i64::from(shell.nnovr);
            radii[(shell_index, potential_index)] = shell.rovr / FEFF_BOHR_ANGSTROM;
        }
    }

    Ok((shell_counts, potential_indices, atom_counts, radii))
}

#[allow(dead_code)]
fn ionicity_for_apot_state(input: &PotInput, column: usize) -> Result<f64> {
    if let Some(potential) = input.potentials.get(column) {
        return Ok(potential.xion);
    }
    input
        .potentials
        .first()
        .map(|potential| potential.xion)
        .context("ATOM pot.inp has no absorber potential row")
}

fn effective_ionicity_for_apot_state(
    input: &PotInput,
    column: usize,
    configuration: &OrbitalConfiguration,
) -> Result<f64> {
    let requested = ionicity_for_apot_state(input, column)?;
    if !(input.warn_ion && input.config_type == 2) {
        return Ok(requested);
    }

    let atomic_number = atomic_number_for_apot_state(input, column)?;
    let electron_count = configuration.electron_counts.iter().copied().sum::<f64>();
    let effective = atomic_number as f64 - electron_count;
    ensure!(
        effective.is_finite(),
        "ATOM WARNION custom configuration for state {column} produced non-finite ionicity"
    );
    Ok(effective)
}

// FixAtomicQuantities interpolates even point nuclei: wfirdf's grid origin
// and COMMON/xx's default-real constants differ before promotion to real*8.
fn atomic_output_quantity(
    state: &AtomicScfState,
    values: ArrayView1<'_, f64>,
) -> Result<Array1<f64>> {
    let source_x = state
        .initial_orbitals
        .radii
        .iter()
        .map(|r| r.ln())
        .collect::<Vec<_>>();
    let values = values.to_vec();
    (0..ATOM_RADIAL_POINTS)
        .map(|row| {
            let target_x = -f64::from(8.8_f32) + row as f64 * f64::from(0.05_f32);
            Ok(terp(&source_x, &values, 3, target_x)?.value)
        })
        .collect()
}

// The finite-nucleus mesh can end before the output mesh. Continuing a
// decaying bound quantity with a cubic polynomial can create a negative density
// at large radii. Continue its terminal exponential decay outside that mesh.
fn atomic_output_bound_quantity(
    state: &AtomicScfState,
    values: ArrayView1<'_, f64>,
) -> Result<Array1<f64>> {
    let mut output = atomic_output_quantity(state, values)?;
    if state.initial_orbitals.nucleus_index <= 1 {
        return Ok(output);
    }
    let radii = &state.initial_orbitals.radii;
    let last = radii.len() - 1;
    let terminal = values[last];
    let previous = values[last - 1];
    let rate = if terminal != 0.0 && previous != 0.0 && terminal.signum() == previous.signum() {
        (terminal / previous).ln() / (radii[last] - radii[last - 1])
    } else {
        f64::NEG_INFINITY
    };
    for (row, value) in output.iter_mut().enumerate() {
        let radius = (-f64::from(8.8_f32) + row as f64 * f64::from(0.05_f32)).exp();
        if radius > radii[last] {
            *value = if rate.is_finite() && rate < 0.0 {
                terminal * (rate * (radius - radii[last])).exp()
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn atomic_output_coulomb(state: &AtomicScfState) -> Result<Array1<f64>> {
    let mut output = atomic_output_quantity(state, state.scf.coulomb_potential.view())?;
    if state.initial_orbitals.nucleus_index > 1 {
        let last = state.initial_orbitals.radii.len() - 1;
        let last_radius = state.initial_orbitals.radii[last];
        let charge = state.scf.coulomb_potential[last] * last_radius;
        for (row, value) in output.iter_mut().enumerate() {
            let radius = (-f64::from(8.8_f32) + row as f64 * f64::from(0.05_f32)).exp();
            if radius > last_radius {
                *value = charge / radius;
            }
        }
    }
    Ok(output)
}

fn atomic_output_state(state: &AtomicScfState) -> Result<AtomicScfState> {
    let mut output = state.clone();
    output.scf.density_4pi = atomic_output_bound_quantity(state, state.scf.density_4pi.view())?;
    output.scf.valence_density_4pi =
        atomic_output_bound_quantity(state, state.scf.valence_density_4pi.view())?;
    output.scf.coulomb_potential = atomic_output_coulomb(state)?;
    for orbital in 0..state.kappas.len() {
        output
            .scf
            .large_components
            .column_mut(orbital)
            .assign(&atomic_output_bound_quantity(
                state,
                state.scf.large_components.column(orbital),
            )?);
        output
            .scf
            .small_components
            .column_mut(orbital)
            .assign(&atomic_output_bound_quantity(
                state,
                state.scf.small_components.column(orbital),
            )?);
    }
    Ok(output)
}

#[allow(dead_code)]
fn atomic_first_radius_times_charge(atomic_number: usize) -> f64 {
    // ATOM/wfirdf.f90 uses default-real arithmetic for `nz*exp(-8.8)`
    // before assigning to double-precision dr1. Keep both the exponential
    // and multiplication in single precision to reproduce its radial mesh.
    f64::from(atomic_number as f32 * (-8.8_f32).exp())
}

#[allow(dead_code)]
fn atomic_nucleus_request_index(input: &PotInput) -> isize {
    if input.finite_nucleus {
        ATOM_FINITE_NUCLEUS_REQUEST_INDEX
    } else {
        ATOM_POINT_NUCLEUS_REQUEST_INDEX
    }
}

fn apot_has_section(apot: &ApotBinData, section_number: usize) -> bool {
    apot.sections
        .iter()
        .any(|section| section.section_number == section_number)
}

fn generated_fpf0_dat(
    apot: &ApotBinData,
    input: &PotInput,
    config_inp: &Path,
) -> Result<Fpf0DatData> {
    let state_count = apot_state_count(input)?;
    let metadata_column = fpf0_initial_absorber_column(input)?;
    ensure!(
        metadata_column < state_count,
        "ATOM fpf0 absorber column {metadata_column} exceeds apot state count {state_count}"
    );
    let component_column = 0_usize;

    let atomic_number = checked_atomic_number(input.potentials[0].z)?;
    let total_energy = fpf0_total_energy(apot, input, config_inp)?;
    let norb = fpf0_norb(apot, metadata_column)?;
    let radii = Array1::from_shape_fn(ATOM_RADIAL_POINTS, |row| {
        refeff_core::loucks_radius(row + 1)
    });
    let core_hole = apot_core_hole_columns(apot).context("ATOM apot.bin core-hole payload")?;
    let density_4pi = real_matrix_column(
        real_matrix_section(apot, ATOM_FPF0_DENSITY_SECTION_NUMBER, "rho")?,
        metadata_column,
        ATOM_RADIAL_POINTS,
        "rho",
    )?;
    let orbital_energies = real_matrix_column_prefix(
        real_matrix_section(apot, ATOM_FPF0_EORB_SECTION_NUMBER, "eorb")?,
        metadata_column,
        norb,
        "eorb",
    )?;
    let kappas = int_matrix_column_prefix(
        int_matrix_section(apot, ATOM_FPF0_KAPPA_SECTION_NUMBER, "kappa")?,
        metadata_column,
        norb,
        "kappa",
    )?;
    let large_components = real_matrix_prefix(
        real_matrix_section(apot, ATOM_FPF0_DGC_SECTION_START + component_column, "dgc")?,
        norb,
        "dgc",
    )?;
    let small_components = real_matrix_prefix(
        real_matrix_section(
            apot,
            ATOM_FPF0_DGC_SECTION_START + state_count + component_column,
            "dpc",
        )?,
        norb,
        "dpc",
    )?;
    let occupations = fpf0_initial_absorber_occupations(input, config_inp, norb)?;
    let hole = checked_hole_index(input.control.ihole)?;

    let form_factor = atomic_form_factor(AtomicFormFactorInput {
        atomic_number,
        hole_orbital_1based: hole,
        radial_step: ATOM_RADIAL_STEP,
        total_energy,
        radii: radii.view(),
        density_4pi: density_4pi.view(),
        initial_large_component: core_hole.large_component.view(),
        initial_small_component: core_hole.small_component.view(),
        large_components: large_components.view(),
        small_components: small_components.view(),
        occupations: &occupations,
        orbital_energies: &orbital_energies,
        kappas: &kappas,
    })
    .context("failed to generate ATOM fpf0.dat from apot.bin")?;
    fpf0_dat_from_form_factor(form_factor)
}

fn fpf0_total_energy(apot: &ApotBinData, input: &PotInput, config_inp: &Path) -> Result<f64> {
    Ok(generated_atomic_total_energy(apot, input, config_inp)?.total)
}

fn generated_atomic_total_energy(
    apot: &ApotBinData,
    input: &PotInput,
    config_inp: &Path,
) -> Result<refeff_core::AtomicTotalEnergy> {
    let state_count = apot_state_count(input)?;
    let column = fpf0_total_energy_column(input, state_count)?;
    generated_atomic_total_energy_for_column(apot, input, config_inp, column)
}

fn generated_atomic_total_energy_for_column(
    apot: &ApotBinData,
    input: &PotInput,
    config_inp: &Path,
    column: usize,
) -> Result<refeff_core::AtomicTotalEnergy> {
    let state_count = apot_state_count(input)?;
    ensure!(
        column < state_count,
        "ATOM total-energy column {column} exceeds apot state count {state_count}"
    );
    let norb = fpf0_norb(apot, column)?;

    let recipe = configuration_recipe(input.config_type);
    let source = ConfigurationRowsSource::new(input, recipe, config_inp)?;
    let configuration = atomic_total_energy_configuration(input, &source, column)?;
    ensure!(
        configuration.orbital_count >= norb,
        "ATOM total-energy configuration has {} orbital(s), but apot.bin column {column} has {norb}",
        configuration.orbital_count
    );

    let kappas = int_matrix_column_prefix(
        int_matrix_section(apot, ATOM_FPF0_KAPPA_SECTION_NUMBER, "kappa")?,
        column,
        norb,
        "kappa",
    )?;
    let occupations = configuration
        .electron_counts
        .iter()
        .take(norb)
        .copied()
        .collect::<Vec<_>>();
    let valence_occupations = atomic_total_energy_valence_occupations(norb);
    let orbital_energies = real_matrix_column_prefix(
        real_matrix_section(apot, ATOM_FPF0_EORB_SECTION_NUMBER, "eorb")?,
        column,
        norb,
        "eorb",
    )?;
    let large_components = real_matrix_prefix(
        real_matrix_section(apot, ATOM_FPF0_DGC_SECTION_START + column, "dgc")?,
        norb,
        "dgc",
    )?;
    let small_components = real_matrix_prefix(
        real_matrix_section(
            apot,
            ATOM_FPF0_DGC_SECTION_START + state_count + column,
            "dpc",
        )?,
        norb,
        "dpc",
    )?;
    let large_coefficients = real_matrix_row_prefix(
        real_matrix_section(
            apot,
            ATOM_FPF0_DGC_SECTION_START + 2 * state_count + column,
            "adgc",
        )?,
        norb,
        "adgc",
    )?;
    let small_coefficients = real_matrix_row_prefix(
        real_matrix_section(
            apot,
            ATOM_FPF0_DGC_SECTION_START + 3 * state_count + column,
            "adpc",
        )?,
        norb,
        "adpc",
    )?;
    let atomic_number = atomic_number_for_apot_state(input, column)?;
    let orbital_powers = atomic_origin_powers(atomic_number, &kappas, input.finite_nucleus)?;
    let active_lengths =
        atomic_active_lengths_from_components(&large_components, &small_components)?;
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })
    .context("failed to generate ATOM total-energy Coulomb coefficients")?;
    // Cached APOT spinors have already passed through FixAtomicQuantities.
    // Integrate them on COMMON/xx's output mesh, not wfirdf's solver mesh.
    let output_step = f64::from(0.05_f32);
    let radii = Array1::from_shape_fn(ATOM_RADIAL_POINTS, |row| {
        (-f64::from(8.8_f32) + row as f64 * output_step).exp()
    });

    atomic_total_energy_from_radials(AtomicTotalEnergyRadialInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
        orbital_energies: &orbital_energies,
        coulomb_coefficients: coulomb_coefficients.view(),
        large_small: false,
        step: output_step,
        radii: radii.view(),
        active_lengths: &active_lengths,
        orbital_powers: &orbital_powers,
        large_components: large_components.view(),
        small_components: small_components.view(),
        large_coefficients: large_coefficients.view(),
        small_coefficients: small_coefficients.view(),
    })
    .context("failed to generate ATOM total energy from apot.bin")
}

fn atomic_total_energy_valence_occupations(norb: usize) -> Vec<f64> {
    // FEFF `scfdat` hard-wires `idfock=1` for the ATOM total-energy pass, so
    // `etotal` receives `xnvalp=0` even though `apot.bin` persists `xnval`.
    vec![0.0; norb]
}

fn atomic_total_energy_configuration(
    input: &PotInput,
    source: &ConfigurationRowsSource,
    column: usize,
) -> Result<OrbitalConfiguration> {
    let unfreeze = input.run.iunf != 0;
    if column < source.potential_count() {
        let hole = if column == 0 && input.run.nohole < 0 {
            checked_hole_index(input.control.ihole)?
        } else {
            0
        };
        return source.compact_configuration(column, hole, input.potentials[column].xion, unfreeze);
    }

    let hole = if input.run.nohole >= 0 {
        checked_hole_index(input.control.ihole)?
    } else {
        0
    };
    source.compact_configuration(0, hole, input.potentials[0].xion, unfreeze)
}

fn fpf0_total_energy_column(input: &PotInput, state_count: usize) -> Result<usize> {
    if input.run.nohole >= 0 {
        return Ok(0);
    }
    state_count
        .checked_sub(1)
        .context("ATOM apot state count must be positive")
}

fn atomic_number_for_apot_state(input: &PotInput, column: usize) -> Result<usize> {
    if let Some(potential) = input.potentials.get(column) {
        return checked_atomic_number(potential.z);
    }
    checked_atomic_number(input.potentials[0].z)
}

fn atomic_origin_powers(
    atomic_number: usize,
    kappas: &[i32],
    finite_nucleus: bool,
) -> Result<Vec<f64>> {
    let charge = atomic_number as f64 / ATOM_TOTAL_ENERGY_SPEED_OF_LIGHT;
    kappas
        .iter()
        .enumerate()
        .map(|(index, &kappa)| {
            let kappa_abs = f64::from(kappa.abs());
            if finite_nucleus {
                return Ok(kappa_abs);
            }
            let radicand = kappa_abs * kappa_abs - charge * charge;
            ensure!(
                radicand > 0.0,
                "ATOM orbital {} has non-positive point-nucleus origin-power radicand {radicand}",
                index + 1
            );
            Ok(radicand.sqrt())
        })
        .collect()
}

fn atomic_active_lengths_from_components(
    large_components: &Array2<f64>,
    small_components: &Array2<f64>,
) -> Result<Vec<usize>> {
    ensure!(
        large_components.dim() == small_components.dim(),
        "ATOM large/small component shape mismatch: {:?} vs {:?}",
        large_components.dim(),
        small_components.dim()
    );
    let (radial_count, orbital_count) = large_components.dim();
    (0..orbital_count)
        .map(|orbital| {
            (0..radial_count)
                .rev()
                .find(|&row| {
                    large_components[(row, orbital)] != 0.0
                        || small_components[(row, orbital)] != 0.0
                })
                .map(|row| {
                    let active_len = row + 1;
                    if active_len % 2 == 0 {
                        active_len - 1
                    } else {
                        active_len
                    }
                })
                .with_context(|| {
                    format!(
                        "ATOM orbital {} has no nonzero radial component rows",
                        orbital + 1
                    )
                })
        })
        .collect()
}

fn apot_state_count(input: &PotInput) -> Result<usize> {
    apot_unique_potential_count(input)?
        .checked_add(1)
        .context("ATOM apot state count overflowed")
}

fn apot_unique_potential_count(input: &PotInput) -> Result<usize> {
    usize::try_from(input.control.nph)
        .context("ATOM nph cannot be represented as usize")?
        .checked_add(1)
        .context("ATOM unique potential count overflowed")
}

fn fpf0_initial_absorber_column(input: &PotInput) -> Result<usize> {
    if input.run.nohole >= 0 {
        return Ok(0);
    }
    usize::try_from(input.control.nph)
        .context("ATOM nph cannot be represented as usize")?
        .checked_add(1)
        .context("ATOM final absorber column overflowed")
}

fn fpf0_norb(apot: &ApotBinData, metadata_column: usize) -> Result<usize> {
    let section = apot_section(apot, ATOM_FPF0_NORB_SECTION_NUMBER, "norb")?;
    let ApotBinPayload::Records(records) = &section.payload else {
        bail!("ATOM apot.bin section 3 norb is not a row-record payload");
    };
    let row = records
        .rows
        .get(metadata_column)
        .with_context(|| format!("ATOM apot.bin section 3 missing norb row {metadata_column}"))?;
    let value = row
        .first()
        .with_context(|| format!("ATOM apot.bin section 3 norb row {metadata_column} is empty"))?;
    let ApotBinValue::Int(value) = value else {
        bail!("ATOM apot.bin section 3 norb row {metadata_column} is not integer-valued");
    };
    ensure!(
        *value > 0,
        "ATOM apot.bin section 3 norb row {metadata_column} must be positive, got {value}"
    );
    usize::try_from(*value).context("ATOM apot.bin norb cannot be represented as usize")
}

fn fpf0_initial_absorber_occupations(
    input: &PotInput,
    config_inp: &Path,
    norb: usize,
) -> Result<Vec<f64>> {
    let recipe = configuration_recipe(input.config_type);
    let source = ConfigurationRowsSource::new(input, recipe, config_inp)?;
    let mut absorber =
        source.compact_configuration(0, 0, input.potentials[0].xion, input.run.iunf != 0)?;
    subtract_valence_from_electron_counts(&mut absorber);
    ensure!(
        absorber.orbital_count >= norb,
        "ATOM compacted absorber has {} orbital(s), but apot.bin fpf0 norb is {norb}",
        absorber.orbital_count
    );
    Ok(absorber
        .electron_counts
        .iter()
        .take(norb)
        .copied()
        .collect())
}

fn fpf0_dat_from_form_factor(form_factor: AtomicFormFactor) -> Result<Fpf0DatData> {
    Ok(Fpf0DatData {
        atomic_number: i32::try_from(form_factor.atomic_number)
            .context("ATOM fpf0 atomic number cannot be represented as i32")?,
        total_energy_fprime: form_factor.total_energy_fprime,
        relativistic_correction: form_factor.relativistic_correction,
        oscillators: form_factor
            .oscillators
            .into_iter()
            .map(|oscillator| Fpf0Oscillator {
                oscillator_strength: oscillator.oscillator_strength,
                excitation_energy: oscillator.excitation_energy,
                orbital_index: oscillator.orbital_index_1based,
            })
            .collect(),
        form_factor_momentum: form_factor.form_factor_momentum,
        form_factor: form_factor.form_factor,
    })
}

fn generated_config_dat(input: &PotInput, config_inp: &Path) -> Result<ConfigDatData> {
    let expected_potentials = usize::try_from(input.control.nph)
        .context("ATOM nph cannot be represented as usize")?
        .checked_add(1)
        .context("ATOM nph potential count overflowed")?;
    ensure!(
        input.potentials.len() == expected_potentials,
        "ATOM pot.inp has {} potential row(s), expected {expected_potentials} from nph={}",
        input.potentials.len(),
        input.control.nph
    );

    let recipe = configuration_recipe(input.config_type);
    let source = ConfigurationRowsSource::new(input, recipe, config_inp)?;
    let configurations = atomic_orbital_configurations(input, &source)?;
    let final_index = source.potential_count();

    let mut potentials = Vec::with_capacity(source.potential_count());
    for potential_index in 0..source.potential_count() {
        let (metadata_index, occupation_index) =
            dump_config_indices(input.run.nohole, final_index, potential_index);
        let (occupations, valence_occupations) = dump_config_occupation_rows(
            &configurations[metadata_index],
            &configurations[occupation_index],
        )?;
        let atomic_number = source.atomic_number(potential_index);
        potentials.push(ConfigDatPotential {
            potential_index: i32::try_from(potential_index)
                .context("ATOM potential index cannot be represented as i32")?,
            atomic_number: i32::try_from(atomic_number)
                .context("ATOM atomic number cannot be represented as i32")?,
            element: atomic_symbol(atomic_number)?.to_string(),
            occupations,
            valence_occupations,
            spin_occupations: None,
        });
    }

    Ok(ConfigDatData {
        header_lines: generated_config_header_lines(),
        potentials,
    })
}

fn configuration_recipe(config_type: i32) -> FeffConfigurationRecipe {
    if config_type == 7 {
        FeffConfigurationRecipe::Feff7
    } else {
        FeffConfigurationRecipe::Feff9
    }
}

fn atomic_orbital_configurations(
    input: &PotInput,
    source: &ConfigurationRowsSource,
) -> Result<Vec<OrbitalConfiguration>> {
    let absorber_ionicity = input.potentials[0].xion;
    let absorber_hole = checked_hole_index(input.control.ihole)?;
    let final_absorber =
        source.compact_configuration(0, absorber_hole, absorber_ionicity, input.run.iunf != 0)?;
    let mut initial_absorber =
        source.compact_configuration(0, 0, absorber_ionicity, input.run.iunf != 0)?;
    if absorber_hole > 0 {
        subtract_valence_from_electron_counts(&mut initial_absorber);
    }

    let mut configurations = Vec::with_capacity(source.potential_count() + 1);
    for potential_index in 0..source.potential_count() {
        if potential_index == 0 {
            configurations.push(if input.run.nohole < 0 {
                final_absorber.clone()
            } else {
                initial_absorber.clone()
            });
        } else {
            configurations.push(source.compact_configuration(
                potential_index,
                0,
                input.potentials[potential_index].xion,
                input.run.iunf != 0,
            )?);
        }
    }
    configurations.push(if input.run.nohole >= 0 {
        final_absorber
    } else {
        initial_absorber
    });
    Ok(configurations)
}

#[allow(dead_code)]
fn atomic_scf_orbital_configurations(
    input: &PotInput,
    source: &ConfigurationRowsSource,
) -> Result<Vec<OrbitalConfiguration>> {
    let absorber_ionicity = input.potentials[0].xion;
    let absorber_hole = checked_hole_index(input.control.ihole)?;
    let final_absorber =
        source.compact_configuration(0, absorber_hole, absorber_ionicity, input.run.iunf != 0)?;
    let initial_absorber =
        source.compact_configuration(0, 0, absorber_ionicity, input.run.iunf != 0)?;

    let mut configurations = Vec::with_capacity(source.potential_count() + 1);
    for potential_index in 0..source.potential_count() {
        if potential_index == 0 {
            configurations.push(if input.run.nohole < 0 {
                final_absorber.clone()
            } else {
                initial_absorber.clone()
            });
        } else {
            configurations.push(source.compact_configuration(
                potential_index,
                0,
                input.potentials[potential_index].xion,
                input.run.iunf != 0,
            )?);
        }
    }
    configurations.push(if input.run.nohole >= 0 {
        final_absorber
    } else {
        initial_absorber
    });
    Ok(configurations)
}

#[allow(dead_code)]
fn atomic_orbital_indices_by_kappa(input: &PotInput, config_inp: &Path) -> Result<Array2<i64>> {
    let state_count = apot_state_count(input)?;
    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );

    let mut values = Array2::<i64>::zeros((FEFF_KAPPA_PROJECTION_COUNT, state_count));
    for (state_index, configuration) in configurations.iter().enumerate() {
        ensure!(
            configuration.projection_orbitals.len() == FEFF_KAPPA_PROJECTION_COUNT,
            "ATOM iorb state {state_index} has {} kappa slot(s), expected {FEFF_KAPPA_PROJECTION_COUNT}",
            configuration.projection_orbitals.len()
        );
        for (slot, &orbital) in configuration.projection_orbitals.iter().enumerate() {
            values[(slot, state_index)] =
                i64::try_from(orbital).context("ATOM iorb slot cannot be represented as i64")?;
        }
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_norman_valence_counts_by_l(input: &PotInput, config_inp: &Path) -> Result<Array2<f64>> {
    let state_count = apot_state_count(input)?;
    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    atomic_norman_valence_counts_from_configurations(&configurations, state_count)
}

#[allow(dead_code)]
fn atomic_norman_valence_counts_from_configurations(
    configurations: &[OrbitalConfiguration],
    state_count: usize,
) -> Result<Array2<f64>> {
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );

    let mut values = Array2::<f64>::zeros((ATOM_NORMAN_VALENCE_CHANNEL_COUNT, state_count));
    for (state_index, configuration) in configurations.iter().enumerate() {
        ensure!(
            configuration.kappa.len() >= configuration.orbital_count
                && configuration.valence_counts.len() >= configuration.orbital_count,
            "ATOM xnvmu state {state_index} has orbital_count={} but kappa len={} and xnval len={}",
            configuration.orbital_count,
            configuration.kappa.len(),
            configuration.valence_counts.len()
        );
        for orbital in 0..configuration.orbital_count {
            let valence_count = configuration.valence_counts[orbital];
            ensure!(
                valence_count.is_finite(),
                "ATOM xnvmu state {state_index} orbital {} has non-finite xnval {valence_count}",
                orbital + 1
            );
            let angular = atomic_angular_momentum_from_kappa(configuration.kappa[orbital])
                .with_context(|| {
                    format!(
                        "ATOM xnvmu state {state_index} orbital {} has invalid kappa {}",
                        orbital + 1,
                        configuration.kappa[orbital]
                    )
                })?;
            if angular < ATOM_NORMAN_VALENCE_CHANNEL_COUNT {
                values[(angular, state_index)] += valence_count;
            }
        }
    }
    Ok(values)
}

#[allow(dead_code)]
fn atomic_angular_momentum_from_kappa(kappa: i32) -> Result<usize> {
    let angular = if kappa < 0 {
        kappa
            .checked_neg()
            .and_then(|value| value.checked_sub(1))
            .context("ATOM kappa angular-momentum conversion overflowed")?
    } else {
        kappa
    };
    usize::try_from(angular).context("ATOM kappa angular momentum cannot be represented as usize")
}

#[allow(dead_code)]
fn atomic_apot_amplitude_reduction_from_states(
    input: &PotInput,
    config_inp: &Path,
    states: &[AtomicScfState],
) -> Result<f64> {
    let state_count = apot_state_count(input)?;
    ensure!(
        states.len() == state_count,
        "ATOM generated {} SCF state(s), expected {state_count}",
        states.len()
    );
    let hole_index = checked_hole_index(input.control.ihole)?;
    if hole_index == 0 {
        return Ok(1.0);
    }

    let configurations = atomic_scf_configurations_from_pot_input(input, config_inp)?;
    ensure!(
        configurations.len() == state_count,
        "ATOM compacted configuration count {} does not match apot state count {state_count}",
        configurations.len()
    );
    let (initial_column, final_column) = atomic_apot_absorber_state_columns(input, state_count)?;
    let initial_configuration = &configurations[initial_column];
    let final_configuration = &configurations[final_column];
    let hole_orbital_1based = final_configuration.hole_position;
    if hole_orbital_1based == 0 {
        return Ok(1.0);
    }

    let overlaps = atomic_apot_relaxed_overlap_integrals(
        initial_configuration,
        final_configuration,
        &states[initial_column],
        &states[final_column],
    )?;
    let occupations = initial_configuration
        .electron_counts
        .iter()
        .zip(initial_configuration.valence_counts.iter())
        .take(initial_configuration.orbital_count)
        .map(|(&electron, &valence)| electron - valence)
        .collect::<Vec<_>>();
    for (orbital, occupation) in occupations.iter().enumerate() {
        ensure!(
            occupation.is_finite(),
            "ATOM s02 occupation for orbital {} is non-finite: {occupation}",
            orbital + 1
        );
    }
    let kappas = initial_configuration
        .kappa
        .iter()
        .take(initial_configuration.orbital_count)
        .copied()
        .collect::<Vec<_>>();

    atomic_overlap_amplitude_reduction(AtomicOverlapAmplitudeReductionInput {
        hole_orbital_1based: Some(hole_orbital_1based),
        kappas: &kappas,
        occupations: &occupations,
        overlap_integrals: overlaps.view(),
    })
    .context("failed to generate ATOM s02 from relaxed absorber overlaps")
}

#[allow(dead_code)]
fn atomic_apot_absorber_state_columns(
    input: &PotInput,
    state_count: usize,
) -> Result<(usize, usize)> {
    ensure!(state_count >= 2, "ATOM apot state count must be at least 2");
    let initial_column = fpf0_initial_absorber_column(input)?;
    ensure!(
        initial_column < state_count,
        "ATOM initial absorber column {initial_column} exceeds apot state count {state_count}"
    );
    let final_column = if initial_column == 0 {
        state_count - 1
    } else {
        0
    };
    Ok((initial_column, final_column))
}

#[allow(dead_code)]
fn atomic_apot_relaxed_overlap_integrals(
    initial_configuration: &OrbitalConfiguration,
    final_configuration: &OrbitalConfiguration,
    initial_state: &AtomicScfState,
    final_state: &AtomicScfState,
) -> Result<Array2<f64>> {
    let orbital_count = initial_configuration.orbital_count;
    atomic_apot_ensure_s02_state("initial absorber", initial_configuration, initial_state)?;
    atomic_apot_ensure_s02_state("final absorber", final_configuration, final_state)?;
    ensure!(
        final_configuration.orbital_count >= orbital_count,
        "ATOM s02 final absorber has {} orbital(s), expected at least {orbital_count}",
        final_configuration.orbital_count
    );

    let initial_origin_powers = initial_state.initial_orbitals.orbital_powers.to_vec();
    let active_lengths = initial_state
        .scf
        .active_lengths
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let radii = &initial_state.initial_orbitals.radii;
    let mut overlaps = Array2::<f64>::zeros((orbital_count, orbital_count));

    for outer in 0..orbital_count {
        let kappa = initial_configuration.kappa[outer];
        let Some(final_orbital) =
            atomic_apot_relaxed_overlap_final_orbital(final_configuration, outer, kappa)
        else {
            for row in 0..=outer {
                overlaps[(row, outer)] = if row == outer { 1.0 } else { 0.0 };
            }
            continue;
        };
        let derivative_large = final_state
            .scf
            .large_components
            .column(final_orbital)
            .to_owned();
        let derivative_small = final_state
            .scf
            .small_components
            .column(final_orbital)
            .to_owned();
        let derivative_large_coefficients = final_state
            .scf
            .large_coefficients
            .column(final_orbital)
            .to_owned();
        let derivative_small_coefficients = final_state
            .scf
            .small_coefficients
            .column(final_orbital)
            .to_owned();

        for inner in 0..orbital_count {
            if initial_configuration.kappa[inner] != kappa {
                continue;
            }
            overlaps[(outer, inner)] =
                atomic_differential_integral(AtomicDifferentialIntegralInput {
                    kind: AtomicDifferentialIntegralKind::DerivativeProjection {
                        large_orbital_1based: inner + 1,
                        small_orbital_1based: inner + 1,
                    },
                    power: 0,
                    origin_power: initial_origin_powers[outer],
                    step: ATOM_RADIAL_STEP,
                    radii: radii.view(),
                    active_lengths: &active_lengths,
                    orbital_powers: &initial_origin_powers,
                    large_components: initial_state.scf.large_components.view(),
                    small_components: initial_state.scf.small_components.view(),
                    large_coefficients: initial_state.scf.large_coefficients.view(),
                    small_coefficients: initial_state.scf.small_coefficients.view(),
                    derivative_large: derivative_large.view(),
                    derivative_small: derivative_small.view(),
                    derivative_large_coefficients: derivative_large_coefficients.view(),
                    derivative_small_coefficients: derivative_small_coefficients.view(),
                })
                .with_context(|| {
                    format!(
                        "failed to generate ATOM s02 overlap outer={} inner={}",
                        outer + 1,
                        inner + 1
                    )
                })?;
        }
    }

    Ok(overlaps)
}

#[allow(dead_code)]
fn atomic_apot_ensure_s02_state(
    label: &'static str,
    configuration: &OrbitalConfiguration,
    state: &AtomicScfState,
) -> Result<()> {
    let orbital_count = configuration.orbital_count;
    ensure!(
        configuration.kappa.len() >= orbital_count
            && configuration.electron_counts.len() >= orbital_count
            && configuration.valence_counts.len() >= orbital_count,
        "ATOM s02 {label} configuration has orbital_count={} but kappa len={}, xnel len={}, xnval len={}",
        orbital_count,
        configuration.kappa.len(),
        configuration.electron_counts.len(),
        configuration.valence_counts.len()
    );
    ensure!(
        state.scf.large_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.small_components.nrows() >= ATOM_RADIAL_POINTS
            && state.scf.large_components.ncols() >= orbital_count
            && state.scf.small_components.ncols() >= orbital_count,
        "ATOM s02 {label} component shapes large={:?}, small={:?}, expected at least {ATOM_RADIAL_POINTS}x{orbital_count}",
        state.scf.large_components.dim(),
        state.scf.small_components.dim()
    );
    ensure!(
        state.scf.large_coefficients.ncols() >= orbital_count
            && state.scf.small_coefficients.ncols() >= orbital_count,
        "ATOM s02 {label} coefficient shapes large={:?}, small={:?}, expected at least one column per orbital",
        state.scf.large_coefficients.dim(),
        state.scf.small_coefficients.dim()
    );
    ensure!(
        state.scf.active_lengths.len() >= orbital_count,
        "ATOM s02 {label} has {} active length(s), expected {orbital_count}",
        state.scf.active_lengths.len()
    );
    Ok(())
}

#[allow(dead_code)]
fn atomic_apot_relaxed_overlap_final_orbital(
    final_configuration: &OrbitalConfiguration,
    initial_orbital: usize,
    initial_kappa: i32,
) -> Option<usize> {
    if final_configuration.kappa.get(initial_orbital) == Some(&initial_kappa) {
        return Some(initial_orbital);
    }
    let shifted = initial_orbital.checked_add(1)?;
    (final_configuration.kappa.get(shifted) == Some(&initial_kappa)).then_some(shifted)
}

fn subtract_valence_from_electron_counts(configuration: &mut OrbitalConfiguration) {
    for (electron_count, valence_count) in configuration
        .electron_counts
        .iter_mut()
        .zip(configuration.valence_counts.iter())
    {
        *electron_count -= *valence_count;
    }
}

fn checked_hole_index(ihole: i32) -> Result<usize> {
    ensure!(ihole >= 0, "ATOM ihole must be non-negative, got {ihole}");
    let ihole = usize::try_from(ihole).context("ATOM ihole cannot be represented as usize")?;
    ensure!(
        ihole <= FEFF_ORBITAL_SLOT_COUNT,
        "ATOM ihole {ihole} is outside FEFF's 0..={FEFF_ORBITAL_SLOT_COUNT} configuration slots"
    );
    Ok(ihole)
}

fn dump_config_indices(nohole: i32, final_index: usize, potential_index: usize) -> (usize, usize) {
    if potential_index == 0 && nohole >= 0 {
        let occupation_index = if nohole == 0 {
            potential_index
        } else {
            final_index
        };
        (final_index, occupation_index)
    } else {
        (potential_index, potential_index)
    }
}

fn dump_config_occupation_rows(
    metadata: &OrbitalConfiguration,
    counts: &OrbitalConfiguration,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let mut occupations = Array1::zeros(FEFF_ORBITAL_SLOT_COUNT);
    let mut valence_occupations = Array1::zeros(FEFF_ORBITAL_SLOT_COUNT);
    for orbital in 0..metadata.orbital_count {
        let slot = feff_configuration_slot(
            metadata.principal_quantum_numbers[orbital],
            metadata.kappa[orbital],
        )
        .with_context(|| {
            format!(
                "ATOM compacted orbital {} with n={} kappa={} has no FEFF configuration slot",
                orbital + 1,
                metadata.principal_quantum_numbers[orbital],
                metadata.kappa[orbital]
            )
        })?;
        occupations[slot] = counts.electron_counts.get(orbital).copied().unwrap_or(0.0);
        valence_occupations[slot] = counts.valence_counts.get(orbital).copied().unwrap_or(0.0);
    }
    Ok((occupations, valence_occupations))
}

fn feff_configuration_slot(principal_quantum_number: i32, kappa: i32) -> Option<usize> {
    FEFF_ORBITAL_PRINCIPAL_QUANTUM_NUMBERS
        .iter()
        .zip(FEFF_ORBITAL_KAPPAS.iter())
        .position(|(&n, &slot_kappa)| n == principal_quantum_number && slot_kappa == kappa)
}

fn generated_config_header_lines() -> Vec<String> {
    vec![
        "# Configuration of all atom types in feff.inp.".to_string(),
        "  # Atomic occupation numbers including core hole, screening, and ionicity (but no SCF)."
            .to_string(),
        "# iph, z,name,  iocc/ival (i=1,40)".to_string(),
    ]
}

#[derive(Debug, Clone)]
struct ConfigurationRowsSource {
    recipe: FeffConfigurationRecipe,
    atomic_numbers: Vec<usize>,
    potential_rows: Vec<ConfigSlotRows>,
}

impl ConfigurationRowsSource {
    fn new(input: &PotInput, recipe: FeffConfigurationRecipe, config_inp: &Path) -> Result<Self> {
        let atomic_numbers = input
            .potentials
            .iter()
            .map(|potential| checked_atomic_number(potential.z))
            .collect::<Result<Vec<_>>>()?;
        let mut potential_rows = atomic_numbers
            .iter()
            .map(|&atomic_number| default_slot_rows(atomic_number, recipe))
            .collect::<Result<Vec<_>>>()?;

        if input.config_type == 2 {
            let config = read_config_inp(config_inp)
                .with_context(|| format!("failed to read {}", config_inp.display()))?;
            for record in &config.records {
                apply_config_record(&mut potential_rows, &atomic_numbers, record)?;
            }
        }

        Ok(Self {
            recipe,
            atomic_numbers,
            potential_rows,
        })
    }

    fn potential_count(&self) -> usize {
        self.atomic_numbers.len()
    }

    fn atomic_number(&self, potential_index: usize) -> usize {
        self.atomic_numbers[potential_index]
    }

    fn compact_configuration(
        &self,
        potential_index: usize,
        hole_index: usize,
        ionicity: f64,
        unfreeze_f_or_higher: bool,
    ) -> Result<OrbitalConfiguration> {
        let atomic_number = self.atomic_number(potential_index);
        let template_atomic_number = template_atomic_number(atomic_number, ionicity)?;
        let rows = if template_atomic_number == atomic_number {
            self.potential_rows[potential_index].clone()
        } else {
            self.rows_for_atomic_number(template_atomic_number)?
        };
        let next_rows = self.rows_for_atomic_number(
            template_atomic_number
                .checked_add(1)
                .context("ATOM template atomic number overflowed")?,
        )?;

        orbital_configuration(OrbitalConfigurationInput {
            atomic_number,
            hole_index,
            ionicity,
            unfreeze_f_or_higher,
            occupations: rows.occupations.view(),
            valence_occupations: rows.valence_occupations.view(),
            spin_occupations: rows.spin_occupations.view(),
            next_occupations: next_rows.occupations.view(),
        })
        .with_context(|| {
            format!(
                "failed to build ATOM orbital configuration for potential {potential_index} (Z={atomic_number})"
            )
        })
    }

    fn rows_for_atomic_number(&self, atomic_number: usize) -> Result<ConfigSlotRows> {
        if let Some((index, _)) = self
            .atomic_numbers
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, potential_atomic_number)| **potential_atomic_number == atomic_number)
        {
            return Ok(self.potential_rows[index].clone());
        }
        default_slot_rows(atomic_number, self.recipe)
    }
}

fn checked_atomic_number(atomic_number: i32) -> Result<usize> {
    ensure!(
        atomic_number > 0,
        "ATOM potential atomic number must be positive, got {atomic_number}"
    );
    usize::try_from(atomic_number).context("ATOM atomic number cannot be represented as usize")
}

fn default_slot_rows(
    atomic_number: usize,
    recipe: FeffConfigurationRecipe,
) -> Result<ConfigSlotRows> {
    let rows = feff_default_configuration_rows(atomic_number, recipe)?;
    Ok(slot_rows_from_default(rows))
}

fn slot_rows_from_default(rows: FeffDefaultConfigurationRows) -> ConfigSlotRows {
    ConfigSlotRows {
        electron_count: rows.occupations.iter().sum(),
        occupations: rows.occupations,
        valence_occupations: rows.valence_occupations,
        spin_occupations: rows.spin_occupations,
    }
}

fn template_atomic_number(atomic_number: usize, ionicity: f64) -> Result<usize> {
    ensure!(
        ionicity.is_finite(),
        "ATOM ionicity must be finite, got {ionicity}"
    );
    let ion = ionicity.round() as isize;
    let template = isize::try_from(atomic_number)
        .context("ATOM atomic number cannot be represented as isize")?
        - ion;
    ensure!(
        (1..139).contains(&template),
        "ATOM template atomic number {template} is outside FEFF's 1..=138 getorb range"
    );
    usize::try_from(template).context("ATOM template atomic number cannot be represented as usize")
}

fn apply_config_record(
    potential_rows: &mut [ConfigSlotRows],
    atomic_numbers: &[usize],
    record: &ConfigRecord,
) -> Result<()> {
    let target = usize::try_from(i64::from(record.potential_index).abs())
        .context("ATOM config.inp potential index cannot be represented as usize")?;
    ensure!(
        target < potential_rows.len(),
        "ATOM config.inp references potential {target}, but only {} potential row(s) are available",
        potential_rows.len()
    );
    let target_element = atomic_symbol(atomic_numbers[target])?;
    ensure!(
        record.element == target_element,
        "ATOM config.inp record element {} does not match potential {target} element {target_element}",
        record.element
    );

    let patched =
        config_record_slot_rows(record, Some(&potential_rows[target])).with_context(|| {
            format!("failed to expand ATOM config.inp record for potential {target}")
        })?;
    if record.potential_index < 0 {
        for (potential_index, &atomic_number) in atomic_numbers.iter().enumerate() {
            if atomic_symbol(atomic_number)? == record.element {
                potential_rows[potential_index] = patched.clone();
            }
        }
    } else {
        potential_rows[target] = patched;
    }
    Ok(())
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

fn write_or_generate_module_log(path: &Path, input: &PotInput) -> Result<usize> {
    if path.is_file() {
        return write_optional_module_log(path);
    }
    write_generated_module_log(path, input)
}

fn write_or_recover_module_log(
    path: &Path,
    input: &PotInput,
    source_handoff_written: bool,
) -> Result<usize> {
    if source_handoff_written && path.is_file() && read_module_log_dat(path).is_err() {
        return write_generated_module_log(path, input);
    }
    write_or_generate_module_log(path, input)
}

fn recover_existing_module_log_if_malformed(
    path: &Path,
    input: &PotInput,
    source_handoff_written: bool,
) -> Result<usize> {
    if !source_handoff_written || !path.is_file() || read_module_log_dat(path).is_ok() {
        return Ok(0);
    }
    write_generated_module_log(path, input)
}

fn write_generated_module_log(path: &Path, input: &PotInput) -> Result<usize> {
    let data = generated_atomic_module_log(input);
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

fn generated_atomic_module_log(input: &PotInput) -> ModuleLogData {
    let mut lines = vec!["Calculating atomic potentials ...".to_string()];
    let overlap_passes = if input
        .potentials
        .iter()
        .any(|potential| potential.xion.abs() > 1.0e-3)
    {
        2
    } else {
        1
    };

    for _ in 0..overlap_passes {
        for potential_index in 0..input.potentials.len() {
            lines.push(format!(
                "    overlapped atomic potential and density for unique potential{potential_index:5}"
            ));
        }
    }
    lines.push("Done with module: atomic potentials.".to_string());

    ModuleLogData {
        line_terminators: vec!["\n".to_string(); lines.len()],
        lines,
    }
}

fn apot_section<'a>(
    apot: &'a ApotBinData,
    section_number: usize,
    label: &'static str,
) -> Result<&'a refeff_io::ApotBinSection> {
    apot.sections
        .iter()
        .find(|section| section.section_number == section_number)
        .with_context(|| format!("ATOM apot.bin is missing section {section_number} {label}"))
}

fn real_matrix_section<'a>(
    apot: &'a ApotBinData,
    section_number: usize,
    label: &'static str,
) -> Result<&'a Array2<f64>> {
    let section = apot_section(apot, section_number, label)?;
    let ApotBinPayload::Matrix(matrix) = &section.payload else {
        bail!("ATOM apot.bin section {section_number} {label} is not a matrix payload");
    };
    match &matrix.values {
        ApotBinMatrixValues::Real(values) => Ok(values),
        _ => bail!("ATOM apot.bin section {section_number} {label} is not real-valued"),
    }
}

fn int_matrix_section<'a>(
    apot: &'a ApotBinData,
    section_number: usize,
    label: &'static str,
) -> Result<&'a Array2<i64>> {
    let section = apot_section(apot, section_number, label)?;
    let ApotBinPayload::Matrix(matrix) = &section.payload else {
        bail!("ATOM apot.bin section {section_number} {label} is not a matrix payload");
    };
    match &matrix.values {
        ApotBinMatrixValues::Int(values) => Ok(values),
        _ => bail!("ATOM apot.bin section {section_number} {label} is not integer-valued"),
    }
}

fn real_matrix_column(
    matrix: &Array2<f64>,
    column: usize,
    row_count: usize,
    label: &'static str,
) -> Result<Array1<f64>> {
    ensure!(
        matrix.nrows() == row_count,
        "ATOM apot.bin {label} has {} row(s), expected {row_count}",
        matrix.nrows()
    );
    ensure!(
        column < matrix.ncols(),
        "ATOM apot.bin {label} has {} column(s), missing column {column}",
        matrix.ncols()
    );
    Ok(matrix.column(column).to_owned())
}

fn real_matrix_column_prefix(
    matrix: &Array2<f64>,
    column: usize,
    count: usize,
    label: &'static str,
) -> Result<Vec<f64>> {
    ensure!(
        matrix.nrows() >= count,
        "ATOM apot.bin {label} has {} row(s), expected at least {count}",
        matrix.nrows()
    );
    ensure!(
        column < matrix.ncols(),
        "ATOM apot.bin {label} has {} column(s), missing column {column}",
        matrix.ncols()
    );
    Ok((0..count).map(|row| matrix[(row, column)]).collect())
}

fn int_matrix_column_prefix(
    matrix: &Array2<i64>,
    column: usize,
    count: usize,
    label: &'static str,
) -> Result<Vec<i32>> {
    ensure!(
        matrix.nrows() >= count,
        "ATOM apot.bin {label} has {} row(s), expected at least {count}",
        matrix.nrows()
    );
    ensure!(
        column < matrix.ncols(),
        "ATOM apot.bin {label} has {} column(s), missing column {column}",
        matrix.ncols()
    );
    (0..count)
        .map(|row| {
            i32::try_from(matrix[(row, column)]).with_context(|| {
                format!("ATOM apot.bin {label} row {row} cannot be represented as i32")
            })
        })
        .collect()
}

fn real_matrix_prefix(
    matrix: &Array2<f64>,
    column_count: usize,
    label: &'static str,
) -> Result<Array2<f64>> {
    ensure!(
        matrix.nrows() == ATOM_RADIAL_POINTS,
        "ATOM apot.bin {label} has {} row(s), expected {ATOM_RADIAL_POINTS}",
        matrix.nrows()
    );
    ensure!(
        matrix.ncols() >= column_count,
        "ATOM apot.bin {label} has {} column(s), expected at least {column_count}",
        matrix.ncols()
    );
    Ok(Array2::from_shape_fn(
        (ATOM_RADIAL_POINTS, column_count),
        |(row, column)| matrix[(row, column)],
    ))
}

fn real_matrix_row_prefix(
    matrix: &Array2<f64>,
    column_count: usize,
    label: &'static str,
) -> Result<Array2<f64>> {
    ensure!(
        matrix.ncols() >= column_count,
        "ATOM apot.bin {label} has {} column(s), expected at least {column_count}",
        matrix.ncols()
    );
    Ok(Array2::from_shape_fn(
        (matrix.nrows(), column_count),
        |(row, column)| matrix[(row, column)],
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AtomicCachePaths {
    pot_inp: PathBuf,
    pot_bin: PathBuf,
    geom_dat: PathBuf,
    apot_bin: PathBuf,
    config_inp: PathBuf,
    config_dat: PathBuf,
    fpf0_dat: PathBuf,
    log1_dat: PathBuf,
    pot_scf_cache: PathBuf,
}

impl AtomicCachePaths {
    fn new(work_dir: &Path) -> Self {
        Self {
            pot_inp: work_dir.join("pot.inp"),
            pot_bin: work_dir.join("pot.bin"),
            geom_dat: work_dir.join("geom.dat"),
            apot_bin: work_dir.join("apot.bin"),
            config_inp: work_dir.join("config.inp"),
            config_dat: work_dir.join("config.dat"),
            fpf0_dat: work_dir.join("fpf0.dat"),
            log1_dat: work_dir.join("log1.dat"),
            pot_scf_cache: work_dir.join(POT_SCF_CACHE_PROVENANCE_FILE),
        }
    }
}

#[cfg(all(test, feature = "full"))]
mod tests;
