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
    MuffinTinInterstitialParametersInput, MuffinTinOverlapMatrixInput, MuffinTinOverlapNeighbor,
    MuffinTinOverlapProjectionInput, MuffinTinOverlapProjectionMode,
    MuffinTinRadiusParametersInput, NormanRadiusInput, OrbitalConfiguration,
    OrbitalConfigurationInput, OverlapDensityIndicesInput, PotScfContourRunInput,
    PotScfContourSourceRows, PotScfContourSourceRowsInput, PotScfOuterIterationStatus, PotScfState,
    PotScfStateAdvance, PotScfStateAdvanceInput, PotentialOverlapInput, PotentialOverlapNeighbor,
    ScfDensityStepInput, ScmtEnergyGrid, ScmtEnergyGridInput, advance_pot_scf_state,
    atomic::{
        AtomicLocalDensityExchangeMode, AtomicScfState, AtomicScfStateInput,
        atomic_coulomb_coefficients, atomic_differential_integral, atomic_form_factor,
        atomic_overlap_amplitude_reduction, atomic_scf_state_from_configuration, atomic_symbol,
        atomic_tabulation, atomic_total_energy_from_radials,
    },
    dirac_hara_exchange_potential, feff_default_configuration_rows, interstitial_fermi_level,
    interstitial_shell_values, karasiev_sjostrom_dufty_trickey_vxc,
    muffin_tin_interstitial_parameters, muffin_tin_overlap_matrix, muffin_tin_radius_parameters,
    norman_radius_from_density, orbital_configuration, overlap_density_indices,
    overlap_potential_density, perdew_zunger_vxc, perrot_dharma_wardana_vxc,
    pot_scf_contour_source_rows, project_muffin_tin_overlap, scmt_energy_grid,
    update_scf_density_potential, von_barth_hedin_potential,
};
use refeff_io::{
    APOT_CORE_HOLE_GRID_STEP, APOT_CORE_HOLE_RADIAL_POINTS, APOT_CORE_HOLE_SECTION_NUMBER,
    ApotAtomicPotsSectionsInput, ApotAtomicScfStateRef, ApotAtomicScfStateSectionsInput,
    ApotBinData, ApotBinMatrixValues, ApotBinPayload, ApotBinSection, ApotBinValue,
    ApotCoreHoleColumns, AtomDatData, ConfigDatData, ConfigDatPotential, ConfigRecord,
    ConfigSlotRows, FEFF_BOHR_ANGSTROM, Fpf0DatData, Fpf0Oscillator, GeomDat, ModuleLogData,
    MtdpData, PotBinData, PotBinScalars, PotInput, PotScfCorvalLdosHandoffInput,
    PotScfFmsSourceGridHandoff, PotScfFovrgSourceGridFromPlanInput, PotScfFovrgSourceGridHandoff,
    PotScfFovrgSourceGridPlan, PotScfFovrgSourceGridPlanInput, apot_atomic_pots_sections,
    apot_atomic_scf_sections_from_states, apot_bin_string, apot_core_hole_columns,
    apot_core_hole_coulomb_from_density, apot_core_hole_radii, config_record_slot_rows,
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

use crate::fms::{PotScfFmsSourceGridInput, build_pot_scf_fms_source_grid_handoff};
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
const POT_NO_SCF_MUFFIN_TIN_WINDOW: usize = 50;
const POT_SCMT_MAX_ENERGY_POINTS: usize = 80;
const POT_SCMT_FLOOR_COUNT: usize = 17;
// Finite-nucleus spectra can need substantially more upward contour steps
// than the point-nucleus seed grid before the electron-count root is bracketed.
const POT_SCMT_MAX_ADAPTIVE_SOURCE_POINTS: usize = POT_SCMT_MAX_ENERGY_POINTS * 8;
const POT_SCMT_CHARGE_SUM_TOLERANCE: f64 = 0.05;
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

pub(crate) fn has_stale_no_scf_pot_bin_from_sources_in_dir(work_dir: &Path) -> Result<bool> {
    let caches = AtomicCachePaths::new(work_dir);
    if !caches.pot_inp.is_file() || !caches.geom_dat.is_file() {
        return Ok(false);
    }

    let input = read_input(work_dir)?;
    if !atomic_enabled(&input) || input.run.nscmt != 0 || input.start_from_file {
        return Ok(false);
    }

    let (pot, mut apot) = match generated_no_scf_pot_bin_and_apot_from_sources(&caches, &input) {
        Ok(generated) => generated,
        Err(error) if no_scf_pot_generation_unsupported(&error) => return Ok(false),
        Err(error) => return Err(error),
    };
    refresh_apot_core_hole_coulomb_payload(&mut apot, input.run.nohole)
        .with_context(|| format!("failed to refresh {}", caches.apot_bin.display()))?;

    Ok(!cached_pot_bin_matches_generated(&caches.pot_bin, &pot)?
        || !cached_apot_bin_matches_generated(&caches.apot_bin, &apot)?)
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
    scf_pot_initial_state_from_generated_pot(
        work_dir,
        input,
        &context.static_arrays,
        &context.config,
        context.pot,
        context.external_source.as_ref(),
        context.restart_pot.as_ref(),
        true,
    )
}

fn generated_scf_pot_run_from_sources(
    caches: &AtomicCachePaths,
    input: &PotInput,
) -> Result<PotScfSourceRun> {
    let work_dir = caches.pot_inp.parent().unwrap_or_else(|| Path::new("."));
    let mut retry_input = input.clone();
    let mut attempt = 1usize;
    let mut run = loop {
        let context = scf_pot_source_context_from_sources(caches, &retry_input, attempt == 1)?;
        let initial = scf_pot_initial_state_from_generated_pot(
            work_dir,
            &retry_input,
            &context.static_arrays,
            &context.config,
            context.pot.clone(),
            context.external_source.as_ref(),
            context.restart_pot.as_ref(),
            attempt == 1,
        )?;
        let run = scf_pot_run_from_initial_state(
            work_dir,
            &retry_input,
            &context.static_arrays,
            &context.config,
            initial,
        )?;
        let mut run = run;
        attach_scf_pot_final_apot(&mut run, &retry_input, &caches.config_inp, &context)?;
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
    ) = match scf_pot_adaptive_source_advance(
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
    ) {
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
            scf_pot_final_pot_from_state(&previous_pot, &current_state, status, config)
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
                current_state = advance.state;
                if scf_pot_status_has_final_pot(status) {
                    final_pot = Some(
                        scf_pot_final_pot_from_state(
                            &previous_pot,
                            &current_state,
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

    let mut final_pot = pot.clone();
    final_pot.raw_text = None;
    final_pot.scalars.fermi_level = state.fermi_energy;
    final_pot.norman_charges = state.norman_charges.clone();
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
    plan: &PotScfFovrgSourceGridPlan,
    energies: ArrayView1<'_, Complex64>,
) -> Result<PotScfFovrgSourceGridHandoff> {
    ensure!(
        !energies.is_empty(),
        "POT SCF FOVRG source grid requires at least one energy row"
    );
    pot_scf_fovrg_source_grid_handoff_from_plan(PotScfFovrgSourceGridFromPlanInput {
        plan,
        energies_hartree: energies,
    })
    .context("failed to build POT SCF FOVRG source grid from reusable plan")
}

fn scf_pot_fms_source_grid_from_initial_state(
    work_dir: &Path,
    input: &PotInput,
    fovrg_grid: &PotScfFovrgSourceGridHandoff,
) -> Result<PotScfFmsSourceGridHandoff> {
    let angular_count = fovrg_grid.phase_shifts.dim().1;
    ensure!(
        angular_count > 0,
        "POT initial SCF FMS source grid requires at least one phase angular channel"
    );
    build_pot_scf_fms_source_grid_handoff(
        work_dir,
        PotScfFmsSourceGridInput {
            pot: input,
            energy_grid_hartree: fovrg_grid.energies_hartree.view(),
            reference_energies_hartree: fovrg_grid.reference_energies_hartree.view(),
            phase_shifts: fovrg_grid.phase_shifts.view(),
            angular_count,
        },
    )
    .context("failed to build POT initial SCF FMS source grid from FOVRG phases")
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
    let output_radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);
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
                if (energy_index + 1 == energies.len() || current < previous)
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
) -> Result<(
    PotScfFovrgSourceGridHandoff,
    PotScfFmsSourceGridHandoff,
    PotScfContourSourceRows,
)> {
    let fovrg_grid = scf_pot_fovrg_source_grid_for_energies_from_plan(fovrg_plan, energies)
        .context("failed to build POT SCF FOVRG source grid")?;
    let fms_grid = scf_pot_fms_source_grid_from_initial_state(work_dir, input, &fovrg_grid)
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
    ) = match scf_pot_adaptive_source_advance(
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
    ) {
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
    pot.scalars.total_volume = pot_output_total_volume_bohr3(input, istprm.interstitial_volume)?;
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
            let (apot, states) = generated_atomic_apot_and_states_from_sources(&caches, &input)?;
            generated_states = Some(states);
            apot
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
                    let (apot, states) =
                        generated_atomic_apot_and_states_from_sources(&caches, &input)?;
                    generated_states = Some(states);
                    apot
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
    let state_inputs = states
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

    let static_arrays = atomic_apot_static_arrays_from_source_geometry(
        input,
        geom,
        Array1::from_elem(unique_count, 1.0),
    )?;
    let overlap_arrays = atomic_apot_overlap_arrays_from_states(input, &static_arrays, states)?;
    let core_hole = atomic_core_hole_columns_from_states(input, config_inp, states)?;
    let energy_scalars =
        atomic_apot_energy_scalars_from_states(input, config_inp, states, &overlap_arrays)?;
    let amplitude_reduction =
        atomic_apot_amplitude_reduction_from_states(input, config_inp, states)?;
    let total_potential = no_scf_pot_total_potential(input, &overlap_arrays)?;
    let valence_potential = no_scf_pot_valence_potential(input, &overlap_arrays, &total_potential)?;
    let radii = no_scf_pot_radial_state(input, &overlap_arrays, &total_potential)?;
    let atomic_numbers = no_scf_pot_atomic_numbers(input, unique_count)?;
    let potential_multiplicities =
        generated_pot_potential_multiplicities(input, geom, unique_count)?;
    let ionization = no_scf_pot_ionization(input, unique_count)?;
    let overlap_factors = no_scf_pot_overlap_factors(input, unique_count)?;
    let total_charge =
        no_scf_pot_total_charge(&atomic_numbers, &potential_multiplicities, &ionization)?;
    let norman_charges =
        Array1::from_shape_fn(unique_count, |potential| atomic_numbers[potential] as f64);
    let projected = no_scf_pot_projected_state(
        input,
        &static_arrays,
        &radii,
        &overlap_arrays,
        &total_potential,
        &valence_potential,
        potential_multiplicities.view(),
        total_charge,
    )?;
    let fermi = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: projected.interstitial_density,
        interstitial_potential: projected.interstitial_potential,
    })
    .context("failed to calculate POT no-SCF Fermi level")?;
    let core_valence = no_scf_pot_core_valence_selection(
        input,
        states,
        &atomic_numbers,
        projected.interstitial_potential,
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
            average_norman_radius: no_scf_pot_average_norman_radius(
                radii.norman_radii.view(),
                potential_multiplicities.view(),
            )?,
            fermi_level: fermi.chemical_potential,
            interstitial_potential: projected.interstitial_potential,
            interstitial_density: projected.interstitial_density,
            edge_position: energy_scalars.edge_energy,
            amplitude_reduction,
            relaxation_energy: energy_scalars.relaxation_energy,
            plasmon_frequency: projected.interstitial_density.sqrt(),
            core_valence_energy: core_valence.core_valence_energy,
            density_radius: fermi.density_parameter,
            fermi_momentum: fermi.fermi_momentum,
            total_charge,
            total_volume: pot_output_total_volume_bohr3(input, projected.interstitial_volume)?,
        },
        muffin_tin_indices: radii.muffin_tin_indices,
        muffin_tin_radii: radii.muffin_tin_radii,
        norman_indices: radii.norman_indices,
        atomic_numbers,
        kappa,
        norman_radii: radii.norman_radii,
        overlap_factors: overlap_factors.clone(),
        max_overlap_factors: overlap_factors,
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
        total_potential: projected.total_potential.clone(),
        valence_density,
        valence_potential: projected.valence_potential,
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
struct NoScfPotRadii {
    muffin_tin_indices: Array1<usize>,
    muffin_tin_radii: Array1<f64>,
    norman_indices: Array1<usize>,
    norman_radii: Array1<f64>,
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
                no_scf_pot_ground_state_vxc(
                    input,
                    density,
                    overlap.magnetization_density[(row, potential)],
                    row,
                    potential,
                )?
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
            let magnetization = overlap.magnetization_density[(row, potential)];
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
                let valence_spin_fraction_twice = if valence_density > 0.0 {
                    1.0 + magnetization * density / valence_density
                } else {
                    1.0
                };
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
                let magnetized_density = density * (1.0 + magnetization);
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

fn no_scf_pot_radial_state(
    input: &PotInput,
    overlap: &AtomicApotOverlapArrays,
    total_potential: &Array2<f64>,
) -> Result<NoScfPotRadii> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        unique_count == overlap.norman_radii.len(),
        "POT no-SCF Norman radius count {} does not match potential count {unique_count}",
        overlap.norman_radii.len()
    );
    ensure!(
        overlap.overlapped_density.ncols() == unique_count
            && total_potential.ncols() == unique_count,
        "POT no-SCF radial tables must have one column per potential"
    );

    let mut muffin_tin_indices = Array1::<usize>::zeros(unique_count);
    let mut muffin_tin_radii = Array1::<f64>::zeros(unique_count);
    let mut norman_indices = Array1::<usize>::zeros(unique_count);
    let mut norman_radii = Array1::<f64>::zeros(unique_count);

    for potential in 0..unique_count {
        let norman_radius = overlap.norman_radii[potential];
        ensure!(
            norman_radius.is_finite() && norman_radius > 0.0,
            "POT no-SCF Norman radius for potential {potential} must be positive and finite, got {norman_radius}"
        );
        let minimum_muffin_tin_index = if unique_count > 1 {
            POT_NO_SCF_MUFFIN_TIN_WINDOW
        } else {
            1
        };
        let mut indices = None;
        let mut last_error = None;
        for muffin_tin_radius in no_scf_pot_muffin_tin_radius_candidates(
            norman_radius,
            input.potentials[potential].folp,
        )? {
            match overlap_density_indices(OverlapDensityIndicesInput {
                overlapped_density: overlap.overlapped_density.column(potential),
                muffin_tin_radius,
                norman_radius,
            }) {
                Ok(value) => {
                    if value.muffin_tin_index < minimum_muffin_tin_index {
                        continue;
                    }
                    indices = Some(value);
                    break;
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }
        let indices = match indices {
            Some(indices) => indices,
            None => {
                let error = last_error.context("POT no-SCF radius candidate list was empty")?;
                return Err(error).with_context(|| {
                    format!("failed to locate POT no-SCF radial indices for potential {potential}")
                });
            }
        };
        ensure!(
            indices.norman_index + 1 < ATOM_RADIAL_POINTS,
            "POT no-SCF Norman index {} for potential {potential} leaves no upper grid point for shell averaging",
            indices.norman_index
        );
        muffin_tin_indices[potential] = indices.muffin_tin_index;
        muffin_tin_radii[potential] = indices.muffin_tin_radius;
        norman_indices[potential] = indices.norman_index;
        norman_radii[potential] = indices.norman_radius;
    }

    Ok(NoScfPotRadii {
        muffin_tin_indices,
        muffin_tin_radii,
        norman_indices,
        norman_radii,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct NoScfPotProjectedState {
    total_potential: Array2<f64>,
    valence_potential: Array2<f64>,
    interstitial_potential: f64,
    interstitial_density: f64,
    interstitial_volume: f64,
}

#[allow(clippy::too_many_arguments)]
fn no_scf_pot_projected_state(
    input: &PotInput,
    static_arrays: &AtomicApotStaticArrays,
    radii: &NoScfPotRadii,
    overlap: &AtomicApotOverlapArrays,
    total_potential: &Array2<f64>,
    valence_potential: &Array2<f64>,
    potential_multiplicities: ArrayView1<'_, f64>,
    total_charge: f64,
) -> Result<NoScfPotProjectedState> {
    let unique_count = apot_unique_potential_count(input)?;
    ensure!(
        valence_potential.dim() == total_potential.dim(),
        "POT no-SCF valence potential shape {:?} does not match total potential shape {:?}",
        valence_potential.dim(),
        total_potential.dim()
    );
    if unique_count == 1 {
        let interstitial = interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: total_potential.column(0),
            overlapped_density: overlap.overlapped_density.column(0),
            muffin_tin_radius: radii.muffin_tin_radii[0],
            muffin_tin_index: radii.muffin_tin_indices[0],
            wigner_seitz_radius: radii.norman_radii[0],
            wigner_seitz_index: radii.norman_indices[0],
        })
        .context("failed to calculate POT no-SCF interstitial shell values")?;
        return Ok(NoScfPotProjectedState {
            total_potential: total_potential.clone(),
            valence_potential: valence_potential.clone(),
            interstitial_potential: interstitial.interstitial_potential,
            interstitial_density: interstitial.interstitial_density,
            interstitial_volume: if input.scattering.totvol > 0.0 {
                pot_input_total_volume_bohr3(input)?
            } else {
                interstitial.shell_volume
            },
        });
    }

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
    let near_neighbor_flags = Array1::<bool>::from_elem(unique_count, false);
    let interstitial_selector = usize::try_from(input.run.inters)
        .context("POT no-SCF interstitial selector cannot be represented as usize")?;
    let initial_volume = no_scf_pot_initial_interstitial_volume(
        input,
        radii.norman_radii.view(),
        potential_multiplicities,
    )?;

    let overlap_matrix = muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
        highest_potential_index: unique_count - 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        potential_multiplicities,
        explicit_overlaps: &explicit_overlap_refs,
        muffin_tin_indices: radii.muffin_tin_indices.view(),
        muffin_tin_radii: radii.muffin_tin_radii.view(),
        norman_radii: radii.norman_radii.view(),
        near_neighbor_flags: near_neighbor_flags.view(),
        interstitial_selector,
        interstitial_volume: initial_volume,
    })
    .context("failed to assemble POT no-SCF muffin-tin overlap matrix")?;

    let projected_potential = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
        highest_potential_index: unique_count - 1,
        values: total_potential.view(),
        radii: overlap_matrix.radii.view(),
        potential_multiplicities,
        norman_indices: radii.norman_indices.view(),
        muffin_tin_indices: radii.muffin_tin_indices.view(),
        muffin_tin_radii: radii.muffin_tin_radii.view(),
        norman_radii: radii.norman_radii.view(),
        near_neighbor_flags: near_neighbor_flags.view(),
        overlap_matrix: &overlap_matrix,
        interstitial_selector,
        interstitial_value: 0.0,
        mode: MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
    })
    .context("failed to project POT no-SCF total potential into muffin tins")?;
    let projected_valence_potential = if no_scf_pot_exchange_branch(input.control.ixc) >= 5 {
        project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            highest_potential_index: unique_count - 1,
            values: valence_potential.view(),
            radii: overlap_matrix.radii.view(),
            potential_multiplicities,
            norman_indices: radii.norman_indices.view(),
            muffin_tin_indices: radii.muffin_tin_indices.view(),
            muffin_tin_radii: radii.muffin_tin_radii.view(),
            norman_radii: radii.norman_radii.view(),
            near_neighbor_flags: near_neighbor_flags.view(),
            overlap_matrix: &overlap_matrix,
            interstitial_selector,
            interstitial_value: 0.0,
            mode: MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
        })
        .context("failed to project POT no-SCF valence potential into muffin tins")?
        .values
    } else {
        projected_potential.values.clone()
    };

    let projected_density = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
        highest_potential_index: unique_count - 1,
        values: overlap.overlapped_density.view(),
        radii: overlap_matrix.radii.view(),
        potential_multiplicities,
        norman_indices: radii.norman_indices.view(),
        muffin_tin_indices: radii.muffin_tin_indices.view(),
        muffin_tin_radii: radii.muffin_tin_radii.view(),
        norman_radii: radii.norman_radii.view(),
        near_neighbor_flags: near_neighbor_flags.view(),
        overlap_matrix: &overlap_matrix,
        interstitial_selector,
        interstitial_value: 0.0,
        mode: MuffinTinOverlapProjectionMode::Density { total_charge },
    })
    .context("failed to project POT no-SCF density into muffin tins")?;
    ensure!(
        overlap_matrix.interstitial_volume.is_finite() && overlap_matrix.interstitial_volume > 0.0,
        "POT no-SCF interstitial volume must be positive and finite, got {}",
        overlap_matrix.interstitial_volume
    );
    let interstitial_density =
        projected_density.interstitial_value / overlap_matrix.interstitial_volume;
    ensure!(
        interstitial_density.is_finite() && interstitial_density > 0.0,
        "POT no-SCF projected interstitial density must be positive and finite, got {interstitial_density}"
    );

    Ok(NoScfPotProjectedState {
        total_potential: projected_potential.values,
        valence_potential: projected_valence_potential,
        interstitial_potential: projected_potential.interstitial_value,
        interstitial_density,
        interstitial_volume: overlap_matrix.interstitial_volume,
    })
}

fn no_scf_pot_muffin_tin_radius_candidates(
    norman_radius: f64,
    overlap_factor: f64,
) -> Result<Vec<f64>> {
    ensure!(
        overlap_factor.is_finite() && overlap_factor > 0.0,
        "POT no-SCF overlap factor must be positive and finite, got {overlap_factor}"
    );
    let divisors = [overlap_factor.max(1.05), 1.25, 1.5, 2.0, 3.0, 5.0, 8.0];
    let mut radii = Vec::new();
    for divisor in divisors {
        let radius = norman_radius / divisor;
        ensure!(
            radius.is_finite() && radius > 0.0 && radius < norman_radius,
            "POT no-SCF muffin-tin radius {radius} must lie inside Norman radius {norman_radius}"
        );
        if !radii
            .iter()
            .any(|candidate: &f64| (*candidate - radius).abs() <= f64::EPSILON)
        {
            radii.push(radius);
        }
    }
    Ok(radii)
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

fn no_scf_pot_initial_interstitial_volume(
    input: &PotInput,
    norman_radii: ArrayView1<'_, f64>,
    potential_multiplicities: ArrayView1<'_, f64>,
) -> Result<f64> {
    if input.scattering.totvol > 0.0 {
        return pot_input_total_volume_bohr3(input);
    }
    ensure!(
        norman_radii.len() == potential_multiplicities.len(),
        "POT no-SCF volume arrays have mismatched lengths rnrm={}, xnatph={}",
        norman_radii.len(),
        potential_multiplicities.len()
    );
    let mut volume = 0.0;
    for potential in 0..norman_radii.len() {
        volume += potential_multiplicities[potential] * norman_radii[potential].powi(3) / 3.0;
    }
    ensure!(
        volume.is_finite() && volume > 0.0,
        "POT no-SCF fallback interstitial volume must be positive and finite, got {volume}"
    );
    Ok(volume)
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

fn pot_output_total_volume_bohr3(input: &PotInput, interstitial_volume: f64) -> Result<f64> {
    if input.scattering.totvol > 0.0 {
        pot_input_total_volume_bohr3(input)
    } else {
        ensure!(
            interstitial_volume.is_finite() && interstitial_volume > 0.0,
            "POT output interstitial volume must be positive and finite, got {interstitial_volume}"
        );
        Ok(interstitial_volume)
    }
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
            let mut energy = state.scf.orbital_energies[orbital];
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
    let radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);
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
    geom: &GeomDat,
    unique_count: usize,
) -> Result<Array1<f64>> {
    if input.run.nscmt > 0 {
        return pot_input_potential_multiplicities(input, unique_count);
    }
    no_scf_pot_geometry_potential_multiplicities(input, geom, unique_count)
}

fn pot_input_potential_multiplicities(
    input: &PotInput,
    unique_count: usize,
) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT no-SCF pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    let mut values = Array1::<f64>::zeros(unique_count);
    for potential in 0..unique_count {
        let multiplicity = input.potentials[potential].xnatph;
        ensure!(
            multiplicity.is_finite() && multiplicity > 0.0,
            "POT no-SCF xnatph for potential {potential} must be positive and finite, got {multiplicity}"
        );
        values[potential] = multiplicity;
    }
    Ok(values)
}

fn no_scf_pot_geometry_potential_multiplicities(
    input: &PotInput,
    geom: &GeomDat,
    unique_count: usize,
) -> Result<Array1<f64>> {
    ensure!(
        input.potentials.len() == unique_count,
        "POT no-SCF pot.inp has {} potential row(s), expected {unique_count}",
        input.potentials.len()
    );
    ensure!(
        geom.nph + 1 == unique_count,
        "POT no-SCF geom.dat nph={} does not match pot.inp nph={}",
        geom.nph,
        input.control.nph
    );
    ensure!(
        geom.nat == geom.atoms.len(),
        "POT no-SCF geom.dat nat {} does not match row count {}",
        geom.nat,
        geom.atoms.len()
    );

    let mut values = Array1::<f64>::zeros(unique_count);
    for (atom_index, atom) in geom.atoms.iter().enumerate() {
        ensure!(
            atom.iph >= 0,
            "POT no-SCF geom.dat atom {} has negative potential index {}",
            atom_index + 1,
            atom.iph
        );
        let potential = usize::try_from(atom.iph)
            .context("POT no-SCF atom potential index cannot be represented as usize")?;
        ensure!(
            potential < unique_count,
            "POT no-SCF geom.dat atom {} potential {potential} exceeds nph={}",
            atom_index + 1,
            unique_count - 1
        );
        values[potential] += 1.0;
    }

    for potential in 0..unique_count {
        ensure!(
            values[potential] > 0.0,
            "POT no-SCF geom.dat has no atom for potential {potential}"
        );
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

fn no_scf_pot_average_norman_radius(
    norman_radii: ArrayView1<'_, f64>,
    potential_multiplicities: ArrayView1<'_, f64>,
) -> Result<f64> {
    ensure!(
        norman_radii.len() == potential_multiplicities.len(),
        "POT no-SCF average Norman radius arrays have mismatched lengths rnrm={}, xnatph={}",
        norman_radii.len(),
        potential_multiplicities.len()
    );
    let mut weighted_sum = 0.0;
    let mut multiplicity_sum = 0.0;
    for potential in 0..norman_radii.len() {
        let radius = norman_radii[potential];
        let multiplicity = potential_multiplicities[potential];
        ensure!(
            radius.is_finite() && radius > 0.0,
            "POT no-SCF Norman radius for average must be positive and finite, got {radius}"
        );
        ensure!(
            multiplicity.is_finite() && multiplicity > 0.0,
            "POT no-SCF multiplicity for average must be positive and finite, got {multiplicity}"
        );
        weighted_sum += radius * multiplicity;
        multiplicity_sum += multiplicity;
    }
    ensure!(
        multiplicity_sum > 0.0,
        "POT no-SCF average Norman radius requires positive multiplicity"
    );
    Ok(weighted_sum / multiplicity_sum)
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
    Ok((large_component, small_component))
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
    let spin_density = Array2::<f64>::zeros((ATOM_RADIAL_POINTS, unique_count));
    for potential_index in 0..unique_count {
        let state = &states[potential_index];
        atomic_apot_ensure_overlap_state(potential_index, state)?;
        for row in 0..ATOM_RADIAL_POINTS {
            electron_density[(row, potential_index)] = state.scf.density_4pi[row];
            valence_density[(row, potential_index)] = state.scf.valence_density_4pi[row];
            coulomb_potential[(row, potential_index)] = state.scf.coulomb_potential[row];
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

    Ok(AtomicApotOverlapArrays {
        norman_radii,
        magnetization_density,
        overlapped_density,
        overlapped_valence_density,
        overlapped_coulomb_potential,
    })
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
    edge_energy +=
        states[0].scf.coulomb_potential[0] - overlap_arrays.overlapped_coulomb_potential[(0, 0)];

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
    let atomic_number = atomic_number_for_apot_state(input, column)?;
    let orbital_powers = atomic_origin_powers(atomic_number, &kappas)?;
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })
    .context("failed to generate ATOM total-energy Coulomb coefficients from state")?;
    let radii = apot_core_hole_radii(ATOM_RADIAL_POINTS);
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
        step: APOT_CORE_HOLE_GRID_STEP,
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

#[allow(dead_code)]
fn atomic_first_radius_times_charge(atomic_number: usize) -> f64 {
    atomic_number as f64 * ATOM_FIRST_RADIUS_LOG.exp()
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
    let radii = apot_core_hole_radii(ATOM_RADIAL_POINTS);
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
        radial_step: APOT_CORE_HOLE_GRID_STEP,
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
    let orbital_powers = atomic_origin_powers(atomic_number, &kappas)?;
    let active_lengths =
        atomic_active_lengths_from_components(&large_components, &small_components)?;
    let coulomb_coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })
    .context("failed to generate ATOM total-energy Coulomb coefficients")?;
    let radii = apot_core_hole_radii(ATOM_RADIAL_POINTS);

    atomic_total_energy_from_radials(AtomicTotalEnergyRadialInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
        orbital_energies: &orbital_energies,
        coulomb_coefficients: coulomb_coefficients.view(),
        large_small: false,
        step: APOT_CORE_HOLE_GRID_STEP,
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

fn atomic_origin_powers(atomic_number: usize, kappas: &[i32]) -> Result<Vec<f64>> {
    let charge = atomic_number as f64 / ATOM_TOTAL_ENERGY_SPEED_OF_LIGHT;
    kappas
        .iter()
        .enumerate()
        .map(|(index, &kappa)| {
            let kappa_abs = f64::from(kappa.abs());
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
        input,
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
    input: &PotInput,
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

    let initial_kappas = initial_configuration
        .kappa
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let atomic_number = checked_atomic_number(input.potentials[0].z)?;
    let initial_origin_powers = atomic_origin_powers(atomic_number, &initial_kappas)?;
    let active_lengths = initial_state
        .scf
        .active_lengths
        .iter()
        .take(orbital_count)
        .copied()
        .collect::<Vec<_>>();
    let radii = apot_core_hole_radii(ATOM_RADIAL_POINTS);
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

#[cfg(test)]
mod tests {
    use super::{
        has_cached_atomic_output, has_supported_atomic_source_handoff,
        has_supported_config_handoff, run_in_dir,
    };
    use anyhow::{Context, Result};
    use ndarray::{Array1, Array2, Array3, Array4};
    use num_complex::{Complex32, Complex64};
    use refeff_core::{
        BroydenWorkspace, GridError, NormanRadiusInput, PotScfContourRunStatus,
        PotScfContourSourceRows, PotScfIterationStatus, PotScfOuterIterationStatus, PotScfState,
        ScmtEnergyGrid, norman_radius_from_density,
    };
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        APOT_ATOMIC_COULOMB_SECTION_NUMBER, APOT_ATOMIC_DENSITY_SECTION_NUMBER,
        APOT_ATOMIC_KAPPA_SECTION_NUMBER, APOT_ATOMIC_NORB_SECTION_NUMBER,
        APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER, APOT_ATOMIC_ORBITAL_SECTION_START,
        APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, ApotBinValue, ConfigDatData, ConfigDatPotential, FEFF_BOHR_ANGSTROM,
        FeffDocument, FeffInput, Fpf0DatData, Fpf0Oscillator, GeomDat, GeomDatRow, ModuleLogData,
        MtdpData, PotBinData, PotBinScalars, PotInput, PotOverlapShell, PotScfFmsSourceGridHandoff,
        PotScfFovrgSourceGridHandoff, apot_bin_string, apot_core_hole_coulomb_from_density,
        apot_core_hole_radii, geom_dat_string, parse_apot_bin, pot_bin_string,
        potential_dat_outputs_from_bins, rdinp, read_apot_bin, read_config_dat, read_fort16,
        read_fpf0_dat, read_module_log_dat, read_pot_bin, write_apot_bin, write_config_dat,
        write_fpf0_dat, write_module_log_dat, write_mtdp, write_pot_bin,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn atomic_module_skips_disabled_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 0)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_advertise_apot_sidecar_without_pot_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        std::fs::write(temp.path().join("pot.inp"), "not a pot.inp handoff\n")?;

        assert!(!has_cached_atomic_output(temp.path())?);
        assert!(!has_supported_atomic_source_handoff(temp.path())?);
        assert!(!has_supported_config_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed pot.inp should fail through the explicit ATOMIC runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("pot.inp"), "{chain}");
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, apot);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, pot);
        assert!(!temp.path().join("config.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_requires_geometry_before_source_apot_generation() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled ATOM should require source geometry")?;

        assert!(
            error
                .to_string()
                .contains("ATOM source apot.bin generation requires geom.dat handoff")
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_config_before_missing_geometry_error() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("enabled ATOM should still require source geometry")?;

        assert!(
            error
                .to_string()
                .contains("ATOM source apot.bin generation requires geom.dat handoff")
        );
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_supported_config_handoff_without_apot_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        assert!(has_supported_config_handoff(temp.path())?);
        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_generates_supported_config_handoff_from_pot_input_without_pot_bin()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;

        assert!(!has_supported_atomic_source_handoff(temp.path())?);
        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        assert!(has_supported_config_handoff(temp.path())?);
        assert!(!has_cached_atomic_output(temp.path())?);
        assert!(!temp.path().join("apot.bin").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_generates_full_apot_from_source_handoffs_without_cached_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_atomic_source_handoffs(temp.path())?;

        assert!(has_supported_atomic_source_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            1
        );
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert_eq!(
            apot.sections
                .iter()
                .map(|section| section.section_number)
                .collect::<Vec<_>>(),
            (1..=29).collect::<Vec<_>>()
        );
        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert!(temp.path().join("log1.dat").is_file());
        assert!(has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_generates_full_apot_from_geometry_source_handoff_without_pot_bin() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_beryllium_atomic_geometry_source_handoffs(temp.path())?;

        assert!(has_supported_atomic_source_handoff(temp.path())?);
        assert!(!temp.path().join("pot.bin").exists());

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            1
        );
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert_eq!(
            apot.sections
                .iter()
                .map(|section| section.section_number)
                .collect::<Vec<_>>(),
            (1..=29).collect::<Vec<_>>()
        );
        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert!(temp.path().join("log1.dat").is_file());
        assert!(has_cached_atomic_output(temp.path())?);
        assert!(!temp.path().join("pot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_generates_finite_nucleus_apot_from_geometry_source_handoff_without_pot_bin()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.finite_nucleus = true;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        let finite_states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        assert!(
            finite_states
                .iter()
                .all(|state| state.initial_orbitals.nucleus_index > 1)
        );
        let expected = super::generated_atomic_apot_bin_from_sources(
            &super::AtomicCachePaths::new(temp.path()),
            &input,
        )?;

        assert!(has_supported_atomic_source_handoff(temp.path())?);
        assert!(!temp.path().join("pot.bin").exists());

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            1
        );
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert_eq!(apot, rendered_apot_bin(&expected)?);
        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert!(temp.path().join("log1.dat").is_file());
        assert!(has_cached_atomic_output(temp.path())?);
        assert!(!temp.path().join("pot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_writes_highz_finite_nucleus_diagnostic() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.finite_nucleus = true;
        input.control.ipr1 = 5;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let atom = std::fs::read_to_string(temp.path().join("atom00.dat"))?;
        let row = atom
            .lines()
            .find(|line| line.trim_start().starts_with("1s"))
            .context("atom00.dat is missing its 1s orbital row")?;
        let binding_energy_ev = row
            .split_whitespace()
            .nth(2)
            .context("atom00.dat 1s row is missing its binding energy")?
            .parse::<f64>()
            .context("atom00.dat 1s binding energy is not numeric")?;
        let highz_reference_ev = 1.287_986e2_f64;
        let relative_error = (binding_energy_ev - highz_reference_ev).abs() / highz_reference_ev;
        assert!(
            relative_error <= 1.0e-3,
            "Be finite-nucleus 1s binding energy {binding_energy_ev:.8e} differs from the pinned FEFF HIGHZ reference {highz_reference_ev:.8e} by {relative_error:.3e}"
        );
        Ok(())
    }

    #[test]
    fn atomic_finite_nucleus_binding_energies_match_highz_reference_range() -> Result<()> {
        let Some(report_path) = reference_highz_report() else {
            crate::require_fixture!("ATOM finite-nucleus HIGHZ reference report; source not found");
        };
        let report = std::fs::read_to_string(&report_path)
            .with_context(|| format!("failed to read {}", report_path.display()))?;

        for atomic_number in [4_usize, 29, 79, 92] {
            let reference_ev = highz_finite_binding_energy(&report, atomic_number)?;
            let input = highz_pot_input(atomic_number)?;
            let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))
                .with_context(|| format!("failed to solve HIGHZ Z={atomic_number}"))?;
            let tabulation = super::atomic_tabulation_from_state(
                states
                    .first()
                    .context("HIGHZ atomic solve returned no SCF state")?,
            )?;
            let actual_ev = tabulation
                .orbitals
                .iter()
                .find(|orbital| {
                    orbital.principal_quantum_number == 1 && orbital.orbital_label.trim() == "s"
                })
                .context("HIGHZ atomic tabulation is missing its 1s orbital")?
                .binding_energy_ev;
            let relative_error = (actual_ev - reference_ev).abs() / reference_ev;
            assert!(
                relative_error <= 1.0e-3,
                "Z={atomic_number} finite-nucleus 1s binding energy {actual_ev:.8e} differs from pinned FEFF HIGHZ {reference_ev:.8e} by {relative_error:.3e}"
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_finite_nucleus_reports_upstream_z119_matching_failure() -> Result<()> {
        let input = highz_pot_input(119)?;
        let error = super::generated_atomic_scf_states(&input, Path::new("config.inp"))
            .err()
            .context("HIGHZ Z=119 unexpectedly completed its production SCF path")?;
        let source = error
            .chain()
            .find_map(|cause| cause.downcast_ref::<refeff_core::AtomMathError>())
            .context("HIGHZ Z=119 failure chain has no typed atomic source")?;
        assert_eq!(
            source,
            &refeff_core::AtomMathError::ScfDiracAttemptFailed { orbital_1based: 32 },
            "unexpected HIGHZ Z=119 typed failure: {error:#}"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_ybco_reference_apot_from_no_scf_pot_handoff() -> Result<()> {
        let Some(reference_dir) = reference_exafs_ybco_dir()? else {
            crate::require_fixture!("ATOM YBCO APOT source reference test; source not found");
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let input = super::read_input(temp.path())?;
        let caches = super::AtomicCachePaths::new(temp.path());
        let pot = super::generated_no_scf_pot_bin_from_sources(&caches, &input)
            .context("failed to prepare YBCO no-SCF pot.bin handoff")?;
        write_pot_bin(temp.path().join("pot.bin"), &pot)?;

        let apot = super::generated_atomic_apot_bin_from_sources(&caches, &input)
            .context("failed to generate YBCO apot.bin from no-SCF pot.bin handoff")?;

        assert_eq!(pot.potential_count(), 5);
        assert!(apot.sections.len() >= 29);
        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
            (POT_BIN_RADIAL_POINTS, 6)
        );
        potential_dat_outputs_from_bins(&pot, &apot)?;
        Ok(())
    }

    #[test]
    fn atomic_module_derives_no_scf_pot_multiplicities_from_geometry_rows() -> Result<()> {
        let mut input = copper_two_potential_pot_input()?;
        input.potentials[0].xnatph = 100.0;
        input.potentials[1].xnatph = 200.0;

        let multiplicities =
            super::generated_pot_potential_multiplicities(&input, &sample_atomic_geom_dat(), 2)?;

        assert_eq!(multiplicities.to_vec(), vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn atomic_module_derives_scf_pot_multiplicities_from_pot_input() -> Result<()> {
        let mut input = copper_two_potential_pot_input()?;
        input.run.nscmt = 4;
        input.potentials[0].xnatph = 100.0;
        input.potentials[1].xnatph = 200.0;

        let multiplicities =
            super::generated_pot_potential_multiplicities(&input, &sample_atomic_geom_dat(), 2)?;

        assert_eq!(multiplicities.to_vec(), vec![100.0, 200.0]);
        Ok(())
    }

    #[test]
    fn atomic_module_builds_iterative_pot_scf_initial_state_from_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 4;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

        assert_eq!(initial.pot.potential_count(), 1);
        assert_eq!(
            initial.last_indices,
            super::scf_pot_rholie_last_indices(&initial.pot)?
        );
        assert_eq!(
            initial.state.fermi_energy,
            initial.istprm.fermi.chemical_potential
        );
        assert_eq!(initial.pot.scalars.fermi_level, initial.state.fermi_energy);
        assert!(initial.pot.scalars.interstitial_density > 0.0);
        assert_eq!(initial.pot.total_potential, initial.istprm.total_potential);
        assert_eq!(
            initial.state.overlapped_density,
            initial.pot.electron_density
        );
        assert!(initial.energy_grid.active_len > 0);
        assert_eq!(
            initial.energy_grid.energies.len(),
            super::POT_SCMT_MAX_ENERGY_POINTS
        );
        assert_eq!(initial.energy_grid.steps.len(), super::POT_SCMT_FLOOR_COUNT);
        assert_eq!(
            initial.energy_grid.energies[0].re,
            initial.pot.scalars.core_valence_energy
        );
        let fovrg_grid = initial.fovrg_grid.as_ref().unwrap_or_else(|| {
            panic!(
                "missing FOVRG source grid: {}",
                initial
                    .fovrg_grid_unavailable
                    .as_deref()
                    .unwrap_or("missing unavailable reason")
            )
        });
        assert!(initial.fovrg_grid_unavailable.is_none());
        assert_eq!(fovrg_grid.rholie_active_counts, initial.last_indices);
        assert!(
            fovrg_grid
                .radial_active_counts
                .iter()
                .zip(fovrg_grid.rholie_active_counts.iter())
                .all(|(radial_count, rholie_count)| radial_count >= rholie_count)
        );
        let source_energy_count = fovrg_grid.energies_hartree.len();
        assert!(source_energy_count > 0);
        assert_eq!(
            fovrg_grid.wave_numbers.dim(),
            (source_energy_count, initial.pot.potential_count())
        );
        assert_eq!(fovrg_grid.regular_large.dim().0, source_energy_count);
        assert_eq!(
            fovrg_grid.regular_large.dim().1,
            initial.pot.potential_count()
        );
        assert!(fovrg_grid.regular_large.dim().2 > 0);
        assert_eq!(
            fovrg_grid.regular_large.dim().3,
            fovrg_grid.source_radii.len()
        );
        if let Some(fms_grid) = &initial.fms_grid {
            assert!(initial.fms_grid_unavailable.is_none());
            assert_eq!(fms_grid.energies_hartree.len(), source_energy_count);
            assert_eq!(fms_grid.scattering_trace.dim().0, source_energy_count);
            assert_eq!(
                fms_grid.scattering_trace.dim().2,
                initial.pot.potential_count()
            );
        } else {
            let reason = initial
                .fms_grid_unavailable
                .as_deref()
                .expect("missing FMS source-grid reason");
            assert!(!reason.is_empty());
        }
        if let Some(contour_rows) = &initial.contour_rows {
            assert!(initial.contour_rows_unavailable.is_none());
            assert_eq!(contour_rows.source_energies.len(), source_energy_count);
            assert_eq!(contour_rows.scattering_trace.dim().0, source_energy_count);
            assert_eq!(
                contour_rows.scattering_trace.dim().2,
                initial.pot.potential_count()
            );
        } else {
            let reason = initial
                .contour_rows_unavailable
                .as_deref()
                .expect("missing contour source-row reason");
            assert!(!reason.is_empty());
        }
        if let Some(advance) = &initial.state_advance {
            assert!(initial.state_advance_unavailable.is_none());
            assert_eq!(
                advance.iteration.contour.embedded_ldos.ncols(),
                initial.pot.potential_count()
            );
            assert_eq!(
                advance.state.overlapped_density.dim(),
                initial.state.overlapped_density.dim()
            );
        } else {
            let reason = initial
                .state_advance_unavailable
                .as_deref()
                .expect("missing state-advance reason");
            assert!(!reason.is_empty());
        }
        if let Some(next) = &initial.next_iteration {
            assert!(initial.next_iteration_unavailable.is_none());
            assert_eq!(next.iteration, 2);
            assert_eq!(next.pot.potential_count(), initial.pot.potential_count());
            assert_eq!(
                next.last_indices,
                super::scf_pot_rholie_last_indices(&next.pot)?
            );
            assert_eq!(next.pot.scalars.fermi_level, next.state.fermi_energy);
            if let Some(contour_rows) = &next.contour_rows {
                assert!(next.contour_rows_unavailable.is_none());
                assert!(!contour_rows.source_energies.is_empty());
                assert_eq!(
                    contour_rows.scattering_trace.dim().2,
                    next.pot.potential_count()
                );
            } else {
                let reason = next
                    .contour_rows_unavailable
                    .as_deref()
                    .expect("missing next contour source-row reason");
                assert!(!reason.is_empty());
            }
            if let Some(advance) = &next.state_advance {
                assert!(next.state_advance_unavailable.is_none());
                assert_eq!(
                    advance.iteration.contour.embedded_ldos.ncols(),
                    next.pot.potential_count()
                );
                assert_eq!(
                    advance.state.overlapped_density.dim(),
                    next.state.overlapped_density.dim()
                );
            } else {
                let reason = next
                    .state_advance_unavailable
                    .as_deref()
                    .expect("missing next state-advance reason");
                assert!(!reason.is_empty());
            }
        } else {
            let reason = initial
                .next_iteration_unavailable
                .as_deref()
                .expect("missing next-iteration reason");
            assert!(!reason.is_empty());
        }
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("apot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_imports_external_potential_as_scf_initial_state() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 2;
        input.external_pot = true;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        let geom = beryllium_single_potential_geom_dat();
        std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;
        let mut seed_input = input.clone();
        seed_input.run.nscmt = 0;
        seed_input.external_pot = false;
        let seed_pot =
            super::generated_no_scf_pot_bin(&seed_input, Path::new("config.inp"), &geom)?;
        let mtdp = sample_scf_mtdp_data(&seed_pot);
        let mtdp_path = temp.path().join("GeCl4.04.dft.mtdp");
        write_mtdp(&mtdp_path, &mtdp)?;
        let imported_mtdp = refeff_io::read_mtdp(&mtdp_path)?;
        let expected_potential_last = imported_mtdp.atom_potential[(POT_BIN_RADIAL_POINTS - 1, 0)];
        let expected_density0 = imported_mtdp.atom_density[(0, 0)];
        let expected_density2 = imported_mtdp.atom_density[(2, 0)];
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;
        assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

        assert!(initial.external_pot_imported);
        assert!(!initial.restart_pot_imported);
        assert_eq!(initial.pot.muffin_tin_indices[0], 7);
        assert!((initial.pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
        assert!((initial.pot.scalars.interstitial_potential + 0.75).abs() < 1.0e-10);
        assert!((initial.pot.scalars.fermi_level + 0.10).abs() < 1.0e-10);
        assert!((initial.pot.total_potential[(0, 0)] + 1.0).abs() < 1.0e-10);
        assert!((initial.pot.total_potential[(2, 0)] + 1.2).abs() < 1.0e-10);
        assert!(
            (initial.pot.total_potential[(POT_BIN_RADIAL_POINTS - 1, 0)] - expected_potential_last)
                .abs()
                < 1.0e-10
        );
        assert!((initial.pot.electron_density[(0, 0)] - expected_density0).abs() < 1.0e-10);
        assert!((initial.pot.electron_density[(2, 0)] - expected_density2).abs() < 1.0e-10);
        assert_eq!(
            initial.state.overlapped_density,
            initial.pot.electron_density
        );
        assert_eq!(initial.state.fermi_energy, initial.pot.scalars.fermi_level);
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("apot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_applies_start_from_file_after_external_potential_for_scf_initial_state()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 2;
        input.external_pot = true;
        input.start_from_file = true;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        let geom = beryllium_single_potential_geom_dat();
        std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;

        let mut seed_input = input.clone();
        seed_input.run.nscmt = 0;
        seed_input.external_pot = false;
        seed_input.start_from_file = false;
        let seed_pot =
            super::generated_no_scf_pot_bin(&seed_input, Path::new("config.inp"), &geom)?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;

        let mut restart = beryllium_single_potential_pot_bin();
        restart.scalars.fermi_level = -0.125;
        restart.scalars.interstitial_potential = -0.275;
        restart.scalars.interstitial_density = 0.019;
        restart.electron_density.fill(0.023);
        restart.total_potential.fill(-0.41);
        restart.coulomb_potential.fill(123.0);
        restart.valence_density.fill(456.0);
        write_pot_bin(temp.path().join("pot.bin"), &restart)?;
        let restart = read_pot_bin(temp.path().join("pot.bin"))?;

        assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

        let caches = super::AtomicCachePaths::new(temp.path());
        let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

        assert!(initial.external_pot_imported);
        assert!(initial.restart_pot_imported);
        assert_eq!(initial.pot.muffin_tin_indices[0], 7);
        assert!((initial.pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
        assert_eq!(initial.pot.total_potential, restart.total_potential);
        assert_eq!(initial.pot.electron_density, restart.electron_density);
        assert_eq!(initial.pot.scalars.fermi_level, restart.scalars.fermi_level);
        assert_eq!(
            initial.pot.scalars.interstitial_potential,
            restart.scalars.interstitial_potential
        );
        assert_eq!(
            initial.pot.scalars.interstitial_density,
            restart.scalars.interstitial_density
        );
        assert_eq!(initial.state.overlapped_density, restart.electron_density);
        assert_eq!(initial.state.fermi_energy, restart.scalars.fermi_level);
        assert_ne!(initial.pot.coulomb_potential, restart.coulomb_potential);
        assert_ne!(initial.pot.valence_density, restart.valence_density);
        assert!(!temp.path().join("apot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_generates_finite_nucleus_scf_state_from_pot_input() -> Result<()> {
        let point_input = beryllium_pot_input()?;
        let point_states =
            super::generated_atomic_scf_states(&point_input, Path::new("config.inp"))?;
        let mut finite_input = point_input.clone();
        finite_input.finite_nucleus = true;

        let finite_states =
            super::generated_atomic_scf_states(&finite_input, Path::new("config.inp"))?;

        assert!(!finite_states.is_empty());
        assert_eq!(finite_states.len(), point_states.len());
        for (point, finite) in point_states.iter().zip(&finite_states) {
            assert_eq!(point.initial_orbitals.nucleus_index, 1);
            assert!(finite.initial_orbitals.nucleus_index > 1);
            assert!(
                finite.initial_orbitals.radii[0] < super::atomic_first_radius_times_charge(4) / 4.0
            );
            assert_ne!(
                finite.initial_orbitals.radii[0],
                point.initial_orbitals.radii[0]
            );
            assert_ne!(
                finite.initial_orbitals.nuclear_potential[0],
                point.initial_orbitals.nuclear_potential[0]
            );
            assert_ne!(finite.scf.density_4pi[0], point.scf.density_4pi[0]);
            assert_ne!(
                finite.scf.large_components[(0, 0)],
                point.scf.large_components[(0, 0)]
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_module_preserves_saved_scmt_call_state_across_retries() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.finite_nucleus = true;
        input.run.nscmt = 2;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;

        let initial_advance = run
            .initial
            .state_advance
            .as_ref()
            .context("finite-nucleus initial SCF advance should be available")?;
        assert_eq!(
            initial_advance.iteration.contour.status,
            PotScfContourRunStatus::Bracketed,
            "finite-nucleus contour exhausted {} points at energy {:?} with electron delta {}",
            initial_advance.iteration.contour.energy_points_used,
            initial_advance.iteration.contour.current_energy,
            initial_advance.iteration.contour.current_electron_delta
        );
        assert!(
            initial_advance.iteration.contour.energy_points_used
                > super::POT_SCMT_MAX_ENERGY_POINTS * 2,
            "finite-nucleus contour should exercise the expanded adaptive source-row cap"
        );
        assert_eq!(
            initial_advance.outer.status,
            PotScfOuterIterationStatus::RepeatRequired
        );
        assert_eq!(
            run.final_status,
            Some(PotScfOuterIterationStatus::RepeatRequired)
        );
        assert_eq!(run.final_iteration, Some(1));
        assert!(run.prepared_iterations.is_empty());
        assert!(run.final_pot.is_none());
        assert!(run.final_apot.is_none());
        assert!(
            run.final_pot_unavailable
                .as_deref()
                .is_some_and(|reason| reason.contains("FEFF-style start attempt")),
            "missing finite-nucleus repeat retry exhaustion reason: {:?}",
            run.final_pot_unavailable
        );
        Ok(())
    }

    #[test]
    fn atomic_module_imports_start_from_file_pot_as_scf_initial_state() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 2;
        input.start_from_file = true;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let mut restart = beryllium_single_potential_pot_bin();
        restart.scalars.fermi_level = -0.125;
        restart.scalars.interstitial_potential = -0.275;
        restart.scalars.interstitial_density = 0.019;
        restart.electron_density.fill(0.023);
        restart.total_potential.fill(-0.41);
        restart.coulomb_potential.fill(123.0);
        restart.valence_density.fill(456.0);
        write_pot_bin(temp.path().join("pot.bin"), &restart)?;
        let restart = read_pot_bin(temp.path().join("pot.bin"))?;

        assert!(super::can_prepare_scf_pot_initial_state_from_sources_in_dir(temp.path())?);

        let caches = super::AtomicCachePaths::new(temp.path());
        let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;

        assert!(initial.restart_pot_imported);
        assert_eq!(initial.pot.total_potential, restart.total_potential);
        assert_eq!(initial.pot.electron_density, restart.electron_density);
        assert_eq!(initial.pot.scalars.fermi_level, restart.scalars.fermi_level);
        assert_eq!(
            initial.pot.scalars.interstitial_potential,
            restart.scalars.interstitial_potential
        );
        assert_eq!(
            initial.pot.scalars.interstitial_density,
            restart.scalars.interstitial_density
        );
        assert_eq!(initial.state.overlapped_density, restart.electron_density);
        assert_eq!(initial.state.fermi_energy, restart.scalars.fermi_level);
        assert_ne!(initial.pot.coulomb_potential, restart.coulomb_potential);
        assert_ne!(initial.pot.valence_density, restart.valence_density);
        Ok(())
    }

    #[test]
    fn atomic_module_runs_iterative_pot_scf_loop_from_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 4;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;
        let initial_fovrg = run
            .initial
            .fovrg_grid
            .as_ref()
            .expect("missing initial POT SCF FOVRG source grid");
        assert!(
            initial_fovrg
                .phase_amplitudes
                .iter()
                .all(|&value| value != Complex64::new(0.0, 0.0))
        );

        assert!(super::can_write_scf_pot_bin_from_sources_in_dir(
            temp.path()
        )?);
        assert_eq!(run.initial.pot.potential_count(), 1);
        assert!(
            run.final_status
                .is_some_and(super::scf_pot_status_has_final_pot),
            "expected final POT status, got {:?}",
            run.final_status
        );
        assert!(
            run.final_iteration.is_some_and(|iteration| {
                iteration >= 1 && iteration <= input.run.nscmt as usize
            })
        );
        assert!(!run.prepared_iterations.is_empty());
        assert!(
            run.terminal_unavailable.is_none(),
            "unexpected terminal POT SCF boundary: {:?}",
            run.terminal_unavailable
        );
        let (final_contour_rows, final_advance) = run
            .prepared_iterations
            .last()
            .and_then(|prepared| {
                prepared
                    .contour_rows
                    .as_ref()
                    .zip(prepared.state_advance.as_ref())
            })
            .or_else(|| {
                run.initial
                    .contour_rows
                    .as_ref()
                    .zip(run.initial.state_advance.as_ref())
            })
            .expect("missing final POT SCF contour source rows");
        assert!(
            final_contour_rows.source_energies.len() < super::POT_SCMT_MAX_ADAPTIVE_SOURCE_POINTS,
            "corrected core-valence contour should avoid exhausting the adaptive source grid"
        );
        assert!(!final_contour_rows.source_energies.is_empty());
        assert_eq!(
            final_advance.iteration.contour.energy_points_used,
            final_contour_rows.source_energies.len()
        );
        if let Some(iteration) = run.final_iteration {
            assert!(iteration >= 1 && iteration <= input.run.nscmt as usize);
        }
        if let Some(status) = run.final_status {
            let expected = run
                .prepared_iterations
                .last()
                .and_then(|prepared| prepared.state_advance.as_ref())
                .map(|advance| advance.outer.status)
                .or_else(|| {
                    run.initial
                        .state_advance
                        .as_ref()
                        .map(|advance| advance.outer.status)
                })
                .expect("final status should have an advance source");
            assert_eq!(status, expected);
        } else {
            assert!(run.terminal_unavailable.is_some());
        }
        for (offset, prepared) in run.prepared_iterations.iter().enumerate() {
            assert_eq!(prepared.iteration, offset + 2);
            assert_eq!(
                prepared.pot.potential_count(),
                run.initial.pot.potential_count()
            );
            assert_eq!(
                prepared.last_indices,
                super::scf_pot_rholie_last_indices(&prepared.pot)?
            );
        }
        if let Some(reason) = &run.terminal_unavailable {
            assert!(!reason.is_empty());
        }
        if matches!(
            run.final_status,
            Some(
                PotScfOuterIterationStatus::Converged
                    | PotScfOuterIterationStatus::ReachedIterationLimit
            )
        ) {
            let final_pot = run.final_pot.as_ref().expect("missing final pot.bin");
            assert!(run.final_pot_unavailable.is_none());
            assert_eq!(
                final_pot.electron_density.dim(),
                run.initial.pot.electron_density.dim()
            );
            assert!(final_pot.raw_text.is_none());
            assert!(!pot_bin_string(final_pot)?.is_empty());
            let final_apot = run.final_apot.as_ref().expect("missing final apot.bin");
            assert!(!apot_bin_string(final_apot)?.is_empty());
            assert!(!potential_dat_outputs_from_bins(final_pot, final_apot)?.is_empty());
        } else {
            assert!(run.final_pot.is_none());
            assert!(run.final_apot.is_none());
            match run.final_pot_unavailable.as_ref() {
                Some(reason) => assert!(!reason.is_empty()),
                None => panic!("missing final pot.bin unavailable reason"),
            }
        }
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("apot.bin").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_builds_true_scf_istprm_from_positive_totvol_reference() -> Result<()> {
        let Some(reference_dir) = reference_bn_true_scf_dir()? else {
            crate::require_fixture!(
                "POT true-SCF istprm reference test; XANES/BN source not found"
            );
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let input = super::read_input(temp.path())?;
        assert!(input.run.nscmt > 0);
        assert!(input.scattering.totvol > 0.0);

        let caches = super::AtomicCachePaths::new(temp.path());
        let geom = super::prepare_scf_pot_initial_state_source_handoff(&caches, &input)?;
        let unique_count = super::apot_unique_potential_count(&input)?;
        let static_arrays = super::atomic_apot_static_arrays_from_source_geometry(
            &input,
            &geom,
            Array1::from_elem(unique_count, 1.0),
        )?;
        let pot = super::generated_no_scf_pot_bin(&input, &caches.config_inp, &geom)?;

        let converted_volume = super::pot_input_total_volume_bohr3(&input)?;
        assert!(converted_volume > input.scattering.totvol);
        let istprm = super::scf_pot_istprm_from_initial_state(&input, &static_arrays, &pot)?;

        assert_eq!(istprm.muffin_tin_radii.len(), unique_count);
        assert!(istprm.interstitial_volume > 0.0);
        assert!(istprm.interstitial_density > 0.0);
        assert!(istprm.fermi.chemical_potential.is_finite());
        assert!(
            istprm
                .muffin_tin_radii
                .iter()
                .zip(istprm.norman_radii.iter())
                .all(|(muffin_tin, norman)| muffin_tin.is_finite()
                    && *muffin_tin > 0.0
                    && norman.is_finite()
                    && *norman > *muffin_tin)
        );
        Ok(())
    }

    #[test]
    fn atomic_module_preserves_bn_bounded_scf_final_valence_density() -> Result<()> {
        let Some(reference_dir) = reference_bn_true_scf_dir()? else {
            crate::require_fixture!(
                "POT BN bounded SCF valence-density test; XANES/BN source not found"
            );
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let mut input = super::read_input(temp.path())?;
        input.run.nscmt = 1;

        let caches = super::AtomicCachePaths::new(temp.path());
        let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;
        let advance = run
            .initial
            .state_advance
            .as_ref()
            .context("missing bounded BN initial SCF advance")?;
        let final_pot = run
            .final_pot
            .as_ref()
            .context("missing bounded BN final pot.bin")?;

        assert_eq!(
            run.final_status,
            Some(PotScfOuterIterationStatus::ReachedIterationLimit)
        );
        assert_close(
            advance.outer.fermi_energy * refeff_core::FEFF_HARTREE_EV,
            -10.374_905_860_627,
            1.0e-6,
            "bounded BN first SCMT Fermi energy",
        );
        assert_close(
            advance.outer.charge_distance,
            0.129_419_310_645,
            1.0e-8,
            "bounded BN first SCMT charge distance",
        );
        assert_close(
            advance.outer.partial_charge_distance,
            3.614_303_291_509,
            2.0e-8,
            "bounded BN first SCMT partial charge distance",
        );
        assert_eq!(
            run.initial.pot.scalars.plasmon_frequency,
            run.initial.pot.scalars.interstitial_density.sqrt(),
            "FEFF POT plasmon frequency is sqrt(rhoint)"
        );
        assert_eq!(
            final_pot.scalars.plasmon_frequency, run.initial.pot.scalars.plasmon_frequency,
            "FEFF freezes the pre-SCF plasmon frequency across SCMT updates"
        );
        assert_close(
            run.initial.pot.valence_density[(0, 0)],
            64.89293612,
            1.0e-6,
            "bounded BN initial absorber valence density",
        );
        assert_close(
            advance.iteration.overlapped_valence_density[(0, 0)],
            run.initial.pot.valence_density[(0, 0)],
            1.0e-10,
            "bounded BN SCMT iteration absorber valence density",
        );
        assert_close(
            advance.outer.overlapped_valence_density[(0, 0)],
            run.initial.pot.valence_density[(0, 0)],
            1.0e-10,
            "bounded BN outer absorber valence density",
        );
        assert_close(
            advance.state.overlapped_valence_density[(0, 0)],
            run.initial.pot.valence_density[(0, 0)],
            1.0e-10,
            "bounded BN state absorber valence density",
        );
        assert_close(
            final_pot.valence_density[(0, 0)],
            run.initial.pot.valence_density[(0, 0)],
            1.0e-10,
            "bounded BN final absorber valence density",
        );
        Ok(())
    }

    #[test]
    fn atomic_module_reaches_core_hole_bounded_iteration_limit_from_sources() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_core_hole_pot_input()?;
        input.run.nohole = -1;
        input.run.nscmt = 2;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let run = super::generated_scf_pot_run_from_sources(&caches, &input)?;

        assert_eq!(
            run.final_status,
            Some(PotScfOuterIterationStatus::ReachedIterationLimit)
        );
        assert_eq!(run.final_iteration, Some(2));
        assert_eq!(run.prepared_iterations.len(), 1);
        let prepared = run
            .prepared_iterations
            .last()
            .context("missing prepared bounded iteration")?;
        assert_close(
            run.initial.pot.scalars.core_valence_energy,
            input.scattering.ecv / refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "converted core-hole ecv",
        );
        assert_close(
            prepared.pot.scalars.core_valence_energy,
            run.initial.pot.scalars.core_valence_energy,
            1.0e-12,
            "prepared bounded ecv",
        );
        let advance = prepared
            .state_advance
            .as_ref()
            .context("missing bounded state advance")?;
        assert_eq!(
            advance.outer.status,
            PotScfOuterIterationStatus::ReachedIterationLimit
        );
        assert_eq!(advance.iteration.bad_occupation_count, 0);
        assert!(advance.iteration.density_step.is_some());
        assert!(run.final_pot.is_some());
        assert!(run.final_apot.is_some());
        assert!(run.final_pot_unavailable.is_none());
        assert!(run.terminal_unavailable.is_none());
        Ok(())
    }

    #[test]
    fn atomic_module_updates_scf_retry_controls_like_feff_nstarts() -> Result<()> {
        let mut input = beryllium_core_hole_pot_input()?;
        input.scattering.ecv = -refeff_core::FEFF_HARTREE_EV;
        input.scattering.ca1 = 0.2;

        let next = super::update_scf_pot_retry_controls(&mut input, -0.8, 1)?;

        assert_eq!(next, 2);
        assert_close(
            input.scattering.ecv,
            -0.8 * refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "changed ecv retry",
        );
        assert_close(
            input.scattering.ca1,
            0.2,
            1.0e-12,
            "unchanged first retry ca1",
        );

        let next = super::update_scf_pot_retry_controls(&mut input, -0.79, next)?;

        assert_eq!(next, super::POT_SCF_MAX_START_ATTEMPTS);
        assert_close(
            input.scattering.ecv,
            -0.79 * refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "carried ecv retry",
        );
        assert_close(
            input.scattering.ca1,
            0.04,
            1.0e-12,
            "third-start reduced ca1",
        );

        let mut skipped = beryllium_core_hole_pot_input()?;
        skipped.scattering.ecv = -0.5 * refeff_core::FEFF_HARTREE_EV;
        skipped.scattering.ca1 = 0.04;

        let next = super::update_scf_pot_retry_controls(&mut skipped, -0.49, 1)?;

        assert_eq!(next, super::POT_SCF_MAX_START_ATTEMPTS);
        assert_close(
            skipped.scattering.ca1,
            super::POT_SCF_MIN_RETRY_MIXING,
            1.0e-12,
            "small ecv change skips to reduced mixing",
        );

        let error = super::update_scf_pot_retry_controls(
            &mut skipped,
            -0.48,
            super::POT_SCF_MAX_START_ATTEMPTS,
        )
        .err()
        .context("max FEFF-style start attempt should not advance")?;
        assert!(
            error.to_string().contains("cannot advance"),
            "unexpected max-attempt retry error: {error:?}"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_keeps_second_istprm_density_projection_positive() -> Result<()> {
        use refeff_core::{
            MuffinTinOverlapMatrixInput, MuffinTinOverlapProjectionInput,
            MuffinTinOverlapProjectionMode, muffin_tin_overlap_matrix, overlap_density_indices,
            project_muffin_tin_overlap,
        };

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 4;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let context = super::scf_pot_source_context_from_sources(&caches, &input, true)?;
        let initial = super::scf_pot_initial_state_from_generated_pot(
            temp.path(),
            &input,
            &context.static_arrays,
            &context.config,
            context.pot,
            context.external_source.as_ref(),
            context.restart_pot.as_ref(),
            true,
        )?;
        let advance = initial
            .state_advance
            .as_ref()
            .context("missing first SCMT advance")?;
        if let Some(density_step) = &advance.iteration.density_step {
            assert!(
                density_step
                    .norman_charges
                    .iter()
                    .all(|charge| charge.abs() <= 1.0e-10),
                "first SCMT Norman-charge history should remain neutral for single-potential Be"
            );
        }
        let unique_count = super::apot_unique_potential_count(&input)?;
        let static_arrays = &context.static_arrays;
        let atom_potentials = super::atomic_apot_usize_potential_indices(
            "iphat",
            &static_arrays.atom_potential_indices,
            unique_count,
        )?;
        let atom_positions = super::atomic_apot_core_atom_positions(static_arrays)?;
        let representative_atoms =
            super::atomic_apot_zero_based_model_atoms(static_arrays, unique_count)?;
        let explicit_overlaps = super::no_scf_pot_muffin_tin_overlaps(static_arrays, unique_count)?;
        let explicit_overlap_refs = explicit_overlaps
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let near_neighbor_flags = Array1::<bool>::from_elem(unique_count, false);
        let interstitial_selector = usize::try_from(input.run.inters)?;

        for (label, density) in [
            ("initial", initial.pot.electron_density.view()),
            ("after_scmt", advance.state.overlapped_density.view()),
        ] {
            let valence_density = if label == "initial" {
                initial.pot.valence_density.view()
            } else {
                advance.state.overlapped_valence_density.view()
            };
            let mut muffin_tin_indices = Array1::<usize>::zeros(unique_count);
            let mut norman_indices = Array1::<usize>::zeros(unique_count);
            let mut norman_radii = initial.pot.norman_radii.clone();
            for potential in 0..unique_count {
                let indices = overlap_density_indices(refeff_core::OverlapDensityIndicesInput {
                    overlapped_density: density.column(potential),
                    muffin_tin_radius: initial.pot.muffin_tin_radii[potential],
                    norman_radius: norman_radii[potential],
                })?;
                muffin_tin_indices[potential] = indices.muffin_tin_index;
                norman_indices[potential] = indices.norman_index;
                norman_radii[potential] = indices.norman_radius;
            }
            let volume_seed = if input.scattering.totvol <= 0.0 {
                let mut norman_volume = 0.0;
                let mut interstitial_volume = 0.0;
                for potential in 0..unique_count {
                    norman_volume += initial.pot.potential_multiplicities[potential]
                        * norman_radii[potential].powi(3);
                    interstitial_volume -= initial.pot.potential_multiplicities[potential]
                        * initial.pot.muffin_tin_radii[potential].powi(3);
                }
                4.0 * std::f64::consts::PI / 3.0 * (interstitial_volume + norman_volume)
            } else {
                let mut interstitial_volume = 0.0;
                for potential in 0..unique_count {
                    interstitial_volume -= initial.pot.potential_multiplicities[potential]
                        * initial.pot.muffin_tin_radii[potential].powi(3);
                }
                4.0 * std::f64::consts::PI / 3.0 * interstitial_volume + input.scattering.totvol
            };
            let overlap_matrix = muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
                highest_potential_index: unique_count - 1,
                atom_potentials: atom_potentials.view(),
                atom_positions: atom_positions.view(),
                representative_atoms: representative_atoms.view(),
                potential_multiplicities: initial.pot.potential_multiplicities.view(),
                explicit_overlaps: &explicit_overlap_refs,
                muffin_tin_indices: muffin_tin_indices.view(),
                muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
                norman_radii: norman_radii.view(),
                near_neighbor_flags: near_neighbor_flags.view(),
                interstitial_selector,
                interstitial_volume: volume_seed,
            })?;
            let projected_density = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
                highest_potential_index: unique_count - 1,
                values: density,
                radii: overlap_matrix.radii.view(),
                potential_multiplicities: initial.pot.potential_multiplicities.view(),
                norman_indices: norman_indices.view(),
                muffin_tin_indices: muffin_tin_indices.view(),
                muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
                norman_radii: norman_radii.view(),
                near_neighbor_flags: near_neighbor_flags.view(),
                overlap_matrix: &overlap_matrix,
                interstitial_selector,
                interstitial_value: 0.0,
                mode: MuffinTinOverlapProjectionMode::Density {
                    total_charge: initial.pot.scalars.total_charge,
                },
            })?;
            let projected_valence = project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
                highest_potential_index: unique_count - 1,
                values: valence_density,
                radii: overlap_matrix.radii.view(),
                potential_multiplicities: initial.pot.potential_multiplicities.view(),
                norman_indices: norman_indices.view(),
                muffin_tin_indices: muffin_tin_indices.view(),
                muffin_tin_radii: initial.pot.muffin_tin_radii.view(),
                norman_radii: norman_radii.view(),
                near_neighbor_flags: near_neighbor_flags.view(),
                overlap_matrix: &overlap_matrix,
                interstitial_selector,
                interstitial_value: 0.0,
                mode: MuffinTinOverlapProjectionMode::Density { total_charge: 0.0 },
            })?;
            let outside_charge = projected_density.interstitial_value;
            let inside_charge = initial.pot.scalars.total_charge - outside_charge;
            let valence_inside = -projected_valence.interstitial_value;
            let interstitial_density =
                4.0 * std::f64::consts::PI * outside_charge / overlap_matrix.interstitial_volume;
            assert!(
                outside_charge.is_finite()
                    && inside_charge.is_finite()
                    && valence_inside.is_finite()
                    && interstitial_density.is_finite()
                    && outside_charge > 0.0
                    && inside_charge > 0.0
                    && valence_inside > 0.0
                    && interstitial_density > 0.0,
                "{label} POT density projection must keep positive interstitial charge, got outside_charge={outside_charge}, inside_charge={inside_charge}, valence_inside={valence_inside}, rhoint={interstitial_density}"
            );
        }

        Ok(())
    }

    #[test]
    fn atomic_module_assembles_terminal_scf_final_pot_candidate() -> Result<()> {
        let mut pot = beryllium_single_potential_pot_bin();
        pot.scalars.fermi_level = -0.10;
        pot.electron_density.fill(0.25);
        pot.valence_density.fill(0.05);
        pot.coulomb_potential.fill(-0.20);

        let potential_count = pot.potential_count();
        let mut final_density = pot.electron_density.clone();
        final_density.fill(0.33);
        let mut final_valence = pot.valence_density.clone();
        final_valence.fill(0.11);
        let mut final_coulomb = pot.coulomb_potential.clone();
        final_coulomb.fill(-0.42);
        let occupancy_by_l = Array2::from_shape_fn(pot.valence_occupancy.dim(), |(angular, _)| {
            angular as f64 + 0.25
        });
        let state = PotScfState {
            fermi_energy: -0.32,
            norman_charges: Array1::from_vec(vec![3.75]),
            norman_charge_reference: Array1::from_vec(vec![3.75]),
            occupancy_by_l: occupancy_by_l.clone(),
            overlapped_density: final_density.clone(),
            overlapped_valence_density: final_valence.clone(),
            coulomb_potential: final_coulomb.clone(),
            workspace: BroydenWorkspace::zeros(2, potential_count),
        };
        let config = beryllium_single_potential_config_dat();

        let final_pot = super::scf_pot_final_pot_from_state(
            &pot,
            &state,
            PotScfOuterIterationStatus::Converged,
            &config,
        )?;

        assert_eq!(final_pot.scalars.fermi_level, state.fermi_energy);
        assert_eq!(final_pot.norman_charges, state.norman_charges);
        assert_eq!(final_pot.valence_occupancy, occupancy_by_l);
        assert_eq!(final_pot.electron_density, final_density);
        assert_eq!(final_pot.valence_density, final_valence);
        assert_eq!(final_pot.coulomb_potential, final_coulomb);
        assert_eq!(final_pot.total_potential, pot.total_potential);
        assert_eq!(final_pot.orbital_occupancy[(0, 0)], 2.0);
        assert!(final_pot.raw_text.is_none());
        assert!(pot_bin_string(&final_pot)?.contains("ATOM source APOT Be smoke test"));

        let iteration_limit_pot = super::scf_pot_final_pot_from_state(
            &pot,
            &state,
            PotScfOuterIterationStatus::ReachedIterationLimit,
            &config,
        )?;
        assert_eq!(iteration_limit_pot, final_pot);

        for status in [
            PotScfOuterIterationStatus::NeedsMoreSourcePoints,
            PotScfOuterIterationStatus::RepeatRequired,
            PotScfOuterIterationStatus::NeedsNextIteration,
        ] {
            let error = super::scf_pot_final_pot_from_state(&pot, &state, status, &config)
                .err()
                .with_context(|| {
                    format!("nonterminal SCF status {status:?} should not produce final pot.bin")
                })?;
            assert!(
                error
                    .to_string()
                    .contains("requires converged or iteration-limit SCF status"),
                "{error:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_module_lifts_fovrg_fms_grids_into_pot_contour_source_rows() -> Result<()> {
        let point_count = 2;
        let potential_count = 2;
        let angular_count = 2;
        let radial_count = 7;
        let source_radii = Array1::from_vec(vec![0.09, 0.14, 0.22, 0.34, 0.53, 0.82, 1.27]);
        let energies =
            Array1::from_vec(vec![Complex64::new(0.20, 0.04), Complex64::new(0.28, 0.04)]);
        let mut pot = sample_pot_bin();
        pot.norman_radii = Array1::from_vec(vec![0.64, 0.72]);

        let wave_numbers =
            Array2::from_shape_fn((point_count, potential_count), |(point, potential)| {
                Complex64::new(
                    0.70 + 0.05 * point as f64 + 0.03 * potential as f64,
                    0.06 + 0.01 * potential as f64,
                )
            });
        let regular_large = Array4::from_shape_fn(
            (point_count, potential_count, angular_count, radial_count),
            |(point, potential, angular, radial)| {
                Complex64::new(
                    0.10 + 0.02 * radial as f64 + 0.03 * angular as f64 + 0.01 * point as f64,
                    0.02 - 0.004 * potential as f64 + 0.003 * radial as f64,
                )
            },
        );
        let regular_small = Array4::from_shape_fn(
            (point_count, potential_count, angular_count, radial_count),
            |(point, potential, angular, radial)| {
                Complex64::new(
                    0.015 + 0.003 * radial as f64 + 0.004 * angular as f64,
                    -0.004 + 0.002 * point as f64 + 0.001 * potential as f64,
                )
            },
        );
        let irregular_large = Array4::from_shape_fn(
            (point_count, potential_count, angular_count, radial_count),
            |(point, potential, angular, radial)| {
                Complex64::new(
                    -0.35 + 0.04 * radial as f64 + 0.02 * potential as f64,
                    0.24 - 0.015 * angular as f64 + 0.01 * point as f64,
                )
            },
        );
        let irregular_small = Array4::from_shape_fn(
            (point_count, potential_count, angular_count, radial_count),
            |(point, potential, angular, radial)| {
                Complex64::new(
                    -0.030 + 0.004 * radial as f64 + 0.002 * angular as f64,
                    0.014 - 0.002 * potential as f64 + 0.001 * point as f64,
                )
            },
        );
        let fovrg_grid = PotScfFovrgSourceGridHandoff {
            source_radii: source_radii.clone(),
            energies_hartree: energies.clone(),
            reference_energies_hartree: Array2::zeros((point_count, potential_count)),
            wave_numbers,
            regular_large,
            regular_small,
            irregular_large,
            irregular_small,
            phase_shifts: Array3::zeros((point_count, angular_count, potential_count)),
            phase_amplitudes: Array3::zeros((point_count, angular_count, potential_count)),
            radial_active_counts: Array1::from_vec(vec![radial_count, radial_count]),
            rholie_active_counts: Array1::from_vec(vec![radial_count, radial_count]),
            muffin_tin_indices_1based: Array1::from_vec(vec![4, 4]),
            norman_indices_1based: Array1::from_vec(vec![6, 6]),
            radial_handoffs: Vec::new(),
        };
        let fms_grid = PotScfFmsSourceGridHandoff {
            energies_hartree: energies,
            scattering_trace: Array3::from_shape_fn(
                (point_count, angular_count, potential_count),
                |(point, angular, potential)| {
                    Complex64::new(
                        0.04 + 0.01 * point as f64 + 0.02 * angular as f64,
                        -0.03 + 0.015 * potential as f64,
                    )
                },
            ),
        };

        let rows =
            super::scf_pot_contour_source_rows_from_initial_state(&pot, &fovrg_grid, &fms_grid)?;

        assert_eq!(rows.source_energies, fovrg_grid.energies_hartree);
        assert_eq!(
            rows.scattering_trace.dim(),
            (point_count, angular_count, potential_count)
        );
        assert_eq!(
            rows.scattering_density.dim(),
            (
                point_count,
                POT_BIN_RADIAL_POINTS,
                angular_count,
                potential_count
            )
        );
        assert_eq!(
            rows.embedded_density_source.dim(),
            (point_count, POT_BIN_RADIAL_POINTS, potential_count)
        );
        let density_norm = rows
            .embedded_density_source
            .iter()
            .map(|value| value.norm())
            .sum::<f64>();
        assert!(density_norm > 0.0);
        Ok(())
    }

    #[test]
    fn atomic_module_advances_initial_pot_scf_state_from_contour_rows() -> Result<()> {
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 4;
        let static_arrays = beryllium_single_potential_static_arrays()?;
        let mut pot = beryllium_single_potential_pot_bin();
        pot.valence_occupancy[(0, 0)] = 1.0;
        let potential_count = pot.potential_count();
        let angular_count = pot.valence_occupancy.nrows();
        let radial_count = POT_BIN_RADIAL_POINTS;
        let energy_grid = ScmtEnergyGrid {
            energies: Array1::from_vec(vec![Complex64::new(0.10, 0.02)]),
            steps: Array1::from_vec(vec![0.01]),
            active_len: 1,
            lower_imaginary_count: 1,
            real_axis_count: 0,
            upper_imaginary_count: 0,
        };
        let rows = PotScfContourSourceRows {
            source_energies: Array1::from_vec(vec![energy_grid.energies[0]]),
            scattering_trace: Array3::<Complex32>::zeros((
                energy_grid.active_len,
                angular_count,
                potential_count,
            )),
            scattering_ldos: Array3::<Complex64>::zeros((
                energy_grid.active_len,
                angular_count,
                potential_count,
            )),
            embedded_ldos_source: Array3::<Complex64>::zeros((
                energy_grid.active_len,
                angular_count,
                potential_count,
            )),
            scattering_density: Array4::<Complex64>::zeros((
                energy_grid.active_len,
                radial_count,
                angular_count,
                potential_count,
            )),
            embedded_density_source: Array3::<Complex64>::zeros((
                energy_grid.active_len,
                radial_count,
                potential_count,
            )),
            density_scale: Array3::<Complex64>::zeros((
                energy_grid.active_len,
                angular_count,
                potential_count,
            )),
        };
        let last_indices = Array1::from_vec(vec![radial_count]);
        let workspace = BroydenWorkspace::zeros(input.run.nscmt as usize, potential_count);
        let initial_norman_charges = Array1::zeros(potential_count);
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

        let advance = super::scf_pot_state_advance_from_rows(
            &input,
            &static_arrays,
            &pot,
            &energy_grid,
            &rows,
            last_indices.view(),
            &state,
            1,
            true,
        )?;

        assert_eq!(
            advance.iteration.status,
            PotScfIterationStatus::NeedsMoreSourcePoints
        );
        assert_eq!(
            advance.outer.status,
            PotScfOuterIterationStatus::NeedsMoreSourcePoints
        );
        assert_eq!(advance.iteration.contour.energy_points_used, 1);
        assert_eq!(advance.state.fermi_energy, state.fermi_energy);
        assert_eq!(advance.state.overlapped_density, state.overlapped_density);
        assert_eq!(advance.state.coulomb_potential, state.coulomb_potential);

        let second = super::scf_pot_state_advance_from_rows(
            &input,
            &static_arrays,
            &pot,
            &energy_grid,
            &rows,
            last_indices.view(),
            &state,
            2,
            false,
        )?;
        assert_eq!(
            second.iteration.status,
            PotScfIterationStatus::NeedsMoreSourcePoints
        );
        assert_eq!(
            second.outer.status,
            PotScfOuterIterationStatus::NeedsMoreSourcePoints
        );
        assert_eq!(second.iteration.contour.energy_points_used, 1);
        assert_eq!(second.state.fermi_energy, state.fermi_energy);
        Ok(())
    }

    #[test]
    fn atomic_module_prepares_next_pot_scf_iteration_after_state_advance() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 4;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        let geom = beryllium_single_potential_geom_dat();
        std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;

        let caches = super::AtomicCachePaths::new(temp.path());
        let initial = super::generated_scf_pot_initial_state_from_sources(&caches, &input)?;
        let static_arrays = super::atomic_apot_static_arrays_from_source_geometry(
            &input,
            &geom,
            Array1::from_elem(initial.pot.potential_count(), 1.0),
        )?;
        let mut state = initial.state.clone();
        state.fermi_energy += 1.0e-4;
        state.overlapped_valence_density = state.overlapped_valence_density.mapv(|value| {
            if value.is_finite() {
                value * 0.99
            } else {
                value
            }
        });
        let config = super::generated_config_dat(&input, &temp.path().join("config.inp"))?;

        let next = super::scf_pot_next_iteration_preparation_from_state(
            temp.path(),
            &input,
            &static_arrays,
            &config,
            &initial.pot,
            &state,
            2,
        )?;

        assert_eq!(next.iteration, 2);
        assert_eq!(next.pot.potential_count(), initial.pot.potential_count());
        assert_eq!(next.state.fermi_energy, state.fermi_energy);
        assert_eq!(next.pot.scalars.fermi_level, state.fermi_energy);
        assert_eq!(next.pot.electron_density, state.overlapped_density);
        assert_eq!(next.pot.valence_density, state.overlapped_valence_density);
        assert_eq!(next.pot.coulomb_potential, state.coulomb_potential);
        assert_eq!(next.pot.total_potential, next.istprm.total_potential);
        assert_eq!(
            next.last_indices,
            super::scf_pot_rholie_last_indices(&next.pot)?
        );
        assert!(next.energy_grid.active_len > 0);
        assert_eq!(
            next.energy_grid.energies[0].re,
            next.pot.scalars.core_valence_energy
        );
        assert_eq!(next.energy_grid.steps.len(), super::POT_SCMT_FLOOR_COUNT);
        if let Some(fovrg_grid) = &next.fovrg_grid {
            assert!(next.fovrg_grid_unavailable.is_none());
            assert_eq!(fovrg_grid.rholie_active_counts, next.last_indices);
            assert!(
                fovrg_grid
                    .radial_active_counts
                    .iter()
                    .zip(fovrg_grid.rholie_active_counts.iter())
                    .all(|(radial_count, rholie_count)| radial_count >= rholie_count)
            );
            assert!(fovrg_grid.energies_hartree.len() >= next.energy_grid.active_len);
            assert_eq!(fovrg_grid.wave_numbers.dim().1, next.pot.potential_count());
        } else {
            let reason = next
                .fovrg_grid_unavailable
                .as_deref()
                .expect("missing next FOVRG source-grid reason");
            assert!(!reason.is_empty());
        }
        if let Some(contour_rows) = &next.contour_rows {
            assert!(next.contour_rows_unavailable.is_none());
            assert!(contour_rows.source_energies.len() >= next.energy_grid.active_len);
            assert_eq!(
                contour_rows.scattering_trace.dim().2,
                next.pot.potential_count()
            );
        } else {
            let reason = next
                .contour_rows_unavailable
                .as_deref()
                .expect("missing next contour source-row reason");
            assert!(!reason.is_empty());
        }
        if let Some(advance) = &next.state_advance {
            assert!(next.state_advance_unavailable.is_none());
            assert!(
                next.contour_rows.as_ref().is_some_and(|rows| advance
                    .iteration
                    .contour
                    .energy_points_used
                    <= rows.source_energies.len())
            );
            assert_eq!(
                advance.iteration.contour.embedded_ldos.ncols(),
                next.pot.potential_count()
            );
            assert_eq!(
                advance.state.overlapped_density.dim(),
                next.state.overlapped_density.dim()
            );
        } else {
            let reason = next
                .state_advance_unavailable
                .as_deref()
                .expect("missing next state-advance reason");
            assert!(!reason.is_empty());
        }
        Ok(())
    }

    #[test]
    fn atomic_module_replaces_malformed_apot_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_atomic_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

        assert!(has_supported_atomic_source_handoff(temp.path())?);
        assert!(!has_cached_atomic_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(super::apot_has_section(&apot, 1));
        assert!(super::apot_has_section(
            &apot,
            APOT_ATOMIC_DENSITY_SECTION_NUMBER
        ));
        Ok(())
    }

    #[test]
    fn atomic_module_generates_source_scf_apot_sections_from_pot_input() -> Result<()> {
        let input = beryllium_pot_input()?;

        let apot = super::generated_atomic_scf_apot_bin(&input, Path::new("config.inp"))?;

        assert_eq!(
            apot.sections
                .iter()
                .map(|section| section.section_number)
                .collect::<Vec<_>>(),
            vec![3, 8, 10, 11, 13, 14, 20, 22, 23, 24, 25, 26, 27, 28, 29]
        );
        let apot = rendered_apot_bin(&apot)?;

        let ApotBinPayload::Records(records) =
            &super::apot_section(&apot, APOT_ATOMIC_NORB_SECTION_NUMBER, "norb")?.payload
        else {
            anyhow::bail!("ATOM generated norb should be row records");
        };
        assert_eq!(records.rows.len(), 2);
        assert!(matches!(records.rows[0][0], ApotBinValue::Int(value) if value > 0));
        assert!(matches!(records.rows[1][0], ApotBinValue::Int(value) if value > 0));

        let density = super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?;
        assert_eq!(density.dim(), (POT_BIN_RADIAL_POINTS, 2));
        assert!(density[[0, 0]].is_finite());
        assert!(density[[0, 0]] > 0.0);
        assert!(density[[0, 1]].is_finite());

        let valence = super::real_matrix_section(
            &apot,
            APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER,
            "xnval",
        )?;
        assert_eq!(valence.dim(), (POT_BIN_ORBITALS, 2));

        let energies =
            super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER, "eorb")?;
        assert_eq!(energies.dim(), (POT_BIN_ORBITALS, 2));
        assert!(energies[[0, 0]].is_finite());

        let kappa = super::int_matrix_section(&apot, APOT_ATOMIC_KAPPA_SECTION_NUMBER, "kappa")?;
        assert_eq!(kappa.dim(), (POT_BIN_ORBITALS, 2));
        assert_eq!(kappa[[0, 0]], -1);

        let first_dgc =
            super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START, "dgc")?;
        assert_eq!(first_dgc.dim(), (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS));
        assert!(first_dgc[[0, 0]].is_finite());

        let final_adpc =
            super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START + 7, "adpc")?;
        assert_eq!(final_adpc.dim(), (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS));
        assert!(final_adpc[[0, 0]].is_finite());
        Ok(())
    }

    #[test]
    fn atomic_module_generates_source_scf_apot_sections_for_core_hole_state() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;

        let apot = super::generated_atomic_scf_apot_bin(&input, Path::new("config.inp"))?;
        let apot = rendered_apot_bin(&apot)?;

        let ApotBinPayload::Records(records) =
            &super::apot_section(&apot, APOT_ATOMIC_NORB_SECTION_NUMBER, "norb")?.payload
        else {
            anyhow::bail!("ATOM generated norb should be row records");
        };
        assert_eq!(records.rows.len(), 2);
        assert!(matches!(records.rows[0][0], ApotBinValue::Int(value) if value > 0));
        assert!(matches!(records.rows[1][0], ApotBinValue::Int(value) if value > 0));

        let density = super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?;
        assert!(density[[0, 0]].is_finite());
        assert!(density[[0, 1]].is_finite());
        assert_ne!(density[[0, 0]], density[[0, 1]]);

        let kappa = super::int_matrix_section(&apot, APOT_ATOMIC_KAPPA_SECTION_NUMBER, "kappa")?;
        assert_eq!(kappa[[0, 1]], -1);

        let final_dgc =
            super::real_matrix_section(&apot, APOT_ATOMIC_ORBITAL_SECTION_START + 1, "dgc")?;
        assert!(final_dgc[[0, 0]].is_finite());
        Ok(())
    }

    #[test]
    fn atomic_module_uses_warnion_custom_config_charge_for_scf_states() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let feff = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE WARNION Cu custom configuration smoke test
EDGE K
CONTROL 1 0 0 0 0 0
CONFIG card 1
0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0
WARNION
COREHOLE FSR
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&feff)?;
        let input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        std::fs::write(
            temp.path().join("config.inp"),
            rdinp::config_inp_string(&document)?,
        )?;

        assert!(input.warn_ion);
        assert_eq!(input.config_type, 2);

        let states = super::generated_atomic_scf_states(&input, &temp.path().join("config.inp"))?;

        assert_eq!(states.len(), 3);
        assert!((states[0].occupations.sum() - 28.0).abs() < 1.0e-10);
        assert!((states[1].occupations.sum() - 29.0).abs() < 1.0e-10);
        assert!((states[2].occupations.sum() - 28.0).abs() < 1.0e-10);
        Ok(())
    }

    #[test]
    fn atomic_module_assembles_full_apot_sections_from_source_arrays() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;
        let static_arrays = beryllium_single_potential_static_arrays()?;

        let apot = ApotBinData {
            sections: super::generated_atomic_apot_sections_from_static_arrays(
                &input,
                Path::new("config.inp"),
                &static_arrays,
            )?,
        };
        assert_eq!(
            apot.sections
                .iter()
                .map(|section| section.section_number)
                .collect::<Vec<_>>(),
            (1..=29).collect::<Vec<_>>()
        );
        let apot = rendered_apot_bin(&apot)?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let overlap_arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        let ApotBinPayload::Records(records) = &super::apot_section(&apot, 1, "scalars")?.payload
        else {
            anyhow::bail!("ATOM full apot section 1 should be scalar records");
        };
        assert_eq!(records.rows.len(), 1);
        assert!(matches!(records.rows[0][0], ApotBinValue::Int(0)));
        assert!(matches!(records.rows[0][1], ApotBinValue::Int(1)));
        assert!(matches!(records.rows[0][2], ApotBinValue::Int(1)));
        let ApotBinValue::Real(relaxation_energy) = &records.rows[0][3] else {
            anyhow::bail!("ATOM full apot erelax should be real-valued");
        };
        let ApotBinValue::Real(edge_energy) = &records.rows[0][4] else {
            anyhow::bail!("ATOM full apot emu should be real-valued");
        };
        let ApotBinValue::Real(amplitude_reduction) = &records.rows[0][5] else {
            anyhow::bail!("ATOM full apot s02 should be real-valued");
        };
        assert!(relaxation_energy.is_finite());
        assert!(edge_energy.is_finite());
        assert!(amplitude_reduction.is_finite());
        assert!(
            *amplitude_reduction > 0.0 && *amplitude_reduction <= 1.0,
            "unexpected full apot s02 {amplitude_reduction}"
        );

        let ApotBinPayload::Records(unique_records) =
            &super::apot_section(&apot, 2, "unique potentials")?.payload
        else {
            anyhow::bail!("ATOM full apot section 2 should be unique-potential records");
        };
        assert_eq!(unique_records.rows.len(), 1);
        assert!(matches!(unique_records.rows[0][0], ApotBinValue::Int(4)));
        let ApotBinValue::Real(norman_radius) = &unique_records.rows[0][3] else {
            anyhow::bail!("ATOM full apot rnrm should be real-valued");
        };
        assert_close(
            *norman_radius,
            overlap_arrays.norman_radii[0],
            1.0e-9 * overlap_arrays.norman_radii[0].abs().max(1.0),
            "full apot rnrm",
        );

        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_DENSITY_SECTION_NUMBER, "rho")?.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert_eq!(
            super::real_matrix_section(
                &apot,
                APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER,
                "rhoval"
            )?
            .dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert_eq!(
            super::real_matrix_section(&apot, APOT_ATOMIC_COULOMB_SECTION_NUMBER, "vcoul")?.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert_eq!(
            super::real_matrix_section(&apot, 12, "xnvmu")?.dim(),
            (4, 2)
        );
        assert_eq!(
            super::real_matrix_section(&apot, 17, "edens")?.dim(),
            (POT_BIN_RADIAL_POINTS, 1)
        );
        assert_eq!(
            super::real_matrix_section(&apot, 19, "vclap")?.dim(),
            (POT_BIN_RADIAL_POINTS, 1)
        );
        assert_eq!(
            super::int_matrix_section(&apot, 21, "iorb")?.dim(),
            (POT_BIN_IORB_SLOTS, 2)
        );
        Ok(())
    }

    #[test]
    fn atomic_module_derives_iorb_section_matrix_from_scf_configurations() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;

        let indices = super::atomic_orbital_indices_by_kappa(&input, Path::new("config.inp"))?;

        assert_eq!(indices.dim(), (10, 2));
        assert_eq!(indices[[4, 0]], 2);
        assert_eq!(indices[[4, 1]], 2);
        assert_eq!(indices[[6, 1]], 3);
        for row in 0..10 {
            if row != 4 {
                assert_eq!(indices[[row, 0]], 0);
            }
            if row != 4 && row != 6 {
                assert_eq!(indices[[row, 1]], 0);
            }
        }
        Ok(())
    }

    #[test]
    fn atomic_module_derives_norman_valence_counts_from_scf_configurations() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;
        let configurations =
            super::atomic_scf_configurations_from_pot_input(&input, Path::new("config.inp"))?;

        let counts = super::atomic_norman_valence_counts_by_l(&input, Path::new("config.inp"))?;

        assert_eq!(counts.dim(), (4, 2));
        for (state_index, configuration) in configurations.iter().enumerate() {
            let expected = configuration
                .valence_counts
                .iter()
                .take(configuration.orbital_count)
                .sum::<f64>();
            let actual = counts.column(state_index).sum();
            assert_close(
                actual,
                expected,
                1.0e-12,
                &format!("xnvmu total state {state_index}"),
            );
        }
        assert_eq!(counts[(2, 0)], 0.0);
        assert_eq!(counts[(3, 0)], 0.0);
        assert!(
            counts[(0, 0)] > 0.0,
            "initial Be absorber should carry s-channel valence count"
        );
        assert!(
            counts[(1, 1)] > 0.0,
            "final Be K-edge state should carry p-channel screening valence count"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_converts_generated_pot_core_valence_energy_from_input_ev() -> Result<()> {
        let input = beryllium_pot_input()?;
        let geom = beryllium_single_potential_geom_dat();

        let pot = super::generated_no_scf_pot_bin(&input, Path::new("config.inp"), &geom)?;

        let guard = super::POT_CORVAL_TOLERANCE_EV / refeff_core::FEFF_HARTREE_EV;
        let input_ecv_hartree = input.scattering.ecv / refeff_core::FEFF_HARTREE_EV;
        assert!(pot.scalars.core_valence_energy.is_finite());
        assert_close(
            pot.scalars.core_valence_energy,
            input_ecv_hartree,
            1.0e-12,
            "converted core-valence separation",
        );
        assert!(
            pot.scalars.core_valence_energy <= pot.scalars.interstitial_potential - guard + 1.0e-12,
            "core-valence separation {} should respect vint {} minus FEFF guard {}",
            pot.scalars.core_valence_energy,
            pot.scalars.interstitial_potential,
            guard
        );
        Ok(())
    }

    #[test]
    fn atomic_module_builds_corval_scan_grid_with_feff_spacing() -> Result<()> {
        let grid = super::pot_corval_scan_energy_grid(-1.2, -70.0)?;

        assert!(grid.len() > 2);
        assert_close(
            grid[0].re,
            -70.0 / refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "corval lower energy",
        );
        assert_close(
            grid[grid.len() - 1].re,
            super::POT_CORVAL_HIGH_EV / refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "corval upper energy",
        );
        assert_close(
            grid[0].im,
            super::POT_CORVAL_LDOS_IMAGINARY_EV / refeff_core::FEFF_HARTREE_EV,
            1.0e-12,
            "corval imaginary broadening",
        );
        assert_close(
            (grid[1].re - grid[0].re) * refeff_core::FEFF_HARTREE_EV,
            0.5,
            5.0e-3,
            "corval approximate energy step",
        );
        Ok(())
    }

    #[test]
    fn atomic_module_detects_corval_ldos_peak_from_previous_sample() -> Result<()> {
        let energies = Array1::from_vec(vec![
            Complex64::new(-2.0, 0.05),
            Complex64::new(-1.5, 0.05),
            Complex64::new(-1.0, 0.05),
            Complex64::new(-0.5, 0.05),
        ]);
        let angular_count = 2;
        let potentials = 1;
        let mut embedded_ldos =
            Array3::<Complex64>::zeros((energies.len(), angular_count, potentials));
        let threshold = 1.0
            / (6.0
                * (super::POT_CORVAL_LDOS_IMAGINARY_EV / refeff_core::FEFF_HARTREE_EV)
                * std::f64::consts::PI);
        embedded_ldos[(0, 0, 0)] = Complex64::new(0.0, 0.1);
        embedded_ldos[(1, 0, 0)] = Complex64::new(0.0, threshold + 0.25);
        embedded_ldos[(2, 0, 0)] = Complex64::new(0.0, threshold + 0.10);
        embedded_ldos[(3, 0, 0)] = Complex64::new(0.0, threshold + 0.05);
        embedded_ldos[(0, 1, 0)] = Complex64::new(0.0, 0.1);
        embedded_ldos[(1, 1, 0)] = Complex64::new(0.0, 0.2);
        embedded_ldos[(2, 1, 0)] = Complex64::new(0.0, 0.15);
        embedded_ldos[(3, 1, 0)] = Complex64::new(0.0, 0.05);

        let peaks = super::pot_corval_ldos_peak_energies(
            energies.view(),
            embedded_ldos.view(),
            angular_count,
        )?;

        assert_close(peaks[(0, 0)], -1.5, 1.0e-12, "l=0 corval peak");
        assert!(
            peaks[(1, 0)].is_nan(),
            "below-threshold l=1 channel should not report a peak"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_uses_scf_exchange_selector_for_no_scf_ground_state_potential() -> Result<()> {
        let mut input = beryllium_pot_input()?;
        input.control.ixc = 0;
        let density = 3.0;
        let coulomb = -0.2;
        let density_radius = (density / 3.0_f64).powf(-1.0 / 3.0);
        let overlap = super::AtomicApotOverlapArrays {
            norman_radii: Array1::from_vec(vec![1.5]),
            magnetization_density: Array2::zeros((1, 1)),
            overlapped_density: Array2::from_shape_vec((1, 1), vec![density])?,
            overlapped_valence_density: Array2::from_shape_vec((1, 1), vec![0.5 * density])?,
            overlapped_coulomb_potential: Array2::from_shape_vec((1, 1), vec![coulomb])?,
        };

        input.control.iscfxc = 11;
        let vbh = super::no_scf_pot_total_potential(&input, &overlap)?;
        assert_close(
            vbh[(0, 0)],
            coulomb + refeff_core::von_barth_hedin_potential(density_radius, 1.0)?,
            1.0e-12,
            "vBH no-SCF total potential",
        );

        input.control.iscfxc = 12;
        let pz = super::no_scf_pot_total_potential(&input, &overlap)?;
        assert_close(
            pz[(0, 0)],
            coulomb + refeff_core::perdew_zunger_vxc(density_radius)?,
            1.0e-12,
            "PZ no-SCF total potential",
        );
        assert!((vbh[(0, 0)] - pz[(0, 0)]).abs() > 1.0e-6);
        Ok(())
    }

    #[test]
    fn atomic_module_builds_no_scf_valence_potential_for_high_exchange_selector() -> Result<()> {
        let mut input = beryllium_pot_input()?;
        input.control.iscfxc = 11;
        let density = 3.0;
        let valence_density = 0.375;
        let coulomb = -0.2;
        let overlap = super::AtomicApotOverlapArrays {
            norman_radii: Array1::from_vec(vec![1.5]),
            magnetization_density: Array2::zeros((1, 1)),
            overlapped_density: Array2::from_shape_vec((1, 1), vec![density])?,
            overlapped_valence_density: Array2::from_shape_vec((1, 1), vec![valence_density])?,
            overlapped_coulomb_potential: Array2::from_shape_vec((1, 1), vec![coulomb])?,
        };

        input.control.ixc = 5;
        let total = super::no_scf_pot_total_potential(&input, &overlap)?;
        let valence = super::no_scf_pot_valence_potential(&input, &overlap, &total)?;
        let valence_radius = (valence_density / 3.0_f64).powf(-1.0 / 3.0);
        assert_close(
            valence[(0, 0)],
            coulomb + refeff_core::von_barth_hedin_potential(valence_radius, 1.0)?,
            1.0e-12,
            "EXCHANGE 5 no-SCF valence potential",
        );
        assert!((valence[(0, 0)] - total[(0, 0)]).abs() > 1.0e-6);

        input.control.ixc = 6;
        let high_branch = super::no_scf_pot_valence_potential(&input, &overlap, &total)?;
        let core_radius = ((density - valence_density) / 3.0_f64).powf(-1.0 / 3.0);
        let magnetized_radius = (density / 3.0_f64).powf(-1.0 / 3.0);
        let magnetized_fermi_momentum = refeff_core::FEFF_FERMI_MOMENTUM_FACTOR / magnetized_radius;
        assert_close(
            high_branch[(0, 0)],
            total[(0, 0)]
                - refeff_core::dirac_hara_exchange_potential(
                    core_radius,
                    magnetized_fermi_momentum,
                )?,
            1.0e-12,
            "EXCHANGE 6 no-SCF valence potential",
        );
        assert!((high_branch[(0, 0)] - total[(0, 0)]).abs() > 1.0e-6);
        Ok(())
    }

    #[test]
    fn atomic_module_uses_corval_ldos_peak_overrides_in_selection() -> Result<()> {
        let mut input = beryllium_pot_input()?;
        input.scattering.corval_emin = -70.0;
        let geom = beryllium_single_potential_geom_dat();
        let pot = super::generated_no_scf_pot_bin(&input, Path::new("config.inp"), &geom)?;
        let mut states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        states[0].scf.orbital_energies.fill(0.0);
        states[0].scf.orbital_energies[0] = super::POT_CORVAL_HIGH_EV
            / refeff_core::FEFF_HARTREE_EV
            - 2.0 * super::POT_CORVAL_TOLERANCE_EV / refeff_core::FEFF_HARTREE_EV;
        states[0].valence_occupations[0] = 0.0;
        let atomic_numbers = super::no_scf_pot_atomic_numbers(&input, pot.potential_count())?;
        let base = super::no_scf_pot_core_valence_selection(
            &input,
            &states,
            &atomic_numbers,
            pot.scalars.interstitial_potential,
            None,
        )?;
        let marker = base
            .markers
            .first()
            .copied()
            .context("Be corval selection should expose at least one marker")?;
        let shifted_energy = marker.energy + 0.01;
        let mut peaks = Array2::<f64>::from_elem((4, pot.potential_count()), f64::NAN);
        peaks[(marker.angular, marker.potential)] = shifted_energy;

        let shifted = super::no_scf_pot_core_valence_selection(
            &input,
            &states,
            &atomic_numbers,
            pot.scalars.interstitial_potential,
            Some(peaks.view()),
        )?;
        let shifted_marker = shifted
            .markers
            .iter()
            .find(|candidate| {
                candidate.angular == marker.angular && candidate.potential == marker.potential
            })
            .context("shifted corval marker should still be present")?;

        assert_close(
            shifted_marker.energy,
            shifted_energy,
            1.0e-12,
            "LDOS peak override energy",
        );
        Ok(())
    }

    #[test]
    fn atomic_module_applies_core_valence_reassignment_to_pot_state() -> Result<()> {
        let potentials = 1;
        let mut orbital_occupancy = Array2::<f64>::zeros((POT_BIN_ORBITALS, potentials));
        let mut valence_occupancy = Array2::<f64>::zeros((4, potentials));
        let mut valence_density =
            Array2::<f64>::from_elem((POT_BIN_RADIAL_POINTS, potentials), 0.5);
        let radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);
        let mut large_components =
            Array3::<f64>::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
        let mut small_components =
            Array3::<f64>::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials));
        for radial in 0..POT_BIN_RADIAL_POINTS {
            let radius = radii[radial];
            large_components[(radial, 2, 0)] = radius * 0.25;
            small_components[(radial, 2, 0)] = radius * 0.05;
            large_components[(radial, 1, 0)] = radius * 0.10;
            small_components[(radial, 1, 0)] = radius * 0.02;
        }
        let selection = super::PotCoreValenceSelection {
            core_valence_energy: -0.35,
            markers: vec![
                super::PotCoreValenceMarker {
                    potential: 0,
                    angular: 1,
                    orbital: 2,
                    energy: -0.40,
                    initial_is_valence: false,
                    is_valence: true,
                },
                super::PotCoreValenceMarker {
                    potential: 0,
                    angular: 0,
                    orbital: 0,
                    energy: -0.10,
                    initial_is_valence: true,
                    is_valence: true,
                },
            ],
        };

        super::no_scf_pot_apply_core_valence_selection(
            &selection,
            &mut orbital_occupancy,
            &mut valence_occupancy,
            &mut valence_density,
            large_components.view(),
            small_components.view(),
        )?;

        assert_eq!(valence_occupancy[(1, 0)], 6.0);
        assert_eq!(valence_occupancy[(0, 0)], 0.0);
        assert_eq!(orbital_occupancy[(2, 0)], 4.0);
        assert_eq!(orbital_occupancy[(1, 0)], 2.0);
        assert_eq!(orbital_occupancy[(0, 0)], 0.0);

        let expected_density = 0.5
            + 4.0 * (0.25_f64.powi(2) + 0.05_f64.powi(2))
            + 2.0 * (0.10_f64.powi(2) + 0.02_f64.powi(2));
        assert_close(
            valence_density[(0, 0)],
            expected_density,
            1.0e-12,
            "core-valence density row 1",
        );
        assert_close(
            valence_density[(POT_BIN_RADIAL_POINTS - 1, 0)],
            expected_density,
            1.0e-12,
            "core-valence density final row",
        );
        Ok(())
    }

    #[test]
    fn atomic_module_ignores_high_l_norman_valence_count_channels() -> Result<()> {
        let configuration = super::OrbitalConfiguration {
            orbital_count: 3,
            core_orbital_count: 0,
            projection_orbitals: Array1::zeros(10),
            hole_position: 0,
            principal_quantum_numbers: Array1::from_vec(vec![2, 4, 5]),
            kappa: Array1::from_vec(vec![-1, 3, -5]),
            electron_counts: Array1::from_vec(vec![2.0, 1.5, 0.5]),
            valence_counts: Array1::from_vec(vec![2.0, 1.5, 0.5]),
            spin_magnetization: Array1::zeros(3),
            ionization_orbital: 0,
            screening_orbital: 0,
            last_occupied_orbital: 0,
            template_atomic_number: 4,
            ionicity_delta: 0.0,
        };

        let counts = super::atomic_norman_valence_counts_from_configurations(&[configuration], 1)?;

        assert_eq!(counts.dim(), (4, 1));
        assert_eq!(counts[(0, 0)], 2.0);
        assert_eq!(counts[(3, 0)], 1.5);
        assert_eq!(counts.column(0).sum(), 3.5);
        Ok(())
    }

    #[test]
    fn atomic_module_keeps_unit_amplitude_reduction_without_core_hole() -> Result<()> {
        let input = beryllium_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;

        let s02 = super::atomic_apot_amplitude_reduction_from_states(
            &input,
            Path::new("config.inp"),
            &states,
        )?;

        assert_eq!(s02, 1.0);
        Ok(())
    }

    #[test]
    fn atomic_module_derives_amplitude_reduction_from_relaxed_absorber_overlaps() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let mut swapped = input.clone();
        swapped.run.nohole = -1;
        let swapped_states = super::generated_atomic_scf_states(&swapped, Path::new("config.inp"))?;

        let s02 = super::atomic_apot_amplitude_reduction_from_states(
            &input,
            Path::new("config.inp"),
            &states,
        )?;
        let swapped_s02 = super::atomic_apot_amplitude_reduction_from_states(
            &swapped,
            Path::new("config.inp"),
            &swapped_states,
        )?;

        assert!(s02.is_finite());
        assert!(s02 > 0.0 && s02 <= 1.0, "unexpected s02 {s02}");
        assert_close(swapped_s02, s02, 1.0e-9 * s02.abs().max(1.0), "swapped s02");
        Ok(())
    }

    #[test]
    fn atomic_module_keeps_zero_energy_scalars_without_core_hole() -> Result<()> {
        let input = beryllium_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let static_arrays = beryllium_single_potential_static_arrays()?;
        let overlap_arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        let scalars = super::atomic_apot_energy_scalars_from_states(
            &input,
            Path::new("config.inp"),
            &states,
            &overlap_arrays,
        )?;
        let initial_total = super::atomic_total_energy_from_state(&input, 0, &states[0])?;

        assert_close(
            scalars.initial_total_energy,
            initial_total.total,
            1.0e-9 * initial_total.total.abs().max(1.0),
            "no-hole initial total energy",
        );
        assert_close(
            scalars.final_total_energy,
            initial_total.total,
            1.0e-9 * initial_total.total.abs().max(1.0),
            "no-hole final total energy",
        );
        assert_eq!(scalars.frozen_orbital_energy, 0.0);
        assert_eq!(scalars.relaxation_energy, 0.0);
        assert_eq!(scalars.edge_energy, 0.0);
        Ok(())
    }

    #[test]
    fn atomic_module_derives_energy_scalars_from_generated_states() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let static_arrays = beryllium_single_potential_static_arrays()?;
        let overlap_arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        let scalars = super::atomic_apot_energy_scalars_from_states(
            &input,
            Path::new("config.inp"),
            &states,
            &overlap_arrays,
        )?;
        let state_count = super::apot_state_count(&input)?;
        let (initial_column, final_column) =
            super::atomic_apot_absorber_state_columns(&input, state_count)?;
        let initial_total =
            super::atomic_total_energy_from_state(&input, initial_column, &states[initial_column])?;
        let final_total =
            super::atomic_total_energy_from_state(&input, final_column, &states[final_column])?;
        let frozen_orbital_energy =
            super::atomic_apot_frozen_orbital_energy(&input, Path::new("config.inp"), &states[0])?;
        let adiabatic_edge = final_total.total - initial_total.total;
        let expected_relaxation = -frozen_orbital_energy - adiabatic_edge;
        let expected_edge = if adiabatic_edge <= 0.0 {
            -frozen_orbital_energy
        } else {
            adiabatic_edge
        } + states[0].scf.coulomb_potential[0]
            - overlap_arrays.overlapped_coulomb_potential[(0, 0)];

        assert!(scalars.frozen_orbital_energy.is_finite());
        assert!(scalars.frozen_orbital_energy < 0.0);
        assert_close(
            scalars.initial_total_energy,
            initial_total.total,
            1.0e-9 * initial_total.total.abs().max(1.0),
            "initial total energy",
        );
        assert_close(
            scalars.final_total_energy,
            final_total.total,
            1.0e-9 * final_total.total.abs().max(1.0),
            "final total energy",
        );
        assert_close(
            scalars.frozen_orbital_energy,
            frozen_orbital_energy,
            1.0e-12 * frozen_orbital_energy.abs().max(1.0),
            "frozen orbital energy",
        );
        assert_close(
            scalars.relaxation_energy,
            expected_relaxation,
            1.0e-9 * expected_relaxation.abs().max(1.0),
            "relaxation energy",
        );
        assert_close(
            scalars.edge_energy,
            expected_edge,
            1.0e-9 * expected_edge.abs().max(1.0),
            "edge energy",
        );
        Ok(())
    }

    #[test]
    fn atomic_module_preserves_absorber_energy_scalars_when_nohole_swaps_columns() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let static_arrays = beryllium_single_potential_static_arrays()?;
        let overlap_arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;
        let scalars = super::atomic_apot_energy_scalars_from_states(
            &input,
            Path::new("config.inp"),
            &states,
            &overlap_arrays,
        )?;

        let mut swapped = input.clone();
        swapped.run.nohole = -1;
        let swapped_states = super::generated_atomic_scf_states(&swapped, Path::new("config.inp"))?;
        let swapped_overlap_arrays = super::atomic_apot_overlap_arrays_from_states(
            &swapped,
            &static_arrays,
            &swapped_states,
        )?;
        let swapped_scalars = super::atomic_apot_energy_scalars_from_states(
            &swapped,
            Path::new("config.inp"),
            &swapped_states,
            &swapped_overlap_arrays,
        )?;

        assert_close(
            swapped_scalars.initial_total_energy,
            scalars.initial_total_energy,
            1.0e-9 * scalars.initial_total_energy.abs().max(1.0),
            "swapped initial total energy",
        );
        assert_close(
            swapped_scalars.final_total_energy,
            scalars.final_total_energy,
            1.0e-9 * scalars.final_total_energy.abs().max(1.0),
            "swapped final total energy",
        );
        assert_close(
            swapped_scalars.frozen_orbital_energy,
            scalars.frozen_orbital_energy,
            1.0e-12 * scalars.frozen_orbital_energy.abs().max(1.0),
            "swapped frozen orbital energy",
        );
        assert_close(
            swapped_scalars.relaxation_energy,
            scalars.relaxation_energy,
            1.0e-9 * scalars.relaxation_energy.abs().max(1.0),
            "swapped relaxation energy",
        );
        assert!(
            swapped_scalars.edge_energy.is_finite(),
            "swapped edge energy should stay finite"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_derives_nohole_core_hole_columns_with_zero_density() -> Result<()> {
        let input = beryllium_core_hole_pot_input()?;

        let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;

        assert_eq!(columns.large_component.len(), POT_BIN_RADIAL_POINTS);
        assert_eq!(columns.small_component.len(), POT_BIN_RADIAL_POINTS);
        assert_eq!(columns.density.len(), POT_BIN_RADIAL_POINTS);
        assert_eq!(columns.coulomb_potential.len(), POT_BIN_RADIAL_POINTS);
        assert!(
            columns.large_component.iter().any(|value| *value != 0.0),
            "core-hole large component should still be copied for fpf0 input"
        );
        assert!(columns.density.iter().all(|value| *value == 0.0));
        assert!(columns.coulomb_potential.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn atomic_module_derives_nohole_one_core_hole_density_from_initial_orbital() -> Result<()> {
        let mut input = beryllium_core_hole_pot_input()?;
        input.run.nohole = 1;

        let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;
        let radii = apot_core_hole_radii(POT_BIN_RADIAL_POINTS);

        assert!(columns.large_component.iter().any(|value| *value != 0.0));
        assert!(columns.small_component.iter().any(|value| *value != 0.0));
        assert!(columns.density.iter().any(|value| *value > 0.0));
        for row in [0_usize, 23, 97] {
            let expected_density = (columns.large_component[row] * columns.large_component[row]
                + columns.small_component[row] * columns.small_component[row])
                / (2.0 * radii[row] * radii[row]);
            assert_close(
                columns.density[row],
                expected_density,
                1.0e-9 * expected_density.abs().max(1.0),
                &format!("drho[{row}]"),
            );
        }

        let expected_coulomb =
            apot_core_hole_coulomb_from_density(columns.density.view(), input.run.nohole)?;
        for row in [0_usize, 23, 97] {
            assert_close(
                columns.coulomb_potential[row],
                expected_coulomb[row],
                1.0e-9 * expected_coulomb[row].abs().max(1.0),
                &format!("dvcoul[{row}]"),
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_module_derives_transition_core_hole_density_from_state_difference() -> Result<()> {
        let mut input = beryllium_core_hole_pot_input()?;
        input.run.nohole = 2;

        let columns = super::generated_atomic_core_hole_columns(&input, Path::new("config.inp"))?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let final_state = states.last().context("missing final absorber state")?;

        assert!(columns.density.iter().any(|value| *value != 0.0));
        for row in [0_usize, 23, 97] {
            let expected_density = 0.5
                * (states[0].scf.density_4pi[row]
                    - states[0].scf.valence_density_4pi[row]
                    - final_state.scf.density_4pi[row]
                    + final_state.scf.valence_density_4pi[row]);
            assert_close(
                columns.density[row],
                expected_density,
                1.0e-9 * expected_density.abs().max(1.0),
                &format!("transition drho[{row}]"),
            );
        }

        let expected_coulomb =
            apot_core_hole_coulomb_from_density(columns.density.view(), input.run.nohole)?;
        for row in [0_usize, 23, 97] {
            assert_close(
                columns.coulomb_potential[row],
                expected_coulomb[row],
                1.0e-9 * expected_coulomb[row].abs().max(1.0),
                &format!("transition dvcoul[{row}]"),
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_module_derives_static_apot_arrays_from_pot_geom_handoffs() -> Result<()> {
        let mut input = copper_two_potential_pot_input()?;
        input.overlap_shells = vec![
            vec![PotOverlapShell {
                iphovr: 1,
                nnovr: 2,
                rovr: 2.0 * FEFF_BOHR_ANGSTROM,
            }],
            vec![
                PotOverlapShell {
                    iphovr: 0,
                    nnovr: 1,
                    rovr: FEFF_BOHR_ANGSTROM,
                },
                PotOverlapShell {
                    iphovr: 1,
                    nnovr: 3,
                    rovr: 3.0 * FEFF_BOHR_ANGSTROM,
                },
            ],
        ];
        let geom = sample_atomic_geom_dat();
        let pot = sample_pot_bin();

        let arrays = super::atomic_apot_static_arrays_from_handoffs(&input, &geom, &pot)?;

        assert_eq!(arrays.unique_potential_count, 2);
        assert_eq!(arrays.atom_count, 3);
        assert_eq!(arrays.atomic_numbers.to_vec(), vec![29, 29]);
        assert_eq!(arrays.model_atom_indices.to_vec(), vec![1, 2]);
        assert_eq!(arrays.overlap_shell_counts.to_vec(), vec![1, 2]);
        assert_eq!(arrays.norman_radii.to_vec(), vec![2.1, 2.1]);
        assert_eq!(arrays.atom_potential_indices.to_vec(), vec![0, 1, 1]);
        assert_eq!(arrays.atom_positions.dim(), (3, 3));
        assert_close(arrays.atom_positions[(0, 1)], 1.0, 1.0e-12, "rat x atom 2");
        assert_close(arrays.atom_positions[(1, 2)], 2.0, 1.0e-12, "rat y atom 3");

        assert_eq!(arrays.overlap_potential_indices.dim(), (2, 2));
        assert_eq!(arrays.overlap_potential_indices[[0, 0]], 1);
        assert_eq!(arrays.overlap_potential_indices[[1, 0]], 0);
        assert_eq!(arrays.overlap_potential_indices[[0, 1]], 0);
        assert_eq!(arrays.overlap_potential_indices[[1, 1]], 1);
        assert_eq!(arrays.overlap_shell_atom_counts[[0, 0]], 2);
        assert_eq!(arrays.overlap_shell_atom_counts[[1, 1]], 3);
        assert_close(arrays.overlap_radii[[0, 0]], 2.0, 1.0e-12, "rovr 0,0");
        assert_close(arrays.overlap_radii[[0, 1]], 1.0, 1.0e-12, "rovr 0,1");
        assert_close(arrays.overlap_radii[[1, 1]], 3.0, 1.0e-12, "rovr 1,1");
        Ok(())
    }

    #[test]
    fn atomic_module_rejects_static_apot_atomic_number_mismatch() -> Result<()> {
        let input = copper_two_potential_pot_input()?;
        let geom = sample_atomic_geom_dat();
        let mut pot = sample_pot_bin();
        pot.atomic_numbers[1] = 30;

        let error = super::atomic_apot_static_arrays_from_handoffs(&input, &geom, &pot)
            .err()
            .context("mismatched pot.bin atomic number should fail")?;

        assert!(error.to_string().contains("does not match pot.inp"));
        Ok(())
    }

    #[test]
    fn atomic_module_derives_overlap_apot_arrays_from_single_potential_shell() -> Result<()> {
        let input = beryllium_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let static_arrays = super::AtomicApotStaticArrays {
            unique_potential_count: 1,
            atom_count: 1,
            atomic_numbers: Array1::from_vec(vec![4]),
            model_atom_indices: Array1::from_vec(vec![1]),
            overlap_shell_counts: Array1::from_vec(vec![1]),
            norman_radii: Array1::from_vec(vec![1.0]),
            atom_potential_indices: Array1::from_vec(vec![0]),
            atom_positions: Array2::zeros((3, 1)),
            overlap_potential_indices: Array2::from_shape_vec((1, 1), vec![0])?,
            overlap_shell_atom_counts: Array2::from_shape_vec((1, 1), vec![1])?,
            overlap_radii: Array2::from_shape_vec((1, 1), vec![2.0])?,
        };

        let arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        assert_eq!(arrays.norman_radii.len(), 1);
        assert!(arrays.norman_radii[0].is_finite());
        assert!(arrays.norman_radii[0] > 0.0);
        assert_eq!(
            arrays.magnetization_density.dim(),
            (POT_BIN_RADIAL_POINTS, 2)
        );
        assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 1));
        assert_eq!(
            arrays.overlapped_valence_density.dim(),
            (POT_BIN_RADIAL_POINTS, 1)
        );
        assert_eq!(
            arrays.overlapped_coulomb_potential.dim(),
            (POT_BIN_RADIAL_POINTS, 1)
        );
        assert!(
            arrays
                .magnetization_density
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
        assert!(arrays.overlapped_valence_density[(23, 0)] > states[0].scf.valence_density_4pi[23]);
        Ok(())
    }

    #[test]
    fn atomic_module_normalizes_isolated_apot_density_for_norman_radius() -> Result<()> {
        let input = beryllium_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let static_arrays = super::AtomicApotStaticArrays {
            unique_potential_count: 1,
            atom_count: 1,
            atomic_numbers: Array1::from_vec(vec![4]),
            model_atom_indices: Array1::from_vec(vec![1]),
            overlap_shell_counts: Array1::from_vec(vec![0]),
            norman_radii: Array1::from_vec(vec![1.0]),
            atom_potential_indices: Array1::from_vec(vec![0]),
            atom_positions: Array2::zeros((3, 1)),
            overlap_potential_indices: Array2::zeros((1, 1)),
            overlap_shell_atom_counts: Array2::zeros((1, 1)),
            overlap_radii: Array2::zeros((1, 1)),
        };

        let raw = norman_radius_from_density(NormanRadiusInput {
            overlapped_density: states[0].scf.density_4pi.view(),
            atomic_number: 4,
        });
        assert!(
            matches!(raw, Err(GridError::InsufficientNormanCharge { .. })),
            "raw isolated generated Be density should exercise the APOT normalization edge"
        );

        let arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 1));
        assert!(arrays.norman_radii[0].is_finite());
        assert!(arrays.norman_radii[0] > 0.0);
        assert!(
            arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23],
            "normalized isolated density should scale the source density upward"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_applies_explicit_overlap_shells_from_static_arrays() -> Result<()> {
        let input = beryllium_two_potential_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let mut atom_positions = Array2::<f64>::zeros((3, 2));
        atom_positions[(0, 1)] = 20.0;
        let static_arrays = super::AtomicApotStaticArrays {
            unique_potential_count: 2,
            atom_count: 2,
            atomic_numbers: Array1::from_vec(vec![4, 4]),
            model_atom_indices: Array1::from_vec(vec![1, 2]),
            overlap_shell_counts: Array1::from_vec(vec![1, 1]),
            norman_radii: Array1::from_vec(vec![1.0, 1.0]),
            atom_potential_indices: Array1::from_vec(vec![0, 1]),
            atom_positions,
            overlap_potential_indices: Array2::from_shape_vec((1, 2), vec![1, 0])?,
            overlap_shell_atom_counts: Array2::from_shape_vec((1, 2), vec![2, 1])?,
            overlap_radii: Array2::from_shape_vec((1, 2), vec![2.0, 2.5])?,
        };

        let arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        assert_eq!(arrays.norman_radii.len(), 2);
        assert_eq!(arrays.overlapped_density.dim(), (POT_BIN_RADIAL_POINTS, 2));
        assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
        assert!(arrays.overlapped_valence_density[(23, 0)] > states[0].scf.valence_density_4pi[23]);
        assert!(arrays.overlapped_density[(23, 1)] > states[1].scf.density_4pi[23]);
        Ok(())
    }

    #[test]
    fn atomic_module_applies_geometry_overlap_from_static_arrays() -> Result<()> {
        let input = beryllium_two_potential_pot_input()?;
        let states = super::generated_atomic_scf_states(&input, Path::new("config.inp"))?;
        let mut atom_positions = Array2::<f64>::zeros((3, 2));
        atom_positions[(0, 1)] = 2.0;
        let static_arrays = super::AtomicApotStaticArrays {
            unique_potential_count: 2,
            atom_count: 2,
            atomic_numbers: Array1::from_vec(vec![4, 4]),
            model_atom_indices: Array1::from_vec(vec![1, 2]),
            overlap_shell_counts: Array1::from_vec(vec![0, 0]),
            norman_radii: Array1::from_vec(vec![1.0, 1.0]),
            atom_potential_indices: Array1::from_vec(vec![0, 1]),
            atom_positions,
            overlap_potential_indices: Array2::zeros((1, 2)),
            overlap_shell_atom_counts: Array2::zeros((1, 2)),
            overlap_radii: Array2::zeros((1, 2)),
        };

        let arrays =
            super::atomic_apot_overlap_arrays_from_states(&input, &static_arrays, &states)?;

        assert!(arrays.overlapped_density[(23, 0)] > states[0].scf.density_4pi[23]);
        assert!(arrays.overlapped_density[(23, 1)] > states[1].scf.density_4pi[23]);
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_malformed_config_handoff_without_apot_solver() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        assert!(has_supported_config_handoff(temp.path())?);
        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_claim_malformed_custom_config_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = copper_two_potential_pot_input()?;
        input.config_type = 2;
        std::fs::write(
            temp.path().join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("config.inp"), "not a config.inp handoff\n")?;

        assert!(!has_supported_config_handoff(temp.path())?);

        let error = super::run_supported_config_handoff_in_dir(temp.path())
            .err()
            .context("malformed custom config.inp should fail through explicit config handoff")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("config.inp"), "{chain}");
        assert!(!temp.path().join("config.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_validates_existing_supported_config_handoff_without_apot_solver() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        super::run_supported_config_handoff_in_dir(temp.path())?;
        let expected = read_config_dat(temp.path().join("config.dat"))?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(read_config_dat(temp.path().join("config.dat"))?, expected);
        assert!(!temp.path().join("apot.bin").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_existing_malformed_log_for_config_handoff_without_apot_solver()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        super::run_supported_config_handoff_in_dir(temp.path())?;
        std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        assert!(!temp.path().join("apot.bin").exists());
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_stale_existing_config_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_config_dat())?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        assert!(has_supported_config_handoff(temp.path())?);
        assert!(!has_cached_atomic_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_advertise_malformed_apot_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

        assert!(!has_cached_atomic_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed apot.bin should fail through the explicit ATOM runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("apot.bin"), "{chain}");
        Ok(())
    }

    #[test]
    fn atomic_module_validates_config_handoff_when_malformed_apot_cache_exists() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

        assert!(!has_cached_atomic_output(temp.path())?);
        assert!(has_supported_config_handoff(temp.path())?);
        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 1);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );

        let error = run_in_dir(temp.path())
            .err()
            .context("source-backed config handoff should reach the ATOM geometry gate")?;

        assert!(
            error
                .to_string()
                .contains("ATOM source apot.bin generation requires geom.dat handoff"),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_config_log_when_pot_cache_is_unusable() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;
        std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;
        std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_supported_config_handoff(temp.path())?);

        let count = super::run_supported_config_handoff_in_dir(temp.path())?;

        assert_eq!(count, 2);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_malformed_config_sidecar_from_pot_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;

        assert!(has_cached_atomic_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_stale_config_sidecar_from_pot_input() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        write_config_dat(temp.path().join("config.dat"), &sample_config_dat())?;

        assert!(has_cached_atomic_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_malformed_module_log_for_config_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        std::fs::write(temp.path().join("config.dat"), "not config.dat\n")?;
        std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(has_cached_atomic_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_advertise_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        write_generated_config_dat(temp.path())?;
        std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_atomic_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed cached log1.dat should fail through the explicit ATOM runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("log1.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn atomic_module_does_not_advertise_malformed_fpf0_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        std::fs::write(temp.path().join("fpf0.dat"), "not fpf0.dat\n")?;

        assert!(!has_cached_atomic_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed fpf0.dat should fail through the explicit ATOM runner")?;
        let chain = format!("{error:?}");
        assert!(chain.contains("failed to read"), "{chain}");
        assert!(chain.contains("fpf0.dat"), "{chain}");
        Ok(())
    }

    #[test]
    fn atomic_module_roundtrips_cached_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        let apot_path = temp.path().join("apot.bin");
        let config_path = temp.path().join("config.dat");
        let fpf0_path = temp.path().join("fpf0.dat");
        let log_path = temp.path().join("log1.dat");
        write_apot_bin(&apot_path, &sample_apot_bin())?;
        write_generated_config_dat(temp.path())?;
        write_fpf0_dat(&fpf0_path, &sample_fpf0_dat())?;
        write_module_log_dat(&log_path, &sample_module_log())?;
        let expected_apot = read_apot_bin(&apot_path)?;
        let expected_config = read_config_dat(&config_path)?;
        let expected_fpf0 = read_fpf0_dat(&fpf0_path)?;
        let expected_log = read_module_log_dat(&log_path)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(has_cached_atomic_output(temp.path())?);
        assert_eq!(read_apot_bin(&apot_path)?, expected_apot);
        assert_eq!(read_config_dat(&config_path)?, expected_config);
        assert_eq!(read_fpf0_dat(&fpf0_path)?, expected_fpf0);
        assert_eq!(read_module_log_dat(&log_path)?, expected_log);
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_module_log_from_cached_apot() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?.potential_count(),
            2
        );
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_log_contains(&log, "Calculating atomic potentials ...");
        assert_log_contains(
            &log,
            "    overlapped atomic potential and density for unique potential    0",
        );
        assert_log_contains(
            &log,
            "    overlapped atomic potential and density for unique potential    1",
        );
        assert_log_contains(&log, "Done with module: atomic potentials.");
        Ok(())
    }

    #[test]
    fn atomic_module_generates_second_overlap_pass_for_ionized_potentials() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        let pot_path = temp.path().join("pot.inp");
        let pot_text = std::fs::read_to_string(&pot_path)?;
        let mut input = PotInput::parse_str(&pot_path, &pot_text)?;
        input.potentials[1].xion = 0.25;
        std::fs::write(&pot_path, refeff_io::pot_input_string(&input)?)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        let log = read_module_log_dat(temp.path().join("log1.dat"))?;
        assert_eq!(
            log_line_occurrences(
                &log,
                "    overlapped atomic potential and density for unique potential    0"
            ),
            2
        );
        assert_eq!(
            log_line_occurrences(
                &log,
                "    overlapped atomic potential and density for unique potential    1"
            ),
            2
        );
        Ok(())
    }

    #[test]
    fn atomic_module_validates_cached_core_hole_coulomb_source() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        set_nohole(temp.path(), 2)?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_transition_apot_bin()?)?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 3);
        Ok(())
    }

    #[test]
    fn atomic_module_regenerates_corrupt_cached_core_hole_coulomb() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        set_nohole(temp.path(), 2)?;
        let mut apot = sample_transition_apot_bin()?;
        let expected = rendered_apot_bin(&sample_transition_apot_bin()?)?;
        let records = apot.sections[0]
            .records()
            .context("sample transition apot should have core-hole records")?;
        let mut rows = records.rows.clone();
        rows[10][3] = ApotBinValue::Real(10.0);
        if let ApotBinPayload::Records(records) = &mut apot.sections[0].payload {
            records.rows = rows;
        }
        write_apot_bin(temp.path().join("apot.bin"), &apot)?;

        let count = run_in_dir(temp.path())?;
        let actual = read_apot_bin(temp.path().join("apot.bin"))?;

        assert_eq!(count, 3);
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn atomic_module_regenerates_nohole_core_hole_coulomb_to_zero() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        let mut apot = sample_apot_bin();
        let records = apot.sections[0]
            .records()
            .context("sample apot should have core-hole records")?;
        let mut rows = records.rows.clone();
        rows[10][3] = ApotBinValue::Real(10.0);
        if let ApotBinPayload::Records(records) = &mut apot.sections[0].payload {
            records.rows = rows;
        }
        write_apot_bin(temp.path().join("apot.bin"), &apot)?;

        let count = run_in_dir(temp.path())?;
        let actual = read_apot_bin(temp.path().join("apot.bin"))?;

        assert_eq!(count, 3);
        assert_eq!(actual, rendered_apot_bin(&sample_apot_bin())?);
        Ok(())
    }

    #[test]
    fn atomic_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            crate::require_fixture!("ATOM reference test; generated EXAFS/Cu reference not found");
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        for name in ["config.dat", "fpf0.dat", "log1.dat"] {
            let source = reference_dir.join(name);
            if source.is_file() {
                std::fs::copy(source, temp.path().join(name))?;
            }
        }
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        let expected_config = optional_config_dat(temp.path().join("config.dat"))?;
        let expected_fpf0 = optional_fpf0_dat(temp.path().join("fpf0.dat"))?;
        let expected_log = optional_module_log(temp.path().join("log1.dat"))?;

        let count = run_in_dir(temp.path())?;

        let optional_count = [
            expected_config.as_ref().map(|_| 1_usize),
            expected_fpf0.as_ref().map(|_| 1_usize),
            Some(1_usize),
        ]
        .into_iter()
        .flatten()
        .sum::<usize>();
        assert_eq!(count, 1 + optional_count);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        if let Some(expected) = expected_config {
            assert_eq!(read_config_dat(temp.path().join("config.dat"))?, expected);
        }
        if let Some(expected) = expected_fpf0 {
            assert_eq!(read_fpf0_dat(temp.path().join("fpf0.dat"))?, expected);
        }
        if let Some(expected) = expected_log {
            assert_eq!(read_module_log_dat(temp.path().join("log1.dat"))?, expected);
        } else {
            let log = read_module_log_dat(temp.path().join("log1.dat"))?;
            assert_log_contains(&log, "Calculating atomic potentials ...");
            assert_log_contains(&log, "Done with module: atomic potentials.");
        }
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_reference_config_when_absent() -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            crate::require_fixture!(
                "ATOM config generation test; generated EXAFS/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_config = read_config_dat(reference_dir.join("config.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?,
            expected_config
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_reference_config_before_missing_geometry_error_when_present()
    -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            crate::require_fixture!(
                "ATOM pre-solver config test; generated EXAFS/Cu reference not found"
            );
        };
        let expected_path = reference_dir.join("config.dat");
        if !expected_path.is_file() {
            crate::require_fixture!(
                "ATOM pre-solver config test; generated EXAFS/Cu config.dat not found"
            );
        }

        let temp = tempfile::tempdir()?;
        std::fs::copy(reference_dir.join("pot.inp"), temp.path().join("pot.inp"))?;
        let expected_config = read_config_dat(expected_path)?;

        let error = run_in_dir(temp.path())
            .err()
            .context("ATOM without apot.bin should still require source geometry")?;

        assert!(
            error
                .to_string()
                .contains("ATOM source apot.bin generation requires geom.dat handoff")
        );
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?,
            expected_config
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_reference_fpf0_when_absent() -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            crate::require_fixture!(
                "ATOM fpf0 generation test; generated EXAFS/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!temp.path().join("fort.16").is_file());
        assert_fpf0_close(
            &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
            &expected_fpf0,
        );
        Ok(())
    }

    #[test]
    fn atomic_total_energy_from_reference_apot_matches_fort16() -> Result<()> {
        let cases = [
            ("EXAFS/Cu", reference_atomic_dir()?),
            ("XANES/Cu", reference_atomic_transition_dir()?),
            ("NRIXS/GeCl_4", reference_atomic_nohole_dir()?),
        ];
        for (label, reference_dir) in cases {
            let Some(reference_dir) = reference_dir else {
                crate::record_missing_fixture!("{label}; reference not found");
                continue;
            };
            let pot_path = reference_dir.join("pot.inp");
            let pot_text = std::fs::read_to_string(&pot_path)?;
            let input = PotInput::parse_str(&pot_path, &pot_text)?;
            let apot = read_apot_bin(reference_dir.join("apot.bin"))?;
            assert!(super::has_fpf0_total_energy_source_sections(&apot, &input)?);
            let config_inp = reference_dir.join("config.inp");
            let actual = super::generated_atomic_total_energy(&apot, &input, &config_inp)?;
            let fort16 = read_fort16(reference_dir.join("fort.16"))?;
            let expected = fort16
                .total_energy_hartree
                .iter()
                .next_back()
                .copied()
                .with_context(|| format!("{label} reference fort.16 has no total-energy rows"))?;
            let difference = (actual.total - expected).abs();
            assert!(
                difference <= 7.5e-5,
                concat!(
                    "{label} total_energy differs by {difference:e}: ",
                    "actual={actual_total:e}, expected={expected:e}, ",
                    "direct={direct:e}, exchange={exchange:e}, ",
                    "magnetic_breit={magnetic_breit:e}, retarded_breit={retarded_breit:e}"
                ),
                label = label,
                difference = difference,
                actual_total = actual.total,
                expected = expected,
                direct = actual.direct_coulomb,
                exchange = actual.exchange_coulomb,
                magnetic_breit = actual.magnetic_breit,
                retarded_breit = actual.retarded_breit,
            );
        }
        Ok(())
    }

    #[test]
    fn atomic_module_recovers_stale_reference_fpf0_when_source_handoff_present() -> Result<()> {
        let Some(reference_dir) = reference_atomic_dir()? else {
            crate::require_fixture!(
                "ATOM stale fpf0 recovery test; generated EXAFS/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;
        let mut stale_fpf0 = expected_fpf0.clone();
        stale_fpf0.atomic_number = 8;
        write_fpf0_dat(temp.path().join("fpf0.dat"), &stale_fpf0)?;

        assert!(has_cached_atomic_output(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!temp.path().join("fort.16").is_file());
        assert_fpf0_close(
            &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
            &expected_fpf0,
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_reference_nohole_config_when_absent() -> Result<()> {
        let Some(reference_dir) = reference_atomic_nohole_dir()? else {
            crate::require_fixture!(
                "ATOM nohole config generation test; generated NRIXS/GeCl_4 reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_config = read_config_dat(reference_dir.join("config.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_config_dat(temp.path().join("config.dat"))?,
            expected_config
        );
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_reference_nohole_fpf0_when_absent() -> Result<()> {
        let Some(reference_dir) = reference_atomic_nohole_dir()? else {
            crate::require_fixture!(
                "ATOM nohole fpf0 generation test; generated NRIXS/GeCl_4 reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!temp.path().join("fort.16").is_file());
        assert_fpf0_close(
            &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
            &expected_fpf0,
        );
        Ok(())
    }

    #[test]
    fn atomic_module_validates_reference_transition_core_hole_when_present() -> Result<()> {
        let Some(reference_dir) = reference_atomic_transition_dir()? else {
            crate::require_fixture!(
                "ATOM transition reference test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }

        run_in_dir(temp.path())?;
        Ok(())
    }

    #[test]
    fn atomic_module_generates_missing_reference_transition_fpf0_when_absent() -> Result<()> {
        let Some(reference_dir) = reference_atomic_transition_dir()? else {
            crate::require_fixture!(
                "ATOM transition fpf0 generation test; generated XANES/Cu reference not found"
            );
        };

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let expected_fpf0 = read_fpf0_dat(reference_dir.join("fpf0.dat"))?;

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert!(!temp.path().join("fort.16").is_file());
        assert_fpf0_close(
            &read_fpf0_dat(temp.path().join("fpf0.dat"))?,
            &expected_fpf0,
        );
        Ok(())
    }

    fn write_pot_input(work_dir: &Path, mpot: i32) -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu atomic smoke test
EDGE K
CONTROL 1 1 1 1 1 1
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
        pot_input.control.mpot = mpot;
        std::fs::write(
            work_dir.join("pot.inp"),
            refeff_io::pot_input_string(&pot_input)?,
        )?;
        Ok(())
    }

    fn beryllium_pot_input() -> Result<PotInput> {
        let mut input = beryllium_core_hole_pot_input()?;
        input.control.ihole = 0;
        input.run.nohole = 0;
        Ok(input)
    }

    fn highz_pot_input(atomic_number: usize) -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            &format!(
                r#"
TITLE HIGHZ Z={atomic_number}
CONTROL 1 0 0 0 0 0
PRINT 5 0 0 0 0 0
NOHOLE
HIGHZ
POTENTIALS
0 {atomic_number} X
ATOMS
0.0 0.0 0.0 0 X0
END
"#
            ),
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        input.control.ihole = 0;
        input.control.ipr1 = 5;
        input.run.nohole = 0;
        input.finite_nucleus = true;
        Ok(input)
    }

    fn highz_finite_binding_energy(report: &str, atomic_number: usize) -> Result<f64> {
        let prefix = format!("{atomic_number}:");
        let line = report
            .lines()
            .find(|line| line.starts_with(&prefix))
            .with_context(|| format!("HIGHZ report is missing Z={atomic_number}"))?;
        line.split_whitespace()
            .nth(2)
            .context("HIGHZ row is missing its finite-nucleus binding energy")?
            .parse::<f64>()
            .with_context(|| format!("HIGHZ Z={atomic_number} binding energy is not numeric"))
    }

    fn beryllium_core_hole_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be atomic SCF smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        input.run.nohole = 0;
        Ok(input)
    }

    fn copper_two_potential_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu atomic static APOT smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
0.0 1.0 0.0 1 Cu2
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        Ok(PotInput::parse_str(
            "pot.inp",
            &rdinp::pot_inp_string(&document)?,
        )?)
    }

    fn beryllium_two_potential_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be two-potential overlap smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 4 Be
1 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
1.0 0.0 0.0 1 Be1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        Ok(PotInput::parse_str(
            "pot.inp",
            &rdinp::pot_inp_string(&document)?,
        )?)
    }

    fn beryllium_single_potential_static_arrays() -> Result<super::AtomicApotStaticArrays> {
        Ok(super::AtomicApotStaticArrays {
            unique_potential_count: 1,
            atom_count: 1,
            atomic_numbers: Array1::from_vec(vec![4]),
            model_atom_indices: Array1::from_vec(vec![1]),
            overlap_shell_counts: Array1::from_vec(vec![1]),
            norman_radii: Array1::from_vec(vec![1.0]),
            atom_potential_indices: Array1::from_vec(vec![0]),
            atom_positions: Array2::zeros((3, 1)),
            overlap_potential_indices: Array2::from_shape_vec((1, 1), vec![0])?,
            overlap_shell_atom_counts: Array2::from_shape_vec((1, 1), vec![1])?,
            overlap_radii: Array2::from_shape_vec((1, 1), vec![2.0])?,
        })
    }

    fn write_beryllium_atomic_source_handoffs(work_dir: &Path) -> Result<()> {
        let input = beryllium_pot_input()?;
        std::fs::write(
            work_dir.join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        write_pot_bin(
            work_dir.join("pot.bin"),
            &beryllium_single_potential_pot_bin(),
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        Ok(())
    }

    fn write_beryllium_atomic_geometry_source_handoffs(work_dir: &Path) -> Result<()> {
        let input = beryllium_pot_input()?;
        std::fs::write(
            work_dir.join("pot.inp"),
            refeff_io::pot_input_string(&input)?,
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        Ok(())
    }

    fn beryllium_single_potential_geom_dat() -> GeomDat {
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

    fn beryllium_single_potential_config_dat() -> ConfigDatData {
        let mut occupations = Array1::zeros(40);
        let mut valence_occupations = Array1::zeros(40);
        occupations[0] = 2.0;
        valence_occupations[0] = 2.0;
        ConfigDatData {
            header_lines: Vec::new(),
            potentials: vec![ConfigDatPotential {
                potential_index: 0,
                atomic_number: 4,
                element: "Be".to_string(),
                occupations,
                valence_occupations,
                spin_occupations: None,
            }],
        }
    }

    fn beryllium_single_potential_pot_bin() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["ATOM source APOT Be smoke test".to_string()],
            pad_width: 8,
            nohole: 0,
            ihole: 0,
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
            atomic_numbers: Array1::from_vec(vec![4]),
            kappa: Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: Array1::from_vec(vec![2.1]),
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

    fn sample_scf_mtdp_data(seed: &PotBinData) -> MtdpData {
        let mut atom_density = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
        let mut atom_potential = Array2::zeros((POT_BIN_RADIAL_POINTS, 1));
        for row in 0..POT_BIN_RADIAL_POINTS {
            atom_density[(row, 0)] = seed.electron_density[(row, 0)];
            atom_potential[(row, 0)] = seed.total_potential[(row, 0)];
        }
        atom_density[(0, 0)] += 1.0e-6;
        atom_density[(2, 0)] += 2.0e-6;
        atom_potential[(0, 0)] = -1.0;
        atom_potential[(1, 0)] = -1.1;
        atom_potential[(2, 0)] = -1.2;
        MtdpData {
            radial_count: POT_BIN_RADIAL_POINTS,
            atomic_numbers: Array1::from_vec(vec![4]),
            atom_coordinates: Array2::zeros((1, 3)),
            atom_radii: Array1::from_vec(vec![1.25]),
            atom_radius_indices: Array1::from_vec(vec![7]),
            atom_density,
            atom_potential,
            empty_sphere_coordinates: Array2::zeros((0, 3)),
            empty_sphere_radii: Array1::zeros(0),
            empty_sphere_radius_indices: Array1::zeros(0),
            empty_sphere_density: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
            empty_sphere_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, 0)),
            interstitial_potential: -0.75,
            homo_energy: -0.12,
            lumo_energy: -0.08,
        }
    }

    fn sample_atomic_geom_dat() -> GeomDat {
        GeomDat {
            nat: 3,
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
                    x: FEFF_BOHR_ANGSTROM,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 1,
                },
                GeomDatRow {
                    index: 3,
                    x: 0.0,
                    y: 2.0 * FEFF_BOHR_ANGSTROM,
                    z: 0.0,
                    iph: 1,
                    boundary: 1,
                },
            ],
        }
    }

    fn set_nohole(work_dir: &Path, nohole: i32) -> Result<()> {
        let pot_path = work_dir.join("pot.inp");
        let pot_text = std::fs::read_to_string(&pot_path)?;
        let mut input = PotInput::parse_str(&pot_path, &pot_text)?;
        input.run.nohole = nohole;
        std::fs::write(&pot_path, refeff_io::pot_input_string(&input)?)?;
        Ok(())
    }

    fn sample_apot_bin() -> ApotBinData {
        ApotBinData {
            sections: vec![
                sample_core_hole_section(None).expect("zero core-hole sample should be renderable"),
                sample_atomic_density_section(),
            ],
        }
    }

    fn sample_transition_apot_bin() -> Result<ApotBinData> {
        let drho = Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            0.012 + 0.0002 * row as f64 + 0.001 * (-0.03 * row as f64).exp()
        });
        Ok(ApotBinData {
            sections: vec![
                sample_core_hole_section(Some(drho.view()))?,
                sample_atomic_density_section(),
            ],
        })
    }

    fn sample_pot_bin() -> PotBinData {
        let potentials = 2;
        PotBinData {
            titles: vec!["ATOM config handoff smoke test".to_string()],
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
            muffin_tin_indices: Array1::from_vec(vec![12, 12]),
            muffin_tin_radii: Array1::from_vec(vec![1.1, 1.1]),
            norman_indices: Array1::from_vec(vec![40, 40]),
            atomic_numbers: Array1::from_vec(vec![29, 29]),
            kappa: Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: Array1::from_vec(vec![2.1, 2.1]),
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

    fn sample_core_hole_section(
        drho: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<ApotBinSection> {
        let drho = drho
            .map(|values| values.to_owned())
            .unwrap_or_else(|| Array1::zeros(POT_BIN_RADIAL_POINTS));
        let dvcoul = apot_core_hole_coulomb_from_density(drho.view(), 2)?;
        Ok(ApotBinSection {
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
                            ApotBinValue::Real(0.05 + 0.001 * row as f64),
                            ApotBinValue::Real(-0.005 - 0.0001 * row as f64),
                            ApotBinValue::Real(drho[row]),
                            ApotBinValue::Real(dvcoul[row]),
                        ]
                    })
                    .collect(),
            }),
            trailing_headers: vec![],
            trailing_header_texts: vec![],
        })
    }

    fn sample_atomic_density_section() -> ApotBinSection {
        ApotBinSection {
            section_number: 8,
            headers: vec!["rho(r,0:nphx+1) - atomic density for each unique potential".to_string()],
            header_texts: vec![
                " rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
            ],
            column_labels: vec![],
            column_label_text: None,
            payload: ApotBinPayload::Matrix(ApotBinMatrix {
                value_type: ApotBinType::Double,
                values: ApotBinMatrixValues::Real(Array2::from_shape_fn(
                    (POT_BIN_RADIAL_POINTS, 2),
                    |(row, potential)| 0.015 * (row + 1) as f64 + 0.25 * potential as f64,
                )),
            }),
            trailing_headers: vec![],
            trailing_header_texts: vec![],
        }
    }

    fn sample_config_dat() -> ConfigDatData {
        ConfigDatData {
            header_lines: Vec::new(),
            potentials: vec![ConfigDatPotential {
                potential_index: 0,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.1),
                valence_occupations: Array1::from_shape_fn(40, |index| index as f64 * 0.01),
                spin_occupations: None,
            }],
        }
    }

    fn write_generated_config_dat(work_dir: &Path) -> Result<()> {
        let input_path = work_dir.join("pot.inp");
        let input_text = std::fs::read_to_string(&input_path)?;
        let input = PotInput::parse_str(&input_path, &input_text)?;
        let data = super::generated_config_dat(&input, &work_dir.join("config.inp"))?;
        write_config_dat(work_dir.join("config.dat"), &data)?;
        Ok(())
    }

    fn sample_fpf0_dat() -> Fpf0DatData {
        Fpf0DatData {
            atomic_number: 29,
            total_energy_fprime: -0.125,
            relativistic_correction: 0.075,
            oscillators: vec![Fpf0Oscillator {
                oscillator_strength: 1.25,
                excitation_energy: -8.98,
                orbital_index: 1,
            }],
            form_factor_momentum: Array1::from_vec(vec![0.0, 0.5, 1.0]),
            form_factor: Array1::from_vec(vec![29.0, 28.1, 25.7]),
        }
    }

    fn sample_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating atomic potentials ...".to_string(),
                "Done with module: atomic potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(), "\n".to_string()],
        }
    }

    fn optional_config_dat(path: impl AsRef<Path>) -> Result<Option<ConfigDatData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_config_dat(path)?));
        }
        Ok(None)
    }

    fn optional_fpf0_dat(path: impl AsRef<Path>) -> Result<Option<Fpf0DatData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_fpf0_dat(path)?));
        }
        Ok(None)
    }

    fn optional_module_log(path: impl AsRef<Path>) -> Result<Option<ModuleLogData>> {
        let path = path.as_ref();
        if path.is_file() {
            return Ok(Some(read_module_log_dat(path)?));
        }
        Ok(None)
    }

    fn rendered_apot_bin(data: &ApotBinData) -> Result<ApotBinData> {
        Ok(parse_apot_bin(&apot_bin_string(data)?)?)
    }

    fn assert_log_contains(log: &ModuleLogData, expected: &str) {
        assert!(
            log.lines.iter().any(|line| line.contains(expected)),
            "expected log to contain {expected:?}, got {:?}",
            log.lines
        );
    }

    fn log_line_occurrences(log: &ModuleLogData, expected: &str) -> usize {
        log.lines.iter().filter(|line| line == &expected).count()
    }

    fn assert_fpf0_close(actual: &Fpf0DatData, expected: &Fpf0DatData) {
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_close(
            actual.total_energy_fprime,
            expected.total_energy_fprime,
            1.0e-6,
            "total_energy_fprime",
        );
        assert_close(
            actual.relativistic_correction,
            expected.relativistic_correction,
            1.0e-6,
            "relativistic_correction",
        );
        assert_eq!(actual.oscillators, expected.oscillators);
        assert_eq!(actual.form_factor_momentum, expected.form_factor_momentum);
        assert_eq!(actual.form_factor_count(), expected.form_factor_count());
        for (index, (&actual, &expected)) in actual
            .form_factor
            .iter()
            .zip(expected.form_factor.iter())
            .enumerate()
        {
            assert_close(actual, expected, 1.5e-4, &format!("form_factor[{index}]"));
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        let difference = (actual - expected).abs();
        assert!(
            difference <= tolerance,
            "{label} differs by {difference:e}: actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
        );
    }

    fn reference_atomic_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/Cu"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("apot.bin").is_file()))
    }

    fn reference_atomic_transition_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/Cu"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("apot.bin").is_file()))
    }

    fn reference_atomic_nohole_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/NRIXS/GeCl_4"));
        Ok(reference.filter(|path| {
            path.join("pot.inp").is_file()
                && path.join("apot.bin").is_file()
                && path.join("config.dat").is_file()
        }))
    }

    fn reference_bn_true_scf_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/BN"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_exafs_ybco_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/YBCO"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_highz_report() -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/HIGHZ/HighZ.out"))
            .filter(|path| path.is_file())
    }
}
