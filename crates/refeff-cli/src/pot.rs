use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use refeff_io::{
    GeomDat, PotInput, cached_pot_stage_module_log, pot_input_string,
    potential_dat_outputs_from_bins, read_apot_bin, read_config_inp, read_module_log_dat,
    read_pot_bin, write_module_log_dat,
};

use crate::{atomic, work_dir_for_input, wpot};

const POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE: &str = concat!(
    "POT required stage needs complete source handoffs that can produce ",
    "pot.bin/apot.bin, or readable pot.bin/apot.bin caches"
);

#[derive(Debug)]
struct CachedNoScfPotPreparation {
    work_dir: PathBuf,
    fingerprint: atomic::NoScfPotSourceFingerprint,
    prepared: atomic::PreparedNoScfPotOutputs,
}

/// Mutable state shared by POT discovery and execution during one pipeline.
///
/// The cache is intentionally owned by a single run. It never crosses
/// workspaces or process boundaries, and its source fingerprint is checked
/// before every reuse.
#[derive(Debug, Default)]
pub(crate) struct PotRunContext {
    no_scf: Option<CachedNoScfPotPreparation>,
    no_scf_unavailable: Option<(PathBuf, atomic::NoScfPotSourceFingerprint)>,
    #[cfg(test)]
    no_scf_preparation_count: usize,
    #[cfg(test)]
    no_scf_attempt_count: usize,
}

impl PotRunContext {
    pub(crate) fn prepared_no_scf(
        &mut self,
        work_dir: &Path,
    ) -> Result<Option<&atomic::PreparedNoScfPotOutputs>> {
        let Some(current_fingerprint) = atomic::no_scf_pot_source_fingerprint_in_dir(work_dir)?
        else {
            self.no_scf = None;
            self.no_scf_unavailable = None;
            return Ok(None);
        };
        let reusable = self.no_scf.as_ref().is_some_and(|cached| {
            cached.work_dir == work_dir && cached.fingerprint == current_fingerprint
        });
        if reusable {
            return Ok(self.no_scf.as_ref().map(|cached| &cached.prepared));
        }
        if self
            .no_scf_unavailable
            .as_ref()
            .is_some_and(|(cached_dir, fingerprint)| {
                cached_dir == work_dir && *fingerprint == current_fingerprint
            })
        {
            return Ok(None);
        }

        self.no_scf = None;
        self.no_scf_unavailable = None;
        // A source may be replaced by another process between fingerprinting
        // and parsing. Retry once, but never publish a mixed-source result.
        for attempt in 0..2 {
            #[cfg(test)]
            {
                self.no_scf_attempt_count += 1;
            }
            let (fingerprint, prepared) = match atomic::prepare_no_scf_pot_outputs_in_dir(work_dir)
            {
                Ok(Some(prepared)) => prepared,
                Ok(None) => {
                    self.no_scf_unavailable =
                        Some((work_dir.to_path_buf(), current_fingerprint.clone()));
                    return Ok(None);
                }
                Err(error) => {
                    self.no_scf_unavailable =
                        Some((work_dir.to_path_buf(), current_fingerprint.clone()));
                    return Err(error);
                }
            };
            #[cfg(test)]
            {
                self.no_scf_preparation_count += 1;
            }
            if atomic::no_scf_pot_source_fingerprint_in_dir(work_dir)?.as_ref()
                == Some(&fingerprint)
            {
                self.no_scf = Some(CachedNoScfPotPreparation {
                    work_dir: work_dir.to_path_buf(),
                    fingerprint,
                    prepared,
                });
                return Ok(self.no_scf.as_ref().map(|cached| &cached.prepared));
            }
            if attempt == 1 {
                bail!("POT no-SCF source handoffs changed repeatedly while preparing atomic state");
            }
        }
        unreachable!("bounded POT no-SCF source preparation loop returned no result")
    }
}

/// Run the FEFF `POT` stage beside an input.
///
/// Existing FEFF `pot.bin`/`apot.bin` caches and the supported source-backed
/// POT subsets can regenerate `potNN.dat` outputs through the `wpot`
/// compatibility writer and preserve or regenerate the FEFF `log1.dat` module
/// wrapper.
pub(crate) fn run_for_input(input: &Path) -> Result<usize> {
    let work_dir = work_dir_for_input(input);
    let mut context = PotRunContext::default();
    if has_cached_pot_output_with_context(work_dir, &mut context)? {
        return run_in_dir_with_context(work_dir, &mut context);
    }
    if has_supported_pot_source_handoff_with_context(work_dir, &mut context)? {
        return run_in_dir_with_context(work_dir, &mut context);
    }
    if has_supported_pot_generation_handoff_with_context(work_dir, &mut context)? {
        return run_in_dir_with_context(work_dir, &mut context);
    }
    if let Some(count) = run_supported_pot_scf_source_handoff_once_in_dir(work_dir)? {
        return Ok(count);
    }
    if has_supported_pot_scf_output_handoff(work_dir)? {
        return run_in_dir(work_dir);
    }
    if has_supported_pot_scf_loop_handoff(work_dir)? {
        return run_supported_pot_scf_loop_handoff_in_dir(work_dir);
    }
    if has_supported_pot_scf_initial_handoff(work_dir)? {
        return run_supported_pot_scf_initial_handoff_in_dir(work_dir);
    }
    if has_supported_pot_input_handoff(work_dir)? {
        return run_supported_pot_input_handoff_in_dir(work_dir);
    }
    run_in_dir(work_dir)
}

pub(crate) fn run_in_dir(work_dir: &Path) -> Result<usize> {
    let mut context = PotRunContext::default();
    run_in_dir_with_context(work_dir, &mut context)
}

pub(crate) fn run_in_dir_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<usize> {
    if !pot_enabled(work_dir)? {
        return Ok(0);
    }
    let restart_scf_input = pot_scf_restart_uses_existing_pot_bin(work_dir)?;
    let source_pot_count = prepare_pot_source_state_with_context(work_dir, context)?;
    let source_scf_pot_count = prepare_pot_scf_output_state_with_context(work_dir, context)?;
    let source_scf_loop_count = if source_scf_pot_count == 0 {
        prepare_pot_scf_loop_state_with_context(work_dir, context)?
    } else {
        0
    };
    let source_scf_initial_count = if source_scf_pot_count == 0 && source_scf_loop_count == 0 {
        prepare_pot_scf_initial_state_with_context(work_dir, context)?
    } else {
        0
    };
    let source_sidecar_count =
        if source_scf_pot_count > 1 || (restart_scf_input && source_scf_pot_count == 0) {
            0
        } else {
            prepare_pot_source_sidecars_with_context(work_dir, source_scf_pot_count > 0, context)?
        };
    let final_pot_files_ready =
        has_cached_pot_files(work_dir) && (!restart_scf_input || source_scf_pot_count > 0);
    if final_pot_files_ready && work_dir.join("pot.inp").is_file() {
        let input = read_input(work_dir)?;
        validate_declared_pot_source_handoffs(work_dir, &input)?;
    }
    if !final_pot_files_ready {
        validate_pot_input_handoff_if_present(work_dir)?;
        bail!(POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE);
    }
    let count =
        wpot::run_in_dir(work_dir).context("failed to run supported wpot stage from POT caches")?;
    Ok(count
        + source_pot_count
        + source_scf_pot_count
        + source_scf_loop_count
        + source_scf_initial_count
        + source_sidecar_count
        + write_or_generate_module_log(&work_dir.join("log1.dat"))?)
}

pub(crate) fn has_cached_pot_output(work_dir: &Path) -> Result<bool> {
    let mut context = PotRunContext::default();
    has_cached_pot_output_with_context(work_dir, &mut context)
}

pub(crate) fn has_cached_pot_output_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<bool> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    if input.control.mpot != 1 || !has_cached_pot_files(work_dir) {
        return Ok(false);
    }
    if validate_declared_pot_source_handoffs(work_dir, &input).is_err() {
        return Ok(false);
    }
    can_render_final_pot_files_with_context(work_dir, context)
}

/// Whether FEFF `POT` can rebuild the ATOM `apot.bin` sidecar from complete
/// typed source handoffs before consuming the existing self-consistent
/// `pot.bin` state through `wpot`.
pub(crate) fn has_supported_pot_source_handoff(work_dir: &Path) -> Result<bool> {
    let mut context = PotRunContext::default();
    has_supported_pot_source_handoff_with_context(work_dir, &mut context)
}

pub(crate) fn has_supported_pot_source_handoff_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<bool> {
    if !pot_enabled_for_discovery(work_dir)?
        || can_render_final_pot_files_with_context(work_dir, context)?
        || pot_start_from_file_uses_existing_pot_bin(work_dir)?
        || source_predicate_or_false(atomic::has_stale_scf_pot_bin_from_sources_in_dir(work_dir))
    {
        return Ok(false);
    }
    if !work_dir.join("pot.bin").is_file() {
        return Ok(false);
    }
    Ok(source_predicate_or_false(
        atomic::can_write_atomic_apot_from_sources_in_dir(work_dir),
    ))
}

/// Whether FEFF `POT` can generate a no-SCF `pot.bin` directly from typed
/// source handoffs before the final source-completion boundary.
pub(crate) fn has_supported_pot_generation_handoff(work_dir: &Path) -> Result<bool> {
    let mut context = PotRunContext::default();
    has_supported_pot_generation_handoff_with_context(work_dir, &mut context)
}

pub(crate) fn has_supported_pot_generation_handoff_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<bool> {
    if !pot_enabled_for_discovery(work_dir)?
        || can_render_final_pot_files_with_context(work_dir, context)?
    {
        return Ok(false);
    }
    if pot_start_from_file_uses_existing_pot_bin(work_dir)? {
        return Ok(source_predicate_or_false(
            atomic::can_write_no_scf_pot_bin_from_sources_in_dir(work_dir),
        ));
    }
    if has_stale_no_scf_pot_bin_from_sources_with_context(work_dir, context) {
        return Ok(true);
    }
    if work_dir.join("pot.bin").is_file() && read_pot_bin(work_dir.join("pot.bin")).is_ok() {
        return Ok(false);
    }
    Ok(matches!(context.prepared_no_scf(work_dir), Ok(Some(_))))
}

/// Whether FEFF `POT` can complete a source-backed iterative SCF run into
/// `pot.bin` before rendering sidecars and `potNN.dat` outputs.
pub(crate) fn has_supported_pot_scf_output_handoff(work_dir: &Path) -> Result<bool> {
    if !pot_enabled_for_discovery(work_dir)?
        || can_render_final_pot_files(work_dir)?
        || has_supported_pot_generation_handoff(work_dir)?
        || has_supported_pot_source_handoff(work_dir)?
    {
        return Ok(false);
    }
    Ok(source_predicate_or_false(
        atomic::can_write_scf_pot_bin_from_sources_in_dir(work_dir),
    ))
}

/// Whether FEFF `POT` can build and validate the source-backed initial state
/// for the iterative SCF loop before the final source-completion boundary.
pub(crate) fn has_supported_pot_scf_initial_handoff(work_dir: &Path) -> Result<bool> {
    if !pot_enabled_for_discovery(work_dir)?
        || can_render_final_pot_files(work_dir)?
        || has_supported_pot_generation_handoff(work_dir)?
        || has_supported_pot_scf_output_handoff(work_dir)?
        || has_supported_pot_source_handoff(work_dir)?
    {
        return Ok(false);
    }
    Ok(source_predicate_or_false(
        atomic::can_prepare_scf_pot_initial_state_from_sources_in_dir(work_dir),
    ))
}

/// Whether FEFF `POT` can run the source-backed SCF loop driver until a
/// terminal convergence status or the next missing source-row boundary.
pub(crate) fn has_supported_pot_scf_loop_handoff(work_dir: &Path) -> Result<bool> {
    if !pot_enabled_for_discovery(work_dir)?
        || can_render_final_pot_files(work_dir)?
        || has_supported_pot_generation_handoff(work_dir)?
        || has_supported_pot_scf_output_handoff(work_dir)?
        || has_supported_pot_source_handoff(work_dir)?
    {
        return Ok(false);
    }
    Ok(source_predicate_or_false(
        atomic::can_prepare_scf_pot_loop_from_sources_in_dir(work_dir),
    ))
}

/// Whether FEFF `POT` can validate the typed `pot.inp` source handoff before
/// the final source-completion boundary.
pub(crate) fn has_supported_pot_input_handoff(work_dir: &Path) -> Result<bool> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file()
        || has_cached_pot_output(work_dir)?
        || has_supported_pot_source_handoff(work_dir)?
        || has_supported_pot_generation_handoff(work_dir)?
        || has_supported_pot_scf_output_handoff(work_dir)?
        || has_supported_pot_scf_loop_handoff(work_dir)?
        || has_supported_pot_scf_initial_handoff(work_dir)?
        || work_dir.join("geom.dat").is_file()
    {
        return Ok(false);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    validate_pot_input_handoff(&input)?;
    Ok(pot_input_handoff_is_supported(&input))
}

/// Validate only the Rust-backed FEFF `pot.inp` source handoff.
///
/// This intentionally does not enter the SCF loop, does not write
/// `log1.dat`, and does not report POT completion.
pub(crate) fn run_supported_pot_input_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file()
        || has_cached_pot_output(work_dir)?
        || work_dir.join("geom.dat").is_file()
    {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    validate_pot_input_handoff(&input)?;
    Ok(usize::from(pot_input_handoff_is_supported(&input)))
}

/// Validate only the Rust-backed initial `pot.bin` state needed by the
/// remaining iterative POT SCF solver. This intentionally does not write a
/// final `pot.bin`, `apot.bin`, `log1.dat`, or `potNN.dat`.
pub(crate) fn run_supported_pot_scf_initial_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file() || has_cached_pot_output(work_dir)? {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    validate_pot_input_handoff(&input)?;
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }
    atomic::prepare_scf_pot_initial_state_from_sources_in_dir(work_dir)
        .context("failed to validate POT initial SCF source handoff")
}

/// Run the Rust-backed SCF loop driver as far as source rows allow. This
/// intentionally does not write a final `pot.bin`, `apot.bin`, `log1.dat`, or
/// `potNN.dat`.
pub(crate) fn run_supported_pot_scf_loop_handoff_in_dir(work_dir: &Path) -> Result<usize> {
    let input_path = work_dir.join("pot.inp");
    if !input_path.is_file() || has_cached_pot_output(work_dir)? {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    validate_pot_input_handoff(&input)?;
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }
    atomic::prepare_scf_pot_loop_from_sources_in_dir(work_dir)
        .context("failed to validate POT SCF loop source handoff")
}

pub(crate) fn run_supported_pot_scf_source_handoff_once_in_dir(
    work_dir: &Path,
) -> Result<Option<usize>> {
    let enabled = match pot_enabled(work_dir) {
        Ok(enabled) => enabled,
        Err(_) => return Ok(None),
    };
    if !enabled || can_render_final_pot_files(work_dir)? {
        return Ok(None);
    }
    if !work_dir.join("pot.inp").is_file() || !work_dir.join("geom.dat").is_file() {
        return Ok(None);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(None);
    };
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(None);
    }
    validate_pot_input_handoff(&input)?;

    let outcome = match atomic::run_scf_pot_source_handoff_once_in_dir(work_dir) {
        Ok(outcome) => outcome,
        Err(_) => return Ok(None),
    };
    match outcome {
        atomic::PotScfSourceHandoffOutcome::NotApplicable => Ok(None),
        atomic::PotScfSourceHandoffOutcome::LoopValidated { count } => Ok(Some(count)),
        atomic::PotScfSourceHandoffOutcome::FinalOutput { count } => {
            if !has_cached_pot_files(work_dir) {
                bail!(
                    "source-backed POT SCF output handoff did not produce complete pot.bin/apot.bin caches"
                );
            }
            let rendered = wpot::run_in_dir(work_dir)
                .context("failed to run supported wpot stage from POT caches")?;
            Ok(Some(
                count + rendered + write_or_generate_module_log(&work_dir.join("log1.dat"))?,
            ))
        }
    }
}

fn has_cached_pot_files(work_dir: &Path) -> bool {
    work_dir.join("pot.bin").is_file() && work_dir.join("apot.bin").is_file()
}

fn validate_declared_pot_source_handoffs(work_dir: &Path, input: &PotInput) -> Result<()> {
    let geom_path = work_dir.join("geom.dat");
    if geom_path.is_file() {
        let geom_text = std::fs::read_to_string(&geom_path)
            .with_context(|| format!("failed to read {}", geom_path.display()))?;
        GeomDat::parse_str(&geom_path, &geom_text)
            .with_context(|| format!("failed to parse {}", geom_path.display()))?;
    }
    let config_path = work_dir.join("config.inp");
    if input.config_type == 2 && config_path.is_file() {
        read_config_inp(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
    }
    if input.external_pot {
        atomic::validate_pot_external_source_handoff_in_dir(work_dir, input.potentials.len())?;
    }
    Ok(())
}

fn can_render_cached_pot_files(work_dir: &Path) -> bool {
    let Ok(pot) = read_pot_bin(work_dir.join("pot.bin")) else {
        return false;
    };
    let Ok(apot) = read_apot_bin(work_dir.join("apot.bin")) else {
        return false;
    };
    potential_dat_outputs_from_bins(&pot, &apot).is_ok()
}

fn can_render_final_pot_files(work_dir: &Path) -> Result<bool> {
    let mut context = PotRunContext::default();
    can_render_final_pot_files_with_context(work_dir, &mut context)
}

fn can_render_final_pot_files_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<bool> {
    if pot_start_from_file_uses_existing_pot_bin(work_dir)?
        && !has_rendered_pot_output_marker(work_dir)
    {
        return Ok(false);
    }
    if has_stale_no_scf_pot_bin_from_sources_with_context(work_dir, context) {
        return Ok(false);
    }
    if source_predicate_or_false(atomic::has_stale_scf_pot_bin_from_sources_in_dir(work_dir)) {
        return Ok(false);
    }
    Ok(can_render_cached_pot_files(work_dir))
}

fn has_stale_no_scf_pot_bin_from_sources_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> bool {
    let prepared = match context.prepared_no_scf(work_dir) {
        Ok(Some(prepared)) => prepared,
        Ok(None) | Err(_) => return false,
    };
    atomic::prepared_no_scf_pot_outputs_match_cached(work_dir, prepared)
        .map(|matches| !matches)
        .unwrap_or(false)
}

fn has_rendered_pot_output_marker(work_dir: &Path) -> bool {
    if !work_dir.join("pot00.dat").is_file() {
        return false;
    }
    let Ok(log) = read_module_log_dat(work_dir.join("log1.dat")) else {
        return false;
    };
    log.lines
        .iter()
        .any(|line| line.contains("Calculating SCF potentials ..."))
        && log
            .lines
            .iter()
            .any(|line| line.contains("Done with module: potentials."))
}

fn source_predicate_or_false(result: Result<bool>) -> bool {
    result.unwrap_or(false)
}

fn pot_start_from_file_uses_existing_pot_bin(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(false);
    }
    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    Ok(input.control.mpot == 1 && input.start_from_file)
}

fn pot_scf_restart_uses_existing_pot_bin(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(false);
    }
    let input = read_input(work_dir)?;
    Ok(input.control.mpot == 1 && input.start_from_file && input.run.nscmt > 0)
}

fn prepare_pot_source_state_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<usize> {
    if let Ok(Some(prepared)) = context.prepared_no_scf(work_dir) {
        if atomic::prepared_no_scf_pot_outputs_match_cached(work_dir, prepared)? {
            return Ok(0);
        }
        return atomic::write_prepared_no_scf_pot_outputs_in_dir(work_dir, prepared)
            .context("failed to write prepared POT no-SCF outputs");
    }
    if can_render_final_pot_files_with_context(work_dir, context)? {
        return Ok(0);
    }

    let refreshed = atomic::refresh_no_scf_pot_bin_from_sources_if_stale_in_dir(work_dir)
        .context("failed to refresh POT pot.bin from no-SCF source handoffs")?;
    if refreshed > 0 {
        return Ok(refreshed);
    }
    if !atomic::can_write_no_scf_pot_bin_from_sources_in_dir(work_dir)? {
        return Ok(0);
    }
    atomic::write_no_scf_pot_bin_from_sources_in_dir(work_dir)
        .context("failed to generate POT pot.bin from no-SCF source handoffs")
}

fn prepare_pot_scf_initial_state_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<usize> {
    if can_render_final_pot_files_with_context(work_dir, context)?
        || has_supported_pot_generation_handoff_with_context(work_dir, context)?
    {
        return Ok(0);
    }
    if !work_dir.join("pot.inp").is_file() || !work_dir.join("geom.dat").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }
    validate_pot_input_handoff(&input)?;

    atomic::prepare_scf_pot_initial_state_from_sources_in_dir(work_dir)
        .context("failed to validate POT initial SCF source handoff")
}

fn prepare_pot_scf_loop_state_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<usize> {
    if can_render_final_pot_files_with_context(work_dir, context)?
        || has_supported_pot_generation_handoff_with_context(work_dir, context)?
    {
        return Ok(0);
    }
    if !work_dir.join("pot.inp").is_file() || !work_dir.join("geom.dat").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }
    validate_pot_input_handoff(&input)?;

    atomic::prepare_scf_pot_loop_from_sources_in_dir(work_dir)
        .context("failed to validate POT SCF loop source handoff")
}

fn prepare_pot_scf_output_state_with_context(
    work_dir: &Path,
    context: &mut PotRunContext,
) -> Result<usize> {
    if can_render_final_pot_files_with_context(work_dir, context)?
        || has_supported_pot_generation_handoff_with_context(work_dir, context)?
    {
        return Ok(0);
    }
    if !work_dir.join("pot.inp").is_file() || !work_dir.join("geom.dat").is_file() {
        return Ok(0);
    }

    let input = read_input(work_dir)?;
    if !pot_input_handoff_is_supported(&input) || input.run.nscmt <= 0 {
        return Ok(0);
    }
    validate_pot_input_handoff(&input)?;

    atomic::try_write_scf_pot_bin_from_sources_in_dir(work_dir)
        .context("failed to generate POT pot.bin from SCF source handoffs")
}

fn prepare_pot_source_sidecars_with_context(
    work_dir: &Path,
    force: bool,
    context: &mut PotRunContext,
) -> Result<usize> {
    if (!force && can_render_final_pot_files_with_context(work_dir, context)?)
        || !work_dir.join("pot.bin").is_file()
    {
        return Ok(0);
    }
    atomic::write_atomic_apot_from_sources_in_dir(work_dir)
        .context("failed to generate POT apot.bin sidecar from ATOM source handoffs")
}

fn pot_enabled(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(true);
    }

    let input = read_input(work_dir)?;
    Ok(input.control.mpot == 1)
}

fn pot_enabled_for_discovery(work_dir: &Path) -> Result<bool> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(true);
    }

    let Ok(input) = read_input(work_dir) else {
        return Ok(false);
    };
    Ok(input.control.mpot == 1)
}

fn read_input(work_dir: &Path) -> Result<PotInput> {
    let input_path = work_dir.join("pot.inp");
    let input_text = std::fs::read_to_string(&input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    PotInput::parse_str(&input_path, &input_text)
        .with_context(|| format!("failed to parse {}", input_path.display()))
}

fn pot_input_handoff_is_supported(input: &PotInput) -> bool {
    input.control.mpot == 1
}

fn validate_pot_input_handoff(input: &PotInput) -> Result<()> {
    pot_input_string(input)
        .map(|_| ())
        .context("failed to validate POT pot.inp source handoff")
}

fn validate_pot_input_handoff_if_present(work_dir: &Path) -> Result<()> {
    if !work_dir.join("pot.inp").is_file() {
        return Ok(());
    }
    let input = read_input(work_dir)?;
    validate_pot_input_handoff(&input)
}

fn write_or_generate_module_log(path: &Path) -> Result<usize> {
    let existing = if path.is_file() {
        read_module_log_dat(path).ok()
    } else {
        None
    };
    let data = cached_pot_stage_module_log(existing.as_ref());
    write_module_log_dat(path, &data)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::{
        POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE, PotRunContext, has_cached_pot_output,
        has_supported_pot_generation_handoff, has_supported_pot_generation_handoff_with_context,
        has_supported_pot_input_handoff, has_supported_pot_scf_initial_handoff,
        has_supported_pot_scf_loop_handoff, has_supported_pot_scf_output_handoff,
        has_supported_pot_source_handoff, run_for_input, run_in_dir, run_in_dir_with_context,
        run_supported_pot_input_handoff_in_dir, run_supported_pot_scf_source_handoff_once_in_dir,
    };
    use anyhow::{Context, Result, bail};
    use ndarray::{Array1, Array2, Array3};
    use refeff_io::pot_bin::{
        POT_BIN_COEFFICIENTS, POT_BIN_IORB_SLOTS, POT_BIN_ORBITALS, POT_BIN_RADIAL_POINTS,
    };
    use refeff_io::{
        ApotBinData, ApotBinMatrix, ApotBinMatrixValues, ApotBinPayload, ApotBinSection,
        ApotBinType, ApotBinValue, FeffDocument, FeffInput, GeomDat, GeomDatRow, ModuleLogData,
        MtdpData, PotBinData, PotBinScalars, PotInput, geom_dat_string, pot_input_string, rdinp,
        read_apot_bin, read_module_log_dat, read_pot_bin, write_apot_bin, write_module_log_dat,
        write_mtdp, write_pot_bin,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[test]
    fn pot_module_rejects_required_stage_without_source_or_caches() -> Result<()> {
        let temp = tempfile::tempdir()?;

        let error = run_in_dir(temp.path())
            .err()
            .context("POT should require source handoffs or caches")?;

        assert!(
            error
                .to_string()
                .contains(POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn pot_module_validates_source_input_handoff_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;

        assert!(has_supported_pot_input_handoff(temp.path())?);
        assert_eq!(run_supported_pot_input_handoff_in_dir(temp.path())?, 1);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 1);
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());

        let error = run_in_dir(temp.path())
            .err()
            .context("required POT stage should still require source handoffs or caches")?;
        assert!(
            error
                .to_string()
                .contains(POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn pot_module_direct_runner_validates_source_input_before_source_requirement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input_with_nan_gamach(temp.path())?;

        let error = run_in_dir(temp.path())
            .err()
            .context("direct POT runner should validate pot.inp before the source requirement")?;
        let chain = format!("{error:#}");

        assert!(chain.contains("failed to validate POT pot.inp source handoff"));
        assert!(chain.contains("gamach must be finite"));
        assert!(!chain.contains(POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE));
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_skips_disabled_source_input_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 0)?;

        assert!(!has_supported_pot_input_handoff(temp.path())?);
        assert_eq!(run_supported_pot_input_handoff_in_dir(temp.path())?, 0);
        assert_eq!(run_for_input(&temp.path().join("feff.inp"))?, 0);
        Ok(())
    }

    #[test]
    fn pot_module_prefers_cached_output_over_source_input_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert!(count > 1);
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(temp.path().join("log1.dat").is_file());
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_orphan_cache_when_input_is_missing() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        assert!(!has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_malformed_input_during_discovery() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 1)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::write(temp.path().join("pot.inp"), "not a pot.inp handoff\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_loop_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_initial_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);
        assert_eq!(
            super::run_supported_pot_scf_source_handoff_once_in_dir(temp.path())?,
            None
        );

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed pot.inp should fail through the explicit POT runner")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("pot.inp"), "{chain}");
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, apot);
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_preserves_cached_output_when_no_scf_source_selector_is_unsupported() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_no_scf_pot_input()?;
        input.control.iscfxc = 0;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;

        assert!(has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_no_cache_no_scf_source_selector_when_unsupported() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_no_scf_pot_input()?;
        input.control.iscfxc = 0;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_loop_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_initial_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);
        assert_eq!(
            super::run_supported_pot_scf_source_handoff_once_in_dir(temp.path())?,
            None
        );

        let error = run_in_dir(temp.path())
            .err()
            .context("unsupported no-SCF selector should not produce POT output")?;
        assert!(
            error
                .to_string()
                .contains(POT_REQUIRED_SOURCE_OR_CACHE_MESSAGE),
            "{error:?}"
        );
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("apot.bin").exists());
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_treats_start_from_file_pot_bin_as_restart_input_not_final_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_oxygen_no_scf_pot_input()?;
        input.run.nscmt = 2;
        input.start_from_file = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&copper_two_potential_geom_dat())?,
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let mut context = PotRunContext::default();

        assert!(super::can_render_cached_pot_files(temp.path()));
        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(context.prepared_no_scf(temp.path())?.is_none());
        assert_eq!(context.no_scf_preparation_count, 0);
        Ok(())
    }

    #[test]
    fn pot_module_generates_missing_apot_sidecar_from_atomic_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_pot_source_handoffs(temp.path())?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 4);
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_replaces_malformed_apot_sidecar_from_atomic_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_pot_source_handoffs(temp.path())?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_source_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_recovers_inconsistent_pot_bin_before_apot_sidecar_from_source_handoffs()
    -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_pot_source_handoffs(temp.path())?;
        let mut pot = beryllium_single_potential_pot_bin();
        pot.atomic_numbers[0] = 8;
        write_pot_bin(temp.path().join("pot.bin"), &pot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(
            read_pot_bin(temp.path().join("pot.bin"))?
                .atomic_numbers
                .to_vec(),
            vec![4]
        );
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_does_not_advertise_apot_source_handoff_without_pot_bin() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
        assert!(pot.norman_radii[0] > pot.muffin_tin_radii[0]);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(pot.large_components.iter().any(|value| *value != 0.0));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_run_context_prepares_no_scf_atomic_state_once_for_discovery_and_execution() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        let mut context = PotRunContext::default();

        assert!(has_supported_pot_generation_handoff_with_context(
            temp.path(),
            &mut context
        )?);
        assert!(has_supported_pot_generation_handoff_with_context(
            temp.path(),
            &mut context
        )?);
        assert_eq!(context.no_scf_preparation_count, 1);

        let count = run_in_dir_with_context(temp.path(), &mut context)?;

        assert_eq!(count, 4);
        assert_eq!(context.no_scf_preparation_count, 1);
        assert!(temp.path().join("pot.bin").is_file());
        assert!(temp.path().join("apot.bin").is_file());
        Ok(())
    }

    #[test]
    fn pot_run_context_invalidates_no_scf_state_when_source_input_changes() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input_path = temp.path().join("pot.inp");
        let mut input = beryllium_no_scf_pot_input()?;
        std::fs::write(&input_path, pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        let mut context = PotRunContext::default();

        assert!(context.prepared_no_scf(temp.path())?.is_some());
        assert_eq!(context.no_scf_preparation_count, 1);

        input.run.nohole = 1;
        std::fs::write(&input_path, pot_input_string(&input)?)?;
        assert!(context.prepared_no_scf(temp.path())?.is_some());
        assert_eq!(context.no_scf_preparation_count, 2);
        Ok(())
    }

    #[test]
    fn pot_run_context_does_not_cache_failed_no_scf_preparation() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(temp.path().join("geom.dat"), "not geom.dat\n")?;
        let mut context = PotRunContext::default();

        assert!(context.prepared_no_scf(temp.path()).is_err());
        assert_eq!(context.no_scf_preparation_count, 0);
        assert_eq!(context.no_scf_attempt_count, 1);
        assert!(context.prepared_no_scf(temp.path())?.is_none());
        assert_eq!(context.no_scf_attempt_count, 1);

        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        assert!(context.prepared_no_scf(temp.path())?.is_some());
        assert_eq!(context.no_scf_preparation_count, 1);
        assert_eq!(context.no_scf_attempt_count, 2);
        Ok(())
    }

    #[test]
    fn pot_module_recover_malformed_pot_bin_from_no_scf_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        std::fs::write(temp.path().join("pot.bin"), "not pot.bin\n")?;
        std::fs::write(temp.path().join("apot.bin"), "not apot.bin\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
        assert!(pot.norman_radii[0] > pot.muffin_tin_radii[0]);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.large_components.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_malformed_no_scf_geometry_source_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(temp.path().join("geom.dat"), "not a geom.dat handoff\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);
        assert_eq!(run_supported_pot_input_handoff_in_dir(temp.path())?, 0);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed POT geom.dat should fail through the explicit POT runner")?;
        let chain = format!("{error:#}");
        assert!(
            chain.contains("failed to refresh POT pot.bin from no-SCF source handoffs"),
            "{chain}"
        );
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("geom.dat"), "{chain}");
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_cached_output_with_malformed_geometry_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::write(temp.path().join("geom.dat"), "not a geom.dat handoff\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed POT geom.dat should block cached POT completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("failed to parse"), "{chain}");
        assert!(chain.contains("geom.dat"), "{chain}");
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_cached_output_with_malformed_config_source_handoff() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        let mut input = copper_custom_config_no_scf_pot_input()?;
        input.run.nscmt = 0;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_oxygen_geom_dat())?,
        )?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::write(temp.path().join("config.inp"), "not a config.inp handoff\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed POT config.inp should block cached POT completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("config.inp"), "{chain}");
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_pot_bin_from_no_scf_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        let mut stale_pot = expected_pot.clone();
        stale_pot.scalars.interstitial_density += 0.25;
        write_pot_bin(temp.path().join("pot.bin"), &stale_pot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_apot_bin_from_no_scf_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        let mut stale_apot = expected_apot.clone();
        add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
        write_apot_bin(temp.path().join("apot.bin"), &stale_apot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_generates_finite_nucleus_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_no_scf_pot_input()?;
        input.finite_nucleus = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(pot.large_components.iter().any(|value| *value != 0.0));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_imports_external_potential_no_scf_pot_bin_from_mtdp_handoff() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_no_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(temp.path().join("GeCl4.04.dft.mtdp"), &sample_mtdp_data())?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.muffin_tin_indices[0], 7);
        assert!((pot.muffin_tin_radii[0] - 1.25).abs() < 1.0e-10);
        assert!((pot.scalars.interstitial_potential + 0.75).abs() < 1.0e-10);
        assert!((pot.scalars.fermi_level + 0.10).abs() < 1.0e-10);
        assert!((pot.total_potential[(0, 0)] + 1.0).abs() < 1.0e-10);
        assert!((pot.total_potential[(2, 0)] + 1.2).abs() < 1.0e-10);
        assert!((pot.total_potential[(3, 0)] + 0.75).abs() < 1.0e-10);
        assert!((pot.electron_density[(0, 0)] - 0.11).abs() < 1.0e-10);
        assert!((pot.electron_density[(2, 0)] - 0.13).abs() < 1.0e-10);
        assert!((pot.electron_density[(3, 0)] - pot.scalars.interstitial_density).abs() < 1.0e-10);
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_writes_external_potential_iterative_scf_outputs_from_mtdp_handoff() -> Result<()>
    {
        let seed = tempfile::tempdir()?;
        std::fs::write(
            seed.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            seed.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(seed.path())?;
        let seed_pot = read_pot_bin(seed.path().join("pot.bin"))?;

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_loop_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert!(
            count >= 4,
            "expected pot.bin, apot.bin, pot00.dat, and log1.dat"
        );
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_ne!(
            pot, seed_pot,
            "EXTPOT SCF source route should not preserve the normal Be seed output"
        );
        assert_eq!(pot.potential_count(), 1);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(pot.electron_density.iter().any(|value| *value != 0.0));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(pot_scf_cache_path(temp.path()).is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_scf_cache_provenance_rejects_modified_external_source_handoff() -> Result<()> {
        let seed = tempfile::tempdir()?;
        write_beryllium_iterative_scf_source_handoffs(seed.path())?;
        run_in_dir(seed.path())?;
        let seed_pot = read_pot_bin(seed.path().join("pot.bin"))?;

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;
        run_in_dir(temp.path())?;
        assert!(has_cached_pot_output(temp.path())?);

        std::fs::write(temp.path().join("sort.aip"), "0  \n")?;
        assert!(!has_cached_pot_output(temp.path())?);
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;
        assert!(has_cached_pot_output(temp.path())?);

        let mut source = std::fs::read_to_string(temp.path().join("GeCl4.04.dft.mtdp"))?;
        source.push('\n');
        std::fs::write(temp.path().join("GeCl4.04.dft.mtdp"), source)?;
        assert!(!has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_external_scf_pot_from_mtdp_handoff() -> Result<()> {
        let seed = tempfile::tempdir()?;
        std::fs::write(
            seed.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            seed.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(seed.path())?;
        let seed_pot = read_pot_bin(seed.path().join("pot.bin"))?;

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;

        let mut stale_pot = expected_pot.clone();
        stale_pot.scalars.interstitial_density += 0.25;
        write_pot_bin(temp.path().join("pot.bin"), &stale_pot)?;
        let mut stale_apot = expected_apot.clone();
        add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
        write_apot_bin(temp.path().join("apot.bin"), &stale_apot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_cached_external_output_with_malformed_sort_source_handoff()
    -> Result<()> {
        let seed = tempfile::tempdir()?;
        std::fs::write(
            seed.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            seed.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(seed.path())?;
        let seed_pot = read_pot_bin(seed.path().join("pot.bin"))?;

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::write(temp.path().join("sort.aip"), "not a sort.aip handoff\n")?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed POT sort.aip should block cached external POT completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("sort.aip"), "{chain}");
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        Ok(())
    }

    #[test]
    fn pot_module_does_not_claim_cached_external_output_with_malformed_mtdp_source_handoff()
    -> Result<()> {
        let seed = tempfile::tempdir()?;
        std::fs::write(
            seed.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            seed.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(seed.path())?;
        let seed_pot = read_pot_bin(seed.path().join("pot.bin"))?;

        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.external_pot = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        write_mtdp(
            temp.path().join("GeCl4.04.dft.mtdp"),
            &sample_scf_mtdp_data(&seed_pot),
        )?;
        std::fs::write(temp.path().join("sort.aip"), "0\n")?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::write(
            temp.path().join("GeCl4.04.dft.mtdp"),
            "not an MTDP handoff\n",
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed POT MTDP should block cached external POT completion")?;
        let chain = format!("{error:#}");
        assert!(chain.contains("GeCl4.04.dft.mtdp"), "{chain}");
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        Ok(())
    }

    #[test]
    fn pot_module_imports_start_from_file_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_no_scf_pot_input()?;
        input.start_from_file = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
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

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.total_potential, restart.total_potential);
        assert_eq!(pot.electron_density, restart.electron_density);
        assert!(
            (pot.scalars.fermi_level - restart.scalars.fermi_level).abs() < 1.0e-10,
            "Fermi level was not imported from restart pot.bin"
        );
        assert!(
            (pot.scalars.interstitial_potential - restart.scalars.interstitial_potential).abs()
                < 1.0e-10,
            "interstitial potential was not imported from restart pot.bin"
        );
        assert!(
            (pot.scalars.interstitial_density - restart.scalars.interstitial_density).abs()
                < 1.0e-10,
            "interstitial density was not imported from restart pot.bin"
        );
        assert_ne!(pot.coulomb_potential, restart.coulomb_potential);
        assert_ne!(pot.valence_density, restart.valence_density);
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_default_exchange_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = beryllium_default_exchange_no_scf_pot_input()?;
        assert_eq!(input.control.ixc, 0);
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(pot.total_potential.iter().all(|value| value.is_finite()));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_high_exchange_no_scf_outputs_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = beryllium_high_exchange_no_scf_pot_input()?;
        assert_eq!(input.control.ixc, 6);
        assert_eq!(input.run.nscmt, 0);
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4]);
        assert!(
            pot.total_potential
                .iter()
                .zip(pot.valence_potential.iter())
                .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
            "EXCHANGE 6 no-SCF source path should preserve separate valence potential"
        );
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_high_exchange_no_scf_outputs_from_source_handoffs() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        let input = beryllium_high_exchange_no_scf_pot_input()?;
        assert_eq!(input.control.ixc, 6);
        assert_eq!(input.run.nscmt, 0);
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(
            expected_pot
                .total_potential
                .iter()
                .zip(expected_pot.valence_potential.iter())
                .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
            "EXCHANGE 6 no-SCF source path should preserve separate valence potential"
        );
        let mut stale_pot = expected_pot.clone();
        stale_pot.scalars.interstitial_density += 0.25;
        write_pot_bin(temp.path().join("pot.bin"), &stale_pot)?;
        let mut stale_apot = expected_apot.clone();
        add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
        write_apot_bin(temp.path().join("apot.bin"), &stale_apot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_writes_iterative_scf_outputs_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_loop_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_initial_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert!(
            count >= 4,
            "expected pot.bin, apot.bin, pot00.dat, and log1.dat"
        );
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.potential_count(), 1);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(pot.electron_density.iter().any(|value| *value != 0.0));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_treats_start_from_file_scf_pot_bin_as_loop_source_not_final_cache() -> Result<()>
    {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.start_from_file = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
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

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);
        assert!(has_supported_pot_scf_loop_handoff(temp.path())?);

        let count = run_for_input(&temp.path().join("feff.inp"))?;

        assert_eq!(count, 1);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, restart);
        assert!(!temp.path().join("apot.bin").exists());
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(pot_scf_cache_path(temp.path()).is_file());
        let mut stale_pot = expected_pot.clone();
        stale_pot.scalars.interstitial_density += 0.25;
        write_pot_bin(temp.path().join("pot.bin"), &stale_pot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_scf_apot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(pot_scf_cache_path(temp.path()).is_file());
        let mut stale_apot = expected_apot.clone();
        add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
        write_apot_bin(temp.path().join("apot.bin"), &stale_apot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_rebuilds_malformed_scf_cache_provenance_from_matching_outputs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_beryllium_iterative_scf_source_handoffs(temp.path())?;
        run_in_dir(temp.path())?;
        let sidecar = pot_scf_cache_path(temp.path());
        assert!(sidecar.is_file());
        std::fs::write(&sidecar, "not POT SCF cache provenance\n")?;

        assert!(has_cached_pot_output(temp.path())?);
        assert!(
            std::fs::read_to_string(&sidecar)?.starts_with("refeff-pot-scf-cache-v1\n"),
            "malformed sidecar should be replaced after legacy cache comparison"
        );
        Ok(())
    }

    #[test]
    fn pot_module_scf_cache_provenance_rejects_modified_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = beryllium_iterative_scf_pot_input()?;
        let geom = beryllium_single_potential_geom_dat();
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;
        run_in_dir(temp.path())?;
        assert!(has_cached_pot_output(temp.path())?);

        let mut changed_input = input.clone();
        changed_input.run.nohole += 1;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&changed_input)?,
        )?;
        assert!(!has_cached_pot_output(temp.path())?);
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        assert!(has_cached_pot_output(temp.path())?);

        let mut changed_geom = geom.clone();
        changed_geom.atoms[0].x += 0.125;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&changed_geom)?,
        )?;
        assert!(!has_cached_pot_output(temp.path())?);
        std::fs::write(temp.path().join("geom.dat"), geom_dat_string(&geom)?)?;
        assert!(has_cached_pot_output(temp.path())?);

        std::fs::write(temp.path().join("config.inp"), "1 1\n")?;
        assert!(!has_cached_pot_output(temp.path())?);
        std::fs::remove_file(temp.path().join("config.inp"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_missing_scf_apot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        std::fs::remove_file(temp.path().join("apot.bin"))?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_validates_finite_nucleus_iterative_scf_loop_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_iterative_scf_pot_input()?;
        input.finite_nucleus = true;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(!has_supported_pot_scf_output_handoff(temp.path())?);
        assert_eq!(
            run_supported_pot_scf_source_handoff_once_in_dir(temp.path())?,
            Some(1)
        );
        assert!(!temp.path().join("pot.bin").exists());
        assert!(!temp.path().join("apot.bin").exists());
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_generates_iterative_scf_outputs_with_high_exchange_selector() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = beryllium_high_exchange_iterative_scf_pot_input()?;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        let count = run_in_dir(temp.path())?;

        assert!(count >= 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert!(
            pot.total_potential
                .iter()
                .zip(pot.valence_potential.iter())
                .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
            "EXCHANGE 5 source path should preserve separate valence potential"
        );
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_regenerates_stale_high_exchange_scf_outputs_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = beryllium_high_exchange_iterative_scf_pot_input()?;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        run_in_dir(temp.path())?;
        let expected_pot = read_pot_bin(temp.path().join("pot.bin"))?;
        let expected_apot = read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(
            expected_pot
                .total_potential
                .iter()
                .zip(expected_pot.valence_potential.iter())
                .any(|(total, valence)| (*total - *valence).abs() > 1.0e-8),
            "EXCHANGE 5 source path should preserve separate valence potential"
        );
        let mut stale_pot = expected_pot.clone();
        stale_pot.scalars.interstitial_density += 0.25;
        write_pot_bin(temp.path().join("pot.bin"), &stale_pot)?;
        let mut stale_apot = expected_apot.clone();
        add_to_first_real_apot_matrix_value(&mut stale_apot, 0.25);
        write_apot_bin(temp.path().join("apot.bin"), &stale_apot)?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(!has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_source_handoff(temp.path())?);
        assert!(has_supported_pot_scf_output_handoff(temp.path())?);
        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        assert_eq!(read_pot_bin(temp.path().join("pot.bin"))?, expected_pot);
        assert_eq!(read_apot_bin(temp.path().join("apot.bin"))?, expected_apot);
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_completes_core_hole_iterative_scf_outputs_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut input = beryllium_core_hole_no_scf_pot_input()?;
        input.run.nscmt = 2;
        input.run.nohole = -1;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert!(pot.scalars.fermi_level.is_finite());
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_core_hole_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_core_hole_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;

        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 4);
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.ihole, 1);
        assert!(pot.nohole < 0);
        assert!(
            pot.initial_large_component
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(pot.scalars.edge_position.is_finite());
        assert!(pot.scalars.amplitude_reduction > 0.0);
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_multi_potential_no_scf_pot_bin_from_source_handoffs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(
            temp.path().join("pot.inp"),
            pot_input_string(&beryllium_oxygen_no_scf_pot_input()?)?,
        )?;
        std::fs::write(
            temp.path().join("geom.dat"),
            geom_dat_string(&beryllium_oxygen_geom_dat())?,
        )?;

        assert!(!has_cached_pot_output(temp.path())?);
        assert!(has_supported_pot_generation_handoff(temp.path())?);
        assert!(!has_supported_pot_input_handoff(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert!(
            count >= 5,
            "expected pot.bin, apot.bin, two potNN.dat files, and log1.dat"
        );
        let pot = read_pot_bin(temp.path().join("pot.bin"))?;
        assert_eq!(pot.potential_count(), 2);
        assert_eq!(pot.atomic_numbers.to_vec(), vec![4, 8]);
        assert!(pot.norman_radii[0] > pot.muffin_tin_radii[0]);
        assert!(pot.norman_radii[1] > pot.muffin_tin_radii[1]);
        assert!(pot.scalars.interstitial_density > 0.0);
        assert!(pot.scalars.interstitial_potential.is_finite());
        assert!(pot.large_components.iter().any(|value| *value != 0.0));
        assert!(temp.path().join("apot.bin").is_file());
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(temp.path().join("pot01.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_writes_outputs_from_cached_state() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert!(temp.path().join("pot00.dat").is_file());
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_does_not_advertise_malformed_pot_bin_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        std::fs::write(temp.path().join("pot.bin"), "not pot.bin\n")?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        assert!(!has_cached_pot_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("malformed pot.bin should fail through the explicit cached POT runner")?;
        let chain = format!("{error:?}");
        assert!(
            chain.contains("failed to run supported wpot stage from POT caches"),
            "{chain}"
        );
        assert!(chain.contains("pot.bin"), "{chain}");
        Ok(())
    }

    #[test]
    fn pot_module_does_not_advertise_incomplete_apot_bin_cache() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(
            temp.path().join("apot.bin"),
            &ApotBinData {
                sections: vec![sample_core_hole_section()],
            },
        )?;

        assert!(!has_cached_pot_output(temp.path())?);

        let error = run_in_dir(temp.path())
            .err()
            .context("incomplete apot.bin should fail through the explicit cached POT runner")?;
        let chain = format!("{error:?}");
        assert!(
            chain.contains("failed to run supported wpot stage from POT caches"),
            "{chain}"
        );
        assert!(
            chain.contains("failed to render FEFF wpot potential outputs"),
            "{chain}"
        );
        Ok(())
    }

    #[test]
    fn pot_module_recovers_malformed_cached_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        std::fs::write(temp.path().join("log1.dat"), [0xff, 0xfe, 0xfd])?;

        assert!(!has_cached_pot_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_skips_disabled_cached_state() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_input(temp.path(), 0)?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;

        assert!(!has_cached_pot_output(temp.path())?);

        let count = run_in_dir(temp.path())?;

        assert_eq!(count, 0);
        assert!(!temp.path().join("pot00.dat").exists());
        assert!(!temp.path().join("log1.dat").exists());
        Ok(())
    }

    #[test]
    fn pot_module_preserves_cached_pot_module_log() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        let expected = ModuleLogData {
            lines: vec![
                "Calculating SCF potentials ...".to_string(),
                "cached FEFF POT detail".to_string(),
                "Done with module: potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 3],
        };
        write_module_log_dat(temp.path().join("log1.dat"), &expected)?;

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert_eq!(read_module_log_dat(temp.path().join("log1.dat"))?, expected);
        Ok(())
    }

    #[test]
    fn pot_module_replaces_atomic_log_when_cached_pot_stage_runs() -> Result<()> {
        let temp = tempfile::tempdir()?;
        write_pot_bin(temp.path().join("pot.bin"), &sample_pot_bin())?;
        write_apot_bin(temp.path().join("apot.bin"), &sample_apot_bin())?;
        write_module_log_dat(temp.path().join("log1.dat"), &sample_atomic_module_log())?;

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert_eq!(
            read_module_log_dat(temp.path().join("log1.dat"))?,
            sample_pot_module_log()
        );
        Ok(())
    }

    #[test]
    fn pot_module_roundtrips_generated_reference_when_present() -> Result<()> {
        let Some(reference_dir) = reference_pot_dir()? else {
            crate::require_fixture!("POT reference test; generated EXAFS/Cu reference not found");
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.bin", "apot.bin"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }

        let count = run_in_dir(temp.path())?;

        assert!(count > 0);
        assert!(temp.path().join("pot00.dat").is_file());
        assert!(temp.path().join("log1.dat").is_file());
        Ok(())
    }

    #[test]
    fn pot_module_generates_reference_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(reference_dir) = reference_pot_dir()? else {
            crate::require_fixture!(
                "POT source reference test; generated EXAFS/Cu reference not found"
            );
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(reference_dir.join(name), temp.path().join(name))?;
        }
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        assert!(!has_cached_pot_output(temp.path())?);

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 5);
        for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }

        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(reference_dir.join("pot.bin"))?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(
            generated
                .norman_radii
                .iter()
                .zip(generated.muffin_tin_radii.iter())
                .all(|(norman, muffin_tin)| norman.is_finite()
                    && muffin_tin.is_finite()
                    && norman > muffin_tin)
        );
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        assert!(
            generated
                .valence_occupancy
                .iter()
                .any(|value| *value != 0.0)
        );
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_gecl4_true_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(source_dir) = reference_xanes_gecl4_source_dir()? else {
            crate::require_fixture!("POT GeCl4 true-SCF source reference test; source not found");
        };
        let Some(zip_path) = reference_xanes_gecl4_pot_zip()? else {
            crate::require_fixture!(
                "POT GeCl4 true-SCF source reference test; reference zip not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!(
                "POT GeCl4 true-SCF source reference test; unzip command not found"
            );
        }

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert!(
            input.run.nscmt > 0,
            "GeCl4 reference should exercise the SCF branch"
        );
        input.run.nscmt = 1;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 5);
        for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_electron_density_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_nio_hubbard_true_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(source_dir) = reference_hubbard_nio_source_dir()? else {
            crate::require_fixture!("POT NiO true-SCF source reference test; source not found");
        };
        let Some(zip_path) = reference_hubbard_nio_pot_zip()? else {
            crate::require_fixture!(
                "POT NiO true-SCF source reference test; reference zip not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!(
                "POT NiO true-SCF source reference test; unzip command not found"
            );
        }

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 2,
            "NiO reference should exercise multiple unique potentials"
        );
        assert_eq!(
            input.run.nohole, -1,
            "NiO reference should exercise the screened core-hole branch"
        );
        assert!(
            input.run.nscmt > 0,
            "NiO reference should exercise the SCF branch"
        );
        input.run.nscmt = 2;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 6);
        for name in [
            "pot.bin",
            "apot.bin",
            "pot00.dat",
            "pot01.dat",
            "pot02.dat",
            "log1.dat",
        ] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_electron_density_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        assert!(
            generated
                .valence_occupancy
                .iter()
                .any(|value| *value != 0.0)
        );
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_matches_nio_hubbard_bounded_feff_reference_when_present() -> Result<()> {
        let Some(reference_pot) = reference_hubbard_nio_bounded_feff_pot_bin()? else {
            crate::require_fixture!(
                "POT NiO bounded FEFF parity test; no REFEFF_NIO_BOUNDED_FEFF_POT_BIN or reference-work/tmp/feff-pot-nio-bounded.*/pot.bin found"
            );
        };
        let Some(source_dir) = reference_hubbard_nio_source_dir()? else {
            crate::require_fixture!("POT NiO bounded FEFF parity test; source not found");
        };

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert!(input.run.nscmt > 2);
        input.run.nscmt = 2;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 6);
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(reference_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        Ok(())
    }

    #[test]
    fn pot_module_generates_ldos_spin_true_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(source_dir) = reference_ldos_cu_spin_source_dir()? else {
            crate::require_fixture!(
                "POT LDOS spin true-SCF source reference test; source not found"
            );
        };

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 1,
            "LDOS spin Cu reference should exercise two potential columns"
        );
        assert_eq!(
            input.run.nohole, 2,
            "LDOS spin Cu reference should exercise final-state screening"
        );
        assert!(
            input.run.nscmt > 0,
            "LDOS spin Cu reference should exercise the SCF branch"
        );
        input.run.nscmt = 1;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 5);
        for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(source_dir.join("pot.bin"))?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.len(),
            reference.potential_multiplicities.len()
        );
        assert!(
            generated
                .potential_multiplicities
                .iter()
                .all(|multiplicity| multiplicity.is_finite() && *multiplicity > 0.0)
        );
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        assert!(
            generated
                .valence_occupancy
                .iter()
                .any(|value| *value != 0.0)
        );
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_bn_positive_totvol_true_scf_outputs_from_source_handoffs() -> Result<()>
    {
        let Some(source_dir) = reference_bn_source_dir()? else {
            crate::require_fixture!("POT BN true-SCF source reference test; source not found");
        };
        let Some(zip_path) = reference_bn_pot_zip()? else {
            crate::require_fixture!(
                "POT BN true-SCF source reference test; reference zip not found"
            );
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!(
                "POT BN true-SCF source reference test; unzip command not found"
            );
        }

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 2,
            "BN reference should exercise three potential columns"
        );
        assert!(
            input.scattering.totvol > 0.0,
            "BN reference should exercise positive totvol"
        );
        assert!(
            input.run.nscmt > 0,
            "BN reference should exercise the SCF branch"
        );
        input.run.nscmt = 1;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 6);
        for name in [
            "pot.bin",
            "apot.bin",
            "pot00.dat",
            "pot01.dat",
            "pot02.dat",
            "log1.dat",
        ] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_close_values(
            "POT total volume",
            [generated.scalars.total_volume],
            [reference.scalars.total_volume],
            1.0e-10,
        );
        assert_pot_bin_reference_geometry_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_matches_bn_positive_totvol_bounded_feff_reference_when_present() -> Result<()> {
        let Some(reference_pot) = reference_bn_positive_totvol_bounded_feff_pot_bin()? else {
            crate::require_fixture!(
                "POT BN bounded FEFF parity test; no REFEFF_BN_POSITIVE_TOTVOL_BOUNDED_FEFF_POT_BIN or reference-work/tmp/feff-pot-bn-positive-totvol-bounded.*/pot.bin found"
            );
        };
        let Some(source_dir) = reference_bn_source_dir()? else {
            crate::require_fixture!("POT BN bounded FEFF parity test; source not found");
        };

        let temp = tempfile::tempdir()?;
        let source_pot = source_dir.join("pot.inp");
        let mut input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 2,
            "BN reference should exercise three potential columns"
        );
        assert!(
            input.scattering.totvol > 0.0,
            "BN reference should exercise positive totvol"
        );
        assert!(input.run.nscmt > 1);
        input.run.nscmt = 1;
        std::fs::write(temp.path().join("pot.inp"), pot_input_string(&input)?)?;
        std::fs::copy(source_dir.join("geom.dat"), temp.path().join("geom.dat"))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 6);
        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(reference_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_close_values(
            "POT total volume",
            [generated.scalars.total_volume],
            [reference.scalars.total_volume],
            1.0e-10,
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_sf6_reference_no_scf_outputs_from_source_zip() -> Result<()> {
        let Some(zip_path) = reference_sf6_pot_zip()? else {
            crate::require_fixture!("POT SF6 source reference test; reference zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("POT SF6 source reference test; unzip command not found");
        }
        let Some(source_dir) = reference_sf6_source_dir()? else {
            crate::require_fixture!(
                "POT SF6 source reference test; generated source handoffs not found"
            );
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(source_dir.join(name), temp.path().join(name))?;
        }
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        assert!(!has_cached_pot_output(temp.path())?);

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 5);
        for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }

        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_ybco_reference_no_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(source_dir) = reference_ybco_source_dir()? else {
            crate::require_fixture!("POT YBCO source reference test; source not found");
        };
        let Some(zip_path) = reference_ybco_pot_zip()? else {
            crate::require_fixture!("POT YBCO source reference test; reference zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("POT YBCO source reference test; unzip command not found");
        }

        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(source_dir.join(name), temp.path().join(name))?;
        }
        let source_pot = source_dir.join("pot.inp");
        let input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 4,
            "YBCO reference should exercise five potential columns"
        );
        assert_eq!(input.run.nscmt, 0, "YBCO reference should be no-SCF");
        assert_eq!(
            input.run.nohole, -1,
            "YBCO reference should exercise screened core-hole bookkeeping"
        );
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        assert!(!has_cached_pot_output(temp.path())?);

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 8);
        for name in [
            "pot.bin",
            "apot.bin",
            "pot00.dat",
            "pot01.dat",
            "pot02.dat",
            "pot03.dat",
            "pot04.dat",
            "log1.dat",
        ] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }

        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_mnf2_xmcd_reference_no_scf_outputs_from_source_zip() -> Result<()> {
        let Some(zip_path) = reference_xmcd_mnf2_pot_zip()? else {
            crate::require_fixture!("POT MnF2 XMCD source reference test; reference zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("POT MnF2 XMCD source reference test; unzip command not found");
        }
        let Some(source_dir) = reference_xmcd_mnf2_source_dir()? else {
            crate::require_fixture!("POT MnF2 XMCD source reference test; source not found");
        };
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(source_dir.join(name), temp.path().join(name))?;
        }
        let source_pot = source_dir.join("pot.inp");
        let input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 3,
            "MnF2 reference should exercise four potential columns"
        );
        assert_eq!(input.run.nscmt, 0, "MnF2 reference should be no-SCF");
        assert_eq!(
            input.run.nohole, -1,
            "MnF2 reference should exercise screened core-hole bookkeeping"
        );
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 7);
        for name in [
            "pot.bin",
            "apot.bin",
            "pot00.dat",
            "pot01.dat",
            "pot02.dat",
            "pot03.dat",
            "log1.dat",
        ] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }

        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    #[test]
    fn pot_module_generates_gd_l1_reference_no_scf_outputs_from_source_handoffs() -> Result<()> {
        let Some(source_dir) = reference_xmcd_gd_l1_source_dir()? else {
            crate::require_fixture!("POT Gd L1 source reference test; source not found");
        };
        let Some(zip_path) = reference_xmcd_gd_l1_pot_zip()? else {
            crate::require_fixture!("POT Gd L1 source reference test; reference zip not found");
        };
        if Command::new("unzip").arg("-v").output().is_err() {
            crate::require_fixture!("POT Gd L1 source reference test; unzip command not found");
        }
        let temp = tempfile::tempdir()?;
        for name in ["pot.inp", "geom.dat"] {
            std::fs::copy(source_dir.join(name), temp.path().join(name))?;
        }
        let source_pot = source_dir.join("pot.inp");
        let input = PotInput::parse_str(&source_pot, &std::fs::read_to_string(&source_pot)?)?;
        assert_eq!(
            input.control.nph, 1,
            "Gd L1 reference should use two potentials"
        );
        assert_eq!(input.run.nscmt, 0, "Gd L1 reference should be no-SCF");
        assert_eq!(
            input.run.nohole, -1,
            "Gd L1 reference should exercise screened core-hole bookkeeping"
        );
        let expected_pot = temp.path().join("expected-pot.bin");
        std::fs::write(
            &expected_pot,
            unzip_reference_entry(&zip_path, "REFERENCE/pot.bin")?,
        )?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(&input_path, "")?;

        let count = run_for_input(&input_path)?;

        assert_eq!(count, 5);
        for name in ["pot.bin", "apot.bin", "pot00.dat", "pot01.dat", "log1.dat"] {
            assert!(
                temp.path().join(name).is_file(),
                "expected source-backed POT run to write {name}"
            );
        }

        let generated = read_pot_bin(temp.path().join("pot.bin"))?;
        let reference = read_pot_bin(&expected_pot)?;
        assert_eq!(generated.potential_count(), reference.potential_count());
        assert_eq!(
            generated.atomic_numbers.to_vec(),
            reference.atomic_numbers.to_vec()
        );
        assert_eq!(generated.nohole, reference.nohole);
        assert_eq!(generated.ihole, reference.ihole);
        assert_eq!(
            generated.potential_multiplicities.to_vec(),
            reference.potential_multiplicities.to_vec()
        );
        assert_pot_bin_reference_rows_close(&generated, &reference);
        assert!(generated.scalars.fermi_level.is_finite());
        assert!(generated.scalars.interstitial_density > 0.0);
        assert!(generated.electron_density.iter().any(|value| *value != 0.0));
        read_apot_bin(temp.path().join("apot.bin"))?;
        assert!(has_cached_pot_output(temp.path())?);
        Ok(())
    }

    fn assert_pot_bin_reference_rows_close(generated: &PotBinData, reference: &PotBinData) {
        assert_pot_bin_reference_electron_density_rows_close(generated, reference);
        assert_close_values(
            "POT valence density",
            generated.valence_density.iter().copied(),
            reference.valence_density.iter().copied(),
            1.0,
        );
    }

    fn assert_pot_bin_reference_electron_density_rows_close(
        generated: &PotBinData,
        reference: &PotBinData,
    ) {
        assert_pot_bin_reference_geometry_rows_close(generated, reference);
        assert_close_values(
            "POT electron density",
            generated.electron_density.iter().copied(),
            reference.electron_density.iter().copied(),
            1.0,
        );
    }

    fn assert_pot_bin_reference_geometry_rows_close(
        generated: &PotBinData,
        reference: &PotBinData,
    ) {
        assert_close_values(
            "POT muffin-tin radii",
            generated.muffin_tin_radii.iter().copied(),
            reference.muffin_tin_radii.iter().copied(),
            2.5e-1,
        );
        assert_close_values(
            "POT Norman radii",
            generated.norman_radii.iter().copied(),
            reference.norman_radii.iter().copied(),
            2.5e-1,
        );
        assert_close_values(
            "POT overlap factors",
            generated.overlap_factors.iter().copied(),
            reference.overlap_factors.iter().copied(),
            2.5e-1,
        );
    }

    fn assert_close_values<A, E>(label: &str, actual: A, expected: E, relative_tolerance: f64)
    where
        A: IntoIterator<Item = f64>,
        E: IntoIterator<Item = f64>,
    {
        let actual = actual.into_iter().collect::<Vec<_>>();
        let expected = expected.into_iter().collect::<Vec<_>>();
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} length changed from FEFF reference"
        );

        let mut compared = 0usize;
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if !actual.is_finite() || !expected.is_finite() {
                assert!(
                    actual == expected,
                    "{label}[{index}] finite mismatch: actual={actual}, expected={expected}"
                );
                continue;
            }
            compared += 1;
            let allowed = relative_tolerance * expected.abs().max(1.0);
            let diff = (actual - expected).abs();
            assert!(
                diff <= allowed,
                "{label}[{index}] differs from FEFF reference: actual={actual}, expected={expected}, diff={diff}, allowed={allowed}"
            );
        }
        assert!(compared > 0, "{label} comparison was empty");
    }

    fn sample_pot_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating SCF potentials ...".to_string(),
                "FEFF-serial using 1 thread.".to_string(),
                "Done with module: potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 3],
        }
    }

    fn sample_atomic_module_log() -> ModuleLogData {
        ModuleLogData {
            lines: vec![
                "Calculating atomic potentials ...".to_string(),
                "Done with module: atomic potentials.".to_string(),
            ],
            line_terminators: vec!["\n".to_string(); 2],
        }
    }

    fn write_pot_input(work_dir: &Path, mpot: i32) -> Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu POT smoke test
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
        std::fs::write(work_dir.join("pot.inp"), pot_input_string(&pot_input)?)?;
        Ok(())
    }

    fn write_pot_input_with_nan_gamach(work_dir: &Path) -> Result<()> {
        write_pot_input(work_dir, 1)?;
        let path = work_dir.join("pot.inp");
        let mut lines = std::fs::read_to_string(&path)?
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let header = lines
            .iter()
            .position(|line| line.trim() == "gamach, rgrd, ca1, ecv, totvol, rfms1, corval_emin")
            .context("pot.inp should contain the scattering header")?;
        let values = lines
            .get(header + 1)
            .context("pot.inp should contain the scattering value line")?
            .split_whitespace()
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 7);
        lines[header + 1] = format!(
            "{:>13}{:>13}{:>13}{:>13}{:>13}{:>13}{:>13}",
            "NaN", values[1], values[2], values[3], values[4], values[5], values[6]
        );
        std::fs::write(path, format!("{}\n", lines.join("\n")))?;
        Ok(())
    }

    fn write_beryllium_pot_source_handoffs(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("pot.inp"),
            pot_input_string(&beryllium_pot_input()?)?,
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

    fn write_beryllium_iterative_scf_source_handoffs(work_dir: &Path) -> Result<()> {
        std::fs::write(
            work_dir.join("pot.inp"),
            pot_input_string(&beryllium_iterative_scf_pot_input()?)?,
        )?;
        std::fs::write(
            work_dir.join("geom.dat"),
            geom_dat_string(&beryllium_single_potential_geom_dat())?,
        )?;
        Ok(())
    }

    fn pot_scf_cache_path(work_dir: &Path) -> PathBuf {
        work_dir.join(".refeff-pot-scf-cache")
    }

    fn beryllium_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be POT source sidecar smoke test
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
        input.control.ihole = 0;
        input.run.nohole = 0;
        Ok(input)
    }

    fn beryllium_no_scf_pot_input() -> Result<PotInput> {
        let mut input = beryllium_pot_input()?;
        input.control.ixc = 2;
        input.run.nscmt = 0;
        Ok(input)
    }

    fn beryllium_default_exchange_no_scf_pot_input() -> Result<PotInput> {
        let mut input = beryllium_pot_input()?;
        assert_eq!(input.control.ixc, 0);
        input.run.nscmt = 0;
        Ok(input)
    }

    fn beryllium_high_exchange_no_scf_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be high-exchange no-SCF POT source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 6 0.0 0.0
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        Ok(PotInput::parse_str(
            "pot.inp",
            &rdinp::pot_inp_string(&document)?,
        )?)
    }

    fn beryllium_iterative_scf_pot_input() -> Result<PotInput> {
        let mut input = beryllium_pot_input()?;
        input.run.nscmt = 2;
        Ok(input)
    }

    fn beryllium_high_exchange_iterative_scf_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be high-exchange iterative POT SCF source run
EDGE K
CONTROL 1 1 1 1 1 1
EXCHANGE 5 0.0 0.0
SCF 5.0 0 2 0.2
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        Ok(PotInput::parse_str(
            "pot.inp",
            &rdinp::pot_inp_string(&document)?,
        )?)
    }

    fn beryllium_core_hole_no_scf_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Be POT core-hole no-SCF source smoke test
EDGE K
CONTROL 1 0 0 0 0 0
EXCHANGE 2 0.0 0.0
POTENTIALS
0 4 Be
ATOMS
0.0 0.0 0.0 0 Be0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        input.run.nscmt = 0;
        Ok(input)
    }

    fn beryllium_oxygen_no_scf_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE BeO POT no-SCF source smoke test
EDGE K
CONTROL 1 0 0 0 0 0
POTENTIALS
0 4 Be
1 8 O
ATOMS
0.0 0.0 0.0 0 Be0
1.6 0.0 0.0 1 O1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        input.control.ihole = 0;
        input.control.ixc = 2;
        input.run.nohole = 0;
        input.run.nscmt = 0;
        Ok(input)
    }

    fn copper_custom_config_no_scf_pot_input() -> Result<PotInput> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu custom-config POT source smoke test
EDGE K
CONTROL 1 1 1 1 1 1
CONFIG card 1
0 Cu 1s -2 2s -2 2p -2 -4 3s -1 3p -2 -4 3d 4 6 4s 1 4p 0 0
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
        let mut input = PotInput::parse_str("pot.inp", &rdinp::pot_inp_string(&document)?)?;
        input.run.nscmt = 0;
        Ok(input)
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

    fn copper_two_potential_geom_dat() -> GeomDat {
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
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 0,
                },
            ],
        }
    }

    fn beryllium_oxygen_geom_dat() -> GeomDat {
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
                    x: 1.6,
                    y: 0.0,
                    z: 0.0,
                    iph: 1,
                    boundary: 0,
                },
            ],
        }
    }

    fn beryllium_single_potential_pot_bin() -> PotBinData {
        let potentials = 1;
        PotBinData {
            titles: vec!["POT APOT source sidecar Be smoke test".to_string()],
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

    fn sample_mtdp_data() -> MtdpData {
        MtdpData {
            radial_count: 3,
            atomic_numbers: Array1::from_vec(vec![4]),
            atom_coordinates: Array2::zeros((1, 3)),
            atom_radii: Array1::from_vec(vec![1.25]),
            atom_radius_indices: Array1::from_vec(vec![7]),
            atom_density: Array2::from_shape_vec((3, 1), vec![0.11, 0.12, 0.13])
                .expect("sample MTDP atom density shape"),
            atom_potential: Array2::from_shape_vec((3, 1), vec![-1.0, -1.1, -1.2])
                .expect("sample MTDP atom potential shape"),
            empty_sphere_coordinates: Array2::zeros((0, 3)),
            empty_sphere_radii: Array1::zeros(0),
            empty_sphere_radius_indices: Array1::zeros(0),
            empty_sphere_density: Array2::zeros((3, 0)),
            empty_sphere_potential: Array2::zeros((3, 0)),
            interstitial_potential: -0.75,
            homo_energy: -0.12,
            lumo_energy: -0.08,
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

    fn sample_pot_bin() -> PotBinData {
        let potentials = 2;
        PotBinData {
            titles: vec!["POT smoke test".to_string()],
            pad_width: 8,
            nohole: -1,
            ihole: 1,
            interstitial_selector: 0,
            automatic_folp: 0,
            jump_mode: 0,
            unfreeze_f: 0,
            scalars: PotBinScalars {
                average_norman_radius: 1.0,
                fermi_level: -0.25,
                interstitial_potential: -0.15,
                interstitial_density: 0.05,
                edge_position: 0.1,
                amplitude_reduction: 0.9,
                relaxation_energy: 0.01,
                plasmon_frequency: 0.02,
                core_valence_energy: -1.0,
                density_radius: 2.0,
                fermi_momentum: 1.2,
                total_charge: 10.0,
                total_volume: 20.0,
            },
            muffin_tin_indices: ndarray::arr1(&[20, 21]),
            muffin_tin_radii: ndarray::arr1(&[1.0, 1.2]),
            norman_indices: ndarray::arr1(&[30, 31]),
            atomic_numbers: ndarray::arr1(&[29, 29]),
            kappa: ndarray::Array1::zeros(POT_BIN_ORBITALS),
            norman_radii: ndarray::arr1(&[1.4, 1.5]),
            overlap_factors: ndarray::arr1(&[1.1, 1.1]),
            max_overlap_factors: ndarray::arr1(&[1.4, 1.4]),
            potential_multiplicities: ndarray::arr1(&[1.0, 12.0]),
            ionization: ndarray::arr1(&[0.0, 0.0]),
            initial_large_component: ndarray::Array1::zeros(POT_BIN_RADIAL_POINTS),
            initial_small_component: ndarray::Array1::zeros(POT_BIN_RADIAL_POINTS),
            large_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            small_components: Array3::zeros((POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials)),
            large_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            small_coefficients: Array3::zeros((POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials)),
            electron_density: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| 0.02 * (row + 1) as f64,
            ),
            coulomb_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, potential)| -0.3 - 0.01 * row as f64 - 0.2 * potential as f64,
            ),
            total_potential: Array2::from_shape_fn(
                (POT_BIN_RADIAL_POINTS, potentials),
                |(row, _)| -0.1 + 0.001 * row as f64,
            ),
            valence_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            valence_potential: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            magnetization_density: Array2::zeros((POT_BIN_RADIAL_POINTS, potentials)),
            orbital_occupancy: Array2::zeros((POT_BIN_ORBITALS, potentials)),
            orbital_energies: ndarray::Array1::zeros(POT_BIN_ORBITALS),
            occupied_orbital_indices: Array2::zeros((POT_BIN_IORB_SLOTS, potentials)),
            norman_charges: ndarray::Array1::zeros(potentials),
            valence_occupancy: Array2::zeros((4, potentials)),
            raw_text: None,
        }
    }

    fn add_to_first_real_apot_matrix_value(apot: &mut ApotBinData, delta: f64) {
        let Some(values) = apot.sections.iter_mut().find_map(|section| {
            if let ApotBinPayload::Matrix(matrix) = &mut section.payload
                && let ApotBinMatrixValues::Real(values) = &mut matrix.values
            {
                return Some(values);
            }
            None
        }) else {
            panic!("sample apot.bin should contain a real matrix section");
        };
        values[(0, 0)] += delta;
    }

    fn sample_apot_bin() -> ApotBinData {
        ApotBinData {
            sections: vec![
                sample_core_hole_section(),
                ApotBinSection {
                    section_number: 8,
                    headers: vec![
                        "rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
                    ],
                    header_texts: vec![
                        " rho(r,0:nphx+1) - atomic density for each unique potential".to_string(),
                    ],
                    column_labels: vec![],
                    column_label_text: None,
                    payload: ApotBinPayload::Matrix(ApotBinMatrix {
                        value_type: ApotBinType::Double,
                        values: ApotBinMatrixValues::Real(Array2::from_shape_fn(
                            (POT_BIN_RADIAL_POINTS, 2),
                            |(row, potential)| 0.01 * (row + 1) as f64 + 0.1 * potential as f64,
                        )),
                    }),
                    trailing_headers: vec![],
                    trailing_header_texts: vec![],
                },
                ApotBinSection {
                    section_number: 11,
                    headers: vec![
                        "vcoul(r,nph) - coulomb potential for each unique potential.".to_string(),
                    ],
                    header_texts: vec![
                        " vcoul(r,nph) - coulomb potential for each unique potential.".to_string(),
                    ],
                    column_labels: vec![],
                    column_label_text: None,
                    payload: ApotBinPayload::Matrix(ApotBinMatrix {
                        value_type: ApotBinType::Double,
                        values: ApotBinMatrixValues::Real(Array2::from_shape_fn(
                            (POT_BIN_RADIAL_POINTS, 2),
                            |(row, potential)| -0.5 * (potential + 1) as f64 - 0.02 * row as f64,
                        )),
                    }),
                    trailing_headers: vec![],
                    trailing_header_texts: vec![],
                },
            ],
        }
    }

    fn sample_core_hole_section() -> ApotBinSection {
        ApotBinSection {
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
                            ApotBinValue::Real(0.0),
                            ApotBinValue::Real(0.0),
                        ]
                    })
                    .collect(),
            }),
            trailing_headers: vec![],
            trailing_header_texts: vec![],
        }
    }

    fn reference_pot_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/Cu"));
        Ok(reference
            .filter(|path| path.join("pot.bin").is_file() && path.join("apot.bin").is_file()))
    }

    fn reference_sf6_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/SF6/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_sf6_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/SF6"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_ybco_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/YBCO/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_ybco_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/EXAFS/YBCO"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_xanes_gecl4_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/GeCl_4"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_xanes_gecl4_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/GeCl_4/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_hubbard_nio_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/HUBBARD/NiO"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_hubbard_nio_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/HUBBARD/NiO/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_hubbard_nio_bounded_feff_pot_bin() -> Result<Option<PathBuf>> {
        if let Some(path) = std::env::var_os("REFEFF_NIO_BOUNDED_FEFF_POT_BIN") {
            let path = PathBuf::from(path);
            return Ok(path.is_file().then_some(path));
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let Some(root) = manifest_dir.parent().and_then(Path::parent) else {
            return Ok(None);
        };
        let tmp_dir = root.join("reference-work/tmp");
        if !tmp_dir.is_dir() {
            return Ok(None);
        }

        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(&tmp_dir)
            .with_context(|| format!("failed to read {}", tmp_dir.display()))?
        {
            let entry = entry.with_context(|| format!("failed to read {}", tmp_dir.display()))?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let pot_bin = path.join("pot.bin");
            if name.starts_with("feff-pot-nio-bounded.") && pot_bin.is_file() {
                candidates.push(pot_bin);
            }
        }
        candidates.sort();
        Ok(candidates.pop())
    }

    fn reference_ldos_cu_spin_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/LDOS/XANES_Cu_spin_no_fms"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_bn_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/BN"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_bn_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XANES/BN/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_bn_positive_totvol_bounded_feff_pot_bin() -> Result<Option<PathBuf>> {
        if let Some(path) = std::env::var_os("REFEFF_BN_POSITIVE_TOTVOL_BOUNDED_FEFF_POT_BIN") {
            let path = PathBuf::from(path);
            return Ok(path.is_file().then_some(path));
        }

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let Some(root) = manifest_dir.parent().and_then(Path::parent) else {
            return Ok(None);
        };
        let tmp_dir = root.join("reference-work/tmp");
        if !tmp_dir.is_dir() {
            return Ok(None);
        }

        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(&tmp_dir)
            .with_context(|| format!("failed to read {}", tmp_dir.display()))?
        {
            let entry = entry.with_context(|| format!("failed to read {}", tmp_dir.display()))?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let pot_bin = path.join("pot.bin");
            if name.starts_with("feff-pot-bn-positive-totvol-bounded.") && pot_bin.is_file() {
                candidates.push(pot_bin);
            }
        }
        candidates.sort();
        Ok(candidates.pop())
    }

    fn reference_xmcd_gd_l1_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XMCD/Gd_L1/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_xmcd_mnf2_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XMCD/MnF2_SPXAS"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn reference_xmcd_mnf2_pot_zip() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XMCD/MnF2_SPXAS/REFERENCE.zip"));
        Ok(reference.filter(|path| path.is_file()))
    }

    fn reference_xmcd_gd_l1_source_dir() -> Result<Option<PathBuf>> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(|root| root.join("reference-work/golden/XMCD/Gd_L1"));
        Ok(reference
            .filter(|path| path.join("pot.inp").is_file() && path.join("geom.dat").is_file()))
    }

    fn unzip_reference_entry(zip_path: &Path, entry: &str) -> Result<Vec<u8>> {
        let output = Command::new("unzip")
            .args(["-p"])
            .arg(zip_path)
            .arg(entry)
            .output()
            .with_context(|| format!("failed to extract {entry} from {}", zip_path.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "failed to extract {entry} from {}: {stderr}",
                zip_path.display()
            );
        }
        Ok(output.stdout)
    }
}
