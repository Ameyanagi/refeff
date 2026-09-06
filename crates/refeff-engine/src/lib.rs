#![forbid(unsafe_code)]
// EXAFS reuses selected POT/SCREEN helpers from the FMS implementation file.
// The remaining private helpers are intentionally unreachable from the
// reduced scheduler and removed by release dead-code elimination.
#![cfg_attr(not(feature = "full"), allow(dead_code))]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

mod atomic;
#[cfg(feature = "full")]
mod band;
#[cfg(feature = "full")]
mod compton;
#[cfg(feature = "full")]
mod crpa;
#[cfg(feature = "full")]
mod dmdw;
#[cfg(feature = "full")]
mod eels;
#[cfg(feature = "full")]
mod eelsmdff;
mod ff2x;
mod fms;
#[cfg(feature = "full")]
mod fullspectrum;
mod genfmt;
#[cfg(feature = "full")]
mod ldos;
#[cfg(feature = "full")]
mod opcons;
mod paths;
mod pot;
#[cfg(feature = "full")]
mod rhorrp;
#[cfg(feature = "full")]
mod rixs;
mod screen;
#[cfg(feature = "sfconv")]
mod sfconv;
mod wpot;
mod xsph;

mod cache;
mod scheduler;
use scheduler::*;
pub mod execution;
mod plan;
pub use plan::{ExecutionPlan, PlannedStage, plan};

use std::cell::Cell;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use refeff_io::{FeffDocument, FeffInput, rdinp};
use serde::Serialize;

/// Typed engine errors that callers may inspect through an `anyhow` error's
/// source chain.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A module name is not part of the FEFF engine surface.
    #[error("unsupported FEFF module `{module}`")]
    UnsupportedModule {
        /// Rejected module spelling.
        module: String,
    },
    /// A known stage is unavailable in the selected Cargo feature set.
    #[error("FEFF module `{module}` requires Cargo feature `{feature}`")]
    FeatureDisabled {
        /// Canonical FEFF module name.
        module: &'static str,
        /// Cargo feature that enables the module.
        feature: &'static str,
    },
}

mod modules;
pub use modules::{MODULES, ModuleDescriptor, ModuleName};

fn ensure_module_available(module: ModuleName) -> Result<()> {
    if let Some(feature) = module.disabled_feature() {
        return Err(EngineError::FeatureDisabled {
            module: module.as_str(),
            feature,
        }
        .into());
    }
    Ok(())
}

#[cfg(not(feature = "full"))]
fn feature_disabled(module: ModuleName, feature: &'static str) -> Result<()> {
    Err(EngineError::FeatureDisabled {
        module: module.as_str(),
        feature,
    }
    .into())
}

/// Summary of the parsed input handled by the `rdinp` compatibility stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RdinpReport {
    /// Number of active FEFF cards parsed from the input.
    pub cards: usize,
    /// Number of atoms extracted from the `ATOMS` table.
    pub atoms: usize,
    /// Number of unique potential rows extracted from `POTENTIALS`.
    pub potentials: usize,
    /// FEFF-style RDINP stdout summary, when currently renderable.
    pub stdout: Option<String>,
}

/// Machine-readable report for `refeff check` (`--json`), also used to
/// render the human `OK: ...` summary line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckReport {
    /// Number of active FEFF cards parsed from the input.
    pub cards: usize,
    /// Number of atoms extracted from the `ATOMS` table.
    pub atoms: usize,
    /// Number of unique potential rows extracted from `POTENTIALS`.
    pub potentials: usize,
    /// Selected absorption edge label, when present.
    pub edge: Option<String>,
    /// FEFF10 pipeline modules the parsed `CONTROL` switches enable (all six
    /// are enabled when `CONTROL` is absent, matching FEFF10's default).
    pub modules_enabled: Vec<&'static str>,
}

/// The FEFF10 module names `CONTROL`'s six switches enable, in FEFF10's
/// `mpot, mphase, mfms, mpath, mfeff, mchi` order (`feff10/src/RDINP/rdinp.f90`).
const CONTROL_MODULE_NAMES: [&str; 6] = ["pot", "xsph", "fms", "path", "genfmt", "ff2x"];

fn modules_enabled(control: Option<[i32; 6]>) -> Vec<&'static str> {
    let switches = control.unwrap_or([1, 1, 1, 1, 1, 1]);
    CONTROL_MODULE_NAMES
        .into_iter()
        .zip(switches)
        .filter(|(_, switch)| *switch != 0)
        .map(|(name, _)| name)
        .collect()
}

/// Output controls installed by a frontend while it invokes engine runners.
///
/// Embedders normally use the quiet [`execute_feff`] boundary and do not need
/// to configure this. It remains public so `refeff-cli` can preserve its
/// human-readable and JSON output contracts without the engine depending on
/// Clap.
/// via a thread-local rather than an extra parameter, since those functions
/// are also called directly from tests and library entry points.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputMode {
    pub verbose: bool,
    pub quiet: bool,
    pub json: bool,
}

thread_local! {
    static OUTPUT_MODE: Cell<OutputMode> = const {
        Cell::new(OutputMode {
            verbose: false,
            quiet: false,
            json: false,
        })
    };
}

/// Restores the previous thread-local [`OutputMode`] on drop.
struct OutputModeGuard {
    previous: OutputMode,
}

impl Drop for OutputModeGuard {
    fn drop(&mut self) {
        OUTPUT_MODE.with(|cell| cell.set(self.previous));
    }
}

fn set_output_mode(mode: OutputMode) -> OutputModeGuard {
    let previous = OUTPUT_MODE.with(|cell| cell.replace(mode));
    OutputModeGuard { previous }
}

/// Run an operation with a temporary frontend output mode.
///
/// The previous mode is restored even if `run` returns an error or unwinds.
pub fn with_output_mode<T>(mode: OutputMode, run: impl FnOnce() -> T) -> T {
    let _guard = set_output_mode(mode);
    run()
}

fn current_output_mode() -> OutputMode {
    OUTPUT_MODE.with(Cell::get)
}

/// Serializes `value` as pretty JSON to stdout (the `--json` machine-readable
/// report for the current command).
fn emit_json<T: Serialize>(value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value).context("failed to serialize JSON report")?;
    println!("{json}");
    Ok(())
}

/// Prints a single-module status line (the one-module equivalent of
/// [`print_stage_line`]), honoring `-q/--quiet` and `--json` — both suppress
/// it, `--json` because `refeff module <name>` emits a structured report
/// instead (see [`run_module`]). Shared by every `run_<module>` function, so
/// it also governs the standalone per-module binaries (`bin/pot.rs`, ...),
/// where the thread-local [`OutputMode`] is always its quiet-off/json-off
/// default.
fn print_module_line(message: std::fmt::Arguments<'_>) {
    let mode = current_output_mode();
    if mode.quiet {
        return;
    }
    // `--json` reserves stdout for the `ModuleReport` document `run_module`
    // emits afterward, so this (normally stdout) line moves to stderr
    // instead of disappearing.
    if mode.json {
        eprintln!("{message}");
    } else {
        println!("{message}");
    }
}

/// Resolves `--threads`/`REFEFF_THREADS`, builds the global `rayon` thread
/// pool once, and mirrors the bound into `refeff_linalg::set_parallelism` so
/// `faer`'s solvers respect it too. `--threads 1` therefore gives a fully
/// serial, deterministic run. A no-op when neither `--threads` nor
/// `REFEFF_THREADS` is set, leaving `rayon`/`faer` at their own defaults.
/// `rayon`'s global pool can only be built once per process; a second call
/// warns and continues rather than failing the run.
fn configure_threads(threads: Option<usize>) {
    let Some(threads) = threads.or_else(|| {
        std::env::var("REFEFF_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    }) else {
        return;
    };
    if let Err(error) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        eprintln!(
            "warning: rayon global thread pool was already initialized ({error}); \
             --threads/REFEFF_THREADS may not take effect for every worker"
        );
    }
    refeff_linalg::set_parallelism(Some(threads));
}

/// Configure process-wide Rayon and faer parallelism for library runners.
///
/// Calling this after another component initialized Rayon's global pool is
/// harmless, but the requested Rayon bound may then be unable to take effect.
pub fn configure_parallelism(threads: Option<usize>) {
    configure_threads(threads);
}

/// Parses `input` and builds a [`FeffDocument`] from it with **zero
/// filesystem writes** — no `.feff.error` sentinel, no `log.dat`, no handoff
/// files — unlike [`execute_rdinp`], which writes all of those as it runs
/// the real `rdinp` stage. A card-located parse/semantic problem surfaces as
/// an `Err` the same way it would from `rdinp`/`run` (see
/// `refeff_cli::exit_code_for`-equivalent handling in `bin/refeff.rs`, which
/// maps non-`Io` [`refeff_io::IoError`] variants to exit code 3).
pub fn check(input: &Path) -> Result<CheckReport> {
    check_with_options(input, false)
}

/// Validate an input without filesystem writes. Syntax-only accepts partial inputs.
pub fn check_with_options(input: &Path, syntax_only: bool) -> Result<CheckReport> {
    let parsed = FeffInput::parse_file(input)?;
    let document = FeffDocument::from_input(&parsed)?;
    if !syntax_only {
        document.validate_calculation(&parsed)?;
        let control = document.control.unwrap_or([1; 6]);
        for (enabled, module) in [
            (control[0] != 0, ModuleName::Pot),
            (control[1] != 0, ModuleName::Xsph),
            (control[2] != 0 && document.fms.is_some(), ModuleName::Fms),
            (control[3] != 0 && document.ispec != 5, ModuleName::Path),
            (control[4] != 0 && document.ispec != 5, ModuleName::Genfmt),
            (control[5] != 0 && document.ispec != 5, ModuleName::Ff2x),
            (document.eels.enabled, ModuleName::Eels),
            (document.rixs.run, ModuleName::Rixs),
            (document.sfconv, ModuleName::Sfconv),
            (document.ldos.is_some(), ModuleName::Ldos),
            (document.band_input.mband != 0, ModuleName::Band),
            (document.crpa.enabled, ModuleName::Crpa),
            (document.compton.do_compton, ModuleName::Compton),
            (
                document.full_spectrum_input.m_full_spectrum != 0,
                ModuleName::Fullspectrum,
            ),
            (document.opcons, ModuleName::Opcons),
        ] {
            if enabled {
                ensure_module_available(module)?;
            }
        }
    }
    Ok(CheckReport {
        cards: parsed.cards().count(),
        atoms: document.atoms.len(),
        potentials: document.potentials.len(),
        edge: document.edge.as_ref().map(|edge| edge.label.clone()),
        modules_enabled: modules_enabled(document.control),
    })
}

/// `refeff check` (alias `inspect`): validate `input` with no side effects
/// and print either the human `OK: ...` summary or (`--json`) a
/// [`CheckReport`] document.
pub fn run_check(input: PathBuf) -> Result<()> {
    let report = check(&input)?;
    let edge = report.edge.as_deref().unwrap_or("none");
    let modules = if report.modules_enabled.is_empty() {
        "none".to_string()
    } else {
        report.modules_enabled.join(", ")
    };
    let line = format!(
        "OK: {} cards, {} atoms, {} potentials, edge={edge}, modules enabled: {modules}",
        report.cards, report.atoms, report.potentials
    );
    if current_output_mode().json {
        eprintln!("{line}");
        emit_json(&report)
    } else {
        println!("{line}");
        Ok(())
    }
}

/// Run the supported FEFF `rdinp` compatibility stage in the current directory.
pub fn run_rdinp(input: PathBuf, output: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Rdinp)?;
    let report = execute_rdinp(&input, &output)?;
    if current_output_mode().json {
        return emit_json(&report);
    }
    print_rdinp_summary(&report)
}

/// Prints the `rdinp` summary (real FEFF-format stdout when renderable,
/// else a plain fallback line) to stdout, unless `--quiet` is active. Called
/// both by `refeff rdinp` and at the start of `refeff run`, so a full run
/// always shows what `rdinp` parsed before its module stages begin.
fn print_rdinp_summary(report: &RdinpReport) -> Result<()> {
    let mode = current_output_mode();
    if mode.quiet {
        return Ok(());
    }
    // In `--json` mode stdout is reserved for the single JSON document
    // emitted at the end of the run, so the human rdinp summary — like
    // every other human-readable line this module prints — goes to stderr
    // instead of being suppressed outright.
    if mode.json {
        let mut stderr = std::io::stderr().lock();
        if let Some(summary) = &report.stdout {
            stderr.write_all(summary.as_bytes())?;
        } else {
            writeln!(
                stderr,
                "rdinp: parsed {} cards, {} atoms, {} potentials",
                report.cards, report.atoms, report.potentials
            )?;
        }
        return Ok(());
    }
    let mut stdout = std::io::stdout().lock();
    if let Some(summary) = &report.stdout {
        stdout.write_all(summary.as_bytes())?;
    } else {
        writeln!(
            stdout,
            "rdinp: parsed {} cards, {} atoms, {} potentials",
            report.cards, report.atoms, report.potentials
        )?;
    }
    Ok(())
}

/// Run the supported FEFF `pot` compatibility stage in the input directory.
pub fn run_pot(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Pot)?;
    run_pot_module(input)
}

/// Run the supported FEFF `atomic` compatibility stage in the input directory.
pub fn run_atomic(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Atomic)?;
    run_atomic_module(input)
}

/// Run the supported FEFF `band` compatibility stage in the input directory.
pub fn run_band(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Band)?;
    #[cfg(feature = "full")]
    {
        run_band_module(input)
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Band, "full")
    }
}

/// Run the supported FEFF `mdff` compatibility stage in the input directory.
pub fn run_mdff(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Mdff)?;
    #[cfg(feature = "full")]
    {
        run_mdff_module(input)
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Mdff, "full")
    }
}

/// Run the supported FEFF `wpot` compatibility stage in the input directory.
pub fn run_wpot(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Wpot)?;
    run_potential_output_module("wpot", input)
}

/// Run the supported FEFF `opcons` compatibility stage in the input directory.
pub fn run_opcons(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Opcons)?;
    #[cfg(feature = "full")]
    {
        let count = opcons::run_for_input(&input)?;
        print_module_line(format_args!(
            "opcons: wrote loss.dat with {count} row(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Opcons, "full")
    }
}

/// Run the supported FEFF `compton` compatibility stage in the input directory.
pub fn run_compton(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Compton)?;
    #[cfg(feature = "full")]
    {
        let count = compton::run_for_input(&input)?;
        print_module_line(format_args!(
            "compton: wrote cached output with {count} row(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Compton, "full")
    }
}

/// Run the supported FEFF `fullspectrum` compatibility stage in the input directory.
pub fn run_fullspectrum(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Fullspectrum)?;
    #[cfg(feature = "full")]
    {
        let count = fullspectrum::run_for_input(&input)?;
        print_module_line(format_args!(
            "fullspectrum: wrote optical constants with {count} row(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Fullspectrum, "full")
    }
}

/// Run the supported FEFF `crpa` compatibility stage in the input directory.
pub fn run_crpa(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Crpa)?;
    #[cfg(feature = "full")]
    {
        let count = crpa::run_for_input(&input)?;
        print_module_line(format_args!(
            "crpa: wrote crpa.dat with {count} result row(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Crpa, "full")
    }
}

/// Run the supported FEFF `screen` compatibility stage in the input directory.
pub fn run_screen(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Screen)?;
    let count = screen::run_for_input(&input)?;
    print_module_line(format_args!(
        "screen: wrote cached or source-backed output with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `ldos` compatibility stage in the input directory.
pub fn run_ldos(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Ldos)?;
    #[cfg(feature = "full")]
    {
        let count = ldos::run_for_input(&input)?;
        print_module_line(format_args!(
            "ldos: validated {count} cached or source-backed output file(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Ldos, "full")
    }
}

/// Run the supported FEFF `eels` compatibility stage in the input directory.
pub fn run_eels(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Eels)?;
    #[cfg(feature = "full")]
    {
        let count = eels::run_for_input(&input)?;
        print_module_line(format_args!(
            "eels: wrote eels.dat with {count} row(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Eels, "full")
    }
}

/// Run the supported FEFF `dmdw` compatibility stage in the input directory.
pub fn run_dmdw(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Dmdw)?;
    #[cfg(feature = "full")]
    {
        let count = dmdw::run_for_input(&input)?;
        print_module_line(format_args!(
            "dmdw: wrote dmdw.out with {count} section(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Dmdw, "full")
    }
}

/// Run the supported FEFF `path` compatibility stage in the input directory.
pub fn run_path(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Path)?;
    let count = paths::run_for_input(&input)?;
    print_module_line(format_args!(
        "path: wrote paths.dat with {count} path(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `genfmt` compatibility stage in the input directory.
pub fn run_genfmt(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Genfmt)?;
    let count = genfmt::run_for_input(&input)?;
    print_module_line(format_args!(
        "genfmt: validated {count} cached output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `ff2x` compatibility stage in the input directory.
pub fn run_ff2x(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Ff2x)?;
    let count = ff2x::run_for_input(&input)?;
    print_module_line(format_args!(
        "ff2x: validated {count} cached spectrum file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `xsph` compatibility stage in the input directory.
pub fn run_xsph(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Xsph)?;
    let count = xsph::run_for_input(&input)?;
    print_module_line(format_args!(
        "xsph: validated {count} cached or source-backed output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `fms` compatibility stage in the input directory.
pub fn run_fms(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Fms)?;
    #[cfg(feature = "full")]
    {
        let count = fms::run_fms_for_input(&input)?;
        print_module_line(format_args!(
            "fms: validated {count} cached Green's-function file(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Fms, "full")
    }
}

/// Run the supported FEFF `mkgtr` compatibility stage in the input directory.
///
pub fn run_mkgtr(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Mkgtr)?;
    #[cfg(feature = "full")]
    {
        let count = fms::run_mkgtr_for_input(&input)?;
        print_module_line(format_args!(
            "mkgtr: validated {count} cached Green's-function trace file(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Mkgtr, "full")
    }
}

/// Run the supported FEFF `rixs` compatibility stage in the input directory.
pub fn run_rixs(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Rixs)?;
    #[cfg(feature = "full")]
    {
        let count = rixs::run_for_input(&input)?;
        print_module_line(format_args!(
            "rixs: validated {count} cached or source-handoff file(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Rixs, "full")
    }
}

/// Run the supported FEFF `rhorrp` compatibility stage in the input directory.
pub fn run_rhorrp(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Rhorrp)?;
    #[cfg(feature = "full")]
    {
        let count = rhorrp::run_for_input(&input)?;
        print_module_line(format_args!(
            "rhorrp: processed {count} density output file(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "full"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Rhorrp, "full")
    }
}

/// Run the supported FEFF `sfconv` compatibility stage in the input directory.
pub fn run_sfconv(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::Sfconv)?;
    #[cfg(feature = "sfconv")]
    {
        sfconv::run_for_input(&input)?;
        print_module_line(format_args!(
            "sfconv: wrote logsfconv.dat beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "sfconv"))]
    {
        let _ = input;
        feature_disabled(ModuleName::Sfconv, "sfconv")
    }
}

/// Run the supported FEFF `self` (self-energy) compatibility stage in the input directory.
pub fn run_self_energy(input: PathBuf) -> Result<()> {
    ensure_module_available(ModuleName::SelfEnergy)?;
    #[cfg(feature = "sfconv")]
    {
        let count = sfconv::run_self_for_input(&input)?;
        print_module_line(format_args!(
            "self: validated {count} excitation pole(s) beside {}",
            input.display()
        ));
        Ok(())
    }
    #[cfg(not(feature = "sfconv"))]
    {
        let _ = input;
        feature_disabled(ModuleName::SelfEnergy, "sfconv")
    }
}

/// Run the complete file-backed FEFF pipeline and render its report.
pub fn run_feff(input: PathBuf, output: PathBuf) -> Result<()> {
    run_feff_to_dir(&input, &output)
}

pub fn run_feff_to_dir(input: &Path, output_dir: &Path) -> Result<()> {
    let report = execute_pipeline(input, output_dir, print_rdinp_summary)?;
    render_run_report(report)
}

/// Execute the file-backed FEFF pipeline without printing CLI progress.
///
/// This is the computational boundary used by the `refeff` facade crate.
/// Applications should normally prefer that crate's typed `Runner` API.
pub fn execute_feff(input: &Path, output_dir: &Path) -> Result<RunReport> {
    let _output_mode_guard = set_output_mode(OutputMode {
        verbose: false,
        quiet: true,
        json: false,
    });
    execute_pipeline(input, output_dir, |_| Ok(()))
}

fn execute_pipeline(
    input: &Path,
    output_dir: &Path,
    after_rdinp: impl FnOnce(&RdinpReport) -> Result<()>,
) -> Result<RunReport> {
    // Full runs may intentionally consume pre-existing handoffs from partial
    // inputs. The explicit `check` command validates standalone calculations.
    execution::start("rdinp")?;
    let rdinp_start = Instant::now();
    let report = execute_rdinp(input, output_dir)?;
    after_rdinp(&report)?;
    let mut rdinp_stage = SupportedModuleReport {
        name: "rdinp",
        count: report.cards,
        unit: "card(s)",
        status: StageStatus::Generated,
        duration_ms: elapsed_ms(rdinp_start),
    };
    print_stage_line(1, &mut rdinp_stage);
    #[allow(unused_mut)] // No scheduler is compiled without the EXAFS feature.
    let mut module_reports = Vec::new();
    #[cfg(feature = "full")]
    let module_result: Result<()> =
        cache::pipeline(output_dir, &mut module_reports, |module_reports| {
            rixs::prepare_two_edge_handoffs(input, output_dir)
                .context("failed to prepare the RIXS two-edge solver workflow")?;
            let mut pot_context = pot::PotRunContext::default();
            run_supported_cached_modules_into(output_dir, module_reports, &mut pot_context)?;
            run_remaining_required_modules(output_dir, module_reports, &mut pot_context)
        });
    #[cfg(all(feature = "exafs", not(feature = "full")))]
    let module_result = cache::pipeline(output_dir, &mut module_reports, |reports| {
        run_exafs_pipeline(output_dir, reports)
    });
    #[cfg(not(feature = "exafs"))]
    let module_result: Result<()> = Err(EngineError::FeatureDisabled {
        module: "run",
        feature: "exafs",
    }
    .into());
    match module_result {
        Ok(()) => Ok(RunReport {
            rdinp: report,
            stages: module_reports,
        }),
        Err(error) => Err(error.context(format!(
            "FEFF run failed after rdinp parsed {} cards, {} atoms, {} potentials from {}; {}",
            report.cards,
            report.atoms,
            report.potentials,
            input.display(),
            supported_module_summary(&module_reports)
        ))),
    }
}

/// Run a derived FEFF input used to prepare one side of the RIXS two-edge
/// handoff. Derived inputs omit `RIXS`, so this recursive pipeline cannot
/// schedule another RIXS edge workflow.
#[cfg(feature = "full")]
pub(crate) fn execute_rixs_edge_pipeline(input: &Path, output_dir: &Path) -> Result<()> {
    execute_pipeline(input, output_dir, |_| Ok(())).map(|_| ())
}

fn render_run_report(report: RunReport) -> Result<()> {
    let mode = current_output_mode();
    let summary_line = (!mode.quiet).then(|| format!("run: {}", report.summary()));
    if mode.json {
        if let Some(line) = &summary_line {
            eprintln!("{line}");
        }
        emit_json(&report)?;
    } else if let Some(line) = &summary_line {
        println!("{line}");
    }
    Ok(())
}

/// Machine-readable report for `refeff module <name>` (`--json`): a single
/// module ran once, so unlike [`RunReport`] there is no per-stage array.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleReport {
    pub module: String,
    pub input: PathBuf,
    pub output: PathBuf,
    pub count: usize,
    pub duration_ms: u64,
    pub rdinp: Option<RdinpReport>,
}

/// Milliseconds elapsed since `start`, saturating rather than panicking if
/// the (practically impossible) `u128` millisecond count overflows `u64`.
fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Execute a module using an explicit handoff/output directory, without printing.
pub fn execute_module(name: ModuleName, input: &Path, work_dir: &Path) -> Result<ModuleReport> {
    ensure_module_available(name)?;
    let start = Instant::now();
    let mut rdinp = None;
    let count = match name {
        ModuleName::Rdinp => {
            let report = execute_rdinp(input, work_dir)?;
            let count = report.cards;
            rdinp = Some(report);
            count
        }
        ModuleName::Pot => pot::run_in_dir(work_dir)?,
        ModuleName::Atomic => atomic::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Band => band::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Mdff => eelsmdff::run_in_dir(work_dir)?,
        ModuleName::Wpot => wpot::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Opcons => opcons::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Compton => compton::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Fullspectrum => fullspectrum::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Crpa => crpa::run_in_dir(work_dir)?,
        ModuleName::Screen => screen::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Ldos => ldos::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Eels => eels::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Dmdw => dmdw::run_in_dir(work_dir)?,
        ModuleName::Path => paths::run_in_dir(work_dir)?,
        ModuleName::Genfmt => genfmt::run_in_dir(work_dir)?,
        ModuleName::Ff2x => ff2x::run_in_dir(work_dir)?,
        ModuleName::Xsph => xsph::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Fms => fms::run_fms_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Mkgtr => fms::run_mkgtr_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Rixs => rixs::run_in_dir(work_dir)?,
        #[cfg(feature = "full")]
        ModuleName::Rhorrp => rhorrp::run_in_dir(work_dir)?,
        #[cfg(feature = "sfconv")]
        ModuleName::Sfconv => sfconv::run_in_dir(work_dir)?,
        #[cfg(feature = "sfconv")]
        ModuleName::SelfEnergy => sfconv::run_self_in_dir(work_dir)?,
        #[allow(unreachable_patterns)]
        _ => anyhow::bail!("module availability was not checked"),
    };
    Ok(ModuleReport {
        module: name.as_str().into(),
        input: input.to_path_buf(),
        output: work_dir.to_path_buf(),
        count,
        duration_ms: elapsed_ms(start),
        rdinp,
    })
}

/// Run a named stage beside its input; frontends can use `execute_module`
/// when the work directory differs from the input parent.
pub fn run_module(name: &str, input: PathBuf) -> Result<()> {
    run_named_module(ModuleName::parse(name)?, input)
}

/// Run one module and emit one report in the current frontend output mode.
pub fn run_named_module(name: ModuleName, input: PathBuf) -> Result<()> {
    let report = execute_module(name, &input, work_dir_for_input(&input))?;
    if current_output_mode().json {
        emit_json(&report)
    } else {
        print_module_line(format_args!(
            "{}: processed {} item(s) in {}",
            report.module,
            report.count,
            report.output.display()
        ));
        Ok(())
    }
}

fn run_potential_output_module(label: &str, input: PathBuf) -> Result<()> {
    let count = wpot::run_for_input(&input)?;
    print_module_line(format_args!(
        "{label}: wrote {count} potential output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

fn run_pot_module(input: PathBuf) -> Result<()> {
    let count = pot::run_for_input(&input)?;
    print_module_line(format_args!(
        "pot: validated or wrote {count} potential handoff file(s) beside {}",
        input.display()
    ));
    Ok(())
}

fn run_atomic_module(input: PathBuf) -> Result<()> {
    let count = atomic::run_for_input(&input)?;
    print_module_line(format_args!(
        "atomic: validated {count} cached or source-handoff file(s) beside {}",
        input.display()
    ));
    Ok(())
}

#[cfg(feature = "full")]
fn run_band_module(input: PathBuf) -> Result<()> {
    let count = band::run_for_input(&input)?;
    print_module_line(format_args!(
        "band: validated {count} cached or source-handoff file(s) beside {}",
        input.display()
    ));
    Ok(())
}

#[cfg(feature = "full")]
fn run_mdff_module(input: PathBuf) -> Result<()> {
    let count = eelsmdff::run_for_input(&input)?;
    print_module_line(format_args!(
        "mdff: wrote or validated {count} EELS-MDFF row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Whether a stage's final output was already present or had to be produced,
/// repaired, or completed from an upstream source handoff during this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StageStatus {
    /// A compatible artifact already existed and was reused.
    Cached,
    /// The stage generated or repaired its artifact.
    Generated,
}

impl StageStatus {
    #[cfg(feature = "full")]
    const fn from_cached(cached: bool) -> Self {
        if cached {
            Self::Cached
        } else {
            Self::Generated
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SupportedModuleReport {
    /// FEFF stage name.
    pub name: &'static str,
    /// Number of artifacts or rows handled by the stage.
    pub count: usize,
    /// Human-readable unit associated with `count`.
    pub unit: &'static str,
    /// Whether the stage reused or generated its result.
    pub status: StageStatus,
    /// Wall-clock execution time in milliseconds.
    pub duration_ms: u64,
}

/// Prints `[i/N] name: reused cached|generated count unit (X.Xs)` to stderr
/// for one pipeline stage, where `i` is `reports.len()` after `report` was
/// pushed. Suppressed by `-q/--quiet` and `--json` (the latter emits a
/// [`RunReport`]/[`ModuleReport`] document instead).
fn print_stage_line(index: usize, report: &mut SupportedModuleReport) {
    if let Ok(module) = ModuleName::parse(report.name) {
        report.name = module.as_str();
    }
    if let Some(status) = execution::take_action(report.name) {
        report.status = status;
    }
    execution::emit(execution::Event::StageCompleted(report.clone()));
    let mode = current_output_mode();
    if mode.quiet {
        return;
    }
    // Stage lines already go to stderr (see D2), so `--json` — which only
    // reserves *stdout* for the machine-readable report — never needs to
    // suppress them, just like it doesn't suppress the rdinp summary.
    let verb = match report.status {
        StageStatus::Cached => "reused cached",
        StageStatus::Generated => "generated",
    };
    // `-v/--verbose` trades the rounded-to-a-decisecond duration for the
    // exact millisecond count, useful when timing many fast, similarly
    // sized stages (e.g. per-potential handoffs) that a single decimal
    // digit of seconds would otherwise show as identical.
    let duration = if mode.verbose {
        format!("{}ms", report.duration_ms)
    } else {
        #[allow(clippy::cast_precision_loss)]
        let seconds = report.duration_ms as f64 / 1000.0;
        format!("{seconds:.1}s")
    };
    eprintln!(
        "[{index}] {}: {verb} {} {} ({duration})",
        report.name, report.count, report.unit
    );
}

/// Machine-readable report for `refeff run` (`--json`): the `rdinp` summary
/// plus one entry per pipeline stage that produced output, in run order.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    /// Parsed-input summary.
    pub rdinp: RdinpReport,
    /// Completed stage reports in execution order.
    pub stages: Vec<SupportedModuleReport>,
}

impl RunReport {
    /// Return the concise stage summary used by the CLI.
    pub fn summary(&self) -> String {
        supported_module_summary(&self.stages)
    }
}

pub(crate) fn work_dir_for_input(input: &Path) -> &Path {
    match input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => parent,
        None => Path::new("."),
    }
}

pub fn execute_rdinp(input: &Path, output_dir: &Path) -> Result<RdinpReport> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let error_sentinel = output_dir.join(".feff.error");
    std::fs::write(&error_sentinel, rdinp::rdinp_error_sentinel_string())
        .with_context(|| format!("failed to write {}", error_sentinel.display()))?;

    let parsed = FeffInput::parse_file(input)?;
    let document = match FeffDocument::from_input(&parsed) {
        Ok(document) => document,
        Err(error) => {
            if let Ok(content) = rdinp::rdinp_error_log_string(&parsed, &error) {
                let output_path = output_dir.join("log.dat");
                std::fs::write(&output_path, content)
                    .with_context(|| format!("failed to write {}", output_path.display()))?;
            }
            return Err(error.into());
        }
    };
    let outputs = rdinp::text_outputs(&document)?;
    let log_dat = rdinp::rdinp_log_dat_string(&document).ok();
    let stdout = rdinp::rdinp_stdout_string(&document).ok();
    for (name, content) in outputs {
        let output_path = output_dir.join(name.as_ref());
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    if let Some(content) = &log_dat {
        let output_path = output_dir.join("log.dat");
        std::fs::write(&output_path, content)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    match std::fs::remove_file(&error_sentinel) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to remove {}", error_sentinel.display()));
        }
    }

    Ok(RdinpReport {
        cards: parsed.cards().count(),
        atoms: document.atoms.len(),
        potentials: document.potentials.len(),
        stdout,
    })
}

#[cfg(all(test, feature = "full"))]
pub(crate) mod tests;
