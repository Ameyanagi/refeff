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

/// A production FEFF10 module supported by the computational engine.
///
/// The engine deliberately parses module names without Clap so embedding it
/// never brings command-line dependencies into the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleName {
    /// Input parsing (`feff10/src/RDINP`).
    Rdinp,
    /// Self-consistent muffin-tin potentials (`feff10/src/POT`).
    Pot,
    /// Free-atom potentials/wavefunctions (`feff10/src/ATOM`).
    Atomic,
    /// Band structure / KKR (`feff10/src/BAND`).
    Band,
    /// EELS mixed dynamic form factor (`feff10/src/EELSMDFF`).
    Mdff,
    /// Potential-file rendering, e.g. `potXX.dat` (part of `feff10/src/POT`).
    Wpot,
    /// Optical constants from a dielectric-function cache
    /// (`feff10/src/OPCONSAT`).
    Opcons,
    /// Compton profiles (`feff10/src/COMPTON`).
    Compton,
    /// Optical constants across the full spectral range
    /// (`feff10/src/FULLSPECTRUM`).
    Fullspectrum,
    /// Constrained-RPA Hubbard parameters (`feff10/src/CRPA`).
    Crpa,
    /// Core-hole screening / Hubbard-U response (`feff10/src/SCREEN`).
    Screen,
    /// Local density of states (`feff10/src/LDOS`).
    Ldos,
    /// Electron energy-loss spectroscopy (`feff10/src/EELS`).
    Eels,
    /// Dynamical-matrix Debye-Waller factors (`feff10/src/DMDW`).
    Dmdw,
    /// Scattering path finder (`feff10/src/PATH`).
    Path,
    /// Path scattering-amplitude tables (`feff10/src/GENFMT`).
    Genfmt,
    /// Final spectrum assembly, EXAFS/XANES/DANES/FPRIME
    /// (`feff10/src/FF2X`).
    Ff2x,
    /// Phase shifts and cross sections (`feff10/src/XSPH`).
    Xsph,
    /// Full multiple scattering / Green's function (`feff10/src/FMS`).
    Fms,
    /// Green's-function trace projection (`feff10/src/MKGTR`).
    Mkgtr,
    /// Resonant inelastic X-ray scattering (`feff10/src/RIXS`).
    Rixs,
    /// Charge-density grid (`feff10/src/RHORRP`).
    Rhorrp,
    /// Many-body spectral-function convolution (`feff10/src/SFCONV`).
    Sfconv,
    /// Self-energy correction poles (`feff10/src/SELF`).
    // Named `SelfEnergy` (rather than `Self`) because `self` is a reserved
    // Rust keyword and cannot be used as an identifier.
    SelfEnergy,
}

impl ModuleName {
    /// Parse a canonical FEFF module name or a historical alias.
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "rdinp" => Ok(Self::Rdinp),
            "pot" => Ok(Self::Pot),
            "atomic" | "atom" => Ok(Self::Atomic),
            "band" => Ok(Self::Band),
            "mdff" | "eelsmdff" => Ok(Self::Mdff),
            "wpot" => Ok(Self::Wpot),
            "opcons" | "opconsat" => Ok(Self::Opcons),
            "compton" => Ok(Self::Compton),
            "fullspectrum" => Ok(Self::Fullspectrum),
            "crpa" => Ok(Self::Crpa),
            "screen" => Ok(Self::Screen),
            "ldos" => Ok(Self::Ldos),
            "eels" => Ok(Self::Eels),
            "dmdw" => Ok(Self::Dmdw),
            "path" | "paths" => Ok(Self::Path),
            "genfmt" => Ok(Self::Genfmt),
            "ff2x" => Ok(Self::Ff2x),
            "xsph" => Ok(Self::Xsph),
            "fms" => Ok(Self::Fms),
            "mkgtr" => Ok(Self::Mkgtr),
            "rixs" => Ok(Self::Rixs),
            "rhorrp" => Ok(Self::Rhorrp),
            "sfconv" => Ok(Self::Sfconv),
            "self" | "selfenergy" => Ok(Self::SelfEnergy),
            _ => Err(EngineError::UnsupportedModule {
                module: value.to_string(),
            }
            .into()),
        }
    }

    /// Return the canonical command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rdinp => "rdinp",
            Self::Pot => "pot",
            Self::Atomic => "atomic",
            Self::Band => "band",
            Self::Mdff => "mdff",
            Self::Wpot => "wpot",
            Self::Opcons => "opcons",
            Self::Compton => "compton",
            Self::Fullspectrum => "fullspectrum",
            Self::Crpa => "crpa",
            Self::Screen => "screen",
            Self::Ldos => "ldos",
            Self::Eels => "eels",
            Self::Dmdw => "dmdw",
            Self::Path => "path",
            Self::Genfmt => "genfmt",
            Self::Ff2x => "ff2x",
            Self::Xsph => "xsph",
            Self::Fms => "fms",
            Self::Mkgtr => "mkgtr",
            Self::Rixs => "rixs",
            Self::Rhorrp => "rhorrp",
            Self::Sfconv => "sfconv",
            Self::SelfEnergy => "self",
        }
    }

    /// Return the Cargo feature required to execute this module, if it is
    /// absent from the current build.
    pub const fn disabled_feature(self) -> Option<&'static str> {
        match self {
            Self::Rdinp
            | Self::Pot
            | Self::Atomic
            | Self::Wpot
            | Self::Screen
            | Self::Path
            | Self::Genfmt
            | Self::Ff2x
            | Self::Xsph => {
                if cfg!(feature = "exafs") {
                    None
                } else {
                    Some("exafs")
                }
            }
            Self::Sfconv | Self::SelfEnergy => {
                if cfg!(feature = "sfconv") {
                    None
                } else {
                    Some("sfconv")
                }
            }
            Self::Band
            | Self::Mdff
            | Self::Opcons
            | Self::Compton
            | Self::Fullspectrum
            | Self::Crpa
            | Self::Ldos
            | Self::Eels
            | Self::Dmdw
            | Self::Fms
            | Self::Mkgtr
            | Self::Rixs
            | Self::Rhorrp => {
                if cfg!(feature = "full") {
                    None
                } else {
                    Some("full")
                }
            }
        }
    }
}

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
/// [`crate::exit_code_for`]-equivalent handling in `bin/refeff.rs`, which
/// maps non-`Io` [`refeff_io::IoError`] variants to exit code 3).
fn check(input: &Path) -> Result<CheckReport> {
    let parsed = FeffInput::parse_file(input)?;
    let document = FeffDocument::from_input(&parsed)?;
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
    let report = execute_rdinp(input, output_dir)?;
    after_rdinp(&report)?;
    let mut module_reports = Vec::new();
    #[cfg(feature = "full")]
    let module_result: Result<()> = (|| {
        rixs::prepare_two_edge_handoffs(input, output_dir)
            .context("failed to prepare the RIXS two-edge solver workflow")?;
        let mut pot_context = pot::PotRunContext::default();
        run_supported_cached_modules_into(output_dir, &mut module_reports, &mut pot_context)?;
        run_remaining_required_modules(output_dir, &mut module_reports, &mut pot_context)
    })();
    #[cfg(all(feature = "exafs", not(feature = "full")))]
    let module_result = run_exafs_pipeline(output_dir, &mut module_reports);
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
struct ModuleReport {
    module: String,
    input: String,
    duration_ms: u64,
}

/// Milliseconds elapsed since `start`, saturating rather than panicking if
/// the (practically impossible) `u128` millisecond count overflows `u64`.
fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn module_name_str(name: ModuleName) -> String {
    name.as_str().to_string()
}

/// Dispatch a single FEFF10 module by name.
///
/// Emits a structured module report when JSON output is active; otherwise
/// each `run_<module>` function prints its own status line.
pub fn run_module(name: &str, input: PathBuf) -> Result<()> {
    run_named_module(ModuleName::parse(name)?, input)
}

/// Dispatch a single FEFF10 module selected by its typed engine name.
pub fn run_named_module(name: ModuleName, input: PathBuf) -> Result<()> {
    ensure_module_available(name)?;
    let start = Instant::now();
    dispatch_module(name, input.clone())?;
    if current_output_mode().json {
        emit_json(&ModuleReport {
            module: module_name_str(name),
            input: input.display().to_string(),
            duration_ms: elapsed_ms(start),
        })?;
    }
    Ok(())
}

fn dispatch_module(name: ModuleName, input: PathBuf) -> Result<()> {
    match name {
        ModuleName::Rdinp => run_rdinp(input, PathBuf::from(".")),
        ModuleName::Pot => run_pot(input),
        ModuleName::Atomic => run_atomic(input),
        ModuleName::Band => run_band(input),
        ModuleName::Mdff => run_mdff(input),
        ModuleName::Wpot => run_wpot(input),
        ModuleName::Opcons => run_opcons(input),
        ModuleName::Compton => run_compton(input),
        ModuleName::Fullspectrum => run_fullspectrum(input),
        ModuleName::Crpa => run_crpa(input),
        ModuleName::Screen => run_screen(input),
        ModuleName::Ldos => run_ldos(input),
        ModuleName::Eels => run_eels(input),
        ModuleName::Dmdw => run_dmdw(input),
        ModuleName::Path => run_path(input),
        ModuleName::Genfmt => run_genfmt(input),
        ModuleName::Ff2x => run_ff2x(input),
        ModuleName::Xsph => run_xsph(input),
        ModuleName::Fms => run_fms(input),
        ModuleName::Mkgtr => run_mkgtr(input),
        ModuleName::Rixs => run_rixs(input),
        ModuleName::Rhorrp => run_rhorrp(input),
        ModuleName::Sfconv => run_sfconv(input),
        ModuleName::SelfEnergy => run_self_energy(input),
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

/// Approximate total FEFF10 pipeline stage count, used only for the `[i/N]`
/// progress header. `N` is the number of distinct modules `refeff run` may
/// invoke; `i` (a running count of stages that actually produced output)
/// does not always reach `N`, since a real run skips whatever upstream
/// products/switches make unnecessary.
const PIPELINE_STAGE_COUNT: usize = 22;

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
fn print_stage_line(index: usize, report: &SupportedModuleReport) {
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
        "[{index}/{}] {}: {verb} {} {} ({duration})",
        PIPELINE_STAGE_COUNT, report.name, report.count, report.unit
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

#[cfg(all(test, feature = "full"))]
fn run_supported_cached_modules(work_dir: &Path) -> Result<Vec<SupportedModuleReport>> {
    let mut reports = Vec::new();
    let mut pot_context = pot::PotRunContext::default();
    run_supported_cached_modules_into(work_dir, &mut reports, &mut pot_context)?;
    Ok(reports)
}

#[cfg(feature = "full")]
fn run_supported_cached_modules_into(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
    pot_context: &mut pot::PotRunContext,
) -> Result<()> {
    let atomic_cached = atomic::has_cached_atomic_output(work_dir)?;
    let stage_start = Instant::now();
    let prepared_no_scf_available =
        !atomic_cached && matches!(pot_context.prepared_no_scf(work_dir), Ok(Some(_)));
    let atomic_source_handoff =
        prepared_no_scf_available || atomic::has_supported_atomic_source_handoff(work_dir)?;
    if atomic_cached || atomic_source_handoff {
        let prepared_no_scf = if prepared_no_scf_available {
            pot_context.prepared_no_scf(work_dir)?
        } else {
            None
        };
        let count = atomic::run_in_dir_with_prepared_no_scf(work_dir, prepared_no_scf)
            .context("failed to run supported atomic stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "atomic",
                count,
                unit: "file(s)",
                status: StageStatus::from_cached(atomic_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if atomic::has_supported_config_handoff(work_dir)? {
        let count = atomic::run_supported_config_handoff_in_dir(work_dir)
            .context("failed to run supported atomic config handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "atomic-config",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if rhorrp::has_supported_rhorrp_output(work_dir)? {
        let stage_start = Instant::now();
        let count = rhorrp::run_in_dir(work_dir).context("failed to run supported rhorrp stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rhorrp",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let pot_cached = pot::has_cached_pot_output_with_context(work_dir, pot_context)?;
    if pot_cached
        || pot::has_supported_pot_source_handoff_with_context(work_dir, pot_context)?
        || pot::has_supported_pot_generation_handoff_with_context(work_dir, pot_context)?
    {
        let stage_start = Instant::now();
        let count = pot::run_in_dir_with_context(work_dir, pot_context)
            .context("failed to run supported pot stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "pot",
                count,
                unit: "file(s)",
                status: StageStatus::from_cached(pot_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if let Some(count) = pot::run_supported_pot_scf_source_handoff_once_in_dir(work_dir)? {
        if count > 0 {
            // The SCF-source handoff already ran as part of evaluating the
            // `if let` above, so this timer (unlike every other branch's)
            // only covers the negligible bookkeeping below it, not the
            // handoff itself.
            let stage_start = Instant::now();
            let completed = pot::has_cached_pot_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "pot" } else { "pot-scf-source" },
                count,
                unit: if completed {
                    "file(s)"
                } else {
                    "source bundle(s)"
                },
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if pot::has_supported_pot_input_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = pot::run_supported_pot_input_handoff_in_dir(work_dir)
            .context("failed to run supported pot input handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "pot-input",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    // Active Hubbard starts with an ordinary spectrum phase because LDOS
    // has not created v_hubbard.bin yet.  Record that bootstrap boundary
    // before the first XSPH/FMS pass.  After LDOS's two internal passes the
    // normal spectrum must be refreshed from the newly active Hubbard
    // source, matching FEFF's POT -> LDOS -> XSPH -> FMS final ordering.
    let active_hubbard_spectrum_bootstrap_pending =
        ldos::active_hubbard_spectrum_bootstrap_pending(work_dir)?;

    // MPSE XSPH consumes the loss function as its excitation-pole source.
    // OPCONS may need the freshly generated POT Norman radii to determine
    // default number densities, so run it after POT but before SCREEN/XSPH.
    // Deferring OPCONS until the later optical-output block leaves XSPH with
    // only an energy-mesh handoff and then prevents FMS from obtaining
    // phase.bin during a fresh full run.
    if opcons::has_complete_table_inputs(work_dir)? {
        let stage_start = Instant::now();
        let count = opcons::run_in_dir(work_dir).context("failed to run supported opcons stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "opcons",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if screen::has_completed_screen_output(work_dir)? {
        let stage_start = Instant::now();
        let count = screen::run_in_dir(work_dir).context("failed to run supported screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if screen::has_recoverable_cached_screen_stage(work_dir)? {
        let stage_start = Instant::now();
        let count = screen::run_recoverable_cached_screen_stage_in_dir(work_dir)
            .context("failed to run recoverable cached screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if screen::has_supported_wscrn_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = screen::run_supported_wscrn_handoff_in_dir(work_dir)
            .context("failed to run supported screen wscrn handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen-wscrn",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if screen::has_supported_screen_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = screen::run_in_dir(work_dir).context("failed to run supported screen stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "screen",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if xsph::has_supported_xsph_output(work_dir)?
        || xsph::has_supported_tdlda_xsedge_output(work_dir)?
    {
        let stage_start = Instant::now();
        let count = xsph::run_in_dir(work_dir).context("failed to run supported xsph stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "xsph",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else {
        if xsph::has_supported_phase_handoff(work_dir)? {
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_handoff_in_dir(work_dir)
                .context("failed to run supported xsph phase handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-phase",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                if let Some(report) = reports.last() {
                    print_stage_line(reports.len(), report);
                }
            }
        }
        if xsph::has_supported_phase_text_handoff(work_dir)? {
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_text_handoff_in_dir(work_dir)
                .context("failed to run supported xsph phase text handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-phase-text",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                if let Some(report) = reports.last() {
                    print_stage_line(reports.len(), report);
                }
            }
        }
        if xsph::has_supported_phase_mesh_handoff(work_dir)? {
            let stage_start = Instant::now();
            let count = xsph::run_supported_phase_mesh_handoff_in_dir(work_dir)
                .context("failed to run supported xsph emesh handoff")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "xsph-emesh",
                    count,
                    unit: "file(s)",
                    status: StageStatus::Generated,
                    duration_ms: elapsed_ms(stage_start),
                });
                if let Some(report) = reports.last() {
                    print_stage_line(reports.len(), report);
                }
            }
        }
    }

    ldos::with_preserved_active_hubbard_ldos_magnetic_sources(work_dir, || {
        if fms::has_runnable_fms_solver(work_dir)? {
            let status = if fms::has_cached_fms_solver_output(work_dir)? {
                StageStatus::Cached
            } else {
                StageStatus::Generated
            };
            let stage_start = Instant::now();
            let count =
                fms::run_fms_in_dir(work_dir).context("failed to run supported fms stage")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "fms",
                    count,
                    unit: "file(s)",
                    status,
                    duration_ms: elapsed_ms(stage_start),
                });
                if let Some(report) = reports.last() {
                    print_stage_line(reports.len(), report);
                }
            }

            let status = if fms::has_cached_mkgtr_output(work_dir)? {
                StageStatus::Cached
            } else {
                StageStatus::Generated
            };
            let stage_start = Instant::now();
            let count =
                fms::run_mkgtr_in_dir(work_dir).context("failed to run supported mkgtr stage")?;
            if count > 0 {
                reports.push(SupportedModuleReport {
                    name: "mkgtr",
                    count,
                    unit: "file(s)",
                    status,
                    duration_ms: elapsed_ms(stage_start),
                });
                if let Some(report) = reports.last() {
                    print_stage_line(reports.len(), report);
                }
            }
        }
        Ok(())
    })?;

    if band::has_cached_band_output(work_dir)? {
        let stage_start = Instant::now();
        let count = band::run_in_dir(work_dir).context("failed to run supported band stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "band",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if band::has_supported_pre_solver_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = band::run_supported_pre_solver_handoff_in_dir(work_dir)
            .context("failed to run supported band pre-solver handoff")?;
        if count > 0 {
            let completed = band::has_cached_band_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "band" } else { "band-handoff" },
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if rixs::has_cached_rixs_output(work_dir)? {
        let stage_start = Instant::now();
        let count = rixs::run_in_dir(work_dir).context("failed to run supported rixs stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "rixs",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if rixs::has_supported_solver_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = rixs::run_supported_solver_handoff_in_dir(work_dir)
            .context("failed to run supported rixs solver handoff")?;
        if count > 0 {
            let completed = rixs::has_cached_rixs_output(work_dir)?;
            reports.push(SupportedModuleReport {
                name: if completed { "rixs" } else { "rixs-handoff" },
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if compton::has_supported_outputs(work_dir)? {
        let stage_start = Instant::now();
        let count =
            compton::run_in_dir(work_dir).context("failed to run supported compton stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "compton",
                count,
                unit: "row(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if fullspectrum::has_cached_optical_inputs(work_dir)? {
        let stage_start = Instant::now();
        let count = fullspectrum::run_in_dir(work_dir)
            .context("failed to run supported fullspectrum stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "fullspectrum",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let crpa_cached = crpa::has_cached_crpa_output(work_dir)?;
    if crpa_cached || crpa::has_supported_crpa_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = crpa::run_in_dir(work_dir).context("failed to run supported crpa stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "crpa",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(crpa_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if crpa::has_supported_wscrn_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = crpa::run_supported_wscrn_handoff_in_dir(work_dir)
            .context("failed to run supported crpa wscrn handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "crpa-wscrn",
                count,
                unit: "row(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if ldos::has_cached_ldos_output(work_dir)? {
        let stage_start = Instant::now();
        let count = ldos::run_in_dir(work_dir).context("failed to run supported ldos stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if ldos::has_recoverable_ldos_output(work_dir)? {
        let stage_start = Instant::now();
        let count =
            ldos::run_in_dir(work_dir).context("failed to run supported recoverable ldos stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if ldos::has_supported_source_output_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = ldos::run_in_dir(work_dir)
            .context("failed to run supported ldos source-output handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    } else if ldos::has_supported_kmesh_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = ldos::run_supported_kmesh_handoff_in_dir(work_dir)
            .context("failed to run supported ldos kmesh handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ldos-kmesh",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if active_hubbard_spectrum_bootstrap_pending && work_dir.join("v_hubbard.bin").is_file() {
        let stage_start = Instant::now();
        let count = xsph::run_in_dir(work_dir)
            .context("failed to refresh active Hubbard xsph spectrum after ldos")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "xsph",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }

        ldos::with_preserved_active_hubbard_ldos_magnetic_sources(work_dir, || {
            if fms::has_runnable_fms_solver(work_dir)? {
                let stage_start = Instant::now();
                let count = fms::run_fms_in_dir(work_dir)
                    .context("failed to refresh active Hubbard fms spectrum after ldos")?;
                if count > 0 {
                    reports.push(SupportedModuleReport {
                        name: "fms",
                        count,
                        unit: "file(s)",
                        status: StageStatus::Generated,
                        duration_ms: elapsed_ms(stage_start),
                    });
                    if let Some(report) = reports.last() {
                        print_stage_line(reports.len(), report);
                    }
                }

                let stage_start = Instant::now();
                let count = fms::run_mkgtr_in_dir(work_dir)
                    .context("failed to refresh active Hubbard mkgtr spectrum after ldos")?;
                if count > 0 {
                    reports.push(SupportedModuleReport {
                        name: "mkgtr",
                        count,
                        unit: "file(s)",
                        status: StageStatus::Generated,
                        duration_ms: elapsed_ms(stage_start),
                    });
                    if let Some(report) = reports.last() {
                        print_stage_line(reports.len(), report);
                    }
                }
            }
            Ok(())
        })?;
    }

    if band::kmesh::has_supported_kmesh_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = band::kmesh::run_supported_kmesh_handoff_in_dir(work_dir)
            .context("failed to run supported kmesh handoff")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "kmesh",
                count,
                unit: "file(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let eels_cached = eels::has_completed_eels_output(work_dir)?;
    if eels::has_cached_eels_output(work_dir)? {
        let stage_start = Instant::now();
        let count = eels::run_in_dir(work_dir).context("failed to run supported eels stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eels",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(eels_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let eelsmdff_cached = eelsmdff::has_completed_mdff_output(work_dir)?;
    if eelsmdff::has_cached_mdff_output(work_dir)? {
        let stage_start = Instant::now();
        let count =
            eelsmdff::run_in_dir(work_dir).context("failed to run supported eelsmdff stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "eelsmdff",
                count,
                unit: "row(s)",
                status: StageStatus::from_cached(eelsmdff_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let dmdw_cached = dmdw::has_cached_dmdw_output(work_dir)?;
    if dmdw_cached || dmdw::has_supported_dmdw_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = dmdw::run_in_dir(work_dir).context("failed to run supported dmdw stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "dmdw",
                count,
                unit: "section(s)",
                status: StageStatus::from_cached(dmdw_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if paths::has_cached_paths_output(work_dir)? {
        let stage_start = Instant::now();
        let count = paths::run_in_dir(work_dir).context("failed to run supported path stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "path",
                count,
                unit: "path(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if genfmt::has_cached_genfmt_output(work_dir)? {
        let stage_start = Instant::now();
        let count = genfmt::run_in_dir(work_dir).context("failed to run supported genfmt stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "genfmt",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if ff2x::has_cached_ff2x_output(work_dir)? {
        let stage_start = Instant::now();
        let count = ff2x::run_in_dir(work_dir).context("failed to run supported ff2x stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "ff2x",
                count,
                unit: "file(s)",
                status: StageStatus::Cached,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    let self_cached = sfconv::has_cached_self_output(work_dir)?;
    if self_cached || sfconv::has_supported_self_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count =
            sfconv::run_self_in_dir(work_dir).context("failed to run supported self stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "self",
                count,
                unit: "pole(s)",
                status: StageStatus::from_cached(self_cached),
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    if sfconv::has_supported_sfconv_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = sfconv::run_in_dir(work_dir).context("failed to run supported sfconv stage")?;
        if count > 0 {
            reports.push(SupportedModuleReport {
                name: "sfconv",
                count,
                unit: "target(s)",
                status: StageStatus::Generated,
                duration_ms: elapsed_ms(stage_start),
            });
            if let Some(report) = reports.last() {
                print_stage_line(reports.len(), report);
            }
        }
    }

    Ok(())
}

fn supported_module_summary(reports: &[SupportedModuleReport]) -> String {
    if reports.is_empty() {
        return "no supported cached stages were run".to_string();
    }

    let details = reports
        .iter()
        .map(|report| format!("{}={} {}", report.name, report.count, report.unit))
        .collect::<Vec<_>>()
        .join(", ");
    format!("supported cached stages run: {details}")
}

#[cfg(feature = "full")]
fn run_remaining_required_modules(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
    pot_context: &mut pot::PotRunContext,
) -> Result<()> {
    if !atomic::has_cached_atomic_output(work_dir)? {
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "pot" && report.count > 0)
        && !pot::has_cached_pot_output_with_context(work_dir, pot_context)?
    {
        run_required_module(reports, "pot", "file(s)", || {
            pot::run_in_dir_with_context(work_dir, pot_context)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "xsph" && report.count > 0)
        && !xsph::has_supported_xsph_output(work_dir)?
        && !xsph::has_supported_tdlda_xsedge_output(work_dir)?
    {
        run_required_module(reports, "xsph", "file(s)", || {
            xsph::run_required_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "fms" && report.count > 0)
        && !fms::has_cached_fms_solver_output(work_dir)?
    {
        run_required_module(reports, "fms", "file(s)", || fms::run_fms_in_dir(work_dir))?;
    }
    if fms::has_cached_fms_solver_output(work_dir)?
        && !reports
            .iter()
            .any(|report| report.name == "mkgtr" && report.count > 0)
    {
        run_required_module(reports, "mkgtr", "file(s)", || {
            fms::run_mkgtr_in_dir(work_dir)
        })?;
    }
    if !band::has_cached_band_output(work_dir)? {
        run_required_module(reports, "band", "file(s)", || band::run_in_dir(work_dir))?;
    }
    if !rixs::has_cached_rixs_output(work_dir)? {
        run_required_module(reports, "rixs", "file(s)", || rixs::run_in_dir(work_dir))?;
    }
    if !rhorrp::has_supported_rhorrp_output(work_dir)? {
        run_required_module(reports, "rhorrp", "file(s)", || {
            rhorrp::run_in_dir(work_dir)
        })?;
    }
    if !opcons::has_complete_table_inputs(work_dir)? {
        run_required_module(reports, "opcons", "row(s)", || opcons::run_in_dir(work_dir))?;
    }
    if !compton::has_supported_outputs(work_dir)? {
        run_required_module(reports, "compton", "row(s)", || {
            compton::run_in_dir(work_dir)
        })?;
    }
    if !fullspectrum::has_cached_optical_inputs(work_dir)? {
        run_required_module(reports, "fullspectrum", "row(s)", || {
            fullspectrum::run_in_dir(work_dir)
        })?;
    }
    if !(crpa::has_cached_crpa_output(work_dir)?
        || crpa::has_supported_crpa_source_handoff(work_dir)?)
    {
        run_required_module(reports, "crpa", "row(s)", || crpa::run_in_dir(work_dir))?;
    }
    if !screen::has_completed_screen_output(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    }
    if !ldos::has_cached_ldos_output(work_dir)? {
        run_required_module(reports, "ldos", "file(s)", || ldos::run_in_dir(work_dir))?;
    }
    if !dmdw::has_cached_dmdw_output(work_dir)? {
        run_required_module(reports, "dmdw", "section(s)", || dmdw::run_in_dir(work_dir))?;
    }
    if !paths::has_cached_paths_output(work_dir)? {
        run_required_module(reports, "path", "path(s)", || paths::run_in_dir(work_dir))?;
    }
    if !genfmt::has_cached_genfmt_output(work_dir)? {
        run_required_module(reports, "genfmt", "file(s)", || {
            genfmt::run_in_dir(work_dir)
        })?;
    }
    if !ff2x::has_cached_ff2x_output(work_dir)? {
        run_required_module(reports, "ff2x", "file(s)", || ff2x::run_in_dir(work_dir))?;
    }
    // EELS consumes the polarization-specific xmu/opcons spectra assembled by
    // the path -> GENFMT -> FF2X producer chain. Keep it after those required
    // stages so a cache-free ELNES/EXELFS run can complete in one scheduler
    // pass instead of stopping before its source spectra exist.
    if !eels::has_completed_eels_output(work_dir)? {
        run_required_module(reports, "eels", "row(s)", || eels::run_in_dir(work_dir))?;
    }
    if !eelsmdff::has_completed_mdff_output(work_dir)? {
        run_required_module(reports, "eelsmdff", "row(s)", || {
            eelsmdff::run_in_dir(work_dir)
        })?;
    }
    if !sfconv::has_cached_self_output(work_dir)? {
        run_required_module(reports, "self", "pole(s)", || {
            sfconv::run_self_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "sfconv" && report.count > 0)
    {
        run_required_module(reports, "sfconv", "target(s)", || {
            sfconv::run_in_dir(work_dir)
        })?;
    }
    Ok(())
}

#[cfg(all(feature = "exafs", not(feature = "full")))]
fn run_exafs_pipeline(work_dir: &Path, reports: &mut Vec<SupportedModuleReport>) -> Result<()> {
    let mut pot_context = pot::PotRunContext::default();

    let atomic_cached = atomic::has_cached_atomic_output(work_dir)?;
    let prepared_no_scf_available =
        !atomic_cached && matches!(pot_context.prepared_no_scf(work_dir), Ok(Some(_)));
    let atomic_source_handoff =
        prepared_no_scf_available || atomic::has_supported_atomic_source_handoff(work_dir)?;
    if atomic_cached || atomic_source_handoff {
        let prepared_no_scf = if prepared_no_scf_available {
            pot_context.prepared_no_scf(work_dir)?
        } else {
            None
        };
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir_with_prepared_no_scf(work_dir, prepared_no_scf)
        })?;
    } else {
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir(work_dir)
        })?;
    }

    let pot_cached = pot::has_cached_pot_output_with_context(work_dir, &mut pot_context)?;
    let _pot_satisfiable = pot_cached
        || pot::has_supported_pot_source_handoff_with_context(work_dir, &mut pot_context)?
        || pot::has_supported_pot_generation_handoff_with_context(work_dir, &mut pot_context)?;
    run_required_module(reports, "pot", "file(s)", || {
        pot::run_in_dir_with_context(work_dir, &mut pot_context)
    })?;

    if screen::has_completed_screen_output(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    } else if screen::has_recoverable_cached_screen_stage(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || {
            screen::run_recoverable_cached_screen_stage_in_dir(work_dir)
        })?;
    } else if screen::has_supported_wscrn_handoff(work_dir)? {
        run_required_module(reports, "screen-wscrn", "row(s)", || {
            screen::run_supported_wscrn_handoff_in_dir(work_dir)
        })?;
    } else if screen::has_supported_screen_source_handoff(work_dir)? {
        run_required_module(reports, "screen", "row(s)", || screen::run_in_dir(work_dir))?;
    }

    // `has_supported_xsph_output` means the stage can run from either caches
    // or source handoffs, not that its outputs already exist. Always invoke
    // the stage so a fresh EXAFS workspace materializes phase.bin before
    // PATH.
    let mut xsph_context = xsph::XsphRunContext::default();
    run_required_module(reports, "xsph", "file(s)", || {
        let xsph_satisfiable =
            xsph::has_supported_xsph_output_with_context(work_dir, &mut xsph_context)?;
        if xsph_satisfiable {
            xsph::run_in_dir_with_context(work_dir, &mut xsph_context)
        } else if xsph::has_supported_tdlda_xsedge_output(work_dir)? {
            xsph::run_in_dir(work_dir)
        } else {
            xsph::run_required_in_dir(work_dir)
        }
    })?;
    // These compatibility predicates report whether a stage is satisfiable
    // from either an existing cache or source handoffs. Invoke each stage so
    // fresh handoffs are actually serialized for its downstream consumer.
    run_required_module(reports, "path", "path(s)", || paths::run_in_dir(work_dir))?;
    run_required_module(reports, "genfmt", "file(s)", || {
        genfmt::run_in_dir(work_dir)
    })?;
    run_required_module(reports, "ff2x", "file(s)", || ff2x::run_in_dir(work_dir))?;

    #[cfg(feature = "sfconv")]
    if sfconv::has_supported_sfconv_source_handoff(work_dir)? {
        run_required_module(reports, "sfconv", "target(s)", || {
            sfconv::run_in_dir(work_dir)
        })?;
    }

    Ok(())
}

/// Runs a single module for which no cached/handoff state satisfied it (the
/// `run_remaining_required_modules` fallback), pushing a `Generated`
/// [`SupportedModuleReport`] into the shared `reports` vec — the same vec
/// [`run_supported_cached_modules_into`] populates — so the final summary
/// and `--json` [`RunReport`] cover every stage the run touched, not just
/// the ones satisfied from cache.
fn run_required_module(
    reports: &mut Vec<SupportedModuleReport>,
    name: &'static str,
    unit: &'static str,
    run: impl FnOnce() -> Result<usize>,
) -> Result<()> {
    let stage_start = Instant::now();
    let count = run().with_context(|| format!("failed to run FEFF {name} stage"))?;
    if count > 0 {
        reports.push(SupportedModuleReport {
            name,
            count,
            unit,
            status: StageStatus::Generated,
            duration_ms: elapsed_ms(stage_start),
        });
        if let Some(report) = reports.last() {
            print_stage_line(reports.len(), report);
        }
    }
    Ok(())
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

fn execute_rdinp(input: &Path, output_dir: &Path) -> Result<RdinpReport> {
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
