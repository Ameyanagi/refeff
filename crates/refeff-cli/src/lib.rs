#![forbid(unsafe_code)]
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
mod band;
mod compton;
mod crpa;
mod dmdw;
mod dym2feffinp;
mod eels;
mod eelsmdff;
mod ff2x;
mod fms;
mod fullspectrum;
mod genfmt;
mod ldos;
mod opcons;
mod paths;
mod pot;
mod rhorrp;
mod rixs;
mod screen;
mod sfconv;
mod wpot;
mod xsph;

use std::cell::Cell;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use refeff_io::{FeffDocument, FeffInput, rdinp};
use serde::Serialize;

/// Typical workflow:
///
/// 1. `refeff check -i feff.inp` — validate an input with no side effects.
/// 2. `refeff run -i feff.inp -o out/` — run the full RDINP-through-FF2X
///    pipeline, writing every generated FEFF-format file under `out/`.
/// 3. `refeff module xsph -i feff.inp` — re-run (or inspect) a single stage
///    in place, next to the input, once cached upstream handoff files exist.
///
/// File placement: `--input`/`--output` are resolved relative to the
/// current directory unless `-C/--dir <DIR>` is given, in which case both
/// (and `module`'s implicit input-parent working directory) are resolved
/// relative to `DIR` instead — the same rule `git -C` uses. This is what
/// makes `refeff module <name> -i feff.inp` and `refeff run -o out/`
/// agree on where files land when combined with a shared `-C`.
///
/// Exit codes: 0 success, 1 internal/IO error, 2 command-line usage error
/// (clap), and 3 invalid `feff.inp`/input. See `refeff --help` (long form)
/// for details.
#[derive(Debug, Parser)]
#[command(
    name = "refeff",
    version,
    about = "Pure-Rust FEFF10 compatibility port",
    after_help = "Typical workflow: `refeff run -i feff.inp -o out/`, then \
                  inspect a single stage with `refeff module <name> -i feff.inp`. \
                  Run `refeff module --help` for the list of supported module names.",
    after_long_help = "Typical workflow:\n  \
                        1. refeff check -i feff.inp        # validate, no side effects\n  \
                        2. refeff run -i feff.inp -o out/  # full RDINP..FF2X pipeline\n  \
                        3. refeff module xsph -i feff.inp  # re-run/inspect one stage\n\n\
                        File placement:\n  \
                        --input/--output resolve relative to the current directory.\n  \
                        -C/--dir DIR (git-style) resolves them relative to DIR instead,\n  \
                        and is also what lets `module` operate in a directory other than\n  \
                        --input's own parent.\n\n\
                        Exit codes:\n  \
                        0  success\n  \
                        1  internal or I/O error\n  \
                        2  command-line usage error (clap)\n  \
                        3  invalid feff.inp / input\n"
)]
pub struct Cli {
    /// The subcommand to run; defaults to `run -i feff.inp -o .` when omitted.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Print one line per pipeline stage as it starts/finishes, plus extra
    /// timing detail, to stderr. Conflicts with `--quiet`.
    #[arg(short, long, global = true, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Suppress per-stage progress lines; only the final summary and errors
    /// are printed.
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Emit a single machine-readable JSON document on stdout instead of
    /// human-readable text; human progress/summary text still goes to
    /// stderr. See `refeff --help` (long form) for the per-stage shape.
    #[arg(long, global = true)]
    pub json: bool,

    /// Operate as if started in DIR (git-style): `--input`/`--output`
    /// defaults and relative paths, and `module`'s implicit
    /// input-parent working directory, all resolve relative to DIR
    /// instead of the current directory.
    #[arg(short = 'C', long = "dir", global = true, value_name = "DIR")]
    pub dir: Option<PathBuf>,

    /// Bound the number of worker threads `rayon`/`faer` use process-wide
    /// (falls back to the `REFEFF_THREADS` environment variable). `--threads
    /// 1` gives a fully serial, deterministic run, useful for golden-output
    /// validation and reproducible HPC batch jobs.
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
}

/// A `refeff` subcommand.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Validate `feff.inp` with zero filesystem writes: parses it and
    /// builds a `FeffDocument` exactly like `rdinp`/`run` do, but never
    /// creates `.feff.error`, `log.dat`, or any handoff file. Prints
    /// `OK: N cards, M atoms, K potentials, edge=..., modules enabled: ...`
    /// on success; a card-located problem is printed and the process exits
    /// with code 3. `inspect` is a historical alias for this command.
    #[command(alias = "inspect")]
    Check {
        /// Path to the `feff.inp`-format input file to validate.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    /// Run only the FEFF10-compatible `rdinp` input-parsing stage, writing
    /// its text handoff files (`atoms.dat`, `geom.dat`, module `.inp`
    /// files, ...) to `--output`.
    Rdinp {
        /// Path to the `feff.inp`-format input file to parse.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Directory to write `rdinp`'s generated handoff files into.
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Run the full supported FEFF10 pipeline (RDINP through FF2X),
    /// writing every generated FEFF-format file under `--output`.
    Run {
        /// Path to the `feff.inp`-format input file to run.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Directory to write all generated FEFF-format output files into.
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Run a single FEFF10 module by name (e.g. to regenerate or inspect
    /// one stage's cached handoff files) in `--input`'s parent directory,
    /// or in `-C/--dir` when given (see `refeff --help`, long form).
    #[command(after_help = "Supported names (see `refeff module --help` for the \
                             possible-values list clap derives from these):\n  \
                             rdinp, pot, atomic (alias: atom), band, mdff (alias: eelsmdff), \
                             wpot, opcons (alias: opconsat), compton, fullspectrum, crpa, \
                             screen, ldos, eels, dmdw, path (alias: paths), genfmt, ff2x, \
                             xsph, fms, mkgtr, rixs, rhorrp, sfconv, \
                             self (alias: selfenergy)")]
    Module {
        /// Name of the FEFF10 module to run (see `after_help` for aliases).
        #[arg(value_enum)]
        name: ModuleName,
        /// Path to the `feff.inp`-format input file; the module's cached
        /// handoff files are read from and written to its parent directory
        /// (override with the global `-C/--dir`).
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    /// Generate a shell completion script for `refeff` on stdout.
    Completions {
        /// Shell to generate the completion script for.
        shell: clap_complete::Shell,
    },
}

/// The production FEFF10 module names `refeff module <name>` accepts, plus
/// their FEFF10-historical aliases (`#[value(alias = ...)]`). Developer
/// prototypes and conversion utilities are intentionally outside this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModuleName {
    /// Input parsing (`feff10/src/RDINP`).
    Rdinp,
    /// Self-consistent muffin-tin potentials (`feff10/src/POT`).
    Pot,
    /// Free-atom potentials/wavefunctions (`feff10/src/ATOM`).
    #[value(alias = "atom")]
    Atomic,
    /// Band structure / KKR (`feff10/src/BAND`).
    Band,
    /// EELS mixed dynamic form factor (`feff10/src/EELSMDFF`).
    #[value(alias = "eelsmdff")]
    Mdff,
    /// Potential-file rendering, e.g. `potXX.dat` (part of `feff10/src/POT`).
    Wpot,
    /// Optical constants from a dielectric-function cache
    /// (`feff10/src/OPCONSAT`).
    #[value(alias = "opconsat")]
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
    #[value(alias = "paths")]
    Path,
    /// Path scattering-amplitude tables (`feff10/src/GENFMT`).
    Genfmt,
    /// Final spectrum assembly, EXAFS/XANES/DANES/FPRIME
    /// (`feff10/src/FF2X`).
    #[value(name = "ff2x")]
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
    #[value(name = "self", alias = "selfenergy")]
    SelfEnergy,
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

/// Global `-v/-q/--json` state, threaded through to `run_*`/module helpers
/// via a thread-local rather than an extra parameter, since those functions
/// are called directly (without a [`Cli`]) from tests and other library
/// entry points and their signatures must stay stable.
#[derive(Debug, Clone, Copy, Default)]
struct OutputMode {
    verbose: bool,
    quiet: bool,
    json: bool,
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

/// Restores the previous thread-local [`OutputMode`] on drop, so a
/// `run_cli` call never leaks its `-v/-q/--json` flags into whatever ran
/// beforehand on the same thread (relevant for the `#[test]` thread pool,
/// where threads are reused across many independent tests).
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

/// Resolves a CLI path argument against `-C/--dir`, git-`-C`-style: a
/// relative `path` is joined onto `dir`; an absolute `path`, or no `dir` at
/// all, passes through unchanged. Applied uniformly to every subcommand's
/// `--input`/`--output` so `-C` is what unifies `run`'s explicit `--output`
/// with `module`'s implicit input-parent working directory (see
/// [`work_dir_for_input`]).
fn resolve_path(dir: Option<&Path>, path: PathBuf) -> PathBuf {
    match dir {
        Some(dir) if path.is_relative() => dir.join(path),
        _ => path,
    }
}

/// Resolves `--threads`/`REFEFF_THREADS`, builds the global `rayon` thread
/// pool once, and mirrors the bound into `refeff_linalg::set_parallelism` so
/// `faer`'s solvers respect it too. `--threads 1` therefore gives a fully
/// serial, deterministic run. A no-op when neither `--threads` nor
/// `REFEFF_THREADS` is set, leaving `rayon`/`faer` at their own defaults.
/// `rayon`'s global pool can only be built once per process; a second call
/// (e.g. a second `run_cli` in the same process, as in-process tests can
/// trigger) warns and continues rather than failing the run.
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

/// Dispatch a parsed `refeff` command.
pub fn run_cli(cli: Cli) -> Result<()> {
    configure_threads(cli.threads);
    let _output_mode_guard = set_output_mode(OutputMode {
        verbose: cli.verbose,
        quiet: cli.quiet,
        json: cli.json,
    });
    let dir = cli.dir.as_deref();
    let command = match cli.command {
        Some(command) => command,
        None => Command::Run {
            input: PathBuf::from("feff.inp"),
            output: PathBuf::from("."),
        },
    };
    match command {
        Command::Check { input } => run_check(resolve_path(dir, input)),
        Command::Rdinp { input, output } => {
            run_rdinp(resolve_path(dir, input), resolve_path(dir, output))
        }
        Command::Run { input, output } => {
            run_feff(resolve_path(dir, input), resolve_path(dir, output))
        }
        Command::Module { name, input } => run_module(name, resolve_path(dir, input)),
        Command::Completions { shell } => print_completions(shell),
    }
}

/// Write a shell completion script for `refeff` to stdout.
fn print_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut command = <Cli as clap::CommandFactory>::command();
    let name = command.get_name().to_string();
    clap_complete::generate(shell, &mut command, name, &mut std::io::stdout());
    Ok(())
}

/// Converts a caller-supplied module name into a [`ModuleName`].
///
/// Implemented for `ModuleName` itself (the CLI entry point, once clap has
/// already validated it) and for `&str` (used by string-based call sites),
/// so `run_module` accepts either without duplicating the dispatch match.
trait IntoModuleName {
    fn into_module_name(self) -> Result<ModuleName>;
}

impl IntoModuleName for ModuleName {
    fn into_module_name(self) -> Result<ModuleName> {
        Ok(self)
    }
}

impl IntoModuleName for &str {
    fn into_module_name(self) -> Result<ModuleName> {
        <ModuleName as ValueEnum>::from_str(self, true).map_err(|message| anyhow::anyhow!(message))
    }
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
fn run_check(input: PathBuf) -> Result<()> {
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
    run_pot_module(input)
}

/// Run the supported FEFF `atomic` compatibility stage in the input directory.
pub fn run_atomic(input: PathBuf) -> Result<()> {
    run_atomic_module(input)
}

/// Run the supported FEFF `band` compatibility stage in the input directory.
pub fn run_band(input: PathBuf) -> Result<()> {
    run_band_module(input)
}

/// Run the supported FEFF `mdff` compatibility stage in the input directory.
pub fn run_mdff(input: PathBuf) -> Result<()> {
    run_mdff_module(input)
}

/// Run the supported FEFF `wpot` compatibility stage in the input directory.
pub fn run_wpot(input: PathBuf) -> Result<()> {
    run_potential_output_module("wpot", input)
}

/// Run the supported FEFF `opcons` compatibility stage in the input directory.
pub fn run_opcons(input: PathBuf) -> Result<()> {
    let count = opcons::run_for_input(&input)?;
    print_module_line(format_args!(
        "opcons: wrote loss.dat with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `compton` compatibility stage in the input directory.
pub fn run_compton(input: PathBuf) -> Result<()> {
    let count = compton::run_for_input(&input)?;
    print_module_line(format_args!(
        "compton: wrote cached output with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `fullspectrum` compatibility stage in the input directory.
pub fn run_fullspectrum(input: PathBuf) -> Result<()> {
    let count = fullspectrum::run_for_input(&input)?;
    print_module_line(format_args!(
        "fullspectrum: wrote optical constants with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `crpa` compatibility stage in the input directory.
pub fn run_crpa(input: PathBuf) -> Result<()> {
    let count = crpa::run_for_input(&input)?;
    print_module_line(format_args!(
        "crpa: wrote crpa.dat with {count} result row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `screen` compatibility stage in the input directory.
pub fn run_screen(input: PathBuf) -> Result<()> {
    let count = screen::run_for_input(&input)?;
    print_module_line(format_args!(
        "screen: wrote cached or source-backed output with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `ldos` compatibility stage in the input directory.
pub fn run_ldos(input: PathBuf) -> Result<()> {
    let count = ldos::run_for_input(&input)?;
    print_module_line(format_args!(
        "ldos: validated {count} cached or source-backed output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `eels` compatibility stage in the input directory.
pub fn run_eels(input: PathBuf) -> Result<()> {
    let count = eels::run_for_input(&input)?;
    print_module_line(format_args!(
        "eels: wrote eels.dat with {count} row(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `dmdw` compatibility stage in the input directory.
pub fn run_dmdw(input: PathBuf) -> Result<()> {
    let count = dmdw::run_for_input(&input)?;
    print_module_line(format_args!(
        "dmdw: wrote dmdw.out with {count} section(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Convert a FEFF `.dym` file into matching `feff.inp` and reordered `.dym` files.
///
/// `center_atom` is one-based to preserve the production
/// `dym2feffinp --c iAbs` command-line contract.
pub fn run_dym2feffinp(
    dym_file: PathBuf,
    center_atom: usize,
    feff_output: PathBuf,
    dym_output: PathBuf,
    spectrum: refeff_io::DymSpectrum,
    write_header: bool,
) -> Result<()> {
    dym2feffinp::run(
        &dym_file,
        center_atom,
        &feff_output,
        &dym_output,
        spectrum,
        write_header,
    )
}

/// Parse and run the standalone FEFF10-compatible `dym2feffinp` CLI.
pub fn dym2feffinp_main() -> Result<()> {
    dym2feffinp::main()
}

/// Run the supported FEFF `path` compatibility stage in the input directory.
pub fn run_path(input: PathBuf) -> Result<()> {
    let count = paths::run_for_input(&input)?;
    print_module_line(format_args!(
        "path: wrote paths.dat with {count} path(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `genfmt` compatibility stage in the input directory.
pub fn run_genfmt(input: PathBuf) -> Result<()> {
    let count = genfmt::run_for_input(&input)?;
    print_module_line(format_args!(
        "genfmt: validated {count} cached output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `ff2x` compatibility stage in the input directory.
pub fn run_ff2x(input: PathBuf) -> Result<()> {
    let count = ff2x::run_for_input(&input)?;
    print_module_line(format_args!(
        "ff2x: validated {count} cached spectrum file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `xsph` compatibility stage in the input directory.
pub fn run_xsph(input: PathBuf) -> Result<()> {
    let count = xsph::run_for_input(&input)?;
    print_module_line(format_args!(
        "xsph: validated {count} cached or source-backed output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `fms` compatibility stage in the input directory.
pub fn run_fms(input: PathBuf) -> Result<()> {
    let count = fms::run_fms_for_input(&input)?;
    print_module_line(format_args!(
        "fms: validated {count} cached Green's-function file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `mkgtr` compatibility stage in the input directory.
///
pub fn run_mkgtr(input: PathBuf) -> Result<()> {
    let count = fms::run_mkgtr_for_input(&input)?;
    print_module_line(format_args!(
        "mkgtr: validated {count} cached Green's-function trace file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `rixs` compatibility stage in the input directory.
pub fn run_rixs(input: PathBuf) -> Result<()> {
    let count = rixs::run_for_input(&input)?;
    print_module_line(format_args!(
        "rixs: validated {count} cached or source-handoff file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `rhorrp` compatibility stage in the input directory.
pub fn run_rhorrp(input: PathBuf) -> Result<()> {
    let count = rhorrp::run_for_input(&input)?;
    print_module_line(format_args!(
        "rhorrp: processed {count} density output file(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `sfconv` compatibility stage in the input directory.
pub fn run_sfconv(input: PathBuf) -> Result<()> {
    sfconv::run_for_input(&input)?;
    print_module_line(format_args!(
        "sfconv: wrote logsfconv.dat beside {}",
        input.display()
    ));
    Ok(())
}

/// Run the supported FEFF `self` (self-energy) compatibility stage in the input directory.
pub fn run_self_energy(input: PathBuf) -> Result<()> {
    let count = sfconv::run_self_for_input(&input)?;
    print_module_line(format_args!(
        "self: validated {count} excitation pole(s) beside {}",
        input.display()
    ));
    Ok(())
}

/// Shared entry point for FEFF10-style standalone module binaries
/// (`bin/pot.rs`, `bin/xsph.rs`, ...): parses a single `-i`/`--input
/// feff.inp` flag and calls `run` with it. Mirrors FEFF10's convention of one
/// executable per module; `refeff module <name>` (see [`Command::Module`])
/// is the equivalent multi-module entry point.
pub fn module_main(
    bin_name: &'static str,
    about: &'static str,
    run: impl FnOnce(PathBuf) -> Result<()>,
) -> Result<()> {
    let matches = clap::Command::new(bin_name)
        .version(env!("CARGO_PKG_VERSION"))
        .about(about)
        .arg(
            clap::Arg::new("input")
                .short('i')
                .long("input")
                .default_value("feff.inp")
                .value_name("INPUT")
                .help("Path to the feff.inp-format input file"),
        )
        .get_matches();
    let input = matches
        .get_one::<String>("input")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("feff.inp"));
    run(input)
}

fn run_feff(input: PathBuf, output: PathBuf) -> Result<()> {
    run_feff_to_dir(&input, &output)
}

fn run_feff_to_dir(input: &Path, output_dir: &Path) -> Result<()> {
    let report = execute_pipeline(input, output_dir, print_rdinp_summary)?;
    render_run_report(report)
}

/// Execute the file-backed FEFF pipeline without printing CLI progress.
///
/// This is the migration boundary used by the `refeff` facade crate. New
/// code should prefer that crate's typed `Runner` API rather than depending
/// directly on `refeff-cli`.
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
    let module_result: Result<()> = (|| {
        run_supported_cached_modules_into(output_dir, &mut module_reports)?;
        run_remaining_required_modules(output_dir, &mut module_reports)
    })();
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

/// The canonical `refeff module <name>` spelling for `name` (its
/// [`clap::ValueEnum`] possible-value name, honoring `#[value(name = ...)]`
/// overrides like `ff2x`/`self`).
fn module_name_str(name: ModuleName) -> String {
    name.to_possible_value()
        .map(|value| value.get_name().to_string())
        .unwrap_or_else(|| format!("{name:?}"))
}

/// Dispatch a single FEFF10 module by name.
///
/// Accepts either a clap-parsed [`ModuleName`] or a free-form `&str` (parsed
/// the same way clap parses `--value-enum` arguments, aliases included) so
/// both the CLI entry point and string-based callers share one dispatch.
/// Emits a [`ModuleReport`] JSON document when `--json` is active; otherwise
/// each `run_<module>` function prints its own status line (see
/// [`print_module_line`]).
fn run_module(name: impl IntoModuleName, input: PathBuf) -> Result<()> {
    let name = name.into_module_name()?;
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

fn run_band_module(input: PathBuf) -> Result<()> {
    let count = band::run_for_input(&input)?;
    print_module_line(format_args!(
        "band: validated {count} cached or source-handoff file(s) beside {}",
        input.display()
    ));
    Ok(())
}

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

#[cfg(test)]
fn run_supported_cached_modules(work_dir: &Path) -> Result<Vec<SupportedModuleReport>> {
    let mut reports = Vec::new();
    run_supported_cached_modules_into(work_dir, &mut reports)?;
    Ok(reports)
}

fn run_supported_cached_modules_into(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
) -> Result<()> {
    let atomic_cached = atomic::has_cached_atomic_output(work_dir)?;
    if atomic_cached || atomic::has_supported_atomic_source_handoff(work_dir)? {
        let stage_start = Instant::now();
        let count = atomic::run_in_dir(work_dir).context("failed to run supported atomic stage")?;
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
        let stage_start = Instant::now();
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

    let pot_cached = pot::has_cached_pot_output(work_dir)?;
    if pot_cached
        || pot::has_supported_pot_source_handoff(work_dir)?
        || pot::has_supported_pot_generation_handoff(work_dir)?
    {
        let stage_start = Instant::now();
        let count = pot::run_in_dir(work_dir).context("failed to run supported pot stage")?;
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

    if fms::has_runnable_fms_solver(work_dir)? {
        let status = if fms::has_cached_fms_solver_output(work_dir)? {
            StageStatus::Cached
        } else {
            StageStatus::Generated
        };
        let stage_start = Instant::now();
        let count = fms::run_fms_in_dir(work_dir).context("failed to run supported fms stage")?;
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

    let eels_cached = eels::has_cached_eels_output(work_dir)?;
    if eels_cached || eels::has_supported_eels_source_handoff(work_dir)? {
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

    let eelsmdff_cached = eelsmdff::has_cached_mdff_output(work_dir)?;
    if eelsmdff_cached || eelsmdff::has_supported_mdff_source_handoff(work_dir)? {
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

fn run_remaining_required_modules(
    work_dir: &Path,
    reports: &mut Vec<SupportedModuleReport>,
) -> Result<()> {
    if !atomic::has_cached_atomic_output(work_dir)? {
        run_required_module(reports, "atomic", "file(s)", || {
            atomic::run_in_dir(work_dir)
        })?;
    }
    if !reports
        .iter()
        .any(|report| report.name == "pot" && report.count > 0)
        && !pot::has_cached_pot_output(work_dir)?
    {
        run_required_module(reports, "pot", "file(s)", || pot::run_in_dir(work_dir))?;
    }
    if !xsph::has_supported_xsph_output(work_dir)?
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
    if !eels::has_cached_eels_output(work_dir)? {
        run_required_module(reports, "eels", "row(s)", || eels::run_in_dir(work_dir))?;
    }
    if !eelsmdff::has_cached_mdff_output(work_dir)? {
        run_required_module(reports, "eelsmdff", "row(s)", || {
            eelsmdff::run_in_dir(work_dir)
        })?;
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

#[cfg(test)]
pub(crate) mod tests;
