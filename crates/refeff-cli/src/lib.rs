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

mod dym2feffinp;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

pub use refeff_engine::{
    CheckReport, RdinpReport, RunReport, StageStatus, SupportedModuleReport, execute_feff,
    run_atomic, run_band, run_compton, run_crpa, run_dmdw, run_eels, run_ff2x, run_fms,
    run_fullspectrum, run_genfmt, run_ldos, run_mdff, run_mkgtr, run_opcons, run_path, run_pot,
    run_rdinp, run_rhorrp, run_rixs, run_screen, run_self_energy, run_sfconv, run_wpot, run_xsph,
};

/// Top-level arguments for the `refeff` and FEFF-compatible frontends.
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

    /// Print one line per pipeline stage plus extra timing detail.
    #[arg(short, long, global = true, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Suppress per-stage progress lines.
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Emit one machine-readable JSON document on stdout.
    #[arg(long, global = true)]
    pub json: bool,

    /// Operate as if started in DIR, like `git -C`.
    #[arg(short = 'C', long = "dir", global = true, value_name = "DIR")]
    pub dir: Option<PathBuf>,

    /// Worker threads for this calculation (0 selects automatic sizing).
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
}

/// A `refeff` subcommand.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Create a runnable ZnSe EXAFS example (refuses to overwrite a file).
    Init {
        #[arg(short, long, default_value = "feff.inp")]
        output: PathBuf,
    },
    /// Inspect a generated FEFF file using the format registry.
    Inspect { path: PathBuf },
    /// Show requested stages, prerequisites and output placement without running.
    Plan {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Validate `feff.inp` without writing files.
    Check {
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Parse a partial document without requiring a complete calculation.
        #[arg(long)]
        syntax_only: bool,
    },
    /// Run only the RDINP input-parsing stage.
    Rdinp {
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Directory for RDINP handoff files.
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Run the complete supported FEFF10 pipeline.
    Run {
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Directory for generated FEFF-format files.
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
        /// Existing output: reuse, recompute (replace the output tree), or error.
        #[arg(long, value_enum, default_value = "reuse")]
        existing: OutputPolicy,
    },
    /// Run one FEFF10 module by name.
    #[command(after_help = "Supported names:\n  \
                             rdinp, pot, atomic (alias: atom), band, mdff (alias: eelsmdff), \
                             wpot, opcons (alias: opconsat), compton, fullspectrum, crpa, \
                             screen, ldos, eels, dmdw, path (alias: paths), genfmt, ff2x, \
                             xsph, fms, mkgtr, rixs, rhorrp, sfconv, \
                             self (alias: selfenergy)")]
    Module {
        /// Module name or FEFF10 historical alias.
        #[arg(value_parser = parse_module)]
        name: ModuleName,
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
        /// Handoff/output directory; defaults to -C DIR or the input parent.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate a shell completion script on stdout.
    Completions {
        /// Shell to generate the completion script for.
        shell: clap_complete::Shell,
    },
}

/// Treatment of an existing calculation directory.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputPolicy {
    Reuse,
    Recompute,
    Error,
}

pub use refeff_engine::ModuleName;

/// Dispatch parsed frontend arguments into the computational engine.
pub fn run_cli(cli: Cli) -> Result<()> {
    use serde_json::json;
    use std::num::NonZeroUsize;
    let threads = match cli.threads {
        Some(n) => Some(n),
        None => std::env::var("REFEFF_THREADS")
            .ok()
            .map(|value| {
                value.parse::<usize>().map_err(|_| {
                    UsageError(format!(
                        "REFEFF_THREADS must be a nonnegative integer, got {value:?}"
                    ))
                })
            })
            .transpose()?,
    };
    let dir = cli.dir.as_deref();
    let command = cli.command.unwrap_or_else(|| Command::Run {
        input: PathBuf::from("feff.inp"),
        output: PathBuf::from("."),
        existing: OutputPolicy::Reuse,
    });
    let (mut data, summary) = match command {
        Command::Init { output } => {
            use std::io::Write;
            let output = resolve_path(dir, output);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)?;
            file.write_all(include_bytes!("../examples/znse.inp"))?;
            (
                json!({"input":output}),
                format!(
                    "Created {}. Next: refeff plan -i {}",
                    output.display(),
                    output.display()
                ),
            )
        }
        Command::Inspect { path } => {
            let path = resolve_path(dir, path);
            let descriptor = refeff_io::codec::identify_format(&path)
                .ok_or_else(|| anyhow::anyhow!("unregistered FEFF file: {}", path.display()))?;
            let bytes = std::fs::metadata(&path)?.len();
            let points = match descriptor.format {
                refeff_io::codec::FileFormat::ChiDat => {
                    Some(refeff_io::chi_dat::read_chi_dat(&path)?.point_count())
                }
                refeff_io::codec::FileFormat::XmuDat => Some(
                    refeff_io::xmu_dat::read_xmu_dat(&path)?
                        .photon_energy_ev
                        .len(),
                ),
                refeff_io::codec::FileFormat::PhaseBin => {
                    Some(refeff_io::read_phase_bin(&path)?.energy_count)
                }
                refeff_io::codec::FileFormat::PathsDat => {
                    Some(refeff_io::read_paths_dat(&path)?.paths.len())
                }
                _ => None,
            };
            let format = format!("{:?}", descriptor.format);
            (
                json!({"path":path,"format":format,"representation":format!("{:?}",descriptor.representation),"producer":descriptor.producer,"bytes":bytes,"points":points,"payload_validated":points.is_some()}),
                format!(
                    "{}: {format}, {bytes} bytes{}",
                    path.display(),
                    points.map(|n| format!(", {n} points")).unwrap_or_default()
                ),
            )
        }
        Command::Plan { input, output } => {
            let input = resolve_path(dir, input);
            let output = resolve_path(dir, output);
            let plan = refeff_engine::plan(&input, &output)?;
            let summary = format!(
                "Output: {}\n{}\nCache decisions are validated during execution.",
                output.display(),
                plan.stages
                    .iter()
                    .map(|stage| format!("{}: {}", stage.name, stage.prerequisites.join(", ")))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            (serde_json::to_value(plan)?, summary)
        }
        Command::Check { input, syntax_only } => {
            let report = refeff_engine::check_with_options(&resolve_path(dir, input), syntax_only)?;
            let summary = format!(
                "OK: {} cards, {} atoms, {} potentials; edge={}",
                report.cards,
                report.atoms,
                report.potentials,
                report.edge.as_deref().unwrap_or("default")
            );
            (serde_json::to_value(report)?, summary)
        }
        Command::Rdinp { input, output } => {
            let input = resolve_path(dir, input);
            let output = resolve_path(dir, output);
            let report = refeff_engine::execute_rdinp(&input, &output)?;
            (
                serde_json::to_value(report)?,
                format!("rdinp: handoffs written to {}", output.display()),
            )
        }
        Command::Run {
            input,
            output,
            existing,
        } => {
            let input = resolve_path(dir, input);
            let output = resolve_path(dir, output);
            let mut runner = refeff::Runner::new();
            if !cli.quiet {
                runner = runner.with_progress_sink(std::sync::Arc::new(ConsoleProgress {
                    verbose: cli.verbose,
                }));
            }
            if let Some(threads) = threads.and_then(NonZeroUsize::new) {
                runner = runner.with_threads(threads);
            }
            let policy = match existing {
                OutputPolicy::Reuse => refeff::ExistingOutputPolicy::ReuseValidated,
                OutputPolicy::Recompute => refeff::ExistingOutputPolicy::Recompute,
                OutputPolicy::Error => refeff::ExistingOutputPolicy::ErrorOnConflict,
            };
            let report = runner.run_files(
                refeff::FileRunRequest::new(&input, &output).with_existing_output_policy(policy),
            )?;
            let spectra = ["xmu.dat", "chi.dat", "eels.dat", "rixsET.dat", "opcons.dat"]
                .into_iter()
                .filter(|name| output.join(name).is_file())
                .map(|name| output.join(name).display().to_string())
                .collect::<Vec<_>>();
            let summary = format!(
                "Completed {} stages in {}{}",
                report.stages.len(),
                output.display(),
                if spectra.is_empty() {
                    String::new()
                } else {
                    format!("\nSpectra: {}", spectra.join(", "))
                }
            );
            (serde_json::to_value(report)?, summary)
        }
        Command::Module {
            name,
            input,
            output,
        } => {
            let input = resolve_path(dir, input);
            let output = output
                .map(|p| resolve_path(dir, p))
                .or_else(|| dir.map(Path::to_path_buf))
                .unwrap_or_else(|| {
                    input
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new("."))
                        .to_path_buf()
                });
            let options = refeff_engine::execution::ExecutionOptions {
                threads,
                ..Default::default()
            };
            let report = refeff_engine::execution::with_execution(&options, || {
                refeff_engine::execute_module(name, &input, &output)
            })?;
            let summary = format!(
                "{}: processed {} item(s) in {}",
                report.module,
                report.count,
                output.display()
            );
            (serde_json::to_value(report)?, summary)
        }
        Command::Completions { shell } => {
            let mut command = <Cli as clap::CommandFactory>::command();
            let name = command.get_name().to_string();
            let mut bytes = Vec::new();
            clap_complete::generate(shell, &mut command, name, &mut bytes);
            let script = String::from_utf8(bytes)?;
            (json!({"shell":shell.to_string(),"script":script}), script)
        }
    };
    data["effective_threads"] = json!(threads.filter(|n| *n > 0).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }));
    if cli.json {
        write_json(&json!({"schema_version": 1, "ok": true, "data": data}))?;
    } else if !cli.quiet {
        use std::io::Write;
        writeln!(std::io::stdout().lock(), "{summary}")?;
    }
    Ok(())
}

fn write_json(value: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    writeln!(stdout)?;
    Ok(())
}

/// Stable exit codes shared by both frontend names and standalone modules.
pub fn exit_code_for(error: &anyhow::Error) -> i32 {
    if error.is::<UsageError>() {
        return 2;
    }
    if let Some(refeff::Error::Pipeline { code, .. }) = error.downcast_ref::<refeff::Error>() {
        return match *code {
            "input" => 3,
            "interrupted" => 130,
            _ => 1,
        };
    }
    if let Some(error) = error.downcast_ref::<refeff_io::IoError>() {
        return if matches!(error, refeff_io::IoError::Io { .. }) {
            1
        } else {
            3
        };
    }
    1
}

/// Terminate a frontend consistently, rendering machine-readable errors when requested.
pub fn finish(result: Result<()>, json: bool) {
    if let Err(error) = result {
        let code = exit_code_for(&error);
        if json {
            let value = serde_json::json!({"schema_version": 1, "ok": false,
                "error": error_details(&error, code)});
            if let Err(write_error) = write_json(&value) {
                eprintln!("error: {write_error:#}");
            }
        } else {
            eprintln!("error: {error:#}");
        }
        std::process::exit(code);
    }
}

/// Entrypoint for the equivalent `refeff` and `feff` frontends.
pub fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    let json = args.iter().any(|arg| arg == "--json");
    match Cli::try_parse_from(args) {
        Ok(cli) => finish(run_cli(cli), json),
        Err(error) if json && error.exit_code() == 0 => {
            finish(
                write_json(
                    &serde_json::json!({"schema_version":1,"ok":true,"data":{"text":error.to_string()}}),
                ),
                true,
            );
        }
        Err(error) if json && error.exit_code() != 0 => {
            let _ = write_json(
                &serde_json::json!({"schema_version":1,"ok":false,"error":{"exit_code":2,"code":"usage","message":error.to_string()}}),
            );
            std::process::exit(2);
        }
        Err(error) => error.exit(),
    }
}

fn resolve_path(dir: Option<&Path>, path: PathBuf) -> PathBuf {
    match dir {
        Some(dir) if path.is_relative() => dir.join(path),
        _ => path,
    }
}

/// Shared parser for FEFF10-style standalone module binaries.
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

/// Convert a FEFF `.dym` file into matching FEFF input and reordered `.dym`.
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

/// Parse and run the standalone `dym2feffinp` frontend.
pub fn dym2feffinp_main() -> Result<()> {
    dym2feffinp::main()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_aliases_map_to_engine_names() -> Result<()> {
        for (alias, expected) in [
            ("atom", refeff_engine::ModuleName::Atomic),
            ("eelsmdff", refeff_engine::ModuleName::Mdff),
            ("opconsat", refeff_engine::ModuleName::Opcons),
            ("paths", refeff_engine::ModuleName::Path),
            ("selfenergy", refeff_engine::ModuleName::SelfEnergy),
        ] {
            let cli = Cli::try_parse_from(["refeff", "module", alias])?;
            let Some(Command::Module { name, .. }) = cli.command else {
                anyhow::bail!("module command was not parsed");
            };
            assert_eq!(refeff_engine::ModuleName::from(name), expected);
        }
        Ok(())
    }

    #[test]
    fn relative_paths_resolve_against_cli_dir() {
        assert_eq!(
            resolve_path(Some(Path::new("work")), PathBuf::from("feff.inp")),
            PathBuf::from("work/feff.inp")
        );
        let absolute = std::env::temp_dir().join("feff.inp");
        assert_eq!(
            resolve_path(Some(Path::new("work")), absolute.clone()),
            absolute
        );
    }

    #[test]
    fn cli_rdinp_routes_output_to_requested_directory() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let input = temp.path().join("feff.inp");
        let output = temp.path().join("rdinp-output");
        std::fs::write(
            &input,
            "TITLE CLI routing smoke\n\
             EDGE K\n\
             CONTROL 1 1 1 1 1 1\n\
             POTENTIALS\n\
             0 29 Cu\n\
             1 29 Cu\n\
             ATOMS\n\
             0.0 0.0 0.0 0 Cu0\n\
             1.8 1.8 0.0 1 Cu1\n\
             END\n",
        )?;

        run_cli(Cli {
            command: Some(Command::Rdinp {
                input,
                output: output.clone(),
            }),
            verbose: false,
            quiet: true,
            json: false,
            dir: None,
            threads: None,
        })?;

        assert!(output.join("global.inp").is_file());
        assert!(output.join("pot.inp").is_file());
        assert!(!temp.path().join("global.inp").exists());
        Ok(())
    }
}

struct ConsoleProgress {
    verbose: bool,
}
impl refeff::ProgressSink for ConsoleProgress {
    fn event(&self, event: refeff::ProgressEvent<'_>) {
        match event {
            refeff::ProgressEvent::StageStarted(name) => eprintln!("{name}: starting"),
            refeff::ProgressEvent::StageCompleted(stage) => eprintln!(
                "{}: {:?} {} {} ({} ms)",
                stage.name, stage.action, stage.count, stage.unit, stage.duration_ms
            ),
            refeff::ProgressEvent::StageProgress {
                name,
                completed,
                total,
            } if self.verbose => {
                eprintln!("{name}: {completed}/{total}");
            }
            _ => {}
        }
    }
}

fn parse_module(value: &str) -> std::result::Result<ModuleName, String> {
    ModuleName::parse(value).map_err(|_| {
        format!(
            "unknown module {value:?}; choose {}",
            refeff_engine::MODULES
                .iter()
                .map(|item| item.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn error_details(error: &anyhow::Error, exit_code: i32) -> serde_json::Value {
    let code = match exit_code {
        2 => "usage",
        3 => "input",
        130 => "interrupted",
        _ => "execution",
    };
    let mut detail =
        serde_json::json!({"code":code,"exit_code":exit_code,"message":format!("{error:#}")});
    if let Some(refeff::Error::Pipeline {
        code,
        module,
        stages,
        ..
    }) = error.downcast_ref::<refeff::Error>()
    {
        detail["code"] = serde_json::json!(code);
        detail["module"] = serde_json::json!(module);
        detail["completed_stages"] = serde_json::json!(stages);
    }
    for source in error.chain() {
        if source.is::<std::io::Error>() {
            detail["code"] = serde_json::json!("io");
        }
        if let Some(refeff_io::IoError::Parse { path, line, .. }) =
            source.downcast_ref::<refeff_io::IoError>()
        {
            detail["path"] = serde_json::json!(path);
            detail["line"] = serde_json::json!(line);
        }
        if let Some(refeff_io::IoError::Io { path, .. } | refeff_io::IoError::Codec { path, .. }) =
            source.downcast_ref::<refeff_io::IoError>()
        {
            detail["path"] = serde_json::json!(path);
        }
    }
    detail
}

#[derive(Debug)]
struct UsageError(String);
impl std::fmt::Display for UsageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for UsageError {}
