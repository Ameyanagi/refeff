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

    /// Bound process-wide Rayon/faer worker threads.
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,
}

/// A `refeff` subcommand.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Validate `feff.inp` without writing files.
    #[command(alias = "inspect")]
    Check {
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
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
        #[arg(value_enum)]
        name: ModuleName,
        /// Path to the FEFF input.
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    /// Generate a shell completion script on stdout.
    Completions {
        /// Shell to generate the completion script for.
        shell: clap_complete::Shell,
    },
}

/// Module names accepted by `refeff module`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModuleName {
    Rdinp,
    Pot,
    #[value(alias = "atom")]
    Atomic,
    Band,
    #[value(alias = "eelsmdff")]
    Mdff,
    Wpot,
    #[value(alias = "opconsat")]
    Opcons,
    Compton,
    Fullspectrum,
    Crpa,
    Screen,
    Ldos,
    Eels,
    Dmdw,
    #[value(alias = "paths")]
    Path,
    Genfmt,
    #[value(name = "ff2x")]
    Ff2x,
    Xsph,
    Fms,
    Mkgtr,
    Rixs,
    Rhorrp,
    Sfconv,
    #[value(name = "self", alias = "selfenergy")]
    SelfEnergy,
}

impl From<ModuleName> for refeff_engine::ModuleName {
    fn from(name: ModuleName) -> Self {
        match name {
            ModuleName::Rdinp => Self::Rdinp,
            ModuleName::Pot => Self::Pot,
            ModuleName::Atomic => Self::Atomic,
            ModuleName::Band => Self::Band,
            ModuleName::Mdff => Self::Mdff,
            ModuleName::Wpot => Self::Wpot,
            ModuleName::Opcons => Self::Opcons,
            ModuleName::Compton => Self::Compton,
            ModuleName::Fullspectrum => Self::Fullspectrum,
            ModuleName::Crpa => Self::Crpa,
            ModuleName::Screen => Self::Screen,
            ModuleName::Ldos => Self::Ldos,
            ModuleName::Eels => Self::Eels,
            ModuleName::Dmdw => Self::Dmdw,
            ModuleName::Path => Self::Path,
            ModuleName::Genfmt => Self::Genfmt,
            ModuleName::Ff2x => Self::Ff2x,
            ModuleName::Xsph => Self::Xsph,
            ModuleName::Fms => Self::Fms,
            ModuleName::Mkgtr => Self::Mkgtr,
            ModuleName::Rixs => Self::Rixs,
            ModuleName::Rhorrp => Self::Rhorrp,
            ModuleName::Sfconv => Self::Sfconv,
            ModuleName::SelfEnergy => Self::SelfEnergy,
        }
    }
}

/// Dispatch parsed frontend arguments into the computational engine.
pub fn run_cli(cli: Cli) -> Result<()> {
    refeff_engine::configure_parallelism(cli.threads);
    let mode = refeff_engine::OutputMode {
        verbose: cli.verbose,
        quiet: cli.quiet,
        json: cli.json,
    };
    let dir = cli.dir.as_deref();
    let command = cli.command.unwrap_or_else(|| Command::Run {
        input: PathBuf::from("feff.inp"),
        output: PathBuf::from("."),
    });
    refeff_engine::with_output_mode(mode, || match command {
        Command::Check { input } => refeff_engine::run_check(resolve_path(dir, input)),
        Command::Rdinp { input, output } => {
            run_rdinp(resolve_path(dir, input), resolve_path(dir, output))
        }
        Command::Run { input, output } => {
            refeff_engine::run_feff(resolve_path(dir, input), resolve_path(dir, output))
        }
        Command::Module { name, input } => {
            refeff_engine::run_named_module(name.into(), resolve_path(dir, input))
        }
        Command::Completions { shell } => print_completions(shell),
    })
}

fn resolve_path(dir: Option<&Path>, path: PathBuf) -> PathBuf {
    match dir {
        Some(dir) if path.is_relative() => dir.join(path),
        _ => path,
    }
}

fn print_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut command = <Cli as clap::CommandFactory>::command();
    let name = command.get_name().to_string();
    clap_complete::generate(shell, &mut command, name, &mut std::io::stdout());
    Ok(())
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
