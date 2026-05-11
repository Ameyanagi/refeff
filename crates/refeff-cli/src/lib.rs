use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use refeff_io::{FeffDocument, FeffInput, rdinp};

#[derive(Debug, Parser)]
#[command(
    name = "refeff",
    version,
    about = "Pure-Rust FEFF10 compatibility port"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Inspect {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Rdinp {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Run {
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
    Module {
        name: String,
        #[arg(short, long, default_value = "feff.inp")]
        input: PathBuf,
    },
}

pub fn run_cli(cli: Cli) -> Result<()> {
    match cli.command.unwrap_or(Command::Run {
        input: PathBuf::from("feff.inp"),
    }) {
        Command::Inspect { input } => inspect(input),
        Command::Rdinp { input } => run_rdinp(input),
        Command::Run { input } => run_feff(input),
        Command::Module { name, input } => run_module(&name, input),
    }
}

fn inspect(input: PathBuf) -> Result<()> {
    let parsed = FeffInput::parse_file(&input)?;
    let cards = parsed.cards().count();
    let atom_rows = parsed.section_rows("ATOMS").count();
    let potential_rows = parsed.section_rows("POTENTIALS").count();
    println!(
        "input={} cards={} atoms={} potentials={}",
        input.display(),
        cards,
        atom_rows,
        potential_rows
    );
    Ok(())
}

pub fn run_rdinp(input: PathBuf) -> Result<()> {
    let parsed = FeffInput::parse_file(&input)?;
    let document = FeffDocument::from_input(&parsed)?;
    for (name, content) in rdinp::text_outputs(&document)? {
        std::fs::write(name, content)?;
    }
    println!(
        "rdinp: parsed {} cards, {} atoms, {} potentials",
        parsed.cards().count(),
        document.atoms.len(),
        document.potentials.len()
    );
    Ok(())
}

fn run_feff(input: PathBuf) -> Result<()> {
    let parsed = FeffInput::parse_file(&input)?;
    bail!(
        "full FEFF numerical execution is not implemented yet; parsed {} active lines from {}",
        parsed.lines.len(),
        input.display()
    )
}

fn run_module(name: &str, input: PathBuf) -> Result<()> {
    if name.eq_ignore_ascii_case("rdinp") {
        return run_rdinp(input);
    }

    let parsed = FeffInput::parse_file(&input)?;
    bail!(
        "module {name} is not implemented yet; parsed {} active lines from {}",
        parsed.lines.len(),
        input.display()
    )
}
