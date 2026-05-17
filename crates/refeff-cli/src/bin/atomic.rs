#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;
use refeff_cli::run_atomic;

#[derive(Debug, Parser)]
struct AtomicCli {
    #[arg(short, long, default_value = "feff.inp")]
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = AtomicCli::parse();
    run_atomic(cli.input)
}
