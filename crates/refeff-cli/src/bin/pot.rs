#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;
use refeff_cli::run_pot;

#[derive(Debug, Parser)]
struct PotCli {
    #[arg(short, long, default_value = "feff.inp")]
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = PotCli::parse();
    run_pot(cli.input)
}
