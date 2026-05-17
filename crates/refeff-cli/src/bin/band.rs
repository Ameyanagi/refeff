#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;
use refeff_cli::run_band;

#[derive(Debug, Parser)]
struct BandCli {
    #[arg(short, long, default_value = "feff.inp")]
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = BandCli::parse();
    run_band(cli.input)
}
