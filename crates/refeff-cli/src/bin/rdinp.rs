use std::path::PathBuf;

use clap::Parser;
use refeff_cli::run_rdinp;

#[derive(Debug, Parser)]
struct RdinpCli {
    #[arg(short, long, default_value = "feff.inp")]
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = RdinpCli::parse();
    run_rdinp(cli.input)
}
