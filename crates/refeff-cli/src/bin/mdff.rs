#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::Parser;
use refeff_cli::run_mdff;

#[derive(Debug, Parser)]
struct MdffCli {
    #[arg(short, long, default_value = "feff.inp")]
    input: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = MdffCli::parse();
    run_mdff(cli.input)
}
