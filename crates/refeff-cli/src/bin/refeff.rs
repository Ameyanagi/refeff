use clap::Parser;
use refeff_cli::{Cli, run_cli};

fn main() -> anyhow::Result<()> {
    run_cli(Cli::parse())
}
