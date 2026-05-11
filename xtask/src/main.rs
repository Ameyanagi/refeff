use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "xtask")]
struct Xtask {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    ReferenceTests,
    BenchE2e,
}

fn main() -> Result<()> {
    let xtask = Xtask::parse();
    match xtask.command {
        Command::ReferenceTests => {
            println!(
                "reference test orchestration will run FEFF10 from FEFF10_REF once module ports land"
            );
        }
        Command::BenchE2e => {
            println!(
                "end-to-end benchmark orchestration will compare Rust and FEFF10 once execution is available"
            );
        }
    }
    Ok(())
}
