#![forbid(unsafe_code)]

use clap::Parser;
use refeff_cli::{Cli, UnsupportedModuleError, run_cli};

/// `refeff`'s exit-code taxonomy (also documented in `refeff --help`,
/// long form):
///
/// - `0`: success.
/// - `1`: internal or I/O error.
/// - `2`: command-line usage error; handled entirely by clap inside
///   `Cli::parse()`, which exits before `main` ever sees an `Err`.
/// - `3`: invalid `feff.inp` / input (a [`refeff_io::IoError`] variant other
///   than `Io`, e.g. a parse or malformed-binary-handoff error).
/// - `4`: a [`UnsupportedModuleError`] — a recognized but not-yet-ported
///   FEFF10 module.
fn exit_code_for(error: &anyhow::Error) -> i32 {
    if error.downcast_ref::<UnsupportedModuleError>().is_some() {
        return 4;
    }
    if let Some(io_error) = error.downcast_ref::<refeff_io::IoError>() {
        return match io_error {
            refeff_io::IoError::Io { .. } => 1,
            _ => 3,
        };
    }
    1
}

fn main() {
    if let Err(error) = run_cli(Cli::parse()) {
        eprintln!("error: {error:?}");
        std::process::exit(exit_code_for(&error));
    }
}
