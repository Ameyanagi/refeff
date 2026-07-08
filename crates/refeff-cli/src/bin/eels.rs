#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "eels",
        "Run FEFF10's EELS module: electron energy-loss spectroscopy.",
        refeff_cli::run_eels,
    )
}
