#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "opconsat",
        "Run FEFF10's OPCONSAT module: optical constants from dielectric data.",
        refeff_cli::run_opcons,
    )
}
