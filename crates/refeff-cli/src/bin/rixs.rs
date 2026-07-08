#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "rixs",
        "Run FEFF10's RIXS module: resonant inelastic X-ray scattering.",
        refeff_cli::run_rixs,
    )
}
