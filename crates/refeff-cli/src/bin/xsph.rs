#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "xsph",
        "Run FEFF10's XSPH module: phase shifts and cross sections.",
        refeff_cli::run_xsph,
    )
}
