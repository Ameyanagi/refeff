#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "ff2x",
        "Run FEFF10's FF2X module: final spectrum assembly (EXAFS/XANES/DANES/FPRIME).",
        refeff_cli::run_ff2x,
    )
}
