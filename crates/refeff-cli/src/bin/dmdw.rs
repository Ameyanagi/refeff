#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "dmdw",
        "Run FEFF10's DMDW module: dynamical-matrix Debye-Waller factors.",
        refeff_cli::run_dmdw,
    )
}
