#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "self",
        "Run FEFF10's SELF module: self-energy correction poles.",
        refeff_cli::run_self_energy,
    )
}
