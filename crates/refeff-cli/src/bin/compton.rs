#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "compton",
        "Run FEFF10's COMPTON module: Compton profiles.",
        refeff_cli::run_compton,
    )
}
