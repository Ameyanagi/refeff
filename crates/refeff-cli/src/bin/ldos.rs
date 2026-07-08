#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "ldos",
        "Run FEFF10's LDOS module: local density of states.",
        refeff_cli::run_ldos,
    )
}
