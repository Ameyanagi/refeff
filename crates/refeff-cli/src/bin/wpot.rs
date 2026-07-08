#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "wpot",
        "Render FEFF10 POT output files (potXX.dat) from cached potential state.",
        refeff_cli::run_wpot,
    )
}
