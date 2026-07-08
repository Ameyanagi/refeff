#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "mdff",
        "Run FEFF10's EELSMDFF module: EELS mixed dynamic form factor.",
        refeff_cli::run_mdff,
    )
}
