#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "crpa",
        "Run FEFF10's CRPA module: constrained-RPA Hubbard parameters.",
        refeff_cli::run_crpa,
    )
}
