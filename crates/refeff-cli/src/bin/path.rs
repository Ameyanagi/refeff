#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "path",
        "Run FEFF10's PATH module: scattering path finder.",
        refeff_cli::run_path,
    )
}
