#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "fullspectrum",
        "Run FEFF10's FULLSPECTRUM module: optical constants across the full spectral range.",
        refeff_cli::run_fullspectrum,
    )
}
