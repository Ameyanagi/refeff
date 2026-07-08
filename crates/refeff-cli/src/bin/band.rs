#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "band",
        "Run FEFF10's BAND module: band structure / KKR calculation.",
        refeff_cli::run_band,
    )
}
