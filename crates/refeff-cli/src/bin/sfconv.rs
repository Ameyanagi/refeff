#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "sfconv",
        "Run FEFF10's SFCONV module: many-body spectral-function convolution.",
        refeff_cli::run_sfconv,
    )
}
