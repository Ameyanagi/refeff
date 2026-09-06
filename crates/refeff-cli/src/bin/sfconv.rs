#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "sfconv",
                "Run FEFF10's SFCONV module: many-body spectral-function convolution.",
                refeff_cli::run_sfconv,
            )
        })(),
        false,
    );
}
