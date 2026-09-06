#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "genfmt",
                "Run FEFF10's GENFMT module: path scattering-amplitude tables.",
                refeff_cli::run_genfmt,
            )
        })(),
        false,
    );
}
