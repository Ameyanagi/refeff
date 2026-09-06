#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "opcons",
                "Run FEFF10's OPCONSAT module: optical constants from a dielectric-function cache.",
                refeff_cli::run_opcons,
            )
        })(),
        false,
    );
}
