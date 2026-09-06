#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "pot",
                "Run FEFF10's POT module: self-consistent muffin-tin potentials.",
                refeff_cli::run_pot,
            )
        })(),
        false,
    );
}
