#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "screen",
                "Run FEFF10's SCREEN module: core-hole screening / Hubbard-U response.",
                refeff_cli::run_screen,
            )
        })(),
        false,
    );
}
