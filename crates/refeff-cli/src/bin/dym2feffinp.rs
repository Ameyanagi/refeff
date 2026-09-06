#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> { refeff_cli::dym2feffinp_main() })(),
        false,
    );
}
