#![forbid(unsafe_code)]

fn main() {
    refeff_cli::finish(
        (|| -> anyhow::Result<()> {
            refeff_cli::module_main(
                "rhorrp",
                "Run FEFF10's RHORRP module: charge-density grid.",
                refeff_cli::run_rhorrp,
            )
        })(),
        false,
    );
}
