#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "mkgtr",
        "Run FEFF10's MKGTR stage: project Green's functions into trace outputs.",
        refeff_cli::run_mkgtr,
    )
}
