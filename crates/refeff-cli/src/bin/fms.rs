#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "fms",
        "Run FEFF10's FMS/MKGTR module: full multiple scattering / Green's function.",
        refeff_cli::run_fms,
    )
}
