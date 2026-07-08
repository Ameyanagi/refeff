#![forbid(unsafe_code)]

fn main() -> anyhow::Result<()> {
    refeff_cli::module_main(
        "atomic",
        "Run FEFF10's ATOM module: free-atom potentials and wavefunctions.",
        refeff_cli::run_atomic,
    )
}
