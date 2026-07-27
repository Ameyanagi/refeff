//! Run the FEFF `rdinp` compatibility stage on an embedded `feff.inp` and
//! write its generated handoff files into a temporary directory, using only
//! `refeff_engine`'s public API.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p refeff-engine --example full_run
//! ```

use std::io::Write as _;

use refeff_engine::run_rdinp;

/// A minimal Cu K-edge EXAFS `feff.inp`.
const CU_FEFF_INP: &str = r#"
TITLE Cu metal, fcc, K edge
EDGE K
CONTROL 1 1 1 1 1 1
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.00000 0.00000 0.00000 0 Cu1
1.80500 1.80500 0.00000 1 Cu2
END
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workdir = tempfile::tempdir()?;
    let input_path = workdir.path().join("feff.inp");
    std::fs::File::create(&input_path)?.write_all(CU_FEFF_INP.as_bytes())?;

    let output_dir = workdir.path().join("run");
    std::fs::create_dir_all(&output_dir)?;

    // `run_rdinp` parses `feff.inp` and writes FEFF's RDINP handoff files
    // (`global.inp`, `pot.inp`, `atoms.dat`, `geom.dat`, ...) into
    // `output_dir`, printing the FEFF-style RDINP summary to stdout.
    run_rdinp(input_path, output_dir.clone())?;

    let mut generated: Vec<String> = std::fs::read_dir(&output_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    generated.sort();
    println!("generated files in {}:", output_dir.display());
    for name in generated {
        println!("  {name}");
    }

    Ok(())
}
