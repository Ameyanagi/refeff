//! Parse an embedded FEFF `feff.inp` and print its cards, atoms, and
//! potentials.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p refeff-io --example parse_feff_inp
//! ```

use refeff_io::FeffInput;

/// A minimal Cu K-edge EXAFS `feff.inp`, equivalent in shape to
/// `feff10/examples/EXAFS/Cu/feff.inp`: a `TITLE`, the absorption `EDGE`, a
/// two-row `POTENTIALS` table (absorber plus one scattering type), and a
/// two-atom `ATOMS` cluster referencing those potential indices.
const CU_FEFF_INP: &str = r#"
TITLE Cu metal, fcc, K edge

EDGE      K
S02       1.0
CONTROL   1 1 1 1 1 1
PRINT     1 0 0 0 0 0

POTENTIALS
   0   29   Cu
   1   29   Cu

ATOMS
   0.00000   0.00000   0.00000   0   Cu1
   1.80500   1.80500   0.00000   1   Cu2

END
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = FeffInput::parse_str("feff.inp (embedded Cu example)", CU_FEFF_INP)?;

    println!("cards:");
    for line in input.cards() {
        println!("  {}", line.raw);
    }

    println!("atoms:");
    for row in input.section_rows("ATOMS") {
        println!("  {}", row.raw);
    }

    println!("potentials:");
    for row in input.section_rows("POTENTIALS") {
        println!("  {}", row.raw);
    }

    let card_count = input.cards().count();
    let atom_count = input.section_rows("ATOMS").count();
    let potential_count = input.section_rows("POTENTIALS").count();
    println!("summary: {card_count} card(s), {atom_count} atom(s), {potential_count} potential(s)");

    Ok(())
}
