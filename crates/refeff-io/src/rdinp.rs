//! Writers for the first `rdinp`-level compatibility outputs.
//!
//! The full FEFF `rdinp` module emits many module input files. This module
//! starts with `atoms.dat`, which is the structural bridge consumed by later
//! FEFF modules, and will grow as the port advances.

use crate::model::{Atom, FeffDocument};
use crate::{IoError, Result};

/// Render FEFF-compatible `atoms.dat` content from an [`FeffDocument`].
pub fn atoms_dat_string(document: &FeffDocument) -> Result<String> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write atoms.dat without ATOMS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_atoms_dat(document, &mut out)?;
    Ok(out)
}

/// Write FEFF-compatible `atoms.dat` content into an arbitrary formatter.
pub fn write_atoms_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "natx =  {:>7}", document.atoms.len()).expect("write to string");
    writeln!(out, "    x       y        z       iph  ").expect("write to string");

    let origin = &document.atoms[0];
    for atom in &document.atoms {
        let distance = atom.distance.unwrap_or_else(|| distance_from(origin, atom));
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:4}{:13.5}",
            atom.x, atom.y, atom.z, atom.ipot, distance
        )
        .expect("write to string");
    }

    Ok(())
}

fn distance_from(origin: &Atom, atom: &Atom) -> f64 {
    let dx = atom.x - origin.x;
    let dy = atom.y - origin.y;
    let dz = atom.z - origin.z;
    dx.hypot(dy).hypot(dz)
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput};

    use super::atoms_dat_string;

    #[test]
    fn writes_atoms_dat_with_feff_widths() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 2.0 2.0 1 Cu1
END
"#,
        )
        .expect("parse");
        let doc = FeffDocument::from_input(&input).expect("document");
        let atoms = atoms_dat_string(&doc).expect("atoms.dat");

        assert_eq!(
            atoms,
            "natx =        2\n    x       y        z       iph  \n      0.00000      0.00000      0.00000   0      0.00000\n      1.00000      2.00000      2.00000   1      3.00000\n"
        );
    }
}
