use super::*;

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

/// Render FEFF-compatible `.dimensions.dat` content from an [`FeffDocument`].
pub fn dimensions_dat_string(document: &FeffDocument) -> Result<String> {
    let (nclusx, lx, nphu, nspu) = dimensions_values(document)?;
    Ok(format!("{nclusx:12}{lx:12}{nphu:12}{nspu:12}\n"))
}

/// Render FEFF-compatible `geom.dat` content from an [`FeffDocument`].
pub fn geom_dat_string(document: &FeffDocument) -> Result<String> {
    if document.atoms.is_empty() {
        return Err(IoError::Parse {
            path: document.source.clone(),
            line: 0,
            message: "cannot write geom.dat without ATOMS rows".to_string(),
        });
    }

    let mut out = String::new();
    write_geom_dat(document, &mut out)?;
    Ok(out)
}

/// Write FEFF-compatible `atoms.dat` content into an arbitrary formatter.
pub fn write_atoms_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    writeln!(out, "natx =  {:>7}", document.atoms.len())?;
    writeln!(out, "    x       y        z       iph  ")?;

    let origin = &document.atoms[0];
    for atom in &document.atoms {
        let distance = distance_from(origin, atom);
        writeln!(
            out,
            "{:13.5}{:13.5}{:13.5}{:4}{:13.5}",
            atom.x, atom.y, atom.z, atom.ipot, distance
        )?;
    }

    Ok(())
}

fn write_geom_dat(document: &FeffDocument, out: &mut impl std::fmt::Write) -> Result<()> {
    let rows = geometry_rows(document)?;
    let iatph = geometry_model_atoms(document, &rows);
    let nph = iatph.len().saturating_sub(1);

    writeln!(out, "nat, nph = {:5}{:5}", rows.len(), nph)?;
    for model_atom in &iatph {
        write!(out, "{model_atom:5}")?;
    }
    writeln!(out)?;
    writeln!(out, " iat     x       y        z       iph  ")?;
    writeln!(out, " {}", "-".repeat(71))?;
    for (idx, row) in rows.iter().enumerate() {
        writeln!(
            out,
            "{:4}{:13.5}{:13.5}{:13.5}{:4}{:4}",
            idx + 1,
            row.x,
            row.y,
            row.z,
            row.ipot,
            1
        )?;
    }
    Ok(())
}
