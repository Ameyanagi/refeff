//! FEFF `paths.dat` text codec.
//!
//! `PATH/pathsd.f90` writes this file for GENFMT. The header is free text
//! terminated by FEFF's dashed separator, followed by path records. Each path
//! starts with `index`, `nleg`, degeneracy, and `r=...`; the next line is a row
//! label, then `nleg` atom rows. Full PATH output includes `rleg`, `beta`, and
//! `eta`; RDINP-generated single-scattering rows omit those trailing values.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};

/// One atom row from FEFF `paths.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatAtom {
    /// Cartesian position in Angstroms as written by FEFF.
    pub position_angstrom: [f64; 3],
    /// Potential index, `ipot`.
    pub potential_index: i32,
    /// FEFF potential label without surrounding quotes.
    pub label: String,
    /// Leg length in Angstroms when present in full PATH output.
    pub leg_distance_angstrom: Option<f64>,
    /// Euler beta angle in degrees when present in full PATH output.
    pub beta_degrees: Option<f64>,
    /// Euler eta angle in degrees when present in full PATH output.
    pub eta_degrees: Option<f64>,
}

/// One FEFF `paths.dat` path record.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatPath {
    /// Path index, `ipath`.
    pub index: usize,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Effective half path length in Angstroms, from the `r=` header field.
    pub effective_half_path_length_angstrom: f64,
    /// The label line between the path header and atom rows.
    pub row_header: String,
    /// Atom rows, including the final absorber row.
    pub atoms: Vec<PathsDatAtom>,
}

impl PathsDatPath {
    /// Number of legs, `nleg`.
    #[must_use]
    pub fn leg_count(&self) -> usize {
        self.atoms.len()
    }

    /// Potential indices as an ndarray vector in path-row order.
    #[must_use]
    pub fn potential_indices(&self) -> Array1<i32> {
        Array1::from_iter(self.atoms.iter().map(|atom| atom.potential_index))
    }
}

/// FEFF `paths.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct PathsDatData {
    /// Header/title lines before FEFF's dashed separator.
    pub titles: Vec<String>,
    /// Path records.
    pub paths: Vec<PathsDatPath>,
}

/// Render FEFF `paths.dat` text.
pub fn paths_dat_string(data: &PathsDatData) -> Result<String> {
    validate_paths_dat(data)?;

    let mut out = String::new();
    for title in &data.titles {
        writeln!(out, "{title}")?;
    }
    writeln!(out, " {}", "-".repeat(71))?;
    for path in &data.paths {
        writeln!(
            out,
            " {:5}{:5}{:8.3}  index, nleg, degeneracy, r={:8.4}",
            path.index,
            path.leg_count(),
            path.degeneracy,
            path.effective_half_path_length_angstrom
        )?;
        if path.row_header.trim().is_empty() {
            writeln!(
                out,
                "      x           y           z     ipot  label      rleg      beta        eta"
            )?;
        } else {
            writeln!(out, "{}", path.row_header)?;
        }
        for atom in &path.atoms {
            write!(
                out,
                "{:12.6}{:12.6}{:12.6}{:4} '{}'",
                atom.position_angstrom[0],
                atom.position_angstrom[1],
                atom.position_angstrom[2],
                atom.potential_index,
                fixed_label(&atom.label)
            )?;
            match (
                atom.leg_distance_angstrom,
                atom.beta_degrees,
                atom.eta_degrees,
            ) {
                (Some(rleg), Some(beta), Some(eta)) => {
                    writeln!(out, " {:10.4}{:10.4}{:10.4}", rleg, beta, eta)?;
                }
                _ => {
                    writeln!(out)?;
                }
            }
        }
    }
    Ok(out)
}

/// Parse FEFF `paths.dat` text.
pub fn parse_paths_dat(text: &str) -> Result<PathsDatData> {
    let mut lines = text.lines().enumerate().peekable();
    let mut titles = Vec::new();
    loop {
        let Some((_, text)) = lines.next() else {
            return Err(IoError::PathsDatMissing { field: "separator" });
        };
        if is_separator(text) {
            break;
        }
        titles.push(text.to_string());
    }

    let mut paths = Vec::new();
    while let Some((line, text)) = next_nonempty(&mut lines) {
        let header = parse_path_header(line, text)?;
        let (row_header_line, row_header) = lines
            .next()
            .map(|(line, text)| (line + 1, text))
            .ok_or(IoError::PathsDatMissing {
                field: "row header",
            })?;
        if row_header.trim().is_empty() {
            return Err(invalid_paths_dat(
                "row header",
                format!("line {row_header_line} is empty"),
            ));
        }

        let mut atoms = Vec::with_capacity(header.leg_count);
        for _ in 0..header.leg_count {
            let (atom_line, atom_text) = lines
                .next()
                .map(|(line, text)| (line + 1, text))
                .ok_or(IoError::PathsDatMissing { field: "atom row" })?;
            atoms.push(parse_atom_row(atom_line, atom_text)?);
        }
        paths.push(PathsDatPath {
            index: header.index,
            degeneracy: header.degeneracy,
            effective_half_path_length_angstrom: header.effective_half_path_length_angstrom,
            row_header: row_header.to_string(),
            atoms,
        });
    }

    let data = PathsDatData { titles, paths };
    validate_paths_dat(&data)?;
    Ok(data)
}

/// Write FEFF `paths.dat` text to a file.
pub fn write_paths_dat(path: impl AsRef<Path>, data: &PathsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, paths_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `paths.dat` text from a file.
pub fn read_paths_dat(path: impl AsRef<Path>) -> Result<PathsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_paths_dat(&text)
}

#[derive(Debug, Clone, Copy)]
struct PathHeader {
    index: usize,
    leg_count: usize,
    degeneracy: f64,
    effective_half_path_length_angstrom: f64,
}

fn validate_paths_dat(data: &PathsDatData) -> Result<()> {
    for path in &data.paths {
        if path.leg_count() == 0 {
            return Err(invalid_paths_dat(
                "nleg",
                "path must contain at least one atom",
            ));
        }
        if !path.degeneracy.is_finite() {
            return Err(invalid_paths_dat("degeneracy", "value must be finite"));
        }
        if !path.effective_half_path_length_angstrom.is_finite() {
            return Err(invalid_paths_dat("r", "value must be finite"));
        }
        for atom in &path.atoms {
            if atom.label.trim().is_empty() {
                return Err(invalid_paths_dat("label", "label must not be empty"));
            }
            let optional_geometry = [
                atom.leg_distance_angstrom,
                atom.beta_degrees,
                atom.eta_degrees,
            ];
            let optional_geometry_count = optional_geometry
                .iter()
                .filter(|value| value.is_some())
                .count();
            if !matches!(optional_geometry_count, 0 | 3) {
                return Err(invalid_paths_dat(
                    "atom row",
                    "rleg, beta, and eta must be all present or all absent",
                ));
            }
            for value in atom.position_angstrom {
                if !value.is_finite() {
                    return Err(invalid_paths_dat("position", "all values must be finite"));
                }
            }
            for value in optional_geometry.into_iter().flatten() {
                if !value.is_finite() {
                    return Err(invalid_paths_dat("atom row", "all values must be finite"));
                }
            }
        }
    }
    Ok(())
}

fn is_separator(line: &str) -> bool {
    let trimmed = line.trim();
    line.get(3..11).is_some_and(|text| text == "--------")
        || (!trimmed.is_empty() && trimmed.chars().all(|ch| ch == '-'))
}

fn next_nonempty<'a>(
    lines: &mut std::iter::Peekable<std::iter::Enumerate<std::str::Lines<'a>>>,
) -> Option<(usize, &'a str)> {
    for (line, text) in lines.by_ref() {
        if !text.trim().is_empty() {
            return Some((line + 1, text));
        }
    }
    None
}

fn parse_path_header(line: usize, text: &str) -> Result<PathHeader> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 3 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 3,
        });
    }
    let effective_half_path_length_angstrom = parse_r_value(line, text)?;
    Ok(PathHeader {
        index: parse_usize(line, "index", tokens[0])?,
        leg_count: parse_usize(line, "nleg", tokens[1])?,
        degeneracy: parse_f64(line, "degeneracy", tokens[2])?,
        effective_half_path_length_angstrom,
    })
}

fn parse_r_value(line: usize, text: &str) -> Result<f64> {
    let Some((_, value)) = text.split_once("r=") else {
        return Err(IoError::PathsDatMissing { field: "r" });
    };
    let token = value
        .split_whitespace()
        .next()
        .ok_or(IoError::PathsDatMissing { field: "r" })?;
    parse_f64(line, "r", token)
}

fn parse_atom_row(line: usize, text: &str) -> Result<PathsDatAtom> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 4 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 4,
        });
    }
    let after_numbers = after_tokens(text, 4).ok_or(IoError::PathsDatMissing { field: "label" })?;
    let (label, tail) = quoted_label(line, after_numbers)?;
    let tail_tokens = tail.split_whitespace().collect::<Vec<_>>();
    let trailing = parse_optional_atom_tail(line, &tail_tokens)?;
    Ok(PathsDatAtom {
        position_angstrom: [
            parse_f64(line, "x", tokens[0])?,
            parse_f64(line, "y", tokens[1])?,
            parse_f64(line, "z", tokens[2])?,
        ],
        potential_index: parse_i32(line, "ipot", tokens[3])?,
        label: label.trim().to_string(),
        leg_distance_angstrom: trailing.map(|values| values.0),
        beta_degrees: trailing.map(|values| values.1),
        eta_degrees: trailing.map(|values| values.2),
    })
}

fn after_tokens(text: &str, count: usize) -> Option<&str> {
    let mut offset = 0_usize;
    let mut seen = 0_usize;
    for token in text.split_whitespace() {
        let relative = text.get(offset..)?.find(token)?;
        offset += relative + token.len();
        seen += 1;
        if seen == count {
            return text.get(offset..);
        }
    }
    None
}

fn quoted_label(line: usize, text: &str) -> Result<(&str, &str)> {
    let start = text
        .find('\'')
        .ok_or(IoError::PathsDatMissing { field: "label" })?;
    let rest = &text[start + 1..];
    let end = rest
        .find('\'')
        .ok_or_else(|| invalid_paths_dat("label", format!("line {line} has unterminated quote")))?;
    Ok((&rest[..end], &rest[end + 1..]))
}

fn parse_optional_atom_tail(line: usize, tokens: &[&str]) -> Result<Option<(f64, f64, f64)>> {
    let Some(first) = tokens.first() else {
        return Ok(None);
    };
    if first.parse::<f64>().is_err() {
        return Ok(None);
    }
    if tokens.len() < 3 {
        return Err(IoError::PathsDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 3,
        });
    }
    Ok(Some((
        parse_f64(line, "rleg", tokens[0])?,
        parse_f64(line, "beta", tokens[1])?,
        parse_f64(line, "eta", tokens[2])?,
    )))
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token.parse::<i32>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::PathsDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn fixed_label(label: &str) -> String {
    format!("{:<6}", label).chars().take(6).collect()
}

fn invalid_paths_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidPathsDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_pathsd_output() -> Result<()> {
        let parsed = parse_paths_dat(FULL_PATHS_DAT)?;
        assert_eq!(parsed.titles, vec!["TITLE Cu crystal"]);
        assert_eq!(parsed.paths.len(), 1);
        let path = &parsed.paths[0];
        assert_eq!(path.index, 1);
        assert_eq!(path.leg_count(), 3);
        assert_eq!(path.potential_indices().to_vec(), vec![1, 1, 0]);
        assert_eq!(path.atoms[0].label, "Cu1");
        assert_eq!(path.atoms[0].leg_distance_angstrom, Some(2.5527));
        Ok(())
    }

    #[test]
    fn parses_single_scattering_rdinp_output() -> Result<()> {
        let parsed = parse_paths_dat(SINGLE_SCATTERING_PATHS_DAT)?;
        assert_eq!(parsed.paths.len(), 1);
        let path = &parsed.paths[0];
        assert_eq!(path.index, 7);
        assert_eq!(path.leg_count(), 2);
        assert_eq!(path.atoms[0].leg_distance_angstrom, None);
        assert_eq!(path.atoms[1].potential_index, 0);
        Ok(())
    }

    #[test]
    fn roundtrips_paths_dat_text() -> Result<()> {
        let data = sample_paths_dat();
        let parsed = parse_paths_dat(&paths_dat_string(&data)?)?;
        assert_eq!(parsed.titles, data.titles);
        assert_eq!(parsed.paths.len(), data.paths.len());
        assert_eq!(parsed.paths[0].index, data.paths[0].index);
        assert_eq!(parsed.paths[0].atoms.len(), data.paths[0].atoms.len());
        Ok(())
    }

    #[test]
    fn rejects_bad_rows() {
        assert!(matches!(
            parse_paths_dat("bad\n"),
            Err(IoError::PathsDatMissing { field: "separator" })
        ));
        assert!(matches!(
            parse_paths_dat("\nTITLE Cu\n"),
            Err(IoError::PathsDatMissing { field: "separator" })
        ));
        assert!(matches!(
            parse_paths_dat(" ---\n  a 2 3 r= 4\n"),
            Err(IoError::PathsDatParse { field: "index", .. })
        ));
        assert!(matches!(
            parse_paths_dat(MALFORMED_TRAILING_PATHS_DAT),
            Err(IoError::PathsDatRowWidth {
                line: 5,
                expected: 3,
                ..
            })
        ));
        let mut partial_optional = sample_paths_dat();
        partial_optional.paths[0].atoms[0].beta_degrees = None;
        assert!(matches!(
            paths_dat_string(&partial_optional),
            Err(IoError::InvalidPathsDat {
                field: "atom row",
                ..
            })
        ));
    }

    fn sample_paths_dat() -> PathsDatData {
        PathsDatData {
            titles: vec!["TITLE Cu crystal".to_string()],
            paths: vec![PathsDatPath {
                index: 1,
                degeneracy: 4.0,
                effective_half_path_length_angstrom: 2.5527,
                row_header:
                    "      x           y           z     ipot  label      rleg      beta        eta"
                        .to_string(),
                atoms: vec![
                    PathsDatAtom {
                        position_angstrom: [1.805, 1.805, 0.0],
                        potential_index: 1,
                        label: "Cu1".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(90.0),
                        eta_degrees: Some(45.0),
                    },
                    PathsDatAtom {
                        position_angstrom: [0.0, 0.0, 0.0],
                        potential_index: 0,
                        label: "Cu0".to_string(),
                        leg_distance_angstrom: Some(2.5527),
                        beta_degrees: Some(90.0),
                        eta_degrees: Some(225.0),
                    },
                ],
            }],
        }
    }

    const FULL_PATHS_DAT: &str = "\
TITLE Cu crystal
 -----------------------------------------------------------------------
     1    3   4.000  index, nleg, degeneracy, r=  2.5527
      x           y           z     ipot  label      rleg      beta        eta
    1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000   45.0000
   -1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000  135.0000
    0.000000    0.000000    0.000000   0 'Cu0   '     2.5527   90.0000  225.0000
";

    const SINGLE_SCATTERING_PATHS_DAT: &str = "\
 TITLE Cu crystal
 Single scattering paths from ss lines cards in feff input
 -----------------------------------------------------------------------
   7   2   1.000  index,nleg,degeneracy,r=  2.0000
 single scattering
    2.000000    0.000000    0.000000   1 'Cu1   '
    0.000000    0.000000    0.000000   0 'Cu0   '  x,y,z,ipot
";

    const MALFORMED_TRAILING_PATHS_DAT: &str = "\
TITLE Cu crystal
 -----------------------------------------------------------------------
     1    1   4.000  index, nleg, degeneracy, r=  2.5527
      x           y           z     ipot  label      rleg      beta        eta
    1.805000    1.805000    0.000000   1 'Cu1   '     2.5527   90.0000
";
}
