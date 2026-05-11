//! Typed extraction of common FEFF input cards.
//!
//! This layer intentionally starts with stable structural cards and grows as
//! each FEFF module is ported. Unknown or module-specific cards remain
//! available in [`crate::FeffInput`] so no information is lost.

use std::path::PathBuf;

use crate::error::{IoError, Result};
use crate::input::{FeffInput, FeffLine, LineKind};

/// FEFF input projected into typed structures used by the Rust modules.
#[derive(Debug, Clone, PartialEq)]
pub struct FeffDocument {
    /// Root input file.
    pub source: PathBuf,
    /// All `TITLE` lines in read order.
    pub titles: Vec<String>,
    /// Selected absorption edge, when present.
    pub edge: Option<Edge>,
    /// Amplitude reduction factor from `S02`, when present.
    pub s02: Option<f64>,
    /// Six execution switches from `CONTROL`, when present.
    pub control: Option<[i32; 6]>,
    /// Six print switches from the common `PRINT` card, when present.
    pub print: Option<[i32; 6]>,
    /// Rows from `POTENTIALS`/`POTENTIAL`.
    pub potentials: Vec<Potential>,
    /// Rows from `ATOMS`/`ATOM`.
    pub atoms: Vec<Atom>,
}

/// Absorption edge label, normalized to uppercase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub label: String,
}

/// One row of the FEFF `POTENTIALS` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    /// FEFF potential index.
    pub ipot: i32,
    /// Parsed atomic number when the field is numeric.
    pub z: Option<i32>,
    /// Original Z token, preserved for `HIGHZ` placeholders such as `XXX`.
    pub z_token: String,
    /// Element or user tag.
    pub tag: Option<String>,
    /// Optional phase-shift angular momentum limit.
    pub lmax1: Option<i32>,
    /// Optional FMS angular momentum limit.
    pub lmax2: Option<i32>,
    /// Optional stoichiometry/count field.
    pub xnatph: Option<f64>,
    /// Optional spin field.
    pub spinph: Option<f64>,
}

/// One row of the FEFF `ATOMS` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Cartesian x coordinate in Angstrom.
    pub x: f64,
    /// Cartesian y coordinate in Angstrom.
    pub y: f64,
    /// Cartesian z coordinate in Angstrom.
    pub z: f64,
    /// Potential index for this atom.
    pub ipot: i32,
    /// Optional atom tag.
    pub tag: Option<String>,
    /// Optional distance field from the input; generated if absent.
    pub distance: Option<f64>,
    /// Optional trailing index.
    pub index: Option<usize>,
}

impl FeffDocument {
    /// Extract the currently supported typed card subset from parsed input.
    pub fn from_input(input: &FeffInput) -> Result<Self> {
        let titles = parse_titles(input)?;
        let edge = parse_edge(input)?;
        let s02 = parse_scalar_card(input, "S02")?;
        let control = parse_i32_6(input, "CONTROL")?;
        let print = parse_i32_6(input, "PRINT")?;
        let potentials = parse_potentials(input)?;
        let atoms = parse_atoms(input)?;

        Ok(Self {
            source: input.source.clone(),
            titles,
            edge,
            s02,
            control,
            print,
            potentials,
            atoms,
        })
    }
}

fn parse_titles(input: &FeffInput) -> Result<Vec<String>> {
    let mut titles = Vec::new();
    for line in input.cards() {
        if let LineKind::Card {
            keyword, raw_args, ..
        } = &line.kind
            && keyword == "TITLE"
        {
            titles.push(raw_args.clone());
        }
    }
    Ok(titles)
}

fn parse_edge(input: &FeffInput) -> Result<Option<Edge>> {
    let Some(line) = input.card("EDGE") else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(label) = args.first() else {
        return Err(parse_error(line, "EDGE requires a label"));
    };
    Ok(Some(Edge {
        label: label.to_ascii_uppercase(),
    }))
}

fn parse_scalar_card(input: &FeffInput, keyword: &str) -> Result<Option<f64>> {
    let Some(line) = input.card(keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let Some(value) = args.first() else {
        return Err(parse_error(
            line,
            format!("{keyword} requires a numeric value"),
        ));
    };
    Ok(Some(parse_f64(line, value)?))
}

fn parse_i32_6(input: &FeffInput, keyword: &str) -> Result<Option<[i32; 6]>> {
    let Some(line) = input.card(keyword) else {
        return Ok(None);
    };
    let args = card_args(line)?;
    let mut values = [0_i32; 6];
    for (idx, slot) in values.iter_mut().enumerate() {
        let Some(value) = args.get(idx) else {
            return Err(parse_error(
                line,
                format!("{keyword} requires 6 integer values"),
            ));
        };
        *slot = parse_i32(line, value)?;
    }
    Ok(Some(values))
}

fn parse_potentials(input: &FeffInput) -> Result<Vec<Potential>> {
    input
        .section_rows("POTENTIALS")
        .map(|line| {
            let fields = section_fields(line)?;
            if fields.len() < 2 {
                return Err(parse_error(line, "POTENTIALS rows require ipot and Z"));
            }

            Ok(Potential {
                ipot: parse_i32(line, &fields[0])?,
                z: parse_i32(line, &fields[1]).ok(),
                z_token: fields[1].clone(),
                tag: fields.get(2).cloned(),
                lmax1: parse_optional_i32(line, fields.get(3))?,
                lmax2: parse_optional_i32(line, fields.get(4))?,
                xnatph: parse_optional_f64(line, fields.get(5))?,
                spinph: parse_optional_f64(line, fields.get(6))?,
            })
        })
        .collect()
}

fn parse_atoms(input: &FeffInput) -> Result<Vec<Atom>> {
    input
        .section_rows("ATOMS")
        .map(|line| {
            let fields = section_fields(line)?;
            if fields.len() < 4 {
                return Err(parse_error(line, "ATOMS rows require x y z ipot"));
            }

            Ok(Atom {
                x: parse_f64(line, &fields[0])?,
                y: parse_f64(line, &fields[1])?,
                z: parse_f64(line, &fields[2])?,
                ipot: parse_i32(line, &fields[3])?,
                tag: fields.get(4).cloned(),
                distance: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| parse_f64(line, value).ok()),
                index: fields
                    .iter()
                    .skip(5)
                    .find_map(|value| value.parse::<usize>().ok()),
            })
        })
        .collect()
}

fn card_args(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::Card { args, .. } => Ok(args),
        LineKind::SectionData { .. } => Err(parse_error(line, "expected card line")),
    }
}

fn section_fields(line: &FeffLine) -> Result<&[String]> {
    match &line.kind {
        LineKind::SectionData { fields, .. } => Ok(fields),
        LineKind::Card { .. } => Err(parse_error(line, "expected section data line")),
    }
}

fn parse_i32(line: &FeffLine, value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .map_err(|_| parse_error(line, format!("invalid integer {value:?}")))
}

fn parse_optional_i32(line: &FeffLine, value: Option<&String>) -> Result<Option<i32>> {
    value.map(|value| parse_i32(line, value)).transpose()
}

fn parse_f64(line: &FeffLine, value: &str) -> Result<f64> {
    value
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error(line, format!("invalid float {value:?}")))
}

fn parse_optional_f64(line: &FeffLine, value: Option<&String>) -> Result<Option<f64>> {
    value.map(|value| parse_f64(line, value)).transpose()
}

fn parse_error(line: &FeffLine, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: line.location.path.clone(),
        line: line.location.line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_common_structure_cards() {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
TITLE Cu crystal
EDGE K
S02 1.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.0 0.0 0.0 1 Cu1 1.0 1
END
"#,
        )
        .expect("parse");

        let doc = FeffDocument::from_input(&input).expect("document");
        assert_eq!(doc.titles, ["Cu crystal"]);
        assert_eq!(doc.edge.unwrap().label, "K");
        assert_eq!(doc.s02, Some(1.0));
        assert_eq!(doc.control, Some([1, 1, 1, 1, 1, 1]));
        assert_eq!(doc.potentials.len(), 2);
        assert_eq!(doc.atoms.len(), 2);
        assert_eq!(doc.atoms[1].tag.as_deref(), Some("Cu1"));
    }
}
