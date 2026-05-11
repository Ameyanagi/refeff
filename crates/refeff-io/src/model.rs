use std::path::PathBuf;

use crate::error::{IoError, Result};
use crate::input::{FeffInput, FeffLine, LineKind};

#[derive(Debug, Clone, PartialEq)]
pub struct FeffDocument {
    pub source: PathBuf,
    pub titles: Vec<String>,
    pub edge: Option<Edge>,
    pub s02: Option<f64>,
    pub control: Option<[i32; 6]>,
    pub print: Option<[i32; 6]>,
    pub potentials: Vec<Potential>,
    pub atoms: Vec<Atom>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Potential {
    pub ipot: i32,
    pub z: i32,
    pub tag: Option<String>,
    pub lmax1: Option<i32>,
    pub lmax2: Option<i32>,
    pub xnatph: Option<f64>,
    pub spinph: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub ipot: i32,
    pub tag: Option<String>,
    pub distance: Option<f64>,
    pub index: Option<usize>,
}

impl FeffDocument {
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
                z: parse_i32(line, &fields[1])?,
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
                distance: parse_optional_f64(line, fields.get(5))?,
                index: parse_optional_usize(line, fields.get(6))?,
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

fn parse_optional_usize(line: &FeffLine, value: Option<&String>) -> Result<Option<usize>> {
    value
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| parse_error(line, format!("invalid unsigned integer {value:?}")))
        })
        .transpose()
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
