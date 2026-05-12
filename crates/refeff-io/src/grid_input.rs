//! Typed reader/writer for FEFF `grid.inp` energy-grid handoff files.
//!
//! `XSPH/rdgrid.f90` reads regular energy, momentum, exponential, and
//! user-specified grid records from this file. RDINP writes `grid.inp` by
//! copying the lines that follow an `EGRID` card without arguments.

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

/// Parsed FEFF `grid.inp` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct GridInput {
    /// Grid records in file order.
    pub records: Vec<GridRecord>,
}

impl GridInput {
    /// Parse FEFF `grid.inp` text.
    pub fn parse_str(text: &str) -> Result<Self> {
        parse_grid_inp(text)
    }
}

/// One `grid.inp` record.
#[derive(Debug, Clone, PartialEq)]
pub enum GridRecord {
    /// Regular generated grid.
    Regular(GridRegularRecord),
    /// User-specified energy points following a `user_grid` marker.
    User(GridUserRecord),
}

/// Regular generated grid record.
#[derive(Debug, Clone, PartialEq)]
pub struct GridRegularRecord {
    /// Grid generator kind.
    pub kind: GridKind,
    /// Minimum grid value or FEFF's `last` continuation marker.
    pub minimum: GridMinimum,
    /// Maximum grid value.
    pub maximum: f64,
    /// Grid step.
    pub step: f64,
}

/// FEFF regular grid kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridKind {
    /// `e_grid`: regular in energy.
    Energy,
    /// `k_grid`: regular in wave number.
    WaveNumber,
    /// `exp_grid`: exponential in energy.
    Exponential,
}

impl GridKind {
    /// FEFF keyword for this grid kind.
    #[must_use]
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Energy => "e_grid",
            Self::WaveNumber => "k_grid",
            Self::Exponential => "exp_grid",
        }
    }
}

/// Regular grid minimum field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridMinimum {
    /// Explicit minimum value.
    Value(f64),
    /// FEFF `last` marker; XSPH starts from the previous grid end.
    Last,
}

/// FEFF `user_grid` record.
#[derive(Debug, Clone, PartialEq)]
pub struct GridUserRecord {
    /// User-specified complex energy points in file order.
    pub points: Vec<GridPoint>,
}

/// One `user_grid` energy point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPoint {
    /// Real part in eV before XSPH converts to Hartree.
    pub real: f64,
    /// Imaginary part in eV before XSPH converts to Hartree.
    pub imaginary: f64,
}

/// Render FEFF `grid.inp` text.
pub fn grid_inp_string(input: &GridInput) -> Result<String> {
    validate_grid_input(input)?;

    let mut out = String::new();
    for record in &input.records {
        match record {
            GridRecord::Regular(record) => {
                write!(out, " {}", record.kind.keyword())?;
                match record.minimum {
                    GridMinimum::Value(value) => write!(out, " {value}")?,
                    GridMinimum::Last => write!(out, " last")?,
                }
                writeln!(out, " {} {} ", record.maximum, record.step)?;
            }
            GridRecord::User(record) => {
                writeln!(out, " user_grid")?;
                for point in &record.points {
                    if point.imaginary == 0.0 {
                        writeln!(out, " {}", point.real)?;
                    } else {
                        writeln!(out, " {} {}", point.real, point.imaginary)?;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Parse FEFF `grid.inp` text.
pub fn parse_grid_inp(text: &str) -> Result<GridInput> {
    let logical_lines = grid_lines(text);
    let mut records = Vec::new();
    let mut index = 0_usize;

    while let Some(line) = logical_lines.get(index) {
        let tokens = split_grid_line(line.text);
        if tokens.is_empty() {
            index += 1;
            continue;
        }
        let keyword = tokens[0].to_ascii_lowercase();
        if keyword == "user_grid" {
            let mut points = Vec::new();
            index += 1;
            while let Some(point_line) = logical_lines.get(index) {
                let point_tokens = split_grid_line(point_line.text);
                if point_tokens.is_empty() {
                    index += 1;
                    continue;
                }
                if !is_numeric_token(point_tokens[0]) {
                    break;
                }
                points.push(parse_grid_point(point_line.line, &point_tokens)?);
                index += 1;
            }
            records.push(GridRecord::User(GridUserRecord { points }));
            continue;
        }

        let kind = parse_grid_kind(line.line, &keyword)?;
        if tokens.len() < 4 {
            return Err(IoError::GridInpRowWidth {
                line: line.line,
                actual: tokens.len(),
                expected: 4,
            });
        }
        let minimum = if tokens[1].eq_ignore_ascii_case("last") {
            GridMinimum::Last
        } else {
            GridMinimum::Value(parse_f64(line.line, "minimum", tokens[1])?)
        };
        records.push(GridRecord::Regular(GridRegularRecord {
            kind,
            minimum,
            maximum: parse_f64(line.line, "maximum", tokens[2])?,
            step: parse_f64(line.line, "step", tokens[3])?,
        }));
        index += 1;
    }

    let input = GridInput { records };
    validate_grid_input(&input)?;
    Ok(input)
}

/// Write FEFF `grid.inp` text to a file.
pub fn write_grid_inp(path: impl AsRef<Path>, input: &GridInput) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, grid_inp_string(input)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `grid.inp` text from a file.
pub fn read_grid_inp(path: impl AsRef<Path>) -> Result<GridInput> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_grid_inp(&text)
}

fn grid_lines(text: &str) -> Vec<GridLine<'_>> {
    text.lines()
        .enumerate()
        .filter_map(|(index, raw)| {
            let line = strip_inline_comment(raw).trim();
            if line.is_empty() || is_comment_line(line) {
                None
            } else {
                Some(GridLine {
                    line: index + 1,
                    text: line,
                })
            }
        })
        .collect()
}

fn split_grid_line(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn strip_inline_comment(line: &str) -> &str {
    let comment_index = line
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '#' | '!' | '%').then_some(index));
    comment_index.map_or(line, |index| &line[..index])
}

fn is_comment_line(line: &str) -> bool {
    line.chars()
        .next()
        .is_some_and(|ch| matches!(ch, '#' | '!' | '*' | 'C' | 'c'))
}

fn parse_grid_kind(line: usize, keyword: &str) -> Result<GridKind> {
    match keyword {
        "e_grid" => Ok(GridKind::Energy),
        "k_grid" => Ok(GridKind::WaveNumber),
        "exp_grid" => Ok(GridKind::Exponential),
        _ => Err(IoError::GridInpParse {
            field: "grid kind",
            line,
            token: keyword.to_string(),
        }),
    }
}

fn parse_grid_point(line: usize, tokens: &[&str]) -> Result<GridPoint> {
    if tokens.is_empty() {
        return Err(IoError::GridInpMissing {
            field: "user grid point",
        });
    }
    Ok(GridPoint {
        real: parse_f64(line, "user grid real energy", tokens[0])?,
        imaginary: tokens.get(1).map_or(Ok(0.0), |token| {
            parse_f64(line, "user grid imaginary energy", token)
        })?,
    })
}

fn is_numeric_token(token: &str) -> bool {
    token.parse::<f64>().is_ok()
}

fn validate_grid_input(input: &GridInput) -> Result<()> {
    if input.records.is_empty() {
        return Err(invalid_grid_inp(
            "records",
            "at least one grid record is required",
        ));
    }

    for record in &input.records {
        match record {
            GridRecord::Regular(record) => {
                if let GridMinimum::Value(minimum) = record.minimum
                    && !minimum.is_finite()
                {
                    return Err(invalid_grid_inp("minimum", "value must be finite"));
                }
                if !record.maximum.is_finite() {
                    return Err(invalid_grid_inp("maximum", "value must be finite"));
                }
                if !record.step.is_finite() || record.step <= 0.0 {
                    return Err(invalid_grid_inp(
                        "step",
                        "value must be finite and positive",
                    ));
                }
            }
            GridRecord::User(record) => {
                if record.points.is_empty() {
                    return Err(invalid_grid_inp(
                        "user_grid",
                        "record must contain at least one point",
                    ));
                }
                for point in &record.points {
                    if !point.real.is_finite() || !point.imaginary.is_finite() {
                        return Err(invalid_grid_inp("user grid point", "values must be finite"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.parse::<f64>().map_err(|_| IoError::GridInpParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn invalid_grid_inp(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidGridInp {
        field,
        message: message.into(),
    }
}

#[derive(Debug, Clone, Copy)]
struct GridLine<'a> {
    line: usize,
    text: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_generated_grid_records() -> Result<()> {
        let parsed = parse_grid_inp(GRID_INP)?;
        assert_eq!(parsed.records.len(), 3);
        assert_eq!(
            parsed.records[0],
            GridRecord::Regular(GridRegularRecord {
                kind: GridKind::Energy,
                minimum: GridMinimum::Value(-15.0),
                maximum: -1.0,
                step: 1.0,
            })
        );
        assert_eq!(
            parsed.records[1],
            GridRecord::Regular(GridRegularRecord {
                kind: GridKind::Energy,
                minimum: GridMinimum::Last,
                maximum: 10.0,
                step: 0.1,
            })
        );
        assert_eq!(
            parsed.records[2],
            GridRecord::Regular(GridRegularRecord {
                kind: GridKind::WaveNumber,
                minimum: GridMinimum::Last,
                maximum: 5.0,
                step: 0.05,
            })
        );
        Ok(())
    }

    #[test]
    fn parses_user_grid_points() -> Result<()> {
        let parsed = parse_grid_inp(USER_GRID_INP)?;
        assert_eq!(
            parsed.records,
            vec![
                GridRecord::User(GridUserRecord {
                    points: vec![
                        GridPoint {
                            real: -1.0,
                            imaginary: 0.0,
                        },
                        GridPoint {
                            real: 2.0,
                            imaginary: 0.25,
                        },
                    ],
                }),
                GridRecord::Regular(GridRegularRecord {
                    kind: GridKind::Energy,
                    minimum: GridMinimum::Last,
                    maximum: 5.0,
                    step: 0.5,
                }),
            ]
        );
        Ok(())
    }

    #[test]
    fn roundtrips_grid_text() -> Result<()> {
        let parsed = parse_grid_inp(GRID_INP)?;
        let reparsed = parse_grid_inp(&grid_inp_string(&parsed)?)?;
        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_grid_records() {
        assert!(matches!(
            parse_grid_inp("e_grid 1 2\n"),
            Err(IoError::GridInpRowWidth {
                line: 1,
                expected: 4,
                ..
            })
        ));
        assert!(matches!(
            parse_grid_inp("bad_grid 1 2 3\n"),
            Err(IoError::GridInpParse {
                field: "grid kind",
                ..
            })
        ));
        assert!(matches!(
            grid_inp_string(&GridInput { records: vec![] }),
            Err(IoError::InvalidGridInp {
                field: "records",
                ..
            })
        ));
    }

    const GRID_INP: &str = "\
 e_grid -15 -1.0 1.0
 e_grid last 10.0 0.1
 k_grid last 5.0 0.05
";

    const USER_GRID_INP: &str = "\
* comment
 user_grid
 -1.0
 2.0 0.25
 e_grid last 5.0 0.5
";
}
