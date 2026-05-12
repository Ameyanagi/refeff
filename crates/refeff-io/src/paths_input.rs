//! Typed reader for FEFF `paths.inp` module handoff files.
//!
//! `paths.inp` controls path expansion and path-filtering thresholds. The Rust
//! port keeps these settings typed so later path generation can be implemented
//! without re-parsing fixed FEFF text.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `paths.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathsInput {
    /// Integer path-generation controls.
    pub control: PathsControl,
    /// Path-filtering criteria and radii.
    pub criteria: PathsCriteria,
    /// NRIXS path-expansion mode selector.
    pub ica: i32,
}

/// First integer control line of `paths.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathsControl {
    pub mpath: i32,
    pub ms: i32,
    pub nncrit: i32,
    pub nlegxx: i32,
    pub ipr4: i32,
}

/// Path-filtering criteria from `paths.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathsCriteria {
    pub critpw: f64,
    pub pcritk: f64,
    pub pcrith: f64,
    pub rmax: f64,
    pub rfms2: f64,
}

impl PathsInput {
    /// Parse a FEFF `paths.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = PathsInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct PathsInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> PathsInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<PathsInput> {
        self.expect_header("mpath, ms, nncrit, nlegxx, ipr4")?;
        let control_values = self.parse_values::<i32>(5, "PATHS control line")?;
        let control = PathsControl {
            mpath: control_values[0],
            ms: control_values[1],
            nncrit: control_values[2],
            nlegxx: control_values[3],
            ipr4: control_values[4],
        };

        self.expect_header("critpw, pcritk, pcrith,  rmax, rfms2")?;
        let criteria_values = self.parse_values::<f64>(5, "PATHS criteria line")?;
        let criteria = PathsCriteria {
            critpw: criteria_values[0],
            pcritk: criteria_values[1],
            pcrith: criteria_values[2],
            rmax: criteria_values[3],
            rfms2: criteria_values[4],
        };

        self.expect_header("ica")?;
        let ica = self.parse_values::<i32>(1, "PATHS ica line")?[0];

        Ok(PathsInput {
            control,
            criteria,
            ica,
        })
    }

    fn expect_header(&mut self, expected: &str) -> Result<()> {
        let (line_number, line) = self.next_line(expected)?;
        if line.trim() == expected {
            Ok(())
        } else {
            Err(self.parse_error(
                line_number,
                format!("expected header {expected:?}, found {line:?}"),
            ))
        }
    }

    fn parse_values<T>(&mut self, count: usize, description: &str) -> Result<Vec<T>>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(description)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < count {
            return Err(self.parse_error(
                line_number,
                format!("{description} requires {count} fields"),
            ));
        }
        fields
            .iter()
            .take(count)
            .map(|field| parse_field(&self.source, line_number, field))
            .collect()
    }

    fn next_line(&mut self, description: &str) -> Result<(usize, &'a str)> {
        self.lines
            .next()
            .map(|(index, line)| (index + 1, line))
            .ok_or_else(|| self.parse_error(0, format!("expected {description}")))
    }

    fn parse_error(&self, line: usize, message: impl Into<String>) -> IoError {
        IoError::Parse {
            path: self.source.clone(),
            line,
            message: message.into(),
        }
    }
}

fn parse_field<T>(source: &Path, line: usize, field: &str) -> Result<T>
where
    T: FromStr,
{
    field.parse::<T>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("invalid numeric field {field:?}"),
    })
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::PathsInput;

    #[test]
    fn parses_generated_paths_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 2 0 0
CRITERIA 3.1 4.2
PCRITERIA 0.5 0.6
RPATH 5.5
NLEG 6
FMS 4.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::paths_inp_string(&document)?;
        let paths = PathsInput::parse_str("paths.inp", &text)?;

        assert_eq!(paths.control.mpath, 1);
        assert_eq!(paths.control.ms, 1);
        assert_eq!(paths.control.nncrit, 0);
        assert_eq!(paths.control.nlegxx, 6);
        assert_eq!(paths.control.ipr4, 2);
        assert_eq!(paths.criteria.critpw, 4.2);
        assert_eq!(paths.criteria.pcritk, 0.5);
        assert_eq!(paths.criteria.pcrith, 0.6);
        assert_eq!(paths.criteria.rmax, 5.5);
        assert_eq!(paths.criteria.rfms2, 4.0);
        assert_eq!(paths.ica, -1);
        Ok(())
    }
}
