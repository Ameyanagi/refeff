//! Typed reader for FEFF `genfmt.inp` module handoff files.
//!
//! GENFMT consumes path-formatting controls and the NRIXS decomposition count.
//! This reader keeps those settings explicit for the Rust path-format stage.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `genfmt.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtInput {
    /// GENFMT run-control fields.
    pub control: GenfmtControl,
    /// NRIXS decomposition-channel count.
    pub decomposition_channels: i32,
}

/// First control line of `genfmt.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenfmtControl {
    pub mfeff: i32,
    pub ipr5: i32,
    pub iorder: i32,
    pub critcw: f64,
    pub wnstar: bool,
}

impl GenfmtInput {
    /// Parse a FEFF `genfmt.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = GenfmtInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct GenfmtInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> GenfmtInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<GenfmtInput> {
        self.expect_header("mfeff, ipr5, iorder, critcw, wnstar")?;
        let (line_number, line) = self.next_line("GENFMT control line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            return Err(self.parse_error(line_number, "GENFMT control line requires 5 fields"));
        }
        let control = GenfmtControl {
            mfeff: parse_field(&self.source, line_number, fields[0])?,
            ipr5: parse_field(&self.source, line_number, fields[1])?,
            iorder: parse_field(&self.source, line_number, fields[2])?,
            critcw: parse_field(&self.source, line_number, fields[3])?,
            wnstar: parse_fortran_bool(&self.source, line_number, fields[4])?,
        };

        self.expect_header("the number of decomposi")?;
        let decomposition_channels = self.parse_values::<i32>(1, "GENFMT decomposition line")?[0];

        Ok(GenfmtInput {
            control,
            decomposition_channels,
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

fn parse_fortran_bool(source: &Path, line: usize, field: &str) -> Result<bool> {
    match field.trim().to_ascii_uppercase().as_str() {
        "T" | ".TRUE." | "TRUE" => Ok(true),
        "F" | ".FALSE." | "FALSE" => Ok(false),
        value => Err(IoError::Parse {
            path: source.to_path_buf(),
            line,
            message: format!("invalid FEFF bool {value:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::GenfmtInput;

    #[test]
    fn parses_generated_genfmt_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 3 0
CRITERIA 2.5 4.4
NRIXS 1 1.0 2.0 -3.0
LDEC 5
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::genfmt_inp_string(&document)?;
        let genfmt = GenfmtInput::parse_str("genfmt.inp", &text)?;

        assert_eq!(genfmt.control.mfeff, 1);
        assert_eq!(genfmt.control.ipr5, 3);
        assert_eq!(genfmt.control.iorder, 2);
        assert_eq!(genfmt.control.critcw, 2.5);
        assert!(!genfmt.control.wnstar);
        assert_eq!(genfmt.decomposition_channels, 5);
        Ok(())
    }
}
