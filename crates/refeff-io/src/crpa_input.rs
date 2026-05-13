//! Typed reader for FEFF `crpa.inp` module handoff files.
//!
//! CRPA input is written as keyword/value rows for the module switch, cutoff
//! radius, and angular momentum channel.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `crpa.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrpaInput {
    /// Whether the CRPA module should run.
    pub enabled: bool,
    /// Real-space cutoff radius.
    pub rcut: f64,
    /// Angular momentum channel.
    pub l: i32,
}

impl CrpaInput {
    /// Parse a FEFF `crpa.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = CrpaInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `crpa.inp` text.
pub fn crpa_input_string(input: &CrpaInput) -> Result<String> {
    if !input.rcut.is_finite() {
        return Err(IoError::Parse {
            path: "crpa.inp".into(),
            line: 0,
            message: "CRPA cutoff radius must be finite".to_string(),
        });
    }
    Ok(format!(
        " do_CRPA{:12}\n rcut{:21.16}     \n l_crpa{:12}\n",
        i32::from(input.enabled),
        input.rcut,
        input.l
    ))
}

struct CrpaInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> CrpaInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<CrpaInput> {
        Ok(CrpaInput {
            enabled: self.parse_keyed_value::<i32>("do_CRPA")? != 0,
            rcut: self.parse_keyed_value("rcut")?,
            l: self.parse_keyed_value("l_crpa")?,
        })
    }

    fn parse_keyed_value<T>(&mut self, expected_key: &str) -> Result<T>
    where
        T: FromStr,
    {
        let (line_number, line) = self.next_line(expected_key)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, format!("expected {expected_key} value row")));
        }
        if fields[0] != expected_key {
            return Err(self.parse_error(
                line_number,
                format!("expected key {expected_key:?}, found {:?}", fields[0]),
            ));
        }
        parse_field(&self.source, line_number, fields[1])
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

    use super::{CrpaInput, crpa_input_string};

    #[test]
    fn parses_generated_crpa_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CRPA 2 3.5
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let crpa = CrpaInput::parse_str("crpa.inp", &rdinp::crpa_inp_string(&document))?;

        assert!(crpa.enabled);
        assert_eq!(crpa.l, 2);
        assert_eq!(crpa.rcut, 3.5);
        Ok(())
    }

    #[test]
    fn renders_generated_crpa_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CRPA 2 3.5
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let crpa = CrpaInput::parse_str("crpa.inp", &rdinp::crpa_inp_string(&document))?;
        let rendered = crpa_input_string(&crpa)?;
        let reparsed = CrpaInput::parse_str("crpa.inp", &rendered)?;

        assert_eq!(rendered, rdinp::crpa_inp_string(&document));
        assert_eq!(reparsed, crpa);
        Ok(())
    }

    #[test]
    fn rejects_invalid_crpa_rendering() {
        let input = CrpaInput {
            enabled: true,
            rcut: f64::NAN,
            l: 2,
        };
        assert!(crpa_input_string(&input).is_err());
    }
}
