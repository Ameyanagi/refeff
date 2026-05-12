//! Typed reader for FEFF `screen.inp` module handoff files.
//!
//! `screen.inp` stores the energy and radial-grid controls for the screening
//! module as keyword/value rows.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `screen.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenInput {
    pub ner: i32,
    pub nei: i32,
    pub maxl: i32,
    pub irrh: i32,
    pub iend: i32,
    pub lfxc: i32,
    pub emin: f64,
    pub emax: f64,
    pub eimax: f64,
    pub ermin: f64,
    pub rfms: f64,
    pub nrptx0: i32,
    pub icore: i32,
}

impl ScreenInput {
    /// Parse a FEFF `screen.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ScreenInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct ScreenInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> ScreenInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<ScreenInput> {
        Ok(ScreenInput {
            ner: self.parse_keyed_value("ner")?,
            nei: self.parse_keyed_value("nei")?,
            maxl: self.parse_keyed_value("maxl")?,
            irrh: self.parse_keyed_value("irrh")?,
            iend: self.parse_keyed_value("iend")?,
            lfxc: self.parse_keyed_value("lfxc")?,
            emin: self.parse_keyed_value("emin")?,
            emax: self.parse_keyed_value("emax")?,
            eimax: self.parse_keyed_value("eimax")?,
            ermin: self.parse_keyed_value("ermin")?,
            rfms: self.parse_keyed_value("rfms")?,
            nrptx0: self.parse_keyed_value("nrptx0")?,
            icore: self.parse_keyed_value("icore")?,
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
    use crate::rdinp;

    use super::ScreenInput;

    #[test]
    fn parses_default_screen_input() -> crate::Result<()> {
        let screen = ScreenInput::parse_str("screen.inp", &rdinp::screen_inp_string())?;
        assert_eq!(screen.ner, 40);
        assert_eq!(screen.nei, 20);
        assert_eq!(screen.maxl, 4);
        assert_eq!(screen.irrh, 1);
        assert_eq!(screen.iend, 0);
        assert_eq!(screen.lfxc, 0);
        assert_eq!(screen.emin, -40.0);
        assert_eq!(screen.emax, 0.0);
        assert_eq!(screen.eimax, 2.0);
        assert_eq!(screen.ermin, 0.001);
        assert_eq!(screen.rfms, 4.0);
        assert_eq!(screen.nrptx0, 251);
        assert_eq!(screen.icore, -1);
        Ok(())
    }
}
