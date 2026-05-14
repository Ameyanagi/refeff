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
    /// Number of real-energy mesh points.
    pub ner: i32,
    /// Number of imaginary-energy mesh points.
    pub nei: i32,
    /// Maximum angular momentum used by the screening calculation.
    pub maxl: i32,
    /// Hedin-Lundqvist screening selector.
    pub irrh: i32,
    /// Endpoint selector for the energy mesh.
    pub iend: i32,
    /// Exchange-correlation selector.
    pub lfxc: i32,
    /// Lower real-energy bound in eV.
    pub emin: f64,
    /// Upper real-energy bound in eV.
    pub emax: f64,
    /// Upper imaginary-energy bound in eV.
    pub eimax: f64,
    /// Minimum imaginary-energy offset in eV.
    pub ermin: f64,
    /// Full multiple-scattering radius in Angstrom.
    pub rfms: f64,
    /// Radial grid size.
    pub nrptx0: i32,
    /// Core-state override.
    pub icore: i32,
}

impl Default for ScreenInput {
    fn default() -> Self {
        Self {
            ner: 40,
            nei: 20,
            maxl: 4,
            irrh: 1,
            iend: 0,
            lfxc: 0,
            emin: -40.0,
            emax: 0.0,
            eimax: 2.0,
            ermin: 0.001,
            rfms: 4.0,
            nrptx0: 251,
            icore: -1,
        }
    }
}

impl ScreenInput {
    /// Parse a FEFF `screen.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ScreenInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `screen.inp` text.
pub fn screen_input_string(input: &ScreenInput) -> Result<String> {
    validate_screen_input(input)?;

    Ok(format!(
        concat!(
            " ner{:12}\n",
            " nei{:12}\n",
            " maxl{:12}\n",
            " irrh{:12}\n",
            " iend{:12}\n",
            " lfxc{:12}\n",
            " emin{:21.15}     \n",
            " emax{:21.16}     \n",
            " eimax{:21.16}     \n",
            " ermin{}\n",
            " rfms{:21.16}     \n",
            " nrptx0{:12}\n",
            " icore{:12}\n",
        ),
        input.ner,
        input.nei,
        input.maxl,
        input.irrh,
        input.iend,
        input.lfxc,
        input.emin,
        input.emax,
        input.eimax,
        screen_fortran_exp(input.ermin, 26, 16),
        input.rfms,
        input.nrptx0,
        input.icore
    ))
}

fn validate_screen_input(input: &ScreenInput) -> Result<()> {
    validate_finite("emin", input.emin)?;
    validate_finite("emax", input.emax)?;
    validate_finite("eimax", input.eimax)?;
    validate_finite("ermin", input.ermin)?;
    validate_finite("rfms", input.rfms)
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "screen.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn screen_fortran_exp(value: f64, width: usize, precision: usize) -> String {
    let raw = format!("{value:.precision$E}");
    let Some((mantissa, exponent)) = raw.split_once('E') else {
        return format!("{raw:>width$}");
    };
    let (sign, digits) = match exponent.as_bytes().first() {
        Some(b'-') => ('-', &exponent[1..]),
        Some(b'+') => ('+', &exponent[1..]),
        _ => ('+', exponent),
    };
    let field = format!("{mantissa}E{sign}{digits:0>3}");
    format!("{field:>width$}")
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

    use super::{ScreenInput, screen_input_string};

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

    #[test]
    fn renders_default_screen_input() -> crate::Result<()> {
        let screen = ScreenInput::parse_str("screen.inp", &rdinp::screen_inp_string())?;
        let rendered = screen_input_string(&screen)?;
        let reparsed = ScreenInput::parse_str("screen.inp", &rendered)?;

        assert_eq!(rendered, rdinp::screen_inp_string());
        assert_eq!(reparsed, screen);
        Ok(())
    }

    #[test]
    fn rejects_invalid_screen_rendering() {
        let input = ScreenInput {
            ner: 40,
            nei: 20,
            maxl: 4,
            irrh: 1,
            iend: 0,
            lfxc: 0,
            emin: f64::NAN,
            emax: 0.0,
            eimax: 2.0,
            ermin: 0.001,
            rfms: 4.0,
            nrptx0: 251,
            icore: -1,
        };
        assert!(screen_input_string(&input).is_err());
    }
}
