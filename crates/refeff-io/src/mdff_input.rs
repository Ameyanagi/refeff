//! Typed reader for FEFF `mdff.inp` EELS-MDFF controls.
//!
//! FEFF's standalone `mdff` program reads one ignored header line followed by
//! two integer controls: the task selector and the q-vector input mode. Keeping
//! this tiny handoff typed lets the Rust CLI validate cached `mdff.dat` output
//! without re-parsing text ad hoc.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `mdff.inp` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MdffInput {
    /// FEFF MDFF task selector: 1 for EELS cross section, 2 for MDFF, 3 for
    /// CAMDFF.
    pub task: i32,
    /// q-vector source selector: 1 for manual q/q-prime vectors, 2 for
    /// experimental EELS geometry.
    pub q_input: i32,
}

impl MdffInput {
    /// Parse a FEFF `mdff.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = MdffInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `mdff.inp` text.
pub fn mdff_input_string(input: &MdffInput) -> Result<String> {
    validate_mdff_input(input)?;

    let mut out = String::new();
    writeln!(out, "task and q-input mode for MDFF")?;
    writeln!(out, "{:4}{:4}", input.task, input.q_input)?;
    Ok(out)
}

fn validate_mdff_input(input: &MdffInput) -> Result<()> {
    if !(1..=3).contains(&input.task) {
        return Err(parse_error_value(
            0,
            format!("MDFF task {} must be 1, 2, or 3", input.task),
        ));
    }
    if !matches!(input.q_input, 1 | 2) {
        return Err(parse_error_value(
            0,
            format!("MDFF q-input mode {} must be 1 or 2", input.q_input),
        ));
    }
    Ok(())
}

struct MdffInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> MdffInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<MdffInput> {
        let _ = self.next_line("MDFF header line")?;
        let (line_number, line) = self.next_line("MDFF task control line")?;
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, "MDFF control line requires 2 fields"));
        }

        let input = MdffInput {
            task: parse_field(&self.source, line_number, "task", fields[0])?,
            q_input: parse_field(&self.source, line_number, "q_input", fields[1])?,
        };
        validate_mdff_input(&input)?;
        Ok(input)
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

fn parse_field<T>(source: &Path, line: usize, field: &'static str, token: &str) -> Result<T>
where
    T: FromStr,
{
    token.parse::<T>().map_err(|_| IoError::Parse {
        path: source.to_path_buf(),
        line,
        message: format!("could not parse {field} from {token:?}"),
    })
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "mdff.inp".into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mdff_input_controls() -> Result<()> {
        let input = MdffInput::parse_str("mdff.inp", "ignored header\n   1   2\n")?;

        assert_eq!(
            input,
            MdffInput {
                task: 1,
                q_input: 2
            }
        );
        Ok(())
    }

    #[test]
    fn renders_mdff_input_controls() -> Result<()> {
        let text = mdff_input_string(&MdffInput {
            task: 3,
            q_input: 1,
        })?;

        assert_eq!(MdffInput::parse_str("mdff.inp", &text)?.task, 3);
        assert_eq!(MdffInput::parse_str("mdff.inp", &text)?.q_input, 1);
        Ok(())
    }

    #[test]
    fn rejects_bad_mdff_input_controls() {
        assert!(MdffInput::parse_str("mdff.inp", "header\n1\n").is_err());
        assert!(MdffInput::parse_str("mdff.inp", "header\n4 2\n").is_err());
        assert!(MdffInput::parse_str("mdff.inp", "header\n1 9\n").is_err());
        assert!(
            mdff_input_string(&MdffInput {
                task: 0,
                q_input: 2
            })
            .is_err()
        );
    }
}
