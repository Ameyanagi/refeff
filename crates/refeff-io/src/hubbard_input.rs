//! Typed reader for FEFF `hubbard.inp` module handoff files.
//!
//! `hubbard.inp` carries the Hubbard correction switches and scalar
//! parameters consumed by the Hubbard extension.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `hubbard.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HubbardInput {
    pub i_hubbard: i32,
    pub mldos_hubb: i32,
    pub u: f64,
    pub j: f64,
    pub fermi_shift: f64,
    pub l: i32,
}

impl HubbardInput {
    /// Parse a FEFF `hubbard.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = HubbardInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct HubbardInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> HubbardInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<HubbardInput> {
        self.expect_header("i_hubbard mldos_hubb U_hubbard J_hubbard fermi_shift l_hubbard")?;
        let values = self.parse_values::<f64>(6, "HUBBARD data line")?;
        Ok(HubbardInput {
            i_hubbard: values[0] as i32,
            mldos_hubb: values[1] as i32,
            u: values[2],
            j: values[3],
            fermi_shift: values[4],
            l: values[5] as i32,
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

    use super::HubbardInput;

    #[test]
    fn parses_generated_hubbard_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
HUBBARD 4.0 0.5 1.5 2
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let hubbard =
            HubbardInput::parse_str("hubbard.inp", &rdinp::hubbard_inp_string(&document))?;

        assert_eq!(hubbard.i_hubbard, 2);
        assert_eq!(hubbard.mldos_hubb, 2);
        assert_eq!(hubbard.u, 4.0);
        assert_eq!(hubbard.j, 0.5);
        assert_eq!(hubbard.fermi_shift, 1.5);
        assert_eq!(hubbard.l, 2);
        Ok(())
    }
}
