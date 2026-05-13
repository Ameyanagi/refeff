//! Typed reader for FEFF `sfconv.inp` module handoff files.
//!
//! `sfconv.inp` controls spectral-function convolution after spectrum
//! assembly.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `sfconv.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvInput {
    /// Spectral-function convolution switches.
    pub control: SfconvControl,
    /// Width and center controls.
    pub window: SfconvWindow,
    /// Spectrum type and print flag.
    pub spectrum: SfconvSpectrum,
    /// Convolution filename, or `NULL`.
    pub cfname: String,
}

/// First control line of `sfconv.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvControl {
    pub msfconv: i32,
    pub ipse: i32,
    pub ipsk: i32,
}

/// Width and center controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvWindow {
    pub wsigk: f64,
    pub cen: f64,
}

/// Spectrum type and print controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfconvSpectrum {
    pub ispec: i32,
    pub ipr6: i32,
}

impl SfconvInput {
    /// Parse a FEFF `sfconv.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = SfconvInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `sfconv.inp` text.
pub fn sfconv_input_string(input: &SfconvInput) -> Result<String> {
    validate_sfconv_input(input)?;

    let mut out = String::new();
    out.push_str("msfconv, ipse, ipsk\n");
    push_i4_row(
        &mut out,
        [
            input.control.msfconv,
            input.control.ipse,
            input.control.ipsk,
        ],
    );
    out.push_str("wsigk, cen\n");
    out.push_str(&format!(
        "{:13.5}{:13.5}\n",
        input.window.wsigk, input.window.cen
    ));
    out.push_str("ispec, ipr6\n");
    push_i4_row(&mut out, [input.spectrum.ispec, input.spectrum.ipr6]);
    out.push_str("cfname\n");
    out.push_str(&sfconv_cfname_line(&input.cfname));
    out.push('\n');
    Ok(out)
}

fn validate_sfconv_input(input: &SfconvInput) -> Result<()> {
    validate_finite("wsigk", input.window.wsigk)?;
    validate_finite("cen", input.window.cen)?;
    if input
        .cfname
        .chars()
        .any(|character| matches!(character, '\n' | '\r'))
    {
        return Err(IoError::Parse {
            path: "sfconv.inp".into(),
            line: 0,
            message: "SFCONV filename must not contain a line terminator".to_string(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "sfconv.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn sfconv_cfname_line(cfname: &str) -> String {
    if cfname.len() <= 12 {
        format!("{cfname:<12}")
    } else {
        cfname.to_string()
    }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) {
    for value in values {
        out.push_str(&format!("{value:4}"));
    }
    out.push('\n');
}

struct SfconvInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> SfconvInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<SfconvInput> {
        self.expect_header("msfconv, ipse, ipsk")?;
        let control_values = self.parse_values::<i32>(3, "SFCONV control line")?;
        let control = SfconvControl {
            msfconv: control_values[0],
            ipse: control_values[1],
            ipsk: control_values[2],
        };

        self.expect_header("wsigk, cen")?;
        let window_values = self.parse_values::<f64>(2, "SFCONV window line")?;
        let window = SfconvWindow {
            wsigk: window_values[0],
            cen: window_values[1],
        };

        self.expect_header("ispec, ipr6")?;
        let spectrum_values = self.parse_values::<i32>(2, "SFCONV spectrum line")?;
        let spectrum = SfconvSpectrum {
            ispec: spectrum_values[0],
            ipr6: spectrum_values[1],
        };

        self.expect_header("cfname")?;
        let (_, line) = self.next_line("SFCONV filename line")?;

        Ok(SfconvInput {
            control,
            window,
            spectrum,
            cfname: line.trim().to_string(),
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

    use super::{SfconvInput, sfconv_input_string};

    #[test]
    fn parses_generated_sfconv_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let sfconv = SfconvInput::parse_str("sfconv.inp", &rdinp::sfconv_inp_string(&document)?)?;

        assert_eq!(sfconv.control.msfconv, 1);
        assert_eq!(sfconv.control.ipse, 0);
        assert_eq!(sfconv.control.ipsk, 0);
        assert_eq!(sfconv.window.wsigk, 0.0);
        assert_eq!(sfconv.window.cen, 0.0);
        assert_eq!(sfconv.spectrum.ispec, 1);
        assert_eq!(sfconv.spectrum.ipr6, 3);
        assert_eq!(sfconv.cfname, "NULL");
        Ok(())
    }

    #[test]
    fn renders_generated_sfconv_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
SFCONV
XANES
PRINT 0 0 0 0 0 3
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let expected = rdinp::sfconv_inp_string(&document)?;
        let sfconv = SfconvInput::parse_str("sfconv.inp", &expected)?;
        let rendered = sfconv_input_string(&sfconv)?;
        let reparsed = SfconvInput::parse_str("sfconv.inp", &rendered)?;

        assert_eq!(rendered, expected);
        assert_eq!(reparsed, sfconv);
        Ok(())
    }

    #[test]
    fn rejects_invalid_sfconv_rendering() {
        let input = SfconvInput {
            control: super::SfconvControl {
                msfconv: 1,
                ipse: 0,
                ipsk: 0,
            },
            window: super::SfconvWindow {
                wsigk: f64::NAN,
                cen: 0.0,
            },
            spectrum: super::SfconvSpectrum { ispec: 1, ipr6: 0 },
            cfname: "NULL".to_string(),
        };
        assert!(sfconv_input_string(&input).is_err());
    }
}
