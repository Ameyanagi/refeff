use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

use super::types::{SfconvControl, SfconvInput, SfconvSpectrum, SfconvWindow};

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

/// Select the FEFF spectrum files consumed by `SO2CONV`.
///
/// This mirrors the filename branch in `SFCONV/so2conv.f90`: base `xmu.dat`
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
