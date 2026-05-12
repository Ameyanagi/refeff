//! Typed reader for FEFF `ff2x.inp` module handoff files.
//!
//! FF2X consumes final spectrum controls, amplitude reduction, path criteria,
//! and momentum-transfer settings. This reader keeps those settings typed for
//! the Rust spectrum assembly stage.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `ff2x.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ff2xInput {
    /// FF2X run-control switches.
    pub control: Ff2xControl,
    /// Corrections and path cutoff used by spectrum assembly.
    pub corrections: Ff2xCorrections,
    /// Temperature and Debye-Waller settings.
    pub debye: Ff2xDebye,
    /// Momentum-transfer vector for EELS/NRIXS-compatible output.
    pub momentum_transfer: [f64; 3],
    /// NRIXS decomposition-channel count.
    pub decomposition_channels: i32,
    /// Electronic temperature in eV.
    pub electronic_temperature: f64,
}

/// First integer control line of `ff2x.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ff2xControl {
    pub mchi: i32,
    pub ispec: i32,
    pub idwopt: i32,
    pub ipr6: i32,
    pub mbconv: i32,
    pub absolu: i32,
    pub i_gamma_ch: i32,
}

/// Scalar correction fields from `ff2x.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ff2xCorrections {
    pub vrcorr: f64,
    pub vicorr: f64,
    pub s02: f64,
    pub critcw: f64,
}

/// Temperature and Debye-Waller controls from `ff2x.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ff2xDebye {
    pub tk: f64,
    pub thetad: f64,
    pub alphat: f64,
    pub thetae: f64,
    pub sig2g: f64,
    pub sig_gk: f64,
}

impl Ff2xInput {
    /// Parse a FEFF `ff2x.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = Ff2xInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct Ff2xInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> Ff2xInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<Ff2xInput> {
        self.expect_header("mchi, ispec, idwopt, ipr6, mbconv, absolu, iGammaCH")?;
        let control_values = self.parse_values::<i32>(7, "FF2X control line")?;
        let control = Ff2xControl {
            mchi: control_values[0],
            ispec: control_values[1],
            idwopt: control_values[2],
            ipr6: control_values[3],
            mbconv: control_values[4],
            absolu: control_values[5],
            i_gamma_ch: control_values[6],
        };

        self.expect_header("vrcorr, vicorr, s02, critcw")?;
        let correction_values = self.parse_values::<f64>(4, "FF2X correction line")?;
        let corrections = Ff2xCorrections {
            vrcorr: correction_values[0],
            vicorr: correction_values[1],
            s02: correction_values[2],
            critcw: correction_values[3],
        };

        self.expect_header("tk, thetad, alphat, thetae, sig2g, sig_gk")?;
        let debye_values = self.parse_values::<f64>(6, "FF2X Debye line")?;
        let debye = Ff2xDebye {
            tk: debye_values[0],
            thetad: debye_values[1],
            alphat: debye_values[2],
            thetae: debye_values[3],
            sig2g: debye_values[4],
            sig_gk: debye_values[5],
        };

        self.expect_header("momentum transfer")?;
        let momentum_values = self.parse_values::<f64>(3, "FF2X momentum-transfer line")?;
        let momentum_transfer = [momentum_values[0], momentum_values[1], momentum_values[2]];

        self.expect_header("the number of decomposi")?;
        let decomposition_channels = self.parse_values::<i32>(1, "FF2X decomposition line")?[0];
        self.expect_header("electronic temperature")?;
        let electronic_temperature = self.parse_values::<f64>(1, "FF2X temperature line")?[0];

        Ok(Ff2xInput {
            control,
            corrections,
            debye,
            momentum_transfer,
            decomposition_channels,
            electronic_temperature,
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

    use super::Ff2xInput;

    #[test]
    fn parses_generated_ff2x_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 2
EXAFS 20
S02 0.8
CORRECTIONS 1.1 2.2
CRITERIA 3.3 4.4
DEBYE 300.0 400.0
ABSOLUTE
NRIXS 1 1.0 2.0 -3.0
LDEC 5
TEMP 0.25
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::ff2x_inp_string(&document)?;
        let ff2x = Ff2xInput::parse_str("ff2x.inp", &text)?;

        assert_eq!(ff2x.control.mchi, 1);
        assert_eq!(ff2x.control.ispec, 1);
        assert_eq!(ff2x.control.idwopt, 0);
        assert_eq!(ff2x.control.ipr6, 2);
        assert_eq!(ff2x.control.absolu, 1);
        assert_eq!(ff2x.corrections.vrcorr, 1.1);
        assert_eq!(ff2x.corrections.vicorr, 2.2);
        assert_eq!(ff2x.corrections.s02, 0.8);
        assert_eq!(ff2x.corrections.critcw, 3.3);
        assert_eq!(ff2x.debye.tk, 300.0);
        assert_eq!(ff2x.debye.thetad, 400.0);
        assert_eq!(ff2x.momentum_transfer, [1.0, 2.0, -3.0]);
        assert_eq!(ff2x.decomposition_channels, 5);
        assert_eq!(ff2x.electronic_temperature, 0.25);
        Ok(())
    }
}
