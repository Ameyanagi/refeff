//! Typed reader for FEFF `fms.inp` module handoff files.
//!
//! The FMS module consumes cluster, Debye-Waller, and angular-momentum
//! controls from `rdinp`. This reader preserves that boundary for the Rust
//! full multiple-scattering implementation.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `fms.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct FmsInput {
    /// FMS run-control switches.
    pub control: FmsControl,
    /// FMS cluster and convergence settings.
    pub cluster: FmsCluster,
    /// Temperature and Debye-Waller settings.
    pub debye: FmsDebye,
    /// Angular-momentum cutoffs indexed by potential.
    pub lmaxph: Vec<i32>,
    /// NRIXS decomposition-channel count.
    pub decomposition_channels: i32,
    /// Whether to save a scattering-matrix slice.
    pub save_gg_slice: bool,
    /// Whether the FMS algorithm should contribute to the spectrum.
    pub do_fms: i32,
}

/// First integer control line of `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmsControl {
    pub mfms: i32,
    pub idwopt: i32,
    pub minv: i32,
}

/// Cluster and convergence controls from `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsCluster {
    pub rfms2: f64,
    pub rdirec: f64,
    pub toler1: f64,
    pub toler2: f64,
}

/// Temperature and Debye-Waller controls from `fms.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmsDebye {
    pub tk: f64,
    pub thetad: f64,
    pub sig2g: f64,
}

impl FmsInput {
    /// Parse a FEFF `fms.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = FmsInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `fms.inp` text.
pub fn fms_input_string(input: &FmsInput) -> Result<String> {
    validate_fms_input(input)?;

    let mut out = String::new();
    writeln!(out, "mfms, idwopt, minv")?;
    push_i4_row(
        &mut out,
        [input.control.mfms, input.control.idwopt, input.control.minv],
    )?;
    writeln!(out, "rfms2, rdirec, toler1, toler2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        input.cluster.rfms2, input.cluster.rdirec, input.cluster.toler1, input.cluster.toler2
    )?;
    writeln!(out, "tk, thetad, sig2g")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.debye.tk, input.debye.thetad, input.debye.sig2g
    )?;
    writeln!(out, " lmaxph(0:nph)")?;
    push_i4_row(&mut out, input.lmaxph.iter().copied())?;
    writeln!(out, " the number of decomposi")?;
    writeln!(out, "{:5}", input.decomposition_channels)?;
    writeln!(out, " save_gg_slice")?;
    writeln!(out, "{}", fortran_bool_field(input.save_gg_slice))?;
    writeln!(out, "do_fms")?;
    push_i4_row(&mut out, [input.do_fms])?;
    Ok(out)
}

fn validate_fms_input(input: &FmsInput) -> Result<()> {
    validate_finite("rfms2", input.cluster.rfms2)?;
    validate_finite("rdirec", input.cluster.rdirec)?;
    validate_finite("toler1", input.cluster.toler1)?;
    validate_finite("toler2", input.cluster.toler2)?;
    validate_finite("tk", input.debye.tk)?;
    validate_finite("thetad", input.debye.thetad)?;
    validate_finite("sig2g", input.debye.sig2g)?;
    if input.lmaxph.is_empty() {
        return Err(IoError::Parse {
            path: "fms.inp".into(),
            line: 0,
            message: "FMS input requires at least one lmaxph value".to_string(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "fms.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    out.push('\n');
    Ok(())
}

struct FmsInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> FmsInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<FmsInput> {
        self.expect_header("mfms, idwopt, minv")?;
        let control_values = self.parse_values::<i32>(3, "FMS control line")?;
        let control = FmsControl {
            mfms: control_values[0],
            idwopt: control_values[1],
            minv: control_values[2],
        };

        self.expect_header("rfms2, rdirec, toler1, toler2")?;
        let cluster_values = self.parse_values::<f64>(4, "FMS cluster line")?;
        let cluster = FmsCluster {
            rfms2: cluster_values[0],
            rdirec: cluster_values[1],
            toler1: cluster_values[2],
            toler2: cluster_values[3],
        };

        self.expect_header("tk, thetad, sig2g")?;
        let debye_values = self.parse_values::<f64>(3, "FMS Debye line")?;
        let debye = FmsDebye {
            tk: debye_values[0],
            thetad: debye_values[1],
            sig2g: debye_values[2],
        };

        self.expect_header("lmaxph(0:nph)")?;
        let lmaxph = self.parse_lmaxph()?;

        self.expect_header("the number of decomposi")?;
        let decomposition_channels = self.parse_values::<i32>(1, "FMS decomposition line")?[0];
        self.expect_header("save_gg_slice")?;
        let save_gg_slice = self.parse_bool_line("FMS save_gg_slice line")?;
        self.expect_header("do_fms")?;
        let do_fms = self.parse_values::<i32>(1, "FMS do_fms line")?[0];

        Ok(FmsInput {
            control,
            cluster,
            debye,
            lmaxph,
            decomposition_channels,
            save_gg_slice,
            do_fms,
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

    fn parse_lmaxph(&mut self) -> Result<Vec<i32>> {
        let (line_number, line) = self.next_line("FMS lmaxph line")?;
        line.split_whitespace()
            .map(|field| parse_field(&self.source, line_number, field))
            .collect()
    }

    fn parse_bool_line(&mut self, description: &str) -> Result<bool> {
        let (line_number, line) = self.next_line(description)?;
        parse_fortran_bool(&self.source, line_number, line.trim())
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

    use super::{FmsInput, fms_input_string};

    #[test]
    fn parses_generated_copper_fms_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FMS 7.0 1 2 0.002 0.003 10.0
DEBYE 300.0 400.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::fms_inp_string(&document)?;
        let fms = FmsInput::parse_str("fms.inp", &text)?;

        assert_eq!(fms.control.mfms, 1);
        assert_eq!(fms.control.idwopt, 0);
        assert_eq!(fms.control.minv, 2);
        assert_eq!(fms.cluster.rfms2, 7.0);
        assert_eq!(fms.cluster.rdirec, 10.0);
        assert_eq!(fms.cluster.toler1, 0.002);
        assert_eq!(fms.cluster.toler2, 0.003);
        assert_eq!(fms.debye.tk, 300.0);
        assert_eq!(fms.debye.thetad, 400.0);
        assert_eq!(fms.lmaxph, [3, 3]);
        assert_eq!(fms.decomposition_channels, -1);
        assert!(!fms.save_gg_slice);
        assert_eq!(fms.do_fms, 1);
        Ok(())
    }

    #[test]
    fn renders_generated_copper_fms_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
FMS 7.0 1 2 0.002 0.003 10.0
DEBYE 300.0 400.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.805 1.805 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::fms_inp_string(&document)?;
        let fms = FmsInput::parse_str("fms.inp", &text)?;

        assert_eq!(fms_input_string(&fms)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_fms_rendering() {
        let input = FmsInput {
            control: super::FmsControl {
                mfms: 1,
                idwopt: -1,
                minv: 0,
            },
            cluster: super::FmsCluster {
                rfms2: f64::NAN,
                rdirec: 10.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            debye: super::FmsDebye {
                tk: 0.0,
                thetad: 0.0,
                sig2g: 0.0,
            },
            lmaxph: vec![3],
            decomposition_channels: -1,
            save_gg_slice: false,
            do_fms: 1,
        };
        assert!(fms_input_string(&input).is_err());

        let mut empty_lmax = input;
        empty_lmax.cluster.rfms2 = 7.0;
        empty_lmax.lmaxph.clear();
        assert!(fms_input_string(&empty_lmax).is_err());
    }
}
