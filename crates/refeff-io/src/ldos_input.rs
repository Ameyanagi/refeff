//! Typed reader for FEFF `ldos.inp` module handoff files.
//!
//! `ldos.inp` combines LDOS mesh controls with FMS convergence settings and
//! per-potential angular momentum cutoffs. Keeping it typed prepares the Rust
//! LDOS module to consume normalized `rdinp` output directly.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `ldos.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosInput {
    /// LDOS run-control switches.
    pub control: LdosControl,
    /// LDOS energy mesh and radial grid settings.
    pub mesh: LdosMesh,
    /// FMS convergence settings reused by LDOS.
    pub fms: LdosFms,
    /// Angular-momentum cutoffs indexed by potential.
    pub lmaxph: Vec<i32>,
    /// LDOS output type selector.
    pub ldostype: i32,
}

/// First integer control line of `ldos.inp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LdosControl {
    pub mldos: i32,
    pub lfms2: i32,
    pub ixc: i32,
    pub ispin: i32,
    pub minv: i32,
    pub neldos: i32,
    pub iscfxc: i32,
}

/// Energy and radial-grid settings from `ldos.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LdosMesh {
    pub rfms2: f64,
    pub emin: f64,
    pub emax: f64,
    pub eimag: f64,
    pub rgrd: f64,
}

/// FMS convergence settings reused by the LDOS module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LdosFms {
    pub rdirec: f64,
    pub toler1: f64,
    pub toler2: f64,
}

impl LdosInput {
    /// Parse a FEFF `ldos.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = LdosInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `ldos.inp` text.
pub fn ldos_input_string(input: &LdosInput) -> Result<String> {
    validate_ldos_input(input)?;

    let mut out = String::new();
    writeln!(out, "mldos, lfms2, ixc, ispin, minv, neldos, iscfxc")?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}{:4} {:7} {:4}",
        input.control.mldos,
        input.control.lfms2,
        input.control.ixc,
        input.control.ispin,
        input.control.minv,
        input.control.neldos,
        input.control.iscfxc
    )?;
    writeln!(out, "rfms2, emin, emax, eimag, rgrd")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}{:13.5}",
        input.mesh.rfms2, input.mesh.emin, input.mesh.emax, input.mesh.eimag, input.mesh.rgrd
    )?;
    writeln!(out, "rdirec, toler1, toler2")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.fms.rdirec, input.fms.toler1, input.fms.toler2
    )?;
    writeln!(out, " lmaxph(0:nph)")?;
    push_i4_row(&mut out, input.lmaxph.iter().copied())?;
    writeln!(out, "ldostype")?;
    push_i4_row(&mut out, [input.ldostype])?;
    Ok(out)
}

fn validate_ldos_input(input: &LdosInput) -> Result<()> {
    validate_finite("rfms2", input.mesh.rfms2)?;
    validate_finite("emin", input.mesh.emin)?;
    validate_finite("emax", input.mesh.emax)?;
    validate_finite("eimag", input.mesh.eimag)?;
    validate_finite("rgrd", input.mesh.rgrd)?;
    validate_finite("rdirec", input.fms.rdirec)?;
    validate_finite("toler1", input.fms.toler1)?;
    validate_finite("toler2", input.fms.toler2)?;
    if input.lmaxph.is_empty() {
        return Err(IoError::Parse {
            path: "ldos.inp".into(),
            line: 0,
            message: "LDOS input requires at least one lmaxph value".to_string(),
        });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "ldos.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn push_i4_row(out: &mut String, values: impl IntoIterator<Item = i32>) -> Result<()> {
    for value in values {
        write!(out, "{value:4}")?;
    }
    out.push('\n');
    Ok(())
}

struct LdosInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> LdosInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<LdosInput> {
        self.expect_header("mldos, lfms2, ixc, ispin, minv, neldos, iscfxc")?;
        let control_values = self.parse_values::<i32>(7, "LDOS control line")?;
        let control = LdosControl {
            mldos: control_values[0],
            lfms2: control_values[1],
            ixc: control_values[2],
            ispin: control_values[3],
            minv: control_values[4],
            neldos: control_values[5],
            iscfxc: control_values[6],
        };

        self.expect_header("rfms2, emin, emax, eimag, rgrd")?;
        let mesh_values = self.parse_values::<f64>(5, "LDOS mesh line")?;
        let mesh = LdosMesh {
            rfms2: mesh_values[0],
            emin: mesh_values[1],
            emax: mesh_values[2],
            eimag: mesh_values[3],
            rgrd: mesh_values[4],
        };

        self.expect_header("rdirec, toler1, toler2")?;
        let fms_values = self.parse_values::<f64>(3, "LDOS FMS line")?;
        let fms = LdosFms {
            rdirec: fms_values[0],
            toler1: fms_values[1],
            toler2: fms_values[2],
        };

        self.expect_header("lmaxph(0:nph)")?;
        let lmaxph = self.parse_variable_i32_line("LDOS lmaxph line")?;
        self.expect_header("ldostype")?;
        let ldostype = self.parse_values::<i32>(1, "LDOS type line")?[0];

        Ok(LdosInput {
            control,
            mesh,
            fms,
            lmaxph,
            ldostype,
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

    fn parse_variable_i32_line(&mut self, description: &str) -> Result<Vec<i32>> {
        let (line_number, line) = self.next_line(description)?;
        line.split_whitespace()
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

    use super::{LdosInput, ldos_input_string};

    #[test]
    fn parses_generated_ldos_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
LDOS -30.0 20.0 0.1 151 2
FMS 4.5 1 2 0.002 0.003 8.0
EXCHANGE 5 0.0 0.0
SPIN 1 0.0 0.0 1.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::ldos_inp_string(&document)?;
        let ldos = LdosInput::parse_str("ldos.inp", &text)?;

        assert_eq!(ldos.control.mldos, 1);
        assert_eq!(ldos.control.lfms2, 1);
        assert_eq!(ldos.control.ixc, 5);
        assert_eq!(ldos.control.ispin, 1);
        assert_eq!(ldos.control.minv, 2);
        assert_eq!(ldos.control.neldos, 151);
        assert_eq!(ldos.mesh.rfms2, 4.5);
        assert_eq!(ldos.mesh.emin, -30.0);
        assert_eq!(ldos.mesh.emax, 20.0);
        assert_eq!(ldos.mesh.eimag, 0.1);
        assert_eq!(ldos.fms.rdirec, 8.0);
        assert_eq!(ldos.fms.toler1, 0.002);
        assert_eq!(ldos.fms.toler2, 0.003);
        assert_eq!(ldos.lmaxph, [3, 3]);
        assert_eq!(ldos.ldostype, 2);
        Ok(())
    }

    #[test]
    fn renders_generated_ldos_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
LDOS -30.0 20.0 0.1 151 2
FMS 4.5 1 2 0.002 0.003 8.0
EXCHANGE 5 0.0 0.0
SPIN 1 0.0 0.0 1.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
1.0 0.0 0.0 1 Cu1
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::ldos_inp_string(&document)?;
        let ldos = LdosInput::parse_str("ldos.inp", &text)?;

        assert_eq!(ldos_input_string(&ldos)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_ldos_rendering() {
        let input = LdosInput {
            control: super::LdosControl {
                mldos: 1,
                lfms2: 1,
                ixc: 0,
                ispin: 0,
                minv: 0,
                neldos: 101,
                iscfxc: 11,
            },
            mesh: super::LdosMesh {
                rfms2: f64::NAN,
                emin: 0.0,
                emax: 0.0,
                eimag: -1.0,
                rgrd: 0.05,
            },
            fms: super::LdosFms {
                rdirec: 10.0,
                toler1: 0.001,
                toler2: 0.001,
            },
            lmaxph: vec![3],
            ldostype: 0,
        };
        assert!(ldos_input_string(&input).is_err());

        let mut empty_lmax = input;
        empty_lmax.mesh.rfms2 = 5.0;
        empty_lmax.lmaxph.clear();
        assert!(ldos_input_string(&empty_lmax).is_err());
    }
}
