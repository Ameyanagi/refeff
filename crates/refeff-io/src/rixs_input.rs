//! Typed reader for FEFF `rixs.inp` module handoff files.
//!
//! RIXS input carries run switches, broadening values in Hartree-scaled units,
//! Fermi-level data, and the edge list assembled from FEFF cards.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `rixs.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsInput {
    /// Whether the RIXS module should run.
    pub run: bool,
    /// Core-hole and experimental broadening values.
    pub broadening: RixsBroadening,
    /// Incident and final energy windows.
    pub energy_window: RixsEnergyWindow,
    /// Fermi level in Hartrees.
    pub xmu: f64,
    /// RIXS calculation switches.
    pub switches: RixsSwitches,
    /// Edge labels in FEFF order.
    pub edges: Vec<String>,
}

/// Broadening fields from `rixs.inp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RixsBroadening {
    pub gam_ch: f64,
    pub gam_exp_1: f64,
    pub gam_exp_2: f64,
}

/// Incident and final energy windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RixsEnergyWindow {
    pub emin_i: f64,
    pub emax_i: f64,
    pub emin_f: f64,
    pub emax_f: f64,
}

/// Boolean RIXS control switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RixsSwitches {
    pub read_poles: bool,
    pub skip_calc: bool,
    pub mbconv: bool,
    pub read_sigma: bool,
}

impl RixsInput {
    /// Parse a FEFF `rixs.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = RixsInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `rixs.inp` text.
pub fn rixs_input_string(input: &RixsInput) -> Result<String> {
    validate_rixs_input(input)?;

    let mut out = String::new();
    writeln!(out, " m_run")?;
    writeln!(out, "{:12}", i32::from(input.run))?;
    writeln!(out, " gam_ch, gam_exp(1), gam_exp(2)")?;
    writeln!(
        out,
        "{:20.10}{:20.10}{:20.10}",
        input.broadening.gam_ch, input.broadening.gam_exp_1, input.broadening.gam_exp_2
    )?;
    writeln!(out, " EMinI, EMaxI, EMinF, EMaxF")?;
    writeln!(
        out,
        "{:20.10}{:20.10}{:20.10}{:20.10}",
        input.energy_window.emin_i,
        input.energy_window.emax_i,
        input.energy_window.emin_f,
        input.energy_window.emax_f
    )?;
    writeln!(out, " xmu")?;
    writeln!(out, " {:20.8}     ", input.xmu)?;
    writeln!(out, " Readpoles, SkipCalc, MBConv, ReadSigma")?;
    writeln!(
        out,
        " {} {} {} {}",
        fortran_bool_field(input.switches.read_poles),
        fortran_bool_field(input.switches.skip_calc),
        fortran_bool_field(input.switches.mbconv),
        fortran_bool_field(input.switches.read_sigma)
    )?;
    writeln!(out, " nEdges")?;
    writeln!(out, "{:12}", input.edges.len())?;
    for (idx, edge) in input.edges.iter().enumerate() {
        writeln!(out, " Edge{:12}", idx + 1)?;
        writeln!(out, " {edge}")?;
    }
    Ok(out)
}

fn validate_rixs_input(input: &RixsInput) -> Result<()> {
    validate_finite("gam_ch", input.broadening.gam_ch)?;
    validate_finite("gam_exp_1", input.broadening.gam_exp_1)?;
    validate_finite("gam_exp_2", input.broadening.gam_exp_2)?;
    validate_finite("emin_i", input.energy_window.emin_i)?;
    validate_finite("emax_i", input.energy_window.emax_i)?;
    validate_finite("emin_f", input.energy_window.emin_f)?;
    validate_finite("emax_f", input.energy_window.emax_f)?;
    validate_finite("xmu", input.xmu)?;
    if input.edges.is_empty() {
        return Err(IoError::Parse {
            path: "rixs.inp".into(),
            line: 0,
            message: "RIXS input requires at least one edge label".to_string(),
        });
    }
    for edge in &input.edges {
        if edge.contains(['\n', '\r']) {
            return Err(IoError::Parse {
                path: "rixs.inp".into(),
                line: 0,
                message: "RIXS edge labels cannot contain line terminators".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "rixs.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

struct RixsInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> RixsInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<RixsInput> {
        self.expect_header("m_run")?;
        let run = self.parse_values::<i32>(1, "RIXS run line")?[0] != 0;

        self.expect_header("gam_ch, gam_exp(1), gam_exp(2)")?;
        let broadening_values = self.parse_values::<f64>(3, "RIXS broadening line")?;
        let broadening = RixsBroadening {
            gam_ch: broadening_values[0],
            gam_exp_1: broadening_values[1],
            gam_exp_2: broadening_values[2],
        };

        self.expect_header("EMinI, EMaxI, EMinF, EMaxF")?;
        let window_values = self.parse_values::<f64>(4, "RIXS energy-window line")?;
        let energy_window = RixsEnergyWindow {
            emin_i: window_values[0],
            emax_i: window_values[1],
            emin_f: window_values[2],
            emax_f: window_values[3],
        };

        self.expect_header("xmu")?;
        let xmu = self.parse_values::<f64>(1, "RIXS xmu line")?[0];

        self.expect_header("Readpoles, SkipCalc, MBConv, ReadSigma")?;
        let switches = self.parse_switches()?;

        self.expect_header("nEdges")?;
        let edge_count = self.parse_values::<usize>(1, "RIXS nEdges line")?[0];
        let mut edges = Vec::with_capacity(edge_count);
        for edge_index in 1..=edge_count {
            self.expect_header(&format!("Edge{edge_index:12}"))?;
            let (_, line) = self.next_line("RIXS edge label")?;
            edges.push(line.trim().to_string());
        }

        Ok(RixsInput {
            run,
            broadening,
            energy_window,
            xmu,
            switches,
            edges,
        })
    }

    fn expect_header(&mut self, expected: &str) -> Result<()> {
        let (line_number, line) = self.next_line(expected)?;
        if line.trim() == expected.trim() {
            Ok(())
        } else {
            Err(self.parse_error(
                line_number,
                format!("expected header {expected:?}, found {line:?}"),
            ))
        }
    }

    fn parse_switches(&mut self) -> Result<RixsSwitches> {
        let (line_number, line) = self.next_line("RIXS switch line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, "RIXS switch line requires 4 fields"));
        }
        Ok(RixsSwitches {
            read_poles: parse_fortran_bool(&self.source, line_number, fields[0])?,
            skip_calc: parse_fortran_bool(&self.source, line_number, fields[1])?,
            mbconv: parse_fortran_bool(&self.source, line_number, fields[2])?,
            read_sigma: parse_fortran_bool(&self.source, line_number, fields[3])?,
        })
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

    use super::{RixsInput, rixs_input_string};

    #[test]
    fn parses_generated_rixs_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K L1 VAL
RIXS 1.0 2.0 3.0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::rixs_inp_string(&document)?;
        let rixs = RixsInput::parse_str("rixs.inp", &text)?;

        const HARTREE_EV: f64 = 27.211_396;
        assert!(rixs.run);
        assert!((rixs.broadening.gam_ch - 0.1 / (HARTREE_EV * HARTREE_EV)).abs() < 1.0e-10);
        assert!((rixs.broadening.gam_exp_1 - 1.0 / HARTREE_EV).abs() < 1.0e-10);
        assert!((rixs.broadening.gam_exp_2 - 2.0 / HARTREE_EV).abs() < 1.0e-10);
        assert_eq!(rixs.energy_window.emin_i, 0.0);
        assert!((rixs.xmu - 3.0 / HARTREE_EV).abs() < 1.0e-8);
        assert!(rixs.switches.read_poles);
        assert!(!rixs.switches.skip_calc);
        assert!(rixs.switches.mbconv);
        assert!(!rixs.switches.read_sigma);
        assert_eq!(rixs.edges, ["K", "L1", "VAL"]);
        Ok(())
    }

    #[test]
    fn renders_generated_rixs_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
EDGE K L1 VAL
RIXS 1.0 2.0 3.0
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::rixs_inp_string(&document)?;
        let rixs = RixsInput::parse_str("rixs.inp", &text)?;

        assert_eq!(rixs_input_string(&rixs)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_rixs_rendering() {
        let input = RixsInput {
            run: true,
            broadening: super::RixsBroadening {
                gam_ch: f64::NAN,
                gam_exp_1: 0.0,
                gam_exp_2: 0.0,
            },
            energy_window: super::RixsEnergyWindow {
                emin_i: 0.0,
                emax_i: 0.0,
                emin_f: 0.0,
                emax_f: 0.0,
            },
            xmu: 0.0,
            switches: super::RixsSwitches {
                read_poles: true,
                skip_calc: false,
                mbconv: true,
                read_sigma: false,
            },
            edges: vec!["K".to_string()],
        };
        assert!(rixs_input_string(&input).is_err());

        let mut bad_edge = input;
        bad_edge.broadening.gam_ch = 0.0;
        bad_edge.edges = vec!["K\nL1".to_string()];
        assert!(rixs_input_string(&bad_edge).is_err());
    }
}
