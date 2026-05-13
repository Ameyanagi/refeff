//! Typed reader for FEFF `compton.inp` module handoff files.
//!
//! The COMPTON handoff carries momentum-grid, spatial-grid, windowing, density
//! output, and q-direction settings normalized by `rdinp`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `compton.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonInput {
    /// Whether the COMPTON module should run.
    pub run: bool,
    /// Momentum-grid controls.
    pub momentum: ComptonMomentum,
    /// Spatial mesh dimensions.
    pub grid: ComptonGrid,
    /// Spatial mesh limits.
    pub limits: ComptonLimits,
    /// Compton calculation switches.
    pub switches: ComptonSwitches,
    /// Window function controls.
    pub window: ComptonWindow,
    /// Electronic temperature in eV.
    pub temperature: f64,
    /// Optional chemical potential controls.
    pub chemical_potential: ComptonChemicalPotential,
    /// Density-output switches.
    pub density_outputs: ComptonDensityOutputs,
    /// Momentum-transfer direction.
    pub qhat: [f64; 3],
}

/// Momentum-grid controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonMomentum {
    pub pqmax: f64,
    pub npq: i32,
}

/// Spatial mesh dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComptonGrid {
    pub ns: i32,
    pub nphi: i32,
    pub nz: i32,
    pub nzp: i32,
}

/// Spatial mesh limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonLimits {
    pub smax: f64,
    pub phimax: f64,
    pub zmax: f64,
    pub zpmax: f64,
}

/// COMPTON calculation switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComptonSwitches {
    pub jpq: bool,
    pub rhozzp: bool,
    pub force_recalc_jzzp: bool,
}

/// Window function controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonWindow {
    pub window_type: i32,
    pub cutoff: f64,
}

/// Optional chemical potential controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComptonChemicalPotential {
    pub enabled: bool,
    pub value: f64,
}

/// Density-output switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComptonDensityOutputs {
    pub rho_xy: bool,
    pub rho_yz: bool,
    pub rho_xz: bool,
    pub rho_vol: bool,
    pub rho_line: bool,
}

impl ComptonInput {
    /// Parse a FEFF `compton.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = ComptonInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `compton.inp` text.
pub fn compton_input_string(input: &ComptonInput) -> Result<String> {
    validate_compton_input(input)?;

    let mut out = String::new();
    writeln!(out, "run compton module?")?;
    writeln!(out, "{:12}", i32::from(input.run))?;
    writeln!(out, "pqmax, npq")?;
    writeln!(
        out,
        "{:13.8}{:16}",
        input.momentum.pqmax, input.momentum.npq
    )?;
    writeln!(out, "ns, nphi, nz, nzp")?;
    writeln!(
        out,
        "{:4}{:4}{:4}{:4}",
        input.grid.ns, input.grid.nphi, input.grid.nz, input.grid.nzp
    )?;
    writeln!(out, "smax, phimax, zmax, zpmax")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}{:13.5}",
        input.limits.smax, input.limits.phimax, input.limits.zmax, input.limits.zpmax
    )?;
    writeln!(out, "jpq? rhozzp? force_recalc_jzzp?")?;
    writeln!(
        out,
        " {} {} {}",
        fortran_bool_field(input.switches.jpq),
        fortran_bool_field(input.switches.rhozzp),
        fortran_bool_field(input.switches.force_recalc_jzzp)
    )?;
    writeln!(out, "window_type (0=Step, 1=Hann), window_cutoff")?;
    writeln!(
        out,
        "{:12}{:13.8}    ",
        input.window.window_type, input.window.cutoff
    )?;
    writeln!(out, "temperature (in eV)")?;
    writeln!(out, "{:13.5}", input.temperature)?;
    writeln!(out, "set_chemical_potential? chemical_potential(eV)")?;
    writeln!(
        out,
        " {}{:13.8}    ",
        fortran_bool_field(input.chemical_potential.enabled),
        input.chemical_potential.value
    )?;
    writeln!(out, "rho_xy? rho_yz? rho_xz? rho_vol? rho_line?")?;
    writeln!(
        out,
        " {} {} {} {} {}",
        fortran_bool_field(input.density_outputs.rho_xy),
        fortran_bool_field(input.density_outputs.rho_yz),
        fortran_bool_field(input.density_outputs.rho_xz),
        fortran_bool_field(input.density_outputs.rho_vol),
        fortran_bool_field(input.density_outputs.rho_line)
    )?;
    writeln!(out, "qhat_x qhat_y qhat_z")?;
    writeln!(
        out,
        "{:21.16}{:26.16}{:26.16}     ",
        input.qhat[0], input.qhat[1], input.qhat[2]
    )?;
    Ok(out)
}

fn validate_compton_input(input: &ComptonInput) -> Result<()> {
    validate_finite("pqmax", input.momentum.pqmax)?;
    validate_finite("smax", input.limits.smax)?;
    validate_finite("phimax", input.limits.phimax)?;
    validate_finite("zmax", input.limits.zmax)?;
    validate_finite("zpmax", input.limits.zpmax)?;
    validate_finite("window_cutoff", input.window.cutoff)?;
    validate_finite("temperature", input.temperature)?;
    validate_finite("chemical_potential", input.chemical_potential.value)?;
    for (index, value) in input.qhat.iter().enumerate() {
        validate_finite(
            match index {
                0 => "qhat_x",
                1 => "qhat_y",
                _ => "qhat_z",
            },
            *value,
        )?;
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "compton.inp".into(),
            line: 0,
            message: format!("{field} must be finite"),
        })
    }
}

fn fortran_bool_field(value: bool) -> &'static str {
    if value { "T" } else { "F" }
}

struct ComptonInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> ComptonInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<ComptonInput> {
        self.expect_header("run compton module?")?;
        let run = self.parse_values::<i32>(1, "COMPTON run line")?[0] != 0;

        self.expect_header("pqmax, npq")?;
        let momentum_values = self.parse_values::<f64>(2, "COMPTON momentum line")?;
        let momentum = ComptonMomentum {
            pqmax: momentum_values[0],
            npq: momentum_values[1] as i32,
        };

        self.expect_header("ns, nphi, nz, nzp")?;
        let grid_values = self.parse_values::<i32>(4, "COMPTON grid line")?;
        let grid = ComptonGrid {
            ns: grid_values[0],
            nphi: grid_values[1],
            nz: grid_values[2],
            nzp: grid_values[3],
        };

        self.expect_header("smax, phimax, zmax, zpmax")?;
        let limit_values = self.parse_values::<f64>(4, "COMPTON limit line")?;
        let limits = ComptonLimits {
            smax: limit_values[0],
            phimax: limit_values[1],
            zmax: limit_values[2],
            zpmax: limit_values[3],
        };

        self.expect_header("jpq? rhozzp? force_recalc_jzzp?")?;
        let switches = self.parse_switches()?;

        self.expect_header("window_type (0=Step, 1=Hann), window_cutoff")?;
        let window_values = self.parse_values::<f64>(2, "COMPTON window line")?;
        let window = ComptonWindow {
            window_type: window_values[0] as i32,
            cutoff: window_values[1],
        };

        self.expect_header("temperature (in eV)")?;
        let temperature = self.parse_values::<f64>(1, "COMPTON temperature line")?[0];

        self.expect_header("set_chemical_potential? chemical_potential(eV)")?;
        let chemical_potential = self.parse_chemical_potential()?;

        self.expect_header("rho_xy? rho_yz? rho_xz? rho_vol? rho_line?")?;
        let density_outputs = self.parse_density_outputs()?;

        self.expect_header("qhat_x qhat_y qhat_z")?;
        let qhat_values = self.parse_values::<f64>(3, "COMPTON qhat line")?;
        let qhat = [qhat_values[0], qhat_values[1], qhat_values[2]];

        Ok(ComptonInput {
            run,
            momentum,
            grid,
            limits,
            switches,
            window,
            temperature,
            chemical_potential,
            density_outputs,
            qhat,
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

    fn parse_switches(&mut self) -> Result<ComptonSwitches> {
        let (line_number, line) = self.next_line("COMPTON switch line")?;
        let fields = bool_fields(line);
        if fields.len() < 3 {
            return Err(self.parse_error(line_number, "COMPTON switch line requires 3 fields"));
        }
        Ok(ComptonSwitches {
            jpq: parse_fortran_bool(&self.source, line_number, fields[0])?,
            rhozzp: parse_fortran_bool(&self.source, line_number, fields[1])?,
            force_recalc_jzzp: parse_fortran_bool(&self.source, line_number, fields[2])?,
        })
    }

    fn parse_chemical_potential(&mut self) -> Result<ComptonChemicalPotential> {
        let (line_number, line) = self.next_line("COMPTON chemical-potential line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(
                line_number,
                "COMPTON chemical-potential line requires 2 fields",
            ));
        }
        Ok(ComptonChemicalPotential {
            enabled: parse_fortran_bool(&self.source, line_number, fields[0])?,
            value: parse_field(&self.source, line_number, fields[1])?,
        })
    }

    fn parse_density_outputs(&mut self) -> Result<ComptonDensityOutputs> {
        let (line_number, line) = self.next_line("COMPTON density-output line")?;
        let fields = bool_fields(line);
        if fields.len() < 5 {
            return Err(
                self.parse_error(line_number, "COMPTON density-output line requires 5 fields")
            );
        }
        Ok(ComptonDensityOutputs {
            rho_xy: parse_fortran_bool(&self.source, line_number, fields[0])?,
            rho_yz: parse_fortran_bool(&self.source, line_number, fields[1])?,
            rho_xz: parse_fortran_bool(&self.source, line_number, fields[2])?,
            rho_vol: parse_fortran_bool(&self.source, line_number, fields[3])?,
            rho_line: parse_fortran_bool(&self.source, line_number, fields[4])?,
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

fn bool_fields(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::{ComptonInput, compton_input_string};

    #[test]
    fn parses_generated_compton_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COMPTON 7.0 300 1
RHOZZP
CGRID 12.0 20 21 22 23
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::compton_inp_string(&document)?;
        let compton = ComptonInput::parse_str("compton.inp", &text)?;

        assert!(compton.run);
        assert_eq!(compton.momentum.pqmax, 7.0);
        assert_eq!(compton.momentum.npq, 300);
        assert_eq!(compton.grid.ns, 20);
        assert_eq!(compton.grid.nphi, 21);
        assert_eq!(compton.grid.nz, 22);
        assert_eq!(compton.grid.nzp, 23);
        assert_eq!(compton.limits.smax, 0.0);
        assert!((compton.limits.phimax - std::f64::consts::TAU).abs() < 1.0e-5);
        assert_eq!(compton.limits.zpmax, 12.0);
        assert!(compton.switches.jpq);
        assert!(compton.switches.rhozzp);
        assert!(compton.switches.force_recalc_jzzp);
        assert_eq!(compton.window.window_type, 1);
        assert_eq!(compton.temperature, 0.0);
        assert!(!compton.chemical_potential.enabled);
        assert_eq!(compton.qhat, [0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn renders_generated_compton_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
COMPTON 7.0 300 1
RHOZZP
CGRID 12.0 20 21 22 23
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::compton_inp_string(&document)?;
        let compton = ComptonInput::parse_str("compton.inp", &text)?;

        assert_eq!(compton_input_string(&compton)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_compton_rendering() {
        let input = ComptonInput {
            run: true,
            momentum: super::ComptonMomentum {
                pqmax: f64::NAN,
                npq: 1000,
            },
            grid: super::ComptonGrid {
                ns: 32,
                nphi: 32,
                nz: 32,
                nzp: 144,
            },
            limits: super::ComptonLimits {
                smax: 0.0,
                phimax: std::f64::consts::TAU,
                zmax: 0.0,
                zpmax: 10.0,
            },
            switches: super::ComptonSwitches {
                jpq: true,
                rhozzp: true,
                force_recalc_jzzp: true,
            },
            window: super::ComptonWindow {
                window_type: 1,
                cutoff: 0.0,
            },
            temperature: 0.0,
            chemical_potential: super::ComptonChemicalPotential {
                enabled: false,
                value: 0.0,
            },
            density_outputs: super::ComptonDensityOutputs {
                rho_xy: false,
                rho_yz: false,
                rho_xz: false,
                rho_vol: false,
                rho_line: false,
            },
            qhat: [0.0, 0.0, 1.0],
        };
        assert!(compton_input_string(&input).is_err());
    }
}
