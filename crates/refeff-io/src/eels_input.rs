//! Typed reader for FEFF `eels.inp` module handoff files.
//!
//! The EELS/ELNES handoff carries beam, detector, q-mesh, and polarization
//! controls normalized by `rdinp`. This reader gives the Rust spectroscopy
//! modules typed access to those settings.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `eels.inp` file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EelsInput {
    /// Whether to calculate ELNES/EXELFS.
    pub calculate_elnes: bool,
    /// Raw FEFF EELS calculation switch (`1` for normal EELS, `9` for GOS).
    pub calculation_mode: i32,
    /// Orientation, relativistic, and input-source controls.
    pub control: EelsControl,
    /// Polarization index range.
    pub polarization: EelsPolarization,
    /// Beam energy in eV.
    pub beam_energy: f64,
    /// Beam direction in arbitrary units.
    pub beam_direction: [f64; 3],
    /// Collection and convergence semiangles in radians.
    pub angles: EelsAngles,
    /// q-mesh dimensions.
    pub qmesh: EelsQMesh,
    /// Detector position angles in radians.
    pub detector: [f64; 2],
    /// Magic-angle calculation switch.
    pub magic: i32,
    /// Energy for magic angle in eV above threshold.
    pub magic_energy: f64,
}

/// EELS control line after the calculate switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EelsControl {
    pub average: i32,
    pub relativistic: i32,
    pub cross_terms: i32,
    pub input: i32,
    pub spectrum_column: i32,
}

/// Polarization index range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EelsPolarization {
    pub min: i32,
    pub step: i32,
    pub max: i32,
}

/// Collection and convergence semiangles in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EelsAngles {
    pub collection: f64,
    pub convergence: f64,
}

/// q-mesh dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EelsQMesh {
    pub radial: i32,
    pub angular: i32,
}

impl EelsInput {
    /// Parse a FEFF `eels.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = EelsInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `eels.inp` text.
pub fn eels_input_string(input: &EelsInput) -> Result<String> {
    validate_eels_input(input)?;

    let mut out = String::new();
    writeln!(out, "calculate ELNES?")?;
    let calculation_mode = if input.calculate_elnes {
        input.calculation_mode
    } else {
        0
    };
    push_i4_row(&mut out, [calculation_mode])?;
    writeln!(out, "average? relativistic? cross-terms? Which input?")?;
    push_i4_row(
        &mut out,
        [
            input.control.average,
            input.control.relativistic,
            input.control.cross_terms,
            input.control.input,
            input.control.spectrum_column,
        ],
    )?;
    writeln!(out, "polarizations to be used ; min step max")?;
    push_i4_row(
        &mut out,
        [
            input.polarization.min,
            input.polarization.step,
            input.polarization.max,
        ],
    )?;
    writeln!(out, "beam energy in eV")?;
    writeln!(out, "{:13.5}", input.beam_energy)?;
    writeln!(out, "beam direction in arbitrary units")?;
    writeln!(
        out,
        "{:13.5}{:13.5}{:13.5}",
        input.beam_direction[0], input.beam_direction[1], input.beam_direction[2]
    )?;
    writeln!(out, "collection and convergence semiangle in rad")?;
    writeln!(
        out,
        "{:13.5}{:13.5}",
        input.angles.collection, input.angles.convergence
    )?;
    writeln!(out, "qmesh - radial and angular grid size")?;
    push_i4_row(&mut out, [input.qmesh.radial, input.qmesh.angular])?;
    writeln!(out, "detector positions - two angles in rad")?;
    writeln!(out, "{:13.5}{:13.5}", input.detector[0], input.detector[1])?;
    writeln!(out, "calculate magic angle if magic=1")?;
    push_i4_row(&mut out, [input.magic])?;
    writeln!(out, "energy for magic angle - eV above threshold")?;
    writeln!(out, "{:13.5}", input.magic_energy)?;
    Ok(out)
}

fn validate_eels_input(input: &EelsInput) -> Result<()> {
    if input.calculate_elnes && input.calculation_mode == 0 {
        return Err(IoError::Parse {
            path: "eels.inp".into(),
            line: 0,
            message: "calculation_mode must be nonzero when EELS is enabled".to_string(),
        });
    }
    validate_finite("beam_energy", input.beam_energy)?;
    for (index, value) in input.beam_direction.iter().enumerate() {
        validate_finite(
            match index {
                0 => "beam_direction_x",
                1 => "beam_direction_y",
                _ => "beam_direction_z",
            },
            *value,
        )?;
    }
    validate_finite("collection_angle", input.angles.collection)?;
    validate_finite("convergence_angle", input.angles.convergence)?;
    validate_finite("detector_theta", input.detector[0])?;
    validate_finite("detector_phi", input.detector[1])?;
    validate_finite("magic_energy", input.magic_energy)
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::Parse {
            path: "eels.inp".into(),
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

struct EelsInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> EelsInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<EelsInput> {
        self.expect_header("calculate ELNES?")?;
        let calculation_mode = self.parse_values::<i32>(1, "EELS calculate line")?[0];
        let calculate_elnes = calculation_mode != 0;

        self.expect_header("average? relativistic? cross-terms? Which input?")?;
        let control_values = self.parse_values::<i32>(5, "EELS control line")?;
        let control = EelsControl {
            average: control_values[0],
            relativistic: control_values[1],
            cross_terms: control_values[2],
            input: control_values[3],
            spectrum_column: control_values[4],
        };

        self.expect_header("polarizations to be used ; min step max")?;
        let polarization_values = self.parse_values::<i32>(3, "EELS polarization line")?;
        let polarization = EelsPolarization {
            min: polarization_values[0],
            step: polarization_values[1],
            max: polarization_values[2],
        };

        self.expect_header("beam energy in eV")?;
        let beam_energy = self.parse_values::<f64>(1, "EELS beam-energy line")?[0];
        self.expect_header("beam direction in arbitrary units")?;
        let beam_values = self.parse_values::<f64>(3, "EELS beam-direction line")?;
        let beam_direction = [beam_values[0], beam_values[1], beam_values[2]];

        self.expect_header("collection and convergence semiangle in rad")?;
        let angle_values = self.parse_values::<f64>(2, "EELS angle line")?;
        let angles = EelsAngles {
            collection: angle_values[0],
            convergence: angle_values[1],
        };

        self.expect_header("qmesh - radial and angular grid size")?;
        let qmesh_values = self.parse_values::<i32>(2, "EELS qmesh line")?;
        let qmesh = EelsQMesh {
            radial: qmesh_values[0],
            angular: qmesh_values[1],
        };

        self.expect_header("detector positions - two angles in rad")?;
        let detector_values = self.parse_values::<f64>(2, "EELS detector line")?;
        let detector = [detector_values[0], detector_values[1]];

        self.expect_header("calculate magic angle if magic=1")?;
        let magic = self.parse_values::<i32>(1, "EELS magic line")?[0];
        self.expect_header("energy for magic angle - eV above threshold")?;
        let magic_energy = self.parse_values::<f64>(1, "EELS magic-energy line")?[0];

        Ok(EelsInput {
            calculate_elnes,
            calculation_mode,
            control,
            polarization,
            beam_energy,
            beam_direction,
            angles,
            qmesh,
            detector,
            magic,
            magic_energy,
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

    use super::{EelsInput, eels_input_string};

    #[test]
    fn parses_generated_elnes_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ELNES
200 0 1 1 2 3
0.0 0.0 2.0
15.0 20.0
8 6
3.0 4.0
MAGIC 12.5
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::eels_inp_string(&document)?;
        let eels = EelsInput::parse_str("eels.inp", &text)?;

        assert!(eels.calculate_elnes);
        assert_eq!(eels.calculation_mode, 1);
        assert_eq!(eels.control.average, 0);
        assert_eq!(eels.control.relativistic, 1);
        assert_eq!(eels.control.cross_terms, 1);
        assert_eq!(eels.control.input, 2);
        assert_eq!(eels.control.spectrum_column, 3);
        assert_eq!(eels.polarization.min, 1);
        assert_eq!(eels.polarization.step, 1);
        assert_eq!(eels.polarization.max, 9);
        assert_eq!(eels.beam_energy, 200000.0);
        assert_eq!(eels.beam_direction, [0.0, 0.0, 1.0]);
        assert_eq!(eels.angles.collection, 0.015);
        assert_eq!(eels.angles.convergence, 0.020);
        assert_eq!(eels.qmesh.radial, 8);
        assert_eq!(eels.qmesh.angular, 6);
        assert_eq!(eels.detector, [0.003, 0.004]);
        assert_eq!(eels.magic, 1);
        assert_eq!(eels.magic_energy, 12.5);
        Ok(())
    }

    #[test]
    fn renders_generated_elnes_input() -> crate::Result<()> {
        let input = FeffInput::parse_str(
            "feff.inp",
            r#"
ELNES
200 0 1 1 2 3
0.0 0.0 2.0
15.0 20.0
8 6
3.0 4.0
MAGIC 12.5
POTENTIALS
0 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0
END
"#,
        )?;
        let document = FeffDocument::from_input(&input)?;
        let text = rdinp::eels_inp_string(&document)?;
        let eels = EelsInput::parse_str("eels.inp", &text)?;

        assert_eq!(eels_input_string(&eels)?, text);
        Ok(())
    }

    #[test]
    fn preserves_gos_calculation_mode() -> crate::Result<()> {
        let text = concat!(
            "calculate ELNES?\n",
            "   9\n",
            "average? relativistic? cross-terms? Which input?\n",
            "   1   1   1   1   4\n",
            "polarizations to be used ; min step max\n",
            "   1   1   9\n",
            "beam energy in eV\n",
            " 300000.00000\n",
            "beam direction in arbitrary units\n",
            "      0.00000      1.00000      0.00000\n",
            "collection and convergence semiangle in rad\n",
            "      0.00240      0.00000\n",
            "qmesh - radial and angular grid size\n",
            "   5   3\n",
            "detector positions - two angles in rad\n",
            "      0.00000      0.00000\n",
            "calculate magic angle if magic=1\n",
            "   0\n",
            "energy for magic angle - eV above threshold\n",
            "      0.00000\n",
        );

        let eels = EelsInput::parse_str("eels.inp", text)?;

        assert!(eels.calculate_elnes);
        assert_eq!(eels.calculation_mode, 9);
        assert_eq!(eels_input_string(&eels)?, text);
        Ok(())
    }

    #[test]
    fn rejects_invalid_eels_rendering() {
        let input = EelsInput {
            calculate_elnes: true,
            calculation_mode: 1,
            control: super::EelsControl {
                average: 0,
                relativistic: 1,
                cross_terms: 1,
                input: 2,
                spectrum_column: 3,
            },
            polarization: super::EelsPolarization {
                min: 1,
                step: 1,
                max: 9,
            },
            beam_energy: f64::NAN,
            beam_direction: [0.0, 0.0, 1.0],
            angles: super::EelsAngles {
                collection: 0.015,
                convergence: 0.020,
            },
            qmesh: super::EelsQMesh {
                radial: 8,
                angular: 6,
            },
            detector: [0.003, 0.004],
            magic: 1,
            magic_energy: 12.5,
        };
        assert!(eels_input_string(&input).is_err());
    }
}
