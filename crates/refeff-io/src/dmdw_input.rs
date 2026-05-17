//! Typed reader for FEFF `dmdw.inp` module handoff files.
//!
//! `dmdw.inp` is either disabled with FEFF's `-999` sentinel or contains a
//! dynamical-matrix Debye-Waller calculation request plus selected path rows.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

/// Parsed contents of a FEFF `dmdw.inp` file.
#[derive(Debug, Clone, PartialEq)]
pub enum DmdwInput {
    /// FEFF sentinel for no standalone DMDW calculation.
    Disabled,
    /// Enabled standalone DMDW calculation.
    Enabled(DmdwCalculation),
}

/// Enabled DMDW calculation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwCalculation {
    /// Module run flag written by FEFF.
    pub run: i32,
    /// Lanczos recursion order.
    pub order: i32,
    /// Temperature selector from the third data line.
    pub temperature_flag: i32,
    /// Sample temperature, or the lower grid bound for multi-temperature runs.
    pub temperature: f64,
    /// Upper temperature grid bound for multi-temperature runs.
    pub temperature_max: Option<f64>,
    /// Dynamical matrix calculation type selector.
    pub calculation_type: i32,
    /// Dynamical matrix filename.
    pub dym_file: String,
    /// Number of path rows.
    pub path_count: usize,
    /// Selected path rows.
    pub paths: Vec<DmdwPath>,
}

/// One path row from `dmdw.inp`.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwPath {
    /// Number of legs in the path row.
    pub leg_count: i32,
    /// Absorber selector written by FEFF.
    pub absorber_selector: i32,
    /// Potential selectors before the distance field.
    pub potentials: Vec<i32>,
    /// Maximum path distance field as written in `dmdw.inp`.
    ///
    /// FEFF's DMDW reader multiplies this value by its Angstrom-to-Bohr
    /// conversion before descriptor expansion.
    pub max_distance: f64,
}

impl DmdwInput {
    /// Parse a FEFF `dmdw.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = DmdwInputParser::new(source.into(), text);
        parser.parse()
    }
}

/// Render FEFF-compatible `dmdw.inp` text.
pub fn dmdw_input_string(input: &DmdwInput) -> Result<String> {
    match input {
        DmdwInput::Disabled => Ok("-999\n".to_string()),
        DmdwInput::Enabled(calculation) => dmdw_calculation_string(calculation),
    }
}

fn dmdw_calculation_string(calculation: &DmdwCalculation) -> Result<String> {
    validate_dmdw_calculation(calculation)?;

    let mut out = String::new();
    out.push_str(&format!("{:4}\n", calculation.run));
    out.push_str(&format!("{:4}\n", calculation.order));
    if calculation.temperature_flag == 1 {
        out.push_str(&format!(
            "{:4}{:11.3}\n",
            calculation.temperature_flag, calculation.temperature
        ));
    } else {
        let temperature_max = calculation.temperature_max.ok_or_else(|| IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW multi-temperature run requires an upper temperature".to_string(),
        })?;
        out.push_str(&format!(
            "{:4}{:11.3}{:11.3}\n",
            calculation.temperature_flag, calculation.temperature, temperature_max
        ));
    }
    out.push_str(&format!("{:4}\n", calculation.calculation_type));
    out.push_str(&calculation.dym_file);
    out.push('\n');
    out.push_str(&format!("{:4}\n", calculation.path_count));
    for path in &calculation.paths {
        out.push_str(&dmdw_path_line(path)?);
    }
    Ok(out)
}

fn validate_dmdw_calculation(calculation: &DmdwCalculation) -> Result<()> {
    if calculation.path_count != calculation.paths.len() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!(
                "DMDW path_count is {} but has {} path row(s)",
                calculation.path_count,
                calculation.paths.len()
            ),
        });
    }
    if calculation.temperature_flag <= 0 {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: format!(
                "DMDW temperature count must be positive, got {}",
                calculation.temperature_flag
            ),
        });
    }
    if !calculation.temperature.is_finite() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW temperature must be finite".to_string(),
        });
    }
    if calculation.temperature_flag == 1 && calculation.temperature_max.is_some() {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW single-temperature run must not set an upper temperature".to_string(),
        });
    }
    if calculation.temperature_flag > 1 {
        let Some(temperature_max) = calculation.temperature_max else {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "DMDW multi-temperature run requires an upper temperature".to_string(),
            });
        };
        if !temperature_max.is_finite() {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "DMDW upper temperature must be finite".to_string(),
            });
        }
    }
    if calculation
        .dym_file
        .chars()
        .any(|character| matches!(character, '\n' | '\r'))
    {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW dynamical-matrix filename must not contain a line terminator"
                .to_string(),
        });
    }
    for (index, path) in calculation.paths.iter().enumerate() {
        if !path.max_distance.is_finite() {
            return Err(IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: format!("DMDW path {index} maximum distance must be finite"),
            });
        }
    }
    Ok(())
}

fn dmdw_path_line(path: &DmdwPath) -> Result<String> {
    let mut prefix = String::new();
    prefix.push_str(&format!("{:4}", path.leg_count));
    prefix.push_str(&format!("{:4}", path.absorber_selector));
    for potential in &path.potentials {
        prefix.push_str(&format!("{potential:4}"));
    }
    if prefix.len() > 20 {
        return Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line: 0,
            message: "DMDW path row has too many integer selector fields".to_string(),
        });
    }
    Ok(format!("{prefix:<20}{:7.2}\n", path.max_distance))
}

struct DmdwInputParser<'a> {
    source: PathBuf,
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> DmdwInputParser<'a> {
    fn new(source: PathBuf, text: &'a str) -> Self {
        Self {
            source,
            lines: text.lines().enumerate(),
        }
    }

    fn parse(&mut self) -> Result<DmdwInput> {
        let run = self.parse_values::<i32>(1, "DMDW run line")?[0];
        if run == -999 {
            return Ok(DmdwInput::Disabled);
        }

        let order = self.parse_values::<i32>(1, "DMDW order line")?[0];
        let (temperature_flag, temperature, temperature_max) = self.parse_temperature()?;
        let calculation_type = self.parse_values::<i32>(1, "DMDW type line")?[0];
        let (_, dym_file) = self.next_line("DMDW dynamical-matrix filename")?;
        let path_count = self.parse_values::<usize>(1, "DMDW path-count line")?[0];
        let mut paths = Vec::with_capacity(path_count);
        for _ in 0..path_count {
            paths.push(self.parse_path()?);
        }

        Ok(DmdwInput::Enabled(DmdwCalculation {
            run,
            order,
            temperature_flag,
            temperature,
            temperature_max,
            calculation_type,
            dym_file: dym_file.trim().to_string(),
            path_count,
            paths,
        }))
    }

    fn parse_temperature(&mut self) -> Result<(i32, f64, Option<f64>)> {
        let (line_number, line) = self.next_line("DMDW temperature line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, "DMDW temperature line requires 2 fields"));
        }
        let temperature_flag = parse_field(&self.source, line_number, fields[0])?;
        let temperature = parse_field(&self.source, line_number, fields[1])?;
        let temperature_max = if temperature_flag == 1 {
            None
        } else {
            let Some(field) = fields.get(2) else {
                return Err(
                    self.parse_error(line_number, "DMDW multi-temperature line requires 3 fields")
                );
            };
            Some(parse_field(&self.source, line_number, field)?)
        };
        Ok((temperature_flag, temperature, temperature_max))
    }

    fn parse_path(&mut self) -> Result<DmdwPath> {
        let (line_number, line) = self.next_line("DMDW path row")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(self.parse_error(line_number, "DMDW path row requires at least 3 fields"));
        }
        let leg_count = parse_field(&self.source, line_number, fields[0])?;
        let absorber_selector = parse_field(&self.source, line_number, fields[1])?;
        let potentials = fields[2..fields.len() - 1]
            .iter()
            .map(|field| parse_field(&self.source, line_number, field))
            .collect::<Result<Vec<i32>>>()?;
        let max_distance = parse_field(&self.source, line_number, fields[fields.len() - 1])?;
        Ok(DmdwPath {
            leg_count,
            absorber_selector,
            potentials,
            max_distance,
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

#[cfg(test)]
mod tests {
    use crate::{FeffDocument, FeffInput, rdinp};

    use super::{DmdwInput, DmdwPath, dmdw_input_string};
    use refeff_core::{DMDW_ANGSTROM_TO_BOHR, DmdwPathDescriptor, dmdw_expand_path_descriptors};

    #[test]
    fn parses_disabled_dmdw_input() -> crate::Result<()> {
        let input = FeffInput::parse_str("feff.inp", "END\n")?;
        let document = FeffDocument::from_input(&input)?;
        let dmdw = DmdwInput::parse_str("dmdw.inp", &rdinp::dmdw_inp_string(&document)?)?;
        assert_eq!(dmdw, DmdwInput::Disabled);
        Ok(())
    }

    #[test]
    fn renders_disabled_dmdw_input() -> crate::Result<()> {
        let input = FeffInput::parse_str("feff.inp", "END\n")?;
        let document = FeffDocument::from_input(&input)?;
        let expected = rdinp::dmdw_inp_string(&document)?;
        let dmdw = DmdwInput::parse_str("dmdw.inp", &expected)?;

        assert_eq!(dmdw_input_string(&dmdw)?, expected);
        Ok(())
    }

    #[test]
    fn parses_generated_dmdw_routes() -> crate::Result<()> {
        let temp = tempfile::tempdir().map_err(|source| crate::IoError::io("tempdir", source))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
            .map_err(|source| crate::IoError::io("feff.dym", source))?;
        let input = FeffInput::parse_str(
            &input_path,
            r#"
DEBYE 450 315 5 feff.dym 6 7 13
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
        let dmdw = DmdwInput::parse_str("dmdw.inp", &rdinp::dmdw_inp_string(&document)?)?;
        let DmdwInput::Enabled(calculation) = dmdw else {
            return Err(crate::IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "expected enabled DMDW calculation".to_string(),
            });
        };

        assert_eq!(calculation.run, 1);
        assert_eq!(calculation.order, 6);
        assert_eq!(calculation.temperature_flag, 1);
        assert_eq!(calculation.temperature, 450.0);
        assert_eq!(calculation.temperature_max, None);
        assert_eq!(calculation.calculation_type, 7);
        assert_eq!(calculation.dym_file, "feff.dym");
        assert_eq!(calculation.path_count, 3);
        assert_eq!(calculation.paths.len(), 3);
        assert_eq!(
            calculation.paths[0],
            DmdwPath {
                leg_count: 2,
                absorber_selector: 0,
                potentials: vec![0],
                max_distance: calculation.paths[0].max_distance,
            }
        );
        assert_eq!(calculation.paths[1].leg_count, 3);
        assert_eq!(calculation.paths[2].leg_count, 4);
        assert!(calculation.paths[2].max_distance > calculation.paths[1].max_distance);
        Ok(())
    }

    #[test]
    fn parses_and_renders_multi_temperature_grid() -> crate::Result<()> {
        let text = concat!(
            "   1\n",
            "   2\n",
            "   3    100.000    500.000\n",
            "   0\n",
            "feff.dym\n",
            "   1\n",
            "   2   0   1          10.00\n",
        );
        let dmdw = DmdwInput::parse_str("dmdw.inp", text)?;
        let DmdwInput::Enabled(calculation) = &dmdw else {
            return Err(crate::IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "expected enabled DMDW calculation".to_string(),
            });
        };

        assert_eq!(calculation.temperature_flag, 3);
        assert_eq!(calculation.temperature, 100.0);
        assert_eq!(calculation.temperature_max, Some(500.0));
        assert_eq!(dmdw_input_string(&dmdw)?, text);
        Ok(())
    }

    #[test]
    fn parses_single_atom_dmdw_descriptor() -> crate::Result<()> {
        let dmdw = DmdwInput::parse_str(
            "dmdw.inp",
            concat!(
                "   1\n",
                "   2\n",
                "   1     77.000\n",
                "   3\n",
                "feff.dym\n",
                "   1\n",
                "   1   2   10.00\n",
            ),
        )?;
        let DmdwInput::Enabled(calculation) = dmdw else {
            return Err(crate::IoError::Parse {
                path: "dmdw.inp".into(),
                line: 0,
                message: "expected enabled DMDW calculation".to_string(),
            });
        };

        assert_eq!(
            calculation.paths[0],
            DmdwPath {
                leg_count: 1,
                absorber_selector: 2,
                potentials: Vec::new(),
                max_distance: 10.0,
            }
        );
        Ok(())
    }

    #[test]
    fn renders_generated_dmdw_routes() -> crate::Result<()> {
        let temp = tempfile::tempdir().map_err(|source| crate::IoError::io("tempdir", source))?;
        let input_path = temp.path().join("feff.inp");
        std::fs::write(temp.path().join("feff.dym"), minimal_dym_text())
            .map_err(|source| crate::IoError::io("feff.dym", source))?;
        let input = FeffInput::parse_str(
            &input_path,
            r#"
DEBYE 450 315 5 feff.dym 6 7 13
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
        let expected = rdinp::dmdw_inp_string(&document)?;
        let dmdw = DmdwInput::parse_str("dmdw.inp", &expected)?;
        let rendered = dmdw_input_string(&dmdw)?;
        let reparsed = DmdwInput::parse_str("dmdw.inp", &rendered)?;

        assert_eq!(rendered, expected);
        assert_eq!(reparsed, dmdw);
        Ok(())
    }

    #[test]
    fn expands_feff_h2o_reference_descriptors_when_available() -> crate::Result<()> {
        let Some(reference_dir) = reference_dmdw_test_dir()? else {
            eprintln!("skipping DMDW H2O descriptor expansion test; feff10 fixture not found");
            return Ok(());
        };

        let input_path = reference_dir.join("H2O.g03.dmdw.inp");
        let dym_path = reference_dir.join("H2O.g03.dym");
        let output_path = reference_dir
            .join("Reference_Results")
            .join("H2O.g03.dmdw.out");
        let input_text = std::fs::read_to_string(&input_path)
            .map_err(|source| crate::IoError::io(input_path.clone(), source))?;
        let dym_text = std::fs::read_to_string(&dym_path)
            .map_err(|source| crate::IoError::io(dym_path.clone(), source))?;
        let output_text = std::fs::read_to_string(&output_path)
            .map_err(|source| crate::IoError::io(output_path.clone(), source))?;
        let DmdwInput::Enabled(calculation) = DmdwInput::parse_str(&input_path, &input_text)?
        else {
            return Err(crate::IoError::Parse {
                path: input_path,
                line: 0,
                message: "expected enabled DMDW reference input".to_string(),
            });
        };
        let dym = crate::parse_dym(&dym_text)?;
        let positions = dym.coordinates.cartesian_positions();
        let descriptors = calculation
            .paths
            .iter()
            .map(|path| DmdwPathDescriptor {
                selectors: std::iter::once(path.absorber_selector)
                    .chain(path.potentials.iter().copied())
                    .collect(),
                max_effective_length: path.max_distance * DMDW_ANGSTROM_TO_BOHR,
            })
            .collect::<Vec<_>>();
        let expanded =
            dmdw_expand_path_descriptors(positions.view(), &descriptors).map_err(|source| {
                crate::IoError::Parse {
                    path: "dmdw.inp".into(),
                    line: 0,
                    message: source.to_string(),
                }
            })?;
        let output = crate::parse_dmdw_out(&output_text)?;
        let expected_paths = output
            .sections
            .iter()
            .filter_map(|section| match &section.subject {
                crate::DmdwOutSubject::PathIndices(indices) => Some(indices.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let actual_paths = expanded
            .iter()
            .map(|path| path.atoms.iter().map(|&atom| atom + 1).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        assert_eq!(actual_paths, expected_paths);
        Ok(())
    }

    #[test]
    fn rejects_invalid_dmdw_rendering() {
        let input = DmdwInput::Enabled(super::DmdwCalculation {
            run: 1,
            order: 6,
            temperature_flag: 1,
            temperature: 450.0,
            temperature_max: None,
            calculation_type: 7,
            dym_file: "feff.dym".to_string(),
            path_count: 2,
            paths: vec![DmdwPath {
                leg_count: 2,
                absorber_selector: 0,
                potentials: vec![0],
                max_distance: 10.0,
            }],
        });
        assert!(dmdw_input_string(&input).is_err());
    }

    fn minimal_dym_text() -> &'static str {
        concat!(
            "    1\n",
            "    1\n",
            "   29\n",
            "   63.546000\n",
            "    0.00000000    0.00000000    0.00000000\n",
            "    1    1\n",
            "  1.000000E+00  0.000000E+00  0.000000E+00\n",
            "  0.000000E+00  1.000000E+00  0.000000E+00\n",
            "  0.000000E+00  0.000000E+00  1.000000E+00\n",
        )
    }

    fn reference_dmdw_test_dir() -> crate::Result<Option<std::path::PathBuf>> {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(std::path::Path::parent)
            .ok_or_else(|| crate::IoError::Parse {
                path: manifest_dir.into(),
                line: 0,
                message: "failed to find workspace root".to_string(),
            })?;
        let path = workspace
            .join("feff10")
            .join("src")
            .join("DMDW")
            .join("Test");
        let required = [
            "H2O.g03.dmdw.inp",
            "H2O.g03.dym",
            "Reference_Results/H2O.g03.dmdw.out",
        ];
        Ok(required
            .iter()
            .all(|name| path.join(name).is_file())
            .then_some(path))
    }
}
