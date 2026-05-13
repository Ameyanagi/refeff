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
    /// Sample temperature.
    pub temperature: f64,
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
    /// Maximum path distance in Bohr-like FEFF units.
    pub max_distance: f64,
}

impl DmdwInput {
    /// Parse a FEFF `dmdw.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = DmdwInputParser::new(source.into(), text);
        parser.parse()
    }
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
        let (temperature_flag, temperature) = self.parse_temperature()?;
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
            calculation_type,
            dym_file: dym_file.trim().to_string(),
            path_count,
            paths,
        }))
    }

    fn parse_temperature(&mut self) -> Result<(i32, f64)> {
        let (line_number, line) = self.next_line("DMDW temperature line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(line_number, "DMDW temperature line requires 2 fields"));
        }
        Ok((
            parse_field(&self.source, line_number, fields[0])?,
            parse_field(&self.source, line_number, fields[1])?,
        ))
    }

    fn parse_path(&mut self) -> Result<DmdwPath> {
        let (line_number, line) = self.next_line("DMDW path row")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(self.parse_error(line_number, "DMDW path row requires at least 4 fields"));
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

    use super::{DmdwInput, DmdwPath};

    #[test]
    fn parses_disabled_dmdw_input() -> crate::Result<()> {
        let input = FeffInput::parse_str("feff.inp", "END\n")?;
        let document = FeffDocument::from_input(&input)?;
        let dmdw = DmdwInput::parse_str("dmdw.inp", &rdinp::dmdw_inp_string(&document)?)?;
        assert_eq!(dmdw, DmdwInput::Disabled);
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
}
