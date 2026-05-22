use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::{IoError, Result};

use super::{DmdwCalculation, DmdwInput, DmdwPath, DmdwPdosOptions, DmdwSelfEnergyOptions};

impl DmdwInput {
    /// Parse a FEFF `dmdw.inp` string.
    pub fn parse_str(source: impl Into<PathBuf>, text: &str) -> Result<Self> {
        let mut parser = DmdwInputParser::new(source.into(), text);
        parser.parse()
    }
}

struct DmdwSelfEnergyPrefix {
    displacement_option: i32,
    energy_option: i32,
    electron_energy: f64,
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
        let (calculation_type, pdos_options) = self.parse_calculation_type()?;
        let self_energy_prefix = if calculation_type == 2 {
            Some(self.parse_self_energy_prefix()?)
        } else {
            None
        };
        let (_, dym_file) = self.next_line("DMDW dynamical-matrix filename")?;
        let (self_energy_options, path_count, paths) = if let Some(prefix) = self_energy_prefix {
            let (_, pds_file) = self.next_line("DMDW self-energy PDS filename")?;
            let (_, a2f_file) = self.next_line("DMDW self-energy a2f filename")?;
            (
                Some(DmdwSelfEnergyOptions {
                    displacement_option: prefix.displacement_option,
                    energy_option: prefix.energy_option,
                    electron_energy: prefix.electron_energy,
                    pds_file: pds_file.trim().to_string(),
                    a2f_file: a2f_file.trim().to_string(),
                }),
                0,
                Vec::new(),
            )
        } else {
            let path_count = self.parse_values::<usize>(1, "DMDW path-count line")?[0];
            let mut paths = Vec::with_capacity(path_count);
            for _ in 0..path_count {
                paths.push(self.parse_path()?);
            }
            (None, path_count, paths)
        };

        Ok(DmdwInput::Enabled(DmdwCalculation {
            run,
            order,
            temperature_flag,
            temperature,
            temperature_max,
            calculation_type,
            self_energy_options,
            pdos_options,
            dym_file: dym_file.trim().to_string(),
            path_count,
            paths,
        }))
    }

    fn parse_calculation_type(&mut self) -> Result<(i32, Option<DmdwPdosOptions>)> {
        let (line_number, line) = self.next_line("DMDW type line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            return Err(self.parse_error(line_number, "DMDW type line requires 1 field"));
        }
        let calculation_type = parse_field(&self.source, line_number, fields[0])?;
        if calculation_type != 5 {
            return Ok((calculation_type, None));
        }

        let mut options = DmdwPdosOptions::default();
        if let Some(format) = fields.get(1) {
            options.format = parse_field(&self.source, line_number, format)?;
        }
        match options.format {
            0 => {
                if let Some(partial) = fields.get(2) {
                    options.write_partial = parse_fortran_bool(line_number, partial)?;
                }
            }
            1 => {
                if let Some(partial) = fields.get(2) {
                    options.write_partial = parse_fortran_bool(line_number, partial)?;
                }
                if let Some(drop_left_edges) = fields.get(3) {
                    options.drop_left_edges = parse_fortran_bool(line_number, drop_left_edges)?;
                }
            }
            2 => {
                if let Some(partial) = fields.get(2) {
                    options.write_partial = parse_fortran_bool(line_number, partial)?;
                }
                if let Some(broadening) = fields.get(3) {
                    options.gaussian_broadening_thz =
                        parse_field(&self.source, line_number, broadening)?;
                }
                if let Some(resolution) = fields.get(4) {
                    options.gaussian_resolution_thz =
                        parse_field(&self.source, line_number, resolution)?;
                }
            }
            10 => {
                if let Some(partial) = fields.get(2) {
                    options.write_partial = parse_fortran_bool(line_number, partial)?;
                }
                if let Some(drop_left_edges) = fields.get(3) {
                    options.drop_left_edges = parse_fortran_bool(line_number, drop_left_edges)?;
                }
                if let Some(broadening) = fields.get(4) {
                    options.gaussian_broadening_thz =
                        parse_field(&self.source, line_number, broadening)?;
                }
                if let Some(resolution) = fields.get(5) {
                    options.gaussian_resolution_thz =
                        parse_field(&self.source, line_number, resolution)?;
                }
            }
            _ => {
                return Err(self.parse_error(
                    line_number,
                    format!(
                        "DMDW PDOS format {} is not supported by FEFF",
                        options.format
                    ),
                ));
            }
        }
        Ok((calculation_type, Some(options)))
    }

    fn parse_self_energy_prefix(&mut self) -> Result<DmdwSelfEnergyPrefix> {
        let displacement_option =
            self.parse_values::<i32>(1, "DMDW self-energy displacement line")?[0];
        let (line_number, line) = self.next_line("DMDW self-energy electron-energy line")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(self.parse_error(
                line_number,
                "DMDW self-energy electron-energy line requires 2 fields",
            ));
        }
        Ok(DmdwSelfEnergyPrefix {
            displacement_option,
            energy_option: parse_field(&self.source, line_number, fields[0])?,
            electron_energy: parse_field(&self.source, line_number, fields[1])?,
        })
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

fn parse_fortran_bool(line: usize, field: &str) -> Result<bool> {
    match field.trim().to_ascii_lowercase().as_str() {
        "t" | ".t." | "true" | ".true." | "1" => Ok(true),
        "f" | ".f." | "false" | ".false." | "0" => Ok(false),
        _ => Err(IoError::Parse {
            path: "dmdw.inp".into(),
            line,
            message: format!("invalid DMDW logical field {field:?}"),
        }),
    }
}
