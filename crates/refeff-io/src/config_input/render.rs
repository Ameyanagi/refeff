use std::fmt::Write as _;
use std::path::Path;

use crate::error::{IoError, Result};

use super::parse::parse_config_inp;
use super::types::{CONFIG_RECORD_WIDTH, ConfigInput, ConfigRecord};
use super::validate::validate_config_input;

pub fn config_inp_string(input: &ConfigInput) -> Result<String> {
    validate_config_input(input)?;
    let lines = input
        .records
        .iter()
        .map(config_record_line)
        .collect::<Result<Vec<_>>>()?;
    config_inp_lines_string(&lines)
}

/// Render raw `CONFIG card` payload lines as FEFF `config.inp` records.
pub fn config_inp_lines_string(lines: &[String]) -> Result<String> {
    let mut out = String::new();
    for line in lines {
        write_config_record_line(&mut out, line)?;
    }
    Ok(out)
}

pub fn write_config_inp(path: impl AsRef<Path>, input: &ConfigInput) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, config_inp_string(input)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `config.inp` text from a file.
pub fn read_config_inp(path: impl AsRef<Path>) -> Result<ConfigInput> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_config_inp(&text)
}

fn config_record_line(record: &ConfigRecord) -> Result<String> {
    let mut line = format!("{} {}", record.potential_index, record.element);
    if let Some(noble_gas) = &record.noble_gas {
        write!(line, " {noble_gas}")?;
    }
    for state in &record.states {
        write!(line, " {}", state.orbital)?;
        for occupation in &state.occupations {
            write!(line, " {}", occupation.occupation)?;
            if let Some(spin) = occupation.spin {
                write!(line, " s {spin}")?;
            }
        }
    }
    Ok(line)
}

fn write_config_record_line(out: &mut String, line: &str) -> Result<()> {
    if line.len() > CONFIG_RECORD_WIDTH {
        return Err(IoError::InvalidConfigInp {
            field: "record",
            message: format!(
                "record length {} exceeds FEFF width {CONFIG_RECORD_WIDTH}",
                line.len()
            ),
        });
    }
    writeln!(out, "{line:<width$}", width = CONFIG_RECORD_WIDTH)?;
    Ok(())
}
