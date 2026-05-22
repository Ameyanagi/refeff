use crate::{IoError, Result};

use super::validate::validate_dmdw_calculation;
use super::{DmdwCalculation, DmdwInput, DmdwPath, DmdwPdosOptions};

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
    out.push_str(&dmdw_type_line(calculation));
    if let Some(options) = &calculation.self_energy_options {
        out.push_str(&format!("{:4}\n", options.displacement_option));
        out.push_str(&format!(
            "{:4}{:11.3}\n",
            options.energy_option, options.electron_energy
        ));
    }
    out.push_str(&calculation.dym_file);
    out.push('\n');
    if let Some(options) = &calculation.self_energy_options {
        out.push_str(&options.pds_file);
        out.push('\n');
        out.push_str(&options.a2f_file);
        out.push('\n');
        return Ok(out);
    }
    out.push_str(&format!("{:4}\n", calculation.path_count));
    for path in &calculation.paths {
        out.push_str(&dmdw_path_line(path)?);
    }
    Ok(out)
}

fn dmdw_type_line(calculation: &DmdwCalculation) -> String {
    if calculation.calculation_type != 5 {
        return format!("{:4}\n", calculation.calculation_type);
    }

    let options = calculation.pdos_options.clone().unwrap_or_default();
    if options == DmdwPdosOptions::default() {
        return format!("{:4}\n", calculation.calculation_type);
    }

    match options.format {
        0 => format!(
            "{:4}{:4}{:>4}\n",
            calculation.calculation_type,
            options.format,
            fortran_bool(options.write_partial)
        ),
        1 => format!(
            "{:4}{:4}{:>4}{:>4}\n",
            calculation.calculation_type,
            options.format,
            fortran_bool(options.write_partial),
            fortran_bool(options.drop_left_edges)
        ),
        2 => format!(
            "{:4}{:4}{:>4}{:11.3}{:11.3}\n",
            calculation.calculation_type,
            options.format,
            fortran_bool(options.write_partial),
            options.gaussian_broadening_thz,
            options.gaussian_resolution_thz
        ),
        10 => format!(
            "{:4}{:4}{:>4}{:>4}{:11.3}{:11.3}\n",
            calculation.calculation_type,
            options.format,
            fortran_bool(options.write_partial),
            fortran_bool(options.drop_left_edges),
            options.gaussian_broadening_thz,
            options.gaussian_resolution_thz
        ),
        _ => format!("{:4}\n", calculation.calculation_type),
    }
}

fn fortran_bool(value: bool) -> &'static str {
    if value { "T" } else { "F" }
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
