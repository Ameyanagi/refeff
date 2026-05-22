use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use refeff_core::DmdwPoleWeightedA2f;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

use super::common::{
    DMDW_A2F_INFO_PATH, DMDW_A2F_INFO_ROW_WIDTH, is_numeric_token, parse_error, parse_error_value,
    parse_i32_field, parse_single_value, parse_two_column_row,
};
use super::types::DmdwA2fInfoData;
use super::validate::validate_dmdw_a2f_info;

/// Build `dmdw_a2f.info` data from the core DMDW pole-weight diagnostic.
pub fn dmdw_a2f_info_from_pole_weighted(
    calculation_type: i32,
    displacement_option: i32,
    lanczos_order: usize,
    diagnostic: &DmdwPoleWeightedA2f,
) -> Result<DmdwA2fInfoData> {
    let data = DmdwA2fInfoData {
        calculation_type,
        displacement_option,
        lanczos_order,
        lanczos_frequency_thz: diagnostic.lanczos_frequency_thz.clone(),
        lanczos_weight: diagnostic.lanczos_weight.clone(),
        normalization: diagnostic.normalization,
        pole_energy_ev: diagnostic.pole_energy_ev.clone(),
        pole_weight: diagnostic.pole_weight.clone(),
        mass_enhancement: diagnostic.mass_enhancement,
        characteristic_energy_ev: diagnostic.characteristic_energy_ev,
    };
    validate_dmdw_a2f_info(&data)?;
    Ok(data)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum A2fInfoSection {
    Header,
    LanczosPoles,
    A2fPoles,
}

/// Parse FEFF `dmdw_a2f.info` text.
pub fn parse_dmdw_a2f_info(text: &str) -> Result<DmdwA2fInfoData> {
    let mut calculation_type = None;
    let mut displacement_option = None;
    let mut lanczos_order = None;
    let mut lanczos_frequency_thz = Vec::new();
    let mut lanczos_weight = Vec::new();
    let mut normalization = None;
    let mut pole_energy_ev = Vec::new();
    let mut pole_weight = Vec::new();
    let mut mass_enhancement = None;
    let mut characteristic_energy_ev = None;
    let mut section = A2fInfoSection::Header;

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("# DMDW Option") {
            calculation_type = Some(parse_i32_field(
                DMDW_A2F_INFO_PATH,
                line_number,
                "DMDW option",
                line,
            )?);
        } else if line.starts_with("# Displacement Option") {
            displacement_option = Some(parse_i32_field(
                DMDW_A2F_INFO_PATH,
                line_number,
                "displacement option",
                line,
            )?);
        } else if line.starts_with("# Lanczos Order") {
            let order = parse_i32_field(DMDW_A2F_INFO_PATH, line_number, "Lanczos order", line)?;
            if order <= 0 {
                return parse_error(
                    DMDW_A2F_INFO_PATH,
                    line_number,
                    "Lanczos order must be positive",
                );
            }
            lanczos_order = Some(usize::try_from(order).map_err(|_| {
                parse_error_value(DMDW_A2F_INFO_PATH, line_number, "invalid Lanczos order")
            })?);
        } else if line.starts_with("# Lanczos Pole") {
            section = A2fInfoSection::LanczosPoles;
        } else if line.starts_with("# norm") {
            normalization = Some(parse_single_value(
                DMDW_A2F_INFO_PATH,
                line_number,
                "normalization",
                line,
            )?);
        } else if line.starts_with("Pole/weight a2f") {
            section = A2fInfoSection::A2fPoles;
        } else if line.starts_with("lambda") {
            mass_enhancement = Some(parse_single_value(
                DMDW_A2F_INFO_PATH,
                line_number,
                "lambda",
                line,
            )?);
        } else if line.starts_with("w0") {
            characteristic_energy_ev = Some(parse_single_value(
                DMDW_A2F_INFO_PATH,
                line_number,
                "w0",
                line,
            )?);
        } else if line.starts_with('#') {
            continue;
        } else if is_numeric_token(line.split_whitespace().next().unwrap_or_default()) {
            let (first, second) = parse_two_column_row(
                DMDW_A2F_INFO_PATH,
                line_number,
                line,
                DMDW_A2F_INFO_ROW_WIDTH,
                "dmdw_a2f.info",
            )?;
            match section {
                A2fInfoSection::LanczosPoles => {
                    lanczos_frequency_thz.push(first);
                    lanczos_weight.push(second);
                }
                A2fInfoSection::A2fPoles => {
                    pole_energy_ev.push(first);
                    pole_weight.push(second);
                }
                A2fInfoSection::Header => {
                    return parse_error(
                        DMDW_A2F_INFO_PATH,
                        line_number,
                        "numeric row appeared before a pole-table heading",
                    );
                }
            }
        } else {
            return parse_error(
                DMDW_A2F_INFO_PATH,
                line_number,
                format!("unrecognized dmdw_a2f.info line {line:?}"),
            );
        }
    }

    let data = DmdwA2fInfoData {
        calculation_type: calculation_type
            .ok_or_else(|| parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing DMDW option"))?,
        displacement_option: displacement_option.ok_or_else(|| {
            parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing displacement option")
        })?,
        lanczos_order: lanczos_order
            .ok_or_else(|| parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing Lanczos order"))?,
        lanczos_frequency_thz: Array1::from_vec(lanczos_frequency_thz),
        lanczos_weight: Array1::from_vec(lanczos_weight),
        normalization: normalization
            .ok_or_else(|| parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing normalization"))?,
        pole_energy_ev: Array1::from_vec(pole_energy_ev),
        pole_weight: Array1::from_vec(pole_weight),
        mass_enhancement: mass_enhancement
            .ok_or_else(|| parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing lambda"))?,
        characteristic_energy_ev: characteristic_energy_ev
            .ok_or_else(|| parse_error_value(DMDW_A2F_INFO_PATH, 0, "missing w0"))?,
    };
    validate_dmdw_a2f_info(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_a2f.info` text.
pub fn dmdw_a2f_info_string(data: &DmdwA2fInfoData) -> Result<String> {
    validate_dmdw_a2f_info(data)?;

    let mut out = String::new();
    writeln!(out, "# DMDW Option {}", data.calculation_type)?;
    writeln!(out, "# Displacement Option {}", data.displacement_option)?;
    writeln!(out, "# Lanczos Order {}", data.lanczos_order)?;
    writeln!(out, "# ")?;
    writeln!(out, "# Lanczos Pole in Thz/weight PHDOS")?;
    for (&frequency, &weight) in data
        .lanczos_frequency_thz
        .iter()
        .zip(data.lanczos_weight.iter())
    {
        write_fortran_zero_scaled_exp(&mut out, frequency, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, weight, 20, 10)?;
        out.push('\n');
    }
    write!(out, "# norm")?;
    write_fortran_zero_scaled_exp(&mut out, data.normalization, 20, 10)?;
    out.push_str("\n\n");
    writeln!(out, "Pole/weight a2f in eV/Arb")?;
    for (&energy, &weight) in data.pole_energy_ev.iter().zip(data.pole_weight.iter()) {
        write_fortran_zero_scaled_exp(&mut out, energy, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, weight, 20, 10)?;
        out.push('\n');
    }
    write!(out, "lambda =")?;
    write_fortran_zero_scaled_exp(&mut out, data.mass_enhancement, 20, 10)?;
    out.push('\n');
    write!(out, "w0 =")?;
    write_fortran_zero_scaled_exp(&mut out, data.characteristic_energy_ev, 20, 10)?;
    out.push('\n');
    Ok(out)
}

/// Read FEFF `dmdw_a2f.info` from disk.
pub fn read_dmdw_a2f_info(path: impl AsRef<Path>) -> Result<DmdwA2fInfoData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_a2f_info(&text)
}

/// Write FEFF `dmdw_a2f.info` text to disk.
pub fn write_dmdw_a2f_info(path: impl AsRef<Path>, data: &DmdwA2fInfoData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_a2f_info_string(data)?).map_err(|source| IoError::io(path, source))
}
