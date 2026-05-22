//! Validation for structured FEFF `dmdw.out` reports.

use crate::error::Result;

use super::common::parse_error;
use super::types::{
    DmdwOutData, DmdwOutHeader, DmdwOutSection, DmdwOutSubject, DmdwOutTemperature,
    DmdwOutTemperatureValue,
};

pub(super) fn validate_dmdw_out(data: &DmdwOutData) -> Result<()> {
    match &data.header {
        Some(header) => {
            if header.lanczos_recursion_order == 0 {
                return parse_error(1, "Lanczos recursion order must be positive");
            }
            validate_temperature(&header.temperature)?;
            if header.dynamical_matrix_file.trim().is_empty() {
                return parse_error(3, "dynamical matrix file must not be empty");
            }
            if data.mass_enhancement_header && !data.sections.is_empty() {
                return parse_error(
                    0,
                    "DMDW mass enhancement header must not be mixed with report sections",
                );
            }
            if data.sections.is_empty() && !data.mass_enhancement_header {
                return parse_error(0, "non-empty dmdw.out must contain at least one section");
            }
            for (index, section) in data.sections.iter().enumerate() {
                validate_section(index + 1, header, section)?;
            }
        }
        None if data.mass_enhancement_header || !data.sections.is_empty() => {
            return parse_error(0, "DMDW output content requires a dmdw.out header");
        }
        None => {}
    }
    Ok(())
}

fn validate_section(index: usize, header: &DmdwOutHeader, section: &DmdwOutSection) -> Result<()> {
    validate_subject(index, &section.subject)?;
    if !matches!(section.subject, DmdwOutSubject::TotalPdos)
        && !section.pdos_poles.is_empty()
        && section.pdos_poles.len() != header.lanczos_recursion_order
    {
        return parse_error(
            index,
            format!(
                "section has {} PDOS pole(s), expected {} from the header",
                section.pdos_poles.len(),
                header.lanczos_recursion_order
            ),
        );
    }
    for pole in &section.pdos_poles {
        validate_finite("pole_frequency_thz", pole.frequency_thz, index)?;
        validate_finite("pole_weight", pole.weight, index)?;
    }
    if let Some(einstein) = &section.einstein {
        validate_finite("einstein_frequency_thz", einstein.frequency_thz, index)?;
        validate_finite(
            "einstein_temperature_kelvin",
            einstein.temperature_kelvin,
            index,
        )?;
        validate_finite(
            "einstein_effective_force_constant_n_per_m",
            einstein.effective_force_constant_n_per_m,
            index,
        )?;
    }
    for moment in &section.moments {
        validate_finite("moment_thz_power_n", moment.moment_thz_power_n, index)?;
        validate_optional_finite("moment_frequency_thz", moment.frequency_thz, index)?;
        validate_optional_finite(
            "moment_temperature_kelvin",
            moment.temperature_kelvin,
            index,
        )?;
        validate_optional_finite(
            "moment_effective_force_constant_n_per_m",
            moment.effective_force_constant_n_per_m,
            index,
        )?;
    }
    validate_optional_finite("reduced_mass_amu", section.reduced_mass_amu, index)?;
    validate_optional_finite("path_length_angstrom", section.path_length_angstrom, index)?;
    validate_optional_finite(
        "sigma2_1e_minus_3_angstrom2",
        section.sigma2_1e_minus_3_angstrom2,
        index,
    )?;
    validate_temperature_values("sigma2", &section.sigma2_by_temperature, index)?;
    validate_optional_finite(
        "vibrational_free_energy_ev",
        section.vibrational_free_energy_ev,
        index,
    )?;
    validate_temperature_values(
        "vibrational_free_energy_ev",
        &section.vibrational_free_energy_by_temperature,
        index,
    )?;
    validate_optional_finite(
        "u2_1e_minus_3_angstrom2",
        section.u2_1e_minus_3_angstrom2,
        index,
    )?;
    validate_temperature_values("u2", &section.u2_by_temperature, index)?;
    Ok(())
}

fn validate_subject(index: usize, subject: &DmdwOutSubject) -> Result<()> {
    match subject {
        DmdwOutSubject::PathIndices(indices) | DmdwOutSubject::AtomIndex { indices, .. } => {
            if indices.is_empty() {
                return parse_error(index, "section index list must not be empty");
            }
            if indices.contains(&0) {
                return parse_error(index, "section indices must be one-based positive values");
            }
        }
        DmdwOutSubject::TotalPdos | DmdwOutSubject::TotalVfe => {}
    }
    Ok(())
}

fn validate_temperature(temperature: &DmdwOutTemperature) -> Result<()> {
    match temperature {
        DmdwOutTemperature::Single(value) => validate_finite("temperature", *value, 2),
        DmdwOutTemperature::ListedBelow => Ok(()),
    }
}

fn validate_temperature_values(
    field: &'static str,
    rows: &[DmdwOutTemperatureValue],
    section_index: usize,
) -> Result<()> {
    for row in rows {
        validate_finite("temperature_kelvin", row.temperature_kelvin, section_index)?;
        validate_finite(field, row.value, section_index)?;
    }
    Ok(())
}

fn validate_optional_finite(field: &'static str, value: Option<f64>, line: usize) -> Result<()> {
    match value {
        Some(value) => validate_finite(field, value, line),
        None => Ok(()),
    }
}

fn validate_finite(field: &'static str, value: f64, line: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(line, format!("{field} must be finite"))
    }
}
