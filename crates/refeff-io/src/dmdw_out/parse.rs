//! Parser for FEFF `dmdw.out` reports.

use crate::error::Result;

use super::common::{parse_error, parse_error_value};
use super::types::{
    DmdwOutData, DmdwOutEinstein, DmdwOutHeader, DmdwOutMoment, DmdwOutPole, DmdwOutSection,
    DmdwOutSubject, DmdwOutTemperature, DmdwOutTemperatureValue,
};
use super::validate::validate_dmdw_out;

/// Parse FEFF `dmdw.out` text.
pub fn parse_dmdw_out(text: &str) -> Result<DmdwOutData> {
    if text.trim().is_empty() {
        return Ok(DmdwOutData {
            header: None,
            mass_enhancement_header: false,
            sections: Vec::new(),
        });
    }

    let lines = text
        .lines()
        .enumerate()
        .map(|(index, raw)| (index + 1, raw.trim()))
        .collect::<Vec<_>>();
    let mut position = 0;
    skip_blank_lines(&lines, &mut position);
    let header = parse_header(&lines, &mut position)?;
    let mass_enhancement_header = parse_mass_enhancement_header(&lines, &mut position);

    let mut sections = Vec::new();
    while position < lines.len() {
        skip_section_gap(&lines, &mut position);
        if position >= lines.len() {
            break;
        }
        sections.push(parse_section(&lines, &mut position)?);
    }

    let data = DmdwOutData {
        header: Some(header),
        mass_enhancement_header,
        sections,
    };
    validate_dmdw_out(&data)?;
    Ok(data)
}

fn parse_header(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutHeader> {
    let (order_line, order_text) = next_line(lines, position, "Lanczos recursion order")?;
    let order = parse_prefixed_usize(order_line, order_text, "# Lanczos recursion order:")?;

    let (temperature_line, temperature_text) = next_line(lines, position, "temperature")?;
    let temperature_value =
        parse_prefixed_tail(temperature_line, temperature_text, "# Temperature:")?;
    let temperature = if temperature_value.eq_ignore_ascii_case("(See list Below)") {
        DmdwOutTemperature::ListedBelow
    } else {
        DmdwOutTemperature::Single(parse_f64(
            temperature_line,
            "temperature",
            temperature_value,
        )?)
    };

    let (dym_line, dym_text) = next_line(lines, position, "dynamical matrix file")?;
    let dynamical_matrix_file =
        parse_prefixed_tail(dym_line, dym_text, "# Dynamical matrix file:")?.to_owned();

    Ok(DmdwOutHeader {
        lanczos_recursion_order: order,
        temperature,
        dynamical_matrix_file,
    })
}

fn parse_mass_enhancement_header(lines: &[(usize, &str)], position: &mut usize) -> bool {
    if *position >= lines.len() {
        return false;
    }
    if lines[*position].1 == "Mass Enchancement Factor" {
        *position += 1;
        true
    } else {
        false
    }
}

fn parse_section(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutSection> {
    let (line_number, line) = lines[*position];
    let subject = parse_subject(line_number, line)?;
    *position += 1;

    let mut section = DmdwOutSection::new(subject);
    if matches!(section.subject, DmdwOutSubject::AtomIndex { .. }) {
        parse_optional_direction(lines, position, &mut section)?;
    }

    while *position < lines.len() {
        let (current_line, current_text) = lines[*position];
        if current_text.is_empty() || is_separator_line(current_text) {
            *position += 1;
            continue;
        }
        if is_section_start(current_text) {
            break;
        }

        if current_text.starts_with("PDOS Poles:") {
            *position += 1;
            section.pdos_poles = parse_pdos_poles(lines, position)?;
        } else if current_text.starts_with("PDOS Einstein freq") {
            *position += 1;
            section.einstein = Some(parse_einstein(lines, position)?);
        } else if current_text.starts_with("pDOS n Moments") {
            *position += 1;
            section.moments = parse_moments(lines, position)?;
        } else if current_text.starts_with("Path Red. Mass (AMU):") {
            section.reduced_mass_amu = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "reduced_mass_amu",
            )?);
            *position += 1;
        } else if current_text.starts_with("Path Length (Ang), s^2") {
            let values = parse_values_after_colon(current_line, current_text)?;
            if values.len() != 2 {
                return parse_error(
                    current_line,
                    format!(
                        "path length/sigma2 row has {} value(s), expected 2",
                        values.len()
                    ),
                );
            }
            section.path_length_angstrom = Some(values[0]);
            section.sigma2_1e_minus_3_angstrom2 = Some(values[1]);
            *position += 1;
        } else if current_text.starts_with("Path Length (Ang):") {
            section.path_length_angstrom = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "path_length_angstrom",
            )?);
            *position += 1;
            skip_blank_lines(lines, position);
            if *position < lines.len() && lines[*position].1.starts_with("Temp (K)") {
                *position += 1;
                section.sigma2_by_temperature =
                    parse_temperature_values(lines, position, "sigma2")?;
            }
        } else if current_text.starts_with("VFE (eV):") {
            section.vibrational_free_energy_ev = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "vibrational_free_energy_ev",
            )?);
            *position += 1;
        } else if current_text.starts_with("Temp (K)") && current_text.contains("VFE") {
            *position += 1;
            section.vibrational_free_energy_by_temperature =
                parse_temperature_values(lines, position, "vibrational_free_energy_ev")?;
        } else if current_text.starts_with("u^2 (1e-3 Ang^2):") {
            section.u2_1e_minus_3_angstrom2 = Some(parse_single_value_after_colon(
                current_line,
                current_text,
                "u2_1e_minus_3_angstrom2",
            )?);
            *position += 1;
        } else if current_text.starts_with("Temp (K)") && current_text.contains("u^2") {
            *position += 1;
            section.u2_by_temperature = parse_temperature_values(lines, position, "u2")?;
        } else if current_text.starts_with("Projected DOS component computed.") {
            section.projected_dos_component_computed = true;
            *position += 1;
        } else {
            return parse_error(
                current_line,
                format!("unexpected DMDW output line {current_text:?}"),
            );
        }
    }

    Ok(section)
}

fn parse_subject(line_number: usize, line: &str) -> Result<DmdwOutSubject> {
    if let Some(tail) = line.strip_prefix("Path Indices:") {
        return Ok(DmdwOutSubject::PathIndices(parse_index_list(
            line_number,
            tail,
        )?));
    }
    if let Some(tail) = line.strip_prefix("Atom Index:") {
        return Ok(DmdwOutSubject::AtomIndex {
            indices: parse_index_list(line_number, tail)?,
            direction: None,
        });
    }
    if line == "Total PDOS results:" {
        return Ok(DmdwOutSubject::TotalPdos);
    }
    if line == "Total VFE for the paths requested:" {
        return Ok(DmdwOutSubject::TotalVfe);
    }
    parse_error(
        line_number,
        format!("expected DMDW section header, found {line:?}"),
    )
}

fn parse_optional_direction(
    lines: &[(usize, &str)],
    position: &mut usize,
    section: &mut DmdwOutSection,
) -> Result<()> {
    if *position >= lines.len() {
        return Ok(());
    }
    let (line_number, line) = lines[*position];
    if !line.contains("Direction") {
        return Ok(());
    }
    let mut tokens = line.split_whitespace();
    let direction = tokens
        .find(|token| token.eq_ignore_ascii_case("Direction"))
        .and_then(|_| tokens.next())
        .filter(|token| token.chars().any(|ch| ch != '-'))
        .map(ToOwned::to_owned)
        .ok_or_else(|| parse_error_value(line_number, "missing atom perturbation direction"))?;
    if let DmdwOutSubject::AtomIndex {
        direction: slot, ..
    } = &mut section.subject
    {
        *slot = Some(direction);
    }
    *position += 1;
    Ok(())
}

fn parse_pdos_poles(lines: &[(usize, &str)], position: &mut usize) -> Result<Vec<DmdwOutPole>> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with("Freq.") {
        *position += 1;
    }

    let mut poles = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return parse_error(
                line_number,
                format!("PDOS pole row has {} token(s), expected 2", tokens.len()),
            );
        }
        poles.push(DmdwOutPole {
            frequency_thz: parse_f64(line_number, "pole_frequency_thz", tokens[0])?,
            weight: parse_f64(line_number, "pole_weight", tokens[1])?,
        });
        *position += 1;
    }
    Ok(poles)
}

fn parse_einstein(lines: &[(usize, &str)], position: &mut usize) -> Result<DmdwOutEinstein> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with("Freq (THz)") {
        *position += 1;
    }
    let (line_number, line) = next_data_line(lines, position, "Einstein row")?;
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 3 {
        return parse_error(
            line_number,
            format!("Einstein row has {} token(s), expected 3", tokens.len()),
        );
    }
    Ok(DmdwOutEinstein {
        frequency_thz: parse_f64(line_number, "einstein_frequency_thz", tokens[0])?,
        temperature_kelvin: parse_f64(line_number, "einstein_temperature_kelvin", tokens[1])?,
        effective_force_constant_n_per_m: parse_f64(
            line_number,
            "einstein_effective_force_constant_n_per_m",
            tokens[2],
        )?,
    })
}

fn parse_moments(lines: &[(usize, &str)], position: &mut usize) -> Result<Vec<DmdwOutMoment>> {
    skip_blank_lines(lines, position);
    if *position < lines.len() && lines[*position].1.starts_with('n') {
        *position += 1;
    }

    let mut moments = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 2 || tokens.len() > 5 {
            return parse_error(
                line_number,
                format!("moment row has {} token(s), expected 2 to 5", tokens.len()),
            );
        }
        moments.push(DmdwOutMoment {
            order: parse_i32(line_number, "moment_order", tokens[0])?,
            moment_thz_power_n: parse_f64(line_number, "moment_thz_power_n", tokens[1])?,
            frequency_thz: parse_optional_f64(
                line_number,
                "moment_frequency_thz",
                tokens.get(2).copied(),
            )?,
            temperature_kelvin: parse_optional_f64(
                line_number,
                "moment_temperature_kelvin",
                tokens.get(3).copied(),
            )?,
            effective_force_constant_n_per_m: parse_optional_f64(
                line_number,
                "moment_effective_force_constant_n_per_m",
                tokens.get(4).copied(),
            )?,
        });
        *position += 1;
    }
    Ok(moments)
}

fn parse_temperature_values(
    lines: &[(usize, &str)],
    position: &mut usize,
    value_field: &'static str,
) -> Result<Vec<DmdwOutTemperatureValue>> {
    let mut rows = Vec::new();
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        if line.is_empty() {
            *position += 1;
            break;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            break;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return parse_error(
                line_number,
                format!(
                    "temperature result row has {} token(s), expected 2",
                    tokens.len()
                ),
            );
        }
        rows.push(DmdwOutTemperatureValue {
            temperature_kelvin: parse_f64(line_number, "temperature_kelvin", tokens[0])?,
            value: parse_f64(line_number, value_field, tokens[1])?,
        });
        *position += 1;
    }
    Ok(rows)
}

fn next_data_line<'a>(
    lines: &'a [(usize, &'a str)],
    position: &mut usize,
    description: &'static str,
) -> Result<(usize, &'a str)> {
    while *position < lines.len() {
        let (line_number, line) = lines[*position];
        *position += 1;
        if line.is_empty() {
            continue;
        }
        if is_section_start(line) || is_section_gap_or_label(line) {
            return parse_error(line_number, format!("missing {description}"));
        }
        return Ok((line_number, line));
    }
    parse_error(0, format!("missing {description}"))
}

fn parse_prefixed_usize(line: usize, text: &str, prefix: &'static str) -> Result<usize> {
    let token = parse_prefixed_tail(line, text, prefix)?;
    parse_usize(line, "lanczos_recursion_order", token)
}

fn parse_prefixed_tail<'a>(line: usize, text: &'a str, prefix: &'static str) -> Result<&'a str> {
    text.strip_prefix(prefix)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| parse_error_value(line, format!("expected {prefix:?} line")))
}

fn parse_index_list(line: usize, tail: &str) -> Result<Vec<usize>> {
    let indices = tail
        .split_whitespace()
        .map(|token| parse_usize(line, "index", token))
        .collect::<Result<Vec<_>>>()?;
    if indices.is_empty() {
        return parse_error(line, "section index list must not be empty");
    }
    if indices.contains(&0) {
        return parse_error(line, "section indices must be positive");
    }
    Ok(indices)
}

fn parse_single_value_after_colon(line: usize, text: &str, field: &'static str) -> Result<f64> {
    let values = parse_values_after_colon(line, text)?;
    if values.len() != 1 {
        return parse_error(
            line,
            format!("{field} line has {} value(s), expected 1", values.len()),
        );
    }
    Ok(values[0])
}

fn parse_values_after_colon(line: usize, text: &str) -> Result<Vec<f64>> {
    let (_, tail) = text
        .split_once(':')
        .ok_or_else(|| parse_error_value(line, "expected ':' separator"))?;
    let values = tail
        .split_whitespace()
        .map(|token| parse_f64(line, "value", token))
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return parse_error(line, "expected numeric value after ':'");
    }
    Ok(values)
}

fn next_line<'a>(
    lines: &'a [(usize, &'a str)],
    position: &mut usize,
    description: &'static str,
) -> Result<(usize, &'a str)> {
    let Some(line) = lines.get(*position).copied() else {
        return parse_error(0, format!("missing {description} line"));
    };
    *position += 1;
    Ok(line)
}

fn skip_blank_lines(lines: &[(usize, &str)], position: &mut usize) {
    while *position < lines.len() && lines[*position].1.is_empty() {
        *position += 1;
    }
}

fn skip_section_gap(lines: &[(usize, &str)], position: &mut usize) {
    while *position < lines.len() {
        let line = lines[*position].1;
        if line.is_empty() || is_separator_line(line) {
            *position += 1;
        } else {
            break;
        }
    }
}

fn is_section_start(line: &str) -> bool {
    line.starts_with("Path Indices:")
        || line.starts_with("Atom Index:")
        || line == "Total PDOS results:"
        || line == "Total VFE for the paths requested:"
}

fn is_section_gap_or_label(line: &str) -> bool {
    is_separator_line(line)
        || line.starts_with("PDOS Poles:")
        || line.starts_with("PDOS Einstein freq")
        || line.starts_with("pDOS n Moments")
        || line.starts_with("Path Red. Mass (AMU):")
        || line.starts_with("Path Length (Ang)")
        || line.starts_with("VFE (eV):")
        || line.starts_with("Temp (K)")
        || line.starts_with("u^2 (1e-3 Ang^2):")
        || line.starts_with("Projected DOS component computed.")
}

fn is_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3 && trimmed.chars().all(|ch| ch == '-')
}

fn parse_optional_f64(
    line: usize,
    field: &'static str,
    token: Option<&str>,
) -> Result<Option<f64>> {
    match token {
        Some(value) if is_placeholder(value) => Ok(None),
        Some(value) => parse_f64(line, field, value).map(Some),
        None => Ok(None),
    }
}

fn is_placeholder(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|ch| ch == '-')
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| parse_error_value(line, format!("could not parse {field} from {token:?}")))
}
