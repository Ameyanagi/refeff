use std::path::Path;

use crate::{IoError, Result};

use super::common::{invalid_cif, required_f64, strip_element_label};
use super::types::{CifAtomSite, CifCell, CifDocument};

#[derive(Debug, Default)]
struct CifBuilder {
    data_block: Option<String>,
    a: Option<f64>,
    b: Option<f64>,
    c: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    space_group_number: Option<i32>,
    space_group_hm: Option<String>,
    symmetry_operations: Vec<String>,
    atom_sites: Vec<CifAtomSite>,
}

/// Parse CIF text into the FEFF structure subset.
///
/// FEFF calls CIFtbx's blank `data_` selector once, so only the first CIF data
/// block is used when a file contains multiple blocks.
pub fn parse_cif(text: &str) -> Result<CifDocument> {
    let lines = text.lines().collect::<Vec<_>>();
    let mut builder = CifBuilder::default();
    let mut index = 0_usize;

    while index < lines.len() {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if let Some(name) = line.strip_prefix("data_") {
            if builder.data_block.is_some() {
                break;
            }
            builder.data_block = Some(name.trim().to_string());
            index += 1;
            continue;
        }
        if line.eq_ignore_ascii_case("loop_") {
            index = parse_loop(&lines, index + 1, &mut builder)?;
            continue;
        }
        if line.starts_with('_') {
            index = parse_scalar(&lines, index, &mut builder)?;
            continue;
        }
        if is_cif_text_field_line(lines[index]) {
            let (_, next_index) = read_cif_text_field(&lines, index)?;
            index = next_index;
            continue;
        }
        index += 1;
    }

    let cell = CifCell {
        a: required_f64(builder.a, "cell_length_a")?,
        b: required_f64(builder.b, "cell_length_b")?,
        c: required_f64(builder.c, "cell_length_c")?,
        alpha: required_f64(builder.alpha, "cell_angle_alpha")?,
        beta: required_f64(builder.beta, "cell_angle_beta")?,
        gamma: required_f64(builder.gamma, "cell_angle_gamma")?,
    };
    if builder.atom_sites.is_empty() {
        return Err(invalid_cif("atom_site", "missing atom-site loop"));
    }

    Ok(CifDocument {
        data_block: builder.data_block,
        cell,
        space_group_number: builder.space_group_number,
        space_group_hm: builder.space_group_hm,
        symmetry_operations: builder.symmetry_operations,
        atom_sites: builder.atom_sites,
    })
}

/// Read and parse CIF text from a file.
pub fn read_cif(path: impl AsRef<Path>) -> Result<CifDocument> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_cif(&text)
}

fn parse_scalar(lines: &[&str], index: usize, builder: &mut CifBuilder) -> Result<usize> {
    let tokens = tokenize_cif_line(lines[index]);
    let Some(key) = tokens.first() else {
        return Ok(index + 1);
    };
    let (value, next_index) = if let Some(value) = tokens.get(1) {
        (Some(value.clone()), index + 1)
    } else if lines
        .get(index + 1)
        .is_some_and(|line| is_cif_text_field_line(line))
    {
        let (value, next_index) = read_cif_text_field(lines, index + 1)?;
        (Some(value), next_index)
    } else {
        (None, index + 1)
    };
    let value = value.as_deref();

    match key.as_str() {
        "_cell_length_a" => builder.a = Some(parse_cif_f64(value, key)?),
        "_cell_length_b" => builder.b = Some(parse_cif_f64(value, key)?),
        "_cell_length_c" => builder.c = Some(parse_cif_f64(value, key)?),
        "_cell_angle_alpha" => builder.alpha = Some(parse_cif_f64(value, key)?),
        "_cell_angle_beta" => builder.beta = Some(parse_cif_f64(value, key)?),
        "_cell_angle_gamma" => builder.gamma = Some(parse_cif_f64(value, key)?),
        "_space_group_IT_number" | "_symmetry_Int_Tables_number" => {
            builder.space_group_number = Some(parse_cif_i32(value, key)?);
        }
        "_symmetry_space_group_name_H-M"
        | "_space_group_name_H-M_alt"
        | "_[local]_cod_cif_authors_sg_H-M"
            if builder.space_group_hm.is_none() =>
        {
            builder.space_group_hm = value.map(normalize_cif_string);
        }
        _ => {}
    }
    Ok(next_index)
}

fn parse_loop(lines: &[&str], mut index: usize, builder: &mut CifBuilder) -> Result<usize> {
    let mut headers = Vec::new();
    let mut row_tokens = Vec::new();
    while let Some(line) = lines.get(index).map(|line| line.trim()) {
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if !line.starts_with('_') {
            break;
        }
        let tokens = tokenize_cif_line(line);
        let header_count = tokens
            .iter()
            .take_while(|token| token.starts_with('_'))
            .count();
        headers.extend(tokens.iter().take(header_count).cloned());
        row_tokens.extend(tokens.into_iter().skip(header_count));
        index += 1;
        if !row_tokens.is_empty() {
            break;
        }
    }

    while !headers.is_empty() && row_tokens.len() >= headers.len() {
        let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
        apply_loop_row(&headers, &row, builder)?;
    }

    while let Some(raw) = lines.get(index) {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if line.eq_ignore_ascii_case("loop_") || line.starts_with("data_") || line.starts_with('_')
        {
            break;
        }
        if is_cif_text_field_line(raw) {
            let (value, next_index) = read_cif_text_field(lines, index)?;
            row_tokens.push(value);
            while !headers.is_empty() && row_tokens.len() >= headers.len() {
                let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
                apply_loop_row(&headers, &row, builder)?;
            }
            index = next_index;
            continue;
        }

        row_tokens.extend(tokenize_cif_line(line));
        while !headers.is_empty() && row_tokens.len() >= headers.len() {
            let row = row_tokens.drain(..headers.len()).collect::<Vec<_>>();
            apply_loop_row(&headers, &row, builder)?;
        }
        index += 1;
    }
    Ok(index)
}

fn apply_loop_row(headers: &[String], row: &[String], builder: &mut CifBuilder) -> Result<()> {
    if let Some(op_index) = headers.iter().position(|header| {
        header == "_symmetry_equiv_pos_as_xyz" || header == "_space_group_symop_operation_xyz"
    }) {
        if let Some(operation) = row.get(op_index) {
            builder
                .symmetry_operations
                .push(normalize_cif_string(operation));
        }
        return Ok(());
    }

    let x_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_x");
    let y_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_y");
    let z_index = headers
        .iter()
        .position(|header| header == "_atom_site_fract_z");
    let Some((x_index, y_index, z_index)) = x_index
        .zip(y_index)
        .zip(z_index)
        .map(|((x, y), z)| (x, y, z))
    else {
        return Ok(());
    };

    let label = headers
        .iter()
        .position(|header| header == "_atom_site_label")
        .and_then(|idx| row.get(idx))
        .map(|value| normalize_cif_string(value));
    let symbol = headers
        .iter()
        .position(|header| header == "_atom_site_type_symbol")
        .and_then(|idx| row.get(idx))
        .map(|value| normalize_cif_string(value))
        .or_else(|| label.as_deref().map(strip_element_label))
        .ok_or_else(|| invalid_cif("atom_site", "missing atom symbol or label"))?;

    builder.atom_sites.push(CifAtomSite {
        symbol,
        label,
        fract_x: parse_cif_f64(row.get(x_index).map(String::as_str), "_atom_site_fract_x")?,
        fract_y: parse_cif_f64(row.get(y_index).map(String::as_str), "_atom_site_fract_y")?,
        fract_z: parse_cif_f64(row.get(z_index).map(String::as_str), "_atom_site_fract_z")?,
    });
    Ok(())
}

pub(super) fn tokenize_cif_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in line.chars() {
        if let Some(quote_char) = quote {
            if ch == quote_char {
                quote = None;
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch == '#' {
            break;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_cif_f64(value: Option<&str>, field: &str) -> Result<f64> {
    let value = value.ok_or_else(|| invalid_cif(field, "missing numeric value"))?;
    let trimmed = numeric_without_uncertainty(value);
    trimmed
        .parse::<f64>()
        .map_err(|_| invalid_cif(field, format!("invalid numeric value {value:?}")))
}

fn parse_cif_i32(value: Option<&str>, field: &str) -> Result<i32> {
    let value = value.ok_or_else(|| invalid_cif(field, "missing integer value"))?;
    numeric_without_uncertainty(value)
        .parse::<i32>()
        .map_err(|_| invalid_cif(field, format!("invalid integer value {value:?}")))
}

fn numeric_without_uncertainty(value: &str) -> &str {
    let value = value.trim();
    value.find('(').map_or(value, |index| &value[..index])
}

fn normalize_cif_string(value: &str) -> String {
    value.trim_matches(['\'', '"']).trim().to_string()
}

fn is_cif_text_field_line(line: &str) -> bool {
    line.trim_start().starts_with(';')
}

fn read_cif_text_field(lines: &[&str], index: usize) -> Result<(String, usize)> {
    let Some(first_line) = lines.get(index) else {
        return Err(invalid_cif("text", "missing semicolon text field"));
    };
    let Some(semicolon_index) = first_line.find(';') else {
        return Err(invalid_cif("text", "missing semicolon text delimiter"));
    };

    let mut parts = Vec::new();
    let opening_remainder = &first_line[(semicolon_index + 1)..];
    if !opening_remainder.is_empty() {
        parts.push(opening_remainder);
    }

    for (offset, line) in lines.iter().enumerate().skip(index + 1) {
        if is_cif_text_field_line(line) {
            return Ok((parts.join("\n"), offset + 1));
        }
        parts.push(line);
    }

    Err(invalid_cif("text", "unterminated semicolon text field"))
}
