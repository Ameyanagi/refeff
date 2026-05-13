//! Minimal Crystallographic Information File reader for FEFF inputs.
//!
//! FEFF's RDINP stage accepts `CIF` cards and expands cell, symmetry, and
//! atom-site data before writing downstream handoff files.  This module starts
//! with the CIF subset used by the FEFF10 reference examples: scalar cell
//! parameters, space-group metadata, symmetry-operation loops, and fractional
//! atom-site loops.

use std::path::Path;

use crate::{IoError, Result};

/// Parsed CIF data needed by FEFF's structure import path.
#[derive(Debug, Clone, PartialEq)]
pub struct CifDocument {
    /// Optional CIF data block name without the `data_` prefix.
    pub data_block: Option<String>,
    /// Unit-cell parameters.
    pub cell: CifCell,
    /// International Tables space-group number, when present.
    pub space_group_number: Option<i32>,
    /// Hermann-Mauguin space-group label, when present.
    pub space_group_hm: Option<String>,
    /// Symmetry operations in CIF text form.
    pub symmetry_operations: Vec<String>,
    /// Fractional atom-site rows.
    pub atom_sites: Vec<CifAtomSite>,
}

/// Unit-cell lengths and angles from CIF scalar fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CifCell {
    /// `a` unit-cell length in angstroms.
    pub a: f64,
    /// `b` unit-cell length in angstroms.
    pub b: f64,
    /// `c` unit-cell length in angstroms.
    pub c: f64,
    /// `alpha` unit-cell angle in degrees.
    pub alpha: f64,
    /// `beta` unit-cell angle in degrees.
    pub beta: f64,
    /// `gamma` unit-cell angle in degrees.
    pub gamma: f64,
}

/// One CIF atom site with fractional coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CifAtomSite {
    /// Element symbol from `_atom_site_type_symbol`, or a stripped label fallback.
    pub symbol: String,
    /// Original atom-site label, when present.
    pub label: Option<String>,
    /// Fractional coordinate along the crystallographic `a` axis.
    pub fract_x: f64,
    /// Fractional coordinate along the crystallographic `b` axis.
    pub fract_y: f64,
    /// Fractional coordinate along the crystallographic `c` axis.
    pub fract_z: f64,
}

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
        if line.starts_with(';') {
            index = skip_multiline(&lines, index + 1);
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
    let value = tokens.get(1).map(String::as_str);
    if value.is_none()
        && lines
            .get(index + 1)
            .is_some_and(|line| line.starts_with(';'))
    {
        return Ok(skip_multiline(lines, index + 2));
    }

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
    Ok(index + 1)
}

fn parse_loop(lines: &[&str], mut index: usize, builder: &mut CifBuilder) -> Result<usize> {
    let mut headers = Vec::new();
    while let Some(line) = lines.get(index).map(|line| line.trim()) {
        if line.is_empty() || line.starts_with('#') {
            index += 1;
            continue;
        }
        if !line.starts_with('_') {
            break;
        }
        let tokens = tokenize_cif_line(line);
        if let Some(header) = tokens.first() {
            headers.push(header.clone());
        }
        index += 1;
    }

    let mut row_tokens = Vec::new();
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
        if line.starts_with(';') {
            index = skip_multiline(lines, index + 1);
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

fn tokenize_cif_line(line: &str) -> Vec<String> {
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

fn strip_element_label(label: &str) -> String {
    label
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
}

fn required_f64(value: Option<f64>, field: &str) -> Result<f64> {
    value.ok_or_else(|| invalid_cif(field, "missing required cell field"))
}

fn skip_multiline(lines: &[&str], mut index: usize) -> usize {
    while index < lines.len() {
        if lines[index].starts_with(';') {
            return index + 1;
        }
        index += 1;
    }
    index
}

fn invalid_cif(field: &str, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "cif".into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_cif, tokenize_cif_line};

    #[test]
    fn tokenizes_quoted_cif_values() {
        assert_eq!(
            tokenize_cif_line("_symmetry_space_group_name_H-M   'P 63/m m c' # comment"),
            ["_symmetry_space_group_name_H-M", "P 63/m m c"]
        );
    }

    #[test]
    fn parses_cell_uncertainties_and_atom_sites() -> crate::Result<()> {
        let cif = r#"
data_demo
_cell_length_a 2.9400(0)
_cell_length_b 2.9400(0)
_cell_length_c 12.1100(0)
_cell_angle_alpha 90.0000(0)
_cell_angle_beta 90.0000(0)
_cell_angle_gamma 120.0000(0)
_symmetry_space_group_name_H-M 'P 63/m m c'
_symmetry_Int_Tables_number 194
loop_
_symmetry_equiv_pos_as_xyz
'+x,+y,+z'
'-x,-y,1/2+z'
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C C1 0.0 0.0 0.0
Cr Cr1 0.6667 0.3333 0.0833
"#;
        let parsed = parse_cif(cif)?;
        assert_eq!(parsed.data_block.as_deref(), Some("demo"));
        assert_eq!(parsed.cell.a, 2.94);
        assert_eq!(parsed.cell.gamma, 120.0);
        assert_eq!(parsed.space_group_number, Some(194));
        assert_eq!(parsed.space_group_hm.as_deref(), Some("P 63/m m c"));
        assert_eq!(parsed.symmetry_operations.len(), 2);
        assert_eq!(parsed.atom_sites.len(), 2);
        assert_eq!(parsed.atom_sites[1].symbol, "Cr");
        Ok(())
    }
}
