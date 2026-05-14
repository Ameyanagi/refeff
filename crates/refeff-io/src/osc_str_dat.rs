//! FEFF `osc_str.dat` oscillator-strength summary codec.
//!
//! `FULLSPECTRUM/fullspectrum.f90` writes this fixed-width text table while
//! collecting each edge's effective electron count from the sum rule.

use std::fmt::Write as _;
use std::path::Path;

use refeff_core::{FullSpectrumEdgeAssembly, edge_index, standard_edge_label};

use crate::error::{IoError, Result};

const OSC_STR_DAT_PATH: &str = "osc_str.dat";
const COMPONENT_WIDTH: usize = 11;
const FEFF_COMPONENT_CHARS: usize = 3;
const EDGE_WIDTH: usize = 6;
const FEFF_EDGE_CHARS: usize = 2;

/// One FEFF `osc_str.dat` edge summary row.
#[derive(Debug, Clone, PartialEq)]
pub struct OscStrRow {
    /// FULLSPECTRUM component name, stored by FEFF as a three-character field.
    pub component: String,
    /// FEFF edge label, stored by FEFF as a two-character field.
    pub edge: String,
    /// FEFF core-hole index returned by `setedg`.
    pub core_hole_index: i32,
    /// Effective electron count from `FULLSPECTRUM/qsum.f90`.
    pub effective_electron_count: f64,
}

/// Parsed FEFF `osc_str.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct OscStrDatData {
    /// Header or comment lines before and around oscillator-strength rows.
    pub header_lines: Vec<String>,
    /// Edge summary rows in file order.
    pub rows: Vec<OscStrRow>,
}

impl OscStrDatData {
    /// Number of oscillator-strength rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// Render FEFF-compatible `osc_str.dat` text.
pub fn osc_str_dat_string(data: &OscStrDatData) -> Result<String> {
    validate_osc_str_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for row in &data.rows {
        let component = feff_char_field(&row.component, FEFF_COMPONENT_CHARS, COMPONENT_WIDTH);
        let edge = feff_char_field(&row.edge, FEFF_EDGE_CHARS, EDGE_WIDTH);
        writeln!(
            out,
            "{component}{edge}{:>4}{:>8.3}",
            row.core_hole_index, row.effective_electron_count
        )?;
    }
    Ok(out)
}

/// Parse FEFF `osc_str.dat` text.
pub fn parse_osc_str_dat(text: &str) -> Result<OscStrDatData> {
    let mut header_lines = Vec::new();
    let mut rows = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if is_osc_str_row(&tokens) {
            rows.push(OscStrRow {
                component: tokens[0].to_string(),
                edge: tokens[1].to_string(),
                core_hole_index: parse_i32(line_number, "core hole index", tokens[2])?,
                effective_electron_count: parse_f64(
                    line_number,
                    "effective electron count",
                    tokens[3],
                )?,
            });
        } else {
            header_lines.push(raw.to_string());
        }
    }

    let data = OscStrDatData { header_lines, rows };
    validate_osc_str_dat(&data)?;
    Ok(data)
}

/// Write FEFF `osc_str.dat` text to a file.
pub fn write_osc_str_dat(path: impl AsRef<Path>, data: &OscStrDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, osc_str_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `osc_str.dat` text from a file.
pub fn read_osc_str_dat(path: impl AsRef<Path>) -> Result<OscStrDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_osc_str_dat(&text)
}

/// Build one `osc_str.dat` row from an assembled FULLSPECTRUM edge.
///
/// This is the typed output adapter for the `fullspectrum.f90` statement that
/// writes `cmpnm`, `edname`, `ihole`, and `neff` after `addedg.f90` returns an
/// edge contribution.
pub fn osc_str_row_from_fullspectrum_edge(
    component: &str,
    edge: &str,
    assembly: &FullSpectrumEdgeAssembly,
) -> Result<OscStrRow> {
    let component = fixed_component_label(component)?;
    let Some(edge) = standard_edge_label(edge) else {
        return invalid_osc_str_dat("edge", format!("unknown FEFF edge label {edge:?}"));
    };
    let Some(core_hole_index) = edge_index(edge) else {
        return invalid_osc_str_dat("edge", format!("unknown FEFF edge label {edge:?}"));
    };
    let row = OscStrRow {
        component,
        edge: edge.to_string(),
        core_hole_index,
        effective_electron_count: assembly.effective_electron_count,
    };
    validate_osc_str_row(&row, 1)?;
    Ok(row)
}

fn is_osc_str_row(tokens: &[&str]) -> bool {
    tokens.len() == 4
        && !tokens[0].starts_with('#')
        && !tokens[1].starts_with('#')
        && tokens[2].parse::<i32>().is_ok()
        && is_numeric_token(tokens[3])
}

fn fixed_component_label(value: &str) -> Result<String> {
    let trimmed = value.trim();
    let component = trimmed
        .chars()
        .take(FEFF_COMPONENT_CHARS)
        .collect::<String>();
    if component.is_empty() {
        invalid_osc_str_dat("component", "value must not be empty")
    } else {
        Ok(component)
    }
}

fn feff_char_field(value: &str, source_width: usize, field_width: usize) -> String {
    let source = format!("{value:<source_width$}");
    format!("{source:>field_width$}")
}

fn validate_osc_str_dat(data: &OscStrDatData) -> Result<()> {
    if data.rows.is_empty() {
        return invalid_osc_str_dat("rows", "at least one oscillator-strength row is required");
    }
    for (row_index, row) in data.rows.iter().enumerate() {
        validate_osc_str_row(row, row_index + 1)?;
    }
    Ok(())
}

fn validate_osc_str_row(row: &OscStrRow, row_index: usize) -> Result<()> {
    validate_label("component", &row.component, FEFF_COMPONENT_CHARS, row_index)?;
    validate_label("edge", &row.edge, FEFF_EDGE_CHARS, row_index)?;
    validate_finite(
        "effective electron count",
        row.effective_electron_count,
        row_index,
    )
}

fn validate_label(field: &'static str, value: &str, max_len: usize, row: usize) -> Result<()> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        invalid_osc_str_dat(field, format!("row {row} value must not be empty"))
    } else if trimmed.len() > max_len {
        invalid_osc_str_dat(
            field,
            format!("row {row} value {value:?} exceeds FEFF width {max_len}"),
        )
    } else if trimmed != value {
        invalid_osc_str_dat(field, format!("row {row} value must already be trimmed"))
    } else {
        Ok(())
    }
}

fn parse_i32(line: usize, field: &'static str, token: &str) -> Result<i32> {
    token
        .parse::<i32>()
        .map_err(|_| parse_error_value(line, format!("invalid {field} value {token:?}")))
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(line, format!("invalid {field} value {token:?}")))
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_osc_str_dat(field, format!("row {row} value must be finite"))
    }
}

fn invalid_osc_str_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: OSC_STR_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn parse_error_value(line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: OSC_STR_DAT_PATH.into(),
        line,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_fullspectrum_osc_str_dat() -> Result<()> {
        let parsed = parse_osc_str_dat(OSC_STR_DAT)?;

        assert_eq!(parsed.header_lines, ["# component  edge  n_eff", " "]);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.rows[0].component, "Cu");
        assert_eq!(parsed.rows[0].edge, "K");
        assert_eq!(parsed.rows[0].core_hole_index, 1);
        assert_eq!(parsed.rows[0].effective_electron_count, 5.123);
        assert_eq!(parsed.rows[1].component, "O");
        assert_eq!(parsed.rows[1].edge, "L1");
        assert_eq!(parsed.rows[1].effective_electron_count, 0.456);
        Ok(())
    }

    #[test]
    fn roundtrips_feff_fullspectrum_osc_str_dat() -> Result<()> {
        let parsed = parse_osc_str_dat(OSC_STR_DAT)?;
        let rendered = osc_str_dat_string(&parsed)?;

        assert_eq!(rendered, OSC_STR_DAT);
        assert_eq!(parse_osc_str_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn builds_row_from_fullspectrum_edge_assembly() -> Result<()> {
        let assembly = sample_edge_assembly(5.1234);

        let row = osc_str_row_from_fullspectrum_edge("Copper", "1", &assembly)?;

        assert_eq!(row.component, "Cop");
        assert_eq!(row.edge, "K");
        assert_eq!(row.core_hole_index, 1);
        assert_eq!(
            row.effective_electron_count,
            assembly.effective_electron_count
        );
        let data = OscStrDatData {
            header_lines: vec!["# component  edge  n_eff".to_string(), " ".to_string()],
            rows: vec![row],
        };
        assert_eq!(
            osc_str_dat_string(&data)?,
            "# component  edge  n_eff\n \n        Cop    K    1   5.123\n"
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_osc_str_dat_inputs() {
        assert!(parse_osc_str_dat("").is_err());
        assert!(parse_osc_str_dat("# only header\n").is_err());
        assert!(parse_osc_str_dat("component edge hole n_eff\n").is_err());
        assert!(parse_osc_str_dat("Cu K bad 1.0\n").is_err());
        assert!(parse_osc_str_dat("Cu K 1 NaN\n").is_err());

        let bad = OscStrDatData {
            header_lines: Vec::new(),
            rows: vec![OscStrRow {
                component: "Copper".to_string(),
                edge: "K".to_string(),
                core_hole_index: 1,
                effective_electron_count: 1.0,
            }],
        };
        assert!(osc_str_dat_string(&bad).is_err());
        assert!(
            osc_str_row_from_fullspectrum_edge("Cu", "Q1", &sample_edge_assembly(1.0)).is_err()
        );
    }

    fn sample_edge_assembly(effective_electron_count: f64) -> FullSpectrumEdgeAssembly {
        FullSpectrumEdgeAssembly {
            scattering_factor: ndarray::Array1::from_elem(2, num_complex::Complex64::new(0.0, 0.0)),
            background: ndarray::Array1::from_elem(2, num_complex::Complex64::new(0.0, 0.0)),
            effective_electron_count,
            zero_energy_fprime: 0.0,
            overlap_points: 1,
        }
    }

    const OSC_STR_DAT: &str = "# component  edge  n_eff\n \n        Cu     K    1   5.123\n        O      L1   3   0.456\n";
}
