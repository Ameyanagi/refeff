//! FEFF energy-grid and scalar output readers.
//!
//! FEFF writes `edges.dat`, `chemical.dat`, and `emesh.dat` during the
//! potential and phase-shift stages. They are small formatted text files used by
//! later modules and by diagnostics, so the Rust port keeps them typed rather
//! than treating them as opaque text.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};

/// One row from FEFF `edges.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgesDatRow {
    /// FEFF `emu` column.
    pub chemical_potential: f64,
    /// FEFF `M_kk` dipole matrix-element column.
    pub matrix_element: f64,
    /// FEFF `gam` core-hole width column.
    pub core_hole_width: f64,
}

/// Parsed contents of FEFF `edges.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgesDatData {
    /// Header or comment lines before and around edge rows.
    pub header_lines: Vec<String>,
    /// Edge rows in file order.
    pub rows: Vec<EdgesDatRow>,
}

/// Parsed contents of FEFF `chemical.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalDatData {
    /// Self-consistency temperature in FEFF's first-column units.
    pub scf_temperature: f64,
    /// Self-consistency temperature converted to kelvin.
    pub scf_temperature_kelvin: f64,
    /// Chemical potential in eV.
    pub chemical_potential_ev: f64,
}

/// Parsed contents of FEFF `emesh.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmeshDatData {
    /// Edge energy in Hartree from the first header line.
    pub edge_hartree: f64,
    /// Bohr radius in Angstrom from the first header line.
    pub bohr_angstrom: f64,
    /// Edge energy in eV from the first header line.
    pub edge_ev: f64,
    /// FEFF spectrum selector, `ispec`.
    pub spectrum: i32,
    /// FEFF Fermi-index marker, `ik0`.
    pub fermi_index: usize,
    /// Explicit row indices from the first table column.
    pub indices: Array1<usize>,
    /// Energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// Photoelectron wave number in inverse Angstrom.
    pub wave_number_inverse_angstrom: Array1<f64>,
}

impl EdgesDatData {
    /// Number of edge rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

impl EmeshDatData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Parse FEFF `edges.dat` text.
pub fn parse_edges_dat(text: &str) -> Result<EdgesDatData> {
    let mut header_lines = Vec::new();
    let mut rows = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        let line_number = index + 1;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            if tokens.len() != 3 {
                return parse_error(
                    "edges.dat",
                    line_number,
                    format!("edge row has {} token(s), expected 3", tokens.len()),
                );
            }
            rows.push(EdgesDatRow {
                chemical_potential: parse_f64("edges.dat", line_number, "emu", tokens[0])?,
                matrix_element: parse_f64("edges.dat", line_number, "M_kk", tokens[1])?,
                core_hole_width: parse_f64("edges.dat", line_number, "gam", tokens[2])?,
            });
        } else if !line.trim().is_empty() {
            header_lines.push(line.to_string());
        }
    }

    let data = EdgesDatData { header_lines, rows };
    validate_edges_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `edges.dat` text.
pub fn edges_dat_string(data: &EdgesDatData) -> Result<String> {
    validate_edges_dat(data)?;
    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for row in &data.rows {
        writeln!(
            out,
            " {:24.16} {:24.16} {:24.16E}",
            row.chemical_potential, row.matrix_element, row.core_hole_width
        )?;
    }
    Ok(out)
}

/// Read FEFF `edges.dat` text from a file.
pub fn read_edges_dat(path: impl AsRef<Path>) -> Result<EdgesDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_edges_dat(&text)
}

/// Write FEFF `edges.dat` text to a file.
pub fn write_edges_dat(path: impl AsRef<Path>, data: &EdgesDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, edges_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `chemical.dat` text.
pub fn parse_chemical_dat(text: &str) -> Result<ChemicalDatData> {
    let mut rows = text.lines().enumerate().filter_map(|(index, raw)| {
        let line = raw.trim();
        (!line.is_empty()).then_some((index + 1, line))
    });
    let (line_number, line) = rows
        .next()
        .ok_or_else(|| parse_error_value("chemical.dat", 0, "missing chemical.dat row"))?;
    if rows.next().is_some() {
        return parse_error("chemical.dat", line_number, "expected exactly one data row");
    }
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 3 {
        return parse_error(
            "chemical.dat",
            line_number,
            format!("chemical row has {} token(s), expected 3", tokens.len()),
        );
    }
    let data = ChemicalDatData {
        scf_temperature: parse_f64("chemical.dat", line_number, "scf_temperature", tokens[0])?,
        scf_temperature_kelvin: parse_f64(
            "chemical.dat",
            line_number,
            "scf_temperature_kelvin",
            tokens[1],
        )?,
        chemical_potential_ev: parse_f64(
            "chemical.dat",
            line_number,
            "chemical_potential_ev",
            tokens[2],
        )?,
    };
    validate_chemical_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `chemical.dat` text.
pub fn chemical_dat_string(data: &ChemicalDatData) -> Result<String> {
    validate_chemical_dat(data)?;
    Ok(format!(
        " {:24.16} {:24.16} {:24.16}\n",
        data.scf_temperature, data.scf_temperature_kelvin, data.chemical_potential_ev
    ))
}

/// Read FEFF `chemical.dat` text from a file.
pub fn read_chemical_dat(path: impl AsRef<Path>) -> Result<ChemicalDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_chemical_dat(&text)
}

/// Write FEFF `chemical.dat` text to a file.
pub fn write_chemical_dat(path: impl AsRef<Path>, data: &ChemicalDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, chemical_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `emesh.dat` text.
pub fn parse_emesh_dat(text: &str) -> Result<EmeshDatData> {
    let mut lines = text.lines().enumerate();
    let (edge_line_number, edge_line) = next_nonempty_line(&mut lines, "emesh.dat", "edge header")?;
    let edge_values = parse_prefixed_f64s(
        "emesh.dat",
        edge_line_number,
        edge_line,
        "# edge, bohr, edge*hart",
        3,
    )?;
    let (ispec_line_number, ispec_line) =
        next_nonempty_line(&mut lines, "emesh.dat", "ispec header")?;
    let ispec_values = parse_prefixed_ints(
        "emesh.dat",
        ispec_line_number,
        ispec_line,
        "# ispec, ik0",
        2,
    )?;
    let (_, label_line) = next_nonempty_line(&mut lines, "emesh.dat", "table label")?;
    if !label_line
        .trim_start()
        .starts_with("# ie, em(ie)*hart, xk(ie)")
    {
        return parse_error("emesh.dat", edge_line_number, "missing emesh table label");
    }

    let mut indices = Vec::new();
    let mut energy_ev = Vec::new();
    let mut wave_number_inverse_angstrom = Vec::new();
    for (index, raw) in lines {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 3 {
            return parse_error(
                "emesh.dat",
                line_number,
                format!("emesh row has {} token(s), expected 3", tokens.len()),
            );
        }
        indices.push(parse_usize("emesh.dat", line_number, "ie", tokens[0])?);
        energy_ev.push(parse_f64("emesh.dat", line_number, "energy_ev", tokens[1])?);
        wave_number_inverse_angstrom.push(parse_f64(
            "emesh.dat",
            line_number,
            "wave_number_inverse_angstrom",
            tokens[2],
        )?);
    }

    let data = EmeshDatData {
        edge_hartree: edge_values[0],
        bohr_angstrom: edge_values[1],
        edge_ev: edge_values[2],
        spectrum: i32::try_from(ispec_values[0]).map_err(|_| {
            parse_error_value("emesh.dat", ispec_line_number, "ispec does not fit in i32")
        })?,
        fermi_index: usize::try_from(ispec_values[1]).map_err(|_| {
            parse_error_value("emesh.dat", ispec_line_number, "ik0 must be non-negative")
        })?,
        indices: Array1::from_vec(indices),
        energy_ev: Array1::from_vec(energy_ev),
        wave_number_inverse_angstrom: Array1::from_vec(wave_number_inverse_angstrom),
    };
    validate_emesh_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `emesh.dat` text.
pub fn emesh_dat_string(data: &EmeshDatData) -> Result<String> {
    validate_emesh_dat(data)?;
    let mut out = String::new();
    writeln!(
        out,
        "# edge, bohr, edge*hart  {:12.5} {:12.5} {:12.5}",
        data.edge_hartree, data.bohr_angstrom, data.edge_ev
    )?;
    writeln!(
        out,
        "# ispec, ik0  {:5} {:5}",
        data.spectrum, data.fermi_index
    )?;
    writeln!(out, " # ie, em(ie)*hart, xk(ie)")?;
    for ((index, energy), wave_number) in data
        .indices
        .iter()
        .zip(data.energy_ev.iter())
        .zip(data.wave_number_inverse_angstrom.iter())
    {
        writeln!(out, "{index:5}{energy:20.5}{wave_number:20.5}")?;
    }
    Ok(out)
}

/// Read FEFF `emesh.dat` text from a file.
pub fn read_emesh_dat(path: impl AsRef<Path>) -> Result<EmeshDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_emesh_dat(&text)
}

/// Write FEFF `emesh.dat` text to a file.
pub fn write_emesh_dat(path: impl AsRef<Path>, data: &EmeshDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, emesh_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_edges_dat(data: &EdgesDatData) -> Result<()> {
    if data.rows.is_empty() {
        return parse_error("edges.dat", 0, "at least one edge row is required");
    }
    for (index, row) in data.rows.iter().enumerate() {
        let row_number = index + 1;
        validate_finite("edges.dat", "emu", row.chemical_potential, row_number)?;
        validate_finite("edges.dat", "M_kk", row.matrix_element, row_number)?;
        validate_finite("edges.dat", "gam", row.core_hole_width, row_number)?;
    }
    Ok(())
}

fn validate_chemical_dat(data: &ChemicalDatData) -> Result<()> {
    validate_finite("chemical.dat", "scf_temperature", data.scf_temperature, 1)?;
    validate_finite(
        "chemical.dat",
        "scf_temperature_kelvin",
        data.scf_temperature_kelvin,
        1,
    )?;
    validate_finite(
        "chemical.dat",
        "chemical_potential_ev",
        data.chemical_potential_ev,
        1,
    )
}

fn validate_emesh_dat(data: &EmeshDatData) -> Result<()> {
    validate_finite("emesh.dat", "edge_hartree", data.edge_hartree, 1)?;
    validate_finite("emesh.dat", "bohr_angstrom", data.bohr_angstrom, 1)?;
    validate_finite("emesh.dat", "edge_ev", data.edge_ev, 1)?;
    if data.point_count() == 0 {
        return parse_error("emesh.dat", 0, "at least one energy-grid row is required");
    }
    if data.indices.len() != data.point_count() {
        return parse_error(
            "emesh.dat",
            0,
            format!(
                "index count {} does not match energy count {}",
                data.indices.len(),
                data.point_count()
            ),
        );
    }
    if data.wave_number_inverse_angstrom.len() != data.point_count() {
        return parse_error(
            "emesh.dat",
            0,
            format!(
                "wave-number count {} does not match energy count {}",
                data.wave_number_inverse_angstrom.len(),
                data.point_count()
            ),
        );
    }
    if data.fermi_index == 0 || data.fermi_index > data.point_count() {
        return parse_error(
            "emesh.dat",
            0,
            format!(
                "ik0 {} must be in 1..={}",
                data.fermi_index,
                data.point_count()
            ),
        );
    }
    for (row, ((index, energy), wave_number)) in data
        .indices
        .iter()
        .zip(data.energy_ev.iter())
        .zip(data.wave_number_inverse_angstrom.iter())
        .enumerate()
    {
        if *index == 0 {
            return parse_error("emesh.dat", row + 1, "energy index must be positive");
        }
        validate_finite("emesh.dat", "energy_ev", *energy, row + 1)?;
        validate_finite(
            "emesh.dat",
            "wave_number_inverse_angstrom",
            *wave_number,
            row + 1,
        )?;
    }
    Ok(())
}

fn next_nonempty_line<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    path: &'static str,
    field: &'static str,
) -> Result<(usize, &'a str)> {
    for (index, raw) in lines {
        let line = raw.trim_end();
        if !line.trim().is_empty() {
            return Ok((index + 1, line));
        }
    }
    parse_error(path, 0, format!("missing {field}"))
}

fn parse_prefixed_f64s(
    path: &'static str,
    line: usize,
    raw: &str,
    prefix: &str,
    width: usize,
) -> Result<Vec<f64>> {
    let values = raw
        .trim_start()
        .strip_prefix(prefix)
        .ok_or_else(|| parse_error_value(path, line, format!("expected {prefix:?}")))?;
    let tokens = values.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != width {
        return parse_error(
            path,
            line,
            format!("header has {} token(s), expected {width}", tokens.len()),
        );
    }
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| parse_f64(path, line, header_field(index), token))
        .collect()
}

fn parse_prefixed_ints(
    path: &'static str,
    line: usize,
    raw: &str,
    prefix: &str,
    width: usize,
) -> Result<Vec<i64>> {
    let values = raw
        .trim_start()
        .strip_prefix(prefix)
        .ok_or_else(|| parse_error_value(path, line, format!("expected {prefix:?}")))?;
    let tokens = values.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != width {
        return parse_error(
            path,
            line,
            format!("header has {} token(s), expected {width}", tokens.len()),
        );
    }
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| parse_i64(path, line, header_field(index), token))
        .collect()
}

fn header_field(index: usize) -> &'static str {
    match index {
        0 => "header_0",
        1 => "header_1",
        2 => "header_2",
        _ => "header",
    }
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
}

fn parse_i64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<i64> {
    token.parse::<i64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
}

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
}

fn validate_finite(path: &'static str, field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, row, format!("{field} must be finite"))
    }
}

fn parse_error<T>(path: &'static str, line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(path, line, message))
}

fn parse_error_value(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
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
    fn parses_edges_dat_rows() -> Result<()> {
        let parsed = parse_edges_dat(EDGES_DAT)?;
        assert_eq!(parsed.header_lines, vec![" # emu, M_kk, gam"]);
        assert_eq!(parsed.row_count(), 1);
        assert_eq!(parsed.rows[0].chemical_potential, 330.319_156_029_843_7);
        assert_eq!(parsed.rows[0].matrix_element, 1.0);
        assert_eq!(parsed.rows[0].core_hole_width, 0.063_546_470_930_994_86);
        assert_eq!(
            parse_edges_dat(&edges_dat_string(&parsed)?)?.rows,
            parsed.rows
        );
        Ok(())
    }

    #[test]
    fn parses_chemical_dat_scalars() -> Result<()> {
        let parsed = parse_chemical_dat(CHEMICAL_DAT)?;
        assert_eq!(parsed.scf_temperature, 0.0);
        assert_eq!(parsed.scf_temperature_kelvin, 0.0);
        assert_eq!(parsed.chemical_potential_ev, -7.729_278_779_143_69);
        assert_eq!(parse_chemical_dat(&chemical_dat_string(&parsed)?)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_emesh_dat_grid() -> Result<()> {
        let parsed = parse_emesh_dat(EMESH_DAT)?;
        assert_eq!(parsed.edge_hartree, -0.28405);
        assert_eq!(parsed.bohr_angstrom, 0.52918);
        assert_eq!(parsed.edge_ev, -7.72928);
        assert_eq!(parsed.spectrum, 3);
        assert_eq!(parsed.fermi_index, 2);
        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.indices.as_slice(), Some(&[1, 2, 3][..]));
        assert_eq!(parsed.energy_ev[0], -16.76504);
        assert_eq!(parsed.wave_number_inverse_angstrom[2], -1.26);

        let rendered = emesh_dat_string(&parsed)?;
        assert_eq!(parse_emesh_dat(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn rejects_bad_energy_output_inputs() {
        assert!(parse_edges_dat("# only header\n").is_err());
        assert!(parse_edges_dat("1 2\n").is_err());
        assert!(parse_chemical_dat("").is_err());
        assert!(parse_chemical_dat("1 2\n").is_err());
        assert!(parse_emesh_dat("# only header\n").is_err());
        assert!(parse_emesh_dat(&EMESH_DAT.replace("# ispec, ik0", "# bad")).is_err());
    }

    const EDGES_DAT: &str = r#" # emu, M_kk, gam
   330.31915602984373        1.0000000000000000        6.3546470930994858E-002
"#;

    const CHEMICAL_DAT: &str =
        "   0.0000000000000000        0.0000000000000000       -7.7292787791436899     \n";

    const EMESH_DAT: &str = r#"# edge, bohr, edge*hart      -0.28405      0.52918     -7.72928
# ispec, ik0      3     2
 # ie, em(ie)*hart, xk(ie)
    1           -16.76504            -1.54000
    2           -15.19685            -1.40000
    3           -13.77801            -1.26000
"#;
}
