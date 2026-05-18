//! FEFF DMDW run-type 2 PDS and `a2f` coupling-table support.
//!
//! FEFF `DMDW/m_dmdw.f90` skips the first ten lines of the PDS and `a2f`
//! handoff files, then reads two numeric columns from each remaining row. This
//! module keeps that boundary typed and feeds the core phonon-coupling
//! transform without re-parsing ad hoc text in orchestration code.

use std::path::Path;

use ndarray::Array1;
use refeff_core::{DebyeError, DmdwPhononCoupling, dmdw_phonon_coupling};

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const DMDW_COUPLING_TABLE_PATH: &str = "dmdw-coupling-table";
const DMDW_A2_DAT_PATH: &str = "dmdw_A2.dat";
const DMDW_COUPLING_HEADER_LINES: usize = 10;
const DMDW_COUPLING_ROW_WIDTH: usize = 2;
const DMDW_A2_DAT_ROW_WIDTH: usize = 2;

/// Two-column FEFF DMDW run-type 2 PDS or `a2f` handoff table.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwCouplingTable {
    /// The ten header lines FEFF skips before reading data rows.
    pub header_lines: Vec<String>,
    /// Phonon energy grid in Hartree.
    pub energy_hartree: Array1<f64>,
    /// Second table column: projected phonon DOS for PDS files, or Eliashberg
    /// coupling for `a2f` files.
    pub values: Array1<f64>,
}

/// Parsed FEFF DMDW `dmdw_A2.dat` matrix-element sidecar.
#[derive(Debug, Clone, PartialEq)]
pub struct DmdwA2DatData {
    /// Phonon energy grid in Hartree.
    pub energy_hartree: Array1<f64>,
    /// FEFF `a2(2,j)`, the Eliashberg coupling divided by projected phonon DOS.
    pub matrix_element: Array1<f64>,
}

impl DmdwCouplingTable {
    /// Number of numeric rows after the ten-line header.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

impl DmdwA2DatData {
    /// Number of matrix-element rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

/// Parse a FEFF DMDW run-type 2 PDS or `a2f` coupling table.
pub fn parse_dmdw_coupling_table(text: &str) -> Result<DmdwCouplingTable> {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() < DMDW_COUPLING_HEADER_LINES {
        return parse_error(
            lines.len() + 1,
            format!(
                "DMDW coupling table has {} header line(s), expected {DMDW_COUPLING_HEADER_LINES}",
                lines.len()
            ),
        );
    }

    let header_lines = lines
        .iter()
        .take(DMDW_COUPLING_HEADER_LINES)
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>();
    let mut energy_hartree = Vec::new();
    let mut values = Vec::new();

    for (offset, raw) in lines.iter().enumerate().skip(DMDW_COUPLING_HEADER_LINES) {
        let line_number = offset + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != DMDW_COUPLING_ROW_WIDTH {
            return parse_error(
                line_number,
                format!(
                    "DMDW coupling table row has {} token(s), expected {DMDW_COUPLING_ROW_WIDTH}",
                    tokens.len()
                ),
            );
        }
        energy_hartree.push(parse_coupling_f64(line_number, "energy", tokens[0])?);
        values.push(parse_coupling_f64(line_number, "value", tokens[1])?);
    }

    let data = DmdwCouplingTable {
        header_lines,
        energy_hartree: Array1::from_vec(energy_hartree),
        values: Array1::from_vec(values),
    };
    validate_dmdw_coupling_table(&data)?;
    Ok(data)
}

/// Render a FEFF-compatible DMDW run-type 2 PDS or `a2f` coupling table.
pub fn dmdw_coupling_table_string(data: &DmdwCouplingTable) -> Result<String> {
    validate_dmdw_coupling_table(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        out.push_str(line);
        out.push('\n');
    }
    for (&energy, &value) in data.energy_hartree.iter().zip(data.values.iter()) {
        write_fortran_zero_scaled_exp(&mut out, energy, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, value, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

/// Read a FEFF DMDW run-type 2 PDS or `a2f` table from disk.
pub fn read_dmdw_coupling_table(path: impl AsRef<Path>) -> Result<DmdwCouplingTable> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_coupling_table(&text)
}

/// Write a FEFF DMDW run-type 2 PDS or `a2f` table to disk.
pub fn write_dmdw_coupling_table(path: impl AsRef<Path>, data: &DmdwCouplingTable) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_coupling_table_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Compute FEFF DMDW run-type 2 phonon coupling from parsed PDS and `a2f` tables.
pub fn dmdw_phonon_coupling_from_tables(
    pds: &DmdwCouplingTable,
    a2f: &DmdwCouplingTable,
) -> std::result::Result<DmdwPhononCoupling, DebyeError> {
    dmdw_phonon_coupling(
        pds.energy_hartree.view(),
        pds.values.view(),
        a2f.energy_hartree.view(),
        a2f.values.view(),
    )
}

/// Build `dmdw_A2.dat` data from FEFF DMDW phonon-coupling output.
pub fn dmdw_a2_dat_from_coupling(coupling: &DmdwPhononCoupling) -> Result<DmdwA2DatData> {
    let data = DmdwA2DatData {
        energy_hartree: coupling.energy_hartree.clone(),
        matrix_element: coupling.matrix_element.clone(),
    };
    validate_dmdw_a2_dat(&data)?;
    Ok(data)
}

/// Parse FEFF DMDW `dmdw_A2.dat` text.
pub fn parse_dmdw_a2_dat(text: &str) -> Result<DmdwA2DatData> {
    let mut energy_hartree = Vec::new();
    let mut matrix_element = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != DMDW_A2_DAT_ROW_WIDTH {
            return parse_a2_error(
                line_number,
                format!(
                    "dmdw_A2.dat row has {} token(s), expected {DMDW_A2_DAT_ROW_WIDTH}",
                    tokens.len()
                ),
            );
        }
        energy_hartree.push(parse_a2_f64(line_number, "energy", tokens[0])?);
        matrix_element.push(parse_a2_f64(line_number, "matrix element", tokens[1])?);
    }

    let data = DmdwA2DatData {
        energy_hartree: Array1::from_vec(energy_hartree),
        matrix_element: Array1::from_vec(matrix_element),
    };
    validate_dmdw_a2_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `dmdw_A2.dat` matrix-element text.
pub fn dmdw_a2_dat_string(data: &DmdwA2DatData) -> Result<String> {
    validate_dmdw_a2_dat(data)?;

    let mut out = String::new();
    for (&energy, &matrix_element) in data.energy_hartree.iter().zip(data.matrix_element.iter()) {
        write_fortran_zero_scaled_exp(&mut out, energy, 20, 10)?;
        write_fortran_zero_scaled_exp(&mut out, matrix_element, 20, 10)?;
        out.push('\n');
    }
    Ok(out)
}

/// Read FEFF DMDW `dmdw_A2.dat` text from disk.
pub fn read_dmdw_a2_dat(path: impl AsRef<Path>) -> Result<DmdwA2DatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_dmdw_a2_dat(&text)
}

/// Write FEFF DMDW `dmdw_A2.dat` text to disk.
pub fn write_dmdw_a2_dat(path: impl AsRef<Path>, data: &DmdwA2DatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, dmdw_a2_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_dmdw_coupling_table(data: &DmdwCouplingTable) -> Result<()> {
    if data.header_lines.len() != DMDW_COUPLING_HEADER_LINES {
        return invalid_dmdw_coupling_table(
            "header_lines",
            format!(
                "got {} header line(s), expected {DMDW_COUPLING_HEADER_LINES}",
                data.header_lines.len()
            ),
        );
    }
    for (index, line) in data.header_lines.iter().enumerate() {
        if line.contains(['\n', '\r']) {
            return invalid_dmdw_coupling_table(
                "header_lines",
                format!("header line {} contains an embedded newline", index + 1),
            );
        }
    }
    if data.point_count() == 0 {
        return invalid_dmdw_coupling_table("rows", "at least one coupling row is required");
    }
    if data.values.len() != data.point_count() {
        return invalid_dmdw_coupling_table(
            "values",
            format!(
                "got {} value(s), expected {}",
                data.values.len(),
                data.point_count()
            ),
        );
    }
    for (row, &energy) in data.energy_hartree.iter().enumerate() {
        validate_finite("energy", energy, row + 1)?;
    }
    for (row, &value) in data.values.iter().enumerate() {
        validate_finite("value", value, row + 1)?;
    }
    Ok(())
}

fn validate_dmdw_a2_dat(data: &DmdwA2DatData) -> Result<()> {
    if data.point_count() == 0 {
        return invalid_dmdw_a2_dat("rows", "at least one dmdw_A2.dat row is required");
    }
    if data.matrix_element.len() != data.point_count() {
        return invalid_dmdw_a2_dat(
            "matrix_element",
            format!(
                "got {} value(s), expected {}",
                data.matrix_element.len(),
                data.point_count()
            ),
        );
    }
    for (row, &energy) in data.energy_hartree.iter().enumerate() {
        validate_a2_finite("energy", energy, row + 1)?;
    }
    for (row, &value) in data.matrix_element.iter().enumerate() {
        validate_a2_finite("matrix element", value, row + 1)?;
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_dmdw_coupling_table(field, format!("row {row} value must be finite"))
    }
}

fn validate_a2_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_dmdw_a2_dat(field, format!("row {row} value must be finite"))
    }
}

fn parse_coupling_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    parse_f64(DMDW_COUPLING_TABLE_PATH, line, field, token)
}

fn parse_a2_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    parse_f64(DMDW_A2_DAT_PATH, line, field, token)
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| parse_error_value(path, line, format!("invalid {field} value {token:?}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error_value(
            path,
            line,
            format!("{field} value must be finite"),
        ))
    }
}

fn invalid_dmdw_coupling_table<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: DMDW_COUPLING_TABLE_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn invalid_dmdw_a2_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(IoError::Parse {
        path: DMDW_A2_DAT_PATH.into(),
        line: 0,
        message: format!("{field}: {}", message.into()),
    })
}

fn parse_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(DMDW_COUPLING_TABLE_PATH, line, message))
}

fn parse_a2_error<T>(line: usize, message: impl Into<String>) -> Result<T> {
    Err(parse_error_value(DMDW_A2_DAT_PATH, line, message))
}

fn parse_error_value(path: &'static str, line: usize, message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: path.into(),
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn parses_dmdw_coupling_table_with_feff_header_skip() -> Result<()> {
        let parsed = parse_dmdw_coupling_table(DMDW_COUPLING_TABLE)?;

        assert_eq!(parsed.header_lines[0], "# header 1");
        assert_eq!(parsed.header_lines[9], "# header 10");
        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.energy_hartree, array![0.001, 0.002, 0.004]);
        assert_eq!(parsed.values, array![10.0, 20.0, 30.0]);
        Ok(())
    }

    #[test]
    fn roundtrips_dmdw_coupling_table() -> Result<()> {
        let parsed = parse_dmdw_coupling_table(DMDW_COUPLING_TABLE)?;
        let rendered = dmdw_coupling_table_string(&parsed)?;
        let reparsed = parse_dmdw_coupling_table(&rendered)?;

        assert_eq!(reparsed, parsed);
        Ok(())
    }

    #[test]
    fn computes_dmdw_phonon_coupling_from_tables() -> anyhow::Result<()> {
        let pds = parse_dmdw_coupling_table(DMDW_COUPLING_TABLE)?;
        let a2f = parse_dmdw_coupling_table(DMDW_A2F_TABLE)?;

        let coupling = dmdw_phonon_coupling_from_tables(&pds, &a2f)?;

        assert_eq!(coupling.point_count(), 3);
        assert_eq!(coupling.matrix_element, array![0.05, 0.05, 0.05]);
        assert_eq!(coupling.eliashberg, array![0.5, 1.0, 1.5]);
        Ok(())
    }

    #[test]
    fn renders_and_parses_dmdw_a2_dat_from_coupling() -> anyhow::Result<()> {
        let pds = parse_dmdw_coupling_table(DMDW_COUPLING_TABLE)?;
        let a2f = parse_dmdw_coupling_table(DMDW_A2F_TABLE)?;
        let coupling = dmdw_phonon_coupling_from_tables(&pds, &a2f)?;

        let data = dmdw_a2_dat_from_coupling(&coupling)?;
        let rendered = dmdw_a2_dat_string(&data)?;
        let parsed = parse_dmdw_a2_dat(&rendered)?;

        assert_eq!(data.point_count(), 3);
        assert_eq!(parsed, data);
        assert_eq!(parsed.energy_hartree, array![0.001, 0.002, 0.004]);
        assert_eq!(parsed.matrix_element, array![0.05, 0.05, 0.05]);
        Ok(())
    }

    #[test]
    fn parses_list_directed_dmdw_a2_dat() -> Result<()> {
        let parsed = parse_dmdw_a2_dat(DMDW_A2_DAT)?;

        assert_eq!(parsed.point_count(), 3);
        assert_eq!(parsed.energy_hartree, array![0.001, 0.002, 0.004]);
        assert_eq!(parsed.matrix_element, array![0.05, 0.05, 0.05]);
        Ok(())
    }

    #[test]
    fn rejects_invalid_dmdw_coupling_tables() {
        assert!(parse_dmdw_coupling_table("# too short\n").is_err());
        assert!(parse_dmdw_coupling_table(BAD_WIDTH_TABLE).is_err());
        assert!(parse_dmdw_coupling_table(BAD_NONFINITE_TABLE).is_err());
        assert!(parse_dmdw_coupling_table(EMPTY_DATA_TABLE).is_err());

        let bad = DmdwCouplingTable {
            header_lines: vec!["# short".to_string()],
            energy_hartree: array![0.001],
            values: array![1.0],
        };
        assert!(dmdw_coupling_table_string(&bad).is_err());
    }

    #[test]
    fn rejects_invalid_dmdw_a2_dat_inputs() {
        assert!(parse_dmdw_a2_dat("").is_err());
        assert!(parse_dmdw_a2_dat("1.0 2.0 3.0\n").is_err());
        assert!(parse_dmdw_a2_dat("1.0 NaN\n").is_err());

        let bad = DmdwA2DatData {
            energy_hartree: array![0.001, 0.002],
            matrix_element: array![0.05],
        };
        assert!(dmdw_a2_dat_string(&bad).is_err());
    }

    const DMDW_COUPLING_TABLE: &str = concat!(
        "# header 1\n",
        "# header 2\n",
        "# header 3\n",
        "# header 4\n",
        "# header 5\n",
        "# header 6\n",
        "# header 7\n",
        "# header 8\n",
        "# header 9\n",
        "# header 10\n",
        " 1.0D-03 1.0D+01\n",
        " 2.0E-03 2.0E+01\n",
        " 4.0e-03 3.0e+01\n",
    );

    const DMDW_A2F_TABLE: &str = concat!(
        "# header 1\n",
        "# header 2\n",
        "# header 3\n",
        "# header 4\n",
        "# header 5\n",
        "# header 6\n",
        "# header 7\n",
        "# header 8\n",
        "# header 9\n",
        "# header 10\n",
        " 1.0D-03 5.0D-01\n",
        " 2.0E-03 1.0E+00\n",
        " 4.0e-03 1.5e+00\n",
    );

    const DMDW_A2_DAT: &str = concat!(
        " 1.0D-03 5.0D-02\n",
        " 2.0E-03 5.0E-02\n",
        " 4.0e-03 5.0e-02\n",
    );

    const BAD_WIDTH_TABLE: &str = concat!(
        "# header 1\n",
        "# header 2\n",
        "# header 3\n",
        "# header 4\n",
        "# header 5\n",
        "# header 6\n",
        "# header 7\n",
        "# header 8\n",
        "# header 9\n",
        "# header 10\n",
        "1.0 2.0 3.0\n",
    );

    const BAD_NONFINITE_TABLE: &str = concat!(
        "# header 1\n",
        "# header 2\n",
        "# header 3\n",
        "# header 4\n",
        "# header 5\n",
        "# header 6\n",
        "# header 7\n",
        "# header 8\n",
        "# header 9\n",
        "# header 10\n",
        "1.0 NaN\n",
    );

    const EMPTY_DATA_TABLE: &str = concat!(
        "# header 1\n",
        "# header 2\n",
        "# header 3\n",
        "# header 4\n",
        "# header 5\n",
        "# header 6\n",
        "# header 7\n",
        "# header 8\n",
        "# header 9\n",
        "# header 10\n",
    );
}
