//! FEFF potential-stage diagnostic output support.
//!
//! The potential stage writes compact and detailed SCF convergence traces as
//! `convergence.scf` and `convergence.scf.fine`. The atomic stage also appends
//! total-energy diagnostics to `fort.16`. These formats are small, but parsing
//! them into typed data makes generated FEFF10 potential runs easy to check.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;

use crate::error::{IoError, Result};

const CONVERGENCE_SCF_PATH: &str = "convergence.scf";
const CONVERGENCE_SCF_FINE_PATH: &str = "convergence.scf.fine";
const FORT16_PATH: &str = "fort.16";

/// Parsed contents of FEFF `convergence.scf` or `convergence.scf.fine`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfConvergenceData {
    /// Non-row lines, including headers and fine-detail blocks.
    pub detail_lines: Vec<String>,
    /// Parsed five-column convergence rows in file order.
    pub rows: Vec<ScfConvergenceRow>,
}

impl ScfConvergenceData {
    /// Number of parsed convergence rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// One SCF convergence row.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfConvergenceRow {
    /// SCF iteration index.
    pub iteration: usize,
    /// Fermi level in eV.
    pub fermi_level_ev: f64,
    /// Maximum charge-distance convergence metric.
    pub charge_distance: f64,
    /// Maximum partial-charge-distance convergence metric.
    pub partial_charge_distance: f64,
    /// Whether FEFF considered the row converged.
    pub converged: bool,
}

/// Parsed contents of FEFF `fort.16` total-energy diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct Fort16Data {
    /// Total energies in Hartree in file order.
    pub total_energy_hartree: Array1<f64>,
}

impl Fort16Data {
    /// Number of total-energy rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.total_energy_hartree.len()
    }
}

/// Parse FEFF `convergence.scf` text.
pub fn parse_convergence_scf(text: &str) -> Result<ScfConvergenceData> {
    parse_scf_convergence(text, CONVERGENCE_SCF_PATH)
}

/// Parse FEFF `convergence.scf.fine` text.
pub fn parse_convergence_scf_fine(text: &str) -> Result<ScfConvergenceData> {
    parse_scf_convergence(text, CONVERGENCE_SCF_FINE_PATH)
}

/// Parse FEFF `fort.16` text.
pub fn parse_fort16(text: &str) -> Result<Fort16Data> {
    let mut energies = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((label, value)) = line.split_once(':') else {
            return parse_error(
                FORT16_PATH,
                line_number,
                format!("expected total-energy line, found {line:?}"),
            );
        };
        if !label.trim().eq_ignore_ascii_case("Total energy") {
            return parse_error(
                FORT16_PATH,
                line_number,
                format!("expected Total energy label, found {label:?}"),
            );
        }
        energies.push(parse_f64(
            FORT16_PATH,
            line_number,
            "total_energy_hartree",
            value.trim(),
        )?);
    }

    let data = Fort16Data {
        total_energy_hartree: Array1::from_vec(energies),
    };
    validate_fort16(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `convergence.scf` text.
pub fn convergence_scf_string(data: &ScfConvergenceData) -> Result<String> {
    scf_convergence_string(data, CONVERGENCE_SCF_PATH)
}

/// Render FEFF-compatible `convergence.scf.fine` text.
pub fn convergence_scf_fine_string(data: &ScfConvergenceData) -> Result<String> {
    scf_convergence_string(data, CONVERGENCE_SCF_FINE_PATH)
}

/// Render FEFF-compatible `fort.16` text.
pub fn fort16_string(data: &Fort16Data) -> Result<String> {
    validate_fort16(data)?;
    let mut out = String::new();
    for value in &data.total_energy_hartree {
        writeln!(out, " Total energy: {:24.16}", value)?;
    }
    Ok(out)
}

/// Read FEFF `convergence.scf` text from a file.
pub fn read_convergence_scf(path: impl AsRef<Path>) -> Result<ScfConvergenceData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_convergence_scf(&text)
}

/// Read FEFF `convergence.scf.fine` text from a file.
pub fn read_convergence_scf_fine(path: impl AsRef<Path>) -> Result<ScfConvergenceData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_convergence_scf_fine(&text)
}

/// Read FEFF `fort.16` text from a file.
pub fn read_fort16(path: impl AsRef<Path>) -> Result<Fort16Data> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_fort16(&text)
}

/// Write FEFF `convergence.scf` text to a file.
pub fn write_convergence_scf(path: impl AsRef<Path>, data: &ScfConvergenceData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, convergence_scf_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `convergence.scf.fine` text to a file.
pub fn write_convergence_scf_fine(path: impl AsRef<Path>, data: &ScfConvergenceData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, convergence_scf_fine_string(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Write FEFF `fort.16` text to a file.
pub fn write_fort16(path: impl AsRef<Path>, data: &Fort16Data) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, fort16_string(data)?).map_err(|source| IoError::io(path, source))
}

fn parse_scf_convergence(text: &str, path: &'static str) -> Result<ScfConvergenceData> {
    let mut detail_lines = Vec::new();
    let mut rows = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if is_convergence_row(&tokens) {
            rows.push(ScfConvergenceRow {
                iteration: parse_usize(path, line_number, "iteration", tokens[0])?,
                fermi_level_ev: parse_f64(path, line_number, "fermi_level_ev", tokens[1])?,
                charge_distance: parse_f64(path, line_number, "charge_distance", tokens[2])?,
                partial_charge_distance: parse_f64(
                    path,
                    line_number,
                    "partial_charge_distance",
                    tokens[3],
                )?,
                converged: parse_converged_flag(path, line_number, tokens[4])?,
            });
        } else {
            detail_lines.push(line.to_owned());
        }
    }

    let data = ScfConvergenceData { detail_lines, rows };
    validate_scf_convergence(path, &data)?;
    Ok(data)
}

fn scf_convergence_string(data: &ScfConvergenceData, path: &'static str) -> Result<String> {
    validate_scf_convergence(path, data)?;
    let mut out = String::new();
    for line in &data.detail_lines {
        writeln!(out, "{line}")?;
    }
    for row in &data.rows {
        writeln!(
            out,
            "{:4}{:12.3}{:15.4}{:15.4}{:6}",
            row.iteration,
            row.fermi_level_ev,
            row.charge_distance,
            row.partial_charge_distance,
            u8::from(row.converged)
        )?;
    }
    Ok(out)
}

fn is_convergence_row(tokens: &[&str]) -> bool {
    tokens.len() == 5
        && tokens[0].parse::<usize>().is_ok()
        && tokens[1].replace(['D', 'd'], "E").parse::<f64>().is_ok()
        && tokens[2].replace(['D', 'd'], "E").parse::<f64>().is_ok()
        && tokens[3].replace(['D', 'd'], "E").parse::<f64>().is_ok()
        && matches!(tokens[4], "0" | "1")
}

fn validate_scf_convergence(path: &'static str, data: &ScfConvergenceData) -> Result<()> {
    for (index, row) in data.rows.iter().enumerate() {
        let line = index + 1;
        validate_finite(path, line, "fermi_level_ev", row.fermi_level_ev)?;
        validate_finite(path, line, "charge_distance", row.charge_distance)?;
        validate_finite(
            path,
            line,
            "partial_charge_distance",
            row.partial_charge_distance,
        )?;
    }
    Ok(())
}

fn validate_fort16(data: &Fort16Data) -> Result<()> {
    if data.total_energy_hartree.is_empty() {
        return parse_error(FORT16_PATH, 0, "at least one total energy is required");
    }
    for (index, value) in data.total_energy_hartree.iter().enumerate() {
        validate_finite(FORT16_PATH, index + 1, "total_energy_hartree", *value)?;
    }
    Ok(())
}

fn parse_converged_flag(path: &'static str, line: usize, token: &str) -> Result<bool> {
    match token {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => parse_error(
            path,
            line,
            format!("could not parse converged flag from {token:?}"),
        ),
    }
}

fn parse_f64(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<f64> {
    let value = token.replace(['D', 'd'], "E").parse::<f64>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })?;
    validate_finite(path, line, field, value)?;
    Ok(value)
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

fn validate_finite(path: &'static str, line: usize, field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, line, format!("{field} must be finite"))
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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn parses_convergence_scf() -> Result<()> {
        let parsed = parse_convergence_scf(CONVERGENCE_SCF)?;
        assert_eq!(parsed.detail_lines.len(), 1);
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.rows[0].iteration, 0);
        assert_eq!(parsed.rows[0].fermi_level_ev, -4.006);
        assert_eq!(parsed.rows[1].charge_distance, 0.3252);
        assert_eq!(parsed.rows[2].partial_charge_distance, 0.5599);
        assert!(!parsed.rows[1].converged);

        let rendered = convergence_scf_string(&parsed)?;
        assert_eq!(parse_convergence_scf(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn parses_convergence_scf_fine_details() -> Result<()> {
        let parsed = parse_convergence_scf_fine(CONVERGENCE_SCF_FINE)?;
        assert_eq!(parsed.row_count(), 2);
        assert!(
            parsed
                .detail_lines
                .iter()
                .any(|line| line == "Electronic configuration")
        );
        assert!(
            parsed
                .detail_lines
                .iter()
                .any(|line| line == "0     2   10.466")
        );

        let rendered = convergence_scf_fine_string(&parsed)?;
        assert_eq!(parse_convergence_scf_fine(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_empty_and_header_only_convergence_files() -> Result<()> {
        let empty = parse_convergence_scf_fine("")?;
        assert_eq!(empty.row_count(), 0);
        let header = parse_convergence_scf("# it. E_fermi(eV)  Charge Distance\n")?;
        assert_eq!(header.detail_lines.len(), 1);
        assert_eq!(header.row_count(), 0);
        Ok(())
    }

    #[test]
    fn parses_fort16() -> Result<()> {
        let parsed = parse_fort16(FORT16)?;
        assert_eq!(parsed.row_count(), 3);
        assert_eq!(parsed.total_energy_hartree[0], -1_322.522_518_926_127_5);
        assert_eq!(parsed.total_energy_hartree[2], -1_652.786_043_284_159_6);

        let rendered = fort16_string(&parsed)?;
        assert_eq!(parse_fort16(&rendered)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_convergence_scf("1 -4.0D+00 1.0D-01 2.0D-02 1\n")?;
        assert_eq!(parsed.rows[0].fermi_level_ev, -4.0);
        assert!(parsed.rows[0].converged);

        let energies = parse_fort16("Total energy: -1.0D+03\n")?;
        assert_eq!(energies.total_energy_hartree[0], -1000.0);
        Ok(())
    }

    #[test]
    fn rejects_bad_pot_diagnostics() {
        assert!(parse_fort16("").is_err());
        assert!(parse_fort16("Energy: 1\n").is_err());
        assert!(parse_fort16("Total energy: NaN\n").is_err());
        assert!(
            fort16_string(&Fort16Data {
                total_energy_hartree: array![1.0, f64::NAN],
            })
            .is_err()
        );
    }

    const CONVERGENCE_SCF: &str = r#" # it. E_fermi(eV)  Charge Distance  Partial Chg. D.  Convergence
   0      -4.006          0.000         0.0000     0
   1      -3.480         0.3252        10.4658     0
   2      -7.136         0.0623         0.5599     0
"#;

    const CONVERGENCE_SCF_FINE: &str = r#" SCF ITERATION NUMBER  1
   Electronic configuration
   type     l     N_el
      0     0    1.159
      0     1    2.006
      0     2   10.466
      0     3    0.000
  Charge transfer:  type  charge 
        0   -0.325
        1    0.002
   0      -4.006          0.000         0.0000     0
   1      -3.480         0.3252        10.4658     0
"#;

    const FORT16: &str = r#" Total energy:   -1322.5225189261275     
 Total energy:   -1652.7860432841596     
 Total energy:   -1652.7860432841596     
"#;
}
