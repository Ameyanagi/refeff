//! FEFF `ldosNN.dat` and `rhocNN.dat` local density-of-states text codecs.
//!
//! FEFF writes non-spin LDOS files with energy plus `s`, `p`, `d`, and `f`
//! orbital density columns. Spin-resolved LDOS output keeps the same energy
//! column and appends the four orbital channels for spin up followed by the
//! four orbital channels for spin down.
//!
//! The LDOS module also writes `rhocNN.dat` embedded-atom reference density
//! files from the same `ff2rho` data path. Those files omit the descriptive
//! header but keep the FEFF five-column energy plus angular-momentum table
//! shape, so the explicit `rhoc` helpers intentionally share this data model.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use refeff_core::FEFF_HARTREE_EV;

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const LDOS_DAT_NON_SPIN_ROW_WIDTH: usize = 5;
const LDOS_DAT_SPIN_ROW_WIDTH: usize = 9;
const LDOS_DAT_NON_SPIN_DENSITY_COLUMNS: usize = 4;
const LDOS_DAT_SPIN_DENSITY_COLUMNS: usize = 8;
const LDOS_DAT_ALLOWED_ROW_WIDTHS: &str = "5 or 9";
const LDOS_DAT_ALLOWED_DENSITY_COLUMNS: &str = "4 or 8";

/// FEFF non-spin LDOS column labels after the energy column.
pub const LDOS_ORBITAL_LABELS: [&str; LDOS_DAT_NON_SPIN_DENSITY_COLUMNS] =
    ["sDOS", "pDOS", "dDOS", "fDOS"];

/// FEFF spin-resolved LDOS column labels after the energy column.
pub const LDOS_SPIN_ORBITAL_LABELS: [&str; LDOS_DAT_SPIN_DENSITY_COLUMNS] = [
    "sDOS_up",
    "pDOS_up",
    "dDOS_up",
    "fDOS_up",
    "sDOS_down",
    "pDOS_down",
    "dDOS_down",
    "fDOS_down",
];

/// Parsed electron count from a FEFF `ldosNN.dat` header.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosElectronCount {
    /// Orbital angular momentum quantum number.
    pub angular_momentum: usize,
    /// Electron count in the corresponding angular momentum channel.
    pub count: f64,
}

/// Parsed FEFF `ldosNN.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct LdosDatData {
    /// Header and comment lines before and around the numeric LDOS table.
    pub header_lines: Vec<String>,
    /// Fermi level in eV when present in the header.
    pub fermi_level_ev: Option<f64>,
    /// Charge transfer when present in the header.
    pub charge_transfer: Option<f64>,
    /// Header electron counts by orbital angular momentum.
    pub electron_counts: Vec<LdosElectronCount>,
    /// Number of atoms in the cluster when present in the header.
    pub atom_count: Option<usize>,
    /// Lorentzian half-width at half-height broadening in eV when present.
    pub lorentzian_hwhh_ev: Option<f64>,
    /// Energy grid in eV.
    pub energy_ev: Array1<f64>,
    /// DOS columns. Non-spin files have four columns in [`LDOS_ORBITAL_LABELS`]
    /// order; spin-resolved files have eight columns in
    /// [`LDOS_SPIN_ORBITAL_LABELS`] order.
    pub density: Array2<f64>,
}

/// FEFF FULLSPECTRUM `rdldos.f90` view of an `ldosNN.dat` table.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSpectrumLdosData {
    /// Fermi level from the LDOS header, converted from eV to Hartree.
    pub fermi_level_hartree: f64,
    /// Photon-energy grid, converted from eV to Hartree.
    pub energy_hartree: Array1<f64>,
    /// `s`, `p`, `d`, and `f` DOS columns converted to states/Hartree/atom.
    pub density_states_per_hartree_atom: Array2<f64>,
}

/// Parsed FEFF `rhocNN.dat` contents.
///
/// FEFF `rhocNN.dat` files use the same energy-grid and angular-momentum
/// density table as non-spin `ldosNN.dat`, usually without header metadata.
pub type RhocDatData = LdosDatData;

impl LdosDatData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether this table has spin-resolved up/down orbital channels.
    #[must_use]
    pub fn is_spin_resolved(&self) -> bool {
        self.density.ncols() == LDOS_DAT_SPIN_DENSITY_COLUMNS
    }
}

impl FullSpectrumLdosData {
    /// Number of energy-grid rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }

    /// Number of angular momentum channels.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.density_states_per_hartree_atom.ncols()
    }
}

/// Render FEFF-compatible `ldosNN.dat` text.
pub fn ldos_dat_string(data: &LdosDatData) -> Result<String> {
    validate_ldos_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data.energy_ev.iter().zip(data.density.axis_iter(Axis(0))) {
        write!(out, "{energy:11.4} ")?;
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                out.push(' ');
            }
            write_fortran_exp(&mut out, *value, 13, 6)?;
        }
        out.push('\n');
    }
    Ok(out)
}

/// Render FEFF-compatible `rhocNN.dat` text.
///
/// The rendered table uses the same numeric layout as [`ldos_dat_string`].
pub fn rhoc_dat_string(data: &RhocDatData) -> Result<String> {
    ldos_dat_string(data)
}

/// Parse FEFF `ldosNN.dat` text.
pub fn parse_ldos_dat(text: &str) -> Result<LdosDatData> {
    let mut header_lines = Vec::new();
    let mut fermi_level_ev = None;
    let mut charge_transfer = None;
    let mut electron_counts = Vec::new();
    let mut atom_count = None;
    let mut lorentzian_hwhh_ev = None;
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut density = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(width, LDOS_DAT_NON_SPIN_ROW_WIDTH | LDOS_DAT_SPIN_ROW_WIDTH) {
                return Err(IoError::LdosDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: LDOS_DAT_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::LdosDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: row_width_label(expected),
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            for token in &tokens[1..] {
                density.push(parse_f64(line_number, "density", token)?);
            }
        } else {
            parse_header_metadata(
                line,
                line_number,
                &mut fermi_level_ev,
                &mut charge_transfer,
                &mut electron_counts,
                &mut atom_count,
                &mut lorentzian_hwhh_ev,
            )?;
            header_lines.push(raw.to_string());
        }
    }

    let point_count = energy_ev.len();
    let density_columns = match row_width {
        Some(LDOS_DAT_NON_SPIN_ROW_WIDTH) => LDOS_DAT_NON_SPIN_DENSITY_COLUMNS,
        Some(LDOS_DAT_SPIN_ROW_WIDTH) => LDOS_DAT_SPIN_DENSITY_COLUMNS,
        _ => 0,
    };
    let density = if density_columns == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, density_columns), density).map_err(|_| {
            invalid_ldos_dat("density", "density payload did not match LDOS table shape")
        })?
    };

    let data = LdosDatData {
        header_lines,
        fermi_level_ev,
        charge_transfer,
        electron_counts,
        atom_count,
        lorentzian_hwhh_ev,
        energy_ev: Array1::from_vec(energy_ev),
        density,
    };
    validate_ldos_dat(&data)?;
    Ok(data)
}

/// Parse FEFF `rhocNN.dat` embedded-density text.
///
/// FEFF writes `rhocNN.dat` with the same five-column table accepted by
/// [`parse_ldos_dat`].
pub fn parse_rhoc_dat(text: &str) -> Result<RhocDatData> {
    parse_ldos_dat(text)
}

/// Convert parsed `ldosNN.dat` content to FEFF `FULLSPECTRUM/rdldos.f90` units.
///
/// The FULLSPECTRUM reader consumes only non-spin LDOS tables with four
/// angular-momentum columns. It converts energies from eV to Hartree and DOS
/// values from states/eV/atom to states/Hartree/atom.
pub fn fullspectrum_ldos_from_ldos_dat(data: &LdosDatData) -> Result<FullSpectrumLdosData> {
    validate_ldos_dat(data)?;
    if data.point_count() < 2 {
        return Err(invalid_ldos_dat(
            "rows",
            "FULLSPECTRUM rdldos requires at least two LDOS rows",
        ));
    }
    if data.is_spin_resolved() {
        return Err(invalid_ldos_dat(
            "density",
            "FULLSPECTRUM rdldos supports only non-spin four-column LDOS data",
        ));
    }
    let fermi_level_ev = data.fermi_level_ev.ok_or_else(|| {
        invalid_ldos_dat(
            "fermi level",
            "FULLSPECTRUM rdldos requires a Fermi-level header",
        )
    })?;

    let fermi_level_hartree = fermi_level_ev / FEFF_HARTREE_EV;
    validate_finite("fermi_level_hartree", fermi_level_hartree)?;

    let energy_hartree = data.energy_ev.mapv(|energy| energy / FEFF_HARTREE_EV);
    for (row, energy) in energy_hartree.iter().copied().enumerate() {
        validate_finite_row("energy_hartree", energy, row + 1)?;
    }

    let density_states_per_hartree_atom = data.density.mapv(|density| density * FEFF_HARTREE_EV);
    for (index, density) in density_states_per_hartree_atom.iter().copied().enumerate() {
        let row = index / LDOS_DAT_NON_SPIN_DENSITY_COLUMNS + 1;
        validate_finite_row("density_states_per_hartree_atom", density, row)?;
    }

    Ok(FullSpectrumLdosData {
        fermi_level_hartree,
        energy_hartree,
        density_states_per_hartree_atom,
    })
}

/// Write FEFF `ldosNN.dat` text to a file.
pub fn write_ldos_dat(path: impl AsRef<Path>, data: &LdosDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, ldos_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Write FEFF `rhocNN.dat` text to a file.
pub fn write_rhoc_dat(path: impl AsRef<Path>, data: &RhocDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhoc_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `ldosNN.dat` text from a file.
pub fn read_ldos_dat(path: impl AsRef<Path>) -> Result<LdosDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_ldos_dat(&text)
}

/// Read FEFF `rhocNN.dat` embedded-density text from a file.
pub fn read_rhoc_dat(path: impl AsRef<Path>) -> Result<RhocDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rhoc_dat(&text)
}

fn parse_header_metadata(
    line: &str,
    line_number: usize,
    fermi_level_ev: &mut Option<f64>,
    charge_transfer: &mut Option<f64>,
    electron_counts: &mut Vec<LdosElectronCount>,
    atom_count: &mut Option<usize>,
    lorentzian_hwhh_ev: &mut Option<f64>,
) -> Result<()> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("fermi level") {
        *fermi_level_ev = Some(parse_f64(
            line_number,
            "fermi level",
            last_numeric_token(line)
                .ok_or_else(|| invalid_ldos_dat("fermi level", "missing numeric header value"))?,
        )?);
    } else if lower.contains("charge transfer") {
        *charge_transfer = Some(parse_f64(
            line_number,
            "charge transfer",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("charge transfer", "missing numeric header value")
            })?,
        )?);
    } else if lower.contains("number of atoms in cluster") {
        *atom_count = Some(parse_usize(
            line_number,
            "number of atoms",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("number of atoms", "missing numeric header value")
            })?,
        )?);
    } else if lower.contains("lorentzian broadening") {
        *lorentzian_hwhh_ev = Some(parse_f64(
            line_number,
            "lorentzian broadening",
            last_numeric_token(line).ok_or_else(|| {
                invalid_ldos_dat("lorentzian broadening", "missing numeric header value")
            })?,
        )?);
    } else if let Some(count) = parse_electron_count_header(line, line_number)? {
        electron_counts.push(count);
    }
    Ok(())
}

fn parse_electron_count_header(
    line: &str,
    line_number: usize,
) -> Result<Option<LdosElectronCount>> {
    let stripped = line.trim_start().strip_prefix('#').map(str::trim);
    let Some(stripped) = stripped else {
        return Ok(None);
    };
    let tokens = stripped.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 2 || !is_usize_token(tokens[0]) || !is_numeric_token(tokens[1]) {
        return Ok(None);
    }
    Ok(Some(LdosElectronCount {
        angular_momentum: parse_usize(line_number, "electron count angular momentum", tokens[0])?,
        count: parse_f64(line_number, "electron count", tokens[1])?,
    }))
}

fn validate_ldos_dat(data: &LdosDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_ldos_dat(
            "rows",
            "at least one LDOS row is required",
        ));
    }
    let (rows, cols) = data.density.dim();
    if rows != point_count {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_DAT_ALLOWED_DENSITY_COLUMNS,
        });
    }
    if !matches!(
        cols,
        LDOS_DAT_NON_SPIN_DENSITY_COLUMNS | LDOS_DAT_SPIN_DENSITY_COLUMNS
    ) {
        return Err(IoError::LdosDatShape {
            field: "density",
            rows,
            cols,
            expected_rows: point_count,
            expected_cols: LDOS_DAT_ALLOWED_DENSITY_COLUMNS,
        });
    }

    if let Some(value) = data.fermi_level_ev {
        validate_finite("fermi level", value)?;
    }
    if let Some(value) = data.charge_transfer {
        validate_finite("charge transfer", value)?;
    }
    if let Some(value) = data.lorentzian_hwhh_ev {
        validate_finite("lorentzian broadening", value)?;
    }
    for count in &data.electron_counts {
        validate_finite("electron count", count.count)?;
    }

    for (row, energy) in data.energy_ev.iter().enumerate() {
        validate_finite_row("energy", *energy, row + 1)?;
    }
    for (index, value) in data.density.iter().enumerate() {
        let row = index / cols + 1;
        validate_finite_row("density", *value, row)?;
    }
    Ok(())
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::LdosDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn parse_usize(line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| IoError::LdosDatParse {
        field,
        line,
        token: token.to_string(),
    })
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_ldos_dat(field, "value must be finite"))
    }
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidLdosDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_ldos_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidLdosDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

fn is_usize_token(token: &str) -> bool {
    token.parse::<usize>().is_ok()
}

fn last_numeric_token(line: &str) -> Option<&str> {
    line.split_whitespace()
        .rev()
        .find(|token| is_numeric_token(token))
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        LDOS_DAT_NON_SPIN_ROW_WIDTH => "5",
        LDOS_DAT_SPIN_ROW_WIDTH => "9",
        _ => LDOS_DAT_ALLOWED_ROW_WIDTHS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feff_ldos_reference_shape_and_metadata() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        assert_eq!(data.point_count(), 3);
        assert!(!data.is_spin_resolved());
        assert_eq!(data.fermi_level_ev, Some(-14.683));
        assert_eq!(data.charge_transfer, Some(0.711));
        assert_eq!(data.atom_count, Some(0));
        assert_eq!(data.lorentzian_hwhh_ev, Some(0.0100));
        assert_eq!(data.electron_counts.len(), 4);
        assert_eq!(data.electron_counts[2].angular_momentum, 2);
        assert_eq!(data.electron_counts[2].count, 10.223);
        assert_eq!(data.energy_ev[0], -30.0);
        assert_eq!(data.density[[0, 0]], 1.342776e-4);
        assert_eq!(data.density[[1, 3]], 2.564170e-5);
        Ok(())
    }

    #[test]
    fn parses_spin_resolved_ldos_shape() -> Result<()> {
        let data = parse_ldos_dat(SPIN_LDOS_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(data.is_spin_resolved());
        assert_eq!(data.density.ncols(), LDOS_SPIN_ORBITAL_LABELS.len());
        assert_eq!(data.density[[0, 4]], 5.0e-3);
        assert_eq!(data.density[[1, 7]], 1.6e-2);
        Ok(())
    }

    #[test]
    fn roundtrips_ldos_text() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        let rendered = ldos_dat_string(&data)?;
        assert_eq!(parse_ldos_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn parses_and_roundtrips_rhoc_text() -> Result<()> {
        let data = parse_rhoc_dat(RHOC_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(!data.is_spin_resolved());
        assert_eq!(data.header_lines.len(), 0);
        assert_eq!(data.energy_ev[0], -10.0);
        assert_eq!(data.density[[1, 3]], 8.0);
        let rendered = rhoc_dat_string(&data)?;
        assert_eq!(parse_rhoc_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn converts_ldos_to_feff_fullspectrum_rdldos_units() -> Result<()> {
        let data = parse_ldos_dat(LDOS_DAT)?;
        let fullspectrum = fullspectrum_ldos_from_ldos_dat(&data)?;

        assert_eq!(fullspectrum.point_count(), 3);
        assert_eq!(fullspectrum.angular_count(), 4);
        assert_close(fullspectrum.fermi_level_hartree, -14.683 / FEFF_HARTREE_EV);
        assert_close(fullspectrum.energy_hartree[0], -30.0 / FEFF_HARTREE_EV);
        assert_close(
            fullspectrum.density_states_per_hartree_atom[[0, 0]],
            1.342_776E-04 * FEFF_HARTREE_EV,
        );
        assert_close(
            fullspectrum.density_states_per_hartree_atom[[1, 3]],
            2.564_170E-05 * FEFF_HARTREE_EV,
        );
        Ok(())
    }

    #[test]
    fn rejects_ldos_tables_not_supported_by_fullspectrum_rdldos() -> Result<()> {
        let spin = parse_ldos_dat(SPIN_LDOS_DAT)?;
        assert!(fullspectrum_ldos_from_ldos_dat(&spin).is_err());

        let missing_fermi = parse_rhoc_dat(RHOC_DAT)?;
        assert!(fullspectrum_ldos_from_ldos_dat(&missing_fermi).is_err());

        let one_row = LdosDatData {
            header_lines: vec!["#  Fermi level (eV): -14.683".to_string()],
            fermi_level_ev: Some(-14.683),
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: Array1::from_vec(vec![1.0]),
            density: Array2::zeros((1, LDOS_DAT_NON_SPIN_DENSITY_COLUMNS)),
        };
        assert!(fullspectrum_ldos_from_ldos_dat(&one_row).is_err());
        Ok(())
    }

    #[test]
    fn rejects_bad_ldos_inputs() {
        assert!(parse_ldos_dat("# no data\n").is_err());
        assert!(parse_ldos_dat("1 2 3 4\n").is_err());
        assert!(parse_ldos_dat("1 2 3 NaN 5\n").is_err());
        assert!(parse_ldos_dat("1 2 3 4 5\n2 3 4 5 6 7 8 9 10\n").is_err());

        let bad_shape = LdosDatData {
            header_lines: Vec::new(),
            fermi_level_ev: None,
            charge_transfer: None,
            electron_counts: Vec::new(),
            atom_count: None,
            lorentzian_hwhh_ev: None,
            energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            density: Array2::zeros((1, LDOS_DAT_NON_SPIN_DENSITY_COLUMNS)),
        };
        assert!(ldos_dat_string(&bad_shape).is_err());
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    const LDOS_DAT: &str = r#"#  Fermi level (eV): -14.683
#  Charge transfer :   0.711
#    Electron counts for each orbital momentum:
#       0      1.428
#       1      1.637
#       2     10.223
#       3      0.000
#  Number of atoms in cluster:   0
#  Lorentzian broadening with HWHH     0.0100 eV
# -----------------------------------------------------------------------
#      e        sDOS           pDOS          dDOS          fDOS    @#
   -30.0000  1.342776E-04  7.462376E-05  4.190590E-04  2.449744E-05
   -29.5500  1.777093E-04  8.534908E-05  3.865858E-04  2.564170E-05
   -29.1000  3.428764E-04  1.101810E-04  3.605636E-04  2.691077E-05
"#;

    const SPIN_LDOS_DAT: &str = r#"#      e        sDOS(up)   pDOS(up)      dDOS(up)    fDOS(up)   sDOS(down)    pDOS(down)   dDOS9(down)   fDOS(down)    @#
     1.0000  1.000000E-03  2.000000E-03  3.000000E-03  4.000000E-03  5.000000E-03  6.000000E-03  7.000000E-03  8.000000E-03
     2.0000  2.000000E-03  4.000000E-03  6.000000E-03  8.000000E-03  1.000000E-02  1.200000E-02  1.400000E-02  1.600000E-02
"#;

    const RHOC_DAT: &str = r#"    -10.0000  1.000000E+00  2.000000E+00  3.000000E+00  4.000000E+00
     -9.5000  5.000000E+00  6.000000E+00  7.000000E+00  8.000000E+00
"#;
}
