//! FEFF `axafs.dat` atomic-XAFS sidecar codec.
//!
//! `XSPH/axafs.f90` writes this optional table when XSPH print level requests
//! AXAFS diagnostics. The table has the same six physical columns as the
//! [`refeff_core::xsph_axafs`] helper returns: absolute energy, edge-relative
//! energy, signed wave number, atomic absorption, fitted atomic background, and
//! atomic fine structure.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, ArrayView2};
use refeff_core::{XSPH_AXAFS_COLUMN_COUNT, XsphAxafs};

use crate::error::{IoError, Result};
use crate::format::write_fortran_exp;

const AXAFS_ROW_WIDTH: usize = XSPH_AXAFS_COLUMN_COUNT;

/// Parsed FEFF `axafs.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct AxafsDatData {
    /// Header and comment lines before the numeric AXAFS table.
    pub header_lines: Vec<String>,
    /// Absolute energy in eV.
    pub energy_ev: Array1<f64>,
    /// Energy relative to the edge in eV.
    pub edge_relative_energy_ev: Array1<f64>,
    /// Signed photoelectron wave number in inverse Angstroms.
    pub wave_number_inverse_angstrom: Array1<f64>,
    /// Normalized atomic absorption, `mu_at`.
    pub atomic_absorption: Array1<f64>,
    /// Normalized fitted atomic background, `mu0_at`.
    pub atomic_background: Array1<f64>,
    /// Atomic fine structure, `chi_at`.
    pub chi_atomic: Array1<f64>,
}

impl AxafsDatData {
    /// Number of AXAFS rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }
}

/// Build `axafs.dat` data from the core XSPH AXAFS helper output.
pub fn axafs_dat_from_xsph_axafs(axafs: &XsphAxafs) -> Result<AxafsDatData> {
    axafs_dat_from_rows(axafs.rows.view())
}

/// Build `axafs.dat` data from a six-column AXAFS table.
pub fn axafs_dat_from_rows(rows: ArrayView2<'_, f64>) -> Result<AxafsDatData> {
    if rows.ncols() != AXAFS_ROW_WIDTH {
        return Err(IoError::AxafsDatShape {
            field: "rows",
            actual: rows.ncols(),
            expected: AXAFS_ROW_WIDTH,
        });
    }

    let data = AxafsDatData {
        header_lines: default_axafs_header_lines(),
        energy_ev: rows.column(0).to_owned(),
        edge_relative_energy_ev: rows.column(1).to_owned(),
        wave_number_inverse_angstrom: rows.column(2).to_owned(),
        atomic_absorption: rows.column(3).to_owned(),
        atomic_background: rows.column(4).to_owned(),
        chi_atomic: rows.column(5).to_owned(),
    };
    validate_axafs_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `axafs.dat` text.
pub fn axafs_dat_string(data: &AxafsDatData) -> Result<String> {
    validate_axafs_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (((((energy, relative), wave_number), absorption), background), chi) in data
        .energy_ev
        .iter()
        .zip(data.edge_relative_energy_ev.iter())
        .zip(data.wave_number_inverse_angstrom.iter())
        .zip(data.atomic_absorption.iter())
        .zip(data.atomic_background.iter())
        .zip(data.chi_atomic.iter())
    {
        write!(out, " {:11.3}{:11.3}{:8.3}", energy, relative, wave_number)?;
        write_fortran_exp(&mut out, *absorption, 13, 5)?;
        write_fortran_exp(&mut out, *background, 13, 5)?;
        write_fortran_exp(&mut out, *chi, 13, 5)?;
        out.push('\n');
    }
    Ok(out)
}

/// Parse FEFF `axafs.dat` text.
pub fn parse_axafs_dat(text: &str) -> Result<AxafsDatData> {
    let mut header_lines = Vec::new();
    let mut energy_ev = Vec::new();
    let mut edge_relative_energy_ev = Vec::new();
    let mut wave_number_inverse_angstrom = Vec::new();
    let mut atomic_absorption = Vec::new();
    let mut atomic_background = Vec::new();
    let mut chi_atomic = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            header_lines.push(raw.trim_end().to_string());
            continue;
        }

        let line_number = index + 1;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != AXAFS_ROW_WIDTH {
            return Err(IoError::AxafsDatRowWidth {
                line: line_number,
                actual: tokens.len(),
                expected: AXAFS_ROW_WIDTH,
            });
        }
        energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
        edge_relative_energy_ev.push(parse_f64(line_number, "edge_relative_energy", tokens[1])?);
        wave_number_inverse_angstrom.push(parse_f64(line_number, "wave_number", tokens[2])?);
        atomic_absorption.push(parse_f64(line_number, "atomic_absorption", tokens[3])?);
        atomic_background.push(parse_f64(line_number, "atomic_background", tokens[4])?);
        chi_atomic.push(parse_f64(line_number, "chi_atomic", tokens[5])?);
    }

    let data = AxafsDatData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        edge_relative_energy_ev: Array1::from_vec(edge_relative_energy_ev),
        wave_number_inverse_angstrom: Array1::from_vec(wave_number_inverse_angstrom),
        atomic_absorption: Array1::from_vec(atomic_absorption),
        atomic_background: Array1::from_vec(atomic_background),
        chi_atomic: Array1::from_vec(chi_atomic),
    };
    validate_axafs_dat(&data)?;
    Ok(data)
}

/// Read FEFF `axafs.dat` text from a file.
pub fn read_axafs_dat(path: impl AsRef<Path>) -> Result<AxafsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_axafs_dat(&text)
}

/// Write FEFF `axafs.dat` text to a file.
pub fn write_axafs_dat(path: impl AsRef<Path>, data: &AxafsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, axafs_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_axafs_dat(data: &AxafsDatData) -> Result<()> {
    let point_count = data.energy_ev.len();
    if point_count == 0 {
        return Err(IoError::InvalidAxafsDat {
            field: "rows",
            message: "at least one AXAFS row is required".to_string(),
        });
    }
    validate_len(
        "edge_relative_energy",
        data.edge_relative_energy_ev.len(),
        point_count,
    )?;
    validate_len(
        "wave_number",
        data.wave_number_inverse_angstrom.len(),
        point_count,
    )?;
    validate_len(
        "atomic_absorption",
        data.atomic_absorption.len(),
        point_count,
    )?;
    validate_len(
        "atomic_background",
        data.atomic_background.len(),
        point_count,
    )?;
    validate_len("chi_atomic", data.chi_atomic.len(), point_count)?;

    for (row, (((((energy, relative), wave_number), absorption), background), chi)) in data
        .energy_ev
        .iter()
        .zip(data.edge_relative_energy_ev.iter())
        .zip(data.wave_number_inverse_angstrom.iter())
        .zip(data.atomic_absorption.iter())
        .zip(data.atomic_background.iter())
        .zip(data.chi_atomic.iter())
        .enumerate()
    {
        validate_finite("energy", row, *energy)?;
        validate_finite("edge_relative_energy", row, *relative)?;
        validate_finite("wave_number", row, *wave_number)?;
        validate_finite("atomic_absorption", row, *absorption)?;
        validate_finite("atomic_background", row, *background)?;
        validate_finite("chi_atomic", row, *chi)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::AxafsDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn validate_finite(field: &'static str, row: usize, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidAxafsDat {
            field,
            message: format!("row {} is not finite", row + 1),
        })
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::AxafsDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn default_axafs_header_lines() -> Vec<String> {
    vec![
        " # File contains AXAFS. See manual for details.".to_string(),
        " #--------------------------------------------------------------".to_string(),
        " #  e, e(wrt edge), k, mu_at=(1+chi_at)*mu0_at, mu0_at, chi_at @#".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        AxafsDatData, axafs_dat_from_rows, axafs_dat_from_xsph_axafs, axafs_dat_string,
        parse_axafs_dat,
    };
    use ndarray::{Array1, arr2};
    use refeff_core::XsphAxafs;

    #[test]
    fn parses_feff_axafs_reference_shape() -> crate::Result<()> {
        let data = parse_axafs_dat(AXAFS_DAT)?;

        assert_eq!(data.point_count(), 2);
        assert_eq!(data.header_lines.len(), 3);
        assert_eq!(data.energy_ev[0], 8979.0);
        assert_eq!(data.edge_relative_energy_ev[1], 1.5);
        assert_eq!(data.wave_number_inverse_angstrom[1], 0.627);
        assert_eq!(data.atomic_absorption[0], 1.23456);
        assert_eq!(data.atomic_background[1], 1.11111);
        assert_eq!(data.chi_atomic[1], -0.045);
        Ok(())
    }

    #[test]
    fn roundtrips_axafs_text() -> crate::Result<()> {
        let data = parse_axafs_dat(AXAFS_DAT)?;
        let rendered = axafs_dat_string(&data)?;
        let reparsed = parse_axafs_dat(&rendered)?;

        assert_eq!(reparsed, data);
        Ok(())
    }

    #[test]
    fn builds_axafs_dat_from_core_rows() -> crate::Result<()> {
        let rows = arr2(&[
            [8979.0, 0.0, 0.0, 1.25, 1.1, 0.136_363_636],
            [8980.5, 1.5, 0.627, 1.4, 1.2, 0.166_666_667],
        ]);

        let data = axafs_dat_from_rows(rows.view())?;
        let axafs = XsphAxafs {
            rows,
            coefficients: [1.0, 2.0, 3.0],
            normalization: 4.0,
        };

        assert_eq!(axafs_dat_from_xsph_axafs(&axafs)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_axafs_inputs() {
        assert!(parse_axafs_dat("# header only\n").is_err());
        assert!(parse_axafs_dat("1 2 3 4 5\n").is_err());
        assert!(parse_axafs_dat("1 2 3 bad 5 6\n").is_err());
        assert!(parse_axafs_dat("1 2 3 4 NaN 6\n").is_err());
        assert!(axafs_dat_string(&empty_axafs()).is_err());
        assert!(axafs_dat_from_rows(arr2(&[[1.0, 2.0, 3.0]]).view()).is_err());
    }

    fn empty_axafs() -> AxafsDatData {
        AxafsDatData {
            header_lines: Vec::new(),
            energy_ev: Array1::from_vec(Vec::new()),
            edge_relative_energy_ev: Array1::from_vec(Vec::new()),
            wave_number_inverse_angstrom: Array1::from_vec(Vec::new()),
            atomic_absorption: Array1::from_vec(Vec::new()),
            atomic_background: Array1::from_vec(Vec::new()),
            chi_atomic: Array1::from_vec(Vec::new()),
        }
    }

    const AXAFS_DAT: &str = "\
 # File contains AXAFS. See manual for details.
 #--------------------------------------------------------------
 #  e, e(wrt edge), k, mu_at=(1+chi_at)*mu0_at, mu0_at, chi_at @#
    8979.000      0.000   0.000  1.23456E+00  1.00000E+00  2.34560E-01
    8980.500      1.500   0.627  1.06111D+00  1.11111D+00 -4.50000D-02
";
}
