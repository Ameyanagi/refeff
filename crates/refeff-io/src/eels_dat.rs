//! FEFF `eels.dat` spectrum text codec.
//!
//! FEFF writes orientation-averaged EELS spectra with four numeric columns:
//! energy loss, total spectrum, atomic background, and fine structure. For
//! orientation-sensitive calculations it appends the nine Cartesian tensor
//! components in `xx, xy, xz, yx, yy, yz, zx, zy, zz` order.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};

use crate::error::{IoError, Result};

const EELS_DAT_AVERAGED_ROW_WIDTH: usize = 4;
const EELS_DAT_TENSOR_ROW_WIDTH: usize = 13;
const EELS_DAT_ALLOWED_ROW_WIDTHS: &str = "4 or 13";
const EELS_DAT_TENSOR_COLUMNS: usize = 9;

/// Cartesian tensor column labels in FEFF `eels.dat` order.
pub const EELS_TENSOR_LABELS: [&str; EELS_DAT_TENSOR_COLUMNS] =
    ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"];

/// Parsed FEFF `eels.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EelsDatData {
    /// Header and comment lines before and around the numeric spectrum table.
    pub header_lines: Vec<String>,
    /// Energy loss in eV.
    pub energy_loss_ev: Array1<f64>,
    /// Total EELS spectrum in `a_0^2 / eV`.
    pub total: Array1<f64>,
    /// Atomic background in `a_0^2 / eV`.
    pub atomic_background: Array1<f64>,
    /// Fine structure contribution, `total - atomic_background`.
    pub fine_structure: Array1<f64>,
    /// Optional 3x3 Cartesian tensor components, row-major in
    /// [`EELS_TENSOR_LABELS`] order.
    pub tensor: Option<Array2<f64>>,
}

impl EelsDatData {
    /// Number of spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_loss_ev.len()
    }

    /// Whether this spectrum carries orientation-sensitive tensor components.
    #[must_use]
    pub fn has_tensor(&self) -> bool {
        self.tensor.is_some()
    }
}

/// Render FEFF-compatible `eels.dat` text.
pub fn eels_dat_string(data: &EelsDatData) -> Result<String> {
    validate_eels_dat(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    if let Some(tensor) = &data.tensor {
        for ((((energy, total), background), fine_structure), components) in data
            .energy_loss_ev
            .iter()
            .zip(data.total.iter())
            .zip(data.atomic_background.iter())
            .zip(data.fine_structure.iter())
            .zip(tensor.axis_iter(Axis(0)))
        {
            write!(
                out,
                "{energy:14.6} {total:14.6E} {background:14.6E} {fine_structure:14.6E}"
            )?;
            for component in components {
                write!(out, " {component:14.6E}")?;
            }
            writeln!(out)?;
        }
    } else {
        for (((energy, total), background), fine_structure) in data
            .energy_loss_ev
            .iter()
            .zip(data.total.iter())
            .zip(data.atomic_background.iter())
            .zip(data.fine_structure.iter())
        {
            writeln!(
                out,
                "{energy:14.6} {total:14.6E} {background:14.6E} {fine_structure:14.6E}"
            )?;
        }
    }

    Ok(out)
}

/// Parse FEFF `eels.dat` text.
pub fn parse_eels_dat(text: &str) -> Result<EelsDatData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_loss_ev = Vec::new();
    let mut total = Vec::new();
    let mut atomic_background = Vec::new();
    let mut fine_structure = Vec::new();
    let mut tensor = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if !matches!(
                width,
                EELS_DAT_AVERAGED_ROW_WIDTH | EELS_DAT_TENSOR_ROW_WIDTH
            ) {
                return Err(IoError::EelsDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: EELS_DAT_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::EelsDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: row_width_label(expected),
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_loss_ev.push(parse_f64(line_number, "energy loss", tokens[0])?);
            total.push(parse_f64(line_number, "total", tokens[1])?);
            atomic_background.push(parse_f64(line_number, "atomic background", tokens[2])?);
            fine_structure.push(parse_f64(line_number, "fine structure", tokens[3])?);
            if width == EELS_DAT_TENSOR_ROW_WIDTH {
                for (offset, label) in EELS_TENSOR_LABELS.iter().enumerate() {
                    tensor.push(parse_f64(line_number, label, tokens[offset + 4])?);
                }
            }
        } else {
            header_lines.push(line.to_string());
        }
    }

    let point_count = energy_loss_ev.len();
    let tensor = if row_width == Some(EELS_DAT_TENSOR_ROW_WIDTH) {
        Some(
            Array2::from_shape_vec((point_count, EELS_DAT_TENSOR_COLUMNS), tensor).map_err(
                |_| invalid_eels_dat("tensor", "tensor payload did not match spectrum length"),
            )?,
        )
    } else {
        None
    };

    let data = EelsDatData {
        header_lines,
        energy_loss_ev: Array1::from_vec(energy_loss_ev),
        total: Array1::from_vec(total),
        atomic_background: Array1::from_vec(atomic_background),
        fine_structure: Array1::from_vec(fine_structure),
        tensor,
    };
    validate_eels_dat(&data)?;
    Ok(data)
}

/// Write FEFF `eels.dat` text to a file.
pub fn write_eels_dat(path: impl AsRef<Path>, data: &EelsDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, eels_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `eels.dat` text from a file.
pub fn read_eels_dat(path: impl AsRef<Path>) -> Result<EelsDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_eels_dat(&text)
}

fn validate_eels_dat(data: &EelsDatData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_eels_dat(
            "rows",
            "at least one spectrum row is required",
        ));
    }
    validate_len("total", data.total.len(), point_count)?;
    validate_len(
        "atomic_background",
        data.atomic_background.len(),
        point_count,
    )?;
    validate_len("fine_structure", data.fine_structure.len(), point_count)?;

    if let Some(tensor) = &data.tensor {
        let shape = tensor.shape();
        if shape != [point_count, EELS_DAT_TENSOR_COLUMNS] {
            return Err(IoError::EelsDatTensorShape {
                rows: shape[0],
                cols: shape[1],
                expected_rows: point_count,
                expected_cols: EELS_DAT_TENSOR_COLUMNS,
            });
        }
    }

    for (row, (((energy, total), background), fine_structure)) in data
        .energy_loss_ev
        .iter()
        .zip(data.total.iter())
        .zip(data.atomic_background.iter())
        .zip(data.fine_structure.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("energy loss", *energy, row)?;
        validate_finite_row("total", *total, row)?;
        validate_finite_row("atomic background", *background, row)?;
        validate_finite_row("fine structure", *fine_structure, row)?;
    }
    if let Some(tensor) = &data.tensor {
        for (index, value) in tensor.iter().enumerate() {
            let row = index / EELS_DAT_TENSOR_COLUMNS + 1;
            let column = index % EELS_DAT_TENSOR_COLUMNS;
            validate_finite_row(EELS_TENSOR_LABELS[column], *value, row)?;
        }
    }

    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::EelsDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::EelsDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidEelsDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_eels_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidEelsDat {
        field,
        message: message.into(),
    }
}

fn is_numeric_token(token: &str) -> bool {
    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        EELS_DAT_AVERAGED_ROW_WIDTH => "4",
        EELS_DAT_TENSOR_ROW_WIDTH => "13",
        _ => EELS_DAT_ALLOWED_ROW_WIDTHS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_orientation_sensitive_eels_reference_shape() -> Result<()> {
        let data = parse_eels_dat(EELS_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(data.has_tensor());
        assert_eq!(data.energy_loss_ev[0], 8979.41);
        assert_eq!(data.total[1], 1.46291e-13);
        assert_eq!(data.atomic_background[0], 1.38421e-13);
        assert_eq!(data.fine_structure[1], -2.00164e-14);
        let tensor = data
            .tensor
            .as_ref()
            .ok_or_else(|| invalid_eels_dat("tensor", "missing tensor"))?;
        assert_eq!(tensor[[0, 0]], 2.53207e-15);
        assert_eq!(tensor[[1, 4]], 1.40271e-13);
        assert_eq!(EELS_TENSOR_LABELS[8], "zz");
        Ok(())
    }

    #[test]
    fn parses_orientation_averaged_eels_shape() -> Result<()> {
        let data = parse_eels_dat(AVERAGED_EELS_DAT)?;
        assert_eq!(data.point_count(), 2);
        assert!(!data.has_tensor());
        assert_eq!(data.fine_structure[0], -3.0e-3);
        Ok(())
    }

    #[test]
    fn roundtrips_eels_text() -> Result<()> {
        let data = parse_eels_dat(EELS_DAT)?;
        let rendered = eels_dat_string(&data)?;
        assert_eq!(parse_eels_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn rejects_bad_eels_inputs() {
        assert!(parse_eels_dat("# no data\n").is_err());
        assert!(parse_eels_dat("1 2 3\n").is_err());
        assert!(parse_eels_dat("1 2 3 NaN\n").is_err());
        assert!(parse_eels_dat("1 2 3 4\n2 3 4 5 6 7 8 9 10 11 12 13 14\n").is_err());

        let invalid_shape = EelsDatData {
            header_lines: Vec::new(),
            energy_loss_ev: Array1::from_vec(vec![1.0, 2.0]),
            total: Array1::from_vec(vec![1.0, 2.0]),
            atomic_background: Array1::from_vec(vec![0.5, 1.5]),
            fine_structure: Array1::from_vec(vec![0.5, 0.5]),
            tensor: Some(Array2::zeros((1, EELS_DAT_TENSOR_COLUMNS))),
        };
        assert!(eels_dat_string(&invalid_shape).is_err());
    }

    const EELS_DAT: &str = r#"# Orientation sensitive EELS calculation - beam energy =   300.keV
#  Energy       total         atomic-bg     fine-struct   xx            xy            xz            yx            yy            yz            zx            zy            zz
   8979.41      0.123021E-12  0.138421E-12 -0.154000E-13  0.253207E-14  0.166012E-37  0.497278E-38 -0.102010E-36  0.117957E-12  0.135182E-36  0.388742E-38  0.301220E-37  0.253207E-14
   8980.98      0.146291E-12  0.166308E-12 -0.200164E-13  0.301002E-14  0.374584E-37  0.166287E-38 -0.156369E-36  0.140271E-12  0.176324E-36  0.502442E-38  0.587747E-37  0.301002E-14
"#;

    const AVERAGED_EELS_DAT: &str = r#"# Orientation averaged EELS calculation - beam energy =   300.keV
#  Energy       total         atomic-bg     fine-struct
  100.0  1.0E-2  1.3E-2 -3.0E-3
  101.0  1.1E-2  1.2E-2 -1.0E-3
"#;
}
