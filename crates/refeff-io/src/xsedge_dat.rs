//! FEFF `TDLDA/xsectd.f90` `xsedge.dat` table codec.
//!
//! `xsedge.dat` is the TDLDA/PMBSE side output written by `xsectd`: each row
//! starts with the output energy in eV, followed by single-particle and
//! screened totals. Spin-orbit split runs add plus/minus branch totals.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use refeff_core::xsph::XsphTdldaXsedgeRows;

use crate::error::{IoError, Result};
use crate::format::write_fortran_zero_scaled_exp;

const XSEDGE_MIN_ROW_WIDTH: usize = 3;
const XSEDGE_SPLIT_ROW_WIDTH: usize = 7;
const XSEDGE_ALLOWED_ROW_WIDTHS: &str = "3 or 7";

/// Parsed FEFF `xsedge.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct XsedgeDatData {
    /// Output energy column in eV, FEFF `(emr(ie) + emu) * hart`.
    pub energy_ev: Array1<f64>,
    /// Sum of active single-particle channel spectra.
    pub total_single_particle: Array1<f64>,
    /// Sum of active TDLDA-screened channel spectra.
    pub total_screened: Array1<f64>,
    /// Optional plus-branch single-particle total, FEFF `l3 + l5`.
    pub plus_branch_single_particle: Option<Array1<f64>>,
    /// Optional minus-branch single-particle total, FEFF `l2 + l4`.
    pub minus_branch_single_particle: Option<Array1<f64>>,
    /// Optional plus-branch screened total, FEFF screened `l3 + l5`.
    pub plus_branch_screened: Option<Array1<f64>>,
    /// Optional minus-branch screened total, FEFF screened `l2 + l4`.
    pub minus_branch_screened: Option<Array1<f64>>,
}

/// Adapter inputs for building `xsedge.dat` from the core TDLDA row helper.
#[derive(Debug, Clone, Copy)]
pub struct XsedgeDatFromTdldaRowsInput<'a> {
    /// Completed core TDLDA `xsedge.dat` rows.
    pub rows: &'a XsphTdldaXsedgeRows,
    /// FEFF `nch`; `1` writes three columns, `2` and `4` write split columns.
    pub channel_count: usize,
}

impl XsedgeDatData {
    /// Number of data rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Whether the FEFF spin-orbit branch columns are present.
    #[must_use]
    pub fn has_branch_columns(&self) -> bool {
        self.plus_branch_single_particle.is_some()
            && self.minus_branch_single_particle.is_some()
            && self.plus_branch_screened.is_some()
            && self.minus_branch_screened.is_some()
    }
}

/// Build FEFF `xsedge.dat` contents from core TDLDA row data.
pub fn xsedge_dat_from_tdlda_rows(input: XsedgeDatFromTdldaRowsInput<'_>) -> Result<XsedgeDatData> {
    let include_branches = match input.channel_count {
        1 => false,
        2 | 4 => true,
        value => {
            return Err(invalid_xsedge_dat(
                "channel_count",
                format!("expected 1, 2, or 4, got {value}"),
            ));
        }
    };
    let data = XsedgeDatData {
        energy_ev: input.rows.energy_ev.clone(),
        total_single_particle: input.rows.total_single_particle.clone(),
        total_screened: input.rows.total_screened.clone(),
        plus_branch_single_particle: include_branches
            .then(|| input.rows.plus_branch_single_particle.clone()),
        minus_branch_single_particle: include_branches
            .then(|| input.rows.minus_branch_single_particle.clone()),
        plus_branch_screened: include_branches.then(|| input.rows.plus_branch_screened.clone()),
        minus_branch_screened: include_branches.then(|| input.rows.minus_branch_screened.clone()),
    };
    validate_xsedge_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible `xsedge.dat` text.
pub fn xsedge_dat_string(data: &XsedgeDatData) -> Result<String> {
    validate_xsedge_dat(data)?;

    let mut out = String::new();
    if data.has_branch_columns() {
        let plus_single = data.plus_branch_single_particle.as_ref().ok_or_else(|| {
            invalid_xsedge_dat("plus_branch_single_particle", "missing branch column")
        })?;
        let minus_single = data.minus_branch_single_particle.as_ref().ok_or_else(|| {
            invalid_xsedge_dat("minus_branch_single_particle", "missing branch column")
        })?;
        let plus_screened = data
            .plus_branch_screened
            .as_ref()
            .ok_or_else(|| invalid_xsedge_dat("plus_branch_screened", "missing branch column"))?;
        let minus_screened = data
            .minus_branch_screened
            .as_ref()
            .ok_or_else(|| invalid_xsedge_dat("minus_branch_screened", "missing branch column"))?;

        for row in 0..data.row_count() {
            write_xsedge_row(
                &mut out,
                [
                    data.energy_ev[row],
                    data.total_single_particle[row],
                    data.total_screened[row],
                    plus_single[row],
                    minus_single[row],
                    plus_screened[row],
                    minus_screened[row],
                ],
            )?;
        }
    } else {
        for row in 0..data.row_count() {
            write_xsedge_row(
                &mut out,
                [
                    data.energy_ev[row],
                    data.total_single_particle[row],
                    data.total_screened[row],
                ],
            )?;
        }
    }
    Ok(out)
}

/// Parse FEFF `xsedge.dat` text.
pub fn parse_xsedge_dat(text: &str) -> Result<XsedgeDatData> {
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut total_single_particle = Vec::new();
    let mut total_screened = Vec::new();
    let mut plus_branch_single_particle = Vec::new();
    let mut minus_branch_single_particle = Vec::new();
    let mut plus_branch_screened = Vec::new();
    let mut minus_branch_screened = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let width = tokens.len();
        if !matches!(width, XSEDGE_MIN_ROW_WIDTH | XSEDGE_SPLIT_ROW_WIDTH) {
            return Err(IoError::XsedgeDatRowWidth {
                line: line_number,
                actual: width,
                expected: XSEDGE_ALLOWED_ROW_WIDTHS,
            });
        }
        if let Some(expected) = row_width {
            if width != expected {
                return Err(IoError::XsedgeDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: row_width_label(expected),
                });
            }
        } else {
            row_width = Some(width);
        }

        energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
        total_single_particle.push(parse_f64(line_number, "total single-particle", tokens[1])?);
        total_screened.push(parse_f64(line_number, "total screened", tokens[2])?);
        if width == XSEDGE_SPLIT_ROW_WIDTH {
            plus_branch_single_particle.push(parse_f64(
                line_number,
                "plus branch single-particle",
                tokens[3],
            )?);
            minus_branch_single_particle.push(parse_f64(
                line_number,
                "minus branch single-particle",
                tokens[4],
            )?);
            plus_branch_screened.push(parse_f64(line_number, "plus branch screened", tokens[5])?);
            minus_branch_screened.push(parse_f64(line_number, "minus branch screened", tokens[6])?);
        }
    }

    let has_branch_columns = row_width == Some(XSEDGE_SPLIT_ROW_WIDTH);
    let data = XsedgeDatData {
        energy_ev: Array1::from_vec(energy_ev),
        total_single_particle: Array1::from_vec(total_single_particle),
        total_screened: Array1::from_vec(total_screened),
        plus_branch_single_particle: has_branch_columns
            .then(|| Array1::from_vec(plus_branch_single_particle)),
        minus_branch_single_particle: has_branch_columns
            .then(|| Array1::from_vec(minus_branch_single_particle)),
        plus_branch_screened: has_branch_columns.then(|| Array1::from_vec(plus_branch_screened)),
        minus_branch_screened: has_branch_columns.then(|| Array1::from_vec(minus_branch_screened)),
    };
    validate_xsedge_dat(&data)?;
    Ok(data)
}

/// Write FEFF `xsedge.dat` text to a file.
pub fn write_xsedge_dat(path: impl AsRef<Path>, data: &XsedgeDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xsedge_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `xsedge.dat` text from a file.
pub fn read_xsedge_dat(path: impl AsRef<Path>) -> Result<XsedgeDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xsedge_dat(&text)
}

fn write_xsedge_row<const N: usize>(out: &mut String, fields: [f64; N]) -> Result<()> {
    write!(out, "{:10.5}  ", fields[0])?;
    write_fortran_zero_scaled_exp(out, fields[1], 10, 5)?;
    write!(out, "  ")?;
    write_fortran_zero_scaled_exp(out, fields[2], 10, 5)?;
    for field in fields.iter().skip(3) {
        write!(out, " ")?;
        write_fortran_zero_scaled_exp(out, *field, 10, 5)?;
    }
    out.push('\n');
    Ok(())
}

fn validate_xsedge_dat(data: &XsedgeDatData) -> Result<()> {
    let row_count = data.row_count();
    if row_count == 0 {
        return Err(invalid_xsedge_dat(
            "rows",
            "at least one xsedge row is required",
        ));
    }
    validate_len(
        "total_single_particle",
        data.total_single_particle.len(),
        row_count,
    )?;
    validate_len("total_screened", data.total_screened.len(), row_count)?;

    let branch_presence = [
        data.plus_branch_single_particle.is_some(),
        data.minus_branch_single_particle.is_some(),
        data.plus_branch_screened.is_some(),
        data.minus_branch_screened.is_some(),
    ];
    if branch_presence.iter().any(|present| *present)
        && !branch_presence.iter().all(|present| *present)
    {
        return Err(invalid_xsedge_dat(
            "branch_columns",
            "plus/minus single-particle and screened columns must be present together",
        ));
    }
    validate_optional_len(
        "plus_branch_single_particle",
        &data.plus_branch_single_particle,
        row_count,
    )?;
    validate_optional_len(
        "minus_branch_single_particle",
        &data.minus_branch_single_particle,
        row_count,
    )?;
    validate_optional_len(
        "plus_branch_screened",
        &data.plus_branch_screened,
        row_count,
    )?;
    validate_optional_len(
        "minus_branch_screened",
        &data.minus_branch_screened,
        row_count,
    )?;

    for (row, value) in data.energy_ev.iter().enumerate() {
        validate_finite("energy", *value, row + 1)?;
    }
    for (row, value) in data.total_single_particle.iter().enumerate() {
        validate_finite("total_single_particle", *value, row + 1)?;
    }
    for (row, value) in data.total_screened.iter().enumerate() {
        validate_finite("total_screened", *value, row + 1)?;
    }
    for (field, values) in [
        (
            "plus_branch_single_particle",
            &data.plus_branch_single_particle,
        ),
        (
            "minus_branch_single_particle",
            &data.minus_branch_single_particle,
        ),
        ("plus_branch_screened", &data.plus_branch_screened),
        ("minus_branch_screened", &data.minus_branch_screened),
    ] {
        if let Some(values) = values {
            for (row, value) in values.iter().enumerate() {
                validate_finite(field, *value, row + 1)?;
            }
        }
    }
    Ok(())
}

fn validate_optional_len(
    field: &'static str,
    values: &Option<Array1<f64>>,
    expected: usize,
) -> Result<()> {
    if let Some(values) = values {
        validate_len(field, values.len(), expected)?;
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::XsedgeDatShape {
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
        .map_err(|_| IoError::XsedgeDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_xsedge_dat(
            field,
            format!("row {row} value must be finite"),
        ))
    }
}

fn invalid_xsedge_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXsedgeDat {
        field,
        message: message.into(),
    }
}

fn row_width_label(width: usize) -> &'static str {
    match width {
        XSEDGE_MIN_ROW_WIDTH => "3",
        XSEDGE_SPLIT_ROW_WIDTH => "7",
        _ => XSEDGE_ALLOWED_ROW_WIDTHS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn parses_three_column_xsedge_data() -> Result<()> {
        let data = parse_xsedge_dat("  10.00000  0.12000E+03  0.13000E+03\n")?;

        assert_eq!(data.row_count(), 1);
        assert!(!data.has_branch_columns());
        assert_eq!(data.energy_ev[0], 10.0);
        assert_eq!(data.total_single_particle[0], 120.0);
        assert_eq!(data.total_screened[0], 130.0);
        assert_eq!(parse_xsedge_dat(&xsedge_dat_string(&data)?)?, data);
        Ok(())
    }

    #[test]
    fn parses_split_xsedge_data_and_roundtrips() -> Result<()> {
        let data = parse_xsedge_dat(
            "  12.50000  0.90000E+02  0.95000E+02 0.70000E+02 0.20000E+02 0.76000E+02 0.19000E+02\n\
               25.00000  0.40000E+02  0.96000E+02 0.14000E+02 0.26000E+02 0.38000E+02 0.58000E+02\n",
        )?;

        assert_eq!(data.row_count(), 2);
        assert!(data.has_branch_columns());
        assert_eq!(data.energy_ev, arr1(&[12.5, 25.0]));
        assert_eq!(data.total_single_particle, arr1(&[90.0, 40.0]));
        assert_eq!(
            data.minus_branch_screened.as_ref().map(|values| values[1]),
            Some(58.0)
        );

        let rendered = xsedge_dat_string(&data)?;
        assert_eq!(parse_xsedge_dat(&rendered)?, data);
        Ok(())
    }

    #[test]
    fn builds_xsedge_from_core_tdlda_rows() -> Result<()> {
        let rows = XsphTdldaXsedgeRows {
            energy_ev: arr1(&[1.0, 2.0]),
            total_single_particle: arr1(&[3.0, 4.0]),
            total_screened: arr1(&[5.0, 6.0]),
            plus_branch_single_particle: arr1(&[7.0, 8.0]),
            minus_branch_single_particle: arr1(&[9.0, 10.0]),
            plus_branch_screened: arr1(&[11.0, 12.0]),
            minus_branch_screened: arr1(&[13.0, 14.0]),
        };

        let unsplit = xsedge_dat_from_tdlda_rows(XsedgeDatFromTdldaRowsInput {
            rows: &rows,
            channel_count: 1,
        })?;
        assert!(!unsplit.has_branch_columns());
        assert_eq!(unsplit.total_screened, arr1(&[5.0, 6.0]));

        let split = xsedge_dat_from_tdlda_rows(XsedgeDatFromTdldaRowsInput {
            rows: &rows,
            channel_count: 4,
        })?;
        assert!(split.has_branch_columns());
        assert_eq!(
            split.plus_branch_screened.as_ref().map(|values| values[0]),
            Some(11.0)
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_xsedge_inputs() {
        assert!(parse_xsedge_dat("").is_err());
        assert!(parse_xsedge_dat("1 2\n").is_err());
        assert!(parse_xsedge_dat("1 2 3\n4 5 6 7 8 9 10\n").is_err());
        assert!(parse_xsedge_dat("1 NaN 3\n").is_err());

        let bad = XsedgeDatData {
            energy_ev: arr1(&[1.0, 2.0]),
            total_single_particle: arr1(&[3.0]),
            total_screened: arr1(&[4.0, 5.0]),
            plus_branch_single_particle: None,
            minus_branch_single_particle: None,
            plus_branch_screened: None,
            minus_branch_screened: None,
        };
        assert!(xsedge_dat_string(&bad).is_err());

        let partial_branch = XsedgeDatData {
            energy_ev: arr1(&[1.0]),
            total_single_particle: arr1(&[2.0]),
            total_screened: arr1(&[3.0]),
            plus_branch_single_particle: Some(arr1(&[4.0])),
            minus_branch_single_particle: None,
            plus_branch_screened: None,
            minus_branch_screened: None,
        };
        assert!(xsedge_dat_string(&partial_branch).is_err());
    }
}
