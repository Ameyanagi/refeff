//! FEFF RIXS output table codecs.
//!
//! `rixsET.dat` stores a two-axis RIXS map followed by one or more channel
//! columns. `herfd.dat` and `herfd-sat.dat` store one-axis line spectra with
//! the same channel-column convention. FEFF separates each map block with a
//! blank line; the parser records those block lengths so rendering can preserve
//! the same table structure.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};

use crate::error::{IoError, Result};

const RIXS_MAP_MIN_ROW_WIDTH: usize = 3;
const RIXS_LINE_MIN_ROW_WIDTH: usize = 2;
const RIXS_MAP_ALLOWED_ROW_WIDTHS: &str = "at least 3";
const RIXS_LINE_ALLOWED_ROW_WIDTHS: &str = "at least 2";
const RIXS_CHANNEL_COLUMNS: &str = "at least 1";

/// Parsed FEFF two-axis RIXS map such as `rixsET.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsMapData {
    /// Header and comment lines before or around the numeric map.
    pub header_lines: Vec<String>,
    /// Number of contiguous numeric rows in each FEFF block.
    pub block_lengths: Vec<usize>,
    /// First energy axis in eV.
    pub first_energy_ev: Array1<f64>,
    /// Second energy axis in eV.
    pub second_energy_ev: Array1<f64>,
    /// RIXS channel columns for each map row.
    pub channels: Array2<f64>,
}

impl RixsMapData {
    /// Number of numeric map rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.first_energy_ev.len()
    }

    /// Number of channel columns after the two energy axes.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.ncols()
    }
}

/// Parsed FEFF one-axis RIXS line spectrum such as `herfd.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct RixsLineData {
    /// Header and comment lines before or around the numeric line spectrum.
    pub header_lines: Vec<String>,
    /// Energy axis in eV.
    pub energy_ev: Array1<f64>,
    /// Spectrum channel columns for each row.
    pub channels: Array2<f64>,
}

impl RixsLineData {
    /// Number of line-spectrum rows.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_ev.len()
    }

    /// Number of channel columns after the energy axis.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.ncols()
    }
}

/// Render FEFF-compatible two-axis RIXS map text.
pub fn rixs_map_string(data: &RixsMapData) -> Result<String> {
    validate_rixs_map(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }

    let block_lengths = normalized_block_lengths(data.block_lengths.as_slice(), data.point_count());
    let mut row_index = 0_usize;
    for block_len in block_lengths {
        for _ in 0..block_len {
            write!(
                out,
                "{first:30.15E} {second:30.15E}",
                first = data.first_energy_ev[row_index],
                second = data.second_energy_ev[row_index]
            )?;
            for value in data.channels.index_axis(Axis(0), row_index) {
                write!(out, " {value:30.15E}")?;
            }
            writeln!(out)?;
            row_index += 1;
        }
        writeln!(out)?;
    }
    Ok(out)
}

/// Parse FEFF two-axis RIXS map text.
pub fn parse_rixs_map(text: &str) -> Result<RixsMapData> {
    let mut header_lines = Vec::new();
    let mut block_lengths = Vec::new();
    let mut current_block_len = 0_usize;
    let mut row_width = None;
    let mut first_energy_ev = Vec::new();
    let mut second_energy_ev = Vec::new();
    let mut channels = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            if current_block_len > 0 {
                block_lengths.push(current_block_len);
                current_block_len = 0;
            }
            continue;
        }

        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if width < RIXS_MAP_MIN_ROW_WIDTH {
                return Err(IoError::RixsDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: RIXS_MAP_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::RixsDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: "same width as previous numeric rows",
                    });
                }
            } else {
                row_width = Some(width);
            }

            first_energy_ev.push(parse_f64(line_number, "first energy", tokens[0])?);
            second_energy_ev.push(parse_f64(line_number, "second energy", tokens[1])?);
            for token in &tokens[2..] {
                channels.push(parse_f64(line_number, "channel", token)?);
            }
            current_block_len += 1;
        } else {
            header_lines.push(line.to_string());
        }
    }

    if current_block_len > 0 {
        block_lengths.push(current_block_len);
    }

    let point_count = first_energy_ev.len();
    let channel_count = row_width.map_or(0, |width| width - 2);
    let channels = if channel_count == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, channel_count), channels)
            .map_err(|_| invalid_rixs_dat("channels", "channel payload did not match map shape"))?
    };

    let data = RixsMapData {
        header_lines,
        block_lengths,
        first_energy_ev: Array1::from_vec(first_energy_ev),
        second_energy_ev: Array1::from_vec(second_energy_ev),
        channels,
    };
    validate_rixs_map(&data)?;
    Ok(data)
}

/// Write FEFF two-axis RIXS map text to a file.
pub fn write_rixs_map(path: impl AsRef<Path>, data: &RixsMapData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rixs_map_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF two-axis RIXS map text from a file.
pub fn read_rixs_map(path: impl AsRef<Path>) -> Result<RixsMapData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rixs_map(&text)
}

/// Render FEFF-compatible one-axis RIXS line spectrum text.
pub fn rixs_line_string(data: &RixsLineData) -> Result<String> {
    validate_rixs_line(data)?;

    let mut out = String::new();
    for line in &data.header_lines {
        writeln!(out, "{line}")?;
    }
    for (energy, row) in data.energy_ev.iter().zip(data.channels.axis_iter(Axis(0))) {
        write!(out, "{energy:30.15E}")?;
        for value in row {
            write!(out, " {value:30.15E}")?;
        }
        writeln!(out)?;
    }
    Ok(out)
}

/// Parse FEFF one-axis RIXS line spectrum text.
pub fn parse_rixs_line(text: &str) -> Result<RixsLineData> {
    let mut header_lines = Vec::new();
    let mut row_width = None;
    let mut energy_ev = Vec::new();
    let mut channels = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim_end();
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }

        if tokens.first().is_some_and(|token| is_numeric_token(token)) {
            let width = tokens.len();
            if width < RIXS_LINE_MIN_ROW_WIDTH {
                return Err(IoError::RixsDatRowWidth {
                    line: line_number,
                    actual: width,
                    expected: RIXS_LINE_ALLOWED_ROW_WIDTHS,
                });
            }
            if let Some(expected) = row_width {
                if width != expected {
                    return Err(IoError::RixsDatRowWidth {
                        line: line_number,
                        actual: width,
                        expected: "same width as previous numeric rows",
                    });
                }
            } else {
                row_width = Some(width);
            }

            energy_ev.push(parse_f64(line_number, "energy", tokens[0])?);
            for token in &tokens[1..] {
                channels.push(parse_f64(line_number, "channel", token)?);
            }
        } else {
            header_lines.push(line.to_string());
        }
    }

    let point_count = energy_ev.len();
    let channel_count = row_width.map_or(0, |width| width - 1);
    let channels = if channel_count == 0 {
        Array2::zeros((0, 0))
    } else {
        Array2::from_shape_vec((point_count, channel_count), channels)
            .map_err(|_| invalid_rixs_dat("channels", "channel payload did not match line shape"))?
    };

    let data = RixsLineData {
        header_lines,
        energy_ev: Array1::from_vec(energy_ev),
        channels,
    };
    validate_rixs_line(&data)?;
    Ok(data)
}

/// Write FEFF one-axis RIXS line spectrum text to a file.
pub fn write_rixs_line(path: impl AsRef<Path>, data: &RixsLineData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rixs_line_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF one-axis RIXS line spectrum text from a file.
pub fn read_rixs_line(path: impl AsRef<Path>) -> Result<RixsLineData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_rixs_line(&text)
}

fn validate_rixs_map(data: &RixsMapData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_rixs_dat(
            "rows",
            "at least one RIXS map row is required",
        ));
    }
    validate_len("second_energy_ev", data.second_energy_ev.len(), point_count)?;
    validate_channels("channels", data.channels.dim(), point_count)?;
    validate_block_lengths(data.block_lengths.as_slice(), point_count)?;

    for (row, (first, second)) in data
        .first_energy_ev
        .iter()
        .zip(data.second_energy_ev.iter())
        .enumerate()
    {
        let row = row + 1;
        validate_finite_row("first energy", *first, row)?;
        validate_finite_row("second energy", *second, row)?;
    }
    validate_channel_values(&data.channels)
}

fn validate_rixs_line(data: &RixsLineData) -> Result<()> {
    let point_count = data.point_count();
    if point_count == 0 {
        return Err(invalid_rixs_dat(
            "rows",
            "at least one RIXS line row is required",
        ));
    }
    validate_channels("channels", data.channels.dim(), point_count)?;

    for (row, energy) in data.energy_ev.iter().enumerate() {
        validate_finite_row("energy", *energy, row + 1)?;
    }
    validate_channel_values(&data.channels)
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::RixsDatShape {
            field,
            rows: actual,
            cols: 1,
            expected_rows: expected,
            expected_cols: "1",
        })
    }
}

fn validate_channels(
    field: &'static str,
    (rows, cols): (usize, usize),
    expected_rows: usize,
) -> Result<()> {
    if rows == expected_rows && cols > 0 {
        Ok(())
    } else {
        Err(IoError::RixsDatShape {
            field,
            rows,
            cols,
            expected_rows,
            expected_cols: RIXS_CHANNEL_COLUMNS,
        })
    }
}

fn validate_block_lengths(block_lengths: &[usize], point_count: usize) -> Result<()> {
    if block_lengths.is_empty() {
        return Ok(());
    }
    let total = block_lengths.iter().try_fold(0_usize, |total, block| {
        if *block == 0 {
            Err(invalid_rixs_dat(
                "block lengths",
                "block length must be positive",
            ))
        } else {
            total
                .checked_add(*block)
                .ok_or_else(|| invalid_rixs_dat("block lengths", "block length sum overflows"))
        }
    })?;
    if total == point_count {
        Ok(())
    } else {
        Err(invalid_rixs_dat(
            "block lengths",
            format!("block length sum {total} does not match row count {point_count}"),
        ))
    }
}

fn validate_channel_values(channels: &Array2<f64>) -> Result<()> {
    let cols = channels.ncols();
    for (index, value) in channels.iter().enumerate() {
        let row = index / cols + 1;
        validate_finite_row("channel", *value, row)?;
    }
    Ok(())
}

fn normalized_block_lengths(block_lengths: &[usize], point_count: usize) -> Vec<usize> {
    if block_lengths.is_empty() {
        vec![point_count]
    } else {
        block_lengths.to_vec()
    }
}

fn parse_f64(line: usize, field: &'static str, token: &str) -> Result<f64> {
    token
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|_| IoError::RixsDatParse {
            field,
            line,
            token: token.to_string(),
        })
}

fn validate_finite_row(field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(IoError::InvalidRixsDat {
            field,
            message: format!("row {row} value must be finite"),
        })
    }
}

fn invalid_rixs_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidRixsDat {
        field,
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
    fn parses_rixs_map_blocks_and_channels() -> Result<()> {
        let data = parse_rixs_map(RIXS_MAP)?;
        assert_eq!(data.point_count(), 4);
        assert_eq!(data.channel_count(), 4);
        assert_eq!(data.block_lengths, vec![2, 2]);
        assert_eq!(data.first_energy_ev[0], 1.15408611465090e4);
        assert_eq!(data.second_energy_ev[0], -15.0000004124133);
        assert_eq!(data.channels[[0, 0]], 1.05265355183925e-6);
        assert_eq!(data.channels[[3, 3]], 3.0e-3);
        Ok(())
    }

    #[test]
    fn parses_rixs_line_reference_shape() -> Result<()> {
        let data = parse_rixs_line(RIXS_LINE)?;
        assert_eq!(data.point_count(), 3);
        assert_eq!(data.channel_count(), 4);
        assert_eq!(data.energy_ev[0], 1.15408611465090e4);
        assert_eq!(data.channels[[1, 0]], 1.14359805938722e-6);
        assert_eq!(data.channels[[2, 3]], 0.0);
        Ok(())
    }

    #[test]
    fn roundtrips_rixs_map_and_line_text() -> Result<()> {
        let map = parse_rixs_map(RIXS_MAP)?;
        assert_eq!(parse_rixs_map(&rixs_map_string(&map)?)?, map);

        let line = parse_rixs_line(RIXS_LINE)?;
        assert_eq!(parse_rixs_line(&rixs_line_string(&line)?)?, line);
        Ok(())
    }

    #[test]
    fn rejects_bad_rixs_inputs() {
        assert!(parse_rixs_map("# no rows\n").is_err());
        assert!(parse_rixs_map("1 2\n").is_err());
        assert!(parse_rixs_map("1 2 3\n4 5 6 7\n").is_err());
        assert!(parse_rixs_map("1 2 NaN\n").is_err());
        assert!(parse_rixs_line("1\n").is_err());
        assert!(parse_rixs_line("1 2\n3 4 5\n").is_err());
        assert!(parse_rixs_line("1 NaN\n").is_err());

        let bad_map = RixsMapData {
            header_lines: Vec::new(),
            block_lengths: vec![3],
            first_energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            second_energy_ev: Array1::from_vec(vec![1.0, 2.0]),
            channels: Array2::zeros((2, 1)),
        };
        assert!(rixs_map_string(&bad_map).is_err());
    }

    const RIXS_MAP: &str = r#"         0.115408611465090E+05        -0.150000004124133E+02         0.105265355183925E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05        -0.150000004124133E+02         0.100255925206130E-05         0.100000000000000E-02         0.200000000000000E-02         0.300000000000000E-02

         0.115408611465090E+05        -0.140000004124133E+02         0.200000000000000E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05        -0.140000004124133E+02         0.300000000000000E-05         0.100000000000000E-02         0.200000000000000E-02         0.300000000000000E-02
"#;

    const RIXS_LINE: &str = r#"         0.115408611465090E+05         0.105265355183925E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115418611463549E+05         0.114359805938722E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
         0.115428611462008E+05         0.125101882959870E-05         0.000000000000000E+00         0.000000000000000E+00         0.000000000000000E+00
"#;
}
