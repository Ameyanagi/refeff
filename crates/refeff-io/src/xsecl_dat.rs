//! FEFF NRIXS `xsecl.dat` and `xsecl2.dat` text table support.
//!
//! The XSPH NRIXS path writes these readable companion tables before the
//! PAD-backed `xsecl.bin` file. Each row contains an energy, real components,
//! imaginary components, and a complex sum over the angular channels.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::{fortran_list_directed_f64, write_fortran_zero_scaled_exp};

/// Header values written as the first row of `xsecl.dat`/`xsecl2.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct XseclDatHeader {
    /// Number of real-axis energy points, `ne1`.
    pub real_energy_count: usize,
    /// Fermi-index/grid split value, `ik0`.
    pub fermi_index: usize,
    /// FEFF edge offset value from XSPH.
    pub edge: f64,
    /// FEFF `emu` scalar from XSPH.
    pub emu: f64,
    /// Core-hole width scalar, `gamach`.
    pub core_hole_width: f64,
}

/// Parsed contents of FEFF `xsecl.dat` or `xsecl2.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct XseclDatData {
    /// Scalar header row.
    pub header: XseclDatHeader,
    /// Energy grid written as `real(em) - edge + emu`.
    pub energy: Array1<f64>,
    /// Per-angular-channel complex cross sections, shaped `(energy, channel)`.
    pub channel_cross_sections: Array2<Complex64>,
    /// Complex row sum written after the channel columns.
    pub channel_sum: Array1<Complex64>,
}

impl XseclDatData {
    /// Number of energy rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.energy.len()
    }

    /// Number of angular channels per energy row.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channel_cross_sections.len_of(Axis(1))
    }
}

/// Parse FEFF `xsecl.dat` text.
pub fn parse_xsecl_dat(text: &str) -> Result<XseclDatData> {
    parse_xsecl_table("xsecl.dat", text)
}

/// Render FEFF-compatible `xsecl.dat` text.
pub fn xsecl_dat_string(data: &XseclDatData) -> Result<String> {
    xsecl_table_string("xsecl.dat", data)
}

/// Read FEFF `xsecl.dat` text from a file.
pub fn read_xsecl_dat(path: impl AsRef<Path>) -> Result<XseclDatData> {
    read_xsecl_table(path, parse_xsecl_dat)
}

/// Write FEFF `xsecl.dat` text to a file.
pub fn write_xsecl_dat(path: impl AsRef<Path>, data: &XseclDatData) -> Result<()> {
    write_text(path, xsecl_dat_string(data)?)
}

/// Parse FEFF `xsecl2.dat` text.
pub fn parse_xsecl2_dat(text: &str) -> Result<XseclDatData> {
    parse_xsecl_table("xsecl2.dat", text)
}

/// Render FEFF-compatible `xsecl2.dat` text.
pub fn xsecl2_dat_string(data: &XseclDatData) -> Result<String> {
    xsecl_table_string("xsecl2.dat", data)
}

/// Read FEFF `xsecl2.dat` text from a file.
pub fn read_xsecl2_dat(path: impl AsRef<Path>) -> Result<XseclDatData> {
    read_xsecl_table(path, parse_xsecl2_dat)
}

/// Write FEFF `xsecl2.dat` text to a file.
pub fn write_xsecl2_dat(path: impl AsRef<Path>, data: &XseclDatData) -> Result<()> {
    write_text(path, xsecl2_dat_string(data)?)
}

fn parse_xsecl_table(path: &'static str, text: &str) -> Result<XseclDatData> {
    let mut lines = text.lines().enumerate();
    let (header_line_number, header_line) = next_nonempty_line(path, &mut lines, "header")?;
    let header = parse_header(path, header_line_number, header_line)?;

    let mut energy = Vec::new();
    let mut channel_cross_sections = Vec::new();
    let mut channel_sum = Vec::new();
    let mut channel_count = None;
    for (index, raw) in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let row = parse_data_row(path, line_number, line)?;
        match channel_count {
            Some(expected) if row.channels.len() != expected => {
                return parse_error(
                    path,
                    line_number,
                    format!(
                        "row has {} channel(s), expected {expected}",
                        row.channels.len()
                    ),
                );
            }
            Some(_) => {}
            None => channel_count = Some(row.channels.len()),
        }
        energy.push(row.energy);
        channel_cross_sections.extend(row.channels);
        channel_sum.push(row.sum);
    }

    let row_count = energy.len();
    let channel_count = channel_count
        .ok_or_else(|| parse_error_value(path, 0, "at least one data row is required"))?;
    let channel_cross_sections =
        Array2::from_shape_vec((row_count, channel_count), channel_cross_sections).map_err(
            |source| parse_error_value(path, 0, format!("invalid channel shape: {source}")),
        )?;
    let data = XseclDatData {
        header,
        energy: Array1::from_vec(energy),
        channel_cross_sections,
        channel_sum: Array1::from_vec(channel_sum),
    };
    validate_xsecl_dat(path, &data)?;
    Ok(data)
}

fn xsecl_table_string(path: &'static str, data: &XseclDatData) -> Result<String> {
    validate_xsecl_dat(path, data)?;
    let mut out = String::new();
    write!(
        out,
        "{:12}{:12}",
        data.header.real_energy_count, data.header.fermi_index
    )?;
    out.push_str(&fortran_list_directed_f64(data.header.edge));
    out.push_str(&fortran_list_directed_f64(data.header.emu));
    out.push_str(&fortran_list_directed_f64(data.header.core_hole_width));
    out.push('\n');
    for row in 0..data.row_count() {
        write_fortran_zero_scaled_exp(&mut out, data.energy[row], 18, 8)?;
        for value in data.channel_cross_sections.row(row) {
            write_fortran_zero_scaled_exp(&mut out, value.re, 18, 8)?;
        }
        for value in data.channel_cross_sections.row(row) {
            write_fortran_zero_scaled_exp(&mut out, value.im, 18, 8)?;
        }
        let sum = data.channel_sum[row];
        write_fortran_zero_scaled_exp(&mut out, sum.re, 18, 8)?;
        write_fortran_zero_scaled_exp(&mut out, sum.im, 18, 8)?;
        out.push('\n');
    }
    Ok(out)
}

fn read_xsecl_table(
    path: impl AsRef<Path>,
    parse: impl FnOnce(&str) -> Result<XseclDatData>,
) -> Result<XseclDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse(&text)
}

fn write_text(path: impl AsRef<Path>, text: String) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, text).map_err(|source| IoError::io(path, source))
}

fn parse_header(path: &'static str, line_number: usize, line: &str) -> Result<XseclDatHeader> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 5 {
        return parse_error(
            path,
            line_number,
            format!("header has {} token(s), expected 5", tokens.len()),
        );
    }
    Ok(XseclDatHeader {
        real_energy_count: parse_usize(path, line_number, "real_energy_count", tokens[0])?,
        fermi_index: parse_usize(path, line_number, "fermi_index", tokens[1])?,
        edge: parse_f64(path, line_number, "edge", tokens[2])?,
        emu: parse_f64(path, line_number, "emu", tokens[3])?,
        core_hole_width: parse_f64(path, line_number, "core_hole_width", tokens[4])?,
    })
}

struct XseclRow {
    energy: f64,
    channels: Vec<Complex64>,
    sum: Complex64,
}

fn parse_data_row(path: &'static str, line_number: usize, line: &str) -> Result<XseclRow> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 5 {
        return parse_error(
            path,
            line_number,
            format!("row has {} token(s), expected at least 5", tokens.len()),
        );
    }
    let channel_tokens = tokens.len() - 3;
    if !channel_tokens.is_multiple_of(2) {
        return parse_error(
            path,
            line_number,
            format!("row has {channel_tokens} channel token(s), expected an even count"),
        );
    }
    let channel_count = channel_tokens / 2;
    let energy = parse_f64(path, line_number, "energy", tokens[0])?;
    let real_values = tokens[1..1 + channel_count]
        .iter()
        .map(|token| parse_f64(path, line_number, "channel_real", token))
        .collect::<Result<Vec<_>>>()?;
    let imag_values = tokens[1 + channel_count..1 + 2 * channel_count]
        .iter()
        .map(|token| parse_f64(path, line_number, "channel_imag", token))
        .collect::<Result<Vec<_>>>()?;
    let channels = real_values
        .into_iter()
        .zip(imag_values)
        .map(|(real, imag)| Complex64::new(real, imag))
        .collect();
    let sum = Complex64::new(
        parse_f64(path, line_number, "sum_real", tokens[tokens.len() - 2])?,
        parse_f64(path, line_number, "sum_imag", tokens[tokens.len() - 1])?,
    );

    Ok(XseclRow {
        energy,
        channels,
        sum,
    })
}

fn validate_xsecl_dat(path: &'static str, data: &XseclDatData) -> Result<()> {
    if data.header.real_energy_count == 0 {
        return parse_error(path, 1, "real energy count must be positive");
    }
    if data.header.fermi_index == 0 {
        return parse_error(path, 1, "fermi index must be positive");
    }
    validate_finite(path, "edge", data.header.edge, 1)?;
    validate_finite(path, "emu", data.header.emu, 1)?;
    validate_finite(path, "core_hole_width", data.header.core_hole_width, 1)?;
    if data.row_count() == 0 {
        return parse_error(path, 0, "at least one data row is required");
    }
    if data.channel_count() == 0 {
        return parse_error(path, 0, "at least one channel is required");
    }
    validate_len(
        path,
        "channel_sum",
        data.channel_sum.len(),
        data.row_count(),
    )?;
    if data.channel_cross_sections.len_of(Axis(0)) != data.row_count() {
        return parse_error(
            path,
            0,
            format!(
                "channel row count {} does not match energy count {}",
                data.channel_cross_sections.len_of(Axis(0)),
                data.row_count()
            ),
        );
    }
    for (index, (energy, sum)) in data.energy.iter().zip(data.channel_sum.iter()).enumerate() {
        let row = index + 1;
        validate_finite(path, "energy", *energy, row)?;
        validate_complex(path, "channel_sum", *sum, row)?;
        for value in data.channel_cross_sections.row(index) {
            validate_complex(path, "channel_cross_sections", *value, row)?;
        }
    }
    Ok(())
}

fn validate_len(
    path: &'static str,
    field: &'static str,
    len: usize,
    expected: usize,
) -> Result<()> {
    if len == expected {
        Ok(())
    } else {
        parse_error(
            path,
            0,
            format!("{field} has {len} row(s), expected {expected}"),
        )
    }
}

fn validate_complex(
    path: &'static str,
    field: &'static str,
    value: Complex64,
    row: usize,
) -> Result<()> {
    validate_finite(path, field, value.re, row)?;
    validate_finite(path, field, value.im, row)
}

fn validate_finite(path: &'static str, field: &'static str, value: f64, row: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        parse_error(path, row, format!("{field} must be finite"))
    }
}

fn next_nonempty_line<'a>(
    path: &'static str,
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    field: &'static str,
) -> Result<(usize, &'a str)> {
    for (index, raw) in lines {
        let line = raw.trim();
        if !line.is_empty() {
            return Ok((index + 1, line));
        }
    }
    parse_error(path, 0, format!("missing {field}"))
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

fn parse_usize(path: &'static str, line: usize, field: &'static str, token: &str) -> Result<usize> {
    token.parse::<usize>().map_err(|_| {
        parse_error_value(
            path,
            line,
            format!("could not parse {field} from {token:?}"),
        )
    })
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
    use super::*;

    #[test]
    fn parses_xsecl_dat() -> Result<()> {
        let parsed = parse_xsecl_dat(XSECL_DAT)?;
        assert_eq!(parsed.header.real_energy_count, 2);
        assert_eq!(parsed.header.fermi_index, 1);
        assert_eq!(parsed.header.edge, -0.25);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.channel_count(), 2);
        assert_eq!(parsed.energy[0], 408.083_58);
        assert_eq!(
            parsed.channel_cross_sections[(0, 0)],
            Complex64::new(-0.000_094_722_801, 0.000_115_562_54)
        );
        assert_eq!(
            parsed.channel_cross_sections[(0, 1)],
            Complex64::new(0.000_058_529_371, -0.000_120_865_91)
        );
        assert_eq!(
            parsed.channel_sum[1],
            Complex64::new(-0.000_160_211_14, -0.000_038_440_289)
        );

        let rendered = xsecl_dat_string(&parsed)?;
        assert_eq!(parse_xsecl_dat(&rendered)?, parsed);
        assert_eq!(parse_xsecl2_dat(XSECL_DAT)?, parsed);
        Ok(())
    }

    #[test]
    fn accepts_fortran_d_exponents() -> Result<()> {
        let parsed = parse_xsecl2_dat("2 1 -2.5D-1 4.0D+2 8.0D-2\n1D+0 2D+0 3D+0 4D+0 5D+0\n")?;
        assert_eq!(parsed.row_count(), 1);
        assert_eq!(parsed.channel_count(), 1);
        assert_eq!(
            parsed.channel_cross_sections[(0, 0)],
            Complex64::new(2.0, 3.0)
        );
        assert_eq!(parsed.channel_sum[0], Complex64::new(4.0, 5.0));
        Ok(())
    }

    #[test]
    fn rejects_bad_xsecl_inputs() {
        assert!(parse_xsecl_dat("").is_err());
        assert!(parse_xsecl_dat("2 1 0 1\n").is_err());
        assert!(parse_xsecl_dat("0 1 0 1 2\n1 2 3 4 5\n").is_err());
        assert!(parse_xsecl_dat("2 0 0 1 2\n1 2 3 4 5\n").is_err());
        assert!(parse_xsecl_dat("2 1 0 1 2\n1 2 3 4\n").is_err());
        assert!(parse_xsecl_dat("2 1 0 1 2\n1 2 NaN 4 5\n").is_err());
        assert!(parse_xsecl_dat(&format!("{XSECL_DAT}3 1 2 3 4\n")).is_err());
    }

    const XSECL_DAT: &str = r#"2 1 -0.25 408.0 0.0839493865
    0.40808358E+03   -0.94722801E-04    0.58529371E-04    0.11556254E-03   -0.12086591E-03   -0.36126732E-04   -0.52785148E-05
    0.40811859E+03   -0.42446685E-04   -0.11776355E-03    0.10570503E-03   -0.14409145E-03   -0.16021114E-03   -0.38440289E-04
"#;
}
