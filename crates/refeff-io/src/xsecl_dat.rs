//! FEFF NRIXS `xsecl.dat` and `xsecl2.dat` text table support.
//!
//! The XSPH NRIXS path writes these readable companion tables before the
//! PAD-backed `xsecl.bin` file. Each row contains an energy, real components,
//! imaginary components, and a complex sum over the angular channels.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};
use crate::format::{fortran_list_directed_f64, write_fortran_zero_scaled_exp};
use crate::xsecl_bin::{XseclBinData, XseclBinTransition};

/// Header values written as the first row of `xsecl.dat`/`xsecl2.dat`.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Completed NRIXS XSPH row data used to build the `xsecl*` handoff files.
#[derive(Debug, Clone)]
pub struct XseclFromXsphNrixsInput<'a> {
    /// Scalar row shared by `xsecl.dat` and `xsecl2.dat`.
    pub header: XseclDatHeader,
    /// Energy column written to the text tables.
    ///
    /// FEFF writes this as the shifted NRIXS spectrum energy; callers that
    /// start from the internal complex phase mesh should supply
    /// `real(em) - edge + emu`.
    pub energy: ArrayView1<'a, f64>,
    /// Angular-decomposition channel cross sections for `xsecl.dat`.
    pub decomposition_cross_sections: ArrayView2<'a, Complex64>,
    /// Total-angular-momentum channel cross sections for `xsecl2.dat`.
    pub total_angular_cross_sections: ArrayView2<'a, Complex64>,
    /// Atomic final-state cross sections written to `xsecl.bin`.
    pub atom_cross_sections: ArrayView2<'a, Complex64>,
    /// Transition-index metadata written before the `xsecl.bin` PAD blocks.
    pub transitions: &'a [XseclBinTransition],
    /// FEFF doubled initial angular momentum, `jinit`.
    pub initial_state_j: i32,
    /// FEFF PAD field width used by neighboring binary handoff files.
    pub pad_width: usize,
}

/// Renderable FEFF XSPH NRIXS handoff payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct XseclFromXsphNrixs {
    /// `xsecl.dat` angular-decomposition table.
    pub xsecl: XseclDatData,
    /// `xsecl2.dat` total-angular-momentum table.
    pub xsecl2: XseclDatData,
    /// `xsecl.bin` PAD-backed atomic final-state table.
    pub xsecl_bin: XseclBinData,
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

/// Build FEFF `xsecl.dat`, `xsecl2.dat`, and `xsecl.bin` from completed XSPH
/// NRIXS work arrays.
///
/// This is the typed output boundary for the `XSPH/xsectjas.f90` final handoff:
/// the caller supplies solved per-energy channel/final-state cross sections,
/// while this adapter computes the text-table row sums and attaches the
/// transition-index metadata required by the PAD-backed atomic table.
pub fn xsecl_from_xsph_nrixs(input: XseclFromXsphNrixsInput<'_>) -> Result<XseclFromXsphNrixs> {
    validate_xsecl_from_xsph_nrixs_input(&input)?;
    let xsecl = xsecl_table_from_rows(
        input.header,
        input.energy,
        input.decomposition_cross_sections,
    );
    let xsecl2 = xsecl_table_from_rows(
        input.header,
        input.energy,
        input.total_angular_cross_sections,
    );
    let xsecl_bin = XseclBinData {
        pad_width: input.pad_width,
        initial_state_j: input.initial_state_j,
        transitions: input.transitions.to_vec(),
        atom_cross_sections: input.atom_cross_sections.to_owned(),
        raw_atom_cross_section_pad: None,
    };

    validate_xsecl_dat("xsecl.dat", &xsecl)?;
    validate_xsecl_dat("xsecl2.dat", &xsecl2)?;
    crate::xsecl_bin::xsecl_bin_string(&xsecl_bin)?;

    Ok(XseclFromXsphNrixs {
        xsecl,
        xsecl2,
        xsecl_bin,
    })
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
    let header = parse_header(path, header_line_number, header_line, &mut lines)?;

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
        if channel_count.is_some() && is_legacy_scalar_noise_row(line) {
            continue;
        }
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

fn parse_header<'a>(
    path: &'static str,
    line_number: usize,
    line: &str,
    lines: &mut std::iter::Enumerate<std::str::Lines<'a>>,
) -> Result<XseclDatHeader> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 4 && tokens.len() != 5 {
        return parse_error(
            path,
            line_number,
            format!("header has {} token(s), expected 4 or 5", tokens.len()),
        );
    }
    let core_hole_width = if tokens.len() == 5 {
        parse_f64(path, line_number, "core_hole_width", tokens[4])?
    } else {
        let (width_line_number, width_line) = next_nonempty_line(path, lines, "core-hole width")?;
        let width_tokens = width_line.split_whitespace().collect::<Vec<_>>();
        if width_tokens.len() != 1 {
            return parse_error(
                path,
                width_line_number,
                format!(
                    "legacy header core-hole width line has {} token(s), expected 1",
                    width_tokens.len()
                ),
            );
        }
        parse_f64(path, width_line_number, "core_hole_width", width_tokens[0])?
    };
    Ok(XseclDatHeader {
        real_energy_count: parse_usize(path, line_number, "real_energy_count", tokens[0])?,
        fermi_index: parse_usize(path, line_number, "fermi_index", tokens[1])?,
        edge: parse_f64(path, line_number, "edge", tokens[2])?,
        emu: parse_f64(path, line_number, "emu", tokens[3])?,
        core_hole_width,
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

fn is_legacy_scalar_noise_row(line: &str) -> bool {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() || tokens.len() > 3 {
        return false;
    }
    let Ok(values) = tokens
        .iter()
        .map(|token| token.replace(['D', 'd'], "E").parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
    else {
        return false;
    };
    values
        .iter()
        .all(|value| value.is_finite() && (*value - values[0]).abs() <= 1.0e-12)
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

fn validate_xsecl_from_xsph_nrixs_input(input: &XseclFromXsphNrixsInput<'_>) -> Result<()> {
    let row_count = input.energy.len();
    if row_count == 0 {
        return parse_error("xsecl.dat", 0, "at least one energy row is required");
    }
    if input.header.real_energy_count == 0 || input.header.real_energy_count > row_count {
        return parse_error(
            "xsecl.dat",
            1,
            format!(
                "real energy count {} must be in 1..={row_count}",
                input.header.real_energy_count
            ),
        );
    }
    if input.header.fermi_index == 0 || input.header.fermi_index > input.header.real_energy_count {
        return parse_error(
            "xsecl.dat",
            1,
            format!(
                "fermi index {} must be in 1..={}",
                input.header.fermi_index, input.header.real_energy_count
            ),
        );
    }

    validate_channel_shape(
        "xsecl.dat",
        "decomposition_cross_sections",
        input.decomposition_cross_sections,
        row_count,
    )?;
    validate_channel_shape(
        "xsecl2.dat",
        "total_angular_cross_sections",
        input.total_angular_cross_sections,
        row_count,
    )?;
    let decomp_channels = input.decomposition_cross_sections.len_of(Axis(1));
    let total_channels = input.total_angular_cross_sections.len_of(Axis(1));
    if decomp_channels != total_channels {
        return parse_error(
            "xsecl2.dat",
            0,
            format!(
                "total-angular channel count {total_channels} must match decomposition channel count {decomp_channels}"
            ),
        );
    }

    let atom_shape = input.atom_cross_sections.shape();
    if atom_shape[0] != row_count {
        return Err(IoError::XseclBinShape {
            field: "atomxsec_energy",
            actual: vec![atom_shape[0]],
            expected: vec![row_count],
        });
    }
    if atom_shape[1] == 0 {
        return Err(IoError::InvalidXseclBin {
            field: "kfinmax",
            message: "at least one final state is required".to_string(),
        });
    }
    if input.transitions.len() > atom_shape[1] {
        return Err(IoError::XseclBinShape {
            field: "indmax",
            actual: vec![input.transitions.len()],
            expected: vec![atom_shape[1]],
        });
    }
    if input.pad_width <= 2 {
        return Err(IoError::InvalidPadWidth(input.pad_width));
    }

    Ok(())
}

fn validate_channel_shape(
    path: &'static str,
    field: &'static str,
    values: ArrayView2<'_, Complex64>,
    row_count: usize,
) -> Result<()> {
    let shape = values.shape();
    if shape[0] != row_count {
        return parse_error(
            path,
            0,
            format!("{field} has {} row(s), expected {row_count}", shape[0]),
        );
    }
    if shape[1] == 0 {
        return parse_error(path, 0, format!("{field} must have at least one channel"));
    }
    Ok(())
}

fn xsecl_table_from_rows(
    header: XseclDatHeader,
    energy: ArrayView1<'_, f64>,
    channels: ArrayView2<'_, Complex64>,
) -> XseclDatData {
    let channel_sum = channels
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().sum())
        .collect::<Vec<_>>();
    XseclDatData {
        header,
        energy: energy.to_owned(),
        channel_cross_sections: channels.to_owned(),
        channel_sum: Array1::from_vec(channel_sum),
    }
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
    fn parses_legacy_split_header_xsecl_dat() -> Result<()> {
        let parsed = parse_xsecl_dat(
            "2 1 -2.5e-1 4.08e2\n8.39493865e-2\n408.08358 -9.4722801e-5 5.8529371e-5 1.1556254e-4 -1.2086591e-4 -3.619343e-5 -5.30337e-6\n408.11859 -4.2446685e-5 -1.1776355e-4 1.0570503e-4 -1.4409145e-4 -1.6021114e-4 -3.8440289e-5\n",
        )?;

        assert_eq!(parsed.header.real_energy_count, 2);
        assert_eq!(parsed.header.fermi_index, 1);
        assert_eq!(parsed.header.edge, -0.25);
        assert_eq!(parsed.header.emu, 408.0);
        assert_eq!(parsed.header.core_hole_width, 8.394_938_65e-2);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.channel_count(), 2);
        Ok(())
    }

    #[test]
    fn skips_legacy_scalar_noise_rows_after_xsecl_data_rows() -> Result<()> {
        let parsed = parse_xsecl_dat(
            "2 1 -2.5e-1 4.08e2\n8.39493865e-2\n408.08358 -9.4722801e-5 5.8529371e-5 1.1556254e-4 -1.2086591e-4 -3.619343e-5 -5.30337e-6\n1.62272661165115 1.62272661165115 1.62272661165115\n1.62272661165115\n408.11859 -4.2446685e-5 -1.1776355e-4 1.0570503e-4 -1.4409145e-4 -1.6021114e-4 -3.8440289e-5\n",
        )?;

        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.channel_count(), 2);
        assert_eq!(parsed.energy[1], 408.11859);
        assert!(
            parse_xsecl_dat(
                "2 1 -2.5e-1 4.08e2\n8.39493865e-2\n408.08358 -9.4722801e-5 5.8529371e-5 1.1556254e-4 -1.2086591e-4 -3.619343e-5 -5.30337e-6\n1.0 2.0 3.0\n",
            )
            .is_err()
        );
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
    fn builds_xsecl_outputs_from_xsph_nrixs_rows() -> Result<()> {
        let header = XseclDatHeader {
            real_energy_count: 2,
            fermi_index: 1,
            edge: -0.25,
            emu: 408.0,
            core_hole_width: 0.083_949_386_5,
        };
        let energy = Array1::from_vec(vec![408.083_58, 408.118_59]);
        let decomposition_cross_sections = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(-0.000_094_722_801, 0.000_115_562_54),
                Complex64::new(0.000_058_529_371, -0.000_120_865_91),
                Complex64::new(-0.000_042_446_685, 0.000_105_705_03),
                Complex64::new(-0.000_117_763_55, -0.000_144_091_45),
            ],
        )
        .expect("test shape");
        let total_angular_cross_sections = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.11, -0.02),
                Complex64::new(-0.03, 0.04),
                Complex64::new(0.07, 0.09),
                Complex64::new(0.05, -0.01),
            ],
        )
        .expect("test shape");
        let atom_cross_sections = Array2::from_shape_fn((2, 3), |(row, state)| {
            Complex64::new(
                0.1 * (row + 1) as f64 + 0.01 * state as f64,
                -0.04 * (row + 1) as f64 - 0.005 * state as f64,
            )
        });
        let transitions = vec![
            XseclBinTransition {
                final_state_kappa: -1,
                decomposition_channel: 0,
                total_angular_momentum_channel: 0,
                orbital_angular_momentum: 0,
            },
            XseclBinTransition {
                final_state_kappa: 2,
                decomposition_channel: 1,
                total_angular_momentum_channel: 1,
                orbital_angular_momentum: 1,
            },
        ];

        let result = xsecl_from_xsph_nrixs(XseclFromXsphNrixsInput {
            header,
            energy: energy.view(),
            decomposition_cross_sections: decomposition_cross_sections.view(),
            total_angular_cross_sections: total_angular_cross_sections.view(),
            atom_cross_sections: atom_cross_sections.view(),
            transitions: &transitions,
            initial_state_j: 1,
            pad_width: 8,
        })?;

        assert_eq!(result.xsecl.header, header);
        assert_eq!(result.xsecl.energy, energy);
        assert_eq!(
            result.xsecl.channel_cross_sections,
            decomposition_cross_sections
        );
        assert_eq!(
            result.xsecl.channel_sum[0],
            decomposition_cross_sections[(0, 0)] + decomposition_cross_sections[(0, 1)]
        );
        assert_eq!(
            result.xsecl2.channel_sum[1],
            total_angular_cross_sections[(1, 0)] + total_angular_cross_sections[(1, 1)]
        );
        assert_eq!(result.xsecl_bin.initial_state_j, 1);
        assert_eq!(result.xsecl_bin.transitions, transitions);
        assert_eq!(result.xsecl_bin.atom_cross_sections, atom_cross_sections);
        assert_eq!(result.xsecl_bin.raw_atom_cross_section_pad, None);

        let parsed_xsecl = parse_xsecl_dat(&xsecl_dat_string(&result.xsecl)?)?;
        assert_eq!(parsed_xsecl.row_count(), result.xsecl.row_count());
        assert_eq!(parsed_xsecl.channel_count(), result.xsecl.channel_count());
        assert_complex_close(parsed_xsecl.channel_sum[0], result.xsecl.channel_sum[0]);
        let parsed_xsecl2 = parse_xsecl2_dat(&xsecl2_dat_string(&result.xsecl2)?)?;
        assert_eq!(parsed_xsecl2.row_count(), result.xsecl2.row_count());
        assert_eq!(parsed_xsecl2.channel_count(), result.xsecl2.channel_count());
        assert_complex_close(parsed_xsecl2.channel_sum[1], result.xsecl2.channel_sum[1]);
        let parsed_bin = crate::xsecl_bin::parse_xsecl_bin(
            &crate::xsecl_bin::xsecl_bin_string(&result.xsecl_bin)?,
            result.xsecl_bin.pad_width,
            result.xsecl_bin.energy_count(),
        )?;
        assert_eq!(parsed_bin.pad_width, result.xsecl_bin.pad_width);
        assert_eq!(parsed_bin.initial_state_j, result.xsecl_bin.initial_state_j);
        assert_eq!(parsed_bin.transitions, result.xsecl_bin.transitions);
        assert_eq!(
            parsed_bin.atom_cross_sections.dim(),
            result.xsecl_bin.atom_cross_sections.dim()
        );
        for (&actual, &expected) in parsed_bin
            .atom_cross_sections
            .iter()
            .zip(result.xsecl_bin.atom_cross_sections.iter())
        {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn xsecl_from_xsph_nrixs_rejects_shape_mismatches() {
        let header = XseclDatHeader {
            real_energy_count: 2,
            fermi_index: 1,
            edge: -0.25,
            emu: 408.0,
            core_hole_width: 0.083_949_386_5,
        };
        let energy = Array1::from_vec(vec![408.083_58, 408.118_59]);
        let decomposition_cross_sections = Array2::from_elem((2, 2), Complex64::new(0.1, -0.2));
        let total_angular_cross_sections = Array2::from_elem((2, 2), Complex64::new(0.2, 0.3));
        let atom_cross_sections = Array2::from_elem((1, 2), Complex64::new(0.4, -0.1));
        let transitions = vec![XseclBinTransition {
            final_state_kappa: -1,
            decomposition_channel: 0,
            total_angular_momentum_channel: 0,
            orbital_angular_momentum: 0,
        }];

        let error = xsecl_from_xsph_nrixs(XseclFromXsphNrixsInput {
            header,
            energy: energy.view(),
            decomposition_cross_sections: decomposition_cross_sections.view(),
            total_angular_cross_sections: total_angular_cross_sections.view(),
            atom_cross_sections: atom_cross_sections.view(),
            transitions: &transitions,
            initial_state_j: 1,
            pad_width: 8,
        })
        .expect_err("atomxsec rows must match the text-table energy rows");

        assert!(matches!(
            error,
            IoError::XseclBinShape {
                field: "atomxsec_energy",
                actual,
                expected,
            } if actual == vec![1] && expected == vec![2]
        ));
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual.re - expected.re).abs() < 1.0e-10,
            "real mismatch: {actual:?} != {expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() < 1.0e-10,
            "imag mismatch: {actual:?} != {expected:?}"
        );
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
