//! FEFF `xsect.dat` cross-section text codec.
//!
//! FEFF10 writes `xsect.dat` from `XSPH/xsphsub.f90` and reads it in FF2X via
//! the historical `rdxbin` routine. The file is formatted text: FEFF title
//! records, a dashed `rdhead` terminator, two commented scalar records, one
//! commented label, and one cross-section row per energy point. FEFF writes the
//! method and table records with unscaled `Ew.d` descriptors, while the
//! `gamach` record uses a `1P` scale factor and stops before its unused trailing
//! literal.

use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{FEFF_HARTREE_EV, wave_number_from_hartree};

use crate::error::{IoError, Result};
use crate::format::{fortran_exp, fortran_zero_scaled_exp};

const XSECT_DAT_SEPARATOR: &str =
    "#  -----------------------------------------------------------------------";
const XSECT_DAT_LABEL: &str = "#       em              xsnorm            xsec  ";
const METHOD_TOKEN_COUNT: usize = 5;
const GAMACH_TOKEN_COUNT: usize = 3;
const ROW_TOKEN_COUNT: usize = 5;

/// Scalar fields from the FEFF `xsect.dat` method record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsectDatScalars {
    /// Many-body amplitude reduction factor, `s02`.
    pub amplitude_reduction: f64,
    /// Relaxation-energy estimate, `erelax`.
    pub relaxation_energy: f64,
    /// Plasmon-frequency estimate, `wp`.
    pub plasmon_frequency: f64,
    /// Edge energy as written by FEFF.
    pub edge_energy: f64,
    /// Chemical-potential position in the absorption spectrum, `emu`.
    pub chemical_potential: f64,
}

/// FEFF `xsect.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct XsectDatData {
    /// Header title records written by FEFF `wthead`.
    pub titles: Vec<String>,
    /// Method/scalar record.
    pub scalars: XsectDatScalars,
    /// Core-hole width in eV, `gamach*hart`.
    pub core_hole_width_ev: f64,
    /// Number of main energy points, `ne1`.
    pub main_energy_count: usize,
    /// FEFF `ik0` index.
    pub fermi_index: usize,
    /// Complex energy grid in eV as written in the first two data columns.
    pub energy_grid_ev: Array1<Complex64>,
    /// Normalized atomic background, `xsnorm`.
    pub normalized_background: Array1<f64>,
    /// Complex cross-section data, `xsec`.
    pub cross_section: Array1<Complex64>,
}

/// FF2X-ready values derived from FEFF `xsect.dat`.
///
/// This is the typed Rust equivalent of the `FF2X/ff2gen.f90` `rdxbin`
/// handoff: the energy mesh and core-hole width are converted from eV to
/// Hartree units, while `edge_energy` and `chemical_potential` remain in the
/// FEFF internal units stored in the method record.
#[derive(Debug, Clone, PartialEq)]
pub struct XsectFf2xHandoff {
    /// Header title records forwarded from `xsect.dat`.
    pub titles: Vec<String>,
    /// Number of header records.
    pub title_count: usize,
    /// Effective amplitude reduction after FEFF's `mbconv`/existing-S02 rule.
    pub amplitude_reduction: f64,
    /// Method-record amplitude reduction before the caller rule is applied.
    pub file_amplitude_reduction: f64,
    /// Relaxation-energy estimate, `erelax`.
    pub relaxation_energy: f64,
    /// Plasmon-frequency estimate, `wp`.
    pub plasmon_frequency: f64,
    /// Edge shift `edgep` in FEFF internal Hartree units.
    pub edge_energy_hartree: f64,
    /// Chemical-potential position `emu` in FEFF internal Hartree units.
    pub chemical_potential_hartree: f64,
    /// Core-hole width `gamach` converted from eV to Hartree.
    pub core_hole_width_hartree: f64,
    /// Number of horizontal-axis points, FEFF `ne1`.
    pub main_energy_count: usize,
    /// FEFF one-based Fermi index, `ik0`.
    pub fermi_index_1based: usize,
    /// Rust zero-based Fermi index.
    pub fermi_index: usize,
    /// Number of cross-section rows read, FEFF `nxsec`.
    pub cross_section_count: usize,
    /// Complex energy grid converted from eV to Hartree, FEFF `emxs`.
    pub energy_grid_hartree: Array1<Complex64>,
    /// Real output energy grid, FEFF `omega`.
    pub omega_hartree: Array1<f64>,
    /// Signed photoelectron wave number, FEFF `xkxs`.
    pub wave_number: Array1<f64>,
    /// Normalized atomic background, FEFF `xsnorm`.
    pub normalized_background: Array1<f64>,
    /// Complex cross-section data, FEFF `xsec`.
    pub cross_section: Array1<Complex64>,
}

impl XsectDatData {
    /// Number of rows in the cross-section table.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.energy_grid_ev.len()
    }
}

impl XsectFf2xHandoff {
    /// Number of rows in the FF2X handoff table.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.energy_grid_hartree.len()
    }
}

/// Convert parsed `xsect.dat` contents into FEFF `rdxbin` handoff arrays.
///
/// `current_amplitude_reduction` is the caller's current `s02` value and
/// `many_body_convolution` is FEFF `mbconv`. FEFF replaces the caller S02 with
/// the file value when `mbconv > 0` or the caller value is `<= 0.1`.
pub fn xsect_dat_ff2x_handoff(
    data: &XsectDatData,
    current_amplitude_reduction: f64,
    many_body_convolution: i32,
) -> Result<XsectFf2xHandoff> {
    validate_xsect_dat(data)?;
    validate_finite_xsect_dat("s02", current_amplitude_reduction)?;
    validate_ff2x_handoff_indices(data)?;

    let amplitude_reduction = if many_body_convolution > 0 || current_amplitude_reduction <= 0.1 {
        data.scalars.amplitude_reduction
    } else {
        current_amplitude_reduction
    };
    let energy_grid_hartree = data.energy_grid_ev.mapv(|energy| energy / FEFF_HARTREE_EV);
    let omega_hartree = energy_grid_hartree
        .mapv(|energy| energy.re - data.scalars.edge_energy + data.scalars.chemical_potential);
    let wave_number = energy_grid_hartree
        .mapv(|energy| wave_number_from_hartree(energy.re - data.scalars.edge_energy));

    Ok(XsectFf2xHandoff {
        titles: data.titles.clone(),
        title_count: data.titles.len(),
        amplitude_reduction,
        file_amplitude_reduction: data.scalars.amplitude_reduction,
        relaxation_energy: data.scalars.relaxation_energy,
        plasmon_frequency: data.scalars.plasmon_frequency,
        edge_energy_hartree: data.scalars.edge_energy,
        chemical_potential_hartree: data.scalars.chemical_potential,
        core_hole_width_hartree: data.core_hole_width_ev / FEFF_HARTREE_EV,
        main_energy_count: data.main_energy_count,
        fermi_index_1based: data.fermi_index,
        fermi_index: data.fermi_index - 1,
        cross_section_count: data.energy_count(),
        energy_grid_hartree,
        omega_hartree,
        wave_number,
        normalized_background: data.normalized_background.clone(),
        cross_section: data.cross_section.clone(),
    })
}

/// Render FEFF `xsect.dat` text.
pub fn xsect_dat_string(data: &XsectDatData) -> Result<String> {
    validate_xsect_dat(data)?;

    let mut out = String::new();
    for title in &data.titles {
        writeln!(out, "# {}", title.trim_end())?;
    }
    writeln!(out, "{XSECT_DAT_SEPARATOR}")?;
    writeln!(
        out,
        "# {}{}{}{}{} method to calculate xsect",
        zero_scaled_e13_5(data.scalars.amplitude_reduction)?,
        zero_scaled_e13_5(data.scalars.relaxation_energy)?,
        zero_scaled_e13_5(data.scalars.plasmon_frequency)?,
        zero_scaled_e15_7(data.scalars.edge_energy)?,
        zero_scaled_e15_7(data.scalars.chemical_potential)?
    )?;
    writeln!(
        out,
        "# {}{main_energy_count:>7}{fermi_index:>7}",
        one_scaled_e15_7(data.core_hole_width_ev)?,
        main_energy_count = data.main_energy_count,
        fermi_index = data.fermi_index
    )?;
    writeln!(out, "{XSECT_DAT_LABEL}")?;

    for ((energy, xsnorm), xsec) in data
        .energy_grid_ev
        .iter()
        .zip(data.normalized_background.iter())
        .zip(data.cross_section.iter())
    {
        writeln!(
            out,
            "{}{}{}{}{}",
            zero_scaled_e17_9(energy.re)?,
            zero_scaled_e13_5(energy.im)?,
            zero_scaled_e13_5(*xsnorm)?,
            zero_scaled_e13_5(xsec.re)?,
            zero_scaled_e13_5(xsec.im)?
        )?;
    }
    Ok(out)
}

/// Parse FEFF `xsect.dat` text.
pub fn parse_xsect_dat(text: &str) -> Result<XsectDatData> {
    let mut lines = XsectDatLines::new(text);
    let titles = lines.titles()?;
    let method = lines.method_record()?;
    let gamach = lines.gamach_record()?;
    lines.label()?;

    let mut energy_grid_ev = Vec::new();
    let mut normalized_background = Vec::new();
    let mut cross_section = Vec::new();
    while let Some(row) = lines.next_data_row()? {
        energy_grid_ev.push(Complex64::new(row[0], row[1]));
        normalized_background.push(row[2]);
        cross_section.push(Complex64::new(row[3], row[4]));
    }

    let data = XsectDatData {
        titles,
        scalars: XsectDatScalars {
            amplitude_reduction: method[0],
            relaxation_energy: method[1],
            plasmon_frequency: method[2],
            edge_energy: method[3],
            chemical_potential: method[4],
        },
        core_hole_width_ev: gamach.0,
        main_energy_count: gamach.1,
        fermi_index: gamach.2,
        energy_grid_ev: Array1::from_vec(energy_grid_ev),
        normalized_background: Array1::from_vec(normalized_background),
        cross_section: Array1::from_vec(cross_section),
    };
    validate_xsect_dat(&data)?;
    Ok(data)
}

/// Write FEFF `xsect.dat` text to a file.
pub fn write_xsect_dat(path: impl AsRef<Path>, data: &XsectDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xsect_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `xsect.dat` text from a file.
pub fn read_xsect_dat(path: impl AsRef<Path>) -> Result<XsectDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xsect_dat(&text)
}

fn validate_xsect_dat(data: &XsectDatData) -> Result<()> {
    let energy_count = data.energy_count();
    if energy_count == 0 {
        return Err(invalid_xsect_dat(
            "energy_grid_ev",
            "at least one energy row is required",
        ));
    }
    validate_len(
        "normalized_background",
        data.normalized_background.len(),
        energy_count,
    )?;
    validate_len("cross_section", data.cross_section.len(), energy_count)?;
    if data.main_energy_count == 0 || data.main_energy_count > energy_count {
        return Err(invalid_xsect_dat(
            "ne1",
            format!(
                "main energy count {} must be in 1..={energy_count}",
                data.main_energy_count
            ),
        ));
    }
    ensure_i_width("ne1", data.main_energy_count, 7)?;
    ensure_i_width("ik0", data.fermi_index, 7)?;

    zero_scaled_e13_5(data.scalars.amplitude_reduction)?;
    zero_scaled_e13_5(data.scalars.relaxation_energy)?;
    zero_scaled_e13_5(data.scalars.plasmon_frequency)?;
    zero_scaled_e15_7(data.scalars.edge_energy)?;
    zero_scaled_e15_7(data.scalars.chemical_potential)?;
    one_scaled_e15_7(data.core_hole_width_ev)?;

    for (index, ((energy, xsnorm), xsec)) in data
        .energy_grid_ev
        .iter()
        .zip(data.normalized_background.iter())
        .zip(data.cross_section.iter())
        .enumerate()
    {
        let row = index + 1;
        zero_scaled_e17_9_field("em.re", energy.re, row)?;
        zero_scaled_e13_5_field("em.im", energy.im, row)?;
        zero_scaled_e13_5_field("xsnorm", *xsnorm, row)?;
        zero_scaled_e13_5_field("xsec.re", xsec.re, row)?;
        zero_scaled_e13_5_field("xsec.im", xsec.im, row)?;
    }
    Ok(())
}

fn validate_ff2x_handoff_indices(data: &XsectDatData) -> Result<()> {
    if data.fermi_index == 0 || data.fermi_index > data.main_energy_count {
        return Err(invalid_xsect_dat(
            "ik0",
            format!(
                "Fermi index {} must be in 1..={}",
                data.fermi_index, data.main_energy_count
            ),
        ));
    }
    Ok(())
}

fn validate_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(IoError::XsectDatShape {
            field,
            actual,
            expected,
        })
    }
}

fn validate_finite_xsect_dat(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_xsect_dat(field, "value must be finite"))
    }
}

struct XsectDatLines<'a> {
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
}

impl<'a> XsectDatLines<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines().enumerate(),
        }
    }

    fn titles(&mut self) -> Result<Vec<String>> {
        let mut titles = Vec::new();
        for (index, line) in self.lines.by_ref() {
            if is_rdhead_separator(line) {
                return Ok(titles);
            }
            if !line.trim().is_empty() {
                titles.push(strip_wthead_prefix(line));
            }
            if index == usize::MAX {
                return Err(invalid_xsect_dat("line", "line index overflowed"));
            }
        }
        Err(IoError::XsectDatMissing { field: "separator" })
    }

    fn method_record(&mut self) -> Result<[f64; METHOD_TOKEN_COUNT]> {
        let (line_number, line) = self.next_required("method")?;
        parse_f64_record::<METHOD_TOKEN_COUNT>(line, line_number, "method")
    }

    fn gamach_record(&mut self) -> Result<(f64, usize, usize)> {
        let (line_number, line) = self.next_required("gamach")?;
        let tokens = record_tokens(line);
        if tokens.len() < GAMACH_TOKEN_COUNT {
            return Err(IoError::XsectDatRowWidth {
                line: line_number,
                actual: tokens.len(),
                expected: GAMACH_TOKEN_COUNT,
            });
        }
        Ok((
            parse_f64_token(tokens[0], line_number, "gamach")?,
            parse_usize_token(tokens[1], line_number, "ne1")?,
            parse_usize_token(tokens[2], line_number, "ik0")?,
        ))
    }

    fn label(&mut self) -> Result<()> {
        let (line_number, line) = self.next_required("label")?;
        let normalized = strip_comment_marker(line).to_ascii_lowercase();
        if normalized.contains("em") && normalized.contains("xsnorm") && normalized.contains("xsec")
        {
            Ok(())
        } else {
            Err(invalid_xsect_dat(
                "label",
                format!("unexpected label on line {line_number}: {line:?}"),
            ))
        }
    }

    fn next_data_row(&mut self) -> Result<Option<[f64; ROW_TOKEN_COUNT]>> {
        for (index, line) in self.lines.by_ref() {
            let line_number = index + 1;
            if line.trim().is_empty() {
                continue;
            }
            return parse_f64_record::<ROW_TOKEN_COUNT>(line, line_number, "row").map(Some);
        }
        Ok(None)
    }

    fn next_required(&mut self, field: &'static str) -> Result<(usize, &'a str)> {
        for (index, line) in self.lines.by_ref() {
            if !line.trim().is_empty() {
                return Ok((index + 1, line));
            }
        }
        Err(IoError::XsectDatMissing { field })
    }
}

fn parse_f64_record<const N: usize>(
    line: &str,
    line_number: usize,
    field: &'static str,
) -> Result<[f64; N]> {
    let tokens = record_tokens(line);
    if tokens.len() < N {
        return Err(IoError::XsectDatRowWidth {
            line: line_number,
            actual: tokens.len(),
            expected: N,
        });
    }

    let mut values = [0.0; N];
    for (index, value) in values.iter_mut().enumerate() {
        *value = parse_f64_token(tokens[index], line_number, field)?;
    }
    Ok(values)
}

fn record_tokens(line: &str) -> Vec<&str> {
    strip_comment_marker(line).split_whitespace().collect()
}

fn strip_comment_marker(line: &str) -> &str {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix('#') {
        rest.trim_start()
    } else {
        trimmed
    }
}

fn parse_f64_token(token: &str, line: usize, field: &'static str) -> Result<f64> {
    token
        .parse::<f64>()
        .map_err(|_| xsect_dat_parse(field, line, token))
}

fn parse_usize_token(token: &str, line: usize, field: &'static str) -> Result<usize> {
    token
        .parse::<usize>()
        .map_err(|_| xsect_dat_parse(field, line, token))
}

fn strip_wthead_prefix(line: &str) -> String {
    let trimmed = line.trim_end();
    if let Some(rest) = trimmed.strip_prefix("# ") {
        rest.to_string()
    } else if let Some(rest) = trimmed.strip_prefix('#') {
        rest.trim_start().to_string()
    } else {
        trimmed.to_string()
    }
}

fn is_rdhead_separator(line: &str) -> bool {
    let bytes = line.as_bytes();
    bytes.len() >= 11 && bytes[3..11].iter().all(|byte| *byte == b'-')
}

fn zero_scaled_e13_5(value: f64) -> Result<String> {
    zero_scaled_exp_field("value", value, 13, 5)
}

fn zero_scaled_e15_7(value: f64) -> Result<String> {
    zero_scaled_exp_field("value", value, 15, 7)
}

fn one_scaled_e15_7(value: f64) -> Result<String> {
    exp_field("value", value, 15, 7)
}

fn zero_scaled_e17_9(value: f64) -> Result<String> {
    zero_scaled_exp_field("value", value, 17, 9)
}

fn zero_scaled_e13_5_field(field: &'static str, value: f64, row: usize) -> Result<String> {
    zero_scaled_exp_field_with_context(field, value, 13, 5, row)
}

fn zero_scaled_e17_9_field(field: &'static str, value: f64, row: usize) -> Result<String> {
    zero_scaled_exp_field_with_context(field, value, 17, 9, row)
}

fn exp_field(field: &'static str, value: f64, width: usize, precision: usize) -> Result<String> {
    exp_field_with_context(field, value, width, precision, 0)
}

fn exp_field_with_context(
    field: &'static str,
    value: f64,
    width: usize,
    precision: usize,
    row: usize,
) -> Result<String> {
    if !value.is_finite() {
        return Err(invalid_xsect_dat(
            field,
            if row == 0 {
                "value must be finite".to_string()
            } else {
                format!("row {row} value must be finite")
            },
        ));
    }
    let formatted = fortran_exp(value, width, precision);
    if formatted.len() > width {
        Err(invalid_xsect_dat(
            field,
            format!("formatted value {formatted:?} exceeds width {width}"),
        ))
    } else {
        Ok(formatted)
    }
}

fn zero_scaled_exp_field(
    field: &'static str,
    value: f64,
    width: usize,
    precision: usize,
) -> Result<String> {
    zero_scaled_exp_field_with_context(field, value, width, precision, 0)
}

fn zero_scaled_exp_field_with_context(
    field: &'static str,
    value: f64,
    width: usize,
    precision: usize,
    row: usize,
) -> Result<String> {
    if !value.is_finite() {
        return Err(invalid_xsect_dat(
            field,
            if row == 0 {
                "value must be finite".to_string()
            } else {
                format!("row {row} value must be finite")
            },
        ));
    }
    let formatted = fortran_zero_scaled_exp(value, width, precision);
    if formatted.len() > width {
        Err(invalid_xsect_dat(
            field,
            format!("formatted value {formatted:?} exceeds width {width}"),
        ))
    } else {
        Ok(formatted)
    }
}

fn ensure_i_width(field: &'static str, value: usize, width: usize) -> Result<()> {
    if value.to_string().len() > width {
        Err(invalid_xsect_dat(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn xsect_dat_parse(field: &'static str, line: usize, token: &str) -> IoError {
    IoError::XsectDatParse {
        field,
        line,
        token: token.to_string(),
    }
}

fn invalid_xsect_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXsectDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_xsect_header_and_rows_like_feff() -> Result<()> {
        let text = xsect_dat_string(&sample_xsect_dat())?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("# Cu crystal"));
        assert_eq!(lines.next(), Some(XSECT_DAT_SEPARATOR));
        assert_eq!(
            lines.next(),
            Some(
                "#   0.85000E+00  0.15000E+00  0.24000E+01  0.9100000E+01 -0.4000000E+00 method to calculate xsect"
            )
        );
        assert_eq!(lines.next(), Some("#   1.2300000E+00      2      1"));
        assert_eq!(lines.next(), Some(XSECT_DAT_LABEL));
        assert_eq!(
            lines.next(),
            Some("  0.125000000E+01  0.10000E-01  0.20000E+01  0.30000E+01 -0.40000E+00")
        );
        Ok(())
    }

    #[test]
    fn roundtrips_xsect_dat_text() -> Result<()> {
        let data = sample_xsect_dat();
        let parsed = parse_xsect_dat(&xsect_dat_string(&data)?)?;
        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn accepts_comment_prefixed_feff_records() -> Result<()> {
        let text = "# Cu crystal\n#  -----------------------------------------------------------------------\n#   8.50000E-01  1.50000E-01  2.40000E+00  9.1000000E+00 -4.0000000E-01 method to calculate xsect\n#   1.2300000E+00      2      1 gamach in eV, # of points on horizontal axis\n#       em              xsnorm            xsec  \n  1.250000000E+00  1.00000E-02  2.00000E+00  3.00000E+00 -4.00000E-01\n  1.500000000E+00  2.00000E-02  2.50000E+00  3.50000E+00 -5.00000E-01\n";
        let parsed = parse_xsect_dat(text)?;
        assert_eq!(parsed.titles, vec!["Cu crystal"]);
        assert_eq!(parsed.main_energy_count, 2);
        assert_eq!(parsed.fermi_index, 1);
        assert_eq!(parsed.energy_count(), 2);
        assert_eq!(parsed.energy_grid_ev[0], Complex64::new(1.25, 0.01));
        assert_eq!(parsed.cross_section[1], Complex64::new(3.5, -0.5));
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes_and_tokens() {
        let mut bad = sample_xsect_dat();
        bad.normalized_background = Array1::from_vec(vec![1.0]);
        assert!(matches!(
            xsect_dat_string(&bad),
            Err(IoError::XsectDatShape {
                field: "normalized_background",
                actual: 1,
                expected: 2,
            })
        ));

        assert!(matches!(
            parse_xsect_dat(
                "# Cu\n#  -----------------------------------------------------------------------\n# nope\n"
            ),
            Err(IoError::XsectDatParse {
                field: "method",
                line: 3,
                ..
            }) | Err(IoError::XsectDatRowWidth { line: 3, .. })
        ));
    }

    #[test]
    fn builds_ff2x_rdxbin_handoff_like_feff() -> Result<()> {
        let data = sample_xsect_dat_for_handoff();
        let handoff = xsect_dat_ff2x_handoff(&data, 0.05, 0)?;

        assert_eq!(handoff.titles, vec!["Cu crystal"]);
        assert_eq!(handoff.title_count, 1);
        assert_close(handoff.amplitude_reduction, 0.85);
        assert_close(handoff.file_amplitude_reduction, 0.85);
        assert_close(handoff.relaxation_energy, 0.15);
        assert_close(handoff.plasmon_frequency, 2.4);
        assert_close(handoff.edge_energy_hartree, 9.1);
        assert_close(handoff.chemical_potential_hartree, -0.4);
        assert_close(handoff.core_hole_width_hartree, 0.045_201_650_073_373_67);
        assert_eq!(handoff.main_energy_count, 2);
        assert_eq!(handoff.fermi_index_1based, 1);
        assert_eq!(handoff.fermi_index, 0);
        assert_eq!(handoff.cross_section_count, 3);
        assert_eq!(handoff.energy_count(), 3);

        let expected_energy = [
            Complex64::new(0.045_936_636_253_428_524, 0.000_367_493_090_027_428_2),
            Complex64::new(0.055_123_963_504_114_23, 0.000_734_986_180_054_856_4),
            Complex64::new(0.367_493_090_027_428_2, 0.001_102_479_270_082_284_6),
        ];
        let expected_omega = [
            -9.454_063_363_746_572,
            -9.444_876_036_495_886,
            -9.132_506_909_972_571,
        ];
        let expected_wave = [
            -4.255_364_464_707_241,
            -4.253_204_917_822_767,
            -4.179_116_392_246_708_5,
        ];

        for row in 0..3 {
            assert_complex_close(handoff.energy_grid_hartree[row], expected_energy[row]);
            assert_close(handoff.omega_hartree[row], expected_omega[row]);
            assert_close(handoff.wave_number[row], expected_wave[row]);
            assert_eq!(handoff.cross_section[row], data.cross_section[row]);
            assert_eq!(
                handoff.normalized_background[row],
                data.normalized_background[row]
            );
        }
        Ok(())
    }

    #[test]
    fn ff2x_handoff_preserves_existing_s02_when_feff_would() -> Result<()> {
        let data = sample_xsect_dat_for_handoff();

        assert_close(
            xsect_dat_ff2x_handoff(&data, 0.9, 0)?.amplitude_reduction,
            0.9,
        );
        assert_close(
            xsect_dat_ff2x_handoff(&data, 0.9, 1)?.amplitude_reduction,
            0.85,
        );
        assert!(matches!(
            xsect_dat_ff2x_handoff(&data, f64::NAN, 0),
            Err(IoError::InvalidXsectDat { field: "s02", .. })
        ));

        let mut bad = data;
        bad.fermi_index = 0;
        assert!(matches!(
            xsect_dat_ff2x_handoff(&bad, 0.9, 0),
            Err(IoError::InvalidXsectDat { field: "ik0", .. })
        ));
        Ok(())
    }

    fn sample_xsect_dat() -> XsectDatData {
        XsectDatData {
            titles: vec!["Cu crystal".to_string()],
            scalars: XsectDatScalars {
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                edge_energy: 9.1,
                chemical_potential: -0.4,
            },
            core_hole_width_ev: 1.23,
            main_energy_count: 2,
            fermi_index: 1,
            energy_grid_ev: Array1::from_vec(vec![
                Complex64::new(1.25, 0.01),
                Complex64::new(1.5, 0.02),
            ]),
            normalized_background: Array1::from_vec(vec![2.0, 2.5]),
            cross_section: Array1::from_vec(vec![
                Complex64::new(3.0, -0.4),
                Complex64::new(3.5, -0.5),
            ]),
        }
    }

    fn sample_xsect_dat_for_handoff() -> XsectDatData {
        XsectDatData {
            titles: vec!["Cu crystal".to_string()],
            scalars: XsectDatScalars {
                amplitude_reduction: 0.85,
                relaxation_energy: 0.15,
                plasmon_frequency: 2.4,
                edge_energy: 9.1,
                chemical_potential: -0.4,
            },
            core_hole_width_ev: 1.23,
            main_energy_count: 2,
            fermi_index: 1,
            energy_grid_ev: Array1::from_vec(vec![
                Complex64::new(1.25, 0.01),
                Complex64::new(1.5, 0.02),
                Complex64::new(10.0, 0.03),
            ]),
            normalized_background: Array1::from_vec(vec![2.0, 2.5, 3.0]),
            cross_section: Array1::from_vec(vec![
                Complex64::new(3.0, -0.4),
                Complex64::new(3.5, -0.5),
                Complex64::new(4.0, -0.6),
            ]),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert_close(actual.re, expected.re);
        assert_close(actual.im, expected.im);
    }
}
