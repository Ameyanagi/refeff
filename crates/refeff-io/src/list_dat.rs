//! FEFF `list.dat` path-selection text codec.
//!
//! `GENFMT/genfmtsub.f90` writes `list.dat` as FEFF title records, a dashed
//! `rdhead` terminator, one label line, and one fixed-format row per generated
//! path. FF2X later reads the first two numeric columns as the path index and
//! user Debye-Waller factor.

use std::fmt::Write as _;
use std::path::Path;

use refeff_core::{
    GenfmtJasDriverOutput, GenfmtJasPathOutputs, GenfmtOrdinaryDriverOutput,
    GenfmtOrdinaryPathOutputs, GenfmtRetainedPathOutput,
};

use crate::error::{IoError, Result};
use crate::format::fortran_zero_scaled_exp;

const LIST_DAT_SEPARATOR: &str =
    " -----------------------------------------------------------------------";
const LIST_DAT_LABEL: &str = "  pathindex     sig2   amp ratio    deg    nlegs  r effective";
const LIST_DAT_ROW_TOKEN_COUNT: usize = 6;

/// One FEFF `list.dat` path row from `GENFMT/genfmtsub.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ListDatEntry {
    /// FEFF path index, `ipath`.
    pub path_index: usize,
    /// User Debye-Waller factor column, `sig2`.
    pub sigma2: f64,
    /// Curved-wave amplitude ratio/criterion, `crit`.
    pub amplitude_ratio: f64,
    /// Path degeneracy.
    pub degeneracy: f64,
    /// Number of legs in the path, `nleg`.
    pub leg_count: usize,
    /// Effective half path length in Angstrom. GENFMT writes `reff*bohr`.
    pub effective_half_path_length_angstrom: f64,
}

impl From<&GenfmtRetainedPathOutput> for ListDatEntry {
    fn from(output: &GenfmtRetainedPathOutput) -> Self {
        Self {
            path_index: output.path_index,
            sigma2: output.list_sigma2,
            amplitude_ratio: output.criterion_percent,
            degeneracy: output.degeneracy,
            leg_count: output.potential_indices.len(),
            effective_half_path_length_angstrom: output.effective_half_path_length_angstrom,
        }
    }
}

impl From<GenfmtRetainedPathOutput> for ListDatEntry {
    fn from(output: GenfmtRetainedPathOutput) -> Self {
        Self::from(&output)
    }
}

/// FEFF `list.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ListDatData {
    /// Header title records written by FEFF `wthead`.
    pub titles: Vec<String>,
    /// Generated path rows.
    pub entries: Vec<ListDatEntry>,
}

impl ListDatData {
    /// Build FEFF `list.dat` data from retained GENFMT path output.
    ///
    /// Entries are copied in caller-supplied order, matching the order FEFF
    /// writes retained paths during the `paths.dat` loop.
    #[must_use]
    pub fn from_genfmt_output(
        titles: &[String],
        retained_paths: &[GenfmtRetainedPathOutput],
    ) -> Self {
        Self {
            titles: titles.to_vec(),
            entries: retained_paths.iter().map(ListDatEntry::from).collect(),
        }
    }

    /// Build FEFF `list.dat` data from ordinary GENFMT path outputs.
    #[must_use]
    pub fn from_genfmt_ordinary_outputs(
        titles: &[String],
        outputs: &GenfmtOrdinaryPathOutputs,
    ) -> Self {
        Self::from_genfmt_output(titles, &outputs.retained_paths)
    }

    /// Build FEFF `list.dat` data from ordinary GENFMT driver output.
    #[must_use]
    pub fn from_genfmt_ordinary_driver_output(
        titles: &[String],
        output: &GenfmtOrdinaryDriverOutput,
    ) -> Self {
        Self::from_genfmt_ordinary_outputs(titles, &output.path_sequence.outputs)
    }

    /// Build FEFF `list.dat` data from GENFMTJAS path outputs.
    #[must_use]
    pub fn from_genfmt_jas_outputs(titles: &[String], outputs: &GenfmtJasPathOutputs) -> Self {
        Self::from_genfmt_output(titles, &outputs.retained_paths)
    }

    /// Build FEFF `list.dat` data from GENFMTJAS driver output.
    #[must_use]
    pub fn from_genfmt_jas_driver_output(
        titles: &[String],
        output: &GenfmtJasDriverOutput,
    ) -> Self {
        Self::from_genfmt_jas_outputs(titles, &output.path_sequence.outputs)
    }
}

/// Render FEFF `list.dat` text.
pub fn list_dat_string(data: &ListDatData) -> Result<String> {
    validate_list_dat(data)?;

    let mut out = String::new();
    for title in &data.titles {
        writeln!(out, "# {}", title.trim_end())?;
    }
    writeln!(out, "{LIST_DAT_SEPARATOR}")?;
    writeln!(out, "{LIST_DAT_LABEL}")?;
    for entry in &data.entries {
        write_entry(&mut out, *entry)?;
    }
    Ok(out)
}

/// Parse FEFF `list.dat` text.
pub fn parse_list_dat(text: &str) -> Result<ListDatData> {
    let scanned = scan_list_dat(text)?;
    let entries = scanned
        .rows
        .iter()
        .map(|row| parse_entry_row(row))
        .collect::<Result<Vec<_>>>()?;
    let data = ListDatData {
        titles: scanned.titles,
        entries,
    };
    validate_list_dat(&data)?;
    Ok(data)
}

/// Write FEFF `list.dat` text to a file.
pub fn write_list_dat(path: impl AsRef<Path>, data: &ListDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, list_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `list.dat` text from a file.
pub fn read_list_dat(path: impl AsRef<Path>) -> Result<ListDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_list_dat(&text)
}

fn write_entry(out: &mut String, entry: ListDatEntry) -> Result<()> {
    let sigma2 = fixed_field("sig2", entry.sigma2, 12, 5)?;
    let amplitude_ratio = exponent_field("crit", entry.amplitude_ratio, 15, 4)?;
    let degeneracy = fixed_field("deg", entry.degeneracy, 10, 3)?;
    let effective_half_path_length =
        fixed_field("reff", entry.effective_half_path_length_angstrom, 9, 4)?;

    writeln!(
        out,
        " {path_index:>8}{sigma2}{amplitude_ratio}{degeneracy}{leg_count:>6}{effective_half_path_length}",
        path_index = entry.path_index,
        leg_count = entry.leg_count
    )?;
    Ok(())
}

fn parse_entry_row(row: &ListDatRow<'_>) -> Result<ListDatEntry> {
    if row.tokens.len() != LIST_DAT_ROW_TOKEN_COUNT {
        return Err(IoError::ListDatRowWidth {
            line: row.line_number,
            actual: row.tokens.len(),
            expected: LIST_DAT_ROW_TOKEN_COUNT,
        });
    }

    Ok(ListDatEntry {
        path_index: parse_usize(row, 0, "path_index")?,
        sigma2: parse_f64(row, 1, "sig2")?,
        amplitude_ratio: parse_f64(row, 2, "crit")?,
        degeneracy: parse_f64(row, 3, "deg")?,
        leg_count: parse_usize(row, 4, "nleg")?,
        effective_half_path_length_angstrom: parse_f64(row, 5, "reff")?,
    })
}

fn validate_list_dat(data: &ListDatData) -> Result<()> {
    for entry in &data.entries {
        validate_entry(*entry)?;
    }
    Ok(())
}

fn validate_entry(entry: ListDatEntry) -> Result<()> {
    ensure_i_width("path_index", entry.path_index, 8)?;
    ensure_i_width("nleg", entry.leg_count, 6)?;
    if entry.leg_count == 0 {
        return Err(invalid_list_dat(
            "nleg",
            "at least one path leg is required",
        ));
    }
    ensure_finite("sig2", entry.sigma2)?;
    ensure_finite("crit", entry.amplitude_ratio)?;
    ensure_finite("deg", entry.degeneracy)?;
    ensure_finite("reff", entry.effective_half_path_length_angstrom)?;
    fixed_field("sig2", entry.sigma2, 12, 5)?;
    exponent_field("crit", entry.amplitude_ratio, 15, 4)?;
    fixed_field("deg", entry.degeneracy, 10, 3)?;
    fixed_field("reff", entry.effective_half_path_length_angstrom, 9, 4)?;
    Ok(())
}

#[derive(Debug)]
struct ScannedListDat<'a> {
    titles: Vec<String>,
    rows: Vec<ListDatRow<'a>>,
}

#[derive(Debug)]
struct ListDatRow<'a> {
    line_number: usize,
    tokens: Vec<&'a str>,
}

fn scan_list_dat(text: &str) -> Result<ScannedListDat<'_>> {
    let mut titles = Vec::new();
    let mut rows = Vec::new();
    let mut body_started = false;

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_separator(trimmed) || is_label(trimmed) {
            body_started = true;
            continue;
        }

        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if first_token_is_path_index(&tokens) {
            body_started = true;
            rows.push(ListDatRow {
                line_number,
                tokens,
            });
        } else if body_started {
            return Err(invalid_list_dat(
                "line",
                format!("unexpected non-row content on line {line_number}: {line:?}"),
            ));
        } else {
            titles.push(strip_wthead_prefix(line));
        }
    }

    Ok(ScannedListDat { titles, rows })
}

fn parse_usize(row: &ListDatRow<'_>, index: usize, field: &'static str) -> Result<usize> {
    let token = token(row, index, field)?;
    token
        .parse::<usize>()
        .map_err(|_| list_dat_parse(field, row.line_number, token))
}

fn parse_f64(row: &ListDatRow<'_>, index: usize, field: &'static str) -> Result<f64> {
    let token = token(row, index, field)?;
    token
        .parse::<f64>()
        .map_err(|_| list_dat_parse(field, row.line_number, token))
}

fn token<'a>(row: &'a ListDatRow<'a>, index: usize, field: &'static str) -> Result<&'a str> {
    row.tokens
        .get(index)
        .copied()
        .ok_or(IoError::ListDatMissing { field })
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

fn is_separator(trimmed: &str) -> bool {
    trimmed.len() >= 8 && trimmed.bytes().all(|byte| byte == b'-')
}

fn is_label(trimmed: &str) -> bool {
    let lower = trimmed.to_ascii_lowercase();
    lower.contains("pathindex") && lower.contains("sig2")
}

fn first_token_is_path_index(tokens: &[&str]) -> bool {
    tokens
        .first()
        .is_some_and(|token| token.parse::<usize>().is_ok())
}

fn ensure_i_width(field: &'static str, value: usize, width: usize) -> Result<()> {
    if value.to_string().len() > width {
        Err(invalid_list_dat(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn ensure_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_list_dat(field, "value must be finite"))
    }
}

fn fixed_field(field: &'static str, value: f64, width: usize, precision: usize) -> Result<String> {
    ensure_finite(field, value)?;
    let formatted = format!("{value:>width$.precision$}");
    ensure_field_width(field, &formatted, width)?;
    Ok(formatted)
}

fn exponent_field(
    field: &'static str,
    value: f64,
    width: usize,
    precision: usize,
) -> Result<String> {
    ensure_finite(field, value)?;
    let formatted = fortran_zero_scaled_exp(value, width, precision);
    ensure_field_width(field, &formatted, width)?;
    Ok(formatted)
}

fn ensure_field_width(field: &'static str, formatted: &str, width: usize) -> Result<()> {
    if formatted.len() <= width {
        Ok(())
    } else {
        Err(invalid_list_dat(
            field,
            format!("formatted value {formatted:?} exceeds width {width}"),
        ))
    }
}

fn list_dat_parse(field: &'static str, line: usize, token: &str) -> IoError {
    IoError::ListDatParse {
        field,
        line,
        token: token.to_string(),
    }
}

fn invalid_list_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidListDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array1, Array2};
    use num_complex::Complex64;
    use refeff_core::{GenfmtFeffBinHeader, GenfmtJasPathSequence, GenfmtOrdinaryPathSequence};

    #[test]
    fn writes_header_label_and_rows_like_feff() -> Result<()> {
        let text = list_dat_string(&sample_list_dat())?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("# PATH  Rmax= 6.000"));
        assert_eq!(lines.next(), Some(LIST_DAT_SEPARATOR));
        assert_eq!(lines.next(), Some(LIST_DAT_LABEL));
        assert_eq!(
            lines.next(),
            Some("       17     0.00000     0.1250E+02     4.000     3   2.5000")
        );
        Ok(())
    }

    #[test]
    fn builds_entry_from_genfmt_retained_path_output() {
        let output = sample_genfmt_retained_path_output();

        let entry = ListDatEntry::from(&output);
        assert_eq!(
            entry,
            ListDatEntry {
                path_index: output.path_index,
                sigma2: output.list_sigma2,
                amplitude_ratio: output.criterion_percent,
                degeneracy: output.degeneracy,
                leg_count: output.potential_indices.len(),
                effective_half_path_length_angstrom: output.effective_half_path_length_angstrom,
            }
        );
        assert_eq!(ListDatEntry::from(output), entry);
    }

    #[test]
    fn builds_data_from_genfmt_retained_path_outputs() {
        let titles = vec!["PATH  Rmax= 6.000".to_string(), "second".to_string()];
        let first = sample_genfmt_retained_path_output();
        let mut second = sample_genfmt_retained_path_output();
        second.path_index = 23;
        second.criterion_percent = 6.25;
        let retained_paths = vec![first.clone(), second.clone()];

        let data = ListDatData::from_genfmt_output(&titles, &retained_paths);

        assert_eq!(data.titles, titles);
        assert_eq!(
            data.entries,
            vec![ListDatEntry::from(first), ListDatEntry::from(second)]
        );
    }

    #[test]
    fn builds_data_from_genfmt_ordinary_path_outputs() {
        let titles = vec!["PATH  Rmax= 6.000".to_string()];
        let first = sample_genfmt_retained_path_output();
        let mut second = sample_genfmt_retained_path_output();
        second.path_index = 23;
        second.criterion_percent = 6.25;
        let outputs = GenfmtOrdinaryPathOutputs {
            examined_path_count: 3,
            retained_path_count: 2,
            final_normalization: Some(4.5),
            path_summaries: Vec::new(),
            retained_paths: vec![first.clone(), second.clone()],
        };

        let data = ListDatData::from_genfmt_ordinary_outputs(&titles, &outputs);

        assert_eq!(data.titles, titles);
        assert_eq!(
            data.entries,
            vec![ListDatEntry::from(first), ListDatEntry::from(second)]
        );
    }

    #[test]
    fn builds_data_from_genfmt_jas_path_outputs() {
        let titles = vec!["PATH  Rmax= 6.000".to_string()];
        let first = sample_genfmt_retained_path_output();
        let mut second = sample_genfmt_retained_path_output();
        second.path_index = 23;
        second.criterion_percent = 6.25;
        let outputs = GenfmtJasPathOutputs {
            examined_path_count: 3,
            retained_path_count: 2,
            final_normalization: Some(4.5),
            path_summaries: Vec::new(),
            retained_paths: vec![first.clone(), second.clone()],
            decomposed_paths: None,
        };

        let data = ListDatData::from_genfmt_jas_outputs(&titles, &outputs);

        assert_eq!(data.titles, titles);
        assert_eq!(
            data.entries,
            vec![ListDatEntry::from(first), ListDatEntry::from(second)]
        );
    }

    #[test]
    fn builds_data_from_genfmt_driver_outputs() {
        let titles = vec!["PATH  Rmax= 6.000".to_string()];
        let first = sample_genfmt_retained_path_output();
        let mut second = sample_genfmt_retained_path_output();
        second.path_index = 23;
        second.criterion_percent = 6.25;
        let ordinary_output = GenfmtOrdinaryDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtOrdinaryPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtOrdinaryPathOutputs {
                    examined_path_count: 3,
                    retained_path_count: 2,
                    final_normalization: Some(4.5),
                    path_summaries: Vec::new(),
                    retained_paths: vec![first.clone(), second.clone()],
                },
            },
            nstar_rows: None,
        };

        let data = ListDatData::from_genfmt_ordinary_driver_output(&titles, &ordinary_output);

        assert_eq!(data.titles, titles);
        assert_eq!(
            data.entries,
            vec![
                ListDatEntry::from(first.clone()),
                ListDatEntry::from(second.clone())
            ]
        );

        let jas_output = GenfmtJasDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtJasPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtJasPathOutputs {
                    examined_path_count: 2,
                    retained_path_count: 1,
                    final_normalization: Some(3.0),
                    path_summaries: Vec::new(),
                    retained_paths: vec![second.clone()],
                    decomposed_paths: None,
                },
            },
            nstar_rows: None,
        };

        let data = ListDatData::from_genfmt_jas_driver_output(&titles, &jas_output);

        assert_eq!(data.titles, titles);
        assert_eq!(data.entries, vec![ListDatEntry::from(second)]);
    }

    #[test]
    fn roundtrips_list_dat_text() -> Result<()> {
        let data = sample_list_dat();
        let rendered = list_dat_string(&data)?;
        assert_eq!(
            rendered,
            "# PATH  Rmax= 6.000\n -----------------------------------------------------------------------\n  pathindex     sig2   amp ratio    deg    nlegs  r effective\n       17     0.00000     0.1250E+02     4.000     3   2.5000\n"
        );
        let parsed = parse_list_dat(&rendered)?;
        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn accepts_headers_with_or_without_wthead_prefix() -> Result<()> {
        let text = "PATH raw\n# PATH written\n -----------------------------------------------------------------------\n  pathindex     sig2   amp ratio    deg    nlegs  r effective\n        1     0.00250     3.5000E+00     2.000     4   1.2500\n";
        let parsed = parse_list_dat(text)?;
        assert_eq!(parsed.titles, vec!["PATH raw", "PATH written"]);
        assert_eq!(parsed.entries[0].path_index, 1);
        assert_eq!(parsed.entries[0].sigma2, 0.0025);
        Ok(())
    }

    #[test]
    fn rejects_invalid_row_width_and_tokens() {
        assert!(matches!(
            parse_list_dat(
                " -----------------------------------------------------------------------\n  pathindex     sig2   amp ratio    deg    nlegs  r effective\n        1     0.00250\n"
            ),
            Err(IoError::ListDatRowWidth {
                line: 3,
                actual: 2,
                expected: LIST_DAT_ROW_TOKEN_COUNT,
            })
        ));

        assert!(matches!(
            parse_list_dat(
                " -----------------------------------------------------------------------\n  pathindex     sig2   amp ratio    deg    nlegs  r effective\n        1     nope     3.5000E+00     2.000     4   1.2500\n"
            ),
            Err(IoError::ListDatParse {
                field: "sig2",
                line: 3,
                ..
            })
        ));
    }

    fn sample_list_dat() -> ListDatData {
        ListDatData {
            titles: vec!["PATH  Rmax= 6.000".to_string()],
            entries: vec![ListDatEntry {
                path_index: 17,
                sigma2: 0.0,
                amplitude_ratio: 12.5,
                degeneracy: 4.0,
                leg_count: 3,
                effective_half_path_length_angstrom: 2.5,
            }],
        }
    }

    fn sample_genfmt_retained_path_output() -> GenfmtRetainedPathOutput {
        GenfmtRetainedPathOutput {
            path_index: 17,
            degeneracy: 4.0,
            criterion_percent: 12.5,
            effective_half_path_length_bohr: 2.4,
            effective_half_path_length_angstrom: 1.270_025_397_6,
            list_sigma2: 0.0,
            potential_indices: Array1::from_vec(vec![1, 2, 0]),
            positions: Array2::zeros((3, 3)),
            beta_angles: Array1::from_vec(vec![0.10, 0.20, 0.30]),
            eta_angles: Array1::from_vec(vec![0.40, 0.50, 0.60]),
            leg_lengths: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            amplitudes: Array1::from_vec(vec![0.2, 0.3, 0.4]),
            phases: Array1::from_vec(vec![0.1, 1.2, 2.3]),
        }
    }

    fn sample_genfmt_feff_bin_header() -> GenfmtFeffBinHeader {
        GenfmtFeffBinHeader {
            version: "refeff-test".to_string(),
            pad_width: 8,
            core_hole: 1,
            order: 2,
            initial_angular_momentum: 0,
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            edge_energy: 9.1,
            potentials: Vec::new(),
            central_phase_shifts: Array1::from_vec(vec![Complex64::new(0.1, -0.01)]),
            complex_momenta: Array1::from_vec(vec![Complex64::new(1.0, 0.1)]),
            wave_numbers: Array1::from_vec(vec![0.5]),
        }
    }
}
