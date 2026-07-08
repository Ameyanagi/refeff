//! FEFF `nstar.dat` path-normalization text codec.
//!
//! `GENFMT/genfmtsub.f90` and `GENFMT/genfmtjas.f90` optionally write
//! `nstar.dat` when `wnstar` is enabled. Each path row uses the Fortran
//! format `(1x,i6,f10.3)`.

use std::fmt::Write as _;
use std::path::Path;

use refeff_core::{
    GenfmtJasDriverOutput, GenfmtNStarRow, GenfmtNStarRows, GenfmtOrdinaryDriverOutput,
};

use crate::error::{IoError, Result};

const NSTAR_DAT_LABEL: &str = " npath     n*";
const NSTAR_DAT_ROW_TOKEN_COUNT: usize = 2;

/// One FEFF `nstar.dat` path row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NStarDatEntry {
    /// Sequential GENFMT path number, `npath`.
    pub path_number: usize,
    /// Polarization-dependent path normalization, `xstar(...)`.
    pub nstar: f64,
}

impl From<&GenfmtNStarRow> for NStarDatEntry {
    fn from(row: &GenfmtNStarRow) -> Self {
        Self {
            path_number: row.path_number,
            nstar: row.nstar,
        }
    }
}

impl From<GenfmtNStarRow> for NStarDatEntry {
    fn from(row: GenfmtNStarRow) -> Self {
        Self::from(&row)
    }
}

/// FEFF `nstar.dat` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct NStarDatData {
    /// Polarization vector written before the path rows.
    pub polarization: [f64; 3],
    /// Generated path normalization rows.
    pub entries: Vec<NStarDatEntry>,
}

impl NStarDatData {
    /// Build FEFF `nstar.dat` data from GENFMT `wnstar` bookkeeping output.
    ///
    /// Entries are copied in caller-supplied traversal order, matching the order
    /// FEFF writes one row per examined `paths.dat` path.
    #[must_use]
    pub fn from_genfmt_nstar_rows(rows: &GenfmtNStarRows) -> Self {
        Self {
            polarization: rows.primary_polarization,
            entries: rows.rows.iter().map(NStarDatEntry::from).collect(),
        }
    }

    /// Build optional FEFF `nstar.dat` data from ordinary GENFMT driver output.
    #[must_use]
    pub fn from_genfmt_ordinary_driver_output(output: &GenfmtOrdinaryDriverOutput) -> Option<Self> {
        output.nstar_rows.as_ref().map(Self::from_genfmt_nstar_rows)
    }

    /// Build optional FEFF `nstar.dat` data from GENFMTJAS driver output.
    #[must_use]
    pub fn from_genfmt_jas_driver_output(output: &GenfmtJasDriverOutput) -> Option<Self> {
        output.nstar_rows.as_ref().map(Self::from_genfmt_nstar_rows)
    }
}

impl From<&GenfmtNStarRows> for NStarDatData {
    fn from(rows: &GenfmtNStarRows) -> Self {
        Self::from_genfmt_nstar_rows(rows)
    }
}

impl From<GenfmtNStarRows> for NStarDatData {
    fn from(rows: GenfmtNStarRows) -> Self {
        Self::from(&rows)
    }
}

/// Render FEFF `nstar.dat` text.
pub fn nstar_dat_string(data: &NStarDatData) -> Result<String> {
    validate_nstar_dat(data)?;

    let mut out = String::new();
    writeln!(
        out,
        " polarization{}{}{}",
        fixed_field("polarization", data.polarization[0], 8, 4)?,
        fixed_field("polarization", data.polarization[1], 8, 4)?,
        fixed_field("polarization", data.polarization[2], 8, 4)?,
    )?;
    writeln!(out, "{NSTAR_DAT_LABEL}")?;
    for entry in &data.entries {
        write_entry(&mut out, *entry)?;
    }
    Ok(out)
}

/// Parse FEFF `nstar.dat` text.
pub fn parse_nstar_dat(text: &str) -> Result<NStarDatData> {
    let mut polarization = None;
    let mut label_line = None;
    let mut rows = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
        if is_polarization_line(&tokens) {
            polarization = Some(parse_polarization(&tokens, line_number)?);
        } else if is_label_line(&tokens) {
            label_line = Some(line_number);
        } else if label_line.is_some() {
            rows.push(NStarDatRow {
                line_number,
                tokens,
            });
        } else {
            return Err(invalid_nstar_dat(
                "line",
                format!(
                    "unexpected content before nstar.dat label on line {line_number}: {line:?}"
                ),
            ));
        }
    }

    let data = NStarDatData {
        polarization: polarization.ok_or(IoError::NStarDatMissing {
            field: "polarization",
        })?,
        entries: rows
            .iter()
            .map(parse_entry_row)
            .collect::<Result<Vec<_>>>()?,
    };
    if label_line.is_none() {
        return Err(IoError::NStarDatMissing { field: "label" });
    }
    validate_nstar_dat(&data)?;
    Ok(data)
}

/// Write FEFF `nstar.dat` text to a file.
pub fn write_nstar_dat(path: impl AsRef<Path>, data: &NStarDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, nstar_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Read FEFF `nstar.dat` text from a file.
pub fn read_nstar_dat(path: impl AsRef<Path>) -> Result<NStarDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_nstar_dat(&text)
}

fn write_entry(out: &mut String, entry: NStarDatEntry) -> Result<()> {
    let nstar = fixed_field("nstar", entry.nstar, 10, 3)?;
    writeln!(
        out,
        " {path_number:>6}{nstar}",
        path_number = entry.path_number
    )?;
    Ok(())
}

fn parse_entry_row(row: &NStarDatRow<'_>) -> Result<NStarDatEntry> {
    if row.tokens.len() != NSTAR_DAT_ROW_TOKEN_COUNT {
        return Err(IoError::NStarDatRowWidth {
            line: row.line_number,
            actual: row.tokens.len(),
            expected: NSTAR_DAT_ROW_TOKEN_COUNT,
        });
    }

    Ok(NStarDatEntry {
        path_number: parse_usize(row, 0, "npath")?,
        nstar: parse_f64(row, 1, "nstar")?,
    })
}

fn validate_nstar_dat(data: &NStarDatData) -> Result<()> {
    for value in data.polarization {
        fixed_field("polarization", value, 8, 4)?;
    }
    for entry in &data.entries {
        validate_entry(*entry)?;
    }
    Ok(())
}

fn validate_entry(entry: NStarDatEntry) -> Result<()> {
    ensure_i_width("npath", entry.path_number, 6)?;
    if entry.path_number == 0 {
        return Err(invalid_nstar_dat(
            "npath",
            "path numbering starts at one in FEFF GENFMT output",
        ));
    }
    fixed_field("nstar", entry.nstar, 10, 3)?;
    Ok(())
}

#[derive(Debug)]
struct NStarDatRow<'a> {
    line_number: usize,
    tokens: Vec<&'a str>,
}

fn parse_polarization(tokens: &[&str], line: usize) -> Result<[f64; 3]> {
    if tokens.len() != 4 {
        return Err(IoError::NStarDatRowWidth {
            line,
            actual: tokens.len(),
            expected: 4,
        });
    }

    Ok([
        parse_f64_token("polarization", line, tokens[1])?,
        parse_f64_token("polarization", line, tokens[2])?,
        parse_f64_token("polarization", line, tokens[3])?,
    ])
}

fn parse_usize(row: &NStarDatRow<'_>, index: usize, field: &'static str) -> Result<usize> {
    let token = token(row, index, field)?;
    token
        .parse::<usize>()
        .map_err(|_| nstar_dat_parse(field, row.line_number, token))
}

fn parse_f64(row: &NStarDatRow<'_>, index: usize, field: &'static str) -> Result<f64> {
    let token = token(row, index, field)?;
    parse_f64_token(field, row.line_number, token)
}

fn parse_f64_token(field: &'static str, line: usize, token: &str) -> Result<f64> {
    token
        .parse::<f64>()
        .map_err(|_| nstar_dat_parse(field, line, token))
}

fn token<'a>(row: &'a NStarDatRow<'a>, index: usize, field: &'static str) -> Result<&'a str> {
    row.tokens
        .get(index)
        .copied()
        .ok_or(IoError::NStarDatMissing { field })
}

fn is_polarization_line(tokens: &[&str]) -> bool {
    tokens
        .first()
        .is_some_and(|token| token.eq_ignore_ascii_case("polarization"))
}

fn is_label_line(tokens: &[&str]) -> bool {
    tokens.len() >= 2
        && tokens[0].eq_ignore_ascii_case("npath")
        && tokens[1].eq_ignore_ascii_case("n*")
}

fn ensure_i_width(field: &'static str, value: usize, width: usize) -> Result<()> {
    if value.to_string().len() > width {
        Err(invalid_nstar_dat(
            field,
            format!("value {value} does not fit FEFF i{width} output"),
        ))
    } else {
        Ok(())
    }
}

fn fixed_field(field: &'static str, value: f64, width: usize, precision: usize) -> Result<String> {
    ensure_finite(field, value)?;
    let formatted = format!("{value:>width$.precision$}");
    ensure_field_width(field, &formatted, width)?;
    Ok(formatted)
}

fn ensure_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_nstar_dat(field, "value must be finite"))
    }
}

fn ensure_field_width(field: &'static str, formatted: &str, width: usize) -> Result<()> {
    if formatted.len() <= width {
        Ok(())
    } else {
        Err(invalid_nstar_dat(
            field,
            format!("formatted value {formatted:?} exceeds width {width}"),
        ))
    }
}

fn nstar_dat_parse(field: &'static str, line: usize, token: &str) -> IoError {
    IoError::NStarDatParse {
        field,
        line,
        token: token.to_string(),
    }
}

fn invalid_nstar_dat(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidNStarDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use num_complex::Complex64;
    use refeff_core::{
        GenfmtFeffBinHeader, GenfmtJasPathOutputs, GenfmtJasPathSequence,
        GenfmtOrdinaryPathOutputs, GenfmtOrdinaryPathSequence,
    };

    #[test]
    fn writes_header_label_and_rows_like_feff() -> Result<()> {
        let text = nstar_dat_string(&sample_nstar_dat())?;
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some(" polarization  1.0000  0.0000 -0.5000"));
        assert_eq!(lines.next(), Some(NSTAR_DAT_LABEL));
        assert_eq!(lines.next(), Some("     17     2.345"));
        assert_eq!(lines.next(), Some("     18    -0.125"));
        Ok(())
    }

    #[test]
    fn builds_data_from_genfmt_nstar_rows() -> Result<()> {
        let rows = GenfmtNStarRows {
            primary_polarization: [0.25, -0.5, 1.0],
            rows: vec![
                GenfmtNStarRow {
                    path_number: 1,
                    nstar: 2.345,
                },
                GenfmtNStarRow {
                    path_number: 2,
                    nstar: -0.125,
                },
            ],
        };

        let data = NStarDatData::from_genfmt_nstar_rows(&rows);

        assert_eq!(data.polarization, [0.25, -0.5, 1.0]);
        assert_eq!(
            data.entries,
            vec![
                NStarDatEntry {
                    path_number: 1,
                    nstar: 2.345,
                },
                NStarDatEntry {
                    path_number: 2,
                    nstar: -0.125,
                },
            ]
        );
        assert_eq!(
            nstar_dat_string(&data)?,
            " polarization  0.2500 -0.5000  1.0000\n npath     n*\n      1     2.345\n      2    -0.125\n"
        );
        Ok(())
    }

    #[test]
    fn builds_optional_data_from_genfmt_driver_outputs() {
        let rows = GenfmtNStarRows {
            primary_polarization: [0.25, -0.5, 1.0],
            rows: vec![GenfmtNStarRow {
                path_number: 1,
                nstar: 2.345,
            }],
        };
        let ordinary_output = GenfmtOrdinaryDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtOrdinaryPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtOrdinaryPathOutputs {
                    examined_path_count: 1,
                    retained_path_count: 0,
                    final_normalization: Some(1.0),
                    path_summaries: Vec::new(),
                    retained_paths: Vec::new(),
                },
            },
            nstar_rows: Some(rows.clone()),
        };
        let data = NStarDatData::from_genfmt_ordinary_driver_output(&ordinary_output)
            .expect("ordinary nstar output");
        assert_eq!(data, NStarDatData::from_genfmt_nstar_rows(&rows));

        let jas_output = GenfmtJasDriverOutput {
            header: sample_genfmt_feff_bin_header(),
            path_sequence: GenfmtJasPathSequence {
                evaluations: Vec::new(),
                outputs: GenfmtJasPathOutputs {
                    examined_path_count: 0,
                    retained_path_count: 0,
                    final_normalization: None,
                    path_summaries: Vec::new(),
                    retained_paths: Vec::new(),
                    decomposed_paths: None,
                },
            },
            nstar_rows: None,
        };
        assert_eq!(
            NStarDatData::from_genfmt_jas_driver_output(&jas_output),
            None
        );
    }

    #[test]
    fn roundtrips_nstar_dat_text() -> Result<()> {
        let data = sample_nstar_dat();
        let rendered = nstar_dat_string(&data)?;
        assert_eq!(
            rendered,
            " polarization  1.0000  0.0000 -0.5000\n npath     n*\n     17     2.345\n     18    -0.125\n"
        );
        let parsed = parse_nstar_dat(&rendered)?;
        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn parses_feff_reference_shape() -> Result<()> {
        let text = " polarization  1.0000  0.0000  0.0000\n npath     n*\n      1     2.345\n     12    -0.125\n";

        let parsed = parse_nstar_dat(text)?;

        assert_eq!(parsed.polarization, [1.0, 0.0, 0.0]);
        assert_eq!(
            parsed.entries,
            vec![
                NStarDatEntry {
                    path_number: 1,
                    nstar: 2.345,
                },
                NStarDatEntry {
                    path_number: 12,
                    nstar: -0.125,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_row_width_and_tokens() {
        assert!(matches!(
            parse_nstar_dat(" polarization  1.0000  0.0000  0.0000\n npath     n*\n      1\n"),
            Err(IoError::NStarDatRowWidth {
                line: 3,
                actual: 1,
                expected: NSTAR_DAT_ROW_TOKEN_COUNT,
            })
        ));

        assert!(matches!(
            parse_nstar_dat(
                " polarization  1.0000  0.0000  0.0000\n npath     n*\n      1      nope\n"
            ),
            Err(IoError::NStarDatParse {
                field: "nstar",
                line: 3,
                ..
            })
        ));
    }

    #[test]
    fn rejects_nonfinite_values_and_overwide_path_numbers() {
        assert!(matches!(
            nstar_dat_string(&NStarDatData {
                polarization: [1.0, f64::NAN, 0.0],
                entries: Vec::new(),
            }),
            Err(IoError::InvalidNStarDat {
                field: "polarization",
                ..
            })
        ));

        assert!(matches!(
            nstar_dat_string(&NStarDatData {
                polarization: [1.0, 0.0, 0.0],
                entries: vec![NStarDatEntry {
                    path_number: 1_000_000,
                    nstar: 1.0,
                }],
            }),
            Err(IoError::InvalidNStarDat { field: "npath", .. })
        ));
    }

    fn sample_nstar_dat() -> NStarDatData {
        NStarDatData {
            polarization: [1.0, 0.0, -0.5],
            entries: vec![
                NStarDatEntry {
                    path_number: 17,
                    nstar: 2.345,
                },
                NStarDatEntry {
                    path_number: 18,
                    nstar: -0.125,
                },
            ],
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
