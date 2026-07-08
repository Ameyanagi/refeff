//! FEFF XSPH `PrintRl` radial-function table (`rl.dat`) support.
//!
//! `XSPH/phase.f90` writes this file with FEFF's generic `WriteData` and
//! `WriteArrayData` helpers. The stream has two scalar header sections followed
//! by one scalar descriptor section and one radial-array section for each
//! absorber energy/channel record.

use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;
use refeff_core::{
    RixsRadialFunctionRecord, RixsRadialFunctionTable, RixsRadialFunctionTableInput,
    rixs_radial_function_table,
};

use crate::apot_bin::{
    ApotBinData, ApotBinPayload, ApotBinRecords, ApotBinSection, ApotBinType, ApotBinValue,
    apot_bin_string, parse_apot_bin,
};
use crate::error::{IoError, Result};

/// Parsed contents of FEFF XSPH `rl.dat`.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRlDatData {
    /// Muffin-tin radius from the first header row, FEFF `rmt`.
    pub muffin_tin_radius: f64,
    /// Active angular cutoff from the first header row, FEFF `lmax`.
    pub angular_limit: usize,
    /// One-based radial match row from the first header row, FEFF `jri`.
    pub radial_match_index_1based: usize,
    /// Loucks logarithmic grid step from the second header row, FEFF `dx`.
    pub log_step: f64,
    /// Loucks-grid origin from the second header row, FEFF `x0`.
    pub grid_origin: f64,
    /// Per-energy/per-angular normalized radial records.
    pub records: Vec<XsphRlDatRecord>,
}

impl XsphRlDatData {
    /// Number of radial records in the table.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Number of radial rows in each record.
    #[must_use]
    pub fn radial_count(&self) -> usize {
        self.radial_match_index_1based
    }

    /// Assemble the RIXS radial-function table FEFF reads from this file.
    pub fn to_rixs_handoff(&self) -> Result<XsphRlDatRixsHandoff> {
        xsph_rl_dat_rixs_handoff(self)
    }
}

/// One FEFF `rl.dat` radial record.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRlDatRecord {
    /// Current phase-grid energy, FEFF `em(ie)`.
    pub energy: Complex64,
    /// Non-negative angular label written as FEFF `Int2 = -ll`.
    pub angular_momentum: usize,
    /// Phase shift written as FEFF `ph(ie,ll)`.
    pub phase_shift: Complex64,
    /// Normalized large radial component, FEFF `p(1:jri) / temp`.
    pub regular_large: Array1<Complex64>,
    /// Normalized small radial component, FEFF `q(1:jri) / temp`.
    pub regular_small: Array1<Complex64>,
}

/// RIXS-ready view of a FEFF XSPH `rl.dat` file.
#[derive(Debug, Clone, PartialEq)]
pub struct XsphRlDatRixsHandoff {
    /// Muffin-tin radius from the first header row, FEFF `rmt`.
    pub muffin_tin_radius: f64,
    /// Active angular cutoff from the first header row, FEFF `lllmax`.
    pub angular_limit: usize,
    /// One-based radial match row from the first header row, FEFF `jri`.
    pub radial_match_index_1based: usize,
    /// Loucks logarithmic grid step from the second header row, FEFF `dx`.
    pub log_step: f64,
    /// Loucks-grid origin from the second header row, FEFF `x0`.
    pub grid_origin: f64,
    /// Number of RIXS energy rows inferred from record count.
    pub energy_count: usize,
    /// FEFF angular column count, `lllmax + 1`.
    pub angular_count: usize,
    /// Large-component radial functions in RIXS `(radial, angular, energy)` order.
    pub table: RixsRadialFunctionTable,
}

/// Parse FEFF XSPH `rl.dat` text.
pub fn parse_xsph_rl_dat(text: &str) -> Result<XsphRlDatData> {
    let stream = parse_apot_bin(text)
        .map_err(|source| invalid_rl_dat_error("section_stream", source.to_string()))?;
    xsph_rl_dat_from_section_stream(&stream)
}

/// Render FEFF-compatible XSPH `rl.dat` text.
pub fn xsph_rl_dat_string(data: &XsphRlDatData) -> Result<String> {
    validate_xsph_rl_dat(data)?;
    apot_bin_string(&xsph_rl_dat_section_stream(data)?)
        .map_err(|source| invalid_rl_dat_error("section_stream", source.to_string()))
}

/// Read FEFF XSPH `rl.dat` from a file.
pub fn read_xsph_rl_dat(path: impl AsRef<Path>) -> Result<XsphRlDatData> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| IoError::io(path, source))?;
    parse_xsph_rl_dat(&text)
}

/// Write FEFF-compatible XSPH `rl.dat` text to a file.
pub fn write_xsph_rl_dat(path: impl AsRef<Path>, data: &XsphRlDatData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, xsph_rl_dat_string(data)?).map_err(|source| IoError::io(path, source))
}

/// Build the FEFF RIXS radial-function handoff from parsed XSPH `rl.dat`.
///
/// `RIXS/rixs.f90` reads only the first complex array from each `WriteArrayData`
/// radial section, so this adapter forwards `regular_large` and preserves the
/// file's `(energy, angular)` record order.
pub fn xsph_rl_dat_rixs_handoff(data: &XsphRlDatData) -> Result<XsphRlDatRixsHandoff> {
    validate_xsph_rl_dat(data)?;
    let angular_count = data
        .angular_limit
        .checked_add(1)
        .ok_or_else(|| invalid_rl_dat_error("angular_limit", "angular count overflow"))?;
    if data.records.is_empty() {
        return invalid_rl_dat("records", "at least one RIXS radial record is required");
    }
    if !data.records.len().is_multiple_of(angular_count) {
        return invalid_rl_dat(
            "records",
            format!(
                "record count {} is not a multiple of angular count {angular_count}",
                data.records.len()
            ),
        );
    }
    let energy_count = data.records.len() / angular_count;
    let records = data
        .records
        .iter()
        .map(|record| {
            let angular_momentum = isize::try_from(record.angular_momentum)
                .map_err(|_| invalid_rl_dat_error("angular_momentum", "value exceeds isize"))?;
            Ok(RixsRadialFunctionRecord {
                energy: record.energy,
                angular_momentum,
                radial_values: record.regular_large.view(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let table = rixs_radial_function_table(RixsRadialFunctionTableInput {
        records: &records,
        energy_count,
        angular_count,
        radial_count: data.radial_match_index_1based,
    })
    .map_err(|source| invalid_rl_dat_error("rixs_radial_function_table", source.to_string()))?;

    Ok(XsphRlDatRixsHandoff {
        muffin_tin_radius: data.muffin_tin_radius,
        angular_limit: data.angular_limit,
        radial_match_index_1based: data.radial_match_index_1based,
        log_step: data.log_step,
        grid_origin: data.grid_origin,
        energy_count,
        angular_count,
        table,
    })
}

fn xsph_rl_dat_section_stream(data: &XsphRlDatData) -> Result<ApotBinData> {
    let mut section_number = 1_usize;
    let mut sections = Vec::with_capacity(
        data.records
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| invalid_rl_dat_error("records", "section count overflow"))?,
    );

    sections.push(record_section(
        section_number,
        vec![ApotBinType::Double, ApotBinType::Int, ApotBinType::Int],
        vec![vec![
            ApotBinValue::Real(data.muffin_tin_radius),
            ApotBinValue::Int(usize_to_i64("angular_limit", data.angular_limit)?),
            ApotBinValue::Int(usize_to_i64(
                "radial_match_index_1based",
                data.radial_match_index_1based,
            )?),
        ]],
    ));
    section_number += 1;

    sections.push(record_section(
        section_number,
        vec![ApotBinType::Double, ApotBinType::Double],
        vec![vec![
            ApotBinValue::Real(data.log_step),
            ApotBinValue::Real(data.grid_origin),
        ]],
    ));
    section_number += 1;

    for record in &data.records {
        sections.push(record_section(
            section_number,
            vec![
                ApotBinType::DComplex,
                ApotBinType::Int,
                ApotBinType::DComplex,
            ],
            vec![vec![
                ApotBinValue::Complex(record.energy),
                ApotBinValue::Int(usize_to_i64("angular_momentum", record.angular_momentum)?),
                ApotBinValue::Complex(record.phase_shift),
            ]],
        ));
        section_number += 1;

        let mut rows = Vec::with_capacity(record.regular_large.len());
        for index in 0..record.regular_large.len() {
            rows.push(vec![
                ApotBinValue::Complex(record.regular_large[index]),
                ApotBinValue::Complex(record.regular_small[index]),
            ]);
        }
        sections.push(record_section(
            section_number,
            vec![ApotBinType::DComplex, ApotBinType::DComplex],
            rows,
        ));
        section_number += 1;
    }

    Ok(ApotBinData { sections })
}

fn xsph_rl_dat_from_section_stream(stream: &ApotBinData) -> Result<XsphRlDatData> {
    if stream.sections.len() < 2 {
        return invalid_rl_dat(
            "sections",
            format!(
                "expected at least 2 section(s), got {}",
                stream.sections.len()
            ),
        );
    }
    if !(stream.sections.len() - 2).is_multiple_of(2) {
        return invalid_rl_dat(
            "sections",
            "radial record sections must appear in descriptor/array pairs",
        );
    }

    let header_row = single_record_row(
        &stream.sections[0],
        &[ApotBinType::Double, ApotBinType::Int, ApotBinType::Int],
        "first_header",
    )?;
    let muffin_tin_radius = real_value(&header_row[0], "muffin_tin_radius")?;
    let angular_limit = usize_value(&header_row[1], "angular_limit")?;
    let radial_match_index_1based = usize_value(&header_row[2], "radial_match_index_1based")?;

    let grid_row = single_record_row(
        &stream.sections[1],
        &[ApotBinType::Double, ApotBinType::Double],
        "grid_header",
    )?;
    let log_step = real_value(&grid_row[0], "log_step")?;
    let grid_origin = real_value(&grid_row[1], "grid_origin")?;

    let mut records = Vec::with_capacity((stream.sections.len() - 2) / 2);
    for pair in stream.sections[2..].chunks_exact(2) {
        let descriptor_row = single_record_row(
            &pair[0],
            &[
                ApotBinType::DComplex,
                ApotBinType::Int,
                ApotBinType::DComplex,
            ],
            "record_descriptor",
        )?;
        let rows = record_rows(
            &pair[1],
            &[ApotBinType::DComplex, ApotBinType::DComplex],
            "record_rows",
        )?;
        let mut regular_large = Vec::with_capacity(rows.len());
        let mut regular_small = Vec::with_capacity(rows.len());
        for row in rows {
            regular_large.push(complex_value(&row[0], "regular_large")?);
            regular_small.push(complex_value(&row[1], "regular_small")?);
        }
        records.push(XsphRlDatRecord {
            energy: complex_value(&descriptor_row[0], "energy")?,
            angular_momentum: usize_value(&descriptor_row[1], "angular_momentum")?,
            phase_shift: complex_value(&descriptor_row[2], "phase_shift")?,
            regular_large: Array1::from_vec(regular_large),
            regular_small: Array1::from_vec(regular_small),
        });
    }

    let data = XsphRlDatData {
        muffin_tin_radius,
        angular_limit,
        radial_match_index_1based,
        log_step,
        grid_origin,
        records,
    };
    validate_xsph_rl_dat(&data)?;
    Ok(data)
}

fn record_section(
    section_number: usize,
    column_types: Vec<ApotBinType>,
    rows: Vec<Vec<ApotBinValue>>,
) -> ApotBinSection {
    ApotBinSection {
        section_number,
        headers: Vec::new(),
        header_texts: Vec::new(),
        column_labels: Vec::new(),
        column_label_text: None,
        payload: ApotBinPayload::Records(ApotBinRecords { column_types, rows }),
        trailing_headers: Vec::new(),
        trailing_header_texts: Vec::new(),
    }
}

fn single_record_row<'a>(
    section: &'a ApotBinSection,
    expected: &[ApotBinType],
    field: &'static str,
) -> Result<&'a [ApotBinValue]> {
    let rows = record_rows(section, expected, field)?;
    if rows.len() != 1 {
        return invalid_rl_dat(field, format!("expected 1 row, got {}", rows.len()));
    }
    Ok(&rows[0])
}

fn record_rows<'a>(
    section: &'a ApotBinSection,
    expected: &[ApotBinType],
    field: &'static str,
) -> Result<&'a [Vec<ApotBinValue>]> {
    let Some(records) = section.records() else {
        return invalid_rl_dat(field, "section does not contain record rows");
    };
    if records.column_types != expected {
        return invalid_rl_dat(
            field,
            format!(
                "unexpected column types {:?}, expected {:?}",
                records.column_types, expected
            ),
        );
    }
    Ok(&records.rows)
}

fn real_value(value: &ApotBinValue, field: &'static str) -> Result<f64> {
    if let ApotBinValue::Real(value) = value {
        return Ok(*value);
    }
    invalid_rl_dat(field, "expected real value")
}

fn complex_value(value: &ApotBinValue, field: &'static str) -> Result<Complex64> {
    if let ApotBinValue::Complex(value) = value {
        return Ok(*value);
    }
    invalid_rl_dat(field, "expected complex value")
}

fn usize_value(value: &ApotBinValue, field: &'static str) -> Result<usize> {
    let ApotBinValue::Int(value) = value else {
        return invalid_rl_dat(field, "expected integer value");
    };
    if *value < 0 {
        return invalid_rl_dat(field, "value must be non-negative");
    }
    usize::try_from(*value).map_err(|_| invalid_rl_dat_error(field, "value exceeds usize"))
}

fn usize_to_i64(field: &'static str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| invalid_rl_dat_error(field, "value exceeds i64"))
}

fn validate_xsph_rl_dat(data: &XsphRlDatData) -> Result<()> {
    validate_positive_finite("muffin_tin_radius", data.muffin_tin_radius)?;
    validate_positive_finite("log_step", data.log_step)?;
    validate_finite("grid_origin", data.grid_origin)?;
    if data.radial_match_index_1based == 0 {
        return invalid_rl_dat("radial_match_index_1based", "value must be positive");
    }

    for (record_index, record) in data.records.iter().enumerate() {
        if record.angular_momentum > data.angular_limit {
            return invalid_rl_dat(
                "angular_momentum",
                format!(
                    "record {} angular momentum {} exceeds lmax {}",
                    record_index + 1,
                    record.angular_momentum,
                    data.angular_limit
                ),
            );
        }
        validate_complex("energy", record.energy)?;
        validate_complex("phase_shift", record.phase_shift)?;
        if record.regular_large.len() != data.radial_match_index_1based {
            return invalid_rl_dat(
                "regular_large",
                format!(
                    "record {} has {} row(s), expected {}",
                    record_index + 1,
                    record.regular_large.len(),
                    data.radial_match_index_1based
                ),
            );
        }
        if record.regular_small.len() != data.radial_match_index_1based {
            return invalid_rl_dat(
                "regular_small",
                format!(
                    "record {} has {} row(s), expected {}",
                    record_index + 1,
                    record.regular_small.len(),
                    data.radial_match_index_1based
                ),
            );
        }
        for value in &record.regular_large {
            validate_complex("regular_large", *value)?;
        }
        for value in &record.regular_small {
            validate_complex("regular_small", *value)?;
        }
    }
    Ok(())
}

fn validate_positive_finite(field: &'static str, value: f64) -> Result<()> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        return invalid_rl_dat(field, "value must be positive");
    }
    Ok(())
}

fn validate_complex(field: &'static str, value: Complex64) -> Result<()> {
    validate_finite(field, value.re)?;
    validate_finite(field, value.im)
}

fn validate_finite(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_rl_dat(field, "value must be finite")
    }
}

fn invalid_rl_dat<T>(field: &'static str, message: impl Into<String>) -> Result<T> {
    Err(invalid_rl_dat_error(field, message))
}

fn invalid_rl_dat_error(field: &'static str, message: impl Into<String>) -> IoError {
    IoError::InvalidXsphRlDat {
        field,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xsph_rl_dat_roundtrips_section_stream() -> Result<()> {
        let data = sample_xsph_rl_dat();

        let text = xsph_rl_dat_string(&data)?;
        assert!(text.contains("#DT#  Double Int Int"));
        assert!(text.contains("#DT#  DComplex Int DComplex"));
        assert!(text.contains("#DT#  DComplex DComplex"));

        let parsed = parse_xsph_rl_dat(&text)?;
        assert_eq!(parsed, data);
        assert_eq!(parsed.record_count(), 2);
        assert_eq!(parsed.radial_count(), 3);
        Ok(())
    }

    #[test]
    fn xsph_rl_dat_rejects_mismatched_radial_rows() {
        let mut data = sample_xsph_rl_dat();
        data.records[0].regular_small = Array1::from_vec(vec![Complex64::new(0.0, 0.0)]);

        assert!(xsph_rl_dat_string(&data).is_err());
    }

    #[test]
    fn xsph_rl_dat_rejects_unpaired_record_sections() -> Result<()> {
        let mut stream = xsph_rl_dat_section_stream(&sample_xsph_rl_dat())?;
        stream.sections.pop();
        let text = apot_bin_string(&stream)?;

        assert!(parse_xsph_rl_dat(&text).is_err());
        Ok(())
    }

    #[test]
    fn xsph_rl_dat_builds_rixs_radial_handoff() -> Result<()> {
        let data = sample_xsph_rl_dat();

        let handoff = xsph_rl_dat_rixs_handoff(&data)?;

        assert_eq!(handoff.energy_count, 1);
        assert_eq!(handoff.angular_count, 2);
        assert_eq!(handoff.table.energies[0], data.records[1].energy);
        assert_eq!(
            handoff.table.radial_functions[(1, 0, 0)],
            data.records[0].regular_large[1]
        );
        assert_eq!(
            handoff.table.radial_functions[(2, 1, 0)],
            data.records[1].regular_large[2]
        );
        Ok(())
    }

    #[test]
    fn xsph_rl_dat_rejects_partial_rixs_record_group() {
        let mut data = sample_xsph_rl_dat();
        data.records.pop();

        assert!(xsph_rl_dat_rixs_handoff(&data).is_err());
    }

    fn sample_xsph_rl_dat() -> XsphRlDatData {
        XsphRlDatData {
            muffin_tin_radius: 1.25,
            angular_limit: 1,
            radial_match_index_1based: 3,
            log_step: 0.0625,
            grid_origin: 8.5,
            records: vec![
                XsphRlDatRecord {
                    energy: Complex64::new(0.5, 0.125),
                    angular_momentum: 0,
                    phase_shift: Complex64::new(0.25, -0.125),
                    regular_large: Array1::from_vec(vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.5, 0.25),
                        Complex64::new(0.25, 0.125),
                    ]),
                    regular_small: Array1::from_vec(vec![
                        Complex64::new(0.0, 0.5),
                        Complex64::new(0.25, 0.125),
                        Complex64::new(0.125, 0.0625),
                    ]),
                },
                XsphRlDatRecord {
                    energy: Complex64::new(1.0, 0.25),
                    angular_momentum: 1,
                    phase_shift: Complex64::new(-0.25, 0.5),
                    regular_large: Array1::from_vec(vec![
                        Complex64::new(0.75, 0.0),
                        Complex64::new(0.375, 0.1875),
                        Complex64::new(0.1875, 0.09375),
                    ]),
                    regular_small: Array1::from_vec(vec![
                        Complex64::new(0.0, 0.375),
                        Complex64::new(0.1875, 0.09375),
                        Complex64::new(0.09375, 0.046875),
                    ]),
                },
            ],
        }
    }
}
