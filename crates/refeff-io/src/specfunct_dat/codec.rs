use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::{IoError, Result};

use super::support::{
    checked_add, checked_product, invalid_specfunct_dat, invalid_specfunct_dat_value,
};
use super::types::{SPECFUNCT_DAT_INFO_COLUMNS, SfconvSpecfunctData};
use super::validation::validate_specfunct_dat;

const HEADER_RECORD_BYTES: usize = 32;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F64_BYTES: usize = 8;

pub fn parse_specfunct_dat(bytes: &[u8]) -> Result<SfconvSpecfunctData> {
    let endian = detect_endian(bytes)?;
    let mut position = 0;

    let header = read_record(bytes, &mut position, endian, "header")?;
    if header.len() != HEADER_RECORD_BYTES {
        return invalid_specfunct_dat(format!(
            "header record has {} byte(s), expected {HEADER_RECORD_BYTES}",
            header.len()
        ));
    }

    let wigner_seitz_radius = read_f64(header, 0, endian)?;
    let core_hole_lifetime = read_f64(header, F64_BYTES, endian)?;
    let asymmetric_phase = read_i32(header, F64_BYTES * 2, endian)?;
    let satellite_type = read_i32(header, F64_BYTES * 2 + INTEGER_BYTES, endian)?;
    let low_q_mode = read_i32(header, F64_BYTES * 2 + INTEGER_BYTES * 2, endian)?;
    let pole_count = parse_nonnegative_i32(
        read_i32(header, F64_BYTES * 2 + INTEGER_BYTES * 3, endian)?,
        "npl",
    )?;

    let pole_energy = parse_f64_vector_record(
        read_record(bytes, &mut position, endian, "pole energy")?,
        endian,
        "pole energy",
    )?;
    let pole_broadening = parse_f64_vector_record(
        read_record(bytes, &mut position, endian, "pole broadening")?,
        endian,
        "pole broadening",
    )?;
    let pole_weight = parse_f64_vector_record(
        read_record(bytes, &mut position, endian, "pole weight")?,
        endian,
        "pole weight",
    )?;

    let spectral_info_payload = read_record(bytes, &mut position, endian, "spectral info")?;
    let momentum_count = infer_row_count(spectral_info_payload, SPECFUNCT_DAT_INFO_COLUMNS)?;
    let spectral_info = parse_column_major_matrix_record(
        spectral_info_payload,
        momentum_count,
        SPECFUNCT_DAT_INFO_COLUMNS,
        endian,
        "spectral info",
    )?;
    let weights = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "weights")?,
        momentum_count,
        SPECFUNCT_DAT_INFO_COLUMNS,
        endian,
        "weights",
    )?;

    let extrinsic_quasiparticle_payload =
        read_record(bytes, &mut position, endian, "extrinsic quasiparticle")?;
    let spectral_point_count = infer_column_count(extrinsic_quasiparticle_payload, momentum_count)?;
    let extrinsic_quasiparticle = parse_column_major_matrix_record(
        extrinsic_quasiparticle_payload,
        momentum_count,
        spectral_point_count,
        endian,
        "extrinsic quasiparticle",
    )?;
    let extrinsic_satellite = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "extrinsic satellite")?,
        momentum_count,
        spectral_point_count,
        endian,
        "extrinsic satellite",
    )?;
    let interference_quasiparticle = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "interference quasiparticle")?,
        momentum_count,
        spectral_point_count,
        endian,
        "interference quasiparticle",
    )?;
    let interference_satellite = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "interference satellite")?,
        momentum_count,
        spectral_point_count,
        endian,
        "interference satellite",
    )?;
    let intrinsic_satellite = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "intrinsic satellite")?,
        momentum_count,
        spectral_point_count,
        endian,
        "intrinsic satellite",
    )?;
    let clipped_extrinsic_satellite = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "clipped extrinsic satellite")?,
        momentum_count,
        spectral_point_count,
        endian,
        "clipped extrinsic satellite",
    )?;
    let energy_grid = parse_column_major_matrix_record(
        read_record(bytes, &mut position, endian, "energy grid")?,
        momentum_count,
        spectral_point_count,
        endian,
        "energy grid",
    )?;

    if position != bytes.len() {
        return invalid_specfunct_dat(format!(
            "specfunct.dat has {} trailing byte(s)",
            bytes.len() - position
        ));
    }

    let data = SfconvSpecfunctData {
        wigner_seitz_radius,
        core_hole_lifetime,
        asymmetric_phase,
        satellite_type,
        low_q_mode,
        pole_count,
        pole_energy,
        pole_broadening,
        pole_weight,
        spectral_info,
        weights,
        extrinsic_quasiparticle,
        extrinsic_satellite,
        interference_quasiparticle,
        interference_satellite,
        intrinsic_satellite,
        clipped_extrinsic_satellite,
        energy_grid,
    };
    validate_specfunct_dat(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian `specfunct.dat` bytes.
pub fn specfunct_dat_bytes(data: &SfconvSpecfunctData) -> Result<Vec<u8>> {
    validate_specfunct_dat(data)?;
    let mut bytes = Vec::new();

    let mut header = Vec::with_capacity(HEADER_RECORD_BYTES);
    header.extend_from_slice(&data.wigner_seitz_radius.to_le_bytes());
    header.extend_from_slice(&data.core_hole_lifetime.to_le_bytes());
    header.extend_from_slice(&data.asymmetric_phase.to_le_bytes());
    header.extend_from_slice(&data.satellite_type.to_le_bytes());
    header.extend_from_slice(&data.low_q_mode.to_le_bytes());
    push_i32(&mut header, data.pole_count, "npl")?;
    write_record(&mut bytes, &header)?;

    write_f64_vector_record(&mut bytes, &data.pole_energy)?;
    write_f64_vector_record(&mut bytes, &data.pole_broadening)?;
    write_f64_vector_record(&mut bytes, &data.pole_weight)?;
    write_column_major_matrix_record(&mut bytes, &data.spectral_info)?;
    write_column_major_matrix_record(&mut bytes, &data.weights)?;
    write_column_major_matrix_record(&mut bytes, &data.extrinsic_quasiparticle)?;
    write_column_major_matrix_record(&mut bytes, &data.extrinsic_satellite)?;
    write_column_major_matrix_record(&mut bytes, &data.interference_quasiparticle)?;
    write_column_major_matrix_record(&mut bytes, &data.interference_satellite)?;
    write_column_major_matrix_record(&mut bytes, &data.intrinsic_satellite)?;
    write_column_major_matrix_record(&mut bytes, &data.clipped_extrinsic_satellite)?;
    write_column_major_matrix_record(&mut bytes, &data.energy_grid)?;
    Ok(bytes)
}

/// Read FEFF `specfunct.dat` from a file.
pub fn read_specfunct_dat(path: impl AsRef<Path>) -> Result<SfconvSpecfunctData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_specfunct_dat(&bytes)
}

/// Write FEFF `specfunct.dat` bytes to a file.
pub fn write_specfunct_dat(path: impl AsRef<Path>, data: &SfconvSpecfunctData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, specfunct_dat_bytes(data)?).map_err(|source| IoError::io(path, source))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

fn detect_endian(bytes: &[u8]) -> Result<Endian> {
    let marker = read_marker_bytes(bytes, 0)?;
    let little = u32::from_le_bytes(marker);
    if little == HEADER_RECORD_BYTES as u32 {
        return Ok(Endian::Little);
    }
    let big = u32::from_be_bytes(marker);
    if big == HEADER_RECORD_BYTES as u32 {
        return Ok(Endian::Big);
    }
    invalid_specfunct_dat(format!(
        "first record marker is {little} little-endian/{big} big-endian, expected {HEADER_RECORD_BYTES}"
    ))
}

fn read_record<'a>(
    bytes: &'a [u8],
    position: &mut usize,
    endian: Endian,
    label: &'static str,
) -> Result<&'a [u8]> {
    let length = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    let end = checked_add(*position, length)?;
    let payload = bytes.get(*position..end).ok_or_else(|| {
        invalid_specfunct_dat_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_specfunct_dat(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn parse_f64_vector_record(
    payload: &[u8],
    endian: Endian,
    label: &'static str,
) -> Result<Array1<f64>> {
    if !payload.len().is_multiple_of(F64_BYTES) {
        return invalid_specfunct_dat(format!(
            "{label} record has {} byte(s), not a multiple of {F64_BYTES}",
            payload.len()
        ));
    }

    let mut values = Vec::with_capacity(payload.len() / F64_BYTES);
    for index in 0..payload.len() / F64_BYTES {
        values.push(read_f64(payload, index * F64_BYTES, endian)?);
    }
    Ok(Array1::from_vec(values))
}

fn parse_column_major_matrix_record(
    payload: &[u8],
    rows: usize,
    cols: usize,
    endian: Endian,
    label: &'static str,
) -> Result<Array2<f64>> {
    let expected_values = checked_product(rows, cols)?;
    let expected_bytes = checked_product(expected_values, F64_BYTES)?;
    if payload.len() != expected_bytes {
        return invalid_specfunct_dat(format!(
            "{label} record has {} byte(s), expected {expected_bytes}",
            payload.len()
        ));
    }

    let mut values = Array2::<f64>::zeros((rows, cols));
    for col in 0..cols {
        for row in 0..rows {
            let offset = checked_product(col * rows + row, F64_BYTES)?;
            values[[row, col]] = read_f64(payload, offset, endian)?;
        }
    }
    Ok(values)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_specfunct_dat_value("record length does not fit in u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
}

fn write_f64_vector_record(bytes: &mut Vec<u8>, values: &Array1<f64>) -> Result<()> {
    let mut payload = Vec::with_capacity(checked_product(values.len(), F64_BYTES)?);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    write_record(bytes, &payload)
}

fn write_column_major_matrix_record(bytes: &mut Vec<u8>, values: &Array2<f64>) -> Result<()> {
    let (rows, cols) = values.dim();
    let mut payload = Vec::with_capacity(checked_product(checked_product(rows, cols)?, F64_BYTES)?);
    for col in 0..cols {
        for row in 0..rows {
            payload.extend_from_slice(&values[[row, col]].to_le_bytes());
        }
    }
    write_record(bytes, &payload)
}

fn read_i32(bytes: &[u8], offset: usize, endian: Endian) -> Result<i32> {
    let raw = read_i32_bytes(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(raw),
        Endian::Big => i32::from_be_bytes(raw),
    })
}

fn read_u32(bytes: &[u8], offset: usize, endian: Endian) -> Result<u32> {
    let raw = read_marker_bytes(bytes, offset)?;
    Ok(match endian {
        Endian::Little => u32::from_le_bytes(raw),
        Endian::Big => u32::from_be_bytes(raw),
    })
}

fn read_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<f64> {
    let raw = read_f64_bytes(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f64::from_le_bytes(raw),
        Endian::Big => f64::from_be_bytes(raw),
    })
}

fn read_marker_bytes(bytes: &[u8], offset: usize) -> Result<[u8; FORTRAN_MARKER_BYTES]> {
    let end = checked_add(offset, FORTRAN_MARKER_BYTES)?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_specfunct_dat_value("missing Fortran record marker"))?;
    let mut raw = [0_u8; FORTRAN_MARKER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_i32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; INTEGER_BYTES]> {
    let end = checked_add(offset, INTEGER_BYTES)?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_specfunct_dat_value("missing i32 payload"))?;
    let mut raw = [0_u8; INTEGER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_f64_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F64_BYTES]> {
    let end = checked_add(offset, F64_BYTES)?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_specfunct_dat_value("missing f64 payload"))?;
    let mut raw = [0_u8; F64_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn parse_nonnegative_i32(value: i32, field: &'static str) -> Result<usize> {
    if value < 0 {
        return invalid_specfunct_dat(format!("{field} must be non-negative"));
    }
    usize::try_from(value)
        .map_err(|_| invalid_specfunct_dat_value(format!("{field} does not fit in usize")))
}

fn push_i32(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_specfunct_dat_value(format!("{field} does not fit in i32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn infer_row_count(payload: &[u8], cols: usize) -> Result<usize> {
    let row_bytes = checked_product(cols, F64_BYTES)?;
    if !payload.len().is_multiple_of(row_bytes) {
        return invalid_specfunct_dat(format!(
            "spectral info record has {} byte(s), not a multiple of {row_bytes}",
            payload.len()
        ));
    }
    let rows = payload.len() / row_bytes;
    if rows == 0 {
        return invalid_specfunct_dat("spectral info record must contain at least one row");
    }
    Ok(rows)
}

fn infer_column_count(payload: &[u8], rows: usize) -> Result<usize> {
    let column_bytes = checked_product(rows, F64_BYTES)?;
    if !payload.len().is_multiple_of(column_bytes) {
        return invalid_specfunct_dat(format!(
            "spectral table record has {} byte(s), not a multiple of {column_bytes}",
            payload.len()
        ));
    }
    let cols = payload.len() / column_bytes;
    if cols == 0 {
        return invalid_specfunct_dat("spectral table record must contain at least one column");
    }
    Ok(cols)
}
