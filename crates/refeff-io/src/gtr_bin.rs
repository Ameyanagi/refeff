//! FEFF `gtrNN.bin` FMS-to-LDOS binary handoff codec.
//!
//! LDOS writes `gtrNN.bin` from `LDOS/fmsdos.f90` as Fortran sequential
//! unformatted records. The first record stores `ne`, `ne1`, `ne3`, `nph`, and
//! `ifms` as 32-bit integers. The second record stores the complex Green's
//! function trace values in FEFF loop order: energy, potential, angular
//! channel.

use std::path::Path;

use ndarray::{Array3, Axis};
use num_complex::Complex64;

use crate::error::{IoError, Result};

const HEADER_RECORD_BYTES: usize = 20;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F64_BYTES: usize = 8;
const COMPLEX64_BYTES: usize = 16;

/// Parsed FEFF `gtrNN.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct GtrBinData {
    /// Number of energy points, FEFF `ne`.
    pub point_count_declared: usize,
    /// Number of horizontal-grid points, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Number of DANES extension points, FEFF `ne3`.
    pub danes_extension_count: usize,
    /// Highest unique potential index, FEFF `nph`.
    pub highest_potential_index: usize,
    /// FEFF FMS selector, `ifms`.
    pub fms_mode: i32,
    /// Complex trace values as `(energy, potential, angular_channel)`.
    pub values: Array3<Complex64>,
}

impl GtrBinData {
    /// Number of parsed energy points.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of potential columns, equal to `nph + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of angular channels stored for each energy and potential.
    #[must_use]
    pub fn angular_channel_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF `gtrNN.bin` bytes.
pub fn parse_gtr_bin(bytes: &[u8]) -> Result<GtrBinData> {
    let endian = detect_endian(bytes)?;
    let mut position = 0;
    let header = read_record(bytes, &mut position, endian, "header")?;
    if header.len() != HEADER_RECORD_BYTES {
        return invalid_gtr_bin(format!(
            "header record has {} byte(s), expected {HEADER_RECORD_BYTES}",
            header.len()
        ));
    }

    let ne = parse_nonnegative_i32(read_i32(header, 0, endian)?, "ne")?;
    let ne1 = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES, endian)?, "ne1")?;
    let ne3 = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES * 2, endian)?, "ne3")?;
    let nph = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES * 3, endian)?, "nph")?;
    let ifms = read_i32(header, INTEGER_BYTES * 4, endian)?;
    let potential_count = checked_add(nph, 1)?;

    let payload = read_record(bytes, &mut position, endian, "Green's-function trace")?;
    if payload.len() % COMPLEX64_BYTES != 0 {
        return invalid_gtr_bin(format!(
            "trace payload has {} byte(s), not a multiple of {COMPLEX64_BYTES}",
            payload.len()
        ));
    }
    let complex_count = payload.len() / COMPLEX64_BYTES;
    let energy_potential_count = checked_product(ne, potential_count)?;
    if energy_potential_count == 0 {
        return invalid_gtr_bin("ne and potential count must be positive");
    }
    if !complex_count.is_multiple_of(energy_potential_count) {
        return invalid_gtr_bin(format!(
            "trace payload has {complex_count} complex value(s), not a whole number of angular channels for ne={ne}, nph={nph}"
        ));
    }
    let angular_channel_count = complex_count / energy_potential_count;
    if angular_channel_count == 0 {
        return invalid_gtr_bin("trace payload must include at least one angular channel");
    }

    let mut values = Vec::with_capacity(complex_count);
    for index in 0..complex_count {
        let offset = checked_product(index, COMPLEX64_BYTES)?;
        let real = read_f64(payload, offset, endian)?;
        let imaginary = read_f64(payload, offset + F64_BYTES, endian)?;
        if !(real.is_finite() && imaginary.is_finite()) {
            return invalid_gtr_bin(format!("trace value {} is not finite", index + 1));
        }
        values.push(Complex64::new(real, imaginary));
    }

    if position != bytes.len() {
        return invalid_gtr_bin(format!(
            "gtrNN.bin has {} trailing byte(s)",
            bytes.len() - position
        ));
    }

    let values = Array3::from_shape_vec((ne, potential_count, angular_channel_count), values)
        .map_err(|source| invalid_gtr_bin_value(format!("invalid trace shape: {source}")))?;
    let data = GtrBinData {
        point_count_declared: ne,
        horizontal_count: ne1,
        danes_extension_count: ne3,
        highest_potential_index: nph,
        fms_mode: ifms,
        values,
    };
    validate_gtr_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian `gtrNN.bin` bytes.
pub fn gtr_bin_bytes(data: &GtrBinData) -> Result<Vec<u8>> {
    validate_gtr_bin(data)?;
    let mut bytes = Vec::new();

    let mut header = Vec::with_capacity(HEADER_RECORD_BYTES);
    push_i32(&mut header, data.point_count_declared, "ne")?;
    push_i32(&mut header, data.horizontal_count, "ne1")?;
    push_i32(&mut header, data.danes_extension_count, "ne3")?;
    push_i32(&mut header, data.highest_potential_index, "nph")?;
    header.extend_from_slice(&data.fms_mode.to_le_bytes());
    write_record(&mut bytes, &header)?;

    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX64_BYTES)?);
    for value in &data.values {
        payload.extend_from_slice(&value.re.to_le_bytes());
        payload.extend_from_slice(&value.im.to_le_bytes());
    }
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `gtrNN.bin` from a file.
pub fn read_gtr_bin(path: impl AsRef<Path>) -> Result<GtrBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_gtr_bin(&bytes)
}

/// Write FEFF `gtrNN.bin` bytes to a file.
pub fn write_gtr_bin(path: impl AsRef<Path>, data: &GtrBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, gtr_bin_bytes(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_gtr_bin(data: &GtrBinData) -> Result<()> {
    if data.point_count_declared == 0 {
        return invalid_gtr_bin("ne must be positive");
    }
    if data.point_count_declared != data.energy_count() {
        return invalid_gtr_bin(format!(
            "ne is {}, but {} energy point(s) were supplied",
            data.point_count_declared,
            data.energy_count()
        ));
    }
    let expected_potential_count = checked_add(data.highest_potential_index, 1)?;
    if data.potential_count() != expected_potential_count {
        return invalid_gtr_bin(format!(
            "nph is {}, but {} potential column(s) were supplied",
            data.highest_potential_index,
            data.potential_count()
        ));
    }
    if data.angular_channel_count() == 0 {
        return invalid_gtr_bin("at least one angular channel is required");
    }
    if data.horizontal_count > data.point_count_declared {
        return invalid_gtr_bin(format!(
            "ne1 {} exceeds ne {}",
            data.horizontal_count, data.point_count_declared
        ));
    }
    ensure_i32("ne", data.point_count_declared)?;
    ensure_i32("ne1", data.horizontal_count)?;
    ensure_i32("ne3", data.danes_extension_count)?;
    ensure_i32("nph", data.highest_potential_index)?;

    for (index, value) in data.values.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_gtr_bin(format!("trace value {} is not finite", index + 1));
        }
    }
    Ok(())
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
    invalid_gtr_bin(format!(
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
        invalid_gtr_bin_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_gtr_bin(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_gtr_bin_value("record length does not fit in u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
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
    let slice = bytes
        .get(offset..offset + FORTRAN_MARKER_BYTES)
        .ok_or_else(|| invalid_gtr_bin_value("missing Fortran record marker"))?;
    let mut raw = [0_u8; FORTRAN_MARKER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_i32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; INTEGER_BYTES]> {
    let slice = bytes
        .get(offset..offset + INTEGER_BYTES)
        .ok_or_else(|| invalid_gtr_bin_value("missing i32 payload"))?;
    let mut raw = [0_u8; INTEGER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_f64_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F64_BYTES]> {
    let slice = bytes
        .get(offset..offset + F64_BYTES)
        .ok_or_else(|| invalid_gtr_bin_value("missing f64 payload"))?;
    let mut raw = [0_u8; F64_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn parse_nonnegative_i32(value: i32, field: &'static str) -> Result<usize> {
    if value < 0 {
        return invalid_gtr_bin(format!("{field} must be non-negative"));
    }
    usize::try_from(value).map_err(|_| invalid_gtr_bin_value(format!("{field} does not fit usize")))
}

fn push_i32(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_gtr_bin_value(format!("{field} does not fit in i32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn ensure_i32(field: &'static str, value: usize) -> Result<()> {
    i32::try_from(value)
        .map(|_| ())
        .map_err(|_| invalid_gtr_bin_value(format!("{field} does not fit in i32")))
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_gtr_bin_value("integer overflow while adding dimensions"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        invalid_gtr_bin_value(format!(
            "integer overflow while multiplying {left} by {right}"
        ))
    })
}

fn invalid_gtr_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_gtr_bin_value(message))
}

fn invalid_gtr_bin_value(message: impl Into<String>) -> IoError {
    IoError::InvalidGtrBin {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_gtr_bin_bytes() -> Result<()> {
        let data = sample_gtr_bin();
        let bytes = gtr_bin_bytes(&data)?;
        let parsed = parse_gtr_bin(&bytes)?;
        assert_eq!(parsed, data);
        assert_eq!(parsed.energy_count(), 2);
        assert_eq!(parsed.potential_count(), 2);
        assert_eq!(parsed.angular_channel_count(), 2);
        Ok(())
    }

    #[test]
    fn parses_fortran_record_shape() -> Result<()> {
        let parsed = parse_gtr_bin(&gtr_bin_bytes(&sample_gtr_bin())?)?;
        assert_eq!(parsed.point_count_declared, 2);
        assert_eq!(parsed.horizontal_count, 1);
        assert_eq!(parsed.danes_extension_count, 0);
        assert_eq!(parsed.highest_potential_index, 1);
        assert_eq!(parsed.fms_mode, 2);
        assert_eq!(parsed.values[(1, 1, 1)], Complex64::new(1.11, -1.11));
        Ok(())
    }

    #[test]
    fn rejects_invalid_gtr_bin_data() -> Result<()> {
        assert!(parse_gtr_bin(&[]).is_err());

        let mut bad_marker = gtr_bin_bytes(&sample_gtr_bin())?;
        bad_marker[0] = 0;
        assert!(parse_gtr_bin(&bad_marker).is_err());

        let bad_shape = GtrBinData {
            point_count_declared: 3,
            ..sample_gtr_bin()
        };
        assert!(gtr_bin_bytes(&bad_shape).is_err());

        let bad_value = GtrBinData {
            values: Array3::from_elem((1, 1, 1), Complex64::new(f64::NAN, 0.0)),
            ..sample_gtr_bin()
        };
        assert!(gtr_bin_bytes(&bad_value).is_err());
        Ok(())
    }

    fn sample_gtr_bin() -> GtrBinData {
        GtrBinData {
            point_count_declared: 2,
            horizontal_count: 1,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            values: Array3::from_shape_fn((2, 2, 2), |(energy, potential, angular)| {
                let value = energy as f64 + 0.1 * potential as f64 + 0.01 * angular as f64;
                Complex64::new(value, -value)
            }),
        }
    }
}
