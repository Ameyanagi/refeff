//! FEFF FF2X `chia.bin` configuration-average handoff codec.
//!
//! FF2X writes one Fortran sequential unformatted record per `COMPLEX*16`
//! value while accumulating spectra across multiple absorbers.

use std::path::Path;

use num_complex::Complex64;

use crate::error::{IoError, Result};

const FORTRAN_MARKER_BYTES: usize = 4;
const F64_BYTES: usize = 8;
const COMPLEX64_BYTES: usize = 16;

/// Parsed FEFF `chia.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiaBinData {
    /// Complex configuration-average accumulator values in FEFF write order.
    pub values: Vec<Complex64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF `chia.bin` bytes.
pub fn parse_chia_bin(bytes: &[u8]) -> Result<ChiaBinData> {
    let endian = detect_endian(bytes)?;
    let mut position = 0;
    let mut values = Vec::new();

    while position < bytes.len() {
        let payload = read_record(bytes, &mut position, endian, "configuration average")?;
        if payload.len() != COMPLEX64_BYTES {
            return invalid_chia_bin(format!(
                "configuration-average record has {} byte(s), expected {COMPLEX64_BYTES}",
                payload.len()
            ));
        }
        let real = read_f64(payload, 0, endian)?;
        let imaginary = read_f64(payload, F64_BYTES, endian)?;
        if !(real.is_finite() && imaginary.is_finite()) {
            return invalid_chia_bin(format!(
                "configuration-average value {} is not finite",
                values.len() + 1
            ));
        }
        values.push(Complex64::new(real, imaginary));
    }

    if values.is_empty() {
        return invalid_chia_bin("chia.bin must contain at least one complex value");
    }
    Ok(ChiaBinData { values })
}

/// Render FEFF-compatible little-endian `chia.bin` bytes.
pub fn chia_bin_bytes(data: &ChiaBinData) -> Result<Vec<u8>> {
    validate_chia_bin(data)?;

    let mut bytes = Vec::new();
    for value in &data.values {
        let mut payload = Vec::with_capacity(COMPLEX64_BYTES);
        payload.extend_from_slice(&value.re.to_le_bytes());
        payload.extend_from_slice(&value.im.to_le_bytes());
        write_record(&mut bytes, &payload)?;
    }
    Ok(bytes)
}

/// Read FEFF `chia.bin` from a file.
pub fn read_chia_bin(path: impl AsRef<Path>) -> Result<ChiaBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_chia_bin(&bytes)
}

/// Write FEFF `chia.bin` bytes to a file.
pub fn write_chia_bin(path: impl AsRef<Path>, data: &ChiaBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, chia_bin_bytes(data)?).map_err(|source| IoError::io(path, source))
}

fn validate_chia_bin(data: &ChiaBinData) -> Result<()> {
    if data.values.is_empty() {
        return invalid_chia_bin("chia.bin must contain at least one complex value");
    }
    for (index, value) in data.values.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_chia_bin(format!(
                "configuration-average value {} is not finite",
                index + 1
            ));
        }
    }
    Ok(())
}

fn detect_endian(bytes: &[u8]) -> Result<Endian> {
    let marker = read_marker_bytes(bytes, 0)?;
    let little = u32::from_le_bytes(marker);
    if little == COMPLEX64_BYTES as u32 {
        return Ok(Endian::Little);
    }
    let big = u32::from_be_bytes(marker);
    if big == COMPLEX64_BYTES as u32 {
        return Ok(Endian::Big);
    }
    invalid_chia_bin(format!(
        "first record marker is {little} little-endian/{big} big-endian, expected {COMPLEX64_BYTES}"
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
        invalid_chia_bin_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_chia_bin(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_chia_bin_value("record length does not fit in u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
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
        .ok_or_else(|| invalid_chia_bin_value("missing Fortran record marker"))?;
    let mut raw = [0_u8; FORTRAN_MARKER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_f64_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F64_BYTES]> {
    let slice = bytes
        .get(offset..offset + F64_BYTES)
        .ok_or_else(|| invalid_chia_bin_value("missing f64 payload"))?;
    let mut raw = [0_u8; F64_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_chia_bin_value("record offset overflows usize"))
}

fn invalid_chia_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_chia_bin_value(message))
}

fn invalid_chia_bin_value(message: impl Into<String>) -> IoError {
    IoError::InvalidChiaBin {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ChiaBinData, chia_bin_bytes, parse_chia_bin};
    use num_complex::Complex64;

    #[test]
    fn roundtrips_chia_bin_records() -> crate::Result<()> {
        let data = ChiaBinData {
            values: vec![Complex64::new(1.25, -0.5), Complex64::new(0.0, 2.75)],
        };

        let parsed = parse_chia_bin(&chia_bin_bytes(&data)?)?;

        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn parses_big_endian_chia_bin_record() -> crate::Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(16_u32).to_be_bytes());
        bytes.extend_from_slice(&1.5_f64.to_be_bytes());
        bytes.extend_from_slice(&(-2.25_f64).to_be_bytes());
        bytes.extend_from_slice(&(16_u32).to_be_bytes());

        let parsed = parse_chia_bin(&bytes)?;

        assert_eq!(parsed.values, vec![Complex64::new(1.5, -2.25)]);
        Ok(())
    }

    #[test]
    fn rejects_bad_record_shape() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(8_u32).to_le_bytes());
        bytes.extend_from_slice(&1.0_f64.to_le_bytes());
        bytes.extend_from_slice(&(8_u32).to_le_bytes());

        let error = parse_chia_bin(&bytes).expect_err("short record should fail");

        assert!(error.to_string().contains("expected 16"), "{error}");
    }
}
