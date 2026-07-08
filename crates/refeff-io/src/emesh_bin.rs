//! FEFF `emesh.bin` Fortran-unformatted energy-grid codec.
//!
//! XSPH writes this binary handoff as two sequential unformatted records: the
//! integer grid counts `ne`, `ne1`, and `ne3`, followed by `ne` `complex*16`
//! energy points in Hartree. The parser supports the little-endian records
//! produced by the generated FEFF10 reference suite and also accepts matching
//! big-endian record markers for portability.

use std::path::Path;

use ndarray::Array1;
use num_complex::Complex64;

use crate::PhaseBinData;
use crate::error::{IoError, Result};

const HEADER_RECORD_BYTES: usize = 12;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F64_BYTES: usize = 8;
const COMPLEX64_BYTES: usize = 16;

/// Parsed FEFF `emesh.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct EmeshBinData {
    /// Number of energy points, FEFF `ne`.
    pub point_count_declared: usize,
    /// Number of horizontal-grid points, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Number of DANES extension points, FEFF `ne3`.
    pub danes_extension_count: usize,
    /// Complex energy grid in Hartree, FEFF `em(1:ne)`.
    pub energy_hartree: Array1<Complex64>,
}

impl EmeshBinData {
    /// Number of parsed energy points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.energy_hartree.len()
    }
}

/// Build FEFF `emesh.bin` contents from the phase mesh stored in `phase.bin`.
///
/// FEFF `XSPH/phmesh2.f90` writes `emesh.bin` before `XSPH/wrxsph.f90` writes
/// the same `em(1:ne)` energy grid into `phase.bin`. This helper reconstructs
/// the mesh sidecar from a typed phase cache without rerunning the phase solver.
pub fn emesh_bin_from_phase_bin(phase: &PhaseBinData) -> Result<EmeshBinData> {
    let data = EmeshBinData {
        point_count_declared: phase.energy_count,
        horizontal_count: phase.main_energy_count,
        danes_extension_count: phase.auxiliary_energy_count,
        energy_hartree: phase.energy_grid.clone(),
    };
    validate_emesh_bin(&data)?;
    Ok(data)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF `emesh.bin` bytes.
pub fn parse_emesh_bin(bytes: &[u8]) -> Result<EmeshBinData> {
    let endian = detect_endian(bytes)?;
    let mut position = 0;
    let header = read_record(bytes, &mut position, endian, "header")?;
    if header.len() != HEADER_RECORD_BYTES {
        return invalid_emesh_bin(format!(
            "header record has {} byte(s), expected {HEADER_RECORD_BYTES}",
            header.len()
        ));
    }

    let ne = parse_nonnegative_i32(read_i32(header, 0, endian)?, "ne")?;
    let ne1 = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES, endian)?, "ne1")?;
    let ne3 = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES * 2, endian)?, "ne3")?;

    let payload = read_record(bytes, &mut position, endian, "energy grid")?;
    let expected_payload = checked_product(ne, COMPLEX64_BYTES)?;
    if payload.len() != expected_payload {
        return invalid_emesh_bin(format!(
            "energy record has {} byte(s), expected {expected_payload}",
            payload.len()
        ));
    }

    let mut values = Vec::with_capacity(ne);
    for index in 0..ne {
        let offset = checked_product(index, COMPLEX64_BYTES)?;
        let real = read_f64(payload, offset, endian)?;
        let imaginary = read_f64(payload, offset + F64_BYTES, endian)?;
        if !(real.is_finite() && imaginary.is_finite()) {
            return invalid_emesh_bin(format!("energy point {} is not finite", index + 1));
        }
        values.push(Complex64::new(real, imaginary));
    }

    if position != bytes.len() {
        return invalid_emesh_bin(format!(
            "emesh.bin has {} trailing byte(s)",
            bytes.len() - position
        ));
    }

    let data = EmeshBinData {
        point_count_declared: ne,
        horizontal_count: ne1,
        danes_extension_count: ne3,
        energy_hartree: Array1::from_vec(values),
    };
    validate_emesh_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian `emesh.bin` bytes.
pub fn emesh_bin_bytes(data: &EmeshBinData) -> Result<Vec<u8>> {
    validate_emesh_bin(data)?;
    let mut bytes = Vec::new();

    let mut header = Vec::with_capacity(HEADER_RECORD_BYTES);
    push_i32(&mut header, data.point_count_declared, "ne")?;
    push_i32(&mut header, data.horizontal_count, "ne1")?;
    push_i32(&mut header, data.danes_extension_count, "ne3")?;
    write_record(&mut bytes, &header)?;

    let mut payload = Vec::with_capacity(checked_product(data.point_count(), COMPLEX64_BYTES)?);
    for value in &data.energy_hartree {
        payload.extend_from_slice(&value.re.to_le_bytes());
        payload.extend_from_slice(&value.im.to_le_bytes());
    }
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `emesh.bin` from a file.
pub fn read_emesh_bin(path: impl AsRef<Path>) -> Result<EmeshBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_emesh_bin(&bytes)
}

/// Write FEFF `emesh.bin` bytes to a file.
pub fn write_emesh_bin(path: impl AsRef<Path>, data: &EmeshBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, emesh_bin_bytes(data)?).map_err(|source| IoError::io(path, source))
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
    invalid_emesh_bin(format!(
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
        invalid_emesh_bin_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_emesh_bin(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_emesh_bin_value("record length does not fit in u32"))?;
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
        .ok_or_else(|| invalid_emesh_bin_value("missing Fortran record marker"))?;
    let mut raw = [0_u8; FORTRAN_MARKER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_i32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; INTEGER_BYTES]> {
    let slice = bytes
        .get(offset..offset + INTEGER_BYTES)
        .ok_or_else(|| invalid_emesh_bin_value("missing i32 payload"))?;
    let mut raw = [0_u8; INTEGER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_f64_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F64_BYTES]> {
    let slice = bytes
        .get(offset..offset + F64_BYTES)
        .ok_or_else(|| invalid_emesh_bin_value("missing f64 payload"))?;
    let mut raw = [0_u8; F64_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn parse_nonnegative_i32(value: i32, field: &'static str) -> Result<usize> {
    if value < 0 {
        return invalid_emesh_bin(format!("{field} must be non-negative"));
    }
    usize::try_from(value)
        .map_err(|_| invalid_emesh_bin_value(format!("{field} does not fit in usize")))
}

fn push_i32(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_emesh_bin_value(format!("{field} does not fit in i32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn validate_emesh_bin(data: &EmeshBinData) -> Result<()> {
    if data.point_count_declared == 0 {
        return invalid_emesh_bin("ne must be positive");
    }
    if data.point_count_declared != data.point_count() {
        return invalid_emesh_bin(format!(
            "ne is {}, but {} energy point(s) were supplied",
            data.point_count_declared,
            data.point_count()
        ));
    }
    if data.horizontal_count > data.point_count_declared {
        return invalid_emesh_bin("ne1 must not exceed ne");
    }
    if data.danes_extension_count > data.point_count_declared {
        return invalid_emesh_bin("ne3 must not exceed ne");
    }
    for (index, value) in data.energy_hartree.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_emesh_bin(format!("energy point {} is not finite", index + 1));
        }
    }
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_emesh_bin_value("byte offset overflows usize"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_emesh_bin_value("record length overflows usize"))
}

fn invalid_emesh_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_emesh_bin_value(message))
}

fn invalid_emesh_bin_value(message: impl Into<String>) -> IoError {
    IoError::InvalidEmeshBin {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_emesh_bin_bytes() -> Result<()> {
        let data = EmeshBinData {
            point_count_declared: 2,
            horizontal_count: 1,
            danes_extension_count: 0,
            energy_hartree: Array1::from_vec(vec![
                Complex64::new(-1.25, 0.05),
                Complex64::new(2.5, -0.125),
            ]),
        };
        let bytes = emesh_bin_bytes(&data)?;
        let parsed = parse_emesh_bin(&bytes)?;
        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn builds_emesh_bin_from_phase_cache() -> Result<()> {
        let phase = PhaseBinData {
            spin_count: 1,
            energy_count: 2,
            main_energy_count: 1,
            auxiliary_energy_count: 1,
            ihole: 1,
            fermi_index: 1,
            pad_width: 8,
            final_state_count: 1,
            transition_count: 1,
            q_count: 1,
            scalars: crate::PhaseBinScalars {
                average_norman_radius: 1.0,
                fermi_level: 0.0,
                edge_energy: 0.25,
            },
            energy_grid: Array1::from_vec(vec![
                Complex64::new(-0.5, 0.02),
                Complex64::new(0.5, 0.03),
            ]),
            reference_energy: ndarray::Array2::zeros((2, 1)),
            potentials: vec![crate::PhaseBinPotential {
                lmax: 0,
                atomic_number: 29,
                label: "Cu".to_string(),
                phase_shifts: ndarray::Array3::zeros((2, 1, 1)),
            }],
            transition_moments: ndarray::Array4::zeros((2, 1, 1, 1)),
            raw_pads: None,
        };

        let emesh = emesh_bin_from_phase_bin(&phase)?;

        assert_eq!(emesh.point_count_declared, 2);
        assert_eq!(emesh.horizontal_count, 1);
        assert_eq!(emesh.danes_extension_count, 1);
        assert_eq!(emesh.energy_hartree, phase.energy_grid);
        Ok(())
    }

    #[test]
    fn rejects_invalid_emesh_bin_bytes() {
        assert!(parse_emesh_bin(&[]).is_err());
        assert!(parse_emesh_bin(&[12, 0, 0, 0]).is_err());

        let bad = EmeshBinData {
            point_count_declared: 2,
            horizontal_count: 1,
            danes_extension_count: 0,
            energy_hartree: Array1::from_vec(vec![Complex64::new(0.0, 0.0)]),
        };
        assert!(emesh_bin_bytes(&bad).is_err());
    }
}
