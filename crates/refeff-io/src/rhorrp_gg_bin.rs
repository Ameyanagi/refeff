//! FEFF `gg_slice.bin` and `gg_diag.bin` RHORRP FMS handoff codecs.
//!
//! `FMS/fmstot.f90` writes both files as sequential Fortran-unformatted
//! records. The headers are 32-bit integer dimensions, followed by default
//! Fortran `complex` payloads, i.e. pairs of 32-bit floating point values. The
//! Rust structs expose energy-first `ndarray` layouts while the byte codec
//! preserves FEFF's column-major array order on disk.

use std::path::Path;

use ndarray::{Array3, Array4, Axis};
use num_complex::{Complex32, Complex64};

use crate::error::{IoError, Result};

const SLICE_HEADER_RECORD_BYTES: usize = 12;
const DIAG_HEADER_RECORD_BYTES: usize = 16;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F32_BYTES: usize = 4;
const COMPLEX32_BYTES: usize = 8;

/// Parsed FEFF `gg_slice.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpGgSliceBinData {
    /// Scattering matrix slice as `(energy, row, column)`.
    pub values: Array3<Complex32>,
}

impl RhorrpGgSliceBinData {
    /// Number of energy points, FEFF `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of row states, FEFF `ldim`.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of column states, FEFF `istate`.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }
}

/// Parsed FEFF `gg_diag.bin` contents.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpGgDiagBinData {
    /// Site-diagonal scattering matrices as `(energy, atom, row, column)`.
    pub values: Array4<Complex32>,
}

/// FEFF `rhoerrp` scattering matrix selection for one point pair.
#[derive(Debug, Clone, PartialEq)]
pub enum RhorrpGgPairMatrix {
    /// The selected matrix is available as `(energy, L, L')`.
    Available(Array3<Complex64>),
    /// FEFF only writes off-diagonal `gg_slice` blocks for `r` near atom 1.
    UnsupportedOffCentralFirstAtom,
}

impl RhorrpGgDiagBinData {
    /// Number of energy points, FEFF `ne`.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of atoms in the FMS inclusion, FEFF `inclus`.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of row states, FEFF `ldim`.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of column states, FEFF second `ldim`.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF `gg_slice.bin` bytes.
pub fn parse_rhorrp_gg_slice_bin(bytes: &[u8]) -> Result<RhorrpGgSliceBinData> {
    let endian = detect_endian(bytes, SLICE_HEADER_RECORD_BYTES)?;
    let mut position = 0;
    let header = read_record(bytes, &mut position, endian, "gg_slice header")?;
    if header.len() != SLICE_HEADER_RECORD_BYTES {
        return invalid_rhorrp_gg_bin(format!(
            "gg_slice header record has {} byte(s), expected {SLICE_HEADER_RECORD_BYTES}",
            header.len()
        ));
    }

    let row_count = parse_positive_i32(read_i32(header, 0, endian)?, "ldim")?;
    let column_count = parse_positive_i32(read_i32(header, INTEGER_BYTES, endian)?, "istate")?;
    let energy_count = parse_positive_i32(read_i32(header, INTEGER_BYTES * 2, endian)?, "ne")?;
    let complex_count = checked_product(checked_product(row_count, column_count)?, energy_count)?;
    let payload = read_record(bytes, &mut position, endian, "gg_slice payload")?;
    let raw_values = parse_complex32_payload(payload, endian, complex_count, "gg_slice")?;
    ensure_consumed(bytes, position, "gg_slice.bin")?;

    let mut values = Array3::from_elem(
        (energy_count, row_count, column_count),
        Complex32::new(0.0, 0.0),
    );
    let mut source_index = 0;
    for energy in 0..energy_count {
        for column in 0..column_count {
            for row in 0..row_count {
                values[(energy, row, column)] = raw_values[source_index];
                source_index += 1;
            }
        }
    }

    let data = RhorrpGgSliceBinData { values };
    validate_rhorrp_gg_slice_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian `gg_slice.bin` bytes.
pub fn rhorrp_gg_slice_bin_bytes(data: &RhorrpGgSliceBinData) -> Result<Vec<u8>> {
    validate_rhorrp_gg_slice_bin(data)?;
    let mut bytes = Vec::new();

    let mut header = Vec::with_capacity(SLICE_HEADER_RECORD_BYTES);
    push_i32(&mut header, data.row_count(), "ldim")?;
    push_i32(&mut header, data.column_count(), "istate")?;
    push_i32(&mut header, data.energy_count(), "ne")?;
    write_record(&mut bytes, &header)?;

    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX32_BYTES)?);
    for energy in 0..data.energy_count() {
        for column in 0..data.column_count() {
            for row in 0..data.row_count() {
                push_complex32(&mut payload, data.values[(energy, row, column)]);
            }
        }
    }
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `gg_slice.bin` from a file.
pub fn read_rhorrp_gg_slice_bin(path: impl AsRef<Path>) -> Result<RhorrpGgSliceBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_rhorrp_gg_slice_bin(&bytes)
}

/// Write FEFF `gg_slice.bin` bytes to a file.
pub fn write_rhorrp_gg_slice_bin(
    path: impl AsRef<Path>,
    data: &RhorrpGgSliceBinData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhorrp_gg_slice_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `gg_diag.bin` bytes.
pub fn parse_rhorrp_gg_diag_bin(bytes: &[u8]) -> Result<RhorrpGgDiagBinData> {
    let endian = detect_endian(bytes, DIAG_HEADER_RECORD_BYTES)?;
    let mut position = 0;
    let header = read_record(bytes, &mut position, endian, "gg_diag header")?;
    if header.len() != DIAG_HEADER_RECORD_BYTES {
        return invalid_rhorrp_gg_bin(format!(
            "gg_diag header record has {} byte(s), expected {DIAG_HEADER_RECORD_BYTES}",
            header.len()
        ));
    }

    let row_count = parse_positive_i32(read_i32(header, 0, endian)?, "ldim")?;
    let column_count = parse_positive_i32(read_i32(header, INTEGER_BYTES, endian)?, "ldim")?;
    let atom_count = parse_positive_i32(read_i32(header, INTEGER_BYTES * 2, endian)?, "inclus")?;
    let energy_count = parse_positive_i32(read_i32(header, INTEGER_BYTES * 3, endian)?, "ne")?;
    if row_count != column_count {
        return invalid_rhorrp_gg_bin(format!(
            "gg_diag ldim values differ: {row_count} and {column_count}"
        ));
    }
    let complex_count = checked_product(
        checked_product(checked_product(row_count, column_count)?, atom_count)?,
        energy_count,
    )?;
    let payload = read_record(bytes, &mut position, endian, "gg_diag payload")?;
    let raw_values = parse_complex32_payload(payload, endian, complex_count, "gg_diag")?;
    ensure_consumed(bytes, position, "gg_diag.bin")?;

    let mut values = Array4::from_elem(
        (energy_count, atom_count, row_count, column_count),
        Complex32::new(0.0, 0.0),
    );
    let mut source_index = 0;
    for energy in 0..energy_count {
        for atom in 0..atom_count {
            for column in 0..column_count {
                for row in 0..row_count {
                    values[(energy, atom, row, column)] = raw_values[source_index];
                    source_index += 1;
                }
            }
        }
    }

    let data = RhorrpGgDiagBinData { values };
    validate_rhorrp_gg_diag_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian `gg_diag.bin` bytes.
pub fn rhorrp_gg_diag_bin_bytes(data: &RhorrpGgDiagBinData) -> Result<Vec<u8>> {
    validate_rhorrp_gg_diag_bin(data)?;
    let mut bytes = Vec::new();

    let mut header = Vec::with_capacity(DIAG_HEADER_RECORD_BYTES);
    push_i32(&mut header, data.row_count(), "ldim")?;
    push_i32(&mut header, data.column_count(), "ldim")?;
    push_i32(&mut header, data.atom_count(), "inclus")?;
    push_i32(&mut header, data.energy_count(), "ne")?;
    write_record(&mut bytes, &header)?;

    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX32_BYTES)?);
    for energy in 0..data.energy_count() {
        for atom in 0..data.atom_count() {
            for column in 0..data.column_count() {
                for row in 0..data.row_count() {
                    push_complex32(&mut payload, data.values[(energy, atom, row, column)]);
                }
            }
        }
    }
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `gg_diag.bin` from a file.
pub fn read_rhorrp_gg_diag_bin(path: impl AsRef<Path>) -> Result<RhorrpGgDiagBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_rhorrp_gg_diag_bin(&bytes)
}

/// Write FEFF `gg_diag.bin` bytes to a file.
pub fn write_rhorrp_gg_diag_bin(path: impl AsRef<Path>, data: &RhorrpGgDiagBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhorrp_gg_diag_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Extract FEFF `gg_diag(:,:,iat,:)` data as `(energy, row, column)`.
///
/// FEFF addresses atoms with 1-based indices. The returned matrix is promoted
/// to `Complex64` so it can be passed directly to RHORRP core routines.
pub fn rhorrp_gg_diag_matrix(
    data: &RhorrpGgDiagBinData,
    atom_index_1based: usize,
) -> Result<Array3<Complex64>> {
    validate_rhorrp_gg_diag_shape(data)?;
    let atom = one_based_index(atom_index_1based, data.atom_count(), "atom")?;
    let energy_count = data.energy_count();
    let row_count = data.row_count();
    let column_count = data.column_count();
    let mut values = Vec::with_capacity(checked_product(
        energy_count,
        checked_product(row_count, column_count)?,
    )?);

    for energy in 0..energy_count {
        for row in 0..row_count {
            for column in 0..column_count {
                let value = data
                    .values
                    .get((energy, atom, row, column))
                    .ok_or_else(|| {
                        invalid_rhorrp_gg_bin_value(format!(
                            "gg_diag index ({energy}, {atom}, {row}, {column}) is outside data"
                        ))
                    })?;
                values.push(complex32_to_complex64(ensure_finite_complex32(
                    *value, "gg_diag", energy, row, column,
                )?));
            }
        }
    }

    Array3::from_shape_vec((energy_count, row_count, column_count), values)
        .map_err(|err| invalid_rhorrp_gg_bin_value(format!("invalid gg_diag matrix shape: {err}")))
}

/// Extract one FEFF `gg_slice` atom block as `(energy, row, column)`.
///
/// `row_atom_index_1based` and `column_atom_index_1based` follow FEFF's
/// 1-based atom numbering. `block_dimension` is the per-atom angular-state
/// dimension used when FEFF flattens atom blocks into `gg_slice`.
pub fn rhorrp_gg_slice_block(
    data: &RhorrpGgSliceBinData,
    row_atom_index_1based: usize,
    column_atom_index_1based: usize,
    block_dimension: usize,
) -> Result<Array3<Complex64>> {
    validate_rhorrp_gg_slice_shape(data)?;
    validate_positive("block_dimension", block_dimension)?;
    let row_start = block_start(row_atom_index_1based, block_dimension, "row atom")?;
    let column_start = block_start(column_atom_index_1based, block_dimension, "column atom")?;
    let row_end = checked_add(row_start, block_dimension)?;
    let column_end = checked_add(column_start, block_dimension)?;
    if row_end > data.row_count() {
        return invalid_rhorrp_gg_bin(format!(
            "row atom block {row_atom_index_1based} spans states {}..{}, but gg_slice has {} rows",
            row_start + 1,
            row_end,
            data.row_count()
        ));
    }
    if column_end > data.column_count() {
        return invalid_rhorrp_gg_bin(format!(
            "column atom block {column_atom_index_1based} spans states {}..{}, but gg_slice has {} columns",
            column_start + 1,
            column_end,
            data.column_count()
        ));
    }

    let energy_count = data.energy_count();
    let mut values = Vec::with_capacity(checked_product(
        energy_count,
        checked_product(block_dimension, block_dimension)?,
    )?);
    for energy in 0..energy_count {
        for row_offset in 0..block_dimension {
            for column_offset in 0..block_dimension {
                let row = checked_add(row_start, row_offset)?;
                let column = checked_add(column_start, column_offset)?;
                let value = data.values.get((energy, row, column)).ok_or_else(|| {
                    invalid_rhorrp_gg_bin_value(format!(
                        "gg_slice index ({energy}, {row}, {column}) is outside data"
                    ))
                })?;
                values.push(complex32_to_complex64(ensure_finite_complex32(
                    *value, "gg_slice", energy, row, column,
                )?));
            }
        }
    }

    Array3::from_shape_vec((energy_count, block_dimension, block_dimension), values)
        .map_err(|err| invalid_rhorrp_gg_bin_value(format!("invalid gg_slice block shape: {err}")))
}

/// Select the FEFF RHORRP scattering matrix for one pair of nearest atoms.
///
/// This mirrors `rhoerrp`: same-site pairs read `gg_diag(:,:,iat,:)`, pairs
/// with `r` near atom 1 read the saved `gg_slice` block, and different-site
/// pairs with `r` away from atom 1 are unavailable because FEFF does not write
/// the full FMS matrix for RHORRP.
pub fn rhorrp_gg_pair_matrix(
    diag: &RhorrpGgDiagBinData,
    slice: &RhorrpGgSliceBinData,
    first_atom_index_1based: usize,
    second_atom_index_1based: usize,
    block_dimension: usize,
) -> Result<RhorrpGgPairMatrix> {
    validate_positive("first_atom_index", first_atom_index_1based)?;
    validate_positive("second_atom_index", second_atom_index_1based)?;

    if first_atom_index_1based == second_atom_index_1based {
        return Ok(RhorrpGgPairMatrix::Available(rhorrp_gg_diag_matrix(
            diag,
            first_atom_index_1based,
        )?));
    }
    if first_atom_index_1based != 1 {
        return Ok(RhorrpGgPairMatrix::UnsupportedOffCentralFirstAtom);
    }

    Ok(RhorrpGgPairMatrix::Available(rhorrp_gg_slice_block(
        slice,
        first_atom_index_1based,
        second_atom_index_1based,
        block_dimension,
    )?))
}

fn validate_rhorrp_gg_slice_bin(data: &RhorrpGgSliceBinData) -> Result<()> {
    validate_rhorrp_gg_slice_shape(data)?;
    validate_complex_values(data.values.iter().copied(), "gg_slice")
}

fn validate_rhorrp_gg_slice_shape(data: &RhorrpGgSliceBinData) -> Result<()> {
    let (energy_count, row_count, column_count) = data.values.dim();
    validate_positive("ne", energy_count)?;
    validate_positive("ldim", row_count)?;
    validate_positive("istate", column_count)?;
    ensure_i32("ne", energy_count)?;
    ensure_i32("ldim", row_count)?;
    ensure_i32("istate", column_count)?;
    Ok(())
}

fn validate_rhorrp_gg_diag_bin(data: &RhorrpGgDiagBinData) -> Result<()> {
    validate_rhorrp_gg_diag_shape(data)?;
    validate_complex_values(data.values.iter().copied(), "gg_diag")
}

fn validate_rhorrp_gg_diag_shape(data: &RhorrpGgDiagBinData) -> Result<()> {
    let (energy_count, atom_count, row_count, column_count) = data.values.dim();
    validate_positive("ne", energy_count)?;
    validate_positive("inclus", atom_count)?;
    validate_positive("ldim", row_count)?;
    validate_positive("ldim", column_count)?;
    if row_count != column_count {
        return invalid_rhorrp_gg_bin(format!(
            "gg_diag matrices must be square, got {row_count}x{column_count}"
        ));
    }
    ensure_i32("ne", energy_count)?;
    ensure_i32("inclus", atom_count)?;
    ensure_i32("ldim", row_count)?;
    Ok(())
}

fn validate_complex_values(
    values: impl IntoIterator<Item = Complex32>,
    field: &'static str,
) -> Result<()> {
    for (index, value) in values.into_iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_rhorrp_gg_bin(format!("{field} value {} is not finite", index + 1));
        }
    }
    Ok(())
}

fn ensure_finite_complex32(
    value: Complex32,
    field: &'static str,
    energy: usize,
    row: usize,
    column: usize,
) -> Result<Complex32> {
    if !(value.re.is_finite() && value.im.is_finite()) {
        return invalid_rhorrp_gg_bin(format!(
            "{field} value at energy {}, row {}, column {} is not finite",
            energy + 1,
            row + 1,
            column + 1
        ));
    }
    Ok(value)
}

fn detect_endian(bytes: &[u8], header_record_bytes: usize) -> Result<Endian> {
    let marker = read_marker_bytes(bytes, 0)?;
    let little = u32::from_le_bytes(marker);
    if little == header_record_bytes as u32 {
        return Ok(Endian::Little);
    }
    let big = u32::from_be_bytes(marker);
    if big == header_record_bytes as u32 {
        return Ok(Endian::Big);
    }
    invalid_rhorrp_gg_bin(format!(
        "first record marker is {little} little-endian/{big} big-endian, expected {header_record_bytes}"
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
        invalid_rhorrp_gg_bin_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_rhorrp_gg_bin(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_rhorrp_gg_bin_value("record length does not fit in u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
}

fn parse_complex32_payload(
    payload: &[u8],
    endian: Endian,
    expected: usize,
    field: &'static str,
) -> Result<Vec<Complex32>> {
    let expected_bytes = checked_product(expected, COMPLEX32_BYTES)?;
    if payload.len() != expected_bytes {
        return invalid_rhorrp_gg_bin(format!(
            "{field} payload has {} byte(s), expected {expected_bytes}",
            payload.len()
        ));
    }
    let mut values = Vec::with_capacity(expected);
    for index in 0..expected {
        let offset = checked_product(index, COMPLEX32_BYTES)?;
        let real = read_f32(payload, offset, endian)?;
        let imaginary = read_f32(payload, offset + F32_BYTES, endian)?;
        if !(real.is_finite() && imaginary.is_finite()) {
            return invalid_rhorrp_gg_bin(format!("{field} value {} is not finite", index + 1));
        }
        values.push(Complex32::new(real, imaginary));
    }
    Ok(values)
}

fn ensure_consumed(bytes: &[u8], position: usize, label: &'static str) -> Result<()> {
    if position != bytes.len() {
        return invalid_rhorrp_gg_bin(format!(
            "{label} has {} trailing byte(s)",
            bytes.len() - position
        ));
    }
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

fn read_f32(bytes: &[u8], offset: usize, endian: Endian) -> Result<f32> {
    let raw = read_f32_bytes(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(raw),
        Endian::Big => f32::from_be_bytes(raw),
    })
}

fn read_marker_bytes(bytes: &[u8], offset: usize) -> Result<[u8; FORTRAN_MARKER_BYTES]> {
    let slice = bytes
        .get(offset..offset + FORTRAN_MARKER_BYTES)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value("missing Fortran record marker"))?;
    let mut raw = [0_u8; FORTRAN_MARKER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_i32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; INTEGER_BYTES]> {
    let slice = bytes
        .get(offset..offset + INTEGER_BYTES)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value("missing i32 payload"))?;
    let mut raw = [0_u8; INTEGER_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn read_f32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F32_BYTES]> {
    let slice = bytes
        .get(offset..offset + F32_BYTES)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value("missing f32 payload"))?;
    let mut raw = [0_u8; F32_BYTES];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn parse_positive_i32(value: i32, field: &'static str) -> Result<usize> {
    if value <= 0 {
        return invalid_rhorrp_gg_bin(format!("{field} must be positive"));
    }
    usize::try_from(value)
        .map_err(|_| invalid_rhorrp_gg_bin_value(format!("{field} does not fit usize")))
}

fn push_i32(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_rhorrp_gg_bin_value(format!("{field} does not fit in i32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_complex32(bytes: &mut Vec<u8>, value: Complex32) {
    bytes.extend_from_slice(&value.re.to_le_bytes());
    bytes.extend_from_slice(&value.im.to_le_bytes());
}

fn ensure_i32(field: &'static str, value: usize) -> Result<()> {
    i32::try_from(value)
        .map(|_| ())
        .map_err(|_| invalid_rhorrp_gg_bin_value(format!("{field} does not fit in i32")))
}

fn validate_positive(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return invalid_rhorrp_gg_bin(format!("{field} must be positive"));
    }
    Ok(())
}

fn one_based_index(index: usize, length: usize, field: &'static str) -> Result<usize> {
    let zero_based = index
        .checked_sub(1)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value(format!("{field} index must be 1-based")))?;
    if zero_based >= length {
        return invalid_rhorrp_gg_bin(format!(
            "{field} index {index} is outside available range 1..={length}"
        ));
    }
    Ok(zero_based)
}

fn block_start(
    atom_index_1based: usize,
    block_dimension: usize,
    field: &'static str,
) -> Result<usize> {
    let atom = atom_index_1based
        .checked_sub(1)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value(format!("{field} index must be 1-based")))?;
    checked_product(atom, block_dimension)
}

fn complex32_to_complex64(value: Complex32) -> Complex64 {
    Complex64::new(f64::from(value.re), f64::from(value.im))
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_rhorrp_gg_bin_value("integer overflow while adding dimensions"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        invalid_rhorrp_gg_bin_value(format!(
            "integer overflow while multiplying {left} by {right}"
        ))
    })
}

fn invalid_rhorrp_gg_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_rhorrp_gg_bin_value(message))
}

fn invalid_rhorrp_gg_bin_value(message: impl Into<String>) -> IoError {
    IoError::InvalidRhorrpGgBin {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEFF_GG_SLICE_BYTES: &[u8] = &[
        0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x0c, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x9a, 0x99, 0x31, 0x41, 0x9a, 0x99,
        0xa9, 0xc1, 0x9a, 0x99, 0x41, 0x41, 0x9a, 0x99, 0xb1, 0xc1, 0x33, 0x33, 0x33, 0x41, 0x33,
        0x33, 0xab, 0xc1, 0x33, 0x33, 0x43, 0x41, 0x33, 0x33, 0xb3, 0xc1, 0xcd, 0xcc, 0x34, 0x41,
        0xcd, 0xcc, 0xac, 0xc1, 0xcd, 0xcc, 0x44, 0x41, 0xcd, 0xcc, 0xb4, 0xc1, 0xcd, 0xcc, 0xa8,
        0x41, 0xcd, 0xcc, 0x24, 0xc2, 0xcd, 0xcc, 0xb0, 0x41, 0xcd, 0xcc, 0x28, 0xc2, 0x9a, 0x99,
        0xa9, 0x41, 0x9a, 0x99, 0x25, 0xc2, 0x9a, 0x99, 0xb1, 0x41, 0x9a, 0x99, 0x29, 0xc2, 0x66,
        0x66, 0xaa, 0x41, 0x66, 0x66, 0x26, 0xc2, 0x66, 0x66, 0xb2, 0x41, 0x66, 0x66, 0x2a, 0xc2,
        0x60, 0x00, 0x00, 0x00,
    ];

    const FEFF_GG_DIAG_BYTES: &[u8] = &[
        0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x33, 0x33,
        0xde, 0x42, 0x33, 0x33, 0x53, 0xc3, 0x33, 0x33, 0xe0, 0x42, 0x33, 0x33, 0x54, 0xc3, 0x66,
        0x66, 0xde, 0x42, 0x66, 0x66, 0x53, 0xc3, 0x66, 0x66, 0xe0, 0x42, 0x66, 0x66, 0x54, 0xc3,
        0x33, 0x33, 0xf2, 0x42, 0x33, 0x33, 0x5d, 0xc3, 0x33, 0x33, 0xf4, 0x42, 0x33, 0x33, 0x5e,
        0xc3, 0x66, 0x66, 0xf2, 0x42, 0x66, 0x66, 0x5d, 0xc3, 0x66, 0x66, 0xf4, 0x42, 0x66, 0x66,
        0x5e, 0xc3, 0x9a, 0x19, 0x53, 0x43, 0x9a, 0x99, 0xcd, 0xc3, 0x9a, 0x19, 0x54, 0x43, 0x9a,
        0x19, 0xce, 0xc3, 0x33, 0x33, 0x53, 0x43, 0x33, 0xb3, 0xcd, 0xc3, 0x33, 0x33, 0x54, 0x43,
        0x33, 0x33, 0xce, 0xc3, 0x9a, 0x19, 0x5d, 0x43, 0x9a, 0x99, 0xd2, 0xc3, 0x9a, 0x19, 0x5e,
        0x43, 0x9a, 0x19, 0xd3, 0xc3, 0x33, 0x33, 0x5d, 0x43, 0x33, 0xb3, 0xd2, 0xc3, 0x33, 0x33,
        0x5e, 0x43, 0x33, 0x33, 0xd3, 0xc3, 0x80, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn parses_feff_gg_slice_reference_bytes() -> Result<()> {
        let parsed = parse_rhorrp_gg_slice_bin(FEFF_GG_SLICE_BYTES)?;
        assert_eq!(parsed.energy_count(), 2);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.column_count(), 3);
        assert_eq!(parsed.values[(0, 0, 0)], Complex32::new(11.1, -21.2));
        assert_eq!(parsed.values[(0, 1, 2)], Complex32::new(12.3, -22.6));
        assert_eq!(parsed.values[(1, 0, 0)], Complex32::new(21.1, -41.2));
        assert_eq!(parsed.values[(1, 1, 2)], Complex32::new(22.3, -42.6));
        assert_eq!(rhorrp_gg_slice_bin_bytes(&parsed)?, FEFF_GG_SLICE_BYTES);
        Ok(())
    }

    #[test]
    fn parses_feff_gg_diag_reference_bytes() -> Result<()> {
        let parsed = parse_rhorrp_gg_diag_bin(FEFF_GG_DIAG_BYTES)?;
        assert_eq!(parsed.energy_count(), 2);
        assert_eq!(parsed.atom_count(), 2);
        assert_eq!(parsed.row_count(), 2);
        assert_eq!(parsed.column_count(), 2);
        assert_eq!(parsed.values[(0, 0, 0, 0)], Complex32::new(111.1, -211.2));
        assert_eq!(parsed.values[(0, 1, 1, 1)], Complex32::new(122.2, -222.4));
        assert_eq!(parsed.values[(1, 0, 0, 1)], Complex32::new(211.2, -411.4));
        assert_eq!(parsed.values[(1, 1, 1, 1)], Complex32::new(222.2, -422.4));
        assert_eq!(rhorrp_gg_diag_bin_bytes(&parsed)?, FEFF_GG_DIAG_BYTES);
        Ok(())
    }

    #[test]
    fn roundtrips_rhorrp_gg_bin_bytes() -> Result<()> {
        let slice = RhorrpGgSliceBinData {
            values: Array3::from_shape_fn((2, 2, 3), |(energy, row, column)| {
                let value = 0.1 * energy as f32 + 0.01 * row as f32 + column as f32;
                Complex32::new(value, -value)
            }),
        };
        let parsed_slice = parse_rhorrp_gg_slice_bin(&rhorrp_gg_slice_bin_bytes(&slice)?)?;
        assert_eq!(parsed_slice, slice);

        let diag = RhorrpGgDiagBinData {
            values: Array4::from_shape_fn((2, 2, 2, 2), |(energy, atom, row, column)| {
                let value =
                    energy as f32 + 0.1 * atom as f32 + 0.01 * row as f32 + 0.001 * column as f32;
                Complex32::new(value, -value)
            }),
        };
        let parsed_diag = parse_rhorrp_gg_diag_bin(&rhorrp_gg_diag_bin_bytes(&diag)?)?;
        assert_eq!(parsed_diag, diag);
        Ok(())
    }

    #[test]
    fn extracts_feff_gg_diag_matrix_for_core_rhorrp() -> Result<()> {
        let parsed = parse_rhorrp_gg_diag_bin(FEFF_GG_DIAG_BYTES)?;
        let matrix = rhorrp_gg_diag_matrix(&parsed, 2)?;
        assert_eq!(matrix.dim(), (2, 2, 2));
        assert_eq!(
            matrix[(0, 0, 0)],
            complex32_to_complex64(parsed.values[(0, 1, 0, 0)])
        );
        assert_eq!(
            matrix[(0, 1, 1)],
            complex32_to_complex64(parsed.values[(0, 1, 1, 1)])
        );
        assert_eq!(
            matrix[(1, 1, 0)],
            complex32_to_complex64(parsed.values[(1, 1, 1, 0)])
        );
        Ok(())
    }

    #[test]
    fn extracts_feff_gg_slice_block_for_core_rhorrp() -> Result<()> {
        let parsed = parse_rhorrp_gg_slice_bin(FEFF_GG_SLICE_BYTES)?;
        let block = rhorrp_gg_slice_block(&parsed, 1, 2, 1)?;
        assert_eq!(block.dim(), (2, 1, 1));
        assert_eq!(
            block[(0, 0, 0)],
            complex32_to_complex64(parsed.values[(0, 0, 1)])
        );
        assert_eq!(
            block[(1, 0, 0)],
            complex32_to_complex64(parsed.values[(1, 0, 1)])
        );

        let wide = RhorrpGgSliceBinData {
            values: Array3::from_shape_fn((2, 4, 4), |(energy, row, column)| {
                let value = 100.0 * energy as f32 + 10.0 * row as f32 + column as f32;
                Complex32::new(value, -value)
            }),
        };
        let wide_block = rhorrp_gg_slice_block(&wide, 1, 2, 2)?;
        assert_eq!(wide_block.dim(), (2, 2, 2));
        assert_eq!(
            wide_block[(0, 0, 0)],
            complex32_to_complex64(wide.values[(0, 0, 2)])
        );
        assert_eq!(
            wide_block[(1, 1, 1)],
            complex32_to_complex64(wide.values[(1, 1, 3)])
        );
        Ok(())
    }

    #[test]
    fn selects_feff_gg_pair_matrix_like_rhoerrp() -> Result<()> {
        let diag = parse_rhorrp_gg_diag_bin(FEFF_GG_DIAG_BYTES)?;
        let slice = parse_rhorrp_gg_slice_bin(FEFF_GG_SLICE_BYTES)?;

        let same = rhorrp_gg_pair_matrix(&diag, &slice, 2, 2, 1)?;
        match same {
            RhorrpGgPairMatrix::Available(matrix) => {
                assert_eq!(matrix.dim(), (2, 2, 2));
                assert_eq!(
                    matrix[(0, 0, 0)],
                    complex32_to_complex64(diag.values[(0, 1, 0, 0)])
                );
            }
            RhorrpGgPairMatrix::UnsupportedOffCentralFirstAtom => {
                return invalid_rhorrp_gg_bin("expected same-site gg_diag matrix");
            }
        }

        let central_to_other = rhorrp_gg_pair_matrix(&diag, &slice, 1, 3, 1)?;
        match central_to_other {
            RhorrpGgPairMatrix::Available(matrix) => {
                assert_eq!(matrix.dim(), (2, 1, 1));
                assert_eq!(
                    matrix[(1, 0, 0)],
                    complex32_to_complex64(slice.values[(1, 0, 2)])
                );
            }
            RhorrpGgPairMatrix::UnsupportedOffCentralFirstAtom => {
                return invalid_rhorrp_gg_bin("expected central gg_slice matrix");
            }
        }

        assert_eq!(
            rhorrp_gg_pair_matrix(&diag, &slice, 2, 1, 1)?,
            RhorrpGgPairMatrix::UnsupportedOffCentralFirstAtom
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_rhorrp_gg_bin_data() -> Result<()> {
        assert!(parse_rhorrp_gg_slice_bin(&[]).is_err());

        let mut bad_marker = FEFF_GG_SLICE_BYTES.to_vec();
        bad_marker[0] = 0;
        assert!(parse_rhorrp_gg_slice_bin(&bad_marker).is_err());

        let bad_slice = RhorrpGgSliceBinData {
            values: Array3::from_elem((1, 1, 1), Complex32::new(f32::NAN, 0.0)),
        };
        assert!(rhorrp_gg_slice_bin_bytes(&bad_slice).is_err());

        let bad_diag = RhorrpGgDiagBinData {
            values: Array4::from_elem((1, 1, 1, 2), Complex32::new(0.0, 0.0)),
        };
        assert!(rhorrp_gg_diag_bin_bytes(&bad_diag).is_err());

        let parsed_diag = parse_rhorrp_gg_diag_bin(FEFF_GG_DIAG_BYTES)?;
        assert!(rhorrp_gg_diag_matrix(&parsed_diag, 0).is_err());
        assert!(rhorrp_gg_diag_matrix(&parsed_diag, 3).is_err());

        let parsed_slice = parse_rhorrp_gg_slice_bin(FEFF_GG_SLICE_BYTES)?;
        assert!(rhorrp_gg_slice_block(&parsed_slice, 1, 1, 0).is_err());
        assert!(rhorrp_gg_slice_block(&parsed_slice, 3, 1, 1).is_err());
        assert!(rhorrp_gg_slice_block(&parsed_slice, 1, 4, 1).is_err());
        assert!(rhorrp_gg_pair_matrix(&parsed_diag, &parsed_slice, 0, 1, 1).is_err());
        assert!(rhorrp_gg_pair_matrix(&parsed_diag, &parsed_slice, 1, 0, 1).is_err());
        Ok(())
    }
}
