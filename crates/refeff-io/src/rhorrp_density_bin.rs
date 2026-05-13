//! FEFF RHORRP binary density-output codec.
//!
//! When a RHORRP density-grid filename ends in `.bin`, `RHORRP/rhorrp.f90`
//! writes sequential Fortran-unformatted records containing the grid
//! dimensionality, origin, axis vectors, point counts, and density values. This
//! module exposes that layout as ndarray-backed Rust data and writes the same
//! little-endian record format used by the generated FEFF10 reference suite.

use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::control_input::FEFF_BOHR_ANGSTROM;
use crate::error::{IoError, Result};

const DIMENSION_RECORD_BYTES: usize = 4;
const VECTOR3_RECORD_BYTES: usize = 24;
const POINT_COUNT_RECORD_BYTES: usize = 4;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F64_BYTES: usize = 8;
const RHORRP_COORDINATE_COLUMNS: usize = 3;
const MAX_RHORRP_DIMENSIONS: usize = 3;

/// Parsed FEFF RHORRP binary density output.
#[derive(Debug, Clone, PartialEq)]
pub struct RhorrpDensityBinData {
    /// Grid origin in Angstroms.
    pub origin_angstrom: [f64; 3],
    /// Grid axis vectors in Angstroms as `(xyz, dimension)`.
    pub axes_angstrom: Array2<f64>,
    /// Number of grid points along each axis.
    pub points_per_axis: Vec<usize>,
    /// Density values in inverse cubic Angstroms, in FEFF point traversal order.
    pub density_per_angstrom3: Array1<f64>,
}

/// Bohr-unit RHORRP density grid ready for FEFF binary-output conversion.
#[derive(Debug, Clone, Copy)]
pub struct RhorrpDensityBinBohrInput<'a> {
    /// Grid origin in Bohr.
    pub origin_bohr: [f64; 3],
    /// Grid axis vectors in Bohr as `(xyz, dimension)`.
    pub axes_bohr: ArrayView2<'a, f64>,
    /// Number of grid points along each active axis.
    pub points_per_axis: &'a [usize],
    /// Density values in inverse cubic Bohr, in FEFF point traversal order.
    pub density_per_bohr3: ArrayView1<'a, f64>,
}

impl RhorrpDensityBinData {
    /// Number of spatial dimensions in the RHORRP grid.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.points_per_axis.len()
    }

    /// Number of density values stored in the final binary record.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.density_per_angstrom3.len()
    }
}

/// Convert RHORRP Bohr-unit calculation output to FEFF binary density data.
///
/// FEFF `calculate_density` writes binary grid metadata in Angstrom units and
/// density in inverse cubic Angstroms. This helper performs only that unit
/// conversion and validates the resulting binary payload shape.
pub fn rhorrp_density_bin_from_bohr(
    input: RhorrpDensityBinBohrInput<'_>,
) -> Result<RhorrpDensityBinData> {
    let (axis_rows, axis_columns) = input.axes_bohr.dim();
    if axis_rows != RHORRP_COORDINATE_COLUMNS || axis_columns != input.points_per_axis.len() {
        return invalid_rhorrp_density_bin(format!(
            "axes_bohr shape is {axis_rows}x{axis_columns}, expected {RHORRP_COORDINATE_COLUMNS}x{}",
            input.points_per_axis.len()
        ));
    }

    let coordinate_scale = FEFF_BOHR_ANGSTROM;
    let density_scale = 1.0 / (FEFF_BOHR_ANGSTROM * FEFF_BOHR_ANGSTROM * FEFF_BOHR_ANGSTROM);

    let mut axes_angstrom = Array2::zeros((axis_rows, axis_columns));
    for dimension in 0..axis_columns {
        for coordinate in 0..axis_rows {
            axes_angstrom[(coordinate, dimension)] =
                input.axes_bohr[(coordinate, dimension)] * coordinate_scale;
        }
    }

    let data = RhorrpDensityBinData {
        origin_angstrom: [
            input.origin_bohr[0] * coordinate_scale,
            input.origin_bohr[1] * coordinate_scale,
            input.origin_bohr[2] * coordinate_scale,
        ],
        axes_angstrom,
        points_per_axis: input.points_per_axis.to_vec(),
        density_per_angstrom3: input
            .density_per_bohr3
            .mapv(|density| density * density_scale),
    };
    validate_rhorrp_density_bin(&data)?;
    Ok(data)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF RHORRP binary density-output bytes.
pub fn parse_rhorrp_density_bin(bytes: &[u8]) -> Result<RhorrpDensityBinData> {
    let endian = detect_endian(bytes)?;
    let mut position = 0;

    let dimensions_record = read_record(bytes, &mut position, endian, "dimension count")?;
    if dimensions_record.len() != DIMENSION_RECORD_BYTES {
        return invalid_rhorrp_density_bin(format!(
            "dimension record has {} byte(s), expected {DIMENSION_RECORD_BYTES}",
            dimensions_record.len()
        ));
    }
    let dimensions = parse_dimensions(read_i32(dimensions_record, 0, endian)?)?;

    let origin_record = read_record(bytes, &mut position, endian, "origin")?;
    let origin_angstrom = read_vector3_record(origin_record, endian, "origin")?;

    let mut axes_angstrom = Array2::<f64>::zeros((RHORRP_COORDINATE_COLUMNS, dimensions));
    let mut points_per_axis = Vec::with_capacity(dimensions);
    for dimension in 0..dimensions {
        let axis_record = read_record(bytes, &mut position, endian, "axis")?;
        let axis = read_vector3_record(axis_record, endian, "axis")?;
        for coordinate in 0..RHORRP_COORDINATE_COLUMNS {
            axes_angstrom[(coordinate, dimension)] = axis[coordinate];
        }

        let count_record = read_record(bytes, &mut position, endian, "axis point count")?;
        if count_record.len() != POINT_COUNT_RECORD_BYTES {
            return invalid_rhorrp_density_bin(format!(
                "axis point-count record has {} byte(s), expected {POINT_COUNT_RECORD_BYTES}",
                count_record.len()
            ));
        }
        points_per_axis.push(parse_positive_i32(
            read_i32(count_record, 0, endian)?,
            "axis point count",
        )?);
    }

    let declared_points = checked_product_all(&points_per_axis)?;
    let density_record = read_record(bytes, &mut position, endian, "density")?;
    let expected_density_bytes = checked_product(declared_points, F64_BYTES)?;
    if density_record.len() != expected_density_bytes {
        return invalid_rhorrp_density_bin(format!(
            "density record has {} byte(s), expected {expected_density_bytes}",
            density_record.len()
        ));
    }

    let mut density = Vec::with_capacity(declared_points);
    for point in 0..declared_points {
        density.push(read_f64(
            density_record,
            checked_product(point, F64_BYTES)?,
            endian,
        )?);
    }

    if position != bytes.len() {
        return invalid_rhorrp_density_bin(format!(
            "RHORRP density binary has {} trailing byte(s)",
            bytes.len() - position
        ));
    }

    let data = RhorrpDensityBinData {
        origin_angstrom,
        axes_angstrom,
        points_per_axis,
        density_per_angstrom3: Array1::from_vec(density),
    };
    validate_rhorrp_density_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian RHORRP binary density-output bytes.
pub fn rhorrp_density_bin_bytes(data: &RhorrpDensityBinData) -> Result<Vec<u8>> {
    validate_rhorrp_density_bin(data)?;

    let density_bytes = checked_product(data.point_count(), F64_BYTES)?;
    let metadata_bytes = checked_add(
        checked_add(DIMENSION_RECORD_BYTES, VECTOR3_RECORD_BYTES)?,
        data.dimensions()
            .checked_mul(checked_add(VECTOR3_RECORD_BYTES, POINT_COUNT_RECORD_BYTES)?)
            .ok_or_else(|| invalid_rhorrp_density_bin_value("metadata length overflows usize"))?,
    )?;
    let mut bytes =
        Vec::with_capacity(recorded_len(metadata_bytes)? + recorded_len(density_bytes)?);

    let mut dimensions = Vec::with_capacity(DIMENSION_RECORD_BYTES);
    push_i32(&mut dimensions, data.dimensions(), "dimension count")?;
    write_record(&mut bytes, &dimensions)?;

    write_vector3_record(&mut bytes, data.origin_angstrom)?;

    for dimension in 0..data.dimensions() {
        write_vector3_record(
            &mut bytes,
            [
                data.axes_angstrom[(0, dimension)],
                data.axes_angstrom[(1, dimension)],
                data.axes_angstrom[(2, dimension)],
            ],
        )?;

        let mut count = Vec::with_capacity(POINT_COUNT_RECORD_BYTES);
        push_i32(
            &mut count,
            data.points_per_axis[dimension],
            "axis point count",
        )?;
        write_record(&mut bytes, &count)?;
    }

    let mut density = Vec::with_capacity(density_bytes);
    for value in &data.density_per_angstrom3 {
        density.extend_from_slice(&value.to_le_bytes());
    }
    write_record(&mut bytes, &density)?;
    Ok(bytes)
}

/// Return whether FEFF RHORRP treats a density-output filename as binary.
///
/// This ports `RHORRP/rhorrp.f90` `filename_is_binary`: use the text after
/// the last dot, copy it into FEFF's four-character extension buffer, lowercase
/// ASCII letters, and compare that padded extension with `bin`.
#[must_use]
pub fn rhorrp_density_filename_is_binary(filename: &str) -> bool {
    let Some(dot_position) = filename.rfind('.') else {
        return false;
    };

    let mut extension = [b' '; 4];
    for (slot, byte) in extension
        .iter_mut()
        .zip(filename[dot_position + 1..].bytes())
    {
        *slot = match byte {
            b'A'..=b'Z' => byte + (b'a' - b'A'),
            _ => byte,
        };
    }
    extension == *b"bin "
}

/// Read FEFF RHORRP binary density output from a file.
pub fn read_rhorrp_density_bin(path: impl AsRef<Path>) -> Result<RhorrpDensityBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_rhorrp_density_bin(&bytes)
}

/// Write FEFF RHORRP binary density output to a file.
pub fn write_rhorrp_density_bin(path: impl AsRef<Path>, data: &RhorrpDensityBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, rhorrp_density_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

fn validate_rhorrp_density_bin(data: &RhorrpDensityBinData) -> Result<()> {
    let dimensions = data.dimensions();
    if !(1..=MAX_RHORRP_DIMENSIONS).contains(&dimensions) {
        return invalid_rhorrp_density_bin(format!(
            "dimension count must be in 1..={MAX_RHORRP_DIMENSIONS}, got {dimensions}"
        ));
    }

    let (axis_rows, axis_columns) = data.axes_angstrom.dim();
    if axis_rows != RHORRP_COORDINATE_COLUMNS || axis_columns != dimensions {
        return invalid_rhorrp_density_bin(format!(
            "axes shape is {axis_rows}x{axis_columns}, expected {RHORRP_COORDINATE_COLUMNS}x{dimensions}"
        ));
    }

    for (index, count) in data.points_per_axis.iter().copied().enumerate() {
        if count == 0 {
            return invalid_rhorrp_density_bin(format!(
                "points_per_axis[{index}] must be positive"
            ));
        }
        ensure_i32("axis point count", count)?;
    }
    ensure_i32("dimension count", dimensions)?;

    let expected_density = checked_product_all(&data.points_per_axis)?;
    if data.point_count() != expected_density {
        return invalid_rhorrp_density_bin(format!(
            "density has {} value(s), expected {expected_density}",
            data.point_count()
        ));
    }

    for (index, value) in data.origin_angstrom.iter().enumerate() {
        validate_finite("origin", *value, index)?;
    }
    for (index, value) in data.axes_angstrom.iter().enumerate() {
        validate_finite("axis", *value, index)?;
    }
    for (index, value) in data.density_per_angstrom3.iter().enumerate() {
        validate_finite("density", *value, index)?;
    }

    Ok(())
}

fn detect_endian(bytes: &[u8]) -> Result<Endian> {
    let marker = read_marker_bytes(bytes, 0)?;
    let little = u32::from_le_bytes(marker);
    if little == DIMENSION_RECORD_BYTES as u32 {
        return Ok(Endian::Little);
    }
    let big = u32::from_be_bytes(marker);
    if big == DIMENSION_RECORD_BYTES as u32 {
        return Ok(Endian::Big);
    }
    invalid_rhorrp_density_bin(format!(
        "first record marker is {little} little-endian/{big} big-endian, expected {DIMENSION_RECORD_BYTES}"
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
        invalid_rhorrp_density_bin_value(format!(
            "{label} record is truncated by payload length {length}"
        ))
    })?;
    *position = end;
    let trailing = read_u32(bytes, *position, endian)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    if trailing != length {
        return invalid_rhorrp_density_bin(format!(
            "{label} record trailing marker is {trailing}, expected {length}"
        ));
    }
    Ok(payload)
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_rhorrp_density_bin_value("record length does not fit in u32"))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
}

fn write_vector3_record(bytes: &mut Vec<u8>, values: [f64; 3]) -> Result<()> {
    let mut payload = Vec::with_capacity(VECTOR3_RECORD_BYTES);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    write_record(bytes, &payload)
}

fn read_vector3_record(bytes: &[u8], endian: Endian, label: &'static str) -> Result<[f64; 3]> {
    if bytes.len() != VECTOR3_RECORD_BYTES {
        return invalid_rhorrp_density_bin(format!(
            "{label} record has {} byte(s), expected {VECTOR3_RECORD_BYTES}",
            bytes.len()
        ));
    }
    Ok([
        read_f64(bytes, 0, endian)?,
        read_f64(bytes, F64_BYTES, endian)?,
        read_f64(bytes, F64_BYTES * 2, endian)?,
    ])
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
    read_fixed_bytes(bytes, offset, "Fortran record marker")
}

fn read_i32_bytes(bytes: &[u8], offset: usize) -> Result<[u8; INTEGER_BYTES]> {
    read_fixed_bytes(bytes, offset, "i32 payload")
}

fn read_f64_bytes(bytes: &[u8], offset: usize) -> Result<[u8; F64_BYTES]> {
    read_fixed_bytes(bytes, offset, "f64 payload")
}

fn read_fixed_bytes<const N: usize>(bytes: &[u8], offset: usize, label: &str) -> Result<[u8; N]> {
    let end = checked_add(offset, N)?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_rhorrp_density_bin_value(format!("missing {label}")))?;
    let mut raw = [0_u8; N];
    raw.copy_from_slice(slice);
    Ok(raw)
}

fn parse_dimensions(value: i32) -> Result<usize> {
    let dimensions = parse_positive_i32(value, "dimension count")?;
    if dimensions > MAX_RHORRP_DIMENSIONS {
        return invalid_rhorrp_density_bin(format!(
            "dimension count must be in 1..={MAX_RHORRP_DIMENSIONS}, got {dimensions}"
        ));
    }
    Ok(dimensions)
}

fn parse_positive_i32(value: i32, field: &'static str) -> Result<usize> {
    if value <= 0 {
        return invalid_rhorrp_density_bin(format!("{field} must be positive"));
    }
    usize::try_from(value)
        .map_err(|_| invalid_rhorrp_density_bin_value(format!("{field} does not fit in usize")))
}

fn push_i32(bytes: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_rhorrp_density_bin_value(format!("{field} does not fit in i32")))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn ensure_i32(field: &'static str, value: usize) -> Result<()> {
    i32::try_from(value)
        .map(|_| ())
        .map_err(|_| invalid_rhorrp_density_bin_value(format!("{field} does not fit in i32")))
}

fn validate_finite(field: &'static str, value: f64, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        invalid_rhorrp_density_bin(format!("{field} value {} is not finite", index + 1))
    }
}

fn recorded_len(payload_len: usize) -> Result<usize> {
    checked_add(
        checked_add(payload_len, FORTRAN_MARKER_BYTES)?,
        FORTRAN_MARKER_BYTES,
    )
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_rhorrp_density_bin_value("byte offset overflows usize"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_rhorrp_density_bin_value("record length overflows usize"))
}

fn checked_product_all(values: &[usize]) -> Result<usize> {
    values.iter().copied().try_fold(1_usize, checked_product)
}

fn invalid_rhorrp_density_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_rhorrp_density_bin_value(message))
}

fn invalid_rhorrp_density_bin_value(message: impl Into<String>) -> IoError {
    IoError::InvalidRhorrpDensityBin {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_HEX: &str = concat!(
        "040000000200000004000000180000009a9999999999b93f9a9999999999c9bf",
        "333333333333d33f1800000018000000000000000000f03f000000000000e03f",
        "000000000000d0bf18000000040000000300000004000000180000009a999999",
        "9999c9bf000000000000f43f000000000000e83f180000000400000002000000",
        "0400000030000000333333333333b33f9a9999999999c93fcdccccccccccd43f",
        "cdccccccccccdc3f666666666666e23f666666666666e63f30000000",
    );

    #[test]
    fn rhorrp_density_bin_matches_feff_reference_bytes() -> Result<()> {
        let data = sample_density_bin();
        let reference = reference_bytes()?;

        assert_eq!(rhorrp_density_bin_bytes(&data)?, reference);
        assert_eq!(parse_rhorrp_density_bin(&reference)?, data);
        Ok(())
    }

    #[test]
    fn rhorrp_density_bin_roundtrips_files() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|source| IoError::Io {
            path: "rhorrp-density-bin-tempdir".into(),
            source,
        })?;
        let path = dir.path().join("density.bin");
        let data = sample_density_bin();

        write_rhorrp_density_bin(&path, &data)?;
        let parsed = read_rhorrp_density_bin(&path)?;

        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn rhorrp_density_bin_rejects_invalid_inputs() {
        assert!(parse_rhorrp_density_bin(&[]).is_err());
        assert!(parse_rhorrp_density_bin(&[4, 0, 0, 0]).is_err());

        let bad_density_len = RhorrpDensityBinData {
            density_per_angstrom3: Array1::zeros(5),
            ..sample_density_bin()
        };
        assert!(rhorrp_density_bin_bytes(&bad_density_len).is_err());

        let bad_axes = RhorrpDensityBinData {
            axes_angstrom: Array2::zeros((2, 2)),
            ..sample_density_bin()
        };
        assert!(rhorrp_density_bin_bytes(&bad_axes).is_err());

        let bad_value = RhorrpDensityBinData {
            density_per_angstrom3: Array1::from_vec(vec![f64::NAN; 6]),
            ..sample_density_bin()
        };
        assert!(rhorrp_density_bin_bytes(&bad_value).is_err());

        assert!(
            rhorrp_density_bin_from_bohr(RhorrpDensityBinBohrInput {
                origin_bohr: [0.0, 0.0, 0.0],
                axes_bohr: Array2::zeros((2, 2)).view(),
                points_per_axis: &[2, 2],
                density_per_bohr3: Array1::zeros(4).view(),
            })
            .is_err()
        );
    }

    #[test]
    fn rhorrp_density_filename_is_binary_matches_feff_reference() {
        let cases = [
            ("density.bin", true),
            ("density.BIN", true),
            ("density.bin1", false),
            ("archive.tar.bin", true),
            ("density", false),
            (".bin", true),
            ("density.", false),
            ("density.b", false),
            ("density.binary", false),
            ("density.bin   ", true),
        ];

        for (filename, expected) in cases {
            assert_eq!(
                rhorrp_density_filename_is_binary(filename),
                expected,
                "{filename}"
            );
        }
    }

    #[test]
    fn converts_bohr_density_bin_like_feff_reference() -> Result<()> {
        let axes_bohr = ndarray::arr2(&[[1.0, -0.2], [0.5, 1.25], [-0.25, 0.75]]);
        let density_per_bohr3 = ndarray::arr1(&[0.5, 2.0, -0.125, 0.0, 1.0, -2.0]);
        let data = rhorrp_density_bin_from_bohr(RhorrpDensityBinBohrInput {
            origin_bohr: [0.1, -0.2, 0.3],
            axes_bohr: axes_bohr.view(),
            points_per_axis: &[3, 2],
            density_per_bohr3: density_per_bohr3.view(),
        })?;

        assert_close(data.origin_angstrom[0], 0.052_917_724_9);
        assert_close(data.origin_angstrom[1], -0.105_835_449_8);
        assert_close(data.origin_angstrom[2], 0.158_753_174_699_999_97);
        assert_close(data.axes_angstrom[(0, 0)], 0.529_177_249);
        assert_close(data.axes_angstrom[(1, 0)], 0.264_588_624_5);
        assert_close(data.axes_angstrom[(2, 0)], -0.132_294_312_25);
        assert_close(data.axes_angstrom[(0, 1)], -0.105_835_449_8);
        assert_close(data.axes_angstrom[(1, 1)], 0.661_471_561_25);
        assert_close(data.axes_angstrom[(2, 1)], 0.396_882_936_749_999_97);
        assert_eq!(data.points_per_axis, [3, 2]);
        assert_close(data.density_per_angstrom3[0], 3.374_166_518_552_075_3);
        assert_close(data.density_per_angstrom3[1], 13.496_666_074_208_301);
        assert_close(data.density_per_angstrom3[2], -0.843_541_629_638_018_8);
        Ok(())
    }

    fn sample_density_bin() -> RhorrpDensityBinData {
        RhorrpDensityBinData {
            origin_angstrom: [0.1, -0.2, 0.3],
            axes_angstrom: ndarray::arr2(&[[1.0, -0.2], [0.5, 1.25], [-0.25, 0.75]]),
            points_per_axis: vec![3, 2],
            density_per_angstrom3: Array1::from_shape_fn(6, |index| {
                0.125 * (index + 1) as f64 - 0.05
            }),
        }
    }

    fn reference_bytes() -> Result<Vec<u8>> {
        let hex = REFERENCE_HEX.as_bytes();
        if !hex.len().is_multiple_of(2) {
            return invalid_rhorrp_density_bin("test reference hex has odd length");
        }
        hex.chunks_exact(2)
            .enumerate()
            .map(|(index, pair)| {
                let high = hex_nibble(pair[0], index)?;
                let low = hex_nibble(pair[1], index)?;
                Ok((high << 4) | low)
            })
            .collect()
    }

    fn hex_nibble(byte: u8, index: usize) -> Result<u8> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            b'A'..=b'F' => Ok(byte - b'A' + 10),
            _ => invalid_rhorrp_density_bin(format!(
                "invalid test reference hex byte at index {index}"
            )),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }
}
