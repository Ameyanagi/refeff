//! FEFF `specfunct.dat` spectral-function cache codec.
//!
//! `SFCONV/so2conv.f90` stores this cache as thirteen Fortran sequential
//! unformatted records. The payloads are scalar material settings, pole arrays,
//! two eight-column momentum tables, and seven spectral-function tables. The
//! parser accepts little-endian and big-endian record markers and payloads; the
//! writer emits the little-endian layout produced by the generated FEFF10
//! reference suite.

use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1};
use refeff_core::{
    Real, SfconvMomentumSpectralInterpolation, SfconvMomentumSpectralInterpolationInput,
    sfconv_interpolate_momentum_spectral_function,
};

use crate::error::{IoError, Result};

const HEADER_RECORD_BYTES: usize = 32;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F64_BYTES: usize = 8;
/// Number of FEFF `sfinfo`/`wgts` columns in `specfunct.dat`.
pub const SPECFUNCT_DAT_INFO_COLUMNS: usize = 8;

/// Parsed FEFF `specfunct.dat` SO2CONV spectral-function cache.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpecfunctData {
    /// Interstitial Wigner-Seitz radius, FEFF `rs`.
    pub wigner_seitz_radius: f64,
    /// Core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: f64,
    /// Asymmetric quasiparticle-phase selector, FEFF `iasym`.
    pub asymmetric_phase: i32,
    /// Satellite approximation selector, FEFF `isattype`.
    pub satellite_type: i32,
    /// Low-q self-energy selector, FEFF `lowq`.
    pub low_q_mode: i32,
    /// Number of active epsilon-inverse poles, FEFF `npl`.
    pub pole_count: usize,
    /// Pole energies for the full FEFF `nplmax` slot capacity, FEFF `plengy`.
    pub pole_energy: Array1<f64>,
    /// Pole broadenings for the full FEFF `nplmax` slot capacity, FEFF `plbrd`.
    pub pole_broadening: Array1<f64>,
    /// Pole weights for the full FEFF `nplmax` slot capacity, FEFF `plwt`.
    pub pole_weight: Array1<f64>,
    /// Momentum-row metadata table, FEFF `sfinfo(nqpts,8)`.
    pub spectral_info: Array2<f64>,
    /// Eight spectral weights for each momentum row, FEFF `wgts(nqpts,8)`.
    pub weights: Array2<f64>,
    /// Extrinsic quasiparticle table, FEFF `emsf(nqpts,nsfpts)`.
    pub extrinsic_quasiparticle: Array2<f64>,
    /// Extrinsic satellite table, FEFF `essf(nqpts,nsfpts)`.
    pub extrinsic_satellite: Array2<f64>,
    /// Interference quasiparticle table, FEFF `xmsf(nqpts,nsfpts)`.
    pub interference_quasiparticle: Array2<f64>,
    /// Interference satellite table, FEFF `xssf(nqpts,nsfpts)`.
    pub interference_satellite: Array2<f64>,
    /// Intrinsic satellite table, FEFF `xissf(nqpts,nsfpts)`.
    pub intrinsic_satellite: Array2<f64>,
    /// Clipped extrinsic satellite table, FEFF `escsf(nqpts,nsfpts)`.
    pub clipped_extrinsic_satellite: Array2<f64>,
    /// Spectral-function energy table, FEFF `engrid(nqpts,nsfpts)`.
    pub energy_grid: Array2<f64>,
}

impl SfconvSpecfunctData {
    /// Number of pole slots serialized in each FEFF pole record.
    #[must_use]
    pub fn pole_capacity(&self) -> usize {
        self.pole_energy.len()
    }

    /// Number of SO2CONV momentum rows, FEFF `nqpts`.
    #[must_use]
    pub fn momentum_count(&self) -> usize {
        self.spectral_info.nrows()
    }

    /// Number of spectral-function energy rows, FEFF `nsfpts`.
    #[must_use]
    pub fn spectral_point_count(&self) -> usize {
        self.energy_grid.ncols()
    }
}

/// Current SO2CONV inputs used to decide whether a cache can be reused.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctCompatibilityInput<'a> {
    /// Current interstitial Wigner-Seitz radius, FEFF `rs`.
    pub wigner_seitz_radius: f64,
    /// Current core-hole lifetime broadening in Hartree, FEFF `gammach`.
    pub core_hole_lifetime: f64,
    /// Current asymmetric quasiparticle-phase selector, FEFF `iasym`.
    pub asymmetric_phase: i32,
    /// Current satellite approximation selector, FEFF `isattype`.
    pub satellite_type: i32,
    /// Current low-q self-energy selector, FEFF `lowq`.
    pub low_q_mode: i32,
    /// Number of active current poles, FEFF `npl`.
    pub pole_count: usize,
    /// Current pole energies, FEFF `plengy`.
    pub pole_energy: ArrayView1<'a, f64>,
    /// Current pole broadenings, FEFF `plbrd`.
    pub pole_broadening: ArrayView1<'a, f64>,
    /// Current pole weights, FEFF `plwt`.
    pub pole_weight: ArrayView1<'a, f64>,
    /// Current minimal SO2CONV momentum grid, FEFF `pgrid`.
    pub momentum_grid: ArrayView1<'a, f64>,
}

/// Parse FEFF `specfunct.dat` bytes.
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

/// Return whether parsed cache data matches the current SO2CONV inputs.
///
/// This mirrors the reuse checks in `SFCONV/so2conv.f90`: material scalars,
/// integer selectors, active pole rows, and the momentum grid must match. FEFF
/// compares the momentum grid after converting both values to default `REAL`,
/// so this function compares those entries as `f32`.
pub fn sfconv_specfunct_matches_so2conv_inputs(
    data: &SfconvSpecfunctData,
    input: SfconvSpecfunctCompatibilityInput<'_>,
) -> Result<bool> {
    validate_specfunct_dat(data)?;
    validate_compatibility_input(input)?;

    if data.wigner_seitz_radius != input.wigner_seitz_radius
        || data.core_hole_lifetime != input.core_hole_lifetime
        || data.asymmetric_phase != input.asymmetric_phase
        || data.low_q_mode != input.low_q_mode
        || data.satellite_type != input.satellite_type
        || data.pole_count != input.pole_count
        || data.momentum_count() != input.momentum_grid.len()
    {
        return Ok(false);
    }

    let active_poles_match = (0..data.pole_count).all(|index| {
        data.pole_energy[index] == input.pole_energy[index]
            && data.pole_broadening[index] == input.pole_broadening[index]
            && data.pole_weight[index] == input.pole_weight[index]
    });
    if !active_poles_match {
        return Ok(false);
    }

    let momentum_matches = (0..data.momentum_count()).all(|index| {
        (data.spectral_info[[index, 0]] as f32) == (input.momentum_grid[index] as f32)
    });
    Ok(momentum_matches)
}

/// Build a validated core interpolation view over a `specfunct.dat` cache.
///
/// This maps the FEFF cache layout to `refeff-core`'s momentum spectral
/// interpolation input without copying the cached arrays. The first `sfinfo`
/// column is FEFF `pgrid`; columns 4 through 8 are `se`, `ce`, `width`, `z1`,
/// and `z1i`.
pub fn sfconv_specfunct_momentum_interpolation_input(
    data: &SfconvSpecfunctData,
    photoelectron_momentum: Real,
) -> Result<SfconvMomentumSpectralInterpolationInput<'_>> {
    validate_specfunct_dat(data)?;
    validate_finite_scalar(photoelectron_momentum, "photoelectron momentum")?;

    Ok(SfconvMomentumSpectralInterpolationInput {
        photoelectron_momentum,
        momentum_grid: data.spectral_info.column(0),
        energy_grid: data.energy_grid.view(),
        extrinsic_quasiparticle: data.extrinsic_quasiparticle.view(),
        extrinsic_satellite: data.extrinsic_satellite.view(),
        interference_quasiparticle: data.interference_quasiparticle.view(),
        interference_satellite: data.interference_satellite.view(),
        intrinsic_satellite: data.intrinsic_satellite.view(),
        clipped_extrinsic_satellite: data.clipped_extrinsic_satellite.view(),
        weights: data.weights.view(),
        self_energy_real: data.spectral_info.column(3),
        energy_correction: data.spectral_info.column(4),
        width: data.spectral_info.column(5),
        renormalization_real: data.spectral_info.column(6),
        renormalization_imag: data.spectral_info.column(7),
    })
}

/// Interpolate one cached `specfunct.dat` spectral row to a photoelectron momentum.
///
/// This is the typed handoff from FEFF's binary SO2CONV cache to the core
/// numerical interpolation kernel used by the future full driver.
pub fn sfconv_specfunct_interpolate_momentum(
    data: &SfconvSpecfunctData,
    photoelectron_momentum: Real,
) -> Result<SfconvMomentumSpectralInterpolation> {
    let input = sfconv_specfunct_momentum_interpolation_input(data, photoelectron_momentum)?;
    sfconv_interpolate_momentum_spectral_function(input)
        .map_err(|source| IoError::SpecfunctDatInterpolation { source })
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

fn validate_specfunct_dat(data: &SfconvSpecfunctData) -> Result<()> {
    validate_finite_scalar(data.wigner_seitz_radius, "rs")?;
    validate_finite_scalar(data.core_hole_lifetime, "gammach")?;
    if data.pole_capacity() == 0 {
        return invalid_specfunct_dat("pole capacity must be positive");
    }
    if data.pole_count > data.pole_capacity() {
        return invalid_specfunct_dat(format!(
            "npl is {}, but pole capacity is {}",
            data.pole_count,
            data.pole_capacity()
        ));
    }
    validate_vector_shape(&data.pole_broadening, data.pole_capacity(), "plbrd")?;
    validate_vector_shape(&data.pole_weight, data.pole_capacity(), "plwt")?;
    validate_finite_vector(&data.pole_energy, "plengy")?;
    validate_finite_vector(&data.pole_broadening, "plbrd")?;
    validate_finite_vector(&data.pole_weight, "plwt")?;

    validate_info_shape(&data.spectral_info, "sfinfo")?;
    let momentum_count = data.momentum_count();
    validate_matrix_shape(
        &data.weights,
        momentum_count,
        SPECFUNCT_DAT_INFO_COLUMNS,
        "wgts",
    )?;
    let spectral_point_count = data.spectral_point_count();
    if spectral_point_count == 0 {
        return invalid_specfunct_dat("spectral table column count must be positive");
    }
    validate_spectral_table(
        &data.extrinsic_quasiparticle,
        momentum_count,
        spectral_point_count,
        "emsf",
    )?;
    validate_spectral_table(
        &data.extrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "essf",
    )?;
    validate_spectral_table(
        &data.interference_quasiparticle,
        momentum_count,
        spectral_point_count,
        "xmsf",
    )?;
    validate_spectral_table(
        &data.interference_satellite,
        momentum_count,
        spectral_point_count,
        "xssf",
    )?;
    validate_spectral_table(
        &data.intrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "xissf",
    )?;
    validate_spectral_table(
        &data.clipped_extrinsic_satellite,
        momentum_count,
        spectral_point_count,
        "escsf",
    )?;
    validate_spectral_table(
        &data.energy_grid,
        momentum_count,
        spectral_point_count,
        "engrid",
    )?;
    validate_finite_matrix(&data.spectral_info, "sfinfo")?;
    validate_finite_matrix(&data.weights, "wgts")?;
    Ok(())
}

fn validate_compatibility_input(input: SfconvSpecfunctCompatibilityInput<'_>) -> Result<()> {
    validate_finite_scalar(input.wigner_seitz_radius, "input rs")?;
    validate_finite_scalar(input.core_hole_lifetime, "input gammach")?;
    if input.pole_count > input.pole_energy.len() {
        return invalid_specfunct_dat("input npl exceeds plengy length");
    }
    if input.pole_count > input.pole_broadening.len() {
        return invalid_specfunct_dat("input npl exceeds plbrd length");
    }
    if input.pole_count > input.pole_weight.len() {
        return invalid_specfunct_dat("input npl exceeds plwt length");
    }
    if input.momentum_grid.is_empty() {
        return invalid_specfunct_dat("input pgrid must not be empty");
    }
    validate_finite_view(input.pole_energy, "input plengy")?;
    validate_finite_view(input.pole_broadening, "input plbrd")?;
    validate_finite_view(input.pole_weight, "input plwt")?;
    validate_finite_view(input.momentum_grid, "input pgrid")?;
    Ok(())
}

fn validate_info_shape(values: &Array2<f64>, field: &'static str) -> Result<()> {
    let (rows, cols) = values.dim();
    if rows == 0 || cols != SPECFUNCT_DAT_INFO_COLUMNS {
        return invalid_specfunct_dat(format!(
            "{field} shape is {rows}x{cols}, expected nonzero rows x {SPECFUNCT_DAT_INFO_COLUMNS}"
        ));
    }
    Ok(())
}

fn validate_spectral_table(
    values: &Array2<f64>,
    rows: usize,
    cols: usize,
    field: &'static str,
) -> Result<()> {
    validate_matrix_shape(values, rows, cols, field)?;
    validate_finite_matrix(values, field)
}

fn validate_vector_shape(
    values: &Array1<f64>,
    expected_len: usize,
    field: &'static str,
) -> Result<()> {
    if values.len() != expected_len {
        return invalid_specfunct_dat(format!(
            "{field} length is {}, expected {expected_len}",
            values.len()
        ));
    }
    Ok(())
}

fn validate_matrix_shape(
    values: &Array2<f64>,
    rows: usize,
    cols: usize,
    field: &'static str,
) -> Result<()> {
    let actual = values.dim();
    if actual != (rows, cols) {
        return invalid_specfunct_dat(format!(
            "{field} shape is {}x{}, expected {rows}x{cols}",
            actual.0, actual.1
        ));
    }
    Ok(())
}

fn validate_finite_scalar(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() {
        return invalid_specfunct_dat(format!("{field} must be finite"));
    }
    Ok(())
}

fn validate_finite_vector(values: &Array1<f64>, field: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!("{field} row {} must be finite", index + 1));
        }
    }
    Ok(())
}

fn validate_finite_view(values: ArrayView1<'_, f64>, field: &'static str) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!("{field} row {} must be finite", index + 1));
        }
    }
    Ok(())
}

fn validate_finite_matrix(values: &Array2<f64>, field: &'static str) -> Result<()> {
    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return invalid_specfunct_dat(format!(
                "{field} row {} column {} must be finite",
                row + 1,
                col + 1
            ));
        }
    }
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_specfunct_dat_value("byte offset overflows usize"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_specfunct_dat_value("record length overflows usize"))
}

fn invalid_specfunct_dat<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_specfunct_dat_value(message))
}

fn invalid_specfunct_dat_value(message: impl Into<String>) -> IoError {
    IoError::InvalidSpecfunctDat {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_specfunct_dat_bytes() -> Result<()> {
        let data = sample_specfunct_data();
        let bytes = specfunct_dat_bytes(&data)?;
        let parsed = parse_specfunct_dat(&bytes)?;
        assert_eq!(parsed, data);
        Ok(())
    }

    #[test]
    fn preserves_column_major_matrix_order() -> Result<()> {
        let data = sample_specfunct_data();
        let bytes = specfunct_dat_bytes(&data)?;
        let parsed = parse_specfunct_dat(&bytes)?;
        assert_eq!(parsed.spectral_info[[0, 1]], data.spectral_info[[0, 1]]);
        assert_eq!(
            parsed.extrinsic_quasiparticle[[2, 1]],
            data.extrinsic_quasiparticle[[2, 1]]
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_specfunct_dat_bytes() -> Result<()> {
        assert!(parse_specfunct_dat(&[]).is_err());

        let mut bytes = specfunct_dat_bytes(&sample_specfunct_data())?;
        bytes[0] = 0;
        assert!(parse_specfunct_dat(&bytes).is_err());

        let truncated = &bytes[..bytes.len() - 1];
        assert!(parse_specfunct_dat(truncated).is_err());
        Ok(())
    }

    #[test]
    fn rejects_invalid_specfunct_shapes() {
        let mut data = sample_specfunct_data();
        data.pole_broadening = Array1::from_vec(vec![0.1, 0.2]);
        assert!(specfunct_dat_bytes(&data).is_err());

        let mut data = sample_specfunct_data();
        data.weights = Array2::zeros((data.momentum_count(), SPECFUNCT_DAT_INFO_COLUMNS - 1));
        assert!(specfunct_dat_bytes(&data).is_err());
    }

    #[test]
    fn checks_so2conv_cache_compatibility() -> Result<()> {
        let data = sample_specfunct_data();
        let input = SfconvSpecfunctCompatibilityInput {
            wigner_seitz_radius: data.wigner_seitz_radius,
            core_hole_lifetime: data.core_hole_lifetime,
            asymmetric_phase: data.asymmetric_phase,
            satellite_type: data.satellite_type,
            low_q_mode: data.low_q_mode,
            pole_count: data.pole_count,
            pole_energy: data.pole_energy.view(),
            pole_broadening: data.pole_broadening.view(),
            pole_weight: data.pole_weight.view(),
            momentum_grid: data.spectral_info.column(0),
        };
        assert!(sfconv_specfunct_matches_so2conv_inputs(&data, input)?);

        let changed = SfconvSpecfunctCompatibilityInput {
            core_hole_lifetime: data.core_hole_lifetime + 1.0e-3,
            ..input
        };
        assert!(!sfconv_specfunct_matches_so2conv_inputs(&data, changed)?);
        Ok(())
    }

    #[test]
    fn builds_momentum_interpolation_input_from_cache() -> Result<()> {
        let data = sample_specfunct_data();
        let input = sfconv_specfunct_momentum_interpolation_input(&data, 0.75)?;
        assert_eq!(input.momentum_grid, data.spectral_info.column(0));
        assert_eq!(input.self_energy_real, data.spectral_info.column(3));
        assert_eq!(input.energy_grid, data.energy_grid.view());
        assert_eq!(input.weights, data.weights.view());
        Ok(())
    }

    #[test]
    fn interpolates_cached_spectral_row_to_momentum() -> Result<()> {
        let data = sample_specfunct_data();
        let interpolated = sfconv_specfunct_interpolate_momentum(&data, 0.75)?;

        assert_eq!(interpolated.energy.len(), data.spectral_point_count());
        assert_eq!(
            interpolated.spectral_function.nrows(),
            SPECFUNCT_DAT_INFO_COLUMNS
        );
        assert!(interpolated.self_energy_real > data.spectral_info[[0, 3]]);
        assert!(interpolated.self_energy_real < data.spectral_info[[1, 3]]);
        assert!(interpolated.weights[0] > data.weights[[0, 0]]);
        assert!(interpolated.weights[0] < data.weights[[1, 0]]);
        Ok(())
    }

    fn sample_specfunct_data() -> SfconvSpecfunctData {
        let momentum_count = 3;
        let spectral_count = 2;
        let pole_capacity = 4;
        let mut spectral_info = Array2::from_shape_fn(
            (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
            |(row, col)| row as f64 + 0.125 * col as f64,
        );
        for row in 0..momentum_count {
            spectral_info[[row, 0]] = 0.25 + row as f64;
        }

        SfconvSpecfunctData {
            wigner_seitz_radius: 2.05,
            core_hole_lifetime: 0.031,
            asymmetric_phase: 1,
            satellite_type: 2,
            low_q_mode: 0,
            pole_count: 3,
            pole_energy: Array1::from_vec(
                (0..pole_capacity).map(|index| 0.5 + index as f64).collect(),
            ),
            pole_broadening: Array1::from_vec(
                (0..pole_capacity)
                    .map(|index| 0.05 + 0.01 * index as f64)
                    .collect(),
            ),
            pole_weight: Array1::from_vec(
                (0..pole_capacity)
                    .map(|index| 1.0 / (index + 1) as f64)
                    .collect(),
            ),
            spectral_info,
            weights: Array2::from_shape_fn(
                (momentum_count, SPECFUNCT_DAT_INFO_COLUMNS),
                |(row, col)| 0.01 * row as f64 + 0.02 * col as f64,
            ),
            extrinsic_quasiparticle: spectral_table(momentum_count, spectral_count, 10.0),
            extrinsic_satellite: spectral_table(momentum_count, spectral_count, 20.0),
            interference_quasiparticle: spectral_table(momentum_count, spectral_count, 30.0),
            interference_satellite: spectral_table(momentum_count, spectral_count, 40.0),
            intrinsic_satellite: spectral_table(momentum_count, spectral_count, 50.0),
            clipped_extrinsic_satellite: spectral_table(momentum_count, spectral_count, 60.0),
            energy_grid: spectral_table(momentum_count, spectral_count, 70.0),
        }
    }

    fn spectral_table(momentum_count: usize, spectral_count: usize, base: f64) -> Array2<f64> {
        Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
            base + row as f64 + 0.1 * col as f64
        })
    }
}
