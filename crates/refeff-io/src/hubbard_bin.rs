//! FEFF Hubbard binary handoff codecs.
//!
//! FEFF writes the Hubbard extension arrays as sequential Fortran-unformatted
//! records without an embedded header. Callers therefore supply the active
//! dimensions from `hubbard.inp`, `.dimensions.dat`, `phase.bin`, or the
//! surrounding module setup.

use std::path::Path;

use ndarray::{Array4, Array5, Array6, Axis};
use num_complex::{Complex32, Complex64};

use crate::error::{IoError, Result};

const HEADER_RECORD_BYTES: usize = 20;
const FORTRAN_MARKER_BYTES: usize = 4;
const INTEGER_BYTES: usize = 4;
const F32_BYTES: usize = 4;
const F64_BYTES: usize = 8;
const COMPLEX32_BYTES: usize = 8;
const COMPLEX64_BYTES: usize = 16;
const HUBBARD_SPIN_COUNT: usize = 2;

/// FEFF `dimsmod::nex`, the fixed leading energy dimension written to
/// `aphase_hubbard.bin`.
pub const HUBBARD_APHASE_ENERGY_COUNT: usize = 2000;

/// Parsed FEFF `v_hubbard.bin` contents.
///
/// Values are exposed as `(potential, spin, angular, magnetic)` where
/// `angular` maps FEFF `0:lx`, `magnetic` maps FEFF `1:(lx+1)**2` to
/// zero-based Rust indices, and `potential` maps FEFF `0:nphx`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardVnlmBinData {
    /// Highest Hubbard angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Hubbard shifts as `(potential, spin, angular, magnetic)`.
    pub values: Array4<f64>,
}

impl HubbardVnlmBinData {
    /// Number of potential blocks, FEFF `nphx + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of spin projections. FEFF Hubbard handoffs always store two.
    #[must_use]
    pub fn spin_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of `l` channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of magnetic slots, equal to `(lx + 1)^2`.
    #[must_use]
    pub fn magnetic_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }

    /// Borrow the FEFF `Vnlm(0:lx,1:(lx+1)**2)` slice for one potential/spin.
    #[must_use]
    pub fn potential_spin(
        &self,
        potential: usize,
        spin: usize,
    ) -> Option<ndarray::ArrayView2<'_, f64>> {
        if potential >= self.potential_count() || spin >= self.spin_count() {
            return None;
        }
        Some(
            self.values
                .view()
                .index_axis_move(Axis(0), potential)
                .index_axis_move(Axis(0), spin),
        )
    }
}

/// Parsed FEFF `aphase_hubbard.bin` contents.
///
/// Values are exposed as `(potential, spin, energy, angular, magnetic)` where
/// `angular` maps FEFF `1:lx+1` to zero-based Rust indices and `magnetic` maps
/// FEFF `1:(lx+1)**2`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardAphaseBinData {
    /// Highest Hubbard angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Hubbard-shifted phase table as `(potential, spin, energy, angular, magnetic)`.
    pub values: Array5<Complex64>,
}

impl HubbardAphaseBinData {
    /// Number of potential blocks, FEFF `nphx + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Number of spin projections. FEFF Hubbard handoffs always store two.
    #[must_use]
    pub fn spin_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of phase-energy rows, FEFF `nex`/active caller count.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of `l` channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }

    /// Number of magnetic slots, equal to `(lx + 1)^2`.
    #[must_use]
    pub fn magnetic_count(&self) -> usize {
        self.values.len_of(Axis(4))
    }
}

/// Parsed FEFF `transformation_hubbard.bin` contents.
///
/// Values are exposed as `(potential, spin, angular, row, column)` where
/// `angular` maps FEFF `0:lx`, `row`/`column` map FEFF
/// `1:(2*l_hubbard+1)`, and `potential` maps FEFF `0:nphx`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardTransformationBinData {
    /// Active Hubbard angular momentum, FEFF `l_hubbard`.
    pub hubbard_l: usize,
    /// Highest FEFF angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Hubbard transformation matrix as `(potential, spin, angular, row, column)`.
    pub transform: Array5<Complex32>,
    /// Inverse Hubbard transformation matrix as `(potential, spin, angular, row, column)`.
    pub inverse: Array5<Complex32>,
}

impl HubbardTransformationBinData {
    /// Number of potential blocks, FEFF `nphx + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.transform.len_of(Axis(0))
    }

    /// Number of spin projections. FEFF Hubbard handoffs always store two.
    #[must_use]
    pub fn spin_count(&self) -> usize {
        self.transform.len_of(Axis(1))
    }

    /// Number of `l` channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.transform.len_of(Axis(2))
    }

    /// Number of Hubbard matrix rows, equal to `2*l_hubbard + 1`.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.transform.len_of(Axis(3))
    }

    /// Number of Hubbard matrix columns, equal to `2*l_hubbard + 1`.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.transform.len_of(Axis(4))
    }
}

/// Parsed FEFF Hubbard LDOS `gtrNN.bin` contents.
///
/// This is the spin-resolved Hubbard variant read by `LDOS/ff2rho_h.f90`, not
/// the ordinary non-Hubbard `gtrNN.bin` codec. Values are exposed as
/// `(spin, energy, potential, angular)`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardLdosGtrBinData {
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
    /// Highest angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Spin-resolved traces as `(spin, energy, potential, angular)`.
    pub values: Array4<Complex32>,
}

impl HubbardLdosGtrBinData {
    /// Number of parsed energy points.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of potential columns, equal to `nph + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of angular channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }
}

/// Parsed FEFF Hubbard LDOS `gtr_mNN.bin` contents.
///
/// Values are exposed as `(spin, energy, potential, angular, magnetic)`. Only
/// FEFF magnetic slots `l**2..(l+1)**2` are read or written for each angular
/// channel; other slots are retained in the in-memory shape and must be zero.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardLdosGtrMBinData {
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
    /// Highest angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Magnetic diagonal traces as `(spin, energy, potential, angular, magnetic)`.
    pub values: Array5<Complex32>,
}

/// LDOS-ready magnetic trace view selected from FEFF Hubbard `gtr_mNN.bin`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardLdosGtrMTraceHandoff {
    /// Number of energy points selected from the source trace.
    pub energy_count: usize,
    /// Number of angular channels selected from the source trace.
    pub angular_count: usize,
    /// Number of magnetic slots, equal to `(lx + 1)^2`.
    pub magnetic_count: usize,
    /// Zero-based FEFF potential index selected from `gtr_mNN.bin`.
    pub potential_index: usize,
    /// Number of horizontal-grid points, FEFF `ne1`.
    pub horizontal_count: usize,
    /// Number of DANES extension points, FEFF `ne3`.
    pub danes_extension_count: usize,
    /// Highest unique potential index, FEFF `nph`.
    pub highest_potential_index: usize,
    /// FEFF FMS selector, `ifms`.
    pub fms_mode: i32,
    /// FEFF `gtr_m(l,im,is,iph,ie)` as `(angular, magnetic, spin, energy)`.
    pub trace: Array4<Complex64>,
}

impl HubbardLdosGtrMBinData {
    /// Number of parsed energy points.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(1))
    }

    /// Number of potential columns, equal to `nph + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of angular channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }

    /// Number of magnetic slots, equal to `(lx + 1)^2`.
    #[must_use]
    pub fn magnetic_count(&self) -> usize {
        self.values.len_of(Axis(4))
    }
}

/// Parsed FEFF Hubbard LDOS `gtr_offNN.bin` contents.
///
/// Values are exposed as `(angular, spin, energy, potential, row, column)`.
/// The row/column order is `(l_hubbard + 1)^2`, matching the FEFF file branch
/// consumed by `ff2rho_h`.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardLdosGtrOffBinData {
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
    /// Active Hubbard angular momentum, FEFF `l_hubbard`.
    pub hubbard_l: usize,
    /// Highest angular channel, FEFF `lx`.
    pub angular_limit: usize,
    /// Off-diagonal Hubbard traces as `(angular, spin, energy, potential, row, column)`.
    pub values: Array6<Complex32>,
}

impl HubbardLdosGtrOffBinData {
    /// Number of parsed energy points.
    #[must_use]
    pub fn energy_count(&self) -> usize {
        self.values.len_of(Axis(2))
    }

    /// Number of potential columns, equal to `nph + 1`.
    #[must_use]
    pub fn potential_count(&self) -> usize {
        self.values.len_of(Axis(3))
    }

    /// Number of angular channels, equal to `lx + 1`.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.values.len_of(Axis(0))
    }

    /// Hubbard row/column order, equal to `(l_hubbard + 1)^2`.
    #[must_use]
    pub fn order(&self) -> usize {
        self.values.len_of(Axis(4))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// Parse FEFF `v_hubbard.bin` bytes.
pub fn parse_v_hubbard_bin(
    bytes: &[u8],
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardVnlmBinData> {
    validate_positive_dimension("potential_count", potential_count)?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let magnetic_count = magnetic_count_from_limit(angular_limit)?;
    let value_count = checked_product_all(&[
        angular_count,
        magnetic_count,
        HUBBARD_SPIN_COUNT,
        potential_count,
    ])?;
    let payload = read_single_record(
        bytes,
        checked_product(value_count, F64_BYTES)?,
        "v_hubbard.bin",
    )?;

    let mut values = Array4::zeros((
        potential_count,
        HUBBARD_SPIN_COUNT,
        angular_count,
        magnetic_count,
    ));
    let endian = detect_payload_endian(bytes, payload.len())?;
    let mut source = 0_usize;
    for potential in 0..potential_count {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for magnetic in 0..magnetic_count {
                for angular in 0..angular_count {
                    let offset = checked_product(source, F64_BYTES)?;
                    let value = read_f64(payload, offset, endian)?;
                    if !value.is_finite() {
                        return invalid_hubbard_bin(format!(
                            "v_hubbard.bin value {} is not finite",
                            source + 1
                        ));
                    }
                    values[(potential, spin, angular, magnetic)] = value;
                    source += 1;
                }
            }
        }
    }

    let data = HubbardVnlmBinData {
        angular_limit,
        values,
    };
    validate_v_hubbard_bin(&data)?;
    Ok(data)
}

/// Parse FEFF `v_hubbard.bin` bytes, inferring `lx` from the record length.
pub fn parse_v_hubbard_bin_inferred(
    bytes: &[u8],
    potential_count: usize,
) -> Result<HubbardVnlmBinData> {
    validate_positive_dimension("potential_count", potential_count)?;
    let (payload_len, _) = single_record_payload_metadata(bytes, "v_hubbard.bin")?;
    let angular_limit = infer_hubbard_angular_limit(
        payload_len,
        F64_BYTES,
        checked_product(HUBBARD_SPIN_COUNT, potential_count)?,
        "v_hubbard.bin",
    )?;
    parse_v_hubbard_bin(bytes, angular_limit, potential_count)
}

/// Render FEFF-compatible little-endian `v_hubbard.bin` bytes.
pub fn v_hubbard_bin_bytes(data: &HubbardVnlmBinData) -> Result<Vec<u8>> {
    validate_v_hubbard_bin(data)?;
    let mut payload = Vec::with_capacity(checked_product(data.values.len(), F64_BYTES)?);
    for potential in 0..data.potential_count() {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for magnetic in 0..data.magnetic_count() {
                for angular in 0..data.angular_count() {
                    payload.extend_from_slice(
                        &data.values[(potential, spin, angular, magnetic)].to_le_bytes(),
                    );
                }
            }
        }
    }

    let mut bytes = Vec::new();
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `v_hubbard.bin` from a file.
pub fn read_v_hubbard_bin(
    path: impl AsRef<Path>,
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardVnlmBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_v_hubbard_bin(&bytes, angular_limit, potential_count)
}

/// Read FEFF `v_hubbard.bin` from a file, inferring `lx` from the record length.
pub fn read_v_hubbard_bin_inferred(
    path: impl AsRef<Path>,
    potential_count: usize,
) -> Result<HubbardVnlmBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_v_hubbard_bin_inferred(&bytes, potential_count)
}

/// Write FEFF `v_hubbard.bin` bytes to a file.
pub fn write_v_hubbard_bin(path: impl AsRef<Path>, data: &HubbardVnlmBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, v_hubbard_bin_bytes(data)?).map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `aphase_hubbard.bin` bytes.
pub fn parse_aphase_hubbard_bin(
    bytes: &[u8],
    energy_count: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardAphaseBinData> {
    validate_positive_dimension("energy_count", energy_count)?;
    validate_positive_dimension("potential_count", potential_count)?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let magnetic_count = magnetic_count_from_limit(angular_limit)?;
    let value_count = checked_product_all(&[
        energy_count,
        angular_count,
        magnetic_count,
        HUBBARD_SPIN_COUNT,
        potential_count,
    ])?;
    let payload = read_single_record(
        bytes,
        checked_product(value_count, COMPLEX64_BYTES)?,
        "aphase_hubbard.bin",
    )?;

    let mut values = Array5::from_elem(
        (
            potential_count,
            HUBBARD_SPIN_COUNT,
            energy_count,
            angular_count,
            magnetic_count,
        ),
        Complex64::new(0.0, 0.0),
    );
    let endian = detect_payload_endian(bytes, payload.len())?;
    let mut source = 0_usize;
    for potential in 0..potential_count {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for magnetic in 0..magnetic_count {
                for angular in 0..angular_count {
                    for energy in 0..energy_count {
                        let offset = checked_product(source, COMPLEX64_BYTES)?;
                        let real = read_f64(payload, offset, endian)?;
                        let imaginary = read_f64(payload, offset + F64_BYTES, endian)?;
                        if !(real.is_finite() && imaginary.is_finite()) {
                            return invalid_hubbard_bin(format!(
                                "aphase_hubbard.bin value {} is not finite",
                                source + 1
                            ));
                        }
                        values[(potential, spin, energy, angular, magnetic)] =
                            Complex64::new(real, imaginary);
                        source += 1;
                    }
                }
            }
        }
    }

    let data = HubbardAphaseBinData {
        angular_limit,
        values,
    };
    validate_aphase_hubbard_bin(&data)?;
    Ok(data)
}

/// Parse FEFF `aphase_hubbard.bin` bytes, inferring `lx` from the record length.
pub fn parse_aphase_hubbard_bin_inferred(
    bytes: &[u8],
    energy_count: usize,
    potential_count: usize,
) -> Result<HubbardAphaseBinData> {
    validate_positive_dimension("energy_count", energy_count)?;
    validate_positive_dimension("potential_count", potential_count)?;
    let (payload_len, _) = single_record_payload_metadata(bytes, "aphase_hubbard.bin")?;
    let fixed_count = checked_product_all(&[energy_count, HUBBARD_SPIN_COUNT, potential_count])?;
    let angular_limit = infer_hubbard_angular_limit(
        payload_len,
        COMPLEX64_BYTES,
        fixed_count,
        "aphase_hubbard.bin",
    )?;
    parse_aphase_hubbard_bin(bytes, energy_count, angular_limit, potential_count)
}

/// Render FEFF-compatible little-endian `aphase_hubbard.bin` bytes.
pub fn aphase_hubbard_bin_bytes(data: &HubbardAphaseBinData) -> Result<Vec<u8>> {
    validate_aphase_hubbard_bin(data)?;
    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX64_BYTES)?);
    for potential in 0..data.potential_count() {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for magnetic in 0..data.magnetic_count() {
                for angular in 0..data.angular_count() {
                    for energy in 0..data.energy_count() {
                        let value = data.values[(potential, spin, energy, angular, magnetic)];
                        payload.extend_from_slice(&value.re.to_le_bytes());
                        payload.extend_from_slice(&value.im.to_le_bytes());
                    }
                }
            }
        }
    }

    let mut bytes = Vec::new();
    write_record(&mut bytes, &payload)?;
    Ok(bytes)
}

/// Read FEFF `aphase_hubbard.bin` from a file.
pub fn read_aphase_hubbard_bin(
    path: impl AsRef<Path>,
    energy_count: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardAphaseBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_aphase_hubbard_bin(&bytes, energy_count, angular_limit, potential_count)
}

/// Read FEFF `aphase_hubbard.bin` from a file, inferring `lx` from the record length.
pub fn read_aphase_hubbard_bin_inferred(
    path: impl AsRef<Path>,
    energy_count: usize,
    potential_count: usize,
) -> Result<HubbardAphaseBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_aphase_hubbard_bin_inferred(&bytes, energy_count, potential_count)
}

/// Write FEFF `aphase_hubbard.bin` bytes to a file.
pub fn write_aphase_hubbard_bin(path: impl AsRef<Path>, data: &HubbardAphaseBinData) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, aphase_hubbard_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF `transformation_hubbard.bin` bytes.
pub fn parse_transformation_hubbard_bin(
    bytes: &[u8],
    hubbard_l: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardTransformationBinData> {
    validate_positive_dimension("potential_count", potential_count)?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let order = transformation_order_from_hubbard_l(hubbard_l)?;
    let expected_len = transformation_hubbard_payload_len(order, angular_count, potential_count)?;
    let (records, endian) = record_payloads(bytes, "transformation_hubbard.bin")?;
    let [transform_payload, inverse_payload] = transformation_hubbard_record_pair(&records)?;

    let transform = parse_transformation_hubbard_payload(
        transform_payload,
        endian,
        "transformation_hubbard.bin transform",
        order,
        angular_count,
        potential_count,
        expected_len,
    )?;
    let inverse = parse_transformation_hubbard_payload(
        inverse_payload,
        endian,
        "transformation_hubbard.bin inverse",
        order,
        angular_count,
        potential_count,
        expected_len,
    )?;

    let data = HubbardTransformationBinData {
        hubbard_l,
        angular_limit,
        transform,
        inverse,
    };
    validate_transformation_hubbard_bin(&data)?;
    Ok(data)
}

/// Parse FEFF `transformation_hubbard.bin` bytes, inferring `lx` from the
/// first record length.
pub fn parse_transformation_hubbard_bin_inferred(
    bytes: &[u8],
    hubbard_l: usize,
    potential_count: usize,
) -> Result<HubbardTransformationBinData> {
    validate_positive_dimension("potential_count", potential_count)?;
    let (records, _) = record_payloads(bytes, "transformation_hubbard.bin")?;
    let [transform_payload, inverse_payload] = transformation_hubbard_record_pair(&records)?;
    if transform_payload.len() != inverse_payload.len() {
        return invalid_hubbard_bin(format!(
            "transformation_hubbard.bin transform payload has {} byte(s), inverse has {}",
            transform_payload.len(),
            inverse_payload.len()
        ));
    }
    let angular_limit = infer_transformation_hubbard_angular_limit(
        transform_payload.len(),
        hubbard_l,
        potential_count,
    )?;
    parse_transformation_hubbard_bin(bytes, hubbard_l, angular_limit, potential_count)
}

/// Render FEFF-compatible little-endian `transformation_hubbard.bin` bytes.
pub fn transformation_hubbard_bin_bytes(data: &HubbardTransformationBinData) -> Result<Vec<u8>> {
    validate_transformation_hubbard_bin(data)?;
    let transform = transformation_hubbard_payload(&data.transform);
    let inverse = transformation_hubbard_payload(&data.inverse);

    let mut bytes = Vec::new();
    write_record(&mut bytes, &transform)?;
    write_record(&mut bytes, &inverse)?;
    Ok(bytes)
}

/// Read FEFF `transformation_hubbard.bin` from a file.
pub fn read_transformation_hubbard_bin(
    path: impl AsRef<Path>,
    hubbard_l: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<HubbardTransformationBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_transformation_hubbard_bin(&bytes, hubbard_l, angular_limit, potential_count)
}

/// Read FEFF `transformation_hubbard.bin` from a file, inferring `lx` from the
/// first record length.
pub fn read_transformation_hubbard_bin_inferred(
    path: impl AsRef<Path>,
    hubbard_l: usize,
    potential_count: usize,
) -> Result<HubbardTransformationBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_transformation_hubbard_bin_inferred(&bytes, hubbard_l, potential_count)
}

/// Write FEFF `transformation_hubbard.bin` bytes to a file.
pub fn write_transformation_hubbard_bin(
    path: impl AsRef<Path>,
    data: &HubbardTransformationBinData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, transformation_hubbard_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF Hubbard LDOS `gtrNN.bin` bytes.
pub fn parse_hubbard_ldos_gtr_bin(
    bytes: &[u8],
    angular_limit: usize,
) -> Result<HubbardLdosGtrBinData> {
    let (records, endian) = record_payloads(bytes, "Hubbard LDOS gtrNN.bin")?;
    let [header, payload] = ldos_hubbard_trace_record_pair(&records, "Hubbard LDOS gtrNN.bin")?;
    let header = parse_ldos_hubbard_trace_header(header, endian, "Hubbard LDOS gtrNN.bin")?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let expected_len = checked_product(
        checked_product_all(&[
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            angular_count,
        ])?,
        COMPLEX32_BYTES,
    )?;
    if payload.len() != expected_len {
        return invalid_hubbard_bin(format!(
            "Hubbard LDOS gtrNN.bin trace payload has {} byte(s), expected {expected_len}",
            payload.len()
        ));
    }

    let mut values = Array4::from_elem(
        (
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            angular_count,
        ),
        Complex32::new(0.0, 0.0),
    );
    let mut source = 0_usize;
    for spin in 0..HUBBARD_SPIN_COUNT {
        for energy in 0..header.energy_count {
            for potential in 0..header.potential_count {
                for angular in 0..angular_count {
                    values[(spin, energy, potential, angular)] =
                        read_complex32_payload(payload, source, endian, "Hubbard LDOS gtrNN.bin")?;
                    source += 1;
                }
            }
        }
    }

    let data = HubbardLdosGtrBinData {
        point_count_declared: header.energy_count,
        horizontal_count: header.horizontal_count,
        danes_extension_count: header.danes_extension_count,
        highest_potential_index: header.highest_potential_index,
        fms_mode: header.fms_mode,
        angular_limit,
        values,
    };
    validate_hubbard_ldos_gtr_bin(&data)?;
    Ok(data)
}

/// Parse FEFF Hubbard LDOS `gtrNN.bin` bytes, inferring `lx` from the payload.
pub fn parse_hubbard_ldos_gtr_bin_inferred(bytes: &[u8]) -> Result<HubbardLdosGtrBinData> {
    let (records, endian) = record_payloads(bytes, "Hubbard LDOS gtrNN.bin")?;
    let [header, payload] = ldos_hubbard_trace_record_pair(&records, "Hubbard LDOS gtrNN.bin")?;
    let header = parse_ldos_hubbard_trace_header(header, endian, "Hubbard LDOS gtrNN.bin")?;
    let angular_limit = infer_linear_angular_limit(
        payload.len(),
        checked_product_all(&[
            COMPLEX32_BYTES,
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
        ])?,
        "Hubbard LDOS gtrNN.bin",
    )?;
    parse_hubbard_ldos_gtr_bin(bytes, angular_limit)
}

/// Render FEFF-compatible little-endian Hubbard LDOS `gtrNN.bin` bytes.
pub fn hubbard_ldos_gtr_bin_bytes(data: &HubbardLdosGtrBinData) -> Result<Vec<u8>> {
    validate_hubbard_ldos_gtr_bin(data)?;
    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX32_BYTES)?);
    for spin in 0..HUBBARD_SPIN_COUNT {
        for energy in 0..data.energy_count() {
            for potential in 0..data.potential_count() {
                for angular in 0..data.angular_count() {
                    push_complex32_payload(
                        &mut payload,
                        data.values[(spin, energy, potential, angular)],
                    );
                }
            }
        }
    }
    ldos_hubbard_trace_bytes(
        data.point_count_declared,
        data.horizontal_count,
        data.danes_extension_count,
        data.highest_potential_index,
        data.fms_mode,
        &payload,
    )
}

/// Read FEFF Hubbard LDOS `gtrNN.bin` from a file.
pub fn read_hubbard_ldos_gtr_bin(
    path: impl AsRef<Path>,
    angular_limit: usize,
) -> Result<HubbardLdosGtrBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_hubbard_ldos_gtr_bin(&bytes, angular_limit)
}

/// Read FEFF Hubbard LDOS `gtrNN.bin` from a file, inferring `lx`.
pub fn read_hubbard_ldos_gtr_bin_inferred(path: impl AsRef<Path>) -> Result<HubbardLdosGtrBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_hubbard_ldos_gtr_bin_inferred(&bytes)
}

/// Write FEFF Hubbard LDOS `gtrNN.bin` bytes to a file.
pub fn write_hubbard_ldos_gtr_bin(
    path: impl AsRef<Path>,
    data: &HubbardLdosGtrBinData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, hubbard_ldos_gtr_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Parse FEFF Hubbard LDOS `gtr_mNN.bin` bytes.
pub fn parse_hubbard_ldos_gtr_m_bin(
    bytes: &[u8],
    angular_limit: usize,
) -> Result<HubbardLdosGtrMBinData> {
    let (records, endian) = record_payloads(bytes, "Hubbard LDOS gtr_mNN.bin")?;
    let [header, payload] = ldos_hubbard_trace_record_pair(&records, "Hubbard LDOS gtr_mNN.bin")?;
    let header = parse_ldos_hubbard_trace_header(header, endian, "Hubbard LDOS gtr_mNN.bin")?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let magnetic_count = magnetic_count_from_limit(angular_limit)?;
    let expected_len = checked_product(
        checked_product_all(&[
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            magnetic_count,
        ])?,
        COMPLEX32_BYTES,
    )?;
    if payload.len() != expected_len {
        return invalid_hubbard_bin(format!(
            "Hubbard LDOS gtr_mNN.bin trace payload has {} byte(s), expected {expected_len}",
            payload.len()
        ));
    }

    let mut values = Array5::from_elem(
        (
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            angular_count,
            magnetic_count,
        ),
        Complex32::new(0.0, 0.0),
    );
    let mut source = 0_usize;
    for spin in 0..HUBBARD_SPIN_COUNT {
        for energy in 0..header.energy_count {
            for potential in 0..header.potential_count {
                for angular in 0..angular_count {
                    for magnetic in angular_magnetic_range(angular)? {
                        values[(spin, energy, potential, angular, magnetic)] =
                            read_complex32_payload(
                                payload,
                                source,
                                endian,
                                "Hubbard LDOS gtr_mNN.bin",
                            )?;
                        source += 1;
                    }
                }
            }
        }
    }

    let data = HubbardLdosGtrMBinData {
        point_count_declared: header.energy_count,
        horizontal_count: header.horizontal_count,
        danes_extension_count: header.danes_extension_count,
        highest_potential_index: header.highest_potential_index,
        fms_mode: header.fms_mode,
        angular_limit,
        values,
    };
    validate_hubbard_ldos_gtr_m_bin(&data)?;
    Ok(data)
}

/// Parse FEFF Hubbard LDOS `gtr_mNN.bin` bytes, inferring `lx`.
pub fn parse_hubbard_ldos_gtr_m_bin_inferred(bytes: &[u8]) -> Result<HubbardLdosGtrMBinData> {
    let (records, endian) = record_payloads(bytes, "Hubbard LDOS gtr_mNN.bin")?;
    let [header, payload] = ldos_hubbard_trace_record_pair(&records, "Hubbard LDOS gtr_mNN.bin")?;
    let header = parse_ldos_hubbard_trace_header(header, endian, "Hubbard LDOS gtr_mNN.bin")?;
    let angular_limit = infer_square_angular_limit(
        payload.len(),
        checked_product_all(&[
            COMPLEX32_BYTES,
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
        ])?,
        "Hubbard LDOS gtr_mNN.bin",
    )?;
    parse_hubbard_ldos_gtr_m_bin(bytes, angular_limit)
}

/// Render FEFF-compatible little-endian Hubbard LDOS `gtr_mNN.bin` bytes.
pub fn hubbard_ldos_gtr_m_bin_bytes(data: &HubbardLdosGtrMBinData) -> Result<Vec<u8>> {
    validate_hubbard_ldos_gtr_m_bin(data)?;
    let mut payload = Vec::with_capacity(checked_product(
        checked_product_all(&[
            HUBBARD_SPIN_COUNT,
            data.energy_count(),
            data.potential_count(),
            data.magnetic_count(),
        ])?,
        COMPLEX32_BYTES,
    )?);
    for spin in 0..HUBBARD_SPIN_COUNT {
        for energy in 0..data.energy_count() {
            for potential in 0..data.potential_count() {
                for angular in 0..data.angular_count() {
                    for magnetic in angular_magnetic_range(angular)? {
                        push_complex32_payload(
                            &mut payload,
                            data.values[(spin, energy, potential, angular, magnetic)],
                        );
                    }
                }
            }
        }
    }
    ldos_hubbard_trace_bytes(
        data.point_count_declared,
        data.horizontal_count,
        data.danes_extension_count,
        data.highest_potential_index,
        data.fms_mode,
        &payload,
    )
}

/// Read FEFF Hubbard LDOS `gtr_mNN.bin` from a file.
pub fn read_hubbard_ldos_gtr_m_bin(
    path: impl AsRef<Path>,
    angular_limit: usize,
) -> Result<HubbardLdosGtrMBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_hubbard_ldos_gtr_m_bin(&bytes, angular_limit)
}

/// Read FEFF Hubbard LDOS `gtr_mNN.bin` from a file, inferring `lx`.
pub fn read_hubbard_ldos_gtr_m_bin_inferred(
    path: impl AsRef<Path>,
) -> Result<HubbardLdosGtrMBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_hubbard_ldos_gtr_m_bin_inferred(&bytes)
}

/// Write FEFF Hubbard LDOS `gtr_mNN.bin` bytes to a file.
pub fn write_hubbard_ldos_gtr_m_bin(
    path: impl AsRef<Path>,
    data: &HubbardLdosGtrMBinData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, hubbard_ldos_gtr_m_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

/// Select one potential's FEFF Hubbard `gtr_m(l,im,is,iph,ie)` trace.
pub fn hubbard_ldos_gtr_m_trace_handoff(
    data: &HubbardLdosGtrMBinData,
    potential_index: usize,
) -> Result<HubbardLdosGtrMTraceHandoff> {
    validate_hubbard_ldos_gtr_m_bin(data)?;
    if potential_index >= data.potential_count() {
        return invalid_hubbard_bin(format!(
            "requested potential index {potential_index} is outside gtr_mNN.bin potential count {}",
            data.potential_count()
        ));
    }

    let mut trace = Array4::from_elem(
        (
            data.angular_count(),
            data.magnetic_count(),
            HUBBARD_SPIN_COUNT,
            data.energy_count(),
        ),
        Complex64::new(0.0, 0.0),
    );
    for angular in 0..data.angular_count() {
        for magnetic in angular_magnetic_range(angular)? {
            for spin in 0..HUBBARD_SPIN_COUNT {
                for energy in 0..data.energy_count() {
                    let value = data.values[(spin, energy, potential_index, angular, magnetic)];
                    trace[(angular, magnetic, spin, energy)] =
                        Complex64::new(value.re as f64, value.im as f64);
                }
            }
        }
    }

    Ok(HubbardLdosGtrMTraceHandoff {
        energy_count: data.energy_count(),
        angular_count: data.angular_count(),
        magnetic_count: data.magnetic_count(),
        potential_index,
        horizontal_count: data.horizontal_count,
        danes_extension_count: data.danes_extension_count,
        highest_potential_index: data.highest_potential_index,
        fms_mode: data.fms_mode,
        trace,
    })
}

/// Parse FEFF Hubbard LDOS `gtr_offNN.bin` bytes.
pub fn parse_hubbard_ldos_gtr_off_bin(
    bytes: &[u8],
    hubbard_l: usize,
    angular_limit: usize,
) -> Result<HubbardLdosGtrOffBinData> {
    let (records, endian) = record_payloads(bytes, "Hubbard LDOS gtr_offNN.bin")?;
    let [header, payload] = ldos_hubbard_trace_record_pair(&records, "Hubbard LDOS gtr_offNN.bin")?;
    let header = parse_ldos_hubbard_trace_header(header, endian, "Hubbard LDOS gtr_offNN.bin")?;
    let angular_count = angular_count_from_limit(angular_limit)?;
    let order = hubbard_offdiag_order_from_l(hubbard_l)?;
    let expected_len = checked_product(
        checked_product_all(&[
            angular_count,
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            order,
            order,
        ])?,
        COMPLEX32_BYTES,
    )?;
    if payload.len() != expected_len {
        return invalid_hubbard_bin(format!(
            "Hubbard LDOS gtr_offNN.bin trace payload has {} byte(s), expected {expected_len}",
            payload.len()
        ));
    }

    let mut values = Array6::from_elem(
        (
            angular_count,
            HUBBARD_SPIN_COUNT,
            header.energy_count,
            header.potential_count,
            order,
            order,
        ),
        Complex32::new(0.0, 0.0),
    );
    let mut source = 0_usize;
    for angular in 0..angular_count {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for energy in 0..header.energy_count {
                for potential in 0..header.potential_count {
                    for column in 0..order {
                        for row in 0..order {
                            values[(angular, spin, energy, potential, row, column)] =
                                read_complex32_payload(
                                    payload,
                                    source,
                                    endian,
                                    "Hubbard LDOS gtr_offNN.bin",
                                )?;
                            source += 1;
                        }
                    }
                }
            }
        }
    }

    let data = HubbardLdosGtrOffBinData {
        point_count_declared: header.energy_count,
        horizontal_count: header.horizontal_count,
        danes_extension_count: header.danes_extension_count,
        highest_potential_index: header.highest_potential_index,
        fms_mode: header.fms_mode,
        hubbard_l,
        angular_limit,
        values,
    };
    validate_hubbard_ldos_gtr_off_bin(&data)?;
    Ok(data)
}

/// Render FEFF-compatible little-endian Hubbard LDOS `gtr_offNN.bin` bytes.
pub fn hubbard_ldos_gtr_off_bin_bytes(data: &HubbardLdosGtrOffBinData) -> Result<Vec<u8>> {
    validate_hubbard_ldos_gtr_off_bin(data)?;
    let mut payload = Vec::with_capacity(checked_product(data.values.len(), COMPLEX32_BYTES)?);
    for angular in 0..data.angular_count() {
        for spin in 0..HUBBARD_SPIN_COUNT {
            for energy in 0..data.energy_count() {
                for potential in 0..data.potential_count() {
                    for column in 0..data.order() {
                        for row in 0..data.order() {
                            push_complex32_payload(
                                &mut payload,
                                data.values[(angular, spin, energy, potential, row, column)],
                            );
                        }
                    }
                }
            }
        }
    }
    ldos_hubbard_trace_bytes(
        data.point_count_declared,
        data.horizontal_count,
        data.danes_extension_count,
        data.highest_potential_index,
        data.fms_mode,
        &payload,
    )
}

/// Read FEFF Hubbard LDOS `gtr_offNN.bin` from a file.
pub fn read_hubbard_ldos_gtr_off_bin(
    path: impl AsRef<Path>,
    hubbard_l: usize,
    angular_limit: usize,
) -> Result<HubbardLdosGtrOffBinData> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| IoError::io(path, source))?;
    parse_hubbard_ldos_gtr_off_bin(&bytes, hubbard_l, angular_limit)
}

/// Write FEFF Hubbard LDOS `gtr_offNN.bin` bytes to a file.
pub fn write_hubbard_ldos_gtr_off_bin(
    path: impl AsRef<Path>,
    data: &HubbardLdosGtrOffBinData,
) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, hubbard_ldos_gtr_off_bin_bytes(data)?)
        .map_err(|source| IoError::io(path, source))
}

#[derive(Debug, Clone, Copy)]
struct LdosHubbardTraceHeader {
    energy_count: usize,
    horizontal_count: usize,
    danes_extension_count: usize,
    highest_potential_index: usize,
    potential_count: usize,
    fms_mode: i32,
}

fn ldos_hubbard_trace_record_pair<'a>(
    records: &'a [&'a [u8]],
    name: &'static str,
) -> Result<[&'a [u8]; 2]> {
    match records {
        [header, payload] => Ok([*header, *payload]),
        _ => invalid_hubbard_bin(format!(
            "{name} must contain exactly two Fortran records, found {}",
            records.len()
        )),
    }
}

fn parse_ldos_hubbard_trace_header(
    header: &[u8],
    endian: Endian,
    name: &'static str,
) -> Result<LdosHubbardTraceHeader> {
    if header.len() != HEADER_RECORD_BYTES {
        return invalid_hubbard_bin(format!(
            "{name} header record has {} byte(s), expected {HEADER_RECORD_BYTES}",
            header.len()
        ));
    }
    let energy_count = parse_nonnegative_i32(read_i32(header, 0, endian)?, "ne")?;
    let horizontal_count = parse_nonnegative_i32(read_i32(header, INTEGER_BYTES, endian)?, "ne1")?;
    let danes_extension_count =
        parse_nonnegative_i32(read_i32(header, INTEGER_BYTES * 2, endian)?, "ne3")?;
    let highest_potential_index =
        parse_nonnegative_i32(read_i32(header, INTEGER_BYTES * 3, endian)?, "nph")?;
    let fms_mode = read_i32(header, INTEGER_BYTES * 4, endian)?;
    validate_positive_dimension("ne", energy_count)?;
    let potential_count = checked_add(highest_potential_index, 1)?;
    Ok(LdosHubbardTraceHeader {
        energy_count,
        horizontal_count,
        danes_extension_count,
        highest_potential_index,
        potential_count,
        fms_mode,
    })
}

fn ldos_hubbard_trace_bytes(
    energy_count: usize,
    horizontal_count: usize,
    danes_extension_count: usize,
    highest_potential_index: usize,
    fms_mode: i32,
    payload: &[u8],
) -> Result<Vec<u8>> {
    validate_positive_dimension("ne", energy_count)?;
    let mut header = Vec::with_capacity(HEADER_RECORD_BYTES);
    push_i32_payload(&mut header, energy_count, "ne")?;
    push_i32_payload(&mut header, horizontal_count, "ne1")?;
    push_i32_payload(&mut header, danes_extension_count, "ne3")?;
    push_i32_payload(&mut header, highest_potential_index, "nph")?;
    header.extend_from_slice(&fms_mode.to_le_bytes());

    let mut bytes = Vec::new();
    write_record(&mut bytes, &header)?;
    write_record(&mut bytes, payload)?;
    Ok(bytes)
}

fn validate_hubbard_ldos_gtr_bin(data: &HubbardLdosGtrBinData) -> Result<()> {
    let expected = vec![
        HUBBARD_SPIN_COUNT,
        data.point_count_declared,
        checked_add(data.highest_potential_index, 1)?,
        angular_count_from_limit(data.angular_limit)?,
    ];
    validate_shape("Hubbard LDOS gtrNN.bin", data.values.shape(), &expected)?;
    validate_positive_dimension("ne", data.energy_count())?;
    validate_complex32_values("Hubbard LDOS gtrNN.bin", data.values.iter())?;
    Ok(())
}

fn validate_hubbard_ldos_gtr_m_bin(data: &HubbardLdosGtrMBinData) -> Result<()> {
    let expected = vec![
        HUBBARD_SPIN_COUNT,
        data.point_count_declared,
        checked_add(data.highest_potential_index, 1)?,
        angular_count_from_limit(data.angular_limit)?,
        magnetic_count_from_limit(data.angular_limit)?,
    ];
    validate_shape("Hubbard LDOS gtr_mNN.bin", data.values.shape(), &expected)?;
    validate_positive_dimension("ne", data.energy_count())?;
    validate_complex32_values("Hubbard LDOS gtr_mNN.bin", data.values.iter())?;
    validate_zero_unused_hubbard_magnetic_slots("Hubbard LDOS gtr_mNN.bin", data.values.view())?;
    Ok(())
}

fn validate_hubbard_ldos_gtr_off_bin(data: &HubbardLdosGtrOffBinData) -> Result<()> {
    let order = hubbard_offdiag_order_from_l(data.hubbard_l)?;
    let expected = vec![
        angular_count_from_limit(data.angular_limit)?,
        HUBBARD_SPIN_COUNT,
        data.point_count_declared,
        checked_add(data.highest_potential_index, 1)?,
        order,
        order,
    ];
    validate_shape("Hubbard LDOS gtr_offNN.bin", data.values.shape(), &expected)?;
    validate_positive_dimension("ne", data.energy_count())?;
    validate_complex32_values("Hubbard LDOS gtr_offNN.bin", data.values.iter())?;
    Ok(())
}

fn validate_zero_unused_hubbard_magnetic_slots(
    label: &'static str,
    values: ndarray::ArrayView5<'_, Complex32>,
) -> Result<()> {
    for angular in 0..values.len_of(Axis(3)) {
        let used = angular_magnetic_range(angular)?;
        for spin in 0..values.len_of(Axis(0)) {
            for energy in 0..values.len_of(Axis(1)) {
                for potential in 0..values.len_of(Axis(2)) {
                    for magnetic in 0..values.len_of(Axis(4)) {
                        if used.contains(&magnetic) {
                            continue;
                        }
                        let value = values[(spin, energy, potential, angular, magnetic)];
                        if value != Complex32::new(0.0, 0.0) {
                            return invalid_hubbard_bin(format!(
                                "{label} unused magnetic slot [{spin},{energy},{potential},{angular},{magnetic}] must be zero"
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_v_hubbard_bin(data: &HubbardVnlmBinData) -> Result<()> {
    let expected = v_hubbard_shape(data.angular_limit, data.potential_count())?;
    validate_shape("v_hubbard.bin", data.values.shape(), &expected)?;
    for (index, value) in data.values.iter().enumerate() {
        if !value.is_finite() {
            return invalid_hubbard_bin(format!("v_hubbard.bin value {} is not finite", index + 1));
        }
    }
    Ok(())
}

fn validate_aphase_hubbard_bin(data: &HubbardAphaseBinData) -> Result<()> {
    validate_positive_dimension("energy_count", data.energy_count())?;
    let expected = aphase_hubbard_shape(
        data.energy_count(),
        data.angular_limit,
        data.potential_count(),
    )?;
    validate_shape("aphase_hubbard.bin", data.values.shape(), &expected)?;
    for (index, value) in data.values.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_hubbard_bin(format!(
                "aphase_hubbard.bin value {} is not finite",
                index + 1
            ));
        }
    }
    Ok(())
}

fn validate_transformation_hubbard_bin(data: &HubbardTransformationBinData) -> Result<()> {
    validate_positive_dimension("potential_count", data.potential_count())?;
    let expected =
        transformation_hubbard_shape(data.hubbard_l, data.angular_limit, data.potential_count())?;
    validate_shape(
        "transformation_hubbard.bin transform",
        data.transform.shape(),
        &expected,
    )?;
    validate_shape(
        "transformation_hubbard.bin inverse",
        data.inverse.shape(),
        &expected,
    )?;
    validate_complex32_values(
        "transformation_hubbard.bin transform",
        data.transform.iter(),
    )?;
    validate_complex32_values("transformation_hubbard.bin inverse", data.inverse.iter())?;
    Ok(())
}

fn v_hubbard_shape(angular_limit: usize, potential_count: usize) -> Result<Vec<usize>> {
    Ok(vec![
        potential_count,
        HUBBARD_SPIN_COUNT,
        angular_count_from_limit(angular_limit)?,
        magnetic_count_from_limit(angular_limit)?,
    ])
}

fn aphase_hubbard_shape(
    energy_count: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<Vec<usize>> {
    Ok(vec![
        potential_count,
        HUBBARD_SPIN_COUNT,
        energy_count,
        angular_count_from_limit(angular_limit)?,
        magnetic_count_from_limit(angular_limit)?,
    ])
}

fn transformation_hubbard_shape(
    hubbard_l: usize,
    angular_limit: usize,
    potential_count: usize,
) -> Result<Vec<usize>> {
    let order = transformation_order_from_hubbard_l(hubbard_l)?;
    Ok(vec![
        potential_count,
        HUBBARD_SPIN_COUNT,
        angular_count_from_limit(angular_limit)?,
        order,
        order,
    ])
}

fn angular_count_from_limit(angular_limit: usize) -> Result<usize> {
    checked_add(angular_limit, 1)
}

fn magnetic_count_from_limit(angular_limit: usize) -> Result<usize> {
    let angular_count = angular_count_from_limit(angular_limit)?;
    checked_product(angular_count, angular_count)
}

fn validate_shape(name: &'static str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid_hubbard_bin(format!("{name} shape is {actual:?}, expected {expected:?}"))
    }
}

fn validate_positive_dimension(name: &'static str, value: usize) -> Result<()> {
    if value > 0 {
        Ok(())
    } else {
        invalid_hubbard_bin(format!("{name} must be positive"))
    }
}

fn transformation_order_from_hubbard_l(hubbard_l: usize) -> Result<usize> {
    checked_add(checked_product(2, hubbard_l)?, 1)
}

fn infer_hubbard_angular_limit(
    payload_len: usize,
    value_bytes: usize,
    fixed_count: usize,
    name: &'static str,
) -> Result<usize> {
    validate_positive_dimension("fixed_count", fixed_count)?;
    let fixed_bytes = checked_product(value_bytes, fixed_count)?;
    if !payload_len.is_multiple_of(fixed_bytes) {
        return invalid_hubbard_bin(format!(
            "{name} payload has {payload_len} byte(s), not a whole Hubbard angular cube"
        ));
    }
    let angular_cube = payload_len / fixed_bytes;
    if angular_cube == 0 {
        return invalid_hubbard_bin(format!("{name} payload is empty"));
    }

    let mut angular_count = 1_usize;
    loop {
        let cube = checked_product(
            checked_product(angular_count, angular_count)?,
            angular_count,
        )?;
        if cube == angular_cube {
            return angular_count
                .checked_sub(1)
                .ok_or_else(|| invalid_hubbard_bin_value("Hubbard angular count underflow"));
        }
        if cube > angular_cube {
            return invalid_hubbard_bin(format!(
                "{name} payload implies angular cube {angular_cube}, which is not (lx + 1)^3"
            ));
        }
        angular_count = checked_add(angular_count, 1)?;
    }
}

fn infer_transformation_hubbard_angular_limit(
    payload_len: usize,
    hubbard_l: usize,
    potential_count: usize,
) -> Result<usize> {
    let order = transformation_order_from_hubbard_l(hubbard_l)?;
    let fixed_count = checked_product_all(&[order, order, HUBBARD_SPIN_COUNT, potential_count])?;
    let fixed_bytes = checked_product(COMPLEX32_BYTES, fixed_count)?;
    if !payload_len.is_multiple_of(fixed_bytes) {
        return invalid_hubbard_bin(format!(
            "transformation_hubbard.bin payload has {payload_len} byte(s), not a whole angular count"
        ));
    }
    let angular_count = payload_len / fixed_bytes;
    if angular_count == 0 {
        return invalid_hubbard_bin("transformation_hubbard.bin payload is empty");
    }
    angular_count
        .checked_sub(1)
        .ok_or_else(|| invalid_hubbard_bin_value("Hubbard angular count underflow"))
}

fn infer_linear_angular_limit(
    payload_len: usize,
    fixed_bytes: usize,
    name: &'static str,
) -> Result<usize> {
    validate_positive_dimension("fixed_bytes", fixed_bytes)?;
    if !payload_len.is_multiple_of(fixed_bytes) {
        return invalid_hubbard_bin(format!(
            "{name} payload has {payload_len} byte(s), not a whole angular count"
        ));
    }
    let angular_count = payload_len / fixed_bytes;
    if angular_count == 0 {
        return invalid_hubbard_bin(format!("{name} payload is empty"));
    }
    angular_count
        .checked_sub(1)
        .ok_or_else(|| invalid_hubbard_bin_value("Hubbard angular count underflow"))
}

fn infer_square_angular_limit(
    payload_len: usize,
    fixed_bytes: usize,
    name: &'static str,
) -> Result<usize> {
    validate_positive_dimension("fixed_bytes", fixed_bytes)?;
    if !payload_len.is_multiple_of(fixed_bytes) {
        return invalid_hubbard_bin(format!(
            "{name} payload has {payload_len} byte(s), not a whole magnetic square"
        ));
    }
    let magnetic_count = payload_len / fixed_bytes;
    if magnetic_count == 0 {
        return invalid_hubbard_bin(format!("{name} payload is empty"));
    }

    let mut angular_count = 1_usize;
    loop {
        let square = checked_product(angular_count, angular_count)?;
        if square == magnetic_count {
            return angular_count
                .checked_sub(1)
                .ok_or_else(|| invalid_hubbard_bin_value("Hubbard angular count underflow"));
        }
        if square > magnetic_count {
            return invalid_hubbard_bin(format!(
                "{name} payload implies magnetic count {magnetic_count}, which is not (lx + 1)^2"
            ));
        }
        angular_count = checked_add(angular_count, 1)?;
    }
}

fn hubbard_offdiag_order_from_l(hubbard_l: usize) -> Result<usize> {
    let angular_count = checked_add(hubbard_l, 1)?;
    checked_product(angular_count, angular_count)
}

fn angular_magnetic_range(angular: usize) -> Result<std::ops::Range<usize>> {
    let start = checked_product(angular, angular)?;
    let end = checked_product(checked_add(angular, 1)?, checked_add(angular, 1)?)?;
    Ok(start..end)
}

fn transformation_hubbard_payload_len(
    order: usize,
    angular_count: usize,
    potential_count: usize,
) -> Result<usize> {
    let value_count = checked_product_all(&[
        order,
        order,
        HUBBARD_SPIN_COUNT,
        angular_count,
        potential_count,
    ])?;
    checked_product(value_count, COMPLEX32_BYTES)
}

fn transformation_hubbard_record_pair<'a>(records: &'a [&'a [u8]]) -> Result<[&'a [u8]; 2]> {
    match records {
        [transform, inverse] => Ok([*transform, *inverse]),
        _ => invalid_hubbard_bin(format!(
            "transformation_hubbard.bin must contain exactly two Fortran records, found {}",
            records.len()
        )),
    }
}

fn parse_transformation_hubbard_payload(
    payload: &[u8],
    endian: Endian,
    label: &'static str,
    order: usize,
    angular_count: usize,
    potential_count: usize,
    expected_len: usize,
) -> Result<Array5<Complex32>> {
    if payload.len() != expected_len {
        return invalid_hubbard_bin(format!(
            "{label} payload has {} byte(s), expected {expected_len}",
            payload.len()
        ));
    }

    let mut values = Array5::from_elem(
        (
            potential_count,
            HUBBARD_SPIN_COUNT,
            angular_count,
            order,
            order,
        ),
        Complex32::new(0.0, 0.0),
    );
    let mut source = 0_usize;
    for potential in 0..potential_count {
        for angular in 0..angular_count {
            for spin in 0..HUBBARD_SPIN_COUNT {
                for column in 0..order {
                    for row in 0..order {
                        let offset = checked_product(source, COMPLEX32_BYTES)?;
                        let real = read_f32(payload, offset, endian)?;
                        let imaginary = read_f32(payload, offset + F32_BYTES, endian)?;
                        if !(real.is_finite() && imaginary.is_finite()) {
                            return invalid_hubbard_bin(format!(
                                "{label} value {} is not finite",
                                source + 1
                            ));
                        }
                        values[(potential, spin, angular, row, column)] =
                            Complex32::new(real, imaginary);
                        source += 1;
                    }
                }
            }
        }
    }
    Ok(values)
}

fn transformation_hubbard_payload(values: &Array5<Complex32>) -> Vec<u8> {
    let mut payload = Vec::with_capacity(values.len() * COMPLEX32_BYTES);
    for potential in 0..values.len_of(Axis(0)) {
        for angular in 0..values.len_of(Axis(2)) {
            for spin in 0..HUBBARD_SPIN_COUNT {
                for column in 0..values.len_of(Axis(4)) {
                    for row in 0..values.len_of(Axis(3)) {
                        let value = values[(potential, spin, angular, row, column)];
                        payload.extend_from_slice(&value.re.to_le_bytes());
                        payload.extend_from_slice(&value.im.to_le_bytes());
                    }
                }
            }
        }
    }
    payload
}

fn read_complex32_payload(
    payload: &[u8],
    source: usize,
    endian: Endian,
    label: &'static str,
) -> Result<Complex32> {
    let offset = checked_product(source, COMPLEX32_BYTES)?;
    let real = read_f32(payload, offset, endian)?;
    let imaginary = read_f32(payload, offset + F32_BYTES, endian)?;
    if !(real.is_finite() && imaginary.is_finite()) {
        return invalid_hubbard_bin(format!("{label} value {} is not finite", source + 1));
    }
    Ok(Complex32::new(real, imaginary))
}

fn push_complex32_payload(payload: &mut Vec<u8>, value: Complex32) {
    payload.extend_from_slice(&value.re.to_le_bytes());
    payload.extend_from_slice(&value.im.to_le_bytes());
}

fn validate_complex32_values<'a>(
    label: &'static str,
    values: impl Iterator<Item = &'a Complex32>,
) -> Result<()> {
    for (index, value) in values.enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return invalid_hubbard_bin(format!("{label} value {} is not finite", index + 1));
        }
    }
    Ok(())
}

fn parse_nonnegative_i32(value: i32, field: &'static str) -> Result<usize> {
    if value < 0 {
        return invalid_hubbard_bin(format!("{field} must be non-negative, got {value}"));
    }
    usize::try_from(value)
        .map_err(|_| invalid_hubbard_bin_value(format!("{field} does not fit usize")))
}

fn push_i32_payload(payload: &mut Vec<u8>, value: usize, field: &'static str) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid_hubbard_bin_value(format!("{field} does not fit i32")))?;
    payload.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn detect_payload_endian(bytes: &[u8], payload_len: usize) -> Result<Endian> {
    let marker = read_marker_bytes(bytes, 0)?;
    let little = u32::from_le_bytes(marker);
    if little as usize == payload_len {
        return Ok(Endian::Little);
    }
    let big = u32::from_be_bytes(marker);
    if big as usize == payload_len {
        return Ok(Endian::Big);
    }
    invalid_hubbard_bin(format!(
        "record marker {} did not match expected payload length {payload_len}",
        little
    ))
}

fn record_payloads<'a>(bytes: &'a [u8], name: &'static str) -> Result<(Vec<&'a [u8]>, Endian)> {
    for endian in [Endian::Little, Endian::Big] {
        if let Ok(records) = record_payloads_for_endian(bytes, endian, name) {
            return Ok((records, endian));
        }
    }
    invalid_hubbard_bin(format!("{name} does not contain complete Fortran records"))
}

fn record_payloads_for_endian<'a>(
    bytes: &'a [u8],
    endian: Endian,
    name: &'static str,
) -> Result<Vec<&'a [u8]>> {
    if bytes.is_empty() {
        return invalid_hubbard_bin(format!("{name} is empty"));
    }
    let mut position = 0_usize;
    let mut records = Vec::new();
    while position < bytes.len() {
        records.push(read_record(bytes, &mut position, endian, name)?);
    }
    Ok(records)
}

fn single_record_payload_metadata(bytes: &[u8], name: &'static str) -> Result<(usize, Endian)> {
    let little = single_record_payload_len_for_endian(bytes, Endian::Little)?;
    if let Some(payload_len) = little {
        return Ok((payload_len, Endian::Little));
    }
    let big = single_record_payload_len_for_endian(bytes, Endian::Big)?;
    if let Some(payload_len) = big {
        return Ok((payload_len, Endian::Big));
    }
    invalid_hubbard_bin(format!(
        "{name} does not contain one complete Fortran record"
    ))
}

fn single_record_payload_len_for_endian(bytes: &[u8], endian: Endian) -> Result<Option<usize>> {
    if bytes.len() < FORTRAN_MARKER_BYTES * 2 {
        return Ok(None);
    }
    let length = read_marker(bytes, 0, endian, "record")? as usize;
    let trailing_offset = checked_add(FORTRAN_MARKER_BYTES, length)?;
    let total = checked_add(trailing_offset, FORTRAN_MARKER_BYTES)?;
    if total != bytes.len() {
        return Ok(None);
    }
    let trailing = read_marker(bytes, trailing_offset, endian, "record")? as usize;
    if trailing == length {
        Ok(Some(length))
    } else {
        Ok(None)
    }
}

fn read_single_record<'a>(
    bytes: &'a [u8],
    expected_len: usize,
    name: &'static str,
) -> Result<&'a [u8]> {
    let endian = detect_payload_endian(bytes, expected_len)?;
    let mut position = 0_usize;
    let payload = read_record(bytes, &mut position, endian, name)?;
    if payload.len() != expected_len {
        return invalid_hubbard_bin(format!(
            "{name} payload has {} byte(s), expected {expected_len}",
            payload.len()
        ));
    }
    if position != bytes.len() {
        return invalid_hubbard_bin(format!(
            "{name} has {} trailing byte(s)",
            bytes.len() - position
        ));
    }
    Ok(payload)
}

fn read_record<'a>(
    bytes: &'a [u8],
    position: &mut usize,
    endian: Endian,
    label: &'static str,
) -> Result<&'a [u8]> {
    let length = read_marker(bytes, *position, endian, label)? as usize;
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    let end = checked_add(*position, length)?;
    if end > bytes.len() {
        return invalid_hubbard_bin(format!("{label} record is truncated"));
    }
    let payload = &bytes[*position..end];
    *position = end;
    let trailing = read_marker(bytes, *position, endian, label)? as usize;
    if trailing != length {
        return invalid_hubbard_bin(format!(
            "{label} record length marker {length} does not match trailing marker {trailing}"
        ));
    }
    *position = checked_add(*position, FORTRAN_MARKER_BYTES)?;
    Ok(payload)
}

fn read_i32(bytes: &[u8], offset: usize, endian: Endian) -> Result<i32> {
    let end = checked_add(offset, INTEGER_BYTES)?;
    let raw: [u8; INTEGER_BYTES] = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_hubbard_bin_value("i32 payload is truncated"))?
        .try_into()
        .map_err(|_| invalid_hubbard_bin_value("invalid i32 payload width"))?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(raw),
        Endian::Big => i32::from_be_bytes(raw),
    })
}

fn write_record(bytes: &mut Vec<u8>, payload: &[u8]) -> Result<()> {
    let length = u32::try_from(payload.len()).map_err(|_| {
        invalid_hubbard_bin_value(format!(
            "record payload has {} byte(s), exceeding 32-bit Fortran marker capacity",
            payload.len()
        ))
    })?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&length.to_le_bytes());
    Ok(())
}

fn read_marker(bytes: &[u8], offset: usize, endian: Endian, label: &'static str) -> Result<u32> {
    let raw = read_marker_bytes(bytes, offset).map_err(|_| {
        invalid_hubbard_bin_value(format!(
            "{label} record marker is truncated at byte {offset}"
        ))
    })?;
    Ok(match endian {
        Endian::Little => u32::from_le_bytes(raw),
        Endian::Big => u32::from_be_bytes(raw),
    })
}

fn read_f32(bytes: &[u8], offset: usize, endian: Endian) -> Result<f32> {
    let end = checked_add(offset, F32_BYTES)?;
    let raw: [u8; F32_BYTES] = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_hubbard_bin_value("f32 payload is truncated"))?
        .try_into()
        .map_err(|_| invalid_hubbard_bin_value("invalid f32 payload width"))?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(raw),
        Endian::Big => f32::from_be_bytes(raw),
    })
}

fn read_marker_bytes(bytes: &[u8], offset: usize) -> Result<[u8; FORTRAN_MARKER_BYTES]> {
    let end = checked_add(offset, FORTRAN_MARKER_BYTES)?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_hubbard_bin_value("record marker is truncated"))?;
    slice
        .try_into()
        .map_err(|_| invalid_hubbard_bin_value("invalid record marker width"))
}

fn read_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<f64> {
    let end = checked_add(offset, F64_BYTES)?;
    let raw: [u8; F64_BYTES] = bytes
        .get(offset..end)
        .ok_or_else(|| invalid_hubbard_bin_value("f64 payload is truncated"))?
        .try_into()
        .map_err(|_| invalid_hubbard_bin_value("invalid f64 payload width"))?;
    Ok(match endian {
        Endian::Little => f64::from_le_bytes(raw),
        Endian::Big => f64::from_be_bytes(raw),
    })
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid_hubbard_bin_value("integer overflow"))
}

fn checked_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_hubbard_bin_value("integer overflow"))
}

fn checked_product_all(values: &[usize]) -> Result<usize> {
    values.iter().copied().try_fold(1_usize, checked_product)
}

fn invalid_hubbard_bin<T>(message: impl Into<String>) -> Result<T> {
    Err(invalid_hubbard_bin_value(message))
}

fn invalid_hubbard_bin_value(message: impl Into<String>) -> IoError {
    IoError::Parse {
        path: "hubbard.bin".into(),
        line: 0,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{Array4, Array5, Array6};

    use super::*;

    #[test]
    fn roundtrips_v_hubbard_bin_in_feff_order() -> Result<()> {
        let data = sample_v_hubbard();
        let bytes = v_hubbard_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 864);

        let reparsed = parse_v_hubbard_bin(&bytes, 2, 2)?;
        let inferred = parse_v_hubbard_bin_inferred(&bytes, 2)?;

        assert_eq!(reparsed, data);
        assert_eq!(inferred, data);
        assert_eq!(reparsed.values[(0, 0, 0, 0)], 1.0);
        assert_eq!(reparsed.values[(0, 0, 2, 8)], 27.0);
        assert_eq!(reparsed.values[(1, 1, 2, 8)], 108.0);
        assert_eq!(reparsed.potential_spin(1, 1).unwrap()[(2, 8)], 108.0);
        Ok(())
    }

    #[test]
    fn roundtrips_aphase_hubbard_bin_in_feff_order() -> Result<()> {
        let data = sample_aphase_hubbard();
        let bytes = aphase_hubbard_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1024);

        let reparsed = parse_aphase_hubbard_bin(&bytes, 2, 1, 2)?;
        let inferred = parse_aphase_hubbard_bin_inferred(&bytes, 2, 2)?;

        assert_eq!(reparsed, data);
        assert_eq!(inferred, data);
        assert_eq!(reparsed.values[(0, 0, 0, 0, 0)], Complex64::new(1.0, -1.0));
        assert_eq!(
            reparsed.values[(0, 0, 1, 1, 3)],
            Complex64::new(16.0, -16.0)
        );
        assert_eq!(
            reparsed.values[(1, 1, 1, 1, 3)],
            Complex64::new(64.0, -64.0)
        );
        Ok(())
    }

    #[test]
    fn roundtrips_transformation_hubbard_bin_in_feff_order() -> Result<()> {
        let data = sample_transformation_hubbard();
        let bytes = transformation_hubbard_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 576);
        assert_eq!(u32::from_le_bytes(bytes[584..588].try_into().unwrap()), 576);

        let reparsed = parse_transformation_hubbard_bin(&bytes, 1, 1, 2)?;
        let inferred = parse_transformation_hubbard_bin_inferred(&bytes, 1, 2)?;

        assert_eq!(reparsed, data);
        assert_eq!(inferred, data);
        assert_eq!(
            reparsed.transform[(0, 0, 0, 0, 0)],
            Complex32::new(1.0, -1.0)
        );
        assert_eq!(
            reparsed.transform[(0, 0, 1, 0, 0)],
            Complex32::new(19.0, -19.0)
        );
        assert_eq!(
            reparsed.transform[(1, 1, 1, 2, 2)],
            Complex32::new(72.0, -72.0)
        );
        assert_eq!(
            reparsed.inverse[(1, 1, 1, 2, 2)],
            Complex32::new(1072.0, -1072.0)
        );
        Ok(())
    }

    #[test]
    fn rejects_wrong_v_hubbard_dimensions() -> Result<()> {
        let bytes = v_hubbard_bin_bytes(&sample_v_hubbard())?;
        let err = parse_v_hubbard_bin(&bytes, 1, 2).expect_err("wrong lx should fail");
        assert!(err.to_string().contains("expected payload length 256"));
        Ok(())
    }

    #[test]
    fn rejects_non_cubic_inferred_hubbard_payload() -> Result<()> {
        let payload = vec![0_u8; F64_BYTES * HUBBARD_SPIN_COUNT * 2 * 2];
        let mut bytes = Vec::new();
        write_record(&mut bytes, &payload)?;

        let err = parse_v_hubbard_bin_inferred(&bytes, 2).expect_err("non-cubic lx should fail");

        assert!(err.to_string().contains("not (lx + 1)^3"));
        Ok(())
    }

    #[test]
    fn rejects_trailing_hubbard_record_bytes() -> Result<()> {
        let mut bytes = v_hubbard_bin_bytes(&sample_v_hubbard())?;
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        let err = parse_v_hubbard_bin(&bytes, 2, 2).expect_err("trailing bytes should fail");
        assert!(err.to_string().contains("trailing byte"));
        Ok(())
    }

    #[test]
    fn rejects_wrong_transformation_hubbard_l() -> Result<()> {
        let bytes = transformation_hubbard_bin_bytes(&sample_transformation_hubbard())?;
        let err =
            parse_transformation_hubbard_bin(&bytes, 2, 1, 2).expect_err("wrong l should fail");
        assert!(err.to_string().contains("expected 1600"));
        Ok(())
    }

    #[test]
    fn rejects_incomplete_transformation_hubbard_records() -> Result<()> {
        let bytes = transformation_hubbard_bin_bytes(&sample_transformation_hubbard())?;
        let first_record_end = FORTRAN_MARKER_BYTES + 576 + FORTRAN_MARKER_BYTES;
        let err = parse_transformation_hubbard_bin(&bytes[..first_record_end], 1, 1, 2)
            .expect_err("single record should fail");
        assert!(err.to_string().contains("exactly two Fortran records"));
        Ok(())
    }

    #[test]
    fn rejects_non_whole_inferred_transformation_payload() -> Result<()> {
        let payload = vec![0_u8; COMPLEX32_BYTES * 5];
        let mut bytes = Vec::new();
        write_record(&mut bytes, &payload)?;
        write_record(&mut bytes, &payload)?;

        let err = parse_transformation_hubbard_bin_inferred(&bytes, 1, 1)
            .expect_err("non-whole angular count should fail");

        assert!(err.to_string().contains("not a whole angular count"));
        Ok(())
    }

    #[test]
    fn rejects_nonfinite_hubbard_values() {
        let mut data = sample_v_hubbard();
        data.values[(0, 0, 0, 0)] = f64::NAN;
        let err = v_hubbard_bin_bytes(&data).expect_err("NaN should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn rejects_nonfinite_transformation_hubbard_values() {
        let mut data = sample_transformation_hubbard();
        data.inverse[(0, 0, 0, 0, 0)] = Complex32::new(f32::INFINITY, 0.0);
        let err = transformation_hubbard_bin_bytes(&data).expect_err("infinite value should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn roundtrips_hubbard_ldos_gtr_bin_in_feff_order() -> Result<()> {
        let data = sample_hubbard_ldos_gtr();
        let bytes = hubbard_ldos_gtr_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 20);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 128);

        let reparsed = parse_hubbard_ldos_gtr_bin(&bytes, 1)?;
        let inferred = parse_hubbard_ldos_gtr_bin_inferred(&bytes)?;

        assert_eq!(reparsed, data);
        assert_eq!(inferred, data);
        assert_eq!(reparsed.values[(0, 0, 0, 0)], Complex32::new(1.0, -1.0));
        assert_eq!(reparsed.values[(0, 0, 0, 1)], Complex32::new(2.0, -2.0));
        assert_eq!(reparsed.values[(1, 1, 1, 1)], Complex32::new(16.0, -16.0));
        Ok(())
    }

    #[test]
    fn roundtrips_hubbard_ldos_gtr_m_bin_in_feff_order() -> Result<()> {
        let data = sample_hubbard_ldos_gtr_m();
        let bytes = hubbard_ldos_gtr_m_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 256);

        let reparsed = parse_hubbard_ldos_gtr_m_bin(&bytes, 1)?;
        let inferred = parse_hubbard_ldos_gtr_m_bin_inferred(&bytes)?;

        assert_eq!(reparsed, data);
        assert_eq!(inferred, data);
        assert_eq!(reparsed.values[(0, 0, 0, 0, 0)], Complex32::new(1.0, -1.0));
        assert_eq!(reparsed.values[(0, 0, 0, 1, 1)], Complex32::new(2.0, -2.0));
        assert_eq!(reparsed.values[(0, 0, 0, 1, 3)], Complex32::new(4.0, -4.0));
        assert_eq!(
            reparsed.values[(1, 1, 1, 1, 3)],
            Complex32::new(32.0, -32.0)
        );
        Ok(())
    }

    #[test]
    fn selects_hubbard_ldos_gtr_m_trace_for_potential() -> Result<()> {
        let data = sample_hubbard_ldos_gtr_m();
        let handoff = hubbard_ldos_gtr_m_trace_handoff(&data, 1)?;

        assert_eq!(handoff.energy_count, 2);
        assert_eq!(handoff.angular_count, 2);
        assert_eq!(handoff.magnetic_count, 4);
        assert_eq!(handoff.potential_index, 1);
        assert_eq!(handoff.trace.dim(), (2, 4, 2, 2));
        assert_eq!(handoff.trace[(0, 0, 0, 0)], Complex64::new(5.0, -5.0));
        assert_eq!(handoff.trace[(1, 3, 1, 1)], Complex64::new(32.0, -32.0));
        assert_eq!(handoff.trace[(0, 1, 0, 0)], Complex64::new(0.0, 0.0));
        assert!(hubbard_ldos_gtr_m_trace_handoff(&data, 2).is_err());
        Ok(())
    }

    #[test]
    fn roundtrips_hubbard_ldos_gtr_off_bin_in_feff_order() -> Result<()> {
        let data = sample_hubbard_ldos_gtr_off();
        let bytes = hubbard_ldos_gtr_off_bin_bytes(&data)?;
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 2048);

        let reparsed = parse_hubbard_ldos_gtr_off_bin(&bytes, 1, 1)?;

        assert_eq!(reparsed, data);
        assert_eq!(
            reparsed.values[(0, 0, 0, 0, 0, 0)],
            Complex32::new(1.0, -1.0)
        );
        assert_eq!(
            reparsed.values[(0, 0, 0, 0, 1, 0)],
            Complex32::new(2.0, -2.0)
        );
        assert_eq!(
            reparsed.values[(0, 0, 0, 0, 0, 1)],
            Complex32::new(5.0, -5.0)
        );
        assert_eq!(
            reparsed.values[(1, 1, 1, 1, 3, 3)],
            Complex32::new(256.0, -256.0)
        );
        Ok(())
    }

    #[test]
    fn rejects_nonzero_unused_hubbard_ldos_gtr_m_slot() {
        let mut data = sample_hubbard_ldos_gtr_m();
        data.values[(0, 0, 0, 0, 1)] = Complex32::new(1.0, 0.0);
        let err = hubbard_ldos_gtr_m_bin_bytes(&data).expect_err("unused slot should fail");
        assert!(err.to_string().contains("unused magnetic slot"));
    }

    #[test]
    fn parses_hubbard_nio_ldos_trace_reference_zip() -> Result<()> {
        let Some(zip_path) = workspace_reference_zip("HUBBARD/NiO") else {
            eprintln!("skipping Hubbard LDOS trace reference test; NiO REFERENCE.zip not found");
            return Ok(());
        };

        let gtr_bytes = unzip_reference_file(&zip_path, "REFERENCE/gtr00.bin")?;
        let gtr = parse_hubbard_ldos_gtr_bin_inferred(&gtr_bytes)?;
        assert_eq!(gtr.energy_count(), 200);
        assert_eq!(gtr.horizontal_count, 200);
        assert_eq!(gtr.danes_extension_count, 0);
        assert_eq!(gtr.potential_count(), 3);
        assert_eq!(gtr.angular_count(), 3);
        assert_eq!(gtr.fms_mode, 2);
        assert_eq!(hubbard_ldos_gtr_bin_bytes(&gtr)?, gtr_bytes);

        let gtr_m_bytes = unzip_reference_file(&zip_path, "REFERENCE/gtr_m00.bin")?;
        let gtr_m = parse_hubbard_ldos_gtr_m_bin_inferred(&gtr_m_bytes)?;
        assert_eq!(gtr_m.energy_count(), 200);
        assert_eq!(gtr_m.potential_count(), 3);
        assert_eq!(gtr_m.angular_count(), 3);
        assert_eq!(gtr_m.magnetic_count(), 9);
        let gtr_m_handoff = hubbard_ldos_gtr_m_trace_handoff(&gtr_m, 0)?;
        assert_eq!(gtr_m_handoff.trace.dim(), (3, 9, 2, 200));
        assert_eq!(hubbard_ldos_gtr_m_bin_bytes(&gtr_m)?, gtr_m_bytes);

        let gtr_off_bytes = unzip_reference_file(&zip_path, "REFERENCE/gtr_off00.bin")?;
        let gtr_off = parse_hubbard_ldos_gtr_off_bin(&gtr_off_bytes, 2, 2)?;
        assert_eq!(gtr_off.energy_count(), 200);
        assert_eq!(gtr_off.potential_count(), 3);
        assert_eq!(gtr_off.angular_count(), 3);
        assert_eq!(gtr_off.order(), 9);
        assert_eq!(hubbard_ldos_gtr_off_bin_bytes(&gtr_off)?, gtr_off_bytes);
        Ok(())
    }

    fn sample_v_hubbard() -> HubbardVnlmBinData {
        let mut next = 1.0;
        let mut values = Array4::zeros((2, 2, 3, 9));
        for potential in 0..2 {
            for spin in 0..2 {
                for magnetic in 0..9 {
                    for angular in 0..3 {
                        values[(potential, spin, angular, magnetic)] = next;
                        next += 1.0;
                    }
                }
            }
        }
        HubbardVnlmBinData {
            angular_limit: 2,
            values,
        }
    }

    fn sample_aphase_hubbard() -> HubbardAphaseBinData {
        let mut next = 1.0;
        let mut values = Array5::from_elem((2, 2, 2, 2, 4), Complex64::new(0.0, 0.0));
        for potential in 0..2 {
            for spin in 0..2 {
                for magnetic in 0..4 {
                    for angular in 0..2 {
                        for energy in 0..2 {
                            values[(potential, spin, energy, angular, magnetic)] =
                                Complex64::new(next, -next);
                            next += 1.0;
                        }
                    }
                }
            }
        }
        HubbardAphaseBinData {
            angular_limit: 1,
            values,
        }
    }

    fn sample_transformation_hubbard() -> HubbardTransformationBinData {
        let mut next = 1.0_f32;
        let mut transform = Array5::from_elem((2, 2, 2, 3, 3), Complex32::new(0.0, 0.0));
        let mut inverse = Array5::from_elem((2, 2, 2, 3, 3), Complex32::new(0.0, 0.0));
        for potential in 0..2 {
            for angular in 0..2 {
                for spin in 0..2 {
                    for column in 0..3 {
                        for row in 0..3 {
                            transform[(potential, spin, angular, row, column)] =
                                Complex32::new(next, -next);
                            let inverse_value = 1000.0 + next;
                            inverse[(potential, spin, angular, row, column)] =
                                Complex32::new(inverse_value, -inverse_value);
                            next += 1.0;
                        }
                    }
                }
            }
        }
        HubbardTransformationBinData {
            hubbard_l: 1,
            angular_limit: 1,
            transform,
            inverse,
        }
    }

    fn sample_hubbard_ldos_gtr() -> HubbardLdosGtrBinData {
        let mut next = 1.0_f32;
        let mut values = Array4::from_elem((2, 2, 2, 2), Complex32::new(0.0, 0.0));
        for spin in 0..2 {
            for energy in 0..2 {
                for potential in 0..2 {
                    for angular in 0..2 {
                        values[(spin, energy, potential, angular)] = Complex32::new(next, -next);
                        next += 1.0;
                    }
                }
            }
        }
        HubbardLdosGtrBinData {
            point_count_declared: 2,
            horizontal_count: 2,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            angular_limit: 1,
            values,
        }
    }

    fn sample_hubbard_ldos_gtr_m() -> HubbardLdosGtrMBinData {
        let mut next = 1.0_f32;
        let mut values = Array5::from_elem((2, 2, 2, 2, 4), Complex32::new(0.0, 0.0));
        for spin in 0..2 {
            for energy in 0..2 {
                for potential in 0..2 {
                    for angular in 0..2 {
                        for magnetic in angular_magnetic_range(angular).unwrap() {
                            values[(spin, energy, potential, angular, magnetic)] =
                                Complex32::new(next, -next);
                            next += 1.0;
                        }
                    }
                }
            }
        }
        HubbardLdosGtrMBinData {
            point_count_declared: 2,
            horizontal_count: 2,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            angular_limit: 1,
            values,
        }
    }

    fn sample_hubbard_ldos_gtr_off() -> HubbardLdosGtrOffBinData {
        let mut next = 1.0_f32;
        let mut values = Array6::from_elem((2, 2, 2, 2, 4, 4), Complex32::new(0.0, 0.0));
        for angular in 0..2 {
            for spin in 0..2 {
                for energy in 0..2 {
                    for potential in 0..2 {
                        for column in 0..4 {
                            for row in 0..4 {
                                values[(angular, spin, energy, potential, row, column)] =
                                    Complex32::new(next, -next);
                                next += 1.0;
                            }
                        }
                    }
                }
            }
        }
        HubbardLdosGtrOffBinData {
            point_count_declared: 2,
            horizontal_count: 2,
            danes_extension_count: 0,
            highest_potential_index: 1,
            fms_mode: 2,
            hubbard_l: 1,
            angular_limit: 1,
            values,
        }
    }

    fn workspace_reference_zip(relative: &str) -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.parent()?.parent()?;
        let zip_path = workspace
            .join("reference-work")
            .join("golden")
            .join(relative)
            .join("REFERENCE.zip");
        zip_path.is_file().then_some(zip_path)
    }

    fn unzip_reference_file(zip_path: &PathBuf, entry: &str) -> Result<Vec<u8>> {
        let output = std::process::Command::new("unzip")
            .arg("-p")
            .arg(zip_path)
            .arg(entry)
            .output()
            .map_err(|source| IoError::io(zip_path, source))?;
        if !output.status.success() {
            return invalid_hubbard_bin(format!(
                "failed to extract {entry} from {}",
                zip_path.display()
            ));
        }
        Ok(output.stdout)
    }
}
