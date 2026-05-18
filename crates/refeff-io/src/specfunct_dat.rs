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
    Real, SFCONV_SO2CONV_BOHR_ANGSTROM, SFCONV_SO2CONV_HARTREE_EV, SfconvConvolution,
    SfconvConvolutionInput, SfconvError, SfconvExafsConvolution, SfconvExafsConvolutionInput,
    SfconvFeffPathInterpolationInput, SfconvFeffPathSignalInput,
    SfconvMomentumSpectralInterpolation, SfconvMomentumSpectralInterpolationInput,
    SfconvPathAverage, SfconvPathAverageInput, SfconvSo2convExafsPreparationInput,
    SfconvSo2convXanesPreparation, SfconvSo2convXanesPreparationInput, SfconvSpectralInterpolation,
    SfconvSpectralInterpolationInput, SfconvXanesConvolution, SfconvXanesConvolutionInput,
    sfconv_convolve, sfconv_exafs_convolution, sfconv_feff_path_signal,
    sfconv_interpolate_feff_path, sfconv_interpolate_momentum_spectral_function,
    sfconv_interpolate_spectral_function, sfconv_path_average, sfconv_so2conv_material_parameters,
    sfconv_so2conv_prepare_exafs_signal, sfconv_so2conv_prepare_xanes_signal,
    sfconv_xanes_convolution,
};

use crate::chi_dat::{ChiDatData, validate_chi_dat};
use crate::error::{IoError, Result};
use crate::sfconv_input::{
    SfconvSo2convFeffPathData, sfconv_so2conv_chi_data_from_convolution_rows,
    sfconv_so2conv_feff_path_data_from_averages, sfconv_so2conv_xmu_data_from_convolution_rows,
};
use crate::xmu_dat::{XmuDatData, validate_xmu_dat};

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

/// Inputs for convolving EXAFS rows with a `specfunct.dat` cache.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctExafsRowsInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Signal energy grid, FEFF `epts2`.
    pub signal_energy: ArrayView1<'a, Real>,
    /// Real EXAFS channel, FEFF `chir`.
    pub real_signal: ArrayView1<'a, Real>,
    /// Imaginary EXAFS channel, FEFF `chii`.
    pub imaginary_signal: ArrayView1<'a, Real>,
    /// Original EXAFS magnitude, FEFF `xmag`.
    pub original_magnitude: ArrayView1<'a, Real>,
    /// Original EXAFS phase, FEFF `phase`.
    pub original_phase: ArrayView1<'a, Real>,
    /// Original phase with `2 k R` removed, FEFF `phm2kr`.
    pub phase_minus_2kr: ArrayView1<'a, Real>,
    /// Photoelectron momentum for each active signal row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Number of target rows to convolve.
    pub active_len: usize,
    /// EXAFS convolution chemical potential, FEFF `cmu`.
    pub chemical_potential: Real,
    /// Apply FEFF's available-energy cutoff, FEFF `icut`.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for convolving prepared XANES rows with a `specfunct.dat` cache.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctXanesRowsInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Prepared padded XANES arrays from the core SO2CONV signal-preparation step.
    pub prepared: &'a SfconvSo2convXanesPreparation,
    /// Photoelectron momentum for each active signal row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Number of target rows to convolve.
    pub active_len: usize,
    /// XANES convolution chemical potential, FEFF `cmu + vint`.
    pub chemical_potential: Real,
    /// Apply FEFF's available-energy cutoff, FEFF `icut`.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `chi.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctChiDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `chi.dat` or `chipNNNN.dat` rows before many-body convolution.
    pub source: &'a ChiDatData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each source row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's padded EXAFS work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `feffNNNN.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctFeffPathDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `feffNNNN.dat` path rows before many-body convolution.
    pub source: &'a SfconvSo2convFeffPathData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each dense uniform path row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's dense uniform path work arrays, FEFF `npts2`.
    pub work_len: usize,
}

/// Inputs for applying a cached `specfunct.dat` convolution to one `xmu.dat`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpecfunctXmuDataInput<'a> {
    /// Parsed SO2CONV spectral-function cache.
    pub cache: &'a SfconvSpecfunctData,
    /// Source `xmu.dat` rows before many-body convolution.
    pub source: &'a XmuDatData,
    /// Material constants scanned from the source FEFF spectrum header.
    pub material: refeff_core::SfconvSo2convMaterialInput,
    /// Corrected photoelectron momentum for each source row, FEFF `pk`.
    pub photoelectron_momentum: ArrayView1<'a, Real>,
    /// Length of FEFF's padded XANES work arrays, FEFF `npts2`.
    pub work_len: usize,
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

/// Convolve prepared XANES rows with cached SO2CONV spectral functions.
///
/// This is the row-level bridge from a reusable `specfunct.dat` cache to the
/// existing core XANES convolution kernels. The returned rows can be applied to
/// `xmu.dat` with `sfconv_so2conv_xmu_data_from_convolution_rows`.
pub fn sfconv_specfunct_xanes_convolution_rows(
    input: SfconvSpecfunctXanesRowsInput<'_>,
) -> Result<Vec<SfconvXanesConvolution>> {
    validate_specfunct_xanes_rows_input(input)?;
    let asymmetric_phase = input.cache.asymmetric_phase != 0;
    (0..input.active_len)
        .map(|row| sfconv_specfunct_xanes_convolution_row(input, row, asymmetric_phase))
        .collect()
}

/// Build a convolved `chi.dat`/`chipNNNN.dat` from a compatible cached `specfunct.dat`.
///
/// This helper performs the FEFF `SO2CONV` EXAFS unit handoff, converts the
/// source inverse-Angstrom wave-number grid into atomic units, pads the energy
/// grid with the same endpoint rule as `so2conv.f90`, applies cached
/// spectral-function convolution rows, and returns a FEFF-style five-column
/// EXAFS table.
pub fn sfconv_specfunct_chi_data_from_cache(
    input: SfconvSpecfunctChiDataInput<'_>,
) -> Result<ChiDatData> {
    validate_specfunct_chi_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_exafs_error)?;
    let momentum = input
        .source
        .wave_number
        .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM);
    let prepared = sfconv_so2conv_prepare_exafs_signal(SfconvSo2convExafsPreparationInput {
        momentum: momentum.view(),
        magnitude: input.source.magnitude.view(),
        phase: input.source.phase.view(),
        phase_minus_2kr: input
            .source
            .phase_minus_2kr
            .as_ref()
            .map(|values| values.view()),
        chemical_potential: material.chemical_potential_offset,
        active_len: input.source.point_count(),
        output_len: input.work_len,
    })
    .map_err(specfunct_exafs_error)?;
    let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
        cache: input.cache,
        signal_energy: prepared.signal_energy.view(),
        real_signal: prepared.real_signal.view(),
        imaginary_signal: prepared.imaginary_signal.view(),
        original_magnitude: prepared.original_magnitude.view(),
        original_phase: prepared.original_phase.view(),
        phase_minus_2kr: prepared.phase_minus_2kr.view(),
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.source.point_count(),
        chemical_potential: material.chemical_potential_offset,
        cutoff: true,
        plasma_frequency: material.plasma_frequency,
    })?;
    sfconv_so2conv_chi_data_from_convolution_rows(input.source, &rows)
}

/// Build a convolved `feffNNNN.dat` path table from a cached `specfunct.dat`.
///
/// FEFF `SO2CONV` first maps the coarse path table onto a dense 0.05
/// inverse-Angstrom grid, convolves that raw EXAFS path signal, then averages
/// the many-body amplitude and phase corrections back onto the original path
/// grid. This helper performs that cache-backed path assembly and preserves the
/// original path table columns except for FEFF's `redfac2` and `caph2`
/// corrections.
pub fn sfconv_specfunct_feff_path_data_from_cache(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<SfconvSo2convFeffPathData> {
    validate_specfunct_feff_path_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_exafs_error)?;
    let source_momentum = sfconv_specfunct_uniform_path_momentum(input.work_len);
    let path_momentum = input
        .source
        .wave_number_inverse_angstrom
        .mapv(|value| value / SFCONV_SO2CONV_BOHR_ANGSTROM);
    let effective_amplitude = input
        .source
        .effective_amplitude
        .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM);
    let mean_free_path = input
        .source
        .mean_free_path_angstrom
        .mapv(|value| value * SFCONV_SO2CONV_BOHR_ANGSTROM);

    let interpolated = sfconv_interpolate_feff_path(SfconvFeffPathInterpolationInput {
        source_momentum: source_momentum.view(),
        path_momentum: path_momentum.view(),
        central_phase: input.source.central_phase.view(),
        effective_amplitude: effective_amplitude.view(),
        effective_phase: input.source.effective_phase.view(),
        reduction_factor: input.source.reduction_factor.view(),
        mean_free_path: mean_free_path.view(),
    })
    .map_err(specfunct_exafs_error)?;
    let signal = sfconv_feff_path_signal(SfconvFeffPathSignalInput {
        momentum: source_momentum.view(),
        central_phase: interpolated.central_phase.view(),
        effective_amplitude: interpolated.effective_amplitude.view(),
        effective_phase: interpolated.effective_phase.view(),
        reduction_factor: interpolated.reduction_factor.view(),
        mean_free_path: interpolated.mean_free_path.view(),
        degeneracy: input.source.degeneracy,
        half_path_length: input.source.effective_half_path_length_angstrom
            * SFCONV_SO2CONV_BOHR_ANGSTROM,
    })
    .map_err(specfunct_exafs_error)?;
    let signal_energy = source_momentum
        .mapv(|momentum| momentum.powi(2) / 2.0 + material.chemical_potential_offset);
    let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
        cache: input.cache,
        signal_energy: signal_energy.view(),
        real_signal: signal.real.view(),
        imaginary_signal: signal.imaginary.view(),
        original_magnitude: signal.magnitude.view(),
        original_phase: signal.phase.view(),
        phase_minus_2kr: signal.phase_minus_2kr.view(),
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.work_len,
        chemical_potential: material.chemical_potential_offset,
        cutoff: true,
        plasma_frequency: material.plasma_frequency,
    })?;
    let averages =
        sfconv_specfunct_feff_path_averages(source_momentum.view(), path_momentum.view(), &rows)?;
    sfconv_so2conv_feff_path_data_from_averages(input.source, &averages)
}

/// Build a convolved `xmu.dat` from a compatible cached `specfunct.dat`.
///
/// This helper performs the FEFF `SO2CONV` XANES unit handoff, pads the signal
/// arrays with the same endpoint rule as `so2conv.f90`, applies cached
/// spectral-function convolution rows, and returns an `xmu.dat` table with the
/// original energy and wave-number columns preserved.
pub fn sfconv_specfunct_xmu_data_from_cache(
    input: SfconvSpecfunctXmuDataInput<'_>,
) -> Result<XmuDatData> {
    validate_specfunct_xmu_data_input(input)?;
    let material =
        sfconv_so2conv_material_parameters(input.material).map_err(specfunct_xanes_error)?;
    let incident_energy = input
        .source
        .photon_energy_ev
        .mapv(|value| value / SFCONV_SO2CONV_HARTREE_EV);
    let excitation_energy = input
        .source
        .relative_energy_ev
        .mapv(|value| value / SFCONV_SO2CONV_HARTREE_EV);
    let prepared = sfconv_so2conv_prepare_xanes_signal(SfconvSo2convXanesPreparationInput {
        incident_energy: incident_energy.view(),
        excitation_energy: excitation_energy.view(),
        absorption: input.source.mu.view(),
        embedded_background: input.source.mu0.view(),
        active_len: input.source.point_count(),
        output_len: input.work_len,
    })
    .map_err(specfunct_xanes_error)?;
    let rows = sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
        cache: input.cache,
        prepared: &prepared,
        photoelectron_momentum: input.photoelectron_momentum,
        active_len: input.source.point_count(),
        chemical_potential: material.chemical_potential_offset + material.interstitial_potential,
        cutoff: false,
        plasma_frequency: material.plasma_frequency,
    })?;
    sfconv_so2conv_xmu_data_from_convolution_rows(input.source, &rows)
}

/// Convolve EXAFS rows with cached SO2CONV spectral functions.
///
/// This is the row-level bridge from a reusable `specfunct.dat` cache to the
/// existing core EXAFS convolution kernels. The returned rows can be applied to
/// `chi.dat`/`chipNNNN.dat` with `sfconv_so2conv_chi_data_from_convolution_rows`
/// or averaged for `feffNNNN.dat` path data.
pub fn sfconv_specfunct_exafs_convolution_rows(
    input: SfconvSpecfunctExafsRowsInput<'_>,
) -> Result<Vec<SfconvExafsConvolution>> {
    validate_specfunct_exafs_rows_input(input)?;
    let mut previous_phase = 0.0;
    let mut phase_jump_count = 0;
    let mut rows = Vec::with_capacity(input.active_len);

    for row in 0..input.active_len {
        let output =
            sfconv_specfunct_exafs_convolution_row(input, row, previous_phase, phase_jump_count)?;
        previous_phase = output.previous_phase;
        phase_jump_count = output.phase_jump_count;
        rows.push(output);
    }

    Ok(rows)
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

fn validate_specfunct_exafs_rows_input(input: SfconvSpecfunctExafsRowsInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.chemical_potential, "exafs chemical potential")?;
    validate_finite_scalar(input.plasma_frequency, "exafs plasma frequency")?;
    if input.cache.asymmetric_phase != 0 {
        return invalid_specfunct_dat("EXAFS convolution requires an iasym=0 specfunct.dat cache");
    }
    if input.active_len == 0 {
        return invalid_specfunct_dat("EXAFS convolution active_len must be positive");
    }

    let signal_len = input.signal_energy.len();
    if signal_len < 2 {
        return invalid_specfunct_dat("EXAFS convolution signal length must be at least 2");
    }
    if input.active_len > signal_len {
        return invalid_specfunct_dat(format!(
            "EXAFS convolution active_len {} exceeds signal length {signal_len}",
            input.active_len
        ));
    }
    if input.active_len > input.photoelectron_momentum.len() {
        return invalid_specfunct_dat(format!(
            "EXAFS convolution active_len {} exceeds photoelectron momentum length {}",
            input.active_len,
            input.photoelectron_momentum.len()
        ));
    }

    validate_exafs_view(input.real_signal, signal_len, "exafs real signal")?;
    validate_exafs_view(input.imaginary_signal, signal_len, "exafs imaginary signal")?;
    validate_exafs_view(
        input.original_magnitude,
        signal_len,
        "exafs original magnitude",
    )?;
    validate_exafs_view(input.original_phase, signal_len, "exafs original phase")?;
    validate_exafs_view(input.phase_minus_2kr, signal_len, "exafs phase minus 2kr")?;
    validate_finite_view(input.signal_energy, "exafs signal energy")?;
    validate_finite_view(input.photoelectron_momentum, "exafs photoelectron momentum")?;
    Ok(())
}

fn validate_exafs_view(
    values: ArrayView1<'_, Real>,
    expected_len: usize,
    field: &'static str,
) -> Result<()> {
    if values.len() != expected_len {
        return invalid_specfunct_dat(format!(
            "{field} length {} does not match signal length {expected_len}",
            values.len()
        ));
    }
    validate_finite_view(values, field)
}

fn sfconv_specfunct_exafs_convolution_row(
    input: SfconvSpecfunctExafsRowsInput<'_>,
    row: usize,
    previous_phase: Real,
    phase_jump_count: i32,
) -> Result<SfconvExafsConvolution> {
    let momentum =
        sfconv_specfunct_interpolate_momentum(input.cache, input.photoelectron_momentum[row])?;
    let spectral = sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
        energy: momentum.energy.view(),
        spectral_function: momentum.spectral_function.view(),
        output_len: input.cache.spectral_point_count(),
    })
    .map_err(specfunct_exafs_error)?;

    let real = sfconv_specfunct_exafs_convolve_signal(
        input,
        row,
        &momentum,
        &spectral,
        input.real_signal,
    )?;
    let imaginary = sfconv_specfunct_exafs_convolve_signal(
        input,
        row,
        &momentum,
        &spectral,
        input.imaginary_signal,
    )?;

    sfconv_exafs_convolution(SfconvExafsConvolutionInput {
        real_convolution_amplitude: real.amplitude,
        real_convolution_phase: real.phase,
        imaginary_convolution_amplitude: imaginary.amplitude,
        imaginary_convolution_phase: imaginary.phase,
        original_magnitude: input.original_magnitude[row],
        original_phase: input.original_phase[row],
        phase_minus_2kr: input.phase_minus_2kr[row],
        previous_phase,
        phase_jump_count,
    })
    .map_err(specfunct_exafs_error)
}

fn sfconv_specfunct_exafs_convolve_signal(
    input: SfconvSpecfunctExafsRowsInput<'_>,
    row: usize,
    momentum: &SfconvMomentumSpectralInterpolation,
    spectral: &SfconvSpectralInterpolation,
    signal: ArrayView1<'_, Real>,
) -> Result<SfconvConvolution> {
    sfconv_convolve(SfconvConvolutionInput {
        photoelectron_energy: input.signal_energy[row],
        chemical_potential: input.chemical_potential,
        core_hole_lifetime: input.cache.core_hole_lifetime,
        signal_energy: input.signal_energy,
        signal,
        spectral_energy: spectral.energy.view(),
        spectral_function: spectral.spectral_function.view(),
        weights: momentum.weights.view(),
        asymmetric_phase: false,
        cutoff: input.cutoff,
        plasma_frequency: input.plasma_frequency,
    })
    .map_err(specfunct_exafs_error)
}

fn specfunct_exafs_error(source: SfconvError) -> IoError {
    IoError::SpecfunctDatExafsConvolution { source }
}

fn validate_specfunct_xanes_rows_input(input: SfconvSpecfunctXanesRowsInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.chemical_potential, "xanes chemical potential")?;
    validate_finite_scalar(input.plasma_frequency, "xanes plasma frequency")?;
    if input.active_len == 0 {
        return invalid_specfunct_dat("XANES convolution active_len must be positive");
    }

    let signal_len = input.prepared.excitation_energy.len();
    if signal_len < 2 {
        return invalid_specfunct_dat("XANES convolution signal length must be at least 2");
    }
    if input.active_len > signal_len {
        return invalid_specfunct_dat(format!(
            "XANES convolution active_len {} exceeds prepared signal length {signal_len}",
            input.active_len
        ));
    }
    if input.active_len > input.photoelectron_momentum.len() {
        return invalid_specfunct_dat(format!(
            "XANES convolution active_len {} exceeds photoelectron momentum length {}",
            input.active_len,
            input.photoelectron_momentum.len()
        ));
    }

    validate_vector_shape(
        &input.prepared.incident_energy,
        signal_len,
        "xanes incident energy",
    )?;
    validate_vector_shape(&input.prepared.absorption, signal_len, "xanes absorption")?;
    validate_vector_shape(
        &input.prepared.embedded_background,
        signal_len,
        "xanes embedded background",
    )?;
    validate_vector_shape(
        &input.prepared.imaginary_fine_structure,
        signal_len,
        "xanes imaginary fine structure",
    )?;
    validate_vector_shape(
        &input.prepared.real_fine_structure,
        signal_len,
        "xanes real fine structure",
    )?;
    validate_finite_view(
        input.prepared.incident_energy.view(),
        "xanes incident energy",
    )?;
    validate_finite_view(
        input.prepared.excitation_energy.view(),
        "xanes excitation energy",
    )?;
    validate_finite_view(input.prepared.absorption.view(), "xanes absorption")?;
    validate_finite_view(
        input.prepared.embedded_background.view(),
        "xanes embedded background",
    )?;
    validate_finite_view(
        input.prepared.imaginary_fine_structure.view(),
        "xanes imaginary fine structure",
    )?;
    validate_finite_view(
        input.prepared.real_fine_structure.view(),
        "xanes real fine structure",
    )?;
    validate_finite_view(input.photoelectron_momentum, "xanes photoelectron momentum")?;
    Ok(())
}

fn validate_specfunct_chi_data_input(input: SfconvSpecfunctChiDataInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_chi_dat(input.source)?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "EXAFS chi.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "EXAFS chi.dat convolution work_len {} is smaller than source row count {}",
            input.work_len,
            input.source.point_count()
        ));
    }
    if input.photoelectron_momentum.len() < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "EXAFS chi.dat convolution momentum count {} is smaller than source row count {}",
            input.photoelectron_momentum.len(),
            input.source.point_count()
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "exafs chi.dat photoelectron momentum",
    )
}

fn validate_specfunct_feff_path_data_input(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_finite_scalar(input.source.degeneracy, "feff path degeneracy")?;
    validate_finite_scalar(
        input.source.effective_half_path_length_angstrom,
        "feff path half length",
    )?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "EXAFS feffNNNN.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < 3 {
        return invalid_specfunct_dat(format!(
            "EXAFS feffNNNN.dat convolution work_len {} is smaller than FEFF minimum 3",
            input.work_len
        ));
    }
    if input.photoelectron_momentum.len() < input.work_len {
        return invalid_specfunct_dat(format!(
            "EXAFS feffNNNN.dat convolution momentum count {} is smaller than work_len {}",
            input.photoelectron_momentum.len(),
            input.work_len
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "exafs feffNNNN.dat photoelectron momentum",
    )?;
    validate_feff_path_view_lengths(input.source)?;
    validate_finite_view(
        input.source.wave_number_inverse_angstrom.view(),
        "feff path wave number",
    )?;
    validate_finite_view(input.source.central_phase.view(), "feff path central phase")?;
    validate_finite_view(
        input.source.effective_amplitude.view(),
        "feff path effective amplitude",
    )?;
    validate_finite_view(
        input.source.effective_phase.view(),
        "feff path effective phase",
    )?;
    validate_finite_view(
        input.source.reduction_factor.view(),
        "feff path reduction factor",
    )?;
    validate_finite_view(
        input.source.mean_free_path_angstrom.view(),
        "feff path mean free path",
    )?;
    validate_finite_view(
        input.source.real_momentum_inverse_angstrom.view(),
        "feff path real momentum",
    )?;
    validate_feff_path_uniform_grid_coverage(input)
}

fn validate_specfunct_xmu_data_input(input: SfconvSpecfunctXmuDataInput<'_>) -> Result<()> {
    validate_specfunct_dat(input.cache)?;
    validate_xmu_dat(input.source)?;
    if input.source.point_count() < 2 {
        return invalid_specfunct_dat(
            "XANES xmu.dat convolution requires at least two source rows",
        );
    }
    if input.work_len < 21 {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution work_len {} is smaller than FEFF minimum 21",
            input.work_len
        ));
    }
    if input.work_len < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution work_len {} is smaller than source row count {}",
            input.work_len,
            input.source.point_count()
        ));
    }
    if input.photoelectron_momentum.len() < input.source.point_count() {
        return invalid_specfunct_dat(format!(
            "XANES xmu.dat convolution momentum count {} is smaller than source row count {}",
            input.photoelectron_momentum.len(),
            input.source.point_count()
        ));
    }
    validate_finite_view(
        input.photoelectron_momentum,
        "xanes xmu.dat photoelectron momentum",
    )
}

fn validate_feff_path_view_lengths(source: &SfconvSo2convFeffPathData) -> Result<()> {
    let point_count = source.point_count();
    validate_feff_path_view_len("caph2", source.central_phase.len(), point_count)?;
    validate_feff_path_view_len("xmfeff2", source.effective_amplitude.len(), point_count)?;
    validate_feff_path_view_len("phfeff2", source.effective_phase.len(), point_count)?;
    validate_feff_path_view_len("redfac2", source.reduction_factor.len(), point_count)?;
    validate_feff_path_view_len("xlam2", source.mean_free_path_angstrom.len(), point_count)?;
    validate_feff_path_view_len(
        "realck2",
        source.real_momentum_inverse_angstrom.len(),
        point_count,
    )
}

fn validate_feff_path_view_len(field: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    invalid_specfunct_dat(format!(
        "feff path {field} length {actual} does not match source row count {expected}"
    ))
}

fn validate_feff_path_uniform_grid_coverage(
    input: SfconvSpecfunctFeffPathDataInput<'_>,
) -> Result<()> {
    let first = input.source.wave_number_inverse_angstrom[0];
    if first > 0.0 {
        return invalid_specfunct_dat(format!(
            "feff path grid starts at {first}, but SO2CONV dense path grid starts at 0"
        ));
    }
    let last = input.source.wave_number_inverse_angstrom[input.source.point_count() - 1];
    let dense_max = 0.05 * (input.work_len - 1) as Real;
    if last < dense_max {
        return invalid_specfunct_dat(format!(
            "feff path grid ends at {last}, below SO2CONV dense grid maximum {dense_max}"
        ));
    }
    Ok(())
}

fn sfconv_specfunct_uniform_path_momentum(work_len: usize) -> Array1<Real> {
    Array1::from_shape_fn(work_len, |row| {
        0.05 * row as Real / SFCONV_SO2CONV_BOHR_ANGSTROM
    })
}

fn sfconv_specfunct_feff_path_averages(
    source_momentum: ArrayView1<'_, Real>,
    path_momentum: ArrayView1<'_, Real>,
    rows: &[SfconvExafsConvolution],
) -> Result<Vec<SfconvPathAverage>> {
    let amplitude_reduction = Array1::from_iter(rows.iter().map(|row| row.amplitude_reduction));
    let phase_shift = Array1::from_iter(rows.iter().map(|row| row.phase_shift));
    (0..path_momentum.len())
        .map(|row| {
            let previous = if row == 0 {
                path_momentum[row]
            } else {
                path_momentum[row - 1]
            };
            let next = if row + 1 == path_momentum.len() {
                path_momentum[row]
            } else {
                path_momentum[row + 1]
            };
            sfconv_path_average(SfconvPathAverageInput {
                source_momentum,
                amplitude_reduction: amplitude_reduction.view(),
                phase_shift: phase_shift.view(),
                previous_momentum: previous,
                center_momentum: path_momentum[row],
                next_momentum: next,
                momentum_step: 0.05,
            })
            .map_err(specfunct_exafs_error)
        })
        .collect()
}

fn sfconv_specfunct_xanes_convolution_row(
    input: SfconvSpecfunctXanesRowsInput<'_>,
    row: usize,
    asymmetric_phase: bool,
) -> Result<SfconvXanesConvolution> {
    let momentum =
        sfconv_specfunct_interpolate_momentum(input.cache, input.photoelectron_momentum[row])?;
    let spectral = sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
        energy: momentum.energy.view(),
        spectral_function: momentum.spectral_function.view(),
        output_len: input.cache.spectral_point_count(),
    })
    .map_err(specfunct_xanes_error)?;

    let embedded_background = sfconv_specfunct_xanes_convolve_signal(
        input,
        row,
        &momentum,
        &spectral,
        input.prepared.embedded_background.view(),
    )?;
    if asymmetric_phase {
        let absorption = sfconv_specfunct_xanes_convolve_signal(
            input,
            row,
            &momentum,
            &spectral,
            input.prepared.absorption.view(),
        )?;
        return sfconv_xanes_convolution(SfconvXanesConvolutionInput {
            asymmetric_phase,
            absorption_convolution: absorption.amplitude,
            embedded_background: embedded_background.amplitude,
            fine_structure_imaginary_amplitude: 0.0,
            fine_structure_imaginary_phase: 0.0,
            fine_structure_real_amplitude: 0.0,
            fine_structure_real_phase: 0.0,
        })
        .map_err(specfunct_xanes_error);
    }

    let imaginary = sfconv_specfunct_xanes_convolve_signal(
        input,
        row,
        &momentum,
        &spectral,
        input.prepared.imaginary_fine_structure.view(),
    )?;
    let real = sfconv_specfunct_xanes_convolve_signal(
        input,
        row,
        &momentum,
        &spectral,
        input.prepared.real_fine_structure.view(),
    )?;
    sfconv_xanes_convolution(SfconvXanesConvolutionInput {
        asymmetric_phase,
        absorption_convolution: 0.0,
        embedded_background: embedded_background.amplitude,
        fine_structure_imaginary_amplitude: imaginary.amplitude,
        fine_structure_imaginary_phase: imaginary.phase,
        fine_structure_real_amplitude: real.amplitude,
        fine_structure_real_phase: real.phase,
    })
    .map_err(specfunct_xanes_error)
}

fn sfconv_specfunct_xanes_convolve_signal(
    input: SfconvSpecfunctXanesRowsInput<'_>,
    row: usize,
    momentum: &SfconvMomentumSpectralInterpolation,
    spectral: &SfconvSpectralInterpolation,
    signal: ArrayView1<'_, Real>,
) -> Result<SfconvConvolution> {
    sfconv_convolve(SfconvConvolutionInput {
        photoelectron_energy: input.prepared.excitation_energy[row],
        chemical_potential: input.chemical_potential,
        core_hole_lifetime: input.cache.core_hole_lifetime,
        signal_energy: input.prepared.excitation_energy.view(),
        signal,
        spectral_energy: spectral.energy.view(),
        spectral_function: spectral.spectral_function.view(),
        weights: momentum.weights.view(),
        asymmetric_phase: input.cache.asymmetric_phase != 0,
        cutoff: input.cutoff,
        plasma_frequency: input.plasma_frequency,
    })
    .map_err(specfunct_xanes_error)
}

fn specfunct_xanes_error(source: SfconvError) -> IoError {
    IoError::SpecfunctDatXanesConvolution { source }
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

    #[test]
    fn convolves_exafs_rows_from_cache() -> Result<()> {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let input = sample_exafs_input(24);
        let momentum = Array1::from_vec(vec![0.75, 1.25, 1.75]);

        let rows = sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
            cache: &data,
            signal_energy: input.signal_energy.view(),
            real_signal: input.real_signal.view(),
            imaginary_signal: input.imaginary_signal.view(),
            original_magnitude: input.original_magnitude.view(),
            original_phase: input.original_phase.view(),
            phase_minus_2kr: input.phase_minus_2kr.view(),
            photoelectron_momentum: momentum.view(),
            active_len: momentum.len(),
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })?;

        assert_eq!(rows.len(), momentum.len());
        for row in rows {
            assert!(row.real.is_finite());
            assert!(row.imaginary.is_finite());
            assert!(row.magnitude.is_finite());
            assert!(row.output_phase.is_finite());
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_exafs_row_inputs() {
        let mut data = sample_specfunct_data();
        let input = sample_exafs_input(4);
        let momentum = Array1::from_vec(vec![0.75]);

        assert!(
            sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
                cache: &data,
                signal_energy: input.signal_energy.view(),
                real_signal: input.real_signal.view(),
                imaginary_signal: input.imaginary_signal.view(),
                original_magnitude: input.original_magnitude.view(),
                original_phase: input.original_phase.view(),
                phase_minus_2kr: input.phase_minus_2kr.view(),
                photoelectron_momentum: momentum.view(),
                active_len: 2,
                chemical_potential: 0.0,
                cutoff: false,
                plasma_frequency: 1.0,
            })
            .is_err()
        );

        data.asymmetric_phase = 1;
        assert!(
            sfconv_specfunct_exafs_convolution_rows(SfconvSpecfunctExafsRowsInput {
                cache: &data,
                signal_energy: input.signal_energy.view(),
                real_signal: input.real_signal.view(),
                imaginary_signal: input.imaginary_signal.view(),
                original_magnitude: input.original_magnitude.view(),
                original_phase: input.original_phase.view(),
                phase_minus_2kr: input.phase_minus_2kr.view(),
                photoelectron_momentum: momentum.view(),
                active_len: 1,
                chemical_potential: 0.0,
                cutoff: false,
                plasma_frequency: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn builds_convoluted_chi_data_from_cache() -> Result<()> {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let source = sample_chi_dat(24);
        let momentum = Array1::from_vec(
            (0..source.point_count())
                .map(|row| 0.75 + 0.01 * row as f64)
                .collect(),
        );

        let output = sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 28,
        })?;

        assert_eq!(output.point_count(), source.point_count());
        assert_eq!(output.header_lines, source.header_lines);
        assert_eq!(output.wave_number, source.wave_number);
        assert!(output.phase_minus_2kr.is_some());
        assert!(output.ckp_real.is_none());
        assert!(output.ckp_imag.is_none());
        assert!(output.chi.iter().all(|value| value.is_finite()));
        assert!(output.magnitude.iter().all(|value| value.is_finite()));
        assert!(output.phase.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn rejects_invalid_chi_cache_convolution_inputs() {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let source = sample_chi_dat(24);
        let short_momentum = Array1::from_vec(vec![0.75]);
        let momentum = Array1::from_vec(vec![0.75; source.point_count()]);

        assert!(
            sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: short_momentum.view(),
                work_len: 28,
            })
            .is_err()
        );
        assert!(
            sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: 2,
            })
            .is_err()
        );
        data.asymmetric_phase = 1;
        assert!(
            sfconv_specfunct_chi_data_from_cache(SfconvSpecfunctChiDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: 28,
            })
            .is_err()
        );
    }

    #[test]
    fn builds_convoluted_feff_path_data_from_cache() -> Result<()> {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let source = sample_feff_path_data(24);
        let momentum = Array1::from_vec((0..24).map(|row| 0.75 + 0.01 * row as f64).collect());

        let output =
            sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: momentum.len(),
            })?;

        assert_eq!(output.point_count(), source.point_count());
        assert_eq!(output.header_lines, source.header_lines);
        assert_eq!(
            output.wave_number_inverse_angstrom,
            source.wave_number_inverse_angstrom
        );
        assert_eq!(output.effective_amplitude, source.effective_amplitude);
        assert_eq!(output.effective_phase, source.effective_phase);
        assert_eq!(
            output.mean_free_path_angstrom,
            source.mean_free_path_angstrom
        );
        assert_eq!(
            output.real_momentum_inverse_angstrom,
            source.real_momentum_inverse_angstrom
        );
        assert!(output.central_phase.iter().all(|value| value.is_finite()));
        assert!(
            output
                .reduction_factor
                .iter()
                .all(|value| value.is_finite())
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_feff_path_cache_convolution_inputs() {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let source = sample_feff_path_data(6);
        let short_momentum = Array1::from_vec(vec![0.75]);
        let momentum = Array1::from_vec(vec![0.75; 6]);

        assert!(
            sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: short_momentum.view(),
                work_len: 6,
            })
            .is_err()
        );

        let mut shifted_source = source.clone();
        shifted_source.wave_number_inverse_angstrom[0] = 0.05;
        assert!(
            sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
                cache: &data,
                source: &shifted_source,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: 6,
            })
            .is_err()
        );

        let mut short_grid = source.clone();
        short_grid.wave_number_inverse_angstrom[5] = 0.20;
        assert!(
            sfconv_specfunct_feff_path_data_from_cache(SfconvSpecfunctFeffPathDataInput {
                cache: &data,
                source: &short_grid,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: 6,
            })
            .is_err()
        );
    }

    #[test]
    fn convolves_xanes_rows_from_cache() -> Result<()> {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 0;
        let prepared = sample_xanes_preparation(24);
        let momentum = Array1::from_vec(vec![0.75, 1.25, 1.75]);

        let rows = sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
            cache: &data,
            prepared: &prepared,
            photoelectron_momentum: momentum.view(),
            active_len: momentum.len(),
            chemical_potential: 0.0,
            cutoff: false,
            plasma_frequency: 1.0,
        })?;

        assert_eq!(rows.len(), momentum.len());
        for row in rows {
            assert!(row.absorption.is_finite());
            assert!(row.embedded_background.is_finite());
            assert!(row.fine_structure.is_finite());
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_xanes_row_inputs() {
        let data = sample_specfunct_data();
        let prepared = sample_xanes_preparation(4);
        let momentum = Array1::from_vec(vec![0.75]);

        assert!(
            sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
                cache: &data,
                prepared: &prepared,
                photoelectron_momentum: momentum.view(),
                active_len: 2,
                chemical_potential: 0.0,
                cutoff: false,
                plasma_frequency: 1.0,
            })
            .is_err()
        );
        assert!(
            sfconv_specfunct_xanes_convolution_rows(SfconvSpecfunctXanesRowsInput {
                cache: &data,
                prepared: &prepared,
                photoelectron_momentum: momentum.view(),
                active_len: 0,
                chemical_potential: 0.0,
                cutoff: false,
                plasma_frequency: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn builds_convoluted_xmu_data_from_cache() -> Result<()> {
        let mut data = sample_specfunct_data();
        data.asymmetric_phase = 1;
        let source = sample_xmu_dat(24);
        let momentum = Array1::from_vec(
            (0..source.point_count())
                .map(|row| 0.75 + 0.01 * row as f64)
                .collect(),
        );

        let output = sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
            cache: &data,
            source: &source,
            material: sample_so2conv_material(),
            photoelectron_momentum: momentum.view(),
            work_len: 28,
        })?;

        assert_eq!(output.point_count(), source.point_count());
        assert_eq!(output.header_lines, source.header_lines);
        assert_eq!(output.photon_energy_ev, source.photon_energy_ev);
        assert_eq!(output.relative_energy_ev, source.relative_energy_ev);
        assert_eq!(output.wave_number, source.wave_number);
        assert!(output.mu.iter().all(|value| value.is_finite()));
        assert!(output.mu0.iter().all(|value| value.is_finite()));
        assert!(output.chi.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn rejects_invalid_xmu_cache_convolution_inputs() {
        let data = sample_specfunct_data();
        let source = sample_xmu_dat(24);
        let short_momentum = Array1::from_vec(vec![0.75]);
        let momentum = Array1::from_vec(vec![0.75; source.point_count()]);

        assert!(
            sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: short_momentum.view(),
                work_len: 28,
            })
            .is_err()
        );
        assert!(
            sfconv_specfunct_xmu_data_from_cache(SfconvSpecfunctXmuDataInput {
                cache: &data,
                source: &source,
                material: sample_so2conv_material(),
                photoelectron_momentum: momentum.view(),
                work_len: 20,
            })
            .is_err()
        );
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

    fn sample_feff_path_data(len: usize) -> SfconvSo2convFeffPathData {
        SfconvSo2convFeffPathData {
            header_lines: vec![
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                    .to_string(),
                "# path 1 reff 4 2.0000 2.5000".to_string(),
                " ------------------------------------------------------------------------------"
                    .to_string(),
            ],
            leg_count: 4,
            degeneracy: 2.0,
            effective_half_path_length_angstrom: 2.5,
            wave_number_inverse_angstrom: Array1::from_shape_fn(len, |row| 0.05 * row as f64),
            central_phase: Array1::from_shape_fn(len, |row| 0.1 + 0.01 * row as f64),
            effective_amplitude: Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64),
            effective_phase: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
            reduction_factor: Array1::from_shape_fn(len, |row| 0.9 + 0.001 * row as f64),
            mean_free_path_angstrom: Array1::from_shape_fn(len, |row| 8.0 + 0.05 * row as f64),
            real_momentum_inverse_angstrom: Array1::from_shape_fn(len, |row| 0.05 * row as f64),
        }
    }

    fn sample_chi_dat(len: usize) -> ChiDatData {
        let wave_number = Array1::from_shape_fn(len, |row| 0.2 + 0.02 * row as f64);
        let magnitude = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
        let phase = Array1::from_shape_fn(len, |row| 0.1 + 0.03 * row as f64);
        ChiDatData {
            header_lines: vec![
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                    .to_string(),
                " ------------------------------------------------------------------------------"
                    .to_string(),
            ],
            wave_number,
            chi: Array1::from_shape_fn(len, |row| 0.01 * row as f64),
            magnitude,
            phase: phase.clone(),
            phase_minus_2kr: Some(Array1::from_shape_fn(len, |row| {
                phase[row] - 0.04 * row as f64
            })),
            ckp_real: None,
            ckp_imag: None,
        }
    }

    fn sample_xmu_dat(len: usize) -> XmuDatData {
        XmuDatData {
            header_lines: vec![
                "# Header Gam_ch= 1.729000 Rs_int= 2.05 Vint= 12.34000 Mu= 18.76000 kf= 1.230000"
                    .to_string(),
                " ------------------------------------------------------------------------------"
                    .to_string(),
            ],
            normalization: Some(1.0),
            photon_energy_ev: Array1::from_shape_fn(len, |row| 100.0 + 2.0 * row as f64),
            relative_energy_ev: Array1::from_shape_fn(len, |row| 1.0 + 2.0 * row as f64),
            wave_number: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
            mu: Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64),
            mu0: Array1::from_shape_fn(len, |row| 0.8 + 0.01 * row as f64),
            chi: Array1::from_shape_fn(len, |row| 0.2 + 0.01 * row as f64),
        }
    }

    fn sample_so2conv_material() -> refeff_core::SfconvSo2convMaterialInput {
        refeff_core::SfconvSo2convMaterialInput {
            core_hole_width_ev: 1.729,
            wigner_seitz_radius: 2.05,
            interstitial_potential_ev: 12.34,
            chemical_potential_ev: 18.76,
            fermi_wave_number_inv_angstrom: 1.23,
        }
    }

    fn spectral_table(momentum_count: usize, spectral_count: usize, base: f64) -> Array2<f64> {
        Array2::from_shape_fn((momentum_count, spectral_count), |(row, col)| {
            base + row as f64 + 0.1 * col as f64
        })
    }

    struct SampleExafsInput {
        signal_energy: Array1<Real>,
        real_signal: Array1<Real>,
        imaginary_signal: Array1<Real>,
        original_magnitude: Array1<Real>,
        original_phase: Array1<Real>,
        phase_minus_2kr: Array1<Real>,
    }

    fn sample_exafs_input(len: usize) -> SampleExafsInput {
        let signal_energy = Array1::from_shape_fn(len, |row| row as f64 * 0.05);
        let real_signal = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
        let imaginary_signal = Array1::from_shape_fn(len, |row| 0.4 + 0.01 * row as f64);
        let original_magnitude = Array1::from_shape_fn(len, |row| {
            let real = real_signal[row];
            let imaginary = imaginary_signal[row];
            (real * real + imaginary * imaginary).sqrt()
        });
        let original_phase =
            Array1::from_shape_fn(len, |row| imaginary_signal[row].atan2(real_signal[row]));
        let phase_minus_2kr =
            Array1::from_shape_fn(len, |row| original_phase[row] - 0.02 * row as f64);

        SampleExafsInput {
            signal_energy,
            real_signal,
            imaginary_signal,
            original_magnitude,
            original_phase,
            phase_minus_2kr,
        }
    }

    fn sample_xanes_preparation(len: usize) -> SfconvSo2convXanesPreparation {
        let excitation_energy = Array1::from_shape_fn(len, |row| row as f64 * 5.0);
        let absorption = Array1::from_shape_fn(len, |row| 1.0 + 0.02 * row as f64);
        let embedded_background = Array1::from_shape_fn(len, |row| 0.8 + 0.01 * row as f64);
        let imaginary_fine_structure = &absorption - &embedded_background;

        SfconvSo2convXanesPreparation {
            incident_energy: Array1::from_shape_fn(len, |row| 100.0 + row as f64 * 5.0),
            excitation_energy,
            absorption,
            embedded_background,
            imaginary_fine_structure,
            real_fine_structure: Array1::from_shape_fn(len, |row| 0.1 + 0.005 * row as f64),
        }
    }
}
