use super::*;

/// Error returned by FEFF convolution helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum ConvolutionError {
    /// Energy and spectrum arrays must have identical lengths.
    #[error("convolution length mismatch: omega has {omega_len}, spectrum has {spectrum_len}")]
    LengthMismatch {
        omega_len: usize,
        spectrum_len: usize,
    },
    /// FEFF `conv` needs at least two points to extrapolate the final interval.
    #[error("convolution requires at least two points, got {points}")]
    InsufficientPoints { points: usize },
    /// The Lorentzian width must be positive and finite.
    #[error("Lorentzian width must be positive and finite, got {width}")]
    InvalidWidth { width: Real },
    /// Energy values must be finite.
    #[error("energy value {name} must be finite, got {value}")]
    NonFiniteEnergy { name: &'static str, value: Real },
    /// Spectrum values must be finite.
    #[error("spectrum value {name} must be finite, got ({real}, {imaginary})")]
    NonFiniteSpectrum {
        name: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF's endpoint extrapolation divides by the final energy spacing.
    #[error("last two energy values must be distinct for endpoint extrapolation")]
    DuplicateEndpointEnergy,
    /// A real-valued spectrum must match its energy grid.
    #[error("excitation convolution length mismatch: omega has {omega_len}, xmu has {xmu_len}")]
    ExcitationLengthMismatch { omega_len: usize, xmu_len: usize },
    /// FEFF `exconv` requires at least two grid points.
    #[error("excitation convolution requires at least two points, got {points}")]
    ExcitationInsufficientPoints { points: usize },
    /// FEFF `exconv` requires a Fermi level inside the grid but below the last point.
    #[error(
        "excitation convolution Fermi level {fermi_energy} is outside the supported energy grid"
    )]
    ExcitationFermiOutOfRange { fermi_energy: Real },
    /// FEFF `exconv` requires distinct adjacent energy points.
    #[error(
        "excitation convolution energy row {row} must increase, got {current} after {previous}"
    )]
    ExcitationNonIncreasingEnergy {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// FEFF `exconv` scalar inputs must be finite.
    #[error("excitation convolution {field} must be finite, got {value}")]
    ExcitationNonFiniteScalar { field: &'static str, value: Real },
    /// FEFF `exconv` spectrum values must be finite.
    #[error("excitation convolution xmu row {row} must be finite, got {value}")]
    ExcitationNonFiniteSpectrum { row: usize, value: Real },
    /// FEFF `exconv` divides by the shake-up weight.
    #[error("excitation convolution shake-up weight must be nonzero, got {value}")]
    ExcitationInvalidShakeupWeight { value: Real },
    /// FEFF `exconv` divides by the distribution width.
    #[error("excitation convolution distribution width must be finite and nonzero, got {value}")]
    ExcitationInvalidDistributionWidth { value: Real },
    /// FEFF interpolation failed inside `exconv`.
    #[error("excitation convolution interpolation failed: {source}")]
    ExcitationInterpolation { source: InterpolationError },
    /// FEFF `xscorratan` input arrays must have identical total lengths.
    #[error(
        "xscorratan length mismatch: energy has {energy_len}, xsec has {xsec_len}, xsnorm has {xsnorm_len}, chia has {chia_len}"
    )]
    AtanLengthMismatch {
        energy_len: usize,
        xsec_len: usize,
        xsnorm_len: usize,
        chia_len: usize,
    },
    /// The horizontal mesh must contain at least one point and fit in the full mesh.
    #[error(
        "xscorratan horizontal length {horizontal_len} is invalid for total length {total_len}"
    )]
    AtanInvalidHorizontalLength {
        horizontal_len: usize,
        total_len: usize,
    },
    /// FEFF `ik0` is a horizontal-mesh index.
    #[error(
        "xscorratan reference Fermi index {fermi_index} is outside horizontal length {horizontal_len}"
    )]
    AtanFermiIndexOutOfRange {
        fermi_index: usize,
        horizontal_len: usize,
    },
    /// Scalar correction inputs must be finite.
    #[error("xscorratan {field} must be finite, got {value}")]
    AtanNonFiniteScalar { field: &'static str, value: Real },
    /// Complex energy mesh values must be finite.
    #[error("xscorratan energy row {row} must be finite, got ({real}, {imaginary})")]
    AtanNonFiniteEnergy {
        row: usize,
        real: Real,
        imaginary: Real,
    },
    /// Complex spectrum values must be finite.
    #[error("xscorratan {field} row {row} must be finite, got ({real}, {imaginary})")]
    AtanNonFiniteSpectrum {
        field: &'static str,
        row: usize,
        real: Real,
        imaginary: Real,
    },
    /// Normalization values must be finite.
    #[error("xscorratan xsnorm row {row} must be finite, got {value}")]
    AtanNonFiniteNormalization { row: usize, value: Real },
    /// FEFF interpolation failed inside `xscorratan`.
    #[error("xscorratan interpolation failed: {source}")]
    AtanInterpolation { source: InterpolationError },
}

/// Inputs for FEFF `FF2X/exconv.f90`.
#[derive(Debug, Clone, Copy)]
pub struct Ff2xExcitationConvolutionInput<'a> {
    /// Energy grid, FEFF `omega`.
    pub energy: ArrayView1<'a, Real>,
    /// Original absorption coefficient, FEFF `xmu`.
    pub xmu: ArrayView1<'a, Real>,
    /// Fermi level, FEFF `efermi`.
    pub fermi_energy: Real,
    /// Relaxed-orbital overlap amplitude, FEFF `s02`.
    pub amplitude_reduction: Real,
    /// Relaxation energy, FEFF `erelax`.
    pub relaxation_energy: Real,
    /// Plasmon frequency, FEFF `wp`.
    pub plasmon_frequency: Real,
}

/// Inputs for FEFF `FF2X/xscorratan.f90`.
#[derive(Debug, Clone, Copy)]
pub struct Ff2xAtanCorrectionInput<'a> {
    /// FEFF spectroscopy selector, `ispec`; `2` uses the emission branch.
    pub spectroscopy: i32,
    /// Complex energy mesh, FEFF `emxs`.
    pub energy: ArrayView1<'a, Complex>,
    /// Number of horizontal-axis points, FEFF `ne1`.
    pub horizontal_len: usize,
    /// Zero-based Rust equivalent of FEFF `ik0`.
    pub fermi_index: usize,
    /// Atomic background cross section, FEFF `xsec`.
    pub xsec: ArrayView1<'a, Complex>,
    /// Normalization multiplier, FEFF `xsnorm`.
    pub xsnorm: ArrayView1<'a, Real>,
    /// Fine-structure contribution, FEFF `chia`.
    pub chia: ArrayView1<'a, Complex>,
    /// Real Fermi-level correction, FEFF `vrcorr`.
    pub real_correction: Real,
    /// Imaginary mesh correction, FEFF `vicorr`.
    pub imaginary_correction: Real,
}
