//! FEFF analytic convolution helpers.
//!
//! This module ports `MATH/conv.f90`. FEFF linearly interpolates a complex
//! spectrum on each energy interval and integrates the Lorentzian kernel
//! analytically with `conv1`; `conv` applies that segment integral to every
//! requested output energy and adds one extrapolated endpoint interval. It also
//! contains the `FF2X/exconv.f90` excitation-spectrum convolution and
//! `FF2X/xscorratan.f90` arctangent correction used by the final spectrum
//! assembly path.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

use crate::interpolation::{InterpolationError, locate_below, terp, terpc};
use crate::{Complex, ComplexVec, Real, RealVec};

const FEFF_REAL_PI: Real = std::f32::consts::PI as Real;
const XSCORR_ATAN_PI: Real = std::f64::consts::PI;
const XSCORR_EPS4: Real = 1.0e-4;

/// Error returned by FEFF convolution helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
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

/// Port of FEFF `conv1`, the analytic integral over one linear segment.
pub fn conv1(
    x1: Real,
    x2: Real,
    y1: Complex,
    y2: Complex,
    x0: Real,
    width: Real,
) -> Result<Complex, ConvolutionError> {
    validate_width(width)?;
    validate_energy("x1", x1)?;
    validate_energy("x2", x2)?;
    validate_energy("x0", x0)?;
    validate_spectrum("y1", y1)?;
    validate_spectrum("y2", y2)?;

    let half_width = (x2 - x1) / 2.0;
    let denominator = (x1 + x2) / 2.0 - x0;
    let t = Complex::new(half_width, 0.0) / Complex::new(denominator, -width);

    let real_part = conv1_component((y2.re - y1.re) / 2.0, (y2.re + y1.re) / 2.0, t);
    let imaginary_part = conv1_component((y2.im - y1.im) / 2.0, (y2.im + y1.im) / 2.0, t);
    Ok(Complex::new(real_part, imaginary_part))
}

/// Convolve a FEFF spectrum with the Lorentzian broadening kernel.
///
/// This ports `conv` and returns a new `ndarray` vector. Use
/// [`conv_in_place`] when the caller needs FEFF's in-place mutation behavior.
pub fn conv(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<ComplexVec, ConvolutionError> {
    let values = convolved_values(omega, spectrum, width)?;
    Ok(Array1::from_vec(values))
}

/// In-place FEFF `conv` behavior for a mutable spectrum slice.
pub fn conv_in_place(
    omega: &[Real],
    spectrum: &mut [Complex],
    width: Real,
) -> Result<(), ConvolutionError> {
    let values = convolved_values(omega, spectrum, width)?;
    for (target, value) in spectrum.iter_mut().zip(values) {
        *target = value;
    }
    Ok(())
}

/// Port of FEFF `FF2X/exconv.f90`: excitation-spectrum convolution.
///
/// FEFF models the excitation spectrum as a quasiparticle delta contribution
/// plus an exponentially decaying shake-up tail. The original routine mutates
/// `xmu` in place; this helper returns a new `ndarray` vector so callers can
/// choose whether to preserve the input.
pub fn ff2x_excitation_convolve(
    input: Ff2xExcitationConvolutionInput<'_>,
) -> Result<RealVec, ConvolutionError> {
    validate_excitation_input(input)?;
    if input.amplitude_reduction >= 0.999 {
        return Ok(input.xmu.to_owned());
    }

    let omega = input.energy.to_vec();
    let xmu = input.xmu.to_vec();
    let nk = omega.len();
    let plasmon_frequency = if input.plasmon_frequency <= 0.0 {
        0.000_01
    } else {
        input.plasmon_frequency
    };
    const PLASMON_WEIGHT: Real = 0.0;
    let shakeup_weight = 1.0 - input.amplitude_reduction - PLASMON_WEIGHT;
    if !(shakeup_weight.is_finite() && shakeup_weight != 0.0) {
        return Err(ConvolutionError::ExcitationInvalidShakeupWeight {
            value: shakeup_weight,
        });
    }
    let plasmon_width = plasmon_frequency;
    let shakeup_width = (input.relaxation_energy
        - PLASMON_WEIGHT * (plasmon_frequency + plasmon_width))
        / shakeup_weight;
    validate_excitation_width(shakeup_width)?;
    validate_excitation_width(plasmon_width)?;

    let fermi_located = locate_below(input.fermi_energy, &omega);
    if fermi_located == 0 || fermi_located >= nk {
        return Err(ConvolutionError::ExcitationFermiOutOfRange {
            fermi_energy: input.fermi_energy,
        });
    }
    let fermi_index = fermi_located - 1;
    let xmu_at_fermi = terp(&omega, &xmu, 1, input.fermi_energy)
        .map_err(|source| ConvolutionError::ExcitationInterpolation { source })?
        .value;

    let mut slope = vec![0.0; nk];
    let mut dmu = vec![0.0; nk];
    let mut xmup = vec![0.0; nk];
    fill_excitation_tail(
        ExcitationTailInput {
            energy: &omega,
            xmu: &xmu,
            fermi_energy: input.fermi_energy,
            xmu_at_fermi,
            fermi_index,
            width: shakeup_width,
        },
        &mut slope,
        &mut dmu,
    );

    for row in 0..nk {
        xmup[row] = input.amplitude_reduction * xmu[row] + shakeup_weight * dmu[row];
    }

    Ok(Array1::from_vec(xmup))
}

/// Port of FEFF `FF2X/xscorratan.f90`: arctangent self-energy correction.
///
/// FEFF builds `xmu = xsec + xsnorm * chia` on the complex mesh, then fills
/// `cchi(1:ne1)` with the horizontal-axis correction. The original routine also
/// shifts and restores the input mesh around the vertical contour. This Rust
/// helper is pure and returns only the active horizontal correction values.
pub fn ff2x_atan_correction(
    input: Ff2xAtanCorrectionInput<'_>,
) -> Result<ComplexVec, ConvolutionError> {
    validate_atan_input(input)?;

    let xmu: Vec<_> = input
        .xsec
        .iter()
        .zip(input.xsnorm.iter())
        .zip(input.chia.iter())
        .map(|((&xsec, &xsnorm), &chia)| xsec + chia * xsnorm)
        .collect();
    let mut fermi_energy = input.energy[input.energy.len() - 1].re;

    if input.real_correction.abs() > XSCORR_EPS4 {
        fermi_energy -= input.real_correction;
        let omega: Vec<_> = input
            .energy
            .iter()
            .take(input.horizontal_len)
            .map(|energy| energy.re)
            .collect();
        let _interpolated_fermi = terpc(&omega, &xmu[..input.horizontal_len], 1, fermi_energy)
            .map_err(|source| ConvolutionError::AtanInterpolation { source })?;
        let _reference_xmu = xmu[input.fermi_index];
    }

    let half_loss = input.energy[0].im / 2.0;
    let values = input
        .energy
        .iter()
        .take(input.horizontal_len)
        .zip(xmu.iter())
        .map(|(energy, &xmu_value)| {
            let delta = energy.re - fermi_energy;
            let lorentz_step = -0.5 + delta.atan2(half_loss) / XSCORR_ATAN_PI;
            let correction = xmu_value * lorentz_step;
            if input.spectroscopy == 2 {
                -correction - xmu_value
            } else {
                correction
            }
        })
        .collect();

    Ok(Array1::from_vec(values))
}

fn convolved_values(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<Vec<Complex>, ConvolutionError> {
    validate_inputs(omega, spectrum, width)?;
    let last = omega.len() - 1;
    let previous = omega.len() - 2;
    let final_spacing = omega[last] - omega[previous];
    if final_spacing == 0.0 {
        return Err(ConvolutionError::DuplicateEndpointEnergy);
    }

    let extrapolated_width = final_spacing.max(50.0 * width);
    let xlast = omega[last] + extrapolated_width;
    let slope_scale = extrapolated_width / final_spacing;
    let ylast = spectrum[last] + (spectrum[last] - spectrum[previous]) * slope_scale;

    omega
        .iter()
        .map(|&omega0| {
            let intervals = omega
                .windows(2)
                .zip(spectrum.windows(2))
                .map(|(x_window, y_window)| {
                    conv1(
                        x_window[0],
                        x_window[1],
                        y_window[0],
                        y_window[1],
                        omega0,
                        width,
                    )
                })
                .try_fold(Complex::new(0.0, 0.0), |sum, value| {
                    value.map(|value| sum + value)
                })?;
            let endpoint = conv1(omega[last], xlast, spectrum[last], ylast, omega0, width)?;
            Ok((intervals + endpoint) / FEFF_REAL_PI)
        })
        .collect()
}

fn validate_excitation_input(
    input: Ff2xExcitationConvolutionInput<'_>,
) -> Result<(), ConvolutionError> {
    if input.energy.len() != input.xmu.len() {
        return Err(ConvolutionError::ExcitationLengthMismatch {
            omega_len: input.energy.len(),
            xmu_len: input.xmu.len(),
        });
    }
    if input.energy.len() < 2 {
        return Err(ConvolutionError::ExcitationInsufficientPoints {
            points: input.energy.len(),
        });
    }
    validate_excitation_scalar("fermi_energy", input.fermi_energy)?;
    validate_excitation_scalar("amplitude_reduction", input.amplitude_reduction)?;
    validate_excitation_scalar("relaxation_energy", input.relaxation_energy)?;
    validate_excitation_scalar("plasmon_frequency", input.plasmon_frequency)?;
    for row in 0..input.energy.len() {
        validate_energy("omega", input.energy[row])?;
        if row > 0 && input.energy[row] <= input.energy[row - 1] {
            return Err(ConvolutionError::ExcitationNonIncreasingEnergy {
                row,
                previous: input.energy[row - 1],
                current: input.energy[row],
            });
        }
    }
    for (row, value) in input.xmu.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ConvolutionError::ExcitationNonFiniteSpectrum { row, value });
        }
    }
    Ok(())
}

fn validate_atan_input(input: Ff2xAtanCorrectionInput<'_>) -> Result<(), ConvolutionError> {
    let energy_len = input.energy.len();
    if energy_len != input.xsec.len()
        || energy_len != input.xsnorm.len()
        || energy_len != input.chia.len()
    {
        return Err(ConvolutionError::AtanLengthMismatch {
            energy_len,
            xsec_len: input.xsec.len(),
            xsnorm_len: input.xsnorm.len(),
            chia_len: input.chia.len(),
        });
    }
    if input.horizontal_len == 0 || input.horizontal_len > energy_len {
        return Err(ConvolutionError::AtanInvalidHorizontalLength {
            horizontal_len: input.horizontal_len,
            total_len: energy_len,
        });
    }
    if input.fermi_index >= input.horizontal_len {
        return Err(ConvolutionError::AtanFermiIndexOutOfRange {
            fermi_index: input.fermi_index,
            horizontal_len: input.horizontal_len,
        });
    }
    validate_atan_scalar("real_correction", input.real_correction)?;
    validate_atan_scalar("imaginary_correction", input.imaginary_correction)?;

    for (row, &energy) in input.energy.iter().enumerate() {
        if !(energy.re.is_finite() && energy.im.is_finite()) {
            return Err(ConvolutionError::AtanNonFiniteEnergy {
                row,
                real: energy.re,
                imaginary: energy.im,
            });
        }
    }
    for (row, &value) in input.xsec.iter().enumerate() {
        validate_atan_complex("xsec", row, value)?;
    }
    for (row, &value) in input.chia.iter().enumerate() {
        validate_atan_complex("chia", row, value)?;
    }
    for (row, &value) in input.xsnorm.iter().enumerate() {
        if !value.is_finite() {
            return Err(ConvolutionError::AtanNonFiniteNormalization { row, value });
        }
    }
    Ok(())
}

struct ExcitationTailInput<'a> {
    energy: &'a [Real],
    xmu: &'a [Real],
    fermi_energy: Real,
    xmu_at_fermi: Real,
    fermi_index: usize,
    width: Real,
}

fn fill_excitation_tail(input: ExcitationTailInput<'_>, slope: &mut [Real], dmu: &mut [Real]) {
    for row in 0..=input.fermi_index {
        slope[row] = 0.0;
        dmu[row] = 0.0;
    }
    for (row, value) in slope
        .iter_mut()
        .enumerate()
        .take(input.energy.len() - 1)
        .skip(input.fermi_index)
    {
        *value = input.width * (input.xmu[row + 1] - input.xmu[row])
            / (input.energy[row + 1] - input.energy[row]);
    }

    let first_tail = input.fermi_index + 1;
    let xmult = ((input.fermi_energy - input.energy[first_tail]) / input.width).exp();
    dmu[first_tail] = input.xmu[first_tail]
        - slope[input.fermi_index]
        - xmult * (input.xmu_at_fermi - slope[input.fermi_index]);
    let mut row = first_tail;
    while row + 1 < input.energy.len() {
        let xmult = ((input.energy[row] - input.energy[row + 1]) / input.width).exp();
        dmu[row + 1] =
            input.xmu[row + 1] - slope[row] + xmult * (dmu[row] - input.xmu[row] + slope[row]);
        row += 1;
    }
}

fn conv1_component(slope: Real, midpoint: Real, t: Complex) -> Real {
    let slope = Complex::new(slope, 0.0);
    let midpoint = Complex::new(midpoint, 0.0);
    let value = if t.norm() >= 0.1 {
        slope * 2.0
            + (midpoint - slope / t)
                * ((Complex::new(1.0, 0.0) + t) / (Complex::new(1.0, 0.0) - t)).ln()
    } else {
        midpoint * (2.0 * (t + t * t * t / 3.0)) - slope * (2.0 * t * t / 3.0)
    };
    value.im
}

fn validate_inputs(
    omega: &[Real],
    spectrum: &[Complex],
    width: Real,
) -> Result<(), ConvolutionError> {
    if omega.len() != spectrum.len() {
        return Err(ConvolutionError::LengthMismatch {
            omega_len: omega.len(),
            spectrum_len: spectrum.len(),
        });
    }
    if omega.len() < 2 {
        return Err(ConvolutionError::InsufficientPoints {
            points: omega.len(),
        });
    }
    validate_width(width)?;
    for &energy in omega {
        validate_energy("omega", energy)?;
    }
    for &value in spectrum {
        validate_spectrum("spectrum", value)?;
    }
    Ok(())
}

fn validate_width(width: Real) -> Result<(), ConvolutionError> {
    if !(width.is_finite() && width > 0.0) {
        return Err(ConvolutionError::InvalidWidth { width });
    }
    Ok(())
}

fn validate_energy(name: &'static str, value: Real) -> Result<(), ConvolutionError> {
    if !value.is_finite() {
        return Err(ConvolutionError::NonFiniteEnergy { name, value });
    }
    Ok(())
}

fn validate_excitation_scalar(field: &'static str, value: Real) -> Result<(), ConvolutionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::ExcitationNonFiniteScalar { field, value })
    }
}

fn validate_excitation_width(value: Real) -> Result<(), ConvolutionError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(ConvolutionError::ExcitationInvalidDistributionWidth { value })
    }
}

fn validate_atan_scalar(field: &'static str, value: Real) -> Result<(), ConvolutionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::AtanNonFiniteScalar { field, value })
    }
}

fn validate_atan_complex(
    field: &'static str,
    row: usize,
    value: Complex,
) -> Result<(), ConvolutionError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ConvolutionError::AtanNonFiniteSpectrum {
            field,
            row,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_spectrum(name: &'static str, value: Complex) -> Result<(), ConvolutionError> {
    if !(value.re.is_finite() && value.im.is_finite()) {
        return Err(ConvolutionError::NonFiniteSpectrum {
            name,
            real: value.re,
            imaginary: value.im,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() < 1.0e-12,
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    fn fixture() -> (Vec<Real>, Vec<Complex>) {
        (
            vec![-2.0, -0.5, 0.25, 1.5, 3.0],
            vec![
                Complex::new(0.0, -0.2),
                Complex::new(1.0, 0.3),
                Complex::new(0.25, 0.8),
                Complex::new(2.0, -0.1),
                Complex::new(1.5, 0.5),
            ],
        )
    }

    fn xscorratan_fixture() -> (
        Array1<Complex>,
        Array1<Complex>,
        Array1<Real>,
        Array1<Complex>,
    ) {
        (
            Array1::from_vec(vec![
                Complex::new(-0.20, 0.08),
                Complex::new(-0.05, 0.08),
                Complex::new(0.10, 0.08),
                Complex::new(0.30, 0.08),
                Complex::new(0.55, 0.08),
                Complex::new(0.10, 0.02),
                Complex::new(0.10, 0.05),
                Complex::new(0.12, 0.08),
            ]),
            Array1::from_vec(vec![
                Complex::new(0.40, -0.08),
                Complex::new(0.55, 0.02),
                Complex::new(0.73, 0.06),
                Complex::new(0.95, -0.01),
                Complex::new(1.10, 0.03),
                Complex::new(0.90, 0.04),
                Complex::new(0.82, 0.02),
                Complex::new(0.76, 0.01),
            ]),
            Array1::from_vec(vec![1.20, 0.85, 1.05, 0.95, 1.10, 0.70, 0.60, 0.50]),
            Array1::from_vec(vec![
                Complex::new(0.10, 0.04),
                Complex::new(-0.03, 0.05),
                Complex::new(0.08, -0.02),
                Complex::new(0.15, 0.03),
                Complex::new(-0.05, 0.06),
                Complex::new(0.02, 0.01),
                Complex::new(-0.04, 0.02),
                Complex::new(0.06, -0.03),
            ]),
        )
    }

    #[test]
    fn conv1_matches_feff_reference() -> Result<(), ConvolutionError> {
        let (omega, spectrum) = fixture();

        let value = conv1(omega[1], omega[2], spectrum[1], spectrum[2], 0.75, 0.2)?;

        assert_complex_close(
            value,
            Complex::new(0.11548114784993968, 0.137_468_645_861_804_9),
        );
        Ok(())
    }

    #[test]
    fn conv_matches_feff_reference() -> Result<(), ConvolutionError> {
        let (omega, spectrum) = fixture();
        let values = conv(&omega, &spectrum, 0.2)?;
        let expected = [
            Complex::new(0.11859251775642239, -0.02108887292319812),
            Complex::new(0.7780612100410474, 0.3262922950470956),
            Complex::new(0.565152671501454, 0.5900533350655084),
            Complex::new(1.6561626472085706, 0.10570908207293085),
            Complex::new(1.4178395137621587, 0.5320419613503408),
        ];

        for (&actual, expected) in values.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn conv_in_place_replaces_spectrum_values() -> Result<(), ConvolutionError> {
        let (omega, mut spectrum) = fixture();

        conv_in_place(&omega, &mut spectrum, 0.2)?;

        assert_complex_close(
            spectrum[0],
            Complex::new(0.11859251775642239, -0.02108887292319812),
        );
        assert_complex_close(
            spectrum[4],
            Complex::new(1.4178395137621587, 0.5320419613503408),
        );
        Ok(())
    }

    #[test]
    fn ff2x_excitation_convolve_matches_feff_exconv_reference() -> Result<(), ConvolutionError> {
        let energy = Array1::from_vec(vec![-0.40, -0.12, 0.00, 0.18, 0.45, 0.90, 1.35, 1.95]);
        let xmu = Array1::from_vec(vec![0.20, 0.34, 0.50, 0.72, 0.95, 1.10, 1.04, 0.88]);

        let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
            energy: energy.view(),
            xmu: xmu.view(),
            fermi_energy: 0.05,
            amplitude_reduction: 0.72,
            relaxation_energy: 0.18,
            plasmon_frequency: 0.55,
        })?;

        let expected = [
            0.144,
            0.244_800_000_000_000_02,
            0.36,
            0.551_374_774_249_927_1,
            0.786_619_177_385_294,
            0.988_662_602_795_670_4,
            0.996_793_437_494_949_4,
            0.892_133_865_612_567,
        ];
        for (&actual, expected) in convolved.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn ff2x_excitation_convolve_matches_second_feff_exconv_reference()
    -> Result<(), ConvolutionError> {
        let energy = Array1::from_vec(vec![-0.30, -0.05, 0.08, 0.21, 0.50, 0.80, 1.20, 1.70]);
        let xmu = Array1::from_vec(vec![1.00, 0.92, 0.83, 0.70, 0.58, 0.54, 0.61, 0.75]);

        let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
            energy: energy.view(),
            xmu: xmu.view(),
            fermi_energy: -0.02,
            amplitude_reduction: 0.35,
            relaxation_energy: 0.22,
            plasmon_frequency: 0.30,
        })?;

        let expected = [
            0.35,
            0.322,
            0.433_980_387_834_570_5,
            0.500_447_825_235_705_8,
            0.547_677_351_474_302_1,
            0.543_921_711_601_378_6,
            0.584_511_611_035_726_7,
            0.696_642_813_433_018,
        ];
        for (&actual, expected) in convolved.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-14 * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn ff2x_atan_correction_matches_feff_xscorratan_reference() -> Result<(), ConvolutionError> {
        let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

        let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
            spectroscopy: 1,
            energy: energy.view(),
            horizontal_len: 5,
            fermi_index: 2,
            xsec: xsec.view(),
            xsnorm: xsnorm.view(),
            chia: chia.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        })?;

        let expected = [
            Complex::new(-0.499_416_619_436_505_95, 0.030_733_330_426_861_907),
            Complex::new(-0.485_918_596_136_024_06, -0.057_902_597_251_671_115),
            Complex::new(-0.527_133_064_767_452_6, -0.025_255_761_088_366_892),
            Complex::new(-0.076_042_902_345_822_37, -0.001_287_683_014_551_682_8),
            Complex::new(-0.030_853_890_139_401_412, -0.002_834_424_357_303_861_3),
        ];
        for (&actual, expected) in correction.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn ff2x_atan_correction_matches_feff_xscorratan_emission_reference()
    -> Result<(), ConvolutionError> {
        let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

        let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
            spectroscopy: 2,
            energy: energy.view(),
            horizontal_len: 5,
            fermi_index: 2,
            xsec: xsec.view(),
            xsnorm: xsnorm.view(),
            chia: chia.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        })?;

        let expected = [
            Complex::new(-0.020_583_380_563_494_07, 0.001_266_669_573_138_094),
            Complex::new(-0.038_581_403_863_976_016, -0.004_597_402_748_328_885),
            Complex::new(-0.286_866_935_232_547_36, -0.013_744_238_911_633_101),
            Complex::new(-1.016_457_097_654_177_6, -0.017_212_316_985_448_31),
            Complex::new(-1.014_146_109_860_598_8, -0.093_165_575_642_696_15),
        ];
        for (&actual, expected) in correction.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn ff2x_atan_correction_matches_feff_xscorratan_real_shift_reference()
    -> Result<(), ConvolutionError> {
        let (energy, xsec, xsnorm, chia) = xscorratan_fixture();

        let correction = ff2x_atan_correction(Ff2xAtanCorrectionInput {
            spectroscopy: 1,
            energy: energy.view(),
            horizontal_len: 5,
            fermi_index: 2,
            xsec: xsec.view(),
            xsnorm: xsnorm.view(),
            chia: chia.view(),
            real_correction: 0.03,
            imaginary_correction: 0.0,
        })?;

        let expected = [
            Complex::new(-0.497_312_650_460_951_86, 0.030_603_855_412_981_65),
            Complex::new(-0.478_036_888_055_366_54, -0.056_963_404_201_068_456),
            Complex::new(-0.343_524_987_872_821_3, -0.016_458_813_915_282_592),
            Complex::new(-0.065_454_696_779_511_84, -0.001_108_386_169_721_710_5),
            Complex::new(-0.028_852_105_893_751_454, -0.002_650_528_388_325_492),
        ];
        for (&actual, expected) in correction.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn ff2x_excitation_convolve_preserves_xmu_when_s02_is_nearly_one()
    -> Result<(), ConvolutionError> {
        let energy = Array1::from_vec(vec![-0.30, -0.05, 0.08, 0.21]);
        let xmu = Array1::from_vec(vec![1.00, 0.92, 0.83, 0.70]);

        let convolved = ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
            energy: energy.view(),
            xmu: xmu.view(),
            fermi_energy: -0.02,
            amplitude_reduction: 1.0,
            relaxation_energy: 0.22,
            plasmon_frequency: 0.30,
        })?;

        assert_eq!(convolved, xmu);
        Ok(())
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            conv(&[1.0], &[Complex::new(1.0, 0.0)], 0.2),
            Err(ConvolutionError::InsufficientPoints { .. })
        ));
        assert!(matches!(
            conv(
                &[1.0, 1.0],
                &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
                0.2
            ),
            Err(ConvolutionError::DuplicateEndpointEnergy)
        ));
        assert!(matches!(
            conv1(
                0.0,
                1.0,
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                0.5,
                0.0
            ),
            Err(ConvolutionError::InvalidWidth { .. })
        ));
        let energy = Array1::from_vec(vec![0.0, 0.2, 0.1]);
        let xmu = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
                energy: energy.view(),
                xmu: xmu.view(),
                fermi_energy: 0.05,
                amplitude_reduction: 0.5,
                relaxation_energy: 0.2,
                plasmon_frequency: 0.3,
            }),
            Err(ConvolutionError::ExcitationNonIncreasingEnergy { row: 2, .. })
        ));
        let energy = Array1::from_vec(vec![0.0, 0.2, 0.4]);
        assert!(matches!(
            ff2x_excitation_convolve(Ff2xExcitationConvolutionInput {
                energy: energy.view(),
                xmu: xmu.view(),
                fermi_energy: -0.1,
                amplitude_reduction: 0.5,
                relaxation_energy: 0.2,
                plasmon_frequency: 0.3,
            }),
            Err(ConvolutionError::ExcitationFermiOutOfRange { .. })
        ));
        let (energy, xsec, xsnorm, chia) = xscorratan_fixture();
        assert!(matches!(
            ff2x_atan_correction(Ff2xAtanCorrectionInput {
                spectroscopy: 1,
                energy: energy.view(),
                horizontal_len: 0,
                fermi_index: 0,
                xsec: xsec.view(),
                xsnorm: xsnorm.view(),
                chia: chia.view(),
                real_correction: 0.0,
                imaginary_correction: 0.0,
            }),
            Err(ConvolutionError::AtanInvalidHorizontalLength { .. })
        ));
        assert!(matches!(
            ff2x_atan_correction(Ff2xAtanCorrectionInput {
                spectroscopy: 1,
                energy: energy.view(),
                horizontal_len: 5,
                fermi_index: 5,
                xsec: xsec.view(),
                xsnorm: xsnorm.view(),
                chia: chia.view(),
                real_correction: 0.0,
                imaginary_correction: 0.0,
            }),
            Err(ConvolutionError::AtanFermiIndexOutOfRange { .. })
        ));
    }
}
