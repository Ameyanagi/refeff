//! FEFF FPRIME/DANES numerical helpers.
//!
//! This module ports the compact helper routines at the end of
//! `FF2X/fprime.f90`. The top-level FEFF routine also handles file I/O and mesh
//! mutation; these helpers keep only the numerical pieces so callers can compose
//! them with Rust-owned input and output handling.

use ndarray::ArrayView1;
use thiserror::Error;

use crate::interpolation::{InterpolationError, terpc};
use crate::{Complex, Real};

const FPRIME_PI: Real = std::f64::consts::PI;
const FPRIME_EPS4: Real = 1.0e-4;

/// Analytic branch used by FEFF `funlog`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FprimeLogCase {
    /// FEFF `icase = 1`, the simplified anomalous correction expression.
    Simplified,
    /// FEFF `icase = 2`, expression for a real frequency.
    RealFrequency,
    /// FEFF `icase = 3`, expression for a purely imaginary frequency.
    ImaginaryFrequency,
}

/// Inputs for FEFF `fpint`.
#[derive(Debug, Clone, Copy)]
pub struct FprimeContourIntegralInput<'a> {
    /// Complex FEFF energy mesh, `emxs`.
    pub energy: ArrayView1<'a, Complex>,
    /// Complex `xmu` values on the same mesh.
    pub xmu: ArrayView1<'a, Complex>,
    /// Zero-based Rust equivalent of FEFF `n1`.
    pub start_index: usize,
    /// Zero-based inclusive Rust equivalent of FEFF `n2`.
    pub end_index: usize,
    /// Energy displacement from the Fermi level, FEFF `dele`.
    pub delta: Real,
    /// Lorentzian loss width, FEFF `xloss`.
    pub loss: Real,
    /// Small equality tolerance, FEFF `eps4`.
    pub epsilon: Real,
    /// Fermi energy used to shift `emxs`.
    pub fermi_energy: Real,
}

/// Inputs for FEFF `fpintp`.
#[derive(Debug, Clone, Copy)]
pub struct FprimePositiveAxisIntegralInput<'a> {
    /// Real positive-axis mesh, FEFF `em`.
    pub energy: ArrayView1<'a, Real>,
    /// Complex `xmu` values on `energy`.
    pub xmu: ArrayView1<'a, Complex>,
    /// Energy displacement from the Fermi level, FEFF `dele`.
    pub delta: Real,
    /// Lorentzian loss width, FEFF `xloss`.
    pub loss: Real,
    /// Fermi energy, FEFF `efermi`.
    pub fermi_energy: Real,
}

/// Error returned by FPRIME/DANES helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FprimeError {
    /// The analytic expression requires a known FEFF branch.
    #[error("invalid FPRIME logarithm branch")]
    InvalidLogCase,
    /// Scalar inputs must be finite.
    #[error("FPRIME {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Loss widths appear in denominators and logarithms.
    #[error("FPRIME loss width must be positive and finite, got {value}")]
    InvalidLoss { value: Real },
    /// Frequencies appear in logarithm denominators.
    #[error("FPRIME frequency must be finite and nonzero, got {value}")]
    InvalidFrequency { value: Real },
    /// Energy and spectrum arrays must match.
    #[error("FPRIME length mismatch: energy has {energy_len}, xmu has {xmu_len}")]
    LengthMismatch { energy_len: usize, xmu_len: usize },
    /// FEFF `fpint` needs a valid zero-based inclusive range.
    #[error(
        "FPRIME contour range start={start_index}, end={end_index} is invalid for length {len}"
    )]
    InvalidContourRange {
        start_index: usize,
        end_index: usize,
        len: usize,
    },
    /// FEFF `fpintp` uses cubic interpolation and a two-point tail model.
    #[error("FPRIME positive-axis integral requires at least 4 points, got {points}")]
    InsufficientPositiveAxisPoints { points: usize },
    /// Energy values must be finite.
    #[error("FPRIME energy row {row} must be finite, got ({real}, {imaginary})")]
    NonFiniteEnergy {
        row: usize,
        real: Real,
        imaginary: Real,
    },
    /// Positive-axis energy values must be finite.
    #[error("FPRIME real energy row {row} must be finite, got {value}")]
    NonFiniteRealEnergy { row: usize, value: Real },
    /// Spectrum values must be finite.
    #[error("FPRIME xmu row {row} must be finite, got ({real}, {imaginary})")]
    NonFiniteSpectrum {
        row: usize,
        real: Real,
        imaginary: Real,
    },
    /// Adjacent positive-axis energies must increase.
    #[error("FPRIME energy row {row} must increase, got {current} after {previous}")]
    NonIncreasingEnergy {
        row: usize,
        previous: Real,
        current: Real,
    },
    /// A denominator in the FEFF formula is zero.
    #[error("FPRIME denominator {field} is singular")]
    SingularDenominator { field: &'static str },
    /// A logarithm argument is zero or non-finite.
    #[error("FPRIME logarithm argument {field} is invalid: ({real}, {imaginary})")]
    InvalidLogArgument {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
    /// FEFF `fpintp` tail fitting requires a real non-negative square-root input.
    #[error("FPRIME tail square-root input must be real, finite, and non-negative, got {value}")]
    InvalidTailSquareRoot { value: Real },
    /// FEFF interpolation failed inside `fpintp`.
    #[error("FPRIME interpolation failed: {source}")]
    Interpolation { source: InterpolationError },
    /// The final value must remain finite.
    #[error("FPRIME {field} result is non-finite: ({real}, {imaginary})")]
    NonFiniteOutput {
        field: &'static str,
        real: Real,
        imaginary: Real,
    },
}

impl TryFrom<i32> for FprimeLogCase {
    type Error = FprimeError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Simplified),
            2 => Ok(Self::RealFrequency),
            3 => Ok(Self::ImaginaryFrequency),
            _ => Err(FprimeError::InvalidLogCase),
        }
    }
}

/// Port of FEFF `funlog`: anomalous FPRIME logarithmic correction.
pub fn fprime_log_correction(
    case: FprimeLogCase,
    loss: Real,
    frequency: Real,
    delta: Real,
) -> Result<Complex, FprimeError> {
    validate_loss(loss)?;
    validate_frequency(frequency)?;
    validate_scalar("delta", delta)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let frequency_c = Complex::new(frequency, 0.0);
    let value = match case {
        FprimeLogCase::Simplified if delta.abs() >= FPRIME_EPS4 => {
            let left = checked_log(
                Complex::new(-loss, delta) / frequency_c,
                "funlog simplified left",
            )?;
            let right = checked_log(
                Complex::new(loss, delta) / frequency_c,
                "funlog simplified right",
            )?;
            imaginary_unit / (2.0 * FPRIME_PI) * (left + right)
        }
        FprimeLogCase::Simplified => imaginary_unit / FPRIME_PI * (loss / frequency).abs().ln(),
        FprimeLogCase::RealFrequency if delta.abs() >= FPRIME_EPS4 => {
            let prefactor =
                imaginary_unit / (2.0 * FPRIME_PI) * (frequency_c + imaginary_unit * loss);
            let left_denominator = Complex::new(frequency + delta, loss);
            let right_denominator = Complex::new(frequency + delta, -loss);
            ensure_nonzero_complex(left_denominator, "funlog real left denominator")?;
            ensure_nonzero_complex(right_denominator, "funlog real right denominator")?;
            let left = checked_log(Complex::new(-loss, delta) / frequency_c, "funlog real left")?
                / left_denominator;
            let right = checked_log(Complex::new(loss, delta) / frequency_c, "funlog real right")?
                / right_denominator;
            prefactor * (left + right)
        }
        FprimeLogCase::RealFrequency => {
            let denominator = Complex::new(frequency, -loss);
            ensure_nonzero_complex(denominator, "funlog real zero denominator")?;
            imaginary_unit / FPRIME_PI
                * (loss / frequency).abs().ln()
                * (Complex::new(1.0, 0.0) + imaginary_unit * loss / denominator)
        }
        FprimeLogCase::ImaginaryFrequency if delta.abs() >= FPRIME_EPS4 => {
            let left_denominator = Complex::new(delta, frequency + loss);
            let right_denominator = Complex::new(delta, frequency - loss);
            ensure_nonzero_complex(left_denominator, "funlog imaginary left denominator")?;
            ensure_nonzero_complex(right_denominator, "funlog imaginary right denominator")?;
            let left = checked_log(
                Complex::new(-loss, delta) / frequency_c,
                "funlog imaginary left",
            )? / left_denominator;
            let right = checked_log(
                Complex::new(loss, delta) / frequency_c,
                "funlog imaginary right",
            )? / right_denominator;
            -(frequency + loss) / (2.0 * FPRIME_PI) * (left + right)
        }
        FprimeLogCase::ImaginaryFrequency => {
            let denominator = frequency - loss;
            if denominator == 0.0 {
                return Err(FprimeError::SingularDenominator {
                    field: "funlog imaginary zero denominator",
                });
            }
            imaginary_unit / FPRIME_PI
                * (loss / frequency).abs().ln()
                * (Complex::new(1.0, 0.0) + loss / denominator)
        }
    };

    ensure_finite_output("funlog", value)?;
    Ok(value)
}

/// Port of FEFF `fpint`: integrate along a complex contour segment.
///
/// `start_index` and `end_index` are zero-based. They correspond to FEFF's
/// one-based `n1` and inclusive `n2` after subtracting one.
pub fn fprime_contour_integral(
    input: FprimeContourIntegralInput<'_>,
) -> Result<Complex, FprimeError> {
    validate_contour_input(input)?;

    let imaginary_unit = Complex::new(0.0, 1.0);
    let delta = Complex::new(input.delta, 0.0);
    let loss_squared = input.loss * input.loss;

    let mut z1 = input.energy[input.end_index] - input.fermi_energy;
    let mut z2 = input.energy[input.end_index - 1] - input.fermi_energy;
    let denominator = Complex::new(loss_squared, 0.0) + (z1 - delta).powi(2);
    ensure_nonzero_complex(denominator, "fpint tail kernel")?;
    let mut value = -imaginary_unit / FPRIME_PI * (z1 - delta) / denominator
        * input.xmu[input.end_index]
        * (2.0 * (z1 - z2));

    if input.start_index < input.end_index.saturating_sub(1) {
        for index in input.start_index..(input.end_index - 1) {
            z1 = input.energy[index] - input.fermi_energy;
            z2 = input.energy[index + 1] - input.fermi_energy;
            let interval = z2 - z1;
            ensure_nonzero_complex(interval, "fpint interval")?;
            let bb = (input.xmu[index + 1] * (z2 - delta) - input.xmu[index] * (z1 - delta))
                / input.loss
                / interval;
            let aa = input.xmu[index] * (z1 - delta) / input.loss - bb * z1;

            let c1 = (aa + bb * (delta + imaginary_unit * input.loss)) / (2.0 * imaginary_unit);
            let negative_ratio = (z2 - delta - imaginary_unit * input.loss)
                / (z1 - delta - imaginary_unit * input.loss);
            let negative_log = if (input.delta - z1.re).abs() < input.epsilon
                && (input.delta - z2.re).abs() < input.epsilon
            {
                Complex::new(
                    checked_real_log(negative_ratio.norm(), "fpint negative real log")?,
                    0.0,
                )
            } else {
                checked_log(negative_ratio, "fpint negative log")?
            };
            value -= imaginary_unit / FPRIME_PI * c1 * negative_log;

            let c1 = -(aa + bb * (delta - imaginary_unit * input.loss)) / (2.0 * imaginary_unit);
            let positive_ratio = (z2 - delta + imaginary_unit * input.loss)
                / (z1 - delta + imaginary_unit * input.loss);
            value -=
                imaginary_unit / FPRIME_PI * c1 * checked_log(positive_ratio, "fpint positive")?;
        }
    }

    ensure_finite_output("fpint", value)?;
    Ok(value)
}

/// Port of FEFF `fpintp`: integrate along the positive real axis and tail.
pub fn fprime_positive_axis_integral(
    input: FprimePositiveAxisIntegralInput<'_>,
) -> Result<Complex, FprimeError> {
    validate_positive_axis_input(input)?;

    let energy = input.energy.to_vec();
    let xmu = input.xmu.to_vec();
    let imaginary_unit = Complex::new(0.0, 1.0);
    let mut value = Complex::new(0.0, 0.0);

    for index in 0..(energy.len() - 1) {
        let x1 = energy[index] - input.fermi_energy;
        let x2 = energy[index + 1] - input.fermi_energy;
        let half_width = (x2 - x1) / 2.0;
        if half_width == 0.0 {
            return Err(FprimeError::SingularDenominator {
                field: "fpintp interval width",
            });
        }
        let center = (energy[index] + energy[index + 1]) / 2.0;
        let interpolated = terpc(&energy, &xmu, 3, center)
            .map_err(|source| FprimeError::Interpolation { source })?
            .value;
        let slope = (xmu[index + 1] - xmu[index]) / (x2 - x1);
        let curvature =
            (xmu[index + 1] - interpolated - slope * half_width) / (half_width * half_width);

        let z1 = Complex::new(input.delta - center + input.fermi_energy, -input.loss);
        let z2 = Complex::new(input.delta - center + input.fermi_energy, input.loss);
        value += positive_axis_interval(half_width, interpolated, slope, curvature, z1)?;
        value += positive_axis_interval(half_width, interpolated, slope, curvature, z2)?;
    }

    let last = energy.len() - 1;
    let previous = energy.len() - 2;
    ensure_nonzero_complex(xmu[last], "fpintp tail xmu")?;
    let tail_sqrt_input = (xmu[previous] / xmu[last]).re;
    if !(tail_sqrt_input.is_finite() && tail_sqrt_input >= 0.0) {
        return Err(FprimeError::InvalidTailSquareRoot {
            value: tail_sqrt_input,
        });
    }
    let tail_scale = tail_sqrt_input.sqrt();
    if tail_scale == 1.0 {
        return Err(FprimeError::SingularDenominator {
            field: "fpintp tail scale",
        });
    }
    let x1 = energy[previous];
    let x2 = energy[last];
    let mut tail_shift = (tail_scale * x1 - x2) / (tail_scale - 1.0);
    if !tail_shift.is_finite() {
        return Err(FprimeError::NonFiniteScalar {
            field: "tail_shift",
            value: tail_shift,
        });
    }
    if tail_shift > x1 {
        tail_shift = 0.0;
    }

    let tail_weight = xmu[last] * (x2 - tail_shift).powi(2);
    let z1 = Complex::new(input.delta - tail_shift, -input.loss);
    let z2 = Complex::new(input.delta - tail_shift, input.loss);
    let x0 = x2 - tail_shift;
    if x0 == 0.0 {
        return Err(FprimeError::SingularDenominator {
            field: "fpintp tail x0",
        });
    }
    value += positive_axis_tail(tail_weight, z1, x0)?;
    value += positive_axis_tail(tail_weight, z2, x0)?;
    value *= -imaginary_unit / (2.0 * FPRIME_PI);

    ensure_finite_output("fpintp", value)?;
    Ok(value)
}

fn positive_axis_interval(
    half_width: Real,
    intercept: Complex,
    slope: Complex,
    curvature: Complex,
    pole: Complex,
) -> Result<Complex, FprimeError> {
    let width = Complex::new(half_width, 0.0);
    let numerator = width - pole;
    let denominator = -width - pole;
    ensure_nonzero_complex(denominator, "fpintp interval log denominator")?;
    let log_term = checked_log(numerator / denominator, "fpintp interval log")?;
    Ok(2.0 * half_width * slope
        + 2.0 * pole * half_width * curvature
        + log_term * (intercept + slope * pole + curvature * pole.powi(2)))
}

fn positive_axis_tail(
    tail_weight: Complex,
    pole: Complex,
    x0: Real,
) -> Result<Complex, FprimeError> {
    ensure_nonzero_complex(pole, "fpintp tail pole")?;
    let shifted = Complex::new(x0, 0.0) - pole;
    ensure_nonzero_complex(shifted, "fpintp tail shifted pole")?;
    Ok(
        checked_log(Complex::new(x0, 0.0) / shifted, "fpintp tail log")? * tail_weight
            / pole.powi(2)
            - tail_weight / pole / x0,
    )
}

fn validate_contour_input(input: FprimeContourIntegralInput<'_>) -> Result<(), FprimeError> {
    validate_matching_complex_arrays(input.energy, input.xmu)?;
    validate_loss(input.loss)?;
    validate_scalar("delta", input.delta)?;
    validate_scalar("epsilon", input.epsilon)?;
    validate_scalar("fermi_energy", input.fermi_energy)?;
    if input.end_index >= input.energy.len()
        || input.end_index == 0
        || input.start_index > input.end_index
    {
        return Err(FprimeError::InvalidContourRange {
            start_index: input.start_index,
            end_index: input.end_index,
            len: input.energy.len(),
        });
    }
    if input.epsilon < 0.0 {
        return Err(FprimeError::NonFiniteScalar {
            field: "epsilon",
            value: input.epsilon,
        });
    }
    Ok(())
}

fn validate_positive_axis_input(
    input: FprimePositiveAxisIntegralInput<'_>,
) -> Result<(), FprimeError> {
    if input.energy.len() != input.xmu.len() {
        return Err(FprimeError::LengthMismatch {
            energy_len: input.energy.len(),
            xmu_len: input.xmu.len(),
        });
    }
    if input.energy.len() < 4 {
        return Err(FprimeError::InsufficientPositiveAxisPoints {
            points: input.energy.len(),
        });
    }
    validate_loss(input.loss)?;
    validate_scalar("delta", input.delta)?;
    validate_scalar("fermi_energy", input.fermi_energy)?;
    for (row, &energy) in input.energy.iter().enumerate() {
        if !energy.is_finite() {
            return Err(FprimeError::NonFiniteRealEnergy { row, value: energy });
        }
        if row > 0 && energy <= input.energy[row - 1] {
            return Err(FprimeError::NonIncreasingEnergy {
                row,
                previous: input.energy[row - 1],
                current: energy,
            });
        }
    }
    for (row, &value) in input.xmu.iter().enumerate() {
        validate_complex_spectrum(row, value)?;
    }
    Ok(())
}

fn validate_matching_complex_arrays(
    energy: ArrayView1<'_, Complex>,
    xmu: ArrayView1<'_, Complex>,
) -> Result<(), FprimeError> {
    if energy.len() != xmu.len() {
        return Err(FprimeError::LengthMismatch {
            energy_len: energy.len(),
            xmu_len: xmu.len(),
        });
    }
    for (row, &value) in energy.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return Err(FprimeError::NonFiniteEnergy {
                row,
                real: value.re,
                imaginary: value.im,
            });
        }
    }
    for (row, &value) in xmu.iter().enumerate() {
        validate_complex_spectrum(row, value)?;
    }
    Ok(())
}

fn validate_complex_spectrum(row: usize, value: Complex) -> Result<(), FprimeError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FprimeError::NonFiniteSpectrum {
            row,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn validate_loss(value: Real) -> Result<(), FprimeError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(FprimeError::InvalidLoss { value })
    }
}

fn validate_frequency(value: Real) -> Result<(), FprimeError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(FprimeError::InvalidFrequency { value })
    }
}

fn validate_scalar(field: &'static str, value: Real) -> Result<(), FprimeError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FprimeError::NonFiniteScalar { field, value })
    }
}

fn checked_real_log(value: Real, field: &'static str) -> Result<Real, FprimeError> {
    if value.is_finite() && value > 0.0 {
        Ok(value.ln())
    } else {
        Err(FprimeError::InvalidLogArgument {
            field,
            real: value,
            imaginary: 0.0,
        })
    }
}

fn checked_log(value: Complex, field: &'static str) -> Result<Complex, FprimeError> {
    if !is_finite_complex(value) || is_zero_complex(value) {
        return Err(FprimeError::InvalidLogArgument {
            field,
            real: value.re,
            imaginary: value.im,
        });
    }
    let result = value.ln();
    ensure_finite_output(field, result)?;
    Ok(result)
}

fn ensure_nonzero_complex(value: Complex, field: &'static str) -> Result<(), FprimeError> {
    if is_finite_complex(value) && !is_zero_complex(value) {
        Ok(())
    } else {
        Err(FprimeError::SingularDenominator { field })
    }
}

fn ensure_finite_output(field: &'static str, value: Complex) -> Result<(), FprimeError> {
    if is_finite_complex(value) {
        Ok(())
    } else {
        Err(FprimeError::NonFiniteOutput {
            field,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn is_finite_complex(value: Complex) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn is_zero_complex(value: Complex) -> bool {
    value.re == 0.0 && value.im == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn assert_complex_close(actual: Complex, expected: Complex) {
        assert!(
            (actual - expected).norm() <= 1.0e-12 * expected.norm().max(1.0),
            "actual={actual:?}, expected={expected:?}, diff={}",
            (actual - expected).norm()
        );
    }

    #[test]
    fn fprime_log_correction_matches_feff_funlog_reference() -> Result<(), FprimeError> {
        let cases = [
            (
                FprimeLogCase::Simplified,
                0.08,
                0.50,
                -0.13,
                Complex::new(0.5, -0.377_675_882_737_221_64),
            ),
            (
                FprimeLogCase::Simplified,
                0.08,
                0.50,
                0.0,
                Complex::new(0.0, -0.583_328_797_148_249_1),
            ),
            (
                FprimeLogCase::RealFrequency,
                0.08,
                0.50,
                0.21,
                Complex::new(-0.320_826_881_246_067_6, -0.223_578_319_342_342_12),
            ),
            (
                FprimeLogCase::RealFrequency,
                0.08,
                0.50,
                0.0,
                Complex::new(0.091_002_932_472_425_77, -0.568_768_327_952_661),
            ),
            (
                FprimeLogCase::ImaginaryFrequency,
                0.08,
                0.50,
                -0.17,
                Complex::new(0.391_399_518_578_323_34, -0.501_116_900_028_590_1),
            ),
            (
                FprimeLogCase::ImaginaryFrequency,
                0.08,
                0.50,
                0.0,
                Complex::new(0.0, -0.694_439_044_224_106_1),
            ),
        ];

        for (case, loss, frequency, delta, expected) in cases {
            assert_complex_close(
                fprime_log_correction(case, loss, frequency, delta)?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn fprime_contour_integral_matches_feff_fpint_reference() -> Result<(), FprimeError> {
        let energy =
            Array1::from_iter((0..8).map(|index| {
                Complex::new(-0.05 + 0.07 * index as Real, 0.02 + 0.025 * index as Real)
            }));
        let xmu = Array1::from_iter((1..=8).map(|index| {
            let index = index as Real;
            Complex::new(
                0.7 + 0.08 * index + 0.01 * index * index,
                -0.04 + 0.03 * index,
            )
        }));

        let first = fprime_contour_integral(FprimeContourIntegralInput {
            energy: energy.view(),
            xmu: xmu.view(),
            start_index: 1,
            end_index: 7,
            delta: 0.11,
            loss: 0.08,
            epsilon: 1.0e-4,
            fermi_energy: 0.03,
        })?;
        assert_complex_close(
            first,
            Complex::new(-0.817_543_004_170_169_5, -0.702_901_398_468_171_2),
        );

        let second = fprime_contour_integral(FprimeContourIntegralInput {
            energy: energy.view(),
            xmu: xmu.view(),
            start_index: 2,
            end_index: 6,
            delta: -0.06,
            loss: 0.08,
            epsilon: 1.0e-4,
            fermi_energy: 0.03,
        })?;
        assert_complex_close(
            second,
            Complex::new(0.031_741_876_462_853_56, -0.536_228_705_848_155_4),
        );
        Ok(())
    }

    #[test]
    fn fprime_positive_axis_integral_matches_feff_fpintp_reference() -> Result<(), FprimeError> {
        let energy = Array1::from_iter(
            (1..=7).map(|index| 0.02 + 0.12 * (index - 1) as Real + 0.01 * (index % 2) as Real),
        );
        let xmu = Array1::from_iter((1..=7).map(|index| {
            let index = index as Real;
            Complex::new(
                0.5 + 0.07 * index + 0.02 * index * index,
                0.03 * index - 0.01,
            )
        }));

        let first = fprime_positive_axis_integral(FprimePositiveAxisIntegralInput {
            energy: energy.view(),
            xmu: xmu.view(),
            delta: 0.09,
            loss: 0.08,
            fermi_energy: 0.03,
        })?;
        assert_complex_close(
            first,
            Complex::new(0.100_296_256_238_197_68, -0.986_754_784_699_495),
        );

        let second = fprime_positive_axis_integral(FprimePositiveAxisIntegralInput {
            energy: energy.view(),
            xmu: xmu.view(),
            delta: -0.14,
            loss: 0.05,
            fermi_energy: -0.02,
        })?;
        assert_complex_close(
            second,
            Complex::new(0.072_117_898_021_539_17, -0.772_729_822_158_172_4),
        );
        Ok(())
    }

    #[test]
    fn fprime_helpers_reject_invalid_inputs() {
        assert!(matches!(
            FprimeLogCase::try_from(9),
            Err(FprimeError::InvalidLogCase)
        ));
        assert!(matches!(
            fprime_log_correction(FprimeLogCase::Simplified, 0.0, 0.5, 0.1),
            Err(FprimeError::InvalidLoss { .. })
        ));
        assert!(matches!(
            fprime_log_correction(FprimeLogCase::Simplified, 0.08, 0.0, 0.1),
            Err(FprimeError::InvalidFrequency { .. })
        ));

        let energy = Array1::from_vec(vec![Complex::new(0.0, 0.1), Complex::new(0.1, 0.2)]);
        let xmu = Array1::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(1.1, 0.1)]);
        assert!(matches!(
            fprime_contour_integral(FprimeContourIntegralInput {
                energy: energy.view(),
                xmu: xmu.view(),
                start_index: 0,
                end_index: 2,
                delta: 0.0,
                loss: 0.1,
                epsilon: 1.0e-4,
                fermi_energy: 0.0,
            }),
            Err(FprimeError::InvalidContourRange { .. })
        ));

        let real_energy = Array1::from_vec(vec![0.0, 0.1, 0.1, 0.3]);
        let xmu = Array1::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.1, 0.1),
            Complex::new(1.2, 0.2),
            Complex::new(1.3, 0.3),
        ]);
        assert!(matches!(
            fprime_positive_axis_integral(FprimePositiveAxisIntegralInput {
                energy: real_energy.view(),
                xmu: xmu.view(),
                delta: 0.1,
                loss: 0.1,
                fermi_energy: 0.0,
            }),
            Err(FprimeError::NonIncreasingEnergy { row: 2, .. })
        ));
    }
}
