//! FEFF FPRIME/DANES numerical routines.
//!
//! This module ports the numerical pieces from `FF2X/fprime.f90`. The original
//! FEFF routine also handles file I/O and mutates its energy mesh in-place; the
//! Rust API keeps those side effects out of the core calculation.

use ndarray::{Array1, ArrayView1};
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

/// Inputs for FEFF `fprime`.
#[derive(Debug, Clone, Copy)]
pub struct FprimeCorrectionInput<'a> {
    /// FEFF `ei`, the edge/Fermi reference used for the negative-frequency pole.
    pub edge_reference_energy: Real,
    /// Complex FEFF energy mesh, `emxs`.
    pub energy: ArrayView1<'a, Complex>,
    /// Number of output points on the main horizontal grid, FEFF `ne1`.
    pub main_energy_count: usize,
    /// Number of positive-axis extension points, FEFF `ne3`.
    pub extension_count: usize,
    /// Zero-based Rust equivalent of FEFF `ik0`.
    pub fermi_index: usize,
    /// Converted cross section, FEFF `xsec`.
    pub cross_section: ArrayView1<'a, Complex>,
    /// Converted atomic background, FEFF `xsnorm`.
    pub background: ArrayView1<'a, Real>,
    /// Path/configuration fine structure, FEFF `chia`.
    pub path_chi: ArrayView1<'a, Complex>,
    /// Real energy-zero correction, FEFF `vrcorr`.
    pub real_correction: Real,
    /// Imaginary broadening correction, FEFF `vicorr`.
    pub imaginary_correction: Real,
}

/// FEFF `danes.dat` diagnostic columns produced while evaluating `fprime`.
#[derive(Debug, Clone, PartialEq)]
pub struct FprimeDanesDiagnostics {
    /// Matsubara contributions from the positive- and negative-frequency poles.
    pub matsubara: Array1<Real>,
    /// Sommerfeld contribution from the positive-frequency pole.
    pub sommerfeld: Array1<Real>,
    /// FEFF's local-plus-logarithmic anomalous contribution.
    pub anomalous: Array1<Real>,
    /// Positive-axis integral contribution, FEFF's historically named `tale`.
    pub tail: Array1<Real>,
    /// Real part of the corrected atomic spectrum.
    pub total: Array1<Real>,
    /// `total - anomalous`.
    pub difference: Array1<Real>,
}

/// FEFF `fprime` correction and optional DANES diagnostics from one evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct FprimeCorrectionOutput {
    /// FEFF `cchi(1:ne1)`.
    pub correction: Array1<Complex>,
    /// Diagnostic columns when the input contains a DANES vertical contour.
    pub danes_diagnostics: Option<FprimeDanesDiagnostics>,
}

/// Error returned by FPRIME/DANES helper routines.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
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
    /// Top-level FEFF `fprime` arrays must have the same mesh length.
    #[error("FPRIME {field} length mismatch: expected {expected}, got {actual}")]
    ArrayLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FEFF `ne1`, `ne2`, and `ne3` must describe a usable mesh partition.
    #[error(
        "FPRIME mesh partition main={main_energy_count}, extension={extension_count} is invalid for length {len}"
    )]
    InvalidMeshPartition {
        main_energy_count: usize,
        extension_count: usize,
        len: usize,
    },
    /// FEFF `ik0` must point into the main horizontal grid.
    #[error("FPRIME fermi index {fermi_index} is invalid for {main_energy_count} main points")]
    InvalidFermiIndex {
        fermi_index: usize,
        main_energy_count: usize,
    },
    /// DANES needs at least three vertical contour points for FEFF's correction.
    #[error("FPRIME DANES vertical contour requires at least 3 points, got {points}")]
    InsufficientVerticalContourPoints { points: usize },
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
    /// Converted background values must be finite.
    #[error("FPRIME background row {row} must be finite, got {value}")]
    NonFiniteBackground { row: usize, value: Real },
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

/// Port of FEFF `fprime`: solid-state/lifetime correction for FPRIME/DANES.
///
/// The returned array is FEFF `cchi(1:ne1)`. The caller remains responsible for
/// assembling `xmu.dat` columns and any diagnostic `danes.dat` output.
pub fn fprime_correction(input: FprimeCorrectionInput<'_>) -> Result<Array1<Complex>, FprimeError> {
    Ok(fprime_correction_with_diagnostics(input)?.correction)
}

/// Port of FEFF `fprime` that also retains the exact `danes.dat` components.
///
/// DANES diagnostics are available only when the mesh contains a vertical
/// contour (`ne2 > 0`). Keeping them inside this evaluation avoids
/// reconstructing the anomalous terms from a bare cross section.
pub fn fprime_correction_with_diagnostics(
    input: FprimeCorrectionInput<'_>,
) -> Result<FprimeCorrectionOutput, FprimeError> {
    validate_correction_input(input)?;

    let mut energy = input.energy.to_vec();
    let ne = energy.len();
    let ne1 = input.main_energy_count;
    let ne3 = input.extension_count;
    let ne2 = ne - ne1 - ne3;
    let imaginary_unit = Complex::new(0.0, 1.0);

    let mut fermi_energy = energy[ne1].re;
    let mut loss = energy[0].im;
    let mut xmu = vec![Complex::new(0.0, 0.0); ne];

    if ne2 > 0 {
        for (row, value) in xmu.iter_mut().take(ne1).enumerate() {
            *value = imaginary_unit * input.background[row]
                + input.background[row] * input.path_chi[row];
        }
        for (row, value) in xmu.iter_mut().enumerate().skip(ne1).take(ne2) {
            *value = input.background[row] * input.path_chi[row];
        }
        for (row, value) in xmu.iter_mut().enumerate().skip(ne - ne3) {
            *value = imaginary_unit * input.background[row];
        }
    } else {
        for (row, value) in xmu.iter_mut().enumerate() {
            *value = input.cross_section[row] + input.background[row] * input.path_chi[row];
        }
    }

    if input.real_correction.abs() > FPRIME_EPS4 && ne2 > 0 {
        let main_energy: Vec<Real> = energy[..ne1].iter().map(|value| value.re).collect();
        let main_xmu = xmu[..ne1].to_vec();
        fermi_energy -= input.real_correction;
        let mut rescale = terpc(&main_energy, &main_xmu, 1, fermi_energy)
            .map_err(|source| FprimeError::Interpolation { source })?
            .value;
        for value in &mut energy[ne1..(ne - ne3)] {
            value.re -= input.real_correction;
        }
        if xmu[input.fermi_index].norm() > FPRIME_EPS4 {
            ensure_nonzero_complex(xmu[input.fermi_index], "fprime vrcorr fermi xmu")?;
            rescale /= xmu[input.fermi_index];
        }
        for value in &mut xmu[ne1..(ne - ne3)] {
            *value *= rescale;
        }
    }

    if input.imaginary_correction > FPRIME_EPS4 && ne2 > 0 {
        loss += input.imaginary_correction;
        validate_loss(loss)?;
        let vertical_width: Vec<Real> = energy[ne1..(ne1 + ne2)]
            .iter()
            .map(|value| value.im)
            .collect();
        let vertical_xmu = xmu[ne1..(ne1 + ne2)].to_vec();
        let vertical_anchor = terpc(&vertical_width, &vertical_xmu, 1, loss)
            .map_err(|source| FprimeError::Interpolation { source })?
            .value;
        let broadening_squared = input.imaginary_correction * input.imaginary_correction;
        for row in 0..ne1 {
            let displacement = energy[row].re - fermi_energy;
            let denominator = broadening_squared + displacement * displacement;
            if denominator == 0.0 {
                return Err(FprimeError::SingularDenominator {
                    field: "fprime vicorr weight",
                });
            }
            let weight = broadening_squared / denominator;
            xmu[row] = xmu[row] * (1.0 - weight) + vertical_anchor * weight;
            energy[row] += imaginary_unit * input.imaginary_correction;
        }
    }

    let (positive_energy, positive_xmu) = fprime_positive_axis_values(input, &energy, &xmu, ne2);
    let mut correction = Vec::with_capacity(ne1);
    let mut diagnostic_matsubara = Vec::with_capacity(ne1);
    let mut diagnostic_sommerfeld = Vec::with_capacity(ne1);
    let mut diagnostic_anomalous = Vec::with_capacity(ne1);
    let mut diagnostic_tail = Vec::with_capacity(ne1);
    let mut diagnostic_total = Vec::with_capacity(ne1);
    let mut diagnostic_difference = Vec::with_capacity(ne1);
    for row in 0..ne1 {
        let raw_delta = energy[row].re - fermi_energy;
        let negative_delta = -raw_delta - 2.0 * input.edge_reference_energy;
        let integration_delta = if ne2 > 0 && raw_delta.abs() < FPRIME_EPS4 {
            0.0
        } else {
            raw_delta
        };
        let mut value = Complex::new(0.0, 0.0);
        let mut matsubara_diagnostic = 0.0;
        let mut sommerfeld_diagnostic = 0.0;

        if ne2 > 0 {
            let w1 = energy[ne1].im;
            let w2 = energy[ne1 + 1].im;
            let w3 = energy[ne1 + 2].im;
            let vertical_widths = [w1, w2, w3];
            let vertical_xmu = [xmu[ne1], xmu[ne1 + 1], xmu[ne1 + 2]];
            let positive_components = fprime_danes_pole_components(
                loss,
                vertical_widths,
                vertical_xmu,
                integration_delta,
            )?;
            let negative_components =
                fprime_danes_pole_components(loss, vertical_widths, vertical_xmu, negative_delta)?;
            value += positive_components.total();
            value += negative_components.total();
            matsubara_diagnostic =
                (positive_components.matsubara + negative_components.matsubara).re;
            sommerfeld_diagnostic = positive_components.sommerfeld.re;

            if integration_delta < FPRIME_EPS4 {
                value -= xmu[row];
            }
            if integration_delta.abs() < FPRIME_EPS4 {
                value += xmu[row] / 2.0;
            }

            value += fprime_contour_integral(FprimeContourIntegralInput {
                energy: ArrayView1::from(&energy[..]),
                xmu: ArrayView1::from(&xmu[..]),
                start_index: ne1 + 1,
                end_index: ne - ne3 - 1,
                delta: integration_delta,
                loss,
                epsilon: FPRIME_EPS4,
                fermi_energy,
            })?;
            value += fprime_contour_integral(FprimeContourIntegralInput {
                energy: ArrayView1::from(&energy[..]),
                xmu: ArrayView1::from(&xmu[..]),
                start_index: ne1 + 1,
                end_index: ne - ne3 - 1,
                delta: negative_delta,
                loss,
                epsilon: FPRIME_EPS4,
                fermi_energy,
            })?;
        }

        let tail = fprime_positive_axis_contribution(
            positive_energy.view(),
            positive_xmu.view(),
            integration_delta,
            negative_delta,
            loss,
            fermi_energy,
        )?;
        value += tail;

        ensure_finite_output("fprime", value)?;
        if ne2 > 0 {
            let mut anomalous = Complex::new(0.0, 0.0);
            if integration_delta >= FPRIME_EPS4 {
                anomalous = xmu[row];
            }
            if integration_delta.abs() < FPRIME_EPS4 {
                anomalous = xmu[row] / 2.0;
            }
            anomalous += xmu[input.fermi_index]
                * fprime_log_correction(
                    FprimeLogCase::Simplified,
                    loss,
                    2.0 * input.edge_reference_energy,
                    integration_delta,
                )?;
            let total = (xmu[row] + value).re;
            let difference = total - anomalous.re;
            for (field, diagnostic) in [
                ("fprime diagnostic matsubara", matsubara_diagnostic),
                ("fprime diagnostic sommerfeld", sommerfeld_diagnostic),
                ("fprime diagnostic anomalous", anomalous.re),
                ("fprime diagnostic tail", tail.re),
                ("fprime diagnostic total", total),
                ("fprime diagnostic difference", difference),
            ] {
                ensure_finite_output(field, Complex::new(diagnostic, 0.0))?;
            }
            diagnostic_matsubara.push(matsubara_diagnostic);
            diagnostic_sommerfeld.push(sommerfeld_diagnostic);
            diagnostic_anomalous.push(anomalous.re);
            diagnostic_tail.push(tail.re);
            diagnostic_total.push(total);
            diagnostic_difference.push(difference);
        }
        correction.push(value);
    }

    let danes_diagnostics = (ne2 > 0).then(|| FprimeDanesDiagnostics {
        matsubara: Array1::from_vec(diagnostic_matsubara),
        sommerfeld: Array1::from_vec(diagnostic_sommerfeld),
        anomalous: Array1::from_vec(diagnostic_anomalous),
        tail: Array1::from_vec(diagnostic_tail),
        total: Array1::from_vec(diagnostic_total),
        difference: Array1::from_vec(diagnostic_difference),
    });
    Ok(FprimeCorrectionOutput {
        correction: Array1::from_vec(correction),
        danes_diagnostics,
    })
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

#[derive(Debug, Clone, Copy)]
struct FprimeDanesPoleComponents {
    matsubara: Complex,
    sommerfeld: Complex,
}

impl FprimeDanesPoleComponents {
    fn total(self) -> Complex {
        self.matsubara + self.sommerfeld
    }
}

fn fprime_danes_pole_components(
    loss: Real,
    widths: [Real; 3],
    xmu: [Complex; 3],
    delta: Real,
) -> Result<FprimeDanesPoleComponents, FprimeError> {
    let [w1, w2, w3] = widths;
    let [xmu1, xmu2, xmu3] = xmu;
    validate_scalar("vertical w1", w1)?;
    validate_scalar("vertical w2", w2)?;
    validate_scalar("vertical w3", w3)?;
    let interval = w3 - w2;
    if interval == 0.0 {
        return Err(FprimeError::SingularDenominator {
            field: "fprime vertical interval",
        });
    }

    let imaginary_unit = Complex::new(0.0, 1.0);
    let matsubara = fprime_lorentz_kernel(loss, w1, delta)? * xmu1 * 2.0 * imaginary_unit * w1;
    let sommerfeld = imaginary_unit * w1 * w1 / 6.0
        * (fprime_lorentz_kernel(loss, w3, delta)? * xmu3
            - fprime_lorentz_kernel(loss, w2, delta)? * xmu2)
        / interval;
    ensure_finite_output("fprime danes Matsubara", matsubara)?;
    ensure_finite_output("fprime danes Sommerfeld", sommerfeld)?;
    Ok(FprimeDanesPoleComponents {
        matsubara,
        sommerfeld,
    })
}

fn fprime_lorentz_kernel(
    loss: Real,
    vertical_frequency: Real,
    delta: Real,
) -> Result<Complex, FprimeError> {
    validate_loss(loss)?;
    validate_scalar("vertical_frequency", vertical_frequency)?;
    validate_scalar("delta", delta)?;

    let pole = Complex::new(-delta, vertical_frequency);
    let denominator = Complex::new(loss * loss, 0.0) + pole.powi(2);
    ensure_nonzero_complex(denominator, "fprime lorentz")?;
    let value = loss / FPRIME_PI / denominator;
    ensure_finite_output("fprime lorentz", value)?;
    Ok(value)
}

fn fprime_positive_axis_values(
    input: FprimeCorrectionInput<'_>,
    energy: &[Complex],
    xmu: &[Complex],
    contour_count: usize,
) -> (Array1<Real>, Array1<Complex>) {
    let ne = energy.len();
    let ne1 = input.main_energy_count;
    let ne3 = input.extension_count;
    let mut positive_energy = Vec::new();
    let mut positive_xmu = Vec::new();

    if contour_count > 0 {
        positive_energy.reserve(ne1 - input.fermi_index + ne3);
        positive_xmu.reserve(ne1 - input.fermi_index + ne3);
        for (row, value) in energy.iter().enumerate().take(ne1).skip(input.fermi_index) {
            positive_energy.push(value.re);
            positive_xmu.push(Complex::new(0.0, input.background[row]));
        }
        for (&energy_value, &xmu_value) in energy[(ne - ne3)..ne].iter().zip(&xmu[(ne - ne3)..ne]) {
            positive_energy.push(energy_value.re);
            positive_xmu.push(xmu_value);
        }
    } else {
        positive_energy.reserve(ne3);
        positive_xmu.reserve(ne3);
        for (&energy_value, &xmu_value) in
            energy[ne1..(ne1 + ne3)].iter().zip(&xmu[ne1..(ne1 + ne3)])
        {
            positive_energy.push(energy_value.re);
            positive_xmu.push(xmu_value);
        }
    }

    (
        Array1::from_vec(positive_energy),
        Array1::from_vec(positive_xmu),
    )
}

fn fprime_positive_axis_contribution(
    energy: ArrayView1<'_, Real>,
    xmu: ArrayView1<'_, Complex>,
    delta: Real,
    negative_delta: Real,
    loss: Real,
    fermi_energy: Real,
) -> Result<Complex, FprimeError> {
    let mut value = fprime_positive_axis_integral(FprimePositiveAxisIntegralInput {
        energy,
        xmu,
        delta,
        loss,
        fermi_energy,
    })?;
    value += fprime_positive_axis_integral(FprimePositiveAxisIntegralInput {
        energy,
        xmu,
        delta: negative_delta,
        loss,
        fermi_energy,
    })?;
    ensure_finite_output("fprime positive axis", value)?;
    Ok(value)
}

fn validate_correction_input(input: FprimeCorrectionInput<'_>) -> Result<(), FprimeError> {
    let len = input.energy.len();
    validate_scalar("edge_reference_energy", input.edge_reference_energy)?;
    validate_scalar("real_correction", input.real_correction)?;
    validate_scalar("imaginary_correction", input.imaginary_correction)?;
    if input.main_energy_count == 0
        || input.main_energy_count >= len
        || input.extension_count > len - input.main_energy_count
    {
        return Err(FprimeError::InvalidMeshPartition {
            main_energy_count: input.main_energy_count,
            extension_count: input.extension_count,
            len,
        });
    }
    if input.fermi_index >= input.main_energy_count {
        return Err(FprimeError::InvalidFermiIndex {
            fermi_index: input.fermi_index,
            main_energy_count: input.main_energy_count,
        });
    }

    validate_array_length("cross_section", input.cross_section.len(), len)?;
    validate_array_length("background", input.background.len(), len)?;
    validate_array_length("path_chi", input.path_chi.len(), len)?;
    for (row, &value) in input.energy.iter().enumerate() {
        if !(value.re.is_finite() && value.im.is_finite()) {
            return Err(FprimeError::NonFiniteEnergy {
                row,
                real: value.re,
                imaginary: value.im,
            });
        }
    }
    for (row, &value) in input.cross_section.iter().enumerate() {
        validate_complex_spectrum(row, value)?;
    }
    for (row, &value) in input.background.iter().enumerate() {
        if !value.is_finite() {
            return Err(FprimeError::NonFiniteBackground { row, value });
        }
    }
    for (row, &value) in input.path_chi.iter().enumerate() {
        validate_complex_spectrum(row, value)?;
    }

    let contour_count = len - input.main_energy_count - input.extension_count;
    if contour_count > 0 {
        validate_loss(input.energy[0].im)?;
        if contour_count < 3 {
            return Err(FprimeError::InsufficientVerticalContourPoints {
                points: contour_count,
            });
        }
    } else {
        // FEFF FPRIME commonly places its horizontal mesh directly on the
        // real axis (`xloss = Im(emxs(1)) = 0`).  Unlike the DANES contour
        // branch, fpintp does not divide by xloss and evaluates the two
        // conjugate poles at the same real location when it is zero.
        validate_nonnegative_loss(input.energy[0].im)?;
    }
    let positive_axis_points = if contour_count > 0 {
        input.main_energy_count - input.fermi_index + input.extension_count
    } else {
        input.extension_count
    };
    if positive_axis_points < 4 {
        return Err(FprimeError::InsufficientPositiveAxisPoints {
            points: positive_axis_points,
        });
    }

    Ok(())
}

fn validate_array_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), FprimeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FprimeError::ArrayLengthMismatch {
            field,
            expected,
            actual,
        })
    }
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
    validate_nonnegative_loss(input.loss)?;
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

fn validate_nonnegative_loss(value: Real) -> Result<(), FprimeError> {
    if value.is_finite() && value >= 0.0 {
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

    fn assert_real_array_close(actual: &Array1<Real>, expected: &[Real]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
                "actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
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
    fn fprime_correction_matches_feff_fprime_reference() -> Result<(), FprimeError> {
        let energy = Array1::from_vec(vec![
            Complex::new(-0.08, 0.06),
            Complex::new(0.02, 0.06),
            Complex::new(0.15, 0.06),
            Complex::new(0.31, 0.06),
            Complex::new(0.38, 0.0),
            Complex::new(0.55, 0.0),
            Complex::new(0.80, 0.0),
            Complex::new(1.10, 0.0),
            Complex::new(1.70, 0.0),
            Complex::new(2.60, 0.0),
        ]);
        let cross_section = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(
                0.4 + 0.03 * index + 0.002 * index * index,
                -0.08 + 0.015 * index,
            )
        }));
        let background = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            0.7 + 0.05 * index + 0.001 * index * index
        }));
        let path_chi = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(0.02 * index - 0.03, 0.01 * index + 0.005)
        }));

        let correction = fprime_correction(FprimeCorrectionInput {
            edge_reference_energy: 0.38,
            energy: energy.view(),
            main_energy_count: 4,
            extension_count: 6,
            fermi_index: 0,
            cross_section: cross_section.view(),
            background: background.view(),
            path_chi: path_chi.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        })?;
        let expected = [
            Complex::new(0.215_889_828_593_304_38, -1.416_291_438_510_892_7),
            Complex::new(0.214_811_886_924_339_1, -1.406_730_739_506_308_7),
            Complex::new(0.218_942_674_544_931_27, -1.443_618_359_654_226_5),
            Complex::new(0.236_657_560_127_544_58, -1.607_767_310_949_670_4),
        ];

        assert_eq!(correction.len(), expected.len());
        for (&actual, expected) in correction.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn fprime_correction_accepts_feff_zero_loss_fprime_mesh() -> Result<(), FprimeError> {
        let energy = Array1::from_vec(vec![
            Complex::new(-0.08, 0.0),
            Complex::new(0.02, 0.0),
            Complex::new(0.15, 0.0),
            Complex::new(0.31, 0.0),
            Complex::new(0.38, 0.0),
            Complex::new(0.55, 0.0),
            Complex::new(0.80, 0.0),
            Complex::new(1.10, 0.0),
            Complex::new(1.70, 0.0),
            Complex::new(2.60, 0.0),
        ]);
        let cross_section = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(
                0.4 + 0.03 * index + 0.002 * index * index,
                -0.08 + 0.015 * index,
            )
        }));
        let background = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            0.7 + 0.05 * index + 0.001 * index * index
        }));
        let path_chi = Array1::from_iter((1..=10).map(|index| {
            let index = index as Real;
            Complex::new(0.02 * index - 0.03, 0.01 * index + 0.005)
        }));

        let correction = fprime_correction(FprimeCorrectionInput {
            edge_reference_energy: 0.38,
            energy: energy.view(),
            main_energy_count: 4,
            extension_count: 6,
            fermi_index: 0,
            cross_section: cross_section.view(),
            background: background.view(),
            path_chi: path_chi.view(),
            real_correction: 0.0,
            imaginary_correction: 0.0,
        })?;

        assert_eq!(correction.len(), 4);
        assert!(
            correction
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        Ok(())
    }

    #[test]
    fn fprime_correction_matches_feff_danes_reference() -> Result<(), FprimeError> {
        let energy = Array1::from_vec(vec![
            Complex::new(0.12, 0.07),
            Complex::new(0.24, 0.07),
            Complex::new(0.34, 0.07),
            Complex::new(0.42, 0.07),
            Complex::new(0.58, 0.07),
            Complex::new(0.42, 0.035),
            Complex::new(0.42, 0.070),
            Complex::new(0.42, 0.120),
            Complex::new(0.42, 0.200),
            Complex::new(0.66, 1.0e-8),
            Complex::new(0.95, 1.0e-8),
            Complex::new(1.35, 1.0e-8),
            Complex::new(2.10, 1.0e-8),
        ]);
        let cross_section = Array1::from_iter((1..=13).map(|index| {
            let index = index as Real;
            Complex::new(0.3 + 0.02 * index, -0.04 + 0.01 * index)
        }));
        let background = Array1::from_iter((1..=13).map(|index| {
            let index = index as Real;
            0.9 + 0.04 * index + 0.002 * index * index
        }));
        let path_chi = Array1::from_iter((1..=13).map(|index| {
            let index = index as Real;
            Complex::new(-0.02 + 0.015 * index, 0.04 - 0.003 * index)
        }));

        let output = fprime_correction_with_diagnostics(FprimeCorrectionInput {
            edge_reference_energy: 0.42,
            energy: energy.view(),
            main_energy_count: 5,
            extension_count: 4,
            fermi_index: 2,
            cross_section: cross_section.view(),
            background: background.view(),
            path_chi: path_chi.view(),
            real_correction: 0.015,
            imaginary_correction: 0.018,
        })?;
        let correction = output.correction;
        let diagnostics = output
            .danes_diagnostics
            .expect("DANES reference mesh should retain diagnostics");
        let expected = [
            Complex::new(2.245_047_339_117_258, -0.985_081_161_806_529_3),
            Complex::new(2.377_815_538_199_815_4, -1.029_333_077_279_869_2),
            Complex::new(2.481_370_833_664_761_7, -1.027_001_567_179_891_3),
            Complex::new(2.314_252_598_270_377, -0.057_621_832_449_236_236),
            Complex::new(2.101_718_212_094_632, -0.025_978_259_095_047_338),
        ];

        assert_eq!(correction.len(), expected.len());
        for (&actual, expected) in correction.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
        assert_real_array_close(
            &diagnostics.matsubara,
            &[
                -0.000_235_452_410_640_824_06,
                0.000_115_567_770_171_685_69,
                0.001_616_904_532_444_044_5,
                -0.011_537_662_500_167_504,
                -0.002_699_476_979_051_247_8,
            ],
        );
        assert_real_array_close(
            &diagnostics.sommerfeld,
            &[
                0.000_056_882_543_936_695_76,
                0.000_143_993_530_272_201_58,
                0.000_286_153_674_704_012_5,
                0.001_790_482_998_460_128_5,
                -0.000_075_188_949_617_717_85,
            ],
        );
        assert_real_array_close(
            &diagnostics.anomalous,
            &[
                0.344_547_746_311_940_8,
                0.492_513_353_731_171_5,
                0.662_395_251_093_048_8,
                0.784_514_916_760_123_5,
                0.508_819_694_718_912_2,
            ],
        );
        assert_real_array_close(
            &diagnostics.tail,
            &[
                2.194_956_632_561_878,
                2.337_458_531_758_847_4,
                2.456_883_740_362_153_7,
                2.351_404_370_716_003_3,
                2.124_167_324_209_779,
            ],
        );
        assert_real_array_close(
            &diagnostics.total,
            &[
                2.240_845_035_142_753_3,
                2.389_026_800_430_135_4,
                2.514_238_455_819_737,
                2.404_788_232_483_441,
                2.165_594_502_546_111,
            ],
        );
        assert_real_array_close(
            &diagnostics.difference,
            &[
                1.896_297_288_830_812_5,
                1.896_513_446_698_964,
                1.851_843_204_726_688_2,
                1.620_273_315_723_317_3,
                1.656_774_807_827_198_9,
            ],
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

        let energy = Array1::from_vec(vec![
            Complex::new(0.0, 0.1),
            Complex::new(0.1, 0.1),
            Complex::new(0.2, 0.0),
            Complex::new(0.3, 0.0),
        ]);
        let complex_values = Array1::from_elem(4, Complex::new(1.0, 0.0));
        let real_values = Array1::from_elem(4, 1.0);
        assert!(matches!(
            fprime_correction(FprimeCorrectionInput {
                edge_reference_energy: 0.2,
                energy: energy.view(),
                main_energy_count: 4,
                extension_count: 0,
                fermi_index: 0,
                cross_section: complex_values.view(),
                background: real_values.view(),
                path_chi: complex_values.view(),
                real_correction: 0.0,
                imaginary_correction: 0.0,
            }),
            Err(FprimeError::InvalidMeshPartition { .. })
        ));
    }
}
