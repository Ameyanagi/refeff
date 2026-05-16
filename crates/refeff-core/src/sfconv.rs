//! FEFF SFCONV numerical helpers.
//!
//! These kernels support spectral-function convolution. The full SFCONV driver
//! also depends on spectrum file orchestration, so this module keeps the
//! reusable numerical transforms independent and directly testable.

use ndarray::{Array1, ArrayView1, ArrayView2};
use thiserror::Error;

use crate::{Real, RealVec};

/// Inputs for FEFF `SFCONV/mkrmu.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvKramersKronigInput<'a> {
    /// Imaginary part of the spectrum-dependent function, FEFF `xmu`.
    pub imaginary: ArrayView1<'a, Real>,
    /// Reference imaginary part to subtract before the transform, FEFF `xmu0`.
    pub reference_imaginary: ArrayView1<'a, Real>,
    /// Energy grid, FEFF `wpts`.
    pub energy: ArrayView1<'a, Real>,
    /// Number of active rows, FEFF `npts`.
    pub active_len: usize,
}

/// Inputs for FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvConvolutionInput<'a> {
    /// Photoelectron energy neglecting collective excitations, FEFF `ekp`.
    pub photoelectron_energy: Real,
    /// Chemical potential / edge position, FEFF `mu`.
    pub chemical_potential: Real,
    /// Core-hole lifetime width, FEFF `gammach`.
    pub core_hole_lifetime: Real,
    /// Signal energy grid, FEFF `wpts2`.
    pub signal_energy: ArrayView1<'a, Real>,
    /// Signal values on `signal_energy`, FEFF `xchi`.
    pub signal: ArrayView1<'a, Real>,
    /// Spectral-function energy grid, FEFF `wpts1`.
    pub spectral_energy: ArrayView1<'a, Real>,
    /// Spectral function values, FEFF `spectf`.
    pub spectral_function: ArrayView1<'a, Real>,
    /// FEFF eight-slot spectral weights array.
    pub weights: ArrayView1<'a, Real>,
    /// Include quasiparticle phase as an asymmetric `1 / omega` term.
    pub asymmetric_phase: bool,
    /// Apply FEFF's available-energy cutoff.
    pub cutoff: bool,
    /// Plasma frequency scale used by the asymmetric phase branch, FEFF `omp`.
    pub plasma_frequency: Real,
}

/// Inputs for FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, Copy)]
pub struct SfconvSpectralInterpolationInput<'a> {
    /// Minimal spectral-function energy grid, FEFF `epts`.
    pub energy: ArrayView1<'a, Real>,
    /// Eight-row spectral-function table, FEFF `spectf(row, point)`.
    pub spectral_function: ArrayView2<'a, Real>,
    /// Number of rows in the uniform output grid, FEFF `npts`.
    pub output_len: usize,
}

/// Magnitude and phase produced by FEFF `SFCONV/sfconvsub.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfconvConvolution {
    /// Magnitude of the convoluted signal, FEFF `cchi`.
    pub amplitude: Real,
    /// Phase of the convoluted signal, FEFF `phase`.
    pub phase: Real,
}

/// Uniform spectral function produced by FEFF `SFCONV/interpsf.f90`.
#[derive(Debug, Clone, PartialEq)]
pub struct SfconvSpectralInterpolation {
    /// Uniform energy grid, FEFF `wpts`.
    pub energy: RealVec,
    /// Interpolated spectral function, FEFF `cspec`.
    pub spectral_function: RealVec,
}

/// Error returned by SFCONV helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SfconvError {
    /// FEFF `mkrmu` smooths rows 20 and 21, so shorter inputs are unsupported.
    #[error("SFCONV {name} count {actual} is below minimum {minimum}")]
    CountTooSmall {
        name: &'static str,
        actual: usize,
        minimum: usize,
    },
    /// Active rows must fit in each input array.
    #[error("SFCONV active row count {active_len} exceeds {field} length {len}")]
    ActiveCountOutOfRange {
        field: &'static str,
        active_len: usize,
        len: usize,
    },
    /// Two related arrays must have the same length.
    #[error("SFCONV {left} length {left_len} does not match {right} length {right_len}")]
    LengthMismatch {
        left: &'static str,
        left_len: usize,
        right: &'static str,
        right_len: usize,
    },
    /// Fixed-size FEFF helper arrays must have the expected number of slots.
    #[error("SFCONV {field} count {actual} does not match expected count {expected}")]
    CountMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Scalar values must be finite.
    #[error("SFCONV {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Array values must be finite.
    #[error("SFCONV {field} row {row} must be finite, got {value}")]
    NonFiniteValue {
        field: &'static str,
        row: usize,
        value: Real,
    },
    /// The energy grid must be strictly increasing to avoid FEFF's pole division.
    #[error("SFCONV {field} row {row} must increase, got {current} after {previous}")]
    NonIncreasingEnergy {
        field: &'static str,
        row: usize,
        previous: Real,
        current: Real,
    },
    /// The asymmetric branch divides by the real quasiparticle weight.
    #[error("SFCONV asymmetric phase requires a nonzero real quasiparticle weight")]
    ZeroAsymmetricWeight,
    /// The asymmetric branch needs a nonzero plasma-frequency scale.
    #[error("SFCONV asymmetric phase requires a nonzero plasma frequency")]
    ZeroPlasmaFrequency,
    /// FEFF normalizes by the total spectral weight.
    #[error("SFCONV normalization weight must be finite and nonzero, got {value}")]
    InvalidNormalization { value: Real },
    /// The transformed value must be finite.
    #[error("SFCONV transformed row {row} must be finite, got {value}")]
    NonFiniteResult { row: usize, value: Real },
}

/// Port of `SFCONV/mkrmu.f90`: discrete Kramers-Kronig transform.
///
/// FEFF integrates `(xmu - xmu0) / (w_i - w_j)` with endpoint/centered energy
/// widths, divides by `pi`, then averages rows 20 and 21 to smooth the legacy
/// phase handoff. The returned array contains exactly `active_len` rows.
pub fn sfconv_kramers_kronig_real_part(
    input: SfconvKramersKronigInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_count_at_least("active_len", input.active_len, 21)?;
    validate_active_len("imaginary", input.active_len, input.imaginary.len())?;
    validate_active_len(
        "reference_imaginary",
        input.active_len,
        input.reference_imaginary.len(),
    )?;
    validate_active_len("energy", input.active_len, input.energy.len())?;

    for row in 0..input.active_len {
        validate_finite_value("imaginary", row, input.imaginary[row])?;
        validate_finite_value("reference_imaginary", row, input.reference_imaginary[row])?;
        validate_finite_value("energy", row, input.energy[row])?;
        if row > 0 && input.energy[row] <= input.energy[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field: "energy",
                row,
                previous: input.energy[row - 1],
                current: input.energy[row],
            });
        }
    }

    let mut real_part = Array1::<Real>::zeros(input.active_len);
    for target in 0..input.active_len {
        let mut sum = 0.0;
        for source in 0..input.active_len {
            if source == target {
                continue;
            }
            let width = integration_width(input.energy, input.active_len, source);
            let numerator = input.imaginary[source] - input.reference_imaginary[source];
            sum += width * numerator / (input.energy[source] - input.energy[target]);
        }
        let value = sum / std::f64::consts::PI;
        if !value.is_finite() {
            return Err(SfconvError::NonFiniteResult { row: target, value });
        }
        real_part[target] = value;
    }

    let smoothed = 0.5 * (real_part[19] + real_part[20]);
    real_part[19] = smoothed;
    real_part[20] = smoothed;

    Ok(real_part)
}

/// Port of `SFCONV/interpsf.f90`: interpolate spectral function to a uniform grid.
///
/// FEFF builds the scalar spectral function from rows 2, 5, and 4 of
/// `spectf` as `spectf(2,j) + spectf(5,j) - 2*spectf(4,j)`, then linearly
/// interpolates that combination from the minimal input grid to `output_len`
/// uniformly spaced points spanning the same energy range.
pub fn sfconv_interpolate_spectral_function(
    input: SfconvSpectralInterpolationInput<'_>,
) -> Result<SfconvSpectralInterpolation, SfconvError> {
    validate_count_at_least("output_len", input.output_len, 2)?;
    validate_count_at_least("energy", input.energy.len(), 2)?;
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_matching_lengths(
        "energy",
        input.energy.len(),
        "spectral_function columns",
        input.spectral_function.ncols(),
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_spectral_rows(input.spectral_function)?;

    let last_input = input.energy.len() - 1;
    let first_energy = input.energy[0];
    let last_energy = input.energy[last_input];
    let step = (last_energy - first_energy) / (input.output_len as Real - 1.0);
    let mut energy = Array1::<Real>::zeros(input.output_len);
    let mut spectral_function = Array1::<Real>::zeros(input.output_len);

    energy[0] = first_energy;
    spectral_function[0] = combined_spectral_function(input.spectral_function, 0);
    energy[input.output_len - 1] = last_energy;
    spectral_function[input.output_len - 1] =
        combined_spectral_function(input.spectral_function, last_input);

    let mut lower = 0usize;
    for output in 1..(input.output_len - 1) {
        let output_energy = first_energy + step * output as Real;
        energy[output] = output_energy;

        while lower + 1 < last_input && output_energy >= input.energy[lower + 1] {
            lower += 1;
        }

        let upper = lower + 1;
        if !(input.energy[lower]..input.energy[upper]).contains(&output_energy) {
            return Err(SfconvError::NonFiniteResult {
                row: output,
                value: output_energy,
            });
        }
        let low = combined_spectral_function(input.spectral_function, lower);
        let high = combined_spectral_function(input.spectral_function, upper);
        let fraction =
            (output_energy - input.energy[lower]) / (input.energy[upper] - input.energy[lower]);
        spectral_function[output] = low + (high - low) * fraction;
    }

    Ok(SfconvSpectralInterpolation {
        energy,
        spectral_function,
    })
}

/// Port of `SFCONV/sfconvsub.f90`: spectral-function convolution.
///
/// The kernel integrates a signal over the spectral function, optionally
/// applying FEFF's available-energy cutoff and asymmetric quasiparticle phase
/// branch. Diagnostic file emission from the Fortran routine is intentionally
/// kept out of this pure numerical helper.
pub fn sfconv_convolve(
    input: SfconvConvolutionInput<'_>,
) -> Result<SfconvConvolution, SfconvError> {
    validate_convolution_input(input)?;

    let weights = input.weights;
    let pi = std::f64::consts::PI;
    let mut real_convolution = 0.0;
    let mut imag_convolution = 0.0;
    let quasiparticle_magnitude = if input.asymmetric_phase {
        weights[0]
    } else {
        weights[0].hypot(weights[1])
    };
    let quasiparticle_phase = if weights[0] != 0.0 && !input.asymmetric_phase {
        (weights[1] / weights[0]).atan()
    } else {
        0.0
    };
    let quasiparticle_reduction = if !input.cutoff {
        1.0
    } else if input.photoelectron_energy - input.chemical_potential != 0.0 {
        input
            .core_hole_lifetime
            .atan2(input.chemical_potential - input.photoelectron_energy)
            / pi
    } else {
        0.5
    };
    let quasiparticle_weight = quasiparticle_reduction * (quasiparticle_magnitude + weights[2]);
    let mut normalization = quasiparticle_weight;

    let mut cutoff_spectral_function = Array1::<Real>::zeros(input.spectral_function.len());
    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let cutoff_weight = cutoff_weight(
            input.cutoff,
            available_energy,
            input.chemical_potential,
            input.core_hole_lifetime,
        );

        let mut value = if !input.cutoff {
            input.spectral_function[row]
        } else if excitation_energy >= 0.0 {
            input.spectral_function[row] * cutoff_weight
        } else {
            (input.spectral_function[row] * cutoff_weight).max(0.0)
        };
        if input.asymmetric_phase {
            let half_width = 0.5 * width;
            let smoothing = 3.0 * width;
            let log_ratio = (((excitation_energy + half_width).powi(2) + smoothing.powi(2))
                / ((excitation_energy - half_width).powi(2) + smoothing.powi(2)))
            .ln();
            value -= quasiparticle_reduction
                * (weights[1] / (pi * quasiparticle_magnitude * width))
                * log_ratio
                * (-(excitation_energy / (2.0 * input.plasma_frequency)).powi(2)).exp()
                / 2.0;
        }
        cutoff_spectral_function[row] = value;
        normalization += value * width;
    }
    if !normalization.is_finite() || normalization == 0.0 {
        return Err(SfconvError::InvalidNormalization {
            value: normalization,
        });
    }

    for row in 0..input.spectral_function.len() {
        let width = integration_width(input.spectral_energy, input.spectral_function.len(), row);
        let excitation_energy = input.spectral_energy[row];
        let available_energy = input.photoelectron_energy - excitation_energy;
        let signal = interpolated_signal(input, available_energy)?;
        if row > 0 && row + 1 < input.spectral_function.len() {
            let left_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row - 1]);
            let right_midpoint = 0.5 * (excitation_energy + input.spectral_energy[row + 1]);
            if left_midpoint < 0.0 && right_midpoint >= 0.0 {
                real_convolution += quasiparticle_weight * signal;
            }
        }
        real_convolution += cutoff_spectral_function[row] * width * signal;
    }

    let stored_real = real_convolution;
    real_convolution =
        stored_real * quasiparticle_phase.cos() - imag_convolution * quasiparticle_phase.sin();
    imag_convolution =
        imag_convolution * quasiparticle_phase.cos() + stored_real * quasiparticle_phase.sin();
    real_convolution /= normalization;
    imag_convolution /= normalization;

    let amplitude = real_convolution.hypot(imag_convolution);
    let phase = imag_convolution.atan2(real_convolution);
    if !amplitude.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 0,
            value: amplitude,
        });
    }
    if !phase.is_finite() {
        return Err(SfconvError::NonFiniteResult {
            row: 1,
            value: phase,
        });
    }

    Ok(SfconvConvolution { amplitude, phase })
}

fn validate_convolution_input(input: SfconvConvolutionInput<'_>) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_finite_scalar("core_hole_lifetime", input.core_hole_lifetime)?;
    validate_finite_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_count_at_least("spectral_function", input.spectral_function.len(), 2)?;
    validate_count_at_least("signal", input.signal.len(), 2)?;
    validate_matching_lengths(
        "spectral_energy",
        input.spectral_energy.len(),
        "spectral_function",
        input.spectral_function.len(),
    )?;
    validate_matching_lengths(
        "signal_energy",
        input.signal_energy.len(),
        "signal",
        input.signal.len(),
    )?;
    validate_count_exact("weights", input.weights.len(), 8)?;
    validate_finite_array("spectral_energy", input.spectral_energy)?;
    validate_finite_array("spectral_function", input.spectral_function)?;
    validate_finite_array("signal_energy", input.signal_energy)?;
    validate_finite_array("signal", input.signal)?;
    validate_finite_array("weights", input.weights)?;
    validate_strictly_increasing("spectral_energy", input.spectral_energy)?;
    validate_strictly_increasing("signal_energy", input.signal_energy)?;
    if input.asymmetric_phase && input.weights[0] == 0.0 {
        return Err(SfconvError::ZeroAsymmetricWeight);
    }
    if input.asymmetric_phase && input.plasma_frequency == 0.0 {
        return Err(SfconvError::ZeroPlasmaFrequency);
    }
    Ok(())
}

fn cutoff_weight(
    cutoff: bool,
    available_energy: Real,
    chemical_potential: Real,
    gamma: Real,
) -> Real {
    if !cutoff {
        1.0
    } else if available_energy - chemical_potential != 0.0 {
        gamma.atan2(chemical_potential - available_energy) / std::f64::consts::PI
    } else {
        0.5
    }
}

fn interpolated_signal(
    input: SfconvConvolutionInput<'_>,
    available_energy: Real,
) -> Result<Real, SfconvError> {
    let last = input.signal.len() - 1;
    if available_energy > input.signal_energy[last] {
        return Ok(input.signal[last]);
    }
    if available_energy <= input.signal_energy[0] {
        let amplitude = input.signal[0];
        let delta = input.chemical_potential - input.signal_energy[0];
        let lambda = delta.powi(2)
            / (std::f64::consts::PI
                * amplitude.abs()
                * (delta.powi(2) + input.core_hole_lifetime.powi(2)));
        let signal = amplitude * (lambda * (available_energy - input.signal_energy[0])).exp();
        if signal.is_finite() {
            return Ok(signal);
        }
        return Err(SfconvError::NonFiniteResult {
            row: 2,
            value: signal,
        });
    }

    for row in 0..last {
        if available_energy > input.signal_energy[row]
            && available_energy <= input.signal_energy[row + 1]
        {
            let fraction = (available_energy - input.signal_energy[row])
                / (input.signal_energy[row + 1] - input.signal_energy[row]);
            return Ok(input.signal[row] + (input.signal[row + 1] - input.signal[row]) * fraction);
        }
    }

    Err(SfconvError::NonFiniteResult {
        row: 3,
        value: available_energy,
    })
}

fn integration_width(energy: ArrayView1<'_, Real>, active_len: usize, row: usize) -> Real {
    if row == 0 {
        energy[1] - energy[0]
    } else if row + 1 == active_len {
        energy[active_len - 1] - energy[active_len - 2]
    } else {
        0.5 * (energy[row + 1] - energy[row - 1])
    }
}

fn combined_spectral_function(spectral_function: ArrayView2<'_, Real>, column: usize) -> Real {
    spectral_function[(1, column)] + spectral_function[(4, column)]
        - 2.0 * spectral_function[(3, column)]
}

fn validate_finite_spectral_rows(
    spectral_function: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    for &row in &[1, 3, 4] {
        for column in 0..spectral_function.ncols() {
            validate_finite_value(
                "spectral_function",
                column,
                spectral_function[(row, column)],
            )?;
        }
    }
    Ok(())
}

fn validate_matching_lengths(
    left: &'static str,
    left_len: usize,
    right: &'static str,
    right_len: usize,
) -> Result<(), SfconvError> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(SfconvError::LengthMismatch {
            left,
            left_len,
            right,
            right_len,
        })
    }
}

fn validate_count_exact(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SfconvError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SfconvError::CountMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn validate_count_at_least(
    name: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), SfconvError> {
    if actual < minimum {
        Err(SfconvError::CountTooSmall {
            name,
            actual,
            minimum,
        })
    } else {
        Ok(())
    }
}

fn validate_active_len(
    field: &'static str,
    active_len: usize,
    len: usize,
) -> Result<(), SfconvError> {
    if active_len > len {
        Err(SfconvError::ActiveCountOutOfRange {
            field,
            active_len,
            len,
        })
    } else {
        Ok(())
    }
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

fn validate_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value)?;
    }
    Ok(())
}

fn validate_finite_value(field: &'static str, row: usize, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteValue { field, row, value })
    }
}

fn validate_strictly_increasing(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for row in 1..values.len() {
        if values[row] <= values[row - 1] {
            return Err(SfconvError::NonIncreasingEnergy {
                field,
                row,
                previous: values[row - 1],
                current: values[row],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, ShapeBuilder, array};

    use crate::Real;

    use super::{
        SfconvConvolutionInput, SfconvError, SfconvKramersKronigInput,
        SfconvSpectralInterpolationInput, sfconv_convolve, sfconv_interpolate_spectral_function,
        sfconv_kramers_kronig_real_part,
    };

    #[test]
    fn kramers_kronig_real_part_matches_feff_mkrmu_reference() -> Result<(), SfconvError> {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(25);

        let real_part = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
            imaginary: imaginary.view(),
            reference_imaginary: reference_imaginary.view(),
            energy: energy.view(),
            active_len: 25,
        })?;

        let expected = [
            0.653_321_127_749_770_8,
            0.750_003_058_275_569_8,
            0.770_088_761_144_957_1,
            0.744_953_602_096_770_5,
            0.685_875_097_053_667_7,
            0.599_956_814_602_449_9,
            0.492_993_575_338_788_3,
            0.370_329_818_936_448_6,
            0.237_144_234_118_930_07,
            0.098_519_596_973_469_21,
            -0.040_581_567_325_286_456,
            -0.175_385_521_001_154_32,
            -0.301_395_336_623_902_3,
            -0.414_483_981_972_534_94,
            -0.510_982_552_336_513_5,
            -0.587_755_578_520_523_2,
            -0.642_255_441_484_044_2,
            -0.672_546_008_587_787_2,
            -0.677_279_884_911_601_4,
            -0.631_242_351_812_862_9,
            -0.631_242_351_812_862_9,
            -0.530_174_264_181_443_8,
            -0.422_544_809_832_420_15,
            -0.273_383_187_221_121_7,
            -0.036_668_636_491_773_95,
        ];
        for (actual, expected) in real_part.iter().zip(expected) {
            assert_close(*actual, expected, 1.0e-13);
        }
        Ok(())
    }

    #[test]
    fn kramers_kronig_real_part_rejects_invalid_inputs() {
        let (imaginary, reference_imaginary, energy) = mkrmu_reference_inputs(21);

        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 20,
            }),
            Err(SfconvError::CountTooSmall {
                name: "active_len",
                ..
            })
        ));
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 22,
            }),
            Err(SfconvError::ActiveCountOutOfRange {
                field: "imaginary",
                ..
            })
        ));

        let mut bad_imaginary = imaginary.clone();
        bad_imaginary[3] = f64::NAN;
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: bad_imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonFiniteValue {
                field: "imaginary",
                row: 3,
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[5] = bad_energy[4];
        assert!(matches!(
            sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
                imaginary: imaginary.view(),
                reference_imaginary: reference_imaginary.view(),
                energy: bad_energy.view(),
                active_len: 21,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 5, .. })
        ));
    }

    #[test]
    fn interpolates_spectral_function_matches_feff_interpsf_reference() -> Result<(), SfconvError> {
        let (energy, spectral_function) = interpsf_reference_inputs();
        let interpolation =
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            })?;

        let expected_energy = [
            -2.0,
            -1.727_590_833_333_333_4,
            -1.455_181_666_666_666_8,
            -1.182_772_5,
            -0.910_363_333_333_333_4,
            -0.637_954_166_666_666_8,
            -0.365_545,
            -0.093_135_833_333_333_42,
            0.179_273_333_333_333_17,
            0.451_682_5,
            0.724_091_666_666_666_4,
            0.996_500_833_333_333_2,
            1.268_91,
        ];
        let expected_spectral_function = [
            -0.03,
            -0.035_578_048_005_086_65,
            -0.040_441_264_512_519_18,
            -0.044_809_714_285_714_24,
            -0.048_809_091_974_223_85,
            -0.052_519_432_577_500_3,
            -0.055_996_334_265_299_72,
            -0.059_278_128_963_028_02,
            -0.062_395_108_746_383_016,
            -0.065_369_121_964_238_19,
            -0.068_218_832_777_920_12,
            -0.070_958_429_921_906_93,
            -0.073_599_999_999_999_89,
        ];

        assert_real_slice_close(&interpolation.energy, &expected_energy, 1.0e-15);
        assert_real_slice_close(
            &interpolation.spectral_function,
            &expected_spectral_function,
            1.0e-14,
        );
        Ok(())
    }

    #[test]
    fn interpolates_spectral_function_rejects_invalid_inputs() {
        let (energy, spectral_function) = interpsf_reference_inputs();

        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 1,
            }),
            Err(SfconvError::CountTooSmall {
                name: "output_len",
                ..
            })
        ));

        let short_rows =
            Array2::from_shape_fn((7, spectral_function.ncols()).f(), |(row, column)| {
                spectral_function[(row, column)]
            });
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: energy.view(),
                spectral_function: short_rows.view(),
                output_len: 13,
            }),
            Err(SfconvError::CountMismatch {
                field: "spectral_function rows",
                actual: 7,
                expected: 8,
            })
        ));

        let short_energy = Array1::from_iter(energy.iter().copied().take(100));
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: short_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::LengthMismatch {
                left: "energy",
                right: "spectral_function columns",
                ..
            })
        ));

        let mut bad_energy = energy.clone();
        bad_energy[10] = bad_energy[9];
        assert!(matches!(
            sfconv_interpolate_spectral_function(SfconvSpectralInterpolationInput {
                energy: bad_energy.view(),
                spectral_function: spectral_function.view(),
                output_len: 13,
            }),
            Err(SfconvError::NonIncreasingEnergy { row: 10, .. })
        ));
    }

    #[test]
    fn convolve_matches_feff_sfconvsub_reference() -> Result<(), SfconvError> {
        let reference = sfconvsub_reference_inputs();

        let cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy: reference.signal_energy.view(),
            signal: reference.signal.view(),
            spectral_energy: reference.spectral_energy.view(),
            spectral_function: reference.spectral_function.view(),
            weights: reference.weights.view(),
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        })?;
        assert_close(cutoff_phase.amplitude, 0.404_768_834_000_475_8, 1.0e-14);
        assert_close(cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let no_cutoff_phase = sfconv_convolve(SfconvConvolutionInput {
            cutoff: false,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(no_cutoff_phase.amplitude, 0.405_036_447_280_840_4, 1.0e-14);
        assert_close(no_cutoff_phase.phase, 0.244_978_663_126_864_14, 1.0e-14);

        let asym_cutoff = sfconv_convolve(SfconvConvolutionInput {
            asymmetric_phase: true,
            ..sfconv_reference_input(
                reference.signal_energy.view(),
                reference.signal.view(),
                reference.spectral_energy.view(),
                reference.spectral_function.view(),
                reference.weights.view(),
            )
        })?;
        assert_close(asym_cutoff.amplitude, 0.394_308_834_584_619_57, 1.0e-14);
        assert_close(asym_cutoff.phase, 0.0, 1.0e-14);
        Ok(())
    }

    #[test]
    fn convolve_rejects_invalid_inputs() {
        let reference = sfconvsub_reference_inputs();

        let short_signal = array![0.62, 0.82, 0.74, 0.48, 0.22];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                signal: short_signal.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::LengthMismatch {
                left: "signal_energy",
                ..
            })
        ));

        let bad_spectral_energy = array![-0.18, -0.04, 0.0, 0.0, 0.31, 0.55, 0.82];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                spectral_energy: bad_spectral_energy.view(),
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::NonIncreasingEnergy {
                field: "spectral_energy",
                row: 3,
                ..
            })
        ));

        let zero_asym_weight = array![0.0, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            sfconv_convolve(SfconvConvolutionInput {
                weights: zero_asym_weight.view(),
                asymmetric_phase: true,
                ..sfconv_reference_input(
                    reference.signal_energy.view(),
                    reference.signal.view(),
                    reference.spectral_energy.view(),
                    reference.spectral_function.view(),
                    reference.weights.view(),
                )
            }),
            Err(SfconvError::ZeroAsymmetricWeight)
        ));
    }

    fn mkrmu_reference_inputs(count: usize) -> (Array1<Real>, Array1<Real>, Array1<Real>) {
        let indices = (1..=count).map(|index| index as Real);
        let imaginary = Array1::from_iter(
            indices
                .clone()
                .map(|index| (0.17 * index).sin() + 0.01 * index),
        );
        let reference_imaginary =
            Array1::from_iter(indices.clone().map(|index| 0.2 * (0.11 * index).cos()));
        let energy = Array1::from_iter((0..count).map(|index| {
            let index = index as Real;
            0.05 * index + 0.002 * index * index
        }));
        (imaginary, reference_imaginary, energy)
    }

    fn interpsf_reference_inputs() -> (Array1<Real>, Array2<Real>) {
        let count = 110usize;
        let energy = Array1::from_shape_fn(count, |index| {
            let i = index as Real;
            -2.0 + 0.018 * i + 0.000_11 * i * i
        });
        let spectral_function = Array2::from_shape_fn((8, count).f(), |(row, column)| {
            let fortran_row = row as Real + 1.0;
            let i = column as Real;
            0.03 * fortran_row + 0.002 * i + 0.000_4 * fortran_row * i + 0.000_01 * i * i
        });
        (energy, spectral_function)
    }

    struct SfconvSubReference {
        spectral_energy: Array1<Real>,
        spectral_function: Array1<Real>,
        signal_energy: Array1<Real>,
        signal: Array1<Real>,
        weights: Array1<Real>,
    }

    fn sfconvsub_reference_inputs() -> SfconvSubReference {
        SfconvSubReference {
            spectral_energy: array![-0.18, -0.04, 0.0, 0.12, 0.31, 0.55, 0.82],
            spectral_function: array![0.05, 0.18, 0.30, 0.23, 0.14, 0.07, 0.02],
            signal_energy: array![0.40, 0.72, 0.95, 1.22, 1.58, 1.95],
            signal: array![0.62, 0.82, 0.74, 0.48, 0.22, 0.12],
            weights: array![0.72, 0.18, 0.11, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    fn sfconv_reference_input<'a>(
        signal_energy: ndarray::ArrayView1<'a, Real>,
        signal: ndarray::ArrayView1<'a, Real>,
        spectral_energy: ndarray::ArrayView1<'a, Real>,
        spectral_function: ndarray::ArrayView1<'a, Real>,
        weights: ndarray::ArrayView1<'a, Real>,
    ) -> SfconvConvolutionInput<'a> {
        SfconvConvolutionInput {
            photoelectron_energy: 1.35,
            chemical_potential: 0.15,
            core_hole_lifetime: 0.08,
            signal_energy,
            signal,
            spectral_energy,
            spectral_function,
            weights,
            asymmetric_phase: false,
            cutoff: true,
            plasma_frequency: 0.55,
        }
    }

    fn assert_close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn assert_real_slice_close(actual: &Array1<Real>, expected: &[Real], tolerance: Real) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_close(actual, expected, tolerance);
        }
    }
}
