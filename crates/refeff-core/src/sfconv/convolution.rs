use ndarray::Array1;

use super::support::*;
use super::*;

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
