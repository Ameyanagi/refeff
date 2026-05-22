use super::*;

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
