use super::*;

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
