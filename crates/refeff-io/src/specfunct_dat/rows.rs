use ndarray::ArrayView1;
use refeff_core::{
    Real, SfconvConvolution, SfconvConvolutionInput, SfconvError, SfconvExafsConvolution,
    SfconvExafsConvolutionInput, SfconvMomentumSpectralInterpolation, SfconvSpectralInterpolation,
    SfconvSpectralInterpolationInput, SfconvXanesConvolution, SfconvXanesConvolutionInput,
    sfconv_convolve, sfconv_exafs_convolution, sfconv_interpolate_spectral_function,
    sfconv_xanes_convolution,
};

use crate::error::{IoError, Result};

use super::spectral::sfconv_specfunct_interpolate_momentum;
use super::types::{SfconvSpecfunctExafsRowsInput, SfconvSpecfunctXanesRowsInput};
use super::validation::{validate_specfunct_exafs_rows_input, validate_specfunct_xanes_rows_input};

pub fn sfconv_specfunct_xanes_convolution_rows(
    input: SfconvSpecfunctXanesRowsInput<'_>,
) -> Result<Vec<SfconvXanesConvolution>> {
    validate_specfunct_xanes_rows_input(input)?;
    let asymmetric_phase = input.cache.asymmetric_phase != 0;
    (0..input.active_len)
        .map(|row| sfconv_specfunct_xanes_convolution_row(input, row, asymmetric_phase))
        .collect()
}
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

pub(super) fn specfunct_exafs_error(source: SfconvError) -> IoError {
    IoError::SpecfunctDatExafsConvolution { source }
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

pub(super) fn specfunct_xanes_error(source: SfconvError) -> IoError {
    IoError::SpecfunctDatXanesConvolution { source }
}
