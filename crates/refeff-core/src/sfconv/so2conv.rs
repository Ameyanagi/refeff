use ndarray::{Array1, Array2};

use super::support::*;
use super::*;

/// Port of the `SO2CONV` `feffNNNN.dat` interpolation loop.
///
/// FEFF first interpolates `caph2`, `xmfeff2`, `phfeff2`, `redfac2`, and
/// `xlam2` from a coarse path grid onto the uniform SO2CONV momentum grid. Rows
/// outside the coarse path range remain zero, while a source point exactly at
/// the final coarse momentum receives the final path row.
pub fn sfconv_interpolate_feff_path(
    input: SfconvFeffPathInterpolationInput<'_>,
) -> Result<SfconvFeffPathInterpolation, SfconvError> {
    validate_feff_path_interpolation_input(input)?;

    let mut output = SfconvFeffPathInterpolation {
        central_phase: Array1::<Real>::zeros(input.source_momentum.len()),
        effective_amplitude: Array1::<Real>::zeros(input.source_momentum.len()),
        effective_phase: Array1::<Real>::zeros(input.source_momentum.len()),
        reduction_factor: Array1::<Real>::zeros(input.source_momentum.len()),
        mean_free_path: Array1::<Real>::zeros(input.source_momentum.len()),
    };

    let last_path_row = input.path_momentum.len() - 1;
    for (source_row, &momentum) in input.source_momentum.iter().enumerate() {
        let mut matched_segment = None;
        for segment in 0..last_path_row {
            if momentum >= input.path_momentum[segment]
                && momentum < input.path_momentum[segment + 1]
            {
                matched_segment = Some(segment);
                break;
            }
        }

        if let Some(segment) = matched_segment {
            set_feff_path_interpolated_row(&mut output, source_row, input, segment)?;
        } else if momentum == input.path_momentum[last_path_row] {
            set_feff_path_exact_row(&mut output, source_row, input, last_path_row);
        }
    }

    validate_finite_array("interpolated central_phase", output.central_phase.view())?;
    validate_finite_array(
        "interpolated effective_amplitude",
        output.effective_amplitude.view(),
    )?;
    validate_finite_array(
        "interpolated effective_phase",
        output.effective_phase.view(),
    )?;
    validate_finite_array(
        "interpolated reduction_factor",
        output.reduction_factor.view(),
    )?;
    validate_finite_array("interpolated mean_free_path", output.mean_free_path.view())?;
    Ok(output)
}

/// Port of the `SO2CONV` raw EXAFS signal loop for `feffNNNN.dat` rows.
///
/// FEFF builds the unconvoluted complex path signal from interpolated path
/// columns before applying the spectral-function convolution. The first
/// magnitude row is linearly extrapolated from rows two and three, matching the
/// historical `xmag(1)` fixup that avoids the singular `k = 0` row.
pub fn sfconv_feff_path_signal(
    input: SfconvFeffPathSignalInput<'_>,
) -> Result<SfconvFeffPathSignal, SfconvError> {
    validate_feff_path_signal_input(input)?;

    let len = input.momentum.len();
    let mut output = SfconvFeffPathSignal {
        magnitude: Array1::<Real>::zeros(len),
        phase_minus_2kr: Array1::<Real>::zeros(len),
        phase: Array1::<Real>::zeros(len),
        real: Array1::<Real>::zeros(len),
        imaginary: Array1::<Real>::zeros(len),
    };

    for row in 0..len {
        output.phase_minus_2kr[row] = input.effective_phase[row] + input.central_phase[row];
        output.phase[row] =
            output.phase_minus_2kr[row] + 2.0 * input.momentum[row] * input.half_path_length;
    }

    for row in 1..len {
        output.magnitude[row] = feff_path_signal_magnitude(input, row)?;
        output.real[row] = output.magnitude[row] * output.phase[row].cos();
        output.imaginary[row] = output.magnitude[row] * output.phase[row].sin();
    }

    let extrapolation_denominator = input.momentum[2] - input.momentum[1];
    validate_nonzero_denominator(
        "feff path signal first-row extrapolation",
        extrapolation_denominator,
    )?;
    output.magnitude[0] = output.magnitude[1]
        + (input.momentum[0] - input.momentum[1]) * (output.magnitude[2] - output.magnitude[1])
            / extrapolation_denominator;
    output.real[0] = output.magnitude[0] * output.phase[0].cos();
    output.imaginary[0] = output.magnitude[0] * output.phase[0].sin();

    validate_finite_array("path signal magnitude", output.magnitude.view())?;
    validate_finite_array("path signal phase_minus_2kr", output.phase_minus_2kr.view())?;
    validate_finite_array("path signal phase", output.phase.view())?;
    validate_finite_array("path signal real", output.real.view())?;
    validate_finite_array("path signal imaginary", output.imaginary.view())?;
    Ok(output)
}

/// Port of the `SO2CONV` EXAFS post-convolution row calculation.
///
/// FEFF convolves the real and imaginary EXAFS channels separately, combines
/// their magnitudes/phases into a complex many-body signal, removes `2 pi`
/// phase jumps with the legacy `npi` state, and stores the amplitude/phase
/// correction arrays later averaged back onto `feffNNNN.dat` path grids.
pub fn sfconv_exafs_convolution(
    input: SfconvExafsConvolutionInput,
) -> Result<SfconvExafsConvolution, SfconvError> {
    validate_exafs_convolution_input(input)?;

    let real = input.real_convolution_amplitude * input.real_convolution_phase.cos()
        - input.imaginary_convolution_amplitude * input.imaginary_convolution_phase.sin();
    let imaginary = input.imaginary_convolution_amplitude * input.imaginary_convolution_phase.cos()
        + input.real_convolution_amplitude * input.real_convolution_phase.sin();
    let magnitude = checked_hypot("exafs convolution magnitude", real, imaginary)?;
    let raw_phase = finite_result("exafs convolution phase", imaginary.atan2(real))?;
    let phase_jump_count =
        so2conv_update_phase_jump_count(input.phase_jump_count, raw_phase, input.previous_phase)?;
    let output_phase = finite_result(
        "exafs output phase",
        raw_phase - std::f64::consts::PI * Real::from(phase_jump_count),
    )?;

    Ok(SfconvExafsConvolution {
        real: finite_result("exafs convolution real", real)?,
        imaginary: finite_result("exafs convolution imaginary", imaginary)?,
        magnitude,
        output_phase,
        output_phase_minus_original: finite_result(
            "exafs output phase correction",
            output_phase + input.phase_minus_2kr - input.original_phase,
        )?,
        amplitude_reduction: finite_result(
            "exafs amplitude reduction",
            magnitude / input.original_magnitude,
        )?,
        phase_shift: finite_result("exafs phase shift", output_phase - input.original_phase)?,
        previous_phase: raw_phase,
        phase_jump_count,
    })
}

/// Port of the `SO2CONV` XANES post-convolution row calculation.
///
/// FEFF either uses a real-valued asymmetric convolution result directly as
/// `xmu2`, or recombines real and imaginary fine-structure convolution channels
/// as `ximu2*cos(phmu) + rmu2*sin(phrmu) + xmu02`. FEFF10 writes the
/// unnormalized fine structure `xmu2 - xmu02`.
pub fn sfconv_xanes_convolution(
    input: SfconvXanesConvolutionInput,
) -> Result<SfconvXanesConvolution, SfconvError> {
    validate_xanes_convolution_input(input)?;

    let background = input.embedded_background;
    let absorption = if input.asymmetric_phase {
        input.absorption_convolution
    } else {
        input.fine_structure_imaginary_amplitude * input.fine_structure_imaginary_phase.cos()
            + input.fine_structure_real_amplitude * input.fine_structure_real_phase.sin()
            + background
    };

    let absorption = finite_result("xanes absorption", absorption)?;
    Ok(SfconvXanesConvolution {
        absorption,
        embedded_background: background,
        fine_structure: finite_result("xanes fine structure", absorption - background)?,
    })
}

/// Port of the `SO2CONV` EXAFS energy-grid padding loop.
///
/// FEFF extends `epts2` from the last two active rows through the full
/// convolution work-array length so endpoint interpolation in `sfconvsub` has a
/// flat continuation beyond the rows read from `chi.dat`, `chipNNNN.dat`, or
/// `feffNNNN.dat`.
pub fn sfconv_so2conv_pad_exafs_energy_grid(
    input: SfconvSo2convExafsEnergyPaddingInput<'_>,
) -> Result<RealVec, SfconvError> {
    validate_so2conv_exafs_energy_padding_input(input)?;

    let mut energy = Array1::<Real>::zeros(input.output_len);
    for row in 0..input.active_len {
        energy[row] = input.energy[row];
    }

    let step = energy[input.active_len - 1] - energy[input.active_len - 2];
    for row in input.active_len..input.output_len {
        energy[row] = finite_result("so2conv padded exafs energy", energy[row - 1] + step)?;
    }

    validate_finite_array("so2conv padded exafs energy", energy.view())?;
    Ok(energy)
}

/// Port of the `SO2CONV` EXAFS channel preparation loops.
///
/// FEFF converts the input `xk` grid to `epts2`, decomposes the magnitude and
/// phase columns into real and imaginary EXAFS channels, leaves padded signal
/// rows at zero, and then extends only the energy grid to the full convolution
/// work-array length.
pub fn sfconv_so2conv_prepare_exafs_signal(
    input: SfconvSo2convExafsPreparationInput<'_>,
) -> Result<SfconvSo2convExafsPreparation, SfconvError> {
    validate_so2conv_exafs_preparation_input(input)?;

    let mut signal_energy = Array1::<Real>::zeros(input.output_len);
    let mut real_signal = Array1::<Real>::zeros(input.output_len);
    let mut imaginary_signal = Array1::<Real>::zeros(input.output_len);
    let mut original_magnitude = Array1::<Real>::zeros(input.output_len);
    let mut original_phase = Array1::<Real>::zeros(input.output_len);
    let mut phase_minus_2kr = Array1::<Real>::zeros(input.output_len);

    for row in 0..input.active_len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        signal_energy[row] = finite_result("so2conv exafs energy", energy)?;
        original_magnitude[row] = input.magnitude[row];
        original_phase[row] = input.phase[row];
        phase_minus_2kr[row] = input.phase_minus_2kr.map_or(0.0, |values| values[row]);
        real_signal[row] = finite_result(
            "so2conv exafs real signal",
            input.magnitude[row] * input.phase[row].cos(),
        )?;
        imaginary_signal[row] = finite_result(
            "so2conv exafs imaginary signal",
            input.magnitude[row] * input.phase[row].sin(),
        )?;
    }

    signal_energy = sfconv_so2conv_pad_exafs_energy_grid(SfconvSo2convExafsEnergyPaddingInput {
        energy: signal_energy.view(),
        active_len: input.active_len,
        output_len: input.output_len,
    })?;

    validate_finite_array("so2conv exafs real signal", real_signal.view())?;
    validate_finite_array("so2conv exafs imaginary signal", imaginary_signal.view())?;
    validate_finite_array(
        "so2conv exafs original magnitude",
        original_magnitude.view(),
    )?;
    validate_finite_array("so2conv exafs original phase", original_phase.view())?;
    validate_finite_array("so2conv exafs phase minus 2kr", phase_minus_2kr.view())?;

    Ok(SfconvSo2convExafsPreparation {
        signal_energy,
        real_signal,
        imaginary_signal,
        original_magnitude,
        original_phase,
        phase_minus_2kr,
    })
}

/// Port of the `SO2CONV` XANES signal preparation loop.
///
/// FEFF pads `xmu.dat` by overwriting rows `j..npts2` with a flat
/// embedded-atom background, then computes `rmu` with `mkrmu` and `ximu` as the
/// residual `xmu - xmu0`. The one-based FEFF row `j` maps to
/// `active_len - 1`, so the last active row is intentionally replaced.
pub fn sfconv_so2conv_prepare_xanes_signal(
    input: SfconvSo2convXanesPreparationInput<'_>,
) -> Result<SfconvSo2convXanesPreparation, SfconvError> {
    validate_so2conv_xanes_preparation_input(input)?;

    let mut incident_energy = Array1::<Real>::zeros(input.output_len);
    let mut excitation_energy = Array1::<Real>::zeros(input.output_len);
    let mut absorption = Array1::<Real>::zeros(input.output_len);
    let mut embedded_background = Array1::<Real>::zeros(input.output_len);

    for row in 0..input.active_len {
        incident_energy[row] = input.incident_energy[row];
        excitation_energy[row] = input.excitation_energy[row];
        absorption[row] = input.absorption[row];
        embedded_background[row] = input.embedded_background[row];
    }

    let step = excitation_energy[input.active_len - 1] - excitation_energy[input.active_len - 2];
    let tail_background = embedded_background[input.active_len - 1];
    for row in (input.active_len - 1)..input.output_len {
        incident_energy[row] = finite_result(
            "so2conv padded xanes incident energy",
            incident_energy[row - 1] + step,
        )?;
        excitation_energy[row] = finite_result(
            "so2conv padded xanes excitation energy",
            excitation_energy[row - 1] + step,
        )?;
        embedded_background[row] = tail_background;
        absorption[row] = tail_background;
    }

    let real_fine_structure = sfconv_kramers_kronig_real_part(SfconvKramersKronigInput {
        imaginary: absorption.view(),
        reference_imaginary: embedded_background.view(),
        energy: excitation_energy.view(),
        active_len: input.output_len,
    })?;
    let mut imaginary_fine_structure = Array1::<Real>::zeros(input.output_len);
    for row in 0..input.output_len {
        imaginary_fine_structure[row] = finite_result(
            "so2conv xanes imaginary fine structure",
            absorption[row] - embedded_background[row],
        )?;
    }

    validate_finite_array(
        "so2conv padded xanes incident energy",
        incident_energy.view(),
    )?;
    validate_finite_array(
        "so2conv padded xanes excitation energy",
        excitation_energy.view(),
    )?;
    validate_finite_array("so2conv padded xanes absorption", absorption.view())?;
    validate_finite_array(
        "so2conv padded xanes embedded_background",
        embedded_background.view(),
    )?;
    validate_finite_array(
        "so2conv xanes imaginary fine structure",
        imaginary_fine_structure.view(),
    )?;

    Ok(SfconvSo2convXanesPreparation {
        incident_energy,
        excitation_energy,
        absorption,
        embedded_background,
        imaginary_fine_structure,
        real_fine_structure,
    })
}

/// Port of the `SO2CONV` triangular average for one FEFF path row.
///
/// FEFF computes `s02list` and `phlist` on a dense uniform momentum grid, then
/// averages nearby dense rows back onto the coarser `feffNNNN.dat` path grid
/// with a triangular finite-element weight. This helper returns the two
/// averaged values before the caller applies them to `redfac2` and `caph2`.
pub fn sfconv_path_average(
    input: SfconvPathAverageInput<'_>,
) -> Result<SfconvPathAverage, SfconvError> {
    validate_path_average_input(input)?;

    let mut amplitude_sum = 0.0;
    let mut phase_sum = 0.0;
    let mut normalization = 0.0;

    for ((&momentum, &amplitude), &phase) in input
        .source_momentum
        .iter()
        .zip(input.amplitude_reduction.iter())
        .zip(input.phase_shift.iter())
    {
        let weight = if momentum == input.center_momentum {
            1.0
        } else if momentum > input.previous_momentum
            && momentum <= input.center_momentum
            && input.previous_momentum != input.center_momentum
        {
            (momentum - input.previous_momentum) / (input.center_momentum - input.previous_momentum)
        } else if momentum > input.center_momentum
            && momentum < input.next_momentum
            && input.next_momentum != input.center_momentum
        {
            (input.next_momentum - momentum) / (input.next_momentum - input.center_momentum)
        } else {
            0.0
        };

        amplitude_sum += amplitude * weight * input.momentum_step;
        phase_sum += phase * weight * input.momentum_step;
        normalization += weight * input.momentum_step;
    }

    validate_nonzero_denominator("path average normalization", normalization)?;
    Ok(SfconvPathAverage {
        amplitude_reduction: finite_result(
            "path average amplitude",
            amplitude_sum / normalization,
        )?,
        phase_shift: finite_result("path average phase", phase_sum / normalization)?,
        normalization: finite_result("path average normalization", normalization)?,
    })
}

/// Port of the SO2CONV spectral-function interpolation over momentum.
///
/// FEFF caches spectral functions on the 66-row `pgrid` and, for each signal
/// row, interpolates those cached tables to the current photoelectron momentum
/// `pk(jj)`. Values at or above the final momentum copy the final cached row.
/// Values below the first momentum copy the first spectral rows and weights,
/// but preserve FEFF's historical endpoint quirk of taking `epts` from the
/// final cached momentum row.
pub fn sfconv_interpolate_momentum_spectral_function(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<SfconvMomentumSpectralInterpolation, SfconvError> {
    validate_momentum_spectral_interpolation_input(input)?;

    let columns = input.energy_grid.ncols();
    let mut output = SfconvMomentumSpectralInterpolation {
        energy: Array1::<Real>::zeros(columns),
        spectral_function: Array2::<Real>::zeros((8, columns)),
        weights: Array1::<Real>::zeros(8),
        self_energy_real: 0.0,
        energy_correction: 0.0,
        width: 0.0,
        renormalization_real: 0.0,
        renormalization_imag: 0.0,
    };

    let last = input.momentum_grid.len() - 1;
    if input.photoelectron_momentum >= input.momentum_grid[last] {
        set_momentum_spectral_exact_row(&mut output, input, last, last);
    } else if input.photoelectron_momentum < input.momentum_grid[0] {
        set_momentum_spectral_exact_row(&mut output, input, last, 0);
    } else {
        let segment = find_momentum_spectral_segment(input)?;
        set_momentum_spectral_interpolated_row(&mut output, input, segment)?;
    }

    validate_finite_array("momentum spectral energy", output.energy.view())?;
    validate_finite_array("momentum spectral weights", output.weights.view())?;
    validate_finite_matrix(
        "momentum spectral function",
        output.spectral_function.view(),
    )?;
    finite_result(
        "momentum spectral self_energy_real",
        output.self_energy_real,
    )?;
    finite_result(
        "momentum spectral energy_correction",
        output.energy_correction,
    )?;
    finite_result("momentum spectral width", output.width)?;
    finite_result(
        "momentum spectral renormalization_real",
        output.renormalization_real,
    )?;
    finite_result(
        "momentum spectral renormalization_imag",
        output.renormalization_imag,
    )?;
    Ok(output)
}

/// Port of the `SO2CONV` photoelectron momentum refinement.
///
/// FEFF first maps the input `xk` grid to `ekpg`, builds a zeroth-order
/// momentum estimate `xpkg`, estimates `zkk` from a finite difference of the
/// supplied self-energy samples, then applies the self-energy correction to
/// produce the momentum `pk` used for spectral-function interpolation.
pub fn sfconv_so2conv_photoelectron_momentum(
    input: SfconvPhotoelectronMomentumInput<'_>,
) -> Result<SfconvPhotoelectronMomentum, SfconvError> {
    validate_photoelectron_momentum_input(input)?;

    let len = input.momentum.len();
    let mut kinetic_energy = Array1::<Real>::zeros(len);
    let mut zero_order_momentum = Array1::<Real>::zeros(len);
    let mut renormalization = Array1::<Real>::zeros(len);
    let mut photoelectron_momentum = Array1::<Real>::zeros(len);

    for row in 0..len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        kinetic_energy[row] = finite_result("photoelectron kinetic energy", energy)?;
        if energy >= 0.0 {
            zero_order_momentum[row] = checked_sqrt(
                "photoelectron zero-order momentum",
                input.fermi_momentum.powi(2) + 2.0 * (energy - input.fermi_level),
            )?;
        }
    }

    for row in 0..len {
        if kinetic_energy[row] < 0.0 {
            continue;
        }

        let (lower_row, upper_row) = if row == 0 {
            (0, 1)
        } else if row + 1 == len {
            (row - 1, row)
        } else {
            (row - 1, row + 1)
        };
        let self_energy_delta = input.self_energy[upper_row] - input.self_energy[lower_row];
        let kinetic_delta = zero_order_momentum[upper_row].powi(2) / 2.0
            - zero_order_momentum[lower_row].powi(2) / 2.0;
        validate_nonzero_denominator("photoelectron momentum finite difference", kinetic_delta)?;

        let denominator = 1.0 + self_energy_delta / kinetic_delta;
        validate_nonzero_denominator("photoelectron momentum renormalization", denominator)?;
        renormalization[row] =
            finite_result("photoelectron momentum renormalization", 1.0 / denominator)?;

        photoelectron_momentum[row] = checked_sqrt(
            "photoelectron momentum",
            zero_order_momentum[row].powi(2)
                - 2.0 * renormalization[row] * (input.self_energy[row] - input.fermi_self_energy),
        )?;
    }

    validate_finite_array("photoelectron kinetic energy", kinetic_energy.view())?;
    validate_finite_array(
        "photoelectron zero-order momentum",
        zero_order_momentum.view(),
    )?;
    validate_finite_array("photoelectron renormalization", renormalization.view())?;
    validate_finite_array("photoelectron momentum", photoelectron_momentum.view())?;
    Ok(SfconvPhotoelectronMomentum {
        kinetic_energy,
        zero_order_momentum,
        renormalization,
        photoelectron_momentum,
    })
}

/// Compute one SO2CONV unbroadened weighted-pole self-energy sample.
///
/// This is the FEFF `brpole = .false.` branch: each active pole contributes
/// `plwt * renergies(energy)`, and the free-electron exchange term is added at
/// the requested photoelectron momentum.
pub fn sfconv_so2conv_unbroadened_self_energy_sample(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<Real, SfconvError> {
    validate_so2conv_self_energy_sample_input(input)?;

    let pole_sum = (1..=input.pole_count).try_fold(0.0, |accumulator, pole_index| {
        let pole = sfconv_select_pole(
            pole_index,
            input.pole_energy,
            input.pole_weight,
            input.pole_broadening,
        )?;
        let context = SfconvSelfEnergyContext {
            fermi_energy: input.material.fermi_energy,
            fermi_momentum: input.material.fermi_momentum,
            plasma_frequency: input.material.plasma_frequency,
            pole_energy: pole.energy,
            quasiparticle_energy: input.quasiparticle_energy,
            photoelectron_momentum: input.photoelectron_momentum,
            accuracy: input.material.accuracy,
            pole_broadening: pole.broadening,
            dispersion_parameter: input.material.dispersion_parameter,
            include_below_fermi: input.include_below_fermi,
        };
        let self_energy = sfconv_real_self_energy(input.energy, context)?.value;
        finite_result(
            "so2conv weighted self energy",
            accumulator + pole.weight * self_energy,
        )
    })?;
    let exchange =
        sfconv_free_electron_exchange(input.photoelectron_momentum, input.material.fermi_momentum)?;
    finite_result("so2conv unbroadened self energy", pole_sum + exchange)
}

/// Build SO2CONV unbroadened self-energy samples for momentum refinement.
///
/// FEFF first maps each input `xk` row to `ekpg`, estimates the zeroth-order
/// photoelectron momentum `xpkg`, evaluates the real self energy `seg` at that
/// momentum, and then calls the momentum-refinement formula. This helper
/// performs the `ekpg`/`xpkg`/`seg` part for the unbroadened `renergies`
/// branch and returns `sef0` for the existing
/// [`sfconv_so2conv_photoelectron_momentum`] helper.
pub fn sfconv_so2conv_unbroadened_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    build_so2conv_self_energy_grid(input, sfconv_so2conv_unbroadened_self_energy_sample)
}

/// Compute one SO2CONV broadened weighted-pole self-energy sample.
///
/// This is the FEFF default `brpole = .true.` branch: each active pole
/// contributes `plwt * brsigma(energy).real`, and the free-electron exchange
/// term is added at the requested photoelectron momentum.
pub fn sfconv_so2conv_broadened_self_energy_sample(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<Real, SfconvError> {
    validate_so2conv_self_energy_sample_input(input)?;

    let pole_sum = (1..=input.pole_count).try_fold(0.0, |accumulator, pole_index| {
        let pole = sfconv_select_pole(
            pole_index,
            input.pole_energy,
            input.pole_weight,
            input.pole_broadening,
        )?;
        let context = SfconvSelfEnergyContext {
            fermi_energy: input.material.fermi_energy,
            fermi_momentum: input.material.fermi_momentum,
            plasma_frequency: input.material.plasma_frequency,
            pole_energy: pole.energy,
            quasiparticle_energy: input.quasiparticle_energy,
            photoelectron_momentum: input.photoelectron_momentum,
            accuracy: input.material.accuracy,
            pole_broadening: pole.broadening,
            dispersion_parameter: input.material.dispersion_parameter,
            include_below_fermi: input.include_below_fermi,
        };
        let self_energy = sfconv_broadened_self_energy(input.energy, context)?.real;
        finite_result(
            "so2conv broadened weighted self energy",
            accumulator + pole.weight * self_energy,
        )
    })?;
    let exchange =
        sfconv_free_electron_exchange(input.photoelectron_momentum, input.material.fermi_momentum)?;
    finite_result("so2conv broadened self energy", pole_sum + exchange)
}

/// Build SO2CONV broadened self-energy samples for momentum refinement.
///
/// This mirrors the FEFF default `brpole = .true.` setup in `so2conv.f90`,
/// using [`sfconv_broadened_self_energy`] for each active pole before the
/// existing photoelectron-momentum refinement step.
pub fn sfconv_so2conv_broadened_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    build_so2conv_self_energy_grid(input, sfconv_so2conv_broadened_self_energy_sample)
}

fn build_so2conv_self_energy_grid(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
    sample: impl Fn(SfconvSo2convSelfEnergySampleInput<'_>) -> Result<Real, SfconvError>,
) -> Result<SfconvSo2convSelfEnergyGrid, SfconvError> {
    validate_so2conv_self_energy_grid_input(input)?;

    let fermi_self_energy = sample(SfconvSo2convSelfEnergySampleInput {
        material: input.material,
        energy: 0.0,
        quasiparticle_energy: input.material.fermi_energy,
        photoelectron_momentum: input.material.fermi_momentum,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
        include_below_fermi: input.include_below_fermi,
    })?;

    let len = input.momentum.len();
    let mut kinetic_energy = Array1::<Real>::zeros(len);
    let mut zero_order_momentum = Array1::<Real>::zeros(len);
    let mut self_energy = Array1::<Real>::zeros(len);

    for row in 0..len {
        let momentum = input.momentum[row];
        let energy = if momentum >= 0.0 {
            momentum.powi(2) / 2.0 + input.chemical_potential
        } else {
            -momentum.powi(2) / 2.0 + input.chemical_potential
        };
        kinetic_energy[row] = finite_result("so2conv self-energy kinetic energy", energy)?;
        if energy >= 0.0 {
            let row_momentum = checked_sqrt(
                "so2conv self-energy zero-order momentum",
                input.material.fermi_momentum.powi(2) + 2.0 * (energy - input.fermi_level),
            )?;
            zero_order_momentum[row] = row_momentum;
            self_energy[row] = sample(SfconvSo2convSelfEnergySampleInput {
                material: input.material,
                energy: 0.0,
                quasiparticle_energy: energy,
                photoelectron_momentum: row_momentum,
                pole_count: input.pole_count,
                pole_energy: input.pole_energy,
                pole_weight: input.pole_weight,
                pole_broadening: input.pole_broadening,
                include_below_fermi: input.include_below_fermi,
            })?;
        }
    }

    validate_finite_array("so2conv self-energy kinetic energy", kinetic_energy.view())?;
    validate_finite_array(
        "so2conv self-energy zero-order momentum",
        zero_order_momentum.view(),
    )?;
    validate_finite_array("so2conv self-energy", self_energy.view())?;
    Ok(SfconvSo2convSelfEnergyGrid {
        kinetic_energy,
        zero_order_momentum,
        self_energy,
        fermi_self_energy,
    })
}

fn set_feff_path_interpolated_row(
    output: &mut SfconvFeffPathInterpolation,
    source_row: usize,
    input: SfconvFeffPathInterpolationInput<'_>,
    lower_row: usize,
) -> Result<(), SfconvError> {
    let upper_row = lower_row + 1;
    let lower_momentum = input.path_momentum[lower_row];
    let upper_momentum = input.path_momentum[upper_row];
    let denominator = upper_momentum - lower_momentum;
    validate_nonzero_denominator("feff path interpolation interval", denominator)?;
    let fraction = (input.source_momentum[source_row] - lower_momentum) / denominator;

    output.central_phase[source_row] = linear_blend(
        input.central_phase[lower_row],
        input.central_phase[upper_row],
        fraction,
    );
    output.effective_amplitude[source_row] = linear_blend(
        input.effective_amplitude[lower_row],
        input.effective_amplitude[upper_row],
        fraction,
    );
    output.effective_phase[source_row] = linear_blend(
        input.effective_phase[lower_row],
        input.effective_phase[upper_row],
        fraction,
    );
    output.reduction_factor[source_row] = linear_blend(
        input.reduction_factor[lower_row],
        input.reduction_factor[upper_row],
        fraction,
    );
    output.mean_free_path[source_row] = linear_blend(
        input.mean_free_path[lower_row],
        input.mean_free_path[upper_row],
        fraction,
    );
    Ok(())
}

fn set_feff_path_exact_row(
    output: &mut SfconvFeffPathInterpolation,
    source_row: usize,
    input: SfconvFeffPathInterpolationInput<'_>,
    path_row: usize,
) {
    output.central_phase[source_row] = input.central_phase[path_row];
    output.effective_amplitude[source_row] = input.effective_amplitude[path_row];
    output.effective_phase[source_row] = input.effective_phase[path_row];
    output.reduction_factor[source_row] = input.reduction_factor[path_row];
    output.mean_free_path[source_row] = input.mean_free_path[path_row];
}

fn find_momentum_spectral_segment(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<usize, SfconvError> {
    for segment in 0..(input.momentum_grid.len() - 1) {
        if input.photoelectron_momentum >= input.momentum_grid[segment]
            && input.photoelectron_momentum < input.momentum_grid[segment + 1]
        {
            return Ok(segment);
        }
    }
    Err(SfconvError::MissingTrigger {
        field: "momentum spectral interval",
    })
}

fn set_momentum_spectral_interpolated_row(
    output: &mut SfconvMomentumSpectralInterpolation,
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    lower_row: usize,
) -> Result<(), SfconvError> {
    let upper_row = lower_row + 1;
    let denominator = input.momentum_grid[upper_row] - input.momentum_grid[lower_row];
    validate_nonzero_denominator("momentum spectral interval", denominator)?;
    let fraction = (input.photoelectron_momentum - input.momentum_grid[lower_row]) / denominator;

    for column in 0..input.energy_grid.ncols() {
        output.energy[column] = linear_blend(
            input.energy_grid[(lower_row, column)],
            input.energy_grid[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(0, column)] = linear_blend(
            input.extrinsic_quasiparticle[(lower_row, column)],
            input.extrinsic_quasiparticle[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(1, column)] = linear_blend(
            input.extrinsic_satellite[(lower_row, column)],
            input.extrinsic_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(2, column)] = linear_blend(
            input.interference_quasiparticle[(lower_row, column)],
            input.interference_quasiparticle[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(3, column)] = linear_blend(
            input.interference_satellite[(lower_row, column)],
            input.interference_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(4, column)] = linear_blend(
            input.intrinsic_satellite[(lower_row, column)],
            input.intrinsic_satellite[(upper_row, column)],
            fraction,
        );
        output.spectral_function[(5, column)] = linear_blend(
            combined_momentum_satellite(input, lower_row, column),
            combined_momentum_satellite(input, upper_row, column),
            fraction,
        );
        output.spectral_function[(6, column)] = linear_blend(
            clipped_momentum_satellite(input, lower_row, column),
            clipped_momentum_satellite(input, upper_row, column),
            fraction,
        );
        output.spectral_function[(7, column)] = linear_blend(
            input.clipped_extrinsic_satellite[(lower_row, column)],
            input.clipped_extrinsic_satellite[(upper_row, column)],
            fraction,
        );
    }

    for slot in 0..8 {
        output.weights[slot] = linear_blend(
            input.weights[(lower_row, slot)],
            input.weights[(upper_row, slot)],
            fraction,
        );
    }
    output.self_energy_real = linear_blend(
        input.self_energy_real[lower_row],
        input.self_energy_real[upper_row],
        fraction,
    );
    output.energy_correction = linear_blend(
        input.energy_correction[lower_row],
        input.energy_correction[upper_row],
        fraction,
    );
    output.width = linear_blend(input.width[lower_row], input.width[upper_row], fraction);
    output.renormalization_real = linear_blend(
        input.renormalization_real[lower_row],
        input.renormalization_real[upper_row],
        fraction,
    );
    output.renormalization_imag = linear_blend(
        input.renormalization_imag[lower_row],
        input.renormalization_imag[upper_row],
        fraction,
    );
    Ok(())
}

fn set_momentum_spectral_exact_row(
    output: &mut SfconvMomentumSpectralInterpolation,
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    energy_row: usize,
    data_row: usize,
) {
    for column in 0..input.energy_grid.ncols() {
        output.energy[column] = input.energy_grid[(energy_row, column)];
        output.spectral_function[(0, column)] = input.extrinsic_quasiparticle[(data_row, column)];
        output.spectral_function[(1, column)] = input.extrinsic_satellite[(data_row, column)];
        output.spectral_function[(2, column)] =
            input.interference_quasiparticle[(data_row, column)];
        output.spectral_function[(3, column)] = input.interference_satellite[(data_row, column)];
        output.spectral_function[(4, column)] = input.intrinsic_satellite[(data_row, column)];
        output.spectral_function[(5, column)] =
            combined_momentum_satellite(input, data_row, column);
        output.spectral_function[(6, column)] = clipped_momentum_satellite(input, data_row, column);
        output.spectral_function[(7, column)] =
            input.clipped_extrinsic_satellite[(data_row, column)];
    }
    for slot in 0..8 {
        output.weights[slot] = input.weights[(data_row, slot)];
    }
    output.self_energy_real = input.self_energy_real[data_row];
    output.energy_correction = input.energy_correction[data_row];
    output.width = input.width[data_row];
    output.renormalization_real = input.renormalization_real[data_row];
    output.renormalization_imag = input.renormalization_imag[data_row];
}

fn combined_momentum_satellite(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    row: usize,
    column: usize,
) -> Real {
    input.extrinsic_satellite[(row, column)] + input.intrinsic_satellite[(row, column)]
        - 2.0 * input.interference_satellite[(row, column)]
}

fn clipped_momentum_satellite(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
    row: usize,
    column: usize,
) -> Real {
    input.extrinsic_satellite[(row, column)] - input.clipped_extrinsic_satellite[(row, column)]
}

fn linear_blend(lower: Real, upper: Real, fraction: Real) -> Real {
    lower + (upper - lower) * fraction
}

fn feff_path_signal_magnitude(
    input: SfconvFeffPathSignalInput<'_>,
    row: usize,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("path signal momentum", input.momentum[row])?;
    let path_factor =
        input.degeneracy * input.effective_amplitude[row] * input.reduction_factor[row];
    if path_factor == 0.0 {
        return Ok(0.0);
    }

    validate_positive_scalar("mean_free_path", input.mean_free_path[row])?;
    let attenuation = (-2.0 * input.half_path_length / input.mean_free_path[row]).exp();
    let denominator = input.momentum[row] * input.half_path_length.powi(2);
    validate_nonzero_denominator("feff path signal magnitude", denominator)?;
    finite_result(
        "feff path signal magnitude",
        path_factor * attenuation / denominator,
    )
}

fn so2conv_update_phase_jump_count(
    phase_jump_count: i32,
    phase: Real,
    previous_phase: Real,
) -> Result<i32, SfconvError> {
    let delta = if phase - previous_phase > 5.0 {
        2
    } else if phase - previous_phase < -5.0 {
        -2
    } else {
        0
    };
    phase_jump_count
        .checked_add(delta)
        .ok_or(SfconvError::PhaseJumpOverflow {
            value: phase_jump_count,
            delta,
        })
}
