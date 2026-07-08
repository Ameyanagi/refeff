use ndarray::{ArrayView1, ArrayView2};

use super::*;

pub(crate) fn validate_convolution_input(
    input: SfconvConvolutionInput<'_>,
) -> Result<(), SfconvError> {
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

pub(crate) fn validate_grater_input(
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    singularities: &[Real],
) -> Result<(), SfconvError> {
    validate_finite_scalar("grater lower", lower)?;
    validate_finite_scalar("grater upper", upper)?;
    if upper <= lower {
        return Err(SfconvError::InvalidIntegrationInterval { lower, upper });
    }
    validate_positive_tolerance("abr", absolute_tolerance)?;
    validate_positive_tolerance("rlr", relative_tolerance)?;
    if singularities.len() > SFCONV_GRATER_MAX_SINGULARITIES {
        return Err(SfconvError::TooManySingularities {
            count: singularities.len(),
            max: SFCONV_GRATER_MAX_SINGULARITIES,
        });
    }

    let mut previous = lower;
    for (index, &singularity) in singularities.iter().enumerate() {
        if !singularity.is_finite()
            || singularity <= lower
            || singularity >= upper
            || singularity <= previous
        {
            return Err(SfconvError::InvalidSingularity {
                index,
                value: singularity,
            });
        }
        previous = singularity;
    }
    Ok(())
}

pub(crate) fn validate_positive_tolerance(
    field: &'static str,
    value: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveTolerance { field, value })
    }
}

pub(crate) fn eval_grater_integrand(
    integrand: &mut impl FnMut(Real) -> Result<Real, SfconvError>,
    argument: Real,
    row: usize,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("grater argument", argument)?;
    let value = integrand(argument)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteValue {
            field: "grater integrand",
            row,
            value,
        })
    }
}

pub(crate) fn validate_satellite_context(
    context: SfconvSatelliteContext,
) -> Result<(), SfconvError> {
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)?;
    validate_positive_scalar("photoelectron_energy", context.photoelectron_energy)?;
    validate_positive_tolerance("accuracy", context.accuracy)
}

pub(crate) fn validate_so2conv_material_input(
    input: SfconvSo2convMaterialInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("core_hole_width_ev", input.core_hole_width_ev)?;
    validate_positive_scalar("wigner_seitz_radius", input.wigner_seitz_radius)?;
    validate_finite_scalar("interstitial_potential_ev", input.interstitial_potential_ev)?;
    validate_finite_scalar("chemical_potential_ev", input.chemical_potential_ev)?;
    validate_finite_scalar(
        "fermi_wave_number_inv_angstrom",
        input.fermi_wave_number_inv_angstrom,
    )
}

pub(crate) fn validate_so2conv_material_parameters(
    parameters: SfconvSo2convMaterialParameters,
) -> Result<(), SfconvError> {
    validate_finite_scalar("core_hole_lifetime", parameters.core_hole_lifetime)?;
    validate_finite_scalar("interstitial_potential", parameters.interstitial_potential)?;
    validate_finite_scalar(
        "chemical_potential_offset",
        parameters.chemical_potential_offset,
    )?;
    validate_finite_scalar("fermi_wave_number", parameters.fermi_wave_number)?;
    validate_positive_scalar("fermi_momentum", parameters.fermi_momentum)?;
    validate_positive_scalar("fermi_energy", parameters.fermi_energy)?;
    validate_positive_scalar("electron_concentration", parameters.electron_concentration)?;
    validate_positive_scalar("plasma_frequency", parameters.plasma_frequency)?;
    validate_finite_scalar("dispersion_parameter", parameters.dispersion_parameter)?;
    validate_finite_scalar(
        "initial_photoelectron_energy",
        parameters.initial_photoelectron_energy,
    )?;
    validate_positive_scalar(
        "initial_photoelectron_momentum",
        parameters.initial_photoelectron_momentum,
    )?;
    validate_positive_tolerance("accuracy", parameters.accuracy)
}

pub(crate) fn validate_self_energy_context(
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_positive_scalar("fermi_energy", context.fermi_energy)?;
    validate_positive_scalar("fermi_momentum", context.fermi_momentum)?;
    validate_positive_scalar("plasma_frequency", context.plasma_frequency)?;
    validate_positive_scalar("pole_energy", context.pole_energy)?;
    validate_finite_scalar("quasiparticle_energy", context.quasiparticle_energy)?;
    validate_positive_scalar("photoelectron_momentum", context.photoelectron_momentum)?;
    validate_positive_tolerance("accuracy", context.accuracy)?;
    validate_finite_scalar("pole_broadening", context.pole_broadening)?;
    validate_finite_scalar("dispersion_parameter", context.dispersion_parameter)
}

pub(crate) fn validate_broadened_self_energy_integrand_input(
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("broadened self-energy momentum", input.momentum)?;
    if input.momentum < 0.0 {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.momentum,
            upper: 0.0,
        });
    }
    validate_finite_scalar("self-energy energy", input.energy)?;
    validate_self_energy_derivative_context(input.context)
}

pub(crate) fn validate_so2conv_self_energy_sample_input(
    input: SfconvSo2convSelfEnergySampleInput<'_>,
) -> Result<(), SfconvError> {
    validate_so2conv_material_parameters(input.material)?;
    validate_finite_scalar("self-energy energy", input.energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("photoelectron_momentum", input.photoelectron_momentum)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)
}

pub(crate) fn validate_so2conv_self_energy_grid_input(
    input: SfconvSo2convSelfEnergyGridInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 1)?;
    validate_finite_array("momentum", input.momentum)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_so2conv_self_energy_sample_input(SfconvSo2convSelfEnergySampleInput {
        material: input.material,
        energy: 0.0,
        quasiparticle_energy: input.material.fermi_energy,
        photoelectron_momentum: input.material.fermi_momentum,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
        include_below_fermi: input.include_below_fermi,
    })
}

pub(crate) fn validate_so2conv_specfunct_input(
    input: SfconvSo2convSpecfunctInput<'_>,
) -> Result<(), SfconvError> {
    validate_so2conv_material_parameters(input.material)?;
    validate_count_at_least("momentum_grid", input.momentum_grid.len(), 2)?;
    validate_finite_array("momentum_grid", input.momentum_grid)?;
    validate_strictly_increasing("momentum_grid", input.momentum_grid)?;
    validate_so2conv_self_energy_sample_input(SfconvSo2convSelfEnergySampleInput {
        material: input.material,
        energy: 0.0,
        quasiparticle_energy: input.material.fermi_energy,
        photoelectron_momentum: input.material.fermi_momentum,
        pole_count: input.pole_count,
        pole_energy: input.pole_energy,
        pole_weight: input.pole_weight,
        pole_broadening: input.pole_broadening,
        include_below_fermi: false,
    })?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
        validate_positive_scalar("pole_weight", input.pole_weight[index])?;
        validate_positive_scalar("pole_broadening", input.pole_broadening[index])?;
    }
    Ok(())
}

pub(crate) fn validate_self_energy_derivative_context(
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_self_energy_context(context)?;
    validate_positive_scalar("pole_broadening", context.pole_broadening)
}

pub(crate) fn validate_real_self_energy_integrand_inputs(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)
}

pub(crate) fn validate_real_self_energy_derivative_integrand_inputs(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)
}

pub(crate) fn validate_satellite_self_energy(
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<(), SfconvError> {
    validate_finite_scalar("on_shell_real", self_energy.on_shell_real)?;
    validate_finite_scalar("satellite width", self_energy.width)?;
    validate_finite_scalar("renormalization_real", self_energy.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", self_energy.renormalization_imag)?;
    validate_finite_scalar("off_shell_real", self_energy.off_shell_real)?;
    validate_finite_scalar("off_shell_imag", self_energy.off_shell_imag)
}

pub(crate) fn validate_momentum_spectral_interpolation_input(
    input: SfconvMomentumSpectralInterpolationInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_momentum", input.photoelectron_momentum)?;
    let rows = input.momentum_grid.len();
    validate_count_at_least("momentum_grid", rows, 2)?;
    validate_finite_array("momentum_grid", input.momentum_grid)?;
    validate_strictly_increasing("momentum_grid", input.momentum_grid)?;

    let columns = input.energy_grid.ncols();
    validate_count_at_least("spectral columns", columns, 1)?;
    validate_matrix_shape("energy_grid", input.energy_grid, rows, columns)?;
    validate_matrix_shape(
        "extrinsic_quasiparticle",
        input.extrinsic_quasiparticle,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "extrinsic_satellite",
        input.extrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "interference_quasiparticle",
        input.interference_quasiparticle,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "interference_satellite",
        input.interference_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "intrinsic_satellite",
        input.intrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape(
        "clipped_extrinsic_satellite",
        input.clipped_extrinsic_satellite,
        rows,
        columns,
    )?;
    validate_matrix_shape("weights", input.weights, rows, 8)?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "self_energy_real",
        input.self_energy_real.len(),
    )?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "energy_correction",
        input.energy_correction.len(),
    )?;
    validate_matching_lengths("momentum_grid", rows, "width", input.width.len())?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "renormalization_real",
        input.renormalization_real.len(),
    )?;
    validate_matching_lengths(
        "momentum_grid",
        rows,
        "renormalization_imag",
        input.renormalization_imag.len(),
    )?;

    validate_finite_matrix("energy_grid", input.energy_grid)?;
    validate_finite_matrix("extrinsic_quasiparticle", input.extrinsic_quasiparticle)?;
    validate_finite_matrix("extrinsic_satellite", input.extrinsic_satellite)?;
    validate_finite_matrix(
        "interference_quasiparticle",
        input.interference_quasiparticle,
    )?;
    validate_finite_matrix("interference_satellite", input.interference_satellite)?;
    validate_finite_matrix("intrinsic_satellite", input.intrinsic_satellite)?;
    validate_finite_matrix(
        "clipped_extrinsic_satellite",
        input.clipped_extrinsic_satellite,
    )?;
    validate_finite_matrix("weights", input.weights)?;
    validate_finite_array("self_energy_real", input.self_energy_real)?;
    validate_finite_array("energy_correction", input.energy_correction)?;
    validate_finite_array("width", input.width)?;
    validate_finite_array("renormalization_real", input.renormalization_real)?;
    validate_finite_array("renormalization_imag", input.renormalization_imag)
}

pub(crate) fn validate_photoelectron_momentum_input(
    input: SfconvPhotoelectronMomentumInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 2)?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "self_energy",
        input.self_energy.len(),
    )?;
    validate_finite_array("momentum", input.momentum)?;
    validate_finite_array("self_energy", input.self_energy)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    validate_positive_scalar("fermi_momentum", input.fermi_momentum)?;
    validate_finite_scalar("fermi_level", input.fermi_level)?;
    validate_finite_scalar("fermi_self_energy", input.fermi_self_energy)
}

pub(crate) fn validate_quasiparticle_peak_input(
    input: SfconvQuasiparticlePeakInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("center_energy", input.center_energy)?;
    validate_finite_scalar("lower_boundary", input.lower_boundary)?;
    validate_finite_scalar("upper_boundary", input.upper_boundary)?;
    if input.upper_boundary <= input.lower_boundary {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.lower_boundary,
            upper: input.upper_boundary,
        });
    }
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)
}

pub(crate) fn validate_exponential_reduction_input(
    input: SfconvExponentialReductionInput<'_>,
) -> Result<(), SfconvError> {
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

pub(crate) fn validate_quasiparticle_pole_input(
    input: SfconvQuasiparticlePoleInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_positive_scalar("width", input.width)?;
    validate_finite_scalar("renormalization_real", input.renormalization.real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization.imaginary)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization.magnitude)
}

pub(crate) fn validate_quasiparticle_table_input(
    input: SfconvQuasiparticleTableInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("energy", input.energy.len(), 1)?;
    validate_matching_lengths(
        "boundaries",
        input.boundaries.len(),
        "energy plus endpoints",
        input.energy.len() + 1,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("endpoint_width", input.endpoint_width)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)
}

pub(crate) fn validate_quasiparticle_interference_input(
    input: SfconvQuasiparticleInterferenceInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_finite_scalar("upper_energy", input.upper_energy)?;
    validate_positive_scalar("bare_photoelectron_energy", input.bare_photoelectron_energy)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_finite_scalar("dispersion_parameter", input.dispersion_parameter)?;
    validate_positive_tolerance("accuracy", input.accuracy)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

pub(crate) fn validate_satellite_pole_contributions_input(
    input: SfconvSatellitePoleContributionsInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("satellite_energy", input.energy)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_positive_scalar("plasma_frequency", input.plasma_frequency)?;
    validate_positive_scalar("bare_photoelectron_energy", input.bare_photoelectron_energy)?;
    validate_finite_scalar("dispersion_parameter", input.dispersion_parameter)?;
    validate_positive_tolerance("accuracy", input.accuracy)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    Ok(())
}

pub(crate) fn validate_extrinsic_satellite_input(
    input: SfconvExtrinsicSatelliteInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("satellite energy", input.energy)?;
    validate_finite_scalar("main_peak", input.main_peak)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)
}

pub(crate) fn validate_spectral_cell_input(
    input: SfconvSpectralCellInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )
}

pub(crate) fn validate_spectral_table_input(
    input: SfconvSpectralTableInput<'_>,
) -> Result<(), SfconvError> {
    let columns = input.energy.len();
    validate_count_at_least("energy", columns, 1)?;
    validate_count_exact("boundaries", input.boundaries.len(), columns + 1)?;
    validate_matching_lengths(
        "off_shell_real",
        input.off_shell_real.len(),
        "energy",
        columns,
    )?;
    validate_matching_lengths(
        "off_shell_imag",
        input.off_shell_imag.len(),
        "energy",
        columns,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_array("off_shell_real", input.off_shell_real)?;
    validate_finite_array("off_shell_imag", input.off_shell_imag)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("quasiparticle_energy", input.quasiparticle_energy)?;
    validate_positive_scalar("quasiparticle_width", input.quasiparticle_width)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("imaginary_derivative", input.imaginary_derivative)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_satellite_context(input.context)?;
    validate_satellite_self_energy(input.self_energy)?;
    validate_count_at_least("pole_count", input.pole_count, 1)?;
    validate_active_len("pole_energy", input.pole_count, input.pole_energy.len())?;
    validate_active_len("pole_weight", input.pole_count, input.pole_weight.len())?;
    validate_active_len(
        "pole_broadening",
        input.pole_count,
        input.pole_broadening.len(),
    )?;
    validate_active_finite_array("pole_energy", input.pole_energy, input.pole_count)?;
    validate_active_finite_array("pole_weight", input.pole_weight, input.pole_count)?;
    validate_active_finite_array("pole_broadening", input.pole_broadening, input.pole_count)?;
    for index in 0..input.pole_count {
        validate_positive_scalar("pole_energy", input.pole_energy[index])?;
    }
    validate_feff_column_index(
        "quasiparticle_lower_column",
        input.quasiparticle_lower_column_1based,
        columns,
    )?;
    validate_feff_column_index(
        "quasiparticle_upper_column",
        input.quasiparticle_upper_column_1based,
        columns,
    )
}

pub(crate) fn validate_satellite_table_input(
    input: SfconvSatelliteTableInput<'_>,
) -> Result<(), SfconvError> {
    let columns = input.extrinsic_satellite.len();
    validate_count_at_least("satellite columns", columns, 1)?;
    validate_matching_lengths(
        "main_peak",
        input.main_peak.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "quasiparticle_interference",
        input.quasiparticle_interference.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "interference_satellite",
        input.interference_satellite.len(),
        "satellite columns",
        columns,
    )?;
    validate_matching_lengths(
        "intrinsic_satellite",
        input.intrinsic_satellite.len(),
        "satellite columns",
        columns,
    )?;
    validate_count_exact("boundaries", input.boundaries.len(), columns + 1)?;
    validate_finite_array("main_peak", input.main_peak)?;
    validate_finite_array(
        "quasiparticle_interference",
        input.quasiparticle_interference,
    )?;
    validate_finite_array("extrinsic_satellite", input.extrinsic_satellite)?;
    validate_finite_array("interference_satellite", input.interference_satellite)?;
    validate_finite_array("intrinsic_satellite", input.intrinsic_satellite)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_feff_column_index(
        "quasiparticle_lower_column",
        input.quasiparticle_lower_column_1based,
        columns,
    )?;
    validate_feff_column_index(
        "quasiparticle_upper_column",
        input.quasiparticle_upper_column_1based,
        columns,
    )
}

pub(crate) fn validate_extrinsic_satellite_split_input(
    input: SfconvExtrinsicSatelliteSplitInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_count_at_least(
        "spectral_function columns",
        input.spectral_function.ncols(),
        3,
    )?;
    validate_matching_lengths(
        "energy",
        input.energy.len(),
        "spectral_function columns",
        input.spectral_function.ncols(),
    )?;
    validate_count_exact(
        "boundaries",
        input.boundaries.len(),
        input.spectral_function.ncols() + 1,
    )?;
    validate_finite_array("energy", input.energy)?;
    validate_strictly_increasing("energy", input.energy)?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_finite_scalar("photoelectron_energy", input.photoelectron_energy)?;
    validate_finite_scalar("beta_zero", input.beta_zero)?;
    validate_finite_array("extrinsic satellite", input.spectral_function.row(1))?;
    validate_finite_array("intrinsic satellite", input.spectral_function.row(4))
}

pub(crate) fn validate_satellite_correction_input(
    input: SfconvSatelliteCorrectionInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_exact("spectral_function rows", input.spectral_function.nrows(), 8)?;
    validate_count_at_least(
        "spectral_function columns",
        input.spectral_function.ncols(),
        1,
    )?;
    validate_count_exact(
        "boundaries",
        input.boundaries.len(),
        input.spectral_function.ncols() + 1,
    )?;
    validate_finite_array("boundaries", input.boundaries)?;
    validate_strictly_increasing("boundaries", input.boundaries)?;
    validate_positive_scalar("uniform_width", input.uniform_width)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_finite_mkspectf_satellite_rows(input.spectral_function)
}

pub(crate) fn validate_spectral_finalization_input(
    input: SfconvSpectralFinalizationInput<'_>,
) -> Result<(), SfconvError> {
    validate_extrinsic_satellite_split_input(SfconvExtrinsicSatelliteSplitInput {
        spectral_function: input.spectral_function,
        energy: input.energy,
        boundaries: input.boundaries,
        photoelectron_energy: input.photoelectron_energy,
        beta_zero: input.beta_zero,
    })?;
    validate_satellite_correction_input(SfconvSatelliteCorrectionInput {
        spectral_function: input.spectral_function,
        boundaries: input.boundaries,
        uniform_width: input.uniform_width,
        exponential_reduction: input.exponential_reduction,
    })?;
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)
}

pub(crate) fn validate_spectral_weights_input(
    input: SfconvSpectralWeightsInput<'_>,
) -> Result<(), SfconvError> {
    validate_finite_scalar("renormalization_real", input.renormalization_real)?;
    validate_finite_scalar("renormalization_imag", input.renormalization_imag)?;
    validate_positive_scalar("renormalization_magnitude", input.renormalization_magnitude)?;
    validate_finite_scalar("interference_amplitude", input.interference_amplitude)?;
    validate_finite_scalar("interference_reduction", input.interference_reduction)?;
    validate_positive_scalar("exponential_reduction", input.exponential_reduction)?;
    validate_count_exact("satellite_weights", input.satellite_weights.len(), 5)?;
    validate_finite_array("satellite_weights", input.satellite_weights)
}

pub(crate) fn validate_feff_path_interpolation_input(
    input: SfconvFeffPathInterpolationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("source_momentum", input.source_momentum.len(), 1)?;
    validate_count_at_least("path_momentum", input.path_momentum.len(), 2)?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "central_phase",
        input.central_phase.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "effective_amplitude",
        input.effective_amplitude.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "effective_phase",
        input.effective_phase.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "reduction_factor",
        input.reduction_factor.len(),
    )?;
    validate_matching_lengths(
        "path_momentum",
        input.path_momentum.len(),
        "mean_free_path",
        input.mean_free_path.len(),
    )?;
    validate_finite_array("source_momentum", input.source_momentum)?;
    validate_strictly_increasing("source_momentum", input.source_momentum)?;
    validate_finite_array("path_momentum", input.path_momentum)?;
    validate_strictly_increasing("path_momentum", input.path_momentum)?;
    validate_finite_array("central_phase", input.central_phase)?;
    validate_finite_array("effective_amplitude", input.effective_amplitude)?;
    validate_finite_array("effective_phase", input.effective_phase)?;
    validate_finite_array("reduction_factor", input.reduction_factor)?;
    validate_finite_array("mean_free_path", input.mean_free_path)
}

pub(crate) fn validate_feff_path_signal_input(
    input: SfconvFeffPathSignalInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("momentum", input.momentum.len(), 3)?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "central_phase",
        input.central_phase.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "effective_amplitude",
        input.effective_amplitude.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "effective_phase",
        input.effective_phase.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "reduction_factor",
        input.reduction_factor.len(),
    )?;
    validate_matching_lengths(
        "momentum",
        input.momentum.len(),
        "mean_free_path",
        input.mean_free_path.len(),
    )?;
    validate_finite_array("momentum", input.momentum)?;
    validate_strictly_increasing("momentum", input.momentum)?;
    validate_finite_array("central_phase", input.central_phase)?;
    validate_finite_array("effective_amplitude", input.effective_amplitude)?;
    validate_finite_array("effective_phase", input.effective_phase)?;
    validate_finite_array("reduction_factor", input.reduction_factor)?;
    validate_finite_array("mean_free_path", input.mean_free_path)?;
    validate_positive_scalar("degeneracy", input.degeneracy)?;
    validate_positive_scalar("half_path_length", input.half_path_length)
}

pub(crate) fn validate_exafs_convolution_input(
    input: SfconvExafsConvolutionInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar(
        "real_convolution_amplitude",
        input.real_convolution_amplitude,
    )?;
    validate_finite_scalar("real_convolution_phase", input.real_convolution_phase)?;
    validate_finite_scalar(
        "imaginary_convolution_amplitude",
        input.imaginary_convolution_amplitude,
    )?;
    validate_finite_scalar(
        "imaginary_convolution_phase",
        input.imaginary_convolution_phase,
    )?;
    validate_positive_scalar("original_magnitude", input.original_magnitude)?;
    validate_finite_scalar("original_phase", input.original_phase)?;
    validate_finite_scalar("phase_minus_2kr", input.phase_minus_2kr)?;
    validate_finite_scalar("previous_phase", input.previous_phase)
}

pub(crate) fn validate_xanes_convolution_input(
    input: SfconvXanesConvolutionInput,
) -> Result<(), SfconvError> {
    validate_finite_scalar("embedded_background", input.embedded_background)?;
    if input.asymmetric_phase {
        validate_finite_scalar("absorption_convolution", input.absorption_convolution)
    } else {
        validate_finite_scalar(
            "fine_structure_imaginary_amplitude",
            input.fine_structure_imaginary_amplitude,
        )?;
        validate_finite_scalar(
            "fine_structure_imaginary_phase",
            input.fine_structure_imaginary_phase,
        )?;
        validate_finite_scalar(
            "fine_structure_real_amplitude",
            input.fine_structure_real_amplitude,
        )?;
        validate_finite_scalar("fine_structure_real_phase", input.fine_structure_real_phase)
    }
}

pub(crate) fn validate_so2conv_exafs_energy_padding_input(
    input: SfconvSo2convExafsEnergyPaddingInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_active_len("energy", input.active_len, input.energy.len())?;
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("energy", input.energy, input.active_len)?;
    validate_active_strictly_increasing("energy", input.energy, input.active_len)
}

pub(crate) fn validate_so2conv_exafs_preparation_input(
    input: SfconvSo2convExafsPreparationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_active_len("momentum", input.active_len, input.momentum.len())?;
    validate_active_len("magnitude", input.active_len, input.magnitude.len())?;
    validate_active_len("phase", input.active_len, input.phase.len())?;
    if let Some(phase_minus_2kr) = input.phase_minus_2kr {
        validate_active_len("phase_minus_2kr", input.active_len, phase_minus_2kr.len())?;
        validate_active_finite_array("phase_minus_2kr", phase_minus_2kr, input.active_len)?;
    }
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("momentum", input.momentum, input.active_len)?;
    validate_active_finite_array("magnitude", input.magnitude, input.active_len)?;
    validate_active_finite_array("phase", input.phase, input.active_len)?;
    validate_finite_scalar("chemical_potential", input.chemical_potential)?;
    for row in 0..input.active_len {
        validate_positive_scalar("magnitude", input.magnitude[row])?;
    }
    Ok(())
}

pub(crate) fn validate_so2conv_xanes_preparation_input(
    input: SfconvSo2convXanesPreparationInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("active_len", input.active_len, 2)?;
    validate_count_at_least("output_len", input.output_len, 21)?;
    validate_active_len(
        "incident_energy",
        input.active_len,
        input.incident_energy.len(),
    )?;
    validate_active_len(
        "excitation_energy",
        input.active_len,
        input.excitation_energy.len(),
    )?;
    validate_active_len("absorption", input.active_len, input.absorption.len())?;
    validate_active_len(
        "embedded_background",
        input.active_len,
        input.embedded_background.len(),
    )?;
    validate_active_len("output_len", input.active_len, input.output_len)?;
    validate_active_finite_array("incident_energy", input.incident_energy, input.active_len)?;
    validate_active_finite_array(
        "excitation_energy",
        input.excitation_energy,
        input.active_len,
    )?;
    validate_active_finite_array("absorption", input.absorption, input.active_len)?;
    validate_active_finite_array(
        "embedded_background",
        input.embedded_background,
        input.active_len,
    )?;
    validate_active_strictly_increasing(
        "excitation_energy",
        input.excitation_energy,
        input.active_len,
    )
}

pub(crate) fn validate_path_average_input(
    input: SfconvPathAverageInput<'_>,
) -> Result<(), SfconvError> {
    validate_count_at_least("source_momentum", input.source_momentum.len(), 1)?;
    validate_matching_lengths(
        "source_momentum",
        input.source_momentum.len(),
        "amplitude_reduction",
        input.amplitude_reduction.len(),
    )?;
    validate_matching_lengths(
        "source_momentum",
        input.source_momentum.len(),
        "phase_shift",
        input.phase_shift.len(),
    )?;
    validate_finite_array("source_momentum", input.source_momentum)?;
    validate_strictly_increasing("source_momentum", input.source_momentum)?;
    validate_finite_array("amplitude_reduction", input.amplitude_reduction)?;
    validate_finite_array("phase_shift", input.phase_shift)?;
    validate_finite_scalar("previous_momentum", input.previous_momentum)?;
    validate_finite_scalar("center_momentum", input.center_momentum)?;
    validate_finite_scalar("next_momentum", input.next_momentum)?;
    if input.previous_momentum > input.center_momentum
        || input.center_momentum > input.next_momentum
    {
        return Err(SfconvError::InvalidIntegrationInterval {
            lower: input.previous_momentum,
            upper: input.next_momentum,
        });
    }
    validate_positive_scalar("momentum_step", input.momentum_step)
}

pub(crate) fn validate_nonzero_denominator(
    field: &'static str,
    value: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value == 0.0 {
        Err(SfconvError::ZeroDenominator { field })
    } else {
        Ok(())
    }
}

pub(crate) fn checked_hypot(
    field: &'static str,
    left: Real,
    right: Real,
) -> Result<Real, SfconvError> {
    validate_finite_scalar(field, left)?;
    validate_finite_scalar(field, right)?;
    let value = left.hypot(right);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

pub(crate) fn finite_result(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

pub(crate) fn validate_dispersion_inputs(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar("momentum", momentum)?;
    validate_positive_scalar("pole_energy", pole_energy)?;
    validate_finite_scalar("dispersion_parameter", dispersion_parameter)
}

pub(crate) fn pole_dispersion_value(
    momentum: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        pole_energy.powi(2) + dispersion_parameter * momentum.powi(2) + momentum.powi(4) / 4.0;
    checked_sqrt("pole_dispersion", radicand)
}

pub(crate) fn checked_sqrt(field: &'static str, value: Real) -> Result<Real, SfconvError> {
    if !value.is_finite() {
        return Err(SfconvError::NonFiniteScalar { field, value });
    }
    if value < 0.0 {
        return Err(SfconvError::NegativeRadicand { field, value });
    }
    Ok(value.sqrt())
}

pub(crate) fn threshold_factor(
    dispersion_parameter: Real,
    pole_energy: Real,
    root: Real,
) -> Result<Real, SfconvError> {
    let radicand =
        dispersion_parameter.powi(2) + (root.powi(2) / 2.0).powi(2) - pole_energy.powi(2);
    Ok(checked_sqrt("qthresh factor", radicand)? - dispersion_parameter)
}

pub(crate) fn roots_sorted_by_imag_descending(
    mut roots: [crate::Complex; 3],
) -> [crate::Complex; 3] {
    loop {
        let mut swaps = 0;
        for index in 0..2 {
            if roots[index].im < roots[index + 1].im {
                roots.swap(index, index + 1);
                swaps += 1;
            }
        }
        if swaps == 0 {
            return roots;
        }
    }
}

pub(crate) const fn feff_index(index_1based: usize) -> usize {
    index_1based - 1
}

pub(crate) fn select_threshold_root<F>(
    roots: [crate::Complex; 3],
    score: F,
) -> Result<crate::Complex, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let index = select_threshold_root_index(roots, score)?;
    Ok(roots[index])
}

pub(crate) fn select_threshold_root_index<F>(
    roots: [crate::Complex; 3],
    mut score: F,
) -> Result<usize, SfconvError>
where
    F: FnMut(Real) -> Result<Real, SfconvError>,
{
    let test0 = score(roots[0].re)?;
    let test1 = score(roots[1].re)?;
    let test2 = score(roots[2].re)?;
    if test0 < test1 && test0 < test2 {
        Ok(0)
    } else if test1 < test2 {
        Ok(1)
    } else {
        Ok(2)
    }
}

pub(crate) fn cutoff_weight(
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

pub(crate) fn interpolated_signal(
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

pub(crate) fn integration_width(
    energy: ArrayView1<'_, Real>,
    active_len: usize,
    row: usize,
) -> Real {
    if row == 0 {
        energy[1] - energy[0]
    } else if row + 1 == active_len {
        energy[active_len - 1] - energy[active_len - 2]
    } else {
        0.5 * (energy[row + 1] - energy[row - 1])
    }
}

pub(crate) fn combined_spectral_function(
    spectral_function: ArrayView2<'_, Real>,
    column: usize,
) -> Real {
    spectral_function[(1, column)] + spectral_function[(4, column)]
        - 2.0 * spectral_function[(3, column)]
}

pub(crate) fn validate_finite_matrix(
    field: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    let columns = values.ncols();
    for row in 0..values.nrows() {
        for column in 0..columns {
            validate_finite_value(field, row * columns + column, values[(row, column)])?;
        }
    }
    Ok(())
}

pub(crate) fn validate_finite_spectral_rows(
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

pub(crate) fn validate_finite_mkspectf_satellite_rows(
    spectral_function: ArrayView2<'_, Real>,
) -> Result<(), SfconvError> {
    let columns = spectral_function.ncols();
    for &row in &[1, 3, 4, 6, 7] {
        for column in 0..columns {
            validate_finite_value(
                "spectral_function",
                row * columns + column,
                spectral_function[(row, column)],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn validate_matching_lengths(
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

pub(crate) fn validate_matrix_shape(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
    rows: usize,
    columns: usize,
) -> Result<(), SfconvError> {
    validate_count_exact(field, matrix.nrows(), rows)?;
    validate_count_exact(field, matrix.ncols(), columns)
}

pub(crate) fn validate_count_exact(
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

pub(crate) fn validate_count_at_least(
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

pub(crate) fn validate_active_len(
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

pub(crate) fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteScalar { field, value })
    }
}

pub(crate) fn validate_positive_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), SfconvError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfconvError::NonPositiveScalar { field, value })
    }
}

pub(crate) fn validate_feff_column_index(
    field: &'static str,
    index_1based: usize,
    len: usize,
) -> Result<(), SfconvError> {
    if index_1based == 0 || index_1based > len {
        Err(SfconvError::IndexOutOfRange {
            field,
            index: index_1based,
            len,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), SfconvError> {
    for (row, value) in values.iter().copied().enumerate() {
        validate_finite_value(field, row, value)?;
    }
    Ok(())
}

pub(crate) fn validate_active_finite_array(
    field: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), SfconvError> {
    for row in 0..active_len {
        validate_finite_value(field, row, values[row])?;
    }
    Ok(())
}

pub(crate) fn validate_finite_value(
    field: &'static str,
    row: usize,
    value: Real,
) -> Result<(), SfconvError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfconvError::NonFiniteValue { field, row, value })
    }
}

pub(crate) fn validate_strictly_increasing(
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

pub(crate) fn validate_active_strictly_increasing(
    field: &'static str,
    values: ArrayView1<'_, Real>,
    active_len: usize,
) -> Result<(), SfconvError> {
    for row in 1..active_len {
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
