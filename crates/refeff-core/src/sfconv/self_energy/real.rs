use super::*;

/// Port of `SFCONV/senergies.f90` `renergies`.
///
/// Returns the real part of the photoelectron self energy for the active pole,
/// with FEFF `grater` diagnostics accumulated across the piecewise momentum
/// integrals.
pub fn sfconv_real_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let qmax = 100.0 * checked_sqrt("self-energy qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut total = SfconvAdaptiveIntegral {
        value: 0.0,
        estimated_error: 0.0,
        evaluations: 0,
        max_regions: 0,
    };

    if context.photoelectron_momentum > context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            context.photoelectron_momentum - context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum - context.fermi_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            context.fermi_momentum - context.photoelectron_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
        if context.include_below_fermi {
            add_real_self_energy_range(
                &mut total,
                0.0,
                context.fermi_momentum - context.photoelectron_momentum,
                absolute_tolerance,
                relative_tolerance,
                |momentum| sfconv_real_self_energy_integrand_lower(momentum, energy, context),
            )?;
        }
    } else {
        add_real_self_energy_range(
            &mut total,
            2.0 * context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_upper(momentum, energy, context),
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            2.0 * context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| sfconv_real_self_energy_integrand_middle(momentum, energy, context),
        )?;
    }

    let scale = -context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    total.value = finite_result("real self energy", total.value * scale)?;
    total.estimated_error *= scale.abs();
    Ok(total)
}

/// Port of `SFCONV/senergies.f90` `drenergies`.
///
/// Returns the energy derivative of the real part of the photoelectron self
/// energy for the active pole, with accumulated FEFF `grater` diagnostics.
pub fn sfconv_real_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let qmax = 100.0 * checked_sqrt("self-energy derivative qmax", context.pole_energy)?;
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut total = SfconvAdaptiveIntegral {
        value: 0.0,
        estimated_error: 0.0,
        evaluations: 0,
        max_regions: 0,
    };

    if context.photoelectron_momentum > context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            context.photoelectron_momentum - context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum - context.fermi_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum {
        add_real_self_energy_range(
            &mut total,
            context.photoelectron_momentum + context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            context.fermi_momentum - context.photoelectron_momentum,
            context.photoelectron_momentum + context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
        if context.include_below_fermi {
            add_real_self_energy_range(
                &mut total,
                0.0,
                context.fermi_momentum - context.photoelectron_momentum,
                absolute_tolerance,
                relative_tolerance,
                |momentum| {
                    sfconv_real_self_energy_derivative_integrand_lower(momentum, energy, context)
                },
            )?;
        }
    } else {
        add_real_self_energy_range(
            &mut total,
            2.0 * context.fermi_momentum,
            qmax,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_upper(momentum, energy, context)
            },
        )?;
        add_real_self_energy_range(
            &mut total,
            0.0,
            2.0 * context.fermi_momentum,
            absolute_tolerance,
            relative_tolerance,
            |momentum| {
                sfconv_real_self_energy_derivative_integrand_middle(momentum, energy, context)
            },
        )?;
    }

    let scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    total.value = finite_result("real self energy derivative", total.value * scale)?;
    total.estimated_error *= scale.abs();
    Ok(total)
}

/// Port of `SFCONV/senergies.f90` `rseint1`.
pub fn sfconv_real_self_energy_integrand_upper(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let numerator = ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    let denominator = ((context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    real_self_energy_log_integrand(
        "real self-energy upper integrand",
        momentum,
        context,
        dispersion,
        numerator,
        denominator,
    )
}

/// Port of `SFCONV/senergies.f90` `rseint2`.
pub fn sfconv_real_self_energy_integrand_middle(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let mut ratio = 1.0;
    if context.include_below_fermi {
        let below_numerator =
            (context.fermi_energy - shifted_energy - dispersion).powi(2) + regularization;
        let below_denominator = ((context.photoelectron_momentum - momentum).powi(2) / 2.0
            - shifted_energy
            - dispersion)
            .powi(2)
            + regularization;
        validate_nonzero_denominator(
            "middle real self-energy below denominator",
            below_denominator,
        )?;
        ratio *= below_numerator / below_denominator;
    }
    let numerator = ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy
        + dispersion)
        .powi(2)
        + regularization;
    let denominator = (context.fermi_energy - shifted_energy + dispersion).powi(2) + regularization;
    ratio *= numerator;
    ratio /= denominator;
    real_self_energy_log_integrand_with_ratio(
        "real self-energy middle integrand",
        momentum,
        context,
        dispersion,
        ratio,
    )
}

/// Port of `SFCONV/senergies.f90` `rseint3`.
pub fn sfconv_real_self_energy_integrand_lower(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let regularization = (context.accuracy * context.pole_energy).powi(2);
    let numerator =
        ((context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy - dispersion)
            .powi(2)
            + regularization;
    let denominator =
        ((context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion)
            .powi(2)
            + regularization;
    real_self_energy_log_integrand(
        "real self-energy lower integrand",
        momentum,
        context,
        dispersion,
        numerator,
        denominator,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint1`.
pub fn sfconv_real_self_energy_derivative_integrand_upper(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let upper =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let lower =
        (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let term = derivative_lorentz_term(upper, context.pole_broadening)?
        - derivative_lorentz_term(lower, context.pole_broadening)?;
    real_self_energy_derivative_integrand(
        "real self-energy derivative upper integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint2`.
pub fn sfconv_real_self_energy_derivative_integrand_middle(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let mut term = 0.0;
    if context.include_below_fermi {
        let below_fermi = context.fermi_energy - shifted_energy - dispersion;
        let below_photoelectron =
            (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion;
        term += derivative_lorentz_term(below_fermi, context.pole_broadening)?;
        term -= derivative_lorentz_term(below_photoelectron, context.pole_broadening)?;
    }
    let upper_photoelectron =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let upper_fermi = context.fermi_energy - shifted_energy + dispersion;
    term += derivative_lorentz_term(upper_photoelectron, context.pole_broadening)?;
    term -= derivative_lorentz_term(upper_fermi, context.pole_broadening)?;
    real_self_energy_derivative_integrand(
        "real self-energy derivative middle integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

/// Port of `SFCONV/senergies.f90` `drseint3`.
pub fn sfconv_real_self_energy_derivative_integrand_lower(
    momentum: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_real_self_energy_derivative_integrand_inputs(momentum, energy, context)?;
    let shifted_energy = energy + context.quasiparticle_energy;
    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    let upper =
        (context.photoelectron_momentum + momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let lower =
        (context.photoelectron_momentum - momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let lower_denominator = checked_sqrt(
        "real self-energy derivative lower denominator",
        lower.powi(2) + context.pole_broadening.powi(2),
    )?;
    validate_nonzero_denominator(
        "real self-energy derivative lower denominator",
        lower_denominator,
    )?;
    let term = derivative_lorentz_term(upper, context.pole_broadening)? - lower / lower_denominator;
    real_self_energy_derivative_integrand(
        "real self-energy derivative lower integrand",
        momentum,
        context,
        dispersion,
        term,
    )
}

fn add_real_self_energy_range(
    total: &mut SfconvAdaptiveIntegral,
    lower: Real,
    upper: Real,
    absolute_tolerance: Real,
    relative_tolerance: Real,
    integrand: impl FnMut(Real) -> Result<Real, SfconvError>,
) -> Result<(), SfconvError> {
    let current = sfconv_grater_integrate(
        integrand,
        lower,
        upper,
        absolute_tolerance,
        relative_tolerance,
        &[],
    )?;
    total.value += current.value;
    total.estimated_error += current.estimated_error;
    total.evaluations += current.evaluations;
    total.max_regions = total.max_regions.max(current.max_regions);
    Ok(())
}

fn real_self_energy_log_integrand(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    numerator: Real,
    denominator: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator(field, denominator)?;
    real_self_energy_log_integrand_with_ratio(
        field,
        momentum,
        context,
        dispersion,
        numerator / denominator,
    )
}

fn real_self_energy_log_integrand_with_ratio(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    ratio: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar(field, ratio)?;
    validate_nonzero_denominator(field, dispersion)?;
    let denominator = dispersion
        * checked_sqrt(
            field,
            momentum.powi(2) + context.pole_energy * context.accuracy,
        )?;
    validate_nonzero_denominator(field, denominator)?;
    finite_result(field, ratio.ln() / (2.0 * denominator))
}

fn derivative_lorentz_term(value: Real, broadening: Real) -> Result<Real, SfconvError> {
    let denominator = value.powi(2) + broadening.powi(2);
    validate_nonzero_denominator("self-energy derivative denominator", denominator)?;
    finite_result("self-energy derivative term", value / denominator)
}

fn real_self_energy_derivative_integrand(
    field: &'static str,
    momentum: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
    term: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator(field, dispersion)?;
    let denominator = dispersion
        * checked_sqrt(
            field,
            momentum.powi(2) + context.pole_energy * context.accuracy,
        )?;
    validate_nonzero_denominator(field, denominator)?;
    finite_result(field, term / denominator)
}
