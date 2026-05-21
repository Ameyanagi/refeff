use ndarray::{Array1, ArrayView1};

use super::plasma::sfconv_q_limits_with_upper;
use super::support::*;
use super::*;

/// Port of `SFCONV/senergies.f90` `exchange`.
///
/// Computes the Hartree-Fock exchange potential for a free electron gas at
/// photoelectron momentum `momentum`.
pub fn sfconv_free_electron_exchange(
    momentum: Real,
    fermi_momentum: Real,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("fermi_momentum", fermi_momentum)?;

    let value = if momentum == fermi_momentum {
        -fermi_momentum / std::f64::consts::PI
    } else {
        let ratio = (momentum + fermi_momentum) / (momentum - fermi_momentum);
        validate_nonzero_denominator("exchange logarithm", ratio)?;
        -(fermi_momentum
            + ((fermi_momentum.powi(2) - momentum.powi(2)) / (2.0 * momentum)) * ratio.abs().ln())
            / std::f64::consts::PI
    };
    finite_result("free electron exchange", value)
}

/// Port of `SFCONV/senergies.f90` `beta`.
///
/// FEFF uses this extrinsic beta function as the analytic imaginary
/// self-energy contribution for the active pole.
pub fn sfconv_extrinsic_beta(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let pole_energy = context.pole_energy;
    let dispersion_parameter = context.dispersion_parameter;
    let fermi_limited_energy =
        (energy + context.quasiparticle_energy - context.fermi_energy).max(pole_energy);
    let qh =
        sfconv_inverse_pole_dispersion(fermi_limited_energy, pole_energy, dispersion_parameter)?;
    let q0 = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - energy - context.quasiparticle_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        energy + context.quasiparticle_energy,
        context.photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        qh,
    )?;

    let above_fermi = if limits.count == 3 {
        let q1 = checked_sqrt(
            "beta q1",
            limits.q1.powi(2) + context.accuracy * pole_energy,
        )?;
        let q2 = checked_sqrt(
            "beta q2",
            limits.q2.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq1 = sfconv_pole_dispersion(q1, pole_energy, dispersion_parameter)?;
        let wq2 = sfconv_pole_dispersion(q2, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q2, wq2, q1, wq1, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    let below_fermi = if limits.q3 < q0 && context.include_below_fermi {
        let q0 = checked_sqrt("beta q0", q0.powi(2) + context.accuracy * pole_energy)?;
        let q3 = checked_sqrt(
            "beta q3",
            limits.q3.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq0 = sfconv_pole_dispersion(q0, pole_energy, dispersion_parameter)?;
        let wq3 = sfconv_pole_dispersion(q3, pole_energy, dispersion_parameter)?;
        beta_prefactor(context)
            * beta_log_argument(q0, wq0, q3, wq3, pole_energy, dispersion_parameter)?.ln()
    } else {
        0.0
    };

    finite_result("extrinsic beta", above_fermi - below_fermi)
}

/// Port of `SFCONV/senergies.f90` `xienergies`.
pub fn sfconv_imaginary_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    finite_result(
        "imaginary self energy",
        -std::f64::consts::PI * sfconv_extrinsic_beta(energy, context)?,
    )
}

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

/// Port of `SFCONV/senergies.f90` `dienergies`.
pub fn sfconv_imaginary_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_context(context)?;

    let pole_energy = context.pole_energy;
    let dispersion_parameter = context.dispersion_parameter;
    let shifted_energy = energy + context.quasiparticle_energy;
    let qh = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let mut q0 = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(pole_energy),
        pole_energy,
        dispersion_parameter,
    )?;
    let upper_limit = 1.0e6 * context.fermi_momentum;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        pole_energy,
        dispersion_parameter,
        upper_limit,
    )?;
    let mut q1 = limits.q1;
    let mut q2 = limits.q2;
    let mut q3 = limits.q3;

    let (dqhdw, dq0dw) = self_energy_fermi_limit_derivatives(shifted_energy, qh, q0, context)?;
    let dq1dw = self_energy_upper_limit_derivative(&mut q1, qh, dqhdw, shifted_energy, context)?;
    let dq2dw = self_energy_upper_limit_derivative(&mut q2, qh, dqhdw, shifted_energy, context)?;
    let dq3dw = self_energy_lower_limit_derivative(&mut q3, q0, dq0dw, shifted_energy, context)?;

    let mut derivative = 0.0;
    let prefactor =
        context.plasma_frequency.powi(2) / (4.0 * context.photoelectron_momentum * pole_energy);

    if limits.count == 3 {
        q1 = checked_sqrt(
            "imaginary derivative q1",
            q1.powi(2) + context.accuracy * pole_energy,
        )?;
        q2 = checked_sqrt(
            "imaginary derivative q2",
            q2.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq1 = sfconv_pole_dispersion(q1, pole_energy, dispersion_parameter)?;
        let wq2 = sfconv_pole_dispersion(q2, pole_energy, dispersion_parameter)?;
        derivative += prefactor
            * dq1dw
            * self_energy_imaginary_derivative_factor(q1, wq1, pole_energy, dispersion_parameter)?;
        derivative -= prefactor
            * dq2dw
            * self_energy_imaginary_derivative_factor(q2, wq2, pole_energy, dispersion_parameter)?;
    }

    if q3 < q0 && context.include_below_fermi {
        q0 = checked_sqrt(
            "imaginary derivative q0",
            q0.powi(2) + context.accuracy * pole_energy,
        )?;
        q3 = checked_sqrt(
            "imaginary derivative q3",
            q3.powi(2) + context.accuracy * pole_energy,
        )?;
        let wq0 = sfconv_pole_dispersion(q0, pole_energy, dispersion_parameter)?;
        let wq3 = sfconv_pole_dispersion(q3, pole_energy, dispersion_parameter)?;
        derivative += prefactor
            * dq0dw
            * self_energy_imaginary_derivative_factor(q0, wq0, pole_energy, dispersion_parameter)?;
        derivative -= prefactor
            * dq3dw
            * self_energy_imaginary_derivative_factor(q3, wq3, pole_energy, dispersion_parameter)?;
    }

    finite_result("imaginary self energy derivative", derivative)
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

/// Port of `SFCONV/mksat.f90` `xmkesat`.
///
/// This is the extrinsic satellite with the quasiparticle pole subtracted and
/// quasiparticle broadening removed.
pub fn sfconv_extrinsic_satellite_debroadened(
    energy: Real,
    context: SfconvSatelliteContext,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_context(context)?;
    validate_satellite_self_energy(self_energy)?;
    validate_nonzero_denominator("satellite energy", energy)?;

    let renormalization_magnitude = checked_hypot(
        "satellite renormalization",
        self_energy.renormalization_real,
        self_energy.renormalization_imag,
    )?;
    validate_nonzero_denominator("satellite renormalization", renormalization_magnitude)?;

    let width_difference = self_energy.width - self_energy.off_shell_imag;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator = energy_difference.powi(2) + width_difference.powi(2);
    validate_nonzero_denominator("extrinsic satellite", denominator)?;

    let total = -width_difference / denominator;
    let main = -self_energy.renormalization_imag
        / (energy * std::f64::consts::PI * renormalization_magnitude)
        * (-(energy / (2.0 * context.plasma_frequency)).powi(2)).exp();
    finite_result(
        "extrinsic satellite",
        total / (std::f64::consts::PI * renormalization_magnitude) - main,
    )
}

/// Port of `SFCONV/mksat.f90` `xmkgwext`.
///
/// This is the full-broadening extrinsic satellite including quasiparticle
/// contributions.
pub fn sfconv_extrinsic_satellite_broadened(
    energy: Real,
    self_energy: SfconvSatelliteSelfEnergy,
) -> Result<Real, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_satellite_self_energy(self_energy)?;
    let energy_difference = energy + self_energy.on_shell_real - self_energy.off_shell_real;
    let denominator =
        std::f64::consts::PI * (energy_difference.powi(2) + self_energy.off_shell_imag.powi(2));
    validate_nonzero_denominator("broadened extrinsic satellite", denominator)?;
    finite_result(
        "broadened extrinsic satellite",
        self_energy.off_shell_imag / denominator,
    )
}

/// Port of `SFCONV/mksat.f90` `xintxsat`.
pub fn sfconv_interference_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let tolerance = 0.2 * context.plasma_frequency;
    let energy_delta = context.photoelectron_energy - energy;
    let lorentzian =
        width / (std::f64::consts::PI * ((energy - dispersion).powi(2) + width.powi(2)));

    let factor = if energy_delta >= 0.0 {
        let wave_number = checked_sqrt("interference wave number", 2.0 * energy_delta)?;
        validate_nonzero_denominator("interference wave number", wave_number)?;
        let numerator = (dispersion - momentum.powi(2) / 2.0 + wave_number * momentum).powi(2)
            + tolerance.powi(2);
        let denominator = (dispersion - momentum.powi(2) / 2.0 - wave_number * momentum).powi(2)
            + tolerance.powi(2);
        validate_nonzero_denominator("interference logarithm", denominator)?;
        (numerator / denominator).ln() / 2.0 / wave_number
    } else {
        let wave_number = checked_sqrt("interference evanescent wave number", -2.0 * energy_delta)?;
        validate_nonzero_denominator("interference evanescent wave number", wave_number)?;
        let denominator = dispersion - momentum.powi(2) / 2.0;
        validate_nonzero_denominator("interference arctangent", denominator)?;
        (wave_number * momentum / denominator).atan() / wave_number
    };

    finite_result(
        "interference satellite integrand",
        momentum * coupling * lorentzian * factor / dispersion,
    )
}

/// Port of `SFCONV/mksat.f90` `xintisat`.
pub fn sfconv_intrinsic_satellite_integrand(
    momentum: Real,
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let lorentzian =
        width / (((energy - dispersion).powi(2) + width.powi(2)) * std::f64::consts::PI);
    finite_result(
        "intrinsic satellite integrand",
        momentum.powi(2) * coupling * lorentzian / dispersion.powi(2),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkxsat`.
pub fn sfconv_interference_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = checked_sqrt(
        "interference satellite q2",
        (2.0 * (energy - context.pole_energy)).max(width),
    )?;
    validate_nonzero_denominator("interference satellite q2", q2)?;
    let qwidth = 10.0 * width / q2;
    let qmin = 0.0_f64.max(q2 - qwidth);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(qmin, q2, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_interference_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, (2.0 * std::f64::consts::PI).powi(2))
}

/// Port of `SFCONV/mksat.f90` `xmkisat`.
pub fn sfconv_intrinsic_satellite(
    energy: Real,
    width: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_positive_scalar("satellite width", width)?;
    validate_satellite_context(context)?;
    let q2 = if energy - context.pole_energy > width {
        checked_sqrt(
            "intrinsic satellite q2",
            2.0 * (energy - context.pole_energy),
        )?
    } else {
        checked_sqrt("intrinsic satellite q2", 2.0 * width)?
    };
    validate_nonzero_denominator("intrinsic satellite q2", q2)?;
    let qwidth = 10.0 * q2.min(width / q2);
    let qmax = q2 + qwidth;
    let first = integrate_mksat_range(0.0, q2, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    let second = integrate_mksat_range(q2, qmax, context, |momentum, context| {
        sfconv_intrinsic_satellite_integrand(momentum, energy, width, context)
    })?;
    combine_satellite_integrals(first, second, 2.0 * std::f64::consts::PI.powi(2))
}

/// Port of `SFCONV/mksat.f90` `xintak`.
pub fn sfconv_interference_quasiparticle_integrand(
    momentum: Real,
    photoelectron_momentum: Real,
    context: SfconvSatelliteContext,
) -> Result<Real, SfconvError> {
    validate_positive_scalar("momentum", momentum)?;
    validate_positive_scalar("photoelectron_momentum", photoelectron_momentum)?;
    validate_satellite_context(context)?;

    let dispersion =
        sfconv_pole_dispersion(momentum, context.pole_energy, context.dispersion_parameter)?;
    validate_nonzero_denominator("pole dispersion", dispersion)?;
    let coupling = sfconv_coupling_potential_squared(
        momentum,
        context.plasma_frequency,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let epsilon = 0.1_f64;
    let numerator = (dispersion + momentum.powi(2) / 2.0 + photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    let denominator = (dispersion + momentum.powi(2) / 2.0 - photoelectron_momentum * momentum)
        .powi(2)
        + (context.pole_energy * epsilon).powi(2);
    validate_nonzero_denominator("quasiparticle logarithm", denominator)?;
    let log_factor = (numerator / denominator).ln() / 2.0;
    finite_result(
        "interference quasiparticle integrand",
        momentum * coupling * log_factor
            / (dispersion * photoelectron_momentum * 4.0 * std::f64::consts::PI.powi(2)),
    )
}

/// Port of `SFCONV/mksat.f90` `xmkak`.
pub fn sfconv_interference_quasiparticle(
    energy: Real,
    upper_energy: Real,
    context: SfconvSatelliteContext,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_finite_scalar("satellite energy", energy)?;
    validate_finite_scalar("satellite upper energy", upper_energy)?;
    validate_satellite_context(context)?;
    if energy <= 0.0 {
        return Ok(SfconvSatelliteIntegral {
            value: 0.0,
            estimated_error: 0.0,
            evaluations: 0,
            max_regions: 0,
        });
    }
    let absolute_tolerance =
        checked_sqrt("quasiparticle tolerance", context.plasma_frequency)? * context.accuracy;
    let upper_momentum = checked_sqrt("quasiparticle upper momentum", 2.0 * upper_energy)?;
    let photoelectron_momentum = checked_sqrt(
        "quasiparticle photoelectron momentum",
        2.0 * context.photoelectron_energy,
    )?;
    validate_nonzero_denominator(
        "quasiparticle photoelectron momentum",
        photoelectron_momentum,
    )?;
    let integral = sfconv_grater_integrate(
        |momentum| {
            sfconv_interference_quasiparticle_integrand(momentum, photoelectron_momentum, context)
        },
        absolute_tolerance,
        upper_momentum,
        absolute_tolerance,
        context.accuracy,
        &[],
    )?;
    Ok(SfconvSatelliteIntegral {
        value: integral.value,
        estimated_error: integral.estimated_error,
        evaluations: integral.evaluations,
        max_regions: integral.max_regions,
    })
}

/// Evaluate the FEFF `brsigma` broadened self-energy integrand family.
///
/// The returned values correspond to the Fortran functions `fqlogrN`,
/// `fqlogiN`, `fqatnrN`, and `fqatniN` for the selected branch `N`. This helper
/// does not perform the `grater` interval integration or final `brsigma`
/// scaling; it keeps the branch formulas directly testable before the full
/// broadened self-energy driver is assembled.
pub fn sfconv_broadened_self_energy_integrands(
    branch: SfconvBroadenedSelfEnergyBranch,
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<SfconvBroadenedSelfEnergyIntegrands, SfconvError> {
    validate_broadened_self_energy_integrand_input(input)?;

    let context = input.context;
    let shifted_energy = finite_result(
        "broadened self-energy shifted energy",
        input.energy + context.quasiparticle_energy,
    )?;
    let dispersion = sfconv_pole_dispersion(
        input.momentum,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let pole_denominator = dispersion.powi(2) + context.pole_broadening.powi(2);
    validate_nonzero_denominator("broadened self-energy pole denominator", pole_denominator)?;
    let log_ratio = broadened_self_energy_log_ratio(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    )?;
    let atan_delta = broadened_self_energy_atan_delta(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    );
    let log_norm = checked_sqrt(
        "broadened self-energy log normalization",
        input.momentum.powi(2) + context.pole_energy * context.accuracy,
    )?;
    let atan_norm = checked_sqrt(
        "broadened self-energy atan normalization",
        input.momentum.powi(2) + context.plasma_frequency * context.accuracy,
    )?;
    validate_nonzero_denominator("broadened self-energy log normalization", log_norm)?;
    validate_nonzero_denominator("broadened self-energy atan normalization", atan_norm)?;

    let log_value = log_ratio.ln();
    let pole_real = dispersion / pole_denominator;
    let pole_imag = context.pole_broadening / pole_denominator;
    Ok(SfconvBroadenedSelfEnergyIntegrands {
        log_real: finite_result(
            "broadened log real integrand",
            pole_real * log_value / log_norm,
        )?,
        log_imag: finite_result(
            "broadened log imag integrand",
            pole_imag * log_value / log_norm,
        )?,
        atan_real: finite_result(
            "broadened atan real integrand",
            pole_imag * atan_delta / atan_norm,
        )?,
        atan_imag: finite_result(
            "broadened atan imag integrand",
            pole_real * atan_delta / atan_norm,
        )?,
    })
}

/// Evaluate the FEFF `dbrsigma` broadened self-energy derivative integrands.
///
/// The returned values correspond to the Fortran functions `dqlogrN`,
/// `dqlogiN`, `dqatnrN`, and `dqatniN` for the selected branch `N`.
pub fn sfconv_broadened_self_energy_derivative_integrands(
    branch: SfconvBroadenedSelfEnergyBranch,
    input: SfconvBroadenedSelfEnergyIntegrandInput,
) -> Result<SfconvBroadenedSelfEnergyDerivativeIntegrands, SfconvError> {
    validate_broadened_self_energy_integrand_input(input)?;

    let context = input.context;
    let shifted_energy = finite_result(
        "broadened self-energy derivative shifted energy",
        input.energy + context.quasiparticle_energy,
    )?;
    let dispersion = sfconv_pole_dispersion(
        input.momentum,
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let pole_denominator = dispersion.powi(2) + context.pole_broadening.powi(2);
    validate_nonzero_denominator(
        "broadened self-energy derivative pole denominator",
        pole_denominator,
    )?;
    let (left, right) = broadened_self_energy_response_arguments(
        branch,
        input.momentum,
        shifted_energy,
        context,
        dispersion,
    );
    let left_denominator = finite_result(
        "broadened self-energy derivative left denominator",
        left.powi(2) + context.pole_broadening.powi(2),
    )?;
    let right_denominator = finite_result(
        "broadened self-energy derivative right denominator",
        right.powi(2) + context.pole_broadening.powi(2),
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative left denominator",
        left_denominator,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative right denominator",
        right_denominator,
    )?;
    let log_derivative = finite_result(
        "broadened self-energy log derivative",
        left / left_denominator - right / right_denominator,
    )?;
    let atan_derivative = finite_result(
        "broadened self-energy atan derivative",
        context.pole_broadening / left_denominator - context.pole_broadening / right_denominator,
    )?;
    let log_norm = checked_sqrt(
        "broadened self-energy derivative log normalization",
        input.momentum.powi(2) + context.pole_energy * context.accuracy,
    )?;
    let atan_norm = checked_sqrt(
        "broadened self-energy derivative atan normalization",
        input.momentum.powi(2) + context.plasma_frequency * context.accuracy,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative log normalization",
        log_norm,
    )?;
    validate_nonzero_denominator(
        "broadened self-energy derivative atan normalization",
        atan_norm,
    )?;

    let pole_real = dispersion / pole_denominator;
    let pole_imag = context.pole_broadening / pole_denominator;
    Ok(SfconvBroadenedSelfEnergyDerivativeIntegrands {
        log_real: finite_result(
            "broadened log real derivative integrand",
            pole_real * log_derivative / log_norm,
        )?,
        log_imag: finite_result(
            "broadened log imag derivative integrand",
            pole_imag * log_derivative / log_norm,
        )?,
        atan_real: finite_result(
            "broadened atan real derivative integrand",
            pole_imag * atan_derivative / atan_norm,
        )?,
        atan_imag: finite_result(
            "broadened atan imag derivative integrand",
            pole_real * atan_derivative / atan_norm,
        )?,
    })
}

/// Port of `SFCONV/senergies.f90` `brsigma`.
///
/// This integrates the broadened log and arctangent branch kernels over FEFF's
/// piecewise momentum intervals, applies the `omp**2/(pi*pk)` scaling, and
/// rotates the complex self energy by FEFF's Lorentzian pole factor
/// `1 - i * brd / ompl`.
pub fn sfconv_broadened_self_energy(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvBroadenedSelfEnergy, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let shifted_energy = finite_result(
        "broadened self-energy shifted energy",
        energy + context.quasiparticle_energy,
    )?;
    let qmax = 100.0 * checked_sqrt("broadened self-energy qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let high_limit = context.photoelectron_momentum + context.fermi_momentum;
    let low_limit = (context.photoelectron_momentum - context.fermi_momentum).abs();
    let high_singularity = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let low_singularity = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        context.pole_energy,
        context.dispersion_parameter,
        qmax,
    )?;
    let singularity_candidates = Array1::from_vec(vec![
        low_singularity,
        limits.q1,
        limits.q2,
        limits.q3,
        high_singularity,
    ]);
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut sums = BroadenedSelfEnergyAccumulator::default();
    let range_input = |branch, lower, upper| BroadenedSelfEnergyRangeInput {
        branch,
        lower,
        upper,
        energy,
        context,
        singularity_candidates: singularity_candidates.view(),
        absolute_tolerance,
        relative_tolerance,
    };

    integrate_broadened_self_energy_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            high_limit,
            qmax,
        ),
    )?;
    integrate_broadened_self_energy_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            low_limit,
            high_limit,
        ),
    )?;
    if context.include_below_fermi {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                low_limit,
                high_limit,
            ),
        )?;
    }

    if context.photoelectron_momentum > context.fermi_momentum {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                0.0,
                low_limit,
            ),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum && context.include_below_fermi
    {
        integrate_broadened_self_energy_range(
            &mut sums,
            range_input(SfconvBroadenedSelfEnergyBranch::HolePair, 0.0, low_limit),
        )?;
    }

    let log_scale = context.plasma_frequency.powi(2)
        / (4.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let atan_scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let unrotated_real = finite_result(
        "broadened self-energy real",
        sums.log_real * log_scale + sums.atan_real * atan_scale,
    )?;
    let unrotated_imaginary = finite_result(
        "broadened self-energy imaginary",
        sums.log_imag * log_scale - sums.atan_imag * atan_scale,
    )?;
    let unrotated_real_error =
        sums.log_real_error * log_scale.abs() + sums.atan_real_error * atan_scale.abs();
    let unrotated_imaginary_error =
        sums.log_imag_error * log_scale.abs() + sums.atan_imag_error * atan_scale.abs();
    let pole_rotation = context.pole_broadening / context.pole_energy;

    Ok(SfconvBroadenedSelfEnergy {
        real: finite_result(
            "broadened self-energy rotated real",
            unrotated_real + unrotated_imaginary * pole_rotation,
        )?,
        imaginary: finite_result(
            "broadened self-energy rotated imaginary",
            unrotated_imaginary - unrotated_real * pole_rotation,
        )?,
        real_estimated_error: unrotated_real_error
            + unrotated_imaginary_error * pole_rotation.abs(),
        imaginary_estimated_error: unrotated_imaginary_error
            + unrotated_real_error * pole_rotation.abs(),
        evaluations: sums.evaluations,
        max_regions: sums.max_regions,
    })
}

/// Port of `SFCONV/senergies.f90` `dbrsigma`.
///
/// This evaluates the energy derivative of [`sfconv_broadened_self_energy`]
/// using FEFF's derivative log and arctangent kernels over the same piecewise
/// momentum intervals.
pub fn sfconv_broadened_self_energy_derivative(
    energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<SfconvBroadenedSelfEnergyDerivative, SfconvError> {
    validate_finite_scalar("self-energy energy", energy)?;
    validate_self_energy_derivative_context(context)?;

    let shifted_energy = finite_result(
        "broadened self-energy derivative shifted energy",
        energy + context.quasiparticle_energy,
    )?;
    let qmax = 100.0 * checked_sqrt("broadened self-energy derivative qmax", context.pole_energy)?
        + context.photoelectron_momentum
        + context.fermi_momentum;
    let high_limit = context.photoelectron_momentum + context.fermi_momentum;
    let low_limit = (context.photoelectron_momentum - context.fermi_momentum).abs();
    let high_singularity = sfconv_inverse_pole_dispersion(
        (shifted_energy - context.fermi_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let low_singularity = sfconv_inverse_pole_dispersion(
        (context.fermi_energy - shifted_energy).max(context.pole_energy),
        context.pole_energy,
        context.dispersion_parameter,
    )?;
    let limits = sfconv_q_limits_with_upper(
        shifted_energy,
        context.photoelectron_momentum,
        context.pole_energy,
        context.dispersion_parameter,
        qmax,
    )?;
    let singularity_candidates = Array1::from_vec(vec![
        low_singularity,
        limits.q1,
        limits.q2,
        limits.q3,
        high_singularity,
    ]);
    let absolute_tolerance = 1.0e-10;
    let relative_tolerance = 1.0e-7;
    let mut sums = BroadenedSelfEnergyAccumulator::default();
    let range_input = |branch, lower, upper| BroadenedSelfEnergyRangeInput {
        branch,
        lower,
        upper,
        energy,
        context,
        singularity_candidates: singularity_candidates.view(),
        absolute_tolerance,
        relative_tolerance,
    };

    integrate_broadened_self_energy_derivative_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticlePair,
            high_limit,
            qmax,
        ),
    )?;
    integrate_broadened_self_energy_derivative_range(
        &mut sums,
        range_input(
            SfconvBroadenedSelfEnergyBranch::ParticleFermi,
            low_limit,
            high_limit,
        ),
    )?;
    if context.include_below_fermi {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::HoleFermi,
                low_limit,
                high_limit,
            ),
        )?;
    }

    if context.photoelectron_momentum > context.fermi_momentum {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(
                SfconvBroadenedSelfEnergyBranch::ParticlePair,
                0.0,
                low_limit,
            ),
        )?;
    } else if context.photoelectron_momentum < context.fermi_momentum && context.include_below_fermi
    {
        integrate_broadened_self_energy_derivative_range(
            &mut sums,
            range_input(SfconvBroadenedSelfEnergyBranch::HolePair, 0.0, low_limit),
        )?;
    }

    let scale = context.plasma_frequency.powi(2)
        / (2.0 * std::f64::consts::PI * context.photoelectron_momentum);
    let unrotated_real = finite_result(
        "broadened self-energy derivative real",
        (sums.log_real + sums.atan_real) * scale,
    )?;
    let unrotated_imaginary = finite_result(
        "broadened self-energy derivative imaginary",
        (sums.log_imag - sums.atan_imag) * scale,
    )?;
    let unrotated_real_error = (sums.log_real_error + sums.atan_real_error) * scale.abs();
    let unrotated_imaginary_error = (sums.log_imag_error + sums.atan_imag_error) * scale.abs();
    let pole_rotation = context.pole_broadening / context.pole_energy;

    Ok(SfconvBroadenedSelfEnergyDerivative {
        real: finite_result(
            "broadened self-energy derivative rotated real",
            unrotated_real + unrotated_imaginary * pole_rotation,
        )?,
        imaginary: finite_result(
            "broadened self-energy derivative rotated imaginary",
            unrotated_imaginary - unrotated_real * pole_rotation,
        )?,
        real_estimated_error: unrotated_real_error
            + unrotated_imaginary_error * pole_rotation.abs(),
        imaginary_estimated_error: unrotated_imaginary_error
            + unrotated_real_error * pole_rotation.abs(),
        evaluations: sums.evaluations,
        max_regions: sums.max_regions,
    })
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

#[derive(Debug, Clone, Copy)]
struct BroadenedSelfEnergyRangeInput<'a> {
    branch: SfconvBroadenedSelfEnergyBranch,
    lower: Real,
    upper: Real,
    energy: Real,
    context: SfconvSelfEnergyContext,
    singularity_candidates: ArrayView1<'a, Real>,
    absolute_tolerance: Real,
    relative_tolerance: Real,
}

#[derive(Debug, Clone, Copy, Default)]
struct BroadenedSelfEnergyAccumulator {
    log_real: Real,
    log_imag: Real,
    atan_real: Real,
    atan_imag: Real,
    log_real_error: Real,
    log_imag_error: Real,
    atan_real_error: Real,
    atan_imag_error: Real,
    evaluations: usize,
    max_regions: usize,
}

impl BroadenedSelfEnergyAccumulator {
    fn add(&mut self, range: BroadenedSelfEnergyRange) {
        self.log_real += range.log_real.value;
        self.log_imag += range.log_imag.value;
        self.atan_real += range.atan_real.value;
        self.atan_imag += range.atan_imag.value;
        self.log_real_error += range.log_real.estimated_error;
        self.log_imag_error += range.log_imag.estimated_error;
        self.atan_real_error += range.atan_real.estimated_error;
        self.atan_imag_error += range.atan_imag.estimated_error;
        self.evaluations += range.log_real.evaluations
            + range.log_imag.evaluations
            + range.atan_real.evaluations
            + range.atan_imag.evaluations;
        self.max_regions = self
            .max_regions
            .max(range.log_real.max_regions)
            .max(range.log_imag.max_regions)
            .max(range.atan_real.max_regions)
            .max(range.atan_imag.max_regions);
    }
}

#[derive(Debug, Clone, Copy)]
struct BroadenedSelfEnergyRange {
    log_real: SfconvAdaptiveIntegral,
    log_imag: SfconvAdaptiveIntegral,
    atan_real: SfconvAdaptiveIntegral,
    atan_imag: SfconvAdaptiveIntegral,
}

fn integrate_broadened_self_energy_range(
    total: &mut BroadenedSelfEnergyAccumulator,
    input: BroadenedSelfEnergyRangeInput<'_>,
) -> Result<(), SfconvError> {
    if input.lower == input.upper {
        return Ok(());
    }
    let singularities =
        sfconv_find_singularities(input.lower, input.upper, input.singularity_candidates)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
    let range = BroadenedSelfEnergyRange {
        log_real: integrate_broadened_self_energy_component(input, &singularities, |integrands| {
            integrands.log_real
        })?,
        log_imag: integrate_broadened_self_energy_component(input, &singularities, |integrands| {
            integrands.log_imag
        })?,
        atan_real: integrate_broadened_self_energy_component(
            input,
            &singularities,
            |integrands| integrands.atan_real,
        )?,
        atan_imag: integrate_broadened_self_energy_component(
            input,
            &singularities,
            |integrands| integrands.atan_imag,
        )?,
    };
    total.add(range);
    Ok(())
}

fn integrate_broadened_self_energy_component(
    input: BroadenedSelfEnergyRangeInput<'_>,
    singularities: &[Real],
    select: impl Fn(SfconvBroadenedSelfEnergyIntegrands) -> Real,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| {
            let integrands = sfconv_broadened_self_energy_integrands(
                input.branch,
                SfconvBroadenedSelfEnergyIntegrandInput {
                    momentum,
                    energy: input.energy,
                    context: input.context,
                },
            )?;
            finite_result("broadened self-energy component", select(integrands))
        },
        input.lower,
        input.upper,
        input.absolute_tolerance,
        input.relative_tolerance,
        singularities,
    )
}

fn integrate_broadened_self_energy_derivative_range(
    total: &mut BroadenedSelfEnergyAccumulator,
    input: BroadenedSelfEnergyRangeInput<'_>,
) -> Result<(), SfconvError> {
    if input.lower == input.upper {
        return Ok(());
    }
    let singularities =
        sfconv_find_singularities(input.lower, input.upper, input.singularity_candidates)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
    let range = BroadenedSelfEnergyRange {
        log_real: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.log_real,
        )?,
        log_imag: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.log_imag,
        )?,
        atan_real: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.atan_real,
        )?,
        atan_imag: integrate_broadened_self_energy_derivative_component(
            input,
            &singularities,
            |integrands| integrands.atan_imag,
        )?,
    };
    total.add(range);
    Ok(())
}

fn integrate_broadened_self_energy_derivative_component(
    input: BroadenedSelfEnergyRangeInput<'_>,
    singularities: &[Real],
    select: impl Fn(SfconvBroadenedSelfEnergyDerivativeIntegrands) -> Real,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| {
            let integrands = sfconv_broadened_self_energy_derivative_integrands(
                input.branch,
                SfconvBroadenedSelfEnergyIntegrandInput {
                    momentum,
                    energy: input.energy,
                    context: input.context,
                },
            )?;
            finite_result(
                "broadened self-energy derivative component",
                select(integrands),
            )
        },
        input.lower,
        input.upper,
        input.absolute_tolerance,
        input.relative_tolerance,
        singularities,
    )
}

fn beta_prefactor(context: SfconvSelfEnergyContext) -> Real {
    context.plasma_frequency.powi(2)
        / (4.0 * std::f64::consts::PI * context.photoelectron_momentum * context.pole_energy)
}

fn beta_log_argument(
    numerator_momentum: Real,
    numerator_dispersion: Real,
    denominator_momentum: Real,
    denominator_dispersion: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    let numerator_denominator = pole_energy
        + numerator_dispersion
        + dispersion_parameter * numerator_momentum.powi(2) / (2.0 * pole_energy);
    let denominator_denominator = pole_energy
        + denominator_dispersion
        + dispersion_parameter * denominator_momentum.powi(2) / (2.0 * pole_energy);
    validate_nonzero_denominator("beta numerator", numerator_denominator)?;
    validate_nonzero_denominator("beta denominator", denominator_momentum)?;
    validate_nonzero_denominator("beta denominator", denominator_denominator)?;
    let argument = numerator_momentum.powi(2) / numerator_denominator * denominator_denominator
        / denominator_momentum.powi(2);
    validate_positive_scalar("beta logarithm", argument)?;
    Ok(argument)
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

fn self_energy_fermi_limit_derivatives(
    shifted_energy: Real,
    qh: Real,
    q0: Real,
    context: SfconvSelfEnergyContext,
) -> Result<(Real, Real), SfconvError> {
    let upper_gap = shifted_energy - context.fermi_energy;
    let lower_gap = context.fermi_energy - shifted_energy;
    if upper_gap > context.pole_energy {
        let denominator = qh
            * checked_sqrt(
                "imaginary derivative high limit",
                context.dispersion_parameter.powi(2) + upper_gap.powi(2)
                    - context.pole_energy.powi(2),
            )?;
        validate_nonzero_denominator("imaginary derivative high limit", denominator)?;
        Ok((upper_gap / denominator, 0.0))
    } else if lower_gap > context.pole_energy {
        let denominator = q0
            * checked_sqrt(
                "imaginary derivative low limit",
                context.dispersion_parameter.powi(2) + lower_gap.powi(2)
                    - context.pole_energy.powi(2),
            )?;
        validate_nonzero_denominator("imaginary derivative low limit", denominator)?;
        Ok((0.0, -lower_gap / denominator))
    } else {
        Ok((0.0, 0.0))
    }
}

fn self_energy_upper_limit_derivative(
    momentum: &mut Real,
    fermi_limit: Real,
    fermi_limit_derivative: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    let dispersion =
        sfconv_pole_dispersion(*momentum, context.pole_energy, context.dispersion_parameter)?;
    let plus_test =
        (context.photoelectron_momentum + *momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    let minus_test =
        (context.photoelectron_momentum - *momentum).powi(2) / 2.0 - shifted_energy + dispersion;
    if *momentum >= fermi_limit {
        *momentum = fermi_limit;
        Ok(fermi_limit_derivative)
    } else if plus_test.abs() < minus_test.abs() {
        self_energy_q_limit_derivative(*momentum, dispersion, context, 1.0, 1.0)
    } else {
        self_energy_q_limit_derivative(*momentum, dispersion, context, -1.0, 1.0)
    }
}

fn self_energy_lower_limit_derivative(
    momentum: &mut Real,
    fermi_limit: Real,
    fermi_limit_derivative: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
) -> Result<Real, SfconvError> {
    let dispersion =
        sfconv_pole_dispersion(*momentum, context.pole_energy, context.dispersion_parameter)?;
    let plus_test =
        (context.photoelectron_momentum + *momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    let minus_test =
        (context.photoelectron_momentum - *momentum).powi(2) / 2.0 - shifted_energy - dispersion;
    if *momentum >= fermi_limit {
        *momentum = fermi_limit;
        Ok(fermi_limit_derivative)
    } else if plus_test.abs() < minus_test.abs() {
        self_energy_q_limit_derivative(*momentum, dispersion, context, 1.0, -1.0)
    } else {
        self_energy_q_limit_derivative(*momentum, dispersion, context, -1.0, -1.0)
    }
}

fn self_energy_q_limit_derivative(
    momentum: Real,
    dispersion: Real,
    context: SfconvSelfEnergyContext,
    momentum_sign: Real,
    dispersion_sign: Real,
) -> Result<Real, SfconvError> {
    let denominator = (momentum + momentum_sign * context.photoelectron_momentum) * dispersion
        + dispersion_sign * (context.dispersion_parameter * momentum + momentum.powi(3) / 2.0);
    validate_nonzero_denominator("imaginary derivative momentum limit", denominator)?;
    finite_result(
        "imaginary derivative momentum limit",
        dispersion / denominator,
    )
}

fn self_energy_imaginary_derivative_factor(
    momentum: Real,
    dispersion: Real,
    pole_energy: Real,
    dispersion_parameter: Real,
) -> Result<Real, SfconvError> {
    validate_nonzero_denominator("imaginary derivative momentum", momentum)?;
    validate_nonzero_denominator("imaginary derivative dispersion", dispersion)?;
    validate_nonzero_denominator("imaginary derivative pole", pole_energy)?;
    let denominator =
        pole_energy + dispersion + dispersion_parameter * momentum.powi(2) / (2.0 * pole_energy);
    validate_nonzero_denominator("imaginary derivative factor", denominator)?;
    let slope = dispersion_parameter * momentum * (1.0 / dispersion + 1.0 / pole_energy)
        + momentum.powi(3) / (2.0 * dispersion);
    finite_result(
        "imaginary derivative factor",
        2.0 / momentum - slope / denominator,
    )
}

fn broadened_self_energy_log_ratio(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> Result<Real, SfconvError> {
    let minus_energy = (context.photoelectron_momentum - momentum).powi(2) / 2.0;
    let plus_energy = (context.photoelectron_momentum + momentum).powi(2) / 2.0;
    let broadening = context.pole_broadening;
    let (numerator_arg, denominator_arg) = match branch {
        SfconvBroadenedSelfEnergyBranch::ParticlePair => (
            minus_energy - shifted_energy + dispersion,
            plus_energy - shifted_energy + dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::ParticleFermi => (
            context.fermi_energy - shifted_energy + dispersion,
            plus_energy - shifted_energy + dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::HoleFermi => (
            minus_energy - shifted_energy - dispersion,
            context.fermi_energy - shifted_energy - dispersion,
        ),
        SfconvBroadenedSelfEnergyBranch::HolePair => (
            minus_energy - shifted_energy - dispersion,
            plus_energy - shifted_energy - dispersion,
        ),
    };
    let numerator = finite_result(
        "broadened self-energy log numerator",
        numerator_arg.powi(2) + broadening.powi(2),
    )?;
    let denominator = finite_result(
        "broadened self-energy log denominator",
        denominator_arg.powi(2) + broadening.powi(2),
    )?;
    validate_nonzero_denominator("broadened self-energy log denominator", denominator)?;
    let ratio = numerator / denominator;
    validate_positive_scalar("broadened self-energy log ratio", ratio)?;
    Ok(ratio)
}

fn broadened_self_energy_atan_delta(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> Real {
    let broadening = context.pole_broadening;
    let (left, right) = broadened_self_energy_response_arguments(
        branch,
        momentum,
        shifted_energy,
        context,
        dispersion,
    );
    (left / broadening).atan() - (right / broadening).atan()
}

fn broadened_self_energy_response_arguments(
    branch: SfconvBroadenedSelfEnergyBranch,
    momentum: Real,
    shifted_energy: Real,
    context: SfconvSelfEnergyContext,
    dispersion: Real,
) -> (Real, Real) {
    let minus_energy = (context.photoelectron_momentum - momentum).powi(2) / 2.0;
    let plus_energy = (context.photoelectron_momentum + momentum).powi(2) / 2.0;
    match branch {
        SfconvBroadenedSelfEnergyBranch::ParticlePair => (
            shifted_energy - dispersion - minus_energy,
            shifted_energy - dispersion - plus_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::ParticleFermi => (
            shifted_energy - dispersion - context.fermi_energy,
            shifted_energy - dispersion - plus_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::HoleFermi => (
            shifted_energy + dispersion - minus_energy,
            shifted_energy + dispersion - context.fermi_energy,
        ),
        SfconvBroadenedSelfEnergyBranch::HolePair => (
            shifted_energy + dispersion - minus_energy,
            shifted_energy + dispersion - plus_energy,
        ),
    }
}

fn integrate_mksat_range(
    lower: Real,
    upper: Real,
    context: SfconvSatelliteContext,
    mut integrand: impl FnMut(Real, SfconvSatelliteContext) -> Result<Real, SfconvError>,
) -> Result<SfconvAdaptiveIntegral, SfconvError> {
    sfconv_grater_integrate(
        |momentum| integrand(momentum, context),
        lower,
        upper,
        context.plasma_frequency * context.accuracy,
        context.accuracy,
        &[],
    )
}

fn combine_satellite_integrals(
    first: SfconvAdaptiveIntegral,
    second: SfconvAdaptiveIntegral,
    normalization: Real,
) -> Result<SfconvSatelliteIntegral, SfconvError> {
    validate_nonzero_denominator("satellite normalization", normalization)?;
    let value = finite_result(
        "satellite integral",
        (first.value + second.value) / normalization,
    )?;
    Ok(SfconvSatelliteIntegral {
        value,
        estimated_error: (first.estimated_error + second.estimated_error) / normalization.abs(),
        evaluations: first.evaluations + second.evaluations,
        max_regions: first.max_regions.max(second.max_regions),
    })
}
