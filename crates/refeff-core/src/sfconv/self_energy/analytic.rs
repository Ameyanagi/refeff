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
