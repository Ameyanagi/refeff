use super::*;

pub(super) fn self_energy_fermi_limit_derivatives(
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

pub(super) fn self_energy_upper_limit_derivative(
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

pub(super) fn self_energy_lower_limit_derivative(
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

pub(super) fn self_energy_q_limit_derivative(
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

pub(super) fn self_energy_imaginary_derivative_factor(
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
