use super::*;

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
