use super::*;

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
