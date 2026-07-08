use super::*;

/// Port of FEFF `Sigma1`.
pub fn self_energy_single_pole(
    input: SelfEnergySinglePoleInput,
) -> Result<Complex, SelfEnergyError> {
    validate_single_pole_input(input)?;

    let momentum_re = input.momentum.re / input.fermi_momentum;
    let first_lower = Complex::new(momentum_re + 1.0, 0.0);
    let first_upper = first_lower * SELF_ENERGY_INF;
    let first_integrand = if input.use_broadened_pole {
        self_energy_bpr2_integrand
    } else {
        self_energy_r2_integrand
    };
    let first =
        integrate_single_pole(input, first_lower, first_upper, false, first_integrand, &[])?;

    let second_lower = Complex::new((momentum_re - 1.0).abs().max(SELF_ENERGY_ZERO_PL), 0.0);
    let second_upper = Complex::new(momentum_re + 1.0, 0.0);
    let second_integrand = if input.use_broadened_pole {
        self_energy_bpr1_integrand
    } else {
        self_energy_r1_integrand
    };
    let second = integrate_single_pole(
        input,
        second_lower,
        second_upper,
        false,
        second_integrand,
        &[],
    )?;

    let third = self_energy_single_pole_tail(input, false)?;
    let coefficient = Complex::new(
        -input.amplitude * input.pole_energy.powi(2)
            / (2.0 * std::f64::consts::PI * input.fermi_energy),
        0.0,
    ) / input.momentum;
    let value = coefficient * (first + second + third);
    ensure_finite_complex("Sigma1", value)?;
    Ok(value)
}

/// Port of FEFF `dSigma`, the energy derivative of `Sigma1`.
pub fn self_energy_single_pole_derivative(
    input: SelfEnergySinglePoleInput,
) -> Result<Complex, SelfEnergyError> {
    validate_single_pole_input(input)?;

    let normalized_momentum = input.momentum / input.fermi_momentum;
    let momentum_re = input.momentum.re / input.fermi_momentum;
    let first_lower = normalized_momentum + Complex::new(1.0, 0.0);
    let first_upper = first_lower * SELF_ENERGY_INF;
    let first_singularities =
        singularities_for_interval(input, first_lower, first_upper, SingularityFunction::Second)?;
    let first = integrate_single_pole(
        input,
        first_lower,
        first_upper,
        true,
        self_energy_dr2_integrand,
        &first_singularities,
    )?;

    let second_lower = Complex::new((momentum_re - 1.0).abs().max(SELF_ENERGY_ZERO_PL), 0.0);
    let second_upper = normalized_momentum + Complex::new(1.0, 0.0);
    let second_singularities = singularities_for_interval(
        input,
        second_lower,
        second_upper,
        SingularityFunction::First,
    )?;
    let second = integrate_single_pole(
        input,
        second_lower,
        second_upper,
        true,
        self_energy_dr1_integrand,
        &second_singularities,
    )?;

    let third = self_energy_single_pole_tail(input, true)?;
    let coefficient = Complex::new(
        -input.amplitude * input.pole_energy.powi(2)
            / (2.0 * std::f64::consts::PI * input.fermi_energy.powi(2)),
        0.0,
    ) / input.momentum;
    let value = coefficient * (first + second + third);
    ensure_finite_complex("dSigma", value)?;
    Ok(value)
}

/// Port of FEFF `CSigZ`.
pub fn many_pole_self_energy(
    input: ManyPoleSelfEnergyInput<'_>,
) -> Result<ManyPoleSelfEnergy, SelfEnergyError> {
    validate_many_pole_input(&input)?;

    let fermi_momentum = SELF_ENERGY_FERMI_MOMENTUM_FACTOR / input.radius;
    let fermi_energy = fermi_momentum.powi(2) / 2.0;
    let relative_energy =
        input.energy.re - input.fermi_level + fermi_energy + input.gap_energy / 2.0;
    let radicand = 2.0 * relative_energy;
    if radicand < 0.0 || !radicand.is_finite() {
        return Err(SelfEnergyError::NegativeRadicand {
            name: "CSigZ ck0",
            value: radicand,
        });
    }
    let momentum = Complex::new(radicand.sqrt(), 0.0);
    ensure_finite_complex("CSigZ ck0", momentum)?;

    let mut self_energy = Complex::new(0.0, 0.0);
    let mut derivative = Complex::new(0.0, 0.0);
    let relative_energy_complex = Complex::new(relative_energy, 0.0);
    for index in 0..input.active_pole_count {
        let width = if input.use_broadened_pole {
            input.pole_widths[index]
        } else {
            0.0
        };
        let single = SelfEnergySinglePoleInput {
            momentum,
            energy: relative_energy_complex,
            pole_energy: input.pole_frequencies[index],
            width,
            amplitude: input.amplitudes[index],
            fermi_momentum,
            fermi_energy,
            on_shell: true,
            use_broadened_pole: input.use_broadened_pole,
        };
        self_energy += self_energy_single_pole(single)?;
        derivative += self_energy_single_pole_derivative(single)?;
    }

    self_energy += hartree_fock_exchange(momentum, fermi_energy, fermi_momentum)?;
    let denominator = Complex::new(1.0, 0.0) - derivative;
    if denominator.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator {
            name: "CSigZ renormalization",
        });
    }
    let renormalization = Complex::new(1.0, 0.0) / denominator;
    ensure_finite_complex("CSigZ SigTot", self_energy)?;
    ensure_finite_complex("CSigZ ZTot", renormalization)?;

    Ok(ManyPoleSelfEnergy {
        self_energy,
        renormalization,
        momentum,
    })
}

fn self_energy_single_pole_tail(
    input: SelfEnergySinglePoleInput,
    derivative: bool,
) -> Result<Complex, SelfEnergyError> {
    let momentum_re = input.momentum.re / input.fermi_momentum;
    let lower = Complex::new(SELF_ENERGY_ZERO_PL, 0.0);
    let span = (momentum_re - 1.0).abs();
    if (input.momentum.re - input.fermi_momentum).abs() < SELF_ENERGY_ZERO_PL
        || span <= SELF_ENERGY_ZERO_PL
    {
        return Ok(Complex::new(0.0, 0.0));
    }

    if input.momentum.re < input.fermi_momentum {
        let upper = Complex::new(1.0 - momentum_re, 0.0);
        if derivative {
            let singularities =
                singularities_for_interval(input, lower, upper, SingularityFunction::Second)?;
            integrate_single_pole(
                input,
                lower,
                upper,
                true,
                self_energy_dr3_integrand,
                &singularities,
            )
        } else {
            let integrand = if input.use_broadened_pole {
                self_energy_bpr3_integrand
            } else {
                self_energy_r3_integrand
            };
            integrate_single_pole(input, lower, upper, false, integrand, &[])
        }
    } else {
        let upper = Complex::new(momentum_re - 1.0, 0.0);
        if derivative {
            let singularities =
                singularities_for_interval(input, lower, upper, SingularityFunction::Second)?;
            integrate_single_pole(
                input,
                lower,
                upper,
                true,
                self_energy_dr2_integrand,
                &singularities,
            )
        } else {
            let integrand = if input.use_broadened_pole {
                self_energy_bpr2_integrand
            } else {
                self_energy_r2_integrand
            };
            integrate_single_pole(input, lower, upper, false, integrand, &[])
        }
    }
}

fn integrate_single_pole(
    input: SelfEnergySinglePoleInput,
    lower: Complex,
    upper: Complex,
    derivative: bool,
    integrand: fn(SelfEnergyIntegrandInput) -> Result<Complex, SelfEnergyError>,
    singularities: &[Real],
) -> Result<Complex, SelfEnergyError> {
    let result = cgratr(
        |q| integrand(integrand_input(input, q, derivative)),
        lower,
        upper,
        SELF_ENERGY_ABS_ERR,
        SELF_ENERGY_REL_ERR,
        singularities,
    )?;
    Ok(result.value)
}

fn integrand_input(
    input: SelfEnergySinglePoleInput,
    q: Complex,
    derivative: bool,
) -> SelfEnergyIntegrandInput {
    let width_over_fermi = input.width / input.fermi_energy;
    let normalized_energy = input.energy / input.fermi_energy;
    SelfEnergyIntegrandInput {
        q,
        normalized_momentum: input.momentum / input.fermi_momentum,
        normalized_energy: if derivative {
            normalized_energy + Complex::new(0.0, width_over_fermi)
        } else {
            normalized_energy
        },
        plasmon_over_fermi: input.pole_energy / input.fermi_energy,
        width_over_fermi,
        gap_energy: 0.0,
        on_shell: input.on_shell,
    }
}

fn singularities_for_interval(
    input: SelfEnergySinglePoleInput,
    lower: Complex,
    upper: Complex,
    function: SingularityFunction,
) -> Result<Vec<Real>, SelfEnergyError> {
    let width_over_fermi = input.width / input.fermi_energy;
    let dp_parameters = [
        input.pole_energy / input.fermi_energy,
        width_over_fermi,
        input.energy.re / input.fermi_energy,
        0.0,
    ];
    let complex_parameters = [
        input.momentum / input.fermi_momentum,
        input.energy / input.fermi_energy + Complex::new(0.0, width_over_fermi),
    ];
    let singularities = find_self_energy_singularities(
        [lower, upper],
        dp_parameters,
        complex_parameters,
        function,
    )?;
    Ok(strict_internal_singularities(
        singularities,
        lower.re,
        upper.re,
    ))
}

fn strict_internal_singularities(singularities: Vec<Real>, lower: Real, upper: Real) -> Vec<Real> {
    let mut filtered = Vec::with_capacity(singularities.len());
    for value in singularities {
        if value <= lower || value >= upper || !value.is_finite() {
            continue;
        }
        if filtered
            .last()
            .is_none_or(|previous: &Real| value > *previous)
        {
            filtered.push(value);
        }
    }
    filtered
}

fn validate_single_pole_input(input: SelfEnergySinglePoleInput) -> Result<(), SelfEnergyError> {
    ensure_finite_complex("ck", input.momentum)?;
    ensure_finite_complex("Energy", input.energy)?;
    ensure_positive_real("Wi", input.pole_energy)?;
    ensure_nonnegative_real("Gamma", input.width)?;
    ensure_finite_real("Amp", input.amplitude)?;
    ensure_positive_real("kFermi", input.fermi_momentum)?;
    ensure_positive_real("EFermi", input.fermi_energy)?;
    if input.momentum.norm() == 0.0 {
        return Err(SelfEnergyError::ZeroDenominator { name: "Sigma1 ck" });
    }
    Ok(())
}

fn validate_many_pole_input(input: &ManyPoleSelfEnergyInput<'_>) -> Result<(), SelfEnergyError> {
    if input.active_pole_count == 0 {
        return Err(SelfEnergyError::InvalidPoleCount);
    }
    ensure_finite_complex("Energy", input.energy)?;
    ensure_finite_real("Mu", input.fermi_level)?;
    ensure_positive_real("Rs", input.radius)?;
    ensure_finite_real("EGap", input.gap_energy)?;
    ensure_min_len(
        "WpScl",
        input.pole_frequencies.len(),
        input.active_pole_count,
    )?;
    ensure_min_len("Gamma", input.pole_widths.len(), input.active_pole_count)?;
    ensure_min_len("AmpFac", input.amplitudes.len(), input.active_pole_count)?;

    for index in 0..input.active_pole_count {
        ensure_positive_real("WpScl", input.pole_frequencies[index])?;
        ensure_nonnegative_real("Gamma", input.pole_widths[index])?;
        ensure_finite_real("AmpFac", input.amplitudes[index])?;
    }
    Ok(())
}

fn ensure_min_len(
    name: &'static str,
    actual: usize,
    required: usize,
) -> Result<(), SelfEnergyError> {
    if actual >= required {
        Ok(())
    } else {
        Err(SelfEnergyError::LengthTooShort {
            name,
            required,
            actual,
        })
    }
}
