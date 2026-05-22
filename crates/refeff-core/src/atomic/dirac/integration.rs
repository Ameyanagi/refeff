use super::*;

/// Port of FEFF `ATOM/intdir.f90`, the real Dirac radial predictor-corrector.
///
/// This is the low-level integration step used by `soldir`: it can search for
/// a matching point, integrate through a supplied matching point, or generate
/// only the inward tail solution. FEFF reports failures through a global
/// `numerr`; this port returns structured errors instead.
pub fn atomic_dirac_integration(
    input: AtomicDiracIntegrationInput<'_>,
) -> Result<AtomicDiracIntegration, AtomMathError> {
    validate_dirac_integration_input(&input)?;
    calculate_atomic_dirac_integration(input)
}
fn calculate_atomic_dirac_integration(
    input: AtomicDiracIntegrationInput<'_>,
) -> Result<AtomicDiracIntegration, AtomMathError> {
    let mut large_component = input.large_source.to_owned();
    let mut small_component = input.small_source.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();
    let predictor = ATOM_INTDIR_PREDICTOR;
    let (corrector, correction_mix) = atom_intdir_corrector_coefficients();
    let mut step = input.step / ATOM_INTDIR_STEP_DIVISOR;
    let energy_scaled = input.energy / input.speed_of_light;
    let doubled_speed = input.speed_of_light + input.speed_of_light;
    let kappa = input.kappa as Real;
    let angular_term = kappa * (kappa + 1.0) / doubled_speed;
    let mut matching_index_1based = input.matching_index_1based;
    let mut max_index_1based = input.max_index_1based;
    let mut tail_large = input.asymptotic_large_component;
    let mut matching_large_component = None;
    let mut matching_small_component = None;

    if input.mode != AtomicDiracIntegrationMode::InwardOnly {
        if input.mode == AtomicDiracIntegrationMode::SearchMatchingPoint {
            matching_index_1based = atom_intdir_search_matching_point(
                input.radii,
                input.potential,
                energy_scaled,
                angular_term,
                input.active_len,
            )?;
        }
        atom_intdir_origin_expansion(
            &mut large_coefficients,
            &mut small_coefficients,
            input,
            energy_scaled,
            doubled_speed,
            kappa,
        )?;

        let matching = matching_index_1based - 1;
        let mut large_derivative = [0.0; ATOM_INTDIR_HISTORY];
        let mut small_derivative = [0.0; ATOM_INTDIR_HISTORY];
        atom_intdir_initial_history(AtomIntdirInitialHistory {
            large_component: &mut large_component,
            small_component: &mut small_component,
            large_derivative: &mut large_derivative,
            small_derivative: &mut small_derivative,
            large_coefficients: &large_coefficients,
            small_coefficients: &small_coefficients,
            radii: input.radii,
            origin_power: input.origin_power,
            coefficient_count: input.coefficient_count,
            step,
        });

        let saved_large = large_component[matching];
        let saved_small = small_component[matching];
        atom_intdir_sweep(AtomIntdirSweep {
            large_component: &mut large_component,
            small_component: &mut small_component,
            large_derivative: &mut large_derivative,
            small_derivative: &mut small_derivative,
            radii: input.radii,
            potential: input.potential,
            predictor,
            corrector,
            correction_mix,
            energy_scaled,
            doubled_speed,
            kappa,
            step,
            start_index_1based: ATOM_INTDIR_HISTORY,
            target_index_1based: matching_index_1based,
            direction: 1,
        })?;
        matching_large_component = Some(large_component[matching]);
        matching_small_component = Some(small_component[matching]);
        large_component[matching] = saved_large;
        small_component[matching] = saved_small;

        if input.mode == AtomicDiracIntegrationMode::SearchMatchingPoint {
            if let Some(outward_large) = matching_large_component {
                let tail_limit = input.matching_precision * outward_large.abs();
                if tail_large > tail_limit {
                    tail_large = tail_limit;
                }
            }
            max_index_1based = input.active_len + 2;
            atom_intdir_adjust_inward_start(
                &mut max_index_1based,
                matching_index_1based,
                input.radii,
                input.potential,
                energy_scaled,
                input.speed_of_light,
            )?;
        }
    }

    let mut large_derivative = [0.0; ATOM_INTDIR_HISTORY];
    let mut small_derivative = [0.0; ATOM_INTDIR_HISTORY];
    loop {
        step = -step;
        let decay = atom_intdir_decay(input.energy, input.speed_of_light)?;
        if decay * input.radii[max_index_1based - 1] >= ATOM_INTDIR_EXPONENT_FLOOR {
            let ratio = decay / (doubled_speed + energy_scaled);
            let mut scale = tail_large / (decay * input.radii[max_index_1based - 1]).exp();
            if scale == 0.0 {
                scale = 1.0;
            }
            for history in 0..ATOM_INTDIR_HISTORY {
                let row_1based = max_index_1based - history;
                let row = row_1based - 1;
                large_component[row] = scale * (decay * input.radii[row]).exp();
                small_component[row] = ratio * large_component[row];
                large_derivative[history] = decay * input.radii[row] * large_component[row] * step;
                small_derivative[history] = ratio * large_derivative[history];
            }
            atom_intdir_sweep(AtomIntdirSweep {
                large_component: &mut large_component,
                small_component: &mut small_component,
                large_derivative: &mut large_derivative,
                small_derivative: &mut small_derivative,
                radii: input.radii,
                potential: input.potential,
                predictor,
                corrector,
                correction_mix,
                energy_scaled,
                doubled_speed,
                kappa,
                step,
                start_index_1based: max_index_1based + 1 - ATOM_INTDIR_HISTORY,
                target_index_1based: matching_index_1based,
                direction: -1,
            })?;
            break;
        }
        atom_intdir_adjust_inward_start(
            &mut max_index_1based,
            matching_index_1based,
            input.radii,
            input.potential,
            energy_scaled,
            input.speed_of_light,
        )?;
    }

    validate_finite_vector("intdir_large_component", large_component.view())?;
    validate_finite_vector("intdir_small_component", small_component.view())?;
    validate_finite_vector("intdir_large_coefficient", large_coefficients.view())?;
    validate_finite_vector("intdir_small_coefficient", small_coefficients.view())?;

    Ok(AtomicDiracIntegration {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        matching_large_component,
        matching_small_component,
        matching_index_1based,
        max_index_1based,
    })
}

fn atom_intdir_corrector_coefficients() -> ([Real; ATOM_INTDIR_HISTORY], Real) {
    let mix = ATOM_INTDIR_MIX_NUMERATOR / ATOM_INTDIR_MIX_DENOMINATOR;
    let complement = 1.0 - mix;
    let mut corrector = ATOM_INTDIR_CORRECTOR_RAW;
    let correction_mix = mix * corrector[ATOM_INTDIR_HISTORY - 1];
    let mut previous = corrector[0];
    for index in 1..ATOM_INTDIR_HISTORY {
        let current = corrector[index];
        corrector[index] = mix * previous + complement * ATOM_INTDIR_PREDICTOR[index];
        previous = current;
    }
    corrector[0] = mix * ATOM_INTDIR_PREDICTOR[0];
    (corrector, correction_mix)
}

fn atom_intdir_search_matching_point(
    radii: ArrayView1<'_, Real>,
    potential: ArrayView1<'_, Real>,
    energy_scaled: Real,
    angular_term: Real,
    active_len: usize,
) -> Result<usize, AtomMathError> {
    let mut matching = ATOM_INTDIR_HISTORY;
    let mut sign = 1.0;
    loop {
        matching += 2;
        if matching >= active_len {
            if energy_scaled > -0.0003 {
                matching = active_len - 12;
            } else {
                return Err(AtomMathError::DiracIntegrationMatchingPointNotFound { active_len });
            }
        }
        let row = matching - 1;
        let value =
            (potential[row] + angular_term / (radii[row] * radii[row]) - energy_scaled) * sign;
        if value <= 0.0 {
            sign = -sign;
            if sign < 0.0 {
                continue;
            }
            if matching >= active_len - ATOM_INTDIR_HISTORY {
                matching = active_len - 12;
            }
            return Ok(matching);
        }
    }
}

fn atom_intdir_origin_expansion(
    large_coefficients: &mut Array1<Real>,
    small_coefficients: &mut Array1<Real>,
    input: AtomicDiracIntegrationInput<'_>,
    energy_scaled: Real,
    doubled_speed: Real,
    kappa: Real,
) -> Result<(), AtomMathError> {
    large_coefficients[0] = input.initial_large_coefficient;
    small_coefficients[0] = input.initial_small_coefficient;
    for coefficient in 1..input.coefficient_count {
        let order = coefficient as Real;
        let large_power = input.origin_power + kappa + order;
        let small_power = input.origin_power - kappa + order;
        let denominator = large_power * small_power + input.potential_coefficients[0].powi(2);
        if denominator == 0.0 {
            return Err(AtomMathError::ZeroDiracIntegrationDevelopmentDenominator {
                coefficient_1based: coefficient + 1,
            });
        }
        let mut large_source = (energy_scaled + doubled_speed)
            * small_coefficients[coefficient - 1]
            + small_coefficients[coefficient];
        let mut small_source =
            energy_scaled * large_coefficients[coefficient - 1] + large_coefficients[coefficient];
        for previous in 0..coefficient {
            large_source -= input.potential_coefficients[previous + 1]
                * small_coefficients[coefficient - 1 - previous];
            small_source -= input.potential_coefficients[previous + 1]
                * large_coefficients[coefficient - 1 - previous];
        }
        large_coefficients[coefficient] = (small_power * large_source
            + input.potential_coefficients[0] * small_source)
            / denominator;
        small_coefficients[coefficient] = (input.potential_coefficients[0] * large_source
            - large_power * small_source)
            / denominator;
    }

    Ok(())
}

struct AtomIntdirInitialHistory<'a> {
    large_component: &'a mut Array1<Real>,
    small_component: &'a mut Array1<Real>,
    large_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    small_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    large_coefficients: &'a Array1<Real>,
    small_coefficients: &'a Array1<Real>,
    radii: ArrayView1<'a, Real>,
    origin_power: Real,
    coefficient_count: usize,
    step: Real,
}

fn atom_intdir_initial_history(input: AtomIntdirInitialHistory<'_>) {
    let AtomIntdirInitialHistory {
        large_component,
        small_component,
        large_derivative,
        small_derivative,
        large_coefficients,
        small_coefficients,
        radii,
        origin_power,
        coefficient_count,
        step,
    } = input;
    for row in 0..ATOM_INTDIR_HISTORY {
        large_component[row] = 0.0;
        small_component[row] = 0.0;
        large_derivative[row] = 0.0;
        small_derivative[row] = 0.0;
        for coefficient in 0..coefficient_count {
            let power = origin_power + coefficient as Real;
            let radial_power = radii[row].powf(power);
            let derivative_scale = power * radial_power * step;
            large_component[row] += radial_power * large_coefficients[coefficient];
            small_component[row] += radial_power * small_coefficients[coefficient];
            large_derivative[row] += derivative_scale * large_coefficients[coefficient];
            small_derivative[row] += derivative_scale * small_coefficients[coefficient];
        }
    }
}

struct AtomIntdirSweep<'a> {
    large_component: &'a mut Array1<Real>,
    small_component: &'a mut Array1<Real>,
    large_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    small_derivative: &'a mut [Real; ATOM_INTDIR_HISTORY],
    radii: ArrayView1<'a, Real>,
    potential: ArrayView1<'a, Real>,
    predictor: [Real; ATOM_INTDIR_HISTORY],
    corrector: [Real; ATOM_INTDIR_HISTORY],
    correction_mix: Real,
    energy_scaled: Real,
    doubled_speed: Real,
    kappa: Real,
    step: Real,
    start_index_1based: usize,
    target_index_1based: usize,
    direction: isize,
}

fn atom_intdir_sweep(input: AtomIntdirSweep<'_>) -> Result<(), AtomMathError> {
    let AtomIntdirSweep {
        large_component,
        small_component,
        large_derivative,
        small_derivative,
        radii,
        potential,
        predictor,
        corrector,
        correction_mix,
        energy_scaled,
        doubled_speed,
        kappa,
        step,
        start_index_1based,
        target_index_1based,
        direction,
    } = input;
    let mut row_1based = start_index_1based;
    let correction_step = correction_mix * step;
    loop {
        let current = row_1based - 1;
        let mut predicted_large = large_component[current] + large_derivative[0] * predictor[0];
        let mut predicted_small = small_component[current] + small_derivative[0] * predictor[0];
        row_1based = row_1based.saturating_add_signed(direction);
        let row = row_1based - 1;
        let previous_small = small_component[row];
        let previous_large = large_component[row];
        large_component[row] = predicted_large - large_derivative[0] * corrector[0];
        small_component[row] = predicted_small - small_derivative[0] * corrector[0];
        for history in 1..ATOM_INTDIR_HISTORY {
            predicted_large += large_derivative[history] * predictor[history];
            predicted_small += small_derivative[history] * predictor[history];
            large_component[row] += large_derivative[history] * corrector[history];
            small_component[row] += small_derivative[history] * corrector[history];
            large_derivative[history - 1] = large_derivative[history];
            small_derivative[history - 1] = small_derivative[history];
        }
        let scaled_potential = (energy_scaled - potential[row]) * radii[row];
        let shifted_potential = scaled_potential + doubled_speed * radii[row];
        large_component[row] += correction_step
            * (shifted_potential * predicted_small - kappa * predicted_large + previous_small);
        small_component[row] += correction_step
            * (kappa * predicted_small - scaled_potential * predicted_large - previous_large);
        large_derivative[ATOM_INTDIR_HISTORY - 1] = step
            * (shifted_potential * small_component[row] - kappa * large_component[row]
                + previous_small);
        small_derivative[ATOM_INTDIR_HISTORY - 1] = step
            * (kappa * small_component[row]
                - scaled_potential * large_component[row]
                - previous_large);
        if row_1based == target_index_1based {
            return Ok(());
        }
    }
}

fn atom_intdir_adjust_inward_start(
    max_index_1based: &mut usize,
    matching_index_1based: usize,
    radii: ArrayView1<'_, Real>,
    potential: ArrayView1<'_, Real>,
    energy_scaled: Real,
    speed_of_light: Real,
) -> Result<(), AtomMathError> {
    let threshold = ATOM_INTDIR_INWARD_THRESHOLD / speed_of_light;
    loop {
        *max_index_1based = max_index_1based.checked_sub(2).ok_or(
            AtomMathError::DiracIntegrationInwardStartTooClose {
                matching_index_1based,
            },
        )?;
        if (*max_index_1based + 1) <= (matching_index_1based + ATOM_INTDIR_HISTORY) {
            return Err(AtomMathError::DiracIntegrationInwardStartTooClose {
                matching_index_1based,
            });
        }
        let row = *max_index_1based - 1;
        if (potential[row] - energy_scaled) * radii[row] * radii[row] <= threshold {
            return Ok(());
        }
    }
}

pub(in crate::atomic) fn atom_intdir_decay(
    energy: Real,
    speed_of_light: Real,
) -> Result<Real, AtomMathError> {
    let energy_scaled = energy / speed_of_light;
    let doubled_speed = speed_of_light + speed_of_light;
    let radicand = -energy_scaled * (doubled_speed + energy_scaled);
    if radicand.is_finite() && radicand > 0.0 {
        Ok(-radicand.sqrt())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationEnergy {
            energy,
            speed_of_light,
        })
    }
}
