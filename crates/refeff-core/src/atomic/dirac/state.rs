use super::*;

/// Port of FEFF `ATOM/soldir.f90` entry-state setup.
///
/// FEFF starts every `soldir` call with `enav = 1`, makes the asymptotic large
/// component nonnegative, saves the originally requested method in `iex`, and
/// maps `method <= 0` to effective method 1.
pub fn atomic_dirac_entry_state(
    input: AtomicDiracEntryStateInput,
) -> Result<AtomicDiracEntryState, AtomMathError> {
    validate_dirac_entry_state_input(&input)?;
    Ok(calculate_atomic_dirac_entry_state(input))
}

/// Port of FEFF `ATOM/soldir.f90` `norm`.
///
/// This helper evaluates the radial and origin-development normalization term
/// used by `soldir` before scaling Dirac components. The `method == 1` branch
/// applies FEFF's matching-point correction to the small component.
pub fn atomic_dirac_normalization(
    input: AtomicDiracNormalizationInput<'_>,
) -> Result<AtomicDiracNormalization, AtomMathError> {
    validate_dirac_normalization_input(&input)?;
    calculate_atomic_dirac_normalization(input)
}

/// Port of FEFF `ATOM/soldir.f90` final normalization/sign-scaling block.
///
/// This helper takes the normalization integral produced by
/// [`atomic_dirac_normalization`], applies FEFF's sign conventions against the
/// initial origin coefficients, scales the active wavefunction prefix, and
/// clears inactive radial rows.
pub fn atomic_dirac_solution_normalization(
    input: AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<AtomicDiracSolutionNormalization, AtomMathError> {
    validate_dirac_solution_normalization_input(&input)?;
    calculate_atomic_dirac_solution_normalization(input)
}

/// Port of FEFF `ATOM/soldir.f90` node counting after solution matching.
///
/// FEFF initializes `nd = 1`, scans through `max(j, mat)`, skips divisions
/// when the previous large-component sample is exactly zero, and counts both
/// sign changes and samples that land exactly on zero.
pub fn atomic_dirac_node_count(
    input: AtomicDiracNodeCountInput<'_>,
) -> Result<AtomicDiracNodeCount, AtomMathError> {
    validate_dirac_node_count_input(&input)?;
    calculate_atomic_dirac_node_count(input)
}

/// Port of FEFF `ATOM/soldir.f90` node-count energy search.
///
/// After counting nodes, FEFF adjusts the trial energy and search brackets before
/// reintegrating. Exhausting `nes` attempts is not a hard error in FEFF: it sets
/// `ifail = 1` and continues to normalization, so this helper reports that state
/// separately from the old `numerr` exits.
pub fn atomic_dirac_node_energy_search(
    input: AtomicDiracNodeEnergySearchInput,
) -> Result<AtomicDiracNodeEnergySearch, AtomMathError> {
    validate_dirac_node_energy_search_input(&input)?;
    calculate_atomic_dirac_node_energy_search(input)
}

/// Port of FEFF `ATOM/soldir.f90` iteration restart setup at labels `101` and `105`.
///
/// FEFF clears the current diagnostic, selects `test1` or `test2` from the
/// current method, restores `en = edep`, resets the energy brackets, clears the
/// small-component matching attempt count, clears the node count, and starts the
/// node-search attempt count at zero for label `105`.
pub fn atomic_dirac_iteration_reset(
    input: AtomicDiracIterationResetInput,
) -> Result<AtomicDiracIterationReset, AtomMathError> {
    validate_dirac_iteration_reset_input(&input)?;
    Ok(calculate_atomic_dirac_iteration_reset(input))
}

/// Port of FEFF `ATOM/soldir.f90` abnormal-exit recovery at label `899`.
///
/// FEFF retries a failed requested inhomogeneous method-1 pass as method 2.
/// Homogeneous requests (`iex == 0`) and failures that already reached method 2
/// return without another restart.
pub fn atomic_dirac_abnormal_exit_recovery(
    input: AtomicDiracAbnormalExitRecoveryInput,
) -> Result<AtomicDiracAbnormalExitRecovery, AtomMathError> {
    calculate_atomic_dirac_abnormal_exit_recovery(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-1 energy correction.
///
/// FEFF uses the small-component disagreement at `mat` to form an additive
/// energy correction `f` and a relative mismatch `c`. When `gpmat` is exactly
/// zero, FEFF leaves `c` unscaled.
pub fn atomic_dirac_method_one_energy_correction(
    input: AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyCorrection, AtomMathError> {
    validate_dirac_method_one_energy_correction_input(&input)?;
    calculate_atomic_dirac_method_one_energy_correction(input)
}

/// Port of FEFF `ATOM/soldir.f90` energy-correction backtracking.
///
/// This applies `en = en + f`, then repeatedly halves `f` while FEFF's
/// positivity, relative-step, or bracket checks reject the trial. The result
/// also reports whether `abs(c) > test`, which is the condition that makes
/// `soldir` run another small-component matching iteration.
pub fn atomic_dirac_energy_step(
    input: AtomicDiracEnergyStepInput,
) -> Result<AtomicDiracEnergyStep, AtomMathError> {
    validate_dirac_energy_step_input(&input)?;
    calculate_atomic_dirac_energy_step(input)
}

/// Port of FEFF `ATOM/soldir.f90` small-component rematch attempt handling.
///
/// After an accepted energy step, FEFF increments `ies` only when
/// `abs(c) > test`. Attempts up to `nes` jump back to label `105`; attempts
/// beyond `nes` set `ifail = 1` and continue normalization with the current
/// solution.
pub fn atomic_dirac_rematch_attempt(
    input: AtomicDiracRematchAttemptInput,
) -> Result<AtomicDiracRematchAttempt, AtomMathError> {
    validate_dirac_rematch_attempt_input(&input)?;
    calculate_atomic_dirac_rematch_attempt(input)
}
fn calculate_atomic_dirac_entry_state(input: AtomicDiracEntryStateInput) -> AtomicDiracEntryState {
    AtomicDiracEntryState {
        previous_energy: 1.0,
        asymptotic_large_component: input.asymptotic_large_component.abs(),
        requested_method: input.method,
        method: if input.method <= 0 { 1 } else { input.method },
    }
}

fn calculate_atomic_dirac_normalization(
    input: AtomicDiracNormalizationInput<'_>,
) -> Result<AtomicDiracNormalization, AtomMathError> {
    let mut radial_terms = input
        .radii
        .iter()
        .zip(input.large_component.iter())
        .zip(input.small_component.iter())
        .take(input.active_len)
        .map(|((&radius, &large), &small)| radius * (large * large + small * small))
        .collect::<Vec<_>>();

    if input.method == 1 {
        let matching = input.matching_index_1based - 1;
        let small = input.small_component[matching];
        radial_terms[matching] += input.radii[matching]
            * (input.matching_small_component * input.matching_small_component - small * small)
            / 2.0;
    }

    let mut norm = 0.0;
    for row in (1..input.active_len).step_by(2) {
        norm += radial_terms[row] + radial_terms[row] + radial_terms[row + 1];
    }
    norm = input.step * (norm + norm + radial_terms[0] - radial_terms[input.active_len - 1]) / 3.0;

    let first_radius = input.radii[0];
    for coefficient in 1..=input.coefficient_count {
        let exponent = input.origin_power + input.origin_power + coefficient as Real;
        if exponent == 0.0 {
            return Err(AtomMathError::ZeroDiracNormalizationOriginExponent);
        }
        let factor = first_radius.powf(exponent) / exponent;
        for left in 0..coefficient {
            let right = coefficient - 1 - left;
            norm += input.large_coefficients[left] * factor * input.large_coefficients[right]
                + input.small_coefficients[left] * factor * input.small_coefficients[right];
        }
    }
    validate_finite_scalar("soldir_norm", norm)?;

    Ok(AtomicDiracNormalization { norm })
}

fn calculate_atomic_dirac_solution_normalization(
    input: AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<AtomicDiracSolutionNormalization, AtomMathError> {
    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    let norm_root = input.norm.sqrt();
    let mut coefficient_divisor = norm_root;
    if large_coefficients[0] * input.initial_large_coefficient < 0.0
        || small_coefficients[0] * input.initial_small_coefficient < 0.0
    {
        coefficient_divisor = -coefficient_divisor;
    }

    for (large, small) in large_coefficients
        .iter_mut()
        .zip(small_coefficients.iter_mut())
        .take(input.coefficient_count)
    {
        *large /= coefficient_divisor;
        *small /= coefficient_divisor;
    }

    let mut component_divisor = norm_root;
    if large_component[0] * input.initial_large_coefficient < 0.0
        || small_component[0] * input.initial_small_coefficient < 0.0
    {
        component_divisor = -component_divisor;
    }

    for (large, small) in large_component
        .iter_mut()
        .zip(small_component.iter_mut())
        .take(input.active_len)
    {
        *large /= component_divisor;
        *small /= component_divisor;
    }
    for (large, small) in large_component
        .iter_mut()
        .zip(small_component.iter_mut())
        .skip(input.active_len)
    {
        *large = 0.0;
        *small = 0.0;
    }

    Ok(AtomicDiracSolutionNormalization {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        component_divisor,
        coefficient_divisor,
    })
}

fn calculate_atomic_dirac_node_count(
    input: AtomicDiracNodeCountInput<'_>,
) -> Result<AtomicDiracNodeCount, AtomMathError> {
    let scan_index_1based = input.matching_index_1based.max(input.scan_index_1based);
    let mut node_count = 1;

    for row in 1..scan_index_1based {
        let previous = input.large_component[row - 1];
        if previous == 0.0 {
            continue;
        }
        if input.large_component[row] / previous <= 0.0 {
            node_count += 1;
        }
    }

    Ok(AtomicDiracNodeCount {
        node_count,
        scan_index_1based,
    })
}

fn calculate_atomic_dirac_node_energy_search(
    input: AtomicDiracNodeEnergySearchInput,
) -> Result<AtomicDiracNodeEnergySearch, AtomMathError> {
    let mut energy = input.energy;
    let mut energy_sup = input.energy_sup;
    let mut energy_inf = input.energy_inf;

    if input.node_count < input.target_node_count {
        energy_sup = energy;
        if energy_inf < 0.0 {
            energy = soldir_node_search_bisect(energy_inf, energy_sup, input.energy_precision)?;
        } else {
            energy *= 8.0e-1;
            if energy.abs() <= input.energy_precision {
                return Err(AtomMathError::DiracNodeEnergyTooSmall {
                    energy,
                    precision: input.energy_precision,
                });
            }
        }
    } else if input.node_count > input.target_node_count {
        energy_inf = energy;
        if energy_sup > input.energy_floor {
            energy = soldir_node_search_bisect(energy_inf, energy_sup, input.energy_precision)?;
        } else {
            energy *= 1.2;
            if energy <= input.energy_floor {
                return Err(AtomMathError::DiracNodeEnergyBelowPotentialFloor {
                    energy,
                    energy_floor: input.energy_floor,
                });
            }
        }
    } else {
        return Ok(AtomicDiracNodeEnergySearch {
            energy,
            energy_sup,
            energy_inf,
            search_attempt_count: input.search_attempt_count,
            needs_reintegration: false,
            attempts_exhausted: false,
        });
    }

    let search_attempt_count = input.search_attempt_count.checked_add(1).ok_or(
        AtomMathError::DiracNodeEnergyAttemptCountOutOfRange {
            search_attempt_count: input.search_attempt_count,
        },
    )?;
    let attempts_exhausted = search_attempt_count > input.max_attempt_count;

    validate_finite_scalar("soldir_node_energy", energy)?;
    validate_finite_scalar("soldir_node_energy_sup", energy_sup)?;
    validate_finite_scalar("soldir_node_energy_inf", energy_inf)?;

    Ok(AtomicDiracNodeEnergySearch {
        energy,
        energy_sup,
        energy_inf,
        search_attempt_count,
        needs_reintegration: !attempts_exhausted,
        attempts_exhausted,
    })
}

fn soldir_node_search_bisect(
    energy_inf: Real,
    energy_sup: Real,
    precision: Real,
) -> Result<Real, AtomMathError> {
    if (energy_inf - energy_sup).abs() > precision {
        Ok((energy_inf + energy_sup) / 2.0)
    } else {
        Err(AtomMathError::DiracNodeEnergyBracketCollapsed {
            energy_inf,
            energy_sup,
            precision,
        })
    }
}

fn calculate_atomic_dirac_iteration_reset(
    input: AtomicDiracIterationResetInput,
) -> AtomicDiracIterationReset {
    let mismatch_precision = if input.method > 1 {
        input.secondary_matching_precision
    } else {
        input.primary_matching_precision
    };

    AtomicDiracIterationReset {
        mismatch_precision,
        energy: input.reference_energy,
        energy_inf: 1.0,
        energy_sup: input.energy_floor,
        match_attempt_count: 0,
        node_count: 0,
        search_attempt_count: 0,
    }
}

fn calculate_atomic_dirac_abnormal_exit_recovery(
    input: AtomicDiracAbnormalExitRecoveryInput,
) -> Result<AtomicDiracAbnormalExitRecovery, AtomMathError> {
    if input.requested_method == 0 || input.method == 2 {
        return Ok(AtomicDiracAbnormalExitRecovery {
            method: input.method,
            needs_restart: false,
        });
    }

    let method = input
        .method
        .checked_add(1)
        .ok_or(AtomMathError::DiracMethodOutOfRange {
            method: input.method,
        })?;
    Ok(AtomicDiracAbnormalExitRecovery {
        method,
        needs_restart: true,
    })
}

fn calculate_atomic_dirac_method_one_energy_correction(
    input: AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyCorrection, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let mut mismatch = input.matching_small_component - input.small_component[matching];
    let correction = input.large_component[matching] * mismatch * input.speed_of_light / input.norm;
    if input.matching_small_component != 0.0 {
        mismatch /= input.matching_small_component;
    }

    validate_finite_scalar("soldir_energy_correction", correction)?;
    validate_finite_scalar("soldir_energy_mismatch", mismatch)?;

    Ok(AtomicDiracEnergyCorrection {
        correction,
        mismatch,
    })
}

fn calculate_atomic_dirac_energy_step(
    input: AtomicDiracEnergyStepInput,
) -> Result<AtomicDiracEnergyStep, AtomMathError> {
    let mut correction = input.correction;
    let mut energy = input.energy + correction;
    let denominator = energy - correction;
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyCorrectionDenominator);
    }
    let mut relative_step = (correction / denominator).abs();
    let needs_rematch = input.mismatch.abs() > input.mismatch_precision;

    loop {
        let rejected = energy >= 0.0
            || relative_step > 2.0e-1
            || (needs_rematch && (energy < input.energy_sup || energy > input.energy_inf));
        if !rejected {
            break;
        }
        correction /= 2.0;
        relative_step /= 2.0;
        energy -= correction;
        if relative_step <= input.zero_energy_precision {
            return Err(AtomMathError::DiracEnergyCorrectionTooSmall { relative_step });
        }
    }

    validate_finite_scalar("soldir_energy_step_energy", energy)?;
    validate_finite_scalar("soldir_energy_step_correction", correction)?;
    validate_finite_scalar("soldir_energy_step_relative", relative_step)?;

    Ok(AtomicDiracEnergyStep {
        energy,
        correction,
        relative_step,
        needs_rematch,
    })
}

fn calculate_atomic_dirac_rematch_attempt(
    input: AtomicDiracRematchAttemptInput,
) -> Result<AtomicDiracRematchAttempt, AtomMathError> {
    if input.mismatch.abs() <= input.mismatch_precision {
        return Ok(AtomicDiracRematchAttempt {
            match_attempt_count: input.match_attempt_count,
            needs_rematch: false,
            attempts_exhausted: false,
        });
    }

    let match_attempt_count = input.match_attempt_count.checked_add(1).ok_or(
        AtomMathError::DiracRematchAttemptCountOutOfRange {
            match_attempt_count: input.match_attempt_count,
        },
    )?;
    let attempts_exhausted = match_attempt_count > input.max_attempt_count;

    Ok(AtomicDiracRematchAttempt {
        match_attempt_count,
        needs_rematch: !attempts_exhausted,
        attempts_exhausted,
    })
}
