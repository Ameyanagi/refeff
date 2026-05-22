use super::*;

/// Port of FEFF `ATOM/soldir.f90` method-0 homogeneous tail matching.
///
/// FEFF scales only the tail rows `mat..=max0` so the outward solution matches
/// the inward large component at `mat`, then sets `j = mat` for node counting.
pub fn atomic_dirac_homogeneous_match(
    input: AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<AtomicDiracHomogeneousMatch, AtomMathError> {
    validate_dirac_homogeneous_match_input(&input)?;
    calculate_atomic_dirac_homogeneous_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-1 large-component matching.
///
/// FEFF computes the large-component mismatch at `mat`, derives a homogeneous
/// tail scale from `hg(mat)`, and adds the homogeneous solution from `mat`
/// through `max0` to match the inward large component.
pub fn atomic_dirac_large_component_match(
    input: AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<AtomicDiracLargeComponentMatch, AtomMathError> {
    validate_dirac_large_component_match_input(&input)?;
    calculate_atomic_dirac_large_component_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 two-component matching.
///
/// FEFF solves a two-by-two matching system using the homogeneous inward and
/// outward values. The prefix scale updates rows before `mat` plus origin
/// coefficients, while the tail scale updates rows from `mat` through `max0`.
pub fn atomic_dirac_two_component_match(
    input: AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<AtomicDiracTwoComponentMatch, AtomMathError> {
    validate_dirac_two_component_match_input(&input)?;
    calculate_atomic_dirac_two_component_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy-disagreement matching.
///
/// This names the second `method >= 2` two-component match in `soldir`, where
/// the matched system is the derivative solution `bg/bp/bgh/bph` produced by
/// the energy-disagreement integration.
pub fn atomic_dirac_energy_disagreement_match(
    input: AtomicDiracEnergyDisagreementMatchInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementMatch, AtomMathError> {
    let match_input = dirac_energy_disagreement_match_as_two_component(&input);
    validate_dirac_two_component_match_input(&match_input)?;
    calculate_atomic_dirac_energy_disagreement_match(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy-disagreement source setup.
///
/// This builds `bg/bp/bgh/bph` immediately before FEFF calls `intdir` for the
/// energy-disagreement system. The first coefficient slots are left zero here
/// because `intdir` overwrites them with `agi/api` before using the expansion.
pub fn atomic_dirac_energy_disagreement_source(
    input: AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementSource, AtomMathError> {
    validate_dirac_energy_disagreement_source_input(&input)?;
    calculate_atomic_dirac_energy_disagreement_source(input)
}

/// Port of FEFF `ATOM/soldir.f90` method-2 energy correction.
///
/// After FEFF integrates and matches the energy-disagreement system, it forms
/// the cross integral between the current solution and the derivative solution,
/// converts the normalization mismatch into an energy correction, and applies
/// that correction to the radial components and origin-development terms.
pub fn atomic_dirac_energy_disagreement_correction(
    input: AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementCorrection, AtomMathError> {
    validate_dirac_energy_disagreement_correction_input(&input)?;
    calculate_atomic_dirac_energy_disagreement_correction(input)
}

/// Port of FEFF `ATOM/soldir.f90` matching-point relocation.
///
/// FEFF searches the maximum `gg(i)^2`, relocates `mat` at most once, keeps
/// matching points odd, and falls back to `max0 - 12` when the peak is too
/// close to the tail for another integration pass.
pub fn atomic_dirac_matching_point_update(
    input: AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<AtomicDiracMatchingPointUpdate, AtomMathError> {
    validate_dirac_matching_point_update_input(&input)?;
    calculate_atomic_dirac_matching_point_update(input)
}

/// Port of FEFF `ATOM/soldir.f90` inhomogeneous `intdir` seed setup.
///
/// This copies `eg/ep` into the component work arrays and shifts `ceg/cep`
/// into coefficient slots `2..=ndor`. The first coefficient slot is left zero
/// here because `intdir` overwrites it with `agi/api` before use.
pub fn atomic_dirac_inhomogeneous_seed(
    input: AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    validate_dirac_inhomogeneous_seed_input(&input)?;
    calculate_atomic_dirac_inhomogeneous_seed(input)
}

/// Port of FEFF `ATOM/soldir.f90` branch after the inhomogeneous `intdir` pass.
///
/// FEFF checks the originally requested method (`iex`) after the first
/// integration. A homogeneous request (`iex == 0`) skips the homogeneous
/// correction pass and directly tail-matches the integrated solution; all
/// nonzero requests jump to the homogeneous correction integration.
pub fn atomic_dirac_inhomogeneous_branch(
    input: AtomicDiracInhomogeneousBranchInput,
) -> Result<AtomicDiracInhomogeneousBranch, AtomMathError> {
    Ok(calculate_atomic_dirac_inhomogeneous_branch(input))
}

/// Port of FEFF `ATOM/soldir.f90` homogeneous `intdir` seed setup.
///
/// FEFF zeros `hg/hp/agh/aph` before integrating the homogeneous system. This
/// helper returns those zeroed arrays in the same structure used by
/// [`atomic_dirac_inhomogeneous_seed`].
pub fn atomic_dirac_homogeneous_seed(
    input: AtomicDiracHomogeneousSeedInput,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    validate_dirac_homogeneous_seed_input(&input)?;
    Ok(calculate_atomic_dirac_homogeneous_seed(input))
}

/// Port of FEFF `ATOM/soldir.f90` homogeneous `intdir` pass setup.
///
/// Before integrating the homogeneous system, FEFF fixes the matching point with
/// `imm = 1`, except for method 1 where it sets `imm = -1` and runs only the
/// inward pass.
pub fn atomic_dirac_homogeneous_pass_setup(
    input: AtomicDiracHomogeneousPassSetupInput,
) -> Result<AtomicDiracHomogeneousPassSetup, AtomMathError> {
    Ok(calculate_atomic_dirac_homogeneous_pass_setup(input))
}

/// Port of FEFF `ATOM/soldir.f90` shooting-pass setup at label `106`.
///
/// FEFF resets `modmat`, chooses whether `intdir` should search a new matching
/// point or reuse the current one from the relative energy change, then updates
/// `enav` to the current trial energy.
pub fn atomic_dirac_shooting_pass_setup(
    input: AtomicDiracShootingPassSetupInput,
) -> Result<AtomicDiracShootingPassSetup, AtomMathError> {
    validate_dirac_shooting_pass_setup_input(&input)?;
    calculate_atomic_dirac_shooting_pass_setup(input)
}
fn calculate_atomic_dirac_homogeneous_match(
    input: AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<AtomicDiracHomogeneousMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let denominator = input.large_component[matching];
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_homogeneous_match_large_component",
        });
    }

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let tail_scale = input.matching_large_component / denominator;

    for row in matching..input.active_len {
        large_component[row] *= tail_scale;
        small_component[row] *= tail_scale;
    }

    validate_finite_scalar("soldir_homogeneous_match_scale", tail_scale)?;
    validate_finite_vector(
        "soldir_homogeneous_match_large_component",
        large_component.view(),
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_small_component",
        small_component.view(),
    )?;

    Ok(AtomicDiracHomogeneousMatch {
        large_component,
        small_component,
        tail_scale,
        scan_index_1based: input.matching_index_1based,
    })
}

fn calculate_atomic_dirac_large_component_match(
    input: AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<AtomicDiracLargeComponentMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let denominator = input.homogeneous_large_component[matching];
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_match_homogeneous_large_component",
        });
    }

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let large_mismatch = large_component[matching] - input.matching_large_component;
    let tail_scale = -large_mismatch / denominator;

    for row in matching..input.active_len {
        large_component[row] += tail_scale * input.homogeneous_large_component[row];
        small_component[row] += tail_scale * input.homogeneous_small_component[row];
    }

    validate_finite_scalar("soldir_large_match_scale", tail_scale)?;
    validate_finite_vector("soldir_large_match_large_component", large_component.view())?;
    validate_finite_vector("soldir_large_match_small_component", small_component.view())?;

    Ok(AtomicDiracLargeComponentMatch {
        large_component,
        small_component,
        tail_scale,
        large_mismatch,
    })
}

fn calculate_atomic_dirac_two_component_match(
    input: AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<AtomicDiracTwoComponentMatch, AtomMathError> {
    let matching = input.matching_index_1based - 1;
    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    let large_mismatch = large_component[matching] - input.matching_large_component;
    let small_mismatch = small_component[matching] - input.matching_small_component;
    let determinant = input.homogeneous_matching_small_component
        * input.homogeneous_large_component[matching]
        - input.homogeneous_matching_large_component * input.homogeneous_small_component[matching];
    if determinant == 0.0 {
        return Err(AtomMathError::ZeroDiracMatchDenominator {
            field: "soldir_match_determinant",
        });
    }

    let prefix_scale = (small_mismatch * input.homogeneous_large_component[matching]
        - large_mismatch * input.homogeneous_small_component[matching])
        / determinant;
    let tail_scale = (small_mismatch * input.homogeneous_matching_large_component
        - large_mismatch * input.homogeneous_matching_small_component)
        / determinant;

    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] +=
            prefix_scale * input.homogeneous_large_coefficients[coefficient];
        small_coefficients[coefficient] +=
            prefix_scale * input.homogeneous_small_coefficients[coefficient];
    }
    for row in 0..matching {
        large_component[row] += prefix_scale * input.homogeneous_large_component[row];
        small_component[row] += prefix_scale * input.homogeneous_small_component[row];
    }
    for row in matching..input.active_len {
        large_component[row] += tail_scale * input.homogeneous_large_component[row];
        small_component[row] += tail_scale * input.homogeneous_small_component[row];
    }

    validate_finite_scalar("soldir_two_match_determinant", determinant)?;
    validate_finite_scalar("soldir_two_match_prefix_scale", prefix_scale)?;
    validate_finite_scalar("soldir_two_match_tail_scale", tail_scale)?;
    validate_finite_vector("soldir_two_match_large_component", large_component.view())?;
    validate_finite_vector("soldir_two_match_small_component", small_component.view())?;
    validate_finite_vector(
        "soldir_two_match_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_two_match_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracTwoComponentMatch {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        determinant,
        prefix_scale,
        tail_scale,
        large_mismatch,
        small_mismatch,
    })
}

fn calculate_atomic_dirac_energy_disagreement_match(
    input: AtomicDiracEnergyDisagreementMatchInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementMatch, AtomMathError> {
    let matched = calculate_atomic_dirac_two_component_match(
        dirac_energy_disagreement_match_as_two_component(&input),
    )?;

    Ok(AtomicDiracEnergyDisagreementMatch {
        large_derivative: matched.large_component,
        small_derivative: matched.small_component,
        large_derivative_coefficients: matched.large_coefficients,
        small_derivative_coefficients: matched.small_coefficients,
        determinant: matched.determinant,
        prefix_scale: matched.prefix_scale,
        tail_scale: matched.tail_scale,
        large_mismatch: matched.large_mismatch,
        small_mismatch: matched.small_mismatch,
    })
}

fn dirac_energy_disagreement_match_as_two_component<'a>(
    input: &AtomicDiracEnergyDisagreementMatchInput<'a>,
) -> AtomicDiracTwoComponentMatchInput<'a> {
    AtomicDiracTwoComponentMatchInput {
        large_component: input.large_derivative,
        small_component: input.small_derivative,
        large_coefficients: input.large_derivative_coefficients,
        small_coefficients: input.small_derivative_coefficients,
        homogeneous_large_component: input.homogeneous_large_component,
        homogeneous_small_component: input.homogeneous_small_component,
        homogeneous_large_coefficients: input.homogeneous_large_coefficients,
        homogeneous_small_coefficients: input.homogeneous_small_coefficients,
        matching_large_component: input.matching_large_derivative,
        matching_small_component: input.matching_small_derivative,
        homogeneous_matching_large_component: input.homogeneous_matching_large_component,
        homogeneous_matching_small_component: input.homogeneous_matching_small_component,
        coefficient_count: input.coefficient_count,
        active_len: input.active_len,
        matching_index_1based: input.matching_index_1based,
    }
}

fn calculate_atomic_dirac_energy_disagreement_source(
    input: AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementSource, AtomMathError> {
    let mut large_source = Array1::<Real>::zeros(input.large_component.len());
    let mut small_source = Array1::<Real>::zeros(input.small_component.len());
    let mut large_coefficients = Array1::<Real>::zeros(input.large_coefficients.len());
    let mut small_coefficients = Array1::<Real>::zeros(input.small_coefficients.len());

    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] =
            input.large_coefficients[coefficient - 1] / input.speed_of_light;
        small_coefficients[coefficient] =
            input.small_coefficients[coefficient - 1] / input.speed_of_light;
    }
    for row in 0..input.active_len {
        let scale = input.radii[row] / input.speed_of_light;
        large_source[row] = input.large_component[row] * scale;
        small_source[row] = input.small_component[row] * scale;
    }

    validate_finite_vector(
        "soldir_energy_disagreement_large_source",
        large_source.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_source",
        small_source.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracEnergyDisagreementSource {
        large_source,
        small_source,
        large_coefficients,
        small_coefficients,
    })
}

fn calculate_atomic_dirac_energy_disagreement_correction(
    input: AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<AtomicDiracEnergyDisagreementCorrection, AtomMathError> {
    let overlap_integral = atomic_dirac_energy_disagreement_overlap(&input)?;
    let denominator = overlap_integral + overlap_integral;
    if denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionIntegral);
    }

    let normalization_mismatch = 1.0 - input.norm;
    let correction = normalization_mismatch / denominator;

    let mut large_component = input.large_component.to_owned();
    let mut small_component = input.small_component.to_owned();
    let mut large_coefficients = input.large_coefficients.to_owned();
    let mut small_coefficients = input.small_coefficients.to_owned();

    for row in 0..input.active_len {
        large_component[row] += correction * input.large_derivative[row];
        small_component[row] += correction * input.small_derivative[row];
    }
    for coefficient in 0..input.coefficient_count {
        large_coefficients[coefficient] +=
            correction * input.large_derivative_coefficients[coefficient];
        small_coefficients[coefficient] +=
            correction * input.small_derivative_coefficients[coefficient];
    }

    validate_finite_scalar(
        "soldir_energy_disagreement_correction_integral",
        overlap_integral,
    )?;
    validate_finite_scalar("soldir_energy_disagreement_correction", correction)?;
    validate_finite_scalar(
        "soldir_energy_disagreement_correction_mismatch",
        normalization_mismatch,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_component",
        large_component.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_component",
        small_component.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_coefficient",
        large_coefficients.view(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_coefficient",
        small_coefficients.view(),
    )?;

    Ok(AtomicDiracEnergyDisagreementCorrection {
        large_component,
        small_component,
        large_coefficients,
        small_coefficients,
        overlap_integral,
        correction,
        normalization_mismatch,
    })
}

fn atomic_dirac_energy_disagreement_overlap(
    input: &AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<Real, AtomMathError> {
    let origin_denominator = input.origin_power + input.origin_power + 1.0;
    if origin_denominator == 0.0 {
        return Err(AtomMathError::ZeroDiracEnergyDisagreementCorrectionOriginExponent);
    }

    let overlap_at = |row: usize| {
        (input.large_component[row] * input.large_derivative[row]
            + input.small_component[row] * input.small_derivative[row])
            * input.radii[row]
    };
    let first = overlap_at(0);
    let last = overlap_at(input.active_len - 1);
    let middle = (1..input.active_len - 1)
        .map(|row| {
            let weight = if row % 2 == 1 { 4.0 } else { 2.0 };
            weight * overlap_at(row)
        })
        .sum::<Real>();
    let overlap = input.step * (first + middle + last) / 3.0 + first / origin_denominator;
    validate_finite_scalar("soldir_energy_disagreement_overlap", overlap)?;
    Ok(overlap)
}

fn calculate_atomic_dirac_matching_point_update(
    input: AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<AtomicDiracMatchingPointUpdate, AtomMathError> {
    let mut peak_index_1based = None;
    let mut peak_square = 0.0;
    for (row, &large) in input
        .large_component
        .iter()
        .take(input.active_len)
        .enumerate()
    {
        let square = large * large;
        validate_finite_scalar("soldir_matching_point_square", square)?;
        if square > peak_square {
            peak_square = square;
            peak_index_1based = Some(row + 1);
        }
    }
    let peak_index_1based = peak_index_1based.ok_or(AtomMathError::DiracMatchingPointNotFound {
        active_len: input.active_len,
    })?;

    let mut matching_index_1based = input.matching_index_1based;
    let mut scan_index_1based = peak_index_1based.max(matching_index_1based);
    let mut relocated = input.already_relocated;
    let mut needs_reintegration = false;

    if peak_index_1based > matching_index_1based && !input.already_relocated {
        relocated = true;
        matching_index_1based = odd_matching_index(peak_index_1based);
        if matching_index_1based < input.active_len - ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN {
            needs_reintegration = true;
            scan_index_1based = peak_index_1based.max(matching_index_1based);
        } else {
            let fallback_index_1based =
                input.active_len - ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET;
            matching_index_1based = odd_matching_index(fallback_index_1based);
            scan_index_1based = fallback_index_1based.max(matching_index_1based);
        }
    }

    Ok(AtomicDiracMatchingPointUpdate {
        matching_index_1based,
        peak_index_1based,
        scan_index_1based,
        relocated,
        needs_reintegration,
    })
}

fn odd_matching_index(index_1based: usize) -> usize {
    if index_1based.is_multiple_of(2) {
        index_1based + 1
    } else {
        index_1based
    }
}

fn calculate_atomic_dirac_inhomogeneous_seed(
    input: AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<AtomicDiracIntegrationSeed, AtomMathError> {
    let large_source = input.large_source.to_owned();
    let small_source = input.small_source.to_owned();
    let mut large_coefficients = Array1::<Real>::zeros(input.large_source_coefficients.len());
    let mut small_coefficients = Array1::<Real>::zeros(input.small_source_coefficients.len());

    for coefficient in 1..input.coefficient_count {
        large_coefficients[coefficient] = input.large_source_coefficients[coefficient - 1];
        small_coefficients[coefficient] = input.small_source_coefficients[coefficient - 1];
    }

    validate_finite_vector("soldir_seed_large_source", large_source.view())?;
    validate_finite_vector("soldir_seed_small_source", small_source.view())?;
    validate_finite_vector("soldir_seed_large_coefficient", large_coefficients.view())?;
    validate_finite_vector("soldir_seed_small_coefficient", small_coefficients.view())?;

    Ok(AtomicDiracIntegrationSeed {
        large_source,
        small_source,
        large_coefficients,
        small_coefficients,
    })
}

fn calculate_atomic_dirac_inhomogeneous_branch(
    input: AtomicDiracInhomogeneousBranchInput,
) -> AtomicDiracInhomogeneousBranch {
    let action = if input.requested_method == 0 {
        AtomicDiracInhomogeneousBranchAction::MatchHomogeneousTail
    } else {
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    };

    AtomicDiracInhomogeneousBranch { action }
}

fn calculate_atomic_dirac_homogeneous_seed(
    input: AtomicDiracHomogeneousSeedInput,
) -> AtomicDiracIntegrationSeed {
    AtomicDiracIntegrationSeed {
        large_source: Array1::<Real>::zeros(input.radial_len),
        small_source: Array1::<Real>::zeros(input.radial_len),
        large_coefficients: Array1::<Real>::zeros(input.coefficient_len),
        small_coefficients: Array1::<Real>::zeros(input.coefficient_len),
    }
}

fn calculate_atomic_dirac_homogeneous_pass_setup(
    input: AtomicDiracHomogeneousPassSetupInput,
) -> AtomicDiracHomogeneousPassSetup {
    let raw_integration_flag = if input.method == 1 { -1 } else { 1 };
    let integration_mode = if input.method == 1 {
        AtomicDiracIntegrationMode::InwardOnly
    } else {
        AtomicDiracIntegrationMode::FixedMatchingPoint
    };

    AtomicDiracHomogeneousPassSetup {
        integration_mode,
        raw_integration_flag,
    }
}

fn calculate_atomic_dirac_shooting_pass_setup(
    input: AtomicDiracShootingPassSetupInput,
) -> Result<AtomicDiracShootingPassSetup, AtomMathError> {
    if input.energy == 0.0 {
        return Err(AtomMathError::ZeroDiracShootingPassEnergy);
    }
    let relative_energy_change = ((input.previous_energy - input.energy) / input.energy).abs();
    validate_finite_scalar(
        "soldir_shooting_pass_relative_energy_change",
        relative_energy_change,
    )?;

    let integration_mode = if relative_energy_change < ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY {
        AtomicDiracIntegrationMode::FixedMatchingPoint
    } else {
        AtomicDiracIntegrationMode::SearchMatchingPoint
    };

    Ok(AtomicDiracShootingPassSetup {
        integration_mode,
        reference_energy: input.energy,
        relative_energy_change,
        relocated: false,
    })
}
