mod types;

pub use types::*;

use ndarray::{Array1, ArrayView1};

use crate::Real;

use super::types::*;
use super::validation::*;

pub(super) const ATOM_INTDIR_HISTORY: usize = 5;
const ATOM_INTDIR_PREDICTOR: [Real; ATOM_INTDIR_HISTORY] =
    [251.0, -1274.0, 2616.0, -2774.0, 1901.0];
const ATOM_INTDIR_CORRECTOR_RAW: [Real; ATOM_INTDIR_HISTORY] = [-19.0, 106.0, -264.0, 646.0, 251.0];
const ATOM_INTDIR_MIX_NUMERATOR: Real = 473.0;
const ATOM_INTDIR_MIX_DENOMINATOR: Real = 502.0;
const ATOM_INTDIR_STEP_DIVISOR: Real = 720.0;
const ATOM_INTDIR_INWARD_THRESHOLD: Real = 700.0;
const ATOM_INTDIR_EXPONENT_FLOOR: Real = -170.0;
const ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN: usize = 10;
pub(super) const ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET: usize = 12;
const ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY: Real = 1.0e-1;

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

/// Port of FEFF `ATOM/soldir.f90` setup before the first `intdir` call.
///
/// This evaluates the deterministic scalar state used by the shooting loop:
/// method fallback, point-nucleus small-component coefficient adjustment,
/// angular term, target node count, and the lower apparent-potential energy
/// bound.
pub fn atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    validate_dirac_solver_setup_input(&input)?;
    calculate_atomic_dirac_solver_setup(input)
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

pub(super) fn atom_intdir_decay(energy: Real, speed_of_light: Real) -> Result<Real, AtomMathError> {
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

fn calculate_atomic_dirac_solver_setup(
    input: AtomicDiracSolverSetupInput<'_>,
) -> Result<AtomicDiracSolverSetup, AtomMathError> {
    let requested_method = input.method;
    let method = if input.method <= 0 { 1 } else { input.method };
    let kappa = input.kappa as Real;
    let doubled_speed_of_light = input.speed_of_light + input.speed_of_light;
    let potential_origin = input.potential_coefficients[0];
    let mut initial_small_coefficient = input.initial_small_coefficient;

    if potential_origin < 0.0 {
        if input.kappa > 0 {
            initial_small_coefficient =
                -input.initial_large_coefficient * (kappa + input.origin_power) / potential_origin;
        } else if input.kappa < 0 {
            let denominator = kappa - input.origin_power;
            if denominator == 0.0 {
                return Err(AtomMathError::ZeroDiracSolverInitialCoefficientDenominator);
            }
            initial_small_coefficient =
                -input.initial_large_coefficient * potential_origin / denominator;
        }
    }

    let angular_term = kappa * (kappa + 1.0) / doubled_speed_of_light;
    let kappa_abs = input.kappa.unsigned_abs();
    let principal = i32::try_from(input.principal_quantum_number).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let kappa_abs_i32 = i32::try_from(kappa_abs).map_err(|_| {
        AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        }
    })?;
    let mut target_nodes = principal - kappa_abs_i32;
    if input.kappa < 0 {
        target_nodes += 1;
    }

    let mut energy_floor = 0.0;
    for row in 0..input.active_len {
        let radius = input.radii[row];
        let apparent =
            (angular_term / (radius * radius) + input.potential[row]) * input.speed_of_light;
        if apparent < energy_floor {
            energy_floor = apparent;
        }
    }
    if energy_floor >= 0.0 {
        return Err(AtomMathError::DiracSolverPotentialNotAttractive { energy_floor });
    }

    let energy = if input.energy < energy_floor {
        energy_floor * 0.9
    } else {
        input.energy
    };

    validate_finite_scalar("soldir_setup_energy", energy)?;
    validate_finite_scalar("soldir_setup_energy_floor", energy_floor)?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        initial_small_coefficient,
    )?;
    validate_finite_scalar("soldir_setup_angular_term", angular_term)?;

    Ok(AtomicDiracSolverSetup {
        requested_method,
        method,
        energy,
        energy_floor,
        initial_small_coefficient,
        angular_term,
        target_nodes,
        doubled_speed_of_light,
    })
}
