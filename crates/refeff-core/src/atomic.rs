//! Atomic lookup tables and small ATOM helper kernels ported from FEFF.
//!
//! This module ports `ATOM/nucmass.f90` and `COMMON/pertab.f90`. FEFF stores
//! many unsuffixed real literals in double-precision arrays, so those values
//! are rounded through single precision before use; the Rust tables keep that
//! behavior explicitly. It also includes compact helper routines from
//! `ATOM/aprdev.f90`, `ATOM/cofcon.f90`, `ATOM/dentfa.f90`,
//! `ATOM/fdmocc.f90`, `ATOM/akeato.f90`, `ATOM/muatco.f90`,
//! `ATOM/inmuat.f90`,
//! `ATOM/lagdat.f90`, `ATOM/ortdat.f90`, `ATOM/tabrat.f90`,
//! `ATOM/fpf0.f90`, `ATOM/nucdev.f90`, `ATOM/dsordf.f90`,
//! `ATOM/yzkteg.f90`, `ATOM/yzkrdf.f90`, `ATOM/fdrirk.f90`,
//! `ATOM/vlda.f90`, `ATOM/potrdf.f90`, `ATOM/intdir.f90`,
//! `ATOM/soldir.f90`,
//! `ATOM/bkmrdf.f90`, and
//! `ATOM/s02at.f90`.

use crate::Real;
use crate::angular::wigner_3j;
use crate::exchange::{dirac_hara_exchange_potential, von_barth_hedin_potential};
use crate::grid::FEFF_FERMI_MOMENTUM_FACTOR;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder, Slice,
};

const ATOM_TABRAT_HARTREE_EV: Real = 27.211_396;
const ATOM_TABRAT_MOMENT_POWERS: [i32; 7] = [6, 4, 2, 1, -1, -2, -3];
const ATOM_TABRAT_LABELS: [&str; 9] = ["s", "p*", "p", "d*", "d", "f*", "f", "g*", "g"];
const ATOM_FPF0_BOHR_ANGSTROM: Real = 0.529_177_249;
const ATOM_FPF0_FINE_STRUCTURE: Real = 1.0 / 137.035_989_56;
const ATOM_FPF0_FORM_FACTOR_POINTS: usize = 81;
const ATOM_FPF0_MOMENTUM_STEP_INV_ANGSTROM: Real = 0.5;
const ATOM_NUCDEV_RADIUS_FACTOR: Real = 2.2677e-05;
const ATOM_INMUAT_WAVEFUNCTION_PRECISION: Real = 1.0e-5;
const ATOM_INMUAT_ENERGY_PRECISION: Real = 5.0e-6;
const ATOM_INMUAT_PRIMARY_RATIO: Real = 100.0;
const ATOM_INMUAT_SECONDARY_RATIO: Real = 10.0;
const ATOM_INMUAT_DEVELOPMENT_ORDER: usize = 10;
const ATOM_INMUAT_ATTEMPT_COUNT: usize = 50;
const ATOM_INMUAT_NUCLEUS_INDEX: usize = 11;
const ATOM_INMUAT_LAGRANGE_CAPACITY: usize = 820;
const ATOM_INMUAT_DEFAULT_RADIAL_COUNT: usize = 251;
const ATOM_INMUAT_ELECTRON_TOLERANCE: Real = 0.001;
const ATOM_INMUAT_DEFAULT_CONVERGENCE: Real = 0.3_f32 as Real;
const ATOM_INTDIR_HISTORY: usize = 5;
const ATOM_INTDIR_PREDICTOR: [Real; ATOM_INTDIR_HISTORY] =
    [251.0, -1274.0, 2616.0, -2774.0, 1901.0];
const ATOM_INTDIR_CORRECTOR_RAW: [Real; ATOM_INTDIR_HISTORY] = [-19.0, 106.0, -264.0, 646.0, 251.0];
const ATOM_INTDIR_MIX_NUMERATOR: Real = 473.0;
const ATOM_INTDIR_MIX_DENOMINATOR: Real = 502.0;
const ATOM_INTDIR_STEP_DIVISOR: Real = 720.0;
const ATOM_INTDIR_INWARD_THRESHOLD: Real = 700.0;
const ATOM_INTDIR_EXPONENT_FLOOR: Real = -170.0;
const ATOM_SOLDIR_MATCHING_POINT_TAIL_MARGIN: usize = 10;
const ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET: usize = 12;
const ATOM_SOLDIR_FIXED_MATCHING_RELATIVE_ENERGY: Real = 1.0e-1;

mod breit;
mod form_factor;
mod nuclear;
mod overlap;
mod tables;
mod tabulation;
mod types;

pub use breit::*;
pub use form_factor::*;
pub use nuclear::*;
pub use overlap::*;
pub use tables::*;
pub use tabulation::*;
pub use types::*;

/// Port of FEFF `ATOM/aprdev.f90`.
///
/// `term_count` is FEFF's 1-based `l` argument. The returned value is the
/// coefficient of power `l - 1` in the product of two coefficient rows.
pub fn atomic_polynomial_product_coefficient(
    left: &[Real],
    right: &[Real],
    term_count: usize,
) -> Result<Real, AtomMathError> {
    if term_count == 0 || term_count > left.len() || term_count > right.len() {
        return Err(AtomMathError::InvalidPolynomialTerm {
            term_count,
            left_len: left.len(),
            right_len: right.len(),
        });
    }
    validate_finite_slice("left coefficient", left)?;
    validate_finite_slice("right coefficient", right)?;

    Ok((0..term_count)
        .map(|index| left[index] * right[term_count - 1 - index])
        .sum())
}

/// Port of FEFF `ATOM/cofcon.f90` convergence acceleration.
///
/// FEFF adjusts the final-iteration weight by `0.1` when consecutive errors
/// have the same or opposite sign, then stores the current error as the next
/// previous error.
pub fn atomic_convergence_mix(
    final_weight: Real,
    current_error: Real,
    previous_error: Real,
) -> Result<AtomicConvergenceMix, AtomMathError> {
    validate_finite_scalar("final_weight", final_weight)?;
    validate_finite_scalar("current_error", current_error)?;
    validate_finite_scalar("previous_error", previous_error)?;

    let product = current_error * previous_error;
    validate_finite_scalar("error_product", product)?;

    let mut updated_final_weight = final_weight;
    if product < 0.0 {
        if updated_final_weight >= 0.2 {
            updated_final_weight -= 0.1;
        }
    } else if product > 0.0 && updated_final_weight <= 0.8 {
        updated_final_weight += 0.1;
    }

    Ok(AtomicConvergenceMix {
        initial_weight: 1.0 - updated_final_weight,
        final_weight: updated_final_weight,
        previous_error: current_error,
    })
}

/// Port of FEFF `ATOM/dentfa.f90`, the Thomas-Fermi density approximation.
///
/// `nuclear_charge + ionicity` is the effective electron count used by FEFF.
/// Values below `1e-4` return zero, matching the Fortran early exit.
pub fn thomas_fermi_density_potential(
    radius: Real,
    nuclear_charge: Real,
    ionicity: Real,
) -> Result<Real, AtomMathError> {
    validate_finite_scalar("radius", radius)?;
    validate_finite_scalar("nuclear_charge", nuclear_charge)?;
    validate_finite_scalar("ionicity", ionicity)?;
    if radius <= 0.0 {
        return Err(AtomMathError::NonPositiveRadius { radius });
    }

    let effective_charge = nuclear_charge + ionicity;
    if effective_charge < 1.0e-4 {
        return Ok(0.0);
    }

    let exponent = Real::from(1.0_f32 / 3.0_f32);
    let mut scaled = radius * effective_charge.powf(exponent);
    scaled = (scaled / Real::from(0.8853_f32)).sqrt();
    let numerator = scaled * (Real::from(0.60112_f32) * scaled + Real::from(1.81061_f32)) + 1.0;
    let denominator = scaled
        * (scaled
            * (scaled
                * (scaled * (Real::from(0.04793_f32) * scaled + Real::from(0.21465_f32))
                    + Real::from(0.77112_f32))
                + Real::from(1.39515_f32))
            + Real::from(1.81061_f32))
        + 1.0;
    let value = effective_charge * (1.0 - (numerator / denominator).powi(2)) / radius;
    validate_finite_scalar("dentfa", value)?;
    Ok(value)
}

/// Port of FEFF `ATOM/fdmocc.f90`, the occupation-number product.
///
/// `left` and `right` are zero-based Rust orbital indices. For equal orbitals,
/// FEFF applies the same degeneracy correction using the orbital kappa.
pub fn atomic_occupation_product(
    occupations: &[Real],
    kappas: &[i32],
    left: usize,
    right: usize,
) -> Result<Real, AtomMathError> {
    validate_occupation_tables(occupations, kappas)?;
    validate_orbital_index(left, occupations.len())?;
    validate_orbital_index(right, occupations.len())?;

    if left == right {
        let kappa_abs = kappas[left].unsigned_abs();
        if kappa_abs == 0 {
            return Err(AtomMathError::ZeroKappa);
        }
        let degeneracy = 2.0 * Real::from(kappa_abs);
        Ok(occupations[left] * (occupations[right] - 1.0) * degeneracy / (degeneracy - 1.0))
    } else {
        Ok(occupations[left] * occupations[right])
    }
}

/// Port of FEFF `ATOM/akeato.f90`, direct Coulomb angular coefficient lookup.
///
/// The orbital indices are zero-based. FEFF uses integer division `k / 2`;
/// odd ranks therefore map to the same channel as the preceding even rank.
pub fn atomic_direct_coulomb_coefficient(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    validate_coefficient_table(coefficients, left, right, rank)?;
    direct_coulomb_coefficient_at(coefficients, left, right, rank)
}

/// Port of FEFF `ATOM/akeato.f90::bkeato`, exchange Coulomb coefficient lookup.
///
/// Equal orbitals return zero, matching FEFF's explicit same-index branch.
pub fn atomic_exchange_coulomb_coefficient(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    validate_coefficient_table(coefficients, left, right, rank)?;
    exchange_coulomb_coefficient_at(coefficients, left, right, rank)
}

/// Port of FEFF `ATOM/muatco.f90`, Coulomb angular coefficient tabulation.
///
/// The returned `Array3` is indexed as `(orbital, orbital, rank / 2)` and keeps
/// FEFF's asymmetric storage convention: direct `F^k` coefficients occupy
/// `(min, max, rank / 2)`, while exchange `G^k` coefficients occupy
/// `(max, min, rank / 2)`.
pub fn atomic_coulomb_coefficients(
    input: AtomicCoulombCoefficientInput<'_>,
) -> Result<Array3<Real>, AtomMathError> {
    const CHANNELS: usize = 5;

    validate_coulomb_coefficient_input(&input)?;
    let orbital_count = input.kappas.len();
    let mut coefficients = Array3::<Real>::zeros((orbital_count, orbital_count, CHANNELS).f());

    for left in 0..orbital_count {
        let left_j2 = doubled_j_usize_from_kappa(input.kappas[left])?;
        for right in 0..=left {
            let right_j2 = doubled_j_usize_from_kappa(input.kappas[right])?;
            let max_rank = left_j2
                .checked_add(right_j2)
                .ok_or(AtomMathError::CoulombRankOutOfRange { rank: left_j2 })?
                / 2;
            let mut min_rank = left_j2.abs_diff(right_j2) / 2;
            if input.kappas[left].signum() != input.kappas[right].signum() {
                min_rank = min_rank
                    .checked_add(1)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank: min_rank })?;
            }

            let same_closed_orbital = left == right && input.valence_occupations[left] <= 0.0;
            let same_orbital_correction = if same_closed_orbital { 1.0 } else { 0.0 };
            coefficients[(right, left, 0)] +=
                input.occupations[left] * (input.occupations[right] - same_orbital_correction);

            if input.valence_occupations[left] > 0.0 && input.valence_occupations[right] > 0.0 {
                continue;
            }

            let mut scale = coefficients[(right, left, 0)];
            if same_closed_orbital {
                let left_j2_real = Real::from(atom_usize_to_i32(left_j2)?);
                scale = -scale * (left_j2_real + 1.0) / left_j2_real;
                min_rank = min_rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank: min_rank })?;
            }

            let mut rank = min_rank;
            while rank <= max_rank {
                let channel = rank / 2;
                if channel >= CHANNELS {
                    return Err(AtomMathError::CoefficientChannelOutOfRange {
                        rank,
                        channel,
                        channels: CHANNELS,
                    });
                }
                let doubled_rank = rank
                    .checked_mul(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
                let wigner = wigner_3j(
                    atom_usize_to_i32(left_j2)?,
                    atom_usize_to_i32(doubled_rank)?,
                    atom_usize_to_i32(right_j2)?,
                    1,
                    0,
                    2,
                )
                .map_err(|source| AtomMathError::CoulombAngular { source })?;
                coefficients[(left, right, channel)] = scale * wigner * wigner;
                rank = rank
                    .checked_add(2)
                    .ok_or(AtomMathError::CoulombRankOutOfRange { rank })?;
            }
        }
    }

    Ok(coefficients)
}

/// Port of FEFF `ATOM/inmuat.f90` after the `getorb` occupation compaction.
///
/// The caller supplies compacted orbital quantum numbers and occupations, for
/// example from [`crate::orbital_configuration`]. This routine mirrors the
/// deterministic ATOM setup that follows `getorb`: electron-count checking,
/// convergence defaults, active lengths, open-shell flags, and the Lagrange
/// pair count.
pub fn atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    validate_orbital_initialization_input(&input)?;
    calculate_atomic_orbital_initialization(input)
}

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

/// Port of FEFF `ATOM/vlda.f90`, local-density exchange potential.
///
/// The routine builds FEFF's total and valence spherical densities from orbital
/// components, evaluates the selected `idfock` exchange branch, and returns the
/// updated potential/development/energy arrays.
pub fn atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    validate_local_density_potential_input(&input)?;
    calculate_atomic_local_density_potential(input)
}

/// Port of FEFF `ATOM/potrdf.f90`, orbital potential/source assembly.
pub fn atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    validate_orbital_potential_input(&input)?;
    calculate_atomic_orbital_potential(input)
}

/// Port of FEFF `ATOM/dsordf.f90`.
///
/// This evaluates FEFF's Simpson-rule radial integral and adds the analytic
/// origin-development correction from `aprdev`. The supported modes cover the
/// `jnd = -2, -1, 1, 2, 3, 4` branches used by FEFF10 `ATOM`.
pub fn atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    validate_differential_integral_input(&input)?;
    calculate_atomic_differential_integral(input)
}

/// Port of FEFF `ATOM/yzkteg.f90`.
///
/// This builds the radial `yk` and `zk` Coulomb kernels using FEFF's four-point
/// point-to-point integration stencil and origin-development correction.
pub fn atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_transform_input(&input)?;
    calculate_atomic_yk_zk_transform(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for positive orbital indices.
///
/// This constructs the source and origin coefficients from ATOM orbital
/// component tables, then delegates to [`atomic_yk_zk_transform`].
pub fn atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    validate_yk_zk_exchange_input(&input)?;
    calculate_atomic_yk_zk_exchange(input)
}

/// Port of FEFF `ATOM/yzkrdf.f90` for caller-prepared sources.
///
/// This mirrors the `i <= 0` branch, where the caller supplies the source and
/// coefficients directly and FEFF uses `k + 2` as the first origin power before
/// delegating to `yzkteg`.
pub fn atomic_yk_zk_prepared_source(
    input: AtomicYkZkPreparedSourceInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let initial_power = input.angular_momentum.checked_add(2).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )? as Real;

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: input.source,
        source_coefficients: input.source_coefficients,
        radii: input.radii,
        initial_power,
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count: input.coefficient_count,
        source_len: input.source_len,
        active_len: input.active_len,
    })
}

/// Port of FEFF `ATOM/fdrirk.f90`.
///
/// This composes the `yzkrdf` first-factor construction with the `dsordf`
/// radial integration branch. When the first pair in the request is zero, the
/// caller must pass the previous first factor to mirror FEFF's common-block
/// sentinel path.
pub fn atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    validate_radial_integral_input(&input)?;
    calculate_atomic_radial_integral(input)
}

/// Port of FEFF `ATOM/ortdat.f90`, Schmidt orthogonalization.
///
/// The supplied callback receives FEFF `dsordf`-style projection and norm
/// requests because the original routine depends on ATOM common-block radial
/// integration state. Returned matrices keep the caller's `(row, orbital)`
/// layout and update only FEFF's active rows for each orthogonalized orbital.
pub fn atomic_schmidt_orthogonalization<F>(
    input: AtomicSchmidtOrthogonalizationInput<'_>,
    overlap_integral: F,
) -> Result<AtomicSchmidtOrthogonalization, AtomMathError>
where
    F: for<'request> FnMut(AtomicSchmidtIntegralRequest<'request>) -> Result<Real, AtomMathError>,
{
    validate_schmidt_orthogonalization_input(&input)?;
    AtomicSchmidtContext {
        input,
        overlap_integral,
    }
    .calculate()
}

/// Port of FEFF `ATOM/etotal.f90`, excluding the radial integral solver.
///
/// The supplied callback receives FEFF-style `fdrirk(i,j,l,m,k)` requests and
/// must return the corresponding radial integral. This function performs the
/// FEFF accumulation of direct Coulomb, exchange Coulomb, magnetic Breit,
/// retarded Breit, and one-electron energy terms.
pub fn atomic_total_energy<F>(
    input: AtomicTotalEnergyInput<'_>,
    radial_integral: F,
) -> Result<AtomicTotalEnergy, AtomMathError>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    validate_total_energy_input(&input)?;
    AtomicTotalEnergyContext {
        input,
        radial_integral,
    }
    .calculate()
}

struct AtomicTotalEnergyContext<'a, F> {
    input: AtomicTotalEnergyInput<'a>,
    radial_integral: F,
}

struct AtomicSchmidtContext<'a, F> {
    input: AtomicSchmidtOrthogonalizationInput<'a>,
    overlap_integral: F,
}

struct AtomicSchmidtTables<'a> {
    large_components: &'a mut Array2<Real>,
    small_components: &'a mut Array2<Real>,
    large_coefficients: &'a mut Array2<Real>,
    small_coefficients: &'a mut Array2<Real>,
    active_lengths: &'a mut [usize],
}

struct AtomicSchmidtProjectionInput<'a> {
    target: usize,
    reference: usize,
    active_len: usize,
    work_large: &'a Array1<Real>,
    work_small: &'a Array1<Real>,
    work_large_coefficients: &'a Array1<Real>,
    work_small_coefficients: &'a Array1<Real>,
    large_components: ArrayView2<'a, Real>,
    small_components: ArrayView2<'a, Real>,
    large_coefficients: ArrayView2<'a, Real>,
    small_coefficients: ArrayView2<'a, Real>,
}

fn calculate_atomic_orbital_initialization(
    input: AtomicOrbitalInitializationInput<'_>,
) -> Result<AtomicOrbitalInitialization, AtomMathError> {
    let orbital_count = input.occupations.len();
    let active_lengths =
        Array1::<usize>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_RADIAL_COUNT);
    let mut shell_markers = Array1::<i32>::from_elem(orbital_count, -1);
    let mut convergence_acceleration =
        Array1::<Real>::from_elem(orbital_count, ATOM_INMUAT_DEFAULT_CONVERGENCE);
    let mut lagrange_pair_count = 0usize;

    for orbital in 0..orbital_count {
        let kappa_abs = kappa_abs_usize(input.kappas[orbital])?;
        let angular_momentum = if input.kappas[orbital] < 0 {
            kappa_abs
                .checked_sub(1)
                .ok_or(AtomMathError::InvalidKappa {
                    kappa: input.kappas[orbital],
                })?
        } else {
            kappa_abs
        };
        let principal_quantum_number = input.principal_quantum_numbers[orbital];
        if angular_momentum >= principal_quantum_number || angular_momentum > 4 {
            return Err(AtomMathError::OrbitalAngularMomentumOutOfRange {
                orbital_1based: orbital + 1,
                principal_quantum_number,
                kappa: input.kappas[orbital],
                angular_momentum,
            });
        }

        let closed_shell_capacity = 2.0 * kappa_abs as Real;
        if input.occupations[orbital] < closed_shell_capacity {
            shell_markers[orbital] = 1;
        }
        if input.occupations[orbital] < 0.5 {
            convergence_acceleration[orbital] = 1.0;
        }
        for previous in 0..orbital {
            if input.kappas[previous] == input.kappas[orbital]
                && (shell_markers[previous] > 0 || shell_markers[orbital] > 0)
            {
                lagrange_pair_count = lagrange_pair_count
                    .checked_add(1)
                    .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })?;
            }
        }
    }

    for value in convergence_acceleration.iter().copied() {
        validate_finite_scalar("inmuat_convergence_acceleration", value)?;
    }

    Ok(AtomicOrbitalInitialization {
        orbital_count,
        self_consistent_count: orbital_count,
        wavefunction_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION,
        energy_precision: ATOM_INMUAT_ENERGY_PRECISION,
        precision_ratios: [ATOM_INMUAT_PRIMARY_RATIO, ATOM_INMUAT_SECONDARY_RATIO],
        primary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION / ATOM_INMUAT_PRIMARY_RATIO,
        secondary_matching_precision: ATOM_INMUAT_WAVEFUNCTION_PRECISION
            / ATOM_INMUAT_SECONDARY_RATIO,
        development_order: ATOM_INMUAT_DEVELOPMENT_ORDER,
        attempt_count: ATOM_INMUAT_ATTEMPT_COUNT,
        nucleus_index: ATOM_INMUAT_NUCLEUS_INDEX,
        radial_count: ATOM_INMUAT_DEFAULT_RADIAL_COUNT,
        orbital_energies: Array1::<Real>::zeros(orbital_count),
        convergence_acceleration,
        wavefunction_errors: Array1::<Real>::zeros(orbital_count),
        energy_errors: Array1::<Real>::zeros(orbital_count),
        active_lengths,
        shell_markers,
        lagrange_pair_count,
        lagrange_parameters: Array1::<Real>::zeros(ATOM_INMUAT_LAGRANGE_CAPACITY),
    })
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

fn atom_intdir_decay(energy: Real, speed_of_light: Real) -> Result<Real, AtomMathError> {
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

fn calculate_atomic_local_density_potential(
    input: AtomicLocalDensityPotentialInput<'_>,
) -> Result<AtomicLocalDensityPotential, AtomMathError> {
    let radial_count = input.radii.len();
    let orbital_count = input.active_lengths.len();
    let mut total_density = Array1::<Real>::zeros(radial_count);
    let mut valence_density = Array1::<Real>::zeros(radial_count);

    for orbital in 0..orbital_count {
        for row in 0..input.active_lengths[orbital] {
            let component_density = input.large_components[(row, orbital)].powi(2)
                + input.small_components[(row, orbital)].powi(2);
            total_density[row] += input.occupations[orbital] * component_density;
            valence_density[row] += input.valence_occupations[orbital] * component_density;
        }
    }

    let mut potential = input.initial_potential.to_owned();
    let mut development_coefficients = input.initial_development_coefficients.to_owned();
    let mut energy_density = input.initial_energy_density.to_owned();

    for row in 0..radial_count {
        let radius_squared = input.radii[row] * input.radii[row];
        let density = total_density[row] / radius_squared;
        if density <= 0.0 {
            continue;
        }

        let comparison_density = match input.mode {
            AtomicLocalDensityExchangeMode::ValenceDensity => valence_density[row] / radius_squared,
            AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
                (total_density[row] - valence_density[row]) / radius_squared
            }
            AtomicLocalDensityExchangeMode::DiracFockOnly => 0.0,
            AtomicLocalDensityExchangeMode::TotalDensity => density,
        };
        let vxc = atomic_local_density_vxc(input.mode, density, comparison_density)?;
        if input.accumulate_energy_density {
            energy_density[row] += vxc * total_density[row];
        }
        let scaled_vxc = vxc / input.speed_of_light;
        if row == 0 {
            development_coefficients[1] += scaled_vxc;
        }
        potential[row] += scaled_vxc;
    }

    validate_finite_vector("vlda_total_density", total_density.view())?;
    validate_finite_vector("vlda_valence_density", valence_density.view())?;
    validate_finite_vector("vlda_potential", potential.view())?;
    validate_finite_vector(
        "vlda_development_coefficient",
        development_coefficients.view(),
    )?;
    validate_finite_vector("vlda_energy_density", energy_density.view())?;

    Ok(AtomicLocalDensityPotential {
        total_density,
        valence_density,
        potential,
        development_coefficients,
        energy_density,
    })
}

fn atomic_local_density_vxc(
    mode: AtomicLocalDensityExchangeMode,
    density: Real,
    comparison_density: Real,
) -> Result<Real, AtomMathError> {
    let density_parameter = (density / 3.0).powf(-1.0 / 3.0);
    let comparison_parameter = if comparison_density > 0.0 {
        (comparison_density / 3.0).powf(-1.0 / 3.0)
    } else {
        101.0
    };

    match mode {
        AtomicLocalDensityExchangeMode::DiracFockOnly => Ok(0.0),
        AtomicLocalDensityExchangeMode::TotalDensity
        | AtomicLocalDensityExchangeMode::ValenceDensity => {
            Ok(von_barth_hedin_potential(comparison_parameter, 1.0)?)
        }
        AtomicLocalDensityExchangeMode::CoreDensitySeparated => {
            let total_vbh = von_barth_hedin_potential(density_parameter, 1.0)?;
            let fermi_momentum = FEFF_FERMI_MOMENTUM_FACTOR / density_parameter;
            let dirac_hara = dirac_hara_exchange_potential(comparison_parameter, fermi_momentum)?;
            Ok(total_vbh - dirac_hara)
        }
    }
}

fn calculate_atomic_orbital_potential(
    input: AtomicOrbitalPotentialInput<'_>,
) -> Result<AtomicOrbitalPotential, AtomMathError> {
    let active_orbital =
        one_based_atomic_orbital_index(input.active_orbital_1based, input.active_lengths.len())?;
    let active_occupation =
        validate_positive_occupation("potrdf_active_orbital", active_orbital, input.occupations)?;
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let active_j2 = kappa_angular_rank(input.kappas[active_orbital])?;

    let mut central_development_coefficients = input.nuclear_development_coefficients.to_owned();
    let mut central_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_small_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_large_work = Array1::<Real>::zeros(radial_count);
    let mut lagrange_small_work = Array1::<Real>::zeros(radial_count);
    let mut exchange_large_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut exchange_small_coefficients = Array1::<Real>::zeros(coefficient_count);

    let mut rank = 0usize;
    loop {
        let (source, source_coefficients, source_len) =
            direct_orbital_potential_source(input, active_orbital, active_occupation, rank)?;
        let transform = atomic_yk_zk_prepared_source(AtomicYkZkPreparedSourceInput {
            source: source.view(),
            source_coefficients: source_coefficients.view(),
            radii: input.radii,
            step: input.step,
            angular_momentum: rank,
            coefficient_count,
            source_len,
            active_len: radial_count,
        })?;

        for (coefficient, &value) in transform.yk_coefficients.iter().enumerate() {
            let target = rank
                .checked_add(coefficient)
                .and_then(|value| value.checked_add(3))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            if target < coefficient_count {
                central_development_coefficients[target] -= value;
            }
        }
        for row in 0..radial_count {
            central_work[row] += transform.yk[row];
        }

        rank = rank
            .checked_add(2)
            .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                angular_momentum: rank,
            })?;
        if rank <= coefficient_count {
            central_development_coefficients[rank - 1] += transform.origin_constant;
        }
        if rank >= active_j2 {
            break;
        }
    }

    if input.include_exchange {
        accumulate_orbital_exchange_terms(
            input,
            active_orbital,
            active_occupation,
            active_j2,
            exchange_large_work.view_mut(),
            exchange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }
    if input.include_lagrange {
        accumulate_orbital_lagrange_terms(
            input,
            active_orbital,
            lagrange_large_work.view_mut(),
            lagrange_small_work.view_mut(),
            exchange_large_coefficients.view_mut(),
            exchange_small_coefficients.view_mut(),
        )?;
    }

    for coefficient in 0..coefficient_count {
        central_development_coefficients[coefficient] /= input.speed_of_light;
        exchange_large_coefficients[coefficient] /= input.speed_of_light;
        exchange_small_coefficients[coefficient] /= input.speed_of_light;
    }

    let central_potential = Array1::from_shape_fn(radial_count, |row| {
        (central_work[row] / input.radii[row] + input.nuclear_potential[row]) / input.speed_of_light
    });
    let exchange_large = Array1::from_shape_fn(radial_count, |row| {
        (exchange_large_work[row] + lagrange_large_work[row] * input.radii[row])
            / input.speed_of_light
    });
    let exchange_small = Array1::from_shape_fn(radial_count, |row| {
        (exchange_small_work[row] + lagrange_small_work[row] * input.radii[row])
            / input.speed_of_light
    });

    validate_finite_vector("potrdf_central_potential", central_potential.view())?;
    validate_finite_vector(
        "potrdf_central_development_coefficient",
        central_development_coefficients.view(),
    )?;
    validate_finite_vector("potrdf_exchange_large", exchange_large.view())?;
    validate_finite_vector("potrdf_exchange_small", exchange_small.view())?;
    validate_finite_vector(
        "potrdf_exchange_large_coefficient",
        exchange_large_coefficients.view(),
    )?;
    validate_finite_vector(
        "potrdf_exchange_small_coefficient",
        exchange_small_coefficients.view(),
    )?;

    Ok(AtomicOrbitalPotential {
        central_potential,
        central_development_coefficients,
        exchange_large,
        exchange_small,
        exchange_large_coefficients,
        exchange_small_coefficients,
    })
}

fn direct_orbital_potential_source(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    rank: usize,
) -> Result<(Array1<Real>, Array1<Real>, usize), AtomMathError> {
    let radial_count = input.radii.len();
    let coefficient_count = input.nuclear_development_coefficients.len();
    let mut source = Array1::<Real>::zeros(radial_count);
    let mut source_coefficients = Array1::<Real>::zeros(coefficient_count);
    let mut source_len = 0usize;

    for orbital in 0..input.active_lengths.len() {
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        if rank > orbital_j2 {
            source_len = source_len.max(input.active_lengths[orbital]);
            continue;
        }

        let scale = atomic_direct_coulomb_coefficient(
            input.coulomb_coefficients,
            active_orbital,
            orbital,
            rank,
        )? / active_occupation;
        if scale != 0.0 {
            for row in 0..input.active_lengths[orbital] {
                source[row] += scale
                    * (input.large_components[(row, orbital)].powi(2)
                        + input.small_components[(row, orbital)].powi(2));
            }

            let origin_power_start = kappa_angular_rank(input.kappas[orbital])?
                .checked_add(1)
                .and_then(|value| value.checked_sub(rank))
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
            let max_terms = coefficient_count
                .checked_add(2)
                .and_then(|value| value.checked_sub(origin_power_start))
                .unwrap_or(0);
            if max_terms > 0 {
                let coefficient_scale = scale * input.origin_scales[orbital].powi(2);
                let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                for term in 1..=max_terms {
                    let target_1based = origin_power_start
                        .checked_add(term)
                        .and_then(|value| value.checked_sub(2))
                        .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                            angular_momentum: rank,
                        })?;
                    if target_1based == 0 || target_1based > coefficient_count {
                        continue;
                    }
                    source_coefficients[target_1based - 1] += coefficient_scale
                        * (polynomial_product_coefficient_view(
                            large_coefficients,
                            large_coefficients,
                            term,
                        )? + polynomial_product_coefficient_view(
                            small_coefficients,
                            small_coefficients,
                            term,
                        )?);
                }
            }
        }
        source_len = source_len.max(input.active_lengths[orbital]);
    }

    Ok((source, source_coefficients, source_len))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_orbital_exchange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    active_occupation: Real,
    active_j2: usize,
    mut exchange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    let coefficient_count = input.nuclear_development_coefficients.len();
    for orbital in 0..input.active_lengths.len() {
        if orbital == active_orbital {
            continue;
        }
        let orbital_j2 = kappa_angular_rank(input.kappas[orbital])?;
        let maximum_rank = (orbital_j2 + active_j2) / 2;
        let mut rank = orbital_j2.abs_diff(maximum_rank);
        if input.kappas[orbital].signum() * input.kappas[active_orbital].signum() < 0 {
            rank = rank
                .checked_add(1)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }

        while rank <= maximum_rank {
            let scale = atomic_exchange_coulomb_coefficient(
                input.coulomb_coefficients,
                orbital,
                active_orbital,
                rank,
            )? / active_occupation;
            if scale != 0.0 {
                let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
                    left_orbital_1based: orbital + 1,
                    right_orbital_1based: active_orbital + 1,
                    large_small: false,
                    angular_momentum: rank,
                    step: input.step,
                    radii: input.radii,
                    active_lengths: input.active_lengths,
                    orbital_powers: input.orbital_powers,
                    large_components: input.large_components,
                    small_components: input.small_components,
                    large_coefficients: input.large_coefficients,
                    small_coefficients: input.small_coefficients,
                })?;
                for row in 0..input.active_lengths[orbital] {
                    exchange_large_work[row] +=
                        scale * transform.yk[row] * input.large_components[(row, orbital)];
                    exchange_small_work[row] +=
                        scale * transform.yk[row] * input.small_components[(row, orbital)];
                }

                let orbital_abs = kappa_abs_usize(input.kappas[orbital])?;
                let active_abs = kappa_abs_usize(input.kappas[active_orbital])?;
                let origin_shift = rank
                    .checked_add(1)
                    .and_then(|value| value.checked_add(orbital_abs))
                    .and_then(|value| value.checked_sub(active_abs))
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if origin_shift <= coefficient_count {
                    let origin_scale =
                        scale * transform.origin_constant * input.origin_scales[orbital]
                            / input.origin_scales[active_orbital];
                    for coefficient_1based in origin_shift..=coefficient_count {
                        let source_index = coefficient_1based - origin_shift;
                        exchange_large_coefficients[coefficient_1based - 1] +=
                            input.large_coefficients[(source_index, orbital)] * origin_scale;
                        exchange_small_coefficients[coefficient_1based - 1] +=
                            input.small_coefficients[(source_index, orbital)] * origin_scale;
                    }
                }

                let exchange_start = kappa_angular_rank(input.kappas[orbital])?
                    .checked_add(2)
                    .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                        angular_momentum: rank,
                    })?;
                if exchange_start <= coefficient_count {
                    let large_coefficients = input.large_coefficients.index_axis(Axis(1), orbital);
                    let small_coefficients = input.small_coefficients.index_axis(Axis(1), orbital);
                    let source_scale = scale * input.origin_scales[orbital].powi(2);
                    for coefficient_1based in exchange_start..=coefficient_count {
                        let term = coefficient_1based + 1 - exchange_start;
                        exchange_large_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                large_coefficients,
                                term,
                            )?;
                        exchange_small_coefficients[coefficient_1based - 1] -= source_scale
                            * polynomial_product_coefficient_view(
                                transform.yk_coefficients.view(),
                                small_coefficients,
                                term,
                            )?;
                    }
                }
            }

            rank = rank
                .checked_add(2)
                .ok_or(AtomMathError::YkZkAngularMomentumOutOfRange {
                    angular_momentum: rank,
                })?;
        }
    }
    Ok(())
}

fn accumulate_orbital_lagrange_terms(
    input: AtomicOrbitalPotentialInput<'_>,
    active_orbital: usize,
    mut lagrange_large_work: ndarray::ArrayViewMut1<'_, Real>,
    mut lagrange_small_work: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_large_coefficients: ndarray::ArrayViewMut1<'_, Real>,
    mut exchange_small_coefficients: ndarray::ArrayViewMut1<'_, Real>,
) -> Result<(), AtomMathError> {
    for orbital in 0..input.self_consistent_count {
        if input.kappas[orbital] != input.kappas[active_orbital] || orbital == active_orbital {
            continue;
        }
        if input.shell_markers[orbital] < 0 && input.shell_markers[active_orbital] < 0 {
            continue;
        }

        let packed = packed_orbital_pair_index(orbital, active_orbital)?;
        let scale = input.lagrange_parameters[packed] * input.occupations[orbital];
        for row in 0..input.active_lengths[orbital] {
            lagrange_large_work[row] += scale * input.large_components[(row, orbital)];
            lagrange_small_work[row] += scale * input.small_components[(row, orbital)];
        }
        for coefficient in 0..input.nuclear_development_coefficients.len() {
            exchange_large_coefficients[coefficient] +=
                input.large_coefficients[(coefficient, orbital)] * scale;
            exchange_small_coefficients[coefficient] +=
                input.small_coefficients[(coefficient, orbital)] * scale;
        }
    }
    Ok(())
}

struct AtomicDifferentialIntegralWork {
    values: Vec<Real>,
    coefficients: Vec<Real>,
    origin_power: Real,
}

fn calculate_atomic_differential_integral(
    input: AtomicDifferentialIntegralInput<'_>,
) -> Result<Real, AtomMathError> {
    let work = atomic_differential_integral_work(&input)?;
    integrate_atomic_differential_work(&input, work)
}

fn atomic_differential_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            false,
        ),
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
        } => atomic_component_overlap_integral_work(
            input,
            left_orbital_1based,
            right_orbital_1based,
            multiply_by_derivative,
            true,
        ),
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => atomic_derivative_projection_integral_work(
            input,
            large_orbital_1based,
            small_orbital_1based,
        ),
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            atomic_derivative_norm_integral_work(input, active_len)
        }
    }
}

fn atomic_component_overlap_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    left_orbital_1based: usize,
    right_orbital_1based: usize,
    multiply_by_derivative: bool,
    large_small: bool,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let left = one_based_atomic_orbital_index(left_orbital_1based, input.active_lengths.len())?;
    let right = one_based_atomic_orbital_index(right_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[left].min(input.active_lengths[right]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            let base = if large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            };
            if multiply_by_derivative {
                base * input.derivative_large[row]
            } else {
                base
            }
        })
        .collect::<Vec<_>>();

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let mut coefficients = (1..=coefficient_count)
        .map(|term| {
            if large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut origin_power = input.orbital_powers[left] + input.orbital_powers[right];
    if multiply_by_derivative {
        origin_power += input.origin_power;
        coefficients = (1..=coefficient_count)
            .map(|term| {
                polynomial_product_coefficient_slice_view(
                    &coefficients,
                    input.derivative_large_coefficients,
                    term,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn atomic_derivative_projection_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    large_orbital_1based: usize,
    small_orbital_1based: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    let large_orbital =
        one_based_atomic_orbital_index(large_orbital_1based, input.active_lengths.len())?;
    let small_orbital =
        one_based_atomic_orbital_index(small_orbital_1based, input.active_lengths.len())?;
    let active_len = input.active_lengths[large_orbital].min(input.active_lengths[small_orbital]);
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.large_components[(row, large_orbital)]
                + input.derivative_small[row] * input.small_components[(row, small_orbital)]
        })
        .collect::<Vec<_>>();
    let large_coefficients = input.large_coefficients.index_axis(Axis(1), large_orbital);
    let small_coefficients = input.small_coefficients.index_axis(Axis(1), small_orbital);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.orbital_powers[large_orbital];
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn atomic_derivative_norm_integral_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    active_len: usize,
) -> Result<AtomicDifferentialIntegralWork, AtomMathError> {
    validate_differential_active_len(active_len, input.radii.len())?;

    let values = (0..active_len)
        .map(|row| {
            input.derivative_large[row] * input.derivative_large[row]
                + input.derivative_small[row] * input.derivative_small[row]
        })
        .collect::<Vec<_>>();
    let coefficient_count = input.derivative_large_coefficients.len();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            Ok(polynomial_product_coefficient_view(
                input.derivative_large_coefficients,
                input.derivative_large_coefficients,
                term,
            )? + polynomial_product_coefficient_view(
                input.derivative_small_coefficients,
                input.derivative_small_coefficients,
                term,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let origin_power = input.origin_power + input.origin_power;
    validate_finite_scalar("dsordf_origin_power", origin_power)?;

    Ok(AtomicDifferentialIntegralWork {
        values,
        coefficients,
        origin_power,
    })
}

fn integrate_atomic_differential_work(
    input: &AtomicDifferentialIntegralInput<'_>,
    work: AtomicDifferentialIntegralWork,
) -> Result<Real, AtomMathError> {
    let radial_power = input
        .power
        .checked_add(1)
        .ok_or(AtomMathError::DifferentialIntegralPowerOutOfRange { power: input.power })?;
    let scaled = work
        .values
        .iter()
        .enumerate()
        .map(|(row, &value)| value * input.radii[row].powi(radial_power))
        .collect::<Vec<_>>();
    validate_finite_slice("dsordf_integrand", &scaled)?;

    let active_len = scaled.len();
    let mut integral = 0.0;
    let mut row_1based = 2;
    while row_1based < active_len {
        let row = row_1based - 1;
        integral += scaled[row] + scaled[row] + scaled[row + 1];
        row_1based += 2;
    }
    integral = input.step * (integral + integral + scaled[0] - scaled[active_len - 1]) / 3.0;

    let mut origin_exponent = work.origin_power + Real::from(input.power);
    validate_finite_scalar("dsordf_origin_exponent", origin_exponent)?;
    for coefficient in work.coefficients {
        origin_exponent += 1.0;
        if origin_exponent == 0.0 {
            return Err(AtomMathError::ZeroDifferentialIntegralOriginExponent);
        }
        let correction = coefficient * input.radii[0].powf(origin_exponent) / origin_exponent;
        validate_finite_scalar("dsordf_origin_correction", correction)?;
        integral += correction;
    }
    validate_finite_scalar("dsordf_integral", integral)?;
    Ok(integral)
}

fn polynomial_product_coefficient_view(
    left: ArrayView1<'_, Real>,
    right: ArrayView1<'_, Real>,
    term_count: usize,
) -> Result<Real, AtomMathError> {
    polynomial_product_coefficient_indexed(
        term_count,
        left.len(),
        right.len(),
        |index| left[index],
        |index| right[index],
    )
}

fn polynomial_product_coefficient_slice_view(
    left: &[Real],
    right: ArrayView1<'_, Real>,
    term_count: usize,
) -> Result<Real, AtomMathError> {
    polynomial_product_coefficient_indexed(
        term_count,
        left.len(),
        right.len(),
        |index| left[index],
        |index| right[index],
    )
}

fn polynomial_product_coefficient_indexed(
    term_count: usize,
    left_len: usize,
    right_len: usize,
    mut left_at: impl FnMut(usize) -> Real,
    mut right_at: impl FnMut(usize) -> Real,
) -> Result<Real, AtomMathError> {
    if term_count == 0 || term_count > left_len || term_count > right_len {
        return Err(AtomMathError::InvalidPolynomialTerm {
            term_count,
            left_len,
            right_len,
        });
    }
    Ok((0..term_count)
        .map(|index| left_at(index) * right_at(term_count - 1 - index))
        .sum())
}

fn one_based_atomic_orbital_index(
    orbital_1based: usize,
    orbital_count: usize,
) -> Result<usize, AtomMathError> {
    if (1..=orbital_count).contains(&orbital_1based) {
        Ok(orbital_1based - 1)
    } else {
        Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: orbital_1based,
            orbital_count,
        })
    }
}

fn calculate_atomic_yk_zk_transform(
    input: AtomicYkZkTransformInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let source_len = input.source_len.min(input.active_len - 2);
    let k_i32 = i32::try_from(input.angular_momentum).map_err(|_| {
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        }
    })?;
    let k_plus_one = input.angular_momentum.checked_add(1).ok_or(
        AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        },
    )?;
    let k_plus_one_i32 =
        i32::try_from(k_plus_one).map_err(|_| AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        })?;
    let k_real = input.angular_momentum as Real;
    let order = (input.angular_momentum + input.angular_momentum + 1) as Real;

    let mut yk = Array1::<Real>::zeros(input.active_len);
    let mut zk = Array1::<Real>::zeros(input.active_len);
    let mut yk_coefficients = Array1::from_iter(
        (0..input.coefficient_count).map(|coefficient| input.source_coefficients[coefficient]),
    );
    let mut zk_coefficients = Array1::<Real>::zeros(input.coefficient_count);

    let mut power = input.initial_power;
    let mut origin_constant = 0.0;
    for coefficient in 0..input.coefficient_count {
        power += 1.0;
        let zk_denominator = power + k_real;
        validate_yk_zk_denominator("zk_origin", zk_denominator)?;
        zk_coefficients[coefficient] = yk_coefficients[coefficient] / zk_denominator;

        if yk_coefficients[coefficient] != 0.0 {
            let radius_power = input.radii[0].powf(power);
            zk[0] += zk_coefficients[coefficient] * radius_power;
            zk[1] += zk_coefficients[coefficient] * input.radii[1].powf(power);

            let yk_denominator = power - k_real - 1.0;
            validate_yk_zk_denominator("yk_origin", yk_denominator)?;
            yk_coefficients[coefficient] = order * zk_coefficients[coefficient] / yk_denominator;
            origin_constant += yk_coefficients[coefficient] * radius_power;
        }
    }

    for row in 0..source_len {
        yk[row] = input.source[row] * input.radii[row];
    }
    yk[source_len] = 0.0;
    yk[source_len + 1] = 0.0;

    let step_exp = input.step.exp();
    let attenuation = step_exp.powi(-k_i32);
    let base_weight = input.step / 24.0;
    let middle_weight = 13.0 * base_weight;
    let leading_weight = attenuation * attenuation * base_weight;
    let trailing_weight = base_weight / attenuation;

    for row_1based in 3..=(source_len + 1) {
        let row = row_1based - 1;
        zk[row] = zk[row - 1] * attenuation
            + (middle_weight * (yk[row] + yk[row - 1] * attenuation)
                - (yk[row - 2] * leading_weight + yk[row + 1] * trailing_weight));
    }

    yk[source_len - 1] = zk[source_len - 1];
    for row in source_len..input.active_len {
        yk[row] = yk[row - 1] * attenuation;
    }

    let backward_trailing_weight = order * trailing_weight * step_exp;
    let backward_leading_weight = order * leading_weight / (step_exp * step_exp);
    let backward_attenuation = attenuation / step_exp;
    let backward_middle_weight = order * middle_weight;
    for row_1based in (2..source_len).rev() {
        let row = row_1based - 1;
        yk[row] = yk[row + 1] * backward_attenuation
            + (backward_middle_weight * (zk[row] + zk[row + 1] * backward_attenuation)
                - (zk[row + 2] * backward_leading_weight + zk[row - 1] * backward_trailing_weight));
    }

    let attenuation_squared = backward_attenuation * backward_attenuation;
    let first_weight = 8.0 * backward_middle_weight / 13.0;
    yk[0] = yk[2] * attenuation_squared
        + first_weight * (zk[2] * attenuation_squared + 4.0 * backward_attenuation * zk[1] + zk[0]);
    origin_constant = (origin_constant + yk[0]) / input.radii[0].powi(k_plus_one_i32);

    validate_finite_scalar("yk_zk_origin_constant", origin_constant)?;
    for row in 0..input.active_len {
        validate_finite_scalar("yk", yk[row])?;
        validate_finite_scalar("zk", zk[row])?;
    }
    for coefficient in 0..input.coefficient_count {
        validate_finite_scalar("yk_coefficient", yk_coefficients[coefficient])?;
        validate_finite_scalar("zk_coefficient", zk_coefficients[coefficient])?;
    }

    Ok(AtomicYkZkTransform {
        yk,
        zk,
        yk_coefficients,
        zk_coefficients,
        origin_constant,
        computed_source_len: source_len,
    })
}

fn validate_yk_zk_denominator(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    if value == 0.0 {
        Err(AtomMathError::ZeroYkZkDenominator { field })
    } else {
        validate_finite_scalar(field, value)
    }
}

fn calculate_atomic_yk_zk_exchange(
    input: AtomicYkZkExchangeInput<'_>,
) -> Result<AtomicYkZkTransform, AtomMathError> {
    let left =
        one_based_atomic_orbital_index(input.left_orbital_1based, input.active_lengths.len())?;
    let right =
        one_based_atomic_orbital_index(input.right_orbital_1based, input.active_lengths.len())?;
    let source_len = input.active_lengths[left].min(input.active_lengths[right]);
    let source = Array1::from_shape_fn(input.radii.len(), |row| {
        if row < source_len {
            if input.large_small {
                input.large_components[(row, left)] * input.small_components[(row, right)]
            } else {
                input.large_components[(row, left)] * input.large_components[(row, right)]
                    + input.small_components[(row, left)] * input.small_components[(row, right)]
            }
        } else {
            0.0
        }
    });

    let left_large_coefficients = input.large_coefficients.index_axis(Axis(1), left);
    let right_large_coefficients = input.large_coefficients.index_axis(Axis(1), right);
    let left_small_coefficients = input.small_coefficients.index_axis(Axis(1), left);
    let right_small_coefficients = input.small_coefficients.index_axis(Axis(1), right);
    let coefficient_count = input.large_coefficients.nrows();
    let coefficients = (1..=coefficient_count)
        .map(|term| -> Result<Real, AtomMathError> {
            if input.large_small {
                polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_small_coefficients,
                    term,
                )
            } else {
                Ok(polynomial_product_coefficient_view(
                    left_large_coefficients,
                    right_large_coefficients,
                    term,
                )? + polynomial_product_coefficient_view(
                    left_small_coefficients,
                    right_small_coefficients,
                    term,
                )?)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_coefficients = Array1::from_vec(coefficients);

    atomic_yk_zk_transform(AtomicYkZkTransformInput {
        source: source.view(),
        source_coefficients: source_coefficients.view(),
        radii: input.radii,
        initial_power: input.orbital_powers[left] + input.orbital_powers[right],
        step: input.step,
        angular_momentum: input.angular_momentum,
        coefficient_count,
        source_len,
        active_len: input.radii.len(),
    })
}

fn calculate_atomic_radial_integral(
    input: AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialIntegral, AtomMathError> {
    let first_factor = if input.request.first_left > 0 && input.request.first_right > 0 {
        Some(calculate_atomic_radial_first_factor(&input)?)
    } else {
        None
    };

    if input.request.second_left == 0 || input.request.second_right == 0 {
        return Ok(AtomicRadialIntegral {
            value: 0.0,
            first_factor,
        });
    }

    let value = match first_factor.as_ref() {
        Some(first_factor) => {
            calculate_atomic_radial_second_integral(&input, first_factor.as_view())?
        }
        None => calculate_atomic_radial_second_integral(
            &input,
            input
                .previous_first_factor
                .ok_or(AtomMathError::MissingRadialFirstFactor)?,
        )?,
    };

    Ok(AtomicRadialIntegral {
        value,
        first_factor,
    })
}

fn calculate_atomic_radial_second_integral(
    input: &AtomicRadialIntegralInput<'_>,
    first_factor: AtomicRadialFirstFactorView<'_>,
) -> Result<Real, AtomMathError> {
    let zero_derivative = Array1::<Real>::zeros(input.radii.len());
    let zero_coefficients = Array1::<Real>::zeros(input.large_coefficients.nrows());
    let kind = if input.large_small {
        AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    } else {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based: input.request.second_left,
            right_orbital_1based: input.request.second_right,
            multiply_by_derivative: true,
        }
    };
    atomic_differential_integral(AtomicDifferentialIntegralInput {
        kind,
        power: -1,
        origin_power: first_factor.origin_power,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
        derivative_large: first_factor.values,
        derivative_small: zero_derivative.view(),
        derivative_large_coefficients: first_factor.coefficients,
        derivative_small_coefficients: zero_coefficients.view(),
    })
}

fn calculate_atomic_radial_first_factor(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<AtomicRadialFirstFactor, AtomMathError> {
    let transform = atomic_yk_zk_exchange(AtomicYkZkExchangeInput {
        left_orbital_1based: input.request.first_left,
        right_orbital_1based: input.request.first_right,
        large_small: input.large_small,
        angular_momentum: input.request.rank,
        step: input.step,
        radii: input.radii,
        active_lengths: input.active_lengths,
        orbital_powers: input.orbital_powers,
        large_components: input.large_components,
        small_components: input.small_components,
        large_coefficients: input.large_coefficients,
        small_coefficients: input.small_coefficients,
    })?;

    let left = one_based_atomic_orbital_index(input.request.first_left, input.kappas.len())?;
    let right = one_based_atomic_orbital_index(input.request.first_right, input.kappas.len())?;
    let left_abs = abs_kappa_i32(input.kappas[left])? as usize;
    let right_abs = abs_kappa_i32(input.kappas[right])? as usize;
    let abs_sum = left_abs
        .checked_add(right_abs)
        .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
    let shifted_start = abs_sum.saturating_sub(input.request.rank).max(1);

    let coefficient_count = transform.yk_coefficients.len();
    let mut coefficients = Array1::<Real>::zeros(coefficient_count);
    for (source, coefficient) in transform.yk_coefficients.iter().copied().enumerate() {
        let target_1based = shifted_start
            .checked_add(source)
            .ok_or(AtomMathError::RadialIntegralIndexOutOfRange)?;
        if target_1based <= coefficient_count {
            coefficients[target_1based - 1] = -coefficient;
        }
    }
    coefficients[0] += transform.origin_constant;

    let origin_power = (input.request.rank as Real) + 1.0;
    validate_finite_scalar("radial_first_factor_origin_power", origin_power)?;
    validate_finite_vector("radial_first_factor", transform.yk.view())?;
    validate_finite_vector("radial_first_factor_coefficient", coefficients.view())?;

    Ok(AtomicRadialFirstFactor {
        values: transform.yk,
        coefficients,
        origin_power,
    })
}

impl<F> AtomicSchmidtContext<'_, F>
where
    F: for<'request> FnMut(AtomicSchmidtIntegralRequest<'request>) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicSchmidtOrthogonalization, AtomMathError> {
        let mut large_components = self.input.large_components.to_owned();
        let mut small_components = self.input.small_components.to_owned();
        let mut large_coefficients = self.input.large_coefficients.to_owned();
        let mut small_coefficients = self.input.small_coefficients.to_owned();
        let mut active_lengths = self.input.active_lengths.to_vec();

        {
            let mut tables = AtomicSchmidtTables {
                large_components: &mut large_components,
                small_components: &mut small_components,
                large_coefficients: &mut large_coefficients,
                small_coefficients: &mut small_coefficients,
                active_lengths: &mut active_lengths,
            };

            if let Some(active_orbital_1based) = self.input.active_orbital_1based {
                let target = active_orbital_1based - 1;
                self.orthogonalize_orbital(target, self.input.kappas.len(), &mut tables)?;
            } else {
                for target in 1..self.input.kappas.len() {
                    self.orthogonalize_orbital(target, target, &mut tables)?;
                }
            }
        }

        Ok(AtomicSchmidtOrthogonalization {
            large_components,
            small_components,
            large_coefficients,
            small_coefficients,
            active_lengths,
        })
    }

    fn orthogonalize_orbital(
        &mut self,
        target: usize,
        reference_limit: usize,
        tables: &mut AtomicSchmidtTables<'_>,
    ) -> Result<(), AtomMathError> {
        let radial_rows = tables.large_components.nrows();
        let coefficient_rows = tables.large_coefficients.nrows();
        let mut active_len = tables.active_lengths[target];
        let mut work_large = Array1::<Real>::zeros(radial_rows);
        let mut work_small = Array1::<Real>::zeros(radial_rows);
        let mut work_large_coefficients = tables
            .large_coefficients
            .index_axis(Axis(1), target)
            .to_owned();
        let mut work_small_coefficients = tables
            .small_coefficients
            .index_axis(Axis(1), target)
            .to_owned();

        for row in 0..active_len {
            work_large[row] = tables.large_components[(row, target)];
            work_small[row] = tables.small_components[(row, target)];
        }

        for reference in 0..reference_limit {
            if reference == target || self.input.kappas[reference] != self.input.kappas[target] {
                continue;
            }
            let reference_len = tables.active_lengths[reference];
            let projection = self.projection(AtomicSchmidtProjectionInput {
                target,
                reference,
                active_len: reference_len,
                work_large: &work_large,
                work_small: &work_small,
                work_large_coefficients: &work_large_coefficients,
                work_small_coefficients: &work_small_coefficients,
                large_components: tables.large_components.view(),
                small_components: tables.small_components.view(),
                large_coefficients: tables.large_coefficients.view(),
                small_coefficients: tables.small_coefficients.view(),
            })?;

            for row in 0..reference_len {
                work_large[row] -= projection * tables.large_components[(row, reference)];
                work_small[row] -= projection * tables.small_components[(row, reference)];
            }
            for coefficient in 0..coefficient_rows {
                work_large_coefficients[coefficient] -=
                    projection * tables.large_coefficients[(coefficient, reference)];
                work_small_coefficients[coefficient] -=
                    projection * tables.small_coefficients[(coefficient, reference)];
            }
            active_len = active_len.max(reference_len);
        }

        tables.active_lengths[target] = active_len;
        let norm = self.norm(
            target,
            active_len,
            &work_large,
            &work_small,
            &work_large_coefficients,
            &work_small_coefficients,
        )?;
        if !norm.is_finite() || norm <= 0.0 {
            return Err(AtomMathError::NonPositiveNorm {
                orbital_1based: target + 1,
                norm,
            });
        }
        let scale = norm.sqrt();
        validate_finite_scalar("schmidt_norm_scale", scale)?;

        for row in 0..active_len {
            tables.large_components[(row, target)] = work_large[row] / scale;
            tables.small_components[(row, target)] = work_small[row] / scale;
            validate_finite_scalar(
                "schmidt_large_component",
                tables.large_components[(row, target)],
            )?;
            validate_finite_scalar(
                "schmidt_small_component",
                tables.small_components[(row, target)],
            )?;
        }
        for coefficient in 0..coefficient_rows {
            tables.large_coefficients[(coefficient, target)] =
                work_large_coefficients[coefficient] / scale;
            tables.small_coefficients[(coefficient, target)] =
                work_small_coefficients[coefficient] / scale;
            validate_finite_scalar(
                "schmidt_large_coefficient",
                tables.large_coefficients[(coefficient, target)],
            )?;
            validate_finite_scalar(
                "schmidt_small_coefficient",
                tables.small_coefficients[(coefficient, target)],
            )?;
        }
        Ok(())
    }

    fn projection(
        &mut self,
        input: AtomicSchmidtProjectionInput<'_>,
    ) -> Result<Real, AtomMathError> {
        let value = {
            let work_large_view = input.work_large.view();
            let work_small_view = input.work_small.view();
            let reference_large_column =
                input.large_components.index_axis(Axis(1), input.reference);
            let reference_small_column =
                input.small_components.index_axis(Axis(1), input.reference);
            let target_large = work_large_view.slice_axis(Axis(0), Slice::from(..input.active_len));
            let target_small = work_small_view.slice_axis(Axis(0), Slice::from(..input.active_len));
            let reference_large =
                reference_large_column.slice_axis(Axis(0), Slice::from(..input.active_len));
            let reference_small =
                reference_small_column.slice_axis(Axis(0), Slice::from(..input.active_len));
            let request = AtomicSchmidtProjectionRequest {
                target_orbital: input.target,
                reference_orbital: input.reference,
                target_power: self.input.orbital_powers[input.target],
                target_large,
                target_small,
                target_large_coefficients: input.work_large_coefficients.view(),
                target_small_coefficients: input.work_small_coefficients.view(),
                reference_large,
                reference_small,
                reference_large_coefficients: input
                    .large_coefficients
                    .index_axis(Axis(1), input.reference),
                reference_small_coefficients: input
                    .small_coefficients
                    .index_axis(Axis(1), input.reference),
            };
            (self.overlap_integral)(AtomicSchmidtIntegralRequest::Projection(request))?
        };
        validate_finite_scalar("schmidt_projection", value)?;
        Ok(value)
    }

    fn norm(
        &mut self,
        target: usize,
        active_len: usize,
        work_large: &Array1<Real>,
        work_small: &Array1<Real>,
        work_large_coefficients: &Array1<Real>,
        work_small_coefficients: &Array1<Real>,
    ) -> Result<Real, AtomMathError> {
        let value = {
            let work_large_view = work_large.view();
            let work_small_view = work_small.view();
            let target_large = work_large_view.slice_axis(Axis(0), Slice::from(..active_len));
            let target_small = work_small_view.slice_axis(Axis(0), Slice::from(..active_len));
            let request = AtomicSchmidtNormRequest {
                target_orbital: target,
                active_len,
                target_power: self.input.orbital_powers[target],
                target_large,
                target_small,
                target_large_coefficients: work_large_coefficients.view(),
                target_small_coefficients: work_small_coefficients.view(),
            };
            (self.overlap_integral)(AtomicSchmidtIntegralRequest::Norm(request))?
        };
        validate_finite_scalar("schmidt_norm", value)?;
        Ok(value)
    }
}

impl<F> AtomicTotalEnergyContext<'_, F>
where
    F: FnMut(AtomicRadialIntegralRequest) -> Result<Real, AtomMathError>,
{
    fn calculate(&mut self) -> Result<AtomicTotalEnergy, AtomMathError> {
        let direct_coulomb = self.direct_coulomb_energy()?;
        let exchange_coulomb = self.exchange_coulomb_energy()?;
        let (magnetic_breit, retarded_breit) = self.breit_energies()?;
        let orbital_energy = self
            .input
            .orbital_energies
            .iter()
            .zip(self.input.occupations)
            .map(|(&energy, &occupation)| energy * occupation)
            .sum::<Real>();
        let total =
            -(direct_coulomb + exchange_coulomb) + magnetic_breit + retarded_breit + orbital_energy;

        Ok(AtomicTotalEnergy {
            total,
            direct_coulomb,
            exchange_coulomb,
            magnetic_breit,
            retarded_breit,
        })
    }

    fn direct_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 0..self.orbital_count() {
            let left_l = self.abs_kappa(left)? - 1;
            for right in 0..=left {
                let symmetry_weight = if right == left { 2.0 } else { 1.0 };
                let right_l = self.abs_kappa(right)? - 1;
                let max_rank = 2 * left_l.min(right_l);
                let mut rank = 0;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, left + 1, right + 1, right + 1, rank)?;
                    energy +=
                        radial * self.direct_coefficient(left, right, rank)? / symmetry_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn exchange_coulomb_energy(&mut self) -> Result<Real, AtomMathError> {
        let mut energy = 0.0;
        for left in 1..self.orbital_count() {
            let valence_weight = if self.input.valence_occupations[left] > 0.0 {
                0.5
            } else {
                1.0
            };
            for right in 0..left {
                if self.input.valence_occupations[right] > 0.0 {
                    continue;
                }
                let left_abs = self.abs_kappa(left)?;
                let right_abs = self.abs_kappa(right)?;
                let mut rank = left_abs.abs_diff(right_abs);
                if self.kappa(left).signum() != self.kappa(right).signum() {
                    rank += 1;
                }
                let max_rank = left_abs + right_abs - 1;
                while rank <= max_rank {
                    let radial = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
                    energy -=
                        radial * self.exchange_coefficient(left, right, rank)? * valence_weight;
                    rank += 2;
                }
            }
        }
        Ok(energy)
    }

    fn breit_energies(&mut self) -> Result<(Real, Real), AtomMathError> {
        let mut magnetic = 0.0;
        let mut retarded = 0.0;
        for right in 0..self.orbital_count() {
            let right_j2 = self.j2(right)?;
            for left in 0..=right {
                let left_j2 = self.j2(left)?;
                let max_rank = left_j2.min(right_j2);
                let mut rank = 1;
                while rank <= max_rank {
                    let radial = self.radial(right + 1, right + 1, left + 1, left + 1, rank)?;
                    if left == right {
                        let coefficients = atomic_breit_angular_coefficients(
                            self.kappa(right),
                            self.kappa(right),
                            rank,
                        )?;
                        let occupation = atomic_occupation_product(
                            self.input.occupations,
                            self.input.kappas,
                            right,
                            right,
                        )?;
                        magnetic +=
                            coefficients.magnetic.iter().sum::<Real>() * radial * occupation / 2.0;
                    }
                    rank += 2;
                }
            }
        }

        for right in 1..self.orbital_count() {
            let right_branch = self.exchange_breit_branch(right)?;
            for left in 0..right {
                let left_branch = self.exchange_breit_branch(left)?;
                let occupation = atomic_occupation_product(
                    self.input.occupations,
                    self.input.kappas,
                    right,
                    left,
                )?;
                let mut rank = left_branch.minimum_rank(right_branch)?;
                let max_rank = left_branch.maximum_rank(right_branch)?;
                let parity_sum = i32::try_from(rank)
                    .map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?
                    .checked_add(left_branch.angular_l)
                    .and_then(|value| value.checked_add(right_branch.angular_l))
                    .ok_or(AtomMathError::BreitBranchOutOfRange)?;
                if parity_sum % 2 == 0 {
                    rank += 1;
                }
                let kappa_sum = self.abs_kappa(right)? + self.abs_kappa(left)?;
                while rank <= max_rank {
                    let coefficients = atomic_breit_angular_coefficients(
                        self.kappa(right),
                        self.kappa(left),
                        rank,
                    )?;
                    let radials = self.exchange_breit_radials(left, right, rank, kappa_sum)?;
                    magnetic += coefficients
                        .magnetic
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    retarded += coefficients
                        .retarded
                        .iter()
                        .zip(radials)
                        .map(|(&coefficient, radial)| coefficient * radial * occupation)
                        .sum::<Real>();
                    rank += 2;
                }
            }
        }

        Ok((magnetic, retarded))
    }

    fn exchange_breit_radials(
        &mut self,
        left: usize,
        right: usize,
        rank: usize,
        kappa_sum: usize,
    ) -> Result<[Real; 3], AtomMathError> {
        let mut radials = [0.0; 3];
        if !(kappa_sum <= rank && self.kappa(left) < 0 && self.kappa(right) > 0) {
            radials[0] = self.radial(left + 1, right + 1, left + 1, right + 1, rank)?;
            radials[1] = self.radial(0, 0, right + 1, left + 1, rank)?;
        }
        if !(kappa_sum <= rank && self.kappa(left) > 0 && self.kappa(right) < 0) {
            radials[2] = self.radial(right + 1, left + 1, right + 1, left + 1, rank)?;
            if radials[1] == 0.0 {
                radials[1] = self.radial(0, 0, left + 1, right + 1, rank)?;
            }
        }
        Ok(radials)
    }

    fn radial(
        &mut self,
        first_left: usize,
        first_right: usize,
        second_left: usize,
        second_right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let request = AtomicRadialIntegralRequest {
            first_left,
            first_right,
            second_left,
            second_right,
            rank,
        };
        let value = (self.radial_integral)(request)?;
        validate_finite_scalar("radial_integral", value)?;
        Ok(value)
    }

    fn direct_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left <= right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        }
    }

    fn exchange_coefficient(
        &self,
        left: usize,
        right: usize,
        rank: usize,
    ) -> Result<Real, AtomMathError> {
        let channel = self.coefficient_channel(rank)?;
        if left < right {
            Ok(self.input.coulomb_coefficients[(right, left, channel)])
        } else if left > right {
            Ok(self.input.coulomb_coefficients[(left, right, channel)])
        } else {
            Ok(0.0)
        }
    }

    fn coefficient_channel(&self, rank: usize) -> Result<usize, AtomMathError> {
        let channel = rank / 2;
        let channels = self.input.coulomb_coefficients.shape()[2];
        if channel >= channels {
            Err(AtomMathError::CoefficientChannelOutOfRange {
                rank,
                channel,
                channels,
            })
        } else {
            Ok(channel)
        }
    }

    fn exchange_breit_branch(&self, orbital: usize) -> Result<BreitExchangeBranch, AtomMathError> {
        let mut angular_l = abs_kappa_i32(self.kappa(orbital))?;
        let mut sign_shift = -1;
        if self.kappa(orbital) < 0 {
            sign_shift = 1;
            angular_l -= 1;
        }
        Ok(BreitExchangeBranch {
            angular_l,
            sign_shift,
        })
    }

    fn orbital_count(&self) -> usize {
        self.input.kappas.len()
    }

    fn kappa(&self, orbital: usize) -> i32 {
        self.input.kappas[orbital]
    }

    fn abs_kappa(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(abs_kappa_i32(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }

    fn j2(&self, orbital: usize) -> Result<usize, AtomMathError> {
        usize::try_from(doubled_j_from_kappa(self.kappa(orbital))?).map_err(|_| {
            AtomMathError::InvalidKappa {
                kappa: self.kappa(orbital),
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct BreitExchangeBranch {
    angular_l: i32,
    sign_shift: i32,
}

impl BreitExchangeBranch {
    fn minimum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = other
            .angular_l
            .checked_add(other.sign_shift)
            .and_then(|value| value.checked_sub(self.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        let second = self
            .angular_l
            .checked_add(self.sign_shift)
            .and_then(|value| value.checked_sub(other.angular_l))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?
            .unsigned_abs();
        usize::try_from(first.min(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }

    fn maximum_rank(self, other: Self) -> Result<usize, AtomMathError> {
        let first = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(other.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        let second = self
            .angular_l
            .checked_add(other.angular_l)
            .and_then(|value| value.checked_add(self.sign_shift))
            .ok_or(AtomMathError::BreitBranchOutOfRange)?;
        usize::try_from(first.max(second)).map_err(|_| AtomMathError::BreitBranchOutOfRange)
    }
}

fn doubled_j_from_kappa(kappa: i32) -> Result<i32, AtomMathError> {
    abs_kappa_i32(kappa)?
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

fn doubled_j_usize_from_kappa(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(doubled_j_from_kappa(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

fn kappa_angular_rank(kappa: i32) -> Result<usize, AtomMathError> {
    doubled_j_usize_from_kappa(kappa)
}

fn kappa_abs_usize(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(abs_kappa_i32(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

fn atom_usize_to_i32(value: usize) -> Result<i32, AtomMathError> {
    i32::try_from(value).map_err(|_| AtomMathError::CoulombRankOutOfRange { rank: value })
}

fn direct_coulomb_coefficient_at(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    let channel = coefficient_channel(coefficients, rank)?;
    if left <= right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(coefficients[(right, left, channel)])
    }
}

fn exchange_coulomb_coefficient_at(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<Real, AtomMathError> {
    let channel = coefficient_channel(coefficients, rank)?;
    if left < right {
        Ok(coefficients[(right, left, channel)])
    } else if left > right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(0.0)
    }
}

fn coefficient_channel(
    coefficients: ArrayView3<'_, Real>,
    rank: usize,
) -> Result<usize, AtomMathError> {
    let channel = rank / 2;
    let channels = coefficients.shape()[2];
    if channel >= channels {
        Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        })
    } else {
        Ok(channel)
    }
}

fn orbital_pair_count(orbital_count: usize) -> Result<usize, AtomMathError> {
    orbital_count
        .checked_mul(orbital_count.saturating_sub(1))
        .map(|count| count / 2)
        .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })
}

fn packed_orbital_pair_index(first: usize, second: usize) -> Result<usize, AtomMathError> {
    let lower = first.min(second);
    let upper = first.max(second);
    upper
        .checked_mul(upper.saturating_sub(1))
        .map(|value| value / 2)
        .and_then(|value| value.checked_add(lower))
        .ok_or(AtomMathError::OrbitalPairTableTooLarge {
            orbital_count: upper.saturating_add(1),
        })
}

fn abs_kappa_i32(kappa: i32) -> Result<i32, AtomMathError> {
    if kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa });
    }
    kappa
        .checked_abs()
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

fn validate_total_energy_input(input: &AtomicTotalEnergyInput<'_>) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_orbital_table_len(
        "orbital_energies",
        orbital_count,
        input.orbital_energies.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_slice("orbital_energy", input.orbital_energies)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_coefficient_table(
        input.coulomb_coefficients,
        orbital_count - 1,
        orbital_count - 1,
        0,
    )
}

fn validate_differential_integral_input(
    input: &AtomicDifferentialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("dsordf_step", input.step)?;
    validate_finite_scalar("dsordf_origin_power", input.origin_power)?;
    if input.radii.is_empty() {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: 0,
            radial_count: 0,
        });
    }
    validate_positive_finite_radii(input.radii)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;

    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_radial_table_len(
        "derivative_large",
        input.radii.len(),
        input.derivative_large.len(),
    )?;
    validate_radial_table_len(
        "derivative_small",
        input.radii.len(),
        input.derivative_small.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_large_coefficients",
        coefficient_count,
        input.derivative_large_coefficients.len(),
    )?;
    validate_coefficient_vector_len(
        "derivative_small_coefficients",
        coefficient_count,
        input.derivative_small_coefficients.len(),
    )?;

    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("derivative_large", input.derivative_large)?;
    validate_finite_vector("derivative_small", input.derivative_small)?;
    validate_finite_vector(
        "derivative_large_coefficient",
        input.derivative_large_coefficients,
    )?;
    validate_finite_vector(
        "derivative_small_coefficient",
        input.derivative_small_coefficients,
    )?;

    match input.kind {
        AtomicDifferentialIntegralKind::ComponentOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        }
        | AtomicDifferentialIntegralKind::LargeSmallOverlap {
            left_orbital_1based,
            right_orbital_1based,
            ..
        } => {
            let left = one_based_atomic_orbital_index(left_orbital_1based, orbital_count)?;
            let right = one_based_atomic_orbital_index(right_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[left].min(input.active_lengths[right]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeProjection {
            large_orbital_1based,
            small_orbital_1based,
        } => {
            let large = one_based_atomic_orbital_index(large_orbital_1based, orbital_count)?;
            let small = one_based_atomic_orbital_index(small_orbital_1based, orbital_count)?;
            validate_differential_active_len(
                input.active_lengths[large].min(input.active_lengths[small]),
                input.radii.len(),
            )?;
        }
        AtomicDifferentialIntegralKind::DerivativeNorm { active_len } => {
            validate_differential_active_len(active_len, input.radii.len())?;
        }
    }
    Ok(())
}

fn validate_yk_zk_transform_input(
    input: &AtomicYkZkTransformInput<'_>,
) -> Result<(), AtomMathError> {
    if input.active_len < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.active_len,
            radial_count: input.active_len,
        });
    }
    if input.source_len < 2 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: input.source_len,
            radial_count: input.active_len,
        });
    }
    validate_coefficient_count("source_coefficients", input.coefficient_count)?;
    validate_radial_table_len("source", input.active_len, input.source.len())?;
    validate_radial_table_len("radii", input.active_len, input.radii.len())?;
    validate_coefficient_vector_len(
        "source_coefficients",
        input.coefficient_count,
        input.source_coefficients.len(),
    )?;
    validate_finite_scalar("yk_zk_initial_power", input.initial_power)?;
    validate_finite_scalar("yk_zk_step", input.step)?;
    if input.step == 0.0 {
        return Err(AtomMathError::ZeroYkZkDenominator { field: "step" });
    }
    if input.angular_momentum > i32::MAX as usize {
        return Err(AtomMathError::YkZkAngularMomentumOutOfRange {
            angular_momentum: input.angular_momentum,
        });
    }
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("yk_zk_source", input.source)?;
    validate_finite_vector("yk_zk_source_coefficient", input.source_coefficients)?;
    Ok(())
}

fn validate_yk_zk_exchange_input(input: &AtomicYkZkExchangeInput<'_>) -> Result<(), AtomMathError> {
    validate_finite_scalar("yk_zk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    let left = one_based_atomic_orbital_index(input.left_orbital_1based, orbital_count)?;
    let right = one_based_atomic_orbital_index(input.right_orbital_1based, orbital_count)?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_differential_active_len(
        input.active_lengths[left].min(input.active_lengths[right]),
        input.radii.len(),
    )?;
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

fn validate_radial_integral_input(
    input: &AtomicRadialIntegralInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("fdrirk_step", input.step)?;
    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        input.radii.len(),
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        input.radii.len(),
        orbital_count,
    )?;
    let coefficient_count = input.large_coefficients.nrows();
    validate_coefficient_count("large_coefficients", coefficient_count)?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;

    if input.request.first_left > 0 && input.request.first_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.first_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.first_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
    }
    if input.request.second_left > 0 && input.request.second_right > 0 {
        let left = one_based_atomic_orbital_index(input.request.second_left, orbital_count)?;
        let right = one_based_atomic_orbital_index(input.request.second_right, orbital_count)?;
        validate_differential_active_len(
            input.active_lengths[left].min(input.active_lengths[right]),
            input.radii.len(),
        )?;
        if (input.request.first_left == 0 || input.request.first_right == 0)
            && input.previous_first_factor.is_none()
        {
            return Err(AtomMathError::MissingRadialFirstFactor);
        }
    }
    if let Some(first_factor) = input.previous_first_factor {
        validate_radial_table_len(
            "previous_first_factor",
            input.radii.len(),
            first_factor.values.len(),
        )?;
        validate_coefficient_vector_len(
            "previous_first_factor_coefficients",
            coefficient_count,
            first_factor.coefficients.len(),
        )?;
        validate_finite_vector("previous_first_factor", first_factor.values)?;
        validate_finite_vector(
            "previous_first_factor_coefficient",
            first_factor.coefficients,
        )?;
        validate_finite_scalar(
            "previous_first_factor_origin_power",
            first_factor.origin_power,
        )?;
    }
    Ok(())
}

fn validate_schmidt_orthogonalization_input(
    input: &AtomicSchmidtOrthogonalizationInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("active_lengths", orbital_count, input.active_lengths.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    if let Some(active_orbital_1based) = input.active_orbital_1based
        && !(1..=orbital_count).contains(&active_orbital_1based)
    {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based,
            orbital_count,
        });
    }

    let radial_rows = input.large_components.nrows();
    let coefficient_rows = input.large_coefficients.nrows();
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_rows,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_rows,
        orbital_count,
    )?;

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_rows {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_rows,
            });
        }
    }
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    Ok(())
}

fn validate_coulomb_coefficient_input(
    input: &AtomicCoulombCoefficientInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    for &kappa in input.kappas {
        doubled_j_from_kappa(kappa)?;
    }
    Ok(())
}

fn validate_orbital_initialization_input(
    input: &AtomicOrbitalInitializationInput<'_>,
) -> Result<(), AtomMathError> {
    if input.atomic_number == 0 {
        return Err(AtomMathError::InvalidAtomicNumber {
            atomic_number: input.atomic_number,
        });
    }
    validate_finite_scalar("inmuat_ionicity", input.ionicity)?;
    let orbital_count = input.occupations.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len(
        "principal_quantum_numbers",
        orbital_count,
        input.principal_quantum_numbers.len(),
    )?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_finite_slice("occupation", input.occupations)?;
    for (orbital, &principal_quantum_number) in input.principal_quantum_numbers.iter().enumerate() {
        if principal_quantum_number == 0 {
            return Err(AtomMathError::InvalidPrincipalQuantumNumber {
                orbital_1based: orbital + 1,
                principal_quantum_number,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_abs_usize(kappa)?;
    }
    let actual = input.occupations.iter().copied().sum::<Real>();
    let expected = input.atomic_number as Real - input.ionicity;
    if (expected - actual).abs() > ATOM_INMUAT_ELECTRON_TOLERANCE {
        return Err(AtomMathError::ElectronCountMismatch {
            atomic_number: input.atomic_number,
            ionicity: input.ionicity,
            expected,
            actual,
            tolerance: ATOM_INMUAT_ELECTRON_TOLERANCE,
        });
    }
    Ok(())
}

fn validate_dirac_entry_state_input(
    input: &AtomicDiracEntryStateInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_entry_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    Ok(())
}

fn validate_dirac_normalization_input(
    input: &AtomicDiracNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_step", input.step)?;
    validate_finite_scalar(
        "soldir_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar("soldir_origin_power", input.origin_power)?;
    validate_dirac_normalization_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_large_component", input.large_component)?;
    validate_finite_vector("soldir_small_component", input.small_component)?;
    validate_coefficient_count("soldir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("soldir_small_coefficient", input.small_coefficients)?;
    if input.method == 1
        && (input.matching_index_1based == 0 || input.matching_index_1based > input.active_len)
    {
        return Err(AtomMathError::DiracNormalizationMatchingIndexOutOfRange {
            matching_index_1based: input.matching_index_1based,
            active_len: input.active_len,
        });
    }
    Ok(())
}

fn validate_dirac_solution_normalization_input(
    input: &AtomicDiracSolutionNormalizationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_solution_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_solution_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_solution_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_dirac_solution_normalization_active_len(
        input.active_len,
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_solution_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_solution_large_component", input.large_component)?;
    validate_finite_vector("soldir_solution_small_component", input.small_component)?;
    validate_coefficient_count(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_solution_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_solution_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_solution_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_node_count_input(
    input: &AtomicDiracNodeCountInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector("soldir_node_large_component", input.large_component)?;
    validate_dirac_node_count_index(
        "matching",
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    validate_dirac_node_count_index("scan", input.scan_index_1based, input.large_component.len())?;
    Ok(())
}

fn validate_dirac_node_energy_search_input(
    input: &AtomicDiracNodeEnergySearchInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_node_search_energy", input.energy)?;
    validate_finite_scalar("soldir_node_search_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_node_search_einf", input.energy_inf)?;
    validate_finite_scalar("soldir_node_search_emin", input.energy_floor)?;
    validate_positive_finite_scalar("soldir_node_search_precision", input.energy_precision)?;
    Ok(())
}

fn validate_dirac_iteration_reset_input(
    input: &AtomicDiracIterationResetInput,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar(
        "soldir_iteration_primary_precision",
        input.primary_matching_precision,
    )?;
    validate_positive_finite_scalar(
        "soldir_iteration_secondary_precision",
        input.secondary_matching_precision,
    )?;
    validate_finite_scalar("soldir_iteration_energy_floor", input.energy_floor)?;
    validate_finite_scalar("soldir_iteration_reference_energy", input.reference_energy)?;
    Ok(())
}

fn validate_dirac_method_one_energy_correction_input(
    input: &AtomicDiracMethodOneEnergyCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("soldir_energy_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_matching_small_component",
        input.matching_small_component,
    )?;
    validate_radial_table_len(
        "soldir_energy_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector("soldir_energy_large_component", input.large_component)?;
    validate_finite_vector("soldir_energy_small_component", input.small_component)?;
    validate_dirac_energy_correction_matching_index(
        input.matching_index_1based,
        input.large_component.len(),
    )?;
    Ok(())
}

fn validate_dirac_energy_step_input(
    input: &AtomicDiracEnergyStepInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_energy_step_energy", input.energy)?;
    validate_finite_scalar("soldir_energy_step_correction", input.correction)?;
    validate_finite_scalar("soldir_energy_step_mismatch", input.mismatch)?;
    validate_finite_scalar("soldir_energy_step_esup", input.energy_sup)?;
    validate_finite_scalar("soldir_energy_step_einf", input.energy_inf)?;
    validate_positive_finite_scalar(
        "soldir_energy_step_mismatch_precision",
        input.mismatch_precision,
    )?;
    validate_positive_finite_scalar(
        "soldir_energy_step_zero_precision",
        input.zero_energy_precision,
    )?;
    Ok(())
}

fn validate_dirac_rematch_attempt_input(
    input: &AtomicDiracRematchAttemptInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_rematch_mismatch", input.mismatch)?;
    validate_positive_finite_scalar(
        "soldir_rematch_mismatch_precision",
        input.mismatch_precision,
    )?;
    Ok(())
}

fn validate_dirac_large_component_match_input(
    input: &AtomicDiracLargeComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_finite_vector("soldir_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    Ok(())
}

fn validate_dirac_homogeneous_match_input(
    input: &AtomicDiracHomogeneousMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_homogeneous_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_homogeneous_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_homogeneous_match_small_component",
        input.small_component,
    )?;
    Ok(())
}

fn validate_dirac_two_component_match_input(
    input: &AtomicDiracTwoComponentMatchInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar(
        "soldir_two_match_matching_large_component",
        input.matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_matching_small_component",
        input.matching_small_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_large_component",
        input.homogeneous_matching_large_component,
    )?;
    validate_finite_scalar(
        "soldir_two_match_homogeneous_matching_small_component",
        input.homogeneous_matching_small_component,
    )?;
    validate_dirac_match_active_len(input.active_len, input.large_component.len())?;
    validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)?;
    validate_radial_table_len(
        "soldir_two_match_small_component",
        input.large_component.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_large_component",
        input.large_component.len(),
        input.homogeneous_large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_two_match_homogeneous_small_component",
        input.large_component.len(),
        input.homogeneous_small_component.len(),
    )?;
    validate_coefficient_count(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_large_coefficients",
        input.coefficient_count,
        input.homogeneous_large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_two_match_homogeneous_small_coefficients",
        input.coefficient_count,
        input.homogeneous_small_coefficients.len(),
    )?;
    validate_finite_vector("soldir_two_match_large_component", input.large_component)?;
    validate_finite_vector("soldir_two_match_small_component", input.small_component)?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_component",
        input.homogeneous_large_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_component",
        input.homogeneous_small_component,
    )?;
    validate_finite_vector(
        "soldir_two_match_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_large_coefficient",
        input.homogeneous_large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_two_match_homogeneous_small_coefficient",
        input.homogeneous_small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_energy_disagreement_source_input(
    input: &AtomicDiracEnergyDisagreementSourceInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar(
        "soldir_energy_disagreement_speed_of_light",
        input.speed_of_light,
    )?;
    validate_dirac_energy_disagreement_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_small_coefficient",
        input.small_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_energy_disagreement_correction_input(
    input: &AtomicDiracEnergyDisagreementCorrectionInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("soldir_energy_disagreement_correction_step", input.step)?;
    validate_finite_scalar("soldir_energy_disagreement_correction_norm", input.norm)?;
    validate_finite_scalar(
        "soldir_energy_disagreement_correction_origin_power",
        input.origin_power,
    )?;
    validate_dirac_energy_disagreement_correction_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_component",
        input.radii.len(),
        input.large_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_component",
        input.radii.len(),
        input.small_component.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_large_derivative",
        input.radii.len(),
        input.large_derivative.len(),
    )?;
    validate_radial_table_len(
        "soldir_energy_disagreement_correction_small_derivative",
        input.radii.len(),
        input.small_derivative.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_coefficient_count(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_large_derivative_coefficients",
        input.coefficient_count,
        input.large_derivative_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_energy_disagreement_correction_small_derivative_coefficients",
        input.coefficient_count,
        input.small_derivative_coefficients.len(),
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_component",
        input.large_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_component",
        input.small_component,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative",
        input.large_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative",
        input.small_derivative,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_coefficient",
        input.large_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_coefficient",
        input.small_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_large_derivative_coefficient",
        input.large_derivative_coefficients,
    )?;
    validate_finite_vector(
        "soldir_energy_disagreement_correction_small_derivative_coefficient",
        input.small_derivative_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_matching_point_update_input(
    input: &AtomicDiracMatchingPointUpdateInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_vector(
        "soldir_matching_point_large_component",
        input.large_component,
    )?;
    if input.active_len > ATOM_SOLDIR_MATCHING_POINT_FALLBACK_OFFSET
        && input.active_len <= input.large_component.len()
    {
        validate_dirac_match_matching_index(input.matching_index_1based, input.active_len)
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len: input.active_len,
            radial_count: input.large_component.len(),
        })
    }
}

fn validate_dirac_inhomogeneous_seed_input(
    input: &AtomicDiracInhomogeneousSeedInput<'_>,
) -> Result<(), AtomMathError> {
    validate_radial_table_len(
        "soldir_seed_small_source",
        input.large_source.len(),
        input.small_source.len(),
    )?;
    validate_coefficient_count("soldir_seed_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "soldir_seed_large_coefficients",
        input.coefficient_count,
        input.large_source_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_seed_small_coefficients",
        input.coefficient_count,
        input.small_source_coefficients.len(),
    )?;
    validate_finite_vector("soldir_seed_large_source", input.large_source)?;
    validate_finite_vector("soldir_seed_small_source", input.small_source)?;
    validate_finite_vector(
        "soldir_seed_large_source_coefficient",
        input.large_source_coefficients,
    )?;
    validate_finite_vector(
        "soldir_seed_small_source_coefficient",
        input.small_source_coefficients,
    )?;
    Ok(())
}

fn validate_dirac_homogeneous_seed_input(
    input: &AtomicDiracHomogeneousSeedInput,
) -> Result<(), AtomMathError> {
    if input.radial_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_radial_len",
            minimum: 1,
            actual: input.radial_len,
        });
    }
    if input.coefficient_len == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "soldir_homogeneous_seed_coefficient_len",
            minimum: 1,
            actual: input.coefficient_len,
        });
    }
    Ok(())
}

fn validate_dirac_shooting_pass_setup_input(
    input: &AtomicDiracShootingPassSetupInput,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_shooting_pass_energy", input.energy)?;
    validate_finite_scalar(
        "soldir_shooting_pass_previous_energy",
        input.previous_energy,
    )?;
    Ok(())
}

fn validate_dirac_integration_input(
    input: &AtomicDiracIntegrationInput<'_>,
) -> Result<(), AtomMathError> {
    validate_positive_finite_scalar("intdir_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_scalar("intdir_step", input.step)?;
    validate_positive_finite_scalar("intdir_matching_precision", input.matching_precision)?;
    validate_finite_scalar("intdir_energy", input.energy)?;
    validate_finite_scalar("intdir_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "intdir_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_finite_scalar(
        "intdir_asymptotic_large_component",
        input.asymptotic_large_component,
    )?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    validate_dirac_integration_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "intdir_large_source",
        input.radii.len(),
        input.large_source.len(),
    )?;
    validate_radial_table_len(
        "intdir_small_source",
        input.radii.len(),
        input.small_source.len(),
    )?;
    validate_radial_table_len("intdir_potential", input.radii.len(), input.potential.len())?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("intdir_large_source", input.large_source)?;
    validate_finite_vector("intdir_small_source", input.small_source)?;
    validate_finite_vector("intdir_potential", input.potential)?;
    validate_coefficient_count("intdir_large_coefficients", input.coefficient_count)?;
    validate_coefficient_vector_capacity(
        "intdir_large_coefficients",
        input.coefficient_count,
        input.large_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_small_coefficients",
        input.coefficient_count,
        input.small_coefficients.len(),
    )?;
    validate_coefficient_vector_capacity(
        "intdir_potential_coefficients",
        input.coefficient_count,
        input.potential_coefficients.len(),
    )?;
    validate_finite_vector("intdir_large_coefficient", input.large_coefficients)?;
    validate_finite_vector("intdir_small_coefficient", input.small_coefficients)?;
    validate_finite_vector("intdir_potential_coefficient", input.potential_coefficients)?;
    atom_intdir_decay(input.energy, input.speed_of_light)?;

    match input.mode {
        AtomicDiracIntegrationMode::SearchMatchingPoint => Ok(()),
        AtomicDiracIntegrationMode::FixedMatchingPoint | AtomicDiracIntegrationMode::InwardOnly => {
            validate_dirac_integration_matching_index(
                input.matching_index_1based,
                input.active_len,
            )?;
            validate_dirac_integration_max_index(
                input.max_index_1based,
                input.matching_index_1based,
                input.active_len,
            )
        }
    }
}

fn validate_dirac_solver_setup_input(
    input: &AtomicDiracSolverSetupInput<'_>,
) -> Result<(), AtomMathError> {
    validate_finite_scalar("soldir_setup_energy", input.energy)?;
    validate_finite_scalar("soldir_setup_origin_power", input.origin_power)?;
    validate_finite_scalar(
        "soldir_setup_initial_large_coefficient",
        input.initial_large_coefficient,
    )?;
    validate_finite_scalar(
        "soldir_setup_initial_small_coefficient",
        input.initial_small_coefficient,
    )?;
    validate_positive_finite_scalar("soldir_setup_speed_of_light", input.speed_of_light)?;
    if input.kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa: input.kappa });
    }
    if input.principal_quantum_number == 0 {
        return Err(AtomMathError::InvalidDiracSolverPrincipalQuantumNumber {
            principal_quantum_number: input.principal_quantum_number,
        });
    }
    validate_dirac_solver_setup_active_len(input.active_len, input.radii.len())?;
    validate_radial_table_len(
        "soldir_setup_potential",
        input.radii.len(),
        input.potential.len(),
    )?;
    validate_coefficient_vector_capacity(
        "soldir_setup_potential_coefficients",
        1,
        input.potential_coefficients.len(),
    )?;
    validate_positive_finite_radii(input.radii)?;
    validate_finite_vector("soldir_setup_potential", input.potential)?;
    validate_finite_vector(
        "soldir_setup_potential_coefficient",
        input.potential_coefficients,
    )?;
    Ok(())
}

fn validate_local_density_potential_input(
    input: &AtomicLocalDensityPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count == 0 {
        return Err(AtomMathError::InvalidCount {
            field: "vlda_radii",
            minimum: 1,
            actual: radial_count,
        });
    }
    validate_positive_finite_scalar("vlda_speed_of_light", input.speed_of_light)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "initial_potential",
        radial_count,
        input.initial_potential.len(),
    )?;
    validate_radial_table_len(
        "initial_energy_density",
        radial_count,
        input.initial_energy_density.len(),
    )?;
    if input.initial_development_coefficients.len() < 2 {
        return Err(AtomMathError::InvalidCount {
            field: "initial_development_coefficients",
            minimum: 2,
            actual: input.initial_development_coefficients.len(),
        });
    }

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len(
        "valence_occupations",
        orbital_count,
        input.valence_occupations.len(),
    )?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }

    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_slice("valence_occupation", input.valence_occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_vector("initial_potential", input.initial_potential)?;
    validate_finite_vector(
        "initial_development_coefficient",
        input.initial_development_coefficients,
    )?;
    validate_finite_vector("initial_energy_density", input.initial_energy_density)?;
    Ok(())
}

fn validate_orbital_potential_input(
    input: &AtomicOrbitalPotentialInput<'_>,
) -> Result<(), AtomMathError> {
    let radial_count = input.radii.len();
    if radial_count < 4 {
        return Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len: radial_count,
            radial_count,
        });
    }
    validate_positive_finite_scalar("potrdf_speed_of_light", input.speed_of_light)?;
    validate_finite_scalar("potrdf_step", input.step)?;
    validate_positive_finite_radii(input.radii)?;
    validate_radial_table_len(
        "nuclear_potential",
        radial_count,
        input.nuclear_potential.len(),
    )?;

    let coefficient_count = input.nuclear_development_coefficients.len();
    validate_coefficient_count("nuclear_development_coefficients", coefficient_count)?;

    let orbital_count = input.active_lengths.len();
    if orbital_count == 0 {
        return Err(AtomMathError::EmptyOrbitalTable);
    }
    if input.self_consistent_count > orbital_count {
        return Err(AtomMathError::ActiveOrbitalOutOfRange {
            active_orbital_1based: input.self_consistent_count,
            orbital_count,
        });
    }
    one_based_atomic_orbital_index(input.active_orbital_1based, orbital_count)?;
    validate_orbital_table_len("kappas", orbital_count, input.kappas.len())?;
    validate_orbital_table_len("orbital_powers", orbital_count, input.orbital_powers.len())?;
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    validate_orbital_table_len("shell_markers", orbital_count, input.shell_markers.len())?;
    validate_orbital_table_len("origin_scales", orbital_count, input.origin_scales.len())?;
    validate_matrix_shape(
        "large_components",
        input.large_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_components",
        input.small_components,
        radial_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "large_coefficients",
        input.large_coefficients,
        coefficient_count,
        orbital_count,
    )?;
    validate_matrix_shape(
        "small_coefficients",
        input.small_coefficients,
        coefficient_count,
        orbital_count,
    )?;

    for (orbital, &active_len) in input.active_lengths.iter().enumerate() {
        if active_len > radial_count {
            return Err(AtomMathError::ActiveLengthOutOfRange {
                orbital_1based: orbital + 1,
                active_len,
                row_count: radial_count,
            });
        }
    }
    for &kappa in input.kappas {
        kappa_angular_rank(kappa)?;
    }
    for &origin_scale in input.origin_scales {
        validate_positive_finite_scalar("origin_scale", origin_scale)?;
    }
    validate_positive_occupation(
        "potrdf_active_orbital",
        input.active_orbital_1based - 1,
        input.occupations,
    )?;

    if input.include_lagrange {
        let expected_pairs = orbital_pair_count(orbital_count)?;
        validate_coefficient_vector_len(
            "lagrange_parameters",
            expected_pairs,
            input.lagrange_parameters.len(),
        )?;
        validate_finite_vector("lagrange_parameter", input.lagrange_parameters)?;
    }

    validate_finite_slice("orbital_power", input.orbital_powers)?;
    validate_finite_slice("occupation", input.occupations)?;
    validate_finite_matrix("large_component", input.large_components)?;
    validate_finite_matrix("small_component", input.small_components)?;
    validate_finite_matrix("large_coefficient", input.large_coefficients)?;
    validate_finite_matrix("small_coefficient", input.small_coefficients)?;
    validate_finite_vector("nuclear_potential", input.nuclear_potential)?;
    validate_finite_vector(
        "nuclear_development_coefficient",
        input.nuclear_development_coefficients,
    )?;
    Ok(())
}

fn validate_positive_finite_scalar(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    validate_finite_scalar(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::NonPositiveScalar { field, value })
    }
}

fn validate_positive_finite_nuclear_scalar(
    field: &'static str,
    value: Real,
) -> Result<(), AtomMathError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialScalar { field, value })
    }
}

fn validate_nuclear_count(
    field: &'static str,
    actual: usize,
    minimum: usize,
) -> Result<(), AtomMathError> {
    if actual >= minimum {
        Ok(())
    } else {
        Err(AtomMathError::InvalidNuclearPotentialCount {
            field,
            minimum,
            actual,
        })
    }
}

fn validate_differential_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDifferentialIntegralActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count && active_len % 2 == 1 {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNormalizationActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_integration_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > ATOM_INTDIR_HISTORY + 12 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracIntegrationActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_solver_setup_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracSolverSetupActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_solution_normalization_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracSolutionNormalizationActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

fn validate_dirac_match_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracMatchActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_energy_disagreement_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracEnergyDisagreementActiveLength {
            active_len,
            radial_count,
        })
    }
}

fn validate_dirac_energy_disagreement_correction_active_len(
    active_len: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if active_len > 0 && active_len % 2 == 1 && active_len <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::InvalidDiracEnergyDisagreementCorrectionActiveLength {
                active_len,
                radial_count,
            },
        )
    }
}

fn validate_dirac_match_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracMatchMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_dirac_node_count_index(
    field: &'static str,
    index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if index_1based > 0 && index_1based <= radial_count {
        Ok(())
    } else {
        Err(AtomMathError::InvalidDiracNodeCountIndex {
            field,
            index_1based,
            radial_count,
        })
    }
}

fn validate_dirac_energy_correction_matching_index(
    matching_index_1based: usize,
    radial_count: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > 0 && matching_index_1based <= radial_count {
        Ok(())
    } else {
        Err(
            AtomMathError::DiracEnergyCorrectionMatchingIndexOutOfRange {
                matching_index_1based,
                radial_count,
            },
        )
    }
}

fn validate_dirac_integration_matching_index(
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if matching_index_1based > ATOM_INTDIR_HISTORY && matching_index_1based <= active_len {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMatchingIndexOutOfRange {
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_dirac_integration_max_index(
    max_index_1based: usize,
    matching_index_1based: usize,
    active_len: usize,
) -> Result<(), AtomMathError> {
    if max_index_1based <= active_len
        && max_index_1based > matching_index_1based + ATOM_INTDIR_HISTORY
    {
        Ok(())
    } else {
        Err(AtomMathError::DiracIntegrationMaxIndexOutOfRange {
            max_index_1based,
            matching_index_1based,
            active_len,
        })
    }
}

fn validate_coefficient_count(table: &'static str, actual_len: usize) -> Result<(), AtomMathError> {
    if actual_len > 0 {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: 1,
            actual_len,
        })
    }
}

fn validate_coefficient_vector_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_coefficient_vector_capacity(
    table: &'static str,
    required_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len >= required_len {
        Ok(())
    } else {
        Err(AtomMathError::CoefficientTableLengthMismatch {
            table,
            expected_len: required_len,
            actual_len,
        })
    }
}

fn validate_matrix_shape(
    table: &'static str,
    matrix: ArrayView2<'_, Real>,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), AtomMathError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows == expected_rows && columns == expected_columns {
        Ok(())
    } else {
        Err(AtomMathError::MatrixShape {
            table,
            expected_rows,
            expected_columns,
            rows,
            columns,
        })
    }
}

fn validate_orbital_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_radial_table_len(
    table: &'static str,
    expected_len: usize,
    actual_len: usize,
) -> Result<(), AtomMathError> {
    if actual_len == expected_len {
        Ok(())
    } else {
        Err(AtomMathError::RadialTableLengthMismatch {
            table,
            expected_len,
            actual_len,
        })
    }
}

fn validate_occupation_tables(occupations: &[Real], kappas: &[i32]) -> Result<(), AtomMathError> {
    if occupations.len() != kappas.len() {
        return Err(AtomMathError::OccupationKappaLengthMismatch {
            occupation_len: occupations.len(),
            kappa_len: kappas.len(),
        });
    }
    validate_finite_slice("occupation", occupations)
}

fn validate_positive_occupation(
    context: &'static str,
    orbital: usize,
    occupations: &[Real],
) -> Result<Real, AtomMathError> {
    let occupation = occupations[orbital];
    if occupation > 0.0 {
        Ok(occupation)
    } else {
        Err(AtomMathError::NonPositiveOccupation {
            context,
            orbital_1based: orbital + 1,
            occupation,
        })
    }
}

fn validate_orbital_index(index: usize, len: usize) -> Result<(), AtomMathError> {
    if index < len {
        Ok(())
    } else {
        Err(AtomMathError::OrbitalIndexOutOfRange { index, len })
    }
}

fn validate_coefficient_table(
    coefficients: ArrayView3<'_, Real>,
    left: usize,
    right: usize,
    rank: usize,
) -> Result<(), AtomMathError> {
    let shape = coefficients.shape();
    let rows = shape[0];
    let columns = shape[1];
    let channels = shape[2];
    if rows == 0 || columns == 0 || rows != columns || channels == 0 {
        return Err(AtomMathError::CoefficientTableShape {
            rows,
            columns,
            channels,
        });
    }
    if left >= rows {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: left,
            len: rows,
        });
    }
    if right >= columns {
        return Err(AtomMathError::OrbitalIndexOutOfRange {
            index: right,
            len: columns,
        });
    }
    let channel = rank / 2;
    if channel >= channels {
        return Err(AtomMathError::CoefficientChannelOutOfRange {
            rank,
            channel,
            channels,
        });
    }
    for value in coefficients.iter().copied() {
        if !value.is_finite() {
            return Err(AtomMathError::NonFiniteScalar {
                field: "coefficient",
                value,
            });
        }
    }
    Ok(())
}

fn validate_finite_slice(field: &'static str, values: &[Real]) -> Result<(), AtomMathError> {
    for &value in values {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_finite_vector(
    field: &'static str,
    values: ArrayView1<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in values.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_finite_matrix(
    field: &'static str,
    matrix: ArrayView2<'_, Real>,
) -> Result<(), AtomMathError> {
    for value in matrix.iter().copied() {
        validate_finite_scalar(field, value)?;
    }
    Ok(())
}

fn validate_positive_finite_radii(values: ArrayView1<'_, Real>) -> Result<(), AtomMathError> {
    for &radius in values {
        validate_finite_scalar("radius", radius)?;
        if radius <= 0.0 {
            return Err(AtomMathError::NonPositiveRadius { radius });
        }
    }
    Ok(())
}

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AtomMathError::NonFiniteScalar { field, value })
    }
}

#[cfg(test)]
mod tests;
