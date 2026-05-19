//! Atomic lookup tables and small ATOM helper kernels ported from FEFF.
//!
//! This module ports `ATOM/nucmass.f90` and `COMMON/pertab.f90`. FEFF stores
//! many unsuffixed real literals in double-precision arrays, so those values
//! are rounded through single precision before use; the Rust tables keep that
//! behavior explicitly. It also includes compact helper routines from
//! `ATOM/aprdev.f90`, `ATOM/cofcon.f90`, `ATOM/dentfa.f90`,
//! `ATOM/fdmocc.f90`, `ATOM/akeato.f90`, `ATOM/muatco.f90`,
//! `ATOM/bkmrdf.f90`, and `ATOM/s02at.f90`.

use crate::angular::{AngularError, wigner_3j};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ShapeBuilder};
use thiserror::Error;

use crate::Real;

/// Error returned by FEFF atomic lookup helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AtomicError {
    /// The requested FEFF atomic lookup table does not contain this atomic number.
    #[error("atomic number {z} is not present in the requested FEFF table")]
    InvalidAtomicNumber { z: usize },
}

/// Error returned by FEFF ATOM helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum AtomMathError {
    /// `aprdev` needs the requested 1-based term to fit both coefficient rows.
    #[error(
        "atomic polynomial term {term_count} is invalid for coefficient lengths {left_len} and {right_len}"
    )]
    InvalidPolynomialTerm {
        term_count: usize,
        left_len: usize,
        right_len: usize,
    },
    /// Scalar inputs must be finite.
    #[error("atomic {field} must be finite, got {value}")]
    NonFiniteScalar { field: &'static str, value: Real },
    /// Relativistic kappa values must be nonzero and fit FEFF's integer algebra.
    #[error("invalid atomic relativistic kappa {kappa}")]
    InvalidKappa { kappa: i32 },
    /// Two FEFF relativistic kappa values overflowed while forming `kap(j)-kap(i)`.
    #[error("atomic kappa difference overflow for left={left_kappa}, right={right_kappa}")]
    KappaDifferenceOutOfRange { left_kappa: i32, right_kappa: i32 },
    /// The Breit rank must fit the FEFF integer arithmetic used by `bkmrdf`.
    #[error("atomic Breit angular rank {rank} is outside FEFF integer range")]
    BreitRankOutOfRange { rank: usize },
    /// FEFF `etotal` Breit exchange branch arithmetic overflowed.
    #[error("atomic Breit exchange branch rank is outside FEFF integer range")]
    BreitBranchOutOfRange,
    /// Wigner 3j construction failed while building Breit angular coefficients.
    #[error("atomic Breit angular coefficient construction failed")]
    BreitAngular(#[from] AngularError),
    /// The FEFF `muatco` multipole rank must fit integer Wigner arithmetic.
    #[error("atomic Coulomb angular rank {rank} is outside FEFF integer range")]
    CoulombRankOutOfRange { rank: usize },
    /// Wigner 3j construction failed while building Coulomb angular coefficients.
    #[error("atomic Coulomb angular coefficient construction failed")]
    CoulombAngular {
        /// Source Wigner-3j error.
        source: AngularError,
    },
    /// Thomas-Fermi density approximation divides by the radius.
    #[error("atomic radius must be positive, got {radius}")]
    NonPositiveRadius { radius: Real },
    /// Occupation and kappa tables must have identical orbital counts.
    #[error(
        "atomic occupation/kappa length mismatch: occupations={occupation_len}, kappas={kappa_len}"
    )]
    OccupationKappaLengthMismatch {
        occupation_len: usize,
        kappa_len: usize,
    },
    /// FEFF atomic orbital tables must all use the same active orbital count.
    #[error(
        "atomic orbital table {table} length mismatch: expected {expected_len}, got {actual_len}"
    )]
    OrbitalTableLengthMismatch {
        table: &'static str,
        expected_len: usize,
        actual_len: usize,
    },
    /// ATOM total-energy accumulation requires at least one orbital.
    #[error("atomic total energy requires at least one orbital")]
    EmptyOrbitalTable,
    /// The requested one-based core-hole orbital is outside the active table.
    #[error("atomic hole orbital {hole_orbital_1based} is outside 1..={orbital_count}")]
    HoleOrbitalOutOfRange {
        hole_orbital_1based: usize,
        orbital_count: usize,
    },
    /// FEFF `s02at` uses fixed 8x8 work matrices for each kappa group.
    #[error("atomic kappa group {kappa} has {count} orbitals, exceeding FEFF limit {limit}")]
    KappaGroupTooLarge {
        kappa: i32,
        count: usize,
        limit: usize,
    },
    /// Relaxed-overlap matrices must be square over the active orbital table.
    #[error("atomic overlap matrix must be {expected}x{expected}, got {rows}x{columns}")]
    OverlapMatrixShape {
        expected: usize,
        rows: usize,
        columns: usize,
    },
    /// Orbital index is outside the supplied table.
    #[error("atomic orbital index {index} is outside table length {len}")]
    OrbitalIndexOutOfRange { index: usize, len: usize },
    /// Same-orbital `fdmocc` needs a valid relativistic kappa.
    #[error("same-orbital occupation product requires nonzero kappa")]
    ZeroKappa,
    /// FEFF `afgk` must be a square rank-3 table with nonempty orbital axes.
    #[error("atomic Coulomb coefficient table has invalid shape ({rows}, {columns}, {channels})")]
    CoefficientTableShape {
        rows: usize,
        columns: usize,
        channels: usize,
    },
    /// The requested `k/2` channel is outside the supplied table.
    #[error("atomic Coulomb rank {rank} maps to channel {channel}, but table has {channels}")]
    CoefficientChannelOutOfRange {
        rank: usize,
        channel: usize,
        channels: usize,
    },
}

/// Result of FEFF `cofcon` convergence acceleration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicConvergenceMix {
    /// Weight `a = 1 - b`.
    pub initial_weight: Real,
    /// Updated final-iteration weight `b`.
    pub final_weight: Real,
    /// Updated previous error `q`, set to the current error `p`.
    pub previous_error: Real,
}

/// FEFF `bkmrdf` angular coefficients for Breit interaction terms.
///
/// The three entries correspond to FEFF's `-1`, `0`, and `+1` magnetic-order
/// slots, respectively.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicBreitAngularCoefficients {
    /// Magnetic-interaction coefficients, FEFF `cmag(1:3)`.
    pub magnetic: [Real; 3],
    /// Retarded-term coefficients, FEFF `cret(1:3)`.
    pub retarded: [Real; 3],
}

/// Inputs for FEFF `ATOM/muatco.f90` Coulomb angular coefficients.
#[derive(Debug, Clone, Copy)]
pub struct AtomicCoulombCoefficientInput<'a> {
    /// Relativistic kappa values for active orbitals, FEFF `kap`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupation flags, FEFF `xnval`; positive pairs skip exchange
    /// coefficients like FEFF.
    pub valence_occupations: &'a [Real],
}

/// Inputs for FEFF `ATOM/etotal.f90` total-energy accumulation.
///
/// The radial integral solver is deliberately supplied as a callback to keep
/// this helper focused on FEFF's Coulomb/Breit energy algebra.
#[derive(Debug, Clone, Copy)]
pub struct AtomicTotalEnergyInput<'a> {
    /// Relativistic kappa values for active orbitals.
    pub kappas: &'a [i32],
    /// Occupation counts, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Valence occupation flags, FEFF `xnval`; positive values trigger FEFF's
    /// half-weight branch in exchange Coulomb accumulation.
    pub valence_occupations: &'a [Real],
    /// One-electron orbital energies, FEFF `en`.
    pub orbital_energies: &'a [Real],
    /// Coulomb angular coefficients, FEFF `afgk`, indexed as
    /// `(orbital, orbital, rank / 2)`.
    pub coulomb_coefficients: ArrayView3<'a, Real>,
}

/// FEFF-style radial integral request passed to `atomic_total_energy`.
///
/// Orbital indices are one-based to mirror FEFF `fdrirk`. A value of `0`
/// preserves the sentinel cases used by FEFF for the already-tabulated first
/// radial factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicRadialIntegralRequest {
    /// First left orbital index, FEFF `i`.
    pub first_left: usize,
    /// First right orbital index, FEFF `j`.
    pub first_right: usize,
    /// Second left orbital index, FEFF `l`.
    pub second_left: usize,
    /// Second right orbital index, FEFF `m`.
    pub second_right: usize,
    /// Radial integral rank, FEFF `k`.
    pub rank: usize,
}

/// Result of FEFF `ATOM/etotal.f90` total-energy accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomicTotalEnergy {
    /// Total atomic energy in FEFF's internal energy units.
    pub total: Real,
    /// Direct Coulomb contribution, FEFF `ener(1)`.
    pub direct_coulomb: Real,
    /// Exchange Coulomb contribution, FEFF `ener(2)`.
    pub exchange_coulomb: Real,
    /// Magnetic Breit contribution, FEFF `ener(3)`.
    pub magnetic_breit: Real,
    /// Retarded Breit contribution, FEFF `ener(4)`.
    pub retarded_breit: Real,
}

/// Inputs for FEFF `ATOM/s02at.f90` relaxed-overlap amplitude reduction.
#[derive(Debug, Clone, Copy)]
pub struct AtomicOverlapAmplitudeReductionInput<'a> {
    /// Optional one-based orbital index containing the core hole, FEFF `ihole`.
    pub hole_orbital_1based: Option<usize>,
    /// Relativistic kappa values for active orbitals, FEFF `nk`.
    pub kappas: &'a [i32],
    /// Active-orbital occupation counts after valence subtraction, FEFF `xnel`.
    pub occupations: &'a [Real],
    /// Relaxed-orbital overlap integrals, FEFF `ovpint(i,j)`.
    pub overlap_integrals: ArrayView2<'a, Real>,
}

/// Port of FEFF `COMMON/pertab.f90::atwtd`: return the periodic-table weight.
///
/// This is the table used by FEFF's Debye and Einstein-model mass calculations.
/// It covers `Z = 1..=139` and intentionally preserves FEFF's single-precision
/// rounding of the literal data statements.
pub fn atomic_weight(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_WEIGHTS
        .get(atomic_number - 1)
        .map(|&weight| Real::from(weight))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `COMMON/pertab.f90::atsym`: return the element symbol.
///
/// FEFF's table is returned trimmed rather than padded to Fortran
/// `character*3` width. The values, including historical placeholder names
/// and table quirks, match FEFF's data statement.
pub fn atomic_symbol(atomic_number: usize) -> Result<&'static str, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_ATOMIC_SYMBOLS
        .get(atomic_number - 1)
        .copied()
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

/// Port of FEFF `nucmass`: return the tabulated standard atomic weight.
///
/// The value is returned in atomic mass units. FEFF uses this table when the
/// `HIGHZ` path requests a finite nuclear-radius model for heavy atoms.
pub fn nuclear_mass(atomic_number: usize) -> Result<Real, AtomicError> {
    if atomic_number == 0 {
        return Err(AtomicError::InvalidAtomicNumber { z: atomic_number });
    }
    FEFF_NUCLEAR_MASSES
        .get(atomic_number - 1)
        .map(|&mass| Real::from(mass))
        .ok_or(AtomicError::InvalidAtomicNumber { z: atomic_number })
}

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
    let channel = rank / 2;
    if left <= right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(coefficients[(right, left, channel)])
    }
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
    let channel = rank / 2;
    if left < right {
        Ok(coefficients[(right, left, channel)])
    } else if left > right {
        Ok(coefficients[(left, right, channel)])
    } else {
        Ok(0.0)
    }
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

/// Port of FEFF `ATOM/bkmrdf.f90`, the Breit angular coefficients.
///
/// `left_kappa` and `right_kappa` are the relativistic kappa values for the
/// two orbitals. `rank` is FEFF's integer `k` for the Breit radial integral.
pub fn atomic_breit_angular_coefficients(
    left_kappa: i32,
    right_kappa: i32,
    rank: usize,
) -> Result<AtomicBreitAngularCoefficients, AtomMathError> {
    let left_j2 = doubled_j_from_kappa(left_kappa)?;
    let right_j2 = doubled_j_from_kappa(right_kappa)?;
    let rank_i32 = i32::try_from(rank).map_err(|_| AtomMathError::BreitRankOutOfRange { rank })?;
    let kappa_difference =
        right_kappa
            .checked_sub(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;
    let kappa_sum =
        right_kappa
            .checked_add(left_kappa)
            .ok_or(AtomMathError::KappaDifferenceOutOfRange {
                left_kappa,
                right_kappa,
            })?;

    let mut coefficients = AtomicBreitAngularCoefficients {
        magnetic: [0.0; 3],
        retarded: [0.0; 3],
    };
    let mut angular_l = rank_i32
        .checked_sub(1)
        .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    for order in 0..3 {
        if angular_l >= 0 {
            accumulate_breit_order(
                BreitOrderContext {
                    left_j2,
                    right_j2,
                    kappa_difference,
                    kappa_sum,
                    rank: rank_i32,
                    rank_usize: rank,
                    angular_l,
                    order,
                },
                &mut coefficients,
            )?;
        }
        angular_l = angular_l
            .checked_add(1)
            .ok_or(AtomMathError::BreitRankOutOfRange { rank })?;
    }

    Ok(coefficients)
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

/// Port of FEFF `ATOM/s02at.f90`, the relaxed-overlap amplitude reduction.
///
/// The overlap matrix is consumed in FEFF group order by kappa, using only the
/// upper triangular entries that FEFF copies into its symmetric work matrix.
pub fn atomic_overlap_amplitude_reduction(
    input: AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<Real, AtomMathError> {
    const KAPPA_MIN: i32 = -5;
    const KAPPA_MAX: i32 = 4;
    const GROUP_LIMIT: usize = 8;

    validate_overlap_amplitude_input(&input)?;
    let hole_zero_based = input.hole_orbital_1based.map(|index| index - 1);
    let mut amplitude = 1.0;

    for kappa in KAPPA_MIN..=KAPPA_MAX {
        if kappa == 0 {
            continue;
        }
        let group = input
            .kappas
            .iter()
            .enumerate()
            .filter_map(|(orbital, &orbital_kappa)| (orbital_kappa == kappa).then_some(orbital))
            .collect::<Vec<_>>();
        if group.is_empty() {
            continue;
        }
        if group.len() > GROUP_LIMIT {
            return Err(AtomMathError::KappaGroupTooLarge {
                kappa,
                count: group.len(),
                limit: GROUP_LIMIT,
            });
        }

        let mut matrix = s02at_overlap_matrix(&group, input.overlap_integrals);
        let determinant_all = s02at_squared_determinant_in_place(&mut matrix, group.len());
        let determinant_without_last =
            s02at_squared_determinant_in_place(&mut matrix, group.len() - 1);
        let last_orbital = *group.last().ok_or(AtomMathError::KappaGroupTooLarge {
            kappa,
            count: 0,
            limit: GROUP_LIMIT,
        })?;
        let occupation = input.occupations[last_orbital];
        let max_occupation = 2.0 * Real::from(kappa.unsigned_abs());
        let hole_vacancy = max_occupation - occupation;
        let hole_position = hole_zero_based.and_then(|hole| {
            group
                .iter()
                .position(|&orbital| orbital == hole)
                .map(|position| position + 1)
        });

        amplitude *= match hole_position {
            None => determinant_all.powf(occupation) * determinant_without_last.powf(hole_vacancy),
            Some(position) if position == group.len() => {
                determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy + 1.0)
            }
            Some(position) => {
                let mut eliminated = s02at_eliminate_hole(matrix.view(), position - 1);
                let determinant_eliminated_all =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len());
                let determinant_eliminated_without_last =
                    s02at_squared_determinant_in_place(&mut eliminated, group.len() - 1);
                let mixed = (determinant_eliminated_without_last * determinant_all * hole_vacancy
                    + determinant_eliminated_all * determinant_without_last * occupation)
                    / max_occupation;
                mixed
                    * determinant_all.powf(occupation - 1.0)
                    * determinant_without_last.powf(hole_vacancy - 1.0)
            }
        };
        validate_finite_scalar("s02", amplitude)?;
    }

    Ok(amplitude)
}

struct AtomicTotalEnergyContext<'a, F> {
    input: AtomicTotalEnergyInput<'a>,
    radial_integral: F,
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

#[derive(Debug, Clone, Copy)]
struct BreitOrderContext {
    left_j2: i32,
    right_j2: i32,
    kappa_difference: i32,
    kappa_sum: i32,
    rank: i32,
    rank_usize: usize,
    angular_l: i32,
    order: usize,
}

#[derive(Debug, Clone, Copy)]
struct BreitOrderTerms {
    cm: Real,
    cz: Real,
    cp: Real,
    d: Real,
    retardation: Option<BreitRetardationTerms>,
}

#[derive(Debug, Clone, Copy)]
struct BreitRetardationTerms {
    am: Real,
    az: Real,
    ap: Real,
    scale: Real,
}

fn accumulate_breit_order(
    context: BreitOrderContext,
    coefficients: &mut AtomicBreitAngularCoefficients,
) -> Result<(), AtomMathError> {
    let wigner_j3 = context
        .angular_l
        .checked_mul(2)
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let wigner = wigner_3j(context.left_j2, context.right_j2, wigner_j3, -1, 1, 2)?;
    if wigner == 0.0 {
        return Ok(());
    }

    let angular_denominator = context
        .angular_l
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(AtomMathError::BreitRankOutOfRange {
            rank: context.rank_usize,
        })?;
    let squared_wigner = wigner * wigner;
    let terms = breit_order_terms(context, angular_denominator);

    if let Some(retardation) = terms.retardation {
        let mut retardation_scale = retardation.scale;
        let denominator = Real::from(angular_denominator.abs()) * terms.d;
        if denominator != 0.0 {
            retardation_scale /= denominator;
        }
        coefficients.retarded[0] +=
            squared_wigner * (retardation.am - retardation_scale * terms.cm);
        coefficients.retarded[1] +=
            (squared_wigner + squared_wigner) * (retardation.az - retardation_scale * terms.cz);
        coefficients.retarded[2] +=
            squared_wigner * (retardation.ap - retardation_scale * terms.cp);
    }

    if terms.d != 0.0 {
        let magnetic_scale = squared_wigner / terms.d;
        coefficients.magnetic[0] += terms.cm * magnetic_scale;
        coefficients.magnetic[1] += terms.cz * (magnetic_scale + magnetic_scale);
        coefficients.magnetic[2] += terms.cp * magnetic_scale;
    }

    Ok(())
}

fn breit_order_terms(context: BreitOrderContext, angular_denominator: i32) -> BreitOrderTerms {
    match context.order {
        0 => {
            let cm = square(context.kappa_difference + context.rank);
            let cz = square(context.kappa_difference) - square(context.rank);
            let cp = square(context.rank - context.kappa_difference);
            let scale = Real::from(context.rank);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
        1 => {
            let cm = square(context.kappa_sum);
            BreitOrderTerms {
                cm,
                cz: cm,
                cp: cm,
                d: Real::from(context.rank) * Real::from(context.rank + 1),
                retardation: None,
            }
        }
        _ => {
            let cm = square(context.kappa_difference - context.angular_l);
            let cz = square(context.kappa_difference) - square(context.angular_l);
            let cp = square(context.kappa_difference + context.angular_l);
            let scale = Real::from(context.angular_l);
            let retardation = breit_retardation_shape(
                context.kappa_difference,
                context.angular_l,
                -angular_denominator,
                scale,
            );
            BreitOrderTerms {
                cm,
                cz,
                cp,
                d: scale * Real::from(context.rank + context.rank + 1),
                retardation: Some(retardation),
            }
        }
    }
}

fn breit_retardation_shape(
    kappa_difference: i32,
    angular_l: i32,
    denominator: i32,
    scale: Real,
) -> BreitRetardationTerms {
    let next_l = angular_l + 1;
    let denominator = Real::from(denominator);
    BreitRetardationTerms {
        am: Real::from((kappa_difference - angular_l) * (kappa_difference + next_l)) / denominator,
        az: Real::from(kappa_difference * kappa_difference + angular_l * next_l) / denominator,
        ap: Real::from((angular_l + kappa_difference) * (kappa_difference - next_l)) / denominator,
        scale,
    }
}

fn square(value: i32) -> Real {
    let value = Real::from(value);
    value * value
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

fn atom_usize_to_i32(value: usize) -> Result<i32, AtomMathError> {
    i32::try_from(value).map_err(|_| AtomMathError::CoulombRankOutOfRange { rank: value })
}

fn abs_kappa_i32(kappa: i32) -> Result<i32, AtomMathError> {
    if kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa });
    }
    kappa
        .checked_abs()
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

fn s02at_overlap_matrix(group: &[usize], overlaps: ArrayView2<'_, Real>) -> Array2<Real> {
    let order = group.len();
    let mut matrix = Array2::zeros((order, order).f());
    for column in 0..order {
        for row in 0..=column {
            let value = overlaps[(group[row], group[column])];
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    matrix
}

fn s02at_eliminate_hole(matrix: ArrayView2<'_, Real>, hole: usize) -> Array2<Real> {
    Array2::from_shape_fn((matrix.nrows(), matrix.ncols()).f(), |(row, column)| {
        if row == hole && column == hole {
            1.0
        } else if row == hole || column == hole {
            0.0
        } else {
            matrix[(row, column)]
        }
    })
}

fn s02at_squared_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let determinant = s02at_determinant_in_place(matrix, order);
    determinant * determinant
}

fn s02at_determinant_in_place(matrix: &mut Array2<Real>, order: usize) -> Real {
    let mut determinant = 1.0;
    for pivot in 0..order {
        if matrix[(pivot, pivot)] == 0.0 {
            let Some(swap_column) = (pivot..order).find(|&column| matrix[(pivot, column)] != 0.0)
            else {
                return 0.0;
            };
            for row in pivot..order {
                let saved = matrix[(row, swap_column)];
                matrix[(row, swap_column)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = saved;
            }
            determinant = -determinant;
        }

        determinant *= matrix[(pivot, pivot)];
        if pivot + 1 < order {
            for row in (pivot + 1)..order {
                for column in (pivot + 1)..order {
                    matrix[(row, column)] -=
                        matrix[(row, pivot)] * matrix[(pivot, column)] / matrix[(pivot, pivot)];
                }
            }
        }
    }
    determinant
}

fn validate_overlap_amplitude_input(
    input: &AtomicOverlapAmplitudeReductionInput<'_>,
) -> Result<(), AtomMathError> {
    let orbital_count = input.kappas.len();
    validate_orbital_table_len("occupations", orbital_count, input.occupations.len())?;
    if let Some(hole_orbital_1based) = input.hole_orbital_1based
        && !(1..=orbital_count).contains(&hole_orbital_1based)
    {
        return Err(AtomMathError::HoleOrbitalOutOfRange {
            hole_orbital_1based,
            orbital_count,
        });
    }
    let [rows, columns] = input.overlap_integrals.shape().try_into().map_err(|_| {
        AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows: input.overlap_integrals.nrows(),
            columns: input.overlap_integrals.ncols(),
        }
    })?;
    if rows != orbital_count || columns != orbital_count {
        return Err(AtomMathError::OverlapMatrixShape {
            expected: orbital_count,
            rows,
            columns,
        });
    }
    for &kappa in input.kappas {
        abs_kappa_i32(kappa)?;
    }
    validate_finite_slice("occupation", input.occupations)?;
    for &value in &input.overlap_integrals {
        validate_finite_scalar("overlap_integral", value)?;
    }
    Ok(())
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

fn validate_occupation_tables(occupations: &[Real], kappas: &[i32]) -> Result<(), AtomMathError> {
    if occupations.len() != kappas.len() {
        return Err(AtomMathError::OccupationKappaLengthMismatch {
            occupation_len: occupations.len(),
            kappa_len: kappas.len(),
        });
    }
    validate_finite_slice("occupation", occupations)
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

fn validate_finite_scalar(field: &'static str, value: Real) -> Result<(), AtomMathError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AtomMathError::NonFiniteScalar { field, value })
    }
}

const FEFF_ATOMIC_WEIGHTS: [f32; 139] = [
    1.0079, 4.0026, 6.941, 9.0122, 10.81, 12.01, 14.007, 15.999, 18.998, 20.18, 22.9898, 24.305,
    26.982, 28.086, 30.974, 32.064, 35.453, 39.948, 39.09, 40.08, 44.956, 47.90, 50.942, 52.00,
    54.938, 55.85, 58.93, 58.71, 63.55, 65.38, 69.72, 72.59, 74.922, 78.96, 79.91, 83.80, 85.47,
    87.62, 88.91, 91.22, 92.91, 95.94, 98.91, 101.07, 102.90, 106.40, 107.87, 112.40, 114.82,
    118.69, 121.75, 127.60, 126.90, 131.30, 132.91, 137.34, 138.91, 140.12, 140.91, 144.24, 145.0,
    150.35, 151.96, 157.25, 158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95,
    183.85, 186.2, 190.20, 192.22, 195.09, 196.97, 200.59, 204.37, 207.19, 208.98, 210.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.04, 231.0, 238.03, 237.05, 244.0, 243.0, 247.0, 247.0, 251.0,
    252.0, 257.0, 258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0,
    285.0, 286.0, 289.0, 289.0, 293.0, 294.0, 294.0, 315.0, 320.0, 330.0, 334.0, 337.0, 340.0,
    344.0, 347.0, 350.0, 354.0, 357.0, 361.0, 364.0, 367.0, 371.0, 374.0, 378.0, 381.0, 385.0,
    388.0, 392.0,
];

const FEFF_ATOMIC_SYMBOLS: [&str; 139] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Te", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo", "Uue", "Ubn", "Ubu", "Ubb", "Ubt", "Ubq", "Ubp", "Ubh",
    "Ubs", "Ubo", "Ube", "Utn", "Utu", "Utb", "Utt", "Utq", "Utp", "Uth", "Uts", "Uto", "Ute",
];

#[allow(clippy::excessive_precision)]
const FEFF_NUCLEAR_MASSES: [f32; 138] = [
    1.00794,
    4.002602,
    6.941,
    9.012182,
    10.811,
    12.0107,
    14.0067,
    15.9994,
    18.9984032,
    20.1797,
    22.98976928,
    24.305,
    26.9815386,
    28.0855,
    30.973762,
    32.065,
    35.453,
    39.948,
    39.0983,
    40.078,
    44.955912,
    47.867,
    50.9415,
    51.9961,
    54.938045,
    55.845,
    58.933195,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.64,
    74.9216,
    78.96,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90585,
    91.224,
    92.90638,
    95.96,
    98.0,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.411,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.9054519,
    137.327,
    138.90547,
    140.116,
    140.90765,
    144.242,
    145.0,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93032,
    167.259,
    168.93421,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.59,
    204.3833,
    207.2,
    208.9804,
    209.0,
    210.0,
    222.0,
    223.0,
    226.0,
    227.0,
    232.03806,
    231.03588,
    238.02891,
    237.0,
    244.0,
    243.0,
    247.0,
    247.0,
    251.0,
    252.0,
    257.0,
    258.0,
    259.0,
    262.0,
    265.0,
    268.0,
    271.0,
    272.0,
    277.0,
    276.0,
    281.0,
    280.0,
    285.0,
    284.0,
    289.0,
    288.0,
    293.0,
    294.0,
    294.0,
    315.0,
    320.0,
    330.0,
    334.0,
    337.0,
    340.0,
    344.0,
    347.0,
    350.0,
    354.0,
    357.0,
    361.0,
    364.0,
    367.0,
    371.0,
    374.0,
    378.0,
    381.0,
    385.0,
    388.0,
];

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3};

    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    fn assert_close_with(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual - expected).abs() < tolerance,
            "actual={actual}, expected={expected}, diff={}",
            (actual - expected).abs()
        );
    }

    #[test]
    fn atomic_weight_matches_feff_pertab_reference() -> Result<(), AtomicError> {
        assert_close(atomic_weight(1)?, 1.007_899_999_618_530_3);
        assert_close(atomic_weight(2)?, 4.002_600_193_023_682);
        assert_close(atomic_weight(26)?, 55.849_998_474_121_094);
        assert_close(atomic_weight(75)?, 186.199_996_948_242_2);
        assert_close(atomic_weight(92)?, 238.029_998_779_296_88);
        assert_close(atomic_weight(118)?, 294.0);
        assert_close(atomic_weight(121)?, 330.0);
        assert_close(atomic_weight(139)?, 392.0);
        Ok(())
    }

    #[test]
    fn atomic_symbol_matches_feff_pertab_reference() -> Result<(), AtomicError> {
        assert_eq!(atomic_symbol(1)?, "H");
        assert_eq!(atomic_symbol(2)?, "He");
        assert_eq!(atomic_symbol(26)?, "Fe");
        assert_eq!(atomic_symbol(75)?, "Te");
        assert_eq!(atomic_symbol(92)?, "U");
        assert_eq!(atomic_symbol(118)?, "Uuo");
        assert_eq!(atomic_symbol(121)?, "Ubu");
        assert_eq!(atomic_symbol(139)?, "Ute");
        Ok(())
    }

    #[test]
    fn nuclear_mass_matches_feff_reference() -> Result<(), AtomicError> {
        assert_close(nuclear_mass(1)?, 1.007_940_053_939_819_3);
        assert_close(nuclear_mass(6)?, 12.010_700_225_830_078);
        assert_close(nuclear_mass(29)?, 63.546_001_434_326_17);
        assert_close(nuclear_mass(57)?, 138.905_471_801_757_8);
        assert_close(nuclear_mass(92)?, 238.028_915_405_273_44);
        assert_close(nuclear_mass(118)?, 294.0);
        assert_close(nuclear_mass(121)?, 330.0);
        assert_close(nuclear_mass(138)?, 388.0);
        Ok(())
    }

    #[test]
    fn nuclear_mass_rejects_invalid_atomic_numbers() {
        assert_eq!(
            nuclear_mass(0),
            Err(AtomicError::InvalidAtomicNumber { z: 0 })
        );
        assert_eq!(
            nuclear_mass(139),
            Err(AtomicError::InvalidAtomicNumber { z: 139 })
        );
        assert_eq!(
            atomic_weight(140),
            Err(AtomicError::InvalidAtomicNumber { z: 140 })
        );
        assert_eq!(
            atomic_symbol(0),
            Err(AtomicError::InvalidAtomicNumber { z: 0 })
        );
    }

    #[test]
    fn atom_helper_kernels_match_feff_reference() -> Result<(), AtomMathError> {
        let left = (1..=10)
            .map(|index| 0.1 * index as Real + 0.03)
            .collect::<Vec<_>>();
        let right = (1..=10)
            .map(|index| -0.04 * index as Real + 0.25)
            .collect::<Vec<_>>();
        assert_close(
            atomic_polynomial_product_coefficient(&left, &right, 3)?,
            0.125_300_000_000_000_02,
        );
        assert_close(
            atomic_polynomial_product_coefficient(&left, &right, 7)?,
            0.382_9,
        );

        let mixed = atomic_convergence_mix(0.5, 0.3, 0.2)?;
        assert_close(mixed.initial_weight, 0.4);
        assert_close(mixed.final_weight, 0.6);
        assert_close(mixed.previous_error, 0.3);

        let mixed = atomic_convergence_mix(0.2, 0.5, -0.4)?;
        assert_close(mixed.initial_weight, 0.9);
        assert_close(mixed.final_weight, 0.1);
        assert_close(mixed.previous_error, 0.5);

        let mixed = atomic_convergence_mix(0.9, 0.5, 0.4)?;
        assert_close(mixed.initial_weight, 0.099_999_999_999_999_98);
        assert_close(mixed.final_weight, 0.9);
        assert_close(mixed.previous_error, 0.5);

        assert_close(
            thomas_fermi_density_potential(0.45, 29.0, -1.0)?,
            43.097_863_212_551_05,
        );
        assert_close(
            thomas_fermi_density_potential(1.25, 8.0, -2.5)?,
            3.548_014_948_104_207,
        );
        assert_close(thomas_fermi_density_potential(1.25, 0.0, 0.0)?, 0.0);

        let mut occupations = vec![0.0; 41];
        let mut kappas = vec![1; 41];
        occupations[1] = 1.5;
        occupations[4] = 3.0;
        kappas[1] = -1;
        kappas[4] = -3;
        assert_close(atomic_occupation_product(&occupations, &kappas, 4, 4)?, 7.2);
        assert_close(atomic_occupation_product(&occupations, &kappas, 1, 4)?, 4.5);
        Ok(())
    }

    #[test]
    fn atom_coulomb_coefficient_lookups_match_feff_reference() -> Result<(), AtomMathError> {
        let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
            1000.0 * (row + 1) as Real + 10.0 * (column + 1) as Real + channel as Real
        });

        assert_close(
            atomic_direct_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
            2052.0,
        );
        assert_close(
            atomic_direct_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
            2052.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
            5022.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
            5022.0,
        );
        assert_close(
            atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 4, 4)?,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn atom_coulomb_coefficients_match_feff_muatco_reference() -> Result<(), AtomMathError> {
        let kappas = [-1, 1, -2, 2, -3];
        let occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
        let valence_occupations = [0.0, 0.5, 0.0, 0.25, 0.0];
        let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
            kappas: &kappas,
            occupations: &occupations,
            valence_occupations: &valence_occupations,
        })?;

        let expected = [
            [
                [2.0, 3.0, 6.0, 1.0, 8.0],
                [0.5, 2.25, 4.5, 0.75, 6.0],
                [1.000_000_000_000_000_7, 0.0, 6.0, 1.5, 12.0],
                [0.0, 0.0, 0.025_000_000_000_000_026, 0.25, 2.0],
                [0.0, 0.0, 1.199_999_999_999_999_3, 0.0, 12.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.450_000_000_000_000_2,
                    -0.400_000_000_000_000_3,
                    0.0,
                    0.0,
                ],
                [
                    0.100_000_000_000_000_03,
                    0.0,
                    0.096_428_571_428_571_31,
                    0.0,
                    0.0,
                ],
                [
                    0.799_999_999_999_999_5,
                    0.428_571_428_571_428_2,
                    0.342_857_142_857_142_47,
                    0.028_571_428_571_428_536,
                    -0.548_571_428_571_427_9,
                ],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.095_238_095_238_094_86,
                    -0.228_571_428_571_427_8,
                ],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ];

        for (channel, rows) in expected.iter().enumerate() {
            for (row, columns) in rows.iter().enumerate() {
                for (column, &expected) in columns.iter().enumerate() {
                    assert_close_with(coefficients[(row, column, channel)], expected, 1.0e-12);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn atom_breit_coefficients_match_feff_bkmrdf_reference() -> Result<(), AtomMathError> {
        let cases = [
            (
                -1,
                -1,
                1,
                [0.5, 0.333_333_333_333_333_2, 0.5],
                [
                    -0.166_666_666_666_666_69,
                    0.333_333_333_333_333_37,
                    -0.166_666_666_666_666_69,
                ],
            ),
            (
                -1,
                1,
                1,
                [
                    1.500_000_000_000_000_4,
                    1.000_000_000_000_000_2,
                    0.166_666_666_666_666_7,
                ],
                [
                    1.500_000_000_000_000_4,
                    3.000_000_000_000_001,
                    0.833_333_333_333_333_6,
                ],
            ),
            (
                1,
                -2,
                1,
                [
                    0.500_000_000_000_000_2,
                    0.333_333_333_333_334_8,
                    0.100_000_000_000_000_06,
                ],
                [
                    -0.166_666_666_666_667_4,
                    -0.666_666_666_666_669_6,
                    -0.126_666_666_666_667_1,
                ],
            ),
            (
                -2,
                2,
                3,
                [
                    0.116_666_666_666_666_78,
                    0.033_333_333_333_333_36,
                    0.002_380_952_380_952_383,
                ],
                [
                    0.070_000_000_000_000_05,
                    0.420_000_000_000_000_3,
                    0.058_571_428_571_428_62,
                ],
            ),
            (
                -3,
                -3,
                5,
                [
                    0.050_505_050_505_050_37,
                    0.072_150_072_150_071_99,
                    0.050_505_050_505_050_37,
                ],
                [
                    -0.039_281_705_948_372_45,
                    0.078_563_411_896_744_9,
                    -0.039_281_705_948_372_45,
                ],
            ),
            (
                2,
                -4,
                3,
                [
                    0.102_380_952_380_952_13,
                    0.201_587_301_587_301_2,
                    0.254_761_904_761_904_3,
                ],
                [
                    0.238_500_881_834_214_8,
                    0.721_305_114_638_447_2,
                    0.264_320_987_654_320_66,
                ],
            ),
        ];

        for (left, right, rank, magnetic, retarded) in cases {
            let actual = atomic_breit_angular_coefficients(left, right, rank)?;
            for index in 0..3 {
                assert_close(actual.magnetic[index], magnetic[index]);
                assert_close(actual.retarded[index], retarded[index]);
            }
        }
        Ok(())
    }

    #[test]
    fn atom_total_energy_matches_feff_etotal_reference() -> Result<(), AtomMathError> {
        let kappas = [-1, 1, -2, 2];
        let occupations = [2.0, 1.5, 3.0, 0.5];
        let valence_occupations = [0.0, 0.0, 1.0, 0.0];
        let orbital_energies = [-0.7, -0.3, -0.12, -0.05];
        let coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
            0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as Real
        });

        let energy = atomic_total_energy(
            AtomicTotalEnergyInput {
                kappas: &kappas,
                occupations: &occupations,
                valence_occupations: &valence_occupations,
                orbital_energies: &orbital_energies,
                coulomb_coefficients: coefficients.view(),
            },
            |request| {
                Ok(0.0001 * (request.rank + 1) as Real
                    + 0.001 * request.first_left as Real
                    + 0.0002 * request.first_right as Real
                    + 0.00003 * request.second_left as Real
                    + 0.000004 * request.second_right as Real)
            },
        )?;

        assert_close(energy.total, -2.230_065_144_829_932);
        assert_close_with(energy.direct_coulomb, 0.109_629, 1.0e-6);
        assert_close_with(energy.exchange_coulomb, -0.055_702_8, 1.0e-6);
        assert_close_with(energy.magnetic_breit, 0.075_902_3, 1.0e-6);
        assert_close_with(energy.retarded_breit, -0.017_041_4, 1.0e-6);
        Ok(())
    }

    #[test]
    fn atom_overlap_amplitude_reduction_matches_feff_s02at_reference() -> Result<(), AtomMathError>
    {
        let kappas = [-1, -1, 1, 1, -2, -3];
        let occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
        let overlaps = sample_s02at_overlaps();
        let cases = [
            (None, 9.680_452_235_999_996e-3),
            (Some(1), 9.680_452_235_999_996e-3),
            (Some(2), 0.327_600_000_000_000_1),
            (Some(3), 9.680_452_235_999_996e-3),
            (Some(4), 9.020_027_472_527_463e-2),
            (Some(5), 9.680_452_235_999_996e-3),
            (Some(6), 9.680_452_235_999_996e-3),
        ];

        for (hole_orbital_1based, expected) in cases {
            let actual =
                atomic_overlap_amplitude_reduction(AtomicOverlapAmplitudeReductionInput {
                    hole_orbital_1based,
                    kappas: &kappas,
                    occupations: &occupations,
                    overlap_integrals: overlaps.view(),
                })?;
            assert_close(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn atom_helper_kernels_reject_invalid_inputs() {
        assert!(matches!(
            atomic_polynomial_product_coefficient(&[1.0], &[2.0], 2),
            Err(AtomMathError::InvalidPolynomialTerm { .. })
        ));
        assert!(matches!(
            atomic_convergence_mix(0.5, Real::INFINITY, 1.0),
            Err(AtomMathError::NonFiniteScalar {
                field: "current_error",
                ..
            })
        ));
        assert!(matches!(
            thomas_fermi_density_potential(0.0, 1.0, 0.0),
            Err(AtomMathError::NonPositiveRadius { .. })
        ));
        assert!(matches!(
            atomic_occupation_product(&[1.0], &[], 0, 0),
            Err(AtomMathError::OccupationKappaLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_occupation_product(&[1.0], &[0], 0, 0),
            Err(AtomMathError::ZeroKappa)
        ));

        let coefficients = Array3::zeros((2, 2, 1));
        assert!(matches!(
            atomic_direct_coulomb_coefficient(coefficients.view(), 0, 0, 4),
            Err(AtomMathError::CoefficientChannelOutOfRange { .. })
        ));

        let coefficients = Array3::zeros((2, 3, 1));
        assert!(matches!(
            atomic_direct_coulomb_coefficient(coefficients.view(), 1, 2, 0),
            Err(AtomMathError::CoefficientTableShape { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[1],
                occupations: &[],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::OrbitalTableLengthMismatch { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[0],
                occupations: &[1.0],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[1],
                occupations: &[Real::NAN],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::NonFiniteScalar {
                field: "occupation",
                ..
            })
        ));
        assert!(matches!(
            atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
                kappas: &[6],
                occupations: &[2.0],
                valence_occupations: &[0.0],
            }),
            Err(AtomMathError::CoefficientChannelOutOfRange { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(0, -1, 1),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(i32::MIN, -1, 1),
            Err(AtomMathError::InvalidKappa { .. })
        ));
        assert!(matches!(
            atomic_breit_angular_coefficients(1, -1, usize::MAX),
            Err(AtomMathError::BreitRankOutOfRange { .. })
        ));

        let coefficients = Array3::zeros((2, 2, 1));
        let input = AtomicTotalEnergyInput {
            kappas: &[1],
            occupations: &[],
            valence_occupations: &[0.0],
            orbital_energies: &[0.0],
            coulomb_coefficients: coefficients.view(),
        };
        assert!(matches!(
            atomic_total_energy(input, |_| Ok(0.0)),
            Err(AtomMathError::OrbitalTableLengthMismatch { .. })
        ));

        let input = AtomicTotalEnergyInput {
            kappas: &[1],
            occupations: &[1.0],
            valence_occupations: &[0.0],
            orbital_energies: &[0.0],
            coulomb_coefficients: coefficients.view(),
        };
        assert!(matches!(
            atomic_total_energy(input, |_| Ok(Real::NAN)),
            Err(AtomMathError::NonFiniteScalar {
                field: "radial_integral",
                ..
            })
        ));

        let single_overlap = Array2::from_elem((1, 1), 1.0);
        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: Some(0),
            kappas: &[1],
            occupations: &[1.0],
            overlap_integrals: single_overlap.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::HoleOrbitalOutOfRange { .. })
        ));

        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: None,
            kappas: &[1, 1],
            occupations: &[1.0, 1.0],
            overlap_integrals: single_overlap.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::OverlapMatrixShape { .. })
        ));

        let too_many_kappas = [1; 9];
        let too_many_occupations = [1.0; 9];
        let too_many_overlaps = Array2::from_diag(&Array1::from_elem(9, 1.0));
        let input = AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based: None,
            kappas: &too_many_kappas,
            occupations: &too_many_occupations,
            overlap_integrals: too_many_overlaps.view(),
        };
        assert!(matches!(
            atomic_overlap_amplitude_reduction(input),
            Err(AtomMathError::KappaGroupTooLarge { .. })
        ));
    }

    fn sample_s02at_overlaps() -> Array2<Real> {
        let mut overlaps =
            Array2::from_shape_fn((6, 6), |(row, column)| 0.02 * (row + column + 2) as Real);
        for index in 0..6 {
            overlaps[(index, index)] = 1.0;
        }
        overlaps[(0, 1)] = 0.91;
        overlaps[(1, 0)] = 0.91;
        overlaps[(2, 3)] = 0.82;
        overlaps[(3, 2)] = 0.82;
        overlaps
    }
}
