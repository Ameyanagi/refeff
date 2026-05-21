use super::*;

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

pub(super) fn polynomial_product_coefficient_view(
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

pub(super) fn polynomial_product_coefficient_slice_view(
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

pub(super) fn polynomial_product_coefficient_indexed(
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

pub(super) fn one_based_atomic_orbital_index(
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

pub(super) fn doubled_j_from_kappa(kappa: i32) -> Result<i32, AtomMathError> {
    abs_kappa_i32(kappa)?
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AtomMathError::InvalidKappa { kappa })
}

pub(super) fn doubled_j_usize_from_kappa(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(doubled_j_from_kappa(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

pub(super) fn kappa_angular_rank(kappa: i32) -> Result<usize, AtomMathError> {
    doubled_j_usize_from_kappa(kappa)
}

pub(super) fn kappa_abs_usize(kappa: i32) -> Result<usize, AtomMathError> {
    usize::try_from(abs_kappa_i32(kappa)?).map_err(|_| AtomMathError::InvalidKappa { kappa })
}

pub(super) fn atom_usize_to_i32(value: usize) -> Result<i32, AtomMathError> {
    i32::try_from(value).map_err(|_| AtomMathError::CoulombRankOutOfRange { rank: value })
}

pub(super) fn direct_coulomb_coefficient_at(
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

pub(super) fn exchange_coulomb_coefficient_at(
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

pub(super) fn coefficient_channel(
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

pub(super) fn orbital_pair_count(orbital_count: usize) -> Result<usize, AtomMathError> {
    orbital_count
        .checked_mul(orbital_count.saturating_sub(1))
        .map(|count| count / 2)
        .ok_or(AtomMathError::OrbitalPairTableTooLarge { orbital_count })
}

pub(super) fn packed_orbital_pair_index(
    first: usize,
    second: usize,
) -> Result<usize, AtomMathError> {
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

pub(super) fn abs_kappa_i32(kappa: i32) -> Result<i32, AtomMathError> {
    if kappa == 0 {
        return Err(AtomMathError::InvalidKappa { kappa });
    }
    kappa
        .checked_abs()
        .ok_or(AtomMathError::InvalidKappa { kappa })
}
