//! FEFF `KSPACE/strbbdd.f90` structure-factor lattice-sum helpers.

use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::support::{validate_basis, validate_vector, validate_vector_component};
use super::{
    KSpaceAngularTables, KSpaceDirectLatticeSetup, KSpaceDirectLatticeTerms,
    KSpaceDirectLatticeTermsInput, KSpaceEnergyDependentTerms, KSpaceEnergyDependentTermsInput,
    KSpaceError, KSpaceEwaldEnergyTables, KSpaceEwaldEnergyTablesInput,
    KSpaceHarmonicPolynomialsInput, KSpaceInitialEwaldTables, KSpaceQPairGroups,
    KSpaceReciprocalLatticeSetup, KSpaceReciprocalPairPhases, KSpaceReciprocalPairPhasesInput,
    KSpaceStrbbddInput, KSpaceStrsetMatrices, KSpaceStrsetNonRelFromLatticeSumInput,
    KSpaceStrsetNonRelInput, KSpaceStrsetRelFromLatticeSumInput, KSpaceStrsetRelInput, PI2,
};
use crate::{Complex, Real, Vector3, wigner_3j};

const STRHARPOL_ZERO_VECTOR_EPSILON: Real = 1.0e-8;
const STRGAUNT_CUTOFF: Real = 1.0e-8;
const STRCONFRA_INITIAL_IMAX: i32 = 100;
const STRCONFRA_INCREMENT: i32 = 20;
const STRCONFRA_TOLERANCE: Real = 1.0e-10;
const STRCONFRA_MAX_IMAX: i32 = 10_000;
const STRCC_D300_MIN_TERMS: usize = 113;
const STRCC_D300_TOLERANCE: Real = 1.0e-10;
const STRCC_D300_UNDERFLOW: Real = 1.0e-50;
const STRCC_D300_MAX_TERMS: usize = 10_000;
const STRCC_EWALD_TERMS_THRESHOLD: Real = 1.0e14;
const CHANGE_ETA_INCREASE_FACTOR: Real = 1.4;
const CHANGE_ETA_MAX: Real = 3.0;
const CHANGE_ETA_MAX_RETRIES: usize = 32;
/// FEFF `KSPACE/strvecgen.f90` componentwise q-pair grouping tolerance.
pub const KSPACE_Q_PAIR_TOLERANCE: Real = 0.001;

/// Build FEFF `STRGAUNT` and `STRAA` angular tables for KSPACE solvers.
///
/// `angular_lmax` is FEFF `LMAX = NL - 1`; the harmonic-polynomial table is
/// built through `LLMAX = 2 * LMAX`, matching `STRINIT -> STRAA`.
pub fn kspace_angular_tables(
    angular_lmax: usize,
    alat_bohr: Real,
) -> Result<KSpaceAngularTables, KSpaceError> {
    validate_vector_component("alat_bohr", 0, alat_bohr)?;
    if alat_bohr <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "alat_bohr",
            value: alat_bohr,
        });
    }

    let harmonic_lmax =
        angular_lmax
            .checked_mul(2)
            .ok_or(KSpaceError::StructureFactorSizeOverflow {
                name: "harmonic_lmax",
            })?;
    let angular_state_count = harmonic_polynomial_count(angular_lmax)?;
    let qjltab = kspace_qjltab(harmonic_lmax)?;
    let (gaunt_counts, gaunt_indices, gaunt_values) =
        kspace_real_gaunt_tables(angular_lmax, alat_bohr)?;
    let cipwl = kspace_cipwl(harmonic_lmax)?;

    Ok(KSpaceAngularTables {
        angular_lmax,
        harmonic_lmax,
        angular_state_count,
        qjltab,
        gaunt_counts,
        gaunt_indices,
        gaunt_values,
        cipwl,
    })
}

/// Build FEFF `QJLTAB(JJ,LL)` real-harmonic normalization values.
///
/// This is the `STRFUNQJL` table filled by `STRAA` for `LL=0..LLMAX`.
pub fn kspace_qjltab(lmax: usize) -> Result<Array2<Real>, KSpaceError> {
    let size = lmax
        .checked_add(1)
        .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "qjltab" })?;
    let mut qjltab = Array2::<Real>::zeros((size, size));

    for ll in 0..=lmax {
        let two_l_plus_one = ll
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "qjltab" })?
            as Real;
        for jj in 0..=ll {
            let weight = if jj == 0 {
                0.5
            } else {
                inverse_factorial_range_product(ll - jj + 1, ll + jj)
            };
            let value = (weight * two_l_plus_one / PI2).sqrt();
            validate_vector_component("qjltab", ll * size + jj, value)?;
            qjltab[(jj, ll)] = value;
        }
    }

    Ok(qjltab)
}

/// Group equivalent site-pair offsets as FEFF `STRVECGEN` does for `IQQP`.
///
/// Input positions are `(site, xyz)` in the same coordinates FEFF stores as
/// `QX/QY/QZ`. The loop order is FEFF's `IQ=1..NQ`, `JQ=1..NQ`; each new
/// representative keeps the first encountered offset, while later offsets
/// whose three components differ by less than `tolerance` are recorded as
/// equivalent site pairs.
pub fn kspace_q_pair_groups(
    positions: ArrayView2<'_, Real>,
    tolerance: Real,
) -> Result<KSpaceQPairGroups, KSpaceError> {
    validate_q_pair_group_input(positions, tolerance)?;

    let site_count = positions.nrows();
    let mut offsets = Vec::<Vector3>::new();
    let mut groups = Vec::<Vec<[usize; 2]>>::new();
    let mut max_offset_norm: Real = 0.0;

    for row_site in 0..site_count {
        for column_site in 0..site_count {
            let offset = [
                positions[(row_site, 0)] - positions[(column_site, 0)],
                positions[(row_site, 1)] - positions[(column_site, 1)],
                positions[(row_site, 2)] - positions[(column_site, 2)],
            ];
            if let Some(q_pair) = matching_q_pair(&offsets, offset, tolerance) {
                groups[q_pair].push([row_site, column_site]);
            } else {
                max_offset_norm = max_offset_norm.max(dot(offset, offset).sqrt());
                offsets.push(offset);
                groups.push(vec![[row_site, column_site]]);
            }
        }
    }

    let q_pair_count = offsets.len();
    let max_equivalent_count = groups.iter().map(Vec::len).max().unwrap_or(0);
    q_pair_count
        .checked_mul(max_equivalent_count)
        .and_then(|count| count.checked_mul(2))
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "q_pair_sites",
        })?;

    let mut offset_array = Array2::<Real>::zeros((q_pair_count, 3));
    for (q_pair, offset) in offsets.into_iter().enumerate() {
        offset_array[(q_pair, 0)] = offset[0];
        offset_array[(q_pair, 1)] = offset[1];
        offset_array[(q_pair, 2)] = offset[2];
    }

    let mut sites = Array3::<usize>::zeros((q_pair_count, max_equivalent_count, 2));
    let mut counts = Vec::with_capacity(q_pair_count);
    for (q_pair, group) in groups.into_iter().enumerate() {
        counts.push(group.len());
        for (equivalent, [row_site, column_site]) in group.into_iter().enumerate() {
            sites[(q_pair, equivalent, 0)] = row_site;
            sites[(q_pair, equivalent, 1)] = column_site;
        }
    }

    Ok(KSpaceQPairGroups {
        offsets: offset_array,
        sites,
        counts,
        max_offset_norm,
    })
}

/// Enumerate FEFF `STRVECGEN` direct-lattice vectors and build `INDR`.
///
/// This mirrors the direct-space half of `strvecgen.f90` plus the `INDR`
/// filling loop from `straa.f90`: vectors are enumerated over the FEFF
/// `-NUMRH..NUMRH` cube, retained when they fall within `rmax` of any q-pair
/// offset, sorted by increasing real-space length with stable tie order, and
/// then mapped back to each q-pair. The first q-pair skips the sorted zero
/// vector, matching FEFF's `SMAX(1)=SMAX(1)-1` and `I1=2` behavior.
pub fn kspace_direct_lattice_setup(
    direct_basis: [Vector3; 3],
    q_pair_offsets: ArrayView2<'_, Real>,
    rmax: Real,
    max_q_pair_offset_norm: Real,
) -> Result<KSpaceDirectLatticeSetup, KSpaceError> {
    validate_direct_lattice_setup_input(direct_basis, q_pair_offsets, rmax)?;
    validate_vector_component("max_q_pair_offset_norm", 0, max_q_pair_offset_norm)?;
    if max_q_pair_offset_norm < 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "max_q_pair_offset_norm",
            value: max_q_pair_offset_norm,
        });
    }

    let index_radius = direct_lattice_index_radius(direct_basis, rmax, max_q_pair_offset_norm)?;
    let mut retained = Vec::<IntegerLatticeVector>::new();

    for i1 in -index_radius..=index_radius {
        for i2 in -index_radius..=index_radius {
            for i3 in -index_radius..=index_radius {
                let indices = [i1, i2, i3];
                let vector = shifted_vector([0.0, 0.0, 0.0], direct_basis, indices);
                if direct_vector_matches_any_q_pair(vector, q_pair_offsets, rmax) {
                    retained.push(IntegerLatticeVector {
                        indices,
                        distance: dot(vector, vector).sqrt(),
                    });
                }
            }
        }
    }

    retained.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut direct_indices = Array2::<i32>::zeros((retained.len(), 3));
    for (row, vector) in retained.iter().enumerate() {
        direct_indices[(row, 0)] = vector.indices[0];
        direct_indices[(row, 1)] = vector.indices[1];
        direct_indices[(row, 2)] = vector.indices[2];
    }

    let direct_counts = direct_lattice_counts(&retained, direct_basis, q_pair_offsets, rmax);
    let max_direct_count = direct_counts.iter().copied().max().unwrap_or(0);
    let q_pair_count = q_pair_offsets.nrows();
    max_direct_count
        .checked_mul(q_pair_count)
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "direct_index_by_pair",
        })?;
    let mut direct_index_by_pair = Array2::<usize>::zeros((max_direct_count, q_pair_count));
    for q_pair in 0..q_pair_count {
        let mut direct_term = 0;
        let start = if q_pair == 0 { 1 } else { 0 };
        for (direct_index, lattice_vector) in retained.iter().enumerate().skip(start) {
            let vector = shifted_vector([0.0, 0.0, 0.0], direct_basis, lattice_vector.indices);
            if direct_vector_matches_q_pair(vector, q_pair_offsets, q_pair, rmax) {
                direct_index_by_pair[(direct_term, q_pair)] = direct_index;
                direct_term += 1;
            }
        }
    }

    Ok(KSpaceDirectLatticeSetup {
        direct_indices,
        direct_index_by_pair,
        direct_counts,
        index_radius,
    })
}

/// Enumerate FEFF `STRVECGEN` reciprocal-lattice vectors for `STRBBDD`.
///
/// Vectors are retained if any half-Brillouin-zone shift and any point on
/// FEFF's four-point reduced-energy probe range satisfies
/// `(k+G)^2 - EDU <= GMAXSQ`. The final list is stable-sorted by increasing
/// `|G|`, matching the selection-sort behavior in `strvecgen.f90`.
pub fn kspace_reciprocal_lattice_setup(
    reciprocal_basis: [Vector3; 3],
    gmax: Real,
    energy_min_reduced: Real,
    energy_max_reduced: Real,
) -> Result<KSpaceReciprocalLatticeSetup, KSpaceError> {
    validate_reciprocal_lattice_setup_input(
        reciprocal_basis,
        gmax,
        energy_min_reduced,
        energy_max_reduced,
    )?;

    let index_radius = reciprocal_lattice_index_radius(reciprocal_basis, gmax)?;
    let gmax_squared = gmax * gmax;
    let mut retained = Vec::<IntegerLatticeVector>::new();

    for i1 in -index_radius..=index_radius {
        for i2 in -index_radius..=index_radius {
            for i3 in -index_radius..=index_radius {
                let indices = [i1, i2, i3];
                let vector = shifted_vector([0.0, 0.0, 0.0], reciprocal_basis, indices);
                if reciprocal_vector_matches_energy_probe(
                    vector,
                    reciprocal_basis,
                    gmax_squared,
                    energy_min_reduced,
                    energy_max_reduced,
                ) {
                    retained.push(IntegerLatticeVector {
                        indices,
                        distance: dot(vector, vector).sqrt(),
                    });
                }
            }
        }
    }

    retained.sort_by(|left, right| {
        left.distance
            .partial_cmp(&right.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut reciprocal_indices = Array2::<i32>::zeros((retained.len(), 3));
    for (row, vector) in retained.iter().enumerate() {
        reciprocal_indices[(row, 0)] = vector.indices[0];
        reciprocal_indices[(row, 1)] = vector.indices[1];
        reciprocal_indices[(row, 2)] = vector.indices[2];
    }

    Ok(KSpaceReciprocalLatticeSetup {
        reciprocal_indices,
        gmax_squared,
        index_radius,
    })
}

/// Evaluate FEFF `STRHARPOL`, the real harmonic polynomials `r^l Y_lm(r)`.
///
/// The returned vector is in FEFF `(l,m)` order:
/// `l=0,m=0`, then `l=1,m=-1..1`, and so on. FEFF's complex-harmonic branch
/// stops at runtime; this Rust port covers the active real-harmonic branch.
pub fn kspace_harmonic_polynomials(
    input: KSpaceHarmonicPolynomialsInput<'_>,
) -> Result<Array1<Real>, KSpaceError> {
    validate_harmonic_polynomial_input(&input)?;
    harmonic_polynomials(input.vector, input.lmax, input.qjltab)
}

/// Build FEFF `STRAA`'s reciprocal pair phase table, `EXPGNQ`.
///
/// The table includes FEFF's `D1TERM1 = -4*pi/ATVOL` prefactor and the
/// Gaussian Ewald factor for each reciprocal vector. It is the source-backed
/// constructor for the `reciprocal_pair_phases` consumed by `STRBBDD`.
pub fn kspace_reciprocal_pair_phases(
    input: KSpaceReciprocalPairPhasesInput<'_>,
) -> Result<KSpaceReciprocalPairPhases, KSpaceError> {
    validate_reciprocal_pair_phases_input(&input)?;

    let reciprocal_count = input.reciprocal_indices.nrows();
    let q_pair_count = input.q_pair_offsets.nrows();
    reciprocal_count
        .checked_mul(q_pair_count)
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "reciprocal_pair_phases",
        })?;
    let mut reciprocal_pair_phases = Array2::<Complex>::zeros((reciprocal_count, q_pair_count));

    let d1term1 = -2.0 * PI2 / direct_cell_volume_factor(input.direct_basis)?;
    validate_vector_component("d1term1", 0, d1term1)?;

    let mut max_index_abs = 0;
    for reciprocal_index in 0..reciprocal_count {
        let g = [
            input.reciprocal_indices[(reciprocal_index, 0)],
            input.reciprocal_indices[(reciprocal_index, 1)],
            input.reciprocal_indices[(reciprocal_index, 2)],
        ];
        for index in g {
            max_index_abs = max_index_abs.max(index.checked_abs().ok_or(
                KSpaceError::StructureFactorSizeOverflow {
                    name: "reciprocal_indices",
                },
            )?);
        }

        let reciprocal_vector = shifted_vector([0.0, 0.0, 0.0], input.reciprocal_basis, g);
        let gaussian = (-dot(reciprocal_vector, reciprocal_vector) / input.eta).exp();
        let factor = d1term1 * gaussian;
        validate_vector_component("reciprocal_pair_factor", reciprocal_index, factor)?;

        for q_pair in 0..q_pair_count {
            let offset = [
                input.q_pair_offsets[(q_pair, 0)],
                input.q_pair_offsets[(q_pair, 1)],
                input.q_pair_offsets[(q_pair, 2)],
            ];
            let phase = complex_phase(PI2 * dot(reciprocal_vector, offset));
            let value = Complex::new(factor, 0.0) * phase;
            validate_complex(
                "reciprocal_pair_phases",
                reciprocal_index * q_pair_count + q_pair,
                value,
            )?;
            reciprocal_pair_phases[(reciprocal_index, q_pair)] = value;
        }
    }

    Ok(KSpaceReciprocalPairPhases {
        reciprocal_pair_phases,
        max_index_abs,
        d1term1,
    })
}

/// Build FEFF `STRAA`'s base direct lattice table, `QQMLRS`.
///
/// This is the energy-independent DLM2 table produced before `STRCC`
/// multiplies in `IILERS`. The accepted direct-vector list and `INDR` mapping
/// are supplied by `kspace_direct_lattice_setup`.
pub fn kspace_direct_lattice_terms(
    input: KSpaceDirectLatticeTermsInput<'_>,
) -> Result<KSpaceDirectLatticeTerms, KSpaceError> {
    let shape = validate_direct_lattice_terms_input(&input)?;
    let mut direct_terms =
        Array3::<Complex>::zeros((shape.mml_count, shape.max_direct_terms, shape.q_pair_count));
    let mut radial_terms = Array4::<Real>::zeros((
        input.j22max + 1,
        input.lmax + 1,
        shape.max_direct_terms,
        shape.q_pair_count,
    ));

    let q1 = -0.5 * (input.eta / (PI2 / 2.0)).sqrt();
    validate_vector_component("direct_lattice_q1", 0, q1)?;
    let q_pair_terms = (0..shape.q_pair_count)
        .into_par_iter()
        .map(
            |q_pair| -> Result<(Array2<Complex>, Array3<Real>, i32), KSpaceError> {
                let mut pair_direct_terms =
                    Array2::<Complex>::zeros((shape.mml_count, shape.max_direct_terms));
                let mut pair_radial_terms = Array3::<Real>::zeros((
                    input.j22max + 1,
                    input.lmax + 1,
                    shape.max_direct_terms,
                ));
                let mut pair_max_index_abs = 0;
                for direct_term in 0..input.direct_counts[q_pair] {
                    let direct_index = input.direct_index_by_pair[(direct_term, q_pair)];
                    let r = [
                        input.direct_indices[(direct_index, 0)],
                        input.direct_indices[(direct_index, 1)],
                        input.direct_indices[(direct_index, 2)],
                    ];
                    for index in r {
                        pair_max_index_abs = pair_max_index_abs.max(index.checked_abs().ok_or(
                            KSpaceError::StructureFactorSizeOverflow {
                                name: "direct_indices",
                            },
                        )?);
                    }

                    let lattice_vector = shifted_vector([0.0, 0.0, 0.0], input.direct_basis, r);
                    let offset = [
                        input.q_pair_offsets[(q_pair, 0)],
                        input.q_pair_offsets[(q_pair, 1)],
                        input.q_pair_offsets[(q_pair, 2)],
                    ];
                    let scaled_delta = [
                        PI2 * (lattice_vector[0] - offset[0]),
                        PI2 * (lattice_vector[1] - offset[1]),
                        PI2 * (lattice_vector[2] - offset[2]),
                    ];
                    let hp = harmonic_polynomials(scaled_delta, input.lmax, input.qjltab)?;
                    let radial_argument = dot(scaled_delta, scaled_delta) * input.eta / 4.0;
                    validate_vector_component(
                        "direct_lattice_radial_argument",
                        direct_term,
                        radial_argument,
                    )?;
                    let gaussian = (-radial_argument).exp();
                    validate_vector_component("direct_lattice_gaussian", direct_term, gaussian)?;

                    let mut angular_factor = 1.0 / (-input.eta / 2.0);
                    let mut mml = 0;
                    for angular_momentum in 0..=input.lmax {
                        angular_factor *= -input.eta / 2.0;
                        let factor = q1 * angular_factor * gaussian;
                        validate_vector_component(
                            "direct_lattice_factor",
                            angular_momentum,
                            factor,
                        )?;
                        for _magnetic in 0..(2 * angular_momentum + 1) {
                            let value = Complex::new(factor * hp[mml], 0.0);
                            validate_complex("direct_terms", mml, value)?;
                            pair_direct_terms[(mml, direct_term)] = value;
                            mml += 1;
                        }

                        let mut radial_factor = 1.0;
                        for j22 in 0..=input.j22max {
                            let aa = angular_momentum as Real - j22 as Real + 0.5;
                            let value = strconfra(aa, radial_argument)? * radial_factor;
                            validate_vector_component("radial_terms", j22, value)?;
                            pair_radial_terms[(j22, angular_momentum, direct_term)] = value;
                            radial_factor /= input.eta * (j22 as Real + 1.0);
                        }
                    }
                }
                Ok((pair_direct_terms, pair_radial_terms, pair_max_index_abs))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let mut max_index_abs = 0;
    for (q_pair, (pair_direct_terms, pair_radial_terms, pair_max_index_abs)) in
        q_pair_terms.iter().enumerate()
    {
        max_index_abs = max_index_abs.max(*pair_max_index_abs);
        for direct_term in 0..input.direct_counts[q_pair] {
            for mml in 0..shape.mml_count {
                direct_terms[(mml, direct_term, q_pair)] = pair_direct_terms[(mml, direct_term)];
            }
            for angular_momentum in 0..=input.lmax {
                for j22 in 0..=input.j22max {
                    radial_terms[(j22, angular_momentum, direct_term, q_pair)] =
                        pair_radial_terms[(j22, angular_momentum, direct_term)];
                }
            }
        }
    }

    Ok(KSpaceDirectLatticeTerms {
        direct_terms,
        radial_terms,
        max_index_abs,
        q1,
    })
}

/// Build FEFF `STRCC` energy-dependent KSPACE tables for one reduced energy.
///
/// The returned `direct_terms`, `d1term3`, and `d300` are the values consumed by
/// `STRBBDD` after FEFF's `IILERS` multiplication. This helper performs the
/// fixed-`ETA` calculation and reports whether FEFF's Ewald threshold would
/// request a `change_eta` rerun.
pub fn kspace_energy_dependent_terms(
    input: KSpaceEnergyDependentTermsInput<'_>,
) -> Result<KSpaceEnergyDependentTerms, KSpaceError> {
    let shape = validate_energy_dependent_terms_input(&input)?;
    let wave_number = input.energy.sqrt();
    validate_complex("reduced_wave_number", 0, wave_number)?;
    if input.lmax > 0 && wave_number == Complex::new(0.0, 0.0) {
        return Err(KSpaceError::DegenerateStructureFactorValue {
            name: "reduced_wave_number",
            index: 0,
        });
    }

    let mut d1term3 = Array1::<Complex>::zeros(input.lmax + 1);
    let mut epwmllh = Array1::<Complex>::zeros(input.lmax + 1);
    d1term3[0] = (input.energy / input.eta).exp();
    epwmllh[0] = Complex::new(1.0, 0.0);
    validate_complex("d1term3", 0, d1term3[0])?;
    for angular_momentum in 1..=input.lmax {
        epwmllh[angular_momentum] =
            epwmllh[angular_momentum - 1] * Complex::new(0.0, 1.0) / wave_number;
        d1term3[angular_momentum] = d1term3[angular_momentum - 1] / wave_number;
        validate_complex("epwmllh", angular_momentum, epwmllh[angular_momentum])?;
        validate_complex("d1term3", angular_momentum, d1term3[angular_momentum])?;
    }

    let mut direct_multipliers =
        Array3::<Complex>::zeros((input.lmax + 1, shape.max_direct_terms, shape.q_pair_count));
    for q_pair in 0..shape.q_pair_count {
        for direct_term in 0..input.direct_counts[q_pair] {
            for angular_momentum in 0..=input.lmax {
                let mut energy_power = Complex::new(1.0, 0.0);
                let mut multiplier = Complex::new(0.0, 0.0);
                for j22 in 0..shape.radial_order_count {
                    multiplier += energy_power
                        * input.radial_terms[(j22, angular_momentum, direct_term, q_pair)];
                    energy_power *= input.energy;
                }
                multiplier *= epwmllh[angular_momentum];
                validate_complex("direct_multipliers", angular_momentum, multiplier)?;
                direct_multipliers[(angular_momentum, direct_term, q_pair)] = multiplier;
            }
        }
    }

    let mut direct_terms =
        Array3::<Complex>::zeros((shape.mml_count, shape.max_direct_terms, shape.q_pair_count));
    for q_pair in 0..shape.q_pair_count {
        for direct_term in 0..input.direct_counts[q_pair] {
            for mml in 0..shape.mml_count {
                let angular_momentum = angular_momentum_for_mml(mml);
                let value = input.base_direct_terms[(mml, direct_term, q_pair)]
                    * direct_multipliers[(angular_momentum, direct_term, q_pair)];
                validate_complex("energy_direct_terms", mml, value)?;
                direct_terms[(mml, direct_term, q_pair)] = value;
            }
        }
    }

    let d300 = strcc_d300(input.energy, input.eta)?;
    let ewald_terms_exceed_threshold =
        strcc_terms_exceed_threshold(d300, direct_terms.view(), d1term3.view());

    Ok(KSpaceEnergyDependentTerms {
        direct_terms,
        direct_multipliers,
        d1term3,
        d300,
        ewald_terms_exceed_threshold,
    })
}

/// Build one-energy KSPACE Ewald tables, applying FEFF `change_eta` retries.
///
/// When `STRCC` reports oversized Ewald terms, FEFF multiplies `ETA` by 1.4,
/// reruns `STRINIT`, and then reruns `STRCC` for the same energy. This helper
/// mirrors that policy for the Rust table constructors and returns the final
/// source-backed tables for `STRBBDD`.
pub fn kspace_ewald_energy_tables(
    input: KSpaceEwaldEnergyTablesInput<'_>,
) -> Result<KSpaceEwaldEnergyTables, KSpaceError> {
    validate_ewald_energy_tables_input(&input)?;

    let reciprocal_pair_phases = kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
        direct_basis: input.direct_basis,
        reciprocal_basis: input.reciprocal_basis,
        reciprocal_indices: input.reciprocal_indices,
        q_pair_offsets: input.q_pair_offsets,
        eta: input.initial_eta,
    })?;
    let direct_lattice_terms = kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
        direct_basis: input.direct_basis,
        direct_indices: input.direct_indices,
        direct_index_by_pair: input.direct_index_by_pair,
        direct_counts: input.direct_counts,
        q_pair_offsets: input.q_pair_offsets,
        lmax: input.lmax,
        j22max: input.j22max,
        qjltab: input.qjltab,
        eta: input.initial_eta,
    })?;
    let initial_tables = KSpaceInitialEwaldTables {
        eta: input.initial_eta,
        reciprocal_pair_phases,
        direct_lattice_terms,
    };
    kspace_ewald_energy_tables_from_initial(input, &initial_tables)
}

/// Build one-energy `STRCC` tables from reusable initial-`ETA` `STRAA` tables.
///
/// The supplied reciprocal phases and base direct terms are used only for the
/// initial `ETA`. If FEFF's Ewald threshold requests `change_eta`, both are
/// rebuilt at every retry `ETA` exactly as in [`kspace_ewald_energy_tables`].
pub fn kspace_ewald_energy_tables_from_initial(
    input: KSpaceEwaldEnergyTablesInput<'_>,
    initial_tables: &KSpaceInitialEwaldTables,
) -> Result<KSpaceEwaldEnergyTables, KSpaceError> {
    validate_ewald_energy_tables_input(&input)?;
    validate_vector_component("initial_ewald_eta", 0, initial_tables.eta)?;
    if initial_tables.eta != input.initial_eta {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "initial_ewald_eta",
            value: initial_tables.eta,
        });
    }

    let mut eta = input.initial_eta;
    let mut retry_count = 0usize;
    loop {
        let (reciprocal_pair_phases, direct_lattice_terms) = if retry_count == 0 {
            (
                initial_tables.reciprocal_pair_phases.clone(),
                initial_tables.direct_lattice_terms.clone(),
            )
        } else {
            (
                kspace_reciprocal_pair_phases(KSpaceReciprocalPairPhasesInput {
                    direct_basis: input.direct_basis,
                    reciprocal_basis: input.reciprocal_basis,
                    reciprocal_indices: input.reciprocal_indices,
                    q_pair_offsets: input.q_pair_offsets,
                    eta,
                })?,
                kspace_direct_lattice_terms(KSpaceDirectLatticeTermsInput {
                    direct_basis: input.direct_basis,
                    direct_indices: input.direct_indices,
                    direct_index_by_pair: input.direct_index_by_pair,
                    direct_counts: input.direct_counts,
                    q_pair_offsets: input.q_pair_offsets,
                    lmax: input.lmax,
                    j22max: input.j22max,
                    qjltab: input.qjltab,
                    eta,
                })?,
            )
        };
        let energy_dependent_terms =
            kspace_energy_dependent_terms(KSpaceEnergyDependentTermsInput {
                energy: input.energy,
                eta,
                lmax: input.lmax,
                base_direct_terms: direct_lattice_terms.direct_terms.view(),
                radial_terms: direct_lattice_terms.radial_terms.view(),
                direct_counts: input.direct_counts,
            })?;

        if !energy_dependent_terms.ewald_terms_exceed_threshold {
            return Ok(KSpaceEwaldEnergyTables {
                eta,
                retry_count,
                reciprocal_pair_phases,
                direct_lattice_terms,
                energy_dependent_terms,
            });
        }

        retry_count =
            retry_count
                .checked_add(1)
                .ok_or(KSpaceError::StructureFactorSizeOverflow {
                    name: "change_eta_retries",
                })?;
        if retry_count > CHANGE_ETA_MAX_RETRIES {
            return Err(KSpaceError::StructureFactorSizeOverflow {
                name: "change_eta_retries",
            });
        }
        eta = change_eta_next(eta)?;
    }
}

/// Evaluate FEFF `STRBBDD` reciprocal/direct lattice sums for one k-point.
///
/// This helper intentionally takes the expensive setup products explicitly:
/// reciprocal/direct lattice lists, pair phase tables, direct terms, and the
/// `QJLTAB` normalization table. It mirrors `strbbdd.f90`'s accumulation order
/// and phase conventions while leaving the surrounding `strinit` setup routine
/// to its own Rust port.
pub fn kspace_strbbdd_lattice_sum(
    input: KSpaceStrbbddInput<'_>,
) -> Result<Array2<Complex>, KSpaceError> {
    let shape = validate_input(&input)?;
    let mut dllmmke = Array2::<Complex>::zeros((shape.mml_count, shape.q_pair_count));

    accumulate_reciprocal_lattice_sum(&input, shape, &mut dllmmke)?;
    apply_missing_reciprocal_pair_phase(&input, &mut dllmmke)?;
    accumulate_direct_lattice_sum(&input, shape, &mut dllmmke)?;

    if shape.mml_count > 0 && shape.q_pair_count > 0 {
        validate_complex("d300", 0, input.d300)?;
        dllmmke[(0, 0)] += input.d300;
    }

    validate_complex_matrix("strbbdd_result", dllmmke.view())?;
    Ok(dllmmke)
}

/// Compose FEFF `STRBBDD` and non-relativistic `STRSET` for one k-point.
///
/// This is the source-backed path used by `structurefactor.f90` before the
/// final FEFF-basis conversion: evaluate the lattice sums into `DLLMMKE`, then
/// contract them with Gaunt coefficients into SPRKKR-basis `TAUKINV`.
pub fn kspace_strset_non_rel_from_lattice_sum(
    input: KSpaceStrsetNonRelFromLatticeSumInput<'_>,
) -> Result<KSpaceStrsetMatrices, KSpaceError> {
    let KSpaceStrsetNonRelFromLatticeSumInput {
        lattice_sum,
        angular_state_count,
        q_pair_sites,
        q_pair_counts,
        site_offsets,
        site_state_counts,
        gaunt_counts,
        gaunt_indices,
        gaunt_values,
        cipwl,
        wave_number,
    } = input;
    let dllmmke = kspace_strbbdd_lattice_sum(lattice_sum)?;
    let taukinv = kspace_strset_non_relativistic(KSpaceStrsetNonRelInput {
        angular_state_count,
        dllmmke: dllmmke.view(),
        q_pair_sites,
        q_pair_counts,
        site_offsets,
        site_state_counts,
        gaunt_counts,
        gaunt_indices,
        gaunt_values,
        cipwl,
        wave_number,
    })?;

    Ok(KSpaceStrsetMatrices { dllmmke, taukinv })
}

/// Compose FEFF `STRBBDD` and relativistic `STRSET` for one k-point.
///
/// This mirrors the `IREL >= 2` branch of `strset.f90` after the common
/// lattice-sum step, returning both `DLLMMKE` and the transformed
/// SPRKKR-basis `TAUKINV` matrix for downstream BAND structure factors.
pub fn kspace_strset_rel_from_lattice_sum(
    input: KSpaceStrsetRelFromLatticeSumInput<'_>,
) -> Result<KSpaceStrsetMatrices, KSpaceError> {
    let KSpaceStrsetRelFromLatticeSumInput {
        lattice_sum,
        angular_state_count,
        q_pair_sites,
        q_pair_counts,
        site_offsets,
        site_state_counts,
        gaunt_counts,
        gaunt_indices,
        gaunt_values,
        cipwl,
        rel_component_counts,
        rel_component_indices,
        rel_component_coefficients,
        wave_number,
    } = input;
    let dllmmke = kspace_strbbdd_lattice_sum(lattice_sum)?;
    let taukinv = kspace_strset_relativistic(KSpaceStrsetRelInput {
        angular_state_count,
        dllmmke: dllmmke.view(),
        q_pair_sites,
        q_pair_counts,
        site_offsets,
        site_state_counts,
        gaunt_counts,
        gaunt_indices,
        gaunt_values,
        cipwl,
        rel_component_counts,
        rel_component_indices,
        rel_component_coefficients,
        wave_number,
    })?;

    Ok(KSpaceStrsetMatrices { dllmmke, taukinv })
}

/// Evaluate FEFF `STRSET`'s non-relativistic Gaunt contraction (`IREL < 2`).
///
/// The returned matrix is FEFF `TAUKINV`: the contracted structure constants
/// are stored with FEFF's leading minus sign, the first representative q-pair
/// receives the diagonal `-i*p` term, and equivalent q-pair blocks are copied
/// from their representative block.
pub fn kspace_strset_non_relativistic(
    input: KSpaceStrsetNonRelInput<'_>,
) -> Result<Array2<Complex>, KSpaceError> {
    let shape = validate_strset_non_rel_input(&input)?;
    let mut taukinv = Array2::<Complex>::zeros((shape.matrix_order, shape.matrix_order));

    for q_pair in 0..shape.q_pair_count {
        contract_representative_q_pair(&input, shape, q_pair, &mut taukinv)?;
        if q_pair == 0 {
            subtract_first_pair_diagonal(&input, shape, q_pair, &mut taukinv)?;
        }
        copy_equivalent_q_pair_blocks(&input, q_pair, &mut taukinv)?;
    }

    validate_complex_matrix("strset_taukinv", taukinv.view())?;
    Ok(taukinv)
}

/// Evaluate FEFF `STRSET`'s relativistic transform (`IREL >= 2`).
///
/// FEFF first builds a representative non-relativistic `GNR` block from
/// `DLLMMKE`, adds the first-pair `+i*p` diagonal in that non-rel basis, then
/// applies `SRREL/IRREL/NRREL` and stores the result as `TAUKINV = -G`.
/// Equivalent q-pair blocks are copied from the representative block.
pub fn kspace_strset_relativistic(
    input: KSpaceStrsetRelInput<'_>,
) -> Result<Array2<Complex>, KSpaceError> {
    let shape = validate_strset_rel_input(&input)?;
    let mut taukinv = Array2::<Complex>::zeros((shape.matrix_order, shape.matrix_order));
    let mut gnr = Array2::<Complex>::zeros((input.angular_state_count, input.angular_state_count));

    for q_pair in 0..shape.q_pair_count {
        build_representative_gnr(&input, q_pair, &mut gnr)?;
        if q_pair == 0 {
            add_first_pair_gnr_diagonal(&input, &mut gnr)?;
        }
        transform_representative_rel_q_pair(&input, q_pair, &gnr, &mut taukinv)?;
        copy_equivalent_rel_q_pair_blocks(&input, q_pair, &mut taukinv)?;
    }

    validate_complex_matrix("strset_rel_taukinv", taukinv.view())?;
    Ok(taukinv)
}

#[derive(Debug, Clone, Copy)]
struct StructureFactorShape {
    mml_count: usize,
    q_pair_count: usize,
    reciprocal_count: usize,
    max_direct_terms: usize,
    radial_order_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct StrsetNonRelShape {
    angular_pair_count: usize,
    matrix_order: usize,
    q_pair_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct StrsetRelShape {
    matrix_order: usize,
    q_pair_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct IntegerLatticeVector {
    indices: [i32; 3],
    distance: Real,
}

fn contract_representative_q_pair(
    input: &KSpaceStrsetNonRelInput<'_>,
    shape: StrsetNonRelShape,
    q_pair: usize,
    taukinv: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let primary_row_site = input.q_pair_sites[(q_pair, 0, 0)];
    let primary_column_site = input.q_pair_sites[(q_pair, 0, 1)];
    let row_offset = input.site_offsets[primary_row_site];
    let column_offset = input.site_offsets[primary_column_site];
    let mut gaunt_cursor = 0;
    let mut angular_pair = 0;

    for lm1 in 0..input.angular_state_count {
        for lm2 in 0..=lm1 {
            let mut csum = Complex::new(0.0, 0.0);
            for _ in 0..input.gaunt_counts[angular_pair] {
                let lm3 = input.gaunt_indices[gaunt_cursor];
                csum += input.dllmmke[(lm3, q_pair)] * input.gaunt_values[gaunt_cursor];
                gaunt_cursor += 1;
            }

            let ratio = input.cipwl[lm1] / input.cipwl[lm2];
            validate_complex("strset_cipwl_ratio", angular_pair, ratio)?;
            taukinv[(row_offset + lm1, column_offset + lm2)] = -csum * ratio;
            taukinv[(row_offset + lm2, column_offset + lm1)] = -csum / ratio;
            angular_pair += 1;
        }
    }

    debug_assert_eq!(angular_pair, shape.angular_pair_count);
    debug_assert_eq!(gaunt_cursor, input.gaunt_counts.iter().sum::<usize>());
    Ok(())
}

fn subtract_first_pair_diagonal(
    input: &KSpaceStrsetNonRelInput<'_>,
    _shape: StrsetNonRelShape,
    q_pair: usize,
    taukinv: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let primary_row_site = input.q_pair_sites[(q_pair, 0, 0)];
    let primary_column_site = input.q_pair_sites[(q_pair, 0, 1)];
    let row_offset = input.site_offsets[primary_row_site];
    let column_offset = input.site_offsets[primary_column_site];
    let diagonal_shift = Complex::new(0.0, 1.0) * input.wave_number;
    validate_complex("strset_diagonal_shift", 0, diagonal_shift)?;
    for lm in 0..input.angular_state_count {
        taukinv[(row_offset + lm, column_offset + lm)] -= diagonal_shift;
    }
    Ok(())
}

fn copy_equivalent_q_pair_blocks(
    input: &KSpaceStrsetNonRelInput<'_>,
    q_pair: usize,
    taukinv: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let primary_row_site = input.q_pair_sites[(q_pair, 0, 0)];
    let primary_column_site = input.q_pair_sites[(q_pair, 0, 1)];
    let primary_row_offset = input.site_offsets[primary_row_site];
    let primary_column_offset = input.site_offsets[primary_column_site];
    let row_count = input.site_state_counts[primary_row_site];
    let column_count = input.site_state_counts[primary_column_site];

    for equivalent in 1..input.q_pair_counts[q_pair] {
        let row_site = input.q_pair_sites[(q_pair, equivalent, 0)];
        let column_site = input.q_pair_sites[(q_pair, equivalent, 1)];
        let row_offset = input.site_offsets[row_site];
        let column_offset = input.site_offsets[column_site];
        for column_state in 0..column_count {
            for row_state in 0..row_count {
                taukinv[(row_offset + row_state, column_offset + column_state)] = taukinv[(
                    primary_row_offset + row_state,
                    primary_column_offset + column_state,
                )];
            }
        }
    }
    Ok(())
}

fn build_representative_gnr(
    input: &KSpaceStrsetRelInput<'_>,
    q_pair: usize,
    gnr: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    gnr.fill(Complex::new(0.0, 0.0));
    let mut gaunt_cursor = 0;
    let mut angular_pair = 0;

    for lm1 in 0..input.angular_state_count {
        for lm2 in 0..=lm1 {
            let mut csum = Complex::new(0.0, 0.0);
            for _ in 0..input.gaunt_counts[angular_pair] {
                let lm3 = input.gaunt_indices[gaunt_cursor];
                csum += input.dllmmke[(lm3, q_pair)] * input.gaunt_values[gaunt_cursor];
                gaunt_cursor += 1;
            }

            let ratio = input.cipwl[lm1] / input.cipwl[lm2];
            validate_complex("strset_rel_cipwl_ratio", angular_pair, ratio)?;
            gnr[(lm1, lm2)] = csum * ratio;
            gnr[(lm2, lm1)] = csum / ratio;
            angular_pair += 1;
        }
    }

    debug_assert_eq!(
        angular_pair,
        triangular_pair_count(input.angular_state_count)?
    );
    debug_assert_eq!(gaunt_cursor, input.gaunt_counts.iter().sum::<usize>());
    Ok(())
}

fn add_first_pair_gnr_diagonal(
    input: &KSpaceStrsetRelInput<'_>,
    gnr: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let diagonal_shift = Complex::new(0.0, 1.0) * input.wave_number;
    validate_complex("strset_rel_diagonal_shift", 0, diagonal_shift)?;
    for lm in 0..input.angular_state_count {
        gnr[(lm, lm)] += diagonal_shift;
    }
    Ok(())
}

fn transform_representative_rel_q_pair(
    input: &KSpaceStrsetRelInput<'_>,
    q_pair: usize,
    gnr: &Array2<Complex>,
    taukinv: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let primary_row_site = input.q_pair_sites[(q_pair, 0, 0)];
    let primary_column_site = input.q_pair_sites[(q_pair, 0, 1)];
    let row_offset = input.site_offsets[primary_row_site];
    let column_offset = input.site_offsets[primary_column_site];
    let row_count = input.site_state_counts[primary_row_site];
    let column_count = input.site_state_counts[primary_column_site];

    for column_state in 0..column_count {
        for row_state in 0..row_count {
            let mut csum1 = Complex::new(0.0, 0.0);
            for spin in 0..2 {
                for row_component in 0..input.rel_component_counts[(spin, row_state)] {
                    let row_lm = input.rel_component_indices[(row_component, spin, row_state)];
                    let mut csum2 = Complex::new(0.0, 0.0);
                    for column_component in 0..input.rel_component_counts[(spin, column_state)] {
                        let column_lm =
                            input.rel_component_indices[(column_component, spin, column_state)];
                        csum2 += gnr[(row_lm, column_lm)]
                            * input.rel_component_coefficients
                                [(column_component, spin, column_state)];
                    }
                    csum1 += input.rel_component_coefficients[(row_component, spin, row_state)]
                        .conj()
                        * csum2;
                }
            }
            validate_complex(
                "strset_rel_transform",
                row_state * column_count + column_state,
                csum1,
            )?;
            taukinv[(row_offset + row_state, column_offset + column_state)] = -csum1;
        }
    }
    Ok(())
}

fn copy_equivalent_rel_q_pair_blocks(
    input: &KSpaceStrsetRelInput<'_>,
    q_pair: usize,
    taukinv: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    let primary_row_site = input.q_pair_sites[(q_pair, 0, 0)];
    let primary_column_site = input.q_pair_sites[(q_pair, 0, 1)];
    let primary_row_offset = input.site_offsets[primary_row_site];
    let primary_column_offset = input.site_offsets[primary_column_site];
    let row_count = input.site_state_counts[primary_row_site];
    let column_count = input.site_state_counts[primary_column_site];

    for equivalent in 1..input.q_pair_counts[q_pair] {
        let row_site = input.q_pair_sites[(q_pair, equivalent, 0)];
        let column_site = input.q_pair_sites[(q_pair, equivalent, 1)];
        let row_offset = input.site_offsets[row_site];
        let column_offset = input.site_offsets[column_site];
        for column_state in 0..column_count {
            for row_state in 0..row_count {
                taukinv[(row_offset + row_state, column_offset + column_state)] = taukinv[(
                    primary_row_offset + row_state,
                    primary_column_offset + column_state,
                )];
            }
        }
    }
    Ok(())
}

fn accumulate_reciprocal_lattice_sum(
    input: &KSpaceStrbbddInput<'_>,
    shape: StructureFactorShape,
    dllmmke: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    if shape.reciprocal_count == 0 || shape.mml_count == 0 {
        return Ok(());
    }

    let reciprocal_power_base = [
        (-2.0 * dot(input.reciprocal_basis[0], input.k) / input.eta).exp(),
        (-2.0 * dot(input.reciprocal_basis[1], input.k) / input.eta).exp(),
        (-2.0 * dot(input.reciprocal_basis[2], input.k) / input.eta).exp(),
    ];
    let f0 = (-dot(input.k, input.k) / input.eta).exp();

    for reciprocal_index in 0..shape.reciprocal_count {
        let g = [
            input.reciprocal_indices[(reciprocal_index, 0)],
            input.reciprocal_indices[(reciprocal_index, 1)],
            input.reciprocal_indices[(reciprocal_index, 2)],
        ];
        let kn = shifted_vector(input.k, input.reciprocal_basis, g);
        let denom = Complex::new(dot(kn, kn), 0.0) - input.energy;
        if denom.re <= input.gmax_squared {
            let hp = harmonic_polynomials(kn, input.lmax, input.qjltab)?;
            let ex2kgn = reciprocal_power_base[0].powi(g[0])
                * reciprocal_power_base[1].powi(g[1])
                * reciprocal_power_base[2].powi(g[2]);
            let f1 = Complex::new(f0 * ex2kgn, 0.0) / denom;
            validate_complex("strbbdd_reciprocal_factor", reciprocal_index, f1)?;

            for q_pair in 0..shape.q_pair_count {
                let reciprocal_pair_phase =
                    input.reciprocal_pair_phases[(reciprocal_index, q_pair)];
                for mml in 0..shape.mml_count {
                    let angular_momentum = angular_momentum_for_mml(mml);
                    dllmmke[(mml, q_pair)] +=
                        f1 * reciprocal_pair_phase * input.d1term3[angular_momentum] * hp[mml];
                }
            }
        }
    }

    Ok(())
}

fn apply_missing_reciprocal_pair_phase(
    input: &KSpaceStrbbddInput<'_>,
    dllmmke: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    for q_pair in 1..dllmmke.ncols() {
        let offset = [
            input.q_pair_offsets[(q_pair, 0)],
            input.q_pair_offsets[(q_pair, 1)],
            input.q_pair_offsets[(q_pair, 2)],
        ];
        let phase = complex_phase(PI2 * dot(input.k, offset));
        validate_complex("strbbdd_q_pair_phase", q_pair, phase)?;
        for mml in 0..dllmmke.nrows() {
            dllmmke[(mml, q_pair)] *= phase;
        }
    }
    Ok(())
}

fn accumulate_direct_lattice_sum(
    input: &KSpaceStrbbddInput<'_>,
    shape: StructureFactorShape,
    dllmmke: &mut Array2<Complex>,
) -> Result<(), KSpaceError> {
    if shape.mml_count == 0 {
        return Ok(());
    }

    let direct_power_base = [
        complex_phase(PI2 * dot(input.direct_basis[0], input.k)),
        complex_phase(PI2 * dot(input.direct_basis[1], input.k)),
        complex_phase(PI2 * dot(input.direct_basis[2], input.k)),
    ];

    for q_pair in 0..shape.q_pair_count {
        for direct_term in 0..input.direct_counts[q_pair] {
            let direct_index = input.direct_index_by_pair[(direct_term, q_pair)];
            let r = [
                input.direct_indices[(direct_index, 0)],
                input.direct_indices[(direct_index, 1)],
                input.direct_indices[(direct_index, 2)],
            ];
            let phase = direct_power_base[0].powi(r[0])
                * direct_power_base[1].powi(r[1])
                * direct_power_base[2].powi(r[2]);
            validate_complex("strbbdd_direct_phase", direct_index, phase)?;
            for mml in 0..shape.mml_count {
                dllmmke[(mml, q_pair)] += phase * input.direct_terms[(mml, direct_term, q_pair)];
            }
        }
    }

    Ok(())
}

fn validate_input(input: &KSpaceStrbbddInput<'_>) -> Result<StructureFactorShape, KSpaceError> {
    validate_vector("k", input.k)?;
    validate_vector_component("eta", 0, input.eta)?;
    validate_vector_component("gmax_squared", 0, input.gmax_squared)?;
    if input.eta <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: input.eta,
        });
    }
    validate_complex("energy", 0, input.energy)?;
    validate_basis(input.reciprocal_basis)?;
    validate_basis(input.direct_basis)?;
    validate_harmonic_shape("qjltab", input.qjltab, input.lmax)?;
    validate_real_matrix("qjltab", input.qjltab)?;
    let mml_count = harmonic_polynomial_count(input.lmax)?;

    let reciprocal_count = input.reciprocal_indices.nrows();
    validate_matrix_shape(
        "reciprocal_indices",
        input.reciprocal_indices.nrows(),
        input.reciprocal_indices.ncols(),
        reciprocal_count,
        3,
    )?;

    let q_pair_count = input.q_pair_offsets.nrows();
    validate_matrix_shape(
        "q_pair_offsets",
        input.q_pair_offsets.nrows(),
        input.q_pair_offsets.ncols(),
        q_pair_count,
        3,
    )?;
    validate_real_matrix("q_pair_offsets", input.q_pair_offsets)?;

    validate_matrix_shape(
        "reciprocal_pair_phases",
        input.reciprocal_pair_phases.nrows(),
        input.reciprocal_pair_phases.ncols(),
        reciprocal_count,
        q_pair_count,
    )?;
    validate_complex_matrix("reciprocal_pair_phases", input.reciprocal_pair_phases)?;

    let direct_count = input.direct_indices.nrows();
    validate_matrix_shape(
        "direct_indices",
        input.direct_indices.nrows(),
        input.direct_indices.ncols(),
        direct_count,
        3,
    )?;

    let max_direct_terms = input.direct_index_by_pair.nrows();
    validate_matrix_shape(
        "direct_index_by_pair",
        input.direct_index_by_pair.nrows(),
        input.direct_index_by_pair.ncols(),
        max_direct_terms,
        q_pair_count,
    )?;
    validate_exact_length("direct_counts", input.direct_counts.len(), q_pair_count)?;

    let direct_terms_shape = input.direct_terms.dim();
    validate_cube_shape(
        "direct_terms",
        direct_terms_shape,
        (direct_terms_shape.0, max_direct_terms, q_pair_count),
    )?;
    validate_complex_cube("direct_terms", input.direct_terms)?;

    if direct_terms_shape.0 != mml_count {
        validate_cube_shape(
            "direct_terms",
            direct_terms_shape,
            (mml_count, max_direct_terms, q_pair_count),
        )?;
    }

    if reciprocal_count > 0 && mml_count > 0 {
        let required_d1term3 = angular_momentum_for_mml(mml_count - 1) + 1;
        validate_length("d1term3", input.d1term3.len(), required_d1term3)?;
    }
    validate_complex_vector("d1term3", input.d1term3)?;

    for (q_pair, &count) in input.direct_counts.iter().enumerate() {
        if count > max_direct_terms {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "direct_counts",
                index: count,
                len: max_direct_terms + 1,
            });
        }
        for direct_term in 0..count {
            let direct_index = input.direct_index_by_pair[(direct_term, q_pair)];
            if direct_index >= direct_count {
                return Err(KSpaceError::StructureFactorIndexOutOfRange {
                    name: "direct_index_by_pair",
                    index: direct_index,
                    len: direct_count,
                });
            }
        }
    }

    Ok(StructureFactorShape {
        mml_count,
        q_pair_count,
        reciprocal_count,
        max_direct_terms,
        radial_order_count: 0,
    })
}

fn validate_strset_non_rel_input(
    input: &KSpaceStrsetNonRelInput<'_>,
) -> Result<StrsetNonRelShape, KSpaceError> {
    if input.angular_state_count == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "angular_state_count",
            count: input.angular_state_count,
        });
    }
    validate_complex("wave_number", 0, input.wave_number)?;

    let q_pair_count = input.dllmmke.ncols();
    let pair_shape = input.q_pair_sites.dim();
    validate_cube_shape("q_pair_sites", pair_shape, (q_pair_count, pair_shape.1, 2))?;
    validate_exact_length("q_pair_counts", input.q_pair_counts.len(), q_pair_count)?;

    let site_count = input.site_offsets.len();
    validate_exact_length(
        "site_state_counts",
        input.site_state_counts.len(),
        site_count,
    )?;
    if site_count == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "site_count",
            count: site_count,
        });
    }

    for (site, &state_count) in input.site_state_counts.iter().enumerate() {
        if state_count != input.angular_state_count {
            return Err(KSpaceError::InvalidStructureFactorCount {
                name: "site_state_counts",
                count: state_count,
            });
        }
        validate_site_range(input.site_offsets[site], state_count)?;
    }
    let matrix_order = matrix_order_from_site_ranges(input.site_offsets, input.site_state_counts)?;

    let angular_pair_count = triangular_pair_count(input.angular_state_count)?;
    validate_exact_length("gaunt_counts", input.gaunt_counts.len(), angular_pair_count)?;
    let gaunt_value_count = input.gaunt_counts.iter().try_fold(0usize, |sum, &count| {
        sum.checked_add(count)
            .ok_or(KSpaceError::StructureFactorSizeOverflow {
                name: "gaunt_counts",
            })
    })?;
    validate_length(
        "gaunt_indices",
        input.gaunt_indices.len(),
        gaunt_value_count,
    )?;
    validate_length("gaunt_values", input.gaunt_values.len(), gaunt_value_count)?;

    let dllmmke_rows = input.dllmmke.nrows();
    validate_complex_matrix("dllmmke", input.dllmmke)?;
    for (index, &gaunt_index) in input
        .gaunt_indices
        .iter()
        .take(gaunt_value_count)
        .enumerate()
    {
        if gaunt_index >= dllmmke_rows {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "gaunt_indices",
                index: gaunt_index,
                len: dllmmke_rows,
            });
        }
        validate_vector_component("gaunt_values", index, input.gaunt_values[index])?;
    }

    validate_length("cipwl", input.cipwl.len(), input.angular_state_count)?;
    validate_complex_vector("cipwl", input.cipwl)?;
    for (index, &value) in input
        .cipwl
        .iter()
        .take(input.angular_state_count)
        .enumerate()
    {
        if value == Complex::new(0.0, 0.0) {
            return Err(KSpaceError::DegenerateStructureFactorValue {
                name: "cipwl",
                index,
            });
        }
    }

    for (q_pair, &count) in input.q_pair_counts.iter().enumerate() {
        if count == 0 || count > pair_shape.1 {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "q_pair_counts",
                index: count,
                len: pair_shape.1 + 1,
            });
        }
        for equivalent in 0..count {
            for axis in 0..2 {
                let site = input.q_pair_sites[(q_pair, equivalent, axis)];
                if site >= site_count {
                    return Err(KSpaceError::StructureFactorIndexOutOfRange {
                        name: "q_pair_sites",
                        index: site,
                        len: site_count,
                    });
                }
            }
        }
    }

    Ok(StrsetNonRelShape {
        angular_pair_count,
        matrix_order,
        q_pair_count,
    })
}

fn validate_strset_rel_input(
    input: &KSpaceStrsetRelInput<'_>,
) -> Result<StrsetRelShape, KSpaceError> {
    if input.angular_state_count == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "angular_state_count",
            count: input.angular_state_count,
        });
    }
    validate_complex("wave_number", 0, input.wave_number)?;

    let q_pair_count = input.dllmmke.ncols();
    let pair_shape = input.q_pair_sites.dim();
    validate_cube_shape("q_pair_sites", pair_shape, (q_pair_count, pair_shape.1, 2))?;
    validate_exact_length("q_pair_counts", input.q_pair_counts.len(), q_pair_count)?;

    let site_count = input.site_offsets.len();
    validate_exact_length(
        "site_state_counts",
        input.site_state_counts.len(),
        site_count,
    )?;
    if site_count == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "site_count",
            count: site_count,
        });
    }
    for (site, &state_count) in input.site_state_counts.iter().enumerate() {
        validate_site_range(input.site_offsets[site], state_count)?;
    }
    let matrix_order = matrix_order_from_site_ranges(input.site_offsets, input.site_state_counts)?;

    let angular_pair_count = triangular_pair_count(input.angular_state_count)?;
    validate_exact_length("gaunt_counts", input.gaunt_counts.len(), angular_pair_count)?;
    let gaunt_value_count = input.gaunt_counts.iter().try_fold(0usize, |sum, &count| {
        sum.checked_add(count)
            .ok_or(KSpaceError::StructureFactorSizeOverflow {
                name: "gaunt_counts",
            })
    })?;
    validate_length(
        "gaunt_indices",
        input.gaunt_indices.len(),
        gaunt_value_count,
    )?;
    validate_length("gaunt_values", input.gaunt_values.len(), gaunt_value_count)?;

    let dllmmke_rows = input.dllmmke.nrows();
    validate_complex_matrix("dllmmke", input.dllmmke)?;
    for (index, &gaunt_index) in input
        .gaunt_indices
        .iter()
        .take(gaunt_value_count)
        .enumerate()
    {
        if gaunt_index >= dllmmke_rows {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "gaunt_indices",
                index: gaunt_index,
                len: dllmmke_rows,
            });
        }
        validate_vector_component("gaunt_values", index, input.gaunt_values[index])?;
    }

    validate_length("cipwl", input.cipwl.len(), input.angular_state_count)?;
    validate_complex_vector("cipwl", input.cipwl)?;
    for (index, &value) in input
        .cipwl
        .iter()
        .take(input.angular_state_count)
        .enumerate()
    {
        if value == Complex::new(0.0, 0.0) {
            return Err(KSpaceError::DegenerateStructureFactorValue {
                name: "cipwl",
                index,
            });
        }
    }

    for (q_pair, &count) in input.q_pair_counts.iter().enumerate() {
        if count == 0 || count > pair_shape.1 {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "q_pair_counts",
                index: count,
                len: pair_shape.1 + 1,
            });
        }
        for equivalent in 0..count {
            for axis in 0..2 {
                let site = input.q_pair_sites[(q_pair, equivalent, axis)];
                if site >= site_count {
                    return Err(KSpaceError::StructureFactorIndexOutOfRange {
                        name: "q_pair_sites",
                        index: site,
                        len: site_count,
                    });
                }
            }
        }
    }

    let max_site_state_count = input.site_state_counts.iter().copied().max().unwrap_or(0);
    validate_matrix_shape(
        "rel_component_counts",
        input.rel_component_counts.nrows(),
        input.rel_component_counts.ncols(),
        2,
        input.rel_component_counts.ncols(),
    )?;
    if input.rel_component_counts.ncols() < max_site_state_count {
        return Err(KSpaceError::StructureFactorIndexOutOfRange {
            name: "rel_component_counts",
            index: max_site_state_count,
            len: input.rel_component_counts.ncols() + 1,
        });
    }

    let rel_shape = input.rel_component_indices.dim();
    validate_cube_shape(
        "rel_component_coefficients",
        input.rel_component_coefficients.dim(),
        rel_shape,
    )?;
    validate_cube_shape(
        "rel_component_indices",
        rel_shape,
        (rel_shape.0, 2, input.rel_component_counts.ncols()),
    )?;
    validate_complex_cube(
        "rel_component_coefficients",
        input.rel_component_coefficients,
    )?;

    for state in 0..max_site_state_count {
        for spin in 0..2 {
            let count = input.rel_component_counts[(spin, state)];
            if count > rel_shape.0 {
                return Err(KSpaceError::StructureFactorIndexOutOfRange {
                    name: "rel_component_counts",
                    index: count,
                    len: rel_shape.0 + 1,
                });
            }
            for component in 0..count {
                let angular_index = input.rel_component_indices[(component, spin, state)];
                if angular_index >= input.angular_state_count {
                    return Err(KSpaceError::StructureFactorIndexOutOfRange {
                        name: "rel_component_indices",
                        index: angular_index,
                        len: input.angular_state_count,
                    });
                }
            }
        }
    }

    Ok(StrsetRelShape {
        matrix_order,
        q_pair_count,
    })
}

fn validate_q_pair_group_input(
    positions: ArrayView2<'_, Real>,
    tolerance: Real,
) -> Result<(), KSpaceError> {
    if positions.nrows() == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_positions",
            count: 0,
        });
    }
    validate_matrix_shape(
        "q_pair_positions",
        positions.nrows(),
        positions.ncols(),
        positions.nrows(),
        3,
    )?;
    validate_real_matrix("q_pair_positions", positions)?;
    validate_vector_component("q_pair_tolerance", 0, tolerance)?;
    if tolerance <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "q_pair_tolerance",
            value: tolerance,
        });
    }
    Ok(())
}

fn validate_direct_lattice_setup_input(
    direct_basis: [Vector3; 3],
    q_pair_offsets: ArrayView2<'_, Real>,
    rmax: Real,
) -> Result<(), KSpaceError> {
    validate_basis(direct_basis)?;
    if q_pair_offsets.nrows() == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_offsets",
            count: 0,
        });
    }
    validate_matrix_shape(
        "q_pair_offsets",
        q_pair_offsets.nrows(),
        q_pair_offsets.ncols(),
        q_pair_offsets.nrows(),
        3,
    )?;
    validate_real_matrix("q_pair_offsets", q_pair_offsets)?;
    validate_vector_component("rmax", 0, rmax)?;
    if rmax <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "rmax",
            value: rmax,
        });
    }
    Ok(())
}

fn validate_reciprocal_lattice_setup_input(
    reciprocal_basis: [Vector3; 3],
    gmax: Real,
    energy_min_reduced: Real,
    energy_max_reduced: Real,
) -> Result<(), KSpaceError> {
    validate_basis(reciprocal_basis)?;
    validate_vector_component("gmax", 0, gmax)?;
    if gmax <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "gmax",
            value: gmax,
        });
    }
    validate_vector_component("energy_min_reduced", 0, energy_min_reduced)?;
    validate_vector_component("energy_max_reduced", 0, energy_max_reduced)?;
    if energy_max_reduced < energy_min_reduced {
        return Err(KSpaceError::InvalidStructureFactorRange {
            name: "reduced_energy_probe",
            min: energy_min_reduced,
            max: energy_max_reduced,
        });
    }
    Ok(())
}

fn validate_reciprocal_pair_phases_input(
    input: &KSpaceReciprocalPairPhasesInput<'_>,
) -> Result<(), KSpaceError> {
    validate_basis(input.direct_basis)?;
    validate_basis(input.reciprocal_basis)?;
    validate_vector_component("eta", 0, input.eta)?;
    if input.eta <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: input.eta,
        });
    }

    validate_matrix_shape(
        "reciprocal_indices",
        input.reciprocal_indices.nrows(),
        input.reciprocal_indices.ncols(),
        input.reciprocal_indices.nrows(),
        3,
    )?;
    if input.q_pair_offsets.nrows() == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_offsets",
            count: 0,
        });
    }
    validate_matrix_shape(
        "q_pair_offsets",
        input.q_pair_offsets.nrows(),
        input.q_pair_offsets.ncols(),
        input.q_pair_offsets.nrows(),
        3,
    )?;
    validate_real_matrix("q_pair_offsets", input.q_pair_offsets)?;
    Ok(())
}

fn validate_direct_lattice_terms_input(
    input: &KSpaceDirectLatticeTermsInput<'_>,
) -> Result<StructureFactorShape, KSpaceError> {
    validate_basis(input.direct_basis)?;
    validate_vector_component("eta", 0, input.eta)?;
    if input.eta <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: input.eta,
        });
    }
    validate_harmonic_shape("qjltab", input.qjltab, input.lmax)?;
    validate_real_matrix("qjltab", input.qjltab)?;
    input
        .j22max
        .checked_add(1)
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "radial_terms",
        })?;
    let mml_count = harmonic_polynomial_count(input.lmax)?;

    let direct_count = input.direct_indices.nrows();
    validate_matrix_shape(
        "direct_indices",
        input.direct_indices.nrows(),
        input.direct_indices.ncols(),
        direct_count,
        3,
    )?;

    let q_pair_count = input.q_pair_offsets.nrows();
    if q_pair_count == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "q_pair_offsets",
            count: 0,
        });
    }
    validate_matrix_shape(
        "q_pair_offsets",
        input.q_pair_offsets.nrows(),
        input.q_pair_offsets.ncols(),
        q_pair_count,
        3,
    )?;
    validate_real_matrix("q_pair_offsets", input.q_pair_offsets)?;

    let max_direct_terms = input.direct_index_by_pair.nrows();
    validate_matrix_shape(
        "direct_index_by_pair",
        input.direct_index_by_pair.nrows(),
        input.direct_index_by_pair.ncols(),
        max_direct_terms,
        q_pair_count,
    )?;
    validate_exact_length("direct_counts", input.direct_counts.len(), q_pair_count)?;

    for (q_pair, &count) in input.direct_counts.iter().enumerate() {
        if count > max_direct_terms {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "direct_counts",
                index: count,
                len: max_direct_terms + 1,
            });
        }
        for direct_term in 0..count {
            let direct_index = input.direct_index_by_pair[(direct_term, q_pair)];
            if direct_index >= direct_count {
                return Err(KSpaceError::StructureFactorIndexOutOfRange {
                    name: "direct_index_by_pair",
                    index: direct_index,
                    len: direct_count,
                });
            }
        }
    }

    Ok(StructureFactorShape {
        mml_count,
        q_pair_count,
        reciprocal_count: 0,
        max_direct_terms,
        radial_order_count: input.j22max + 1,
    })
}

fn validate_energy_dependent_terms_input(
    input: &KSpaceEnergyDependentTermsInput<'_>,
) -> Result<StructureFactorShape, KSpaceError> {
    validate_complex("energy", 0, input.energy)?;
    validate_vector_component("eta", 0, input.eta)?;
    if input.eta <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "eta",
            value: input.eta,
        });
    }

    let mml_count = harmonic_polynomial_count(input.lmax)?;
    let direct_terms_shape = input.base_direct_terms.dim();
    if direct_terms_shape.0 != mml_count {
        validate_cube_shape(
            "base_direct_terms",
            direct_terms_shape,
            (mml_count, direct_terms_shape.1, direct_terms_shape.2),
        )?;
    }
    validate_complex_cube("base_direct_terms", input.base_direct_terms)?;

    let radial_shape = input.radial_terms.dim();
    if radial_shape.0 == 0 {
        return Err(KSpaceError::InvalidStructureFactorCount {
            name: "radial_terms",
            count: radial_shape.0,
        });
    }
    validate_real_array4(
        "radial_terms",
        input.radial_terms,
        (
            radial_shape.0,
            input.lmax + 1,
            direct_terms_shape.1,
            direct_terms_shape.2,
        ),
    )?;

    let q_pair_count = direct_terms_shape.2;
    validate_exact_length("direct_counts", input.direct_counts.len(), q_pair_count)?;
    for &count in input.direct_counts {
        if count > direct_terms_shape.1 {
            return Err(KSpaceError::StructureFactorIndexOutOfRange {
                name: "direct_counts",
                index: count,
                len: direct_terms_shape.1 + 1,
            });
        }
    }

    Ok(StructureFactorShape {
        mml_count,
        q_pair_count,
        reciprocal_count: 0,
        max_direct_terms: direct_terms_shape.1,
        radial_order_count: radial_shape.0,
    })
}

fn validate_ewald_energy_tables_input(
    input: &KSpaceEwaldEnergyTablesInput<'_>,
) -> Result<(), KSpaceError> {
    validate_complex("energy", 0, input.energy)?;
    validate_vector_component("initial_eta", 0, input.initial_eta)?;
    if input.initial_eta <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "initial_eta",
            value: input.initial_eta,
        });
    }
    if input.initial_eta > CHANGE_ETA_MAX {
        return Err(KSpaceError::EwaldEtaExceeded {
            eta: input.initial_eta,
            max: CHANGE_ETA_MAX,
        });
    }
    Ok(())
}

fn change_eta_next(eta: Real) -> Result<Real, KSpaceError> {
    validate_vector_component("eta", 0, eta)?;
    let next_eta = eta * CHANGE_ETA_INCREASE_FACTOR;
    validate_vector_component("next_eta", 0, next_eta)?;
    if next_eta > CHANGE_ETA_MAX {
        return Err(KSpaceError::EwaldEtaExceeded {
            eta: next_eta,
            max: CHANGE_ETA_MAX,
        });
    }
    Ok(next_eta)
}

fn direct_cell_volume_factor(direct_basis: [Vector3; 3]) -> Result<Real, KSpaceError> {
    let cross = cross(direct_basis[1], direct_basis[2]);
    let determinant = dot(direct_basis[0], cross);
    validate_vector_component("direct_lattice_determinant", 0, determinant)?;
    let volume_factor = determinant.abs() * PI2.powi(3);
    validate_vector_component("direct_lattice_volume", 0, volume_factor)?;
    if volume_factor <= 0.0 {
        return Err(KSpaceError::DegenerateStructureFactorValue {
            name: "direct_lattice_volume",
            index: 0,
        });
    }
    Ok(volume_factor)
}

fn reciprocal_lattice_index_radius(
    reciprocal_basis: [Vector3; 3],
    gmax: Real,
) -> Result<i32, KSpaceError> {
    let min_reciprocal_length =
        reciprocal_basis
            .into_iter()
            .try_fold(Real::INFINITY, |minimum, basis| {
                let length = dot(basis, basis).sqrt();
                validate_vector_component("reciprocal_basis_length", 0, length)?;
                Ok::<Real, KSpaceError>(minimum.min(length))
            })?;
    if min_reciprocal_length <= 0.0 {
        return Err(KSpaceError::DegenerateStructureFactorValue {
            name: "reciprocal_basis_length",
            index: 0,
        });
    }
    let radius = (gmax / min_reciprocal_length).trunc() + 2.0;
    if radius > i32::MAX as Real {
        return Err(KSpaceError::StructureFactorSizeOverflow {
            name: "reciprocal_lattice_index_radius",
        });
    }
    Ok(radius as i32)
}

fn reciprocal_vector_matches_energy_probe(
    vector: Vector3,
    reciprocal_basis: [Vector3; 3],
    gmax_squared: Real,
    energy_min_reduced: Real,
    energy_max_reduced: Real,
) -> bool {
    for j1 in -1..=1 {
        for j2 in -1..=1 {
            for j3 in -1..=1 {
                let shifted = [
                    vector[0]
                        + 0.5
                            * (Real::from(j1) * reciprocal_basis[0][0]
                                + Real::from(j2) * reciprocal_basis[1][0]
                                + Real::from(j3) * reciprocal_basis[2][0]),
                    vector[1]
                        + 0.5
                            * (Real::from(j1) * reciprocal_basis[0][1]
                                + Real::from(j2) * reciprocal_basis[1][1]
                                + Real::from(j3) * reciprocal_basis[2][1]),
                    vector[2]
                        + 0.5
                            * (Real::from(j1) * reciprocal_basis[0][2]
                                + Real::from(j2) * reciprocal_basis[1][2]
                                + Real::from(j3) * reciprocal_basis[2][2]),
                ];
                let ksq = dot(shifted, shifted);
                for energy_index in 0..=3 {
                    let reduced_energy = energy_min_reduced
                        + energy_index as Real * (energy_max_reduced - energy_min_reduced) / 3.0;
                    if ksq - reduced_energy <= gmax_squared {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn direct_lattice_index_radius(
    direct_basis: [Vector3; 3],
    rmax: Real,
    max_q_pair_offset_norm: Real,
) -> Result<i32, KSpaceError> {
    let min_direct_length =
        direct_basis
            .into_iter()
            .try_fold(Real::INFINITY, |minimum, basis| {
                let length = dot(basis, basis).sqrt();
                validate_vector_component("direct_basis_length", 0, length)?;
                Ok::<Real, KSpaceError>(minimum.min(length))
            })?;
    if min_direct_length <= 0.0 {
        return Err(KSpaceError::DegenerateStructureFactorValue {
            name: "direct_basis_length",
            index: 0,
        });
    }
    let search_radius = rmax + max_q_pair_offset_norm * 1.001;
    validate_vector_component("direct_lattice_search_radius", 0, search_radius)?;
    let radius = (search_radius / min_direct_length).trunc() + 2.0;
    if radius > i32::MAX as Real {
        return Err(KSpaceError::StructureFactorSizeOverflow {
            name: "direct_lattice_index_radius",
        });
    }
    Ok(radius as i32)
}

fn direct_lattice_counts(
    retained: &[IntegerLatticeVector],
    direct_basis: [Vector3; 3],
    q_pair_offsets: ArrayView2<'_, Real>,
    rmax: Real,
) -> Vec<usize> {
    (0..q_pair_offsets.nrows())
        .map(|q_pair| {
            let start = if q_pair == 0 { 1 } else { 0 };
            retained
                .iter()
                .skip(start)
                .filter(|lattice_vector| {
                    let vector =
                        shifted_vector([0.0, 0.0, 0.0], direct_basis, lattice_vector.indices);
                    direct_vector_matches_q_pair(vector, q_pair_offsets, q_pair, rmax)
                })
                .count()
        })
        .collect()
}

fn direct_vector_matches_any_q_pair(
    vector: Vector3,
    q_pair_offsets: ArrayView2<'_, Real>,
    rmax: Real,
) -> bool {
    (0..q_pair_offsets.nrows())
        .any(|q_pair| direct_vector_matches_q_pair(vector, q_pair_offsets, q_pair, rmax))
}

fn direct_vector_matches_q_pair(
    vector: Vector3,
    q_pair_offsets: ArrayView2<'_, Real>,
    q_pair: usize,
    rmax: Real,
) -> bool {
    let offset = [
        q_pair_offsets[(q_pair, 0)],
        q_pair_offsets[(q_pair, 1)],
        q_pair_offsets[(q_pair, 2)],
    ];
    let delta = [
        vector[0] - offset[0],
        vector[1] - offset[1],
        vector[2] - offset[2],
    ];
    dot(delta, delta).sqrt() <= rmax
}

fn matching_q_pair(offsets: &[Vector3], offset: Vector3, tolerance: Real) -> Option<usize> {
    offsets.iter().position(|candidate| {
        (candidate[0] - offset[0]).abs() < tolerance
            && (candidate[1] - offset[1]).abs() < tolerance
            && (candidate[2] - offset[2]).abs() < tolerance
    })
}

fn validate_site_range(offset: usize, count: usize) -> Result<(), KSpaceError> {
    offset
        .checked_add(count)
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "site_offsets",
        })?;
    Ok(())
}

fn matrix_order_from_site_ranges(
    offsets: &[usize],
    counts: &[usize],
) -> Result<usize, KSpaceError> {
    offsets
        .iter()
        .zip(counts)
        .try_fold(0usize, |order, (&offset, &count)| {
            let end =
                offset
                    .checked_add(count)
                    .ok_or(KSpaceError::StructureFactorSizeOverflow {
                        name: "site_offsets",
                    })?;
            Ok(order.max(end))
        })
}

fn triangular_pair_count(count: usize) -> Result<usize, KSpaceError> {
    let next = count
        .checked_add(1)
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "gaunt_counts",
        })?;
    count
        .checked_mul(next)
        .and_then(|value| value.checked_div(2))
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "gaunt_counts",
        })
}

type RealGauntTables = (Vec<usize>, Vec<usize>, Vec<Real>);

fn kspace_real_gaunt_tables(
    angular_lmax: usize,
    alat_bohr: Real,
) -> Result<RealGauntTables, KSpaceError> {
    let angular_state_count = harmonic_polynomial_count(angular_lmax)?;
    let pair_count = triangular_pair_count(angular_state_count)?;
    let mut gaunt_counts = Vec::with_capacity(pair_count);
    let mut gaunt_indices = Vec::new();
    let mut gaunt_values = Vec::new();
    let prefactor = 2.0 * PI2 * PI2 / alat_bohr;

    for l1 in 0..=angular_lmax {
        let l1_i32 = usize_to_i32_for_wigner(l1)?;
        for m1 in -l1_i32..=l1_i32 {
            let lm1 = lm_index(l1, m1)?;
            for l2 in 0..=angular_lmax {
                let l2_i32 = usize_to_i32_for_wigner(l2)?;
                for m2 in -l2_i32..=l2_i32 {
                    let lm2 = lm_index(l2, m2)?;
                    if lm2 > lm1 {
                        continue;
                    }

                    let mut count = 0usize;
                    let l3_min = l1.abs_diff(l2);
                    let l3_max = l1
                        .checked_add(l2)
                        .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "gaunt_l3" })?;
                    for l3 in l3_min..=l3_max {
                        let l3_i32 = usize_to_i32_for_wigner(l3)?;
                        for m3 in -l3_i32..=l3_i32 {
                            let real_gaunt = real_gaunt_coefficient(l1, m1, l2, m2, l3, m3)?;
                            if real_gaunt.abs() > STRGAUNT_CUTOFF {
                                let lm3 = lm_index(l3, m3)?;
                                let value = prefactor * real_gaunt;
                                validate_vector_component(
                                    "gaunt_values",
                                    gaunt_values.len(),
                                    value,
                                )?;
                                gaunt_indices.push(lm3);
                                gaunt_values.push(value);
                                count += 1;
                            }
                        }
                    }
                    gaunt_counts.push(count);
                }
            }
        }
    }

    validate_exact_length("gaunt_counts", gaunt_counts.len(), pair_count)?;
    Ok((gaunt_counts, gaunt_indices, gaunt_values))
}

fn real_gaunt_coefficient(
    l1: usize,
    m1: i32,
    l2: usize,
    m2: i32,
    l3: usize,
    m3: i32,
) -> Result<Real, KSpaceError> {
    let mut total = 0.0;
    for component1 in 0..2 {
        let (xm1, cc1) = real_harmonic_component(m1, component1);
        if cc1 == Complex::new(0.0, 0.0) {
            continue;
        }
        for component2 in 0..2 {
            let (xm2, cc2) = real_harmonic_component(m2, component2);
            if cc2 == Complex::new(0.0, 0.0) {
                continue;
            }
            for component3 in 0..2 {
                let (xm3, cc3) = real_harmonic_component(m3, component3);
                if cc3 == Complex::new(0.0, 0.0) || xm1 + xm2 + xm3 != 0 {
                    continue;
                }
                let complex_gaunt = complex_harmonic_gaunt(l1, l2, l3, xm1, xm2, xm3)?;
                total += (cc1 * cc2 * cc3).re * complex_gaunt;
            }
        }
    }
    validate_vector_component("real_gaunt", 0, total)?;
    Ok(total)
}

fn complex_harmonic_gaunt(
    l1: usize,
    l2: usize,
    l3: usize,
    m1: i32,
    m2: i32,
    m3: i32,
) -> Result<Real, KSpaceError> {
    let two_l1_plus_one = checked_two_l_plus_one(l1)? as Real;
    let two_l2_plus_one = checked_two_l_plus_one(l2)? as Real;
    let two_l3_plus_one = checked_two_l_plus_one(l3)? as Real;
    let normalization = (two_l1_plus_one * two_l2_plus_one / (2.0 * PI2 * two_l3_plus_one)).sqrt();
    let coefficient = normalization
        * parity_sign(-m3)
        * clebsch_gordan(l1, l2, l3, 0, 0, 0)?
        * clebsch_gordan(l1, l2, l3, m1, m2, -m3)?;
    validate_vector_component("complex_gaunt", 0, coefficient)?;
    Ok(coefficient)
}

fn clebsch_gordan(
    j1: usize,
    j2: usize,
    j3: usize,
    m1: i32,
    m2: i32,
    m3: i32,
) -> Result<Real, KSpaceError> {
    if m1.checked_add(m2) != Some(m3) {
        return Ok(0.0);
    }
    let j1_i32 = usize_to_i32_for_wigner(j1)?;
    let j2_i32 = usize_to_i32_for_wigner(j2)?;
    let j3_i32 = usize_to_i32_for_wigner(j3)?;
    let phase = j1_i32
        .checked_sub(j2_i32)
        .and_then(|value| value.checked_add(m3))
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "clebsch_gordan_phase",
        })?;
    Ok(parity_sign(phase)
        * (checked_two_l_plus_one(j3)? as Real).sqrt()
        * wigner_3j(j1_i32, j2_i32, j3_i32, m1, m2, 1)?)
}

fn real_harmonic_component(magnetic: i32, component: usize) -> (i32, Complex) {
    let root_half = 0.5_f64.sqrt();
    let abs_magnetic = magnetic.abs();
    if component == 0 {
        let coefficient = match magnetic.cmp(&0) {
            std::cmp::Ordering::Less => Complex::new(0.0, root_half),
            std::cmp::Ordering::Equal => Complex::new(1.0, 0.0),
            std::cmp::Ordering::Greater => Complex::new(root_half, 0.0),
        };
        (-abs_magnetic, coefficient)
    } else {
        let sign = parity_sign(abs_magnetic);
        let coefficient = match magnetic.cmp(&0) {
            std::cmp::Ordering::Less => Complex::new(0.0, -root_half * sign),
            std::cmp::Ordering::Equal => Complex::new(0.0, 0.0),
            std::cmp::Ordering::Greater => Complex::new(root_half * sign, 0.0),
        };
        (abs_magnetic, coefficient)
    }
}

fn kspace_cipwl(harmonic_lmax: usize) -> Result<Array1<Complex>, KSpaceError> {
    let count = harmonic_polynomial_count(harmonic_lmax)?;
    let mut values = Array1::<Complex>::zeros(count);
    let mut index = 0usize;
    for angular_momentum in 0..=harmonic_lmax {
        let phase = match angular_momentum % 4 {
            0 => Complex::new(1.0, 0.0),
            1 => Complex::new(0.0, 1.0),
            2 => Complex::new(-1.0, 0.0),
            _ => Complex::new(0.0, -1.0),
        };
        let width = angular_momentum
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "cipwl" })?;
        for _ in 0..width {
            values[index] = phase;
            index += 1;
        }
    }
    validate_exact_length("cipwl", index, count)?;
    Ok(values)
}

fn inverse_factorial_range_product(start: usize, end: usize) -> Real {
    (start..=end).fold(1.0, |product, value| product / value as Real)
}

fn lm_index(angular_momentum: usize, magnetic: i32) -> Result<usize, KSpaceError> {
    let base = angular_momentum
        .checked_mul(
            angular_momentum
                .checked_add(1)
                .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "lm_index" })?,
        )
        .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "lm_index" })?;
    let offset = usize::try_from(magnetic.unsigned_abs())
        .map_err(|_| KSpaceError::StructureFactorSizeOverflow { name: "lm_index" })?;
    if magnetic < 0 {
        base.checked_sub(offset)
            .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "lm_index" })
    } else {
        base.checked_add(offset)
            .ok_or(KSpaceError::StructureFactorSizeOverflow { name: "lm_index" })
    }
}

fn checked_two_l_plus_one(angular_momentum: usize) -> Result<usize, KSpaceError> {
    angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "angular_momentum",
        })
}

fn usize_to_i32_for_wigner(value: usize) -> Result<i32, KSpaceError> {
    i32::try_from(value)
        .map_err(|_| KSpaceError::Angular(crate::AngularError::IndexTooLarge { value }))
}

fn parity_sign(exponent: i32) -> Real {
    if exponent % 2 == 0 { 1.0 } else { -1.0 }
}

fn validate_harmonic_polynomial_input(
    input: &KSpaceHarmonicPolynomialsInput<'_>,
) -> Result<(), KSpaceError> {
    validate_vector("vector", input.vector)?;
    validate_harmonic_shape("qjltab", input.qjltab, input.lmax)?;
    validate_real_matrix("qjltab", input.qjltab)?;
    Ok(())
}

fn harmonic_polynomials(
    vector: Vector3,
    lmax: usize,
    qjltab: ArrayView2<'_, Real>,
) -> Result<Array1<Real>, KSpaceError> {
    let count = harmonic_polynomial_count(lmax)?;
    let mut hp = Array1::<Real>::zeros(count);
    let mut t = Array2::<Real>::zeros((lmax + 1, lmax + 1));
    t[(0, 0)] = 1.0;
    for ll in 1..=lmax {
        t[(ll, ll)] = t[(ll - 1, ll - 1)] * (2.0 * ll as Real - 1.0);
    }

    hp[0] = qjltab[(0, 0)] * t[(0, 0)];
    if vector[0].abs() + vector[1].abs() + vector[2].abs() < STRHARPOL_ZERO_VECTOR_EPSILON {
        return Ok(hp);
    }

    let mut shp = vec![0.0; lmax + 1];
    let mut chp = vec![0.0; lmax + 1];
    chp[0] = 1.0;

    let x = vector[0];
    let y = vector[1];
    let z = vector[2];
    let xy = x * x + y * y;
    let zsq = z * z;
    let rsq = xy + zsq;

    for jj in 0..lmax {
        let jjp1 = jj + 1;
        chp[jjp1] = x * chp[jj] - y * shp[jj];
        shp[jjp1] = x * shp[jj] + y * chp[jj];
    }

    if lmax >= 1 {
        t[(1, 0)] = z;
    }

    let mut f1 = z;
    let mut f2 = 0.0;
    let mut f3 = 1.0;
    for ll in 1..lmax {
        f1 += z + z;
        f2 += rsq;
        f3 += 1.0;
        t[(ll + 1, 0)] = (f1 * t[(ll, 0)] - f2 * t[(ll - 1, 0)]) / f3;
    }

    if xy > zsq {
        let f20 = z / xy;
        let f10 = 1.0 + z * f20;

        for jj in 0..lmax {
            let jjp1 = jj + 1;
            let mut f1 = f10 * (jj + jj + 1) as Real;
            let mut f2 = f20;
            for ll in (jj + 2)..=lmax {
                f1 += f10;
                f2 += f20;
                t[(ll, jjp1)] = f1 * t[(ll - 1, jj)] - f2 * t[(ll, jj)];
            }
        }
    } else if lmax >= 2 {
        let f1 = -xy / z;
        let f20 = rsq / z;

        for ll in 2..=lmax {
            let mut jj = ll;
            let mut f2 = f20 * (ll + jj) as Real;
            let mut f3 = (ll - jj) as Real;

            for _ in 1..ll {
                jj -= 1;
                f2 -= f20;
                f3 += 1.0;
                t[(ll, jj)] = (f1 * t[(ll, jj + 1)] + f2 * t[(ll - 1, jj)]) / f3;
            }
        }
    }

    for ll in 1..=lmax {
        let mm0ll = ll * ll + ll;
        hp[mm0ll] = qjltab[(0, ll)] * t[(ll, 0)];
        for jj in 1..=ll {
            let factor = qjltab[(jj, ll)] * t[(ll, jj)];
            hp[mm0ll + jj] = factor * chp[jj];
            hp[mm0ll - jj] = factor * shp[jj];
        }
    }

    for (index, &value) in hp.indexed_iter() {
        validate_vector_component("harmonic_polynomials", index, value)?;
    }
    Ok(hp)
}

fn angular_momentum_for_mml(index: usize) -> usize {
    let mut remaining = index;
    let mut angular_momentum = 0;
    loop {
        let width = 2 * angular_momentum + 1;
        if remaining < width {
            return angular_momentum;
        }
        remaining -= width;
        angular_momentum += 1;
    }
}

fn shifted_vector(k: Vector3, basis: [Vector3; 3], indices: [i32; 3]) -> Vector3 {
    [
        k[0] + Real::from(indices[0]) * basis[0][0]
            + Real::from(indices[1]) * basis[1][0]
            + Real::from(indices[2]) * basis[2][0],
        k[1] + Real::from(indices[0]) * basis[0][1]
            + Real::from(indices[1]) * basis[1][1]
            + Real::from(indices[2]) * basis[2][1],
        k[2] + Real::from(indices[0]) * basis[0][2]
            + Real::from(indices[1]) * basis[1][2]
            + Real::from(indices[2]) * basis[2][2],
    ]
}

fn complex_phase(phase: Real) -> Complex {
    Complex::new(0.0, phase).exp()
}

fn dot(left: Vector3, right: Vector3) -> Real {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: Vector3, right: Vector3) -> Vector3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn strconfra(aa: Real, x: Real) -> Result<Real, KSpaceError> {
    validate_vector_component("strconfra_a", 0, aa)?;
    validate_vector_component("strconfra_x", 0, x)?;
    if x <= 0.0 {
        return Err(KSpaceError::InvalidStructureFactorPositiveParameter {
            name: "strconfra_x",
            value: x,
        });
    }

    let mut imax = STRCONFRA_INITIAL_IMAX - STRCONFRA_INCREMENT;
    let mut previous = None;
    loop {
        imax += STRCONFRA_INCREMENT;
        let mut value = Real::from(imax) / x;
        let mut i = imax;
        for _ in 2..=imax {
            value += 1.0;
            value = x + (Real::from(i) - aa) / value;
            value = Real::from(i - 1) / value;
            i -= 1;
        }

        value += 1.0;
        value = x + (1.0 - aa) / value;
        value = 1.0 / value;
        validate_vector_component("strconfra", imax as usize, value)?;

        let Some(previous_value) = previous else {
            previous = Some(value);
            continue;
        };
        if ((value - previous_value) / value).abs() <= STRCONFRA_TOLERANCE {
            return Ok(value);
        }
        if imax >= STRCONFRA_MAX_IMAX {
            return Err(KSpaceError::StructureFactorSizeOverflow {
                name: "strconfra_iterations",
            });
        }
        previous = Some(value);
    }
}

fn strcc_d300(energy: Complex, eta: Real) -> Result<Complex, KSpaceError> {
    let mut alpha = eta.sqrt() / PI2;
    validate_vector_component("alpha0", 0, alpha)?;
    let mut energy_power = Complex::new(1.0, 0.0);
    let mut d300 = Complex::new(0.0, 0.0);
    let mut j13 = 0usize;

    loop {
        d300 += energy_power * alpha;
        validate_complex("d300", j13, d300)?;

        let numerator = 2.0 * j13 as Real - 1.0;
        let denominator = eta * (j13 as Real + 1.0) * (2.0 * j13 as Real + 1.0);
        alpha *= numerator / denominator;
        energy_power *= energy;
        if energy_power.norm() < STRCC_D300_UNDERFLOW {
            energy_power = Complex::new(0.0, 0.0);
        }

        let next_term = energy_power * alpha;
        validate_complex("d300_next_term", j13, next_term)?;
        if (next_term / d300).norm() <= STRCC_D300_TOLERANCE && j13 >= STRCC_D300_MIN_TERMS {
            return Ok(d300);
        }
        if j13 >= STRCC_D300_MAX_TERMS {
            return Err(KSpaceError::StructureFactorSizeOverflow {
                name: "d300_iterations",
            });
        }
        j13 += 1;
    }
}

fn strcc_terms_exceed_threshold(
    d300: Complex,
    direct_terms: ArrayView3<'_, Complex>,
    d1term3: ndarray::ArrayView1<'_, Complex>,
) -> bool {
    let direct_term = direct_terms
        .get((0, 0, 0))
        .copied()
        .unwrap_or_else(|| Complex::new(0.0, 0.0));
    let d1term = d1term3
        .get(1)
        .copied()
        .unwrap_or_else(|| Complex::new(0.0, 0.0));
    d300.norm().max(direct_term.norm()).max(d1term.norm()) > STRCC_EWALD_TERMS_THRESHOLD
}

fn harmonic_polynomial_count(lmax: usize) -> Result<usize, KSpaceError> {
    lmax.checked_add(1)
        .and_then(|size| size.checked_mul(size))
        .ok_or(KSpaceError::StructureFactorSizeOverflow {
            name: "harmonic_polynomials",
        })
}

fn validate_harmonic_shape(
    name: &'static str,
    values: ArrayView2<'_, Real>,
    lmax: usize,
) -> Result<(), KSpaceError> {
    let expected = lmax
        .checked_add(1)
        .ok_or(KSpaceError::StructureFactorSizeOverflow { name })?;
    validate_matrix_shape(name, values.nrows(), values.ncols(), expected, expected)
}

fn validate_matrix_shape(
    name: &'static str,
    rows: usize,
    columns: usize,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), KSpaceError> {
    if rows == expected_rows && columns == expected_columns {
        Ok(())
    } else {
        Err(KSpaceError::InvalidStructureFactorShape {
            name,
            rows,
            columns,
            expected_rows,
            expected_columns,
        })
    }
}

fn validate_cube_shape(
    name: &'static str,
    actual: (usize, usize, usize),
    expected: (usize, usize, usize),
) -> Result<(), KSpaceError> {
    if actual == expected {
        Ok(())
    } else {
        Err(KSpaceError::InvalidStructureFactorCubeShape {
            name,
            first: actual.0,
            second: actual.1,
            third: actual.2,
            expected_first: expected.0,
            expected_second: expected.1,
            expected_third: expected.2,
        })
    }
}

fn validate_exact_length(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), KSpaceError> {
    if actual == expected {
        Ok(())
    } else {
        Err(KSpaceError::InvalidStructureFactorLength {
            name,
            actual,
            expected,
        })
    }
}

fn validate_length(name: &'static str, actual: usize, expected: usize) -> Result<(), KSpaceError> {
    if actual >= expected {
        Ok(())
    } else {
        Err(KSpaceError::InvalidStructureFactorLength {
            name,
            actual,
            expected,
        })
    }
}

fn validate_real_matrix(
    name: &'static str,
    values: ArrayView2<'_, Real>,
) -> Result<(), KSpaceError> {
    for ((row, column), &value) in values.indexed_iter() {
        validate_vector_component(name, row * values.ncols() + column, value)?;
    }
    Ok(())
}

fn validate_complex_vector(
    name: &'static str,
    values: ndarray::ArrayView1<'_, Complex>,
) -> Result<(), KSpaceError> {
    for (index, &value) in values.indexed_iter() {
        validate_complex(name, index, value)?;
    }
    Ok(())
}

fn validate_complex_matrix(
    name: &'static str,
    values: ArrayView2<'_, Complex>,
) -> Result<(), KSpaceError> {
    for ((row, column), &value) in values.indexed_iter() {
        validate_complex(name, row * values.ncols() + column, value)?;
    }
    Ok(())
}

fn validate_complex_cube(
    name: &'static str,
    values: ArrayView3<'_, Complex>,
) -> Result<(), KSpaceError> {
    let shape = values.dim();
    for ((first, second, third), &value) in values.indexed_iter() {
        let index = (first * shape.1 + second) * shape.2 + third;
        validate_complex(name, index, value)?;
    }
    Ok(())
}

fn validate_real_array4(
    name: &'static str,
    values: ArrayView4<'_, Real>,
    expected: (usize, usize, usize, usize),
) -> Result<(), KSpaceError> {
    let actual = values.dim();
    if actual != expected {
        return Err(KSpaceError::InvalidStructureFactorArray4Shape {
            name,
            first: actual.0,
            second: actual.1,
            third: actual.2,
            fourth: actual.3,
            expected_first: expected.0,
            expected_second: expected.1,
            expected_third: expected.2,
            expected_fourth: expected.3,
        });
    }
    for ((first, second, third, fourth), &value) in values.indexed_iter() {
        let index = ((first * actual.1 + second) * actual.2 + third) * actual.3 + fourth;
        validate_vector_component(name, index, value)?;
    }
    Ok(())
}

fn validate_complex(name: &'static str, index: usize, value: Complex) -> Result<(), KSpaceError> {
    validate_vector_component(name, index * 2, value.re)?;
    validate_vector_component(name, index * 2 + 1, value.im)?;
    Ok(())
}
