//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels for final-state planning,
//! angular coefficient tables, NRIXS transition weights, q-Bessel tables,
//! initial-state occupation normalization, and angular-decomposition spectrum
//! updates, plus phase-mesh primitive, FEFF84 grid construction, and
//! `grid.inp` user-grid phase mesh composition.

use ndarray::{Array1, Array2, Array5, ArrayView1, ShapeBuilder};
use refeff_linalg::feff_determinant;

use crate::{
    Complex, Real, spherical_bessel_j_y, terp, wigner_3j,
    xsph_occ_norm::{
        XSPH_OCC_NORM_ATOMIC_NUMBER_MAX, XSPH_OCC_NORM_HOLE_COUNT, xsph_occ_norm_denominator,
        xsph_occ_norm_numerator,
    },
};

const QBESSEL_MAX_LJ: usize = 39;
const QBESSEL_ZERO_CUTOFF: Real = 1.0e8;
const CWIG3J_MAX_DOUBLED_ARGUMENT: i32 = 116;
const XSPH_MAX_LX: usize = 20;
const XSPH_HARTREE_EV: Real = 27.211_396;
const XSPH_BOHR_ANGSTROM: Real = 0.529_177_249;
const XSPH_HOLE_ORBITAL_X0: Real = 8.80;
const XSPH_HOLE_ORBITAL_TAIL_CUTOFF: Real = 1.0e-11;
const XSPH_PHASE_SORT_TOLERANCE: Real = 0.001;
const XSPH_USER_PHASE_GRID_MAX_RECORDS: usize = 10;

mod mesh;
mod spectrum;
mod types;

pub use mesh::*;
pub use spectrum::*;
pub use types::*;

/// Port of FEFF `XSPH/mincalc.f90`.
///
/// The input arrays may contain extra capacity, mirroring FEFF's `kfinmax`;
/// `active_len` is FEFF's `indmax` and selects the active prefix. The returned
/// `calculations` table contains one row for each distinct `kind`, with the
/// maximum `ljind` observed for that kind folded into column 1.
pub fn xsph_minimize_calculations(
    kind: ArrayView1<'_, i32>,
    orbital_l: ArrayView1<'_, i32>,
    final_lj: ArrayView1<'_, i32>,
    active_len: usize,
) -> Result<XsphCalculationPlan, XsphError> {
    validate_active_len("kind", kind.len(), active_len)?;
    validate_active_len("orbital_l", orbital_l.len(), active_len)?;
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;

    let mut calculations = Array2::<i32>::zeros((active_len, 3));
    let mut index_map = Array1::<i32>::zeros(active_len);

    let mut calculation_count = 1_usize;
    calculations[(0, 0)] = kind[0];
    calculations[(0, 1)] = final_lj[0];
    calculations[(0, 2)] = orbital_l[0];
    index_map[0] = 1;
    let mut max_lj = final_lj[0];

    for index in 1..active_len {
        let current_kind = kind[index];
        let current_lj = final_lj[index];
        max_lj = max_lj.max(current_lj);

        let existing = (0..calculation_count)
            .find(|&row| current_kind == calculations[(row, 0)])
            .map(|row| row + 1);

        if let Some(one_based_row) = existing {
            index_map[index] =
                -i32::try_from(one_based_row).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            let row = one_based_row - 1;
            calculations[(row, 1)] = calculations[(row, 1)].max(current_lj);
        } else {
            let row = calculation_count;
            calculation_count += 1;
            index_map[index] =
                i32::try_from(calculation_count).map_err(|_| XsphError::IndexMapOverflow {
                    index,
                    value: i32::MIN,
                })?;
            calculations[(row, 0)] = current_kind;
            calculations[(row, 1)] = current_lj;
            calculations[(row, 2)] = orbital_l[index];
        }
    }

    let compact_calculations = Array2::from_shape_fn((calculation_count, 3), |(row, column)| {
        calculations[(row, column)]
    });

    Ok(XsphCalculationPlan {
        max_lj,
        calculations: compact_calculations,
        index_map,
    })
}

/// Port of FEFF `XSPH/ljneeded0.f90`.
///
/// Returns FEFF's integer flags for angular channels `0..=ljmax` that are used
/// by the one-based shared calculation `calculation_index`.
pub fn xsph_lj_needed_flags(
    ljmax: usize,
    final_lj: ArrayView1<'_, i32>,
    index_map: ArrayView1<'_, i32>,
    active_len: usize,
    calculation_index: i32,
) -> Result<Array1<i32>, XsphError> {
    validate_active_len("final_lj", final_lj.len(), active_len)?;
    validate_active_len("index_map", index_map.len(), active_len)?;
    validate_final_lj(final_lj, active_len)?;
    if calculation_index <= 0 {
        return Err(XsphError::NonPositiveCalculationIndex { calculation_index });
    }

    let output_len = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut needed = Array1::<i32>::zeros(output_len);
    for index in 0..active_len {
        let mapped = index_map[index]
            .checked_abs()
            .ok_or(XsphError::IndexMapOverflow {
                index,
                value: index_map[index],
            })?;
        if mapped == calculation_index {
            let angular_momentum = usize::try_from(final_lj[index]).map_err(|_| {
                XsphError::NegativeAngularMomentum {
                    name: "final_lj",
                    index,
                    value: final_lj[index],
                }
            })?;
            if angular_momentum > ljmax {
                return Err(XsphError::AngularMomentumOutOfRange {
                    angular_momentum,
                    ljmax,
                });
            }
            needed[angular_momentum] = 1;
        }
    }
    Ok(needed)
}

/// Port of FEFF `XSPH/qbesselget.f90`.
///
/// Builds a Fortran-order table `j_l(qtrans * r)` with rows over radii and
/// columns over `l = 0..=ljmax`. FEFF skips Bessel evaluation and stores zeros
/// when `qtrans * r >= 1e8`; this adapter keeps the same cutoff.
pub fn xsph_q_bessel_table(
    qtrans: Real,
    radii: ArrayView1<'_, Real>,
    ljmax: usize,
) -> Result<Array2<Real>, XsphError> {
    validate_finite_real("qtrans", qtrans)?;
    if ljmax > QBESSEL_MAX_LJ {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: ljmax,
            ljmax: QBESSEL_MAX_LJ,
        });
    }

    let column_count = ljmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax })?;
    let mut table = Array2::<Real>::zeros((radii.len(), column_count).f());
    for (radius_index, &radius) in radii.iter().enumerate() {
        validate_finite_real("radius", radius)?;
        let argument = qtrans * radius;
        validate_finite_real("qtrans * radius", argument)?;
        if argument < QBESSEL_ZERO_CUTOFF {
            let values = spherical_bessel_j_y(Complex::new(argument, 0.0), ljmax)?;
            for angular_momentum in 0..=ljmax {
                table[(radius_index, angular_momentum)] = values.j[angular_momentum].re;
            }
        }
    }
    Ok(table)
}

/// Port of FEFF `XSPH/getoccnorm.f90`.
///
/// FEFF normalizes partially occupied initial states by dividing the default
/// occupation for `(Z, ihole)` by the reference occupation for `Z = 100`.
/// The inputs are the original one-based FEFF atomic number and hole selector.
/// Hole selectors whose FEFF denominator is zero are reported as errors instead
/// of returning a non-finite quotient.
pub fn xsph_occupation_normalization(
    atomic_number: usize,
    hole_index: usize,
) -> Result<Real, XsphError> {
    if !(1..=XSPH_OCC_NORM_ATOMIC_NUMBER_MAX).contains(&atomic_number) {
        return Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number,
            max_atomic_number: XSPH_OCC_NORM_ATOMIC_NUMBER_MAX,
        });
    }
    if !(1..=XSPH_OCC_NORM_HOLE_COUNT).contains(&hole_index) {
        return Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index,
            max_hole_index: XSPH_OCC_NORM_HOLE_COUNT,
        });
    }

    let numerator = xsph_occ_norm_numerator(atomic_number, hole_index).ok_or(
        XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number,
            max_atomic_number: XSPH_OCC_NORM_ATOMIC_NUMBER_MAX,
        },
    )?;
    let denominator =
        xsph_occ_norm_denominator(hole_index).ok_or(XsphError::InvalidOccupationNormHoleIndex {
            hole_index,
            max_hole_index: XSPH_OCC_NORM_HOLE_COUNT,
        })?;
    if denominator == 0 {
        return Err(XsphError::ZeroOccupationNormDenominator { hole_index });
    }

    Ok(Real::from(numerator) / Real::from(denominator))
}

/// Port of FEFF `XSPH/getholeorb0.f90`.
///
/// The surrounding FEFF routine first calls `getorb` to locate `iholep`, then
/// passes `dgc(:, iholep, 0)` and `dpc(:, iholep, 0)` through this interpolation
/// step. This pure helper accepts those selected components directly, finds the
/// last source sample above FEFF's `1e-11` tail cutoff, interpolates with
/// FEFF-compatible cubic `terp`, and zero-fills the output tail after `jnew`.
pub fn xsph_initial_hole_orbital(
    input: XsphHoleOrbitalInput<'_>,
) -> Result<XsphHoleOrbital, XsphError> {
    validate_finite_real("original_step", input.original_step)?;
    validate_finite_real("new_step", input.new_step)?;
    if input.large_component.len() != input.small_component.len() {
        return Err(XsphError::HoleOrbitalLengthMismatch {
            large_len: input.large_component.len(),
            small_len: input.small_component.len(),
        });
    }
    if input.output_count > input.output_capacity {
        return Err(XsphError::InvalidHoleOrbitalOutputCount {
            output_count: input.output_count,
            output_capacity: input.output_capacity,
        });
    }

    for (&large, &small) in input
        .large_component
        .iter()
        .zip(input.small_component.iter())
    {
        validate_finite_real("large_component", large)?;
        validate_finite_real("small_component", small)?;
    }

    let last_nonzero = input
        .large_component
        .iter()
        .zip(input.small_component.iter())
        .rposition(|(&large, &small)| {
            large.abs() >= XSPH_HOLE_ORBITAL_TAIL_CUTOFF
                || small.abs() >= XSPH_HOLE_ORBITAL_TAIL_CUTOFF
        })
        .ok_or(XsphError::EmptyHoleOrbital)?;
    let source_count = last_nonzero
        .saturating_add(2)
        .min(input.large_component.len());
    let source_x: Vec<_> = (0..source_count)
        .map(|index| -XSPH_HOLE_ORBITAL_X0 + index as Real * input.original_step)
        .collect();
    let large_source: Vec<_> = input
        .large_component
        .iter()
        .take(source_count)
        .copied()
        .collect();
    let small_source: Vec<_> = input
        .small_component
        .iter()
        .take(source_count)
        .copied()
        .collect();

    let mut large_component = Array1::<Real>::zeros(input.output_capacity);
    let mut small_component = Array1::<Real>::zeros(input.output_capacity);
    for index in 0..input.output_count {
        let x = -XSPH_HOLE_ORBITAL_X0 + index as Real * input.new_step;
        large_component[index] = terp(&source_x, &large_source, 3, x)?.value;
        small_component[index] = terp(&source_x, &small_source, 3, x)?.value;
    }

    Ok(XsphHoleOrbital {
        large_component,
        small_component,
        active_count: input.output_count,
        source_count,
    })
}

/// Port of FEFF `XSPH/axafs.f90`.
///
/// FEFF writes `axafs.dat` directly. This pure helper returns the same six data
/// columns as an `ndarray` table so callers can render the file or feed the
/// values into downstream analysis without filesystem side effects.
pub fn xsph_axafs(input: XsphAxafsInput<'_>) -> Result<XsphAxafs, XsphError> {
    validate_finite_real("fermi_energy", input.fermi_energy)?;
    validate_active_len("energies", input.energies.len(), input.horizontal_count)?;
    validate_active_len(
        "cross_section",
        input.cross_section.len(),
        input.horizontal_count,
    )?;
    if input.zero_wave_index >= input.horizontal_count {
        return Err(XsphError::InvalidAxafsGridIndex {
            zero_wave_index: input.zero_wave_index,
            horizontal_count: input.horizontal_count,
        });
    }
    let point_count = input.horizontal_count - input.zero_wave_index - 1;
    if point_count < 3 {
        return Err(XsphError::InsufficientAxafsPoints { point_count });
    }

    let reference_energy = input.energies[input.zero_wave_index];
    validate_finite_complex("energies", input.zero_wave_index, reference_energy)?;
    let mut energies = Vec::with_capacity(point_count);
    let mut absorption = Vec::with_capacity(point_count);
    for row in 0..point_count {
        let source = input.zero_wave_index + row + 1;
        let energy = input.energies[source];
        let cross_section = input.cross_section[source];
        validate_finite_complex("energies", source, energy)?;
        validate_finite_complex("cross_section", source, cross_section)?;
        energies.push((energy - reference_energy).re + input.fermi_energy);
        absorption.push(cross_section.im);
    }

    let weights = xsph_axafs_weights(&energies, input.fermi_energy);
    let mut moments = [0.0; 5];
    let mut rhs = [0.0; 3];
    for row in 0..point_count {
        let energy = energies[row];
        let weight = weights[row];
        for (power, moment) in moments.iter_mut().enumerate() {
            *moment += weight * energy.powi(power as i32);
        }
        for (power, value) in rhs.iter_mut().enumerate() {
            *value += weight * absorption[row] * energy.powi(power as i32);
        }
    }

    let coefficients = xsph_axafs_quadratic_coefficients(moments, rhs)?;
    let normalization_energy = energies[0] + 100.0 / XSPH_HARTREE_EV;
    let normalization = xsph_axafs_background(coefficients, normalization_energy);
    validate_finite_real("axafs normalization", normalization)?;
    if normalization == 0.0 {
        return Err(XsphError::ZeroAxafsNormalization);
    }

    let mut rows = Array2::<Real>::zeros((point_count, XSPH_AXAFS_COLUMN_COUNT));
    for row in 0..point_count {
        let energy = energies[row];
        let background = xsph_axafs_background(coefficients, energy);
        validate_finite_real("axafs background", background)?;
        if background == 0.0 {
            return Err(XsphError::ZeroAxafsBackground { index: row });
        }
        let relative_energy = energy - input.fermi_energy;
        let wave_number = if relative_energy >= 0.0 {
            (2.0 * relative_energy).sqrt() / XSPH_BOHR_ANGSTROM
        } else {
            -(-2.0 * relative_energy).sqrt() / XSPH_BOHR_ANGSTROM
        };
        let chi_atomic = (absorption[row] - background) / background;
        rows[(row, 0)] = energy * XSPH_HARTREE_EV;
        rows[(row, 1)] = relative_energy * XSPH_HARTREE_EV;
        rows[(row, 2)] = wave_number;
        rows[(row, 3)] = absorption[row] / normalization;
        rows[(row, 4)] = background / normalization;
        rows[(row, 5)] = chi_atomic;
        for column in 0..XSPH_AXAFS_COLUMN_COUNT {
            validate_finite_real("axafs row", rows[(row, column)])?;
        }
    }

    Ok(XsphAxafs {
        rows,
        coefficients,
        normalization,
    })
}

/// Port of FEFF `XSPH/xmultjas.f90`.
///
/// Returns the longitudinal multipole prefactor for
/// `<k|exp(i*q*z)|k'>`. FEFF declares `xm` as `complex*16`, but this helper is
/// real-valued because `xmultjas` removes the `i**ls` phase and applies it in
/// the caller.
pub fn xsph_longitudinal_multipole_factor(
    kappa: i32,
    kappa_prime: i32,
    multipole_l: i32,
) -> Result<Complex, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    if multipole_l < 0 {
        return Err(XsphError::NegativeAngularMomentum {
            name: "multipole_l",
            index: 0,
            value: multipole_l,
        });
    }

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };
    let doubled_multipole = multipole_l
        .checked_mul(2)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "multipole_l",
            value: multipole_l,
        })?;

    let parity_check =
        i64::from(j2) + i64::from(j2_prime) + i64::from(doubled_multipole) + i64::from(parity)
            - i64::from(parity_prime);
    let doubled_difference = (i64::from(j2) - i64::from(j2_prime)).abs();
    let doubled_sum = i64::from(j2) + i64::from(j2_prime);
    if parity_check.rem_euclid(4) == 0
        || i64::from(doubled_multipole) < doubled_difference
        || i64::from(doubled_multipole) > doubled_sum
    {
        return Ok(Complex::new(0.0, 0.0));
    }

    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    validate_cwig3j_doubled_argument("multipole_l", multipole_l, doubled_multipole)?;
    let angular_weight = ((f64::from(j2) + 1.0) * (f64::from(j2_prime) + 1.0)).sqrt()
        * (f64::from(doubled_multipole) + 1.0);
    let value = angular_weight * wigner_3j(j2, doubled_multipole, j2_prime, 1, 0, 2)?;
    Ok(Complex::new(value, 0.0))
}

/// Port of FEFF `XSPH/xmult.f90`.
///
/// Returns FEFF's `xm1` and `xm2` angular prefactors for Grant equation 6.30,
/// used by `radint.f90` before the relativistic radial integrals. `bessel_l`
/// is FEFF `ls`, the spherical-Bessel order, and `multipole_l` is FEFF `lb`,
/// the vector multipole order.
pub fn xsph_relativistic_multipole_factors(
    kappa: i32,
    kappa_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
) -> Result<XsphRelativisticMultipoleFactors, XsphError> {
    if kappa == 0 || kappa_prime == 0 {
        return Err(XsphError::ZeroKappa);
    }
    validate_nonnegative_angular_momentum("bessel_l", bessel_l)?;
    validate_nonnegative_angular_momentum("multipole_l", multipole_l)?;

    let Some(ls_lb_factor) = xsph_ls_lb_factor(bessel_l, multipole_l)? else {
        return Ok(XsphRelativisticMultipoleFactors {
            p_q_prime: Complex::new(0.0, 0.0),
            q_p_prime: Complex::new(0.0, 0.0),
        });
    };

    let j2 = doubled_j_from_kappa("kappa", kappa)?;
    validate_cwig3j_doubled_argument("kappa", kappa, j2)?;
    let parity = if kappa > 0 { -1 } else { 1 };
    let j2_prime = doubled_j_from_kappa("kappa_prime", kappa_prime)?;
    validate_cwig3j_doubled_argument("kappa_prime", kappa_prime, j2_prime)?;
    let parity_prime = if kappa_prime > 0 { -1 } else { 1 };

    let p_q_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 - parity) / 2,
        (j2_prime + parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        1.0,
    )?;
    let q_p_prime = xsph_relativistic_multipole_component(
        ls_lb_factor,
        (j2 + parity) / 2,
        (j2_prime - parity_prime) / 2,
        bessel_l,
        multipole_l,
        j2,
        j2_prime,
        -1.0,
    )?;

    Ok(XsphRelativisticMultipoleFactors {
        p_q_prime,
        q_p_prime,
    })
}

/// Port of FEFF `XSPH/acoef.f90`.
///
/// Builds the energy-independent angular matrix used by `szlz` for
/// occupation, spin, orbital, and magnetic-dipole density channels. The output
/// is Fortran-order with shape `(2 * lmax + 1, 2, 2, 3, lmax + 1)`. The first
/// axis stores magnetic quantum numbers as `ml + lmax`; the second and third
/// axes are FEFF's two relativistic branches; the fourth axis stores the three
/// operator channels; and the last axis stores `l = 0..=lmax`.
///
/// FEFF declares the coefficient table and intermediate angular factors as
/// default `real`, so this routine rounds the same inner products through
/// `f32` before exposing the values as [`Real`].
pub fn xsph_angular_density_coefficients(
    spin_selector: i32,
    lmax: usize,
) -> Result<Array5<Real>, XsphError> {
    if lmax > XSPH_MAX_LX {
        return Err(XsphError::AngularMomentumOutOfRange {
            angular_momentum: lmax,
            ljmax: XSPH_MAX_LX,
        });
    }
    let abs_spin = spin_selector
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange {
            name: "spin_selector",
            value: spin_selector,
        })?;
    let lmax_i32 = usize_to_i32("lmax", lmax)?;
    let ml_count = lmax
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: lmax })?;
    let l_count = lmax
        .checked_add(1)
        .ok_or(XsphError::AngularMomentumCapacityOverflow { ljmax: lmax })?;

    let mut coefficients = Array5::<Real>::zeros((ml_count, 2, 2, 3, l_count).f());
    let mut t3j = vec![0.0_f32; l_count * l_count * 2];
    let spin_projection = if spin_selector < 0 { 0_i32 } else { 1_i32 };

    for ml in -lmax_i32..=lmax_i32 {
        let mj = 2 * ml + (2 * spin_projection - 1);
        let magnetic_j = 0.5_f32 * mj as f32;
        let cwig_mj = -mj;
        let ml_index =
            usize::try_from(ml + lmax_i32).map_err(|_| XsphError::IntegerOutOfRange {
                name: "magnetic_l",
                value: ml,
            })?;

        for lp in 0..=lmax {
            let lp_i32 = usize_to_i32("lp", lp)?;
            let lp2 = lp_i32.checked_mul(2).ok_or(XsphError::SizeOutOfRange {
                name: "lp",
                value: lp,
            })?;
            for jp in 0..=lmax {
                let jp_i32 = usize_to_i32("jp", jp)?;
                let jp2 = jp_i32
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(XsphError::SizeOutOfRange {
                        name: "jp",
                        value: jp,
                    })?;
                for mp in 0..=1 {
                    let mp2 = 2 * usize_to_i32("mp", mp)? - 1;
                    let angular = wigner_3j(1, jp2, lp2, mp2, cwig_mj, 2)? as f32;
                    t3j[t3j_index(lp, jp, mp, l_count)] =
                        alternating_sign(lp_i32) as f32 * ((jp2 as f32) + 1.0).sqrt() * angular;
                }
            }
        }

        for lpp in 0..=lmax {
            let lpp_i32 = usize_to_i32("lpp", lpp)?;
            let mut operator = [[[0.0_f32; 3]; 2]; 2];
            for m1 in 0..=1 {
                for m2 in 0..=1 {
                    for (iop, operator_value) in operator[m1][m2].iter_mut().enumerate() {
                        if m1 == m2 {
                            let spin_m = m1 as f32 - 0.5;
                            let orbital_m = magnetic_j - spin_m;
                            if (ml + spin_projection - usize_to_i32("m1", m1)?).abs() <= lpp_i32 {
                                if spin_selector == 0 {
                                    *operator_value = 2.0;
                                } else if iop == 0 {
                                    *operator_value = spin_m;
                                } else if iop == 1 && abs_spin == 1 {
                                    *operator_value = orbital_m;
                                } else if iop == 1 && abs_spin == 2 {
                                    *operator_value = 1.0;
                                } else if iop == 2 && abs_spin == 1 {
                                    let numerator = spin_m
                                        * 2.0
                                        * (3.0 * orbital_m * orbital_m
                                            - (lpp_i32 * (lpp_i32 + 1)) as f32);
                                    *operator_value = numerator
                                        / (2 * lpp_i32 + 3) as f32
                                        / (2 * lpp_i32 - 1) as f32;
                                } else if iop == 2 && abs_spin == 2 {
                                    let value = t3j[t3j_index(lpp, lpp, m1, l_count)];
                                    *operator_value = value * value;
                                }
                            }
                        } else if iop == 2
                            && abs_spin <= 1
                            && nint(0.5 + f64::from(magnetic_j.abs())) < lpp_i32
                        {
                            let radial = ((lpp_i32 * (lpp_i32 + 1)) as f32
                                - (magnetic_j * magnetic_j - 0.25))
                                .sqrt();
                            *operator_value = 3.0 * magnetic_j * radial
                                / (2 * lpp_i32 + 3) as f32
                                / (2 * lpp_i32 - 1) as f32;
                        } else if iop == 2 && abs_spin > 1 {
                            *operator_value = t3j[t3j_index(lpp, lpp, m1, l_count)]
                                * t3j[t3j_index(lpp, lpp, m2, l_count)];
                        }
                    }
                }
            }

            for branch_1 in 0..=1 {
                let Some(jj) = xsph_acoef_j(branch_1, lpp) else {
                    continue;
                };
                for branch_2 in 0..=1 {
                    let Some(jp) = xsph_acoef_j(branch_2, lpp) else {
                        continue;
                    };
                    for iop in 0..3 {
                        let mut value =
                            coefficients[(ml_index, branch_1, branch_2, iop, lpp)] as f32;
                        for m2 in 0..=1 {
                            for m1 in 0..=1 {
                                value += operator[m1][m2][iop]
                                    * t3j[t3j_index(lpp, jp, spin_projection as usize, l_count)]
                                    * t3j[t3j_index(lpp, jp, m1, l_count)]
                                    * t3j[t3j_index(lpp, jj, m2, l_count)]
                                    * t3j[t3j_index(lpp, jj, spin_projection as usize, l_count)];
                            }
                        }
                        coefficients[(ml_index, branch_1, branch_2, iop, lpp)] = value as Real;
                    }
                }
            }
        }
    }

    Ok(coefficients)
}

fn validate_active_len(
    name: &'static str,
    actual: usize,
    active_len: usize,
) -> Result<(), XsphError> {
    if active_len == 0 {
        return Err(XsphError::EmptyIndexSet);
    }
    if actual < active_len {
        return Err(XsphError::LengthTooShort {
            name,
            required: active_len,
            actual,
        });
    }
    Ok(())
}

fn t3j_index(lp: usize, jp: usize, mp: usize, l_count: usize) -> usize {
    ((lp * l_count) + jp) * 2 + mp
}

fn xsph_acoef_j(branch: usize, lpp: usize) -> Option<usize> {
    if branch == 0 {
        if lpp == 0 { None } else { Some(lpp - 1) }
    } else {
        Some(lpp)
    }
}

fn validate_final_lj(final_lj: ArrayView1<'_, i32>, active_len: usize) -> Result<(), XsphError> {
    for index in 0..active_len {
        let value = final_lj[index];
        if value < 0 {
            return Err(XsphError::NegativeAngularMomentum {
                name: "final_lj",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_finite_real(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteScalar { name, value })
    }
}

fn validate_phase_mesh_capacity(capacity: usize) -> Result<(), XsphError> {
    if capacity == 0 {
        Err(XsphError::InvalidPhaseMeshCapacity { capacity })
    } else {
        Ok(())
    }
}

fn validate_phase_mesh_step(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() && value != 0.0 {
        Ok(())
    } else {
        Err(XsphError::InvalidPhaseMeshStep { name, value })
    }
}

fn validate_phase_mesh_endpoint(name: &'static str, value: Real) -> Result<(), XsphError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(XsphError::InvalidPhaseMeshEndpoint { name, value })
    }
}

fn phase_mesh_count(final_offset: i32) -> Result<usize, XsphError> {
    usize::try_from(final_offset)
        .map_err(|_| XsphError::IntegerOutOfRange {
            name: "phase_mesh_offset",
            value: final_offset,
        })?
        .checked_add(1)
        .ok_or(XsphError::IntegerOutOfRange {
            name: "phase_mesh_offset",
            value: final_offset,
        })
}

fn append_phase_mesh_segment(values: &mut Vec<Complex>, capacity: usize, segment: Array1<Complex>) {
    let remaining = capacity.saturating_sub(values.len());
    values.extend(segment.iter().take(remaining).copied());
}

fn append_danes_phase_extension(
    values: &mut Vec<Complex>,
    horizontal_count: usize,
    capacity: usize,
) -> Result<usize, XsphError> {
    let extension_limit = capacity.min(150);
    let extension_slots = extension_limit.saturating_sub(values.len());
    if extension_slots <= 1 {
        return Ok(0);
    }

    let extension_count = extension_slots - 1;
    let previous = horizontal_count
        .checked_sub(2)
        .ok_or(XsphError::InvalidPhaseMeshCapacity { capacity })?;
    let min_energy = 2.0 * values[horizontal_count - 1].re - values[previous].re;
    let max_energy = 7.0e4;
    validate_phase_mesh_endpoint("danes_min_energy", min_energy)?;
    let exponent_step = (max_energy / min_energy).ln() / extension_count as Real;
    validate_phase_mesh_step("danes_exponent_step", exponent_step)?;
    values.extend(
        (0..extension_count)
            .map(|index| Complex::new(min_energy * (exponent_step * index as Real).exp(), 2.0e-8)),
    );
    Ok(extension_count)
}

fn xsph_thermal_contour_height(temperature: Real) -> Result<(usize, Real), XsphError> {
    validate_phase_mesh_endpoint("thermal_temperature", temperature)?;
    let minimum_height = 0.05;
    let period = 2.0 * std::f64::consts::PI * temperature;
    validate_phase_mesh_endpoint("thermal_period", period)?;
    let pole_count = if period < minimum_height {
        let count = (minimum_height / period).ceil();
        validate_phase_mesh_endpoint("thermal_pole_count", count)?;
        count as usize
    } else {
        1
    };
    let upper_imaginary = pole_count as Real * period;
    validate_phase_mesh_endpoint("thermal_upper_imaginary", upper_imaginary)?;
    Ok((pole_count, upper_imaginary))
}

fn xsph_default_thermal_horizontal_grid(
    core_valence_separation: Real,
    upper_imaginary: Real,
) -> Result<Array1<Complex>, XsphError> {
    validate_finite_real("core_valence_separation", core_valence_separation)?;
    validate_phase_mesh_endpoint("thermal_upper_imaginary", upper_imaginary)?;
    let maximum_energy = 5.8;
    let trial_step = upper_imaginary / 4.0;
    validate_phase_mesh_endpoint("thermal_trial_step", trial_step)?;

    let below_count = ((0.0 - core_valence_separation) / trial_step)
        .ceil()
        .min(21.0);
    validate_phase_mesh_endpoint("thermal_below_count", below_count)?;
    let below_count = below_count as usize;
    let energy_step = (0.0 - core_valence_separation) / below_count as Real;
    validate_phase_mesh_endpoint("thermal_energy_step", energy_step)?;

    let horizontal_count = ((maximum_energy - core_valence_separation) / energy_step).ceil();
    validate_phase_mesh_endpoint("thermal_horizontal_count", horizontal_count)?;
    let horizontal_count = horizontal_count as usize;
    Ok(Array1::from_shape_fn(horizontal_count, |index| {
        Complex::new(
            core_valence_separation + (index + 1) as Real * energy_step,
            0.0,
        )
    }))
}

fn xsph_thermal_phase_mesh_count(
    horizontal_count: usize,
    pole_count: usize,
) -> Result<usize, XsphError> {
    horizontal_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(10))
        .and_then(|count| count.checked_add(pole_count))
        .and_then(|count| count.checked_add(1))
        .ok_or(XsphError::SizeOutOfRange {
            name: "thermal_phase_mesh_count",
            value: horizontal_count,
        })
}

#[derive(Debug, Clone, Copy)]
struct ResolvedPhaseUserGrid {
    kind: XsphPhaseUserGridKind,
    minimum: Real,
    maximum: Real,
    step: Real,
}

#[derive(Debug, Clone, Copy)]
struct PreviousPhaseUserGrid {
    kind: XsphPhaseUserGridKind,
    maximum: Real,
}

fn xsph_user_phase_horizontal_grid(
    records: &[XsphPhaseUserGridRecord<'_>],
    capacity: usize,
) -> Result<Vec<Complex>, XsphError> {
    validate_phase_mesh_capacity(capacity)?;

    let mut user_points = Vec::new();
    let mut regular_records = Vec::new();
    let mut previous = None;

    for (record_index, record) in records.iter().enumerate() {
        match record {
            XsphPhaseUserGridRecord::Regular(record) => {
                let resolved = resolve_phase_user_regular_grid(record_index, *record, previous)?;
                previous = Some(PreviousPhaseUserGrid {
                    kind: resolved.kind,
                    maximum: resolved.maximum,
                });
                regular_records.push(resolved);
            }
            XsphPhaseUserGridRecord::User(points) => {
                if points.is_empty() {
                    return Err(XsphError::EmptyPhaseGridRecords);
                }
                for (point_index, &point) in points.iter().enumerate() {
                    validate_finite_complex("user_grid", point_index, point)?;
                    if user_points.len() < capacity {
                        user_points.push(point / XSPH_HARTREE_EV);
                    }
                }
                let last = points[points.len() - 1].re;
                validate_finite_real("user_grid_maximum", last)?;
                previous = Some(PreviousPhaseUserGrid {
                    kind: XsphPhaseUserGridKind::Energy,
                    maximum: last,
                });
            }
        }
    }

    let mut values = user_points;
    for record in regular_records {
        append_phase_user_regular_grid(&mut values, capacity, record)?;
    }

    Ok(values)
}

fn validate_phase_user_grid_records(
    records: &[XsphPhaseUserGridRecord<'_>],
) -> Result<(), XsphError> {
    if records.is_empty() {
        return Err(XsphError::EmptyPhaseGridRecords);
    }
    if records.len() > XSPH_USER_PHASE_GRID_MAX_RECORDS {
        return Err(XsphError::TooManyPhaseGridRecords {
            count: records.len(),
            max: XSPH_USER_PHASE_GRID_MAX_RECORDS,
        });
    }
    Ok(())
}

fn resolve_phase_user_regular_grid(
    _record_index: usize,
    record: XsphPhaseUserRegularGrid,
    previous: Option<PreviousPhaseUserGrid>,
) -> Result<ResolvedPhaseUserGrid, XsphError> {
    validate_finite_real("user_grid_maximum", record.maximum)?;
    validate_phase_mesh_endpoint("user_grid_step", record.step)?;
    let minimum = match record.minimum {
        XsphPhaseUserGridMinimum::Value(value) => {
            validate_finite_real("user_grid_minimum", value)?;
            value
        }
        XsphPhaseUserGridMinimum::Last => {
            let value = previous.map_or(0.0, |previous| {
                phase_user_last_minimum(record.kind, previous, record.step)
            });
            validate_finite_real("user_grid_last_minimum", value)?;
            value
        }
    };
    validate_finite_real("user_grid_minimum", minimum)?;
    Ok(ResolvedPhaseUserGrid {
        kind: record.kind,
        minimum,
        maximum: record.maximum,
        step: record.step,
    })
}

fn phase_user_last_minimum(
    current_kind: XsphPhaseUserGridKind,
    previous: PreviousPhaseUserGrid,
    step: Real,
) -> Real {
    let current_is_k = current_kind == XsphPhaseUserGridKind::WaveNumber;
    let previous_is_k = previous.kind == XsphPhaseUserGridKind::WaveNumber;
    if current_is_k == previous_is_k {
        previous.maximum + step
    } else if current_is_k {
        (2.0 * previous.maximum / XSPH_HARTREE_EV).sqrt() / XSPH_BOHR_ANGSTROM + step
    } else {
        (previous.maximum * XSPH_BOHR_ANGSTROM).powi(2) / 2.0 * XSPH_HARTREE_EV + step
    }
}

fn append_phase_user_regular_grid(
    values: &mut Vec<Complex>,
    capacity: usize,
    record: ResolvedPhaseUserGrid,
) -> Result<(), XsphError> {
    if values.len() >= capacity {
        return Ok(());
    }
    let remaining = capacity - values.len();
    let segment = match record.kind {
        XsphPhaseUserGridKind::Energy => xsph_even_energy_mesh(
            record.minimum / XSPH_HARTREE_EV,
            record.maximum / XSPH_HARTREE_EV,
            record.step / XSPH_HARTREE_EV,
            remaining,
        )?,
        XsphPhaseUserGridKind::WaveNumber => xsph_k_energy_mesh(
            record.minimum * XSPH_BOHR_ANGSTROM,
            record.maximum * XSPH_BOHR_ANGSTROM,
            record.step * XSPH_BOHR_ANGSTROM,
            remaining,
        )?,
        XsphPhaseUserGridKind::Exponential => {
            let minimum = record.minimum / XSPH_HARTREE_EV;
            let maximum = record.maximum / XSPH_HARTREE_EV;
            let step = record.step / XSPH_HARTREE_EV;
            let span = maximum - minimum + 1.0;
            validate_phase_mesh_endpoint("user_exp_grid_span", span)?;
            let exponential = xsph_exponential_energy_mesh(1.0, span, step, remaining)?;
            Array1::from_iter(
                exponential
                    .iter()
                    .map(|energy| Complex::new(energy.re + minimum - 1.0, 0.0)),
            )
        }
    };
    append_phase_mesh_segment(values, capacity, segment);
    Ok(())
}

fn validate_nonnegative_angular_momentum(name: &'static str, value: i32) -> Result<(), XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum {
            name,
            index: 0,
            value,
        })
    } else {
        Ok(())
    }
}

fn validate_indexed_angular_momentum(
    name: &'static str,
    index: usize,
    value: i32,
) -> Result<i32, XsphError> {
    if value < 0 {
        Err(XsphError::NegativeAngularMomentum { name, index, value })
    } else {
        Ok(value)
    }
}

fn usize_to_i32(name: &'static str, value: usize) -> Result<i32, XsphError> {
    i32::try_from(value).map_err(|_| XsphError::SizeOutOfRange { name, value })
}

fn xsph_axafs_weights(energies: &[Real], fermi_energy: Real) -> Vec<Real> {
    let point_count = energies.len();
    (0..point_count)
        .map(|row| {
            if row == 0 {
                (energies[1] - fermi_energy) * (energies[0] - fermi_energy).abs()
            } else if row + 1 == point_count {
                (energies[row] - energies[row - 1]) * (energies[row] - fermi_energy)
            } else {
                (energies[row + 1] - energies[row - 1]) * (energies[row] - fermi_energy)
            }
        })
        .collect()
}

fn xsph_axafs_quadratic_coefficients(
    moments: [Real; 5],
    rhs: [Real; 3],
) -> Result<[Real; 3], XsphError> {
    let base = Array2::from_shape_fn((3, 3), |(row, column)| moments[row + column]);
    let denominator = feff_determinant(base.view())?;
    validate_finite_real("axafs fit denominator", denominator)?;
    if denominator == 0.0 {
        return Err(XsphError::SingularAxafsFit);
    }

    let mut coefficients = [0.0; 3];
    for (target_column, coefficient) in coefficients.iter_mut().enumerate() {
        let matrix = Array2::from_shape_fn((3, 3), |(row, column)| {
            if column == target_column {
                rhs[row]
            } else {
                moments[row + column]
            }
        });
        let numerator = feff_determinant(matrix.view())?;
        validate_finite_real("axafs fit numerator", numerator)?;
        *coefficient = numerator / denominator;
        validate_finite_real("axafs fit coefficient", *coefficient)?;
    }
    Ok(coefficients)
}

fn xsph_axafs_background(coefficients: [Real; 3], energy: Real) -> Real {
    coefficients[0] + coefficients[1] * energy + coefficients[2] * energy * energy
}

fn validate_finite_complex(
    name: &'static str,
    index: usize,
    value: Complex,
) -> Result<(), XsphError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(XsphError::NonFiniteComplex {
            name,
            index,
            real: value.re,
            imaginary: value.im,
        })
    }
}

fn doubled_j_from_kappa(name: &'static str, kappa: i32) -> Result<i32, XsphError> {
    let abs_kappa = kappa
        .checked_abs()
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })?;
    abs_kappa
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(XsphError::IntegerOutOfRange { name, value: kappa })
}

fn validate_cwig3j_doubled_argument(
    name: &'static str,
    original_value: i32,
    doubled_value: i32,
) -> Result<(), XsphError> {
    if doubled_value <= CWIG3J_MAX_DOUBLED_ARGUMENT {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange {
            name,
            value: original_value,
        })
    }
}

fn validate_cwig3j_integer_argument(name: &'static str, value: i32) -> Result<(), XsphError> {
    if value <= CWIG3J_MAX_DOUBLED_ARGUMENT / 2 {
        Ok(())
    } else {
        Err(XsphError::IntegerOutOfRange { name, value })
    }
}

fn xsph_ls_lb_factor(bessel_l: i32, multipole_l: i32) -> Result<Option<Complex>, XsphError> {
    let real_factor = if multipole_l > 0 && bessel_l == multipole_l - 1 {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l - 1) * f64::from(multipole_l + 1) / 2.0;
        value.sqrt()
    } else if bessel_l > 0 && bessel_l - 1 == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        let value = f64::from(2 * multipole_l + 3) * f64::from(multipole_l) / 2.0;
        value.sqrt()
    } else if bessel_l == multipole_l {
        validate_cwig3j_integer_argument("bessel_l", bessel_l)?;
        validate_cwig3j_integer_argument("multipole_l", multipole_l)?;
        f64::from(2 * multipole_l + 1) / 2.0_f64.sqrt()
    } else {
        return Ok(None);
    };
    Ok(Some(imaginary_unit_power(bessel_l) * real_factor))
}

#[allow(clippy::too_many_arguments)]
fn xsph_relativistic_multipole_component(
    ls_lb_factor: Complex,
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    multipole_l: i32,
    j2: i32,
    j2_prime: i32,
    imaginary_sign: Real,
) -> Result<Complex, XsphError> {
    let nine_j = xsph_nine_j(lambda, lambda_prime, bessel_l, j2, j2_prime, multipole_l);
    let three_j = wigner_3j(lambda, bessel_l, lambda_prime, 0, 0, 1)?;
    let angular_weight = (6.0
        * (f64::from(j2) + 1.0)
        * (f64::from(j2_prime) + 1.0)
        * f64::from(2 * multipole_l + 1)
        * f64::from(2 * lambda + 1)
        * f64::from(2 * lambda_prime + 1))
    .sqrt();
    Ok(ls_lb_factor
        * (nine_j * three_j * alternating_sign(lambda) * angular_weight)
        * Complex::new(0.0, imaginary_sign))
}

fn xsph_nine_j(
    lambda: i32,
    lambda_prime: i32,
    bessel_l: i32,
    j2: i32,
    j2_prime: i32,
    multipole_l: i32,
) -> Real {
    if bessel_l > multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                2 * multipole_l,
                bessel_l + multipole_l,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * bessel_l,
                2 * lambda,
                j2,
                2 * lambda_prime,
            )
    } else if bessel_l < multipole_l {
        -f64::from(bessel_l + multipole_l + 1)
            * xsph_six_j(1, 2, 2 * multipole_l, bessel_l + multipole_l, 2 * bessel_l)
            * xsph_six_j(
                bessel_l + multipole_l,
                2 * multipole_l,
                j2_prime,
                2 * lambda_prime,
                j2,
            )
            * xsph_six_j(
                2 * bessel_l,
                bessel_l + multipole_l,
                j2,
                2 * lambda,
                2 * lambda_prime,
            )
    } else {
        let first_term = -f64::from(2 * bessel_l + 2)
            * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l + 1, 2 * multipole_l)
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                2 * lambda_prime,
                j2_prime,
                j2,
            )
            * xsph_six_j(
                2 * multipole_l,
                2 * multipole_l + 1,
                j2,
                2 * lambda,
                2 * lambda_prime,
            );
        if bessel_l == 0 {
            first_term
        } else {
            first_term
                - f64::from(2 * bessel_l)
                    * xsph_six_j(1, 2, 2 * multipole_l, 2 * multipole_l - 1, 2 * multipole_l)
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        j2_prime,
                        2 * lambda_prime,
                        j2,
                    )
                    * xsph_six_j(
                        2 * multipole_l - 1,
                        2 * multipole_l,
                        2 * lambda,
                        j2,
                        2 * lambda_prime,
                    )
        }
    }
}

fn xsph_six_j(j1: i32, j2: i32, j3: i32, j4: i32, j5: i32) -> Real {
    if j2 != j1 + 1 {
        return 0.0;
    }
    if j4 == j3 + 1 {
        let g2 = j5 - 1;
        if g2 < (j1 - j3).abs() || g2 > j1 + j3 {
            return 0.0;
        }
        let value = (1.0 + f64::from(g2 + j1 - j3) / 2.0) * (1.0 + f64::from(g2 - j1 + j3) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j3 + 1)
            / f64::from(j3 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j3) / 2.0))
    } else if j3 == j4 + 1 {
        let g2 = j5;
        if g2 < (j1 - j4).abs() || g2 > j1 + j4 {
            return 0.0;
        }
        let value = (1.0 - f64::from(g2 - j1 - j4) / 2.0) * (2.0 + f64::from(g2 + j1 + j4) / 2.0)
            / f64::from(j1 + 1)
            / f64::from(j1 + 2)
            / f64::from(j4 + 1)
            / f64::from(j4 + 2);
        value.sqrt() * alternating_sign(nint(1.0 + f64::from(g2 + j1 + j4) / 2.0))
    } else {
        0.0
    }
}

fn nint(value: Real) -> i32 {
    value.round() as i32
}

fn alternating_sign(exponent: i32) -> Real {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn imaginary_unit_power(exponent: i32) -> Complex {
    match exponent.rem_euclid(4) {
        0 => Complex::new(1.0, 0.0),
        1 => Complex::new(0.0, 1.0),
        2 => Complex::new(-1.0, 0.0),
        _ => Complex::new(0.0, -1.0),
    }
}

#[cfg(test)]
mod tests;
