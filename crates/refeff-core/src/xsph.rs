//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels from `XSPH/mincalc.f90` and
//! `XSPH/ljneeded0.f90` that decide which final-state calculations can be
//! shared and which angular channels are needed for each shared calculation.

use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder};
use thiserror::Error;

use crate::{AngularError, BesselError, Complex, Real, spherical_bessel_j_y, wigner_3j};

const QBESSEL_MAX_LJ: usize = 39;
const QBESSEL_ZERO_CUTOFF: Real = 1.0e8;
const CWIG3J_MAX_DOUBLED_ARGUMENT: i32 = 116;

/// Shared final-state calculation plan returned by [`xsph_minimize_calculations`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XsphCalculationPlan {
    /// Maximum `lj` encountered in the active final-state index list, FEFF `ljj`.
    pub max_lj: i32,
    /// Rows `[kind, max_lj_for_kind, representative_l]`, FEFF `indcalc`.
    pub calculations: Array2<i32>,
    /// Per-final-state map to a calculation row, FEFF `indmap`.
    ///
    /// Positive values mark the first occurrence of a final-state `kind`.
    /// Negative values reuse the absolute calculation index from an earlier
    /// occurrence, matching FEFF's convention.
    pub index_map: Array1<i32>,
}

/// Error returned by XSPH planning helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum XsphError {
    /// FEFF `mincalc` expects at least one active final-state index.
    #[error("XSPH calculation planning requires at least one active index")]
    EmptyIndexSet,
    /// A supplied index row is shorter than the requested active prefix.
    #[error("{name} length {actual} is shorter than active length {required}")]
    LengthTooShort {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// Angular momentum indices used as output slots must be non-negative.
    #[error("{name} entry {index} must be non-negative, got {value}")]
    NegativeAngularMomentum {
        name: &'static str,
        index: usize,
        value: i32,
    },
    /// FEFF `ljneeded0` would stop when an `lj` index exceeds `ljmax`.
    #[error("XSPH angular momentum {angular_momentum} exceeds ljmax {ljmax}")]
    AngularMomentumOutOfRange {
        angular_momentum: usize,
        ljmax: usize,
    },
    /// Shared calculation indices are one-based in FEFF.
    #[error("XSPH calculation index must be positive, got {calculation_index}")]
    NonPositiveCalculationIndex { calculation_index: i32 },
    /// The FEFF map convention cannot represent `abs(i32::MIN)`.
    #[error("XSPH index map entry {index} cannot be negated: {value}")]
    IndexMapOverflow { index: usize, value: i32 },
    /// Requested output size overflows `usize`.
    #[error("XSPH ljmax {ljmax} cannot be represented as an output vector length")]
    AngularMomentumCapacityOverflow { ljmax: usize },
    /// XSPH scalar inputs must be finite.
    #[error("{name} must be finite, got {value}")]
    NonFiniteScalar { name: &'static str, value: Real },
    /// Spherical Bessel evaluation failed.
    #[error(transparent)]
    Bessel(#[from] BesselError),
    /// Wigner-symbol evaluation failed.
    #[error(transparent)]
    Angular(#[from] AngularError),
    /// Relativistic kappa values must be nonzero.
    #[error("XSPH relativistic kappa must be nonzero")]
    ZeroKappa,
    /// Integer angular inputs must stay in the supported FEFF range.
    #[error("{name} value {value} is outside the supported XSPH integer range")]
    IntegerOutOfRange { name: &'static str, value: i32 },
}

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

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} expected {expected}"
        );
    }

    #[test]
    fn xsph_minimize_calculations_matches_feff_reference() -> Result<(), XsphError> {
        let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
        let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

        let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 8)?;

        assert_eq!(plan.max_lj, 6);
        assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2, 4, -3, -1]));
        assert_eq!(
            plan.calculations,
            arr2(&[[2, 5, 1], [4, 4, 2], [-3, 6, 1], [5, 0, 0]])
        );
        Ok(())
    }

    #[test]
    fn xsph_minimize_calculations_honors_active_prefix() -> Result<(), XsphError> {
        let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
        let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

        let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 5)?;

        assert_eq!(plan.max_lj, 5);
        assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2]));
        assert_eq!(plan.calculations, arr2(&[[2, 5, 1], [4, 4, 2], [-3, 3, 1]]));
        Ok(())
    }

    #[test]
    fn xsph_lj_needed_flags_match_feff_reference() -> Result<(), XsphError> {
        let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);
        let index_map = arr1(&[1, 2, -1, 3, -2, 4, -3, -1]);

        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 1)?,
            arr1(&[0, 1, 1, 0, 0, 1, 0])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 2)?,
            arr1(&[0, 1, 0, 0, 1, 0, 0])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 3)?,
            arr1(&[0, 0, 0, 1, 0, 0, 1])
        );
        assert_eq!(
            xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 4)?,
            arr1(&[1, 0, 0, 0, 0, 0, 0])
        );
        Ok(())
    }

    #[test]
    fn xsph_q_bessel_table_matches_feff_reference() -> Result<(), XsphError> {
        let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
        let table = xsph_q_bessel_table(0.35, radii.view(), 4)?;

        assert_eq!(table.shape(), &[4, 5]);
        assert_eq!(table.strides(), &[1, 4]);
        let expected = arr2(&[
            [
                9.997_958_458_381_769e-1,
                1.166_523_756_252_462e-2,
                8.165_952_107_648_562e-5,
                4.083_055_447_551_5e-7,
                1.587_874_544_380_937_5e-9,
            ],
            [
                9.797_080_213_012_896e-1,
                1.152_437_384_397_447_3e-1,
                8.095_451_039_379_387e-3,
                4.055_621_228_179_726_3e-4,
                1.579_141_698_006_595_3e-5,
            ],
            [
                8.261_173_577_085_878e-1,
                3.129_012_474_446_291e-1,
                6.788_620_641_892_411e-2,
                1.036_640_216_929_531_6e-2,
                1.223_141_376_378_009_1e-3,
            ],
            [
                9.385_522_838_839_835e-2,
                -9.429_243_227_927_261e-2,
                -1.342_662_707_938_009e-1,
                -1.612_046_859_156_612_8e-3,
                1.326_542_239_346_443e-1,
            ],
        ]);
        for ((row, column), &expected_value) in expected.indexed_iter() {
            assert_close(table[(row, column)], expected_value);
        }
        Ok(())
    }

    #[test]
    fn xsph_q_bessel_table_applies_feff_large_argument_cutoff() -> Result<(), XsphError> {
        let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
        let table = xsph_q_bessel_table(1.0e8, radii.view(), 4)?;

        let expected_first_row = [
            4.205_477_931_907_825e-8,
            9.072_704_282_365_188e-8,
            -4.205_475_210_096_54e-8,
            -9.072_706_385_102_794e-8,
            4.205_468_859_202_071e-8,
        ];
        for (column, &expected_value) in expected_first_row.iter().enumerate() {
            assert_close(table[(0, column)], expected_value);
        }
        for row in 1..4 {
            for column in 0..5 {
                assert_close(table[(row, column)], 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn xsph_longitudinal_multipole_factor_matches_feff_reference() -> Result<(), XsphError> {
        let cases = [
            (-1, -1, 0, -std::f64::consts::SQRT_2),
            (-1, 1, 1, 2.449_489_742_783_178),
            (1, -1, 1, 2.449_489_742_783_178),
            (-2, 1, 1, 0.0),
            (2, -1, 2, -4.472_135_954_999_58),
            (-3, 2, 3, 0.0),
            (3, -2, 2, 2.927_700_218_845_598),
            (-2, -2, 5, 0.0),
        ];

        for (kappa, kappa_prime, multipole_l, expected) in cases {
            let value = xsph_longitudinal_multipole_factor(kappa, kappa_prime, multipole_l)?;
            assert_close(value.re, expected);
            assert_close(value.im, 0.0);
        }
        Ok(())
    }

    #[test]
    fn xsph_planning_helpers_reject_invalid_inputs() {
        let kind = arr1(&[2]);
        let orbital_l = arr1(&[1]);
        let final_lj = arr1(&[2]);

        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 0),
            Err(XsphError::EmptyIndexSet)
        ));
        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 2),
            Err(XsphError::LengthTooShort { name: "kind", .. })
        ));

        let bad_lj = arr1(&[-1]);
        assert!(matches!(
            xsph_minimize_calculations(kind.view(), orbital_l.view(), bad_lj.view(), 1),
            Err(XsphError::NegativeAngularMomentum { .. })
        ));

        let index_map = arr1(&[1]);
        assert!(matches!(
            xsph_lj_needed_flags(1, final_lj.view(), index_map.view(), 1, 1),
            Err(XsphError::AngularMomentumOutOfRange { .. })
        ));
        assert!(matches!(
            xsph_lj_needed_flags(2, final_lj.view(), index_map.view(), 1, 0),
            Err(XsphError::NonPositiveCalculationIndex { .. })
        ));

        let overflow_map = arr1(&[i32::MIN]);
        assert!(matches!(
            xsph_lj_needed_flags(2, final_lj.view(), overflow_map.view(), 1, 1),
            Err(XsphError::IndexMapOverflow { .. })
        ));

        let radii = arr1(&[1.0]);
        assert!(matches!(
            xsph_q_bessel_table(Real::NAN, radii.view(), 4),
            Err(XsphError::NonFiniteScalar { name: "qtrans", .. })
        ));
        let bad_radii = arr1(&[Real::INFINITY]);
        assert!(matches!(
            xsph_q_bessel_table(1.0, bad_radii.view(), 4),
            Err(XsphError::NonFiniteScalar { name: "radius", .. })
        ));
        assert!(matches!(
            xsph_q_bessel_table(1.0, radii.view(), 40),
            Err(XsphError::AngularMomentumOutOfRange {
                angular_momentum: 40,
                ljmax: 39,
            })
        ));
        assert!(matches!(
            xsph_q_bessel_table(0.0, radii.view(), 4),
            Err(XsphError::Bessel(
                BesselError::NonPositiveRealArgument { .. }
            ))
        ));

        assert!(matches!(
            xsph_longitudinal_multipole_factor(0, 1, 1),
            Err(XsphError::ZeroKappa)
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(1, 1, -1),
            Err(XsphError::NegativeAngularMomentum {
                name: "multipole_l",
                ..
            })
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(i32::MIN, 1, 1),
            Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
        ));
        assert!(matches!(
            xsph_longitudinal_multipole_factor(60, -60, 1),
            Err(XsphError::IntegerOutOfRange { name: "kappa", .. })
        ));
    }
}
