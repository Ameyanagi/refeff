//! Small XSPH planning helpers ported from FEFF.
//!
//! The full XSPH phase-shift driver is still being ported incrementally. This
//! module contains self-contained helper kernels from `XSPH/mincalc.f90` and
//! `XSPH/ljneeded0.f90` that decide which final-state calculations can be
//! shared and which angular channels are needed for each shared calculation.

use ndarray::{Array1, Array2, ArrayView1};
use thiserror::Error;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
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

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;

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
    }
}
