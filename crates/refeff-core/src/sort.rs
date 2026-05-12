//! Companion-index sorting helpers for FEFF arrays.
//!
//! FEFF `QSORTD` and `QSORTI` leave the input values in place and return an
//! order vector. Rust's standard sort is used here instead of porting the old
//! iterative quicksort, while preserving the one-based order form expected by
//! translated FEFF call sites. The `PATH/sortix.f90` helpers are also ported
//! directly because their heap-sort tie ordering is used by path logic.

use thiserror::Error;

use crate::Real;

/// Error returned by FEFF sort helpers.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum SortError {
    /// Sorting finite FEFF grids with NaN or infinities is undefined.
    #[error("sort value at index {index} must be finite, got {value}")]
    NonFiniteValue { index: usize, value: Real },
    /// `QSORTI` compatibility requires values to fit into a 32-bit integer key.
    #[error("sort value at index {index} cannot be represented as i32 key: {value}")]
    IntegerKeyOutOfRange { index: usize, value: Real },
    /// `SORTIR` compares single-precision keys.
    #[error("sort value at index {index} cannot be represented as f32 key: {value}")]
    SinglePrecisionKeyOutOfRange { index: usize, value: Real },
}

/// Return a zero-based ascending companion order, equivalent to `QSORTD`.
pub fn sort_order(values: &[Real]) -> Result<Vec<usize>, SortError> {
    validate_finite(values)?;
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });
    Ok(order)
}

/// Return a one-based ascending companion order, matching FEFF `QSORTD`.
pub fn qsortd_order_1based(values: &[Real]) -> Result<Vec<usize>, SortError> {
    sort_order_1based(values)
}

/// Return a one-based ascending companion order for real values.
pub fn sort_order_1based(values: &[Real]) -> Result<Vec<usize>, SortError> {
    Ok(sort_order(values)?
        .into_iter()
        .map(|index| index + 1)
        .collect())
}

/// Return a zero-based `QSORTI`-compatible order.
///
/// FEFF's `QSORTI` stores comparison temporaries as integers even though the
/// array is double precision. This helper sorts by the truncated integer key,
/// then by the real value and original index for deterministic tie handling.
pub fn qsorti_compatible_order(values: &[Real]) -> Result<Vec<usize>, SortError> {
    let keys = integer_sort_keys(values)?;
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&left, &right| {
        keys[left]
            .cmp(&keys[right])
            .then_with(|| values[left].total_cmp(&values[right]))
            .then_with(|| left.cmp(&right))
    });
    Ok(order)
}

/// Return a one-based `QSORTI`-compatible companion order.
pub fn qsorti_order_1based(values: &[Real]) -> Result<Vec<usize>, SortError> {
    Ok(qsorti_compatible_order(values)?
        .into_iter()
        .map(|index| index + 1)
        .collect())
}

/// Port of FEFF `PATH/sortix.f90` `sortir` for single-precision real keys.
///
/// The returned order is one-based and preserves FEFF's heap-sort tie behavior.
pub fn sortir_order_1based(values: &[Real]) -> Result<Vec<usize>, SortError> {
    let keys = single_precision_sort_keys(values)?;
    Ok(feff_heap_index_order(keys.len(), |index| keys[index])
        .into_iter()
        .map(|index| index + 1)
        .collect())
}

/// Port of FEFF `PATH/sortix.f90` `sortii` for integer keys.
///
/// The returned order is one-based and preserves FEFF's heap-sort tie behavior.
pub fn sortii_order_1based(values: &[i32]) -> Vec<usize> {
    feff_heap_index_order(values.len(), |index| values[index])
        .into_iter()
        .map(|index| index + 1)
        .collect()
}

/// Port of FEFF `PATH/sortix.f90` `sortid` for double-precision real keys.
///
/// The returned order is one-based and preserves FEFF's heap-sort tie behavior.
pub fn sortid_order_1based(values: &[Real]) -> Result<Vec<usize>, SortError> {
    validate_finite(values)?;
    Ok(feff_heap_index_order(values.len(), |index| values[index])
        .into_iter()
        .map(|index| index + 1)
        .collect())
}

fn validate_finite(values: &[Real]) -> Result<(), SortError> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SortError::NonFiniteValue { index, value });
        }
    }
    Ok(())
}

fn integer_sort_keys(values: &[Real]) -> Result<Vec<i32>, SortError> {
    validate_finite(values)?;
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value < i32::MIN as Real || value > i32::MAX as Real {
                return Err(SortError::IntegerKeyOutOfRange { index, value });
            }
            Ok(value as i32)
        })
        .collect()
}

fn single_precision_sort_keys(values: &[Real]) -> Result<Vec<f32>, SortError> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(SortError::NonFiniteValue { index, value });
            }
            let key = value as f32;
            if !key.is_finite() {
                return Err(SortError::SinglePrecisionKeyOutOfRange { index, value });
            }
            Ok(key)
        })
        .collect()
}

fn feff_heap_index_order<T, F>(len: usize, key: F) -> Vec<usize>
where
    T: PartialOrd + Copy,
    F: Fn(usize) -> T,
{
    let mut order: Vec<_> = (0..len).collect();
    if len <= 1 {
        return order;
    }

    let mut left = len / 2 + 1;
    let mut right = len;
    loop {
        let stored = if left > 1 {
            left -= 1;
            order[left - 1]
        } else {
            let stored = order[right - 1];
            order[right - 1] = order[0];
            right -= 1;
            if right == 1 {
                order[0] = stored;
                return order;
            }
            stored
        };

        let mut child = left;
        loop {
            let parent = child;
            child *= 2;
            if child != right {
                if child > right {
                    order[parent - 1] = stored;
                    break;
                }
                if key(order[child - 1]) < key(order[child]) {
                    child += 1;
                }
            }
            if key(stored) >= key(order[child - 1]) {
                order[parent - 1] = stored;
                break;
            }
            order[parent - 1] = order[child - 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qsortd_order_matches_feff_reference() -> Result<(), SortError> {
        let values = [3.25, -1.5, 2.75, 2.0, -0.25, 7.0, -3.0, 4.5];

        assert_eq!(qsortd_order_1based(&values)?, vec![7, 2, 5, 4, 3, 1, 8, 6]);
        assert_eq!(sort_order(&values)?, vec![6, 1, 4, 3, 2, 0, 7, 5]);
        Ok(())
    }

    #[test]
    fn qsorti_order_matches_feff_integer_key_reference() -> Result<(), SortError> {
        let values = [1.9, 1.1, -1.9, -1.1, 0.4, -0.4, 2.2, 2.8];

        assert_eq!(qsorti_order_1based(&values)?, vec![3, 4, 6, 5, 2, 1, 7, 8]);
        Ok(())
    }

    #[test]
    fn sortix_orders_match_feff_reference() -> Result<(), SortError> {
        let real_values = [3.25, -1.5, 2.75, 2.0, -0.25, 7.0, -3.0, 4.5];
        assert_eq!(
            sortir_order_1based(&real_values)?,
            vec![7, 2, 5, 4, 3, 1, 8, 6]
        );

        let real_ties = [1.0, 1.0, -2.0, 3.0, -2.0, 0.0, 3.0, 1.0];
        assert_eq!(
            sortir_order_1based(&real_ties)?,
            vec![3, 5, 6, 8, 2, 1, 7, 4]
        );

        let int_ties = [3, -1, 2, 2, -1, 5, 0];
        assert_eq!(sortii_order_1based(&int_ties), vec![2, 5, 7, 3, 4, 1, 6]);

        let double_ties = [3.0, -1.0, 2.0, 2.0, -1.0, 5.0];
        assert_eq!(sortid_order_1based(&double_ties)?, vec![5, 2, 3, 4, 1, 6]);

        let double_precision = [
            1.000_000_000_000_001,
            1.000_000_000_000_002,
            0.999_999_999_999_999,
            -4.0,
            8.0,
            0.0,
        ];
        assert_eq!(
            sortid_order_1based(&double_precision)?,
            vec![4, 6, 3, 1, 2, 5]
        );
        Ok(())
    }

    #[test]
    fn rejects_non_finite_values() {
        assert!(matches!(
            sort_order(&[1.0, Real::NAN]),
            Err(SortError::NonFiniteValue { index: 1, .. })
        ));
        assert!(matches!(
            sortir_order_1based(&[1.0, Real::MAX]),
            Err(SortError::SinglePrecisionKeyOutOfRange { index: 1, .. })
        ));
        assert!(matches!(
            sortid_order_1based(&[1.0, Real::INFINITY]),
            Err(SortError::NonFiniteValue { index: 1, .. })
        ));
    }
}
