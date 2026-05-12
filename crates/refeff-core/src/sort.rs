//! Companion-index sorting helpers for FEFF arrays.
//!
//! FEFF `QSORTD` and `QSORTI` leave the input values in place and return an
//! order vector. Rust's standard sort is used here instead of porting the old
//! iterative quicksort, while preserving the one-based order form expected by
//! translated FEFF call sites.

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
    fn rejects_non_finite_values() {
        assert!(matches!(
            sort_order(&[1.0, Real::NAN]),
            Err(SortError::NonFiniteValue { index: 1, .. })
        ));
    }
}
