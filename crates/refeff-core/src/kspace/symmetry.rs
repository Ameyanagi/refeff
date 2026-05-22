use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use refeff_linalg::feff_inverse;

use super::support::*;
use super::*;

const POINT_GROUP_EPSILON: Real = 1.0e-8;

#[derive(Debug, Clone, Copy)]
struct PointGroupCandidate {
    vector: Vector3,
    norm: Real,
}

/// Port of FEFF `KSPACE/kmesh.f90` `sdef`.
///
/// FEFF rewrites symmetry-operation matrices for alternate centered-lattice
/// labels derived from CXY settings. `lattice` is interpreted as FEFF's
/// three-character `lattic` code, padded with spaces and uppercased for Rust
/// caller convenience.
pub fn redefine_lattice_symmetry_operations(
    operations: ArrayView3<'_, i32>,
    lattice: &str,
) -> Result<Array3<i32>, KSpaceError> {
    validate_symmetry_operation_shape(operations)?;

    let mut redefined = operations.to_owned();
    match lattice_code3(lattice) {
        [b'C', b'X', b'Z'] | [b'B', b'O', b' '] => {
            for operation in 0..redefined.shape()[0] {
                swap_operation_entries(&mut redefined, operation, (1, 1), (2, 2));
                swap_operation_entries(&mut redefined, operation, (0, 1), (0, 2));
                swap_operation_entries(&mut redefined, operation, (1, 2), (2, 1));
                swap_operation_entries(&mut redefined, operation, (1, 0), (2, 0));
            }
        }
        [b'C', b'Y', b'Z'] | [b'A', b'O', b' '] => {
            for operation in 0..redefined.shape()[0] {
                swap_operation_entries(&mut redefined, operation, (0, 0), (2, 2));
                swap_operation_entries(&mut redefined, operation, (0, 1), (2, 1));
                swap_operation_entries(&mut redefined, operation, (0, 2), (2, 0));
                swap_operation_entries(&mut redefined, operation, (1, 2), (1, 0));
            }
        }
        _ => {}
    }
    Ok(redefined)
}

/// Port of FEFF `KSPACE/kmesh.f90` `sdefl`.
///
/// FEFF transforms LAPW-style symmetry operations into the current lattice
/// basis as `rbas * operation * transpose(gbas) / (2*pi)` for orthogonal
/// lattices, and also for the special non-orthogonal `CXZ` setting. Other
/// non-orthogonal settings keep the integer operation table unchanged.
pub fn transform_lapw_symmetry_operations(
    direct_vectors: ArrayView2<'_, Real>,
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    lattice: &str,
    orthogonal: bool,
) -> Result<Array3<i32>, KSpaceError> {
    validate_matrix("direct_vectors", direct_vectors)?;
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_symmetry_operation_shape(operations)?;

    if !orthogonal && lattice_code3(lattice) != [b'C', b'X', b'Z'] {
        return Ok(operations.to_owned());
    }

    let mut transformed = Array3::<i32>::zeros(operations.raw_dim());
    for operation in 0..operations.shape()[0] {
        for row in 0..3 {
            for column in 0..3 {
                let value = transformed_symmetry_entry(
                    direct_vectors,
                    reciprocal_vectors,
                    operations,
                    operation,
                    row,
                    column,
                );
                transformed[(operation, row, column)] =
                    round_symmetry_operation_entry(value, operation, row, column)?;
            }
        }
    }
    Ok(transformed)
}

/// Build FEFF's reciprocal-space metric `b(i,j) = dot(b_i, b_j)`.
///
/// `reciprocal_vectors` stores the reciprocal basis vectors as columns,
/// matching `KSPACE/pointgroup.f90` inputs.
pub fn reciprocal_metric(reciprocal_vectors: ArrayView2<'_, Real>) -> Result<RealMat, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    let mut metric = Array2::<Real>::zeros((3, 3));
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                metric[(row, col)] +=
                    reciprocal_vectors[(axis, row)] * reciprocal_vectors[(axis, col)];
            }
        }
    }
    Ok(metric)
}

/// Port of FEFF `KSPACE/pointgroup.f90`.
///
/// `reciprocal_vectors` stores reciprocal basis vectors as columns. `metric`
/// should normally come from [`reciprocal_metric`]. `max_operations` is FEFF's
/// `ntot` capacity; if the lattice produces more operations, a typed error is
/// returned instead of stopping the process.
pub fn point_group_operations(
    reciprocal_vectors: ArrayView2<'_, Real>,
    metric: ArrayView2<'_, Real>,
    max_operations: usize,
) -> Result<PointGroup, KSpaceError> {
    validate_matrix("reciprocal_vectors", reciprocal_vectors)?;
    validate_matrix("metric", metric)?;
    if max_operations == 0 {
        return Err(KSpaceError::InvalidPointGroupCapacity {
            capacity: max_operations,
        });
    }

    let transposed = Array2::from_shape_fn((3, 3), |(row, col)| reciprocal_vectors[(col, row)]);
    let inverse_transposed = feff_inverse(transposed.view())?;
    let cutoffs = point_group_cutoffs(metric)?;
    let candidates = point_group_candidate_vectors(transposed.view(), cutoffs)?;
    let mut operations = Vec::<[[Real; 3]; 3]>::new();

    for first in 0..candidates.len() {
        if (candidates[first].norm - metric[(0, 0)]).abs() > POINT_GROUP_EPSILON {
            continue;
        }
        for second in 0..candidates.len() {
            if second == first
                || (candidates[second].norm - metric[(1, 1)]).abs() > POINT_GROUP_EPSILON
            {
                continue;
            }
            let first_second_dot = dot(candidates[first].vector, candidates[second].vector);
            if (first_second_dot - metric[(1, 0)]).abs() > POINT_GROUP_EPSILON {
                continue;
            }
            for third in 0..candidates.len() {
                if third == first
                    || third == second
                    || (candidates[third].norm - metric[(2, 2)]).abs() > POINT_GROUP_EPSILON
                {
                    continue;
                }
                let first_third_dot = dot(candidates[first].vector, candidates[third].vector);
                let second_third_dot = dot(candidates[second].vector, candidates[third].vector);
                if (first_third_dot - metric[(0, 2)]).abs() <= POINT_GROUP_EPSILON
                    && (second_third_dot - metric[(1, 2)]).abs() <= POINT_GROUP_EPSILON
                {
                    if operations.len() >= max_operations {
                        return Err(KSpaceError::TooManyPointGroupOperations {
                            capacity: max_operations,
                        });
                    }
                    operations.push(point_group_operation(
                        inverse_transposed.view(),
                        candidates[first].vector,
                        candidates[second].vector,
                        candidates[third].vector,
                    ));
                }
            }
        }
    }

    if operations.is_empty() {
        return Err(KSpaceError::NoPointGroupOperations);
    }

    Ok(PointGroup {
        operations: Array3::from_shape_fn((operations.len(), 3, 3), |(op, row, col)| {
            operations[op][row][col]
        }),
    })
}

/// Port of FEFF `KSPACE/symmetrycheck.f90`.
///
/// `operations` stores integer rotation matrices as `(operation, row, column)`.
/// `translations` stores FEFF translation vectors as `(operation, xyz)` in
/// radians. The returned multiplication table intentionally keeps FEFF's
/// positive one-based operation indices and `-1` translation-failure sentinel
/// so callers can compare it directly with legacy outputs.
pub fn symmetry_check(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
) -> Result<SymmetryCheck, KSpaceError> {
    validate_symmetry_inputs(operations, translations)?;

    let operation_count = operations.shape()[0];
    if operation_count > i32::MAX as usize {
        return Err(KSpaceError::SymmetryOperationCountOverflow {
            count: operation_count,
        });
    }
    if operation_count == 1 {
        return Ok(SymmetryCheck {
            multiplication: Array2::<i32>::zeros((1, 1)),
            ierr: 0,
        });
    }

    let mut multiplication = Array2::<i32>::zeros((operation_count, operation_count));
    for left in 0..operation_count {
        for right in 0..operation_count {
            let product_index = symmetry_product_index(operations, left, right)?.ok_or(
                KSpaceError::SymmetryProductMissing {
                    left: left + 1,
                    right: right + 1,
                },
            )?;
            multiplication[(left, right)] = product_index;
        }
    }

    mark_invalid_symmetry_translations(operations, translations, &mut multiplication);
    let ierr = symmetry_error_index(multiplication.view());
    Ok(SymmetryCheck {
        multiplication,
        ierr,
    })
}

fn point_group_cutoffs(metric: ArrayView2<'_, Real>) -> Result<[i32; 3], KSpaceError> {
    let mut max_metric = 0.0;
    let mut diagonal = [0.0; 3];
    for index in 0..3 {
        let value = metric[(index, index)];
        if value <= POINT_GROUP_EPSILON {
            return Err(KSpaceError::DegenerateMetricDiagonal { index, value });
        }
        if value > max_metric {
            max_metric = value;
        }
        diagonal[index] = 1.0 / value;
    }

    let bhelp = metric[(0, 1)] * metric[(0, 2)] * metric[(1, 2)];
    let first = metric[(0, 0)]
        - metric[(0, 1)].powi(2) * diagonal[1]
        - metric[(2, 0)].powi(2) * diagonal[2]
        + 2.0 * bhelp * diagonal[1] * diagonal[2];
    let second = metric[(1, 1)]
        - metric[(0, 1)].powi(2) * diagonal[0]
        - metric[(2, 1)].powi(2) * diagonal[2]
        + 2.0 * bhelp * diagonal[0] * diagonal[2];
    let third = metric[(2, 2)]
        - metric[(0, 2)].powi(2) * diagonal[0]
        - metric[(2, 1)].powi(2) * diagonal[1]
        + 2.0 * bhelp * diagonal[0] * diagonal[1];

    Ok([
        point_group_cutoff(0, max_metric, first)?,
        point_group_cutoff(1, max_metric, second)?,
        point_group_cutoff(2, max_metric, third)?,
    ])
}

fn point_group_cutoff(index: usize, max_metric: Real, value: Real) -> Result<i32, KSpaceError> {
    if value <= POINT_GROUP_EPSILON {
        return Err(KSpaceError::DegenerateMetricDenominator { index, value });
    }
    let cutoff = (max_metric / value).sqrt() + 1.0;
    if cutoff > Real::from(i32::MAX) {
        return Err(KSpaceError::PointGroupSearchOverflow);
    }
    Ok(cutoff as i32)
}

fn point_group_candidate_vectors(
    transposed_basis: ArrayView2<'_, Real>,
    cutoffs: [i32; 3],
) -> Result<Vec<PointGroupCandidate>, KSpaceError> {
    let first_count = point_group_axis_count(cutoffs[0])?;
    let second_count = point_group_axis_count(cutoffs[1])?;
    let third_count = point_group_axis_count(cutoffs[2])?;
    let capacity = first_count
        .checked_mul(second_count)
        .and_then(|value| value.checked_mul(third_count))
        .ok_or(KSpaceError::PointGroupSearchOverflow)?;
    let mut candidates = Vec::with_capacity(capacity);

    for i in -cutoffs[0]..=cutoffs[0] {
        let first = [
            Real::from(i) * transposed_basis[(0, 0)],
            Real::from(i) * transposed_basis[(1, 0)],
            Real::from(i) * transposed_basis[(2, 0)],
        ];
        for j in -cutoffs[1]..=cutoffs[1] {
            let second = [
                first[0] + Real::from(j) * transposed_basis[(0, 1)],
                first[1] + Real::from(j) * transposed_basis[(1, 1)],
                first[2] + Real::from(j) * transposed_basis[(2, 1)],
            ];
            for k in -cutoffs[2]..=cutoffs[2] {
                let vector = [
                    second[0] + Real::from(k) * transposed_basis[(0, 2)],
                    second[1] + Real::from(k) * transposed_basis[(1, 2)],
                    second[2] + Real::from(k) * transposed_basis[(2, 2)],
                ];
                candidates.push(PointGroupCandidate {
                    vector,
                    norm: dot(vector, vector),
                });
            }
        }
    }
    Ok(candidates)
}

fn point_group_axis_count(cutoff: i32) -> Result<usize, KSpaceError> {
    let cutoff = usize::try_from(cutoff).map_err(|_| KSpaceError::PointGroupSearchOverflow)?;
    cutoff
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(KSpaceError::PointGroupSearchOverflow)
}

fn point_group_operation(
    inverse_transposed: ArrayView2<'_, Real>,
    first: Vector3,
    second: Vector3,
    third: Vector3,
) -> [[Real; 3]; 3] {
    let columns = [first, second, third];
    let mut operation = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for axis in 0..3 {
                operation[row][col] += columns[axis][row] * inverse_transposed[(axis, col)];
            }
        }
    }
    operation
}

fn symmetry_product_index(
    operations: ArrayView3<'_, i32>,
    left: usize,
    right: usize,
) -> Result<Option<i32>, KSpaceError> {
    for candidate in 0..operations.shape()[0] {
        if symmetry_product_matches(operations, left, right, candidate) {
            let index = i32::try_from(candidate + 1).map_err(|_| {
                KSpaceError::SymmetryOperationCountOverflow {
                    count: operations.shape()[0],
                }
            })?;
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn symmetry_product_matches(
    operations: ArrayView3<'_, i32>,
    left: usize,
    right: usize,
    candidate: usize,
) -> bool {
    (0..3).all(|row| {
        (0..3).all(|col| {
            let product = (0..3)
                .map(|axis| {
                    i128::from(operations[(left, row, axis)])
                        * i128::from(operations[(right, axis, col)])
                })
                .sum::<i128>();
            product == i128::from(operations[(candidate, row, col)])
        })
    })
}

fn mark_invalid_symmetry_translations(
    operations: ArrayView3<'_, i32>,
    translations: ArrayView2<'_, Real>,
    multiplication: &mut Array2<i32>,
) {
    let operation_count = operations.shape()[0];
    for left in 0..operation_count {
        for right in 0..operation_count {
            let product = multiplication[(left, right)];
            if product <= 0 {
                continue;
            }
            let product_index = product as usize - 1;
            for axis in 0..3 {
                let mut ttest = translations[(right, axis)];
                for component in 0..3 {
                    ttest += Real::from(operations[(left, component, axis)])
                        * (translations[(left, component)]
                            - translations[(product_index, component)]);
                }
                let normalized = ttest.abs() / PI2;
                let integer_part = (normalized * 1.001) as i32;
                if (normalized - Real::from(integer_part)).abs() >= 0.0001 {
                    multiplication[(left, right)] = -1;
                    break;
                }
            }
        }
    }
}

fn symmetry_error_index(multiplication: ArrayView2<'_, i32>) -> usize {
    let operation_count = multiplication.nrows();
    let mut errors = vec![0usize; operation_count];
    for left in 0..operation_count {
        for right in 0..operation_count {
            if multiplication[(left, right)] <= 0 {
                errors[left] += 1;
                errors[right] += 1;
            }
        }
    }

    let mut ierr = 0usize;
    let mut max_errors = 0usize;
    for (index, count) in errors.into_iter().enumerate() {
        if count > max_errors {
            max_errors = count;
            ierr = index + 1;
        }
    }
    ierr
}

fn swap_operation_entries(
    operations: &mut Array3<i32>,
    operation: usize,
    first: (usize, usize),
    second: (usize, usize),
) {
    let first_value = operations[(operation, first.0, first.1)];
    operations[(operation, first.0, first.1)] = operations[(operation, second.0, second.1)];
    operations[(operation, second.0, second.1)] = first_value;
}

fn transformed_symmetry_entry(
    direct_vectors: ArrayView2<'_, Real>,
    reciprocal_vectors: ArrayView2<'_, Real>,
    operations: ArrayView3<'_, i32>,
    operation: usize,
    row: usize,
    column: usize,
) -> Real {
    let mut value = 0.0;
    for left in 0..3 {
        for right in 0..3 {
            value += direct_vectors[(row, left)]
                * Real::from(operations[(operation, left, right)])
                * reciprocal_vectors[(column, right)]
                / PI2;
        }
    }
    value
}

fn round_symmetry_operation_entry(
    value: Real,
    operation: usize,
    row: usize,
    column: usize,
) -> Result<i32, KSpaceError> {
    let rounded = value.round();
    if rounded.is_finite() && rounded >= i32::MIN as Real && rounded <= i32::MAX as Real {
        Ok(rounded as i32)
    } else {
        Err(KSpaceError::SymmetryOperationValueOverflow {
            operation: operation + 1,
            row: row + 1,
            column: column + 1,
            value,
        })
    }
}

fn dot(left: Vector3, right: Vector3) -> Real {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
