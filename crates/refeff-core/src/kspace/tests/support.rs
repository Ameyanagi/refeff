use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, arr2, array};

use super::*;
use crate::Complex;

pub(super) fn sample_sdef_operations() -> Array3<i32> {
    array![
        [[111, 112, 113], [121, 122, 123], [131, 132, 133]],
        [[211, 212, 213], [221, 222, 223], [231, 232, 233]]
    ]
}

pub(super) fn sample_sdefl_operations() -> Array3<i32> {
    array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ]
}

pub(super) fn sign_flip_symmetry_operations() -> Array3<i32> {
    array![
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
}

pub(super) fn skew_reciprocal_basis() -> RealMat {
    arr2(&[
        [
            3.138_666_777_779_998_4,
            -7.979_661_299_440_674e-2,
            -1.489_536_775_895_592_7e-1,
        ],
        [
            -3.404_655_487_761_354_4e-1,
            2.138_549_228_250_1,
            -1.968_316_453_862_033e-1,
        ],
        [
            1.994_915_324_860_168_6e-1,
            -2.713_084_841_809_829e-1,
            1.587_952_598_588_694,
        ],
    ])
}

pub(super) fn assert_vector_close(
    actual: Option<Vector3>,
    expected: Vector3,
) -> Result<(), KSpaceError> {
    let Some(actual) = actual else {
        return Err(KSpaceError::KPathDefinitionIncomplete {
            bravais: 0,
            kpath: 0,
            available: 0,
            required: 1,
        });
    };
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected);
    }
    Ok(())
}

pub(super) fn assert_vector_values_close(actual: Vector3, expected: Vector3) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected);
    }
}

pub(super) fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: ArrayView2<'_, Real>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_close(actual, expected[(row, column)]);
    }
}

pub(super) fn assert_matrix_i32_eq(actual: ArrayView2<'_, i32>, expected: ArrayView2<'_, i32>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_eq!(actual, expected[(row, column)]);
    }
}

pub(super) fn assert_array1_close(actual: ArrayView1<'_, Real>, expected: ArrayView1<'_, Real>) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(*actual, *expected);
    }
}

pub(super) fn assert_complex_array1_close(
    actual: ArrayView1<'_, Complex>,
    expected: ArrayView1<'_, Complex>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_complex_close(*actual, *expected);
    }
}

pub(super) fn assert_complex_matrix_close(
    actual: ArrayView2<'_, Complex>,
    expected: ArrayView2<'_, Complex>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_complex_close(actual, expected[(row, column)]);
    }
}

pub(super) fn assert_complex_cube_close(
    actual: ArrayView3<'_, Complex>,
    expected: ArrayView3<'_, Complex>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((first, second, third), &actual) in actual.indexed_iter() {
        assert_complex_close(actual, expected[(first, second, third)]);
    }
}

pub(super) fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_close(actual.re, expected.re);
    assert_close(actual.im, expected.im);
}

pub(super) fn assert_operation_close(
    actual: Option<[[Real; 3]; 3]>,
    expected: [[Real; 3]; 3],
) -> Result<(), KSpaceError> {
    let Some(actual) = actual else {
        return Err(KSpaceError::NoPointGroupOperations);
    };
    for row in 0..3 {
        for col in 0..3 {
            assert_close(actual[row][col], expected[row][col]);
        }
    }
    Ok(())
}

pub(super) fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:?}, got {actual:?}"
    );
}
