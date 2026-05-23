use super::*;

pub(super) fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-18,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

pub(super) fn assert_slice_close(actual: &[Real], expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_dmdw_close(*actual, *expected);
    }
}

pub(super) fn assert_matrix_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
    assert_eq!(actual.shape(), &[3, 3]);
    for row in 0..3 {
        for column in 0..3 {
            assert_dmdw_close(actual[(row, column)], expected[row][column]);
        }
    }
}

pub(super) fn assert_matrix_abs_close(actual: ArrayView2<'_, Real>, expected: &[[Real; 3]; 3]) {
    assert_eq!(actual.shape(), &[3, 3]);
    for row in 0..3 {
        for column in 0..3 {
            assert_dmdw_close(actual[(row, column)].abs(), expected[row][column].abs());
        }
    }
}

pub(super) fn sample_dmdw_type2_blocks() -> (ndarray::Array4<Real>, ndarray::Array1<Real>) {
    let masses = ndarray::arr1(&[63.546, 63.546, 63.546]);
    let mut force_blocks = ndarray::Array4::zeros((3, 3, 3, 3));
    for atom in 0..3 {
        for component in 0..3 {
            force_blocks[(atom, atom, component, component)] =
                0.02 + 0.003 * atom as Real + 0.001 * component as Real;
        }
    }
    for component in 0..3 {
        force_blocks[(0, 1, component, component)] = -0.004;
        force_blocks[(1, 0, component, component)] = -0.004;
        force_blocks[(1, 2, component, component)] = -0.003;
        force_blocks[(2, 1, component, component)] = -0.003;
    }
    (force_blocks, masses)
}

pub(super) fn sample_dmdw_type2_coupling() -> DmdwPhononCoupling {
    DmdwPhononCoupling {
        energy_hartree: ndarray::arr1(&[0.001, 0.002, 0.004]),
        energy_ev: ndarray::arr1(&[
            0.001 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.002 * DMDW_COUPLING_ENERGY_HARTREE_EV,
            0.004 * DMDW_COUPLING_ENERGY_HARTREE_EV,
        ]),
        eliashberg: ndarray::arr1(&[0.5, 1.0, 1.5]),
        matrix_element: ndarray::arr1(&[0.05, 0.05, 0.05]),
        normalization: 1.0,
    }
}

pub(super) fn assert_vector_close(actual: &Array1<Real>, expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_dmdw_close(*actual, *expected);
    }
}

pub(super) fn assert_complex_dmdw_close(actual: Complex, expected: Complex) {
    assert_dmdw_close(actual.re, expected.re);
    assert_dmdw_close(actual.im, expected.im);
}

pub(super) fn assert_complex_dmdw_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "actual={} expected={} diff={}",
        actual.re,
        expected.re,
        (actual.re - expected.re).abs()
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "actual={} expected={} diff={}",
        actual.im,
        expected.im,
        (actual.im - expected.im).abs()
    );
}

pub(super) fn column_dot(matrix: ArrayView2<'_, Real>, left: usize, right: usize) -> Real {
    let left_column = matrix.column(left);
    let right_column = matrix.column(right);
    left_column
        .iter()
        .zip(right_column.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

pub(super) fn assert_moment_summary(
    actual: &DmdwMomentSummary,
    expected_moment: Real,
    expected_frequency: Real,
    reduced_mass: Real,
) -> Result<(), DebyeError> {
    assert_dmdw_close(actual.moment_thz_power_n, expected_moment);
    let expected = dmdw_single_pole_einstein_summary(expected_frequency, reduced_mass)?;
    assert_dmdw_close(
        actual.frequency_thz.ok_or(DebyeError::NonFiniteOutput {
            name: "test moment frequency",
            value: Real::NAN,
        })?,
        expected.frequency_thz,
    );
    assert_dmdw_close(
        actual
            .temperature_kelvin
            .ok_or(DebyeError::NonFiniteOutput {
                name: "test moment temperature",
                value: Real::NAN,
            })?,
        expected.temperature_kelvin,
    );
    assert_dmdw_close(
        actual
            .effective_force_constant_n_per_m
            .ok_or(DebyeError::NonFiniteOutput {
                name: "test moment force constant",
                value: Real::NAN,
            })?,
        expected.effective_force_constant_n_per_m,
    );
    Ok(())
}

pub(super) fn assert_dmdw_close(actual: Real, expected: Real) {
    let tolerance = expected.abs().max(1.0) * 1.0e-14;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}

pub(super) fn assert_relative_close(actual: Real, expected: Real) {
    let tolerance = expected.abs().max(1.0) * 1.0e-14;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} diff={}",
        (actual - expected).abs()
    );
}
