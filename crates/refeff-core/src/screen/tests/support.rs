use super::*;

pub(super) fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

pub(super) fn assert_complex_close(
    actual: crate::Complex,
    expected_re: f64,
    expected_im: f64,
    tolerance: f64,
) {
    assert_close(actual.re, expected_re, tolerance);
    assert_close(actual.im, expected_im, tolerance);
}

pub(super) fn assert_complex32_close(
    actual: Complex32,
    expected_re: f64,
    expected_im: f64,
    tolerance: f64,
) {
    assert_close(actual.re as f64, expected_re, tolerance);
    assert_close(actual.im as f64, expected_im, tolerance);
}

pub(super) fn assert_array_close(actual: &RealVec, expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual_value, &expected_value) in actual.iter().zip(expected) {
        assert_close(actual_value, expected_value, tolerance);
    }
}
