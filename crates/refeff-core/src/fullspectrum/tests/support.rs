use super::*;

pub(super) fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

pub(super) fn sample_edge_background(point_count: usize) -> FullSpectrumBackground {
    FullSpectrumBackground {
        scattering_factor: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(100.0 + row as Real, 10.0 + row as Real)
        }),
        effective_electron_count: 2.5,
        zero_energy_fprime: 1.0,
    }
}

pub(super) fn sample_edge_fine_structure(point_count: usize) -> FullSpectrumFineStructure {
    FullSpectrumFineStructure {
        scattering_factor: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(200.0 + row as Real, 20.0 + row as Real)
        }),
        background: Array1::from_shape_fn(point_count, |row| {
            Complex64::new(300.0 + row as Real, 30.0 + row as Real)
        }),
        real_energy_interval: [2.0, 6.0],
        imaginary_energy_interval: [2.0, 6.0],
        real_transition_interval: [2.0, 6.0],
        imaginary_transition_interval: [2.0, 6.0],
    }
}
