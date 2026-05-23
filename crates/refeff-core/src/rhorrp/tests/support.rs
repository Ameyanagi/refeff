use super::*;

pub(super) fn reference_grid_input<'a>(axes: &'a Array2<Real>) -> RhorrpDensityGridInput<'a> {
    RhorrpDensityGridInput {
        origin: [0.1, -0.2, 0.3],
        axes: axes.view(),
        points_per_axis: &[3, 2, 4],
    }
}

pub(super) fn reference_axes() -> Array2<Real> {
    arr2(&[[1.2, -0.3, 0.4], [-0.4, 0.9, 0.1], [0.2, 0.5, 1.1]])
}

pub(super) fn reference_positions() -> Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

pub(super) fn reference_nearest_points() -> Array2<Real> {
    arr2(&[
        [0.7, 0.0, 0.2, 0.0],
        [0.2, 0.1, 0.9, 0.5],
        [0.1, 0.8, 0.1, 0.5],
    ])
}

pub(super) fn reference_inclusion_positions() -> Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 1.4],
        [1.5, 1.5, 0.0],
        [-0.5, 0.2, 0.3],
    ])
}

pub(super) fn sample_density(point: Vector3) -> Real {
    point[0] + 2.0 * point[1] - 0.5 * point[2] + point[0] * point[1]
}

pub(super) fn reference_wavefunctions() -> Array3<Complex> {
    Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
        let ie = (energy + 1) as Real;
        let il = angular as Real;
        let ir = (radial + 1) as Real;
        Complex::new(10.0 * ir + il + 0.1 * ie, -5.0 * ir + 0.25 * il - 0.2 * ie)
    })
}

pub(super) struct ReferenceSameSiteWavefunctions {
    pub(super) regular_large: Array3<Complex>,
    pub(super) irregular_large: Array3<Complex>,
    pub(super) regular_small: Array3<Complex>,
    pub(super) irregular_small: Array3<Complex>,
}

pub(super) fn reference_same_site_wavefunctions() -> ReferenceSameSiteWavefunctions {
    ReferenceSameSiteWavefunctions {
        regular_large: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.10 * ie + 0.03 * il + 0.01 * ir,
                -0.06 * ie + 0.02 * il - 0.015 * ir,
            )
        }),
        irregular_large: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.08 * ie + 0.04 * il + 0.025 * ir,
                0.05 * ie - 0.01 * il + 0.02 * ir,
            )
        }),
        regular_small: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.07 * ie - 0.02 * il + 0.018 * ir,
                0.04 * ie + 0.015 * il - 0.012 * ir,
            )
        }),
        irregular_small: Array3::from_shape_fn((3, 3, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.03 * ie + 0.025 * il - 0.02 * ir,
                0.02 * ie + 0.018 * il + 0.017 * ir,
            )
        }),
    }
}

pub(super) struct ReferencePairEnergyTables {
    pub(super) first_regular_large: Array3<Complex>,
    pub(super) first_irregular_large: Array3<Complex>,
    pub(super) first_regular_small: Array3<Complex>,
    pub(super) first_irregular_small: Array3<Complex>,
    pub(super) second_regular_large: Array3<Complex>,
    pub(super) second_regular_small: Array3<Complex>,
    pub(super) first_phase: Array2<Complex>,
    pub(super) second_phase: Array2<Complex>,
    pub(super) scattering_matrix: Array3<Complex>,
}

pub(super) fn reference_pair_energies() -> Array1<Complex> {
    Array1::from_vec(vec![
        Complex::new(0.2, 0.05),
        Complex::new(-0.1, 0.0),
        Complex::new(1.5, -0.2),
    ])
}

pub(super) fn reference_pair_energy_tables() -> ReferencePairEnergyTables {
    reference_pair_energy_tables_with_energy_count(3)
}

pub(super) fn reference_pair_energy_tables_with_energy_count(
    energy_count: usize,
) -> ReferencePairEnergyTables {
    ReferencePairEnergyTables {
        first_regular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.10 * ie + 0.03 * il + 0.01 * ir,
                    -0.06 * ie + 0.02 * il - 0.015 * ir,
                )
            },
        ),
        first_irregular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.08 * ie + 0.04 * il + 0.025 * ir,
                    0.05 * ie - 0.01 * il + 0.02 * ir,
                )
            },
        ),
        first_regular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.07 * ie - 0.02 * il + 0.018 * ir,
                    0.04 * ie + 0.015 * il - 0.012 * ir,
                )
            },
        ),
        first_irregular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.03 * ie + 0.025 * il - 0.02 * ir,
                    0.02 * ie + 0.018 * il + 0.017 * ir,
                )
            },
        ),
        second_regular_large: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    -0.05 * ie + 0.02 * il + 0.014 * ir,
                    0.03 * ie - 0.012 * il + 0.011 * ir,
                )
            },
        ),
        second_regular_small: Array3::from_shape_fn(
            (energy_count, 2, 6),
            |(energy, angular, radial)| {
                let ie = (energy + 1) as Real;
                let il = angular as Real;
                let ir = (radial + 1) as Real;
                Complex::new(
                    0.045 * ie + 0.018 * il - 0.009 * ir,
                    -0.025 * ie + 0.013 * il + 0.016 * ir,
                )
            },
        ),
        first_phase: Array2::from_shape_fn((energy_count, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(0.015 * ie + 0.04 * il, -0.006 * ie + 0.02 * il)
        }),
        second_phase: Array2::from_shape_fn((energy_count, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(-0.011 * ie + 0.03 * il, 0.007 * ie - 0.015 * il)
        }),
        scattering_matrix: Array3::from_shape_fn((energy_count, 4, 4), |(energy, row, column)| {
            let ie = (energy + 1) as Real;
            let row = (row + 1) as Real;
            let column = (column + 1) as Real;
            Complex::new(
                0.002 * ie + 0.004 * row - 0.003 * column,
                -0.0015 * ie + 0.0025 * row + 0.001 * column,
            )
        }),
    }
}

pub(super) struct ReferenceScatteringGreenTables {
    pub(super) first_regular_large: Array3<Complex>,
    pub(super) first_regular_small: Array3<Complex>,
    pub(super) second_regular_large: Array3<Complex>,
    pub(super) second_regular_small: Array3<Complex>,
    pub(super) first_phase: Array2<Complex>,
    pub(super) second_phase: Array2<Complex>,
    pub(super) scattering_matrix: Array3<Complex>,
}

pub(super) fn reference_scattering_green_tables() -> ReferenceScatteringGreenTables {
    ReferenceScatteringGreenTables {
        first_regular_large: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.10 * ie + 0.03 * il + 0.01 * ir,
                -0.06 * ie + 0.02 * il - 0.015 * ir,
            )
        }),
        first_regular_small: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.07 * ie - 0.02 * il + 0.018 * ir,
                0.04 * ie + 0.015 * il - 0.012 * ir,
            )
        }),
        second_regular_large: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                -0.05 * ie + 0.02 * il + 0.014 * ir,
                0.03 * ie - 0.012 * il + 0.011 * ir,
            )
        }),
        second_regular_small: Array3::from_shape_fn((3, 2, 4), |(energy, angular, radial)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            let ir = (radial + 1) as Real;
            Complex::new(
                0.045 * ie + 0.018 * il - 0.009 * ir,
                -0.025 * ie + 0.013 * il + 0.016 * ir,
            )
        }),
        first_phase: Array2::from_shape_fn((3, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(0.015 * ie + 0.04 * il, -0.006 * ie + 0.02 * il)
        }),
        second_phase: Array2::from_shape_fn((3, 2), |(energy, angular)| {
            let ie = (energy + 1) as Real;
            let il = angular as Real;
            Complex::new(-0.011 * ie + 0.03 * il, 0.007 * ie - 0.015 * il)
        }),
        scattering_matrix: Array3::from_shape_fn((3, 4, 4), |(energy, row, column)| {
            let ie = (energy + 1) as Real;
            let row = (row + 1) as Real;
            let column = (column + 1) as Real;
            Complex::new(
                0.002 * ie + 0.004 * row - 0.003 * column,
                -0.0015 * ie + 0.0025 * row + 0.001 * column,
            )
        }),
    }
}

pub(super) fn reference_irregular_solution() -> (Vec<Real>, ComplexVec) {
    let radii = (1..=120)
        .map(|index| {
            let index = index as Real;
            0.02 * index + 0.0001 * index * index
        })
        .collect::<Vec<_>>();
    let values = ComplexVec::from_shape_fn(120, |index| {
        let one_based = (index + 1) as Real;
        Complex::new(
            (0.07 * one_based).sin() + 0.002 * one_based,
            (0.05 * one_based).cos() - 0.001 * one_based,
        )
    });
    (radii, values)
}

pub(super) struct ReferenceAtomicDensityTables {
    pub(super) radii: Vec<Real>,
    pub(super) positions: Array2<Real>,
    pub(super) potentials: [usize; 4],
    pub(super) large: Array3<Real>,
    pub(super) small: Array3<Real>,
}

pub(super) fn reference_atomic_density_tables() -> ReferenceAtomicDensityTables {
    let radii = (1..=12)
        .map(|index| 0.015 + 0.035 * index as Real + 0.001 * (index as Real - 1.0).powi(2))
        .collect::<Vec<_>>();
    let positions = arr2(&[
        [0.0, 0.0, 0.0],
        [0.7, -0.2, 0.15],
        [-0.5, 0.55, -0.25],
        [1.85, 0.2, -0.1],
    ]);
    let potentials = [0, 1, 2, 1];
    let large = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
        let index = (radial + 1) as Real;
        let orbital = (orbital + 1) as Real;
        (0.13 * index).sin() + 0.031 * orbital + 0.047 * potential as Real + 0.12 * radii[radial]
    });
    let small = Array3::from_shape_fn((12, 3, 3), |(radial, orbital, potential)| {
        let index = (radial + 1) as Real;
        let orbital = (orbital + 1) as Real;
        (0.09 * index).cos() - 0.019 * orbital + 0.023 * potential as Real - 0.08 * radii[radial]
    });
    ReferenceAtomicDensityTables {
        radii,
        positions,
        potentials,
        large,
        small,
    }
}

pub(super) fn reference_density_integration_inputs() -> (Array1<Complex>, Array1<Complex>) {
    let energies = Array1::from_vec(vec![
        Complex::new(-0.030, 0.070),
        Complex::new(-0.030, 0.035),
        Complex::new(-0.030, 0.000),
        Complex::new(0.010, 0.000),
        Complex::new(0.065, 0.000),
        Complex::new(0.130, 0.000),
        Complex::new(0.045, 0.021_991_148_575_128_55),
        Complex::new(0.045, 0.043_982_297_150_257_1),
    ]);
    let energy_density = Array1::from_shape_fn(8, |index| {
        let energy = energies[index];
        let one_based = (index + 1) as Real;
        Complex::new(
            0.40 + 0.07 * one_based + 0.02 * energy.re - 0.15 * energy.im,
            -0.25 + 0.04 * one_based + 0.18 * energy.re + 0.03 * energy.im,
        )
    });
    (energies, energy_density)
}

pub(super) fn column(points: &RealMat, index: usize) -> Vector3 {
    [points[(0, index)], points[(1, index)], points[(2, index)]]
}

pub(super) fn row(points: &RealMat, index: usize) -> Vector3 {
    [points[(index, 0)], points[(index, 1)], points[(index, 2)]]
}

pub(super) fn assert_vector_close(actual: Vector3, expected: Vector3) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
            (actual - expected).abs()
        );
    }
}

pub(super) fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_complex_close_tol(actual, expected, 1.0e-12);
}

pub(super) fn assert_complex_close_tol(actual: Complex, expected: Complex, tolerance: Real) {
    assert!(
        (actual.re - expected.re).abs() < tolerance,
        "real actual={:.17e}, expected={:.17e}, diff={:.17e}",
        actual.re,
        expected.re,
        (actual.re - expected.re).abs()
    );
    assert!(
        (actual.im - expected.im).abs() < tolerance,
        "imag actual={:.17e}, expected={:.17e}, diff={:.17e}",
        actual.im,
        expected.im,
        (actual.im - expected.im).abs()
    );
}

pub(super) fn assert_real_close_scaled(actual: Real, expected: Real) {
    let tolerance = 1.0e-11 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}, tolerance={tolerance:.17e}",
        (actual - expected).abs()
    );
}

pub(super) fn assert_real_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual:.17e}, expected={expected:.17e}, diff={:.17e}",
        (actual - expected).abs()
    );
}
