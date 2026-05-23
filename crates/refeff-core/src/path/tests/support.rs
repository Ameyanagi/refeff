use super::*;

pub(super) fn mrb_reference_positions() -> ndarray::Array2<Real> {
    arr2(&[
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
        [2.0, 1.0, 0.4],
        [-0.5, 1.7, 0.3],
        [0.7, -1.2, 0.8],
        [0.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
    ])
}

pub(super) fn assert_path_geometry_close(
    actual: &PathGeometry,
    expected_distances: &[Real],
    expected_cosines: &[Real],
) {
    assert_eq!(actual.leg_distances.len(), expected_distances.len());
    assert_eq!(actual.angle_cosines.len(), expected_cosines.len());

    for (&actual, &expected) in actual.leg_distances.iter().zip(expected_distances) {
        assert!(
            (actual - expected).abs() <= MRB_TOLERANCE,
            "leg distance {actual} != {expected}"
        );
    }
    for (&actual, &expected) in actual.angle_cosines.iter().zip(expected_cosines) {
        assert!(
            (actual - expected).abs() <= MRB_TOLERANCE,
            "angle cosine {actual} != {expected}"
        );
    }

    let expected_total = expected_distances
        .iter()
        .fold(0.0_f32, |sum, &distance| sum + distance as f32);
    assert!(
        (actual.total_path_length - Real::from(expected_total)).abs() <= MRB_TOLERANCE,
        "total path length {} != {}",
        actual.total_path_length,
        expected_total
    );
}

pub(super) fn assert_output_parameters_close(
    actual: &PathOutputParameters,
    expected_distances: &[Real],
    expected_angles: &[Real],
    expected_eta: &[Real],
) {
    assert_real_slice_close(
        &actual.leg_distances,
        expected_distances,
        "output leg distance",
        OUTPUT_PARAMETER_TOLERANCE,
    );
    assert_real_slice_close(
        &actual.scattering_angles,
        expected_angles,
        "output scattering angle",
        OUTPUT_PARAMETER_TOLERANCE,
    );
    assert_real_slice_close(
        &actual.eta_angles,
        expected_eta,
        "output eta angle",
        OUTPUT_PARAMETER_TOLERANCE,
    );
}

pub(super) fn assert_real_slice_close(
    actual: &[Real],
    expected: &[Real],
    label: &str,
    tolerance: Real,
) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} {actual} != {expected}"
        );
    }
}

pub(super) fn assert_hash_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() <= HASH_TOLERANCE,
        "path hash {actual} != {expected}"
    );
}

pub(super) fn assert_standard_coordinates(
    actual: PathStandardCoordinates,
    expected_case: u8,
    expected: &[[Real; 3]],
) {
    assert_eq!(actual.symmetry_case, expected_case);
    assert_eq!(actual.coordinates.nrows(), expected.len());
    assert_eq!(actual.coordinates.ncols(), 3);
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_value) in expected_row.iter().enumerate() {
            let actual_value = actual.coordinates[(row, column)];
            assert!(
                (actual_value - expected_value).abs() <= STANDARD_TOLERANCE,
                "standard coordinate ({row}, {column}) {actual_value} != {expected_value}"
            );
        }
    }
}

pub(super) fn assert_canonical_representation(
    actual: PathCanonicalRepresentation,
    expected_path: &[usize],
    expected_case: u8,
    expected_reversed: bool,
    expected_hash: Real,
    expected_coordinates: &[[Real; 3]],
) {
    assert_eq!(actual.path_indices, expected_path);
    assert_eq!(actual.reversed, expected_reversed);
    assert_hash_close(actual.degeneracy_hash, expected_hash);
    assert_standard_coordinates(
        PathStandardCoordinates {
            coordinates: actual.coordinates,
            symmetry_case: actual.symmetry_case,
        },
        expected_case,
        expected_coordinates,
    );
}

pub(super) fn reference_atom_potentials() -> Vec<usize> {
    (0..=8).map(|index| index % 4).collect()
}

pub(super) fn prcrit_reference_inputs()
-> (Vec<Complex>, Vec<Complex>, Array3<Complex>, Array2<usize>) {
    let energy_count = 43;
    let potential_count = 3;
    let angular_channels = 4;
    let energies = (0..energy_count)
        .map(|index| {
            let ie = (index + 1) as Real;
            Complex::new(0.02 * (ie - 2.0) + 0.001 * (ie - 1.0), 0.005 + 0.0003 * ie)
        })
        .collect::<Vec<_>>();
    let references = vec![Complex::new(-0.015, -0.002); energy_count];
    let phase_shifts = Array3::from_shape_fn(
        (energy_count, angular_channels, potential_count).f(),
        |(energy, angular, potential)| {
            let ie = (energy + 1) as Real;
            let il = (angular + 1) as Real;
            let iph = potential as Real;
            Complex::new(
                0.02 * ie + 0.11 * il + 0.03 * iph,
                0.004 * ie - 0.002 * il + 0.001 * iph,
            )
        },
    );
    let angular_limits = Array2::from_shape_fn(
        (energy_count, potential_count).f(),
        |(energy, potential)| (energy + 1 + potential) % angular_channels,
    );
    (energies, references, phase_shifts, angular_limits)
}

pub(super) fn reference_fbeta_table() -> Array3<Real> {
    Array3::from_shape_fn((81, 4, 3), |(beta_row, potential, criterion)| {
        let beta_index = beta_row as i32 - 40;
        Real::from(
            0.5_f32
                + 0.01_f32 * potential as f32
                + 0.002_f32 * (criterion + 1) as f32
                + 0.003_f32 * beta_index.abs() as f32
                + 0.0001_f32 * beta_index as f32,
        )
    })
}

pub(super) fn reference_fbeta_output_table() -> Array3<Real> {
    Array3::from_shape_fn((81, 4, 5), |(beta_row, potential, energy)| {
        let beta_index = beta_row as i32 - 40;
        Real::from(
            0.45_f32
                + 0.008_f32 * potential as f32
                + 0.015_f32 * (energy + 1) as f32
                + 0.0025_f32 * beta_index.abs() as f32
                + 0.0002_f32 * beta_index as f32,
        )
    })
}

pub(super) fn assert_option_close(actual: Option<Real>, expected: Option<Real>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_close(actual, expected),
        (actual, expected) => assert_eq!(actual, expected),
    }
}

pub(super) fn assert_phase_close(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "path phase criteria {actual} != {expected}"
    );
}

pub(super) fn assert_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() <= CRITERION_TOLERANCE,
        "path criterion {actual} != {expected}"
    );
}
