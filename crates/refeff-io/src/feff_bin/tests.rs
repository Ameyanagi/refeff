use super::*;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::error::{IoError, Result};

#[test]
fn writes_header_and_path_records_like_feff() -> Result<()> {
    let data = sample_feff_bin_data();
    let text = feff_bin_string(&data)?;
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("#_feff.bin v03: refeff-test"));
    assert_eq!(lines.next(), Some("#_    1    3    8"));
    assert!(text.lines().any(|line| line == "#@ Cu     O       29   8"));
    assert!(
        text.lines()
            .any(|line| line == "##    17   3   4.000   2.5000000        1.2500e1  0  1  0")
    );
    Ok(())
}

#[test]
fn roundtrips_feff_bin_text_with_pad_tolerance() -> Result<()> {
    let data = sample_feff_bin_data();
    let parsed = parse_feff_bin(&feff_bin_string(&data)?)?;
    assert_eq!(parsed.version, data.version);
    assert_eq!(parsed.pad_width, data.pad_width);
    assert_eq!(parsed.ihole, data.ihole);
    assert_eq!(parsed.order, data.order);
    assert_eq!(
        parsed.initial_angular_momentum,
        data.initial_angular_momentum
    );
    assert_eq!(parsed.potentials, data.potentials);
    assert_close_reals(
        [
            parsed.average_norman_radius,
            parsed.fermi_level,
            parsed.edge_energy,
        ],
        [
            data.average_norman_radius,
            data.fermi_level,
            data.edge_energy,
        ],
    );
    assert_close_complex(parsed.central_phase_shift, data.central_phase_shift);
    assert_close_complex(parsed.complex_momentum, data.complex_momentum);
    assert_close_reals(parsed.real_momentum, data.real_momentum);
    assert_eq!(parsed.paths.len(), data.paths.len());
    for (actual, expected) in parsed.paths.iter().zip(&data.paths) {
        assert_eq!(actual.index, expected.index);
        assert_close_reals(
            [
                actual.degeneracy,
                actual.effective_half_path_length_bohr,
                actual.criterion,
            ],
            [
                expected.degeneracy,
                expected.effective_half_path_length_bohr,
                expected.criterion,
            ],
        );
        assert_eq!(actual.potential_indices, expected.potential_indices);
        assert_close_reals(
            actual.positions.iter().copied(),
            expected.positions.iter().copied(),
        );
        assert_close_reals(actual.beta.iter().copied(), expected.beta.iter().copied());
        assert_close_reals(actual.eta.iter().copied(), expected.eta.iter().copied());
        assert_close_reals(
            actual.leg_distances.iter().copied(),
            expected.leg_distances.iter().copied(),
        );
        assert_close_reals(
            actual.amplitude.iter().copied(),
            expected.amplitude.iter().copied(),
        );
        assert_close_reals(actual.phase.iter().copied(), expected.phase.iter().copied());
    }
    Ok(())
}

#[test]
fn rejects_invalid_shapes_and_tokens() {
    let mut bad = sample_feff_bin_data();
    bad.real_momentum = Array1::from_vec(vec![1.0]);
    assert!(matches!(
        feff_bin_string(&bad),
        Err(IoError::FeffBinShape {
            field: "xk",
            actual,
            expected,
        }) if actual == vec![1] && expected == vec![3]
    ));

    assert!(matches!(
        parse_feff_bin("#_not-feff"),
        Err(IoError::InvalidFeffBin {
            field: "version",
            ..
        })
    ));
}

#[test]
fn preserves_matching_raw_text() -> Result<()> {
    let data = sample_feff_bin_data();
    let text = feff_bin_string(&data)?;
    let mut parsed = parse_feff_bin(&text)?;
    let raw_text = parsed
        .raw_text
        .as_mut()
        .ok_or(IoError::FeffBinMissing { field: "raw_text" })?;
    raw_text.push('\n');

    let mut expected = text.clone();
    expected.push('\n');
    assert_eq!(feff_bin_string(&parsed)?, expected);

    parsed.edge_energy += 1.0;
    assert_ne!(feff_bin_string(&parsed)?, expected);
    Ok(())
}

fn sample_feff_bin_data() -> FeffBinData {
    FeffBinData {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        ihole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 9.1,
        potentials: vec![
            FeffBinPotential {
                label: "Cu".to_string(),
                atomic_number: 29,
            },
            FeffBinPotential {
                label: "O".to_string(),
                atomic_number: 8,
            },
        ],
        central_phase_shift: Array1::from_vec(vec![
            Complex64::new(0.1, -0.01),
            Complex64::new(0.2, -0.02),
            Complex64::new(0.3, -0.03),
        ]),
        complex_momentum: Array1::from_vec(vec![
            Complex64::new(1.0, 0.1),
            Complex64::new(1.1, 0.2),
            Complex64::new(1.2, 0.3),
        ]),
        real_momentum: Array1::from_vec(vec![0.5, 0.6, 0.7]),
        paths: vec![FeffBinPath {
            index: 17,
            degeneracy: 4.0,
            effective_half_path_length_bohr: 2.5 / FEFF_BIN_BOHR,
            criterion: 12.5,
            potential_indices: Array1::from_vec(vec![0, 1, 0]),
            positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
                (0, 0..=2) => 0.0,
                (1, 0) => 1.0,
                (1, 1) => 0.5,
                (1, 2) => 0.0,
                (2, 0) => -1.0,
                (2, 1) => 0.25,
                (2, 2) => 0.0,
                _ => 0.0,
            }),
            beta: Array1::from_vec(vec![0.1, 0.2, 0.3]),
            eta: Array1::from_vec(vec![0.4, 0.5, 0.6]),
            leg_distances: Array1::from_vec(vec![1.0, 1.1, 1.2]),
            amplitude: Array1::from_vec(vec![2.0, 2.1, 2.2]),
            phase: Array1::from_vec(vec![-0.1, -0.2, -0.3]),
        }],
        raw_text: None,
    }
}

fn assert_close_reals(
    actual: impl IntoIterator<Item = f64>,
    expected: impl IntoIterator<Item = f64>,
) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-6,
            "{actual} != {expected}"
        );
    }
}

fn assert_close_complex(
    actual: impl IntoIterator<Item = Complex64>,
    expected: impl IntoIterator<Item = Complex64>,
) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual.re - expected.re).abs() <= expected.re.abs().max(1.0) * 1.0e-6,
            "{actual} != {expected}"
        );
        assert!(
            (actual.im - expected.im).abs() <= expected.im.abs().max(1.0) * 1.0e-6,
            "{actual} != {expected}"
        );
    }
}
