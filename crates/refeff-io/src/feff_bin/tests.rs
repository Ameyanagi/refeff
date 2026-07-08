use super::*;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use refeff_core::{
    GenfmtFeffBinHeader, GenfmtFeffBinPotential, GenfmtJasDriverOutput, GenfmtJasPathOutputs,
    GenfmtJasPathSequence, GenfmtOrdinaryDriverOutput, GenfmtOrdinaryPathOutputs,
    GenfmtOrdinaryPathSequence, GenfmtRetainedPathOutput,
};

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
fn builds_path_from_genfmt_retained_path_output() {
    let output = sample_genfmt_retained_path_output();

    let path = FeffBinPath::from(&output);
    assert_eq!(path.index, 17);
    assert_eq!(path.degeneracy, output.degeneracy);
    assert_eq!(
        path.effective_half_path_length_bohr,
        output.effective_half_path_length_bohr
    );
    assert_eq!(path.criterion, output.criterion_percent);
    assert_eq!(path.potential_indices, output.potential_indices);
    assert_eq!(path.positions, output.positions);
    assert_eq!(path.beta, output.beta_angles);
    assert_eq!(path.eta, output.eta_angles);
    assert_eq!(path.leg_distances, output.leg_lengths);
    assert_eq!(path.amplitude, output.amplitudes);
    assert_eq!(path.phase, output.phases);

    assert_eq!(FeffBinPath::from(output.clone()), path);
}

#[test]
fn builds_data_from_genfmt_feff_bin_header() {
    let header = sample_genfmt_feff_bin_header();

    let data = FeffBinData::from(&header);
    assert_eq!(data.version, header.version);
    assert_eq!(data.pad_width, header.pad_width);
    assert_eq!(data.ihole, header.core_hole);
    assert_eq!(data.order, header.order);
    assert_eq!(
        data.initial_angular_momentum,
        header.initial_angular_momentum
    );
    assert_eq!(data.average_norman_radius, header.average_norman_radius);
    assert_eq!(data.fermi_level, header.fermi_level);
    assert_eq!(data.edge_energy, header.edge_energy);
    assert_eq!(data.potentials[0].label, "Cu");
    assert_eq!(data.potentials[0].atomic_number, 29);
    assert_eq!(data.potentials[1].label, "O");
    assert_eq!(data.potentials[1].atomic_number, 8);
    assert_eq!(data.central_phase_shift, header.central_phase_shifts);
    assert_eq!(data.complex_momentum, header.complex_momenta);
    assert_eq!(data.real_momentum, header.wave_numbers);
    assert!(data.paths.is_empty());
    assert!(data.raw_text.is_none());

    assert_eq!(FeffBinData::from(header.clone()), data);
}

#[test]
fn builds_complete_data_from_genfmt_output() -> Result<()> {
    let header = sample_genfmt_feff_bin_header();
    let first_path = sample_genfmt_retained_path_output();
    let mut second_path = sample_genfmt_retained_path_output();
    second_path.path_index = 23;
    second_path.criterion_percent = 6.25;
    second_path.amplitudes = Array1::from_vec(vec![0.5, 0.6, 0.7]);
    let retained_paths = vec![first_path.clone(), second_path.clone()];

    let data = FeffBinData::from_genfmt_output(&header, &retained_paths);

    assert_eq!(data.version, header.version);
    assert_eq!(data.potential_count(), header.potentials.len());
    assert_eq!(
        data.paths,
        vec![
            FeffBinPath::from(&first_path),
            FeffBinPath::from(&second_path)
        ]
    );
    assert!(data.raw_text.is_none());

    let rendered = feff_bin_string(&data)?;
    let parsed = parse_feff_bin(&rendered)?;
    assert_eq!(parsed.paths.len(), 2);
    assert_eq!(parsed.paths[0].index, 17);
    assert_eq!(parsed.paths[1].index, 23);
    Ok(())
}

#[test]
fn builds_complete_data_from_genfmt_output_collectors() {
    let header = sample_genfmt_feff_bin_header();
    let first_path = sample_genfmt_retained_path_output();
    let mut second_path = sample_genfmt_retained_path_output();
    second_path.path_index = 23;

    let ordinary_outputs = GenfmtOrdinaryPathOutputs {
        examined_path_count: 3,
        retained_path_count: 2,
        final_normalization: Some(4.5),
        path_summaries: Vec::new(),
        retained_paths: vec![first_path.clone(), second_path.clone()],
    };
    let ordinary_data = FeffBinData::from_genfmt_ordinary_outputs(&header, &ordinary_outputs);
    assert_eq!(
        ordinary_data.paths,
        vec![
            FeffBinPath::from(&first_path),
            FeffBinPath::from(&second_path)
        ]
    );

    let jas_outputs = GenfmtJasPathOutputs {
        examined_path_count: 2,
        retained_path_count: 1,
        final_normalization: Some(3.0),
        path_summaries: Vec::new(),
        retained_paths: vec![second_path.clone()],
        decomposed_paths: None,
    };
    let jas_data = FeffBinData::from_genfmt_jas_outputs(&header, &jas_outputs);
    assert_eq!(jas_data.paths, vec![FeffBinPath::from(&second_path)]);
}

#[test]
fn builds_complete_data_from_genfmt_driver_outputs() {
    let header = sample_genfmt_feff_bin_header();
    let first_path = sample_genfmt_retained_path_output();
    let mut second_path = sample_genfmt_retained_path_output();
    second_path.path_index = 23;

    let ordinary_output = GenfmtOrdinaryDriverOutput {
        header: header.clone(),
        path_sequence: GenfmtOrdinaryPathSequence {
            evaluations: Vec::new(),
            outputs: GenfmtOrdinaryPathOutputs {
                examined_path_count: 3,
                retained_path_count: 2,
                final_normalization: Some(4.5),
                path_summaries: Vec::new(),
                retained_paths: vec![first_path.clone(), second_path.clone()],
            },
        },
        nstar_rows: None,
    };
    let ordinary_data = FeffBinData::from_genfmt_ordinary_driver_output(&ordinary_output);
    assert_eq!(
        ordinary_data.paths,
        vec![
            FeffBinPath::from(&first_path),
            FeffBinPath::from(&second_path)
        ]
    );

    let jas_output = GenfmtJasDriverOutput {
        header: header.clone(),
        path_sequence: GenfmtJasPathSequence {
            evaluations: Vec::new(),
            outputs: GenfmtJasPathOutputs {
                examined_path_count: 2,
                retained_path_count: 1,
                final_normalization: Some(3.0),
                path_summaries: Vec::new(),
                retained_paths: vec![second_path.clone()],
                decomposed_paths: None,
            },
        },
        nstar_rows: None,
    };
    let jas_data = FeffBinData::from_genfmt_jas_driver_output(&jas_output);
    assert_eq!(jas_data.paths, vec![FeffBinPath::from(&second_path)]);
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

fn sample_genfmt_retained_path_output() -> GenfmtRetainedPathOutput {
    GenfmtRetainedPathOutput {
        path_index: 17,
        degeneracy: 4.0,
        criterion_percent: 12.5,
        effective_half_path_length_bohr: 2.4,
        effective_half_path_length_angstrom: 2.4 * FEFF_BIN_BOHR,
        list_sigma2: 0.0,
        potential_indices: Array1::from_vec(vec![1, 2, 0]),
        positions: Array2::from_shape_fn((3, 3), |(leg, axis)| match (leg, axis) {
            (0, 0) => 1.0,
            (0, 1) => 0.5,
            (0, 2) => -0.25,
            (1, 0) => 0.4,
            (1, 1) => -0.3,
            (1, 2) => 1.2,
            _ => 0.0,
        }),
        beta_angles: Array1::from_vec(vec![0.10, 0.20, 0.30]),
        eta_angles: Array1::from_vec(vec![0.40, 0.50, 0.60]),
        leg_lengths: Array1::from_vec(vec![1.0, 1.1, 1.2]),
        amplitudes: Array1::from_vec(vec![0.2, 0.3, 0.4]),
        phases: Array1::from_vec(vec![0.1, 1.2, 2.3]),
    }
}

fn sample_genfmt_feff_bin_header() -> GenfmtFeffBinHeader {
    GenfmtFeffBinHeader {
        version: "refeff-test".to_string(),
        pad_width: FEFF_BIN_DEFAULT_PAD_WIDTH,
        core_hole: 1,
        order: 2,
        initial_angular_momentum: 0,
        average_norman_radius: 1.25,
        fermi_level: -0.4,
        edge_energy: 9.1,
        potentials: vec![
            GenfmtFeffBinPotential {
                label: "Cu".to_string(),
                atomic_number: 29,
            },
            GenfmtFeffBinPotential {
                label: "O".to_string(),
                atomic_number: 8,
            },
            GenfmtFeffBinPotential {
                label: "C".to_string(),
                atomic_number: 6,
            },
        ],
        central_phase_shifts: Array1::from_vec(vec![
            Complex64::new(0.1, -0.01),
            Complex64::new(0.2, -0.02),
            Complex64::new(0.3, -0.03),
        ]),
        complex_momenta: Array1::from_vec(vec![
            Complex64::new(1.0, 0.1),
            Complex64::new(1.1, 0.2),
            Complex64::new(1.2, 0.3),
        ]),
        wave_numbers: Array1::from_vec(vec![0.5, 0.6, 0.7]),
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
