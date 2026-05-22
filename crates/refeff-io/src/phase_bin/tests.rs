use super::*;

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;

use crate::error::{IoError, Result};

#[test]
fn writes_phase_bin_header_like_feff() -> Result<()> {
    let data = sample_phase_bin_data();
    let text = phase_bin_string(&data)?;
    assert_eq!(
        text.lines().next(),
        Some("    2    3    2    1    1    4    2    8    4    3    2")
    );
    assert!(text.lines().any(|line| line == "   1  29 Cu    "));
    assert!(text.lines().any(|line| line == "   2   8 O     "));
    Ok(())
}

#[test]
fn roundtrips_phase_bin_text_with_pad_tolerance() -> Result<()> {
    let data = sample_phase_bin_data();
    let parsed = parse_phase_bin(&phase_bin_string(&data)?)?;
    assert_eq!(parsed.spin_count, data.spin_count);
    assert_eq!(parsed.energy_count, data.energy_count);
    assert_eq!(parsed.main_energy_count, data.main_energy_count);
    assert_eq!(parsed.auxiliary_energy_count, data.auxiliary_energy_count);
    assert_eq!(parsed.ihole, data.ihole);
    assert_eq!(parsed.fermi_index, data.fermi_index);
    assert_eq!(parsed.pad_width, data.pad_width);
    assert_eq!(parsed.final_state_count, data.final_state_count);
    assert_eq!(parsed.transition_count, data.transition_count);
    assert_eq!(parsed.q_count, data.q_count);
    assert_close_reals(parsed.scalars.as_array(), data.scalars.as_array());
    assert_close_complex(parsed.energy_grid, data.energy_grid);
    assert_close_complex(parsed.reference_energy, data.reference_energy);
    assert_close_complex(parsed.transition_moments, data.transition_moments);
    assert_eq!(parsed.potentials.len(), data.potentials.len());
    for (actual, expected) in parsed.potentials.iter().zip(data.potentials.iter()) {
        assert_eq!(actual.lmax, expected.lmax);
        assert_eq!(actual.atomic_number, expected.atomic_number);
        assert_eq!(actual.label, expected.label);
        assert_close_complex(
            actual.phase_shifts.iter().copied(),
            expected.phase_shifts.iter().copied(),
        );
    }
    Ok(())
}

#[test]
fn parses_legacy_eight_integer_header() -> Result<()> {
    let mut text = phase_bin_string(&legacy_phase_bin_data())?;
    text.replace_range(
        0..text.lines().next().map_or(0, str::len),
        "    2    3    2    1    1    4    2    8",
    );
    let parsed = parse_phase_bin(&text)?;
    assert_eq!(parsed.final_state_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
    assert_eq!(parsed.transition_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
    assert_eq!(parsed.q_count, 1);
    Ok(())
}

#[test]
fn preserves_matching_raw_pad_blocks() -> Result<()> {
    let data = sample_phase_bin_data();
    let text = phase_bin_string(&data)?;
    let mut parsed = parse_phase_bin(&text)?;
    let raw_pads = parsed
        .raw_pads
        .as_mut()
        .ok_or(IoError::PhaseBinMissing { field: "raw_pads" })?;
    let scalars = raw_pads
        .scalars
        .as_mut()
        .ok_or(IoError::PhaseBinMissing { field: "dum" })?;
    scalars.push('\n');

    let energy_start = text
        .find('$')
        .ok_or(IoError::PhaseBinMissing { field: "em" })?;
    let mut expected = text.clone();
    expected.insert(energy_start, '\n');
    assert_eq!(phase_bin_string(&parsed)?, expected);

    parsed.scalars.fermi_level += 1.0;
    assert_ne!(phase_bin_string(&parsed)?, expected);
    Ok(())
}

#[test]
fn rejects_invalid_shapes_and_tokens() {
    let mut bad = sample_phase_bin_data();
    bad.energy_grid = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
    assert!(matches!(
        phase_bin_string(&bad),
        Err(IoError::PhaseBinShape {
            field: "em",
            actual,
            expected,
        }) if actual == vec![1] && expected == vec![3]
    ));

    assert!(matches!(
        parse_phase_bin("not-an-int"),
        Err(IoError::PhaseBinParse {
            field: "header",
            ..
        })
    ));
}

fn sample_phase_bin_data() -> PhaseBinData {
    let spin_count = 2;
    let energy_count = 3;
    let q_count = 2;
    let transition_count = 3;
    PhaseBinData {
        spin_count,
        energy_count,
        main_energy_count: 2,
        auxiliary_energy_count: 1,
        ihole: 4,
        fermi_index: 2,
        pad_width: PHASE_BIN_DEFAULT_PAD_WIDTH,
        final_state_count: 4,
        transition_count,
        q_count,
        scalars: PhaseBinScalars {
            average_norman_radius: 1.2,
            fermi_level: -0.35,
            edge_energy: 9.8,
        },
        energy_grid: Array1::from_shape_fn(energy_count, |energy| {
            Complex64::new(0.5 + energy as f64, 0.1 * energy as f64)
        }),
        reference_energy: Array2::from_shape_fn((energy_count, spin_count), |(energy, spin)| {
            Complex64::new(-1.0 + energy as f64 * 0.2, 0.05 * spin as f64)
        }),
        potentials: vec![
            sample_potential(1, 29, "Cu", energy_count, spin_count, 0.1),
            sample_potential(2, 8, "O", energy_count, spin_count, 0.2),
        ],
        transition_moments: Array4::from_shape_fn(
            (energy_count, q_count, transition_count, spin_count),
            |(energy, q_index, transition, spin)| {
                Complex64::new(
                    0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                    -0.02 * spin as f64,
                )
            },
        ),
        raw_pads: None,
    }
}

fn legacy_phase_bin_data() -> PhaseBinData {
    let mut data = sample_phase_bin_data();
    data.final_state_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
    data.transition_count = PHASE_BIN_DEFAULT_TRANSITION_COUNT;
    data.q_count = 1;
    data.transition_moments = Array4::from_shape_fn(
        (
            data.energy_count,
            data.q_count,
            data.transition_count,
            data.spin_count,
        ),
        |(energy, q_index, transition, spin)| {
            Complex64::new(
                0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                -0.02 * spin as f64,
            )
        },
    );
    data
}

fn sample_potential(
    lmax: usize,
    atomic_number: usize,
    label: &str,
    energy_count: usize,
    spin_count: usize,
    scale: f64,
) -> PhaseBinPotential {
    let l_count = 2 * lmax + 1;
    PhaseBinPotential {
        lmax,
        atomic_number,
        label: label.to_string(),
        phase_shifts: Array3::from_shape_fn(
            (energy_count, l_count, spin_count),
            |(energy, l_slot, spin)| {
                Complex64::new(
                    scale + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        ),
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
