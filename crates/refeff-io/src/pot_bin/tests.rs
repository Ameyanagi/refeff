use ndarray::{Array1, Array2, Array3};

use crate::Result;
use crate::error::IoError;

use super::*;

#[test]
fn writes_header_and_integer_chunks_like_feff() -> Result<()> {
    let data = sample_pot_bin_data();
    let text = pot_bin_string(&data)?;
    assert_eq!(
        text.lines().next(),
        Some("    2    1    8   -1    2    3    4    5    6")
    );

    assert!(text.lines().any(|line| {
        line == "  -20  -19  -18  -17  -16  -15  -14  -13  -12  -11  -10   -9   -8   -7   -6   -5   -4   -3   -2   -1"
    }));
    assert!(text.lines().any(|line| {
        line == "    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19"
    }));
    assert!(text.lines().any(|line| line == "   20"));
    assert!(text.lines().any(|line| line == " -5 -4 -3 -2 -1  0  1  2"));
    assert!(text.lines().any(|line| line == "  3  4"));
    Ok(())
}

#[test]
fn roundtrips_pot_bin_text_with_pad_tolerance() -> Result<()> {
    let data = sample_pot_bin_data();
    let parsed = parse_pot_bin(&pot_bin_string(&data)?)?;
    assert_eq!(parsed.titles, data.titles);
    assert_eq!(parsed.pad_width, data.pad_width);
    assert_eq!(parsed.nohole, data.nohole);
    assert_eq!(parsed.ihole, data.ihole);
    assert_eq!(parsed.interstitial_selector, data.interstitial_selector);
    assert_eq!(parsed.automatic_folp, data.automatic_folp);
    assert_eq!(parsed.jump_mode, data.jump_mode);
    assert_eq!(parsed.unfreeze_f, data.unfreeze_f);
    assert_eq!(parsed.muffin_tin_indices, data.muffin_tin_indices);
    assert_eq!(parsed.norman_indices, data.norman_indices);
    assert_eq!(parsed.atomic_numbers, data.atomic_numbers);
    assert_eq!(parsed.kappa, data.kappa);
    assert_eq!(
        parsed.occupied_orbital_indices,
        data.occupied_orbital_indices
    );
    assert_close_iter(parsed.scalars.as_array(), data.scalars.as_array());
    assert_close_iter(parsed.muffin_tin_radii, data.muffin_tin_radii);
    assert_close_iter(parsed.large_components, data.large_components);
    assert_close_iter(parsed.large_coefficients, data.large_coefficients);
    assert_close_iter(parsed.electron_density, data.electron_density);
    assert_close_iter(parsed.orbital_occupancy, data.orbital_occupancy);
    assert_close_iter(parsed.valence_occupancy, data.valence_occupancy);
    Ok(())
}

#[test]
fn derives_fullspectrum_number_density_from_pot_bin() -> Result<()> {
    let data = sample_pot_bin_data();
    let copper_density = fullspectrum_number_density_from_pot_bin(29, &data)?;
    let oxygen_density = fullspectrum_number_density_from_pot_bin(8, &data)?;
    let missing_density = fullspectrum_number_density_from_pot_bin(26, &data)?;

    assert!((copper_density - 0.004_604_023_193_216_264).abs() < 1.0e-16);
    assert!((oxygen_density - 0.018_416_092_772_865_055).abs() < 1.0e-16);
    assert_eq!(missing_density, 0.0);
    Ok(())
}

#[test]
fn exposes_fullspectrum_rdpotp_fields_from_pot_bin() -> Result<()> {
    let data = sample_pot_bin_data();
    let state = fullspectrum_potential_state_from_pot_bin(&data)?;

    assert_eq!(state.title_count(), data.titles.len());
    assert_eq!(state.nph(), data.potential_count() - 1);
    assert_eq!(state.titles[0], data.titles[0]);
    assert!(state.atomic_numbers.iter().eq(data.atomic_numbers.iter()));
    assert!(
        state
            .potential_multiplicities
            .iter()
            .eq(data.potential_multiplicities.iter())
    );
    assert!(state.norman_radii.iter().eq(data.norman_radii.iter()));
    Ok(())
}

#[test]
fn rejects_bad_fullspectrum_rdpotp_view_inputs() {
    let mut data = sample_pot_bin_data();
    data.norman_radii = Array1::zeros(data.potential_count().saturating_sub(1));

    assert!(matches!(
        fullspectrum_potential_state_from_pot_bin(&data),
        Err(IoError::PotBinShape { field: "rnrm", .. })
    ));
}

#[test]
fn preserves_feff_title_record_spacing() -> Result<()> {
    let mut data = sample_pot_bin_data();
    data.titles[0] =
        " POT  SCF 100  4.0000   0, screened core-hole, AFOLP (folp(0)= 1.150)".to_string();
    let text = pot_bin_string(&data)?;
    assert_eq!(text.lines().nth(1), Some(data.titles[0].as_str()));
    let parsed = parse_pot_bin(&text)?;
    assert_eq!(parsed.titles[0], data.titles[0]);
    Ok(())
}

#[test]
fn preserves_matching_raw_text() -> Result<()> {
    let data = sample_pot_bin_data();
    let text = pot_bin_string(&data)?;
    let mut parsed = parse_pot_bin(&text)?;
    let raw_text = parsed
        .raw_text
        .as_mut()
        .ok_or(IoError::PotBinMissing { field: "raw_text" })?;
    raw_text.push('\n');

    let mut expected = text.clone();
    expected.push('\n');
    assert_eq!(pot_bin_string(&parsed)?, expected);

    parsed.scalars.fermi_level += 1.0;
    assert_ne!(pot_bin_string(&parsed)?, expected);
    Ok(())
}

#[test]
fn rejects_invalid_shapes_and_bad_tokens() {
    let mut bad = sample_pot_bin_data();
    bad.kappa = Array1::from_vec(vec![1]);
    assert!(matches!(
        pot_bin_string(&bad),
        Err(IoError::PotBinShape {
            field: "kappa",
            actual,
            expected,
        }) if actual == vec![1] && expected == vec![POT_BIN_ORBITALS]
    ));

    assert!(matches!(
        parse_pot_bin("not-an-int"),
        Err(IoError::PotBinParse {
            field: "header",
            ..
        })
    ));
}

fn sample_pot_bin_data() -> PotBinData {
    let potentials = 2;
    let angular_count = 4;
    PotBinData {
        titles: vec!["Cu crystal".to_string(), "second title".to_string()],
        pad_width: POT_BIN_DEFAULT_PAD_WIDTH,
        nohole: -1,
        ihole: 2,
        interstitial_selector: 3,
        automatic_folp: 4,
        jump_mode: 5,
        unfreeze_f: 6,
        scalars: PotBinScalars {
            average_norman_radius: 1.25,
            fermi_level: -0.4,
            interstitial_potential: -1.2,
            interstitial_density: 0.03,
            edge_position: 9.1,
            amplitude_reduction: 0.85,
            relaxation_energy: 0.15,
            plasmon_frequency: 2.4,
            core_valence_energy: -3.0,
            density_radius: 1.7,
            fermi_momentum: 0.9,
            total_charge: 42.0,
            total_volume: 11.0,
        },
        muffin_tin_indices: Array1::from_vec(vec![12, 13]),
        muffin_tin_radii: Array1::from_vec(vec![1.1, 1.2]),
        norman_indices: Array1::from_vec(vec![20, 21]),
        atomic_numbers: Array1::from_vec(vec![29, 8]),
        kappa: Array1::from_iter(-20..=20),
        norman_radii: Array1::from_vec(vec![2.1, 2.2]),
        overlap_factors: Array1::from_vec(vec![0.9, 0.8]),
        max_overlap_factors: Array1::from_vec(vec![1.3, 1.4]),
        potential_multiplicities: Array1::from_vec(vec![1.0, 4.0]),
        ionization: Array1::from_vec(vec![0.0, 1.0]),
        initial_large_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            0.001 * (row + 1) as f64
        }),
        initial_small_component: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| {
            -0.001 * (row + 1) as f64
        }),
        large_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                0.0001 * (row + 1) as f64 + 0.01 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_components: Array3::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, POT_BIN_ORBITALS, potentials),
            |(row, orbital, potential)| {
                -0.0001 * (row + 1) as f64 - 0.01 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        large_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                0.01 * (coef + 1) as f64 + 0.001 * orbital as f64 + 0.1 * potential as f64
            },
        ),
        small_coefficients: Array3::from_shape_fn(
            (POT_BIN_COEFFICIENTS, POT_BIN_ORBITALS, potentials),
            |(coef, orbital, potential)| {
                -0.01 * (coef + 1) as f64 - 0.001 * orbital as f64 - 0.1 * potential as f64
            },
        ),
        electron_density: radial_matrix(potentials, 0.01),
        coulomb_potential: radial_matrix(potentials, -0.02),
        total_potential: radial_matrix(potentials, -0.03),
        valence_density: radial_matrix(potentials, 0.004),
        valence_potential: radial_matrix(potentials, -0.005),
        magnetization_density: radial_matrix(potentials, 0.0002),
        orbital_occupancy: Array2::from_shape_fn(
            (POT_BIN_ORBITALS, potentials),
            |(orbital, potential)| 0.2 * orbital as f64 + potential as f64,
        ),
        orbital_energies: Array1::from_shape_fn(POT_BIN_ORBITALS, |orbital| {
            -10.0 + orbital as f64 * 0.25
        }),
        occupied_orbital_indices: Array2::from_shape_fn(
            (POT_BIN_IORB_SLOTS, potentials),
            |(slot, _)| slot as i32 - 5,
        ),
        norman_charges: Array1::from_vec(vec![28.5, 7.5]),
        valence_occupancy: Array2::from_shape_fn(
            (angular_count, potentials),
            |(angular, potential)| 0.5 * angular as f64 + potential as f64,
        ),
        raw_text: None,
    }
}

fn radial_matrix(potentials: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, potential)| {
        scale * (row + 1) as f64 + potential as f64 * 0.125
    })
}

fn assert_close_iter(
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
