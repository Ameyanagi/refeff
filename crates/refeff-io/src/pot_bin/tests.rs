use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use refeff_core::{Complex, RhorrpWavefunctionGridPreparation};

use crate::Result;
use crate::config_dat::{ConfigDatData, ConfigDatPotential, rhorrp_orbital_tables_from_config_dat};
use crate::error::IoError;
use crate::pad::encode_reals;
use crate::{RhorrpFmsInputHandoff, RhorrpPhaseBinHandoff, RhorrpPotInputControls};

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
fn roundtrips_full_iorb_payload_for_many_potentials() -> Result<()> {
    let data = sample_pot_bin_data_with_potentials(5);
    let parsed = parse_pot_bin(&pot_bin_string(&data)?)?;

    assert_eq!(parsed.potential_count(), 5);
    assert_eq!(
        parsed.occupied_orbital_indices.dim(),
        (POT_BIN_IORB_SLOTS, 5)
    );
    assert_eq!(
        parsed.occupied_orbital_indices,
        data.occupied_orbital_indices
    );
    assert_close_iter(parsed.norman_charges, data.norman_charges);
    assert_close_iter(parsed.valence_occupancy, data.valence_occupancy);
    Ok(())
}

#[test]
fn parses_legacy_thirty_orbital_pot_bin_by_padding_to_feff10_shape() -> Result<()> {
    let data = sample_pot_bin_data();
    let legacy_orbitals = 30;
    let parsed = parse_pot_bin(&legacy_pot_bin_string(&data, legacy_orbitals)?)?;

    assert_eq!(parsed.kappa.len(), POT_BIN_ORBITALS);
    for orbital in 0..legacy_orbitals {
        assert_eq!(parsed.kappa[orbital], data.kappa[orbital]);
        assert!((parsed.orbital_energies[orbital] - data.orbital_energies[orbital]).abs() < 1.0e-6);
    }
    for orbital in legacy_orbitals..POT_BIN_ORBITALS {
        assert_eq!(parsed.kappa[orbital], 0);
        assert_eq!(parsed.orbital_energies[orbital], 0.0);
    }

    assert_eq!(parsed.large_components.dim(), data.large_components.dim());
    assert_eq!(
        parsed.large_coefficients.dim(),
        data.large_coefficients.dim()
    );
    assert_eq!(parsed.orbital_occupancy.dim(), data.orbital_occupancy.dim());
    for potential in 0..data.potential_count() {
        for orbital in 0..legacy_orbitals {
            assert!(
                (parsed.large_components[(0, orbital, potential)]
                    - data.large_components[(0, orbital, potential)])
                    .abs()
                    < 1.0e-6
            );
            assert!(
                (parsed.large_coefficients[(0, orbital, potential)]
                    - data.large_coefficients[(0, orbital, potential)])
                    .abs()
                    < 1.0e-6
            );
            assert!(
                (parsed.orbital_occupancy[(orbital, potential)]
                    - data.orbital_occupancy[(orbital, potential)])
                    .abs()
                    < 1.0e-6
            );
        }
        for orbital in legacy_orbitals..POT_BIN_ORBITALS {
            assert_eq!(parsed.large_components[(0, orbital, potential)], 0.0);
            assert_eq!(parsed.small_components[(0, orbital, potential)], 0.0);
            assert_eq!(parsed.large_coefficients[(0, orbital, potential)], 0.0);
            assert_eq!(parsed.small_coefficients[(0, orbital, potential)], 0.0);
            assert_eq!(parsed.orbital_occupancy[(orbital, potential)], 0.0);
        }
        for slot in 0..8 {
            assert_eq!(
                parsed.occupied_orbital_indices[(slot, potential)],
                data.occupied_orbital_indices[(slot, potential)]
            );
        }
        for slot in 8..POT_BIN_IORB_SLOTS {
            assert_eq!(parsed.occupied_orbital_indices[(slot, potential)], 0);
        }
    }
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
fn exposes_rhorrp_wavefunction_handoff_from_pot_bin() -> Result<()> {
    let data = sample_pot_bin_data();
    let handoff = rhorrp_wavefunction_handoff_from_pot_bin(&data)?;

    assert_eq!(handoff.potential_count(), 2);
    assert_eq!(handoff.muffin_tin_radii(), &[1.1, 1.2]);
    assert_eq!(handoff.norman_radii(), &[2.1, 2.2]);
    assert_eq!(handoff.atomic_numbers(), &[29.0, 8.0]);

    let grid = handoff.grid_preparation_input(0.04, 5, POT_BIN_RADIAL_POINTS);
    assert_eq!(grid.muffin_tin_radii, handoff.muffin_tin_radii());
    assert_eq!(
        grid.bound_large_components.dim(),
        data.large_components.dim()
    );
    assert_eq!(
        grid.bound_small_components.dim(),
        data.small_components.dim()
    );
    assert_eq!(grid.electron_density.dim(), data.electron_density.dim());
    assert_eq!(
        grid.interstitial_potential,
        data.scalars.interstitial_potential
    );
    assert_eq!(grid.interstitial_density, data.scalars.interstitial_density);
    assert_eq!(grid.original_radial_dx, RHORRP_POT_BIN_RADIAL_DX);
    assert_eq!(grid.target_radial_dx, 0.04);
    assert_eq!(grid.jump_mode, data.jump_mode);
    assert_eq!(grid.exchange_index, 5);
    assert_eq!(grid.radial_count, POT_BIN_RADIAL_POINTS);
    Ok(())
}

#[test]
fn combines_rhorrp_pot_and_config_handoffs_for_prepared_tables() -> Result<()> {
    let data = sample_pot_bin_data();
    let handoff = rhorrp_wavefunction_handoff_from_pot_bin(&data)?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_rhorrp_config_dat())?;
    let prepared = sample_rhorrp_prepared(data.potential_count());
    let energies = Array1::from_vec(vec![Complex::new(0.15, 0.02), Complex::new(0.20, 0.04)]);

    let input = handoff.prepared_wavefunction_tables_input(
        &prepared,
        &orbital_tables,
        energies.view(),
        14,
        3,
    )?;

    assert_eq!(input.muffin_tin_radii, handoff.muffin_tin_radii());
    assert_eq!(input.norman_radii, handoff.norman_radii());
    assert_eq!(input.atomic_numbers, handoff.atomic_numbers());
    assert_eq!(
        input.bound_large_coefficients_by_potential.dim(),
        data.large_coefficients.dim()
    );
    assert_eq!(
        input.bound_small_coefficients_by_potential.dim(),
        data.small_coefficients.dim()
    );
    assert_eq!(
        input.electron_counts_by_potential.dim(),
        orbital_tables.electron_counts_by_potential.dim()
    );
    assert_eq!(
        input.valence_counts_by_potential.dim(),
        data.orbital_occupancy.dim()
    );
    assert_eq!(
        input.kappa_by_potential.dim(),
        orbital_tables.kappa_by_potential.dim()
    );
    assert_eq!(input.bound_orbital_counts, &[2, 3]);
    assert_eq!(input.exchange_index, 14);
    assert_eq!(input.angular_momentum_count, 3);
    assert_eq!(
        input.valence_counts_by_potential[(2, 1)],
        data.orbital_occupancy[(2, 1)]
    );
    Ok(())
}

#[test]
fn composes_rhorrp_wavefunction_tables_from_handoff_files() -> Result<()> {
    let data = sample_pot_bin_data();
    let config = sample_rhorrp_config_dat();
    let phase = sample_rhorrp_phase_handoff(data.potential_count(), 1);
    let fms = sample_rhorrp_fms_handoff(data.potential_count(), 0);
    let controls = sample_rhorrp_pot_controls();

    let handoff = rhorrp_wavefunction_tables_from_handoffs(RhorrpWavefunctionTablesHandoffInput {
        pot: &data,
        config: &config,
        phase: &phase,
        fms,
        controls,
        radial_count: RHORRP_WAVEFUNCTION_RADIAL_COUNT,
    })?;

    assert_eq!(handoff.prepared.potential_count(), data.potential_count());
    assert_eq!(handoff.wavefunctions.energy_count(), phase.energy_count());
    assert_eq!(handoff.wavefunctions.angular_momentum_count(), 1);
    assert_eq!(
        handoff.wavefunctions.potential_count(),
        data.potential_count()
    );
    assert_eq!(
        handoff.wavefunctions.radial_count(),
        RHORRP_WAVEFUNCTION_RADIAL_COUNT
    );
    assert_eq!(handoff.radial_x0, RHORRP_WAVEFUNCTION_RADIAL_X0);
    assert_eq!(handoff.radial_dx, controls.target_radial_dx);
    assert!(
        handoff.reference_energy_hartree.re.is_finite()
            && handoff.reference_energy_hartree.im.is_finite()
    );
    Ok(())
}

#[test]
fn rejects_inconsistent_rhorrp_wavefunction_handoffs() -> Result<()> {
    let data = sample_pot_bin_data();
    let handoff = rhorrp_wavefunction_handoff_from_pot_bin(&data)?;
    let orbital_tables = rhorrp_orbital_tables_from_config_dat(&sample_rhorrp_config_dat())?;
    let energies = Array1::from_vec(vec![Complex::new(0.15, 0.02)]);

    let wrong_prepared_potentials = sample_rhorrp_prepared(1);
    assert!(
        handoff
            .prepared_wavefunction_tables_input(
                &wrong_prepared_potentials,
                &orbital_tables,
                energies.view(),
                14,
                3,
            )
            .is_err()
    );

    let mut one_potential_config = sample_rhorrp_config_dat();
    one_potential_config.potentials.pop();
    let one_potential_tables = rhorrp_orbital_tables_from_config_dat(&one_potential_config)?;
    let prepared = sample_rhorrp_prepared(data.potential_count());
    assert!(
        handoff
            .prepared_wavefunction_tables_input(
                &prepared,
                &one_potential_tables,
                energies.view(),
                14,
                3,
            )
            .is_err()
    );

    let phase = sample_rhorrp_phase_handoff(data.potential_count(), 1);
    let bad_fms = sample_rhorrp_fms_handoff(1, 0);
    assert!(matches!(
        rhorrp_wavefunction_tables_from_handoffs(RhorrpWavefunctionTablesHandoffInput {
            pot: &data,
            config: &sample_rhorrp_config_dat(),
            phase: &phase,
            fms: bad_fms,
            controls: sample_rhorrp_pot_controls(),
            radial_count: RHORRP_WAVEFUNCTION_RADIAL_COUNT,
        }),
        Err(IoError::InvalidPotBin { .. })
    ));
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
    sample_pot_bin_data_with_potentials(2)
}

fn sample_pot_bin_data_with_potentials(potentials: usize) -> PotBinData {
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
        muffin_tin_indices: Array1::from_shape_fn(potentials, |potential| 12 + potential),
        muffin_tin_radii: Array1::from_shape_fn(potentials, |potential| match potential {
            0 => 1.1,
            1 => 1.2,
            _ => 1.1 + 0.1 * potential as f64,
        }),
        norman_indices: Array1::from_shape_fn(potentials, |potential| 20 + potential),
        atomic_numbers: Array1::from_shape_fn(potentials, |potential| match potential {
            0 => 29,
            1 => 8,
            _ => 20 + potential,
        }),
        kappa: Array1::from_iter(-20..=20),
        norman_radii: Array1::from_shape_fn(potentials, |potential| 2.1 + 0.1 * potential as f64),
        overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            0.9 - 0.1 * potential as f64
        }),
        max_overlap_factors: Array1::from_shape_fn(potentials, |potential| {
            1.3 + 0.1 * potential as f64
        }),
        potential_multiplicities: Array1::from_shape_fn(potentials, |potential| {
            if potential == 0 {
                1.0
            } else if potential == 1 {
                4.0
            } else {
                potential as f64 + 1.0
            }
        }),
        ionization: Array1::from_shape_fn(potentials, |potential| potential as f64),
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
        norman_charges: Array1::from_shape_fn(potentials, |potential| match potential {
            0 => 28.5,
            1 => 7.5,
            _ => potential as f64 + 0.5,
        }),
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

fn legacy_pot_bin_string(data: &PotBinData, orbital_count: usize) -> Result<String> {
    let mut out = String::new();
    out.push_str(&format!(
        "{} {} {} {} {} {} {} {} {}\n",
        data.titles.len(),
        data.potential_count() - 1,
        data.pad_width,
        data.nohole,
        data.ihole,
        data.interstitial_selector,
        data.automatic_folp,
        data.jump_mode,
        data.unfreeze_f
    ));
    for title in &data.titles {
        out.push_str(title);
        out.push('\n');
    }

    out.push_str(&encode_reals(&data.scalars.as_array(), data.pad_width)?);
    push_int_line(&mut out, data.muffin_tin_indices.iter().copied());
    out.push_str(&encode_reals(
        &data.muffin_tin_radii.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?);
    push_int_line(&mut out, data.norman_indices.iter().copied());
    push_int_line(&mut out, data.atomic_numbers.iter().copied());
    push_int_line(&mut out, data.kappa.iter().take(orbital_count).copied());

    for values in [
        data.norman_radii.view(),
        data.overlap_factors.view(),
        data.max_overlap_factors.view(),
        data.potential_multiplicities.view(),
        data.ionization.view(),
        data.initial_large_component.view(),
        data.initial_small_component.view(),
    ] {
        out.push_str(&encode_reals(
            &values.iter().copied().collect::<Vec<_>>(),
            data.pad_width,
        )?);
    }

    for values in [
        data.large_components.view(),
        data.small_components.view(),
        data.large_coefficients.view(),
        data.small_coefficients.view(),
    ] {
        out.push_str(&encode_reals(
            &flatten3_orbitals(values, orbital_count),
            data.pad_width,
        )?);
    }

    for values in [
        data.electron_density.view(),
        data.coulomb_potential.view(),
        data.total_potential.view(),
        data.valence_density.view(),
        data.valence_potential.view(),
        data.magnetization_density.view(),
    ] {
        out.push_str(&encode_reals(&flatten2(values), data.pad_width)?);
    }
    out.push_str(&encode_reals(
        &flatten2_orbitals(data.orbital_occupancy.view(), orbital_count),
        data.pad_width,
    )?);
    out.push_str(&encode_reals(
        &data
            .orbital_energies
            .iter()
            .take(orbital_count)
            .copied()
            .collect::<Vec<_>>(),
        data.pad_width,
    )?);
    for potential in 0..data.potential_count() {
        push_int_line(
            &mut out,
            (0..8).map(|slot| data.occupied_orbital_indices[(slot, potential)]),
        );
    }
    out.push_str(&encode_reals(
        &data.norman_charges.iter().copied().collect::<Vec<_>>(),
        data.pad_width,
    )?);
    out.push_str(&encode_reals(
        &flatten2(data.valence_occupancy.view()),
        data.pad_width,
    )?);
    Ok(out)
}

fn push_int_line<T: std::fmt::Display>(out: &mut String, values: impl IntoIterator<Item = T>) {
    let mut first = true;
    for value in values {
        if !first {
            out.push(' ');
        }
        first = false;
        out.push_str(&value.to_string());
    }
    out.push('\n');
}

fn flatten2(values: ArrayView2<'_, f64>) -> Vec<f64> {
    let (rows, cols) = values.dim();
    let mut flat = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            flat.push(values[(row, col)]);
        }
    }
    flat
}

fn flatten2_orbitals(values: ArrayView2<'_, f64>, orbital_count: usize) -> Vec<f64> {
    let (_, cols) = values.dim();
    let mut flat = Vec::with_capacity(orbital_count * cols);
    for col in 0..cols {
        for row in 0..orbital_count {
            flat.push(values[(row, col)]);
        }
    }
    flat
}

fn flatten3_orbitals(values: ArrayView3<'_, f64>, orbital_count: usize) -> Vec<f64> {
    let (rows, _, planes) = values.dim();
    let mut flat = Vec::with_capacity(rows * orbital_count * planes);
    for plane in 0..planes {
        for col in 0..orbital_count {
            for row in 0..rows {
                flat.push(values[(row, col, plane)]);
            }
        }
    }
    flat
}

fn sample_rhorrp_config_dat() -> ConfigDatData {
    let mut first_occupations = Array1::zeros(crate::CONFIG_DAT_ORBITAL_COUNT);
    let mut first_valence = Array1::zeros(crate::CONFIG_DAT_ORBITAL_COUNT);
    first_occupations[0] = 1.0;
    first_occupations[1] = 2.0;
    first_valence[1] = 0.5;

    let mut second_occupations = Array1::zeros(crate::CONFIG_DAT_ORBITAL_COUNT);
    let mut second_valence = Array1::zeros(crate::CONFIG_DAT_ORBITAL_COUNT);
    second_occupations[0] = 2.0;
    second_occupations[1] = 2.0;
    second_occupations[2] = 1.0;
    second_valence[2] = 1.0;

    ConfigDatData {
        header_lines: Vec::new(),
        potentials: vec![
            ConfigDatPotential {
                potential_index: 0,
                atomic_number: 29,
                element: "Cu".to_string(),
                occupations: first_occupations,
                valence_occupations: first_valence,
                spin_occupations: None,
            },
            ConfigDatPotential {
                potential_index: 1,
                atomic_number: 8,
                element: "O".to_string(),
                occupations: second_occupations,
                valence_occupations: second_valence,
                spin_occupations: None,
            },
        ],
    }
}

fn sample_rhorrp_phase_handoff(potentials: usize, angular_count: usize) -> RhorrpPhaseBinHandoff {
    RhorrpPhaseBinHandoff {
        energies_hartree: Array1::from_vec(vec![
            Complex::new(0.15, 0.02),
            Complex::new(0.20, 0.04),
        ]),
        chemical_potential_hartree: 0.045,
        real_axis_count: 2,
        xsph_phase_shifts: Array3::zeros((2, angular_count, potentials)),
    }
}

fn sample_rhorrp_fms_handoff(
    potential_count: usize,
    max_angular_momentum: usize,
) -> RhorrpFmsInputHandoff {
    RhorrpFmsInputHandoff {
        fms_radius_bohr: 1.0,
        potential_count,
        max_angular_momentum,
        angular_momentum_count: max_angular_momentum + 1,
    }
}

fn sample_rhorrp_pot_controls() -> RhorrpPotInputControls {
    RhorrpPotInputControls {
        exchange_index: 0,
        target_radial_dx: RHORRP_POT_BIN_RADIAL_DX,
        raw_temperature_hartree: 0.001,
        temperature_hartree: 0.001,
    }
}

fn sample_rhorrp_prepared(potentials: usize) -> RhorrpWavefunctionGridPreparation {
    RhorrpWavefunctionGridPreparation {
        radii: Array1::from_shape_fn(POT_BIN_RADIAL_POINTS, |row| 0.01 * (row + 1) as f64),
        radial_dx: RHORRP_POT_BIN_RADIAL_DX,
        potential_jumps: Array1::zeros(potentials),
        reference_indices_1based: vec![12; potentials],
        reference_energies_hartree: Array1::from_shape_fn(potentials, |potential| {
            Complex::new(-0.2 + 0.01 * potential as f64, 0.0)
        }),
        total_potential: Array2::from_shape_fn((POT_BIN_RADIAL_POINTS, potentials), |(row, _)| {
            Complex::new(-0.01 * (row + 1) as f64, 0.0)
        }),
        valence_potential: Array2::from_shape_fn(
            (POT_BIN_RADIAL_POINTS, potentials),
            |(row, _)| Complex::new(-0.02 * (row + 1) as f64, 0.0),
        ),
        bound_large_components: Array3::zeros((
            POT_BIN_RADIAL_POINTS,
            POT_BIN_ORBITALS,
            potentials,
        )),
        bound_small_components: Array3::zeros((
            POT_BIN_RADIAL_POINTS,
            POT_BIN_ORBITALS,
            potentials,
        )),
        bound_active_lengths: Array2::zeros((POT_BIN_ORBITALS, potentials)),
    }
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
