use crate::error::Result;

use ndarray::{Array1, Array2};

use super::common::invalid_apot_bin;
use super::parse::parse_matrix_shape;
use super::*;

#[test]
fn parses_records_and_matrix_sections() -> Result<()> {
    let data = parse_apot_bin(APOT_BIN)?;
    assert_eq!(data.section_count(), 3);
    assert_eq!(data.matrix_count(), 1);

    let first = &data.sections[0];
    assert_eq!(first.section_number, 1);
    assert_eq!(first.column_labels, ["nph", "nat", "ihole", "s02"]);
    let Some(records) = first.records() else {
        return invalid_apot_bin(0, "first section should contain records");
    };
    assert_eq!(records.row_count(), 1);
    assert_eq!(records.column_count(), 4);
    assert_eq!(records.rows[0][0], ApotBinValue::Int(1));
    assert_eq!(records.rows[0][3], ApotBinValue::Real(0.95));

    let Some(matrix) = data.sections[2].matrix() else {
        return invalid_apot_bin(0, "third section should contain a matrix");
    };
    assert_eq!(matrix.shape(), (2, 3));
    match &matrix.values {
        ApotBinMatrixValues::Real(values) => assert_eq!(values[[1, 2]], 6.0),
        _ => return invalid_apot_bin(0, "third section should contain real matrix values"),
    }
    assert_eq!(data.sections[2].trailing_headers, ["next block"]);

    Ok(())
}

#[test]
fn parses_compact_i4_shape_fields() -> Result<()> {
    assert_eq!(parse_matrix_shape(1, "   34000")?, (3, 4000));
    Ok(())
}

#[test]
fn roundtrips_apot_bin_data() -> Result<()> {
    let data = parse_apot_bin(APOT_BIN)?;
    let rendered = apot_bin_string(&data)?;
    let reparsed = parse_apot_bin(&rendered)?;
    assert_eq!(reparsed, data);
    Ok(())
}

#[test]
fn builds_atomic_scf_state_sections_for_source_consumers() -> Result<()> {
    let state_count = 2;
    let state_index = 1;
    let orbital_count = 2;
    let density_4pi = Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| 10.0 + row as f64);
    let coulomb_potential =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| -20.0 - row as f64);
    let valence_density_4pi =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| 1.0 + 0.5 * row as f64);
    let valence_occupations = Array1::from_vec(vec![0.25, 1.75]);
    let orbital_energies = Array1::from_vec(vec![-32.0, -4.5]);
    let kappas = Array1::from_vec(vec![-1, 1]);
    let large_components = Array2::from_shape_fn(
        (APOT_ATOMIC_RADIAL_POINTS, orbital_count),
        |(row, orbital)| 1000.0 + 10.0 * row as f64 + orbital as f64,
    );
    let small_components = Array2::from_shape_fn(
        (APOT_ATOMIC_RADIAL_POINTS, orbital_count),
        |(row, orbital)| -1000.0 - 10.0 * row as f64 - orbital as f64,
    );
    let large_coefficients = Array2::from_shape_fn(
        (APOT_ATOMIC_COEFFICIENTS, orbital_count),
        |(row, orbital)| 0.01 * row as f64 + orbital as f64,
    );
    let small_coefficients = Array2::from_shape_fn(
        (APOT_ATOMIC_COEFFICIENTS, orbital_count),
        |(row, orbital)| -0.02 * row as f64 - orbital as f64,
    );

    let sections = apot_atomic_scf_state_sections(ApotAtomicScfStateSectionsInput {
        state_count,
        state_index,
        orbital_count,
        density_4pi: density_4pi.view(),
        coulomb_potential: coulomb_potential.view(),
        valence_density_4pi: valence_density_4pi.view(),
        valence_occupations: valence_occupations.view(),
        orbital_energies: orbital_energies.view(),
        kappas: kappas.view(),
        large_components: large_components.view(),
        small_components: small_components.view(),
        large_coefficients: large_coefficients.view(),
        small_coefficients: small_coefficients.view(),
    })?;

    assert_eq!(
        sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        vec![3, 8, 10, 11, 13, 14, 20, 23, 25, 27, 29]
    );

    let reparsed = parse_apot_bin(&apot_bin_string(&ApotBinData { sections })?)?;
    let ApotBinPayload::Records(records) =
        &section(&reparsed, APOT_ATOMIC_NORB_SECTION_NUMBER)?.payload
    else {
        return invalid_apot_bin(0, "norb should be row records");
    };
    assert_eq!(records.rows[0][0], ApotBinValue::Int(0));
    assert_eq!(records.rows[1][0], ApotBinValue::Int(2));

    let density = real_matrix_values(&reparsed, APOT_ATOMIC_DENSITY_SECTION_NUMBER)?;
    assert_eq!(density.dim(), (APOT_ATOMIC_RADIAL_POINTS, state_count));
    assert_close(density[[0, 0]], 0.0);
    assert_close(density[[3, state_index]], density_4pi[3]);

    let valence_density =
        real_matrix_values(&reparsed, APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER)?;
    assert_close(valence_density[[4, state_index]], valence_density_4pi[4]);

    let coulomb = real_matrix_values(&reparsed, APOT_ATOMIC_COULOMB_SECTION_NUMBER)?;
    assert_close(coulomb[[4, state_index]], coulomb_potential[4]);

    let valence = real_matrix_values(&reparsed, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER)?;
    assert_eq!(valence.dim(), (APOT_ATOMIC_ORBITAL_SLOTS, state_count));
    assert_close(valence[[1, state_index]], valence_occupations[1]);
    assert_close(valence[[2, state_index]], 0.0);

    let energies = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER)?;
    assert_close(energies[[0, state_index]], orbital_energies[0]);

    let kappa = int_matrix_values(&reparsed, APOT_ATOMIC_KAPPA_SECTION_NUMBER)?;
    assert_eq!(kappa.dim(), (APOT_ATOMIC_ORBITAL_SLOTS, state_count));
    assert_eq!(kappa[[0, state_index]], -1);
    assert_eq!(kappa[[1, state_index]], 1);
    assert_eq!(kappa[[2, state_index]], 0);

    let dgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + state_index)?;
    assert_eq!(
        dgc.dim(),
        (APOT_ATOMIC_RADIAL_POINTS, APOT_ATOMIC_ORBITAL_SLOTS)
    );
    assert_close(dgc[[5, 1]], large_components[[5, 1]]);
    assert_close(dgc[[5, 2]], 0.0);

    let dpc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + state_count + state_index,
    )?;
    assert_close(dpc[[5, 1]], small_components[[5, 1]]);

    let adgc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + 2 * state_count + state_index,
    )?;
    assert_eq!(
        adgc.dim(),
        (APOT_ATOMIC_COEFFICIENTS, APOT_ATOMIC_ORBITAL_SLOTS)
    );
    assert_close(adgc[[3, 1]], large_coefficients[[3, 1]]);

    let adpc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + 3 * state_count + state_index,
    )?;
    assert_close(adpc[[3, 1]], small_coefficients[[3, 1]]);

    Ok(())
}

#[test]
fn builds_atomic_scf_state_sections_from_core_atom_state() -> anyhow::Result<()> {
    let configuration = refeff_core::OrbitalConfiguration {
        orbital_count: 2,
        core_orbital_count: 2,
        projection_orbitals: Array1::zeros(10),
        hole_position: 0,
        principal_quantum_numbers: Array1::from_vec(vec![1, 2]),
        kappa: Array1::from_vec(vec![-1, -1]),
        electron_counts: Array1::from_vec(vec![2.0, 2.0]),
        valence_counts: Array1::from_vec(vec![0.0, 0.0]),
        spin_magnetization: Array1::zeros(2),
        ionization_orbital: 0,
        screening_orbital: 0,
        last_occupied_orbital: 2,
        template_atomic_number: 4,
        ionicity_delta: 0.0,
    };
    let state = refeff_core::atomic::atomic_scf_state_from_configuration(
        refeff_core::atomic::AtomicScfStateInput {
            atomic_number: 4,
            ionicity: 0.0,
            thomas_fermi_ionicity: -1.0,
            configuration: &configuration,
            exchange_mode: refeff_core::atomic::AtomicLocalDensityExchangeMode::DiracFockOnly,
            max_orbital_iterations: 40,
            speed_of_light: 137.0373,
            step: 0.05,
            first_radius_times_charge: 4.0 * (-8.8_f64).exp(),
            requested_nucleus_index: 11,
        },
    )?;

    let state_count = 2;
    let state_index = 1;
    let sections = apot_atomic_scf_state_sections_from_state(state_count, state_index, &state)?;

    assert_eq!(
        sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        vec![3, 8, 10, 11, 13, 14, 20, 23, 25, 27, 29]
    );

    let merged = apot_atomic_scf_sections_from_states(
        state_count,
        &[ApotAtomicScfStateRef {
            state_index,
            state: &state,
        }],
    )?;
    assert_eq!(
        merged
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        vec![3, 8, 10, 11, 13, 14, 20, 23, 25, 27, 29]
    );

    let reparsed = parse_apot_bin(&apot_bin_string(&ApotBinData { sections })?)?;
    let ApotBinPayload::Records(records) =
        &section(&reparsed, APOT_ATOMIC_NORB_SECTION_NUMBER)?.payload
    else {
        return Err(crate::error::IoError::InvalidApotBin {
            line: 0,
            message: "norb should be row records".to_string(),
        }
        .into());
    };
    assert_eq!(records.rows[0][0], ApotBinValue::Int(0));
    assert_eq!(records.rows[state_index][0], ApotBinValue::Int(2));

    let density = real_matrix_values(&reparsed, APOT_ATOMIC_DENSITY_SECTION_NUMBER)?;
    assert_close(density[[0, 0]], 0.0);
    assert_close(density[[7, state_index]], state.scf.density_4pi[7]);

    let valence_density =
        real_matrix_values(&reparsed, APOT_ATOMIC_VALENCE_DENSITY_SECTION_NUMBER)?;
    assert_close(
        valence_density[[7, state_index]],
        state.scf.valence_density_4pi[7],
    );

    let coulomb = real_matrix_values(&reparsed, APOT_ATOMIC_COULOMB_SECTION_NUMBER)?;
    assert_close(coulomb[[7, state_index]], state.scf.coulomb_potential[7]);

    let valence = real_matrix_values(&reparsed, APOT_ATOMIC_VALENCE_OCCUPATION_SECTION_NUMBER)?;
    assert_close(valence[[0, state_index]], state.valence_occupations[0]);
    assert_close(valence[[1, state_index]], state.valence_occupations[1]);
    assert_close(valence[[2, state_index]], 0.0);

    let energies = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER)?;
    assert_close(energies[[0, state_index]], state.scf.orbital_energies[0]);
    assert_close(energies[[1, state_index]], state.scf.orbital_energies[1]);

    let kappa = int_matrix_values(&reparsed, APOT_ATOMIC_KAPPA_SECTION_NUMBER)?;
    assert_eq!(kappa[[0, state_index]], state.kappas[0] as i64);
    assert_eq!(kappa[[1, state_index]], state.kappas[1] as i64);
    assert_eq!(kappa[[2, state_index]], 0);

    let dgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + state_index)?;
    assert_close(dgc[[9, 0]], state.scf.large_components[[9, 0]]);
    assert_close(dgc[[9, 1]], state.scf.large_components[[9, 1]]);

    let dpc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + state_count + state_index,
    )?;
    assert_close(dpc[[9, 0]], state.scf.small_components[[9, 0]]);

    let adgc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + 2 * state_count + state_index,
    )?;
    assert_close(adgc[[3, 0]], state.scf.large_coefficients[[3, 0]]);

    let adpc = real_matrix_values(
        &reparsed,
        APOT_ATOMIC_ORBITAL_SECTION_START + 3 * state_count + state_index,
    )?;
    assert_close(adpc[[3, 0]], state.scf.small_coefficients[[3, 0]]);

    Ok(())
}

#[test]
fn builds_merged_atomic_scf_sections_in_feff_group_order() -> Result<()> {
    let state_count = 2;
    let first = SyntheticScfState::new(100.0, 1);
    let second = SyntheticScfState::new(200.0, 2);

    let sections =
        apot_atomic_scf_sections(&[second.input(state_count, 1), first.input(state_count, 0)])?;

    assert_eq!(
        sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        vec![3, 8, 10, 11, 13, 14, 20, 22, 23, 24, 25, 26, 27, 28, 29]
    );

    let reparsed = parse_apot_bin(&apot_bin_string(&ApotBinData { sections })?)?;
    let ApotBinPayload::Records(records) =
        &section(&reparsed, APOT_ATOMIC_NORB_SECTION_NUMBER)?.payload
    else {
        return invalid_apot_bin(0, "norb should be row records");
    };
    assert_eq!(records.rows[0][0], ApotBinValue::Int(1));
    assert_eq!(records.rows[1][0], ApotBinValue::Int(2));

    let density = real_matrix_values(&reparsed, APOT_ATOMIC_DENSITY_SECTION_NUMBER)?;
    assert_close(density[[4, 0]], first.density_4pi[4]);
    assert_close(density[[4, 1]], second.density_4pi[4]);

    let energies = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_ENERGY_SECTION_NUMBER)?;
    assert_close(energies[[0, 0]], first.orbital_energies[0]);
    assert_close(energies[[1, 0]], 0.0);
    assert_close(energies[[1, 1]], second.orbital_energies[1]);

    let kappa = int_matrix_values(&reparsed, APOT_ATOMIC_KAPPA_SECTION_NUMBER)?;
    assert_eq!(kappa[[0, 0]], first.kappas[0] as i64);
    assert_eq!(kappa[[1, 0]], 0);
    assert_eq!(kappa[[1, 1]], second.kappas[1] as i64);

    let first_dgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START)?;
    assert_close(first_dgc[[5, 0]], first.large_components[[5, 0]]);

    let second_dgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 1)?;
    assert_close(second_dgc[[5, 1]], second.large_components[[5, 1]]);

    let first_dpc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 2)?;
    assert_close(first_dpc[[5, 0]], first.small_components[[5, 0]]);

    let second_dpc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 3)?;
    assert_close(second_dpc[[5, 1]], second.small_components[[5, 1]]);

    let first_adgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 4)?;
    assert_close(first_adgc[[3, 0]], first.large_coefficients[[3, 0]]);

    let second_adgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 5)?;
    assert_close(second_adgc[[3, 1]], second.large_coefficients[[3, 1]]);

    let first_adpc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 6)?;
    assert_close(first_adpc[[3, 0]], first.small_coefficients[[3, 0]]);

    let second_adpc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 7)?;
    assert_close(second_adpc[[3, 1]], second.small_coefficients[[3, 1]]);

    Ok(())
}

#[test]
fn builds_full_atomic_pots_sections_in_writeatomicpots_order() -> Result<()> {
    let unique_potential_count = 0;
    let unique_count = 1;
    let state_count = 2;
    let atom_count = 1;
    let initial = SyntheticScfState::new(300.0, 1);
    let final_state = SyntheticScfState::new(400.0, 2);
    let state_inputs = [
        initial.input(state_count, 0),
        final_state.input(state_count, 1),
    ];

    let atomic_numbers = Array1::from_vec(vec![29_i64]);
    let model_atom_indices = Array1::from_vec(vec![1_i64]);
    let overlap_shell_counts = Array1::from_vec(vec![1_i64]);
    let norman_radii = Array1::from_vec(vec![2.25]);
    let atom_potential_indices = Array1::from_vec(vec![0_i64]);
    let core_hole_large_component =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| 0.01 * row as f64);
    let core_hole_small_component =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| -0.001 * row as f64);
    let core_hole_density =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| 0.02 + 0.0001 * row as f64);
    let core_hole_coulomb_potential =
        Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| -0.03 - 0.0002 * row as f64);
    let overlap_potential_indices = Array2::from_elem((1, unique_count), 0_i64);
    let overlap_shell_atom_counts = Array2::from_elem((1, unique_count), 1_i64);
    let magnetization_density =
        Array2::from_shape_fn((APOT_ATOMIC_RADIAL_POINTS, state_count), |(row, state)| {
            0.001 * row as f64 + state as f64
        });
    let norman_valence_counts = Array2::from_shape_fn((4, state_count), |(row, state)| {
        row as f64 + 0.1 * state as f64
    });
    let atom_positions = Array2::from_shape_fn((3, atom_count), |(axis, _)| axis as f64 + 0.5);
    let overlap_radii = Array2::from_elem((1, unique_count), 2.5);
    let overlapped_density =
        Array2::from_shape_fn((APOT_ATOMIC_RADIAL_POINTS, unique_count), |(row, _)| {
            0.2 + 0.001 * row as f64
        });
    let overlapped_valence_density =
        Array2::from_shape_fn((APOT_ATOMIC_RADIAL_POINTS, unique_count), |(row, _)| {
            0.1 + 0.0005 * row as f64
        });
    let overlapped_coulomb_potential =
        Array2::from_shape_fn((APOT_ATOMIC_RADIAL_POINTS, unique_count), |(row, _)| {
            -0.4 - 0.002 * row as f64
        });
    let orbital_indices_by_kappa =
        Array2::from_shape_fn((10, state_count), |(slot, state)| (slot + state) as i64);

    let sections = apot_atomic_pots_sections(ApotAtomicPotsSectionsInput {
        unique_potential_count,
        atom_count,
        hole_index: 1,
        relaxation_energy: -1.25,
        edge_energy: 2.5,
        amplitude_reduction: 0.95,
        atomic_numbers: atomic_numbers.view(),
        model_atom_indices: model_atom_indices.view(),
        overlap_shell_counts: overlap_shell_counts.view(),
        norman_radii: norman_radii.view(),
        atom_potential_indices: atom_potential_indices.view(),
        core_hole_large_component: core_hole_large_component.view(),
        core_hole_small_component: core_hole_small_component.view(),
        core_hole_density: core_hole_density.view(),
        core_hole_coulomb_potential: core_hole_coulomb_potential.view(),
        overlap_potential_indices: overlap_potential_indices.view(),
        overlap_shell_atom_counts: overlap_shell_atom_counts.view(),
        magnetization_density: magnetization_density.view(),
        norman_valence_counts: norman_valence_counts.view(),
        atom_positions: atom_positions.view(),
        overlap_radii: overlap_radii.view(),
        overlapped_density: overlapped_density.view(),
        overlapped_valence_density: overlapped_valence_density.view(),
        overlapped_coulomb_potential: overlapped_coulomb_potential.view(),
        orbital_indices_by_kappa: orbital_indices_by_kappa.view(),
        states: &state_inputs,
    })?;

    assert_eq!(
        sections
            .iter()
            .map(|section| section.section_number)
            .collect::<Vec<_>>(),
        (1..=29).collect::<Vec<_>>()
    );

    let reparsed = parse_apot_bin(&apot_bin_string(&ApotBinData { sections })?)?;
    assert_eq!(reparsed.section_count(), 29);
    assert_eq!(reparsed.matrix_count(), 24);

    let ApotBinPayload::Records(scalars) = &section(&reparsed, 1)?.payload else {
        return invalid_apot_bin(0, "section 1 should be scalar records");
    };
    assert_eq!(scalars.rows[0][0], ApotBinValue::Int(0));
    assert_eq!(scalars.rows[0][1], ApotBinValue::Int(1));
    assert_eq!(scalars.rows[0][2], ApotBinValue::Int(1));
    assert_eq!(scalars.rows[0][5], ApotBinValue::Real(0.95));

    let ApotBinPayload::Records(unique) = &section(&reparsed, 2)?.payload else {
        return invalid_apot_bin(0, "section 2 should be unique-potential records");
    };
    assert_eq!(unique.rows[0][0], ApotBinValue::Int(29));
    assert_eq!(unique.rows[0][3], ApotBinValue::Real(2.25));

    let core_hole = apot_core_hole_columns(&reparsed)?;
    assert_close(core_hole.large_component[5], core_hole_large_component[5]);
    assert_close(core_hole.density[5], core_hole_density[5]);

    let density = real_matrix_values(&reparsed, APOT_ATOMIC_DENSITY_SECTION_NUMBER)?;
    assert_close(density[[4, 0]], initial.density_4pi[4]);
    assert_close(density[[4, 1]], final_state.density_4pi[4]);

    let dmag = real_matrix_values(&reparsed, 9)?;
    assert_eq!(dmag.dim(), (APOT_ATOMIC_RADIAL_POINTS, state_count));
    assert_close(dmag[[7, 1]], magnetization_density[[7, 1]]);

    let xnvmu = real_matrix_values(&reparsed, 12)?;
    assert_eq!(xnvmu.dim(), (4, state_count));
    assert_close(xnvmu[[3, 1]], norman_valence_counts[[3, 1]]);

    let rat = real_matrix_values(&reparsed, 15)?;
    assert_eq!(rat.dim(), (3, atom_count));
    assert_close(rat[[2, 0]], atom_positions[[2, 0]]);

    let edens = real_matrix_values(&reparsed, 17)?;
    assert_eq!(edens.dim(), (APOT_ATOMIC_RADIAL_POINTS, unique_count));
    assert_close(edens[[8, 0]], overlapped_density[[8, 0]]);

    let iorb = int_matrix_values(&reparsed, 21)?;
    assert_eq!(iorb.dim(), (10, state_count));
    assert_eq!(iorb[[9, 1]], orbital_indices_by_kappa[[9, 1]]);

    let first_dgc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START)?;
    assert_close(first_dgc[[5, 0]], initial.large_components[[5, 0]]);

    let second_adpc = real_matrix_values(&reparsed, APOT_ATOMIC_ORBITAL_SECTION_START + 7)?;
    assert_close(second_adpc[[3, 1]], final_state.small_coefficients[[3, 1]]);

    Ok(())
}

#[test]
fn rejects_bad_apot_bin_data() {
    assert!(parse_apot_bin("").is_err());
    assert!(parse_apot_bin("1 2 3\n").is_err());
    assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\n").is_err());
    assert!(
        parse_apot_bin("#SN#   Section:    1\n#DT# 2D double array with sizes    2   2\n1\n")
            .is_err()
    );
    assert!(parse_apot_bin("#SN#   Section:    1\n#DT# Int\nNaN\n").is_err());
}

#[test]
fn refreshes_core_hole_coulomb_from_section_five_density() -> Result<()> {
    let mut data = ApotBinData {
        sections: vec![sample_core_hole_section(Some(
            sample_core_hole_density().view(),
        ))?],
    };
    let expected = data.clone();
    let ApotBinPayload::Records(records) = &mut data.sections[0].payload else {
        return invalid_apot_bin(0, "sample should contain records");
    };
    records.rows[10][3] = ApotBinValue::Real(10.0);

    refresh_apot_core_hole_coulomb_payload(&mut data, 2)?;

    assert_eq!(data, expected);
    Ok(())
}

#[test]
fn extracts_core_hole_columns_from_section_five() -> Result<()> {
    let data = ApotBinData {
        sections: vec![sample_core_hole_section(Some(
            sample_core_hole_density().view(),
        ))?],
    };

    let columns = apot_core_hole_columns(&data)?;

    assert_eq!(columns.large_component.len(), APOT_CORE_HOLE_RADIAL_POINTS);
    assert_eq!(columns.small_component.len(), APOT_CORE_HOLE_RADIAL_POINTS);
    assert_eq!(columns.density[0], sample_core_hole_density()[0]);
    assert_eq!(
        columns.coulomb_potential.len(),
        APOT_CORE_HOLE_RADIAL_POINTS
    );
    Ok(())
}

#[test]
fn refreshes_nohole_core_hole_coulomb_to_zero() -> Result<()> {
    let mut data = ApotBinData {
        sections: vec![sample_core_hole_section(None)?],
    };
    let ApotBinPayload::Records(records) = &mut data.sections[0].payload else {
        return invalid_apot_bin(0, "sample should contain records");
    };
    records.rows[10][3] = ApotBinValue::Real(10.0);

    refresh_apot_core_hole_coulomb_payload(&mut data, 0)?;

    let ApotBinPayload::Records(records) = &data.sections[0].payload else {
        return invalid_apot_bin(0, "sample should contain records");
    };
    assert_eq!(records.rows[10][3], ApotBinValue::Real(0.0));
    Ok(())
}

#[test]
fn rejects_nohole_core_hole_density_when_nonzero() -> Result<()> {
    let mut data = ApotBinData {
        sections: vec![sample_core_hole_section(Some(
            sample_core_hole_density().view(),
        ))?],
    };

    let error = refresh_apot_core_hole_coulomb_payload(&mut data, 0)
        .err()
        .ok_or_else(|| crate::error::IoError::InvalidApotBin {
            line: 0,
            message: "expected nohole density validation failure".to_string(),
        })?;

    assert!(error.to_string().contains("expected zero for nohole<=0"));
    Ok(())
}

fn sample_core_hole_density() -> Array1<f64> {
    Array1::from_shape_fn(APOT_CORE_HOLE_RADIAL_POINTS, |row| {
        0.012 + 0.0002 * row as f64 + 0.001 * (-0.03 * row as f64).exp()
    })
}

fn sample_core_hole_section(drho: Option<ndarray::ArrayView1<'_, f64>>) -> Result<ApotBinSection> {
    let drho = drho
        .map(|values| values.to_owned())
        .unwrap_or_else(|| Array1::zeros(APOT_CORE_HOLE_RADIAL_POINTS));
    let dvcoul = apot_core_hole_coulomb_from_density(drho.view(), 2)?;
    Ok(ApotBinSection {
        section_number: APOT_CORE_HOLE_SECTION_NUMBER,
        headers: vec![
            "dgc0   - upper component of core hole orbital".to_string(),
            "dpc0   - lower component of core hole orbital".to_string(),
            "drho   - core hole density.".to_string(),
            "dvcoul - core hole coulomb potential.".to_string(),
        ],
        header_texts: vec![
            " dgc0   - upper component of core hole orbital".to_string(),
            " dpc0   - lower component of core hole orbital".to_string(),
            " drho   - core hole density.".to_string(),
            " dvcoul - core hole coulomb potential.".to_string(),
        ],
        column_labels: vec![
            "dgc0".to_string(),
            "dpc0".to_string(),
            "drho".to_string(),
            "dvcoul".to_string(),
        ],
        column_label_text: Some(
            "            dgc0                 dpc0                 drho               dvcoul "
                .to_string(),
        ),
        payload: ApotBinPayload::Records(ApotBinRecords {
            column_types: vec![ApotBinType::Double; 4],
            rows: (0..APOT_CORE_HOLE_RADIAL_POINTS)
                .map(|row| {
                    vec![
                        ApotBinValue::Real(0.05 + 0.001 * row as f64),
                        ApotBinValue::Real(-0.005 - 0.0001 * row as f64),
                        ApotBinValue::Real(drho[row]),
                        ApotBinValue::Real(dvcoul[row]),
                    ]
                })
                .collect(),
        }),
        trailing_headers: vec![],
        trailing_header_texts: vec![],
    })
}

struct SyntheticScfState {
    orbital_count: usize,
    density_4pi: Array1<f64>,
    coulomb_potential: Array1<f64>,
    valence_density_4pi: Array1<f64>,
    valence_occupations: Array1<f64>,
    orbital_energies: Array1<f64>,
    kappas: Array1<i32>,
    large_components: Array2<f64>,
    small_components: Array2<f64>,
    large_coefficients: Array2<f64>,
    small_coefficients: Array2<f64>,
}

impl SyntheticScfState {
    fn new(seed: f64, orbital_count: usize) -> Self {
        Self {
            orbital_count,
            density_4pi: Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| seed + row as f64),
            coulomb_potential: Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| {
                -seed - row as f64
            }),
            valence_density_4pi: Array1::from_shape_fn(APOT_ATOMIC_RADIAL_POINTS, |row| {
                0.1 * seed + 0.5 * row as f64
            }),
            valence_occupations: Array1::from_shape_fn(orbital_count, |orbital| {
                seed / 100.0 + 0.25 * orbital as f64
            }),
            orbital_energies: Array1::from_shape_fn(orbital_count, |orbital| {
                -seed - orbital as f64
            }),
            kappas: Array1::from_shape_fn(
                orbital_count,
                |orbital| {
                    if orbital % 2 == 0 { -1 } else { 1 }
                },
            ),
            large_components: Array2::from_shape_fn(
                (APOT_ATOMIC_RADIAL_POINTS, orbital_count),
                |(row, orbital)| seed * 10.0 + 10.0 * row as f64 + orbital as f64,
            ),
            small_components: Array2::from_shape_fn(
                (APOT_ATOMIC_RADIAL_POINTS, orbital_count),
                |(row, orbital)| -seed * 10.0 - 10.0 * row as f64 - orbital as f64,
            ),
            large_coefficients: Array2::from_shape_fn(
                (APOT_ATOMIC_COEFFICIENTS, orbital_count),
                |(row, orbital)| seed + 0.01 * row as f64 + orbital as f64,
            ),
            small_coefficients: Array2::from_shape_fn(
                (APOT_ATOMIC_COEFFICIENTS, orbital_count),
                |(row, orbital)| -seed - 0.02 * row as f64 - orbital as f64,
            ),
        }
    }

    fn input(&self, state_count: usize, state_index: usize) -> ApotAtomicScfStateSectionsInput<'_> {
        ApotAtomicScfStateSectionsInput {
            state_count,
            state_index,
            orbital_count: self.orbital_count,
            density_4pi: self.density_4pi.view(),
            coulomb_potential: self.coulomb_potential.view(),
            valence_density_4pi: self.valence_density_4pi.view(),
            valence_occupations: self.valence_occupations.view(),
            orbital_energies: self.orbital_energies.view(),
            kappas: self.kappas.view(),
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            large_coefficients: self.large_coefficients.view(),
            small_coefficients: self.small_coefficients.view(),
        }
    }
}

fn section(data: &ApotBinData, section_number: usize) -> Result<&ApotBinSection> {
    data.sections
        .iter()
        .find(|section| section.section_number == section_number)
        .ok_or_else(|| crate::error::IoError::InvalidApotBin {
            line: 0,
            message: format!("missing section {section_number}"),
        })
}

fn real_matrix_values(data: &ApotBinData, section_number: usize) -> Result<&Array2<f64>> {
    let Some(matrix) = section(data, section_number)?.matrix() else {
        return invalid_apot_bin(0, format!("section {section_number} should be a matrix"));
    };
    match &matrix.values {
        ApotBinMatrixValues::Real(values) => Ok(values),
        _ => invalid_apot_bin(0, format!("section {section_number} should be real-valued")),
    }
}

fn int_matrix_values(data: &ApotBinData, section_number: usize) -> Result<&Array2<i64>> {
    let Some(matrix) = section(data, section_number)?.matrix() else {
        return invalid_apot_bin(0, format!("section {section_number} should be a matrix"));
    };
    match &matrix.values {
        ApotBinMatrixValues::Int(values) => Ok(values),
        _ => invalid_apot_bin(
            0,
            format!("section {section_number} should be integer-valued"),
        ),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual={actual} expected={expected}"
    );
}

const APOT_BIN: &str = r#"#SN#   Section:    1
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Int Int Double
#H# first section
#CL# nph nat ihole s02
     1         79          1     0.9500000000E+00
#SN#   Section:    2
#DF# This section written in TXT .
#H#
#H# The following data types are written in this section.
#DT#  Int Double
    29     0.2838535628D+01
    30     0.2632330371E+01
#SN#   Section:    3
#DF# This section written in TXT .
#H#
#DT# 2D double array with sizes    2   3
#H# File is organized as follows:  Array(1,i)     Array(1,i+1)    Array(1,i+2)  . . .
#H#                                Array(2,i)
#H#                                     .
#H#                                     .
#H#                                     .
#H# matrix
1.0000000000E+00    2.0000000000E+00    3.0000000000E+00
4.0000000000E+00    5.0000000000E+00    6.0000000000E+00
#H# next block
"#;
