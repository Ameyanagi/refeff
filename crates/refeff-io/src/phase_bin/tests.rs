use super::*;

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, Array4, arr2};
use num_complex::Complex64;
use refeff_core::{
    Complex, FEFF_HARTREE_EV, GenfmtDriverSetupInput, GenfmtJasDriverSetupInput,
    GenfmtJasTransitionMatrices, GenfmtLegendreNormalizationInput, GenfmtNStarRowsInput,
    TransitionBMatrixInput, genfmt_driver_setup, genfmt_jas_driver_setup,
    genfmt_legendre_normalization_table, genfmt_nstar_rows, transition_b_matrix,
};

use crate::error::{IoError, Result};
use crate::{
    BandEnergyMesh, BandInput, CfAverage, GenfmtControl, GenfmtInput, GlobalControl, GlobalInput,
    GlobalNorms, GlobalQControl, PathsDatGenfmtPath,
};

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
fn builds_genfmt_phase_tables_from_phase_bin() -> Result<()> {
    let data = sample_phase_bin_data();
    let genfmt = phase_bin_genfmt_data(&data)?;

    assert_eq!(genfmt.signed_angular_offset, 2);
    assert_eq!(genfmt.angular_limits, arr2(&[[1, 2], [1, 2], [1, 2]]));
    assert_eq!(genfmt.spin_phase_shifts.dim(), (3, 5, 2, 2));
    assert_eq!(genfmt.potential_labels, ["Cu", "O"]);
    assert_eq!(genfmt.potential_label_refs(), ["Cu", "O"]);
    assert_eq!(genfmt.atomic_numbers, Array1::from_vec(vec![29, 8]));

    assert_eq!(
        genfmt.spin_phase_shifts[(1, 1, 1, 0)],
        data.potentials[0].phase_shifts[(1, 0, 1)]
    );
    assert_eq!(
        genfmt.spin_phase_shifts[(1, 3, 1, 0)],
        data.potentials[0].phase_shifts[(1, 2, 1)]
    );
    assert_eq!(
        genfmt.spin_phase_shifts[(1, 0, 1, 0)],
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        genfmt.spin_phase_shifts[(1, 4, 1, 0)],
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        genfmt.spin_phase_shifts[(2, 4, 0, 1)],
        data.potentials[1].phase_shifts[(2, 4, 0)]
    );
    Ok(())
}

#[test]
fn builds_band_phase_tables_from_phase_bin() -> Result<()> {
    let data = sample_phase_bin_data();
    let band = phase_bin_band_data(&data)?;

    assert_eq!(band, data.to_band_data()?);
    assert_eq!(band.signed_angular_offset, 2);
    assert_eq!(band.energies_hartree, data.energy_grid);
    assert_eq!(band.reference_energies_hartree, data.reference_energy);
    assert_eq!(band.spin_phase_shifts.dim(), (3, 5, 2, 2));
    assert_eq!(band.potential_lmax, vec![1, 2]);
    assert_eq!(band.potential_labels, ["Cu", "O"]);

    assert_eq!(
        band.spin_phase_shifts[(1, 1, 1, 0)],
        data.potentials[0].phase_shifts[(1, 0, 1)]
    );
    assert_eq!(
        band.spin_phase_shifts[(1, 3, 1, 0)],
        data.potentials[0].phase_shifts[(1, 2, 1)]
    );
    assert_eq!(
        band.spin_phase_shifts[(1, 0, 1, 0)],
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        band.spin_phase_shifts[(1, 4, 1, 0)],
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        band.spin_phase_shifts[(2, 4, 0, 1)],
        data.potentials[1].phase_shifts[(2, 4, 0)]
    );
    Ok(())
}

#[test]
fn builds_band_search_setup_from_phase_handoffs() -> Result<()> {
    let phase = sample_band_phase_bin_data();
    let band = sample_band_input();

    let setup = band_search_setup_from_handoffs(&band, &phase)?;

    assert_eq!(setup, phase.to_band_search_setup(&band)?);
    assert_eq!(setup.phase_handoff.potential_lmax, vec![1, 2]);
    assert_eq!(setup.energy_mesh.point_count(), 3);
    assert_close_reals(
        setup.energy_mesh.energies_hartree.iter().copied(),
        [1.0, 2.0, 3.0],
    );
    assert_eq!(
        setup.phase_interpolation.reference_energies_hartree.dim(),
        (3, 2)
    );
    assert_eq!(setup.phase_interpolation.phase_shifts.dim(), (3, 2, 5, 2));

    let phase_value = setup.phase_interpolation.phase_shifts[(1, 1, 4, 1)];
    assert!((phase_value.re - 0.62_f32).abs() <= 1.0e-6);
    assert!((phase_value.im - 0.001_f32).abs() <= 1.0e-6);
    assert_eq!(
        setup.phase_interpolation.phase_shifts[(1, 0, 0, 0)],
        num_complex::Complex32::new(0.0, 0.0)
    );
    assert_eq!(
        setup.phase_interpolation.phase_shifts[(1, 0, 4, 0)],
        num_complex::Complex32::new(0.0, 0.0)
    );
    Ok(())
}

#[test]
fn band_search_setup_rejects_invalid_active_prefix() {
    let mut phase = sample_band_phase_bin_data();
    phase.main_energy_count = 0;
    let band = sample_band_input();

    assert!(matches!(
        band_search_setup_from_handoffs(&band, &phase),
        Err(IoError::InvalidPhaseBin { field: "ne1", .. })
    ));
}

#[test]
fn builds_path_phase_handoff_from_phase_bin() -> Result<()> {
    let data = sample_phase_bin_data();
    let handoff = phase_bin_path_handoff_from_phase_bin(&data)?;

    assert_eq!(handoff, data.to_path_handoff()?);
    assert_eq!(handoff.energy_count(), data.energy_count);
    assert_eq!(handoff.potential_count(), data.potential_count());
    assert_eq!(handoff.potential_labels, ["Cu", "O"]);
    assert_eq!(handoff.output_energy_count, data.main_energy_count);
    assert_eq!(handoff.zero_wave_energy_index, 1);
    assert_eq!(handoff.angular_limits, arr2(&[[1, 2], [1, 2], [1, 2]]));
    assert_eq!(handoff.phase_shifts.dim(), (3, 3, 2));
    assert_eq!(
        handoff.phase_shifts[(1, 0, 0)],
        data.potentials[0].phase_shifts[(1, 1, 0)]
    );
    assert_eq!(
        handoff.phase_shifts[(1, 1, 0)],
        data.potentials[0].phase_shifts[(1, 0, 0)]
    );
    assert_eq!(
        handoff.phase_shifts[(2, 2, 1)],
        data.potentials[1].phase_shifts[(2, 0, 0)]
    );
    assert_eq!(handoff.reference_energies[2], data.reference_energy[(2, 0)]);

    let tables = handoff.criteria_tables()?;
    assert_eq!(tables.output_energy_count, data.main_energy_count);
    assert_eq!(tables.zero_wave_energy_index, 1);
    assert_eq!(tables.fbeta.shape(), &[81, 2, 3]);
    assert_eq!(tables.critical_energy_indices, vec![1]);
    assert_eq!(tables.fbeta_critical.shape(), &[81, 2, 1]);
    assert_eq!(path_phase_criteria_tables_from_phase_bin(&data)?, tables);
    Ok(())
}

#[test]
fn path_phase_handoff_rejects_invalid_zero_wave_index() {
    let mut data = sample_phase_bin_data();
    data.fermi_index = 0;

    assert!(matches!(
        phase_bin_path_handoff_from_phase_bin(&data),
        Err(IoError::InvalidPhaseBin { field: "ik0", .. })
    ));
}

#[test]
fn genfmt_phase_tables_feed_core_driver_setup() -> Result<()> {
    let phase = sample_phase_bin_data();
    let genfmt = phase.to_genfmt_data()?;
    let potential_labels = genfmt.potential_label_refs();

    let ordinary = genfmt_driver_setup(GenfmtDriverSetupInput {
        version: "refeff-test",
        pad_width: phase.pad_width,
        core_hole: phase.ihole,
        order: 2,
        average_norman_radius: phase.scalars.average_norman_radius,
        fermi_level: phase.scalars.fermi_level,
        edge_energy: phase.scalars.edge_energy,
        spin_selector: 1,
        available_spin_channels: phase.spin_count,
        energies: phase.energy_grid.view(),
        spin_reference_energies: phase.reference_energy.view(),
        spin_phase_shifts: genfmt.spin_phase_shifts.view(),
        angular_limits: genfmt.angular_limits.view(),
        signed_angular_offset: genfmt.signed_angular_offset,
        initial_orbital_l: 0,
        initial_kappa: -1,
        potential_labels: &potential_labels,
        atomic_numbers: genfmt.atomic_numbers.view(),
    })
    .expect("ordinary GENFMT driver setup");

    assert_eq!(ordinary.header.pad_width, phase.pad_width);
    assert_eq!(
        ordinary.header.average_norman_radius,
        phase.scalars.average_norman_radius
    );
    assert_eq!(ordinary.header.potentials[0].label, "Cu");
    assert_eq!(ordinary.header.potentials[1].atomic_number, 8);

    let jas = genfmt_jas_driver_setup(GenfmtJasDriverSetupInput {
        version: "refeff-test",
        pad_width: phase.pad_width,
        core_hole: phase.ihole,
        order: 2,
        average_norman_radius: phase.scalars.average_norman_radius,
        fermi_level: phase.scalars.fermi_level,
        edge_energy: phase.scalars.edge_energy,
        spin_selector: 1,
        available_spin_channels: phase.spin_count,
        energies: phase.energy_grid.view(),
        spin_reference_energies: phase.reference_energy.view(),
        spin_phase_shifts: genfmt.spin_phase_shifts.view(),
        spin_radial_factors: phase.transition_moments.view(),
        angular_limits: genfmt.angular_limits.view(),
        signed_angular_offset: genfmt.signed_angular_offset,
        initial_orbital_l: 0,
        initial_kappa: -1,
        potential_labels: &potential_labels,
        atomic_numbers: genfmt.atomic_numbers.view(),
    })
    .expect("GENFMTJAS driver setup");

    assert_eq!(jas.header.potentials[1].label, "O");
    assert_eq!(
        jas.radial_factors.radial_factors.dim(),
        (phase.energy_count, phase.q_count, phase.transition_count)
    );
    Ok(())
}

#[test]
fn builds_core_driver_setup_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let global = sample_global_input();

    let ordinary = genfmt_driver_setup_from_handoffs("refeff-test", &genfmt, &global, &phase)?;

    assert_eq!(ordinary.spin_channel_count, phase.spin_count);
    assert_eq!(ordinary.header.core_hole, phase.ihole);
    assert_eq!(ordinary.header.order, genfmt.control.iorder);
    assert_eq!(ordinary.header.initial_angular_momentum, 2);
    assert_eq!(
        ordinary.header.average_norman_radius,
        phase.scalars.average_norman_radius
    );
    assert_eq!(ordinary.header.potentials[0].label, "Cu");

    let jas = genfmt_jas_driver_setup_from_handoffs("refeff-test", &genfmt, &global, &phase)?;

    assert_eq!(jas.spin_selection.spin_index, 1);
    assert_eq!(jas.header.order, genfmt.control.iorder);
    assert_eq!(jas.header.initial_angular_momentum, 2);
    assert_eq!(
        jas.radial_factors.radial_factors.dim(),
        (phase.energy_count, phase.q_count, phase.transition_count)
    );
    Ok(())
}

#[test]
fn builds_core_path_setups_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let global = sample_global_input();
    let paths = vec![sample_genfmt_path()];

    let ordinary = genfmt_ordinary_path_setups_from_handoffs(&genfmt, &global, &phase, &paths)?;

    assert_eq!(ordinary.len(), 1);
    assert_eq!(ordinary[0].rotations.real_leg_count, paths[0].leg_count());
    assert_eq!(ordinary[0].lambda.order, 6);
    assert_eq!(ordinary[0].lambda.max_m_plus_one, 3);
    assert!(
        ordinary[0]
            .rotations
            .polarized_extra_rotation()
            .expect("ordinary polarized pseudo-leg")
            .is_some()
    );

    let jas = genfmt_jas_path_setups_from_handoffs(&genfmt, &global, &phase, &paths)?;

    assert_eq!(jas.len(), 1);
    assert_eq!(jas[0].rotations.real_leg_count, paths[0].leg_count());
    assert_eq!(jas[0].lambda.order, ordinary[0].lambda.order);
    assert!(
        jas[0]
            .rotations
            .polarized_extra_rotation()
            .expect("JAS polarized pseudo-leg")
            .is_some()
    );
    Ok(())
}

#[test]
fn builds_jas_transition_indices_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_jas_phase_bin_data();
    let mut global = sample_global_input();
    global.control.do_nrixs = 1;
    global.control.le2 = 0;

    let indices = genfmt_jas_transition_indices_from_handoffs(&global, &phase)?;

    assert_eq!(indices.initial_j2, 3);
    assert_eq!(indices.final_j2_max, 3);
    assert_eq!(indices.final_lj_max, 0);
    assert_eq!(indices.final_state_capacity, 40);
    assert_eq!(indices.transitions.len(), 1);
    assert_eq!(indices.transitions[0].final_state_kappa, -2);
    assert_eq!(indices.transitions[0].orbital_angular_momentum, 1);
    Ok(())
}

#[test]
fn rejects_mismatched_jas_transition_index_handoff() {
    let phase = sample_phase_bin_data();
    let mut global = sample_global_input();
    global.control.do_nrixs = 1;
    global.control.le2 = 0;

    assert!(matches!(
        genfmt_jas_transition_indices_from_handoffs(&global, &phase),
        Err(IoError::InvalidPhaseBin {
            field: "indmax",
            ..
        })
    ));
}

#[test]
fn builds_jas_transition_setups_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_jas_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let mut global = sample_global_input();
    global.control.do_nrixs = 1;
    global.control.le2 = 0;
    global.control.l2lp = 1;
    let paths = vec![sample_genfmt_path()];
    let path_setups = genfmt_jas_path_setups_from_handoffs(&genfmt, &global, &phase, &paths)?;

    let transition_setups =
        genfmt_jas_transition_setups_from_handoff_setups(&global, &phase, &path_setups)?;

    assert_eq!(transition_setups.len(), 1);
    assert_eq!(transition_setups[0].effective_initial_j.initial_j2, 3);
    assert_eq!(transition_setups[0].transition_count.transition_count, 1);
    match &transition_setups[0].matrices {
        GenfmtJasTransitionMatrices::LeftRight(matrices) => {
            assert_eq!(matrices.left_matrix.shape(), &[4, 5, 1, 1]);
            assert_eq!(matrices.right_matrix.shape(), &[4, 5, 1, 1]);
            assert_eq!(matrices.generated_final_j2, vec![3]);
        }
        GenfmtJasTransitionMatrices::Spherical(_) => {
            panic!("expected q-resolved GENFMTJAS transition matrices")
        }
    }
    Ok(())
}

#[test]
fn rejects_unpolarized_jas_transition_setups_from_handoffs() -> Result<()> {
    let phase = sample_jas_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let mut global = sample_global_input();
    global.control.do_nrixs = 1;
    global.control.ipol = 0;
    global.control.l2lp = 1;
    let paths = vec![sample_genfmt_path()];
    let path_setups = genfmt_jas_path_setups_from_handoffs(&genfmt, &global, &phase, &paths)?;

    assert!(matches!(
        genfmt_jas_transition_setups_from_handoff_setups(&global, &phase, &path_setups),
        Err(IoError::InvalidPhaseBin { field: "ipol", .. })
    ));
    Ok(())
}

#[test]
fn builds_nstar_rows_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let mut global = sample_global_input();
    global.control.elpty = 0.6;
    let paths = vec![sample_nstar_genfmt_path()];
    let path_inputs = crate::genfmt_nstar_path_inputs(&paths);

    let rows =
        genfmt_nstar_rows_from_handoffs(&genfmt, &global, &phase, &paths)?.expect("nstar rows");
    let expected = genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: global.evec,
        ellipticity_vector: global.xivec,
        initial_l: 2,
        ellipticity: global.control.elpty,
        path_inputs: &path_inputs,
    })
    .expect("expected ordinary nstar rows");

    assert_eq!(rows, expected);
    assert_eq!(rows.rows.len(), 1);

    let mut jas_global = global;
    jas_global.control.do_nrixs = 1;
    let jas_rows = genfmt_nstar_rows_from_handoffs(&genfmt, &jas_global, &phase, &paths)?
        .expect("JAS nstar rows");
    let expected_jas = genfmt_nstar_rows(GenfmtNStarRowsInput {
        primary_polarization: jas_global.evec,
        ellipticity_vector: jas_global.xivec,
        initial_l: 2,
        ellipticity: 0.0,
        path_inputs: &path_inputs,
    })
    .expect("expected JAS nstar rows");

    assert_eq!(jas_rows, expected_jas);
    assert_ne!(jas_rows, rows);
    Ok(())
}

#[test]
fn builds_ordinary_transition_b_matrix_from_genfmt_handoff_files() -> Result<()> {
    let phase = sample_phase_bin_data();
    let mut global = sample_global_input();
    global.control.ipol = 0;

    let matrix = genfmt_ordinary_transition_b_matrix_from_handoffs(&global, &phase)?;
    let expected = transition_b_matrix(TransitionBMatrixInput {
        lmax: 2,
        initial_kappa: -2,
        polarization: 0,
        polarization_tensor: [[Complex::new(0.0, 0.0); 3]; 3],
        multipole: global.control.le2,
        trace_orbital: false,
        spin: global.control.ispin,
        spin_channels: phase.spin_count,
        spin_vector_angle: global.control.angks,
    })
    .expect("expected ordinary transition B matrix");

    assert_eq!(matrix, expected);
    assert_eq!(matrix.l_offset, 2);
    assert_eq!(matrix.matrix.shape(), &[5, 2, 8, 5, 2, 8]);
    assert_eq!(matrix.orbital_momenta[0..3], [2, 2, 0]);
    Ok(())
}

#[test]
fn builds_ordinary_transition_matrices_from_handoff_setups() -> Result<()> {
    let phase = sample_phase_bin_data();
    let genfmt = sample_genfmt_input();
    let mut global = sample_global_input();
    global.control.ipol = 0;
    let paths = vec![sample_genfmt_path()];
    let path_setups = genfmt_ordinary_path_setups_from_handoffs(&genfmt, &global, &phase, &paths)?;
    let transition_b_matrix = genfmt_ordinary_transition_b_matrix_from_handoffs(&global, &phase)?;

    let matrices = genfmt_ordinary_transition_matrices_from_handoff_setups(
        &global,
        &phase,
        &path_setups,
        &transition_b_matrix,
    )?;

    assert_eq!(matrices.len(), 1);
    assert_eq!(matrices[0].matrices.shape(), &[2, 9, 8, 9, 8]);
    assert_eq!(matrices[0].b_matrix_spin_indices, vec![1, 1]);
    assert_ne!(
        matrices[0].matrices[(0, 4, 0, 4, 0)],
        Complex::new(0.0, 0.0)
    );

    let mut single_spin_global = global.clone();
    single_spin_global.control.ispin = 2;
    let transition_b_matrix =
        genfmt_ordinary_transition_b_matrix_from_handoffs(&single_spin_global, &phase)?;
    let matrices = genfmt_ordinary_transition_matrices_from_handoff_setups(
        &single_spin_global,
        &phase,
        &path_setups,
        &transition_b_matrix,
    )?;
    assert_eq!(matrices[0].matrices.shape(), &[1, 9, 8, 9, 8]);
    assert_eq!(matrices[0].b_matrix_spin_indices, vec![0]);
    Ok(())
}

#[test]
fn extracts_ordinary_genfmt_spin_radial_factors_from_phase_bin() -> Result<()> {
    let phase = legacy_phase_bin_data();
    let factors = genfmt_ordinary_spin_radial_factors_from_phase(&phase)?;

    assert_eq!(
        factors.dim(),
        (
            phase.energy_count,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            phase.spin_count
        )
    );
    assert_eq!(factors[(1, 3, 1)], phase.transition_moments[(1, 0, 3, 1)]);
    Ok(())
}

#[test]
fn extracts_rhorrp_phase_table_from_phase_bin() -> Result<()> {
    let phase = sample_phase_bin_data();
    let table = rhorrp_phase_table_from_phase_bin(&phase, 0)?;

    assert_eq!(
        table.dim(),
        (phase.energy_count, 3, phase.potential_count())
    );
    assert_eq!(
        table[(1, 0, 0)],
        phase.potentials[0].phase_shifts[(1, 1, 0)]
    );
    assert_eq!(
        table[(1, 1, 0)],
        phase.potentials[0].phase_shifts[(1, 2, 0)]
    );
    assert_eq!(table[(1, 2, 0)], Complex64::new(0.0, 0.0));
    assert_eq!(
        table[(2, 2, 1)],
        phase.potentials[1].phase_shifts[(2, 4, 0)]
    );
    Ok(())
}

#[test]
fn rhorrp_phase_table_averages_active_spin_channels() -> Result<()> {
    let phase = sample_phase_bin_data();
    let table = rhorrp_phase_table_from_phase_bin(&phase, 1)?;
    let first_spin = phase.potentials[1].phase_shifts[(2, 3, 0)];
    let last_spin = phase.potentials[1].phase_shifts[(2, 3, 1)];

    assert_eq!(table[(2, 1, 1)], (first_spin + last_spin) * 0.5);
    Ok(())
}

#[test]
fn extracts_rhorrp_phase_handoff_from_phase_bin() -> Result<()> {
    let phase = sample_phase_bin_data();
    let handoff = rhorrp_phase_handoff_from_phase_bin(&phase, 0)?;

    assert_eq!(handoff, phase.to_rhorrp_handoff(0)?);
    assert_eq!(handoff.energy_count(), phase.energy_count);
    assert_eq!(handoff.real_axis_count, phase.main_energy_count);
    assert_eq!(
        handoff.chemical_potential_hartree,
        phase.scalars.fermi_level
    );
    assert_eq!(handoff.angular_momentum_count(), 3);
    assert_eq!(handoff.potential_count(), phase.potential_count());
    assert_eq!(handoff.energies_hartree, phase.energy_grid);
    assert_eq!(
        handoff.xsph_phase_shifts[(1, 1, 0)],
        phase.potentials[0].phase_shifts[(1, 2, 0)]
    );
    Ok(())
}

#[test]
fn rhorrp_phase_handoff_averages_active_spin_channels() -> Result<()> {
    let phase = sample_phase_bin_data();
    let handoff = rhorrp_phase_handoff_from_phase_bin(&phase, 1)?;
    let first_spin = phase.potentials[1].phase_shifts[(2, 3, 0)];
    let last_spin = phase.potentials[1].phase_shifts[(2, 3, 1)];

    assert_eq!(
        handoff.xsph_phase_shifts[(2, 1, 1)],
        (first_spin + last_spin) * 0.5
    );
    Ok(())
}

#[test]
fn builds_rixs_handoff_from_phase_bin() -> Result<()> {
    let mut phase = legacy_phase_bin_data();
    let oxygen_lmax = phase.potentials[1].lmax;
    for spin in 0..phase.spin_count {
        phase.potentials[1].phase_shifts[(0, oxygen_lmax + 2, spin)] = Complex64::new(0.0, 0.0);
    }
    let copper_lmax = phase.potentials[0].lmax;
    for spin in 0..phase.spin_count {
        phase.potentials[0].phase_shifts[(2, copper_lmax + 1, spin)] = Complex64::new(0.0, 0.0);
    }

    let handoff = phase_bin_rixs_handoff_from_phase_bin(&phase)?;

    assert_eq!(handoff, phase.to_rixs_handoff()?);
    assert_eq!(handoff.spin_count, phase.spin_count);
    assert_eq!(handoff.energy_count, phase.energy_count);
    assert_eq!(handoff.main_energy_count, phase.main_energy_count);
    assert_eq!(handoff.auxiliary_energy_count, phase.auxiliary_energy_count);
    assert_eq!(handoff.ihole, phase.ihole);
    assert_eq!(handoff.fermi_index, phase.fermi_index);
    assert_eq!(handoff.scalars, phase.scalars);
    assert_eq!(handoff.energy_grid, phase.energy_grid);
    assert_eq!(handoff.reference_energy, phase.reference_energy);
    assert_eq!(handoff.potentials, phase.potentials);
    assert_eq!(handoff.potential_count(), phase.potential_count());
    assert_eq!(handoff.angular_limits, arr2(&[[1, 1], [1, 2], [0, 2]]));
    assert_eq!(handoff.max_angular_limit_plus_one, 3);
    assert_eq!(
        rixs_angular_limits_from_phase_bin(&phase)?,
        handoff.angular_limits
    );
    assert_eq!(
        handoff.transition_moments.dim(),
        (
            phase.energy_count,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            phase.spin_count
        )
    );
    assert_eq!(
        handoff.transition_moments[(1, 3, 1)],
        phase.transition_moments[(1, 0, 3, 1)]
    );
    assert_eq!(
        rixs_transition_moments_from_phase_bin(&phase)?,
        handoff.transition_moments
    );
    Ok(())
}

#[test]
fn builds_rixs_transition_setup_from_global_and_phase_handoffs() -> Result<()> {
    let phase = phase_bin_rixs_handoff_from_phase_bin(&legacy_phase_bin_data())?;
    let mut global = sample_global_input();
    global.polarization_tensor = [
        [0.20, -0.05, -0.10, 0.04, 0.03, 0.02],
        [0.11, -0.07, 0.50, 0.00, -0.08, 0.09],
        [0.06, 0.01, 0.13, -0.02, 0.17, 0.03],
    ];

    let setup = phase_bin_rixs_transition_setup_from_handoffs(&global, &phase)?;

    assert_eq!(setup.initial_angular_momentum, 1);
    assert_eq!(
        setup.b_matrix_diagonal.dim(),
        (
            phase.max_angular_limit_plus_one * phase.max_angular_limit_plus_one,
            PHASE_BIN_DEFAULT_TRANSITION_COUNT,
            phase.spin_count,
        )
    );
    assert_eq!(
        setup.transition_angular_momenta.len(),
        PHASE_BIN_DEFAULT_TRANSITION_COUNT
    );
    assert!(
        setup
            .transition_angular_momenta
            .iter()
            .any(|&angular_momentum| angular_momentum >= 0)
    );
    Ok(())
}

#[test]
fn selects_rixs_transition_phase_shifts_from_phase_handoff() -> Result<()> {
    let mut phase_data = legacy_phase_bin_data();
    phase_data.potentials[0] = sample_potential(
        2,
        29,
        "Cu",
        phase_data.energy_count,
        phase_data.spin_count,
        0.1,
    );
    let lmax = phase_data.potentials[0].lmax;
    phase_data.potentials[0].phase_shifts[(0, lmax, 0)] = Complex64::new(1.0, -0.1);
    phase_data.potentials[0].phase_shifts[(0, lmax - 1, 0)] = Complex64::new(2.0, -0.2);
    phase_data.potentials[0].phase_shifts[(0, lmax - 2, 0)] = Complex64::new(3.0, -0.3);

    let phase = phase_bin_rixs_handoff_from_phase_bin(&phase_data)?;
    let mut global = sample_global_input();
    global.polarization_tensor = [
        [0.20, -0.05, -0.10, 0.04, 0.03, 0.02],
        [0.11, -0.07, 0.50, 0.00, -0.08, 0.09],
        [0.06, 0.01, 0.13, -0.02, 0.17, 0.03],
    ];
    let setup = phase_bin_rixs_transition_setup_from_handoffs(&global, &phase)?;

    let selected = phase_bin_rixs_transition_phase_shifts_from_handoff(&phase, &setup)?;

    assert_eq!(
        selected.dim(),
        (phase.energy_count, setup.transition_angular_momenta.len())
    );
    let mut checked = 0;
    for (transition, &angular_momentum) in setup.transition_angular_momenta.iter().enumerate() {
        if let Ok(angular_momentum) = usize::try_from(angular_momentum)
            && angular_momentum <= lmax
        {
            let expected = phase_data.potentials[0].phase_shifts[(0, lmax - angular_momentum, 0)];
            assert_eq!(selected[(0, transition)], expected);
            checked += 1;
        }
    }
    assert!(checked > 0);
    Ok(())
}

#[test]
fn rixs_handoff_rejects_short_transition_table() {
    let phase = sample_phase_bin_data();

    assert!(matches!(
        phase_bin_rixs_handoff_from_phase_bin(&phase),
        Err(IoError::InvalidPhaseBin { field: "rkk", .. })
    ));
}

#[test]
fn rixs_handoff_matches_feff_rdxsphrxs_reference_phase_bin() -> Result<()> {
    let Some(path) = reference_exafs_cu_phase_bin() else {
        return Ok(());
    };
    let phase = read_phase_bin(path)?;
    let handoff = phase_bin_rixs_handoff_from_phase_bin(&phase)?;

    assert_eq!(handoff.energy_count, 80);
    assert_eq!(handoff.main_energy_count, 59);
    assert_eq!(handoff.auxiliary_energy_count, 0);
    assert_eq!(handoff.potential_count(), 2);
    assert_eq!(handoff.ihole, 1);
    assert_eq!(handoff.fermi_index, 1);
    assert_eq!(handoff.max_angular_limit_plus_one, 21);
    assert_close_reals(
        handoff.scalars.as_array(),
        [
            2.635_147_358_699_168,
            -0.138_801_301_525_098_76,
            -0.138_801_301_525_098_76,
        ],
    );
    assert_eq!(
        (0..8)
            .map(|energy| handoff.angular_limits[(energy, 0)])
            .collect::<Vec<_>>(),
        vec![5, 5, 5, 5, 5, 5, 5, 5]
    );
    assert_eq!(
        (0..8)
            .map(|energy| handoff.angular_limits[(energy, 1)])
            .collect::<Vec<_>>(),
        vec![5, 5, 5, 5, 5, 5, 6, 6]
    );
    assert_close_complex(
        [
            handoff.energy_grid[0],
            handoff.reference_energy[(0, 0)],
            handoff.transition_moments[(0, 0, 0)],
            handoff.transition_moments[(1, 2, 0)],
        ],
        [
            Complex64::new(-0.138_801_301_525_098_76, 0.031_773_268_817_226_29),
            Complex64::new(-0.603_644_561_767_596_7, 0.0),
            Complex64::new(-0.560_537_151_799_813, 1.290_526_357_547_372_7),
            Complex64::new(0.0, 0.0),
        ],
    );
    Ok(())
}

#[test]
fn rejects_inconsistent_rhorrp_phase_handoff_energy_counts() {
    let mut short_real_axis = sample_phase_bin_data();
    short_real_axis.main_energy_count = 1;
    assert!(matches!(
        rhorrp_phase_handoff_from_phase_bin(&short_real_axis, 0),
        Err(IoError::InvalidPhaseBin { field: "ne1", .. })
    ));

    let mut overflowing_auxiliary = sample_phase_bin_data();
    overflowing_auxiliary.main_energy_count = overflowing_auxiliary.energy_count;
    assert!(matches!(
        rhorrp_phase_handoff_from_phase_bin(&overflowing_auxiliary, 0),
        Err(IoError::InvalidPhaseBin { field: "ne3", .. })
    ));
}

#[test]
fn rejects_rhorrp_phase_table_with_invalid_spin_count() {
    let mut phase = sample_phase_bin_data();
    phase.spin_count = 3;
    phase.reference_energy = Array2::from_shape_fn((phase.energy_count, 3), |(energy, spin)| {
        Complex64::new(-1.0 + energy as f64 * 0.2, 0.05 * spin as f64)
    });
    for potential in &mut phase.potentials {
        let l_count = 2 * potential.lmax + 1;
        potential.phase_shifts = Array3::from_shape_fn(
            (phase.energy_count, l_count, 3),
            |(energy, l_slot, spin)| {
                Complex64::new(
                    0.1 + 0.01 * energy as f64 + 0.1 * l_slot as f64,
                    0.001 * spin as f64,
                )
            },
        );
    }
    phase.transition_moments = Array4::from_shape_fn(
        (phase.energy_count, phase.q_count, phase.transition_count, 3),
        |(energy, q_index, transition, spin)| {
            Complex64::new(
                0.01 * (energy + 1) as f64 + 0.1 * q_index as f64 + transition as f64,
                -0.02 * spin as f64,
            )
        },
    );

    assert!(matches!(
        rhorrp_phase_table_from_phase_bin(&phase, 1),
        Err(IoError::InvalidPhaseBin {
            field: "rhorrp_phase",
            ..
        })
    ));
}

#[test]
fn rejects_short_ordinary_genfmt_spin_radial_factor_handoff() {
    let phase = sample_phase_bin_data();

    assert!(matches!(
        genfmt_ordinary_spin_radial_factors_from_phase(&phase),
        Err(IoError::InvalidPhaseBin {
            field: "indmax",
            ..
        })
    ));
}

#[test]
fn converts_genfmt_edge_start_index_from_phase_bin() -> Result<()> {
    let mut phase = sample_phase_bin_data();
    phase.fermi_index = 1;

    assert_eq!(genfmt_edge_start_index_from_phase(&phase)?, 0);

    let mut bad = phase;
    bad.fermi_index = 0;
    assert!(matches!(
        genfmt_edge_start_index_from_phase(&bad),
        Err(IoError::InvalidPhaseBin { field: "ik0", .. })
    ));
    Ok(())
}

#[test]
fn builds_legendre_normalization_from_genfmt_dimensions() -> Result<()> {
    let table = genfmt_legendre_normalization_from_feff_dims()?;
    let expected = genfmt_legendre_normalization_table(GenfmtLegendreNormalizationInput {
        lmaxp1: 25,
        mmaxp1: 25,
    })
    .expect("expected FEFF snlm table");

    assert_eq!(table, expected);
    assert_eq!(table.shape(), &[25, 25]);
    assert_eq!(table[(0, 0)], 1.0);
    assert_eq!(table[(0, 1)], 0.0);
    Ok(())
}

#[test]
fn builds_core_legendre_normalization_from_genfmt_dimensions() -> Result<()> {
    let table = genfmt_core_legendre_normalization_from_feff_dims()?;
    let snlm = genfmt_legendre_normalization_from_feff_dims()?;

    assert_eq!(table.shape(), &[25, 25]);
    assert_eq!(table[(0, 0)], snlm[(0, 0)]);
    assert_eq!(table[(0, 1)], snlm[(1, 0)]);
    assert_eq!(table[(1, 1)], snlm[(1, 1)]);
    Ok(())
}

#[test]
fn rejects_invalid_core_hole_for_genfmt_handoff_setup() {
    let mut phase = sample_phase_bin_data();
    phase.ihole = 31;
    let genfmt = sample_genfmt_input();
    let global = sample_global_input();

    assert!(matches!(
        genfmt_driver_setup_from_handoffs("refeff-test", &genfmt, &global, &phase),
        Err(IoError::InvalidPhaseBin { field: "ihole", .. })
    ));
    assert!(matches!(
        genfmt_jas_driver_setup_from_handoffs("refeff-test", &genfmt, &global, &phase),
        Err(IoError::InvalidPhaseBin { field: "ihole", .. })
    ));
    assert!(matches!(
        genfmt_ordinary_transition_b_matrix_from_handoffs(&global, &phase),
        Err(IoError::InvalidPhaseBin { field: "ihole", .. })
    ));
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
fn parses_legacy_ten_integer_header_without_q_count() -> Result<()> {
    let data = legacy_phase_bin_data();
    let mut text = phase_bin_string(&data)?;
    text.replace_range(
        0..text.lines().next().map_or(0, str::len),
        "    2    3    2    1    1    4    2    8    8    8",
    );
    let parsed = parse_phase_bin(&text)?;
    assert_eq!(parsed.final_state_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
    assert_eq!(parsed.transition_count, PHASE_BIN_DEFAULT_TRANSITION_COUNT);
    assert_eq!(parsed.q_count, 1);
    assert_close_complex(parsed.transition_moments, data.transition_moments);
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

fn reference_exafs_cu_phase_bin() -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent().and_then(Path::parent)?;
    let path = workspace.join("reference-work/golden/EXAFS/Cu/phase.bin");
    path.is_file().then_some(path)
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

fn sample_band_phase_bin_data() -> PhaseBinData {
    let mut data = sample_phase_bin_data();
    data.energy_count = 4;
    data.main_energy_count = 4;
    data.auxiliary_energy_count = 0;
    data.fermi_index = 1;
    data.scalars.fermi_level = 0.0;
    data.energy_grid = Array1::from_shape_fn(data.energy_count, |energy| {
        Complex64::new(energy as f64, 0.0)
    });
    data.reference_energy =
        Array2::from_shape_fn((data.energy_count, data.spin_count), |(energy, spin)| {
            Complex64::new(0.25 * energy as f64, 0.05 * spin as f64)
        });
    data.potentials = vec![
        sample_potential(1, 29, "Cu", data.energy_count, data.spin_count, 0.1),
        sample_potential(2, 8, "O", data.energy_count, data.spin_count, 0.2),
    ];
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

fn sample_band_input() -> BandInput {
    BandInput {
        mband: 1,
        energy_mesh: BandEnergyMesh {
            emin: FEFF_HARTREE_EV,
            emax: 3.0 * FEFF_HARTREE_EV,
            estep: FEFF_HARTREE_EV,
        },
        nkp: 3,
        ikpath: 1,
        freeprop: false,
    }
}

fn sample_genfmt_input() -> GenfmtInput {
    GenfmtInput {
        control: GenfmtControl {
            mfeff: 1,
            ipr5: 2,
            iorder: 3,
            critcw: 4.5,
            wnstar: true,
        },
        decomposition_channels: 1,
    }
}

fn sample_global_input() -> GlobalInput {
    GlobalInput {
        cfaverage: CfAverage {
            nabs: 1,
            iphabs: 0,
            rclabs: 100000.0,
        },
        control: GlobalControl {
            ipol: 1,
            ispin: 1,
            le2: 0,
            elpty: 0.0,
            angks: 0.0,
            l2lp: 0,
            do_nrixs: 0,
            ldecmx: -1,
            lj: -1,
        },
        evec: [0.0, 0.0, 1.0],
        xivec: [1.0, 0.0, 0.0],
        spvec: [0.0, 0.0, 1.0],
        polarization_tensor: [[0.0; 6]; 3],
        norms: GlobalNorms {
            evnorm: 1.0,
            xivnorm: 1.0,
            spvnorm: 1.0,
        },
        q_control: GlobalQControl {
            nq: 0,
            imdff: 0,
            qaverage: true,
            mixdff: false,
        },
        q_vectors: Vec::new(),
        mdff: None,
    }
}

fn sample_genfmt_path() -> PathsDatGenfmtPath {
    PathsDatGenfmtPath {
        index: 17,
        degeneracy: 4.0,
        effective_half_path_length_bohr: 1.0,
        potential_indices: Array1::from_vec(vec![1, 0]),
        positions_bohr: arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
}

fn sample_nstar_genfmt_path() -> PathsDatGenfmtPath {
    PathsDatGenfmtPath {
        index: 23,
        degeneracy: 3.0,
        effective_half_path_length_bohr: 1.0,
        potential_indices: Array1::from_vec(vec![1, 0]),
        positions_bohr: arr2(&[[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]]),
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

fn sample_jas_phase_bin_data() -> PhaseBinData {
    let mut data = sample_phase_bin_data();
    data.final_state_count = 40;
    data.transition_count = 1;
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
