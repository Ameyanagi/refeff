use super::{support::*, *};

#[test]
fn rdgeom_atomic_units_match_feff_setup_reference() -> Result<(), ScreenError> {
    let positions = array![
        [0.0, 0.529_177_249, -1.058_354_498],
        [1.322_943_122_5, -0.264_588_624_5, 0.0]
    ];

    let setup = screen_rdgeom_atomic_units(ScreenRdgeomAtomicUnitsInput {
        atom_positions_angstrom: positions.view(),
        rfms2_angstrom: 1.058_354_498,
        direct_radius_angstrom: 2.645_886_245,
        min_real_energy_ev: -40.0,
        max_real_energy_ev: 0.0,
        max_imaginary_energy_ev: 2.0,
        screen_rfms_angstrom: 4.0,
        min_imaginary_energy_ev: 0.001,
        max_l: 4,
        angular_capacity_lx: 2,
    })?;

    assert_eq!(setup.atom_positions_bohr.strides(), &[1, 2]);
    assert_close(setup.atom_positions_bohr[(0, 0)], 0.0, 1.0e-15);
    assert_close(setup.atom_positions_bohr[(0, 1)], 1.0, 1.0e-15);
    assert_close(setup.atom_positions_bohr[(0, 2)], -2.0, 1.0e-15);
    assert_close(setup.atom_positions_bohr[(1, 0)], 2.5, 1.0e-15);
    assert_close(setup.atom_positions_bohr[(1, 1)], -0.5, 1.0e-15);
    assert_close(setup.atom_positions_bohr[(1, 2)], 0.0, 1.0e-15);
    assert_close(setup.rfms2_bohr, 2.0, 1.0e-15);
    assert_close(setup.direct_radius_bohr, 5.0, 1.0e-15);
    assert_close(
        setup.min_real_energy_hartree,
        -1.469_972_360_109_712_8,
        1.0e-15,
    );
    assert_close(setup.max_real_energy_hartree, 0.0, 1.0e-15);
    assert_close(
        setup.max_imaginary_energy_hartree,
        0.073_498_618_005_485_64,
        1.0e-15,
    );
    assert_close(setup.screen_rfms_bohr, 7.558_903_954_315_693, 1.0e-15);
    assert_close(
        setup.min_imaginary_energy_hartree,
        3.674_930_900_274_282_3e-5,
        1.0e-18,
    );
    assert_eq!(setup.max_l, 3);
    Ok(())
}

#[test]
fn phase_potential_shift_matches_feff_prep_reference() -> Result<(), ScreenError> {
    let total = array![10.0, 11.0, 12.0, 13.0, 14.0];
    let valence = array![20.0, 21.0, 22.0, 23.0, 24.0];

    let low_exchange = screen_phase_potential_reference_shift(ScreenPhasePotentialInput {
        total_potential: total.view(),
        valence_potential: valence.view(),
        muffin_tin_next_index_1based: 3,
        exchange_selector: 4,
    })?;
    assert_close(low_exchange.reference_energy, 12.0, 1.0e-15);
    assert_array_close(
        &low_exchange.total_potential,
        &[-2.0, -1.0, 0.0, 13.0, 14.0],
        1.0e-15,
    );
    assert_array_close(
        &low_exchange.valence_potential,
        &[-2.0, -1.0, 0.0, 23.0, 24.0],
        1.0e-15,
    );

    let high_exchange = screen_phase_potential_reference_shift(ScreenPhasePotentialInput {
        total_potential: total.view(),
        valence_potential: valence.view(),
        muffin_tin_next_index_1based: 3,
        exchange_selector: 5,
    })?;
    assert_array_close(
        &high_exchange.total_potential,
        &[-2.0, -1.0, 0.0, 13.0, 14.0],
        1.0e-15,
    );
    assert_array_close(
        &high_exchange.valence_potential,
        &[8.0, 9.0, 10.0, 23.0, 24.0],
        1.0e-15,
    );
    Ok(())
}

#[test]
fn lda_exchange_correlation_kernel_matches_feff_ldafxc_reference() -> Result<(), ScreenError> {
    let radii = [0.5, 0.75, 1.0, 1.5, 2.0];
    let density = [0.04, 0.10, 0.0, -1.0, 0.25];

    let full = screen_lda_exchange_correlation_kernel(&radii, &density, 0, radii.len())?;
    assert_close(full[0], -16.919_199_214_545_813, 1.0e-13);
    assert_close(full[1], -3.960_989_192_391_738_6, 1.0e-13);
    assert_close(full[2], 0.0, 1.0e-15);
    assert_close(full[3], 0.0, 1.0e-15);
    assert_close(full[4], -0.294_609_719_384_913, 1.0e-13);

    let exchange_only = screen_lda_exchange_correlation_kernel(&radii, &density, 2, radii.len())?;
    assert_close(exchange_only[0], -14.488_412_060_289_518, 1.0e-13);
    assert_close(exchange_only[1], -3.495_786_749_594_309_6, 1.0e-13);
    assert_close(exchange_only[4], -0.266_878_831_976_939_35, 1.0e-13);
    Ok(())
}

#[test]
fn coulomb_kernel_matrix_matches_feff_response_setup_reference() -> Result<(), ScreenError> {
    let radii = [0.5, 1.0, 2.0];
    let local_kernel = [0.1, -0.2, 0.0];
    let matrix = screen_coulomb_kernel_matrix(&radii, radii.len(), Some(&local_kernel))?;
    let pi = std::f64::consts::PI;

    assert_close(matrix[(0, 0)], 8.4 * pi, 1.0e-14);
    assert_close(matrix[(0, 1)], 4.0 * pi, 1.0e-14);
    assert_close(matrix[(1, 0)], 4.0 * pi, 1.0e-14);
    assert_close(matrix[(0, 2)], 2.0 * pi, 1.0e-14);
    assert_close(matrix[(2, 0)], 2.0 * pi, 1.0e-14);
    assert_close(matrix[(1, 1)], 3.2 * pi, 1.0e-14);
    assert_close(matrix[(1, 2)], 2.0 * pi, 1.0e-14);
    assert_close(matrix[(2, 1)], 2.0 * pi, 1.0e-14);
    assert_close(matrix[(2, 2)], 2.0 * pi, 1.0e-14);
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            assert_close(matrix[(row, column)], matrix[(column, row)], 1.0e-14);
        }
    }
    Ok(())
}

#[test]
fn bare_core_hole_potential_matches_feff_loop_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 4.0];
    let large = [1.0, 0.5, 0.25];
    let small = [0.0, 0.25, 0.0];
    let potential = screen_bare_core_hole_potential(&radii, &large, &small, 0.1, radii.len())?;

    assert_close(potential[0], 0.1375, 1.0e-14);
    assert_close(potential[1], 0.0875, 1.0e-14);
    assert_close(potential[2], 0.046875, 1.0e-14);
    Ok(())
}

#[test]
fn radial_coulomb_potential_matches_feff_shell_weight_loop() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 3.0];
    let shell_weights = [0.5, 0.5, 0.0];
    let potential = screen_radial_coulomb_potential(&radii, &shell_weights, radii.len())?;

    assert_close(potential[0], 0.75, 1.0e-14);
    assert_close(potential[1], 0.5, 1.0e-14);
    assert_close(potential[2], 1.0 / 3.0, 1.0e-14);
    Ok(())
}

#[test]
fn crpa_density_weights_match_feff_normalization_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 3.0];
    let density = [2.0, 4.0, 6.0];
    let weights = screen_crpa_density_weights(&radii, &density, 0.1, radii.len(), 2, None)?;

    assert_close(weights.normalization, 2.8, 1.0e-14);
    assert_close(weights.normalized_density[0], 5.0 / 7.0, 1.0e-14);
    assert_close(weights.normalized_density[1], 10.0 / 7.0, 1.0e-14);
    assert_close(weights.normalized_density[2], 15.0 / 7.0, 1.0e-14);
    assert_close(weights.shell_weights[0], 1.0 / 14.0, 1.0e-14);
    assert_close(weights.shell_weights[1], 2.0 / 7.0, 1.0e-14);
    assert_close(weights.shell_weights[2], 0.0, 1.0e-14);

    let projected = screen_crpa_density_weights(
        &radii,
        &density,
        0.1,
        radii.len(),
        radii.len(),
        Some(ScreenCrpaProjectionWindow {
            inner_radius: 1.0,
            outer_radius: 3.0,
        }),
    )?;
    assert_close(projected.normalization, 0.4, 1.0e-14);
    assert_close(projected.normalized_density[0], 5.0, 1.0e-14);
    assert_close(projected.normalized_density[1], 2.5, 1.0e-14);
    assert_close(projected.normalized_density[2], 0.0, 1.0e-14);
    assert_close(projected.shell_weights[0], 0.5, 1.0e-14);
    assert_close(projected.shell_weights[1], 0.5, 1.0e-14);
    assert_close(projected.shell_weights[2], 0.0, 1.0e-14);
    Ok(())
}

#[test]
fn crpa_hubbard_summary_matches_feff_accumulation_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 3.0];
    let screened = [0.5, 1.0, 1.5];
    let bare = [2.0, 1.0, 0.5];
    let total_density = [5.0 / 7.0, 10.0 / 7.0, 15.0 / 7.0];
    let orbital_density = [0.2, 0.3, 0.4];

    let summary = screen_crpa_hubbard_summary(
        &radii,
        &screened,
        &bare,
        &total_density,
        &orbital_density,
        0.1,
        radii.len(),
    )?;

    assert_close(summary.screened_density_potential[0], 0.1, 1.0e-14);
    assert_close(summary.screened_density_potential[1], 0.3, 1.0e-14);
    assert_close(summary.screened_density_potential[2], 0.6, 1.0e-14);
    assert_close(summary.hubbard_u, 9.0 / 7.0, 1.0e-14);
    assert_close(summary.occupation, 1.0, 1.0e-14);
    assert_close(summary.bare_u, 0.75, 1.0e-14);
    Ok(())
}
