use super::{
    RealVec, ScreenContourEnergyGridInput, ScreenCrpaProjectionWindow,
    ScreenCrpaResponseSliceInput, ScreenEnergyStateInput, ScreenError,
    ScreenExactRadialContinuationInput, ScreenFmsResponseSliceInput, ScreenGetphRadialBoundsInput,
    ScreenIrregularInitialConditionInput, ScreenIrregularWronskianScaleInput,
    ScreenPhasePotentialInput, ScreenRadialBoundsInput, ScreenRdgeomAtomicUnitsInput,
    ScreenSolutionNormalizationInput, screen_atomic_response_slice,
    screen_bare_core_hole_potential, screen_contour_energy_grid, screen_coulomb_kernel_matrix,
    screen_crpa_density_weights, screen_crpa_hubbard_summary, screen_crpa_orbital_density,
    screen_crpa_response_slice, screen_energy_integration_delta, screen_energy_state,
    screen_exact_radial_continuation, screen_exponential_energy_grid,
    screen_fms_cluster_green_trace, screen_fms_response_slice, screen_getph_lmax,
    screen_getph_radial_bounds, screen_integrate_response_step, screen_irregular_initial_condition,
    screen_irregular_wronskian_scale, screen_lda_exchange_correlation_kernel,
    screen_phase_potential_reference_shift, screen_radial_bounds, screen_radial_coulomb_potential,
    screen_radial_grid, screen_radial_index_1based, screen_rdgeom_atomic_units,
    screen_response_system_matrix, screen_solution_normalization, screen_solve_response_potential,
    screen_symmetrize_response_upper,
};
use ndarray::array;
use num_complex::Complex32;
use refeff_linalg::LinalgError;

use crate::Complex;

#[test]
fn exponential_energy_grid_matches_feff_setegrid_reference() -> Result<(), ScreenError> {
    let grid = screen_exponential_energy_grid(8.0, 5)?;

    assert_complex_close(grid[0], 0.0, 8.000_000_000_000_002, 1.0e-14);
    assert_complex_close(grid[1], 0.0, 4.196_152_422_706_632, 1.0e-14);
    assert_complex_close(grid[2], 0.0, 2.000_000_000_000_000_4, 1.0e-14);
    assert_complex_close(grid[3], 0.0, 0.732_050_807_568_877_4, 1.0e-14);
    assert_complex_close(grid[4], 0.0, 0.0, 1.0e-14);
    Ok(())
}

#[test]
fn contour_energy_grid_matches_feff_setegi_reference() -> Result<(), ScreenError> {
    let grid = screen_contour_energy_grid(ScreenContourEnergyGridInput {
        min_real_energy: -0.2,
        max_real_energy: 0.4,
        max_imaginary_energy: 0.5,
        min_imaginary_energy: 0.0,
        real_points: 4,
        imaginary_points: 4,
        max_points: 20,
    })?;

    assert_eq!(grid.active_len, 10);
    assert_close(grid.effective_min_imaginary_energy, 0.05, 1.0e-15);
    assert_complex_close(grid.energies[0], -0.2, 0.05, 1.0e-14);
    assert_complex_close(grid.energies[1], -0.2, 0.2, 1.0e-14);
    assert_complex_close(grid.energies[2], -0.2, 0.35, 1.0e-14);
    assert_complex_close(grid.energies[3], -0.2, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[4], -5.551_115_123_125_783e-17, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[5], 0.2, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[6], 0.4, 0.5, 1.0e-14);
    assert_complex_close(grid.energies[7], 0.4, 0.35, 1.0e-14);
    assert_complex_close(grid.energies[8], 0.4, 0.2, 1.0e-14);
    assert_complex_close(grid.energies[9], 0.4, 0.05, 1.0e-14);
    assert_complex_close(grid.energies[10], 0.0, 0.0, 1.0e-15);
    Ok(())
}

#[test]
fn radial_grid_matches_feff_setri_reference() -> Result<(), ScreenError> {
    let grid = screen_radial_grid(0.05, 8.8, 5)?;

    assert_close(grid[0], 0.000_150_733_075_095_476_5, 1.0e-15);
    assert_close(grid[1], 0.000_158_461_325_115_751_26, 1.0e-15);
    assert_close(grid[2], 0.000_166_585_810_987_633_24, 1.0e-15);
    assert_close(grid[3], 0.000_175_126_848_157_658_42, 1.0e-15);
    assert_close(grid[4], 0.000_184_105_793_667_578_87, 1.0e-15);
    assert_eq!(screen_radial_index_1based(8.8, 0.05, grid[2])?, 3);
    assert_eq!(screen_radial_index_1based(8.8, 0.05, 1.0)?, 177);
    assert_eq!(screen_radial_index_1based(0.0, 1.0, 0.01)?, -3);
    Ok(())
}

#[test]
fn radial_bounds_match_feff_screensub_reference() -> Result<(), ScreenError> {
    let bounds = screen_radial_bounds(ScreenRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        tail_extension: 3,
        radial_capacity: 251,
        response_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.muffin_tin_next_index_1based, 165);
    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 190);
    Ok(())
}

#[test]
fn radial_bounds_clamp_ilast_to_response_capacity() -> Result<(), ScreenError> {
    let bounds = screen_radial_bounds(ScreenRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        tail_extension: 3,
        radial_capacity: 251,
        response_capacity: 185,
    })?;

    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 185);
    Ok(())
}

#[test]
fn getph_radial_bounds_match_feff_reference() -> Result<(), ScreenError> {
    let bounds = screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 1.2,
        radial_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.norman_index_1based, 181);
    assert_eq!(bounds.active_count, 187);
    Ok(())
}

#[test]
fn getph_radial_bounds_clamp_ilast_to_radial_capacity() -> Result<(), ScreenError> {
    let bounds = screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
        x0: 8.8,
        dx: 0.05,
        muffin_tin_radius: 0.5,
        norman_radius: 38.474_666_049_032_14,
        radial_capacity: 251,
    })?;

    assert_eq!(bounds.muffin_tin_index_1based, 164);
    assert_eq!(bounds.norman_index_1based, 251);
    assert_eq!(bounds.active_count, 251);
    Ok(())
}

#[test]
fn energy_state_matches_feff_per_energy_reference() -> Result<(), ScreenError> {
    let state = screen_energy_state(ScreenEnergyStateInput {
        energy: Complex::new(0.4, 0.5),
        reference_energy: Complex::new(0.1, 0.05),
        muffin_tin_radius: 1.7,
        exchange_selector: 7,
    })?;

    assert_complex_close(state.kinetic_energy, 0.3, 0.45, 1.0e-15);
    assert_complex_close(
        state.wave_number,
        0.916_970_019_128_716_1,
        0.490_754_528_006_756_5,
        1.0e-14,
    );
    assert_complex32_close(
        state.fms_wave_number,
        0.916_970_014_572_143_6,
        0.490_754_514_932_632_45,
        1.0e-6,
    );
    assert_complex_close(
        state.muffin_tin_argument,
        1.558_849_032_518_817_3,
        0.834_282_697_611_486,
        1.0e-14,
    );
    assert_eq!(state.dirac_cycle_count, 3);

    let low_exchange = screen_energy_state(ScreenEnergyStateInput {
        exchange_selector: 14,
        ..ScreenEnergyStateInput {
            energy: Complex::new(0.4, 0.5),
            reference_energy: Complex::new(0.1, 0.05),
            muffin_tin_radius: 1.7,
            exchange_selector: 7,
        }
    })?;
    assert_eq!(low_exchange.dirac_cycle_count, 0);
    Ok(())
}

#[test]
fn getph_lmax_matches_feff_light_element_overrides() -> Result<(), ScreenError> {
    assert_eq!(screen_getph_lmax(29, 5, 3)?, 3);
    assert_eq!(screen_getph_lmax(8, 2, 3)?, 2);
    assert_eq!(screen_getph_lmax(4, 5, 10)?, 2);
    assert_eq!(screen_getph_lmax(2, 5, 10)?, 1);
    assert_eq!(screen_getph_lmax(1, 0, 0)?, 1);
    Ok(())
}

#[test]
fn solution_normalization_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(1.25, -0.4),
    })?;

    assert_complex_close(
        normalization.small_component_factor,
        -0.001_459_482_078_780_620_7,
        -0.001_824_332_682_938_356_4,
        1.0e-16,
    );
    assert_complex_close(
        normalization.relativistic_scale,
        1.000_000_599_040_804_3,
        -0.000_002_662_585_641_506_650_3,
        1.0e-16,
    );
    assert_complex_close(
        normalization.regular_solution_scale,
        0.725_690_457_959_513_5,
        0.232_218_816_478_531_07,
        1.0e-16,
    );

    let zero_amplitude = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number: Complex::new(0.4, 0.5),
        phase_amplitude: Complex::new(0.0, 0.0),
    })?;
    assert_complex_close(zero_amplitude.regular_solution_scale, 0.0, 0.0, 1.0e-16);
    Ok(())
}

#[test]
fn irregular_initial_condition_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let input = ScreenIrregularInitialConditionInput {
        muffin_tin_radius: 1.7,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.8, 0.1),
        neumann_l: Complex::new(-0.3, 0.05),
        bessel_j_l_plus_1: Complex::new(0.25, -0.03),
        neumann_l_plus_1: Complex::new(-0.6, 0.2),
        hankel_l: Complex::new(0.1, 0.7),
        hankel_l_plus_1: Complex::new(-0.2, 0.3),
        use_hankel_boundary: false,
    };

    let standing = screen_irregular_initial_condition(input)?;
    assert_complex_close(
        standing.large_component,
        -0.215_795_629_731_268_06,
        -0.025_994_455_746_676_352,
        1.0e-16,
    );
    assert_complex_close(
        standing.small_component,
        0.001_838_866_245_442_668,
        0.001_316_132_001_240_697_2,
        1.0e-17,
    );

    let hankel = screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
        use_hankel_boundary: true,
        ..input
    })?;
    assert_complex_close(
        hankel.large_component,
        -0.077_143_175_772_786_6,
        1.326_264_690_969_657_8,
        1.0e-15,
    );
    assert_complex_close(
        hankel.small_component,
        0.001_572_486_508_374_408_2,
        0.000_178_855_217_613_778_5,
        1.0e-17,
    );
    Ok(())
}

#[test]
fn irregular_wronskian_scale_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let scale = screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        regular_large_at_match: Complex::new(0.3, 0.2),
        regular_small_at_match: Complex::new(-0.01, 0.04),
        irregular_large_at_match: Complex::new(0.7, -0.2),
        irregular_small_at_match: Complex::new(0.02, 0.03),
    })?;

    assert_complex_close(
        scale.phase_factor,
        1.083_141_079_608_063_2,
        0.219_563_566_708_252_36,
        1.0e-15,
    );
    assert_complex_close(
        scale.denominator,
        -0.726_137_142_242_051_2,
        5.106_772_750_294_418,
        1.0e-14,
    );
    assert_complex_close(
        scale.reciprocal_wave_scale,
        -0.260_696_573_980_254_4,
        -0.153_973_620_782_305_84,
        1.0e-15,
    );
    assert_complex_close(
        scale.irregular_solution_scale,
        -0.248_564_171_233_149_1,
        -0.224_014_623_457_035_68,
        1.0e-15,
    );

    let zero = screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.0, 0.0),
        regular_large_at_match: Complex::new(0.0, 0.0),
        regular_small_at_match: Complex::new(0.0, 0.0),
        irregular_large_at_match: Complex::new(0.0, 0.0),
        irregular_small_at_match: Complex::new(0.0, 0.0),
    })?;
    assert_complex_close(zero.reciprocal_wave_scale, 0.0, 0.0, 1.0e-16);
    assert_complex_close(zero.irregular_solution_scale, 0.0, 0.0, 1.0e-16);
    Ok(())
}

#[test]
fn exact_radial_continuation_matches_feff_screensub_reference() -> Result<(), ScreenError> {
    let continued = screen_exact_radial_continuation(ScreenExactRadialContinuationInput {
        radius: 2.0,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.6, 0.2),
        neumann_l: Complex::new(-0.4, 0.1),
        bessel_j_l_plus_1: Complex::new(0.3, 0.05),
        neumann_l_plus_1: Complex::new(-0.2, 0.2),
        hankel_l: Complex::new(0.1, 0.7),
        hankel_l_plus_1: Complex::new(-0.2, 0.3),
    })?;

    assert_complex_close(
        continued.regular_large_component,
        1.314_103_542_373_494,
        0.299_396_383_930_798,
        1.0e-15,
    );
    assert_complex_close(
        continued.regular_small_component,
        -0.000_934_743_791_234_705_6,
        -0.001_135_887_639_152_749_7,
        1.0e-17,
    );
    assert_complex_close(
        continued.irregular_large_component,
        -0.090_756_677_379_748_95,
        1.560_311_401_140_773_7,
        1.0e-15,
    );
    assert_complex_close(
        continued.irregular_small_component,
        0.001_849_984_127_499_303_5,
        0.000_210_417_903_075_033_55,
        1.0e-17,
    );
    Ok(())
}

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

#[test]
fn energy_integration_delta_matches_feff_trapezoid_reference() -> Result<(), ScreenError> {
    let energies = array![
        Complex::new(0.0, 0.1),
        Complex::new(1.0, 0.2),
        Complex::new(3.0, 0.5),
        Complex::new(6.0, 1.1)
    ];

    assert_complex_close(
        screen_energy_integration_delta(energies.view(), 0)?,
        0.5,
        0.05,
        1.0e-14,
    );
    assert_complex_close(
        screen_energy_integration_delta(energies.view(), 1)?,
        1.5,
        0.2,
        1.0e-14,
    );
    assert_complex_close(
        screen_energy_integration_delta(energies.view(), 2)?,
        2.5,
        0.45,
        1.0e-14,
    );
    assert_complex_close(
        screen_energy_integration_delta(energies.view(), 3)?,
        1.5,
        0.3,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn response_integration_and_symmetry_match_feff_upper_triangle() -> Result<(), ScreenError> {
    let accumulated = array![
        [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)],
        [Complex::new(9.0, 0.0), Complex::new(4.0, 1.0)]
    ];
    let response_at_energy = array![
        [Complex::new(0.5, 1.0), Complex::new(-1.0, 0.5)],
        [Complex::new(3.0, 3.0), Complex::new(2.0, -1.0)]
    ];
    let integrated = screen_integrate_response_step(
        accumulated.view(),
        response_at_energy.view(),
        Complex::new(0.2, 0.1),
        2,
    )?;

    assert_eq!(integrated.strides(), &[1, 2]);
    assert_complex_close(integrated[(0, 0)], 1.0, 1.25, 1.0e-14);
    assert_complex_close(integrated[(0, 1)], 1.75, 0.0, 1.0e-14);
    assert_complex_close(integrated[(1, 0)], 9.0, 0.0, 1.0e-14);
    assert_complex_close(integrated[(1, 1)], 4.5, 1.0, 1.0e-14);

    let symmetric = screen_symmetrize_response_upper(integrated.view(), 2)?;
    assert_complex_close(symmetric[(0, 1)], 1.75, 0.0, 1.0e-14);
    assert_complex_close(symmetric[(1, 0)], 1.75, 0.0, 1.0e-14);
    assert_complex_close(symmetric[(1, 1)], 4.5, 1.0, 1.0e-14);
    Ok(())
}

#[test]
fn crpa_orbital_density_matches_feff_density_row_reference() -> Result<(), ScreenError> {
    let regular = array![
        Complex::new(1.0, 0.0),
        Complex::new(0.5, 0.25),
        Complex::new(0.0, 1.0)
    ];
    let irregular = array![
        Complex::new(0.2, 0.1),
        Complex::new(0.4, -0.2),
        Complex::new(-0.3, 0.2)
    ];

    let density = screen_crpa_orbital_density(
        regular.view(),
        irregular.view(),
        Complex::new(0.1, 0.2),
        Complex::new(0.7, 0.3),
        2,
        regular.len(),
    )?;

    assert_close(density[0], 1.909_859_317_102_744_5, 1.0e-14);
    assert_close(density[1], 0.696_302_876_027_042_2, 1.0e-14);
    assert_close(density[2], -2.801_126_998_417_358, 1.0e-14);
    Ok(())
}

#[test]
fn atomic_response_slice_matches_feff_upper_triangle_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0];
    let regular = array![Complex::new(1.0, 0.0), Complex::new(0.5, 0.25)];
    let irregular = array![Complex::new(0.2, 0.1), Complex::new(-0.3, 0.2)];

    let response = screen_atomic_response_slice(
        &radii,
        regular.view(),
        irregular.view(),
        Complex::new(0.7, 0.3),
        0.1,
        1,
        radii.len(),
    )?;

    assert_eq!(response.strides(), &[1, 2]);
    assert_complex_close(
        response[(0, 0)],
        2.918_050_088_899_328_5e-5,
        -0.000_173_867_151_130_251_67,
        1.0e-14,
    );
    assert_complex_close(
        response[(0, 1)],
        -0.000_855_961_359_410_469_6,
        0.000_328_280_635_001_174_44,
        1.0e-14,
    );
    assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(
        response[(1, 1)],
        -0.000_485_125_827_279_513_3,
        -0.000_304_875_441_579_794_35,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn fms_response_slice_matches_feff_cluster_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 3.0];
    let regular = array![
        Complex::new(1.0, 0.0),
        Complex::new(0.5, 0.25),
        Complex::new(0.25, -0.1)
    ];
    let irregular = array![
        Complex::new(0.2, 0.1),
        Complex::new(-0.3, 0.2),
        Complex::new(0.4, 0.05)
    ];

    let response = screen_fms_response_slice(ScreenFmsResponseSliceInput {
        radii: &radii,
        regular_solution: regular.view(),
        irregular_solution: irregular.view(),
        cluster_green: Complex::new(0.1, 0.2),
        wave_number: Complex::new(0.7, 0.3),
        dx: 0.1,
        angular_momentum: 1,
        active_count: radii.len(),
        fms_count: 2,
    })?;

    assert_eq!(response.strides(), &[1, 3]);
    assert_complex_close(
        response[(0, 0)],
        0.000_430_412_388_112_651,
        -0.000_263_840_362_204_647_56,
        1.0e-14,
    );
    assert_complex_close(
        response[(0, 1)],
        -0.000_063_832_345_694_672_8,
        0.000_699_876_076_009_448_3,
        1.0e-14,
    );
    assert_complex_close(response[(0, 2)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(
        response[(1, 1)],
        -0.000_373_875_167_640_226_43,
        0.000_230_537_355_656_206_72,
        1.0e-14,
    );
    assert_complex_close(response[(2, 2)], 0.0, 0.0, 1.0e-14);
    Ok(())
}

#[test]
fn crpa_response_slice_matches_feff_projected_reference() -> Result<(), ScreenError> {
    let radii = [1.0, 2.0, 3.0];
    let regular = array![
        Complex::new(1.0, 0.0),
        Complex::new(0.5, 0.25),
        Complex::new(0.25, -0.1)
    ];
    let irregular = array![
        Complex::new(0.2, 0.1),
        Complex::new(-0.3, 0.2),
        Complex::new(0.4, 0.05)
    ];

    let response = screen_crpa_response_slice(ScreenCrpaResponseSliceInput {
        radii: &radii,
        regular_solution: regular.view(),
        irregular_solution: irregular.view(),
        cluster_green: Complex::new(0.1, 0.2),
        wave_number: Complex::new(0.7, 0.3),
        dx: 0.1,
        angular_momentum: 1,
        crpa_angular_momentum: 1,
        projection_window: Some(ScreenCrpaProjectionWindow {
            inner_radius: 1.0,
            outer_radius: 3.0,
        }),
        active_count: radii.len(),
    })?;

    assert_eq!(response.strides(), &[1, 3]);
    assert_complex_close(response[(0, 0)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(response[(0, 1)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(response[(0, 2)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(response[(1, 0)], 0.0, 0.0, 1.0e-14);
    assert_complex_close(
        response[(1, 1)],
        -0.000_053_687_562_182_483_7,
        -0.000_004_646_130_370_224_231,
        1.0e-14,
    );
    assert_complex_close(
        response[(1, 2)],
        0.000_182_520_613_470_705_1,
        -0.000_287_665_880_223_793_86,
        1.0e-14,
    );
    assert_complex_close(
        response[(2, 2)],
        -0.000_427_456_676_939_791_8,
        -0.000_205_369_424_409_379_63,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn fms_cluster_green_trace_matches_feff_phase_trace_reference() -> Result<(), ScreenError> {
    let scattering = array![
        [
            Complex32::new(9.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0)
        ],
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0)
        ],
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(-0.25, 0.75),
            Complex32::new(0.0, 0.0)
        ],
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.125, -0.375)
        ]
    ];

    let trace = screen_fms_cluster_green_trace(scattering.view(), Complex::new(0.2, 0.05), 1)?;

    assert_complex_close(
        trace,
        0.140_306_297_914_067_32,
        0.345_849_798_891_802_3,
        1.0e-14,
    );
    Ok(())
}

#[test]
fn response_system_matrix_matches_feff_inversion_setup_reference() -> Result<(), ScreenError> {
    let kernel = array![[2.0, 0.5], [0.5, 1.0]];
    let susceptibility = array![
        [Complex::new(1.0, 0.1), Complex::new(2.0, 0.2)],
        [Complex::new(3.0, 0.3), Complex::new(4.0, 0.05)]
    ];

    let system = screen_response_system_matrix(kernel.view(), susceptibility.view(), 2)?;

    assert_eq!(system.strides(), &[1, 2]);
    assert_close(system[(0, 0)], 0.65, 1.0e-14);
    assert_close(system[(0, 1)], -0.425, 1.0e-14);
    assert_close(system[(1, 0)], -0.35, 1.0e-14);
    assert_close(system[(1, 1)], 0.85, 1.0e-14);
    Ok(())
}

#[test]
fn screened_response_potential_matches_feff_dgetrs_reference() -> Result<(), ScreenError> {
    let kernel = array![[2.0, 0.5], [0.5, 1.0]];
    let susceptibility = array![
        [Complex::new(1.0, 0.1), Complex::new(2.0, 0.2)],
        [Complex::new(3.0, 0.3), Complex::new(4.0, 0.05)]
    ];
    let bare = array![0.8, 0.2];

    let screened =
        screen_solve_response_potential(kernel.view(), susceptibility.view(), bare.view(), 2)?;

    assert_close(screened[0], 612.0 / 323.0, 1.0e-14);
    assert_close(screened[1], 328.0 / 323.0, 1.0e-14);
    Ok(())
}

#[test]
fn screen_helpers_reject_invalid_inputs() {
    assert!(matches!(
        screen_radial_grid(0.0, 8.8, 5),
        Err(ScreenError::NonPositiveInput { name: "dx", .. })
    ));
    assert!(matches!(
        screen_radial_grid(0.05, 8.8, 0),
        Err(ScreenError::EmptyRadialGrid)
    ));
    assert!(matches!(
        screen_exponential_energy_grid(8.0, 1),
        Err(ScreenError::CountTooSmall { name: "energy", .. })
    ));
    assert!(matches!(
        screen_radial_index_1based(8.8, 0.05, -1.0),
        Err(ScreenError::NonPositiveInput { name: "radius", .. })
    ));
    assert!(matches!(
        screen_radial_bounds(ScreenRadialBoundsInput {
            x0: 8.8,
            dx: 0.05,
            muffin_tin_radius: 0.5,
            norman_radius: 1.2,
            tail_extension: 3,
            radial_capacity: 164,
            response_capacity: 251,
        }),
        Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: 165,
            capacity: 164
        })
    ));
    assert!(matches!(
        screen_radial_bounds(ScreenRadialBoundsInput {
            x0: 0.0,
            dx: 1.0,
            muffin_tin_radius: 0.01,
            norman_radius: 1.2,
            tail_extension: 3,
            radial_capacity: 251,
            response_capacity: 251,
        }),
        Err(ScreenError::NonPositiveRadialBound {
            name: "muffin_tin_index_1based",
            value: -2
        })
    ));
    assert!(matches!(
        screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
            x0: 8.8,
            dx: 0.05,
            muffin_tin_radius: 0.5,
            norman_radius: 1.2,
            radial_capacity: 163,
        }),
        Err(ScreenError::RadialBoundOutOfRange {
            name: "getph_muffin_tin_index_1based",
            value: 164,
            capacity: 163
        })
    ));
    assert!(matches!(
        screen_getph_radial_bounds(ScreenGetphRadialBoundsInput {
            x0: 0.0,
            dx: 1.0,
            muffin_tin_radius: 0.01,
            norman_radius: 1.2,
            radial_capacity: 251,
        }),
        Err(ScreenError::NonPositiveRadialBound {
            name: "getph_muffin_tin_index_1based",
            value: -2
        })
    ));
    assert!(matches!(
        screen_energy_state(ScreenEnergyStateInput {
            energy: Complex::new(f64::NAN, 0.0),
            reference_energy: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.0,
            exchange_selector: 0,
        }),
        Err(ScreenError::NonFiniteComplexInput { name: "energy", .. })
    ));
    assert!(matches!(
        screen_energy_state(ScreenEnergyStateInput {
            energy: Complex::new(0.0, 0.0),
            reference_energy: Complex::new(0.0, 0.0),
            muffin_tin_radius: 0.0,
            exchange_selector: 0,
        }),
        Err(ScreenError::NonPositiveInput {
            name: "muffin_tin_radius",
            ..
        })
    ));
    assert!(matches!(
        screen_getph_lmax(0, 4, 3),
        Err(ScreenError::CountTooSmall {
            name: "atomic_number",
            ..
        })
    ));
    assert!(matches!(
        screen_solution_normalization(ScreenSolutionNormalizationInput {
            wave_number: Complex::new(f64::NAN, 0.0),
            phase_amplitude: Complex::new(1.0, 0.0),
        }),
        Err(ScreenError::NonFiniteComplexInput {
            name: "wave_number",
            ..
        })
    ));
    assert!(matches!(
        screen_solution_normalization(ScreenSolutionNormalizationInput {
            wave_number: Complex::new(1.0, 0.0),
            phase_amplitude: Complex::new(0.0, f64::INFINITY),
        }),
        Err(ScreenError::NonFiniteComplexInput {
            name: "phase_amplitude",
            ..
        })
    ));
    assert!(matches!(
        screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
            muffin_tin_radius: 0.0,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
            bessel_j_l: Complex::new(0.8, 0.1),
            neumann_l: Complex::new(-0.3, 0.05),
            bessel_j_l_plus_1: Complex::new(0.25, -0.03),
            neumann_l_plus_1: Complex::new(-0.6, 0.2),
            hankel_l: Complex::new(0.1, 0.7),
            hankel_l_plus_1: Complex::new(-0.2, 0.3),
            use_hankel_boundary: false,
        }),
        Err(ScreenError::NonPositiveInput {
            name: "muffin_tin_radius",
            ..
        })
    ));
    assert!(matches!(
        screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
            muffin_tin_radius: 1.7,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
            bessel_j_l: Complex::new(0.8, 0.1),
            neumann_l: Complex::new(-0.3, 0.05),
            bessel_j_l_plus_1: Complex::new(0.25, -0.03),
            neumann_l_plus_1: Complex::new(-0.6, 0.2),
            hankel_l: Complex::new(f64::NAN, 0.7),
            hankel_l_plus_1: Complex::new(-0.2, 0.3),
            use_hankel_boundary: true,
        }),
        Err(ScreenError::NonFiniteComplexInput {
            name: "hankel_l",
            ..
        })
    ));
    assert!(matches!(
        screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.0, 0.0),
            regular_large_at_match: Complex::new(0.3, 0.2),
            regular_small_at_match: Complex::new(-0.01, 0.04),
            irregular_large_at_match: Complex::new(0.7, -0.2),
            irregular_small_at_match: Complex::new(0.02, 0.03),
        }),
        Err(ScreenError::ZeroComplexResult {
            name: "wave_number"
        })
    ));
    assert!(matches!(
        screen_exact_radial_continuation(ScreenExactRadialContinuationInput {
            radius: -1.0,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
            bessel_j_l: Complex::new(0.6, 0.2),
            neumann_l: Complex::new(-0.4, 0.1),
            bessel_j_l_plus_1: Complex::new(0.3, 0.05),
            neumann_l_plus_1: Complex::new(-0.2, 0.2),
            hankel_l: Complex::new(0.1, 0.7),
            hankel_l_plus_1: Complex::new(-0.2, 0.3),
        }),
        Err(ScreenError::NonPositiveInput { name: "radius", .. })
    ));
    let bad_screen_positions = array![[1.0, 2.0]];
    assert!(matches!(
        screen_rdgeom_atomic_units(ScreenRdgeomAtomicUnitsInput {
            atom_positions_angstrom: bad_screen_positions.view(),
            rfms2_angstrom: 1.0,
            direct_radius_angstrom: 2.0,
            min_real_energy_ev: -40.0,
            max_real_energy_ev: 0.0,
            max_imaginary_energy_ev: 2.0,
            screen_rfms_angstrom: 4.0,
            min_imaginary_energy_ev: 0.001,
            max_l: 4,
            angular_capacity_lx: 2,
        }),
        Err(ScreenError::AtomPositionColumnCount { columns: 2 })
    ));
    let nonfinite_screen_positions = array![[1.0, f64::NAN, 3.0]];
    assert!(matches!(
        screen_rdgeom_atomic_units(ScreenRdgeomAtomicUnitsInput {
            atom_positions_angstrom: nonfinite_screen_positions.view(),
            rfms2_angstrom: 1.0,
            direct_radius_angstrom: 2.0,
            min_real_energy_ev: -40.0,
            max_real_energy_ev: 0.0,
            max_imaginary_energy_ev: 2.0,
            screen_rfms_angstrom: 4.0,
            min_imaginary_energy_ev: 0.001,
            max_l: 4,
            angular_capacity_lx: 2,
        }),
        Err(ScreenError::NonFiniteMatrixInput {
            name: "atom_positions_angstrom",
            row: 0,
            column: 1,
            ..
        })
    ));
    let total = array![1.0, 2.0];
    let valence = array![1.0];
    assert!(matches!(
        screen_phase_potential_reference_shift(ScreenPhasePotentialInput {
            total_potential: total.view(),
            valence_potential: valence.view(),
            muffin_tin_next_index_1based: 0,
            exchange_selector: 0,
        }),
        Err(ScreenError::CountTooSmall {
            name: "muffin_tin_next_index_1based",
            ..
        })
    ));
    assert!(matches!(
        screen_phase_potential_reference_shift(ScreenPhasePotentialInput {
            total_potential: total.view(),
            valence_potential: valence.view(),
            muffin_tin_next_index_1based: 2,
            exchange_selector: 0,
        }),
        Err(ScreenError::RadialBoundOutOfRange {
            name: "muffin_tin_next_index_1based",
            value: 2,
            capacity: 1
        })
    ));
    let bad_total = array![1.0, f64::NAN];
    assert!(matches!(
        screen_phase_potential_reference_shift(ScreenPhasePotentialInput {
            total_potential: bad_total.view(),
            valence_potential: total.view(),
            muffin_tin_next_index_1based: 2,
            exchange_selector: 0,
        }),
        Err(ScreenError::NonFiniteInput {
            name: "reference_potential",
            ..
        })
    ));
    assert!(matches!(
        screen_lda_exchange_correlation_kernel(&[1.0], &[0.1], 0, 2),
        Err(ScreenError::ActiveCountOutOfRange { .. })
    ));
    assert!(matches!(
        screen_lda_exchange_correlation_kernel(&[0.0], &[0.1], 0, 1),
        Err(ScreenError::NonPositiveInput { name: "radius", .. })
    ));
    assert!(matches!(
        screen_lda_exchange_correlation_kernel(&[1.0], &[f64::NAN], 0, 1),
        Err(ScreenError::NonFiniteInput {
            name: "electron_density",
            ..
        })
    ));
    assert!(matches!(
        screen_coulomb_kernel_matrix(&[1.0], 2, None),
        Err(ScreenError::ActiveCountOutOfRange { .. })
    ));
    assert!(matches!(
        screen_coulomb_kernel_matrix(&[1.0], 1, Some(&[f64::NAN])),
        Err(ScreenError::NonFiniteInput {
            name: "local_kernel",
            ..
        })
    ));
    assert!(matches!(
        screen_bare_core_hole_potential(&[1.0], &[1.0], &[0.0], 0.0, 1),
        Err(ScreenError::NonPositiveInput { name: "dx", .. })
    ));
    assert!(matches!(
        screen_bare_core_hole_potential(&[1.0], &[f64::INFINITY], &[0.0], 0.1, 1),
        Err(ScreenError::NonFiniteInput {
            name: "large_component",
            ..
        })
    ));
    assert!(matches!(
        screen_radial_coulomb_potential(&[1.0], &[f64::NAN], 1),
        Err(ScreenError::NonFiniteInput {
            name: "shell_weight",
            ..
        })
    ));
    assert!(matches!(
        screen_crpa_density_weights(&[1.0], &[0.0], 0.1, 1, 1, None),
        Err(ScreenError::NonPositiveResult {
            name: "crpa_density_normalization",
            ..
        })
    ));
    assert!(matches!(
        screen_crpa_hubbard_summary(&[1.0], &[1.0], &[1.0], &[1.0], &[f64::NAN], 0.1, 1,),
        Err(ScreenError::NonFiniteInput {
            name: "orbital_density",
            ..
        })
    ));
    assert!(matches!(
        screen_atomic_response_slice(
            &[1.0],
            array![Complex::new(1.0, 0.0)].view(),
            array![Complex::new(1.0, 0.0)].view(),
            Complex::new(f64::NAN, 0.0),
            0.1,
            0,
            1,
        ),
        Err(ScreenError::NonFiniteComplexInput {
            name: "wave_number",
            ..
        })
    ));
    assert!(matches!(
        screen_fms_response_slice(ScreenFmsResponseSliceInput {
            radii: &[1.0],
            regular_solution: array![Complex::new(1.0, 0.0)].view(),
            irregular_solution: array![Complex::new(1.0, 0.0)].view(),
            cluster_green: Complex::new(0.0, 0.0),
            wave_number: Complex::new(1.0, 0.0),
            dx: 0.1,
            angular_momentum: 0,
            active_count: 1,
            fms_count: 2,
        }),
        Err(ScreenError::ActiveCountOutOfRange {
            active_count: 2,
            len: 1
        })
    ));
    assert!(matches!(
        screen_crpa_response_slice(ScreenCrpaResponseSliceInput {
            radii: &[1.0],
            regular_solution: array![Complex::new(1.0, 0.0)].view(),
            irregular_solution: array![Complex::new(1.0, 0.0)].view(),
            cluster_green: Complex::new(0.0, 0.0),
            wave_number: Complex::new(1.0, 0.0),
            dx: 0.1,
            angular_momentum: 0,
            crpa_angular_momentum: 0,
            projection_window: Some(ScreenCrpaProjectionWindow {
                inner_radius: 2.0,
                outer_radius: 1.0,
            }),
            active_count: 1,
        }),
        Err(ScreenError::NonIncreasingInput {
            upper_name: "projection_outer_radius",
            ..
        })
    ));
    assert!(matches!(
        screen_fms_cluster_green_trace(
            array![[Complex32::new(1.0, 0.0)]].view(),
            Complex::new(0.0, 0.0),
            1,
        ),
        Err(ScreenError::MatrixTooSmall {
            name: "fms_scattering",
            active_count: 4,
            ..
        })
    ));
    assert!(matches!(
        screen_fms_cluster_green_trace(
            array![[Complex32::new(f32::NAN, 0.0)]].view(),
            Complex::new(0.0, 0.0),
            0,
        ),
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name: "fms_scattering",
            row: 0,
            column: 0,
            ..
        })
    ));
    let two_energies = array![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
    assert!(matches!(
        screen_energy_integration_delta(two_energies.view(), 2),
        Err(ScreenError::EnergyIndexOutOfRange { index: 2, len: 2 })
    ));
    let regular = array![Complex::new(f64::NAN, 0.0)];
    let irregular = array![Complex::new(1.0, 0.0)];
    assert!(matches!(
        screen_crpa_orbital_density(
            regular.view(),
            irregular.view(),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            0,
            1,
        ),
        Err(ScreenError::NonFiniteComplexInput {
            name: "regular_solution",
            ..
        })
    ));
    assert!(matches!(
        screen_crpa_density_weights(
            &[1.0],
            &[1.0],
            0.1,
            1,
            1,
            Some(ScreenCrpaProjectionWindow {
                inner_radius: 2.0,
                outer_radius: 1.0,
            }),
        ),
        Err(ScreenError::NonIncreasingInput {
            upper_name: "projection_outer_radius",
            ..
        })
    ));
    let kernel = array![[1.0]];
    let susceptibility = array![[Complex::new(0.0, 0.0)]];
    assert!(matches!(
        screen_response_system_matrix(kernel.view(), susceptibility.view(), 2),
        Err(ScreenError::MatrixTooSmall { name: "kernel", .. })
    ));
    let bad_susceptibility = array![[Complex::new(f64::NAN, 0.0)]];
    assert!(matches!(
        screen_response_system_matrix(kernel.view(), bad_susceptibility.view(), 1),
        Err(ScreenError::NonFiniteComplexMatrixInput {
            name: "susceptibility",
            row: 0,
            column: 0,
            ..
        })
    ));
    let bare = array![f64::NAN];
    assert!(matches!(
        screen_solve_response_potential(kernel.view(), susceptibility.view(), bare.view(), 1),
        Err(ScreenError::NonFiniteInput {
            name: "bare_potential",
            ..
        })
    ));
    let singular_susceptibility = array![
        [Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]
    ];
    let identity_kernel = array![[1.0, 0.0], [0.0, 1.0]];
    let singular_rhs = array![1.0, 1.0];
    assert!(matches!(
        screen_solve_response_potential(
            identity_kernel.view(),
            singular_susceptibility.view(),
            singular_rhs.view(),
            2
        ),
        Err(ScreenError::Linalg(LinalgError::SingularMatrix {
            pivot: 0
        }))
    ));
    assert!(matches!(
        screen_contour_energy_grid(ScreenContourEnergyGridInput {
            min_real_energy: 0.4,
            max_real_energy: 0.4,
            max_imaginary_energy: 0.5,
            min_imaginary_energy: 0.05,
            real_points: 4,
            imaginary_points: 4,
            max_points: 20,
        }),
        Err(ScreenError::NonIncreasingInput {
            upper_name: "max_real_energy",
            ..
        })
    ));
    assert!(matches!(
        screen_contour_energy_grid(ScreenContourEnergyGridInput {
            min_real_energy: -0.2,
            max_real_energy: 0.4,
            max_imaginary_energy: 0.04,
            min_imaginary_energy: 0.0,
            real_points: 4,
            imaginary_points: 4,
            max_points: 20,
        }),
        Err(ScreenError::NonIncreasingInput {
            upper_name: "max_imaginary_energy",
            ..
        })
    ));
    assert!(matches!(
        screen_contour_energy_grid(ScreenContourEnergyGridInput {
            min_real_energy: -0.2,
            max_real_energy: 0.4,
            max_imaginary_energy: 0.5,
            min_imaginary_energy: 0.0,
            real_points: 4,
            imaginary_points: 4,
            max_points: 8,
        }),
        Err(ScreenError::EnergyGridTooLong {
            required: 10,
            available: 8
        })
    ));
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn assert_complex_close(
    actual: crate::Complex,
    expected_re: f64,
    expected_im: f64,
    tolerance: f64,
) {
    assert_close(actual.re, expected_re, tolerance);
    assert_close(actual.im, expected_im, tolerance);
}

fn assert_complex32_close(actual: Complex32, expected_re: f64, expected_im: f64, tolerance: f64) {
    assert_close(actual.re as f64, expected_re, tolerance);
    assert_close(actual.im as f64, expected_im, tolerance);
}

fn assert_array_close(actual: &RealVec, expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual_value, &expected_value) in actual.iter().zip(expected) {
        assert_close(actual_value, expected_value, tolerance);
    }
}
