use super::support::{assert_dmdw_close, assert_relative_close};
use super::*;

#[test]
fn spring_input_parser_matches_feff_cards() -> Result<(), DebyeError> {
    let input = parse_spring_input(
        "
*             resolution     a_cut     wmax  vdosfit
  VDOS            0.03       0.5       1.0      2.5

 * type
  STRETCHES
 * i       j    alpha(N/m)  dR_ij (%)
   0       1      27.9        2.
  ANGLES
   0       1       2       4.5       3.
  PRDOS 4
",
    )?;

    assert_eq!(input.stretches.len(), 1);
    assert_eq!(input.angles.len(), 1);
    assert_dmdw_close(input.resolution, 0.03);
    assert_dmdw_close(input.max_frequency, 0.5);
    assert_dmdw_close(input.dos_fit, 1.0);
    assert_dmdw_close(input.cutoff, 2.5);
    assert_eq!(input.print_projected_dos, 4);
    assert_eq!(input.stretches[0].first_atom, 0);
    assert_eq!(input.stretches[0].second_atom, 1);
    assert_dmdw_close(input.stretches[0].force_constant, 27.9);
    assert_dmdw_close(input.stretches[0].distance_tolerance_percent, 2.0);
    assert_eq!(input.angles[0].center_atom, 1);
    Ok(())
}

#[test]
fn spring_recursion_debye_waller_factor_updates_feff_state() -> Result<(), DebyeError> {
    let spring = parse_spring_input(
        "
VDOS 0.03 0.5 1.0 2.5
STRETCHES
0 1 27.9 2.0
1 2 12.0 2.0
END
",
    )?;
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.8, 0.0, 0.0]]);
    let atomic_numbers = [29, 30, 31];
    let potentials = [0, 1, 2];
    let matrix = spring_dynamical_matrix(SpringDynamicalMatrixInput {
        spring: &spring,
        atom_positions_angstrom: positions.view(),
        atomic_numbers: &atomic_numbers,
        potential_indices: &potentials,
        absorber_index: 0,
    })?;
    assert_relative_close(matrix.characteristic_frequency, 0.977_869_638_680_937_1);
    assert_relative_close(matrix.interaction_radius_angstrom, 2.0);

    let path = ndarray::arr2(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let result = recursion_debye_waller_factor(SpringRecursionInput {
        matrix: &matrix,
        temperature: 190.0,
        path_positions_angstrom: path.view(),
        state: None,
    })?;

    assert!(!result.fallback_used);
    assert_relative_close(result.sigma2, 0.031_891_103_846_101_55);
    assert_relative_close(result.reduced_mass, 33.246_086_960_041_21);
    assert_relative_close(result.einstein_frequency, 13.701_812_438_315_125);
    assert_relative_close(
        result
            .two_pole_frequencies
            .expect("RM poles should be present")[0],
        24.409_125_359_573_647,
    );
    assert_relative_close(
        result
            .two_pole_weights
            .expect("RM weights should be present")[1],
        0.894_773_078_763_220_9,
    );
    let mut state = SpringRecursionState::new(3);
    update_spring_recursion_state(&mut state, &matrix, path.view(), result.sigma2)?;
    assert_relative_close(state.max_sigma2, result.sigma2);
    assert_relative_close(state.pair_sigma2[(0, 2)], result.sigma2);
    assert_relative_close(state.pair_sigma2[(2, 0)], result.sigma2);
    Ok(())
}

#[test]
fn spring_equation_of_motion_debye_waller_factor_matches_feff_grid() -> Result<(), DebyeError> {
    let spring = parse_spring_input(
        "
VDOS 0.03 0.5 1.0 2.5
STRETCHES
0 1 27.9 2.0
1 2 12.0 2.0
END
",
    )?;
    let positions = ndarray::arr2(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.8, 0.0, 0.0]]);
    let atomic_numbers = [29, 30, 31];
    let potentials = [0, 1, 2];
    let matrix = spring_dynamical_matrix(SpringDynamicalMatrixInput {
        spring: &spring,
        atom_positions_angstrom: positions.view(),
        atomic_numbers: &atomic_numbers,
        potential_indices: &potentials,
        absorber_index: 0,
    })?;
    let path = ndarray::arr2(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    let result = equation_of_motion_debye_waller_factor(SpringEquationOfMotionInput {
        matrix: &matrix,
        spring: &spring,
        temperature: 190.0,
        path_positions_angstrom: path.view(),
    })?;

    assert_relative_close(result.sigma2, 0.130_988_883_045_134_37);
    assert_relative_close(result.reduced_mass, 33.246_086_960_041_21);
    assert_relative_close(result.density_normalization, 2.307_601_619_915_61);
    assert_relative_close(result.normalization_check_percent, 262.477_214_826_939_3);
    assert_relative_close(result.moment_frequency, 13.701_812_438_315_125);
    assert!(!result.capped);
    Ok(())
}
