use super::{support::*, *};
use ndarray::{Array1, Array2};

use crate::{FovrgDiracSolverInput, Real, fovrg_dirac_solver};

struct ScreenDfovrgReferenceInputs {
    exchange_cycle_count: usize,
    target_kappa: i32,
    muffin_tin_radius: Real,
    target_last_index: usize,
    energy: Complex,
    radii: Array1<Real>,
    exchange_correlation_potential: Array1<Complex>,
    valence_exchange_correlation_potential: Array1<Complex>,
    bound_large_components: Array2<Real>,
    bound_small_components: Array2<Real>,
    bound_large_coefficients: Array2<Real>,
    bound_small_coefficients: Array2<Real>,
    electron_counts: Array1<Real>,
    valence_counts: Array1<Real>,
    kappa: Array1<i32>,
    radial_match_index: usize,
    bound_orbital_count: usize,
}

impl ScreenDfovrgReferenceInputs {
    fn to_input(&self, irregular: bool) -> FovrgDiracSolverInput<'_> {
        FovrgDiracSolverInput {
            exchange_cycle_count: self.exchange_cycle_count,
            target_kappa: self.target_kappa,
            muffin_tin_radius: self.muffin_tin_radius,
            target_last_index: self.target_last_index,
            energy: self.energy,
            step: 0.45,
            radii: self.radii.view(),
            exchange_correlation_potential: self.exchange_correlation_potential.view(),
            valence_exchange_correlation_potential: self
                .valence_exchange_correlation_potential
                .view(),
            bound_large_components: self.bound_large_components.view(),
            bound_small_components: self.bound_small_components.view(),
            bound_large_coefficients: self.bound_large_coefficients.view(),
            bound_small_coefficients: self.bound_small_coefficients.view(),
            electron_counts: self.electron_counts.view(),
            valence_counts: self.valence_counts.view(),
            kappa: self.kappa.view(),
            muffin_tin_large_component: Complex::new(0.0, 0.0),
            muffin_tin_small_component: Complex::new(0.0, 0.0),
            atomic_number: 29.0,
            irregular,
            c3_scale: 0,
            radial_match_index: self.radial_match_index,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn screen_dfovrg_reference_inputs() -> ScreenDfovrgReferenceInputs {
    let count = 40;
    let bound_orbitals = 3;
    let radii = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        (-8.8 + 0.45 * (row - 1.0)).exp()
    }));
    let exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.16 + 0.006 * row, 0.002 * (0.31 * row).cos())
    }));
    let valence_exchange_correlation_potential = Array1::from_iter((1..=count).map(|row| {
        let row = row as Real;
        Complex::new(-0.12 + 0.004 * row, 0.001 * (0.27 * row).sin())
    }));
    let bound_large_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.012 * orbital * (0.08 * row * orbital).sin() * (-0.010 * row).exp()
        });
    let bound_small_components =
        Array2::from_shape_fn((count, bound_orbitals), |(row, orbital)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            0.009 * orbital * (0.07 * row * orbital).cos() * (-0.012 * row).exp()
        });
    let bound_large_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        0.008 * row + 0.0011 * orbital * (0.19 * row * orbital).cos()
    });
    let bound_small_coefficients = Array2::from_shape_fn((10, bound_orbitals), |(row, orbital)| {
        let row = (row + 1) as Real;
        let orbital = (orbital + 1) as Real;
        -0.005 * row + 0.0008 * orbital * (0.16 * row * orbital).sin()
    });

    ScreenDfovrgReferenceInputs {
        exchange_cycle_count: 0,
        target_kappa: -2,
        muffin_tin_radius: 1.42,
        target_last_index: 15,
        energy: Complex::new(0.38, 0.020),
        radii,
        exchange_correlation_potential,
        valence_exchange_correlation_potential,
        bound_large_components,
        bound_small_components,
        bound_large_coefficients,
        bound_small_coefficients,
        electron_counts: Array1::from_vec(vec![1.80, 1.00, 0.70]),
        valence_counts: Array1::from_vec(vec![0.0, 0.20, 0.0]),
        kappa: Array1::from_vec(vec![-1, 1, -2]),
        radial_match_index: 9,
        bound_orbital_count: bound_orbitals,
    }
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
fn exact_radial_continuation_tail_evaluates_bessel_hankel_rows() -> Result<(), ScreenError> {
    let radii = array![0.5, 0.75, 1.0, 1.25];
    let phase_shift = Complex::new(0.2, -0.1);
    let wave_number = Complex::new(0.6, 0.25);
    let angular_momentum = 1;

    let tail = screen_exact_radial_continuation_tail(ScreenExactRadialContinuationTailInput {
        radii: radii.view(),
        phase_shift,
        wave_number,
        angular_momentum,
        radial_match_index_1based: 2,
        active_count: 4,
    })?;

    assert_eq!(tail.start_index_1based, 2);
    assert_eq!(
        tail.rows[0],
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(0.0, 0.0),
            regular_small_component: Complex::new(0.0, 0.0),
            irregular_large_component: Complex::new(0.0, 0.0),
            irregular_small_component: Complex::new(0.0, 0.0),
        }
    );
    for index in 1..4 {
        let argument = wave_number * radii[index];
        let bessel = crate::besjn(argument, angular_momentum + 1)?;
        let hankel = crate::besjh(argument, angular_momentum + 1)?;
        let expected = screen_exact_radial_continuation(ScreenExactRadialContinuationInput {
            radius: radii[index],
            phase_shift,
            wave_number,
            bessel_j_l: bessel.j[angular_momentum],
            neumann_l: bessel.y[angular_momentum],
            bessel_j_l_plus_1: bessel.j[angular_momentum + 1],
            neumann_l_plus_1: bessel.y[angular_momentum + 1],
            hankel_l: hankel.h[angular_momentum],
            hankel_l_plus_1: hankel.h[angular_momentum + 1],
        })?;
        assert_eq!(tail.rows[index], expected);
    }
    Ok(())
}

#[test]
fn exact_radial_continuation_tail_rejects_bad_bounds() {
    let radii = array![0.5, 0.75];
    let error = screen_exact_radial_continuation_tail(ScreenExactRadialContinuationTailInput {
        radii: radii.view(),
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.6, 0.25),
        angular_momentum: 0,
        radial_match_index_1based: 3,
        active_count: 2,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        ScreenError::RadialBoundOutOfRange {
            name: "radial_match_index_1based",
            value: 3,
            capacity: 2
        }
    ));
}

#[test]
fn radial_channel_assembly_normalizes_wronskian_and_exact_tail() -> Result<(), ScreenError> {
    let regular_large = array![
        Complex::new(0.20, 0.05),
        Complex::new(0.24, 0.06),
        Complex::new(0.30, 0.02),
        Complex::new(0.33, 0.01)
    ];
    let regular_small = array![
        Complex::new(-0.010, 0.004),
        Complex::new(-0.012, 0.003),
        Complex::new(-0.015, 0.002),
        Complex::new(-0.017, 0.001)
    ];
    let irregular_large = array![
        Complex::new(0.70, -0.20),
        Complex::new(0.66, -0.18),
        Complex::new(0.61, -0.16),
        Complex::new(0.58, -0.14)
    ];
    let irregular_small = array![
        Complex::new(0.020, 0.030),
        Complex::new(0.018, 0.028),
        Complex::new(0.016, 0.026),
        Complex::new(0.014, 0.024)
    ];
    let exact = array![
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(0.0, 0.0),
            regular_small_component: Complex::new(0.0, 0.0),
            irregular_large_component: Complex::new(0.0, 0.0),
            irregular_small_component: Complex::new(0.0, 0.0),
        },
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(0.0, 0.0),
            regular_small_component: Complex::new(0.0, 0.0),
            irregular_large_component: Complex::new(0.0, 0.0),
            irregular_small_component: Complex::new(0.0, 0.0),
        },
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(1.10, -0.05),
            regular_small_component: Complex::new(0.010, -0.002),
            irregular_large_component: Complex::new(-0.20, 0.90),
            irregular_small_component: Complex::new(0.003, 0.007),
        },
        ScreenExactRadialContinuation {
            regular_large_component: Complex::new(1.20, -0.04),
            regular_small_component: Complex::new(0.011, -0.001),
            irregular_large_component: Complex::new(-0.18, 0.85),
            irregular_small_component: Complex::new(0.004, 0.006),
        }
    ];
    let wave_number = Complex::new(0.4, 0.5);
    let phase_shift = Complex::new(0.2, -0.1);
    let phase_amplitude = Complex::new(1.25, -0.4);

    let assembled = screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        exact_continuation: Some(exact.view()),
        phase_shift,
        phase_amplitude,
        wave_number,
        radial_match_index_1based: 3,
        active_count: 4,
    })?;

    let normalization = screen_solution_normalization(ScreenSolutionNormalizationInput {
        wave_number,
        phase_amplitude,
    })?;
    let match_index = 2;
    let expected_regular_match = regular_large[match_index] * normalization.regular_solution_scale;
    let expected_regular_small_match =
        regular_small[match_index] * normalization.regular_solution_scale;
    let expected_wronskian =
        screen_irregular_wronskian_scale(ScreenIrregularWronskianScaleInput {
            phase_shift,
            wave_number,
            regular_large_at_match: expected_regular_match,
            regular_small_at_match: expected_regular_small_match,
            irregular_large_at_match: irregular_large[match_index],
            irregular_small_at_match: irregular_small[match_index],
        })?;

    let expected_regular0 = regular_large[0] * normalization.regular_solution_scale;
    let expected_regular_small1 = regular_small[1] * normalization.regular_solution_scale;
    let expected_irregular0 = irregular_large[0] * expected_wronskian.irregular_solution_scale;
    let expected_irregular_small1 =
        irregular_small[1] * expected_wronskian.irregular_solution_scale;

    assert_complex_close(
        assembled.regular_large[0],
        expected_regular0.re,
        expected_regular0.im,
        1.0e-14,
    );
    assert_complex_close(
        assembled.regular_small[1],
        expected_regular_small1.re,
        expected_regular_small1.im,
        1.0e-14,
    );
    assert_complex_close(
        assembled.irregular_large[0],
        expected_irregular0.re,
        expected_irregular0.im,
        1.0e-14,
    );
    assert_complex_close(
        assembled.irregular_small[1],
        expected_irregular_small1.re,
        expected_irregular_small1.im,
        1.0e-14,
    );
    assert_eq!(assembled.regular_large[2], exact[2].regular_large_component);
    assert_eq!(assembled.regular_small[3], exact[3].regular_small_component);
    assert_eq!(
        assembled.irregular_large[2],
        exact[2].irregular_large_component
    );
    assert_eq!(
        assembled.irregular_small[3],
        exact[3].irregular_small_component
    );
    assert_eq!(assembled.normalization, normalization);
    assert_eq!(assembled.irregular_wronskian_scale, expected_wronskian);
    Ok(())
}

#[test]
fn fovrg_channel_assembly_drives_raw_solutions_and_exact_tail() -> Result<(), ScreenError> {
    let fixture = screen_dfovrg_reference_inputs();
    let regular_solver = fixture.to_input(false);
    let irregular_solver = fixture.to_input(true);
    let phase_shift = Complex::new(0.17, -0.04);
    let phase_amplitude = Complex::new(1.18, -0.22);
    let wave_number = Complex::new(0.84, 0.03);
    let angular_momentum = 1;
    let radial_match_index_1based = regular_solver.radial_match_index + 1;
    let active_count = 18;

    let channel = screen_fovrg_channel_assembly(ScreenFovrgChannelAssemblyInput {
        regular_solver,
        irregular_solver,
        phase_shift,
        phase_amplitude,
        wave_number,
        angular_momentum,
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: true,
    })?;

    let bessel_order = angular_momentum + 1;
    let muffin_tin_argument = wave_number * regular_solver.muffin_tin_radius;
    let bessel = crate::besjn(muffin_tin_argument, bessel_order)?;
    let hankel = crate::besjh(muffin_tin_argument, bessel_order)?;
    let expected_initial =
        screen_irregular_initial_condition(ScreenIrregularInitialConditionInput {
            muffin_tin_radius: regular_solver.muffin_tin_radius,
            phase_shift,
            wave_number,
            bessel_j_l: bessel.j[angular_momentum],
            neumann_l: bessel.y[angular_momentum],
            bessel_j_l_plus_1: bessel.j[bessel_order],
            neumann_l_plus_1: bessel.y[bessel_order],
            hankel_l: hankel.h[angular_momentum],
            hankel_l_plus_1: hankel.h[bessel_order],
            use_hankel_boundary: true,
        })?;
    let expected_irregular_solver = FovrgDiracSolverInput {
        muffin_tin_large_component: expected_initial.large_component,
        muffin_tin_small_component: expected_initial.small_component,
        ..irregular_solver
    };
    let expected_regular = fovrg_dirac_solver(regular_solver)?;
    let expected_irregular = fovrg_dirac_solver(expected_irregular_solver)?;
    let expected_tail =
        screen_exact_radial_continuation_tail(ScreenExactRadialContinuationTailInput {
            radii: regular_solver.radii,
            phase_shift,
            wave_number,
            angular_momentum,
            radial_match_index_1based,
            active_count,
        })?;
    let expected_assembled = screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
        regular_large: expected_regular.large_component.view(),
        regular_small: expected_regular.small_component.view(),
        irregular_large: expected_irregular.large_component.view(),
        irregular_small: expected_irregular.small_component.view(),
        exact_continuation: Some(expected_tail.rows.view()),
        phase_shift,
        phase_amplitude,
        wave_number,
        radial_match_index_1based,
        active_count,
    })?;

    assert_eq!(channel.irregular_initial_condition, expected_initial);
    assert_eq!(channel.regular_solution, expected_regular);
    assert_eq!(channel.irregular_solution, expected_irregular);
    assert_eq!(channel.exact_continuation, expected_tail);
    assert_eq!(channel.assembled, expected_assembled);
    assert_eq!(
        channel.assembled.regular_large[radial_match_index_1based - 1],
        channel.exact_continuation.rows[radial_match_index_1based - 1].regular_large_component
    );
    Ok(())
}

#[test]
fn fovrg_matched_channel_assembly_recovers_phase_amplitude_from_regular_solve()
-> Result<(), ScreenError> {
    let fixture = screen_dfovrg_reference_inputs();
    let regular_solver = FovrgDiracSolverInput {
        target_kappa: -2,
        ..fixture.to_input(false)
    };
    let irregular_solver = FovrgDiracSolverInput {
        target_kappa: -2,
        ..fixture.to_input(true)
    };
    let wave_number = Complex::new(0.84, 0.03);
    let angular_momentum = 1;
    let radial_match_index_1based = regular_solver.radial_match_index + 1;
    let active_count = 18;

    let matched = screen_fovrg_matched_channel_assembly(ScreenFovrgMatchedChannelAssemblyInput {
        regular_solver,
        irregular_solver,
        wave_number,
        angular_momentum,
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: true,
    })?;

    let regular_solution = fovrg_dirac_solver(regular_solver)?;
    let phase = crate::xsph_regular_phase(crate::XsphRegularPhaseInput {
        muffin_tin_radius: regular_solver.muffin_tin_radius,
        wave_number,
        regular_large_at_muffin_tin: regular_solution.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular_solution.muffin_tin_small_component,
        kappa: regular_solver.target_kappa,
    })?;
    let explicit = screen_fovrg_channel_assembly(ScreenFovrgChannelAssemblyInput {
        regular_solver,
        irregular_solver,
        phase_shift: phase.phase_shift,
        phase_amplitude: phase.phase_amplitude,
        wave_number,
        angular_momentum,
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: true,
    })?;

    assert_eq!(matched.phase_shift, phase.phase_shift);
    assert_eq!(matched.phase_amplitude, phase.phase_amplitude);
    assert_eq!(matched, explicit);
    Ok(())
}

#[test]
fn fovrg_channel_assembly_rejects_regular_solver_branch_mismatch() {
    let fixture = screen_dfovrg_reference_inputs();
    let error = screen_fovrg_channel_assembly(ScreenFovrgChannelAssemblyInput {
        regular_solver: fixture.to_input(true),
        irregular_solver: fixture.to_input(true),
        phase_shift: Complex::new(0.17, -0.04),
        phase_amplitude: Complex::new(1.18, -0.22),
        wave_number: Complex::new(0.84, 0.03),
        angular_momentum: 1,
        radial_match_index_1based: 10,
        active_count: 18,
        use_hankel_boundary: false,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        ScreenError::FovrgSolverBranchMismatch {
            name: "regular_solver",
            expected: false,
            actual: true
        }
    ));
}

#[test]
fn fovrg_cube_assembly_loops_solver_grid_in_energy_angular_order() -> Result<(), ScreenError> {
    let fixture = screen_dfovrg_reference_inputs();
    let energy_count = 2;
    let angular_count = 2;
    let active_count = 18;
    let phase_shifts = array![
        [Complex::new(0.17, -0.04), Complex::new(0.21, -0.03)],
        [Complex::new(0.19, -0.02), Complex::new(0.24, -0.01)]
    ];
    let phase_amplitudes = array![
        [Complex::new(1.18, -0.22), Complex::new(1.12, -0.18)],
        [Complex::new(1.15, -0.20), Complex::new(1.09, -0.16)]
    ];
    let wave_numbers = array![Complex::new(0.84, 0.03), Complex::new(0.90, 0.04)];

    let mut regular_solvers = Vec::new();
    let mut irregular_solvers = Vec::new();
    for energy_index in 0..energy_count {
        for angular_momentum in 0..angular_count {
            let solver_energy = Complex::new(
                fixture.energy.re + 0.015 * energy_index as Real,
                fixture.energy.im + 0.004 * angular_momentum as Real,
            );
            regular_solvers.push(FovrgDiracSolverInput {
                energy: solver_energy,
                ..fixture.to_input(false)
            });
            irregular_solvers.push(FovrgDiracSolverInput {
                energy: solver_energy,
                ..fixture.to_input(true)
            });
        }
    }

    let radial_match_index_1based = regular_solvers[0].radial_match_index + 1;
    let cubes = screen_fovrg_cube_assembly(ScreenFovrgCubeAssemblyInput {
        regular_solvers: &regular_solvers,
        irregular_solvers: &irregular_solvers,
        phase_shifts: phase_shifts.view(),
        phase_amplitudes: phase_amplitudes.view(),
        wave_numbers: wave_numbers.view(),
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: true,
    })?;

    assert_eq!(
        cubes.radial_cubes.regular_large.dim(),
        (energy_count, active_count, angular_count)
    );
    for energy_index in 0..energy_count {
        for angular_momentum in 0..angular_count {
            let solver_index = energy_index * angular_count + angular_momentum;
            let expected = screen_fovrg_channel_assembly(ScreenFovrgChannelAssemblyInput {
                regular_solver: regular_solvers[solver_index],
                irregular_solver: irregular_solvers[solver_index],
                phase_shift: phase_shifts[(energy_index, angular_momentum)],
                phase_amplitude: phase_amplitudes[(energy_index, angular_momentum)],
                wave_number: wave_numbers[energy_index],
                angular_momentum,
                radial_match_index_1based,
                active_count,
                use_hankel_boundary: true,
            })?;

            assert_eq!(
                cubes.irregular_initial_large[(energy_index, angular_momentum)],
                expected.irregular_initial_condition.large_component
            );
            assert_eq!(
                cubes.irregular_initial_small[(energy_index, angular_momentum)],
                expected.irregular_initial_condition.small_component
            );
            assert_eq!(
                cubes.regular_iteration_counts[(energy_index, angular_momentum)],
                expected.regular_solution.iteration_count
            );
            assert_eq!(
                cubes.irregular_iteration_counts[(energy_index, angular_momentum)],
                expected.irregular_solution.iteration_count
            );
            assert_eq!(
                cubes.difficult_iterations[(energy_index, angular_momentum)],
                expected.regular_solution.difficult_iterations
                    + expected.irregular_solution.difficult_iterations
            );
            for radial_index in 0..active_count {
                assert_eq!(
                    cubes.radial_cubes.regular_large
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.regular_large[radial_index]
                );
                assert_eq!(
                    cubes.radial_cubes.regular_small
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.regular_small[radial_index]
                );
                assert_eq!(
                    cubes.radial_cubes.irregular_large
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.irregular_large[radial_index]
                );
                assert_eq!(
                    cubes.radial_cubes.irregular_small
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.irregular_small[radial_index]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fovrg_matched_cube_assembly_loops_and_returns_phase_tables() -> Result<(), ScreenError> {
    let fixture = screen_dfovrg_reference_inputs();
    let energy_count = 2;
    let angular_count = 2;
    let active_count = 18;
    let wave_numbers = array![Complex::new(0.84, 0.03), Complex::new(0.90, 0.04)];
    let mut regular_solvers = Vec::new();
    let mut irregular_solvers = Vec::new();
    for energy_index in 0..energy_count {
        for angular_momentum in 0..angular_count {
            let solver_energy = Complex::new(
                fixture.energy.re + 0.015 * energy_index as Real,
                fixture.energy.im + 0.004 * angular_momentum as Real,
            );
            let target_kappa = -((angular_momentum as i32) + 1);
            regular_solvers.push(FovrgDiracSolverInput {
                target_kappa,
                energy: solver_energy,
                ..fixture.to_input(false)
            });
            irregular_solvers.push(FovrgDiracSolverInput {
                target_kappa,
                energy: solver_energy,
                ..fixture.to_input(true)
            });
        }
    }
    let radial_match_index_1based = regular_solvers[0].radial_match_index + 1;

    let cubes = screen_fovrg_matched_cube_assembly(ScreenFovrgMatchedCubeAssemblyInput {
        regular_solvers: &regular_solvers,
        irregular_solvers: &irregular_solvers,
        wave_numbers: wave_numbers.view(),
        angular_count,
        radial_match_index_1based,
        active_count,
        use_hankel_boundary: true,
    })?;

    assert_eq!(
        cubes.solved.radial_cubes.regular_large.dim(),
        (energy_count, active_count, angular_count)
    );
    assert_eq!(cubes.phase_shifts.dim(), (energy_count, angular_count));
    assert_eq!(cubes.phase_amplitudes.dim(), (energy_count, angular_count));

    for energy_index in 0..energy_count {
        for angular_momentum in 0..angular_count {
            let solver_index = energy_index * angular_count + angular_momentum;
            let expected =
                screen_fovrg_matched_channel_assembly(ScreenFovrgMatchedChannelAssemblyInput {
                    regular_solver: regular_solvers[solver_index],
                    irregular_solver: irregular_solvers[solver_index],
                    wave_number: wave_numbers[energy_index],
                    angular_momentum,
                    radial_match_index_1based,
                    active_count,
                    use_hankel_boundary: true,
                })?;

            assert_eq!(
                cubes.phase_shifts[(energy_index, angular_momentum)],
                expected.phase_shift
            );
            assert_eq!(
                cubes.phase_amplitudes[(energy_index, angular_momentum)],
                expected.phase_amplitude
            );
            assert_eq!(
                cubes.solved.irregular_initial_large[(energy_index, angular_momentum)],
                expected.irregular_initial_condition.large_component
            );
            for radial_index in 0..active_count {
                assert_eq!(
                    cubes.solved.radial_cubes.regular_large
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.regular_large[radial_index]
                );
                assert_eq!(
                    cubes.solved.radial_cubes.irregular_small
                        [(energy_index, radial_index, angular_momentum)],
                    expected.assembled.irregular_small[radial_index]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fovrg_cube_assembly_rejects_solver_count_mismatch() {
    let fixture = screen_dfovrg_reference_inputs();
    let regular_solvers = vec![fixture.to_input(false), fixture.to_input(false)];
    let irregular_solvers = vec![fixture.to_input(true)];
    let phase_shifts = array![[Complex::new(0.17, -0.04), Complex::new(0.21, -0.03)]];
    let phase_amplitudes = array![[Complex::new(1.18, -0.22), Complex::new(1.12, -0.18)]];
    let wave_numbers = array![Complex::new(0.84, 0.03)];

    let error = screen_fovrg_cube_assembly(ScreenFovrgCubeAssemblyInput {
        regular_solvers: &regular_solvers,
        irregular_solvers: &irregular_solvers,
        phase_shifts: phase_shifts.view(),
        phase_amplitudes: phase_amplitudes.view(),
        wave_numbers: wave_numbers.view(),
        radial_match_index_1based: 10,
        active_count: 18,
        use_hankel_boundary: false,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        ScreenError::FovrgSolverCountMismatch {
            expected: 2,
            regular: 2,
            irregular: 1
        }
    ));
}

#[test]
fn radial_channel_assembly_rejects_short_raw_solutions() {
    let short = array![Complex::new(0.1, 0.0)];
    let full = array![Complex::new(0.1, 0.0), Complex::new(0.2, 0.0)];

    let error = screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
        regular_large: short.view(),
        regular_small: full.view(),
        irregular_large: full.view(),
        irregular_small: full.view(),
        exact_continuation: None,
        phase_shift: Complex::new(0.1, 0.0),
        phase_amplitude: Complex::new(1.0, 0.0),
        wave_number: Complex::new(0.4, 0.1),
        radial_match_index_1based: 1,
        active_count: 2,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        ScreenError::ActiveCountOutOfRange {
            active_count: 2,
            len: 1
        }
    ));
}

#[test]
fn radial_cube_assembly_lifts_channel_assembly_over_energy_and_angular_grid()
-> Result<(), ScreenError> {
    let energy_count = 2;
    let active_count = 4;
    let angular_count = 2;
    let regular_large =
        ndarray::Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
            let scale = (e + 1) as f64 * (r + 1) as f64 * (l + 1) as f64;
            Complex::new(0.10 * scale, 0.02 * scale)
        });
    let regular_small =
        ndarray::Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
            let scale = (e + 1) as f64 * (r + 1) as f64 * (l + 1) as f64;
            Complex::new(-0.004 * scale, 0.001 * scale)
        });
    let irregular_large =
        ndarray::Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
            let scale = (e + 1) as f64 * (r + 1) as f64 * (l + 1) as f64;
            Complex::new(0.30 * scale, -0.04 * scale)
        });
    let irregular_small =
        ndarray::Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
            let scale = (e + 1) as f64 * (r + 1) as f64 * (l + 1) as f64;
            Complex::new(0.006 * scale, 0.002 * scale)
        });
    let exact =
        ndarray::Array3::from_shape_fn((energy_count, active_count, angular_count), |(e, r, l)| {
            if r < 2 {
                return ScreenExactRadialContinuation {
                    regular_large_component: Complex::new(0.0, 0.0),
                    regular_small_component: Complex::new(0.0, 0.0),
                    irregular_large_component: Complex::new(0.0, 0.0),
                    irregular_small_component: Complex::new(0.0, 0.0),
                };
            }
            let scale = (e + 1) as f64 * (r + 1) as f64 * (l + 1) as f64;
            ScreenExactRadialContinuation {
                regular_large_component: Complex::new(1.0 + scale * 0.01, -0.03 * scale),
                regular_small_component: Complex::new(0.001 * scale, -0.0002 * scale),
                irregular_large_component: Complex::new(-0.2 * scale, 0.8 + 0.04 * scale),
                irregular_small_component: Complex::new(0.0003 * scale, 0.0007 * scale),
            }
        });
    let phase_shifts = array![
        [Complex::new(0.20, -0.10), Complex::new(0.25, -0.05)],
        [Complex::new(0.18, -0.08), Complex::new(0.22, -0.04)]
    ];
    let phase_amplitudes = array![
        [Complex::new(1.10, -0.20), Complex::new(1.05, -0.15)],
        [Complex::new(1.20, -0.25), Complex::new(1.15, -0.18)]
    ];
    let wave_numbers = array![Complex::new(0.4, 0.5), Complex::new(0.45, 0.35)];

    let cubes = screen_radial_cube_assembly(ScreenRadialCubeAssemblyInput {
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        irregular_large: irregular_large.view(),
        irregular_small: irregular_small.view(),
        exact_continuation: Some(exact.view()),
        phase_shifts: phase_shifts.view(),
        phase_amplitudes: phase_amplitudes.view(),
        wave_numbers: wave_numbers.view(),
        radial_match_index_1based: 3,
        active_count,
    })?;

    assert_eq!(
        cubes.regular_large.dim(),
        (energy_count, active_count, angular_count)
    );
    for energy in 0..energy_count {
        let regular_large_energy = regular_large.index_axis(ndarray::Axis(0), energy);
        let regular_small_energy = regular_small.index_axis(ndarray::Axis(0), energy);
        let irregular_large_energy = irregular_large.index_axis(ndarray::Axis(0), energy);
        let irregular_small_energy = irregular_small.index_axis(ndarray::Axis(0), energy);
        let exact_energy = exact.index_axis(ndarray::Axis(0), energy);
        for angular in 0..angular_count {
            let expected = screen_radial_channel_assembly(ScreenRadialChannelAssemblyInput {
                regular_large: regular_large_energy.index_axis(ndarray::Axis(1), angular),
                regular_small: regular_small_energy.index_axis(ndarray::Axis(1), angular),
                irregular_large: irregular_large_energy.index_axis(ndarray::Axis(1), angular),
                irregular_small: irregular_small_energy.index_axis(ndarray::Axis(1), angular),
                exact_continuation: Some(exact_energy.index_axis(ndarray::Axis(1), angular)),
                phase_shift: phase_shifts[(energy, angular)],
                phase_amplitude: phase_amplitudes[(energy, angular)],
                wave_number: wave_numbers[energy],
                radial_match_index_1based: 3,
                active_count,
            })?;
            for radial in 0..active_count {
                assert_eq!(
                    cubes.regular_large[(energy, radial, angular)],
                    expected.regular_large[radial]
                );
                assert_eq!(
                    cubes.regular_small[(energy, radial, angular)],
                    expected.regular_small[radial]
                );
                assert_eq!(
                    cubes.irregular_large[(energy, radial, angular)],
                    expected.irregular_large[radial]
                );
                assert_eq!(
                    cubes.irregular_small[(energy, radial, angular)],
                    expected.irregular_small[radial]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn radial_cube_assembly_rejects_short_channel_axes() {
    let full = ndarray::Array3::from_elem((1, 2, 1), Complex::new(0.1, 0.0));
    let short_radial = ndarray::Array3::from_elem((1, 1, 1), Complex::new(0.1, 0.0));
    let phase = array![[Complex::new(0.1, 0.0)]];
    let wave_numbers = array![Complex::new(0.4, 0.1)];

    let error = screen_radial_cube_assembly(ScreenRadialCubeAssemblyInput {
        regular_large: full.view(),
        regular_small: full.view(),
        irregular_large: short_radial.view(),
        irregular_small: full.view(),
        exact_continuation: None,
        phase_shifts: phase.view(),
        phase_amplitudes: phase.view(),
        wave_numbers: wave_numbers.view(),
        radial_match_index_1based: 1,
        active_count: 2,
    })
    .unwrap_err();

    assert!(matches!(
        error,
        ScreenError::CountTooSmall {
            name: "irregular_large",
            actual: 1,
            minimum: 2
        }
    ));
}
