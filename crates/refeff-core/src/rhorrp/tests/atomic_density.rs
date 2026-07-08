use super::{support::*, *};
use crate::{
    DiracSpinorOrbitalsGridInput, FovrgDiracSolverInput, PotentialGridInput,
    fix_dirac_spinor_orbitals_grid, fix_potential_grid, fovrg_dirac_solver, radial_radius,
};

#[test]
fn fix_irregular_origin_matches_feff_reference() -> Result<(), RhorrpError> {
    let (radii, values) = reference_irregular_solution();
    let fixed = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
        radii: &radii,
        values: values.view(),
    })?;

    assert_complex_close_tol(
        fixed[0],
        Complex::new(9.791_151_469_085_387, 3.741_459_448_683_99),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[49],
        Complex::new(-2.047_179_619_930_901_1e-1, -8.434_737_680_311_137e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[74],
        Complex::new(-6.916_158_567_064_077e-1, -8.929_639_586_361_882e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[99],
        Complex::new(8.811_645_823_831e-1, 1.866_102_289_679_183_5e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[100],
        Complex::new(9.101_077_089_878_837e-1, 2.302_339_202_367_545e-1),
        1.0e-8,
    );
    assert_complex_close_tol(
        fixed[119],
        Complex::new(1.094_598_908_088_280_5, 8.401_702_866_503_66e-1),
        1.0e-8,
    );
    Ok(())
}

#[test]
fn fix_irregular_origin_rejects_invalid_inputs() {
    let (radii, values) = reference_irregular_solution();
    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..99],
            values: values.slice_axis(Axis(0), Slice::from(..99)),
        }),
        Err(RhorrpError::InsufficientIrregularFixPoints {
            points: 99,
            required: 100,
        })
    ));

    assert!(matches!(
        rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
            radii: &radii[..100],
            values: values.view(),
        }),
        Err(RhorrpError::IrregularFixLengthMismatch {
            radii: 100,
            values: 120,
        })
    ));
}

#[test]
fn potential_reference_shift_matches_feff_init_wavefunctions() -> Result<(), RhorrpError> {
    let total = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    let valence = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let shifted = rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
        muffin_tin_radius: 1.0,
        radial_x0: 0.7,
        radial_dx: 0.2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        exchange_index: 5,
    })?;

    assert_eq!(shifted.reference_index_1based, 6);
    assert_complex_close(shifted.reference_energy_hartree, Complex::new(15.0, 0.0));
    assert_eq!(
        shifted.total_potential,
        Array1::from_vec(vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 16.0, 17.0])
    );
    assert_eq!(
        shifted.valence_potential,
        Array1::from_vec(vec![-14.0, -13.0, -12.0, -11.0, -10.0, -9.0, 7.0, 8.0])
    );
    Ok(())
}

#[test]
fn potential_reference_shift_copies_total_for_non_energy_dependent_exchange()
-> Result<(), RhorrpError> {
    let total = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    let valence = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let shifted = rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
        muffin_tin_radius: 1.0,
        radial_x0: 0.7,
        radial_dx: 0.2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        exchange_index: 4,
    })?;

    assert_eq!(
        shifted.valence_potential,
        Array1::from_vec(vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 7.0, 8.0])
    );
    Ok(())
}

#[test]
fn potential_reference_shifts_match_feff_init_wavefunctions_potential_loop()
-> Result<(), RhorrpError> {
    let total = arr2(&[
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0],
        [13.0, 23.0],
        [14.0, 24.0],
        [15.0, 25.0],
        [16.0, 26.0],
    ]);
    let valence = arr2(&[
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0],
        [6.0, 12.0],
        [7.0, 14.0],
    ]);

    let shifted = rhorrp_potential_reference_shifts(RhorrpPotentialReferenceShiftsInput {
        muffin_tin_radii: &[1.0, 0.82],
        radial_x0: 0.7,
        radial_dx: 0.2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        exchange_index: 5,
    })?;

    assert_eq!(shifted.reference_indices_1based, vec![6, 5]);
    assert_eq!(
        shifted.reference_energies_hartree,
        Array1::from_vec(vec![Complex::new(15.0, 0.0), Complex::new(24.0, 0.0)])
    );
    assert_eq!(
        shifted.total_potential,
        arr2(&[
            [-5.0, -4.0],
            [-4.0, -3.0],
            [-3.0, -2.0],
            [-2.0, -1.0],
            [-1.0, 0.0],
            [0.0, 25.0],
            [16.0, 26.0],
        ])
    );
    assert_eq!(
        shifted.valence_potential,
        arr2(&[
            [-14.0, -22.0],
            [-13.0, -20.0],
            [-12.0, -18.0],
            [-11.0, -16.0],
            [-10.0, -14.0],
            [-9.0, 12.0],
            [7.0, 14.0],
        ])
    );
    Ok(())
}

#[test]
fn wavefunction_grid_preparation_matches_feff_init_wavefunctions_setup() -> Result<(), RhorrpError>
{
    let source_rows = 12;
    let potentials = 2;
    let orbitals = 3;
    let dx = 0.05;
    let muffin_tin_radii = [radial_radius(4, dx), radial_radius(5, dx)];
    let electron_density = Array2::from_shape_fn((source_rows, potentials), |(row, potential)| {
        4.0 * std::f64::consts::PI * (0.15 + 0.01 * row as Real + 0.02 * potential as Real)
    });
    let valence_density = Array2::from_shape_fn((source_rows, potentials), |(row, potential)| {
        4.0 * std::f64::consts::PI * (0.07 + 0.006 * row as Real + 0.015 * potential as Real)
    });
    let total_potential = Array2::from_shape_fn((source_rows, potentials), |(row, potential)| {
        -1.60 + 0.035 * row as Real + 0.24 * potential as Real
    });
    let valence_potential = Array2::from_shape_fn((source_rows, potentials), |(row, potential)| {
        -1.10 + 0.020 * row as Real + 0.18 * potential as Real
    });
    let magnetization = Array2::from_shape_fn((source_rows, potentials), |(row, potential)| {
        0.002 * (row + 1) as Real - 0.001 * potential as Real
    });
    let bound_large = Array3::from_shape_fn(
        (source_rows, orbitals, potentials),
        |(row, orbital, potential)| {
            if row < 8 + orbital + potential {
                0.01 * (orbital + 1) as Real * (0.2 * (row + 1) as Real).sin()
            } else {
                0.0
            }
        },
    );
    let bound_small = Array3::from_shape_fn(
        (source_rows, orbitals, potentials),
        |(row, orbital, potential)| {
            if row < 8 + orbital + potential {
                0.008 * (orbital + 1) as Real * (0.17 * (row + 1) as Real).cos()
            } else {
                0.0
            }
        },
    );

    let prepared = rhorrp_prepare_wavefunction_grids(RhorrpWavefunctionGridPreparationInput {
        muffin_tin_radii: &muffin_tin_radii,
        electron_density: electron_density.view(),
        total_potential: total_potential.view(),
        valence_density: valence_density.view(),
        valence_potential: valence_potential.view(),
        magnetization: magnetization.view(),
        bound_large_components: bound_large.view(),
        bound_small_components: bound_small.view(),
        interstitial_potential: -0.80,
        interstitial_density: 0.21,
        original_radial_dx: dx,
        target_radial_dx: dx,
        jump_mode: 1,
        potential_jump: 0.0,
        exchange_index: 5,
        radial_count: source_rows,
    })?;

    assert_eq!(prepared.potential_count(), potentials);
    assert_eq!(prepared.radial_count(), source_rows);
    assert_eq!(prepared.orbital_count(), orbitals);
    assert_real_close_scaled(prepared.radial_dx, dx);

    for (potential, &muffin_tin_radius) in muffin_tin_radii.iter().enumerate().take(potentials) {
        let total_grid = fix_potential_grid(PotentialGridInput {
            muffin_tin_radius,
            electron_density: electron_density.column(potential),
            total_potential: total_potential.column(potential),
            magnetization: magnetization.column(potential),
            interstitial_potential: -0.80,
            interstitial_density: 0.21,
            original_delta: dx,
            new_delta: dx,
            jump_mode: 1,
            potential_jump: 0.0,
            output_len: source_rows,
        })?;
        let valence_grid = fix_potential_grid(PotentialGridInput {
            muffin_tin_radius,
            electron_density: valence_density.column(potential),
            total_potential: valence_potential.column(potential),
            magnetization: magnetization.column(potential),
            interstitial_potential: -0.80,
            interstitial_density: 0.21,
            original_delta: dx,
            new_delta: dx,
            jump_mode: 2,
            potential_jump: total_grid.potential_jump,
            output_len: source_rows,
        })?;
        let shifted = rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
            muffin_tin_radius,
            radial_x0: 8.8,
            radial_dx: dx,
            total_potential: total_grid.total_potential.view(),
            valence_potential: valence_grid.total_potential.view(),
            exchange_index: 5,
        })?;
        let spinors = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: dx,
            new_delta: dx,
            large_components: bound_large.index_axis(Axis(2), potential),
            small_components: bound_small.index_axis(Axis(2), potential),
            output_len: source_rows,
        })?;

        assert_real_close(
            prepared.potential_jumps[potential],
            total_grid.potential_jump,
        );
        assert_eq!(
            prepared.reference_indices_1based[potential],
            shifted.reference_index_1based
        );
        assert_complex_close(
            prepared.reference_energies_hartree[potential],
            shifted.reference_energy_hartree,
        );
        for row in [0, shifted.reference_index_1based - 1, source_rows - 1] {
            assert_complex_close(
                prepared.total_potential[(row, potential)],
                Complex::new(shifted.total_potential[row], 0.0),
            );
            assert_complex_close(
                prepared.valence_potential[(row, potential)],
                Complex::new(shifted.valence_potential[row], 0.0),
            );
        }
        for orbital in 0..orbitals {
            assert_eq!(
                prepared.bound_active_lengths[(orbital, potential)],
                spinors.active_lengths[orbital]
            );
            assert_real_close(
                prepared.bound_large_components[(0, orbital, potential)],
                spinors.large_components[(0, orbital)],
            );
            assert_real_close(
                prepared.bound_small_components[(source_rows - 1, orbital, potential)],
                spinors.small_components[(source_rows - 1, orbital)],
            );
        }
    }
    Ok(())
}

#[test]
fn potential_reference_shift_rejects_invalid_inputs() {
    let total = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let valence = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    assert!(matches!(
        rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
            muffin_tin_radius: 0.0,
            radial_x0: 0.7,
            radial_dx: 0.2,
            total_potential: total.view(),
            valence_potential: valence.view(),
            exchange_index: 5,
        }),
        Err(RhorrpError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
            muffin_tin_radius: 1.0,
            radial_x0: 0.7,
            radial_dx: 0.2,
            total_potential: total.view(),
            valence_potential: valence.slice_axis(Axis(0), Slice::from(..2)),
            exchange_index: 5,
        }),
        Err(RhorrpError::PotentialReferenceShiftLengthMismatch {
            total: 3,
            valence: 2,
        })
    ));

    assert!(matches!(
        rhorrp_potential_reference_shift(RhorrpPotentialReferenceShiftInput {
            muffin_tin_radius: 10.0,
            radial_x0: 0.7,
            radial_dx: 0.2,
            total_potential: total.view(),
            valence_potential: valence.view(),
            exchange_index: 5,
        }),
        Err(RhorrpError::PotentialReferenceIndexOutOfRange {
            radial_count: 3,
            ..
        })
    ));

    let empty_total = Array2::<Real>::zeros((3, 0));
    let empty_valence = Array2::<Real>::zeros((3, 0));
    assert!(matches!(
        rhorrp_potential_reference_shifts(RhorrpPotentialReferenceShiftsInput {
            muffin_tin_radii: &[],
            radial_x0: 0.7,
            radial_dx: 0.2,
            total_potential: empty_total.view(),
            valence_potential: empty_valence.view(),
            exchange_index: 5,
        }),
        Err(RhorrpError::InvalidPotentialReferencePotentialCount { potential_count: 0 })
    ));

    let total_matrix = Array2::<Real>::zeros((3, 2));
    let valence_matrix = Array2::<Real>::zeros((3, 1));
    assert!(matches!(
        rhorrp_potential_reference_shifts(RhorrpPotentialReferenceShiftsInput {
            muffin_tin_radii: &[1.0, 1.1],
            radial_x0: 0.7,
            radial_dx: 0.2,
            total_potential: total_matrix.view(),
            valence_potential: valence_matrix.view(),
            exchange_index: 5,
        }),
        Err(RhorrpError::PotentialReferenceShiftShapeMismatch {
            total_radial: 3,
            total_potentials: 2,
            valence_radial: 3,
            valence_potentials: 1,
            muffin_tin_radii: 2,
        })
    ));
}

#[test]
fn wavefunction_setup_matches_feff_init_wavefunctions() -> Result<(), RhorrpError> {
    let setup = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
        energy_hartree: Complex::new(0.4, 0.5),
        reference_energy_hartree: Complex::new(0.1, 0.05),
        muffin_tin_radius: 1.7,
        norman_radius: 1.2,
        radial_x0: 8.8,
        radial_dx: 0.05,
        radial_capacity: 251,
        exchange_index: 7,
    })?;

    assert_eq!(setup.last_integration_index_1based, 187);
    assert_eq!(setup.dirac_cycle_count, 3);
    assert_complex_close(setup.kinetic_energy_hartree, Complex::new(0.3, 0.45));
    assert_complex_close_tol(
        setup.wave_number,
        Complex::new(0.916_970_019_128_716_1, 0.490_754_528_006_756_5),
        1.0e-14,
    );
    assert_complex_close_tol(
        setup.muffin_tin_wave_number,
        Complex::new(1.558_849_032_518_817_3, 0.834_282_697_611_486),
        1.0e-14,
    );

    let low_exchange = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
        exchange_index: 14,
        ..RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(0.4, 0.5),
            reference_energy_hartree: Complex::new(0.1, 0.05),
            muffin_tin_radius: 1.7,
            norman_radius: 1.2,
            radial_x0: 8.8,
            radial_dx: 0.05,
            radial_capacity: 251,
            exchange_index: 7,
        }
    })?;
    assert_eq!(low_exchange.dirac_cycle_count, 0);

    let clamped = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
        norman_radius: 38.474_666_049_032_14,
        ..RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(0.4, 0.5),
            reference_energy_hartree: Complex::new(0.1, 0.05),
            muffin_tin_radius: 1.7,
            norman_radius: 1.2,
            radial_x0: 8.8,
            radial_dx: 0.05,
            radial_capacity: 251,
            exchange_index: 7,
        }
    })?;
    assert_eq!(clamped.last_integration_index_1based, 251);
    Ok(())
}

#[test]
fn wavefunction_setup_rejects_invalid_inputs() {
    assert!(matches!(
        rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(f64::NAN, 0.0),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.0,
            norman_radius: 1.0,
            radial_x0: 8.8,
            radial_dx: 0.05,
            radial_capacity: 251,
            exchange_index: 0,
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "wavefunction_energy_hartree.real",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(0.0, 0.0),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.0,
            norman_radius: 0.0,
            radial_x0: 8.8,
            radial_dx: 0.05,
            radial_capacity: 251,
            exchange_index: 0,
        }),
        Err(RhorrpError::InvalidPositiveRadius {
            name: "norman_radius",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(0.0, 0.0),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.0,
            norman_radius: 1.0,
            radial_x0: 8.8,
            radial_dx: 0.05,
            radial_capacity: 0,
            exchange_index: 0,
        }),
        Err(RhorrpError::InvalidRadialCount { radial_count: 0 })
    ));

    assert!(matches!(
        rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: Complex::new(0.0, 0.0),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            muffin_tin_radius: 1.0,
            norman_radius: 1.0,
            radial_x0: -20.0,
            radial_dx: 1.0,
            radial_capacity: 251,
            exchange_index: 0,
        }),
        Err(RhorrpError::WavefunctionSetupIndexOutOfRange {
            radial_capacity: 251,
            ..
        })
    ));
}

#[test]
fn muffin_tin_match_composes_feff_init_wavefunctions_sequence() -> Result<(), RhorrpError> {
    let matched = rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
        muffin_tin_radius: 1.7,
        wave_number: Complex::new(1.1, 0.15),
        angular_momentum: 2,
        regular_large_at_muffin_tin: Complex::new(0.8, 0.2),
        regular_small_at_muffin_tin: Complex::new(-0.3, 0.4),
        kappa: -2,
    })?;

    assert_complex_close_tol(
        matched.muffin_tin_wave_number,
        Complex::new(1.87, 0.255),
        1.0e-15,
    );
    assert_complex_close_tol(
        matched.bessel_j_l,
        Complex::new(0.180_997_622_266_464_1, 0.036_476_724_461_219_01),
        1.0e-15,
    );
    assert_complex_close_tol(
        matched.neumann_l,
        Complex::new(-0.792_852_028_987_480_5, 0.223_907_623_892_841_88),
        1.0e-15,
    );
    assert_complex_close_tol(
        matched.bessel_j_l_plus_1,
        Complex::new(0.049_515_580_781_725_854, 0.018_167_305_853_358_465),
        1.0e-15,
    );
    assert_complex_close_tol(
        matched.neumann_l_plus_1,
        Complex::new(-1.589_298_128_172_463_2, 0.717_274_179_389_014_9),
        1.0e-14,
    );
    assert_complex_close_tol(
        matched.phase_shift,
        Complex::new(2.942_804_262_966_890_7, -0.101_383_444_078_318_16),
        1.0e-14,
    );
    assert_complex_close_tol(
        matched.phase_amplitude,
        Complex::new(105.165_268_721_644, -187.519_881_928_796_3),
        1.0e-12,
    );
    assert_complex_close_tol(
        matched.regular_solution_scale,
        Complex::new(0.002_275_150_205_854_981_6, 0.004_056_813_653_007_83),
        1.0e-15,
    );
    Ok(())
}

#[test]
fn muffin_tin_match_rejects_invalid_inputs() {
    assert!(matches!(
        rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
            muffin_tin_radius: 0.0,
            wave_number: Complex::new(1.1, 0.15),
            angular_momentum: 2,
            regular_large_at_muffin_tin: Complex::new(0.8, 0.2),
            regular_small_at_muffin_tin: Complex::new(-0.3, 0.4),
            kappa: -2,
        }),
        Err(RhorrpError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
            muffin_tin_radius: 1.7,
            wave_number: Complex::new(1.1, 0.15),
            angular_momentum: 9,
            regular_large_at_muffin_tin: Complex::new(0.8, 0.2),
            regular_small_at_muffin_tin: Complex::new(-0.3, 0.4),
            kappa: -2,
        }),
        Err(RhorrpError::BesselEvaluation {
            source: crate::BesselError::ExactOrderOutOfRange {
                order: 10,
                max_order: 9,
            },
        })
    ));
}

#[test]
fn radial_solution_scalars_match_feff_init_wavefunctions() -> Result<(), RhorrpError> {
    let scale = rhorrp_regular_solution_scale(RhorrpRegularSolutionScaleInput {
        phase_amplitude: Complex::new(1.25, -0.4),
    })?;
    assert_complex_close_tol(
        scale.scale,
        Complex::new(0.725_689_404_934_687_9, 0.232_220_609_579_100_12),
        1.0e-16,
    );

    let irregular = rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
        muffin_tin_radius: 1.7,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.8, 0.1),
        neumann_l: Complex::new(-0.3, 0.05),
        bessel_j_l_plus_1: Complex::new(0.25, -0.03),
        neumann_l_plus_1: Complex::new(-0.6, 0.2),
    })?;
    assert_complex_close_tol(
        irregular.large_component,
        Complex::new(-0.215_795_431_247_046_34, -0.025_995_014_748_418_55),
        1.0e-16,
    );
    assert_complex_close_tol(
        irregular.small_component,
        Complex::new(0.001_838_861_639_564_406_4, 0.001_316_136_108_948_062),
        1.0e-17,
    );

    let wronskian = rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        regular_large_at_match: Complex::new(0.3, 0.2),
        regular_small_at_match: Complex::new(-0.01, 0.04),
        irregular_large_at_match: Complex::new(0.7, -0.2),
        irregular_small_at_match: Complex::new(0.02, 0.03),
    })?;
    assert_complex_close_tol(
        wronskian.phase_factor,
        Complex::new(1.083_141_079_608_063_2, 0.219_563_566_708_252_36),
        1.0e-15,
    );
    assert_complex_close_tol(
        wronskian.denominator,
        Complex::new(-0.726_137_142_242_051_2, 5.106_772_750_294_418),
        1.0e-14,
    );
    assert_complex_close_tol(
        wronskian.reciprocal_wave_scale,
        Complex::new(-0.260_696_573_980_254_4, -0.153_973_620_782_305_84),
        1.0e-15,
    );

    let transformed = rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
        phase_factor: wronskian.phase_factor,
        reciprocal_wave_scale: wronskian.reciprocal_wave_scale,
        regular_large_component: Complex::new(0.11, 0.22),
        regular_small_component: Complex::new(-0.04, 0.05),
        irregular_large_component: Complex::new(0.6, -0.15),
        irregular_small_component: Complex::new(0.03, 0.08),
    })?;
    assert_complex_close_tol(
        transformed.large_component,
        Complex::new(-0.037_259_303_741_555_2, 0.207_124_148_389_249_03),
        1.0e-15,
    );
    assert_complex_close_tol(
        transformed.small_component,
        Complex::new(-0.060_464_244_739_568_386, -0.013_394_427_597_637),
        1.0e-15,
    );

    let continued = rhorrp_exact_radial_continuation(RhorrpExactRadialContinuationInput {
        radius: 2.0,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
        bessel_j_l: Complex::new(0.6, 0.2),
        neumann_l: Complex::new(-0.4, 0.1),
        bessel_j_l_plus_1: Complex::new(0.3, 0.05),
        neumann_l_plus_1: Complex::new(-0.2, 0.2),
    })?;
    assert_complex_close_tol(
        continued.regular_large_component,
        Complex::new(1.314_101_957_995_448_5, 0.299_399_703_487_163_66),
        1.0e-15,
    );
    assert_complex_close_tol(
        continued.regular_small_component,
        Complex::new(-0.000_934_740_206_884_246_9, -0.001_135_889_447_534_475),
        1.0e-17,
    );
    assert_complex_close_tol(
        continued.irregular_large_component,
        Complex::new(-0.513_092_568_622_136_6, 0.143_135_451_704_084_54),
        1.0e-16,
    );
    assert_complex_close_tol(
        continued.irregular_small_component,
        Complex::new(0.001_030_678_096_154_484_5, -9.748_977_376_318_495e-6),
        1.0e-17,
    );
    Ok(())
}

#[test]
fn exact_radial_tail_matches_feff_init_wavefunctions_loop() -> Result<(), RhorrpError> {
    let radii = [0.9, 1.1, 1.4, 1.8];
    let tail = rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
        radii: &radii,
        start_index_1based: 2,
        angular_momentum: 2,
        phase_shift: Complex::new(0.2, -0.1),
        wave_number: Complex::new(0.4, 0.5),
    })?;

    assert_eq!(tail.start_index_1based, 2);
    assert_eq!(tail.row_count(), 3);
    assert_complex_close_tol(
        tail.regular_large_components[0],
        Complex::new(-2.050_540_180_418_029, -0.119_113_885_080_750_09),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.regular_small_components[0],
        Complex::new(0.034_421_495_198_108_36, 0.001_532_680_042_627_237),
        1.0e-15,
    );
    assert_complex_close_tol(
        tail.irregular_large_components[0],
        Complex::new(7.724_218_475_604_785, 4.770_266_774_546_087),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.irregular_small_components[0],
        Complex::new(-0.132_248_029_619_076_95, -0.075_997_664_147_609_21),
        1.0e-15,
    );

    assert_complex_close_tol(
        tail.regular_large_components[1],
        Complex::new(-1.239_107_687_370_063_5, -0.080_448_475_098_744_81),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.regular_small_components[1],
        Complex::new(0.016_554_864_369_938_12, 0.001_226_223_433_787_182),
        1.0e-15,
    );
    assert_complex_close_tol(
        tail.irregular_large_components[1],
        Complex::new(4.511_076_150_131_777, 3.113_694_562_704_52),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.irregular_small_components[1],
        Complex::new(-0.062_554_693_722_132_51, -0.038_411_769_538_196_396),
        1.0e-15,
    );

    assert_complex_close_tol(
        tail.regular_large_components[2],
        Complex::new(-0.711_471_770_825_952_3, 0.005_752_433_782_165_725),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.regular_small_components[2],
        Complex::new(0.007_688_926_229_843_904, 0.000_953_376_106_575_112_2),
        1.0e-15,
    );
    assert_complex_close_tol(
        tail.irregular_large_components[2],
        Complex::new(2.405_099_072_873_88, 2.034_179_702_325_563),
        1.0e-14,
    );
    assert_complex_close_tol(
        tail.irregular_small_components[2],
        Complex::new(-0.028_114_434_990_648_862, -0.019_179_563_860_727_895),
        1.0e-15,
    );
    Ok(())
}

#[test]
fn radial_solution_assembly_matches_feff_init_wavefunctions_steps() -> Result<(), RhorrpError> {
    let radii = [0.9, 1.1, 1.4, 1.8];
    let raw_regular_large = Array1::from_vec(vec![
        Complex::new(0.3, 0.2),
        Complex::new(0.35, 0.24),
        Complex::new(0.41, 0.29),
        Complex::new(0.5, 0.33),
    ]);
    let raw_regular_small = Array1::from_vec(vec![
        Complex::new(-0.01, 0.04),
        Complex::new(-0.012, 0.05),
        Complex::new(-0.014, 0.055),
        Complex::new(-0.016, 0.06),
    ]);
    let raw_irregular_large = Array1::from_vec(vec![
        Complex::new(0.7, -0.2),
        Complex::new(0.68, -0.18),
        Complex::new(0.62, -0.16),
        Complex::new(0.58, -0.12),
    ]);
    let raw_irregular_small = Array1::from_vec(vec![
        Complex::new(0.02, 0.03),
        Complex::new(0.024, 0.035),
        Complex::new(0.027, 0.038),
        Complex::new(0.03, 0.042),
    ]);

    let assembled = rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
        radii: &radii,
        raw_regular_large: raw_regular_large.view(),
        raw_regular_small: raw_regular_small.view(),
        raw_irregular_large: raw_irregular_large.view(),
        raw_irregular_small: raw_irregular_small.view(),
        phase_shift: Complex::new(0.2, -0.1),
        phase_amplitude: Complex::new(1.25, -0.4),
        wave_number: Complex::new(0.4, 0.5),
        angular_momentum: 2,
        match_index_1based: 2,
        exact_tail_start_index_1based: 3,
    })?;

    assert_eq!(assembled.row_count(), 4);
    assert_complex_close_tol(
        assembled.regular_solution_scale,
        Complex::new(0.725_689_404_934_687_9, 0.232_220_609_579_100_12),
        1.0e-16,
    );
    assert_complex_close_tol(
        assembled.irregular_wronskian_scale.denominator,
        Complex::new(-1.874_577_217_189_589_6, 3.750_960_864_218_263_2),
        1.0e-14,
    );
    assert_complex_close_tol(
        assembled.irregular_wronskian_scale.reciprocal_wave_scale,
        Complex::new(-0.364_154_368_523_117_8, -0.078_106_459_632_476_4),
        1.0e-15,
    );

    assert_complex_close_tol(
        assembled.regular_large_components[0],
        Complex::new(0.171_262_699_564_586_34, 0.214_804_063_860_667_6),
        1.0e-15,
    );
    assert_complex_close_tol(
        assembled.regular_small_components[0],
        Complex::new(-0.016_545_718_432_510_886, 0.026_705_370_101_596_52),
        1.0e-16,
    );
    assert_complex_close_tol(
        assembled.irregular_large_components[0],
        Complex::new(0.082_203_861_642_917_79, 0.210_995_197_860_541_83),
        1.0e-15,
    );
    assert_complex_close_tol(
        assembled.irregular_small_components[0],
        Complex::new(-0.024_096_406_051_351_62, -0.001_936_174_802_402_815),
        1.0e-16,
    );

    assert_complex_close_tol(
        assembled.regular_large_components[1],
        Complex::new(0.198_258_345_428_156_73, 0.255_442_670_537_010_14),
        1.0e-15,
    );
    assert_complex_close_tol(
        assembled.irregular_large_components[1],
        Complex::new(0.030_728_523_566_529_83, 0.242_245_361_244_317_25),
        1.0e-15,
    );

    assert_complex_close_tol(
        assembled.regular_large_components[2],
        Complex::new(-1.239_107_687_370_063_5, -0.080_448_475_098_744_81),
        1.0e-14,
    );
    assert_complex_close_tol(
        assembled.irregular_large_components[3],
        Complex::new(2.405_099_072_873_88, 2.034_179_702_325_563),
        1.0e-14,
    );
    Ok(())
}

#[test]
fn radial_solution_assembly_applies_feff_s_wave_irregular_fix() -> Result<(), RhorrpError> {
    let row_count = 120;
    let radii = Array1::from_iter((0..row_count).map(|row| 0.2 + 0.035 * row as Real));
    let raw_regular_large = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.20 + 0.004 * row, 0.05 + 0.002 * row)
    }));
    let raw_regular_small = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(-0.010 + 0.0002 * row, 0.020 + 0.0003 * row)
    }));
    let raw_irregular_large = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.70 - 0.001 * row, -0.20 + 0.002 * row)
    }));
    let raw_irregular_small = Array1::from_iter((0..row_count).map(|row| {
        let row = row as Real;
        Complex::new(0.020 + 0.0001 * row, 0.030 + 0.0002 * row)
    }));

    let base_input = RhorrpRadialSolutionAssemblyInput {
        radii: radii.as_slice().expect("contiguous test radii"),
        raw_regular_large: raw_regular_large.view(),
        raw_regular_small: raw_regular_small.view(),
        raw_irregular_large: raw_irregular_large.view(),
        raw_irregular_small: raw_irregular_small.view(),
        phase_shift: Complex::new(0.2, -0.1),
        phase_amplitude: Complex::new(1.25, -0.4),
        wave_number: Complex::new(0.4, 0.5),
        angular_momentum: 1,
        match_index_1based: 80,
        exact_tail_start_index_1based: 101,
    };
    let unsmoothed = rhorrp_assemble_radial_solutions(base_input)?;
    let smoothed = rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
        angular_momentum: 0,
        ..base_input
    })?;
    let expected_large = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
        radii: radii.as_slice().expect("contiguous test radii"),
        values: unsmoothed.irregular_large_components.view(),
    })?;
    let expected_small = rhorrp_fix_irregular_origin(RhorrpIrregularFixInput {
        radii: radii.as_slice().expect("contiguous test radii"),
        values: unsmoothed.irregular_small_components.view(),
    })?;

    assert!(!unsmoothed.irregular_origin_smoothed);
    assert!(smoothed.irregular_origin_smoothed);
    for row in [0, 49, 74, 99] {
        assert_complex_close_tol(
            smoothed.irregular_large_components[row],
            expected_large[row],
            1.0e-10,
        );
        assert_complex_close_tol(
            smoothed.irregular_small_components[row],
            expected_small[row],
            1.0e-10,
        );
    }
    assert_ne!(
        smoothed.irregular_large_components[0],
        unsmoothed.irregular_large_components[0]
    );
    Ok(())
}

#[test]
fn wavefunction_channel_composes_feff_init_wavefunctions_flow() -> Result<(), RhorrpError> {
    let reference = reference_wavefunction_channel_inputs();
    let input = RhorrpWavefunctionChannelInput {
        solver: reference.to_input(),
        angular_momentum: 1,
        wave_number: wave_number_from_kinetic_energy(reference.energy),
    };
    let channel = rhorrp_wavefunction_channel(input)?;

    let regular = fovrg_dirac_solver(FovrgDiracSolverInput {
        irregular: false,
        muffin_tin_large_component: Complex::new(0.0, 0.0),
        muffin_tin_small_component: Complex::new(0.0, 0.0),
        ..reference.to_input()
    })?;
    let matched = rhorrp_muffin_tin_match(RhorrpMuffinTinMatchInput {
        muffin_tin_radius: reference.muffin_tin_radius,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        regular_large_at_muffin_tin: regular.muffin_tin_large_component,
        regular_small_at_muffin_tin: regular.muffin_tin_small_component,
        kappa: reference.target_kappa,
    })?;
    let irregular_initial =
        rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
            muffin_tin_radius: reference.muffin_tin_radius,
            phase_shift: matched.phase_shift,
            wave_number: input.wave_number,
            bessel_j_l: matched.bessel_j_l,
            neumann_l: matched.neumann_l,
            bessel_j_l_plus_1: matched.bessel_j_l_plus_1,
            neumann_l_plus_1: matched.neumann_l_plus_1,
        })?;
    let irregular = fovrg_dirac_solver(FovrgDiracSolverInput {
        irregular: true,
        muffin_tin_large_component: irregular_initial.large_component,
        muffin_tin_small_component: irregular_initial.small_component,
        ..reference.to_input()
    })?;
    let active_radii = reference
        .radii
        .slice_axis(Axis(0), Slice::from(..regular.active_len))
        .to_vec();
    let expected = rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
        radii: &active_radii,
        raw_regular_large: regular.large_component.view(),
        raw_regular_small: regular.small_component.view(),
        raw_irregular_large: irregular.large_component.view(),
        raw_irregular_small: irregular.small_component.view(),
        phase_shift: matched.phase_shift,
        phase_amplitude: matched.phase_amplitude,
        wave_number: input.wave_number,
        angular_momentum: input.angular_momentum,
        match_index_1based: reference.radial_match_index + 1,
        exact_tail_start_index_1based: reference.radial_match_index + 1,
    })?;

    assert_eq!(channel.regular_active_len, 29);
    assert_eq!(channel.irregular_active_len, 29);
    assert_eq!(channel.regular_iteration_count, 2);
    assert_eq!(channel.radial_solutions.row_count(), 29);
    assert_complex_close_tol(
        channel.muffin_tin_match.phase_shift,
        matched.phase_shift,
        1.0e-12,
    );
    assert_complex_close_tol(
        channel.irregular_initial_condition.large_component,
        irregular_initial.large_component,
        1.0e-12,
    );
    for row in [
        0,
        reference.radial_match_index,
        channel.radial_solutions.row_count() - 1,
    ] {
        assert_complex_close_tol(
            channel.radial_solutions.regular_large_components[row],
            expected.regular_large_components[row],
            1.0e-10,
        );
        assert_complex_close_tol(
            channel.radial_solutions.irregular_small_components[row],
            expected.irregular_small_components[row],
            1.0e-10,
        );
    }
    Ok(())
}

#[test]
fn photoelectron_kappa_matches_feff_init_wavefunctions_channels() -> Result<(), RhorrpError> {
    assert_eq!(rhorrp_photoelectron_kappa(0)?, -1);
    assert_eq!(rhorrp_photoelectron_kappa(1)?, -2);
    assert_eq!(rhorrp_photoelectron_kappa(4)?, -5);
    assert_eq!(rhorrp_c3_scale_for_angular_momentum(0), 0);
    assert_eq!(rhorrp_c3_scale_for_angular_momentum(1), 1);
    assert_eq!(rhorrp_c3_scale_for_angular_momentum(4), 1);
    assert!(matches!(
        rhorrp_photoelectron_kappa(i32::MAX as usize),
        Err(RhorrpError::PhotoelectronKappaOutOfRange {
            angular_momentum,
        }) if angular_momentum == i32::MAX as usize
    ));
    Ok(())
}

#[test]
fn potential_wavefunctions_compose_feff_init_wavefunctions_loop() -> Result<(), RhorrpError> {
    let reference = reference_wavefunction_channel_inputs();
    let energies = Array1::from_vec(vec![
        reference.energy,
        reference.energy + Complex::new(0.045, 0.006),
    ]);
    let reference_energy = Complex::new(0.0, 0.0);
    let norman_radius = reference.radii[8];
    let input = RhorrpPotentialWavefunctionsInput {
        solver: reference.to_input(),
        energies_hartree: energies.view(),
        reference_energy_hartree: reference_energy,
        norman_radius,
        radial_x0: 8.8,
        radial_dx: 0.45,
        exchange_index: 14,
        angular_momentum_count: 2,
    };
    let wavefunctions = rhorrp_potential_wavefunctions(input)?;

    assert_eq!(wavefunctions.energy_count(), 2);
    assert_eq!(wavefunctions.angular_momentum_count(), 2);
    assert_eq!(wavefunctions.radial_count(), 29);
    assert_eq!(wavefunctions.setups.len(), 2);
    assert_eq!(wavefunctions.regular_iteration_count, 0);
    assert_eq!(wavefunctions.irregular_iteration_count, 0);

    for energy_index in 0..energies.len() {
        let setup = rhorrp_wavefunction_setup(RhorrpWavefunctionSetupInput {
            energy_hartree: energies[energy_index],
            reference_energy_hartree: reference_energy,
            muffin_tin_radius: reference.muffin_tin_radius,
            norman_radius,
            radial_x0: 8.8,
            radial_dx: 0.45,
            radial_capacity: reference.radii.len(),
            exchange_index: 14,
        })?;
        assert_eq!(wavefunctions.setups[energy_index], setup);
        assert_complex_close_tol(
            wavefunctions.wave_numbers[energy_index],
            setup.wave_number,
            1.0e-14,
        );

        for angular in 0..2 {
            let expected = rhorrp_wavefunction_channel(RhorrpWavefunctionChannelInput {
                solver: FovrgDiracSolverInput {
                    exchange_cycle_count: setup.dirac_cycle_count,
                    target_kappa: rhorrp_photoelectron_kappa(angular)?,
                    target_last_index: setup.last_integration_index_1based - 1,
                    energy: setup.kinetic_energy_hartree,
                    c3_scale: rhorrp_c3_scale_for_angular_momentum(angular),
                    irregular: false,
                    muffin_tin_large_component: Complex::new(0.0, 0.0),
                    muffin_tin_small_component: Complex::new(0.0, 0.0),
                    ..reference.to_input()
                },
                angular_momentum: angular,
                wave_number: setup.wave_number,
            })?;

            assert_complex_close_tol(
                wavefunctions.phase_shifts[(energy_index, angular)],
                expected.muffin_tin_match.phase_shift,
                1.0e-12,
            );
            for radial in [
                0,
                reference.radial_match_index,
                wavefunctions.radial_count() - 1,
            ] {
                assert_complex_close_tol(
                    wavefunctions.regular_large[(energy_index, angular, radial)],
                    expected.radial_solutions.regular_large_components[radial],
                    1.0e-10,
                );
                assert_complex_close_tol(
                    wavefunctions.irregular_small[(energy_index, angular, radial)],
                    expected.radial_solutions.irregular_small_components[radial],
                    1.0e-10,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn prepared_potential_wavefunctions_builds_feff_fovrg_input() -> Result<(), RhorrpError> {
    let reference = reference_wavefunction_channel_inputs();
    let prepared = prepared_wavefunction_grids_from_reference(&reference, 2);
    let energies = Array1::from_vec(vec![
        reference.energy,
        reference.energy + Complex::new(0.045, 0.006),
    ]);
    let norman_radius = reference.radii[8];

    let wavefunctions =
        rhorrp_prepared_potential_wavefunctions(RhorrpPreparedPotentialWavefunctionsInput {
            prepared: &prepared,
            potential_index: 0,
            energies_hartree: energies.view(),
            muffin_tin_radius: reference.muffin_tin_radius,
            norman_radius,
            bound_large_coefficients: reference.bound_large_coefficients.view(),
            bound_small_coefficients: reference.bound_small_coefficients.view(),
            electron_counts: reference.electron_counts.view(),
            valence_counts: reference.valence_counts.view(),
            kappa: reference.kappa.view(),
            atomic_number: 29.0,
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_count: reference.bound_orbital_count,
        })?;
    let expected = rhorrp_potential_wavefunctions(RhorrpPotentialWavefunctionsInput {
        solver: reference.to_input(),
        energies_hartree: energies.view(),
        reference_energy_hartree: prepared.reference_energies_hartree[0],
        norman_radius,
        radial_x0: 8.8,
        radial_dx: prepared.radial_dx,
        exchange_index: 14,
        angular_momentum_count: 2,
    })?;

    assert_eq!(wavefunctions.setups, expected.setups);
    assert_eq!(wavefunctions.energy_count(), expected.energy_count());
    assert_eq!(
        wavefunctions.angular_momentum_count(),
        expected.angular_momentum_count()
    );
    assert_eq!(wavefunctions.radial_count(), expected.radial_count());
    for energy in 0..wavefunctions.energy_count() {
        assert_complex_close_tol(
            wavefunctions.wave_numbers[energy],
            expected.wave_numbers[energy],
            1.0e-14,
        );
        for angular in 0..wavefunctions.angular_momentum_count() {
            assert_complex_close_tol(
                wavefunctions.phase_shifts[(energy, angular)],
                expected.phase_shifts[(energy, angular)],
                1.0e-12,
            );
            for radial in [
                0,
                reference.radial_match_index,
                wavefunctions.radial_count() - 1,
            ] {
                assert_complex_close_tol(
                    wavefunctions.regular_large[(energy, angular, radial)],
                    expected.regular_large[(energy, angular, radial)],
                    1.0e-10,
                );
                assert_complex_close_tol(
                    wavefunctions.irregular_large[(energy, angular, radial)],
                    expected.irregular_large[(energy, angular, radial)],
                    1.0e-10,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn wavefunction_tables_compose_feff_init_wavefunctions_potential_loop() -> Result<(), RhorrpError> {
    let first = reference_wavefunction_channel_inputs();
    let mut second = reference_wavefunction_channel_inputs();
    second.muffin_tin_radius = 1.50;
    let energies = Array1::from_vec(vec![
        first.energy,
        first.energy + Complex::new(0.045, 0.006),
    ]);
    let first_input = RhorrpPotentialWavefunctionsInput {
        solver: first.to_input(),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.0, 0.0),
        norman_radius: first.radii[8],
        radial_x0: 8.8,
        radial_dx: 0.45,
        exchange_index: 14,
        angular_momentum_count: 2,
    };
    let second_input = RhorrpPotentialWavefunctionsInput {
        solver: second.to_input(),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.018, 0.002),
        norman_radius: second.radii[9],
        radial_x0: 8.8,
        radial_dx: 0.45,
        exchange_index: 14,
        angular_momentum_count: 2,
    };
    let potential_inputs = [first_input, second_input];
    let tables = rhorrp_wavefunction_tables(RhorrpWavefunctionTablesInput {
        potentials: &potential_inputs,
    })?;

    assert_eq!(tables.energy_count(), 2);
    assert_eq!(tables.angular_momentum_count(), 2);
    assert_eq!(tables.radial_count(), 29);
    assert_eq!(tables.potential_count(), 2);
    assert_eq!(tables.setups_by_potential.len(), 2);

    for (potential_index, &potential_input) in potential_inputs.iter().enumerate() {
        let expected = rhorrp_potential_wavefunctions(potential_input)?;
        assert_eq!(tables.setups_by_potential[potential_index], expected.setups);
        for energy in 0..expected.energy_count() {
            assert_complex_close_tol(
                tables.wave_numbers[(energy, potential_index)],
                expected.wave_numbers[energy],
                1.0e-14,
            );
            for angular in 0..expected.angular_momentum_count() {
                assert_complex_close_tol(
                    tables.phase_shifts[(energy, angular, potential_index)],
                    expected.phase_shifts[(energy, angular)],
                    1.0e-12,
                );
                for radial in [0, first.radial_match_index, expected.radial_count() - 1] {
                    assert_complex_close_tol(
                        tables.regular_large[(energy, angular, radial, potential_index)],
                        expected.regular_large[(energy, angular, radial)],
                        1.0e-10,
                    );
                    assert_complex_close_tol(
                        tables.irregular_small[(energy, angular, radial, potential_index)],
                        expected.irregular_small[(energy, angular, radial)],
                        1.0e-10,
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn prepared_wavefunction_tables_compose_feff_init_wavefunctions_potential_loop()
-> Result<(), RhorrpError> {
    let reference = reference_wavefunction_channel_inputs();
    let prepared = prepared_wavefunction_grids_from_reference(&reference, 2);
    let energies = Array1::from_vec(vec![
        reference.energy,
        reference.energy + Complex::new(0.045, 0.006),
    ]);
    let muffin_tin_radii = [reference.muffin_tin_radius, 1.50];
    let norman_radii = [reference.radii[8], reference.radii[9]];
    let atomic_numbers = [29.0, 30.0];
    let bound_large_coefficients_by_potential = Array3::from_shape_fn(
        (
            reference.bound_large_coefficients.nrows(),
            reference.bound_large_coefficients.ncols(),
            2,
        ),
        |(coefficient, orbital, potential)| {
            reference.bound_large_coefficients[(coefficient, orbital)]
                * (1.0 + 0.015 * potential as Real)
        },
    );
    let bound_small_coefficients_by_potential = Array3::from_shape_fn(
        (
            reference.bound_small_coefficients.nrows(),
            reference.bound_small_coefficients.ncols(),
            2,
        ),
        |(coefficient, orbital, potential)| {
            reference.bound_small_coefficients[(coefficient, orbital)]
                * (1.0 - 0.012 * potential as Real)
        },
    );
    let electron_counts_by_potential = Array2::from_shape_fn(
        (reference.electron_counts.len(), 2),
        |(orbital, potential)| {
            reference.electron_counts[orbital] + 0.05 * orbital as Real * potential as Real
        },
    );
    let valence_counts_by_potential = Array2::from_shape_fn(
        (reference.valence_counts.len(), 2),
        |(orbital, potential)| {
            reference.valence_counts[orbital] + 0.03 * (orbital + 1) as Real * potential as Real
        },
    );
    let mut kappa_by_potential =
        Array2::from_shape_fn((reference.kappa.len(), 2), |(orbital, _)| {
            reference.kappa[orbital]
        });
    kappa_by_potential[(1, 1)] = -2;
    let bound_orbital_counts = [
        reference.bound_orbital_count,
        reference.bound_orbital_count - 1,
    ];

    let tables = rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
        prepared: &prepared,
        energies_hartree: energies.view(),
        muffin_tin_radii: &muffin_tin_radii,
        norman_radii: &norman_radii,
        bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
        bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
        electron_counts_by_potential: electron_counts_by_potential.view(),
        valence_counts_by_potential: valence_counts_by_potential.view(),
        kappa_by_potential: kappa_by_potential.view(),
        atomic_numbers: &atomic_numbers,
        exchange_index: 14,
        angular_momentum_count: 2,
        bound_orbital_counts: &bound_orbital_counts,
    })?;

    let potential_inputs = [
        RhorrpPotentialWavefunctionsInput {
            solver: prepared_reference_solver(
                &prepared,
                &reference,
                PreparedReferencePotentialInput {
                    index: 0,
                    muffin_tin_radius: muffin_tin_radii[0],
                    atomic_number: 29.0,
                    bound_large_coefficients: bound_large_coefficients_by_potential
                        .index_axis(Axis(2), 0),
                    bound_small_coefficients: bound_small_coefficients_by_potential
                        .index_axis(Axis(2), 0),
                    electron_counts: electron_counts_by_potential.index_axis(Axis(1), 0),
                    valence_counts: valence_counts_by_potential.index_axis(Axis(1), 0),
                    kappa: kappa_by_potential.index_axis(Axis(1), 0),
                    bound_orbital_count: bound_orbital_counts[0],
                },
            ),
            energies_hartree: energies.view(),
            reference_energy_hartree: prepared.reference_energies_hartree[0],
            norman_radius: norman_radii[0],
            radial_x0: 8.8,
            radial_dx: prepared.radial_dx,
            exchange_index: 14,
            angular_momentum_count: 2,
        },
        RhorrpPotentialWavefunctionsInput {
            solver: prepared_reference_solver(
                &prepared,
                &reference,
                PreparedReferencePotentialInput {
                    index: 1,
                    muffin_tin_radius: muffin_tin_radii[1],
                    atomic_number: 30.0,
                    bound_large_coefficients: bound_large_coefficients_by_potential
                        .index_axis(Axis(2), 1),
                    bound_small_coefficients: bound_small_coefficients_by_potential
                        .index_axis(Axis(2), 1),
                    electron_counts: electron_counts_by_potential.index_axis(Axis(1), 1),
                    valence_counts: valence_counts_by_potential.index_axis(Axis(1), 1),
                    kappa: kappa_by_potential.index_axis(Axis(1), 1),
                    bound_orbital_count: bound_orbital_counts[1],
                },
            ),
            energies_hartree: energies.view(),
            reference_energy_hartree: prepared.reference_energies_hartree[1],
            norman_radius: norman_radii[1],
            radial_x0: 8.8,
            radial_dx: prepared.radial_dx,
            exchange_index: 14,
            angular_momentum_count: 2,
        },
    ];
    let expected = rhorrp_wavefunction_tables(RhorrpWavefunctionTablesInput {
        potentials: &potential_inputs,
    })?;

    assert_eq!(tables.energy_count(), expected.energy_count());
    assert_eq!(
        tables.angular_momentum_count(),
        expected.angular_momentum_count()
    );
    assert_eq!(tables.radial_count(), expected.radial_count());
    assert_eq!(tables.potential_count(), expected.potential_count());
    assert_eq!(tables.setups_by_potential, expected.setups_by_potential);
    for potential in 0..tables.potential_count() {
        for energy in 0..tables.energy_count() {
            assert_complex_close_tol(
                tables.wave_numbers[(energy, potential)],
                expected.wave_numbers[(energy, potential)],
                1.0e-14,
            );
            for angular in 0..tables.angular_momentum_count() {
                assert_complex_close_tol(
                    tables.phase_shifts[(energy, angular, potential)],
                    expected.phase_shifts[(energy, angular, potential)],
                    1.0e-12,
                );
                for radial in [0, reference.radial_match_index, tables.radial_count() - 1] {
                    assert_complex_close_tol(
                        tables.regular_large[(energy, angular, radial, potential)],
                        expected.regular_large[(energy, angular, radial, potential)],
                        1.0e-10,
                    );
                    assert_complex_close_tol(
                        tables.irregular_small[(energy, angular, radial, potential)],
                        expected.irregular_small[(energy, angular, radial, potential)],
                        1.0e-10,
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn density_reference_energy_matches_feff_eref0_handoff() -> Result<(), RhorrpError> {
    let reference = reference_wavefunction_channel_inputs();
    let prepared = prepared_wavefunction_grids_from_reference(&reference, 3);

    assert_complex_close_tol(
        rhorrp_density_reference_energy_hartree(&prepared)?,
        Complex::new(0.036, 0.004),
        1.0e-14,
    );
    Ok(())
}

#[test]
fn radial_solution_scalars_reject_invalid_inputs() {
    assert!(matches!(
        rhorrp_regular_solution_scale(RhorrpRegularSolutionScaleInput {
            phase_amplitude: Complex::new(0.0, 0.0),
        }),
        Err(RhorrpError::ZeroComplexResult {
            name: "regular_solution_phase_amplitude",
        })
    ));

    assert!(matches!(
        rhorrp_irregular_initial_condition(RhorrpIrregularInitialConditionInput {
            muffin_tin_radius: 0.0,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
            bessel_j_l: Complex::new(0.8, 0.1),
            neumann_l: Complex::new(-0.3, 0.05),
            bessel_j_l_plus_1: Complex::new(0.25, -0.03),
            neumann_l_plus_1: Complex::new(-0.6, 0.2),
        }),
        Err(RhorrpError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_exact_radial_continuation(RhorrpExactRadialContinuationInput {
            radius: 0.0,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
            bessel_j_l: Complex::new(0.6, 0.2),
            neumann_l: Complex::new(-0.4, 0.1),
            bessel_j_l_plus_1: Complex::new(0.3, 0.05),
            neumann_l_plus_1: Complex::new(-0.2, 0.2),
        }),
        Err(RhorrpError::InvalidPositiveRadius { name: "radius", .. })
    ));

    assert!(matches!(
        rhorrp_irregular_wronskian_scale(RhorrpIrregularWronskianScaleInput {
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.0, 0.0),
            regular_large_at_match: Complex::new(0.3, 0.2),
            regular_small_at_match: Complex::new(-0.01, 0.04),
            irregular_large_at_match: Complex::new(0.7, -0.2),
            irregular_small_at_match: Complex::new(0.02, 0.03),
        }),
        Err(RhorrpError::ZeroComplexResult {
            name: "wronskian_wave_number",
        })
    ));

    assert!(matches!(
        rhorrp_irregular_solution_transform(RhorrpIrregularSolutionTransformInput {
            phase_factor: Complex::new(f64::NAN, 0.0),
            reciprocal_wave_scale: Complex::new(1.0, 0.0),
            regular_large_component: Complex::new(0.11, 0.22),
            regular_small_component: Complex::new(-0.04, 0.05),
            irregular_large_component: Complex::new(0.6, -0.15),
            irregular_small_component: Complex::new(0.03, 0.08),
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "irregular_transform_phase_factor",
            ..
        })
    ));

    let radii = [0.9, 1.1, 1.4];
    assert!(matches!(
        rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
            radii: &radii,
            start_index_1based: 4,
            angular_momentum: 2,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
        }),
        Err(RhorrpError::ExactRadialTailStartOutOfRange {
            start_index_1based: 4,
            radial_count: 3,
        })
    ));

    assert!(matches!(
        rhorrp_exact_radial_tail(RhorrpExactRadialTailInput {
            radii: &[1.0],
            start_index_1based: 1,
            angular_momentum: 9,
            phase_shift: Complex::new(0.2, -0.1),
            wave_number: Complex::new(0.4, 0.5),
        }),
        Err(RhorrpError::BesselEvaluation {
            source: crate::BesselError::ExactOrderOutOfRange {
                order: 10,
                max_order: 9,
            },
        })
    ));

    let radii = [0.9, 1.1, 1.4];
    let two_rows = Array1::from_vec(vec![Complex::new(0.1, 0.0), Complex::new(0.2, 0.0)]);
    let three_rows = Array1::from_vec(vec![
        Complex::new(0.1, 0.0),
        Complex::new(0.2, 0.0),
        Complex::new(0.3, 0.0),
    ]);
    assert!(matches!(
        rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
            radii: &radii,
            raw_regular_large: two_rows.view(),
            raw_regular_small: three_rows.view(),
            raw_irregular_large: three_rows.view(),
            raw_irregular_small: three_rows.view(),
            phase_shift: Complex::new(0.2, -0.1),
            phase_amplitude: Complex::new(1.25, -0.4),
            wave_number: Complex::new(0.4, 0.5),
            angular_momentum: 2,
            match_index_1based: 2,
            exact_tail_start_index_1based: 2,
        }),
        Err(RhorrpError::RadialSolutionLengthMismatch {
            component: "raw_regular_large",
            expected: 3,
            actual: 2,
        })
    ));

    assert!(matches!(
        rhorrp_assemble_radial_solutions(RhorrpRadialSolutionAssemblyInput {
            radii: &radii,
            raw_regular_large: three_rows.view(),
            raw_regular_small: three_rows.view(),
            raw_irregular_large: three_rows.view(),
            raw_irregular_small: three_rows.view(),
            phase_shift: Complex::new(0.2, -0.1),
            phase_amplitude: Complex::new(1.25, -0.4),
            wave_number: Complex::new(0.4, 0.5),
            angular_momentum: 2,
            match_index_1based: 0,
            exact_tail_start_index_1based: 2,
        }),
        Err(RhorrpError::RadialSolutionMatchIndexOutOfRange {
            match_index_1based: 0,
            radial_count: 3,
        })
    ));

    let reference = reference_wavefunction_channel_inputs();
    assert!(matches!(
        rhorrp_wavefunction_channel(RhorrpWavefunctionChannelInput {
            solver: reference.to_input(),
            angular_momentum: 1,
            wave_number: Complex::new(f64::NAN, 0.0),
        }),
        Err(RhorrpError::NonFiniteValue {
            name: "wavefunction_channel_wave_number",
            ..
        })
    ));

    assert!(matches!(
        rhorrp_potential_wavefunctions(RhorrpPotentialWavefunctionsInput {
            solver: reference.to_input(),
            energies_hartree: Array1::<Complex>::zeros(0).view(),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            norman_radius: reference.radii[8],
            radial_x0: 8.8,
            radial_dx: 0.45,
            exchange_index: 14,
            angular_momentum_count: 2,
        }),
        Err(RhorrpError::InvalidWavefunctionShape {
            energy: 0,
            angular: 2,
            radial: 40,
        })
    ));

    assert!(matches!(
        rhorrp_potential_wavefunctions(RhorrpPotentialWavefunctionsInput {
            solver: reference.to_input(),
            energies_hartree: Array1::from_vec(vec![reference.energy]).view(),
            reference_energy_hartree: Complex::new(0.0, 0.0),
            norman_radius: reference.radii[8],
            radial_x0: 8.8,
            radial_dx: 0.45,
            exchange_index: 14,
            angular_momentum_count: 0,
        }),
        Err(RhorrpError::InvalidWavefunctionShape {
            energy: 1,
            angular: 0,
            radial: 40,
        })
    ));

    assert!(matches!(
        rhorrp_wavefunction_tables(RhorrpWavefunctionTablesInput { potentials: &[] }),
        Err(RhorrpError::InvalidWavefunctionPotentialCount { potential_count: 0 })
    ));

    let energies = Array1::from_vec(vec![reference.energy]);
    let first_input = RhorrpPotentialWavefunctionsInput {
        solver: reference.to_input(),
        energies_hartree: energies.view(),
        reference_energy_hartree: Complex::new(0.0, 0.0),
        norman_radius: reference.radii[8],
        radial_x0: 8.8,
        radial_dx: 0.45,
        exchange_index: 14,
        angular_momentum_count: 2,
    };
    let mismatched_input = RhorrpPotentialWavefunctionsInput {
        angular_momentum_count: 1,
        ..first_input
    };
    let mismatched_potentials = [first_input, mismatched_input];
    assert!(matches!(
        rhorrp_wavefunction_tables(RhorrpWavefunctionTablesInput {
            potentials: &mismatched_potentials,
        }),
        Err(RhorrpError::WavefunctionPotentialShapeMismatch {
            potential: 1,
            expected_energy: 1,
            expected_angular: 2,
            expected_radial: 29,
            actual_energy: 1,
            actual_angular: 1,
            actual_radial: 29,
        })
    ));

    let prepared = prepared_wavefunction_grids_from_reference(&reference, 1);
    let energies = Array1::from_vec(vec![reference.energy]);
    assert!(matches!(
        rhorrp_prepared_potential_wavefunctions(RhorrpPreparedPotentialWavefunctionsInput {
            prepared: &prepared,
            potential_index: 1,
            energies_hartree: energies.view(),
            muffin_tin_radius: reference.muffin_tin_radius,
            norman_radius: reference.radii[8],
            bound_large_coefficients: reference.bound_large_coefficients.view(),
            bound_small_coefficients: reference.bound_small_coefficients.view(),
            electron_counts: reference.electron_counts.view(),
            valence_counts: reference.valence_counts.view(),
            kappa: reference.kappa.view(),
            atomic_number: 29.0,
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_count: reference.bound_orbital_count,
        }),
        Err(RhorrpError::PreparedWavefunctionPotentialOutOfRange {
            potential: 1,
            potential_count: 1,
        })
    ));

    let mut bad_prepared = prepared.clone();
    bad_prepared.reference_indices_1based[0] = 1;
    assert!(matches!(
        rhorrp_prepared_potential_wavefunctions(RhorrpPreparedPotentialWavefunctionsInput {
            prepared: &bad_prepared,
            potential_index: 0,
            energies_hartree: energies.view(),
            muffin_tin_radius: reference.muffin_tin_radius,
            norman_radius: reference.radii[8],
            bound_large_coefficients: reference.bound_large_coefficients.view(),
            bound_small_coefficients: reference.bound_small_coefficients.view(),
            electron_counts: reference.electron_counts.view(),
            valence_counts: reference.valence_counts.view(),
            kappa: reference.kappa.view(),
            atomic_number: 29.0,
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_count: reference.bound_orbital_count,
        }),
        Err(RhorrpError::PreparedWavefunctionReferenceIndexOutOfRange {
            potential: 0,
            index_1based: 1,
            radial_count: 40,
        })
    ));

    let bound_large_coefficients_by_potential = reference
        .bound_large_coefficients
        .clone()
        .insert_axis(Axis(2));
    let bound_small_coefficients_by_potential = reference
        .bound_small_coefficients
        .clone()
        .insert_axis(Axis(2));
    let electron_counts_by_potential = reference.electron_counts.clone().insert_axis(Axis(1));
    let valence_counts_by_potential = reference.valence_counts.clone().insert_axis(Axis(1));
    let kappa_by_potential = reference.kappa.clone().insert_axis(Axis(1));
    let bound_orbital_counts = [reference.bound_orbital_count];
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "norman_radii",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_bound_large_coefficients_by_potential = Array3::zeros((
        reference.bound_large_coefficients.nrows(),
        reference.bound_large_coefficients.ncols(),
        0,
    ));
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: empty_bound_large_coefficients_by_potential
                .view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "bound_large_coefficients_by_potential",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_bound_small_coefficients_by_potential = Array3::zeros((
        reference.bound_small_coefficients.nrows(),
        reference.bound_small_coefficients.ncols(),
        0,
    ));
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: empty_bound_small_coefficients_by_potential
                .view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "bound_small_coefficients_by_potential",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_electron_counts_by_potential = Array2::zeros((reference.electron_counts.len(), 0));
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: empty_electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "electron_counts_by_potential",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_valence_counts_by_potential = Array2::zeros((reference.valence_counts.len(), 0));
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: empty_valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "valence_counts_by_potential",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_kappa_by_potential = Array2::zeros((reference.kappa.len(), 0));
    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: empty_kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &bound_orbital_counts,
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "kappa_by_potential",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    assert!(matches!(
        rhorrp_prepared_wavefunction_tables(RhorrpPreparedWavefunctionTablesInput {
            prepared: &prepared,
            energies_hartree: energies.view(),
            muffin_tin_radii: &[reference.muffin_tin_radius],
            norman_radii: &[reference.radii[8]],
            bound_large_coefficients_by_potential: bound_large_coefficients_by_potential.view(),
            bound_small_coefficients_by_potential: bound_small_coefficients_by_potential.view(),
            electron_counts_by_potential: electron_counts_by_potential.view(),
            valence_counts_by_potential: valence_counts_by_potential.view(),
            kappa_by_potential: kappa_by_potential.view(),
            atomic_numbers: &[29.0],
            exchange_index: 14,
            angular_momentum_count: 2,
            bound_orbital_counts: &[],
        }),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "bound_orbital_counts",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));

    let empty_prepared = prepared_wavefunction_grids_from_reference(&reference, 0);
    assert!(matches!(
        rhorrp_density_reference_energy_hartree(&empty_prepared),
        Err(RhorrpError::InvalidWavefunctionGridPotentialCount { potential_count: 0 })
    ));

    let mut bad_reference_prepared = prepared.clone();
    bad_reference_prepared.reference_energies_hartree = Array1::zeros(0);
    assert!(matches!(
        rhorrp_density_reference_energy_hartree(&bad_reference_prepared),
        Err(RhorrpError::PreparedWavefunctionMetadataLengthMismatch {
            component: "reference_energies_hartree",
            expected_potentials: 1,
            actual_potentials: 0,
        })
    ));
}

struct PreparedReferencePotentialInput<'a> {
    index: usize,
    muffin_tin_radius: Real,
    atomic_number: Real,
    bound_large_coefficients: ndarray::ArrayView2<'a, Real>,
    bound_small_coefficients: ndarray::ArrayView2<'a, Real>,
    electron_counts: ndarray::ArrayView1<'a, Real>,
    valence_counts: ndarray::ArrayView1<'a, Real>,
    kappa: ndarray::ArrayView1<'a, i32>,
    bound_orbital_count: usize,
}

fn prepared_reference_solver<'a>(
    prepared: &'a RhorrpWavefunctionGridPreparation,
    reference: &'a ReferenceWavefunctionChannelInputs,
    potential: PreparedReferencePotentialInput<'a>,
) -> FovrgDiracSolverInput<'a> {
    FovrgDiracSolverInput {
        exchange_cycle_count: 1,
        target_kappa: reference.target_kappa,
        muffin_tin_radius: potential.muffin_tin_radius,
        target_last_index: reference.target_last_index,
        energy: reference.energy,
        step: prepared.radial_dx,
        radii: prepared.radii.view(),
        exchange_correlation_potential: prepared
            .total_potential
            .index_axis(Axis(1), potential.index),
        valence_exchange_correlation_potential: prepared
            .valence_potential
            .index_axis(Axis(1), potential.index),
        bound_large_components: prepared
            .bound_large_components
            .index_axis(Axis(2), potential.index),
        bound_small_components: prepared
            .bound_small_components
            .index_axis(Axis(2), potential.index),
        bound_large_coefficients: potential.bound_large_coefficients,
        bound_small_coefficients: potential.bound_small_coefficients,
        electron_counts: potential.electron_counts,
        valence_counts: potential.valence_counts,
        kappa: potential.kappa,
        muffin_tin_large_component: Complex::new(0.0, 0.0),
        muffin_tin_small_component: Complex::new(0.0, 0.0),
        atomic_number: potential.atomic_number,
        irregular: false,
        c3_scale: 0,
        radial_match_index: prepared.reference_indices_1based[potential.index] - 2,
        bound_orbital_count: potential.bound_orbital_count,
    }
}

fn prepared_wavefunction_grids_from_reference(
    reference: &ReferenceWavefunctionChannelInputs,
    potential_count: usize,
) -> RhorrpWavefunctionGridPreparation {
    let radial_count = reference.radii.len();
    let orbital_count = reference.bound_orbital_count;
    let mut reference_indices_1based = Vec::with_capacity(potential_count);
    for _ in 0..potential_count {
        reference_indices_1based.push(reference.radial_match_index + 2);
    }

    RhorrpWavefunctionGridPreparation {
        radii: reference.radii.clone(),
        radial_dx: 0.45,
        potential_jumps: Array1::zeros(potential_count),
        reference_indices_1based,
        reference_energies_hartree: Array1::from_iter(
            (0..potential_count).map(|potential| {
                Complex::new(0.018 * potential as Real, 0.002 * potential as Real)
            }),
        ),
        total_potential: Array2::from_shape_fn(
            (radial_count, potential_count),
            |(row, potential)| {
                reference.exchange_correlation_potential[row]
                    + Complex::new(0.010 * potential as Real, -0.001 * potential as Real)
            },
        ),
        valence_potential: Array2::from_shape_fn(
            (radial_count, potential_count),
            |(row, potential)| {
                reference.valence_exchange_correlation_potential[row]
                    + Complex::new(0.006 * potential as Real, 0.0015 * potential as Real)
            },
        ),
        bound_large_components: Array3::from_shape_fn(
            (radial_count, orbital_count, potential_count),
            |(row, orbital, potential)| {
                reference.bound_large_components[(row, orbital)] * (1.0 + 0.01 * potential as Real)
            },
        ),
        bound_small_components: Array3::from_shape_fn(
            (radial_count, orbital_count, potential_count),
            |(row, orbital, potential)| {
                reference.bound_small_components[(row, orbital)] * (1.0 - 0.01 * potential as Real)
            },
        ),
        bound_active_lengths: Array2::from_elem((orbital_count, potential_count), radial_count),
    }
}

struct ReferenceWavefunctionChannelInputs {
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

impl ReferenceWavefunctionChannelInputs {
    fn to_input(&self) -> FovrgDiracSolverInput<'_> {
        FovrgDiracSolverInput {
            exchange_cycle_count: 1,
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
            irregular: false,
            c3_scale: 0,
            radial_match_index: self.radial_match_index,
            bound_orbital_count: self.bound_orbital_count,
        }
    }
}

fn reference_wavefunction_channel_inputs() -> ReferenceWavefunctionChannelInputs {
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

    ReferenceWavefunctionChannelInputs {
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

fn wave_number_from_kinetic_energy(kinetic_energy: Complex) -> Complex {
    let alpha_kinetic = kinetic_energy / 137.035_989_56;
    (kinetic_energy * 2.0 + alpha_kinetic * alpha_kinetic).sqrt()
}

#[test]
fn atomic_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let reference = reference_atomic_density_tables();

    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.08, 0.04, -0.03],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        9.746_265_921_948_757,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.72, -0.15, 0.18],
            orbital_index_1based: 2,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        2.182_748_347_338_233e1,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 3,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        7.107_185_239_762_148e6,
    );
    assert_real_close_scaled(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [4.2, 3.9, -2.5],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        })?,
        0.0,
    );
    Ok(())
}

#[test]
fn atomic_density_rejects_invalid_inputs() {
    let reference = reference_atomic_density_tables();
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 0,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityOrbital {
            orbital: 0,
            orbital_count: 3,
        })
    ));

    let bad_potentials = [0, 1, 3, 1];
    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &bad_potentials,
            radii: &reference.radii,
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::InvalidAtomicDensityPotential {
            atom_index_1based: 3,
            potential: 3,
            max_potential: 2,
        })
    ));

    assert!(matches!(
        rhorrp_atomic_density(RhorrpAtomicDensityInput {
            point: [0.0, 0.0, 0.0],
            orbital_index_1based: 1,
            atom_positions: reference.positions.view(),
            atom_potentials: &reference.potentials,
            radii: &reference.radii[..11],
            large_components: reference.large.view(),
            small_components: reference.small.view(),
        }),
        Err(RhorrpError::AtomicDensityRadialLengthMismatch {
            radii: 11,
            components: 12,
        })
    ));
}

#[test]
fn integrate_density_matches_feff_reference() -> Result<(), RhorrpError> {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        })?,
        -4.627_669_214_946_009e-2,
    );
    assert_real_close(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: -0.010,
            temperature_hartree: 0.000_001,
            chemical_potential_override_hartree: None,
        })?,
        -1.115_611_780_024_965e-3,
    );
    Ok(())
}

#[test]
fn integrate_density_rejects_invalid_inputs() {
    let (energies, energy_density) = reference_density_integration_inputs();

    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.slice_axis(Axis(0), Slice::from(..7)),
            energy_density: energy_density.view(),
            real_axis_count: 6,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::DensityIntegrationLengthMismatch {
            energies: 7,
            densities: 8,
        })
    ));
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: energies.view(),
            energy_density: energy_density.view(),
            real_axis_count: 1,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::InvalidDensityIntegrationRealAxisCount {
            real_axis_count: 1,
            energy_count: 8,
        })
    ));

    let vertical_only = Array1::from_vec(vec![
        Complex::new(-0.03, 0.09),
        Complex::new(-0.03, 0.06),
        Complex::new(-0.03, 0.03),
        Complex::new(-0.03, 0.00),
    ]);
    let vertical_density = Array1::from_vec(vec![Complex::new(0.3, 0.1); 4]);
    assert!(matches!(
        rhorrp_integrate_density(RhorrpDensityIntegrationInput {
            energies_hartree: vertical_only.view(),
            energy_density: vertical_density.view(),
            real_axis_count: 4,
            chemical_potential_hartree: 0.045,
            temperature_hartree: 0.0035,
            chemical_potential_override_hartree: None,
        }),
        Err(RhorrpError::MissingDensityIntegrationCorner)
    ));
}
