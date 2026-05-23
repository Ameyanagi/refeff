use super::*;

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
