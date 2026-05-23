use super::{support::*, *};

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
