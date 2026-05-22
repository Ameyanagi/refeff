use super::{support::*, *};

#[test]
fn converts_energy_to_signed_wave_number() {
    assert_eq!(wave_number_from_hartree(2.0), 2.0);
    assert_eq!(wave_number_from_hartree(-2.0), -2.0);
    assert_eq!(wave_number_from_hartree(0.0), 0.0);
}

#[test]
fn reproduces_loucks_log_grid_points() {
    assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
    assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
    assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
}

#[test]
fn maps_radius_to_index_below() -> Result<(), GridError> {
    let radius = loucks_radius(42);
    assert_eq!(loucks_index_below(radius)?, 42);

    let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
    assert_eq!(loucks_index_below(midpoint)?, 42);
    Ok(())
}

#[test]
fn rejects_invalid_radius_or_delta() {
    assert!(matches!(
        loucks_index_below(0.0),
        Err(GridError::InvalidRadius { .. })
    ));
    assert!(matches!(
        radial_index_below(1.0, 0.0),
        Err(GridError::InvalidDelta { .. })
    ));
}

#[test]
fn fix_dirac_spinor_grid_matches_feff_fixdsp_reference() -> Result<(), GridError> {
    let mut large = vec![0.0; 251];
    let mut small = vec![0.0; 251];
    for i in 1..=80 {
        let i_real = i as Real;
        large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
        small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
    }
    let large = Array1::from_vec(large);
    let small = Array1::from_vec(small);

    let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_component: large.view(),
        small_component: small.view(),
        output_len: 180,
    })?;

    assert_eq!(result.active_len, 161);
    assert_spinor_value(
        &result,
        1,
        0.098_856_582_548_901_49,
        0.981_461_262_295_415_9,
    );
    assert_spinor_value(&result, 2, 0.146_525_001_614_189, 0.969_970_868_040_543_4);
    assert_spinor_value(
        &result,
        3,
        0.192_879_394_911_354_22,
        0.957_050_307_749_104_5,
    );
    assert_spinor_value(&result, 10, 0.473_738_853_193_487_96, 0.830_355_320_320_026);
    assert_spinor_value(
        &result,
        80,
        -0.310_280_702_093_608_3,
        -0.562_325_207_440_241_6,
    );
    assert_spinor_value(
        &result,
        120,
        -0.008_407_166_503_866_128,
        0.021_105_137_955_943_806,
    );
    assert_spinor_value(
        &result,
        160,
        0.191_266_534_139_204_64,
        0.176_750_359_590_577_94,
    );
    assert_spinor_value(&result, 161, 0.0, 0.0);
    assert_spinor_value(&result, 180, 0.0, 0.0);
    Ok(())
}

#[test]
fn fix_dirac_spinor_grid_zero_fills_empty_spinor() -> Result<(), GridError> {
    let large = Array1::zeros(251);
    let small = Array1::zeros(251);

    let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_component: large.view(),
        small_component: small.view(),
        output_len: 16,
    })?;

    assert_eq!(result.active_len, 0);
    assert!(result.large_component.iter().all(|&value| value == 0.0));
    assert!(result.small_component.iter().all(|&value| value == 0.0));
    Ok(())
}

#[test]
fn fix_dirac_spinor_grid_rejects_invalid_inputs() {
    let large = Array1::zeros(4);
    let small = Array1::zeros(3);
    assert_eq!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 16,
        }),
        Err(GridError::SpinorLengthMismatch {
            large_len: 4,
            small_len: 3,
        })
    );

    let nonfinite = Array1::from_vec(vec![0.0, f64::NAN, 0.0, 0.0]);
    let zeros = Array1::zeros(4);
    assert!(matches!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: nonfinite.view(),
            small_component: zeros.view(),
            output_len: 16,
        }),
        Err(GridError::NonFiniteGridValue {
            name: "large_component",
            index: 1,
            ..
        })
    ));

    assert_eq!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.0,
            new_delta: 0.025,
            large_component: zeros.view(),
            small_component: zeros.view(),
            output_len: 16,
        }),
        Err(GridError::InvalidDelta { delta: 0.0 })
    );
}

#[test]
fn fix_dirac_spinor_orbitals_grid_matches_feff_fixdsx_reference() -> Result<(), GridError> {
    let mut large = Array2::<Real>::zeros((251, 4).f());
    let mut small = Array2::<Real>::zeros((251, 4).f());
    for i in 1..=40 {
        let i_real = i as Real;
        large[(i - 1, 0)] = (0.07 * i_real).sin() * (-0.01 * i_real).exp();
        small[(i - 1, 0)] = (0.05 * i_real).cos() * (-0.02 * i_real).exp();
    }
    for i in 1..=75 {
        let i_real = i as Real;
        large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
        small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
    }
    for i in 1..=5 {
        let i_real = i as Real;
        large[(i - 1, 3)] = 0.05 * i_real;
        small[(i - 1, 3)] = -0.04 * i_real;
    }

    let result = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_components: large.view(),
        small_components: small.view(),
        output_len: 260,
    })?;

    assert_eq!(result.large_components.shape(), &[260, 4]);
    assert_eq!(result.large_components.strides(), &[1, 260]);
    assert_eq!(result.active_lengths.to_vec(), vec![81, 0, 151, 11]);
    assert_orbital_value(
        &result,
        1,
        1,
        0.069_246_904_378_467_77,
        0.978_973_680_203_922_3,
    );
    assert_orbital_value(&result, 81, 1, 0.0, 0.0);
    assert_orbital_value(&result, 82, 1, 0.0, 0.0);
    assert_orbital_value(&result, 1, 2, 0.0, 0.0);
    assert_orbital_value(&result, 100, 2, 0.0, 0.0);
    assert_orbital_value(
        &result,
        1,
        3,
        0.023_955_660_167_434_965,
        0.297_785_819_903_598_26,
    );
    assert_orbital_value(
        &result,
        150,
        3,
        0.228_834_221_332_933_4,
        0.130_219_461_349_623_98,
    );
    assert_orbital_value(&result, 151, 3, 0.0, 0.0);
    assert_orbital_value(&result, 1, 4, 0.05, -0.04);
    assert_orbital_value(&result, 11, 4, 0.0, 0.0);
    assert_orbital_value(&result, 12, 4, 0.0, 0.0);
    Ok(())
}

#[test]
fn fix_dirac_spinor_orbitals_grid_rejects_shape_mismatch() {
    let large = Array2::<Real>::zeros((4, 2));
    let small = Array2::<Real>::zeros((4, 3));

    assert_eq!(
        fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_components: large.view(),
            small_components: small.view(),
            output_len: 16,
        }),
        Err(GridError::SpinorShapeMismatch {
            large_rows: 4,
            large_columns: 2,
            small_rows: 4,
            small_columns: 3,
        })
    );
}
