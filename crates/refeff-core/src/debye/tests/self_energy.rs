use super::{support::*, *};

#[test]
fn dmdw_self_energy_matches_zero_energy_feff_identity() -> Result<(), DebyeError> {
    let temperature = 300.0;
    let pole_energy = ndarray::arr1(&[0.012, 0.024]);
    let pole_weight = ndarray::arr1(&[0.35, 0.65]);

    let self_energy = dmdw_self_energy_from_a2f_poles(
        temperature,
        Complex::new(0.0, 0.0),
        pole_energy.view(),
        pole_weight.view(),
    )?;
    let expected_imaginary = pole_energy
        .iter()
        .zip(pole_weight.iter())
        .map(|(&energy, &weight)| {
            let argument = energy / (DMDW_SELF_ENERGY_BOLTZMANN_EV_PER_K * temperature);
            -DMDW_SELF_ENERGY_TWO_PI * weight / argument.sinh()
        })
        .sum::<Real>();

    assert_complex_dmdw_close_tol(self_energy, Complex::new(0.0, expected_imaginary), 1.0e-10);
    Ok(())
}

#[test]
fn dmdw_self_energy_grid_matches_scalar_evaluation() -> Result<(), DebyeError> {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0, 7.0]),
        lanczos_weight: ndarray::arr1(&[0.4, 0.6]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.010, 0.030]),
        pole_weight: ndarray::arr1(&[0.15, 0.25]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.0225,
    };
    let energies = ndarray::arr1(&[-0.02, 0.0, 0.04]);

    let grid = dmdw_self_energy_grid_from_a2f_poles(450.0, energies.view(), &diagnostic)?;

    assert_eq!(grid.point_count(), energies.len());
    assert_vector_close(&grid.energy_ev, &[-0.02, 0.0, 0.04]);
    for (&energy, &actual) in energies.iter().zip(grid.self_energy.iter()) {
        let expected = dmdw_self_energy_from_a2f_poles(
            450.0,
            Complex::new(energy, 0.0),
            diagnostic.pole_energy_ev.view(),
            diagnostic.pole_weight.view(),
        )?;
        assert_complex_dmdw_close(actual, expected);
    }
    Ok(())
}

#[test]
fn dmdw_self_energy_rejects_invalid_inputs() {
    let energies = ndarray::arr1(&[0.01]);
    let weights = ndarray::arr1(&[0.2]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            0.0,
            Complex::new(0.0, 0.0),
            energies.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW self-energy temperature",
            ..
        })
    ));
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(Real::NAN, 0.0),
            energies.view(),
            weights.view()
        ),
        Err(DebyeError::NonFiniteComplex {
            name: "DMDW self-energy energy",
            ..
        })
    ));

    let short_weights = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            energies.view(),
            short_weights.view()
        ),
        Err(DebyeError::InvalidDmdwSelfEnergyPoleTableShape { .. })
    ));

    let empty = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(300.0, Complex::new(0.0, 0.0), empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwPoleTable)
    ));

    let zero_energy = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            zero_energy.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW self-energy pole energy",
            ..
        })
    ));

    let zero_weight = ndarray::arr1(&[0.0]);
    assert_eq!(
        dmdw_self_energy_from_a2f_poles(
            300.0,
            Complex::new(0.0, 0.0),
            zero_energy.view(),
            zero_weight.view()
        ),
        Ok(Complex::new(0.0, 0.0))
    );
}

#[test]
fn dmdw_self_energy_grid_rejects_empty_energy_grid() {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[1.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.01]),
        pole_weight: ndarray::arr1(&[0.2]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.01,
    };
    let empty = ndarray::arr1(&[]);

    assert!(matches!(
        dmdw_self_energy_grid_from_a2f_poles(300.0, empty.view(), &diagnostic),
        Err(DebyeError::EmptyDmdwSelfEnergyGrid)
    ));
}

#[test]
fn dmdw_spectral_function_handles_zero_coupling_symmetry() -> Result<(), DebyeError> {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.020]),
        pole_weight: ndarray::arr1(&[0.0]),
        mass_enhancement: 0.0,
        characteristic_energy_ev: 0.020,
    };
    let energy = ndarray::arr1(&[-1.0, -0.5, 0.0, 0.5, 1.0]);

    let spectral = dmdw_spectral_function_from_a2f_poles(
        300.0,
        energy.view(),
        0.0,
        diagnostic.characteristic_energy_ev,
        &diagnostic,
        20.0,
        101,
    )?;

    assert_eq!(spectral.point_count(), energy.len());
    assert_close(spectral.gamma_w0, 0.005);
    assert!(spectral.normalization.is_finite());
    assert!(spectral.normalization > 0.0);
    for value in &spectral.spectral_function {
        assert!(value.re.is_finite());
        assert!(value.im.is_finite());
    }
    assert_complex_dmdw_close_tol(
        spectral.spectral_function[0],
        spectral.spectral_function[4].conj(),
        1.0e-10,
    );
    assert_complex_dmdw_close_tol(
        spectral.spectral_function[1],
        spectral.spectral_function[3].conj(),
        1.0e-10,
    );
    Ok(())
}

#[test]
fn dmdw_spectral_function_rejects_invalid_grids() {
    let diagnostic = DmdwPoleWeightedA2f {
        lanczos_frequency_thz: ndarray::arr1(&[3.0]),
        lanczos_weight: ndarray::arr1(&[1.0]),
        normalization: 1.0,
        pole_energy_ev: ndarray::arr1(&[0.020]),
        pole_weight: ndarray::arr1(&[0.1]),
        mass_enhancement: 1.0,
        characteristic_energy_ev: 0.020,
    };
    let one_point = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            one_point.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            101,
        ),
        Err(DebyeError::InvalidDmdwSpectralEnergyGrid { points: 1 })
    ));

    let nonuniform = ndarray::arr1(&[-1.0, 0.0, 0.25]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            nonuniform.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            101,
        ),
        Err(DebyeError::NonUniformDmdwSpectralEnergyGrid { .. })
    ));

    let energy = ndarray::arr1(&[-1.0, 0.0, 1.0]);
    assert!(matches!(
        dmdw_spectral_function_from_a2f_poles(
            300.0,
            energy.view(),
            0.0,
            diagnostic.characteristic_energy_ev,
            &diagnostic,
            20.0,
            100,
        ),
        Err(DebyeError::InvalidDmdwSpectralTimeGrid { points: 100 })
    ));
}
