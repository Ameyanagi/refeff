use super::{support::*, *};

#[test]
fn dmdw_debye_waller_factors_from_poles_match_feff_accumulation() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[300.0, 600.0]);
    let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
    let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
    let factors = dmdw_debye_waller_factors_from_poles(
        temperatures.view(),
        5.0,
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(&factors, &[5.459_186_287_610_058, 10.914_330_842_743_967]);
    Ok(())
}

#[test]
fn dmdw_debye_waller_factors_use_zero_temperature_coth_limit() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[0.001]);
    let angular_frequencies = ndarray::arr1(&[2.0]);
    let weights = ndarray::arr1(&[1.0]);
    let factors = dmdw_debye_waller_factors_from_poles(
        temperatures.view(),
        5.0,
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(&factors, &[0.317_544_517_206_879_8]);
    Ok(())
}

#[test]
fn dmdw_vibrational_free_energy_from_poles_matches_feff_accumulation() -> Result<(), DebyeError> {
    let temperatures = ndarray::arr1(&[300.0, 600.0]);
    let angular_frequencies = ndarray::arr1(&[2.0, 4.0, -3.0, 0.0]);
    let weights = ndarray::arr1(&[0.25, 0.75, 0.9, 10.0]);
    let free_energy = dmdw_vibrational_free_energy_from_poles(
        temperatures.view(),
        angular_frequencies.view(),
        weights.view(),
    )?;

    assert_vector_close(
        &free_energy,
        &[-6_129.431_830_672_452, -15_718.169_449_997_833],
    );
    Ok(())
}

#[test]
fn dmdw_einstein_and_moment_summaries_match_feff_print_formulas() -> Result<(), DebyeError> {
    let reduced_mass = 10.0;
    let summary = dmdw_single_pole_einstein_summary(3.5, reduced_mass)?;
    assert_dmdw_close(summary.frequency_thz, 3.5);
    assert_dmdw_close(summary.temperature_kelvin, 3.5 * DMDW_THZ_TO_KELVIN);
    assert_dmdw_close(
        summary.effective_force_constant_n_per_m,
        reduced_mass
            * (2.0 * std::f64::consts::PI * 3.5).powi(2)
            * DMDW_AMU_THZ2_TO_NEWTON_PER_METER,
    );

    let frequencies = ndarray::arr1(&[-1.0, 2.0, 4.0]);
    let weights = ndarray::arr1(&[0.2, 0.2, 0.6]);
    let moments =
        dmdw_moment_summaries_from_poles(reduced_mass, frequencies.view(), weights.view())?;

    assert_eq!(
        moments
            .iter()
            .map(|moment| moment.order)
            .collect::<Vec<_>>(),
        vec![-2, -1, 0, 1, 2]
    );
    assert_moment_summary(
        &moments[0],
        0.109_375,
        0.109_375_f64.powf(-0.5),
        reduced_mass,
    )?;
    assert_moment_summary(&moments[1], 0.312_5, 3.2, reduced_mass)?;
    assert_dmdw_close(moments[2].moment_thz_power_n, 1.0);
    assert_eq!(moments[2].frequency_thz, None);
    assert_eq!(moments[2].temperature_kelvin, None);
    assert_eq!(moments[2].effective_force_constant_n_per_m, None);
    assert_moment_summary(&moments[3], 3.5, 3.5, reduced_mass)?;
    assert_moment_summary(&moments[4], 13.0, 13.0_f64.sqrt(), reduced_mass)?;
    Ok(())
}

#[test]
fn dmdw_pole_thermal_helpers_reject_invalid_inputs() {
    let temperatures = ndarray::arr1(&[300.0]);
    let frequencies = ndarray::arr1(&[1.0, 2.0]);
    let weights = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            1.0,
            frequencies.view(),
            weights.view()
        ),
        Err(DebyeError::InvalidDmdwPoleTableShape { .. })
    ));

    let empty_temperatures = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_vibrational_free_energy_from_poles(
            empty_temperatures.view(),
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::EmptyDmdwTemperatureTable)
    ));

    let bad_temperatures = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_vibrational_free_energy_from_poles(
            bad_temperatures.view(),
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW temperature",
            ..
        })
    ));

    assert!(matches!(
        dmdw_debye_waller_factors_from_poles(
            temperatures.view(),
            0.0,
            weights.view(),
            weights.view()
        ),
        Err(DebyeError::NonPositive {
            name: "DMDW reduced mass",
            ..
        })
    ));
}

#[test]
fn dmdw_pole_summary_helpers_reject_invalid_inputs() {
    assert!(matches!(
        dmdw_single_pole_einstein_summary(0.0, 1.0),
        Err(DebyeError::NonPositive {
            name: "DMDW Einstein frequency",
            ..
        })
    ));

    let empty = ndarray::arr1(&[]);
    assert!(matches!(
        dmdw_moment_summaries_from_poles(1.0, empty.view(), empty.view()),
        Err(DebyeError::EmptyDmdwPoleTable)
    ));

    let imaginary_frequencies = ndarray::arr1(&[-1.0]);
    let weights = ndarray::arr1(&[1.0]);
    assert!(matches!(
        dmdw_moment_summaries_from_poles(1.0, imaginary_frequencies.view(), weights.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW positive pole weight normalization",
            ..
        })
    ));
}
