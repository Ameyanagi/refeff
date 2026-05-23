use super::{support::*, *};

#[test]
fn dmdw_mass_weighted_dynamical_matrix_matches_feff_make_dm() -> Result<(), DebyeError> {
    let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    blocks[(0, 0, 0, 0)] = 2.0;
    blocks[(0, 1, 0, 1)] = 3.0;
    blocks[(1, 0, 1, 0)] = 6.0;
    blocks[(1, 1, 2, 2)] = 18.0;
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let scale = 1_556.892_791_61 * 602.214_198_280;

    let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

    assert_eq!(result.matrix.shape(), &[6, 6]);
    assert_dmdw_close(result.matrix[(0, 0)], 0.5 * scale);
    assert_dmdw_close(result.matrix[(0, 3)], 0.5 * scale);
    assert_dmdw_close(result.matrix[(3, 0)], scale);
    assert_dmdw_close(result.matrix[(5, 5)], 2.0 * scale);
    assert_dmdw_close(result.average_value, scale / 9.0);
    assert_dmdw_close(result.average_asymmetry, scale / 36.0);
    assert_dmdw_close(result.asymmetry_percent_average, 25.0);
    assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 72.0);
    assert!(result.passes_feff_symmetry_check());
    Ok(())
}

#[test]
fn dmdw_mass_weighted_dynamical_matrix_reports_feff_asymmetry_warning() -> Result<(), DebyeError> {
    let mut blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    blocks[(0, 1, 0, 1)] = 6.0;
    let masses = ndarray::arr1(&[4.0, 9.0]);

    let result = dmdw_mass_weighted_dynamical_matrix(blocks.view(), masses.view())?;

    assert_dmdw_close(result.asymmetry_percent_average, 200.0);
    assert_dmdw_close(result.asymmetry_percent_max, 100.0 / 18.0);
    assert!(!result.passes_feff_symmetry_check());
    Ok(())
}

#[test]
fn dmdw_mass_weighted_dynamical_matrix_rejects_invalid_inputs() {
    let masses = ndarray::arr1(&[4.0, 9.0]);
    let bad_shape = ndarray::Array4::<Real>::zeros((1, 2, 3, 3));
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(bad_shape.view(), masses.view()),
        Err(DebyeError::InvalidDmdwBlockShape { .. })
    ));

    let empty_blocks = ndarray::Array4::<Real>::zeros((0, 0, 3, 3));
    let empty_masses = ndarray::Array1::<Real>::zeros(0);
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(empty_blocks.view(), empty_masses.view()),
        Err(DebyeError::EmptyDmdwAtomTable)
    ));

    let bad_masses = ndarray::arr1(&[4.0, 0.0]);
    let blocks = ndarray::Array4::<Real>::zeros((2, 2, 3, 3));
    assert!(matches!(
        dmdw_mass_weighted_dynamical_matrix(blocks.view(), bad_masses.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW atom mass",
            ..
        })
    ));
}

#[test]
fn dmdw_lanczos_coefficients_match_feff_recurrence() -> Result<(), DebyeError> {
    let matrix = ndarray::arr2(&[[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]);
    let seed = ndarray::arr1(&[1.0, 1.0, 1.0]);

    let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

    assert_vector_close(
        &coefficients.alpha,
        &[4.666_666_666_666_667, 5.639_455_782_312_925],
    );
    assert_vector_close(
        &coefficients.beta,
        &[0.0, 3.299_831_645_537_221_6, 2.120_878_539_880_258],
    );
    assert_dmdw_close(coefficients.single_pole_frequency, 0.343_813_972_349_477_75);
    Ok(())
}

#[test]
fn dmdw_lanczos_coefficients_preserve_feff_column_product() -> Result<(), DebyeError> {
    let matrix = ndarray::arr2(&[[1.0, 10.0], [0.0, 2.0]]);
    let seed = ndarray::arr1(&[1.0, 0.0]);

    let coefficients = dmdw_lanczos_coefficients(matrix.view(), seed.view(), 1)?;

    assert_vector_close(&coefficients.alpha, &[1.0, 2.0]);
    assert_vector_close(&coefficients.beta, &[0.0, 10.0, 10.0]);
    Ok(())
}

#[test]
fn dmdw_lanczos_coefficients_reject_invalid_inputs() {
    let matrix = ndarray::arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let seed = ndarray::arr1(&[1.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_coefficients(matrix.view(), seed.view(), 0),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole count",
            ..
        })
    ));

    let bad_matrix = ndarray::Array2::<Real>::zeros((2, 3));
    assert!(matches!(
        dmdw_lanczos_coefficients(bad_matrix.view(), seed.view(), 1),
        Err(DebyeError::InvalidDmdwLanczosShape { .. })
    ));

    let eigen_seed = ndarray::arr1(&[1.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_coefficients(matrix.view(), eigen_seed.view(), 1),
        Err(DebyeError::DmdwLanczosBreakdown { iteration: 1 })
    ));
}

#[test]
fn dmdw_lanczos_polynomials_match_feff_recurrences() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[4.666_666_666_666_667, 5.639_455_782_312_925]);
    let beta = ndarray::arr1(&[0.0, 3.299_831_645_537_221_6]);

    assert_dmdw_close(
        dmdw_lanczos_s_polynomial(2, 7.0, alpha.view(), beta.view())?,
        -7.714_285_714_285_713_5,
    );
    assert_dmdw_close(
        dmdw_lanczos_r_polynomial(2, 7.0, alpha.view(), beta.view())?,
        1.360_544_217_687_074_6,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial_derivative(2, 7.0, alpha.view(), beta.view())?,
        3.693_877_551_020_406_7,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial(1, 7.0, alpha.view(), beta.view())?,
        2.333_333_333_333_333,
    );
    assert_dmdw_close(
        dmdw_lanczos_r_polynomial(1, 7.0, alpha.view(), beta.view())?,
        1.0,
    );
    assert_dmdw_close(
        dmdw_lanczos_s_polynomial_derivative(1, 7.0, alpha.view(), beta.view())?,
        1.0,
    );
    Ok(())
}

#[test]
fn dmdw_lanczos_polynomials_reject_invalid_inputs() {
    let alpha = ndarray::arr1(&[1.0]);
    let beta = ndarray::arr1(&[0.0]);
    assert!(matches!(
        dmdw_lanczos_s_polynomial(0, 1.0, alpha.view(), beta.view()),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_s_polynomial(2, 1.0, alpha.view(), beta.view()),
        Err(DebyeError::InvalidDmdwLanczosPolynomialShape { .. })
    ));
    assert!(matches!(
        dmdw_lanczos_s_polynomial(1, Real::NAN, alpha.view(), beta.view()),
        Err(DebyeError::NonFinite {
            name: "DMDW Lanczos polynomial x",
            ..
        })
    ));
}

#[test]
fn dmdw_lanczos_pole_spectrum_matches_feff_scan() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[16.0, 16.0]);
    let beta = ndarray::arr1(&[0.0, 8.0]);
    let spectrum =
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

    assert!(spectrum.has_expected_pole_count());
    assert_vector_close(&spectrum.squared_angular_frequencies, &[8.0, 24.0]);
    assert_vector_close(
        &spectrum.angular_frequencies,
        &[8.0_f64.sqrt(), 24.0_f64.sqrt()],
    );
    assert_vector_close(
        &spectrum.frequencies,
        &[
            8.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
            24.0_f64.sqrt() / (2.0 * std::f64::consts::PI),
        ],
    );
    assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
    assert!(spectrum.imaginary_warnings.is_empty());
    Ok(())
}

#[test]
fn dmdw_lanczos_pole_spectrum_reports_imaginary_weight_warnings() -> Result<(), DebyeError> {
    let alpha = ndarray::arr1(&[-16.0, -16.0]);
    let beta = ndarray::arr1(&[0.0, 8.0]);
    let spectrum =
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 32.0, 8192)?;

    assert!(spectrum.has_expected_pole_count());
    assert_vector_close(&spectrum.squared_angular_frequencies, &[-24.0, -8.0]);
    assert_vector_close(
        &spectrum.angular_frequencies,
        &[-24.0_f64.sqrt(), -8.0_f64.sqrt()],
    );
    assert_vector_close(&spectrum.weights, &[0.5, 0.5]);
    assert_eq!(spectrum.imaginary_warnings.len(), 2);
    assert_eq!(
        spectrum.imaginary_warnings[0].severity,
        DmdwImaginaryPoleSeverity::LargeWeight
    );
    assert_eq!(spectrum.imaginary_warnings[0].pole_index, 0);
    assert_dmdw_close(spectrum.imaginary_warnings[0].weight, 0.5);
    Ok(())
}

#[test]
fn dmdw_lanczos_pole_spectrum_rejects_invalid_inputs() {
    let alpha = ndarray::arr1(&[1.0, 1.0]);
    let beta = ndarray::arr1(&[0.0, 0.0]);
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(0, alpha.view(), beta.view(), 2.0, 1),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos polynomial order",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 0.0, 1),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole search limit",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(1, alpha.view(), beta.view(), 2.0, 0),
        Err(DebyeError::NonPositive {
            name: "DMDW Lanczos pole samples per pole",
            ..
        })
    ));
    assert!(matches!(
        dmdw_lanczos_pole_spectrum_with_search(2, alpha.view(), beta.view(), 2.0, 2),
        Err(DebyeError::ZeroDmdwLanczosPoleDerivative { .. })
    ));
}
