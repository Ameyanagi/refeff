use super::{
    AngularError, BasisTransformMode, PolarizationTensorMode, TransitionBMatrixInput,
    basis_transform_matrices, change_basis_representation, legendre_normalization,
    legendre_normalization_table, legendre_polynomials, mkgtr_clebsch_gordan_coefficients,
    polarization_tensor, relativistic_clebsch_gordan_coefficients, relativistic_state_index_1based,
    spherical_harmonics, spin_orbit_coupling_tables, transition_b_matrix, wigner_3j,
    wigner_rotation,
};
use crate::Complex;
use ndarray::{Array2, ArrayView2, ShapeBuilder, arr2};
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

#[test]
fn computes_snlm_values() -> Result<(), AngularError> {
    assert_close(legendre_normalization(0, 0)?, 1.0);
    assert_close(legendre_normalization(1, 0)?, 3.0_f64.sqrt());
    assert_close(legendre_normalization(1, 1)?, (3.0_f64 / 2.0).sqrt());
    assert_close(legendre_normalization(2, 2)?, (5.0_f64 / 24.0).sqrt());
    assert_eq!(legendre_normalization(1, 2)?, 0.0);
    Ok(())
}

#[test]
fn computes_cpl0_legendre_polynomials() {
    let values = legendre_polynomials(0.25, 4);
    let expected = [1.0, 0.25, -0.40625, -0.3359375, 0.15771484375];

    for (&actual, expected) in values.iter().zip(expected) {
        assert_close(actual, expected);
    }
}

#[test]
fn builds_fortran_order_xnlm_table() -> Result<(), AngularError> {
    let table = legendre_normalization_table(3)?;

    assert_eq!(table.shape(), &[4, 4]);
    assert_eq!(table.strides(), &[1, 4]);
    assert_close(table[[0, 2]], 5.0_f64.sqrt());
    assert_close(table[[2, 2]], (5.0_f64 / 24.0).sqrt());
    assert_eq!(table[[3, 2]], 0.0);
    Ok(())
}

#[test]
fn computes_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
    assert_close(wigner_3j(1, 1, 0, 0, 0, 1)?, -1.0 / 3.0_f64.sqrt());
    assert_eq!(wigner_3j(1, 1, 3, 0, 0, 1)?, 0.0);
    Ok(())
}

#[test]
fn computes_half_integer_wigner_3j_coefficients() -> Result<(), AngularError> {
    assert_close(wigner_3j(0, 1, 1, 0, -1, 2)?, -1.0 / 2.0_f64.sqrt());
    assert_eq!(
        wigner_3j(2, 2, 2, 1, 0, 2),
        Err(AngularError::InvalidWignerParity {
            argument: 3,
            scale: 2,
        })
    );
    Ok(())
}

#[test]
fn computes_wigner_rotation_elements() -> Result<(), AngularError> {
    assert_close(wigner_rotation(0.7, 2, 1, -1, 1)?, 0.2974375221921237);
    assert_close(wigner_rotation(1.1, 3, -2, 1, 1)?, 0.4544222701103565);
    assert_close(wigner_rotation(0.7, 3, 1, -1, 2)?, -0.5648429673316498);
    assert_close(wigner_rotation(-0.9, 5, -3, 1, 2)?, 0.494867123375203);
    assert_close(wigner_rotation(0.4, 4, -2, -4, 2)?, -0.3740481938792059);
    Ok(())
}

#[test]
fn rejects_invalid_wigner_rotation_inputs() {
    assert_eq!(
        wigner_rotation(0.1, 1, 0, 0, 3),
        Err(AngularError::InvalidWignerScale { scale: 3 })
    );
    assert_eq!(
        wigner_rotation(f64::NAN, 1, 0, 0, 1),
        Err(AngularError::NonFiniteRotationAngle)
    );
    assert!(matches!(
        wigner_rotation(0.1, 3, 1, 0, 2),
        Err(AngularError::InvalidWignerParity { .. })
    ));
}

#[test]
fn computes_feff_spherical_harmonics() -> Result<(), AngularError> {
    let values = spherical_harmonics([1.0, 2.0, 3.0], 3)?;
    let expected = [
        Complex::new(0.28209478784887687, 0.0),
        Complex::new(0.09233719417642765, -0.1846743883528553),
        Complex::new(0.3917535369473459, 0.0),
        Complex::new(-0.09233719417642765, -0.1846743883528553),
        Complex::new(-0.08277304213899862, -0.1103640561853315),
        Complex::new(0.16554608427799725, -0.3310921685559945),
        Complex::new(0.29286359223107555, 0.0),
        Complex::new(-0.16554608427799725, -0.3310921685559945),
        Complex::new(-0.08277304213899862, 0.1103640561853315),
        Complex::new(-0.08761323662775768, 0.015929679386865035),
        Complex::new(-0.17558813818777735, -0.23411751758370314),
        Complex::new(0.1912556872248307, -0.3825113744496614),
        Complex::new(0.0641157227438533, 0.0),
        Complex::new(-0.1912556872248307, -0.3825113744496614),
        Complex::new(-0.17558813818777735, 0.23411751758370314),
        Complex::new(0.08761323662775768, 0.015929679386865035),
    ];

    assert_complex_iter_close(values.iter().copied(), &expected);
    Ok(())
}

#[test]
fn spherical_harmonics_match_feff_axis_and_zero_vector_cases() -> Result<(), AngularError> {
    let xz = spherical_harmonics([-0.5, 0.0, 2.0], 2)?;
    assert_complex_iter_close(
        xz.iter().copied(),
        &[
            Complex::new(0.28209478784887687, 0.0),
            Complex::new(-0.08379463832252748, 0.0),
            Complex::new(0.47401405587946666, 0.0),
            Complex::new(0.08379463832252748, 0.0),
            Complex::new(0.022722011567568246, 0.0),
            Complex::new(-0.18177609254054597, 0.0),
            Complex::new(0.5751257874583111, 0.0),
            Complex::new(0.18177609254054597, 0.0),
            Complex::new(0.022722011567568246, 0.0),
        ],
    );

    let zero = spherical_harmonics([0.0, 0.0, 0.0], 2)?;
    assert_complex_iter_close(
        zero.iter().copied(),
        &[
            Complex::new(0.28209478784887687, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.4886025051046183, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.6307831217284703, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    );
    Ok(())
}

#[test]
fn rejects_non_finite_spherical_harmonic_vector() {
    assert_eq!(
        spherical_harmonics([f64::NAN, 0.0, 1.0], 2),
        Err(AngularError::NonFiniteVector)
    );
}

#[test]
fn builds_feff_spherical_polarization_tensors() -> Result<(), AngularError> {
    for selector in 1..=9 {
        let tensor = polarization_tensor(selector, PolarizationTensorMode::Spherical)?;
        let selected = selector - 1;
        for row in -1..=1 {
            for column in -1..=1 {
                let index = polarization_test_index(row) * 3 + polarization_test_index(column);
                let expected = if index == selected {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert_tensor_entry(tensor.view(), row, column, expected);
            }
        }
    }

    let averaged = polarization_tensor(10, PolarizationTensorMode::Spherical)?;
    assert_tensor_nonzeros(
        averaged.view(),
        &[
            (-1, -1, Complex::new(1.0 / 3.0, 0.0)),
            (0, 0, Complex::new(1.0 / 3.0, 0.0)),
            (1, 1, Complex::new(1.0 / 3.0, 0.0)),
        ],
    );
    Ok(())
}

#[test]
fn builds_feff_cartesian_polarization_tensors() -> Result<(), AngularError> {
    let half = 0.5;
    let root_half = 1.0 / 2.0_f64.sqrt();
    let one = Complex::new(1.0, 0.0);
    let imaginary = Complex::new(0.0, 1.0);

    assert_tensor_nonzeros(
        polarization_tensor(1, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, -1, one * half),
            (-1, 1, -one * half),
            (1, -1, -one * half),
            (1, 1, one * half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(2, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, -1, -imaginary * half),
            (-1, 1, -imaginary * half),
            (1, -1, imaginary * half),
            (1, 1, imaginary * half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(3, PolarizationTensorMode::Cartesian)?.view(),
        &[(-1, 0, one * root_half), (1, 0, -one * root_half)],
    );
    assert_tensor_nonzeros(
        polarization_tensor(4, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, -1, imaginary * half),
            (-1, 1, -imaginary * half),
            (1, -1, imaginary * half),
            (1, 1, -imaginary * half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(5, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, -1, one * half),
            (-1, 1, one * half),
            (1, -1, one * half),
            (1, 1, one * half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(6, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, 0, -imaginary * root_half),
            (1, 0, -imaginary * root_half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(7, PolarizationTensorMode::Cartesian)?.view(),
        &[(0, -1, one * root_half), (0, 1, -one * root_half)],
    );
    assert_tensor_nonzeros(
        polarization_tensor(8, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (0, -1, imaginary * root_half),
            (0, 1, imaginary * root_half),
        ],
    );
    assert_tensor_nonzeros(
        polarization_tensor(9, PolarizationTensorMode::Cartesian)?.view(),
        &[(0, 0, one)],
    );
    assert_tensor_nonzeros(
        polarization_tensor(10, PolarizationTensorMode::Cartesian)?.view(),
        &[
            (-1, -1, Complex::new(1.0 / 3.0, 0.0)),
            (0, 0, Complex::new(1.0 / 3.0, 0.0)),
            (1, 1, Complex::new(1.0 / 3.0, 0.0)),
        ],
    );
    Ok(())
}

#[test]
fn rejects_invalid_polarization_tensor_selector() {
    assert_eq!(
        polarization_tensor(0, PolarizationTensorMode::Spherical),
        Err(AngularError::InvalidPolarizationTensorIndex { index: 0 })
    );
    assert_eq!(
        polarization_tensor(11, PolarizationTensorMode::Cartesian),
        Err(AngularError::InvalidPolarizationTensorIndex { index: 11 })
    );
}

#[test]
fn builds_feff_averaged_transition_b_matrix() -> Result<(), AngularError> {
    let result = transition_b_matrix(TransitionBMatrixInput {
        polarization: 0,
        ..sample_transition_b_matrix_input()
    })?;

    assert_eq!(result.kappa_indices, [-2, 1, 0, 0, 0, -1, 2, -3]);
    assert_eq!(result.orbital_momenta, [1, 1, -1, -1, -1, 0, 2, 2]);
    assert_transition_values(
        &result,
        &[
            Complex::new(-0.05555555555555555, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.02, 0.0),
            Complex::new(0.02, 0.0),
        ],
    );
    assert_complex_close(
        matrix_sum(&result.matrix),
        Complex::new(-0.06666666666666678, 0.0),
    );
    assert_eq!(nonzero_count(&result.matrix), 34);
    Ok(())
}

#[test]
fn builds_feff_polarized_transition_b_matrix() -> Result<(), AngularError> {
    let result = transition_b_matrix(sample_transition_b_matrix_input())?;

    assert_eq!(result.kappa_indices, [-2, 1, 0, 0, 0, -1, 2, -3]);
    assert_eq!(result.orbital_momenta, [1, 1, -1, -1, -1, 0, 2, 2]);
    assert_transition_values(
        &result,
        &[
            Complex::new(-0.06273441744900005, -0.002816883591316175),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.028021517059414094, 0.0012286136649168643),
            Complex::new(0.032783586221432195, 0.001621399287040505),
        ],
    );
    assert_complex_close(
        matrix_sum(&result.matrix),
        Complex::new(-0.1498383428670292, 1.0054312926918387),
    );
    assert_eq!(nonzero_count(&result.matrix), 776);
    Ok(())
}

#[test]
fn applies_feff_trace_and_spin_folding_to_transition_b_matrix() -> Result<(), AngularError> {
    let traced = transition_b_matrix(TransitionBMatrixInput {
        trace_orbital: true,
        ..sample_transition_b_matrix_input()
    })?;
    assert_transition_values(
        &traced,
        &[
            Complex::new(-0.13927806177774438, -0.006120603534129609),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.08851128918371495, -0.0038584833449001833),
            Complex::new(0.08464910571422107, 0.0008909740920669523),
        ],
    );
    assert_complex_close(
        matrix_sum(&traced.matrix),
        Complex::new(-0.31619310063452616, 0.8078421709741431),
    );
    assert_eq!(nonzero_count(&traced.matrix), 728);

    let spin_folded = transition_b_matrix(TransitionBMatrixInput {
        spin: 0,
        ..sample_transition_b_matrix_input()
    })?;
    assert_transition_values(
        &spin_folded,
        &[
            Complex::new(-0.12775761018690238, -0.0018521924356471728),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.05865146715590615, -0.0005346566843031169),
            Complex::new(0.06730612780091635, 0.0012482092313231155),
        ],
    );
    assert_complex_close(
        matrix_sum(&spin_folded.matrix),
        Complex::new(-0.2170418650429801, 1.3409109998393136),
    );
    assert_eq!(nonzero_count(&spin_folded.matrix), 828);
    Ok(())
}

#[test]
fn builds_feff_magnetic_dipole_transition_b_matrix() -> Result<(), AngularError> {
    let result = transition_b_matrix(TransitionBMatrixInput {
        initial_kappa: 2,
        multipole: 1,
        spin: 2,
        spin_vector_angle: -0.4,
        ..sample_transition_b_matrix_input()
    })?;

    assert_eq!(result.kappa_indices, [1, -2, 3, 0, -1, 2, -3, 0]);
    assert_eq!(result.orbital_momenta, [1, 1, 3, -1, 0, 2, 2, -1]);
    assert_transition_values(
        &result,
        &[
            Complex::new(-0.04864195024406099, 0.003463869265078493),
            Complex::new(0.0017631363458188638, -0.0002788047964405202),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.035984450619728485, 0.0004976156588231403),
            Complex::new(0.0, 0.0),
        ],
    );
    assert_complex_close(
        matrix_sum(&result.matrix),
        Complex::new(0.21773917179567237, 1.463795753151296),
    );
    assert_eq!(nonzero_count(&result.matrix), 1432);
    Ok(())
}

#[test]
fn rejects_invalid_transition_b_matrix_inputs() {
    assert_eq!(
        transition_b_matrix(TransitionBMatrixInput {
            spin_channels: 0,
            ..sample_transition_b_matrix_input()
        }),
        Err(AngularError::InvalidSpinChannelCount { value: 0 })
    );

    let mut input = sample_transition_b_matrix_input();
    input.polarization_tensor[0][2] = Complex::new(f64::NAN, 0.0);
    assert_eq!(
        transition_b_matrix(input),
        Err(AngularError::NonFinitePolarizationTensor { row: -1, column: 1 })
    );
}

#[test]
fn builds_spin_orbit_coupling_tables() -> Result<(), AngularError> {
    let tables = spin_orbit_coupling_tables(1)?;

    assert_eq!(tables.plus.shape(), &[2, 3, 2]);
    assert_eq!(tables.plus.strides(), &[1, 2, 6]);
    assert_eq!(tables.m_offset, 1);
    assert_close(tables.plus[[0, 1, 0]], 1.0);
    assert_close(tables.plus[[0, 1, 1]], 1.0);
    assert_close(tables.minus[[0, 1, 0]], 0.0);
    Ok(())
}

#[test]
fn relativistic_clebsch_gordan_coefficients_match_feff_calccgc_reference()
-> Result<(), AngularError> {
    let tables = relativistic_clebsch_gordan_coefficients(2)?;

    assert_eq!(tables.orbital_momentum, vec![0, 1, 1, 2, 2]);
    assert_eq!(tables.kappa, vec![-1, 1, -2, 2, -3]);
    assert_eq!(tables.spin_multiplicity, vec![2, 2, 4, 4, 6]);
    assert_eq!(tables.coefficients.shape(), &[18, 2]);
    assert_eq!(tables.coefficients.strides(), &[1, 18]);
    assert_real_matrix_close(
        tables.coefficients.view(),
        arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.577_350_269_190, -0.816_496_580_928],
            [0.816_496_580_928, -0.577_350_269_190],
            [1.0, 0.0],
            [0.816_496_580_928, 0.577_350_269_190],
            [0.577_350_269_190, 0.816_496_580_928],
            [0.0, 1.0],
            [0.447_213_595_500, -0.894_427_191_000],
            [0.632_455_532_034, -0.774_596_669_241],
            [0.774_596_669_241, -0.632_455_532_034],
            [0.894_427_191_000, -0.447_213_595_500],
            [1.0, 0.0],
            [0.894_427_191_000, 0.447_213_595_500],
            [0.774_596_669_241, 0.632_455_532_034],
            [0.632_455_532_034, 0.774_596_669_241],
            [0.447_213_595_500, 0.894_427_191_000],
            [0.0, 1.0],
        ])
        .view(),
    );
    Ok(())
}

#[test]
fn relativistic_clebsch_gordan_coefficients_rejects_oversized_lmax() {
    assert_eq!(
        relativistic_clebsch_gordan_coefficients(usize::MAX),
        Err(AngularError::IndexTooLarge { value: usize::MAX })
    );
}

#[test]
fn mkgtr_clebsch_gordan_coefficients_match_feff_calclbcoef_reference() -> Result<(), AngularError> {
    let coefficients = mkgtr_clebsch_gordan_coefficients(2, 3, 6)?;
    let expected = [
        (1, 1, 0, 0, FRAC_1_SQRT_2),
        (2, 1, 1, 0, FRAC_1_SQRT_2),
        (1, 1, 0, 1, 4.082_482_904_638_63e-1),
        (2, 1, 0, 1, 5.773_502_691_896_258e-1),
        (1, 2, 0, 1, 5.000_000_000_000_001e-1),
        (2, 2, 0, 1, 4.082_482_904_638_631_3e-1),
        (3, 2, 0, 1, 2.886_751_345_948_129e-1),
        (1, 1, 1, 1, -5.773_502_691_896_258e-1),
        (2, 1, 1, 1, -4.082_482_904_638_63e-1),
        (2, 2, 1, 1, 2.886_751_345_948_129e-1),
        (3, 2, 1, 1, 4.082_482_904_638_631_3e-1),
        (4, 2, 1, 1, 5.000_000_000_000_001e-1),
        (1, 2, 0, 2, 2.236_067_977_499_79e-1),
        (2, 2, 0, 2, 3.162_277_660_168_38e-1),
        (3, 2, 0, 2, 3.872_983_346_207_417_6e-1),
        (4, 2, 0, 2, 4.472_135_954_999_579e-1),
        (1, 3, 0, 2, 4.082_482_904_638_629_6e-1),
        (2, 3, 0, 2, 3.651_483_716_701_106e-1),
        (3, 3, 0, 2, 3.162_277_660_168_378_3e-1),
        (4, 3, 0, 2, 2.581_988_897_471_610_4e-1),
        (5, 3, 0, 2, 1.825_741_858_350_553_3e-1),
        (1, 2, 1, 2, -4.472_135_954_999_579e-1),
        (2, 2, 1, 2, -3.872_983_346_207_417_6e-1),
        (3, 2, 1, 2, -3.162_277_660_168_38e-1),
        (4, 2, 1, 2, -2.236_067_977_499_79e-1),
        (2, 3, 1, 2, 1.825_741_858_350_553_3e-1),
        (3, 3, 1, 2, 2.581_988_897_471_610_4e-1),
        (4, 3, 1, 2, 3.162_277_660_168_379e-1),
        (5, 3, 1, 2, 3.651_483_716_701_106e-1),
        (6, 3, 1, 2, 4.082_482_904_638_629_6e-1),
    ];

    assert_eq!(coefficients.shape(), &[6, 3, 2, 3]);
    assert_eq!(coefficients.strides(), &[1, 6, 18, 36]);
    for (im, ii, is, ll, expected) in expected {
        assert_close(coefficients[(im - 1, ii - 1, is, ll)], expected);
    }
    assert_eq!(
        coefficients
            .iter()
            .filter(|&&coefficient| coefficient.abs() > 1.0e-14)
            .count(),
        expected.len()
    );
    assert_close(coefficients[(0, 1, 0, 0)], 0.0);
    Ok(())
}

#[test]
fn mkgtr_clebsch_gordan_coefficients_reject_invalid_dimensions() {
    assert_eq!(
        mkgtr_clebsch_gordan_coefficients(2, 0, 6),
        Err(AngularError::InvalidAngularTableDimension {
            name: "j_lmax",
            value: 0,
            minimum: 1,
        })
    );
    assert_eq!(
        mkgtr_clebsch_gordan_coefficients(2, 3, 5),
        Err(AngularError::InvalidAngularTableDimension {
            name: "mj_lmax",
            value: 5,
            minimum: 6,
        })
    );
}

#[test]
fn relativistic_state_index_matches_feff_ikapmue_reference() -> Result<(), AngularError> {
    let cases = [
        (-1, -1, 1),
        (-1, 0, 2),
        (1, -1, 3),
        (1, 0, 4),
        (-2, -2, 5),
        (-2, -1, 6),
        (-2, 0, 7),
        (-2, 1, 8),
        (2, -2, 9),
        (2, -1, 10),
        (2, 0, 11),
        (2, 1, 12),
        (-3, -3, 13),
        (-3, -2, 14),
        (-3, -1, 15),
        (-3, 0, 16),
        (-3, 1, 17),
        (-3, 2, 18),
    ];

    for (kappa, mu_minus_half, expected) in cases {
        assert_eq!(
            relativistic_state_index_1based(kappa, mu_minus_half)?,
            expected
        );
    }
    Ok(())
}

#[test]
fn relativistic_state_index_rejects_invalid_quantum_numbers() {
    assert_eq!(
        relativistic_state_index_1based(0, 0),
        Err(AngularError::InvalidRelativisticKappa { kappa: 0 })
    );
    assert_eq!(
        relativistic_state_index_1based(i32::MIN, 0),
        Err(AngularError::InvalidRelativisticKappa { kappa: i32::MIN })
    );
    assert_eq!(
        relativistic_state_index_1based(-2, -3),
        Err(AngularError::RelativisticMagneticIndexOutOfRange {
            kappa: -2,
            mu_minus_half: -3,
        })
    );
    assert_eq!(
        relativistic_state_index_1based(-2, 2),
        Err(AngularError::RelativisticMagneticIndexOutOfRange {
            kappa: -2,
            mu_minus_half: 2,
        })
    );
}

#[test]
fn basis_transform_matrices_match_feff_bastrmat_reference() -> Result<(), AngularError> {
    let transforms = basis_transform_matrices(1)?;

    assert_eq!(transforms.lmax, 1);
    assert_eq!(transforms.order, 8);
    assert_matrix_summary(
        transforms.real_to_complex.view(),
        MatrixSummary {
            count: 12,
            total: Complex::new(4.0, -2.828_427_124_746),
            trace: Complex::new(4.0 - SQRT_2, -SQRT_2),
            weighted: Complex::new(12.464_466_094_067, -14.606_601_717_798),
            norm2: 8.0,
        },
        &[
            (1, 1, Complex::new(1.0, 0.0)),
            (1, 2, Complex::new(0.0, 0.0)),
            (2, 1, Complex::new(0.0, 0.0)),
            (5, 5, Complex::new(1.0, 0.0)),
            (8, 8, Complex::new(-FRAC_1_SQRT_2, 0.0)),
        ],
    );
    assert_matrix_summary(
        transforms.complex_to_relativistic.view(),
        MatrixSummary {
            count: 12,
            total: Complex::new(6.787_693_700_235, 0.0),
            trace: Complex::new(4.787_693_700_235, 0.0),
            weighted: Complex::new(25.996_074_262_560, -8.589_788_840_816),
            norm2: 8.0,
        },
        &[
            (1, 1, Complex::new(1.0, 0.0)),
            (1, 2, Complex::new(0.0, 0.0)),
            (2, 1, Complex::new(0.0, 0.0)),
            (5, 5, Complex::new(0.0, 0.0)),
            (8, 8, Complex::new(1.0, 0.0)),
        ],
    );
    assert_matrix_summary(
        transforms.real_to_relativistic.view(),
        MatrixSummary {
            count: 18,
            total: Complex::new(2.478_292_623_476, -2.230_710_143_301),
            trace: Complex::new(1.109_389_799_741, -0.408_248_290_464),
            weighted: Complex::new(-0.037_314_010_809, -8.229_960_687_575),
            norm2: 8.0,
        },
        &[
            (1, 1, Complex::new(1.0, 0.0)),
            (1, 2, Complex::new(0.0, 0.0)),
            (2, 1, Complex::new(0.0, 0.0)),
            (5, 5, Complex::new(0.0, 0.0)),
            (8, 8, Complex::new(-FRAC_1_SQRT_2, 0.0)),
        ],
    );
    Ok(())
}

#[test]
fn change_basis_representation_matches_feff_changerep_reference() -> Result<(), AngularError> {
    let transforms = basis_transform_matrices(1)?;
    let input = sample_basis_transform_input(transforms.order);
    let cases = [
        (
            BasisTransformMode::RelativisticToReal,
            MatrixSummary {
                count: 64,
                total: Complex::new(4.390_729_316_824, 0.876_977_453_087),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(17.449_943_098_879, -7.739_339_036_806),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(-0.575_333_004_299, 0.430_350_724_562)),
                (2, 1, Complex::new(-0.055_564_086_460, -0.921_048_461_819)),
                (5, 5, Complex::new(0.26, 0.10)),
                (8, 8, Complex::new(0.562_634_665_738, 0.216_397_948_361)),
            ],
        ),
        (
            BasisTransformMode::RealToRelativistic,
            MatrixSummary {
                count: 64,
                total: Complex::new(3.846_058_831_001, 1.764_637_032_210),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(20.413_098_444_828, 11.817_486_246_457),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(0.25, 0.33)),
                (2, 1, Complex::new(0.53, -0.03)),
                (5, 5, Complex::new(0.30, 0.08)),
                (8, 8, Complex::new(1.0, 0.42)),
            ],
        ),
        (
            BasisTransformMode::RelativisticToComplex,
            MatrixSummary {
                count: 64,
                total: Complex::new(30.318_524_912_599, 11.660_971_120_231),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(153.171_238_628_134, 9.571_742_854_461),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(0.25, 0.33)),
                (2, 1, Complex::new(0.53, -0.03)),
                (5, 5, Complex::new(0.26, 0.10)),
                (8, 8, Complex::new(1.04, 0.40)),
            ],
        ),
        (
            BasisTransformMode::ComplexToRelativistic,
            MatrixSummary {
                count: 64,
                total: Complex::new(22.938_940_635_365, 8.822_669_475_140),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(153.171_238_628_134, 9.571_742_854_461),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(0.25, 0.33)),
                (2, 1, Complex::new(0.53, -0.03)),
                (5, 5, Complex::new(0.26, 0.10)),
                (8, 8, Complex::new(1.04, 0.40)),
            ],
        ),
        (
            BasisTransformMode::ComplexToReal,
            MatrixSummary {
                count: 60,
                total: Complex::new(10.310_984_130_223, 3.282_354_980_122),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(54.968_991_359_991, -0.025_929_291_126),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(-0.268_700_576_851, 0.268_700_576_851)),
                (2, 1, Complex::new(0.014_142_135_624, -0.466_690_475_583)),
                (5, 5, Complex::new(0.65, 0.25)),
                (8, 8, Complex::new(0.0, 0.0)),
            ],
        ),
        (
            BasisTransformMode::RealToComplex,
            MatrixSummary {
                count: 64,
                total: Complex::new(12.48, 4.8),
                trace: Complex::new(4.68, 1.8),
                weighted: Complex::new(57.017_519_497_415, 13.494_356_415_872),
                norm2: 30.5856,
            },
            [
                (1, 1, Complex::new(0.13, 0.05)),
                (1, 2, Complex::new(0.240_416_305_603, 0.070_710_678_119)),
                (2, 1, Complex::new(0.282_842_712_475, 0.155_563_491_861)),
                (5, 5, Complex::new(0.65, 0.25)),
                (8, 8, Complex::new(1.0, 0.42)),
            ],
        ),
    ];

    for (mode, summary, entries) in cases {
        let output = change_basis_representation(input.view(), mode, &transforms)?;
        assert_matrix_summary(output.view(), summary, &entries);
    }
    Ok(())
}

#[test]
fn change_basis_representation_rejects_invalid_shapes() -> Result<(), AngularError> {
    let transforms = basis_transform_matrices(1)?;
    let bad_input = Array2::<Complex>::zeros((2, 2));
    assert_eq!(
        change_basis_representation(
            bad_input.view(),
            BasisTransformMode::RelativisticToReal,
            &transforms,
        ),
        Err(AngularError::InvalidBasisTransformShape {
            name: "input",
            rows: 2,
            columns: 2,
            expected: 8,
        })
    );
    Ok(())
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected}"
    );
}

fn assert_real_matrix_close(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), &actual) in actual.indexed_iter() {
        assert_close(actual, expected[(row, column)]);
    }
}

#[derive(Debug, Clone, Copy)]
struct MatrixSummary {
    count: usize,
    total: Complex,
    trace: Complex,
    weighted: Complex,
    norm2: f64,
}

fn sample_basis_transform_input(order: usize) -> Array2<Complex> {
    Array2::from_shape_fn((order, order).f(), |(row, column)| {
        let row_feff = row as f64 + 1.0;
        let column_feff = column as f64 + 1.0;
        Complex::new(
            0.1 * row_feff + 0.03 * column_feff,
            -0.02 * row_feff + 0.07 * column_feff,
        )
    })
}

fn matrix_summary(matrix: ArrayView2<'_, Complex>) -> MatrixSummary {
    let mut count = 0usize;
    let mut total = Complex::new(0.0, 0.0);
    let mut trace = Complex::new(0.0, 0.0);
    let mut weighted = Complex::new(0.0, 0.0);
    let mut norm2 = 0.0;
    for ((row, column), &value) in matrix.indexed_iter() {
        total += value;
        weighted += value * Complex::new(row as f64 + 1.0, -0.25 * (column as f64 + 1.0));
        norm2 += value.norm_sqr();
        if value.norm() > 1.0e-12 {
            count += 1;
        }
    }
    for index in 0..matrix.nrows().min(matrix.ncols()) {
        trace += matrix[(index, index)];
    }
    MatrixSummary {
        count,
        total,
        trace,
        weighted,
        norm2,
    }
}

fn assert_matrix_summary(
    matrix: ArrayView2<'_, Complex>,
    expected: MatrixSummary,
    entries: &[(usize, usize, Complex)],
) {
    let actual = matrix_summary(matrix);
    assert_eq!(actual.count, expected.count);
    assert_complex_close(actual.total, expected.total);
    assert_complex_close(actual.trace, expected.trace);
    assert_complex_close(actual.weighted, expected.weighted);
    assert_close(actual.norm2, expected.norm2);
    for &(row, column, expected) in entries {
        assert_complex_close(matrix[(row - 1, column - 1)], expected);
    }
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_complex_iter_close(actual: impl ExactSizeIterator<Item = Complex>, expected: &[Complex]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, &expected) in actual.zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
}

fn assert_tensor_nonzeros(tensor: ArrayView2<'_, Complex>, expected: &[(isize, isize, Complex)]) {
    assert_eq!(tensor.shape(), &[3, 3]);
    for row in -1..=1 {
        for column in -1..=1 {
            let expected_value = expected
                .iter()
                .find_map(|&(expected_row, expected_column, value)| {
                    if expected_row == row && expected_column == column {
                        Some(value)
                    } else {
                        None
                    }
                })
                .unwrap_or(Complex::new(0.0, 0.0));
            assert_tensor_entry(tensor, row, column, expected_value);
        }
    }
}

fn assert_tensor_entry(
    tensor: ArrayView2<'_, Complex>,
    row: isize,
    column: isize,
    expected: Complex,
) {
    assert_complex_close(
        tensor[(
            polarization_test_index(row),
            polarization_test_index(column),
        )],
        expected,
    );
}

fn polarization_test_index(magnetic: isize) -> usize {
    match magnetic {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 0,
    }
}

fn sample_transition_b_matrix_input() -> TransitionBMatrixInput {
    TransitionBMatrixInput {
        lmax: 3,
        initial_kappa: -1,
        polarization: 1,
        polarization_tensor: [
            [
                Complex::new(0.20, -0.05),
                Complex::new(-0.10, 0.04),
                Complex::new(0.03, 0.02),
            ],
            [
                Complex::new(0.11, -0.07),
                Complex::new(0.50, 0.00),
                Complex::new(-0.08, 0.09),
            ],
            [
                Complex::new(0.06, 0.01),
                Complex::new(0.13, -0.02),
                Complex::new(0.17, 0.03),
            ],
        ],
        multipole: 2,
        trace_orbital: false,
        spin: 1,
        spin_channels: 1,
        spin_vector_angle: 0.3,
    }
}

fn assert_transition_values(result: &super::TransitionBMatrix, expected: &[Complex; 6]) {
    let indices = [
        (0, 0, 1, 0, 0, 1),
        (1, 0, 2, -1, 0, 2),
        (0, 1, 4, 0, 1, 4),
        (-1, 0, 5, 1, 1, 6),
        (0, 0, 7, 0, 0, 7),
        (0, 0, 8, 0, 0, 8),
    ];
    for (index, &expected) in indices.iter().zip(expected.iter()) {
        let actual = result.value(index.0, index.1, index.2, index.3, index.4, index.5);
        assert!(actual.is_some(), "missing transition matrix value");
        if let Some(actual) = actual {
            assert_complex_close(actual, expected);
        }
    }
}

fn matrix_sum(matrix: &ndarray::Array6<Complex>) -> Complex {
    matrix
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), |accumulator, value| {
            accumulator + value
        })
}

fn nonzero_count(matrix: &ndarray::Array6<Complex>) -> usize {
    matrix.iter().filter(|value| value.norm() > 1.0e-14).count()
}
