use super::*;
use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder, array};
use num_complex::{Complex32, Complex64};

#[test]
fn multiplies_real_matrices_through_faer() {
    let lhs = array![[1.0, 2.0], [3.0, 4.0]];
    let rhs = array![[5.0], [6.0]];
    let out = real_matmul(lhs.view(), rhs.view());
    assert_eq!(out, array![[17.0], [39.0]]);
}

#[test]
fn roundtrips_complex_matrix_layout() {
    let input = array![[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]];
    let matrix = complex_to_faer(input.view());
    let out = complex_from_faer(&matrix);
    assert_eq!(out, input);
}

#[test]
fn roundtrips_complex32_matrix_layout() {
    let input = array![[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]];
    let matrix = complex32_to_faer(input.view());
    let out = complex32_from_faer(&matrix);
    assert_eq!(out, input);
}

#[test]
fn real_lu_matches_feff_dgetrf_dgetrs_reference() -> Result<(), LinalgError> {
    let matrix = array![[0.0, 2.0, -1.0], [3.0, -1.0, 4.0], [1.0, 0.5, 2.0]];
    let right_hand_side = array![[1.0, -2.0], [0.0, 3.0], [2.0, -1.0]];

    let lu = real_lu_factor(matrix.view())?;
    assert_eq!(lu.pivots(), &[2, 2, 3]);
    assert_matrix_close(
        lu.factors(),
        array![
            [3.0, -1.0, 4.0],
            [0.0, 2.0, -1.0],
            [
                0.333_333_333_333_333_3,
                0.41666666666666663,
                1.0833333333333335
            ]
        ]
        .view(),
    );

    let solution = real_lu_solve(&lu, right_hand_side.view())?;
    assert_matrix_close(
        solution.view(),
        array![
            [-1.5384615384615383, 1.9230769230769231],
            [1.2307692307692308, -1.5384615384615383],
            [1.4615384615384615, -1.0769230769230769]
        ]
        .view(),
    );

    let vector_solution = real_lu_solve_vector(&lu, right_hand_side.column(0))?;
    assert_eq!(vector_solution.len(), 3);
    assert_close(vector_solution[0], -1.5384615384615383);
    Ok(())
}

#[test]
fn complex_lu_matches_feff_zgetrf_zgetrs_reference() -> Result<(), LinalgError> {
    let matrix = array![
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 0.5)
        ],
        [
            Complex64::new(3.0, 2.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(4.0, -1.0)
        ],
        [
            Complex64::new(1.0, -3.0),
            Complex64::new(0.5, 2.0),
            Complex64::new(2.0, 0.0)
        ]
    ];
    let right_hand_side = array![
        [Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
        [Complex64::new(0.0, -1.0), Complex64::new(3.0, 0.0)],
        [Complex64::new(2.0, 2.0), Complex64::new(-1.0, -0.5)]
    ];

    let lu = complex_lu_factor(matrix.view())?;
    assert_eq!(lu.pivots(), &[2, 2, 3]);
    assert_complex_matrix_close(
        lu.factors(),
        array![
            [
                Complex64::new(3.0, 2.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(4.0, -1.0)
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5)
            ],
            [
                Complex64::new(-0.23076923076923078, -0.846_153_846_153_846_1),
                Complex64::new(-0.12307692307692306, 0.515_384_615_384_615_3),
                Complex64::new(3.9038461538461537, 3.730_769_230_769_231)
            ]
        ]
        .view(),
    );

    let solution = complex_lu_solve(&lu, right_hand_side.view())?;
    assert_complex_matrix_close(
        solution.view(),
        array![
            [
                Complex64::new(-0.233_470_733_718_054_4, 0.43198680956306673),
                Complex64::new(-0.13470733718054406, -0.280_131_904_369_332_2)
            ],
            [
                Complex64::new(0.600_164_880_461_665_2, 0.281_615_828_524_319_9),
                Complex64::new(-0.798_351_195_383_347_1, 0.216_158_285_243_198_7)
            ],
            [
                Complex64::new(0.600_329_760_923_330_6, -0.236_768_342_951_360_3),
                Complex64::new(0.403_297_609_233_305_9, 0.432_316_570_486_397_3)
            ]
        ]
        .view(),
    );

    let vector_solution = complex_lu_solve_vector(&lu, right_hand_side.column(0))?;
    assert_complex_close(
        vector_solution[0],
        Complex64::new(-0.233_470_733_718_054_4, 0.43198680956306673),
    );
    Ok(())
}

#[test]
fn complex32_lu_matches_feff_cgetrf_cgetrs_reference() -> Result<(), LinalgError> {
    let matrix = array![
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(-1.0, 0.5)
        ],
        [
            Complex32::new(3.0, 2.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(4.0, -1.0)
        ],
        [
            Complex32::new(1.0, -3.0),
            Complex32::new(0.5, 2.0),
            Complex32::new(2.0, 0.0)
        ]
    ];
    let right_hand_side = array![
        [Complex32::new(1.0, 0.5), Complex32::new(-2.0, 1.0)],
        [Complex32::new(0.0, -1.0), Complex32::new(3.0, 0.0)],
        [Complex32::new(2.0, 2.0), Complex32::new(-1.0, -0.5)]
    ];

    let lu = complex32_lu_factor(matrix.view())?;
    assert_eq!(lu.pivots(), &[2, 2, 3]);
    assert_complex32_matrix_close(
        lu.factors(),
        array![
            [
                Complex32::new(3.0, 2.0),
                Complex32::new(-1.0, 0.0),
                Complex32::new(4.0, -1.0)
            ],
            [
                Complex32::new(0.0, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(-1.0, 0.5)
            ],
            [
                Complex32::new(-0.23076928, -0.8461538),
                Complex32::new(-0.12307697, 0.515_384_7),
                Complex32::new(3.9038463, 3.730_769)
            ]
        ]
        .view(),
    );

    let solution = complex32_lu_solve(&lu, right_hand_side.view())?;
    assert_complex32_matrix_close(
        solution.view(),
        array![
            [
                Complex32::new(-0.23347072, 0.43198672),
                Complex32::new(-0.13470735, -0.280_131_9)
            ],
            [
                Complex32::new(0.60016483, 0.28161582),
                Complex32::new(-0.798_351_2, 0.2161583)
            ],
            [
                Complex32::new(0.6003297, -0.236_768_3),
                Complex32::new(0.4032976, 0.43231657)
            ]
        ]
        .view(),
    );

    let vector_solution = complex32_lu_solve_vector(&lu, right_hand_side.column(0))?;
    assert_complex32_close(vector_solution[0], Complex32::new(-0.23347072, 0.43198672));
    Ok(())
}

#[test]
fn complex32_faer_lu_solve_matches_feff_compatible_lu() -> Result<(), LinalgError> {
    let matrix = array![
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(-1.0, 0.5)
        ],
        [
            Complex32::new(3.0, 2.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(4.0, -1.0)
        ],
        [
            Complex32::new(1.0, -3.0),
            Complex32::new(0.5, 2.0),
            Complex32::new(2.0, 0.0)
        ]
    ];
    let right_hand_side = array![
        [Complex32::new(1.0, 0.5), Complex32::new(-2.0, 1.0)],
        [Complex32::new(0.0, -1.0), Complex32::new(3.0, 0.0)],
        [Complex32::new(2.0, 2.0), Complex32::new(-1.0, -0.5)]
    ];

    let compat_lu = complex32_lu_factor(matrix.view())?;
    let compat_solution = complex32_lu_solve(&compat_lu, right_hand_side.view())?;
    let faer_lu = complex32_faer_lu_factor(matrix.view())?;
    assert_eq!(faer_lu.order(), 3);
    let faer_solution = complex32_faer_lu_solve(&faer_lu, right_hand_side.view())?;

    assert_complex32_matrix_close(faer_solution.view(), compat_solution.view());
    Ok(())
}

#[test]
fn complex32_faer_lu_solve_in_place_matches_owned_result() -> Result<(), LinalgError> {
    let matrix = array![
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(-1.0, 0.5)
        ],
        [
            Complex32::new(3.0, 2.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(4.0, -1.0)
        ],
        [
            Complex32::new(1.0, -3.0),
            Complex32::new(0.5, 2.0),
            Complex32::new(2.0, 0.0)
        ]
    ];
    let right_hand_side = array![
        [Complex32::new(1.0, 0.5), Complex32::new(-2.0, 1.0)],
        [Complex32::new(0.0, -1.0), Complex32::new(3.0, 0.0)],
        [Complex32::new(2.0, 2.0), Complex32::new(-1.0, -0.5)]
    ];

    let lu = complex32_faer_lu_factor(matrix.view())?;
    let owned = complex32_faer_lu_solve(&lu, right_hand_side.view())?;

    // Row-major (default `array!` layout) input takes the copy-based
    // fallback path inside `complex32_faer_lu_solve_in_place`.
    let mut row_major = right_hand_side.clone();
    complex32_faer_lu_solve_in_place(&lu, row_major.view_mut())?;
    assert_complex32_matrix_close(row_major.view(), owned.view());

    // Column-major storage (as FMS builds with `ndarray`'s `.f()` shape)
    // exercises the zero-copy `faer` `solve_in_place` path directly.
    let mut column_major = Array2::from_shape_fn(right_hand_side.raw_dim().f(), |(row, col)| {
        right_hand_side[(row, col)]
    });
    complex32_faer_lu_solve_in_place(&lu, column_major.view_mut())?;
    assert_complex32_matrix_close(column_major.view(), owned.view());

    Ok(())
}

#[test]
fn set_parallelism_accepts_sequential_and_threaded_modes() {
    set_parallelism(Some(1));
    set_parallelism(Some(4));
    set_parallelism(Some(0));
    set_parallelism(None);
}

#[test]
fn lu_rejects_singular_and_mismatched_inputs() {
    let singular = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]];
    assert_eq!(
        real_lu_factor(singular.view()),
        Err(LinalgError::SingularMatrix { pivot: 2 })
    );

    let non_square = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(
        real_lu_factor(non_square.view()),
        Err(LinalgError::NonSquare { rows: 2, cols: 3 })
    );

    let lu = real_lu_factor(array![[1.0, 0.0], [0.0, 1.0]].view());
    let rhs = array![[1.0], [2.0], [3.0]];
    assert!(matches!(
        lu.and_then(|lu| real_lu_solve(&lu, rhs.view())),
        Err(LinalgError::LengthMismatch { .. })
    ));

    let complex32_non_square = array![
        [Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
        [Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)],
        [Complex32::new(5.0, 0.0), Complex32::new(6.0, 0.0)]
    ];
    assert_eq!(
        complex32_lu_factor(complex32_non_square.view()),
        Err(LinalgError::NonSquare { rows: 3, cols: 2 })
    );
    assert!(matches!(
        complex32_faer_lu_factor(complex32_non_square.view()),
        Err(LinalgError::NonSquare { rows: 3, cols: 2 })
    ));
}

#[test]
fn determinant_matches_feff_reference_with_column_swap() -> Result<(), LinalgError> {
    let matrix = array![[0.0, 2.0, -1.0], [1.0, -3.0, 2.0], [4.0, 1.0, 0.5]];

    assert_close(feff_determinant(matrix.view())?, 2.0);
    Ok(())
}

#[test]
fn determinant_returns_zero_for_singular_matrix() -> Result<(), LinalgError> {
    let matrix = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, 0.0, 1.0]];

    assert_eq!(feff_determinant(matrix.view())?, 0.0);
    Ok(())
}

#[test]
fn determinant_rejects_non_square_matrix() {
    let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    assert_eq!(
        feff_determinant(matrix.view()),
        Err(LinalgError::NonSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn inverse_matches_feff_reference() -> Result<(), LinalgError> {
    let matrix = array![[2.0, -1.0, 0.5], [1.0, 3.0, -2.0], [0.25, -0.5, 1.5]];
    let inverse = feff_inverse(matrix.view())?;
    let expected = array![
        [
            0.41791044776119407,
            0.14925373134328357,
            0.05970149253731341
        ],
        [
            -0.23880597014925373,
            0.34328358208955223,
            0.5373134328358209
        ],
        [-0.14925373134328357, 0.08955223880597014, 0.835820895522388]
    ];

    assert_matrix_close(inverse.view(), expected.view());
    assert_matrix_close(
        real_matmul(matrix.view(), inverse.view()).view(),
        Array2::eye(3).view(),
    );
    Ok(())
}

#[test]
fn inverse_rejects_singular_matrix() {
    let matrix = array![[1.0, 2.0], [2.0, 4.0]];

    assert_eq!(
        feff_inverse(matrix.view()),
        Err(LinalgError::SingularMatrix { pivot: 1 })
    );
}

#[test]
fn inverse_rejects_non_square_matrix() {
    let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    assert_eq!(
        feff_inverse(matrix.view()),
        Err(LinalgError::NonSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn complex_polyfit_matches_feff_reference_order2() -> Result<(), LinalgError> {
    let x = [-1.0, 0.0, 1.5, 2.0, 3.5];
    let y = array![
        Complex64::new(1.0, -1.0),
        Complex64::new(0.5, 0.25),
        Complex64::new(3.0, -0.5),
        Complex64::new(4.2, 1.1),
        Complex64::new(10.0, 2.0)
    ];

    let coefficients = complex_polyfit(&x, y.view(), 2)?;
    assert_complex_vec_close(
        coefficients.view(),
        &[
            Complex64::new(0.7367647058823517, -0.4195433436532505),
            Complex64::new(0.47303921568627294, 0.3809984520123841),
            Complex64::new(0.6245098039215693, 0.08521671826625377),
        ],
    );

    let evaluated = complex_polyval(coefficients.view(), &[-0.5, 0.25, 2.25, 4.0]);
    assert_complex_vec_close(
        evaluated.view(),
        &[
            Complex64::new(0.6563725490196075, -0.5887383900928791),
            Complex64::new(0.8940563725490179, -0.31896768575851364),
            Complex64::new(4.96268382352941, 0.8691128095975234),
            Complex64::new(12.621078431372553, 2.467917956656346),
        ],
    );
    Ok(())
}

#[test]
fn complex_polyfit_matches_feff_reference_order3() -> Result<(), LinalgError> {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = array![
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(1.0, 0.5),
        Complex64::new(2.5, -0.25),
        Complex64::new(5.0, 1.0)
    ];

    let coefficients = complex_polyfit(&x, y.view(), 3)?;
    assert_complex_vec_close(
        coefficients.view(),
        &[
            Complex64::new(1.0, -0.4428571428571429),
            Complex64::new(1.1666666666666663, 0.5833333333333331),
            Complex64::new(0.25, 0.44642857142857145),
            Complex64::new(0.08333333333333344, -0.208_333_333_333_333_3),
        ],
    );

    let evaluated = complex_polyval(coefficients.view(), &[-1.5, 0.5, 2.5]);
    assert_complex_vec_close(
        evaluated.view(),
        &[
            Complex64::new(-0.4687499999999998, 0.389732142857143),
            Complex64::new(1.6562499999999998, -0.06562500000000011),
            Complex64::new(6.781250000000001, 0.550446428571429),
        ],
    );
    Ok(())
}

#[test]
fn complex_least_squares_rejects_invalid_inputs() {
    let design = array![[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]];
    let y = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    assert!(matches!(
        complex_least_squares_normal_eq(design.view(), y.view()),
        Err(LinalgError::LengthMismatch { .. })
    ));

    let singular_y = array![Complex64::new(1.0, 0.0)];
    assert!(matches!(
        complex_least_squares_normal_eq(design.view(), singular_y.view()),
        Err(LinalgError::SingularMatrix { .. })
    ));
}

#[test]
fn real32_symmetric_2x2_helpers_match_feff_slaev2_reference() -> Result<(), LinalgError> {
    let cases = [
        (
            4.0,
            1.25,
            1.0,
            [4.452_562_3, 0.547_437_6],
            [-0.940_271_56, -0.340_425_25],
        ),
        (
            -5.0,
            2.0,
            -1.0,
            [-5.828_427_3, -0.171_572_86],
            [-0.923_879_5, 0.382_683_4],
        ),
        (2.0, 0.0, -3.0, [-3.0, 2.0], [-0.0, 1.0]),
        (0.0, 0.0, 0.0, [0.0, -0.0], [-0.0, 1.0]),
    ];

    for (diagonal_a, off_diagonal, diagonal_c, expected_values, expected_vector) in cases {
        let values = real32_symmetric_2x2_eigenvalues(diagonal_a, off_diagonal, diagonal_c)?;
        assert_f32_slice_close(&values, &expected_values);
        let eigensystem = real32_symmetric_2x2_eigen(diagonal_a, off_diagonal, diagonal_c)?;
        assert_f32_close(eigensystem.larger_abs_eigenvalue(), expected_values[0]);
        assert_f32_close(eigensystem.smaller_abs_eigenvalue(), expected_values[1]);
        assert_f32_slice_close(&eigensystem.larger_abs_eigenvector(), &expected_vector);
    }

    Ok(())
}

#[test]
fn real32_symmetric_2x2_helpers_reject_non_finite_inputs() {
    assert_eq!(
        real32_symmetric_2x2_eigenvalues(f32::NAN, 0.0, 1.0),
        Err(LinalgError::NonFiniteScalar { name: "a" })
    );
    assert_eq!(
        real32_symmetric_2x2_eigen(1.0, f32::INFINITY, 0.0),
        Err(LinalgError::NonFiniteScalar { name: "b" })
    );
}

#[test]
fn real32_symmetric_eigen_matches_feff_ssyev_reference() -> Result<(), LinalgError> {
    let matrix = array![
        [4.0_f32, 99.0, 99.0, 99.0],
        [-1.0, 3.0, 99.0, 99.0],
        [0.5, 1.25, 2.5, 99.0],
        [0.75, -0.5, 1.5, 1.0],
    ];

    let eigensystem = real32_symmetric_eigen(matrix.view(), SymmetricTriangle::Lower)?;
    assert_f32_vec_close(
        eigensystem.eigenvalues(),
        &[-0.301_708_43, 1.816_907, 4.142_762_7, 4.842_039],
    );
    assert_f32_matrix_abs_close(
        eigensystem.eigenvectors(),
        array![
            [0.008_082_926, -0.523_904_4, -0.127_553_43, 0.842_133_64],
            [0.328_703_9, -0.589_783_8, -0.578_473_57, -0.457_686_87],
            [-0.556_576_4, 0.343_050_42, -0.749_305_3, 0.105_265_945],
            [0.762_962_2, 0.509_897_6, -0.296_040_98, 0.265_052_56],
        ]
        .view(),
    );

    let values_only = real32_symmetric_eigenvalues(matrix.view(), SymmetricTriangle::Lower)?;
    assert_f32_vec_close(values_only.view(), &eigensystem.eigenvalues().to_vec());
    Ok(())
}

#[test]
fn real32_symmetric_eigen_uses_selected_triangle_only() -> Result<(), LinalgError> {
    let lower = array![[2.0_f32, f32::NAN], [0.5, 3.0]];
    let upper = array![[2.0_f32, 0.5], [f32::NAN, 3.0]];

    let lower_values = real32_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Lower)?;
    let upper_values = real32_symmetric_eigenvalues(upper.view(), SymmetricTriangle::Upper)?;
    assert_f32_vec_close(lower_values.view(), &upper_values.to_vec());

    assert_eq!(
        real32_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Upper),
        Err(LinalgError::NonFiniteMatrixEntry { row: 0, col: 1 })
    );
    Ok(())
}

#[test]
fn real32_symmetric_eigen_rejects_invalid_inputs() {
    let non_square = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(
        real32_symmetric_eigen(non_square.view(), SymmetricTriangle::Lower),
        Err(LinalgError::NonSquare { rows: 2, cols: 3 })
    );

    let non_finite = array![[1.0_f32, 0.0], [f32::INFINITY, 2.0]];
    assert_eq!(
        real32_symmetric_eigenvalues(non_finite.view(), SymmetricTriangle::Lower),
        Err(LinalgError::NonFiniteMatrixEntry { row: 1, col: 0 })
    );
}

#[test]
fn complex32_general_eigenvalues_match_upper_triangular_reference() -> Result<(), LinalgError> {
    let matrix = array![
        [
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.0)
        ],
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(-2.0, 0.25),
            Complex32::new(3.0, 4.0)
        ],
        [
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.5, -1.5)
        ],
    ];

    let mut eigenvalues = complex32_general_eigenvalues(matrix.view())?.to_vec();
    eigenvalues.sort_by(|left, right| {
        left.re
            .total_cmp(&right.re)
            .then(left.im.total_cmp(&right.im))
    });
    assert_eq!(eigenvalues.len(), matrix.nrows());
    assert_complex32_slice_close(
        &eigenvalues,
        &[
            Complex32::new(-2.0, 0.25),
            Complex32::new(0.5, -1.5),
            Complex32::new(1.0, 0.5),
        ],
    );
    Ok(())
}

#[test]
fn complex32_general_eigenvalues_reject_invalid_inputs() {
    let non_square = array![[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)]];
    assert_eq!(
        complex32_general_eigenvalues(non_square.view()),
        Err(LinalgError::NonSquare { rows: 1, cols: 2 })
    );

    let non_finite = array![
        [Complex32::new(1.0, 0.0), Complex32::new(0.0, f32::NAN)],
        [Complex32::new(0.0, 0.0), Complex32::new(2.0, 0.0)],
    ];
    assert_eq!(
        complex32_general_eigenvalues(non_finite.view()),
        Err(LinalgError::NonFiniteMatrixEntry { row: 0, col: 1 })
    );
}

#[test]
fn real64_symmetric_eigen_matches_known_reference() -> Result<(), LinalgError> {
    let matrix = array![[2.0_f64, f64::NAN], [1.0, 2.0]];

    let eigensystem = real64_symmetric_eigen(matrix.view(), SymmetricTriangle::Lower)?;
    assert_vec_close(eigensystem.eigenvalues(), &[1.0, 3.0]);
    let one_over_root_two = std::f64::consts::FRAC_1_SQRT_2;
    assert_matrix_abs_close(
        eigensystem.eigenvectors(),
        array![
            [one_over_root_two, one_over_root_two],
            [one_over_root_two, one_over_root_two],
        ]
        .view(),
    );

    let values_only = real64_symmetric_eigenvalues(matrix.view(), SymmetricTriangle::Lower)?;
    assert_vec_close(values_only.view(), &eigensystem.eigenvalues().to_vec());
    Ok(())
}

#[test]
fn real64_symmetric_eigen_uses_selected_triangle_only() -> Result<(), LinalgError> {
    let lower = array![[2.0_f64, f64::NAN], [0.5, 3.0]];
    let upper = array![[2.0_f64, 0.5], [f64::NAN, 3.0]];

    let lower_values = real64_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Lower)?;
    let upper_values = real64_symmetric_eigenvalues(upper.view(), SymmetricTriangle::Upper)?;
    assert_vec_close(lower_values.view(), &upper_values.to_vec());

    assert_eq!(
        real64_symmetric_eigenvalues(lower.view(), SymmetricTriangle::Upper),
        Err(LinalgError::NonFiniteMatrixEntry { row: 0, col: 1 })
    );
    Ok(())
}

#[test]
fn real64_symmetric_eigen_rejects_invalid_inputs() {
    let non_square = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(
        real64_symmetric_eigen(non_square.view(), SymmetricTriangle::Lower),
        Err(LinalgError::NonSquare { rows: 2, cols: 3 })
    );

    let non_finite = array![[1.0_f64, 0.0], [f64::INFINITY, 2.0]];
    assert_eq!(
        real64_symmetric_eigenvalues(non_finite.view(), SymmetricTriangle::Lower),
        Err(LinalgError::NonFiniteMatrixEntry { row: 1, col: 0 })
    );
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual} expected={expected}"
    );
}

fn assert_matrix_close(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, col), &value) in actual.indexed_iter() {
        assert_close(value, expected[(row, col)]);
    }
}

fn assert_vec_close(actual: ArrayView1<'_, f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "index={index} actual={actual} expected={expected}"
        );
    }
}

fn assert_matrix_abs_close(actual: ArrayView2<'_, f64>, expected: ArrayView2<'_, f64>) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());
    for ((row, col), &actual) in actual.indexed_iter() {
        let expected = expected[(row, col)];
        assert!(
            (actual.abs() - expected.abs()).abs() < 1.0e-12,
            "({row},{col}) actual={actual} expected={expected}"
        );
    }
}

fn assert_f32_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 2.0e-5,
        "actual={actual} expected={expected}"
    );
}

fn assert_f32_slice_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2.0e-5,
            "index={index} actual={actual} expected={expected}"
        );
    }
}

fn assert_f32_vec_close(actual: ArrayView1<'_, f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2.0e-5,
            "index={index} actual={actual} expected={expected}"
        );
    }
}

fn assert_f32_matrix_abs_close(actual: ArrayView2<'_, f32>, expected: ArrayView2<'_, f32>) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());
    for ((row, col), &actual) in actual.indexed_iter() {
        let expected = expected[(row, col)];
        assert!(
            (actual.abs() - expected.abs()).abs() < 2.0e-5,
            "({row},{col}) actual={actual} expected={expected}"
        );
    }
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert!(
        (actual - expected).norm() < 5.0e-6,
        "actual={actual:?} expected={expected:?}"
    );
}

fn assert_complex32_slice_close(actual: &[Complex32], expected: &[Complex32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).norm() < 5.0e-6,
            "index={index} actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_complex_matrix_close(
    actual: ArrayView2<'_, Complex64>,
    expected: ArrayView2<'_, Complex64>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, col), &value) in actual.indexed_iter() {
        assert_complex_close(value, expected[(row, col)]);
    }
}

fn assert_complex32_matrix_close(
    actual: ArrayView2<'_, Complex32>,
    expected: ArrayView2<'_, Complex32>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, col), &value) in actual.indexed_iter() {
        assert_complex32_close(value, expected[(row, col)]);
    }
}

fn assert_complex_vec_close(actual: ArrayView1<'_, Complex64>, expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
}
