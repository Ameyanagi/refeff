use super::*;

#[test]
fn mkgtr_green_trace_matches_feff_getgtr_loop() -> Result<(), Box<dyn Error>> {
    let mut first_matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    first_matrix.matrix[(0, 0, 0, 0, 0, 0)] = Complex::new(2.0, 0.5);
    let mut second_matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    second_matrix.matrix[(0, 0, 0, 0, 0, 0)] = Complex::new(-1.0, 0.25);
    let matrices = [first_matrix, second_matrix];
    let mut green = Array3::zeros((2, 1, 1).f());
    green[(0, 0, 0)] = Complex32::new(1.0, 2.0);
    green[(1, 0, 0)] = Complex32::new(-0.5, 0.75);
    let mut rkk = Array3::zeros((2, 8, 1).f());
    rkk[(0, 0, 0)] = Complex::new(3.0, -1.0);
    rkk[(1, 0, 0)] = Complex::new(0.5, 2.0);

    let result = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 1,
        green_functions: green.view(),
        transition_matrices: &matrices,
        transition_moments: rkk.view(),
    })?;

    assert_eq!(result.traces.shape(), &[2, 2]);
    assert_complex_close(
        result.traces[(0, 0)],
        widen_complex32_for_test(green[(0, 0, 0)])
            * matrices[0].matrix[(0, 0, 0, 0, 0, 0)]
            * rkk[(0, 0, 0)]
            * rkk[(0, 0, 0)],
    );
    assert_complex_close(
        result.traces[(1, 1)],
        widen_complex32_for_test(green[(1, 0, 0)])
            * matrices[1].matrix[(0, 0, 0, 0, 0, 0)]
            * rkk[(1, 0, 0)]
            * rkk[(1, 0, 0)],
    );
    Ok(())
}

#[test]
fn mkgtr_green_trace_uses_feff_spin_channel_indexing() -> Result<(), Box<dyn Error>> {
    let mut matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    matrix.matrix[(0, 1, 0, 0, 0, 0)] = Complex::new(1.5, -0.25);
    let matrices = [matrix];
    let mut green = Array3::zeros((1, 2, 2).f());
    green[(0, 0, 1)] = Complex32::new(0.5, -0.25);
    let mut rkk = Array3::zeros((1, 8, 2).f());
    rkk[(0, 0, 0)] = Complex::new(2.0, 0.0);
    rkk[(0, 0, 1)] = Complex::new(3.0, 0.5);

    let result = mkgtr_green_trace(MkgtrGreenTraceInput {
        active_spin_channels: 2,
        green_functions: green.view(),
        transition_matrices: &matrices,
        transition_moments: rkk.view(),
    })?;

    assert_complex_close(
        result.traces[(0, 0)],
        widen_complex32_for_test(green[(0, 0, 1)])
            * matrices[0].matrix[(0, 1, 0, 0, 0, 0)]
            * rkk[(0, 0, 0)]
            * rkk[(0, 0, 1)],
    );
    Ok(())
}

#[test]
fn mkgtr_green_trace_rejects_invalid_inputs() {
    let matrix = sample_mkgtr_transition_matrix([0, -1, -1, -1, -1, -1, -1, -1]);
    let matrices = [matrix];
    let green = Array3::from_elem((1, 1, 1).f(), Complex32::new(f32::NAN, 0.0));
    let rkk = Array3::from_elem((1, 8, 1).f(), Complex::new(1.0, 0.0));

    assert_eq!(
        mkgtr_green_trace(MkgtrGreenTraceInput {
            active_spin_channels: 1,
            green_functions: green.view(),
            transition_matrices: &matrices,
            transition_moments: rkk.view(),
        }),
        Err(FmsError::NonFiniteComplexValue {
            table: "gg",
            index: 0,
        })
    );

    let short_rkk = Array3::zeros((1, 8, 0).f());
    assert_eq!(
        mkgtr_green_trace(MkgtrGreenTraceInput {
            active_spin_channels: 1,
            green_functions: Array3::zeros((1, 1, 1).f()).view(),
            transition_matrices: &matrices,
            transition_moments: short_rkk.view(),
        }),
        Err(FmsError::SpinChannelCountMismatch {
            table: "rkk",
            expected: 1,
            actual: 0,
        })
    );
}

#[test]
fn mkgtr_jas_green_trace_covers_rotated_diagonal_and_mdff_q_pairs() -> Result<(), Box<dyn Error>> {
    let indices = xsph_nrixs_transition_indices(XsphNrixsTransitionIndicesInput {
        initial_kappa: -1,
        multipole: 1,
        max_angular_momentum: 2,
    })?;
    let transitions = indices
        .transitions
        .iter()
        .map(|transition| MkgtrJasTransition {
            final_state_kappa: transition.final_state_kappa,
            decomposition_channel: transition.decomposition_channel as usize,
            multipole: transition.total_angular_momentum_channel as usize,
            orbital_angular_momentum: transition.orbital_angular_momentum as usize,
        })
        .collect::<Vec<_>>();
    let max_angular_momentum = transitions
        .iter()
        .map(|transition| transition.decomposition_channel)
        .max()
        .unwrap_or(0);
    let channel_count = (max_angular_momentum + 1).pow(2);
    let green = Array3::from_shape_fn((1, channel_count, channel_count).f(), |(_, row, column)| {
        let scale = 0.1 + 0.03 * row as f32 + 0.02 * column as f32;
        Complex32::new(scale, -0.4 * scale)
    });
    let rkk = Array4::from_shape_fn((1, 2, transitions.len(), 1).f(), |(_, q, transition, _)| {
        Complex::new(
            0.4 + 0.1 * q as Real + 0.02 * transition as Real,
            -0.03 * (q + transition) as Real,
        )
    });
    let phases = Array1::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.8, -0.6)]);
    let beta = Array1::from_vec(vec![0.0, 0.7]);
    let weights = Array1::from_vec(vec![Complex::new(0.9, 0.1), Complex::new(0.7, -0.2)]);
    let cosines = array![[1.0, 0.25], [0.25, 1.0]];
    let run = |q_pair_mode| {
        mkgtr_jas_green_trace(MkgtrJasGreenTraceInput {
            active_spin_channels: 1,
            max_angular_momentum,
            green_functions: green.view(),
            transition_moments: rkk.view(),
            initial_kappa: -1,
            initial_j2: indices.initial_j2,
            final_j2_max: indices.final_j2_max,
            final_lj_max: indices.final_lj_max,
            transitions: &transitions,
            q_phases: phases.view(),
            q_beta_angles: beta.view(),
            q_weights: weights.view(),
            q_pair_cosines: cosines.view(),
            ellipticity: 0.0,
            q_pair_mode,
            max_decomposition_channel: Some(max_angular_momentum),
        })
    };

    let diagonal = run(MkgtrJasQPairMode::Diagonal)?;
    let all_pairs = run(MkgtrJasQPairMode::AllPairs)?;
    let first_to_second = run(MkgtrJasQPairMode::FirstToSecond)?;
    assert!(diagonal.trace[0].norm() > 1.0e-12);
    assert!((all_pairs.trace[0] - diagonal.trace[0]).norm() > 1.0e-12);
    assert!((first_to_second.trace[0] - diagonal.trace[0]).norm() > 1.0e-12);
    for result in [&diagonal, &all_pairs, &first_to_second] {
        let decomposed = result
            .decomposed_traces
            .as_ref()
            .expect("decomposition requested");
        let sum = decomposed.index_axis(Axis(0), 0).iter().copied().sum();
        assert_complex_close(result.trace[0], sum);
    }
    Ok(())
}

#[test]
fn xclmz_matches_feff_reference_lx3() -> Result<(), FmsError> {
    let table = rehr_albers_polynomials(3, 4, 4, Complex32::new(1.25, 0.4))?;

    assert_eq!(table.shape(), &[5, 9]);
    assert_eq!(table.strides(), &[1, 5]);
    assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(table[(1, 0)], Complex32::new(1.2322206, 0.725_689_4));
    assert_complex32_close(table[(3, 0)], Complex32::new(-10.012509, 5.438_266));
    assert_complex32_close(table[(2, 1)], Complex32::new(-2.1395304, 4.1993084));
    assert_complex32_close(table[(3, 2)], Complex32::new(-23.036537, -6.8588142));
    assert_complex32_close(table[(4, 3)], Complex32::new(8.928_719, -161.62775));
    assert_complex32_close(
        matrix_sum(table.view()),
        Complex32::new(-58.983994, -154.61885),
    );
    assert_eq!(nonzero_count(table.view()), 11);
    Ok(())
}

#[test]
fn xclmz_matches_feff_reference_with_limited_m() -> Result<(), FmsError> {
    let table = rehr_albers_polynomials(4, 3, 2, Complex32::new(-0.8, 1.1))?;

    assert_eq!(table.shape(), &[6, 11]);
    assert_eq!(table.strides(), &[1, 6]);
    assert_complex32_close(table[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(table[(1, 0)], Complex32::new(1.5945946, -0.432_432_4));
    assert_complex32_close(table[(2, 0)], Complex32::new(3.2834187, -2.840029));
    assert_complex32_close(table[(1, 1)], Complex32::new(0.5945946, -0.432_432_4));
    assert_complex32_close(table[(2, 1)], Complex32::new(2.7830534, -4.382761));
    assert_complex32_close(
        matrix_sum(table.view()),
        Complex32::new(9.255661, -8.087655),
    );
    assert_eq!(nonzero_count(table.view()), 5);
    Ok(())
}

#[test]
fn xclmz_rejects_invalid_inputs() {
    assert_eq!(
        rehr_albers_polynomials(3, 0, 1, Complex32::new(1.0, 0.0)),
        Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 0,
            lx: 3,
        })
    );
    assert_eq!(
        rehr_albers_polynomials(3, 5, 1, Complex32::new(1.0, 0.0)),
        Err(FmsError::InvalidAngularLimit {
            name: "lmaxp1",
            value: 5,
            lx: 3,
        })
    );
    assert_eq!(
        rehr_albers_polynomials(3, 1, 1, Complex32::new(0.0, 0.0)),
        Err(FmsError::ZeroRho)
    );
    assert_eq!(
        rehr_albers_polynomials(3, 1, 1, Complex32::new(f32::NAN, 0.0)),
        Err(FmsError::NonFiniteRho)
    );
}

#[test]
fn rotxan_matches_feff_reference_forward_and_backward() -> Result<(), FmsError> {
    let forward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Forward)?;
    let backward = fms_rotation_matrix(3, 3, 0.7, 1.1, FmsRotationDirection::Backward)?;

    assert_eq!(forward.shape(), &[7, 7, 4]);
    assert_eq!(forward.strides(), &[1, 7, 49]);
    assert_complex32_close(
        rotation_sum(forward.view()),
        Complex32::new(1.159_583_6, 0.288_981_8),
    );
    assert_complex32_close(
        rotation_sum(backward.view()),
        Complex32::new(1.159_583_1, 0.288_981_74),
    );
    assert_eq!(rotation_nonzero_count(forward.view()), 84);
    assert_eq!(rotation_nonzero_count(backward.view()), 84);

    assert_complex32_close(rotation_value(&forward, 0, 0, 0), Complex32::new(1.0, 0.0));
    assert_complex32_close(
        rotation_value(&forward, 1, -1, 1),
        Complex32::new(-0.053_333_33, -0.104_787_19),
    );
    assert_complex32_close(
        rotation_value(&forward, -1, 1, 1),
        Complex32::new(-0.053_333_33, 0.104_787_19),
    );
    assert_complex32_close(
        rotation_value(&forward, 2, -1, 2),
        Complex32::new(-0.044_576_85, 0.061_240_695),
    );
    assert_complex32_close(
        rotation_value(&forward, -2, 1, 3),
        Complex32::new(0.116_102_73, 0.159_504_58),
    );
    assert_complex32_close(
        rotation_value(&forward, 3, 3, 3),
        Complex32::new(0.678_509_35, 0.108_389_09),
    );

    assert_complex32_close(
        rotation_value(&backward, 2, -1, 2),
        Complex32::new(-0.034_358_274, -0.067_505_76),
    );
    assert_complex32_close(
        rotation_value(&backward, -2, 1, 3),
        Complex32::new(0.089_487_91, -0.175_822_26),
    );
    assert_complex32_close(
        rotation_value(&backward, 3, 3, 3),
        Complex32::new(0.678_509_35, -0.108_389_09),
    );
    Ok(())
}

#[test]
fn rotxan_rejects_invalid_inputs() {
    assert_eq!(
        fms_rotation_matrix(25, 1, 0.0, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::InvalidAngularLimit {
            name: "lmax",
            value: 25,
            lx: 24,
        })
    );
    assert_eq!(
        fms_rotation_matrix(3, 4, 0.0, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::InvalidAngularLimit {
            name: "mmax",
            value: 4,
            lx: 3,
        })
    );
    assert_eq!(
        fms_rotation_matrix(3, 3, f32::NAN, 0.0, FmsRotationDirection::Forward),
        Err(FmsError::NonFiniteRotationAngle { name: "beta" })
    );
}
