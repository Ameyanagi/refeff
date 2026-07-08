use super::*;

#[test]
fn fms_iterative_system_matrix_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let system = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: 0.0,
    })?;

    assert_eq!(system.shape(), &[8, 8]);
    assert_eq!(system.strides(), &[1, 8]);
    assert_complex32_close(
        matrix_sum(system.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(system[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(system[(1, 3)], Complex32::new(0.001_4, -0.003_199_999_7));
    assert_complex32_close(
        system[(4, 5)],
        Complex32::new(0.001_230_000_3, -0.011_239_999),
    );
    assert_complex32_close(system[(6, 7)], Complex32::new(0.001_789_999_7, -0.020_9));

    let cutoff_system = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: 0.09,
    })?;

    assert_complex32_close(
        matrix_sum(cutoff_system.view()),
        Complex32::new(7.922_833_4, -0.471_125_07),
    );
    assert_complex32_close(cutoff_system[(1, 3)], Complex32::new(0.0, 0.0));
    assert_complex32_close(
        cutoff_system[(4, 5)],
        Complex32::new(0.001_230_000_3, -0.011_239_999),
    );
    Ok(())
}

#[test]
fn fms_iterative_system_matrix_rejects_invalid_tolerance() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_iterative_system_matrix(FmsIterativeSystemInput {
        states: &state_set.states,
        spin_channels: 2,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        zero_tolerance: -1.0,
    });

    assert!(matches!(
        result,
        Err(FmsError::InvalidTolerance {
            name: "toler2",
            value: -1.0,
        })
    ));
    Ok(())
}

#[test]
fn fms_bicgstab_scattering_matches_feff_ggbi_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_bicgstab_scattering(FmsBiCgStabInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 2);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.949_217_6, 4.806_942),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_855_818, -0.003_201_462_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.066_029_795, 0.044_123_195),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_492_656, 0.140_840_8),
    );
    Ok(())
}

#[test]
fn fms_bicgstab_scattering_respects_lcalc_mask() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_bicgstab_scattering(FmsBiCgStabInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, false],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_855_818, -0.003_201_462_3),
    );
    assert_complex32_close(result.scattering[(2, 2, 0)], Complex32::new(0.0, 0.0));
    assert_complex32_close(result.scattering[(7, 7, 0)], Complex32::new(0.0, 0.0));
    Ok(())
}

#[test]
fn fms_recursion_scattering_matches_feff_ggrm_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_recursion_scattering(FmsRecursionInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 3);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_324, 4.799_402),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_021, -0.003_244_287_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_52, 0.044_093_154),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_72, 0.140_520_17),
    );
    Ok(())
}

#[test]
fn fms_graves_morris_scattering_matches_feff_gggm_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_graves_morris_scattering(FmsGravesMorrisInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 4);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(0.090_419_99, 0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_321_6, 4.799_405),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_049_4, -0.003_244_209),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_47, 0.044_093_188),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_895, 0.140_520_08),
    );
    Ok(())
}

#[test]
fn fms_tfqmr_scattering_matches_feff_ggtf_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_tfqmr_scattering(FmsTfqmrInput {
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
        calculated_l: &[true, true],
        convergence_tolerance: 1.0e-5,
        zero_tolerance: 0.0,
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.multiple_scattering_order, 4);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(7.909_579_3, -0.516_9),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_7, 4.799_402_7),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_021_4, -0.003_244_287_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_43, 0.044_093_173),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_91, 0.140_520_1),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_matches_feff_gglu_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: false,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_eq!(result.full_scattering, None);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(8.107_28, -0.542_959_87),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.007_797_020_5, -0.003_244_286_6),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.065_967_42, 0.044_093_15),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_285_9, 0.140_520_07),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_returns_feff_gg_full_when_requested() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0, 1], &[1, 0], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: true,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1, 0],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 1,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    let Some(full_scattering) = result.full_scattering else {
        return Err("missing full scattering matrix".into());
    };
    assert_eq!(full_scattering.shape(), &[10, 10]);
    assert_eq!(result.scattering.shape(), &[8, 8, 2]);
    assert_complex32_close(
        matrix_sum(full_scattering.view()),
        Complex32::new(-6.616_672_5, 8.779_471),
    );
    assert_complex32_close(
        full_scattering[(0, 9)],
        Complex32::new(-0.189_542, 0.041_967_187),
    );
    assert_complex32_close(
        full_scattering[(9, 0)],
        Complex32::new(0.063_354_82, 0.163_031_2),
    );

    for potential in 0..=1 {
        let lmax = [1, 0][potential];
        let ipart = 2 * (lmax + 1) * (lmax + 1);
        let offset = match state_set.representative_offsets[potential] {
            Some(offset) => offset,
            None => return Err("missing representative offset".into()),
        };
        for column in 0..ipart {
            for row in 0..ipart {
                assert_complex32_close(
                    result.scattering[(row, column, potential)],
                    full_scattering[(offset + row, offset + column)],
                );
            }
        }
    }
    Ok(())
}

#[test]
fn fms_full_potential_lu_scattering_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, _) = reference_gglu_inputs(state_set.states.len());
    let t_matrix = reference_full_potential_t_matrix(state_set.states.len());

    let result = fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
        calculate_full_scattering: false,
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    assert_eq!(result.system_matrix.shape(), &[8, 8]);
    assert_eq!(result.system_matrix.strides(), &[1, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.scattering.strides(), &[1, 8, 64]);
    assert_complex32_close(
        matrix_sum(result.system_matrix.view()),
        Complex32::new(8.191_353, -0.610_848),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.843_191_9, 4.688_064),
    );
    assert_complex32_close(
        result.scattering[(0, 0, 0)],
        Complex32::new(-0.006_074_232, -0.004_277_690_3),
    );
    assert_complex32_close(
        result.scattering[(1, 3, 0)],
        Complex32::new(-0.063_446_34, 0.043_493_286),
    );
    assert_complex32_close(
        result.scattering[(6, 7, 0)],
        Complex32::new(-0.096_970_54, 0.136_094_53),
    );
    assert_eq!(result.full_scattering, None);
    Ok(())
}

#[test]
fn fms_full_potential_lu_scattering_returns_full_matrix_when_requested()
-> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, _) = reference_gglu_inputs(state_set.states.len());
    let t_matrix = reference_full_potential_t_matrix(state_set.states.len());

    let result = fms_full_potential_lu_scattering(FmsFullPotentialLuInput {
        calculate_full_scattering: true,
        states: &state_set.states,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &state_set.representative_offsets,
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    })?;

    let Some(full_scattering) = result.full_scattering else {
        return Err("missing full full-potential scattering matrix".into());
    };
    assert_eq!(full_scattering.shape(), &[8, 8]);
    assert_complex32_close(
        matrix_sum(full_scattering.view()),
        Complex32::new(-2.843_191_9, 4.688_064),
    );
    Ok(())
}

#[test]
fn fms_lu_scattering_rejects_missing_representative() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_lu_scattering(FmsLuInput {
        states: &state_set.states,
        calculate_full_scattering: false,
        spin_channels: 2,
        global_lmax: 1,
        potential_lmax: &[1],
        representative_offsets: &[None],
        potential_start: 0,
        potential_end: 0,
        free_propagator: free_propagator.view(),
        t_matrix: t_matrix.view(),
    });

    assert!(matches!(
        result,
        Err(FmsError::MissingRepresentativePotential { potential: 0 })
    ));
    Ok(())
}
