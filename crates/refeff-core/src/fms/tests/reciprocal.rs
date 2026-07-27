use ndarray::{Array2, ShapeBuilder, array};
use num_complex::Complex32;

use super::super::{
    FmsReciprocalAccumulator, FmsReciprocalCoreHoleInput, FmsReciprocalError, FmsReciprocalPlan,
    fms_reciprocal_apply_core_hole,
};

fn assert_complex_close(actual: Complex32, expected: Complex32) {
    assert!(
        (actual - expected).norm() <= 2.0e-6,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn integrates_one_site_s_wave_resolvents_in_mesh_order() -> Result<(), FmsReciprocalError> {
    let t = Complex32::new(0.2, -0.1);
    let plan = FmsReciprocalPlan::new(array![[t]].view())?;
    let structure_factors = [Complex32::new(0.3, 0.05), Complex32::new(-0.1, 0.2)];
    let weights = [1.0, 3.0];
    let mut accumulator = FmsReciprocalAccumulator::new(1)?;
    let mut expected = Complex32::new(0.0, 0.0);
    for (&structure_factor, &weight) in structure_factors.iter().zip(weights.iter()) {
        let matrix = array![[structure_factor]];
        let solved = plan.solve_k_point(matrix.view())?;
        let scalar = structure_factor / (Complex32::new(1.0, 0.0) - structure_factor * t);
        assert_complex_close(solved[(0, 0)], scalar);
        accumulator.push(weight, solved.view())?;
        expected += scalar * weight as f32;
    }
    expected /= weights.iter().sum::<f64>() as f32;
    let integrated = accumulator.finish()?;
    assert_complex_close(integrated[(0, 0)], expected);
    Ok(())
}

#[test]
fn core_hole_dyson_update_matches_scalar_identity() -> Result<(), FmsReciprocalError> {
    let green_value = Complex32::new(0.4, 0.1);
    let t_difference = Complex32::new(-0.15, 0.08);
    let green = array![[green_value]];
    let difference = array![[t_difference]];
    let corrected = fms_reciprocal_apply_core_hole(FmsReciprocalCoreHoleInput {
        green: green.view(),
        absorber_state_offset: 0,
        site_block_order: 1,
        t_difference: difference.view(),
    })?;
    let expected = green_value / (Complex32::new(1.0, 0.0) + t_difference * green_value);
    assert_complex_close(corrected[(0, 0)], expected);
    Ok(())
}

#[test]
fn core_hole_dyson_update_preserves_noncommuting_matrix_order() -> Result<(), FmsReciprocalError> {
    let green = array![
        [Complex32::new(0.4, 0.1), Complex32::new(-0.2, 0.05)],
        [Complex32::new(0.3, -0.08), Complex32::new(0.15, 0.2)]
    ];
    let difference = array![
        [Complex32::new(-0.15, 0.08), Complex32::new(0.06, 0.02)],
        [Complex32::new(-0.04, 0.03), Complex32::new(0.11, -0.05)]
    ];
    let corrected = fms_reciprocal_apply_core_hole(FmsReciprocalCoreHoleInput {
        green: green.view(),
        absorber_state_offset: 0,
        site_block_order: 2,
        t_difference: difference.view(),
    })?;

    // Independent two-by-two form of the same Dyson identity:
    // G - G (I+C G)^-1 C G = (I+G C)^-1 G.
    let gc = green.dot(&difference);
    let system = array![
        [Complex32::new(1.0, 0.0) + gc[(0, 0)], gc[(0, 1)]],
        [gc[(1, 0)], Complex32::new(1.0, 0.0) + gc[(1, 1)]]
    ];
    let determinant = system[(0, 0)] * system[(1, 1)] - system[(0, 1)] * system[(1, 0)];
    let inverse = array![
        [system[(1, 1)] / determinant, -system[(0, 1)] / determinant],
        [-system[(1, 0)] / determinant, system[(0, 0)] / determinant]
    ];
    let expected = inverse.dot(&green);
    for column in 0..2 {
        for row in 0..2 {
            assert_complex_close(corrected[(row, column)], expected[(row, column)]);
        }
    }
    Ok(())
}

#[test]
fn reciprocal_fms_rejects_empty_bad_weight_and_bad_absorber_block() {
    assert!(matches!(
        FmsReciprocalPlan::new(Array2::zeros((0, 0)).view()),
        Err(FmsReciprocalError::InvalidMatrixShape { .. })
    ));

    let mut accumulator = FmsReciprocalAccumulator::new(1).expect("valid order");
    assert!(matches!(
        accumulator.push(0.0, array![[Complex32::new(1.0, 0.0)]].view()),
        Err(FmsReciprocalError::InvalidKPointWeight { .. })
    ));
    assert!(matches!(
        accumulator.finish(),
        Err(FmsReciprocalError::EmptyKPointMesh)
    ));

    let green = Array2::from_elem((2, 2).f(), Complex32::new(0.1, 0.0));
    let difference = array![[Complex32::new(0.2, 0.0)]];
    assert!(matches!(
        fms_reciprocal_apply_core_hole(FmsReciprocalCoreHoleInput {
            green: green.view(),
            absorber_state_offset: 2,
            site_block_order: 1,
            t_difference: difference.view(),
        }),
        Err(FmsReciprocalError::AbsorberBlockOutOfRange { .. })
    ));
}

#[test]
fn reciprocal_fms_rejects_singular_kkr_matrix() -> Result<(), FmsReciprocalError> {
    let plan = FmsReciprocalPlan::new(array![[Complex32::new(1.0, 0.0)]].view())?;
    assert!(matches!(
        plan.solve_k_point(array![[Complex32::new(1.0, 0.0)]].view()),
        Err(FmsReciprocalError::Linalg(_))
    ));
    Ok(())
}
