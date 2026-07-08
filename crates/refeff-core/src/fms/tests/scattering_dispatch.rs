use super::*;

#[test]
fn fms_scattering_method_selection_matches_feff_minv_rules() {
    assert_eq!(
        fms_scattering_method_selection(0, false),
        FmsScatteringMethodSelection {
            effective_minv: 0,
            method: FmsScatteringMethod::Lu,
            forced_lu_for_full_scattering: false,
        }
    );
    assert_eq!(
        fms_scattering_method_selection(1, false).method,
        FmsScatteringMethod::BiCgStab
    );
    assert_eq!(
        fms_scattering_method_selection(2, false).method,
        FmsScatteringMethod::Recursion
    );
    assert_eq!(
        fms_scattering_method_selection(3, false).method,
        FmsScatteringMethod::GravesMorris
    );
    assert_eq!(
        fms_scattering_method_selection(4, false),
        FmsScatteringMethodSelection {
            effective_minv: 4,
            method: FmsScatteringMethod::Tfqmr,
            forced_lu_for_full_scattering: false,
        }
    );
    assert_eq!(
        fms_scattering_method_selection(-1, false).method,
        FmsScatteringMethod::Tfqmr
    );
    assert_eq!(
        fms_scattering_method_selection(3, true),
        FmsScatteringMethodSelection {
            effective_minv: 0,
            method: FmsScatteringMethod::Lu,
            forced_lu_for_full_scattering: true,
        }
    );
    assert_eq!(FmsScatteringMethod::Lu.feff_label(), "LUD");
    assert_eq!(FmsScatteringMethod::BiCgStab.feff_label(), "VdV");
    assert_eq!(FmsScatteringMethod::Recursion.feff_label(), "LLU");
    assert_eq!(FmsScatteringMethod::GravesMorris.feff_label(), "GMS");
    assert_eq!(FmsScatteringMethod::Tfqmr.feff_label(), "TF");
}

#[test]
fn fms_scattering_dispatches_lu_branch() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let result = fms_scattering(reference_scattering_input(
        FmsScatteringMethod::Lu,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    ))?;

    assert_eq!(result.method, FmsScatteringMethod::Lu);
    assert_eq!(result.multiple_scattering_order, None);
    let Some(system_matrix) = result.system_matrix.as_ref() else {
        return Err("missing system matrix".into());
    };
    assert_eq!(system_matrix.shape(), &[8, 8]);
    assert_eq!(result.scattering.shape(), &[8, 8, 1]);
    assert_eq!(result.full_scattering, None);
    assert_complex32_close(
        matrix_sum(system_matrix.view()),
        Complex32::new(8.107_28, -0.542_959_87),
    );
    assert_complex32_close(
        scattering_sum(result.scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    Ok(())
}

#[test]
fn fms_scattering_dispatches_lu_full_matrix_request() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());
    let mut input = reference_scattering_input(
        FmsScatteringMethod::Lu,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    );
    input.calculate_full_scattering = true;

    let result = fms_scattering(input)?;

    assert_eq!(result.method, FmsScatteringMethod::Lu);
    assert_eq!(result.multiple_scattering_order, None);
    let Some(full_scattering) = result.full_scattering else {
        return Err("missing full scattering matrix".into());
    };
    assert_eq!(full_scattering.shape(), &[8, 8]);
    assert_complex32_close(
        matrix_sum(full_scattering.view()),
        Complex32::new(-2.944_320_4, 4.799_401_3),
    );
    Ok(())
}

#[test]
fn fms_scattering_dispatches_iterative_branches() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());

    let cases = [
        (
            FmsScatteringMethod::BiCgStab,
            2,
            Complex32::new(-2.949_217_6, 4.806_942),
        ),
        (
            FmsScatteringMethod::Recursion,
            3,
            Complex32::new(-2.944_324, 4.799_402),
        ),
        (
            FmsScatteringMethod::GravesMorris,
            4,
            Complex32::new(-2.944_321_6, 4.799_405),
        ),
        (
            FmsScatteringMethod::Tfqmr,
            4,
            Complex32::new(-2.944_320_7, 4.799_402_7),
        ),
    ];

    for (method, order, scattering_reference) in cases {
        let result = fms_scattering(reference_scattering_input(
            method,
            &state_set.states,
            &state_set.representative_offsets,
            free_propagator.view(),
            t_matrix.view(),
        ))?;

        assert_eq!(result.method, method);
        assert_eq!(result.multiple_scattering_order, Some(order));
        let Some(system_matrix) = result.system_matrix.as_ref() else {
            return Err("missing system matrix".into());
        };
        assert_eq!(system_matrix.shape(), &[8, 8]);
        assert_eq!(result.scattering.shape(), &[8, 8, 1]);
        assert_eq!(result.full_scattering, None);
        assert_complex32_close(
            scattering_sum(result.scattering.view()),
            scattering_reference,
        );
    }
    Ok(())
}

#[test]
fn fms_scattering_rejects_iterative_full_matrix_request() -> Result<(), Box<dyn Error>> {
    let state_set = construct_state_kets(2, &[0], &[1], 1)?;
    let (free_propagator, t_matrix) = reference_gglu_inputs(state_set.states.len());
    let mut input = reference_scattering_input(
        FmsScatteringMethod::BiCgStab,
        &state_set.states,
        &state_set.representative_offsets,
        free_propagator.view(),
        t_matrix.view(),
    );
    input.calculate_full_scattering = true;

    assert!(matches!(
        fms_scattering(input),
        Err(FmsError::FullScatteringRequiresLu {
            method: FmsScatteringMethod::BiCgStab,
        })
    ));
    Ok(())
}
