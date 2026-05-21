use super::*;

#[test]
fn xgllm_matches_feff_reference() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };

    assert_complex32_close(
        rehr_albers_z_axis_propagator(0, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(415.546_9, -1006.2809),
    );
    assert_complex32_close(
        rehr_albers_z_axis_propagator(1, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(-307.497_3, 722.469_5),
    );
    assert_complex32_close(
        rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(115.08963, -235.94589),
    );
    Ok(())
}

#[test]
fn xgllm_matches_feff_empty_sum_case() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 1,
        magnetic: 0,
        spin: 1,
    };

    assert_complex32_close(
        rehr_albers_z_axis_propagator(2, first, second, xclm.view(), xnlm.view())?,
        Complex32::new(0.0, 0.0),
    );
    Ok(())
}

#[test]
fn xgllm_rejects_invalid_inputs() -> Result<(), Box<dyn Error>> {
    let (xclm, xnlm) = reference_xgllm_tables()?;
    let first = StateKet {
        atom: 1,
        angular_momentum: 2,
        magnetic: 0,
        spin: 1,
    };
    let second = StateKet {
        atom: 2,
        angular_momentum: 3,
        magnetic: 0,
        spin: 1,
    };

    assert_eq!(
        rehr_albers_z_axis_propagator(3, first, second, xclm.view(), xnlm.view()),
        Err(FmsError::MuOutOfRange {
            mu: 3,
            angular_momentum: 2,
        })
    );
    assert_eq!(
        rehr_albers_z_axis_propagator(
            0,
            StateKet { atom: 0, ..first },
            second,
            xclm.view(),
            xnlm.view(),
        ),
        Err(FmsError::InvalidStateAtom { atom: 0 })
    );

    let mut bad_xnlm = xnlm.clone();
    bad_xnlm[(0, 2)] = 0.0;
    assert_eq!(
        rehr_albers_z_axis_propagator(0, first, second, xclm.view(), bad_xnlm.view()),
        Err(FmsError::InvalidNormalization {
            mu: 0,
            angular_momentum: 2,
        })
    );
    Ok(())
}
