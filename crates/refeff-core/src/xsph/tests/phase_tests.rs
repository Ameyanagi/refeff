use super::{support::*, *};
use ndarray::Axis;

#[test]
fn xsph_phase_angular_limit_matches_feff_phase_reference() -> Result<(), XsphError> {
    let floor = arr1(&[
        Complex::new(-0.2, 0.01),
        Complex::new(0.0, 0.02),
        Complex::new(0.5, 0.03),
        Complex::new(2.0, 0.04),
    ]);
    let result = xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
        energies: floor.view(),
        energy_count: floor.len(),
        auxiliary_count: 0,
        muffin_tin_radius: 2.3,
        max_angular_momentum: 10,
    })?;
    assert_eq!(result.angular_limit, 5);
    assert_eq!(result.uncapped_limit, 5);
    assert_close(result.max_wave_number, 2.0);
    assert_eq!(result.accuracy_warning_wave_number, None);

    let cap = arr1(&[
        Complex::new(1.0, 0.01),
        Complex::new(20.0, 0.02),
        Complex::new(50.0, 0.03),
    ]);
    let result = xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
        energies: cap.view(),
        energy_count: cap.len(),
        auxiliary_count: 0,
        muffin_tin_radius: 3.0,
        max_angular_momentum: 8,
    })?;
    assert_eq!(result.angular_limit, 8);
    assert_eq!(result.uncapped_limit, 21);
    assert_close(result.max_wave_number, 10.0);
    assert_eq!(result.accuracy_warning_wave_number, Some(7));

    let exclude_ne3 = arr1(&[
        Complex::new(1.0, 0.01),
        Complex::new(9.0, 0.02),
        Complex::new(16.0, 0.03),
        Complex::new(1000.0, 0.04),
        Complex::new(2000.0, 0.05),
    ]);
    let result = xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
        energies: exclude_ne3.view(),
        energy_count: exclude_ne3.len(),
        auxiliary_count: 2,
        muffin_tin_radius: 1.5,
        max_angular_momentum: 12,
    })?;
    assert_eq!(result.angular_limit, 5);
    assert_eq!(result.uncapped_limit, 5);
    assert_close(result.max_wave_number, 5.656_854_249_492_381);
    assert_eq!(result.accuracy_warning_wave_number, None);

    let negative = arr1(&[Complex::new(-5.0, 0.01), Complex::new(-1.0, 0.02)]);
    let result = xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
        energies: negative.view(),
        energy_count: negative.len(),
        auxiliary_count: 0,
        muffin_tin_radius: 2.0,
        max_angular_momentum: 4,
    })?;
    assert_eq!(result.angular_limit, 4);
    assert_eq!(result.uncapped_limit, 5);
    assert_close(result.max_wave_number, 0.0);
    assert_eq!(result.accuracy_warning_wave_number, Some(5));

    Ok(())
}

#[test]
fn xsph_phase_angular_limit_rejects_invalid_inputs() {
    let energies = arr1(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);

    assert_eq!(
        xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
            energies: energies.view(),
            energy_count: 0,
            auxiliary_count: 0,
            muffin_tin_radius: 1.0,
            max_angular_momentum: 10,
        }),
        Err(XsphError::EmptyPhaseMesh)
    );
    assert_eq!(
        xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
            energies: energies.view(),
            energy_count: 1,
            auxiliary_count: 2,
            muffin_tin_radius: 1.0,
            max_angular_momentum: 10,
        }),
        Err(XsphError::InvalidAuxiliaryEnergyCount {
            auxiliary_count: 2,
            energy_count: 1,
        })
    );
    assert_eq!(
        xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
            energies: energies.view(),
            energy_count: 3,
            auxiliary_count: 0,
            muffin_tin_radius: 1.0,
            max_angular_momentum: 10,
        }),
        Err(XsphError::LengthTooShort {
            name: "energies",
            required: 3,
            actual: 2,
        })
    );
    assert_eq!(
        xsph_phase_angular_limit(XsphPhaseAngularLimitInput {
            energies: energies.view(),
            energy_count: 2,
            auxiliary_count: 0,
            muffin_tin_radius: 0.0,
            max_angular_momentum: 10,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0,
        })
    );
}

#[test]
fn xsph_phase_energy_setup_matches_feff_phase_reference() -> Result<(), XsphError> {
    let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
        energy: Complex::new(2.0, 0.25),
        reference_energy: Complex::new(0.4, 0.05),
        muffin_tin_potential: 0.1,
        lreal: 0,
        energy_index: 1,
        real_mesh_count: 3,
        muffin_tin_radius: 2.3,
        exchange_selector: 0,
    })?;
    assert_eq!(result.decision, XsphPhaseEnergyDecision::Active);
    assert_eq!(result.cycle_count, Some(0));
    assert_phase_setup_dynamics(
        result.dynamics.expect("active dynamics"),
        Complex::new(1.6, 0.2),
        Complex::new(1.5, 0.2),
        Complex::new(1.792_369_197_066_116_6, 0.111_593_660_928_35),
        Complex::new(1.735_912_980_471_197_5, 0.115_222_351_384_390_38),
        Complex::new(4.122_449_153_252_067_5, 0.256_665_420_135_205),
        Complex::new(3.992_599_855_083_754, 0.265_011_408_184_097_85),
    );

    let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
        energy: Complex::new(2.0, 0.25),
        reference_energy: Complex::new(0.4, 0.05),
        muffin_tin_potential: 0.1,
        lreal: 2,
        energy_index: 1,
        real_mesh_count: 3,
        muffin_tin_radius: 2.3,
        exchange_selector: 0,
    })?;
    assert_eq!(result.decision, XsphPhaseEnergyDecision::Active);
    assert_eq!(result.cycle_count, Some(0));
    assert_phase_setup_dynamics(
        result.dynamics.expect("real-mesh dynamics"),
        Complex::new(1.6, 0.0),
        Complex::new(1.5, 0.2),
        Complex::new(1.788_892_485_166_876, 0.0),
        Complex::new(1.735_912_980_471_197_5, 0.115_222_351_384_390_38),
        Complex::new(4.114_452_715_883_814_4, 0.0),
        Complex::new(3.992_599_855_083_754, 0.265_011_408_184_097_85),
    );

    let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
        energy: Complex::new(2.0, 0.25),
        reference_energy: Complex::new(0.4, 0.05),
        muffin_tin_potential: 0.1,
        lreal: 2,
        energy_index: 3,
        real_mesh_count: 3,
        muffin_tin_radius: 2.3,
        exchange_selector: 0,
    })?;
    assert_eq!(result.decision, XsphPhaseEnergyDecision::Active);
    assert_complex_close(
        result.dynamics.expect("post-ne1 dynamics").momentum_squared,
        Complex::new(1.6, 0.2),
    );

    let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
        energy: Complex::new(2.0, 0.25),
        reference_energy: Complex::new(0.4, 0.05),
        muffin_tin_potential: 0.1,
        lreal: 0,
        energy_index: 1,
        real_mesh_count: 3,
        muffin_tin_radius: 2.3,
        exchange_selector: 5,
    })?;
    assert_eq!(result.decision, XsphPhaseEnergyDecision::Active);
    assert_eq!(result.cycle_count, Some(3));

    Ok(())
}

#[test]
fn xsph_phase_energy_setup_preserves_feff_skip_branches() -> Result<(), XsphError> {
    for energy in [Complex::new(-11.0, 0.25), Complex::new(301.0, 0.25)] {
        let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
            energy,
            reference_energy: Complex::new(0.4, 0.05),
            muffin_tin_potential: 0.1,
            lreal: 0,
            energy_index: 1,
            real_mesh_count: 3,
            muffin_tin_radius: 2.3,
            exchange_selector: 0,
        })?;
        assert_eq!(
            result.decision,
            XsphPhaseEnergyDecision::OutsideEnergyWindow
        );
        assert_eq!(result.dynamics, None);
        assert_eq!(result.cycle_count, None);
    }

    let result = xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
        energy: Complex::new(0.2, 0.0),
        reference_energy: Complex::new(0.5, 0.1),
        muffin_tin_potential: 0.1,
        lreal: 0,
        energy_index: 1,
        real_mesh_count: 3,
        muffin_tin_radius: 2.3,
        exchange_selector: 0,
    })?;
    assert_eq!(
        result.decision,
        XsphPhaseEnergyDecision::NonPositiveMomentum
    );
    assert_eq!(result.cycle_count, None);
    assert_phase_setup_dynamics(
        result.dynamics.expect("nonpositive dynamics"),
        Complex::new(-0.3, -0.1),
        Complex::new(-0.4, -0.1),
        Complex::new(0.127_386_695_296_459_6, -0.784_998_796_196_273_9),
        Complex::new(0.110_951_183_840_342_42, -0.901_278_080_001_552_6),
        Complex::new(0.292_989_399_181_857_07, -1.805_497_231_251_429_7),
        Complex::new(0.255_187_722_832_787_5, -2.072_939_584_003_570_7),
    );

    Ok(())
}

#[test]
fn xsph_phase_energy_setup_rejects_invalid_inputs() {
    assert!(matches!(
        xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
            energy: Complex::new(1.0, 0.0),
            reference_energy: Complex::new(0.0, 0.0),
            muffin_tin_potential: 0.0,
            lreal: 0,
            energy_index: 0,
            real_mesh_count: 1,
            muffin_tin_radius: 0.0,
            exchange_selector: 0,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0
        })
    ));

    assert!(matches!(
        xsph_phase_energy_setup(XsphPhaseEnergySetupInput {
            energy: Complex::new(Real::NAN, 0.0),
            reference_energy: Complex::new(0.0, 0.0),
            muffin_tin_potential: 0.0,
            lreal: 0,
            energy_index: 0,
            real_mesh_count: 1,
            muffin_tin_radius: 1.0,
            exchange_selector: 0,
        }),
        Err(XsphError::NonFiniteComplex { name: "energy", .. })
    ));
}

#[test]
fn xsph_phase_channel_plan_matches_feff_phase_reference() -> Result<(), XsphError> {
    let plan = xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
        angular_limit: 2,
        log_step: 0.10,
        initial_cycle_count: 3,
        spin_channels: 2,
        spin: 1,
    })?;
    assert_eq!(
        plan.channels,
        vec![
            phase_channel(-2, 3, 4, -3, 0, 3, false),
            phase_channel(-1, 2, 3, -2, 0, 3, false),
            phase_channel(0, 1, 2, -1, 0, 3, false),
            phase_channel(1, 2, 1, 1, 0, 3, false),
            phase_channel(2, 3, 2, 2, 0, 3, false),
        ]
    );

    let plan = xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
        angular_limit: 2,
        log_step: 0.10,
        initial_cycle_count: 3,
        spin_channels: 1,
        spin: 0,
    })?;
    assert_eq!(
        plan.channels,
        vec![
            phase_channel(-2, 3, 4, -3, 1, 3, false),
            phase_channel(-1, 2, 3, -2, 1, 3, false),
            phase_channel(0, 1, 2, -1, 0, 3, false),
            phase_channel(1, 2, 3, -2, 1, 3, false),
            phase_channel(2, 3, 4, -3, 1, 3, false),
        ]
    );

    Ok(())
}

#[test]
fn xsph_phase_channel_plan_preserves_local_exchange_cycle_state() -> Result<(), XsphError> {
    let plan = xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
        angular_limit: 3,
        log_step: 0.20,
        initial_cycle_count: 3,
        spin_channels: 2,
        spin: 1,
    })?;
    assert_eq!(
        plan.channels,
        vec![
            phase_channel(-3, 4, 5, -4, 0, 0, true),
            phase_channel(-2, 3, 4, -3, 0, 0, true),
            phase_channel(-1, 2, 3, -2, 0, 0, false),
            phase_channel(0, 1, 2, -1, 0, 0, false),
            phase_channel(1, 2, 1, 1, 0, 0, false),
            phase_channel(2, 3, 2, 2, 0, 0, true),
            phase_channel(3, 4, 3, 3, 0, 0, true),
        ]
    );

    Ok(())
}

#[test]
fn xsph_phase_channel_plan_rejects_invalid_inputs() {
    assert!(matches!(
        xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
            angular_limit: 1,
            log_step: Real::NAN,
            initial_cycle_count: 3,
            spin_channels: 2,
            spin: 1,
        }),
        Err(XsphError::NonFiniteScalar {
            name: "log_step",
            ..
        })
    ));

    assert!(matches!(
        xsph_phase_channel_plan(XsphPhaseChannelPlanInput {
            angular_limit: usize::MAX,
            log_step: 0.10,
            initial_cycle_count: 3,
            spin_channels: 2,
            spin: 1,
        }),
        Err(XsphError::SizeOutOfRange {
            name: "angular_limit",
            ..
        })
    ));
}

#[test]
fn xsph_phase_cutoff_matches_feff_phase_reference() -> Result<(), XsphError> {
    assert_phase_cutoff(
        4,
        Complex::new(5.0e-7, 0.0),
        Complex::new(5.0e-7, 0.0),
        false,
        true,
    )?;
    assert_phase_cutoff(
        3,
        Complex::new(2.0e-6, 0.0),
        Complex::new(0.0, 0.0),
        true,
        false,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(2.0e-6, 0.0),
        Complex::new(0.0, 0.0),
        true,
        true,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(8.0e-6, 0.0),
        Complex::new(8.0e-6, 0.0),
        false,
        true,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(1.2e-5, 0.0),
        Complex::new(1.2e-5, 0.0),
        false,
        false,
    )?;
    assert_phase_cutoff(
        -4,
        Complex::new(2.0e-6, 0.0),
        Complex::new(0.0, 0.0),
        true,
        false,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(2.0e-6, 1.0e-6),
        Complex::new(0.0, 0.0),
        true,
        true,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(1.0e-6, 0.0),
        Complex::new(0.0, 0.0),
        true,
        true,
    )?;
    assert_phase_cutoff(
        4,
        Complex::new(1.0e-5, 0.0),
        Complex::new(1.0e-5, 0.0),
        false,
        false,
    )?;

    Ok(())
}

#[test]
fn xsph_phase_cutoff_rejects_invalid_inputs() {
    assert!(matches!(
        xsph_phase_cutoff(XsphPhaseCutoffInput {
            angular_channel: 4,
            phase_shift: Complex::new(Real::NAN, 0.0),
        }),
        Err(XsphError::NonFiniteComplex {
            name: "phase_shift",
            ..
        })
    ));

    assert!(matches!(
        xsph_phase_cutoff(XsphPhaseCutoffInput {
            angular_channel: 1,
            phase_shift: Complex::new(0.0, -1000.0),
        }),
        Err(XsphError::NonFiniteComplex {
            name: "phase_scattering_change",
            ..
        })
    ));
}

#[test]
fn xsph_phase_reference_tail_matches_feff_phase_reference() -> Result<(), XsphError> {
    let mut normal = phase_reference_values(200, 5);
    let result = xsph_phase_reference_tail(normal.view_mut(), 5, 3, 2)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 4,
            filled_count: 2,
        }
    );
    assert_eq!(
        normal,
        arr1(&[
            Complex::new(201.0, -21.0),
            Complex::new(202.0, -22.0),
            Complex::new(203.0, -23.0),
            Complex::new(203.0, -23.0),
            Complex::new(203.0, -23.0),
        ])
    );

    let mut no_aux = phase_reference_values(200, 4);
    let result = xsph_phase_reference_tail(no_aux.view_mut(), 4, 2, 0)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 5,
            filled_count: 0,
        }
    );
    assert_eq!(no_aux, phase_reference_values(200, 4));

    let mut all_aux = phase_reference_values(300, 4);
    let result = xsph_phase_reference_tail(all_aux.view_mut(), 4, 2, 4)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 1,
            filled_count: 4,
        }
    );
    assert_eq!(
        all_aux,
        arr1(&[
            Complex::new(302.0, -32.0),
            Complex::new(302.0, -32.0),
            Complex::new(302.0, -32.0),
            Complex::new(302.0, -32.0),
        ])
    );

    let mut over_aux = phase_reference_values(100, 3);
    let result = xsph_phase_reference_tail(over_aux.view_mut(), 3, 2, 5)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 1,
            filled_count: 3,
        }
    );
    assert_eq!(
        over_aux,
        arr1(&[
            Complex::new(102.0, -12.0),
            Complex::new(102.0, -12.0),
            Complex::new(102.0, -12.0),
        ])
    );

    let mut zero_ne = phase_reference_values(100, 1);
    let result = xsph_phase_reference_tail(zero_ne.view_mut(), 0, 1, 2)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 1,
            filled_count: 0,
        }
    );
    assert_eq!(zero_ne, phase_reference_values(100, 1));

    Ok(())
}

#[test]
fn xsph_phase_reference_tail_rejects_invalid_inputs() {
    let mut short = phase_reference_values(100, 2);
    assert_eq!(
        xsph_phase_reference_tail(short.view_mut(), 3, 1, 1),
        Err(XsphError::LengthTooShort {
            name: "reference_energies",
            required: 3,
            actual: 2,
        })
    );

    let mut references = phase_reference_values(100, 3);
    assert_eq!(
        xsph_phase_reference_tail(references.view_mut(), 3, 0, 2),
        Err(XsphError::InvalidRealEnergyCount {
            real_mesh_count: 0,
            energy_count: 3,
        })
    );
    assert_eq!(
        xsph_phase_reference_tail(references.view_mut(), 3, 4, 2),
        Err(XsphError::InvalidRealEnergyCount {
            real_mesh_count: 4,
            energy_count: 3,
        })
    );
}

#[test]
fn xsph_hubbard_phase_reference_tail_matches_feff_phase_h_reference() -> Result<(), XsphError> {
    let mut normal = phase_reference_values(200, 5);
    let result = xsph_hubbard_phase_reference_tail(normal.view_mut(), 5, 3, 2)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 4,
            filled_count: 2,
        }
    );
    assert_eq!(
        normal,
        arr1(&[
            Complex::new(201.0, -21.0),
            Complex::new(202.0, -22.0),
            Complex::new(203.0, -23.0),
            Complex::new(203.0, -23.0),
            Complex::new(203.0, -23.0),
        ])
    );

    let mut no_aux = phase_reference_values(300, 4);
    let result = xsph_hubbard_phase_reference_tail(no_aux.view_mut(), 4, 2, 0)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 5,
            filled_count: 0,
        }
    );
    assert_eq!(no_aux, phase_reference_values(300, 4));

    let mut all_aux = phase_reference_values(400, 4);
    let result = xsph_hubbard_phase_reference_tail(all_aux.view_mut(), 4, 2, 4)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 1,
            filled_count: 4,
        }
    );
    assert_eq!(
        all_aux,
        arr1(&[
            Complex::new(402.0, -42.0),
            Complex::new(402.0, -42.0),
            Complex::new(402.0, -42.0),
            Complex::new(402.0, -42.0),
        ])
    );

    let mut last_one = phase_reference_values(500, 6);
    let result = xsph_hubbard_phase_reference_tail(last_one.view_mut(), 6, 1, 1)?;
    assert_eq!(
        result,
        XsphPhaseReferenceTail {
            start_index_1based: 6,
            filled_count: 1,
        }
    );
    assert_complex_close(last_one[5], Complex::new(501.0, -51.0));

    Ok(())
}

#[test]
fn xsph_hubbard_phase_reference_tail_rejects_invalid_inputs() {
    let mut short = phase_reference_values(100, 2);
    assert_eq!(
        xsph_hubbard_phase_reference_tail(short.view_mut(), 3, 1, 1),
        Err(XsphError::LengthTooShort {
            name: "reference_energies",
            required: 3,
            actual: 2,
        })
    );

    let mut references = phase_reference_values(100, 3);
    assert_eq!(
        xsph_hubbard_phase_reference_tail(references.view_mut(), 3, 2, 5),
        Err(XsphError::InvalidAuxiliaryEnergyCount {
            auxiliary_count: 5,
            energy_count: 3,
        })
    );
    assert_eq!(
        xsph_hubbard_phase_reference_tail(references.view_mut(), 3, 0, 2),
        Err(XsphError::InvalidRealEnergyCount {
            real_mesh_count: 0,
            energy_count: 3,
        })
    );
    assert_eq!(
        xsph_hubbard_phase_reference_tail(references.view_mut(), 3, 4, 2),
        Err(XsphError::InvalidRealEnergyCount {
            real_mesh_count: 4,
            energy_count: 3,
        })
    );
}

#[test]
fn xsph_phase_radial_indices_match_feff_phase_reference() -> Result<(), XsphError> {
    let cases = [
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: 2.30,
                grid_origin: 8.80,
                log_step: 0.05,
                radial_capacity: 251,
            },
            193.658_182_458_702_1,
            193,
            194,
            195,
        ),
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: 0.85,
                grid_origin: 8.80,
                log_step: 0.05,
                radial_capacity: 251,
            },
            173.749_621_410_044_5,
            173,
            174,
            175,
        ),
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: (4.25_f64 - 1.00).exp(),
                grid_origin: 1.00,
                log_step: 0.25,
                radial_capacity: 32,
            },
            18.0,
            18,
            19,
            20,
        ),
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: 1.75,
                grid_origin: 3.10,
                log_step: 0.37,
                radial_capacity: 40,
            },
            10.890_853_480_906_548,
            10,
            11,
            12,
        ),
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: 1.75,
                grid_origin: 3.10,
                log_step: 0.37,
                radial_capacity: 12,
            },
            10.890_853_480_906_548,
            10,
            11,
            12,
        ),
        (
            XsphPhaseRadialIndicesInput {
                muffin_tin_radius: 1.00,
                grid_origin: -1.40,
                log_step: 1.00,
                radial_capacity: 20,
            },
            -0.399_999_999_999_999_9,
            0,
            1,
            2,
        ),
    ];

    for (input, raw_muffin_tin_index, muffin_tin_index, radial_match, reference) in cases {
        let result = xsph_phase_radial_indices(input)?;
        assert_close(result.raw_muffin_tin_index, raw_muffin_tin_index);
        assert_eq!(result.muffin_tin_index, muffin_tin_index);
        assert_eq!(result.radial_match_index_1based, radial_match);
        assert_eq!(result.reference_index_1based, reference);
    }

    Ok(())
}

#[test]
fn xsph_phase_radial_indices_reject_invalid_inputs() {
    assert_eq!(
        xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius: 0.0,
            grid_origin: 8.80,
            log_step: 0.05,
            radial_capacity: 251,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0,
        })
    );

    assert_eq!(
        xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius: 1.0,
            grid_origin: 8.80,
            log_step: 0.0,
            radial_capacity: 251,
        }),
        Err(XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: 0.0,
        })
    );

    assert_eq!(
        xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius: 1.75,
            grid_origin: 3.10,
            log_step: 0.37,
            radial_capacity: 11,
        }),
        Err(XsphError::LengthTooShort {
            name: "radial_grid",
            required: 12,
            actual: 11,
        })
    );

    assert_eq!(
        xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius: 1.0,
            grid_origin: -2.40,
            log_step: 1.0,
            radial_capacity: 20,
        }),
        Err(XsphError::IntegerOutOfRange {
            name: "radial_match_index_1based",
            value: 0,
        })
    );

    assert!(matches!(
        xsph_phase_radial_indices(XsphPhaseRadialIndicesInput {
            muffin_tin_radius: 1.0,
            grid_origin: i32::MAX as Real + 1.0e6,
            log_step: 1.0,
            radial_capacity: 20,
        }),
        Err(XsphError::RealIntegerOutOfRange {
            name: "muffin_tin_index",
            ..
        })
    ));
}

#[test]
fn xsph_phase_self_energy_summary_matches_feff_mpse_reference() -> Result<(), XsphError> {
    let mut electron_density = Array1::<Real>::from_elem(200, 1.0e-6);
    electron_density[194] = 0.018;
    electron_density[11] = 0.0045;
    electron_density[1] = 0.00075;
    electron_density[0] = 1.0;

    let cases = [
        (195, 0.018, 2.367_080_141_024_981, 12.941_720_228_732_04),
        (12, 0.0045, 3.757_505_505_956_089, 6.470_860_114_366_022),
        (2, 0.00075, 6.827_840_632_552_956, 2.641_717_579_520_726),
        (1, 1.0, 0.620_350_490_899_400_1, 96.461_887_257_429_9),
    ];

    for (reference_index_1based, density, wigner_seitz_radius, plasma_frequency_ev) in cases {
        let summary = xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
            electron_density: electron_density.view(),
            reference_index_1based,
        })?;
        assert_close(summary.electron_density, density);
        assert_close(summary.wigner_seitz_radius, wigner_seitz_radius);
        assert_close(summary.plasma_frequency_ev, plasma_frequency_ev);
    }

    Ok(())
}

#[test]
fn xsph_phase_self_energy_summary_rejects_invalid_inputs() {
    let density = arr1(&[0.018, 0.0045]);

    assert_eq!(
        xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
            electron_density: density.view(),
            reference_index_1based: 0,
        }),
        Err(XsphError::InvalidPhaseRadialReferenceIndex { index_1based: 0 })
    );

    assert_eq!(
        xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
            electron_density: density.view(),
            reference_index_1based: 3,
        }),
        Err(XsphError::LengthTooShort {
            name: "electron_density",
            required: 3,
            actual: 2,
        })
    );

    let zero_density = arr1(&[0.0]);
    assert_eq!(
        xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
            electron_density: zero_density.view(),
            reference_index_1based: 1,
        }),
        Err(XsphError::InvalidPositiveScalar {
            name: "electron_density",
            value: 0.0,
        })
    );

    let nan_density = arr1(&[Real::NAN]);
    assert!(matches!(
        xsph_phase_self_energy_summary(XsphPhaseSelfEnergySummaryInput {
            electron_density: nan_density.view(),
            reference_index_1based: 1,
        }),
        Err(XsphError::NonFiniteScalar {
            name: "electron_density",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn xsph_phase_plasmon_pole_setup_matches_feff_phase_reference() -> Result<(), XsphError> {
    let density = phase_plasmon_density();
    let dense = xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
        plasmon_selector: 1,
        exchange_selector: 0,
        electron_density: density.view(),
        reference_index_1based: 3,
        excitation_poles: &[
            phase_excitation_pole(14.4, 0.1, 0.012),
            phase_excitation_pole(40.0, 0.25, 0.04),
            phase_excitation_pole(88.0, 0.6, 0.08),
        ],
    })?
    .expect("FEFF enters MPSE pole setup for iPl > 0 and ixc == 0");
    assert_close(dense.electron_density, 0.018);
    assert_close(dense.wigner_seitz_radius, 2.367_080_141_024_981);
    assert_close(dense.plasma_frequency_hartree, 0.475_599_275_712_721_26);
    assert_close(dense.plasma_frequency_ev, 12.941_720_228_732_041);
    assert_eq!(dense.poles.len(), 3);
    assert_phase_plasmon_pole(
        dense.poles[0],
        1.112_680_520_479_064_2,
        0.003_674_930_900_274_282,
        0.012,
    );
    assert_phase_plasmon_pole(
        dense.poles[2],
        6.799_714_291_816_502,
        0.022_049_585_401_645_692,
        0.08,
    );

    let sparse = xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
        plasmon_selector: 2,
        exchange_selector: 0,
        electron_density: density.view(),
        reference_index_1based: 6,
        excitation_poles: &[
            phase_excitation_pole(2.5, 0.1, 0.0025),
            phase_excitation_pole(12.5, 0.4, 0.018),
        ],
    })?
    .expect("FEFF enters MPSE pole setup for positive iPl");
    assert_close(sparse.wigner_seitz_radius, 3.757_505_505_956_088_7);
    assert_close(sparse.plasma_frequency_hartree, 0.237_799_637_856_360_68);
    assert_close(sparse.plasma_frequency_ev, 6.470_860_114_366_022);
    assert_phase_plasmon_pole(
        sparse.poles[0],
        0.386_347_402_944_119_4,
        0.003_674_930_900_274_282,
        0.0025,
    );
    assert_phase_plasmon_pole(
        sparse.poles[1],
        1.931_737_014_720_597,
        0.014_699_723_601_097_128,
        0.018,
    );

    Ok(())
}

#[test]
fn xsph_phase_plasmon_pole_setup_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let density = arr1(&[Real::NAN]);
    let poison = [phase_excitation_pole(Real::NAN, Real::NAN, Real::NAN)];

    let skipped_plasmon = xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
        plasmon_selector: 0,
        exchange_selector: 0,
        electron_density: density.view(),
        reference_index_1based: 0,
        excitation_poles: &poison,
    })?;
    assert_eq!(skipped_plasmon, None);

    let skipped_exchange = xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
        plasmon_selector: 1,
        exchange_selector: 1,
        electron_density: density.view(),
        reference_index_1based: 0,
        excitation_poles: &poison,
    })?;
    assert_eq!(skipped_exchange, None);

    Ok(())
}

#[test]
fn xsph_phase_plasmon_pole_setup_rejects_invalid_inputs() {
    let density = phase_plasmon_density();

    assert_eq!(
        xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            electron_density: density.view(),
            reference_index_1based: 3,
            excitation_poles: &[],
        }),
        Err(XsphError::EmptyIndexSet)
    );

    assert_eq!(
        xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            electron_density: density.view(),
            reference_index_1based: 3,
            excitation_poles: &[phase_excitation_pole(0.0, 0.1, 0.012)],
        }),
        Err(XsphError::InvalidPositiveScalar {
            name: "plasmon_pole_energy",
            value: 0.0,
        })
    );

    assert_eq!(
        xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            electron_density: density.view(),
            reference_index_1based: 3,
            excitation_poles: &[phase_excitation_pole(14.4, 0.0, 0.012)],
        }),
        Err(XsphError::InvalidPositiveScalar {
            name: "plasmon_pole_width",
            value: 0.0,
        })
    );

    assert!(matches!(
        xsph_phase_plasmon_pole_setup(XsphPhasePlasmonPoleSetupInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            electron_density: density.view(),
            reference_index_1based: 3,
            excitation_poles: &[phase_excitation_pole(14.4, 0.1, Real::NAN)],
        }),
        Err(XsphError::NonFiniteScalar {
            name: "plasmon_pole_amplitude",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn xsph_phase_radial_header_matches_feff_print_rl_reference() -> Result<(), XsphError> {
    let first = xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
        print_radial: true,
        potential_index: 0,
        muffin_tin_radius: 2.30,
        angular_limit: 5,
        radial_match_index_1based: 194,
        log_step: 0.05,
        grid_origin: 8.80,
    })?
    .expect("FEFF writes rl.dat header for absorber PrintRl");
    assert_close(first.muffin_tin_radius, 2.30);
    assert_eq!(first.angular_limit, 5);
    assert_eq!(first.radial_match_index_1based, 194);
    assert_close(first.log_step, 0.05);
    assert_close(first.grid_origin, 8.80);

    let second = xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
        print_radial: true,
        potential_index: 0,
        muffin_tin_radius: 1.75,
        angular_limit: 12,
        radial_match_index_1based: 11,
        log_step: 0.37,
        grid_origin: 3.10,
    })?
    .expect("FEFF writes rl.dat header for absorber PrintRl");
    assert_close(second.muffin_tin_radius, 1.75);
    assert_eq!(second.angular_limit, 12);
    assert_eq!(second.radial_match_index_1based, 11);
    assert_close(second.log_step, 0.37);
    assert_close(second.grid_origin, 3.10);

    Ok(())
}

#[test]
fn xsph_phase_radial_header_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let skipped_print = xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
        print_radial: false,
        potential_index: 0,
        muffin_tin_radius: 0.0,
        angular_limit: 5,
        radial_match_index_1based: 0,
        log_step: 0.0,
        grid_origin: Real::NAN,
    })?;
    assert_eq!(skipped_print, None);

    let skipped_potential = xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
        print_radial: true,
        potential_index: 1,
        muffin_tin_radius: 2.30,
        angular_limit: 5,
        radial_match_index_1based: 194,
        log_step: 0.05,
        grid_origin: 8.80,
    })?;
    assert_eq!(skipped_potential, None);

    Ok(())
}

#[test]
fn xsph_phase_radial_header_rejects_invalid_inputs() {
    assert_eq!(
        xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
            print_radial: true,
            potential_index: 0,
            muffin_tin_radius: 0.0,
            angular_limit: 5,
            radial_match_index_1based: 194,
            log_step: 0.05,
            grid_origin: 8.80,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: 0.0,
        })
    );

    assert_eq!(
        xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
            print_radial: true,
            potential_index: 0,
            muffin_tin_radius: 2.30,
            angular_limit: 5,
            radial_match_index_1based: 0,
            log_step: 0.05,
            grid_origin: 8.80,
        }),
        Err(XsphError::InvalidPhaseRadialMatchIndex { index_1based: 0 })
    );

    assert_eq!(
        xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
            print_radial: true,
            potential_index: 0,
            muffin_tin_radius: 2.30,
            angular_limit: 5,
            radial_match_index_1based: 194,
            log_step: 0.0,
            grid_origin: 8.80,
        }),
        Err(XsphError::InvalidPositiveScalar {
            name: "log_step",
            value: 0.0,
        })
    );

    assert!(matches!(
        xsph_phase_radial_header(XsphPhaseRadialHeaderInput {
            print_radial: true,
            potential_index: 0,
            muffin_tin_radius: 2.30,
            angular_limit: 5,
            radial_match_index_1based: 194,
            log_step: 0.05,
            grid_origin: Real::NAN,
        }),
        Err(XsphError::NonFiniteScalar {
            name: "grid_origin",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn xsph_empty_cell_phase_matches_feff_phase_reference() -> Result<(), XsphError> {
    let cases = [
        (
            XsphEmptyCellPhaseInput {
                muffin_tin_radius: 2.3,
                wave_number: Complex::new(1.4, 0.2),
                empty_cell_wave_number: Complex::new(1.1, 0.15),
                kappa: -1,
            },
            0,
            1,
            Complex::new(-6.697_134_502_390_13e-1, -1.452_847_589_234_203e-1),
            Complex::new(1.334_620_985_836_210_8, -1.291_247_161_504_714e-2),
            Complex::new(5.081_834_410_440_77e-1, -3.312_070_573_088_552_5e-1),
            Complex::new(-5.059_124_293_153_827e-3, -3.019_458_503_905_245e-4),
            Complex::new(-4.705_477_909_201_361e-2, -1.407_721_587_808_134_2e-1),
            Complex::new(3.344_615_381_778_479e-1, -5.936_861_117_008_973e-2),
        ),
        (
            XsphEmptyCellPhaseInput {
                muffin_tin_radius: 2.3,
                wave_number: Complex::new(1.4, 0.2),
                empty_cell_wave_number: Complex::new(1.1, 0.15),
                kappa: 1,
            },
            1,
            0,
            Complex::new(-6.697_134_502_390_132e-1, -1.452_847_589_234_202_4e-1),
            Complex::new(1.334_620_985_836_210_8, -1.291_247_161_504_729_8e-2),
            Complex::new(9.788_960_568_224_105e-1, -8.072_275_912_067_747e-2),
            Complex::new(2.837_486_907_546_769e-3, -1.321_002_375_355_779_4e-3),
            Complex::new(3.140_199_898_115_473_3e-1, -1.001_664_517_080_757e-1),
            Complex::new(1.462_661_771_822_243_2e-1, 1.081_616_450_256_623_3e-1),
        ),
        (
            XsphEmptyCellPhaseInput {
                muffin_tin_radius: 1.7,
                wave_number: Complex::new(0.8, 0.06),
                empty_cell_wave_number: Complex::new(0.52, 0.03),
                kappa: -3,
            },
            2,
            3,
            Complex::new(-1.352_397_299_068_919_6e-3, -7.719_836_788_959_729e-4),
            Complex::new(4.662_528_448_616_549e-1, -7.147_187_601_918_269e-3),
            Complex::new(8.352_618_137_239_29e-2, 9.118_185_073_141_22e-3),
            Complex::new(-3.058_001_760_351_089e-5, -7.570_833_835_939_667e-6),
            Complex::new(1.076_190_345_908_640_7e-1, 1.399_267_223_372_004_6e-2),
            Complex::new(-1.640_688_165_449_007_5, 2.883_835_492_155_909e-1),
        ),
        (
            XsphEmptyCellPhaseInput {
                muffin_tin_radius: 2.9,
                wave_number: Complex::new(2.2, 0.45),
                empty_cell_wave_number: Complex::new(1.75, 0.25),
                kappa: 2,
            },
            2,
            1,
            Complex::new(5.030_039_088_871_637_5, -4.847_946_482_709_506e-1),
            Complex::new(1.149_266_563_163_625_2, 3.121_069_309_134_368e-2),
            Complex::new(3.701_395_805_529_878e-1, -3.969_259_396_277_603e-1),
            Complex::new(-2.939_630_117_355_670_7e-3, -3.202_414_658_320_532_4e-3),
            Complex::new(-1.924_819_333_002_102e-1, -1.704_317_929_243_727e-1),
            Complex::new(2.150_628_178_433_905e-1, -1.771_218_028_752_171_4e-1),
        ),
    ];

    for (
        input,
        large_l,
        small_l,
        phase_shift,
        amplitude,
        regular_large,
        regular_small,
        bessel_j_large,
        neumann_large,
    ) in cases
    {
        let result = xsph_empty_cell_phase(input)?;
        assert_eq!(result.large_l, large_l);
        assert_eq!(result.small_l, small_l);
        assert_complex_close(result.phase_shift, phase_shift);
        assert_complex_close(result.phase_amplitude, amplitude);
        assert_complex_close(result.regular_large_at_muffin_tin, regular_large);
        assert_complex_close(result.regular_small_at_muffin_tin, regular_small);
        assert_complex_close(result.bessel_j_large, bessel_j_large);
        assert_complex_close(result.neumann_large, neumann_large);
    }
    Ok(())
}

#[test]
fn xsph_empty_cell_phase_rejects_invalid_inputs() {
    assert!(matches!(
        xsph_empty_cell_phase(XsphEmptyCellPhaseInput {
            muffin_tin_radius: -1.0,
            wave_number: Complex::new(1.0, 0.0),
            empty_cell_wave_number: Complex::new(1.0, 0.0),
            kappa: -1,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: -1.0
        })
    ));

    assert!(matches!(
        xsph_empty_cell_phase(XsphEmptyCellPhaseInput {
            muffin_tin_radius: 1.0,
            wave_number: Complex::new(1.0, 0.0),
            empty_cell_wave_number: Complex::new(1.0, 0.0),
            kappa: 0,
        }),
        Err(XsphError::ZeroKappa)
    ));
}

#[test]
fn xsph_regular_phase_matches_feff_phase_reference() -> Result<(), XsphError> {
    let cases = [
        (
            XsphRegularPhaseInput {
                muffin_tin_radius: 2.3,
                wave_number: Complex::new(1.4, 0.2),
                regular_large_at_muffin_tin: Complex::new(0.48, -0.12),
                regular_small_at_muffin_tin: Complex::new(-0.032, 0.018),
                kappa: -1,
            },
            0,
            1,
        ),
        (
            XsphRegularPhaseInput {
                muffin_tin_radius: 2.3,
                wave_number: Complex::new(1.4, 0.2),
                regular_large_at_muffin_tin: Complex::new(0.32, 0.06),
                regular_small_at_muffin_tin: Complex::new(0.021, -0.009),
                kappa: 1,
            },
            1,
            0,
        ),
        (
            XsphRegularPhaseInput {
                muffin_tin_radius: 1.7,
                wave_number: Complex::new(0.8, 0.06),
                regular_large_at_muffin_tin: Complex::new(0.018, -0.004),
                regular_small_at_muffin_tin: Complex::new(-0.0014, 0.0007),
                kappa: -3,
            },
            2,
            3,
        ),
        (
            XsphRegularPhaseInput {
                muffin_tin_radius: 2.9,
                wave_number: Complex::new(2.2, 0.45),
                regular_large_at_muffin_tin: Complex::new(-0.14, 0.22),
                regular_small_at_muffin_tin: Complex::new(0.010, 0.006),
                kappa: 2,
            },
            2,
            1,
        ),
    ];

    for (input, large_l, small_l) in cases {
        let active = crate::besjn(
            input.wave_number * input.muffin_tin_radius,
            large_l.max(small_l),
        )?;
        let expected = crate::muffin_tin_phase_amplitude(
            input.muffin_tin_radius,
            input.regular_large_at_muffin_tin,
            input.regular_small_at_muffin_tin,
            input.wave_number,
            active.j[large_l],
            active.y[large_l],
            active.j[small_l],
            active.y[small_l],
            input.kappa,
        )?;
        let result = xsph_regular_phase(input)?;
        assert_eq!(result.large_l, large_l);
        assert_eq!(result.small_l, small_l);
        assert_complex_close(result.phase_shift, expected.phase);
        assert_complex_close(result.phase_amplitude, expected.amplitude);
        assert_complex_close(
            result.regular_large_at_muffin_tin,
            input.regular_large_at_muffin_tin,
        );
        assert_complex_close(
            result.regular_small_at_muffin_tin,
            input.regular_small_at_muffin_tin,
        );
        assert_complex_close(result.bessel_j_large, active.j[large_l]);
        assert_complex_close(result.neumann_large, active.y[large_l]);
        assert_complex_close(result.bessel_j_small, active.j[small_l]);
        assert_complex_close(result.neumann_small, active.y[small_l]);
    }
    Ok(())
}

#[test]
fn xsph_regular_phase_rejects_invalid_inputs() {
    assert!(matches!(
        xsph_regular_phase(XsphRegularPhaseInput {
            muffin_tin_radius: -1.0,
            wave_number: Complex::new(1.0, 0.0),
            regular_large_at_muffin_tin: Complex::new(0.1, 0.0),
            regular_small_at_muffin_tin: Complex::new(0.01, 0.0),
            kappa: -1,
        }),
        Err(XsphError::InvalidPositiveRadius {
            name: "muffin_tin_radius",
            value: -1.0
        })
    ));

    assert!(matches!(
        xsph_regular_phase(XsphRegularPhaseInput {
            muffin_tin_radius: 1.0,
            wave_number: Complex::new(1.0, 0.0),
            regular_large_at_muffin_tin: Complex::new(Real::NAN, 0.0),
            regular_small_at_muffin_tin: Complex::new(0.01, 0.0),
            kappa: -1,
        }),
        Err(XsphError::NonFiniteComplex {
            name: "regular_large_at_muffin_tin",
            index: 0,
            real,
            imaginary: 0.0,
        }) if real.is_nan()
    ));

    assert!(matches!(
        xsph_regular_phase(XsphRegularPhaseInput {
            muffin_tin_radius: 1.0,
            wave_number: Complex::new(1.0, 0.0),
            regular_large_at_muffin_tin: Complex::new(0.1, 0.0),
            regular_small_at_muffin_tin: Complex::new(0.01, 0.0),
            kappa: 0,
        }),
        Err(XsphError::ZeroKappa)
    ));
}

#[test]
fn xsph_phase_grid_preparation_matches_xsphsub_fixvar_sequence() -> Result<(), XsphError> {
    let source_count = 220;
    let radial_count = 120;
    let potential_count = 2;
    let orbital_count = 2;
    let muffin_tin_radii = [0.65, 0.72];
    let electron_density = phase_grid_matrix(source_count, potential_count, 0.18, 0.011);
    let total_potential = phase_grid_matrix(source_count, potential_count, -0.42, 0.006);
    let valence_density = phase_grid_matrix(source_count, potential_count, 0.07, 0.004);
    let valence_potential = phase_grid_matrix(source_count, potential_count, -0.25, 0.003);
    let magnetization = phase_grid_matrix(source_count, potential_count, 0.02, 0.001);
    let bound_large_components =
        phase_grid_spinors(source_count, orbital_count, potential_count, 0.012);
    let bound_small_components =
        phase_grid_spinors(source_count, orbital_count, potential_count, -0.007);

    let prepared = xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
        muffin_tin_radii: &muffin_tin_radii,
        electron_density: electron_density.view(),
        total_potential: total_potential.view(),
        valence_density: valence_density.view(),
        valence_potential: valence_potential.view(),
        magnetization: magnetization.view(),
        bound_large_components: bound_large_components.view(),
        bound_small_components: bound_small_components.view(),
        interstitial_potential: -0.03,
        interstitial_density: 0.004,
        original_radial_dx: 0.05,
        target_radial_dx: 0.10,
        jump_mode: 1,
        potential_jump: 0.0,
        exchange_selector: 5,
        radial_count,
    })?;

    assert_eq!(
        prepared.total_potential.dim(),
        (radial_count, potential_count)
    );
    assert_eq!(
        prepared.bound_active_lengths.dim(),
        (orbital_count, potential_count)
    );

    let total = crate::fix_potential_grid(crate::PotentialGridInput {
        muffin_tin_radius: muffin_tin_radii[0],
        electron_density: electron_density.index_axis(Axis(1), 0),
        total_potential: total_potential.index_axis(Axis(1), 0),
        magnetization: magnetization.index_axis(Axis(1), 0),
        interstitial_potential: -0.03,
        interstitial_density: 0.004,
        original_delta: 0.05,
        new_delta: 0.10,
        jump_mode: 1,
        potential_jump: 0.0,
        output_len: radial_count,
    })?;
    let valence = crate::fix_potential_grid(crate::PotentialGridInput {
        muffin_tin_radius: muffin_tin_radii[0],
        electron_density: valence_density.index_axis(Axis(1), 0),
        total_potential: valence_potential.index_axis(Axis(1), 0),
        magnetization: magnetization.index_axis(Axis(1), 0),
        interstitial_potential: -0.03,
        interstitial_density: 0.004,
        original_delta: 0.05,
        new_delta: 0.10,
        jump_mode: 2,
        potential_jump: total.potential_jump,
        output_len: radial_count,
    })?;
    let spinors = crate::fix_dirac_spinor_orbitals_grid(crate::DiracSpinorOrbitalsGridInput {
        original_delta: 0.05,
        new_delta: 0.10,
        large_components: bound_large_components.index_axis(Axis(2), 0),
        small_components: bound_small_components.index_axis(Axis(2), 0),
        output_len: radial_count,
    })?;

    assert_eq!(prepared.radii, total.radii);
    assert_close(prepared.potential_jumps[0], total.potential_jump);
    assert_eq!(
        prepared.total_potential.index_axis(Axis(1), 0),
        total.total_potential
    );
    assert_eq!(
        prepared.electron_density.index_axis(Axis(1), 0),
        total.charge_density
    );
    assert_eq!(
        prepared.magnetization.index_axis(Axis(1), 0),
        total.magnetization
    );
    assert_eq!(
        prepared.valence_potential.index_axis(Axis(1), 0),
        valence.total_potential
    );
    assert_eq!(
        prepared.valence_density.index_axis(Axis(1), 0),
        valence.charge_density
    );
    assert_eq!(
        prepared.bound_large_components.index_axis(Axis(2), 0),
        spinors.large_components
    );
    assert_eq!(
        prepared.bound_small_components.index_axis(Axis(2), 0),
        spinors.small_components
    );
    assert_eq!(
        prepared.bound_active_lengths.index_axis(Axis(1), 0),
        spinors.active_lengths
    );
    Ok(())
}

#[test]
fn xsph_phase_grid_preparation_clones_total_potential_for_low_exchange() -> Result<(), XsphError> {
    let source_count = 220;
    let radial_count = 120;
    let potential_count = 1;
    let orbital_count = 1;
    let muffin_tin_radii = [0.65];
    let electron_density = phase_grid_matrix(source_count, potential_count, 0.18, 0.011);
    let total_potential = phase_grid_matrix(source_count, potential_count, -0.42, 0.006);
    let valence_density = phase_grid_matrix(source_count, potential_count, 0.07, 0.004);
    let valence_potential = phase_grid_matrix(source_count, potential_count, -0.25, 0.003);
    let magnetization = phase_grid_matrix(source_count, potential_count, 0.02, 0.001);
    let bound_large_components =
        phase_grid_spinors(source_count, orbital_count, potential_count, 0.012);
    let bound_small_components =
        phase_grid_spinors(source_count, orbital_count, potential_count, -0.007);

    let prepared = xsph_phase_grid_preparation(XsphPhaseGridPreparationInput {
        muffin_tin_radii: &muffin_tin_radii,
        electron_density: electron_density.view(),
        total_potential: total_potential.view(),
        valence_density: valence_density.view(),
        valence_potential: valence_potential.view(),
        magnetization: magnetization.view(),
        bound_large_components: bound_large_components.view(),
        bound_small_components: bound_small_components.view(),
        interstitial_potential: -0.03,
        interstitial_density: 0.004,
        original_radial_dx: 0.05,
        target_radial_dx: 0.10,
        jump_mode: 0,
        potential_jump: 0.0,
        exchange_selector: 3,
        radial_count,
    })?;

    assert_eq!(prepared.valence_potential, prepared.total_potential);
    assert!(prepared.valence_density.iter().all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn xsph_phase_radial_output_matches_feff_print_rl_reference() -> Result<(), XsphError> {
    let (large, small) = phase_radial_output_fixture();

    let ll0 = xsph_phase_radial_output(phase_radial_output_input(
        &large,
        &small,
        PhaseRadialCase::new(true, 0, 0, 3, 3, Complex::new(1.25, -0.5)),
    ))?
    .expect("FEFF PrintRl branch should be active for ll=0");
    assert_eq!(ll0.angular_channel, 0);
    assert_eq!(ll0.output_angular_momentum, 0);
    assert_complex_close(ll0.energy, Complex::new(7.5, 0.2));
    assert_complex_close(ll0.phase_shift, Complex::new(0.125, -0.0625));
    assert_eq!(ll0.regular_large.len(), 3);
    assert_eq!(ll0.regular_small.len(), 3);
    assert_complex_close(
        ll0.regular_large[0],
        Complex::new(1.103_448_275_862_069, 1.241_379_310_344_827_6),
    );
    assert_complex_close(
        ll0.regular_small[2],
        Complex::new(1.931_034_482_758_620_6, 0.172_413_793_103_448_3),
    );

    let llneg = xsph_phase_radial_output(phase_radial_output_input(
        &large,
        &small,
        PhaseRadialCase::new(true, 0, -2, 3, 4, Complex::new(1.25, -0.5)),
    ))?
    .expect("FEFF PrintRl branch should be active for negative ll within lmax");
    assert_eq!(llneg.angular_channel, -2);
    assert_eq!(llneg.output_angular_momentum, 2);
    assert_eq!(llneg.regular_large.len(), 4);
    assert_complex_close(
        llneg.regular_large[3],
        Complex::new(3.034_482_758_620_69, 0.413_793_103_448_275_9),
    );
    assert_complex_close(llneg.regular_small[3], Complex::new(0.0, -1.0));

    Ok(())
}

#[test]
fn xsph_phase_radial_output_preserves_feff_skip_conditions() -> Result<(), XsphError> {
    let (large, small) = phase_radial_output_fixture();

    for (print_radial, potential_index, angular_channel, angular_limit) in [
        (true, 0, 1, 3),
        (true, 1, -1, 3),
        (false, 0, -1, 3),
        (true, 0, -4, 3),
    ] {
        let result = xsph_phase_radial_output(phase_radial_output_input(
            &large,
            &small,
            PhaseRadialCase::new(
                print_radial,
                potential_index,
                angular_channel,
                angular_limit,
                3,
                Complex::new(1.25, -0.5),
            ),
        ))?;
        assert_eq!(result, None);
    }

    Ok(())
}

#[test]
fn xsph_phase_radial_output_rejects_invalid_inputs() {
    let (large, small) = phase_radial_output_fixture();

    assert_eq!(
        xsph_phase_radial_output(phase_radial_output_input(
            &large,
            &small,
            PhaseRadialCase::new(true, 0, 0, 3, 0, Complex::new(1.25, -0.5)),
        )),
        Err(XsphError::EmptyIndexSet)
    );

    let short = arr1(&[Complex::new(1.0, 0.0)]);
    assert_eq!(
        xsph_phase_radial_output(XsphPhaseRadialOutputInput {
            regular_large: short.view(),
            ..phase_radial_output_input(
                &large,
                &small,
                PhaseRadialCase::new(true, 0, 0, 3, 2, Complex::new(1.25, -0.5)),
            )
        }),
        Err(XsphError::LengthTooShort {
            name: "regular_large",
            required: 2,
            actual: 1,
        })
    );

    assert_eq!(
        xsph_phase_radial_output(phase_radial_output_input(
            &large,
            &small,
            PhaseRadialCase::new(true, 0, 0, 3, 3, Complex::new(0.0, 0.0)),
        )),
        Err(XsphError::ZeroPhaseAmplitude)
    );

    assert_eq!(
        xsph_phase_radial_output(phase_radial_output_input(
            &large,
            &small,
            PhaseRadialCase::new(true, 0, i32::MIN, 3, 3, Complex::new(1.25, -0.5)),
        )),
        Err(XsphError::IntegerOutOfRange {
            name: "angular_channel",
            value: i32::MIN,
        })
    );
}

#[test]
fn xsph_hubbard_phase_potential_shifts_match_feff_phase_h_reference() -> Result<(), XsphError> {
    let (total, valence, hubbard) = hubbard_phase_fixture();

    let down_l1 = xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
        angular_channel: 1,
        spin_projection: 1,
        total_potential: total.view(),
        valence_potential: valence.view(),
        hubbard_potential: hubbard.view(),
        active_len: 3,
    })?;
    assert_eq!(down_l1.len(), 3);
    assert_hubbard_shift(&down_l1[0], 1, -0.25, 10.75, -1.0, 22.75, -6.0);
    assert_hubbard_shift(&down_l1[1], 2, -0.50, 10.50, -1.0, 22.50, -6.0);
    assert_hubbard_shift(&down_l1[2], 3, -0.75, 10.25, -1.0, 22.25, -6.0);

    let up_l1 = xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
        angular_channel: 1,
        spin_projection: 2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        hubbard_potential: hubbard.view(),
        active_len: 3,
    })?;
    assert_hubbard_shift(&up_l1[0], 1, 0.25, 11.25, -1.0, 23.25, -6.0);
    assert_hubbard_shift(&up_l1[1], 2, 0.50, 11.50, -1.0, 23.50, -6.0);
    assert_hubbard_shift(&up_l1[2], 3, 0.75, 11.75, -1.0, 23.75, -6.0);

    let down_l0 = xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
        angular_channel: 0,
        spin_projection: 1,
        total_potential: total.view(),
        valence_potential: valence.view(),
        hubbard_potential: hubbard.view(),
        active_len: 3,
    })?;
    assert_eq!(down_l0.len(), 1);
    assert_hubbard_shift(&down_l0[0], 0, -0.125, 10.875, -1.0, 22.875, -6.0);

    let up_l2 = xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
        angular_channel: 2,
        spin_projection: 2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        hubbard_potential: hubbard.view(),
        active_len: 3,
    })?;
    assert_eq!(up_l2.len(), 5);
    assert_hubbard_shift(&up_l2[0], 4, 0.10, 11.10, -1.0, 23.10, -6.0);
    assert_hubbard_shift(&up_l2[1], 5, 0.20, 11.20, -1.0, 23.20, -6.0);
    assert_hubbard_shift(&up_l2[2], 6, 0.30, 11.30, -1.0, 23.30, -6.0);
    assert_hubbard_shift(&up_l2[3], 7, 0.40, 11.40, -1.0, 23.40, -6.0);
    assert_hubbard_shift(&up_l2[4], 8, 0.50, 11.50, -1.0, 23.50, -6.0);

    Ok(())
}

#[test]
fn xsph_hubbard_phase_assignments_match_feff_phase_h_reference() -> Result<(), XsphError> {
    let l0 = arr1(&[Complex::new(0.145, -0.045)]);
    let assignments = xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
        energy_index: 1,
        angular_channel: 0,
        hubbard_angular_limit: 2,
        magnetic_phase_shifts: l0.view(),
    })?;
    assert_eq!(assignments.len(), 1);
    assert_hubbard_assignment(&assignments[0], 1, 0, 0, Complex::new(0.145, -0.045));

    let l1 = arr1(&[
        Complex::new(0.260, -0.100),
        Complex::new(0.385, -0.175),
        Complex::new(0.510, -0.250),
    ]);
    let assignments = xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
        energy_index: 2,
        angular_channel: 1,
        hubbard_angular_limit: 2,
        magnetic_phase_shifts: l1.view(),
    })?;
    assert_eq!(assignments.len(), 3);
    assert_hubbard_assignment(&assignments[0], 2, 1, 1, Complex::new(0.260, -0.100));
    assert_hubbard_assignment(&assignments[1], 2, 1, 2, Complex::new(0.385, -0.175));
    assert_hubbard_assignment(&assignments[2], 2, 1, 3, Complex::new(0.510, -0.250));

    let l2 = arr1(&[
        Complex::new(0.625, -0.305),
        Complex::new(0.750, -0.380),
        Complex::new(0.875, -0.455),
        Complex::new(1.000, -0.530),
        Complex::new(1.125, -0.605),
    ]);
    let assignments = xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
        energy_index: 3,
        angular_channel: 2,
        hubbard_angular_limit: 2,
        magnetic_phase_shifts: l2.view(),
    })?;
    assert_eq!(assignments.len(), 5);
    assert_hubbard_assignment(&assignments[0], 3, 2, 4, Complex::new(0.625, -0.305));
    assert_hubbard_assignment(&assignments[4], 3, 2, 8, Complex::new(1.125, -0.605));

    Ok(())
}

#[test]
fn xsph_hubbard_phase_assignments_preserve_feff_skip_conditions() -> Result<(), XsphError> {
    let poison = arr1(&[Complex::new(Real::NAN, Real::NAN)]);

    let negative = xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
        energy_index: 0,
        angular_channel: -1,
        hubbard_angular_limit: 2,
        magnetic_phase_shifts: poison.view(),
    })?;
    assert!(negative.is_empty());

    let above_lx = xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
        energy_index: 0,
        angular_channel: 3,
        hubbard_angular_limit: 2,
        magnetic_phase_shifts: poison.view(),
    })?;
    assert!(above_lx.is_empty());

    Ok(())
}

#[test]
fn xsph_hubbard_phase_assignments_reject_invalid_inputs() {
    let short = arr1(&[
        Complex::new(0.625, -0.305),
        Complex::new(0.750, -0.380),
        Complex::new(0.875, -0.455),
        Complex::new(1.000, -0.530),
    ]);
    assert_eq!(
        xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
            energy_index: 3,
            angular_channel: 2,
            hubbard_angular_limit: 2,
            magnetic_phase_shifts: short.view(),
        }),
        Err(XsphError::LengthTooShort {
            name: "hubbard_phase_shifts",
            required: 5,
            actual: 4,
        })
    );

    let nonfinite = arr1(&[Complex::new(Real::NAN, 0.0)]);
    assert!(matches!(
        xsph_hubbard_phase_assignments(XsphHubbardPhaseAssignmentInput {
            energy_index: 0,
            angular_channel: 0,
            hubbard_angular_limit: 2,
            magnetic_phase_shifts: nonfinite.view(),
        }),
        Err(XsphError::NonFiniteComplex {
            name: "hubbard_phase_shift",
            index: 0,
            real,
            imaginary: 0.0,
        }) if real.is_nan()
    ));
}

#[test]
fn xsph_hubbard_phase_potential_shifts_reject_invalid_inputs() {
    let (total, valence, hubbard) = hubbard_phase_fixture();

    assert!(matches!(
        xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
            angular_channel: -1,
            spin_projection: 1,
            total_potential: total.view(),
            valence_potential: valence.view(),
            hubbard_potential: hubbard.view(),
            active_len: 3,
        }),
        Err(XsphError::NegativeAngularMomentum {
            name: "angular_channel",
            value: -1,
            ..
        })
    ));

    assert_eq!(
        xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
            angular_channel: 1,
            spin_projection: 3,
            total_potential: total.view(),
            valence_potential: valence.view(),
            hubbard_potential: hubbard.view(),
            active_len: 3,
        }),
        Err(XsphError::InvalidHubbardSpinProjection { spin_projection: 3 })
    );

    let short_hubbard = Array2::<Real>::zeros((2, 9));
    assert!(matches!(
        xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
            angular_channel: 2,
            spin_projection: 2,
            total_potential: total.view(),
            valence_potential: valence.view(),
            hubbard_potential: short_hubbard.view(),
            active_len: 3,
        }),
        Err(XsphError::MatrixTooSmall {
            name: "hubbard_potential",
            required: [3, 9],
            actual: [2, 9],
        })
    ));

    assert!(matches!(
        xsph_hubbard_phase_potential_shifts(XsphHubbardPhasePotentialInput {
            angular_channel: 1,
            spin_projection: 1,
            total_potential: total.view(),
            valence_potential: valence.view(),
            hubbard_potential: hubbard.view(),
            active_len: 0,
        }),
        Err(XsphError::EmptyIndexSet)
    ));
}

fn phase_reference_values(base: i32, len: usize) -> Array1<Complex> {
    Array1::from_iter((1..=len).map(|index| {
        let index = index as f64;
        Complex::new(base as f64 + index, -(base as f64 / 10.0 + index))
    }))
}

fn phase_plasmon_density() -> Array1<Real> {
    let mut density = Array1::<Real>::from_elem(8, 1.0e-6);
    density[2] = 0.018;
    density[5] = 0.0045;
    density
}

fn phase_excitation_pole(energy: Real, width: Real, amplitude: Real) -> crate::ExcitationPole {
    crate::ExcitationPole {
        energy,
        width,
        amplitude,
        loss_height: 0.0,
    }
}

fn hubbard_phase_fixture() -> (Array1<Complex>, Array1<Complex>, Array2<Real>) {
    let total =
        Array1::from_iter((1..=5).map(|index| Complex::new(10.0 + index as Real, -index as Real)));
    let valence = Array1::from_iter(
        (1..=5).map(|index| Complex::new(20.0 + index as Real, -2.0 * index as Real)),
    );
    let mut hubbard = Array2::<Real>::zeros((3, 9));
    hubbard[(0, 0)] = -0.125;
    hubbard[(1, 1)] = -0.25;
    hubbard[(1, 2)] = 0.50;
    hubbard[(1, 3)] = -0.75;
    hubbard[(2, 4)] = -0.10;
    hubbard[(2, 5)] = 0.20;
    hubbard[(2, 6)] = -0.30;
    hubbard[(2, 7)] = 0.40;
    hubbard[(2, 8)] = -0.50;
    (total, valence, hubbard)
}

fn phase_radial_output_fixture() -> (Array1<Complex>, Array1<Complex>) {
    (
        arr1(&[
            Complex::new(2.0, 1.0),
            Complex::new(-1.5, 0.5),
            Complex::new(0.25, -2.0),
            Complex::new(4.0, -1.0),
        ]),
        arr1(&[
            Complex::new(-3.0, 0.5),
            Complex::new(1.0, 1.5),
            Complex::new(2.5, -0.75),
            Complex::new(-0.5, -1.25),
        ]),
    )
}

#[derive(Debug, Clone, Copy)]
struct PhaseRadialCase {
    print_radial: bool,
    potential_index: i32,
    angular_channel: i32,
    angular_limit: usize,
    active_len: usize,
    phase_amplitude: Complex,
}

impl PhaseRadialCase {
    fn new(
        print_radial: bool,
        potential_index: i32,
        angular_channel: i32,
        angular_limit: usize,
        active_len: usize,
        phase_amplitude: Complex,
    ) -> Self {
        Self {
            print_radial,
            potential_index,
            angular_channel,
            angular_limit,
            active_len,
            phase_amplitude,
        }
    }
}

fn phase_radial_output_input<'a>(
    regular_large: &'a Array1<Complex>,
    regular_small: &'a Array1<Complex>,
    case: PhaseRadialCase,
) -> XsphPhaseRadialOutputInput<'a> {
    XsphPhaseRadialOutputInput {
        print_radial: case.print_radial,
        potential_index: case.potential_index,
        angular_channel: case.angular_channel,
        angular_limit: case.angular_limit,
        energy: Complex::new(7.5, 0.2),
        phase_shift: Complex::new(0.125, -0.0625),
        phase_amplitude: case.phase_amplitude,
        regular_large: regular_large.view(),
        regular_small: regular_small.view(),
        active_len: case.active_len,
    }
}

fn assert_hubbard_shift(
    actual: &XsphHubbardPhasePotentialShift,
    magnetic_channel: usize,
    shift: Real,
    total_first_re: Real,
    total_first_im: Real,
    valence_last_re: Real,
    valence_last_im: Real,
) {
    assert_eq!(actual.magnetic_channel, magnetic_channel);
    assert_close(actual.shift, shift);
    assert_complex_close(
        actual.total_potential[0],
        Complex::new(total_first_re, total_first_im),
    );
    assert_complex_close(
        actual.valence_potential[actual.valence_potential.len() - 1],
        Complex::new(valence_last_re, valence_last_im),
    );
}

fn assert_hubbard_assignment(
    actual: &XsphHubbardPhaseAssignment,
    energy_index: usize,
    angular_channel: usize,
    magnetic_channel: usize,
    phase_shift: Complex,
) {
    assert_eq!(actual.energy_index, energy_index);
    assert_eq!(actual.angular_channel, angular_channel);
    assert_eq!(actual.magnetic_channel, magnetic_channel);
    assert_complex_close(actual.phase_shift, phase_shift);
}

fn assert_phase_plasmon_pole(
    actual: XsphPhasePlasmonPole,
    energy_over_plasma: Real,
    width_hartree: Real,
    amplitude: Real,
) {
    assert_close(actual.energy_over_plasma, energy_over_plasma);
    assert_close(actual.width_hartree, width_hartree);
    assert_close(actual.amplitude, amplitude);
}

fn phase_channel(
    angular_channel: i32,
    orbital_index: usize,
    partner_orbital_index: usize,
    kappa: i32,
    c3_derivative: i32,
    cycle_count: usize,
    forces_local_exchange: bool,
) -> XsphPhaseChannel {
    XsphPhaseChannel {
        angular_channel,
        orbital_index,
        partner_orbital_index,
        kappa,
        c3_derivative,
        cycle_count,
        forces_local_exchange,
    }
}

fn assert_phase_cutoff(
    angular_channel: i32,
    phase_shift: Complex,
    expected_phase_shift: Complex,
    zeroed: bool,
    terminate_energy: bool,
) -> Result<(), XsphError> {
    let result = xsph_phase_cutoff(XsphPhaseCutoffInput {
        angular_channel,
        phase_shift,
    })?;
    assert_complex_close(result.phase_shift, expected_phase_shift);
    assert_eq!(result.zeroed, zeroed);
    assert_eq!(result.terminate_energy, terminate_energy);
    Ok(())
}

fn assert_phase_setup_dynamics(
    actual: XsphPhaseEnergyDynamics,
    momentum_squared: Complex,
    empty_cell_momentum_squared: Complex,
    wave_number: Complex,
    empty_cell_wave_number: Complex,
    muffin_tin_argument: Complex,
    empty_cell_muffin_tin_argument: Complex,
) {
    assert_complex_close(actual.momentum_squared, momentum_squared);
    assert_complex_close(
        actual.empty_cell_momentum_squared,
        empty_cell_momentum_squared,
    );
    assert_complex_close(actual.wave_number, wave_number);
    assert_complex_close(actual.empty_cell_wave_number, empty_cell_wave_number);
    assert_complex_close(actual.muffin_tin_argument, muffin_tin_argument);
    assert_complex_close(
        actual.empty_cell_muffin_tin_argument,
        empty_cell_muffin_tin_argument,
    );
}

fn phase_grid_matrix(
    radial_count: usize,
    potential_count: usize,
    base: Real,
    slope: Real,
) -> Array2<Real> {
    Array2::from_shape_fn((radial_count, potential_count), |(row, potential)| {
        let row = (row + 1) as Real;
        let potential = potential as Real;
        base + slope * row + 0.002 * potential + 0.0003 * (0.17 * row).sin()
    })
}

fn phase_grid_spinors(
    radial_count: usize,
    orbital_count: usize,
    potential_count: usize,
    scale: Real,
) -> Array3<Real> {
    Array3::from_shape_fn(
        (radial_count, orbital_count, potential_count),
        |(row, orbital, potential)| {
            let row = (row + 1) as Real;
            let orbital = (orbital + 1) as Real;
            let potential = (potential + 1) as Real;
            scale * orbital * potential * (0.05 * row * orbital).sin() * (-0.004 * row).exp()
        },
    )
}
