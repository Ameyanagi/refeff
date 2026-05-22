use super::{support::*, *};

#[test]
fn coulomb_potential_slw_matches_feff_potslw_long_reference() -> Result<(), GridError> {
    let radii = (1..=251)
        .map(|index| (-8.8 + 0.05 * (index - 1) as Real).exp())
        .collect::<Array1<_>>();
    let density = (1..=251)
        .map(|index| {
            let radius = radii[index - 1];
            (0.015 * index as Real + 0.002 * (index % 5) as Real) * radius * radius
        })
        .collect::<Array1<_>>();

    let result = coulomb_potential_slw(CoulombPotentialSlwInput {
        density: density.view(),
        radii: radii.view(),
        delta: 0.05,
        active_len: 251,
    })?;

    assert_eq!(result.active_len, 251);
    assert_close(result.potential[0], 2.668_218_180_150_684e3);
    assert_close(result.potential[1], 2.668_218_180_150_706e3);
    assert_close(result.potential[2], 2.668_218_180_150_729e3);
    assert_close(result.potential[63], 2.668_218_178_677_377_6e3);
    assert_close(result.potential[127], 2.668_216_102_613_817_4e3);
    assert_close(result.potential[250], 1.715_191_594_675_573_5e3);
    Ok(())
}

#[test]
fn coulomb_potential_slw_matches_feff_potslw_short_reference() -> Result<(), GridError> {
    let radii = (1..=251)
        .map(|index| 0.2 + 0.04 * index as Real)
        .collect::<Array1<_>>();
    let mut density = (1..=251)
        .map(|index| 0.03 * index as Real + 0.001 * (index * index) as Real)
        .collect::<Array1<_>>();
    density[8] = Real::NAN;

    let result = coulomb_potential_slw(CoulombPotentialSlwInput {
        density: density.view(),
        radii: radii.view(),
        delta: 0.07,
        active_len: 8,
    })?;

    let expected = [
        5.611_527_327_297_855e-2,
        5.254_100_056_797_561_5e-2,
        4.981_230_728_432_756e-2,
        4.749_189_715_919_542_6e-2,
        4.522_151_874_857_458e-2,
        4.271_756_035_420_61e-2,
        3.967_487_563_606_690_6e-2,
        3.662_296_212_560_022e-2,
    ];
    for (actual, expected) in result.potential.iter().take(8).zip(expected) {
        assert_close(*actual, expected);
    }
    assert_eq!(result.potential[8], 0.0);
    assert_eq!(result.potential[250], 0.0);
    Ok(())
}

#[test]
fn coulomb_potential_slw_rejects_invalid_inputs() {
    let density = Array1::<Real>::zeros(4);
    let radii = Array1::<Real>::ones(4);
    let short_radii = Array1::<Real>::ones(3);
    assert!(matches!(
        coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: short_radii.view(),
            delta: 0.05,
            active_len: 3,
        }),
        Err(GridError::CoulombLengthMismatch {
            density_len: 4,
            radii_len: 3,
        })
    ));
    assert!(matches!(
        coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: radii.view(),
            delta: 0.05,
            active_len: 2,
        }),
        Err(GridError::SourceGridTooShort {
            name: "active",
            required: 3,
            available: 2,
        })
    ));
    assert!(matches!(
        coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: radii.view(),
            delta: 0.05,
            active_len: 5,
        }),
        Err(GridError::SourceGridTooShort {
            name: "density",
            required: 5,
            available: 4,
        })
    ));

    let invalid_radii = Array1::from_vec(vec![1.0, 1.1, 0.0, 1.3]);
    assert!(matches!(
        coulomb_potential_slw(CoulombPotentialSlwInput {
            density: density.view(),
            radii: invalid_radii.view(),
            delta: 0.05,
            active_len: 4,
        }),
        Err(GridError::InvalidRadius { radius: 0.0 })
    ));
}

#[test]
fn scmt_energy_grid_matches_feff_grids_reference() -> Result<(), GridError> {
    let result = scmt_energy_grid(ScmtEnergyGridInput {
        core_valence_energy: -0.50,
        fermi_energy: 0.20,
        max_points: 120,
        step_count: 9,
    })?;

    assert_eq!(result.active_len, 74);
    assert_eq!(result.lower_imaginary_count, 5);
    assert_eq!(result.real_axis_count, 61);
    assert_eq!(result.upper_imaginary_count, 8);
    assert_energy(&result, 1, -0.5, 1.837_465_450_137_141e-3);
    assert_energy(&result, 2, -0.5, 7.349_861_800_548_564e-3);
    assert_energy(&result, 3, -0.5, 1.653_718_905_123_426_8e-2);
    assert_energy(&result, 4, -0.5, 2.939_944_720_219_425_7e-2);
    assert_energy(&result, 5, -0.5, 4.593_663_625_342_852e-2);
    assert_energy(
        &result,
        37,
        -1.327_868_852_459_011_8e-1,
        4.593_663_625_342_852e-2,
    );
    assert_energy(&result, 72, 0.2, 7.349_861_800_548_564e-3);
    assert_energy(&result, 73, 0.2, 4.134_297_262_808_567e-3);
    assert_energy(&result, 74, 0.2, 1.837_465_450_137_141e-3);
    assert_eq!(result.energies[74], Complex::new(0.0, 0.0));
    assert_step(&result, 1, 4.593_663_625_342_853e-4);
    assert_step(&result, 5, 4.134_297_262_808_567e-3);
    assert_step(&result, 9, 1.148_415_906_335_713_1e-2);
    Ok(())
}

#[test]
fn scmt_energy_grid_matches_feff_grids_clamped_reference() -> Result<(), GridError> {
    let result = scmt_energy_grid(ScmtEnergyGridInput {
        core_valence_energy: -0.20,
        fermi_energy: 20.00,
        max_points: 42,
        step_count: 8,
    })?;

    assert_eq!(result.active_len, 42);
    assert_eq!(result.lower_imaginary_count, 4);
    assert_eq!(result.real_axis_count, 31);
    assert_eq!(result.upper_imaginary_count, 7);
    assert_energy(&result, 1, -0.2, 1.837_465_450_137_141e-3);
    assert_energy(&result, 4, -0.2, 2.939_944_720_219_425_7e-2);
    assert_energy(
        &result,
        5,
        4.516_129_032_258_064_4e-1,
        2.939_944_720_219_425_7e-2,
    );
    assert_energy(
        &result,
        21,
        1.087_741_935_483_871_1e1,
        2.939_944_720_219_425_7e-2,
    );
    assert_energy(&result, 40, 20.0, 7.349_861_800_548_564e-3);
    assert_energy(&result, 41, 20.0, 4.134_297_262_808_567e-3);
    assert_energy(&result, 42, 20.0, 1.837_465_450_137_141e-3);
    assert_step(&result, 1, 4.593_663_625_342_853e-4);
    assert_step(&result, 7, 7.349_861_800_548_564e-3);
    assert_step(&result, 8, 7.349_861_800_548_564e-3);
    Ok(())
}

#[test]
fn scmt_energy_grid_rejects_invalid_inputs() {
    assert!(matches!(
        scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: f64::NAN,
            fermi_energy: 0.2,
            max_points: 120,
            step_count: 9,
        }),
        Err(GridError::NonFiniteScalar {
            name: "core_valence_energy",
            ..
        })
    ));
    assert_eq!(
        scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.5,
            fermi_energy: 0.2,
            max_points: 120,
            step_count: 0,
        }),
        Err(GridError::InvalidGridLength { name: "step" })
    );
    assert_eq!(
        scmt_energy_grid(ScmtEnergyGridInput {
            core_valence_energy: -0.5,
            fermi_energy: 0.2,
            max_points: 14,
            step_count: 8,
        }),
        Err(GridError::OutputGridTooShort {
            required: 15,
            available: 14,
        })
    );
}
