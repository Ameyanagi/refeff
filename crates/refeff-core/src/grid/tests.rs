use super::*;
use ndarray::{Array1, Array2, ShapeBuilder};

#[test]
fn converts_energy_to_signed_wave_number() {
    assert_eq!(wave_number_from_hartree(2.0), 2.0);
    assert_eq!(wave_number_from_hartree(-2.0), -2.0);
    assert_eq!(wave_number_from_hartree(0.0), 0.0);
}

#[test]
fn reproduces_loucks_log_grid_points() {
    assert!((loucks_x(1) + 8.8).abs() < 1.0e-12);
    assert!((loucks_x(2) + 8.75).abs() < 1.0e-12);
    assert!((loucks_radius(1) - (-8.8_f64).exp()).abs() < 1.0e-16);
}

#[test]
fn maps_radius_to_index_below() -> Result<(), GridError> {
    let radius = loucks_radius(42);
    assert_eq!(loucks_index_below(radius)?, 42);

    let midpoint = (loucks_x(42) + 0.5 * LOUCKS_DELTA).exp();
    assert_eq!(loucks_index_below(midpoint)?, 42);
    Ok(())
}

#[test]
fn rejects_invalid_radius_or_delta() {
    assert!(matches!(
        loucks_index_below(0.0),
        Err(GridError::InvalidRadius { .. })
    ));
    assert!(matches!(
        radial_index_below(1.0, 0.0),
        Err(GridError::InvalidDelta { .. })
    ));
}

#[test]
fn fix_dirac_spinor_grid_matches_feff_fixdsp_reference() -> Result<(), GridError> {
    let mut large = vec![0.0; 251];
    let mut small = vec![0.0; 251];
    for i in 1..=80 {
        let i_real = i as Real;
        large[i - 1] = (0.1 * i_real).sin() * (-0.02 * i_real).exp() + 0.001 * i_real;
        small[i - 1] = (0.08 * i_real).cos() * (-0.015 * i_real).exp() - 0.0005 * i_real;
    }
    let large = Array1::from_vec(large);
    let small = Array1::from_vec(small);

    let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_component: large.view(),
        small_component: small.view(),
        output_len: 180,
    })?;

    assert_eq!(result.active_len, 161);
    assert_spinor_value(
        &result,
        1,
        0.098_856_582_548_901_49,
        0.981_461_262_295_415_9,
    );
    assert_spinor_value(&result, 2, 0.146_525_001_614_189, 0.969_970_868_040_543_4);
    assert_spinor_value(
        &result,
        3,
        0.192_879_394_911_354_22,
        0.957_050_307_749_104_5,
    );
    assert_spinor_value(&result, 10, 0.473_738_853_193_487_96, 0.830_355_320_320_026);
    assert_spinor_value(
        &result,
        80,
        -0.310_280_702_093_608_3,
        -0.562_325_207_440_241_6,
    );
    assert_spinor_value(
        &result,
        120,
        -0.008_407_166_503_866_128,
        0.021_105_137_955_943_806,
    );
    assert_spinor_value(
        &result,
        160,
        0.191_266_534_139_204_64,
        0.176_750_359_590_577_94,
    );
    assert_spinor_value(&result, 161, 0.0, 0.0);
    assert_spinor_value(&result, 180, 0.0, 0.0);
    Ok(())
}

#[test]
fn fix_dirac_spinor_grid_zero_fills_empty_spinor() -> Result<(), GridError> {
    let large = Array1::zeros(251);
    let small = Array1::zeros(251);

    let result = fix_dirac_spinor_grid(DiracSpinorGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_component: large.view(),
        small_component: small.view(),
        output_len: 16,
    })?;

    assert_eq!(result.active_len, 0);
    assert!(result.large_component.iter().all(|&value| value == 0.0));
    assert!(result.small_component.iter().all(|&value| value == 0.0));
    Ok(())
}

#[test]
fn fix_dirac_spinor_grid_rejects_invalid_inputs() {
    let large = Array1::zeros(4);
    let small = Array1::zeros(3);
    assert_eq!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: large.view(),
            small_component: small.view(),
            output_len: 16,
        }),
        Err(GridError::SpinorLengthMismatch {
            large_len: 4,
            small_len: 3,
        })
    );

    let nonfinite = Array1::from_vec(vec![0.0, f64::NAN, 0.0, 0.0]);
    let zeros = Array1::zeros(4);
    assert!(matches!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_component: nonfinite.view(),
            small_component: zeros.view(),
            output_len: 16,
        }),
        Err(GridError::NonFiniteGridValue {
            name: "large_component",
            index: 1,
            ..
        })
    ));

    assert_eq!(
        fix_dirac_spinor_grid(DiracSpinorGridInput {
            original_delta: 0.0,
            new_delta: 0.025,
            large_component: zeros.view(),
            small_component: zeros.view(),
            output_len: 16,
        }),
        Err(GridError::InvalidDelta { delta: 0.0 })
    );
}

#[test]
fn fix_dirac_spinor_orbitals_grid_matches_feff_fixdsx_reference() -> Result<(), GridError> {
    let mut large = Array2::<Real>::zeros((251, 4).f());
    let mut small = Array2::<Real>::zeros((251, 4).f());
    for i in 1..=40 {
        let i_real = i as Real;
        large[(i - 1, 0)] = (0.07 * i_real).sin() * (-0.01 * i_real).exp();
        small[(i - 1, 0)] = (0.05 * i_real).cos() * (-0.02 * i_real).exp();
    }
    for i in 1..=75 {
        let i_real = i as Real;
        large[(i - 1, 2)] = 0.2 * (0.11 * i_real).sin() + 0.002 * i_real;
        small[(i - 1, 2)] = 0.3 * (0.09 * i_real).cos() - 0.001 * i_real;
    }
    for i in 1..=5 {
        let i_real = i as Real;
        large[(i - 1, 3)] = 0.05 * i_real;
        small[(i - 1, 3)] = -0.04 * i_real;
    }

    let result = fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
        original_delta: 0.05,
        new_delta: 0.025,
        large_components: large.view(),
        small_components: small.view(),
        output_len: 260,
    })?;

    assert_eq!(result.large_components.shape(), &[260, 4]);
    assert_eq!(result.large_components.strides(), &[1, 260]);
    assert_eq!(result.active_lengths.to_vec(), vec![81, 0, 151, 11]);
    assert_orbital_value(
        &result,
        1,
        1,
        0.069_246_904_378_467_77,
        0.978_973_680_203_922_3,
    );
    assert_orbital_value(&result, 81, 1, 0.0, 0.0);
    assert_orbital_value(&result, 82, 1, 0.0, 0.0);
    assert_orbital_value(&result, 1, 2, 0.0, 0.0);
    assert_orbital_value(&result, 100, 2, 0.0, 0.0);
    assert_orbital_value(
        &result,
        1,
        3,
        0.023_955_660_167_434_965,
        0.297_785_819_903_598_26,
    );
    assert_orbital_value(
        &result,
        150,
        3,
        0.228_834_221_332_933_4,
        0.130_219_461_349_623_98,
    );
    assert_orbital_value(&result, 151, 3, 0.0, 0.0);
    assert_orbital_value(&result, 1, 4, 0.05, -0.04);
    assert_orbital_value(&result, 11, 4, 0.0, 0.0);
    assert_orbital_value(&result, 12, 4, 0.0, 0.0);
    Ok(())
}

#[test]
fn fix_dirac_spinor_orbitals_grid_rejects_shape_mismatch() {
    let large = Array2::<Real>::zeros((4, 2));
    let small = Array2::<Real>::zeros((4, 3));

    assert_eq!(
        fix_dirac_spinor_orbitals_grid(DiracSpinorOrbitalsGridInput {
            original_delta: 0.05,
            new_delta: 0.025,
            large_components: large.view(),
            small_components: small.view(),
            output_len: 16,
        }),
        Err(GridError::SpinorShapeMismatch {
            large_rows: 4,
            large_columns: 2,
            small_rows: 4,
            small_columns: 3,
        })
    );
}

#[test]
fn fix_potential_grid_matches_feff_fixvar_nojump_reference() -> Result<(), GridError> {
    let result = run_sample_potential_grid(0, 0.125)?;

    assert_eq!(result.muffin_tin_index, 121);
    assert_eq!(result.interstitial_index, 122);
    assert_close(result.potential_jump, 0.125);
    assert_potential_value(
        &result,
        1,
        1.507_330_750_954_765e-4,
        -1.935_022_498_312_550_8,
        3.208_561_106_457_231e-2,
        6.991_469_396_917_269e-4,
    );
    assert_potential_value(
        &result,
        2,
        1.545_489_010_585_363e-4,
        -1.927_550_614_879_478_5,
        3.221_287_457_552_784e-2,
        1.047_124_998_462_492_8e-3,
    );
    assert_potential_value(
        &result,
        60,
        6.588_596_634_060_351e-4,
        -1.512_010_471_807_159,
        3.892_714_881_737_269e-2,
        3.404_343_790_471_498e-3,
    );
    assert_potential_value(
        &result,
        121,
        3.027_554_745_375_812_7e-3,
        -1.097_815_545_411_376,
        4.308_030_270_342_605e-2,
        -1.595_986_127_361_670_4e-2,
    );
    assert_potential_value(
        &result,
        122,
        3.104_197_658_649_308_7e-3,
        -1.091_039_022_689_942_5,
        4.312_310_486_424_463e-2,
        -1.593_525_191_056_929e-2,
    );
    assert_potential_value(
        &result,
        123,
        3.182_780_796_509_667e-3,
        -0.75,
        2.228_169_203_286_535e-2,
        0.0,
    );
    assert_potential_value(
        &result,
        180,
        1.323_355_009_654_092_8e-2,
        -0.75,
        2.228_169_203_286_535e-2,
        0.0,
    );
    Ok(())
}

#[test]
fn fix_potential_grid_matches_feff_fixvar_auto_jump_reference() -> Result<(), GridError> {
    let result = run_sample_potential_grid(1, 0.125)?;

    assert_close(result.potential_jump, 3.423_945_657_555_365e-1);
    assert_potential_value(
        &result,
        1,
        1.507_330_750_954_765e-4,
        -1.592_627_932_557_014_3,
        3.208_561_106_457_231e-2,
        6.991_469_396_917_269e-4,
    );
    assert_potential_value(
        &result,
        2,
        1.545_489_010_585_363e-4,
        -1.585_156_049_123_942,
        3.221_287_457_552_784e-2,
        1.047_124_998_462_492_8e-3,
    );
    assert_potential_value(
        &result,
        60,
        6.588_596_634_060_351e-4,
        -1.169_615_906_051_622_5,
        3.892_714_881_737_269e-2,
        3.404_343_790_471_498e-3,
    );
    assert_potential_value(
        &result,
        121,
        3.027_554_745_375_812_7e-3,
        -7.554_209_796_558_395e-1,
        4.308_030_270_342_605e-2,
        -1.595_986_127_361_670_4e-2,
    );
    assert_potential_value(
        &result,
        122,
        3.104_197_658_649_308_7e-3,
        -7.486_444_569_344_06e-1,
        4.312_310_486_424_463e-2,
        -1.593_525_191_056_929e-2,
    );
    assert_potential_value(
        &result,
        123,
        3.182_780_796_509_667e-3,
        -0.75,
        2.228_169_203_286_535e-2,
        0.0,
    );
    assert_potential_value(
        &result,
        180,
        1.323_355_009_654_092_8e-2,
        -0.75,
        2.228_169_203_286_535e-2,
        0.0,
    );
    Ok(())
}

#[test]
fn fix_potential_grid_rejects_invalid_inputs() {
    let density = Array1::<Real>::zeros(4);
    let potential = Array1::<Real>::zeros(5);
    let magnetization = Array1::<Real>::zeros(4);
    assert!(matches!(
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: radial_radius(2, 0.05),
            electron_density: density.view(),
            total_potential: potential.view(),
            magnetization: magnetization.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode: 0,
            potential_jump: 0.125,
            output_len: 8,
        }),
        Err(GridError::PotentialLengthMismatch {
            density_len: 4,
            potential_len: 5,
            magnetization_len: 4,
        })
    ));

    assert!(matches!(
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: radial_radius(2, 0.05),
            electron_density: density.view(),
            total_potential: density.view(),
            magnetization: density.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode: 0,
            potential_jump: f64::NAN,
            output_len: 8,
        }),
        Err(GridError::NonFiniteScalar {
            name: "potential_jump",
            ..
        })
    ));

    let nonfinite_density = Array1::from_vec(vec![0.0, f64::INFINITY, 0.0, 0.0]);
    assert!(matches!(
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: radial_radius(2, 0.05),
            electron_density: nonfinite_density.view(),
            total_potential: density.view(),
            magnetization: density.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode: 0,
            potential_jump: 0.125,
            output_len: 8,
        }),
        Err(GridError::NonFiniteGridValue {
            name: "electron_density",
            index: 1,
            ..
        })
    ));

    assert_eq!(
        fix_potential_grid(PotentialGridInput {
            muffin_tin_radius: radial_radius(6, 0.05),
            electron_density: density.view(),
            total_potential: density.view(),
            magnetization: density.view(),
            interstitial_potential: -0.75,
            interstitial_density: 0.28,
            original_delta: 0.05,
            new_delta: 0.025,
            jump_mode: 0,
            potential_jump: 0.125,
            output_len: 16,
        }),
        Err(GridError::SourceGridTooShort {
            name: "total_potential",
            required: 7,
            available: 4,
        })
    );
}

#[test]
fn fix_atomic_quantities_grid_matches_feff_scfdat_reference() -> Result<(), GridError> {
    let sample = sample_atomic_quantities();
    let result = fix_atomic_quantities_grid(sample.input())?;

    assert_eq!(result.large_components.shape(), &[251, 41]);
    assert_eq!(result.large_components.strides(), &[1, 251]);
    assert_atomic_quantity_value(
        &result,
        1,
        [
            2.791_223_374_813_335e-1,
            4.476_972_310_016_830_7e-1,
            -6.927_869_216_005_332e-2,
            7.429_002_860_424_964e-2,
            5.975_952_039_958_033e-3,
            -3.950_027_944_054_473_5e-3,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        2,
        [
            3.182_786_671_366_479e-1,
            5.455_503_623_374_722e-1,
            -1.036_498_625_688_917_2e-1,
            9.194_830_208_194_059e-2,
            8.967_257_481_430_235e-3,
            -5.902_243_782_988_271e-3,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        7,
        [
            5.124_030_373_249_69e-1,
            1.033_378_683_971_634_8,
            -2.762_683_654_671_763_7e-1,
            1.459_496_267_664_38e-1,
            2.421_247_561_760_581e-2,
            -1.560_559_355_284_406_3e-2,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        13,
        [
            7.381_902_178_641_675e-1,
            1.615_051_935_363_908_6,
            -4.863_362_194_530_379e-1,
            1.934_092_324_774_363_8e-1,
            4.312_221_991_984_726e-2,
            -2.711_094_420_196_304e-2,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        51,
        [
            1.709_145_907_021_589_3,
            5.230_653_512_208_42,
            -1.984_144_943_656_441_1,
            3.590_573_514_219_042_6e-1,
            1.790_021_071_432_657_5e-1,
            -9.679_447_371_383_787e-2,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        127,
        [
            8.710_325_782_441_283e-1,
            1.234_932_656_246_927_8e1,
            -5.021_214_730_612_967,
            5.605_615_949_265_137e-1,
            5.340_648_757_062_982e-1,
            -2.195_164_573_819_312_6e-1,
        ],
    );
    assert_atomic_quantity_value(
        &result,
        251,
        [
            3.574_883_608_998_099_8,
            2.476_483_338_260_353_8e1,
            -9.904_288_186_993_792,
            7.900_999_927_015_696e-1,
            1.351_727_616_897_123,
            -3.720_632_469_425_549e-1,
        ],
    );

    assert_atomic_orbital_quantity_value(
        &result,
        1,
        [
            2.574_628_049_407_745_5e-3,
            1.359_765_371_929_873_8e-2,
            6.932_165_062_956_984e-3,
            1.077_554_283_628_114_8e-2,
            8.946_972_105_857_141e-2,
            -4.878_878_770_458_183e-2,
        ],
    );
    assert_atomic_orbital_quantity_value(
        &result,
        13,
        [
            1.668_258_309_461_783_5e-2,
            5.112_013_023_337_827e-3,
            4.456_638_917_632_386e-2,
            -1.428_226_000_723_848_6e-2,
            5.739_297_089_585_34e-1,
            -3.883_217_851_300_009e-1,
        ],
    );
    assert_atomic_orbital_quantity_value(
        &result,
        251,
        [
            2.593_551_064_852_843e-1,
            -1.877_818_289_985_89e-1,
            7.531_974_620_885_585e-1,
            -5.337_865_926_961_277e-1,
            1.013_544_901_980_431e1,
            -7.101_575_507_219_226,
        ],
    );
    Ok(())
}

#[test]
fn fix_atomic_quantities_grid_rejects_invalid_inputs() {
    let sample = sample_atomic_quantities();
    let short = Array1::<Real>::zeros(250);
    assert!(matches!(
        fix_atomic_quantities_grid(AtomicQuantitiesGridInput {
            initial_small_component: short.view(),
            ..sample.input()
        }),
        Err(GridError::AtomicQuantitiesLengthMismatch { small_len: 250, .. })
    ));

    let bad_radii = Array1::from_vec(vec![1.0, 1.1, 0.0, 1.3]);
    let zeros = Array1::<Real>::zeros(4);
    let matrix = Array2::<Real>::zeros((4, 1));
    assert_eq!(
        fix_atomic_quantities_grid(AtomicQuantitiesGridInput {
            source_radii: bad_radii.view(),
            coulomb_potential: zeros.view(),
            charge_density: zeros.view(),
            magnetization: zeros.view(),
            valence_density: zeros.view(),
            initial_large_component: zeros.view(),
            initial_small_component: zeros.view(),
            large_components: matrix.view(),
            small_components: matrix.view(),
            output_len: 4,
        }),
        Err(GridError::InvalidRadius { radius: 0.0 })
    );

    let bad_shape = Array2::<Real>::zeros((250, 41));
    assert!(matches!(
        fix_atomic_quantities_grid(AtomicQuantitiesGridInput {
            large_components: bad_shape.view(),
            small_components: bad_shape.view(),
            ..sample.input()
        }),
        Err(GridError::AtomicQuantitiesSpinorRowMismatch {
            radial_len: 251,
            rows: 250,
            columns: 41,
        })
    ));
}

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

#[test]
fn sum_loucks_spherical_overlap_matches_feff_sumax_wide_reference() -> Result<(), GridError> {
    let (source, base) = sample_sumax_grids();
    let result = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
        neighbor_distance: 2.35,
        multiplicity: 1.75,
        source: source.view(),
        accumulated: base.view(),
    })?;

    assert_eq!(result.active_len, 194);
    assert_overlap_value(
        &result,
        &base,
        1,
        1.745_028_012_500_681_4,
        1.735_031_657_279_253,
    );
    assert_overlap_value(
        &result,
        &base,
        2,
        1.745_017_080_247_046_8,
        1.735_031_656_704_451_3,
    );
    assert_overlap_value(
        &result,
        &base,
        10,
        1.744_669_358_295_808_8,
        1.735_031_649_332_149_8,
    );
    assert_overlap_value(
        &result,
        &base,
        97,
        1.726_426_742_568_854_9,
        1.735_092_022_832_586,
    );
    assert_overlap_value(
        &result,
        &base,
        193,
        1.768_292_002_760_516,
        1.763_509_941_444_896,
    );
    assert_overlap_value(
        &result,
        &base,
        194,
        1.772_250_425_997_878,
        1.767_233_009_588_076_4,
    );
    assert_overlap_value(&result, &base, 195, 5.249_114_029_620_047e-3, 0.0);
    assert_overlap_value(&result, &base, 250, 8.930_063_446_890_768e-3, 0.0);
    Ok(())
}

#[test]
fn sum_loucks_spherical_overlap_matches_feff_sumax_near_reference() -> Result<(), GridError> {
    let (source, base) = sample_sumax_grids();
    let result = sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
        neighbor_distance: 0.012,
        multiplicity: 0.60,
        source: source.view(),
        accumulated: base.view(),
    })?;

    assert_eq!(result.active_len, 88);
    assert_overlap_value(
        &result,
        &base,
        1,
        3.436_843_996_472_091_5e-1,
        3.336_880_444_257_808e-1,
    );
    assert_overlap_value(
        &result,
        &base,
        2,
        3.436_695_121_985_222e-1,
        3.336_840_886_559_266e-1,
    );
    assert_overlap_value(
        &result,
        &base,
        10,
        3.432_708_426_104_191e-1,
        3.336_331_336_467_602e-1,
    );
    assert_overlap_value(
        &result,
        &base,
        44,
        3.373_894_297_682_521_5e-1,
        3.336_542_711_118_750_7e-1,
    );
    assert_overlap_value(
        &result,
        &base,
        87,
        3.321_150_695_075_532_6e-1,
        3.391_350_820_293_816e-1,
    );
    assert_overlap_value(
        &result,
        &base,
        88,
        3.326_278_771_348_888e-1,
        3.398_375_950_972_270_5e-1,
    );
    assert_overlap_value(&result, &base, 89, -7.394_167_837_740_848e-3, 0.0);
    assert_overlap_value(&result, &base, 250, 8.930_063_446_890_768e-3, 0.0);
    Ok(())
}

#[test]
fn sum_loucks_spherical_overlap_rejects_invalid_inputs() {
    let source = Array1::<Real>::zeros(250);
    let accumulated = Array1::<Real>::zeros(249);
    assert_eq!(
        sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: 2.35,
            multiplicity: 1.0,
            source: source.view(),
            accumulated: accumulated.view(),
        }),
        Err(GridError::OverlapLengthMismatch {
            source_len: 250,
            accumulated_len: 249,
        })
    );

    let short = Array1::<Real>::zeros(16);
    assert!(matches!(
        sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: 2.35,
            multiplicity: 1.0,
            source: short.view(),
            accumulated: short.view(),
        }),
        Err(GridError::SourceGridTooShort { name: "source", .. })
    ));

    assert!(matches!(
        sum_loucks_spherical_overlap(LoucksSphericalOverlapInput {
            neighbor_distance: 2.35,
            multiplicity: f64::NAN,
            source: source.view(),
            accumulated: source.view(),
        }),
        Err(GridError::NonFiniteScalar {
            name: "multiplicity",
            ..
        })
    ));
}

#[test]
fn muffin_tin_overlap_matrix_matches_feff_movrlp_explicit_reference() -> Result<(), GridError> {
    let sample = sample_movrlp_state();
    let explicit = sample.explicit_overlaps();
    let result = muffin_tin_overlap_matrix(sample.input(&explicit))?;
    let factors = result.lu.factors();

    assert_eq!(result.active_order, 101);
    assert_close(result.interstitial_volume, 1.250_001_131_628_848e1);
    assert_close(result.radii[0], 1.507_330_750_954_765e-4);
    assert_close(result.radii[94], 1.657_267_540_176_123_7e-2);
    assert_close(result.radii[99], 2.127_973_643_837_715_8e-2);
    assert_eq!(
        [
            result.lu.pivots()[0],
            result.lu.pivots()[1],
            result.lu.pivots()[49],
            result.lu.pivots()[50],
            result.lu.pivots()[99],
            result.lu.pivots()[100],
            result.lu.pivots()[74],
            result.lu.pivots()[89],
        ],
        [1, 2, 50, 51, 100, 101, 75, 90]
    );

    assert_complex32_close(factors[(0, 0)], Complex32::new(1.0, 0.0));
    assert_complex32_close(factors[(0, 100)], Complex32::new(1.0e-2, 0.0));
    assert_complex32_close(factors[(100, 0)], Complex32::new(0.0, 0.0));
    assert_complex32_close(factors[(99, 99)], Complex32::new(9.738_406_5e-1, 0.0));
    assert_complex32_close(factors[(100, 100)], Complex32::new(8.354_477e-3, 0.0));
    assert_complex32_close(factors[(29, 98)], Complex32::new(-3.502_009_4e-2, 0.0));
    assert_complex32_close(factors[(29, 99)], Complex32::new(4.868_523_8e-2, 0.0));
    assert_complex32_close(factors[(34, 98)], Complex32::new(-2.731_694e-1, 0.0));
    assert_complex32_close(factors[(34, 99)], Complex32::new(4.531_623e-1, 0.0));
    Ok(())
}

#[test]
fn muffin_tin_overlap_matrix_rejects_invalid_inputs() {
    let sample = sample_movrlp_state();
    let explicit = sample.explicit_overlaps();
    let bad_indices = Array1::from_vec(vec![49, 100]);
    assert_eq!(
        muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
            muffin_tin_indices: bad_indices.view(),
            ..sample.input(&explicit)
        }),
        Err(GridError::MuffinTinIndexTooSmall {
            name: "muffin_tin_indices",
            potential: 0,
            minimum: MOVRLP_NOVP,
            index: 49,
        })
    );

    let bad_positions = Array2::<Real>::zeros((2, 2));
    assert_eq!(
        muffin_tin_overlap_matrix(MuffinTinOverlapMatrixInput {
            atom_positions: bad_positions.view(),
            ..sample.input(&explicit)
        }),
        Err(GridError::InvalidPositionShape {
            rows: 2,
            columns: 2,
        })
    );
}

#[test]
fn project_muffin_tin_overlap_matches_feff_ovp2mt_reference() -> Result<(), GridError> {
    let sample = sample_movrlp_state();
    let explicit = sample.explicit_overlaps();
    let overlap = muffin_tin_overlap_matrix(sample.input(&explicit))?;
    let values = sample_ovp2mt_values(overlap.radii.view());

    let estimated = project_muffin_tin_overlap(sample.projection_input(
        values.view(),
        overlap.radii.view(),
        &overlap,
        MuffinTinOverlapProjectionMode::PotentialEstimateInterstitial,
        0.0,
    ))?;
    assert_close_with_tolerance(estimated.interstitial_value, 3.529_647_445_678_711e-1, 1e-6);
    assert_close_with_tolerance(estimated.values[(45, 0)], 1.671_886_152_029_037_3e-1, 1e-6);
    assert_close_with_tolerance(estimated.values[(94, 0)], 2.667_137_837_409_973e-1, 1e-6);
    assert_close_with_tolerance(estimated.values[(95, 0)], 2.724_320_558_655_516_4e-1, 1e-6);
    assert_close_with_tolerance(
        estimated.values[(96, 0)],
        estimated.interstitial_value,
        1e-12,
    );
    assert_close_with_tolerance(estimated.values[(50, 1)], 2.770_467_555_522_918_6e-1, 1e-6);
    assert_close_with_tolerance(estimated.values[(99, 1)], 4.113_857_826_590_538e-1, 1e-6);
    assert_close_with_tolerance(estimated.values[(100, 1)], 4.158_367_065_544_319_5e-1, 1e-6);
    assert_close_with_tolerance(
        estimated.values[(101, 1)],
        estimated.interstitial_value,
        1e-12,
    );

    let fixed = project_muffin_tin_overlap(sample.projection_input(
        values.view(),
        overlap.radii.view(),
        &overlap,
        MuffinTinOverlapProjectionMode::PotentialFixedInterstitial,
        0.75,
    ))?;
    assert_close_with_tolerance(fixed.interstitial_value, 0.75, 1e-12);
    assert_close_with_tolerance(fixed.values[(45, 0)], 1.671_885_848_045_349e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(94, 0)], 4.144_923_090_934_753_4e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(95, 0)], 4.280_226_014_160_147e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(96, 0)], 0.75, 1e-12);
    assert_close_with_tolerance(fixed.values[(50, 1)], 2.770_467_400_550_842_3e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(99, 1)], 4.414_542_019_367_218e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(100, 1)], 4.492_451_647_553_490_4e-1, 1e-6);
    assert_close_with_tolerance(fixed.values[(101, 1)], 0.75, 1e-12);

    let density = project_muffin_tin_overlap(sample.projection_input(
        values.view(),
        overlap.radii.view(),
        &overlap,
        MuffinTinOverlapProjectionMode::Density { total_charge: 22.5 },
        -99.0,
    ))?;
    assert_close_with_tolerance(density.interstitial_value, 2.249_999_617_054_582e1, 1e-6);
    assert_close_with_tolerance(density.values[(45, 0)], values[(45, 0)], 1e-12);
    assert_close_with_tolerance(density.values[(99, 1)], values[(99, 1)], 1e-12);
    Ok(())
}

#[test]
fn project_muffin_tin_overlap_rejects_invalid_inputs() -> Result<(), GridError> {
    let sample = sample_movrlp_state();
    let explicit = sample.explicit_overlaps();
    let overlap = muffin_tin_overlap_matrix(sample.input(&explicit))?;
    let values = sample_ovp2mt_values(overlap.radii.view());
    let short_values = Array2::<Real>::zeros((250, 2));

    assert_eq!(
        project_muffin_tin_overlap(MuffinTinOverlapProjectionInput {
            values: short_values.view(),
            ..sample.projection_input(
                values.view(),
                overlap.radii.view(),
                &overlap,
                MuffinTinOverlapProjectionMode::PotentialFixedInterstitial,
                0.0,
            )
        }),
        Err(GridError::ShapeTooSmall {
            name: "values",
            rows: 250,
            columns: 2,
            required_rows: 251,
            required_columns: 2,
        })
    );
    Ok(())
}

#[test]
fn sphere_overlap_volumes_match_feff_calcvl_reference() -> Result<(), GridError> {
    let cases = [
        (
            1.25,
            0.95,
            1.10,
            5.612_978_874_413_764e-1,
            1.664_520_507_626_991_6,
        ),
        (
            2.40,
            1.70,
            2.15,
            2.962_352_981_526_981,
            9.622_705_147_348_121,
        ),
        (
            0.80,
            1.60,
            1.25,
            1.356_786_629_672_262_6,
            1.562_880_822_304_789_6,
        ),
        (3.10, 2.90, 4.80, 3.020_854_048_429_17, 6.324_026_011_676_25),
    ];

    for (radius_a, radius_b, distance, expected_cap, expected_lens) in cases {
        assert_close(
            sphere_overlap_cap_volume(radius_a, radius_b, distance)?,
            expected_cap,
        );
        assert_close(
            sphere_overlap_lens_volume(radius_a, radius_b, distance)?,
            expected_lens,
        );
    }
    Ok(())
}

#[test]
fn sphere_overlap_volumes_reject_invalid_inputs() {
    assert_eq!(
        sphere_overlap_cap_volume(0.0, 1.0, 1.0),
        Err(GridError::NonPositiveScalar {
            name: "sphere_radius",
            value: 0.0,
        })
    );
    assert!(matches!(
        sphere_overlap_lens_volume(1.0, Real::NAN, 1.0),
        Err(GridError::NonPositiveScalar {
            name: "other_radius",
            ..
        })
    ));
}

#[test]
fn interstitial_shell_values_match_feff_istval_wide_reference() -> Result<(), GridError> {
    let (potential, density) = sample_istval_grids();
    let muffin_tin_radius = (loucks_x(45) + 0.021).exp();
    let wigner_seitz_radius = (loucks_x(116) + 0.034).exp();
    let muffin_tin_index = loucks_index_below(muffin_tin_radius)?;
    let wigner_seitz_index = loucks_index_below(wigner_seitz_radius)?;

    assert_eq!(muffin_tin_index, 45);
    assert_eq!(wigner_seitz_index, 116);
    let result = interstitial_shell_values(InterstitialShellValuesInput {
        total_potential: potential.view(),
        overlapped_density: density.view(),
        muffin_tin_radius,
        muffin_tin_index,
        wigner_seitz_radius,
        wigner_seitz_index,
    })?;

    assert_interstitial_values(
        result,
        -1.294_131_834_592_241_2,
        8.430_358_921_763_391e-1,
        3.920_777_855_274_227_4e-5,
    );
    Ok(())
}

#[test]
fn interstitial_shell_values_match_feff_istval_tight_reference() -> Result<(), GridError> {
    let (potential, density) = sample_istval_grids();
    let muffin_tin_radius = (loucks_x(70) + 0.010).exp();
    let wigner_seitz_radius = (loucks_x(70) + 0.037).exp();
    let muffin_tin_index = loucks_index_below(muffin_tin_radius)?;
    let wigner_seitz_index = loucks_index_below(wigner_seitz_radius)?;

    assert_eq!(muffin_tin_index, 70);
    assert_eq!(wigner_seitz_index, 70);
    let result = interstitial_shell_values(InterstitialShellValuesInput {
        total_potential: potential.view(),
        overlapped_density: density.view(),
        muffin_tin_radius,
        muffin_tin_index,
        wigner_seitz_radius,
        wigner_seitz_index,
    })?;

    assert_interstitial_values(
        result,
        -1.347_852_330_921_851,
        7.333_517_443_187_345e-1,
        3.102_227_388_939_98e-9,
    );
    Ok(())
}

#[test]
fn interstitial_shell_values_rejects_invalid_inputs() {
    let values = Array1::<Real>::zeros(8);
    assert_eq!(
        interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: values.view(),
            overlapped_density: values.view(),
            muffin_tin_radius: loucks_radius(4),
            muffin_tin_index: 4,
            wigner_seitz_radius: loucks_radius(4),
            wigner_seitz_index: 4,
        }),
        Err(GridError::InvalidRadiusOrder {
            inner_radius: loucks_radius(4),
            outer_radius: loucks_radius(4),
        })
    );

    assert_eq!(
        interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: values.view(),
            overlapped_density: values.view(),
            muffin_tin_radius: loucks_radius(4),
            muffin_tin_index: 0,
            wigner_seitz_radius: loucks_radius(5),
            wigner_seitz_index: 5,
        }),
        Err(GridError::InvalidGridIndex {
            name: "muffin_tin",
            index: 0,
        })
    );

    assert_eq!(
        interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: values.view(),
            overlapped_density: values.view(),
            muffin_tin_radius: loucks_radius(6),
            muffin_tin_index: 6,
            wigner_seitz_radius: loucks_radius(7),
            wigner_seitz_index: 5,
        }),
        Err(GridError::InvalidGridIndexRange {
            lower_index: 6,
            upper_index: 5,
        })
    );

    let short = Array1::<Real>::zeros(4);
    assert_eq!(
        interstitial_shell_values(InterstitialShellValuesInput {
            total_potential: short.view(),
            overlapped_density: short.view(),
            muffin_tin_radius: loucks_radius(3),
            muffin_tin_index: 3,
            wigner_seitz_radius: loucks_radius(4),
            wigner_seitz_index: 4,
        }),
        Err(GridError::SourceGridTooShort {
            name: "total_potential",
            required: 5,
            available: 4,
        })
    );
}

#[test]
fn overlap_density_indices_match_feff_sidx_keep_reference() -> Result<(), GridError> {
    let density = sample_sidx_keep_density();
    let muffin_tin_radius = (feff_legacy_loucks_x(30) + 0.020).exp();
    let norman_radius = (feff_legacy_loucks_x(90) + 0.030).exp();

    let result = overlap_density_indices(OverlapDensityIndicesInput {
        overlapped_density: density.view(),
        muffin_tin_radius,
        norman_radius,
    })?;

    assert_eq!(result.max_density_index, 250);
    assert_eq!(result.muffin_tin_index, 30);
    assert_eq!(result.norman_index, 90);
    assert!(!result.moved_norman_radius);
    assert_close(result.muffin_tin_radius, 6.555_734_762_497_377e-4);
    assert_close(result.norman_radius, 1.329_988_188_760_991_2e-2);
    Ok(())
}

#[test]
fn overlap_density_indices_match_feff_sidx_move_norman_reference() -> Result<(), GridError> {
    let density = sample_sidx_cutoff_density();
    let muffin_tin_radius = (feff_legacy_loucks_x(30) + 0.020).exp();
    let norman_radius = (feff_legacy_loucks_x(130) + 0.010).exp();

    let result = overlap_density_indices(OverlapDensityIndicesInput {
        overlapped_density: density.view(),
        muffin_tin_radius,
        norman_radius,
    })?;

    assert_eq!(result.max_density_index, 92);
    assert_eq!(result.muffin_tin_index, 30);
    assert_eq!(result.norman_index, 92);
    assert!(result.moved_norman_radius);
    assert_close(result.muffin_tin_radius, 6.555_734_762_497_377e-4);
    assert_close(result.norman_radius, 1.426_423_215_543_176_1e-2);
    Ok(())
}

#[test]
fn overlap_density_indices_rejects_invalid_inputs() {
    let density = Array1::<Real>::from_elem(8, 0.1);
    assert_eq!(
        overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: density.view(),
            muffin_tin_radius: 0.0,
            norman_radius: loucks_radius(4),
        }),
        Err(GridError::InvalidRadius { radius: 0.0 })
    );

    assert_eq!(
        overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: density.view(),
            muffin_tin_radius: loucks_radius(9),
            norman_radius: loucks_radius(10),
        }),
        Err(GridError::SourceGridTooShort {
            name: "overlapped_density",
            required: 9,
            available: 8,
        })
    );

    let zero_tail = Array1::<Real>::from_elem(16, 1.0e-6);
    assert_eq!(
        overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: zero_tail.view(),
            muffin_tin_radius: loucks_radius(4),
            norman_radius: loucks_radius(8),
        }),
        Err(GridError::NoActiveDensityTail {
            start_index: 4,
            threshold: SIDX_DENSITY_CUTOFF,
        })
    );

    let mut nonfinite = Array1::<Real>::from_elem(16, 0.1);
    nonfinite[2] = Real::NAN;
    assert!(matches!(
        overlap_density_indices(OverlapDensityIndicesInput {
            overlapped_density: nonfinite.view(),
            muffin_tin_radius: loucks_radius(4),
            norman_radius: loucks_radius(8),
        }),
        Err(GridError::NonFiniteGridValue {
            name: "overlapped_density",
            index: 2,
            ..
        })
    ));
}

#[test]
fn norman_radius_matches_feff_frnrm_oxygen_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_oxygen_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 8,
    })?;

    assert_close(result.radius, 1.063_980_446_859_560_2);
    Ok(())
}

#[test]
fn norman_radius_matches_feff_frnrm_iron_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_iron_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 26,
    })?;

    assert_close(result.radius, 8.688_945_443_598_616e-1);
    Ok(())
}

#[test]
fn norman_radius_matches_feff_frnrm_gold_like_reference() -> Result<(), GridError> {
    let density = sample_frnrm_gold_density();

    let result = norman_radius_from_density(NormanRadiusInput {
        overlapped_density: density.view(),
        atomic_number: 79,
    })?;

    assert_close(result.radius, 6.973_687_583_509_427e-1);
    Ok(())
}

#[test]
fn norman_radius_rejects_invalid_inputs() {
    let density = Array1::<Real>::from_elem(FRNRM_DENSITY_POINTS, 1.0);
    assert_eq!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: density.view(),
            atomic_number: 0,
        }),
        Err(GridError::InvalidAtomicNumber { atomic_number: 0 })
    );

    let short_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS - 1);
    assert_eq!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: short_density.view(),
            atomic_number: 1,
        }),
        Err(GridError::SourceGridTooShort {
            name: "overlapped_density",
            required: FRNRM_DENSITY_POINTS,
            available: FRNRM_DENSITY_POINTS - 1,
        })
    );

    let zero_density = Array1::<Real>::zeros(FRNRM_DENSITY_POINTS);
    assert!(matches!(
        norman_radius_from_density(NormanRadiusInput {
            overlapped_density: zero_density.view(),
            atomic_number: 1,
        }),
        Err(GridError::InsufficientNormanCharge {
            atomic_number: 1,
            ..
        })
    ));
}

#[test]
fn interstitial_fermi_level_matches_feff_fermi_reference() -> Result<(), GridError> {
    let shell = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: 8.430_358_921_763_391e-1,
        interstitial_potential: -1.294_131_834_592_241_2,
    })?;
    assert_fermi_level(
        shell,
        -5.040_450_363_824_843e-1,
        1.526_716_490_479_997_5,
        1.257_049_560_049_051_4,
    );

    let dense = interstitial_fermi_level(FermiLevelInput {
        interstitial_density: 3.2,
        interstitial_potential: -0.42,
    })?;
    assert_fermi_level(
        dense,
        1.502_548_984_343_600_6,
        9.787_169_102_922_159e-1,
        1.960_892_135_913_447_2,
    );
    Ok(())
}

#[test]
fn interstitial_fermi_level_rejects_invalid_inputs() {
    assert_eq!(
        interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 0.0,
            interstitial_potential: -1.0,
        }),
        Err(GridError::NonPositiveScalar {
            name: "interstitial_density",
            value: 0.0,
        })
    );
    assert!(matches!(
        interstitial_fermi_level(FermiLevelInput {
            interstitial_density: 1.0,
            interstitial_potential: Real::NAN,
        }),
        Err(GridError::NonFiniteScalar {
            name: "interstitial_potential",
            ..
        })
    ));
}

fn assert_spinor_value(
    spinor: &DiracSpinorGrid,
    index_1based: usize,
    expected_large: Real,
    expected_small: Real,
) {
    let index = index_1based - 1;
    assert_close(spinor.large_component[index], expected_large);
    assert_close(spinor.small_component[index], expected_small);
}

fn assert_orbital_value(
    spinor: &DiracSpinorOrbitalsGrid,
    index_1based: usize,
    orbital_1based: usize,
    expected_large: Real,
    expected_small: Real,
) {
    let radial = index_1based - 1;
    let orbital = orbital_1based - 1;
    assert_close(spinor.large_components[(radial, orbital)], expected_large);
    assert_close(spinor.small_components[(radial, orbital)], expected_small);
}

fn assert_potential_value(
    grid: &PotentialGrid,
    index_1based: usize,
    expected_radius: Real,
    expected_potential: Real,
    expected_density: Real,
    expected_magnetization: Real,
) {
    let index = index_1based - 1;
    assert_close(grid.radii[index], expected_radius);
    assert_close(grid.total_potential[index], expected_potential);
    assert_close(grid.charge_density[index], expected_density);
    assert_close(grid.magnetization[index], expected_magnetization);
}

fn assert_atomic_quantity_value(
    grid: &AtomicQuantitiesGrid,
    index_1based: usize,
    expected: [Real; 6],
) {
    let index = index_1based - 1;
    assert_scfdat_fix_close(grid.coulomb_potential[index], expected[0]);
    assert_scfdat_fix_close(grid.charge_density[index], expected[1]);
    assert_scfdat_fix_close(grid.magnetization[index], expected[2]);
    assert_scfdat_fix_close(grid.valence_density[index], expected[3]);
    assert_scfdat_fix_close(grid.initial_large_component[index], expected[4]);
    assert_scfdat_fix_close(grid.initial_small_component[index], expected[5]);
}

fn assert_atomic_orbital_quantity_value(
    grid: &AtomicQuantitiesGrid,
    index_1based: usize,
    expected: [Real; 6],
) {
    let index = index_1based - 1;
    assert_scfdat_fix_close(grid.large_components[(index, 0)], expected[0]);
    assert_scfdat_fix_close(grid.small_components[(index, 0)], expected[1]);
    assert_scfdat_fix_close(grid.large_components[(index, 2)], expected[2]);
    assert_scfdat_fix_close(grid.small_components[(index, 2)], expected[3]);
    assert_scfdat_fix_close(grid.large_components[(index, 40)], expected[4]);
    assert_scfdat_fix_close(grid.small_components[(index, 40)], expected[5]);
}

fn assert_scfdat_fix_close(actual: Real, expected: Real) {
    assert_close_with_tolerance(actual, expected, 5.0e-7_f64.max(expected.abs() * 5.0e-7));
}

fn assert_energy(
    grid: &ScmtEnergyGrid,
    index_1based: usize,
    expected_real: Real,
    expected_imaginary: Real,
) {
    let value = grid.energies[index_1based - 1];
    assert_close(value.re, expected_real);
    assert_close(value.im, expected_imaginary);
}

fn assert_step(grid: &ScmtEnergyGrid, index_1based: usize, expected: Real) {
    assert_close(grid.steps[index_1based - 1], expected);
}

fn assert_overlap_value(
    overlap: &LoucksSphericalOverlap,
    base: &Array1<Real>,
    index_1based: usize,
    expected_total: Real,
    expected_contribution: Real,
) {
    let index = index_1based - 1;
    const SUMAX_ORACLE_TOLERANCE: Real = 5.0e-9;

    assert_close_with_tolerance(
        overlap.accumulated[index],
        expected_total,
        SUMAX_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        overlap.accumulated[index] - base[index],
        expected_contribution,
        SUMAX_ORACLE_TOLERANCE,
    );
}

fn assert_interstitial_values(
    values: InterstitialShellValues,
    expected_potential: Real,
    expected_density: Real,
    expected_volume: Real,
) {
    const ISTVAL_ORACLE_TOLERANCE: Real = 5.0e-10;

    assert_close_with_tolerance(
        values.interstitial_potential,
        expected_potential,
        ISTVAL_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        values.interstitial_density,
        expected_density,
        ISTVAL_ORACLE_TOLERANCE,
    );
    assert_close_with_tolerance(
        values.shell_volume,
        expected_volume,
        1.0e-15_f64.max(expected_volume.abs() * 5.0e-7),
    );
}

fn assert_fermi_level(
    value: FermiLevel,
    expected_chemical_potential: Real,
    expected_density_parameter: Real,
    expected_fermi_momentum: Real,
) {
    assert_close(value.chemical_potential, expected_chemical_potential);
    assert_close(value.density_parameter, expected_density_parameter);
    assert_close(value.fermi_momentum, expected_fermi_momentum);
}

fn run_sample_potential_grid(
    jump_mode: i32,
    potential_jump: Real,
) -> Result<PotentialGrid, GridError> {
    let (density, potential, magnetization) = sample_potential_sources();
    fix_potential_grid(PotentialGridInput {
        muffin_tin_radius: (-8.8 + 60.4 * 0.05_f64).exp(),
        electron_density: density.view(),
        total_potential: potential.view(),
        magnetization: magnetization.view(),
        interstitial_potential: -0.75,
        interstitial_density: 0.28,
        original_delta: 0.05,
        new_delta: 0.025,
        jump_mode,
        potential_jump,
        output_len: 180,
    })
}

fn sample_potential_sources() -> (Array1<Real>, Array1<Real>, Array1<Real>) {
    let source_len = 251;
    let density = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            0.4 + 0.002 * i + 0.03 * (0.04 * i).sin()
        })
        .collect::<Array1<_>>();
    let potential = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            -2.0 + 0.015 * i + 0.05 * (0.03 * i).cos()
        })
        .collect::<Array1<_>>();
    let magnetization = (1..=source_len)
        .map(|index| {
            let i = index as Real;
            0.01 * (0.08 * i).sin() - 0.0001 * i
        })
        .collect::<Array1<_>>();
    (density, potential, magnetization)
}

#[derive(Debug, Clone)]
struct AtomicQuantitiesSample {
    radii: Array1<Real>,
    coulomb_potential: Array1<Real>,
    charge_density: Array1<Real>,
    magnetization: Array1<Real>,
    valence_density: Array1<Real>,
    initial_large_component: Array1<Real>,
    initial_small_component: Array1<Real>,
    large_components: Array2<Real>,
    small_components: Array2<Real>,
}

impl AtomicQuantitiesSample {
    fn input(&self) -> AtomicQuantitiesGridInput<'_> {
        AtomicQuantitiesGridInput {
            source_radii: self.radii.view(),
            coulomb_potential: self.coulomb_potential.view(),
            charge_density: self.charge_density.view(),
            magnetization: self.magnetization.view(),
            valence_density: self.valence_density.view(),
            initial_large_component: self.initial_large_component.view(),
            initial_small_component: self.initial_small_component.view(),
            large_components: self.large_components.view(),
            small_components: self.small_components.view(),
            output_len: 251,
        }
    }
}

fn sample_atomic_quantities() -> AtomicQuantitiesSample {
    let source_len = 251;
    let radii = (1..=source_len)
        .map(|index| {
            (-8.85 + 0.051 * (index - 1) as Real + 1.0e-4 * (0.37 * index as Real).cos()).exp()
        })
        .collect::<Array1<_>>();
    let coulomb_potential = (1..=source_len)
        .map(|index| 0.2 + 0.01 * index as Real + (0.03 * index as Real).sin())
        .collect::<Array1<_>>();
    let charge_density = (1..=source_len)
        .map(|index| 0.1 * index as Real + 0.25 * (0.02 * index as Real).cos())
        .collect::<Array1<_>>();
    let magnetization = (1..=source_len)
        .map(|index| -0.04 * index as Real + 0.1 * (0.05 * index as Real).sin())
        .collect::<Array1<_>>();
    let valence_density = (1..=source_len)
        .map(|index| 0.05 * (index as Real).sqrt() + 0.002 * (index % 5) as Real)
        .collect::<Array1<_>>();
    let initial_large_component = (1..=source_len)
        .map(|index| 0.003 * index as Real + 1.0e-5 * (index * index) as Real)
        .collect::<Array1<_>>();
    let initial_small_component = (1..=source_len)
        .map(|index| -0.002 * index as Real + 2.0e-6 * (index * index) as Real)
        .collect::<Array1<_>>();
    let large_components = Array2::from_shape_fn((source_len, 41).f(), |(row, column)| {
        let i = (row + 1) as Real;
        let j = (column + 1) as Real;
        0.001 * i * j + 0.02 * (0.01 * (i + j)).sin()
    });
    let small_components = Array2::from_shape_fn((source_len, 41).f(), |(row, column)| {
        let i = (row + 1) as Real;
        let j = (column + 1) as Real;
        -0.0007 * i * j + 0.015 * (0.012 * (i + 2.0 * j)).cos()
    });
    AtomicQuantitiesSample {
        radii,
        coulomb_potential,
        charge_density,
        magnetization,
        valence_density,
        initial_large_component,
        initial_small_component,
        large_components,
        small_components,
    }
}

#[derive(Debug, Clone)]
struct MovrlpSample {
    atom_potentials: Array1<usize>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    potential_multiplicities: Array1<Real>,
    neighbors0: [MuffinTinOverlapNeighbor; 1],
    neighbors1: [MuffinTinOverlapNeighbor; 1],
    norman_indices: Array1<usize>,
    muffin_tin_indices: Array1<usize>,
    muffin_tin_radii: Array1<Real>,
    norman_radii: Array1<Real>,
    near_neighbor_flags: Array1<bool>,
}

impl MovrlpSample {
    fn explicit_overlaps(&self) -> [&[MuffinTinOverlapNeighbor]; 2] {
        [&self.neighbors0, &self.neighbors1]
    }

    fn input<'a>(
        &'a self,
        explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
    ) -> MuffinTinOverlapMatrixInput<'a> {
        MuffinTinOverlapMatrixInput {
            highest_potential_index: 1,
            atom_potentials: self.atom_potentials.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            potential_multiplicities: self.potential_multiplicities.view(),
            explicit_overlaps,
            muffin_tin_indices: self.muffin_tin_indices.view(),
            muffin_tin_radii: self.muffin_tin_radii.view(),
            norman_radii: self.norman_radii.view(),
            near_neighbor_flags: self.near_neighbor_flags.view(),
            interstitial_selector: 0,
            interstitial_volume: 12.5,
        }
    }

    fn projection_input<'a>(
        &'a self,
        values: ArrayView2<'a, Real>,
        radii: ArrayView1<'a, Real>,
        overlap_matrix: &'a MuffinTinOverlapMatrix,
        mode: MuffinTinOverlapProjectionMode,
        interstitial_value: Real,
    ) -> MuffinTinOverlapProjectionInput<'a> {
        MuffinTinOverlapProjectionInput {
            highest_potential_index: 1,
            values,
            radii,
            potential_multiplicities: self.potential_multiplicities.view(),
            norman_indices: self.norman_indices.view(),
            muffin_tin_indices: self.muffin_tin_indices.view(),
            muffin_tin_radii: self.muffin_tin_radii.view(),
            norman_radii: self.norman_radii.view(),
            near_neighbor_flags: self.near_neighbor_flags.view(),
            overlap_matrix,
            interstitial_selector: 0,
            interstitial_value,
            mode,
        }
    }
}

fn sample_movrlp_state() -> MovrlpSample {
    let atom_potentials = Array1::from_vec(vec![0, 1]);
    let atom_positions = Array2::<Real>::zeros((2, 3));
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let potential_multiplicities = Array1::from_vec(vec![1.0, 2.0]);
    let neighbors0 = [MuffinTinOverlapNeighbor {
        source_potential: 1,
        multiplicity: 2,
        distance: 0.030,
    }];
    let neighbors1 = [MuffinTinOverlapNeighbor {
        source_potential: 0,
        multiplicity: 1,
        distance: 0.031,
    }];
    let norman_indices = Array1::from_vec(vec![90, 92]);
    let muffin_tin_indices = Array1::from_vec(vec![95, 100]);
    let muffin_tin_radii = Array1::from_vec(vec![0.020, 0.024]);
    let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
    let near_neighbor_flags = Array1::from_vec(vec![false, false]);
    MovrlpSample {
        atom_potentials,
        atom_positions,
        representative_atoms,
        potential_multiplicities,
        neighbors0,
        neighbors1,
        norman_indices,
        muffin_tin_indices,
        muffin_tin_radii,
        norman_radii,
        near_neighbor_flags,
    }
}

fn sample_ovp2mt_values(radii: ArrayView1<'_, Real>) -> Array2<Real> {
    Array2::from_shape_fn((251, 2), |(radial, potential)| {
        let index = (radial + 1) as Real;
        0.1 * (potential + 1) as Real
            + 0.001 * index
            + 0.00001 * index * index
            + 0.02 * radii[radial]
    })
}

fn sample_sumax_grids() -> (Array1<Real>, Array1<Real>) {
    let len = 250;
    let source = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.2 + 0.004 * i + 0.03 * (0.035 * i).sin()
        })
        .collect::<Array1<_>>();
    let base = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.01 * (0.027 * i).cos()
        })
        .collect::<Array1<_>>();
    (source, base)
}

fn sample_istval_grids() -> (Array1<Real>, Array1<Real>) {
    let len = 1251;
    let potential = (1..=len)
        .map(|index| {
            let i = index as Real;
            -1.5 + 0.002 * i + 0.04 * (0.017 * i).cos()
        })
        .collect::<Array1<_>>();
    let density = (1..=len)
        .map(|index| {
            let i = index as Real;
            0.5 + 0.003 * i + 0.02 * (0.023 * i).sin()
        })
        .collect::<Array1<_>>();
    (potential, density)
}

fn sample_frnrm_oxygen_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            50.0 * (-1.2 * radius).exp() + 0.1 * (-0.05 * radius).exp()
        })
        .collect::<Array1<_>>()
}

fn sample_frnrm_iron_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            220.0 * (-0.85 * radius).exp() / (1.0 + 0.12 * radius)
        })
        .collect::<Array1<_>>()
}

fn sample_frnrm_gold_density() -> Array1<Real> {
    (1..=FRNRM_DENSITY_POINTS)
        .map(|index| {
            let radius = feff_legacy_loucks_radius(index);
            950.0 * (-0.55 * radius).exp() / (1.0 + 0.08 * radius * radius)
        })
        .collect::<Array1<_>>()
}

fn sample_sidx_keep_density() -> Array1<Real> {
    (1..=250)
        .map(|index| {
            let i = index as Real;
            0.08 + 0.0004 * i + 0.002 * (0.05 * i).sin()
        })
        .collect::<Array1<_>>()
}

fn sample_sidx_cutoff_density() -> Array1<Real> {
    (1..=250)
        .map(|index| {
            if index <= 92 {
                0.04 + 0.0002 * index as Real
            } else {
                1.0e-6
            }
        })
        .collect::<Array1<_>>()
}

fn assert_close(actual: Real, expected: Real) {
    let tolerance = 1.0e-12_f64.max(expected.abs() * 1.0e-12);
    assert_close_with_tolerance(actual, expected, tolerance);
}

fn assert_close_with_tolerance(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert_close_with_tolerance(actual.re as Real, expected.re as Real, 5.0e-6);
    assert_close_with_tolerance(actual.im as Real, expected.im as Real, 5.0e-6);
}
