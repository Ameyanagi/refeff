use super::{support::*, *};

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
