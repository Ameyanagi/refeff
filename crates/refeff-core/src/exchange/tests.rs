use super::*;

fn assert_real_close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() < 1.0e-14,
        "actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

fn assert_real_near(actual: Real, expected: Real, tolerance: Real) {
    assert!(
        (actual - expected).abs() < tolerance,
        "actual={actual}, expected={expected}, diff={}, tolerance={tolerance}",
        (actual - expected).abs()
    );
}

fn assert_complex_close(actual: Complex, expected: Complex) {
    assert_real_close(actual.re, expected.re);
    assert_real_close(actual.im, expected.im);
}

fn assert_complex_near(actual: Complex, expected: Complex, tolerance: Real) {
    assert_real_near(actual.re, expected.re, tolerance);
    assert_real_near(actual.im, expected.im, tolerance);
}

#[test]
fn dirac_hara_exchange_potential_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        dirac_hara_exchange_potential(2.0, 1.3)?,
        -0.127_197_685_551_846_32,
    );
    assert_eq!(dirac_hara_exchange_potential(150.0, 0.4)?, 0.0);
    Ok(())
}

#[test]
fn hedin_lundqvist_ffq_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        hedin_lundqvist_ffq(0.8, 0.42, 1.2, 0.7, 4.0 / 3.0)?,
        0.086_644_481_840_666_34,
    );
    assert_real_close(
        hedin_lundqvist_ffq(1.6, 1.8, 2.4, 0.35, 0.9)?,
        0.062_528_814_886_981_41,
    );
    Ok(())
}

#[test]
fn von_barth_hedin_potential_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        von_barth_hedin_potential(2.5, 1.2)?,
        -0.318_654_527_096_978_5,
    );
    assert_eq!(von_barth_hedin_potential(1200.0, 0.8)?, 0.0);
    Ok(())
}

#[test]
fn perdew_zunger_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(perdew_zunger_vxc(0.75)?, -0.888_417_338_140_178);
    assert_real_close(perdew_zunger_vxc(2.0)?, -0.357_256_470_778_624_55);
    assert_real_close(perdew_zunger_vxc(10.0)?, -0.083_694_351_354_168_7);

    let full = perdew_zunger_exchange_correlation(2.0)?;
    assert_real_close(full.potential, -0.357_256_470_778_624_55);
    Ok(())
}

#[test]
fn perrot_dharma_wardana_vxc_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        perrot_dharma_wardana_vxc(2.0, 0.0)?,
        -0.365_286_030_418_622_73,
    );
    assert_real_close(
        perrot_dharma_wardana_vxc(2.0, 0.05)?,
        -0.372_727_169_113_195_8,
    );
    assert_real_close(
        perrot_dharma_wardana_vxc(0.75, 0.12)?,
        -0.898_713_301_179_314_3,
    );
    assert_real_close(
        perrot_dharma_wardana_reduced_vxc(2.0, 0.5)?,
        -0.371_754_474_403_717,
    );
    assert_real_close(
        perrot_dharma_wardana_reduced_vxc(8.0, 4.0)?,
        -0.094_263_857_322_040_14,
    );
    Ok(())
}

#[test]
fn karasiev_sjostrom_dufty_trickey_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(2.0, 0.0)?,
        -0.356_646_002_509_705_63,
    );
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(2.0, 0.05)?,
        -0.359_324_656_674_432_86,
    );
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_vxc(0.75, 0.12)?,
        -0.887_463_405_178_325_2,
    );

    let zero = karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.0, KsdTSpin::Unpolarized)?;
    assert_real_close(zero.free_energy_per_particle, -0.273_783_738_220_404_75);
    assert_real_close(zero.potential, -0.356_646_002_509_705_63);

    let first = karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.5, KsdTSpin::Unpolarized)?;
    assert_real_close(first.free_energy_per_particle, -0.258_509_037_442_536_34);
    assert_real_close(first.potential, -0.351_894_120_250_991_87);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(2.0, 0.5, KsdTSpin::Unpolarized)?,
        -0.287_859_963_317_935_5,
    );

    let second = karasiev_sjostrom_dufty_trickey_free_energy(8.0, 4.0, KsdTSpin::Unpolarized)?;
    assert_real_close(second.free_energy_per_particle, -0.055_636_903_924_648_526);
    assert_real_close(second.potential, -0.080_007_059_893_872_29);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(8.0, 4.0, KsdTSpin::Unpolarized)?,
        -0.071_401_338_006_861_6,
    );

    let polarized =
        karasiev_sjostrom_dufty_trickey_free_energy(2.0, 0.5, KsdTSpin::FullyPolarized)?;
    assert_real_close(polarized.free_energy_per_particle, -0.271_613_018_966_040_1);
    assert_real_close(polarized.potential, -0.386_717_363_845_231_3);
    assert_real_close(
        karasiev_sjostrom_dufty_trickey_internal_energy(2.0, 0.5, KsdTSpin::FullyPolarized)?,
        -0.321_359_319_863_624_4,
    );
    Ok(())
}

#[test]
fn quinn_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    assert_real_close(
        quinn_imaginary_self_energy(1.15, 2.0, 0.65, 0.42)?,
        -0.002_466_676_087_856_595,
    );
    assert_real_close(
        quinn_imaginary_self_energy(2.4, 4.0, 0.35, 0.18)?,
        -0.000_005_424_606_390_748_76,
    );
    Ok(())
}

#[test]
fn hedin_lundqvist_imaginary_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    let first = hedin_lundqvist_imaginary_self_energy(2.0, 1.3)?;
    assert_real_close(first.value, -0.014_367_116_812_766_725);
    assert!(!first.cusp);

    let second = hedin_lundqvist_imaginary_self_energy(5.0, 0.8)?;
    assert_real_close(second.value, -0.062_054_667_501_585_545);
    assert!(second.cusp);
    Ok(())
}

#[test]
fn hedin_lundqvist_self_energy_matches_feff_reference() -> Result<(), ExchangeError> {
    let first = hedin_lundqvist_self_energy(2.0, 1.3)?;
    assert_real_close(first.real, -0.368_652_535_973_926_2);
    assert_real_close(first.imaginary, -0.014_367_116_812_766_725);
    assert!(!first.cusp);

    let second = hedin_lundqvist_self_energy(5.0, 0.8)?;
    assert_real_close(second.real, -0.205_217_998_385_377_2);
    assert_real_close(second.imaginary, -0.062_054_667_501_585_545);
    assert!(second.cusp);

    let third = hedin_lundqvist_self_energy(0.15, 3.5)?;
    assert_real_close(third.real, -4.207_016_089_629_206);
    assert_real_close(third.imaginary, -0.000_000_000_589_933_275_068_458_5);
    Ok(())
}

#[test]
fn xcpot_many_pole_density_grid_matches_feff_reference() -> Result<(), ExchangeError> {
    let bulk_density = ndarray::arr1(&[
        0.050, 0.040, 0.030, 0.022, 0.018, 0.010, 0.008, 0.006, 0.004, 0.003, 0.002, 0.001,
    ]);
    let bulk = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 1,
        density: bulk_density.view(),
        radial_match_index_1based: 5,
    })?;
    assert_real_close(bulk.interstitial_radius, 2.879_411_911_484_860_6);
    assert_real_close(bulk.core_radius, 1.683_890_300_960_629_6);
    assert_real_close(bulk.min_radius, 1.683_890_300_960_629_6);
    assert_real_close(bulk.max_radius, 2.879_411_911_484_860_6);
    assert_real_close(bulk.radius_step, 0.149_440_201_315_528_88);
    for radius in bulk.radii {
        assert_real_close(radius, 2.879_411_911_484_860_6);
    }

    let clamp_density = ndarray::arr1(&[
        0.050, 0.040, 0.004, 0.022, 0.020, 0.018, 0.012, 0.008, 0.004, 0.003, 0.002, 0.001,
    ]);
    let clamp = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 2,
        density: clamp_density.view(),
        radial_match_index_1based: 5,
    })?;
    assert_real_close(clamp.interstitial_radius, 2.367_080_141_024_981);
    assert_real_close(clamp.max_radius, 3.907_963_208_983_859_6);
    assert_xcpot_radii_close(
        &clamp.radii,
        &[
            1.683_890_300_960_629_6,
            1.961_899_414_463_533_3,
            2.239_908_527_966_437,
            2.367_080_141_024_981,
            2.517_917_641_469_341,
            2.795_926_754_972_244_4,
            3.073_935_868_475_148_3,
            3.351_944_981_978_052_2,
            3.629_954_095_480_956,
            3.907_963_208_983_859_6,
        ],
    );

    let linear_density = ndarray::arr1(&[
        0.050, 0.040, 0.030, 0.022, 0.020, 0.010, 0.012, 0.018, 0.020, 0.030, 0.040, 0.050,
    ]);
    let linear = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 2,
        density: linear_density.view(),
        radial_match_index_1based: 5,
    })?;
    assert_xcpot_radii_close(
        &linear.radii,
        &[
            1.683_890_300_960_629_6,
            1.833_330_502_276_158_6,
            1.982_770_703_591_687_5,
            2.132_210_904_907_216_4,
            2.281_651_106_222_745_4,
            2.431_091_307_538_274_3,
            2.580_531_508_853_803,
            2.729_971_710_169_331_7,
            2.879_411_911_484_860_6,
            3.028_852_112_800_389_6,
        ],
    );

    let mixed_density = ndarray::arr1(&[
        0.050, 0.040, 0.030, 0.022, 0.020, 0.000, 0.012, 0.008, 0.004, 0.003, 0.002, 0.001,
    ]);
    let mixed = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 2,
        density: mixed_density.view(),
        radial_match_index_1based: 5,
    })?;
    assert_real_close(mixed.interstitial_radius, 10.0);
    assert_real_close(mixed.max_radius, 20.0);
    assert_xcpot_radii_close(
        &mixed.radii,
        &[
            1.683_890_300_960_629_6,
            3.973_404_013_340_551,
            6.262_917_725_720_472,
            8.552_431_438_100_392,
            10.0,
            10.841_945_150_480_313,
            13.131_458_862_860_235,
            15.420_972_575_240_157,
            17.710_486_287_620_08,
            20.0,
        ],
    );

    let nonpositive_density = ndarray::Array1::<Real>::from_elem(12, -1.0);
    let nonpositive = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 2,
        density: nonpositive_density.view(),
        radial_match_index_1based: 5,
    })?;
    assert_real_close(nonpositive.interstitial_radius, 10.0);
    assert_real_close(nonpositive.core_radius, 101.0);
    assert_real_close(nonpositive.min_radius, 0.01);
    assert_real_close(nonpositive.max_radius, 20.0);
    assert_xcpot_radii_close(
        &nonpositive.radii,
        &[
            0.01,
            2.508_749_999_999_999_6,
            5.007_499_999_999_999,
            7.506_25,
            10.0,
            10.004_999_999_999_999,
            12.503_749_999_999_998,
            15.0025,
            17.501_25,
            20.0,
        ],
    );

    Ok(())
}

#[test]
fn xcpot_many_pole_control_matches_feff_reference() -> Result<(), ExchangeError> {
    let sentinel_after_minus_one = ndarray::arr1(&[0.20, 0.40, -1.00, 0.60, -1.0001, 0.90]);
    let control = xcpot_many_pole_control(XcpotManyPoleControlInput {
        plasmon_selector: 1,
        exchange_selector: 0,
        pole_frequencies: sentinel_after_minus_one.view(),
    })?;
    assert!(control.enabled);
    assert_eq!(control.active_pole_count, 4);

    let sentinel_first = ndarray::arr1(&[-2.00, 0.30, 0.40, 0.50, 0.60, 0.70]);
    let zero_poles = xcpot_many_pole_control(XcpotManyPoleControlInput {
        plasmon_selector: 2,
        exchange_selector: 10,
        pole_frequencies: sentinel_first.view(),
    })?;
    assert!(zero_poles.enabled);
    assert_eq!(zero_poles.active_pole_count, 0);

    let poison = ndarray::arr1(&[Real::NAN]);
    let disabled_by_plasmon = xcpot_many_pole_control(XcpotManyPoleControlInput {
        plasmon_selector: 0,
        exchange_selector: 0,
        pole_frequencies: poison.view(),
    })?;
    assert_eq!(
        disabled_by_plasmon,
        XcpotManyPoleControl {
            enabled: false,
            active_pole_count: 0,
        }
    );

    let disabled_by_exchange = xcpot_many_pole_control(XcpotManyPoleControlInput {
        plasmon_selector: 1,
        exchange_selector: 6,
        pole_frequencies: poison.view(),
    })?;
    assert_eq!(
        disabled_by_exchange,
        XcpotManyPoleControl {
            enabled: false,
            active_pole_count: 0,
        }
    );

    Ok(())
}

#[test]
fn xcpot_many_pole_delta_table_matches_feff_reference() -> Result<(), ExchangeError> {
    let fermi = xcpot_mpse_fermi_fixture();
    let energy = xcpot_mpse_energy_fixture();
    let renormalization = Complex::new(1.25, -0.15);

    let bulk = xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
        plasmon_selector: 1,
        fermi_self_energy: fermi.view(),
        energy_self_energy: energy.view(),
        renormalization,
    })?;
    assert_xcpot_complex_table_close(
        &bulk.fermi_self_energy,
        &[Complex::new(1.95, -0.38); XCPOT_MPSE_GRID_POINTS],
    );
    assert_xcpot_complex_table_close(
        &bulk.delta_self_energy,
        &[Complex::new(2.634_5, 0.292_500_000_000_000_15); XCPOT_MPSE_GRID_POINTS],
    );
    assert_xcpot_real_table_close(&bulk.real, &[2.634_5; XCPOT_MPSE_GRID_POINTS]);
    assert_xcpot_real_table_close(
        &bulk.imaginary,
        &[0.292_500_000_000_000_15; XCPOT_MPSE_GRID_POINTS],
    );

    let radial = xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
        plasmon_selector: 2,
        fermi_self_energy: fermi.view(),
        energy_self_energy: energy.view(),
        renormalization,
    })?;
    assert_xcpot_complex_table_close(
        &radial.fermi_self_energy,
        &[
            Complex::new(0.150_000_000_000_000_02, -0.02),
            Complex::new(0.350_000_000_000_000_03, -0.06),
            Complex::new(0.55, -0.1),
            Complex::new(0.75, -0.14),
            Complex::new(0.95, -0.180_000_000_000_000_02),
            Complex::new(1.150_000_000_000_000_1, -0.22),
            Complex::new(1.35, -0.26),
            Complex::new(1.55, -0.3),
            Complex::new(1.75, -0.339_999_999_999_999_97),
            Complex::new(1.95, -0.38),
        ],
    );
    assert_xcpot_real_table_close(
        &radial.real,
        &[
            1.235,
            1.390_499_999_999_999_8,
            1.546,
            1.701_500_000_000_000_2,
            1.857_000_000_000_000_2,
            2.012_499_999_999_999_7,
            2.167_999_999_999_999_3,
            2.323_499_999_999_999_7,
            2.478_999_999_999_999_6,
            2.634_5,
        ],
    );
    assert_xcpot_real_table_close(
        &radial.imaginary,
        &[
            -0.909,
            -0.775_499_999_999_999_9,
            -0.641_999_999_999_999_9,
            -0.508_5,
            -0.374_999_999_999_999_9,
            -0.241_499_999_999_999_9,
            -0.107_999_999_999_999_82,
            0.025_500_000_000_000_078,
            0.158_999_999_999_999_97,
            0.292_500_000_000_000_15,
        ],
    );
    assert_xcpot_real_table_close(
        &radial.real,
        &radial.delta_self_energy.map(|value| value.re),
    );
    assert_xcpot_real_table_close(
        &radial.imaginary,
        &radial.delta_self_energy.map(|value| value.im),
    );

    Ok(())
}

#[test]
fn xcpot_many_pole_self_energy_delta_table_matches_feff_reference() -> Result<(), ExchangeError> {
    let bulk_radii = [2.1; XCPOT_MPSE_GRID_POINTS];
    let bulk_grid = XcpotManyPoleDensityGrid {
        interstitial_radius: 2.1,
        core_radius: 2.1,
        min_radius: 2.1,
        max_radius: 2.1,
        radius_step: 0.0,
        radii: bulk_radii,
    };
    let bulk_frequencies = ndarray::arr1(&[0.35, 0.80, 1.60, -2.0]);
    let bulk_widths = ndarray::arr1(&[0.02, 0.03, 0.04, 0.0]);
    let bulk_amplitudes = ndarray::arr1(&[0.25, 0.40, 0.15, 0.0]);
    let bulk = xcpot_many_pole_self_energy_delta_table(XcpotManyPoleSelfEnergyTableInput {
        plasmon_selector: 1,
        energy: Complex::new(0.40, 0.0),
        fermi_level: 0.05,
        density_grid: bulk_grid,
        pole_frequencies: bulk_frequencies.view(),
        pole_widths: bulk_widths.view(),
        amplitudes: bulk_amplitudes.view(),
        gap_energy: 0.03,
        active_pole_count: 3,
        use_broadened_pole: false,
    })?;
    assert_complex_table_near(
        &bulk.fermi_self_energy,
        &[Complex::new(-3.235_450_358_541_453_5e-1, -9.706_472_595_428_154e-12);
            XCPOT_MPSE_GRID_POINTS],
        1.0e-8,
    );
    assert_complex_table_near(
        &bulk.delta_self_energy,
        &[Complex::new(2.790_576_938_755_429e-2, -1.442_841_136_785_682_4e-2);
            XCPOT_MPSE_GRID_POINTS],
        1.0e-8,
    );

    let radial_grid = XcpotManyPoleDensityGrid {
        interstitial_radius: 3.2,
        core_radius: 1.4,
        min_radius: 1.4,
        max_radius: 3.2,
        radius_step: 0.2,
        radii: [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
    };
    let radial_frequencies = ndarray::arr1(&[0.32, 0.75, -2.0]);
    let radial_widths = ndarray::arr1(&[0.01, 0.02, 0.0]);
    let radial_amplitudes = ndarray::arr1(&[0.40, 0.20, 0.0]);
    let radial = xcpot_many_pole_self_energy_delta_table(XcpotManyPoleSelfEnergyTableInput {
        plasmon_selector: 2,
        energy: Complex::new(0.32, 0.0),
        fermi_level: 0.03,
        density_grid: radial_grid,
        pole_frequencies: radial_frequencies.view(),
        pole_widths: radial_widths.view(),
        amplitudes: radial_amplitudes.view(),
        gap_energy: 0.02,
        active_pole_count: 2,
        use_broadened_pole: false,
    })?;
    assert_complex_table_near(
        &radial.fermi_self_energy,
        &[
            Complex::new(-4.451_809_680_606_597e-1, -1.206_369_334_315_174_7e-11),
            Complex::new(-3.894_912_712_879_034e-1, -1.049_081_269_766_274_6e-11),
            Complex::new(-3.460_391_207_438_889e-1, -9.274_024_644_354_811e-12),
            Complex::new(-3.111_756_991_168_569e-1, -8.306_016_197_286_569e-12),
            Complex::new(-2.825_650_854_491_354_5e-1, -7.518_527_765_263_462e-12),
            Complex::new(-2.586_532_113_478_711_7e-1, -6.866_145_741_396_383e-12),
            Complex::new(-2.383_631_387_691_121_7e-1, -6.317_511_389_518_123e-12),
            Complex::new(-2.209_248_653_384_096_6e-1, -5.850_233_057_171_287_5e-12),
            Complex::new(-2.057_731_503_774_479e-1, -5.447_920_421_829_496e-12),
            Complex::new(-1.924_836_204_175_405_8e-1, -5.098_296_121_745_192e-12),
        ],
        1.0e-8,
    );
    assert_complex_table_near(
        &radial.delta_self_energy,
        &[
            Complex::new(6.456_173_591_052_022e-2, -1.660_428_194_501_701e-11),
            Complex::new(5.431_289_546_035_571e-2, -3.533_032_305_744_311e-11),
            Complex::new(4.729_100_350_558_293_5e-2, -2.136_058_187_351_263e-2),
            Complex::new(5.192_369_235_878_823e-2, -2.680_451_939_734_807_7e-2),
            Complex::new(5.351_639_516_634_02e-2, -2.891_304_202_680_845e-2),
            Complex::new(5.326_725_417_463_32e-2, -2.936_384_183_623_174_5e-2),
            Complex::new(5.135_124_013_948_820_5e-2, -2.859_194_683_814_302_7e-2),
            Complex::new(4.689_403_817_926_209e-2, -2.613_628_590_834_874_6e-2),
            Complex::new(3.272_351_445_486_075e-2, -3.471_187_271_413_82e-2),
            Complex::new(3.585_511_467_803_860_5e-2, -4.186_728_660_237_111_5e-2),
        ],
        1.0e-8,
    );

    Ok(())
}

#[test]
fn xcpot_many_pole_self_energy_delta_table_forwards_bpr_selector() -> Result<(), ExchangeError> {
    let radii = [2.1; XCPOT_MPSE_GRID_POINTS];
    let density_grid = XcpotManyPoleDensityGrid {
        interstitial_radius: 2.1,
        core_radius: 2.1,
        min_radius: 2.1,
        max_radius: 2.1,
        radius_step: 0.0,
        radii,
    };
    let frequencies = ndarray::arr1(&[0.35, 0.80, 1.60, -2.0]);
    let widths = ndarray::arr1(&[0.02, 0.03, 0.04, 0.0]);
    let amplitudes = ndarray::arr1(&[0.25, 0.40, 0.15, 0.0]);

    let broadened = xcpot_many_pole_self_energy_delta_table(XcpotManyPoleSelfEnergyTableInput {
        plasmon_selector: 1,
        energy: Complex::new(0.40, 0.0),
        fermi_level: 0.05,
        density_grid,
        pole_frequencies: frequencies.view(),
        pole_widths: widths.view(),
        amplitudes: amplitudes.view(),
        gap_energy: 0.03,
        active_pole_count: 3,
        use_broadened_pole: true,
    })?;
    assert_complex_table_near(
        &broadened.fermi_self_energy,
        &[Complex::new(-3.278_038_018_649_215_3e-1, -1.587_262_382_066_107_7e-6);
            XCPOT_MPSE_GRID_POINTS],
        1.0e-9,
    );
    assert_complex_table_near(
        &broadened.delta_self_energy,
        &[Complex::new(2.195_464_602_869_098e-2, -2.056_664_185_489_893e-2);
            XCPOT_MPSE_GRID_POINTS],
        1.0e-9,
    );
    Ok(())
}

#[test]
fn xcpot_many_pole_row_delta_matches_feff_reference() -> Result<(), ExchangeError> {
    let density_grid = xcpot_mpse_row_density_grid_fixture();
    let delta_table = xcpot_mpse_row_delta_table_fixture();
    let cases = [
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 1,
                radius: 2.25,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 10.0,
                imaginary: -1.0,
            },
        ),
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 2,
                radius: 2.25,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 22.5,
                imaginary: -2.25,
            },
        ),
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 2,
                radius: 1.0,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 10.0,
                imaginary: -1.0,
            },
        ),
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 2,
                radius: 10.0,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 100.0,
                imaginary: -10.0,
            },
        ),
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 2,
                radius: 4.75,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 47.5,
                imaginary: -4.75,
            },
        ),
        (
            XcpotManyPoleRowDeltaInput {
                plasmon_selector: 2,
                radius: 0.5,
                density_grid,
                delta_table,
            },
            XcpotSigma {
                real: 0.0,
                imaginary: 0.0,
            },
        ),
    ];

    for (input, expected) in cases {
        assert_xcpot_sigma_near(xcpot_many_pole_row_delta(input)?, expected);
    }

    Ok(())
}

#[test]
fn xcpot_self_energy_delta_application_matches_feff_reference() -> Result<(), ExchangeError> {
    let total = xcpot_real_total_fixture();
    let valence = xcpot_real_valence_fixture();
    let delta_real = xcpot_delta_real_fixture();
    let delta_imaginary = xcpot_delta_imaginary_fixture();
    let valence_delta_real = xcpot_valence_delta_real_fixture();
    let valence_delta_imaginary = xcpot_valence_delta_imaginary_fixture();
    let poison = ndarray::arr1(&[Real::NAN]);

    let ixc0 = xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
        exchange_selector: 0,
        total_potential: total.view(),
        valence_potential: poison.view(),
        delta_real: delta_real.view(),
        delta_imaginary: delta_imaginary.view(),
        valence_delta_real: poison.view(),
        valence_delta_imaginary: poison.view(),
        active_len: 4,
    })?;
    assert_complex_slice_close(
        &ixc0.total_potential,
        &[
            Complex::new(10.1, -0.01),
            Complex::new(11.3, -0.02),
            Complex::new(13.6, 0.03),
            Complex::new(15.5, 0.04),
        ],
    );
    assert!(ixc0.valence_potential.is_empty());

    let ixc5 = xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
        exchange_selector: 15,
        total_potential: total.view(),
        valence_potential: valence.view(),
        delta_real: delta_real.view(),
        delta_imaginary: poison.view(),
        valence_delta_real: valence_delta_real.view(),
        valence_delta_imaginary: valence_delta_imaginary.view(),
        active_len: 4,
    })?;
    assert_complex_slice_close(
        &ixc5.total_potential,
        &[
            Complex::new(10.1, 0.21),
            Complex::new(11.3, 0.22),
            Complex::new(13.6, -0.23),
            Complex::new(15.5, -0.24),
        ],
    );
    assert_complex_slice_close(
        &ixc5.valence_potential,
        &[
            Complex::new(21.1, 0.21),
            Complex::new(22.7, 0.22),
            Complex::new(21.95, -0.23),
            Complex::new(23.6, -0.24),
        ],
    );

    let ixc6 = xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
        exchange_selector: 6,
        total_potential: total.view(),
        valence_potential: valence.view(),
        delta_real: delta_real.view(),
        delta_imaginary: delta_imaginary.view(),
        valence_delta_real: valence_delta_real.view(),
        valence_delta_imaginary: valence_delta_imaginary.view(),
        active_len: 4,
    })?;
    assert_complex_slice_close(&ixc6.total_potential, &ixc0.total_potential.to_vec());
    assert_eq!(ixc6.valence_potential, ixc5.valence_potential);

    Ok(())
}

#[test]
fn xcpot_local_scales_match_feff_reference() -> Result<(), ExchangeError> {
    let ixc0 = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 0,
        density: 0.02,
        magnetization: 0.05,
        valence_density: 0.004,
    })?;
    assert_xcpot_local_scale_close(
        ixc0,
        XcpotLocalScales {
            radius: 2.285_390_748_670_416,
            fermi_momentum: 0.839_750_617_610_591,
            magnetized_radius: 2.248_523_160_622_389,
            magnetized_fermi_momentum: 0.853_519_468_372_427_9,
            valence_radius: None,
            valence_fermi_momentum: None,
            core_radius: None,
        },
    );

    let ixc5_threshold = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 5,
        density: 0.02,
        magnetization: 0.05,
        valence_density: 0.000_01,
    })?;
    assert_real_close(ixc5_threshold.valence_radius.unwrap(), 10.0);
    assert_real_close(
        ixc5_threshold.valence_fermi_momentum.unwrap(),
        0.191_915_829_267_751_28,
    );
    assert_eq!(ixc5_threshold.core_radius, None);

    let ixc5_clamped = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 5,
        density: 0.02,
        magnetization: 0.05,
        valence_density: 0.000_02,
    })?;
    assert_real_close(ixc5_clamped.valence_radius.unwrap(), 10.0);
    assert_real_close(
        ixc5_clamped.valence_fermi_momentum.unwrap(),
        0.191_915_829_267_751_28,
    );

    let index15 = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 15,
        density: 0.02,
        magnetization: 0.05,
        valence_density: 0.02,
    })?;
    assert_real_close(index15.valence_radius.unwrap(), 2.285_390_748_670_416);
    assert_real_close(
        index15.valence_fermi_momentum.unwrap(),
        0.839_750_617_610_591,
    );

    let ixc6_fallback = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 6,
        density: 0.03,
        magnetization: 0.10,
        valence_density: 0.04,
    })?;
    assert_real_close(ixc6_fallback.radius, 1.996_472_712_327_54);
    assert_real_close(ixc6_fallback.fermi_momentum, 0.961_274_492_171_800_3);
    assert_real_close(ixc6_fallback.magnetized_radius, 1.934_041_625_363_389_9);
    assert_real_close(
        ixc6_fallback.magnetized_fermi_momentum,
        0.992_304_543_764_366_7,
    );
    assert_eq!(ixc6_fallback.valence_radius, None);
    assert_eq!(ixc6_fallback.valence_fermi_momentum, None);
    assert_real_close(ixc6_fallback.core_radius.unwrap(), 101.0);

    let index16_core = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 16,
        density: 0.08,
        magnetization: 0.20,
        valence_density: 0.03,
    })?;
    assert_real_close(index16_core.radius, 1.439_705_955_742_430_3);
    assert_real_close(index16_core.fermi_momentum, 1.333_021_013_785_997_5);
    assert_real_close(index16_core.magnetized_radius, 1.354_815_175_348_222_3);
    assert_real_close(
        index16_core.magnetized_fermi_momentum,
        1.416_546_203_200_181_9,
    );
    assert_real_close(index16_core.core_radius.unwrap(), 1.683_890_300_960_629_6);

    let nonpositive_density = xcpot_local_scales(XcpotLocalScalesInput {
        exchange_selector: 6,
        density: -0.50,
        magnetization: 0.0,
        valence_density: 0.40,
    })?;
    assert_real_close(nonpositive_density.radius, 10.0);
    assert_real_close(nonpositive_density.fermi_momentum, 0.191_915_829_267_751_28);
    assert_real_close(nonpositive_density.magnetized_radius, 10.0);
    assert_real_close(
        nonpositive_density.magnetized_fermi_momentum,
        0.191_915_829_267_751_28,
    );
    assert_real_close(nonpositive_density.core_radius.unwrap(), 101.0);

    Ok(())
}

#[test]
fn xcpot_sigma_matches_feff_reference() -> Result<(), ExchangeError> {
    let cases = [
        (
            XcpotSigmaInput {
                exchange_selector: 0,
                radius: 2.0,
                core_radius: 101.0,
                momentum: 1.3,
            },
            XcpotSigma {
                real: -0.368_652_535_973_926_2,
                imaginary: -0.014_367_116_812_766_725,
            },
        ),
        (
            XcpotSigmaInput {
                exchange_selector: 1,
                radius: 2.0,
                core_radius: 101.0,
                momentum: 1.3,
            },
            XcpotSigma {
                real: -0.127_197_685_551_846_32,
                imaginary: 0.0,
            },
        ),
        (
            XcpotSigmaInput {
                exchange_selector: 3,
                radius: 2.0,
                core_radius: 101.0,
                momentum: 1.3,
            },
            XcpotSigma {
                real: -0.127_197_685_551_846_32,
                imaginary: -0.014_367_116_812_766_725,
            },
        ),
        (
            XcpotSigmaInput {
                exchange_selector: 5,
                radius: 2.0,
                core_radius: 101.0,
                momentum: 1.3,
            },
            XcpotSigma {
                real: -0.368_652_535_973_926_2,
                imaginary: -0.014_367_116_812_766_725,
            },
        ),
        (
            XcpotSigmaInput {
                exchange_selector: 6,
                radius: 2.0,
                core_radius: 1.5,
                momentum: 1.3,
            },
            XcpotSigma {
                real: 0.007_216_185_319_526_447,
                imaginary: -0.014_367_116_812_766_725,
            },
        ),
    ];

    for (input, expected) in cases {
        assert_xcpot_sigma_near(xcpot_sigma(input)?, expected);
    }

    Ok(())
}

#[test]
fn xcpot_fermi_cache_matches_feff_reference() -> Result<(), ExchangeError> {
    let cases = [
        (
            XcpotFermiCacheInput {
                exchange_selector: 0,
                radius: 2.0,
                core_radius: 101.0,
                valence_radius: None,
                interstitial: false,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                valence_self_energy: XcpotSigma {
                    real: 0.0,
                    imaginary: 0.0,
                },
                ground_state_ratio: 1.0,
            },
        ),
        (
            XcpotFermiCacheInput {
                exchange_selector: 1,
                radius: 2.0,
                core_radius: 101.0,
                valence_radius: None,
                interstitial: false,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.305_373_198_526_817_94,
                    imaginary: 0.0,
                },
                valence_self_energy: XcpotSigma {
                    real: 0.0,
                    imaginary: 0.0,
                },
                ground_state_ratio: 1.0,
            },
        ),
        (
            XcpotFermiCacheInput {
                exchange_selector: 5,
                radius: 2.0,
                core_radius: 101.0,
                valence_radius: Some(3.0),
                interstitial: false,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                valence_self_energy: XcpotSigma {
                    real: -0.258_167_243_559_313_1,
                    imaginary: -0.000_000_000_007_295_955,
                },
                ground_state_ratio: 1.0,
            },
        ),
        (
            XcpotFermiCacheInput {
                exchange_selector: 5,
                radius: 2.0,
                core_radius: 101.0,
                valence_radius: Some(3.0),
                interstitial: true,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                valence_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                ground_state_ratio: 1.0,
            },
        ),
        (
            XcpotFermiCacheInput {
                exchange_selector: 6,
                radius: 2.0,
                core_radius: 1.5,
                valence_radius: None,
                interstitial: false,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                valence_self_energy: XcpotSigma {
                    real: 0.038_219_342_944_246_39,
                    imaginary: -0.000_000_000_013_683_98,
                },
                ground_state_ratio: 1.0,
            },
        ),
        (
            XcpotFermiCacheInput {
                exchange_selector: 6,
                radius: 2.0,
                core_radius: 1.5,
                valence_radius: None,
                interstitial: true,
            },
            XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                valence_self_energy: XcpotSigma {
                    real: -0.368_988_923_572_785,
                    imaginary: -0.000_000_000_013_683_98,
                },
                ground_state_ratio: 1.0,
            },
        ),
    ];

    for (input, expected) in cases {
        assert_xcpot_fermi_cache_near(xcpot_fermi_cache(input)?, expected);
    }

    Ok(())
}

#[test]
fn xcpot_self_energy_correction_matches_feff_reference() -> Result<(), ExchangeError> {
    let cases = [
        (
            xcpot_correction_input(0, false, 1.2, 101.0, None, 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.766_576_388_976_207_4,
                corrected_momentum: 1.777_156_570_227_309_3,
                total_delta: XcpotSigma {
                    real: -0.019_327_522_407_820_43,
                    imaginary: -0.135_820_265_392_827_7,
                },
                valence_delta: None,
            },
        ),
        (
            xcpot_correction_input(1, false, 1.2, 101.0, None, 0.1)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.783_591_366_301_124_7,
                corrected_momentum: 1.632_162_334_739_870_4,
                total_delta: XcpotSigma {
                    real: 0.229_225_011_279_819_98,
                    imaginary: 0.0,
                },
                valence_delta: None,
            },
        ),
        (
            xcpot_correction_input(3, false, 1.2, 101.0, None, 0.2)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.799_943_969_256_179,
                corrected_momentum: 1.632_162_334_739_870_4,
                total_delta: XcpotSigma {
                    real: 0.229_225_011_279_819_98,
                    imaginary: -0.088_858_033_200_113_55,
                },
                valence_delta: None,
            },
        ),
        (
            xcpot_correction_input(5, true, 1.2, 101.0, Some(3.0), 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.766_576_388_976_207_4,
                corrected_momentum: 1.777_156_570_227_309_3,
                total_delta: XcpotSigma {
                    real: -0.019_327_522_407_820_43,
                    imaginary: -0.135_820_265_392_827_7,
                },
                valence_delta: Some(XcpotSigma {
                    real: -0.019_327_522_407_820_43,
                    imaginary: -0.135_820_265_392_827_7,
                }),
            },
        ),
        (
            xcpot_correction_input(5, false, 1.2, 101.0, Some(3.0), 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.766_576_388_976_207_4,
                corrected_momentum: 1.777_156_570_227_309_3,
                total_delta: XcpotSigma {
                    real: -0.019_327_522_407_820_43,
                    imaginary: -0.135_820_265_392_827_7,
                },
                valence_delta: Some(XcpotSigma {
                    real: 0.048_976_836_887_211_356,
                    imaginary: -0.133_607_865_020_133_67,
                }),
            },
        ),
        (
            xcpot_correction_input(6, true, 1.2, 1.5, None, 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.766_576_388_976_207_4,
                corrected_momentum: 1.684_991_539_327_440_2,
                total_delta: XcpotSigma {
                    real: 0.131_274_225_669_682_25,
                    imaginary: -0.110_907_463_457_444_25,
                },
                valence_delta: Some(XcpotSigma {
                    real: 0.131_274_225_669_682_25,
                    imaginary: -0.110_907_463_457_444_25,
                }),
            },
        ),
        (
            xcpot_correction_input(6, false, 1.2, 1.5, None, 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 1.766_576_388_976_207_4,
                corrected_momentum: 1.684_991_539_327_440_2,
                total_delta: XcpotSigma {
                    real: 0.131_274_225_669_682_25,
                    imaginary: -0.110_907_463_457_444_25,
                },
                valence_delta: Some(XcpotSigma {
                    real: -0.275_934_040_847_349_2,
                    imaginary: -0.110_907_463_457_444_25,
                }),
            },
        ),
        (
            xcpot_correction_input(5, false, 0.11, 101.0, Some(3.0), 0.0)?,
            XcpotSelfEnergyCorrection {
                magnetized_momentum: 0.969_944_399_482_886,
                corrected_momentum: 0.969_555_861_826_545_4,
                total_delta: XcpotSigma {
                    real: 0.000_377_386_765_188_669_76,
                    imaginary: -0.000_014_706_040_797_751_255,
                },
                valence_delta: Some(XcpotSigma {
                    real: 0.000_152_544_464_668_058_5,
                    imaginary: -0.000_039_101_705_104_229_7,
                }),
            },
        ),
    ];

    for (input, expected) in cases {
        assert_xcpot_self_energy_correction_near(xcpot_self_energy_correction(input)?, expected);
    }

    Ok(())
}

#[test]
fn xcpot_reference_shift_matches_feff_reference() -> Result<(), ExchangeError> {
    let total = xcpot_total_fixture();
    let poison_valence = ndarray::arr1(&[Complex::new(Real::NAN, Real::NAN)]);

    let ixc0 = xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: 0,
        lreal: 0,
        total_potential: total.view(),
        valence_potential: poison_valence.view(),
        active_len: 4,
    })?;
    assert_complex_close(ixc0.reference_energy, Complex::new(15.0, 1.5));
    assert_complex_close(ixc0.total_potential[0], Complex::new(-5.0, -2.5));
    assert_complex_close(ixc0.total_potential[3], Complex::new(0.0, 0.0));
    assert_eq!(ixc0.valence_potential, ixc0.total_potential);

    let valence = xcpot_valence_fixture();
    let ixc6 = xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: 6,
        lreal: 0,
        total_potential: total.view(),
        valence_potential: valence.view(),
        active_len: 4,
    })?;
    assert_complex_close(ixc6.reference_energy, Complex::new(15.0, 1.5));
    assert_complex_close(ixc6.total_potential[1], Complex::new(-3.5, -2.0));
    assert_complex_close(ixc6.valence_potential[0], Complex::new(5.0, -3.5));
    assert_complex_close(ixc6.valence_potential[3], Complex::new(10.0, 1.0));

    let ixc0_lreal = xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: 0,
        lreal: 1,
        total_potential: total.view(),
        valence_potential: poison_valence.view(),
        active_len: 4,
    })?;
    assert_complex_close(ixc0_lreal.reference_energy, Complex::new(15.0, 0.0));
    assert_complex_close(ixc0_lreal.total_potential[0], Complex::new(-5.0, 0.0));
    assert_complex_close(ixc0_lreal.total_potential[2], Complex::new(-1.75, 0.0));
    assert_complex_close(ixc0_lreal.valence_potential[0], Complex::new(-5.0, -2.5));
    assert_complex_close(ixc0_lreal.valence_potential[2], Complex::new(-1.75, -1.25));

    let index15_lreal = xcpot_reference_shift(XcpotReferenceShiftInput {
        exchange_selector: 15,
        lreal: 1,
        total_potential: total.view(),
        valence_potential: valence.view(),
        active_len: 4,
    })?;
    assert_complex_close(index15_lreal.reference_energy, Complex::new(15.0, 0.0));
    assert_complex_close(index15_lreal.total_potential[1], Complex::new(-3.5, 0.0));
    assert_complex_close(index15_lreal.valence_potential[0], Complex::new(5.0, 0.0));
    assert_complex_close(index15_lreal.valence_potential[3], Complex::new(10.0, 0.0));

    Ok(())
}

#[test]
fn xcpot_ground_state_branch_matches_feff_reference() -> Result<(), ExchangeError> {
    let total = xcpot_real_total_fixture();
    let poison_valence = ndarray::arr1(&[Real::NAN]);

    let ixc2 = xcpot_ground_state_branch(XcpotGroundStateBranchInput {
        exchange_selector: 12,
        lreal: 0,
        energy: Complex::new(8.5, 0.25),
        fermi_level: 2.0,
        total_potential: total.view(),
        valence_potential: poison_valence.view(),
        active_len: 4,
    })?
    .expect("FEFF enters static branch for ixc == 2");
    assert_complex_close(ixc2.reference_energy, Complex::new(15.0, 0.0));
    assert_complex_close(ixc2.total_potential[0], Complex::new(-5.0, 0.0));
    assert_complex_close(ixc2.total_potential[3], Complex::new(0.0, 0.0));
    assert_eq!(ixc2.valence_potential, ixc2.total_potential);

    let valence = xcpot_real_valence_fixture();
    let ixc6 = xcpot_ground_state_branch(XcpotGroundStateBranchInput {
        exchange_selector: 6,
        lreal: 0,
        energy: Complex::new(1.5, 0.25),
        fermi_level: 2.0,
        total_potential: total.view(),
        valence_potential: valence.view(),
        active_len: 4,
    })?
    .expect("FEFF enters static branch below the Fermi level");
    assert_complex_close(ixc6.reference_energy, Complex::new(15.0, 0.0));
    assert_complex_close(ixc6.total_potential[1], Complex::new(-3.5, 0.0));
    assert_complex_close(ixc6.valence_potential[0], Complex::new(5.0, 0.0));
    assert_complex_close(ixc6.valence_potential[3], Complex::new(10.0, 0.0));

    let index15_lreal = xcpot_ground_state_branch(XcpotGroundStateBranchInput {
        exchange_selector: 15,
        lreal: 1,
        energy: Complex::new(2.0, 0.25),
        fermi_level: 2.0,
        total_potential: total.view(),
        valence_potential: valence.view(),
        active_len: 4,
    })?
    .expect("FEFF enters static branch for equality to the Fermi level");
    assert_complex_close(index15_lreal.reference_energy, Complex::new(15.0, 0.0));
    assert_complex_close(index15_lreal.total_potential[2], Complex::new(-1.75, 0.0));
    assert_complex_close(index15_lreal.valence_potential[2], Complex::new(8.25, 0.0));

    Ok(())
}

#[test]
fn xcpot_ground_state_branch_preserves_feff_skip_condition() -> Result<(), ExchangeError> {
    let poison = ndarray::arr1(&[Real::NAN]);
    let skipped = xcpot_ground_state_branch(XcpotGroundStateBranchInput {
        exchange_selector: 0,
        lreal: 0,
        energy: Complex::new(2.5, 0.25),
        fermi_level: 2.0,
        total_potential: poison.view(),
        valence_potential: poison.view(),
        active_len: 4,
    })?;
    assert_eq!(skipped, None);
    Ok(())
}

#[test]
fn xcpot_composed_dynamic_hl_matches_feff_reference() -> Result<(), ExchangeError> {
    let total = ndarray::arr1(&[10.0, 11.1, 14.9, 15.46]);
    let valence = ndarray::arr1(&[20.0, 21.2, 23.0, 24.5]);
    let density = ndarray::arr1(&[0.040, 0.030, 0.020, 0.015]);
    let magnetization = ndarray::arr1(&[0.00, 0.02, 0.05, 0.10]);
    let valence_density = ndarray::arr1(&[0.010, 0.012, 0.014, 0.015]);

    let result = xcpot(XcpotInput {
        exchange_selector: 0,
        lreal: 0,
        energy: Complex::new(1.2, 0.25),
        fermi_level: 0.1,
        total_potential: total.view(),
        valence_potential: valence.view(),
        density: density.view(),
        magnetization: magnetization.view(),
        valence_density: valence_density.view(),
        active_len: 4,
        plasmon_selector: 0,
        many_pole_delta_table: None,
        many_pole_self_energy: None,
        fermi_cache: None,
    })?;

    assert_complex_close(
        result.reference_energy,
        Complex::new(15.481_502_175_856_003, -0.140_764_098_399_660_06),
    );
    assert_complex_slice_close(
        &result.total_potential,
        &[
            Complex::new(-5.524_431_393_020_59, 0.017_304_446_670_349_652),
            Complex::new(-4.401_207_209_867_412, 0.005_094_373_886_891_851),
            Complex::new(-0.575_149_615_522_073_9, -0.000_861_494_439_687_693_3),
            Complex::new(0.0, 0.0),
        ],
    );
    assert_eq!(result.valence_potential, result.total_potential);
    assert_xcpot_fermi_cache_near(
        result.fermi_cache[0],
        XcpotFermiCache {
            total_self_energy: XcpotSigma {
                real: -0.402_609_747_291_588_2,
                imaginary: -0.000_000_000_015_881_689_339_979_514,
            },
            valence_self_energy: XcpotSigma {
                real: 0.0,
                imaginary: 0.0,
            },
            ground_state_ratio: 1.0,
        },
    );
    assert_xcpot_fermi_cache_near(
        result.fermi_cache[3],
        XcpotFermiCache {
            total_self_energy: XcpotSigma {
                real: -0.301_206_273_895_566_7,
                imaginary: -0.000_000_000_009_609_971_829_232_23,
            },
            valence_self_energy: XcpotSigma {
                real: 0.0,
                imaginary: 0.0,
            },
            ground_state_ratio: 1.0,
        },
    );

    let reused = xcpot(XcpotInput {
        fermi_cache: Some(result.fermi_cache.view()),
        ..XcpotInput {
            exchange_selector: 0,
            lreal: 0,
            energy: Complex::new(1.2, 0.25),
            fermi_level: 0.1,
            total_potential: total.view(),
            valence_potential: valence.view(),
            density: density.view(),
            magnetization: magnetization.view(),
            valence_density: valence_density.view(),
            active_len: 4,
            plasmon_selector: 0,
            many_pole_delta_table: None,
            many_pole_self_energy: None,
            fermi_cache: None,
        }
    })?;
    assert_eq!(reused, result);

    Ok(())
}

#[test]
fn xcpot_composed_dynamic_valence_matches_feff_reference() -> Result<(), ExchangeError> {
    let total = ndarray::arr1(&[8.0, 9.25, 10.75, 12.0]);
    let valence = ndarray::arr1(&[3.0, 3.5, 4.25, 4.8]);
    let density = ndarray::arr1(&[0.070, 0.052, 0.031, 0.022]);
    let magnetization = ndarray::arr1(&[0.01, 0.03, 0.04, 0.02]);
    let valence_density = ndarray::arr1(&[0.014, 0.018, 0.020, 0.022]);

    let result = xcpot(XcpotInput {
        exchange_selector: 5,
        lreal: 1,
        energy: Complex::new(1.35, 0.15),
        fermi_level: 0.2,
        total_potential: total.view(),
        valence_potential: valence.view(),
        density: density.view(),
        magnetization: magnetization.view(),
        valence_density: valence_density.view(),
        active_len: 4,
        plasmon_selector: 0,
        many_pole_delta_table: None,
        many_pole_self_energy: None,
        fermi_cache: None,
    })?;

    assert_complex_close(
        result.reference_energy,
        Complex::new(12.007_002_034_985_957, 0.0),
    );
    assert_complex_slice_close(
        &result.total_potential,
        &[
            Complex::new(-4.079_571_289_764_123, 0.0),
            Complex::new(-2.818_629_935_649_841_7, 0.0),
            Complex::new(-1.272_187_064_404_565_9, 0.0),
            Complex::new(0.0, 0.0),
        ],
    );
    assert_complex_slice_close(
        &result.valence_potential,
        &[
            Complex::new(-8.965_513_587_072_822, 0.0),
            Complex::new(-8.479_329_332_228_906, 0.0),
            Complex::new(-7.740_978_108_497_407, 0.0),
            Complex::new(-7.2, 0.0),
        ],
    );
    assert_xcpot_fermi_cache_near(
        result.fermi_cache[0],
        XcpotFermiCache {
            total_self_energy: XcpotSigma {
                real: -0.476_135_099_741_379_4,
                imaginary: -0.000_000_000_021_057_777_905_370_05,
            },
            valence_self_energy: XcpotSigma {
                real: -0.295_178_007_956_640_45,
                imaginary: -0.000_000_000_009_272_407_099_441_631,
            },
            ground_state_ratio: 1.0,
        },
    );
    assert_xcpot_fermi_cache_near(
        result.fermi_cache[3],
        XcpotFermiCache {
            total_self_energy: XcpotSigma {
                real: -0.337_139_662_906_284_16,
                imaginary: -0.000_000_000_011_707_838_794_564_854,
            },
            valence_self_energy: XcpotSigma {
                real: -0.337_139_662_906_284_16,
                imaginary: -0.000_000_000_011_707_838_794_564_854,
            },
            ground_state_ratio: 1.0,
        },
    );

    Ok(())
}

#[test]
fn xcpot_composed_mpse_uses_prepared_delta_table() -> Result<(), ExchangeError> {
    let radius_density = |radius: Real| 3.0 / (4.0 * FEFF_PI * radius.powi(3));
    let total = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
    let poison = ndarray::arr1(&[Real::NAN]);
    let density = ndarray::arr1(&[
        radius_density(1.0),
        radius_density(2.0),
        radius_density(3.0),
        radius_density(4.0),
    ]);
    let zeros = ndarray::Array1::<Real>::zeros(4);
    let density_grid = xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
        plasmon_selector: 2,
        density: density.view(),
        radial_match_index_1based: 3,
    })?;
    let real = density_grid.radii.map(|radius| 10.0 * radius);
    let imaginary = density_grid.radii.map(|radius| -radius);
    let delta_table = XcpotManyPoleDeltaTable {
        fermi_self_energy: [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS],
        delta_self_energy: real.map(|value| Complex::new(value, -0.1 * value)),
        real,
        imaginary,
    };

    let result = xcpot(XcpotInput {
        exchange_selector: 0,
        lreal: 0,
        energy: Complex::new(1.0, 0.0),
        fermi_level: 0.0,
        total_potential: total.view(),
        valence_potential: poison.view(),
        density: density.view(),
        magnetization: zeros.view(),
        valence_density: zeros.view(),
        active_len: 4,
        plasmon_selector: 2,
        many_pole_delta_table: Some(delta_table),
        many_pole_self_energy: None,
        fermi_cache: None,
    })?;

    assert_complex_close(result.reference_energy, Complex::new(44.0, -4.0));
    assert_complex_slice_close(
        &result.total_potential,
        &[
            Complex::new(-33.0, 3.0),
            Complex::new(-22.0, 2.0),
            Complex::new(-11.0, 1.0),
            Complex::new(0.0, 0.0),
        ],
    );
    assert_eq!(result.valence_potential, result.total_potential);
    assert!(result.fermi_cache.is_empty());
    assert_eq!(result.density_grid, Some(density_grid));

    Ok(())
}

#[test]
fn xcpot_composed_mpse_computes_csigz_delta_table() -> Result<(), ExchangeError> {
    let radius_density = |radius: Real| 3.0 / (4.0 * FEFF_PI * radius.powi(3));
    let total = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
    let poison = ndarray::arr1(&[Real::NAN]);
    let density = ndarray::arr1(&[radius_density(2.1); 4]);
    let zeros = ndarray::Array1::<Real>::zeros(4);
    let frequencies = ndarray::arr1(&[0.35, 0.80, 1.60, -2.0]);
    let widths = ndarray::arr1(&[0.02, 0.03, 0.04, 0.0]);
    let amplitudes = ndarray::arr1(&[0.25, 0.40, 0.15, 0.0]);

    let result = xcpot(XcpotInput {
        exchange_selector: 0,
        lreal: 0,
        energy: Complex::new(0.40, 0.0),
        fermi_level: 0.05,
        total_potential: total.view(),
        valence_potential: poison.view(),
        density: density.view(),
        magnetization: zeros.view(),
        valence_density: zeros.view(),
        active_len: 4,
        plasmon_selector: 1,
        many_pole_delta_table: None,
        many_pole_self_energy: Some(XcpotManyPoleSelfEnergyInput {
            pole_frequencies: frequencies.view(),
            pole_widths: widths.view(),
            amplitudes: amplitudes.view(),
            gap_energy: 0.03,
            use_broadened_pole: false,
        }),
        fermi_cache: None,
    })?;

    assert_complex_near(
        result.reference_energy,
        Complex::new(4.027_905_769_387_554, -1.442_841_136_785_682_4e-2),
        1.0e-8,
    );
    assert_complex_slice_close(
        &result.total_potential,
        &[
            Complex::new(-3.0, 0.0),
            Complex::new(-2.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    );
    assert_eq!(result.valence_potential, result.total_potential);
    assert!(result.fermi_cache.is_empty());
    assert!(result.density_grid.is_some());

    Ok(())
}

#[test]
fn xcpot_composed_reports_missing_mpse_reference_data() {
    let total = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]);
    let poison = ndarray::arr1(&[Real::NAN]);
    let density = ndarray::arr1(&[0.04, 0.03, 0.02, 0.015]);
    let zeros = ndarray::Array1::<Real>::zeros(4);

    assert!(matches!(
        xcpot(XcpotInput {
            exchange_selector: 0,
            lreal: 0,
            energy: Complex::new(1.0, 0.0),
            fermi_level: 0.0,
            total_potential: total.view(),
            valence_potential: poison.view(),
            density: density.view(),
            magnetization: zeros.view(),
            valence_density: zeros.view(),
            active_len: 4,
            plasmon_selector: 1,
            many_pole_delta_table: None,
            many_pole_self_energy: None,
            fermi_cache: None,
        }),
        Err(ExchangeError::MissingReferenceData {
            name: "many_pole_self_energy",
            value: 1,
            data: "WpCorr/Gamma/AmpFac",
        })
    ));
}

#[test]
fn exchange_helpers_reject_invalid_inputs() {
    assert!(matches!(
        dirac_hara_exchange_potential(0.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_ffq(0.0, 1.0, 1.0, 1.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "q", .. })
    ));
    assert!(matches!(
        von_barth_hedin_potential(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "xmag", .. })
    ));
    assert!(matches!(
        perdew_zunger_vxc(0.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        perrot_dharma_wardana_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "temp", .. })
    ));
    assert!(matches!(
        perrot_dharma_wardana_reduced_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "t", .. })
    ));
    assert!(matches!(
        karasiev_sjostrom_dufty_trickey_vxc(1.0, -0.1),
        Err(ExchangeError::NegativeInput { name: "temp", .. })
    ));
    assert!(matches!(
        karasiev_sjostrom_dufty_trickey_free_energy(1.0, -0.1, KsdTSpin::Unpolarized,),
        Err(ExchangeError::NegativeInput { name: "t", .. })
    ));
    assert!(matches!(
        quinn_imaginary_self_energy(0.0, 1.0, 1.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "x", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_imaginary_self_energy(1.0, 0.0),
        Err(ExchangeError::NonPositiveInput { name: "xk", .. })
    ));
    assert!(matches!(
        hedin_lundqvist_self_energy(0.0, 1.0),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
}

#[test]
fn xcpot_many_pole_density_grid_rejects_invalid_inputs() {
    let density = ndarray::arr1(&[0.1, 0.2]);

    assert_eq!(
        xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
            plasmon_selector: 2,
            density: density.view(),
            radial_match_index_1based: 0,
        }),
        Err(ExchangeError::InvalidIndex {
            name: "radial_match_index_1based",
            index: 0,
        })
    );

    assert_eq!(
        xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
            plasmon_selector: 2,
            density: density.view(),
            radial_match_index_1based: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "density",
            required: 3,
            actual: 2,
        })
    );

    let nan_density = ndarray::arr1(&[0.1, Real::NAN]);
    assert!(matches!(
        xcpot_many_pole_density_grid(XcpotManyPoleDensityGridInput {
            plasmon_selector: 2,
            density: nan_density.view(),
            radial_match_index_1based: 1,
        }),
        Err(ExchangeError::NonFiniteInput { name: "density", value }) if value.is_nan()
    ));
}

#[test]
fn xcpot_many_pole_control_rejects_invalid_inputs() {
    let no_sentinel = ndarray::arr1(&[0.2, -1.0, 0.4]);
    assert_eq!(
        xcpot_many_pole_control(XcpotManyPoleControlInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            pole_frequencies: no_sentinel.view(),
        }),
        Err(ExchangeError::MissingManyPoleSentinel {
            name: "pole_frequencies",
            len: 3,
        })
    );

    let nan_before_sentinel = ndarray::arr1(&[0.2, Real::NAN, -2.0]);
    assert!(matches!(
        xcpot_many_pole_control(XcpotManyPoleControlInput {
            plasmon_selector: 1,
            exchange_selector: 0,
            pole_frequencies: nan_before_sentinel.view(),
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "pole_frequencies",
            value,
        }) if value.is_nan()
    ));

    let nan_after_sentinel = ndarray::arr1(&[0.2, -2.0, Real::NAN]);
    let control = xcpot_many_pole_control(XcpotManyPoleControlInput {
        plasmon_selector: 1,
        exchange_selector: 0,
        pole_frequencies: nan_after_sentinel.view(),
    })
    .expect("values after the FEFF sentinel are not part of the active prefix");
    assert_eq!(control.active_pole_count, 1);
}

#[test]
fn xcpot_many_pole_delta_table_rejects_invalid_inputs() {
    let fermi = xcpot_mpse_fermi_fixture();
    let energy = xcpot_mpse_energy_fixture();

    assert_eq!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 0,
            fermi_self_energy: fermi.view(),
            energy_self_energy: energy.view(),
            renormalization: Complex::new(1.0, 0.0),
        }),
        Err(ExchangeError::InvalidSelector {
            name: "plasmon_selector",
            value: 0,
        })
    );

    let short = ndarray::arr1(&[Complex::new(1.0, 0.0)]);
    assert_eq!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 2,
            fermi_self_energy: short.view(),
            energy_self_energy: energy.view(),
            renormalization: Complex::new(1.0, 0.0),
        }),
        Err(ExchangeError::LengthTooShort {
            name: "fermi_self_energy",
            required: XCPOT_MPSE_GRID_POINTS,
            actual: 1,
        })
    );
    assert_eq!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 2,
            fermi_self_energy: fermi.view(),
            energy_self_energy: short.view(),
            renormalization: Complex::new(1.0, 0.0),
        }),
        Err(ExchangeError::LengthTooShort {
            name: "energy_self_energy",
            required: XCPOT_MPSE_GRID_POINTS,
            actual: 1,
        })
    );

    assert!(matches!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 2,
            fermi_self_energy: fermi.view(),
            energy_self_energy: energy.view(),
            renormalization: Complex::new(Real::NAN, 0.0),
        }),
        Err(ExchangeError::NonFiniteComplex {
            name: "renormalization",
            value,
        }) if value.re.is_nan()
    ));

    let mut poisoned_fermi = fermi.clone();
    let mut poisoned_energy = energy.clone();
    poisoned_fermi[0] = Complex::new(Real::NAN, 0.0);
    poisoned_energy[0] = Complex::new(0.0, Real::NAN);
    assert!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 1,
            fermi_self_energy: poisoned_fermi.view(),
            energy_self_energy: poisoned_energy.view(),
            renormalization: Complex::new(1.0, 0.0),
        })
        .is_ok()
    );
    assert!(matches!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 2,
            fermi_self_energy: poisoned_fermi.view(),
            energy_self_energy: energy.view(),
            renormalization: Complex::new(1.0, 0.0),
        }),
        Err(ExchangeError::NonFiniteComplex {
            name: "fermi_self_energy",
            value,
        }) if value.re.is_nan()
    ));

    let mut poisoned_last = energy.clone();
    poisoned_last[XCPOT_MPSE_GRID_POINTS - 1] = Complex::new(0.0, Real::NAN);
    assert!(matches!(
        xcpot_many_pole_delta_table(XcpotManyPoleDeltaTableInput {
            plasmon_selector: 1,
            fermi_self_energy: fermi.view(),
            energy_self_energy: poisoned_last.view(),
            renormalization: Complex::new(1.0, 0.0),
        }),
        Err(ExchangeError::NonFiniteComplex {
            name: "energy_self_energy",
            value,
        }) if value.im.is_nan()
    ));
}

#[test]
fn xcpot_many_pole_row_delta_rejects_invalid_inputs() {
    let density_grid = xcpot_mpse_row_density_grid_fixture();
    let delta_table = xcpot_mpse_row_delta_table_fixture();

    assert_eq!(
        xcpot_many_pole_row_delta(XcpotManyPoleRowDeltaInput {
            plasmon_selector: 0,
            radius: 2.25,
            density_grid,
            delta_table,
        }),
        Err(ExchangeError::InvalidSelector {
            name: "plasmon_selector",
            value: 0,
        })
    );
    assert!(matches!(
        xcpot_many_pole_row_delta(XcpotManyPoleRowDeltaInput {
            plasmon_selector: 1,
            radius: 0.0,
            density_grid,
            delta_table,
        }),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));

    let mut poisoned_table = delta_table;
    poisoned_table.real[0] = Real::NAN;
    assert!(matches!(
        xcpot_many_pole_row_delta(XcpotManyPoleRowDeltaInput {
            plasmon_selector: 1,
            radius: 2.25,
            density_grid,
            delta_table: poisoned_table,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "delta_table.real",
            value,
        }) if value.is_nan()
    ));

    let mut poisoned_grid = density_grid;
    poisoned_grid.radii[0] = Real::NAN;
    assert!(matches!(
        xcpot_many_pole_row_delta(XcpotManyPoleRowDeltaInput {
            plasmon_selector: 2,
            radius: 1.25,
            density_grid: poisoned_grid,
            delta_table,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "density_grid.radii",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn xcpot_self_energy_delta_application_rejects_invalid_inputs() {
    let total = xcpot_real_total_fixture();
    let valence = xcpot_real_valence_fixture();
    let delta_real = xcpot_delta_real_fixture();
    let delta_imaginary = xcpot_delta_imaginary_fixture();
    let valence_delta_real = xcpot_valence_delta_real_fixture();
    let valence_delta_imaginary = xcpot_valence_delta_imaginary_fixture();

    assert_eq!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 0,
            total_potential: total.view(),
            valence_potential: valence.view(),
            delta_real: delta_real.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 0,
        }),
        Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: 0,
        })
    );

    let short = ndarray::arr1(&[1.0]);
    assert_eq!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 0,
            total_potential: short.view(),
            valence_potential: valence.view(),
            delta_real: delta_real.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "total_potential",
            required: 2,
            actual: 1,
        })
    );
    assert_eq!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 0,
            total_potential: total.view(),
            valence_potential: valence.view(),
            delta_real: short.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "delta_real",
            required: 2,
            actual: 1,
        })
    );
    assert_eq!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 6,
            total_potential: total.view(),
            valence_potential: short.view(),
            delta_real: delta_real.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "valence_potential",
            required: 2,
            actual: 1,
        })
    );

    let nan = ndarray::arr1(&[Real::NAN, 1.0]);
    assert!(matches!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 0,
            total_potential: nan.view(),
            valence_potential: valence.view(),
            delta_real: delta_real.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 2,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "total_potential",
            value,
        }) if value.is_nan()
    ));
    assert!(matches!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 0,
            total_potential: total.view(),
            valence_potential: valence.view(),
            delta_real: delta_real.view(),
            delta_imaginary: nan.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: valence_delta_imaginary.view(),
            active_len: 2,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "delta_imaginary",
            value,
        }) if value.is_nan()
    ));
    assert!(matches!(
        xcpot_apply_self_energy_deltas(XcpotSelfEnergyApplicationInput {
            exchange_selector: 5,
            total_potential: total.view(),
            valence_potential: valence.view(),
            delta_real: delta_real.view(),
            delta_imaginary: delta_imaginary.view(),
            valence_delta_real: valence_delta_real.view(),
            valence_delta_imaginary: nan.view(),
            active_len: 2,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "valence_delta_imaginary",
            value,
        }) if value.is_nan()
    ));
}

#[test]
fn xcpot_local_scales_rejects_invalid_inputs() {
    assert!(matches!(
        xcpot_local_scales(XcpotLocalScalesInput {
            exchange_selector: 0,
            density: Real::NAN,
            magnetization: 0.0,
            valence_density: 0.0,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "density",
            value,
        }) if value.is_nan()
    ));
    assert!(matches!(
        xcpot_local_scales(XcpotLocalScalesInput {
            exchange_selector: 0,
            density: 0.02,
            magnetization: Real::INFINITY,
            valence_density: 0.0,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "magnetization",
            value,
        }) if value.is_infinite()
    ));
    assert!(matches!(
        xcpot_local_scales(XcpotLocalScalesInput {
            exchange_selector: 5,
            density: 0.02,
            magnetization: 0.0,
            valence_density: Real::NAN,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "valence_density",
            value,
        }) if value.is_nan()
    ));
    assert_eq!(
        xcpot_local_scales(XcpotLocalScalesInput {
            exchange_selector: 0,
            density: 0.02,
            magnetization: -1.0,
            valence_density: 0.0,
        }),
        Err(ExchangeError::NonPositiveInput {
            name: "magnetization_factor",
            value: 0.0,
        })
    );
}

#[test]
fn xcpot_sigma_rejects_invalid_inputs() {
    assert!(matches!(
        xcpot_sigma(XcpotSigmaInput {
            exchange_selector: 0,
            radius: Real::NAN,
            core_radius: 101.0,
            momentum: 1.3,
        }),
        Err(ExchangeError::NonFiniteInput { name: "rs", .. })
    ));
    assert!(matches!(
        xcpot_sigma(XcpotSigmaInput {
            exchange_selector: 0,
            radius: 2.0,
            core_radius: 101.0,
            momentum: 0.0,
        }),
        Err(ExchangeError::NonPositiveInput { name: "xk", .. })
    ));
    assert!(matches!(
        xcpot_sigma(XcpotSigmaInput {
            exchange_selector: 6,
            radius: 2.0,
            core_radius: 0.0,
            momentum: 1.3,
        }),
        Err(ExchangeError::NonPositiveInput { name: "rscore", .. })
    ));
    assert!(matches!(
        xcpot_sigma(XcpotSigmaInput {
            exchange_selector: 2,
            radius: 2.0,
            core_radius: 101.0,
            momentum: 1.3,
        }),
        Err(ExchangeError::InvalidSelector {
            name: "exchange_selector % 10",
            value: 2,
        })
    ));
    assert!(matches!(
        xcpot_sigma(XcpotSigmaInput {
            exchange_selector: 10,
            radius: 2.0,
            core_radius: 101.0,
            momentum: 1.3,
        }),
        Err(ExchangeError::MissingReferenceData {
            name: "index / 10",
            value: 1,
            data: "bphl.dat",
        })
    ));
}

#[test]
fn xcpot_fermi_cache_rejects_invalid_inputs() {
    assert!(matches!(
        xcpot_fermi_cache(XcpotFermiCacheInput {
            exchange_selector: 0,
            radius: 0.0,
            core_radius: 101.0,
            valence_radius: None,
            interstitial: false,
        }),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        xcpot_fermi_cache(XcpotFermiCacheInput {
            exchange_selector: 5,
            radius: 2.0,
            core_radius: 101.0,
            valence_radius: None,
            interstitial: false,
        }),
        Err(ExchangeError::MissingRequiredInput {
            name: "valence_radius",
            value: 5,
        })
    ));
    assert!(matches!(
        xcpot_fermi_cache(XcpotFermiCacheInput {
            exchange_selector: 5,
            radius: 2.0,
            core_radius: 101.0,
            valence_radius: Some(0.0),
            interstitial: false,
        }),
        Err(ExchangeError::NonPositiveInput {
            name: "valence_radius",
            ..
        })
    ));
    assert!(matches!(
        xcpot_fermi_cache(XcpotFermiCacheInput {
            exchange_selector: 15,
            radius: 2.0,
            core_radius: 101.0,
            valence_radius: Some(3.0),
            interstitial: false,
        }),
        Err(ExchangeError::MissingReferenceData {
            name: "index / 10",
            value: 1,
            data: "bphl.dat",
        })
    ));
}

#[test]
fn xcpot_self_energy_correction_rejects_invalid_inputs() {
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            radius: 0.0,
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::NonPositiveInput { name: "rs", .. })
    ));
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            energy: -1.0,
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::NegativeRadicand {
            name: "initial_momentum",
            ..
        })
    ));
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            exchange_selector: 5,
            valence_radius: None,
            valence_fermi_momentum: Some(FEFF_FA / 3.0),
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::MissingRequiredInput {
            name: "valence_radius",
            value: 5,
        })
    ));
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            exchange_selector: 5,
            valence_radius: Some(3.0),
            valence_fermi_momentum: None,
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::MissingRequiredInput {
            name: "valence_fermi_momentum",
            value: 5,
        })
    ));
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            fermi_cache: XcpotFermiCache {
                total_self_energy: XcpotSigma {
                    real: Real::NAN,
                    imaginary: 0.0,
                },
                ..xcpot_valid_correction_input().fermi_cache
            },
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "total_self_energy.real",
            value,
        }) if value.is_nan()
    ));
    assert!(matches!(
        xcpot_self_energy_correction(XcpotSelfEnergyCorrectionInput {
            exchange_selector: 10,
            ..xcpot_valid_correction_input()
        }),
        Err(ExchangeError::MissingReferenceData {
            name: "index / 10",
            value: 1,
            data: "bphl.dat",
        })
    ));
}

#[test]
fn xcpot_reference_shift_rejects_invalid_inputs() {
    let total = xcpot_total_fixture();
    let valence = xcpot_valence_fixture();

    assert_eq!(
        xcpot_reference_shift(XcpotReferenceShiftInput {
            exchange_selector: 0,
            lreal: 0,
            total_potential: total.view(),
            valence_potential: valence.view(),
            active_len: 0,
        }),
        Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: 0,
        })
    );

    let short = ndarray::arr1(&[Complex::new(1.0, 0.0)]);
    assert_eq!(
        xcpot_reference_shift(XcpotReferenceShiftInput {
            exchange_selector: 0,
            lreal: 0,
            total_potential: short.view(),
            valence_potential: valence.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "total_potential",
            required: 2,
            actual: 1,
        })
    );

    assert_eq!(
        xcpot_reference_shift(XcpotReferenceShiftInput {
            exchange_selector: 6,
            lreal: 0,
            total_potential: total.view(),
            valence_potential: short.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "valence_potential",
            required: 2,
            actual: 1,
        })
    );

    let nan_total = ndarray::arr1(&[Complex::new(Real::NAN, 0.0)]);
    assert!(matches!(
        xcpot_reference_shift(XcpotReferenceShiftInput {
            exchange_selector: 0,
            lreal: 0,
            total_potential: nan_total.view(),
            valence_potential: valence.view(),
            active_len: 1,
        }),
        Err(ExchangeError::NonFiniteComplex {
            name: "total_potential",
            value,
        }) if value.re.is_nan()
    ));

    let nan_valence = ndarray::arr1(&[Complex::new(Real::NAN, 0.0)]);
    assert!(matches!(
        xcpot_reference_shift(XcpotReferenceShiftInput {
            exchange_selector: 6,
            lreal: 0,
            total_potential: total.view(),
            valence_potential: nan_valence.view(),
            active_len: 1,
        }),
        Err(ExchangeError::NonFiniteComplex {
            name: "valence_potential",
            value,
        }) if value.re.is_nan()
    ));
}

#[test]
fn xcpot_ground_state_branch_rejects_invalid_inputs() {
    let total = xcpot_real_total_fixture();
    let valence = xcpot_real_valence_fixture();

    assert_eq!(
        xcpot_ground_state_branch(XcpotGroundStateBranchInput {
            exchange_selector: 2,
            lreal: 0,
            energy: Complex::new(8.5, 0.25),
            fermi_level: 2.0,
            total_potential: total.view(),
            valence_potential: valence.view(),
            active_len: 0,
        }),
        Err(ExchangeError::InvalidIndex {
            name: "active_len",
            index: 0,
        })
    );

    let short = ndarray::arr1(&[1.0]);
    assert_eq!(
        xcpot_ground_state_branch(XcpotGroundStateBranchInput {
            exchange_selector: 2,
            lreal: 0,
            energy: Complex::new(8.5, 0.25),
            fermi_level: 2.0,
            total_potential: short.view(),
            valence_potential: valence.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "total_potential",
            required: 2,
            actual: 1,
        })
    );

    assert_eq!(
        xcpot_ground_state_branch(XcpotGroundStateBranchInput {
            exchange_selector: 6,
            lreal: 0,
            energy: Complex::new(1.5, 0.25),
            fermi_level: 2.0,
            total_potential: total.view(),
            valence_potential: short.view(),
            active_len: 2,
        }),
        Err(ExchangeError::LengthTooShort {
            name: "valence_potential",
            required: 2,
            actual: 1,
        })
    );

    let nan_total = ndarray::arr1(&[Real::NAN]);
    assert!(matches!(
        xcpot_ground_state_branch(XcpotGroundStateBranchInput {
            exchange_selector: 2,
            lreal: 0,
            energy: Complex::new(8.5, 0.25),
            fermi_level: 2.0,
            total_potential: nan_total.view(),
            valence_potential: valence.view(),
            active_len: 1,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "total_potential",
            value,
        }) if value.is_nan()
    ));

    let nan_valence = ndarray::arr1(&[Real::NAN]);
    assert!(matches!(
        xcpot_ground_state_branch(XcpotGroundStateBranchInput {
            exchange_selector: 6,
            lreal: 0,
            energy: Complex::new(1.5, 0.25),
            fermi_level: 2.0,
            total_potential: total.view(),
            valence_potential: nan_valence.view(),
            active_len: 1,
        }),
        Err(ExchangeError::NonFiniteInput {
            name: "valence_potential",
            value,
        }) if value.is_nan()
    ));
}

fn xcpot_total_fixture() -> ndarray::Array1<Complex> {
    ndarray::arr1(&[
        Complex::new(10.0, -1.0),
        Complex::new(11.5, -0.5),
        Complex::new(13.25, 0.25),
        Complex::new(15.0, 1.5),
        Complex::new(99.0, 99.0),
    ])
}

fn xcpot_real_total_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[10.0, 11.5, 13.25, 15.0, 99.0])
}

fn xcpot_real_valence_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[20.0, 21.5, 23.25, 25.0, 199.0])
}

fn xcpot_valence_fixture() -> ndarray::Array1<Complex> {
    ndarray::arr1(&[
        Complex::new(20.0, -2.0),
        Complex::new(21.5, -1.0),
        Complex::new(23.25, 0.5),
        Complex::new(25.0, 2.5),
        Complex::new(199.0, 199.0),
    ])
}

fn xcpot_delta_real_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[0.10, -0.20, 0.35, 0.50, 9.0])
}

fn xcpot_delta_imaginary_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[-0.01, -0.02, 0.03, 0.04, 9.0])
}

fn xcpot_valence_delta_real_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[1.10, 1.20, -1.30, -1.40, 9.0])
}

fn xcpot_valence_delta_imaginary_fixture() -> ndarray::Array1<Real> {
    ndarray::arr1(&[0.21, 0.22, -0.23, -0.24, 9.0])
}

fn xcpot_mpse_fermi_fixture() -> ndarray::Array1<Complex> {
    ndarray::Array1::from_iter((1..=XCPOT_MPSE_GRID_POINTS).map(|index| {
        let index = index as Real;
        Complex::new(0.2 * index - 0.05, -0.04 * index + 0.02)
    }))
}

fn xcpot_mpse_energy_fixture() -> ndarray::Array1<Complex> {
    ndarray::Array1::from_iter((1..=XCPOT_MPSE_GRID_POINTS).map(|index| {
        let index = index as Real;
        Complex::new(0.9 + 0.31 * index, -0.7 + 0.08 * index)
    }))
}

fn xcpot_mpse_row_density_grid_fixture() -> XcpotManyPoleDensityGrid {
    XcpotManyPoleDensityGrid {
        interstitial_radius: 10.0,
        core_radius: 1.0,
        min_radius: 1.0,
        max_radius: 10.0,
        radius_step: 1.0,
        radii: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
}

fn xcpot_mpse_row_delta_table_fixture() -> XcpotManyPoleDeltaTable {
    let real = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let imaginary = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0];
    XcpotManyPoleDeltaTable {
        fermi_self_energy: [Complex::new(0.0, 0.0); XCPOT_MPSE_GRID_POINTS],
        delta_self_energy: real.map(|value| Complex::new(value, -0.1 * value)),
        real,
        imaginary,
    }
}

fn xcpot_correction_input(
    exchange_selector: i32,
    interstitial: bool,
    energy: Real,
    core_radius: Real,
    valence_radius: Option<Real>,
    magnetization: Real,
) -> Result<XcpotSelfEnergyCorrectionInput, ExchangeError> {
    let radius = 2.0;
    let fermi_momentum = FEFF_FA / radius;
    let valence_fermi_momentum = valence_radius.map(|radius| FEFF_FA / radius);
    let fermi_cache = xcpot_fermi_cache(XcpotFermiCacheInput {
        exchange_selector,
        radius,
        core_radius,
        valence_radius,
        interstitial,
    })?;

    Ok(XcpotSelfEnergyCorrectionInput {
        exchange_selector,
        energy,
        fermi_level: 0.1,
        radius,
        core_radius,
        fermi_momentum,
        magnetized_fermi_momentum: fermi_momentum * (1.0 + magnetization).powf(1.0 / 3.0),
        valence_radius,
        valence_fermi_momentum,
        interstitial,
        fermi_cache,
    })
}

fn xcpot_valid_correction_input() -> XcpotSelfEnergyCorrectionInput {
    xcpot_correction_input(0, false, 1.2, 101.0, None, 0.0).expect("valid xcpot correction fixture")
}

fn assert_xcpot_radii_close(actual: &[Real; XCPOT_MPSE_GRID_POINTS], expected: &[Real]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_real_close(actual, expected);
    }
}

fn assert_xcpot_real_table_close(
    actual: &[Real; XCPOT_MPSE_GRID_POINTS],
    expected: &[Real; XCPOT_MPSE_GRID_POINTS],
) {
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_real_close(actual, expected);
    }
}

fn assert_xcpot_complex_table_close(
    actual: &[Complex; XCPOT_MPSE_GRID_POINTS],
    expected: &[Complex; XCPOT_MPSE_GRID_POINTS],
) {
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
}

fn assert_complex_table_near(
    actual: &[Complex; XCPOT_MPSE_GRID_POINTS],
    expected: &[Complex; XCPOT_MPSE_GRID_POINTS],
    tolerance: Real,
) {
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_real_near(actual.re, expected.re, tolerance);
        assert_real_near(actual.im, expected.im, tolerance);
    }
}

fn assert_complex_slice_close(actual: &ndarray::Array1<Complex>, expected: &[Complex]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected);
    }
}

fn assert_xcpot_local_scale_close(actual: XcpotLocalScales, expected: XcpotLocalScales) {
    assert_real_close(actual.radius, expected.radius);
    assert_real_close(actual.fermi_momentum, expected.fermi_momentum);
    assert_real_close(actual.magnetized_radius, expected.magnetized_radius);
    assert_real_close(
        actual.magnetized_fermi_momentum,
        expected.magnetized_fermi_momentum,
    );
    assert_optional_real_close(actual.valence_radius, expected.valence_radius);
    assert_optional_real_close(
        actual.valence_fermi_momentum,
        expected.valence_fermi_momentum,
    );
    assert_optional_real_close(actual.core_radius, expected.core_radius);
}

fn assert_xcpot_sigma_near(actual: XcpotSigma, expected: XcpotSigma) {
    assert_real_near(actual.real, expected.real, 1.0e-12);
    assert_real_near(actual.imaginary, expected.imaginary, 1.0e-12);
}

fn assert_xcpot_fermi_cache_near(actual: XcpotFermiCache, expected: XcpotFermiCache) {
    assert_xcpot_sigma_near(actual.total_self_energy, expected.total_self_energy);
    assert_xcpot_sigma_near(actual.valence_self_energy, expected.valence_self_energy);
    assert_real_close(actual.ground_state_ratio, expected.ground_state_ratio);
}

fn assert_xcpot_self_energy_correction_near(
    actual: XcpotSelfEnergyCorrection,
    expected: XcpotSelfEnergyCorrection,
) {
    assert_real_near(
        actual.magnetized_momentum,
        expected.magnetized_momentum,
        1.0e-12,
    );
    assert_real_near(
        actual.corrected_momentum,
        expected.corrected_momentum,
        1.0e-12,
    );
    assert_xcpot_sigma_near(actual.total_delta, expected.total_delta);
    assert_optional_xcpot_sigma_near(actual.valence_delta, expected.valence_delta);
}

fn assert_optional_xcpot_sigma_near(actual: Option<XcpotSigma>, expected: Option<XcpotSigma>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_xcpot_sigma_near(actual, expected),
        (None, None) => {}
        _ => panic!("actual={actual:?}, expected={expected:?}"),
    }
}

fn assert_optional_real_close(actual: Option<Real>, expected: Option<Real>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert_real_close(actual, expected),
        (None, None) => {}
        _ => panic!("actual={actual:?}, expected={expected:?}"),
    }
}
