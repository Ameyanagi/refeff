use super::{support::*, *};

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
