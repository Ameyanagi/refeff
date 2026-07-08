use super::{support::*, *};
use crate::exchange::ExchangeError;

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
fn muffin_tin_radius_parameters_match_feff_istprm_explicit_reference() -> Result<(), GridError> {
    let atom_potentials = Array1::from_vec(vec![0, 1]);
    let atom_positions = Array2::<Real>::zeros((2, 3));
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
    let overlap_factors = Array1::from_vec(vec![1.0, 1.0]);
    let max_overlap_factors = Array1::from_vec(vec![1.15, 1.15]);
    let coulomb = sample_istprm_coulomb_table();
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
    let explicit: [&[MuffinTinOverlapNeighbor]; 2] = [&neighbors0, &neighbors1];

    let result = muffin_tin_radius_parameters(MuffinTinRadiusParametersInput {
        highest_potential_index: 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        explicit_overlaps: &explicit,
        norman_radii: norman_radii.view(),
        overlap_factors: overlap_factors.view(),
        max_overlap_factors: max_overlap_factors.view(),
        coulomb_potential: coulomb.view(),
        afolp_enabled: false,
        interstitial_selector: 8,
    })?;

    assert_eq!(result.interstitial_selector, 2);
    assert_eq!(result.norman_indices, Array1::from_vec(vec![93, 96]));
    assert_eq!(
        result.nearest_neighbor_potentials,
        Array1::from_vec(vec![1, 0])
    );
    assert_eq!(
        result.near_neighbor_flags,
        Array1::from_vec(vec![false, false])
    );
    assert_eq!(
        result.norman_radius_fallbacks,
        Array1::from_vec(vec![false, false])
    );
    assert_close_with_tolerance(result.nearest_neighbor_distances[0], 0.030, 1e-14);
    assert_close_with_tolerance(result.nearest_neighbor_distances[1], 0.031, 1e-14);
    assert_close_with_tolerance(
        result.muffin_tin_radii[0],
        1.363_636_363_636_363_6e-2,
        1e-14,
    );
    assert_close_with_tolerance(
        result.muffin_tin_radii[1],
        1.690_909_090_909_090_5e-2,
        1e-14,
    );
    assert_close_with_tolerance(result.max_overlap_factors[0], 1.07, 1e-14);
    assert_close_with_tolerance(
        result.max_overlap_factors[1],
        1.045_161_290_322_580_6,
        1e-14,
    );
    Ok(())
}

#[test]
fn muffin_tin_radius_parameters_match_feff_istprm_geometry_reference() -> Result<(), GridError> {
    let atom_potentials = Array1::from_vec(vec![0, 1]);
    let atom_positions =
        Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.030, 0.0, 0.0]).unwrap();
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
    let overlap_factors = Array1::from_vec(vec![1.0, 1.0]);
    let max_overlap_factors = Array1::from_vec(vec![1.15, 1.15]);
    let coulomb = sample_istprm_coulomb_table();
    let empty0: [MuffinTinOverlapNeighbor; 0] = [];
    let empty1: [MuffinTinOverlapNeighbor; 0] = [];
    let explicit: [&[MuffinTinOverlapNeighbor]; 2] = [&empty0, &empty1];

    let result = muffin_tin_radius_parameters(MuffinTinRadiusParametersInput {
        highest_potential_index: 1,
        atom_potentials: atom_potentials.view(),
        atom_positions: atom_positions.view(),
        representative_atoms: representative_atoms.view(),
        explicit_overlaps: &explicit,
        norman_radii: norman_radii.view(),
        overlap_factors: overlap_factors.view(),
        max_overlap_factors: max_overlap_factors.view(),
        coulomb_potential: coulomb.view(),
        afolp_enabled: false,
        interstitial_selector: 0,
    })?;

    assert_eq!(result.interstitial_selector, 0);
    assert_eq!(
        result.nearest_neighbor_potentials,
        Array1::from_vec(vec![1, 0])
    );
    assert_close_with_tolerance(result.nearest_neighbor_distances[0], 0.030, 1e-14);
    assert_close_with_tolerance(result.nearest_neighbor_distances[1], 0.030, 1e-14);
    assert_close_with_tolerance(
        result.muffin_tin_radii[0],
        1.363_636_363_636_363_6e-2,
        1e-14,
    );
    assert_close_with_tolerance(result.muffin_tin_radii[1], 1.636_363_636_363_636e-2, 1e-14);
    assert_close_with_tolerance(result.max_overlap_factors[0], 1.07, 1e-14);
    assert_close_with_tolerance(result.max_overlap_factors[1], 1.07, 1e-14);
    Ok(())
}

#[test]
fn muffin_tin_radius_parameters_reject_invalid_explicit_neighbor() {
    let atom_potentials = Array1::from_vec(vec![0, 1]);
    let atom_positions = Array2::<Real>::zeros((2, 3));
    let representative_atoms = Array1::from_vec(vec![0, 1]);
    let norman_radii = Array1::from_vec(vec![0.015, 0.018]);
    let overlap_factors = Array1::from_vec(vec![1.0, 1.0]);
    let max_overlap_factors = Array1::from_vec(vec![1.15, 1.15]);
    let coulomb = sample_istprm_coulomb_table();
    let bad_neighbors = [MuffinTinOverlapNeighbor {
        source_potential: 2,
        multiplicity: 1,
        distance: 0.030,
    }];
    let empty: [MuffinTinOverlapNeighbor; 0] = [];
    let explicit: [&[MuffinTinOverlapNeighbor]; 2] = [&bad_neighbors, &empty];

    assert_eq!(
        muffin_tin_radius_parameters(MuffinTinRadiusParametersInput {
            highest_potential_index: 1,
            atom_potentials: atom_potentials.view(),
            atom_positions: atom_positions.view(),
            representative_atoms: representative_atoms.view(),
            explicit_overlaps: &explicit,
            norman_radii: norman_radii.view(),
            overlap_factors: overlap_factors.view(),
            max_overlap_factors: max_overlap_factors.view(),
            coulomb_potential: coulomb.view(),
            afolp_enabled: false,
            interstitial_selector: 0,
        }),
        Err(GridError::InvalidPotentialIndex {
            name: "explicit_overlaps.source_potential",
            index: 2,
            available: 2,
        })
    );
}

#[test]
fn muffin_tin_interstitial_parameters_match_feff_istprm_reference() -> Result<(), GridError> {
    let sample = sample_istprm_interstitial_state();
    let explicit = sample.explicit_overlaps();

    let result = muffin_tin_interstitial_parameters(sample.input(&explicit, 10.0, 12))?;

    assert_eq!(result.max_density_indices, Array1::from_vec(vec![250, 250]));
    assert_eq!(result.muffin_tin_indices, Array1::from_vec(vec![98, 102]));
    assert_eq!(result.norman_indices, Array1::from_vec(vec![120, 123]));
    assert_close_with_tolerance(
        result.average_norman_radius,
        6.554_735_680_074_165e-2,
        1e-14,
    );
    assert_close_with_tolerance(result.interstitial_volume, 3.389_636_054_356_424e-3, 1e-14);
    assert!(!result.interstitial_potential_limited);
    assert!(result.interstitial_density > 0.0);
    assert_fermi_level(
        result.fermi,
        result.interstitial_potential + result.fermi.fermi_momentum.powi(2) / 2.0,
        (3.0 / result.interstitial_density).powf(1.0 / 3.0),
        FEFF_FERMI_MOMENTUM_FACTOR / result.fermi.density_parameter,
    );

    let rs = (sample.electron_density[(0, 0)] / 3.0).powf(-1.0 / 3.0);
    let expected_first_total =
        sample.coulomb_potential[(0, 0)] + crate::exchange::perdew_zunger_vxc(rs)?;
    assert_close_with_tolerance(result.total_potential[(0, 0)], expected_first_total, 1e-12);
    assert_close_with_tolerance(
        result.total_potential[(110, 0)],
        result.interstitial_potential,
        1e-12,
    );
    assert!(result.valence_potential.iter().all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn muffin_tin_interstitial_parameters_applies_feff_vint_limit() -> Result<(), GridError> {
    let sample = sample_istprm_interstitial_state();
    let explicit = sample.explicit_overlaps();

    let result = muffin_tin_interstitial_parameters(sample.input(&explicit, -10.0, 12))?;

    assert!(result.interstitial_potential_limited);
    assert_close_with_tolerance(result.interstitial_potential, -10.05, 1e-12);
    assert_close_with_tolerance(result.total_potential[(110, 0)], -10.05, 1e-12);
    assert_close_with_tolerance(
        result.fermi.chemical_potential,
        -10.05 + result.fermi.fermi_momentum.powi(2) / 2.0,
        1e-12,
    );
    Ok(())
}

#[test]
fn muffin_tin_interstitial_parameters_rejects_invalid_scf_exchange_selector() {
    let sample = sample_istprm_interstitial_state();
    let explicit = sample.explicit_overlaps();

    assert!(matches!(
        muffin_tin_interstitial_parameters(sample.input(&explicit, 10.0, 99)),
        Err(GridError::Exchange(ExchangeError::InvalidSelector {
            name: "iscfxc",
            value: 99,
        }))
    ));
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

fn sample_istprm_coulomb_table() -> Array2<Real> {
    Array2::from_shape_fn((251, 2), |(radial, potential)| {
        -2.0 + 0.001 * radial as Real + 0.05 * potential as Real
    })
}

#[derive(Debug, Clone)]
struct IstprmInterstitialSample {
    atom_potentials: Array1<usize>,
    atom_positions: Array2<Real>,
    representative_atoms: Array1<usize>,
    potential_multiplicities: Array1<Real>,
    neighbors0: [MuffinTinOverlapNeighbor; 1],
    neighbors1: [MuffinTinOverlapNeighbor; 1],
    electron_density: Array2<Real>,
    valence_density: Array2<Real>,
    magnetization: Array2<Real>,
    coulomb_potential: Array2<Real>,
    muffin_tin_radii: Array1<Real>,
    norman_radii: Array1<Real>,
    near_neighbor_flags: Array1<bool>,
}

impl IstprmInterstitialSample {
    fn explicit_overlaps(&self) -> [&[MuffinTinOverlapNeighbor]; 2] {
        [&self.neighbors0, &self.neighbors1]
    }

    fn input<'a>(
        &'a self,
        explicit_overlaps: &'a [&'a [MuffinTinOverlapNeighbor]],
        fermi_level: Real,
        scf_exchange_selector: i32,
    ) -> MuffinTinInterstitialParametersInput<'a> {
        MuffinTinInterstitialParametersInput {
            highest_potential_index: 1,
            atom_potentials: self.atom_potentials.view(),
            atom_positions: self.atom_positions.view(),
            representative_atoms: self.representative_atoms.view(),
            potential_multiplicities: self.potential_multiplicities.view(),
            explicit_overlaps,
            electron_density: self.electron_density.view(),
            valence_density: self.valence_density.view(),
            magnetization: self.magnetization.view(),
            coulomb_potential: self.coulomb_potential.view(),
            muffin_tin_radii: self.muffin_tin_radii.view(),
            norman_radii: self.norman_radii.view(),
            near_neighbor_flags: self.near_neighbor_flags.view(),
            exchange_selector: 2,
            scf_exchange_selector,
            spin_polarization: 0,
            scf_temperature_hartree: 0.0,
            total_charge: 10.0,
            fermi_level,
            total_volume: 0.0,
            interstitial_selector: 0,
        }
    }
}

fn sample_istprm_interstitial_state() -> IstprmInterstitialSample {
    IstprmInterstitialSample {
        atom_potentials: Array1::from_vec(vec![0, 1]),
        atom_positions: Array2::<Real>::zeros((2, 3)),
        representative_atoms: Array1::from_vec(vec![0, 1]),
        potential_multiplicities: Array1::from_vec(vec![1.0, 2.0]),
        neighbors0: [MuffinTinOverlapNeighbor {
            source_potential: 1,
            multiplicity: 2,
            distance: 0.090,
        }],
        neighbors1: [MuffinTinOverlapNeighbor {
            source_potential: 0,
            multiplicity: 1,
            distance: 0.092,
        }],
        electron_density: Array2::from_shape_fn((251, 2), |(row, potential)| {
            0.4 + 0.001 * (row + 1) as Real + 0.05 * potential as Real
        }),
        valence_density: Array2::from_shape_fn((251, 2), |(row, potential)| {
            0.18 + 0.0005 * (row + 1) as Real + 0.02 * potential as Real
        }),
        magnetization: Array2::from_shape_fn((251, 2), |(row, potential)| {
            0.01 * potential as Real + 1.0e-5 * row as Real
        }),
        coulomb_potential: Array2::from_shape_fn((251, 2), |(row, potential)| {
            -1.2 + 0.002 * row as Real + 0.04 * potential as Real
        }),
        muffin_tin_radii: Array1::from_vec(vec![0.020, 0.024]),
        norman_radii: Array1::from_vec(vec![0.060, 0.068]),
        near_neighbor_flags: Array1::from_vec(vec![false, false]),
    }
}
