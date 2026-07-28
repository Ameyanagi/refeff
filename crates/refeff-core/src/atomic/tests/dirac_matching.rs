#![allow(clippy::excessive_precision)]

use super::*;

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_integration_seeds_match_feff_soldir_reference() -> Result<(), AtomMathError> {
    let radial_count = 8;
    let coefficient_count = 5;
    let large_source = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.05 * index + 0.003 * index * index
    });
    let small_source = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.04 * index + 0.002 * index * index
    });
    let large_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.11 * index - 0.004 * index * index
    });
    let small_source_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.09 * index + 0.005 * index * index
    });

    let inhomogeneous = atomic_dirac_inhomogeneous_seed(AtomicDiracInhomogeneousSeedInput {
        large_source: large_source.view(),
        small_source: small_source.view(),
        large_source_coefficients: large_source_coefficients.view(),
        small_source_coefficients: small_source_coefficients.view(),
        coefficient_count,
    })?;
    assert_close_with(
        inhomogeneous.large_source[0],
        5.300_000_000_000_000_5e-2,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.large_source[7],
        5.920_000_000_000_000_8e-1,
        1.0e-17,
    );
    assert_close_with(
        inhomogeneous.small_source[4],
        -1.500_000_000_000_000_2e-1,
        1.0e-18,
    );
    assert_close_with(inhomogeneous.large_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        inhomogeneous.large_coefficients[1],
        1.060_000_000_000_000_0e-1,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.large_coefficients[4],
        3.760_000_000_000_000_0e-1,
        1.0e-18,
    );
    assert_close_with(inhomogeneous.small_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        inhomogeneous.small_coefficients[1],
        -8.499_999_999_999_999_2e-2,
        1.0e-18,
    );
    assert_close_with(
        inhomogeneous.small_coefficients[4],
        -2.799_999_999_999_999_7e-1,
        1.0e-18,
    );

    let homogeneous = atomic_dirac_homogeneous_seed(AtomicDiracHomogeneousSeedInput {
        radial_len: radial_count,
        coefficient_len: coefficient_count,
    })?;
    assert_eq!(homogeneous.large_source.len(), radial_count);
    assert_eq!(homogeneous.small_source.len(), radial_count);
    assert_eq!(homogeneous.large_coefficients.len(), coefficient_count);
    assert_eq!(homogeneous.small_coefficients.len(), coefficient_count);
    assert!(homogeneous.large_source.iter().all(|&value| value == 0.0));
    assert!(homogeneous.small_source.iter().all(|&value| value == 0.0));
    assert!(
        homogeneous
            .large_coefficients
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        homogeneous
            .small_coefficients
            .iter()
            .all(|&value| value == 0.0)
    );
    Ok(())
}

#[test]
fn atom_dirac_inhomogeneous_branch_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let homogeneous_request =
        atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
            requested_method: 0,
        })?;
    assert_eq!(
        homogeneous_request.action,
        AtomicDiracInhomogeneousBranchAction::MatchHomogeneousTail
    );

    let method1_request = atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
        requested_method: 1,
    })?;
    assert_eq!(
        method1_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );

    let method2_request = atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
        requested_method: 2,
    })?;
    assert_eq!(
        method2_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );

    let negative_request =
        atomic_dirac_inhomogeneous_branch(AtomicDiracInhomogeneousBranchInput {
            requested_method: -1,
        })?;
    assert_eq!(
        negative_request.action,
        AtomicDiracInhomogeneousBranchAction::IntegrateHomogeneousSystem
    );
    Ok(())
}

#[test]
fn atom_dirac_homogeneous_pass_setup_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let method1 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 1 })?;
    assert_eq!(
        method1.integration_mode,
        AtomicDiracIntegrationMode::InwardOnly
    );
    assert_eq!(method1.raw_integration_flag, -1);

    let method2 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 2 })?;
    assert_eq!(
        method2.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(method2.raw_integration_flag, 1);

    let method3 =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: 3 })?;
    assert_eq!(
        method3.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(method3.raw_integration_flag, 1);

    let negative =
        atomic_dirac_homogeneous_pass_setup(AtomicDiracHomogeneousPassSetupInput { method: -2 })?;
    assert_eq!(
        negative.integration_mode,
        AtomicDiracIntegrationMode::FixedMatchingPoint
    );
    assert_eq!(negative.raw_integration_flag, 1);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_matching_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let large_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        0.08 * index - 0.006 * index * index
    });
    let small_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        -0.025 * index + 0.0015 * index * index
    });
    let homogeneous_large_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        0.018 * index + 0.0007 * index * index
    });
    let homogeneous_small_component = Array1::from_shape_fn(8, |row| {
        let index = (row + 1) as Real;
        -0.012 * index + 0.0004 * index * index
    });

    let homogeneous_match = atomic_dirac_homogeneous_match(AtomicDiracHomogeneousMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        matching_large_component: 0.240,
        active_len: 7,
        matching_index_1based: 4,
    })?;
    assert_close_with(
        homogeneous_match.tail_scale,
        1.071_428_571_428_571_4,
        1.0e-16,
    );
    assert_eq!(homogeneous_match.scan_index_1based, 4);
    assert_close_with(homogeneous_match.large_component[0], 7.4e-2, 1.0e-18);
    assert_close_with(
        homogeneous_match.large_component[3],
        2.399_999_999_999_999_9e-1,
        1.0e-18,
    );
    assert_close_with(
        homogeneous_match.large_component[6],
        2.850_000_000_000_000_3e-1,
        1.0e-16,
    );
    assert_close_with(
        homogeneous_match.large_component[7],
        large_component[7],
        1.0e-18,
    );
    assert_close_with(
        homogeneous_match.small_component[3],
        -8.142_857_142_857_143_3e-2,
        1.0e-17,
    );
    assert_close_with(
        homogeneous_match.small_component[6],
        -1.087_500_000_000_000_0e-1,
        1.0e-17,
    );

    let large_match = atomic_dirac_large_component_match(AtomicDiracLargeComponentMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        homogeneous_large_component: homogeneous_large_component.view(),
        homogeneous_small_component: homogeneous_small_component.view(),
        matching_large_component: 0.240,
        active_len: 7,
        matching_index_1based: 4,
    })?;
    assert_close_with(large_match.tail_scale, 1.923_076_923_076_921_5e-1, 1.0e-16);
    assert_close_with(large_match.large_mismatch, -1.6e-2, 1.0e-16);
    assert_close_with(large_match.large_component[3], 2.4e-1, 1.0e-18);
    assert_close_with(
        large_match.large_component[6],
        2.968_269_230_769_230_4e-1,
        1.0e-16,
    );
    assert_close_with(
        large_match.small_component[6],
        -1.138_846_153_846_153_9e-1,
        1.0e-16,
    );
    assert_close_with(large_match.large_component[7], large_component[7], 1.0e-18);

    let large_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        0.11 * index - 0.004 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        -0.07 * index + 0.003 * index * index
    });
    let homogeneous_large_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        0.012 * index + 0.0005 * index * index
    });
    let homogeneous_small_coefficients = Array1::from_shape_fn(4, |row| {
        let index = (row + 1) as Real;
        -0.009 * index + 0.0003 * index * index
    });

    let two_match = atomic_dirac_two_component_match(AtomicDiracTwoComponentMatchInput {
        large_component: large_component.view(),
        small_component: small_component.view(),
        large_coefficients: large_coefficients.view(),
        small_coefficients: small_coefficients.view(),
        homogeneous_large_component: homogeneous_large_component.view(),
        homogeneous_small_component: homogeneous_small_component.view(),
        homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
        homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
        matching_large_component: 0.285,
        matching_small_component: -0.068,
        homogeneous_matching_large_component: 0.087,
        homogeneous_matching_small_component: -0.047,
        coefficient_count: 4,
        active_len: 8,
        matching_index_1based: 5,
    })?;
    assert_close_with(two_match.determinant, -7.025e-4, 1.0e-18);
    assert_close_with(two_match.tail_scale, 4.756_583_629_893_235_4, 1.0e-15);
    assert_close_with(two_match.prefix_scale, 5.475_088_967_971_526_4, 1.0e-15);
    assert_close_with(two_match.large_mismatch, -3.5e-2, 1.0e-16);
    assert_close_with(two_match.small_mismatch, -1.95e-2, 1.0e-16);
    assert_close_with(
        two_match.large_component[0],
        1.763_841_637_010_675_2e-1,
        1.0e-16,
    );
    assert_close_with(
        two_match.large_component[4],
        7.613_327_402_135_228_2e-1,
        1.0e-15,
    );
    assert_close_with(
        two_match.small_component[4],
        -3.253_291_814_946_617_2e-1,
        1.0e-15,
    );
    assert_close_with(
        two_match.large_coefficients[3],
        6.826_049_822_064_055_3e-1,
        1.0e-15,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_match_matches_feff_soldir_reference() -> Result<(), AtomMathError>
{
    let radial_count = 8;
    let coefficient_count = 5;
    let large_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.004 * index + 0.0005 * index * index
    });
    let small_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.003 * index + 0.0002 * index * index
    });
    let homogeneous_large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.018 * index + 0.0007 * index * index
    });
    let homogeneous_small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.012 * index + 0.0004 * index * index
    });
    let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.0008 * index + 0.00007 * index * index
    });
    let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.0006 * index + 0.00005 * index * index
    });
    let homogeneous_large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.012 * index + 0.0005 * index * index
    });
    let homogeneous_small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.009 * index + 0.0003 * index * index
    });

    let matched =
        atomic_dirac_energy_disagreement_match(AtomicDiracEnergyDisagreementMatchInput {
            large_derivative: large_derivative.view(),
            small_derivative: small_derivative.view(),
            large_derivative_coefficients: large_derivative_coefficients.view(),
            small_derivative_coefficients: small_derivative_coefficients.view(),
            homogeneous_large_component: homogeneous_large_component.view(),
            homogeneous_small_component: homogeneous_small_component.view(),
            homogeneous_large_coefficients: homogeneous_large_coefficients.view(),
            homogeneous_small_coefficients: homogeneous_small_coefficients.view(),
            matching_large_derivative: 0.037,
            matching_small_derivative: -0.011,
            homogeneous_matching_large_component: 0.087,
            homogeneous_matching_small_component: -0.047,
            coefficient_count,
            active_len: radial_count,
            matching_index_1based: 5,
        })?;

    assert_close_with(matched.determinant, -7.025e-4, 1.0e-18);
    assert_close_with(matched.prefix_scale, 1.672_597_864_768_679_6e-1, 1.0e-16);
    assert_close_with(matched.tail_scale, 1.772_241_992_882_559_2e-1, 1.0e-16);
    assert_close_with(matched.large_mismatch, -4.499_999_999_999_997_1e-3, 1.0e-18);
    assert_close_with(matched.small_mismatch, 1.000_000_000_000_000_9e-3, 1.0e-18);
    assert_close_with(
        matched.large_derivative[0],
        7.627_758_007_117_430_9e-3,
        1.0e-18,
    );
    assert_close_with(
        matched.large_derivative[4],
        5.155_160_142_348_751_1e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.small_derivative[4],
        -1.886_120_996_441_279_3e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.large_derivative_coefficients[4],
        1.787_633_451_957_292_7e-2,
        1.0e-17,
    );
    assert_close_with(
        matched.small_derivative_coefficients[4],
        -8.022_241_992_882_548_1e-3,
        1.0e-18,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_source_matches_feff_soldir_reference() -> Result<(), AtomMathError>
{
    let radial_count = 8;
    let coefficient_count = 5;
    let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
    let large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.06 * index - 0.002 * index * index
    });
    let small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.015 * index + 0.0007 * index * index
    });
    let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.12 * index - 0.004 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.08 * index + 0.003 * index * index
    });

    let source =
        atomic_dirac_energy_disagreement_source(AtomicDiracEnergyDisagreementSourceInput {
            large_component: large_component.view(),
            small_component: small_component.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            radii: radii.view(),
            speed_of_light: 137.0373,
            coefficient_count,
            active_len: 7,
        })?;

    assert_close_with(source.large_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        source.large_coefficients[1],
        8.464_848_621_506_699_7e-4,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[2],
        1.634_591_457_946_121_3e-3,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[3],
        2.364_319_787_386_353_7e-3,
        1.0e-18,
    );
    assert_close_with(
        source.large_coefficients[4],
        3.035_669_850_471_368_2e-3,
        1.0e-18,
    );
    assert_close_with(source.small_coefficients[0], 0.0, 1.0e-18);
    assert_close_with(
        source.small_coefficients[1],
        -5.618_908_136_689_792_0e-4,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[2],
        -1.079_997_927_571_544_4e-3,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[3],
        -1.554_321_341_707_695_8e-3,
        1.0e-18,
    );
    assert_close_with(
        source.small_coefficients[4],
        -1.984_861_056_077_433_2e-3,
        1.0e-18,
    );
    assert_close_with(source.large_source[0], 3.385_939_448_602_680_0e-5, 1.0e-19);
    assert_close_with(source.large_source[3], 1.689_008_004_217_633_1e-4, 1.0e-18);
    assert_close_with(source.large_source[6], 3.636_984_276_120_176_0e-4, 1.0e-18);
    assert_close_with(source.large_source[7], 0.0, 1.0e-18);
    assert_close_with(source.small_source[0], -8.348_092_088_796_263_3e-6, 1.0e-20);
    assert_close_with(source.small_source[3], -3.962_672_625_279_831_4e-5, 1.0e-19);
    assert_close_with(source.small_source[6], -7.985_552_432_350_822_2e-5, 1.0e-19);
    assert_close_with(source.small_source[7], 0.0, 1.0e-18);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_energy_disagreement_correction_matches_feff_soldir_reference()
-> Result<(), AtomMathError> {
    let radial_count = 8;
    let coefficient_count = 5;
    let radii = Array1::from_shape_fn(radial_count, |row| 0.08 * (0.11 * row as Real).exp());
    let large_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.08 * index - 0.003 * index * index
    });
    let small_component = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.018 * index + 0.0008 * index * index
    });
    let large_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        0.002 * index + 0.0003 * index * index
    });
    let small_derivative = Array1::from_shape_fn(radial_count, |row| {
        let index = (row + 1) as Real;
        -0.0014 * index + 0.0001 * index * index
    });
    let large_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.13 * index - 0.005 * index * index
    });
    let small_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.09 * index + 0.0035 * index * index
    });
    let large_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        0.0007 * index + 0.00004 * index * index
    });
    let small_derivative_coefficients = Array1::from_shape_fn(coefficient_count, |row| {
        let index = (row + 1) as Real;
        -0.0005 * index + 0.00003 * index * index
    });

    let correction = atomic_dirac_energy_disagreement_correction(
        AtomicDiracEnergyDisagreementCorrectionInput {
            radii: radii.view(),
            large_component: large_component.view(),
            small_component: small_component.view(),
            large_derivative: large_derivative.view(),
            small_derivative: small_derivative.view(),
            large_coefficients: large_coefficients.view(),
            small_coefficients: small_coefficients.view(),
            large_derivative_coefficients: large_derivative_coefficients.view(),
            small_derivative_coefficients: small_derivative_coefficients.view(),
            norm: 0.913,
            step: 0.11,
            origin_power: 1.30,
            coefficient_count,
            active_len: 7,
        },
    )?;

    assert_close_with(
        correction.overlap_integral,
        3.960_742_076_990_347_3e-4,
        1.0e-18,
    );
    assert_close_with(correction.correction, 1.098_279_038_483_979_4e2, 1.0e-12);
    assert_close_with(
        correction.normalization_mismatch,
        8.699_999_999_999_996_6e-2,
        1.0e-18,
    );
    assert_close_with(
        correction.large_component[0],
        3.296_041_788_513_152_7e-1,
        1.0e-16,
    );
    assert_close_with(
        correction.large_component[3],
        1.677_797_169_259_493_5,
        1.0e-15,
    );
    assert_close_with(
        correction.large_component[6],
        3.565_060_840_449_020_5,
        1.0e-15,
    );
    assert_close_with(correction.large_component[7], 4.48e-1, 1.0e-18);
    assert_close_with(
        correction.small_component[0],
        -1.599_762_750_029_173_1e-1,
        1.0e-16,
    );
    assert_close_with(
        correction.small_component[6],
        -6.249_567_288_571_498_1e-1,
        1.0e-16,
    );
    assert_close_with(correction.small_component[7], -9.28e-2, 1.0e-18);
    assert_close_with(
        correction.large_coefficients[4],
        1.019_225_567_317_790_8,
        1.0e-15,
    );
    assert_close_with(
        correction.small_coefficients[4],
        -5.546_988_317_346_963_6e-1,
        1.0e-16,
    );
    Ok(())
}

#[test]
fn atom_dirac_matching_point_update_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let mut large_component = Array1::<Real>::zeros(25);
    large_component[2] = 0.60;
    large_component[4] = 0.40;
    let no_update = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 13,
        matching_index_1based: 5,
        already_relocated: false,
    })?;
    assert_eq!(no_update.matching_index_1based, 5);
    assert_eq!(no_update.peak_index_1based, 3);
    assert_eq!(no_update.scan_index_1based, 5);
    assert!(!no_update.relocated);
    assert!(!no_update.needs_reintegration);

    large_component.fill(0.0);
    large_component[5] = 0.90;
    let reintegrate_even =
        atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
            large_component: large_component.view(),
            active_len: 21,
            matching_index_1based: 3,
            already_relocated: false,
        })?;
    assert_eq!(reintegrate_even.matching_index_1based, 7);
    assert_eq!(reintegrate_even.peak_index_1based, 6);
    assert_eq!(reintegrate_even.scan_index_1based, 7);
    assert!(reintegrate_even.relocated);
    assert!(reintegrate_even.needs_reintegration);

    large_component.fill(0.0);
    large_component[17] = 0.90;
    let fallback_tail = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 21,
        matching_index_1based: 5,
        already_relocated: false,
    })?;
    assert_eq!(fallback_tail.matching_index_1based, 9);
    assert_eq!(fallback_tail.peak_index_1based, 18);
    assert_eq!(fallback_tail.scan_index_1based, 9);
    assert!(fallback_tail.relocated);
    assert!(!fallback_tail.needs_reintegration);

    let already_moved = atomic_dirac_matching_point_update(AtomicDiracMatchingPointUpdateInput {
        large_component: large_component.view(),
        active_len: 21,
        matching_index_1based: 5,
        already_relocated: true,
    })?;
    assert_eq!(already_moved.matching_index_1based, 5);
    assert_eq!(already_moved.peak_index_1based, 18);
    assert_eq!(already_moved.scan_index_1based, 18);
    assert!(already_moved.relocated);
    assert!(!already_moved.needs_reintegration);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_solver_setup_matches_feff_soldir_reference() -> Result<(), AtomMathError> {
    let fixture = sample_soldir_setup_fixture();

    let clamped = atomic_dirac_solver_setup(fixture.input(-8.0, 0, -2, 2, true))?;
    assert_eq!(clamped.requested_method, 0);
    assert_eq!(clamped.method, 1);
    assert_eq!(clamped.target_nodes, 1);
    assert_close_with(clamped.energy, -5.963_839_259_330_666_4, 1.0e-14);
    assert_close_with(clamped.energy_floor, -6.626_488_065_922_962_4, 1.0e-14);
    assert_close_with(
        clamped.initial_small_coefficient,
        -1.472_928_410_311_296_5e-2,
        1.0e-16,
    );
    assert_close_with(clamped.angular_term, 7.297_283_294_402_327_9e-3, 1.0e-18);
    assert_close_with(clamped.doubled_speed_of_light, 274.0746, 1.0e-12);

    let positive_kappa = atomic_dirac_solver_setup(fixture.input(-0.2, 2, 1, 3, true))?;
    assert_eq!(positive_kappa.requested_method, 2);
    assert_eq!(positive_kappa.method, 2);
    assert_eq!(positive_kappa.target_nodes, 2);
    assert_close_with(positive_kappa.energy, -0.2, 1.0e-18);
    assert_close_with(
        positive_kappa.energy_floor,
        -6.626_488_065_922_962_4,
        1.0e-14,
    );
    assert_close_with(
        positive_kappa.initial_small_coefficient,
        3.160_423_066_381_816_8e1,
        1.0e-13,
    );
    assert_close_with(
        positive_kappa.angular_term,
        7.297_283_294_402_327_9e-3,
        1.0e-18,
    );

    let no_adjust = atomic_dirac_solver_setup(fixture.input(-0.1, -1, -1, 1, false))?;
    assert_eq!(no_adjust.requested_method, -1);
    assert_eq!(no_adjust.method, 1);
    assert_eq!(no_adjust.target_nodes, 1);
    assert_close_with(no_adjust.energy, -0.1, 1.0e-18);
    assert_close_with(no_adjust.energy_floor, -5.619_077_423_139_916_0e1, 1.0e-13);
    assert_close_with(no_adjust.initial_small_coefficient, -6.0e-3, 1.0e-18);
    assert_close_with(no_adjust.angular_term, 0.0, 1.0e-18);
    Ok(())
}

#[test]
fn atom_dirac_bound_orbital_composes_soldir_driver() -> Result<(), AtomMathError> {
    let fixture = sample_intdir_fixture();
    let solution = atomic_dirac_bound_orbital(AtomicDiracBoundOrbitalInput {
        large_source: fixture.large_source.view(),
        small_source: fixture.small_source.view(),
        large_source_coefficients: fixture.large_coefficients.view(),
        small_source_coefficients: fixture.small_coefficients.view(),
        radii: fixture.radii.view(),
        potential: fixture.potential.view(),
        potential_coefficients: fixture.potential_coefficients.view(),
        energy: -0.08,
        origin_power: 0.999,
        initial_large_coefficient: 0.85,
        initial_small_coefficient: -0.004,
        asymptotic_large_component: 0.02,
        principal_quantum_number: 1,
        kappa: -1,
        speed_of_light: 137.0373,
        step: 0.05,
        primary_matching_precision: 1.0e-7,
        secondary_matching_precision: 1.0e-6,
        coefficient_count: 6,
        active_len: 151,
        initial_max_index_1based: 151,
        max_attempt_count: 50,
        method: 1,
    })?;

    assert_eq!(solution.method, 1);
    assert!(solution.attempts_exhausted);
    assert_eq!(solution.active_len, 105);
    assert_eq!(solution.matching_index_1based, 69);
    assert_eq!(solution.node_count, 1);
    assert_eq!(solution.search_attempt_count, 0);
    assert_eq!(solution.match_attempt_count, 51);
    assert_close_with(solution.energy, -2.463_836_906_999_279_8e1, 1.0e-10);
    assert_close_with(solution.norm, 3.655_262_276_748_030_6e-3, 1.0e-15);
    assert_close_with(
        solution.large_component[0],
        3.322_202_624_147_516_0e-1,
        1.0e-15,
    );
    assert_close_with(
        solution.large_component[50],
        1.474_836_017_186_380_6,
        1.0e-14,
    );
    assert_close_with(
        solution.large_component[104],
        -6.452_040_959_794_344_0e-6,
        1.0e-18,
    );
    assert_close_with(
        solution.small_component[0],
        -9.902_083_655_412_203_0e-3,
        1.0e-17,
    );
    assert_close_with(
        solution.small_component[104],
        1.653_072_303_996_776_8e-7,
        1.0e-19,
    );
    assert_close_with(
        solution.large_coefficients[0],
        1.405_916_906_625_304_1e1,
        1.0e-12,
    );
    assert_close_with(
        solution.small_coefficients[0],
        -4.105_802_483_655_679_5e-1,
        1.0e-14,
    );
    assert_close_with(solution.large_component[solution.active_len], 0.0, 1.0e-18);
    assert_close_with(solution.small_component[solution.active_len], 0.0, 1.0e-18);
    Ok(())
}

#[test]
fn atom_dirac_bound_orbital_composes_method2_soldir_driver() -> Result<(), AtomMathError> {
    let fixture = sample_intdir_fixture();
    let solution = atomic_dirac_bound_orbital(AtomicDiracBoundOrbitalInput {
        large_source: fixture.large_source.view(),
        small_source: fixture.small_source.view(),
        large_source_coefficients: fixture.large_coefficients.view(),
        small_source_coefficients: fixture.small_coefficients.view(),
        radii: fixture.radii.view(),
        potential: fixture.potential.view(),
        potential_coefficients: fixture.potential_coefficients.view(),
        energy: -0.08,
        origin_power: 0.999,
        initial_large_coefficient: 0.85,
        initial_small_coefficient: -0.004,
        asymptotic_large_component: 0.02,
        principal_quantum_number: 2,
        kappa: -1,
        speed_of_light: 137.0373,
        step: 0.05,
        primary_matching_precision: 1.0e-7,
        secondary_matching_precision: 1.0e-6,
        coefficient_count: 6,
        active_len: 151,
        initial_max_index_1based: 151,
        max_attempt_count: 50,
        method: 2,
    })?;

    assert_eq!(solution.method, 2);
    assert!(!solution.attempts_exhausted);
    assert_eq!(solution.active_len, 107);
    assert_eq!(solution.matching_index_1based, 81);
    assert_eq!(solution.node_count, 2);
    assert_eq!(solution.search_attempt_count, 0);
    assert_eq!(solution.match_attempt_count, 41);
    // Iterative matching takes a slightly different final floating-point step
    // across platform libm implementations. These tolerances cover the
    // macOS/Linux spread while remaining at or below the solver's 1e-7
    // primary matching precision.
    assert_close_with(solution.energy, -1.852_679_703_022_842_3e1, 1.0e-8);
    assert_close_with(solution.norm, 1.000_000_000_011_237_7, 1.0e-10);
    assert_close_with(
        solution.large_component[0],
        2.013_842_378_173_847_0e-1,
        1.0e-9,
    );
    assert_close_with(
        solution.large_component[70],
        -1.000_684_726_233_834_5,
        1.0e-8,
    );
    assert_close_with(
        solution.large_component[106],
        -7.173_794_714_792_370_0e-7,
        1.0e-14,
    );
    assert_close_with(
        solution.small_component[0],
        -6.105_178_569_893_397_0e-3,
        1.0e-10,
    );
    assert_close_with(
        solution.small_component[106],
        1.593_684_930_837_893_0e-8,
        1.0e-15,
    );
    assert_close_with(
        solution.large_coefficients[0],
        8.540_080_903_008_144,
        1.0e-7,
    );
    assert_close_with(
        solution.small_coefficients[0],
        -2.494_022_599_554_403_8e-1,
        1.0e-9,
    );
    assert_close_with(solution.large_component[solution.active_len], 0.0, 1.0e-18);
    assert_close_with(solution.small_component[solution.active_len], 0.0, 1.0e-18);
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_dirac_integration_matches_feff_intdir_reference() -> Result<(), AtomMathError> {
    let fixture = sample_intdir_fixture();

    let searched = atomic_dirac_integration(fixture.input(
        AtomicDiracIntegrationMode::SearchMatchingPoint,
        0,
        0,
    ))?;
    assert_eq!(searched.matching_index_1based, 127);
    assert_eq!(searched.max_index_1based, 151);
    assert_some_close(
        searched.matching_large_component,
        7.844_180_279_031_651_7e-1,
        1.0e-12,
    );
    assert_some_close(
        searched.matching_small_component,
        6.433_852_518_326_962_0e-4,
        1.0e-15,
    );
    assert_close_with(
        searched.large_component[126],
        3.946_584_591_497_206_1e2,
        1.0e-9,
    );
    assert_close_with(
        searched.small_component[126],
        -5.380_100_169_329_787_9e-1,
        1.0e-12,
    );
    assert_close_with(
        searched.large_coefficients[1],
        -1.096_438_489_149_803_4,
        1.0e-12,
    );
    assert_close_with(
        searched.small_coefficients[1],
        2.146_028_457_009_671_9e-2,
        1.0e-14,
    );
    assert_close_with(
        searched.large_component[150],
        7.844_180_279_031_651_3e-8,
        1.0e-20,
    );
    assert_close_with(
        searched.small_component[150],
        -1.144_825_333_416_651_0e-10,
        1.0e-22,
    );

    let fixed = atomic_dirac_integration(fixture.input(
        AtomicDiracIntegrationMode::FixedMatchingPoint,
        65,
        139,
    ))?;
    assert_eq!(fixed.matching_index_1based, 65);
    assert_eq!(fixed.max_index_1based, 139);
    assert_some_close(
        fixed.matching_large_component,
        -4.787_017_896_869_409_0e-2,
        1.0e-13,
    );
    assert_some_close(
        fixed.matching_small_component,
        2.893_471_976_931_037_7e-3,
        1.0e-15,
    );
    assert_close_with(fixed.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
    assert_close_with(
        fixed.small_component[64],
        1.444_514_204_264_709_7e-2,
        1.0e-15,
    );
    assert_close_with(
        fixed.large_coefficients[1],
        -1.096_438_489_149_803_4,
        1.0e-12,
    );
    assert_close_with(
        fixed.small_coefficients[1],
        2.146_028_457_009_671_9e-2,
        1.0e-14,
    );
    assert_close_with(fixed.large_component[138], 2.0e-2, 1.0e-20);
    assert_close_with(
        fixed.small_component[138],
        -2.918_916_426_428_632_8e-5,
        1.0e-22,
    );

    let inward =
        atomic_dirac_integration(fixture.input(AtomicDiracIntegrationMode::InwardOnly, 65, 139))?;
    assert_eq!(inward.matching_large_component, None);
    assert_eq!(inward.matching_small_component, None);
    assert_eq!(inward.matching_index_1based, 65);
    assert_eq!(inward.max_index_1based, 139);
    assert_close_with(inward.large_component[64], 2.250_038_459_307_619_5, 1.0e-13);
    assert_close_with(
        inward.small_component[64],
        1.444_514_204_264_709_7e-2,
        1.0e-15,
    );
    assert_close_with(inward.large_coefficients[1], 4.0e-4, 1.0e-18);
    assert_close_with(inward.small_coefficients[1], -3.0e-4, 1.0e-18);
    assert_close_with(inward.large_component[138], 2.0e-2, 1.0e-20);
    assert_close_with(
        inward.small_component[138],
        -2.918_916_426_428_632_8e-5,
        1.0e-22,
    );
    Ok(())
}
