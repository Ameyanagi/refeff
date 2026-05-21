#![allow(clippy::excessive_precision)]

use super::*;

#[allow(clippy::excessive_precision)]
#[test]
fn atom_radial_integral_matches_feff_fdrirk_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkrdf_fixture();
    let kappas = [-1, 1, -2];

    let overlap = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 1,
            first_right: 2,
            second_left: 1,
            second_right: 3,
            rank: 2,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close_with(overlap.value, 3.844_030_024_958_072_30e-9, 1.0e-20);
    let overlap_factor = overlap
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    assert_close_with(
        overlap_factor.values[0],
        1.109_878_400_538_443_00e-5,
        1.0e-17,
    );
    assert_close_with(
        overlap_factor.values[3],
        1.171_927_755_618_356_82e-5,
        1.0e-17,
    );
    assert_close_with(
        overlap_factor.coefficients[0],
        -2.561_250_012_646_588_91,
        1.0e-12,
    );
    assert_close_with(
        overlap_factor.coefficients[3],
        -8.575_701_162_755_210_16e-2,
        1.0e-16,
    );

    let large_small = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 2,
            first_right: 3,
            second_left: 1,
            second_right: 2,
            rank: 1,
        },
        &kappas,
        true,
        None,
    ))?;
    assert_close_with(large_small.value, 2.056_815_682_976_472_25e-10, 1.0e-21);
    let large_small_factor = large_small
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    assert_close_with(
        large_small_factor.coefficients[0],
        -2.237_401_842_533_894_71e-2,
        1.0e-14,
    );
    assert_close_with(
        large_small_factor.coefficients[3],
        9.462_409_003_166_756_97e-4,
        1.0e-17,
    );

    let first = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 2,
            first_right: 1,
            second_left: 2,
            second_right: 1,
            rank: 1,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close_with(first.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
    let previous = first
        .first_factor
        .as_ref()
        .ok_or(AtomMathError::MissingRadialFirstFactor)?;
    let sentinel = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 0,
            first_right: 0,
            second_left: 1,
            second_right: 2,
            rank: 1,
        },
        &kappas,
        false,
        Some(previous.as_view()),
    ))?;
    assert_close_with(sentinel.value, -3.712_970_151_907_870_88e-9, 1.0e-20);
    assert!(sentinel.first_factor.is_none());

    let no_second = atomic_radial_integral(fixture.fdrirk_input(
        AtomicRadialIntegralRequest {
            first_left: 1,
            first_right: 2,
            second_left: 0,
            second_right: 0,
            rank: 2,
        },
        &kappas,
        false,
        None,
    ))?;
    assert_close(no_second.value, 0.0);
    assert!(no_second.first_factor.is_some());
    Ok(())
}

#[test]
fn atom_form_factor_matches_feff_fpf0_reference() -> Result<(), AtomMathError> {
    let radial_count = 251;
    let orbital_count = 5;
    let radial_step = 0.05;
    let radii = Array1::from_shape_fn(radial_count, |index| {
        (-8.8 + radial_step * index as Real).exp()
    });
    let density_4pi = Array1::from_shape_fn(radial_count, |index| {
        0.3 * (-0.7 * radii[index]).exp() + 0.01 * (index + 1).rem_euclid(7) as Real
    });
    let initial_large_component = Array1::from_shape_fn(radial_count, |index| {
        0.2 * (-0.4 * radii[index]).exp() + 0.001 * (index + 1) as Real
    });
    let initial_small_component = Array1::from_shape_fn(radial_count, |index| {
        -0.05 * (-0.3 * radii[index]).exp() + 0.0002 * (index + 1) as Real
    });
    let large_components = Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
        let orbital = (col + 1) as Real;
        (0.03 * orbital + 0.0007 * (row + 1) as Real) * (-0.05 * orbital * radii[row]).exp()
    });
    let small_components = Array2::from_shape_fn((radial_count, orbital_count), |(row, col)| {
        let orbital = (col + 1) as Real;
        (-0.01 * orbital + 0.0003 * (row + 1) as Real) * (-0.03 * orbital * radii[row]).exp()
    });
    let occupations = [2.0, 2.0, 1.5, 0.5, 0.0];
    let orbital_energies = [-0.85, -0.55, -0.21, -0.08, 0.04];
    let kappas = [-1, 1, -2, 2, -1];

    let form_factor = atomic_form_factor(AtomicFormFactorInput {
        atomic_number: 26,
        hole_orbital_1based: 2,
        radial_step,
        total_energy: -2.345,
        radii: radii.view(),
        density_4pi: density_4pi.view(),
        initial_large_component: initial_large_component.view(),
        initial_small_component: initial_small_component.view(),
        large_components: large_components.view(),
        small_components: small_components.view(),
        occupations: &occupations,
        orbital_energies: &orbital_energies,
        kappas: &kappas,
    })?;

    assert_eq!(form_factor.atomic_number, 26);
    assert_close_with(form_factor.total_energy_fprime, -2.081_24e-4, 5.0e-10);
    assert_close_with(form_factor.relativistic_correction, -6.478_75e-2, 5.0e-8);
    assert_eq!(form_factor.oscillators.len(), 3);
    let expected_oscillators = [(2.0, -0.55, 2), (0.104_07, -0.85, 1), (0.003_60, -0.08, 4)];
    for (actual, (strength, energy, index)) in
        form_factor.oscillators.iter().zip(expected_oscillators)
    {
        assert_close_with(actual.oscillator_strength, strength, 5.0e-6);
        assert_close_with(actual.excitation_energy, energy, 5.0e-13);
        assert_eq!(actual.orbital_index_1based, index);
    }
    assert_eq!(form_factor.form_factor.len(), 81);
    let expected_rows = [
        (0, 0.0, 760.5215),
        (1, 0.5, -4.0195),
        (2, 1.0, 16.7054),
        (3, 1.5, -1.1065),
        (4, 2.0, -0.5452),
        (10, 5.0, 1.4707),
        (20, 10.0, -0.1129),
        (40, 20.0, -0.6736),
        (80, 40.0, 0.1214),
    ];
    for (index, momentum, value) in expected_rows {
        assert_close_with(form_factor.form_factor_momentum[index], momentum, 1.0e-13);
        assert_close_with(form_factor.form_factor[index], value, 5.5e-5);
    }
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_schmidt_orthogonalization_matches_feff_ortdat_reference() -> Result<(), AtomMathError> {
    let fixture = sample_schmidt_fixture();
    let all_orbitals =
        atomic_schmidt_orthogonalization(fixture.as_input(None), sample_schmidt_integral)?;
    assert_eq!(all_orbitals.active_lengths, vec![3, 4, 3, 5]);
    assert_columns_close(
        &all_orbitals.large_components,
        &[
            [0.18, 0.25, 0.32, 0.39, 0.46],
            [
                0.333_475_933_348_347_96,
                0.403_443_338_654_020_99,
                0.473_410_743_959_694_18,
                0.697_998_855_802_804_52,
                0.57,
            ],
            [
                0.487_117_140_335_587_17,
                0.572_362_639_894_314_91,
                0.657_608_139_453_042_64,
                0.61,
                0.68,
            ],
            [
                0.086_758_208_000_696_446,
                0.041_346_281_239_887_581,
                -0.004_065_645_520_921_706_5,
                -0.041_673_823_238_614_134,
                0.979_213_171_940_273_24,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.small_components,
        &[
            [0.01, 0.04, 0.07, 0.1, 0.13],
            [
                -0.017_924_610_617_016_022,
                0.012_061_420_228_272_458,
                0.042_047_451_073_560_942,
                0.111_679_816_928_448_71,
                0.11,
            ],
            [
                -0.036_533_785_525_169_032,
                0.0,
                0.036_533_785_525_169_032,
                0.06,
                0.09,
            ],
            [
                -0.043_493_187_919_062_107,
                -0.062_955_442_245_123_172,
                -0.082_417_696_571_184_237,
                -0.099_878_989_604_138_421,
                0.086_765_724_095_973_565,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.large_coefficients,
        &[
            [0.25, 0.45, 0.65, 0.85],
            [
                0.319_683_475_957_684_54,
                0.519_590_348_259_607_70,
                0.719_497_220_561_530_87,
                0.919_404_092_863_454_04,
            ],
            [
                0.426_227_497_793_638_78,
                0.669_786_067_961_432_36,
                0.913_344_638_129_225_95,
                1.156_903_208_297_019_4,
            ],
            [
                -0.069_671_028_191_237_896,
                -0.199_419_390_364_978_30,
                -0.329_167_752_538_718_77,
                -0.458_916_114_712_459_13,
            ],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &all_orbitals.small_coefficients,
        &[
            [0.01, -0.02, -0.05, -0.08],
            [
                0.065_835_252_079_320_519,
                0.035_849_221_234_032_044,
                0.005_863_190_388_743_565_9,
                -0.024_122_840_456_544_916,
            ],
            [
                0.109_601_356_575_507_10,
                0.073_067_571_050_338_065,
                0.036_533_785_525_169_032,
                0.0,
            ],
            [
                0.067_524_121_512_063_162,
                0.086_986_375_838_124_214,
                0.106_448_630_164_185_28,
                0.125_910_884_490_246_34,
            ],
        ],
        1.0e-12,
    );

    let active_two =
        atomic_schmidt_orthogonalization(fixture.as_input(Some(2)), sample_schmidt_integral)?;
    assert_eq!(active_two.active_lengths, vec![3, 5, 3, 5]);
    assert_columns_close(
        &active_two.large_components,
        &[
            [0.18, 0.25, 0.32, 0.39, 0.46],
            [
                -0.257_731_473_167_008_73,
                -0.271_503_234_760_490_32,
                -0.285_274_996_353_971_69,
                -0.160_996_405_265_147_69,
                -0.860_433_208_548_678_89,
            ],
            [0.4, 0.47, 0.54, 0.61, 0.68],
            [0.51, 0.58, 0.65, 0.72, 0.79],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.small_components,
        &[
            [0.01, 0.04, 0.07, 0.1, 0.13],
            [
                0.038_454_127_655_123_280,
                0.032_551_944_115_059_794,
                0.026_649_760_574_996_302,
                0.056_145_103_363_729_076,
                -0.076_240_917_213_174_053,
            ],
            [-0.03, 0.0, 0.03, 0.06, 0.09],
            [-0.05, -0.02, 0.01, 0.04, 0.07],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.large_coefficients,
        &[
            [0.25, 0.45, 0.65, 0.85],
            [
                -0.150_238_668_255_056_88,
                -0.189_586_558_522_146_90,
                -0.228_934_448_789_236_94,
                -0.268_282_339_056_326_81,
            ],
            [0.35, 0.55, 0.75, 0.95],
            [0.4, 0.6, 0.8, 1.0],
        ],
        1.0e-12,
    );
    assert_columns_close(
        &active_two.small_coefficients,
        &[
            [0.01, -0.02, -0.05, -0.08],
            [
                -0.082_810_438_850_310_059,
                -0.076_908_255_310_246_559,
                -0.071_006_071_770_183_060,
                -0.065_103_888_230_119_589,
            ],
            [0.09, 0.06, 0.03, 0.0],
            [0.13, 0.10, 0.07, 0.04],
        ],
        1.0e-12,
    );

    Ok(())
}

#[test]
fn atom_overlap_amplitude_reduction_matches_feff_s02at_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, -1, 1, 1, -2, -3];
    let occupations = [2.0, 1.0, 1.5, 0.5, 3.0, 2.5];
    let overlaps = sample_s02at_overlaps();
    let cases = [
        (None, 9.680_452_235_999_996e-3),
        (Some(1), 9.680_452_235_999_996e-3),
        (Some(2), 0.327_600_000_000_000_1),
        (Some(3), 9.680_452_235_999_996e-3),
        (Some(4), 9.020_027_472_527_463e-2),
        (Some(5), 9.680_452_235_999_996e-3),
        (Some(6), 9.680_452_235_999_996e-3),
    ];

    for (hole_orbital_1based, expected) in cases {
        let actual = atomic_overlap_amplitude_reduction(AtomicOverlapAmplitudeReductionInput {
            hole_orbital_1based,
            kappas: &kappas,
            occupations: &occupations,
            overlap_integrals: overlaps.view(),
        })?;
        assert_close(actual, expected);
    }
    Ok(())
}
