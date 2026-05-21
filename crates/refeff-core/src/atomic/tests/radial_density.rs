#![allow(clippy::excessive_precision)]

use super::*;

#[test]
fn atom_total_energy_matches_feff_etotal_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, 1, -2, 2];
    let occupations = [2.0, 1.5, 3.0, 0.5];
    let valence_occupations = [0.0, 0.0, 1.0, 0.0];
    let orbital_energies = [-0.7, -0.3, -0.12, -0.05];
    let coefficients = Array3::from_shape_fn((4, 4, 6), |(row, column, channel)| {
        0.01 * (100 * (row + 1) + 10 * (column + 1) + channel + 1) as Real
    });

    let energy = atomic_total_energy(
        AtomicTotalEnergyInput {
            kappas: &kappas,
            occupations: &occupations,
            valence_occupations: &valence_occupations,
            orbital_energies: &orbital_energies,
            coulomb_coefficients: coefficients.view(),
        },
        |request| {
            Ok(0.0001 * (request.rank + 1) as Real
                + 0.001 * request.first_left as Real
                + 0.0002 * request.first_right as Real
                + 0.00003 * request.second_left as Real
                + 0.000004 * request.second_right as Real)
        },
    )?;

    assert_close(energy.total, -2.230_065_144_829_932);
    assert_close_with(energy.direct_coulomb, 0.109_629, 1.0e-6);
    assert_close_with(energy.exchange_coulomb, -0.055_702_8, 1.0e-6);
    assert_close_with(energy.magnetic_breit, 0.075_902_3, 1.0e-6);
    assert_close_with(energy.retarded_breit, -0.017_041_4, 1.0e-6);
    Ok(())
}

#[test]
fn atom_lagrange_parameters_match_feff_lagdat_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, -1, 1, 1, -2];
    let occupations = [2.0, 1.0, 1.5, 0.5, 3.0];
    let valence_occupations = [0.0, 0.0, 0.25, 0.0, 0.0];
    let shell_markers = [-1, 1, 1, 1, -1];
    let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let all_parameters = atomic_lagrange_parameters(
        AtomicLagrangeParametersInput {
            active_orbital_1based: None,
            include_exchange: true,
            kappas: &kappas,
            occupations: &occupations,
            shell_markers: &shell_markers,
            coulomb_coefficients: coefficients.view(),
        },
        sample_atomic_radial_integral,
    )?;
    let expected_all = [
        -1.780_000_000_000_000_1e-3,
        0.0,
        0.0,
        0.0,
        0.0,
        -6.871_000_000_000_001e-3,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (&actual, expected) in all_parameters.iter().zip(expected_all) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    let active_parameters = atomic_lagrange_parameters(
        AtomicLagrangeParametersInput {
            active_orbital_1based: Some(2),
            include_exchange: false,
            kappas: &kappas,
            occupations: &occupations,
            shell_markers: &shell_markers,
            coulomb_coefficients: coefficients.view(),
        },
        sample_atomic_radial_integral,
    )?;
    let expected_active = [
        -1.200_000_000_000_000_1e-3,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (&actual, expected) in active_parameters.iter().zip(expected_active) {
        assert_close_with(actual, expected, 1.0e-12);
    }

    Ok(())
}

#[test]
fn atom_tabulation_matches_feff_tabrat_reference() -> Result<(), AtomMathError> {
    let principal_quantum_numbers = [1, 2, 2, 3, 3];
    let kappas = [-1, -1, 1, -2, 1];
    let occupations = [2.0, 1.5, 0.5, 3.0, 0.25];
    let orbital_energies = [-0.70, -0.25, -0.18, -0.09, -0.04];
    let tabulation = atomic_tabulation(
        AtomicTabulationInput {
            principal_quantum_numbers: &principal_quantum_numbers,
            kappas: &kappas,
            occupations: &occupations,
            orbital_energies: &orbital_energies,
        },
        sample_atomic_tabrat_integral,
    )?;

    let expected = [
        (
            1,
            "s",
            2.0,
            19.047_977_2,
            [0.136, 0.134, 0.132, 0.131, 0.129, 0.128, 0.0],
            6,
        ),
        (
            2,
            "s",
            1.5,
            6.802_849,
            [0.166, 0.164, 0.162, 0.161, 0.159, 0.158, 0.0],
            6,
        ),
        (
            2,
            "p*",
            0.5,
            4.898_051_28,
            [0.196, 0.194, 0.192, 0.191, 0.189, 0.188, 0.0],
            6,
        ),
        (
            3,
            "p",
            3.0,
            2.449_025_64,
            [0.226, 0.224, 0.222, 0.221, 0.219, 0.218, 0.217],
            7,
        ),
        (
            3,
            "p*",
            0.25,
            1.088_455_84,
            [0.256, 0.254, 0.252, 0.251, 0.249, 0.248, 0.0],
            6,
        ),
    ];
    for (orbital, (nq, label, occupation, binding_energy_ev, moments, moment_count)) in
        tabulation.orbitals.iter().zip(expected)
    {
        assert_eq!(orbital.principal_quantum_number, nq);
        assert_eq!(orbital.orbital_label, label);
        assert_close(orbital.occupation, occupation);
        assert_close_with(orbital.binding_energy_ev, binding_energy_ev, 1.0e-10);
        assert_eq!(orbital.moments.len(), moment_count);
        for ((moment, &expected_value), &expected_power) in orbital
            .moments
            .iter()
            .zip(moments.iter())
            .zip(ATOM_TABRAT_MOMENT_POWERS.iter())
        {
            assert_eq!(moment.power, expected_power);
            assert_close(moment.value, expected_value);
        }
    }
    assert_eq!(tabulation.overlaps.len(), 2);
    assert_eq!(tabulation.overlaps[0].left, 0);
    assert_eq!(tabulation.overlaps[0].right, 1);
    assert_eq!(tabulation.overlaps[0].left_orbital_label, "s");
    assert_eq!(tabulation.overlaps[0].right_orbital_label, "s");
    assert_close(tabulation.overlaps[0].value, 0.15);
    assert_eq!(tabulation.overlaps[1].left, 2);
    assert_eq!(tabulation.overlaps[1].right, 4);
    assert_eq!(tabulation.overlaps[1].left_orbital_label, "p*");
    assert_eq!(tabulation.overlaps[1].right_orbital_label, "p*");
    assert_close(tabulation.overlaps[1].value, 0.23);
    Ok(())
}
#[test]
fn atom_differential_integral_matches_feff_dsordf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_dsordf_fixture();
    let cases = [
        (
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 1,
                right_orbital_1based: 2,
                multiply_by_derivative: false,
            },
            2,
            0.0,
            4.983_995_991_889_760_16e-9,
        ),
        (
            AtomicDifferentialIntegralKind::ComponentOverlap {
                left_orbital_1based: 1,
                right_orbital_1based: 3,
                multiply_by_derivative: true,
            },
            -1,
            0.4,
            4.174_834_158_519_188_87e-5,
        ),
        (
            AtomicDifferentialIntegralKind::LargeSmallOverlap {
                left_orbital_1based: 2,
                right_orbital_1based: 3,
                multiply_by_derivative: false,
            },
            1,
            0.0,
            -5.798_475_020_316_198_31e-8,
        ),
        (
            AtomicDifferentialIntegralKind::LargeSmallOverlap {
                left_orbital_1based: 2,
                right_orbital_1based: 1,
                multiply_by_derivative: true,
            },
            0,
            0.3,
            -4.232_100_062_570_746_56e-8,
        ),
        (
            AtomicDifferentialIntegralKind::DerivativeProjection {
                large_orbital_1based: 2,
                small_orbital_1based: 3,
            },
            0,
            0.45,
            1.816_237_327_192_537_93e-5,
        ),
        (
            AtomicDifferentialIntegralKind::DerivativeNorm { active_len: 9 },
            0,
            0.45,
            5.411_954_636_180_096_36e-5,
        ),
    ];

    for (kind, power, origin_power, expected) in cases {
        let actual = atomic_differential_integral(fixture.input(kind, power, origin_power))?;
        assert_close_with(actual, expected, 1.0e-17);
    }
    Ok(())
}
#[test]
fn atom_local_density_potential_matches_feff_vlda_reference() -> Result<(), AtomMathError> {
    let fixture = sample_vlda_fixture();

    let valence = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::ValenceDensity, true),
    )?;
    assert_close_with(
        valence.total_density[0],
        6.809_505_899_999_999_42e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.total_density[4],
        8.670_367_500_000_001_48e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.total_density[9],
        4.974_400_000_000_001_22e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[0],
        2.049_973_999_999_999_98e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[4],
        2.672_390_000_000_000_62e-3,
        1.0e-17,
    );
    assert_close_with(
        valence.valence_density[9],
        1.243_600_000_000_000_30e-3,
        1.0e-17,
    );
    assert_close_with(valence.potential[0], -7.054_707_605_385_910_14e-3, 2.0e-10);
    assert_close_with(valence.potential[4], -6.362_810_615_972_094_51e-3, 2.0e-10);
    assert_close_with(valence.potential[9], -3.663_730_681_720_983_07e-3, 2.0e-10);
    assert_close_with(valence.potential[12], 1.300_000_000_000_000_16e-3, 1.0e-18);
    assert_close_with(
        valence.development_coefficients[1],
        1.284_529_239_461_408_91e-2,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[0],
        -4.676_397_112_407_516_99e-3,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[4],
        1.845_934_601_355_665_38e-3,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[9],
        1.682_086_596_903_880_49e-2,
        2.0e-10,
    );
    assert_close_with(
        valence.energy_density[12],
        2.600_000_000_000_000_23e-2,
        1.0e-18,
    );

    let core = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::CoreDensitySeparated, true),
    )?;
    assert_close_with(core.potential[0], -4.639_483_986_312_321_55e-3, 2.0e-10);
    assert_close_with(core.potential[4], -4.094_974_008_849_363_44e-3, 2.0e-10);
    assert_close_with(core.potential[9], -1.989_064_683_335_639_83e-3, 2.0e-10);
    assert_close_with(
        core.development_coefficients[1],
        1.526_051_601_368_767_77e-2,
        2.0e-10,
    );
    assert_close_with(
        core.energy_density[0],
        -2.422_637_366_298_145_52e-3,
        2.0e-10,
    );
    assert_close_with(core.energy_density[4], 4.540_470_272_335_868_71e-3, 2.0e-10);
    assert_close_with(core.energy_density[9], 1.796_243_867_752_029_75e-2, 2.0e-10);

    let total = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::TotalDensity, false),
    )?;
    assert_close_with(total.potential[0], -1.030_418_779_316_292_88e-2, 2.0e-10);
    assert_close_with(total.potential[4], -9.399_113_926_789_406_24e-3, 2.0e-10);
    assert_close_with(total.potential[9], -6.124_858_580_930_082_20e-3, 2.0e-10);
    assert_close_with(
        total.energy_density[0],
        2.000_000_000_000_000_04e-3,
        1.0e-18,
    );
    assert_close_with(
        total.energy_density[4],
        1.000_000_000_000_000_02e-2,
        1.0e-18,
    );
    assert_close_with(
        total.energy_density[9],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );

    let dirac = atomic_local_density_potential(
        fixture.input(AtomicLocalDensityExchangeMode::DiracFockOnly, true),
    )?;
    assert_close_with(dirac.potential[0], 1.000_000_000_000_000_05e-4, 1.0e-19);
    assert_close_with(dirac.potential[4], 5.000_000_000_000_000_10e-4, 1.0e-19);
    assert_close_with(dirac.potential[9], 1.000_000_000_000_000_02e-3, 1.0e-18);
    assert_close_with(
        dirac.development_coefficients[1],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[0],
        2.000_000_000_000_000_04e-3,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[4],
        1.000_000_000_000_000_02e-2,
        1.0e-18,
    );
    assert_close_with(
        dirac.energy_density[9],
        2.000_000_000_000_000_04e-2,
        1.0e-18,
    );
    Ok(())
}
#[test]
fn atom_orbital_potential_matches_feff_potrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_potrdf_fixture();

    let full = atomic_orbital_potential(fixture.input(true, true))?;
    for (index, expected) in [
        (0, -1.451_464_734_879_546_50e-3),
        (4, -1.422_294_851_220_632_99e-3),
        (9, -1.385_920_785_309_911_19e-3),
        (12, -1.364_108_165_051_381_58e-3),
    ] {
        assert_close_with(full.central_potential[index], expected, 1.0e-15);
    }
    for (index, expected) in [
        (0, -2.189_205_772_127_074_25e-4),
        (1, -4.371_323_520_763_144_61e-4),
        (3, -8.773_080_991_906_762_18e-4),
        (5, -1.317_263_492_825_318_69e-3),
    ] {
        assert_close_with(
            full.central_development_coefficients[index],
            expected,
            1.0e-15,
        );
    }
    for (index, expected) in [
        (0, 1.702_743_222_291_228_72e-7),
        (4, 2.294_031_531_020_954_80e-7),
        (9, 0.0),
    ] {
        assert_close_with(full.exchange_large[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, -4.763_258_868_894_551_20e-8),
        (4, -4.776_069_610_481_555_18e-8),
        (9, 0.0),
    ] {
        assert_close_with(full.exchange_small[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, 2.307_477_389_651_008_40e-5),
        (2, 4.794_137_619_410_912_88e-5),
        (5, 7.832_202_463_049_932_73e-5),
    ] {
        assert_close_with(full.exchange_large_coefficients[index], expected, 1.0e-16);
    }
    for (index, expected) in [
        (0, 1.845_981_911_720_806_31e-6),
        (2, -4.841_519_661_027_267_90e-6),
        (5, -1.331_336_809_940_218_72e-5),
    ] {
        assert_close_with(full.exchange_small_coefficients[index], expected, 1.0e-16);
    }

    let direct = atomic_orbital_potential(fixture.input(false, false))?;
    for (actual, expected) in direct
        .central_potential
        .iter()
        .zip(full.central_potential.iter())
    {
        assert_close_with(*actual, *expected, 1.0e-16);
    }
    for (actual, expected) in direct
        .central_development_coefficients
        .iter()
        .zip(full.central_development_coefficients.iter())
    {
        assert_close_with(*actual, *expected, 1.0e-16);
    }
    for value in direct
        .exchange_large
        .iter()
        .chain(direct.exchange_small.iter())
        .chain(direct.exchange_large_coefficients.iter())
        .chain(direct.exchange_small_coefficients.iter())
    {
        assert_close_with(*value, 0.0, 1.0e-20);
    }
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_transform_matches_feff_yzkteg_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkteg_fixture();
    let transform = atomic_yk_zk_transform(fixture.input())?;

    assert_eq!(transform.computed_source_len, 9);
    assert_close_with(
        transform.origin_constant,
        1.024_939_588_738_283_48e2,
        1.0e-11,
    );
    assert_close_with(transform.yk[0], 3.871_202_667_947_041_34e-4, 1.0e-16);
    assert_close_with(transform.yk[1], 4.476_978_947_879_065_22e-4, 1.0e-16);
    assert_close_with(transform.yk[4], 6.350_731_526_853_801_77e-4, 1.0e-16);
    assert_close_with(transform.yk[8], 6.665_230_606_586_294_07e-4, 1.0e-16);
    assert_close_with(transform.yk[12], 4.467_837_687_045_075_67e-4, 1.0e-16);
    assert_close_with(transform.zk[0], 1.055_350_291_449_006_03e-5, 1.0e-17);
    assert_close_with(transform.zk[1], 1.147_457_094_885_342_41e-5, 1.0e-17);
    assert_close_with(transform.zk[4], 1.675_242_796_907_188_86e-4, 1.0e-16);
    assert_close_with(transform.zk[9], 7.118_915_805_710_559_43e-4, 1.0e-16);
    assert_close_with(
        transform.yk_coefficients[0],
        -3.906_646_372_399_797_53e-2,
        1.0e-16,
    );
    assert_close_with(
        transform.yk_coefficients[3],
        6.197_311_460_469_354_11e-2,
        1.0e-16,
    );
    assert_close_with(
        transform.zk_coefficients[0],
        1.054_794_520_547_945_24e-2,
        1.0e-17,
    );
    assert_close_with(
        transform.zk_coefficients[3],
        2.045_112_781_954_887_27e-2,
        1.0e-17,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_prepared_source_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkteg_fixture();
    let rank_two = atomic_yk_zk_prepared_source(fixture.prepared_input(9, 2))?;
    assert_eq!(rank_two.computed_source_len, 9);
    assert_close_with(
        rank_two.origin_constant,
        1.110_957_296_725_969_88e2,
        1.0e-11,
    );
    assert_close_with(rank_two.yk[0], 3.746_164_822_999_324_47e-4, 1.0e-16);
    assert_close_with(rank_two.yk[1], 4.361_981_443_957_904_09e-4, 1.0e-16);
    assert_close_with(rank_two.yk[4], 6.265_729_070_725_439_66e-4, 1.0e-16);
    assert_close_with(rank_two.yk[8], 6.608_249_600_892_370_22e-4, 1.0e-16);
    assert_close_with(rank_two.yk[12], 4.429_642_176_685_166_84e-4, 1.0e-16);
    assert_close_with(rank_two.zk[0], 4.277_638_252_436_042_60e-12, 1.0e-22);
    assert_close_with(rank_two.zk[1], 5.499_800_258_296_022_76e-12, 1.0e-22);
    assert_close_with(rank_two.zk[4], 1.590_237_125_316_554_21e-4, 1.0e-16);
    assert_close_with(rank_two.zk[9], 7.067_357_259_641_375_48e-4, 1.0e-16);
    assert_close_with(
        rank_two.yk_coefficients[0],
        1.374_999_999_999_999_83e-2,
        1.0e-17,
    );
    assert_close_with(
        rank_two.yk_coefficients[3],
        1.360_000_000_000_000_10e-2,
        1.0e-17,
    );

    let rank_one = atomic_yk_zk_prepared_source(fixture.prepared_input(7, 1))?;
    assert_eq!(rank_one.computed_source_len, 7);
    assert_close_with(rank_one.origin_constant, 1.293_492_132_385_440_25, 1.0e-13);
    assert_close_with(rank_one.yk[0], 2.908_635_211_432_032_27e-4, 1.0e-16);
    assert_close_with(rank_one.yk[1], 3.220_388_501_435_997_46e-4, 1.0e-16);
    assert_close_with(rank_one.yk[4], 4.003_521_683_966_694_17e-4, 1.0e-16);
    assert_close_with(rank_one.yk[8], 3.610_570_331_017_010_91e-4, 1.0e-16);
    assert_close_with(rank_one.yk[12], 2.956_084_966_154_574_63e-4, 1.0e-16);
    assert_close_with(rank_one.zk[0], 3.988_806_776_811_954_55e-10, 1.0e-20);
    assert_close_with(rank_one.zk[1], 4.878_024_038_015_732_55e-10, 1.0e-20);
    assert_close_with(rank_one.zk[4], 1.686_537_565_491_518_30e-4, 1.0e-16);
    assert_close_with(rank_one.zk[9], 0.0, 1.0e-20);
    assert_close_with(
        rank_one.yk_coefficients[0],
        1.155_000_000_000_000_12e-2,
        1.0e-17,
    );
    assert_close_with(
        rank_one.yk_coefficients[3],
        1.020_000_000_000_000_07e-2,
        1.0e-17,
    );
    Ok(())
}

#[allow(clippy::excessive_precision)]
#[test]
fn atom_yk_zk_exchange_matches_feff_yzkrdf_reference() -> Result<(), AtomMathError> {
    let fixture = sample_yzkrdf_fixture();
    let overlap = atomic_yk_zk_exchange(fixture.yzkrdf_input(1, 2, 2, false))?;
    assert_eq!(overlap.computed_source_len, 9);
    assert_close_with(overlap.origin_constant, -2.571_240_643_442_588_96, 1.0e-12);
    assert_close_with(overlap.yk[0], 1.109_878_400_538_443_00e-5, 1.0e-17);
    assert_close_with(overlap.yk[1], 1.135_633_080_766_094_54e-5, 1.0e-17);
    assert_close_with(overlap.yk[4], 1.178_867_152_957_986_59e-5, 1.0e-17);
    assert_close_with(overlap.yk[8], 1.017_973_162_090_520_64e-5, 1.0e-17);
    assert_close_with(overlap.yk[12], 6.823_678_168_755_628_77e-6, 1.0e-18);
    assert_close_with(overlap.zk[0], 5.468_221_372_334_369_25e-6, 1.0e-18);
    assert_close_with(overlap.zk[1], 5.909_940_448_128_294_29e-6, 1.0e-18);
    assert_close_with(overlap.zk[4], 7.024_129_238_136_815_07e-6, 1.0e-18);
    assert_close_with(overlap.zk[9], 1.014_708_699_883_866_62e-5, 1.0e-17);
    assert_close_with(
        overlap.yk_coefficients[0],
        -9.990_630_795_999_924_30e-3,
        1.0e-17,
    );
    assert_close_with(
        overlap.yk_coefficients[3],
        8.575_701_162_755_210_16e-2,
        1.0e-16,
    );

    let large_small = atomic_yk_zk_exchange(fixture.yzkrdf_input(2, 3, 1, true))?;
    assert_eq!(large_small.computed_source_len, 7);
    assert_close_with(
        large_small.origin_constant,
        -2.237_401_842_533_894_71e-2,
        1.0e-14,
    );
    assert_close_with(large_small.yk[0], -1.770_958_131_971_287_30e-6, 1.0e-18);
    assert_close_with(large_small.yk[1], -2.024_241_049_179_754_12e-6, 1.0e-18);
    assert_close_with(large_small.yk[4], -2.505_938_578_653_440_58e-6, 1.0e-18);
    assert_close_with(large_small.yk[8], -2.208_316_861_767_755_49e-6, 1.0e-18);
    assert_close_with(large_small.yk[12], -1.808_016_927_269_919_53e-6, 1.0e-18);
    assert_close_with(large_small.zk[0], 3.406_624_777_460_352_47e-7, 1.0e-19);
    assert_close_with(large_small.zk[1], 3.708_373_404_554_750_70e-7, 1.0e-19);
    assert_close_with(large_small.zk[4], -1.328_125_640_689_300_04e-6, 1.0e-18);
    assert_close_with(large_small.zk[9], 0.0, 1.0e-19);
    assert_close_with(
        large_small.yk_coefficients[0],
        -3.957_309_029_859_694_58e-3,
        1.0e-17,
    );
    assert_close_with(
        large_small.yk_coefficients[3],
        -2.038_402_989_657_719_41e-3,
        1.0e-17,
    );
    Ok(())
}
