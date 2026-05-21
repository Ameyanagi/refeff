#![allow(clippy::excessive_precision)]

use super::*;

#[test]
fn atom_coulomb_coefficient_lookups_match_feff_reference() -> Result<(), AtomMathError> {
    let coefficients = Array3::from_shape_fn((41, 41, 5), |(row, column, channel)| {
        1000.0 * (row + 1) as Real + 10.0 * (column + 1) as Real + channel as Real
    });

    assert_close(
        atomic_direct_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
        2052.0,
    );
    assert_close(
        atomic_direct_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
        2052.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 1, 4, 4)?,
        5022.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 1, 4)?,
        5022.0,
    );
    assert_close(
        atomic_exchange_coulomb_coefficient(coefficients.view(), 4, 4, 4)?,
        0.0,
    );
    Ok(())
}

#[test]
fn atom_coulomb_coefficients_match_feff_muatco_reference() -> Result<(), AtomMathError> {
    let kappas = [-1, 1, -2, 2, -3];
    let occupations = [2.0, 1.5, 3.0, 0.5, 4.0];
    let valence_occupations = [0.0, 0.5, 0.0, 0.25, 0.0];
    let coefficients = atomic_coulomb_coefficients(AtomicCoulombCoefficientInput {
        kappas: &kappas,
        occupations: &occupations,
        valence_occupations: &valence_occupations,
    })?;

    let expected = [
        [
            [2.0, 3.0, 6.0, 1.0, 8.0],
            [0.5, 2.25, 4.5, 0.75, 6.0],
            [1.000_000_000_000_000_7, 0.0, 6.0, 1.5, 12.0],
            [0.0, 0.0, 0.025_000_000_000_000_026, 0.25, 2.0],
            [0.0, 0.0, 1.199_999_999_999_999_3, 0.0, 12.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.450_000_000_000_000_2,
                -0.400_000_000_000_000_3,
                0.0,
                0.0,
            ],
            [
                0.100_000_000_000_000_03,
                0.0,
                0.096_428_571_428_571_31,
                0.0,
                0.0,
            ],
            [
                0.799_999_999_999_999_5,
                0.428_571_428_571_428_2,
                0.342_857_142_857_142_47,
                0.028_571_428_571_428_536,
                -0.548_571_428_571_427_9,
            ],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.095_238_095_238_094_86,
                -0.228_571_428_571_427_8,
            ],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ];

    for (channel, rows) in expected.iter().enumerate() {
        for (row, columns) in rows.iter().enumerate() {
            for (column, &expected) in columns.iter().enumerate() {
                assert_close_with(coefficients[(row, column, channel)], expected, 1.0e-12);
            }
        }
    }

    Ok(())
}

#[test]
fn atom_breit_coefficients_match_feff_bkmrdf_reference() -> Result<(), AtomMathError> {
    let cases = [
        (
            -1,
            -1,
            1,
            [0.5, 0.333_333_333_333_333_2, 0.5],
            [
                -0.166_666_666_666_666_69,
                0.333_333_333_333_333_37,
                -0.166_666_666_666_666_69,
            ],
        ),
        (
            -1,
            1,
            1,
            [
                1.500_000_000_000_000_4,
                1.000_000_000_000_000_2,
                0.166_666_666_666_666_7,
            ],
            [
                1.500_000_000_000_000_4,
                3.000_000_000_000_001,
                0.833_333_333_333_333_6,
            ],
        ),
        (
            1,
            -2,
            1,
            [
                0.500_000_000_000_000_2,
                0.333_333_333_333_334_8,
                0.100_000_000_000_000_06,
            ],
            [
                -0.166_666_666_666_667_4,
                -0.666_666_666_666_669_6,
                -0.126_666_666_666_667_1,
            ],
        ),
        (
            -2,
            2,
            3,
            [
                0.116_666_666_666_666_78,
                0.033_333_333_333_333_36,
                0.002_380_952_380_952_383,
            ],
            [
                0.070_000_000_000_000_05,
                0.420_000_000_000_000_3,
                0.058_571_428_571_428_62,
            ],
        ),
        (
            -3,
            -3,
            5,
            [
                0.050_505_050_505_050_37,
                0.072_150_072_150_071_99,
                0.050_505_050_505_050_37,
            ],
            [
                -0.039_281_705_948_372_45,
                0.078_563_411_896_744_9,
                -0.039_281_705_948_372_45,
            ],
        ),
        (
            2,
            -4,
            3,
            [
                0.102_380_952_380_952_13,
                0.201_587_301_587_301_2,
                0.254_761_904_761_904_3,
            ],
            [
                0.238_500_881_834_214_8,
                0.721_305_114_638_447_2,
                0.264_320_987_654_320_66,
            ],
        ),
    ];

    for (left, right, rank, magnetic, retarded) in cases {
        let actual = atomic_breit_angular_coefficients(left, right, rank)?;
        for index in 0..3 {
            assert_close(actual.magnetic[index], magnetic[index]);
            assert_close(actual.retarded[index], retarded[index]);
        }
    }
    Ok(())
}
