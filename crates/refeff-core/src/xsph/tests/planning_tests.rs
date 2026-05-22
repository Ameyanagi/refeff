use super::{support::*, *};

#[test]
fn xsph_minimize_calculations_matches_feff_reference() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 8)?;

    assert_eq!(plan.max_lj, 6);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2, 4, -3, -1]));
    assert_eq!(
        plan.calculations,
        arr2(&[[2, 5, 1], [4, 4, 2], [-3, 6, 1], [5, 0, 0]])
    );
    Ok(())
}

#[test]
fn xsph_minimize_calculations_honors_active_prefix() -> Result<(), XsphError> {
    let kind = arr1(&[2, 4, 2, -3, 4, 5, -3, 2]);
    let orbital_l = arr1(&[1, 2, 3, 1, 4, 0, 5, 6]);
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);

    let plan = xsph_minimize_calculations(kind.view(), orbital_l.view(), final_lj.view(), 5)?;

    assert_eq!(plan.max_lj, 5);
    assert_eq!(plan.index_map, arr1(&[1, 2, -1, 3, -2]));
    assert_eq!(plan.calculations, arr2(&[[2, 5, 1], [4, 4, 2], [-3, 3, 1]]));
    Ok(())
}

#[test]
fn xsph_lj_needed_flags_match_feff_reference() -> Result<(), XsphError> {
    let final_lj = arr1(&[2, 1, 5, 3, 4, 0, 6, 1]);
    let index_map = arr1(&[1, 2, -1, 3, -2, 4, -3, -1]);

    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 1)?,
        arr1(&[0, 1, 1, 0, 0, 1, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 2)?,
        arr1(&[0, 1, 0, 0, 1, 0, 0])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 3)?,
        arr1(&[0, 0, 0, 1, 0, 0, 1])
    );
    assert_eq!(
        xsph_lj_needed_flags(6, final_lj.view(), index_map.view(), 8, 4)?,
        arr1(&[1, 0, 0, 0, 0, 0, 0])
    );
    Ok(())
}

#[test]
fn xsph_q_bessel_table_matches_feff_reference() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(0.35, radii.view(), 4)?;

    assert_eq!(table.shape(), &[4, 5]);
    assert_eq!(table.strides(), &[1, 4]);
    let expected = arr2(&[
        [
            9.997_958_458_381_769e-1,
            1.166_523_756_252_462e-2,
            8.165_952_107_648_562e-5,
            4.083_055_447_551_5e-7,
            1.587_874_544_380_937_5e-9,
        ],
        [
            9.797_080_213_012_896e-1,
            1.152_437_384_397_447_3e-1,
            8.095_451_039_379_387e-3,
            4.055_621_228_179_726_3e-4,
            1.579_141_698_006_595_3e-5,
        ],
        [
            8.261_173_577_085_878e-1,
            3.129_012_474_446_291e-1,
            6.788_620_641_892_411e-2,
            1.036_640_216_929_531_6e-2,
            1.223_141_376_378_009_1e-3,
        ],
        [
            9.385_522_838_839_835e-2,
            -9.429_243_227_927_261e-2,
            -1.342_662_707_938_009e-1,
            -1.612_046_859_156_612_8e-3,
            1.326_542_239_346_443e-1,
        ],
    ]);
    for ((row, column), &expected_value) in expected.indexed_iter() {
        assert_close(table[(row, column)], expected_value);
    }
    Ok(())
}

#[test]
fn xsph_q_bessel_table_applies_feff_large_argument_cutoff() -> Result<(), XsphError> {
    let radii = arr1(&[0.1, 1.0, 3.0, 20.0]);
    let table = xsph_q_bessel_table(1.0e8, radii.view(), 4)?;

    let expected_first_row = [
        4.205_477_931_907_825e-8,
        9.072_704_282_365_188e-8,
        -4.205_475_210_096_54e-8,
        -9.072_706_385_102_794e-8,
        4.205_468_859_202_071e-8,
    ];
    for (column, &expected_value) in expected_first_row.iter().enumerate() {
        assert_close(table[(0, column)], expected_value);
    }
    for row in 1..4 {
        for column in 0..5 {
            assert_close(table[(row, column)], 0.0);
        }
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_matches_feff_getoccnorm_reference() -> Result<(), XsphError> {
    let cases = [
        (1, 1, 0.5),
        (6, 4, 0.25),
        (8, 4, 0.5),
        (26, 9, 1.0 / 3.0),
        (29, 10, 0.5),
        (47, 17, 0.5),
        (58, 15, 1.0 / 6.0),
        (79, 24, 0.5),
        (80, 24, 1.0),
        (92, 22, 0.5),
        (100, 16, 1.0),
        (100, 29, 1.0),
    ];

    for (atomic_number, hole_index, expected) in cases {
        let actual = xsph_occupation_normalization(atomic_number, hole_index)?;
        assert_close_tol(actual, expected, 5.0e-13);
    }
    Ok(())
}

#[test]
fn xsph_occupation_normalization_rejects_invalid_inputs() {
    assert_eq!(
        xsph_occupation_normalization(0, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 0,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(101, 1),
        Err(XsphError::InvalidOccupationNormAtomicNumber {
            atomic_number: 101,
            max_atomic_number: 100,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 0),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 0,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(26, 30),
        Err(XsphError::InvalidOccupationNormHoleIndex {
            hole_index: 30,
            max_hole_index: 29,
        })
    );
    assert_eq!(
        xsph_occupation_normalization(92, 27),
        Err(XsphError::ZeroOccupationNormDenominator { hole_index: 27 })
    );
}
