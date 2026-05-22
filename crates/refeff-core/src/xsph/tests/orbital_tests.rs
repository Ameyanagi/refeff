use super::{support::*, *};

#[test]
fn xsph_initial_hole_orbital_matches_feff_getholeorb0_reference() -> Result<(), XsphError> {
    let (large_source, small_source) = hole_orbital_source();
    let orbital = xsph_initial_hole_orbital(XsphHoleOrbitalInput {
        large_component: large_source.view(),
        small_component: small_source.view(),
        original_step: 0.05,
        new_step: 0.035,
        output_count: 12,
        output_capacity: 16,
    })?;

    assert_eq!(orbital.active_count, 12);
    assert_eq!(orbital.source_count, 16);
    let expected_large = [
        1.184_910_404_133_227e-1,
        1.324_776_258_980_802e-1,
        1.473_025_254_311_882e-1,
        1.629_521_321_304_354e-1,
        1.794_130_641_284_993e-1,
        1.966_760_790_140_799e-1,
        2.147_356_523_616_736e-1,
        2.335_893_236_221_459e-1,
        2.532_385_421_404_321e-1,
        2.736_894_375_417_349e-1,
        2.949_509_263_611_023e-1,
        3.170_346_434_301_027e-1,
    ];
    let expected_small = [
        -2.843_108_757_828_936e-2,
        -2.154_487_654_885_214e-2,
        -1.507_873_522_927_23e-2,
        -9.029_598_949_854_223e-3,
        -3.394_352_127_428_596e-3,
        1.831_137_246_489_437e-3,
        6.651_487_119_769_044e-3,
        1.107_164_454_957_789e-2,
        1.509_688_426_219_543e-2,
        1.873_254_703_597_43e-2,
        2.198_385_316_345_285e-2,
        2.485_593_326_716_564e-2,
    ];
    for index in 0..12 {
        assert_close_tol(
            orbital.large_component[index],
            expected_large[index],
            5.0e-14,
        );
        assert_close_tol(
            orbital.small_component[index],
            expected_small[index],
            5.0e-14,
        );
    }
    for index in 12..16 {
        assert_close(orbital.large_component[index], 0.0);
        assert_close(orbital.small_component[index], 0.0);
    }
    Ok(())
}

#[test]
fn xsph_initial_hole_orbital_rejects_invalid_inputs() {
    let (large_source, small_source) = hole_orbital_source();
    let small_short: Array1<_> = small_source.iter().take(250).copied().collect();
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: large_source.view(),
            small_component: small_short.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::HoleOrbitalLengthMismatch {
            large_len: 251,
            small_len: 250,
        })
    );
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: large_source.view(),
            small_component: small_source.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 17,
            output_capacity: 16,
        }),
        Err(XsphError::InvalidHoleOrbitalOutputCount {
            output_count: 17,
            output_capacity: 16,
        })
    );
    let zero = Array1::<Real>::zeros(251);
    assert_eq!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: zero.view(),
            small_component: zero.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::EmptyHoleOrbital)
    );
    let mut bad = large_source.clone();
    bad[4] = Real::NAN;
    assert!(matches!(
        xsph_initial_hole_orbital(XsphHoleOrbitalInput {
            large_component: bad.view(),
            small_component: small_source.view(),
            original_step: 0.05,
            new_step: 0.035,
            output_count: 12,
            output_capacity: 16,
        }),
        Err(XsphError::NonFiniteScalar {
            name: "large_component",
            ..
        })
    ));
}
