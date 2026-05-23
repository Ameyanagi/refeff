use super::{support::*, *};

#[test]
fn background_from_fprime_matches_feff_rdbkg_reference_algorithm() -> Result<(), FullSpectrumError>
{
    let omega = array![
        5.0 / FEFF_HARTREE_EV,
        10.0 / FEFF_HARTREE_EV,
        20.0 / FEFF_HARTREE_EV,
        30.0 / FEFF_HARTREE_EV,
    ];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let f_prime = array![1.0, 3.0, 5.0];
    let f_double_prime = array![0.1, 0.3, 0.5];
    let segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    let expected_second = 1.0 + 0.5 * (3.0 - 1.0) * (10.0_f64 / 15.0).powi(2);
    let expected_third =
        1.0 + ((2.0 / 15.0_f64.powi(2) + 4.0 / 25.0_f64.powi(2)) / 2.0) * 20.0_f64.powi(2);

    assert_eq!(background.point_count(), 4);
    assert_close(background.zero_energy_fprime, 1.0, 0.0);
    assert_close(background.scattering_factor[0].re, 1.0, 0.0);
    assert_close(background.scattering_factor[0].im, 0.0, 0.0);
    assert_close(background.scattering_factor[1].re, expected_second, 1.0e-14);
    assert_close(background.scattering_factor[1].im, 0.2, 1.0e-14);
    assert_close(background.scattering_factor[2].re, expected_third, 1.0e-14);
    assert_close(background.scattering_factor[2].im, 0.4, 1.0e-14);
    assert_close(background.scattering_factor[3].re, 0.0, 0.0);
    assert_close(background.scattering_factor[3].im, 0.0, 0.0);
    assert!(background.effective_electron_count.is_finite());
    assert!(background.effective_electron_count > 0.0);
    Ok(())
}

#[test]
fn background_from_fprime_applies_feff_file_priority() -> Result<(), FullSpectrumError> {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let high_priority_f_prime = array![1.0, 3.0, 5.0];
    let high_priority_f_double_prime = array![0.1, 0.3, 0.5];
    let low_priority_f_prime = array![1.0, 12.0, 14.0];
    let low_priority_f_double_prime = array![9.0, 9.0, 9.0];
    let segments = [
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: photon_energy_ev.view(),
            f_prime: high_priority_f_prime.view(),
            f_double_prime: high_priority_f_double_prime.view(),
        },
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: photon_energy_ev.view(),
            f_prime: low_priority_f_prime.view(),
            f_double_prime: low_priority_f_double_prime.view(),
        },
    ];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    assert_close(
        background.scattering_factor[0].re,
        1.444_444_444_444_444_4,
        1.0e-14,
    );
    assert_close(background.scattering_factor[0].im, 0.2, 1.0e-14);
    Ok(())
}

#[test]
fn background_from_fprime_updates_fp0_in_feff_reverse_order() -> Result<(), FullSpectrumError> {
    let omega = array![30.0 / FEFF_HARTREE_EV];
    let high_priority_energy_ev = array![5.0, 15.0];
    let high_priority_f_prime = array![1.0, 3.0];
    let high_priority_f_double_prime = array![0.1, 0.3];
    let low_priority_energy_ev = array![20.0, 40.0];
    let low_priority_f_prime = array![10.0, 18.0];
    let low_priority_f_double_prime = array![1.0, 3.0];
    let segments = [
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: high_priority_energy_ev.view(),
            f_prime: high_priority_f_prime.view(),
            f_double_prime: high_priority_f_double_prime.view(),
        },
        FullSpectrumBackgroundSegmentInput {
            photon_energy_ev: low_priority_energy_ev.view(),
            f_prime: low_priority_f_prime.view(),
            f_double_prime: low_priority_f_double_prime.view(),
        },
    ];

    let background = full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
        omega: omega.view(),
        segments: &segments,
    })?;

    assert_close(background.zero_energy_fprime, 1.0, 0.0);
    assert_close(background.scattering_factor[0].re, 12.25, 1.0e-14);
    assert_close(background.scattering_factor[0].im, 2.0, 1.0e-14);
    Ok(())
}

#[test]
fn background_from_fprime_rejects_invalid_inputs() {
    let omega = array![10.0 / FEFF_HARTREE_EV];
    let photon_energy_ev = array![5.0, 15.0, 25.0];
    let f_prime = array![1.0, 3.0, 5.0];
    let f_double_prime = array![0.1, 0.3, 0.5];
    let valid_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];

    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: Array1::<Real>::zeros(0).view(),
            segments: &valid_segments,
        }),
        Err(FullSpectrumError::EmptyTable { name: "omega" })
    ));
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &[],
        }),
        Err(FullSpectrumError::EmptyTable {
            name: "background_segments"
        })
    ));

    let too_short_energy = array![5.0];
    let too_short_f_prime = array![1.0];
    let too_short_f_double_prime = array![0.1];
    let too_short_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: too_short_energy.view(),
        f_prime: too_short_f_prime.view(),
        f_double_prime: too_short_f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &too_short_segments,
        }),
        Err(FullSpectrumError::SegmentTooShort {
            name: "background",
            segment: 0,
            len: 1
        })
    ));

    let mismatched_f_prime = array![1.0, 3.0];
    let mismatched_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: photon_energy_ev.view(),
        f_prime: mismatched_f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &mismatched_segments,
        }),
        Err(FullSpectrumError::SegmentLengthMismatch {
            field: "f_prime",
            segment: 0,
            actual: 2,
            expected: 3
        })
    ));

    let non_increasing_energy = array![5.0, 5.0, 25.0];
    let non_increasing_segments = [FullSpectrumBackgroundSegmentInput {
        photon_energy_ev: non_increasing_energy.view(),
        f_prime: f_prime.view(),
        f_double_prime: f_double_prime.view(),
    }];
    assert!(matches!(
        full_spectrum_background_from_fprime(FullSpectrumBackgroundInput {
            omega: omega.view(),
            segments: &non_increasing_segments,
        }),
        Err(FullSpectrumError::SegmentNonIncreasingEnergy {
            segment: 0,
            row: 1,
            ..
        })
    ));
}
