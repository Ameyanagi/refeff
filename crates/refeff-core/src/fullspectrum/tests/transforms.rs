use super::{support::*, *};

#[test]
fn scattering_to_dielectric_matches_feff_fullspectrum_reference_algorithm()
-> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0];
    let scattering_factor = array![Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)];
    let background_scattering_factor = array![Complex64::new(0.25, 0.5), Complex64::new(0.1, 0.05)];

    let dielectric =
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: omega.view(),
            scattering_factor: scattering_factor.view(),
            background_scattering_factor: background_scattering_factor.view(),
        })?;

    assert_eq!(dielectric.point_count(), 2);
    assert_eq!(dielectric.omega[1], omega[1]);
    assert_close(
        dielectric.epsilon_minus_one[0].re,
        -0.125_663_706_143_591_74,
        1.0e-15,
    );
    assert_close(
        dielectric.epsilon_minus_one[0].im,
        -0.251_327_412_287_183_47,
        1.0e-15,
    );
    assert_close(
        dielectric.background_epsilon_minus_one[0].re,
        -0.031_415_926_535_897_934,
        1.0e-15,
    );
    assert_close(
        dielectric.background_epsilon_minus_one[0].im,
        -0.062_831_853_071_795_87,
        1.0e-15,
    );
    assert_close(dielectric.sigma[0], -0.052_118_634_285_441_2, 1.0e-15);
    assert_close(
        dielectric.epsilon_minus_one[1].re,
        0.015_707_963_267_948_967,
        1.0e-15,
    );
    assert_close(
        dielectric.epsilon_minus_one[1].im,
        -0.007_853_981_633_974_483,
        1.0e-15,
    );
    assert_close(dielectric.sigma[1], -0.003_257_414_642_840_075, 1.0e-15);
    Ok(())
}

#[test]
fn kramers_kronig_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0, 4.0, 7.0, 11.0];
    let epsilon2 = array![0.2, 0.5, 0.25, 0.4, 0.15];

    let epsilon1 = full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
        omega: omega.view(),
        epsilon2: epsilon2.view(),
    })?;

    assert_close(epsilon1[0], 0.459_691_458_481_860_66, 1.0e-14);
    assert_close(epsilon1[1], 0.459_691_458_481_860_66, 1.0e-14);
    assert_close(epsilon1[2], 0.160_612_887_833_759_8, 1.0e-14);
    assert_close(epsilon1[3], 0.014_010_911_454_772_346, 1.0e-14);
    assert_close(epsilon1[4], 0.014_010_911_454_772_346, 1.0e-14);
    Ok(())
}

#[test]
fn optical_constants_match_feff_eps2opt_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![0.5, 1.0];
    let epsilon_minus_one = array![Complex64::new(3.0, 4.0), Complex64::new(-0.5, 0.25)];

    let constants = full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
        omega: omega.view(),
        epsilon_minus_one: epsilon_minus_one.view(),
    })?;

    assert_eq!(constants.point_count(), 2);
    assert_eq!(constants.omega[1], omega[1]);
    assert_eq!(constants.epsilon_minus_one[0], epsilon_minus_one[0]);
    assert_close(
        constants.refractive_index_minus_one[0].re,
        1.197_368_226_935_62,
        1.0e-14,
    );
    assert_close(
        constants.refractive_index_minus_one[0].im,
        0.910_179_721_124_454_7,
        1.0e-14,
    );
    assert_close(
        constants.absorption_coefficient[0],
        12.551_376_312_230_127,
        1.0e-14,
    );
    assert_close(constants.reflectivity[0], 0.204_687_076_850_633_86, 1.0e-14);
    assert_close(constants.loss[0], 0.125, 1.0e-14);
    assert_close(
        constants.refractive_index_minus_one[1].re,
        -0.272_326_654_887_322_55,
        1.0e-14,
    );
    assert_close(
        constants.refractive_index_minus_one[1].im,
        0.171_780_374_861_256_22,
        1.0e-14,
    );
    assert_close(
        constants.absorption_coefficient[1],
        4.737_701_967_861_726,
        1.0e-14,
    );
    assert_close(constants.reflectivity[1], 0.034_392_102_279_900_92, 1.0e-14);
    assert_close(constants.loss[1], 0.8, 1.0e-14);
    Ok(())
}

#[test]
fn hamaker_transform_matches_feff_reference_algorithm() -> Result<(), FullSpectrumError> {
    let omega = array![1.0, 2.0, 4.0, 7.0, 11.0];
    let epsilon = array![
        Complex64::new(0.1, 0.2),
        Complex64::new(0.3, 0.5),
        Complex64::new(0.2, 0.25),
        Complex64::new(0.4, 0.4),
        Complex64::new(0.1, 0.15),
    ];

    let imaginary_axis = full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
        omega: omega.view(),
        epsilon: epsilon.view(),
    })?;

    assert_close(imaginary_axis[0], 0.354_646_982_533_894_65, 1.0e-14);
    assert_close(imaginary_axis[1], 0.354_646_982_533_894_65, 1.0e-14);
    assert_close(imaginary_axis[2], 0.223_104_490_219_296_74, 1.0e-14);
    assert_close(imaginary_axis[3], 0.126_886_660_083_068_56, 1.0e-14);
    assert_close(imaginary_axis[4], 0.126_886_660_083_068_56, 1.0e-14);
    Ok(())
}

#[test]
fn fullspectrum_transforms_reject_invalid_inputs() {
    assert!(matches!(
        full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
            omega: array![1.0].view(),
            epsilon2: array![0.2].view(),
        }),
        Err(FullSpectrumError::TooFewRows {
            name: "kramers_kronig",
            len: 1
        })
    ));
    assert!(matches!(
        full_spectrum_kramers_kronig(FullSpectrumKramersKronigInput {
            omega: array![1.0, 1.0].view(),
            epsilon2: array![0.2, 0.3].view(),
        }),
        Err(FullSpectrumError::NonIncreasingOmega { row: 1, .. })
    ));
    assert!(matches!(
        full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
            omega: array![1.0, 2.0].view(),
            epsilon: array![Complex64::new(0.1, f64::NAN), Complex64::new(0.2, 0.3)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon imaginary",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_hamaker_transform(FullSpectrumHamakerInput {
            omega: array![1.0, 2.0].view(),
            epsilon: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "epsilon",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![0.0].view(),
            epsilon_minus_one: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![1.0].view(),
            epsilon_minus_one: array![Complex64::new(f64::NAN, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "epsilon real",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_optical_constants(FullSpectrumOpticalConstantsInput {
            omega: array![1.0, 2.0].view(),
            epsilon_minus_one: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "epsilon_minus_one",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.0,
            omega: array![1.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveInput {
            name: "number_density",
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![0.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonPositiveValue {
            field: "omega",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![1.0].view(),
            scattering_factor: array![Complex64::new(0.1, f64::NAN)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::NonFiniteValue {
            field: "scattering_factor imaginary",
            row: 0,
            ..
        })
    ));
    assert!(matches!(
        full_spectrum_scattering_to_dielectric(FullSpectrumScatteringDielectricInput {
            number_density: 0.01,
            omega: array![1.0, 2.0].view(),
            scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
            background_scattering_factor: array![Complex64::new(0.1, 0.2)].view(),
        }),
        Err(FullSpectrumError::LengthMismatch {
            field: "scattering_factor",
            ..
        })
    ));
}
